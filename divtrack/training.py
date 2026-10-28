"""Zero-annotation training: one round of self-training, mother state head, fusion weight.

Steps (all on the training sequences, which are pooled; no ground truth is read):
  1. Geometric solution (division score = geometric prior, birth cost 2) -> training labels: the
     divisions of that lineage that pass the rules (divtrack.rules).
  2. Train the triplet network for every seed on these labels (weighted BCE, positive weight
     min(N-/N+, 200)); for every epoch, solve all training sequences with F = z (birth cost 2) and
     count rule-passing divisions; keep the epoch with the largest total (ties: earliest).
  3. Mother state head on features of (t-1, t, t+1) from the frozen encoder of the selected triplet
     network; positives = mothers of the training labels, negatives = reliable continuations; epoch
     chosen by the balanced BCE on the development frames.
  4. Fusion weight w_u of F = z + w_u * u: class-balanced logistic regression on the development
     frames (positives = training labels, negatives = candidates whose mother is a reliable continuation).
Every solve during training uses the birth cost ``cost.birth_label`` (2). The deployment birth cost is
not trained or selected: inference uses ``cost.birth_deploy`` (fixed at 1).
Every GPU step runs in this process, one after the other; integer programs are solved by CPU workers.
"""
import json
import os
import shutil
import time

import numpy as np

from .cells import make_split, reliable_continuations, state_targets
from .config import save_config
from .data import Sequence, CellTable
from .inference import check_work_dir
from .patches import build_patch_tables, load_patch_tables
from .rules import make_labels, rule_count
from .tracking import Candidates, Lineage, Solver, division_costs_score


def auc(scores, labels):
    """Area under the ROC curve (ties counted half)."""
    from scipy.stats import rankdata
    scores, labels = np.asarray(scores, float), np.asarray(labels).astype(bool)
    n1, n0 = labels.sum(), (~labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# =========================================================================== networks
def train_triplet_net(data, cfg, seed, label_name, out_dir, device, log):
    """Train the triplet network on pooled sequences; one checkpoint per epoch in ``out_dir``.

    ``data``: list of dicts with p16 (cells x patch, float16), tri_rows, keys, split, labels, name.
    """
    import torch
    import torch.nn.functional as Fn
    from .models import TripletNet

    J = cfg["judge"]
    geo = cfg["geometry"]
    PL, TL, YL, IL = [], [], [], []
    offp = offc = 0
    for d in data:
        T = d["keys"][:, 0]
        SEG = d["split"]
        T1 = np.minimum(T + 1, len(SEG) - 1)
        ok = lambda a: (SEG[a] == 0) | (SEG[a] == -1)
        itr = np.where(ok(T) & ok(T1))[0]
        K2I = {tuple(int(v) for v in k): i for i, k in enumerate(d["keys"])}
        y = np.zeros(len(d["keys"]), np.float32)
        hit = 0
        for row in d["labels"]:
            t, lm, l1, l2 = (int(v) for v in row)
            i = K2I.get((t, lm, l1, l2))
            i = K2I.get((t, lm, l2, l1)) if i is None else i
            if i is not None:
                y[i] = 1.0
                hit += 1
        log("  %s: labels %d (found among candidates %d) | candidates %d, in training frames %d, positives %d"
            % (d["name"], len(d["labels"]), hit, len(d["keys"]), len(itr), int(y[itr].sum())))
        PL.append(d["p16"]); TL.append(d["tri_rows"] + offp); YL.append(y); IL.append(itr + offc)
        offp += len(d["p16"]); offc += len(d["keys"])
    PATCH = np.concatenate(PL)
    TRI = np.concatenate(TL)
    y = np.concatenate(YL)
    itr = np.concatenate(IL)
    del PL
    if y[itr].sum() < J["min_positives"]:
        raise RuntimeError("only %d positive training candidates (< %d)" % (y[itr].sum(), J["min_positives"]))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = TripletNet(J["feature_dim"]).to(device)
    log("  seed %d: %d training candidates, %d positives" % (seed, len(itr), int(y[itr].sum())))
    EP, BS = J["epochs"], J["batch_size"]
    opt = torch.optim.AdamW(net.parameters(), J["lr"], weight_decay=J["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP)
    ytr = y[itr]
    pw = torch.tensor([min((ytr == 0).sum() / max((ytr == 1).sum(), 1), J["pos_weight_cap"])], device=device)
    os.makedirs(out_dir, exist_ok=True)
    for ep in range(EP):
        net.train()
        perm = rng.permutation(len(itr))
        tot, nb = 0.0, 0
        for i in range(0, len(perm), BS):
            j = perm[i:i + BS]
            ix = itr[j]
            p = PATCH[TRI[ix].reshape(-1)].astype(np.float32).reshape(len(ix), 3, *PATCH.shape[1:])
            P = torch.from_numpy(p).to(device)
            yb = torch.from_numpy(ytr[j]).to(device)
            if rng.random() < J["swap_daughters_p"]:      # no effect on the output; see configs/default.yaml
                P = P.clone()
                P[:, [1, 2]] = P[:, [2, 1]]
            opt.zero_grad()
            loss = Fn.binary_cross_entropy_with_logits(net(P), yb, pos_weight=pw)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        sch.step()
        log("  seed %d epoch %d: training loss %.4f" % (seed, ep, tot / max(nb, 1)))
        torch.save({"model": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                    "gate": [geo["max_sister_dist"], geo["max_mother_dist"]], "label": label_name, "seed": seed,
                    "epoch": ep, "train_seqs": [d["name"] for d in data]},
                   os.path.join(out_dir, "judge_e%d.pth" % ep))


def train_state_head(data, cfg, seed, device, log):
    """Train the mother state head on pooled sequences. ``data``: dicts with feats, valid, lab, train_ok, val_ok.

    Returns (state_dict of the selected epoch, selected epoch, validation BCE, counts)."""
    import torch
    import torch.nn.functional as Fn
    from .models import MotherStateHead

    S = cfg["state_head"]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    F = np.concatenate([d["feats"] for d in data]).astype(np.float32)
    V = np.concatenate([d["valid"] for d in data])
    lab = np.concatenate([d["lab"].astype(np.int64) for d in data])
    tro = np.concatenate([d["train_ok"] for d in data])
    vao = np.concatenate([d["val_ok"] for d in data])
    N = len(lab)
    ok3 = V.all(1)
    itr = np.where(tro & (lab >= 0) & ok3)[0]
    iva = np.where(vao & (lab >= 0) & ok3)[0]
    counts = dict(train_pos=int((lab[itr] == 1).sum()), train_neg=int((lab[itr] == 0).sum()),
                  val_pos=int((lab[iva] == 1).sum()), val_neg=int((lab[iva] == 0).sum()))
    if counts["train_pos"] < S["min_train_positives"] or counts["val_pos"] < S["min_val_positives"]:
        raise RuntimeError("too few state-head positives: %s" % counts)
    net = MotherStateHead(d=F.shape[2], hidden=S["hidden"], heads=S["heads"]).to(device)
    Ft = torch.from_numpy(F).to(device)
    Vt = torch.from_numpy(V).to(device)
    y = torch.from_numpy(lab.astype(np.float32)).to(device)
    w = torch.zeros(N, device=device)
    for c in (0, 1):
        w[torch.from_numpy(itr[lab[itr] == c]).to(device)] = len(itr) / (2.0 * int((lab[itr] == c).sum()))
    opt = torch.optim.AdamW(net.parameters(), lr=S["lr"], weight_decay=S["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, S["epochs"])

    def predict():
        net.eval()
        o = []
        with torch.no_grad():
            for a in range(0, len(Ft), cfg["runtime"]["state_batch"]):
                b = a + cfg["runtime"]["state_batch"]
                o.append(net(Ft[a:b], Vt[a:b]).cpu().numpy())
        o = np.concatenate(o).astype(np.float32)
        assert np.isfinite(o).all()
        return o

    def val_bce(raw):
        l = Fn.binary_cross_entropy_with_logits(torch.from_numpy(raw[iva]), torch.from_numpy(lab[iva].astype(np.float32)),
                                                reduction="none").numpy()
        yv = lab[iva]
        return float(0.5 * (l[yv == 1].mean() + l[yv == 0].mean()))

    best = None
    for ep in range(S["epochs"]):
        net.train()
        perm = rng.permutation(itr)
        for a in range(0, len(perm), S["batch_size"]):
            ix = torch.from_numpy(perm[a:a + S["batch_size"]]).to(device)
            opt.zero_grad(set_to_none=True)
            loss = (Fn.binary_cross_entropy_with_logits(net(Ft[ix], Vt[ix]), y[ix], reduction="none") * w[ix]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite state-head loss")
            loss.backward()
            opt.step()
        sch.step()
        vb = val_bce(predict())
        if best is None or vb < best[0]:
            best = (vb, ep, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
        log("  state head seed %d epoch %d: development BCE %.4f" % (seed, ep, vb))
    return best[2], best[1], best[0], counts


def fit_fusion(z, u, y, dev, l2):
    """w_u and intercept b of p(division) = sigmoid(z + w_u * u + b), class-balanced, on development frames."""
    from scipy.optimize import minimize
    mf = (y >= 0) & dev
    ym = y[mf].astype(float)
    zm, um = z[mf], u[mf]
    sw = np.where(ym == 1, 0.5 / max(ym.sum(), 1), 0.5 / max((1 - ym).sum(), 1))

    def nll(th):
        wu, b0 = th
        f = zm + wu * um + b0
        return float(np.sum(sw * (np.logaddexp(0, -f) * ym + np.logaddexp(0, f) * (1 - ym))) + l2 * (wu ** 2))

    wu, b0 = (float(v) for v in minimize(nll, x0=np.zeros(2), method="L-BFGS-B").x)
    return wu, b0, dict(n_pos=int(ym.sum()), n_neg=int((1 - ym).sum()),
                        auc_dev={"z": auc(zm, ym), "u": auc(um, ym), "fusion": auc(zm + wu * um, ym)})


# =========================================================================== pipeline
class _Seq:
    """Per training sequence: detections, cells, split, candidates, patch tables."""

    def __init__(self, spec, root, cfg, workers, log):
        self.name = spec["name"]
        self.dir = os.path.join(root, "datasets", self.name)
        os.makedirs(self.dir, exist_ok=True)
        self.seq = Sequence(spec["image_dir"], spec["mask_dir"], spec["voxel_um"], spacing_um=spec.get("spacing_um"),
                            cache_file=os.path.join(self.dir, "detections.npz"), workers=workers, name=self.name)
        check_work_dir(self.dir, self.seq, cfg)
        self.cells = CellTable(self.seq)
        sp = cfg["split"]
        self.split = make_split(self.seq.frames, sp["dev_start"], sp["dev_fraction"], sp["guard_frames"])
        self.cand_file = os.path.join(self.dir, "candidates.npz")
        if not os.path.exists(self.cand_file):
            Candidates.build(self.cells, cfg["geometry"]).save(self.cand_file)
        self.cands = Candidates.load(self.cand_file)
        self.keys, self.tri_rows = self.cands.table(self.cells)
        self.volumes = self.cells.volume_dict()
        self.labels_per_frame = [self.cells.LAB[self.cells.rows(t)] for t in self.cands.frames]
        build_patch_tables(self.seq, self.cells, cfg, os.path.join(self.dir, "patches"),
                           workers=cfg["runtime"]["patch_workers"], log=log)
        dev = np.where(self.split == 1)[0]
        log("%s: %d frames, %d detections, s = %.3f um, %d division candidates, development frames [%d, %d)"
            % (self.name, len(self.seq.frames), len(self.cells), self.seq.spacing, len(self.keys), dev[0], dev[-1] + 1))

    def patches(self):
        return load_patch_tables(os.path.join(self.dir, "patches"))

    def p16(self):
        return np.asarray(load_patch_tables(os.path.join(self.dir, "patches"), glimpses=False)).astype(np.float16)

    def ctc(self, lin):
        return lin.ctc(self.labels_per_frame)


class Trainer:
    def __init__(self, cfg, train_specs, work_dir, device="cuda", seeds=None, log=print):
        self.cfg = cfg
        self.specs = train_specs
        self.root = os.path.abspath(work_dir)
        self.device = device
        self.seeds = list(cfg["judge"]["seeds"] if seeds is None else seeds)
        self.log = log
        os.makedirs(self.root, exist_ok=True)
        save_config(cfg, os.path.join(self.root, "config_used.yaml"))
        self.rec_path = os.path.join(self.root, "records.json")
        self.rec = json.load(open(self.rec_path)) if os.path.exists(self.rec_path) else {}
        self.ds = []
        self.solver = None

    # ------------------------------------------------------------------ bookkeeping
    def _save_rec(self):
        tmp = self.rec_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.rec, f, indent=1, sort_keys=True)
        os.replace(tmp, self.rec_path)

    def _set(self, key, value):
        self.rec[key] = value
        self._save_rec()

    def _sd(self, seed):
        """Work folder of one seed: checkpoints of every epoch, scores, solutions, state head, fusion weight."""
        d = os.path.join(self.root, "seed%d" % seed)
        os.makedirs(d, exist_ok=True)
        return d

    # ------------------------------------------------------------------ steps
    def prepare(self):
        t0 = time.time()
        self.ds = [_Seq(spec, self.root, self.cfg, self.cfg["runtime"]["patch_workers"], self.log) for spec in self.specs]
        self.solver = Solver({d.name: d.cand_file for d in self.ds}, workers=self.cfg["runtime"]["solver_workers"])
        self.log("prepared %d training sequences in %.0f s" % (len(self.ds), time.time() - t0))

    def _solve(self, d, hyper, birth, path):
        if os.path.exists(path):
            return Lineage.load(path)
        lin = self.solver.solve(d.name, hyper, birth)
        lin.save(path + ".tmp.npz")
        os.replace(path + ".tmp.npz", path)
        return lin

    def geometric_labels(self):
        """Training labels: rule-passing divisions of the geometric solution (birth cost ``birth_label``)."""
        cost = self.cfg["cost"]
        for d in self.ds:
            out = os.path.join(d.dir, "labels.npy")
            if os.path.exists(out):
                continue
            hyper = [division_costs_score(p["tri_dsum"], p["tri_g"], cost) if p is not None else None for p in d.cands.pairs]
            lin = self._solve(d, hyper, cost["birth_label"], os.path.join(d.dir, "geometric_solution.npz"))
            tracks, det_of, _ = d.ctc(lin)
            labels, n_decl = make_labels(tracks, det_of, d.volumes, self.cfg["rules"])
            np.save(out, labels)
            self._set("labels/%s" % d.name, [int(len(labels)), int(n_decl)])
            self.log("training labels %s: %d of %d declared divisions pass the rules" % (d.name, len(labels), n_decl))

    def labels(self):
        return {d.name: np.load(os.path.join(d.dir, "labels.npy")) for d in self.ds}

    def train_round(self):
        """Train the triplet network of every seed on the labels, then select its epoch by the rule count."""
        import torch
        J = self.cfg["judge"]
        labels = self.labels()
        todo = [s for s in self.seeds if not os.path.exists(os.path.join(self._sd(s), "judge_e%d.pth" % (J["epochs"] - 1)))]
        if todo:
            data = [dict(name=d.name, p16=d.p16(), tri_rows=d.tri_rows, keys=d.keys, split=d.split, labels=labels[d.name])
                    for d in self.ds]
            for s in todo:
                self.log("training the triplet network, seed %d" % s)
                train_triplet_net(data, self.cfg, s, "rules", self._sd(s), self.device, self.log)
            del data
        cost = self.cfg["cost"]
        from .models import load_triplet_net
        for s in self.seeds:
            if ("pick/s%d" % s) in self.rec:
                continue
            totals = {}
            for e in range(J["epochs"]):
                net, _ = load_triplet_net(os.path.join(self._sd(s), "judge_e%d.pth" % e), self.device)
                tot = 0
                for d in self.ds:
                    key = "rc/s%d_e%d_%s" % (s, e, d.name)
                    if key not in self.rec:
                        scores = self._scores(net, d, s, e)
                        hyper = [division_costs_score(p["tri_dsum"], scores[fi], cost) if p is not None else None
                                 for fi, p in enumerate(d.cands.pairs)]
                        lin = self._solve(d, hyper, cost["birth_label"],
                                          os.path.join(self._sd(s), "solution_e%d_%s_b%.1f.npz" % (e, d.name, cost["birth_label"])))
                        tracks, det_of, _ = d.ctc(lin)
                        n_decl, n_pass = rule_count(tracks, det_of, d.volumes, self.cfg["rules"])
                        self._set(key, [n_pass, n_decl])
                        self.log("seed %d epoch %d %s: %d divisions, %d pass the rules" % (s, e, d.name, n_decl, n_pass))
                    tot += self.rec[key][0]
                totals[e] = tot
                del net
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            pick = sorted(totals, key=lambda e: (-totals[e], e))[0]
            self._set("pick/s%d" % s, pick)
            self.log("seed %d: rule-passing divisions per epoch %s -> epoch %d" % (s, totals, pick))

    def _scores(self, net, d, s, e):
        """Judge probabilities of all candidates of ``d`` (cached as .npz)."""
        from .scoring import judge_scores
        path = os.path.join(self._sd(s), "scores_e%d_%s.npz" % (e, d.name))
        if os.path.exists(path):
            with np.load(path) as z:
                flat, lens = z["scores"], z["lens"]
            off = np.concatenate([[0], np.cumsum(lens)])
            return [flat[off[i]:off[i + 1]] for i in range(len(lens))]
        p32 = load_patch_tables(os.path.join(d.dir, "patches"), glimpses=False)
        scores = judge_scores(net, p32, d.cells, d.cands, self.device, chunk=self.cfg["runtime"]["score_chunk"])
        np.savez(path + ".tmp.npz", scores=np.concatenate(scores), lens=np.array([len(x) for x in scores]))
        os.replace(path + ".tmp.npz", path)
        return scores

    def train_state_heads(self):
        """State head of every seed on the frozen encoder of its selected triplet network."""
        import torch
        from .models import load_triplet_net
        from .scoring import encode_cells, state_scores
        S = self.cfg["state_head"]
        rel = S["reliable"]
        labels = self.labels()
        base = []
        for d in self.ds:
            reliable = reliable_continuations(d.cells, rel["max_disp"], rel["min_nn_ratio"], tuple(rel["volume_ratio"]))
            lab, tro, vao = state_targets(d.cells, labels[d.name], reliable, d.split)
            np.save(os.path.join(d.dir, "reliable.npy"), reliable)
            base.append(dict(name=d.name, lab=lab, train_ok=tro, val_ok=vao))
        for s in self.seeds:
            out = os.path.join(self._sd(s), "state.pth")
            if os.path.exists(out):
                continue
            e = self.rec["pick/s%d" % s]
            net, _ = load_triplet_net(os.path.join(self._sd(s), "judge_e%d.pth" % e), self.device)
            data = []
            for d, b in zip(self.ds, base):
                fpath = os.path.join(self._sd(s), "features_%s.npy" % d.name)
                p32, prev16, next16, valid = d.patches()
                if not os.path.exists(fpath):
                    np.save(fpath, encode_cells(net, p32, prev16, next16, self.device, batch=self.cfg["runtime"]["encode_batch"]))
                data.append(dict(b, feats=np.load(fpath), valid=valid.copy()))
            self.log("state head seed %d (features of the triplet network of epoch %d)" % (s, e))
            sd, ep, vb, counts = train_state_head(data, self.cfg, s, self.device, self.log)
            from .models import MotherStateHead
            head = MotherStateHead(d=data[0]["feats"].shape[2], hidden=S["hidden"], heads=S["heads"])
            head.load_state_dict(sd)
            head.to(self.device).eval()
            for d, dd in zip(self.ds, data):
                u = state_scores(head, dd["feats"], dd["valid"], self.device, batch=self.cfg["runtime"]["state_batch"])
                np.save(os.path.join(self._sd(s), "u_%s.npy" % d.name), u)
            torch.save({"model": sd, "seed": s, "epoch": ep, "heads": S["heads"], "train_ds": [d.name for d in self.ds],
                        "feat": "seed%d_epoch%d" % (s, e)}, out)
            self._set("state/s%d" % s, dict(epoch=ep, val_bce=vb, **counts))
            self.log("state head seed %d: epoch %d, development BCE %.4f, %s" % (s, ep, vb, counts))
            del net

    def fit_fusion(self):
        """w_u of every seed on the development frames (positives: training labels)."""
        from .models import load_triplet_net
        from .scoring import judge_logits
        F_ = self.cfg["fusion"]
        labels = self.labels()
        for s in self.seeds:
            out = os.path.join(self._sd(s), "fusion.json")
            if os.path.exists(out):
                continue
            e = self.rec["pick/s%d" % s]
            net, _ = load_triplet_net(os.path.join(self._sd(s), "judge_e%d.pth" % e), self.device)
            Z, U, Y, DEV = [], [], [], []
            for d in self.ds:
                z = np.clip(judge_logits(net, d.p16(), d.tri_rows, self.device, batch=F_["score_batch"]), -F_["z_clip"], F_["z_clip"])
                reliable = np.load(os.path.join(d.dir, "reliable.npy"))
                posk = {(int(a), int(b), min(int(c), int(dd)), max(int(c), int(dd))) for a, b, c, dd in labels[d.name]}
                y = np.full(len(d.keys), -1, np.int8)
                for i, k in enumerate(d.keys):
                    kk = (int(k[0]), int(k[1]), int(k[2]), int(k[3]))
                    if kk in posk:
                        y[i] = 1
                    elif reliable[d.cells.row[(kk[0], kk[1])]]:
                        y[i] = 0
                S_ = d.split
                dev = np.array([S_[int(k[0])] == 1 and S_[min(int(k[0]) + 1, len(S_) - 1)] == 1 and S_[max(int(k[0]) - 1, 0)] == 1
                                for k in d.keys])
                uc = np.load(os.path.join(self._sd(s), "u_%s.npy" % d.name)).astype(np.float64)
                u = uc[d.tri_rows[:, 0]]
                Z.append(z); U.append(u); Y.append(y); DEV.append(dev)
            z, u, y, dev = (np.concatenate(v) for v in (Z, U, Y, DEV))
            wu, b0, info = fit_fusion(z, u, y, dev, F_["l2"])
            res = {"w_u": wu, "w_v": 0.0, "intercept": b0, "intercept_deployed": False, "form": "zu", "z_clip": F_["z_clip"],
                   "train_ds": [d.name for d in self.ds], "judge_epoch": e, **info}
            with open(out, "w") as f:
                json.dump(res, f, indent=1)
            self._set("fusion/s%d" % s, dict(w_u=wu, intercept=b0, **info))
            self.log("fusion seed %d: w_u = %.4f (development positives %d, negatives %d, AUC %s)"
                     % (s, wu, info["n_pos"], info["n_neg"], {k: round(v, 4) for k, v in info["auc_dev"].items()}))

    def export(self):
        """weights/seed<s>/{judge.pth, state.pth, fusion.json} + the selected epochs (selection.json)."""
        wdir = os.path.join(self.root, "weights")
        for s in self.seeds:
            o = os.path.join(wdir, "seed%d" % s)
            os.makedirs(o, exist_ok=True)
            e = self.rec["pick/s%d" % s]
            shutil.copyfile(os.path.join(self._sd(s), "judge_e%d.pth" % e), os.path.join(o, "judge.pth"))
            shutil.copyfile(os.path.join(self._sd(s), "state.pth"), os.path.join(o, "state.pth"))
            shutil.copyfile(os.path.join(self._sd(s), "fusion.json"), os.path.join(o, "fusion.json"))
        birth = float(self.cfg["cost"]["birth_deploy"])          # fixed in the configuration, not selected
        with open(os.path.join(wdir, "selection.json"), "w") as f:
            json.dump({"birth_cost_deploy": birth, "epochs": {k: v for k, v in self.rec.items() if k.startswith("pick/")}}, f, indent=1)
        self.log("exported weights to %s (deployment birth cost %.1f, fixed in the configuration)" % (wdir, birth))

    def run(self):
        t0 = time.time()
        self.prepare()
        try:
            self.geometric_labels()
            self.train_round()
            self.train_state_heads()
            self.fit_fusion()
            self.export()
        finally:
            if self.solver is not None:
                self.solver.close()
        self.log("training finished in %.2f h" % ((time.time() - t0) / 3600))
