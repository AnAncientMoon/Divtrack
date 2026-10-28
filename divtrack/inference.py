"""Tracking one sequence with trained weights (no training, no ground truth)."""
import json
import os
import time

import numpy as np

from .config import load_config
from .data import Sequence, CellTable
from .rules import rule_count
from .tracking import (Candidates, Solver, write_ctc, division_costs_fused, division_costs_geometry_only)


def resolve_device(device):
    import torch
    if device in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_fusion(path):
    """Fusion weight w_u of F = z + w_u * u (accepts the released and the newly trained format)."""
    with open(path) as f:
        d = json.load(f)
    if float(d.get("w_v", 0.0)) != 0.0 or any(float(r) != 1.0 for r in d.get("r", [1.0])):
        raise ValueError("%s is not a fusion file of the form F = z + w_u * u" % path)
    return float(d["w_u"]), d


class Logger:
    def __init__(self, path=None):
        self.path = path
        self.t0 = time.time()

    def __call__(self, msg):
        line = "[divtrack %s] %s" % (time.strftime("%H:%M:%S"), msg)
        print(line, flush=True)
        if self.path:
            with open(self.path, "a") as f:
                f.write(line + "\n")


def check_work_dir(work_dir, seq, cfg):
    """Cached files in ``work_dir`` are only reused for the same inputs and the same configuration."""
    fp = json.loads(json.dumps({**seq.fingerprint(), "geometry": cfg["geometry"], "patches": cfg["patches"]}))
    fpp = os.path.join(work_dir, "fingerprint.json")
    if os.path.exists(fpp):
        with open(fpp) as f:
            if json.load(f) != fp:
                raise RuntimeError("work directory %s was created for other inputs or another configuration; "
                                   "remove it or choose another work directory" % work_dir)
    else:
        with open(fpp, "w") as f:
            json.dump(fp, f, indent=1)


def prepare_sequence(image_dir, mask_dir, voxel_um, work_dir, cfg, spacing_um=None, workers=1, name="sequence", log=print):
    """Detections, cell table and candidate variables of a sequence (cached in ``work_dir``)."""
    os.makedirs(work_dir, exist_ok=True)
    seq = Sequence(image_dir, mask_dir, voxel_um, spacing_um=spacing_um, workers=workers, name=name,
                   cache_file=os.path.join(work_dir, "detections.npz"))
    check_work_dir(work_dir, seq, cfg)
    cells = CellTable(seq)
    cpath = os.path.join(work_dir, "candidates.npz")
    if not os.path.exists(cpath):
        Candidates.build(cells, cfg["geometry"]).save(cpath)
    log("%s: %d frames (%d-%d), %d detections, cell spacing s = %.3f um%s"
        % (name, len(seq.frames), seq.frames[0], seq.frames[-1], len(cells), seq.spacing,
           "" if not spacing_um else " (manual; measured %.3f um)" % seq.spacing_auto))
    return seq, cells, cpath


def track(image_dir, mask_dir, voxel_um, out_dir, weights_dir=None, cfg=None, device="auto", birth_cost=None,
          geometry_only=False, spacing_um=None, work_dir=None, workers=None, threads=None, name="sequence"):
    """Track one sequence and write the CTC result into ``out_dir``. Returns a summary dict.

    birth_cost: None (default) = cost.birth_deploy, fixed at 1.0 (the largest continuation cost within
        the gate); a number overrides it.
    geometry_only: switch the learned score off (F = 0); same candidates and the same program.
    Worker processes are started with the "spawn" method: call this function from a script that is
    protected by ``if __name__ == "__main__":``.
    """
    cfg = cfg or load_config()
    out_dir = os.path.abspath(out_dir)
    work_dir = os.path.abspath(work_dir or out_dir.rstrip("/") + "_work")
    os.makedirs(work_dir, exist_ok=True)
    log = Logger(os.path.join(work_dir, "log.txt"))
    rt = cfg["runtime"]
    solver_workers = workers if workers is not None else rt["solver_workers"]
    if threads:
        import torch
        torch.set_num_threads(int(threads))
    t_start = time.time()
    seq, cells, cpath = prepare_sequence(image_dir, mask_dir, voxel_um, work_dir, cfg, spacing_um,
                                         workers=max(1, solver_workers), name=name, log=log)
    cands = Candidates.load(cpath)
    cost = cfg["cost"]
    summary = {"name": name, "frames": [seq.frames[0], seq.frames[-1]], "voxel_um": seq.voxel.tolist(),
               "spacing_um": seq.spacing, "spacing_auto_um": seq.spacing_auto, "n_detections": len(cells),
               "n_division_candidates": cands.n_hyperedges(), "model": "geometry_only" if geometry_only else "full"}

    if geometry_only:
        hyper = [division_costs_geometry_only(p["tri_dsum"], cost) if p is not None else None for p in cands.pairs]
    else:
        from .models import load_triplet_net, load_state_head
        from .patches import build_patch_tables, load_patch_tables
        from .scoring import judge_scores, encode_cells, state_scores
        if weights_dir is None:
            raise ValueError("weights_dir is required unless geometry_only=True")
        dev = resolve_device(device)
        judge, _ = load_triplet_net(os.path.join(weights_dir, "judge.pth"), dev)
        head, _ = load_state_head(os.path.join(weights_dir, "state.pth"), dev)
        w_u, _ = load_fusion(os.path.join(weights_dir, "fusion.json"))
        log("weights %s | device %s | w_u = %.4f" % (weights_dir, dev, w_u))
        build_patch_tables(seq, cells, cfg, os.path.join(work_dir, "patches"), workers=rt["patch_workers"], log=log)
        p32, prev16, next16, valid = load_patch_tables(os.path.join(work_dir, "patches"))
        feats = encode_cells(judge, p32, prev16, next16, dev, batch=rt["encode_batch"])
        u = state_scores(head, feats, valid, dev, batch=rt["state_batch"]).astype(np.float64)
        np.save(os.path.join(work_dir, "u.npy"), u.astype(np.float32))
        log("mother state u: median %.2f, fraction > 0 = %.4f" % (np.median(u), (u > 0).mean()))
        scores = judge_scores(judge, p32, cells, cands, dev, chunk=rt["score_chunk"])
        hyper = []
        for fi, p in enumerate(cands.pairs):
            if p is None:
                hyper.append(None)
                continue
            u_m = u[cells.start[cands.frames[fi - 1]] + p["tri_i"]]
            hyper.append(division_costs_fused(p["tri_dsum"], scores[fi], u_m, w_u, cost))
        log("scored %d division candidates" % sum(len(s) for s in scores))

    b = float(cost["birth_deploy"] if birth_cost is None else birth_cost)      # 1.0 unless overridden
    solver = Solver({name: cpath}, workers=solver_workers)
    volumes = cells.volume_dict()
    labels = [cells.LAB[cells.rows(t)] for t in cands.frames]
    try:
        lin = solver.solve(name, hyper, b)
        tracks, det_of, _ = lin.ctc(labels)
        n_decl, n_pass = rule_count(tracks, det_of, volumes, cfg["rules"])
        log("birth cost %.1f: %d tracks, %d divisions, %d pass the rules (rate %.4f)"
            % (b, len(tracks), n_decl, n_pass, n_pass / n_decl if n_decl else 0.0))
    finally:
        solver.close()
    n_tracks = write_ctc(out_dir, seq, cells, lin, workers=rt["mask_workers"])
    summary.update({"birth_cost": b, "n_tracks": n_tracks, "declared_divisions": n_decl, "rule_passing_divisions": n_pass,
                    "rule_pass_rate": n_pass / n_decl if n_decl else 0.0, "seconds": round(time.time() - t_start, 1)})
    with open(os.path.join(work_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log("wrote %s (%d tracks, %d divisions, birth cost %.1f) in %.0f s" % (out_dir, n_tracks, n_decl, b, time.time() - t_start))
    return summary
