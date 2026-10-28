"""Linking with divisions: one 0-1 integer program per pair of consecutive frames.

For frames t and t+1 with m and n detections (positions in units of the cell spacing s):

    x_ij    continuation i -> j          allowed if d_ij <= max_move,   cost d_ij / max_move
    x_ijk   division, mother i at t, daughters j, k at t+1,           cost (d_ij + d_ik) / max_move
                                                                         + c_div - lambda * F_ijk
    b_j     birth of j                                                  cost c_birth

    each cell of t+1 has exactly one source:  sum_i x_ij + sum_{i,k} x_ijk + b_j = 1
    each cell of t is used at most once:      sum_j x_ij + sum_{j,k} x_ijk <= 1   (no death variable)

Division candidates: for every mother i, the ``daughter_topk`` cells of t+1 ranked by a distance
softmax (temperature ``softmax_tau``) are paired up; a pair (j, k) is kept if |x_j - x_k| <=
max_sister_dist and max(|x_i - x_j|, |x_i - x_k|) <= max_mother_dist. Frame pairs are independent
problems; track identities are assigned afterwards in frame order. Problems are solved with
``scipy.optimize.milp`` (HiGHS).

The division score F_ijk enters in one of three forms:
    geometric prior  (training labels):  F = clip(logit(g_ijk)), where g_ijk is a purely geometric
                                          score (distance-softmax bimodality times back-consistency)
    judge            (epoch selection):  F = z_ijk
    full model       (deployment):       F = z_ijk + w_u * u_i
z is obtained from the judge probability clipped to [1e-6, 1 - 1e-6], i.e. |z| <= 13.8155.
The arithmetic follows the original implementation operation by operation so that results are
bit-identical with the published runs.
"""
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np


# ======================================================================== candidates
def softmax_rows(X, mask=None):
    """Row-wise softmax; entries where ``mask`` is True get probability 0."""
    X = X.copy()
    if mask is not None:
        X[mask] = -np.inf
    with np.errstate(invalid="ignore"):
        X = X - np.nanmax(np.where(np.isfinite(X), X, -np.inf), axis=1, keepdims=True)
    E = np.exp(X)
    E[~np.isfinite(E)] = 0.0
    return E / (E.sum(axis=1, keepdims=True) + 1e-30)


def pair_candidates(ppos, pos, geo):
    """Candidate variables between two frames with positions ``ppos`` (m, 3) and ``pos`` (n, 3), float32."""
    max_move = geo["max_move"]
    m, n = len(ppos), len(pos)
    D = np.linalg.norm(ppos[:, None, :] - pos[None, :, :], axis=-1)
    gate = D > max_move
    Cg = D / max(max_move, 1e-6)
    cont_i, cont_j, cont_cost = [], [], []
    for i in range(m):
        for j in np.where(~gate[i])[0]:
            cont_i.append(i)
            cont_j.append(int(j))
            cont_cost.append(float(Cg[i, j]))
    tri_i, tri_j, tri_k, tri_g, tri_d = [], [], [], [], []
    if n >= 2:
        tau = geo["softmax_tau"]
        Af = softmax_rows(-D / tau, gate)
        Ab = softmax_rows(-D.T / tau, gate.T)
        for i in range(m):
            cand = [int(j) for j in np.argsort(-Af[i])[:geo["daughter_topk"]] if not gate[i, j]]
            for a in range(len(cand)):
                for b in range(a + 1, len(cand)):
                    j, k = cand[a], cand[b]
                    sd = np.linalg.norm(pos[j] - pos[k])
                    md = max(np.linalg.norm(pos[j] - ppos[i]), np.linalg.norm(pos[k] - ppos[i]))
                    if sd > geo["max_sister_dist"] or md > geo["max_mother_dist"]:
                        continue
                    p1, p2 = Af[i, j], Af[i, k]
                    g = (min(p1, p2) / (p1 + p2 + 1e-12)) * (Ab[j, i] * Ab[k, i])
                    tri_i.append(i)
                    tri_j.append(j)
                    tri_k.append(k)
                    tri_g.append(float(g))
                    tri_d.append(float(Cg[i, j] + Cg[i, k]))
    return dict(m=m, n=n,
                cont_i=np.array(cont_i, np.int64), cont_j=np.array(cont_j, np.int64), cont_cost=np.array(cont_cost, np.float64),
                tri_i=np.array(tri_i, np.int64), tri_j=np.array(tri_j, np.int64), tri_k=np.array(tri_k, np.int64),
                tri_g=np.array(tri_g, np.float64), tri_dsum=np.array(tri_d, np.float64))


_PAIR_KEYS = ("cont_i", "cont_j", "cont_cost", "tri_i", "tri_j", "tri_k", "tri_g", "tri_dsum")


class Candidates:
    """Candidate variables of a whole sequence.

    ``pairs[fi]`` describes the problem between frame ``frames[fi-1]`` and ``frames[fi]``; it is None
    when there is nothing to link (first frame, an empty frame, or a gap in the frame numbers).
    """

    def __init__(self, frames, n_det, pairs):
        self.frames, self.n_det, self.pairs = list(frames), list(n_det), pairs

    @classmethod
    def build(cls, cells, geo):
        pairs = [None]
        for fi in range(1, len(cells.frames)):
            t0, t1 = cells.frames[fi - 1], cells.frames[fi]
            if cells.count[t0] == 0 or cells.count[t1] == 0 or t1 != t0 + 1:
                pairs.append(None)
                continue
            pairs.append(pair_candidates(cells.pos(t0), cells.pos(t1), geo))
        return cls(cells.frames, [cells.count[t] for t in cells.frames], pairs)

    def n_hyperedges(self):
        return int(sum(len(p["tri_i"]) for p in self.pairs if p is not None))

    def save(self, path):
        has = np.array([p is not None for p in self.pairs])
        out = dict(frames=np.array(self.frames, np.int64), n_det=np.array(self.n_det, np.int64), has=has)
        for key in _PAIR_KEYS:
            dtype = np.float64 if key in ("cont_cost", "tri_g", "tri_dsum") else np.int64
            parts = [np.asarray(p[key], dtype) for p in self.pairs if p is not None]
            out[key] = np.concatenate(parts) if parts else np.zeros(0, dtype)
            out[key + "_len"] = np.array([len(p[key]) for p in self.pairs if p is not None], np.int64)
        out["m"] = np.array([p["m"] for p in self.pairs if p is not None], np.int64)
        out["n"] = np.array([p["n"] for p in self.pairs if p is not None], np.int64)
        tmp = path + ".tmp.npz"
        np.savez(tmp, **out)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        with np.load(path) as z:
            arr = {key: z[key] for key in z.files}          # read every array once
        pairs, q = [], 0
        offs = {key: np.concatenate([[0], np.cumsum(arr[key + "_len"])]) for key in _PAIR_KEYS}
        for has in arr["has"]:
            if not has:
                pairs.append(None)
                continue
            p = dict(m=int(arr["m"][q]), n=int(arr["n"][q]))
            for key in _PAIR_KEYS:
                p[key] = arr[key][offs[key][q]:offs[key][q + 1]]
            pairs.append(p)
            q += 1
        return cls(arr["frames"].tolist(), arr["n_det"].tolist(), pairs)

    def table(self, cells):
        """All division candidates in solver order: keys (t, mother label, daughter labels sorted) and
        cell rows (mother, daughter with the smaller label, daughter with the larger label)."""
        keys, rows = [], []
        for fi, p in enumerate(self.pairs):
            if p is None or len(p["tri_i"]) == 0:
                continue
            t0, t1 = self.frames[fi - 1], self.frames[fi]
            a0, b0 = cells.start[t0], cells.start[t1]
            lm = cells.LAB[a0 + p["tri_i"]]
            lj, lk = cells.LAB[b0 + p["tri_j"]], cells.LAB[b0 + p["tri_k"]]
            swap = lj > lk
            l1, l2 = np.where(swap, lk, lj), np.where(swap, lj, lk)
            r1 = np.where(swap, b0 + p["tri_k"], b0 + p["tri_j"])
            r2 = np.where(swap, b0 + p["tri_j"], b0 + p["tri_k"])
            keys.append(np.stack([np.full(len(lm), t0), lm, l1, l2], 1))
            rows.append(np.stack([a0 + p["tri_i"], r1, r2], 1))
        if not keys:
            return np.zeros((0, 4), np.int64), np.zeros((0, 3), np.int64)
        return np.concatenate(keys).astype(np.int64), np.concatenate(rows).astype(np.int64)


# ======================================================================== division costs
def division_costs_score(dsum, scores, cost):
    """c_ijk with F = clip(logit(p)), p = judge probability or geometric score (training-time solves)."""
    eps, clip = cost["prob_clip"], cost["score_clip"]
    c_div, lam = cost["division_offset"], cost["score_weight"]
    s = np.clip(np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0), 0.0, 1.0)
    out = np.empty(len(s))
    for n, (d, sc) in enumerate(zip(dsum, s)):
        p = min(max(float(sc), eps), 1.0 - eps)
        z = float(np.clip(np.log(p / (1.0 - p)), -clip, clip))
        out[n] = float(d) + c_div - lam * z
    return out


def division_costs_fused(dsum, scores, u_mother, w_u, cost):
    """c_ijk with F = z + w_u * u_i (deployed model)."""
    eps, clip = cost["prob_clip"], cost["score_clip"]
    c_div, lam = cost["division_offset"], cost["score_weight"]
    s = np.clip(np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0), 0.0, 1.0)
    out = np.empty(len(s))
    for n, (d, sc, u) in enumerate(zip(dsum, s, u_mother)):
        p = min(max(float(sc), eps), 1.0 - eps)
        z = float(np.clip(np.log(p / (1.0 - p)), -clip, clip))
        F = z + w_u * u
        out[n] = float(d) + (c_div - lam * float(np.clip(F, -clip, clip)))
    return out


def division_costs_geometry_only(dsum, cost):
    """c_ijk with F = 0 (learned score switched off; same candidates and same program)."""
    return np.array([float(d) + (cost["division_offset"] - cost["score_weight"] * 0.0) for d in dsum], dtype=np.float64)


# ======================================================================== frame-pair program
def solve_pair(p, hyper_cost, birth_cost):
    """Solve one frame-pair program. Returns boolean selections of continuations and hyperedges."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    import scipy.sparse as ssp

    m, n = p["m"], p["n"]
    ci, cj = p["cont_i"].tolist(), p["cont_j"].tolist()
    ti, tj, tk = p["tri_i"].tolist(), p["tri_j"].tolist(), p["tri_k"].tolist()
    cost = [float(v) for v in p["cont_cost"]] + [float(v) for v in hyper_cost] + [float(birth_cost)] * n
    rows, cols, vals = [], [], []
    c = 0
    for i, j in zip(ci, cj):
        rows += [m + j, i]; cols += [c, c]; vals += [1.0, 1.0]; c += 1
    for i, j, k in zip(ti, tj, tk):
        rows += [m + j, m + k, i]; cols += [c, c, c]; vals += [1.0, 1.0, 1.0]; c += 1
    for j in range(n):
        rows += [m + j]; cols += [c]; vals += [1.0]; c += 1
    A = ssp.csr_matrix((vals, (rows, cols)), shape=(m + n, c))
    lb = np.r_[np.zeros(m), np.ones(n)]
    ub = np.ones(m + n)
    res = milp(c=np.array(cost), constraints=[LinearConstraint(A, lb, ub)],
               integrality=np.ones(c), bounds=Bounds(0, 1))
    if not (res.success and res.x is not None):
        raise RuntimeError("frame-pair program failed: status=%s message=%s" % (getattr(res, "status", None), getattr(res, "message", None)))
    sel = res.x > 0.5
    nc, nt = len(ci), len(ti)
    return sel[:nc], sel[nc:nc + nt]


_SOLVER = {}


def _init_solver(cand_files):
    _SOLVER["cands"] = {name: Candidates.load(path) for name, path in cand_files.items()}


def _solve_job(job):
    name, fi, hyper_cost, birth = job
    return fi, solve_pair(_SOLVER["cands"][name].pairs[fi], hyper_cost, birth)


class Solver:
    """Solves all frame pairs of one or several sequences, optionally with a process pool.

    ``cand_files`` maps a sequence name to its saved :class:`Candidates` (.npz).
    """

    def __init__(self, cand_files, workers=1):
        self.cand_files = dict(cand_files)
        self.cands = {name: Candidates.load(path) for name, path in self.cand_files.items()}
        self.pool = None
        if workers > 1:
            self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                            initializer=_init_solver, initargs=(self.cand_files,))

    def solve(self, name, hyper_costs, birth_cost):
        """``hyper_costs[fi]``: cost array of the hyperedges of pair fi. Returns a :class:`Lineage`."""
        cands = self.cands[name]
        todo = [fi for fi, p in enumerate(cands.pairs) if p is not None]
        jobs = [(name, fi, np.asarray(hyper_costs[fi], dtype=np.float64), float(birth_cost)) for fi in todo]
        if self.pool is None:
            results = [(fi, solve_pair(cands.pairs[fi], c, b)) for (_, fi, c, b) in jobs]
        else:
            results = list(self.pool.map(_solve_job, jobs, chunksize=4))
        selections = dict(results)
        return assemble_lineage(cands, selections)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()
            self.pool = None


# ======================================================================== lineage
class Lineage:
    """Track identity of every detection plus the division parents (internal ids from 0)."""

    def __init__(self, frames, assign, parent):
        self.frames, self.assign, self.parent = list(frames), assign, dict(parent)

    def n_divisions(self):
        return len(self.parent) // 2

    def ctc(self, labels):
        """CTC view: tracks [(L, B, E, P)] sorted by L (ids from 1), {L: {t: mask label}}, and per-frame
        {mask label: L}. ``labels[fi]`` are the mask labels of frame ``frames[fi]`` in detection order."""
        seen = set()
        for a in self.assign:
            seen.update(int(v) for v in a)
        remap = {tid: i + 1 for i, tid in enumerate(sorted(seen))}
        parent = {remap[d]: remap[m] for d, m in self.parent.items() if d in remap and m in remap}
        span, det_of, per_frame = {}, {}, []
        for fi, t in enumerate(self.frames):
            mapping = {}
            for j, lab in enumerate(labels[fi]):
                tid = remap[int(self.assign[fi][j])]
                mapping[int(lab)] = tid
                if tid in span:
                    span[tid][1] = t
                else:
                    span[tid] = [t, t]
                det_of.setdefault(tid, {})[t] = int(lab)
            per_frame.append(mapping)
        tracks = [(tid, span[tid][0], span[tid][1], parent.get(tid, 0)) for tid in sorted(span)]
        return tracks, det_of, per_frame

    def save(self, path):
        np.savez(path, frames=np.array(self.frames, np.int64),
                 assign=np.concatenate(self.assign) if self.assign else np.zeros(0, np.int64),
                 lens=np.array([len(a) for a in self.assign], np.int64),
                 parent=np.array(sorted(self.parent.items()), np.int64).reshape(-1, 2))

    @classmethod
    def load(cls, path):
        with np.load(path) as z:
            frames, flat, lens, parent = z["frames"], z["assign"], z["lens"], z["parent"]
        off = np.concatenate([[0], np.cumsum(lens)])
        assign = [flat[off[i]:off[i + 1]] for i in range(len(lens))]
        return cls(frames.tolist(), assign, {int(d): int(m) for d, m in parent})


def assemble_lineage(cands, selections):
    """Assign track ids frame by frame from the solved frame-pair programs."""
    assign, parent = [], {}
    next_tid = 0
    prev = None
    for fi, n in enumerate(cands.n_det):
        p = cands.pairs[fi]
        if p is None:
            cur = np.arange(next_tid, next_tid + n, dtype=np.int64)
            next_tid += n
        else:
            sel_c, sel_t = selections[fi]
            cur = np.full(n, -1, np.int64)
            for q in np.where(sel_c)[0]:
                cur[p["cont_j"][q]] = prev[p["cont_i"][q]]
            for q in np.where(sel_t)[0]:
                mother = int(prev[p["tri_i"][q]])
                d1, d2 = next_tid, next_tid + 1
                next_tid += 2
                cur[p["tri_j"][q]] = d1
                cur[p["tri_k"][q]] = d2
                parent[d1] = mother
                parent[d2] = mother
            for j in range(n):
                if cur[j] < 0:
                    cur[j] = next_tid
                    next_tid += 1
        assign.append(cur)
        prev = cur
    return Lineage(cands.frames, assign, parent)


# ======================================================================== CTC output
def _write_one_mask(job):
    import tifffile
    src_path, out_path, mapping = job
    src = tifffile.imread(src_path)
    lut = np.zeros(int(src.max()) + 1, dtype=np.uint16)
    for lab, tid in mapping.items():
        if 0 <= lab < len(lut):
            lut[lab] = tid
    tifffile.imwrite(out_path, lut[src], compression="zlib")
    return out_path


def write_ctc(out_dir, seq, cells, lineage, workers=1):
    """Write ``mask###.tif`` (detections relabelled with track ids) and ``res_track.txt`` (L B E P)."""
    os.makedirs(out_dir, exist_ok=True)
    labels = [cells.LAB[cells.rows(t)] for t in lineage.frames]
    tracks, _, per_frame = lineage.ctc(labels)
    width = max(3, len(str(max(lineage.frames))))
    jobs = [(seq.mask_path(t), os.path.join(out_dir, "mask%0*d.tif" % (width, t)), per_frame[fi])
            for fi, t in enumerate(lineage.frames)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            list(ex.map(_write_one_mask, jobs, chunksize=4))
    else:
        for job in jobs:
            _write_one_mask(job)
    with open(os.path.join(out_dir, "res_track.txt"), "w") as f:
        for L, B, E, P in tracks:
            f.write("%d %d %d %d\n" % (L, B, E, P))
    return len(tracks)
