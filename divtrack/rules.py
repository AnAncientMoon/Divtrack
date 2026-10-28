"""Biological rules on divisions found by the tracker (no ground truth involved).

A division of mother track M (last frame t) into daughter tracks D1, D2 (first frame t+1)
passes the rules if
    * both daughter tracks last at least ``min_daughter_frames`` frames,
    * the mother track exists for at least ``min_mother_frames`` frames, counting frame t,
    * 0.6 <= (V_D1 + V_D2) / V_M <= 1.6, with V the voxel counts of the given detection masks
      (mother at t, daughters at t+1).

The same rules are used for two things:
    make_labels   training labels (t, mother label, daughter label, daughter label)
    rule_count    label-free model selection: number of declared divisions and of rule-passing ones

A lineage is given in CTC form: ``tracks`` = [(L, B, E, P)] sorted by L, and ``det_of[L][t]`` = mask
label of track L in frame t. ``volumes[(t, label)]`` = voxel count.
"""
import collections
import os

import numpy as np


def _rule_params(rules):
    lo, hi = rules["volume_ratio"]
    return int(rules["min_daughter_frames"]), int(rules["min_mother_frames"]), float(lo), float(hi)


def make_labels(tracks, det_of, volumes, rules):
    """Training labels from a lineage. Returns (int64 array (n, 4) of (t, mother, daughter1, daughter2), n_declared)."""
    ks, ka, vlo, vhi = _rule_params(rules)
    BEG = {L: B for L, B, E, P in tracks}
    END = {L: E for L, B, E, P in tracks}
    PAR = {L: P for L, B, E, P in tracks}
    kid = collections.defaultdict(list)
    for L, P in PAR.items():
        if P:
            kid[P].append(L)
    rows, n_decl = [], 0
    for P, ch in kid.items():
        if len(ch) != 2 or P not in END:
            continue
        t = END[P]
        lm = det_of.get(P, {}).get(t)
        l1 = det_of.get(ch[0], {}).get(t + 1)
        l2 = det_of.get(ch[1], {}).get(t + 1)
        if None in (lm, l1, l2):
            continue
        n_decl += 1
        surv = min(END[ch[0]] - BEG[ch[0]] + 1, END[ch[1]] - BEG[ch[1]] + 1)
        age = t - BEG[P] + 1
        vr = (volumes.get((t + 1, l1), 0) + volumes.get((t + 1, l2), 0)) / max(volumes.get((t, lm), 1), 1)
        if surv >= ks and age >= ka and vlo <= vr <= vhi:
            rows.append((t, lm, l1, l2))
    return np.array(rows, dtype=np.int64).reshape(-1, 4), n_decl


def rule_count(tracks, det_of, volumes, rules):
    """(number of declared divisions, number passing the rules)."""
    ks, ka, vlo, vhi = _rule_params(rules)
    rowB = {L: B for L, B, E, P in tracks}
    rowE = {L: E for L, B, E, P in tracks}
    par = {L: P for L, B, E, P in tracks}
    kids = collections.defaultdict(list)
    for L, P in par.items():
        if P:
            kids[P].append(L)
    n_decl, passing = 0, set()
    for m, ch in kids.items():
        if len(ch) != 2:
            continue
        n_decl += 1
        te, tb = rowE[m], rowB[ch[0]]
        if rowB[ch[1]] != tb or tb != te + 1:
            continue
        if (rowE[ch[0]] - rowB[ch[0]] + 1) < ks or (rowE[ch[1]] - rowB[ch[1]] + 1) < ks:
            continue
        if (te - rowB[m] + 1) < ka:
            continue
        lm = det_of.get(m, {}).get(te)
        l1 = det_of.get(ch[0], {}).get(tb)
        l2 = det_of.get(ch[1], {}).get(tb)
        if None in (lm, l1, l2):
            continue
        vm = volumes.get((te, lm), 0)
        v1 = volumes.get((tb, l1), 0)
        v2 = volumes.get((tb, l2), 0)
        if vm <= 0 or not (vlo <= (v1 + v2) / vm <= vhi):
            continue
        passing.add((te, lm, min(l1, l2), max(l1, l2)))
    return n_decl, len(passing)


def load_ctc_result(res_dir, mask_dir):
    """Read a CTC result folder (res_track.txt + mask###.tif) produced on the detections in ``mask_dir``.

    Returns (tracks, det_of, volumes); a result label is matched to the detection it covers most.
    """
    import tifffile
    from .data import frame_files

    tr = np.loadtxt(os.path.join(res_dir, "res_track.txt"), dtype=np.int64).reshape(-1, 4)
    tracks = [tuple(int(v) for v in r) for r in tr]
    res_files, det_files = frame_files(res_dir, "mask"), frame_files(mask_dir, "mask")
    det_of, volumes = {}, {}
    for t in sorted(det_files):
        if t not in res_files:
            continue
        r = tifffile.imread(res_files[t]).ravel().astype(np.int64)
        e = tifffile.imread(det_files[t]).ravel().astype(np.int64)
        fg = e > 0
        labs, vols = np.unique(e[fg], return_counts=True)
        for L, v in zip(labs, vols):
            volumes[(t, int(L))] = int(v)
        both = fg & (r > 0)
        pairs, counts = np.unique(np.stack([e[both], r[both]], 1), axis=0, return_counts=True)
        best = {}
        for (L, R), c in zip(pairs, counts):        # per detection: the result label covering most of it
            if (int(c), int(R)) > best.get(int(L), (-1, -1)):
                best[int(L)] = (int(c), int(R))
        for L, (c, R) in best.items():
            det_of.setdefault(R, {})[t] = L
    return tracks, det_of, volumes
