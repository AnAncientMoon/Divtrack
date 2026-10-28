"""Cell-level quantities used during training (no ground truth involved).

split                  per sequence: training frames (0), development frames (1), guard frames (2)
reliable_continuations cells whose identity is unambiguous over t-1 -> t -> t+1 -> t+2
state_targets          targets of the mother state head
"""
import numpy as np


def make_split(frames, dev_start=0.35, dev_fraction=0.30, guard=2):
    """0 = training, 1 = development, 2 = guard. Indexed by frame number (length max(frames) + 1)."""
    N = max(frames) + 1
    s = np.zeros(N, np.int64)
    a = int(round(dev_start * N))
    b = a + int(round(dev_fraction * N))
    s[a:b] = 1
    s[max(a - guard, 0):a] = 2
    s[b:min(b + guard, N)] = 2
    return s


def reliable_continuations(cells, max_disp=0.5, min_nn_ratio=2.0, volume_ratio=(0.8, 1.25)):
    """Boolean per cell: the cell is on a reliable continuation.

    A link a -> b between consecutive frames is reliable if a and b are mutual nearest neighbours,
    |x_a - x_b| <= max_disp (units of s), the second-nearest candidate is at least ``min_nn_ratio``
    times farther in both directions, and V_b / V_a is within ``volume_ratio``. A cell at t is
    reliable if it has a reliable predecessor, a reliable successor, and that successor has a
    reliable successor (t-1 -> t -> t+1 -> t+2).
    """
    T = cells.T
    POS = cells.POS.astype(np.float64)
    VOL = cells.VOL
    N = len(T)
    vlo, vhi = volume_ratio
    link = -np.ones(N, int)
    rev = -np.ones(N, int)
    byT = {int(t): np.where(T == t)[0] for t in np.unique(T)}
    for t in sorted(byT):
        if t + 1 not in byT:
            continue
        a, b = byT[t], byT[t + 1]
        Dm = np.linalg.norm(POS[a][:, None] - POS[b][None], axis=-1)
        if min(Dm.shape) < 2:
            continue
        fw = np.argsort(Dm, 1)
        bw = np.argsort(Dm, 0)
        for i in range(len(a)):
            j = fw[i, 0]
            if bw[0, j] != i:
                continue
            d = Dm[i, j]
            fr = Dm[i, fw[i, 1]] / max(d, 1e-8)
            br = Dm[bw[1, j], j] / max(d, 1e-8)
            ratio = VOL[b[j]] / VOL[a[i]]
            if d > max_disp or min(fr, br) < min_nn_ratio or not (vlo <= ratio <= vhi):
                continue
            link[a[i]] = b[j]
            rev[b[j]] = a[i]
    reliable = np.zeros(N, bool)
    for m in range(N):
        c = link[m]
        if rev[m] >= 0 and c >= 0 and link[c] >= 0:
            reliable[m] = True
    return reliable


def state_targets(cells, labels, reliable, split):
    """Targets of the mother state head.

    Returns (lab, train_ok, val_ok): lab = 1 for mothers of the training labels that are not
    reliable continuations, 0 for reliable continuations that are not labelled mothers, -1 otherwise;
    train_ok / val_ok: frames t-1 .. t+2 all outside / all inside the development block.
    """
    N = len(cells)
    mother = np.zeros(N, bool)
    for t, lm, l1, l2 in np.asarray(labels, np.int64).reshape(-1, 4):
        mother[cells.row[(int(t), int(lm))]] = True
    lab = np.full(N, -1, np.int8)
    lab[mother & ~reliable] = 1
    lab[reliable & ~mother] = 0
    tmax = int(max(cells.frames))
    assert len(split) == tmax + 1
    train_ok = np.zeros(N, bool)
    val_ok = np.zeros(N, bool)
    for i in range(N):
        t = int(cells.T[i])
        if t < 1 or t + 2 > tmax:
            continue
        s = split[t - 1:t + 3]
        train_ok[i] = bool(np.all(s != 1))
        val_ok[i] = bool(np.all(s == 1))
    return lab, train_ok, val_ok
