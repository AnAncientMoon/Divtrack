"""Unit tests on small synthetic data (no download needed).

    python tests/test_units.py            (plain runner)
    python -m pytest tests/test_units.py  (if pytest is installed)
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from divtrack.cells import make_split, reliable_continuations  # noqa: E402
from divtrack.config import load_config  # noqa: E402
from divtrack.data import CellTable, Sequence, detections_from_mask  # noqa: E402
from divtrack.rules import load_ctc_result, make_labels, rule_count  # noqa: E402
from divtrack.tracking import (Candidates, Solver, division_costs_geometry_only,  # noqa: E402
                               division_costs_score, write_ctc)

SHAPE = (16, 96, 96)
VOXEL = [1.0, 0.25, 0.25]          # z voxel 4x larger, like most confocal stacks


def _ball(vol, centre, radius_um, label):
    zz, yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
    d = (((zz - centre[0]) * VOXEL[0]) ** 2 + ((yy - centre[1]) * VOXEL[1]) ** 2 + ((xx - centre[2]) * VOXEL[2]) ** 2)
    vol[d <= radius_um ** 2] = label


def make_sequence(root, n_frames=24, division_at=10, seed=0):
    """Four nuclei drifting slowly; nucleus 0 divides between frames division_at and division_at + 1.
    Mask labels are shuffled in every frame. Returns the true lineage as {frame: [(track, centre)]}."""
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(root, "01"), exist_ok=True)
    os.makedirs(os.path.join(root, "01_ERR_SEG"), exist_ok=True)
    cells = {0: np.array([8.0, 30.0, 30.0]), 1: np.array([8.0, 30.0, 70.0]), 2: np.array([8.0, 70.0, 30.0]),
             3: np.array([8.0, 70.0, 70.0])}
    truth = {}
    for t in range(n_frames):
        if t == division_at + 1:
            c = cells.pop(0)
            cells[4] = c + np.array([0.0, 0.0, -8.0])         # daughters 2 um apart, radius 2 um
            cells[5] = c + np.array([0.0, 0.0, 8.0])
        for k in cells:
            cells[k] = cells[k] + np.array([0.0, 0.3, 0.2])
        mask = np.zeros(SHAPE, np.uint16)
        img = np.zeros(SHAPE, np.float32)
        keys = sorted(cells)
        labels = rng.permutation(len(keys)) + 1
        for lab, k in zip(labels, keys):
            _ball(mask, cells[k], 2.5 if k < 4 else 2.0, lab)
        img[mask > 0] = 200.0
        img += rng.normal(10, 2, SHAPE).astype(np.float32)
        tifffile.imwrite(os.path.join(root, "01", "t%03d.tif" % t), img.astype(np.uint16))
        tifffile.imwrite(os.path.join(root, "01_ERR_SEG", "mask%03d.tif" % t), mask)
        truth[t] = [(k, cells[k].copy()) for k in keys]
    return truth


_CACHE = {}
_TMP = []


def _tmpdir():
    d = tempfile.mkdtemp(prefix="divtrack_test_")
    _TMP.append(d)
    return d


def synthetic():
    if "syn" not in _CACHE:
        root = _tmpdir()
        truth = make_sequence(root)
        seq = Sequence(os.path.join(root, "01"), os.path.join(root, "01_ERR_SEG"), VOXEL, name="synthetic")
        _CACHE["syn"] = (root, truth, seq, CellTable(seq))
    return _CACHE["syn"]


def test_detections_and_spacing():
    root, truth, seq, cells = synthetic()
    assert seq.frames == list(range(24))
    for t in (0, 15):
        got = seq.coords[t]
        assert len(got) == len(truth[t])
        for _, c in truth[t]:                            # centroids of the discretised balls
            assert np.linalg.norm(got - c, axis=1).min() < 0.6
    # spacing = median nearest-neighbour distance in micrometres
    assert 5.0 < seq.spacing < 11.0
    labs, cents, vols = detections_from_mask(tifffile.imread(os.path.join(root, "01_ERR_SEG", "mask000.tif")))
    assert len(labs) == 4 and (vols > 0).all()


def test_geometric_tracking_finds_the_division():
    root, truth, seq, cells = synthetic()
    tmp = _tmpdir()
    cfg = load_config()
    cands = Candidates.build(cells, cfg["geometry"])
    cpath = os.path.join(tmp, "cands.npz")
    cands.save(cpath)
    cands2 = Candidates.load(cpath)
    assert cands2.n_hyperedges() == cands.n_hyperedges() > 0
    hyper = [division_costs_score(p["tri_dsum"], p["tri_g"], cfg["cost"]) if p is not None else None for p in cands2.pairs]
    solver = Solver({"syn": cpath}, workers=1)
    lin = solver.solve("syn", hyper, cfg["cost"]["birth_label"])
    assert lin.n_divisions() == 1
    labels = [cells.LAB[cells.rows(t)] for t in lin.frames]
    tracks, det_of, _ = lin.ctc(labels)
    assert len(tracks) == 3 + 1 + 2                       # three cells, one mother, two daughters
    mothers = {P for _, _, _, P in tracks if P}
    assert len(mothers) == 1
    m = mothers.pop()
    mother = [tr for tr in tracks if tr[0] == m][0]
    assert mother[1] == 0 and mother[2] == 10              # mother lives in frames 0..10
    daughters = [tr for tr in tracks if tr[3] == m]
    assert all(d[1] == 11 and d[2] == 23 for d in daughters)

    # the same program with the score switched off still links all non-dividing cells
    lin0 = solver.solve("syn", [division_costs_geometry_only(p["tri_dsum"], cfg["cost"]) if p is not None else None
                                for p in cands2.pairs], 1.0)
    assert sum(1 for tr in lin0.ctc(labels)[0] if tr[1] == 0 and tr[2] == 23) == 3


def test_birth_cost_fixed_at_deployment():
    """Deployment solves once, with the fixed birth cost cost.birth_deploy = 1.0 (the largest continuation
    cost within the gate); the solves of training keep cost.birth_label = 2.0."""
    import divtrack.inference as inference
    import divtrack.training as training
    from divtrack.tracking import pair_candidates
    cfg = load_config(overrides={"runtime": {"solver_workers": 1, "mask_workers": 1, "patch_workers": 1}})
    cost, geo = cfg["cost"], cfg["geometry"]
    assert cost["birth_deploy"] == 1.0 and cost["birth_label"] == 2.0 and "birth_grid" not in cost
    # a continuation exactly at the gate costs d / max_move = 1 = birth_deploy; farther cells are not linked
    g = geo["max_move"]
    p = pair_candidates(np.zeros((1, 3), np.float32), np.array([[0.0, 0.0, g], [0.0, 0.0, g + 0.5]], np.float32), geo)
    assert p["cont_j"].tolist() == [0] and float(p["cont_cost"][0]) == cost["birth_deploy"]

    births = []

    class RecordingSolver(Solver):
        def solve(self, name, hyper_costs, birth_cost):
            births.append(float(birth_cost))
            return Solver.solve(self, name, hyper_costs, birth_cost)

    root, truth, seq, cells = synthetic()
    tmp = _tmpdir()
    spec = dict(name="syn", image_dir=os.path.join(root, "01"), mask_dir=os.path.join(root, "01_ERR_SEG"), voxel_um=VOXEL)
    saved = inference.Solver, training.Solver
    inference.Solver = training.Solver = RecordingSolver
    try:
        s = inference.track(spec["image_dir"], spec["mask_dir"], VOXEL, os.path.join(tmp, "res"), cfg=cfg,
                            geometry_only=True, name="synthetic")
        assert births == [1.0] and s["birth_cost"] == 1.0           # deployment: one solve with c_b = 1
        del births[:]
        tr = training.Trainer(cfg, [spec], os.path.join(tmp, "train"), device="cpu", seeds=[0], log=lambda m: None)
        tr.prepare()
        tr.geometric_labels()
        tr.solver.close()
        assert births == [2.0]                                       # training: c_b = birth_label
    finally:
        inference.Solver, training.Solver = saved


def test_one_round_pipeline():
    """One round of self-training: labels -> triplet network with epoch selection -> state head -> w_u;
    no second round and no birth-cost selection, neither in the pipeline nor in the configuration."""
    from divtrack.training import Trainer
    cfg = load_config()
    assert "round2_label_seed" not in cfg["judge"] and "label_round" not in cfg["state_head"] \
        and "label_round" not in cfg["fusion"] and "birth_grid" not in cfg["cost"]
    root, truth, seq, cells = synthetic()
    spec = dict(name="syn", image_dir=os.path.join(root, "01"), mask_dir=os.path.join(root, "01_ERR_SEG"), voxel_um=VOXEL)
    steps = []
    tr = Trainer(cfg, [spec], os.path.join(_tmpdir(), "steps"), device="cpu", seeds=[0], log=lambda m: None)
    for n in ("prepare", "geometric_labels", "train_round", "train_state_heads", "fit_fusion", "export"):
        setattr(tr, n, (lambda n: lambda: steps.append(n))(n))
    tr.run()
    assert steps == ["prepare", "geometric_labels", "train_round", "train_state_heads", "fit_fusion", "export"]
    assert not any(hasattr(Trainer, n) for n in ("round2_labels", "select_birth"))


def test_ctc_output_round_trip():
    root, truth, seq, cells = synthetic()
    tmp = _tmpdir()
    cfg = load_config()
    cands = Candidates.build(cells, cfg["geometry"])
    cpath = os.path.join(tmp, "cands.npz")
    cands.save(cpath)
    hyper = [division_costs_score(p["tri_dsum"], p["tri_g"], cfg["cost"]) if p is not None else None for p in cands.pairs]
    lin = Solver({"syn": cpath}).solve("syn", hyper, 2.0)
    out = os.path.join(tmp, "res")
    n = write_ctc(out, seq, cells, lin)
    tr = np.loadtxt(os.path.join(out, "res_track.txt"), dtype=int).reshape(-1, 4)
    assert len(tr) == n == 6
    labels = [cells.LAB[cells.rows(t)] for t in lin.frames]
    tracks, det_of, _ = lin.ctc(labels)
    vols = cells.volume_dict()
    tracks_f, det_of_f, vols_f = load_ctc_result(out, os.path.join(root, "01_ERR_SEG"))
    assert tracks_f == tracks and vols_f == vols
    rules = dict(cfg["rules"], min_daughter_frames=5, min_mother_frames=5)
    assert rule_count(tracks, det_of, vols, rules) == rule_count(tracks_f, det_of_f, vols_f, rules) == (1, 1)
    labs, n_decl = make_labels(tracks, det_of, vols, rules)
    assert n_decl == 1 and labs.shape == (1, 4) and labs[0, 0] == 10


def test_rules_thresholds():
    rules = {"min_daughter_frames": 10, "min_mother_frames": 6, "volume_ratio": [0.6, 1.6]}
    # mother 1 lives 0..9 (10 frames), daughters 2, 3 live 10..19 (10 frames)
    tracks = [(1, 0, 9, 0), (2, 10, 19, 1), (3, 10, 19, 1)]
    det_of = {1: {t: 1 for t in range(10)}, 2: {t: 2 for t in range(10, 20)}, 3: {t: 3 for t in range(10, 20)}}
    vols = {(9, 1): 100, (10, 2): 50, (10, 3): 50}
    assert rule_count(tracks, det_of, vols, rules) == (1, 1)
    assert rule_count(tracks, det_of, {(9, 1): 100, (10, 2): 100, (10, 3): 80}, rules) == (1, 0)    # ratio 1.8
    short = [(1, 0, 9, 0), (2, 10, 18, 1), (3, 10, 19, 1)]                                           # 9 frames
    assert rule_count(short, det_of, vols, rules) == (1, 0)
    young = [(1, 5, 9, 0), (2, 10, 19, 1), (3, 10, 19, 1)]                                           # 5 frames
    assert rule_count(young, det_of, vols, rules) == (1, 0)


def test_split_matches_paper_convention():
    s = make_split(list(range(195)))
    dev = np.where(s == 1)[0]
    assert (dev[0], dev[-1] + 1) == (68, 126) and (s == 2).sum() == 4


def test_reliable_continuations():
    root, truth, seq, cells = synthetic()
    rel = reliable_continuations(cells, 0.5, 2.0, (0.8, 1.25))
    # cells that move 0.06 s per frame and are far apart are reliable, except near the first/last frames
    # and around the division
    t = cells.T
    assert rel[(t >= 2) & (t <= 7)].all()
    assert not rel[t == 0].any() and not rel[t == 23].any()


def test_triplet_network_is_symmetric_in_the_daughters():
    import torch
    from divtrack.models import MotherStateHead, TripletNet
    torch.manual_seed(0)
    net = TripletNet().eval()
    P = torch.rand(5, 3, 11, 32, 32)
    with torch.no_grad():
        a = net(P)
        b = net(P[:, [0, 2, 1]])
    assert a.shape == (5,) and torch.equal(a, b)
    head = MotherStateHead().eval()
    f = torch.rand(4, 3, 96)
    valid = torch.tensor([[True, True, True], [False, True, True], [True, True, False], [False, True, False]])
    with torch.no_grad():
        assert torch.isfinite(head(f, valid)).all()


def test_fusion_fit_recovers_weight():
    from divtrack.training import fit_fusion
    rng = np.random.default_rng(0)
    n = 20000
    z, u = rng.normal(0, 2, n), rng.normal(0, 2, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(z + 0.8 * u - 1.0)))).astype(np.int8)
    w_u, b, info = fit_fusion(z, u, y, np.ones(n, bool), l2=1e-3)
    assert abs(w_u - 0.8) < 0.1 and info["n_pos"] + info["n_neg"] == n


if __name__ == "__main__":
    tests = [(name, f) for name, f in sorted(globals().items()) if name.startswith("test_") and callable(f)]
    failed = 0
    for name, f in tests:
        try:
            f()
            print("PASS", name, flush=True)
        except Exception as e:                      # report and continue
            failed += 1
            print("FAIL", name, repr(e), flush=True)
    if os.environ.get("DIVTRACK_KEEP_TMP") != "1":      # set DIVTRACK_KEEP_TMP=1 to keep the synthetic data
        for d in _TMP:
            shutil.rmtree(d, ignore_errors=True)
    print("%d tests, %d failed" % (len(tests), failed))
    sys.exit(1 if failed else 0)
