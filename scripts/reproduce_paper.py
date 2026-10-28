#!/usr/bin/env python
"""Track and evaluate all test sequences of the paper with the released (or newly trained) weights.

    python scripts/reproduce_paper.py --datasets configs/datasets_paper.yaml --out results/paper [--seeds 0 1 2 3 4]

For every sequence of the ``test`` list and every seed, the sequence is tracked with
weights/seed<k> (birth cost 1.0) and evaluated against its ground truth; the script prints
OP_CLB per seed and mean +- standard deviation, and writes results.json. With --geometry-only
the baseline without the learned score is evaluated instead (it does not depend on the seed).
"""
import argparse
import json
import os
import sys

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from divtrack.config import load_config  # noqa: E402
from divtrack.evaluate import evaluate  # noqa: E402
from divtrack.inference import track  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", required=True, help="YAML with the list 'test' (see configs/datasets_paper.yaml)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights-root", default=os.path.join(REPO, "weights"), help="folder with seed<k>/ subfolders")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--geometry-only", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    with open(a.datasets, encoding="utf-8") as f:
        tests = yaml.safe_load(f)["test"]
    cfg = load_config()
    seeds = [None] if a.geometry_only else a.seeds
    results = {}
    for d in tests:
        work = os.path.join(a.out, d["name"] + "_work")
        for s in seeds:
            tag = "geometry_only" if s is None else "seed%d" % s
            res = os.path.join(a.out, d["name"], tag)
            track(d["image_dir"], d["mask_dir"], d["voxel_um"], res,
                  weights_dir=None if s is None else os.path.join(a.weights_root, "seed%d" % s), cfg=cfg,
                  device=a.device, geometry_only=s is None, spacing_um=d.get("spacing_um"), work_dir=work,
                  workers=a.workers, threads=a.threads, name=d["name"])
            m = evaluate(d["gt_dir"], res)
            results.setdefault(d["name"], {})[tag] = m
            print("RESULT %-8s %-14s OP_CLB %.5f" % (d["name"], tag, m["OP_CLB"]), flush=True)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("\nsequence   OP_CLB (mean +- sd over %d run%s)" % (len(seeds), "s" if len(seeds) > 1 else ""))
    for name, per in results.items():
        v = np.array([m["OP_CLB"] for m in per.values()])
        print("%-10s %.4f +- %.4f   %s" % (name, v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0, " ".join("%.5f" % x for x in v)))


if __name__ == "__main__":
    main()
