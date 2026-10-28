#!/usr/bin/env python
"""Evaluate a CTC result folder with the Cell Tracking Challenge measures (py-ctcmetrics).

    python scripts/evaluate.py --gt-dir Fluo-N3DH-CE/02_GT --res-dir results/ce02_seed0 [--json out.json]

Optionally also prints the label-free rule count when --mask-dir (the input detections) is given.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from divtrack.config import load_config  # noqa: E402
from divtrack.evaluate import evaluate, format_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-dir", required=True, help="CTC ground truth folder (<seq>_GT with TRA/)")
    ap.add_argument("--res-dir", required=True, help="result folder (mask###.tif + res_track.txt)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--mask-dir", default=None, help="input detections; adds the rule count")
    ap.add_argument("--json", default=None, help="write the measures to this file")
    a = ap.parse_args()
    m = evaluate(a.gt_dir, a.res_dir, threads=a.threads)
    print(format_metrics(m))
    if a.mask_dir:
        from divtrack.rules import load_ctc_result, rule_count
        tracks, det_of, volumes = load_ctc_result(a.res_dir, a.mask_dir)
        n_decl, n_pass = rule_count(tracks, det_of, volumes, load_config()["rules"])
        m.update(declared_divisions=n_decl, rule_passing_divisions=n_pass)
        print("divisions declared %d, passing the rules %d" % (n_decl, n_pass))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(m, f, indent=1)


if __name__ == "__main__":
    main()
