#!/usr/bin/env python
"""Track one sequence with trained weights (zero-shot inference; no training, no ground truth needed).

Example (CTC Fluo-N3DH-CE, sequence 02, released weights of seed 0):
    python scripts/infer.py --image-dir Fluo-N3DH-CE/02 --mask-dir Fluo-N3DH-CE/02_ERR_SEG \
        --voxel-um 1.0 0.09 0.09 --out-dir results/ce02_seed0 --weights weights/seed0 \
        --gt-dir Fluo-N3DH-CE/02_GT

The result folder contains CTC files only (mask###.tif, res_track.txt); intermediate files,
a log and summary.json go to --work-dir (default: <out-dir>_work), which can be deleted afterwards.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from divtrack.config import load_config  # noqa: E402
from divtrack.inference import track  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-dir", required=True, help="folder with t000.tif, t001.tif, ... (3D, z y x)")
    ap.add_argument("--mask-dir", required=True, help="folder with mask000.tif, ... (label mask of the detections)")
    ap.add_argument("--voxel-um", type=float, nargs=3, required=True, metavar=("Z", "Y", "X"), help="voxel size in micrometres")
    ap.add_argument("--out-dir", required=True, help="output folder (CTC format)")
    ap.add_argument("--weights", default=os.path.join(REPO, "weights", "seed0"), help="folder with judge.pth, state.pth, fusion.json")
    ap.add_argument("--geometry-only", action="store_true", help="switch the learned division score off (baseline)")
    ap.add_argument("--birth-cost", type=float, default=None,
                    help="override the deployment birth cost (default: cost.birth_deploy = 1.0, the paper setting)")
    ap.add_argument("--spacing-um", type=float, default=None, help="override the measured cell spacing s (micrometres)")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    ap.add_argument("--workers", type=int, default=None, help="processes for the integer programs (default from the config)")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--config", default=None, help="YAML overriding configs/default.yaml")
    ap.add_argument("--name", default="sequence", help="name used in logs")
    ap.add_argument("--gt-dir", default=None, help="optional CTC ground truth (<seq>_GT) to evaluate the result")
    a = ap.parse_args()

    summary = track(a.image_dir, a.mask_dir, a.voxel_um, a.out_dir, weights_dir=None if a.geometry_only else a.weights,
                    cfg=load_config(a.config), device=a.device, birth_cost=a.birth_cost, geometry_only=a.geometry_only,
                    spacing_um=a.spacing_um, work_dir=a.work_dir, workers=a.workers, threads=a.threads, name=a.name)
    if a.gt_dir:
        from divtrack.evaluate import evaluate, format_metrics
        m = evaluate(a.gt_dir, a.out_dir)
        print("evaluation:", format_metrics(m), flush=True)
        summary["evaluation"] = m
        work = a.work_dir or os.path.abspath(a.out_dir).rstrip("/") + "_work"
        with open(os.path.join(work, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
