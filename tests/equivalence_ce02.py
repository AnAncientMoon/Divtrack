#!/usr/bin/env python
"""Equivalence test: the released weights on CTC Fluo-N3DH-CE sequence 02 must reproduce the published
tracking results file by file.

    python tests/equivalence_ce02.py --ce-dir /path/to/Fluo-N3DH-CE --out /tmp/divtrack_equiv [--seeds 0 1 2 3 4]

For every seed (and for the geometry-only baseline) the script tracks CE/02 with the official
02_ERR_SEG detections and the default settings (birth cost 1.0), then compares the MD5 of
res_track.txt and of all 190 mask###.tif files with tests/reference/ce02_outputs_md5.json. The
reference files are the outputs of the original research code that produced the paper's numbers.
The TIFF bytes depend on tifffile/imagecodecs; use the versions in requirements.txt.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from divtrack.config import load_config  # noqa: E402
from divtrack.inference import track  # noqa: E402


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def compare(res_dir, ref):
    produced = {os.path.basename(p) for p in glob.glob(os.path.join(res_dir, "mask*.tif"))} | {"res_track.txt"}
    missing = sorted(set(ref) - produced)
    extra = sorted(produced - set(ref))
    differ = sorted(f for f in set(ref) & produced if md5(os.path.join(res_dir, f)) != ref[f])
    return missing, extra, differ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ce-dir", required=True, help="Fluo-N3DH-CE folder containing 02 and 02_ERR_SEG")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--no-geometry", action="store_true", help="skip the geometry-only baseline")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    with open(os.path.join(REPO, "tests", "reference", "ce02_outputs_md5.json")) as f:
        reference = json.load(f)
    cfg = load_config()
    work = os.path.join(a.out, "_work")
    runs = [("seed%d" % s, os.path.join(REPO, "weights", "seed%d" % s)) for s in a.seeds]
    if not a.no_geometry:
        runs.append(("geometry_only", None))
    failures = 0
    for key, weights in runs:
        res = os.path.join(a.out, key)
        track(os.path.join(a.ce_dir, "02"), os.path.join(a.ce_dir, "02_ERR_SEG"), [1.0, 0.09, 0.09], res,
              weights_dir=weights, cfg=cfg, device=a.device, geometry_only=weights is None, work_dir=work,
              workers=a.workers, threads=a.threads, name="ce02")
        missing, extra, differ = compare(res, reference[key])
        ok = not (missing or extra or differ)
        failures += not ok
        print("EQUIVALENCE %-14s %s: %d reference files, %d differ, %d missing, %d extra%s"
              % (key, "PASS" if ok else "FAIL", len(reference[key]), len(differ), len(missing), len(extra),
                 "" if ok else " (first: %s)" % (differ + missing + extra)[:3]), flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
