#!/usr/bin/env python
"""Train divtrack from scratch without annotation (one round of self-training + state head + fusion).

    python scripts/train.py --datasets configs/datasets_paper.yaml --work-dir runs/paper --device cuda

--datasets is a YAML file with a list ``train`` of sequences (name, image_dir, mask_dir, voxel_um);
see configs/datasets_paper.yaml. The trained weights end up in <work-dir>/weights/seed<k>/ and can be
used with scripts/infer.py. Every step is resumable: rerun the same command after an interruption.
"""
import argparse
import os
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from divtrack.config import load_config  # noqa: E402
from divtrack.inference import Logger, resolve_device  # noqa: E402
from divtrack.training import Trainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", required=True, help="YAML with the list 'train'")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--config", default=None, help="YAML overriding configs/default.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="default: judge.seeds of the config (0 1 2 3 4)")
    ap.add_argument("--device", default="cuda", help="cuda, cuda:N or cpu")
    a = ap.parse_args()
    with open(a.datasets, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    train = spec["train"]
    os.makedirs(a.work_dir, exist_ok=True)
    log = Logger(os.path.join(a.work_dir, "train_log.txt"))
    cfg = load_config(a.config)
    log("training sequences: %s | device %s | seeds %s" % ([d["name"] for d in train], a.device, a.seeds or cfg["judge"]["seeds"]))
    Trainer(cfg, train, a.work_dir, device=resolve_device(a.device), seeds=a.seeds, log=log).run()


if __name__ == "__main__":
    main()
