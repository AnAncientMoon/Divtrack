#!/bin/bash
# Download the Cell Tracking Challenge training dataset Fluo-N3DH-CE (C. elegans developing embryo).
#
# usage: bash scripts/prepare_data/download_ctc_ce.sh <target folder>
#
# Source: https://celltrackingchallenge.net/3d-datasets/ (training data, about 3.1 GB). According to the
# CTC "Naming and file content conventions", a dataset folder contains for each sequence N:
#   0N/          images t000.tif ...                      (used)
#   0N_ERR_SEG/  imperfect segmentation masks to be linked (used as the given detections)
#   0N_GT/       gold truth, TRA/ is used for evaluation only
#   0N_ST/       silver truth (not used)
# Voxel size: 1.0 x 0.09 x 0.09 um (z, y, x). Sequence 01 is used for training, 02 for testing.
set -eu
DEST=${1:?usage: download_ctc_ce.sh <target folder>}
URL=https://data.celltrackingchallenge.net/training-datasets/Fluo-N3DH-CE.zip
mkdir -p "$DEST"
cd "$DEST"
if [ ! -f Fluo-N3DH-CE.zip ]; then
  curl -L --fail -o Fluo-N3DH-CE.zip.part "$URL"
  mv Fluo-N3DH-CE.zip.part Fluo-N3DH-CE.zip
fi
unzip -q -n Fluo-N3DH-CE.zip
for d in 01 01_ERR_SEG 01_GT 02 02_ERR_SEG 02_GT; do
  if [ -d "Fluo-N3DH-CE/$d" ]; then
    echo "Fluo-N3DH-CE/$d: $(ls Fluo-N3DH-CE/$d | wc -l) entries"
  else
    echo "missing Fluo-N3DH-CE/$d (check the archive content on the CTC website)" >&2
  fi
done
