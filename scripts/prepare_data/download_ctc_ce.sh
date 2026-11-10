#!/bin/bash
# Download the Cell Tracking Challenge training dataset Fluo-N3DH-CE (C. elegans developing embryo).
#
# usage: bash scripts/prepare_data/download_ctc_ce.sh <target folder>
#
# Source: https://celltrackingchallenge.net/3d-datasets/ (training data, about 3.4 GB). According to the
# CTC "Naming and file content conventions", a dataset folder contains for each sequence N:
#   0N/          images t000.tif ...                      (used)
#   0N_ERR_SEG/  imperfect segmentation masks to be linked (used as the given detections)
#   0N_GT/       gold truth, TRA/ is used for evaluation only
#   0N_ST/       silver truth (not used)
# Voxel size: 1.0 x 0.09 x 0.09 um (z, y, x). Sequence 01 is used for training, 02 for testing.
# A failed download is retried, and running the script again continues an interrupted download.
set -eu
DEST=${1:?usage: download_ctc_ce.sh <target folder>}
URL=https://data.celltrackingchallenge.net/training-datasets/Fluo-N3DH-CE.zip
TRIES=5
mkdir -p "$DEST"
cd "$DEST"
if [ ! -f Fluo-N3DH-CE.zip ]; then
  # Give up on a connection after 30 s and on a transfer that stays below 1 kB/s for 60 s.
  # -C - continues from the bytes already in the .part file.
  n=1
  until curl -L --fail --connect-timeout 30 --speed-limit 1024 --speed-time 60 \
      -C - -o Fluo-N3DH-CE.zip.part "$URL"; do
    if [ "$n" -ge "$TRIES" ]; then
      echo "Download failed $TRIES times. Run this script again to continue it, or download" >&2
      echo "$URL in a browser and put the file in $DEST." >&2
      exit 1
    fi
    echo "Download failed (attempt $n of $TRIES), retrying in 10 s" >&2
    n=$((n + 1))
    sleep 10
  done
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
