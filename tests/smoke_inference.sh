#!/bin/bash
# Inference smoke test: the released weights (seed 0) on the first 20 frames of CTC Fluo-N3DH-CE
# sequence 01, on CPU, with evaluation. Takes about a minute. The score on 20 frames is not
# representative: the cell spacing s measured on the early, sparse frames is too large.
#
# usage: bash tests/smoke_inference.sh <Fluo-N3DH-CE dir> <output dir> [python]
set -eu
CE=$1; OUT=$2; PY=${3:-python}
HERE=$(cd "$(dirname "$0")/.." && pwd)
$PY "$HERE/tests/make_subset.py" "$CE/01" "$CE/01_ERR_SEG" "$CE/01_GT" 20 "$OUT/data"
$PY "$HERE/scripts/infer.py" --image-dir "$OUT/data/01" --mask-dir "$OUT/data/01_ERR_SEG" --voxel-um 1.0 0.09 0.09 \
    --out-dir "$OUT/result" --weights "$HERE/weights/seed0" --device cpu --gt-dir "$OUT/data/01_GT" --name ce01_first20
echo "SMOKE_INFERENCE_OK"
