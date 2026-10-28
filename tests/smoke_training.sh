#!/bin/bash
# Training smoke test: the whole zero-annotation pipeline (geometric labels, triplet network with epoch
# selection, state head, fusion weight, export) on small subsets, then inference and evaluation with
# the newly trained weights (deployment birth cost 1.0). The numbers are not meaningful; the test
# checks that every step runs. Takes a few minutes on a GPU, longer on CPU.
#
# usage: bash tests/smoke_training.sh <Fluo-N3DH-CE dir> <mskcc-s1 CTC dir (01, 01_ERR_SEG)> <output dir> [python] [device]
set -eu
CE=$1; MS=$2; OUT=$3; PY=${4:-python}; DEV=${5:-cpu}
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
$PY "$HERE/tests/make_subset.py" "$CE/01" "$CE/01_ERR_SEG" - 80 "$OUT/data/ce01_first80"
$PY "$HERE/tests/make_subset.py" "$MS/01" "$MS/01_ERR_SEG" - 100 "$OUT/data/ms1_first100"
$PY "$HERE/tests/make_subset.py" "$CE/02" "$CE/02_ERR_SEG" "$CE/02_GT" 60 "$OUT/data/ce02_first60"
cat > "$OUT/datasets.yaml" <<YAML
train:
  - {name: ce01_first80, image_dir: $OUT/data/ce01_first80/01, mask_dir: $OUT/data/ce01_first80/01_ERR_SEG, voxel_um: [1.0, 0.09, 0.09]}
  - {name: ms1_first100, image_dir: $OUT/data/ms1_first100/01, mask_dir: $OUT/data/ms1_first100/01_ERR_SEG, voxel_um: [0.75, 0.15, 0.15]}
YAML
$PY "$HERE/scripts/train.py" --datasets "$OUT/datasets.yaml" --work-dir "$OUT/train" --config "$HERE/tests/configs/smoke.yaml" --device "$DEV"
$PY "$HERE/scripts/infer.py" --image-dir "$OUT/data/ce02_first60/01" --mask-dir "$OUT/data/ce02_first60/01_ERR_SEG" \
    --voxel-um 1.0 0.09 0.09 --out-dir "$OUT/test/ce02_first60" --weights "$OUT/train/weights/seed0" --device "$DEV" \
    --config "$HERE/tests/configs/smoke.yaml" --gt-dir "$OUT/data/ce02_first60/01_GT" --name ce02_first60
echo "SMOKE_TRAINING_OK"
