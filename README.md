# Divtrack

Tracking of cells, including cell divisions, in 3D time-lapse images.

## Installation

```bash
conda create -n divtrack -c conda-forge python=3.11 pip
conda activate divtrack
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130   # or cu126, cpu
pip install -r requirements.txt
```

Run all commands from the repository root.

## Data

```bash
DATA=/path/to/data
bash scripts/prepare_data/download_ctc_ce.sh $DATA
```

This downloads Fluo-N3DH-CE of the Cell Tracking Challenge. For mskcc-confocal and nih-ls, unzip [mskcc_confocal.zip](https://doi.org/10.5281/zenodo.6460303) and [nih_ls.zip](https://doi.org/10.5281/zenodo.6460375), then convert them:

```bash
for s in s1 s2 s3; do
  python scripts/prepare_data/zenodo_to_ctc.py mskcc --src /path/to/mskcc-confocal/mskcc_confocal_$s --name $s --out $DATA/mskcc-$s
done
for n in 1 2 3; do
  python scripts/prepare_data/zenodo_to_ctc.py nihls --src /path/to/nih-ls/nih_diSPIM_deconv_$n --name $n \
      --out $DATA/nihls-$n --max-frames 270 --relabel-base $((1000 * n))
done
```

## Training

Set the data paths in `configs/datasets_paper.yaml`, then run

```bash
python scripts/train.py --datasets configs/datasets_paper.yaml --work-dir runs/train --device cuda
```

The five models (seeds 0-4) are saved in `runs/train/weights/`. Add `--seeds 0` to train only one; rerun the same command to resume.

## Tracking

The input is a folder of 3D images (`t000.tif`, ...) and a folder of detection masks (`mask000.tif`, ...).

```bash
python scripts/infer.py --image-dir $DATA/Fluo-N3DH-CE/02 --mask-dir $DATA/Fluo-N3DH-CE/02_ERR_SEG \
    --voxel-um 1.0 0.09 0.09 --weights runs/train/weights/seed0 --out-dir results/ce02_seed0
```

`--voxel-um` is the voxel size in µm (z, y, x). The tracks are saved in CTC format.

## Evaluation

```bash
python scripts/evaluate.py --gt-dir $DATA/Fluo-N3DH-CE/02_GT --res-dir results/ce02_seed0
```

This prints OP_CLB and the other CTC measures; BIO uses BC(1).

## All test sequences

```bash
python scripts/reproduce_paper.py --datasets configs/datasets_paper.yaml --out results/test --device cuda
python scripts/reproduce_paper.py --datasets configs/datasets_paper.yaml --out results/test_geometry --geometry-only
```

The first command uses the five trained models; the second switches the learned division score off. All scores are saved in `results.json`.

## License

MIT, see [LICENSE](LICENSE).
