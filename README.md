# Divtrack

Code for the paper "Efficient 3D Cell Tracking by Integer Programming and Self-Supervised Division Detection" by Tong Hu, Hironori Shigeta, Fuyu Zheng, Yuki Matsumoto and Shigeto Seno, submitted to ISBI 2027.

## Installation

```bash
conda create -n divtrack -c conda-forge python=3.11 pip
conda activate divtrack
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
python tests/test_units.py        # optional: unit tests on synthetic data
```

For another CUDA version or CPU only, replace `cu130` by e.g. `cu126` or `cpu`; `conda env create -f environment.yml` creates the same environment in one step. Run all commands from the repository root; the package itself does not need to be installed.

## Data

The first command downloads CTC Fluo-N3DH-CE (sequence 01 for training, 02 for testing). For mskcc-confocal and nih-ls (Hirsch et al., MICCAI 2022), download and unzip [mskcc_confocal.zip](https://doi.org/10.5281/zenodo.6460303) and [nih_ls.zip](https://doi.org/10.5281/zenodo.6460375) first; the two loops then convert every sequence to the CTC layout, with images in `01/`, detections in `01_ERR_SEG/` and ground truth (used only for evaluation) in `01_GT/`.

```bash
bash scripts/prepare_data/download_ctc_ce.sh /path/to/data
for s in s1 s2 s3; do
  python scripts/prepare_data/zenodo_to_ctc.py mskcc --src /path/to/mskcc-confocal/mskcc_confocal_$s \
      --name $s --out /path/to/data/mskcc-$s
done
for n in 1 2 3; do
  python scripts/prepare_data/zenodo_to_ctc.py nihls --src /path/to/nih-ls/nih_diSPIM_deconv_$n \
      --name $n --out /path/to/data/nihls-$n --max-frames 270 --relabel-base $((1000 * n))
done
```

## Tracking with the released weights

The tracker takes 3D images (`t000.tif`, `t001.tif`, ...) and, for every frame, a label mask of the detected nuclei (`mask000.tif`, ...). The five models of the paper are in `weights/seed0` to `weights/seed4`; to track CE/02 with seed 0:

```bash
python scripts/infer.py \
    --image-dir /path/to/data/Fluo-N3DH-CE/02 --mask-dir /path/to/data/Fluo-N3DH-CE/02_ERR_SEG \
    --voxel-um 1.0 0.09 0.09 --weights weights/seed0 --out-dir results/ce02_seed0
```

`--voxel-um` is the voxel size in µm in the order z, y, x, and `--device` takes `auto` (default), `cpu`, `cuda` or `cuda:N`. The tracks are written to `results/ce02_seed0` in CTC format (`mask###.tif`, `res_track.txt`); intermediate files go to `results/ce02_seed0_work` and can be deleted.

## Evaluation

```bash
python scripts/evaluate.py --gt-dir /path/to/data/Fluo-N3DH-CE/02_GT --res-dir results/ce02_seed0
ctc_evaluate --gt /path/to/data/Fluo-N3DH-CE/02_GT --res results/ce02_seed0 --tra --det --lnk --ct --tf --bc 3 --cca
```

`scripts/evaluate.py` runs `ctc_evaluate` from py-ctcmetrics and prints the CTC measures, including BC(3) and OP_CLB = (LNK + BIO(3)) / 2; adding `--gt-dir` to `scripts/infer.py` does the same right after tracking. The paper reports BC(1) and (LNK + BIO(1)) / 2, which can be read from the full output of the second command.

## Training from scratch

Replace `/path/to/data` in `configs/datasets_paper.yaml` by your data folder, then run

```bash
python scripts/train.py --datasets configs/datasets_paper.yaml --work-dir runs/paper --device cuda
```

This trains five models (seeds 0-4) on CE/01 and mskcc s1, s2; `--seeds 0` trains only one. The weights are saved in `runs/paper/weights/seed<k>/`, which `scripts/infer.py --weights` accepts, and an interrupted run continues when you run the same command again.

## Reproducing the paper

With the `test` paths of `configs/datasets_paper.yaml` filled in, run

```bash
python scripts/reproduce_paper.py --datasets configs/datasets_paper.yaml --out results/paper --device cuda
python scripts/reproduce_paper.py --datasets configs/datasets_paper.yaml --out results/paper_geometry --geometry-only
```

The first command tracks and evaluates CE/02, mskcc s3 and nih-ls 1-3 with the five released models, and the second does the same for the baseline without the learned division score. Both print OP_CLB for every run and save all measures in `results.json` in the output folder. For your own models, add `--weights-root runs/paper/weights` (and `--seeds 0` if you trained only seed 0).

## Citation

```bibtex
@unpublished{hu2026divtrack,
  title  = {Efficient {3D} Cell Tracking by Integer Programming and Self-Supervised Division Detection},
  author = {Hu, Tong and Shigeta, Hironori and Zheng, Fuyu and Matsumoto, Yuki and Seno, Shigeto},
  note   = {Submitted to the IEEE International Symposium on Biomedical Imaging (ISBI) 2027},
  year   = {2026}
}
```

If you use the data, please also cite the Cell Tracking Challenge, Hirsch et al. (MICCAI 2022) and, for nih-ls, Moyle et al. (Nature 2021).

## License

MIT License, see [LICENSE](LICENSE).
