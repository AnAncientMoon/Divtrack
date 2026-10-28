# divtrack

Tracking of nuclei, including divisions, in 3D time-lapse images of *C. elegans* embryos, with image
scores of division trained **without manual lineage or division annotations**.

This is the code of the paper

> **Efficient Self-Supervised 3D Cell Tracking by Integer Programming with Visual Evidence of Division**<br>
> Tong Hu, Hironori Shigeta, Fuyu Zheng, Shigeto Seno<br>
> Graduate School of Information Science and Technology, The University of Osaka, Japan<br>
> submitted to ISBI 2027

It contains the tracker, the self-supervised training pipeline, the five released models used in the
paper, data preparation scripts and tests.

* Input: 3D images and, for every frame, a label mask of detected nuclei (the detections are taken as given).
* Output: a lineage in the [Cell Tracking Challenge](https://celltrackingchallenge.net) (CTC) format.

## Method in short

Detections of consecutive frames *t* and *t+1* are linked by one 0-1 integer program per frame pair.
All distances *d* are measured in units of the cell spacing *s*, the median nearest-neighbour
distance between nuclei over the whole sequence, which is computed from the detections.

| variable | meaning | cost |
|---|---|---|
| x<sub>ij</sub> | continuation i &rarr; j, allowed if d<sub>ij</sub> &le; 3 | d<sub>ij</sub> / 3 |
| x<sub>ijk</sub> | division of mother i into daughters j, k | (d<sub>ij</sub> + d<sub>ik</sub>) / 3 + 0.5 &minus; 0.03 F<sub>ijk</sub> |
| b<sub>j</sub> | birth of j | c<sub>b</sub> |

At deployment the birth cost is fixed at c<sub>b</sub> = 1, the largest continuation cost within
the gate (d<sub>ij</sub> / 3 &le; 1 for d<sub>ij</sub> &le; 3). Every solve during training uses
c<sub>b</sub> = 2.

Every cell of frame *t+1* has exactly one source (continuation, division or birth); every cell
of frame *t* is used at most once (there is no death variable). Division candidates: each mother
pairs up its four top-ranked neighbours under a distance softmax; a pair is kept if the sisters
are at most 3 *s* apart and both daughters at most 2 *s* from the mother. The program is solved
with `scipy.optimize.milp` (HiGHS).

The learned division score is F<sub>ijk</sub> = z<sub>ijk</sub> + w<sub>u</sub> u<sub>i</sub>:

* **z<sub>ijk</sub> - triplet network.** Patches around the mother (frame *t*) and the two
  candidate daughters (frame *t+1*) are cubes of side 1.5 *s* resampled to 11&times;32&times;32;
  a shared 3D CNN maps each to a 96-d feature f, and an MLP (288 &rarr; 128 &rarr; 1) scores
  [f<sub>i</sub>; (f<sub>j</sub>+f<sub>k</sub>)/2; |f<sub>j</sub>&minus;f<sub>k</sub>|]. The network
  sees images only, no geometric quantity. z is clipped to |z| &le; 13.8.
* **u<sub>i</sub> - mother state head.** The frozen encoder of the triplet network embeds the
  image at the position of cell *i* in frames *t&minus;1*, *t*, *t+1*; one attention layer with
  frame *t* as query and a small MLP give u<sub>i</sub>.
* **w<sub>u</sub>** is fitted by class-balanced logistic regression on held-out frames of the
  training sequences.

**Training without manual lineage or division annotations** (one round of self-training, all training sequences pooled):

1. Solve the training sequences with a purely geometric division score (birth cost 2).
2. Keep the divisions of that lineage that pass three rules as training labels: both daughter
   tracks last at least 10 frames, the mother track has existed at least 6 frames (counting the
   division frame), and the volume ratio (V<sub>j</sub>+V<sub>k</sub>)/V<sub>i</sub> of the given
   detection masks is between 0.6 and 1.6.
3. Train the triplet network (5 seeds, weighted BCE with positive weight min(N<sub>&minus;</sub>/N<sub>+</sub>, 200), 8 epochs).
4. Select the epoch without ground truth: solve the training sequences with F = z (birth cost 2) and
   keep the epoch with the most rule-passing divisions.
5. Train the mother state head on the frozen encoder of the selected triplet network (positives:
   mothers of the training labels; negatives: cells on reliable continuations), then fit
   w<sub>u</sub> on the development frames (positives: training labels; negatives: candidates whose
   mother is a reliable continuation).

The development frames (the middle 30% of every training sequence, plus two guard frames on each
side) are not used to train the triplet network or the state head; they serve to choose the epoch of
the state head and to fit w<sub>u</sub>. The label-free epoch selection of step 4 solves whole sequences.

All hyperparameters are listed in [`configs/default.yaml`](configs/default.yaml).

## Repository layout

```
divtrack/                  Python package
  data.py                  reading CTC-style sequences, centroids, cell spacing s, positions
  patches.py               image patches (cube of 1.5 s -> 11x32x32) and patch tables
  tracking.py              candidate generation, frame-pair integer program, lineage, CTC writer
  models.py                triplet network (z) and mother state head (u)
  scoring.py               running the networks on a sequence
  rules.py                 rule-based training labels and the label-free rule count
  cells.py                 train/development frame split, reliable continuations, state-head targets
  training.py              self-training (labels, triplet network, epoch selection), state head, fusion
  inference.py             tracking a sequence with trained weights
  evaluate.py              CTC measures through py-ctcmetrics
configs/default.yaml       all hyperparameters
configs/datasets_paper.yaml  training/test sequences of the paper (paths to fill in)
scripts/infer.py           track one sequence
scripts/train.py           train from scratch
scripts/evaluate.py        evaluate a result against CTC ground truth
scripts/reproduce_paper.py track and evaluate all test sequences of the paper for all seeds
scripts/prepare_data/      download (CTC) and conversion (Zenodo -> CTC layout) scripts
weights/seed0 ... seed4    released models (see weights/README.md)
tests/                     unit tests, smoke tests, equivalence test
LICENSE                    MIT license
```

## Installation

Verified environment: Linux x86_64, Python 3.11.17, PyTorch 2.13.0 (CUDA 13.0 build), numpy 2.4.6,
scipy 1.17.1, tifffile 2026.3.3, imagecodecs 2026.3.6, PyYAML 6.0.3, py-ctcmetrics 1.3.3.
Training was run on one NVIDIA RTX PRO 6000 Blackwell Max-Q (96 GB); inference also runs on CPU.

```bash
conda create -n divtrack -c conda-forge python=3.11 pip
conda activate divtrack
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130   # or .../whl/cu126, .../whl/cpu
pip install -r requirements.txt
pip install -e .            # optional (editable mode); the scripts also run from the repository without it
```

or `conda env create -f environment.yml`. No commercial solver is needed (HiGHS ships with SciPy).
The released results are reproduced bit-exactly only with the versions above (in particular
tifffile/imagecodecs for the output files); other versions should give the same tracks up to
rare ties.

## Data

**CTC Fluo-N3DH-CE** (training sequence 01, test sequence 02; voxel 1.0 &times; 0.09 &times; 0.09 &micro;m):

```bash
bash scripts/prepare_data/download_ctc_ce.sh /path/to/data
```

The archive (training data of the CTC, about 3.1 GB) contains `01`, `02` (images), `01_ERR_SEG`,
`02_ERR_SEG` (the imperfect segmentation masks distributed for the Cell Linking Benchmark, used
here as the given detections) and `01_GT`, `02_GT` (ground truth, used for evaluation only).

**mskcc-confocal and nih-ls** (released with Hirsch et al., *Tracking by weakly-supervised learning and graph
optimization for whole-embryo C. elegans lineages*, MICCAI 2022) come without segmentation. Download and unzip
[mskcc_confocal.zip](https://doi.org/10.5281/zenodo.6460303) (about 24 GB) and
[nih_ls.zip](https://doi.org/10.5281/zenodo.6460375) (about 25 GB), then convert every sequence:

```bash
for s in s1 s2 s3; do
  python scripts/prepare_data/zenodo_to_ctc.py mskcc --src /path/to/mskcc-confocal/mskcc_confocal_$s \
      --name $s --out /path/to/data/mskcc-$s
done
for n in 1 2 3; do
  python scripts/prepare_data/zenodo_to_ctc.py nihls --src /path/to/nih-ls/nih_diSPIM_deconv_$n \
      --name $n --out /path/to/data/nihls-$n --max-frames 270 --relabel-base $((1000 * n))
done
```

The detections are drawn from the annotated positions and radii (ellipsoids of 0.8 &times; the
radius) and the same masks serve as ground truth. For nih-ls only the first 270 frames are used, as
in Hirsch et al., and the detection labels are permuted randomly in every frame. mskcc s1 and s2
are training sequences, s3 is a test sequence; nih-ls is used only for zero-shot testing.

Every sequence then has the layout

```
<sequence>/01/          t000.tif t001.tif ...          3D images
<sequence>/01_ERR_SEG/  mask000.tif mask001.tif ...    detections (0 = background)
<sequence>/01_GT/TRA/   man_track000.tif ... man_track.txt   ground truth (evaluation only)
```

(for CTC data the folders are `0N`, `0N_ERR_SEG`, `0N_GT`).

## Quick start: tracking with the released weights

```bash
python scripts/infer.py \
    --image-dir /path/to/data/Fluo-N3DH-CE/02 --mask-dir /path/to/data/Fluo-N3DH-CE/02_ERR_SEG \
    --voxel-um 1.0 0.09 0.09 --weights weights/seed0 --out-dir results/ce02_seed0 \
    --gt-dir /path/to/data/Fluo-N3DH-CE/02_GT          # optional: evaluate
```

* `--out-dir` receives `mask###.tif` (detections relabelled with track ids) and `res_track.txt`;
  intermediate files, `log.txt` and `summary.json` go to `<out-dir>_work` (can be deleted).
* `--device auto|cpu|cuda|cuda:N`, `--threads` (CPU threads), `--workers` (processes for the integer programs).
* The birth cost is fixed at 1.0 (`cost.birth_deploy`, the paper setting); `--birth-cost` overrides it.
* `--geometry-only` switches the learned score off (F = 0) with everything else unchanged.
* `--spacing-um` overrides the measured cell spacing. The spacing is measured over the whole
  sequence; on a short excerpt of an early, sparse embryo it is too large and results degrade.
* Your own data: 3D only; provide the voxel size in &micro;m as (z, y, x); detections are never added or removed.

CE/02 (190 frames, 22,317 detections) takes about 3 minutes on a CPU (10 threads, 8 solver processes)
and under a minute on a GPU, including the extraction of image patches; mskcc s3 (370 frames,
111,175 detections) takes about 3.5 minutes on a GPU.

## Evaluation

`scripts/evaluate.py` calls `ctc_evaluate` from [py-ctcmetrics](https://pypi.org/project/py-ctcmetrics/) 1.3.3:

```bash
python scripts/evaluate.py --gt-dir /path/to/data/Fluo-N3DH-CE/02_GT --res-dir results/ce02_seed0
# = ctc_evaluate --gt <GT> --res <RES> --tra --det --lnk --ct --tf --bc 3 --cca
```

The Cell Linking Benchmark score is OP<sub>CLB</sub> = (LNK + BIO(i)) / 2 with
BIO(i) = mean(CT, TF, BC(i), CCA), where BC(i), the F1 score of division detection, accepts a detected
division within i frames of a reference division. The paper uses i = 1, as Ulman et al.
(Nature Methods 2017) did for Fluo-N3DH-CE. The scripts of this repository (`scripts/evaluate.py`,
`scripts/infer.py --gt-dir` and `scripts/reproduce_paper.py`) report only i = 3, that is BC(3), BIO(3)
and OP<sub>CLB</sub> = (LNK + BIO(3)) / 2. The full output of the `ctc_evaluate` command above lists
BC(i) and BIO(i) for i = 0-3, and the values of the paper are BC(1) and (LNK + BIO(1)) / 2 from that
output. `--mask-dir <detections>` additionally prints the label-free rule count.

## Training from scratch

1. Prepare the data (above) and fill in the paths of `configs/datasets_paper.yaml`
   (training sequences CE/01, mskcc s1, mskcc s2, in this order).
2. Run

   ```bash
   python scripts/train.py --datasets configs/datasets_paper.yaml --work-dir runs/paper --device cuda
   ```

   `--seeds 0` trains a single model. The work directory receives the patch tables (about 23 GB for the
   three training sequences), the checkpoints of every epoch, all intermediate solutions,
   `records.json` (label counts, rule counts per epoch, selected epochs, w<sub>u</sub>)
   and the final weights in `runs/paper/weights/seed<k>/`, which `scripts/infer.py --weights` accepts.
   Every step is resumable: rerun the same command after an interruption.
3. Track the test sequences with the new weights (birth cost 1.0, fixed by `cost.birth_deploy`) and
   evaluate them as above, e.g. with `scripts/reproduce_paper.py --weights-root runs/paper/weights`.

GPU work (training, scoring) runs in one process, one step after the other; the integer programs
are solved by CPU processes (`runtime.solver_workers`). In the paper, training one seed took 2.3 h on
one machine with one NVIDIA RTX PRO 6000 GPU and 8 CPU cores, including the preprocessing and the
integer programs solved for the labels and for the epoch selection.

With the verified environment on the same type of GPU, `scripts/train.py --seeds 0` reproduced the
released seed-0 model: identical training labels, identical rule counts for all 8 epochs, the same
selected epoch (2), and bit-identical triplet network and state head. It gives w<sub>u</sub> =
0.6153648115245174, whereas the released `fusion.json` has 0.6153642513919886: the released value was
fitted on another GPU (RTX A6000), where the triplet scores differ in the last bits. With either value
the tracks on CE/02 are identical. Other GPUs or library versions give slightly different weights.

## Main results

OP<sub>CLB</sub> = (LNK + BIO(1)) / 2 and BC(1) of the released weights (mean &plusmn; standard deviation over the 5 seeds,
birth cost 1.0), from the evaluation of the original runs. Recomputing all 25 runs (5 test sequences
&times; 5 seeds) and the geometry-only baselines with this code reproduced every output file byte for
byte (verified environment); on CE/02, `tests/equivalence_ce02.py` repeats this check on a CPU against
stored MD5 checksums.

| test sequence | detections | OP<sub>CLB</sub> | BC(1) | OP<sub>CLB</sub>, learned score off (`--geometry-only`) |
|---|---|---|---|---|
| CTC Fluo-N3DH-CE 02 | official 02_ERR_SEG | 0.9523 &plusmn; 0.0035 | 0.9264 &plusmn; 0.0089 | 0.8804 |
| mskcc-confocal s3 | drawn from the annotation | 0.9833 &plusmn; 0.0026 | 0.9695 &plusmn; 0.0057 | 0.9442 |
| nih-ls 1 / 2 / 3 (zero-shot, 270 frames) | drawn from the annotation | 0.9680 / 0.9809 / 0.9826 | 0.9284 / 0.9486 / 0.9560 | - |
| nih-ls, mean of the three embryos | | 0.9772 &plusmn; 0.0063 | 0.9444 &plusmn; 0.0176 | - |

Means with six decimals: OP<sub>CLB</sub> 0.952253 (CE/02), 0.983263 (mskcc s3), 0.977178 (nih-ls);
BC(1) 0.926372, 0.969518, 0.944361.
Per seed 0-4 (OP<sub>CLB</sub>): CE/02 0.949285, 0.954700, 0.954555, 0.955100, 0.947625;
mskcc s3 0.985860, 0.981795, 0.981895, 0.986315, 0.980450;
nih-ls (mean of the three embryos) 0.983848, 0.977160, 0.974165, 0.982468, 0.968250.
Mean division counts on CE/02 (true positive / false positive / false negative, BC(1) matching):
329.8 / 21.2 / 31.2 with the learned score, 276 / 99 / 85 without it.
With i = 3, which the scripts print, every value above stays the same except those of the
geometry-only baseline on CE/02 (OP<sub>CLB</sub> 0.8811, 278 / 97 / 83 divisions).
See the paper for the comparison with other methods.

To recompute the table: fill in the `test` paths of `configs/datasets_paper.yaml`, then

```bash
python scripts/reproduce_paper.py --datasets configs/datasets_paper.yaml --out results/paper --device cuda
python scripts/reproduce_paper.py --datasets configs/datasets_paper.yaml --out results/paper_geometry --geometry-only
```

Both commands print OP<sub>CLB</sub> with i = 3; for the i = 1 values, run the `ctc_evaluate` command of
the Evaluation section on the result folders `results/paper/<name>/seed<k>` and
`results/paper_geometry/<name>/geometry_only` (`<name>` as in `configs/datasets_paper.yaml`).

## Tests

```bash
python tests/test_units.py                                            # synthetic data, seconds
bash tests/smoke_inference.sh /path/to/data/Fluo-N3DH-CE /tmp/divtrack_smoke   # CE/01, 20 frames, CPU, < 1 min
bash tests/smoke_training.sh /path/to/data/Fluo-N3DH-CE /path/to/data/mskcc-s1 /tmp/divtrack_train_smoke python cuda
python tests/equivalence_ce02.py --ce-dir /path/to/data/Fluo-N3DH-CE --out /tmp/divtrack_equiv   # ~12 min on CPU
```

* `test_units.py`: data loading, candidates and integer program on a synthetic division, CTC output
  round trip, rules, split, network symmetry, fusion fit, birth costs (1.0 at deployment, 2.0 in the
  training solves), steps of the one-round training pipeline (runs with or without pytest).
* `smoke_training.sh`: the whole training pipeline on small subsets (first 80 frames of CE/01 and
  100 of mskcc s1) with short schedules (`tests/configs/smoke.yaml`), then inference and evaluation
  on the first 60 frames of CE/02. Checks that every step runs; the numbers are not meaningful.
* `equivalence_ce02.py`: tracks CE/02 with the five released models and the geometry-only baseline
  and compares the MD5 of `res_track.txt` and of all 190 masks with the outputs of the original
  research code (`tests/reference/ce02_outputs_md5.json`).

## Notes

* No ground truth is read during training, epoch selection or inference; `--gt-dir` is used only for evaluation.
* The training labels and the epoch selection depend only on the rules and on the detections, so
  they transfer to new data; the thresholds are counted in frames (about 1-1.5 min per frame for all
  data used here).
* Floating-point results of the networks differ slightly between CPU and GPU, between GPU models and
  between library versions. On CE/02 the tracks are identical; in general rare ties in the integer
  programs may resolve differently.
* With `swap_daughters_p: 0.5` (default) the training code consumes the random stream exactly as the
  code that produced the released weights; the swap itself does not change the output of the
  network, which is symmetric in the two daughters.

## Citation

The paper has been submitted to ISBI 2027 and is not yet accepted; until it is published, please cite it as

```bibtex
@unpublished{hu2026divtrack,
  title  = {Efficient Self-Supervised {3D} Cell Tracking by Integer Programming with Visual Evidence of Division},
  author = {Hu, Tong and Shigeta, Hironori and Zheng, Fuyu and Seno, Shigeto},
  note   = {Submitted to the IEEE International Symposium on Biomedical Imaging (ISBI) 2027},
  year   = {2026}
}
```

Please also cite the datasets: the Cell Tracking Challenge and Hirsch et al. (MICCAI 2022) for
mskcc-confocal and nih-ls (and Moyle et al., Nature 2021, for nih-ls).

## License

MIT License, see [LICENSE](LICENSE).
