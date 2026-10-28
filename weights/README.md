# Released weights

Five models (seeds 0-4) of the final system of the paper, trained without annotation on the pooled
sequences CTC Fluo-N3DH-CE 01 + mskcc-confocal s1 + s2 with one round of self-training. These are the
files that produced the results reported in the paper; they are byte-identical to the files of the
original runs (MD5 below).

| seed | judge.pth (epoch) | state.pth (epoch) | w_u (fusion.json) |
|---|---|---|---|
| 0 | epoch 2 | epoch 9 | 0.6154 |
| 1 | epoch 5 | epoch 5 | 0.9506 |
| 2 | epoch 6 | epoch 7 | 0.7002 |
| 3 | epoch 4 | epoch 2 | 0.7181 |
| 4 | epoch 6 | epoch 4 | 0.9648 |

* The epoch of the triplet network was selected by the rule count (solutions of the training
  sequences with F = z); the training labels and the epoch selection were computed with the birth
  cost 2.0.
* The state head runs on the frozen encoder of the triplet network of the same seed; its epoch was
  selected by the balanced BCE on the development frames.
* w_u was fitted on the development frames with the training labels as positives.
* At deployment the birth cost is fixed at 1.0 (`cost.birth_deploy` in `configs/default.yaml`).

Provenance (files of the original research code, copied without any change):

| file | original file | created |
|---|---|---|
| `seed<k>/judge.pth` | `judge_r1_s<k>_e<epoch>.pth`, the checkpoint of the selected epoch of the training run | 2026-10-01 |
| `seed<k>/state.pth` | `state_uni1fx_s<k>.pth`, state head trained on the frozen encoder features of that network | 2026-10-02 |
| `seed<k>/fusion.json` | `fusion_oneround_s<k>.json`, w_u fitted on an NVIDIA RTX A6000 | 2026-10-19 |

Files of one seed:

* `judge.pth` - triplet network (`divtrack.models.TripletNet`), dict with `model` (state dict) and
  metadata (`gate` = sister/mother distance limits [3.0, 2.0], `label`, `seed`, `epoch`, `train_seqs`).
* `state.pth` - mother state head (`divtrack.models.MotherStateHead`, 4 attention heads), dict with
  `model` and metadata (`feat` = encoder whose features it uses, i.e. the triplet network of the same
  seed; `tag`, `seed`, `epoch`, `train_ds`).
* `fusion.json` - fusion weight of F = z + w_u * u. Only `w_u` is used. The file also stores fitting
  diagnostics (development-frame AUCs, sample counts, the unused intercept) and entries of a
  density-dependent reliability curve that the final model does not use (`r` is identically 1,
  `w_v` = 0, `form` = "zu").

MD5:

```
5642482d29e366e7ac480d783238f3fd  seed0/judge.pth
47101be063d1cf71564169dc5ebba8fb  seed0/state.pth
62299ed2508ab3aa505fb7e63996cc6c  seed0/fusion.json
e73bb1bfaad97c1d952667d50da5114e  seed1/judge.pth
b76c5dcb22b06874268dd7939f61cfca  seed1/state.pth
e89a05e1bcb5cc1a0f374fc64a4955ce  seed1/fusion.json
cb527f5eaa8b8f3bfbfabf3638f0289b  seed2/judge.pth
88577a84fe975ce4a46ba221dfb2f4e4  seed2/state.pth
9fe5b33fb3e470245ce3e7f399429e19  seed2/fusion.json
63726cad1bbb0e0f6b8a164af59619cd  seed3/judge.pth
caa80d6a12026c52654936997c491985  seed3/state.pth
e8a8f3d3b0d2ed36e3674dd3d8ea178d  seed3/fusion.json
bb798888c5ab12088d297f294e22c7fe  seed4/judge.pth
a8df2f57f4686529c3a7e2328954e712  seed4/state.pth
369c27200000bd60170f0720081587bb  seed4/fusion.json
```

Total size about 5.6 MB, so the weights are kept in the repository.
