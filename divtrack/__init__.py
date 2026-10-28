"""divtrack: lineage tracking of nuclei with a learned division score, trained without annotation.

Given 3D images and per-frame nucleus detections (label masks), divtrack links the
detections of consecutive frames by solving one 0-1 integer program per frame pair.
A division is a hyperedge (mother -> two daughters) whose cost contains a score
F = z + w_u * u learned from images only:
  z  triplet network that looks at the mother and the two candidate daughters,
  u  mother state head that looks at the mother at t-1, t and t+1.
Both are trained by one round of self-training on the tracker's own lineage (the solution with a
purely geometric division score), filtered by simple biological rules; no manual annotation is used.

The package keeps imports light (no torch at import time) so that solver worker
processes start quickly.
"""

__version__ = "1.0.0"
