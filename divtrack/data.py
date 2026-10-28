"""Reading one time-lapse sequence in Cell Tracking Challenge (CTC) layout.

    image_dir/  t000.tif t001.tif ...        3D images (z, y, x), one file per frame
    mask_dir/   mask000.tif mask001.tif ...  label masks of the given detections
                                             (0 = background, one positive label per nucleus;
                                             labels do not need to be consistent over time)

The detections are taken as given: nuclei are never added or removed, only linked.
Every length used by the method is expressed in units of the cell spacing s, the
median nearest-neighbour distance between detection centroids over all frames.
"""
import json
import os
import glob
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import tifffile
from scipy.spatial import cKDTree


def frame_files(folder, prefix):
    """{frame index: path} for files named <prefix><3 or 4 digits>.tif in ``folder``."""
    out = {}
    for path in glob.glob(os.path.join(folder, prefix + "*.tif")):
        stem = os.path.basename(path)[len(prefix):-len(".tif")]
        if stem.isdigit() and len(stem) in (3, 4):
            out[int(stem)] = path
    return out


def detections_from_mask(mask):
    """Labels (ascending), centroids (z, y, x in voxels, float64) and voxel counts of all objects."""
    labs = np.unique(mask)
    labs = labs[labs > 0]
    if labs.size == 0:
        return np.zeros(0, np.int64), np.zeros((0, 3), np.float64), np.zeros(0, np.int64)
    flat = mask.ravel()
    idx = np.flatnonzero(flat)
    vals = flat[idx]
    coords = np.array(np.unravel_index(idx, mask.shape), dtype=np.float64).T
    order = np.argsort(vals, kind="stable")
    vals, coords = vals[order], coords[order]
    lo = np.searchsorted(vals, labs, side="left")
    hi = np.searchsorted(vals, labs, side="right")
    cents = np.array([coords[b:e].mean(axis=0) for b, e in zip(lo, hi)], dtype=np.float64)
    return labs.astype(np.int64), cents, (hi - lo).astype(np.int64)


def _read_mask_detections(path):
    mask = tifffile.imread(path)
    if mask.ndim != 3:
        raise ValueError("only 3D label masks are supported: %s has shape %s" % (path, mask.shape))
    return detections_from_mask(mask) + (tuple(int(v) for v in mask.shape),)


def normalize_coords(coords_vox, voxel_um, spacing, center_um):
    """Voxel coordinates -> physical position relative to the volume centre, in units of s (float32)."""
    P = np.asarray(coords_vox, dtype=np.float64).reshape(-1, 3) * np.asarray(voxel_um, dtype=np.float64)
    return ((P - np.asarray(center_um, dtype=np.float64)) / max(spacing, 1e-6)).astype(np.float32)


class Sequence:
    """Detections, cell spacing and image access for one sequence.

    Args:
        image_dir, mask_dir: folders in CTC layout (see module docstring).
        voxel_um: voxel size (z, y, x) in micrometres.
        spacing_um: optional manual value of the cell spacing s; by default it is measured.
        cache_file: optional .npz used to cache the detections (centroids and volumes).
        workers: processes used to read the masks.
    """

    def __init__(self, image_dir, mask_dir, voxel_um, spacing_um=None, cache_file=None, workers=1, name="sequence"):
        self.name = name
        self.image_dir = os.path.abspath(os.path.expanduser(image_dir))
        self.mask_dir = os.path.abspath(os.path.expanduser(mask_dir))
        self.voxel = np.asarray(voxel_um, dtype=np.float64)
        if self.voxel.shape != (3,) or not (self.voxel > 0).all():
            raise ValueError("voxel_um must be three positive numbers (z, y, x), got %s" % (voxel_um,))
        self.mask_files = frame_files(self.mask_dir, "mask")
        self.image_files = frame_files(self.image_dir, "t")
        if not self.mask_files:
            raise FileNotFoundError("no mask###.tif files in %s" % self.mask_dir)
        self.frames = sorted(self.mask_files)
        missing = [t for t in self.frames if t not in self.image_files]
        if missing:
            raise FileNotFoundError("frames without an image (first 5): %s; expected t%03d.tif in %s"
                                    % (missing[:5], missing[0], self.image_dir))
        self._load_detections(cache_file, workers)
        self.spacing_auto = self._median_spacing()
        self.spacing = float(spacing_um) if spacing_um else self.spacing_auto
        # A fixed origin (the volume centre) keeps positions comparable across frames.
        self.center_um = np.asarray(self.vol_shape, dtype=np.float64) / 2.0 * self.voxel

    # ------------------------------------------------------------------ detections
    def _fingerprint(self, images=False):
        """Names, sizes and modification times of the input files (independent of the folder path)."""
        out = {}
        for kind, files in (("masks", self.mask_files),) + ((("images", self.image_files),) if images else ()):
            rows = []
            for t in self.frames:
                st = os.stat(files[t])
                rows.append([os.path.basename(files[t]), int(st.st_size), int(st.st_mtime_ns)])
            out[kind] = rows
        return json.dumps(out)

    def fingerprint(self):
        """Fingerprint of the inputs and of the measured quantities, used to validate cached work files."""
        return {"files": json.loads(self._fingerprint(images=True)), "voxel_um": self.voxel.tolist(),
                "spacing_um": self.spacing, "frames": [self.frames[0], self.frames[-1], len(self.frames)]}

    def _load_detections(self, cache_file, workers):
        fp = self._fingerprint()
        if cache_file and os.path.exists(cache_file):
            with np.load(cache_file, allow_pickle=False) as z:
                c = {key: z[key] for key in z.files}            # read every array once
            if str(c["fingerprint"]) == fp:
                off = c["offsets"]
                self.labels = {t: c["labels"][off[i]:off[i + 1]] for i, t in enumerate(self.frames)}
                self.coords = {t: c["coords"][off[i]:off[i + 1]] for i, t in enumerate(self.frames)}
                self.volumes = {t: c["volumes"][off[i]:off[i + 1]] for i, t in enumerate(self.frames)}
                self.vol_shape = tuple(int(v) for v in c["vol_shape"])
                return
        paths = [self.mask_files[t] for t in self.frames]
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
                res = list(ex.map(_read_mask_detections, paths, chunksize=4))
        else:
            res = [_read_mask_detections(p) for p in paths]
        shapes = {r[3] for r in res}
        if len(shapes) != 1:
            raise ValueError("masks of %s do not all have the same shape: %s" % (self.mask_dir, sorted(shapes)))
        self.vol_shape = res[0][3]
        self.labels = {t: r[0] for t, r in zip(self.frames, res)}
        self.coords = {t: r[1] for t, r in zip(self.frames, res)}
        self.volumes = {t: r[2] for t, r in zip(self.frames, res)}
        if cache_file:
            off = np.cumsum([0] + [len(self.labels[t]) for t in self.frames])
            tmp = cache_file + ".tmp.npz"
            np.savez(tmp, fingerprint=np.array(fp), offsets=off, vol_shape=np.array(self.vol_shape),
                     labels=np.concatenate([self.labels[t] for t in self.frames]),
                     coords=np.concatenate([self.coords[t] for t in self.frames]).reshape(-1, 3),
                     volumes=np.concatenate([self.volumes[t] for t in self.frames]))
            os.replace(tmp, cache_file)

    def _median_spacing(self):
        """Median distance (micrometres) from every detection to its nearest neighbour in the same frame."""
        vals = []
        for t in self.frames:
            P = self.coords[t] * self.voxel
            if len(P) >= 2:
                dd, _ = cKDTree(P).query(P, k=2)
                vals.extend(dd[:, 1].tolist())
        return float(np.median(vals)) if vals else float("nan")

    # ------------------------------------------------------------------ access
    def positions(self, t):
        """Centroids of frame t in units of s (float32, shape (n, 3))."""
        return normalize_coords(self.coords[t], self.voxel, self.spacing, self.center_um)

    def read_volume(self, t):
        v = np.asarray(tifffile.imread(self.image_files[t]))
        if v.ndim != 3:
            raise ValueError("only 3D images are supported: %s has shape %s" % (self.image_files[t], v.shape))
        return v

    def mask_path(self, t):
        return self.mask_files[t]

    def n_cells(self):
        return int(sum(len(self.labels[t]) for t in self.frames))


class CellTable:
    """All detections of a sequence in a fixed row order: frames ascending, labels ascending.

    Attributes: T (frame), LAB (mask label), COORD (voxel centroid), POS (float32 position in units
    of s), VOL (voxel count), and ``row[(t, label)]``.
    """

    def __init__(self, seq):
        self.frames = list(seq.frames)
        self.start, self.count = {}, {}
        T, LAB, COORD, POS, VOL = [], [], [], [], []
        n = 0
        for t in self.frames:
            labs = seq.labels[t]
            self.start[t], self.count[t] = n, len(labs)
            n += len(labs)
            T.append(np.full(len(labs), t, np.int64))
            LAB.append(np.asarray(labs, np.int64))
            COORD.append(np.asarray(seq.coords[t], np.float64).reshape(-1, 3))
            POS.append(seq.positions(t))
            VOL.append(np.asarray(seq.volumes[t], np.int64))
        self.T = np.concatenate(T)
        self.LAB = np.concatenate(LAB)
        self.COORD = np.concatenate(COORD)
        self.POS = np.concatenate(POS)
        self.VOL = np.concatenate(VOL)
        self.row = {(int(t), int(l)): i for i, (t, l) in enumerate(zip(self.T, self.LAB))}

    def __len__(self):
        return len(self.T)

    def rows(self, t):
        return slice(self.start[t], self.start[t] + self.count[t])

    def pos(self, t):
        return self.POS[self.rows(t)]

    def volume_dict(self):
        """{(t, label): voxel count}."""
        return {(int(t), int(l)): int(v) for t, l, v in zip(self.T, self.LAB, self.VOL)}
