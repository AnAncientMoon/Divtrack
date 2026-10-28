"""Image patches around nuclei.

A patch is a cube of physical side ``scale * s`` (s = cell spacing) centred on a detection
centroid, resampled to a fixed grid (11 x 32 x 32 by default) and normalised by its own
1% / 99.9% intensity quantiles. Because both the size and the grid are defined in physical
units relative to s, the same network sees comparable neighbourhoods on data with
different voxel sizes and cell densities.

Tables built per sequence (memory-mapped .npy files, rows in CellTable order):
    patches_f32.npy   (N, 11, 32, 32) float32  patch of every cell at its own frame t
    glimpse_prev.npy  (N, 11, 32, 32) float16  frame t-1 sampled at the cell's position at t
    glimpse_next.npy  (N, 11, 32, 32) float16  frame t+1 sampled at the cell's position at t
    glimpse_valid.npy (N, 3) bool              which of (t-1, t, t+1) exist
The triplet network scores candidates on the float32 patches; the features of the mother
state head (and the training of the triplet network) use the patches rounded to float16.
"""
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np


def extract_patches(vol, coords_vox, voxel_um, phys_size_um, out_shape, q_low=0.01, q_high=0.999):
    """Cubes of side ``phys_size_um`` around ``coords_vox`` (N, 3), resampled to ``out_shape``.

    Returns a float32 tensor of shape (N, 1, *out_shape). Out-of-volume parts are zero-padded.
    """
    import torch
    import torch.nn.functional as F

    Z, Y, X = vol.shape
    half_vox = (phys_size_um / 2.0) / np.asarray(voxel_um, dtype=np.float64)
    half_vox = np.maximum(half_vox, 1.0)
    half_vox = np.minimum(half_vox, np.array([Z, Y, X], dtype=np.float64) / 2.0)
    v = torch.from_numpy(vol.astype(np.float32))
    out = torch.zeros((len(coords_vox), 1, *out_shape), dtype=torch.float32)
    for i, c in enumerate(coords_vox):
        lo = np.floor(c - half_vox).astype(int)
        hi = np.ceil(c + half_vox).astype(int) + 1
        pad_lo = np.maximum(-lo, 0)
        pad_hi = np.maximum(hi - np.array([Z, Y, X]), 0)
        lo_c = np.maximum(lo, 0)
        hi_c = np.minimum(hi, np.array([Z, Y, X]))
        if np.any(hi_c <= lo_c):
            continue
        crop = v[lo_c[0]:hi_c[0], lo_c[1]:hi_c[1], lo_c[2]:hi_c[2]]
        if pad_lo.any() or pad_hi.any():
            crop = F.pad(crop, (int(pad_lo[2]), int(pad_hi[2]),
                                int(pad_lo[1]), int(pad_hi[1]),
                                int(pad_lo[0]), int(pad_hi[0])))
        crop = crop[None, None]
        crop = F.interpolate(crop, size=out_shape, mode="trilinear", align_corners=False)[0]
        lo_p = torch.quantile(crop, q_low)
        hi_p = torch.quantile(crop, q_high)
        crop = ((crop - lo_p) / (hi_p - lo_p + 1e-6)).clamp(0, 1)
        out[i] = crop
    return out


def patch_params(cfg):
    p = cfg["patches"]
    return float(p["scale"]), tuple(int(v) for v in p["shape"]), float(p["low_quantile"]), float(p["high_quantile"])


# --------------------------------------------------------------------- per-frame worker
_W = {}


def _init_worker(image_files, voxel, spacing, scale, shape, q_low, q_high, threads=None):
    if threads:
        import torch
        torch.set_num_threads(int(threads))
    _W.update(image_files=image_files, voxel=voxel, spacing=spacing, scale=scale, shape=shape, q_low=q_low, q_high=q_high)


def _frame_job(job):
    """Everything sampled from the image of frame ``t``.

    own:    cells of frame t (their patch at t)
    after:  cells of frame t+1 (frame t is their t-1 glimpse)
    before: cells of frame t-1 (frame t is their t+1 glimpse)
    """
    import tifffile
    t, own, after, before = job
    vol = np.asarray(tifffile.imread(_W["image_files"][t]))
    phys = _W["scale"] * _W["spacing"]

    def sample(c):
        if c is None:
            return None
        return extract_patches(vol, c, _W["voxel"], phys, _W["shape"], _W["q_low"], _W["q_high"]).numpy()[:, 0]

    return t, sample(own), sample(after), sample(before)


def _bounded_map(ex, fn, jobs, max_pending):
    """Like ``ex.map`` (results in order) but with at most ``max_pending`` jobs in flight, so that
    finished results do not pile up in memory."""
    import collections
    pending = collections.deque()
    it = iter(jobs)
    for job in it:
        pending.append(ex.submit(fn, job))
        if len(pending) >= max_pending:
            break
    while pending:
        res = pending.popleft().result()
        nxt = next(it, None)
        if nxt is not None:
            pending.append(ex.submit(fn, nxt))
        yield res


def build_patch_tables(seq, cells, cfg, out_dir, workers=1, threads_per_worker=1, glimpses=True, log=print):
    """Write the patch tables of a sequence into ``out_dir``; existing tables are kept."""
    scale, shape, q_low, q_high = patch_params(cfg)
    N = len(cells)
    os.makedirs(out_dir, exist_ok=True)
    p32 = os.path.join(out_dir, "patches_f32.npy")
    pprev, pnext, pval = (os.path.join(out_dir, n) for n in ("glimpse_prev.npy", "glimpse_next.npy", "glimpse_valid.npy"))
    need_own = not os.path.exists(p32)
    need_gl = glimpses and not os.path.exists(pval)
    if not need_own and not need_gl:
        return
    own = np.lib.format.open_memmap(p32 + ".tmp", mode="w+", dtype=np.float32, shape=(N, *shape)) if need_own else None
    if need_gl:
        prev = np.lib.format.open_memmap(pprev + ".tmp", mode="w+", dtype=np.float16, shape=(N, *shape))
        nxt = np.lib.format.open_memmap(pnext + ".tmp", mode="w+", dtype=np.float16, shape=(N, *shape))
        prev[:] = 0
        nxt[:] = 0
        valid = np.zeros((N, 3), bool)
        valid[:, 1] = True
    frames = set(cells.frames)
    jobs = []
    for t in cells.frames:
        c_own = cells.COORD[cells.rows(t)] if need_own else None
        c_after = cells.COORD[cells.rows(t + 1)] if (need_gl and t + 1 in frames) else None
        c_before = cells.COORD[cells.rows(t - 1)] if (need_gl and t - 1 in frames) else None
        jobs.append((t, c_own, c_after, c_before))
    init = (seq.image_files, seq.voxel, seq.spacing, scale, shape, q_low, q_high)
    ex = None
    if workers > 1:
        ex = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                 initializer=_init_worker, initargs=init + (threads_per_worker,))
        results = _bounded_map(ex, _frame_job, jobs, 2 * workers)
    else:
        _init_worker(*init)
        results = map(_frame_job, jobs)
    for done, (t, P_own, P_after, P_before) in enumerate(results, 1):
        if P_own is not None:
            own[cells.rows(t)] = P_own
        if P_after is not None and len(P_after):
            r = cells.rows(t + 1)
            prev[r] = P_after.astype(np.float16)
            valid[r, 0] = True
        if P_before is not None and len(P_before):
            r = cells.rows(t - 1)
            nxt[r] = P_before.astype(np.float16)
            valid[r, 2] = True
        if done % 50 == 0 or done == len(jobs):
            log("  patches %s: %d/%d frames" % (seq.name, done, len(jobs)))
    if ex is not None:
        ex.shutdown()
    if need_own:
        own.flush()
        del own
        os.replace(p32 + ".tmp", p32)
    if need_gl:
        prev.flush()
        nxt.flush()
        del prev, nxt
        os.replace(pprev + ".tmp", pprev)
        os.replace(pnext + ".tmp", pnext)
        np.save(pval, valid)


def load_patch_tables(out_dir, glimpses=True):
    """Memory-mapped patch tables written by :func:`build_patch_tables`."""
    p32 = np.load(os.path.join(out_dir, "patches_f32.npy"), mmap_mode="r")
    if not glimpses:
        return p32
    prev = np.load(os.path.join(out_dir, "glimpse_prev.npy"), mmap_mode="r")
    nxt = np.load(os.path.join(out_dir, "glimpse_next.npy"), mmap_mode="r")
    valid = np.load(os.path.join(out_dir, "glimpse_valid.npy"))
    return p32, prev, nxt, valid
