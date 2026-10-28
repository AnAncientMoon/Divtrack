#!/usr/bin/env python
"""Convert the C. elegans datasets of Hirsch et al. (MICCAI 2022) to the CTC layout used by divtrack.

    mskcc-confocal  https://doi.org/10.5281/zenodo.6460303  (mskcc_confocal.zip, ~24 GB; series s1, s2, s3)
    nih-ls          https://doi.org/10.5281/zenodo.6460375  (nih_ls.zip, ~25 GB; samples 1, 2, 3)

Both provide images and tracks (tracks/tracks.txt: t z y x cell_id parent_id track_id radius ...),
but no segmentation. As in the paper, detections are drawn from the annotation ("ideal detection"):
every annotated nucleus becomes an ellipsoid with 0.8 x the annotated radius (larger nuclei drawn
first, no overwriting); the same masks are written as detections (01_ERR_SEG) and as tracking ground
truth (01_GT/TRA with man_track.txt rebuilt from the parent links). Images are symbolic links.

    python scripts/prepare_data/zenodo_to_ctc.py mskcc --src <unzipped>/mskcc-confocal/mskcc_confocal_s1 --name s1 --out data/mskcc-s1
    python scripts/prepare_data/zenodo_to_ctc.py nihls --src <unzipped>/nih-ls/nih_diSPIM_deconv_1 --name 1 --out data/nihls-1 \
        --max-frames 270 --relabel-base 1000

Paper settings: mskcc s1, s2 (training) and s3 (test) without relabelling; nih-ls 1-3 restricted to
the first 270 frames (as in Hirsch et al.; the embryo twitches later) and with the detection labels
randomly permuted in every frame (--relabel-base 1000 * sample, per-frame seed = base + t), so that
mask labels carry no identity information. Voxel size: mskcc 0.75 x 0.15 x 0.15 um, nih-ls 0.1625 um.
"""
import argparse
import collections
import json
import os

import numpy as np
import tifffile

SETTINGS = {
    # z_scale: tracks.txt stores z multiplied by this factor; image file prefix
    "mskcc": dict(voxel=[0.75, 0.15, 0.15], z_scale=5.0, image="images/mskcc_confocal_%s_t%03d.tif"),
    "nihls": dict(voxel=[0.1625, 0.1625, 0.1625], z_scale=1.0, image="images/nih_diSPIM_deconv_%s_t%03d.tif"),
}


def build_tracks(t, cid, pid):
    """CTC tracks from per-node parent links: a single child in the next frame continues the track,
    two children start two daughter tracks."""
    row = {int(c): i for i, c in enumerate(cid)}
    kids = collections.defaultdict(list)
    for i in range(len(cid)):
        if pid[i] >= 0 and pid[i] in row:
            kids[int(pid[i])].append(int(cid[i]))
    order = np.argsort(t, kind="stable")
    track_of, tracks, nxt = {}, {}, 1
    stats = dict(gap_splits=0, missing_parent=0, parents_gt2=0)
    for i in order:
        c, p = int(cid[i]), int(pid[i])
        if p >= 0 and p in row and len(kids[p]) == 1 and p in track_of and t[row[p]] == t[i] - 1:
            tid = track_of[p]
            track_of[c] = tid
            tracks[tid]["end"] = int(t[i])
            continue
        par = 0
        if p >= 0 and p in row:
            if len(kids[p]) == 2 and t[row[p]] == t[i] - 1:
                par = track_of[p]
            elif len(kids[p]) > 2:
                stats["parents_gt2"] += 1
                par = track_of[p]
            else:
                stats["gap_splits"] += 1          # single child not in the next frame: new track
        elif p >= 0:
            stats["missing_parent"] += 1
        tid = nxt
        nxt += 1
        track_of[c] = tid
        tracks[tid] = {"start": int(t[i]), "end": int(t[i]), "parent": int(par)}
    assert nxt < 65535
    stats["n_div"] = sum(1 for v in kids.values() if len(v) == 2)
    return track_of, tracks, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=sorted(SETTINGS))
    ap.add_argument("--src", required=True, help="folder of one series/sample (contains images/ and tracks/)")
    ap.add_argument("--name", required=True, help="series or sample id used in the image file names (s1, s2, s3 or 1, 2, 3)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--radius-scale", type=float, default=0.8)
    ap.add_argument("--max-frames", type=int, default=None, help="keep only frames t < max_frames (nih-ls: 270)")
    ap.add_argument("--relabel-base", type=int, default=None, help="randomly permute detection labels per frame (seed = base + t)")
    a = ap.parse_args()
    S = SETTINGS[a.dataset]
    rs = a.radius_scale
    tab = np.genfromtxt(os.path.join(a.src, "tracks", "tracks.txt"), names=True, dtype=None, encoding=None)
    t = tab["t"].astype(int)
    z = tab["z"].astype(float) / S["z_scale"]
    y, x, r = tab["y"].astype(float), tab["x"].astype(float), tab["radius"].astype(float)
    cid, pid = tab["cell_id"].astype(int), tab["parent_id"].astype(int)
    if a.max_frames is not None:
        keep = t < a.max_frames
        t, z, y, x, r, cid, pid = t[keep], z[keep], y[keep], x[keep], r[keep], cid[keep], pid[keep]
    track_of, tracks, stats = build_tracks(t, cid, pid)
    print("%s %s: %d frames, %d nodes, %d tracks, %d divisions, %s"
          % (a.dataset, a.name, t.max() + 1, len(t), len(tracks), stats["n_div"], stats), flush=True)
    for d in ("01", "01_ERR_SEG", os.path.join("01_GT", "TRA")):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    with open(os.path.join(a.out, "01_GT", "TRA", "man_track.txt"), "w") as f:
        for tid in sorted(tracks):
            f.write("%d %d %d %d\n" % (tid, tracks[tid]["start"], tracks[tid]["end"], tracks[tid]["parent"]))
    image = lambda T: os.path.join(os.path.abspath(a.src), S["image"] % (a.name, T))
    shape = tifffile.imread(image(0)).shape
    byt = collections.defaultdict(list)
    for i in range(len(t)):
        byt[int(t[i])].append(i)
    overlap = 0
    for T in range(t.max() + 1):
        vol = np.zeros(shape, np.uint16)
        for i in sorted(byt[T], key=lambda k: -r[k]):          # larger nuclei first
            lab = track_of[int(cid[i])]
            rxy = max(r[i] * rs, 1.0)
            rz = max(r[i] * rs / S["z_scale"], 1.0)
            z0, z1 = int(max(0, np.floor(z[i] - rz))), int(min(shape[0] - 1, np.ceil(z[i] + rz)))
            y0, y1 = int(max(0, np.floor(y[i] - rxy))), int(min(shape[1] - 1, np.ceil(y[i] + rxy)))
            x0, x1 = int(max(0, np.floor(x[i] - rxy))), int(min(shape[2] - 1, np.ceil(x[i] + rxy)))
            if z1 < z0 or y1 < y0 or x1 < x0:
                continue
            zz, yy, xx = np.mgrid[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
            e = ((zz - z[i]) / rz) ** 2 + ((yy - y[i]) / rxy) ** 2 + ((xx - x[i]) / rxy) ** 2 <= 1.0
            sub = vol[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
            overlap += int((e & (sub > 0)).sum())
            e &= sub == 0
            if not e.any():                                      # fully covered: keep at least the centre voxel
                e[min(int(round(z[i])) - z0, e.shape[0] - 1), min(int(round(y[i])) - y0, e.shape[1] - 1),
                  min(int(round(x[i])) - x0, e.shape[2] - 1)] = True
            sub[e] = lab
        tifffile.imwrite(os.path.join(a.out, "01_GT", "TRA", "man_track%03d.tif" % T), vol, compression="zlib")
        det = vol
        if a.relabel_base is not None:
            u = np.unique(vol)
            u = u[u > 0]
            perm = np.random.default_rng(a.relabel_base + T).permutation(len(u)) + 1
            lut = np.zeros(int(vol.max()) + 1, vol.dtype)
            lut[u] = perm.astype(vol.dtype)
            det = lut[vol]
        tifffile.imwrite(os.path.join(a.out, "01_ERR_SEG", "mask%03d.tif" % T), det, compression="zlib")
        dst = os.path.join(a.out, "01", "t%03d.tif" % T)
        if not os.path.lexists(dst):
            os.symlink(image(T), dst)
        if T % 50 == 0:
            print("  frame %d written (overlapping voxels so far: %d)" % (T, overlap), flush=True)
    meta = dict(dataset=a.dataset, name=a.name, frames=int(t.max() + 1), voxel_um=S["voxel"], radius_scale=rs,
                max_frames=a.max_frames, relabel_base=a.relabel_base, n_nodes=int(len(t)), n_tracks=len(tracks),
                overlap_voxels=overlap, **stats)
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("done:", a.out)


if __name__ == "__main__":
    main()
