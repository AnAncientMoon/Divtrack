#!/usr/bin/env python
"""Make a small CTC-style subset (first N frames) of a sequence for smoke tests.

    python tests/make_subset.py <image_dir> <mask_dir> <gt_dir or -> <N> <out_dir>

Images and masks are symbolic links; when a ground-truth folder is given, its TRA markers are
linked as well and man_track.txt is truncated to the first N frames. Output:
    out_dir/01, out_dir/01_ERR_SEG, out_dir/01_GT/TRA
"""
import glob
import os
import re
import sys


def link_frames(src, dst, pattern, n):
    os.makedirs(dst, exist_ok=True)
    k = 0
    for p in sorted(glob.glob(os.path.join(src, "*.tif"))):
        m = re.fullmatch(pattern, os.path.basename(p))
        if m and int(m.group(1)) < n:
            q = os.path.join(dst, os.path.basename(p))
            if not os.path.lexists(q):
                os.symlink(os.path.abspath(p), q)
            k += 1
    return k


def main():
    img, msk, gt, n, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    print("images", link_frames(img, os.path.join(out, "01"), r"t(\d{3,4})\.tif", n))
    print("masks", link_frames(msk, os.path.join(out, "01_ERR_SEG"), r"mask(\d{3,4})\.tif", n))
    if gt != "-":
        tra = os.path.join(out, "01_GT", "TRA")
        print("ground-truth markers", link_frames(os.path.join(gt, "TRA"), tra, r"man_track(\d{3,4})\.tif", n))
        rows = [l.split() for l in open(os.path.join(gt, "TRA", "man_track.txt")) if l.strip()]
        keep = {int(r[0]) for r in rows if int(r[1]) < n}
        with open(os.path.join(tra, "man_track.txt"), "w") as f:
            for L, B, E, P in ((int(a), int(b), int(c), int(d)) for a, b, c, d in rows):
                if L in keep:
                    f.write("%d %d %d %d\n" % (L, B, min(E, n - 1), P if P in keep else 0))
        print("ground-truth tracks", len(keep))


if __name__ == "__main__":
    main()
