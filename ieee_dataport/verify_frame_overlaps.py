#!/usr/bin/env python3
"""
verify_frame_overlaps.py -- Finding 17 of Section 2.3.4: clip pairs of different partitions that show the same
source frames, found from the file names and confirmed on the pixels.

    python verify_frame_overlaps.py --root IEEE [--listing listing.txt] [--out shared_frames.csv]

1. Candidates. ieee_audit_tools.frame_overlaps() lists every pair of clips of the same source title, in two
   different partitions, whose frame ranges may intersect. Frame numbers in a name count from the start of the
   segment that the name gives (-splitted-A-B with B - A > 60 s), so pairs from different segments are compared
   in source time over frame rates from 23 to 31 fps with 10 s of slack.
2. Decoding. Every candidate clip is decoded to grey frames of 96 x 54 pixels (OpenCV, area interpolation).
3. Alignment. For each pair the mean squared difference between frame i of one clip and frame i + lag of the
   other is computed for every lag with at least 3 frames in common, and the lag with the smallest mean is kept.
4. Verdict. Matching frames could also come from a static camera filmed at another time, so the test is on
   motion: the frame-to-frame differences of the two clips are compared at the best lag, and their squared
   difference divided by their mean energy (the motion residual) is about 0 for the same frames and about 2 for
   unrelated frames. A pair shares frames when the residual is below 1 over at least 5 common frames. On the
   supplied partition the 390 candidate pairs fall in two groups, residuals of at most 0.77 (78 pairs: 39
   training and validation, 31 training and testing, 8 validation and testing) and of at least 1.85.

--root is the corpus tree <root>/<partition>/<class>/<clip>.mp4 (a doubled partition level is accepted).
--listing, a text file with one partition/class/name.mp4 per line, replaces the names of the tree when the
candidates are to be drawn from the full corpus while only the candidate clips are present under --root.

Output: one row per checked pair with the best lag, the frames in common, the mean squared difference, the
motion residual and the verdict, and the count per pair of partitions on standard output.
"""
import argparse, collections, csv, itertools, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ieee_audit_tools import PARTS, enumerate_tree, read_listing, frame_overlaps

W, H, MINOV, MIN_SHARED, RESIDUAL = 96, 54, 3, 5, 1.0


def decode(path):
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (W, H), interpolation=cv2.INTER_AREA))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded: {path}")
    return np.stack(frames).astype(np.float32), fps


def compare(A, B):
    """best lag (frame i of A is frame i + lag of B), frames in common, mean squared difference, motion residual"""
    a, b = A.reshape(len(A), -1), B.reshape(len(B), -1)
    D = np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T, 0) / a.shape[1]
    lags = range(-(len(A) - MINOV), len(B) - MINOV + 1)
    means = [np.diagonal(D, offset=L).mean() for L in lags]
    k = int(np.argmin(means))
    lag = lags[k]
    common = len(np.diagonal(D, offset=lag))
    i0, j0 = max(0, -lag), max(0, lag)
    da, db = np.diff(A[i0:i0 + common], axis=0), np.diff(B[j0:j0 + common], axis=0)
    energy = 0.5 * (np.mean(da * da) + np.mean(db * db))
    residual = float(np.mean((da - db) ** 2) / energy) if energy > 1e-6 else float("nan")
    return lag, common, float(means[k]), residual


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--listing")
    ap.add_argument("--out", default="ieee_shared_frames_check.csv")
    a = ap.parse_args()

    tree = enumerate_tree(a.root)
    rows = read_listing(a.listing) if a.listing else tree
    files = {p: [r["name"] for r in rows if r["partition"] == p] for p in PARTS}
    where = collections.defaultdict(list)                      # (partition, name) -> [(class, path)]
    for r in tree:
        where[(r["partition"], r["name"])].append((r["cls"], r["path"]))

    cands = []
    for x, y in itertools.combinations(PARTS, 2):
        for o in frame_overlaps(files, x, y):
            cands.append((x, o["left"], y, o["right"], o["same_segment"]))
    missing = sorted({(p, n) for x, l, y, r, _ in cands for p, n in ((x, l), (y, r)) if (p, n) not in where})
    if missing:
        sys.exit(f"{len(missing)} candidate clips are not under --root, first: {missing[0]}")
    print(f"{len(cands)} candidate pairs over {len({(x, l) for x, l, *_ in cands} | {(y, r) for _, _, y, r, _ in cands})} clips")

    cache, out = {}, []
    def frames(path):
        if path not in cache:
            if len(cache) > 80:
                cache.clear()
            cache[path] = decode(path)
        return cache[path]
    for x, l, y, r, same in cands:
        for (cl, pl), (cr, pr) in itertools.product(where[(x, l)], where[(y, r)]):
            (A, fa), (B, fb) = frames(pl), frames(pr)
            if abs(fa - fb) > 0.01 * fa:
                print(f"  frame rates differ ({fa:.2f}, {fb:.2f}), compared frame by frame: {l} | {r}")
            lag, common, msd, res = compare(A, B)
            shares = bool(res < RESIDUAL and common >= MIN_SHARED)
            out.append(dict(partition_a=x, class_a=cl, name_a=l, partition_b=y, class_b=cr, name_b=r,
                            same_segment=same, lag=lag, frames_in_common=common, mean_sq_diff=round(msd, 2),
                            motion_residual=round(res, 3), shares_frames=shares))
    with open(a.out, "w", newline="", encoding="utf8") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)
    yes = [o for o in out if o["shares_frames"]]
    print(f"{len(out)} pairs checked, {len(yes)} share frames -> {a.out}")
    for x, y in itertools.combinations(PARTS, 2):
        s = [o for o in yes if (o["partition_a"], o["partition_b"]) == (x, y)]
        print(f"  {x} / {y}: {len(s)} pairs, {len({o['name_b'] for o in s})} {y} clips")
    ok = [o["motion_residual"] for o in out if o["shares_frames"]]
    no = [o["motion_residual"] for o in out if not o["shares_frames"] and o["frames_in_common"] >= MIN_SHARED
          and o["motion_residual"] == o["motion_residual"]]
    if ok and no:
        print(f"  motion residual: at most {max(ok):.2f} for shared pairs, at least {min(no):.2f} for the others")


if __name__ == "__main__":
    main()
