#!/usr/bin/env python3
"""
motion_features.py -- per-frame motion signals for the training-free temporal baseline and
the clip-level motion classifier on CADP (Sections 6.3.4 and 6.4.4).

    python motion_features.py --frames_root CADP/clips --out CADP/motion/clips.npz [--workers 2]
    python motion_features.py --frames_root CADP/full_frames --out CADP/motion/full.npz

For every directory of frames (<name>/<n>.jpg, time order) it computes, per frame:
    diff      mean absolute difference between consecutive grayscale frames (320 px wide)
    flow_mean mean magnitude of dense optical flow (Farneback) between consecutive frames
    flow_p90  90th percentile of the flow magnitude
    flow_div  mean absolute divergence of the flow field (expansion / contraction)
and stores them in one compressed npz keyed by directory name. Nothing is learned here.
"""
import os, sys, argparse, time
import numpy as np
import cv2
from multiprocessing import Pool

W = 320


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    if w != W:
        img = cv2.resize(img, (W, int(round(h * W / w))), interpolation=cv2.INTER_AREA)
    return img


def features_for_dir(d):
    files = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
    prev = None
    rows = []
    for f in files:
        g = load_gray(os.path.join(d, f))
        if g is None:
            continue
        if prev is None or prev.shape != g.shape:
            rows.append((0.0, 0.0, 0.0, 0.0))
        else:
            diff = float(np.mean(np.abs(g.astype(np.int16) - prev.astype(np.int16))))
            flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            dx = np.gradient(flow[..., 0], axis=1); dy = np.gradient(flow[..., 1], axis=0)
            rows.append((diff, float(mag.mean()), float(np.percentile(mag, 90)), float(np.mean(np.abs(dx + dy)))))
        prev = g
    return os.path.basename(d), np.array(rows, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--update", action="store_true",
                    help="keep the entries of an existing --out whose directory is unchanged (same frame count, no newer frame), recompute the rest")
    a = ap.parse_args()
    dirs = sorted(os.path.join(a.frames_root, d) for d in os.listdir(a.frames_root)
                  if os.path.isdir(os.path.join(a.frames_root, d)))
    t0 = time.time()
    out = {}
    if a.update and os.path.exists(a.out):
        prev = np.load(a.out)
        prev_time = os.path.getmtime(a.out)
        keep = []
        for d in dirs:
            name = os.path.basename(d)
            files = [f for f in os.listdir(d) if f.endswith(".jpg")]
            newest = max((os.path.getmtime(os.path.join(d, f)) for f in files), default=0.0)
            if name in prev.files and len(prev[name]) == len(files) and newest < prev_time:
                out[name] = prev[name]; keep.append(d)
        dirs = [d for d in dirs if d not in set(keep)]
        print(f"update: {len(out)} directories kept, {len(dirs)} to compute")
    with Pool(a.workers) as pool:
        for i, (name, arr) in enumerate(pool.imap_unordered(features_for_dir, dirs, chunksize=4)):
            out[name] = arr
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(dirs)} dirs, {time.time()-t0:.0f} s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f"wrote {a.out}: {len(out)} directories, {sum(len(v) for v in out.values())} frames, {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
