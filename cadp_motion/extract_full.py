#!/usr/bin/env python3
"""extract_full.py -- full-length frames at --fps for the videos of the given splits
(for the training-free temporal baseline, which needs the whole video, not the capped
working set).   python extract_full.py --cadp_root CADP --videos mp4dir --splits val test"""
import os, json, argparse, subprocess, time
ap = argparse.ArgumentParser()
ap.add_argument("--cadp_root", required=True); ap.add_argument("--videos", required=True)
ap.add_argument("--splits", nargs="+", default=["val", "test"])
ap.add_argument("--fps", type=float, default=10.0); ap.add_argument("--max_w", type=int, default=320)
a = ap.parse_args()
mapping = json.load(open(os.path.join(a.cadp_root, "id_mapping.json")))
inv = {v: k for k, v in mapping.items()}
split = json.load(open(os.path.join(a.cadp_root, "split.json")))
out_root = os.path.join(a.cadp_root, "full_frames"); os.makedirs(out_root, exist_ok=True)
t0 = time.time(); n = 0
for sp in a.splits:
    for vid in split[sp]:
        out = os.path.join(out_root, vid)
        if os.path.isdir(out) and os.listdir(out):
            continue
        os.makedirs(out, exist_ok=True)
        subprocess.run(["nice", "-n", "10", "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                        "-i", os.path.join(a.videos, inv[vid] + ".mp4"),
                        "-vf", f"fps={a.fps},scale='min({a.max_w},iw)':-2", "-q:v", "5", os.path.join(out, "%05d.jpg")])
        n += 1
        print(f"  {vid} ({sp}) {len(os.listdir(out))} frames, {time.time()-t0:.0f} s", flush=True)
print("done", n, "videos")
