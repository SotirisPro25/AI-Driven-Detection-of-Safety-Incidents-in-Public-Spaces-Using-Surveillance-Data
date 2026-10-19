#!/usr/bin/env python3
"""
make_accident_windows.py -- ground-truth accident windows for evaluate_cadp.py.

    python make_accident_windows.py --cadp_root CADP [--stage1 annotations_stage1.json]
                                    [--eval_fps 10] [--cap 300]

Two sources, combined when both exist:
  1. The stage-1 temporal annotations of the CADP release (segment start and end in
     seconds, one entry per accident clip). Every segment becomes one window. The endpoints
     are read as extract_frames of the release reads them: the keyframe timestamps are sorted
     and the first two are the start and the end; a segment with fewer than two keyframes is
     not a clip.
  2. The stage-2 VATIC files in <cadp_root>/annotations_vatic/, named by the release clip number
     (<id>.txt, the files as distributed; the clip number is resolved to its video and segment
     start through clips_manifest.csv) or, for files prepared per video, by the six-digit video
     id (<vid>.txt). The span of frames carrying the "Colliding" attribute is the precise
     collision window. Its frame indices count from the start of the clip at the native frame
     rate, so they are shifted by the segment start. The Colliding span replaces the stage-1
     segment that contains it (or is added when no segment contains it).

Writes <cadp_root>/accident_windows.json as {vid: [[start, end], ...]} in WORKING-SET frame
indices (extracted_frames/<vid>/<n>.jpg sampled at --eval_fps with a cap of --cap frames), and
a conditions.csv template (vid, camera_type, illumination) if none exists.
"""
import os, sys, json, csv, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accident_dataset_toolkit import load_cadp_vatic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadp_root", required=True)
    ap.add_argument("--stage1", default="", help="stage-1 JSON of the CADP release")
    ap.add_argument("--eval_fps", type=float, default=10.0)
    ap.add_argument("--cap", type=int, default=300)
    a = ap.parse_args()
    manifest = {m["vid"]: m for m in json.load(open(os.path.join(a.cadp_root, "processing_manifest.json")))}
    mapping = json.load(open(os.path.join(a.cadp_root, "id_mapping.json")))      # youtube_id -> vid
    windows = {}

    def clip(vid, s, e):
        n_ext = int(manifest.get(vid, {}).get("frames_extracted", a.cap) or a.cap)
        if s >= n_ext:
            return None
        return [max(0, s), max(0, min(e, n_ext - 1))]

    n_stage1 = 0
    if a.stage1:
        ann = json.load(open(a.stage1))
        for yt, vid in mapping.items():
            for o in ann.get(yt + ".mp4") or []:
                fr = sorted(float(k["frame"]) for k in o.get("keyframes", []))
                if len(fr) < 2:          # extract_frames of the release: fewer than two endpoints, not a clip
                    continue
                w = clip(vid, int(round(fr[0] * a.eval_fps)), int(round(fr[1] * a.eval_fps)))   # first two timestamps
                if w:
                    windows.setdefault(vid, []).append(w); n_stage1 += 1
    anno_dir = os.path.join(a.cadp_root, "annotations_vatic")
    by_clip = {}                                    # release clip number -> (vid, segment start in s)
    cm = os.path.join(a.cadp_root, "clips_manifest.csv")
    if os.path.exists(cm):
        for r in csv.DictReader(open(cm)):
            if r["kind"] == "accident" and r["cadp_clip_id"] != "":
                by_clip[int(r["cadp_clip_id"])] = (r["vid"], float(r["start_s"]))
    n_vatic = 0
    if os.path.isdir(anno_dir):
        for f in sorted(os.listdir(anno_dir)):
            if not f.endswith(".txt"):
                continue
            name = f[:-4]
            if name in manifest:                    # one file per video, frames count from the video start
                vid, offset_s = name, 0.0
            elif name.isdigit() and int(name) in by_clip:   # the release's naming, frames count from the segment start
                vid, offset_s = by_clip[int(name)]
            else:
                print("skipping", f, "(neither a video id nor a clip number of the processed corpus)")
                continue
            src_fps = float(manifest.get(vid, {}).get("src_fps", 30.0) or 30.0)
            c = load_cadp_vatic(os.path.join(anno_dir, f), clip_id=name, fps=src_fps)
            if c["accident_window"] is None:
                continue
            w = clip(vid, int(round((offset_s + c["accident_window"][0] / src_fps) * a.eval_fps)),
                     int(round((offset_s + c["accident_window"][1] / src_fps) * a.eval_fps)))
            if not w:
                continue
            n_vatic += 1
            ws = windows.setdefault(vid, [])
            replaced = False
            for i, (s, e) in enumerate(ws):
                if s <= w[0] <= e or s <= w[1] <= e:
                    ws[i] = w; replaced = True; break
            if not replaced:
                ws.append(w)
    for vid in windows:
        windows[vid].sort()
    out = os.path.join(a.cadp_root, "accident_windows.json")
    json.dump(windows, open(out, "w"), indent=1)
    print(f"{n_stage1} stage-1 segment windows, {n_vatic} VATIC Colliding spans -> {len(windows)} videos in {out}")
    cond = os.path.join(a.cadp_root, "conditions.csv")
    if not os.path.exists(cond):
        with open(cond, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["vid", "camera_type", "illumination"])
            for vid in sorted(manifest):
                wr.writerow([vid, "", ""])
        print(f"template written: {cond}  (camera_type in fixed/dashcam/broadcast, illumination in day/night)")


if __name__ == "__main__":
    main()
