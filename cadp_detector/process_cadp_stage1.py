#!/usr/bin/env python3
"""
process_cadp_stage1.py -- segment the processed CADP videos into accident clips and
non-accident footage from the stage-1 temporal annotations of the CADP release
(annotations_<timestamp>_stage1.json, Section 2.2 a of the release README), the way
extract_frames in analysis/annotations/utils.py of the CADP repository builds its clips.

    python process_cadp_stage1.py --stage1 annotations_stage1.json --cadp_root CADP \
           --videos /path/to/mp4s [--fps 10] [--cap 300] [--max_w 640] [--native]

Reads from <cadp_root>: id_mapping.json (youtube_id -> vid), split.json (video-level split),
processing_manifest.json (duration_s, src_fps per vid).

Writes to <cadp_root>:
    cadp_clip_index.csv         the clip number of the CADP release (the name of its stage-2 VATIC
                                file) for every segment of the stage-1 file, all 226 videos
    cadp_segments.json          per processed video: the stage-1 segments in seconds, with the
                                stage-1 id and the release clip number
    clips_manifest.csv          one row per extracted clip (accident or normal), with its split
    clips/<clip_uid>/<n>.jpg    frames at --fps, at most --cap per clip, longest side <= --max_w
    clips_native/<id:06d>/<k>.jpg   with --native: every native frame of every accident segment,
                                the layout of extract_frames, so that the stage-2 VATIC file
                                <id>.txt and generate_csv.py / create_crowd_far.py apply directly
    split_clips.json            {split: [release clip numbers]} for the detector
    accident_windows.json       {vid: [[start, end], ...]} in WORKING-SET frame indices of
                                extracted_frames/<vid>/ (fps --fps, cap --cap), for evaluate_cadp.py
    accident_windows_full.json  the same without the cap (full-length videos at --fps), for
                                temporal_baseline.py and temporal_references.py
    stage1_stats.json           the numbers quoted in Sections 6.1 and 6.2 of the thesis

Segment rule (identical to extract_frames): the keyframe timestamps of a segment are sorted and
the first two are its start and end. A segment with fewer than two keyframes is not a clip.
Clip numbering (identical to extract_frames): the videos are visited in file order of the JSON,
the segments of a video in order of start time, and the k-th segment overall is clip k (zero
based). Verified on the sample VATIC file 1407 of the release (221 annotated frames against a
7.35 s segment at 30 fps).

Clip identifiers of this thesis are <vid>_a<k> (accident, k-th segment in time order) or
<vid>_n<k> (normal footage, k-th gap). The split of a clip is the split of its video, so
no source video contributes to more than one partition.
"""
import os, sys, json, csv, argparse, subprocess, math, time, shutil


def run_ffmpeg(src, start, dur, out_pattern, fps, max_w):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
           "-i", src, "-vf", f"fps={fps},scale='min({max_w},iw)':-2", "-q:v", "4", out_pattern]
    return subprocess.run(cmd, capture_output=True, text=True).returncode


def extract_native(src, clip_rows, out_root):
    """extract_frames of analysis/annotations/utils.py for the segments of one video, with the
    release clip numbers of the rows as directory names. Returns the number of frames written."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("cannot open", src, file=sys.stderr)
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    endpoints = []
    for r in clip_rows:
        t0, t1 = int(r["start_s"] * fps), int(r["end_s"] * fps)
        endpoints.append([t0, t1 + 1, int(r["cadp_clip_id"])])
    endpoints.sort(key=lambda k: k[0])
    last = {}
    for _, _, cid in endpoints:
        os.makedirs(os.path.join(out_root, f"{cid:06d}"), exist_ok=True)
        last[cid] = 0
    counter, written = 0, 0
    stop = max(e[1] for e in endpoints)
    while counter <= stop:
        segs = [cid for s0, e0, cid in endpoints if s0 <= counter < e0]
        if not segs:
            if not cap.grab():
                break
        else:
            status, img = cap.read()
            if not status or img is None:
                break
            for cid in segs:
                cv2.imwrite(os.path.join(out_root, f"{cid:06d}", f"{last[cid]}.jpg"), img)
                last[cid] += 1; written += 1
        counter += 1
    cap.release()
    return written


def segment_endpoints(obj):
    """Start and end of a stage-1 segment in seconds, as extract_frames reads them, or None."""
    ts = sorted(float(k["frame"]) for k in obj.get("keyframes", []))
    if len(ts) < 2:
        return None
    return ts[0], ts[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--cadp_root", required=True)
    ap.add_argument("--videos", required=True, help="directory with <youtube_id>.mp4")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--cap", type=int, default=300, help="max frames per clip")
    ap.add_argument("--max_w", type=int, default=640)
    ap.add_argument("--neg_margin", type=float, default=1.0, help="seconds kept clear of every accident segment")
    ap.add_argument("--neg_min", type=float, default=4.0, help="minimum length of a normal-footage clip")
    ap.add_argument("--neg_max_len", type=float, default=30.0, help="normal-footage clips longer than this are cut into pieces")
    ap.add_argument("--neg_max_per_video", type=int, default=12)
    ap.add_argument("--skip_extract", action="store_true", help="only write the manifests and statistics")
    ap.add_argument("--native", action="store_true",
                    help="also extract every native frame of each accident segment into clips_native/<id:06d>/<k>.jpg")
    a = ap.parse_args()

    root = a.cadp_root
    ann = json.load(open(a.stage1))
    mapping = json.load(open(os.path.join(root, "id_mapping.json")))          # youtube_id -> vid
    split = json.load(open(os.path.join(root, "split.json")))
    inv_split = {v: sp for sp, vs in split.items() for v in vs}
    manifest = {m["vid"]: m for m in json.load(open(os.path.join(root, "processing_manifest.json")))}

    # ---- clip numbering of the release (extract_frames): file order of videos, start order within a video
    clip_index, index_rows, counter = {}, [], 0
    for vname, objs in ann.items():
        segs = []
        for o in objs:
            se = segment_endpoints(o)
            if se is None:
                continue
            segs.append((se[0], se[1], o.get("id")))
        segs.sort(key=lambda t: t[0])
        for s0, e0, sid in segs:
            clip_index[(vname, sid)] = counter
            index_rows.append([counter, vname, sid, round(s0, 3), round(e0, 3)])
            counter += 1
    with open(os.path.join(root, "cadp_clip_index.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["cadp_clip_id", "video", "stage1_id", "start_s", "end_s"]); wr.writerows(index_rows)

    segments, rows = {}, []
    stats = {"videos": len(mapping), "videos_in_stage1_file": 0, "videos_with_segments": 0,
             "segments": 0, "segments_dropped_single_keyframe": 0, "accident_seconds": 0.0, "footage_seconds": 0.0,
             "release_clips_total": counter,
             "per_split": {sp: {"videos": 0, "segments": 0, "accident_seconds": 0.0, "footage_seconds": 0.0,
                                "accident_clips": 0, "normal_clips": 0, "accident_frames": 0, "normal_frames": 0}
                           for sp in ("train", "val", "test")},
             "segments_inside_cap": 0, "segments_partially_inside_cap": 0, "segments_beyond_cap": 0,
             "videos_whole_single_segment": 0, "videos_multi_segment": 0, "fps": a.fps, "cap": a.cap}
    cap_s = a.cap / a.fps
    for yt, vid in sorted(mapping.items(), key=lambda kv: kv[1]):
        m = manifest[vid]
        dur = float(m["duration_s"])
        sp = inv_split[vid]
        vname = yt + ".mp4"
        objs = ann.get(vname)
        segs = []
        if objs is not None:
            stats["videos_in_stage1_file"] += 1
            for o in objs:
                se = segment_endpoints(o)
                if se is None:
                    stats["segments_dropped_single_keyframe"] += 1
                    continue
                s, e = max(0.0, se[0]), min(dur, se[1])
                if e <= s:
                    e = min(dur, s + 0.5)
                segs.append({"start_s": s, "end_s": e, "stage1_id": o.get("id"), "type": o.get("type"),
                             "cadp_clip_id": clip_index[(vname, o.get("id"))]})
        segs.sort(key=lambda x: x["start_s"])
        segments[vid] = {"youtube_id": yt, "split": sp, "duration_s": dur, "src_fps": m["src_fps"],
                         "in_stage1_file": objs is not None, "segments": segs}
        ps = stats["per_split"][sp]
        ps["videos"] += 1; ps["footage_seconds"] += dur; stats["footage_seconds"] += dur
        if segs:
            stats["videos_with_segments"] += 1
        if len(segs) > 1:
            stats["videos_multi_segment"] += 1
        if len(segs) == 1 and segs[0]["start_s"] <= 0.5 and segs[0]["end_s"] >= dur - 1.0:
            stats["videos_whole_single_segment"] += 1
        for k, sg in enumerate(segs):
            stats["segments"] += 1; ps["segments"] += 1
            L = sg["end_s"] - sg["start_s"]
            stats["accident_seconds"] += L; ps["accident_seconds"] += L
            if sg["start_s"] >= cap_s:
                stats["segments_beyond_cap"] += 1
            elif sg["end_s"] <= cap_s:
                stats["segments_inside_cap"] += 1
            else:
                stats["segments_partially_inside_cap"] += 1
            rows.append({"clip_uid": f"{vid}_a{k:02d}", "vid": vid, "youtube_id": yt, "split": sp, "kind": "accident",
                         "start_s": round(sg["start_s"], 3), "end_s": round(sg["end_s"], 3),
                         "duration_s": round(L, 3), "stage1_id": sg["stage1_id"], "cadp_clip_id": sg["cadp_clip_id"]})
        # normal footage: gaps between segments, with a margin, cut into pieces of <= neg_max_len
        gaps, cursor = [], 0.0
        for sg in segs:
            gaps.append((cursor, sg["start_s"] - a.neg_margin))
            cursor = max(cursor, sg["end_s"] + a.neg_margin)
        gaps.append((cursor, dur))
        k = 0
        for g0, g1 in gaps:
            while g1 - g0 >= a.neg_min and k < a.neg_max_per_video:
                piece = min(a.neg_max_len, g1 - g0)
                rows.append({"clip_uid": f"{vid}_n{k:02d}", "vid": vid, "youtube_id": yt, "split": sp, "kind": "normal",
                             "start_s": round(g0, 3), "end_s": round(g0 + piece, 3), "duration_s": round(piece, 3),
                             "stage1_id": "", "cadp_clip_id": ""})
                g0 += piece; k += 1

    # accident windows: working set (frame indices of extracted_frames/<vid>/, fps, cap) and full length
    windows, windows_full = {}, {}
    for vid, d in segments.items():
        n_ext = int(manifest[vid].get("frames_extracted", a.cap) or a.cap)
        ws, wf = [], []
        for sg in d["segments"]:
            s = int(round(sg["start_s"] * a.fps)); e = int(round(sg["end_s"] * a.fps))
            wf.append([s, e])
            if s >= n_ext:
                continue
            ws.append([max(0, s), max(0, min(e, n_ext - 1))])
        if ws:
            windows[vid] = ws
        if wf:
            windows_full[vid] = wf
    json.dump(segments, open(os.path.join(root, "cadp_segments.json"), "w"), indent=1)
    json.dump(windows, open(os.path.join(root, "accident_windows.json"), "w"), indent=1)
    json.dump(windows_full, open(os.path.join(root, "accident_windows_full.json"), "w"), indent=1)

    # frame extraction. Every clip directory carries _span.json (source span and sampling); a directory
    # whose marker is missing or differs from the manifest row is rebuilt, so a rerun after a change
    # of the segment rule or of the clip numbering never keeps frames of another span.
    clips_dir = os.path.join(root, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    t0 = time.time()
    for i, r in enumerate(rows):
        out = os.path.join(clips_dir, r["clip_uid"])
        expected = min(a.cap, int(math.floor(r["duration_s"] * a.fps)))
        span = {"youtube_id": r["youtube_id"], "start_s": r["start_s"], "end_s": r["end_s"],
                "fps": a.fps, "cap": a.cap, "max_w": a.max_w}
        marker = os.path.join(out, "_span.json")
        if a.skip_extract:
            r["n_frames"] = len([f for f in os.listdir(out) if f.endswith(".jpg")]) if os.path.isdir(out) else expected
        else:
            os.makedirs(out, exist_ok=True)
            existing = len([f for f in os.listdir(out) if f.endswith(".jpg")])
            current = os.path.exists(marker) and json.load(open(marker)) == span
            if existing == 0 or not current:
                if existing:
                    shutil.rmtree(out); os.makedirs(out)
                src = os.path.join(a.videos, r["youtube_id"] + ".mp4")
                dur = min(r["duration_s"], a.cap / a.fps)
                rc = run_ffmpeg(src, r["start_s"], dur, os.path.join(out, "%04d.jpg"), a.fps, a.max_w)
                if rc != 0:
                    print("ffmpeg failed for", r["clip_uid"], file=sys.stderr)
                else:
                    json.dump(span, open(marker, "w"))
            r["n_frames"] = len([f for f in os.listdir(out) if f.endswith(".jpg")])
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(rows)} clips, {time.time()-t0:.0f} s", flush=True)
        ps = stats["per_split"][r["split"]]
        ps[r["kind"] + "_clips"] += 1
        ps[r["kind"] + "_frames"] += r["n_frames"]
    if not a.skip_extract:
        wanted = {r["clip_uid"] for r in rows}
        for d in os.listdir(clips_dir):
            if d not in wanted and os.path.isdir(os.path.join(clips_dir, d)):
                shutil.rmtree(os.path.join(clips_dir, d))
                print("removed stale clip directory", d)
    with open(os.path.join(root, "clips_manifest.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["clip_uid", "vid", "youtube_id", "split", "kind", "start_s", "end_s",
                                           "duration_s", "n_frames", "stage1_id", "cadp_clip_id"])
        wr.writeheader(); wr.writerows(rows)
    # split membership by release clip number (for the detector trained on the native-frame clips)
    split_clips = {sp: sorted(f"{r['cadp_clip_id']:06d}" for r in rows if r["kind"] == "accident" and r["split"] == sp)
                   for sp in ("train", "val", "test")}
    json.dump(split_clips, open(os.path.join(root, "split_clips.json"), "w"), indent=1)
    # native-frame extraction in the repository layout, decoded frame by frame with OpenCV exactly
    # as extract_frames of the release does (frame k of clip c is the k-th decoded frame from
    # int(start * fps) to int(end * fps) inclusive), so that the stage-2 VATIC frame indices apply
    if a.native:
        nat = os.path.join(root, "clips_native")
        os.makedirs(nat, exist_ok=True)
        t1 = time.time()
        by_video = {}
        for r in rows:
            if r["kind"] == "accident":
                by_video.setdefault(r["youtube_id"], []).append(r)
        done = 0
        for i, (yt, rs) in enumerate(sorted(by_video.items())):
            todo = [r for r in rs if not (os.path.isdir(os.path.join(nat, f"{r['cadp_clip_id']:06d}"))
                                          and os.listdir(os.path.join(nat, f"{r['cadp_clip_id']:06d}")))]
            if not todo:
                continue
            n = extract_native(os.path.join(a.videos, yt + ".mp4"), todo, nat)
            done += len(todo)
            print(f"  native: {yt} {len(todo)} clips, {n} frames, {done} clips so far, {time.time()-t1:.0f} s", flush=True)
    stats["accident_clips"] = sum(1 for r in rows if r["kind"] == "accident")
    stats["normal_clips"] = sum(1 for r in rows if r["kind"] == "normal")
    stats["accident_frames"] = sum(r["n_frames"] for r in rows if r["kind"] == "accident")
    stats["normal_frames"] = sum(r["n_frames"] for r in rows if r["kind"] == "normal")
    stats["normal_seconds"] = sum(r["duration_s"] for r in rows if r["kind"] == "normal")
    json.dump(stats, open(os.path.join(root, "stage1_stats.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "per_split"}, indent=1))
    for sp, d in stats["per_split"].items():
        print(sp, d)


if __name__ == "__main__":
    main()
