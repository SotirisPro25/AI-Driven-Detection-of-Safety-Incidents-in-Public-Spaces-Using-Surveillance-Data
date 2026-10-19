"""
accident_dataset_toolkit.py
===========================
A reusable toolkit to PROCESS, CHARACTERISE, and COMPARE traffic-accident
surveillance datasets for the thesis:
  "AI-Driven Detection of Safety Incidents in Public Spaces Using Surveillance Data"

Design goal
-----------
Two datasets with different native formats (CADP = VATIC per-video track files;
IEEE = TBD) are each normalised into ONE common schema, so that:
  * the same statistics are computed for both, and
  * the same evaluation metrics (detection mAP@0.5, and temporal accident-window
    metrics) are computed for both,
enabling a fair side-by-side comparison.

Common normalised schema (per clip)
-----------------------------------
Clip = {
  "clip_id": str,
  "dataset": str,                  # "CADP" | "IEEE" | ...
  "width": int, "height": int,     # source resolution (may be None if unknown)
  "num_frames": int,
  "fps": float,                    # nominal frames per second (for time conversions)
  "objects": [ Track, ... ],       # object tracks (real objects only)
  "mask_regions": [ Track, ... ],  # CrowdRegion / FarRegion etc. (excluded from detection)
  "accident_window": (start_f, end_f) or None,   # frame span of the incident, inclusive
}
Track = {
  "track_id": int,
  "label": str,                    # normalised class name (see CLASS_CANON)
  "frames": { frame_id(int): Box } # Box = {"bbox": [x1,y1,x2,y2], "visible": bool, "attributes": [str]}
}

Everything downstream (stats, plots, metrics) operates on this schema only,
so adding the IEEE loader is the ONLY new code needed when that data arrives.
"""

from __future__ import annotations
import os, re, json, math
from collections import defaultdict, Counter

# --------------------------------------------------------------------------
# Class canonicalisation: map every dataset's native labels onto one taxonomy
# so the two datasets can be compared on equal terms. Extend when IEEE arrives.
# --------------------------------------------------------------------------
CLASS_CANON = {
    # CADP native -> canonical
    "Car": "vehicle",
    "Bus": "vehicle",
    "Three-wheeler": "vehicle",
    "Others": "vehicle",
    "Two-wheeler": "two_wheeler",
    "Person": "person",
    # region labels (not objects)
    "FarRegion": "__region__",
    "CrowdRegion": "__region__",
    "Separator": "__ignore__",
}
REGION_CANON = "__region__"
IGNORE_CANON = "__ignore__"


def canon_label(native: str) -> str:
    return CLASS_CANON.get(native, native.lower())


# ==========================================================================
# LOADERS  (one per dataset format) -> produce Clip dicts in the common schema
# ==========================================================================

_QUOTED = re.compile(r'"([^"]*)"')


def load_cadp_vatic(vatic_path, clip_id=None, width=None, height=None,
                    fps=30.0, accident_attr="Colliding"):
    """
    Load one CADP VATIC .txt file into a normalised Clip.
    VATIC line: trackId xmin ymin xmax ymax frame lost occluded generated "label" ["attr"...]
    """
    if clip_id is None:
        clip_id = os.path.splitext(os.path.basename(vatic_path))[0]

    tracks = {}         # track_id -> dict
    max_frame = -1
    acc_start, acc_end = None, None

    with open(vatic_path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            quoted = _QUOTED.findall(line)
            head = line.split('"', 1)[0].split()
            if len(head) < 9 or not quoted:
                continue

            tid = int(head[0])
            xmin, ymin, xmax, ymax = map(int, head[1:5])
            frame_id = int(head[5])
            lost = int(head[6])
            occluded = int(head[7])
            attrs = quoted[1:]
            native_label = quoted[0]

            max_frame = max(max_frame, frame_id)

            if tid not in tracks:
                tracks[tid] = {"track_id": tid, "native_label": native_label,
                               "label": canon_label(native_label), "frames": {}}
            tracks[tid]["frames"][frame_id] = {
                "bbox": [xmin, ymin, xmax, ymax],   # already x1,y1,x2,y2
                "visible": (lost == 0),
                "occluded": bool(occluded),
                "attributes": attrs,
            }
            if accident_attr in attrs:
                acc_start = frame_id if acc_start is None else min(acc_start, frame_id)
                acc_end = frame_id if acc_end is None else max(acc_end, frame_id)

    objects, mask_regions, separators = [], [], []
    for t in tracks.values():
        if t["label"] == REGION_CANON:
            mask_regions.append(t)
        elif t["label"] == IGNORE_CANON:
            separators.append(t)            # "Separator" tracks delimit the accident span
        else:
            objects.append(t)
    # analysis/annotations.find_boundary of the CADP repository: with at least two Separator
    # tracks, the span runs from the earliest to the latest first visible frame of a separator;
    # otherwise the whole clip counts ([0, -1])
    starts = []
    for t in separators:
        vis = [f for f, fr in t["frames"].items() if fr["visible"]]
        if vis:
            starts.append(min(vis))
    separator_boundary = [min(starts), max(starts)] if len(starts) >= 2 else [0, -1]

    # infer resolution from max coordinates if not supplied
    if width is None or height is None:
        mx = my = 0
        for t in tracks.values():
            for fr in t["frames"].values():
                x1, y1, x2, y2 = fr["bbox"]
                mx = max(mx, x2); my = max(my, y2)
        width = width or _round_res(mx)
        height = height or _round_res(my)

    return {
        "clip_id": str(clip_id),
        "dataset": "CADP",
        "width": width, "height": height,
        "num_frames": max_frame + 1,
        "fps": fps,
        "objects": objects,
        "mask_regions": mask_regions,
        "accident_window": (acc_start, acc_end) if acc_start is not None else None,
        "separator_boundary": separator_boundary,
    }


def _round_res(v):
    # snap an observed max coordinate up to a plausible standard dimension
    for std in (240, 268, 320, 344, 360, 480, 512, 516, 640, 720, 1080, 1280):
        if v <= std:
            return std
    return int(v)




# ==========================================================================
# STATISTICS  (identical computation for any Clip, any dataset)
# ==========================================================================

def clip_stats(clip):
    """Per-clip statistics on the common schema."""
    n_frames = clip["num_frames"]
    fps = clip["fps"] or 30.0
    W, H = clip["width"], clip["height"]

    label_counter = Counter()          # instances (track-level)
    box_area_frac = []                 # each visible box area / frame area
    visible_boxes = 0
    frame_area = (W * H) if (W and H) else None

    for t in clip["objects"]:
        label_counter[t["label"]] += 1
        for fr in t["frames"].values():
            if not fr["visible"]:
                continue
            visible_boxes += 1
            if frame_area:
                x1, y1, x2, y2 = fr["bbox"]
                a = max(0, x2 - x1) * max(0, y2 - y1)
                box_area_frac.append(a / frame_area)

    aw = clip["accident_window"]
    if aw:
        acc_len_frames = aw[1] - aw[0] + 1
        acc_len_sec = acc_len_frames / fps
    else:
        acc_len_frames = acc_len_sec = 0

    return {
        "clip_id": clip["clip_id"],
        "dataset": clip["dataset"],
        "resolution": f"{W}x{H}" if (W and H) else "unknown",
        "num_frames": n_frames,
        "duration_s": round(n_frames / fps, 2),
        "num_object_tracks": len(clip["objects"]),
        "num_mask_regions": len(clip["mask_regions"]),
        "class_instances": dict(label_counter),
        "visible_boxes": visible_boxes,
        "has_accident_window": aw is not None,
        "accident_len_frames": acc_len_frames,
        "accident_len_s": round(acc_len_sec, 2),
        "mean_box_area_frac": round(sum(box_area_frac) / len(box_area_frac), 4) if box_area_frac else None,
        "min_box_area_frac": round(min(box_area_frac), 5) if box_area_frac else None,
    }


def dataset_stats(clips):
    """Aggregate statistics across a list of Clips (one dataset)."""
    if not clips:
        return {}
    per = [clip_stats(c) for c in clips]
    total_frames = sum(p["num_frames"] for p in per)
    fps0 = clips[0]["fps"] or 30.0
    class_totals = Counter()
    for p in per:
        class_totals.update(p["class_instances"])
    n_acc = sum(1 for p in per if p["has_accident_window"])
    res_set = Counter(p["resolution"] for p in per)
    return {
        "dataset": clips[0]["dataset"],
        "num_clips": len(clips),
        "total_frames": total_frames,
        "total_duration_min": round(total_frames / fps0 / 60, 2),
        "clips_with_accident_window": n_acc,
        "class_instance_totals": dict(class_totals),
        "resolutions": dict(res_set),
        "mean_tracks_per_clip": round(sum(p["num_object_tracks"] for p in per) / len(per), 2),
        "mean_accident_len_s": round(
            sum(p["accident_len_s"] for p in per if p["has_accident_window"]) / max(1, n_acc), 2),
        "per_clip": per,
    }


# ==========================================================================
# EVALUATION METRICS  (same for both datasets)
# ==========================================================================

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def average_precision(preds, gts, iou_thresh=0.5):
    """
    Detection AP for a single class on one dataset.
    preds: list of (clip_id, frame_id, bbox, score)
    gts:   list of (clip_id, frame_id, bbox)
    Standard VOC-style greedy matching, one GT per prediction.
    """
    gt_index = defaultdict(list)   # (clip,frame) -> list of [bbox, matched]
    for cid, fid, box in gts:
        gt_index[(cid, fid)].append([box, False])
    n_gt = len(gts)
    if n_gt == 0:
        return 0.0

    preds_sorted = sorted(preds, key=lambda p: p[3], reverse=True)
    tp = [0] * len(preds_sorted)
    fp = [0] * len(preds_sorted)

    for i, (cid, fid, box, score) in enumerate(preds_sorted):
        cands = gt_index.get((cid, fid), [])
        best_iou, best_j = 0.0, -1
        for j, (gbox, matched) in enumerate(cands):
            if matched:
                continue
            v = iou(box, gbox)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_thresh and best_j >= 0:
            cands[best_j][1] = True
            tp[i] = 1
        else:
            fp[i] = 1

    # precision-recall
    ctp = cfp = 0
    precisions, recalls = [], []
    for i in range(len(preds_sorted)):
        ctp += tp[i]; cfp += fp[i]
        precisions.append(ctp / (ctp + cfp))
        recalls.append(ctp / n_gt)

    # 11-point interpolated AP (VOC2007-style; robust and simple to report)
    ap = 0.0
    for r in [x / 10 for x in range(11)]:
        p = max([precisions[i] for i in range(len(recalls)) if recalls[i] >= r], default=0.0)
        ap += p / 11
    return ap


def precision_recall_curve(preds, gts, iou_thresh=0.5):
    """Same greedy VOC matching as average_precision, but returns the raw
    (precision, recall) sequence so that a PR curve can be plotted."""
    gt_index = defaultdict(list)
    for cid, fid, box in gts:
        gt_index[(cid, fid)].append([box, False])
    n_gt = len(gts)
    if n_gt == 0:
        return [], []
    preds_sorted = sorted(preds, key=lambda p: p[3], reverse=True)
    ctp = cfp = 0
    precisions, recalls = [], []
    for cid, fid, box, score in preds_sorted:
        cands = gt_index.get((cid, fid), [])
        best_iou, best_j = 0.0, -1
        for j, (gbox, matched) in enumerate(cands):
            if matched:
                continue
            v = iou(box, gbox)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_thresh and best_j >= 0:
            cands[best_j][1] = True
            ctp += 1
        else:
            cfp += 1
        precisions.append(ctp / (ctp + cfp))
        recalls.append(ctp / n_gt)
    return precisions, recalls


def mean_average_precision(preds_by_class, gts_by_class, iou_thresh=0.5):
    """preds_by_class/gts_by_class: {class_name: [...]}. Returns (mAP, per-class dict)."""
    per = {}
    for cls in gts_by_class:
        per[cls] = average_precision(preds_by_class.get(cls, []),
                                     gts_by_class[cls], iou_thresh)
    mAP = sum(per.values()) / len(per) if per else 0.0
    return mAP, per


def temporal_accident_metrics(pred_window, gt_window, num_frames, fps=30.0):
    """
    Evaluate accident-WINDOW detection (a thesis-relevant, timeliness-aware metric).
    pred_window: (start_frame, end_frame) inclusive, or None.
    gt_window:   one window, a list of windows (a video may contain several accident
                 segments, as in the stage-1 CADP annotations), or None.
    Returns temporal IoU against the best-overlapping ground-truth window, the detection
    latency (s) = pred_start - gt_start of that window (negative = early), the status
    hit / miss / false_alarm / true_negative and the number of ground-truth windows.
    """
    def to_set(w):
        return set(range(int(w[0]), int(w[1]) + 1)) if w else set()
    if gt_window is None:
        gts = []
    elif isinstance(gt_window, (list, tuple)) and gt_window and isinstance(gt_window[0], (list, tuple)):
        gts = [tuple(w) for w in gt_window]
    else:
        gts = [tuple(gt_window)]
    p = to_set(pred_window)
    if not gts and not p:
        return {"t_iou": 1.0, "latency_s": 0.0, "status": "true_negative", "n_gt": 0}
    if gts and not p:
        return {"t_iou": 0.0, "latency_s": None, "status": "miss", "n_gt": len(gts)}
    if p and not gts:
        return {"t_iou": 0.0, "latency_s": None, "status": "false_alarm", "n_gt": 0}
    best, best_iou = None, -1.0
    for g in gts:
        gs = to_set(g)
        inter = len(gs & p); union = len(gs | p)
        t_iou = inter / union if union else 0.0
        if inter > 0 and t_iou > best_iou:
            best, best_iou = g, t_iou
    if best is None:
        return {"t_iou": 0.0, "latency_s": None, "status": "false_alarm", "n_gt": len(gts)}
    latency = (pred_window[0] - best[0]) / fps
    return {"t_iou": round(best_iou, 3), "latency_s": round(latency, 3), "status": "hit",
            "n_gt": len(gts), "matched_gt": list(best)}


# ==========================================================================
# COMPARISON  (put two dataset_stats side by side)
# ==========================================================================

def compare_datasets(stats_a, stats_b):
    """Return a dict aligning the two datasets' aggregate stats for reporting."""
    keys = ["dataset", "num_clips", "total_frames", "total_duration_min",
            "clips_with_accident_window", "mean_tracks_per_clip",
            "mean_accident_len_s"]
    rows = []
    for k in keys:
        rows.append({"metric": k, "A": stats_a.get(k), "B": stats_b.get(k)})
    # class coverage union
    classes = sorted(set(stats_a.get("class_instance_totals", {})) |
                     set(stats_b.get("class_instance_totals", {})))
    class_rows = [{"class": c,
                   "A": stats_a.get("class_instance_totals", {}).get(c, 0),
                   "B": stats_b.get("class_instance_totals", {}).get(c, 0)}
                  for c in classes]
    return {"summary_rows": rows, "class_rows": class_rows}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        clip = load_cadp_vatic(sys.argv[1])
        print(json.dumps(clip_stats(clip), indent=2))
