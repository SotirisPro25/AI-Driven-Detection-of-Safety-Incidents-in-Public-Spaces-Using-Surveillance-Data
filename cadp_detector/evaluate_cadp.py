#!/usr/bin/env python3
"""
evaluate_cadp.py -- TESTING & VALIDATION for the trained CADP detector.

Runs the trained model over the 'test' (or 'val') split and reports:
  1. Detection mAP@0.5 with per-class AP (vehicle / two_wheeler / person),
     overall, per video and, if a conditions file is given, per condition
     (camera type and illumination);
  2. Precision-recall curves per class;
  3. Temporal accident metrics per video (temporal IoU, signed latency,
     hit / miss / false alarm) from the Colliding ground-truth windows and
     false alarms per hour of non-incident footage;
  4. Inference time per frame.

The thesis did not train the detector to the point of reporting 1 to 3. Its inference path
(detect_frame) is the one timed by bench_detector.py for Section 6.4.1 (Tables 6.6 and 6.8).

Everything is written to ONE results file
    <out>/detector_n<n>_seed<seed>_<split>.json

USAGE
  python evaluate_cadp.py --cadp_root /path/to/CADP --split test \
         --weights checkpoints_n3_s0 --n_contexts 3 --seed 0 \
         --conditions conditions.csv --out results

conditions.csv (optional, one line per video):
    vid,camera_type,illumination
    001000,fixed,day
    001001,dashcam,night
camera_type in {fixed, dashcam, broadcast}; illumination in {day, night}.

Predictions of accident windows are derived from detections via a simple,
transparent rule (documented in the thesis): a frame is 'incident-active' when
>=2 detected objects of score>=0.7 overlap with IoU>=0.3 (proxy for collision
proximity); the predicted window is the longest run of active frames. This
rule is deliberately simple -- the contribution under test is the detector; the
window rule is held fixed across all model variants so comparisons are fair.
"""
import os, json, argparse, csv, time, platform, math
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accident_dataset_toolkit import (mean_average_precision, precision_recall_curve,
                                      temporal_accident_metrics, iou)
from train_cadp_tf2 import (build_models, load_split_rows, resize_keep_ratio,
                            rpn_to_rois, nms, Config, C, NUM_CLASSES)

INV_CLASS = {v: k for k, v in C.class_mapping.items()}
EVAL_FPS = 10.0          # frame rate of the working set (Section 6.2.1)


def apply_regr(x, y, w, h, tx, ty, tw, th):
    """roi_helpers.apply_regr of the release (feature-map units, rounded)."""
    try:
        cx, cy = x + w / 2.0, y + h / 2.0
        cx1, cy1 = tx * w + cx, ty * h + cy
        w1, h1 = math.exp(tw) * w, math.exp(th) * h
        return int(round(cx1 - w1 / 2.0)), int(round(cy1 - h1 / 2.0)), int(round(w1)), int(round(h1))
    except (ValueError, OverflowError):
        return x, y, w, h


def detect_frame(model_rpn, model_cls, img, score_thresh=0.0, nms_thresh=0.5):
    """Two-stage inference on one image, as measure_map.py of the release runs it:
    300 proposals after suppression at 0.7, head in batches of num_rois,
    class = argmax of the softmax (background skipped), class-specific regression
    divided by classifier_regr_std and applied with apply_regr, boxes scaled by
    the stride, greedy suppression at 0.5 per class. No score threshold for the
    precision-recall curve. Returns [(box, label, score)], resized-image coordinates."""
    import tensorflow as tf
    x = img[None]
    p_cls, p_reg, fmap = model_rpn(x, training=False)
    rois, _ = rpn_to_rois(p_cls.numpy()[0], p_reg.numpy()[0],
                          fmap.shape[1], fmap.shape[2], top_n=300)
    by_class = {}
    for i in range(0, len(rois), C.num_rois):
        batch = rois[i:i+C.num_rois]
        n_real = len(batch)
        if n_real < C.num_rois:
            batch = np.concatenate(
                [batch, np.repeat(batch[:1], C.num_rois-n_real, 0)])
        pc, pr = model_cls([fmap, tf.convert_to_tensor(batch[None], tf.float32)],
                           training=False)
        pc = pc.numpy()[0]; pr = pr.numpy()[0]
        for j in range(n_real):
            cls = int(pc[j].argmax()); score = float(pc[j].max())
            if INV_CLASS[cls] == "bg" or score < score_thresh:
                continue
            x_, y_, w_, h_ = batch[j]
            tx, ty, tw, th = pr[j, 4*cls:4*(cls+1)]
            sx, sy, sw, sh = C.classifier_regr_std
            x_, y_, w_, h_ = apply_regr(x_, y_, w_, h_, tx/sx, ty/sy, tw/sw, th/sh)
            st = C.rpn_stride
            by_class.setdefault(INV_CLASS[cls], []).append(
                [st*x_, st*y_, st*(x_+w_), st*(y_+h_), score])
    dets = []
    for label, lst in by_class.items():
        arr = np.array(lst, dtype=np.float64)
        keep = nms(arr[:, :4], arr[:, 4], nms_thresh, 300)
        for k in keep:
            dets.append((arr[k, :4].tolist(), label, float(arr[k, 4])))
    return dets


# ------------------------------------------------------------------------------------------
# The measurement protocol of the release (measure_map.py): per image, predictions sorted by
# score are matched greedily to unmatched ground truth of the same class at IoU >= 0.5, every
# unmatched ground-truth box is appended as a zero-score positive, scikit-learn's step-wise
# average precision is computed on the lists ACCUMULATED over the images seen so far, and the
# reported mAP is the running mean of those cumulative APs over images and classes. It is
# reproduced here, in the order of the CSV rows, so that the paper's figures can be compared
# under their own definition next to the VOC-style AP.
# ------------------------------------------------------------------------------------------
def sklearn_average_precision(T, P):
    """sklearn.metrics.average_precision_score for binary labels T and scores P."""
    T = np.asarray(T, dtype=np.float64); P = np.asarray(P, dtype=np.float64)
    if T.sum() == 0:
        return 0.0
    order = np.argsort(-P, kind="mergesort")
    T, P = T[order], P[order]
    distinct = np.where(np.diff(P) != 0)[0]
    idx = np.r_[distinct, T.size - 1]
    tps = np.cumsum(T)[idx]
    fps = 1 + idx - tps
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    # sklearn reverses both and appends (precision 1, recall 0)
    precision = np.r_[precision[::-1], 1.0]
    recall = np.r_[recall[::-1], 0.0]
    return float(-np.sum(np.diff(recall) * precision[:-1]))


class ReleaseProtocol:
    def __init__(self):
        self.T, self.P = {}, {}
        self.map_sum, self.count = 0.0, 0
        self.by_class = {}

    def add_image(self, dets, gt_boxes):
        """dets: [(box, label, score)], gt_boxes: [{'x1','y1','x2','y2','label'}] (same coordinates)."""
        gts = [dict(g, matched=False) for g in gt_boxes]
        T, P = {}, {}
        for box, label, score in sorted(dets, key=lambda d: -d[2]):
            P.setdefault(label, []); T.setdefault(label, [])
            P[label].append(score)
            found = False
            for g in gts:
                if g["label"] != label or g["matched"]:
                    continue
                if iou(box, [g["x1"], g["y1"], g["x2"], g["y2"]]) >= 0.5:
                    found = True; g["matched"] = True; break
            T[label].append(int(found))
        for g in gts:
            if not g["matched"]:
                T.setdefault(g["label"], []).append(1); P.setdefault(g["label"], []).append(0.0)
        for k in T:
            self.T.setdefault(k, []).extend(T[k]); self.P.setdefault(k, []).extend(P[k])
        aps = []
        for k in self.T:
            ap = sklearn_average_precision(self.T[k], self.P[k])
            self.by_class.setdefault(k, []).append(ap); aps.append(ap)
        self.map_sum += sum(aps); self.count += len(aps)

    def result(self):
        final = {k: sklearn_average_precision(self.T[k], self.P[k]) for k in self.T}
        return {"map_running_mean": self.map_sum / self.count if self.count else None,
                "per_class_running_mean": {k: float(np.mean(v)) for k, v in self.by_class.items()},
                "final_cumulative_ap": final,
                "map_final_cumulative": float(np.mean(list(final.values()))) if final else None,
                "definition": "measure_map.py of the CADP release: greedy matching at IoU 0.5, unmatched "
                              "ground truth appended as zero-score positives, scikit-learn step-wise AP on "
                              "the cumulative lists, running mean over images and classes"}


def predicted_accident_window(frame_dets, min_score=0.7, iou_t=0.3):
    """Longest run of frames where >=2 confident detections overlap."""
    active = []
    for fid in sorted(frame_dets):
        ds = [(b, s) for (b, l, s) in frame_dets[fid] if s >= min_score]
        hit = any(iou(a[0], b[0]) >= iou_t
                  for i, a in enumerate(ds) for b in ds[i+1:])
        active.append((fid, hit))
    best, cur = None, None
    for fid, hit in active:
        if hit:
            cur = (cur[0], fid) if cur else (fid, fid)
            if best is None or cur[1]-cur[0] > best[1]-best[0]:
                best = cur
        else:
            cur = None
    return best


def subset_map(preds, gts, vids):
    """mAP and per-class AP restricted to a set of video ids."""
    p = {c: [x for x in v if x[0] in vids] for c, v in preds.items()}
    g = {c: [x for x in v if x[0] in vids] for c, v in gts.items()}
    g = {c: v for c, v in g.items() if v}
    if not g:
        return None, {}
    return mean_average_precision(p, g, iou_thresh=0.5)


def load_conditions(path):
    cond = {}
    if path and os.path.exists(path):
        with open(path) as f:
            for row in csv.DictReader(f):
                cond[row["vid"].strip()] = {"camera_type": row.get("camera_type", "").strip().lower(),
                                            "illumination": row.get("illumination", "").strip().lower()}
    return cond


def hardware_string():
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            d = tf.config.experimental.get_device_details(gpus[0])
            return d.get("device_name", "GPU")
    except Exception:
        pass
    return platform.processor() or platform.machine() or "CPU"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadp_root", required=True)
    ap.add_argument("--csv", default="csv/cadp_all.csv")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--weights", default="checkpoints")
    ap.add_argument("--n_contexts", type=int, default=3)
    ap.add_argument("--step_stride", type=int, default=2, help="as used in training")
    ap.add_argument("--seed", type=int, default=0, help="seed tag of the trained model")
    ap.add_argument("--conditions", default="", help="csv: vid,camera_type,illumination")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    import cv2
    import tensorflow as tf
    rows = load_split_rows(args.cadp_root, args.csv, args.split)
    model_rpn, model_cls = build_models(args.n_contexts, args.step_stride)
    model_rpn.load_weights(os.path.join(args.weights, "rpn_best.weights.h5"))
    model_cls.load_weights(os.path.join(args.weights, "head_best.weights.h5"))

    gts, preds = {}, {}
    frame_dets_by_vid = {}
    frames_by_vid = {}
    times_ms = []
    release = ReleaseProtocol()
    for path, boxes in rows.items():
        img = cv2.imread(path)
        if img is None:
            continue
        b = np.array([[x["x1"], x["y1"], x["x2"], x["y2"]] for x in boxes])
        img_r, b_r, scale = resize_keep_ratio(img.astype("float32"), b, C.im_size)
        img_r -= np.array([103.939, 116.779, 123.68])
        vid = os.path.basename(os.path.dirname(path))
        fid = int(os.path.basename(path).split(".")[0])
        frames_by_vid[vid] = frames_by_vid.get(vid, 0) + 1
        for x, bb in zip(boxes, b_r):
            gts.setdefault(x["label"], []).append((vid, fid, list(bb)))
        t0 = time.perf_counter()
        dets = detect_frame(model_rpn, model_cls, img_r)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        frame_dets_by_vid.setdefault(vid, {})[fid] = dets
        for box, label, score in dets:
            preds.setdefault(label, []).append((vid, fid, box, score))
        release.add_image(dets, [{"x1": bb[0], "y1": bb[1], "x2": bb[2], "y2": bb[3], "label": x["label"]}
                                 for x, bb in zip(boxes, b_r)])

    # ---- 1. detection, overall ----
    mAP, per = mean_average_precision(preds, gts, iou_thresh=0.5)
    print(f"\n=== {args.split} split: detection ===")
    for c, v in per.items():
        print(f"  AP@0.5 {c:12s}: {v:.3f}")
    print(f"  mAP@0.5          : {mAP:.3f}")
    rel = release.result()
    print(f"  release protocol (measure_map.py): running-mean mAP {rel['map_running_mean']:.3f}, "
          f"final cumulative {rel['map_final_cumulative']:.3f}")
    n_frames = sum(frames_by_vid.values())

    # ---- per video (for the paired Wilcoxon test over videos) ----
    per_video = {}
    for vid in frames_by_vid:
        m, p = subset_map(preds, gts, {vid})
        if m is not None:
            per_video[vid] = {"map": m, "per_class_ap": p, "n_frames": frames_by_vid[vid]}

    # ---- per condition ----
    cond = load_conditions(args.conditions)
    by_condition = {}
    if cond:
        groups = {"fixed": set(), "dashcam": set(), "broadcast": set(), "day": set(), "night": set()}
        for vid, c in cond.items():
            if vid in frames_by_vid:
                if c["camera_type"] in groups:
                    groups[c["camera_type"]].add(vid)
                if c["illumination"] in groups:
                    groups[c["illumination"]].add(vid)
        for name, vids in groups.items():
            if not vids:
                continue
            m, p = subset_map(preds, gts, vids)
            if m is not None:
                by_condition[name] = {"map": m, "per_class_ap": p,
                                      "n_frames": sum(frames_by_vid[v] for v in vids),
                                      "n_videos": len(vids)}

    # ---- 2. PR curves ----
    pr_curves = {}
    for c in gts:
        pr, rc = precision_recall_curve(preds.get(c, []), gts[c], iou_thresh=0.5)
        step = max(1, len(pr) // 400)          # thin very long curves
        pr_curves[c] = {"precision": pr[::step], "recall": rc[::step]}

    # ---- 3. accident-window timeliness ----
    temporal = {"available": False}
    win_file = os.path.join(args.cadp_root, "accident_windows.json")
    if os.path.exists(win_file):
        gt_windows = json.load(open(win_file))          # {vid: [[s, e], ...]} or {vid: [s, e]}
        print(f"\n=== {args.split} split: accident timeliness ===")
        per_video_temporal = {}
        hits, misses, false_alarms = [], [], []
        gt_lengths_s = []
        non_incident_frames = 0
        for vid, fd in frame_dets_by_vid.items():
            pw = predicted_accident_window(fd)
            gw = gt_windows.get(vid)
            if gw and not isinstance(gw[0], (list, tuple)):
                gw = [gw]
            gw = [tuple(w) for w in gw] if gw else None
            m = temporal_accident_metrics(pw, gw, len(fd), fps=EVAL_FPS)
            per_video_temporal[vid] = {"pred": pw, "gt": gw, **m, "n_frames": len(fd)}
            print(f"  {vid}: pred={pw} gt={gw} -> {m}")
            covered = set()
            if gw:
                for w in gw:
                    gt_lengths_s.append((w[1] - w[0] + 1) / EVAL_FPS)
                    covered |= set(range(int(w[0]), int(w[1]) + 1))
            non_incident_frames += len(fd) - len(covered & set(fd))
            if gw:
                if m["status"] == "hit":
                    hits.append(m)
                else:
                    misses.append(vid)
            if m["status"] == "false_alarm":
                false_alarms.append(vid)
        n_gt = sum(1 for v in per_video_temporal.values() if v["gt"])
        lat = [h["latency_s"] for h in hits]
        tiou = [h["t_iou"] for h in hits]
        non_incident_hours = non_incident_frames / EVAL_FPS / 3600.0
        temporal = {
            "available": True,
            "eval_fps": EVAL_FPS,
            "n_videos": len(frame_dets_by_vid),
            "n_videos_with_gt_window": n_gt,
            "n_gt_windows": sum(len(v["gt"]) for v in per_video_temporal.values() if v["gt"]),
            "hits": len(hits), "misses": len(misses), "missed_videos": misses,
            "hit_rate": (len(hits) / n_gt) if n_gt else None,
            "median_t_iou": float(np.median(tiou)) if tiou else None,
            "median_latency_s": float(np.median(lat)) if lat else None,
            "latency_iqr_s": [float(np.percentile(lat, 25)), float(np.percentile(lat, 75))] if lat else None,
            "median_gt_window_s": float(np.median(gt_lengths_s)) if gt_lengths_s else None,
            "false_alarm_videos": false_alarms,
            "non_incident_hours": non_incident_hours,
            "false_alarms_per_hour": (len(false_alarms) / non_incident_hours) if non_incident_hours > 0 else None,
            "per_video": per_video_temporal,
        }
        print(f"  window hit rate: {len(hits)}/{n_gt}")
    else:
        print("\n(no accident_windows.json -- run process_cadp_stage1.py or make_accident_windows.py)")

    # ---- 4. inference time ----
    warm = times_ms[10:] if len(times_ms) > 20 else times_ms
    latency = {"ms_per_frame_mean": float(np.mean(warm)) if warm else None,
               "ms_per_frame_median": float(np.median(warm)) if warm else None,
               "n_timed_frames": len(warm), "batch_size": 1, "excludes_decoding": True,
               "hardware": hardware_string(), "tensorflow": tf.__version__}

    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f"detector_n{args.n_contexts}_seed{args.seed}_{args.split}.json")
    json.dump({"kind": "detector", "split": args.split, "n_contexts": args.n_contexts, "step_stride": args.step_stride,
               "seed": args.seed, "n_test_frames": n_frames, "n_videos": len(frames_by_vid),
               "map": mAP, "per_class_ap": per, "per_video": per_video,
               "by_condition": by_condition, "pr_curves": pr_curves,
               "release_protocol": rel, "inference": {"proposals": 300, "rpn_nms": C.rpn_nms_thresh,
                                                      "class_nms": 0.5, "score_threshold": 0.0,
                                                      "regression_applied": True},
               "temporal": temporal, "latency": latency},
              open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
