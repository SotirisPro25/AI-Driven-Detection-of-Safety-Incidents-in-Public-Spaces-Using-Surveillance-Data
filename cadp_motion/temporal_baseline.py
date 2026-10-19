#!/usr/bin/env python3
"""
temporal_baseline.py -- training-free accident-window detection from motion cues on CADP,
evaluated against the stage-1 accident segments (Section 6.4.4, Table 6.9).

    python temporal_baseline.py --features CADP/motion/full.npz --windows CADP/accident_windows_full.json \
           --split CADP/split.json --out results/temporal_baseline.json [--fps 10]

Method (no learning): one motion signal per frame (frame difference, mean flow magnitude,
90th-percentile flow magnitude or flow divergence) is median-smoothed over k frames and
standardized per video (robust z-score: median and median absolute deviation). Frames whose
z-score exceeds a threshold form active runs, and runs shorter than min_run frames are
discarded. Each remaining run is a predicted accident window.

Calibration: the signal, k, the threshold and min_run are chosen on the VALIDATION videos by
maximizing the window-level F1 (precision of predicted windows against recall of ground-truth
segments). The chosen setting is applied once to the TEST videos. Two readings of the result
are reported: the thesis rule of Section 6.3.3 (longest active run, one window per video,
hit rate, median temporal IoU, median signed latency, false alarms per hour) and the
window-level reading (precision, recall, F1, false alarms per hour of non-incident footage).
"""
import os, json, argparse, itertools
import numpy as np

SIGNALS = {"diff": 0, "flow_mean": 1, "flow_p90": 2, "flow_div": 3}


def smooth(x, k):
    if k <= 1:
        return x
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(xp, k), axis=1)


def robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    return (x - med) / (1.4826 * mad)


def runs(active, min_run):
    a = np.concatenate(([0], active.astype(np.int8), [0]))
    d = np.diff(a)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s + 1 >= min_run]


def overlaps(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


_CACHE = {}


def zsignal(series, key, sig, k):
    ck = (key, sig, k)
    if ck not in _CACHE:
        _CACHE[ck] = robust_z(smooth(series[:, SIGNALS[sig]], k))
    return _CACHE[ck]


def predict(series, sig, k, z, min_run, key=None):
    zs = zsignal(series, key if key is not None else id(series), sig, k)
    return runs(zs > z, min_run)


def window_eval(preds_by_vid, gt_by_vid, n_frames_by_vid, fps):
    """Window-level precision / recall and the thesis rule reading."""
    tp = fp = 0
    gt_total = gt_hit = 0
    non_incident_frames = 0
    longest = {"n_gt_videos": 0, "hits": 0, "misses": 0, "t_iou": [], "latency": [], "false_alarm": 0}
    for vid, preds in preds_by_vid.items():
        gts = [tuple(w) for w in gt_by_vid.get(vid, [])]
        covered = set()
        for g in gts:
            covered |= set(range(g[0], g[1] + 1))
        non_incident_frames += n_frames_by_vid[vid] - len(covered)
        for p in preds:
            if any(overlaps(p, g) for g in gts):
                tp += 1
            else:
                fp += 1
        for g in gts:
            gt_total += 1
            if any(overlaps(p, g) for p in preds):
                gt_hit += 1
        # thesis rule: longest run only
        if preds:
            L = max(preds, key=lambda w: w[1] - w[0])
        else:
            L = None
        if gts:
            longest["n_gt_videos"] += 1
            best = None
            if L is not None:
                for g in gts:
                    if overlaps(L, g):
                        inter = min(L[1], g[1]) - max(L[0], g[0]) + 1
                        union = max(L[1], g[1]) - min(L[0], g[0]) + 1
                        if best is None or inter / union > best[0]:
                            best = (inter / union, g)
            if best:
                longest["hits"] += 1
                longest["t_iou"].append(best[0])
                longest["latency"].append((L[0] - best[1][0]) / fps)
            else:
                longest["misses"] += 1
                if L is not None:
                    longest["false_alarm"] += 1
        elif L is not None:
            longest["false_alarm"] += 1
    hours = non_incident_frames / fps / 3600.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = gt_hit / gt_total if gt_total else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    lat = longest["latency"]
    return {
        "window_level": {"predicted_windows": tp + fp, "true_positive_windows": tp, "false_positive_windows": fp,
                         "gt_windows": gt_total, "gt_windows_detected": gt_hit,
                         "precision": prec, "recall": rec, "f1": f1,
                         "non_incident_hours": hours, "false_alarms_per_hour": fp / hours if hours else None},
        "thesis_rule": {"n_videos_with_gt_window": longest["n_gt_videos"], "hits": longest["hits"],
                        "misses": longest["misses"],
                        "hit_rate": longest["hits"] / longest["n_gt_videos"] if longest["n_gt_videos"] else None,
                        "median_t_iou": float(np.median(longest["t_iou"])) if longest["t_iou"] else None,
                        "median_latency_s": float(np.median(lat)) if lat else None,
                        "latency_iqr_s": [float(np.percentile(lat, 25)), float(np.percentile(lat, 75))] if lat else None,
                        "false_alarm_windows": longest["false_alarm"],
                        "false_alarms_per_hour": longest["false_alarm"] / hours if hours else None},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--windows", required=True, help="{vid: [[s, e], ...]} in the frame index space of --features")
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=10.0)
    a = ap.parse_args()
    F = np.load(a.features)
    series = {k: F[k] for k in F.files}
    gt = json.load(open(a.windows))
    split = json.load(open(a.split))
    n_frames = {k: len(v) for k, v in series.items()}
    grid = list(itertools.product(SIGNALS.keys(), (1, 5, 9), (1.0, 1.5, 2.0, 2.5, 3.0, 4.0), (3, 5, 10, 20)))
    val_vids = [v for v in split["val"] if v in series]
    test_vids = [v for v in split["test"] if v in series]
    best, best_cfg, table = -1.0, None, []
    for sig, k, z, mr in grid:
        preds = {v: predict(series[v], sig, k, z, mr, key=v) for v in val_vids}
        r = window_eval(preds, gt, n_frames, a.fps)
        f1 = r["window_level"]["f1"]
        table.append({"signal": sig, "k": k, "z": z, "min_run": mr, "val_f1": f1,
                      "val_precision": r["window_level"]["precision"], "val_recall": r["window_level"]["recall"]})
        if f1 > best:
            best, best_cfg = f1, (sig, k, z, mr)
    sig, k, z, mr = best_cfg
    preds_test = {v: predict(series[v], sig, k, z, mr, key=v) for v in test_vids}
    res_test = window_eval(preds_test, gt, n_frames, a.fps)
    preds_val = {v: predict(series[v], sig, k, z, mr, key=v) for v in val_vids}
    res_val = window_eval(preds_val, gt, n_frames, a.fps)
    out = {"kind": "temporal_baseline", "method": "robust z-score of a smoothed motion signal, active runs",
           "calibrated_on": "val", "selected": {"signal": sig, "smoothing_frames": k, "z_threshold": z, "min_run_frames": mr},
           "val": res_val, "test": res_test, "fps": a.fps,
           "n_val_videos": len(val_vids), "n_test_videos": len(test_vids),
           "test_predictions": {v: [list(map(int, w)) for w in preds_test[v]] for v in test_vids},
           "grid": sorted(table, key=lambda r: -r["val_f1"])[:10]}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("selected:", out["selected"])
    print("val :", json.dumps(res_val["window_level"]), "\n     ", json.dumps(res_val["thesis_rule"]))
    print("test:", json.dumps(res_test["window_level"]), "\n     ", json.dumps(res_test["thesis_rule"]))


if __name__ == "__main__":
    main()
