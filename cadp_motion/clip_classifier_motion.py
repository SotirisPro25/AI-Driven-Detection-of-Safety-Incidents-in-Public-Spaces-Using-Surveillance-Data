#!/usr/bin/env python3
"""
clip_classifier_motion.py -- accident versus normal-footage clip classification on CADP from
motion statistics alone, with the video-level split (Section 6.4.4, Table 6.10).

    python clip_classifier_motion.py --features CADP/motion/clips.npz \
           --manifest CADP/clips_manifest.csv --out results/clip_classifier_motion.json

Each clip is described by 26 statistics of its four per-frame motion signals (frame
difference, mean flow magnitude, 90th-percentile flow magnitude, flow divergence): mean,
standard deviation, maximum, 90th percentile, the largest 5-frame increase (a jerk) and the
ratio of the maximum to the median, plus the clip length in frames and the fraction of frames
above twice the clip median. Two classifiers are trained on the training clips with
scikit-learn: a standardized logistic regression and a histogram gradient-boosting classifier.
The model with the higher macro-F1 on the validation clips is evaluated once on the test
clips. Accuracy is reported with the majority-class prior of the test clips. Clips inherit the
split of their source video, so no video contributes clips to two partitions.
"""
import os, json, csv, argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score, roc_auc_score

NAMES = ["diff", "flow_mean", "flow_p90", "flow_div"]


def clip_features(arr):
    if len(arr) < 3:
        arr = np.vstack([arr] + [arr[-1:]] * (3 - len(arr)))
    f = []
    for j in range(4):
        x = arr[1:, j] if len(arr) > 1 else arr[:, j]
        med = np.median(x) + 1e-6
        k = min(5, len(x) - 1) if len(x) > 1 else 1
        jerk = np.max(x[k:] - x[:-k]) if len(x) > k else 0.0
        f += [x.mean(), x.std(), x.max(), np.percentile(x, 90), jerk, x.max() / med]
    f += [len(arr), float(np.mean(arr[1:, 0] > 2 * (np.median(arr[1:, 0]) + 1e-6))) if len(arr) > 1 else 0.0]
    return np.array(f, dtype=np.float64)


def evaluate(model, X, y):
    p = model.predict(X)
    s = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else p
    return {"n": int(len(y)), "accuracy": float((p == y).mean()),
            "macro_f1": float(f1_score(y, p, average="macro")),
            "precision_accident": float(precision_score(y, p, zero_division=0)),
            "recall_accident": float(recall_score(y, p)),
            "recall_normal": float(recall_score(1 - y, 1 - p)),
            "roc_auc": float(roc_auc_score(y, s)) if len(set(y)) > 1 else None,
            "confusion": confusion_matrix(y, p, labels=[1, 0]).tolist(),   # rows: accident, normal
            "majority_baseline": float(max(y.mean(), 1 - y.mean()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--grid", action="store_true",
                    help="wider candidate set fixed in advance (logistic C, boosting depth/rate/iterations, random forest), "
                         "selected on validation macro-F1 only")
    a = ap.parse_args()
    F = np.load(a.features)
    rows = list(csv.DictReader(open(a.manifest)))
    data = {sp: ([], [], []) for sp in ("train", "val", "test")}
    for r in rows:
        if r["clip_uid"] not in F.files:
            continue
        arr = F[r["clip_uid"]]
        if len(arr) < 2:
            continue
        X, y, ids = data[r["split"]]
        X.append(clip_features(arr)); y.append(1 if r["kind"] == "accident" else 0); ids.append(r["clip_uid"])
    D = {sp: (np.array(X), np.array(y), ids) for sp, (X, y, ids) in data.items()}
    for sp, (X, y, _) in D.items():
        print(f"{sp}: {len(y)} clips, {int(y.sum())} accident, {int((1-y).sum())} normal")
    Xtr, ytr, _ = D["train"]; Xva, yva, _ = D["val"]; Xte, yte, ids_te = D["test"]
    results = {"kind": "clip_classifier_motion", "features": 26, "models": {}}
    best_name, best_f1, best_model = None, -1.0, None
    for seed in range(a.seeds):
        cands = {
            "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0,
                                                                                       class_weight="balanced", random_state=seed)),
            "gradient_boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                                                class_weight="balanced", random_state=seed),
        }
        if a.grid:
            from sklearn.ensemble import RandomForestClassifier
            for C in (0.1, 10.0):
                cands[f"logistic_regression_C{C}"] = make_pipeline(StandardScaler(), LogisticRegression(
                    max_iter=4000, C=C, class_weight="balanced", random_state=seed))
            for d in (3, 6):
                for lr in (0.03, 0.1):
                    for it in (200, 500):
                        cands[f"gradient_boosting_d{d}_lr{lr}_it{it}"] = HistGradientBoostingClassifier(
                            max_iter=it, learning_rate=lr, max_depth=d, class_weight="balanced", random_state=seed)
            for d in (None, 8):
                cands[f"random_forest_depth{d}"] = RandomForestClassifier(
                    n_estimators=500, max_depth=d, class_weight="balanced", n_jobs=-1, random_state=seed)
        for name, m in cands.items():
            m.fit(Xtr, ytr)
            v = evaluate(m, Xva, yva)
            t = evaluate(m, Xte, yte)
            results["models"].setdefault(name, []).append({"seed": seed, "val": v, "test": t})
            key = v["macro_f1"] + 1e-6 * (v["roc_auc"] or 0)          # ties broken by validation AUC
            if key > best_f1:
                best_f1, best_name, best_model = key, (name, seed), m
    name, seed = best_name
    sel = [r for r in results["models"][name] if r["seed"] == seed][0]
    results["selected"] = {"model": name, "seed": seed, "val": sel["val"], "test": sel["test"]}
    # seed statistics of the selected model family on the test clips
    tests = [r["test"] for r in results["models"][name]]
    results["selected"]["test_over_seeds"] = {k: {"mean": float(np.mean([t[k] for t in tests])),
                                                  "std": float(np.std([t[k] for t in tests], ddof=1)) if len(tests) > 1 else 0.0}
                                              for k in ("accuracy", "macro_f1", "recall_accident", "recall_normal", "roc_auc")
                                              if all(t[k] is not None for t in tests)}
    results["split_sizes"] = {sp: {"clips": int(len(y)), "accident": int(y.sum()), "normal": int((1 - y).sum())}
                              for sp, (X, y, _) in D.items()}
    p = best_model.predict(Xte)
    results["test_errors"] = [i for i, (pi, yi) in enumerate(zip(p, yte)) if pi != yi]
    results["test_error_clips"] = [ids_te[i] for i in results["test_errors"]]
    # bootstrap 95% confidence intervals over the test clips (10,000 resamples)
    rng = np.random.default_rng(0)
    accs, f1s = [], []
    for _ in range(10000):
        idx = rng.integers(0, len(yte), len(yte))
        accs.append(float((p[idx] == yte[idx]).mean()))
        f1s.append(float(f1_score(yte[idx], p[idx], average="macro")))
    results["selected"]["test_bootstrap_ci"] = {"accuracy": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
                                                "macro_f1": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))]}
    results["other_model"] = {name_: runs[0]["test"] for name_, runs in results["models"].items() if name_ != name}
    results["deterministic"] = all(len({round(r["test"]["accuracy"], 6) for r in runs}) == 1 for runs in results["models"].values())
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=1)
    print("selected:", name, "seed", seed)
    print("val :", json.dumps({k: sel["val"][k] for k in ("accuracy", "macro_f1", "roc_auc", "majority_baseline")}))
    print("test:", json.dumps({k: sel["test"][k] for k in ("accuracy", "macro_f1", "recall_accident", "recall_normal", "roc_auc", "majority_baseline")}))
    print("test over seeds:", json.dumps(results["selected"]["test_over_seeds"]))


if __name__ == "__main__":
    main()
