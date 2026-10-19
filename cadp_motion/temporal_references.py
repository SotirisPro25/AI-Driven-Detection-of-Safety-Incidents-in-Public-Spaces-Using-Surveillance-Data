#!/usr/bin/env python3
"""temporal_references.py -- the always-on and random reference predictors for Table 6.9.
    python temporal_references.py --features CADP/motion/full.npz --windows CADP/accident_windows_full.json \
           --split CADP/split.json --out CADP/results/references.json"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from temporal_baseline import window_eval

ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True); ap.add_argument("--windows", required=True)
ap.add_argument("--split", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=float, default=10.0); ap.add_argument("--draws", type=int, default=20)
a = ap.parse_args()
F = np.load(a.features); gt = json.load(open(a.windows)); split = json.load(open(a.split))
n = {k: len(F[k]) for k in F.files}
test = [v for v in split["test"] if v in n]
always = window_eval({v: [(0, n[v] - 1)] for v in test}, gt, n, a.fps)
rng = np.random.default_rng(0); draws = []
for d in range(a.draws):
    preds = {}
    for v in test:
        L = n[v]; k = max(1, int(round(L / 300)))
        starts = rng.integers(0, max(1, L - 30), size=k)
        preds[v] = [(int(s), int(s + 29)) for s in starts]
    draws.append(window_eval(preds, gt, n, a.fps))
rand = {"precision": float(np.mean([x["window_level"]["precision"] for x in draws])),
        "recall": float(np.mean([x["window_level"]["recall"] for x in draws])),
        "false_alarms_per_hour": float(np.mean([x["window_level"]["false_alarms_per_hour"] for x in draws])),
        "thesis_rule_hit_rate": float(np.mean([x["thesis_rule"]["hit_rate"] for x in draws])), "draws": a.draws}
acc = sum(sum(e - s + 1 for s, e in gt.get(v, [])) for v in test); tot = sum(n[v] for v in test)
out = {"always_on": always, "random": rand, "accident_share_test": acc / tot, "n_test_videos": len(test)}
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({"always_on": always["thesis_rule"], "random": rand, "accident_share_test": acc / tot}, indent=1))
