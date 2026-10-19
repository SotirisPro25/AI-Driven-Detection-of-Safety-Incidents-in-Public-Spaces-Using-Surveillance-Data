#!/usr/bin/env python3
"""parse_kaggle_log.py -- rebuild the result files of kaggle_train.py from the RESULT_* lines of one or
more Kaggle session logs (copied from the log viewer, one log line per text line; timestamps and line
numbers in front of the message are ignored).

    python parse_kaggle_log.py OUT_DIR log1.txt [log2.txt ...]

RESULT_JSON <file> k/n <piece>   compact result JSON without the per-clip list
RESULT_PRED <file> k/n <piece>   predicted class index of every test clip, one digit per clip
RESULT_TRUE <split> k/n <piece>  true class index of every test clip of that split
RESULT_NAMES <split> k/n <piece> JSON list of the test clip names of that split
RESULT_NAMESZ <split> k/n <piece> the same list, zlib-compressed and base64-encoded
Each rebuilt file is checked: accuracy and confusion matrix recomputed from the digits must match the JSON."""
import os, re, sys, json
from collections import defaultdict

PAT = re.compile(r"(RESULT_(?:JSON|PRED|TRUE|NAMESZ|NAMES)) (\S+) (\d+)/(\d+) ?(.*)$")


def collect(paths):
    parts = defaultdict(dict)          # (tag, name) -> {k: piece}
    total = {}
    for p in paths:
        for line in open(p, encoding="utf-8", errors="replace"):
            m = PAT.search(line.rstrip("\n"))
            if not m:
                continue
            tag, name, k, n, piece = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), m.group(5)
            parts[(tag, name)][k] = piece.rstrip()
            total[(tag, name)] = n
    out = {}
    for key, d in parts.items():
        n = total[key]
        missing = [k for k in range(1, n + 1) if k not in d]
        if missing:
            print(f"warning: {key} misses pieces {missing}", file=sys.stderr)
            continue
        out[key] = "".join(d[k] for k in range(1, n + 1))
    return out


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out_dir, logs = sys.argv[1], sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    got = collect(logs)
    trues = {name: s for (tag, name), s in got.items() if tag == "RESULT_TRUE"}
    names = {name: json.loads(s) for (tag, name), s in got.items() if tag == "RESULT_NAMES"}
    import base64, zlib
    names.update({name: json.loads(zlib.decompress(base64.b64decode(s)).decode("utf8"))
                  for (tag, name), s in got.items() if tag == "RESULT_NAMESZ"})
    n_ok = 0
    for (tag, fname), s in sorted(got.items()):
        if tag != "RESULT_JSON":
            continue
        j = json.loads(s)
        pred = got.get(("RESULT_PRED", fname))
        split = j.get("split", "") + ("_raw" if j.get("raw") else "")
        true = trues.get(split)
        if pred and true and len(pred) == len(true):
            nm = names.get(split) or [""] * len(true)
            if len(nm) != len(true):
                print(f"warning: {fname}: {len(nm)} names for {len(true)} clips", file=sys.stderr)
                nm = [""] * len(true)
            j["test_predictions"] = [{"name": a, "true": int(t), "pred": int(p)} for a, t, p in zip(nm, true, pred)]
            # consistency with the stored metrics
            acc = sum(t == p for t, p in zip(true, pred)) / len(true)
            K = len(j["confusion"])
            conf = [[0] * K for _ in range(K)]
            for t, p in zip(true, pred):
                conf[int(t)][int(p)] += 1
            if abs(acc - j["accuracy"]) > 1e-9 or conf != j["confusion"]:
                print(f"warning: {fname}: per-clip digits disagree with the stored metrics", file=sys.stderr)
        else:
            print(f"note: {fname}: no per-clip predictions rebuilt (pred {bool(pred)}, true {bool(true)})", file=sys.stderr)
        json.dump(j, open(os.path.join(out_dir, fname), "w"), indent=1)
        n_ok += 1
        print(f"{fname:46s} ep {j.get('epochs_run')!s:>3s}  acc {j['accuracy']:.4f}  macro-F1 {j['macro_f1']:.4f}  "
              f"{j.get('wall_clock_s', 0) / 60:5.0f} min  {j.get('hardware')}")
    print(n_ok, "result files written to", out_dir)


if __name__ == "__main__":
    main()
