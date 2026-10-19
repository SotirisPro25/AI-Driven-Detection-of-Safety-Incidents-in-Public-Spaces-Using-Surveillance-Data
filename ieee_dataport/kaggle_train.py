#!/usr/bin/env python3
"""kaggle_train.py -- the 18 classifier runs of Section 6.3.2 on Kaggle (GPU session), from the frame
cache produced by kaggle_prepare.py (its saved output added as input), resumable across sessions
(result files found under /kaggle/input are taken over). train_clip_classifier.py must sit next
to this file."""
import os, sys, glob, json, shutil, subprocess, time, zipfile, threading, queue

INPUT = os.environ.get("KAGGLE_INPUT", "/kaggle/input")
WORK = os.environ.get("KAGGLE_WORKING", "/kaggle/working")
HERE = os.path.dirname(os.path.abspath(__file__))
CLF = os.path.join(HERE, "train_clip_classifier.py")
RESULTS = os.path.join(WORK, "results")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
MAX_RUNS = int(os.environ.get("MAX_RUNS", "0"))          # 0 = all; e.g. 6 for a session budget
HARD_STOP_H = float(os.environ.get("HARD_STOP_H", "11.3"))         # Kaggle ends a session at 12 h, outputs must be written before
TIME_BUDGET_H = float(os.environ.get("TIME_BUDGET_H", "10.2"))   # no new run is started this many hours after the
T_START = float(os.environ.get("SESSION_START", time.time()))     # session start (Kaggle stops a session at 12 h)


WEIGHT_FILES = {"resnet50-11ad3fa6.pth": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
                "r2plus1d_18-91a641e6.pth": "https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth"}


def ensure_weights():
    """The ImageNet weights of ResNet-50 (IMAGENET1K_V2) and the Kinetics-400 weights of R(2+1)D-18, from a
    dataset among the inputs (any folder holding the two files) copied into the torch hub cache, else
    downloaded by torchvision when the notebook has Internet enabled. Stops early when neither is possible."""
    import torch
    ckpt = os.path.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(ckpt, exist_ok=True)
    missing = []
    for name in WEIGHT_FILES:
        dst = os.path.join(ckpt, name)
        if os.path.exists(dst):
            continue
        found = glob.glob(os.path.join(INPUT, "**", name), recursive=True)
        if found:
            shutil.copy(found[0], dst); print("weights from input:", found[0], flush=True)
        else:
            missing.append(name)
    if missing:
        try:
            for name in missing:
                torch.hub.download_url_to_file(WEIGHT_FILES[name], os.path.join(ckpt, name), progress=False)
            print("weights downloaded:", missing, flush=True)
        except Exception as e:
            sys.exit(f"pretrained weights {missing} are neither among the inputs nor downloadable ({e}). Either enable "
                     "Internet in the notebook settings or add a dataset that holds these two files: "
                     + " ".join(WEIGHT_FILES.values()))


def log_lines(tag, name, text, width=1800):
    """long payloads go to the log in numbered pieces, so they can be read back from the Kaggle log viewer"""
    parts = [text[i:i + width] for i in range(0, len(text), width)] or [""]
    for k, part in enumerate(parts, 1):
        print(f"{tag} {name} {k}/{len(parts)} {part}", flush=True)


def log_result(path, printed_names):
    """the result of a run, written into the log as compact JSON (without the per-clip list), the predicted
    class of every test clip as a digit string, and the clip names once per split"""
    j = json.load(open(path))
    tp = j.get("test_predictions", [])
    compact = {k: v for k, v in j.items() if k != "test_predictions"}
    name = os.path.basename(path)
    log_lines("RESULT_JSON", name, json.dumps(compact, separators=(",", ":")))
    log_lines("RESULT_PRED", name, "".join(str(p["pred"]) for p in tp))
    split = j.get("split", "") + ("_raw" if j.get("raw") else "")
    if split not in printed_names:
        printed_names.add(split)
        log_lines("RESULT_TRUE", split, "".join(str(p["true"]) for p in tp))
        names = json.dumps([p["name"] for p in tp], separators=(",", ":"), ensure_ascii=False).encode("utf8")
        import base64, zlib                  # the clip names compressed (about a quarter of the plain length)
        log_lines("RESULT_NAMESZ", split, base64.b64encode(zlib.compress(names, 9)).decode())


def audit_addendum():
    """Checks on the data actually used, printed as AUDIT lines: the source-disjoint split must not share a
    source video between its partitions, also when titles are compared on their letters and digits only
    (the key that reproduces the audit figures of Chapter 2 on the full corpus), and the leakage figures of
    the supplied partition recomputed on this copy with the same key."""
    import csv, re, itertools, collections
    def parse(n):
        stem = n[:-4] if n.lower().endswith(".mp4") else n
        if re.fullmatch(r"\d{6}", stem):
            return None, None, None
        m = re.match(r"^(.*?)(?:_(\d+)_(\d+))?-splitted-(\d+)-(\d+)$", stem)
        if m and int(m.group(5)) - int(m.group(4)) <= 60:
            return m.group(1), (int(m.group(2)) if m.group(2) else None), (int(m.group(3)) if m.group(3) else None)
        m = re.match(r"^(.*?)(?:-splitted-[\d-]+)?_(\d+)_(\d+)$", stem)
        if m:
            return m.group(1), int(m.group(2)), int(m.group(3))
        return stem, None, None
    key = lambda s: re.sub(r"[^0-9a-z\u0430-\u044f\u0451]", "", s.lower())
    def report(tag, rows):                  # rows: (partition, name)
        idx = {}
        for part, n in rows:
            src, f0, f1 = parse(n)
            if src:
                idx.setdefault(part, collections.defaultdict(list))[key(src)].append((f0, f1))
        parts = [p for p in ("train", "val", "test") if p in idx]
        out = {"sources": {p: len(idx[p]) for p in parts}}
        for a, b in itertools.combinations(parts, 2):
            sh = set(idx[a]) & set(idx[b])
            # overlap_pairs compares the frame numbers of the names directly, also across the -splitted-A-B
            # segments, whose frame numbers start again from zero. It is kept as run (the logged figures of the
            # notebook), and verify_frame_overlaps.py gives the count of the thesis (Finding 17).
            ov = sum(1 for s in sh for l in idx[a][s] for r in idx[b][s]
                     if l[0] is not None and r[0] is not None and l[0] < r[1] and r[0] < l[1])
            out[f"{a}-{b}"] = {"shared": len(sh), "files": [sum(len(idx[p][s]) for s in sh) for p in (a, b)], "overlap_pairs": ov}
        common = set.intersection(*(set(idx[p]) for p in parts))
        out["all_three"] = {"sources": len(common), "files": sum(len(idx[p][s]) for p in parts for s in common)}
        n_all = len(rows)
        out["files"] = n_all
        print("AUDIT", tag, json.dumps(out, separators=(",", ":")), flush=True)
    try:
        lst = glob.glob(os.path.join(INPUT, "**", "ieee_listing.csv"), recursive=True)
        dsj = glob.glob(os.path.join(INPUT, "**", "ieee_split_disjoint.csv"), recursive=True)
        if lst:
            report("supplied", [(r["partition"], r["name"]) for r in csv.DictReader(open(lst[0], encoding="utf8"))])
        if dsj:
            report("disjoint", [(r["disjoint_partition"], r["name"]) for r in csv.DictReader(open(dsj[0], encoding="utf8"))])
    except Exception as e:
        print("AUDIT failed:", e, flush=True)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    audit_addendum()
    ensure_weights()
    # the cache: a folder with the manifests (preparation output) or ieee_cache.zip
    man = glob.glob(os.path.join(INPUT, "**", "manifest_IEEE_normalized.json"), recursive=True)
    cache = os.path.join(WORK, "cache")
    if man and not os.path.exists(cache):
        # The preparation may come in shards (SHARD=k/n), each with its own cache folder and manifests: the
        # cached frames of all of them are linked into one folder and the manifests merged. The cache names a
        # clip by tree, class and file name, so a (class, name) pair that occurs in two supplied partitions
        # (5 pairs in the Kaggle copy of the corpus) would share one entry: every such pair is left out of all splits.
        import collections
        tmp, srcs = "/tmp/cachep", sorted(os.path.dirname(m) for m in man)
        for src in srcs:
            for d in os.listdir(src):
                p = os.path.join(src, d)
                if os.path.isdir(p):
                    os.makedirs(os.path.join(tmp, d), exist_ok=True)
                    for f in os.listdir(p):
                        q = os.path.join(tmp, d, f)
                        if not os.path.lexists(q):
                            os.symlink(os.path.join(p, f), q)
        for b in sorted({os.path.basename(f) for src in srcs for f in glob.glob(os.path.join(src, "manifest_*.json"))}):
            M = {}
            for src in srcs:
                if os.path.exists(os.path.join(src, b)):
                    for k, v in json.load(open(os.path.join(src, b))).items():
                        M.setdefault(k, set()).update(tuple(e) for e in v)
            c = collections.Counter(e for k in ("train", "val", "test") for e in M.get(k, ()))
            dup = {e for e, v in c.items() if v > 1}
            N = {k: sorted(e for e in v if e not in dup) for k, v in M.items()}
            json.dump(N, open(os.path.join(tmp, b), "w"))
            print("EXCLUDED", b, sorted(dup), {k: len(v) for k, v in N.items()}, flush=True)
        os.symlink(tmp, cache)
        print("cache shards merged:", len(srcs), flush=True)
    if man:
        src = os.path.dirname(man[0])
        if not os.path.exists(cache):
            os.symlink(src, cache)
    elif not os.path.isdir(cache):
        zips = glob.glob(os.path.join(INPUT, "**", "ieee_cache.zip"), recursive=True)
        if not zips:
            sys.exit("no frame cache found under " + INPUT)
        with zipfile.ZipFile(zips[0]) as zf:
            zf.extractall(WORK)
    print("cache:", cache, "224 px:", len(glob.glob(os.path.join(cache, "224", "*.npy"))),
          "112 px:", len(glob.glob(os.path.join(cache, "112", "*.npy"))), flush=True)
    # results of earlier sessions
    n_prev = 0
    for f in glob.glob(os.path.join(INPUT, "**", "results_*_seed*.json"), recursive=True):
        dst = os.path.join(RESULTS, os.path.basename(f))
        if not os.path.exists(dst):
            shutil.copy(f, dst); n_prev += 1
    print("result files taken over from earlier sessions:", n_prev, flush=True)
    subset = glob.glob(os.path.join(INPUT, "**", "fixed_viewpoint.txt"), recursive=True)
    # the classes present in all three supplied partitions (an incomplete upload leaves some out)
    man_n = json.load(open(os.path.join(cache, "manifest_IEEE_normalized.json")))
    present = None
    for d in ("train", "val", "test"):
        cl = {c for c, _ in man_n.get(d, [])}
        present = cl if present is None else present & cl
    order = ["Backend", "Backend Rollover", "Frontend", "Frontend Rollover", "General Augmented Crash",
             "No Accident Normal Traffic", "Sidehit", "Sidehit Rollover"]
    used = [c for c in order if c in present]
    os.environ["IEEE_CLASSES"] = ",".join(used)
    json.dump({"classes_used": used, "classes_missing": [c for c in order if c not in present],
               "counts": {d: len(v) for d, v in man_n.items()}}, open(os.path.join(RESULTS, "classes_used.json"), "w"), indent=1)
    print("classes used:", used, "| missing from at least one partition:", [c for c in order if c not in present], flush=True)
    common = ["--cache", cache, "--batch", "8", "--accum", "2", "--amp", "--workers", "2", "--out", RESULTS]
    if os.environ.get("EPOCHS"):                       # only for the smoke test; the protocol is 30 with early stopping
        common += ["--epochs", os.environ["EPOCHS"]]
    runs = []
    for seed in SEEDS:
        for backbone in ("r2d", "r2p1d"):
            for split in ("supplied", "disjoint"):
                runs.append(dict(root=os.path.join(WORK, "IEEE_normalized"), split=split, backbone=backbone, seed=seed, raw=False))
        for backbone in ("r2d", "r2p1d"):
            runs.append(dict(root=os.path.join(WORK, "IEEE_as_distributed"), split="supplied", backbone=backbone, seed=seed, raw=True))
    # one run per GPU at a time (Kaggle "GPU T4 x2" gives two cards); the longer ResNet-50 runs first
    import torch
    n_gpu = max(1, torch.cuda.device_count())
    print("GPUs:", n_gpu, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
    runs.sort(key=lambda r: (r["seed"], r["backbone"] != "r2d"))
    todo = []
    for r in runs:
        r["name"] = f"results_{r['backbone']}_{r['split']}{'_raw' if r['raw'] else ''}_seed{r['seed']}.json"
        if os.path.exists(os.path.join(RESULTS, r["name"])):
            print("done already:", r["name"], flush=True)
        else:
            todo.append(r)
    if MAX_RUNS:
        todo = todo[:MAX_RUNS]
    events = queue.Queue()
    lock = threading.Lock()

    def run_one(r, gpu):
        cmd = [sys.executable, CLF, "--root", r["root"], "--split", r["split"], "--backbone", r["backbone"],
               "--seed", str(r["seed"])] + (["--raw"] if r["raw"] else []) + common
        if subset:
            cmd += ["--subset", subset[0]]
        tag = r["name"][8:-5]
        with lock:
            print(f"\n=== {r['name']} on GPU {gpu} {time.strftime('%H:%M:%S')}\n", " ".join(cmd), flush=True)
        t0 = time.time()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        with open(os.path.join(RESULTS, r["name"][:-5] + ".log"), "w") as log:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
            for line in p.stdout:                   # full output to the run log, epoch lines also to the notebook log
                log.write(line)
                if line.startswith(("epoch", "test:", "note:")):
                    with lock:
                        print(f"   [{tag}]", line.rstrip(), flush=True)
            p.wait()
        events.put((r, gpu, p.returncode, time.time() - t0))

    free = list(range(n_gpu))
    active = 0
    durations, printed_names, failures, stop = [], set(), 0, False
    while todo or active:
        while todo and free and not stop:
            elapsed_h = (time.time() - T_START) / 3600
            expect_h = 1.3 * max(durations) / 3600 if durations else 1.6     # safety margin on the longest run so far
            if elapsed_h > TIME_BUDGET_H or elapsed_h + expect_h > HARD_STOP_H:
                print(f"stopping at {elapsed_h:.2f} h: the next run (about {expect_h:.2f} h) would not finish before "
                      f"{HARD_STOP_H} h; the remaining runs continue in the next session (add this version's output "
                      "as input)", flush=True)
                stop = True; break
            r = todo.pop(0)
            threading.Thread(target=run_one, args=(r, free.pop(0)), daemon=True).start()
            active += 1
        if not active:
            break
        r, gpu, rc, dt = events.get()
        active -= 1; free.append(gpu); durations.append(dt)
        with lock:
            print(f"{r['name']}: exit code {rc} after {dt / 60:.0f} min", flush=True)
            if rc != 0:
                print(open(os.path.join(RESULTS, r["name"][:-5] + ".log")).read()[-3000:], flush=True)
                failures += 1
                if failures >= 2:
                    print("two failed runs, no new runs are started", flush=True); stop = True
            else:
                failures = 0
                log_result(os.path.join(RESULTS, r["name"]), printed_names)
        if stop:
            todo = []
    # summary and zip
    rows = []
    for f in sorted(glob.glob(os.path.join(RESULTS, "results_*.json"))):
        j = json.load(open(f))
        rows.append((os.path.basename(f), j.get("epochs_run"), j.get("best_val_macro_f1", 0), j["accuracy"], j["macro_f1"],
                     j.get("wall_clock_s", 0) / 60, j.get("hardware")))
    print(f"\n{'result file':46s} {'ep':>3s} {'val F1':>7s} {'test acc':>9s} {'test F1':>8s} {'min':>4s}  hardware")
    for r in rows:
        print(f"{r[0]:46s} {str(r[1]):>3s} {r[2]:7.4f} {r[3]:9.4f} {r[4]:8.4f} {r[5]:4.0f}  {r[6]}")
    print(len(rows), "of", len(runs), "runs present", flush=True)
    shutil.make_archive(os.path.join(WORK, "ieee_results"), "zip", RESULTS)
    if os.path.islink(cache):              # the cache stays with the preparation output, not copied into this one
        os.remove(cache)
    print("zip: ieee_results.zip (Save Version with outputs to keep it)", flush=True)


if __name__ == "__main__":
    main()
