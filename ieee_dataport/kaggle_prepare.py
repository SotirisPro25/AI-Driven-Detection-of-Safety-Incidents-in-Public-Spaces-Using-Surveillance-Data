#!/usr/bin/env python3
"""kaggle_prepare.py -- the preparation of the IEEE DataPort corpus on Kaggle (CPU session):
audit (Section 2.3.4), source-disjoint split, normalization and container check (Section 6.2.3), frame
cache for the GPU runs, contact sheets for the viewpoint labeling. Resumable. Expects the corpus
as distributed under /kaggle/input (train/val/test partitions, class folders inside) and the
scripts ieee_audit_tools.py and train_clip_classifier.py next to this file."""
import os, re, sys, glob, json, csv, shutil, subprocess, zipfile, time

INPUT = os.environ.get("KAGGLE_INPUT", "/kaggle/input")
WORK = os.environ.get("KAGGLE_WORKING", "/kaggle/working")
HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "ieee_audit_tools.py")
CLF = os.path.join(HERE, "train_clip_classifier.py")
PY = sys.executable
AUDIT = os.path.join(WORK, "audit_outputs")
SHARD = os.environ.get("SHARD", "")          # "k/n": this session prepares part k of n of the clips
T_START = time.time()


def run(args, cwd=None, timeout=None, required=True):
    print("\n$", " ".join(args), flush=True)
    t0 = time.time()
    try:
        subprocess.run(args, check=True, cwd=cwd, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  stopped after {timeout:.0f} s; the preparation continues without this step", flush=True)
        if required:
            raise
    print(f"  ({time.time() - t0:.0f} s)", flush=True)


sys.path.insert(0, HERE)
from ieee_audit_tools import PARTS, CLASSES8, canon_class, norm_part  # noqa: E402


def has_classes(pdir):
    """at least two of the eight class folders directly inside pdir"""
    try:
        subs = [x for x in os.listdir(pdir) if os.path.isdir(os.path.join(pdir, x))]
    except OSError:
        return False
    return sum(1 for x in subs if canon_class(x) in CLASSES8) >= 2


def partition_dirs(base, max_depth=6):
    """{partition: [dirs]} for every folder named like a partition that holds the class folders,
    directly or through a doubled level (train/train/<class>), anywhere under base"""
    found = {}
    base = os.path.abspath(base)
    for d, subs, files in os.walk(base):
        if d[len(base):].count(os.sep) > max_depth:
            subs[:] = []
            continue
        for s in list(subs):
            part = norm_part(s)
            if part is None or s.lower().endswith("_disjoint"):
                continue
            pdir = os.path.join(d, s)
            inner = [x for x in sorted(os.listdir(pdir)) if os.path.isdir(os.path.join(pdir, x)) and norm_part(x) == part]
            if has_classes(pdir) or (inner and has_classes(os.path.join(pdir, inner[0]))):
                found.setdefault(part, []).append(pdir)
                subs.remove(s)                       # do not descend into a partition that was found
    return found


def infer_partition(dirs):
    """partition from the directory components (dataset slug included), the innermost match wins"""
    part = None
    for d in dirs:
        k = d.lower()
        for p, keys in (("train", ("train",)), ("val", ("validat", "valid", "val")), ("test", ("test",))):
            if any(x in k for x in keys):
                part = p
    return part


def infer_class(dirs):
    """thesis class name from the directory components, the innermost match wins"""
    cls = None
    for d in dirs:
        k = re.sub(r"[^a-z]", "", d.lower())
        c = None
        if "general" in k or "augment" in k:
            c = "General Augmented Crash"
        elif "noacc" in k or "normtraf" in k or "normaltraf" in k:
            c = "No Accident Normal Traffic"
        else:
            for base in ("sidehit", "frontend", "backend"):
                i = k.find(base)
                if i >= 0:
                    c = base.capitalize() + (" Rollover" if "rol" in k[i + len(base):] else "")
                    break
        if c:
            cls = c
    return cls


def assemble_root(base, work):
    """Every .mp4 under the inputs is placed by partition and class, inferred from its directory path
    (dataset names included), so the corpus may arrive as one dataset or as one dataset per partition or
    per class folder. Duplicates (same partition, class and file name) are kept once. The result is a
    tree of links <work>/IEEE_dataset/<partition>/<class>/<clip>.mp4 and a census of what is available."""
    root = os.path.join(work, "IEEE_dataset")
    shutil.rmtree(root, ignore_errors=True)
    census, skipped, seen = {}, [], set()
    for d, subs, files in os.walk(base, followlinks=True):
        rel = os.path.relpath(d, base)
        dirs = [] if rel == "." else rel.split(os.sep)
        for f in files:
            if not f.lower().endswith(".mp4"):
                continue
            part, cls = infer_partition(dirs), infer_class(dirs)
            if part is None or cls is None:
                skipped.append(os.path.join(rel, f)); continue
            key = (part, cls, f)
            if key in seen:
                continue
            seen.add(key)
            dst_dir = os.path.join(root, part, cls)
            os.makedirs(dst_dir, exist_ok=True)
            os.symlink(os.path.join(d, f), os.path.join(dst_dir, f))
            census.setdefault(part, {}).setdefault(cls, 0)
            census[part][cls] += 1
    expected = {"train": 3912, "val": 1052, "test": 726}
    print("available clips (partition x class):", flush=True)
    for p in PARTS:
        row = census.get(p, {})
        print(f"  {p:5s} {sum(row.values()):5d} of {expected[p]:5d} expected | "
              + ", ".join(f"{c}: {row.get(c, 0)}" for c in CLASSES8), flush=True)
    if skipped:
        print(len(skipped), "clips with no recognizable partition or class were left out, e.g.", skipped[:3], flush=True)
    json.dump({"census": census, "expected_totals": expected, "skipped": skipped},
              open(os.path.join(AUDIT, "available_census.json"), "w"), indent=1)
    missing = [p for p in PARTS if p not in census]
    return (None, missing) if missing else (root, [])


def ensure_ffmpeg():
    """ffmpeg and ffprobe on the PATH (the Kaggle image ships them; otherwise apt-get, which needs the
    notebook's Internet option, or the static ffmpeg of imageio-ffmpeg for the normalization alone)"""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    print("ffmpeg/ffprobe not on PATH: trying apt-get (needs Internet enabled in the notebook settings)", flush=True)
    subprocess.run("apt-get update -qq && apt-get install -y -qq ffmpeg", shell=True, check=False)
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        bindir = os.path.join(WORK, "bin"); os.makedirs(bindir, exist_ok=True)
        if not os.path.exists(os.path.join(bindir, "ffmpeg")):
            os.symlink(exe, os.path.join(bindir, "ffmpeg"))
        os.environ["PATH"] = bindir + os.pathsep + os.environ["PATH"]
        print("using the static ffmpeg of imageio-ffmpeg:", exe, flush=True)
    except Exception as e:
        print("imageio-ffmpeg not usable:", e, flush=True)
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        sys.exit("ffmpeg and ffprobe are required (normalization and container check). Enable Internet in the "
                 "notebook settings so that apt-get can install them, then run again.")


def main():
    os.makedirs(os.path.join(AUDIT, "results"), exist_ok=True)
    ensure_ffmpeg()
    print("inputs:", sorted(os.listdir(INPUT)) if os.path.isdir(INPUT) else "none", flush=True)
    root, missing = assemble_root(INPUT, WORK)
    if root is None:
        zips = glob.glob(os.path.join(INPUT, "**", "*.zip"), recursive=True)
        for z in zips:
            print("extracting", z, flush=True)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(os.path.join(WORK, "IEEE_dataset_zip"))
        if zips:
            root, missing = assemble_root(os.path.join(WORK, "IEEE_dataset_zip"), WORK)
    if root is None:
        sys.exit("partition folders with class folders inside are missing for: " + ", ".join(missing)
                 + f" (looked under {INPUT}; add the dataset(s) with the train, val and test folders as input)")
    print("corpus root:", root, flush=True)
    print("ffmpeg:", shutil.which("ffmpeg"), "ffprobe:", shutil.which("ffprobe"), "cpus:", os.cpu_count(), flush=True)
    links = os.path.join(WORK, "links")
    norm = os.path.join(WORK, "IEEE_normalized")
    raw = os.path.join(WORK, "IEEE_as_distributed")
    cache = os.path.join(WORK, "cache")
    nw = str(max(1, min(4, os.cpu_count() or 2)))
    # D.1 to D.3
    run([PY, TOOL, "listing", "--root", root, "--out", "ieee_listing.csv"], cwd=AUDIT)
    # D.4
    run([PY, TOOL, "disjoint", "--root", root, "--seed", "0", "--results", "results",
         "--out", "ieee_split_disjoint.csv", "--links_root", links], cwd=AUDIT)
    print(json.dumps(json.load(open(os.path.join(AUDIT, "results", "disjoint_counts.json"))), indent=1), flush=True)
    # D.5
    shard = ["--shard", SHARD] if SHARD else []
    run([PY, TOOL, "normalize", "--root", root, "--links_root", links, "--out", norm, "--workers", nw] + shard, cwd=AUDIT)
    run([PY, TOOL, "link-tree", "--root", root, "--out", raw], cwd=AUDIT)
    if not SHARD:                     # the container check needs every clip; a sharded run skips it
        run([PY, TOOL, "probe", "--root", root, "--out", "ieee_probe_raw.json", "--workers", nw], cwd=AUDIT)
        run([PY, TOOL, "probe", "--root", norm, "--out", "ieee_probe_normalized.json", "--workers", nw], cwd=AUDIT)
        with open(os.path.join(AUDIT, "metadata_check.txt"), "w") as f:
            for name in ("ieee_probe_raw.json", "ieee_probe_normalized.json"):
                out = subprocess.run([PY, TOOL, "metadata-check", "--probe", name], cwd=AUDIT, capture_output=True, text=True)
                print(name, out.stdout.strip(), flush=True)
                f.write(name + ": " + out.stdout.strip() + "\n")
    # frame cache for the GPU runs: both trees, both input sizes. A step that would run past 11 h of
    # the 12 h session is stopped, so that the session ends normally and keeps its output.
    for tree in (norm, raw):          # one decode per clip gives both input sizes (224 px and 112 px)
        left = 11 * 3600 - (time.time() - T_START)
        run([PY, CLF, "--root", tree, "--split", "supplied", "--backbone", "r2d", "--sizes", "224,112",
             "--cache", cache, "--build_cache_only", "--workers", nw] + shard,
            timeout=max(60, left), required=False)
    n224 = len(glob.glob(os.path.join(cache, "224", "*.npy"))); n112 = len(glob.glob(os.path.join(cache, "112", "*.npy")))
    print("cache: 224 px", n224, "| 112 px", n112, "| manifests", sorted(os.path.basename(m) for m in glob.glob(os.path.join(cache, "manifest_*.json"))), flush=True)
    # contact sheets of the test partition for the fixed-viewpoint labeling
    if not SHARD or SHARD.startswith("0/"):
        contact_sheets(os.path.join(AUDIT, "ieee_listing.csv"), os.path.join(WORK, "contact_sheets"))
    # keep the output small
    shutil.rmtree(norm, ignore_errors=True)
    shutil.rmtree(os.path.join(WORK, "IEEE_dataset"), ignore_errors=True)
    shutil.rmtree(os.path.join(WORK, "IEEE_dataset_zip"), ignore_errors=True)
    shutil.rmtree(raw, ignore_errors=True)
    shutil.rmtree(links, ignore_errors=True)
    shutil.make_archive(os.path.join(WORK, "audit_outputs"), "zip", AUDIT)
    print("\nPREPARATION COMPLETE. Outputs: cache/, audit_outputs.zip, contact_sheets/. Save Version with outputs.", flush=True)


def contact_sheets(listing_csv, out_dir, per_sheet=20):
    import cv2, numpy as np
    rows = [r for r in csv.DictReader(open(listing_csv, encoding="utf8")) if r["partition"] == "test"]
    os.makedirs(out_dir, exist_ok=True)
    tiles, sheet_no = [], 0
    with open(os.path.join(out_dir, "index.csv"), "w", newline="", encoding="utf8") as f:
        wr = csv.writer(f); wr.writerow(["index", "sheet", "cls", "name"])
        for i, r in enumerate(rows):
            img = None
            try:
                cap = cv2.VideoCapture(r["path"]); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2)); ok, img = cap.read(); cap.release()
                if not ok:
                    img = None
            except Exception:
                img = None
            if img is None:
                img = np.zeros((180, 320, 3), np.uint8)
            img = cv2.resize(img, (320, 180))
            cv2.putText(img, str(i), (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(img, str(i), (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            tiles.append(img); wr.writerow([i, sheet_no, r["cls"], r["name"]])
            if len(tiles) == per_sheet or i == len(rows) - 1:
                while len(tiles) < per_sheet:
                    tiles.append(np.zeros((180, 320, 3), np.uint8))
                sheet = np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, per_sheet, 4)])
                cv2.imwrite(os.path.join(out_dir, f"sheet_{sheet_no:03d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
                tiles, sheet_no = [], sheet_no + 1
    print(len(rows), "test clips on", sheet_no, "contact sheets in", out_dir, flush=True)


if __name__ == "__main__":
    main()
