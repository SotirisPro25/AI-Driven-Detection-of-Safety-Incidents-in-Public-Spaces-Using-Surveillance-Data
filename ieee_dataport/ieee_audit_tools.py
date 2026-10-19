#!/usr/bin/env python3
"""
ieee_audit_tools.py -- the IEEE DataPort audit of Section 2.3.4 and the processing of Section 6.2.3
as one runnable tool, and the inputs that train_clip_classifier.py needs (Tables 6.7 and 6.8).

The corpus as distributed is a tree  <root>/<partition>/<class>/<clip>.mp4  with the partitions
train, val, test (any spelling of case is accepted) and eight class folders.

    python ieee_audit_tools.py listing  --root IEEE                      # -> ieee_listing.csv, census, leakage report
    python ieee_audit_tools.py probe    --root IEEE [--sample 0]         # -> ieee_probe.json (container facts per clip)
    python ieee_audit_tools.py disjoint --root IEEE [--seed 0]           # -> <root>/{train,val,test}_disjoint/<class>/ (symlinks),
                                                                         #    results/disjoint_counts.json (Section 6.2.3), ieee_split_disjoint.csv
    python ieee_audit_tools.py normalize --root IEEE --out IEEE_normalized [--workers 2]
                                                                         # -> re-encoding of every clip of every tree (Section 6.2.3)
    python ieee_audit_tools.py link-tree --root IEEE --out IEEE_as_distributed
                                                                         # -> the clips as distributed, thesis class names (--raw runs)
    python ieee_audit_tools.py metadata-check --probe ieee_probe.json     # -> accuracy of a classifier on container facts alone

`listing` needs only the file names, so it also runs on a text listing: --listing <file> with one
path per line (partition/class/name.mp4), no video files required.
"""
import os, re, sys, json, csv, argparse, collections, itertools, random, subprocess
from multiprocessing import Pool

PARTS = ("train", "val", "test")
CLASSES8 = ["Backend", "Backend Rollover", "Frontend", "Frontend Rollover",
            "General Augmented Crash", "No Accident Normal Traffic", "Sidehit", "Sidehit Rollover"]


def canon_class(folder):
    """Map a class folder name as found on disk (Backend_rollover, General_Augmented_Crash_Download,
    Noaccident_norm_traffic, sidehit_rolover, ...) onto the eight class names of the thesis."""
    k = re.sub(r"[^a-z]", "", folder.lower())
    if k.startswith("general"):
        return "General Augmented Crash"
    if k.startswith("noacc") or "normal" in k or "normtraffic" in k:
        return "No Accident Normal Traffic"
    roll = "rol" in k
    for base in ("backend", "frontend", "sidehit"):
        if k.startswith(base):
            return f"{base.capitalize()} Rollover" if roll else base.capitalize()
    return folder


# ------------------------------------------------------------------ D.1 filename provenance parser
def parse_clip_name(basename):
    """Resolve an IEEE DataPort clip filename into its provenance record: kind
    ('frame-range' | 'sec-window' | 'numeric' | 'other'), source video title,
    frame indices f0/f1, second offsets s0/s1 within the parent. f0/f1 count from the start of the
    segment named in the file (-splitted-A-B, B - A > 60 s) when the source was split, so they are
    source frames only for unsplit sources. The source key is used for grouping and is kept as it was
    in the runs, see clip_extent() for positions in the source."""
    stem = basename[:-4] if basename.lower().endswith('.mp4') else basename
    rec = dict(kind='other', source=stem, f0=None, f1=None, s0=None, s1=None)
    if re.fullmatch(r'\d{6}', stem):     # imported corpus, six-digit index
        return dict(rec, kind='numeric', source=None)
    # Family 2: fixed five-second window of a source video or of a segment.
    m = re.match(r'^(.*?)(?:_(\d+)_(\d+))?-splitted-(\d+)-(\d+)$', stem)
    if m and int(m.group(5)) - int(m.group(4)) <= 60:
        f0, f1 = (int(g) if g else None for g in m.group(2, 3))
        return dict(rec, kind='sec-window', source=m.group(1), f0=f0, f1=f1,
                    s0=int(m.group(4)), s1=int(m.group(5)))
    # Family 1: event-centered cut, start and end frame index of the source.
    m = re.match(r'^(.*?)(?:-splitted-[\d-]+)?_(\d+)_(\d+)$', stem)
    if m:
        return dict(rec, kind='frame-range', source=m.group(1),
                    f0=int(m.group(2)), f1=int(m.group(3)))
    return rec


def in_shard(cls, name, shard):
    """shard "k/n": the clip belongs to part k of n by a CRC of its class and file name, the same rule in
    the normalization and in the frame cache, so that n sessions can share the work"""
    if not shard:
        return True
    import zlib
    k, n = (int(x) for x in shard.split("/"))
    return zlib.crc32(f"{cls}/{name}".encode("utf8")) % n == k


def link_or_copy(src, dst):
    """Symbolic link, else hard link (same volume, no privileges on Windows), else copy."""
    import shutil
    if os.path.lexists(dst):
        return
    src = os.path.abspath(src)
    try:
        os.symlink(src, dst)
    except OSError:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


# ------------------------------------------------------------------ tree enumeration
def norm_part(name):
    n = name.lower()
    for p in PARTS:
        if n == p or n.startswith(p):
            return p
    return None


def enumerate_tree(root):
    """[{partition, cls, cls_folder, name, path}] for <root>/<partition>/<class>/*.mp4. A doubled partition
    level, as in <root>/train/train/<class>/, is stepped through. Class folders are mapped onto the
    thesis names (cls); the folder name on disk is kept in cls_folder."""
    rows = []
    for d in sorted(os.listdir(root)):
        part = norm_part(d)
        pdir = os.path.join(root, d)
        if part is None or not os.path.isdir(pdir) or d.lower().endswith("_disjoint"):
            continue
        subs = [x for x in sorted(os.listdir(pdir)) if os.path.isdir(os.path.join(pdir, x))]
        inner = [x for x in subs if norm_part(x) == part]           # train/train/<class>, other folders ignored
        if inner:
            pdir = os.path.join(pdir, inner[0])
        for cls in sorted(os.listdir(pdir)):
            cdir = os.path.join(pdir, cls)
            if not os.path.isdir(cdir):
                continue
            for f in sorted(os.listdir(cdir)):
                if f.lower().endswith(".mp4"):
                    rows.append(dict(partition=part, cls=canon_class(cls), cls_folder=cls, name=f,
                                     path=os.path.join(cdir, f)))
    return rows


def read_listing(path):
    rows = []
    for line in open(path, encoding="utf8", errors="replace"):
        line = line.strip().replace("\\", "/")
        if not line.lower().endswith(".mp4"):
            continue
        parts = [p for p in line.split("/") if p]
        if len(parts) < 3:
            continue
        part = norm_part(parts[-3])
        if part is None:
            continue
        rows.append(dict(partition=part, cls=canon_class(parts[-2]), cls_folder=parts[-2], name=parts[-1], path=line))
    return rows


# ------------------------------------------------------------------ D.2 census
def census(rows, partition=None):
    counts = collections.Counter(r["cls"] for r in rows if partition is None or r["partition"] == partition)
    total = sum(counts.values())
    for name, n in counts.most_common():
        print(f'  {name:<34} {n:>5}  {100 * n / total:5.1f}%')
    print(f'  {"TOTAL":<34} {total:>5}  100.0%')
    if total:
        print(f'  majority-class baseline: {100 * max(counts.values()) / total:.1f}%')
    return counts


# ------------------------------------------------------------------ D.3 leakage
def index_by_source(files_by_partition):
    index = {}
    for part, names in files_by_partition.items():
        index[part] = src = collections.defaultdict(list)
        for rec in map(parse_clip_name, names):
            if rec['source']:
                src[rec['source']].append(rec)
    return index


def leakage_report(index):
    out = {"pairs": {}, "all_three": None}
    for a, b in itertools.combinations(index, 2):
        shared = set(index[a]) & set(index[b])
        touched = [sum(len(index[p][s]) for s in shared) for p in (a, b)]
        out["pairs"][f"{a}-{b}"] = {"shared_sources": len(shared), "files_touched": {a: touched[0], b: touched[1]}}
        print(f"  {a} / {b}: {len(shared)} shared source videos, touching {touched[0]} and {touched[1]} files")
    common = set.intersection(*(set(p) for p in index.values()))
    n_common_files = sum(len(index[p][s]) for p in index for s in common)
    out["all_three"] = {"sources": len(common), "files": n_common_files}
    print(f"  present in all partitions: {len(common)} sources, {n_common_files} files")
    return out


def clip_extent(basename):
    """Where a clip lies in its source video, from the file name alone: (title, segment start in seconds, kind,
    values), or None for the numeric family. A long source was first cut into segments of about ten minutes,
    named -splitted-A-B with B - A > 60 seconds, and the frame numbers of a clip count from the start of its
    segment, not of the source. kind 'f': frames f0..f1 of the segment. kind 's': seconds s0..s1 of the segment.
    kind 'fs': the five-second window s0..s1 of the cut f0..f1."""
    stem = basename[:-4] if basename.lower().endswith('.mp4') else basename
    if re.fullmatch(r'\d{6}', stem):
        return None
    seg, base, rest = 0, stem, ''
    m = re.match(r'^(.*?)-splitted-(\d+)-(\d+)(.*)$', stem)
    if m and int(m.group(3)) - int(m.group(2)) > 60:
        base, seg, rest = m.group(1), int(m.group(2)), m.group(4)
    full = base + rest
    m = re.match(r'^(.*?)_(\d+)_(\d+)-splitted-(\d+)-(\d+)$', full)
    if m:
        return m.group(1), seg, 'fs', tuple(int(g) for g in m.group(2, 3, 4, 5))
    m = re.match(r'^(.*?)_(\d+)_(\d+)$', full)
    if m:
        return m.group(1), seg, 'f', (int(m.group(2)), int(m.group(3)))
    m = re.match(r'^(.*?)-splitted-(\d+)-(\d+)$', full)
    if m and int(m.group(3)) - int(m.group(2)) <= 60:
        return m.group(1), seg, 's', (int(m.group(2)), int(m.group(3)))
    return None


def _segment_frames(ext, fps):
    """frames of its segment that a clip covers, at a given frame rate of the source"""
    kind, v = ext[2], ext[3]
    if kind == 'f':
        return v[0], v[1]
    if kind == 's':
        return v[0] * fps, v[1] * fps
    return v[0] + v[2] * fps, min(v[1], v[0] + v[3] * fps)


def frame_overlaps(files_by_partition, a, b, fps_grid=(23, 24, 25, 26, 27, 28, 29, 29.97, 30, 31), slack_s=10.0):
    """Pairs of clips, one in partition a and one in partition b, that may show the same source frames.
    Within one segment the frame numbers are compared directly (the frame rate matters only for second
    windows), and such a pair is returned with exact=True when its frame ranges intersect at every frame rate
    of the grid. Across segments the ranges are compared in source time with slack_s seconds of tolerance,
    because segments start at key frames and the source frame rate is not in the name, and the pair is
    returned with exact=False, a candidate. The count of Finding 17 is the set of candidates confirmed on the
    pixels by verify_frame_overlaps.py, not this name-level list. Before rev8f this function compared frame
    numbers of different segments as if they were source frames, which gave the 176 pairs of the old text."""
    def recs(p):
        out = collections.defaultdict(list)
        for n in files_by_partition[p]:
            e = clip_extent(n)
            if e:
                out[e[0]].append((n, e))
        return out
    left, right = recs(a), recs(b)
    out = []
    for title in set(left) & set(right):
        for (x, ex), (y, ey) in itertools.product(left[title], right[title]):
            hits = []
            for fps in fps_grid:
                ix, iy = _segment_frames(ex, fps), _segment_frames(ey, fps)
                if ex[1] == ey[1]:
                    hits.append(min(ix[1], iy[1]) > max(ix[0], iy[0]))
                else:
                    ox, oy = ex[1] * fps, ey[1] * fps
                    hits.append(min(ox + ix[1], oy + iy[1]) + slack_s * fps > max(ox + ix[0], oy + iy[0]))
            if any(hits):
                out.append(dict(title=title, left=x, right=y, same_segment=ex[1] == ey[1],
                                exact=ex[1] == ey[1] and all(hits)))
    return out


def leaked_share(rows, index):
    """share of files whose source appears in another partition (the 48.5% of Section 2.3.4)"""
    by_part = {p: set(index[p]) for p in index}
    n_leak = 0
    for r in rows:
        src = parse_clip_name(r["name"])["source"]
        if src and any(src in by_part[q] for q in by_part if q != r["partition"]):
            n_leak += 1
    return n_leak, len(rows)


# ------------------------------------------------------------------ D.4 source-disjoint split
def source_disjoint_split(records, ratios=(0.70, 0.15, 0.15), seed=0):
    rng, groups, orphans = random.Random(seed), collections.defaultdict(list), []
    for rec in records:
        source = parse_clip_name(rec['name'])['source']
        (groups[source] if source else orphans).append(rec)
    rng.shuffle(orphans)
    names = ['train', 'val', 'test']
    quota, assigned = dict(zip(names, ratios)), {n: [] for n in names}
    order = sorted(groups.values(), key=len, reverse=True)
    total = sum(len(g) for g in order) + len(orphans)
    for group in order + [[r] for r in orphans]:
        target = min(names, key=lambda n: len(assigned[n]) / (quota[n] * total))
        assigned[target].extend(group)
    return assigned


def verify_disjoint(assigned):
    seen = {}
    for part, recs in assigned.items():
        for rec in recs:
            src = parse_clip_name(rec['name'])['source']
            if src and seen.setdefault(src, part) != part:
                raise AssertionError(f'source {src!r} in both {seen[src]} and {part}')
    return True


# ------------------------------------------------------------------ D.5 container normalization
def normalize(src_path, dst_path, seconds=5.0, fps=25, width=1280, height=720):
    vf = (f'fps={fps},'
          f'scale={width}:{height}:force_original_aspect_ratio=decrease,'
          f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,'
          f'tpad=stop_mode=clone:stop_duration={seconds}')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-nostdin', '-i', src_path, '-vf', vf,
                    '-t', str(seconds), '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
                    '-pix_fmt', 'yuv420p', dst_path], check=True)


def probe(path):
    out = subprocess.run([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration,bit_rate',
        '-show_entries', 'stream=codec_type,width,height,avg_frame_rate,nb_frames',
        '-of', 'json', path], capture_output=True, text=True).stdout
    try:
        j = json.loads(out)
    except json.JSONDecodeError:
        return None
    v = next((s for s in j.get("streams", []) if s.get("codec_type") == "video"), {})
    num, den = (v.get("avg_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    return {"duration_s": float(j.get("format", {}).get("duration") or 0), "bit_rate": int(j.get("format", {}).get("bit_rate") or 0),
            "width": int(v.get("width") or 0), "height": int(v.get("height") or 0), "fps": round(fps, 3),
            "nb_frames": int(v.get("nb_frames") or 0), "audio": any(s.get("codec_type") == "audio" for s in j.get("streams", []))}


def _probe_row(r):
    return r["path"], probe(r["path"])


def _normalize_row(args):
    src, dst = args
    if os.path.exists(dst):
        return dst, "exists"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst[:-4] + ".part.mp4"            # written under a temporary name, renamed when complete,
    try:                                    # so an interrupted run never leaves a truncated clip behind
        normalize(src, tmp)
        os.replace(tmp, dst)
        return dst, "ok"
    except subprocess.CalledProcessError as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        return dst, f"failed: {e}"


# ------------------------------------------------------------------ commands
def cmd_listing(a):
    rows = read_listing(a.listing) if a.listing else enumerate_tree(a.root)
    if not rows:
        sys.exit("no clips found")
    for r in rows:
        rec = parse_clip_name(r["name"])
        r.update(kind=rec["kind"], source=rec["source"] or "", f0=rec["f0"], f1=rec["f1"], s0=rec["s0"], s1=rec["s1"])
    with open(a.out, "w", newline="", encoding="utf8") as f:
        wr = csv.DictWriter(f, fieldnames=["partition", "cls", "cls_folder", "name", "kind", "source", "f0", "f1", "s0", "s1", "path"])
        wr.writeheader(); wr.writerows(rows)
    print(f"{len(rows)} clips -> {a.out}")
    folders = sorted({(r["cls_folder"], r["cls"]) for r in rows})
    print("class folders on disk -> thesis names:", ", ".join(f"{a_} -> {b_}" for a_, b_ in folders))
    for p in PARTS:
        if any(r["partition"] == p for r in rows):
            print(f"partition {p}:"); census(rows, p)
    print("families:", collections.Counter(r["kind"] for r in rows))
    index = index_by_source({p: [r["name"] for r in rows if r["partition"] == p] for p in PARTS})
    print("leakage:")
    rep = leakage_report(index)
    files = {p: [r["name"] for r in rows if r["partition"] == p] for p in PARTS}
    for x, y in itertools.combinations(PARTS, 2):
        ov = frame_overlaps(files, x, y)
        n_exact = sum(o["exact"] for o in ov)
        rep["pairs"][f"{x}-{y}"]["same_segment_frame_pairs"] = n_exact
        rep["pairs"][f"{x}-{y}"]["candidate_pairs_to_verify"] = len(ov) - n_exact
        print(f"  {x} / {y}: {n_exact} clip pairs with intersecting frame ranges in the same segment, "
              f"{len(ov) - n_exact} further candidates for verify_frame_overlaps.py")
    n_leak, n_all = leaked_share(rows, index)
    rep["files_with_source_in_another_partition"] = {"n": n_leak, "share": n_leak / n_all if n_all else None}
    print(f"  files whose source appears in another partition: {n_leak} of {n_all} ({100*n_leak/n_all:.1f}%)")
    json.dump(rep, open(os.path.splitext(a.out)[0] + "_leakage.json", "w"), indent=1)


def cmd_probe(a):
    rows = enumerate_tree(a.root)
    if a.sample:
        random.Random(a.seed).shuffle(rows); rows = rows[:a.sample]
    out = {}
    with Pool(a.workers) as pool:
        for i, (path, info) in enumerate(pool.imap_unordered(_probe_row, rows, chunksize=8)):
            out[path] = info
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(rows)}", flush=True)
    by_path = {r["path"]: r for r in rows}
    data = [dict(by_path[p], **(info or {})) for p, info in out.items() if info]
    json.dump(data, open(a.out, "w"), indent=0)
    print(f"{len(data)} clips probed -> {a.out}")


def cmd_disjoint(a):
    rows = enumerate_tree(a.root)
    recs = [dict(name=r["name"], cls=r["cls"], path=r["path"], partition=r["partition"]) for r in rows]
    unknown = sorted({r["cls"] for r in recs if r["cls"] not in CLASSES8})
    if unknown:
        print("warning: class folders not recognized, kept as named:", unknown)
    assigned = source_disjoint_split(recs, seed=a.seed)
    verify_disjoint(assigned)
    counts = {p: {c: sum(1 for r in assigned[p] if r["cls"] == c) for c in CLASSES8 + unknown} for p in assigned}
    os.makedirs(a.results, exist_ok=True)
    json.dump(counts, open(os.path.join(a.results, "disjoint_counts.json"), "w"), indent=1)
    with open(a.out, "w", newline="", encoding="utf8") as f:
        wr = csv.writer(f); wr.writerow(["disjoint_partition", "cls", "name", "distributed_partition"])
        for p in PARTS:
            for r in assigned[p]:
                wr.writerow([p, r["cls"], r["name"], r["partition"]])
    links_root = a.links_root or a.root
    for p in PARTS:
        d = os.path.join(links_root, f"{p}_disjoint")
        for r in assigned[p]:
            cdir = os.path.join(d, r["cls"]); os.makedirs(cdir, exist_ok=True)
            link_or_copy(r["path"], os.path.join(cdir, r["name"]))
    print("source-disjoint split verified;", {p: len(assigned[p]) for p in assigned}, "->", a.out,
          "and", os.path.join(a.results, "disjoint_counts.json"), f"and {links_root}/{{train,val,test}}_disjoint/")


def cmd_normalize(a):
    rows = enumerate_tree(a.root)
    jobs = [(r["path"], os.path.join(a.out, r["partition"], r["cls"], r["name"])) for r in rows
            if in_shard(r["cls"], r["name"], a.shard)]
    if a.shard:
        print(f"shard {a.shard}: {len(jobs)} of {len(rows)} clips", flush=True)
    n_ok = 0
    with Pool(a.workers) as pool:
        for i, (dst, status) in enumerate(pool.imap_unordered(_normalize_row, jobs, chunksize=4)):
            n_ok += status in ("ok", "exists")
            if status.startswith("failed"):
                print(dst, status, file=sys.stderr)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(jobs)}", flush=True)
    print(f"{n_ok} of {len(jobs)} clips normalized into {a.out}", flush=True)
    links_root = a.links_root or a.root
    by_real = {os.path.realpath(x["path"]): x for x in rows}     # one lookup per clip instead of a scan
    for p in PARTS:                      # the disjoint trees, as links into the normalized files
        d = os.path.join(links_root, f"{p}_disjoint")
        if os.path.isdir(d):
            for cls in os.listdir(d):
                for f in os.listdir(os.path.join(d, cls)):
                    src = os.path.realpath(os.path.join(d, cls, f))
                    r = by_real.get(src)
                    if r:
                        dst_dir = os.path.join(a.out, f"{p}_disjoint", cls); os.makedirs(dst_dir, exist_ok=True)
                        link_or_copy(os.path.join(a.out, r["partition"], r["cls"], r["name"]), os.path.join(dst_dir, f))
    print("disjoint trees linked into", a.out)


def cmd_link_tree(a):
    """<out>/{train,val,test}/<thesis class>/<clip> pointing at the clips as distributed (no re-encoding),
    the layout train_clip_classifier.py reads for the --raw runs."""
    rows = enumerate_tree(a.root)
    for r in rows:
        d = os.path.join(a.out, r["partition"], r["cls"]); os.makedirs(d, exist_ok=True)
        link_or_copy(r["path"], os.path.join(d, r["name"]))
    print(f"{len(rows)} clips linked into {a.out}")


def cmd_metadata_check(a):
    """D.5: a classifier on container facts alone, before (and after) normalization"""
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    data = json.load(open(a.probe))
    X = np.array([[d["duration_s"], d["nb_frames"], d["fps"], d["width"], d["height"], d["bit_rate"], int(d["audio"])] for d in data], float)
    y = np.array([d["cls"] for d in data])
    maj = max(collections.Counter(y).values()) / len(y)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    acc = cross_val_score(HistGradientBoostingClassifier(random_state=0), X, y, cv=cv).mean()
    print(f"{len(y)} clips: metadata-only accuracy {100*acc:.1f}% (5-fold) against the majority class {100*maj:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("listing"); s.add_argument("--root", default=""); s.add_argument("--listing", default="")
    s.add_argument("--out", default="ieee_listing.csv"); s.set_defaults(fn=cmd_listing)
    s = sub.add_parser("probe"); s.add_argument("--root", required=True); s.add_argument("--sample", type=int, default=0)
    s.add_argument("--seed", type=int, default=0); s.add_argument("--workers", type=int, default=2)
    s.add_argument("--out", default="ieee_probe.json"); s.set_defaults(fn=cmd_probe)
    s = sub.add_parser("disjoint"); s.add_argument("--root", required=True); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--results", default="results"); s.add_argument("--out", default="ieee_split_disjoint.csv")
    s.add_argument("--links_root", default="", help="where to create {train,val,test}_disjoint (default: --root; "
                                                     "needed when the root is read-only, e.g. /kaggle/input)")
    s.set_defaults(fn=cmd_disjoint)
    s = sub.add_parser("normalize"); s.add_argument("--root", required=True); s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=2)
    s.add_argument("--links_root", default="", help="where the {train,val,test}_disjoint trees are (default: --root)")
    s.add_argument("--shard", default="", help="k/n: normalize only part k of n (the disjoint links cover all clips)")
    s.set_defaults(fn=cmd_normalize)
    s = sub.add_parser("link-tree"); s.add_argument("--root", required=True); s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_link_tree)
    s = sub.add_parser("metadata-check"); s.add_argument("--probe", required=True); s.set_defaults(fn=cmd_metadata_check)
    a = ap.parse_args()
    if a.cmd == "listing" and not (a.root or a.listing):
        ap.error("listing needs --root or --listing")
    a.fn(a)


if __name__ == "__main__":
    main()
