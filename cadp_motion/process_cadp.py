#!/usr/bin/env python3
"""
process_cadp.py  -- idempotent, re-runnable CADP video processor.

Scans <videos> for *.mp4, and for EVERY video (across all upload
batches) assigns a stable 6-digit CADP id, extracts frames into the standard
layout, records probe metadata, and rebuilds a unified manifest + split.

Re-running after new uploads:
  - existing ids are preserved (stable mapping keyed by youtube_id),
  - only NEW videos get extracted,
  - manifest + split are regenerated over the FULL set.

So each time you upload more CADP streams, just run this again.

USAGE
  python process_cadp.py <videos> <cadp_root>

To reproduce the partition of the thesis, copy data/splits/cadp/id_mapping.json and
data/splits/cadp/split.json into <cadp_root> first: existing assignments are kept.
"""
import os, sys, json, subprocess, random

UPLOADS = sys.argv[1] if len(sys.argv) > 1 else "videos"     # folder with <youtube_id>.mp4
ROOT = sys.argv[2] if len(sys.argv) > 2 else "CADP"          # output root
FRAMES = os.path.join(ROOT, "extracted_frames")
PROBE = os.path.join(ROOT, "probe")
MAP_FILE = os.path.join(ROOT, "id_mapping.json")
FPS = 10          # extraction rate
CAP = 300         # max frames per video (sandbox-friendly; use native for full runs)
ID_START = 1000   # 6-digit ids begin at 001000

for d in (ROOT, FRAMES, PROBE):
    os.makedirs(d, exist_ok=True)

# ---- APPEND-ONLY stable id mapping ----
# Existing youtube_id -> vid assignments are NEVER changed. New videos get the
# next free id after the current maximum, in sorted order. This guarantees an
# id like 001007 always refers to the same clip regardless of later uploads.
mapping = {}
if os.path.exists(MAP_FILE):
    mapping = json.load(open(MAP_FILE))

vids = sorted(f[:-4] for f in os.listdir(UPLOADS) if f.endswith(".mp4"))
# next id continues after the current maximum used id (not len, to be robust)
used = [int(v) for v in mapping.values()] if mapping else []
next_id = (max(used) + 1) if used else ID_START
for yid in vids:                      # sorted, but only NEW ones get assigned
    if yid not in mapping:
        mapping[yid] = f"{next_id:06d}"
        next_id += 1
json.dump(mapping, open(MAP_FILE, "w"), indent=2)

def probe(yid):
    r = subprocess.run(["ffprobe","-v","error","-select_streams","v:0",
        "-show_entries","stream=width,height,avg_frame_rate,nb_frames",
        "-show_entries","format=duration","-of","json", f"{UPLOADS}/{yid}.mp4"],
        capture_output=True, text=True)
    try: return json.loads(r.stdout)
    except: return {}

def extract(yid, vid6):
    outdir = os.path.join(FRAMES, vid6)
    if os.path.isdir(outdir) and len(os.listdir(outdir)) > 0:
        return len(os.listdir(outdir))  # already extracted; skip
    os.makedirs(outdir, exist_ok=True)
    subprocess.run(["ffmpeg","-v","error","-i",f"{UPLOADS}/{yid}.mp4",
        "-vf",f"fps={FPS}","-frames:v",str(CAP),"-q:v","3",
        os.path.join(outdir,"%d.jpg")], capture_output=True)
    return len(os.listdir(outdir))

# ---- process every video ----
rows = []
newly = 0
for yid in vids:
    vid6 = mapping[yid]
    pj = probe(yid)
    json.dump({"youtube_id":yid,"vid":vid6,"probe":pj},
              open(os.path.join(PROBE, f"{vid6}.json"),"w"))
    before = os.path.isdir(os.path.join(FRAMES,vid6)) and len(os.listdir(os.path.join(FRAMES,vid6)))>0
    nf = extract(yid, vid6)
    if not before: newly += 1
    st = (pj.get("streams") or [{}])[0]; fmt = pj.get("format",{})
    fr = st.get("avg_frame_rate","0/1")
    try:
        num,den = fr.split("/"); fps = round(float(num)/float(den),1) if float(den) else 0
    except: fps = 0
    rows.append({"vid":vid6,"youtube_id":yid,
                 "width":st.get("width"),"height":st.get("height"),
                 "duration_s":round(float(fmt.get("duration",0)),1),
                 "src_fps":fps,"frames_extracted":nf})

# ---- APPEND-ONLY split: preserve existing assignments, place only new vids ----
SPLIT_FILE = os.path.join(ROOT,"split.json")
split = {"train":[], "val":[], "test":[]}
if os.path.exists(SPLIT_FILE):
    prev_split = json.load(open(SPLIT_FILE))
    for k in split: split[k] = list(prev_split.get(k, []))
assigned = set(split["train"]) | set(split["val"]) | set(split["test"])
allv = sorted(r["vid"] for r in rows)
new_vids = [v for v in allv if v not in assigned]
# assign new videos deterministically to keep ~70/15/15 overall
random.seed(42)
random.shuffle(new_vids)
for v in new_vids:
    n = sum(len(split[k]) for k in split) or 1
    # current fractions; add to whichever bucket is most under target
    targets = {"train":0.70,"val":0.15,"test":0.15}
    deficits = {k:(targets[k]-(len(split[k])/n)) for k in split}
    bucket = max(deficits, key=deficits.get)
    split[bucket].append(v)
for k in split: split[k] = sorted(split[k])
json.dump(split, open(SPLIT_FILE,"w"), indent=2)

# image-set lists
ism = os.path.join(ROOT,"voc/ImageSets/Main"); os.makedirs(ism, exist_ok=True)
def frames_for(v):
    d=os.path.join(FRAMES,v)
    return sorted(os.listdir(d), key=lambda x:int(x.split('.')[0]))
for name,group in [("trainval",split["train"]+split["val"]),
                   ("train",split["train"]),("val",split["val"]),("test",split["test"])]:
    with open(os.path.join(ism,f"{name}.txt"),"w") as f:
        for v in sorted(group):
            for fr in frames_for(v):
                f.write(f"{v}/{fr.split('.')[0]}\n")

# annotation folders
for d in ("annotations_vatic","masked_frames","csv","voc/Annotations","voc/JPEGImages"):
    os.makedirs(os.path.join(ROOT,d), exist_ok=True)

# manifests
split_of = {v:k for k,g in split.items() for v in g}
for r in rows:
    r["split"] = split_of[r["vid"]]
    r["annotation_expected"] = f"annotations_vatic/{r['vid']}.txt"
rows.sort(key=lambda r:r["vid"])
json.dump(rows, open(os.path.join(ROOT,"processing_manifest.json"),"w"), indent=2)
import csv
with open(os.path.join(ROOT,"processing_manifest.csv"),"w",newline="") as f:
    wr=csv.DictWriter(f,fieldnames=["vid","youtube_id","split","width","height",
        "duration_s","src_fps","frames_extracted","annotation_expected"])
    wr.writeheader(); wr.writerows(rows)

# summary
tot_dur = sum(r["duration_s"] for r in rows)/60
tot_fr = sum(r["frames_extracted"] for r in rows)
res = {}
for r in rows: res[f"{r['width']}x{r['height']}"] = res.get(f"{r['width']}x{r['height']}",0)+1
print(f"TOTAL videos: {len(rows)}  (newly extracted this run: {newly})")
print(f"total duration: {tot_dur:.1f} min | frames: {tot_fr}")
print(f"split train/val/test: {len(split['train'])}/{len(split['val'])}/{len(split['test'])}")
print(f"resolutions: {res}")
print(f"id range: {rows[0]['vid']}..{rows[-1]['vid']}")
