#!/usr/bin/env python3
"""
train_clip_classifier.py  --  eight-class accident clip classifier on IEEE DataPort.

Two backbones (Table 6.5):
  --backbone r2d   : ResNet-50 frame encoder, features averaged over 16 frames
  --backbone r2p1d : R(2+1)D-18 spatio-temporal network (Kinetics-400 weights)
Two splits (Section 6.3.2):
  --split supplied : the partition as distributed
  --split disjoint : the source-disjoint split of Section 6.2.3
Optional:
  --subset FILE    : text file with one clip basename per line (the fixed-viewpoint
                     subset of Section 6.2.3); the test metrics are also reported on it
  --raw            : tag a run on the un-normalized clips (Sections 6.3.2 and 6.4.2)

Each run writes ONE results file
    <out>/results_<backbone>_<split>[_raw]_seed<seed>.json
with accuracy, macro-F1, per-class recall, the confusion matrix, the majority
baseline, the subset metrics, the training history, epochs run, wall-clock
time and the inference time per clip.

USAGE
  python train_clip_classifier.py --root IEEE_normalized --split disjoint --backbone r2p1d --seed 0
"""
import os, sys, json, argparse, random, time, platform
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from sklearn.metrics import f1_score, confusion_matrix, recall_score

CLASSES = ["Backend", "Backend Rollover", "Frontend", "Frontend Rollover",
           "General Augmented Crash", "No Accident Normal Traffic",
           "Sidehit", "Sidehit Rollover"]
if os.environ.get("IEEE_CLASSES"):             # restrict to the classes present in all three partitions
    CLASSES = [c for c in CLASSES if c in os.environ["IEEE_CLASSES"].split(",")]
N_FRAMES = 16


def decode_clip(path):
    """Every frame of a clip as a (T, H, W, C) uint8 tensor, RGB. OpenCV first (portable, no
    torchvision video backend needed), torchvision.io.read_video as the fallback."""
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(img[:, :, ::-1].copy())            # BGR -> RGB
        cap.release()
        if frames:
            return torch.from_numpy(np.stack(frames))
    except Exception:
        pass
    from torchvision.io import read_video
    v, _, _ = read_video(path, pts_unit="sec")
    return v


def decode_resized(path, size, max_frames=20000):
    """Every frame of a clip resized to size x size as soon as it is decoded, (T, size, size, C) uint8 RGB.
    Same frames as resizing the output of decode_clip, but the memory grows with the small frames only,
    so a long full-HD clip as distributed cannot exhaust the memory of the session."""
    import cv2
    cap = cv2.VideoCapture(path)
    out = []
    while len(out) < max_frames:
        ok, img = cap.read()
        if not ok:
            break
        out.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)[:, :, ::-1])   # BGR -> RGB
    cap.release()
    if len(out) >= max_frames:
        print(f"note: {path} decoded up to its first {max_frames} frames", file=sys.stderr, flush=True)
    if out:
        return np.stack(out)
    v = decode_clip(path).numpy()
    return np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR) for f in v])


def cache_key(tree_tag, cls, basename):
    """A clip is identified by the tree it comes from (IEEE_normalized or IEEE_as_distributed), its
    class and its file name, so the supplied and the disjoint split share one cache entry and the
    cache can travel to another machine (Colab, Kaggle) without the videos."""
    import hashlib
    return hashlib.sha1(f"{tree_tag}/{cls}/{basename}".encode("utf8")).hexdigest()


def sample_frames(path, size, cache_dir=None, key=None):
    """The 16 uniformly spaced frames of a clip, resized to size x size, as a (16, size, size, 3)
    uint8 array. With cache_dir, the frames are decoded once and kept as JPEG bytes
    (<cache_dir>/<size>/<key>.npy), so that every later epoch reads about 0.3 MB per clip
    instead of decoding a 125-frame 1280 x 720 video, and a machine that holds only the cache
    needs no videos at all."""
    import cv2
    cpath = None
    if cache_dir:
        if key is None:
            import hashlib
            key = hashlib.sha1(os.path.abspath(path).encode()).hexdigest()
        cpath = os.path.join(cache_dir, str(size), key + ".npy")
        if os.path.exists(cpath):
            blobs = np.load(cpath, allow_pickle=True)
            return np.stack([cv2.cvtColor(cv2.imdecode(b, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for b in blobs])
    v = decode_resized(path, size)                             # (T,size,size,C) uint8 RGB
    idx = np.linspace(0, len(v) - 1, N_FRAMES).astype(int)     # 16 uniform frames
    frames = np.ascontiguousarray(v[idx])
    if cpath:
        write_cache(frames, cpath)
    return frames


def write_cache(frames, cpath):
    import cv2
    os.makedirs(os.path.dirname(cpath), exist_ok=True)
    blobs = np.empty(len(frames), dtype=object)
    for k, fr in enumerate(frames):
        blobs[k] = cv2.imencode(".jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])[1]
    tmp = cpath + ".tmp.npy"
    np.save(tmp, blobs, allow_pickle=True); os.replace(tmp, cpath)


def cache_sizes(path, sizes, cache_dir, key, max_frames=20000):
    """The cache entries of one clip at several input sizes from a single decode: the same frames
    as sample_frames gives for each size separately."""
    import cv2
    todo = [z for z in sizes if not os.path.exists(os.path.join(cache_dir, str(z), key + ".npy"))]
    if not todo:
        return
    cap = cv2.VideoCapture(path)
    outs, n = {z: [] for z in todo}, 0
    while n < max_frames:
        ok, img = cap.read()
        if not ok:
            break
        for z in todo:
            outs[z].append(cv2.resize(img, (z, z), interpolation=cv2.INTER_LINEAR)[:, :, ::-1])
        n += 1
    cap.release()
    if n == 0:                                                  # OpenCV could not read it: the fallback path
        for z in todo:
            sample_frames(path, z, cache_dir, key)
        return
    idx = np.linspace(0, n - 1, N_FRAMES).astype(int)
    for z in todo:
        write_cache(np.ascontiguousarray(np.stack([outs[z][j] for j in idx])), os.path.join(cache_dir, str(z), key + ".npy"))


class ClipDataset(Dataset):
    """Reads <root>/<split_dir>/<class>/<clip>.mp4 (normalized clips, Section 6.2.3). When the
    directory is absent but a frame cache with a manifest for this tree exists, the items come
    from the manifest and the frames from the cache (no videos needed)."""
    def __init__(self, root, split_dir, size, train, cache_dir=None):
        self.items, self.keys = [], []
        self.tree_tag = os.path.basename(os.path.normpath(root))
        sdir = os.path.join(root, split_dir)
        manifest = os.path.join(cache_dir, f"manifest_{self.tree_tag}.json") if cache_dir else None
        if os.path.isdir(sdir):
            for ci, c in enumerate(CLASSES):
                d = os.path.join(sdir, c)
                if os.path.isdir(d):
                    for f in sorted(os.listdir(d)):
                        if f.endswith(".mp4"):
                            self.items.append((os.path.join(d, f), ci)); self.keys.append(cache_key(self.tree_tag, c, f))
        elif manifest and os.path.exists(manifest):
            for c, f in json.load(open(manifest))[split_dir]:
                if c not in CLASSES:
                    continue
                self.items.append((os.path.join(sdir, c, f), CLASSES.index(c))); self.keys.append(cache_key(self.tree_tag, c, f))
        else:
            raise FileNotFoundError(f"neither {sdir} nor a cache manifest for {self.tree_tag}")
        self.size, self.train, self.cache_dir = size, train, cache_dir

    def __len__(self): return len(self.items)

    def manifest_entries(self):
        return [(CLASSES[y], os.path.basename(p)) for p, y in self.items]

    def __getitem__(self, i):
        path, y = self.items[i]
        frames = sample_frames(path, self.size, self.cache_dir, self.keys[i])   # (16,S,S,3) uint8
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # (16,C,S,S)
        if self.train and random.random() < 0.5:
            x = torch.flip(x, dims=[3])                        # horizontal flip
        mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1)
        std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1)
        return (x - mean) / std, y


class R2D(nn.Module):
    """ResNet-50 per frame, temporal average pooling, 8-way softmax."""
    def __init__(self):
        super().__init__()
        m = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        m.fc = nn.Identity()
        self.enc, self.fc = m, nn.Linear(2048, len(CLASSES))

    def forward(self, x):                                      # x: (B,16,C,H,W)
        b, t = x.shape[:2]
        f = self.enc(x.flatten(0, 1)).view(b, t, -1).mean(1)
        return self.fc(f)


class R2P1D(nn.Module):
    """R(2+1)D-18 with Kinetics-400 initialization."""
    def __init__(self):
        super().__init__()
        m = torchvision.models.video.r2plus1d_18(weights="KINETICS400_V1")
        m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
        self.net = m

    def forward(self, x):                                      # (B,16,C,H,W)->(B,C,16,H,W)
        return self.net(x.permute(0, 2, 1, 3, 4))


def _cache_one(args):
    path, size, cache, key = args
    import signal

    def _timeout(*_):
        raise TimeoutError("no result after 600 s")
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(600)                       # one clip never holds up the whole cache
    try:
        if isinstance(size, (list, tuple)):
            cache_sizes(path, size, cache, key)
        else:
            sample_frames(path, size, cache, key)
    except Exception as e:
        print("cache failed for", path, e, file=sys.stderr, flush=True)
    finally:
        signal.alarm(0)
    return path


def predict(model, loader, dev):
    """Returns (true labels, predicted labels, mean inference ms per clip)."""
    model.eval(); ys, ps, ms = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev)
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(x)
            if dev == "cuda":
                torch.cuda.synchronize()
            ms.append((time.perf_counter() - t0) * 1000.0 / len(y))
            ps += out.argmax(1).cpu().tolist(); ys += y.tolist()
    return ys, ps, (float(np.mean(ms[1:])) if len(ms) > 1 else float(np.mean(ms)))


def metrics(ys, ps):
    acc = float(np.mean(np.array(ys) == np.array(ps)))
    counts = np.bincount(ys, minlength=len(CLASSES))
    return {"accuracy": acc,
            "macro_f1": float(f1_score(ys, ps, average="macro")),
            "recall_per_class": recall_score(ys, ps, average=None,
                                             labels=list(range(len(CLASSES)))).tolist(),
            "confusion": confusion_matrix(ys, ps, labels=list(range(len(CLASSES)))).tolist(),
            "test_counts": counts.tolist(),
            "majority_baseline": float(counts.max() / counts.sum()),
            "n": int(len(ys))}


def evaluate(model, loader, dev):
    ys, ps, ms = predict(model, loader, dev)
    res = metrics(ys, ps)
    res["ms_per_clip"] = ms
    return res, ys, ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", choices=["supplied", "disjoint"], required=True)
    ap.add_argument("--backbone", choices=["r2d", "r2p1d"], required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset", default="", help="fixed-viewpoint clip basenames, one per line")
    ap.add_argument("--raw", action="store_true", help="run on un-normalized clips (tag only)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--batch", type=int, default=16, help="clips per forward pass (memory)")
    ap.add_argument("--accum", type=int, default=1,
                    help="gradient-accumulation steps; the effective batch of Table 6.5 is batch x accum = 16")
    ap.add_argument("--amp", action="store_true", help="mixed precision (float16 autocast), halves activation memory")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader workers (each decodes one clip at a time)")
    ap.add_argument("--cache", default="", help="directory for the JPEG frame cache (decode every clip once)")
    ap.add_argument("--shard", default="", help="k/n: with --build_cache_only, cache only part k of n")
    ap.add_argument("--sizes", default="", help="with --build_cache_only: input sizes cached from one decode, e.g. 224,112")
    ap.add_argument("--build_cache_only", action="store_true",
                    help="decode and cache the frames of the three splits at this backbone's size, then exit (no GPU needed)")
    a = ap.parse_args()
    if a.batch * a.accum != 16:
        print(f"note: effective batch {a.batch * a.accum} differs from the 16 of Table 6.5")
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    size = 224 if a.backbone == "r2d" else 112
    sd = {"supplied": ("train", "val", "test"),
          "disjoint": ("train_disjoint", "val_disjoint", "test_disjoint")}[a.split]
    cache = a.cache or None
    tr, va, te = (ClipDataset(a.root, sd[0], size, True, cache),
                  ClipDataset(a.root, sd[1], size, False, cache),
                  ClipDataset(a.root, sd[2], size, False, cache))
    if a.build_cache_only:
        if not cache:
            sys.exit("--build_cache_only needs --cache DIR")
        from multiprocessing import Pool
        sizes = [int(z) for z in a.sizes.split(",")] if a.sizes else [size]
        # every split directory of the tree goes into the manifest; the clips are cached once each
        all_dirs = ("train", "val", "test", "train_disjoint", "val_disjoint", "test_disjoint")
        manifest, seen, jobs = {}, set(), []
        for d in all_dirs:
            if not os.path.isdir(os.path.join(a.root, d)):
                continue
            ds = ClipDataset(a.root, d, size, False, cache)
            manifest[d] = ds.manifest_entries()
            for (p_, y_), k in zip(ds.items, ds.keys):
                if k in seen or not os.path.exists(p_):          # a link to a clip of another shard
                    continue
                if a.shard:                                       # same rule as ieee_audit_tools.in_shard
                    import zlib
                    k_, n_ = (int(x) for x in a.shard.split("/"))
                    if zlib.crc32(f"{CLASSES[y_]}/{os.path.basename(p_)}".encode("utf8")) % n_ != k_:
                        continue
                seen.add(k); jobs.append((p_, sizes if len(sizes) > 1 else size, cache, k))
        os.makedirs(cache, exist_ok=True)
        json.dump(manifest, open(os.path.join(cache, f"manifest_{tr.tree_tag}.json"), "w"))
        t0 = time.time()
        with Pool(max(1, a.workers)) as pool:
            for i, _ in enumerate(pool.imap_unordered(_cache_one, jobs, chunksize=4)):
                if (i + 1) % 200 == 0:
                    print(f"  cached {i+1}/{len(jobs)} clips, {time.time()-t0:.0f} s", flush=True)
        print(f"cache complete: {len(jobs)} clips at {','.join(map(str, sizes))} px in {cache}, manifest for {tr.tree_tag} "
              f"with {', '.join(f'{k}: {len(v)}' for k, v in manifest.items())}")
        return
    counts = np.bincount([y for _, y in tr.items], minlength=len(CLASSES))
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32)
    model = (R2D() if a.backbone == "r2d" else R2P1D()).to(dev)
    crit = nn.CrossEntropyLoss(weight=weights.to(dev))          # inverse-frequency weights
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    L = lambda ds, sh: DataLoader(ds, batch_size=a.batch, shuffle=sh, num_workers=a.workers,
                                  pin_memory=(dev == "cuda"), persistent_workers=a.workers > 0)
    scaler = torch.amp.GradScaler("cuda", enabled=(a.amp and dev == "cuda"))
    best, bad, best_state = -1.0, 0, None
    history = {"train_loss": [], "val_accuracy": [], "val_macro_f1": []}
    t_start = time.time()
    epochs_run = 0
    for ep in range(a.epochs):
        model.train(); losses = []
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(L(tr, True)):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(a.amp and dev == "cuda")):
                loss = crit(model(x.to(dev, non_blocking=True)), y.to(dev, non_blocking=True))
            scaler.scale(loss / a.accum).backward()
            if (i + 1) % a.accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
        if (i + 1) % a.accum != 0:                              # flush a trailing partial accumulation
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        v, _, _ = evaluate(model, L(va, False), dev)
        epochs_run = ep + 1
        history["train_loss"].append(float(np.mean(losses)))
        history["val_accuracy"].append(v["accuracy"]); history["val_macro_f1"].append(v["macro_f1"])
        print(f"epoch {ep+1}: val acc {v['accuracy']:.4f} macro-F1 {v['macro_f1']:.4f}")
        if v["macro_f1"] > best:
            best, bad, best_state = v["macro_f1"], 0, {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience: break                         # early stopping on macro-F1
    model.load_state_dict(best_state)
    res, ys, ps = evaluate(model, L(te, False), dev)
    # per-clip test predictions, so that a fixed-viewpoint list can be applied later
    res["test_predictions"] = [{"name": os.path.basename(p_), "true": int(y_), "pred": int(p2)}
                               for (p_, _), y_, p2 in zip(te.items, ys, ps)]
    # fixed-viewpoint subset (Section 6.2.3): same predictions, restricted to listed clips
    if a.subset and os.path.exists(a.subset):
        names = {l.strip() for l in open(a.subset) if l.strip()}
        keep = [i for i, (p, _) in enumerate(te.items) if os.path.basename(p) in names]
        if keep:
            sub = metrics([ys[i] for i in keep], [ps[i] for i in keep])
            res["subset"] = {"name": "fixed_viewpoint", **sub}
    res.update(kind="classifier", split=a.split, backbone=a.backbone, seed=a.seed, raw=a.raw,
               batch=a.batch, accum=a.accum, effective_batch=a.batch * a.accum, amp=a.amp,
               classes=CLASSES, train_counts=counts.tolist(), epochs_run=epochs_run,
               epochs_requested=a.epochs, best_val_macro_f1=best, history=history,
               wall_clock_s=time.time() - t_start,
               hardware=(torch.cuda.get_device_name(0) if dev == "cuda" else platform.processor() or "CPU"),
               torch_version=torch.__version__, torchvision_version=torchvision.__version__)
    os.makedirs(a.out, exist_ok=True)
    tag = "_raw" if a.raw else ""
    out = os.path.join(a.out, f"results_{a.backbone}_{a.split}{tag}_seed{a.seed}.json")
    json.dump(res, open(out, "w"), indent=1)
    print("test:", {k: res[k] for k in ("accuracy", "macro_f1", "majority_baseline")}, "->", out)


if __name__ == "__main__":
    main()
