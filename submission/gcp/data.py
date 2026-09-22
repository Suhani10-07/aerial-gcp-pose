"""Dataset indexing, label cleaning, splits and on-disk cache."""
import json, re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png"}
# multi-part download wrappers, e.g. "<name>-<timestamp>Z-1-001/"
_ZIP_WRAP = re.compile(r".*-\d{8}T\d{6}Z-\d+-\d{3}$")
CLASSES = ["Cross", "Square", "L-Shape"]
_ALIASES = {"cross": 0, "square": 1, "l-shape": 2, "l-shaped": 2, "l shape": 2}

# Stage-1 canvas: long side 1280, fixed 1280x960 (4:3 covers 4096x3068/2730 and 4000x3000)
S1_W, S1_H = 1280, 960
# Stage-2 cache: full-res patch around GT, big enough for jitter + scale augmentation
S2_CACHE = 768


def canon_key(rel: str) -> str:
    """Drop Drive zip wrapper folders (and a nested train_dataset/) so paths match label keys."""
    parts = [p for p in rel.split("/") if not _ZIP_WRAP.match(p)]
    # keys start below the dataset folder, whatever mount prefix precedes it
    anchors = [i for i, p in enumerate(parts) if p in ("train_dataset", "test_dataset")]
    if anchors:
        parts = parts[anchors[-1] + 1:]
    return "/".join(parts)


def scan_images(roots, exclude=None):
    """Index images under one or more roots (multi-part downloads) by canonical key.
    `exclude`: skip files with this folder name in their path (keeps test out of training)."""
    out = {}
    for root in [roots] if isinstance(roots, (str, Path)) else roots:
        root = Path(root)
        for f in root.rglob("*"):
            if f.suffix.lower() in IMG_EXT and exclude not in f.parts:
                out[canon_key(f.relative_to(root).as_posix())] = f
    return out


def norm_shape(s):
    return _ALIASES.get(str(s).strip().lower()) if s else None


def load_records(train_roots, label_file=None):
    """List of dicts {key, path, x, y, cls, project, gcp}. Skips labels without image;
    imputes missing shape from other images of the same GCP (a marker has one shape)."""
    roots = [Path(r) for r in ([train_roots] if isinstance(train_roots, (str, Path)) else train_roots)]
    label_file = Path(label_file) if label_file else next(f for r in roots for f in r.rglob("*gcp_marks*.json"))
    labels, imgs = json.load(open(label_file)), scan_images(roots, exclude="test_dataset")
    recs = []
    for k, v in labels.items():
        k2 = canon_key(k)
        if k2 not in imgs or not isinstance(v, dict) or "mark" not in v:
            continue
        c = norm_shape(v.get("verified_shape"))
        recs.append(dict(key=k, path=str(imgs[k2]), x=float(v["mark"]["x"]), y=float(v["mark"]["y"]),
                         cls=-1 if c is None else c, project=k2.split("/")[0], gcp=k2.rsplit("/", 1)[0]))
    if not recs:
        raise RuntimeError(f"no label keys matched images under {roots}; e.g. label '{next(iter(labels))}' "
                           f"vs image '{next(iter(imgs), None)}'")
    votes = {}
    for r in recs:
        if r["cls"] >= 0:
            votes.setdefault(r["gcp"], Counter())[r["cls"]] += 1
    for r in recs:
        if r["cls"] < 0 and r["gcp"] in votes:
            r["cls"] = votes[r["gcp"]].most_common(1)[0][0]
    return recs


def split_by_project(recs, val_projects):
    """Hold out whole projects: shape is constant per project, so a random split would leak."""
    tr = [r for r in recs if r["project"] not in val_projects]
    va = [r for r in recs if r["project"] in val_projects]
    return tr, va


def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"unreadable image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_canvas(img):
    """Resize to stage-1 canvas (top-left aligned, zero pad). Returns canvas, scale."""
    h, w = img.shape[:2]
    s = min(S1_W / w, S1_H / h)
    small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((S1_H, S1_W, 3), np.uint8)
    canvas[:small.shape[0], :small.shape[1]] = small
    return canvas, s


def crop_padded(img, cx, cy, size):
    """size x size crop centred at (cx, cy), zero padded. Returns crop, (x0, y0) of crop origin."""
    x0, y0 = int(round(cx)) - size // 2, int(round(cy)) - size // 2
    h, w = img.shape[:2]
    out = np.zeros((size, size, 3), img.dtype)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + size, w), min(y0 + size, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out, (x0, y0)


def _cache_one(args):
    i, r, out, n_neg, seed = args
    rng = np.random.default_rng(seed + i)
    img = read_image(r["path"])
    h, w = img.shape[:2]
    canvas, s = to_canvas(img)
    cv2.imwrite(str(out / f"s1_{i}.jpg"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    pos, (x0, y0) = crop_padded(img, r["x"], r["y"], S2_CACHE)
    cv2.imwrite(str(out / f"s2p_{i}.jpg"), cv2.cvtColor(pos, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    for j in range(n_neg):  # background patches far from the marker
        while True:
            nx, ny = rng.uniform(0, w), rng.uniform(0, h)
            if np.hypot(nx - r["x"], ny - r["y"]) > 400:
                break
        neg, _ = crop_padded(img, nx, ny, 384)
        cv2.imwrite(str(out / f"s2n_{i}_{j}.jpg"), cv2.cvtColor(neg, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return dict(r, idx=i, s1_scale=s, s2_x=r["x"] - x0, s2_y=r["y"] - y0, n_neg=n_neg)


def build_cache(recs, out_dir, n_neg=2, workers=4, seed=0):
    """Decode each 12MP JPEG once: stage-1 canvas + stage-2 positive/negative patches."""
    from multiprocessing import Pool
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    meta_f = out / "meta.json"
    if meta_f.exists():
        meta = json.load(open(meta_f))
        if [m["key"] for m in meta] == [r["key"] for r in recs]:
            return meta
    jobs = [(i, r, out, n_neg, seed) for i, r in enumerate(recs)]
    if workers <= 1:
        meta = [_cache_one(j) for j in jobs]
    else:
        with Pool(workers) as p:
            meta = p.map(_cache_one, jobs, chunksize=8)
    json.dump(meta, open(meta_f, "w"))
    return meta
