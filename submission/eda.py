"""Exploratory data analysis: integrity checks on labels and images.
Usage: python eda.py --train <train_dataset dir> --test <test_dataset dir>
"""
import argparse, json, hashlib
from collections import Counter
from pathlib import Path
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--train", required=True)
p.add_argument("--test", required=True)
p.add_argument("--labels", default=None)
a = p.parse_args()
train, test = Path(a.train), Path(a.test)
lbl_path = Path(a.labels) if a.labels else next(train.rglob("*gcp_marks*.json"))
labels = json.load(open(lbl_path))
print("labels:", len(labels), "file:", lbl_path.name)

from gcp.data import scan_images as scan

tr, te = scan(train), scan(test)
print("train imgs:", len(tr), "test imgs:", len(te))
print("labels w/o image:", len(set(labels) - set(tr)), list(set(labels) - set(tr))[:3])
print("images w/o label:", len(set(tr) - set(labels)), list(set(tr) - set(labels))[:3])

# label schema / values
keys, shapes, bad, sizes, exif, oob = Counter(), Counter(), [], Counter(), Counter(), []
for k, v in labels.items():
    keys[tuple(sorted(v.keys())) if isinstance(v, dict) else type(v).__name__] += 1
    shapes[repr(v.get("verified_shape")) if isinstance(v, dict) else None] += 1
    m = v.get("mark") if isinstance(v, dict) else None
    try:
        x, y = float(m["x"]), float(m["y"])
    except Exception:
        bad.append((k, m)); continue
    if k in tr:
        im = Image.open(tr[k]); sizes[im.size] += 1
        exif[im.getexif().get(274)] += 1
        if not (0 <= x < im.size[0] and 0 <= y < im.size[1]): oob.append((k, x, y, im.size))
print("label key schemas:", dict(keys))
print("shape values:", dict(shapes))
print("unparseable marks:", len(bad), bad[:3])
print("train sizes:", dict(sizes)); print("train EXIF orientation:", dict(exif))
print("out-of-bounds marks:", len(oob), oob[:5])

xs = [float(v["mark"]["x"]) for v in labels.values() if isinstance(v, dict) and v.get("mark")]
ys = [float(v["mark"]["y"]) for v in labels.values() if isinstance(v, dict) and v.get("mark")]
if xs:
    import numpy as np
    for n, arr in (("x", xs), ("y", ys)):
        print(n, "min/p5/med/p95/max:", np.percentile(arr, [0, 5, 50, 95, 100]).round(1))

# hierarchy + duplicates
proj = Counter(k.split("/")[0] for k in labels); print("projects:", len(proj), "gcps:", len({k.rsplit('/', 1)[0] for k in labels}))
print("depths:", Counter(len(k.split("/")) for k in labels))
print("test sizes:", Counter(Image.open(f).size for f in list(te.values())))
print("test EXIF:", Counter(Image.open(f).getexif().get(274) for f in te.values()))
print("test projects overlap w/ train:", len({k.split('/')[0] for k in te} & set(proj)), "/", len({k.split('/')[0] for k in te}))
h = Counter(hashlib.md5(f.read_bytes()).hexdigest() for f in tr.values())
print("duplicate train files:", sum(c - 1 for c in h.values() if c > 1))
# shape per project (label noise check: same gcp with different shapes)
g = {}
for k, v in labels.items():
    if isinstance(v, dict): g.setdefault(k.rsplit("/", 1)[0], set()).add(v.get("verified_shape"))
print("gcps with conflicting shapes:", sum(len(s) > 1 for s in g.values()))
print("shape by project:", {pj: dict(Counter(labels[k].get("verified_shape") for k in labels if k.startswith(pj + "/"))) for pj in proj})
