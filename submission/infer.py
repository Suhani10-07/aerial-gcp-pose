"""Run the two-stage model on a folder of images and write predictions.json.

python infer.py --images <test_dataset dir> --s1 weights/s1.pt --s2 weights/s2.pt --out predictions.json
Evaluate on held-out training projects instead:
python infer.py --eval --train-roots <dir> ... --s1 ... --s2 ...
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from gcp.data import CLASSES, load_records, read_image, scan_images
from gcp.pipeline import Predictor, metrics
from train import DEFAULT_VAL


def run(pred, items, gcp_vote):
    """items: list of (key, path). Returns key -> (x, y, cls_idx)."""
    raw = {}
    for k, p in tqdm(items):
        x, y, prob, _ = pred(read_image(p))
        raw[k] = (x, y, prob)
    if gcp_vote:  # all images in a gcp folder show the same physical marker -> share shape
        groups = defaultdict(list)
        for k in raw:
            groups[k.rsplit("/", 1)[0]].append(k)
        mean_p = {g: np.mean([raw[k][2] for k in ks], 0) for g, ks in groups.items()}
        return {k: (x, y, int(mean_p[k.rsplit("/", 1)[0]].argmax())) for k, (x, y, _) in raw.items()}
    return {k: (x, y, int(p.argmax())) for k, (x, y, p) in raw.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", help="test_dataset root")
    ap.add_argument("--s1", required=True)
    ap.add_argument("--s2", required=True)
    ap.add_argument("--out", default="predictions.json")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--no-gcp-vote", action="store_true")
    ap.add_argument("--l-name", default="L-Shape", help="label string for the L class (train json uses 'L-Shape')")
    ap.add_argument("--no-gcp-prior", action="store_true", help="ignore shapes of GCPs already labelled in train")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--train-roots", nargs="+")
    ap.add_argument("--val-projects", default=DEFAULT_VAL)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pred = Predictor(a.s1, a.s2, dev, tta=not a.no_tta)
    names = CLASSES[:2] + [a.l_name]

    if a.eval:
        vp = set(a.val_projects.split(","))
        recs = [r for r in load_records(a.train_roots) if r["project"] in vp]
        res = run(pred, [(r["key"], r["path"]) for r in recs], not a.no_gcp_vote)
        print(json.dumps(metrics(res, {r["key"]: (r["x"], r["y"], r["cls"]) for r in recs}), indent=1))
        return

    items = sorted(scan_images(a.images).items())
    res = run(pred, items, not a.no_gcp_vote)
    if a.train_roots and not a.no_gcp_prior:
        # same survey/gcp folder as a labelled training image -> same physical marker -> known shape
        known = {r["gcp"]: r["cls"] for r in load_records(a.train_roots) if r["cls"] >= 0}
        hits = [k for k in res if k.rsplit("/", 1)[0] in known]
        for k in hits:
            res[k] = (*res[k][:2], known[k.rsplit("/", 1)[0]])
        print(f"gcp prior applied to {len(hits)} images")
    out = {k: {"mark": {"x": round(float(x), 2), "y": round(float(y), 2)}, "verified_shape": names[c]}
           for k, (x, y, c) in res.items()}
    Path(a.out).write_text(json.dumps(out, indent=4))
    print(f"wrote {len(out)} predictions -> {a.out}")


if __name__ == "__main__":
    main()
