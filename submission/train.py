"""Train stage 1 (coarse localisation) or stage 2 (fine localisation + shape).

python train.py --stage 1 --train-roots <dir> [<dir2> ...] --out weights/s1.pt
python train.py --stage 2 --train-roots <dir> ... --out weights/s2.pt
Add --full to train on all projects (no hold-out) for the final model.
"""
import argparse
import json
import random
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from gcp.data import build_cache, load_records, split_by_project
from gcp.datasets import Stage1Dataset, Stage2Dataset, hm_to_img
from gcp.heatmap import decode, focal_loss
from gcp.model import HeatNet

# one held-out project per class (shape is constant within a project)
DEFAULT_VAL = "RDCW-Reddipalayam Limestone Mine,Egypt-New city,Deora Limestone Mine"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, choices=[1, 2], required=True)
    p.add_argument("--train-roots", nargs="+", required=True)
    p.add_argument("--cache", default="cache")
    p.add_argument("--out", required=True)
    p.add_argument("--backbone", default="resnet34")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-projects", default=DEFAULT_VAL)
    p.add_argument("--full", action="store_true", help="train on all data, save last epoch")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def seed_worker(wid):
    s = torch.initial_seed() % 2 ** 32
    random.seed(s + wid); np.random.seed((s + wid) % 2 ** 32)


@torch.no_grad()
def evaluate(model, loader, stage, dev):
    """Stage 1: PCK in full-res px (canvas px / scale ~ x3.2). Stage 2: PCK in crop px + macro F1."""
    model.eval()
    errs, ys, ps = [], [], []
    for batch in loader:
        x, gt = batch[0].to(dev), batch[2]
        h, logit = model(x)
        pt = decode(torch.sigmoid(h.float()).cpu())[:, 0, :2]
        errs += torch.linalg.norm(hm_to_img(pt) - gt, dim=1).tolist()
        if stage == 2:
            ys += batch[3].tolist(); ps += logit.argmax(1).cpu().tolist()
    e = np.array(errs) * (3.2 if stage == 1 else 1.0)
    r = {f"PCK@{t}": float((e <= t).mean()) for t in (10, 25, 50)}
    r["median_px"] = float(np.median(e))
    if stage == 2:
        r["macroF1"] = float(f1_score(ys, ps, average="macro"))
        r["score"] = (r["PCK@10"] + r["PCK@25"]) / 2 + r["macroF1"]
    else:
        r["score"] = r["PCK@50"] + r["PCK@25"]
    model.train()
    return r


def main():
    a = get_args()
    seed_all(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = build_cache(load_records(a.train_roots), a.cache, workers=a.workers)
    val_projects = set() if a.full else set(a.val_projects.split(","))
    tr, va = split_by_project(meta, val_projects)
    print(f"train {len(tr)}  val {len(va)}  classes {Counter(m['cls'] for m in tr)}")

    n_cls = 3 if a.stage == 2 else 0
    DS = Stage1Dataset if a.stage == 1 else Stage2Dataset
    bs = a.bs or (8 if a.stage == 1 else 32)
    dl = DataLoader(DS(tr, a.cache, True), bs, shuffle=True, num_workers=a.workers, drop_last=True,
                    worker_init_fn=seed_worker, pin_memory=True, persistent_workers=a.workers > 0)
    vl = DataLoader(DS(va, a.cache, False), bs, num_workers=a.workers) if va else None

    model = HeatNet(a.backbone, n_cls).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(dl), pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")
    # inverse-frequency class weights: macro F1 weights all shapes equally
    cnt = Counter(m["cls"] for m in tr if m["cls"] >= 0)
    cw = torch.tensor([len(tr) / (3 * cnt.get(c, 1)) for c in range(3)], dtype=torch.float32, device=dev)

    best, hist = -1, []
    for ep in range(a.epochs):
        tot = 0.0
        for batch in dl:
            x, hm = batch[0].to(dev, non_blocking=True), batch[1].to(dev, non_blocking=True)
            with torch.autocast("cuda", enabled=dev == "cuda"):
                h, logit = model(x)
            loss = focal_loss(h, hm)
            if a.stage == 2:
                y = batch[3].to(dev)
                if (y >= 0).any():
                    loss = loss + 0.5 * F.cross_entropy(logit.float(), y, weight=cw, ignore_index=-1, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        r = evaluate(model, vl, a.stage, dev) if vl else {}
        hist.append(dict(epoch=ep, loss=tot / len(dl), **r))
        print(json.dumps(hist[-1]))
        ck = dict(state_dict=model.state_dict(), backbone=a.backbone, n_classes=n_cls, epoch=ep, val=r)
        if a.full or r["score"] > best:
            best = r.get("score", best)
            torch.save(ck, a.out)
    json.dump(hist, open(a.out + ".log.json", "w"), indent=1)


if __name__ == "__main__":
    main()
