"""Torch datasets for both stages. Geometric aug is one affine warp applied to image + keypoint."""
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .data import S1_H, S1_W
from .heatmap import gaussian_target

MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255
STRIDE = 4


def to_tensor(img):
    return torch.from_numpy(((img.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1).copy())


def img_to_hm(v):
    """Pixel coordinate -> stride-4 heatmap coordinate (cell centres at 4i+1.5)."""
    return (v - 1.5) / STRIDE


def hm_to_img(v):
    return v * STRIDE + 1.5


def photometric(img):
    """Illumination/sensor variation across sites, seasons and cameras."""
    img = img.astype(np.float32)
    if random.random() < 0.8:
        img = img * random.uniform(0.7, 1.3) + random.uniform(-25, 25)
    if random.random() < 0.5:
        gray = img.mean(2, keepdims=True)
        img = gray + (img - gray) * random.uniform(0.5, 1.4)
    if random.random() < 0.3:
        img = 255 * (np.clip(img, 0, 255) / 255) ** random.uniform(0.7, 1.4)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if random.random() < 0.25:
        img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.3, 1.2))
    if random.random() < 0.2:
        img = np.clip(img + np.random.normal(0, random.uniform(2, 8), img.shape), 0, 255).astype(np.uint8)
    return img


def affine(img, pt, out_wh, scale, angle, flip, dst_pt):
    """Warp so that source point `pt` lands at `dst_pt` in output, with scale/rotation/h-flip."""
    a = math.radians(angle)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) * scale
    if flip:
        R = R @ np.array([[-1, 0], [0, 1]])
    t = np.asarray(dst_pt) - R @ np.asarray(pt)
    M = np.hstack([R, t[:, None]]).astype(np.float32)
    out = cv2.warpAffine(img, M, out_wh, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return out, M


class Stage1Dataset(Dataset):
    """Whole image on a 1280x960 canvas -> coarse marker heatmap."""

    def __init__(self, meta, cache_dir, train, sigma=1.5):
        self.meta, self.dir, self.train, self.sigma = meta, Path(cache_dir), train, sigma

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        m = self.meta[i]
        img = cv2.cvtColor(cv2.imread(str(self.dir / f"s1_{m['idx']}.jpg")), cv2.COLOR_BGR2RGB)
        x, y = m["x"] * m["s1_scale"], m["y"] * m["s1_scale"]
        if self.train:
            c = np.array([S1_W / 2, S1_H / 2])
            ang = random.choice([0, 90, 180, 270]) + random.uniform(-15, 15)
            s = math.exp(random.uniform(math.log(0.75), math.log(1.35)))
            shift = np.array([random.uniform(-200, 200), random.uniform(-150, 150)])
            img, M = affine(img, c, (S1_W, S1_H), s, ang, random.random() < 0.5, c + shift)
            x, y = M @ np.array([x, y, 1.0])
            img = photometric(img)
        inside = 0 <= x < S1_W and 0 <= y < S1_H
        hm = gaussian_target(S1_H // STRIDE, S1_W // STRIDE, img_to_hm(x) if inside else None, img_to_hm(y), self.sigma)
        return to_tensor(img), torch.from_numpy(hm)[None], torch.tensor([x, y], dtype=torch.float32)


class Stage2Dataset(Dataset):
    """384px full-resolution crop around (jittered) marker -> fine heatmap + shape class.
    A fraction of samples are background crops (empty heatmap, class ignored) so the stage-2
    peak score can reject stage-1 false positives at inference."""

    SIZE = 384

    def __init__(self, meta, cache_dir, train, neg_ratio=0.3, jitter=110, sigma=2.0):
        self.meta, self.dir, self.train = meta, Path(cache_dir), train
        self.neg_ratio, self.jitter, self.sigma = (neg_ratio if train else 0.0), jitter, sigma
        self.n_pos = len(meta)

    def __len__(self):
        return self.n_pos + int(self.n_pos * self.neg_ratio)

    def __getitem__(self, i):
        S, hw = self.SIZE, self.SIZE // STRIDE
        if i >= self.n_pos:  # background crop
            m = random.choice(self.meta)
            img = cv2.cvtColor(cv2.imread(str(self.dir / f"s2n_{m['idx']}_{random.randrange(m['n_neg'])}.jpg")), cv2.COLOR_BGR2RGB)
            img = photometric(np.ascontiguousarray(np.rot90(img, random.randrange(4))))
            return to_tensor(img), torch.zeros(1, hw, hw), torch.tensor([-1.0, -1.0]), torch.tensor(-1)
        m = self.meta[i]
        img = cv2.cvtColor(cv2.imread(str(self.dir / f"s2p_{m['idx']}.jpg")), cv2.COLOR_BGR2RGB)
        pt = (m["s2_x"], m["s2_y"])
        if self.train:
            s = math.exp(random.uniform(math.log(0.7), math.log(1.45)))  # altitude / GSD variation
            j = self.jitter if random.random() < 0.85 else 2 * self.jitter  # simulate stage-1 error
            dst = (S / 2 + random.uniform(-j, j), S / 2 + random.uniform(-j, j))
            dst = tuple(np.clip(dst, 12, S - 12))
            img, _ = affine(img, pt, (S, S), s, random.uniform(0, 360), random.random() < 0.5, dst)
            img = photometric(img)
        else:  # deterministic moderate offset, mimics inference
            rng = np.random.default_rng(i)
            dst = (S / 2 + rng.uniform(-40, 40), S / 2 + rng.uniform(-40, 40))
            img, _ = affine(img, pt, (S, S), 1.0, 0, False, dst)
        x, y = dst
        hm = gaussian_target(hw, hw, img_to_hm(x), img_to_hm(y), self.sigma)
        return to_tensor(img), torch.from_numpy(hm)[None], torch.tensor([x, y], dtype=torch.float32), torch.tensor(m["cls"])
