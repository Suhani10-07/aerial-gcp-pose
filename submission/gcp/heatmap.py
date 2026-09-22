"""Heatmap targets, CenterNet focal loss and sub-pixel decoding."""
import numpy as np
import torch
import torch.nn.functional as F


def gaussian_target(h, w, cx, cy, sigma):
    """(h, w) Gaussian peak at heatmap coords (cx, cy); all zeros if cx is None."""
    if cx is None:
        return np.zeros((h, w), np.float32)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)


def focal_loss(logits, target, alpha=2, beta=4):
    """Penalty-reduced pixel focal loss (CenterNet). Robust to extreme fg/bg imbalance of tiny markers."""
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = target.ge(0.99).float()
    pos_loss = -torch.log(p) * (1 - p) ** alpha * pos
    neg_loss = -torch.log(1 - p) * p ** alpha * (1 - target) ** beta * (1 - pos)
    return (pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1)


@torch.no_grad()
def decode(heat, k=1, win=2):
    """heat: (B,1,H,W) probabilities. Top-k local maxima refined by weighted centroid in a
    (2*win+1)^2 window -> sub-pixel coords. Returns (B,k,3) tensor of (x, y, score) in heatmap units."""
    B, _, H, W = heat.shape
    peaks = heat * (F.max_pool2d(heat, 3, 1, 1) == heat)
    scores, idx = peaks.view(B, -1).topk(k)
    out = torch.zeros(B, k, 3)
    for b in range(B):
        for j in range(k):
            y, x = divmod(idx[b, j].item(), W)
            y0, y1, x0, x1 = max(y - win, 0), min(y + win + 1, H), max(x - win, 0), min(x + win + 1, W)
            patch = heat[b, 0, y0:y1, x0:x1].float()
            ys, xs = torch.meshgrid(torch.arange(y0, y1), torch.arange(x0, x1), indexing="ij")
            wsum = patch.sum().clamp(min=1e-8)
            out[b, j] = torch.tensor([(patch * xs).sum() / wsum, (patch * ys).sum() / wsum, scores[b, j]])
    return out
