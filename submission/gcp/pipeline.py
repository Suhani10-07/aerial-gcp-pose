"""Two-stage inference on one full-resolution image, plus metrics."""
import numpy as np
import torch
from sklearn.metrics import f1_score

from .data import crop_padded, to_canvas
from .datasets import Stage2Dataset, hm_to_img, to_tensor
from .heatmap import decode
from .model import HeatNet


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = HeatNet(ck["backbone"], ck["n_classes"], pretrained=False)
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()


@torch.no_grad()
def _heat(model, x, tta):
    """Sigmoid heatmap (and class probs) with optional horizontal-flip TTA."""
    h, logit = model(x)
    h = torch.sigmoid(h.float())
    p = torch.softmax(logit.float(), 1) if logit is not None else None
    if tta:
        h2, l2 = model(x.flip(3))
        h = (h + torch.sigmoid(h2.float()).flip(3)) / 2
        if p is not None:
            p = (p + torch.softmax(l2.float(), 1)) / 2
    return h, p


class Predictor:
    def __init__(self, s1_path, s2_path, device="cuda", topk=3, tta=True, amp=True):
        self.dev = device
        self.s1, self.s2 = load_model(s1_path, device), load_model(s2_path, device)
        self.topk, self.tta, self.amp = topk, tta, amp and device.startswith("cuda")

    def _stage2(self, img, cx, cy):
        S = Stage2Dataset.SIZE
        crop, (x0, y0) = crop_padded(img, cx, cy, S)
        with torch.autocast("cuda", enabled=self.amp):
            h, p = _heat(self.s2, to_tensor(crop)[None].to(self.dev), self.tta)
        hx, hy, score = decode(h.cpu())[0, 0].tolist()
        return x0 + hm_to_img(hx), y0 + hm_to_img(hy), score, p[0].cpu().numpy()

    def __call__(self, img):
        """img: RGB uint8 full-res. Returns x, y, class probs, confidence."""
        canvas, s = to_canvas(img)
        with torch.autocast("cuda", enabled=self.amp):
            h, _ = _heat(self.s1, to_tensor(canvas)[None].to(self.dev), self.tta)
        cands = decode(h.cpu(), k=self.topk)[0]
        best = None
        for hx, hy, s1score in cands.tolist():
            cx, cy = hm_to_img(hx) / s, hm_to_img(hy) / s
            x, y, sc, p = self._stage2(img, cx, cy)
            x, y, sc, p = self._stage2(img, x, y)  # re-centre: marker near crop centre as in training
            conf = sc * s1score ** 0.25
            if best is None or conf > best[3]:
                best = (x, y, p, conf)
        return best


def metrics(preds, gts, thresholds=(10, 25, 50)):
    """preds/gts: dict key -> (x, y, cls). PCK at pixel thresholds + macro F1."""
    keys = [k for k in gts if k in preds]
    d = np.array([np.hypot(preds[k][0] - gts[k][0], preds[k][1] - gts[k][1]) for k in keys])
    out = {f"PCK@{t}": float((d <= t).mean()) for t in thresholds}
    out["median_px"] = float(np.median(d))
    out["macroF1"] = float(f1_score([gts[k][2] for k in keys], [preds[k][2] for k in keys], average="macro"))
    return out
