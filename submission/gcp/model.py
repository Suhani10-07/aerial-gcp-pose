"""Heatmap network: timm CNN backbone + light FPN decoder to stride 4, optional class head."""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatNet(nn.Module):
    def __init__(self, backbone="resnet34", n_classes=0, pretrained=True, ch=96):
        super().__init__()
        self.enc = timm.create_model(backbone, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        chs = self.enc.feature_info.channels()  # strides 4, 8, 16, 32
        self.lat = nn.ModuleList(nn.Conv2d(c, ch, 1) for c in chs)
        self.smooth = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(ch, 1, 1))
        nn.init.constant_(self.head[-1].bias, -4.6)  # prior p~0.01 stabilises focal loss
        self.cls = nn.Sequential(nn.Dropout(0.3), nn.Linear(chs[-1] + ch, n_classes)) if n_classes else None

    def forward(self, x):
        feats = self.enc(x)
        y = self.lat[-1](feats[-1])
        for f, lat in zip(reversed(feats[:-1]), reversed(self.lat[:-1])):
            y = F.interpolate(y, size=f.shape[-2:], mode="nearest") + lat(f)
        y = self.smooth(y)
        heat = self.head(y)  # stride-4 logits
        if self.cls is None:
            return heat, None
        # global context + heatmap-attended local features (focus on the marker itself)
        att = torch.softmax(heat.flatten(2), -1)
        local = (y.flatten(2) * att).sum(-1)
        glob = feats[-1].mean((2, 3))
        return heat, self.cls(torch.cat([glob, local], 1))
