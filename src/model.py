"""Encoder, aggregators, heads (§15.4, §2).

    cached 512-d per-view feats  ->  aggregate  ->  z in R^256
                                                     |-> Linear -> class logits   (run A)
                                                     `-> ConvTranspose3d x3 -> 32^3 (run B)

BOTH HEADS READ ONLY `z` (§3.3). Neither sees images, neither sees the mesh. The backbone
is frozen and its features cached (§3.4), so the trainable part is only aggregator + heads.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# aggregation over the view dimension (§15.4)
# --------------------------------------------------------------------------

class MaxPool(nn.Module):
    """MVCNN: elementwise max over views. No parameters -- the control arm."""

    def forward(self, f: torch.Tensor) -> torch.Tensor:   # [B, V, D] -> [B, D]
        return f.max(dim=1).values


class AttnPool(nn.Module):
    """GMViT-lite: one learnable query, softmax over views, weighted sum (~5 lines).

    This is the only SHARED trainable component when the backbone is frozen, which is why
    §3.3 flags Phase 2 as a weakened version of the joint-training test.
    """

    def __init__(self, dim: int = 512):
        super().__init__()
        self.q = nn.Parameter(torch.randn(dim) * dim ** -0.5)

    def forward(self, f: torch.Tensor) -> torch.Tensor:   # [B, V, D] -> [B, D]
        w = torch.softmax((f @ self.q) / math.sqrt(f.shape[-1]), dim=1)   # [B, V]
        return (w.unsqueeze(-1) * f).sum(dim=1)

    def weights(self, f: torch.Tensor) -> torch.Tensor:
        """Per-view attention weights -- useful for "which views mattered" in the meeting."""
        return torch.softmax((f @ self.q) / math.sqrt(f.shape[-1]), dim=1)


def build_aggregator(kind: str, dim: int = 512) -> nn.Module:
    if kind == "max":
        return MaxPool()
    if kind == "attn":
        return AttnPool(dim)
    raise ValueError(f"unknown aggregator {kind!r}")


# --------------------------------------------------------------------------
# heads
# --------------------------------------------------------------------------

class ClassifierHead(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, num_classes)

    def forward(self, z):
        return self.fc(z)


class VoxelDecoder(nn.Module):
    """z -> res^3 occupancy LOGITS (no sigmoid; BCEWithLogits handles it, §15.6).

    Kept behind this interface so the occupancy/SDF MLP stays a one-class swap (§3.2).
    """

    def __init__(self, latent_dim: int = 256, res: int = 32, base: int = 256):
        super().__init__()
        if res not in (32, 64):
            raise ValueError(f"unsupported voxel res {res}")
        self.res = res
        self.base = base
        self.fc = nn.Linear(latent_dim, base * 4 * 4 * 4)
        chans = [base, 128, 64] + ([32] if res == 64 else [])
        layers = []
        for i in range(len(chans) - 1):
            layers += [nn.ConvTranspose3d(chans[i], chans[i + 1], 4, 2, 1),
                       nn.BatchNorm3d(chans[i + 1]), nn.ReLU(inplace=True)]
        layers += [nn.ConvTranspose3d(chans[-1], 1, 4, 2, 1)]     # -> res^3 logits
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        x = self.fc(z).view(-1, self.base, 4, 4, 4)
        return self.net(x).squeeze(1)                              # [B, res, res, res]


# --------------------------------------------------------------------------
# the whole thing
# --------------------------------------------------------------------------

class MVAE(nn.Module):
    """Multi-view encoder + optional heads. Runs A / B / C differ only in which heads
    are built and which losses are used -- the architecture is identical (§3.3)."""

    def __init__(self, feat_dim=512, latent_dim=256, num_classes=43, res=32,
                 aggregator="max", use_cls=True, use_dec=True):
        super().__init__()
        self.aggregator = build_aggregator(aggregator, feat_dim)
        self.to_latent = nn.Linear(feat_dim, latent_dim)
        self.cls_head = ClassifierHead(latent_dim, num_classes) if use_cls else None
        self.decoder = VoxelDecoder(latent_dim, res) if use_dec else None

    def encode(self, feats: torch.Tensor) -> torch.Tensor:
        """[B, V, 512] cached features -> z [B, latent]."""
        return self.to_latent(self.aggregator(feats))

    def forward(self, feats):
        z = self.encode(feats)
        logits = self.cls_head(z) if self.cls_head is not None else None
        voxels = self.decoder(z) if self.decoder is not None else None
        return z, logits, voxels

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
