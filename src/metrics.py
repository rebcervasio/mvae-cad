"""IoU, Chamfer, linear probe (§15.7).

NEVER report loss as a headline metric -- report IoU (§15.6). With pos_weight in the loss,
a falling BCE is compatible with a completely empty reconstruction.
"""

from __future__ import annotations

import numpy as np
import torch


# --------------------------------------------------------------------------
# reconstruction
# --------------------------------------------------------------------------

def iou(logits: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    """Per-sample IoU of occupied voxels. logits [B,r,r,r] (raw), target same shape."""
    pred = (torch.sigmoid(logits) > thresh)
    tgt = target > 0.5
    dims = tuple(range(1, pred.ndim))
    inter = (pred & tgt).sum(dims).float()
    union = (pred | tgt).sum(dims).float()
    return torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(inter))


def best_threshold_iou(logits: torch.Tensor, target: torch.Tensor,
                       grid=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7)) -> tuple[float, float]:
    """(best_iou, best_thresh). 0.5 is the reported default; this shows how sensitive
    the number is to that choice, which is worth knowing before quoting one figure."""
    best = (-1.0, 0.5)
    for t in grid:
        v = float(iou(logits, target, t).mean())
        if v > best[0]:
            best = (v, t)
    return best


def chamfer_voxel(pred: np.ndarray, target: np.ndarray, pitch: float = 2.0 / 32) -> float:
    """Bidirectional nearest-neighbour distance between occupied voxel CENTRES (§15.7).

    Coarse at 32^3 -- report it, but lead with IoU. Returns world units (object radius = 1).
    """
    from scipy.spatial import cKDTree

    def centres(g):
        i, j, k = np.nonzero(g)
        return np.stack([-1 + (i + .5) * pitch, -1 + (j + .5) * pitch, -1 + (k + .5) * pitch], 1)

    a, b = centres(pred), centres(target)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    da, _ = cKDTree(b).query(a)
    db, _ = cKDTree(a).query(b)
    return float(da.mean() + db.mean())


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(1) == labels).float().mean())


def linear_probe(z_train: np.ndarray, y_train: np.ndarray,
                 z_test: np.ndarray, y_test: np.ndarray, seed: int = 0) -> dict:
    """Q2 (§15.7): freeze the encoder, fit logistic regression on standardized z.

    This is NOT training -- it asks whether a latent that may never have seen a label
    still separates classes linearly.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(z_train)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(sc.transform(z_train), y_train)
    return {
        "probe_acc": float(clf.score(sc.transform(z_test), y_test)),
        "probe_train_acc": float(clf.score(sc.transform(z_train), y_train)),
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
    }


# --------------------------------------------------------------------------
# per-group breakdown -- so one average cannot hide a broken subset
# --------------------------------------------------------------------------

def by_group(values: np.ndarray, groups: np.ndarray, names: list[str] | None = None) -> dict:
    """Mean of `values` per group id. Used to report IoU per class, because a single
    average would hide that thin classes behave differently (see NOTES.md)."""
    out = {}
    for g in np.unique(groups):
        key = names[int(g)] if names is not None else int(g)
        out[key] = float(values[groups == g].mean())
    return out
