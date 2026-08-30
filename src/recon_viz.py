"""Plot 2 (§0.5, §7): where the reconstruction error lives.

    python src/recon_viz.py results/ckpt_B.pt [--n 4] [--class Flange_Like_Parts]

Per part: original voxels | reconstruction | error map, plus a CROSS-SECTION, because the
whole occlusion argument is about INTERNAL geometry and an exterior render cannot show it.
Error is split into two colours, which is the point of the figure:

    MISSED  (true=1, pred=0)  -- geometry the latent failed to carry
    ADDED   (true=0, pred=1)  -- geometry the decoder invented

A part with an internal bore reconstructing as SOLID shows up as a concentrated blob of
ADDED voxels inside the part. That is the §1 occlusion argument as a measurement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import MVAE  # noqa: E402
from train import load_cache  # noqa: E402
from metrics import iou  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_model(ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c = ck["config"]
    model = MVAE(feat_dim=512, latent_dim=c["latent_dim"], num_classes=len(ck["classes"]),
                 res=ck["res"], aggregator=c["aggregator"],
                 use_cls=c["run"] in ("A", "C"), use_dec=c["run"] in ("B", "C"))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def figure(ckpt_path: Path, n: int = 4, only_class: str | None = None, out: Path | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    model, ck = load_model(ckpt_path)
    cache = load_cache(ck["config"]["dataset"])
    classes = [str(c) for c in cache["classes"]]
    feats = cache["feats"].astype(np.float32)[:, ck["view_idx"], :]
    voxels = cache["voxels"]
    labels = cache["labels"].astype(np.int64)

    idx = np.array(ck["test_idx"])
    if only_class:
        want = classes.index(only_class)
        idx = idx[labels[idx] == want]
        if len(idx) == 0:
            raise SystemExit(f"no test parts in class {only_class}")
    idx = idx[:n]

    with torch.no_grad():
        _, _, logits = model(torch.tensor(feats[idx]))
        ious = iou(logits, torch.tensor(voxels[idx], dtype=torch.float32)).numpy()
        pred = (torch.sigmoid(logits) > 0.5).numpy()
    true = voxels[idx] > 0.5

    rows = len(idx)
    fig, axes = plt.subplots(rows, 4, figsize=(13.2, 3.15 * rows),
                             subplot_kw=None)
    axes = np.atleast_2d(axes)
    res = true.shape[-1]

    for r in range(rows):
        t, p = true[r], pred[r]
        missed, added = t & ~p, p & ~t

        for col, (grid, title, color) in enumerate([
            (t, "original (target)", "#4C9BE8"),
            (p, "reconstruction", "#E8834C"),
        ]):
            ax = fig.add_subplot(rows, 4, r * 4 + col + 1, projection="3d")
            axes[r, col].axis("off")
            if grid.any():
                ax.voxels(grid, facecolors=color, edgecolor="none")
            ax.set_xlim(0, res); ax.set_ylim(0, res); ax.set_zlim(0, res)
            ax.set_box_aspect((1, 1, 1)); ax.view_init(25, 45)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            if r == 0:
                ax.set_title(title, fontsize=10)

        # error map in 3D
        ax = fig.add_subplot(rows, 4, r * 4 + 3, projection="3d")
        axes[r, 2].axis("off")
        both = (t & p) | missed | added
        if both.any():
            colors = np.empty(both.shape, dtype=object)
            colors[t & p] = "#BBBBBB"
            colors[missed] = "#C0392B"
            colors[added] = "#27AE60"
            ax.voxels(both, facecolors=colors, edgecolor="none")
        ax.set_xlim(0, res); ax.set_ylim(0, res); ax.set_zlim(0, res)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(25, 45)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        if r == 0:
            ax.set_title("error: red=MISSED  green=ADDED", fontsize=10)

        # cross-section -- the only place internal geometry is visible
        ax = axes[r, 3]
        k = int(np.argmax(t.sum(axis=(0, 1)))) if t.any() else res // 2
        for i in range(res):
            for j in range(res):
                c = None
                if t[i, j, k] and p[i, j, k]:
                    c = "#BBBBBB"
                elif t[i, j, k]:
                    c = "#C0392B"
                elif p[i, j, k]:
                    c = "#27AE60"
                if c:
                    ax.add_patch(Rectangle((i, j), 1, 1, facecolor=c, edgecolor="white", linewidth=0.2))
        ax.set_xlim(0, res); ax.set_ylim(0, res); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"cross-section  |  IoU {ious[r]:.3f}" if r else
                     f"cross-section (internal)\nIoU {ious[r]:.3f}", fontsize=9)
        ax.set_ylabel(f"{classes[labels[idx[r]]]}\n{cache['ids'][idx[r]]}", fontsize=7)

    fig.suptitle(f"Where the reconstruction error lives — {ckpt_path.stem}", fontsize=12)
    fig.tight_layout()
    out = out or RESULTS / f"plot2_error_{ckpt_path.stem}.png"
    fig.savefig(out, dpi=115)
    plt.close(fig)

    print(f"[recon_viz] {ckpt_path.name}")
    for r in range(rows):
        t, p = true[r], pred[r]
        print(f"  {classes[labels[idx[r]]]:24} IoU {ious[r]:.3f}  "
              f"missed {int((t & ~p).sum()):5}  added {int((p & ~t).sum()):5}  true {int(t.sum()):5}")
    print(f"  wrote {out}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--class", dest="cls", default=None)
    a = ap.parse_args(argv)
    figure(Path(a.ckpt), a.n, a.cls)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
