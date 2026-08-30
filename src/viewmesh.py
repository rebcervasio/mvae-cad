"""Visualisation helper: look at a mesh, and see what voxelizing costs.

Not part of the pipeline -- this exists to answer "what does the .off actually look like,
and how much shape does the 32^3 target throw away?"

    python src/viewmesh.py <mesh-file> [out.png]

Renders, side by side and through the SAME camera:
    original mesh  |  32^3 voxels  |  64^3 voxels  |  128^3 voxels
and prints the quantitative loss at each resolution.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mesh import load_mesh, normalize, check_watertight, voxelize  # noqa: E402
from render import look_at, dodecahedron_cameras  # noqa: E402


def _depth_from_points(pts: np.ndarray, cam: np.ndarray, size: int = 256, splat: int = 1) -> np.ndarray:
    """Orthographic z-buffer of an arbitrary point set. Same convention as render.py."""
    p = torch.as_tensor(pts, dtype=torch.float32) @ torch.as_tensor(look_at(cam), dtype=torch.float32).T
    H = W = size
    u = ((p[:, 0] + 1.0) * 0.5 * W).long()
    v = ((1.0 - p[:, 1]) * 0.5 * H).long()
    d = (p[:, 2] + 1.0) * 0.5
    buf = torch.full((H * W,), 1.0)
    rng = range(-splat, splat + 1)
    for du in rng:
        for dv in rng:
            uu, vv = u + du, v + dv
            ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            buf.scatter_reduce_(0, vv[ok] * W + uu[ok], d[ok], reduce="amin", include_self=True)
    return buf.view(H, W).numpy()


def voxel_centres(grid: np.ndarray) -> np.ndarray:
    """World-space centres of occupied cells, on the [-1,1]^3 lattice."""
    res = grid.shape[0]
    pitch = 2.0 / res
    i, j, k = np.nonzero(grid)
    return np.stack([-1 + (i + 0.5) * pitch, -1 + (j + 0.5) * pitch, -1 + (k + 0.5) * pitch], axis=1)


def surface_voxels(grid: np.ndarray) -> np.ndarray:
    """Occupied cells with at least one empty 6-neighbour. Interior cells are invisible,
    so dropping them makes the render honest and much cheaper."""
    from scipy import ndimage
    er = ndimage.binary_erosion(grid, ndimage.generate_binary_structure(3, 1), border_value=0)
    return grid & ~er


def compare(path: str | Path, out_path: str | Path, resolutions=(32, 64, 128), cam_idx: int = 16) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mesh = normalize(load_mesh(path))
    mesh, rep = check_watertight(mesh)
    cam = dodecahedron_cameras()[cam_idx]

    # reference: the true surface, densely sampled
    ref_pts, _ = trimesh.sample.sample_surface(mesh, 400_000, seed=0)
    ref_depth = _depth_from_points(np.asarray(ref_pts), cam, splat=1)
    ref_sil = ref_depth < 1.0

    panels = [("original mesh", ref_depth, None)]
    stats = []
    for res in resolutions:
        grid, method = voxelize(mesh, res=res)
        pts = voxel_centres(surface_voxels(grid))
        # splat scales with cell size so each resolution renders at its true blockiness
        d = _depth_from_points(pts, cam, splat=max(1, round(256 / res / 2)))
        sil = d < 1.0
        iou = float((sil & ref_sil).sum() / (sil | ref_sil).sum())
        panels.append((f"{res}³ voxels", d, iou))
        stats.append((res, grid, method, iou))

    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.5))
    for ax, (title, d, iou) in zip(axes, panels):
        ax.imshow(1.0 - d, cmap="magma", vmin=0, vmax=1)
        ax.set_title(title + (f"\nsilhouette IoU {iou:.3f}" if iou is not None else "\n(ground truth)"), fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{Path(path).name} — what the voxel target can represent", fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    print(f"[viewmesh] {path}")
    print(f"  watertight={rep.watertight_final}  faces={rep.n_faces}")
    print(f"  {'res':>5} {'cell size':>10} {'~mm on a 100mm part':>21} {'occupied':>10} {'silhouette IoU':>15}")
    for res, grid, method, iou in stats:
        pitch = 2.0 / res
        print(f"  {res:>5} {pitch:>10.4f} {pitch*50:>20.1f}mm {int(grid.sum()):>10} {iou:>15.3f}   ({method})")
    print(f"  wrote {out_path}")
    return out_path


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    src = Path(argv[1])
    out = Path(argv[2]) if len(argv) > 2 else Path(__file__).resolve().parents[1] / "results" / f"voxcompare_{src.stem}.png"
    compare(src, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
