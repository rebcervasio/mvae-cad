"""Side-by-side of the two voxelization definitions, with the true mesh outline overlaid.

    python src/voxcompare.py <mesh> [out.png]

Top row    : 3D render of mesh / exact / conservative
Bottom row : a CROSS-SECTION with the real mesh boundary drawn on top, which is the only
             view where the difference between the two definitions is unambiguous.

exact        = cell ON if its CENTRE is inside the solid       -> true volume, thin parts vanish
conservative = exact OR the surface passes through the cell    -> nothing vanishes, walls fatten
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mesh import load_mesh, normalize, check_watertight, voxelize, _rasterize_to_lattice, lattice_centres  # noqa: E402
from viewmesh import _depth_from_points, voxel_centres, surface_voxels  # noqa: E402
from render import dodecahedron_cameras  # noqa: E402


def conservative(m: trimesh.Trimesh, res: int = 32) -> np.ndarray:
    """Surface shell UNION exact interior. One layer of dilation, never loses a feature."""
    shell = _rasterize_to_lattice(m.voxelized(pitch=2.0 / res), res)
    inside = np.asarray(m.ray.contains_points(lattice_centres(res)), bool).reshape(res, res, res)
    return shell | inside


def compare(path, out_path, res: int = 32, cam_idx: int = 16, axis: int = 2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    m = normalize(load_mesh(path))
    m, rep = check_watertight(m)
    cam = dodecahedron_cameras()[cam_idx]

    ex, _ = voxelize(m, res=res, method="exact")
    co = conservative(m, res)

    ref_pts, _ = trimesh.sample.sample_surface(m, 400_000, seed=0)
    panels = [("original mesh", _depth_from_points(np.asarray(ref_pts), cam, splat=1)),
              (f"exact — {int(ex.sum())} cells", _depth_from_points(voxel_centres(surface_voxels(ex)), cam, splat=4) if ex.sum() else None),
              (f"conservative — {int(co.sum())} cells", _depth_from_points(voxel_centres(surface_voxels(co)), cam, splat=4))]

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.2))
    for ax, (title, d) in zip(axes[0], panels):
        if d is None:
            ax.text(0.5, 0.5, "EMPTY\n(no cell centre\nlands inside)", ha="center", va="center",
                    fontsize=13, color="#C0392B", weight="bold", transform=ax.transAxes)
            ax.set_facecolor("#1a1a1a")
        else:
            ax.imshow(1.0 - d, cmap="magma", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # ---- cross-section: the view where the difference is unambiguous ----
    pitch = 2.0 / res
    k = res // 2
    zc = -1 + (k + 0.5) * pitch
    try:
        sec = m.section(plane_origin=[0, 0, zc], plane_normal=[0, 0, 1])
        seg = sec.to_2D()[0].discrete if sec is not None else []
    except Exception:
        seg = []

    for ax, (grid, name, color) in zip(axes[1], [(ex, "exact", "#4C9BE8"), (co, "conservative", "#E8834C")]):
        sl = grid[:, :, k]
        for i in range(res):
            for j in range(res):
                if sl[i, j]:
                    ax.add_patch(Rectangle((-1 + i * pitch, -1 + j * pitch), pitch, pitch,
                                           facecolor=color, edgecolor="white", linewidth=0.3))
        for s in seg:
            ax.plot(s[:, 0], s[:, 1], color="#C0392B", linewidth=1.8, zorder=5)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"cross-section: {name} — {int(sl.sum())} cells in this slice", fontsize=10)

    axes[1, 2].axis("off")
    diff = int((co & ~ex).sum())
    axes[1, 2].text(0.02, 0.95,
                    f"{Path(path).name}\n\n"
                    f"watertight: {rep.watertight_final}\n"
                    f"resolution: {res}³   cell = {pitch:.4f}  (~{pitch*50:.1f} mm on a 100 mm part)\n\n"
                    f"exact          {int(ex.sum()):6} cells   ({ex.mean()*100:5.2f}%)\n"
                    f"conservative   {int(co.sum()):6} cells   ({co.mean()*100:5.2f}%)\n"
                    f"added by dilation {diff:6} cells\n\n"
                    f"red outline = the TRUE mesh boundary\n"
                    f"in this cross-section.\n\n"
                    f"exact keeps only cells whose centre\n"
                    f"falls inside that outline.\n"
                    f"conservative also keeps every cell\n"
                    f"the outline passes through.",
                    va="top", ha="left", fontsize=9, family="monospace", transform=axes[1, 2].transAxes)

    fig.suptitle("exact vs conservative voxelization", fontsize=12)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=115)
    plt.close(fig)
    print(f"[voxcompare] {path}")
    print(f"  exact {int(ex.sum())} cells | conservative {int(co.sum())} cells | dilation adds {diff}")
    print(f"  wrote {out_path}")
    return out_path


def main(argv):
    if len(argv) < 2:
        print(__doc__); return 2
    src = Path(argv[1])
    out = Path(argv[2]) if len(argv) > 2 else Path(__file__).resolve().parents[1] / "results" / f"voxdef_{src.stem}.png"
    compare(src, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
