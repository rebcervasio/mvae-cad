"""Verify what dilation actually does to thin parts.

    python src/thincheck.py                 # synthetic validation + CADNET gallery
    python src/thincheck.py <class-name>    # gallery for one CADNET class

The question is not "does conservative fatten things" -- it does, by construction. It is
whether a sub-cell wall becomes ONE cell thick (the thinnest a voxel grid can represent, so
unavoidable) or TWO-PLUS (genuinely over-fat, and a real distortion of the geometry).

Thickness is measured with a Euclidean distance transform: for the occupied region,
2 * max(EDT) is the local thickness in cells. A 1-cell plate gives EDT max 0.5.
"""

from __future__ import annotations

import glob
import io
import contextlib
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mesh import load_mesh, normalize, check_watertight, voxelize  # noqa: E402
from viewmesh import _depth_from_points, voxel_centres, surface_voxels  # noqa: E402
from voxcompare import conservative  # noqa: E402
from render import dodecahedron_cameras  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
QUIET = io.StringIO()


def _runs_along(grid: np.ndarray, axis: int) -> np.ndarray:
    """For every occupied cell, the length of the contiguous occupied run it belongs to
    along `axis`. Zero where empty."""
    g = np.moveaxis(grid, axis, -1)
    out = np.zeros(g.shape, dtype=np.int32)
    for idx in np.ndindex(g.shape[:-1]):
        line = g[idx]
        if not line.any():
            continue
        # label contiguous runs, then write each run's length back over its cells
        lab, n = ndimage.label(line)
        if n:
            sizes = ndimage.sum_labels(line, lab, index=np.arange(1, n + 1)).astype(np.int32)
            out[idx][line] = sizes[lab[line] - 1]
    return np.moveaxis(out, -1, axis)


def wall_thickness_cells(grid: np.ndarray) -> tuple[float, float]:
    """(median, max) LOCAL WALL THICKNESS in cells.

    For each occupied cell take the shortest occupied run through it along x, y or z --
    for plate-like geometry that is the wall thickness. Reported as the median over cells.

    NOTE: an earlier version used 2*max(distance_transform_edt), which cannot tell a
    1-cell plate from a 2-cell plate (every cell of both touches empty space, so EDT=1
    for both). Run length is unambiguous.
    """
    if grid.sum() == 0:
        return 0.0, 0.0
    runs = np.stack([_runs_along(grid, a) for a in (0, 1, 2)])
    runs[:, ~grid] = 10_000                      # ignore empty cells in the min
    thick = runs.min(axis=0)[grid]
    return float(np.median(thick)), float(thick.max())


def synthetic_validation() -> None:
    print("=== SYNTHETIC: a plate of known thickness -> how many cells does it become? ===")
    print("   (pitch = 0.0625; 'true cells' = actual thickness / pitch)")
    print()
    print(f"  {'true thickness':>15} {'true cells':>11} | {'exact':>16} | {'conservative':>18}")
    print("  " + "-" * 70)
    for t in [0.30, 0.20, 0.12, 0.08, 0.05, 0.03, 0.015]:
        box = trimesh.creation.box(extents=(1.6, 1.6, t))
        s = float(np.linalg.norm(box.vertices - box.bounding_box.centroid, axis=1).max())
        mm = normalize(box)
        true_cells = (t / s) / (2.0 / 32)
        row = f"  {t/s:>15.4f} {true_cells:>11.2f} |"
        for meth in ("exact", "conservative"):
            with contextlib.redirect_stdout(QUIET):
                g = conservative(mm) if meth == "conservative" else voxelize(mm, method="exact")[0]
            if g.sum() == 0:
                cell = "EMPTY"
            else:
                med, _ = wall_thickness_cells(g)
                cell = f"{med:.1f} cells"
            row += f" {cell:>16} |" if meth == "exact" else f" {cell:>18} |"
        print(row)
    print()
    print("  A voxel grid CANNOT represent anything thinner than 1 cell.")
    print("  So 1.0 = minimal/unavoidable. 2.0+ = genuine over-dilation.")


def gallery(classes: list[str], per_class: int = 3, out_name: str = "thin_gallery.png") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cam = dodecahedron_cameras()[16]
    rows = []
    for cls in classes:
        for fp in sorted(glob.glob(str(ROOT / "data" / "CADNET_3317" / cls / "*.stl")))[:per_class]:
            try:
                with contextlib.redirect_stdout(QUIET):
                    m = normalize(load_mesh(fp))
                    m, _ = check_watertight(m)
                    ex, _ = voxelize(m, method="exact")
                    co = conservative(m)
                pts, _ = trimesh.sample.sample_surface(m, 300_000, seed=0)
                rows.append({
                    "cls": cls, "id": Path(fp).stem, "mesh": m,
                    "ref": _depth_from_points(np.asarray(pts), cam, splat=1),
                    "ex": ex, "co": co,
                    "co_img": _depth_from_points(voxel_centres(surface_voxels(co)), cam, splat=4),
                })
            except Exception as e:
                print(f"  skip {fp}: {e}")

    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(13.5, 3.05 * n))
    axes = np.atleast_2d(axes)
    pitch = 2.0 / 32
    for r, d in enumerate(rows):
        axes[r, 0].imshow(1.0 - d["ref"], cmap="magma", vmin=0, vmax=1)
        axes[r, 0].set_ylabel(f"{d['cls']}\n{d['id']}", fontsize=7)
        axes[r, 0].set_title("mesh" if r == 0 else "", fontsize=9)

        axes[r, 1].imshow(1.0 - d["co_img"], cmap="magma", vmin=0, vmax=1)
        axes[r, 1].set_title("conservative voxels" if r == 0 else "", fontsize=9)

        # cross-section through the densest slice, with the true outline
        co = d["co"]
        k = int(np.argmax(co.sum(axis=(0, 1))))
        ax = axes[r, 2]
        sl = co[:, :, k]
        for i in range(32):
            for j in range(32):
                if sl[i, j]:
                    ax.add_patch(Rectangle((-1 + i * pitch, -1 + j * pitch), pitch, pitch,
                                           facecolor="#E8834C", edgecolor="white", linewidth=0.25))
        try:
            zc = -1 + (k + 0.5) * pitch
            sec = d["mesh"].section(plane_origin=[0, 0, zc], plane_normal=[0, 0, 1])
            for s in (sec.to_2D()[0].discrete if sec is not None else []):
                ax.plot(s[:, 0], s[:, 1], color="#C0392B", linewidth=1.6, zorder=5)
        except Exception:
            pass
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
        ax.set_title("cross-section + true outline" if r == 0 else "", fontsize=9)

        med_c, max_c = wall_thickness_cells(co)
        med_e, _ = wall_thickness_cells(d["ex"])
        ax = axes[r, 3]; ax.axis("off")
        ax.text(0.0, 0.92,
                f"exact        {int(d['ex'].sum()):5} cells\n"
                f"conservative {int(d['co'].sum()):5} cells\n\n"
                f"wall thickness (conservative)\n"
                f"  median {med_c:.1f} cells\n"
                f"  max    {max_c:.1f} cells\n\n"
                f"wall thickness (exact)\n"
                f"  median {med_e:.1f} cells\n\n"
                + ("EXACT LOSES THIS PART\n" if d['ex'].sum() == 0 else "")
                + ("minimal dilation\n(1 cell = thinnest possible)" if med_c <= 1.6
                   else "over-dilated (>1.5 cells)"),
                va="top", ha="left", fontsize=8, family="monospace", transform=ax.transAxes,
                color="#C0392B" if d['ex'].sum() == 0 else "black")

        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        for c in (0, 1):
            axes[r, c].axis("off")

    fig.suptitle("Thin parts: what conservative voxelization actually produces", fontsize=12)
    fig.tight_layout()
    out = ROOT / "results" / out_name
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"\n  wrote {out}")
    return out


def main(argv):
    synthetic_validation()
    print()
    classes = [argv[1]] if len(argv) > 1 else ["Thin_Plates", "Slender_Thin_Plates", "Clips"]
    print(f"=== REAL CADNET PARTS: {', '.join(classes)} ===")
    gallery(classes, per_class=2 if len(classes) > 1 else 4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
