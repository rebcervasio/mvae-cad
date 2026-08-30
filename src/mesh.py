"""Mesh loading, normalization, watertight gate, voxelization (§15.1, §15.3).

THE normalization lives here and nowhere else (§15.1 / §16 trap 2). Every consumer --
render.py, cache.py, demo.py -- imports `normalize` from this module. Never inline it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import trimesh

VOXEL_RES = 32


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load any mesh file to a single Trimesh. Scenes are concatenated, loudly."""
    obj = trimesh.load(str(path), force="mesh", process=True)
    if isinstance(obj, trimesh.Scene):
        parts = list(obj.geometry.values())
        print(f"  [load] scene with {len(parts)} geometries -> concatenated")
        obj = trimesh.util.concatenate(parts)
    if not isinstance(obj, trimesh.Trimesh):
        raise TypeError(f"{path}: loaded {type(obj).__name__}, not a Trimesh")
    if len(obj.faces) == 0:
        raise ValueError(f"{path}: mesh has no faces")
    return obj


# --------------------------------------------------------------------------
# §15.1 normalization -- THE ONE PLACE
# --------------------------------------------------------------------------

def normalize(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Center on the bounding-box centroid, scale so the furthest vertex sits at r=1.

    Isotropic on purpose: NOT per-axis bounding-box normalization, which would destroy
    aspect ratio -- a real classification cue (§15.1). Result lies inside the unit
    sphere, hence inside the [-1,1]^3 cube that both the renderer and the voxelizer use.
    """
    mesh = mesh.copy()
    center = mesh.bounding_box.centroid
    verts = mesh.vertices - center
    scale = float(np.linalg.norm(verts, axis=1).max())
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError(f"degenerate mesh: normalization scale = {scale}")
    mesh.vertices = verts / scale
    return mesh


# --------------------------------------------------------------------------
# §5 watertight gate
# --------------------------------------------------------------------------

@dataclass
class WatertightReport:
    watertight_raw: bool      # as loaded
    watertight_final: bool    # after repair attempt
    repaired: bool            # repair was attempted AND changed the verdict
    n_faces: int
    volume: float             # signed volume; negative means inverted normals

    def to_dict(self) -> dict:
        return asdict(self)


def check_watertight(mesh: trimesh.Trimesh, repair: bool = True) -> tuple[trimesh.Trimesh, WatertightReport]:
    """Gate on watertightness, attempting repair. Never silent -- the caller logs the rate.

    `.fill()` voxelization needs a closed surface, and IoU is meaningless without a
    well-defined volume (§5). The pass RATE across a dataset is itself a reportable
    finding, so this returns the verdict rather than raising.
    """
    raw = bool(mesh.is_watertight)
    out = mesh
    if not raw and repair:
        out = mesh.copy()
        out.merge_vertices()
        out.update_faces(out.nondegenerate_faces())
        out.update_faces(out.unique_faces())
        out.remove_unreferenced_vertices()
        trimesh.repair.fill_holes(out)
        trimesh.repair.fix_winding(out)
        trimesh.repair.fix_inversion(out)
        trimesh.repair.fix_normals(out)

    final = bool(out.is_watertight)
    return out, WatertightReport(
        watertight_raw=raw,
        watertight_final=final,
        repaired=(not raw and final),
        n_faces=int(len(out.faces)),
        volume=float(out.volume) if final else float("nan"),
    )


# --------------------------------------------------------------------------
# §15.3 voxelization
# --------------------------------------------------------------------------

def _rasterize_to_lattice(vg, res: int) -> np.ndarray:
    """Map a trimesh VoxelGrid's occupied cell CENTRES into our fixed lattice.

    DEVIATION FROM §15.3, recorded in NOTES.md: the spec says to pad/centre-crop
    trimesh's returned matrix to res^3. We instead map world coordinates into our own
    fixed lattice. Same fixed shape, but alignment is derived from world coordinates
    rather than assumed to be centred -- trimesh anchors its grid on the mesh bounds,
    not on the origin, so centre-cropping would silently shift the target relative to
    the rendered views. That is exactly the §16 trap 2 normalization drift, and it
    would be invisible in the loss.
    """
    pitch = 2.0 / res
    grid = np.zeros((res, res, res), dtype=bool)
    pts = np.asarray(vg.points, dtype=np.float64)
    if pts.size == 0:
        return grid
    idx = np.floor((pts + 1.0) / pitch).astype(np.int64)
    n_out = int(((idx < 0) | (idx >= res)).any(axis=1).sum())
    if n_out:
        # Expected to be tiny: cell centres can sit a half-pitch past the surface.
        print(f"  [voxelize] {n_out}/{len(idx)} cells clipped to the lattice bounds")
    idx = np.clip(idx, 0, res - 1)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return grid


def lattice_centres(res: int = VOXEL_RES) -> np.ndarray:
    """The res^3 cell centres of our fixed lattice over [-1,1]^3, in world coords.

    Cell i is centred at -1 + (i+0.5)*pitch, so the lattice is symmetric about the origin.
    This is THE lattice: voxel targets, decoder output and every metric live on it.
    """
    pitch = 2.0 / res
    c = np.arange(res) * pitch - 1.0 + pitch / 2.0
    return np.stack(np.meshgrid(c, c, c, indexing="ij"), axis=-1).reshape(-1, 3)


def voxelize(mesh: trimesh.Trimesh, res: int = VOXEL_RES, method: str = "auto") -> tuple[np.ndarray, str]:
    """Solid occupancy on a FIXED res^3 lattice spanning the normalized [-1,1]^3 cube.

    Returns (grid, method_used). Input MUST already be normalized.

    Four routes. "auto" picks `conservative` when watertight and `morphological`
    otherwise. The method used is RETURNED, never hidden -- §18 forbids silent fallbacks,
    so callers log it and the cache records it per part.

      "conservative"  -- surface shell UNION exact interior. A cell is ON if its centre
                         is inside the solid OR the surface passes through it.
                         **Primary route.** Definitionally equivalent to binvox `-e`
                         unioned with parity/ray-stabbing solid fill, which is what
                         3D-R2N2 / Pix2Vox / ShapeNet use. Measured: adds at most one
                         cell per face, never loses a feature (0% empty grids on CADNET
                         vs 7.5% for `exact`), and preserves bores identically to exact.
      "exact"         -- point-in-solid test at each lattice centre: "is this cell's
                         centre inside the part?". True volume (+0.1% vs analytic), but
                         **deletes any feature thinner than one cell** -- on CADNET that
                         empties 7.5% of parts and 80-100% of Thin_Plates. Kept for the
                         thin-wall diagnostic and for volume checks, NOT for targets.
      "surface_fill"  -- trimesh `.voxelized().fill()`, the literal §15.3 route.
                         DILATES: it marks every cell the surface passes through, so a
                         solid comes out ~1 cell too fat (measured: toilet 7.1% vs 4.2%
                         true occupancy, IoU 0.59). Kept for comparison, not for targets.
      "morphological" -- surface-voxelize then `scipy.ndimage.binary_fill_holes`. Needs
                         no watertight surface, so it is the fallback for meshes that
                         fail the gate. Inherits surface_fill's dilation. Correct on
                         bores: a through-bore connects to the exterior so it is NOT
                         filled; an enclosed cavity IS.
    """
    if mesh.vertices.size and np.abs(mesh.vertices).max() > 1.0 + 1e-6:
        raise ValueError("voxelize() requires a normalized mesh -- call normalize() first")

    from scipy import ndimage

    pitch = 2.0 / res
    if method == "auto":
        method = "conservative" if mesh.is_watertight else "morphological"

    if method in ("exact", "conservative"):
        if not mesh.is_watertight:
            raise ValueError(f"{method} voxelization needs a watertight mesh (ray parity test)")
        inside = mesh.ray.contains_points(lattice_centres(res))
        grid = np.asarray(inside, dtype=bool).reshape(res, res, res)
        if method == "conservative":
            grid = grid | _rasterize_to_lattice(mesh.voxelized(pitch=pitch), res)
    elif method == "surface_fill":
        grid = _rasterize_to_lattice(mesh.voxelized(pitch=pitch).fill(), res)
    elif method == "morphological":
        shell = _rasterize_to_lattice(mesh.voxelized(pitch=pitch), res)
        grid = ndimage.binary_fill_holes(shell)
    else:
        raise ValueError(f"unknown voxelize method {method!r}")

    if grid.sum() == 0:
        print("  [voxelize] WARNING: zero occupied cells")
    assert grid.shape == (res, res, res), f"voxel shape {grid.shape} != {(res, res, res)}"
    return grid, method


def occupancy(grid: np.ndarray) -> float:
    return float(grid.mean())


# --------------------------------------------------------------------------
# artifact: the voxel plot that stage 2 is accepted on
# --------------------------------------------------------------------------

def plot_voxels(grid: np.ndarray, out_path: str | Path, title: str = "") -> Path:
    """Save a 3D voxel render. §18: save a PNG at every stage."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 3.6))
    for i, (elev, azim) in enumerate([(25, 45), (25, 135), (80, 45)]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.voxels(grid, facecolors="#4C9BE8", edgecolor="#20486E", linewidth=0.12)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(0, grid.shape[0]); ax.set_ylim(0, grid.shape[1]); ax.set_zlim(0, grid.shape[2])
        ax.set_box_aspect((1, 1, 1))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(f"elev={elev} azim={azim}", fontsize=8)
    fig.suptitle(title or "voxels", fontsize=10)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# CLI: one mesh, end to end -> stage 2 acceptance artifact
# --------------------------------------------------------------------------

def process(path: str | Path, res: int = VOXEL_RES) -> tuple[np.ndarray, WatertightReport, str]:
    """load -> normalize -> watertight gate -> voxelize. The canonical order."""
    mesh = load_mesh(path)
    mesh = normalize(mesh)
    mesh, rep = check_watertight(mesh)
    if not rep.watertight_final:
        print("  [gate] NOT watertight after repair -> morphological fill (grid-based)")
    grid, method = voxelize(mesh, res)
    return grid, rep, method


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python src/mesh.py <mesh-file> [out.png]")
        return 2
    src = Path(argv[1])
    out = Path(argv[2]) if len(argv) > 2 else Path(__file__).resolve().parents[1] / "results" / "stage2_voxels.png"

    print(f"[mesh] {src}")
    grid, rep, method = process(src)
    print(f"  watertight raw={rep.watertight_raw} final={rep.watertight_final} "
          f"repaired={rep.repaired} faces={rep.n_faces} volume={rep.volume:.4f}")
    print(f"  voxels {grid.shape} occupied={int(grid.sum())} ({occupancy(grid)*100:.1f}%) "
          f"via {method}")

    npy = out.with_suffix(".npy")
    npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy, grid)
    plot_voxels(grid, out, title=f"{src.name}  |  {int(grid.sum())} occupied ({occupancy(grid)*100:.1f}%)")
    print(f"  wrote {npy}\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
