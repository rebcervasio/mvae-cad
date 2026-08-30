"""Multi-view orthographic depth maps, pure numpy/torch (§4 plan A, §15.2).

No pyrender, no PyTorch3D, no OSMesa, no EGL. Deliberate (§13): installing those on
Apple Silicon is the single most likely way to lose two days. This is ~150 lines and
runs identically on Mac and Colab.

Convention (§15.2): +z in the CAMERA frame points from the camera toward the origin, so
SMALLER z = NEARER. Background is 1.0 = far.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

IMAGE_SIZE = 128
N_SURFACE_POINTS = 200_000
UP_REF = np.array([0.0, 0.0, 1.0])


# --------------------------------------------------------------------------
# §15.2 camera positions: the 20 dodecahedron vertices
# --------------------------------------------------------------------------

def dodecahedron_cameras() -> np.ndarray:
    """The 20 vertices of a regular dodecahedron, on the unit sphere.

    Same camera set as CADNET and the pruning paper, so the view-count sweep is
    comparable to the literature (§4).
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    inv = 1.0 / phi
    v = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                v.append((sx * 1.0, sy * 1.0, sz * 1.0))          # 8
    for sy in (-1, 1):
        for sz in (-1, 1):
            v.append((0.0, sy * inv, sz * phi))                    # 4
    for sx in (-1, 1):
        for sy in (-1, 1):
            v.append((sx * inv, sy * phi, 0.0))                    # 4
    for sx in (-1, 1):
        for sz in (-1, 1):
            v.append((sx * phi, 0.0, sz * inv))                    # 4
    cams = np.array(v, dtype=np.float64)
    assert cams.shape == (20, 3), f"expected 20 cameras, got {cams.shape}"
    cams /= np.linalg.norm(cams, axis=1, keepdims=True)
    return cams


def view_subset(n: int, cams: np.ndarray | None = None, seed_idx: int = 0) -> np.ndarray:
    """Deterministic farthest-point subset of the SAME 20 cameras (§15.2, §4).

    Never a different camera scheme per N -- the sweep must vary exactly one thing.
    Returns indices into `cams`.
    """
    cams = dodecahedron_cameras() if cams is None else cams
    if n >= len(cams):
        return np.arange(len(cams))
    sel = [seed_idx]
    d = np.linalg.norm(cams - cams[seed_idx], axis=1)
    while len(sel) < n:
        nxt = int(np.argmax(d))
        sel.append(nxt)
        d = np.minimum(d, np.linalg.norm(cams - cams[nxt], axis=1))
    return np.array(sorted(sel))


def look_at(cam_pos: np.ndarray) -> np.ndarray:
    """World->camera rotation. +z_cam points from the camera toward the origin.

    Guards the degenerate case where the camera direction is parallel to UP_REF
    (§15.2 step 2), which would make the cross product zero.
    """
    z_axis = -cam_pos / np.linalg.norm(cam_pos)   # camera looks toward origin
    up = UP_REF
    if abs(float(np.dot(z_axis, up))) > 0.999:    # degenerate: pick another up
        up = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=0)  # rows -> p_cam = R @ p_world


# --------------------------------------------------------------------------
# §15.2 rasteriser
# --------------------------------------------------------------------------

def render_views(
    mesh: trimesh.Trimesh,
    cams: np.ndarray | None = None,
    size: int = IMAGE_SIZE,
    n_points: int = N_SURFACE_POINTS,
    seed: int = 0,
    splat: bool = True,
) -> np.ndarray:
    """Orthographic depth maps. Mesh MUST already be normalized (mesh.normalize).

    Returns float32 [n_views, size, size], background 1.0 (far), object < 1.
    """
    if mesh.vertices.size and np.abs(mesh.vertices).max() > 1.0 + 1e-6:
        raise ValueError("render_views() requires a normalized mesh -- call normalize() first")

    cams = dodecahedron_cameras() if cams is None else cams
    rng = np.random.default_rng(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, n_points, seed=int(rng.integers(1 << 30)))
    pts = torch.as_tensor(np.asarray(pts), dtype=torch.float32)

    H = W = size
    out = torch.empty(len(cams), H, W, dtype=torch.float32)

    # 3x3 splat: fills pinholes, and per §15.2 matters more than raw point count.
    offs = [(0, 0)] if not splat else [(du, dv) for du in (-1, 0, 1) for dv in (-1, 0, 1)]

    for i, c in enumerate(cams):
        R = torch.as_tensor(look_at(c), dtype=torch.float32)
        p = pts @ R.T                                    # world -> camera

        u = ((p[:, 0] + 1.0) * 0.5 * W).long()           # orthographic projection
        v = ((1.0 - p[:, 1]) * 0.5 * H).long()           # v flipped: +y is up
        depth = (p[:, 2] + 1.0) * 0.5                    # -> [0,1], smaller = nearer

        buf = torch.full((H * W,), 1.0, dtype=torch.float32)   # background = far
        for du, dv in offs:
            uu, vv = u + du, v + dv
            ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            idx = (vv[ok] * W + uu[ok])
            buf.scatter_reduce_(0, idx, depth[ok], reduce="amin", include_self=True)
        out[i] = buf.view(H, W)

    return out.numpy()


def to_resnet_input(views: np.ndarray) -> torch.Tensor:
    """[N,H,W] depth -> [N,3,H,W] ImageNet-normalized (§15.2).

    Replicates the single channel to 3 and applies ImageNet mean/std. Yes it is a depth
    map and not a photo -- pretrained low-level filters still transfer, and this is what
    PointCLIP does.
    """
    x = torch.as_tensor(views, dtype=torch.float32).unsqueeze(1).repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


# --------------------------------------------------------------------------
# artifact: the 4x5 view grid that stage 3 is accepted on
# --------------------------------------------------------------------------

def plot_view_grid(views: np.ndarray, out_path: str | Path, title: str = "") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(views)
    rows, cols = (4, 5) if n == 20 else (int(np.ceil(n / 5)), min(n, 5))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.05))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i < n:
            # display 1-depth so the object is bright against a dark background
            ax.imshow(1.0 - views[i], cmap="magma", vmin=0.0, vmax=1.0)
            ax.set_title(f"view {i}", fontsize=7, pad=2)
        ax.axis("off")
    fig.suptitle(title or f"{n} orthographic depth views", fontsize=10)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def main(argv: list[str]) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mesh import load_mesh, normalize

    if len(argv) < 2:
        print("usage: python src/render.py <mesh-file> [out.png]")
        return 2
    src = Path(argv[1])
    out = Path(argv[2]) if len(argv) > 2 else Path(__file__).resolve().parents[1] / "results" / "stage3_views.png"

    import time
    mesh = normalize(load_mesh(src))
    t0 = time.perf_counter()
    views = render_views(mesh)
    dt = time.perf_counter() - t0

    cov = [float((v < 1.0).mean()) for v in views]
    print(f"[render] {src}")
    print(f"  views {views.shape} dtype={views.dtype}  in {dt:.2f}s ({dt/len(views)*1000:.0f} ms/view)")
    print(f"  depth range [{views.min():.3f}, {views.max():.3f}]  (background=1.0)")
    print(f"  silhouette coverage: min={min(cov)*100:.1f}% max={max(cov)*100:.1f}% mean={np.mean(cov)*100:.1f}%")
    plot_view_grid(views, out, title=f"{src.parent.name}/{src.name}  |  20 orthographic depth views")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
