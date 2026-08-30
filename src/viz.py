"""Interactive 3D viewer. Not part of the pipeline -- this is for looking at things.

    python src/viz.py <mesh>                    # spin the mesh around
    python src/viz.py <mesh> --voxels           # mesh | 32^3 voxels, side by side
    python src/viz.py <mesh> --voxels --res 64  # ...at another resolution
    python src/viz.py <mesh> --cameras          # the 20 dodecahedron viewpoints
    python src/viz.py --cls Nuts                # one CADNET part
    python src/viz.py --cls Flange_Like_Parts --n 4 --index 6   # browse parts 6..9
    python src/viz.py --cls toilet --n 4        # ModelNet10 still works
    python src/viz.py --list                    # what classes are available
    python src/viz.py <mesh> --save out.png     # build the scene, save, don't open a window

`--cls` searches CADNET first, then ModelNet10. Matching is case-insensitive and a unique
substring is enough, so `--cls flange` finds `Flange_Like_Parts`.

Controls once the window is open:
    drag        rotate            scroll      zoom
    shift-drag  pan               w           wireframe
    a           toggle axes       z           reset view
    q / esc     quit
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mesh import load_mesh, normalize, check_watertight, voxelize  # noqa: E402
from render import dodecahedron_cameras  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SLOT = 2.6  # spacing between side-by-side objects (each is <=2 units across)

MESH_COLOR = [150, 160, 175, 255]
VOXEL_COLOR = [76, 155, 232, 255]
CAM_COLOR = [232, 106, 76, 255]


def voxels_to_mesh(grid: np.ndarray, color=VOXEL_COLOR) -> trimesh.Trimesh:
    """Occupancy grid -> box mesh on the SAME [-1,1]^3 lattice mesh.py uses."""
    res = grid.shape[0]
    pitch = 2.0 / res
    tf = np.eye(4)
    tf[:3, :3] *= pitch
    tf[:3, 3] = -1.0 + 0.5 * pitch          # cell (0,0,0) centre
    boxes = trimesh.voxel.VoxelGrid(grid, transform=tf).as_boxes()
    boxes.visual.face_colors = color
    return boxes


def camera_rig(radius: float = 1.55, dot: float = 0.055) -> list[trimesh.Trimesh]:
    """The 20 dodecahedron viewpoints as dots, with a stalk pointing at the origin."""
    out = []
    for c in dodecahedron_cameras():
        s = trimesh.creation.icosphere(subdivisions=2, radius=dot)
        s.apply_translation(c * radius)
        s.visual.face_colors = CAM_COLOR
        out.append(s)
        # a thin stalk from the camera toward the object, so direction is legible
        seg = trimesh.creation.cylinder(radius=dot * 0.18, segment=np.array([c * radius, c * 1.05]))
        seg.visual.face_colors = CAM_COLOR
        out.append(seg)
    return out


# --------------------------------------------------------------------------
# class lookup -- CADNET and ModelNet10 have different layouts and extensions
# --------------------------------------------------------------------------

DATASETS = {
    "cadnet":     (ROOT / "data" / "CADNET_3317", "{cls}",         ("*.stl", "*.STL")),
    "modelnet10": (ROOT / "data" / "ModelNet10",  "{cls}/train",   ("*.off",)),
}


def available_classes(dataset: str) -> list[str]:
    root = DATASETS[dataset][0]
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("_"))


def resolve_class(name: str, dataset: str = "auto") -> tuple[str, str]:
    """(dataset, exact class name). Case-insensitive, unique substring accepted."""
    order = [dataset] if dataset != "auto" else ["cadnet", "modelnet10"]
    for ds in order:
        classes = available_classes(ds)
        exact = [c for c in classes if c.lower() == name.lower()]
        if exact:
            return ds, exact[0]
        partial = [c for c in classes if name.lower() in c.lower()]
        if len(partial) == 1:
            return ds, partial[0]
        if len(partial) > 1:
            raise SystemExit(f"'{name}' matches {len(partial)} {ds} classes: {', '.join(partial)}")

    lines = [f"no class matching '{name}'."]
    for ds in order:
        cs = available_classes(ds)
        if cs:
            lines.append(f"\n{ds} ({len(cs)}):\n  " + "\n  ".join(cs))
    raise SystemExit("\n".join(lines))


def class_files(name: str, dataset: str, n: int, index: int) -> list[str]:
    ds, cls = resolve_class(name, dataset)
    root, sub, patterns = DATASETS[ds]
    d = root / sub.format(cls=cls)
    files = sorted({str(f) for pat in patterns for f in d.glob(pat)})
    if not files:
        raise SystemExit(f"no meshes in {d}")
    print(f"  [{ds}] {cls}: {len(files)} parts, showing {index}..{min(index + n, len(files)) - 1}")
    return files[index : index + n]


def build_scene(args) -> trimesh.Scene:
    scene = trimesh.Scene()

    if args.cls:
        files = class_files(args.cls, args.dataset, args.n, args.index)
    else:
        files = [args.mesh]

    for slot, fp in enumerate(files):
        m = normalize(load_mesh(fp))
        m, rep = check_watertight(m)
        m.visual.face_colors = MESH_COLOR
        x = slot * SLOT * (2 if args.voxels else 1)

        mm = m.copy()
        mm.apply_translation([x, 0, 0])
        scene.add_geometry(mm, node_name=f"mesh_{slot}")
        print(f"  [{slot}] {Path(fp).name:24} faces={len(m.faces):6}  watertight={rep.watertight_final}")

        if args.voxels:
            grid, method = voxelize(m, res=args.res)
            vx = voxels_to_mesh(grid)
            vx.apply_translation([x + SLOT, 0, 0])
            scene.add_geometry(vx, node_name=f"voxels_{slot}")
            print(f"       {args.res}^3 voxels: {int(grid.sum())} occupied "
                  f"({grid.mean()*100:.1f}%) via {method}")

        if args.cameras and slot == 0:
            for i, g in enumerate(camera_rig()):
                g.apply_translation([x, 0, 0])
                scene.add_geometry(g, node_name=f"cam_{i}")

    return scene


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mesh", nargs="?", help="path to a mesh file (.off/.stl/.obj/.ply)")
    p.add_argument("--voxels", action="store_true", help="show the voxelization beside the mesh")
    p.add_argument("--res", type=int, default=32, help="voxel resolution (default 32)")
    p.add_argument("--cameras", action="store_true", help="show the 20 dodecahedron viewpoints")
    p.add_argument("--cls", help="browse a class instead of one file, e.g. Nuts or flange")
    p.add_argument("--dataset", default="auto", choices=["auto", "cadnet", "modelnet10"],
                   help="which dataset --cls refers to (default auto: CADNET, then ModelNet10)")
    p.add_argument("--n", type=int, default=1, help="how many parts when using --cls (default 1)")
    p.add_argument("--index", type=int, default=0, help="skip the first N parts of the class")
    p.add_argument("--list", action="store_true", help="list available classes and exit")
    p.add_argument("--save", help="render to this PNG instead of opening a window")
    args = p.parse_args(argv)

    if args.list:
        for ds in ("cadnet", "modelnet10"):
            cs = available_classes(ds)
            print(f"\n{ds} ({len(cs)} classes)")
            for c in cs:
                print(f"  {c}")
        return 0

    if not args.mesh and not args.cls:
        p.print_help()
        return 2

    print("[viz] building scene")
    scene = build_scene(args)
    print(f"  {len(scene.geometry)} geometries")

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        png = scene.save_image(resolution=(1400, 800), visible=True)
        out.write_bytes(png)
        print(f"  wrote {out}")
        return 0

    print("\n  drag=rotate  scroll=zoom  shift-drag=pan  w=wireframe  a=axes  z=reset  q=quit\n")
    scene.show(smooth=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
