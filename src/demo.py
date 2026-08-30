"""The acceptance test (§6, §17.11):  python src/demo.py part.step

1. load and tessellate the STEP file
2. split into solids if it is an assembly (the demo file has 37)
3. render N views, save the view grid PNG -- so we see what the model sees
4. encoder -> predicted class + confidence per part
5. decoder -> original vs reconstructed voxels, plus an error map
6. encode/decode time in ms, per part and batched

HONESTY (§6): the demo file's parts are almost certainly NOT in the training taxonomy, so
any class prediction is out-of-taxonomy and is labelled as such. **No classification claim
is made from this demo.** The decoder works regardless of taxonomy -- that is what it shows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from device import resolve_device, sync  # noqa: E402
from mesh import normalize, check_watertight, voxelize  # noqa: E402
from render import render_views, to_resnet_input, dodecahedron_cameras, plot_view_grid  # noqa: E402
from metrics import iou  # noqa: E402
from recon_viz import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


# --------------------------------------------------------------------------
# STEP loading (§16 trap 7: timeboxed, with a documented fallback)
# --------------------------------------------------------------------------

def load_step_solids(path: Path, lin_defl: float = 0.1, ang_defl: float = 0.5) -> list[trimesh.Trimesh]:
    """STEP -> one Trimesh per solid. Never write a CAD kernel (§1) -- this is OCC.

    Falls back to trimesh for .stl/.obj/.ply so the demo is testable without a CAD kernel.
    """
    suffix = path.suffix.lower()
    if suffix not in (".step", ".stp"):
        m = trimesh.load(str(path), force="mesh")
        return [m]

    try:
        from OCP.STEPControl import STEPControl_Reader
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE
        from OCP.BRep import BRep_Tool
        from OCP.TopLoc import TopLoc_Location
        from OCP.TopoDS import TopoDS
        from OCP.IFSelect import IFSelect_RetDone
    except ImportError as e:
        raise SystemExit(
            f"STEP loading needs the OCP/cadquery kernel and it is not installed ({e}).\n"
            f"  install:  pip install cadquery-ocp\n"
            f"  fallback (§16 trap 7): convert the STEP to STL once by hand and pass that."
        )

    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise SystemExit(f"OCC could not read {path}")
    reader.TransferRoots()
    shape = reader.OneShape()

    BRepMesh_IncrementalMesh(shape, lin_defl, False, ang_defl, True)

    solids = []
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        solid = TopoDS.Solid_s(exp.Current())
        verts, faces = [], []
        fexp = TopExp_Explorer(solid, TopAbs_FACE)
        while fexp.More():
            face = TopoDS.Face_s(fexp.Current())
            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation_s(face, loc)
            if tri is not None:
                tf = loc.Transformation()
                base = len(verts)
                for i in range(1, tri.NbNodes() + 1):
                    p = tri.Node(i).Transformed(tf)
                    verts.append([p.X(), p.Y(), p.Z()])
                for i in range(1, tri.NbTriangles() + 1):
                    a, b, c = tri.Triangle(i).Get()
                    faces.append([base + a - 1, base + b - 1, base + c - 1])
            fexp.Next()
        if faces:
            m = trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces), process=True)
            m.merge_vertices()
            solids.append(m)
        exp.Next()
    return solids


# --------------------------------------------------------------------------
# demo
# --------------------------------------------------------------------------

def run(step_path: Path, ckpt_path: Path, max_parts: int, out_prefix: str) -> dict:
    dev = resolve_device("cpu")   # measured faster for this decoder; see NOTES
    model, ck = load_model(ckpt_path)
    model = model.to(dev)
    classes = [str(c) for c in ck["classes"]]
    cams = dodecahedron_cameras()[ck["view_idx"]]

    from cache import build_backbone
    backbone = build_backbone(dev)

    print(f"[demo] {step_path}")
    t0 = time.perf_counter()
    solids = load_step_solids(step_path)
    t_load = time.perf_counter() - t0
    print(f"  tessellated {len(solids)} solid(s) in {t_load:.1f}s")
    if len(solids) > max_parts:
        print(f"  showing the first {max_parts} (of {len(solids)})")
    solids = solids[:max_parts]

    rows = []
    for i, raw in enumerate(solids):
        rec = {"part": i, "faces": int(len(raw.faces))}
        try:
            m = normalize(raw)
            m, wt = check_watertight(m)
            rec["watertight"] = wt.watertight_final
            grid, method = voxelize(m)
            rec["voxel_method"] = method
            rec["occupancy"] = float(grid.mean())

            t0 = time.perf_counter(); views = render_views(m, cams=cams); t_r = time.perf_counter() - t0
            x = to_resnet_input(views).to(dev)

            sync(dev); t0 = time.perf_counter()
            with torch.no_grad():
                f = backbone(x).unsqueeze(0)
                z = model.encode(f)
            sync(dev); t_e = time.perf_counter() - t0

            rec.update(render_ms=t_r * 1000, encode_ms=t_e * 1000)

            if model.decoder is not None:
                sync(dev); t0 = time.perf_counter()
                with torch.no_grad():
                    logits = model.decoder(z)
                sync(dev); t_d = time.perf_counter() - t0
                tgt = torch.tensor(grid[None], dtype=torch.float32)
                rec["decode_ms"] = t_d * 1000
                rec["iou"] = float(iou(logits, tgt).mean())
                rec["_pred"] = (torch.sigmoid(logits)[0] > 0.5).numpy()
                rec["_true"] = grid

            if model.cls_head is not None:
                with torch.no_grad():
                    p = torch.softmax(model.cls_head(z), dim=1)[0]
                k = int(p.argmax())
                # OUT OF TAXONOMY -- reported as a curiosity only (§6)
                rec["pred_class_OUT_OF_TAXONOMY"] = classes[k]
                rec["confidence"] = float(p[k])

            if i == 0:
                plot_view_grid(views, RESULTS / f"{out_prefix}_views.png",
                               title=f"{step_path.name} part 0 — what the model sees")
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"  part {i}: FAILED {rec['error']}")
        rows.append(rec)
        if "error" not in rec:
            msg = f"  part {i:3}  faces {rec['faces']:6}  wt={str(rec.get('watertight')):5}"
            if "iou" in rec:
                msg += f"  IoU {rec['iou']:.3f}"
            msg += f"  render {rec['render_ms']:6.0f}ms  encode {rec['encode_ms']:6.1f}ms"
            if "decode_ms" in rec:
                msg += f"  decode {rec['decode_ms']:5.1f}ms"
            print(msg)

    ok = [r for r in rows if "iou" in r]
    if ok:
        plot_parts(ok[:4], RESULTS / f"{out_prefix}_recon.png", step_path.name)

    summary = {
        "step_file": str(step_path), "checkpoint": str(ckpt_path),
        "n_solids_total": len(solids), "n_ok": len(ok),
        "classification_claim": "NONE - demo parts are out of the training taxonomy (§6)",
        "mean_iou": float(np.mean([r["iou"] for r in ok])) if ok else None,
        "parts": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
    }
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"{out_prefix}.json"
    p.write_text(json.dumps(summary, indent=2))
    print(f"\n  parts reconstructed: {len(ok)}/{len(rows)}"
          + (f"   mean IoU {summary['mean_iou']:.3f}" if ok else ""))
    print(f"  wrote {p}")
    return summary


def plot_parts(rows, out: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    fig = plt.figure(figsize=(9.6, 3.2 * n))
    for r, rec in enumerate(rows):
        t, p = rec["_true"], rec["_pred"]
        for c, (g, lbl, col) in enumerate([(t, "original", "#4C9BE8"), (p, "reconstruction", "#E8834C")]):
            ax = fig.add_subplot(n, 3, r * 3 + c + 1, projection="3d")
            if g.any():
                ax.voxels(g, facecolors=col, edgecolor="none")
            ax.set_box_aspect((1, 1, 1)); ax.view_init(25, 45)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            if r == 0:
                ax.set_title(lbl, fontsize=10)
        ax = fig.add_subplot(n, 3, r * 3 + 3, projection="3d")
        both = (t & p) | (t & ~p) | (p & ~t)
        if both.any():
            colors = np.empty(both.shape, dtype=object)
            colors[t & p] = "#BBBBBB"; colors[t & ~p] = "#C0392B"; colors[p & ~t] = "#27AE60"
            ax.voxels(both, facecolors=colors, edgecolor="none")
        ax.set_box_aspect((1, 1, 1)); ax.view_init(25, 45)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        if r == 0:
            ax.set_title("red=MISSED  green=ADDED", fontsize=10)
    fig.suptitle(f"{title} — reconstruction from 20 views only", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=115)
    plt.close(fig)
    print(f"  wrote {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", help="path to a .step/.stp file (or .stl/.obj fallback)")
    ap.add_argument("--ckpt", default=str(ROOT / "results" / "ckpt_B.pt"))
    ap.add_argument("--max-parts", dest="max_parts", type=int, default=8)
    ap.add_argument("--prefix", default="demo")
    a = ap.parse_args(argv)
    run(Path(a.step), Path(a.ckpt), a.max_parts, a.prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
