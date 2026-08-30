"""Build the feature/voxel cache once (§15.5). This is what makes the week fit (§3.4).

The ResNet18 backbone is FROZEN, so its 512-d output per view is computed exactly once for
the whole dataset and written to disk. Every training run afterwards loads this and trains
only the aggregator + heads, dropping runs from hours to under a minute -- which is what
makes the view sweep and the latent sweep affordable on a laptop.

    python src/cache.py --dataset modelnet10 --limit 200
    python src/cache.py --dataset cadnet

Writes data/cache_<dataset>.npz:
    feats   [N, 20, 512]   float16
    voxels  [N, 32,32,32]  bool
    labels  [N]            int64
    ids     [N]            str
plus per-part provenance (voxel method, watertight verdict, thin-wall flag) and a
data/cache_<dataset>_report.json with the rates that §5 asks to be reported.
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
from device import resolve_device  # noqa: E402
from mesh import load_mesh, normalize, check_watertight, voxelize, VOXEL_RES  # noqa: E402
from render import render_views, to_resnet_input, dodecahedron_cameras  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
N_VIEWS = 20
FEAT_DIM = 512


# --------------------------------------------------------------------------
# dataset discovery
# --------------------------------------------------------------------------

def find_parts(dataset: str, split: str = "train", limit_per_class: int | None = None):
    """Return [(path, class_name)]. Kept dumb on purpose -- one glob per dataset layout."""
    if dataset == "modelnet10":
        root = DATA / "ModelNet10"
        classes = sorted(p.name for p in root.iterdir() if p.is_dir())
        out = []
        for c in classes:
            files = sorted((root / c / split).glob("*.off"))
            out += [(f, c) for f in (files[:limit_per_class] if limit_per_class else files)]
        return out, classes

    if dataset == "cadnet":
        root = DATA / "CADNET_3317"
        if not root.exists():
            cands = [p for p in DATA.glob("**/CADNET*") if p.is_dir()]
            if cands:
                root = cands[0]
        classes = sorted(p.name for p in root.iterdir() if p.is_dir())
        out = []
        for c in classes:
            files = sorted(list((root / c).glob("*.stl")) + list((root / c).glob("*.STL")))
            # A per-class cap is blind to geometry, so it cannot correlate with thinness.
            # Selecting whole CLASSES would: class size correlates -0.211 with thinness,
            # so "the N largest classes" silently drops the thin ones (see NOTES.md).
            out += [(f, c) for f in (files[:limit_per_class] if limit_per_class else files)]
        return out, classes

    raise ValueError(f"unknown dataset {dataset!r}")


# --------------------------------------------------------------------------
# backbone
# --------------------------------------------------------------------------

def build_backbone(device: torch.device) -> torch.nn.Module:
    """ImageNet-pretrained ResNet18 with the fc removed, frozen (§15.4, §3.4)."""
    from torchvision.models import resnet18, ResNet18_Weights
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


# --------------------------------------------------------------------------
# the thin-wall check (see NOTES: exact containment drops sub-cell features)
# --------------------------------------------------------------------------

def thin_wall_ratio(mesh: trimesh.Trimesh, grid: np.ndarray, res: int) -> float:
    """What fraction of this part's target is sub-cell geometry?

    Compares the stored `conservative` target against `exact`, which keeps only cells whose
    CENTRE is inside the solid and therefore drops anything thinner than one cell. So
    (conservative - exact) / conservative is "how much of this part exists only because we
    dilate": 0 = chunky, ->1 = the part is essentially all sub-cell wall (a plate, a clip).

    ratio == 1.0 means `exact` would have produced an EMPTY grid -- the part exists in the
    target purely by dilation. Those parts are counted and reported, never dropped (§18);
    for them, IoU measures silhouette agreement more than volumetric agreement.
    """
    n = int(grid.sum())
    if n == 0:
        return float("nan")
    import contextlib, io
    try:
        # `exact` legitimately returns an empty grid for sub-cell parts -- that is the
        # signal being measured here, not an error, so its warning is suppressed.
        with contextlib.redirect_stdout(io.StringIO()):
            ex, _ = voxelize(mesh, res=res, method="exact")
    except Exception:
        return float("nan")
    return float((n - int(ex.sum())) / n)


# --------------------------------------------------------------------------
# main build
# --------------------------------------------------------------------------

def build(dataset: str, split: str, limit: int | None, res: int, size: int,
          n_points: int, seed: int, batch_views: int = 20) -> Path:
    device = resolve_device("auto")
    parts, classes = find_parts(dataset, split, limit)
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"[cache] dataset={dataset} split={split} parts={len(parts)} classes={len(classes)}")
    print(f"[cache] device={device} views={N_VIEWS} res={res}^3 image={size}")

    net = build_backbone(device)
    cams = dodecahedron_cameras()

    feats, voxels, labels, ids = [], [], [], []
    prov = []          # per-part provenance
    skipped = []       # (id, reason) -- counted and printed, never silent
    t_start = time.perf_counter()

    for i, (path, cname) in enumerate(parts):
        try:
            m = load_mesh(path)
            m = normalize(m)
            m, wt = check_watertight(m)
            grid, method = voxelize(m, res=res)
            if grid.sum() == 0:
                skipped.append((path.stem, "empty voxel grid"))
                continue
            thin = thin_wall_ratio(m, grid, res)

            views = render_views(m, cams=cams, size=size, n_points=n_points, seed=seed)
            x = to_resnet_input(views).to(device)
            with torch.no_grad():
                f = net(x)                       # [20, 512]
            feats.append(f.detach().to("cpu").numpy().astype(np.float16))
            voxels.append(grid)
            labels.append(cls_to_idx[cname])
            ids.append(path.stem)
            prov.append({"id": path.stem, "cls": cname, "voxel_method": method,
                         "watertight_raw": wt.watertight_raw,
                         "watertight_final": wt.watertight_final,
                         "repaired": wt.repaired,
                         "occupancy": float(grid.mean()),
                         "thin_wall_ratio": thin})
        except Exception as e:
            skipped.append((path.stem, f"{type(e).__name__}: {e}"))
            continue

        if (i + 1) % 50 == 0 or i + 1 == len(parts):
            el = time.perf_counter() - t_start
            rate = (i + 1) / el
            print(f"  {i+1:5}/{len(parts)}  {el:6.1f}s  {rate:5.2f} parts/s  "
                  f"eta {(len(parts)-i-1)/rate/60:5.1f} min  kept={len(ids)} skipped={len(skipped)}")

    if not ids:
        raise SystemExit("[cache] nothing was cached -- every part failed. See errors above.")

    feats = np.stack(feats)
    voxels = np.stack(voxels)
    labels = np.asarray(labels, dtype=np.int64)
    ids_arr = np.asarray(ids)

    out = DATA / f"cache_{dataset}.npz"
    np.savez_compressed(out, feats=feats, voxels=voxels, labels=labels, ids=ids_arr,
                        classes=np.asarray(classes))
    assert feats.shape == (len(ids), N_VIEWS, FEAT_DIM), f"bad feats shape {feats.shape}"
    assert voxels.shape == (len(ids), res, res, res), f"bad voxels shape {voxels.shape}"

    report = summarize(dataset, split, res, size, seed, classes, prov, skipped,
                       feats, voxels, labels, time.perf_counter() - t_start)
    (DATA / f"cache_{dataset}_report.json").write_text(json.dumps(report, indent=2))
    print_report(report, out)
    return out


def summarize(dataset, split, res, size, seed, classes, prov, skipped,
              feats, voxels, labels, elapsed) -> dict:
    n = len(prov)
    wt_raw = sum(p["watertight_raw"] for p in prov)
    wt_fin = sum(p["watertight_final"] for p in prov)
    thin = [p["thin_wall_ratio"] for p in prov if not np.isnan(p["thin_wall_ratio"])]
    occ = voxels.reshape(len(voxels), -1).mean(axis=1)
    empty = int(voxels.sum())
    total = int(voxels.size)
    return {
        "dataset": dataset, "split": split, "seed": seed,
        "voxel_res": res, "image_size": size, "n_views": N_VIEWS,
        "n_kept": n, "n_skipped": len(skipped),
        "skipped": skipped[:50],
        "classes": classes,
        "class_counts": {c: int((labels == i).sum()) for i, c in enumerate(classes)},
        "watertight_raw_rate": wt_raw / n,
        "watertight_final_rate": wt_fin / n,
        "voxel_method_counts": {m: sum(1 for p in prov if p["voxel_method"] == m)
                                for m in {p["voxel_method"] for p in prov}},
        "occupancy_mean": float(occ.mean()),
        "occupancy_p05": float(np.percentile(occ, 5)),
        "occupancy_p95": float(np.percentile(occ, 95)),
        # §15.6: pos_weight = #empty / #occupied, computed once from the data
        "pos_weight": float((total - empty) / max(empty, 1)),
        "thin_wall_ratio_mean": float(np.mean(thin)) if thin else None,
        "thin_wall_frac_over_0.3": float(np.mean([t > 0.3 for t in thin])) if thin else None,
        # parts that exist in the target ONLY because of dilation (exact would be empty)
        "frac_exact_would_be_empty": float(np.mean([t >= 0.999 for t in thin])) if thin else None,
        "thin_wall_ratio_by_class": {
            c: float(np.mean([p["thin_wall_ratio"] for p in prov
                              if p["cls"] == c and not np.isnan(p["thin_wall_ratio"])] or [np.nan]))
            for c in classes},
        "feats_shape": list(feats.shape), "voxels_shape": list(voxels.shape),
        "elapsed_sec": round(elapsed, 1),
    }


def print_report(r: dict, out: Path) -> None:
    print()
    print(f"[cache] wrote {out}  ({out.stat().st_size/1024**2:.1f} MB)")
    print(f"  kept {r['n_kept']}  skipped {r['n_skipped']}")
    print(f"  feats  {r['feats_shape']}  float16")
    print(f"  voxels {r['voxels_shape']}  bool")
    print()
    print(f"  watertight as loaded : {r['watertight_raw_rate']*100:5.1f}%   <- §5 asks this be reported")
    print(f"  watertight after repair: {r['watertight_final_rate']*100:5.1f}%")
    print(f"  voxel methods: {r['voxel_method_counts']}")
    print()
    print(f"  occupancy mean {r['occupancy_mean']*100:.2f}%  (p05 {r['occupancy_p05']*100:.2f}%  p95 {r['occupancy_p95']*100:.2f}%)")
    print(f"  pos_weight = {r['pos_weight']:.1f}   <- §15.6, use this in BCEWithLogitsLoss")
    if r["thin_wall_ratio_mean"] is not None:
        print(f"  thin-wall ratio mean {r['thin_wall_ratio_mean']:.3f}; "
              f"{r['thin_wall_frac_over_0.3']*100:.1f}% of parts over 0.3")
        print(f"  parts that exist ONLY via dilation (exact would be empty): "
              f"{r['frac_exact_would_be_empty']*100:.1f}%")
        worst = sorted(((v, k) for k, v in r["thin_wall_ratio_by_class"].items()
                        if v == v), reverse=True)[:5]
        print("  thinnest classes: " + ", ".join(f"{k} {v:.2f}" for v, k in worst))
    print(f"  elapsed {r['elapsed_sec']}s")
    if r["n_skipped"]:
        print(f"\n  SKIPPED ({r['n_skipped']}):")
        for pid, why in r["skipped"][:10]:
            print(f"    {pid}: {why}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="modelnet10", choices=["modelnet10", "cadnet"])
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=None, help="max parts per class")
    p.add_argument("--res", type=int, default=VOXEL_RES)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--points", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    build(a.dataset, a.split, a.limit, a.res, a.size, a.points, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
