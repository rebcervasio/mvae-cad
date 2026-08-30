"""Latency: encode/decode ms per part, batched and unbatched (§0.5 plot 4, §15.7).

    python src/bench.py results/ckpt_B.pt

Reports batch 1 vs batch 64, at 12 and 20 views, on every available device -- so the
"GPU or CPU?" question in §0.8 can be answered with our own numbers instead of a guess.

Timing discipline (§15.7): warm up 10 iterations, then an explicit device sync before and
after. Stage 1 measured a ~207 ms first-call kernel compilation on MPS; without warmup this
table would report compile time as inference time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from device import sync  # noqa: E402
from recon_viz import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

WARMUP = 10
ITERS = 30


def time_it(fn, dev: torch.device, warmup: int = WARMUP, iters: int = ITERS) -> float:
    """Mean seconds per call, warmed up and synced."""
    for _ in range(warmup):
        fn()
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(dev)
    return (time.perf_counter() - t0) / iters


@torch.no_grad()
def bench(ckpt_path: Path, devices: list[str], batches=(1, 64), views=(12, 20)) -> dict:
    model, ck = load_model(ckpt_path)
    latent = ck["config"]["latent_dim"]
    res = ck["res"]
    rows = []

    for dname in devices:
        dev = torch.device(dname)
        try:
            m = model.to(dev)
        except Exception as e:
            print(f"  {dname}: unavailable ({e})")
            continue
        for nv in views:
            for bs in batches:
                feats = torch.randn(bs, nv, 512, device=dev)
                enc = time_it(lambda: m.encode(feats), dev)
                row = {"device": dname, "views": nv, "batch": bs,
                       "encode_ms_per_part": enc / bs * 1000}
                if m.decoder is not None:
                    z = m.encode(feats)
                    dec = time_it(lambda: m.decoder(z), dev)
                    row["decode_ms_per_part"] = dec / bs * 1000
                    row["total_ms_per_part"] = row["encode_ms_per_part"] + row["decode_ms_per_part"]
                rows.append(row)
                print(f"  {dname:5} views={nv:2} batch={bs:3}  "
                      f"encode {row['encode_ms_per_part']:8.3f} ms/part"
                      + (f"   decode {row['decode_ms_per_part']:8.3f}"
                         f"   total {row['total_ms_per_part']:8.3f}" if "decode_ms_per_part" in row else ""))
    return {"checkpoint": str(ckpt_path), "latent_dim": latent, "voxel_res": res,
            "warmup": WARMUP, "iters": ITERS, "rows": rows,
            "note": "NB: encoder timing EXCLUDES rendering and the frozen ResNet18 "
                    "(features are cached, §3.4). See render_ms below for the real cost."}


def bench_frontend(devices: list[str]) -> dict:
    """The part the cache hides: rendering + the frozen backbone. An end-to-end latency
    claim that omits these would be dishonest, since deployment cannot cache them."""
    from render import render_views, to_resnet_input, dodecahedron_cameras
    from mesh import load_mesh, normalize
    from cache import build_backbone
    import glob

    fp = sorted(glob.glob(str(ROOT / "data" / "CADNET_3317" / "*" / "*.stl")))[0]
    mesh = normalize(load_mesh(fp))
    cams = dodecahedron_cameras()
    t_render = time_it(lambda: render_views(mesh, cams=cams), torch.device("cpu"), warmup=1, iters=3)

    out = {"render_ms_per_part": t_render * 1000, "backbone": {}}
    print(f"\n  render 20 views: {t_render*1000:.1f} ms/part")
    for dname in devices:
        dev = torch.device(dname)
        net = build_backbone(dev)
        x = to_resnet_input(render_views(mesh, cams=cams)).to(dev)
        t = time_it(lambda: net(x), dev, warmup=5, iters=10)
        out["backbone"][dname] = t * 1000
        print(f"  frozen ResNet18, 20 views on {dname:5}: {t*1000:8.1f} ms/part")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--no-frontend", action="store_true")
    a = ap.parse_args(argv)

    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    print(f"[bench] devices: {devices}")
    out = bench(Path(a.ckpt), devices)
    if not a.no_frontend:
        out["frontend"] = bench_frontend(devices)

    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / "bench.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
