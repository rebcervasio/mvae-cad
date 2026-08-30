"""View-count and latent-size sweeps, and the plots they feed (§0.5, §3.5, §17.8).

    python src/sweep.py views       # 1/3/6/12/20 views x runs A,B   -> plot 1
    python src/sweep.py latent      # z in {64,256,1024} on run B    -> §3.5 check 2
    python src/sweep.py all

Every run is a slice of the SAME cache (§15.5) -- no re-rendering, no re-encoding.
Results are JSON in results/, so plots regenerate without retraining.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

VIEW_COUNTS = [1, 3, 6, 12, 20]
LATENT_SIZES = [64, 256, 1024]


def cfg(**kw):
    """A config namespace with train.py's defaults, overridden by kw."""
    p = T.main.__wrapped__ if hasattr(T.main, "__wrapped__") else None
    import argparse as ap
    d = dict(run="A", dataset="cadnet", name=None, n_views=20, latent_dim=256,
             aggregator="max", epochs=60, batch_size=32, lr=1e-3, weight_decay=1e-4,
             val_frac=0.2, seed=0, device="cpu", overfit=0, gate_iou=0.95, lam=None,
             eval_every=10, dedup=True, dedup_thresh=0.999)
    d.update(kw)
    return ap.Namespace(**d)


def sweep_views(epochs: int, seed: int) -> list[dict]:
    out = []
    for run in ("A", "B"):
        for n in VIEW_COUNTS:
            print(f"\n{'='*64}\n  VIEW SWEEP  run={run}  views={n}\n{'='*64}")
            r = T.run_training(cfg(run=run, n_views=n, epochs=epochs, seed=seed,
                                   name=f"sweep_{run}_v{n}"))
            out.append({"run": run, "n_views": n,
                        "acc": r.get("final_acc"), "iou": r.get("final_iou"),
                        "probe": r.get("probe", {}).get("probe_acc")})
    (RESULTS / "sweep_views.json").write_text(json.dumps(out, indent=2))
    return out


def sweep_latent(epochs: int, seed: int) -> list[dict]:
    """§3.5 check 2. IoU climbing with latent size -> the bottleneck IS the bottleneck.
    IoU flat -> the bottleneck is upstream, in the 2D projection. Flat is the interesting
    case: capacity cannot recover geometry no camera ever saw."""
    out = []
    for z in LATENT_SIZES:
        print(f"\n{'='*64}\n  LATENT SWEEP  z={z}\n{'='*64}")
        r = T.run_training(cfg(run="B", latent_dim=z, epochs=epochs, seed=seed,
                               name=f"sweep_B_z{z}"))
        out.append({"latent_dim": z, "iou": r.get("final_iou"),
                    "probe": r.get("probe", {}).get("probe_acc")})
    (RESULTS / "sweep_latent.json").write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------

def plot_views(rows: list[dict]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
    a = [r for r in rows if r["run"] == "A"]
    b = [r for r in rows if r["run"] == "B"]

    ax1.plot([r["n_views"] for r in a], [r["acc"] for r in a], "o-",
             color="#4C9BE8", label="A: classification accuracy")
    if any(r.get("probe") is not None for r in b):
        ax1.plot([r["n_views"] for r in b], [r["probe"] for r in b], "s--",
                 color="#7EC8E3", label="B: linear probe on z (no labels seen)")
    ax1.set_xlabel("number of views")
    ax1.set_ylabel("accuracy", color="#2C6FA8")
    ax1.set_ylim(0, 1)
    ax1.set_xticks(VIEW_COUNTS)

    ax2 = ax1.twinx()
    ax2.plot([r["n_views"] for r in b], [r["iou"] for r in b], "^-",
             color="#E8834C", label="B: reconstruction IoU")
    ax2.set_ylabel("reconstruction IoU", color="#B35A28")
    ax2.set_ylim(0, 1)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
    ax1.grid(alpha=0.25)
    ax1.set_title("Do the two curves flatten in the same place?", fontsize=11)
    fig.tight_layout()
    p = RESULTS / "plot1_view_sweep.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def plot_latent(rows: list[dict]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot([r["latent_dim"] for r in rows], [r["iou"] for r in rows], "o-",
            color="#E8834C", linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(LATENT_SIZES)
    ax.set_xticklabels([str(z) for z in LATENT_SIZES])
    ax.set_xlabel("latent size |z|")
    ax.set_ylabel("reconstruction IoU")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ious = [r["iou"] for r in rows]
    spread = max(ious) - min(ious)
    verdict = ("IoU climbs with |z| -> the bottleneck IS the latent; a bigger latent is a cheap lever"
               if spread > 0.05 else
               "IoU is FLAT -> the bottleneck is UPSTREAM, in the 2D projection.\n"
               "Capacity cannot recover geometry no camera ever saw (§3.5 check 2).")
    ax.set_title("Latent-size probe", fontsize=11)
    ax.text(0.02, 0.03, verdict, transform=ax.transAxes, fontsize=8,
            va="bottom", ha="left", wrap=True)
    fig.tight_layout()
    p = RESULTS / "plot3_latent_probe.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["views", "latent", "all", "plot"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    RESULTS.mkdir(exist_ok=True)
    if a.what in ("views", "all"):
        print(plot_views(sweep_views(a.epochs, a.seed)))
    if a.what in ("latent", "all"):
        print(plot_latent(sweep_latent(a.epochs, a.seed)))
    if a.what == "plot":  # regenerate plots from existing JSON, no retraining
        for f, fn in [("sweep_views.json", plot_views), ("sweep_latent.json", plot_latent)]:
            p = RESULTS / f
            if p.exists():
                print(fn(json.loads(p.read_text())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
