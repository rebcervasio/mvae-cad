"""The latent table (§0.5 artifact 3, §3.7): each run scored on BOTH objectives.

    python src/table.py            # assemble from results/run_*.json
    python src/table.py --png      # also render it as an image for the slide

|          | classification | reconstruction IoU |
|----------|----------------|--------------------|
| A cls    | (expect best)  | (no decoder)       |
| B recon  | **probe <- the cell that matters** | (expect best) |
| C joint  | Q3: C vs A     | ?                  |

The off-diagonal cells are the finding. B-left is the one to discuss: if a latent trained
ONLY to reconstruct -- never having seen a label -- still separates classes under a linear
probe, then reconstruction quality really is a proxy for information content.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

ROWS = [("A", "A — classifier only", "cross-entropy only, labels used"),
        ("B", "B — autoencoder only", "reconstruction only, NO labels at all"),
        ("C", "C — joint (Phase 2)", "L_cls + lambda*L_recon")]


def load(run: str) -> dict | None:
    p = RESULTS / f"run_{run}.json"
    return json.loads(p.read_text()) if p.exists() else None


def build() -> dict:
    out = {}
    for key, label, note in ROWS:
        r = load(key)
        if r is None:
            continue
        out[key] = {
            "label": label, "note": note,
            "trained_acc": r.get("final_acc"),
            "probe_acc": (r.get("probe") or {}).get("probe_acc"),
            "iou": r.get("final_iou"),
            "n_train": r.get("n_train"), "n_test": r.get("n_test"),
            "seed": r.get("seed"), "elapsed_sec": r.get("elapsed_sec"),
        }
    return out


def fmt(v, pct=True):
    if v is None:
        return "     —"
    return f"{v*100:5.1f}%" if pct else f"{v:6.3f}"


def render_text(t: dict) -> str:
    L = []
    L.append("")
    L.append("  latent table — every run scored on BOTH objectives (§3.7)")
    L.append("  " + "-" * 74)
    L.append(f"  {'run':22} {'trained acc':>12} {'linear probe':>13} {'recon IoU':>11}")
    L.append("  " + "-" * 74)
    for key, _, _ in ROWS:
        if key not in t:
            continue
        r = t[key]
        L.append(f"  {r['label']:22} {fmt(r['trained_acc']):>12} {fmt(r['probe_acc']):>13} "
                 f"{fmt(r['iou'], pct=False):>11}")
    L.append("  " + "-" * 74)
    a, b, c = t.get("A"), t.get("B"), t.get("C")
    if b and b.get("probe_acc") is not None and a and a.get("trained_acc") is not None:
        gap = a["trained_acc"] - b["probe_acc"]
        L.append("")
        L.append(f"  Q2: B's latent never saw a label. Its probe scores {fmt(b['probe_acc'])} "
                 f"vs A's {fmt(a['trained_acc'])} ceiling (gap {gap*100:.1f} pts).")
        L.append("      " + ("Reconstruction alone yields a nearly class-separable latent."
                             if gap < 0.10 else
                             "Reconstruction alone does NOT give a class-separable latent — "
                             "the §5.1 worry, with evidence."))
    if c and a and c.get("trained_acc") is not None and a.get("trained_acc") is not None:
        d = c["trained_acc"] - a["trained_acc"]
        L.append("")
        L.append(f"  Q3 = C - A = {d*100:+.1f} pts. " +
                 ("The reconstruction objective HELPS classification." if d > 0.01 else
                  "The two objectives are orthogonal." if abs(d) <= 0.01 else
                  "They COMPETE — the §3.7 invariance conflict, concretely."))
    else:
        L.append("")
        L.append("  Q3: Run C not present (Phase 2 not run or abandoned — a reportable outcome, §3.3).")
    return "\n".join(L)


def render_png(t: dict, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [(k, t[k]) for k, _, _ in ROWS if k in t]
    fig, ax = plt.subplots(figsize=(8.4, 1.1 + 0.75 * len(rows)))
    ax.axis("off")
    cells, labels = [], []
    for k, r in rows:
        labels.append(r["label"])
        cells.append([fmt(r["trained_acc"]), fmt(r["probe_acc"]), fmt(r["iou"], pct=False)])
    tb = ax.table(cellText=cells, rowLabels=labels,
                  colLabels=["trained acc", "linear probe", "recon IoU"],
                  cellLoc="center", loc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(10)
    tb.scale(1, 1.7)
    # highlight B-left: the cell that matters (§3.7)
    for i, (k, _) in enumerate(rows):
        if k == "B":
            tb[(i + 1, 1)].set_facecolor("#FFE9A8")
    ax.set_title("Every latent scored on both objectives\n"
                 "highlighted: a latent that never saw a label, probed for class", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args(argv)
    t = build()
    if not t:
        raise SystemExit("no results/run_{A,B,C}.json found — train something first")
    print(render_text(t))
    (RESULTS / "table.json").write_text(json.dumps(t, indent=2))
    print(f"\n  wrote {RESULTS / 'table.json'}")
    if a.png:
        print(f"  wrote {render_png(t, RESULTS / 'plot3_latent_table.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
