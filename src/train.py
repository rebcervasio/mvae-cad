"""Runs A / B / C and the overfit gate, selected by config (§3.3, §15.6, §17).

    python src/train.py --run A                      # classifier only, cross-entropy
    python src/train.py --run B                      # autoencoder only, NO labels used
    python src/train.py --run C --lam auto            # joint (Phase 2)
    python src/train.py --run B --overfit 20          # §3.5 gate: memorise 20 parts

Everything trains on CACHED frozen-backbone features (§3.4), so a run is a minute or two,
which is what makes the §0.5 sweeps affordable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from device import resolve_device, sync  # noqa: E402
from model import MVAE  # noqa: E402
from metrics import iou, accuracy, linear_probe, by_group, best_threshold_iou  # noqa: E402
from render import view_subset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS = ROOT / "data", ROOT / "results"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_cache(dataset: str) -> dict:
    p = DATA / f"cache_{dataset}.npz"
    if not p.exists():
        raise SystemExit(f"no cache at {p} -- run: python src/cache.py --dataset {dataset}")
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def duplicate_groups(feats: np.ndarray, voxels: np.ndarray, thresh: float = 0.999) -> np.ndarray:
    """Group near-identical parts so they cannot be split across train/test.

    CADNET contains duplicate part variants: 17.5% of parts share an IDENTICAL 32^3 grid
    with another part, and a plain random split scatters those across both halves. Measured
    consequence: 1-nearest-neighbour alone scored 95.5% -- HIGHER than the trained model --
    i.e. the task had become "retrieve the copy you already saw", not classification. It
    would inflate reconstruction IoU the same way, by scoring the decoder on memorised parts.

    Two parts are joined if they share an identical voxel grid OR their pooled features have
    cosine similarity >= `thresh`. Groups are the connected components of that relation.
    0.999 is deliberately strict: distinct-but-similar parts (all bolts look alike) sit
    around 0.99, and merging those would collapse whole classes.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(feats)
    P = feats.max(axis=1)
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    S = P @ P.T
    np.fill_diagonal(S, 0.0)
    r, c = np.nonzero(S >= thresh)

    # identical voxel grids join too, even if features drift slightly
    seen: dict[bytes, int] = {}
    vr, vc = [], []
    for i, g in enumerate(voxels.reshape(n, -1)):
        k = g.tobytes()
        if k in seen:
            vr.append(seen[k]); vc.append(i)
        else:
            seen[k] = i
    r = np.concatenate([r, np.array(vr, dtype=int)])
    c = np.concatenate([c, np.array(vc, dtype=int)])

    adj = coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    _, comp = connected_components(adj, directed=False)
    return comp


def stratified_split(labels: np.ndarray, val_frac: float, seed: int,
                     groups: np.ndarray | None = None):
    """Per-class split, so every class appears in both halves (§16 trap 8: seeded).

    If `groups` is given, WHOLE GROUPS are assigned to one side, so duplicate parts cannot
    leak across the split.
    """
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.arange(len(labels))
    tr, te = [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        gs = np.unique(groups[idx])
        rng.shuffle(gs)
        target = max(1, int(round(len(idx) * val_frac)))
        chosen, n_te = set(), 0
        for g in gs:
            if n_te >= target or len(chosen) == len(gs) - 1:
                break
            chosen.add(g)
            n_te += int((groups[idx] == g).sum())
        mask = np.isin(groups[idx], list(chosen))
        te += list(idx[mask])
        tr += list(idx[~mask])
    return np.array(sorted(tr)), np.array(sorted(te))


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def run_training(cfg) -> dict:
    set_seed(cfg.seed)
    dev = resolve_device(cfg.device)
    cache = load_cache(cfg.dataset)
    classes = [str(c) for c in cache["classes"]]

    feats_all = cache["feats"].astype(np.float32)          # [N, 20, 512]
    voxels_all = cache["voxels"]                           # [N, r, r, r] bool
    labels_all = cache["labels"].astype(np.int64)
    res = voxels_all.shape[-1]

    # --- view-count sweep is a SLICE of the cache (§15.5); no re-render, no re-encode ---
    vidx = view_subset(cfg.n_views)
    feats_all = feats_all[:, vidx, :]

    if cfg.overfit:
        # §3.5 gate: train == eval on N parts, no held-out set, no regularisation.
        rng = np.random.default_rng(cfg.seed)
        sel = rng.choice(len(labels_all), size=min(cfg.overfit, len(labels_all)), replace=False)
        tr = te = np.sort(sel)
    else:
        groups = None
        if cfg.dedup:
            groups = duplicate_groups(cache["feats"].astype(np.float32), voxels_all, cfg.dedup_thresh)
            n_dup = len(groups) - len(np.unique(groups))
            print(f"  dedup: {len(np.unique(groups))} groups from {len(groups)} parts "
                  f"({n_dup} duplicates, {100*n_dup/len(groups):.1f}%)")
        tr, te = stratified_split(labels_all, cfg.val_frac, cfg.seed, groups)

    F = torch.tensor(feats_all, device=dev)
    V = torch.tensor(voxels_all, dtype=torch.float32, device=dev)
    Y = torch.tensor(labels_all, device=dev)
    tr_t = torch.tensor(tr, device=dev)
    te_t = torch.tensor(te, device=dev)

    use_cls = cfg.run in ("A", "C")
    use_dec = cfg.run in ("B", "C")
    model = MVAE(feat_dim=F.shape[-1], latent_dim=cfg.latent_dim, num_classes=len(classes),
                 res=res, aggregator=cfg.aggregator, use_cls=use_cls, use_dec=use_dec).to(dev)

    # §15.6 pos_weight from the TRAIN set only. Without it the model predicts empty
    # everywhere, reports a beautiful loss, and produces an empty grid.
    occ = float(V[tr_t].mean())
    pos_weight = torch.tensor((1 - occ) / max(occ, 1e-8), device=dev)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0 if cfg.overfit else cfg.weight_decay)
    lam = cfg.lam

    print(f"[train] run={cfg.run} dataset={cfg.dataset} device={dev} "
          f"views={cfg.n_views} latent={cfg.latent_dim} agg={cfg.aggregator}")
    print(f"  parts train={len(tr)} test={len(te)} classes={len(classes)} res={res}^3")
    print(f"  occupancy={occ*100:.2f}%  pos_weight={float(pos_weight):.1f}  "
          f"trainable={model.n_trainable()/1e6:.2f}M" + ("  [OVERFIT GATE]" if cfg.overfit else ""))

    hist = []
    t0 = time.perf_counter()
    for ep in range(cfg.epochs):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), device=dev)]
        sums = {"cls": 0.0, "rec": 0.0, "n": 0}
        for i in range(0, len(perm), cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            z, logits, vox = model(F[b])
            l_cls = ce(logits, Y[b]) if use_cls else torch.zeros((), device=dev)
            l_rec = bce(vox, V[b]) if use_dec else torch.zeros((), device=dev)

            # §15.6: lambda from the loss ratio at initialisation, measured once.
            if cfg.run == "C" and lam is None:
                lam = float(l_cls.detach()) / max(float(l_rec.detach()), 1e-8)
                print(f"  lambda = L_cls(0)/L_recon(0) = {lam:.4f}")

            loss = l_cls + (lam or 1.0) * l_rec if cfg.run == "C" else (l_cls + l_rec)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sums["cls"] += float(l_cls.detach()) * len(b)
            sums["rec"] += float(l_rec.detach()) * len(b)
            sums["n"] += len(b)

        # ALWAYS log both terms separately (§15.6): a falling total with a flat L_cls
        # means the decoder is eating the whole gradient, invisible in the sum.
        # Evaluating every epoch is a large share of runtime, so do it every
        # `eval_every` epochs and always on the last one.
        last = ep == cfg.epochs - 1
        rec = (evaluate(model, F, V, Y, te_t, cfg, use_cls, use_dec)
               if (last or ep % cfg.eval_every == 0) else {})
        hist.append({"epoch": ep, "l_cls": sums["cls"] / sums["n"], "l_rec": sums["rec"] / sums["n"], **rec})
        if rec and (ep % max(1, cfg.epochs // 10) == 0 or last):
            msg = f"  ep {ep:3}  L_cls {hist[-1]['l_cls']:.4f}  L_rec {hist[-1]['l_rec']:.4f}"
            if use_cls:
                msg += f"  acc {rec['acc']*100:5.1f}%"
            if use_dec:
                msg += f"  IoU {rec['iou']:.4f}"
            print(msg)

    elapsed = time.perf_counter() - t0
    final = hist[-1]
    assert ("iou" in final) or not use_dec, "last epoch must be evaluated"

    out = {
        "config": vars(cfg), "seed": cfg.seed, "device": str(dev),
        "n_train": len(tr), "n_test": len(te), "classes": classes,
        "voxel_res": res, "occupancy": occ, "pos_weight": float(pos_weight),
        "lam": lam, "elapsed_sec": round(elapsed, 1),
        "trainable_params": model.n_trainable(),
        "history": hist, **{f"final_{k}": v for k, v in final.items()},
    }

    # per-class IoU: one average would hide that thin classes behave differently
    if use_dec:
        model.eval()
        with torch.no_grad():
            _, _, vx = model(F[te_t])
            per = iou(vx, V[te_t]).cpu().numpy()
        out["iou_by_class"] = by_group(per, labels_all[te], classes)
        out["iou_best_thresh"], out["iou_best_thresh_value"] = best_threshold_iou(vx, V[te_t])[::-1]

    # Q2: linear probe on the frozen latent (§15.7). Runs for every run type -- for B
    # this is the cell that matters, since B never saw a label.
    model.eval()
    with torch.no_grad():
        z_tr = model.encode(F[tr_t]).cpu().numpy()
        z_te = model.encode(F[te_t]).cpu().numpy()
    if not cfg.overfit:
        out["probe"] = linear_probe(z_tr, labels_all[tr], z_te, labels_all[te], cfg.seed)
        print(f"  linear probe on z: {out['probe']['probe_acc']*100:.1f}%")

    RESULTS.mkdir(exist_ok=True)
    name = cfg.name or f"{cfg.run}_v{cfg.n_views}_z{cfg.latent_dim}_{cfg.aggregator}" + ("_overfit" if cfg.overfit else "")
    path = RESULTS / f"run_{name}.json"

    # Save weights so plot 2 and demo.py can reconstruct without retraining.
    ckpt = RESULTS / f"ckpt_{name}.pt"
    torch.save({"state_dict": model.state_dict(), "config": vars(cfg),
                "classes": classes, "res": res, "view_idx": vidx.tolist(),
                "test_idx": te.tolist(), "train_idx": tr.tolist()}, ckpt)
    out["checkpoint"] = str(ckpt)
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"  elapsed {elapsed:.1f}s -> wrote {path}")

    if cfg.overfit and use_dec:
        ok = final["iou"] >= cfg.gate_iou
        print(f"\n  OVERFIT GATE: IoU {final['iou']:.4f} vs required {cfg.gate_iou} -> "
              f"{'PASS - decoder can express this geometry, proceed' if ok else 'FAIL - STOP AND FIX (§3.5)'}")
        out["gate_pass"] = bool(ok)
        path.write_text(json.dumps(out, indent=2, default=str))
    return out


@torch.no_grad()
def evaluate(model, F, V, Y, idx, cfg, use_cls, use_dec) -> dict:
    model.eval()
    out = {}
    _, logits, vox = model(F[idx])
    if use_cls:
        out["acc"] = accuracy(logits, Y[idx])
    if use_dec:
        out["iou"] = float(iou(vox, V[idx]).mean())
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="A", choices=["A", "B", "C"])
    p.add_argument("--dataset", default="cadnet")
    p.add_argument("--name", default=None)
    p.add_argument("--n-views", dest="n_views", type=int, default=20)
    p.add_argument("--latent-dim", dest="latent_dim", type=int, default=256)
    p.add_argument("--aggregator", default="max", choices=["max", "attn"])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-4)
    p.add_argument("--val-frac", dest="val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    # MEASURED, not assumed (§18): the decoder is small enough that MPS dispatch
    # overhead dominates -- CPU is 2.4x faster (10.7s vs 25.7s, identical IoU to
    # 3 decimal places). Cache building still uses MPS, where ResNet18 wins.
    p.add_argument("--device", default="cpu",
                   help="cpu is ~2.4x faster than mps for this decoder; use auto to override")
    p.add_argument("--overfit", type=int, default=0, help="§3.5 gate: train==eval on N parts")
    p.add_argument("--gate-iou", dest="gate_iou", type=float, default=0.95)
    p.add_argument("--no-dedup", dest="dedup", action="store_false", default=True,
                   help="disable duplicate-aware splitting (NOT recommended: inflates every metric)")
    p.add_argument("--dedup-thresh", dest="dedup_thresh", type=float, default=0.999)
    p.add_argument("--eval-every", dest="eval_every", type=int, default=5,
                   help="evaluate every N epochs (always on the last)")
    p.add_argument("--lam", default=None, help="run C lambda; 'auto' = L_cls(0)/L_recon(0)")
    cfg = p.parse_args(argv)
    cfg.lam = None if cfg.lam in (None, "auto") else float(cfg.lam)
    run_training(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
