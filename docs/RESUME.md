# RESUME — start here after the reboot

**Status: Q1 and Q2 are answered and defensible. Phase 1 is complete.**
§9 requires exactly this by end of day 5, so the week is on track. Everything below is
optional upside.

---

## 1. Get back into the environment

```bash
conda activate mvae
cd /Users/rebecca.cervasio/Downloads/papers/project/mvae
python -c "import torch,trimesh,embreex; print('ok', torch.__version__)"
```

Env is `mvae` (Python 3.11). If anything is missing:
`pip install torch torchvision trimesh numpy scipy scikit-learn matplotlib embreex rtree "pyglet<2"`

**`embreex` is not optional** — without it voxelization runs 380× slower.

---

## 2. Nothing needs recomputing

The expensive artifacts are all on disk:

| artifact | cost to rebuild | status |
|---|---|---|
| `data/cache_cadnet.npz` (35 MB, 2127 parts) | **33 min** | ✅ keep |
| `data/CADNET_3317/` (990 MB) | download | ✅ keep |
| `data/MCB_A.tar.gz` / `MCB_B.tar.gz` (4 GB) | download | ✅ unused, week 2 |
| `results/*.json` (36 runs) + `ckpt_*.pt` | hours | ✅ keep |
| `results/*.png` (12 plots) | seconds | ✅ regenerable |

Regenerate any plot without retraining: `python src/sweep.py plot` · `python src/table.py --png`

---

## 3. What the numbers say (full detail in NOTES.md)

| run | trained acc | linear probe | recon IoU |
|---|---|---|---|
| **A** — classifier only | **91.0%** | 89.8% | — |
| **B** — autoencoder only, *never saw a label* | — | **88.2%** | **0.743** |

- **Overfit gate PASSED at IoU 1.0000** → 0.743 is a real information limit, not a weak decoder.
- **Q2 lands well:** a latent trained purely to reconstruct probes 2.8 pts behind supervised.
- **The headline:** reconstruction IoU responds to neither view count (20×, effect = 0.63×
  seed noise) nor latent size (16×, non-monotonic). **One view reconstructs as well as
  twenty** — so the decoder is doing *category-level shape completion*, not geometric
  reconstruction.
- **Classification does benefit:** +5.1 pts from 1→6 views, then saturates (5 seeds, ±1.2).
- **Occlusion premise tested:** only **0.2%** of surface is invisible to all 20 views. Error
  lives at **boundaries** (19.3% at the surface → 0.0% six cells deep), not "inside".

---

## 4. Next steps, in priority order

### (a) Latency table — 5 minutes, needed for §0.8's "GPU or CPU?" question
```bash
python src/bench.py results/ckpt_B.pt
```
Writes `results/bench.json`. Note it deliberately also times **rendering + the frozen
ResNet18**, which the cache hides — an end-to-end latency claim that omits those would be
dishonest, since deployment cannot cache them.

### (b) STEP demo — the acceptance test. Budget half a day (§16 trap 7)
```bash
pip install cadquery-ocp          # NOT yet installed; this is the risky step
python src/demo.py ../resources/500-1212.step --ckpt results/ckpt_B.pt
```
The file is **AP242, 37 solids, 200 cylindrical surfaces** — an assembly, so `demo.py` loops
over parts. If the install fights back, the documented fallback is to convert the STEP to
STL by hand and pass that; `demo.py` accepts `.stl` directly.

**Say up front:** the demo makes **no classification claim** — its parts are out of the
training taxonomy. The decoder works regardless of taxonomy; that is what it shows.

### (c) Run C / Phase 2 — Q3, only if (a) and (b) are clean
```bash
python src/train.py --run C --lam auto --epochs 60 --name C
python src/table.py --png      # fills in the C row, computes Q3 = C - A
```
λ is measured as `L_cls(0)/L_recon(0)` at init and both loss terms are logged separately.
Three λ values max, hard cap (§3.3). **Abandon out loud if it does not converge** — that is
a reportable outcome, and Q1/Q2 are already banked.

### (d) Optional tightening
- Run B seeds to full 60 epochs (currently stopped at 30): the verdict will not change,
  but the noise floor gets tighter.
  ```bash
  for v in 1 20; do for s in 0 1 2 3; do
    python src/train.py --run B --n-views $v --epochs 60 --seed $s --name seedB_v${v}_s${s}
  done; done
  ```
  **Run at most 2–3 in parallel** — 8 concurrent jobs drained the battery in ~30 min.
- Error bars on plot 1 (needs the above).
- Multi-seed Run B across all 5 view counts, if the flat curve is going on a slide.

---

## 5. Open decisions for Sai (§0.8)

Unchanged, plus two new ones the data raised:

1. GPU or CPU deployment target — **now answerable with our own numbers** after step (a).
2. Voxels the right decode target, given C-Infinity's differentiable rendering?
3. Which downstream tasks share a latent (the §3.7 invariance conflict).
4. Internal labelled data, or public for now?
5. **NEW — CADNET contains ~19% duplicates.** Does C-Infinity's internal data too? A random
   split over duplicated parts inflates every metric; 1-NN scored 95.5% before the fix.
6. **NEW — normalization destroys absolute scale**, so an M6 and an M8 bolt become the
   *same input*. 14% of the duplicate groups are pure scale variants. That is the
   good-recall/bad-precision signature, visible in our own pipeline.

---

## 6. Things to be careful about (learned the hard way — see NOTES.md)

- **Never trust a number that does not respond sensibly to the thing you varied.** Three
  measurement bugs were caught this way: a silhouette test that returned the same 0.061 for
  three different shapes; a bore probe that was non-monotonic in bore size; a thickness
  metric that could not tell a 1-cell plate from a 2-cell one.
- **Check for duplicate leakage before believing any accuracy.** 1-NN beating the trained
  model is the tell.
- **Single-seed differences are not results.** Measured noise: ±1.2 pts (accuracy),
  ±0.010 (IoU).
- **Do not pipe long background jobs through `grep`** — it buffers and you lose the output.
  Write to a log and grep the file.
- **`pgrep -f "foo.py"` matches the waiter's own command line**, creating a shell that waits
  on itself forever. Two of those were left running for hours.
