# mvae — multiview encoder + voxel decoder

Does the latent a multiview encoder produces actually *contain* the part's geometry, or has
it silently thrown most of it away? To test invertibility you need an inverse, so this
builds one: the same encoder, plus a decoder that turns the latent back into a 3D volume.

```
STEP/STL → mesh → 20 depth views → ResNet18 (frozen, shared) → aggregate → z ∈ R²⁵⁶
                                                                    │
                                              ┌─────────────────────┴──────┐
                                              ↓                            ↓
                                       class logits              32³ occupancy grid
```

**Both heads read only `z`.** Neither sees images, neither sees the mesh. The mesh is a
*label*, not an input — a voxel grid is just a very high-dimensional label. At inference:
views in, `z` out.

| | question | answered by |
|---|---|---|
| **Q1** | reconstruct 3D from multiview with low error? | Run B's IoU + the view sweep |
| **Q2** | classify from that same latent? | Run B's linear probe, Run A as ceiling |
| **Q3** | does training both together help either? | Run C vs Run A |

---

## Setup

```bash
conda create -n mvae python=3.11 -y && conda activate mvae
pip install torch torchvision trimesh numpy scipy scikit-learn matplotlib
pip install embreex rtree          # REQUIRED: 380x faster voxelization, see NOTES.md
pip install "pyglet<2"             # optional: interactive viewer
pip install cadquery-ocp           # optional: STEP loading for demo.py
```

`embreex` is not optional. Without it, exact point-in-solid testing runs at 7.63 s/part
instead of 0.02 s — the difference between 4 hours and 40 seconds for the dataset.

---

## Run it, in order

Each stage writes JSON **and** a PNG to `results/`, so plots regenerate without retraining
and most bugs are visible instantly (§18).

```bash
# 0. does ConvTranspose3d work on this machine?  (§16 trap 1)
python src/smoke_mps.py

# 1. one mesh end to end: normalize → watertight gate → voxelize → picture
python src/mesh.py data/CADNET_3317/Nuts/0001.stl

# 2. 20 depth-map views as a 4x5 grid
python src/render.py data/CADNET_3317/Nuts/0001.stl

# 3. build the cache ONCE (~40 min for 2127 parts). This is what makes the week fit.
python src/cache.py --dataset cadnet --limit 50

# 4. Run A — classifier only, cross-entropy
python src/train.py --run A --name A

# 5. THE GATE (§3.5). Must pass before any number is believed.
python src/train.py --run B --overfit 20 --epochs 200 --name gate

# 6. Run B — autoencoder only, NO labels used at all
python src/train.py --run B --name B

# 7. sweeps → plots 1 and 3
python src/sweep.py all

# 8. plot 2 — where the error lives (needs a Run B checkpoint)
python src/recon_viz.py results/ckpt_B.pt --n 4

# 9. latency, batch 1 vs 64, 12 vs 20 views
python src/bench.py results/ckpt_B.pt

# 10. the acceptance test
python src/demo.py ../resources/500-1212.step --ckpt results/ckpt_B.pt

# 11. Phase 2, ONLY if 0-10 are clean
python src/train.py --run C --lam auto --name C
```

### Looking at things

```bash
python src/viz.py <mesh> --voxels        # interactive: mesh | voxels
python src/viz.py <mesh> --cameras       # the 20 viewpoints, in place
python src/viz.py --cls Nuts --n 4       # browse a class
python src/viewmesh.py <mesh>            # what voxel resolution costs
python src/voxcompare.py <mesh>          # exact vs conservative, with true outline
python src/thincheck.py                  # what dilation does to thin walls
```

---

## The gate that makes the numbers trustworthy

A bad reconstruction number is ambiguous between *"the views lost the geometry"* (the
finding) and *"my decoder is broken"* (bad engineering). Reporting the first when the truth
is the second is worse than having no result.

**So: `--overfit 20` trains on 20 parts and evaluates on the same 20.** Memorising 20 grids
should be trivial. IoU ≈ 0.95 → the decoder can express this geometry, so a poor score on
the full set is a real information limit. Can't memorise → something is broken; stop and fix
before generating any number.

Like calibrating a thermometer in boiling water: you know it should read 100°C, so 40°C
means you fix the thermometer, not publish the measurement.

---

## Things that will bite, and where they are handled

| trap | where |
|---|---|
| `ConvTranspose3d` unimplemented on MPS | `smoke_mps.py`, run it first |
| normalization drift between views and voxels | `mesh.normalize()` — defined once, imported everywhere |
| voxel grid not exactly 32³ | asserted in `mesh.voxelize()` |
| missing `pos_weight` → empty reconstructions that look like a finding | `train.py`, computed from the train split |
| depth maps that look like noise | 200k surface points + 3×3 splat; always eyeball the PNG |
| non-watertight meshes → hollow voxels | gated, repaired, and the **pass rate is reported** |
| seeds | set for torch/numpy/random, written into every results JSON |

**Never report loss as a headline metric. Report IoU.** With `pos_weight`, a beautifully
falling BCE is entirely compatible with a completely empty reconstruction.

See `NOTES.md` for the implementer's log: measured findings, deviations from the spec, and
the bugs that were caught along the way.
