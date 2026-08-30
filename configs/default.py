"""Single config dataclass. Every run is a variation of this (§13: no Hydra, no Lightning)."""

from dataclasses import dataclass, asdict, field


@dataclass
class Config:
    # --- run identity -----------------------------------------------------
    name: str = "default"          # results/run_<name>.json
    run: str = "A"                 # "A" cls-only | "B" recon-only | "C" joint
    seed: int = 0                  # written into every results JSON (§16 trap 8)

    # --- data -------------------------------------------------------------
    dataset: str = "modelnet10"    # "modelnet10" | "cadnet" | "mcb"
    data_root: str = "data"
    n_views: int = 20              # sweep {1,3,6,12,20}; slices cached feats axis 1
    image_size: int = 128
    n_surface_points: int = 200_000  # §15.2: 200k, not 50k
    voxel_res: int = 32            # §15.3

    # --- model (§15.4) ----------------------------------------------------
    backbone: str = "resnet18"     # frozen, features cached (§3.4)
    aggregator: str = "max"        # "max" (MVCNN) | "attn" (GMViT-lite)
    feat_dim: int = 512            # ResNet18 penultimate
    latent_dim: int = 256          # sweep {64, 256, 1024} (§3.5 check 2)

    # --- training ---------------------------------------------------------
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_frac: float = 0.2

    # --- losses (§15.6) ---------------------------------------------------
    pos_weight: float | None = None  # None -> computed from train set (#empty/#occupied)
    lam: float | None = None         # Run C only; None -> L_cls(0)/L_recon(0) at init

    # --- device -----------------------------------------------------------
    device: str = "auto"             # resolved once; never .cuda() (§13)
    decoder_device: str | None = None  # §16 trap 1 escape hatch: "cpu" if ConvTranspose3d fails on MPS

    # --- overfit gate (§3.5 check 1) --------------------------------------
    overfit_n: int = 20              # train == eval on these, IoU must hit ~0.95

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT = Config()
