"""§17 stage 1 / §16 trap 1: does ConvTranspose3d run forward+backward on MPS?

Historically flaky on Apple Silicon. Run this before building anything around it.
Exits non-zero if the decoder stack cannot run on the resolved device, and prints
the fallback to put in configs/default.py (decoder_device="cpu").
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from device import resolve_device, sync  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def decoder_stack(latent_dim: int = 256) -> nn.Module:
    """Exactly the §15.4 decoder head, so the smoke test tests the real thing."""
    return nn.Sequential(
        nn.Linear(latent_dim, 256 * 4 * 4 * 4),
        nn.Unflatten(1, (256, 4, 4, 4)),
        nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),  # 8^3
        nn.BatchNorm3d(128),
        nn.ReLU(inplace=True),
        nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),   # 16^3
        nn.BatchNorm3d(64),
        nn.ReLU(inplace=True),
        nn.ConvTranspose3d(64, 1, kernel_size=4, stride=2, padding=1),     # 32^3 logits
    )


def try_device(dev: torch.device, latent_dim: int = 256, batch: int = 8) -> dict:
    out = {"device": str(dev), "ok": False, "error": None}
    try:
        net = decoder_stack(latent_dim).to(dev)
        z = torch.randn(batch, latent_dim, device=dev)
        target = (torch.rand(batch, 1, 32, 32, 32, device=dev) > 0.9).float()

        def step():
            net.zero_grad(set_to_none=False)
            logits = net(z)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            loss.backward()
            return logits, loss

        # First call on MPS pays kernel compilation; warm up so the number is honest.
        sync(dev)
        t0 = time.perf_counter()
        step()
        sync(dev)
        cold = time.perf_counter() - t0

        for _ in range(4):
            step()
        sync(dev)
        t0 = time.perf_counter()
        logits, loss = step()
        sync(dev)
        dt = time.perf_counter() - t0

        assert logits.shape == (batch, 1, 32, 32, 32), f"bad shape {tuple(logits.shape)}"
        grad = net[0].weight.grad
        assert grad is not None, "no gradient reached the first Linear"
        assert torch.isfinite(grad).all(), "non-finite gradient"
        assert torch.isfinite(loss), "non-finite loss"

        out.update(
            ok=True,
            out_shape=list(logits.shape),
            loss=float(loss.detach()),
            grad_absmean=float(grad.abs().mean()),
            fwd_bwd_ms=round(dt * 1000, 2),
            first_call_ms=round(cold * 1000, 2),
        )
    except Exception as e:  # no silent fallbacks (§18) -- record and report
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def main() -> int:
    dev = resolve_device("auto")
    print(f"torch {torch.__version__} | resolved device: {dev}")

    report = {"torch": torch.__version__, "resolved_device": str(dev), "attempts": []}

    primary = try_device(dev)
    report["attempts"].append(primary)
    print(f"  {dev}: {'PASS' if primary['ok'] else 'FAIL'}  {primary if not primary['ok'] else ''}")
    if primary["ok"]:
        print(f"    out={primary['out_shape']} loss={primary['loss']:.4f} "
              f"grad|.|={primary['grad_absmean']:.3e} "
              f"fwd+bwd={primary['fwd_bwd_ms']}ms (first call {primary['first_call_ms']}ms)")

    cpu = try_device(torch.device("cpu"))
    report["attempts"].append(cpu)
    print(f"  cpu: {'PASS' if cpu['ok'] else 'FAIL'}  {cpu if not cpu['ok'] else ''}")
    if cpu["ok"]:
        print(f"    fwd+bwd={cpu['fwd_bwd_ms']}ms")

    if primary["ok"]:
        report["decoder_device"] = str(dev)
        verdict = f"ConvTranspose3d works on {dev}. No fallback needed."
    elif cpu["ok"]:
        report["decoder_device"] = "cpu"
        verdict = ("ConvTranspose3d FAILED on the accelerator but works on CPU. "
                   'Set decoder_device="cpu" in configs/default.py and note it in NOTES.md.')
    else:
        report["decoder_device"] = None
        verdict = "ConvTranspose3d failed on BOTH devices. Substitute Upsample(2)+Conv3d (§16 trap 1)."

    report["verdict"] = verdict
    print(f"\nVERDICT: {verdict}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "smoke_mps.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {RESULTS / 'smoke_mps.json'}")

    return 0 if report["decoder_device"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
