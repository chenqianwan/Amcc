#!/usr/bin/env python3
"""
Compare MiniViT forward behavior on several synthetic 32×32 inputs (no training).

Writes outputs/compare/comparison.png and prints a short numeric summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np
import torch

from mini_vit import MiniViT, set_seed
from mini_vit.utils import (
    _chw01_to_hwc_rgb,
    _savefig_png,
    prepare_demo_image_32x3x32,
)

SEED = 42
IMG_SIZE = 32
PATCH_SIZE = 4
GRID = IMG_SIZE // PATCH_SIZE  # 8
# At least three distinct synthetic patterns (no FakeData / no download)
INPUT_SPECS: list[tuple[str, str]] = [
    ("checkerboard", "Checkerboard"),
    ("stripes", "Horizontal stripes"),
    ("gradient", "RGB gradient"),
]


def mean_cls_patch_heat_8x8(z: dict, block_index: int = 0) -> np.ndarray:
    """Mean-over-heads attention: CLS (row 0) → patch keys, reshaped to 8×8."""
    aw = z[f"attn_block_{block_index}"]
    row = aw.mean_over_heads[0, 0, 1:].detach().cpu().numpy()
    return row.reshape(GRID, GRID)


def main() -> None:
    set_seed(SEED)
    device = torch.device("cpu")
    out_dir = (ROOT / "outputs" / "compare").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = MiniViT().eval()

    rows: list[tuple[str, str, torch.Tensor, np.ndarray, np.ndarray]] = []

    for image_type, short_title in INPUT_SPECS:
        x, source = prepare_demo_image_32x3x32(
            seed=SEED,
            device=device,
            image_type=image_type,
            prefer_fake_data=False,
        )
        with torch.no_grad():
            logits, z = model(x)
        heat = mean_cls_patch_heat_8x8(z, block_index=0)
        lv = logits[0].detach().float().cpu().numpy()
        rows.append((short_title, source, x, heat, lv))

    n = len(rows)
    fig, axes = plt.subplots(3, n, figsize=(3.6 * n, 8.2))
    if n == 1:
        axes = axes.reshape(3, 1)

    fig.suptitle(
        "MiniViT comparison (same seed, untrained weights)\n"
        "Inputs → block-0 CLS→patch attention (mean heads) → logits",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    for j, (title, _src, x, heat, lv) in enumerate(rows):
        axes[0, j].imshow(
            _chw01_to_hwc_rgb(x[0]),
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        axes[0, j].set_aspect("equal")
        axes[0, j].set_title(title, fontsize=11)
        axes[0, j].set_xticks([])
        axes[0, j].set_yticks([])

        im = axes[1, j].imshow(
            heat,
            cmap="magma",
            origin="upper",
            interpolation="nearest",
        )
        axes[1, j].set_aspect("equal")
        axes[1, j].set_title("Attention (block 0, mean heads)", fontsize=10)
        axes[1, j].set_xlabel("patch col")
        if j == 0:
            axes[1, j].set_ylabel("patch row")
        plt.colorbar(im, ax=axes[1, j], fraction=0.046, pad=0.04)

        axes[2, j].bar(
            np.arange(len(lv)),
            lv,
            color="steelblue",
            edgecolor="black",
            linewidth=0.4,
        )
        axes[2, j].set_title("Logits (10 classes)", fontsize=10)
        axes[2, j].set_xlabel("class")
        axes[2, j].set_xticks(np.arange(10))
        axes[2, j].axhline(0.0, color="gray", linewidth=0.6, linestyle="--")
        if j == 0:
            axes[2, j].set_ylabel("value")

    fig.tight_layout()
    out_png = out_dir / "comparison.png"
    _savefig_png(fig, out_png)
    plt.close(fig)

    print("compare_inputs.py — summary (deterministic, seed=%d)" % SEED)
    print("-" * 60)
    for title, _src, _x, heat, lv in rows:
        am = int(lv.argmax())
        l2 = float(np.linalg.norm(lv))
        print(
            f"  {title:22s}  logits: argmax={am}  L2={l2:.3f}  "
            f"range=[{lv.min():+.3f},{lv.max():+.3f}]  |  "
            f"CLS→patch attn: [{heat.min():.4f}, {heat.max():.4f}]"
        )
    print("-" * 60)
    print(
        "Same architecture and weights; differences come only from the input "
        "tensor (different patterns → different activations, attention, and logits)."
    )
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
