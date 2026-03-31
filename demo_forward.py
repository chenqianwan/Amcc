#!/usr/bin/env python3
"""
Complete forward-only demo: 32×32 RGB → MiniViT → shapes + figures.

Run with no arguments for the original default (auto image, seed 42, ./outputs/).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from mini_vit import MiniViT, set_seed
from mini_vit.utils import (
    DEMO_IMAGE_TYPES,
    prepare_demo_image_32x3x32,
    print_forward_shapes,
    save_attention_cls_patches_gridded,
    save_attention_cls_patches_only_figure,
    save_attention_head0_cls_to_patches,
    save_attention_mean_cls_to_patches,
    save_cls_attention_overlay_on_image,
    save_input_image,
    save_logits_bar_chart,
    save_patch_grid,
)

IMG_SIZE = 32
PATCH_SIZE = 4
# First transformer block is clearest for teaching (earlier layer).
ATTENTION_BLOCK_INDEX = 0


def _out_name(prefix: str, basename: str) -> str:
    return f"{prefix}{basename}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Mini Vision Transformer — forward-only demo. "
            "One forward pass on a 32×32 RGB image; saves PNGs and optional shape table."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for weights and synthetic image (default: 42)",
    )
    p.add_argument(
        "--image-type",
        choices=list(DEMO_IMAGE_TYPES),
        default="auto",
        metavar="TYPE",
        help=(
            "Demo input: auto = try torchvision FakeData else procedural wave; "
            "or random / checkerboard / stripes / gradient (all synthetic, no download)"
        ),
    )
    p.add_argument(
        "--save-prefix",
        default="",
        help="Optional string prepended to each output filename (e.g. exp1_ → exp1_input_image.png)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write PNGs (default: outputs/ next to this script)",
    )
    p.add_argument(
        "--print-shapes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print the forward tensor shape table (default: true; use --no-print-shapes to skip)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (ROOT / "outputs").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    x, source = prepare_demo_image_32x3x32(
        seed=args.seed,
        device=device,
        image_type=args.image_type,
    )
    grid = IMG_SIZE // PATCH_SIZE
    prefix = args.save_prefix
    b = ATTENTION_BLOCK_INDEX

    model = MiniViT().to(device)
    model.eval()

    with torch.no_grad():
        logits, z = model(x)

    if args.print_shapes:
        print_forward_shapes(logits, z, source=source)

    save_input_image(x[0].cpu(), out_dir / _out_name(prefix, "input_image.png"))
    save_patch_grid(
        x[0].cpu(),
        PATCH_SIZE,
        out_dir / _out_name(prefix, "patch_grid.png"),
        title=f"Input + {PATCH_SIZE}×{PATCH_SIZE} patch grid",
    )

    attn_block = z[f"attn_block_{b}"].cpu()
    num_heads = attn_block.per_head.shape[1]

    save_attention_head0_cls_to_patches(
        attn_block,
        (grid, grid),
        out_dir / _out_name(prefix, f"attention_block{b}_head0.png"),
        block_index=b,
        num_heads=num_heads,
    )
    save_attention_mean_cls_to_patches(
        attn_block,
        (grid, grid),
        out_dir / _out_name(prefix, f"attention_block{b}_mean_heads.png"),
        block_index=b,
        num_heads=num_heads,
    )
    save_attention_cls_patches_only_figure(
        attn_block,
        (grid, grid),
        out_dir / _out_name(prefix, "attention_cls_to_patches_only.png"),
        block_index=b,
        num_heads=num_heads,
    )
    save_attention_cls_patches_gridded(
        attn_block,
        (grid, grid),
        out_dir / _out_name(prefix, f"attention_block{b}_patch_grid_heatmap.png"),
        block_index=b,
        num_heads=num_heads,
    )
    save_cls_attention_overlay_on_image(
        x[0].cpu(),
        attn_block,
        (grid, grid),
        PATCH_SIZE,
        out_dir / _out_name(prefix, f"attention_block{b}_overlay_on_input.png"),
        block_index=b,
    )

    save_logits_bar_chart(
        logits.cpu(),
        out_dir / _out_name(prefix, "logits_bar_chart.png"),
    )

    print(f"Wrote demo artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
