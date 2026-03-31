#!/usr/bin/env python3
"""
Lightweight self-check for the Mini ViT project (no pytest).

Runs tensor/shape assertions on the model, then runs demo_forward.py into a
temporary directory and verifies expected PNGs exist and are non-empty.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from mini_vit import AttentionWeights, MiniViT, set_seed
from mini_vit.utils import prepare_demo_image_32x3x32

# Must stay in sync with demo_forward.py defaults / outputs
IMG_SIZE = 32
PATCH_SIZE = 4
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 64
SEQ_LEN = 1 + NUM_PATCHES  # CLS + patches
NUM_HEADS = 4
NUM_CLASSES = 10
EMBED_DIM = 64
DEPTH = 2

EXPECTED_DEMO_PNGS = [
    "input_image.png",
    "patch_grid.png",
    "attention_block0_head0.png",
    "attention_block0_mean_heads.png",
    "attention_cls_to_patches_only.png",
    "attention_block0_patch_grid_heatmap.png",
    "attention_block0_overlay_on_input.png",
    "logits_bar_chart.png",
]


def err(msg: str) -> None:
    print(f"ERROR: {msg}")
    sys.exit(1)


def check(name: str, ok: bool, detail: str) -> None:
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        err(f"check failed: {name}")


def run_model_checks() -> None:
    print("Model & forward pass")
    set_seed(0)
    device = torch.device("cpu")
    x, _ = prepare_demo_image_32x3x32(
        seed=0, device=device, image_type="random", prefer_fake_data=False
    )

    check(
        "input tensor shape (B,3,H,W)",
        tuple(x.shape) == (1, 3, IMG_SIZE, IMG_SIZE),
        str(tuple(x.shape)),
    )

    model = MiniViT().eval()
    check(
        "positional embedding length == num_patches",
        model.pos_embed.shape == (1, NUM_PATCHES, EMBED_DIM),
        f"pos_embed {tuple(model.pos_embed.shape)}, expect (1,{NUM_PATCHES},{EMBED_DIM})",
    )

    with torch.no_grad():
        logits, z = model(x)

    pt = z["patch_tokens"]
    check(
        "patch token count (N patches)",
        pt.shape == (1, NUM_PATCHES, EMBED_DIM),
        f"patch_tokens {tuple(pt.shape)}",
    )

    tok = z["tokens_after_cls_pos"]
    check(
        "CLS + patches sequence length",
        tok.shape == (1, SEQ_LEN, EMBED_DIM),
        f"tokens_after_cls_pos {tuple(tok.shape)} (expect seq {SEQ_LEN}=1+{NUM_PATCHES})",
    )

    check(
        "logits shape",
        tuple(logits.shape) == (1, NUM_CLASSES),
        str(tuple(logits.shape)),
    )

    for i in range(DEPTH):
        key = f"attn_block_{i}"
        aw = z[key]
        check(
            f"{key} is AttentionWeights",
            isinstance(aw, AttentionWeights),
            type(aw).__name__,
        )
        if isinstance(aw, AttentionWeights):
            ph = aw.per_head.shape
            mh = aw.mean_over_heads.shape
            exp_ph = (1, NUM_HEADS, SEQ_LEN, SEQ_LEN)
            exp_mh = (1, SEQ_LEN, SEQ_LEN)
            check(
                f"{key}.per_head shape",
                ph == exp_ph,
                f"{tuple(ph)} vs {exp_ph}",
            )
            check(
                f"{key}.mean_over_heads shape",
                mh == exp_mh,
                f"{tuple(mh)} vs {exp_mh}",
            )


def run_demo_output_checks() -> None:
    print("\nDemo script & output files")
    with tempfile.TemporaryDirectory(prefix="mini_vit_validate_") as tmp:
        tmp_path = Path(tmp)
        cmd = [
            sys.executable,
            str(ROOT / "demo_forward.py"),
            "--no-print-shapes",
            "--output-dir",
            str(tmp_path),
            "--image-type",
            "random",
            "--seed",
            "0",
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            err(f"demo_forward.py exited with code {proc.returncode}")

        for name in EXPECTED_DEMO_PNGS:
            path = tmp_path / name
            exists = path.is_file()
            size = path.stat().st_size if exists else 0
            check(
                f"PNG created: {name}",
                exists and size > 0,
                f"{size} bytes" if exists else "missing",
            )


def main() -> None:
    print("Mini ViT — validate_project.py\n")
    run_model_checks()
    run_demo_output_checks()
    print("\nAll checks passed. Model shapes, attention layout, and demo outputs look good.")


if __name__ == "__main__":
    main()
