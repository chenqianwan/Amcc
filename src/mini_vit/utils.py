"""Demo helpers: reproducibility, image prep, shape logging, figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .attention_viz import AttentionWeights


def _chw01_to_hwc_rgb(image_chw: torch.Tensor) -> np.ndarray:
    """(C,H,W) float in ~[0,1] -> contiguous (H,W,3) RGB for matplotlib."""
    x = image_chw.detach().float().cpu().clamp(0, 1).numpy()
    if x.ndim != 3:
        raise ValueError(f"Expected CHW image, got shape {x.shape}")
    c, h, w = x.shape
    if c == 1:
        x = np.repeat(x, 3, axis=0)
    elif c > 3:
        x = x[:3]
    img = np.transpose(x, (1, 2, 0))
    return np.ascontiguousarray(img)


def _savefig_png(fig: plt.Figure, out_path: Path) -> None:
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(out_path),
        format="png",
        dpi=120,
        bbox_inches="tight",
        pad_inches=0.05,
    )


DEMO_IMAGE_TYPES = ("auto", "random", "checkerboard", "stripes", "gradient")


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _procedural_wave_image(seed: int, device: torch.device) -> tuple[torch.Tensor, str]:
    g = torch.Generator()
    g.manual_seed(seed)
    base = torch.rand(1, 3, 32, 32, generator=g, dtype=torch.float32)
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, 32),
        torch.linspace(0, 1, 32),
        indexing="ij",
    )
    wave = 0.25 * (
        torch.sin(xx * 6.28 + base[0, 0, 0, 0] * 5)
        + torch.cos(yy * 6.28 + base[0, 1, 0, 0] * 5)
    ).unsqueeze(0).unsqueeze(0)
    x = (base + wave).clamp(0, 1).to(device)
    return x, "procedural synthetic (torch, seed=%d)" % seed


def _build_typed_synthetic(
    seed: int,
    image_type: str,
    device: torch.device,
) -> tuple[torch.Tensor, str]:
    """Deterministic 32×32 RGB in [0,1], shape (1,3,32,32). ``image_type`` not ``auto``."""
    if image_type not in DEMO_IMAGE_TYPES or image_type == "auto":
        raise ValueError(f"image_type must be one of {DEMO_IMAGE_TYPES!r} (not 'auto' here)")

    set_seed(seed)

    if image_type == "random":
        g = torch.Generator()
        g.manual_seed(seed)
        x = torch.rand(1, 3, 32, 32, generator=g, dtype=torch.float32).to(device)
        return x, f"random uniform RGB (seed={seed})"

    if image_type == "checkerboard":
        cell = 4
        torch.manual_seed(seed)
        a = torch.rand(3)
        b = torch.rand(3)
        img = torch.zeros(3, 32, 32)
        for i in range(0, 32, cell):
            for j in range(0, 32, cell):
                use_a = ((i // cell) + (j // cell)) % 2 == 0
                color = a if use_a else b
                img[:, i : i + cell, j : j + cell] = color.view(3, 1, 1)
        return img.unsqueeze(0).to(device), f"checkerboard cell={cell} (seed={seed})"

    if image_type == "stripes":
        torch.manual_seed(seed)
        c0 = torch.rand(3, 1, 1)
        c1 = torch.rand(3, 1, 1)
        band = 4
        img = torch.zeros(3, 32, 32)
        for y in range(32):
            img[:, y : y + 1, :] = c0 if (y // band) % 2 == 0 else c1
        return img.unsqueeze(0).to(device), f"horizontal stripes band={band} (seed={seed})"

    if image_type == "gradient":
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, 32),
            torch.linspace(0, 1, 32),
            indexing="ij",
        )
        r = xx
        g = yy
        b = 1.0 - (xx * 0.5 + yy * 0.5)
        phase = (seed % 997) / 997.0
        img = torch.stack(
            [
                (r + phase * 0.2).clamp(0, 1),
                (g + phase * 0.15).clamp(0, 1),
                (b * (0.5 + phase * 0.5)).clamp(0, 1),
            ],
            dim=0,
        )
        return img.unsqueeze(0).to(device), f"RGB gradient (seed={seed})"

    raise ValueError(f"unknown image_type: {image_type}")


def prepare_demo_image_32x3x32(
    seed: int = 42,
    prefer_fake_data: bool = True,
    device: torch.device | None = None,
    image_type: str = "auto",
) -> tuple[torch.Tensor, str]:
    """
    Build one 32×32 RGB image in [0, 1], shape (1, 3, 32, 32).

    - ``image_type='auto'`` (default): try torchvision FakeData, else procedural wave.
    - Other types: ``random``, ``checkerboard``, ``stripes``, ``gradient`` (no FakeData).
    """
    if device is None:
        device = torch.device("cpu")

    if image_type != "auto":
        if image_type not in DEMO_IMAGE_TYPES:
            raise ValueError(
                f"image_type must be one of {DEMO_IMAGE_TYPES}, got {image_type!r}"
            )
        return _build_typed_synthetic(seed, image_type, device)

    if prefer_fake_data:
        try:
            from torchvision.datasets import FakeData

            ds = FakeData(
                size=1,
                image_size=(3, 32, 32),
                num_classes=10,
                random_offset=seed,
            )
            img_u8, _ = ds[0]
            x = img_u8.unsqueeze(0).float().to(device) / 255.0
            return x, "torchvision FakeData (random_offset=%d)" % seed
        except Exception:
            pass

    return _procedural_wave_image(seed, device)


def print_forward_shapes(
    logits: torch.Tensor,
    intermediates: dict[str, torch.Tensor],
    source: str,
) -> None:
    """Print logits and intermediate tensors in a fixed, readable order."""
    order = [
        "x_in",
        "patch_tokens",
        "tokens_after_cls_pos",
        "attn_block_0",
        "tokens_after_block_0",
        "attn_block_1",
        "tokens_after_block_1",
        "tokens_after_final_norm",
        "cls_token",
        "logits",
    ]
    line = "-" * 56
    print(line)
    print("MiniViT forward pass — tensor shapes")
    print(f"  Demo image: {source}")
    print(line)
    print(f"  {'name':<28} {'shape':<26}")
    print(line)
    for key in order:
        if key not in intermediates:
            continue
        val = intermediates[key]
        if isinstance(val, AttentionWeights):
            print(f"  {key + '.per_head':<28} {str(tuple(val.per_head.shape)):<26}")
            print(
                f"  {key + '.mean_over_heads':<28} "
                f"{str(tuple(val.mean_over_heads.shape)):<26}"
            )
        else:
            t = val
            print(f"  {key:<28} {str(tuple(t.shape)):<26}")
    print(f"  {'logits (returned)':<28} {str(tuple(logits.shape)):<26}")
    print(line)


def save_input_image(
    image_chw: torch.Tensor,
    out_path: Path,
    title: str = "Demo input (32×32 RGB)",
) -> None:
    """Save (3, H, W) float [0,1] as RGB PNG."""
    img = _chw01_to_hwc_rgb(image_chw)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img, vmin=0, vmax=1, interpolation="nearest")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    _savefig_png(fig, out_path)
    plt.close(fig)


def save_patch_grid(
    image_chw: torch.Tensor,
    patch_size: int,
    out_path: Path,
    title: str = "32×32 with patch grid",
) -> None:
    """Overlay patch grid on the input image (CHW RGB, same layout as the model)."""
    img = _chw01_to_hwc_rgb(image_chw)
    h, w = img.shape[0], img.shape[1]
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(
            f"Image ({h},{w}) must be divisible by patch_size={patch_size}"
        )

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img, vmin=0, vmax=1, interpolation="nearest")
    ax.set_aspect("equal")
    ax.set_title(title)
    # Pixel boundaries for origin='upper': lines at ±0.5 around each index.
    for y in range(0, h + 1, patch_size):
        ax.axhline(y - 0.5, color="cyan", linewidth=0.8, alpha=0.9)
    for x in range(0, w + 1, patch_size):
        ax.axvline(x - 0.5, color="cyan", linewidth=0.8, alpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    _savefig_png(fig, out_path)
    plt.close(fig)


def _coerce_attention_weights(
    w: AttentionWeights | torch.Tensor,
) -> AttentionWeights:
    if isinstance(w, AttentionWeights):
        return w
    if isinstance(w, torch.Tensor) and w.dim() == 4:
        return AttentionWeights(per_head=w, mean_over_heads=w.mean(dim=1))
    raise TypeError(
        "Expected AttentionWeights or (B,H,L,L) tensor, got "
        f"{type(w).__name__}"
    )


def _cls_to_patch_heatmap_bll(
    attn_bll: torch.Tensor,
    grid_hw: tuple[int, int],
) -> np.ndarray:
    """
    CLS → patch keys heatmap from a single (B, L, L) attention matrix.

    Token 0 = CLS; columns 1: are patch tokens (ViT order).
    """
    gh, gw = grid_hw
    row = attn_bll[0, 0, 1:].detach().cpu().numpy()
    if row.size != gh * gw:
        raise ValueError(f"Expected {gh * gw} patch tokens, got {row.size}")
    return row.reshape(gh, gw)


def _upsample_patch_map_to_pixels(
    heat_gh_gw: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    """Repeat each patch cell to (gh*P, gw*P) for overlay on the RGB image."""
    return np.repeat(np.repeat(heat_gh_gw, patch_size, axis=0), patch_size, axis=1)


def save_attention_head_cls_to_patches(
    weights: AttentionWeights | torch.Tensor,
    grid_hw: tuple[int, int],
    out_path: Path,
    head_index: int,
    block_index: int = 0,
    num_heads: int | None = None,
) -> None:
    """
    Heatmap: one attention head, CLS query → patch keys only (8×8 for 32×32 / patch 4).
    """
    aw = _coerce_attention_weights(weights)
    if num_heads is None:
        num_heads = aw.per_head.shape[1]
    heat = _cls_to_patch_heatmap_bll(aw.select_head(head_index), grid_hw)
    gh, gw = grid_hw

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(heat, cmap="viridis", origin="upper", interpolation="nearest")
    ax.set_xticks(np.arange(gw))
    ax.set_yticks(np.arange(gh))
    ax.set_xlabel("Patch column (left → right)")
    ax.set_ylabel("Patch row (top → bottom)")
    ax.set_title(
        f"Transformer block {block_index}, head {head_index} (of {num_heads})\n"
        "CLS token → each image patch (attention weight)"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.12)
    cbar.set_label("Weight (softmax over keys)")
    fig.tight_layout()
    _savefig_png(fig, out_path)
    plt.close(fig)


def save_attention_head0_cls_to_patches(
    weights: AttentionWeights | torch.Tensor,
    grid_hw: tuple[int, int],
    out_path: Path,
    block_index: int = 0,
    num_heads: int | None = None,
) -> None:
    """Convenience: head index 0."""
    save_attention_head_cls_to_patches(
        weights,
        grid_hw,
        out_path,
        head_index=0,
        block_index=block_index,
        num_heads=num_heads,
    )


def save_attention_mean_cls_to_patches(
    weights: AttentionWeights | torch.Tensor,
    grid_hw: tuple[int, int],
    out_path: Path,
    block_index: int = 0,
    num_heads: int | None = None,
) -> None:
    """Mean softmax weights over heads: CLS → patches."""
    aw = _coerce_attention_weights(weights)
    if num_heads is None:
        num_heads = aw.per_head.shape[1]
    heat = _cls_to_patch_heatmap_bll(aw.mean_over_heads, grid_hw)
    gh, gw = grid_hw

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(heat, cmap="magma", origin="upper", interpolation="nearest")
    ax.set_xticks(np.arange(gw))
    ax.set_yticks(np.arange(gh))
    ax.set_xlabel("Patch column")
    ax.set_ylabel("Patch row")
    ax.set_title(
        f"Transformer block {block_index}: mean over {num_heads} heads\n"
        "CLS → image patches (averaged head weights)"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.12)
    cbar.set_label("Mean weight")
    fig.tight_layout()
    _savefig_png(fig, out_path)
    plt.close(fig)


def save_attention_cls_patches_only_figure(
    weights: AttentionWeights | torch.Tensor,
    grid_hw: tuple[int, int],
    out_path: Path,
    block_index: int = 0,
    num_heads: int | None = None,
) -> None:
    """
    Same numeric map as mean CLS→patches, framed for teaching: only patch *keys*
    (not CLS self-column), row-major patch order from the conv embedding.
    """
    aw = _coerce_attention_weights(weights)
    if num_heads is None:
        num_heads = aw.per_head.shape[1]
    heat = _cls_to_patch_heatmap_bll(aw.mean_over_heads, grid_hw)
    gh, gw = grid_hw

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    im = ax.imshow(heat, cmap="cividis", origin="upper", interpolation="nearest")
    ax.set_xticks(np.arange(gw))
    ax.set_yticks(np.arange(gh))
    ax.set_xlabel("Patch column (0 … 7)")
    ax.set_ylabel("Patch row (0 … 7)")
    ax.set_title(
        f"Block {block_index}: CLS looks at which patches?\n"
        f"(mean over {num_heads} heads; keys = {gh}×{gw} image patches only)"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.12)
    cbar.set_label("Attention to patch")
    fig.tight_layout()
    _savefig_png(fig, out_path)
    plt.close(fig)


def save_attention_cls_patches_gridded(
    weights: AttentionWeights | torch.Tensor,
    grid_hw: tuple[int, int],
    out_path: Path,
    block_index: int = 0,
    num_heads: int | None = None,
) -> None:
    """8×8 mean CLS→patch heatmap with patch-cell boundaries drawn on top."""
    aw = _coerce_attention_weights(weights)
    if num_heads is None:
        num_heads = aw.per_head.shape[1]
    heat = _cls_to_patch_heatmap_bll(aw.mean_over_heads, grid_hw)
    gh, gw = grid_hw

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    im = ax.imshow(
        heat,
        cmap="inferno",
        origin="upper",
        interpolation="nearest",
        extent=(-0.5, gw - 0.5, gh - 0.5, -0.5),
    )
    for t in range(gh + 1):
        ax.axhline(t - 0.5, color="white", linewidth=0.9, alpha=0.85)
    for t in range(gw + 1):
        ax.axvline(t - 0.5, color="white", linewidth=0.9, alpha=0.85)
    ax.set_xlim(-0.5, gw - 0.5)
    ax.set_ylim(gh - 0.5, -0.5)
    ax.set_xticks(np.arange(gw))
    ax.set_yticks(np.arange(gh))
    ax.set_xlabel("Patch column")
    ax.set_ylabel("Patch row")
    ax.set_title(
        f"Block {block_index}: patch-grid layout\n"
        f"(mean CLS→patch weights; {gh}×{gw} cells = image patches)"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.12)
    cbar.set_label("Mean weight")
    fig.tight_layout()
    _savefig_png(fig, out_path)
    plt.close(fig)


def save_cls_attention_overlay_on_image(
    image_chw: torch.Tensor,
    weights: AttentionWeights | torch.Tensor,
    grid_hw: tuple[int, int],
    patch_size: int,
    out_path: Path,
    block_index: int = 0,
    alpha: float = 0.48,
) -> None:
    """
    Map mean CLS→patch importance onto the 32×32 RGB image (each patch filled flat).
    """
    aw = _coerce_attention_weights(weights)
    heat = _cls_to_patch_heatmap_bll(aw.mean_over_heads, grid_hw)
    up = _upsample_patch_map_to_pixels(heat, patch_size)
    img = _chw01_to_hwc_rgb(image_chw)
    if up.shape[0] != img.shape[0] or up.shape[1] != img.shape[1]:
        raise ValueError("Upsampled heatmap must match image H×W")

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.imshow(img, vmin=0, vmax=1, interpolation="nearest")
    im = ax.imshow(
        up,
        cmap="hot",
        alpha=alpha,
        interpolation="nearest",
        vmin=float(np.min(heat)),
        vmax=float(np.max(heat)),
    )
    ax.set_aspect("equal")
    ax.set_title(
        f"Block {block_index}: where CLS attends (on the input image)\n"
        "mean over heads, each 4×4 pixel block = one patch"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.08)
    cbar.set_label("CLS→patch weight")
    fig.tight_layout()
    _savefig_png(fig, out_path)
    plt.close(fig)


def save_logits_bar_chart(
    logits: torch.Tensor,
    out_path: Path,
    title: str = "Classifier logits (no training)",
) -> None:
    """Bar chart for (B, num_classes) — uses first batch row."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vals = logits[0].detach().float().cpu().numpy()
    n = len(vals)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(np.arange(n), vals, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("class index")
    ax.set_ylabel("logit")
    ax.set_title(title)
    ax.set_xticks(np.arange(n))
    fig.tight_layout()
    _savefig_png(fig, out_path)
    plt.close(fig)


# Backwards-compatible names
def print_tensor_shapes(name: str, t: torch.Tensor) -> None:
    print(f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}")


def save_patch_grid_figure(
    image_chw: torch.Tensor,
    patch_size: int,
    out_path: Path,
    title: str = "Image with patch grid",
) -> None:
    save_patch_grid(image_chw, patch_size, out_path, title=title)


def save_attention_cls_to_patches(
    weights: AttentionWeights | torch.Tensor,
    grid_hw: tuple[int, int],
    out_path: Path,
    title: str = "CLS -> patches (mean heads)",
) -> None:
    """Mean CLS→patches heatmap with a caller-chosen title (legacy API)."""
    aw = _coerce_attention_weights(weights)
    heat = _cls_to_patch_heatmap_bll(aw.mean_over_heads, grid_hw)
    gh, gw = grid_hw
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(heat, cmap="magma", origin="upper", interpolation="nearest")
    ax.set_xticks(np.arange(gw))
    ax.set_yticks(np.arange(gh))
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.1)
    fig.tight_layout()
    _savefig_png(fig, out_path)
    plt.close(fig)
