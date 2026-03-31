"""Tiny Vision Transformer — forward pass only, explicit MultiheadAttention."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .attention_viz import AttentionWeights


class PatchEmbed(nn.Module):
    """Learnable patch embedding: Conv2d (P×P kernel, stride P) -> (B, N, D)."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    """Two-layer MLP with GELU (compact: hidden = 2 * dim)."""

    def __init__(self, dim: int, hidden_ratio: float = 2.0) -> None:
        super().__init__()
        hidden = int(dim * hidden_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm, residual MHA + residual MLP.

    MultiheadAttention uses batch_first=True everywhere. Expected layout:
      x, norm1(x): (N, L, E)  — batch, sequence length, embed_dim
      attn output: (N, L, E)
      attn weights (average_attn_weights=False): (N, num_heads, L, L)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = dim
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_hidden_ratio)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(
                "TransformerBlock expects x of shape (N, L, E) with batch_first=True; "
                f"got dim={x.dim()}, shape={tuple(x.shape)}"
            )
        n_batch, seq_len, model_dim = x.shape
        if model_dim != self.embed_dim:
            raise ValueError(
                f"Last dim must be embed_dim={self.embed_dim}, got {model_dim}"
            )

        h = self.norm1(x).contiguous()
        attn_out, attn_weights = self.attn(
            query=h,
            key=h,
            value=h,
            need_weights=True,
            average_attn_weights=False,
            is_causal=False,
        )
        if attn_out.shape != (n_batch, seq_len, self.embed_dim):
            raise RuntimeError(
                f"MHA output shape mismatch: expected {(n_batch, seq_len, self.embed_dim)}, "
                f"got {tuple(attn_out.shape)} (check batch_first and (N,L,E) layout)"
            )
        if attn_weights.shape != (n_batch, self.num_heads, seq_len, seq_len):
            raise RuntimeError(
                "Attention weight shape mismatch for batch_first=True + "
                f"average_attn_weights=False: expected {(n_batch, self.num_heads, seq_len, seq_len)}, "
                f"got {tuple(attn_weights.shape)}"
            )

        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class MiniViT(nn.Module):
    """
    32×32 RGB, patch 4, 64-dim, 2 blocks, 4 heads, 10 logits.
    Forward returns logits and a dict of tensors for debugging / visualization.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        num_classes: int = 10,
        mlp_hidden_ratio: float = 2.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        n_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # One learned position per *patch* only; CLS is prepended without sharing this table.
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        self.blocks = nn.ModuleList(
            TransformerBlock(embed_dim, num_heads, mlp_hidden_ratio, attn_dropout)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        x: (B, 3, 32, 32)

        Returns:
            logits: (B, num_classes)
            intermediates: tensors per stage; each ``attn_block_i`` is an
            :class:`AttentionWeights` (``per_head`` and ``mean_over_heads``).
        """
        b = x.shape[0]
        ih, iw = int(x.shape[2]), int(x.shape[3])
        expected = self.patch_embed.img_size
        if ih != expected or iw != expected:
            raise ValueError(
                f"Expected input (B,3,{expected},{expected}), got (…,{ih},{iw}). "
                "Spatial size must match the patch grid used to size positional embeddings."
            )

        intermediates: dict[str, Any] = {}
        intermediates["x_in"] = x

        patch_tokens = self.patch_embed(x)
        intermediates["patch_tokens"] = patch_tokens
        if patch_tokens.shape[1] != self.pos_embed.shape[1]:
            raise RuntimeError(
                f"Patch token length {patch_tokens.shape[1]} does not match "
                f"pos_embed length {self.pos_embed.shape[1]}."
            )

        patch_tokens = patch_tokens + self.pos_embed
        cls = self.cls_token.expand(b, -1, -1)
        # Standard order: [CLS], then patch tokens (each patch already has its position added).
        tokens = torch.cat([cls, patch_tokens], dim=1)
        intermediates["tokens_after_cls_pos"] = tokens

        for i, block in enumerate(self.blocks):
            tokens, attn_w = block(tokens)
            intermediates[f"attn_block_{i}"] = AttentionWeights(
                per_head=attn_w,
                mean_over_heads=attn_w.mean(dim=1),
            )
            intermediates[f"tokens_after_block_{i}"] = tokens

        tokens = self.norm(tokens)
        intermediates["tokens_after_final_norm"] = tokens

        cls_out = tokens[:, 0]
        intermediates["cls_token"] = cls_out

        logits = self.head(cls_out)
        intermediates["logits"] = logits

        return logits, intermediates
