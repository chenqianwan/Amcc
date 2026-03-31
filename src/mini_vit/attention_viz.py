"""Structured attention weights for per-head and mean-head visualization."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AttentionWeights:
    """
    Self-attention from ``nn.MultiheadAttention`` with ``batch_first=True``.

    - ``per_head``: ``(batch, num_heads, seq_len, seq_len)`` — full softmax weights per head.
    - ``mean_over_heads``: ``(batch, seq_len, seq_len)`` — mean across heads (same ``L×L`` layout).
    """

    per_head: torch.Tensor
    mean_over_heads: torch.Tensor

    def __post_init__(self) -> None:
        if self.per_head.dim() != 4:
            raise ValueError(
                f"per_head must be (B, H, L, L), got shape {tuple(self.per_head.shape)}"
            )
        b, h, l, s = self.per_head.shape
        if l != s:
            raise ValueError("per_head last two dims must match (square attention)")
        exp_mean = (b, l, s)
        if tuple(self.mean_over_heads.shape) != exp_mean:
            raise ValueError(
                f"mean_over_heads must be {exp_mean}, got {tuple(self.mean_over_heads.shape)}"
            )

    def select_head(self, index: int) -> torch.Tensor:
        """Weights for one head: ``(batch, seq_len, seq_len)``."""
        return self.per_head[:, index, :, :]

    def to(self, device: torch.device | str) -> AttentionWeights:
        return AttentionWeights(
            self.per_head.to(device),
            self.mean_over_heads.to(device),
        )

    def cpu(self) -> AttentionWeights:
        return AttentionWeights(self.per_head.cpu(), self.mean_over_heads.cpu())
