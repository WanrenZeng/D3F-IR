"""DiT transformer block."""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .attention import CrossAttention, NormAttention
from .conditioning import ddt_gate, ddt_modulate
from .rotary import VisionRotaryEmbeddingFast


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, mlp_ratio: float = 4.0, enable_rgb_cross: bool = True) -> None:
        super().__init__()
        self.enable_rgb_cross = enable_rgb_cross
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = NormAttention(dim, num_heads=num_heads, attn_drop=dropout)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_ratio, dropout)
        if enable_rgb_cross:
            self.cross = CrossAttention(dim, num_heads=num_heads, dropout=dropout)
            self.cross_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        rgb_ctx: Optional[torch.Tensor],
        mod_params: Optional[Tuple[torch.Tensor, ...]] = None,
        rope: Optional[VisionRotaryEmbeddingFast] = None,
        time_tokens: int = 0,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mod_params is None:
            shift_self = scale_self = gate_self = None
            shift_ff = scale_ff = gate_ff = None
        else:
            (shift_self, scale_self, gate_self, shift_ff, scale_ff, gate_ff,) = mod_params

        h = ddt_modulate(self.norm_self(x), shift_self, scale_self)
        self_out = self.self_attn(h, rope=rope, time_tokens=time_tokens)
        x = x + ddt_gate(self_out, gate_self)

        h = ddt_modulate(self.norm_ff(x), shift_ff, scale_ff)
        ff_out = self.ff(h)
        x = x + ddt_gate(ff_out, gate_ff)

        if self.enable_rgb_cross and rgb_ctx is not None and time_tokens < x.size(1):
            time_part = x[:, :time_tokens, :]
            feat_part = x[:, time_tokens:, :]
            feat_part = self.cross(feat_part, rgb_ctx)
            x = torch.cat([time_part, feat_part], dim=1)
        return x


