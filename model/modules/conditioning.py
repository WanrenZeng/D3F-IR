"""Time embedding and adaptive normalization utilities."""

import math
from typing import Optional

import torch
import torch.nn as nn

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, dim))

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        freq = torch.exp(torch.arange(half_dim, device=device, dtype=timesteps.dtype) * -(math.log(10000.0) / max(1, half_dim - 1)))
        args = timesteps[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        return self.mlp(emb)


class TimeTokenEmbed(nn.Module):
    def __init__(self, embed_dim: int, token_count: int = 2) -> None:
        super().__init__()
        self.token_count = token_count
        self.proj = nn.Linear(embed_dim, embed_dim * token_count)

    def forward(self, time_embed: torch.Tensor) -> torch.Tensor:
        b, c = time_embed.shape
        tokens = self.proj(time_embed).view(b, self.token_count, c)
        return tokens


def _repeat_to_length(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    if tensor.size(1) == target_len:
        return tensor
    if target_len % tensor.size(1) != 0:
        raise ValueError(f"Sequence length {target_len} is not divisible by tensor length {tensor.size(1)}.")
    repeat = target_len // tensor.size(1)
    return tensor.repeat_interleave(repeat, dim=1)


def ddt_modulate(x: torch.Tensor, shift: Optional[torch.Tensor], scale: Optional[torch.Tensor]) -> torch.Tensor:
    if shift is None or scale is None:
        return x
    if shift.dim() == 2:
        shift = shift.unsqueeze(1)
    if scale.dim() == 2:
        scale = scale.unsqueeze(1)
    shift = _repeat_to_length(shift, x.size(1))
    scale = _repeat_to_length(scale, x.size(1))
    return x * (1 + scale) + shift


def ddt_gate(x: torch.Tensor, gate: Optional[torch.Tensor]) -> torch.Tensor:
    if gate is None:
        return x
    if gate.dim() == 2:
        gate = gate.unsqueeze(1)
    gate = _repeat_to_length(gate, x.size(1))
    return x * gate


