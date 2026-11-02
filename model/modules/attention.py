"""Attention and transformer blocks."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rotary import VisionRotaryEmbeddingFast
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed * self.weight


class NormAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        fused_attn: bool = False,
        use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        norm_layer = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rope: Optional[VisionRotaryEmbeddingFast] = None, time_tokens: int = 0) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            if time_tokens > 0:
                q_time, q_feat = q[:, :, :time_tokens, :], q[:, :, time_tokens:, :]
                k_time, k_feat = k[:, :, :time_tokens, :], k[:, :, time_tokens:, :]
                if q_feat.numel() > 0:
                    q_feat = rope(q_feat)
                    k_feat = rope(k_feat)
                q = torch.cat([q_time, q_feat], dim=2)
                k = torch.cat([k_time, k_feat], dim=2)
            else:
                q = rope(q)
                k = rope(k)
        if self.fused_attn:
            q = q.to(v.dtype)
            k = k.to(v.dtype)
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = NormAttention(dim, num_heads=num_heads, attn_drop=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
        rope: Optional[VisionRotaryEmbeddingFast] = None,
    ) -> torch.Tensor:
        attn_input = self.norm1(x)
        if scale is not None and shift is not None:
            attn_input = attn_input * (1 + scale) + shift
        attn_output = self.attn(attn_input, rope=rope)
        x = x + attn_output
        mlp_input = self.norm2(x)
        if scale is not None and shift is not None:
            mlp_input = mlp_input * (1 + scale) + shift
        x = x + self.mlp(mlp_input)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(query)
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.attn(q, kv_norm, kv_norm)
        out = self.drop(self.act(self.proj(attn_out)))
        return query + out


