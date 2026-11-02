from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.blocks import (
    DiTBlock,
    SinusoidalTimeEmbedding,
    TimeTokenEmbed,
    VisionRotaryEmbeddingFast,
)

class ConditionalDiT(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        rgb_dim: int,
        backbone_hidden: int = 384,
        head_hidden: int = 384,
        backbone_depth: int = 14,
        head_depth: int = 2,
        backbone_heads: int = 8,
        head_heads: int = 8,
        backbone_mlp_ratio: float = 4.0,
        head_mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_patches: int = 196,
        time_token_count: int = 2,
        align_dim: Optional[int] = None,
        align_layer: Optional[int] = None,
        rope_resolution: Tuple[int, int] = (32, 40),
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.backbone_hidden = backbone_hidden
        self.head_hidden = head_hidden
        self.time_token_count = time_token_count
        self.num_patches = num_patches


        self.input_proj = nn.Linear(latent_dim, backbone_hidden)
        self.rgb_proj = nn.Linear(rgb_dim, backbone_hidden)
        self.time_embed = SinusoidalTimeEmbedding(backbone_hidden)
        self.time_token = TimeTokenEmbed(backbone_hidden, token_count=time_token_count)
        self.time_pos = nn.Parameter(torch.zeros(1, time_token_count * 2, backbone_hidden))
        self.cond_proj = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, backbone_hidden))

        bb_head_dim = backbone_hidden // backbone_heads // 2


        rope_seq_len = rope_resolution
        self.rope_resolution = rope_resolution
        self.backbone_rope = VisionRotaryEmbeddingFast(dim=bb_head_dim, pt_seq_len=rope_seq_len)

        self.backbone_blocks = nn.ModuleList([DiTBlock(backbone_hidden, backbone_heads, dropout=dropout, mlp_ratio=backbone_mlp_ratio, enable_rgb_cross=True) for _ in range(backbone_depth)])
        self.backbone_mods = nn.ModuleList([nn.Sequential(nn.SiLU(), nn.Linear(backbone_hidden, backbone_hidden * 6)) for _ in range(backbone_depth)])
        for mod in self.backbone_mods:
            nn.init.constant_(mod[-1].weight, 0)
            nn.init.constant_(mod[-1].bias, 0)

        self.backbone_norm = nn.LayerNorm(backbone_hidden)


        self.head_proj = nn.Linear(backbone_hidden, head_hidden)
        self.head_cond_proj = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, head_hidden))
        self.head_time_proj = nn.Linear(backbone_hidden, head_hidden)
        self.head_time_token = TimeTokenEmbed(head_hidden, token_count=time_token_count)
        self.head_time_pos = nn.Parameter(torch.zeros(1, time_token_count * 2, head_hidden))
        self.head_rgb_proj = nn.Linear(rgb_dim, head_hidden)

        hd_head_dim = head_hidden // head_heads // 2
        self.head_rope = VisionRotaryEmbeddingFast(dim=hd_head_dim, pt_seq_len=rope_seq_len)
        self.head_blocks = nn.ModuleList([DiTBlock(head_hidden, head_heads, dropout=dropout, mlp_ratio=head_mlp_ratio, enable_rgb_cross=True) for _ in range(head_depth)])
        self.head_mods = nn.ModuleList([nn.Sequential(nn.SiLU(), nn.Linear(head_hidden, head_hidden * 6)) for _ in range(head_depth)])
        for mod in self.head_mods:
            nn.init.constant_(mod[-1].weight, 0)
            nn.init.constant_(mod[-1].bias, 0)

        self.head_norm = nn.LayerNorm(head_hidden)
        self.output_proj = nn.Linear(head_hidden, latent_dim)

        self.align_dim = align_dim or backbone_hidden
        self.default_align_layer = align_layer if align_layer is not None else backbone_depth // 2
        projector_dim = backbone_hidden
        self.align_conv = nn.Sequential(
            nn.Conv2d(backbone_hidden, projector_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(projector_dim, self.align_dim, kernel_size=3, padding=1),
        )

    @staticmethod
    def _reshape_mod_params(mod_tensor: torch.Tensor, hidden: int) -> Tuple[torch.Tensor, ...]:
        chunks = mod_tensor.view(mod_tensor.size(0), 6, hidden).unbind(dim=1)
        return tuple(chunk for chunk in chunks)

    def forward(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor,
        r: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        rgb_tokens: Optional[torch.Tensor] = None,
        return_features: bool = False,
        feature_layer: Optional[int] = None,
        align_only: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if align_only and not return_features:
            raise ValueError("align_only=True requires return_features=True")

        x = self.input_proj(tokens)
        cond_vec = cond.mean(dim=1) if cond is not None else torch.zeros(tokens.size(0), self.backbone_hidden, device=tokens.device, dtype=tokens.dtype)
        cond_vec = self.cond_proj(cond_vec)
        time_embs = [self.time_embed(t)]
        if r is not None:
            time_embs.append(self.time_embed(r))
        cond_vec = cond_vec

        time_tokens = torch.cat([self.time_token(te) for te in time_embs], dim=1)
        time_tokens = time_tokens + self.time_pos[:, : time_tokens.size(1), :]
        x = torch.cat([time_tokens, x], dim=1)
        time_len = time_tokens.size(1)

        rgb_ctx = self.rgb_proj(rgb_tokens) if rgb_tokens is not None else None

        capture_idx = feature_layer if feature_layer is not None else (self.default_align_layer if return_features else -1)
        mid_features = None

        for idx, (block, mod) in enumerate(zip(self.backbone_blocks, self.backbone_mods)):
            mod_params = self._reshape_mod_params(mod(cond_vec), self.backbone_hidden)
            x = block(x, rgb_ctx=rgb_ctx, mod_params=mod_params, rope=self.backbone_rope, time_tokens=time_len)
            if return_features and capture_idx == idx:
                mid_features = self._project_align_features(x[:, time_len:, :])

        x = self.backbone_norm(x)
        feat_tokens = x[:, time_len:, :]
        if return_features and mid_features is None:
            mid_features = self._project_align_features(feat_tokens)


        head_tokens = self.head_proj(feat_tokens)
        head_cond = cond.mean(dim=1) if cond is not None else torch.zeros(tokens.size(0), self.head_hidden, device=tokens.device, dtype=tokens.dtype)
        head_cond = self.head_cond_proj(head_cond)
        head_time_embs = [self.head_time_proj(te) for te in time_embs]
        head_cond = head_cond + sum(head_time_embs)
        head_time_tokens = torch.cat([self.head_time_token(te_proj) for te_proj in head_time_embs], dim=1)
        head_time_tokens = head_time_tokens + self.head_time_pos[:, : head_time_tokens.size(1), :]
        head_x = torch.cat([head_time_tokens, head_tokens], dim=1)
        head_time_len = head_time_tokens.size(1)
        head_rgb_ctx = self.head_rgb_proj(rgb_tokens) if rgb_tokens is not None else None

        for block, mod in zip(self.head_blocks, self.head_mods):
            mod_params = self._reshape_mod_params(mod(head_cond), self.head_hidden)
            head_x = block(head_x, rgb_ctx=head_rgb_ctx, mod_params=mod_params, rope=self.head_rope, time_tokens=head_time_len)

        head_x = self.head_norm(head_x)
        head_feat_tokens = head_x[:, head_time_len:, :]
        velocity = self.output_proj(head_feat_tokens)

        if return_features:
            projected = mid_features if mid_features is not None else self._project_align_features(feat_tokens)
            if align_only:
                return {"mid_features": projected}
            return {"velocity": velocity, "mid_features": projected}
        return velocity

    def _project_align_features(self, features: torch.Tensor) -> torch.Tensor:
        b, n, c = features.shape


        h, w = self.rope_resolution

        if h * w != n:
            raise RuntimeError(f"Align projection expects grid ({h}x{w}={h*w}). Got {n}.")
        fmap = features.permute(0, 2, 1).reshape(b, c, h, w)
        projected = self.align_conv(fmap)
        return projected.flatten(2).permute(0, 2, 1)


