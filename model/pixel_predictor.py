import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.blocks import (
    DAB,
    RRG,
    TransformerBlock,
    VisionRotaryEmbeddingFast,
    cycleisp_conv,
)

class GatedAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, query: torch.Tensor, key: Optional[torch.Tensor] = None, value: Optional[torch.Tensor] = None) -> torch.Tensor:
        key = query if key is None else key
        value = query if value is None else value
        attn_output, _ = self.mha(query, key, value)
        return attn_output


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.rgb_norm = nn.LayerNorm(dim)
        self.draw_norm = nn.LayerNorm(dim)
        self.rgb_to_draw = GatedAttention(dim, num_heads, dropout=dropout)
        self.proj = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, rgb_tokens: torch.Tensor, draw_tokens: torch.Tensor) -> torch.Tensor:
        rgb_q = self.rgb_norm(rgb_tokens)
        draw_q = self.draw_norm(draw_tokens)
        attn = self.rgb_to_draw(draw_q, rgb_q, rgb_q)
        return self.act(self.proj(attn + draw_q))


class GammaNetwork(nn.Module):
    def __init__(self, input_dim: int, num_gammas: int = 10) -> None:
        super().__init__()
        self.num_gammas = num_gammas
        self.proj = nn.Linear(input_dim, 2 * num_gammas)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=1)
        x = self.tanh(self.proj(x))
        return x


class LocalRefinementHead(nn.Module):
    def __init__(self, latent_dim: int, rgb_patch_size: int = 16, rggb_patch_size: int = 8, base_channels: int = 64) -> None:
        super().__init__()
        if rgb_patch_size % 8 != 0:
            raise ValueError("rgb_patch_size must be divisible by 8")
        if rggb_patch_size != rgb_patch_size // 2:
            raise ValueError("rggb_patch_size must equal rgb_patch_size // 2")
        self.rgb_patch_size = rgb_patch_size
        self.rggb_patch_size = rggb_patch_size

        self.down8 = self._make_down_block(3, base_channels)
        self.down4 = self._make_down_block(base_channels, (latent_dim // 2) - 64)
        self.down2 = self._make_down_block((latent_dim // 2) - 64, (latent_dim // 2) - 32)
        self.down1 = self._make_down_block((latent_dim // 2) - 32, latent_dim // 2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(latent_dim // 2, latent_dim // 2, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(num_groups=4, num_channels=latent_dim // 2),
            nn.SiLU(),
        )

        self.up2 = self._make_up_block(latent_dim // 2, (latent_dim // 2) - 32)
        self.up4 = self._make_up_block((latent_dim // 2) - 32, (latent_dim // 2) - 64)
        self.up8 = self._make_up_block((latent_dim // 2) - 64, base_channels)
        self.out_channels = base_channels

    def _make_down_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0),
            nn.GroupNorm(num_groups=4, num_channels=out_ch),
            nn.SiLU(),
        )

    def _make_up_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0),
            nn.GroupNorm(num_groups=4, num_channels=out_ch),
            nn.SiLU(),
        )

    def forward(self, rgb_patches: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        x8 = self.down8(rgb_patches)
        x4 = self.down4(x8)
        x2 = self.down2(x4)
        x1 = self.down1(x2)
        token_feat = tokens.unsqueeze(-1).unsqueeze(-1)
        scale, shift = torch.chunk(token_feat, 2, dim=1)
        scale = scale.sigmoid()
        x1 = x1 * scale + shift
        x1 = self.bottleneck(x1)
        x = self.up2(x1) + x2
        x = self.up4(x) + x4
        x = self.up8(x) + x8
        return x


class fusion(nn.Module):
    def __init__(self, channel: int, reduction: int = 16) -> None:
        super().__init__()

        self.act = nn.GELU()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.body = RRG(cycleisp_conv, channel, kernel_size=3, reduction=reduction, act=nn.PReLU(channel), num_dab=3)

    def forward(self, tokens: torch.Tensor, y: torch.Tensor) -> torch.Tensor:


        b, c, h, w = y.shape
        num_tokens = tokens.shape[1]


        patch_size = 16


        patches = self._extract_patches(y, patch_size)

        token_feat = tokens.unsqueeze(-1).unsqueeze(-1)
        scale, shift = torch.chunk(token_feat, 2, dim=2)
        scale = scale.sigmoid()

        refined_patches = patches * scale + shift
        out = self._reconstruct_from_patches(refined_patches, (h, w), patch_size)
        out = self.act(self.conv(out))
        out = self.body(out)
        out = out

        return out

    @staticmethod
    def _extract_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:
        b, c, h, w = image.shape
        if h % patch_size != 0 or w % patch_size != 0:
            raise RuntimeError("Image size must be divisible by patch size.")
        patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        return patches.view(b, -1, c, patch_size, patch_size)

    @staticmethod
    def _reconstruct_from_patches(patches: torch.Tensor, image_size: Tuple[int, int], patch_size: int) -> torch.Tensor:
        b, num_patches, c, ph, pw = patches.shape
        h, w = image_size
        patches_h = h // patch_size
        patches_w = w // patch_size
        if patches_h * patches_w != num_patches:
            raise RuntimeError(f"Patch count {num_patches} does not match image size {image_size} with patch size {patch_size} (expected {patches_h}x{patches_w}).")
        patches = patches.view(b, patches_h, patches_w, c, ph, pw)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        return patches.view(b, c, h, w)


class CTXFusion(nn.Module):
    def __init__(self, channel: int, ctx_dim: int = 384) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(ctx_dim, channel), nn.Sigmoid())

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        out = ctx.mean(dim=1)
        out = self.proj(out)
        return out.unsqueeze(-1).unsqueeze(-1)


class ShallowUNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 96,
        token_dim: int = 512,
        num_fusion_blocks: int = 6,
        num_heads: int = 8,
    ) -> None:
        super().__init__()

        act = nn.GELU()


        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, base_channels),
            act
        )
        self.dab_start = nn.Sequential(
            DAB(cycleisp_conv, base_channels, 3, 8),
            nn.GroupNorm(4, base_channels),
            act
        )


        ch1 = int(base_channels * 1.5)
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, ch1, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, ch1),
            act
        )
        self.dab_down1 = nn.Sequential(
            DAB(cycleisp_conv, ch1, 3, 8),
            nn.GroupNorm(4, ch1),
            act
        )


        ch2 = base_channels * 2
        self.down2 = nn.Sequential(
            nn.Conv2d(ch1, ch2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, ch2),
            act
        )
        self.dab_down2 = nn.Sequential(
            DAB(cycleisp_conv, ch2, 3, 8),
            nn.GroupNorm(4, ch2),
            act
        )


        self.fusion_blocks = nn.ModuleList([
            TransformerBlock(token_dim, num_heads, 4.0) for _ in range(num_fusion_blocks)
        ])
        self.fusion_projs = nn.ModuleList([
            nn.Linear(token_dim, 2 * ch2) for _ in range(num_fusion_blocks)
        ])
        self.fusion_dabs = nn.ModuleList([
            nn.Sequential(
                DAB(cycleisp_conv, ch2, 3, 8),
                nn.GroupNorm(4, ch2),
                act
            ) for _ in range(num_fusion_blocks)
        ])


        self.mid_rrg = nn.Sequential(
            RRG(cycleisp_conv, ch2, 3, 8, act, 3),
            nn.GroupNorm(4, ch2),
            act
        )


        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ch2, ch1, kernel_size=2, stride=2),
            nn.GroupNorm(4, ch1),
            act
        )
        self.dab_up1 = nn.Sequential(
            DAB(cycleisp_conv, ch1, 3, 8),
            nn.GroupNorm(4, ch1),
            act
        )


        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ch1, base_channels, kernel_size=2, stride=2),
            nn.GroupNorm(4, base_channels),
            act
        )
        self.dab_up2 = nn.Sequential(
            DAB(cycleisp_conv, base_channels, 3, 8),
            nn.GroupNorm(4, base_channels),
            act
        )


        self.out_rrg = nn.Sequential(
            RRG(cycleisp_conv, base_channels, 3, 8, act, 3),
            nn.GroupNorm(4, base_channels),
            act
        )


        self.tail = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)


        fusion_head_dim = token_dim // num_heads // 2

        self.rope_seq_len = (32, 40)
        self.rope = VisionRotaryEmbeddingFast(dim=fusion_head_dim, pt_seq_len=self.rope_seq_len)

        self.CTX_fusion = CTXFusion(base_channels, ctx_dim=384)

    def forward(self, rgb: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:

        x = self.head(rgb)
        skip1 = self.dab_start(x)

        x = self.down1(skip1)
        skip2 = self.dab_down1(x)

        x = self.down2(skip2)
        feat = self.dab_down2(x)


        rope = self.rope
        if tokens.shape[1] != rope.freqs_cos.shape[0]:
            h_token = feat.shape[2] // 4
            w_token = feat.shape[3] // 4
            if h_token * w_token == tokens.shape[1]:


                base_dim = self.rope.freqs_cos.shape[1] // 2
                rope = VisionRotaryEmbeddingFast(
                    dim=base_dim,
                    pt_seq_len=(h_token, w_token)
                ).to(tokens.device)

        for block, proj, dab in zip(self.fusion_blocks, self.fusion_projs, self.fusion_dabs):
            tokens = block(tokens, rope=rope)

            affine = proj(tokens)
            scale, shift = torch.chunk(affine, 2, dim=-1)


            b, n, c = scale.shape
            h_feat, w_feat = feat.shape[2], feat.shape[3]


            h_token = h_feat // 4
            w_token = w_feat // 4


            if h_token * w_token != n:


                pass

            scale_grid = scale.transpose(1, 2).reshape(b, c, h_token, w_token)
            shift_grid = shift.transpose(1, 2).reshape(b, c, h_token, w_token)


            feat_view = feat.view(b, c, h_token, 4, w_token, 4)


            scale_exp = scale_grid.unsqueeze(3).unsqueeze(5)
            shift_exp = shift_grid.unsqueeze(3).unsqueeze(5)


            feat_view = feat_view * (1 + scale_exp) + shift_exp


            feat = feat_view.reshape(b, c, h_feat, w_feat)

            feat = dab(feat)

        x = self.mid_rrg(feat)


        x = self.up1(x)
        x = self.dab_up1(x)
        x = x + skip2

        x = self.up2(x)
        x = self.dab_up2(x)
        x = x + skip1

        x = self.out_rrg(x)
        out = self.tail(x)

        return out


class PixelPredictor(nn.Module):
    """Decoder that fuses RGB & draw tokens and outputs RGGB patches."""

    def __init__(
        self,
        latent_dim: int,
        rgb_dim: int,
        num_patches: int,
        patch_size: int,
        image_size: Union[int, Tuple[int, int]],
        decoder_hidden: int,
        decoder_layers: int,
        decoder_heads: int,
        decoder_mlp_ratio: float,
        layer_norm_eps: float,
        fusion_heads: int = 8,
        gamma_input_dim: Optional[int] = None,
        local_refine_channels: int = 64,
    ) -> None:
        super().__init__()


        self.patch_size = patch_size
        self.latent_fusion = CrossAttentionFusion(latent_dim, fusion_heads)

        self.draw_up = nn.Sequential(
            nn.Linear(latent_dim, rgb_dim),
            nn.GELU()
        )
        self.fusion = CrossAttentionFusion(rgb_dim, fusion_heads)

        dec_half_head_dim = decoder_hidden // decoder_heads // 2
        self.rope_dim = dec_half_head_dim

        if isinstance(image_size, int):
             H, W = image_size, image_size
        else:
             H, W = image_size
        h_patches = H // patch_size
        w_patches = W // patch_size
        hw_seq_len = (h_patches, w_patches)

        self.rope = VisionRotaryEmbeddingFast(dim=dec_half_head_dim, pt_seq_len=hw_seq_len)
        self.act = nn.GELU()
        self.decoder_embed = nn.Linear(rgb_dim, decoder_hidden)


        self.gamma_net = GammaNetwork(gamma_input_dim) if gamma_input_dim is not None else None
        self.shallow_branch: Optional[ShallowUNetEncoder] = None
        self._last_intermediates: Dict[str, torch.Tensor] = {}

    def _reset_intermediates(self) -> None:
        self._last_intermediates = {}

    def forward(
        self,
        rgb_tokens_full: torch.Tensor,
        draw_tokens: torch.Tensor,
        rgb: Optional[torch.Tensor] = None,
        context_tokens: Optional[torch.Tensor] = None,
        rgb_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._reset_intermediates()

        if rgb_latent is not None and self.latent_fusion is not None:
            draw_tokens = self.latent_fusion(draw_tokens, rgb_latent)
        draw_tokens_up = self.draw_up(draw_tokens)
        fused = self.fusion(draw_tokens_up, rgb_tokens_full)
        x = self.decoder_embed(fused)
        x = self.act(x)


        tokens = x


        final_image = None
        intermediates: Dict[str, torch.Tensor] = {"tokens": tokens.detach()}

        if self.shallow_branch is not None and rgb is not None:
            final_image = self.shallow_branch(rgb, tokens)
            intermediates["decoder_refined"] = torch.sigmoid(final_image.detach())


        self._last_intermediates = intermediates
        return tokens, final_image

    def get_intermediates(self) -> Dict[str, torch.Tensor]:
        return self._last_intermediates


