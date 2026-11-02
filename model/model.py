"""Top-level D3F-IR model: DINO encoder, DiT, and pixel predictor."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .dit import ConditionalDiT
from .encoder import DinoEncoder
from .pixel_predictor import PixelPredictor, ShallowUNetEncoder


class D3FIRModel(nn.Module):
    """Compose the three configured model components."""

    def __init__(self, config: ModelConfig, dino_checkpoint: str, load_pretrained: bool = True) -> None:
        super().__init__()
        self.config = config
        image_size = config.image_size
        height, width = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.encoder = DinoEncoder(
            input_size=(height, width),
            patch_size=config.patch_size,
            ckpt_path=dino_checkpoint,
            trainable=True,
            latent_reduce_dim=config.latent_dim,
            load_pretrained=load_pretrained,
        )
        self.dit = ConditionalDiT(
            latent_dim=config.latent_dim,
            context_dim=self.encoder.output_channels,
            rgb_dim=self.encoder.output_channels,
            backbone_hidden=config.dit_hidden,
            head_hidden=config.dit_head_hidden or config.dit_hidden,
            backbone_depth=config.dit_depth,
            head_depth=config.dit_head_depth,
            backbone_heads=config.dit_heads,
            head_heads=config.dit_head_heads or config.dit_heads,
            backbone_mlp_ratio=config.dit_mlp_ratio,
            head_mlp_ratio=config.dit_head_mlp_ratio or config.dit_mlp_ratio,
            num_patches=self.encoder.num_patches,
            time_token_count=config.time_token_count,
            align_dim=config.align_dim or self.encoder.output_channels,
            align_layer=config.align_layer,
            rope_resolution=(height // config.patch_size, width // config.patch_size),
        )
        self.pixel_predictor = PixelPredictor(
            latent_dim=config.latent_dim,
            rgb_dim=self.encoder.output_channels,
            num_patches=self.encoder.num_patches,
            patch_size=config.patch_size,
            image_size=(height, width),
            decoder_hidden=config.pixel_predictor_hidden,
            decoder_layers=config.pixel_predictor_layers,
            decoder_heads=config.pixel_predictor_heads,
            decoder_mlp_ratio=config.pixel_predictor_mlp_ratio,
            layer_norm_eps=1e-6,
            fusion_heads=config.fusion_heads,
            gamma_input_dim=self.encoder.output_channels,
            local_refine_channels=config.shallow_channels,
        )
        if config.shallow_channels > 0:
            self.pixel_predictor.shallow_branch = ShallowUNetEncoder(
                in_channels=3,
                base_channels=config.shallow_channels,
                token_dim=config.pixel_predictor_hidden,
                num_fusion_blocks=6,
                num_heads=8,
            )
        self.pixel_predictor.pixel_v_proj = nn.Conv2d(
            3, config.latent_dim, kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.pixel_predictor.latent_v_proj = nn.Linear(
            config.latent_dim, config.latent_dim
        )
        self.encoder.rgb_adapter = (
            nn.Sequential(
                nn.Linear(self.encoder.output_channels, config.latent_dim),
                nn.SiLU(),
            )
        )

    def compute_sim_loss(self, v_latent: torch.Tensor, v_pixel: torch.Tensor) -> torch.Tensor:
        image_features = self.pixel_predictor.pixel_v_proj(v_pixel)
        image_features = image_features.flatten(2).transpose(1, 2)
        latent_features = self.pixel_predictor.latent_v_proj(v_latent)
        image_features = F.normalize(image_features, dim=-1)
        latent_features = F.normalize(latent_features, dim=-1)
        return 1.0 - torch.sum(image_features * latent_features, dim=-1).mean()

    def encode_rgb(self, rgb: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        tokens, _ = self.encoder(rgb)
        adapter = self.encoder.rgb_adapter
        return tokens, adapter(tokens) if adapter is not None else None

    def encode_infrared(self, infrared: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens, latent = self.encoder(infrared)
        if latent is None:
            raise RuntimeError("DinoEncoder requires a latent adapter for infrared targets.")
        return tokens, latent

    def forward_velocity(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        r: Optional[torch.Tensor],
        context_tokens: Optional[torch.Tensor],
        rgb_tokens: torch.Tensor,
    ) -> torch.Tensor:
        condition = context_tokens if context_tokens is not None else rgb_tokens
        return self.dit(z, t=t, r=r, cond=condition, rgb_tokens=rgb_tokens)

    def decode(
        self,
        rgb: torch.Tensor,
        rgb_latent: torch.Tensor,
        velocity: torch.Tensor,
        rgb_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # rgb_tokens: visible-light DINO tokens (merged former context_tokens).
        # rgb_latent: source latent (kept separate; not equivalent to rgb_tokens).
        tokens, image = self.pixel_predictor(
            rgb_tokens,
            velocity,
            rgb=rgb,
            rgb_latent=rgb_latent,
        )
        outputs: Dict[str, torch.Tensor] = {"decoder_tokens": tokens}
        outputs.update(self.pixel_predictor.get_intermediates())
        return image, outputs


FlowMatchingModel = D3FIRModel
