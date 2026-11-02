"""Architecture configuration for the three D3F-IR model components."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union


@dataclass(frozen=True)
class ModelConfig:
    image_size: Union[int, Tuple[int, int]] = (512, 640)
    patch_size: int = 16
    latent_dim: int = 256
    time_token_count: int = 2
    dit_depth: int = 14
    dit_heads: int = 8
    dit_hidden: int = 384
    dit_mlp_ratio: float = 4.0
    dit_head_depth: int = 2
    dit_head_hidden: Optional[int] = None
    dit_head_heads: Optional[int] = None
    dit_head_mlp_ratio: Optional[float] = None
    pixel_predictor_hidden: int = 512
    pixel_predictor_layers: int = 6
    pixel_predictor_heads: int = 16
    pixel_predictor_mlp_ratio: float = 4.0
    fusion_heads: int = 4
    shallow_channels: int = 96
    align_layer: Optional[int] = None
    align_dim: Optional[int] = None
