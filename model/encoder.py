from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel

# DINOv3 ViT-S/16 architecture used by D3F-IR. Inference can build the encoder
# from these values alone (weights are then taken from the released ckpt.pt), so
# the gated original DINOv3 files are only required when load_pretrained=True.
DEFAULT_DINOV3_CONFIG = {
    "hidden_size": 384,
    "intermediate_size": 1536,
    "num_hidden_layers": 12,
    "num_attention_heads": 6,
    "hidden_act": "gelu",
    "patch_size": 16,
    "num_register_tokens": 4,
    "rope_theta": 100.0,
    "key_bias": False,
    "query_bias": True,
    "value_bias": True,
    "proj_bias": True,
    "mlp_bias": True,
    "use_gated_mlp": False,
    "pos_embed_rescale": 2.0,
    "layerscale_value": 1.0,
    "attention_dropout": 0.0,
    "drop_path_rate": 0.0,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-05,
    "image_size": 224,
    "num_channels": 3,
}


class DinoEncoder(nn.Module):
    """DINO encoder that always applies the draw adapter when configured."""

    def __init__(
        self,
        input_size: Tuple[int, int] = (512, 640),
        patch_size: int = 16,
        ckpt_path: str = "",
        dino_model: str = "vit_small_patch16_224.dino",
        trainable: bool = False,
        latent_reduce_dim: Optional[int] = None,
        load_pretrained: bool = True,
    ) -> None:
        super().__init__()


        self.input_size = input_size
        self.patch_size = patch_size


        if isinstance(input_size, int):
            H, W = input_size, input_size
        else:
            H, W = input_size

        self.num_patches = (H // patch_size) * (W // patch_size)

        checkpoint_path = Path(ckpt_path).expanduser() if ckpt_path else None
        model_path = None
        if checkpoint_path is not None and checkpoint_path.exists():
            model_path = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent

        if load_pretrained:
            # Training: initialise the encoder from the original DINOv3 weights.
            if model_path is None:
                raise FileNotFoundError(
                    f"DINO checkpoint does not exist: {ckpt_path}"
                )
            self.backbone = DINOv3ViTModel.from_pretrained(
                str(model_path), local_files_only=True, output_hidden_states=True
            )
            print(f"DINO weights loaded from {ckpt_path}")
        else:
            # Inference: build the architecture from the embedded config; all
            # weights are subsequently loaded from the model checkpoint (ckpt.pt).
            dino_config = DINOv3ViTConfig(**DEFAULT_DINOV3_CONFIG, output_hidden_states=True)
            self.backbone = DINOv3ViTModel(dino_config)
            print("DINO backbone built from embedded config; weights loaded from model checkpoint")

        self.trainable_backbone = trainable
        self.backbone.requires_grad_(trainable)
        self.backbone.train(trainable)

        if hasattr(self.backbone, "config") and hasattr(self.backbone.config, "hidden_size"):
            self.embed_dim = self.backbone.config.hidden_size
        else:
            raise RuntimeError("Could not find hidden_size in DINOv3 model config.")

        cfg = getattr(self.backbone, "default_cfg", {})
        mean = cfg.get("mean", (0.485, 0.456, 0.406))
        std = cfg.get("std", (0.229, 0.224, 0.225))
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

        self.draw_adapter: Optional[nn.Module] = None
        if latent_reduce_dim is not None:
            self.draw_adapter = nn.Sequential(nn.Linear(self.embed_dim, latent_reduce_dim), nn.SiLU())
        self.output_channels = self.embed_dim

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.trainable_backbone:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = (x - self.mean) / self.std
        outputs = self.backbone(x)
        last_hidden_states = outputs.last_hidden_state
        num_register = self.backbone.config.num_register_tokens
        patch_tokens = last_hidden_states[:, 1 + num_register :, :]


        draw_tokens = self.draw_adapter(patch_tokens) if self.draw_adapter is not None else None
        return patch_tokens, draw_tokens

    def extract_layer_features(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        x = (x - self.mean) / self.std
        outputs = self.backbone(x)


        hidden_states = outputs.hidden_states
        if layer_idx < 0:
            layer_idx = len(hidden_states) + layer_idx


        feat = hidden_states[layer_idx]
        num_register = self.backbone.config.num_register_tokens
        feat = feat[:, 1 + num_register :, :]
        return feat


