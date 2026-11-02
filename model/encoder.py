from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModel

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
    ) -> None:
        super().__init__()


        self.input_size = input_size
        self.patch_size = patch_size


        if isinstance(input_size, int):
            H, W = input_size, input_size
        else:
            H, W = input_size

        self.num_patches = (H // patch_size) * (W // patch_size)

        checkpoint_path = Path(ckpt_path).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"DINO checkpoint does not exist: {checkpoint_path}"
            )
        model_path = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
        self.backbone = AutoModel.from_pretrained(
            str(model_path), local_files_only=True, output_hidden_states=True
        )
        print(f"DINO weights loaded from {ckpt_path}")

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


