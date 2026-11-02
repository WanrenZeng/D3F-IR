#!/usr/bin/env python3
"""Run visible-light-to-infrared inference with a dataset default checkpoint."""

import argparse
from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d3f_ir.checkpoint import load_pretrained_weights
from model.config import ModelConfig
from model.model import D3FIRModel

DATASET_CONFIGS = {
    "FLIR-align": {"resolution": (640, 512), "checkpoint": "FLIR-align/ckpt.pt"},
    "KAIST_clean": {"resolution": (640, 512), "checkpoint": "KAIST_clean/ckpt.pt"},
    "M3FD_Detection": {"resolution": (512, 384), "checkpoint": "M3FD_Detection/ckpt.pt"},
}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_CONFIGS), required=True)
    parser.add_argument("--input", type=Path, required=True, help="An RGB image or a directory of images")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None, help="Override the dataset default checkpoint")
    parser.add_argument("--dino_ckpt", type=Path, default=PROJECT_ROOT / "ckpt/DINOv3/model.safetensors")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    return parser.parse_args()


def build_model(dataset: str, dino_checkpoint: Path, device: torch.device) -> D3FIRModel:
    width, height = DATASET_CONFIGS[dataset]["resolution"]
    config = ModelConfig(
        image_size=(height, width),
        patch_size=16,
        latent_dim=256,
        time_token_count=2,
        dit_depth=14,
        dit_heads=16,
        dit_hidden=128,
        dit_mlp_ratio=4.0,
        dit_head_depth=2,
        dit_head_hidden=256,
        dit_head_heads=16,
        dit_head_mlp_ratio=4.0,
        pixel_predictor_hidden=384,
        pixel_predictor_layers=6,
        pixel_predictor_heads=16,
        pixel_predictor_mlp_ratio=4.0,
        fusion_heads=4,
        shallow_channels=80,
        align_dim=384,
    )
    return D3FIRModel(config, str(dino_checkpoint)).to(device)


@torch.inference_mode()
def infer_image(
    model: D3FIRModel,
    image_path: Path,
    output_path: Path,
    resolution: tuple[int, int],
    steps: int,
    device: torch.device,
    precision: str,
) -> None:
    width, height = resolution
    with Image.open(image_path) as source:
        image = source.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
    visible = ToTensor()(image).unsqueeze(0).to(device)
    visible_tokens, source_latent = model.encode_rgb(visible)
    if source_latent is None:
        raise RuntimeError("The selected model did not produce visible-light source latents.")

    latent = source_latent
    output = visible.clone()
    dt = 1.0 / steps
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    amp_enabled = device.type == "cuda" and precision != "fp32"
    for step in range(steps):
        time = torch.full(
            (visible.shape[0],), step / steps, device=device, dtype=latent.dtype
        )
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            velocity = model.dit(
                latent, t=time, cond=visible_tokens, rgb_tokens=visible_tokens
            )
            pixel_velocity, _ = model.decode(
                visible_tokens,
                velocity,
                rgb=visible,
                context_tokens=visible_tokens,
                rgb_latent=source_latent,
            )
        latent = latent + velocity * dt
        output = output + pixel_velocity * dt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ToPILImage()(output[0].float().clamp(0.0, 1.0).cpu()).save(output_path)


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be greater than zero")
    if not args.input.exists():
        raise FileNotFoundError(f"Input does not exist: {args.input}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    checkpoint_path = args.weights or PROJECT_ROOT / "ckpt" / DATASET_CONFIGS[args.dataset]["checkpoint"]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if not args.dino_ckpt.is_file():
        raise FileNotFoundError(f"DINO checkpoint does not exist: {args.dino_ckpt}")

    output_dir = args.output_dir or PROJECT_ROOT / "outputs" / "inference" / args.dataset
    if args.input.is_file():
        if args.input.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {args.input.suffix}")
        image_paths = [args.input]
        relative_root = args.input.parent
    else:
        image_paths = sorted(
            path for path in args.input.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        relative_root = args.input
    if not image_paths:
        raise FileNotFoundError(f"No supported images found under {args.input}")

    model = build_model(args.dataset, args.dino_ckpt, device)
    load_pretrained_weights(model, checkpoint_path, strict=True)
    model.eval()
    model.requires_grad_(False)
    print(f"Dataset: {args.dataset}; checkpoint: {checkpoint_path}; device: {device}")
    for image_path in tqdm(image_paths, desc="Inference"):
        relative_path = image_path.relative_to(relative_root)
        output_path = output_dir / relative_path.with_suffix(".png")
        infer_image(
            model,
            image_path,
            output_path,
            DATASET_CONFIGS[args.dataset]["resolution"],
            args.steps,
            device,
            args.precision,
        )
    print(f"Saved {len(image_paths)} result(s) to {output_dir}")


if __name__ == "__main__":
    main()