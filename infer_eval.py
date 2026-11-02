#!/usr/bin/env python3
"""Evaluate this repo's checkpoints on paired IR datasets (PSNR/SSIM/LPIPS/FID).

This is a copy-and-adapt of ``infer.py`` that keeps the current-repo model
(``model.model.D3FIRModel``) instead of the legacy parent-directory model used
by ``evaluate_fm.py``. It runs visible-light -> infrared flow-matching inference
over a dataset manifest and reports the same metric pipeline as ``evaluate_fm.py``
so results can be compared for consistency.
"""

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import lpips
import torch
import torch.nn.functional as F
from cleanfid import fid
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.config import ModelConfig
from model.model import D3FIRModel

# resolution = (W, H); checkpoint/manifest paths follow the repo layout.
DATASET_CONFIGS = {
    "FLIR-align": {
        "resolution": (640, 512),
        "checkpoint": "FLIR-align/ckpt.pt",
        "manifest": "annotations_validation.json",
    },
    "KAIST_clean": {
        "resolution": (640, 512),
        "checkpoint": "KAIST_clean/ckpt.pt",
        "manifest": "test_set.json",
    },
    "M3FD_Detection": {
        "resolution": (512, 384),
        "checkpoint": "M3FD_Detection/ckpt.pt",
        "manifest": "test_set.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASET_CONFIGS),
                        default=list(DATASET_CONFIGS))
    parser.add_argument("--data_root", type=Path, default=Path("/share401/public/upload"),
                        help="Directory holding <dataset>/ manifests and images")
    parser.add_argument("--checkpoint_root", type=Path, default=PROJECT_ROOT / "ckpt")
    parser.add_argument("--weights", type=Path, default=None,
                        help="Override checkpoint path (single-dataset runs only)")
    parser.add_argument("--dino_ckpt", type=Path, default=None,
                        help="Optional: only needed to build the encoder from the original DINOv3 weights; "
                             "evaluation uses the fine-tuned backbone inside ckpt.pt")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "eval_repo_results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--steps", type=int, default=10, help="Flow-matching sampling steps")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of evaluation pairs (smoke test)")
    parser.add_argument("--save_images", action="store_true",
                        help="Also dump generated IR images under output_dir/<dataset>/generated")
    return parser.parse_args()


class PairedInfraredDataset(Dataset):
    def __init__(self, data_root: Path, manifest_path: Path, resolution):
        self.data_root = data_root
        self.samples = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.width, self.height = resolution
        self.to_tensor = ToTensor()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.samples[index]
        with Image.open(self.data_root / item["vision_path"]) as image:
            visible = image.convert("RGB").resize((self.width, self.height))
        with Image.open(self.data_root / item["infrared_path"]) as image:
            infrared = image.convert("RGB").resize((self.width, self.height))
        return {"visible": self.to_tensor(visible), "infrared": self.to_tensor(infrared)}


def load_pretrained_weights(model, ckpt_path, strict=False):
    # Inlined from d3f_ir.checkpoint to avoid the matplotlib dependency pulled
    # in by d3f_ir.common; identical remapping of legacy checkpoint keys.
    print(f"Loading pre-trained weights from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict_raw = checkpoint["model"] if "model" in checkpoint else checkpoint

    model_dict = model.state_dict()
    state_dict = {}
    unmatched_keys = []
    shape_mismatches = []

    for original_key, v in state_dict_raw.items():
        if original_key.startswith("decoder."):
            k = "pixel_predictor." + original_key.removeprefix("decoder.")
        elif original_key == "rgb_adapter" or original_key.startswith("rgb_adapter."):
            k = "encoder." + original_key
        elif original_key == "pixel_v_proj" or original_key.startswith("pixel_v_proj."):
            k = "pixel_predictor." + original_key
        elif original_key == "latent_v_proj" or original_key.startswith("latent_v_proj."):
            k = "pixel_predictor." + original_key
        else:
            k = original_key
        if k in model_dict:
            target_shape = model_dict[k].shape
            if v.shape != target_shape:
                if "position_embeddings" in k and v.dim() == 3:
                    print(f"Resizing position embedding {k}: {v.shape} -> {target_shape}")
                    num_prefix_tokens = 1
                    n_old = v.shape[1]
                    n_new = target_shape[1]
                    d = v.shape[2]
                    if n_old == 257:
                        h_old, w_old = 16, 16
                    else:
                        side = int(math.sqrt(n_old - 1))
                        if side * side == n_old - 1:
                            h_old, w_old = side, side
                        else:
                            print(f"Warning: ambiguous source shape for {k}, skipping resize.")
                            continue
                    cls_token = v[:, :1, :]
                    pos_tokens = v[:, 1:, :]
                    pos_tokens = pos_tokens.transpose(1, 2).reshape(1, d, h_old, w_old)
                    n_patches_new = n_new - 1
                    if hasattr(model, "encoder") and hasattr(model.encoder, "input_size"):
                        H_new, W_new = model.encoder.input_size
                        h_new = H_new // model.encoder.patch_size
                        w_new = W_new // model.encoder.patch_size
                        if h_new * w_new != n_patches_new:
                            print(f"Warning: target parameter shape {n_patches_new} != config {h_new*w_new}")
                            side = int(math.sqrt(n_patches_new))
                            if side * side == n_patches_new:
                                h_new, w_new = side, side
                            else:
                                print(f"Warning: ambiguous target shape for {k}, skipping resize.")
                                continue
                    else:
                        side = int(math.sqrt(n_patches_new))
                        h_new, w_new = side, side
                    print(f"Interpolating pos embed from {h_old}x{w_old} to {h_new}x{w_new}")
                    pos_tokens = F.interpolate(pos_tokens, size=(h_new, w_new), mode="bicubic", align_corners=False)
                    pos_tokens = pos_tokens.flatten(2).transpose(1, 2)
                    v_new = torch.cat((cls_token, pos_tokens), dim=1)
                    state_dict[k] = v_new
                elif "rope.freqs_cos" in k or "rope.freqs_sin" in k:
                    print(f"Skipping resolution-dependent RoPE buffer {k}: {v.shape} vs {target_shape}")
                    continue
                else:
                    print(f"Skipping parameter {k} due to shape mismatch: {v.shape} vs {target_shape}")
                    shape_mismatches.append((original_key, k))
            else:
                state_dict[k] = v
        else:
            if not original_key.startswith("inception_loss_module."):
                unmatched_keys.append(original_key)

    missing_keys = sorted(set(model_dict) - set(state_dict))
    if strict and (unmatched_keys or shape_mismatches or missing_keys):
        raise RuntimeError(
            "Checkpoint is incompatible with the model: "
            f"{len(missing_keys)} missing model keys, "
            f"{len(unmatched_keys)} unmatched checkpoint keys, "
            f"{len(shape_mismatches)} shape mismatches. "
            f"Missing examples: {missing_keys[:10]}; "
            f"unmatched examples: {unmatched_keys[:10]}; "
            f"shape mismatch examples: {shape_mismatches[:10]}"
        )

    model.load_state_dict(state_dict, strict=False)
    print(
        f"Pre-trained weights loaded: {len(state_dict)} tensors matched; "
        f"{len(missing_keys)} model keys missing, "
        f"{len(unmatched_keys)} checkpoint keys unmatched, "
        f"{len(shape_mismatches)} shape mismatches."
    )


def build_model(resolution, dino_checkpoint: Path, device: torch.device) -> D3FIRModel:
    # Mirror infer.py's fixed architecture for the bundled best checkpoints.
    width, height = resolution
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
    return D3FIRModel(config, str(dino_checkpoint) if dino_checkpoint else "", load_pretrained=False).to(device)


@torch.inference_mode()
def sample_fm(model: D3FIRModel, visible: torch.Tensor, steps: int,
              device: torch.device, precision: str) -> torch.Tensor:
    rgb_tokens, source_latent = model.encode_rgb(visible)
    if source_latent is None:
        raise RuntimeError("The selected model did not produce visible-light source latents.")

    latent = source_latent
    output = visible.clone()
    dt = 1.0 / steps
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    amp_enabled = device.type == "cuda" and precision != "fp32"
    for step in range(steps):
        time = torch.full((visible.shape[0],), step / steps, device=device, dtype=latent.dtype)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            velocity = model.dit(latent, t=time, cond=rgb_tokens, rgb_tokens=rgb_tokens)
            pixel_velocity, _ = model.decode(
                rgb=visible,
                rgb_latent=source_latent,
                velocity=velocity,
                rgb_tokens=rgb_tokens,
            )
        latent = latent + velocity * dt
        output = output + pixel_velocity * dt
    return output.clamp(0.0, 1.0)


def evaluate_dataset(args: argparse.Namespace, dataset_name: str, device: torch.device,
                     lpips_model: torch.nn.Module, ssim_metric: StructuralSimilarityIndexMeasure) -> Dict[str, Any]:
    spec = DATASET_CONFIGS[dataset_name]
    width, height = spec["resolution"]
    data_dir = args.data_root / dataset_name
    manifest_path = data_dir / spec["manifest"]
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    checkpoint_path = args.weights or (args.checkpoint_root / spec["checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    dataset = PairedInfraredDataset(data_dir, manifest_path, (width, height))
    if args.max_samples is not None:
        dataset.samples = dataset.samples[: args.max_samples]
    if len(dataset) == 0:
        raise ValueError(f"No evaluation samples in {manifest_path}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")

    result_dir = args.output_dir / dataset_name
    result_dir.mkdir(parents=True, exist_ok=True)
    generated_dir_saved = result_dir / "generated"
    if args.save_images:
        generated_dir_saved.mkdir(parents=True, exist_ok=True)

    model = build_model(spec["resolution"], args.dino_ckpt, device)
    load_pretrained_weights(model, checkpoint_path, strict=True)
    model.eval()
    model.requires_grad_(False)

    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    total_images = 0
    to_pil = ToPILImage()
    with tempfile.TemporaryDirectory(prefix="fid_", dir=result_dir) as temp_dir:
        real_dir = Path(temp_dir) / "real"
        generated_dir = Path(temp_dir) / "generated"
        real_dir.mkdir()
        generated_dir.mkdir()
        with torch.inference_mode():
            for batch_index, batch in enumerate(tqdm(loader, desc=dataset_name)):
                visible = batch["visible"].to(device, non_blocking=True)
                infrared = batch["infrared"].to(device, non_blocking=True).float()
                generated = sample_fm(model, visible, args.steps, device, args.precision).float()
                batch_count = visible.shape[0]
                lpips_scores = lpips_model(generated * 2.0 - 1.0, infrared * 2.0 - 1.0)
                ssim_score = ssim_metric(generated, infrared)
                mse = (generated - infrared).square().flatten(1).mean(dim=1)
                psnr_scores = -10.0 * torch.log10(mse.clamp_min(1e-12))
                totals["psnr"] += psnr_scores.sum().item()
                totals["ssim"] += ssim_score.item() * batch_count
                totals["lpips"] += lpips_scores.reshape(batch_count, -1).mean(dim=1).sum().item()
                total_images += batch_count
                for sample_index in range(batch_count):
                    image_id = batch_index * args.batch_size + sample_index
                    filename = f"{image_id:07d}.png"
                    to_pil(infrared[sample_index].cpu()).save(real_dir / filename)
                    to_pil(generated[sample_index].cpu()).save(generated_dir / filename)
                    if args.save_images:
                        to_pil(generated[sample_index].cpu()).save(generated_dir_saved / filename)
        fid_score = float(fid.compute_fid(
            str(generated_dir), str(real_dir), mode="clean",
            model_name="inception_v3", num_workers=args.num_workers,
            batch_size=max(1, min(32, args.batch_size * 8)),
        ))

    metrics = {
        "model": "repo.D3FIRModel",
        "checkpoint": str(checkpoint_path),
        "dataset": dataset_name,
        "num_images": total_images,
        "steps": args.steps,
        "precision": args.precision,
        "psnr": totals["psnr"] / total_images,
        "ssim": totals["ssim"] / total_images,
        "lpips": totals["lpips"] / total_images,
        "fid": fid_score,
    }
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be greater than zero")
    if args.weights is not None and len(args.datasets) != 1:
        raise ValueError("--weights override is only valid for a single --datasets run")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    results: List[Dict[str, Any]] = []
    for dataset_name in args.datasets:
        results.append(evaluate_dataset(args, dataset_name, device, lpips_model, ssim_metric))

    (args.output_dir / "summary.json").write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    csv_path = args.output_dir / "summary.csv"
    header = ["dataset", "psnr", "ssim", "lpips", "fid", "num_images", "steps", "precision", "checkpoint"]
    lines = [",".join(header)]
    for m in results:
        lines.append(",".join(str(m[k]) for k in header))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary: {csv_path}")


if __name__ == "__main__":
    main()
