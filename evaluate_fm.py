#!/usr/bin/env python3
"""Evaluate the saved visible-light-to-infrared flow-matching checkpoints."""

import argparse
import ast
import csv
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lpips
import torch
from cleanfid import fid
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent
RED_DIR = PROJECT_DIR.parent
LEGACY_MODEL_PATH = RED_DIR / "model" / "modelU_sim_DAB.py"
if str(RED_DIR) not in sys.path:
    sys.path.insert(0, str(RED_DIR))
legacy_spec = importlib.util.spec_from_file_location("legacy_model_u_sim_dab", LEGACY_MODEL_PATH)
if legacy_spec is None or legacy_spec.loader is None:
    raise ImportError(f"Could not load legacy model from {LEGACY_MODEL_PATH}")
legacy_model = importlib.util.module_from_spec(legacy_spec)
legacy_spec.loader.exec_module(legacy_model)
FlowMatchingModel = legacy_model.FlowMatchingModel

CHECKPOINT_ROOT = PROJECT_DIR / "ckpt"
DATA_ROOT = RED_DIR / "data"
DINO_CHECKPOINT = CHECKPOINT_ROOT / "DINOv3" / "model.safetensors"
DEFAULT_CHECKPOINT_FOLDERS = (
    "red-v-DAB-final_finetune1",
    "red-v-DAB-final_finetune2",
    "red-v-DAB-final_finetune2_fid",
)
DATASET_MANIFESTS = {
    "FLIR-align": "annotations_validation.json",
    "KAIST_clean": "test_set.json",
    "M3FD_Detection": "test_set.json",
}


class PairedInfraredDataset(Dataset):
    def __init__(self, data_root: Path, manifest_path: Path, resolution: Tuple[int, int]):
        self.data_root = data_root
        self.samples = json.loads(manifest_path.read_text(encoding="utf-8"))
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
        return {
            "visible": self.to_tensor(visible),
            "infrared": self.to_tensor(infrared),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_folders", nargs="+", default=list(DEFAULT_CHECKPOINT_FOLDERS))
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASET_MANIFESTS), default=list(DATASET_MANIFESTS))
    parser.add_argument("--weights", nargs="+", default=["best.pt", "best copy.pt"])
    parser.add_argument("--data_root", type=Path, default=DATA_ROOT)
    parser.add_argument("--checkpoint_root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--dino_ckpt", type=Path, default=DINO_CHECKPOINT)
    parser.add_argument("--results_dir", type=Path, default=PROJECT_DIR / "evaluation_results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def read_checkpoint_config(checkpoint_dir: Path) -> Dict[str, Any]:
    config_path = checkpoint_dir / "cfg.py"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint config not found: {config_path}")
    module = ast.parse(config_path.read_text(encoding="utf-8"))
    for statement in module.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "cfg" for target in statement.targets
        ):
            config = ast.literal_eval(statement.value)
            if isinstance(config, dict):
                return config
    raise ValueError(f"Could not read cfg dict from {config_path}")


def build_model(config: Dict[str, Any], dino_checkpoint: Path, device: torch.device) -> FlowMatchingModel:
    width, height = config["input_size"]
    if config.get("source_dist") != "rgb_reduce":
        raise ValueError("Expected checkpoint trained with visible-light latent source (rgb_reduce).")
    if config.get("prediction_mode") != "v" or config.get("flow_type", "FM") != "FM":
        raise ValueError("Expected FM velocity-prediction checkpoint.")
    return FlowMatchingModel(
        input_size=(height, width),
        patch_size=config["patch_size"],
        latent_dim=config["latent_dim"],
        flow_source="rgb_reduce",
        time_token_count=config["time_token_count"],
        dit_depth=config["dit_depth"],
        dit_heads=config["dit_heads"],
        dit_hidden=config["dit_hidden"],
        dit_mlp_ratio=config["dit_mlp_ratio"],
        dit_head_depth=config["dit_head_depth"],
        dit_head_hidden=config["dit_head_hidden"],
        dit_head_heads=config["dit_head_heads"],
        dit_head_mlp_ratio=config["dit_head_mlp_ratio"],
        decoder_hidden=config["decoder_hidden"],
        decoder_layers=config["decoder_layers"],
        decoder_heads=config["decoder_heads"],
        decoder_mlp_ratio=config["decoder_mlp_ratio"],
        fusion_heads=config["fusion_heads"],
        shallow_channels=config["shallow_channels"],
        dino_ckpt=str(dino_checkpoint),
        align_dim=384,
        align_layer=config.get("align_layer"),
    ).to(device)


def load_weights(model: FlowMatchingModel, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint has no 'model' state dict: {checkpoint_path}")
    result = model.load_state_dict(checkpoint["model"], strict=False)
    if result.missing_keys:
        raise RuntimeError(f"Missing model weights: {result.missing_keys[:20]}")
    unexpected = [key for key in result.unexpected_keys if not key.startswith("inception_loss_module.")]
    if unexpected:
        raise RuntimeError(f"Unexpected model weights: {unexpected[:20]}")
    print(f"Loaded model tensors; ignored {len(result.unexpected_keys)} training-only Inception tensors.")


def sample_fm(model: FlowMatchingModel, visible: torch.Tensor, steps: int, amp_enabled: bool) -> torch.Tensor:
    visible_tokens, source_latent = model.encode_rgb(visible)
    if source_latent is None:
        raise RuntimeError("Model did not produce visible-light source latents.")
    image = visible.clone()
    latent = source_latent
    dt = 1.0 / steps
    for step in range(steps):
        time = torch.full((visible.shape[0],), step / steps, device=visible.device, dtype=latent.dtype)
        with torch.autocast(device_type=visible.device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            velocity = model.dit(latent, t=time, cond=visible_tokens, rgb_tokens=visible_tokens)
            pixel_velocity, _ = model.decode(
                visible_tokens, velocity, rgb=visible, context_tokens=visible_tokens, rgb_latent=source_latent
            )
        latent = latent + velocity * dt
        image = image + pixel_velocity * dt
    return image.clamp(0.0, 1.0)


def evaluate_checkpoint(
    args: argparse.Namespace,
    checkpoint_folder: str,
    dataset_name: str,
    weight_name: str,
    device: torch.device,
    lpips_model: torch.nn.Module,
    ssim_metric: StructuralSimilarityIndexMeasure,
) -> Dict[str, Any]:
    checkpoint_dir = args.checkpoint_root / checkpoint_folder / dataset_name
    checkpoint_path = checkpoint_dir / weight_name
    config = read_checkpoint_config(checkpoint_dir)
    if config.get("camera_name") != dataset_name:
        raise ValueError(f"Dataset mismatch in {checkpoint_path}: {config.get('camera_name')!r}")

    data_dir = args.data_root / dataset_name
    manifest_path = data_dir / DATASET_MANIFESTS[dataset_name]
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    width, height = config["input_size"]
    dataset = PairedInfraredDataset(data_dir, manifest_path, (width, height))
    if args.max_samples is not None:
        dataset.samples = dataset.samples[: args.max_samples]
    if len(dataset) == 0:
        raise ValueError(f"No evaluation samples in {manifest_path}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    result_dir = args.results_dir / checkpoint_folder / dataset_name / Path(weight_name).stem.replace(" ", "_")
    result_dir.mkdir(parents=True, exist_ok=True)
    steps = args.steps or int(config.get("val_flow_steps", 10))
    amp_enabled = args.precision == "bf16" and device.type == "cuda"
    model = build_model(config, args.dino_ckpt, device)
    load_weights(model, checkpoint_path)
    model.eval()

    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    total_images = 0
    to_pil = ToPILImage()
    with tempfile.TemporaryDirectory(prefix="fid_", dir=result_dir) as temporary_dir:
        real_dir = Path(temporary_dir) / "real"
        generated_dir = Path(temporary_dir) / "generated"
        real_dir.mkdir()
        generated_dir.mkdir()
        with torch.inference_mode():
            for batch_index, batch in enumerate(tqdm(loader, desc=f"{checkpoint_folder}/{dataset_name}/{weight_name}")):
                visible = batch["visible"].to(device, non_blocking=True)
                infrared = batch["infrared"].to(device, non_blocking=True)
                generated = sample_fm(model, visible, steps, amp_enabled).float()
                infrared = infrared.float()
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
        fid_score = fid.compute_fid(
            str(generated_dir),
            str(real_dir),
            mode="clean",
            model_name="inception_v3",
            num_workers=args.num_workers,
            batch_size=max(1, min(32, args.batch_size * 8)),
        )

    metrics = {
        "checkpoint": str(checkpoint_path),
        "dataset": dataset_name,
        "num_images": total_images,
        "steps": steps,
        "psnr": totals["psnr"] / total_images,
        "ssim": totals["ssim"] / total_images,
        "lpips": totals["lpips"] / total_images,
        "fid": float(fid_score),
    }
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if not args.dino_ckpt.exists():
        raise FileNotFoundError(f"DINO checkpoint not found: {args.dino_ckpt}")
    device = torch.device(args.device)
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    results: List[Dict[str, Any]] = []
    errors: List[str] = []
    for checkpoint_folder in args.checkpoint_folders:
        for dataset_name in args.datasets:
            for weight_name in args.weights:
                checkpoint_path = args.checkpoint_root / checkpoint_folder / dataset_name / weight_name
                if not checkpoint_path.is_file():
                    print(f"Skipping missing checkpoint: {checkpoint_path}")
                    continue
                try:
                    metrics = evaluate_checkpoint(
                        args, checkpoint_folder, dataset_name, weight_name, device, lpips_model, ssim_metric
                    )
                    results.append(metrics)
                    with (args.results_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
                        writer = csv.DictWriter(file, fieldnames=list(metrics))
                        writer.writeheader()
                        writer.writerows(results)
                except Exception as error:
                    message = f"{checkpoint_path}: {type(error).__name__}: {error}"
                    print(f"ERROR: {message}")
                    errors.append(message)
    (args.results_dir / "summary.json").write_text(
        json.dumps({"results": results, "errors": errors}, indent=2), encoding="utf-8"
    )
    print(f"Completed {len(results)} evaluations; {len(errors)} failed.")
    print(f"Summary: {args.results_dir / 'summary.json'}")
    if errors:
        raise RuntimeError(f"{len(errors)} checkpoint evaluations failed; see summary.json")


if __name__ == "__main__":
    main()
