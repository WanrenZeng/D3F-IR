#!/usr/bin/env python3
"""Command-line entry point for D3F-IR fine-tuning."""

import json
import os
from pathlib import Path
import sys

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d3f_ir.common import (
    DinoEncoder,
    FlowMatchingModel,
    configure_precision,
    log_parameter_counts,
    save_cfg,
    set_seed,
)
from d3f_ir.config import parse_args
from d3f_ir.data import build_infrared_loader
from d3f_ir.checkpoint import load_pretrained_weights
from d3f_ir.engine import train_one_epoch
from d3f_ir.validation import validate
from model.config import ModelConfig
def main() -> None:
    args = parse_args()

    if args.camera_name == "M3FD_Detection":
        args.input_size = [512, 384]
    else:
        args.input_size = [640, 512]

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)
    save_cfg(args.output_dir, args)

    train_loader = build_infrared_loader(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, input_size=tuple(args.input_size),
        split="train", args=args,
    )
    val_loader = build_infrared_loader(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, input_size=tuple(args.input_size),
        split="val", args=args,
    )

    input_size_hw = (args.input_size[1], args.input_size[0])
    teacher_encoder = DinoEncoder(
        input_size=input_size_hw, patch_size=args.patch_size,
        ckpt_path=args.dino_ckpt, trainable=False,
    ).to(device)
    teacher_encoder.eval()
    teacher_encoder.requires_grad_(False)

    model_config = ModelConfig(
        image_size=input_size_hw,
        patch_size=args.patch_size,
        latent_dim=args.latent_dim,
        time_token_count=args.time_token_count,
        dit_depth=args.dit_depth,
        dit_heads=args.dit_heads,
        dit_hidden=args.dit_hidden,
        dit_mlp_ratio=args.dit_mlp_ratio,
        dit_head_depth=args.dit_head_depth,
        dit_head_hidden=args.dit_head_hidden,
        dit_head_heads=args.dit_head_heads,
        dit_head_mlp_ratio=args.dit_head_mlp_ratio,
        pixel_predictor_hidden=args.decoder_hidden,
        pixel_predictor_layers=args.decoder_layers,
        pixel_predictor_heads=args.decoder_heads,
        pixel_predictor_mlp_ratio=args.decoder_mlp_ratio,
        fusion_heads=args.fusion_heads,
        shallow_channels=args.shallow_channels,
        align_dim=teacher_encoder.output_channels,
        align_layer=args.align_layer,
    )
    model = FlowMatchingModel(
        config=model_config, dino_checkpoint=args.dino_ckpt
    ).to(device)

    log_parameter_counts(model, teacher_encoder)
    if args.pretrain_ckpt:
        load_pretrained_weights(model, args.pretrain_ckpt)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=25, T_mult=1, eta_min=args.lr * 0.01
    )
    scaler, autocast_kwargs = configure_precision(args.precision)

    start_epoch = 0
    best_fid = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint.get("optimizer", optimizer.state_dict()))
        start_epoch = int(checkpoint.get("epoch", 0))
        best_fid = float(checkpoint.get("best_fid", best_fid))
        print(
            f"[Resume] Loaded checkpoint from {args.resume} at epoch "
            f"{start_epoch} with Best FID {best_fid}"
        )

    global_step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, args.epochs):
        train_stats, global_step = train_one_epoch(
            model, optimizer, train_loader, device, scaler,
            autocast_kwargs, args, epoch, global_step, teacher_encoder,
        )
        metrics = validate(
            model, val_loader, device, args,
            epoch=epoch,
        )
        metrics_dir = Path(args.vis_dir) / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_record = {
            "epoch": epoch + 1, "train": train_stats, "validate": metrics,
        }
        (metrics_dir / f"epoch_{epoch + 1:03d}.json").write_text(
            json.dumps(metrics_record, indent=2), encoding="utf-8"
        )
        print(
            f"[Epoch {epoch + 1:03d}] flow={train_stats['flow']:.4f} "
            f"recon={train_stats['recon']:.4f} total={train_stats['total']:.4f} "
            f"PSNR={metrics['psnr']:.3f} SSIM={metrics['ssim']:.4f} "
            f"LPIPS={metrics['lpips']:.4f} FID={metrics.get('fid', 999.0):.4f}"
        )

        current_fid = metrics.get("fid", 999.0)
        if current_fid < best_fid and current_fid > 0.0:
            best_fid = current_fid
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_fid": best_fid,
                },
                Path(args.output_dir) / "best.pt",
            )
        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_fid": best_fid,
            },
            Path(args.output_dir) / "latest.pt",
        )
        scheduler.step()


if __name__ == "__main__":
    main()
