"""Command-line configuration."""

from pathlib import Path

from .common import *


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train D3F-IR with conditional flow matching")
    parser.add_argument(
        "--data_dir", type=str, default=str(project_root / "data" / "FLIR-align")
    )
    parser.add_argument("--camera_name", type=str, default="FLIR-align")
    parser.add_argument("--train_file", type=str, default="")
    parser.add_argument("--val_file", type=str, default="")
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--input_size", nargs=2, type=int, default=[512, 384], help="Input size in (W, H)")
    parser.add_argument("--rggb_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument(
        "--output_dir", type=str, default=str(project_root / "ckpt" / "FLIR-align")
    )
    parser.add_argument(
        "--vis_dir", type=str, default=str(project_root / "outputs" / "visualizations")
    )
    parser.add_argument("--vis_interval", type=int, default=5000)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--time_token_count", type=int, default=2)
    parser.add_argument("--flow_weight", type=float, default=1.0)
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--align_weight", type=float, default=1.0)
    parser.add_argument("--align_layer", type=int, default=None)
    parser.add_argument("--val_flow_steps", type=int, default=1)
    parser.add_argument(
        "--dino_ckpt",
        type=str,
        default=str(project_root / "ckpt" / "DINOv3" / "model.safetensors"),
    )
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--dit_depth", type=int, default=14)
    parser.add_argument("--dit_heads", type=int, default=8)
    parser.add_argument("--dit_hidden", type=int, default=384)
    parser.add_argument("--dit_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--dit_head_depth", type=int, default=2)
    parser.add_argument("--dit_head_hidden", type=int, default=None)
    parser.add_argument("--dit_head_heads", type=int, default=None)
    parser.add_argument("--dit_head_mlp_ratio", type=float, default=None)
    parser.add_argument("--decoder_hidden", type=int, default=512)
    parser.add_argument("--decoder_layers", type=int, default=6)
    parser.add_argument("--decoder_heads", type=int, default=16)
    parser.add_argument("--decoder_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--fusion_heads", type=int, default=4)
    parser.add_argument("--shallow_channels", type=int, default=96)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--pretrain_ckpt", type=str, default="", help="Path to pre-trained model checkpoint (from pretrain.py)")
    return parser.parse_args()
