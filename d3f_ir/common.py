#!/usr/bin/env python3

"""Shared utilities for visible-light-to-infrared flow matching."""

import argparse
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import transforms
import lpips
from tqdm import tqdm

from torchmetrics.image import StructuralSimilarityIndexMeasure
import torchvision.models as models

class InceptionLoss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.inception = models.inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception.eval()


        self.inception.aux_logits = False


        self.inception.fc = torch.nn.Identity()

    def forward(self, x, y):


        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        y = F.interpolate(y, size=(299, 299), mode='bilinear', align_corners=False)


        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        y = (y - mean) / std

        feat_x = self.inception(x)
        with torch.no_grad():
            feat_y = self.inception(y)

        return F.mse_loss(feat_x, feat_y)

from .evaluation_metrics import calculate_fid

def calculate_psnr_manual(img1, img2):


    mse = (img1 - img2) ** 2
    mse = mse.view(mse.shape[0], -1).mean(dim=1)


    psnr_val = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-10))
    return psnr_val

from model.encoder import DinoEncoder
from model.model import D3FIRModel, FlowMatchingModel
from torch.utils.data import DataLoader, Dataset
from dataset.data import InfraredDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    params = (p for p in module.parameters() if p.requires_grad or not trainable_only)
    return sum(p.numel() for p in params)


def log_parameter_counts(model: FlowMatchingModel, teacher: DinoEncoder) -> None:
    def fmt(n: int) -> str:
        return f"{n/1e6:.2f}M" if n >= 1e6 else str(n)

    encoder_params = _count_parameters(model.encoder, trainable_only=False)
    dit_params = _count_parameters(model.dit, trainable_only=False)
    predictor_params = _count_parameters(model.pixel_predictor, trainable_only=False)
    predictor_shallow = _count_parameters(model.pixel_predictor.shallow_branch, trainable_only=False) if model.pixel_predictor.shallow_branch is not None else 0
    predictor_gamma = _count_parameters(model.pixel_predictor.gamma_net, trainable_only=False) if model.pixel_predictor.gamma_net is not None else 0
    model_total = _count_parameters(model, trainable_only=False)
    model_trainable = _count_parameters(model, trainable_only=True)
    teacher_total = _count_parameters(teacher, trainable_only=False)

    print("[Params] model total", fmt(model_total), "trainable", fmt(model_trainable))
    print("[Params] encoder", fmt(encoder_params), "dit", fmt(dit_params), "Pixel Predictor", fmt(predictor_params))
    if model.pixel_predictor.shallow_branch is not None:
        print("[Params] Pixel Predictor shallow branch", fmt(predictor_shallow))
    if model.pixel_predictor.gamma_net is not None:
        print("[Params] Pixel Predictor gamma network", fmt(predictor_gamma))
    print("[Params] teacher (frozen) total", fmt(teacher_total))


def save_cfg(output_dir: str, args: argparse.Namespace) -> None:
    cfg_path = Path(output_dir) / "cfg.py"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Auto-generated training configuration\n", "cfg = {\n"]
    for k, v in sorted(vars(args).items()):
        lines.append(f"    '{k}': {repr(v)},\n")
    lines.append("}\n")
    cfg_path.write_text("".join(lines), encoding="utf-8")


def configure_precision(mode: str):
    if mode == "fp16":
        return GradScaler(), {"enabled": True, "dtype": torch.float16}
    if mode == "bf16":
        return None, {"enabled": True, "dtype": torch.bfloat16}
    return None, {"enabled": False}


def gamma_correct(img: torch.Tensor, gamma: float = 0.2) -> torch.Tensor:
    return torch.pow(torch.clamp(img, 0.0, 1.0), gamma)


def tensor_to_numpy(img: torch.Tensor) -> np.ndarray:
    return np.clip(img.detach().cpu().float().permute(1, 2, 0).numpy(), 0.0, 1.0)


def rggb_to_rgb_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[1] == 3:
        return tensor
    if tensor.dim() == 4:
        r, g1, g2, b = torch.chunk(tensor, 4, dim=1)
    else:
        r, g1, g2, b = torch.chunk(tensor, 4, dim=0)
    g = 0.5 * (g1 + g2)
    return torch.cat([r, g, b], dim=1 if tensor.dim() == 4 else 0)
