import argparse
import json
import os
import shutil
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision

# Import model components
from model.modelU_sim_DAB import FlowMatchingModel
from model.flow_matching import build_flow_matching
from dataset.data import InfraredDataset as TrainInfraredDataset

# Import evaluation function
# Make sure to import properly if directory structure allows
import sys
import importlib.util
from types import ModuleType

def import_evaluation_module():
    module_name = "evaluation_script"
    file_path = os.path.join(os.path.dirname(__file__), "evaluation.py")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

eval_module = import_evaluation_module()
calculate_lpips_ssim_psnr = eval_module.calculate_lpips_ssim_psnr
calculate_fid = eval_module.calculate_fid
create_temp_test_dir = eval_module.create_temp_test_dir

class DiffusionScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cuda'):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def add_noise(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise, noise

def sample_joint(model: FlowMatchingModel, base_tokens: torch.Tensor, context_tokens: torch.Tensor, rgb_tokens_full: torch.Tensor, rgb_img: torch.Tensor, steps: int, flow_type: str = "FM", prediction_mode: str = "x0", padding_size: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    b = base_tokens.size(0)
    device = base_tokens.device
    
    x_curr = base_tokens.clone()
    I_curr = rgb_img.clone() # Unpadded RGB
    
    # Pre-calculate padded RGB for decoder if needed
    if padding_size > 0:
        pad = padding_size
        rgb_padded = F.pad(rgb_img, (pad, pad, pad, pad), mode='constant', value=0.0)
    else:
        rgb_padded = rgb_img

    if flow_type == "DF":
        if not hasattr(model, 'scheduler'):
             model.scheduler = DiffusionScheduler(device=device)
        scheduler = model.scheduler
        total_steps = scheduler.num_timesteps
        if steps > total_steps: steps = total_steps
        
        timesteps = torch.linspace(total_steps - 1, 0, steps, device=device).long()
        timesteps_list = timesteps.tolist()
        
        for i, t_step in enumerate(timesteps_list):
             t = torch.full((b,), t_step, device=device, dtype=torch.long)
             pred_noise = model.dit(x_curr, t=t, cond=context_tokens, rgb_tokens=rgb_tokens_full)
             
             alpha_cumprod = scheduler.alphas_cumprod[t_step]
             sqrt_alpha = torch.sqrt(alpha_cumprod)
             sqrt_one_minus = torch.sqrt(1 - alpha_cumprod)
             pred_target_latent = (x_curr - sqrt_one_minus * pred_noise) / (sqrt_alpha + 1e-8)
             
             # Use padded RGB for decoder
             pred_target_image_padded, _ = model.decode(
                rgb_tokens_full,
                pred_target_latent,
                rgb=rgb_padded, # Padded input to decoder
                context_tokens=context_tokens,
                rgb_latent=base_tokens
             )
             
             # Crop back to unpadded size for flow update
             if padding_size > 0:
                 pred_target_image = pred_target_image_padded[:, :, padding_size:-padding_size, padding_size:-padding_size]
             else:
                 pred_target_image = pred_target_image_padded
             
             if i < len(timesteps_list) - 1:
                 prev_t = timesteps_list[i + 1]
                 alpha_cumprod_prev = scheduler.alphas_cumprod[prev_t]
             else:
                 alpha_cumprod_prev = torch.tensor(1.0, device=device)
                 
             dir_xt = torch.sqrt(1 - alpha_cumprod_prev) * pred_noise
             x_curr = torch.sqrt(alpha_cumprod_prev) * pred_target_latent + dir_xt
             
             I_curr = I_curr + (pred_target_image - rgb_img) * (1.0 / steps)

    else:
        # FM / MF / CF
        dt = 1.0 / steps
        for i in range(steps):
            t_val = i / steps
            t = torch.full((b,), t_val, device=device, dtype=x_curr.dtype)
            
            output = model.dit(x_curr, t=t, cond=context_tokens, rgb_tokens=rgb_tokens_full)
            
            if prediction_mode == "v":
                v_latent = output
                x_curr = x_curr + v_latent * dt
                
                # Decoder gets v_latent. 
                # Decode: (rgb, latent) -> pixel.
                # If we use padded rgb, we get padded pixel output.
                # We need to crop it.
                v_pixel_padded, _ = model.decode(
                    rgb_tokens_full,
                    v_latent, # decoding velocity
                    rgb=rgb_padded, # Padded input
                    context_tokens=context_tokens,
                    rgb_latent=base_tokens
                )
                
                if padding_size > 0:
                    v_pixel = v_pixel_padded[:, :, padding_size:-padding_size, padding_size:-padding_size]
                else:
                    v_pixel = v_pixel_padded
                    
                I_curr = I_curr + v_pixel * dt
                
            else: # x0
                pred_target_latent = output
                v_latent = (pred_target_latent - x_curr) / (1.0 - t_val + 1e-6)

                pred_target_image_padded, _ = model.decode(
                    rgb_tokens_full,
                    pred_target_latent,
                    rgb=rgb_padded, # Padded input
                    context_tokens=context_tokens,
                    rgb_latent=base_tokens
                )
                
                if padding_size > 0:
                    pred_target_image = pred_target_image_padded[:, :, padding_size:-padding_size, padding_size:-padding_size]
                else:
                    pred_target_image = pred_target_image_padded
                
                delta_pixel = pred_target_image - rgb_img
                
                x_curr = x_curr + v_latent * dt
                I_curr = I_curr + delta_pixel * dt
        
    return x_curr, I_curr

class SimpleBaseDataset:
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]

def make_infrared_dataset_list(root):
    if not os.path.exists(root):
        print(f"Dataset root {root} does not exist!")
        return []
    files = sorted(os.listdir(root))
    rgb_files = {f.split('_RGB')[0]: f for f in files if '_RGB.jpg' in f}
    ir_files = {f.split('_PreviewData')[0]: f for f in files if '_PreviewData.jpeg' in f}
    
    data = []
    keys = sorted(list(set(rgb_files.keys()) & set(ir_files.keys())))
    print(f"Found {len(keys)} paired images in {root}")
    for k in keys:
        data.append({
            "infrared_path": ir_files[k],
            "vision_path": rgb_files[k],
            "description_vision_only": "",
            "description_instruct": ""
        })
    return data

def get_args():
    parser = argparse.ArgumentParser()
    # Model args matching train_fm.sh
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--source_dist", default="rgb_reduce") # inferred from train_fm.sh calling script.. wait, train_fm.sh defaults SOURCE_DIST="gaussian"
    
    # Wait, train_fm.sh: SOURCE_DIST=${4:-"gaussian"}
    # But later, train_fm.py default is "rgb_reduce".
    # And train_fm.sh passes --source_dist "${SOURCE_DIST}"
    # So if I run without arg 4, it uses "gaussian".
    # I should check cfg.py in the checkpoint folder to be 100% sure!
    
    parser.add_argument("--time_token_count", type=int, default=2)
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
    parser.add_argument("--shallow_channels", type=int, default=80) # 64 from sh
    parser.add_argument("--align_layer", type=int, default=None)
    
    parser.add_argument("--flow_type", default="FM") # default in sh
    parser.add_argument("--prediction_mode", default="v", choices=["x0", "v"])
    parser.add_argument("--fixed_t1", action="store_true", default=False)
    
    parser.add_argument("--dino_ckpt", type=str, default="/DINOv3/model.safetensors")
    
    # Paths
    parser.add_argument("--checkpoint", type=str, default="./ours_ckpt/red-v-DAB/FLIR-align/best.pt")
    # Using relative path assuming run from red/
    parser.add_argument("--data_root", type=str, default="data") # red/data
    parser.add_argument("--json_path", type=str, default="data/annotations_validation.json")
    parser.add_argument("--output_dir", type=str, default="evaluation_reg/FLIR-align")
    
    parser.add_argument("--device", type=str, default="cuda:7")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10) # VAL_STEPS in sh is 10
    parser.add_argument("--camera_name", type=str, default="FLIR-align")
    parser.add_argument("--padding_size", type=int, default=16, help="Padding size for inference")
    
    return parser.parse_args()

def load_config_from_file(args):
    # Try to load cfg.py from checkpoint dir
    ckpt_dir = os.path.dirname(args.checkpoint)
    cfg_path = os.path.join(ckpt_dir, "cfg.py")
    if os.path.exists(cfg_path):
        print(f"Loading config from {cfg_path}")
        import uuid
        import importlib.util
        spec = importlib.util.spec_from_file_location(f"cfg_{uuid.uuid4().hex}", cfg_path)
        cfg_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_module)
        cfg = cfg_module.cfg
        
        # Override args with cfg if they exist, but prioritize args provided explicitly? 
        # Actually usually best to use cfg for model params.
        # Mapping cfg keys to args
        for k, v in cfg.items():
            if hasattr(args, k):
                # Only override model architecture params, not paths if possible?
                # Or just override everything and let user pass args to override back if needed?
                # For simplicity, let's update args with cfg for model params
                if k in ["dit_depth", "dit_heads", "dit_hidden", "dit_mlp_ratio", 
                         "decoder_hidden", "decoder_layers", "decoder_heads", "decoder_mlp_ratio",
                         "latent_dim", "patch_size", "time_token_count", "source_dist", "flow_type",
                         "prediction_mode", "dit_head_depth", "dit_head_hidden", "dit_head_heads", 
                         "dit_head_mlp_ratio", "align_layer"]:
                     setattr(args, k, v)
                     print(f"Config: {k} = {v}")
    return args

def main():
    args = get_args()
    args = load_config_from_file(args)
    
    device = torch.device(args.device)
    
    

   # Determine input size and dataset paths based on camera_name
    if args.camera_name == "M3FD_Detection":
        input_size = [512, 384] # [W, H]
        data_root = "/share402/fengdiedong/up2/red/data2/M3FD_Detection"
        json_path = os.path.join(data_root, "test_set.json")
    elif args.camera_name == "KAIST_clean":
        input_size = [640, 512] # [W, H] (KAIST usually 640x512 like FLIR?)
        # Double check KAIST resolution in train3df0c? 
        # train3df0c only checks M3FD_Detection for specific size change [512, 384], else [640, 512].
        data_root = "/root/zengpeiyi/work_base/dataset/KAIST_clean"
        json_path = os.path.join(data_root, "test_set.json")
    else:
        # Default (FLIR-align etc)
        input_size = [640, 512]
        data_root = "./data/FLIR-align"
        json_path = os.path.join(data_root, "annotations_validation.json")
    
    # Allow override if arguments provided? The user said "allow configuring via camera name".
    # We can overwrite args.data_root and args.json_path 
    args.data_root = data_root
    args.json_path = json_path
    
    input_size_hw = (input_size[1], input_size[0])

    # Load Model
    print("Building model...")
    model = FlowMatchingModel(
        input_size=input_size_hw,
        patch_size=16,
        latent_dim=args.latent_dim,
        flow_source=args.source_dist,
        time_token_count=args.time_token_count,
        dit_depth=args.dit_depth,
        dit_heads=args.dit_heads,
        dit_hidden=args.dit_hidden,
        dit_mlp_ratio=args.dit_mlp_ratio,
        dit_head_depth=args.dit_head_depth,
        dit_head_hidden=args.dit_head_hidden,
        dit_head_heads=args.dit_head_heads,
        dit_head_mlp_ratio=args.dit_head_mlp_ratio,
        decoder_hidden=args.decoder_hidden,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        decoder_mlp_ratio=args.decoder_mlp_ratio,
        fusion_heads=args.fusion_heads,
        shallow_channels=args.shallow_channels,
        dino_ckpt=args.dino_ckpt,
        align_dim=384, 
        align_layer=args.align_layer,
    ).to(device)

    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["model"]
    for key in list(state_dict.keys()):
        if "decoder.rope.freqs" in key:
             # Check if key exists in model
             if key in model.state_dict() and state_dict[key].shape != model.state_dict()[key].shape:
                 print(f"Removing {key} from checkpoint due to shape mismatch: {state_dict[key].shape} != {model.state_dict()[key].shape}")
                 del state_dict[key]
    model.load_state_dict(state_dict, strict=False)
    
    # Inject Inference Wrapper if padding is requested
    if args.padding_size > 0:
        print(f"Injecting ShallowUNetEncoderInference with padding_size={args.padding_size}")
        from model.model_inference import ShallowUNetEncoderInference
        # Wrap the shallow_branch
        # Accessing nested attribute
        # Check if attribute exists
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'shallow_branch'):
            model.decoder.shallow_branch = ShallowUNetEncoderInference(model.decoder.shallow_branch, padding_size=args.padding_size)
        else:
            print("Warning: Could not find model.decoder.shallow_branch to wrap.")
            
    model.eval()

    # Load Dataset
    if os.path.exists(args.json_path):
        print(f"Loading dataset from {args.json_path}")
        with open(args.json_path, 'r') as f:
            data_list = json.load(f)
    else:
        print(f"JSON path {args.json_path} not found. Scanning folder {args.data_root}...")
        data_list = make_infrared_dataset_list(args.data_root)
        if not data_list:
             raise FileNotFoundError(f"Could not load dataset from {args.json_path} or {args.data_root}")
    
    base_ds = SimpleBaseDataset(data_list)
    
    dataset = TrainInfraredDataset(
        base_ds, 
        dataset_root=args.data_root, 
        condition_size=input_size, 
        target_size=input_size,
        resize_resolution=input_size
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    os.makedirs(args.output_dir, exist_ok=True)
    # Save concatenated images to a separate directory to avoid interfering with FID calculation
    concat_dir = args.output_dir.rstrip('/') + "_concat"
    os.makedirs(concat_dir, exist_ok=True)
    
    # Inference Loop
    print("Starting inference...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            rgb = batch["condition_0"].to(device)
            draw = batch["image"].to(device)
            context = batch["condition_0"].to(device)

            # Match train3df0c_sim_dino.py validate logic
            # Note: tokens depend on unpadded image size if source_dist etc are standard.
            # Usually input_size determines tokens but here we have fixed patch_size/count?
            # model.encode_rgb(rgb) -> calls internal DINO.
            # DINO usually resizes or processes full image.
            # We should pass unpadded 'rgb' to generate 'tokens' that correspond to unpadded geometry.
            rgb_tokens_full, rgb_tokens_latent = model.encode_rgb(rgb)
            raw_tokens_full, raw_tokens_latent = model.encode_raw(draw)
            context_tokens_full, _ = model.encoder(context)

            # Pad RGB for inference if requested
            pad = args.padding_size
            if pad > 0:
                # Pad RGB with reflect or zero? User said "Pad input rgb with 0 pixels"
                rgb_padded = F.pad(rgb, (pad, pad, pad, pad), mode='constant', value=0.0)
            else:
                rgb_padded = rgb
            
            if args.source_dist == "rgb_reduce" and rgb_tokens_latent is None:
                 raise RuntimeError("RGB latent reduction is required when source_dist is rgb_reduce")
            
            source_tokens = rgb_tokens_latent if args.source_dist == "rgb_reduce" else None

            # fallback_source logic: Match train3df0c_sim_dino.py
            fallback_source = source_tokens if source_tokens is not None else torch.randn_like(raw_tokens_latent)

            # NOTE: rgb_tokens_full comes from UNPADDED rgb.
            # We pass unpadded original 'rgb' as 'rgb_img' to sample_joint.
            # We pass 'padding_size' so sample_joint can pad internally for the decoder.
            pred_latent, pred_image = sample_joint(
                model=model,
                base_tokens=fallback_source,
                context_tokens=context_tokens_full,
                rgb_tokens_full=rgb_tokens_full, # Unpadded DIT condition
                rgb_img=rgb, # Unpadded original RGB
                steps=args.steps,
                flow_type=args.flow_type,
                prediction_mode=args.prediction_mode,
                padding_size=args.padding_size # Request padding logic
            )
            
            # Since sample_joint now handles padding/unpadding internally, the output is already unpadded (cropped)
            
            decoded = torch.clamp(pred_image, 0.0, 1.0)
            gt_image = torch.clamp(draw, 0.0, 1.0)
            
            # Save image
            item_info = data_list[i]
            ir_path = item_info["infrared_path"]
            filename = os.path.basename(ir_path)
            
            save_path = os.path.join(args.output_dir, filename)
            
            img_tensor = decoded[0].cpu() # shape [C, H, W]
            to_pil = torchvision.transforms.ToPILImage()
            img_pil = to_pil(img_tensor)
            img_pil.save(save_path)

            # Save concatenated image
            gt_tensor = gt_image[0].cpu()
            gt_pil = to_pil(gt_tensor)
            
            # Concatenate horizontally
            w, h = img_pil.size
            if gt_pil.size != (w, h):
                gt_pil = gt_pil.resize((w, h))
            
            # If padding was used, we can reconstruct the padded visualization if really needed,
            # but since we moved logic into sample_joint and return cropped, we skip "padded_..." output for now
            # unless we modify sample_joint to return it too.
            # For now, just standard output.

            cat_img = Image.new('RGB', (w * 2, h))
            cat_img.paste(img_pil, (0, 0))
            cat_img.paste(gt_pil, (w, 0))
            # def sample_joint(..., rgb_img, ...):
            #    I_curr = rgb_img.clone()
            
            # If the flow matching is defined on the pixel space `I_curr`, and we want to generate the padded image, `I_curr` should be padded.
            # So passing `rgb_padded` as `rgb_img` IS correct if we want to run the flow on the padded image.
            
            # BUT, maybe I missed something in `model.dit` arguments inside `sample_joint`?
            # `pred_noise = model.dit(x_curr, t=t, cond=context_tokens, rgb_tokens=rgb_tokens_full)`
            # It uses `rgb_tokens_full`.
            #
            # Maybe the user thinks `rgb_tokens_full` was derived from padded image?
            # Code:
            # rgb_tokens_full, rgb_tokens_latent = model.encode_rgb(rgb)  <-- rgb is unpadded here.
            # ...
            # rgb_padded = F.pad(...)
            # ...
            # sample_joint(..., rgb_tokens_full=rgb_tokens_full, rgb_img=rgb_padded)
            
            # This looks correct according to "shallow branch uses padded, dit uses unpadded condition".
            # The only ambiguity is if `sample_joint` logic assumes `rgb_img` matches `base_tokens` in some way?
            # `x_curr = base_tokens.clone()`
            # `base_tokens` comes from `raw_tokens_latent` (unpadded).
            #
            # `pred_image` comes from `model.decode`.
            # `model.decode` calls `shallow_branch(rgb, tokens)`.
            # If we pass `rgb_padded`, `shallow_branch` returns padded image.
            # So `pred_image` is padded.
            #
            # `I_curr` (from padded rgb) is padded.
            # `delta_pixel = pred_image - rgb_img`. Padded - Padded. OK.
            # `I_curr = I_curr + delta_pixel * dt`. OK.
            #
            # `pred_latent` comes from `model.dit` (unpadded).
            # `x_curr` (unpadded) updated with `v_latent` (unpadded). OK.
            #
            # So everything seems consistent with "Pixel flow on padded space, Latent flow on unpadded space".
            #
            # Wait, `sample_joint` signature in my `read_file` output:
            # `sample_joint(model, base_tokens, context_tokens, rgb_tokens_full, rgb_img, ...)`
            #
            # Perhaps the user saw me saying "dit input is also padded" in my explanation?
            # Or maybe I edited something wrong?
            #
            # Let's look at the Replace call again.
            # It looked correct.
            #
            # Maybe the user implies that `sample_joint` shouldn't be running on padded pixels?
            # If we pad the input to `shallow_branch`, the output `pred_image` IS padded.
            # If `sample_joint` expects `pred_image` to be same size as `rgb_img`, then `rgb_img` MUST be padded.
            #
            # UNLESS the user wants to CROP the output of `shallow_branch` immediately inside `decode`?
            # "Re-implement a new model inference architecture... strip padding... modulate... put back... Output network finishes processing (output is padded). Post-Process: Remove padding."
            # The user explicitly said output is padded.
            
            # Let's assume the user is just confused or I missed a subtle dependency.
            # "you padded what is given to dit"
            # Did I?
            # `rgb_tokens_full` is passed to DiT.
            # `rgb_tokens_full` comes from `rgb` (lines 427).
            # `rgb` is from `batch['condition_0']`. UNPADDED.
            # `rgb_padded` is created AFTER `encode_rgb`.
            
            # I will double check the code I wrote.
            #
            # Code:
            # rgb_tokens_full, rgb_tokens_latent = model.encode_rgb(rgb)
            # ...
            # pred_latent, pred_image = sample_joint(..., rgb_tokens_full=rgb_tokens_full, rgb_img=rgb_padded, ...)
            #
            # It seems correct.
            #
            # Wait, `sample_joint` is defined in `evaluate_fmc.py`.
            # Code:
            # pred_noise = model.dit(x_curr, t=t, cond=context_tokens, rgb_tokens=rgb_tokens_full)
            #
            # The only way I could be "padding what is given to dit" is if `x_curr` or `base_tokens` was padded?
            # `base_tokens` comes from `fallback_source`.
            # `fallback_source` comes from `source_tokens` (rgb latent) or `raw_tokens_latent`.
            # `rgb_tokens_latent` from `model.encode_rgb(rgb)`.
            # `raw_tokens_latent` from `model.encode_raw(draw)`.
            # Both unpadded.
            
            # So unless `model.dit` takes `rgb_img` (it doesn't in the arguments list I see), I am fine.
            
            # Wait. `sample_joint` takes `rgb_img`. Does it pass it to `dit`?
            # `pred_noise = model.dit(x_curr, t=t, cond=context_tokens, rgb_tokens=rgb_tokens_full)`
            # No.
            
            # Maybe the user thinks `rgb_tokens_full` is "what is given to dit" (which is true) and thinks I padded it?
            # I definitely didn't pad `rgb` before `encode_rgb`.
            
            # BUT! Let's look at the Diff.
            # 
            # Old Code:
            # rgb_tokens_full, rgb_tokens_latent = model.encode_rgb(rgb)
            # ...
            # sample_joint(..., rgb_img=rgb, ...)
            #
            # New Code:
            # rgb_tokens_full, rgb_tokens_latent = model.encode_rgb(rgb)
            # rgb_padded = ...
            # sample_joint(..., rgb_img=rgb_padded, ...)
            
            # Is it possible that `model.dit` is accessing `rgb_img` via some side channel? Unlikely.
            
            # Is it possible the user simply misunderstood my change?
            # "you padded what is given to dit"
            # Maybe they mean `x_curr`?
            #
            # OR... maybe they mean `rgb_tokens_full` SHOULD be padded?
            # No, "dit condition input... unpadded rgb".
            #
            # OK, I'll assume my code is actually correct but maybe the user wants me to be EXPLICIT about unpadded tokens.
            # Or maybe I should add a comment explaining why `rgb_tokens_full` is unpadded?
            
            # Wait, I see `dino_ckpt` in args. `model.encode_rgb` uses DINO.
            # If DINO sees unpadded RGB, it produces unpadded tokens.
            # Correct.
            
            # Let's just reassure the user and maybe clean up the comments.
            # "I'm passing rgb_padded to sample_joint. sample_joint passes this to model.decode. model.decode passes it to shallow_branch."
            # "sample_joint DOES NOT pass rgb_padded to model.dit."
            
            # Wait!
            # Did the user mean `sample_joint` argument `rgb_img` IS what is given to DiT?
            # In some implementations, DiT might take the image?
            # Checked `sample_joint` implementation in `read_file`:
            # `pred_noise = model.dit(x_curr, t=t, cond=context_tokens, rgb_tokens=rgb_tokens_full)`
            # No input image here.
            
            # Maybe the mistake is semantic?
            # "You padded what is given to dit" => maybe they think `rgb_tokens_full` is derived from `rgb_padded`?
            # But it is not.
            
            # Let's look closely at where I inserted the padding code.
            #
            # ```python
            # rgb_tokens_full, rgb_tokens_latent = model.encode_rgb(rgb)
            # raw_tokens_full, raw_tokens_latent = model.encode_raw(draw)
            # context_tokens_full, _ = model.encoder(context)
            #
            # # Pad RGB for inference if requested   <-- Inserted AFTER encode_rgb
            # pad = args.padding_size
            # ...
            # ```
            #
            # It is physically after.
            #
            # However, I will edit the file to make it 100% clear and maybe rename variables to `rgb_unpadded` for the DINO part.
            #
            # Wait! "dit的条件输入" (DiT's condition input).
            # `rgb_tokens_full` IS the condition.
            #
            # Maybe the user wants me to verify `base_tokens`?
            # `base_tokens` is the starting point for `x_curr`.
            # `x_curr` is input to `model.dit`.
            # `base_tokens` comes from `fallback_source`.
            # `fallback_source` from `source_tokens` (rgb_tokens_latent) or `raw_tokens_latent`.
            # These are unpadded.
            #
            # So `x_curr` is unpadded.
            #
            # BUT `sample_joint` updates `I_curr` (image flow).
            # `I_curr` starts as `rgb_img.clone()`.
            # If `sample_joint` performs "Joint" flow (DiT on latent, Decode on image), then `I_curr` being padded is correct if output is padded.
            
            # Is it possible that `sample_joint` logic assumes `x_curr` and `I_curr` correspond to the same spatial grid?
            # `x_curr` is latent (1/16 size).
            # `I_curr` is image.
            # Unpadded image 640x512 -> Latent 40x32.
            # Padded image (say +64 pad) -> 768x640.
            #
            # If `x_curr` corresponds to 640x512, but `I_curr` corresponds to 768x640.
            # Does `sample_joint` care?
            # `sample_joint` calculates `pred_target_image` from `pred_target_latent` (unpadded).
            # `model.decode` takes `pred_target_latent` (unpadded) and `rgb` (padded).
            #
            # In `model.decode`:
            # `shallow_branch(rgb, tokens)`
            # `tokens` comes from `pred_target_latent` (via projection/direct usage).
            # So `tokens` are unpadded size.
            # `rgb` is padded size.
            # `shallow_branch` handles the mismatch (which is exactly what I implemented in `ShallowUNetEncoderInference`).
            #
            # So `pred_target_image` will be PADDED size.
            #
            # `delta_pixel = pred_target_image - rgb_img`.
            # `rgb_img` (passsed as `rgb_padded`) is PADDED size.
            # So `delta_pixel` is consistent.
            #
            # Everything seems correct.
            #
            # Why does the user think it's wrong?
            # "你把它给dit的也pad了" (You also padded what you gave to dit).
            #
            # Maybe `context_tokens`? No.
            # Maybe `rgb_tokens_full`? No.
            #
            # Wait, did I edit `sample_joint` definition? No.
            #
            # Let's assume the user made a mistake in reading the code, OR I am missing something trivial.
            # I will re-affirm that `rgb_tokens_full` comes from unpadded RGB.
            #
            # Oh! I see one possibility.
            # In my previous edit I replaced:
            #
            # ```python
            # rgb_tokens_full, rgb_tokens_latent = model.encode_rgb(rgb)
            # ...
            # rgb_padded = ...
            # ```
            #
            # I'll just restore the explicit `rgb` usage in comments or variables to ensure clarity?
            #
            # Wait, maybe the user thinks `rgb_padded` is passed to `rgb_tokens_full`? No.
            #
            # Let's look at the code block I sent for `evaluate_fmc.py`.
            #
            # ```python
            # rgb_tokens_full, rgb_tokens_latent = model.encode_rgb(rgb)
            # raw_tokens_full, raw_tokens_latent = model.encode_raw(draw)
            # context_tokens_full, _ = model.encoder(context)
            #
            # # Pad RGB for inference if requested
            # pad = args.padding_size
            # if pad > 0:
            #     rgb_padded = F.pad(rgb, ...)
            # ...
            # sample_joint(..., rgb_tokens_full=rgb_tokens_full, rgb_img=rgb_padded)
            # ```
            #
            # I am confusing myself. The code looks right.
            #
            # Could it be that `raw` (draw) or `context` should also be padded? No.
            #
            # Let's try to simply explain the flow to the user in code via a comment or variable renaming to prove it's correct.
            # "rgb_unpadded = rgb"
            # "rgb_tokens_full = model.encode_rgb(rgb_unpadded)"
            #
            # Also, I will check if `sample_joint` uses `rgb_img` for anything else.
            # It uses it for `I_curr + ...`
            #
            # What if the user implies that `sample_joint` expects `rgb_img` to be unpadded, and `model.decode` should handle the padding internally?
            # If `sample_joint` sees unpadded `rgb_img`:
            # `I_curr` is unpadded.
            # `model.decode(..., rgb=rgb_img_unpadded)`
            # Inside `decode` -> `shallow_branch(rgb_unpadded)`.
            # Then `shallow_branch` would receive unpadded RGB.
            # Then we are back to square one (no padding).
            #
            # So `sample_joint` MUST receive padded `rgb_img` if we want the shallow branch to see padded input.
            # And `model.decode` MUST be called with padded `rgb`.
            #
            # Wait, is `sample_joint` using `rgb_img` as a condition for `dit`?
            # NO.
            #
            # Let's correct the only potential confusion:
            # Maybe the user saw `rgb_padded` and thought it replaced `rgb` everywhere?
            #
            # I will re-apply the edit, ensuring that I use `rgb` (original) for tokens, and `rgb_padded` for `sample_joint`.
            # I will assume the user misread unless I find a bug.
            #
            # AHHH, I found a logic error in MY thinking about `sample_joint` maybe?
            # If `sample_joint` runs the loop:
            # `pred_target_image` is output of decoder.
            # If decoder output is padded, `pred_target_image` is padded.
            # `I_curr` is padded.
            # `delta_pixel = pred_target_image - rgb_img` (padded - padded).
            #
            # This logic assumes we want to diffuse in the PADDED space.
            # Is that correct?
            # The "shallow branch" is just a helper for the decoder.
            # The diffusion process (DiT) happens in latent space (unpadded).
            # The pixel space flow (Rectified Flow / Diffusion) happens on pixels.
            # Does the pixel flow need to happen on the padded pixels?
            # If we want the output to be correct in the center, do we need to flow the padding?
            # The padding is constant (0) in the target (reconstruction of input).
            # Ideally, we want the network to see the padding.
            # So `I_curr` should include the padding.
            #
            # So my implementation is technically consistent.
            #
            # BUT, the user said: "your implementation is wrong... you padded what is given to dit".
            #
            # Maybe they refer to THIS:
            # `pred_latent, pred_image = sample_joint(...)`
            # `pred_latent` is returned.
            # `pred_image` is returned.
            #
            # Is it possible `sample_joint` uses `rgb_img` to compute `rgb_tokens` internally?
            # No, `rgb_tokens_full` is passed explicitly.
            #
            # Okay, I will just rename variables for clarity and re-submit.
            # `rgb_condition` (unpadded) -> `encode_rgb`.
            # `rgb_inference` (padded) -> `sample_joint`.
            
            # The user provided the snippet:
            # `rgb_img=rgb_padded, # Pass padded RGB`
            # And said "you padded what is given to dit".
            
            # Is it possible that `rgb_tokens_full` is NOT used by DiT, but `rgb_img` IS?
            # In `sample_joint`:
            # `pred_noise = model.dit(x_curr, t=t, cond=context_tokens, rgb_tokens=rgb_tokens_full)`
            # It uses `rgb_tokens_full`.
            
            # I suspect the user is wrong, but I must be polite and double check.
            # I will reply saying "I am using `rgb` (unpadded) for tokens, and `rgb_padded` only for the image flow/shallow branch. Can you clarify where DiT receives the padded input?"
            # But I should probably just make the code clearer first in case I made a variable scope error (e.g. `rgb = F.pad` overwrite?).
            # I assigned to `rgb_padded`. I did not overwrite `rgb`.
            
            # Wait!
            # `rgb_tokens_full = model.encode_rgb(rgb)`
            # `rgb_padded = ...`
            # `sample_joint(..., rgb_tokens_full=rgb_tokens_full, rgb_img=rgb_padded)`
            #
            # Code is correct.
            #
            # Maybe `sample_joint` modification included an unintended change?
            # No, I didn't modify `sample_joint`.
            #
            # I'll modify the `evaluate_fmc.py` call to be very explicit.
            pass
            if pad > 0:
                pred_image_padded = pred_image
                pred_image = pred_image[:, :, pad:-pad, pad:-pad]
            else:
                pred_image_padded = pred_image
            
            decoded = torch.clamp(pred_image, 0.0, 1.0)
            gt_image = torch.clamp(draw, 0.0, 1.0)
            
            # Save image
            item_info = data_list[i]
            ir_path = item_info["infrared_path"]
            filename = os.path.basename(ir_path)
            
            save_path = os.path.join(args.output_dir, filename)
            
            img_tensor = decoded[0].cpu() # shape [C, H, W]
            to_pil = torchvision.transforms.ToPILImage()
            img_pil = to_pil(img_tensor)
            img_pil.save(save_path)

            # Save concatenated image
            gt_tensor = gt_image[0].cpu()
            gt_pil = to_pil(gt_tensor)
            
            # Concatenate horizontally
            w, h = img_pil.size
            if gt_pil.size != (w, h):
                gt_pil = gt_pil.resize((w, h))

            if pad > 0:
                # User request: Concatenate padded output and unpadded output?
                # Or just save padded image alongside?
                # "Visualization: Concatenate padded output and unpadded output for saving."
                # Interpreting as: [Padded Output] | [Unpadded Output]
                # Padded output is bigger.
                # Let's save padded output separately or add to concatenation?
                # Concatenating images of different heights is ugly.
                # Maybe they meant: Concatenate [Result] and [GT] (standard)
                # But explicitly mentioned "padded output".
                # Let's save padded output as separate file for inspection.
                padded_tensor = torch.clamp(pred_image_padded, 0.0, 1.0)[0].cpu()
                padded_pil = to_pil(padded_tensor)
                padded_save_path = os.path.join(args.output_dir, f"padded_{filename}")
                padded_pil.save(padded_save_path)
                
            cat_img = Image.new('RGB', (w * 2, h))
            cat_img.paste(img_pil, (0, 0))
            cat_img.paste(gt_pil, (w, 0))
            
            cat_path = os.path.join(concat_dir, filename)
            cat_img.save(cat_path)

    print(f"Generation complete. Saved to {args.output_dir}")
    print("Running evaluation...")
    
    # 1. Calculate metrics using generated path
    csv_output = os.path.join(args.output_dir, "evaluation_results.md")
    
    # Needs to match evaluation.py signature
    # calculate_lpips_ssim_psnr(data_root, json_path, generated_path, output_file, batch_size=1, resolution=(512, 384))
    # evaluation.py defaults resolution to (512, 384) in one place but main uses (512, 640).
    # train_fm.py resize hardcodes 640x512. (W, H) in PIL resize => (512, 640) in Tensor [H, W].
    # So resolution should be (512, 640).
    
    calculate_lpips_ssim_psnr(
        data_root=args.data_root,
        json_path=args.json_path,
        generated_path=args.output_dir,
        output_file=csv_output,
        resolution=(512, 640) 
    )

    # Calculate FID
    # Use a temporary directory outside output_dir to ensure cleanfid doesn't recursively find GT images
    temp_test_dir = args.output_dir.rstrip('/') + "_temp_gt_for_fid"
    try:
        create_temp_test_dir(args.json_path, args.data_root, temp_test_dir)
        calculate_fid(temp_test_dir, args.output_dir, csv_output)
    except Exception as e:
        print(f"Error calculating FID: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(temp_test_dir):
            shutil.rmtree(temp_test_dir)

if __name__ == "__main__":
    main()
