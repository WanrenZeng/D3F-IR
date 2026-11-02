"""Training loop and per-epoch optimization."""

from .common import *
from .sampling import sample_joint
from .visualization import visualize_flow_triplet


def train_one_epoch(
    model: FlowMatchingModel,
    optimizer: torch.optim.Optimizer,
    loader,
    device: torch.device,
    scaler: Optional[GradScaler],
    autocast_kwargs: Dict[str, object],
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    teacher_encoder: DinoEncoder,
) -> Tuple[Dict[str, float], int]:
    model.train()
    stats = {"flow": 0.0, "recon": 0.0, "align": 0.0, "total": 0.0}
    steps = 0
    accumulation_steps = 8

    ssim_criterion = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_criterion = lpips.LPIPS(net="alex").to(device)
    l2_criterion = torch.nn.MSELoss().to(device)

    desc = f"Train Epoch {epoch+1:03d}"

    optimizer.zero_grad(set_to_none=True)

    with tqdm(loader, desc=desc, unit="batch") as progress:
        model.train()
        for batch in progress:
            visible = batch["condition_0"].to(device, non_blocking=True)
            infrared = batch["image"].to(device, non_blocking=True)
            context = visible

            with torch.no_grad():
                teacher_visible_tokens, _ = teacher_encoder(visible)
                teacher_infrared_tokens, _ = teacher_encoder(infrared)

            with autocast(**autocast_kwargs):
                rgb_tokens, rgb_tokens_latent = model.encode_rgb(visible)
                infrared_tokens_full, infrared_tokens_latent = model.encode_infrared(infrared)
                infrared_tokens_latent = infrared_tokens_latent.detach()
                if rgb_tokens_latent is None:
                    raise RuntimeError("The visible-light latent adapter is required for flow matching.")
                source_latent = rgb_tokens_latent.detach()
                t = torch.rand(visible.size(0), device=device, dtype=visible.dtype)
                t_latent = t.view(-1, 1, 1)
                mixed_latent = (1 - t_latent) * source_latent + t_latent * infrared_tokens_latent
                model_out = model.dit(
                    mixed_latent, t=t, cond=rgb_tokens, rgb_tokens=rgb_tokens
                )
                target_velocity = infrared_tokens_latent - source_latent
                flow_loss = l2_criterion(model_out, target_velocity)
                align_loss = flow_loss.new_zeros(())
                if args.align_weight > 0:
                    align_base_grad = rgb_tokens_latent
                    t_align = torch.rand(align_base_grad.size(0), device=device, dtype=align_base_grad.dtype)
                    mix_align = align_base_grad + t_align.view(-1, 1, 1) * (infrared_tokens_latent - align_base_grad)
                    align_outputs = model.dit(
                        mix_align,
                        t=t_align,
                        cond=rgb_tokens,
                        rgb_tokens=rgb_tokens,
                        return_features=True,
                        feature_layer=args.align_layer,
                        align_only=True,
                    )
                    student_feats = F.normalize(align_outputs["mid_features"], dim=-1)
                    teacher_tokens_norm = teacher_infrared_tokens.detach()
                    teacher_mean = teacher_tokens_norm.mean(dim=1, keepdim=True)
                    teacher_tokens_norm = teacher_tokens_norm - 0.7 * teacher_mean
                    teacher_tokens_norm = teacher_tokens_norm / (teacher_tokens_norm.std(dim=1, keepdim=True) + 1e-6)
                    teacher_feats = F.normalize(teacher_tokens_norm, dim=-1)
                    align_loss = -torch.mean(torch.sum(student_feats * teacher_feats, dim=-1))

                predicted_pixel_velocity, pred_ae_tokens = model.decode(
                    rgb=visible,
                    rgb_latent=rgb_tokens_latent,
                    velocity=model_out,
                    rgb_tokens=rgb_tokens,
                )
                decoded_pred_img = visible + predicted_pixel_velocity

                decoded_pred = torch.clamp(decoded_pred_img, 0.0, 1.0)
                recon_pred_loss = l2_criterion(decoded_pred, infrared) + 0.2 * (1 - ssim_criterion(decoded_pred, infrared)) + 0.1 * lpips_criterion(decoded_pred, infrared).mean()

                target_velocity = infrared_tokens_latent - source_latent
                target_pixel_velocity, _ = model.decode(
                    rgb=visible,
                    rgb_latent=rgb_tokens_latent,
                    velocity=target_velocity,
                    rgb_tokens=rgb_tokens,
                )
                decoded_gt_img = visible + target_pixel_velocity

                decoded_gt = torch.clamp(decoded_gt_img, 0.0, 1.0)
                recon_gt_loss = l2_criterion(decoded_gt, infrared) + 0.2 * (1 - ssim_criterion(decoded_gt, infrared)) + 0.1 * lpips_criterion(decoded_gt, infrared).mean()

                recon_loss = recon_pred_loss + recon_gt_loss

                total_loss = args.flow_weight * flow_loss + args.recon_weight * recon_loss + args.align_weight * align_loss

                loss_accum = total_loss / accumulation_steps

            if scaler is not None:
                scaler.scale(loss_accum).backward()
                if (steps + 1) % accumulation_steps == 0:
                    if args.clip_grad > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss_accum.backward()
                if (steps + 1) % accumulation_steps == 0:
                    if args.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            stats["flow"] += flow_loss.detach().item()
            stats["recon"] += recon_loss.detach().item()
            stats["align"] += align_loss.detach().item()
            stats["total"] += total_loss.detach().item()
            steps += 1
            progress.set_postfix(flow=f"{stats['flow']/max(1,steps):.4f}", recon=f"{stats['recon']/max(1,steps):.4f}", align=f"{stats['align']/max(1,steps):.4f}", total=f"{stats['total']/max(1,steps):.4f}")

            global_step += 1

            if args.vis_interval > 0 and global_step % args.vis_interval == 0:
                model.eval()
                with torch.no_grad():
                    train_vis_dir = os.path.join(args.vis_dir, "train")
                    os.makedirs(train_vis_dir, exist_ok=True)
                    vis_path = os.path.join(
                        train_vis_dir, f"epoch_{epoch+1:03d}_step_{global_step:06d}.png"
                    )
                    with autocast(**autocast_kwargs):
                        vis_pred_latent, vis_pred_image = sample_joint(
                            model,
                            rgb_tokens_latent.detach(),
                            rgb_tokens,
                            visible,
                            max(1, args.val_flow_steps),
                        )
                    student_features = {
                        "visible": rgb_tokens[:1].detach().cpu().float(),
                        "pred": vis_pred_latent[:1].detach().cpu().float(),
                        "infrared": infrared_tokens_full[:1].detach().cpu().float(),
                    }
                    teacher_tokens = {
                        "visible": teacher_visible_tokens[:1].detach().cpu().float(),
                        "infrared": teacher_infrared_tokens[:1].detach().cpu().float(),
                    }
                    visualize_flow_triplet(
                        visible[:1].cpu(),
                        context[:1].cpu(),
                        vis_pred_image[:1].cpu(),
                        infrared[:1].cpu(),
                        vis_path,
                        gamma=args.gamma,
                        img_size=(args.input_size[1], args.input_size[0]),
                        patch_size=model.encoder.patch_size,
                        student_features=student_features,
                        teacher_tokens=teacher_tokens,
                        ae_tokens={k: v[:1].detach().cpu() for k, v in pred_ae_tokens.items()},
                    )
                model.train()

    return {k: v / max(1, steps) for k, v in stats.items()}, global_step
