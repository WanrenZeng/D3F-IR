"""Validation and image-quality evaluation."""

from .common import *
from .sampling import sample_joint
from .visualization import visualize_flow_triplet


def validate(model: FlowMatchingModel, loader, device: torch.device, args: argparse.Namespace, epoch: int = 0, max_samples: Optional[int] = None) -> Dict[str, float]:
    model.eval()


    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_criterion = lpips.LPIPS(net="alex").to(device)

    total_loss = 0.0

    total_lpips = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    total_samples = 0
    batches = 0


    fid_real_dir = os.path.join(args.vis_dir, "temp_fid_real")
    fid_fake_dir = os.path.join(args.vis_dir, "temp_fid_fake")
    if os.path.exists(fid_real_dir): shutil.rmtree(fid_real_dir)
    if os.path.exists(fid_fake_dir): shutil.rmtree(fid_fake_dir)
    os.makedirs(fid_real_dir, exist_ok=True)
    os.makedirs(fid_fake_dir, exist_ok=True)

    with torch.no_grad():
        with tqdm(loader, desc="Validate", unit="batch") as progress:
            for batch in progress:

                if max_samples is not None and total_samples >= max_samples:
                    break

                visible = batch["condition_0"].to(device, non_blocking=True)
                infrared = batch["image"].to(device, non_blocking=True)
                context = visible

                b_size = visible.shape[0]

                rgb_tokens, rgb_tokens_latent = model.encode_rgb(visible)
                infrared_tokens_full, infrared_tokens_latent = model.encode_infrared(infrared)
                if rgb_tokens_latent is None:
                    raise RuntimeError("The visible-light latent adapter is required for flow matching.")
                pred_latent, pred_image = sample_joint(
                    model, rgb_tokens_latent, rgb_tokens, visible,
                    args.val_flow_steps,
                )


                for idx in range(b_size):

                    r_img = torch.clamp(infrared[idx], 0, 1).cpu()
                    f_img = torch.clamp(pred_image[idx], 0, 1).cpu()

                    real_pil = transforms.ToPILImage()(r_img)
                    fake_pil = transforms.ToPILImage()(f_img)

                    fname = f"b{batches}_i{idx}.png"
                    real_pil.save(os.path.join(fid_real_dir, fname))
                    fake_pil.save(os.path.join(fid_fake_dir, fname))


                if batches == 0 and args.vis_interval > 0:
                    val_vis_dir = os.path.join(args.vis_dir, "val")
                    os.makedirs(val_vis_dir, exist_ok=True)
                    vis_path = os.path.join(val_vis_dir, f"epoch_{epoch+1:03d}.png")


                    student_features = {
                        "visible": rgb_tokens[:1].detach().cpu().float(),
                        "pred": pred_latent[:1].detach().cpu().float(),
                        "infrared": infrared_tokens_full[:1].detach().cpu().float(),
                    }

                    visualize_flow_triplet(
                        visible[:1].cpu(),
                        context[:1].cpu(),
                        pred_image[:1].cpu(),
                        infrared[:1].cpu(),
                        vis_path,
                        gamma=args.gamma,
                        img_size=(args.input_size[1], args.input_size[0]),
                        patch_size=model.encoder.patch_size,
                        student_features=student_features,
                        teacher_tokens=None,
                        ae_tokens=None,
                    )

                latent_loss = F.mse_loss(pred_latent, infrared_tokens_latent)
                decoded = torch.clamp(pred_image, 0.0, 1.0)


                lpips_batch = lpips_criterion(decoded * 2 - 1, infrared * 2 - 1)
                total_lpips += lpips_batch.sum().item()


                psnr_batch = calculate_psnr_manual(decoded, infrared)
                total_psnr += psnr_batch.sum().item()


                ssim_val_mean = ssim_metric(decoded, infrared)
                total_ssim += ssim_val_mean.item() * b_size

                total_loss += latent_loss.item()
                batches += 1
                total_samples += b_size

                current_lpips = lpips_batch.mean().item()
                progress.set_postfix(latent=f"{latent_loss.item():.4f}", lpips=f"{current_lpips:.4f}")


    fid_score = 999.0
    if calculate_fid is not None and total_samples > 1:
        try:
            print("Calculating FID...")


            dummy_fid_log = os.path.join(args.vis_dir, f"fid_epoch_{epoch}.md")
            fid_score = calculate_fid(fid_real_dir, fid_fake_dir, dummy_fid_log)
            print(f"FID: {fid_score}")
        except Exception as e:
            print(f"FID calc failed: {e}")
            fid_score = 999.0
    elif calculate_fid is not None:
        print("Skipping FID: at least two validation samples are required.")


    if os.path.exists(fid_real_dir): shutil.rmtree(fid_real_dir)
    if os.path.exists(fid_fake_dir): shutil.rmtree(fid_fake_dir)

    avg_psnr = total_psnr / max(1, total_samples)
    avg_ssim = total_ssim / max(1, total_samples)
    avg_lpips = total_lpips / max(1, total_samples)

    return {"latent_loss": total_loss / max(1, batches), "psnr": avg_psnr, "ssim": avg_ssim, "lpips": avg_lpips, "fid": fid_score}
