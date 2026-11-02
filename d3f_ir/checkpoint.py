"""Pretrained model checkpoint loading."""

from .common import *


def load_pretrained_weights(model, ckpt_path, strict=False):
    print(f"Loading pre-trained weights from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if "model" in checkpoint:
        state_dict_raw = checkpoint["model"]
    else:
        state_dict_raw = checkpoint

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


                     if hasattr(model, 'encoder') and hasattr(model.encoder, 'input_size'):
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

                     pos_tokens = F.interpolate(pos_tokens, size=(h_new, w_new), mode='bicubic', align_corners=False)
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
