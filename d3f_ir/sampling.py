"""Flow-matching sampling routines."""

from .common import *


def sample_joint(model: FlowMatchingModel, base_tokens: torch.Tensor, context_tokens: torch.Tensor, rgb_tokens_full: torch.Tensor, rgb_img: torch.Tensor, steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
    b = base_tokens.size(0)
    device = base_tokens.device

    x_curr = base_tokens.clone()
    I_curr = rgb_img.clone()

    dt = 1.0 / steps
    for i in range(steps):
        t_val = i / steps
        t = torch.full((b,), t_val, device=device, dtype=x_curr.dtype)
        v_latent = model.dit(x_curr, t=t, cond=context_tokens, rgb_tokens=rgb_tokens_full)
        x_curr = x_curr + v_latent * dt
        v_pixel, _ = model.decode(
            rgb_tokens_full,
            v_latent,
            rgb=rgb_img,
            context_tokens=context_tokens,
            rgb_latent=base_tokens,
        )
        I_curr = I_curr + v_pixel * dt

    return x_curr, I_curr
