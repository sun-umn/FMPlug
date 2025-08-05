import logging
import pickle
from typing import Callable

import torch

# ----------------------------
# Setup logger
# ----------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# Open the polynomial model variance data for
# normalization
with open("poly13_model_var.pkl", "rb") as f:
    reg = pickle.load(f)


def normalize_latent(z_x: torch.Tensor, t_x: torch.Tensor):
    """
    Normalize z_x at time t_x using interpolated mean and variance per channel.
    z_x: (B, C, H, W)
    t_x: scalar (float or 0-dim tensor)
    """
    z_var = reg(t_x.detach().cpu().numpy())
    z_var = torch.tensor(z_var, dtype=z_x.dtype, device=z_x.device)
    z_x = torch.sqrt(z_var / torch.var(z_x, unbiased=False)) * z_x
    return z_x


def integrate_euler(
    f: Callable,
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    step_index: int,
    guidance_scale: float = 2.0,
) -> torch.Tensor:
    """
    Function that implements the Euler ODE solver.
    """
    do_guidance = guidance_scale > 1.0

    for idx, t0 in enumerate(timesteps):
        sigma = sigmas[idx]
        sigma_next = sigmas[idx + 1]

        logger.debug(
            f"Step {idx}: t0={t0.item():.4f}, "
            f"sigma={sigma.item():.4f}, "
            f"sigma_next={sigma_next.item():.4f}"
        )

        latent_input = torch.cat([x0] * 2) if do_guidance else x0
        timestep = t0.expand(latent_input.shape[0])
        dt = sigma_next - sigma

        logger.debug(f"dt: dt = {dt:.6f}")

        noise_pred = f(
            x=latent_input,
            t=timestep,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )

        logger.debug(
            "Noise BEFORE guidance: "
            f"min={noise_pred.min().item():.6f}, "
            f"max={noise_pred.max().item():.6f}"
        )

        if do_guidance:
            uncond, text = noise_pred.chunk(2)
            noise_pred = uncond + guidance_scale * (text - uncond)

        logger.debug(
            "Noise AFTER guidance: "
            f"min={noise_pred.min().item():.6f}, "
            f"max={noise_pred.max().item():.6f}"
        )

        prev_sample = x0 + dt * noise_pred
        x0 = prev_sample

        logger.debug(
            f"Updated sample: min={x0.min().item():.6f}, max={x0.max().item():.6f}"
        )

        step_index += 1

    return x0
