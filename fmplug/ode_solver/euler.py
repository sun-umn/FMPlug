import logging
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
