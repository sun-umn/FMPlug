# stdlib
import logging
import pickle
from typing import Callable

# third party
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


# To normalize the latent variable upon entry
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


def integrate_heun(
    f: Callable,
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
    step_index: int,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    guidance_scale: float = 2.0,
    s_churn: float = 0.0,
    s_tmin: float = 0.0,
    s_tmax: float = float("inf"),
) -> torch.Tensor:
    """
    Function that implements the Heun ODE solver.
    """
    do_guidance = guidance_scale > 1.0
    prev_derivative = None
    dt = None
    state_in_first_order = dt is None

    logger.debug(f"Timesteps: {timesteps.cpu().numpy()}")

    for idx, t0 in enumerate(timesteps):
        if state_in_first_order:
            sigma = sigmas[step_index]
            sigma_next = sigmas[step_index + 1]
        else:
            sigma = sigmas[step_index - 1]
            sigma_next = sigmas[step_index]

        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigma <= s_tmax
            else 0.0
        )
        sigma_hat = sigma * (gamma + 1)

        logger.debug(
            f"Step {step_index}: sigma={sigma.item():.4f}, "
            f"sigma_next={sigma_next.item():.4f}, "
            f"sigma_hat={sigma_hat.item():.4f}"
        )

        latent_input = torch.cat([x0] * 2) if do_guidance else x0
        timestep = t0.expand(latent_input.shape[0])

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

        if state_in_first_order:
            denoised = x0 - noise_pred * sigma
            derivative = (x0 - denoised) / sigma_hat
            dt = sigma_next - sigma_hat
            prev_derivative = derivative
            prev_x0 = x0
            prev_sample = x0 + derivative * dt
            x0 = prev_sample

            logger.debug(
                "First order step: "
                f"x0.min={x0.min().item():.6f}, "
                f"x0.max={x0.max().item():.6f}"
            )

            state_in_first_order = dt is None
        else:
            denoised = x0 - noise_pred * sigma_next
            derivative = (x0 - denoised) / sigma_next
            derivative = 0.5 * (prev_derivative + derivative)
            x0 = prev_x0 + derivative * dt

            logger.debug(
                "Second order step: "
                f"x0.min={x0.min().item():.6f}, "
                f"x0.max={x0.max().item():.6f}"
            )

            prev_derivative = None
            dt = None
            state_in_first_order = dt is None
            prev_x0 = None

        step_index += 1

    return x0
