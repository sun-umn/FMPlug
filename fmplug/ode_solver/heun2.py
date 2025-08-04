# stdlib
from typing import Callable

# third party
import torch


def integrate_heun_v2(
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
    s_noise: float = 1.0,
) -> torch.Tensor:
    """
    Function that implements the Euler ode solver.
    """
    # Start ODE solver
    do_classifier_free_guidance = guidance_scale > 1.0

    prev_derivative = None
    dt = None
    state_in_first_order = dt is None

    # print("Heun 2")
    print(timesteps)
    for idx, t0 in enumerate(timesteps):
        # This is what is triggered in the code
        if state_in_first_order:
            sigma = sigmas[step_index]
            sigma_next = sigmas[step_index + 1]

        else:
            sigma = sigmas[step_index - 1]
            sigma_next = sigmas[step_index]

        # print("Sigma Info")
        # print(step_index)
        # print(sigmas)

        # Compute gamma
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigma <= s_tmax
            else 0.0
        )

        # Compute sigma hat
        sigma_hat = sigma * (gamma + 1)

        # Debugging purposes
        # print(t0, sigma, sigma_next)
        # Print the mean and variance to observe during solving the ode
        # x0 will be the latent variable
        latent_model_input = torch.cat([x0] * 2) if do_classifier_free_guidance else x0
        # print("latent")
        # print(latent_model_input.shape)
        # print(latent_model_input.min())
        # print(latent_model_input.max())

        # broadcast to batch dimension in a way that's compatible with ONNX / Core ML
        timestep = t0.expand(latent_model_input.shape[0])

        noise_pred = f(
            x=latent_model_input,
            t=timestep,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )
        # print("noise pred tensor")
        # print(noise_pred.min())
        # print(noise_pred.max())
        # print("\n")

        # If the classifier free guidance flag is True then we want
        # a linear combination of the unconditional noise prediction and
        # signal from the prompt prediction
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        # print("noise pred tensor - after")
        # print(noise_pred.min())
        # print(noise_pred.max())
        # print("\n")

        # Heun update
        if state_in_first_order:
            # print("Euler")
            # 1. compute predicted original sample (x_0)
            # from sigma-scaled predicted noise
            denoised = x0 - noise_pred * sigma
            # print("Denoised")
            # print(denoised.min())
            # print(denoised.max())

            # 2. convert to an ODE derivative for 1st order
            derivative = (x0 - denoised) / sigma_hat
            # print("Derivative")
            # print(derivative.min())
            # print(derivative.max())

            # 3. Delta timestep
            dt = sigma_next - sigma_hat
            # print("dt")
            # print(dt)
            # print(sigma)
            # print(sigma_next)
            # print(sigma_hat)

            # store for 2nd order step
            prev_derivative = derivative
            prev_x0 = x0

            prev_sample = x0 + derivative * dt
            x0 = prev_sample

            # print("prev sample")
            # print(x0.min())
            # print(x0.max())
            # print("\n")

            state_in_first_order = dt is None

        else:
            # print("Is this line executed")
            # 1. compute predicted original sample (x_0)
            # from sigma-scaled predicted noise
            denoised = x0 - noise_pred * sigma_next
            # print("Heun Denoised")
            # print(denoised.min())
            # print(denoised.max())

            # 2. 2nd order / Heun's method
            derivative = (x0 - denoised) / sigma_next
            derivative = 0.5 * (prev_derivative + derivative)

            # print("Heun derivative")
            # print(derivative.min())
            # print(derivative.max())

            # 3. take prev timestep & sample
            # dt = self.dt
            # sample = self.sample
            # print("prev x0")
            # print(prev_x0.min())
            # print(prev_x0.max())

            # print("dt")
            # print(dt)
            x0 = prev_x0 + derivative * dt

            # print("prev sample")
            # print(x0.min())
            # print(x0.max())
            # print("\n")

            # free dt and derivative
            # Note, this puts the scheduler in "first order mode"
            prev_derivative = None
            dt = None
            state_in_first_order = dt is None
            prev_x0 = None

        step_index += 1

    return x0


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
    s_noise: float = 1.0,
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
