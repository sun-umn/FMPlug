# stdlib
from typing import Callable

# third party
import torch


def spherical_projection(
    x: torch.Tensor, center: torch.Tensor, radius: float
) -> torch.Tensor:
    direction = x - center
    direction_norm = torch.norm(
        direction.view(direction.size(0), -1), dim=1, keepdim=True
    )
    direction_unit = direction / (
        direction_norm.view(-1, *([1] * (x.dim() - 1))) + 1e-8
    )
    return center + radius * direction_unit


def integrate_euler_v2(
    f: Callable,
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    guidance_scale: float = 2.0,
) -> torch.Tensor:
    """
    Function that implements the Euler ode solver.
    """
    # Start ODE solver
    do_classifier_free_guidance = guidance_scale > 1.0

    for idx, t0 in enumerate(timesteps):
        sigma = sigmas[idx]
        sigma_next = sigmas[idx + 1]

        # Debugging purposes
        # print(t0, sigma, sigma_next)
        # Print the mean and variance to observe during solving the ode
        # x0 will be the latent variable
        latent_model_input = torch.cat([x0] * 2) if do_classifier_free_guidance else x0

        # broadcast to batch dimension in a way that's compatible with ONNX / Core ML
        timestep = t0.expand(latent_model_input.shape[0])

        # Euler delta
        dt = sigma_next - sigma

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

        # Update step for Euler
        prev_sample = x0 + dt * noise_pred
        x0 = prev_sample
        # print("prev sample tensor")
        # print(x0.min())
        # print(x0.max())

        # # Optional: estimate the denoised mean (if you have access)
        # # For now, assume mu_theta ≈ x0
        # # (could also use x0 - sigma * noise_pred if needed)
        # mu_theta = x0

        # # Compute spherical projection radius: sqrt(n) * sigma
        # latent_dim = x0[0].numel()  # total elements per sample (e.g., C*H*W)
        # radius = (latent_dim**0.5) * sigma.item()

        # # Project back to Gaussian shell
        # x0 = spherical_projection(prev_sample, center=mu_theta, radius=radius)

    return x0
