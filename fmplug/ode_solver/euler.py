# stdlib
from typing import Callable

# third party
import torch


@torch.compile
def integrate_euler(
    f: Callable,
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
    prompt_embedding: torch.Tensor,
    pooled_embedding: torch.Tensor,
    device: torch.device,
    guidance_scale: float = 2.0,
) -> torch.Tensor:
    """
    Function that implements the Euler ode solver.
    """
    # Start ODE solver
    current_timesteps = timesteps[:-1]
    previous_timesteps = timesteps[1:]
    current_sigmas = sigmas[:-1]
    previous_sigmas = sigmas[1:]

    integrate_parameters = zip(
        current_timesteps,
        previous_timesteps,
        current_sigmas,
        previous_sigmas,
    )

    do_classifier_free_guidance = guidance_scale > 1.0

    for i, (t0, t1, sigma, sigma_next) in enumerate(integrate_parameters):
        # Print the mean and variance to observe during solving the ode
        # print(x0.mean())
        # print(x0.var())

        # print(x0.norm(), x0.mean(), x0.var())
        # x0 will be the latent variable
        latent_model_input = torch.cat([x0] * 2) if do_classifier_free_guidance else x0

        # broadcast to batch dimension in a way that's compatible with ONNX / Core ML
        timestep = t0.expand(latent_model_input.shape[0])

        # upcast to avoid precision issues
        sample = x0
        dt = sigma_next - sigma

        # Euler
        noise_pred = f(
            x=latent_model_input,
            t=timestep,
            prompt_embedding=prompt_embedding,
            pooled_embedding=pooled_embedding,
            device=device,
        )

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # if do_classifier_free_guidance:
        #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        #     noise_pred = noise_pred_uncond + guidance_scale * (
        #         noise_pred_text - noise_pred_uncond
        #     )

        # else:
        #     noise_pred, noise_pred_text = noise_pred.chunk(2)

        # Update step for euler
        prev_sample = sample + dt * noise_pred

        # prev_sample = prev_sample

        x0 = prev_sample

    return x0
