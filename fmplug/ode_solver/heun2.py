# stdlib
from typing import Callable

# third party
import torch
import tqdm


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
    Function that implements the Heun 2 ode solver.
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

    for _, (t0, t1, sigma, sigma_next) in tqdm.tqdm(enumerate(integrate_parameters)):
        # print(x0.norm(), x0.mean(), x0.var())
        # x0 will be the latent variable
        latent_model_input = torch.cat([x0] * 2) if do_classifier_free_guidance else x0

        # broadcast to batch dimension in a way that's compatible with ONNX / Core ML
        timestep = t0.expand(latent_model_input.shape[0])
        prev_timestep = t1.expand(latent_model_input.shape[0])

        # upcast to avoid precision issues
        sample = x0.to(torch.float32)
        dt = sigma_next - sigma

        # Heun2
        k1 = f(
            x=latent_model_input,
            t=timestep,
            prompt_embedding=prompt_embedding,
            pooled_embedding=pooled_embedding,
            device=device,
        )

        # Predict next latent using Euler step
        x1_pred = latent_model_input + dt * k1

        # k2
        k2 = f(
            x=x1_pred,
            t=prev_timestep,
            prompt_embedding=prompt_embedding,
            pooled_embedding=pooled_embedding,
            device=device,
        )

        # Heun2 step (average slope)
        noise_pred = 0.5 * dt * (k1 + k2)

        # TODO: Keep here for now this is the RK4 implementation
        # Rk4
        # half_dt = 0.5 * dt
        # k1 = f(
        #     x=latent_model_input,
        #     t=timestep,
        #     prompt_embedding=prompt_embedding,
        #     pooled_embedding=pooled_embedding,
        #     device=device,
        # )

        # k2 = f(
        #     x=(latent_model_input + half_dt * k1),
        #     t=(timestep + half_dt),
        #     prompt_embedding=prompt_embedding,
        #     pooled_embedding=pooled_embedding,
        #     device=device,
        # )

        # k3 = f(
        #     x=(latent_model_input + half_dt * k2),
        #     t=(timestep + half_dt),
        #     prompt_embedding=prompt_embedding,
        #     pooled_embedding=pooled_embedding,
        #     device=device,
        # )

        # k4 = f(
        #     x=(latent_model_input + dt * k3),
        #     t=prev_timestep,
        #     prompt_embedding=prompt_embedding,
        #     pooled_embedding=pooled_embedding,
        #     device=device,
        # )

        # noise_pred = (k1 + 2 * (k2 + k3) + k4) * dt * (1 / 6)
        # noise_pred = noise_pred.to(noise_pred.dtype)

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        else:
            noise_pred, noise_pred_text = noise_pred.chunk(2)

        # Update step for huen2
        prev_sample = sample + noise_pred

        prev_sample = prev_sample.to(torch.float32)

        x0 = prev_sample

    return x0
