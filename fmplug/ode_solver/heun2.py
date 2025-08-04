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
