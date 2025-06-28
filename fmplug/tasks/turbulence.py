# stdlib
import os
import random
import time

# third party
import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import wandb
from diffusers import AutoencoderTiny, StableDiffusion3Img2ImgPipeline
from skimage.metrics import peak_signal_noise_ratio

# first party
from fmplug.ode_solver.euler import integrate_euler
from fmplug.tasks.utils import compute_ssim, prepare_measurement
from fmplug.utils.measurements import get_noise, get_operator

# These presets are used for torch compile for SD3
torch.set_float32_matmul_precision("high")

torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True


def set_seed(seed):
    """
    Function to set the seed for the run
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def total_variation_loss(x):
    return torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + torch.mean(
        torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
    )


def turbulence_task(config_name: str) -> None:
    # Configuration
    image_size = 512
    scale_factor = 4
    num_inference_steps = 3
    batch_size = 1
    guidance_scale = 2.0
    optimizer_name = "Adam"
    lr = 1e-2
    weight_decay = 0.0
    epochs = 3500
    dtype = torch.float32

    # NOTE: Seed was 123
    set_seed(123)

    # Log into huggingface to be able to pull the SD3.0
    # print("Log into HuggingFace ...")
    # login(os.environ.get("HF_ACCESS_TOKEN"))

    # Global variables for wandb
    API_KEY = os.environ.get("WANDB_API_KEY")
    PROJECT_NAME = "FMPlug"

    # Enable wandb
    print("Initialize Project ...")
    wandb.login(key=API_KEY)  # type: ignore

    wandb_instance = wandb.init(  # type: ignore
        # set the wandb project where this run will be logged
        project=PROJECT_NAME,
        tags=["Experimental", "super resolution"],
        config={
            "optimizer_name": optimizer_name,
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "image_size": image_size,
            "scale_factor": scale_factor,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        },
    )

    # Create the directory to save all of the model results
    wandb_experiment_id = wandb_instance.id
    save_file_path = f"/users/5/dever120/FMPlug/experiments/{wandb_experiment_id}"
    os.makedirs(save_file_path, exist_ok=True)

    # Setup device as cuda
    device = torch.device("cuda")

    # Initialize LPIPS model (use net='vgg' for VGG-based)
    lpips_loss_fn = lpips.LPIPS(net="vgg").to(device)

    # Super turbulence config
    config = {
        "measurement": {
            "operator": {"name": "turbulence"},
            "noise": {"name": "gaussian", "sigma": 0.03},
        },
        "kernel": "gaussian",
        "kernel_size": 64,
        "intensity": 3.0,
    }

    # Load in an image & measurement
    img_outputs = prepare_measurement(
        image_path="/users/5/dever120/FMPlug/data/div2k_example.png",  # Hardcode for now  # noqa
        image_size=image_size,
        config=config,
        get_operator_fn=get_operator,
        get_noise_fn=get_noise,
        device=device,
    )

    # Let's save the turbulence images
    # measurement_image = img_outputs["y"]

    # # Image is in [-1, 1] need to convert to [0, 1]
    # measurement_image = (measurement_image + 1.0) / 2.0

    print("Load in SD3 image to image model ...")
    # Enable a tiny autoencoder
    vae = AutoencoderTiny.from_pretrained("madebyollin/taesd3", torch_dtype=dtype)

    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        text_encoder_3=None,
        tokenizer_3=None,
        torch_dtype=dtype,
    )

    # Set new vae
    pipe.vae = vae
    pipe.vae.config.shift_factor = 0.0

    pipe = pipe.to(device)

    # Add these lines to enable torch compile which is expected to
    # increase the speed
    pipe.set_progress_bar_config(disable=True)

    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)

    # pipe.transformer = torch.compile(
    #     pipe.transformer, mode="max-autotune", fullgraph=True
    # )
    # pipe.vae.decode = torch.compile(
    #     pipe.vae.decode, mode="max-autotune", fullgraph=True
    # )

    pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=True)
    pipe.vae.decode = torch.compile(pipe.vae.decode, mode="default", fullgraph=True)

    # Extract different components of the pipeline
    prompt_encoder = pipe.encode_prompt
    image_processor = pipe.image_processor
    prepare_latents = pipe.prepare_latents

    # Transformer block
    transformer = pipe.transformer  # type: ignore
    transformer.eval()  # type: ignore
    transformer.requires_grad_(False)  # type: ignore
    # transformer.enable_gradient_checkpointing()

    # AE / VAE
    vae = pipe.vae
    vae.eval()
    vae.requires_grad_(False)
    # vae.enable_gradient_checkpointing()

    # Define the prompts and embeddings
    prompt = "a high quality photo of animal, bush, close-up, fox, grass, green, greenery, hide, panda, red, red panda, stare"  # noqa
    prompt_2 = None
    prompt_3 = None

    negative_prompt = ""
    negative_prompt_2 = None
    negative_prompt_3 = None

    do_classifier_free_guidance = True
    prompt_embeds = None
    negative_prompt_embeds = None
    pooled_prompt_embeds = None
    negative_pooled_prompt_embeds = None
    clip_skip = None
    num_images_per_prompt = 1
    max_sequence_length = 128
    lora_scale = None

    # encode prompt
    print("Encode prompt ...")
    with torch.no_grad():
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = prompt_encoder(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

    # prompt embeds with classifier free guidance
    prompt_embedding = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pooled_embedding = torch.cat(
        [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
    )

    # Define the latent time steps here
    timesteps = pipe.scheduler.timesteps
    sigmas = pipe.scheduler.sigmas

    # After digging in I understand more how this works so we an set
    # a paramter to say lets start in the range of timesteps 600.0
    mask = timesteps <= 1000.0
    timesteps = timesteps[mask]
    sigmas = sigmas[mask]

    # Now get num_inference spaced timesteps and sigmas
    num_inference_mask = torch.linspace(
        0, len(timesteps) - 1, num_inference_steps + 1
    ).long()
    timesteps = timesteps[num_inference_mask].to(device=device, dtype=dtype)
    sigmas = sigmas[num_inference_mask].to(device, dtype=dtype)

    print("Process image, build timesteps & latent variables ...")
    # When we are feeding in the image it needs to be in the range [0, 1]
    # The output of the image preprocessor will be in the range [-1, 1]

    # Now the real test will be to produce the image using the degraded image
    # Degraded img and label
    y_n = img_outputs["y_n"]

    # # Start from the measurement
    # upsampled_img = torch.nn.functional.interpolate(
    #     (y_n + 1.0) / 2.0,
    #     size=(image_size, image_size),
    #     mode="bilinear",
    #     align_corners=False,
    # )

    # Start from a random image
    # Image is expecting an input between [0, 1]
    random_image = torch.rand(
        (1, 3, image_size, image_size), dtype=dtype, device=device
    )
    image = image_processor.preprocess(
        random_image, height=image_size, width=image_size
    )

    # Here is where the magic starts to happen - in the Img2Img
    # we can take an image to an encoded latent using prepare_latents. This function
    # has a couple of major steps:
    # 1. Encode
    # 2. Shift and scale encoded latent with vae.config values
    # 3. Add Guassian noise with the first sigma strength
    # - when decoding images this may be
    # a key step.

    num_images_per_prompt = 1
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

    # We need to define z first because this is what we are training
    z = prepare_latents(
        image,  # Range [-1, 1],
        latent_timestep,
        batch_size,
        num_images_per_prompt,
        prompt_embeds.dtype,
        device,
        generator=None,
    )

    # Setup model parameter
    z = torch.nn.parameter.Parameter(z)
    z = z.to(device=device, dtype=dtype)

    # Setup optimizer
    optimizer: torch.optim.Adam | torch.optim.AdamW | None = None
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam([z], lr=lr)

    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW([z], lr=lr, weight_decay=weight_decay)

    # Export the measurment operator
    operator = img_outputs["operator"]

    # Create the loss function
    criterion = torch.nn.MSELoss()

    # Try the grad scaler for fp16
    scaler = torch.amp.GradScaler()

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=lr,  # Peak learning rate
    #     steps_per_epoch=1,
    #     epochs=epochs,  # Total number of epochs
    #     pct_start=0.05,  # % of total steps for warmup (default = 30%)
    #     anneal_strategy="linear",  # Use cosine annealing
    #     cycle_momentum=False,  # If using AdamW or Adam
    #     final_div_factor=1e4,  # make this the same as initial
    #     div_factor=25.0,
    # )

    # Function for integration
    # @torch.compile
    def f(x, t, prompt_embedding, pooled_embedding, device):
        with torch.amp.autocast(device.type, dtype=torch.float16):
            result = transformer(
                hidden_states=x,
                timestep=t,
                encoder_hidden_states=prompt_embedding,
                pooled_projections=pooled_embedding,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

        return result

    # Should we try a compile warmup here -
    # z will not be updated as long as we are not updating
    print("Staring compile warmup ...")
    noise = torch.randn(z.shape, generator=None, dtype=dtype, layout=None).to(device)

    # with torch.no_grad():
    #     for _ in range(3):
    #         x_t = integrate_euler(
    #             f=f,
    #             x0=z,
    #             timesteps=timesteps,
    #             sigmas=sigmas,
    #             prompt_embedding=prompt_embedding,
    #             pooled_embedding=pooled_embedding,
    #             device=device,
    #             guidance_scale=guidance_scale,
    #         )

    #         # Implment steps to rescale x_t
    #         last_sigma = sigmas[-1]

    #         # Step 1: Add noise inverse
    #         decoded_latent = (x_t - last_sigma * noise) / (1 - last_sigma)

    #         # Step 2: Add shift and scale inverse
    #         decoded_latent = (
    #             decoded_latent / vae.config.scaling_factor
    #         ) + vae.config.shift_factor

    #         # Step 3: Decode using VAE / AE - this output is [-1, 1]
    #         decoded_output = torch.clamp(vae.decode(decoded_latent).sample, -1.0, 1.0)

    # print("Ending compile warmup ...")
    noise = torch.randn(z.shape, generator=None, dtype=dtype, layout=None).to(device)
    lpips_scores = []
    early_stopping_criterion = 0

    start = time.time()
    for idx, epoch in tqdm.tqdm(enumerate(range(epochs))):
        torch.compiler.cudagraph_mark_step_begin()

        # First zero gradients
        optimizer.zero_grad()  # type: ignore

        # If this is the first pass we have our latent variable established
        # pass it through the ODE solver
        x_t = integrate_euler(
            f=f,
            x0=z,
            timesteps=timesteps,
            sigmas=sigmas,
            prompt_embedding=prompt_embedding,
            pooled_embedding=pooled_embedding,
            device=device,
            guidance_scale=guidance_scale,
        )

        # Implment steps to rescale x_t
        last_sigma = sigmas[-1]

        # Step 1: Add noise inverse
        decoded_latent = (x_t - last_sigma * noise) / (1 - last_sigma)
        # decoded_latent = x_t

        # Step 2: Add shift and scale inverse
        decoded_latent = (
            decoded_latent / vae.config.scaling_factor
        ) + vae.config.shift_factor

        # Step 3: Decode using VAE / AE - this output is [-1, 1]
        decoded_output = torch.clamp(vae.decode(decoded_latent).sample, -1.0, 1.0)

        # Now apply the degradation
        operator_decoded_output = operator.forward(decoded_output)  # type: ignore

        # Apply the loss function - this expects [-1, 1]
        mse_loss = criterion(operator_decoded_output, y_n)
        lpips_loss = lpips_loss_fn(operator_decoded_output, y_n)
        tv_loss = total_variation_loss(decoded_output)

        # What if we also wanted to measure mse in the latent space?
        # We would want to prepare the latents based on the last time
        # step
        # Why are we using the last timestep to encode? This is because
        # We will add minimum noise to encode the measurment
        latent_y_n = vae.encode(y_n).latents

        # Encode the degraded output
        latent_operator_decoded_output = vae.encode(operator_decoded_output).latents
        latent_mse = criterion(latent_operator_decoded_output, latent_y_n)

        loss = mse_loss + latent_mse + lpips_loss + 0.1 * tv_loss

        # Update gradients of z
        scaler.scale(loss).backward()

        # Unscale gradients before clipping
        # scaler.unscale_(optimizer)

        # Clip gradients (example: max norm = 1.0)
        # torch.nn.utils.clip_grad_norm_([z], max_norm=0.05)

        scaler.step(optimizer)  # type: ignore
        scaler.update()

        # scheduler.step()

        # What is the gradient norm?
        grad_norm = z.grad.norm()

        # Compute the PSNR & print loss
        with torch.no_grad():
            lpips_score = lpips_loss_fn(decoded_output, img_outputs.get("ref_img"))

            model_img = (decoded_output + 1.0) / 2.0  # type: ignore
            model_img = model_img.squeeze(0).detach().cpu().numpy()  # type: ignore

            img = img_outputs.get("ref_img")
            img = (img + 1.0) / 2.0  # type: ignore
            img = img.squeeze(0).detach().cpu().numpy()  # type: ignore

            ssim_score = compute_ssim(
                img,
                model_img,
            )

            img = img.transpose(1, 2, 0)  # type: ignore
            model_img = model_img.transpose(1, 2, 0)  # type: ignore

            psnr_score = peak_signal_noise_ratio(img, model_img)
            mse_score = ((img - model_img) ** 2).mean()

            lpips_scores.append(lpips_score.item())

            metrics_to_log = {
                "epoch": epoch,
                # "current_lr": scheduler.get_last_lr()[0],
                "mse_loss": mse_score,
                "psnr": psnr_score,
                "ssim": ssim_score,
                "lpips": lpips_score.item(),
                "grad_norm": grad_norm,
            }
            wandb.log(metrics_to_log)  # type: ignore

            current_lpips = lpips_scores[-1]

            if len(lpips_scores) > 1:
                best_lpips = np.min(lpips_scores[:-1])
            else:
                best_lpips = current_lpips

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

            ax1.imshow(img)
            ax1.set_title("Original Image")

            ax2.imshow(model_img)
            ax2.set_title("Reconstructed Image")

            image_save_path = os.path.join(
                save_file_path, f"reference_vs_generated_image_epoch_{idx + 1}.png"
            )
            fig.savefig(image_save_path, bbox_inches="tight")
            plt.close(fig)

            if current_lpips <= best_lpips:
                # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

                # ax1.imshow(img)
                # ax1.set_title("Original Image")

                # ax2.imshow(model_img)
                # ax2.set_title("Reconstructed Image")

                # image_save_path = os.path.join(
                #     save_file_path, f"reference_vs_generated_image_epoch_{idx + 1}.png"  # noqa
                # )
                # fig.savefig(image_save_path, bbox_inches="tight")

                # Reset early stopping criterion
                early_stopping_criterion = 0

            else:
                early_stopping_criterion += 1

            # if early_stopping_criterion >= 500:
            #     break

    end = time.time()
    print(end - start)

    # The prompt is playing a major role in how good of a solution we can obtain
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

    ax1.imshow(img)
    ax1.set_title("Original Image")

    ax2.imshow(model_img)
    ax2.set_title("Reconstructed Image")

    image_save_path = os.path.join(save_file_path, "final_output.png")
    fig.savefig(image_save_path, bbox_inches="tight")
