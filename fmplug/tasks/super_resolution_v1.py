# stdlib
import os
import time

# third party
import lpips
import matplotlib.pyplot as plt
import torch
import tqdm
import wandb
from diffusers import AutoencoderTiny, StableDiffusion3Img2ImgPipeline
from skimage.metrics import peak_signal_noise_ratio

# first party
from fmplug.ode_solver.euler import integrate_euler
from fmplug.tasks.utils import compute_ssim, prepare_super_resolution_measurement
from fmplug.utils.measurements import get_noise, get_operator

# These presets are used for torch compile for SD3
torch.set_float32_matmul_precision("high")

torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True


def super_resolution_task() -> None:
    # Configuration
    image_size = 512
    scale_factor = 4
    num_inference_steps = 3
    batch_size = 1
    guidance_scale = 2.0
    lr = 1e-1
    epochs = 5000
    dtype = torch.float32

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
            "lr": lr,
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
    lpips_loss_fn = lpips.LPIPS(net="alex").to(device)

    # Load in an image & measurement
    noise_sigma = 0.03
    img_outputs = prepare_super_resolution_measurement(
        image_path="/users/5/dever120/FMPlug/data/div2k_example.png",  # Hardcode for now  # noqa
        image_size=image_size,
        scale_factor=scale_factor,
        noise_sigma=noise_sigma,
        device=device,
        get_operator_fn=get_operator,
        get_noise_fn=get_noise,
    )

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

    pipe.transformer = torch.compile(
        pipe.transformer, mode="max-autotune", fullgraph=True
    )
    pipe.vae.decode = torch.compile(
        pipe.vae.decode, mode="max-autotune", fullgraph=True
    )

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
    max_sequence_length = 256
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
    timesteps = timesteps[num_inference_mask].to(device)
    sigmas = sigmas[num_inference_mask].to(device)

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
    random_image = torch.rand((1, 3, image_size, image_size))
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

    # Setup optimizer
    optimizer = torch.optim.AdamW([z], lr=lr, weight_decay=0.0, amsgrad=True)

    # Export the measurment operator
    operator = img_outputs["operator"]

    # Create the loss function
    criterion = torch.nn.MSELoss()

    # Try the grad scaler for fp16
    scaler = torch.amp.GradScaler()

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
    with torch.no_grad():
        for _ in range(3):
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
            noise = torch.randn(x_t.shape, generator=None, dtype=dtype, layout=None).to(
                device
            )
            last_sigma = sigmas[-1]

            # Step 1: Add noise inverse
            decoded_latent = (x_t - last_sigma * noise) / (1 - last_sigma)

            # Step 2: Add shift and scale inverse
            decoded_latent = (
                decoded_latent / vae.config.scaling_factor
            ) + vae.config.shift_factor

            # Step 3: Decode using VAE / AE - this output is [-1, 1]
            decoded_output = torch.clamp(vae.decode(decoded_latent).sample, -1.0, 1.0)

    print("Ending compile warmup ...")

    start = time.time()
    for idx, epoch in tqdm.tqdm(enumerate(range(epochs))):
        # First zero gradients
        optimizer.zero_grad()

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
        noise = torch.randn(x_t.shape, generator=None, dtype=dtype, layout=None).to(
            device
        )
        last_sigma = sigmas[-1]

        # Step 1: Add noise inverse
        decoded_latent = (x_t - last_sigma * noise) / (1 - last_sigma)

        # Step 2: Add shift and scale inverse
        decoded_latent = (
            decoded_latent / vae.config.scaling_factor
        ) + vae.config.shift_factor

        # Step 3: Decode using VAE / AE - this output is [-1, 1]
        decoded_output = torch.clamp(vae.decode(decoded_latent).sample, -1.0, 1.0)

        # Now apply the degradation
        operator_decoded_output = operator.forward(decoded_output)

        # Apply the loss function - this expects [-1, 1]
        lpips_loss = lpips_loss_fn(operator_decoded_output, y_n)
        loss = criterion(operator_decoded_output, y_n) + lpips_loss

        # Update gradients of z
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Compute the PSNR & print loss
        with torch.no_grad():
            lpips_score = lpips_loss_fn(decoded_output, img_outputs.get("ref_img"))

            model_img = (decoded_output + 1.0) / 2.0  # type: ignore
            model_img = model_img.squeeze(0).detach().cpu().numpy()  # type: ignore

            img = img_outputs.get("ref_img")
            img = (img + 1.0) / 2.0
            img = img.squeeze(0).detach().cpu().numpy()

            ssim_score = compute_ssim(
                img,
                model_img,
            )

            img = img.transpose(1, 2, 0)  # type: ignore
            model_img = model_img.transpose(1, 2, 0)  # type: ignore

            psnr_score = peak_signal_noise_ratio(img, model_img)
            mse_score = ((img - model_img) ** 2).mean()

            metrics_to_log = {
                "epoch": epoch,
                "mse_loss": mse_score,
                "psnr": psnr_score,
                "ssim": ssim_score,
                "lpips": lpips_score.item(),
            }
            wandb.log(metrics_to_log)  # type: ignore

            if idx == 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

                ax1.imshow(img)
                ax1.set_title("Original Image")

                ax2.imshow(model_img)
                ax2.set_title("Reconstructed Image")

                image_save_path = os.path.join(
                    save_file_path, "reference_vs_first_generated_image.png"
                )
                fig.savefig(image_save_path, bbox_inches="tight")

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
