# third party
import torch
from diffusers import StableDiffusion3Img2ImgPipeline
from skimage.metrics import peak_signal_noise_ratio

# first party
from fmplug.ode_solver.euler import integrate_euler
from fmplug.tasks.utils import prepare_super_resolution_measurement
from fmplug.utils.measurements import get_noise, get_operator


def super_resolution_task() -> None:
    # Configuration
    image_size = 512
    scale_factor = 4
    # num_inference_steps = 10
    batch_size = 1
    guidance_scale = 2.0
    # lr = 1.0
    # epochs = 10
    # loss_multplier = 1.0
    # loss_fn = "mse"
    # max_iter = 20

    # Log into huggingface to be able to pull the SD3.0
    # print("Log into HuggingFace ...")
    # login(os.environ.get("HF_ACCESS_TOKEN"))

    # Setup device as cuda
    device = torch.device("cuda")

    # Load in an image & measurement
    noise_sigma = 0.01
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
    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        text_encoder_3=None,
        tokenizer_3=None,
        torch_dtype=torch.float32,
    )
    pipe = pipe.to(device)

    # Extract different components of the pipeline
    prompt_encoder = pipe.encode_prompt
    image_processor = pipe.image_processor
    prepare_latents = pipe.prepare_latents

    # Transformer block
    transformer = pipe.transformer
    transformer.eval()
    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()

    # AE / VAE
    vae = pipe.vae
    vae.eval()
    vae.requires_grad_(False)
    vae.enable_gradient_checkpointing()

    # Define the prompts and embeddings
    prompt = "a high quality, unclose photo of a red panda's face in the jungle"
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
    mask = timesteps < 600.0
    timesteps = timesteps[mask]
    sigmas = sigmas[mask]

    # Now get num_inference spaced timesteps and sigmas
    num_inference_mask = torch.linspace(0, len(timesteps) - 1, 25).long()
    timesteps = timesteps[num_inference_mask].to(device)
    sigmas = sigmas[num_inference_mask].to(device)

    print("Process image, build timesteps & latent variables ...")
    # When we are feeding in the image it needs to be in the range [0, 1]
    # The output of the image preprocessor will be in the range [-1, 1]

    # Now the real test will be to produce the image using the degraded image
    # Degraded img and label
    y_n = img_outputs["y_n"]

    upsampled_img = torch.nn.functional.interpolate(
        (y_n + 1.0) / 2.0,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )

    image = image_processor.preprocess(
        upsampled_img, height=image_size, width=image_size
    )

    print(
        image.min(),
        image.max(),
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
    optimizer = torch.optim.AdamW([z], lr=1e-1, weight_decay=0.0)

    # Export the measurment operator
    operator = img_outputs["operator"]

    # Create the loss function
    criterion = torch.nn.MSELoss()

    # Function for integration
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

    epochs = 150
    for idx, epoch in enumerate(range(epochs)):
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
        noise = torch.randn(
            x_t.shape, generator=None, dtype=torch.float32, layout=None
        ).to(device)
        last_sigma = sigmas[-1]

        # Step 1: Add noise inverse
        decoded_latent = (x_t - last_sigma * noise) / (1 - last_sigma)

        # Step 2: Add shift and scale inverse
        decoded_latent = (
            decoded_latent / vae.config.scaling_factor
        ) + vae.config.shift_factor

        # Step 3: Decode using VAE / AE - this output is [-1, 1]
        decoded_output = vae.decode(decoded_latent).sample

        # Now apply the degradation
        operator_decoded_output = operator.forward(decoded_output)

        # Apply the loss function - this expects [-1, 1]
        loss = criterion(operator_decoded_output, y_n)

        # Update gradients of z
        loss.backward()
        optimizer.step()
        # print(z.grad.min(), z.grad.max(), z.grad.norm(), z.norm())

        # Compute the PSNR & print loss
        with torch.no_grad():
            model_img = (decoded_output + 1.0) / 2.0
            model_img = model_img.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)

            img = img_outputs.get("ref_img")
            img = (img + 1.0) / 2.0
            img = img.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)

            psnr = peak_signal_noise_ratio(img, model_img)
            print(idx, loss.item(), psnr)
