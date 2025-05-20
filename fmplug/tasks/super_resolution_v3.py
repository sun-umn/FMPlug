# stdlib
import os
import random

# third party
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import wandb
import yaml  # type: ignore
from diffusers import StableDiffusion3Img2ImgPipeline
from huggingface_hub import login
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from torchvision import transforms

# first party
from fmplug.utils.measurements import get_noise, get_operator


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


@torch.compile
def linear_interp(t0, t1, y0, y1, t):
    if t == t0:
        return y0

    if t == t1:
        return y1

    slope = (t - t0) / (t1 - t0)
    return y0 + slope * (y1 - y0)


def integrate(
    f,
    x0,
    timesteps,
    sigmas,
    prompt_embedding,
    pooled_embedding,
    device,
    guidance_scale: float = 7.0,
):
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
        # x0 will be the latent variable
        latent_model_input = torch.cat([x0] * 2) if do_classifier_free_guidance else x0

        # broadcast to batch dimension in a way that's compatible with ONNX / Core ML
        timestep = t0.expand(latent_model_input.shape[0])
        # prev_timestep = t1.expand(latent_model_input.shape[0])

        noise_pred = f(
            x=latent_model_input,
            t=timestep,
            prompt_embedding=prompt_embedding,
            pooled_embedding=pooled_embedding,
            device=device,
        )[0]

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        # upcast to avoid precision issues
        sample = x0.to(torch.float32)
        dt = sigma_next - sigma
        prev_sample = sample + dt * noise_pred
        prev_sample = prev_sample.to(noise_pred.dtype)

        # NOTE: There was a bug in the notebook and solution is very blurry
        # and does not provide a good prior. Commenting out for now.
        # while j < len(timesteps) and t1 >= timesteps[j]:
        #     solution = linear_interp(
        #         sigma, sigma_next, sample, prev_sample, timesteps[j]
        #     )
        #     j += 1

        x0 = prev_sample

    return x0


def super_resolution_task(config_name: str) -> None:
    # Load in the configuration
    base_config_path = "/users/5/dever120/FMPlug/fmplug/configs"
    with open(os.path.join(base_config_path, config_name + ".yaml"), "r") as file:
        config = yaml.safe_load(file)["config"]

    image_size = height = width = config["image_size"]
    scale_factor = config["scale_factor"]
    num_inference_steps = config["num_inference_steps"]
    batch_size = config["batch_size"]
    guidance_scale = config["guidance_scale"]
    lr = config["lr"]
    epochs = config["epochs"]
    loss_multiplier = config["loss_multiplier"]
    loss_fn = config["loss_fn"]
    max_iter = config["max_iter"]

    # Fix a seed
    set_seed(1985)

    # Log into huggingface to be able to pull the SD3.0
    print("Log into HuggingFace ...")
    login("REMOVED")

    # Setup device as cuda
    device = torch.device("cuda")

    # Global variables for wandb
    API_KEY = "2080070c4753d0384b073105ed75e1f46669e4bf"
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
            "loss_multiplier": loss_multiplier,
            "image_size": image_size,
            "scale_factor": scale_factor,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "lbfgs_max_iter": max_iter,
        },
    )

    # Create the directory to save all of the model results
    wandb_experiment_id = wandb_instance.id
    save_file_path = f"/users/5/dever120/FMPlug/experiments/{wandb_experiment_id}"
    os.makedirs(save_file_path, exist_ok=True)

    gt_img_path = "/users/5/dever120/FMPlug/data/div2k_example.png"
    gt_img = Image.open(gt_img_path).convert("RGB")

    tf = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    gt_img = tf(gt_img)

    ref_numpy = np.array(gt_img)
    x = gt_img * 2.0 - 1.0  # type: ignore

    ref_img = torch.Tensor(x).to(torch.float32).to(device).unsqueeze(0)
    ref_img.requires_grad = False

    # Super resolution
    # TODO: make image size configurable
    config = {
        "measurement": {
            "operator": {
                "name": "super_resolution",
                "in_shape": (1, 3, image_size, image_size),
                "scale_factor": scale_factor,
            },
            "noise": {"name": "gaussian", "sigma": 0.01},
        },
    }

    measure_config = config["measurement"]

    # Initalize operator
    operator = get_operator(device=device, **measure_config["operator"])  # type: ignore
    noiser = get_noise(**measure_config["noise"])  # type: ignore

    # Forward measurement model (Ax + n)
    y = operator.forward(ref_img)
    y_n = noiser(y)

    print("Load in SD3 image to image model ...")
    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
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

    # VAE
    vae = pipe.vae
    vae.eval()
    # vae.requires_grad_(False)
    vae.enable_gradient_checkpointing()

    # Start the problem by defining z
    # The inputs for the image to image pipeline need to be between
    # 0 and 1
    z = torch.randn(3, image_size, image_size)

    # prompt
    prompt = "a high quality, unclose photo of a red panda's face in the jungle"
    # prompt_2 = (
    #     "Ultra-detailed photo of a red panda in the wild, realistic fur, "
    #     "bright expressive eyes, eating bamboo leaves, DSLR wildlife photography, "
    #     "bokeh background"
    # )
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

    # preprocess image
    print("Process image, build timesteps & latent variables ...")
    image = image_processor.preprocess(z, height=height, width=width)

    # 4. Prepare timesteps
    timesteps = torch.linspace(1000.0, 0.0, num_inference_steps + 1).to(device)
    sigmas = timesteps / 1000.0

    # Prepare the latent timestep so we can build the latent variable
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

    # 5. Prepare latent variables
    latents = prepare_latents(
        image,
        latent_timestep,
        batch_size,
        num_images_per_prompt,
        prompt_embeds.dtype,
        device,
        generator=None,
    )
    z = torch.nn.parameter.Parameter(latents)

    # Solve the ODE
    print("Solve inverse problem ...")

    def f(x, t, prompt_embedding, pooled_embedding, device):
        with torch.amp.autocast(device.type, dtype=torch.float16):
            return transformer(
                hidden_states=x,
                timestep=t,
                encoder_hidden_states=prompt_embedding,
                pooled_projections=pooled_embedding,
                joint_attention_kwargs=None,
                return_dict=False,
            )

    # Criterion for learning
    if loss_fn == "l1":
        criterion = torch.nn.L1Loss().to(device)

    elif loss_fn == "mse":
        criterion = torch.nn.MSELoss().to(device)  # type: ignore

    # Setup perceptual loss
    # percep_loss_fn = PerceptualLoss(layer="relu3_3").to(device)

    # parameter groups for LBFGS
    params_group = {"params": z, "lr": lr}

    optimizer = torch.optim.AdamW(
        [params_group],
        weight_decay=1e-2,
        amsgrad=True,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",  # or 'max' for metrics like PSNR
        factor=0.5,  # multiply LR by this factor
        patience=1,  # wait N epochs before reducing
        threshold=1e-3,  # minimum change to qualify as "improvement"
        verbose=True,  # logs LR changes
    )

    def closure():
        optimizer.zero_grad()

        with (torch.cuda.amp.autocast(enabled=True, dtype=torch.float16),):
            x_t = integrate(
                f,
                z,
                timesteps,
                sigmas,
                prompt_embedding,
                pooled_embedding,
                device,
                guidance_scale=guidance_scale,
            )
            x_t = (x_t / vae.config.scaling_factor) + vae.config.shift_factor
            decoded_output = vae.decode(x_t.half())
            # decoded_output = vae.decode(x_t)

        # decoded_output = decoded_output.to(torch.float32)
        # decoded_output = torch.tanh(decoded_output.sample.to(torch.float32))
        decoded_output = decoded_output.sample.to(torch.float32)
        decoded_output = torch.clamp(decoded_output, -1.0, 1.0)

        # Output shapes checkout
        # Verified that outputs are between -1 and 1
        # print(decoded_output.min(), decoded_output.max())

        # Store for access after step
        decoded_output_holder[0] = decoded_output.detach()

        operator_decoded_output = operator.forward(decoded_output)

        loss = loss_multiplier * criterion(operator_decoded_output, y_n)

        loss.backward()
        optimizer.step()

        # print out gradients so we can understand
        print(z.grad.min(), z.grad.max())

        del x_t
        torch.cuda.empty_cache()

        return loss

    psnrs = []
    losses = []
    best_images = []

    for iterator in tqdm.tqdm(range(epochs)):
        decoded_output_holder = [None]  # mutable object to hold output

        # first-order solver AdamW is wrapped in this
        # closure
        loss = closure()

        losses.append(loss.item())

        with torch.no_grad():
            x_t = integrate(
                f,
                z,
                timesteps,
                sigmas,
                prompt_embedding,
                pooled_embedding,
                device,
                guidance_scale=guidance_scale,
            )
            x_t = (x_t / vae.config.scaling_factor) + vae.config.shift_factor
            decoded_output = vae.decode(x_t.to(torch.float32)).sample

        decoded_output = decoded_output.to(torch.float32)
        decoded_output = torch.clamp(decoded_output, -1.0, 1.0)

        # Evaluate
        with torch.no_grad():
            output_numpy = decoded_output.detach().cpu().squeeze().numpy()
            output_numpy = (output_numpy + 1) / 2
            output_numpy = np.transpose(
                output_numpy, (1, 2, 0)
            )  # Keep out for now lets evaluate

            # calculate psnr
            tmp_psnr = peak_signal_noise_ratio(
                ref_numpy.transpose(1, 2, 0), output_numpy
            )
            print(tmp_psnr)

            scheduler.step(tmp_psnr)

            # calculate mse
            mse_score = ((ref_numpy.transpose(1, 2, 0) - output_numpy) ** 2).mean()

            metrics_to_log = {
                "epoch": iterator,
                "psnr": tmp_psnr,
                "mse_loss": mse_score,
            }
            wandb.log(metrics_to_log)  # type: ignore

            psnrs.append(tmp_psnr)

            if len(psnrs) == 1 or (len(psnrs) > 1 and tmp_psnr > np.max(psnrs[:-1])):
                best_img = output_numpy
                best_images.append(best_img)

    display_ref_img = ref_numpy.transpose(1, 2, 0)

    # Create a figure with 1 row, 3 columns
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 10))

    # Ground truth
    ax1.imshow(display_ref_img)
    ax1.set_title("Ground Truth Image")
    ax1.axis("off")

    # Display the corrupted image
    y_n_numpy = y_n.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    y_n_numpy = (y_n_numpy + 1.0) / 2.0
    ax2.imshow(y_n_numpy)
    ax2.set_title("Corrupted Image")
    ax2.axis("off")

    # Reconstructed image
    ax3.imshow(best_img.astype(float))
    ax3.set_title("Reconstructed Image (FM)")
    ax3.axis("off")

    # Pixel-wise absolute difference (grayscale)
    diff = np.abs(display_ref_img - best_img).astype(float).mean(axis=-1)
    ax4.imshow(diff, cmap="hot")
    ax4.set_title("Pixel Difference (FM)")
    ax4.axis("off")

    # Save the figure
    plt.tight_layout()
    fig.savefig(os.path.join(save_file_path, "best_img.png"), bbox_inches="tight")
    plt.close()

    # Save the raw data as well
    np.save("ground_truth_image.npy", display_ref_img)
    np.save("reconstructed_image.npy", best_img)
