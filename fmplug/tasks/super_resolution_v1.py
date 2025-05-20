# third party
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm
import wandb
from flow_matching.utils import ModelWrapper
from huggingface_hub import login
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from torchvision import transforms

# first party
from fmplug.losses.losses import PerceptualLoss
from fmplug.models.stable_diffusion import StableDiffusion3Base
from fmplug.utils.measurements import get_noise, get_operator


class CFGSSD3caledModel(ModelWrapper):
    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.model = model
        self.nfe_counter = 0

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        label,
        cfg_scale,
        prompt_embedding: torch.Tensor,
        pooled_embedding: torch.Tensor,
    ):
        t = torch.zeros(x.shape[0], device=x.device) + t

        with (
            torch.cuda.amp.autocast(enabled=True, dtype=torch.float32),
            torch.no_grad(),
        ):
            result = self.model.predict_vector(
                x,
                t,
                prompt_embedding,
                pooled_embedding,
            )

        result = self.model.decode(result)
        result = self.model.encode(result)

        self.nfe_counter += 1

        return result.to(dtype=torch.float32)

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter


# Custom Runge-Kutta 4
def rk4_step_jit(f, t0, dt, t1, x0, prompt_embedding, pooled_embedding):
    half_dt = 0.5 * dt

    k1 = f(x0, t0, prompt_embedding, pooled_embedding)

    k2 = f(x0 + half_dt * k1, t0 + half_dt, prompt_embedding, pooled_embedding)

    k3 = f(x0 + half_dt * k2, t0 + half_dt, prompt_embedding, pooled_embedding)

    k4 = f(x0 + dt * k3, t1, prompt_embedding, pooled_embedding)

    return (k1 + 2 * (k2 + k3) + k4) * dt * (1 / 6)


# @torch.jit.script
@torch.compile
def cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t):
    h = (t - t0) / (t1 - t0)

    h00 = (1 + 2 * h) * (1 - h) * (1 - h)

    h10 = h * (1 - h) * (1 - h)

    h01 = h * h * (3 - 2 * h)

    h11 = h * h * (h - 1)

    dt = t1 - t0

    return h00 * y0 + h10 * dt * f0 + h01 * y1 + h11 * dt * f1


@torch.compile
def linear_interp(t0, t1, y0, y1, t):
    if t == t0:
        return y0

    if t == t1:
        return y1

    slope = (t - t0) / (t1 - t0)
    return y0 + slope * (y1 - y0)


def integrate(f, x0, t_grid, prompt_embedding, pooled_embedding):
    # x0 will be the initial encoded v
    x = x0
    j = 1

    t_grid_pairs = zip(t_grid[:-1], t_grid[1:])

    for t0, t1 in t_grid_pairs:
        dt = t1 - t0

        # Get the initial solution
        f0 = f(x, t0, prompt_embedding, pooled_embedding)

        # # RK4
        # dx = rk4_step_jit(f, t0, dt, t1, x, prompt_embedding, pooled_embedding)

        # Euler
        dx = dt * f0

        x1 = x + dx

        while j < len(t_grid) and t1 >= t_grid[j]:
            # f1 = f(x1, t1, prompt_embedding, pooled_embedding)
            # solution = cubic_hermite_interp(t0, x, f0, t1, x1, f1, t_grid[j])
            solution = linear_interp(t0, t1, x, x1, t_grid[j])

            j += 1

        x = x1

        del f0
        del dx

        torch.cuda.empty_cache()

    return solution


def fft_loss(pred, target):
    pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
    target_fft = torch.fft.fft2(target, dim=(-2, -1))
    return torch.abs(pred_fft - target_fft).mean()


def super_resolution_task() -> None:
    # Log into huggingface to be able to pull the SD3.0
    print("Log into HuggingFace ...")
    login("hf_access_token")

    # Setup device as cuda
    device = torch.device("cuda")

    # Global variables for wandb
    API_KEY = "2080070c4753d0384b073105ed75e1f46669e4bf"
    PROJECT_NAME = "FMPlug"

    # Enable wandb
    print("Initialize Project ...")
    wandb.login(key=API_KEY)  # type: ignore

    _ = wandb.init(  # type: ignore
        # set the wandb project where this run will be logged
        project=PROJECT_NAME,
        tags=["Experimental", "super resolution"],
    )

    # Setup image size
    image_size = 768
    scale_factor = 12

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

    # Load in SD3.0
    print("Load in Stable Diffusion 3.0 ...")
    sd3_base = StableDiffusion3Base(device=device, dtype=torch.float32)

    # Setup prompt and null embedding
    prompts = [
        "a high quality photo of animal, bush, close-up, fox, grass, green, greenery, hide, panda, red, red panda, stare"  # noqa
    ]

    batch_size = 1
    # with torch.no_grad():
    prompt_embedding, pooled_embedding = sd3_base.encode_prompt(prompts, batch_size)
    null_prompt_embedding, null_pooled_embedding = sd3_base.encode_prompt(
        [""], batch_size
    )

    # Put on cuda device
    prompt_embedding = prompt_embedding.to(device)
    pooled_embedding = pooled_embedding.to(device)

    # Setup model
    cfg_scaled_model = CFGSSD3caledModel(model=sd3_base)  # type: ignore
    cfg_scaled_model = cfg_scaled_model.requires_grad_(False)

    def f(x, t, prompt_embedding, pooled_embedding):
        return cfg_scaled_model(
            x=x,
            t=t,
            label=None,
            cfg_scale=0.0,
            prompt_embedding=prompt_embedding,
            pooled_embedding=pooled_embedding,
        )

    # Setup time grid
    time_grid = torch.clamp(torch.linspace(0, 1, 20), 0, 1).to(device)

    # Setup z
    z = torch.load("/users/5/dever120/FMPlug/div2k-encoded-z-v3.pt")
    # z = torch.load("div2k-encoded-z-256-v1.pt")
    z = torch.nn.parameter.Parameter(z, True)
    z = z.to(torch.float32)

    # Make prompt embedding and pooled embedding trainable
    prompt_embedding = torch.nn.parameter.Parameter(prompt_embedding, True)
    pooled_embedding = torch.nn.parameter.Parameter(pooled_embedding, True)

    # TODO: Configs should be at the top of the function
    lr = 5e-1
    epochs = 25

    # Criterion for learning
    criterion = torch.nn.L1Loss().to(device)

    # Setup perceptual loss
    _ = PerceptualLoss(layer="relu3_3").to(device)  # type: ignore

    # parameter groups for LBFGS
    params_group = {"params": z, "lr": lr}

    optimizer = torch.optim.LBFGS(
        [params_group], max_iter=10, history_size=10, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()

        with (torch.cuda.amp.autocast(enabled=True, dtype=torch.float32),):
            x_t = integrate(
                f,
                z,
                time_grid,
                prompt_embedding,
                pooled_embedding,
            )
            decoded_output = sd3_base.decode(x_t)

        # decoded_output = decoded_output.to(torch.float32)
        decoded_output = decoded_output.to(torch.float32)

        # Store for access after step
        decoded_output_holder[0] = decoded_output.detach()

        operator_decoded_output = operator.forward(decoded_output)

        # perceptual loss
        # percep_pred_input = torch.clamp((operator_decoded_output + 1.0) / 2.0, 0.0, 1.0)  # noqa
        # percep_target_input = torch.clamp((y_n + 1.0) / 2.0, 0.0, 1.0)

        # perceptual_loss = percep_loss_fn(percep_pred_input, percep_target_input)

        loss = 10.0 * criterion(operator_decoded_output, y_n)

        # + 0.1 * fft_loss(
        #     operator_decoded_output, y_n
        # )
        # loss = 10.0 * criterion(operator_decoded_output, y_n) + 10.0 * perceptual_loss
        # loss = torch.linalg.norm(operator_decoded_output - y_n)

        loss.backward()

        print(z.grad.min(), z.grad.max())

        del x_t
        torch.cuda.empty_cache()

        return loss

    psnrs = []
    losses = []
    best_images = []

    for iterator in tqdm.tqdm(range(epochs)):
        decoded_output_holder = [None]  # mutable object to hold output

        loss = optimizer.step(closure)
        losses.append(loss.item())

        # decoded_output = decoded_output_holder[0]

        with torch.no_grad():
            x_t = integrate(
                f,
                z,
                time_grid,
                prompt_embedding,
                pooled_embedding,
            )
            decoded_output = sd3_base.decode(x_t)

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
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10))

    # Ground truth
    ax1.imshow(display_ref_img)
    ax1.set_title("Ground Truth Image")
    ax1.axis("off")

    # Reconstructed image
    ax2.imshow(best_img.astype(float))
    ax2.set_title("Reconstructed Image (FM)")
    ax2.axis("off")

    # Pixel-wise absolute difference (grayscale)
    diff = np.abs(display_ref_img - best_img).astype(float).mean(axis=-1)
    ax3.imshow(diff, cmap="hot")
    ax3.set_title("Pixel Difference (FM)")
    ax3.axis("off")

    # Save the figure
    plt.tight_layout()
    fig.savefig("/users/5/dever120/FMPlug/best_img.png", dpi=300)
    plt.close()
