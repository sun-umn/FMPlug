![](https://github.com/sun-umn/FMPlug/blob/main/banner/flow_matching_gh_banner.png)

# FMPlug

Official code release for **FMPlug: Plug-and-Play Flow Matching for Image Inverse Problems**.

- Paper: https://arxiv.org/abs/2511.16520
- Project page: https://sun-umn.github.io/xm-plug/

FMPlug solves image inverse problems by optimizing in the latent space of a pretrained flow matching model. This repository contains the FMPlug implementation, example inputs, task configurations, and command-line entry points for running reconstruction experiments.

## Repository Layout

```text
.
|-- fmplug/                  # Main FMPlug package
|   |-- configs/             # YAML task configurations
|   |-- models/              # Model wrappers and network components
|   |-- tasks/               # Reconstruction task implementations
|   `-- utils/               # Measurements, logging, image utilities, etc.
|-- example/                 # Small example data layout for image tasks
|-- run_fmplug.py            # Standard image inverse-problem CLI
|-- cli_mri.py               # Scientific InverseBench-backed CLI
|-- bkse/                    # External blur kernel space repo; clone separately
|-- motionblur/              # External motion blur repo; clone separately
`-- InverseBench/            # External scientific benchmark repo; clone separately
```

## External Repositories

`bkse`, `motionblur`, and `InverseBench` are external repositories used by this project. They are included in this working tree for development convenience, but users should clone them from their original sources when setting up the project.

From the repository root:

```bash
git clone https://github.com/VinAIResearch/blur-kernel-space-exploring bkse
git clone https://github.com/LeviBorodenko/motionblur motionblur
git clone https://github.com/devzhk/InverseBench.git InverseBench
```

The `motionblur` package is imported by FMPlug's blur measurement operators. `bkse` provides blur-kernel-space-exploring utilities and pretrained kernel assets used by blur-related experiments.

`InverseBench` is required only for scientific tasks such as MRI, inverse scattering, FWI, and black-hole imaging. The scientific runner imports InverseBench modules directly from the clone path configured in `fmplug/configs/scientific_mri_config.yaml`.

## Environment Setup with `uv`

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create and synchronize the Python environment:

```bash
uv python pin 3.9
uv venv --python 3.9
uv sync
```

Activate the environment:

```bash
source .venv/bin/activate
```

This project is GPU-oriented. The default `uv` configuration uses the PyTorch CUDA 12.1 wheel index for `torch` and `torchvision`. If your system requires a different CUDA build, edit the PyTorch index in `pyproject.toml` before running `uv sync`.

Scientific tasks may also need optional dependencies from InverseBench, depending on the problem. Install the InverseBench environment or add the task-specific dependencies to this environment before running those tasks.

## Model and Data Requirements

FMPlug loads Stable Diffusion 3 Medium through Hugging Face:

```text
stabilityai/stable-diffusion-3-medium-diffusers
```

Make sure you have access to the model and are logged in:

```bash
huggingface-cli login
```


For standard image tasks, place task data under `example/` or update `fmplug.data_folder` in the YAML config. Each sample directory should contain:

```text
gt.png
prompt.txt
```

For scientific tasks, place or link the InverseBench datasets under `scientific.data_root`, or set `INVERSEBENCH_DATA_ROOT`.

The output directory is controlled by `fmplug.save_folder` in the config and defaults to `./experiment-FMPlug`.

## Running FMPlug

Available example configs are in `fmplug/configs/`:

- `superresolution_config.yaml`
- `gaussian_blur_config.yaml`
- `motion_blur_config.yaml`
- `inpainting_config.yaml`

Run a standard image task by passing the config name without the `.yaml` suffix:

```bash
uv run python run_fmplug.py run-FMPlug-task --config_name superresolution_config
```

Other standard examples:

```bash
uv run python run_fmplug.py run-FMPlug-task --config_name gaussian_blur_config
uv run python run_fmplug.py run-FMPlug-task --config_name motion_blur_config
uv run python run_fmplug.py run-FMPlug-task --config_name inpainting_config
```

## Running Scientific Tasks

Scientific tasks use `cli_mri.py` and require a local InverseBench clone:

```bash
git clone https://github.com/devzhk/InverseBench.git InverseBench
```

Edit `fmplug/configs/scientific_mri_config.yaml` before running:

- `scientific.inversebench_path`: path to the cloned InverseBench repository. This can also be set with `INVERSEBENCH_ROOT`.
- `scientific.data_root`: directory containing the InverseBench datasets. This can also be set with `INVERSEBENCH_DATA_ROOT`.
- `scientific.datasets.*`: task-specific dataset folders, such as `mri_test` and `mri_val`.
- `scientific.norm_checkpoint`: path to `best_meanvar_model.pth`.
- `scientific.wandb_mode`: defaults to `offline`; set `WANDB_API_KEY` and use `online` if you want remote W&B logging.

Run the scientific MRI example:

```bash
python run_fmplug_inversebench.py run-FMPlug-scientific-task --config_name inversebench/mri_config
```

## Configuration

Task settings live in `fmplug/configs/*.yaml`. The main sections are:

- `measurement`: forward operator and noise model.
- `fmplug`: optimization settings, image size, solver settings, data path, and output path.
- `scientific`: InverseBench path, scientific dataset paths, W&B behavior, CUDA device, and scientific task defaults.

Common settings to change:

- `fmplug.data_folder`: root directory containing task inputs for standard image tasks.
- `fmplug.save_folder`: output directory for reconstructions and logs.
- `fmplug.epochs`: optimization iterations.
- `fmplug.NFE`: number of function evaluations for the flow solver.
- `measurement.operator.name`: inverse problem type.
- `scientific.inversebench_path`: InverseBench clone path for scientific tasks.
- `scientific.data_root` and `scientific.datasets.*`: dataset locations for scientific tasks.

## Notes

- The runners currently assume CUDA is available.
- The SD3 model download may require substantial disk space and GPU memory.
- `bkse`, `motionblur`, and `InverseBench` keep their original licenses and attribution. Check their upstream repositories for details.

## Citation

If you use this code, please cite the FMPlug paper:

```bibtex
@misc{wan2026savingfoundationflowmatchingpriors,
      title={Saving Foundation Flow-Matching Priors for Inverse Problems}, 
      author={Yuxiang Wan and Ryan Devera and Wenjie Zhang and Ju Sun},
      year={2026},
      eprint={2511.16520},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.16520}, 
}
```
