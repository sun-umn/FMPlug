# Super Resolution Configuration

This directory contains configuration files for the super resolution task in FMPlug.

## Configuration Files

Each configuration file is a YAML file that contains all the parameters needed for the super resolution task. The naming convention is `config_XXXX.yaml` where XXXX is a unique identifier.

## Example Configuration

The `config_0000.yaml` file contains the following parameters:

### Image and Processing Parameters
- `image_size`: Size of the input image (default: 512)
- `scale_factor`: Super resolution scale factor (default: 4)
- `num_inference_steps`: Number of inference steps (default: 10)
- `guidance_scale`: Guidance scale for generation (default: 3.0)
- `lr`: Learning rate (default: 1e-2)
- `epochs`: Number of training epochs (default: 2500)
- `strength`: Strength parameter (default: 0.85)
- `ode_solver`: ODE solver type (default: "euler")

### Text and Device Configuration
- `prompt`: Text prompt for generation (default: "")
- `device`: Device to use (default: "cuda")
- `seed`: Random seed (default: 0)

### Noise and Paths
- `noise_sigma`: Noise sigma value (default: 0.03)
- `image_path`: Path to the input image
- `prompt_image_path`: Path to the prompt image

### Wandb and Save Configuration
- `wandb_project`: Wandb project name (default: "FMPlug")
- `wandb_tags`: List of wandb tags (default: ["Super Resolution"])
- `save_base_path`: Base path for saving results

## Usage

To use a configuration, call the `super_resolution_task` function with the configuration name (without the `.yaml` extension):

```python
from fmplug.tasks.inverse_solver_template import super_resolution_task

# Use config_0000.yaml
super_resolution_task("config_0000")
```

## Creating New Configurations

To create a new configuration:

1. Copy an existing configuration file
2. Modify the parameters as needed
3. Use a unique name following the `config_XXXX.yaml` pattern
4. Call the function with the new configuration name

## Example

```python
# Load and use configuration
super_resolution_task("config_0000")
```

This will load all parameters from `config_0000.yaml` and run the super resolution task with those settings.





