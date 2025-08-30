# Multi-Image Super Resolution with OpenAI VLM Prompts

This module extends the original inverse solver template to process multiple images with automatically generated prompts using OpenAI's Vision Language Model (GPT-4V).

## Features

1. **Multi-Image Processing**: Processes all PNG images in a specified directory
2. **Automatic Prompt Generation**: Uses OpenAI GPT-4V to generate descriptive prompts for each image
3. **Sigma Trajectory Visualization**: Plots initial vs final sigma trajectories to show optimization progress
4. **Individual Image Tracking**: Each image gets its own set of saved outputs and metrics

## Files

- `inverse_solver_multi_image.py`: Main implementation with multi-image support
- `run_multi_image_task.py`: Script to run the multi-image task
- `config_multi_image.yaml`: Configuration file for multi-image processing

## Configuration

The `config_multi_image.yaml` file contains all necessary parameters:

```yaml
# Image directory containing multiple PNG images
image_directory: "/path/to/your/images"

# OpenAI API key (can also be set via environment variable OPENAI_API_KEY)
openai_api_key: ""  # Leave empty to use environment variable

# Text prompt for generation (optional - if provided, will use this instead of VLM)
# Leave empty to use OpenAI VLM for automatic prompt generation
prompt: ""

# Other parameters remain the same as single-image version
```

## Usage

1. **Set up your image directory**: Place all PNG images you want to process in a directory
2. **Set OpenAI API key**: Either in the config file or as environment variable `OPENAI_API_KEY`
3. **Configure prompts** (optional):
   - Leave `prompt: ""` empty to use OpenAI VLM for automatic prompt generation
   - Set `prompt: "your custom prompt"` to use the same prompt for all images
4. **Run the task**:
   ```bash
   python fmplug/tasks/run_multi_image_task.py config_multi_image
   ```

## Output Structure

For each image, the following files are generated:

- `{image_name}_final_output.png`: Final reconstructed image comparison
- `{image_name}_lpips_assessment.png`: LPIPS layer analysis
- `sigma_trajectory_{image_name}.png`: Sigma trajectory evolution plot
- `{image_name}_reference_vs_generated_image_epoch_*.png`: Training progress images
- `prompts/{image_name}_prompt.txt`: Generated or custom prompt used for the image

Additionally, aggregate metrics are saved:

- `final_metrics.csv`: CSV file containing final metrics for all images (image_name, mse, psnr, ssim, lpips)

## Key Changes from Original

1. **Loop over images**: The main training loop (lines 441-804 from original) is now wrapped in a for loop over all images
2. **OpenAI VLM integration**: `generate_prompt_with_openai_vlm()` function generates prompts for each image
3. **Conditional prompt usage**: If a prompt is provided in config, it uses that instead of VLM generation
4. **Prompt saving**: All prompts (custom or generated) are saved to text files in a prompts directory
5. **Metrics aggregation**: Final metrics (MSE, PSNR, SSIM, LPIPS) are collected and saved to CSV using polars
6. **Sigma trajectory plotting**: `plot_sigma_trajectories()` function visualizes how sigma values change during training
7. **Individual file naming**: All saved files include the image name to avoid conflicts

## Dependencies

Additional dependencies beyond the original:
- `openai`: For GPT-4V API access
- `polars`: For efficient CSV data handling and metrics export
- `glob`: For finding PNG files in directory
- `pathlib`: For path manipulation

## Error Handling

- Falls back to a generic prompt if OpenAI API fails
- Validates that the image directory contains PNG files
- Continues processing other images if one fails

## Notes

- Each image is processed independently with its own optimization run
- The OpenAI VLM prompt generation adds some time to the overall process
- All metrics are logged to wandb with the image name for easy filtering
