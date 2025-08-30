#!/usr/bin/env python3
"""
Example usage of the multi-image super resolution task.

This script demonstrates how to set up and run the multi-image task
with OpenAI VLM prompt generation.
"""

# stdlib
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import fmplug
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# first party
from fmplug.tasks.inverse_solver_multi_image import super_resolution_multi_image_task


def setup_example_environment():
    """
    Set up the example environment with sample images and configuration.
    """
    # Create directories if they don't exist
    data_dir = Path("/users/5/dever120/FMPlug/data/images")
    data_dir.mkdir(parents=True, exist_ok=True)

    experiments_dir = Path("/users/5/dever120/FMPlug/experiments")
    experiments_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {data_dir}")
    print(f"Experiments directory: {experiments_dir}")

    # Check if there are any PNG images in the data directory
    png_files = list(data_dir.glob("*.png"))
    if not png_files:
        print("No PNG images found in data directory.")
        print("Please add some PNG images to:", data_dir)
        print("Example: cp /path/to/your/images/*.png", data_dir)
        return False

    print(f"Found {len(png_files)} PNG images:")
    for png_file in png_files:
        print(f"  - {png_file.name}")

    return True


def check_openai_api_key():
    """
    Check if OpenAI API key is available.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("You can set it with: export OPENAI_API_KEY='your-api-key-here'")
        print("Or add it to the config file.")
        return False

    print("OpenAI API key found in environment variables.")
    return True


def main():
    """
    Main function to demonstrate multi-image task usage.
    """
    print("Multi-Image Super Resolution Task Example")
    print("=" * 50)

    # Check environment setup
    if not setup_example_environment():
        print("Environment setup failed. Please check the data directory.")
        return

    if not check_openai_api_key():
        print("OpenAI API key not found. Prompt generation will use fallback prompts.")

    # Configuration name
    config_name = "config_multi_image"

    print(f"\nRunning multi-image task with config: {config_name}")
    print("This will:")
    print("1. Process all PNG images in the data directory")
    print("2. Generate prompts using OpenAI GPT-4V (if API key available)")
    print("3. Run super resolution optimization for each image")
    print("4. Save results with individual image names")
    print("5. Plot sigma trajectories for each image")

    # Ask for confirmation
    response = input("\nContinue? (y/N): ")
    if response.lower() != "y":
        print("Aborted.")
        return

    try:
        super_resolution_multi_image_task(config_name)
        print("\nTask completed successfully!")
        print("Check the experiments directory for results.")
    except Exception as e:
        print(f"Error running task: {e}")
        # stdlib
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
