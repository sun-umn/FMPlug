#!/usr/bin/env python3
"""
Script to run the multi-image super resolution task with OpenAI VLM prompts.
"""

# stdlib
import os
import sys

# Add the parent directory to the path so we can import fmplug
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# first party
from fmplug.tasks.inverse_solver_multi_image import super_resolution_multi_image_task


def main():
    """
    Main function to run the multi-image super resolution task.
    """
    if len(sys.argv) != 2:
        print("Usage: python run_multi_image_task.py <config_name>")
        print("Example: python run_multi_image_task.py config_multi_image")
        sys.exit(1)

    config_name = sys.argv[1]

    print(f"Running multi-image super resolution task with config: {config_name}")

    try:
        super_resolution_multi_image_task(config_name)
        print("Task completed successfully!")
    except Exception as e:
        print(f"Error running task: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
