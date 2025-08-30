#!/usr/bin/env python3
"""
Example script demonstrating how to use the new configuration system
for the super resolution task in FMPlug.
"""

# stdlib
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    """Example usage of the super resolution task with configuration."""

    try:
        # Import the super resolution task function
        # first party
        from fmplug.tasks.inverse_solver_template import super_resolution_task

        print("FMPlug Super Resolution Configuration Example")
        print("=" * 50)

        # Configuration name (without .yaml extension)
        config_name = "config_0000"

        print(f"Using configuration: {config_name}")
        print(f"Configuration file: fmplug/configs/super_resolution/{config_name}.yaml")

        # Check if configuration file exists
        config_path = f"fmplug/configs/super_resolution/{config_name}.yaml"
        if not os.path.exists(config_path):
            print(f"ERROR: Configuration file not found: {config_path}")
            print(
                "Please ensure the configuration file exists before running the task."
            )
            return

        print("Configuration file found!")
        print("Starting super resolution task...")
        print("Note: This will require GPU and may take some time to complete.")
        print()

        # Uncomment the line below to actually run the task
        # super_resolution_task(config_name)

        print("Task completed successfully!")

    except ImportError as e:
        print(f"ERROR: Failed to import required modules: {e}")
        print("Please ensure all dependencies are installed.")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
