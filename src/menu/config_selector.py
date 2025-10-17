"""
Configuration selection menu for Takenoko
"""

import os
import sys
import glob
from typing import List, Optional


def find_config_files(config_dir: str = "configs") -> List[str]:
    """Find all TOML configuration files in the specified directory"""
    if not os.path.exists(config_dir):
        raise ValueError(f"Configuration directory not found: {config_dir}")

    config_files = glob.glob(os.path.join(config_dir, "*.toml"))
    config_files = [f.replace("\\", "/") for f in config_files if os.path.exists(f)]

    if not config_files:
        raise ValueError(f"No TOML configuration files found in {config_dir}")

    return sorted(config_files)


def display_config_menu(config_files: List[str]) -> str:
    """Display configuration selection menu and return selected config"""
    print("=" * 50)
    print("Takenoko - Configuration Selection")
    print("=" * 50)
    print("0. Quit")

    for i, config_file in enumerate(config_files, 1):
        display_name = config_file.replace("\\", "/").split("/")[-1]
        print(f"{i}. {display_name}")

    print("=" * 50)

    while True:
        try:
            choice = input(
                f"Enter the number of the config to run (0-{len(config_files)}): "
            ).strip()

            # Validate input: must be non-empty digits only
            if not choice.isdigit():
                print("Invalid input. Please enter a number.")
                continue

            choice_num = int(choice)

            if choice_num == 0:
                print("Exiting...")
                sys.exit(0)

            if 1 <= choice_num <= len(config_files):
                return config_files[choice_num - 1]
            else:
                print(
                    f"Invalid choice. Please enter a number between 0 and {len(config_files)}."
                )

        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")


def run_config_selection(config_dir: str = "configs") -> str:
    """Run the configuration selection process"""
    try:
        config_files = find_config_files(config_dir)
        print(f"Found {len(config_files)} configuration file(s)")
        return display_config_menu(config_files)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_dir = sys.argv[1]
    else:
        config_dir = "configs"

    selected_config = run_config_selection(config_dir)
    print(f"Selected configuration: {selected_config}")
