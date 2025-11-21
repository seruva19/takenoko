"""
Takenoko specific menu implementation
"""

import sys
import logging
from typing import Dict, Any, Optional, List
from .menu_base import BaseMenu, MenuItem


class OperationsMenu(BaseMenu):
    """Main menu for operations"""

    def __init__(self, trainer):
        super().__init__("Takenoko - Unified Operations Menu")
        self.trainer = trainer
        self._setup_menu_items()

    def _setup_menu_items(self):
        """Setup all menu items"""
        self.add_item("1", "Cache Latents", self.trainer.cache_latents)
        self.add_item(
            "2", "Cache Text Encoder Outputs", self.trainer.cache_text_encoder_outputs
        )
        self.add_item("3", "Train Model", self.trainer.train_model)
        self.add_item("4", "Estimate VRAM Usage", self._estimate_vram)
        self.add_item("5", "Estimate latent cache chunks", self._estimate_cache_chunks)
        self.add_item("6", "Analyze Dataset Buckets", self._analyze_buckets)
        self.add_item("7", "Reload Config File", self.trainer.reload_config)
        self.add_item("8", "Free VRAM", self.trainer.free_vram_aggressively)
        self.add_item("9", "Memory Diagnostics", self.show_memory_diagnostics)
        self.add_item("0", "Return to Config Selection", self._exit_to_config_selection)

    def _estimate_vram(self) -> bool:
        """Estimate VRAM usage"""
        try:
            import sys
            import os

            sys.path.insert(
                0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            from common.vram_estimator import estimate_and_log_vram

            # Use print() for better visibility in menu context
            print(f"\n🔍 Estimating VRAM usage for current configuration...")

            # Create a simple logger class that prints instead of logging
            class SimpleLogger:
                def info(self, msg, *args):
                    if args:
                        print(f"   {msg % args}")
                    else:
                        print(f"   {msg}")

                def error(self, msg, *args):
                    if args:
                        print(f"   ❌ {msg % args}")
                    else:
                        print(f"   ❌ {msg}")

                def warning(self, msg, *args):
                    if args:
                        print(f"   ⚠️ {msg % args}")
                    else:
                        print(f"   ⚠️ {msg}")

            gb, details = estimate_and_log_vram(self.trainer.config, SimpleLogger())

            print(f"\n📊 Summary:")
            print(f"   Total estimated VRAM: {gb:.2f} GB")

            return True
        except Exception as e:
            print(f"❌ Error estimating VRAM usage: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _estimate_cache_chunks(self) -> bool:
        """Estimate latent cache chunks"""
        try:
            from caching.chunk_estimator import (
                estimate_latent_cache_chunks,
                estimate_latent_cache_chunks_per_dataset,
            )

            per_ds = estimate_latent_cache_chunks_per_dataset(
                self.trainer.args.dataset_config, self.trainer.args
            )
            total_chunks = sum(entry["chunks"] for entry in per_ds)
            total_effective = sum(
                entry.get("effective_chunks", entry["chunks"]) for entry in per_ds
            )

            # Use print() instead of logging.info() for better visibility in menu context
            print(f"\n🧮 Estimated latent cache chunks (cache workload): {total_chunks}")
            if total_effective != total_chunks:
                print(
                    f"🌀 Estimated per-epoch video batches (epoch_slide applied): {total_effective}"
                )
            else:
                print(f"🌀 Estimated per-epoch video batches: {total_effective}")

            # Show each dataset with distinguishing information
            for idx, entry in enumerate(per_ds, 1):
                vdir = entry["video_directory"]
                chunks = entry["chunks"]
                effective = entry.get("effective_chunks", chunks)
                caption_ext = entry.get("caption_extension", ".txt")
                cache_dir = entry.get("latents_cache_dir")

                # Build info string with distinguishing details
                info_parts = [f"{chunks} cache"]
                if effective != chunks:
                    cycle = entry.get("epoch_cycle_max")
                    per_epoch = f"{effective} per-epoch"
                    if cycle:
                        per_epoch += f", cycle≤{cycle} epochs"
                    info_parts.append(per_epoch)
                if caption_ext and caption_ext != ".txt":
                    info_parts.append(f"captions={caption_ext}")
                if cache_dir:
                    # Show just the last part of cache path for brevity
                    cache_name = cache_dir.split("/")[-1].split("\\")[-1]
                    info_parts.append(f"cache={cache_name}")

                info_str = ", ".join(info_parts)
                print(f"   [{idx}] {vdir}: {info_str}")

            return True
        except Exception as e:
            print(f"❌ Error estimating cache chunks: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _analyze_buckets(self) -> bool:
        """Analyze bucket distribution for datasets"""
        try:
            from menu.bucket_analyzer import analyze_dataset_buckets

            print(f"\n📊 Analyzing dataset bucket distribution...")
            print(f"{'='*80}")

            analyze_dataset_buckets(self.trainer.config, self.trainer.args)

            return True
        except Exception as e:
            print(f"❌ Error analyzing buckets: {e}")
            import traceback

            traceback.print_exc()
            return False

    def show_memory_diagnostics(self) -> bool:
        """Display comprehensive memory diagnostics."""
        try:
            import sys
            import os

            sys.path.insert(
                0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            from utils.memory_tracking_manager import (
                show_memory_diagnostics as show_memory_diagnostics_func,
            )

            print(f"\n🧠 Memory Diagnostics")
            print(f"{'='*50}")
            show_memory_diagnostics_func()
            return True
        except Exception as e:
            print(f"❌ Error during memory diagnostics: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _exit_to_config_selection(self) -> bool:
        """Exit to config selection menu"""
        logging.info("Returning to config selection menu...")
        sys.exit(100)


class ConfigSelectionMenu(BaseMenu):
    """Menu for selecting configuration files"""

    def __init__(self):
        super().__init__("Takenoko - Configuration Selection")
        self.selected_config = None

    def setup_from_configs(self, config_files: List[str]):
        """Setup menu items from list of config files"""
        self.add_item("0", "Quit", self._quit)

        for i, config_file in enumerate(config_files, 1):
            display_name = config_file.replace("\\", "/").split("/")[
                -1
            ]  # Get just filename
            self.add_item(
                str(i), display_name, lambda f=config_file: self._select_config(f)
            )

    def _select_config(self, config_file: str) -> bool:
        """Select a configuration file"""
        self.selected_config = config_file
        return True

    def _quit(self) -> bool:
        """Quit the application"""
        sys.exit(0)

    def get_selected_config(self) -> Optional[str]:
        """Get the selected configuration file"""
        return self.selected_config


def create_operations_menu(trainer) -> OperationsMenu:
    """Factory function to create the main operations menu"""
    return OperationsMenu(trainer)


def create_config_menu_from_directory(
    config_dir: str = "configs",
) -> ConfigSelectionMenu:
    """Factory function to create config selection menu from directory"""
    import os
    import glob

    config_files = glob.glob(os.path.join(config_dir, "*.toml"))
    config_files = [f.replace("\\", "/") for f in config_files if os.path.exists(f)]

    if not config_files:
        raise ValueError(f"No TOML configuration files found in {config_dir}")

    menu = ConfigSelectionMenu()
    menu.setup_from_configs(config_files)
    return menu
