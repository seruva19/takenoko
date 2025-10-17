"""
Main Launcher - Entry point that combines config selection and operations menu
"""

import sys
import os
from .config_selector import run_config_selection
from ..takenoko import UnifiedTrainer


def main():
    """Main launcher function"""
    # Add src directory to Python path if not already there
    if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    try:
        # Run config selection menu
        config_path = run_config_selection()

        # Create and run the unified trainer with selected config
        trainer = UnifiedTrainer(config_path)
        trainer.run()

    except SystemExit:
        # Let SystemExit exceptions bubble up (used for menu navigation)
        raise
    except Exception as e:
        print(f"💥 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()