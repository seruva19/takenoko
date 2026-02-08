"""
PolyLoRA-specific interactive menu shown when enable_polylora is true.
Guides the user through prerequisite steps before running PolyLoRA training.
"""

from .menu_base import BaseMenu


class PolyLoRAMenu(BaseMenu):
    """Menu for PolyLoRA side-pipeline steps."""

    def __init__(self, trainer):
        super().__init__("Takenoko - PolyLoRA Pipeline")
        self.trainer = trainer
        self._setup_menu_items()

    def _setup_menu_items(self) -> None:
        self.add_item("1", "Collect target spec from LoRA checkpoints", self.trainer.polylora_controller.collect_spec)
        self.add_item("2", "Build embedding/LoRA pair dataset (identity/base optional)", self.trainer.polylora_controller.build_pairs)
        self.add_item("3", "Train PolyLoRA network (identity/perceiver/base flags respected)", self.trainer.polylora_controller.train_network)
        self.add_item("4", "Predict LoRA for sanity check", self.trainer.polylora_controller.predict_sample)
        self.add_item("5", "QA LoRA corpus (spec + frames)", self.trainer.polylora_controller.qa_corpus)
        self.add_item("0", "Return to Config Selection", self._exit_to_config_selection)

    def _exit_to_config_selection(self) -> bool:
        """Exit with code 100 to reuse outer config loop."""
        import sys

        print("\nReturning to config selection...")
        sys.exit(100)


def create_polylora_menu(trainer) -> PolyLoRAMenu:
    """Factory to build the PolyLoRA menu."""
    return PolyLoRAMenu(trainer)
