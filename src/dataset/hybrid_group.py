import random
from typing import Optional, Sized, Protocol

import torch


class _LenGetItem(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int): ...


class HybridDatasetGroup(torch.utils.data.Dataset):
    """A lightweight wrapper that mixes items from a main group and a correction group.

    - Non-invasive: does not alter underlying datasets
    - Probabilistic routing controlled by correction_ratio
    - Forwards epoch/step/max_step calls to both groups
    """

    def __init__(
        self,
        main_group: _LenGetItem,
        correction_group: _LenGetItem,
        correction_ratio: float = 0.2,
        rng: Optional[random.Random] = None,
    ) -> None:
        super().__init__()
        self.main_group = main_group
        self.correction_group = correction_group
        self.correction_ratio = max(0.0, min(1.0, float(correction_ratio)))
        self.rng = rng or random.Random()
        # Expose common metadata expected by trainer for logging
        try:
            self.num_train_items = getattr(self.main_group, "num_train_items")
        except Exception:
            try:
                self.num_train_items = len(self.main_group)
            except Exception:
                self.num_train_items = 0

        # Mirror dataset listing if available for log output
        self._datasets = getattr(self.main_group, "datasets", [])

    @property
    def datasets(self):  # for trainer logs
        return self._datasets

    def __len__(self) -> int:  # type: ignore[override]
        # Keep epoch length identical to main dataset to avoid altering scheduler math
        return len(self.main_group)

    def __getitem__(self, idx: int):  # type: ignore[override]
        use_correction = self.rng.random() < self.correction_ratio
        if use_correction and len(self.correction_group) > 0:
            # Map index into correction range to keep stochasticity but avoid OOB
            mapped_idx = idx % len(self.correction_group)
            return self.correction_group[mapped_idx]
        mapped_idx = idx % len(self.main_group)
        return self.main_group[mapped_idx]

    # Forwarders for Takenoko dataset lifecycle hooks
    def set_current_epoch(self, epoch, force_shuffle=None, reason=None):
        """
        Set current epoch for all datasets in the hybrid group.

        Note: With shared_epoch approach, actual shuffling happens in __getitem__.
        Parameters forwarded for backward compatibility.
        """
        for ds in (self.main_group, self.correction_group):
            try:
                ds.set_current_epoch(epoch, force_shuffle=force_shuffle, reason=reason)  # type: ignore[attr-defined]
            except Exception:
                pass

    def set_current_step(self, step):
        for ds in (self.main_group, self.correction_group):
            try:
                ds.set_current_step(step)  # type: ignore[attr-defined]
            except Exception:
                pass

    def set_max_train_steps(self, max_train_steps):
        for ds in (self.main_group, self.correction_group):
            try:
                ds.set_max_train_steps(max_train_steps)  # type: ignore[attr-defined]
            except Exception:
                pass
