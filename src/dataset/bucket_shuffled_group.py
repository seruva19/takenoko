import random
from typing import Any, Optional, Sequence

import torch


class BucketShuffledDatasetGroup(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets: Sequence[torch.utils.data.Dataset],
        *,
        seed: int = 0,
        shared_epoch: Any = None,
    ) -> None:
        super().__init__()
        self._datasets = list(datasets)
        self.seed = int(seed)
        self.shared_epoch = shared_epoch
        self.current_epoch = 0

        # (dataset_index, dataset_batch_index)
        self._global_order: list[tuple[int, int]] = []
        self._last_epoch_built: Optional[int] = None

        self.num_train_items = 0
        for ds in self._datasets:
            self.num_train_items += int(getattr(ds, "num_train_items", 0) or 0)

        self._rebuild_global_order(force=True)

    @property
    def datasets(self):
        return getattr(self, "_datasets", [])

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._global_order)

    def __getitem__(self, idx: int):  # type: ignore[override]
        self._check_and_rebuild_on_epoch_change()
        ds_idx, local_idx = self._global_order[idx]
        return self._datasets[ds_idx][local_idx]

    def set_current_epoch(self, epoch, force_shuffle=None, reason=None):
        try:
            epoch_int = int(epoch)
        except Exception:
            epoch_int = 0
        self.current_epoch = epoch_int
        self._check_and_rebuild_on_epoch_change()

        for ds in self._datasets:
            try:
                ds.set_current_epoch(epoch, force_shuffle=force_shuffle, reason=reason)  # type: ignore[attr-defined]
            except Exception:
                pass

    def set_current_step(self, step):
        for ds in self._datasets:
            try:
                ds.set_current_step(step)  # type: ignore[attr-defined]
            except Exception:
                pass

    def set_max_train_steps(self, max_train_steps):
        for ds in self._datasets:
            try:
                ds.set_max_train_steps(max_train_steps)  # type: ignore[attr-defined]
            except Exception:
                pass

    def validate_resume_compatibility(
        self, expected_dataset_info: Optional[dict] = None
    ) -> bool:
        main_group = self._datasets[0] if self._datasets else None
        fn = getattr(main_group, "validate_resume_compatibility", None)
        if callable(fn):
            try:
                return bool(fn(expected_dataset_info))
            except Exception:
                return True
        return True

    def get_dataset_info_for_resume(self) -> dict:
        main_group = self._datasets[0] if self._datasets else None
        fn = getattr(main_group, "get_dataset_info_for_resume", None)
        if callable(fn):
            try:
                return dict(fn())
            except Exception:
                return {}
        return {}

    def log_dataset_compatibility_info(self) -> None:
        main_group = self._datasets[0] if self._datasets else None
        fn = getattr(main_group, "log_dataset_compatibility_info", None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass

    def _check_and_rebuild_on_epoch_change(self) -> None:
        epoch = self.current_epoch
        if self.shared_epoch is not None:
            try:
                epoch = int(self.shared_epoch.value)
            except Exception:
                pass

        if self._last_epoch_built == epoch:
            return

        self.current_epoch = epoch

        for ds in self._datasets:
            check_fn = getattr(ds, "_check_and_shuffle_on_epoch_change", None)
            if callable(check_fn):
                try:
                    check_fn()
                except Exception:
                    pass

        self._rebuild_global_order(force=True)

    def _rebuild_global_order(self, *, force: bool = False) -> None:
        epoch = self.current_epoch
        if not force and self._last_epoch_built == epoch:
            return

        rng = random.Random(self.seed + epoch * 10007)

        bucket_to_indices: dict[Any, list[tuple[int, int]]] = {}

        for ds_idx, ds in enumerate(self._datasets):
            bm = getattr(ds, "batch_manager", None)
            bbi = getattr(bm, "bucket_batch_indices", None)
            if bm is None or bbi is None:
                raise ValueError(
                    "BucketShuffledDatasetGroup requires datasets with a BucketBatchManager "
                    "(prepare_for_training must have been run)."
                )

            for local_idx, (bucket_key, _batch_idx) in enumerate(bbi):
                bucket_to_indices.setdefault(bucket_key, []).append((ds_idx, local_idx))

        bucket_keys = list(bucket_to_indices.keys())
        rng.shuffle(bucket_keys)
        for key in bucket_keys:
            rng.shuffle(bucket_to_indices[key])

        global_order: list[tuple[int, int]] = []
        for key in bucket_keys:
            global_order.extend(bucket_to_indices[key])

        self._global_order = global_order
        self._last_epoch_built = epoch
