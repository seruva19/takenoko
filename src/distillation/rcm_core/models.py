"""Common utilities for preparing teacher and student networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch.nn as nn


def _freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)
    module.eval()


def _unfreeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(True)
    module.train()


@dataclass(slots=True)
class RCMModelBundle:
    """Aggregates teacher/student networks manipulated by the runner."""

    teacher: nn.Module
    student: nn.Module
    teacher_ema: Optional[nn.Module] = None
    fake_score: Optional[nn.Module] = None
    auxiliary_modules: Iterable[nn.Module] = ()

    def freeze_teacher(self) -> None:
        _freeze_module(self.teacher)
        if self.teacher_ema is not None:
            _freeze_module(self.teacher_ema)

    def unfreeze_teacher(self) -> None:
        _unfreeze_module(self.teacher)
        if self.teacher_ema is not None:
            _unfreeze_module(self.teacher_ema)

    def train_student(self) -> None:
        _unfreeze_module(self.student)

    def eval_student(self) -> None:
        _freeze_module(self.student)


def prepare_model_bundle(
    teacher: nn.Module,
    student: nn.Module,
    *,
    teacher_ema: Optional[nn.Module] = None,
    fake_score: Optional[nn.Module] = None,
    freeze_teacher: bool = True,
    auxiliary_modules: Iterable[nn.Module] = (),
) -> RCMModelBundle:
    """Return a bundle with consistent freezing semantics."""

    bundle = RCMModelBundle(
        teacher=teacher,
        student=student,
        teacher_ema=teacher_ema,
        fake_score=fake_score,
        auxiliary_modules=tuple(auxiliary_modules),
    )

    if freeze_teacher:
        bundle.freeze_teacher()
    else:
        bundle.unfreeze_teacher()

    bundle.train_student()
    for module in bundle.auxiliary_modules:
        module.train()
    if bundle.fake_score is not None:
        bundle.fake_score.train()
    return bundle
