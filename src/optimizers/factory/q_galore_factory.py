"""Q-GaLore optimizer creation helpers for WAN network trainer."""
from typing import Any, Dict, List, Tuple

import torch


def create_q_galore_adamw8bit_optimizer(
    trainable_params: List[Any],
    lr: float,
    optimizer_kwargs: Dict[str, Any],
    logger: Any,
) -> Tuple[Any, torch.optim.Optimizer]:
    logger.info(f"using QGaLoreAdamW8bit optimizer | {optimizer_kwargs}")
    try:
        from vendor.q_galore_torch.q_galore_adamw8bit import (
            AdamW8bit as QGaLoreAdamW8bit,
        )
    except Exception as err:
        try:
            from q_galore_torch import QGaLoreAdamW8bit
        except Exception as err2:
            raise ImportError(
                "QGaLoreAdamW8bit requires q-galore-torch and bitsandbytes. "
                "Install with `pip install q-galore`."
            ) from err2
    optimizer_class = QGaLoreAdamW8bit
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
    return optimizer_class, optimizer
