from typing import Optional, Tuple, Dict, Any, List

import torch
from torch.optim import Optimizer

# Import from torchao
# We use the _AdamW class which corresponds to the main AdamW implementation in torchao
from torchao.optim import _AdamW as VendorAdamW

class StochasticAdamW(VendorAdamW):
    """
    AdamW optimizer with support for stochastic rounding in BF16 training.
    
    This optimizer wraps the torchao-derived implementation to provide
    stochastic rounding support, which can be beneficial for low-precision
    (BF16) training stability.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (bool, optional): whether to use the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and Beyond"
            (default: False)
        bf16_stochastic_round (bool, optional): whether to use stochastic rounding
            for BF16 parameters. (default: True)
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        *,
        bf16_stochastic_round=True,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            bf16_stochastic_round=bf16_stochastic_round,
        )
