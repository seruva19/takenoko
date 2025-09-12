"""
Stochastic Rounding for BF16 Training

Provides stochastic rounding operations for improved BF16 training stability.
Essential for full fine-tuning of large models where gradient precision matters.
"""

try:
    # Try to import the optimized CUDA version first
    from .stochastic_ops import (
        copy_stochastic_cuda_, add_stochastic_cuda_, mul_stochastic_cuda_,
        div_stochastic_cuda_, lerp_stochastic_cuda_, addcmul_stochastic_cuda_,
        addcdiv_stochastic_cuda_
    )
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

# Always import the pure Python fallback versions
from .stochastic_ops import (
    copy_stochastic_, add_stochastic_, mul_stochastic_,
    div_stochastic_, lerp_stochastic_, addcmul_stochastic_,
    addcdiv_stochastic_
)

# Import the high-level interface
from .stoch_lib import add_, mul_, div_, lerp_, addcmul_, addcdiv_

__all__ = [
    'CUDA_AVAILABLE',
    'copy_stochastic_', 'add_stochastic_', 'mul_stochastic_',
    'div_stochastic_', 'lerp_stochastic_', 'addcmul_stochastic_',
    'addcdiv_stochastic_', 'add_', 'mul_', 'div_', 'lerp_', 'addcmul_', 'addcdiv_'
]

if CUDA_AVAILABLE:
    __all__.extend([
        'copy_stochastic_cuda_', 'add_stochastic_cuda_', 'mul_stochastic_cuda_',
        'div_stochastic_cuda_', 'lerp_stochastic_cuda_', 'addcmul_stochastic_cuda_',
        'addcdiv_stochastic_cuda_'
    ])