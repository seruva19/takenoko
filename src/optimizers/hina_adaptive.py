import torch
import torch.nn as nn
from typing import Any, Dict, List, Tuple, Optional
import math
import array
import random
import concurrent.futures
from threading import Thread
from collections import defaultdict
import time

import logging
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class MemoryMonitor:
    """Dynamic memory monitor"""

    def __init__(self, target_vram_gb: float = 16):
        self.target_vram = target_vram_gb * 1024**3
        self.current_usage = 0

    def check_memory_pressure(self) -> float:
        """Check memory pressure ratio"""
        if torch.cuda.is_available():
            current_allocated = torch.cuda.memory_allocated()
            pressure_ratio = current_allocated / self.target_vram
            return pressure_ratio
        return 0.0

    def suggest_optimizations(self, pressure_ratio: float) -> Dict[str, bool]:
        """Suggest optimization strategies based on memory pressure"""
        if pressure_ratio > 0.9:
            return {
                "reduce_buffer_pool": True,
                "increase_gc_frequency": True,
                "use_checkpoint_offload": True,
                "reduce_precision": True,
            }
        elif pressure_ratio > 0.7:
            return {
                "reduce_buffer_pool": True,
                "increase_gc_frequency": False,
                "use_checkpoint_offload": False,
                "reduce_precision": False,
            }
        return {
            "reduce_buffer_pool": False,
            "increase_gc_frequency": False,
            "use_checkpoint_offload": False,
            "reduce_precision": False,
        }


class EnhancedBufferPool:
    """Enhanced buffer pool for intelligent memory management"""

    def __init__(self, max_total_memory_mb: int = 500):
        self._buffer_pool = {}
        self._usage_stats = defaultdict(int)
        self._max_total_memory = max_total_memory_mb * 1024 * 1024
        self._current_memory = 0

    def get_buffer_with_priority(
        self,
        shape: Tuple,
        dtype: torch.dtype,
        device: torch.device,
        priority: str = "normal",
    ) -> torch.Tensor:
        """Get buffer based on priority"""
        key = (shape, dtype, device)
        self._usage_stats[key] += 1

        if key in self._buffer_pool and self._buffer_pool[key]:
            return self._buffer_pool[key].pop()

        # Check memory budget
        tensor_size = torch.prod(torch.tensor(shape)).item() * self._get_dtype_size(
            dtype
        )
        if (
            self._current_memory + tensor_size > self._max_total_memory
            and priority != "critical"
        ):
            # Not enough memory and not critical priority, return new tensor (do not add to pool)
            return torch.empty(shape, dtype=dtype, device=device)

        return torch.empty(shape, dtype=dtype, device=device)

    def return_buffer(self, tensor: torch.Tensor, priority: str = "normal"):
        """Return buffer to pool"""
        key = (tuple(tensor.shape), tensor.dtype, tensor.device)

        if key not in self._buffer_pool:
            self._buffer_pool[key] = []

        # Decide whether to keep based on usage frequency
        usage_freq = self._usage_stats.get(key, 0)
        max_buffers = max(
            1, min(3, usage_freq // 10)
        )  # Dynamically adjust buffer count

        if len(self._buffer_pool[key]) < max_buffers:
            tensor.zero_()
            self._buffer_pool[key].append(tensor)
            tensor_size = torch.prod(
                torch.tensor(tensor.shape)
            ).item() * self._get_dtype_size(tensor.dtype)
            self._current_memory += tensor_size

    def smart_cleanup(self, memory_pressure: float):
        """Smartly clean up buffer pool"""
        if memory_pressure > 0.8:
            # Clean up buffers with low usage frequency
            keys_to_clean = sorted(
                self._usage_stats.keys(), key=lambda k: self._usage_stats[k]
            )[: len(self._usage_stats) // 2]
            for key in keys_to_clean:
                if key in self._buffer_pool:
                    del self._buffer_pool[key]
            self._current_memory = 0  # Reset counter

    @staticmethod
    def _get_dtype_size(dtype: torch.dtype) -> int:
        """Get data type size"""
        size_map = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.int32: 4,
            torch.int16: 2,
            torch.int8: 1,
            torch.bool: 1,
        }
        return size_map.get(dtype, 4)


class CompactStateDict:
    """Compact state storage"""

    __slots__ = ["tensor_states", "scalar_states", "bool_states", "string_states"]

    def __init__(self):
        self.tensor_states = {}
        self.scalar_states = {}
        self.bool_states = {}
        self.string_states = {}

    def set_tensor(
        self, key: str, value: torch.Tensor, use_half_precision: bool = False
    ):
        """Set tensor state, optionally using half precision"""
        if use_half_precision and value.dtype == torch.float32:
            value = value.to(torch.bfloat16)
        self.tensor_states[key] = value

    def get_tensor(
        self,
        key: str,
        target_dtype: torch.dtype = None,  # type: ignore
        target_device: torch.device = None,  # type: ignore
    ) -> torch.Tensor:
        """Get tensor state, optionally converting to target dtype and device"""
        tensor = self.tensor_states.get(key)
        if tensor is not None:
            if target_dtype is not None and tensor.dtype != target_dtype:
                tensor = tensor.to(target_dtype)
            if target_device is not None and tensor.device != target_device:
                tensor = tensor.to(target_device)
        return tensor  # type: ignore

    def set_scalar(self, key: str, value: float):
        """Set scalar state"""
        self.scalar_states[key] = value

    def get_scalar(self, key: str, default: float = 0.0) -> float:
        """Get scalar state"""
        return self.scalar_states.get(key, default)


class CompressedRelationships:
    """Compressed parameter relationship storage"""

    def __init__(self):
        self.param_pairs = []
        self.compatibility_scores = torch.tensor([], dtype=torch.float16)
        self.interaction_types = []
        self._type_pool = {
            "matmul_12": 0,
            "matmul_21": 1,
            "matmul_12t": 2,
            "matmul_1t2": 3,
            "norm_based": 4,
        }
        self._reverse_type_pool = {v: k for k, v in self._type_pool.items()}

    def add_relationship(
        self,
        param1_id: int,
        param2_id: int,
        compatibility: float,
        interaction_type: str,
    ):
        """Add parameter relationship"""
        self.param_pairs.append((param1_id, param2_id))

        # Expand compatibility score tensor
        new_score = torch.tensor([compatibility], dtype=torch.float16)
        self.compatibility_scores = torch.cat([self.compatibility_scores, new_score])

        # Use type pool
        type_id = self._type_pool.get(interaction_type, 4)
        self.interaction_types.append(type_id)

    def get_relationship(self, param1_id: int) -> Optional[Dict]:
        """Get parameter relationship"""
        for i, (p1_id, p2_id) in enumerate(self.param_pairs):
            if p1_id == param1_id:
                return {
                    "partner_id": p2_id,
                    "compatibility": self.compatibility_scores[i].item(),
                    "interaction_type": self._reverse_type_pool[
                        self.interaction_types[i]
                    ],
                }
        return None


@torch.jit.script
def quantize_importance_score(score: float) -> int:
    """Quantize importance score to int16"""
    return int(torch.clamp(torch.round(torch.tensor(score * 6553.5)), 0, 65535).item())


@torch.jit.script
def dequantize_importance_score(quantized: int) -> float:
    """Dequantize importance score"""
    return float(quantized) / 6553.5


@torch.jit.script
def compute_lr_mask_update_core(
    lr_mask: torch.Tensor,
    sign_agree: torch.Tensor,
    lr_bump: float,
    min_lr: float,
    max_lr: float,
) -> torch.Tensor:
    """JIT-compiled core logic for lr_mask update"""
    lr_adjustment = torch.where(sign_agree > 0, lr_bump, -lr_bump)
    new_lr_mask = lr_mask + lr_adjustment
    return torch.clamp(new_lr_mask, min=min_lr, max=max_lr)


@torch.jit.script
def orthogonal_gradient_core_optimized(
    grad_flat: torch.Tensor, param_flat: torch.Tensor, eps: float
) -> torch.Tensor:
    """Optimized orthogonal gradient projection core computation"""
    grad_norm = torch.norm(grad_flat, p=2)
    if grad_norm <= eps:
        return grad_flat

    dot_product = torch.dot(param_flat, grad_flat)
    param_norm_sq = torch.dot(param_flat, param_flat) + eps
    proj_coeff = dot_product / param_norm_sq

    orthogonal_grad_flat = grad_flat - proj_coeff * param_flat
    orth_norm = torch.norm(orthogonal_grad_flat, p=2) + eps
    scale_factor = grad_norm / orth_norm

    return orthogonal_grad_flat * scale_factor


class AsyncComputeManager:
    """Asynchronous computation manager"""

    def __init__(self, max_workers: int = 2):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.pending_futures = []

    def submit_async_task(self, func, *args, **kwargs):
        """Submit asynchronous task"""
        future = self.executor.submit(func, *args, **kwargs)
        self.pending_futures.append(future)
        return future

    def collect_completed_tasks(self, timeout: float = 0.001):
        """Collect completed tasks"""
        completed = []
        remaining = []

        for future in self.pending_futures:
            if future.done():
                try:
                    result = future.result(timeout=timeout)
                    completed.append(result)
                except concurrent.futures.TimeoutError:
                    remaining.append(future)
                except Exception as e:
                    logger.warning(f"Asynchronous task execution failed: {e}")
                    remaining.append(future)
            else:
                remaining.append(future)

        self.pending_futures = remaining
        return completed

    def shutdown(self):
        """Shutdown executor"""
        self.executor.shutdown(wait=True)


class HinaAdaptive(torch.optim.Optimizer):
    """
    Memory-optimized version of the adaptive HinaAdaptive optimizer

    Main optimization features:
    1. Precision grading: critical states keep high precision, secondary states use low precision
    2. Intelligent buffer pool: dynamic memory management
    3. Compressed state storage: reduce Python object overhead
    4. Asynchronous computation: non-critical computations run asynchronously
    5. Adaptive memory management: adjust strategies based on memory pressure
    6. Edge overfitting control:
       - Edge suppression: detect and suppress edge gradients to prevent edge overfitting
       - Frequency awareness: control high-frequency noise for training stability
       - Spatial awareness: regularization based on spatial variance
       - LoRA low-rank regularization: rank penalty mechanism for LoRA layers
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        optim_bits: int = 32,
        args: Any = None,
        percentile_clipping: int = 100,
        block_wise: bool = True,
        is_paged: bool = False,
        # Enhanced features configuration
        use_spd: bool = True,
        spd_lambda: float = 0.06,
        use_cautious: bool = True,
        use_orthogonal_grad: bool = False,
        use_adopt_stability: bool = True,
        use_grams: bool = True,
        use_agr: bool = True,
        use_tam: bool = True,
        tam_beta: float = 0.999,
        # Dynamic adaptive learning rate functionality
        use_dynamic_adaptation: bool = True,
        adaptation_strength: float = 1.0,
        relationship_discovery_interval: int = 100,
        importance_decay: float = 0.95,
        compatibility_threshold: float = 0.3,
        # lr_mask mechanism configuration
        use_lr_mask: bool = True,
        lr_bump: float = 3e-6,
        min_lr: float = 1e-7,
        max_lr: float = 1e-3,
        warmup_steps: int = 500,
        # Dynamic weight decay configuration
        dynamic_weight_decay: bool = True,
        wd_transition_steps: int = 1000,
        wd_decay_factor: float = 0.7,
        wd_min_ratio: float = 0.1,
        # Memory optimization configuration
        memory_efficient: bool = True,
        vram_budget_gb: float = 16.0,
        cpu_offload_states: bool = True,
        reduce_precision: bool = True,
        adaptive_features: bool = True,
        emergency_simplify: bool = True,
        max_buffer_memory_mb: int = 500,
        # Edge overfitting control parameters
        edge_suppression: bool = False,
        edge_penalty: float = 0.1,
        # Spatial awareness
        spatial_awareness: bool = False,
        frequency_penalty: float = 0.05,
        detail_preservation: float = 0.8,
        edge_threshold: float = 0.6,
        # LoRA low-rank regularization
        lora_rank_penalty: bool = False,
        rank_penalty_strength: float = 0.01,
        low_rank_emphasis: float = 1.2,
        **kwargs,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            optim_bits=optim_bits,
            args=args,
            percentile_clipping=percentile_clipping,
            block_wise=block_wise,
            is_paged=is_paged,
        )

        super().__init__(params, defaults)

        # Original feature switches
        self.use_spd = use_spd
        self.spd_lambda = spd_lambda
        self.use_cautious = use_cautious
        self.use_orthogonal_grad = use_orthogonal_grad
        self.use_adopt_stability = use_adopt_stability
        self.use_grams = use_grams
        self.use_agr = use_agr
        self.use_tam = use_tam
        self.tam_beta = tam_beta

        # Dynamic adaptive functionality configuration
        self.use_dynamic_adaptation = use_dynamic_adaptation
        self.adaptation_strength = adaptation_strength
        self.relationship_discovery_interval = relationship_discovery_interval
        self.importance_decay = importance_decay
        self.compatibility_threshold = compatibility_threshold

        # lr_mask mechanism configuration
        self.use_lr_mask = use_lr_mask
        self.lr_bump = lr_bump
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps

        # Dynamic weight decay configuration
        self.dynamic_weight_decay = dynamic_weight_decay
        self.wd_transition_steps = wd_transition_steps
        self.wd_decay_factor = wd_decay_factor
        self.wd_min_ratio = wd_min_ratio

        # Memory optimization configuration
        self.memory_efficient = memory_efficient
        self.vram_budget_gb = vram_budget_gb
        self.cpu_offload_states = cpu_offload_states
        self.reduce_precision = reduce_precision
        self.adaptive_features = adaptive_features
        self.emergency_simplify = emergency_simplify

        # Edge overfitting control configuration
        self.edge_suppression = edge_suppression
        self.edge_penalty = edge_penalty
        self.spatial_awareness = spatial_awareness
        self.frequency_penalty = frequency_penalty
        self.detail_preservation = detail_preservation
        self.edge_threshold = edge_threshold
        self.lora_rank_penalty = lora_rank_penalty
        self.rank_penalty_strength = rank_penalty_strength
        self.low_rank_emphasis = low_rank_emphasis

        # Initialize memory management components
        self.memory_monitor = MemoryMonitor(vram_budget_gb)
        self.buffer_pool = EnhancedBufferPool(max_buffer_memory_mb)
        self.async_manager = AsyncComputeManager()

        # Initialize edge overfitting control components
        if self.edge_suppression:
            self.edge_cache = {}  # Edge computation cache

        # Compressed state storage
        self.compressed_relationships = CompressedRelationships()
        self.quantized_importance_scores = {}  # param_id -> int16
        self.last_relationship_update = 0

        # Initialize parameter group metadata
        self._initialize_adaptive_metadata()

        # Store initial parameters (for SPD)
        if self.use_spd:
            self._store_initial_parameters()

        # Enable PyTorch optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

        logger.info(f"HinaAdaptive initialized, VRAM budget: {vram_budget_gb}GB")

    def _initialize_adaptive_metadata(self):
        """Initialize adaptive version metadata (memory optimized version)"""
        self.param_groups_metadata = {}

        for group_idx, group in enumerate(self.param_groups):
            self.param_groups_metadata[group_idx] = {
                "param_count": len(group["params"]),
                "param_list": list(group["params"]),
                "compact_states": {},  # Use compact state storage
            }

            # Initialize tracking information for each parameter
            for param in group["params"]:
                param_id = id(param)

                # Use quantized importance score
                self.quantized_importance_scores[param_id] = quantize_importance_score(
                    1.0
                )

                # Use compact state dictionary
                compact_state = CompactStateDict()
                compact_state.set_scalar("initial_norm", 0.0)
                compact_state.set_scalar("change_rate", 0.0)
                compact_state.set_scalar("stability", 1.0)

                # lr_mask state initialization (using half precision)
                if self.use_lr_mask:
                    device = param.device if hasattr(param, "device") else "cuda"
                    shape = param.shape

                    if self.reduce_precision:
                        lr_mask = (
                            torch.ones(shape, device=device, dtype=torch.bfloat16)
                            * self.defaults["lr"]
                        )
                    else:
                        lr_mask = (
                            torch.ones(shape, device=device, dtype=torch.float32)
                            * self.defaults["lr"]
                        )

                    compact_state.set_tensor(
                        "lr_mask", lr_mask, use_half_precision=self.reduce_precision
                    )
                    compact_state.set_tensor(
                        "last_polarity",
                        torch.zeros(shape, dtype=torch.bool, device=device),
                    )
                    compact_state.set_scalar("lr_max", self.defaults["lr"])
                    compact_state.set_scalar("avg_lr", self.defaults["lr"])
                    compact_state.set_scalar(
                        "warmup_complete", 0.0
                    )  # 0.0 = False, 1.0 = True

                # Edge overfitting control state initialization
                if self.edge_suppression:
                    device = param.device if hasattr(param, "device") else "cuda"
                    shape = param.shape

                    # Edge history tracking
                    compact_state.set_tensor(
                        "edge_history",
                        torch.zeros(shape, device=device, dtype=torch.float32),
                    )
                    compact_state.set_tensor(
                        "edge_momentum",
                        torch.zeros(shape, device=device, dtype=torch.float32),
                    )
                    compact_state.set_scalar("edge_strength", 0.0)

                if self.spatial_awareness:
                    device = param.device if hasattr(param, "device") else "cuda"
                    shape = param.shape

                    # Spatial awareness state
                    compact_state.set_tensor(
                        "spatial_variance",
                        torch.ones(shape, device=device, dtype=torch.float32),
                    )
                    compact_state.set_tensor(
                        "detail_tracker",
                        torch.zeros(shape, device=device, dtype=torch.float32),
                    )
                    compact_state.set_scalar("spatial_activity", 0.0)

                if self.lora_rank_penalty and len(param.shape) == 2:
                    device = param.device if hasattr(param, "device") else "cuda"
                    min_dim = min(param.shape)

                    # LoRA low-rank tracking
                    compact_state.set_tensor(
                        "rank_tracker",
                        torch.zeros(min_dim, device=device, dtype=torch.float32),
                    )
                    compact_state.set_scalar("rank_penalty_history", 0.0)

                self.param_groups_metadata[group_idx]["compact_states"][
                    param_id
                ] = compact_state

    def _store_initial_parameters(self):
        """Store initial parameters for SPD (memory optimized version)"""
        self.initial_params = {}
        for group_idx, group in enumerate(self.param_groups):
            for param in group["params"]:
                if param.requires_grad:
                    if self.cpu_offload_states:
                        # Store initial parameters on CPU to save VRAM
                        self.initial_params[param] = param.data.clone().detach().cpu()
                    else:
                        self.initial_params[param] = param.data.clone().detach()

    def _get_optimized_buffer(
        self,
        shape: Tuple,
        dtype: torch.dtype,
        device: torch.device,
        priority: str = "normal",
    ) -> torch.Tensor:
        """Get optimized buffer"""
        return self.buffer_pool.get_buffer_with_priority(shape, dtype, device, priority)

    def _return_optimized_buffer(self, tensor: torch.Tensor, priority: str = "normal"):
        """Return optimized buffer"""
        self.buffer_pool.return_buffer(tensor, priority)

    def _check_memory_and_adapt(self):
        """Check memory and adapt"""
        if not self.memory_efficient:
            return

        memory_pressure = self.memory_monitor.check_memory_pressure()
        optimizations = self.memory_monitor.suggest_optimizations(memory_pressure)

        if optimizations["reduce_buffer_pool"]:
            self.buffer_pool.smart_cleanup(memory_pressure)

        if optimizations["increase_gc_frequency"]:
            torch.cuda.empty_cache()

        if optimizations["reduce_precision"] and not self.reduce_precision:
            logger.warning(
                "Memory pressure too high, consider enabling reduce_precision=True"
            )

        if optimizations["use_checkpoint_offload"] and hasattr(
            self, "_enable_emergency_offload"
        ):
            self._enable_emergency_offload()  # type: ignore

    def _compute_parameter_contribution_score_optimized(
        self, param, compact_state, param_id
    ):
        """
        Optimized parameter contribution score calculation
        Use batch operations and reduce memory allocation
        """
        # 1. Gradient-related contribution analysis
        grad_contribution = 0.0
        if param.grad is not None:
            current_grad_norm = torch.norm(param.grad).item()
            grad_contribution = current_grad_norm

        # 2. Parameter change-related contribution analysis
        change_contribution = 0.0
        initial_norm = compact_state.get_scalar("initial_norm")

        if initial_norm == 0.0:
            # First record
            initial_norm = torch.norm(param.data).item()
            compact_state.set_scalar("initial_norm", initial_norm)
        else:
            # Calculate relative change rate
            current_norm = torch.norm(param.data).item()
            if initial_norm > 0:
                change_rate = abs(current_norm - initial_norm) / initial_norm
                compact_state.set_scalar("change_rate", change_rate)
                change_contribution = change_rate

        # 3. Parameter intrinsic characteristic analysis (sample calculation to save time)
        if random.random() < 0.1:  # 10% probability for full calculation
            param_variance = torch.var(param.data).item()
            param_sparsity = (param.data.abs() < 1e-6).float().mean().item()
            intrinsic_contribution = param_variance * (1.0 - param_sparsity)
        else:
            # Use quick estimation
            intrinsic_contribution = 0.5

        # Overall contribution score
        total_contribution = (
            grad_contribution * 0.4
            + change_contribution * 0.3
            + intrinsic_contribution * 0.3
        )

        return max(0.01, total_contribution)

    def _discover_parameter_relationships_async(self, group_metadata):
        """Asynchronously discover parameter relationships"""

        def discover_relationships():
            param_list = group_metadata["param_list"]
            new_relationships = []

            # Limit the number of parameter pairs to control computation
            max_pairs = min(100, len(param_list) * (len(param_list) - 1) // 2)
            pair_count = 0

            for i, param1 in enumerate(param_list):
                if pair_count >= max_pairs:
                    break
                if param1.dim() != 2:
                    continue

                for j, param2 in enumerate(param_list[i + 1 :], i + 1):
                    if pair_count >= max_pairs:
                        break
                    if param2.dim() != 2:
                        continue

                    compatibility = self._compute_parameter_compatibility_fast(
                        param1, param2
                    )
                    if compatibility > self.compatibility_threshold:
                        param1_id = id(param1)
                        param2_id = id(param2)
                        interaction_type = self._determine_interaction_type_fast(
                            param1, param2
                        )

                        new_relationships.append(
                            {
                                "param1_id": param1_id,
                                "param2_id": param2_id,
                                "compatibility": compatibility,
                                "interaction_type": interaction_type,
                            }
                        )
                        pair_count += 1

            return new_relationships

        # Submit asynchronous task
        return self.async_manager.submit_async_task(discover_relationships)

    def _compute_parameter_compatibility_fast(self, param1, param2):
        """Fast version of parameter compatibility calculation"""
        if param1.dim() != 2 or param2.dim() != 2:
            return 0.0

        shape1, shape2 = param1.shape, param2.shape

        # Check matrix multiplication possibility
        multiplication_checks = [
            shape1[1] == shape2[0],
            shape1[0] == shape2[1],
            shape1[1] == shape2[1],
            shape1[0] == shape2[0],
        ]

        if not any(multiplication_checks):
            return 0.0

        # Simplified correlation calculation (sampled version)
        try:
            # Use only the first 1000 elements for correlation calculation
            flat1 = param1.data.flatten()[:1000]
            flat2 = param2.data.flatten()[:1000]

            min_size = min(len(flat1), len(flat2))
            if min_size > 1:
                flat1 = flat1[:min_size]
                flat2 = flat2[:min_size]

                correlation = torch.corrcoef(torch.stack([flat1, flat2]))[0, 1]
                if torch.isnan(correlation):
                    correlation = 0.0
                else:
                    correlation = abs(correlation.item())
            else:
                correlation = 0.0

            shape_compatibility = sum(multiplication_checks) / len(
                multiplication_checks
            )
            total_compatibility = shape_compatibility * 0.7 + correlation * 0.3

            return total_compatibility

        except Exception as e:
            logger.debug(f"Fast compatibility calculation failed: {e}")
            return 0.0

    @staticmethod
    def _determine_interaction_type_fast(
        param1: torch.Tensor, param2: torch.Tensor
    ) -> str:
        """Fastly determine interaction type"""
        shape1, shape2 = param1.shape, param2.shape

        if shape1[1] == shape2[0]:
            return "matmul_12"
        elif shape1[0] == shape2[1]:
            return "matmul_21"
        elif shape1[1] == shape2[1]:
            return "matmul_12t"
        elif shape1[0] == shape2[0]:
            return "matmul_1t2"
        else:
            return "norm_based"

    def _update_importance_scores_batch(self, group_metadata):
        """Batch update importance scores"""
        param_ids = []
        contribution_scores = []

        # Batch collect contribution scores
        for param in group_metadata["param_list"]:
            param_id = id(param)
            compact_state = group_metadata["compact_states"].get(param_id)

            if compact_state is not None:
                contribution = self._compute_parameter_contribution_score_optimized(
                    param, compact_state, param_id
                )
                param_ids.append(param_id)
                contribution_scores.append(contribution)

        # Batch update quantized importance scores
        for param_id, contribution in zip(param_ids, contribution_scores):
            old_quantized = self.quantized_importance_scores.get(
                param_id, quantize_importance_score(1.0)
            )
            old_importance = dequantize_importance_score(old_quantized)

            new_importance = (
                self.importance_decay * old_importance
                + (1 - self.importance_decay) * contribution
            )

            self.quantized_importance_scores[param_id] = quantize_importance_score(
                new_importance
            )

    def _apply_orthogonal_gradient_optimized(
        self, grad: torch.Tensor, param: torch.Tensor
    ) -> torch.Tensor:
        """
        Memory-optimized orthogonal gradient projection
        """
        param_norm = torch.norm(param.data, p=2)
        if param_norm <= 1e-30:
            return grad

        param_flat = param.data.view(-1)
        grad_flat = grad.view(-1)

        if param_flat.shape != grad_flat.shape:
            return grad

        # Use JIT-compiled core function
        orthogonal_grad_flat = orthogonal_gradient_core_optimized(
            grad_flat, param_flat, 1e-30
        )

        return orthogonal_grad_flat.view_as(grad)

    def _compute_adaptive_lr_scale_optimized(
        self, param, group_metadata, state, grad=None, global_step=None
    ):
        """
        Optimized adaptive learning rate scaling calculation
        """
        param_id = id(param)
        compact_state = group_metadata["compact_states"].get(param_id)

        if compact_state is None:
            return 1.0

        # === lr_mask basic adjustment ===
        lr_mask_scale = 1.0
        if self.use_lr_mask and grad is not None and global_step is not None:
            lr_mask_scale = self._update_lr_mask_optimized(
                compact_state, grad, global_step
            )

        # === Adaptive advanced adjustment ===
        adaptive_scale = 1.0
        if self.use_dynamic_adaptation:
            # 1. Adjust based on importance
            quantized_importance = self.quantized_importance_scores.get(
                param_id, quantize_importance_score(1.0)
            )
            importance = dequantize_importance_score(quantized_importance)
            importance_factor = min(
                3.0, max(0.1, importance * self.adaptation_strength)
            )

            # 2. Adjust based on parameter relationships (simplified version)
            relationship_scale = 1.0
            rel_info = self.compressed_relationships.get_relationship(param_id)
            if rel_info is not None:
                compatibility_bonus = rel_info["compatibility"]
                relationship_scale = 1.0 + compatibility_bonus * 0.2

            adaptive_scale = importance_factor * relationship_scale
            adaptive_scale = max(0.01, min(5.0, adaptive_scale))

        # === Combine final scaling factor ===
        # Handle lr_mask_scale being a tensor
        if isinstance(lr_mask_scale, torch.Tensor):
            # If lr_mask_scale is a tensor, multiply directly
            final_scale = lr_mask_scale * adaptive_scale
            # Use torch.clamp instead of max/min for tensors
            final_scale = torch.clamp(final_scale, min=0.001, max=10.0)
        else:
            # If it's a scalar, use the original logic
            final_scale = lr_mask_scale * adaptive_scale
            final_scale = max(0.001, min(10.0, final_scale))

        return final_scale

    def _update_lr_mask_optimized(self, compact_state, grad, global_step):
        """Optimized lr_mask update"""
        if not self.use_lr_mask:
            return 1.0

        # Get or initialize lr_mask
        lr_mask = compact_state.get_tensor("lr_mask", torch.float32)
        if lr_mask is None:
            device = grad.device
            shape = grad.shape
            dtype = torch.bfloat16 if self.reduce_precision else torch.float32
            lr_mask = (
                torch.ones(shape, device=device, dtype=dtype) * self.defaults["lr"]
            )
            compact_state.set_tensor(
                "lr_mask", lr_mask, use_half_precision=self.reduce_precision
            )

        if global_step < self.warmup_steps:
            return self._update_warmup_lr_mask_optimized(
                compact_state, grad, global_step
            )
        else:
            return self._update_post_warmup_lr_mask_optimized(
                compact_state, grad, global_step
            )

    def _update_warmup_lr_mask_optimized(self, compact_state, grad, global_step):
        """Optimized warmup lr_mask update"""
        # Use new get_tensor method to directly get the tensor on the correct device
        last_polarity = compact_state.get_tensor(
            "last_polarity", target_device=grad.device
        )
        current_polarity = grad > 0

        if last_polarity is not None:
            sign_agree = torch.where(last_polarity == current_polarity, 1.0, -1.0)
        else:
            sign_agree = torch.ones_like(
                current_polarity, dtype=torch.float32, device=grad.device
            )

        compact_state.set_tensor("last_polarity", current_polarity)

        # Use new get_tensor method to get lr_mask on the correct device and dtype
        lr_mask = compact_state.get_tensor(
            "lr_mask", target_dtype=torch.float32, target_device=grad.device
        )

        # Use JIT-compiled core update function
        new_lr_mask = compute_lr_mask_update_core(
            lr_mask, sign_agree, self.lr_bump, self.min_lr, self.max_lr
        )

        # Update state
        if self.reduce_precision:
            compact_state.set_tensor(
                "lr_mask", new_lr_mask.to(torch.bfloat16), use_half_precision=True
            )
        else:
            compact_state.set_tensor("lr_mask", new_lr_mask)

        compact_state.set_scalar("avg_lr", torch.mean(new_lr_mask).item())

        # Return relative scaling factor
        base_lr = self.defaults["lr"]
        lr_scale = new_lr_mask / base_lr if base_lr > 0 else new_lr_mask

        return lr_scale

    def _update_post_warmup_lr_mask_optimized(self, compact_state, grad, global_step):
        """Optimized post-warmup lr_mask update"""
        warmup_complete = compact_state.get_scalar("warmup_complete")
        if warmup_complete < 0.5:  # False
            compact_state.set_scalar("warmup_complete", 1.0)  # True

        # Use new get_tensor method to get lr_mask on the correct device and dtype
        lr_mask = compact_state.get_tensor(
            "lr_mask", target_dtype=torch.float32, target_device=grad.device
        )

        # If lr_mask is None, return scalar scaling factor
        if lr_mask is None:
            return 1.0

        # In post-warmup stage, maintain stability
        base_lr = self.defaults["lr"]
        lr_scale = lr_mask / base_lr if base_lr > 0 else lr_mask

        return lr_scale

    def _apply_spd_regularization_optimized(self, param, group, state):
        """Apply SPD regularization (memory optimized version)"""
        if param not in self.initial_params:
            return 0

        initial_param = self.initial_params[param]

        # If initial parameter is on CPU, move it to the same device
        if initial_param.device != param.data.device:
            if self.cpu_offload_states:
                # Temporarily move to GPU for calculation
                initial_param_gpu = initial_param.to(param.data.device)
                param_diff = param.data - initial_param_gpu
                # Immediately clean up temporary tensor
                del initial_param_gpu
            else:
                param_diff = param.data - initial_param
        else:
            param_diff = param.data - initial_param

        # Calculate bias ratio
        param_norm = torch.norm(param.data)
        diff_norm = torch.norm(param_diff)

        if param_norm > 0:
            bias_ratio = diff_norm / param_norm
        else:
            bias_ratio = 0

        # SPD penalty term
        spd_penalty = self.spd_lambda * bias_ratio * param_diff

        return spd_penalty

    @staticmethod
    @torch.jit.script
    def _apply_agr_regularization_optimized(grad: torch.Tensor) -> torch.Tensor:
        """Apply AGR regularization (JIT optimized version)"""
        grad_norm = torch.norm(grad)
        if grad_norm > 1.0:
            clip_factor = 1.0 / grad_norm
            return grad * clip_factor
        return grad

    @staticmethod
    @torch.jit.script
    def _apply_cautious_update_optimized(
        update: torch.Tensor, grad: torch.Tensor
    ) -> torch.Tensor:
        """Apply cautious update strategy (JIT optimized version)"""
        update_flat = update.view(-1)
        grad_flat = grad.view(-1)

        update_norm = torch.norm(update_flat)
        grad_norm = torch.norm(grad_flat)

        if update_norm > 0 and grad_norm > 0:
            alignment = torch.dot(update_flat, grad_flat) / (update_norm * grad_norm)
            if alignment < 0.1:
                return update * 0.5

        return update

    def _apply_tam_damping_optimized(self, momentum, grad, state):
        """Apply TAM damping (memory optimized version)"""
        if "momentum_alignment" not in state:
            state["momentum_alignment"] = 0.0

        try:
            momentum_norm = torch.norm(momentum)
            grad_norm = torch.norm(grad)

            if momentum_norm > 0 and grad_norm > 0:
                momentum_flat = momentum.view(-1)
                grad_flat = grad.view(-1)

                if momentum_flat.size() == grad_flat.size():
                    alignment = torch.dot(momentum_flat, grad_flat) / (
                        momentum_norm * grad_norm
                    )
                    alignment = alignment.item()
                else:
                    alignment = 0.0
            else:
                alignment = 0.0
        except Exception as e:
            logger.debug(f"TAM alignment calculation failed: {e}")
            alignment = 0.0

        # Smooth alignment estimate
        state["momentum_alignment"] = (
            self.tam_beta * state["momentum_alignment"]
            + (1 - self.tam_beta) * alignment
        )

        # Calculate damping factor
        damping_factor = (1 + state["momentum_alignment"]) / 2
        return damping_factor

    def _compute_edge_penalty_optimized(
        self,
        grad: torch.Tensor,
        threshold: float = 0.6,
        cache_key: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Optimized edge penalty calculation, used to control edge overfitting

        Args:
            grad: Gradient tensor
            threshold: Edge detection threshold
            cache_key: Cache key (optional)

        Returns:
            Edge penalty tensor
        """
        if len(grad.shape) < 2:
            return torch.zeros_like(grad)

        # Check cache (if enabled)
        if cache_key and hasattr(self, "edge_cache"):
            cached = self.edge_cache.get(cache_key)
            if cached is not None and cached.shape == grad.shape:
                return cached

        with torch.no_grad():
            # Get tensor using buffer pool
            laplacian = self.buffer_pool.get_buffer_with_priority(
                grad.shape, grad.dtype, grad.device, priority="normal"
            )

            # Simplified edge detection: calculate Laplacian operator
            if len(grad.shape) == 2 and grad.shape[0] > 2 and grad.shape[1] > 2:
                # Horizontal second derivative
                laplacian[1:-1, :] = grad[2:, :] - 2 * grad[1:-1, :] + grad[:-2, :]
                # Vertical second derivative
                laplacian[:, 1:-1] += grad[:, 2:] - 2 * grad[:, 1:-1] + grad[:, :-2]

            # Calculate edge strength
            edge_strength = torch.abs(laplacian)
            edge_mask = (edge_strength > threshold).float()
            result = edge_mask * edge_strength

            # Cache result
            if cache_key and hasattr(self, "edge_cache"):
                self.edge_cache[cache_key] = result.clone()

            # Return buffer
            self.buffer_pool.return_buffer(laplacian)

            return result

    def _compute_frequency_penalty_simplified(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Simplified frequency penalty calculation, used to control high-frequency noise

        Args:
            grad: Gradient tensor

        Returns:
            Frequency penalty tensor
        """
        if len(grad.shape) < 2:
            return torch.zeros_like(grad)

        with torch.no_grad():
            # Use simplified high-frequency detection: calculate adjacent element differences
            if len(grad.shape) == 2:
                h, w = grad.shape
                if h > 1 and w > 1:
                    # Get buffer
                    result = self.buffer_pool.get_buffer_with_priority(
                        grad.shape, grad.dtype, grad.device, priority="normal"
                    )

                    # Calculate horizontal and vertical differences
                    h_diff = torch.abs(grad[:, 1:] - grad[:, :-1])
                    v_diff = torch.abs(grad[1:, :] - grad[:-1, :])

                    # Combine difference information
                    result[:, 1:] = h_diff
                    result[1:, :] += v_diff

                    return result

            return torch.zeros_like(grad)

    def _lora_rank_regularization_fast(
        self, param: torch.Tensor, rank_strength: float = 0.01, use_approx: bool = True
    ) -> torch.Tensor:
        """
        Fast LoRA low-rank regularization, used to control LoRA layer overfitting

        Args:
            param: Parameter tensor
            rank_strength: Rank regularization strength
            use_approx: Whether to use an approximate method

        Returns:
            Low-rank regularization penalty tensor
        """
        if len(param.shape) != 2:
            return torch.zeros_like(param)

        with torch.no_grad():
            if use_approx:
                # Use approximate method: only consider the largest singular values
                # Calculate covariance matrix
                if param.shape[0] <= param.shape[1]:
                    cov = torch.mm(param, param.t())
                else:
                    cov = torch.mm(param.t(), param)

                # Calculate eigenvalues (only take the first few)
                try:
                    eigenvals, _ = torch.linalg.eigh(cov)
                    # Penalize larger eigenvalues (promote low-rank)
                    large_eigenvals = eigenvals[eigenvals.argsort(descending=True)[:10]]
                    rank_penalty_scalar = torch.sum(large_eigenvals) * rank_strength

                    # Create gradient approximation
                    return param * rank_penalty_scalar
                except Exception as e:
                    logger.debug(f"LoRA rank regularization calculation failed: {e}")
                    return torch.zeros_like(param)
            else:
                # Full SVD (if needed)
                try:
                    U, S, Vh = torch.linalg.svd(param, full_matrices=False)
                    # Penalize larger singular values
                    large_s = S[S.argsort(descending=True)[:10]]
                    rank_penalty = torch.sum(large_s) * rank_strength
                    penalty_grad = U @ torch.diag(S * rank_penalty / torch.sum(S)) @ Vh
                    return penalty_grad
                except Exception as e:
                    logger.debug(
                        f"Full SVD rank regularization calculation failed: {e}"
                    )
                    return torch.zeros_like(param)

    def _apply_spatial_awareness_regularization(
        self, grad: torch.Tensor, state: dict
    ) -> torch.Tensor:
        """
        Apply spatial awareness regularization, used to control spatial overfitting

        Args:
            grad: Gradient tensor
            state: Parameter state

        Returns:
            Regularized gradient
        """
        if len(grad.shape) < 2:
            return grad

        with torch.no_grad():
            # Initialize spatial variance tracking
            if "spatial_variance" not in state:
                state["spatial_variance"] = torch.ones_like(grad)

            if "detail_tracker" not in state:
                state["detail_tracker"] = torch.zeros_like(grad)

            # Calculate local variance
            local_variance = torch.zeros_like(grad)
            if len(grad.shape) == 2 and grad.shape[0] > 2 and grad.shape[1] > 2:
                # Calculate variance for 3x3 neighborhood
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i == 0 and j == 0:
                            continue

                        # Calculate difference after offset
                        if i == 0:
                            if j == 1:
                                local_variance[:, :-1] += torch.pow(
                                    grad[:, 1:] - grad[:, :-1], 2
                                )
                            elif j == -1:
                                local_variance[:, 1:] += torch.pow(
                                    grad[:, :-1] - grad[:, 1:], 2
                                )
                        elif j == 0:
                            if i == 1:
                                local_variance[:-1, :] += torch.pow(
                                    grad[1:, :] - grad[:-1, :], 2
                                )
                            elif i == -1:
                                local_variance[1:, :] += torch.pow(
                                    grad[:-1, :] - grad[1:, :], 2
                                )

            # Update spatial variance tracking
            state["spatial_variance"] = (
                0.9 * state["spatial_variance"] + 0.1 * local_variance
            )

            # Adjust gradient based on spatial variance
            regularization_factor = 1.0 / (
                1.0 + state["spatial_variance"] * self.detail_preservation
            )

            return grad * regularization_factor

    @torch.no_grad()
    def step(self, closure=None):
        """Execute optimization step - memory optimized version"""
        loss = None
        if closure is not None:
            loss = closure()

        # Check memory and adapt
        self._check_memory_and_adapt()

        # Global step count
        if not hasattr(self, "global_step"):
            self.global_step = 0
        self.global_step += 1

        # Collect completed asynchronous tasks
        completed_relationships = self.async_manager.collect_completed_tasks()
        for relationship_batch in completed_relationships:
            for rel in relationship_batch:
                self.compressed_relationships.add_relationship(
                    rel["param1_id"],
                    rel["param2_id"],
                    rel["compatibility"],
                    rel["interaction_type"],
                )

        for group_idx, group in enumerate(self.param_groups):
            group_metadata = self.param_groups_metadata[group_idx]

            # Periodically update parameter relationships and importance scores
            should_update_relationships = (
                self.global_step - self.last_relationship_update
                >= self.relationship_discovery_interval
            )

            if should_update_relationships:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Step {self.global_step}: Updating parameter relationships and importance scores"
                    )

                # Batch update importance scores
                self._update_importance_scores_batch(group_metadata)

                # Asynchronously rediscover parameter relationships
                if self.use_dynamic_adaptation and self.adaptive_features:
                    self._discover_parameter_relationships_async(group_metadata)

                self.last_relationship_update = self.global_step

            # Process each parameter
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError("HinaAdaptive does not support sparse gradients")

                state = self.state[param]
                param_id = id(param)
                compact_state = group_metadata["compact_states"].get(param_id)

                # State initialization
                if len(state) == 0:
                    state["step"] = 0

                    # Select precision based on memory settings
                    if self.reduce_precision:
                        state["exp_avg"] = torch.zeros_like(
                            param.data, dtype=torch.bfloat16
                        )
                        state["exp_avg_sq"] = torch.zeros_like(
                            param.data, dtype=torch.bfloat16
                        )
                        if self.use_adopt_stability:
                            state["exp_avg_sq_prev"] = torch.zeros_like(
                                param.data, dtype=torch.bfloat16
                            )
                    else:
                        state["exp_avg"] = torch.zeros_like(
                            param.data, dtype=torch.float32
                        )
                        state["exp_avg_sq"] = torch.zeros_like(
                            param.data, dtype=torch.float32
                        )
                        if self.use_adopt_stability:
                            state["exp_avg_sq_prev"] = torch.zeros_like(
                                param.data, dtype=torch.float32
                            )

                state["step"] += 1

                beta1, beta2 = group["betas"]
                step_size = group["lr"]

                # AGR regularization
                if self.use_agr:
                    grad = HinaAdaptive._apply_agr_regularization_optimized(grad)

                # === Edge overfitting control ===
                if len(grad.shape) >= 2:
                    # Edge-aware gradient regularization
                    if self.edge_suppression:
                        cache_key = f"edge_p_{param_id}_{state['step']}"
                        edge_penalty = self._compute_edge_penalty_optimized(
                            grad, self.edge_threshold, cache_key
                        )

                        # Apply edge penalty
                        if edge_penalty.numel() > 0:
                            edge_factor = 1.0 + self.edge_penalty * edge_penalty
                            grad = grad * (1.0 / edge_factor)

                            # Update edge history
                            if compact_state is not None:
                                edge_history = compact_state.get_tensor(
                                    "edge_history", target_device=grad.device
                                )
                                if edge_history is not None:
                                    edge_history = (
                                        0.9 * edge_history + 0.1 * edge_penalty
                                    )
                                    compact_state.set_tensor(
                                        "edge_history", edge_history
                                    )
                                    compact_state.set_scalar(
                                        "edge_strength", torch.mean(edge_penalty).item()
                                    )

                    # Frequency-aware gradient adjustment
                    if self.spatial_awareness:
                        freq_penalty = self._compute_frequency_penalty_simplified(grad)
                        if freq_penalty.numel() > 0:
                            grad = grad - self.frequency_penalty * freq_penalty

                            # Update spatial activity
                            if compact_state is not None:
                                spatial_activity = torch.mean(
                                    torch.abs(freq_penalty)
                                ).item()
                                compact_state.set_scalar(
                                    "spatial_activity", spatial_activity
                                )

                    # Apply spatial awareness regularization
                    if self.spatial_awareness:
                        grad = self._apply_spatial_awareness_regularization(grad, state)

                # LoRA low-rank regularization
                if self.lora_rank_penalty and len(param.shape) == 2:
                    rank_penalty = self._lora_rank_regularization_fast(
                        param, self.rank_penalty_strength, use_approx=True
                    )
                    if rank_penalty.numel() > 0:
                        grad = grad + rank_penalty

                        # Update rank tracking
                        if compact_state is not None:
                            rank_penalty_magnitude = torch.mean(
                                torch.abs(rank_penalty)
                            ).item()
                            compact_state.set_scalar(
                                "rank_penalty_history", rank_penalty_magnitude
                            )

                # Orthogonal gradient projection
                if self.use_orthogonal_grad:
                    grad = self._apply_orthogonal_gradient_optimized(grad, param)

                # Bias-corrected learning rate
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Update momentum estimate (considering precision)
                if self.use_adopt_stability and "exp_avg_sq_prev" in state:
                    state["exp_avg_sq_prev"] = state["exp_avg_sq"].clone()

                # Ensure calculations are performed at the correct precision
                if self.reduce_precision:
                    # Calculate in bfloat16 precision
                    grad_bf16 = grad.to(torch.bfloat16)
                    state["exp_avg"].mul_(beta1).add_(grad_bf16, alpha=1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(
                        grad_bf16, grad_bf16, value=1 - beta2
                    )
                else:
                    # Calculate in float32 precision
                    state["exp_avg"].mul_(beta1).add_(grad, alpha=1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(
                        grad, grad, value=1 - beta2
                    )

                # Calculate update
                if self.use_adopt_stability and "exp_avg_sq_prev" in state:
                    exp_avg_sq_hat = torch.maximum(
                        state["exp_avg_sq"], state["exp_avg_sq_prev"]
                    )
                    denom = (exp_avg_sq_hat.sqrt() / math.sqrt(bias_correction2)).add_(
                        group["eps"]
                    )
                else:
                    denom = (
                        state["exp_avg_sq"].sqrt() / math.sqrt(bias_correction2)
                    ).add_(group["eps"])

                # Convert update to float32 to maintain precision
                if self.reduce_precision:
                    exp_avg_corrected = (
                        state["exp_avg"].to(torch.float32) / bias_correction1
                    )
                    denom = denom.to(torch.float32)
                else:
                    exp_avg_corrected = state["exp_avg"] / bias_correction1

                update = exp_avg_corrected / denom

                # TAM damping
                if self.use_tam:
                    damping_factor = self._apply_tam_damping_optimized(
                        state["exp_avg"], grad, state
                    )
                    update = update * damping_factor

                # Cautious update
                if self.use_cautious:
                    update = HinaAdaptive._apply_cautious_update_optimized(update, grad)

                # Dynamic adaptive learning rate adjustment
                current_step_size = step_size
                if (
                    self.use_dynamic_adaptation or self.use_lr_mask
                ) and compact_state is not None:
                    lr_scale = self._compute_adaptive_lr_scale_optimized(
                        param,
                        group_metadata,
                        state,
                        grad=grad,
                        global_step=self.global_step,
                    )

                    # Handle lr_scale being a tensor or scalar
                    if isinstance(lr_scale, torch.Tensor):
                        if lr_scale.numel() == 1:
                            current_step_size *= lr_scale.item()
                        else:
                            # Ensure lr_scale is on the same device as update
                            if lr_scale.device != update.device:
                                lr_scale = lr_scale.to(update.device)
                            # Element-wise learning rate adjustment
                            param.data.add_(update * lr_scale, alpha=-step_size)
                            current_step_size = 0  # Skip subsequent update application
                    else:
                        current_step_size *= lr_scale

                # Apply update (if not already applied)
                if current_step_size != 0:
                    param.data.add_(update, alpha=-current_step_size)

                # Weight decay
                current_weight_decay = group["weight_decay"]

                # Dynamic weight decay
                if self.dynamic_weight_decay:
                    if state["step"] > self.wd_transition_steps:
                        progress = (
                            state["step"] - self.wd_transition_steps
                        ) / self.wd_transition_steps
                        decay_multiplier = max(
                            self.wd_min_ratio,
                            self.wd_decay_factor ** min(progress, 2.0),
                        )
                        current_weight_decay *= decay_multiplier

                if current_weight_decay != 0:
                    param.data.add_(
                        param.data, alpha=-group["lr"] * current_weight_decay
                    )

                # SPD regularization
                if self.use_spd:
                    spd_penalty = self._apply_spd_regularization_optimized(
                        param, group, state
                    )
                    if isinstance(spd_penalty, torch.Tensor):
                        param.data.add_(spd_penalty, alpha=-group["lr"])

        return loss

    def update_device(self, device):
        """When the model is moved to a new device, update the optimizer's internal state"""
        if hasattr(self, "initial_params"):
            for param, initial_param in self.initial_params.items():
                if initial_param.device != device:
                    self.initial_params[param] = initial_param.to(device)

        # Update tensors in all states
        for state in self.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.device != device:
                    state[key] = value.to(device)

        # Update tensors in compact states
        for group_metadata in self.param_groups_metadata.values():
            for compact_state in group_metadata["compact_states"].values():
                for key, tensor in compact_state.tensor_states.items():
                    if tensor.device != device:
                        compact_state.tensor_states[key] = tensor.to(device)

    def get_optimization_info(self) -> Dict[str, Any]:
        """Get detailed optimization information"""
        info = {
            "optimizer_type": "HinaAdaptive",
            "version": "Memory optimized version v1.0",
            "total_params": sum(len(group["params"]) for group in self.param_groups),
            "features": {
                "spd": self.use_spd,
                "cautious": self.use_cautious,
                "orthogonal_grad": self.use_orthogonal_grad,
                "adopt_stability": self.use_adopt_stability,
                "grams": self.use_grams,
                "agr": self.use_agr,
                "tam": self.use_tam,
                "dynamic_adaptation": self.use_dynamic_adaptation,
                "lr_mask": self.use_lr_mask,
                "dynamic_weight_decay": self.dynamic_weight_decay,
                "edge_suppression": self.edge_suppression,
                "spatial_awareness": self.spatial_awareness,
                "lora_rank_penalty": self.lora_rank_penalty,
            },
            "adaptation_config": {
                "adaptation_strength": self.adaptation_strength,
                "relationship_discovery_interval": self.relationship_discovery_interval,
                "importance_decay": self.importance_decay,
                "compatibility_threshold": self.compatibility_threshold,
            },
            "memory_optimization": {
                "memory_efficient": self.memory_efficient,
                "vram_budget_gb": self.vram_budget_gb,
                "cpu_offload_states": self.cpu_offload_states,
                "reduce_precision": self.reduce_precision,
                "adaptive_features": self.adaptive_features,
                "emergency_simplify": self.emergency_simplify,
            },
            "edge_overfitting_control": {
                "edge_penalty": self.edge_penalty,
                "edge_threshold": self.edge_threshold,
                "frequency_penalty": self.frequency_penalty,
                "detail_preservation": self.detail_preservation,
                "rank_penalty_strength": self.rank_penalty_strength,
                "low_rank_emphasis": self.low_rank_emphasis,
            },
            "current_memory_pressure": self.memory_monitor.check_memory_pressure(),
        }

        # Add dynamic statistics
        if hasattr(self, "global_step"):
            total_relationships = len(self.compressed_relationships.param_pairs)
            total_importance_scores = len(self.quantized_importance_scores)

            avg_importance = 0.0
            if total_importance_scores > 0:
                quantized_scores = list(self.quantized_importance_scores.values())
                avg_importance = (
                    sum(dequantize_importance_score(q) for q in quantized_scores)
                    / total_importance_scores
                )

            info["training_stats"] = {
                "global_step": self.global_step,
                "total_relationships": total_relationships,
                "total_importance_scores": total_importance_scores,
                "avg_importance_score": avg_importance,
                "pending_async_tasks": len(self.async_manager.pending_futures),
            }

        return info

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get detailed memory statistics"""
        stats = {
            "memory_pressure": self.memory_monitor.check_memory_pressure(),
            "buffer_pool_stats": {
                "total_buffer_types": len(self.buffer_pool._buffer_pool),
                "current_memory_mb": self.buffer_pool._current_memory / (1024 * 1024),
                "max_memory_mb": self.buffer_pool._max_total_memory / (1024 * 1024),
            },
            "state_compression": {
                "quantized_importance_scores": len(self.quantized_importance_scores),
                "compressed_relationships": len(
                    self.compressed_relationships.param_pairs
                ),
            },
        }

        if torch.cuda.is_available():
            stats["cuda_memory"] = {
                "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
            }

        return stats

    def optimize_for_vram(self, target_vram_gb: float):
        """Automatically optimize settings based on target VRAM"""
        self.vram_budget_gb = target_vram_gb
        self.memory_monitor.target_vram = target_vram_gb * 1024**3

        current_pressure = self.memory_monitor.check_memory_pressure()

        if current_pressure > 0.9:
            logger.warning(
                f"VRAM usage {current_pressure:.1%} is high, enabling emergency optimization mode"
            )
            # Enable all memory optimizations
            self.reduce_precision = True
            self.cpu_offload_states = True
            self.emergency_simplify = True

            # Reduce asynchronous tasks
            self.relationship_discovery_interval *= 2

            # Force buffer pool cleanup
            self.buffer_pool.smart_cleanup(current_pressure)
            torch.cuda.empty_cache()

        elif current_pressure > 0.7:
            logger.info(
                f"VRAM usage {current_pressure:.1%}, enabling standard optimization mode"
            )
            self.reduce_precision = True
            self.cpu_offload_states = True

        else:
            logger.info(f"VRAM usage {current_pressure:.1%}, memory sufficient")

    def cleanup_resources(self):
        """Clean up resources and release memory"""
        # Clean up buffer pool
        self.buffer_pool._buffer_pool.clear()
        self.buffer_pool._current_memory = 0

        # Shutdown asynchronous manager
        self.async_manager.shutdown()

        # Clean up compressed relationships
        self.compressed_relationships.param_pairs.clear()
        self.compressed_relationships.compatibility_scores = torch.tensor(
            [], dtype=torch.float16
        )
        self.compressed_relationships.interaction_types.clear()

        # Clean up quantized importance scores
        self.quantized_importance_scores.clear()

        # Clean up edge cache
        if hasattr(self, "edge_cache"):
            self.edge_cache.clear()

        # Force garbage collection
        torch.cuda.empty_cache()

        logger.info("All optimizer resources cleaned up")

    def __del__(self):
        """Destructor, ensure resources are cleaned up correctly"""
        try:
            self.cleanup_resources()
        except Exception as e:
            logger.warning(f"Error cleaning up resources: {e}")
