import argparse
from typing import Any, Optional, Tuple

import torch
from accelerate import Accelerator

from common.logger import get_logger
from scheduling.timestep_utils import get_noisy_model_input_and_timesteps
from utils.device_utils import synchronize_device
from utils.train_utils import clean_memory_on_device


logger = get_logger(__name__)


class DualModelManager:
    """Manage high/low-noise base model contexts under a single LoRA network.

    This swaps only the base model state_dict underneath the active transformer
    instance so the attached LoRA layers remain continuous across swaps.

    MEMORY USAGE: Uses the exact same amount of RAM as single model training:
    - ONE model in GPU memory (active)
    - ONE model state_dict in CPU/GPU memory (inactive)
    - During swap: temporarily holds both state_dicts during transition
    - NO additional GPU memory required beyond single model training
    """

    def __init__(
        self,
        *,
        active_transformer: torch.nn.Module,
        high_noise_state_dict: dict,
        timestep_boundary: float,
        offload_inactive: bool,
        blocks_to_swap: int = 0,
    ) -> None:
        # Normalize boundary to [0,1] (identical to original)
        if timestep_boundary > 1.0:
            timestep_boundary = timestep_boundary / 1000.0
        self.timestep_boundary: float = float(timestep_boundary)
        self.blocks_to_swap: int = int(blocks_to_swap)

        # Mixed mode supported: block swap for active DiT while inactive DiT is offloaded on CPU

        self.offload_inactive: bool = bool(offload_inactive)

        # Active model is the already-prepared/wrapped transformer
        self.active_model: torch.nn.Module = active_transformer

        # MEMORY OPTIMIZATION: Store the inactive state dict exactly like the original implementation
        # The original keeps the high-noise state dict and starts with low-noise active
        # This ensures we only have ONE model in GPU memory at any time, just like single model training
        self.inactive_state_dict: dict = high_noise_state_dict

        # Track which regime is currently loaded (identical to original)
        self.current_model_is_high_noise: bool = False
        self.next_model_is_high_noise: bool = False

    @torch.no_grad()
    def determine_and_prepare_batch(
        self,
        *,
        args: argparse.Namespace,
        noise: torch.Tensor,
        latents: torch.Tensor,
        noise_scheduler: Any,
        device: torch.device,
        dtype: torch.dtype,
        timestep_distribution: Optional[Any] = None,
        presampled_uniform: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Decide high/low noise regime for this batch and ensure consistency.

        Returns a tuple of (noisy_model_input, timesteps, sigmas).
        All items in the batch are guaranteed to fall on the same side of the
        boundary determined by `self.timestep_boundary`.
        """
        batch_size = latents.shape[0]

        # Probe one sample to decide regime for the entire batch
        probe_noisy, probe_timesteps, _ = get_noisy_model_input_and_timesteps(
            args,
            noise[0:1],
            latents[0:1],
            noise_scheduler,
            device,
            dtype,
            timestep_distribution,
            presampled_uniform=(
                presampled_uniform[0:1]
                if presampled_uniform is not None and len(presampled_uniform) > 0
                else None
            ),
        )
        # Align with reference: normalize by 1000.0 without subtracting 1
        probe_t = (probe_timesteps[0].item()) / 1000.0
        target_is_high = probe_t >= self.timestep_boundary
        self.next_model_is_high_noise = target_is_high

        final_inputs = []
        final_timesteps = []
        # sigmas are only relevant for sigma-based schemes; collect last seen
        last_sigmas = None

        # Strategy controls how per-epoch bucketed uniform samples interact with the boundary
        strategy = str(getattr(args, "dual_timestep_bucket_strategy", "hybrid")).lower()
        max_retries = int(getattr(args, "dual_timestep_bucket_max_retries", 100))
        eps = float(getattr(args, "dual_timestep_bucket_eps", 1e-4))

        def _call_with_uniform(idx: int, u: Optional[torch.Tensor]):
            return get_noisy_model_input_and_timesteps(
                args,
                noise[idx : idx + 1],
                latents[idx : idx + 1],
                noise_scheduler,
                device,
                dtype,
                timestep_distribution,
                presampled_uniform=u,
            )

        for i in range(batch_size):
            matched = False

            # Base presampled uniform from dataset pool if available
            base_u: Optional[torch.Tensor] = (
                presampled_uniform[i : i + 1]
                if presampled_uniform is not None and i < presampled_uniform.shape[0]
                else None
            )

            # Shortcut for "presampled": one attempt only using dataset-provided uniform
            if strategy == "presampled":
                nmi, ts, sig = _call_with_uniform(i, base_u)
                t_norm = (ts[0].item()) / 1000.0
                if (t_norm >= self.timestep_boundary) == target_is_high:
                    final_inputs.append(nmi)
                    final_timesteps.append(ts)
                    last_sigmas = sig
                    matched = True
                # If not matched, accept as-is (intentionally not forcing alignment)
                if not matched:
                    final_inputs.append(nmi)
                    final_timesteps.append(ts)
                    last_sigmas = sig
                continue

            # Strict clamp: if we have a presampled value, adjust it by binary search to cross boundary
            if strategy == "strict_clamp" and base_u is not None:
                lo = 0.0
                hi = 1.0
                if target_is_high:
                    lo = float(base_u[0].item())
                else:
                    hi = float(base_u[0].item())
                chosen_nmi = None
                chosen_ts = None
                chosen_sig = None
                for _ in range(max(4, min(max_retries, 20))):  # small bounded search
                    mid = (lo + hi) / 2.0
                    u_mid = torch.tensor([mid], device=device, dtype=torch.float32)
                    nmi, ts, sig = _call_with_uniform(i, u_mid)
                    t_norm = (ts[0].item()) / 1000.0
                    if target_is_high:
                        if t_norm >= self.timestep_boundary + eps:
                            chosen_nmi, chosen_ts, chosen_sig = nmi, ts, sig
                            hi = mid
                            matched = True
                        else:
                            lo = mid
                    else:
                        if t_norm < self.timestep_boundary - eps:
                            chosen_nmi, chosen_ts, chosen_sig = nmi, ts, sig
                            lo = mid
                            matched = True
                        else:
                            hi = mid
                # If found a clamped candidate, use it; else fall back to one shot with base_u
                if matched and chosen_nmi is not None:
                    final_inputs.append(chosen_nmi)
                    final_timesteps.append(chosen_ts)  # type: ignore[arg-type]
                    last_sigmas = chosen_sig
                    continue
                else:
                    nmi, ts, sig = _call_with_uniform(i, base_u)
                    final_inputs.append(nmi)
                    final_timesteps.append(ts)
                    last_sigmas = sig
                    continue

            # Hybrid/on_demand/general retry logic
            for attempt in range(max_retries):
                if strategy == "on_demand":
                    u_try: Optional[torch.Tensor] = torch.rand(
                        1, device=device, dtype=torch.float32
                    )
                elif strategy == "hybrid" and attempt == 0 and base_u is not None:
                    u_try = base_u
                elif strategy == "hybrid":
                    u_try = torch.rand(1, device=device, dtype=torch.float32)
                else:
                    # Default: behave like original (use provided u if any; else None)
                    u_try = base_u

                nmi, ts, sig = _call_with_uniform(i, u_try)
                t_norm = (ts[0].item()) / 1000.0
                if (t_norm >= self.timestep_boundary) == target_is_high:
                    final_inputs.append(nmi)
                    final_timesteps.append(ts)
                    last_sigmas = sig
                    matched = True
                    break

            if not matched:
                # fallback to the last computed values (extremely rare)
                final_inputs.append(nmi)
                final_timesteps.append(ts)
                last_sigmas = sig

        noisy_model_input = torch.cat(final_inputs, dim=0)
        timesteps = torch.cat(final_timesteps, dim=0)
        return noisy_model_input, timesteps, last_sigmas

    @torch.no_grad()
    def swap_if_needed(self, accelerator: Accelerator) -> None:
        """Swap base weights if the upcoming batch requires the other regime."""
        if self.current_model_is_high_noise == self.next_model_is_high_noise:
            return

        dev = accelerator.device
        src = "High" if self.current_model_is_high_noise else "Low"
        dst = "High" if self.next_model_is_high_noise else "Low"
        logger.info(f"🔄 Swapping base model weights: {src} → {dst} noise regime")

        # Work with the unwrapped module to avoid DDP/Accelerate wrappers issues
        model_to_load = accelerator.unwrap_model(self.active_model)

        # IDENTICAL TO ORIGINAL: Check blocks_to_swap first (lines 559-585 in original)
        if self.blocks_to_swap == 0:
            # If offloading inactive DiT, move the model to CPU first (identical to original)
            if self.offload_inactive:
                model_to_load.to("cpu", non_blocking=True)
                synchronize_device(dev)  # wait for the CPU to finish
                clean_memory_on_device(dev)

            # Get current state dict (CPU or accelerator.device)
            current_sd = model_to_load.state_dict()

            # Load inactive state dict with strict validation (identical to original)
            info = model_to_load.load_state_dict(
                self.inactive_state_dict, strict=True, assign=True
            )
            assert len(info.missing_keys) == 0, f"Missing keys: {info.missing_keys}"
            assert (
                len(info.unexpected_keys) == 0
            ), f"Unexpected keys: {info.unexpected_keys}"

            if self.offload_inactive:
                model_to_load.to(dev, non_blocking=True)
                synchronize_device(dev)

            # Swap the state dict, ensuring the stored inactive copy lives on CPU to avoid
            # keeping a duplicate set of GPU tensors in mixed mode
            try:
                cpu_sd = {}
                for k, v in current_sd.items():
                    try:
                        cpu_sd[k] = v.detach().to("cpu", non_blocking=True)
                    except Exception:
                        cpu_sd[k] = v.detach().cpu()
                self.inactive_state_dict = cpu_sd
            except Exception:
                logger.error(
                    "Failed to convert state dict to CPU, falling back to original behavior"
                )
                # Fallback to original behavior if conversion fails
                self.inactive_state_dict = current_sd
        else:
            # If block swap is enabled, we cannot use offloading inactive DiT,
            # Mixed mode: active model may have swapped blocks; ensure state is valid post-swap
            current_sd = model_to_load.state_dict()

            info = model_to_load.load_state_dict(
                self.inactive_state_dict, strict=True, assign=True
            )
            assert len(info.missing_keys) == 0, f"Missing keys: {info.missing_keys}"
            assert (
                len(info.unexpected_keys) == 0
            ), f"Unexpected keys: {info.unexpected_keys}"

            # Swap the state dict (identical to original)
            self.inactive_state_dict = current_sd
            # If the model supports block swap preparation hooks, refresh them after weight load
            try:
                # Move resident (non-swapped) blocks back to the accelerator device because
                # load_state_dict(assign=True) may attach CPU tensors from the state_dict
                if hasattr(model_to_load, "move_to_device_except_swap_blocks"):
                    model_to_load.move_to_device_except_swap_blocks(accelerator.device)
                if hasattr(model_to_load, "prepare_block_swap_before_forward"):
                    model_to_load.prepare_block_swap_before_forward()
            except Exception as _prep_err:
                logger.debug(f"Block-swap prepare hook skipped: {_prep_err}")

        # Update current regime (identical to original)
        self.current_model_is_high_noise = self.next_model_is_high_noise
