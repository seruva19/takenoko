from concurrent.futures import ThreadPoolExecutor
import gc
import time
from typing import Optional, List
import argparse
import torch
import torch.nn as nn

import logging
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


def clean_memory_on_device(device: torch.device):
    """
    Clean memory on the specified device, will be called from training scripts.
    """
    gc.collect()

    # device may "cuda" or "cuda:0", so we need to check the type of device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()


def synchronize_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def swap_weight_devices_cuda(
    device: torch.device,
    layer_to_cpu: nn.Module,
    layer_to_cuda: nn.Module,
    enhanced_swapping: bool = False,
    pinned_memory_enabled: bool = False,
    non_blocking: bool = True,
    timing_enabled: bool = False,
    verbose_logging: bool = False,
):
    """Swap weights between CUDA and CPU devices with optional enhanced features."""
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    start_time = time.perf_counter() if timing_enabled else None

    weight_swap_jobs = []

    # This is not working for all cases (e.g. SD3), so we need to find the corresponding modules
    modules_to_cpu = {k: v for k, v in layer_to_cpu.named_modules()}
    for module_to_cuda_name, module_to_cuda in layer_to_cuda.named_modules():
        if hasattr(module_to_cuda, "weight") and module_to_cuda.weight is not None:
            module_to_cpu = modules_to_cpu.get(module_to_cuda_name, None)
            if (
                module_to_cpu is not None
                and module_to_cpu.weight.shape == module_to_cuda.weight.shape
            ):
                weight_swap_jobs.append(
                    (
                        module_to_cpu,
                        module_to_cuda,
                        module_to_cpu.weight.data,
                        module_to_cuda.weight.data,
                    )
                )
            else:
                if module_to_cuda.weight.data.device.type != device.type:
                    if verbose_logging:
                        logger.info(
                            f"Module {module_to_cuda_name} not found in CPU model or shape mismatch, moving to device"
                        )
                    module_to_cuda.weight.data = module_to_cuda.weight.data.to(device)  # type: ignore

    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value

    # Legacy entrypoint: always use original swapping algorithm to preserve backward compatibility.
    # Enhanced algorithm is available via Offloader.swap_weight_devices_cuda instance method.
    _swap_weights_original(device, weight_swap_jobs, non_blocking)

    if timing_enabled:
        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Swapped weights in {elapsed:.2f}s. Count of modules swapped: {len(weight_swap_jobs)}"
        )


def swap_weight_devices_no_cuda(
    device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module
):
    """
    CPU-only fallback path for platforms without CUDA support.
    """
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs = []

    modules_to_cpu = {name: mod for name, mod in layer_to_cpu.named_modules()}
    for module_to_cuda_name, module_to_cuda in layer_to_cuda.named_modules():
        if not hasattr(module_to_cuda, "weight") or module_to_cuda.weight is None:
            continue

        module_to_cpu = modules_to_cpu.get(module_to_cuda_name)
        if (
            module_to_cpu is None
            or not hasattr(module_to_cpu, "weight")
            or module_to_cpu.weight is None
        ):
            continue

        if module_to_cpu.weight.shape != module_to_cuda.weight.shape:
            continue

        weight_swap_jobs.append(
            (
                module_to_cpu,
                module_to_cuda,
                module_to_cpu.weight.data,
                module_to_cuda.weight.data,
            )
        )

    for module_to_cpu, module_to_cuda, _, _ in weight_swap_jobs:
        # Clone to avoid in-place overwrite before the swap completes
        cpu_clone = module_to_cpu.weight.data.clone()
        module_to_cpu.weight.data.copy_(module_to_cuda.weight.data)
        module_to_cuda.weight.data.copy_(cpu_clone)


def weights_to_device(
    layer: nn.Module,
    device: torch.device,
    non_blocking: bool = True,
    move_buffers: bool = True,
):
    """
    Move parameters and buffers of a layer to the target device,
    preserving non-blocking semantics when possible.
    """
    for module in layer.modules():
        if hasattr(module, "weight") and module.weight is not None:
            use_non_blocking = non_blocking and device.type != "cpu"
            module.weight.data = module.weight.data.to(  # type: ignore[attr-defined]
                device, non_blocking=use_non_blocking
            )

        if move_buffers:
            for buffer in module.buffers(recurse=False):
                if buffer is not None:
                    use_non_blocking = non_blocking and device.type != "cpu"
                    buffer.data = buffer.data.to(  # type: ignore[attr-defined]
                        device, non_blocking=use_non_blocking
                    )


def _swap_weights_original(
    device: torch.device, weight_swap_jobs: List, non_blocking: bool = True
):
    """Original weight swapping algorithm for backwards compatibility."""
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):  # type: ignore
        # cuda to cpu
        for (
            module_to_cpu,
            module_to_cuda,
            cpu_weight_tensor,
            gpu_weight_tensor,
        ) in weight_swap_jobs:
            cpu_weight_tensor.record_stream(stream)
            module_to_cpu.weight.data = cpu_weight_tensor.to(
                "cpu", non_blocking=non_blocking
            )

        stream.synchronize()

        # cpu to cuda
        for (
            module_to_cpu,
            module_to_cuda,
            cpu_weight_tensor,
            gpu_weight_tensor,
        ) in weight_swap_jobs:
            cpu_weight_tensor.copy_(module_to_cuda.weight.data, non_blocking=non_blocking)
            module_to_cuda.weight.data = cpu_weight_tensor

    stream.synchronize()
    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value


class Offloader:
    """
    common offloading class with enhanced features
    """

    def __init__(
        self,
        block_type: str,
        num_blocks: int,
        blocks_to_swap: int,
        device: torch.device,
        debug: bool = False,
        enhanced_enabled: bool = False,
        pinned_memory_enabled: bool = False,
        non_blocking_transfers: bool = True,
        timing_enabled: bool = False,
        verbose_logging: bool = False,
        target_memory_type: str = "auto",
        cpu_memory_priority: bool = False,
    ):
        self.block_type = block_type
        self.num_blocks = num_blocks
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.debug = debug

        # Enhanced offloading features
        self.enhanced_enabled = enhanced_enabled
        self.pinned_memory_enabled = pinned_memory_enabled
        self.non_blocking_transfers = non_blocking_transfers
        self.timing_enabled = timing_enabled
        self.verbose_logging = verbose_logging
        self.target_memory_type = target_memory_type
        self.cpu_memory_priority = cpu_memory_priority

        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.futures = {}
        self.cuda_available = device.type == "cuda"

        # Dedicated CUDA stream for asynchronous transfers when available
        self.stream = torch.cuda.Stream(device=device) if self.cuda_available else None

        # Persistent staging buffers for enhanced swapping
        self.staging_buffer_a = None
        self.staging_buffer_b = None
        # Reusable pinned buffers when pinned-memory mode is active
        self.pinned_buffer = None

        if self.verbose_logging:
            logger.info(
                f"Offloader initialized for {block_type}: enhanced={enhanced_enabled}, "
                f"pinned_memory={pinned_memory_enabled}, "
                f"target_memory_type={target_memory_type}, cpu_memory_priority={cpu_memory_priority}"
            )

    def swap_weight_devices(self, block_to_cpu: nn.Module, block_to_cuda: nn.Module):
        if self.cuda_available:
            return self.swap_weight_devices_cuda(self.device, block_to_cpu, block_to_cuda)
        return swap_weight_devices_no_cuda(self.device, block_to_cpu, block_to_cuda)

    def swap_weight_devices_cuda(
        self, device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module
    ):
        """Enhanced weight swapping with persistent staging buffers."""
        assert layer_to_cpu.__class__ == layer_to_cuda.__class__

        start_time = time.perf_counter() if self.timing_enabled else None

        weight_swap_jobs = []

        modules_to_cpu = {k: v for k, v in layer_to_cpu.named_modules()}
        for module_to_cuda_name, module_to_cuda in layer_to_cuda.named_modules():
            if hasattr(module_to_cuda, "weight") and module_to_cuda.weight is not None:
                module_to_cpu = modules_to_cpu.get(module_to_cuda_name, None)
                if (
                    module_to_cpu is not None
                    and module_to_cpu.weight.shape == module_to_cuda.weight.shape
                ):
                    weight_swap_jobs.append(
                        (
                            module_to_cpu,
                            module_to_cuda,
                            module_to_cpu.weight.data,
                            module_to_cuda.weight.data,
                        )
                    )
                else:
                    if module_to_cuda.weight.data.device.type != device.type:
                        if self.verbose_logging:
                            logger.info(
                                f"Module {module_to_cuda_name} not found in CPU model or shape mismatch, moving to device"
                            )
                        module_to_cuda.weight.data = module_to_cuda.weight.data.to(device)  # type: ignore

        torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value

        sync_event: Optional[torch.cuda.Event] = None

        # Master switch check: if enhanced offloading is disabled, always use original algorithm
        if not getattr(self, "enhanced_enabled", False):
            _swap_weights_original(device, weight_swap_jobs, self.non_blocking_transfers)
            if self.timing_enabled and start_time is not None:
                elapsed = time.perf_counter() - start_time
                logger.info(
                    f"Swapped weights (original) in {elapsed:.2f}s. Count of modules swapped: {len(weight_swap_jobs)}"
                )
            return sync_event

        # Mode selection: shared_gpu mode uses original algorithm (compatibility)
        if getattr(self, "target_memory_type", "auto") == "shared_gpu":
            _swap_weights_original(device, weight_swap_jobs, self.non_blocking_transfers)
            if self.timing_enabled and start_time is not None:
                elapsed = time.perf_counter() - start_time
                logger.info(
                    f"Swapped weights (shared_gpu) in {elapsed:.2f}s. Count of modules swapped: {len(weight_swap_jobs)}"
                )
            return sync_event

        # Use enhanced staging logic and optionally return an async completion event
        sync_event = self._swap_weights_enhanced_persistent_staging(
            device, weight_swap_jobs, start_time
        )
        return sync_event

    def _swap_weights_enhanced_persistent_staging(
        self, device: torch.device, weight_swap_jobs: List, start_time: Optional[float]
    ) -> Optional[torch.cuda.Event]:
        """Enhanced weight swapping with persistent staging buffers."""
        if not weight_swap_jobs:
            return None

        if self.pinned_memory_enabled:
            stream = self.stream
            if stream is None:
                stream = torch.cuda.Stream(device=device if device.type == "cuda" else None)
                self.stream = stream
            # Ensure we have reusable pinned buffers sized for the current jobs
            def _needs_reallocation(
                pin_buf: torch.Tensor, gpu_view: torch.Tensor
            ) -> bool:
                return (
                    pin_buf.shape != gpu_view.shape
                    or pin_buf.dtype != gpu_view.dtype
                    or pin_buf.device.type != "cpu"
                )

            if (
                self.pinned_buffer is None
                or len(self.pinned_buffer) != len(weight_swap_jobs)
                or any(
                    _needs_reallocation(pin_buf, gpu_view)
                    for pin_buf, (_, _, _, gpu_view) in zip(
                        self.pinned_buffer or [],
                        weight_swap_jobs,
                    )
                )
            ):
                with torch.cuda.stream(stream):
                    self.pinned_buffer = [
                        torch.empty_like(cuda_data_view, device="cpu").pin_memory(
                            device=device
                        )
                        for _, _, cuda_data_view, _ in weight_swap_jobs
                    ]
                stream.synchronize()

            events = [torch.cuda.Event() for _ in weight_swap_jobs]

            # Phase 1: move GPU weights to pinned CPU buffers asynchronously
            for event, pin_buf, (_module_to_cpu, _module_to_cuda, cpu_weight_tensor, gpu_weight_tensor) in zip(
                events, self.pinned_buffer, weight_swap_jobs
            ):
                with torch.cuda.stream(stream):
                    pin_buf.copy_(gpu_weight_tensor, non_blocking=True)
                    event.record(stream)

            # Phase 2: move CPU weights back to GPU once GPU->CPU copies complete
            for event, (module_to_cpu, module_to_cuda, cpu_weight_tensor, gpu_weight_tensor) in zip(
                events, weight_swap_jobs
            ):
                with torch.cuda.stream(stream):
                    stream.wait_event(event)
                    gpu_weight_tensor.copy_(module_to_cpu.weight.data, non_blocking=True)

            # Ensure both phases have completed before weights are reassigned
            stream.synchronize()

            # Update module references now that transfers have completed
            for pin_buf, (module_to_cpu, module_to_cuda, _cpu_weight_tensor, gpu_weight_tensor) in zip(
                self.pinned_buffer, weight_swap_jobs
            ):
                module_to_cuda.weight.data = gpu_weight_tensor
                module_to_cpu.weight.data = pin_buf

            sync_event = torch.cuda.Event()
            sync_event.record(stream)

            if self.timing_enabled and start_time is not None:
                elapsed = time.perf_counter() - start_time
                logger.info(
                    f"[{self.block_type}] Swapped weights (pinned) in {elapsed:.2f}s. Count of modules swapped: {len(weight_swap_jobs)}"
                )

            return sync_event

        # Non-pinned path keeps existing staging-buffer behaviour
        stream = torch.cuda.Stream(device=device if device.type == "cuda" else None)
        def _needs_staging_reallocation(
            buffers: Optional[List[torch.Tensor]],
            jobs: List,
        ) -> bool:
            if buffers is None or len(buffers) != len(jobs):
                return True
            return any(
                buf.shape != cuda_view.shape or buf.dtype != cuda_view.dtype
                for buf, (_, _, cuda_view, _) in zip(buffers, jobs)
            )

        with torch.cuda.stream(stream):
            if _needs_staging_reallocation(self.staging_buffer_a, weight_swap_jobs):
                # Create staging buffers pinned to device for correct multi-GPU behavior
                self.staging_buffer_a = [
                    torch.empty_like(cuda_data_view, device="cpu").pin_memory(
                        device=device
                    )
                    for _, _, cuda_data_view, _ in weight_swap_jobs
                ]
            if _needs_staging_reallocation(self.staging_buffer_b, weight_swap_jobs):
                self.staging_buffer_b = [
                    torch.empty_like(cuda_data_view, device="cpu").pin_memory(
                        device=device
                    )
                    for _, _, cuda_data_view, _ in weight_swap_jobs
                ]

            events = [torch.cuda.Event() for _ in weight_swap_jobs]

            # Phase 1: copy to staging and record events
            for (
                event,
                sbuf_a,
                sbuf_b,
                (
                    module_to_cpu,
                    module_to_cuda,
                    cuda_data_view,
                    cpu_data_view,
                ),
            ) in zip(
                events, self.staging_buffer_a, self.staging_buffer_b, weight_swap_jobs
            ):
                sbuf_a.copy_(cuda_data_view.data, non_blocking=True)  # CUDA -> pinned
                event.record(stream)

                sbuf_b.copy_(module_to_cuda.weight.data)  # CPU -> pinned (sync)

        with torch.cuda.stream(stream):
            # Phase 2: event-synchronized transfers from staging
            for (
                event,
                sbuf_a,
                sbuf_b,
                (
                    module_to_cpu,
                    module_to_cuda,
                    cuda_data_view,
                    cpu_data_view,
                ),
            ) in zip(
                events, self.staging_buffer_a, self.staging_buffer_b, weight_swap_jobs
            ):
                event.synchronize()
                cuda_data_view.copy_(sbuf_b, non_blocking=True)  # pinned -> CUDA
                cpu_data_view.copy_(sbuf_a)  # pinned -> CPU (sync)

                module_to_cuda.weight.data = cuda_data_view
                module_to_cpu.weight.data = cpu_data_view

        stream.synchronize()
        torch.cuda.current_stream().synchronize()

        if self.timing_enabled and start_time is not None:
            elapsed = time.perf_counter() - start_time
            logger.info(
                f"[{self.block_type}] Swapped weights in {elapsed:.2f}s. Count of modules swapped: {len(weight_swap_jobs)}"
            )
        return None

    def _submit_move_blocks(self, blocks, block_idx_to_cpu, block_idx_to_cuda):
        def move_blocks(bidx_to_cpu, block_to_cpu, bidx_to_cuda, block_to_cuda):
            if self.debug:
                start_time = time.perf_counter()
                print(
                    f"[{self.block_type}] Move block {bidx_to_cpu} to CPU and block {bidx_to_cuda} to {'CUDA' if self.cuda_available else 'device'}"
                )

            sync_event = self.swap_weight_devices(block_to_cpu, block_to_cuda)

            if self.debug:
                print(
                    f"[{self.block_type}] Moved blocks {bidx_to_cpu} and {bidx_to_cuda} in {time.perf_counter()-start_time:.2f}s"
                )
            return bidx_to_cpu, bidx_to_cuda, sync_event

        block_to_cpu = blocks[block_idx_to_cpu]
        block_to_cuda = blocks[block_idx_to_cuda]

        self.futures[block_idx_to_cuda] = self.thread_pool.submit(
            move_blocks,
            block_idx_to_cpu,
            block_to_cpu,
            block_idx_to_cuda,
            block_to_cuda,
        )

    def _wait_blocks_move(self, block_idx):
        if block_idx not in self.futures:
            return

        if self.debug:
            print(f"[{self.block_type}] Wait for block {block_idx}")
            start_time = time.perf_counter()

        future = self.futures.pop(block_idx)
        _, bidx_to_cuda, sync_event = future.result()

        assert (
            block_idx == bidx_to_cuda
        ), f"Block index mismatch: {block_idx} != {bidx_to_cuda}"

        if self.cuda_available and sync_event is not None:
            torch.cuda.current_stream().wait_event(sync_event)

        if self.debug:
            print(
                f"[{self.block_type}] Waited for block {block_idx}: {time.perf_counter()-start_time:.2f}s"
            )


class ModelOffloader(Offloader):
    """
    supports forward offloading with enhanced features
    """

    def __init__(
        self,
        block_type: str,
        blocks: list[nn.Module],
        num_blocks: int,
        blocks_to_swap: int,
        supports_backward: bool,
        device: torch.device,
        debug: bool = False,
        enhanced_enabled: bool = False,
        pinned_memory_enabled: bool = False,
        non_blocking_transfers: bool = True,
        timing_enabled: bool = False,
        verbose_logging: bool = False,
        target_memory_type: str = "auto",
        cpu_memory_priority: bool = False,
    ):
        super().__init__(
            block_type,
            num_blocks,
            blocks_to_swap,
            device,
            debug,
            enhanced_enabled,
            pinned_memory_enabled,
            non_blocking_transfers,
            timing_enabled,
            verbose_logging,
            target_memory_type,
            cpu_memory_priority,
        )

        self.supports_backward = supports_backward
        self.forward_only = (
            not supports_backward
        )  # forward only offloading: can be changed to True for inference

        if self.supports_backward:
            # register backward hooks
            self.remove_handles = []
            for i, block in enumerate(blocks):
                hook = self.create_backward_hook(blocks, i)
                if hook is not None:
                    handle = block.register_full_backward_hook(hook)
                    self.remove_handles.append(handle)

    def set_forward_only(self, forward_only: bool):
        self.forward_only = forward_only

    def __del__(self):
        if self.supports_backward:
            for handle in self.remove_handles:
                handle.remove()

    def create_backward_hook(
        self, blocks: list[nn.Module], block_index: int
    ) -> Optional[callable]:  # type: ignore
        # -1 for 0-based index
        num_blocks_propagated = self.num_blocks - block_index - 1
        swapping = (
            num_blocks_propagated > 0 and num_blocks_propagated <= self.blocks_to_swap
        )
        waiting = block_index > 0 and block_index <= self.blocks_to_swap

        if not swapping and not waiting:
            return None

        # create  hook
        block_idx_to_cpu = self.num_blocks - num_blocks_propagated
        block_idx_to_cuda = self.blocks_to_swap - num_blocks_propagated
        block_idx_to_wait = block_index - 1

        def backward_hook(module, grad_input, grad_output):
            if self.debug:
                print(f"Backward hook for block {block_index}")

            if swapping:
                self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
            if waiting:
                self._wait_blocks_move(block_idx_to_wait)
            return None

        return backward_hook

    def prepare_block_devices_before_forward(self, blocks: list[nn.Module]):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        if self.debug:
            print(f"[{self.block_type}] Prepare block devices before forward")

        # Determine target memory for offloaded blocks
        cpu_device = torch.device("cpu")
        offload_device = cpu_device
        tmt = getattr(self, "target_memory_type", "auto")
        if tmt == "shared_gpu":
            offload_device = self.device
        elif tmt == "cpu":
            offload_device = cpu_device
        else:  # auto
            if getattr(self, "cpu_memory_priority", False):
                offload_device = cpu_device

        for b in blocks[0 : self.num_blocks - self.blocks_to_swap]:
            b.to(self.device)
            weights_to_device(b, self.device, non_blocking=self.non_blocking_transfers)

        for b in blocks[self.num_blocks - self.blocks_to_swap :]:
            b.to(self.device)  # ensure buffers on device
            if offload_device.type == "cpu":
                weights_to_device(
                    b,
                    offload_device,
                    non_blocking=False,
                    move_buffers=False,
                )

        synchronize_device(self.device)
        clean_memory_on_device(self.device)

    def wait_for_block(self, block_idx: int):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self._wait_blocks_move(block_idx)

    def submit_move_blocks_forward(self, blocks: list[nn.Module], block_idx: int):
        # check if blocks_to_swap is enabled
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        if not self.forward_only:
            # if backward is enabled, we do not swap blocks in forward pass more than blocks_to_swap,
            # because it should already be on GPU
            if block_idx >= self.blocks_to_swap:
                return

            block_idx_to_cpu = block_idx
            block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
            block_idx_to_cuda = block_idx_to_cuda % self.num_blocks
            self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
            return

        block_idx_to_cpu = block_idx

        if self.blocks_to_swap < (self.num_blocks // 2):
            # strategy 1: no wrap around
            # If the current block is in the middle blocks that are not swapped, do nothing
            if self.blocks_to_swap <= block_idx < self.num_blocks - self.blocks_to_swap:
                return

            if block_idx < self.blocks_to_swap:
                # move the next block to cuda
                block_idx_to_cuda = (
                    self.num_blocks - self.blocks_to_swap + block_idx
                ) % self.num_blocks
            else:
                # move the previous block to cuda
                block_idx_to_cuda = block_idx - (self.num_blocks - self.blocks_to_swap)
        else:
            # strategy 2: with wrap around
            block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
            block_idx_to_cuda = block_idx_to_cuda % self.num_blocks

        self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)


def create_enhanced_offloader(
    block_type: str,
    num_blocks: int,
    blocks_to_swap: int,
    device: torch.device,
    debug: bool = False,
    config_args: Optional[argparse.Namespace] = None,
) -> Offloader:
    """
    Create an Offloader with enhanced features based on configuration parameters.
    """
    enhanced_enabled = False
    pinned_memory_enabled = False
    non_blocking_transfers = True
    timing_enabled = False
    verbose_logging = False
    target_memory_type = "auto"
    cpu_memory_priority = False

    if config_args is not None:
        enhanced_enabled = getattr(config_args, "offload_enhanced_enabled", False)
        pinned_memory_enabled = getattr(
            config_args, "offload_pinned_memory_enabled", False
        )
        non_blocking_transfers = getattr(
            config_args, "offload_non_blocking_transfers", True
        )
        timing_enabled = getattr(config_args, "offload_timing_enabled", False)
        verbose_logging = getattr(config_args, "offload_verbose_logging", False)
        target_memory_type = getattr(config_args, "offload_target_memory_type", "auto")
        cpu_memory_priority = getattr(config_args, "offload_cpu_memory_priority", False)

    return Offloader(
        block_type=block_type,
        num_blocks=num_blocks,
        blocks_to_swap=blocks_to_swap,
        device=device,
        debug=debug,
        enhanced_enabled=enhanced_enabled,
        pinned_memory_enabled=pinned_memory_enabled,
        non_blocking_transfers=non_blocking_transfers,
        timing_enabled=timing_enabled,
        verbose_logging=verbose_logging,
        target_memory_type=target_memory_type,
        cpu_memory_priority=cpu_memory_priority,
    )


def create_enhanced_model_offloader(
    block_type: str,
    blocks: list[nn.Module],
    num_blocks: int,
    blocks_to_swap: int,
    supports_backward: bool,
    device: torch.device,
    debug: bool = False,
    config_args: Optional[argparse.Namespace] = None,
) -> ModelOffloader:
    """
    Create a ModelOffloader with enhanced features based on configuration parameters.
    """
    enhanced_enabled = False
    pinned_memory_enabled = False
    non_blocking_transfers = True
    timing_enabled = False
    verbose_logging = False
    target_memory_type = "auto"
    cpu_memory_priority = False

    if config_args is not None:
        enhanced_enabled = getattr(config_args, "offload_enhanced_enabled", False)
        pinned_memory_enabled = getattr(
            config_args, "offload_pinned_memory_enabled", False
        )
        non_blocking_transfers = getattr(
            config_args, "offload_non_blocking_transfers", True
        )
        timing_enabled = getattr(config_args, "offload_timing_enabled", False)
        verbose_logging = getattr(config_args, "offload_verbose_logging", False)
        target_memory_type = getattr(config_args, "offload_target_memory_type", "auto")
        cpu_memory_priority = getattr(config_args, "offload_cpu_memory_priority", False)

    return ModelOffloader(
        block_type=block_type,
        blocks=blocks,
        num_blocks=num_blocks,
        blocks_to_swap=blocks_to_swap,
        supports_backward=supports_backward,
        device=device,
        debug=debug,
        enhanced_enabled=enhanced_enabled,
        pinned_memory_enabled=pinned_memory_enabled,
        non_blocking_transfers=non_blocking_transfers,
        timing_enabled=timing_enabled,
        verbose_logging=verbose_logging,
        target_memory_type=target_memory_type,
        cpu_memory_priority=cpu_memory_priority,
    )
