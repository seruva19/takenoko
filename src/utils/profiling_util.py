import torch
import os
from common.logger import get_logger

logger = get_logger(__name__)


class TrainingProfiler:
    """Helper class for PyTorch Profiler integration."""

    def __init__(self, args, log_dir):
        self.enabled = getattr(args, "profiling_enabled", False)
        if not self.enabled:
            self.profiler = None
            return

        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.wait_steps = getattr(args, "profiling_wait_steps", 5)
        self.warmup_steps = getattr(args, "profiling_warmup_steps", 2)
        self.active_steps = getattr(args, "profiling_active_steps", 3)
        self.repeat = getattr(args, "profiling_repeat", 1)
        self.skip_first = getattr(args, "profiling_skip_first", 10)

        self.record_shapes = getattr(args, "profiling_record_shapes", True)
        self.profile_memory = getattr(args, "profiling_profile_memory", True)
        self.with_stack = getattr(args, "profiling_with_stack", True)
        self.with_flops = getattr(args, "profiling_with_flops", True)

        # Create tensorboard trace handler
        self.handler = torch.profiler.tensorboard_trace_handler(self.log_dir)

        logger.info(
            f"🚀 Profiling enabled: wait={self.wait_steps}, warmup={self.warmup_steps}, "
            f"active={self.active_steps}, repeat={self.repeat}, skip_first={self.skip_first}"
        )
        logger.info(f"📁 Profiler logs will be saved to: {self.log_dir}")

        self.profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=self.wait_steps,
                warmup=self.warmup_steps,
                active=self.active_steps,
                repeat=self.repeat,
                skip_first=self.skip_first,
            ),
            on_trace_ready=self.handler,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops,
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        )

    def start(self):
        if self.profiler:
            self.profiler.start()

    def stop(self):
        if self.profiler:
            self.profiler.stop()

    def step(self):
        if self.profiler:
            self.profiler.step()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
