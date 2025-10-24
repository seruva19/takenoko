from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def generate_crop_positions(
    frame_count: int,
    target_frames: Optional[List[int]],
    mode: str,
    frame_stride: Optional[int] = 1,
    frame_sample: Optional[int] = 1,
    max_frames: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """
    Generate a list of (start_index, window_size) pairs for extracting frame windows.

    Args:
        frame_count: Total number of frames available in the video (F).
        target_frames: List of target window sizes (each is the number of frames in a clip).
        mode: Strategy name. Supported:
            - "head"
            - "middle"
            - "chunk" (non-overlapping contiguous windows)
            - "slide" (sliding window with stride)
            - "slide_end" (sliding with stride; ensure last window ends at the last frame; if video is shorter than target, return a single full-length window)
            - "uniform" (fixed count of evenly spaced starts)
            - "multiple_overlapping" (cover video with minimal number of windows, end-aligned)
            - "adaptive" (intelligently handles variable clip lengths: if shorter than target, takes whole clip; if longer, takes multiple fragments with overlap, max_frames caps the effective video length)
            - "uniform_adaptive" (uniform sampling that accepts short videos: if shorter than target, takes whole clip; if longer, uses uniform sampling, max_frames caps the effective video length)
            - "full" (use up to max_frames, rounded to N*4+1)
        frame_stride: Stride for "slide"/"slide_end" modes.
        frame_sample: Number of samples for "uniform" mode.
        max_frames: Maximum frames for "full" mode.

    Returns:
        List of (start_index, window_size) tuples.
    """
    if not target_frames:
        return []

    crop_pos_and_frames: List[Tuple[int, int]] = []

    normalized_mode = mode

    for target_frame in target_frames:
        if frame_count < target_frame and normalized_mode not in [
            "adaptive",
            "uniform_adaptive",
        ]:
            # Not enough frames to extract this window size (except for adaptive and uniform_adaptive modes)
            continue

        if normalized_mode == "head":
            crop_pos_and_frames.append((0, target_frame))

        elif normalized_mode == "middle":
            start = (frame_count - target_frame) // 2
            crop_pos_and_frames.append((start, target_frame))

        elif normalized_mode == "chunk":
            for i in range(0, frame_count, target_frame):
                if i + target_frame <= frame_count:
                    crop_pos_and_frames.append((i, target_frame))

        elif normalized_mode == "slide":
            stride = frame_stride or 1
            for i in range(0, frame_count - target_frame + 1, stride):
                crop_pos_and_frames.append((i, target_frame))

        elif normalized_mode == "slide_end":
            stride = frame_stride or 1
            last_start = frame_count - target_frame
            if last_start < 0:
                # Video shorter than target; take full length as a single window
                crop_pos_and_frames.append((0, frame_count))
            else:
                for i in range(0, last_start + 1, stride):
                    crop_pos_and_frames.append((i, target_frame))
                if not crop_pos_and_frames or crop_pos_and_frames[-1][0] != last_start:
                    crop_pos_and_frames.append((last_start, target_frame))

        elif normalized_mode == "uniform":
            samples = frame_sample or 1
            starts = np.linspace(0, frame_count - target_frame, samples, dtype=int)
            for i in starts:
                crop_pos_and_frames.append((int(i), target_frame))

        elif normalized_mode == "multiple_overlapping":
            # Cover the whole video with minimal number of clips, end-aligned (may overlap)
            num_clips = ((frame_count - 1) // target_frame) + 1
            starts = np.linspace(0, frame_count - target_frame, num_clips, dtype=int)
            for i in starts:
                crop_pos_and_frames.append((int(i), target_frame))

        elif normalized_mode == "adaptive":
            # Adaptive method: handles variable clip lengths intelligently
            # Apply max_frames cap to effective frame count if specified
            effective_frame_count = frame_count
            if max_frames is not None and max_frames > 0:
                effective_frame_count = min(frame_count, max_frames)

            # Skip if no frames available
            if effective_frame_count <= 0:
                continue

            if effective_frame_count < target_frame:
                # If effective clip is shorter than target, take it as whole
                crop_pos_and_frames.append((0, effective_frame_count))
            else:
                # Calculate how many fragments we can fit in the effective length
                num_fragments = effective_frame_count // target_frame
                # num_fragments will be at least 1 when effective_frame_count >= target_frame
                if num_fragments == 1:
                    # Single fragment - take from the beginning
                    crop_pos_and_frames.append((0, target_frame))
                else:
                    # Multiple fragments - distribute evenly across effective length
                    for i in range(num_fragments):
                        start = int(
                            i
                            * (effective_frame_count - target_frame)
                            / (num_fragments - 1)
                        )
                        crop_pos_and_frames.append((start, target_frame))

        elif normalized_mode == "uniform_adaptive":
            # Uniform-like method that accepts short videos (like adaptive)
            # Apply max_frames cap to effective frame count if specified
            effective_frame_count = frame_count
            if max_frames is not None and max_frames > 0:
                effective_frame_count = min(frame_count, max_frames)

            # Skip if no frames available
            if effective_frame_count <= 0:
                continue

            if effective_frame_count < target_frame:
                # If effective clip is shorter than target, take it as whole
                crop_pos_and_frames.append((0, effective_frame_count))
            else:
                # Use uniform sampling on the effective length
                samples = frame_sample or 1
                starts = np.linspace(
                    0, effective_frame_count - target_frame, samples, dtype=int
                )
                for i in starts:
                    crop_pos_and_frames.append((int(i), target_frame))

        elif normalized_mode == "full":
            if max_frames is None or max_frames <= 0:
                use_frames = frame_count
            else:
                use_frames = min(frame_count, max_frames)
            # round to N*4+1 as per original implementation
            use_frames = (use_frames - 1) // 4 * 4 + 1
            crop_pos_and_frames.append((0, use_frames))

        else:
            raise ValueError(f"frame_extraction {mode} is not supported")

    return crop_pos_and_frames
