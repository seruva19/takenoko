import torch
import random
import logging
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


def _fluxflow_perturb_single_video(
    video_frames: torch.Tensor,  # Expected shape (F, C, H, W) or (F, other_dims...)
    mode: str,
    frame_perturb_ratio: float,
    block_size: int,
    block_perturb_prob: float,
) -> torch.Tensor:
    """
    Applies FLUXFLOW temporal perturbation to a single video tensor.
    The first dimension is assumed to be the frame/time dimension.

    Args:
        video_frames: Input video tensor with shape (F, ...)
        mode: "frame" or "block"
        frame_perturb_ratio: Alpha - ratio of frames to shuffle in frame mode
        block_size: k - size of blocks in block mode
        block_perturb_prob: Beta - ratio of blocks to reorder in block mode (NOT probability!)
    """
    num_frames = video_frames.shape[0]
    if num_frames <= 1:  # Cannot perturb if 0 or 1 frame
        logger.info(f"FLUXFLOW: Skipping single video - only {num_frames} frame(s)")
        return video_frames

    if mode == "frame":
        # --- Frame-Level Perturbations (following paper Algorithm 1) ---
        num_to_shuffle = int(num_frames * frame_perturb_ratio)
        if num_to_shuffle <= 1:  # Need at least 2 frames to shuffle
            logger.info(
                f"FLUXFLOW: Frame mode - not enough frames to shuffle ({num_to_shuffle} calculated from {num_frames} frames)"
            )
            return video_frames

        logger.info(
            f"FLUXFLOW: Frame mode - shuffling {num_to_shuffle}/{num_frames} frames"
        )

        # Create index permutation following paper's approach
        all_indices = list(range(num_frames))
        indices_to_permute = sorted(random.sample(all_indices, num_to_shuffle))
        shuffled_subset = random.sample(indices_to_permute, len(indices_to_permute))

        new_order = list(range(num_frames))
        for i in range(num_to_shuffle):
            original_pos = indices_to_permute[i]
            new_val = shuffled_subset[i]
            new_order[original_pos] = new_val

        return video_frames[new_order]

    elif mode == "block":
        # --- Block-Level Perturbations (following paper Algorithm 1) ---
        if block_size <= 0:
            logger.warning(
                f"FLUXFLOW: Invalid block_size {block_size}, returning original video."
            )
            return video_frames
        if num_frames < block_size:  # Not enough frames for even one block
            logger.info(
                f"FLUXFLOW: Block mode - not enough frames ({num_frames}) for block size {block_size}"
            )
            return video_frames

        num_blocks = num_frames // block_size
        if num_blocks <= 1:  # Need at least 2 blocks to shuffle their order
            logger.info(
                f"FLUXFLOW: Block mode - only {num_blocks} block(s), need at least 2 for shuffling"
            )
            return video_frames

        # Calculate number of blocks to reorder (following paper's beta usage)
        num_blocks_to_reorder = int(num_blocks * block_perturb_prob)
        if num_blocks_to_reorder <= 1:
            logger.info(
                f"FLUXFLOW: Block mode - not enough blocks to reorder ({num_blocks_to_reorder} calculated from {num_blocks} blocks)"
            )
            return video_frames

        logger.info(
            f"FLUXFLOW: Block mode - reordering {num_blocks_to_reorder}/{num_blocks} blocks of size {block_size}"
        )

        # Create block index permutation following paper's approach
        all_block_indices = list(range(num_blocks))
        block_indices_to_permute = sorted(
            random.sample(all_block_indices, num_blocks_to_reorder)
        )
        shuffled_block_subset = random.sample(
            block_indices_to_permute, len(block_indices_to_permute)
        )

        new_block_order = list(range(num_blocks))
        for i in range(num_blocks_to_reorder):
            original_pos = block_indices_to_permute[i]
            new_val = shuffled_block_subset[i]
            new_block_order[original_pos] = new_val

        # Reconstruct video from reordered blocks
        v_perturbed_blocks = []
        for block_idx in new_block_order:
            start_frame = block_idx * block_size
            end_frame = start_frame + block_size
            v_perturbed_blocks.append(video_frames[start_frame:end_frame])

        # Concatenate reordered blocks
        perturbed_video = torch.cat(v_perturbed_blocks, dim=0)

        # Handle any remaining frames (if num_frames % block_size != 0)
        remaining_frames = num_frames % block_size
        if remaining_frames > 0:
            remaining_start = num_blocks * block_size
            remaining_frames_tensor = video_frames[remaining_start:]
            perturbed_video = torch.cat(
                [perturbed_video, remaining_frames_tensor], dim=0
            )

        return perturbed_video

    else:
        raise ValueError(
            f"FLUXFLOW: Unknown mode '{mode}'. Must be 'frame' or 'block'."
        )


def apply_fluxflow_to_batch(
    video_batch_tensor: torch.Tensor, fluxflow_config: dict
) -> torch.Tensor:
    """
    Applies FLUXFLOW temporal perturbation to a batch of video tensors.

    Args:
        video_batch_tensor: The input batch of video latents/frames.
        fluxflow_config: A dictionary with FLUXFLOW parameters:
            'mode': str, "frame" or "block"
            'frame_perturb_ratio': float, alpha - ratio for frame mode
            'block_size': int, k - block size for block mode
            'block_perturb_prob': float, beta - ratio of blocks to reorder (NOT probability!)
            'frame_dim_in_batch': int, index of the frame/time dimension in the batch tensor.
                                      Assumes batch dim is 0.

    Returns:
        The batch of temporally perturbed video tensors.
    """
    mode = fluxflow_config.get("mode", "frame")
    frame_perturb_ratio = fluxflow_config.get("frame_perturb_ratio", 0.25)
    block_size = fluxflow_config.get("block_size", 4)
    block_perturb_prob = fluxflow_config.get("block_perturb_prob", 0.5)
    frame_dim_in_batch = fluxflow_config.get(
        "frame_dim_in_batch", 2
    )  # Default for B,C,F,H,W

    logger.info(
        f"FLUXFLOW: apply_fluxflow_to_batch called - batch shape={video_batch_tensor.shape}, "
        f"mode={mode}, frame_dim={frame_dim_in_batch}, "
        f"frame_perturb_ratio={frame_perturb_ratio}, block_size={block_size}, block_perturb_ratio={block_perturb_prob}"
    )

    if not (0 <= frame_dim_in_batch < video_batch_tensor.ndim):
        raise ValueError(
            f"FLUXFLOW: frame_dim_in_batch {frame_dim_in_batch} is out of bounds "
            f"for tensor with {video_batch_tensor.ndim} dims."
        )

    # Permute to (B, F, ...) for easier iteration if frames are not already the second dimension (after batch dim 0)
    original_dims = list(range(video_batch_tensor.ndim))
    if (
        frame_dim_in_batch != 1
    ):  # If frame dimension is not at index 1 (after batch dim 0)
        permuted_dims = [original_dims[0]]  # Batch dim always first
        permuted_dims.append(
            original_dims[frame_dim_in_batch]
        )  # Bring frame dim to second position
        for i in range(
            1, len(original_dims)
        ):  # Add other dimensions, skipping original frame_dim_in_batch if it wasn't 0
            if i != frame_dim_in_batch:
                permuted_dims.append(original_dims[i])

        video_batch_bf_ellipsis = video_batch_tensor.permute(*permuted_dims)
    else:  # Frame dimension is already at index 1
        permuted_dims = original_dims  # No actual permutation needed for processing
        video_batch_bf_ellipsis = video_batch_tensor

    perturbed_video_list = []
    for i in range(video_batch_bf_ellipsis.shape[0]):  # Iterate over batch
        single_video_f_ellipsis = video_batch_bf_ellipsis[i]  # Shape (F, ...)
        perturbed_single_video = _fluxflow_perturb_single_video(
            single_video_f_ellipsis,
            mode,
            frame_perturb_ratio,
            block_size,
            block_perturb_prob,
        )
        perturbed_video_list.append(perturbed_single_video)

    perturbed_batch_bf_ellipsis = torch.stack(
        perturbed_video_list, dim=0
    )  # Stack along batch dimension

    # Permute back to original batch dimension order if a permutation occurred
    if frame_dim_in_batch != 1:
        # To get the inverse permutation:
        inv_permuted_dims = [0] * len(permuted_dims)
        for original_pos, permuted_val in enumerate(permuted_dims):
            inv_permuted_dims[permuted_val] = original_pos

        perturbed_batch_original_dims = perturbed_batch_bf_ellipsis.permute(
            *inv_permuted_dims
        )
    else:  # No permutation was done, so no inverse needed
        perturbed_batch_original_dims = perturbed_batch_bf_ellipsis

    return perturbed_batch_original_dims


def get_fluxflow_config_from_args(args) -> dict:
    """
    Extracts FLUXFLOW configuration parameters from args namespace
    and returns them as a dictionary.
    """
    return {
        "enable_fluxflow": getattr(args, "enable_fluxflow", False),
        "mode": getattr(args, "fluxflow_mode", "frame"),
        "frame_perturb_ratio": getattr(args, "fluxflow_frame_perturb_ratio", 0.25),
        "block_size": getattr(args, "fluxflow_block_size", 4),
        "block_perturb_prob": getattr(args, "fluxflow_block_perturb_prob", 0.5),
        "frame_dim_in_batch": getattr(args, "fluxflow_frame_dim_in_batch", 2),
    }


if __name__ == "__main__":
    # Example Usage and Basic Tests
    print("Running FLUXFLOW Augmentation tests...")

    # Test _fluxflow_perturb_single_video
    print("\n--- Testing _fluxflow_perturb_single_video ---")
    test_video_frames = torch.arange(10).view(10, 1)  # 10 frames, 1 channel
    print(f"Original video (frames 0-9):\n{test_video_frames.squeeze().tolist()}")

    # Frame mode test
    perturbed_frame_mode = _fluxflow_perturb_single_video(
        test_video_frames, "frame", 0.5, 4, 0.5
    )
    print(f"Frame mode (50% shuffle):\n{perturbed_frame_mode.squeeze().tolist()}")
    assert (
        not torch.equal(test_video_frames, perturbed_frame_mode)
        or test_video_frames.shape[0] <= 1
        or int(test_video_frames.shape[0] * 0.5) <= 1
    ), "Frame mode did not change video"

    # Block mode test (now using beta as ratio, not probability)
    # Use block_size=2 and beta=1.0 to ensure we get enough blocks to reorder
    perturbed_block_mode = _fluxflow_perturb_single_video(
        test_video_frames, "block", 0.25, 2, 1.0  # beta=1.0 means reorder all blocks
    )
    print(
        f"Block mode (block_size=2, beta=1.0):\n{perturbed_block_mode.squeeze().tolist()}"
    )
    num_blocks = test_video_frames.shape[0] // 2  # 10 // 2 = 5 blocks
    assert not torch.equal(test_video_frames, perturbed_block_mode) or (
        num_blocks <= 1
    ), "Block mode did not change video"

    # Test with fewer frames than block size
    short_video = torch.arange(2).view(2, 1)
    perturbed_short_block = _fluxflow_perturb_single_video(
        short_video, "block", 0.25, 3, 1.0
    )
    assert torch.equal(
        short_video, perturbed_short_block
    ), "Block mode changed video with < block_size frames"
    print(
        f"Block mode (short video, block_size=3): {perturbed_short_block.squeeze().tolist()}"
    )

    # Test apply_fluxflow_to_batch
    print("\n--- Testing apply_fluxflow_to_batch ---")
    # B, C, F, H, W -> Frame dim = 2
    test_batch_bcfhw = torch.stack(
        [torch.arange(i * 10, (i + 1) * 10).view(1, 10, 1, 1) for i in range(2)], dim=0
    )  # Batch=2, C=1, F=10, H=1, W=1
    print(
        f"Original batch (B,C,F,H,W), frame_dim=2, Video 0 (0-9), Video 1 (10-19):\n{test_batch_bcfhw.squeeze().tolist()}"
    )

    config_bcfhw = {
        "mode": "frame",
        "frame_perturb_ratio": 0.5,
        "block_size": 2,
        "block_perturb_prob": 1.0,  # Now interpreted as ratio, not probability
        "frame_dim_in_batch": 2,
    }
    perturbed_batch_bcfhw = apply_fluxflow_to_batch(
        test_batch_bcfhw.clone(), config_bcfhw
    )
    print(
        f"Perturbed batch (B,C,F,H,W), frame_dim=2:\n{perturbed_batch_bcfhw.squeeze().tolist()}"
    )
    assert not torch.equal(
        test_batch_bcfhw, perturbed_batch_bcfhw
    ), "Batch B,C,F,H,W did not change"

    # B, F, C, H, W -> Frame dim = 1
    test_batch_bfchw = test_batch_bcfhw.permute(0, 2, 1, 3, 4)  # Permute to B,F,C,H,W
    print(
        f"Original batch (B,F,C,H,W), frame_dim=1, Video 0 (0-9), Video 1 (10-19):\n{test_batch_bfchw.squeeze().tolist()}"
    )
    config_bfchw = {
        "mode": "block",
        "frame_perturb_ratio": 0.3,
        "block_size": 2,
        "block_perturb_prob": 1.0,  # Now interpreted as ratio, not probability
        "frame_dim_in_batch": 1,
    }
    perturbed_batch_bfchw = apply_fluxflow_to_batch(
        test_batch_bfchw.clone(), config_bfchw
    )
    print(
        f"Perturbed batch (B,F,C,H,W), frame_dim=1:\n{perturbed_batch_bfchw.squeeze().tolist()}"
    )
    assert not torch.equal(
        test_batch_bfchw, perturbed_batch_bfchw
    ), "Batch B,F,C,H,W did not change"
    assert (
        perturbed_batch_bfchw.shape == test_batch_bfchw.shape
    ), "Shape changed for B,F,C,H,W"

    print("\nAll basic tests completed.")
