import torch
from safetensors.torch import load_file, save_file
import argparse
import os
import logging
import numpy as np
import re


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_lora(path):
    """Load LoRA from safetensors or pytorch format"""
    if path.endswith(".safetensors"):
        return load_file(path)
    else:
        return torch.load(path, map_location="cpu")


def save_lora(tensors, path):
    """Save LoRA in safetensors format"""
    save_file(tensors, path)


def extract_block_index(base_key):
    """Extract numeric block index from key name."""
    # Common patterns for block indices in LoRA keys
    patterns = [
        r'\.(\d+)\.',  # General pattern like "transformer.h.0." or "blocks.5."
        r'_(\d+)_',    # Pattern with underscores like "block_12_"
        r'block(\d+)', # Pattern like "block12"
        r'layer(\d+)', # Pattern like "layer5"
        r'h\.(\d+)',   # Transformer pattern like "h.0"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, base_key)
        if match:
            return int(match.group(1))
    
    return None


def is_block_targeted(base_key, target_blocks=None, target_indices=None):
    """Check if block should be targeted based on name patterns and/or indices."""
    # Backward compatibility: if target_blocks is a list and target_indices is None
    # behave like the original function
    if target_indices is None and target_blocks is not None:
        if not target_blocks:
            return False
        return any(pattern.lower() in base_key.lower() for pattern in target_blocks)
    
    # New logic: If both criteria are specified, BOTH must match
    if target_indices is not None and target_blocks is not None:
        block_index = extract_block_index(base_key)
        index_match = block_index is not None and block_index in target_indices
        pattern_match = any(pattern.lower() in base_key.lower() for pattern in target_blocks)
        return index_match and pattern_match
    
    # If only indices are specified
    if target_indices is not None:
        block_index = extract_block_index(base_key)
        return block_index is not None and block_index in target_indices
    
    # If only patterns are specified
    if target_blocks:
        return any(pattern.lower() in base_key.lower() for pattern in target_blocks)
    
    return False


def analyze_pattern_matches(base_keys, target_blocks=None, target_indices=None):
    """Analyze which patterns match and provide detailed feedback."""
    if not target_blocks and not target_indices:
        return
    
    logging.info("=== PATTERN MATCHING ANALYSIS ===")
    
    # Analyze pattern matches
    if target_blocks:
        logging.info(f"Searching for patterns: {target_blocks}")
        for pattern in target_blocks:
            matches = [key for key in base_keys if pattern.lower() in key.lower()]
            if matches:
                logging.info(f"  ✓ Pattern '{pattern}' found in {len(matches)} layers")
                # Show first few examples
                examples = matches[:3]
                if len(matches) > 3:
                    examples_str = ", ".join(examples) + f" (and {len(matches)-3} more)"
                else:
                    examples_str = ", ".join(examples)
                logging.info(f"    Examples: {examples_str}")
            else:
                logging.warning(f"  ✗ Pattern '{pattern}' NOT FOUND in any layers!")
    
    # Analyze index matches
    if target_indices:
        logging.info(f"Searching for indices: {target_indices}")
        found_indices = set()
        for key in base_keys:
            idx = extract_block_index(key)
            if idx is not None:
                found_indices.add(idx)
        
        for target_idx in target_indices:
            if target_idx in found_indices:
                matches = [key for key in base_keys if extract_block_index(key) == target_idx]
                logging.info(f"  ✓ Index {target_idx} found in {len(matches)} layers")
            else:
                logging.warning(f"  ✗ Index {target_idx} NOT FOUND in any layers!")
        
        logging.info(f"Available indices: {sorted(found_indices)}")
    
    # Show combined results
    if target_blocks and target_indices:
        logging.info("Combined targeting (both pattern AND index must match):")
        targeted_keys = [k for k in base_keys if is_block_targeted(k, target_blocks, target_indices)]
        logging.info(f"  Final matches: {len(targeted_keys)} layers")
    elif target_blocks:
        targeted_keys = [k for k in base_keys if is_block_targeted(k, target_blocks, target_indices)]
        logging.info(f"  Pattern-only matches: {len(targeted_keys)} layers")
    elif target_indices:
        targeted_keys = [k for k in base_keys if is_block_targeted(k, target_blocks, target_indices)]
        logging.info(f"  Index-only matches: {len(targeted_keys)} layers")
    
    logging.info("=== END ANALYSIS ===")
    return len(targeted_keys) > 0


def get_block_info(lora_dict):
    """Analyze and display block information for debugging."""
    base_keys = get_base_keys(lora_dict)
    
    block_info = {}
    for key in base_keys:
        index = extract_block_index(key)
        if index is not None:
            if index not in block_info:
                block_info[index] = []
            block_info[index].append(key)
    
    logging.info("Block analysis:")
    for index in sorted(block_info.keys()):
        sample_keys = block_info[index][:3]  # Show first 3 keys as examples
        logging.info(f"  Index {index}: {len(block_info[index])} layers (e.g., {', '.join(sample_keys)})")
    
    return block_info


def get_base_keys(lora_dict):
    """Extract base keys from LoRA dictionary"""
    base_keys = set()
    for k in lora_dict.keys():
        if k.endswith(".lora_down.weight"):
            base_keys.add(k.replace(".lora_down.weight", ""))
    return base_keys


def show_available_patterns(base_keys, max_examples=20):
    """Show available patterns in the LoRA for debugging."""
    logging.info("=== AVAILABLE LAYER PATTERNS ===")
    
    # Extract unique patterns
    patterns = set()
    for key in base_keys:
        parts = key.split('.')
        for part in parts:
            patterns.add(part)
    
    # Show patterns
    sorted_patterns = sorted(patterns)
    logging.info(f"Available patterns ({len(sorted_patterns)} total):")
    for pattern in sorted_patterns:
        # Count how many layers contain this pattern
        count = sum(1 for key in base_keys if pattern in key)
        logging.info(f"  {pattern} (appears in {count} layers)")
    
    # Show sample layer names
    logging.info(f"\nSample layer names (first {max_examples}):")
    for i, key in enumerate(sorted(base_keys)):
        if i >= max_examples:
            logging.info(f"  ... and {len(base_keys) - max_examples} more")
            break
        logging.info(f"  {key}")
    
    logging.info("=== END AVAILABLE PATTERNS ===")


def compute_lora_delta(lora_dict, base_key, device="cpu"):
    """Compute the delta matrix for a LoRA layer: B @ A"""
    A_key = f"{base_key}.lora_down.weight"
    B_key = f"{base_key}.lora_up.weight"
    
    if A_key not in lora_dict or B_key not in lora_dict:
        raise KeyError(f"Missing LoRA weights for {base_key}")
    
    A = lora_dict[A_key].to(torch.float32).to(device)
    B = lora_dict[B_key].to(torch.float32).to(device)
    
    return torch.matmul(B, A)


def decompose_delta_to_lora(delta, rank, dtype=torch.float16):
    """Decompose delta matrix back to LoRA A/B matrices using SVD"""
    try:
        # Ensure delta is float32 for SVD
        delta = delta.float()
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    except Exception as e:
        logging.warning(f"torch.linalg.svd failed: {e}")
        raise
    
    # Truncate to desired rank
    rank = min(rank, len(S))
    U = U[:, :rank]
    S = S[:rank]
    Vh = Vh[:rank, :]
    
    # Distribute singular values more evenly
    # Put more weight in A (down) to match typical LoRA conventions
    A_new = (Vh * S.unsqueeze(1)).to(dtype).contiguous().cpu()
    B_new = U.to(dtype).contiguous().cpu()
    
    return A_new, B_new


def decompose_delta_to_lora_randomized(delta, rank, dtype=torch.float16, n_iter=2, oversamples=10):
    """Faster, approximate SVD using randomization."""
    try:
        delta = delta.float()
        m, n = delta.size()
        
        # Create a random projection matrix
        proj_rank = min(rank + oversamples, n)
        rand_proj = torch.randn((n, proj_rank), device=delta.device, dtype=delta.dtype)
        
        # Project the matrix
        Y = delta @ rand_proj
        
        # Power iterations for better accuracy
        for _ in range(n_iter):
            Q, _ = torch.linalg.qr(Y)
            Z = delta.T @ Q
            Y = delta @ Z
            
        Q, _ = torch.linalg.qr(Y)
        
        # Project the original matrix onto the low-rank basis
        B_proj = Q.T @ delta
        
        # Perform SVD on the much smaller projected matrix
        U_proj, S, Vh = torch.linalg.svd(B_proj, full_matrices=False)
        U = Q @ U_proj

    except Exception as e:
        logging.warning(f"Randomized SVD failed: {e}")
        raise

    rank = min(rank, len(S))
    U = U[:, :rank]
    S = S[:rank]
    Vh = Vh[:rank, :]

    A_new = (torch.diag(S) @ Vh).to(dtype).contiguous().cpu()
    B_new = U.to(dtype).contiguous().cpu()

    return A_new, B_new


def create_zero_lora(reference_lora, base_key, dtype=torch.float16):
    """Create zero LoRA matrices with same shape as reference"""
    A_key = f"{base_key}.lora_down.weight"
    B_key = f"{base_key}.lora_up.weight"
    
    A_ref = reference_lora[A_key]
    B_ref = reference_lora[B_key]
    
    A_zero = torch.zeros_like(A_ref, dtype=dtype).cpu()
    B_zero = torch.zeros_like(B_ref, dtype=dtype).cpu()
    
    return A_zero, B_zero


def process_targeted_and_nullified_lora(
    lora_a,
    lora_b, 
    target_blocks,
    rank=16,
    alpha=1.0,
    dtype=torch.float16,
    device="cpu",
    magnitude_threshold=1e-6,
    fast_svd=False,
    target_indices=None,
    auto_threshold_percentile=None,
    target_norm=1.0,
    analyze_blocks=False,
    exclude_non_targeted=False
):
    """
    Create a LoRA that:
    1. For targeted blocks: contains the difference (lora_a - lora_b)
    2. For non-targeted blocks: contains zeros (nullified) or excluded completely
    """
    
    base_keys_a = get_base_keys(lora_a)
    base_keys_b = get_base_keys(lora_b)
    
    # Use intersection to ensure both LoRAs have the keys
    common_keys = base_keys_a.intersection(base_keys_b)
    
    if not common_keys:
        raise ValueError("No common keys found between LoRAs")

    # Analyze blocks if requested
    if analyze_blocks:
        get_block_info(lora_a)

    # ENHANCED: Analyze pattern matches and show detailed feedback
    patterns_found = analyze_pattern_matches(common_keys, target_blocks, target_indices)
    
    if not patterns_found:
        logging.error("❌ NO MATCHING PATTERNS FOUND!")
        show_available_patterns(common_keys)
        raise ValueError("No layers match target patterns - see available patterns above")

    # Auto-threshold calculation
    if auto_threshold_percentile is not None:
        logging.info(f"Automatically calculating magnitude threshold to discard the weakest {auto_threshold_percentile}% of layers...")
        all_magnitudes = []
        for base_key in common_keys:
            try:
                delta_a_mag = compute_lora_delta(lora_a, base_key, device)
                delta_b_mag = compute_lora_delta(lora_b, base_key, device)
                difference_delta_mag = delta_a_mag - delta_b_mag
                magnitude = torch.norm(difference_delta_mag).item()
                all_magnitudes.append(magnitude)
            except Exception as e:
                logging.warning(f"Skipping {base_key} during magnitude analysis: {e}")
                continue
        
        if all_magnitudes:
            magnitude_threshold = np.percentile(all_magnitudes, auto_threshold_percentile)
            logging.info(f"Auto-calculated magnitude threshold: {magnitude_threshold:.4e}")
        else:
            logging.warning("Could not calculate auto-threshold, falling back to default.")
    
    # Check if any keys match the target patterns
    if target_indices is not None or target_blocks:
        targeted_keys = [k for k in common_keys if is_block_targeted(k, target_blocks, target_indices)]
    else:
        targeted_keys = [k for k in common_keys if is_block_targeted(k, target_blocks)]
    
    if not targeted_keys:
        logging.error("❌ NO LAYERS MATCH THE TARGET PATTERNS!")
        logging.error("This should not happen after pattern analysis - please report this bug!")
        raise ValueError("No layers match target patterns")
        
    logging.info(f"Processing {len(common_keys)} common layers")
    logging.info(f"Target blocks: {target_blocks}")
    if target_indices is not None:
        logging.info(f"Target indices: {target_indices}")
    logging.info(f"Targeted layers: {len(targeted_keys)}")
    
    # Log the processing mode
    if exclude_non_targeted:
        logging.info("Mode: EXCLUDE non-targeted blocks (smaller file size)")
    else:
        logging.info("Mode: NULLIFY non-targeted blocks (same structure as input)")
    
    # Log which SVD method is being used
    if fast_svd:
        logging.info("Using FAST (Randomized) SVD. Quality may be slightly lower.")
        decompose_func = decompose_delta_to_lora_randomized
    else:
        logging.info("Using FULL (Exact) SVD for maximum quality.")
        decompose_func = decompose_delta_to_lora

    # Auto-alpha calculation
    if alpha is None:
        delta_norms = []
        for base_key in targeted_keys:
            try:
                delta_a = compute_lora_delta(lora_a, base_key, device)
                delta_b = compute_lora_delta(lora_b, base_key, device)
                delta = delta_a - delta_b
                norm = torch.norm(delta).item()
                if norm >= magnitude_threshold:
                    delta_norms.append(norm)
            except Exception as e:
                logging.warning(f"Skipping {base_key} during alpha analysis: {e}")
                continue

        if not delta_norms:
            raise ValueError("Could not compute alpha: all delta magnitudes too small")
        avg_norm = sum(delta_norms) / len(delta_norms)
        alpha = target_norm / (avg_norm + 1e-8)
        logging.info(f"Auto-tuned alpha: {alpha:.4f} (target norm {target_norm}, avg delta {avg_norm:.4f})")
    
    result_lora = {}
    targeted_count = 0
    nullified_count = 0
    excluded_count = 0
    skipped_count = 0
    
    for base_key in common_keys:
        try:
            if target_indices is not None:
                is_targeted = is_block_targeted(base_key, target_blocks, target_indices)
            else:
                is_targeted = is_block_targeted(base_key, target_blocks)
            
            if is_targeted:
                # TARGETED: Compute difference (lora_a - lora_b)
                delta_a = compute_lora_delta(lora_a, base_key, device)
                delta_b = compute_lora_delta(lora_b, base_key, device)
                
                difference_delta = delta_a - delta_b
                
                # Check if difference is meaningful
                magnitude = torch.norm(difference_delta).item()
                if magnitude < magnitude_threshold:
                    if exclude_non_targeted:
                        # Skip completely - don't add to result
                        skipped_count += 1
                        continue
                    else:
                        logging.debug(f"Skipping {base_key}: magnitude {magnitude:.2e} below threshold")
                        # Create zero LoRA for consistency
                        A_new, B_new = create_zero_lora(lora_a, base_key, dtype)
                        skipped_count += 1
                else:
                    # IMPORTANT FIX: Don't divide by alpha here!
                    # The alpha scaling should be handled during inference, not during creation
                    # If you want to pre-scale, multiply by alpha instead of dividing
                    scaled_delta = difference_delta * alpha  # Scale up the difference
                    A_new, B_new = decompose_func(scaled_delta, rank, dtype)
                    targeted_count += 1
                    block_idx = extract_block_index(base_key)
                    logging.info(f"TARGETED {base_key} (index {block_idx}): magnitude={magnitude:.2e}")
                    
            else:
                if exclude_non_targeted:
                    # Skip completely - don't add to result
                    excluded_count += 1
                    block_idx = extract_block_index(base_key)
                    logging.debug(f"EXCLUDED {base_key} (index {block_idx}): not included in output")
                    continue
                else:
                    # NON-TARGETED: Create zero LoRA (nullified)
                    A_new, B_new = create_zero_lora(lora_a, base_key, dtype)
                    nullified_count += 1
                    block_idx = extract_block_index(base_key)
                    logging.debug(f"NULLIFIED {base_key} (index {block_idx}): zero weights")
            
            # Store results (only reached if not excluded)
            A_key = f"{base_key}.lora_down.weight"
            B_key = f"{base_key}.lora_up.weight"
            result_lora[A_key] = A_new
            result_lora[B_key] = B_new
            
        except Exception as e:
            logging.error(f"Error processing {base_key}: {e}")
            continue

    # Log results based on mode
    if exclude_non_targeted:
        logging.info(f"Results: {targeted_count} targeted, {excluded_count} excluded, {skipped_count} skipped due to low magnitude")
    else:
        logging.info(f"Results: {targeted_count} targeted, {nullified_count} nullified, {skipped_count} skipped")
    
    if targeted_count == 0:
        logging.error("❌ NO LAYERS WERE SUCCESSFULLY PROCESSED!")
        return {}
    
    return result_lora


def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Create LoRA with targeted differences and nullified non-targeted blocks"
    )
    parser.add_argument("--lora_a", required=True, help="First LoRA path")
    parser.add_argument("--lora_b", required=True, help="Second LoRA path") 
    parser.add_argument("--output", required=True, help="Output LoRA path")
    parser.add_argument("--target_blocks", nargs="*", default=None,
                       help="Block patterns to target (e.g., attn up_blocks)")
    parser.add_argument("--target_indices", nargs="*", type=int, default=None,
                       help="Block indices to target (e.g., 0 1 2 3)")
    parser.add_argument("--exclude_non_targeted", action='store_true', default=False,
                       help="Exclude non-targeted blocks completely from output (smaller file size). Default is to nullify them (zero weights).")
    parser.add_argument("--analyze_blocks", action='store_true', default=False,
                       help="Analyze and display block structure before processing.")
    parser.add_argument("--rank", type=int, default=16, help="SVD rank")
    parser.add_argument("--alpha", type=float, default=None, help="Scaling factor (leave empty for auto-tuning)")
    parser.add_argument("--target_norm", type=float, default=1.0, help="Target norm for auto-alpha calculation")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--device", default="cpu", help="Computation device")
    
    threshold_group = parser.add_mutually_exclusive_group()
    threshold_group.add_argument("--magnitude_threshold", type=float, default=1e-7,
                               help="Manually set threshold for skipping layers.")
    threshold_group.add_argument("--auto_threshold_percentile", type=float, default=None,
                               help="Automatically set threshold by discarding the weakest X%% of layers (e.g., 25.0). Overrides --magnitude_threshold.")
    
    parser.add_argument("--fast_svd", action='store_true', default=False,
                       help="Use faster, approximate SVD for quick previews.")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.target_blocks and not args.target_indices:
        raise ValueError("Must specify at least one of --target_blocks or --target_indices")
    
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    
    # Load LoRAs
    logging.info(f"Loading LoRA A: {args.lora_a}")
    lora_a = load_lora(args.lora_a)
    
    logging.info(f"Loading LoRA B: {args.lora_b}")
    lora_b = load_lora(args.lora_b)
    
    # Process
    result = process_targeted_and_nullified_lora(
        lora_a=lora_a,
        lora_b=lora_b,
        target_blocks=args.target_blocks,
        target_indices=args.target_indices,
        rank=args.rank,
        alpha=args.alpha,
        dtype=dtype,
        device=args.device,
        magnitude_threshold=args.magnitude_threshold,
        auto_threshold_percentile=args.auto_threshold_percentile,
        target_norm=args.target_norm,
        fast_svd=args.fast_svd,
        analyze_blocks=args.analyze_blocks,
        exclude_non_targeted=args.exclude_non_targeted
    )
    
    if not result:
        logging.error("❌ NO LAYERS WERE PROCESSED!")
        return
    
    # Save result
    save_lora(result, args.output)
    logging.info(f"✅ Result saved to {args.output}")
    print(f"✅ Created LoRA with {len(result)//2} layers")
    if args.target_blocks and args.target_indices:
        print(f"Targeted: {args.target_blocks} within indices {args.target_indices}")
    elif args.target_blocks:
        print(f"Targeted blocks: {args.target_blocks}")
    elif args.target_indices:
        print(f"Targeted indices: {args.target_indices}")
    
    if args.exclude_non_targeted:
        print("Non-targeted blocks: excluded (not included in output)")
    else:
        print("Non-targeted blocks: nullified (zero weights)")


if __name__ == "__main__":
    main()