#!/usr/bin/env python3

"""
Convert Takenoko VB-LoRA checkpoints to regular LoRA checkpoints.

This is intended for inference compatibility with loaders that expect
`lora_down.weight` / `lora_up.weight` tensors (e.g. common ComfyUI LoRA paths).

Notes:
- Input VB-LoRA checkpoints can be either:
  - full logits: `*.vblora_logits_A/B`
  - compressed logits: `*.vblora_logits_*_topk_indices/_topk_weights`
- The output is regular LoRA tensors only for converted modules.
- Conversion is inference-oriented; it does not preserve VB-LoRA trainability.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, Optional, Tuple

import torch


logger = logging.getLogger("convert_vblora_to_lora")

TOPK_INDICES_SUFFIX = "_topk_indices"
TOPK_WEIGHTS_SUFFIX = "_topk_weights"
DEFAULT_TOPK = 2


def _load_state_dict(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file

        metadata: Dict[str, str] = {}
        with safe_open(path, framework="pt", device="cpu") as handle:
            raw_metadata = handle.metadata()
            if raw_metadata:
                metadata = dict(raw_metadata)
        return load_file(path), metadata

    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):
        if "state_dict" in raw and isinstance(raw["state_dict"], dict):
            return raw["state_dict"], {}
        return raw, {}
    raise ValueError(f"Unsupported checkpoint structure in {path!r}.")


def _save_state_dict(path: str, state_dict: Dict[str, torch.Tensor], metadata: Dict[str, str]) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        from safetensors.torch import save_file

        save_file(state_dict, path, metadata=metadata)
        return
    torch.save(state_dict, path)


def _decode_topk_logits(
    state_dict: Dict[str, torch.Tensor],
    num_vectors_hint: Optional[int] = None,
) -> Tuple[Dict[str, torch.Tensor], Optional[int]]:
    decoded = dict(state_dict)
    inferred_topk: Optional[int] = None

    num_vectors = int(max(1, num_vectors_hint or 1))
    bank = decoded.get("vblora_vector_bank", None)
    if isinstance(bank, torch.Tensor) and bank.ndim == 2:
        num_vectors = int(bank.shape[0])

    keys = list(decoded.keys())
    for key in keys:
        if not key.endswith(TOPK_INDICES_SUFFIX):
            continue
        if key not in decoded:
            continue

        weights_key = key[: -len(TOPK_INDICES_SUFFIX)] + TOPK_WEIGHTS_SUFFIX
        if weights_key not in decoded:
            raise KeyError(f"Missing {weights_key!r} for compressed key {key!r}.")

        indices = decoded[key].to(torch.long)
        topk_weights = decoded[weights_key]
        if indices.ndim == 0 or topk_weights.ndim == 0:
            raise ValueError(f"Invalid compressed tensor rank for key {key!r}.")
        if tuple(indices.shape[:-1]) != tuple(topk_weights.shape[:-1]):
            raise ValueError(f"Shape mismatch between {key!r} and {weights_key!r}.")

        topk = int(indices.shape[-1])
        if inferred_topk is None:
            inferred_topk = topk
        elif inferred_topk != topk:
            logger.warning(
                "Detected mixed compressed top-k values (%s vs %s). Using per-module shape.",
                inferred_topk,
                topk,
            )

        if topk_weights.shape[-1] == topk - 1:
            full_weights = torch.cat(
                [topk_weights, 1 - topk_weights.sum(dim=-1, keepdim=True)],
                dim=-1,
            )
        elif topk_weights.shape[-1] == topk:
            full_weights = topk_weights
        else:
            raise ValueError(
                f"Unexpected topk_weights last dim for {weights_key!r}: "
                f"got {topk_weights.shape[-1]}, expected {topk - 1} or {topk}."
            )

        if indices.numel() > 0:
            max_index = int(indices.max().item()) + 1
            num_vectors = max(num_vectors, max_index)

        # Clamp avoids NaN from log(0); values may still become very negative.
        topk_logits = torch.log(full_weights.clamp_min(1e-12))
        recovered = torch.full(
            [*topk_logits.shape[:-1], num_vectors],
            fill_value=float("-inf"),
            dtype=topk_logits.dtype,
            device=topk_logits.device,
        ).scatter(-1, indices, topk_logits)

        original_key = key[: -len(TOPK_INDICES_SUFFIX)]
        decoded[original_key] = recovered
        del decoded[key]
        del decoded[weights_key]

    return decoded, inferred_topk


def _materialize_matrix(logits: torch.Tensor, vector_bank: torch.Tensor, topk: int) -> torch.Tensor:
    effective_topk = max(1, min(int(topk), int(logits.shape[-1])))
    topk_logits, indices = logits.topk(effective_topk, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1)
    return (topk_weights.unsqueeze(-1) * vector_bank[indices]).sum(dim=-2)


def _rename_module_name(name: str, from_prefix: str, to_prefix: str) -> str:
    if from_prefix and name.startswith(from_prefix):
        return to_prefix + name[len(from_prefix) :]
    if name.startswith("vblora_"):
        return "lora_" + name[len("vblora_") :]
    return name


def convert_vblora_to_lora(
    state_dict: Dict[str, torch.Tensor],
    topk: int,
    from_prefix: str,
    to_prefix: str,
) -> Tuple[Dict[str, torch.Tensor], int]:
    if "vblora_vector_bank" not in state_dict:
        raise KeyError("Input checkpoint missing required key 'vblora_vector_bank'.")

    vector_bank = state_dict["vblora_vector_bank"]
    if vector_bank.ndim != 2:
        raise ValueError(
            "Expected 'vblora_vector_bank' to be rank-2 tensor, "
            f"got shape {tuple(vector_bank.shape)}."
        )

    logits_a: Dict[str, torch.Tensor] = {}
    logits_b: Dict[str, torch.Tensor] = {}
    alphas: Dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if key.endswith(".vblora_logits_A"):
            logits_a[key[: -len(".vblora_logits_A")]] = value
        elif key.endswith(".vblora_logits_B"):
            logits_b[key[: -len(".vblora_logits_B")]] = value
        elif key.endswith(".alpha"):
            alphas[key[: -len(".alpha")]] = value

    module_names = sorted(set(logits_a.keys()) & set(logits_b.keys()))
    if not module_names:
        raise ValueError("No VB-LoRA module logits found in checkpoint.")

    converted: Dict[str, torch.Tensor] = {}
    converted_count = 0
    for module_name in module_names:
        logit_a = logits_a[module_name]
        logit_b = logits_b[module_name]

        if logit_a.ndim != 3 or logit_b.ndim != 3:
            logger.warning("Skipping %s due to unexpected logits rank.", module_name)
            continue

        rank = int(logit_a.shape[0])
        if int(logit_b.shape[1]) != rank:
            logger.warning(
                "Skipping %s due to rank mismatch (A rank=%s, B rank=%s).",
                module_name,
                rank,
                int(logit_b.shape[1]),
            )
            continue

        # A: (rank, in_tile, vec_len) -> (rank, in_features)
        lora_down = _materialize_matrix(logit_a, vector_bank, topk).reshape(rank, -1)
        # B: (out_tile, rank, vec_len) -> (out_features, rank)
        lora_up = (
            _materialize_matrix(logit_b, vector_bank, topk)
            .transpose(1, 2)
            .reshape(-1, rank)
        )

        out_name = _rename_module_name(module_name, from_prefix=from_prefix, to_prefix=to_prefix)
        converted[f"{out_name}.lora_down.weight"] = lora_down
        converted[f"{out_name}.lora_up.weight"] = lora_up

        alpha = alphas.get(module_name, torch.tensor(float(rank), dtype=torch.float32))
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(float(alpha), dtype=torch.float32)
        converted[f"{out_name}.alpha"] = alpha.to(dtype=torch.float32, device="cpu")
        converted_count += 1

    if converted_count == 0:
        raise ValueError("No modules were converted; check checkpoint compatibility.")
    return converted, converted_count


def _cast_floating_tensors(
    state_dict: Dict[str, torch.Tensor],
    dtype_name: str,
) -> Dict[str, torch.Tensor]:
    if dtype_name == "auto":
        return state_dict

    if dtype_name == "fp16":
        target_dtype = torch.float16
    elif dtype_name == "bf16":
        target_dtype = torch.bfloat16
    elif dtype_name == "fp32":
        target_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype_name}")

    casted: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        tensor = value.detach().clone().to("cpu")
        if torch.is_floating_point(tensor):
            tensor = tensor.to(target_dtype)
        casted[key] = tensor
    return casted


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Takenoko VB-LoRA checkpoint to regular LoRA checkpoint.",
    )
    parser.add_argument("input_path", help="Path to VB-LoRA checkpoint (.safetensors or .pt/.ckpt)")
    parser.add_argument("output_path", help="Path to output regular LoRA checkpoint")
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help=(
            "VB-LoRA top-k used to materialize weights. "
            "If omitted, inferred from compressed checkpoints, otherwise defaults to 2."
        ),
    )
    parser.add_argument(
        "--from-prefix",
        default="vblora_unet_",
        help="Module name prefix to replace (default: vblora_unet_).",
    )
    parser.add_argument(
        "--to-prefix",
        default="lora_unet_",
        help="Replacement prefix for regular LoRA module names (default: lora_unet_).",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "fp16", "bf16", "fp32"],
        default="auto",
        help="Output floating tensor dtype (default: auto, preserve source).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output path if it already exists.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not os.path.exists(args.input_path):
        logger.error("Input not found: %s", args.input_path)
        sys.exit(1)
    if os.path.exists(args.output_path) and not args.force:
        logger.error("Output exists: %s (use --force to overwrite)", args.output_path)
        sys.exit(1)

    try:
        state_dict, metadata = _load_state_dict(args.input_path)
        logger.info("Loaded %s tensors from %s", len(state_dict), args.input_path)

        decoded_state_dict, inferred_topk = _decode_topk_logits(state_dict)
        if inferred_topk is not None:
            logger.info("Detected compressed VB-LoRA logits (inferred top-k=%s).", inferred_topk)

        selected_topk = int(args.topk) if args.topk is not None else (inferred_topk or DEFAULT_TOPK)
        if selected_topk < 1:
            raise ValueError("--topk must be >= 1.")
        if inferred_topk is not None and args.topk is not None and int(args.topk) != int(inferred_topk):
            logger.warning(
                "Overriding inferred top-k (%s) with user value (%s).",
                inferred_topk,
                args.topk,
            )
        logger.info("Using top-k=%s for VB-LoRA materialization.", selected_topk)

        converted, converted_count = convert_vblora_to_lora(
            decoded_state_dict,
            topk=selected_topk,
            from_prefix=args.from_prefix,
            to_prefix=args.to_prefix,
        )
        converted = _cast_floating_tensors(converted, args.dtype)

        out_metadata = dict(metadata)
        out_metadata["takenoko_converted_from"] = "vblora"
        out_metadata["takenoko_conversion_tool"] = "convert_vblora_to_lora.py"
        out_metadata["takenoko_conversion_topk"] = str(selected_topk)
        out_metadata["takenoko_conversion_note"] = (
            "Inference compatibility export; not suitable for resuming VB-LoRA training."
        )

        _save_state_dict(args.output_path, converted, metadata=out_metadata)
        logger.info(
            "Converted %s VB-LoRA modules and wrote %s tensors to %s",
            converted_count,
            len(converted),
            args.output_path,
        )
    except Exception as exc:
        logger.error("Conversion failed: %s", exc)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
