import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from common.logger import get_logger

try:
    from vendor.ramtorch.ramtorch_linear import Linear as VendoredRamLinear  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    VendoredRamLinear = None

logger = get_logger(__name__, level=logging.INFO)


@dataclass
class RamTorchLinearConfig:
    enabled: bool = False
    device: Optional[str] = None  # e.g., "cuda" or "cuda:0"
    strength: float = 1.0  # fraction [0,1] of eligible linears to replace
    min_params: int = 0  # only replace if in_features*out_features >= min_params
    prefer_dtype: Optional[torch.dtype] = None  # dtype to initialize weights with
    fp32_io: bool = True  # upcast grads/inputs around RamTorch to fp32 for stability
    verbose: bool = False  # log every replacement


_cfg = RamTorchLinearConfig()


def configure_ramtorch_linear(
    *,
    enabled: bool,
    device: Optional[str] = None,
    strength: Optional[float] = None,
    min_params: Optional[int] = None,
    prefer_dtype: Optional[torch.dtype] = None,
    fp32_io: Optional[bool] = None,
    verbose: Optional[bool] = None,
) -> None:
    """
    Configure the RamTorch Linear replacement behavior.

    Parameters
    ----------
    enabled: bool
        Whether RamTorch replacement is enabled.
    device: Optional[str]
        Device string to pass to RamTorch Linear (e.g., "cuda").
    strength: Optional[float]
        Fraction in [0,1] controlling how many eligible layers are replaced.
    min_params: Optional[int]
        Minimum parameter count (in_features*out_features) to qualify for replacement.
    prefer_dtype: Optional[torch.dtype]
        Preferred dtype for RamTorch parameters (respected even if constructor lacks dtype).
    """
    global _cfg
    _cfg.enabled = bool(enabled)
    if device is not None:
        _cfg.device = device
    if strength is not None:
        # clamp
        _cfg.strength = max(0.0, min(1.0, float(strength)))
    if min_params is not None:
        _cfg.min_params = int(min_params)
    if prefer_dtype is not None:
        _cfg.prefer_dtype = prefer_dtype
    if fp32_io is not None:
        _cfg.fp32_io = bool(fp32_io)
    if verbose is not None:
        _cfg.verbose = bool(verbose)
    if _cfg.enabled:
        logger.info(
            "🌲 RamTorch Linear ENABLED: device=%s, strength=%.3f, min_params=%d, dtype=%s, fp32_io=%s, verbose=%s",
            _cfg.device,
            _cfg.strength,
            _cfg.min_params,
            str(_cfg.prefer_dtype),
            str(_cfg.fp32_io),
            str(_cfg.verbose),
        )
    else:
        logger.info("🌲 RamTorch Linear DISABLED")


def _should_replace(in_features: int, out_features: int, tag: Optional[str]) -> bool:
    if not _cfg.enabled:
        return False
    num_params = int(in_features) * int(out_features)
    if num_params < _cfg.min_params:
        return False
    # Deterministic pseudo-random selection based on shape and tag
    basis = (in_features * 1315423911) ^ (out_features * 2654435761)
    if tag:
        # simple stable fold-in of tag hash (do not rely on Python's randomized hash)
        tag_val = 0
        for ch in tag:
            tag_val = (tag_val * 131 + ord(ch)) & 0xFFFFFFFF
        basis ^= tag_val
    # map to [0,1)
    frac = (basis & 0xFFFFFFFF) / 4294967296.0
    return frac < _cfg.strength


def _init_like_wan(m: nn.Module) -> None:
    """Initialize weights to match Takenoko's Wan init for Linear layers."""
    try:
        if hasattr(m, "weight") and isinstance(m.weight, torch.nn.Parameter):
            nn.init.xavier_uniform_(m.weight)
        if (
            hasattr(m, "bias")
            and isinstance(m.bias, torch.nn.Parameter)
            and m.bias is not None
        ):
            nn.init.zeros_(m.bias)
    except Exception as e:
        logger.debug("RamTorch init fallback skipped: %s", e)


_first_replacement_logged: bool = False


def make_linear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    *,
    tag: Optional[str] = None,
    prefer_dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """
    Factory returning either torch.nn.Linear or RamTorch Linear according to config.

    Parameters
    ----------
    in_features, out_features, bias
        Usual Linear arguments.
    tag: Optional[str]
        Semantic tag (e.g., "q", "k", "v", "o", "ffn_in", "ffn_out") for
        deterministic strength gating across consistent runs.
    prefer_dtype: Optional[torch.dtype]
        Preferred dtype for the module parameters; if RamTorch constructor does not
        accept dtype (unpatched), we will cast parameters post-construction.
    """
    if not _should_replace(in_features, out_features, tag):
        return nn.Linear(in_features, out_features, bias=bias)

    ram_linear_cls = VendoredRamLinear
    if ram_linear_cls is None:
        try:
            from ramtorch import Linear as RamLinear  # type: ignore
        except Exception as e:
            logger.warning("RamTorch unavailable (%s); falling back to nn.Linear", e)
            return nn.Linear(in_features, out_features, bias=bias)
        ram_linear_cls = RamLinear

    # Resolve dtype preference
    dtype = prefer_dtype or _cfg.prefer_dtype or torch.float32

    # Respect the unmerged patch adding dtype to constructor; fall back if missing
    linear: Optional[nn.Module] = None
    try:
        linear = ram_linear_cls(
            in_features,
            out_features,
            bias=bias,
            dtype=dtype,  # may not be supported in current release
            device=_cfg.device,
        )
    except TypeError:
        # Older RamTorch without dtype kwarg
        linear = ram_linear_cls(
            in_features,
            out_features,
            bias=bias,
            device=_cfg.device,
        )
        # Cast parameters to target dtype
        try:
            if hasattr(linear, "weight") and isinstance(
                linear.weight, torch.nn.Parameter
            ):
                linear.weight.data = linear.weight.data.to(dtype=dtype)
            if (
                hasattr(linear, "bias")
                and isinstance(linear.bias, torch.nn.Parameter)
                and linear.bias is not None
            ):
                linear.bias.data = linear.bias.data.to(dtype=dtype)
        except Exception as e:
            logger.debug("RamTorch cast-to-dtype skipped: %s", e)

    # Verbose or one-time usage log
    global _first_replacement_logged
    if _cfg.verbose:
        try:
            logger.info(
                "🌲 RamTorch Linear replace tag=%s, shape=(%d,%d), device=%s, dtype=%s",
                str(tag),
                int(out_features),
                int(in_features),
                str(_cfg.device),
                str(dtype),
            )
        except Exception:
            pass
    elif not _first_replacement_logged:
        try:
            logger.info(
                "🌲 Using RamTorch Linear (additional layers may be replaced; set ramtorch_verbose=true to log all) for tag=%s, shape=(%d,%d), device=%s, dtype=%s",
                str(tag),
                int(out_features),
                int(in_features),
                str(_cfg.device),
                str(dtype),
            )
        except Exception:
            pass
        _first_replacement_logged = True

    # Initialize to match Takenoko defaults
    _init_like_wan(linear)

    # Optional stability guard for mixed precision: run RamTorch Linear in float32
    if _cfg.fp32_io:
        orig_forward = linear.forward

        def wrapped_forward(x: torch.Tensor, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
            # Upcast inputs and keep outputs float32 so RamTorch backward receives grad_out=float32
            y = orig_forward(x.to(torch.float32), *args, **kwargs)
            return y

        linear.forward = wrapped_forward  # type: ignore[method-assign]
    return linear


def is_linear_like(m: nn.Module) -> bool:
    """Duck-typing check for Linear-like modules (torch.nn.Linear or RamTorch Linear)."""
    try:
        if isinstance(m, nn.Linear):
            return True
        has_if = hasattr(m, "in_features") and hasattr(m, "out_features")
        has_params = (
            hasattr(m, "weight")
            and (isinstance(getattr(m, "weight"), torch.nn.Parameter))
            and hasattr(m, "bias")
            and (
                getattr(m, "bias") is None
                or isinstance(getattr(m, "bias"), torch.nn.Parameter)
            )
        )
        has_forward = callable(getattr(m, "forward", None))
        return bool(has_if and has_params and has_forward)
    except Exception:
        return False


def ramtorch_enabled() -> bool:
    """Return whether RamTorch Linear replacement is currently enabled."""
    return _cfg.enabled


# -----------------------
# High-level config helpers
# -----------------------


def configure_ramtorch_from_args(args: Any) -> None:
    """
    Centralized configuration entrypoint. Extracts RamTorch-related options
    from a typical argparse.Namespace (or any object with attributes) and
    applies sane defaults and conversions.

    This keeps configuration logic out of config_parser and call sites.
    """
    try:
        # Derive preferred dtype from DiT dtype string when available
        prefer_dtype = None
        dit_dtype_str = getattr(args, "dit_dtype", None)
        if isinstance(dit_dtype_str, str):
            try:
                # Local import to avoid cycles on module import
                from utils.model_utils import str_to_dtype  # type: ignore

                prefer_dtype = str_to_dtype(dit_dtype_str)
            except Exception:
                prefer_dtype = None

        configure_ramtorch_linear(
            enabled=bool(getattr(args, "use_ramtorch_linear", False)),
            device=getattr(args, "ramtorch_device", None),
            strength=float(getattr(args, "ramtorch_strength", 1.0)),
            min_params=int(getattr(args, "ramtorch_min_features", 0)),
            prefer_dtype=prefer_dtype,
            fp32_io=bool(getattr(args, "ramtorch_fp32_io", True)),
            verbose=bool(getattr(args, "ramtorch_verbose", False)),
        )
    except Exception as e:
        logger.debug(f"RamTorch configuration skipped: {e}")


def configure_ramtorch_from_config(config: dict) -> None:
    """
    Alternative entrypoint accepting a raw dict (TOML-loaded). Useful if
    callers prefer to configure before creating args.
    """

    class _Obj:
        pass

    obj = _Obj()
    for key, val in config.items():
        setattr(obj, key, val)
    configure_ramtorch_from_args(obj)
