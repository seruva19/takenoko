import ast
import math
from typing import Dict, List, Optional, Tuple, Type, cast

import torch
import torch.nn as nn
from torch.nn.init import _calculate_correct_fan
from torch import Tensor

from common.logger import get_logger
from networks.lora_wan import LoRAModule, LoRANetwork, WAN_TARGET_REPLACE_MODULES


logger = get_logger(__name__)

_DEFAULT_VERA_SEED = 0
_DEFAULT_VERA_D_INITIAL = 1.0
_DEFAULT_VERA_MATRIX_INIT = "kaiming_uniform"
_VALID_VERA_INITS = {"kaiming_uniform", "kaiming_normal"}


class _VeRAProjectionBank:
    """Deterministic projection bank keyed by shape and seed."""

    _cache: Dict[Tuple[int, int, int, int, str], Tuple[Tensor, Tensor]] = {}

    @staticmethod
    def _kaiming_init(
        shape: Tuple[int, int],
        generator: torch.Generator,
        init_mode: str,
    ) -> Tensor:
        tensor = torch.empty(shape, device="cpu", dtype=torch.float32)
        fan = _calculate_correct_fan(tensor, "fan_in")
        gain = math.sqrt(2.0)
        std = gain / math.sqrt(float(max(1, fan)))
        if init_mode == "kaiming_normal":
            with torch.no_grad():
                return tensor.normal_(mean=0.0, std=std, generator=generator)

        bound = math.sqrt(3.0) * std
        with torch.no_grad():
            return tensor.uniform_(-bound, bound, generator=generator)

    @classmethod
    def get(
        cls,
        in_features: int,
        out_features: int,
        rank: int,
        seed: int,
        init_mode: str = _DEFAULT_VERA_MATRIX_INIT,
    ) -> Tuple[Tensor, Tensor]:
        normalized_init = str(init_mode).strip().lower() or _DEFAULT_VERA_MATRIX_INIT
        if normalized_init not in _VALID_VERA_INITS:
            normalized_init = _DEFAULT_VERA_MATRIX_INIT

        key = (
            int(in_features),
            int(out_features),
            int(rank),
            int(seed),
            normalized_init,
        )
        cached = cls._cache.get(key, None)
        if cached is not None:
            return cached

        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        vera_A = cls._kaiming_init(
            (int(out_features), int(rank)),
            generator,
            normalized_init,
        )
        vera_B = cls._kaiming_init(
            (int(rank), int(in_features)),
            generator,
            normalized_init,
        )

        cls._cache[key] = (vera_A, vera_B)
        return cls._cache[key]


def _parse_bool(raw: object) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(raw)


def _parse_seed_marker(initialize: Optional[str]) -> int:
    seed, _, _ = _parse_initialize_marker(initialize)
    return seed


def _parse_initialize_marker(initialize: Optional[str]) -> Tuple[int, float, str]:
    marker = str(initialize or "").strip().lower()
    if not marker:
        return (
            _DEFAULT_VERA_SEED,
            _DEFAULT_VERA_D_INITIAL,
            _DEFAULT_VERA_MATRIX_INIT,
        )

    seed = _DEFAULT_VERA_SEED
    d_initial = _DEFAULT_VERA_D_INITIAL
    matrix_init = _DEFAULT_VERA_MATRIX_INIT

    for token in marker.split("|"):
        token = token.strip()
        if not token:
            continue
        if token.startswith("vera_seed_"):
            try:
                seed = max(0, int(token.rsplit("_", 1)[-1]))
            except Exception:
                seed = _DEFAULT_VERA_SEED
            continue
        if token.startswith("dinit_"):
            try:
                value = float(token.split("_", 1)[1])
                if value > 0.0:
                    d_initial = value
            except Exception:
                d_initial = _DEFAULT_VERA_D_INITIAL
            continue
        if token.startswith("init_"):
            candidate = token.split("_", 1)[1].strip().lower()
            if candidate in _VALID_VERA_INITS:
                matrix_init = candidate

    return seed, d_initial, matrix_init


class VeRAModule(LoRAModule):
    """VeRA: frozen shared random projections + trainable scaling vectors."""

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        split_dims: Optional[List[int]] = None,
        initialize: Optional[str] = None,
        pissa_niter: Optional[int] = None,
        ggpo_sigma: Optional[float] = None,
        ggpo_beta: Optional[float] = None,
    ) -> None:
        del pissa_niter, ggpo_sigma, ggpo_beta  # VeRA does not use these.

        if split_dims is not None:
            raise ValueError(
                "VeRA does not currently support split_dims modules. "
                "Use non-split linear targets."
            )
        if org_module.__class__.__name__ == "Conv2d":
            raise ValueError("VeRA currently supports linear-like modules only.")

        in_features = getattr(org_module, "in_features", None)
        out_features = getattr(org_module, "out_features", None)
        if in_features is None or out_features is None:
            raise RuntimeError("VeRA requires linear-like module with in/out features.")

        rank = max(1, int(lora_dim))
        vera_seed, vera_d_initial, vera_matrix_init = _parse_initialize_marker(
            initialize
        )

        super().__init__(
            lora_name=lora_name,
            org_module=org_module,
            multiplier=multiplier,
            lora_dim=rank,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            split_dims=None,
            initialize="kaiming",
            pissa_niter=None,
            ggpo_sigma=None,
            ggpo_beta=None,
        )

        # VeRA does not optimize lora_down/lora_up weights.
        if hasattr(self, "lora_down") and hasattr(self.lora_down, "weight"):
            cast(nn.Module, self.lora_down).weight.requires_grad_(False)  # type: ignore[attr-defined]
        if hasattr(self, "lora_up") and hasattr(self.lora_up, "weight"):
            cast(nn.Module, self.lora_up).weight.requires_grad_(False)  # type: ignore[attr-defined]
        self._ggpo_enabled = False

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.vera_rank = int(rank)
        self.vera_projection_prng_key = int(vera_seed)
        self.vera_d_initial = float(vera_d_initial)
        self.vera_matrix_init = str(vera_matrix_init)
        self._shared_vera_A: Optional[Tensor] = None
        self._shared_vera_B: Optional[Tensor] = None

        # Start from no-op adaptation (b=0; d scale does not matter while b=0).
        self.vera_b = nn.Parameter(torch.zeros(self.out_features, dtype=torch.float32))
        self.vera_d = nn.Parameter(
            torch.full(
                (self.vera_rank,),
                fill_value=self.vera_d_initial,
                dtype=torch.float32,
            )
        )

        shared_A, shared_B = _VeRAProjectionBank.get(
            self.in_features,
            self.out_features,
            self.vera_rank,
            self.vera_projection_prng_key,
            init_mode=self.vera_matrix_init,
        )
        # Shared random projections are recreated from seed, so keep them out of checkpoints.
        self.register_buffer("vera_A", shared_A.clone(), persistent=False)
        self.register_buffer("vera_B", shared_B.clone(), persistent=False)
        self.register_buffer(
            "vera_seed",
            torch.tensor(self.vera_projection_prng_key, dtype=torch.int64),
            persistent=True,
        )

    def set_shared_projections(self, shared_A: Tensor, shared_B: Tensor) -> None:
        self._shared_vera_A = shared_A
        self._shared_vera_B = shared_B

    def _resolve_projection_pair(
        self,
        override_A: Optional[Tensor] = None,
        override_B: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        source_A = override_A
        source_B = override_B

        if source_A is None:
            source_A = (
                self._shared_vera_A if self._shared_vera_A is not None else self.vera_A
            )
        if source_B is None:
            source_B = (
                self._shared_vera_B if self._shared_vera_B is not None else self.vera_B
            )

        source_A = cast(Tensor, source_A)
        source_B = cast(Tensor, source_B)
        return (
            source_A[: self.out_features, : self.vera_rank],
            source_B[: self.vera_rank, : self.in_features],
        )

    def _apply_rank_vector(self, tensor: Tensor, vector: Tensor) -> Tensor:
        if tensor.ndim == 2:
            return tensor * vector.view(1, -1)
        if tensor.ndim == 3:
            return tensor * vector.view(1, 1, -1)
        if tensor.ndim == 4:
            return tensor * vector.view(1, -1, 1, 1)
        return tensor

    def _apply_output_vector(self, tensor: Tensor, vector: Tensor) -> Tensor:
        if tensor.ndim == 2:
            return tensor * vector.view(1, -1)
        if tensor.ndim == 3:
            return tensor * vector.view(1, 1, -1)
        if tensor.ndim == 4:
            return tensor * vector.view(1, -1, 1, 1)
        return tensor

    def _compute_delta(self, x: Tensor) -> Tuple[Tensor, float]:
        dtype = x.dtype
        device = x.device
        vera_A, vera_B = self._resolve_projection_pair()

        rank_proj = torch.nn.functional.linear(
            x,
            vera_B.to(device=device, dtype=dtype),
        )

        if self.rank_dropout is not None and self.training:
            mask = (
                torch.rand((rank_proj.size(0), self.vera_rank), device=device)
                > float(self.rank_dropout)
            )
            if rank_proj.ndim == 3:
                mask = mask.unsqueeze(1)
            elif rank_proj.ndim == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)
            rank_proj = rank_proj * mask.to(dtype=dtype)
            dropout_scale = 1.0 / (1.0 - float(self.rank_dropout))
        else:
            dropout_scale = 1.0

        rank_proj = self._apply_rank_vector(
            rank_proj,
            self.vera_d.to(device=device, dtype=dtype),
        )
        delta = torch.nn.functional.linear(
            rank_proj,
            vera_A.to(device=device, dtype=dtype),
        )
        delta = self._apply_output_vector(
            delta,
            self.vera_b.to(device=device, dtype=dtype),
        )
        return delta, dropout_scale

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        hidden = x
        if self.dropout is not None and self.training:
            hidden = torch.nn.functional.dropout(hidden, p=self.dropout)

        delta, dropout_scale = self._compute_delta(hidden)
        return org_forwarded + delta * self.multiplier * self.scale * dropout_scale

    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier

        vera_A, vera_B = self._resolve_projection_pair()
        scaled_A = vera_A.to(torch.float32) * self.vera_b.to(torch.float32).view(-1, 1)
        scaled_B = vera_B.to(torch.float32) * self.vera_d.to(torch.float32).view(-1, 1)
        delta = scaled_A @ scaled_B
        return delta * float(multiplier) * float(self.scale)

    def merge_to(self, sd, dtype, device, non_blocking=False):
        del non_blocking  # not used

        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"]
        org_dtype = weight.dtype
        org_device = weight.device
        work_device = org_device if device is None else device
        work_dtype = org_dtype if dtype is None else dtype

        if isinstance(sd, dict):
            vera_A = cast(
                Tensor,
                sd.get(
                    "vera_A",
                    sd.get("vera_B.default", self.vera_A),
                ),
            )
            vera_B = cast(
                Tensor,
                sd.get(
                    "vera_B",
                    sd.get("vera_A.default", self.vera_B),
                ),
            )
            vera_b = cast(
                Tensor,
                sd.get(
                    "vera_b",
                    sd.get(
                        "vera_lambda_b.default",
                        sd.get("vera_lambda_b", self.vera_b),
                    ),
                ),
            )
            vera_d = cast(
                Tensor,
                sd.get(
                    "vera_d",
                    sd.get(
                        "vera_lambda_d.default",
                        sd.get("vera_lambda_d", self.vera_d),
                    ),
                ),
            )
        else:
            vera_A = None
            vera_B = None
            vera_b = self.vera_b
            vera_d = self.vera_d

        active_vera_A, active_vera_B = self._resolve_projection_pair(vera_A, vera_B)

        scaled_A = active_vera_A.to(device=work_device, dtype=torch.float32) * vera_b.to(
            device=work_device, dtype=torch.float32
        ).view(-1, 1)
        scaled_B = active_vera_B.to(device=work_device, dtype=torch.float32) * vera_d.to(
            device=work_device, dtype=torch.float32
        ).view(-1, 1)
        delta = scaled_A @ scaled_B

        merged = weight.to(device=work_device, dtype=torch.float32) + (
            delta * float(self.multiplier) * float(self.scale)
        )
        org_sd["weight"] = merged.to(device=org_device, dtype=work_dtype)
        self.org_module.load_state_dict(org_sd)


class VeRAInfModule(VeRAModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.org_module_ref = [self.org_module]  # type: ignore[attr-defined]
        self.enabled = True
        self.network: Optional[VeRANetwork] = None

    def set_network(self, network) -> None:
        self.network = network

    def default_forward(self, x):
        delta, _ = self._compute_delta(x)
        return self.org_forward(x) + delta * self.multiplier * self.scale

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        return self.default_forward(x)


class VeRANetwork(LoRANetwork):
    def __init__(
        self,
        *args,
        module_class: Type[object] = VeRAModule,
        vera_projection_prng_key: int = _DEFAULT_VERA_SEED,
        **kwargs,
    ) -> None:
        super().__init__(*args, module_class=module_class, **kwargs)
        self.vera_projection_prng_key = int(max(0, vera_projection_prng_key))
        self._configure_shared_projections()

    def _configure_shared_projections(self) -> None:
        modules: List[VeRAModule] = [
            cast(VeRAModule, module)
            for module in (self.text_encoder_loras + self.unet_loras)
            if isinstance(module, VeRAModule)
        ]
        if not modules:
            return

        max_in_features = max(module.in_features for module in modules)
        max_out_features = max(module.out_features for module in modules)
        max_rank = max(module.vera_rank for module in modules)
        matrix_init = modules[0].vera_matrix_init
        seed = modules[0].vera_projection_prng_key

        shared_A, shared_B = _VeRAProjectionBank.get(
            in_features=max_in_features,
            out_features=max_out_features,
            rank=max_rank,
            seed=seed,
            init_mode=matrix_init,
        )
        for module in modules:
            module.set_shared_projections(shared_A, shared_B)

    def prepare_network(self, args) -> None:
        logger.info(
            "VeRA enabled (projection_prng_key=%s, modules=%s, shared_projections=True).",
            self.vera_projection_prng_key,
            len(self.text_encoder_loras) + len(self.unet_loras),
        )


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
):
    return create_network_from_weights(
        WAN_TARGET_REPLACE_MODULES,
        multiplier,
        weights_sd,
        text_encoders=text_encoders,
        unet=unet,
        for_inference=for_inference,
        **kwargs,
    )


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    include_time_modules = kwargs.get("include_time_modules", False)
    if isinstance(include_time_modules, str):
        include_time_modules = _parse_bool(include_time_modules)
    include_time_modules = bool(include_time_modules)

    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)

    excluded_parts = ["patch_embedding", "text_embedding", "norm", "head"]
    if not include_time_modules:
        excluded_parts.extend(["time_embedding", "time_projection"])
    exclude_patterns.append(r".*(" + "|".join(excluded_parts) + r").*")
    kwargs["exclude_patterns"] = exclude_patterns

    return create_network(
        WAN_TARGET_REPLACE_MODULES,
        "vera_unet",
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )


def create_network(
    target_replace_modules: List[str],
    prefix: str,
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    del vae  # API compatibility

    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = 1.0

    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    verbose = kwargs.get("verbose", False)
    if verbose is not None:
        verbose = True if verbose == "True" else False

    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is not None and isinstance(exclude_patterns, str):
        exclude_patterns = ast.literal_eval(exclude_patterns)
    include_patterns = kwargs.get("include_patterns", None)
    if include_patterns is not None and isinstance(include_patterns, str):
        include_patterns = ast.literal_eval(include_patterns)
    extra_exclude_patterns = kwargs.get("extra_exclude_patterns", None)
    if extra_exclude_patterns is not None and isinstance(extra_exclude_patterns, str):
        extra_exclude_patterns = ast.literal_eval(extra_exclude_patterns)
    extra_include_patterns = kwargs.get("extra_include_patterns", None)
    if extra_include_patterns is not None and isinstance(extra_include_patterns, str):
        extra_include_patterns = ast.literal_eval(extra_include_patterns)

    include_time_modules = kwargs.get("include_time_modules", False)
    if isinstance(include_time_modules, str):
        include_time_modules = _parse_bool(include_time_modules)
    if include_time_modules:
        if extra_include_patterns is None:
            extra_include_patterns = []
        for pattern in ("^time_embedding\\.", "^time_projection\\."):
            if pattern not in extra_include_patterns:
                extra_include_patterns.append(pattern)

    vera_projection_prng_key = kwargs.get(
        "vera_projection_prng_key", _DEFAULT_VERA_SEED
    )
    try:
        vera_projection_prng_key = max(0, int(vera_projection_prng_key))
    except Exception as exc:
        raise ValueError(
            f"vera_projection_prng_key must be an integer >= 0, got {vera_projection_prng_key!r}"
        ) from exc

    vera_d_initial = kwargs.get("vera_d_initial", _DEFAULT_VERA_D_INITIAL)
    try:
        vera_d_initial = float(vera_d_initial)
    except Exception as exc:
        raise ValueError(
            f"vera_d_initial must be a float > 0, got {vera_d_initial!r}"
        ) from exc
    if vera_d_initial <= 0.0:
        raise ValueError(f"vera_d_initial must be > 0, got {vera_d_initial}")

    vera_matrix_init = str(
        kwargs.get("vera_matrix_init", _DEFAULT_VERA_MATRIX_INIT)
    ).strip().lower()
    if vera_matrix_init not in _VALID_VERA_INITS:
        raise ValueError(
            "vera_matrix_init must be one of "
            f"{sorted(_VALID_VERA_INITS)}, got {vera_matrix_init!r}"
        )

    initialize = (
        f"vera_seed_{vera_projection_prng_key}"
        f"|dinit_{vera_d_initial}"
        f"|init_{vera_matrix_init}"
    )

    network = VeRANetwork(
        target_replace_modules,
        prefix,
        text_encoders,  # type: ignore[arg-type]
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        conv_lora_dim=None,  # VeRA linear-only for this integration.
        conv_alpha=None,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        extra_exclude_patterns=extra_exclude_patterns,
        extra_include_patterns=extra_include_patterns,
        verbose=verbose,
        ggpo_sigma=None,
        ggpo_beta=None,
        initialize=initialize,
        pissa_niter=None,
        module_class=VeRAModule,
        vera_projection_prng_key=vera_projection_prng_key,
    )

    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio", None)
    loraplus_lr_ratio = (
        float(loraplus_lr_ratio) if loraplus_lr_ratio is not None else None
    )
    if loraplus_lr_ratio is not None:
        network.set_loraplus_lr_ratio(loraplus_lr_ratio)

    return network


def create_network_from_weights(
    target_replace_modules: List[str],
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> LoRANetwork:
    modules_dim: Dict[str, int] = {}
    modules_alpha: Dict[str, Tensor] = {}
    try:
        inferred_seed = max(
            0,
            int(kwargs.get("vera_projection_prng_key", _DEFAULT_VERA_SEED)),
        )
    except Exception:
        inferred_seed = _DEFAULT_VERA_SEED
    try:
        inferred_d_initial = float(
            kwargs.get("vera_d_initial", _DEFAULT_VERA_D_INITIAL)
        )
    except Exception:
        inferred_d_initial = _DEFAULT_VERA_D_INITIAL
    if inferred_d_initial <= 0.0:
        inferred_d_initial = _DEFAULT_VERA_D_INITIAL
    inferred_matrix_init = str(
        kwargs.get("vera_matrix_init", _DEFAULT_VERA_MATRIX_INIT)
    ).strip().lower()
    if inferred_matrix_init not in _VALID_VERA_INITS:
        inferred_matrix_init = _DEFAULT_VERA_MATRIX_INIT

    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if key.endswith(".alpha"):
            modules_alpha[lora_name] = value
        elif key.endswith(".vera_d"):
            modules_dim[lora_name] = int(value.shape[0])
        elif key.endswith(".vera_lambda_d"):
            modules_dim[lora_name] = int(value.shape[0])
        elif key.endswith(".vera_lambda_d.default"):
            modules_dim[lora_name] = int(value.shape[0])
        elif key.endswith(".vera_seed"):
            try:
                inferred_seed = int(value.item())
            except Exception:
                inferred_seed = _DEFAULT_VERA_SEED

    module_class: Type[object] = VeRAInfModule if for_inference else VeRAModule

    extra_include_patterns = kwargs.get("extra_include_patterns", None)
    if extra_include_patterns is not None and isinstance(extra_include_patterns, str):
        extra_include_patterns = ast.literal_eval(extra_include_patterns)
    extra_exclude_patterns = kwargs.get("extra_exclude_patterns", None)
    if extra_exclude_patterns is not None and isinstance(extra_exclude_patterns, str):
        extra_exclude_patterns = ast.literal_eval(extra_exclude_patterns)

    include_time_modules = kwargs.get("include_time_modules", False)
    if isinstance(include_time_modules, str):
        include_time_modules = _parse_bool(include_time_modules)
    include_time_modules = bool(include_time_modules)

    if not extra_include_patterns:
        extra_include_patterns = []
    if include_time_modules:
        for pattern in ("^time_embedding\\.", "^time_projection\\."):
            if pattern not in extra_include_patterns:
                extra_include_patterns.append(pattern)
    else:
        for lora_name in modules_dim.keys():
            if "time_embedding" in lora_name or "time_projection" in lora_name:
                for pattern in ("^time_embedding\\.", "^time_projection\\."):
                    if pattern not in extra_include_patterns:
                        extra_include_patterns.append(pattern)
                break

    initialize = (
        f"vera_seed_{max(0, int(inferred_seed))}"
        f"|dinit_{inferred_d_initial}"
        f"|init_{inferred_matrix_init}"
    )

    network = VeRANetwork(
        target_replace_modules,
        "vera_unet",
        text_encoders,  # type: ignore[arg-type]
        unet,  # type: ignore[arg-type]
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        extra_exclude_patterns=extra_exclude_patterns,
        extra_include_patterns=extra_include_patterns,
        ggpo_sigma=None,
        ggpo_beta=None,
        initialize=initialize,
        pissa_niter=None,
        vera_projection_prng_key=max(0, int(inferred_seed)),
    )
    return network
