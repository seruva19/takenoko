"""Model management for WAN network trainer.

This module handles model loading, configuration, and modification for the training system.
Extracted from wan_network_trainer.py to improve code organization and maintainability.
"""

import argparse
import importlib
import sys
import os
from typing import Any, Dict, Optional, Tuple
import torch
from accelerate import Accelerator
from memory.safetensors_loader import load_file

import utils.fluxflow_augmentation as fluxflow_augmentation
import logging
from common.logger import get_logger
from utils import model_utils
from common.model_downloader import download_model_if_needed
from wan.configs.config import WAN_CONFIGS
from wan.modules.model import WanModel, detect_wan_sd_dtype, load_wan_model
from wan.modules.vae import WanVAE
from core.dual_model_manager import DualModelManager

logger = get_logger(__name__, level=logging.INFO)


class ModelManager:
    """Handles model loading, configuration, and management."""

    def __init__(self):
        self.pos_embed_cache = {}
        self.config = None
        self.dit_dtype = None
        self.default_guidance_scale = 1.0
        self.fluxflow_config = {}
        self._downloaded_dit_path = (
            None  # Store downloaded path to avoid double downloading
        )

    def detect_wan_sd_dtype(self, dit_path: str) -> torch.dtype:
        """Detect the dtype of the WAN model from the checkpoint."""
        return detect_wan_sd_dtype(dit_path)

    def get_attention_mode(self, args: argparse.Namespace) -> str:
        """Get the attention mode based on arguments."""
        if args.sdpa:
            return "torch"
        elif args.flash_attn:
            return "flash"
        elif args.sage_attn:
            return "sageattn"
        elif args.xformers:
            return "xformers"
        elif args.flash3:
            return "flash3"
        else:
            raise ValueError(
                "Either --sdpa, --flash-attn, --flash3, --sage-attn or --xformers must be specified"
            )

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: Optional[torch.dtype],
        config: Any,
    ) -> tuple[WanModel, Optional[DualModelManager]]:
        """Load the transformer model and optionally configure dual-model training."""
        # Use already downloaded path if available, otherwise download
        if self._downloaded_dit_path is not None:
            dit_path = self._downloaded_dit_path
            logger.info(f"Using already downloaded DiT model: {dit_path}")
        elif dit_path.startswith(("http://", "https://")):
            logger.info(f"Detected URL for DiT model, downloading: {dit_path}")
            cache_dir = getattr(args, "model_cache_dir", None)
            dit_path = download_model_if_needed(dit_path, cache_dir=cache_dir)
            logger.info(f"Downloaded DiT model to: {dit_path}")

        # Get sparse_algo parameter if nabla sparse attention is enabled
        sparse_algo = None
        if hasattr(args, "nabla_sparse_attention") and args.nabla_sparse_attention:
            sparse_algo = args.nabla_sparse_algo
            if sparse_algo is None:
                default_algo = "nabla-0.7_sta-11-24-24"
                logger.warning(
                    f"--nabla_sparse_attention is set, but --nabla_sparse_algo is not. Using a default: '{default_algo}'"
                )
                sparse_algo = default_algo
            logger.info(
                f"🔧 Loading WAN model with sparse attention algorithm: {sparse_algo}"
            )

        # Pass FVDM flag through to the model by inlining the constructor after load
        use_fvdm_flag = hasattr(args, "enable_fvdm") and args.enable_fvdm

        # Honor mixed_precision_transformer: keep per-tensor dtypes by not forcing a uniform cast
        mp_transformer = bool(getattr(args, "mixed_precision_transformer", False))
        dit_weight_dtype_to_use = None if mp_transformer else dit_weight_dtype

        # Optional override via config: dit_cast_dtype = "fp16" | "bf16" | None
        # Ignored when fp8_scaled is True (weights are converted to fp8 at load) or when mixed_precision_transformer is True
        cast_pref = getattr(args, "dit_cast_dtype", None)
        if not getattr(args, "fp8_scaled", False) and not mp_transformer and cast_pref:
            cast_pref_str = str(cast_pref).lower()
            if cast_pref_str in ("fp16", "float16"):  # explicit uniform cast
                dit_weight_dtype_to_use = torch.float16
                logger.info("Overriding DiT uniform cast to float16 via dit_cast_dtype")
            elif cast_pref_str in ("bf16", "bfloat16"):
                dit_weight_dtype_to_use = torch.bfloat16
                logger.info(
                    "Overriding DiT uniform cast to bfloat16 via dit_cast_dtype"
                )
            elif cast_pref_str in ("none", "null", "auto", ""):  # treat as None
                pass
            else:
                logger.warning(
                    f"Unknown dit_cast_dtype='{cast_pref}'. Expected 'fp16'|'bf16'|None. Using default behavior."
                )

        transformer = load_wan_model(
            config,
            accelerator.device,
            dit_path,
            attn_mode,
            split_attn,
            loading_device,
            dit_weight_dtype_to_use,
            getattr(args, "fp8_scaled", False),
            sparse_algo=sparse_algo,
            use_fvdm=use_fvdm_flag,
            quant_dtype=(
                torch.float32 if getattr(args, "upcast_quantization", False) else None
            ),
            upcast_linear=bool(getattr(args, "upcast_linear", False)),
            exclude_ffn_from_scaled_mm=bool(
                getattr(args, "exclude_ffn_from_scaled_mm", False)
            ),
            scale_input_tensor=getattr(args, "scale_input_tensor", None),
            rope_on_the_fly=bool(getattr(args, "rope_on_the_fly", False)),
            broadcast_time_embed=bool(getattr(args, "broadcast_time_embed", False)),
            strict_e_slicing_checks=bool(
                getattr(args, "strict_e_slicing_checks", False)
            ),
            lower_precision_attention=bool(
                getattr(args, "lower_precision_attention", False)
            ),
            simple_modulation=bool(getattr(args, "simple_modulation", False)),
            optimized_torch_compile=bool(
                getattr(args, "optimized_torch_compile", False)
            ),
            rope_func=str(getattr(args, "rope_func", "default")),
            rope_use_float32=bool(getattr(args, "rope_use_float32", False)),
            lean_attention_fp32_default=bool(
                getattr(args, "lean_attention_fp32_default", False)
            ),
            compile_args=(
                list(args.compile_args)
                if hasattr(args, "compile_args") and args.compile_args is not None
                else None
            ),
            enable_memory_mapping=bool(
                getattr(args, "enable_memory_mapping", False)
            ),
            enable_zero_copy_loading=bool(
                getattr(args, "enable_zero_copy_loading", False)
            ),
            enable_non_blocking_transfers=bool(
                getattr(args, "enable_non_blocking_transfers", False)
            ),
            memory_mapping_threshold=int(
                getattr(args, "memory_mapping_threshold", 10 * 1024 * 1024)
            ),
        )

        # Optional: enable lean attention math from config
        try:
            if bool(getattr(args, "lean_attn_math", False)):
                setattr(transformer, "_lean_attn_math", True)  # type: ignore
                logger.info("Lean attention math enabled for Wan blocks")
            if bool(getattr(args, "lower_precision_attention", False)):
                setattr(transformer, "_lower_precision_attention", True)  # type: ignore
                try:
                    for _blk in transformer.blocks:  # type: ignore[attr-defined]
                        setattr(_blk, "_lower_precision_attention", True)
                except Exception:
                    pass
                logger.info("Lower precision attention enabled (fp16 compute path)")
            if bool(getattr(args, "simple_modulation", False)):
                setattr(transformer, "_simple_modulation", True)  # type: ignore
                logger.info("Simple modulation enabled (Wan 2.1 style for 2.2)")
        except Exception:
            pass
        # WanModel was constructed with use_fvdm above; no runtime mutation needed

        # If dual-model training is disabled, return as-is
        enable_dual = bool(getattr(args, "enable_dual_model_training", False))
        if not enable_dual:
            return transformer, None

        # Validate required args
        high_noise_path = getattr(args, "dit_high_noise", None)
        if not high_noise_path or not str(high_noise_path).strip():
            raise ValueError(
                "enable_dual_model_training=True requires 'dit_high_noise' to be set"
            )

        # Download high-noise model if needed (without building a second module on GPU)
        if str(high_noise_path).startswith(("http://", "https://")):
            logger.info(
                f"Detected URL for high-noise DiT model, downloading: {high_noise_path}"
            )
            cache_dir = getattr(args, "model_cache_dir", None)
            high_noise_path = download_model_if_needed(
                high_noise_path, cache_dir=cache_dir
            )
            logger.info(f"Downloaded high-noise DiT model to: {high_noise_path}")

        timestep_boundary = float(getattr(args, "timestep_boundary", 900))
        blocks_to_swap = int(getattr(args, "blocks_to_swap", 0))

        # Allow mixed mode ONLY if explicitly enabled in config
        allow_mixed = bool(getattr(args, "allow_mixed_block_swap_offload", False))
        if blocks_to_swap > 0 and bool(getattr(args, "offload_inactive_dit", True)):
            if allow_mixed:
                logger.info(
                    "Mixed mode enabled: block swap on active DiT + offloaded inactive DiT on CPU"
                )
            else:
                logger.info(
                    "Block swap specified; offload_inactive_dit disabled (allow_mixed_block_swap_offload=false)"
                )

        # Respect explicit offload flag; if mixed not allowed and blocks_to_swap>0, disable offload
        if not allow_mixed and blocks_to_swap > 0:
            offload_inactive = False
        else:
            offload_inactive = bool(getattr(args, "offload_inactive_dit", True))

        # Load high-noise model to get its state_dict, identical to original implementation
        logger.info("Loading high-noise model to extract state_dict...")
        high_noise_model = load_wan_model(
            config,
            accelerator.device,
            high_noise_path,
            attn_mode,
            split_attn,
            "cpu" if offload_inactive else loading_device,
            dit_weight_dtype_to_use,
            getattr(args, "fp8_scaled", False),
            sparse_algo=sparse_algo,
            use_fvdm=use_fvdm_flag,
            quant_dtype=(
                torch.float32 if getattr(args, "upcast_quantization", False) else None
            ),
            upcast_linear=bool(getattr(args, "upcast_linear", False)),
            exclude_ffn_from_scaled_mm=bool(
                getattr(args, "exclude_ffn_from_scaled_mm", False)
            ),
            scale_input_tensor=getattr(args, "scale_input_tensor", None),
            rope_use_float32=bool(getattr(args, "rope_use_float32", False)),
            enable_memory_mapping=bool(
                getattr(args, "enable_memory_mapping", False)
            ),
            enable_zero_copy_loading=bool(
                getattr(args, "enable_zero_copy_loading", False)
            ),
            enable_non_blocking_transfers=bool(
                getattr(args, "enable_non_blocking_transfers", False)
            ),
            memory_mapping_threshold=int(
                getattr(args, "memory_mapping_threshold", 10 * 1024 * 1024)
            ),
        )

        # Enable block swap for the temporary high-noise model only when not offloading it
        # When offloading inactive (mixed mode), keep the temp model strictly on CPU to extract state_dict
        if blocks_to_swap > 0 and not offload_inactive:
            logger.info(
                f"Prepare block swap for high noise model, blocks_to_swap={blocks_to_swap}"
            )
            high_noise_model.enable_block_swap(
                blocks_to_swap, accelerator.device, supports_backward=True
            )
            high_noise_model.move_to_device_except_swap_blocks(accelerator.device)
            high_noise_model.prepare_block_swap_before_forward()

        # Extract state_dict from the loaded model (will be on CPU if offload_inactive)
        high_sd = high_noise_model.state_dict()

        # Clean up the temporary model to save memory
        del high_noise_model

        dual_manager = DualModelManager(
            active_transformer=transformer,
            high_noise_state_dict=high_sd,
            timestep_boundary=timestep_boundary,
            offload_inactive=offload_inactive,
            blocks_to_swap=blocks_to_swap,
        )

        logger.info(
            "✅ Dual model manager initialized (single LoRA, base swap strategy)"
        )
        return transformer, dual_manager

    def create_network(
        self,
        args: argparse.Namespace,
        transformer: WanModel,
        vae: Optional[WanVAE],
        control_signal_processor: Any,
    ) -> Any:
        """Create and configure the network for training."""
        # Load network module
        sys.path.append(os.path.dirname(__file__))
        logger.info(f"import network module: {args.network_module}")
        network_module = importlib.import_module(args.network_module)

        if args.base_weights is not None:
            # if base_weights is specified, merge the weights to DiT model
            for i, weight_path in enumerate(args.base_weights):
                if (
                    args.base_weights_multiplier is None
                    or len(args.base_weights_multiplier) <= i
                ):
                    multiplier = 1.0
                else:
                    multiplier = args.base_weights_multiplier[i]

                logger.info(
                    f"merging module: {weight_path} with multiplier {multiplier}"
                )

                weights_sd = load_file(weight_path)
                module = network_module.create_arch_network_from_weights(
                    multiplier, weights_sd, unet=transformer, for_inference=True
                )
                module.merge_to(None, transformer, weights_sd, torch.float32, "cpu")

            logger.info(f"all weights merged: {', '.join(args.base_weights)}")

        # prepare network
        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                key, value = net_arg.split("=")
                net_kwargs[key] = value

        sparse_algo_to_pass = None
        if hasattr(args, "nabla_sparse_attention") and args.nabla_sparse_attention:
            # The main switch is on. Now get the algorithm string.
            sparse_algo_to_pass = args.nabla_sparse_algo
            if sparse_algo_to_pass is None:
                # If the user enables sparse attention but forgets the algo string,
                # we can provide a sensible default or raise an error. A default is more user-friendly.
                default_algo = "nabla-0.7_sta-11-24-24"
                logger.warning(
                    f"--nabla_sparse_attention is set, but --nabla_sparse_algo is not. Using a default: '{default_algo}'"
                )
                sparse_algo_to_pass = default_algo

        # Add the result to net_kwargs. It will be None if the flag is off.
        net_kwargs["sparse_algo"] = sparse_algo_to_pass

        # Check if control LoRA is enabled
        control_config = None
        if hasattr(args, "enable_control_lora") and args.enable_control_lora:
            # Build control config using consistent key names expected by utils processor
            control_config = {
                "control_lora_type": getattr(args, "control_lora_type", "tile"),
                "control_preprocessing": getattr(args, "control_preprocessing", "blur"),
                "control_blur_kernel_size": getattr(
                    args, "control_blur_kernel_size", 15
                ),
                "control_blur_sigma": getattr(args, "control_blur_sigma", 4.0),
                "control_scale_factor": getattr(args, "control_scale_factor", 1.0),
                "control_concatenation_dim": getattr(
                    args, "control_concatenation_dim", -2
                ),
                "control_inject_noise": getattr(args, "control_inject_noise", 0.0),
            }
            logger.info(f"🎯 Control LoRA enabled with config: {control_config}")

        from networks.lora_wan import WAN_TARGET_REPLACE_MODULES

        # Handle VAE training separately
        if args.network_module == "networks.vae_wan":
            logger.info("🎨 Creating VAE network for VAE training")
            if vae is None:
                raise ValueError(
                    "VAE model is required for VAE training but was not loaded"
                )

            # Parse VAE-specific arguments
            vae_training_mode = getattr(args, "vae_training_mode", "full")

            # Import VAE network module
            import networks.vae_wan as vae_network_module

            if args.dim_from_weights:
                # Load VAE network from weights
                weights_sd = load_file(args.dim_from_weights)
                network = vae_network_module.create_network_from_weights(
                    1.0,  # multiplier
                    args.network_dim,
                    args.network_alpha,
                    vae,
                    weights_sd=weights_sd,
                    transformer=transformer,
                    training_mode=vae_training_mode,
                    **net_kwargs,
                )
            else:
                # Create new VAE network
                network = vae_network_module.create_network(
                    1.0,  # multiplier
                    args.network_dim,
                    args.network_alpha,
                    vae,
                    transformer=transformer,
                    training_mode=vae_training_mode,
                    **net_kwargs,
                )

            logger.info(
                f"✅ VAE network created with training mode: {vae_training_mode}"
            )
            return network

        if args.dim_from_weights:
            logger.info(f"Loading network from weights: {args.dim_from_weights}")
            weights_sd = load_file(args.dim_from_weights)
            if control_config is not None:
                # Use control LoRA network from weights
                from networks.control_lora_wan import (
                    create_control_network_from_weights,
                )

                network = create_control_network_from_weights(
                    WAN_TARGET_REPLACE_MODULES,
                    1,
                    weights_sd,
                    unet=transformer,
                    control_config=control_config,
                )
            else:
                network, _ = network_module.create_arch_network_from_weights(
                    1, weights_sd, unet=transformer
                )
        else:
            # We use the name create_arch_network for compatibility with LyCORIS
            if hasattr(network_module, "create_arch_network"):
                if control_config is not None:
                    # Use control LoRA network
                    from networks.control_lora_wan import create_control_arch_network

                    network = create_control_arch_network(
                        1.0,
                        args.network_dim,
                        args.network_alpha,
                        vae,  # type: ignore
                        None,
                        transformer,
                        neuron_dropout=args.network_dropout,
                        control_config=control_config,
                        verbose=getattr(args, "verbose_network", False),
                        **net_kwargs,
                    )
                else:
                    network = network_module.create_arch_network(
                        1.0,
                        args.network_dim,
                        args.network_alpha,
                        vae,
                        None,
                        transformer,
                        neuron_dropout=args.network_dropout,
                        verbose=getattr(args, "verbose_network", False),
                        **net_kwargs,
                    )
            else:
                # LyCORIS compatibility
                network = network_module.create_network(
                    1.0,
                    args.network_dim,
                    args.network_alpha,
                    vae,
                    None,
                    transformer,
                    **net_kwargs,
                )

        if network is None:
            return None

        if hasattr(network_module, "prepare_network"):
            network.prepare_network(args)

        # apply network to DiT
        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)

        # Modify model for control LoRA if enabled
        if hasattr(args, "enable_control_lora") and args.enable_control_lora:
            self.modify_model_for_control_lora(transformer, args)

        # Optionally create ControlNet module (parallel network) if enabled
        self.controlnet = None
        if hasattr(args, "enable_controlnet") and args.enable_controlnet:
            try:
                import networks.controlnet_wan as controlnet_module

                cn_kwargs = {}
                if isinstance(getattr(args, "controlnet", None), dict):
                    cn_kwargs.update(args.controlnet)
                # mypy: pass torch.nn.Module | None for vae arg type
                self.controlnet = controlnet_module.create_network(
                    1.0,
                    args.network_dim,
                    args.network_alpha,
                    vae if isinstance(vae, torch.nn.Module) else None,
                    None,
                    transformer,
                    **cn_kwargs,
                )
                logger.info("✅ ControlNet module created and ready for training")
            except Exception as e:
                logger.warning(f"Failed to set up ControlNet module: {e}")

        if args.network_weights is not None:
            # FIXME consider alpha of weights: this assumes that the alpha is not changed
            info = network.load_weights(args.network_weights)
            logger.info(f"load network weights from {args.network_weights}: {info}")

        if args.gradient_checkpointing:
            transformer.enable_gradient_checkpointing(
                args.gradient_checkpointing_cpu_offload
            )
            network.enable_gradient_checkpointing()  # may have no effect

        return network

    def handle_model_specific_args(self, args: argparse.Namespace) -> None:
        """Handle model-specific argument processing and validation."""
        self.pos_embed_cache = {}
        self.config = WAN_CONFIGS[args.task]

        # Download model if it's a URL before detecting dtype
        dit_path = args.dit
        if dit_path.startswith(("http://", "https://")):
            logger.info(f"Detected URL for DiT model, downloading: {dit_path}")
            cache_dir = getattr(args, "model_cache_dir", None)
            dit_path = download_model_if_needed(dit_path, cache_dir=cache_dir)
            logger.info(f"Downloaded DiT model to: {dit_path}")
            # Update args.dit to point to the local downloaded path
            args.dit = dit_path
            self._downloaded_dit_path = dit_path  # Store for reuse

        self.dit_dtype = detect_wan_sd_dtype(dit_path)

        if self.dit_dtype == torch.float16:
            assert args.mixed_precision in [
                "fp16",
                "no",
            ], "DiT weights are in fp16, mixed precision must be fp16 or no"
        elif self.dit_dtype == torch.bfloat16:
            assert args.mixed_precision in [
                "bf16",
                "no",
            ], "DiT weights are in bf16, mixed precision must be bf16 or no"

        if args.fp8_scaled and self.dit_dtype.itemsize == 1:
            raise ValueError(
                "DiT weights is already in fp8 format, cannot scale to fp8. Please use fp16/bf16 weights / DiTの重みはすでにfp8形式です。fp8にスケーリングできません。fp16/bf16の重みを使用してください"
            )

        # dit_dtype cannot be fp8, so we select the appropriate dtype
        if self.dit_dtype.itemsize == 1:
            self.dit_dtype = (
                torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
            )

        args.dit_dtype = model_utils.dtype_to_str(self.dit_dtype)

        self.default_guidance_scale = 1.0  # not used
        self.fluxflow_config = fluxflow_augmentation.get_fluxflow_config_from_args(args)

    def load_vae(
        self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str
    ) -> WanVAE:
        """Load the VAE model."""
        # Download model if it's a URL
        if vae_path.startswith(("http://", "https://")):
            logger.info(f"Detected URL for VAE model, downloading: {vae_path}")
            cache_dir = getattr(args, "model_cache_dir", None)
            vae_path = download_model_if_needed(vae_path, cache_dir=cache_dir)
            logger.info(f"Downloaded VAE model to: {vae_path}")
            # Update args.vae to point to the local downloaded path
            args.vae = vae_path

        logger.info(f"Loading VAE model from {vae_path}")
        cache_device = torch.device("cpu") if args.vae_cache_cpu else None

        # Handle device parameter - ensure it's a valid device string
        device = "cuda"  # default
        if hasattr(args, "device") and args.device is not None:
            if isinstance(args.device, str):
                device = args.device
            elif isinstance(args.device, torch.device):
                device = str(args.device)
            else:
                logger.warning(f"Invalid device type {type(args.device)}, using 'cuda'")
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            logger.info("CUDA not available, using CPU")

        vae = WanVAE(
            vae_path=vae_path,
            device=device,
            dtype=vae_dtype,
            cache_device=cache_device,
        )
        return vae

    @staticmethod
    def scale_shift_latents(latents: torch.Tensor) -> torch.Tensor:
        """Apply scaling and shifting to latents (currently no-op for WAN)."""
        return latents

    def modify_model_for_control_lora(
        self, transformer: WanModel, args: argparse.Namespace
    ) -> None:
        """Modify the model's patch embedding layer to accept additional channels for control LoRA.

        This aligns with the reference implementation.
        """
        # Re-entrancy guard – return early if already patched
        if getattr(transformer, "_control_lora_patched", False):
            logger.debug("Control LoRA patch already applied – skipping.")
            return

        if hasattr(transformer, "patch_embedding"):
            with torch.no_grad():
                in_cls = transformer.patch_embedding.__class__  # nn.Conv3d
                old_in_dim = transformer.in_dim  # 16
                new_in_dim = old_in_dim * 2  # Double the input channels

                new_in = in_cls(
                    in_channels=new_in_dim,
                    out_channels=transformer.patch_embedding.out_channels,
                    kernel_size=transformer.patch_embedding.kernel_size,  # type: ignore
                    stride=transformer.patch_embedding.stride,  # type: ignore
                    padding=transformer.patch_embedding.padding,  # type: ignore
                ).to(
                    device=transformer.patch_embedding.weight.device,
                    dtype=transformer.patch_embedding.weight.dtype,
                )

                new_in.weight.zero_()
                # Copy original weights to first half of new weights
                new_in.weight[:, :old_in_dim, :, :, :] = (
                    transformer.patch_embedding.weight
                )
                # Copy original bias so the behaviour matches the reference implementation
                if transformer.patch_embedding.bias is not None:
                    new_in.bias.copy_(transformer.patch_embedding.bias)  # type: ignore

                # Replace the original patch embedding
                transformer.patch_embedding = new_in
                transformer.in_dim = new_in_dim

                # Update HuggingFace config so that any model save/load cycle retains the new input channel size
                if hasattr(transformer, "register_to_config"):
                    # WanModel may inherit from ConfigMixin in some versions
                    transformer.register_to_config(in_dim=new_in_dim)  # type: ignore

                logger.info(
                    f"Modified model for control LoRA: input channels {old_in_dim} -> {new_in_dim}"
                )

                # Ensure gradients are enabled for the new patch_embedding so it can learn
                transformer.patch_embedding.requires_grad_(True)

                # mark patched
                transformer._control_lora_patched = True  # type: ignore

    def get_model_config(self) -> Dict[str, Any]:
        """Get the current model configuration."""
        if self.config is None:
            raise ValueError(
                "Model config not initialized. Call handle_model_specific_args first."
            )
        return self.config

    def get_dit_dtype(self) -> torch.dtype:
        """Get the DiT data type."""
        if self.dit_dtype is None:
            raise ValueError(
                "DiT dtype not initialized. Call handle_model_specific_args first."
            )
        return self.dit_dtype

    def get_fluxflow_config(self) -> Dict[str, Any]:
        """Get the FluxFlow configuration."""
        return self.fluxflow_config

    def get_default_guidance_scale(self) -> float:
        """Get the default guidance scale."""
        return self.default_guidance_scale

    def validate_attention_mode(self, args: argparse.Namespace) -> str:
        """Validate and return the attention mode based on arguments."""
        if args.sdpa:
            return "torch"
        elif args.flash_attn:
            return "flash"
        elif args.sage_attn:
            return "sageattn"
        elif args.xformers:
            return "xformers"
        elif args.flash3:
            return "flash3"
        else:
            raise ValueError(
                "Either --sdpa, --flash-attn, --flash3, --sage-attn or --xformers must be specified"
            )

    def prepare_model_for_training(
        self,
        transformer: WanModel,
        args: argparse.Namespace,
        dit_weight_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Prepare the model for training (dtype casting, etc.)."""
        if dit_weight_dtype != self.dit_dtype and dit_weight_dtype is not None:
            logger.info(f"casting model to {dit_weight_dtype}")
            transformer.to(dit_weight_dtype)

        # Apply control LoRA modifications if needed (centralized entry point)
        if hasattr(args, "enable_control_lora") and args.enable_control_lora:
            logger.info("Applying control LoRA model modifications...")
            self.modify_model_for_control_lora(transformer, args)

        # Set appropriate training/eval mode
        if args.gradient_checkpointing:
            transformer.train()
        else:
            transformer.eval()

    def setup_block_swapping(
        self, transformer: WanModel, accelerator: Accelerator, blocks_to_swap: int
    ) -> WanModel:
        """Setup block swapping for memory optimization."""
        if blocks_to_swap > 0:
            logger.info(
                f"enable swap {blocks_to_swap} blocks to CPU from device: {accelerator.device}"
            )
            transformer.enable_block_swap(
                blocks_to_swap, accelerator.device, supports_backward=True
            )
            transformer.move_to_device_except_swap_blocks(accelerator.device)

            # Prepare with device placement
            transformer = accelerator.prepare(
                transformer, device_placement=[not blocks_to_swap > 0]
            )
            accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(
                accelerator.device
            )  # reduce peak memory usage
            accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()
        else:
            transformer = accelerator.prepare(transformer)

        return transformer
