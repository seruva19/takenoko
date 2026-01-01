"""MuonClip - Muon optimizer with QK-Clip attention stabilization.

MuonClip combines:
1. Muon (MomentUm Orthogonalized by Newton-schulz) for 2D weight matrices
2. QK-Clip for attention stability via per-head logit clipping

The QK-Clip mechanism constrains attention logits by applying per-head scaling
factors to Query and Key parameters, preventing attention collapse or explosion.

For LoRA training:
- Apply Muon to LoRA matrices
- Apply QK-Clip to attention LoRA modules if attention_params are registered
- Use AdamW for bias/gain parameters

Example Configuration (TOML):
--------------------------
optimizer_type = "MuonClip"
learning_rate = 5e-5

optimizer_args = [
    "muonclip_lr=0.001",              # LR for matrix params (Muon)
    "muonclip_adam_lr=5e-5",          # LR for bias/gain params (AdamW)
    "muonclip_tau=100.0",             # QK-Clip threshold
    "muonclip_momentum=0.95",         # Momentum for Muon
    "muonclip_ns_steps=5",            # Newton-Schulz iterations
    "weight_decay=0.01",              # Weight decay
    "muonclip_betas=[0.9,0.95]",      # AdamW betas
    "muonclip_nesterov=true",         # Use Nesterov momentum
    "muonclip_auto_detect_attention=true"  # Auto-detect attention params
]

When to Use:
-----------
- Training attention-heavy architectures (DiT, transformers)
- Experiencing attention instability (NaN losses, attention collapse)
- Full fine-tuning or training attention LoRA modules

QK-Clip Parameters:
------------------
- tau (default: 100.0): Attention logit clipping threshold
  - Lower tau (e.g., 50.0): More aggressive clipping, more stable
  - Higher tau (e.g., 200.0): Less clipping, more expressive
- Auto-detection: Automatically finds Q/K projection parameters by name patterns
  - Patterns: 'q_proj', 'k_proj', 'wq', 'wk', 'query', 'key', etc.

See configs/examples/wan22_lora_muonclip.toml for a complete example.
See src/optimizers/MUONCLIP_README.md for full documentation.
"""

import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Set, Union
from optimizers.optimizer_utils import apply_weight_decay

import logging
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


def zeropower_via_newtonschulz5(G, steps: int = 5):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    This is the same implementation as in muon.py for consistency.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    """Standard Muon update with Newton-Schulz orthogonalization."""
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    """Standard Adam update for non-Muon parameters."""
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class QKClipState:
    """Manages QK-Clip attention stabilization state.

    Attributes:
        tau: Threshold for attention logit clipping (default: 100.0)
        attention_params: Dict mapping parameter names to their role ('q', 'k', 'q_rope', 'k_rope')
        param_id_to_name: Reverse mapping from parameter id to name
        attention_logits: Dict storing max attention logits per parameter
    """

    def __init__(self, tau: float = 100.0):
        self.tau = tau
        self.attention_params: Dict[str, str] = {}  # param_name -> role
        self.param_id_to_name: Dict[int, str] = {}  # id(param) -> param_name
        self.attention_logits: Dict[str, float] = {}  # param_name -> max_logit
        self.enabled = False

    def register_attention_param(self, name: str, param: torch.nn.Parameter, role: str):
        """Register an attention parameter for QK-Clip.

        Args:
            name: Parameter name (e.g., "layer.0.attn.q_proj")
            param: The parameter tensor
            role: One of ['q', 'k', 'q_rope', 'k_rope']
        """
        valid_roles = {"q", "k", "q_rope", "k_rope"}
        if role not in valid_roles:
            raise ValueError(f"Role must be one of {valid_roles}, got {role}")

        self.attention_params[name] = role
        self.param_id_to_name[id(param)] = name
        self.enabled = True
        logger.debug(f"Registered attention param: {name} as {role}")

    def get_param_role(self, param: torch.nn.Parameter) -> Optional[str]:
        """Get the role of a parameter if it's registered for attention."""
        name = self.param_id_to_name.get(id(param))
        if name is None:
            return None
        return self.attention_params.get(name)

    def update_logits(self, attention_logits: Dict[str, float]):
        """Update the maximum attention logits for registered parameters."""
        self.attention_logits.update(attention_logits)

    def compute_scaling_factor(self, param: torch.nn.Parameter) -> float:
        """Compute QK-Clip scaling factor for a parameter.

        Returns:
            gamma: Scaling factor where gamma = min(1, tau / max_logit)
            Returns 1.0 if no logit data or not an attention param
        """
        name = self.param_id_to_name.get(id(param))
        if name is None or name not in self.attention_logits:
            return 1.0

        max_logit = self.attention_logits[name]
        if max_logit <= 0:
            return 1.0

        gamma = min(1.0, self.tau / max_logit)
        return gamma

    def apply_qk_clip_scaling(
        self, param: torch.nn.Parameter, update: torch.Tensor
    ) -> torch.Tensor:
        """Apply QK-Clip scaling to an update based on parameter role.

        Args:
            param: The parameter being updated
            update: The update tensor (gradient or Muon update)

        Returns:
            Scaled update tensor
        """
        if not self.enabled:
            return update

        role = self.get_param_role(param)
        if role is None:
            return update

        gamma = self.compute_scaling_factor(param)

        if role in {"q", "k"}:  # Head-specific components
            scale = gamma**0.5
        elif role == "q_rope":  # Query rotary
            scale = gamma
        elif role == "k_rope":  # Key rotary - no scaling
            scale = 1.0
        else:
            scale = 1.0

        if scale != 1.0:
            logger.debug(
                f"QK-Clip scaling {role} param by {scale:.4f} (gamma={gamma:.4f})"
            )

        return update * scale


class SingleDeviceMuonClip(torch.optim.Optimizer):
    """MuonClip optimizer for single-device training.

    This optimizer applies Muon updates to 2D parameters and optionally applies
    QK-Clip attention stabilization to registered attention parameters.

    Args:
        params: Parameters to optimize
        lr: Learning rate
        weight_decay: Weight decay coefficient
        momentum: Momentum coefficient for Muon (default: 0.95)
        tau: QK-Clip threshold for attention logit clipping (default: 100.0)
        ns_steps: Number of Newton-Schulz iterations (default: 5)
        nesterov: Whether to use Nesterov momentum (default: True)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0,
        momentum: float = 0.95,
        tau: float = 100.0,
        ns_steps: int = 5,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            tau=tau,
            ns_steps=ns_steps,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

        # Initialize QK-Clip state
        self.qk_clip = QKClipState(tau=tau)

    def register_attention_params(
        self,
        attention_params: Dict[str, tuple],
    ):
        """Register attention parameters for QK-Clip stabilization.

        Args:
            attention_params: Dict mapping param_name -> (param, role)
                where role is one of ['q', 'k', 'q_rope', 'k_rope']

        Example:
            >>> optimizer.register_attention_params({
            ...     "layer.0.attn.q_proj.lora_A": (model.layer[0].attn.q_proj.lora_A, 'q'),
            ...     "layer.0.attn.k_proj.lora_A": (model.layer[0].attn.k_proj.lora_A, 'k'),
            ... })
        """
        for name, (param, role) in attention_params.items():
            self.qk_clip.register_attention_param(name, param, role)

        logger.info(
            f"Registered {len(attention_params)} attention parameters for QK-Clip"
        )

    def update_attention_logits(self, attention_logits: Dict[str, float]):
        """Update maximum attention logits for QK-Clip scaling.

        Call this during training with the maximum attention logits observed
        in the forward pass.

        Args:
            attention_logits: Dict mapping param_name -> max_logit_value
        """
        self.qk_clip.update_logits(attention_logits)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                # Compute Muon update
                update = muon_update(
                    p.grad,
                    state["momentum_buffer"],
                    beta=group["momentum"],
                    ns_steps=group["ns_steps"],
                    nesterov=group["nesterov"],
                )

                # Apply QK-Clip scaling if this is an attention parameter
                update = self.qk_clip.apply_qk_clip_scaling(p, update)

                # Apply weight decay and learning rate
                apply_weight_decay(
                    p,
                    update.reshape(p.shape),
                    group["lr"],
                    group["weight_decay"],
                    group.get("weight_decay_type", "default"),
                    group.get("initial_lr", group["lr"]),
                )
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


class SingleDeviceMuonClipWithAuxAdam(torch.optim.Optimizer):
    """MuonClip with auxiliary AdamW for non-2D parameters.

    This optimizer applies:
    - MuonClip to 2D parameters (matrices)
    - AdamW to <2D parameters (biases, gains)

    Args:
        param_groups: List of parameter groups with 'use_muon' flag
    """

    def __init__(self, param_groups):
        # Validate and set defaults for each group
        for group in param_groups:
            assert "use_muon" in group, "Each param group must have 'use_muon' flag"

            if group["use_muon"]:
                # MuonClip defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["tau"] = group.get("tau", 100.0)
                group["ns_steps"] = group.get("ns_steps", 5)
                group["nesterov"] = group.get("nesterov", True)
            else:
                # AdamW defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)

        super().__init__(param_groups, dict())

        # Initialize QK-Clip state (shared across all Muon groups)
        tau = next((g["tau"] for g in param_groups if g.get("use_muon")), 100.0)
        self.qk_clip = QKClipState(tau=tau)

    def register_attention_params(self, attention_params: Dict[str, tuple]):
        """Register attention parameters for QK-Clip stabilization.

        Args:
            attention_params: Dict mapping param_name -> (param, role)
        """
        for name, (param, role) in attention_params.items():
            self.qk_clip.register_attention_param(name, param, role)

        logger.info(
            f"Registered {len(attention_params)} attention parameters for QK-Clip"
        )

    def update_attention_logits(self, attention_logits: Dict[str, float]):
        """Update maximum attention logits for QK-Clip scaling."""
        self.qk_clip.update_logits(attention_logits)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                # MuonClip update
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=group["ns_steps"],
                        nesterov=group["nesterov"],
                    )

                    # Apply QK-Clip scaling
                    update = self.qk_clip.apply_qk_clip_scaling(p, update)

                    apply_weight_decay(
                        p,
                        update.reshape(p.shape),
                        group["lr"],
                        group["weight_decay"],
                        group.get("weight_decay_type", "default"),
                        group.get("initial_lr", group["lr"]),
                    )
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                # AdamW update
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0

                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )

                    apply_weight_decay(
                        p,
                        update,
                        group["lr"],
                        group["weight_decay"],
                        group.get("weight_decay_type", "default"),
                        group.get("initial_lr", group["lr"]),
                    )
                    p.add_(update, alpha=-group["lr"])

        return loss


def auto_detect_attention_params(
    model: torch.nn.Module,
    trainable_params: List[torch.nn.Parameter],
    q_patterns: List[str] = None,
    k_patterns: List[str] = None,
) -> Dict[str, tuple]:
    """Automatically detect attention parameters by module name patterns.

    This is a helper function to identify Q and K projection parameters for QK-Clip
    based on common naming patterns in transformer models.

    Args:
        model: The model containing the parameters
        trainable_params: List of parameters that are being trained
        q_patterns: List of string patterns to match query projection names
            (default: ['q_proj', 'wq', 'query', 'to_q'])
        k_patterns: List of string patterns to match key projection names
            (default: ['k_proj', 'wk', 'key', 'to_k'])

    Returns:
        Dict mapping parameter names to (parameter, role) tuples

    Example:
        >>> attention_params = auto_detect_attention_params(model, trainable_params)
        >>> optimizer.register_attention_params(attention_params)
    """
    if q_patterns is None:
        q_patterns = ["q_proj", "wq", "query", "to_q", "attn.q"]
    if k_patterns is None:
        k_patterns = ["k_proj", "wk", "key", "to_k", "attn.k"]

    trainable_ids = {id(p) for p in trainable_params}
    attention_params = {}

    for name, param in model.named_parameters():
        if id(param) not in trainable_ids:
            continue

        name_lower = name.lower()

        # Check for query projection
        for pattern in q_patterns:
            if pattern in name_lower:
                # Check if it's a RoPE parameter
                if "rope" in name_lower or "rotary" in name_lower:
                    attention_params[name] = (param, "q_rope")
                else:
                    attention_params[name] = (param, "q")
                break

        # Check for key projection
        for pattern in k_patterns:
            if pattern in name_lower:
                # Check if it's a RoPE parameter
                if "rope" in name_lower or "rotary" in name_lower:
                    attention_params[name] = (param, "k_rope")
                else:
                    attention_params[name] = (param, "k")
                break

    if attention_params:
        logger.info(f"Auto-detected {len(attention_params)} attention parameters:")
        for name, (_, role) in attention_params.items():
            logger.info(f"  - {name}: {role}")
    else:
        logger.warning(
            "No attention parameters auto-detected. QK-Clip will not be applied."
        )

    return attention_params
