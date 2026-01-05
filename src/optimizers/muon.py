import torch
import torch.distributed as dist
from optimizers.optimizer_utils import (
    adam_update,
    apply_weight_decay,
    muon_update,
    track_gradient_consistency,
    track_update_ratio,
)


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.

    Arguments:
        lr: The learning rate, in units of spectral norm per update.
        weight_decay: The AdamW-style weight decay.
        momentum: The momentum. A value of 0.95 here is usually fine.
    """

    def __init__(
        self, params, lr=0.02, weight_decay=0, momentum=0.95, log_muon_metrics=False
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            log_muon_metrics=log_muon_metrics,
        )
        assert (
            isinstance(params, list)
            and len(params) >= 1
            and isinstance(params[0], torch.nn.Parameter)
        )
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (
                dist.get_world_size() - len(params) % dist.get_world_size()
            )
            for base_i in range(len(params))[:: dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    if group.get("log_muon_metrics", False):
                        track_gradient_consistency(
                            state, p.grad, state["momentum_buffer"]
                        )

                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=group.get("ns_steps", 5),
                        nesterov=group.get("nesterov", True),
                    )

                    if group.get("log_muon_metrics", False):
                        track_update_ratio(state, p, update, group["lr"])

                    apply_weight_decay(
                        p,
                        update.reshape(p.shape),
                        group["lr"],
                        group["weight_decay"],
                        group.get("weight_decay_type", "default"),
                        group.get("initial_lr", group.get("lr")),
                    )
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(
                    params_pad[base_i : base_i + dist.get_world_size()],
                    params_pad[base_i + dist.get_rank()],
                )

        return loss


class SingleDeviceMuon(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """

    def __init__(
        self, params, lr=0.02, weight_decay=0, momentum=0.95, log_muon_metrics=False
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            log_muon_metrics=log_muon_metrics,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                if group.get("log_muon_metrics", False):
                    track_gradient_consistency(state, p.grad, state["momentum_buffer"])

                update = muon_update(
                    p.grad,
                    state["momentum_buffer"],
                    beta=group["momentum"],
                    ns_steps=group.get("ns_steps", 5),
                    nesterov=group.get("nesterov", True),
                )

                if group.get("log_muon_metrics", False):
                    track_update_ratio(state, p, update, group["lr"])

                apply_weight_decay(
                    p,
                    update.reshape(p.shape),
                    group["lr"],
                    group["weight_decay"],
                    group.get("weight_decay_type", "default"),
                    group.get("initial_lr", group.get("lr")),
                )
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed Muon variant that can be used for all parameters in the network, since it runs an
    internal AdamW for the parameters that are not compatible with Muon. The user must manually
    specify which parameters shall be optimized with Muon and which with Adam by passing in a
    list of param_groups with the `use_muon` flag set.

    The point of this class is to allow the user to have a single optimizer in their code, rather
    than having both a Muon and an Adam which each need to be stepped.

    You can see an example usage below:

    https://github.com/KellerJordan/modded-nanogpt/blob/master/records/052525_MuonWithAuxAdamExample/b01550f9-03d8-4a9c-86fe-4ab434f1c5e0.txt#L470
    ```
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    from muon import MuonWithAuxAdam
    adam_groups = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
    muon_group = dict(params=hidden_matrix_params, lr=0.05, momentum=0.95, use_muon=True)
    param_groups = [*adam_groups, muon_group]
    optimizer = MuonWithAuxAdam(param_groups)
    ```
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(
                    group["params"], key=lambda x: x.size(), reverse=True
                )
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["ns_steps"] = group.get("ns_steps", 5)
                group["nesterov"] = group.get("nesterov", True)
                group["log_muon_metrics"] = group.get("log_muon_metrics", False)
                assert set(group.keys()) == set(
                    [
                        "params",
                        "lr",
                        "momentum",
                        "weight_decay",
                        "use_muon",
                        "ns_steps",
                        "nesterov",
                        "initial_lr",
                        "weight_decay_type",
                        "log_muon_metrics",
                    ]
                )
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(
                    [
                        "params",
                        "lr",
                        "betas",
                        "eps",
                        "weight_decay",
                        "use_muon",
                        "initial_lr",
                        "weight_decay_type",
                    ]
                )
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (
                    dist.get_world_size() - len(params) % dist.get_world_size()
                )
                for base_i in range(len(params))[:: dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            # continue
                            p.grad = torch.zeros_like(p)  # Force synchronization
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)

                        if group.get("log_muon_metrics", False):
                            track_gradient_consistency(
                                state, p.grad, state["momentum_buffer"]
                            )

                        update = muon_update(
                            p.grad,
                            state["momentum_buffer"],
                            beta=group["momentum"],
                            ns_steps=group.get("ns_steps", 5),
                            nesterov=group.get("nesterov", True),
                        )

                        if group.get("log_muon_metrics", False):
                            track_update_ratio(state, p, update, group["lr"])

                        p.grad = update.reshape(p.shape).to(p.grad.dtype)
                        apply_weight_decay(
                            p,
                            p.grad,
                            group["lr"],
                            group["weight_decay"],
                            group.get("weight_decay_type", "default"),
                            group.get("initial_lr", group.get("lr")),
                        )
                        p.add_(p.grad, alpha=-group["lr"])
                    dist.all_gather(
                        params_pad[base_i : base_i + dist.get_world_size()],
                        params_pad[base_i + dist.get_rank()],
                    )
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
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
                        group.get("initial_lr", group.get("lr")),
                    )
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["ns_steps"] = group.get("ns_steps", 5)
                group["nesterov"] = group.get("nesterov", True)
                group["log_muon_metrics"] = group.get("log_muon_metrics", False)
                assert set(group.keys()) == set(
                    [
                        "params",
                        "lr",
                        "momentum",
                        "weight_decay",
                        "use_muon",
                        "ns_steps",
                        "nesterov",
                        "initial_lr",
                        "weight_decay_type",
                        "log_muon_metrics",
                    ]
                )
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(
                    [
                        "params",
                        "lr",
                        "betas",
                        "eps",
                        "weight_decay",
                        "use_muon",
                        "initial_lr",
                        "weight_decay_type",
                    ]
                )
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    if group.get("log_muon_metrics", False):
                        track_gradient_consistency(
                            state, p.grad, state["momentum_buffer"]
                        )

                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=group.get("ns_steps", 5),
                        nesterov=group.get("nesterov", True),
                    )

                    if group.get("log_muon_metrics", False):
                        track_update_ratio(state, p, update, group["lr"])

                    apply_weight_decay(
                        p,
                        update.reshape(p.shape),
                        group["lr"],
                        group["weight_decay"],
                        group.get("weight_decay_type", "default"),
                        group.get("initial_lr", group.get("lr")),
                    )
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
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
                        group.get("initial_lr", group.get("lr")),
                    )
                    p.add_(update, alpha=-group["lr"])

        return loss
