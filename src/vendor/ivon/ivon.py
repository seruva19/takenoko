from contextlib import contextmanager
from math import pow
from typing import Callable, Optional, Tuple

import torch
import torch.distributed as dist
import torch.optim
from torch import Tensor


ClosureType = Callable[[], Tensor]


def _welford_mean(avg: Optional[Tensor], newval: Tensor, count: int) -> Tensor:
    return newval if avg is None else avg + (newval - avg) / count


class IVON(torch.optim.Optimizer):
    hessian_approx_methods = (
        "price",
        "gradsq",
    )

    def __init__(
        self,
        params,
        lr: float,
        ess: float,
        hess_init: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.99999,
        weight_decay: float = 1e-4,
        mc_samples: int = 1,
        hess_approx: str = "price",
        clip_radius: float = float("inf"),
        sync: bool = False,
        debias: bool = True,
        rescale_lr: bool = True,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 1 <= mc_samples:
            raise ValueError("Invalid number of MC samples: {}".format(mc_samples))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        if not 0.0 < hess_init:
            raise ValueError("Invalid Hessian initialization: {}".format(hess_init))
        if not 0.0 < ess:
            raise ValueError("Invalid effective sample size: {}".format(ess))
        if not 0.0 < clip_radius:
            raise ValueError("Invalid clipping radius: {}".format(clip_radius))
        if not 0.0 <= beta1 <= 1.0:
            raise ValueError("Invalid beta1 parameter: {}".format(beta1))
        if not 0.0 <= beta2 <= 1.0:
            raise ValueError("Invalid beta2 parameter: {}".format(beta2))
        if hess_approx not in self.hessian_approx_methods:
            raise ValueError("Invalid hess_approx parameter: {}".format(beta2))

        defaults = dict(
            lr=lr,
            mc_samples=mc_samples,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            hess_init=hess_init,
            ess=ess,
            clip_radius=clip_radius,
        )
        super().__init__(params, defaults)

        self.mc_samples = mc_samples
        self.hess_approx = hess_approx
        self.sync = sync
        self._numel, self._device, self._dtype = self._get_param_configs()
        self.current_step = 0
        self.debias = debias
        self.rescale_lr = rescale_lr

        self._reset_samples()
        self._init_buffers()

    def _get_param_configs(self):
        all_params = []
        for pg in self.param_groups:
            pg["numel"] = sum(p.numel() for p in pg["params"] if p is not None)
            all_params += [p for p in pg["params"] if p is not None]
        if len(all_params) == 0:
            return 0, torch.device("cpu"), torch.get_default_dtype()

        devices = {p.device for p in all_params}
        if len(devices) > 1:
            raise ValueError(
                "Parameters are on different devices: " f"{[str(d) for d in devices]}"
            )
        dtypes = {p.dtype for p in all_params}
        if len(dtypes) > 1:
            raise ValueError(
                "Parameters are on different dtypes: " f"{[str(d) for d in dtypes]}"
            )

        device = next(iter(devices))
        dtype = next(iter(dtypes))
        total = sum(pg["numel"] for pg in self.param_groups)
        return total, device, dtype

    def _sync_buffers_to_params(self) -> None:
        """Ensure internal buffers/state live on the current parameter device.

        This makes IVON more robust if the model is moved between devices after
        optimizer initialization.
        """

        numel, device, dtype = self._get_param_configs()
        self._numel = numel
        self._device = device
        self._dtype = dtype

        for group in self.param_groups:
            if "momentum" in group and group["momentum"].device != device:
                group["momentum"] = group["momentum"].to(device=device)
            if "hess" in group and group["hess"].device != device:
                group["hess"] = group["hess"].to(device=device)

        for key in ("avg_grad", "avg_nxg", "avg_gsq"):
            value = self.state.get(key)
            if isinstance(value, torch.Tensor) and value.device != device:
                self.state[key] = value.to(device=device)

    def _reset_samples(self):
        self.state["count"] = 0
        self.state["avg_grad"] = None
        self.state["avg_nxg"] = None
        self.state["avg_gsq"] = None

    def _init_buffers(self):
        for group in self.param_groups:
            hess_init, numel = group["hess_init"], group["numel"]

            # Internal optimizer buffers are kept in fp32 for numerical stability.
            group["momentum"] = torch.zeros(
                numel, device=self._device, dtype=torch.float32
            )
            group["hess"] = torch.zeros(
                numel, device=self._device, dtype=torch.float32
            ).add(torch.as_tensor(hess_init, device=self._device, dtype=torch.float32))

    @contextmanager
    def sampled_params(self, train: bool = False):
        param_avg, noise = self._sample_params()
        yield
        self._restore_param_average(train, param_avg, noise)

    def _restore_param_average(self, train: bool, param_avg: Tensor, noise: Tensor):
        param_grads = []
        offset = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p is None:
                    continue

                p_slice = slice(offset, offset + p.numel())

                # Restore mean params and preserve parameter dtype
                p.data = param_avg[p_slice].view(p.shape).to(dtype=p.dtype)
                if train:
                    if p.grad is not None:
                        param_grads.append(p.grad.flatten())
                    else:
                        param_grads.append(
                            torch.zeros(p.numel(), device=p.device, dtype=p.dtype)
                        )
                offset += p.numel()
        assert offset == self._numel

        if train:
            self._sync_buffers_to_params()

            grad_sample = torch.cat(param_grads, 0)
            count = int(self.state.get("count", 0)) + 1
            self.state["count"] = count

            if self.state.get("avg_grad") is None:
                self.state["avg_grad"] = torch.zeros(
                    self._numel, device=self._device, dtype=torch.float32
                )
            if self.state.get("avg_nxg") is None:
                self.state["avg_nxg"] = torch.zeros_like(self.state["avg_grad"])
            if self.state.get("avg_gsq") is None:
                self.state["avg_gsq"] = torch.zeros_like(self.state["avg_grad"])

            grad_sample_fp32 = grad_sample.to(dtype=torch.float32)
            noise_fp32 = noise.to(dtype=torch.float32)

            self.state["avg_grad"] = _welford_mean(
                self.state["avg_grad"], grad_sample_fp32, count
            )
            if self.hess_approx == "price":
                self.state["avg_nxg"] = _welford_mean(
                    self.state["avg_nxg"], noise_fp32 * grad_sample_fp32, count
                )
            elif self.hess_approx == "gradsq":
                self.state["avg_gsq"] = _welford_mean(
                    self.state["avg_gsq"], grad_sample_fp32.square(), count
                )

    @torch.no_grad()
    def step(self, closure: ClosureType = None) -> Optional[Tensor]:
        self._sync_buffers_to_params()

        if closure is None:
            loss = None
        else:
            losses = []
            for _ in range(self.mc_samples):
                with torch.enable_grad():
                    loss = closure()
                losses.append(loss)
            loss = sum(losses) / self.mc_samples
        if self.sync and dist.is_initialized():
            self._sync_samples()

        # Ensure state exists even if no sample was collected (defensive).
        if self.state.get("avg_grad") is None:
            self.state["avg_grad"] = torch.zeros(
                self._numel, device=self._device, dtype=torch.float32
            )
        if self.state.get("avg_nxg") is None:
            self.state["avg_nxg"] = torch.zeros_like(self.state["avg_grad"])
        if self.state.get("avg_gsq") is None:
            self.state["avg_gsq"] = torch.zeros_like(self.state["avg_grad"])

        self._update()
        self._reset_samples()
        return loss

    def _sync_samples(self):
        world_size = dist.get_world_size()
        dist.all_reduce(self.state["avg_grad"])
        self.state["avg_grad"].div_(world_size)
        dist.all_reduce(self.state["avg_nxg"])
        self.state["avg_nxg"].div_(world_size)

    def _sample_params(self) -> Tuple[Tensor, Tensor]:
        self._sync_buffers_to_params()

        noise_samples = []
        param_avgs = []

        offset = 0
        for group in self.param_groups:
            gnumel = group["numel"]
            hess_buffer = group["hess"] + float(group["weight_decay"])
            noise_sample = (
                torch.randn(gnumel, device=hess_buffer.device, dtype=torch.float32)
                / (float(group["ess"]) * hess_buffer).sqrt()
            )
            noise_samples.append(noise_sample)

            goffset = 0
            for p in group["params"]:
                if p is None:
                    continue

                p_avg = p.data.flatten()
                numel = p.numel()
                p_noise = noise_sample[goffset : goffset + numel]

                param_avgs.append(p_avg)
                p.data = (
                    (p_avg.to(dtype=torch.float32) + p_noise)
                    .to(dtype=p.dtype)
                    .view(p.shape)
                )
                goffset += numel
                offset += numel
            assert goffset == group["numel"]
        assert offset == self._numel

        return (
            torch.cat([p.to(dtype=torch.float32) for p in param_avgs], 0),
            torch.cat(noise_samples, 0),
        )

    def _update(self):
        self.current_step += 1

        self._sync_buffers_to_params()

        offset = 0
        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["beta1"]
            b2 = group["beta2"]
            pg_slice = slice(offset, offset + group["numel"])

            param_avg = torch.cat(
                [
                    p.flatten().to(dtype=torch.float32)
                    for p in group["params"]
                    if p is not None
                ],
                0,
            )

            group["momentum"] = self._new_momentum(
                self.state["avg_grad"][pg_slice], group["momentum"], b1
            )

            group["hess"] = self._new_hess(
                self.hess_approx,
                group["hess"],
                self.state["avg_nxg"],
                self.state["avg_gsq"],
                pg_slice,
                group["ess"],
                b2,
                group["weight_decay"],
            )

            param_avg = self._new_param_averages(
                param_avg,
                group["hess"],
                group["momentum"],
                (
                    lr * (group["hess_init"] + group["weight_decay"])
                    if self.rescale_lr
                    else lr
                ),
                group["weight_decay"],
                group["clip_radius"],
                1.0 - pow(b1, float(self.current_step)) if self.debias else 1.0,
                group["hess_init"],
            )

            pg_offset = 0
            for p in group["params"]:
                if p is not None:
                    p.data = (
                        param_avg[pg_offset : pg_offset + p.numel()]
                        .view(p.shape)
                        .to(dtype=p.dtype)
                    )
                    pg_offset += p.numel()
            assert pg_offset == group["numel"]
            offset += group["numel"]
        assert offset == self._numel

    @staticmethod
    def _get_nll_hess(method: str, hess, avg_nxg, avg_gsq, pg_slice) -> Tensor:
        if method == "price":
            return avg_nxg[pg_slice] * hess
        elif method == "gradsq":
            return avg_gsq[pg_slice]
        else:
            raise NotImplementedError(f"unknown hessian approx.: {method}")

    @staticmethod
    def _new_momentum(avg_grad, m, b1) -> Tensor:
        return b1 * m + (1.0 - b1) * avg_grad

    @staticmethod
    def _new_hess(method, hess, avg_nxg, avg_gsq, pg_slice, ess, beta2, wd) -> Tensor:
        f = IVON._get_nll_hess(method, hess + wd, avg_nxg, avg_gsq, pg_slice) * ess
        return (
            beta2 * hess
            + (1.0 - beta2) * f
            + (0.5 * (1 - beta2) ** 2) * (hess - f).square() / (hess + wd)
        )

    @staticmethod
    def _new_param_averages(
        param_avg, hess, momentum, lr, wd, clip_radius, debias, hess_init
    ) -> Tensor:
        return param_avg - lr * torch.clip(
            (momentum / debias + wd * param_avg) / (hess + wd),
            min=-clip_radius,
            max=clip_radius,
        )
