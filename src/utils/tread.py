import torch
from dataclasses import dataclass
from typing import Optional, Any, cast


@dataclass
class MaskInfo:
    """Book-keeping for one TREAD routing window."""

    mask: torch.Tensor  # bool tensor, True where token was dropped
    ids_keep: torch.Tensor  # long tensor (B, num_keep)
    ids_mask: torch.Tensor  # long tensor (B, num_mask)
    ids_shuffle: torch.Tensor  # long tensor, permutation that packs kept tokens first
    ids_restore: torch.Tensor  # long tensor, inverse permutation


class TREADRouter:
    """
    Minimal implementation of the token router used in TREAD.

    Public API
    ----------
    get_mask(x, mask_ratio, ...) -> MaskInfo
        Sample a permutation & binary mask that decides which tokens are kept.
    start_route(x, mask_info) -> x_small
        Apply the permutation and truncate to kept tokens.
    end_route(x_small, mask_info, original_x=None) -> x_full
        Undo the permutation, re‑insert the skipped tokens (either the
        original representations or a mask token), and return a full‑length
        sequence so gradients can flow everywhere.
    """

    def __init__(self, seed: int = 42, device: Any = None):
        # Maintain one RNG generator per device type to avoid device mismatch
        self.seed = int(seed)
        self._generators: dict[str, torch.Generator] = {}
        if device is not None:
            dev_type = torch.device(device).type
            gen = torch.Generator(device=dev_type)
            gen.manual_seed(self.seed)
            self._generators[dev_type] = gen

    def _get_generator_for(self, device: Any) -> torch.Generator:
        dev_type = torch.device(device).type
        gen = self._generators.get(dev_type)
        if gen is None:
            gen = torch.Generator(device=dev_type)
            gen.manual_seed(self.seed)
            self._generators[dev_type] = gen
        return gen

    # --------------------------------------------------------------------- #
    # helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _importance(x: torch.Tensor) -> torch.Tensor:
        """
        Per‑token importance score  ∝  L1‑norm of the feature vector.

        Normalised to [0, 1] **per sample** to avoid scale drift.
        """
        # x: (B, S, D)
        mags = x.abs().sum(-1)  # (B,S)
        rng = mags.max(dim=1, keepdim=True)[0] - mags.min(dim=1, keepdim=True)[0]
        mags = (mags - mags.min(dim=1, keepdim=True)[0]) / (rng + 1e-8)
        return mags

    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def get_mask(
        self,
        x: torch.Tensor,  # (B, S, D)
        mask_ratio: float = 0.0,
        l1_reg: float = 0.0,
        inverse: bool = False,
        force_keep: Optional[torch.Tensor] = None,  # (B, S) bool
    ) -> MaskInfo:
        """
        Decide which tokens to keep.

        * `mask_ratio` = 0.75  → keep 25 % of the *image* tokens.
        * `force_keep` marks tokens that are **never** allowed to drop.
        """
        B, S, _ = x.shape

        # ---------------------------------------------------------------------
        # 1) book‑keeping for "must‑keep" tokens
        # ---------------------------------------------------------------------
        if force_keep is None:
            force_keep = torch.zeros(B, S, dtype=torch.bool, device=x.device)
        # narrow type for checkers
        assert force_keep is not None
        num_force = force_keep.sum(1)  # (B,)  how many per sample

        # overall keep budget (one scalar K for the whole batch)
        base_keep = S - int(round(S * float(mask_ratio)))
        keep_budget = max(base_keep, int(num_force.max()))
        K = keep_budget  # make name explicit

        # ---------------------------------------------------------------------
        # 2) importance + randomness mix
        # ---------------------------------------------------------------------
        score = self._importance(x)  # in [0,1]
        if inverse:
            score = 1.0 - score

        noise = torch.rand(
            score.shape,
            dtype=score.dtype,
            device=score.device,
            generator=self._get_generator_for(score.device),
        )

        mix = (1.0 - l1_reg) * noise + l1_reg * score  # convex combination
        mix = mix.masked_fill(force_keep, -1.0)  # force‑keep ⇒ lowest rank

        # ---------------------------------------------------------------------
        # 3) build permutations
        # ---------------------------------------------------------------------
        ids_shuffle = torch.argsort(mix, dim=1)  # (B, S) smallest first
        ids_keep = ids_shuffle[:, :K]  # (B, K)
        ids_mask = ids_shuffle[:, K:]  # (B, S-K)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # bool mask where True = *masked* (dropped) token
        mask = torch.ones(B, S, dtype=torch.bool, device=x.device)
        mask.scatter_(1, ids_keep, False)

        # runtime dtype assertions for clarity
        assert mask.dtype == torch.bool
        assert ids_keep.dtype == torch.long
        assert ids_mask.dtype == torch.long
        assert ids_shuffle.dtype == torch.long
        assert ids_restore.dtype == torch.long

        return MaskInfo(mask, ids_keep, ids_mask, ids_shuffle, ids_restore)

    def start_route(self, x: torch.Tensor, info: MaskInfo) -> torch.Tensor:
        """
        Permute tokens so the kept ones come first, then truncate.
        """
        # x: (B,S,D)
        x_shuf = torch.take_along_dim(
            x, info.ids_shuffle.unsqueeze(-1).expand_as(x), dim=1
        )
        return x_shuf[:, : info.ids_keep.size(1), :]

    def end_route(
        self,
        routed_x: torch.Tensor,  # (B, num_keep, D)
        info: MaskInfo,
        original_x: Optional[torch.Tensor] = None,
        mask_token: float | int = 0.0,
    ) -> torch.Tensor:
        """
        Rebuild a sequence of length S so gradients can flow everywhere.

        If `original_x` is provided we copy the *actual* skipped
        representations back in (recommended for training).  Otherwise we
        fill them with `mask_token`.
        """
        B, S = info.mask.shape
        D = routed_x.shape[2]

        # Create buffer for all tokens in shuffled order
        x_shuf = torch.empty(B, S, D, device=routed_x.device, dtype=routed_x.dtype)
        x_shuf[:, : routed_x.size(1), :] = routed_x

        if original_x is not None:
            # put original skipped tokens back behind the kept ones
            orig_shuf = torch.take_along_dim(
                original_x,
                info.ids_shuffle.unsqueeze(-1).expand_as(original_x),
                dim=1,
            )
            x_shuf[:, routed_x.size(1) :, :] = orig_shuf[:, routed_x.size(1) :, :]
        else:
            x_shuf[:, routed_x.size(1) :, :].fill_(mask_token)

        # Undo the permutation
        x_full = torch.take_along_dim(
            x_shuf, info.ids_restore.unsqueeze(-1).expand_as(x_shuf), dim=1
        )
        return x_full
