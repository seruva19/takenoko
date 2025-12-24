"""CDC-FM preprocessing and geometry-aware noise utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import hashlib
import json
import os

import numpy as np
import torch
from safetensors import safe_open

from common.logger import get_logger

logger = get_logger(__name__)

try:
    import faiss  # type: ignore

    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False


@dataclass
class LatentSample:
    latent: np.ndarray
    cache_path: str
    shape: Tuple[int, ...]


def _hash_cdc_config(values: Dict[str, object]) -> str:
    payload = json.dumps(values, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:8]


def _get_latent_key(keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        if key.startswith("latents_"):
            return key
    return None


def _load_latent_tensor(cache_path: str) -> torch.Tensor:
    with safe_open(cache_path, framework="pt", device="cpu") as f:
        key = _get_latent_key(f.keys())
        if key is None:
            raise ValueError(f"No latent tensor found in cache: {cache_path}")
        return f.get_tensor(key).float()


def _shape_to_tag(shape: Tuple[int, ...]) -> str:
    if len(shape) == 4:
        _, frames, height, width = shape
        return f"{frames}x{height}x{width}"
    if len(shape) == 3:
        height, width = shape[-2], shape[-1]
        return f"{height}x{width}"
    return "x".join(str(s) for s in shape)


def get_cdc_npz_path(
    latent_cache_path: str,
    config_hash: str,
    latent_shape: Tuple[int, ...],
) -> str:
    cache_path = Path(latent_cache_path)
    shape_tag = _shape_to_tag(latent_shape)
    return str(cache_path.with_name(f"{cache_path.stem}_cdc_{shape_tag}_{config_hash}.npz"))


class CarreDuChampComputer:
    def __init__(
        self,
        k_neighbors: int = 256,
        k_bandwidth: int = 8,
        d_cdc: int = 8,
        gamma: float = 1.0,
        device: str = "cuda",
    ) -> None:
        self.k = k_neighbors
        self.k_bw = k_bandwidth
        self.d_cdc = d_cdc
        self.gamma = gamma
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def compute_knn_graph(self, latents_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not _FAISS_AVAILABLE:
            raise RuntimeError("faiss is not available for CDC-FM preprocessing")

        num_samples, dim = latents_np.shape
        k_actual = min(self.k, num_samples - 1)

        if latents_np.dtype != np.float32:
            latents_np = latents_np.astype(np.float32)

        index = faiss.IndexFlatL2(dim)
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        index.add(latents_np)  # type: ignore
        distances, indices = index.search(latents_np, k_actual + 1)  # type: ignore
        return distances, indices

    @torch.no_grad()
    def compute_gamma_b_single(
        self,
        point_idx: int,
        latents_np: np.ndarray,
        distances: np.ndarray,
        indices: np.ndarray,
        epsilon: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dim = latents_np.shape[1]
        neighbor_idx = indices[point_idx, 1:]
        neighbor_points = latents_np[neighbor_idx]

        max_distance = 1e10
        neighbor_dists = np.clip(distances[point_idx, 1:], 0, max_distance)
        neighbor_dists_sq = neighbor_dists ** 2

        eps_i = max(float(epsilon[point_idx]), 1e-10)
        eps_neighbors = np.maximum(epsilon[neighbor_idx], 1e-10)
        denom = np.maximum(eps_i * eps_neighbors, 1e-20)
        exp_arg = -neighbor_dists_sq / denom
        exp_arg = np.clip(exp_arg, -50, 0)
        weights = np.exp(exp_arg)

        weight_sum = weights.sum()
        if weight_sum < 1e-20 or not np.isfinite(weight_sum):
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / weight_sum

        centered = neighbor_points - latents_np[point_idx]
        weighted_centered = centered * weights[:, None]
        weighted_centered_torch = torch.from_numpy(weighted_centered).to(
            self.device, dtype=torch.float32
        )

        try:
            _, s_vals, v_h = torch.linalg.svd(weighted_centered_torch, full_matrices=False)
        except RuntimeError as exc:
            logger.debug("CDC SVD failed on GPU, retrying on CPU: %s", exc)
            u_np, s_np, v_np = np.linalg.svd(weighted_centered, full_matrices=False)
            s_vals = torch.from_numpy(s_np).to(self.device)
            v_h = torch.from_numpy(v_np).to(self.device)

        eigenvalues_full = s_vals ** 2
        if len(eigenvalues_full) >= self.d_cdc:
            top_eigenvalues, top_idx = torch.topk(eigenvalues_full, self.d_cdc)
            top_eigenvectors = v_h[top_idx, :]
        else:
            top_eigenvalues = eigenvalues_full
            top_eigenvectors = v_h
            pad = self.d_cdc - len(eigenvalues_full)
            if pad > 0:
                top_eigenvalues = torch.cat(
                    [top_eigenvalues, torch.zeros(pad, device=self.device)]
                )
                top_eigenvectors = torch.cat(
                    [top_eigenvectors, torch.zeros(pad, dim, device=self.device)]
                )

        max_eigenval = float(top_eigenvalues[0].item()) if len(top_eigenvalues) else 1.0
        if max_eigenval > 1e-10:
            top_eigenvalues = top_eigenvalues / max_eigenval
        top_eigenvalues = torch.clamp(
            top_eigenvalues * self.gamma, 1e-3, self.gamma * 1.0
        )

        eigvecs = top_eigenvectors.cpu().half()
        eigvals = top_eigenvalues.cpu().half()
        return eigvecs, eigvals

    def compute_for_batch(
        self,
        latents_np: np.ndarray,
        sample_ids: List[int],
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        num_samples, dim = latents_np.shape
        if len(sample_ids) != num_samples:
            raise ValueError(
                f"CDC sample id mismatch: {num_samples} latents vs {len(sample_ids)} ids"
            )

        if num_samples < 5:
            results: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
            for sample_id in sample_ids:
                eigvecs = torch.zeros(self.d_cdc, dim, dtype=torch.float16)
                eigvals = torch.zeros(self.d_cdc, dtype=torch.float16)
                results[sample_id] = (eigvecs, eigvals)
            return results

        distances, indices = self.compute_knn_graph(latents_np)
        k_bw_actual = min(self.k_bw, distances.shape[1] - 1)
        epsilon = distances[:, k_bw_actual]

        results = {}
        for local_idx, sample_id in enumerate(sample_ids):
            eigvecs, eigvals = self.compute_gamma_b_single(
                local_idx, latents_np, distances, indices, epsilon
            )
            results[sample_id] = (eigvecs, eigvals)
        return results


class CDCPreprocessor:
    def __init__(
        self,
        latent_cache_paths: List[str],
        config_hash: str,
        k_neighbors: int,
        k_bandwidth: int,
        d_cdc: int,
        gamma: float,
        min_bucket_size: int,
        device: str = "cuda",
    ) -> None:
        self.latent_cache_paths = latent_cache_paths
        self.config_hash = config_hash
        self.min_bucket_size = min_bucket_size
        self.computer = CarreDuChampComputer(
            k_neighbors=k_neighbors,
            k_bandwidth=k_bandwidth,
            d_cdc=d_cdc,
            gamma=gamma,
            device=device,
        )

    def _write_cdc_cache(
        self,
        cache_path: str,
        eigvecs: torch.Tensor,
        eigvals: torch.Tensor,
        latent_shape: Tuple[int, ...],
    ) -> None:
        np.savez_compressed(
            cache_path,
            eigvecs=eigvecs.cpu().numpy(),
            eigvals=eigvals.cpu().numpy(),
            shape=np.asarray(latent_shape, dtype=np.int32),
        )

    def compute_all(self, force_recache: bool = False) -> bool:
        if not _FAISS_AVAILABLE:
            logger.warning("CDC-FM preprocessing skipped: faiss is not available.")
            return False

        buckets: Dict[Tuple[int, ...], List[LatentSample]] = {}
        for cache_path in self.latent_cache_paths:
            latent_tensor = _load_latent_tensor(cache_path)
            latent_shape = tuple(int(x) for x in latent_tensor.shape)
            flat = latent_tensor.reshape(-1).cpu().numpy().astype(np.float32)
            buckets.setdefault(latent_shape, []).append(
                LatentSample(latent=flat, cache_path=cache_path, shape=latent_shape)
            )

        for shape, samples in buckets.items():
            if len(samples) < max(self.min_bucket_size, self.computer.k):
                logger.warning(
                    "CDC-FM skipped bucket %s (size=%d, k_neighbors=%d).",
                    shape,
                    len(samples),
                    self.computer.k,
                )
                for sample in samples:
                    cache_path = get_cdc_npz_path(
                        sample.cache_path, self.config_hash, sample.shape
                    )
                    if force_recache or not os.path.exists(cache_path):
                        eigvecs = torch.zeros(
                            self.computer.d_cdc, sample.latent.shape[0], dtype=torch.float16
                        )
                        eigvals = torch.zeros(self.computer.d_cdc, dtype=torch.float16)
                        self._write_cdc_cache(cache_path, eigvecs, eigvals, sample.shape)
                continue

            latents_np = np.stack([sample.latent for sample in samples], axis=0)
            sample_ids = list(range(len(samples)))
            results = self.computer.compute_for_batch(latents_np, sample_ids)

            for local_idx, sample in enumerate(samples):
                eigvecs, eigvals = results[local_idx]
                cache_path = get_cdc_npz_path(
                    sample.cache_path, self.config_hash, sample.shape
                )
                if force_recache or not os.path.exists(cache_path):
                    self._write_cdc_cache(cache_path, eigvecs, eigvals, sample.shape)

        return True


class GammaBDataset:
    def __init__(self, config_hash: str, device: Optional[str] = None) -> None:
        self.config_hash = config_hash
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._warned_missing: set[str] = set()
        self._warned_shape: set[str] = set()

    def _load_npz(self, cache_path: str) -> Optional[dict]:
        if not os.path.exists(cache_path):
            if cache_path not in self._warned_missing:
                logger.warning("CDC cache missing: %s", cache_path)
                self._warned_missing.add(cache_path)
            return None
        return dict(np.load(cache_path))

    def load_for_item(
        self,
        latent_cache_path: str,
        latent_shape: Tuple[int, ...],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        cache_path = get_cdc_npz_path(latent_cache_path, self.config_hash, latent_shape)
        data = self._load_npz(cache_path)
        if data is None:
            return None
        cached_shape = tuple(int(x) for x in data.get("shape", ()))
        if cached_shape and cached_shape != latent_shape:
            if cache_path not in self._warned_shape:
                logger.warning(
                    "CDC shape mismatch for %s: cached=%s current=%s",
                    latent_cache_path,
                    cached_shape,
                    latent_shape,
                )
                self._warned_shape.add(cache_path)
            return None
        eigvecs = torch.from_numpy(data["eigvecs"]).to(self.device).float()
        eigvals = torch.from_numpy(data["eigvals"]).to(self.device).float()
        return eigvecs, eigvals

    @torch.no_grad()
    def compute_sigma_t_x(
        self,
        eigenvectors: torch.Tensor,
        eigenvalues: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)

        t = t.view(-1, 1)
        if torch.allclose(t, torch.zeros_like(t), atol=1e-8):
            return x.reshape(orig_shape)

        if torch.allclose(eigenvalues, torch.zeros_like(eigenvalues), atol=1e-8):
            return x.reshape(orig_shape)

        vt_x = torch.einsum("bkd,bd->bk", eigenvectors, x)
        sqrt_eigenvalues = torch.sqrt(eigenvalues.clamp(min=1e-10))
        sqrt_lambda_vt_x = sqrt_eigenvalues * vt_x
        gamma_sqrt_x = torch.einsum("bkd,bk->bd", eigenvectors, sqrt_lambda_vt_x)
        result = (1 - t) * x + t * gamma_sqrt_x
        return result.reshape(orig_shape)


def apply_cdc_noise_transformation(
    noise: torch.Tensor,
    t_normalized: torch.Tensor,
    gamma_b_dataset: Optional[GammaBDataset],
    item_info: Optional[List[object]],
) -> torch.Tensor:
    if gamma_b_dataset is None or item_info is None:
        return noise
    if noise.shape[0] != len(item_info):
        logger.warning(
            "CDC-FM skipped: batch size %d does not match item_info %d",
            noise.shape[0],
            len(item_info),
        )
        return noise

    noise_device = noise.device
    t_normalized = t_normalized.to(device=noise_device, dtype=noise.dtype).clamp(0.0, 1.0)
    if t_normalized.dim() == 0:
        t_normalized = t_normalized.expand(noise.shape[0])

    latent_shape = tuple(int(x) for x in noise.shape[1:])
    noise_out = []
    for idx, item in enumerate(item_info):
        latent_cache_path = getattr(item, "latent_cache_path", None)
        if not latent_cache_path:
            noise_out.append(noise[idx])
            continue
        cached = gamma_b_dataset.load_for_item(latent_cache_path, latent_shape)
        if cached is None:
            noise_out.append(noise[idx])
            continue
        eigvecs, eigvals = cached
        sample_noise = noise[idx].reshape(1, -1)
        t_single = t_normalized[idx : idx + 1]
        transformed = gamma_b_dataset.compute_sigma_t_x(eigvecs, eigvals, sample_noise, t_single)
        noise_out.append(transformed.reshape(noise[idx].shape))

    return torch.stack(noise_out, dim=0)
