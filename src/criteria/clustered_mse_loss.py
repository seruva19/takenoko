# Based on: https://github.com/Anzhc/Timestep-Attention-and-other-shenanigans/blob/main/Code/Clustered_MSE_loss.py (AGPL-3.0)

import torch
from sklearn.cluster import KMeans


def adaptive_clustered_mse_loss(
    input,
    target,
    timesteps,
    loss_map,
    reduction="none",
    min_clusters=4,
    max_clusters=100,
):

    device = input.device
    target = target.to(device)

    batch_size = input.size(0)
    adjusted_loss = torch.zeros_like(input, dtype=torch.float32)

    for i in range(batch_size):
        initial_loss = (input[i] - target[i]) ** 2

        timestep_loss = loss_map.get(timesteps[i].item(), 1.0)
        n_clusters = min_clusters + (timestep_loss - min(loss_map.values())) / (
            max(loss_map.values()) - min(loss_map.values())
        ) * (max_clusters - min_clusters)
        n_clusters = max(min(int(n_clusters), max_clusters), min_clusters)

        loss_flat = initial_loss.view(-1).detach().cpu().numpy().reshape(-1, 1)

        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(loss_flat)

        clusters = clusters.reshape(initial_loss.shape)

        clusters = torch.tensor(clusters, device=device, dtype=torch.long)

        unique_clusters = torch.unique(clusters)
        adjusted_loss_i = torch.zeros_like(initial_loss)
        for cluster in unique_clusters:
            cluster_mask = (clusters == cluster).float()
            cluster_loss = initial_loss * cluster_mask
            cluster_mean_loss = cluster_loss.sum() / cluster_mask.sum()
            adjusted_loss_i += cluster_mask * cluster_mean_loss

        adjusted_loss[i] = adjusted_loss_i

    # Apply the reduction
    if reduction == "mean":
        return adjusted_loss.mean()
    elif reduction == "sum":
        return adjusted_loss.sum()
    elif reduction == "none":
        return adjusted_loss
    else:
        raise ValueError(f"Invalid reduction type: {reduction}")
