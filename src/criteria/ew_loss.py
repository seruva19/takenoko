import torch
import torch.nn.functional as F


def exponential_weighted_loss(
    noise_pred,
    target,
    alphas_cumprod,
    timesteps,
    loss_map,
    reduction="none",
    boundary_shift=0.0,
):
    mae_loss = F.l1_loss(noise_pred, target, reduction="none")
    mean_mae_loss = mae_loss.mean()
    mean_mae_loss += boundary_shift
    weight_map = torch.exp(mae_loss - mean_mae_loss)
    weighted_loss = mae_loss * weight_map
    mean_weighted_loss = weighted_loss.mean(dim=(2, 3), keepdim=True)
    std_weighted_loss = weighted_loss.std(dim=(2, 3), keepdim=True)
    ac = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    final_weighted_loss = mean_weighted_loss * ac + std_weighted_loss * (1 - ac)
    mse_loss = F.mse_loss(noise_pred, target, reduction="none")
    mean_final_weighted_loss = final_weighted_loss.mean()
    below_mean_mask = final_weighted_loss < mean_final_weighted_loss
    above_mean_mask = ~below_mean_mask
    loss = torch.zeros_like(final_weighted_loss)
    mse_loss_mean = mse_loss.mean(dim=(2, 3), keepdim=True)
    loss[below_mean_mask] = torch.min(
        final_weighted_loss[below_mean_mask],
        mse_loss_mean.expand_as(final_weighted_loss)[below_mean_mask],
    )
    loss[above_mean_mask] = (
        0.85 * final_weighted_loss[above_mean_mask]
        + 0.15 * mse_loss_mean.expand_as(final_weighted_loss)[above_mean_mask]
    )
    all_loss_values = torch.tensor(
        list(loss_map.values()), dtype=torch.float32, device=loss.device
    )
    min_loss_value = all_loss_values.min()
    max_loss_value = all_loss_values.max()
    normalized_factors = torch.tensor(
        [loss_map.get(t.item(), 1.0) for t in timesteps],
        dtype=torch.float32,
        device=loss.device,
    )
    interpolation_factors = (max_loss_value - normalized_factors) / (
        max_loss_value - min_loss_value
    )
    median_interpolation_factors = interpolation_factors.median()
    scaled_interpolation_factors = (
        0.5 * interpolation_factors + 0.5 * median_interpolation_factors
    )
    loss = (
        1 - scaled_interpolation_factors.view(-1, 1, 1, 1)
    ) * mse_loss + scaled_interpolation_factors.view(-1, 1, 1, 1) * loss

    # Apply the reduction
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
