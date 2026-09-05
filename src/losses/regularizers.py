import torch
import torch.nn.functional as F


def total_variation_loss(img: torch.Tensor, squared: bool = True) -> torch.Tensor:
    """
    Computes Total Variation (TV) loss to encourage spatial smoothness and suppress high-frequency noise.

    Formula (squared=True, standard in style transfer):
        L_tv = (1/2) * (mean((I[:, :, 1:, :] - I[:, :, :-1, :])^2) + mean((I[:, :, :, 1:] - I[:, :, :, :-1])^2))

    Args:
        img (torch.Tensor): Image tensor of shape (B, C, H, W).
        squared (bool): If True, use squared differences (L2 norm). If False, use absolute differences (L1 norm).

    Returns:
        torch.Tensor: Scalar TV loss tensor.
    """
    diff_h = img[:, :, 1:, :] - img[:, :, :-1, :]
    diff_w = img[:, :, :, 1:] - img[:, :, :, :-1]

    if squared:
        tv_h = torch.mean(diff_h ** 2)
        tv_w = torch.mean(diff_w ** 2)
    else:
        tv_h = torch.mean(torch.abs(diff_h))
        tv_w = torch.mean(torch.abs(diff_w))

    return 0.5 * (tv_h + tv_w)


def compute_consistency_loss(generated_noise: torch.Tensor, generated_clean: torch.Tensor) -> torch.Tensor:
    """
    Computes consistency loss between stylized noisy image and stylized clean image.

    Args:
        generated_noise (torch.Tensor): Output from stylizing noisy input.
        generated_clean (torch.Tensor): Output from stylizing clean input.

    Returns:
        torch.Tensor: MSE loss between the two representations.
    """
    return F.mse_loss(generated_noise, generated_clean)
