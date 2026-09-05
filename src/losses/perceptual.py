import torch
import torch.nn.functional as F


def gram_matrix(y: torch.Tensor) -> torch.Tensor:
    """
    Computes the normalized Gram matrix of a batch of feature maps.

    Args:
        y (torch.Tensor): Feature map tensor of shape (B, C, H, W).

    Returns:
        torch.Tensor: Gram matrix of shape (B, C, C).
    """
    b, ch, h, w = y.size()
    features = y.view(b, ch, h * w)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def compute_content_loss(features_generated, features_content) -> torch.Tensor:
    """
    Computes perceptual content loss between generated and content feature maps at relu2_2.

    Args:
        features_generated: VggOutputs namedtuple from stylize network output.
        features_content: VggOutputs namedtuple from original content image.

    Returns:
        torch.Tensor: MSE content loss.
    """
    return F.mse_loss(features_generated.relu2_2, features_content.relu2_2)


def compute_style_loss(features_generated, gram_style_targets) -> torch.Tensor:
    """
    Computes style loss across relu1_2, relu2_2, relu3_3, relu4_3 layers.
    Uses expand_as to match shapes without memory allocation, avoiding broadcasting warnings.

    Args:
        features_generated: Iterable of feature tensors (relu1_2, relu2_2, relu3_3, relu4_3).
        gram_style_targets: List of Gram matrices of the style image for each layer.

    Returns:
        torch.Tensor: Aggregated style MSE loss.
    """
    style_loss = torch.tensor(0.0, device=features_generated[0].device)
    for ft_gen, gm_target in zip(features_generated, gram_style_targets):
        gm_gen = gram_matrix(ft_gen)
        # expand_as creates a non-allocating view matching the batch size
        gm_target_exp = gm_target.expand_as(gm_gen)
        style_loss = style_loss + F.mse_loss(gm_gen, gm_target_exp)
    return style_loss
