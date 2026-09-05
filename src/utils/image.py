import os
import torch
from PIL import Image

# Compatibility with Pillow < 10 and Pillow >= 10
RESAMPLE_FILTER = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS', getattr(Image, 'ANTIALIAS', None))


def load_image(filename, size=None, scale=None):
    """
    Load an image from disk and optionally resize or scale it.

    Args:
        filename (str): Path to image file.
        size (int or tuple/list): Target (width, height) or single int for square resize.
        scale (float): Scale down factor (e.g., 2.0 halves the dimensions).

    Returns:
        PIL.Image.Image: Loaded and processed RGB image.
    """
    img = Image.open(filename).convert('RGB')

    if size is not None:
        if isinstance(size, int):
            target_size = (size, size)
        else:
            target_size = (size[0], size[1])
        img = img.resize(target_size, RESAMPLE_FILTER)

    if scale is not None and scale != 1.0:
        target_size = (int(img.size[0] / scale), int(img.size[1] / scale))
        img = img.resize(target_size, RESAMPLE_FILTER)

    return img


def save_image(filename, data):
    """
    Save a PyTorch Tensor or PIL Image to disk, creating parent directories if needed.

    Args:
        filename (str): Destination file path.
        data (torch.Tensor or PIL.Image.Image): Image data. If Tensor, expected values in [0, 255]
                                               or [0, 1]. Shape can be (C, H, W) or (1, C, H, W).
    """
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

    if isinstance(data, torch.Tensor):
        tensor = data.detach().cpu()
        if tensor.dim() == 4:
            tensor = tensor[0]
        # If float in range [0, 1], scale to [0, 255]
        if tensor.is_floating_point() and tensor.max() <= 1.0 and tensor.min() >= 0.0:
            tensor = tensor * 255.0
        np_img = tensor.clamp(0, 255).numpy().transpose(1, 2, 0).astype("uint8")
        img = Image.fromarray(np_img)
    elif isinstance(data, Image.Image):
        img = data
    else:
        raise TypeError(f"Unsupported data type for save_image: {type(data)}")

    img.save(filename)


def normalize_batch(batch):
    """
    Normalize image batch using ImageNet mean and std for VGG perceptual loss.

    Args:
        batch (torch.Tensor): Batch tensor with pixel values in range [0, 255], shape (B, 3, H, W).

    Returns:
        torch.Tensor: Normalized tensor with zero mean and unit variance.
    """
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch.div(255.0) - mean) / std
