"""
Stylization entry point for Fast Neural Style Transfer.
Supports both CLI usage and direct function import.
"""
import os
import re
import argparse
import torch
from torchvision import transforms
from PIL import Image

from src.models import get_model
from src.utils import get_device, load_image, save_image


def stylize_image(
    content_image_path: str,
    model_path: str,
    model_type: str = 'ae',
    scale: float | None = None,
    size: int | tuple | None = None,
    device: torch.device | str | None = None,
) -> Image.Image:
    """
    Stylizes a single image using a trained fast neural style transfer checkpoint.

    Args:
        content_image_path (str): Path to the content image.
        model_path (str): Path to the saved .pth model checkpoint.
        model_type (str): Model architecture name ('ae', 'bo', 'res', 'dense', 'ae_attn', 'ae_old').
        scale (float, optional): Downsampling scale factor.
        size (int or tuple, optional): Target size for the content image.
        device (torch.device or str, optional): Inference device.

    Returns:
        PIL.Image.Image: The stylized result.
    """
    if not os.path.exists(content_image_path):
        raise FileNotFoundError(f"Content image not found: '{content_image_path}'")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: '{model_path}'")

    if not isinstance(device, torch.device):
        device = get_device(device)

    # Load content image with optional scale or resize
    content_img = load_image(content_image_path, size=size, scale=scale)

    # Prepare tensor (scaled to [0, 255])
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255.0))
    ])
    content_tensor = content_transform(content_img).unsqueeze(0).to(device)

    # Initialize model
    style_model = get_model(model_type).to(device)

    # Load weights and strip legacy InstanceNorm running stats if present
    state_dict = torch.load(model_path, map_location=device)
    state_dict = {k: v for k, v in state_dict.items() if not re.search(r'in\d+\.running_(mean|var)$', k)}
    style_model.load_state_dict(state_dict)
    style_model.eval()

    with torch.no_grad():
        output_tensor = style_model(content_tensor)

    # Convert to PIL Image
    output_tensor = output_tensor[0].detach().cpu().clamp(0, 255)
    np_img = output_tensor.numpy().transpose(1, 2, 0).astype("uint8")
    return Image.fromarray(np_img)


def main():
    parser = argparse.ArgumentParser(
        description="Fast Neural Style Transfer - Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--content-image", "-c", type=str, required=True,
                        help="Path to content image you want to stylize.")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Path to saved .pth model checkpoint.")
    parser.add_argument("--output-path", type=str, default="./",
                        help="Directory for saving the output image.")
    parser.add_argument("--output-name", type=str, default="stylized.jpg",
                        help="Filename of the stylized output image.")
    parser.add_argument("--model-type", type=str, default="ae",
                        help="Model architecture ('ae', 'bo', 'res', 'dense', 'ae_attn', 'ae_old').")
    parser.add_argument("--content-scale", type=float, default=None,
                        help="Downscaling factor for content image (e.g. 2.0 halves width and height).")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to run inference on ('auto', 'cuda', 'mps', 'cpu', 'cuda:0').")
    parser.add_argument("--mps", action="store_true", default=False,
                        help="Convenience flag to enable macOS Apple Silicon GPU.")

    args = parser.parse_args()

    # Determine device
    device_arg = 'mps' if args.mps else args.device
    selected_device = get_device(device_arg)
    print(f"Running stylization on device: {selected_device}")

    # Stylize
    stylized_img = stylize_image(
        content_image_path=args.content_image,
        model_path=args.model,
        model_type=args.model_type,
        scale=args.content_scale,
        device=selected_device
    )

    # Ensure output directory exists and save
    out_file = os.path.join(args.output_path, args.output_name)
    save_image(out_file, stylized_img)
    print(f"Stylized image successfully saved to: {os.path.abspath(out_file)}")


if __name__ == "__main__":
    main()
