"""
Training entry point for Fast Neural Style Transfer.
Trains a feedforward transformation network to stylize images in real time using perceptual losses.
"""
import os
import time
import argparse
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models import get_model, Vgg16
from src.data import ImageFolderDataset
from src.losses import (
    gram_matrix,
    compute_content_loss,
    compute_style_loss,
    total_variation_loss,
    compute_consistency_loss,
)
from src.utils import get_device, load_image, normalize_batch


def train(args):
    # Set device
    device_arg = 'mps' if args.mps else args.device
    device = get_device(device_arg)
    print(f"Training on device: {device}")

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    # Content dataset transforms (scale pixels to [0, 255])
    content_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.mul(255.0))
    ])

    print(f"Loading dataset from '{args.dataset}'...")
    train_dataset = ImageFolderDataset(args.dataset, transform=content_transform)
    print(f"Found {len(train_dataset)} training images.")

    pin_memory = (device.type == 'cuda')
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    # Initialize stylization model
    stylize_network = get_model(args.model_type).to(device)
    optimizer = Adam(stylize_network.parameters(), lr=args.lr)

    # Initialize frozen VGG-16 perceptual feature extractor
    vgg = Vgg16(requires_grad=False).to(device)

    # Load and preprocess style image
    style_size = args.style_size if args.style_size is not None else args.image_size
    style_img = load_image(args.style_image, size=[style_size, style_size])
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.mul(255.0))
    ])
    style_tensor = style_transform(style_img).unsqueeze(0).to(device)

    # Precompute Gram matrices for the style target
    with torch.no_grad():
        features_style = vgg(normalize_batch(style_tensor))
        gram_style_targets = [gram_matrix(f) for f in features_style]

    print(f"Starting training for {args.epochs} epoch(s)...")
    start_time = time.time()

    for epoch in range(args.epochs):
        stylize_network.train()
        agg_content_loss = 0.0
        agg_style_loss = 0.0
        agg_tv_loss = 0.0
        agg_consistency_loss = 0.0
        agg_total_loss = 0.0

        step_start_time = time.time()
        num_batches = len(train_loader)

        for batch_id, content_batch in enumerate(train_loader):
            optimizer.zero_grad()
            content_batch = content_batch.to(device, non_blocking=True)

            # Generate stylized image
            generated_batch = stylize_network(content_batch)

            # Extract perceptual features from VGG
            features_content = vgg(normalize_batch(content_batch))
            features_generated = vgg(normalize_batch(generated_batch))

            # 1. Content loss (at relu2_2)
            content_loss = args.content_weight * compute_content_loss(features_generated, features_content)

            # 2. Style loss (across relu1_2, relu2_2, relu3_3, relu4_3)
            style_loss = args.style_weight * compute_style_loss(features_generated, gram_style_targets)

            # 3. Total variation loss (smoothing regularization)
            tv_loss = args.tv_weight * total_variation_loss(generated_batch)

            # 4. Consistency loss (optional robustness regularization)
            if args.consistency_weight > 0.0:
                noise = torch.randn_like(content_batch) * 255.0 * 0.05
                content_noisy = 0.95 * content_batch + noise
                generated_noisy = stylize_network(content_noisy)
                consistency_loss = args.consistency_weight * compute_consistency_loss(generated_noisy, generated_batch)
            else:
                consistency_loss = torch.tensor(0.0, device=device)

            # Total loss (includes tv_loss and consistency_loss in gradient backpropagation)
            total_loss = content_loss + style_loss + tv_loss + consistency_loss
            total_loss.backward()
            optimizer.step()

            # Accumulate loss metrics
            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()
            agg_tv_loss += tv_loss.item()
            agg_consistency_loss += consistency_loss.item()
            agg_total_loss += total_loss.item()

            if (batch_id + 1) % args.log_interval == 0 or (batch_id + 1) == num_batches:
                elapsed_sec = time.time() - step_start_time
                imgs_processed = (batch_id + 1) * args.batch_size
                fps = imgs_processed / max(elapsed_sec, 1e-5)
                n = batch_id + 1

                log_msg = (
                    f"Epoch [{epoch + 1}/{args.epochs}] "
                    f"Step [{n}/{num_batches}] "
                    f"Speed: {fps:.1f} img/s | "
                    f"Content: {agg_content_loss / n:.2f} | "
                    f"Style: {agg_style_loss / n:.2f} | "
                    f"TV: {agg_tv_loss / n:.2f} | "
                    f"Cons: {agg_consistency_loss / n:.2f} | "
                    f"Total: {agg_total_loss / n:.2f}"
                )
                print(log_msg)

        # Save checkpoint after each epoch
        stylize_network.eval()
        if args.model_name is None:
            model_base = os.path.splitext(os.path.basename(args.style_image))[0]
        else:
            model_base = args.model_name

        c_w_str = format(args.content_weight, '.1E').replace('+', '').replace('.', 'p')
        s_w_str = format(args.style_weight, '.1E').replace('+', '').replace('.', 'p')
        tv_w_str = format(args.tv_weight, '.0E').replace('+', '')
        cons_w_str = format(args.consistency_weight, '.0E').replace('+', '')

        checkpoint_name = f"{model_base}_{args.model_type}_cont{c_w_str}_sty{s_w_str}_tv{tv_w_str}_cons{cons_w_str}_epoch{epoch + 1}.pth"
        os.makedirs(args.save_model_dir, exist_ok=True)
        save_path = os.path.join(args.save_model_dir, checkpoint_name)
        torch.save(stylize_network.state_dict(), save_path)
        print(f"Checkpoint successfully saved to: {os.path.abspath(save_path)}")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time / 60:.2f} minutes.")


def main():
    parser = argparse.ArgumentParser(
        description="Fast Neural Style Transfer - Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Paths & dataset
    parser.add_argument("--dataset", "-d", type=str, required=True,
                        help="Path to folder containing content training images.")
    parser.add_argument("--style-image", "-i", type=str, required=True,
                        help="Path to style reference image.")
    parser.add_argument("--save-model-dir", type=str, default="./checkpoints",
                        help="Directory where trained model checkpoints will be saved.")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Prefix name for the saved model (defaults to style image name).")

    # Architecture & optimization
    parser.add_argument("--model-type", type=str, default="ae",
                        help="Architecture ('ae', 'bo', 'res', 'dense', 'ae_attn', 'ae_old').")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for Adam optimizer.")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Size (H=W) to which training images are cropped.")
    parser.add_argument("--style-size", type=int, default=None,
                        help="Size (H=W) of style image. Defaults to image-size if omitted.")

    # Loss weights
    parser.add_argument("--content-weight", "-c", type=float, default=1e5,
                        help="Weight for content loss.")
    parser.add_argument("--style-weight", "-s", type=float, default=1e10,
                        help="Weight for style loss.")
    parser.add_argument("--tv-weight", "-tv", type=float, default=1e0,
                        help="Weight for total variation (TV) loss.")
    parser.add_argument("--consistency-weight", "-cs", type=float, default=0.0,
                        help="Weight for noise consistency loss (default 0.0).")

    # Hardware & performance
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to train on ('auto', 'cuda', 'mps', 'cpu', 'cuda:0').")
    parser.add_argument("--mps", action="store_true", default=False,
                        help="Convenience flag to enable macOS Apple Silicon GPU.")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Number of DataLoader worker subprocesses.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="Number of batches between progress logs.")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
