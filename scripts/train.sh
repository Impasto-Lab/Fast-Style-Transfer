#!/usr/bin/env bash
# Training and hyperparameter tuning script
set -euo pipefail

COMMAND="python train.py"
DATASET_PATH="${1:-/path/to/coco/train2017}"
STYLE_IMAGE="${2:-pretrained_models/Fauvism_André-Derain_Pier.jpg}"
SAVE_DIR="${3:-./checkpoints}"

echo "Starting training workflow..."
echo "Dataset: $DATASET_PATH"
echo "Style Image: $STYLE_IMAGE"
echo "Save Directory: $SAVE_DIR"

# Example 1: Train baseline model
$COMMAND --dataset "$DATASET_PATH" \
         --style-image "$STYLE_IMAGE" \
         --save-model-dir "$SAVE_DIR" \
         --batch-size 16 \
         --epochs 1 \
         --content-weight 1e5 \
         --style-weight 1e10 \
         --tv-weight 1e0 \
         --consistency-weight 0.0

# Example 2: Train different network architectures (uncomment to run)
# architectures=("ae" "bo" "res" "dense" "ae_attn")
# for arch in "${architectures[@]}"; do
#     echo "Training architecture: $arch"
#     $COMMAND --dataset "$DATASET_PATH" \
#              --style-image "$STYLE_IMAGE" \
#              --model-type "$arch" \
#              --save-model-dir "$SAVE_DIR" \
#              --batch-size 16 \
#              --epochs 1
# done
