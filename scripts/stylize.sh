#!/usr/bin/env bash
# Batch stylization script
set -euo pipefail

COMMAND="python stylize.py"
IMAGES_DIR="${1:-./pretrained_models}"
MODEL_PATH="${2:-./pretrained_models/Fauvism_André-Derain_Pier.pth}"
OUTPUT_DIR="${3:-./stylized_images}"

mkdir -p "$OUTPUT_DIR"

echo "Stylizing images from '$IMAGES_DIR' using model '$MODEL_PATH'..."

for img in "$IMAGES_DIR"/*.jpg; do
    [ -e "$img" ] || continue
    filename=$(basename "$img")
    echo "Processing $filename..."
    $COMMAND --content-image "$img" \
             --model "$MODEL_PATH" \
             --output-path "$OUTPUT_DIR" \
             --output-name "$filename"
done

echo "Batch stylization completed. Output directory: $OUTPUT_DIR"
