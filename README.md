# Fast Neural Style Transfer in PyTorch :art: :rocket:

A modern, high-performance PyTorch refactoring of [**Perceptual Losses for Real-Time Style Transfer and Super-Resolution**](https://arxiv.org/abs/1603.08155) by *Justin Johnson, Alexandre Alahi, and Fei-Fei Li (ECCV 2016)*.

This implementation feeds content images directly through an image transformation network trained with perceptual loss functions derived from a pretrained VGG-16 network, enabling real-time, feedforward artistic stylization across consumer GPUs, Apple Silicon (MPS), and CPU environments.

---

## Visual Showcase :framed_picture:

| Content Image | Style Image | Stylized Output |
| :---: | :---: | :---: |
| <img src="pretrained_models\bear.jpg" height="220" alt="Content: Golden Gate"> | <img src="pretrained_models\Fauvism_André-Derain_Pier.jpg" height="220" alt="Style: Starry Night"> | <img src="pretrained_models\stylized.jpg" height="220" alt="Stylized Result"> |

---

## Key Improvements & Algorithmic Enhancements :sparkles:

Compared to the original formulation in Johnson et al. (2016), this implementation introduces several key algorithmic and architectural enhancements:

- **Resize-Convolution (Checkerboard Artifact Elimination)**: Replaces transposed convolutions (`ConvTranspose2d`) from the original paper with nearest-neighbor upsampling followed by convolution (`UpsampleConvLayer`), completely eliminating high-frequency checkerboard artifacts ([Odena et al., 2016](http://distill.pub/2016/deconv-checkerboard/)).
- **Reflection Padding**: Replaces zero-padding across all convolutional layers with reflection padding, preventing border discoloration, unnatural boundary halos, and edge artifacts.
- **Instance Normalization**: Utilizes Instance Normalization (`InstanceNorm2d`) instead of batch normalization, preventing contrast degradation and accelerating visual stylization convergence ([Ulyanov et al., 2016](https://arxiv.org/abs/1607.08022)).
- **Anisotropic Total Variation (TV) Regularization**: Incorporates spatial total variation loss ($\mathcal{L}_{tv}$) directly into the backpropagation chain, penalizing high-frequency noise and encouraging clean, coherent artistic stroke boundaries.
- **Noise Consistency Regularization**: Introduces an optional consistency loss between clean and Gaussian-perturbed inputs to improve model robustness and structural stability.
- **Advanced Generator Backbones**: Extends the classic 5-residual Autoencoder with multiple modern architectural variants:
  - **Bottleneck Residuals (`BottleNetwork`)**: 1×1 → 3×3 → 1×1 bottleneck blocks for parameter-efficient styling.
  - **Dense Connectivity (`DenseNet`)**: Multi-scale feature reuse via concatenated dense bottleneck blocks.
  - **Grouped Convolutions (`ResNeXt`)**: Cardinality-based aggregated transformations for richer texture expression.
  - **Self-Attention Mechanism (`AutoencoderAttention`)**: Global spatial context awareness to capture non-local style patterns.

---

## Getting Started :rocket:

### Prerequisites & Setup
* Python >= 3.10 (tested up to Python 3.12 / 3.13)
* PyTorch >= 2.0.0 with CUDA or MPS support

```bash
# 1. Clone repository
git clone https://github.com/Impasto-Lab/Fast-Style-Transfer.git
cd Fast-Style-Transfer

# 2. Create and activate a conda environment
conda create -n fst python=3.10 -y
conda activate fst

# 3. Install dependencies
pip install -r requirements.txt
```

> [!NOTE]
> On first run of training, pretrained VGG-16 weights (~528 MB) will be automatically downloaded and cached in your PyTorch hub directory.

---

## How to Use :computer:

### 1. Stylizing an Image (Inference)

#### Via Command Line (CLI)
Stylize any content image using a pretrained checkpoint:

```bash
python stylize.py -c pretrained_models/bear.jpg -m pretrained_models/Fauvism_André-Derain_Pier.pth --output-path ./outputs --output-name stylized_bear.jpg
```

Downscale large content images for faster processing or lower VRAM usage:
```bash
python stylize.py -c large_photo.jpg -m model.pth --content-scale 2.0 -o ./outputs
```

#### Via Python API (Direct Function Import)
You can directly import and use the stylization function in your own Python projects:

```python
from stylize import stylize_image

# Returns a PIL.Image instance
stylized_img = stylize_image(
    content_image_path="pretrained_models/bear.jpg",
    model_path="pretrained_models/Fauvism_André-Derain_Pier.pth",
    model_type="ae",
    scale=None,
    device="auto"  # or 'cuda', 'mps', 'cpu'
)

stylized_img.save("output.jpg")
```

---

### 2. Training a Style Model

To train your own artistic style transformation network, prepare a dataset of content images (e.g. [COCO 2017 Train](https://cocodataset.org/#download) or ImageNet):

```bash
python train.py --dataset /path/to/coco/train2017 \
                --style-image pretrained_models/Fauvism_André-Derain_Pier.jpg \
                --save-model-dir ./checkpoints \
                --batch-size 16 \
                --epochs 1
```

#### Tuning Loss Weights
Different style artworks may require balancing the content, style, and total variation weights:
```bash
python train.py --dataset /path/to/coco/train2017 \
                --style-image path/to/art.jpg \
                --content-weight 1e5 \
                --style-weight 1e10 \
                --tv-weight 1e0 \
                --batch-size 16
```

#### Training with Different Network Architectures
Select alternate generator architectures via `--model-type`:
```bash
# Options: 'ae' (Autoencoder), 'bo' (BottleNetwork), 'res' (ResNeXt), 'dense' (DenseNet), 'ae_attn' (Self-Attention)
python train.py --dataset /path/to/dataset \
                --style-image path/to/style.jpg \
                --model-type res \
                --batch-size 16
```

---

### 3. Batch Scripts
Batch shell scripts are available in the [`scripts/`](file:///e:/Projects/Fast-Style-Transfer/scripts/) directory:

* **Batch Stylization**:
  ```bash
  bash scripts/stylize.sh /path/to/input/images /path/to/model.pth ./batch_output
  ```
* **Hyperparameter Search / Training**:
  ```bash
  bash scripts/train.sh /path/to/coco/train2017 pretrained_models/Fauvism_André-Derain_Pier.jpg ./checkpoints
  ```

---

## Options & Command-Line Arguments :gear:

### `stylize.py` Options

| Flag | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--content-image` | `-c` | `str` | *Required* | Path to the content image you want to stylize. |
| `--model` | `-m` | `str` | *Required* | Path to the saved `.pth` model checkpoint. |
| `--output-path` | | `str` | `./` | Directory where the stylized image will be saved. |
| `--output-name` | | `str` | `stylized.jpg` | Filename of the output stylized image. |
| `--model-type` | | `str` | `ae` | Generator architecture (`ae`, `bo`, `res`, `dense`, `ae_attn`, `ae_old`). |
| `--content-scale` | | `float` | `None` | Downscaling factor for content image (e.g. `2.0` halves width and height). |
| `--device` | | `str` | `auto` | Execution device (`auto`, `cuda`, `mps`, `cpu`, `cuda:0`). |
| `--mps` | | `flag` | `False` | Convenience flag to force Apple Silicon GPU on macOS. |

---

### `train.py` Options

#### 1. Dataset & Paths
| Flag | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--dataset` | `-d` | `str` | *Required* | Path to directory containing content training images (flat or recursive). |
| `--style-image` | `-i` | `str` | *Required* | Path to reference style image. |
| `--save-model-dir` | | `str` | `./checkpoints` | Directory to save trained model checkpoints. |
| `--model-name` | | `str` | `None` | Base prefix for saved model filename (defaults to style image name). |

#### 2. Model Architecture & Optimization
| Flag | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--model-type` | | `str` | `ae` | Network architecture (`ae`, `bo`, `res`, `dense`, `ae_attn`, `ae_old`). |
| `--epochs` | | `int` | `1` | Total number of training epochs. |
| `--batch-size` | | `int` | `16` | Batch size for training. |
| `--lr` | | `float` | `1e-3` | Learning rate for Adam optimizer. |
| `--image-size` | | `int` | `256` | Crop size ($H=W$) of content training images. |
| `--style-size` | | `int` | `None` | Size ($H=W$) of style image (defaults to `--image-size`). |

#### 3. Loss Weights
| Flag | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--content-weight` | `-c` | `float` | `1e5` | Weight for VGG-16 content loss (at `relu2_2`). |
| `--style-weight` | `-s` | `float` | `1e10` | Weight for VGG-16 Gram matrix style loss. |
| `--tv-weight` | `-tv` | `float` | `1e0` | Weight for Total Variation (TV) spatial smoothing loss. |
| `--consistency-weight`| `-cs` | `float` | `0.0` | Weight for noise consistency loss (0.0 disables noisy pass). |

#### 4. Hardware & Logging
| Flag | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--device` | | `str` | `auto` | Execution device (`auto`, `cuda`, `mps`, `cpu`, `cuda:0`). |
| `--mps` | | `flag` | `False` | Convenience flag to enable Apple Silicon GPU. |
| `--num-workers` | | `int` | `2` | Number of worker processes for data loading. |
| `--seed` | | `int` | `42` | Random seed for reproducibility. |
| `--log-interval` | | `int` | `100` | Number of batches between console progress logs. |

---

## Project Structure :open_file_folder:

```text
Fast-Style-Transfer/
├── stylize.py                     # Primary CLI and Python API entry point for stylization
├── train.py                       # Primary CLI entry point for neural style transfer training
├── requirements.txt               # Required dependencies (torch, torchvision, pillow, numpy)
├── README.md                      # Project documentation and visual guides
├── pretrained_models/             # Pretrained weights and sample images
│   ├── bear.jpg                   # Sample content image
│   ├── Fauvism_André-Derain_Pier.jpg # Reference style image
│   ├── Fauvism_André-Derain_Pier.pth # Trained model weights
│   └── stylized.jpg               # Example stylized output
├── scripts/                       # Shell automation utilities
│   ├── stylize.sh                 # Batch stylization script
│   └── train.sh                   # Hyperparameter exploration and training script
└── src/                           # Core source modules
    ├── __init__.py
    ├── models/                    # Generator and feature extractor networks
    │   ├── __init__.py            # Model factory get_model() and registry
    │   ├── common.py              # Shared layers (ConvLayer, ResidualBlock, UpsampleConvLayer)
    │   ├── autoencoder.py         # Standard Johnson et al. Autoencoder
    │   ├── autoencoder_attention.py # Generator with bottleneck Self-Attention
    │   ├── autoencoder_old.py     # Transposed-convolution Autoencoder
    │   ├── bottle_net.py          # Bottleneck residual generator
    │   ├── dense_net.py           # DenseNet-style generator
    │   ├── resnext.py             # ResNeXt grouped-convolution generator
    │   └── vgg.py                 # Sliced VGG-16 feature extractor
    ├── losses/                    # Loss functions
    │   ├── __init__.py
    │   ├── perceptual.py          # Content loss, Gram matrix, Style loss
    │   └── regularizers.py        # Corrected Total Variation Loss, Consistency Loss
    ├── data/                      # Dataset handling
    │   ├── __init__.py
    │   └── dataset.py             # Robust ImageFolderDataset (recursive search & error tolerance)
    └── utils/                     # Helper utilities
        ├── __init__.py
        ├── image.py               # Robust Pillow 10+ image I/O and normalization
        └── device.py              # CUDA / MPS / CPU hardware detection
```

---

## References :books:

1. **J. Johnson, A. Alahi, L. Fei-Fei (2016)**. "Perceptual Losses for Real-Time Style Transfer and Super-Resolution". *European Conference on Computer Vision (ECCV)*. [[arXiv:1603.08155](https://arxiv.org/abs/1603.08155)]
2. **L. A. Gatys, A. S. Ecker, M. Bethge (2015)**. "A Neural Algorithm of Artistic Style". *Nature Communications / arXiv:1508.06576*. [[arXiv](https://arxiv.org/abs/1508.06576)]
3. **A. Odena, V. Dumoulin, C. Olah (2016)**. "Deconvolution and Checkerboard Artifacts". *Distill*. [[Article](http://distill.pub/2016/deconv-checkerboard/)]
4. **H. Zhang, I. Goodfellow, D. Metaxas, A. Odena (2019)**. "Self-Attention Generative Adversarial Networks". *ICML*. [[arXiv:1805.08318](https://arxiv.org/abs/1805.08318)]
5. **S. Xie, R. Girshick, P. Dollár, Z. Tu, K. He (2017)**. "Aggregated Residual Transformations for Deep Neural Networks". *CVPR*. [[arXiv:1611.05431](https://arxiv.org/abs/1611.05431)]
6. **D. Ulyanov, A. Vedaldi, V. Lempitsky (2016)**. "Instance Normalization: The Missing Ingredient for Fast Stylization". *arXiv:1607.08022*. [[arXiv:1607.08022](https://arxiv.org/abs/1607.08022)]

> ## Acknowledgments & License
>
> - This project is licensed under the terms of the [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
.
> - Code architecture and implementation are inspired by Justin Johnson's [fast-neural-style](https://github.com/jcjohnson/fast-neural-style) (BSD 3-Clause License).
> - The pretrained weights included in this repository were trained by us from scratch and are also provided under the MIT License.
