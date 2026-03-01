# OOD Detection Baselines

A modular PyTorch framework for benchmarking out-of-distribution (OOD) detection methods on CIFAR-10, CIFAR-100, and ImageNet-1K. Supports multiple architectures and a comprehensive suite of detection methods. Our method is represented by cosine_layers.

---

## Table of Contents

- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Data Setup](#data-setup)
- [Supported Models and Datasets](#supported-models-and-datasets)
- [Running the Code](#running-the-code)
  - [CIFAR-10 / CIFAR-100 (main.py)](#cifar-10--cifar-100-mainpy)
  - [ImageNet with ResNet-50 (main_for_resnet.py)](#imagenet-with-resnet-50-main_for_resnetpy)


---

## Dependencies

Install via pip:

```bash
pip install torch torchvision
pip install scikit-learn scipy numpy
```

**Full `requirements.txt`:**

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scikit-learn>=1.2.0
scipy>=1.10.0
```

A CUDA-capable GPU is strongly recommended. The code auto-detects GPU availability via `torch.cuda.is_available()`.

---

## Project Structure

```
.
├── main.py                  # Entry point for CIFAR-10/100 experiments
├── main_for_resnet.py       # Entry point for ImageNet / ResNet-50
├── models.py                # Model definitions (ResNet18, DenseNet100) + forward adapters
├── methods.py               # All OOD detection methods + metrics
├── loaders.py               # Dataset loaders, transforms, calibration loaders
├── evaluate_baselines.py    # Standalone baseline evaluation utilities
└── calibration_set_code.py  # Calibration set construction helpers
```

---

## Data Setup

All data is expected under a single root directory (default: `./data`). Set this with `--data_root`.

### CIFAR-10 / CIFAR-100

Downloaded automatically by torchvision on first run — no manual setup needed.

### ImageNet-1K

Place the validation set as an `ImageFolder`-compatible directory:

```
data/
└── imagenet-val/
    ├── n01440764/
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   └── ...
    └── ...
```

### OOD Datasets

ImageFolder-based OODs must be placed under `data/` with the following folder names:

| `--ood` argument | Expected folder path       |
|------------------|----------------------------|
| `inat`           | `data/iNaturalist/`        |
| `places`         | `data/Places/`             |
| `textures`       | `data/Textures/`           |
| `sun`            | `data/SUN/`                |
| `lsun`           | `data/LSUN/`               |
| `isun`           | `data/iSUN/`               |
| `svhn`           | Downloaded automatically   |
| `cifar10`        | Downloaded automatically   |
| `cifar100`       | Downloaded automatically   |

Each ImageFolder directory must contain one or more class subdirectories with image files (`.jpg`, `.png`, `.jpeg`, etc.).

For a **custom OOD dataset** in ImageFolder format, pass `--ood_path /path/to/your/folder`.

---

## Supported Models and Datasets

| `--model`      | `--id`           | Pretrained weights source |
|----------------|------------------|--------------------------|
| `resnet18`     | `cifar10`, `cifar100` | Custom checkpoint (`--ckpt`) |
| `densenet100`  | `cifar10`, `cifar100` | Custom checkpoint (`--ckpt`) |
| `resnet50`     | `imagenet`       | `torchvision` (ImageNet pretrained) |
| `mobilenet`    | `imagenet`       | `torchvision` (ImageNet pretrained) |

> **Note:** For `resnet50` and `mobilenet`, `--ckpt` is ignored. The torchvision pretrained weights are loaded automatically.

---

## Running the Code

### CIFAR-10 / CIFAR-100 (`main.py`)

Use this script for experiments with `resnet18` or `densenet100` on CIFAR-10/100.

**Minimal example — CIFAR-10 with DenseNet100 vs SVHN:**

```bash
python main.py \
  --id cifar10 \
  --ood svhn \
  --model densenet100 \
  --ckpt /path/to/densenet100_cif10/best.pth \
  --data_root ./data
```

**CIFAR-100 with ResNet18:**

```bash
python main.py \
  --id cifar100 \
  --ood cifar10 \
  --model resnet18 \
  --ckpt /path/to/resnet18_cifar100/best.pth \
  --data_root ./data \
  --batch 512
```

**Run only specific methods:**

```bash
python main.py \
  --id cifar10 \
  --ood svhn \
  --model resnet18 \
  --ckpt /path/to/best.pth \
  --methods msp energy maha cosine_layers
```

**Tuning the `cosine_layers` method:**

```bash
python main.py \
  --id cifar10 \
  --ood svhn \
  --model densenet100 \
  --ckpt /path/to/best.pth \
  --methods cosine_layers \
  --cos_layers 1,2,3 \
  --cos_layer_weights 1,1,1 \
  --cos_calib_per_class 5000
```

---

### ImageNet with ResNet-50 (`main_for_resnet.py`)

This script is optimised for ImageNet-scale evaluation. Training-set feature collection is skipped (not needed for `cosine_layers`). Only `cosine_layers` is active by default; other methods are commented out.

**Minimal example:**

```bash
python main_for_resnet.py \
  --id imagenet \
  --ood inat \
  --data_root /path/to/data
```

**Changing the OOD benchmark:**

```bash
python main_for_resnet.py --ood places
python main_for_resnet.py --ood textures
python main_for_resnet.py --ood sun
```

**Selecting which intermediate layers to use:**

```bash
python main_for_resnet.py \
  --ood inat \
  --cos_layers 3,4 \
  --cos_layer_weights 1,1
```

Layer indices map to the following feature tensors:

| Layer index | Feature                          | Architecture |
|-------------|----------------------------------|--------------|
| `1`         | After `layer2` (`out2`)          | ResNet-50    |
| `2`         | After `layer3` (`out3`)          | ResNet-50    |
| `3`         | After `layer4` (`out4`)          | ResNet-50    |
| `4`         | Penultimate (post-GAP)           | ResNet-50    |
| `1`         | After DenseBlock2 (`out2`)       | DenseNet100  |
| `2`         | After DenseBlock3 (`out3`)       | DenseNet100  |
| `3`         | After norm_final (`out4`)        | DenseNet100  |


