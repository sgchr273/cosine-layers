import os
import random
from typing import Optional, Dict

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision
from torchvision.transforms import RandAugment
from torchvision.datasets.folder import IMG_EXTENSIONS


# Defaults are overridden by runner via CLI
BATCH = 256
NUM_WORKERS = 4

NORMS: Dict[str, Dict[str, tuple]] = {
    "cifar10": dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    "cifar100": dict(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2761)),
    "imagenet": dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    "cif10_densenet": dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
}


def make_calib_loader(dataset_name: str, tf, calib_num:int, data_root: str = "./data"):
    dn = dataset_name.lower()
    if dn == "cifar10":
        calib_set = datasets.CIFAR10(data_root, train=True, download=True, transform=tf)
    elif dn == "cifar100":
        calib_set = datasets.CIFAR100(data_root, train=True, download=True, transform=tf)
    elif dn == "imagenet":
        calib_set = datasets.ImageFolder(os.path.join(data_root, "imagenet-val"), transform=tf)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    calib_subset, _ = torch.utils.data.random_split(calib_set, [calib_num, len(calib_set) - calib_num])
    calib_loader = DataLoader(calib_subset, batch_size=BATCH, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    return calib_loader

def build_transform(dataset_name: str, model: str, input_size: int, train: bool = True) -> transforms.Compose:
    dn = dataset_name.lower()
    arch = model.lower()

    is_cifar = dn in {"cifar10", "cifar100"}
    is_densenet = arch in {"densenet100", "densenet-100"}

    # --- Special case: CIFAR-{10,100} with DenseNet → shared transform ---
    if is_cifar and is_densenet:
        # pick whatever norm you want for DenseNet on CIFAR; two options shown:
        # 1) unified DenseNet-on-CIFAR stats:
        norm = NORMS.get("cif10_densenet", NORMS["cifar10"])
        # 2) OR per-dataset stats (uncomment this to use dataset-specific norms):
        # norm = NORMS[dn]

        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                # add RandAugment if desired for this special case too
                # RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(norm["mean"], norm["std"]),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(norm["mean"], norm["std"]),
            ])

    # --- Default path: choose norms by dataset ---
    if dn.startswith("imagenet"):
        norm = NORMS["imagenet"]
    elif dn == "cifar100":
        norm = NORMS["cifar100"]
    elif dn == "cifar10":
        norm = NORMS["cifar10"]
    else:
        norm = NORMS["cifar10"]  # sensible fallback

    # --- Train/Test transforms (default path) ---
    if train:
        return transforms.Compose([
            transforms.RandomCrop(input_size, padding=4),
            transforms.RandomHorizontalFlip(),
            RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(norm["mean"], norm["std"]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(norm["mean"], norm["std"]),
        ])



def load_id_dataset(name: str, split: str, tfm: transforms.Compose, data_root: str = "./data"):
    name = name.lower()
    if name == "cifar10":
        return datasets.CIFAR10(data_root, train=(split == "train"), download=True, transform=tfm)
    if name == "cifar100":
        return datasets.CIFAR100(data_root, train=(split == "train"), download=True, transform=tfm)
    if name in ["imagenet", "imagenet1k", "imagenet-1k"]:
        sub = "train" if split == "train" else "val"
        return torchvision.datasets.ImageFolder(os.path.join(data_root, "imagenet-val"), transform=tfm)
    raise ValueError(f"Unsupported ID dataset: {name}")

def load_ood_dataset(
    name: str,
    tfm: transforms.Compose,
    data_root: str = "./data",
    path_override: Optional[str] = None,
):
    """
    OOD datasets:
      - Built-ins: svhn, cifar10, cifar100  (torchvision datasets)
      - ImageFolder-based: isun, lsun, places, inat, textures
      - Optional custom ImageFolder via path_override (NOT for CIFAR)
    """
    name = name.lower()

    # Custom ImageFolder override (not for CIFAR)
    if path_override is not None:
        if name in {"cifar10", "cifar100"}:
            raise ValueError(
                "Do not use path_override for CIFAR. Use name='cifar10' or 'cifar100' "
                "to load via torchvision (they are not stored as image files)."
            )
        ds = torchvision.datasets.ImageFolder(path_override, transform=tfm)
        if len(ds.samples) == 0:
            raise FileNotFoundError(
                f"No images found in '{path_override}'. Expected extensions: {IMG_EXTENSIONS}"
            )
        return ds

    # Built-in datasets (non-ImageFolder)
    if name == "svhn":
        return datasets.SVHN(data_root, split="test", download=True, transform=tfm)
    if name == "cifar10":
        return datasets.CIFAR10(data_root, train=False, download=True, transform=tfm)
    if name == "cifar100":
        return datasets.CIFAR100(data_root, train=False, download=True, transform=tfm)

    # ImageFolder-based OODs
    folder_map = {
        "isun":     "iSUN",
        "lsun":     "LSUN",
        "places":   "Places",
        "inat":     "iNaturalist",
        "textures": "Textures",
        "sun":      "SUN", 
    }
    if name in folder_map:
        root = os.path.join(data_root, folder_map[name])
        ds = torchvision.datasets.ImageFolder(root, transform=tfm)
        if len(ds.samples) == 0:
            raise FileNotFoundError(
                f"No images found in '{root}'. Expected extensions: {IMG_EXTENSIONS}"
            )
        return ds

    raise ValueError(f"Unsupported OOD dataset: {name}")

def make_loader(ds, batch: int = BATCH, shuffle: bool = False, num_workers: int = NUM_WORKERS):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

def make_bank_subset(id_train_set, per_class: int, num_classes: int):
    cls_to_idx = {c: [] for c in range(num_classes)}
    for idx in range(len(id_train_set)):
        _, y = id_train_set[idx]
        cls_to_idx[int(y)].append(idx)
    bank_indices = []
    for c in range(num_classes):
        idxs = cls_to_idx[c]
        random.shuffle(idxs)
        bank_indices.extend(idxs[:per_class])
    return Subset(id_train_set, bank_indices)

def make_calib_subset(id_train_set, per_class: int, num_classes: int):
    # For cosine_layers calibration
    return make_bank_subset(id_train_set, per_class, num_classes)