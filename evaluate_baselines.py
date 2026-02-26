# Project layout

```
project/
├─ loaders.py        # datasets, transforms, loaders, helpers
├─ models.py         # model factory, unified forward, classifier params, seeding
├─ methods.py        # all OOD methods incl. cosine_layers
└─ runner.py         # CLI entrypoint that wires everything together
```

---

## loaders.py
```python
import os
import random
from typing import Optional, Dict

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision

# Defaults are overridden by runner via CLI
BATCH = 256
NUM_WORKERS = 4

NORMS: Dict[str, Dict[str, tuple]] = {
    "cifar10": dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    "cifar100": dict(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2761)),
    "imagenet": dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
}

def build_transform(dataset_name: str, input_size: int) -> transforms.Compose:
    norm = NORMS["imagenet"] if dataset_name.startswith("imagenet") else (
           NORMS["cifar100"] if dataset_name == "cifar100" else NORMS["cifar10"])
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
        return torchvision.datasets.ImageFolder(os.path.join(data_root, "imagenet-1k", sub), transform=tfm)
    raise ValueError(f"Unsupported ID dataset: {name}")

def load_ood_dataset(name: str, tfm: transforms.Compose, data_root: str = "./data", path_override: Optional[str] = None):
    name = name.lower()
    if path_override:
        return torchvision.datasets.ImageFolder(path_override, transform=tfm)
    if name == "svhn":
        return datasets.SVHN(data_root, split="test", download=True, transform=tfm)
    if name in ["isun", "lsun", "places", "inat", "textures", "cifar100", "cifar10"]:
        folder = {
            "isun": "ood_data/iSUN",
            "lsun": "ood_data/LSUN",
            "places": "ood_data/Places",
            "inat": "ood_data/iNaturalist",
            "textures": "ood_data/Textures",
            "cifar100": "ood_data/CIFAR100",
            "cifar10": "ood_data/CIFAR10",
        }[name]
        return torchvision.datasets.ImageFolder(os.path.join(data_root, folder), transform=tfm)
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
```

---

## models.py
```python
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm

# Import your custom DenseNet (la, lb, penult, outs)
from eval_dense_ood import load_densenet_ckpt_into_ood_model, DenseNet100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class ModelSpec:
    arch: str                 # "resnet18" | "densenet100" | "vit" | "swin"
    num_classes: int
    ckpt: Optional[str] = None

def set_all_seeds(seed: int = 0):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False


def _load_strict(m: nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model", ckpt)
    m.load_state_dict(sd, strict=False)
    m.eval()


def build_model(spec: ModelSpec) -> nn.Module:
    arch = spec.arch.lower()
    nc = spec.num_classes

    if arch == "densenet100":
        m = DenseNet100(num_classes=nc)
        if spec.ckpt:
            load_densenet_ckpt_into_ood_model(m, spec.ckpt, DEVICE)
        return m.to(DEVICE)

    if arch == "resnet18":
        m = tvm.resnet18(num_classes=nc)
        if spec.ckpt:
            _load_strict(m, spec.ckpt)
        return m.to(DEVICE)

    if arch == "vit":
        m = tvm.vit_b_16(weights=None)
        m.heads.head = nn.Linear(m.heads.head.in_features, nc)
        if spec.ckpt:
            _load_strict(m, spec.ckpt)
        return m.to(DEVICE)

    if arch == "swin":
        m = tvm.swin_t(weights=None)
        m.head = nn.Linear(m.head.in_features, nc)
        if spec.ckpt:
            _load_strict(m, spec.ckpt)
        return m.to(DEVICE)

    raise ValueError(f"Unsupported model arch: {spec.arch}")

# ---- Unified forward that returns (logits, logits_aux, penult, extra) ----

def forward_adapt(model: nn.Module, x: torch.Tensor, arch: str):
    model.eval()
    with torch.no_grad():
        out = model(x)
        if isinstance(out, tuple) and len(out) >= 3:
            logits, logits_aux, penult = out[0], out[1], out[2]
            extra = out[3] if len(out) >= 4 else None
            return logits, logits_aux, penult, extra
        # torchvision paths
        penult = _get_penultimate(model, x, arch)
        logits = model(x)
        extra = _get_resnet_intermediates(model, x) if arch.startswith("resnet") else None
        return logits, None, penult, extra


def _get_penultimate(model: nn.Module, x: torch.Tensor, arch: str) -> torch.Tensor:
    grabbed = {}

    def hook_fn(_, __, output):
        grabbed["feat"] = output.flatten(1).detach()

    handle = None
    try:
        if arch.startswith("resnet"):
            handle = model.avgpool.register_forward_hook(hook_fn)
            _ = model(x)
        elif arch == "densenet100":
            feat = model.features(x).relu()
            pooled = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
            grabbed["feat"] = pooled
        elif arch == "vit":
            handle = model.encoder.ln.register_forward_hook(hook_fn)
            _ = model(x)
        elif arch == "swin":
            handle = model.norm.register_forward_hook(hook_fn)
            _ = model(x)
        else:
            feat = list(model.children())[-2](x)
            grabbed["feat"] = feat.flatten(1)
    finally:
        if handle is not None:
            handle.remove()

    if "feat" not in grabbed:
        raise RuntimeError("Failed to grab penultimate features for this architecture.")
    return grabbed["feat"].to(x.device)

# resnet taps -> [out1,out2,out3,out4]

def _get_resnet_intermediates(model: nn.Module, x: torch.Tensor) -> List[torch.Tensor]:
    out = model.relu(model.bn1(model.conv1(x)))
    out1 = model.layer1(out)
    out2 = model.layer2(out1)
    out3 = model.layer3(out2)
    out4 = model.layer4(out3)
    return [out1, out2, out3, out4]


def get_classifier_params(model: nn.Module):
    import numpy as np
    m = getattr(model, "module", model)
    for attr in ("classifier", "linear", "fc", "head"):
        layer = getattr(m, attr, None)
        if isinstance(layer, nn.Linear):
            W = layer.weight.detach().cpu().numpy()
            b = layer.bias.detach().cpu().numpy() if layer.bias is not None else np.zeros(W.shape[0])
            return W, b
    last_linear = None
    for _, layer in m.named_modules():
        if isinstance(layer, nn.Linear):
            last_linear = layer
    if last_linear is None:
        raise AttributeError("No nn.Linear classifier layer found.")
    W = last_linear.weight.detach().cpu().numpy()
    b = last_linear.bias.detach().cpu().numpy() if last_linear.bias is not None else np.zeros(W.shape[0])
    return W, b
```

---

## methods.py
```python
from typing import Tuple, List, Dict
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import EmpiricalCovariance
from numpy.linalg import norm, pinv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numpy import linalg as LA

from models import DEVICE, forward_adapt

# ---------- Metrics & utils ----------
RECALL = 0.95

def logsumexp(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(arr, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(arr - m), axis=axis, keepdims=True))).squeeze(axis)

class Metrics:
    @staticmethod
    def auc(id_scores, ood_scores):
        from sklearn.metrics import roc_auc_score
        y = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
        s = np.concatenate([id_scores, ood_scores])
        return float(roc_auc_score(y, s))

    @staticmethod
    def fpr_recall(id_scores, ood_scores, target_recall=RECALL):
        from sklearn.metrics import roc_curve
        y = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
        s = np.concatenate([id_scores, ood_scores])
        fpr, tpr, _ = roc_curve(y, s)
        mask = tpr >= target_recall
        idx = np.where(mask)[0][0] if np.any(mask) else np.argmax(tpr)
        return float(fpr[idx]), float(tpr[idx])

# ---------- Collectors ----------
@torch.no_grad()
def collect_logits_softmax(model, loader, arch):
    all_logits, all_softmax = [], []
    for x, *_ in loader:
        x = x.to(DEVICE, non_blocking=True)
        logits, *_ = forward_adapt(model, x, arch)
        sm = F.softmax(logits, dim=-1)
        all_logits.append(logits.float().cpu())
        all_softmax.append(sm.float().cpu())
    logits_np = torch.cat(all_logits, dim=0).numpy()
    softmax_np = torch.cat(all_softmax, dim=0).numpy()
    return logits_np, softmax_np

@torch.no_grad()
def collect_penultimate_and_labels(model, loader, arch):
    feats, labels = [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        logits, logits_aux, penult, _ = forward_adapt(model, x, arch)
        feats.append(penult.detach().cpu())
        labels.append(y)
    feats = torch.cat(feats, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return feats, labels

# ---------- Baselines (MSP, MaxLogit, Energy) ----------

def method_msp(id_sm, ood_sm, name):
    score_id = id_sm.max(axis=-1)
    score_ood = ood_sm.max(axis=-1)
    auc = Metrics.auc(-score_id, -score_ood)
    fpr, _ = Metrics.fpr_recall(-score_id, -score_ood)
    print(f"MSP: {name} auroc {auc:.2%}, fpr {fpr:.2%}")

def method_maxlogit(id_logits, ood_logits, name):
    score_id = id_logits.max(axis=-1)
    score_ood = ood_logits.max(axis=-1)
    auc = Metrics.auc(-score_id, -score_ood)
    fpr, _ = Metrics.fpr_recall(-score_id, -score_ood)
    print(f"MaxLogit: {name} auroc {auc:.2%}, fpr {fpr:.2%}")

def method_energy(id_logits, ood_logits, name):
    score_id = logsumexp(id_logits, axis=-1)
    score_ood = logsumexp(ood_logits, axis=-1)
    auc = Metrics.auc(-score_id, -score_ood)
    fpr, _ = Metrics.fpr_recall(-score_id, -score_ood)
    print(f"Energy: {name} auroc {auc:.2%}, fpr {fpr:.2%}")

# ---------- Mahalanobis ----------

def method_mahalanobis(feature_id_train, train_labels, feature_id_val, feature_ood, num_classes, name):
    print('[Mahalanobis] fitting class means/precision...')
    train_means, centered = [], []
    for c in range(num_classes):
        fs = feature_id_train[train_labels == c]
        m = fs.mean(axis=0)
        train_means.append(m)
        centered.append(fs - m)
    centered = np.concatenate(centered, axis=0)
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(centered.astype(np.float64))
    mean = torch.from_numpy(np.stack(train_means)).to(DEVICE).float()
    prec = torch.from_numpy(ec.precision_).to(DEVICE).float()

    def min_maha_sq(X):
        X = torch.from_numpy(X).to(DEVICE).float()
        XC = X[:, None, :] - mean[None, :, :]
        t = torch.matmul(XC, prec)
        d2 = (t * XC).sum(dim=-1)
        return d2.min(dim=1).values.detach().cpu().numpy()

    score_id = -min_maha_sq(feature_id_val)
    score_ood = -min_maha_sq(feature_ood)
    auc = Metrics.auc(score_id, score_ood)
    fpr, _ = Metrics.fpr_recall(score_id, score_ood)
    print(f"Mahalanobis: {name} auroc {auc:.2%}, fpr {fpr:.2%}")

# ---------- GradNorm ----------

def gradnorm(x, w, b, num_classes):
    fc = torch.nn.Linear(*w.shape[::-1], bias=True).to(DEVICE)
    with torch.no_grad():
        fc.weight.copy_(torch.from_numpy(w))
        fc.bias.copy_(torch.from_numpy(b))
    x = torch.from_numpy(x).float().to(DEVICE)
    logsoftmax = torch.nn.LogSoftmax(dim=-1).to(DEVICE)
    confs = []
    for i in x:
        targets = torch.ones((1, num_classes), device=DEVICE)
        fc.zero_grad(set_to_none=True)
        loss = torch.mean(torch.sum(-targets * logsoftmax(fc(i[None])), dim=-1))
        loss.backward()
        layer_grad_norm = torch.sum(torch.abs(fc.weight.grad)).detach().cpu().numpy()
        confs.append(layer_grad_norm)
    return np.array(confs)

# ---------- NNGuide ----------

def baseconf_msp_torch(logits: torch.Tensor):
    return logits.softmax(dim=-1).amax(dim=-1)

def baseconf_maxlogit_torch(logits: torch.Tensor):
    return logits.max(dim=-1).values

def baseconf_energy_torch(logits: torch.Tensor):
    m = logits.max(dim=-1, keepdim=True).values
    return (m + (logits - m).exp().sum(dim=-1, keepdim=True).log()).squeeze(-1)

@torch.no_grad()
def collect_feats_and_conf(model, loader, conf_fn, arch):
    feats, confs = [], []
    for x, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        logits, logits_aux, penult, _ = forward_adapt(model, x, arch)
        feats.append(penult)
        confs.append(conf_fn(logits))
    Z = torch.cat(feats, dim=0)
    S = torch.cat(confs, dim=0).unsqueeze(1)
    return Z, S

@torch.no_grad()
def nnguide_scores(z, s, Z_bank, S_bank, k):
    z_norm = F.normalize(z, p=2, dim=-1)
    Z_bank_norm = F.normalize(Z_bank, p=2, dim=-1)
    Z_weighted = Z_bank_norm * S_bank
    sim = z_norm @ Z_weighted.t()
    g = sim.topk(k, dim=1).values.mean(dim=1)
    guided = s * g
    return guided, -guided

@torch.no_grad()
def nnguide_score_loader(model, loader, Z_bank, S_bank, k, conf_fn, arch, return_ood_higher=True):
    all_guided = []
    for x, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        logits, _, penult, _ = forward_adapt(model, x, arch)
        s = conf_fn(logits)
        guided_id, guided_ood = nnguide_scores(penult, s, Z_bank, S_bank, k)
        all_guided.append(guided_ood if return_ood_higher else guided_id)
    return torch.cat(all_guided, dim=0).cpu().numpy()

# ---------- NECO ----------

def neco(feature_id_train, feature_id_val, feature_ood, logit_id_val, logit_ood, model_architecture_type, neco_dim):
    ss = StandardScaler()
    complete_vectors_train = ss.fit_transform(feature_id_train)
    complete_vectors_test = ss.transform(feature_id_val)
    complete_vectors_ood = ss.transform(feature_ood)
    pca_estimator = PCA(feature_id_train.shape[1])
    _ = pca_estimator.fit_transform(complete_vectors_train)
    cls_test_reduced_all = pca_estimator.transform(complete_vectors_test)
    cls_ood_reduced_all = pca_estimator.transform(complete_vectors_ood)
    score_id_maxlogit = logit_id_val.max(axis=-1)
    score_ood_maxlogit = logit_ood.max(axis=-1)
    if model_architecture_type in ['deit', 'swin']:
        complete_vectors_train = feature_id_train
        complete_vectors_test = feature_id_val
        complete_vectors_ood = feature_ood
    cls_test_reduced = cls_test_reduced_all[:, :neco_dim]
    cls_ood_reduced = cls_ood_reduced_all[:, :neco_dim]

    def _ratio_rows(Afull, Ared):
        out = []
        for i in range(Ared.shape[0]):
            sc_complet = LA.norm(Afull[i, :])
            sc = LA.norm(Ared[i, :])
            out.append(sc / sc_complet if sc_complet > 0 else 0.0)
        return np.array(out)

    l_ID = _ratio_rows(complete_vectors_test, cls_test_reduced)
    l_OOD = _ratio_rows(complete_vectors_ood, cls_ood_reduced)
    score_id, score_ood = l_ID, l_OOD
    if model_architecture_type != 'resnet':
        score_id *= score_id_maxlogit
        score_ood *= score_ood_maxlogit
    auc = Metrics.auc(-score_id, -score_ood)
    fpr, _ = Metrics.fpr_recall(-score_id, -score_ood, RECALL)
    print(f'NECO: auroc {auc:.2%}, fpr {fpr:.2%}')

# ---------- ReAct ----------

def react(feature_id_val, feature_ood, clip, w, b, name):
    logit_id_clip = np.clip(feature_id_val, a_min=None, a_max=clip) @ w.T + b
    logit_ood_clip = np.clip(feature_ood, a_min=None, a_max=clip) @ w.T + b
    score_id = logsumexp(logit_id_clip, axis=-1)
    score_ood = logsumexp(logit_ood_clip, axis=-1)
    auc = Metrics.auc(-score_id, -score_ood)
    fpr, _ = Metrics.fpr_recall(-score_id, -score_ood, RECALL)
    print(f"ReAct: {name} auroc {auc:.2%}, fpr {fpr:.2%}")

# ---------- ViM ----------

def vim(feature_id_train, feature_id_val, feature_ood,
        logit_id_train, logit_id_val, ood_logits,
        name, model_architecture_type, model_name, u):
    DIM = 1000 if feature_id_val.shape[-1] >= 2048 else 512
    if model_architecture_type == "resnet" and model_name in ["resnet34", "resnet18"]:
        DIM = 300
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(feature_id_train - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
    vlogit_id_train = norm(np.matmul(feature_id_train - u, NS), axis=-1)
    alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
    vlogit_id_val = norm(np.matmul(feature_id_val - u, NS), axis=-1) * alpha
    vlogit_ood = norm(np.matmul(feature_ood - u, NS), axis=-1) * alpha
    energy_id_val = logsumexp(logit_id_val, axis=-1)
    energy_ood = logsumexp(ood_logits, axis=-1)
    score_id = -( -vlogit_id_val + energy_id_val )
    score_ood = -( -vlogit_ood + energy_ood )
    auc = Metrics.auc(score_id, score_ood)
    fpr, _ = Metrics.fpr_recall(score_id, score_ood, RECALL)
    print(f"ViM: {name} auroc {auc:.2%}, fpr {fpr:.2%}")

# ---------- Cosine-prototype (cosine_layers) ----------
from models import _get_resnet_intermediates  # for taps

def _gap(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=(2, 3))

@torch.no_grad()
def _iter_feats_layers(model: torch.nn.Module, loader, arch: str, LAYERS: List[int]):
    if arch in ("vit", "swin"):
        raise NotImplementedError("cosine_layers supports resnet-like & custom DenseNet paths only.")
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        logits, logits_aux, penult, extra = forward_adapt(model, x, arch)
        if arch.startswith("resnet"):
            outs = _get_resnet_intermediates(model, x)
        else:
            if extra is None:
                raise RuntimeError("DenseNet path did not return intermediates for cosine_layers.")
            outs = extra
        feats = []
        for li in LAYERS:  # 1->out2, 2->out3, 3->out4
            feats.append(F.normalize(_gap(outs[li]), dim=1))
        yield feats, y

@torch.no_grad()
def cosine_build_prototypes(model, calib_loader, arch: str, LAYERS: List[int], num_classes: int):
    first_feats, first_y = next(_iter_feats_layers(model, calib_loader, arch, LAYERS))
    dims = [f.size(1) for f in first_feats]
    sums: Dict[int, List[torch.Tensor]] = {li: [torch.zeros(dims[k], device=DEVICE) for _ in range(num_classes)]
                                           for k, li in enumerate(LAYERS)}
    counts: Dict[int, torch.Tensor] = {li: torch.zeros(num_classes, device=DEVICE) for li in LAYERS}

    def update(feats_list, y):
        for k, li in enumerate(LAYERS):
            Fk = feats_list[k]
            for cls in range(num_classes):
                mask = (y == cls)
                if mask.any():
                    sums[li][cls] += Fk[mask].sum(dim=0)
                    counts[li][cls] += mask.sum()

    update(first_feats, first_y)
    for feats_list, y in _iter_feats_layers(model, calib_loader, arch, LAYERS):
        update(feats_list, y)

    prototypes: Dict[int, torch.Tensor] = {}
    for li in LAYERS:
        plist = []
        for cls in range(num_classes):
            if counts[li][cls] > 0:
                p = sums[li][cls] / counts[li][cls]
                p = F.normalize(p, dim=0)
            else:
                p = F.normalize(torch.randn_like(sums[li][0]), dim=0)
            plist.append(p)
        prototypes[li] = torch.stack(plist, dim=0)
    return prototypes

@torch.no_grad()
def cosine_score_loader(model, loader, arch: str, prototypes: Dict[int, torch.Tensor],
                        LAYERS: List[int], layer_weights: Dict[int, float]):
    wsum = sum(layer_weights[li] for li in LAYERS)
    scores_out = []
    for feats_list, _ in _iter_feats_layers(model, loader, arch, LAYERS):
        per_layer_max = []
        for k, li in enumerate(LAYERS):
            f = feats_list[k]
            P = prototypes[li].to(f.device)
            cos = f @ P.t()
            max_cos, _ = cos.max(dim=1)
            per_layer_max.append(layer_weights[li] * max_cos)
        avg_max_cos = per_layer_max[0]
        for t in per_layer_max[1:]:
            avg_max_cos = avg_max_cos + t
        avg_max_cos = avg_max_cos / wsum
        ood_score = 1.0 - avg_max_cos
        scores_out.append(ood_score.detach().cpu())
    return torch.cat(scores_out, dim=0).numpy()

def method_cosine_layers(model, arch: str, id_train_set, id_val_loader, ood_loader,
                         num_classes: int, cos_layers: List[int], layer_weights: Dict[int, float],
                         calib_subset_maker, calib_per_class: int):
    calib_subset = calib_subset_maker(id_train_set, calib_per_class, num_classes)
    calib_loader = torch.utils.data.DataLoader(calib_subset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    print("[cosine_layers] Building prototypes...")
    protos = cosine_build_prototypes(model, calib_loader, arch, cos_layers, num_classes)
    print("[cosine_layers] Scoring ID and OOD...")
    id_scores = cosine_score_loader(model, id_val_loader, arch, protos, cos_layers, layer_weights)
    ood_scores = cosine_score_loader(model, ood_loader, arch, protos, cos_layers, layer_weights)
    auc = Metrics.auc(id_scores, ood_scores)
    fpr, _ = Metrics.fpr_recall(id_scores, ood_scores, RECALL)
    print(f"cosine_layers: AUROC {auc:.2%}, FPR@95 {fpr:.2%}")
```

---

## runner.py
```python
#!/usr/bin/env python3
import argparse
import numpy as np
import torch

from loaders import build_transform, load_id_dataset, load_ood_dataset, make_loader, make_bank_subset, make_calib_subset
from models import ModelSpec, build_model, set_all_seeds, DEVICE
from models import get_classifier_params
from methods import (
    collect_logits_softmax, collect_penultimate_and_labels,
    method_msp, method_maxlogit, method_energy,
    method_mahalanobis, gradnorm,
    baseconf_msp_torch, collect_feats_and_conf, nnguide_score_loader,
    neco, react, vim, Metrics
)
from methods import method_cosine_layers
from numpy.linalg import pinv

DEFAULT_METHODS = [
    "msp", "maxlogit", "energy", "maha", "gradnorm",
    "nnguide", "neco", "react", "vim", "cosine_layers"
]


def parse_args():
    p = argparse.ArgumentParser(description="Flexible OOD Baselines Runner (modular)")
    p.add_argument("--id", required=True, choices=["cifar10", "cifar100", "imagenet"], help="ID dataset")
    p.add_argument("--ood", required=True, help="OOD name (svhn, isun, lsun, places, inat, textures, cifar10, cifar100) or custom folder with --ood_path")
    p.add_argument("--ood_path", default=None, help="ImageFolder path for custom OOD")
    p.add_argument("--model", required=True, choices=["resnet18", "densenet100", "vit", "swin"])
    p.add_argument("--ckpt", default=None, help="Model checkpoint path")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--bank_per_class", type=int, default=100)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--data_root", default="./data")
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)

    # cosine_layers
    p.add_argument("--cos_layers", default="1,2,3", help="Comma list among {1,2,3} mapping to out2,out3,out4")
    p.add_argument("--cos_layer_weights", default="1,1,1", help="Comma list matching cos_layers")
    p.add_argument("--cos_calib_per_class", type=int, default=100)
    return p.parse_args()


def _parse_layers_arg(arg: str):
    parts = [p.strip() for p in arg.split(',') if p.strip()]
    layers = [int(p) for p in parts]
    for li in layers:
        if li not in (1, 2, 3):
            raise ValueError("--cos_layers must be among 1,2,3 (maps to out2,out3,out4)")
    return layers


def main():
    args = parse_args()
    set_all_seeds(0)

    # Input size
    input_size = 224 if args.id == "imagenet" or args.model in ["vit", "swin"] else 32

    # Datasets & loaders
    id_train_tf = build_transform(args.id, input_size)
    id_test_tf = build_transform(args.id, input_size)
    ood_tf = build_transform(args.id, input_size)

    id_train_set = load_id_dataset(args.id, "train", id_train_tf, data_root=args.data_root)
    id_test_set = load_id_dataset(args.id, "val", id_test_tf, data_root=args.data_root)
    ood_set = load_ood_dataset(args.ood, ood_tf, data_root=args.data_root, path_override=args.ood_path)

    id_train_loader = make_loader(id_train_set, batch=args.batch, shuffle=False, num_workers=args.workers)
    id_val_loader = make_loader(id_test_set, batch=args.batch, shuffle=False, num_workers=args.workers)
    ood_loader = make_loader(ood_set, batch=args.batch, shuffle=False, num_workers=args.workers)

    # Model
    num_classes = 1000 if args.id == "imagenet" else (100 if args.id == "cifar100" else 10)
    spec = ModelSpec(arch=args.model, num_classes=num_classes, ckpt=args.ckpt)
    model = build_model(spec)
    arch = args.model.lower()

    # Sanity: ID accuracy
    @torch.no_grad()
    def evaluate_id_acc(m):
        m.eval()
        total = correct = 0
        for x, y in id_val_loader:
            x = x.to(DEVICE); y = y.to(DEVICE)
            logits, _, _, _ = forward_adapt(m, x, arch)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
        return 100.0 * correct / total

    from models import forward_adapt  # late import to avoid cycle
    print(f"[Sanity] ID top-1: {evaluate_id_acc(model):.2f}%")

    # Shared collections
    id_logits, id_softmax = collect_logits_softmax(model, id_val_loader, arch)
    id_tr_logits, _ = collect_logits_softmax(model, id_train_loader, arch)
    ood_logits, ood_softmax = collect_logits_softmax(model, ood_loader, arch)

    feat_id_train, y_train = collect_penultimate_and_labels(model, id_train_loader, arch)
    feat_id_val, _ = collect_penultimate_and_labels(model, id_val_loader, arch)
    feat_ood, _ = collect_penultimate_and_labels(model, ood_loader, arch)

    # Methods selection
    methods = set(m.lower() for m in args.methods)

    if "msp" in methods:
        method_msp(id_softmax, ood_softmax, args.ood)
    if "maxlogit" in methods:
        method_maxlogit(id_logits, ood_logits, args.ood)
    if "energy" in methods:
        method_energy(id_logits, ood_logits, args.ood)
    if "maha" in methods:
        method_mahalanobis(feat_id_train, y_train, feat_id_val, feat_ood, num_classes, args.ood)
    if "gradnorm" in methods:
        W, b = get_classifier_params(model)
        gn_id = gradnorm(feat_id_val.astype(np.float32), W, b, num_classes)
        gn_ood = gradnorm(feat_ood.astype(np.float32), W, b, num_classes)
        auc = Metrics.auc(gn_id, gn_ood); fpr, _ = Metrics.fpr_recall(gn_id, gn_ood)
        print(f"GradNorm: {args.ood} auroc {auc:.2%}, fpr {fpr:.2%}")
    if "nnguide" in methods:
        bank_subset = make_bank_subset(id_train_set, per_class=args.bank_per_class, num_classes=num_classes)
        bank_loader = make_loader(bank_subset, batch=args.batch, shuffle=False, num_workers=args.workers)
        Z_bank, S_bank = collect_feats_and_conf(model, bank_loader, baseconf_msp_torch, arch)
        id_scores = nnguide_score_loader(model, id_val_loader, Z_bank, S_bank, args.k, baseconf_msp_torch, arch, True)
        ood_scores = nnguide_score_loader(model, ood_loader, Z_bank, S_bank, args.k, baseconf_msp_torch, arch, True)
        from sklearn.metrics import roc_auc_score, roc_curve
        y = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
        s = np.concatenate([id_scores, ood_scores])
        auc = roc_auc_score(y, s)
        fpr, tpr, _ = roc_curve(y, s)
        idx = np.argmax(tpr >= 0.95) if np.any(tpr >= 0.95) else np.argmax(tpr)
        print(f"NNGuide (k={args.k}): AUROC {auc:.2%}, FPR@95 {fpr[idx]:.2%}")
    if "neco" in methods:
        model_type = "resnet" if arch.startswith("resnet") else ("deit" if arch == "vit" else arch)
        neco(feat_id_train, feat_id_val, feat_ood, id_logits, ood_logits, model_type, neco_dim=100)
    if "react" in methods:
        from models import get_classifier_params
        W, b = get_classifier_params(model)
        clip = np.quantile(feat_id_train, 0.99)
        react(feat_id_val, feat_ood, clip, W, b, args.ood)
    if "vim" in methods:
        W, b = get_classifier_params(model)
        u = -np.matmul(pinv(W), b)
        model_type = "resnet" if arch.startswith("resnet") else arch
        vim(feat_id_train, feat_id_val, feat_ood, id_tr_logits, id_logits, ood_logits, args.ood, model_type, arch, u)
    if "cosine_layers" in methods:
        cos_layers = _parse_layers_arg(args.cos_layers)
        w_vals = [float(s) for s in args.cos_layer_weights.split(',') if s.strip()]
        if len(w_vals) != len(cos_layers):
            raise ValueError("--cos_layer_weights length must match --cos_layers")
        layer_weights = {li: w_vals[i] for i, li in enumerate(cos_layers)}
        method_cosine_layers(model, arch, id_train_set, id_val_loader, ood_loader,
                             num_classes, cos_layers, layer_weights,
                             make_calib_subset, calib_per_class=args.cos_calib_per_class)

if __name__ == "__main__":
    main()
```
