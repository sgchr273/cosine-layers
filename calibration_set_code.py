
import argparse
import numpy as np
import torch


from loaders import build_transform, load_id_dataset, load_ood_dataset, make_loader, make_calib_loader, make_bank_subset
from models import ModelSpec, build_model, set_all_seeds, DEVICE
import math
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from typing import Dict, List
from models import DEVICE, forward_adapt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RECALL = 0.95  # FPR@95
DEFAULT_METHODS = [
    "msp", "maxlogit", "energy", "maha", "gradnorm",
    "nnguide", "neco", "react", "vim", "cosine_layers"
]

############################
# Metrics helpers
############################
class Metrics:
    @staticmethod
    def auc(id_scores, ood_scores):
        from sklearn import metrics
        y_true = np.concatenate([
            np.zeros_like(id_scores),
            np.ones_like(ood_scores)
        ])
        y_score = np.concatenate([id_scores, ood_scores])
        return metrics.roc_auc_score(y_true, y_score)

    @staticmethod
    def fpr_recall(id_scores, ood_scores, recall_level=0.95):
        from sklearn import metrics
        y_true = np.concatenate([
            np.zeros_like(id_scores),
            np.ones_like(ood_scores)
        ])
        y_score = np.concatenate([id_scores, ood_scores])

        fpr, tpr, thresh = metrics.roc_curve(y_true, y_score, pos_label=1)
        idxs = np.where(tpr >= recall_level)[0]
        if len(idxs) == 0:
            best_i = np.argmax(tpr)
            return fpr[best_i], thresh[best_i]
        i = idxs[0]
        return fpr[i], thresh[i]

############################
# Global average pooling
############################
@torch.no_grad()
def _gap(z: torch.Tensor) -> torch.Tensor:
    # (N,C,H,W) -> (N,C)
    return z.mean(dim=(2,3), keepdim=False)

############################
# Feature iterator:
# returns normalized feats for requested layers
############################
@torch.no_grad()
def _iter_feats_layers(model, loader, arch: str, LAYERS: List[int]):
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        # user-provided forward wrapper
        logits, logits_aux, penult, extra = forward_adapt(model, x, arch)
        # extra is dict: {layer_id: feat_tensor}

        feats = []
        for li in LAYERS:
            z = extra[li]
            if z.dim() == 4:
                z = _gap(z)  # GAP if it's a map
            feats.append(F.normalize(z, dim=1))
        yield feats, y

############################
# Build per-class prototypes
############################
@torch.no_grad()
def cosine_build_prototypes(model, calib_loader, arch: str,
                            LAYERS: List[int], num_classes: int):
    first_feats, first_y = next(_iter_feats_layers(model, calib_loader, arch, LAYERS))
    dims = [f.size(1) for f in first_feats]

    sums: Dict[int, List[torch.Tensor]] = {
        li: [torch.zeros(dims[k], device=DEVICE) for _ in range(num_classes)]
        for k, li in enumerate(LAYERS)
    }
    counts: Dict[int, torch.Tensor] = {
        li: torch.zeros(num_classes, device=DEVICE)
        for li in LAYERS
    }

    def update(feats_list, y):
        for k, li in enumerate(LAYERS):
            Fk = feats_list[k]  # (N, dim)
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
                # fallback if we never saw this class
                p = F.normalize(torch.randn_like(sums[li][0]), dim=0)
            plist.append(p)
        prototypes[li] = torch.stack(plist, dim=0)  # (num_classes, dim)

    return prototypes

############################
# OOD score for a loader
# (1 - weighted cosine-to-nearest-prototype)
############################
@torch.no_grad()
def cosine_score_loader(model, loader, arch: str,
                        prototypes: Dict[int, torch.Tensor],
                        LAYERS: List[int],
                        layer_weights: Dict[int, float]):
    wsum = sum(layer_weights[li] for li in LAYERS)
    all_scores = []

    for feats_list, _ in _iter_feats_layers(model, loader, arch, LAYERS):
        per_layer_max = []
        for k, li in enumerate(LAYERS):
            f = feats_list[k]               # (N, dim)
            P = prototypes[li].to(f.device) # (C, dim)
            cos = f @ P.t()                 # (N, C)
            max_cos, _ = cos.max(dim=1)     # best-matching class
            per_layer_max.append(layer_weights[li] * max_cos)

        avg_max_cos = per_layer_max[0]
        for t in per_layer_max[1:]:
            avg_max_cos = avg_max_cos + t
        avg_max_cos = avg_max_cos / wsum

        ood_score = 1.0 - avg_max_cos      # higher = more OOD-like
        all_scores.append(ood_score.detach().cpu())

    return torch.cat(all_scores, dim=0).numpy()

############################
# ID classification accuracy using prototypes
# combine cosine across layers per class, argmax
############################
@torch.no_grad()
def cosine_id_accuracy(model, loader, arch: str,
                       prototypes: Dict[int, torch.Tensor],
                       LAYERS: List[int],
                       layer_weights: Dict[int, float]):
    correct = 0
    total = 0
    wsum = sum(layer_weights[li] for li in LAYERS)

    for feats_list, y in _iter_feats_layers(model, loader, arch, LAYERS):
        # we'll build class scores of shape (N, num_classes)
        class_scores = None

        for k, li in enumerate(LAYERS):
            f = feats_list[k]                # (N, dim)
            P = prototypes[li].to(f.device)  # (C, dim)
            cos = f @ P.t()                  # (N, C)
            wcos = layer_weights[li] * cos   # weighted
            if class_scores is None:
                class_scores = wcos
            else:
                class_scores = class_scores + wcos

        class_scores = class_scores / wsum   # (N, C)

        preds = class_scores.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()

    acc = 100.0 * correct / max(total, 1)
    return acc

############################
# build calib loader for a given percent/class
############################
def _load_id_trainset(id_name: str, train_tf, data_root: str):
    """
    you must fill this with your dataset logic.
    for example:
    - CIFAR10 train split
    - CIFAR100 train split
    - etc.
    must return a Dataset with .targets or .labels
    """
    import torchvision
    id_name = id_name.lower()
    if id_name == "cifar10":
        ds = torchvision.datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=train_tf,
        )
    elif id_name == "cifar100":
        ds = torchvision.datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=train_tf,
        )
    else:
        raise ValueError(f"Add your dataset '{id_name}' in _load_id_trainset.")
    return ds

def make_calib_loader(
    id_name: str,
    train_tf,
    calib_pct_per_class: float,
    data_root: str,
    batch_size: int = 512,
    num_workers: int = 4,
    seed: int = 0,
):
    full_train = _load_id_trainset(id_name, train_tf, data_root)

    # pull labels
    if hasattr(full_train, "targets"):
        targets = np.array(full_train.targets)
    elif hasattr(full_train, "labels"):
        targets = np.array(full_train.labels)
    else:
        raise RuntimeError("dataset missing .targets/.labels; update make_calib_loader")

    num_classes = int(targets.max() + 1)

    g = torch.Generator()
    g.manual_seed(seed)

    keep_indices = []
    for cls in range(num_classes):
        cls_idx = np.where(targets == cls)[0]
        cls_idx = torch.as_tensor(cls_idx, dtype=torch.long)

        take_n = int(math.ceil(len(cls_idx) * (calib_pct_per_class / 100.0)))
        take_n = min(take_n, len(cls_idx))

        if take_n > 0:
            perm = torch.randperm(len(cls_idx), generator=g)
            chosen = cls_idx[perm[:take_n]]
            keep_indices.append(chosen)

    if len(keep_indices) == 0:
        raise RuntimeError("no samples chosen for calibration; pct too small?")

    keep_indices = torch.cat(keep_indices, dim=0).tolist()
    calib_subset = Subset(full_train, keep_indices)

    calib_loader = DataLoader(
        calib_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return calib_loader, len(calib_subset), num_classes

############################
# core evaluation for ONE calib %
############################
def eval_one_calib_pct(model,
                       arch,
                       id_val_loader,
                       ood_loader,
                       num_classes,
                       cos_layers,
                       layer_weights,
                       calib_loader):
    # prototypes
    print("[cosine_layers] Building prototypes (fast)…")
    protos = cosine_build_prototypes(model, calib_loader, arch, cos_layers, num_classes)

    # score ID+OOD
    print("[cosine_layers] Scoring ID and OOD (fast)…")
    id_scores  = cosine_score_loader(model, id_val_loader, arch, protos, cos_layers, layer_weights)
    ood_scores = cosine_score_loader(model, ood_loader,    arch, protos, cos_layers, layer_weights)

    # accuracy on ID
    print("[cosine_layers] Computing ID accuracy (fast)…")
    acc = cosine_id_accuracy(model, id_val_loader, arch, protos, cos_layers, layer_weights)

    auc  = Metrics.auc(id_scores, ood_scores)
    fpr, _ = Metrics.fpr_recall(id_scores, ood_scores, RECALL)

    return fpr, auc, acc

############################
# evaluate for ALL calib % and pretty print
############################
def evaluate_cosine_layers_over_calib_pcts(
    model,
    arch,
    id_name,
    id_train_tf,
    data_root,
    id_val_loader,
    ood_loader,
    cos_layers,
    layer_weights,
    calib_pcts=(0.5, 1, 5, 10, 25, 50, 100),
):
    results = {}  # pct -> dict(fpr, auc, acc)

    for pct in calib_pcts:
        # build calib loader for this pct
        calib_loader, n_imgs, num_classes = make_calib_loader(
            id_name,
            id_train_tf,
            calib_pct_per_class=pct,
            data_root=data_root,
        )

        print("")
        print(f"=== Calibration {pct}% of train (~{n_imgs} imgs) ===")

        fpr, auc, acc = eval_one_calib_pct(
            model=model,
            arch=arch,
            id_val_loader=id_val_loader,
            ood_loader=ood_loader,
            num_classes=num_classes,
            cos_layers=cos_layers,
            layer_weights=layer_weights,
            calib_loader=calib_loader,
        )

        print(f"cosine_layers (fast): AUROC {auc*100:.2f}%, "
              f"FPR@95 {fpr*100:.2f}%, Acc {acc:.2f}%")

        results[pct] = {
            "fpr": fpr,
            "auroc": auc,
            "acc": acc,
            "n_imgs": n_imgs,
        }

    return results


def parse_args():
    p = argparse.ArgumentParser(description="Flexible OOD Baselines Runner (modular)")
    p.add_argument("--id",  choices=["cifar10", "cifar100", "imagenet"], help="ID dataset",default='cifar10')
    p.add_argument("--ood",  help="OOD name (svhn, isun, lsun, sun, places, inat, textures, cifar10, cifar100) or custom folder with --ood_path",default='svhn')
    p.add_argument("--ood_path", default=None, help="ImageFolder path for custom OOD")
    p.add_argument("--model",  choices=["resnet18", "densenet100", "resnet50", "mobilenet"], default='resnet18')
    p.add_argument("--ckpt", help="Model checkpoint path",default='/home/sgchr/Documents/cosine_layers/runs/ckpts_resnet18_cifar10/best.pth')
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--bank_per_class", type=int, default=50)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--data_root", default="/home/sgchr/Documents/cosine_layers/data")
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)

    # cosine_layers
    p.add_argument("--cos_layers", default="1,2,3", help="Comma list among {1,2,3} mapping to out2,out3,out4")
    p.add_argument("--cos_layer_weights", default="1,1,1", help="Comma list matching cos_layers")
    # p.add_argument("--cos_calib_per_class", type=int, default=5000)
    p.add_argument(
    "--cos_calib_per_class",
    type=float,
    default=50.0,
    help="Percentage of training data PER CLASS to use for cosine calibration. "
         "Examples: 0.5, 1, 5, 10, 25, 50, 100"
    )
    return p.parse_args()


def _parse_layers_arg(arg: str):
    parts = [p.strip() for p in arg.split(',') if p.strip()]
    layers = [int(p) for p in parts]
    # for li in layers:
    #     if li not in (1, 2, 3):
    #         raise ValueError("--cos_layers must be among 1,2,3 (maps to out2,out3,out4)")
    return layers

def main():
    args = parse_args()
    set_all_seeds(0)

    # Input size
    input_size = 224 if args.id == "imagenet" or args.model in ["vit", "swin"] else 32

    # # Datasets & loaders
    id_train_tf = build_transform(args.id, args.model, input_size, train=True)
    id_test_tf = build_transform(args.id, args.model, input_size, train=False)
    ood_tf = build_transform(args.id, args.model, input_size,train=False)

    id_train_set = load_id_dataset(args.id, "train", id_train_tf, data_root=args.data_root)
    id_test_set = load_id_dataset(args.id, "val", id_test_tf, data_root=args.data_root)
    ood_set = load_ood_dataset(args.ood, ood_tf, data_root=args.data_root, path_override=args.ood_path)

    # id_train_loader = make_loader(id_train_set, batch=args.batch, shuffle=False, num_workers=args.workers)
    id_val_loader = make_loader(id_test_set, batch=args.batch, shuffle=False, num_workers=args.workers)
    ood_loader = make_loader(ood_set, batch=args.batch, shuffle=False, num_workers=args.workers)

    # Model
    num_classes = 1000 if args.id == "imagenet" else (100 if args.id == "cifar100" else 10)
    spec = ModelSpec(arch=args.model, num_classes=num_classes, ckpt=args.ckpt)
    model = build_model(spec)

    arch = args.model.lower()

    # # Sanity: ID accuracy
    # @torch.no_grad()
    # def evaluate_id_acc(m):
    #     m.eval()
    #     total = correct = 0
    #     for x, y in id_val_loader:
    #         x = x.to(DEVICE); y = y.to(DEVICE)
    #         logits, _, _, _ = forward_adapt(m, x, arch)
    #         pred = logits.argmax(1)
    #         correct += (pred == y).sum().item()
    #         total += y.numel()
    #     return 100.0 * correct / total

    # from models import forward_adapt  # late import to avoid cycle
    # print(f"[Sanity] ID top-1: {evaluate_id_acc(model):.2f}%")

    # # Shared collections
    # id_logits, id_softmax = collect_logits_softmax(model, id_val_loader, arch)
    # id_tr_logits, _ = collect_logits_softmax(model, id_train_loader, arch)
    # ood_logits, ood_softmax = collect_logits_softmax(model, ood_loader, arch)

    # feat_id_train, y_train = collect_penultimate_and_labels(model, id_train_loader, arch)
    # feat_id_val, _ = collect_penultimate_and_labels(model, id_val_loader, arch)
    # feat_ood, _ = collect_penultimate_and_labels(model, ood_loader, arch)

    # Methods selection
    methods = set(m.lower() for m in args.methods)

    # if "msp" in methods:
    #     method_msp(id_softmax, ood_softmax, args.ood)
    # if "maxlogit" in methods:
    #     method_maxlogit(id_logits, ood_logits, args.ood)
    # if "energy" in methods:
    #     method_energy(id_logits, ood_logits, args.ood)
    # if "maha" in methods:
    #     method_mahalanobis(feat_id_train, y_train, feat_id_val, feat_ood, num_classes, args.ood)
    # if "gradnorm" in methods:
    #     from models import get_classifier_params
    #     W, b = get_classifier_params(model)
    #     gn_id = gradnorm(feat_id_val.astype(np.float32), W, b, num_classes)
    #     gn_ood = gradnorm(feat_ood.astype(np.float32), W, b, num_classes)
    #     auc = Metrics.auc(-gn_id, -gn_ood); fpr, _ = Metrics.fpr_recall(-gn_id, -gn_ood)
    #     print(f"GradNorm: {args.ood} auroc {auc:.2%}, fpr {fpr:.2%}")
    # if "nnguide" in methods:
    #     bank_subset = make_bank_subset(id_train_set, per_class=args.bank_per_class, num_classes=num_classes)
    #     bank_loader = make_loader(bank_subset, batch=args.batch, shuffle=False, num_workers=args.workers)
    #     Z_bank, S_bank = collect_feats_and_conf(model, bank_loader, baseconf_msp_torch, arch)
    #     id_scores = nnguide_score_loader(model, id_val_loader, Z_bank, S_bank, args.k, baseconf_msp_torch, arch, True)
    #     ood_scores = nnguide_score_loader(model, ood_loader, Z_bank, S_bank, args.k, baseconf_msp_torch, arch, True)
    #     from sklearn.metrics import roc_auc_score, roc_curve
    #     y = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    #     s = np.concatenate([id_scores, ood_scores])
    #     auc = roc_auc_score(y, s)
    #     fpr, tpr, _ = roc_curve(y, s)
    #     idx = np.argmax(tpr >= 0.95) if np.any(tpr >= 0.95) else np.argmax(tpr)
    #     print(f"NNGuide (k={args.k}): AUROC {auc:.2%}, FPR@95 {fpr[idx]:.2%}")
    # if "neco" in methods:
    #     model_type = "resnet" if arch.startswith("resnet") else ("deit" if arch == "vit" else arch)
    #     neco(feat_id_train, feat_id_val, feat_ood, id_logits, ood_logits, model_type, neco_dim=100)
    # if "react" in methods:
    #     from models import get_classifier_params
    #     W, b = get_classifier_params(model)
    #     clip = np.quantile(feat_id_train, 0.99)
    #     react(feat_id_val, feat_ood, clip, W, b, args.ood)
    # if "vim" in methods:
    #     W, b = get_classifier_params(model)
    #     u = -np.matmul(pinv(W), b)
    #     model_type = "resnet" if arch.startswith("resnet") else arch
    #     vim(feat_id_train, feat_id_val, feat_ood, id_tr_logits, id_logits, ood_logits, args.ood, model_type, arch, u)
    cos_layers = _parse_layers_arg(args.cos_layers)
    w_vals = [float(s) for s in args.cos_layer_weights.split(',') if s.strip()]
    if len(w_vals) != len(cos_layers):
        raise ValueError("--cos_layer_weights length must match --cos_layers")
    layer_weights = {li: w_vals[i] for i, li in enumerate(cos_layers)}

    all_results = evaluate_cosine_layers_over_calib_pcts(
        model=model,
        arch=arch,
        id_name=args.id,           # e.g. "cifar10"
        id_train_tf=id_train_tf,   # your train transform
        data_root=args.data_root,  # dataset root
        id_val_loader=id_val_loader,
        ood_loader=ood_loader,
        cos_layers=cos_layers,
        layer_weights=layer_weights,
        calib_pcts=(0.5, 1, 5, 10, 25, 50, 100),  # <- you can add 8 here if you want
    )

    # optional: print a summary table at the end
    print("\n==== Summary ====")
    for pct, stats in all_results.items():
        print(f"{pct}% -> AUROC {stats['auroc']*100:.2f}%, "
            f"FPR@95 {stats['fpr']*100:.2f}%, "
            f"Acc {stats['acc']:.2f}%, "
            f"N={stats['n_imgs']}")

       
if __name__ == "__main__":
    main()


# accuracies of networks for cifar10 id:
# resnet18 92.92 -->95.71
# densenet 94.39
# swin 98.3
# vit 98

##accuracy when cifar100=id:
#resnet18  80.27
#accuracy for densenet on cifar100 = 74.69
##resnet50 on imagenet give accuracy of 80.12

