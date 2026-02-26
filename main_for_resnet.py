import argparse
import numpy as np
import torch, torchvision
from torchvision.models import (
    ResNet50_Weights,
    MobileNet_V3_Large_Weights
)

from loaders import build_transform, load_id_dataset, load_ood_dataset, make_loader, make_calib_loader, make_bank_subset
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
#change the batchsize in args to 512, remove args.spec from the ModelSpec,id_test_set for calib_loader


def parse_args():
    p = argparse.ArgumentParser(description="Flexible OOD Baselines Runner (modular)")
    p.add_argument("--id",  choices=["cifar10", "cifar100", "imagenet"], help="ID dataset",default='imagenet')
    p.add_argument("--ood",  help="OOD name (svhn, isun, lsun, sun, places, inat, textures, cifar10, cifar100) or custom folder with --ood_path",default='inat')
    p.add_argument("--ood_path", default=None, help="ImageFolder path for custom OOD")
    p.add_argument("--model",  choices=["resnet18", "densenet100", "resnet50", "swin"], default='resnet50')
    p.add_argument("--ckpt", help="Model checkpoint path",default='/home/sgchr/Documents/cosine_layers/runs/ckpts_resnet18_cifar10/best.pth')
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--bank_per_class", type=int, default=50)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--data_root", default="/home/sgchr/Documents/cosine_layers/data")
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)

    # cosine_layers
    p.add_argument("--cos_layers", default="5", help="Comma list among {1,2,3} mapping to out2,out3,out4")
    p.add_argument("--cos_layer_weights", default="1", help="Comma list matching cos_layers")
    p.add_argument("--cos_calib_per_class", type=int, default=5000)
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

    # Datasets & loaders
    # id_train_tf = build_transform(args.id, args.model, input_size, train=True)
    id_test_tf = build_transform(args.id, args.model, input_size, train=False)
    ood_tf = build_transform(args.id, args.model, input_size,train=False)

    # id_train_set = load_id_dataset(args.id, "train", id_train_tf, data_root=args.data_root)
    id_test_set = load_id_dataset(args.id, "val", id_test_tf, data_root=args.data_root)
    ood_set = load_ood_dataset(args.ood, ood_tf, data_root=args.data_root, path_override=args.ood_path)

    # id_train_loader = make_loader(id_train_set, batch=args.batch, shuffle=False, num_workers=args.workers)
    id_val_loader = make_loader(id_test_set, batch=args.batch, shuffle=False, num_workers=args.workers)
    ood_loader = make_loader(ood_set, batch=args.batch, shuffle=False, num_workers=args.workers)

    # Model
    num_classes = 1000 if args.id == "imagenet" else (100 if args.id == "cifar100" else 10)
    spec = ModelSpec(arch=args.model, num_classes=num_classes)  #ckpt=args.ckpt
    # model = build_model(spec)

    # arch = args.model.lower()
    arch = 'resnet50'
    model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model = model.to(DEVICE)
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
    if "cosine_layers" in methods:
        cos_layers = _parse_layers_arg(args.cos_layers)
        w_vals = [float(s) for s in args.cos_layer_weights.split(',') if s.strip()]
        if len(w_vals) != len(cos_layers):
            raise ValueError("--cos_layer_weights length must match --cos_layers")
        layer_weights = {li: w_vals[i] for i, li in enumerate(cos_layers)}
        # change the folllowing to id_train_tf for the rest of the datasets except imagenet
        calib_loader = make_calib_loader(args.id, id_test_tf, args.cos_calib_per_class, args.data_root)
        method_cosine_layers(model, arch, id_val_loader, ood_loader,
                             num_classes, cos_layers, layer_weights,
                             calib_loader)

if __name__ == "__main__":
    main()

# accuracies of networks for cifar10 id:
# resnet18 92.92 -->95.71
# densenet 94.39
# resnet50 80.12
# vit 98

# ##############reuslts  for resnet50  when layer_weight=(1,1,1) and calib_loader=5k
# ##inat     
# [Sanity] ID top-1: 80.12%
# [cosine_layers] Building prototypes...
# [cosine_layers] Scoring ID and OOD...
# cosine_layers: AUROC 93.95%, FPR@95 24.75%
# ##places
# [Sanity] ID top-1: 80.12%
# [cosine_layers] Building prototypes...
# [cosine_layers] Scoring ID and OOD...
# cosine_layers: AUROC 78.24%, FPR@95 62.56%
# ##Textures
# [Sanity] ID top-1: 80.12%
# [cosine_layers] Building prototypes...
# [cosine_layers] Scoring ID and OOD...
# cosine_layers: AUROC 96.90%, FPR@95 16.53%
# ##Sun
# [Sanity] ID top-1: 80.12%
# [cosine_layers] Building prototypes...
# [cosine_layers] Scoring ID and OOD...
# cosine_layers: AUROC 81.24%, FPR@95 59.30%
# ##############reuslts  for resnet50  when layer_weight=(1,1,1) and calib_loader=5k
##Sun
# [cosine_layers] Building prototypes...
# [cosine_layers] Scoring ID and OOD...
# cosine_layers: AUROC 76.22%, FPR@95 62.01%