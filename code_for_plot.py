import argparse
import numpy as np
import torch
from methods import cosine_score_loader, cosine_build_prototypes
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib as mpl
import matplotlib.pyplot as plt

# Force Times New Roman everywhere
mpl.rcParams.update({
    "font.family": "DejaVu Serif",     # safe built-in
    "mathtext.fontset": "stix",        # math looks more “Times-like”
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})
# from matplotlib import font_manager as fm, rcParams
# fm.fontManager.addfont("/path/to/Times New Roman.ttf")
# rcParams["font.family"] = "Times New Roman"
# import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "DejaVu Serif", # from earlier
    "axes.labelsize": 14,              # X/Y axis label font size
    "xtick.labelsize": 12,             # X tick labels
    "ytick.labelsize": 12,             # Y tick labels
    "legend.fontsize": 12,             # legend text (if used)
})




from loaders import build_transform, load_id_dataset, load_ood_dataset, make_loader, make_calib_loader, make_bank_subset
from models import ModelSpec, build_model, set_all_seeds, DEVICE
from models import get_classifier_params
from methods import  Metrics

from methods import method_cosine_layers
from numpy.linalg import pinv

DEFAULT_METHODS = [
    "msp", "maxlogit", "energy", "maha", "gradnorm",
    "nnguide", "neco", "react", "vim", "cosine_layers"
]


def parse_args():
    p = argparse.ArgumentParser(description="Flexible OOD Baselines Runner (modular)")
    p.add_argument("--id",  choices=["cifar10", "cifar100", "imagenet"], help="ID dataset",default='imagenet')
    p.add_argument("--ood",  help="OOD name (svhn, isun, lsun, sun, places, inat, textures, cifar10, cifar100) or custom folder with --ood_path",default='svhn')
    p.add_argument("--ood_path", default=None, help="ImageFolder path for custom OOD")
    p.add_argument("--model",  choices=["resnet18", "densenet100", "resnet50", "mobilenet"], default='resnet50')
    p.add_argument("--ckpt", help="Model checkpoint path",default='/home/sgchr/Documents/cosine_layers/runs/ckpts_resnet18_cifar10/best.pth')
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--bank_per_class", type=int, default=50)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--data_root", default="/home/sgchr/Documents/cosine_layers/data")
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)

    # cosine_layers
    p.add_argument("--cos_layers", default="1,2,3", help="Comma list among {1,2,3} mapping to out2,out3,out4")
    p.add_argument("--cos_layer_weights", default="1,1,1", help="Comma list matching cos_layers")
    p.add_argument("--cos_calib_per_class", type=int, default=5000)
    return p.parse_args()


def _parse_layers_arg(arg: str):
    parts = [p.strip() for p in arg.split(',') if p.strip()]
    layers = [int(p) for p in parts]
    for li in layers:
        if li not in (1, 2, 3):
            raise ValueError("--cos_layers must be among 1,2,3 (maps to out2,out3,out4)")
    return layers

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

def _to_layer_weight_dict(LAYERS: List[int], w: Tuple[float, ...]) -> Dict[int, float]:
    assert len(LAYERS) == len(w), "LAYERS and weight tuple must have same length"
    return {li: float(w_i) for li, w_i in zip(LAYERS, w)}

# def sweep_weights_with_metrics(
#     model, arch: str,
#     id_val_loader, ood_loader,
#     prototypes: Dict[int, torch.Tensor],
#     LAYERS: List[int],
#     weight_presets: Dict[str, Tuple[float, ...]],
#     RECALL: float,
# ):
#     """
#     Uses RAW weights (no extra normalization here).
#     Note: cosine_score_loader already divides by wsum internally.
#     """
#     results, order = {}, []
#     for name, wtuple in weight_presets.items():
#         layer_weights = _to_layer_weight_dict(LAYERS, wtuple)  # NO normalization

#         id_scores  = cosine_score_loader(model, id_val_loader,  arch, prototypes, LAYERS, layer_weights)
#         ood_scores = cosine_score_loader(model, ood_loader,     arch, prototypes, LAYERS, layer_weights)

#         auc = Metrics.auc(id_scores, ood_scores)
#         fpr, _ = Metrics.fpr_recall(id_scores, ood_scores, RECALL)

#         results[name] = {
#             "layer_weights_raw": wtuple,
#             "layer_weights_used": layer_weights,  # raw dict
#             "id_scores": id_scores,
#             "ood_scores": ood_scores,
#             "auroc": float(auc),
#             "fpr": float(fpr),
#         }
#         order.append(name)

#     return results, order

def sweep_weights_with_metrics_multi(
    model,
    arch: str,
    id_val_loader,
    ood_loaders: Dict[str, object],  # name -> DataLoader
    prototypes: Dict[int, torch.Tensor],
    LAYERS: List[int],
    weight_presets: Dict[str, Tuple[float, ...]],
    RECALL: float,
):
    results, order = {}, []
    ds_names = list(ood_loaders.keys())

    for preset_name, wtuple in weight_presets.items():
        layer_weights = _to_layer_weight_dict(LAYERS, wtuple)  # NO normalization
        preset_bucket = {
            "layer_weights_raw": wtuple,
            "layer_weights_used": layer_weights,
            "datasets": {}
        }

        # compute ID scores once per preset
        id_scores = cosine_score_loader(model, id_val_loader, arch, prototypes, LAYERS, layer_weights)

        for ds_name, ood_loader in ood_loaders.items():
            ood_scores = cosine_score_loader(model, ood_loader, arch, prototypes, LAYERS, layer_weights)
            auroc = Metrics.auc(id_scores, ood_scores)
            fpr, _ = Metrics.fpr_recall(id_scores, ood_scores, RECALL)
            preset_bucket["datasets"][ds_name] = {
                "id_scores": id_scores,
                "ood_scores": ood_scores,
                "auroc": float(auroc),
                "fpr": float(fpr),
            }

        results[preset_name] = preset_bucket
        order.append(preset_name)

    return results, order, ds_names

def plot_auroc_and_fpr_multi(results, order, ds_names, arch: str, recall_value: float):
    x = np.arange(len(order))

    # AUROC plot (one line per dataset)
    plt.figure()
    for ds_name in ds_names:
        y = [results[p]["datasets"][ds_name]["auroc"] for p in order]
        plt.plot(x, y, marker="o", label=ds_name)
    plt.xticks(x, order, rotation=20)
    plt.xlabel("Weight preset")
    plt.ylabel("AUROC")
    # plt.title(f"{arch}: AUROC vs layer_weights")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('auroc_weights_res50')

    # FPR@Recall plot (one line per dataset)
    plt.figure()
    for ds_name in ds_names:
        y = [results[p]["datasets"][ds_name]["fpr"] for p in order]
        plt.plot(x, y, marker="o", label=ds_name)
    plt.xticks(x, order, rotation=20)
    plt.xlabel("Weight preset")
    plt.ylabel(f"FPR")
    # plt.title(f"{arch}: FPR@Recall vs layer_weights")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('fpr_weights_res50')

def print_weight_table_multi(results, order, ds_names, decimals=4):
    fmt = f"{{:.{decimals}f}}"
    header = ["Preset"] + [f"AUROC({d})" for d in ds_names] + [f"FPR({d})" for d in ds_names] + ["Weights (raw)"]
    print("\t".join(header))
    for p in order:
        wraw = results[p]["layer_weights_raw"]
        wtxt = ", ".join([f"{i+1}:{fmt.format(wraw[i])}" for i in range(len(wraw))])
        aurocs = [fmt.format(results[p]["datasets"][d]["auroc"]) for d in ds_names]
        fprs   = [fmt.format(results[p]["datasets"][d]["fpr"])   for d in ds_names]
        print("\t".join([p] + aurocs + fprs + [wtxt]))

import numpy as np

def print_avg_fpr_auroc(results, order, ds_names=None, decimals=4):
    """
    Prints: Preset  AvgAUROC±Std  AvgFPR±Std
    Works with:
      - multi-OOD `results[preset]["datasets"][ds]["auroc"/"fpr"]`
      - single-OOD  `results[preset]["auroc"/"fpr"]`
    """
    fmt = f"{{:.{decimals}f}}"
    print("Preset\tAvg AUROC ± Std\tAvg FPR ± Std")

    for p in order:
        # Detect multi-OOD structure
        if "datasets" in results[p]:
            # If ds_names not provided, infer from keys
            ds_keys = ds_names if ds_names is not None else list(results[p]["datasets"].keys())
            aurocs = np.array([results[p]["datasets"][d]["auroc"] for d in ds_keys], dtype=float)
            fprs   = np.array([results[p]["datasets"][d]["fpr"]   for d in ds_keys], dtype=float)
        else:
            # Single OOD dataset case
            aurocs = np.array([results[p]["auroc"]], dtype=float)
            fprs   = np.array([results[p]["fpr"]],   dtype=float)

        auroc_mean, auroc_std = float(aurocs.mean()), float(aurocs.std(ddof=0))
        fpr_mean,   fpr_std   = float(fprs.mean()),   float(fprs.std(ddof=0))

        print(
            f"{p}\t{fmt.format(auroc_mean)} ± {fmt.format(auroc_std)}\t"
            f"{fmt.format(fpr_mean)} ± {fmt.format(fpr_std)}"
        )


def main():
    args = parse_args()
    set_all_seeds(0)

    # Input size
    input_size = 224 if args.id == "imagenet" or args.model in ["vit", "swin"] else 32

    # Datasets & loaders
    # id_train_tf = build_transform(args.id, args.model, input_size, train=True)
    id_test_tf = build_transform(args.id, args.model, input_size, train=False)
    # ood_tf = build_transform(args.id, args.model, input_size,train=False)

    # id_train_set = load_id_dataset(args.id, "train", id_train_tf, data_root=args.data_root)
    id_test_set = load_id_dataset(args.id, "val", id_test_tf, data_root=args.data_root)
    # ood_set = load_ood_dataset(args.ood, ood_tf, data_root=args.data_root, path_override=args.ood_path)

    # id_train_loader = make_loader(id_train_set, batch=args.batch, shuffle=False, num_workers=args.workers)
    id_val_loader = make_loader(id_test_set, batch=args.batch, shuffle=False, num_workers=args.workers)
    # ood_loader = make_loader(ood_set, batch=args.batch, shuffle=False, num_workers=args.workers)

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
        calib_loader = make_calib_loader(args.id, id_test_tf, args.cos_calib_per_class, args.data_root)
        # method_cosine_layers(model, arch, id_val_loader, ood_loader,
        #                      num_classes, cos_layers, layer_weights,
        #                      calib_loader)
        #Following i sfor cifar10 resnet18
        # CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
        # CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
        #Following is for cifar100,
        # CIFAR10_MEAN = (0.5071, 0.4865, 0.4409)
        # CIFAR10_STD  = (0.2673, 0.2564, 0.2761)
        #Following is for densenet100, id=cif10
        # CIFAR10_MEAN=(0.4914, 0.4822, 0.4465)
        # CIFAR10_STD=(0.2470, 0.2435, 0.2616)
        #Folloing is for imagenet
        # CIFAR10_MEAN =(0.485, 0.456, 0.406)
        # CIFAR10_STD=(0.229, 0.224, 0.225)
        # tf = transforms.Compose([
        #     transforms.Resize((32, 32)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        # ])

        svhn_set = torchvision.datasets.SVHN('./data', split='test', download=True, transform=id_test_tf)
        isun_set = torchvision.datasets.ImageFolder('/home/sgchr/Documents/Multiple_spaces/CIFAR10/ood_data/iSUN', transform=id_test_tf)  
        lsun_set = torchvision.datasets.ImageFolder('/home/sgchr/Documents/Multiple_spaces/CIFAR10/ood_data/LSUN', transform=id_test_tf) 
        pla_set = torchvision.datasets.ImageFolder('/home/sgchr/Documents/Multiple_spaces/Imagenet/ood_data/Places', transform=id_test_tf)  
        inat_set = torchvision.datasets.ImageFolder('/home/sgchr/Documents/Multiple_spaces/Imagenet/ood_data/iNaturalist', transform=id_test_tf)  
        dtd_set = torchvision.datasets.ImageFolder('/home/sgchr/Documents/Multiple_spaces/Imagenet/ood_data/Textures', transform=id_test_tf)  
        sun_set = torchvision.datasets.ImageFolder('/home/sgchr/Documents/Multiple_spaces/Imagenet/ood_data/SUN', transform=id_test_tf)  

        svhn_loader = DataLoader(svhn_set, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
        isun_loader = DataLoader(isun_set, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
        lsun_loader = DataLoader(lsun_set, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
        pla_loader = DataLoader(pla_set, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
        inat_loader = DataLoader(inat_set, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
        dtd_loader = DataLoader(dtd_set, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
        sun_loader = DataLoader(sun_set, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
        
        prototypes = cosine_build_prototypes(model, calib_loader, arch, cos_layers, num_classes)
        weight_presets = {
            "uniform":       (0.33, 0.33, 0.34),
            "deep_heavy":    (0.20, 0.30, 0.50),  # strong default for ResNet50 & DenseNet
            "mid_heavy":     (0.25, 0.50, 0.25),
            "shallow_heavy": (0.50, 0.30, 0.20),
        }
        # weight_presets = {
        #     "uniform":       (1, 1, 1),
        #     "deep_heavy":    (.5, .5, 3),  # strong default for ResNet50 & DenseNet
        #     "mid_heavy":     (.5, 3, .5),
        #     "shallow_heavy": (3, 0.3, 0.2),
        # }

        ood_loaders = {
            "iNaturalist": inat_loader,
            "Places":      pla_loader,
            "Textures":    dtd_loader,
            # "SVHN": svhn_loader,    
            # "iSUN": isun_loader,
            # "LSUN":lsun_loader
            # add more…
            "SUN": sun_loader
        }

        results, order, ds_names = sweep_weights_with_metrics_multi(
            model=model,
            arch=arch,
            id_val_loader=id_val_loader,
            ood_loaders=ood_loaders,
            prototypes=prototypes,
            LAYERS=cos_layers,
            weight_presets=weight_presets,
            RECALL=0.95,
        )

        print_weight_table_multi(results, order, ds_names, decimals=4)
        plot_auroc_and_fpr_multi(results, order, ds_names, arch=arch, recall_value=0.95)
        # results, order, ds_names = sweep_weights_with_metrics_multi(...)
        print_avg_fpr_auroc(results, order, ds_names)


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