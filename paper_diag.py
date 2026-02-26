# plot_id_ood_prototypes.py
# Usage:
#   python plot_id_ood_prototypes.py --assets ./diag_assets --view vis \
#       --png out_scatter.png --pdf out_scatter.pdf --max_id 20000 --max_ood 8000
#
# Inputs expected in --assets:
#   Z_id_vis.npy, Z_ood_vis.npy, Z_P_vis.npy, id_labels.npy      (for --view vis)
#   or
#   Z_id_concat.npy, Z_ood_concat.npy, Z_P_concat.npy, id_labels.npy  (for --view concat)

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_arrays(assets_dir: str, view: str):
    if view == "vis":
        Z_id  = np.load(os.path.join(assets_dir, "Z_id_vis.npy"))
        Z_ood = np.load(os.path.join(assets_dir, "Z_ood_vis.npy"))
        Z_P   = np.load(os.path.join(assets_dir, "Z_P_vis.npy"))
    elif view == "concat":
        Z_id  = np.load(os.path.join(assets_dir, "Z_id_concat.npy"))
        Z_ood = np.load(os.path.join(assets_dir, "Z_ood_concat.npy"))
        Z_P   = np.load(os.path.join(assets_dir, "Z_P_concat.npy"))
    else:
        raise ValueError("--view must be 'vis' or 'concat'")
    y_id = np.load(os.path.join(assets_dir, "id_labels.npy"))
    return Z_id, Z_ood, Z_P, y_id

def subsample(X: np.ndarray, y: np.ndarray | None, max_n: int, seed: int = 0):
    if max_n is None or len(X) <= max_n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_n, replace=False)
    if y is None:
        return X[idx], None
    return X[idx], y[idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=str,  help="Directory with saved Z_*.npy and id_labels.npy", default="/home/sgchr/Documents/cosine_layers/diag_assets")
    ap.add_argument("--view", type=str, default="vis", choices=["vis", "concat"], help="Use PCA of vis layer or concatenated features")
    ap.add_argument("--png", type=str, default="id_ood_prototype_pca_1.png")
    ap.add_argument("--pdf", type=str, default="id_ood_prototype_pca.pdf")
    ap.add_argument("--title", type=str, default="ID clusters & class prototypes vs OOD (PCA of normalized GAP features)")
    ap.add_argument("--max_id", type=int, default=8000, help="Max ID points to plot (subsample for clarity)")
    ap.add_argument("--max_ood", type=int, default=5000, help="Max OOD points to plot (subsample for clarity)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--id_marker_size", type=float, default=6.0)
    ap.add_argument("--ood_marker_size", type=float, default=7.5)
    ap.add_argument("--proto_marker_size", type=float, default=160.0)
    args = ap.parse_args()

    # Load arrays
    Z_id, Z_ood, Z_P, y_id = load_arrays(args.assets, args.view)

    # Subsample for a clean figure
    Z_id,  y_id  = subsample(Z_id,  y_id,  args.max_id,  seed=args.seed)
    Z_ood, _     = subsample(Z_ood, None,  args.max_ood, seed=args.seed)

    # Plot
    fig, ax = plt.subplots(figsize=(8.8, 7.2))

    # ID: colored by class labels (use tab10)
    if y_id is None:
        # no labels—just plot as light dots
        ax.scatter(Z_id[:,0], Z_id[:,1], s=args.id_marker_size, c="0.55", alpha=0.7, linewidths=0, label="ID")
    else:
        num_classes = int(np.max(y_id)) + 1
        cmap = plt.cm.get_cmap("tab10", max(10, num_classes))
        # Draw per class to get consistent colors; keep legend minimal later
        for c in range(num_classes):
            sel = (y_id == c)
            if np.any(sel):
                ax.scatter(
                    Z_id[sel, 0], Z_id[sel, 1],
                    s=args.id_marker_size, c=[cmap(c)], alpha=0.75, linewidths=0
                )

    # OOD: gray triangles
    ax.scatter(
        Z_ood[:,0], Z_ood[:,1],
        s=args.ood_marker_size, c="0.45", alpha=0.65, marker="^", linewidths=0, label="OOD"
    )

    # Prototypes: stars with black edges
    ax.scatter(
        Z_P[:,0], Z_P[:,1],
        s=args.proto_marker_size, marker="*", c="gold",
        edgecolors="k", linewidths=0.8, zorder=5, label="Prototype"
    )

    # Cosmetics
    ax.set_title(args.title, fontsize=12, weight="bold", pad=10)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.spines[:].set_visible(False)

    # Minimal legend (proxy artists)
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="k", markersize=6, label="ID"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor="0.45", markersize=7, label="OOD"),
        Line2D([0],[0], marker="*", color="w", markeredgecolor="k", markerfacecolor="gold", markersize=12, label="Prototype"),
    ]
    ax.legend(handles=legend_elems, loc="best", frameon=False)

    plt.tight_layout()
    # Save PNG (300 dpi) and PDF
    plt.savefig(args.png, dpi=300, bbox_inches="tight")
    plt.savefig(args.pdf, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
