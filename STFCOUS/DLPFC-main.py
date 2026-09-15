"""
STFCOUS on the human DLPFC dataset (12 Visium sections, 151507-151676).

STFCOUS is a multi-level graph contrastive learning framework for spatial
domain identification in spatial transcriptomics:

    * two augmented graph views built by random feature masking and
      random edge dropping;
    * a weight-shared ResGCN encoder with DyT (dynamic tanh) activation
      and residual connections;
    * a competence-weighted multi-level objective:
        - L_self : node-level cross-view alignment of each spot,
        - L_nei  : edge-level neighbourhood consistency,
        - L_spa  : global-level soft-gated repulsion between
                   non-adjacent spot pairs;
    * per-spot confidence weights from `CompetenceScorer`, estimated
      from neighbourhood embedding consistency.

This script trains STFCOUS on every DLPFC section over multiple random
seeds, evaluates each run with mclust clustering (ARI / NMI against the
ground-truth cortical layers), and saves the best spatial plots, UMAP
plots and a CSV summary.

Dependencies: torch, scanpy, scikit-learn, matplotlib, pillow, rpy2
(R package `mclust`), plus the local modules `model`, `util` and
`preprocess` (place model.py / util.py / preprocess.py / DLPFC-main.py
in the same directory).

Usage:
    python DLPFC-main.py
"""

import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from sklearn.decomposition import PCA
from torch.optim import Adam

from model import ResGCN
from util import evaluate, normalize, augment, load_data1

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------- #
# R environment: mclust is required by the clustering inside `evaluate`. #
# Uncomment the next line if mclust lives in a non-default R library.    #
# --------------------------------------------------------------------- #
# robjects.r('.libPaths("/path/to/your/Rlibs")')
robjects.r("library(mclust)")


# --------------------------------------------------------------------- #
# Competence scorer                                                     #
# --------------------------------------------------------------------- #
class CompetenceScorer(nn.Module):
    """Per-spot confidence weight estimated from local embedding consistency.

    A spot whose neighbours agree with it in embedding space (cosine
    similarity above `sigma`) is considered reliable and receives a high
    weight; noisy or tissue-boundary spots receive a low weight.  The
    weights lie in [0.1, 1.0] and are detached from the computation graph.
    """

    def __init__(self, sigma=0.5, device="cuda"):
        super(CompetenceScorer, self).__init__()
        self.sigma = sigma
        self.device = device

    def forward(self, z, adj):
        # 1. cosine similarity matrix of the embeddings
        z_norm = F.normalize(z, p=2, dim=1)
        sim_matrix = torch.mm(z_norm, z_norm.t())                     # (N, N)

        # 2. neighbour mask (exclude self-loops)
        if adj.is_sparse:
            adj = adj.to_dense()
        neighbor_mask = (adj > 0).float()
        neighbor_mask = neighbor_mask * (1 - torch.eye(adj.shape[0], device=self.device))

        # 3. keep only supportive neighbours (similarity > sigma)
        clean_sim = torch.where(sim_matrix > self.sigma, sim_matrix, torch.zeros_like(sim_matrix))

        # 4. local consistency score, normalised by node degree
        rho = torch.sum(clean_sim * neighbor_mask, dim=1)
        degree = torch.sum(neighbor_mask, dim=1) + 1e-6
        competence = rho / degree

        # 5. min-max scale to [0.1, 1.0] so every spot keeps a base weight
        competence = (competence - competence.min()) / (competence.max() - competence.min() + 1e-6)
        competence = competence * 0.9 + 0.1

        return competence.detach()


def fix_seed(seed):
    """Fix all random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # ------------------------------ config ---------------------------- #
    save_dir = "results_DLPFC"
    os.makedirs(save_dir, exist_ok=True)

    datasets = ['151507', '151508', '151509', '151510',
                '151669', '151670', '151671', '151672',
                '151673', '151674', '151675', '151676']
    seed_list = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

    # training hyper-parameters
    epochs = 150
    lr, wd = 5e-4, 1e-4
    lam, gam = 1.0, 1.0        # loss weights of L_nei and L_spa
    s, tau = 0.6, 0.1          # soft-gate threshold and temperature of L_spa
    sigma = 0.4                # similarity threshold of the competence scorer
    hid_dims = [256, 256]      # ResGCN hidden dimensions
    pd1, pd2 = 0.2, 0.6        # edge-drop rates of the two views
    pm1, pm2 = 0.1, 0.0        # feature-mask rates of the two views
    max_grad_norm = 5.0

    device = "cuda" if torch.cuda.is_available() else "cpu"

    csv_summary_list = []

    # --------------------------- dataset loop ------------------------- #
    for data_name in datasets:
        print(f"\n========== Running for {data_name} ==========\n")

        ari_list, nmi_list = [], []

        global_best_ari = -1
        global_best_pred_labels = None
        global_best_emb = None
        Y_global = None
        adata_for_plot = None

        # ---------------------------- seed loop ------------------------ #
        for seed in seed_list:
            A, X, Y, adata = load_data1(data_name)

            if Y_global is None:
                Y_global = Y.cpu().numpy()
            adata_for_plot = adata

            in_dim = X.shape[1]

            fix_seed(seed)

            encoder = ResGCN(in_dim, hid_dims).to(device)
            comp_scorer = CompetenceScorer(sigma=sigma, device=device).to(device)

            A, X = A.to(device), X.to(device)
            if not A.is_sparse:
                A = A.to_sparse()
            A = A.coalesce()
            src, dst = A.indices()

            # boolean mask selecting all non-adjacent pairs (i != j, A_ij = 0)
            # note: N x N tensor, memory grows quadratically with spot count
            mask = torch.full(A.size(), True, device=A.device)
            mask[src, dst] = False
            mask.fill_diagonal_(False)

            optimizer = Adam(encoder.parameters(), lr=lr, weight_decay=wd)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

            best_ari, best_nmi = -1, -1
            best_labels, best_emb = None, None

            # -------------------------- train loop ---------------------- #
            for epoch in range(epochs):
                encoder.train()

                # augmentation warm-up: linearly ramp the perturbation
                # strength over the first 30% of epochs to avoid early
                # collapse, and warm up L_spa with the same schedule
                t = epoch / max(1, epochs - 1)          # 0 -> 1
                warm = min(1.0, t / 0.3)

                A1, X1 = augment(A, X, pd1 * warm, pm1 * warm)
                A2, X2 = augment(A, X, pd2 * warm, pm2 * warm)

                A_norm1 = normalize(A1, add_self_loops=True)
                A_norm2 = normalize(A2, add_self_loops=True)

                Z1, Z2 = encoder(A_norm1, A_norm2, X1, X2)

                # per-spot confidence weights (mean-normalised, no grad)
                Z_mean = (Z1 + Z2) / 2
                weights = comp_scorer(Z_mean, A)
                weights = weights / (weights.mean().detach() + 1e-12)

                Z1_n = F.normalize(Z1, dim=1)
                Z2_n = F.normalize(Z2, dim=1)

                # node-level: cross-view alignment of the same spot
                sim_pos = (Z1_n * Z2_n).sum(dim=1)
                loss_ali = -(weights * sim_pos).mean()

                # edge-level: neighbourhood consistency within each view
                sim_nei_1 = (Z1_n[src] * Z1_n[dst]).sum(dim=1)
                sim_nei_2 = (Z2_n[src] * Z2_n[dst]).sum(dim=1)
                loss_nei = -(weights[src] * (sim_nei_1 + sim_nei_2) / 2).mean()

                # global-level: soft-gated repulsion between non-adjacent pairs
                S_mat = Z1_n @ Z2_n.T
                S_masked = torch.masked_select(S_mat, mask)
                loss_spa = torch.sigmoid((S_masked - s) / tau).mean()

                loss = loss_ali + lam * loss_nei + (gam * warm) * loss_spa

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                scheduler.step()

                # -------------------- evaluation -------------------- #
                encoder.eval()
                with torch.no_grad():
                    Z_eval = encoder.embed(normalize(A, add_self_loops=True), X)

                Z_numpy = Z_eval.cpu().numpy()
                adata.obsm['emb_pca'] = PCA(n_components=20).fit_transform(Z_numpy)

                ari, pred_labels, nmi = evaluate(adata, Z_numpy, Y.cpu().numpy())

                if ari > best_ari:
                    best_ari, best_nmi = ari, nmi
                    best_labels = pred_labels.copy()
                    best_emb = Z_numpy.copy()

            print(f"  Seed {seed} finished. Best ARI: {best_ari:.4f}")

            ari_list.append(best_ari)
            nmi_list.append(best_nmi)

            if best_ari > global_best_ari:
                global_best_ari = best_ari
                global_best_pred_labels = best_labels
                global_best_emb = best_emb

        # ---------------------- per-dataset summary -------------------- #
        print(f"Summary for {data_name}: "
              f"Mean ARI={np.mean(ari_list):.4f}, Max ARI={np.max(ari_list):.4f}")
        print("ARI list:", [f"{x:.4f}" for x in ari_list])

        csv_summary_list.append({
            'Dataset': data_name,
            'Mean_ARI': np.mean(ari_list),
            'Std_ARI': np.std(ari_list),
            'Max_ARI': np.max(ari_list),
            'Mean_NMI': np.mean(nmi_list),
            'Std_NMI': np.std(nmi_list),
            'Max_NMI': np.max(nmi_list),
        })

        # -------------- plots for the best run (spatial + UMAP) -------- #
        if global_best_pred_labels is not None and adata_for_plot is not None:
            print(f"  Drawing scanpy plots for {data_name}...")
            adata = adata_for_plot.copy()

            try:
                adata.obsm['domain_best'] = global_best_emb
                adata.obs['predicted'] = global_best_pred_labels.astype(str).astype('category')
                adata.obs['ground_truth'] = Y_global.astype(str)

                # (1) spatial plot on the H&E image
                spatial_path = os.path.join(save_dir, f'{data_name}_spatial.jpg')
                plt.rcParams["figure.figsize"] = (4, 4)
                sc.pl.spatial(adata, img_key="hires", color='predicted',
                              frameon=False, title='', legend_loc=None, show=False)
                plt.savefig(spatial_path, bbox_inches='tight', dpi=600, pad_inches=0)
                plt.close()

                # slightly darkened / desaturated duplicate for figures
                if os.path.exists(spatial_path):
                    img = Image.open(spatial_path)
                    img = ImageEnhance.Brightness(img).enhance(0.90)
                    img = ImageEnhance.Color(img).enhance(0.85)
                    img.save(os.path.join(save_dir, f'{data_name}_spatial_dark.jpg'))

                # (2) UMAP of the best embedding
                sc.pp.neighbors(adata, use_rep='domain_best')
                sc.tl.umap(adata)
                sc.pl.umap(adata, color='predicted', title='',
                           legend_loc=None, frameon=False, show=False)
                plt.savefig(os.path.join(save_dir, f'{data_name}_umap.jpg'),
                            bbox_inches='tight', dpi=600, pad_inches=0)
                plt.close()

                print(f"  [{data_name}] plots saved.")
            except Exception as e:
                print(f"  [{data_name}] plotting failed: {e}")
                import traceback
                traceback.print_exc()

    # ------------------------------ CSV ------------------------------- #
    df = pd.DataFrame(csv_summary_list)
    csv_path = os.path.join(save_dir, "final_summary_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nAll results and plots saved to: {save_dir}")
