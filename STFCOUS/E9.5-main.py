"""
STFCOUS on the MOSTA Mouse Brain E9.5 dataset (spatial transcriptomics).

Trains the DyT-based residual graph encoder with competence-weighted
multi-level graph contrastive learning on an embryonic mouse brain section.
No ground-truth annotation is used: for every random seed the epoch with
the highest Calinski-Harabasz (CH) score is kept, and internal clustering
metrics (CH / DB / SC) are reported.

Companion modules (place in the same directory):
    model.py : DyT-based residual GCN encoder (ResGCN)
    util.py  : data loading, graph augmentation, evaluation

Outputs are saved under ./results_E9.5/ :
    {section}_Seed{seed}_predicted_Normal.jpg : spatial domain map
    {section}_Seed{seed}_predicted_Clean.jpg  : map without axes and legend
    {section}_Seed{seed}_predicted.h5ad       : embedding and predicted labels
    training_metrics.csv / Final_Summary_Posterior.csv : metric records
"""

import os
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn import metrics
from sklearn.decomposition import PCA

from model import ResGCN
from util import evaluate, normalize, augment, load_data1

warnings.filterwarnings('ignore')


def fix_seed(seed):
    """Fix all random seeds and force deterministic cuDNN behaviour."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CompetenceScorer(nn.Module):
    """Per-spot confidence weight from neighbourhood embedding consistency."""

    def __init__(self, sigma=0.5, device='cuda:0'):
        super(CompetenceScorer, self).__init__()
        self.sigma = sigma
        self.device = device

    def forward(self, z, adj):
        # 1. cosine similarity between all spot pairs
        z_norm = F.normalize(z, p=2, dim=1)
        sim_matrix = torch.mm(z_norm, z_norm.t())

        # 2. densify the adjacency matrix if sparse
        if adj.is_sparse:
            adj = adj.to_dense()

        # 3. mask of spatial neighbours (self-connections removed)
        neighbor_mask = (adj > 0).float()
        neighbor_mask = neighbor_mask * (1 - torch.eye(adj.shape[0], device=self.device))

        # 4. discard low-similarity neighbours
        clean_sim = torch.where(sim_matrix > self.sigma, sim_matrix, torch.zeros_like(sim_matrix))

        # 5. local consistency per spot
        rho = torch.sum(clean_sim * neighbor_mask, dim=1)

        # 6. normalize by neighbour count
        degree = torch.sum(neighbor_mask, dim=1) + 1e-6
        competence = rho / degree

        # 7. rescale to [0.1, 1.0]
        competence = (competence - competence.min()) / (competence.max() - competence.min() + 1e-6)
        competence = competence * 0.9 + 0.1

        return competence.detach()


if __name__ == "__main__":
    # ================= Configuration =================
    data_name = "E9.5_E2S2"
    save_dir = "./results_E9.5"
    os.makedirs(save_dir, exist_ok=True)

    epochs = 100
    lam, gam = 1., 1.
    s, tau = 0.6, 0.1
    hid_dims = [256, 256]
    lr, wd = 5e-04, 1e-04
    pd1, pd2, pm1, pm2 = 0.2, 0.6, 0.1, 0.0

    seed_list = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    csv_records = []

    print(f"\n========== Running for {data_name} ==========\n")
    print(f"Plan to run seeds: {seed_list}")

    # ================= Seed loop =================
    for i, seed in enumerate(seed_list):
        print(f"\n[{i + 1}/10] --- Running Seed: {seed} ---")
        fix_seed(seed)

        # 1. load data
        A, X, Y, adata = load_data1(data_name)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        in_dim = X.shape[1]

        # 2. build model
        encoder = ResGCN(in_dim, hid_dims).to(device)
        comp_scorer = CompetenceScorer(sigma=0.5, device=device).to(device)

        A, X = A.to(device), X.to(device)
        if not A.is_sparse:
            A_sparse = A.to_sparse()
        else:
            A_sparse = A
        A_sparse = A_sparse.coalesce()
        src, dst = A_sparse.indices()

        # pairs NOT connected by an edge (used by the spatial repulsion loss)
        mask = torch.full((A.shape[0], A.shape[0]), True, device=device)
        mask[src, dst] = False
        mask.fill_diagonal_(False)

        optimizer = Adam(encoder.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # best run of this seed
        best_ch = -1
        best_db = 0
        best_sc = -1
        best_emb = None
        best_labels = None

        # ================= Training loop =================
        for epoch in range(epochs):
            encoder.train()

            # warm-up: augmentation strength ramps up over the first 30% of epochs
            t = epoch / max(1, (epochs - 1))
            warm = min(1.0, t / 0.3)

            pd1_e = pd1 * warm
            pd2_e = pd2 * warm
            pm1_e = pm1 * warm
            pm2_e = pm2 * warm

            A1, X1 = augment(A, X, pd1_e, pm1_e)
            A2, X2 = augment(A, X, pd2_e, pm2_e)

            A_norm1 = normalize(A1, add_self_loops=True)
            A_norm2 = normalize(A2, add_self_loops=True)

            Z1, Z2 = encoder(A_norm1, A_norm2, X1, X2)

            # competence weights, mean-normalized to stabilise the gradient
            Z_mean = (Z1 + Z2) / 2
            weights = comp_scorer(Z_mean, A)
            weights = weights / (weights.mean().detach() + 1e-12)

            Z1_n = F.normalize(Z1, dim=1)
            Z2_n = F.normalize(Z2, dim=1)

            # node-level cross-view alignment loss
            sim_pos = (Z1_n * Z2_n).sum(dim=1)
            loss_ali = - (weights * sim_pos).mean()

            # edge-level neighbourhood consistency loss
            sim_nei_1 = (Z1_n[src] * Z1_n[dst]).sum(dim=1)
            sim_nei_2 = (Z2_n[src] * Z2_n[dst]).sum(dim=1)
            edge_weight = weights[src]
            loss_nei = - (edge_weight * (sim_nei_1 + sim_nei_2) / 2).mean()

            # global soft-gated repulsion loss on non-adjacent pairs
            S_mat = Z1_n @ Z2_n.T
            S_masked = torch.masked_select(S_mat, mask)
            S_sig = torch.sigmoid((S_masked - s) / tau)
            loss_spa = S_sig.mean()

            gam_e = gam * warm
            loss = loss_ali + lam * loss_nei + gam_e * loss_spa

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            # --- evaluation: internal clustering metrics (no ground truth) ---
            encoder.eval()
            with torch.no_grad():
                Z_eval = encoder.embed(normalize(A, add_self_loops=True), X)
            Z_numpy = Z_eval.cpu().numpy()
            adata.obsm['emb_pca'] = PCA(n_components=20).fit_transform(Z_numpy)

            try:
                Y_dummy = Y.cpu().numpy() if Y is not None else None
                pred_labels, _ = evaluate(adata, Z_numpy, Y_dummy)

                if isinstance(pred_labels, list):
                    pred_labels = np.array(pred_labels)

                emb_pca = PCA(n_components=20).fit_transform(Z_numpy)
                current_ch = metrics.calinski_harabasz_score(emb_pca, pred_labels)
                current_db = metrics.davies_bouldin_score(emb_pca, pred_labels)
                current_sc = metrics.silhouette_score(emb_pca, pred_labels)
            except Exception as e:
                current_ch, current_db, current_sc = -1, 0, -1

            # keep the epoch with the highest CH score
            if current_ch > best_ch:
                best_ch = current_ch
                best_db = current_db
                best_sc = current_sc
                best_emb = Z_numpy.copy()
                best_labels = pred_labels.copy()

        # ================= Save results of this seed =================
        print(f"[{data_name}] Seed:{seed} Best CH: {best_ch:.4f}, DB: {best_db:.4f}, SC: {best_sc:.4f}")

        adata.obsm['domain_best'] = best_emb
        adata.obs['predicted'] = best_labels.astype(str)

        # spatial domain map with title
        plt.rcParams["figure.figsize"] = (3, 3)
        out_path_v1 = os.path.join(save_dir, f"{data_name}_Seed{seed}_predicted_Normal.jpg")
        sc.pl.spatial(
            adata, img_key="hires", color='predicted', spot_size=1,
            show=False, title=f'STFCOUS (CH={best_ch:.2f})'
        )
        plt.savefig(out_path_v1, bbox_inches='tight', dpi=600)
        plt.close()

        # spatial domain map without axes and legend
        out_path_v2 = os.path.join(save_dir, f"{data_name}_Seed{seed}_predicted_Clean.jpg")
        fig, ax = plt.subplots(figsize=(3, 3))
        sc.pl.spatial(
            adata, img_key="hires", color='predicted', spot_size=1,
            show=False,
            title='', legend_loc=None, ax=ax
        )
        ax.axis('off')
        plt.savefig(out_path_v2, bbox_inches='tight', dpi=600)
        plt.close()

        adata.write(os.path.join(save_dir, f"{data_name}_Seed{seed}_predicted.h5ad"))

        csv_records.append({
            "Dataset": data_name,
            "Seed": seed,
            "Best_CH": best_ch,
            "Best_DB": best_db,
            "Best_SC": best_sc
        })
        pd.DataFrame(csv_records).to_csv(os.path.join(save_dir, "training_metrics.csv"), index=False)

    # ================= Final summary =================
    df = pd.DataFrame(csv_records)
    print("\n======== Summary Across All 10 Runs ========")
    print(df)
    print(f"Avg CH: {df['Best_CH'].mean():.4f}")

    df.to_csv(os.path.join(save_dir, "Final_Summary_Posterior.csv"), index=False)
