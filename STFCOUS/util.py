"""
Utilities for STFCOUS.

Contains:
    * load_data1      -- load a Visium section, build the spatial adjacency
                         matrix, ground-truth labels and PCA features;
    * augment, mask_feat, drop_edge
                     -- graph augmentation operators (random feature
                         masking and random edge dropping);
    * normalize, add_self_loop, sparse_identity, sparse_diag
                     -- sparse graph normalization helpers;
    * mclust_R, evaluate
                     -- mclust clustering + ARI / NMI / ACC / F1 evaluation;
    * refine_label, metrics, fix_seed, get_logger
                     -- label refinement, clustering metrics and misc helpers;
    * Cal_Spatial_Net, Stats_Spatial_Net
                     -- spatial network construction / statistics.

Dependencies: torch, scanpy, scikit-learn, scipy, pandas, matplotlib,
rpy2 (R package `mclust`) and the local module `preprocess`
(spatial network construction and expression preprocessing).
"""

import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
import scanpy as sc
import scipy.sparse as sp
import torch
from munkres import Munkres
from rpy2.robjects import r
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, \
    normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from sklearn import neighbors

import preprocess

# --------------------------------------------------------------------- #
# R environment: mclust is required for clustering.                      #
# Uncomment the next line if mclust lives in a non-default R library.    #
# --------------------------------------------------------------------- #
# robjects.r('.libPaths("/path/to/your/Rlibs")')
robjects.r('library(mclust)')
mclust = rpackages.importr('mclust')


def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    """\
    Construct the spatial neighbor network.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is
        connected to spots whose distance is less than rad_cutoff. When
        model=='KNN', the spot is connected to its first k_cutoff nearest
        neighbors.

    Returns
    -------
    The spatial network is saved in adata.uns['Spatial_Net']
    """

    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))

    if model == 'KNN':
        nbrs = neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index),))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net


def Stats_Spatial_Net(adata):
    """Plot the distribution of the number of neighbors per spot."""
    Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
    Mean_edge = Num_edge / adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df / adata.shape[0]
    fig, ax = plt.subplots(figsize=[3, 2])
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)' % Mean_edge)
    ax.bar(plot_df.index, plot_df)


def load_data1(dataset):
    """Load a Visium section and return (A, X, Y, adata):
    A -- dense spatial adjacency matrix (float32, self-loops removed);
    X -- PCA-reduced expression features (n_components=300);
    Y -- integer-encoded ground-truth labels;
    adata -- the AnnData object.
    """
    # 1. read the Visium section and the ground-truth layer annotations
    adata = sc.read_visium(f'./data/{dataset}/')
    adata.var_names_make_unique()

    df_meta = pd.read_csv(f'./data/{dataset}/metadata.tsv', sep='\t')
    adata.obs['ground_truth'] = df_meta['layer_guess'].values

    # filter out NA nodes
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]

    # 2. spatial network construction + preprocessing
    preprocess.Cal_Spatial_Net(adata, rad_cutoff=200)
    preprocess.preprocess(adata)
    preprocess.Stats_Spatial_Net(adata)

    if 'highly_variable' in adata.var.columns:
        adata_Vars = adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata

    data1 = preprocess.Transfer_pytorch_Data(adata_Vars, dim_reduction=False)

    # 3. integer labels
    le = LabelEncoder()
    label = le.fit_transform(adata.obs['ground_truth'])
    Y = torch.tensor(label, dtype=torch.long)

    # 4. PCA-reduced features
    x = data1.x.to(torch.float32)
    X = PCA(n_components=300).fit_transform(x)
    X = torch.FloatTensor(X)

    # 5. dense adjacency matrix (self-loops removed)
    Stats_Spatial_Net(adata)
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])),
                      shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])
    adj = G - sp.dia_matrix((G.diagonal()[np.newaxis, :], [0]), shape=G.shape)
    adj.eliminate_zeros()
    adj = adj.tocoo()

    A = torch.tensor(adj.toarray(), dtype=torch.float32)

    return A, X, Y, adata


def fix_seed(seed):
    """Fix random seeds (python / numpy / torch / cudnn)."""
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def metrics(y_true, y_pred):
    """Evaluate clustering performance.

    :param y_true: ground truth
    :param y_pred: prediction
    :returns acc, nmi, ari, f1:
    - accuracy (after optimal label matching via the Hungarian algorithm)
    - normalized mutual information
    - adjusted rand index
    - macro f1 score
    """
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    ari = adjusted_rand_score(y_true, y_pred)

    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                if ind < len(y_pred):
                    y_pred[ind] = i
                    ind += 1
        l2 = list(set(y_pred))
        num_class2 = len(l2)
        if num_class1 != num_class2:
            print('Warning: Number of classes in y_true and y_pred do not match.')
            return 0.0, nmi, ari, 0.0
    cost = np.zeros((num_class1, num_class2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = accuracy_score(y_true, new_predict)
    f1 = f1_score(y_true, new_predict, average='macro')

    return acc, nmi, ari, f1


def refine_label(adata, radius=20, key='label'):
    """Refine labels by majority vote over the `radius` nearest neighbours
    in physical space."""
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    distance = cdist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = np.array(new_type)

    # cast to int if the labels came back as strings / objects
    if new_type.dtype.kind in {'U', 'S', 'O'}:
        new_type = new_type.astype(int)

    return new_type


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """Cluster the embedding in adata.obsm[used_obsm] with the R package
    mclust; the resulting labels are stored in adata.obs['mclust']."""
    import rpy2.robjects.numpy2ri as numpy2ri
    numpy2ri.activate()

    robjects.r(f'set.seed({random_seed})')
    robjects.r('library(mclust)')

    np.random.seed(random_seed)
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')

    return adata


def evaluate(adata, Z: np.ndarray, Y: np.ndarray, logger=None):
    """Cluster the embedding Z with mclust (on adata.obsm['emb_pca']),
    refine the labels and return (ari, Y_, nmi)."""
    n_clusters = len(np.unique(Y))

    # fix the seed so clustering is reproducible
    fix_seed(0)

    # mclust clustering
    adata = mclust_R(adata, n_clusters, modelNames='EEE',
                     used_obsm='emb_pca', random_seed=2020)
    Y_ = adata.obs['mclust']
    adata.obs['nb'] = Y_

    # majority-vote refinement over spatial neighbours
    new_type = refine_label(adata, 20, key='nb')
    Y_ = new_type

    acc, nmi, ari, f1 = metrics(Y, Y_)

    return ari, Y_, nmi


def get_logger(filename, verbosity=1, name=None, mode='a'):
    """Return a logger writing both to file and to stdout."""
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("%(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, mode)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# ------------------- Graph augmentation ------------------- #
def augment(A: torch.sparse.Tensor, X: torch.Tensor,
            edge_mask_rate: float, feat_drop_rate: float):
    """Randomly drop edges and mask features to build an augmented view."""
    A = drop_edge(A, edge_mask_rate)
    X = mask_feat(X, feat_drop_rate)

    return A, X


def mask_feat(X: torch.Tensor, mask_prob: float):
    """Zero out a random subset of feature columns with probability mask_prob."""
    drop_mask = (
            torch.empty((X.size(1),), dtype=torch.float32, device=X.device).uniform_()
            < mask_prob
    )
    X = X.clone()
    X[:, drop_mask] = 0

    return X


def drop_edge(A: torch.sparse.Tensor, drop_prob: float):
    """Randomly remove edges with probability drop_prob."""
    n_edges = A._nnz()
    mask_rates = torch.full((n_edges,), fill_value=drop_prob,
                            dtype=torch.float)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)

    E = A._indices()
    V = A._values()

    E = E[:, mask_idx]
    V = V[mask_idx]
    A = torch.sparse_coo_tensor(E, V, A.shape, device=A.device)

    return A


# ------------------- Graph normalization ------------------- #
def add_self_loop(A: torch.sparse.Tensor):
    return A + sparse_identity(A.shape[0], device=A.device)


def normalize(A: torch.sparse.Tensor, add_self_loops=True, returnA=False):
    """Normalize the graph's adjacency matrix (symmetric normalization)
    in the torch.sparse.Tensor format."""
    if add_self_loops:
        A_hat = add_self_loop(A)
    else:
        A_hat = A

    D_hat_invsqrt = torch.sparse.sum(A_hat, dim=0).to_dense() ** -0.5
    D_hat_invsqrt[D_hat_invsqrt == torch.inf] = 0
    D_hat_invsqrt = sparse_diag(D_hat_invsqrt)
    A_norm = D_hat_invsqrt @ A_hat @ D_hat_invsqrt
    if returnA:
        return A_hat, A_norm
    else:
        return A_norm


def sparse_identity(dim, device):
    indices = torch.arange(dim).unsqueeze(0).repeat(2, 1)
    values = torch.ones(dim)
    identity_matrix = torch.sparse_coo_tensor(indices, values,
                                              size=(dim, dim), device=device)
    return identity_matrix


def sparse_diag(V: torch.Tensor):
    size = V.size(0)
    indices = torch.arange(size).unsqueeze(0).repeat(2, 1)
    values = V
    diagonal_matrix = torch.sparse_coo_tensor(indices, values,
                                              size=(size, size), device=V.device)
    return diagonal_matrix
