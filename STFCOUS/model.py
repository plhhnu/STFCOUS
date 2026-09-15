"""
Model definitions for STFCOUS.

Contains:
    * DyT         -- dynamic tanh activation with learnable per-channel
                     parameters, used in place of normalization layers;
    * GCNConv     -- graph convolution layer for sparse adjacency;
    * ResGCNLayer -- one residual GCN block (GCN + residual + DyT);
    * ResGCN      -- the STFCOUS encoder, weight-shared across the two
                     augmented graph views;
    * GCN         -- plain GCN baseline without residual / DyT
                     (kept for ablation studies).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DyT(nn.Module):
    """Dynamic tanh activation: gamma * tanh(alpha * x) + beta.

    `alpha` is a learnable scalar that maps features into the linear or
    saturation region of tanh; `gamma` and `beta` are learnable per-channel
    affine parameters. Unlike normalization layers it keeps no cross-sample
    statistics, so distribution differences between spatial domains are
    preserved.
    """

    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias


class GCNConv(nn.Module):
    """Graph convolution layer operating on a sparse normalized adjacency."""

    def __init__(self, in_dim, out_dim, activation=None):
        super(GCNConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, A_norm, X):
        assert isinstance(X, torch.Tensor)
        assert isinstance(A_norm, torch.sparse.Tensor)
        output = torch.matmul(X, self.weight)
        output = torch.spmm(A_norm, output) + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output


class ResGCNLayer(nn.Module):
    """One residual GCN block: graph convolution + residual connection,
    followed by DyT activation."""

    def __init__(self, in_dim, out_dim):
        super(ResGCNLayer, self).__init__()

        # graph convolution (no activation inside)
        self.gcn = GCNConv(in_dim, out_dim, activation=None)
        self.activation = DyT(out_dim)

        # project the skip connection when the feature dimension changes
        if in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim)
        else:
            self.res_proj = nn.Identity()

    def forward(self, A_norm, X):
        out = self.gcn(A_norm, X)
        res = self.res_proj(X)
        return self.activation(out + res)


class ResGCN(nn.Module):
    """STFCOUS encoder: a stack of ResGCNLayers shared between the two
    augmented graph views. `embed` returns L2-normalized embeddings."""

    def __init__(self, in_dim, hid_dims):
        super(ResGCN, self).__init__()

        self.encoder = nn.ModuleList()
        hid_dims = [in_dim] + list(hid_dims)
        for i in range(len(hid_dims) - 1):
            self.encoder.append(ResGCNLayer(hid_dims[i], hid_dims[i + 1]))

    def embed(self, A_norm, X):
        Z = X
        for layer in self.encoder:
            Z = layer(A_norm, Z)
        Z = F.normalize(Z, p=2, dim=1)
        return Z

    def forward(self, A_norm1, A_norm2, X1, X2):
        Z1 = self.embed(A_norm1, X1)
        Z2 = self.embed(A_norm2, X2)
        return Z1, Z2


class GCN(nn.Module):
    """Plain GCN baseline (SELU activations, no residual connection and
    no DyT), kept for ablation studies."""

    def __init__(self, in_dim, hid_dims):
        super(GCN, self).__init__()

        self.encoder = nn.ModuleList()
        hid_dims = [in_dim] + list(hid_dims)
        if len(hid_dims) > 2:
            for i in range(len(hid_dims) - 2):
                self.encoder.append(GCNConv(hid_dims[i], hid_dims[i + 1], activation=F.selu))
        self.encoder.append(GCNConv(hid_dims[-2], hid_dims[-1], activation=None))

    def embed(self, A_norm, X):
        Z = X
        for layer in self.encoder:
            Z = layer(A_norm, Z)
        Z = F.normalize(Z, p=2, dim=1)
        return Z

    def forward(self, A_norm1, A_norm2, X1, X2):
        # Z1 and Z2 have been normalized.
        Z1 = self.embed(A_norm1, X1)
        Z2 = self.embed(A_norm2, X2)
        return Z1, Z2
