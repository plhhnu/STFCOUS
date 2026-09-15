# STFCOUS: Multi-Level Graph Contrastive Learning for Spatial Domain Identification in Spatial Transcriptomics

STFCOUS is a self-supervised deep learning framework for spatial domain identification in spatial transcriptomics. It learns spot-level representations from gene expression profiles and spatial coordinates through competence-weighted multi-level graph contrastive learning, without requiring any histological annotation.

## Framework Overview
![STFCOUS Framework](Figure/STFCOUS.png)

The framework mainly includes five components:

- **Asymmetric graph augmentation:** Two augmented graph views are built by random feature masking and random edge dropping, with one view perturbed on the topology only so that the expression semantics are fully preserved; perturbation strength is warmed up over the first 30% of epochs.
- **DyT-based residual graph encoder:** A weight-shared ResGCN encoder aggregates neighbourhood information through graph convolution, injects the node's own expression back via residual connections, and replaces fixed activations and normalization layers with the dynamic tanh (DyT) module, preserving distribution differences between spatial domains.
- **Competence scoring:** A per-spot confidence weight is estimated from neighbourhood embedding consistency, so that noisy and tissue-boundary spots exert a weaker influence on the objective.
- **Multi-level contrastive optimization:** Three complementary terms are jointly optimized: a node-level cross-view alignment loss, an edge-level neighbourhood consistency loss, and a global-level soft-gated spatial repulsion loss that pushes non-adjacent but similar spot pairs apart.
- **Clustering and evaluation:** Embeddings are clustered with mclust (number of clusters set to the number of annotated layers), followed by majority-vote spatial label refinement; performance is measured by ARI and NMI against the ground truth.

## Data availability
The DLPFC dataset (12 sections, 151507-151676) used in this study is publicly available from the spatialLIBD resource (https://research.libd.org/spatialLIBD/). The Human Breast Cancer (Block A Section 1) and Adult Mouse Brain (FFPE) datasets used in additional experiments were obtained from the 10x Genomics public datasets portal: https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-0-0 and https://www.10xgenomics.com/datasets/adult-mouse-brain-ffpe-1-standard-1-3-0.

### Data preparation
Place each Visium section under `./data/` in the standard folder layout, together with a `metadata.tsv` file containing the `layer_guess` annotation column:

```
data/
├── 151507/
│   ├── filtered_feature_bc_matrix.h5
│   ├── spatial/
│   └── metadata.tsv          # must contain a `layer_guess` column
├── 151508/
...
```

## Installation

#### GPU Acceleration Notice

To fully utilize GPU acceleration, please make sure that the installed PyTorch version is compatible with your CUDA environment. This project has been tested on Linux with Python 3.8.20, NVIDIA GeForce RTX 2080 Ti GPUs, and CUDA Toolkit 11.8. We recommend installing a PyTorch version compatible with CUDA 11.8, such as PyTorch 2.2.2+cu118.

If PyTorch is installed using the default command without specifying the CUDA version, a CPU-only version may be installed, which will prevent the model from using GPU acceleration. Please refer to the official PyTorch installation guide and select the appropriate command according to your CUDA version. For example, for CUDA 11.8, you can use:

```bash
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118
```

After installation, you can check whether GPU acceleration is available using:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

If `torch.cuda.is_available()` returns `True`, the GPU environment has been successfully configured.

### Option 1: Install with `environment.yml` (Recommended)

#### Step 1. Create a new conda environment:

```
conda create -n stfcous python=3.8 r-base r-essentials -y
```

Activate the environment:

```
conda activate stfcous
```

Install the required R package `mclust`:

```
R
```

In the R console, run:

```
install.packages("mclust")
library(mclust)
q()
```

Then return to the terminal and update the environment using `environment.yml`:

```
conda env update -n stfcous -f environment.yml
```

#### Step 2. Final Check

After installation, run the following commands:

```
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python -c "import scanpy as sc; import anndata; import sklearn; print('Basic packages installed successfully')"
python -c "import torch_geometric; import rpy2; from munkres import Munkres; print('Project packages installed successfully')"
```

If all commands run without errors, the environment is ready.

### Option 2: Install Manually Without `environment.yml`

If the automatic installation using `environment.yml` fails, you can install the environment manually.

#### Step 1: Create a conda environment

```
conda create -n stfcous python=3.8
conda activate stfcous
```

#### Step 2: Install R and mclust

```
conda install r-base r-essentials
```

Start R:

```
R
```

Install `mclust` in the R console:

```
install.packages("mclust")
library(mclust)
q()
```

#### Step 3: Install PyTorch with CUDA 11.8

```
pip install torch==2.2.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

#### Step 4: Install PyG

```
pip install torch_geometric==2.5.2
pip install torch_scatter==2.1.2+pt22cu118 torch_sparse==0.6.18+pt22cu118 -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
```

#### Step 5: Install Common Packages

```
pip install scanpy==1.9.3 anndata==0.9.1 numpy==1.22.4 pandas==2.0.3 scipy==1.10.1 scikit-learn==1.3.0 matplotlib==3.7.1 pillow==10.2.0
```

#### Step 6: Install Other Packages

```
pip install umap-learn==0.5.7 rpy2==3.5.11 munkres==1.1.4
```

#### Step 7: Final Check

```
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python -c "import scanpy as sc; import anndata; import sklearn; print('Basic packages installed successfully')"
python -c "import torch_geometric; import rpy2; from munkres import Munkres; print('Project packages installed successfully')"
```

If all commands run without errors, the environment is ready.

## Usage

Place `DLPFC-main.py`, `model.py`, `util.py` and `preprocess.py` in the same directory, prepare the data as described above, and run:

```
python DLPFC-main.py
```

The script trains STFCOUS on all 12 DLPFC sections over 10 random seeds (42-51). For each section it saves, under `results_DLPFC/`:

- `{section}_spatial.jpg` — spatial domain map of the best run on the H&E image;
- `{section}_spatial_dark.jpg` — slightly darkened version for figures;
- `{section}_umap.jpg` — UMAP of the best embedding;
- `final_summary_results.csv` — mean / std / max ARI and NMI over seeds for every section.

## Parameter Settings

STFCOUS uses a unified training configuration across sections; only the spatial graph radius is adjusted per dataset according to spot density.

### Core Parameters

| Parameter      | Description                                                | Value       |
| -------------- | ---------------------------------------------------------- | ----------- |
| `epochs`       | Training epochs                                            | 150         |
| `lr`           | Learning rate (Adam, cosine annealing)                     | 5e-4        |
| `wd`           | Weight decay                                               | 1e-4        |
| `hid_dims`     | ResGCN hidden dimensions                                   | [256, 256]  |
| `lam`          | Weight of the neighbourhood consistency loss (L_nei)       | 1.0         |
| `gam`          | Weight of the spatial repulsion loss (L_spa)               | 1.0         |
| `s`            | Soft-gate similarity threshold of L_spa                    | 0.6         |
| `tau`          | Soft-gate temperature of L_spa                             | 0.1         |
| `sigma`        | Similarity threshold of the competence scorer              | 0.4         |
| `pd1` / `pd2`  | Edge-drop rates of the two augmented views                 | 0.2 / 0.6   |
| `pm1` / `pm2`  | Feature-mask rates of the two augmented views              | 0.1 / 0.0   |
| `warm-up`      | Fraction of epochs used to ramp up augmentation strength   | 0.3         |
| `max_grad_norm`| Gradient clipping norm                                     | 5.0         |
| `rad_cutoff`   | Radius for spatial network construction (DLPFC)            | 200         |
| `PCA (input)`  | PCA dimension of input expression features                 | 300         |
| `PCA (cluster)`| PCA dimension before mclust clustering                    | 20          |
| `seeds`        | Random seeds for repeated runs                             | 42 - 51     |

### Notes

- Clustering uses mclust on the 20-dim PCA of the learned embedding, with the number of clusters set to the number of annotated layers; labels are then refined by majority vote over the 20 nearest spatial neighbours.
- For new datasets, we recommend starting from the default configuration and mainly tuning `rad_cutoff` according to the spot density and resolution of the platform.

## Reference:

Please consider citing the following reference:

```

```
