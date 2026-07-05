<div align="center">

# SpaceRec

**predicting high resolution gene expression and cell type composition in spatial transcriptomics.**

[Tutorial](https://spacerecproject.github.io/SpaceRec/tutorials/) · [Notebook](spacerec/notebooks/run_spacerec.ipynb) · [API Overview](#api-overview) · [Workflow](#workflow)

</div>

---

SpaceRec builds a histology-guided model for high-resolution spatial
reconstruction, learning to predict dense grid-level gene expression and
cell-type probabilities from H&E image features under Visium supervision.


## At A Glance

| Stage | Purpose | Main output |
| --- | --- | --- |
| 1. Deconvolution    | Estimate spot-level cell-type proportions with RCTD. | `results/brca/deconv/deconv.csv` |
| 2. Grid Embedding | Extract dense18 Virchow2 H&E grid features. | `results/brca/grid_embedding/grid_embedding.h5` |
| 3. Train | Train expression and type projection heads. | `results/brca/train/grid_predictions.h5` |
| 4. Aggregate | Aggregate grid predictions to polygon-level expression and type assignments. | `results/brca/aggregate/spacerec_ct.csv` |
| 5. Evaluation | Render side-by-side type and expression checks. | `results/brca/Evaluation/` |

## Data

Large artifacts are not stored in GitHub. After downloading, place them at the
repository root:

```text
SpaceRec/
  data/
  results/
```

Download link:

```text
Google Drive: https://drive.google.com/open?id=1Fxyag8rx4A-DDvfdCk6xUd_SKmtTz5vw
```

`data/` contains the packaged BRCA example inputs. `results/` contains generated
deconvolution, grid embedding, training, aggregation, and evaluation outputs.

## Environment Setup

Clone the repository, enter it, and create a dedicated conda environment:

```bash
git clone https://github.com/SpaceRecProject/SpaceRec.git
cd SpaceRec
conda create -n spacerec python=3.10 -y
conda activate spacerec
pip install torch lightning numpy pandas scipy h5py anndata pillow matplotlib shapely timm
```

RCTD also needs the R packages `Matrix`, `Seurat`, and `spacexr`. Check the
setup from the repository root before running the notebook:

```bash
python -c "import spacerec.api as spacerec; print(spacerec.__file__)"
python -c "import torch; print(torch.cuda.is_available())"
Rscript -e "library(Matrix); library(Seurat); library(spacexr)"
```

## Quick Start

Open the [tutorial](https://spacerecproject.github.io/SpaceRec/tutorials/) or run the notebook directly:

```text
spacerec/notebooks/run_spacerec.ipynb
```

The notebook is organized into five explicit execution stages:

```text
Step 1: Deconvolution
Step 2: Grid Embedding
Step 3: Train
Step 4: Aggregate
Step 5: Evaluation
```

Current BRCA notebook settings:

| Step | Setting | Value |
| --- | --- | --- |
| Step 2 | `max_patches` | `None` |
| Step 3 | `projection_dim` | `512` |
| Step 3 | `max_epochs` | `60` |
| Step 3 | `batch_size` | `4` |
| Step 3 | `limit_spots` | `None` |

For smoke tests, use a small `max_patches`, a small `limit_spots`, and fewer
epochs.

## API Overview

```python
import spacerec.api as spacerec

spacerec.deconv(...)    # spot-level cell-type proportions
spacerec.ge(...)        # dense18 Virchow2 grid embeddings
spacerec.train(...)     # projection-heads model training
spacerec.agg(...)       # grid-to-polygon/cell aggregation
spacerec.plottype(...)  # type visualization
spacerec.plotexpr(...)  # expression visualization
```

Default outputs are written under:

```text
results/<dataset>/<step>/
```

Supported API dataset names are `brca` and `crc`; this packaged example is BRCA.

## Workflow

### Step 1: Deconvolution

```python
spacerec.deconv(...)
```

Estimates spot-level cell-type proportions from Visium counts and an annotated
single-cell reference. These proportions supervise the type head during
training.

For spot `s`:

$$
x_s \approx l_s \sum_k p_{s,k} r_k,\quad
p_{s,k} \ge 0,\quad
\sum_k p_{s,k}=1.
$$

Main output:

```text
results/brca/deconv/deconv.csv
```

### Step 2: Grid Embedding

```python
spacerec.ge(...)
```

Generates dense grid features from H&E using Virchow2. The retained path is
`dense18_virchow2` with final 6400-dimensional grid embeddings:

```text
token feature:          1280
same-patch tile:        2560
neighbor tile context:  2560
final feature:          6400
```

Each grid view is:

$$
h_{g,v} = [t_{g,v}; u_v; n_v] \in \mathbb{R}^{6400}.
$$

Overlapping patch views are combined after concatenation with raised-cosine /
Hann-like weights:

$$
h_g = \frac{\sum_v w_{g,v} h_{g,v}}{\sum_v w_{g,v}}.
$$

Main output:

```text
results/brca/grid_embedding/grid_embedding.h5
```

### Step 3: Train

```python
spacerec.train(...)
```

Trains the projection-heads model:

```text
grid feature h_g
  -> Linear + GELU + LayerNorm
  -> GeneHead
  -> TypeHead
```

With the notebook setting, the first projection is:

$$
6400 \rightarrow 512.
$$

The model predicts grid-level expression and type probabilities, then aggregates
them to Visium spots for supervision:

$$
\hat{x}_s = \sum_{g \in G_s} \hat{x}_g,\quad
\hat{p}_s = \frac{1}{|G_s|}\sum_{g \in G_s}\hat{q}_g.
$$

Loss:

$$
\mathcal{L}_{expr}
= \mathrm{Huber}\left(\log(1+\hat{x}_s), \log(1+x_s)\right)
$$

$$
\mathcal{L}_{deconv}
= \mathrm{KL}\left(p_s \Vert \hat{p}_s\right)
$$

$$
\mathcal{L}_{conf}
= -\mathbb{E}_g \log \max_k \hat{q}_{g,k}
$$

$$
\mathcal{L}
= \mathcal{L}_{expr}
+ \lambda_{type}
\left[
\alpha \mathcal{L}_{conf}
+ (1-\alpha)\mathcal{L}_{deconv}
\right].
$$

Main outputs:

```text
results/brca/train/grid_predictions.h5
results/brca/train/grid_type.csv
results/brca/train/grid_expr.h5ad
results/brca/train/model/best_train_model.ckpt
```

### Step 4: Aggregate

```python
spacerec.agg(...)
```

After training, the model generated predictions for all retained dense grids:

$$
\{\hat{e}_g,\hat{p}_g\}_{g=1}^{G}.
$$

For a target polygon $a$, let $\mathcal{G}(a)$ denote the set of overlapping
dense grids. Let $A_g$ be the area of one dense grid and $A_{a,g}$ be the
intersection area between polygon $a$ and grid $g$. The fractional overlap
weight was defined as:

$$
\rho_{a,g} = \frac{A_{a,g}}{A_g}.
$$

Because grid-level expression predictions represent additive expression
contributions, polygon-level expression was computed by area-fraction weighted
summation:

$$
\tilde{e}_a
= \sum_{g\in\mathcal{G}(a)}
\rho_{a,g}\hat{e}_g.
$$

The exported polygon-level expression was then log-transformed:

$$
\hat{e}_a
= \log\left(1+\max(\tilde{e}_a,0)\right).
$$

Polygon-level cell-type probabilities were computed from the area-weighted sum
of grid-level probabilities:

$$
\tilde{p}_a
= \sum_{g\in\mathcal{G}(a)}
\rho_{a,g}\hat{p}_g.
$$

The result was normalized across cell types:

$$
\hat{p}_{a,c}
= \frac{\tilde{p}_{a,c}}{\sum_{c'=1}^{C}\tilde{p}_{a,c'}}.
$$

The final cell-type assignment was:

$$
\hat{c}_a
= \arg\max_c \hat{p}_{a,c}.
$$

Main outputs:

```text
results/brca/aggregate/spacerec_ct.csv
results/brca/aggregate/spacerec_polygon.csv
results/brca/aggregate/spacerec_expr.h5ad
```

### Step 5: Evaluation

```python
spacerec.plottype(...)
spacerec.plotexpr(...)
```

Generates side-by-side visual checks:

```text
xen_type.png   vs grid_type.png
xen_expr.png   vs grid_expr.png
```

`window_label` and `polygon_window_label` are display labels only. True Xenium
polygons are selected by full-resolution `window` coordinates.

## Current BRCA Run

The current full BRCA run reports:

| Step | Metric | Value |
| --- | --- | --- |
| Grid embedding | `n_export_patches` | `24423` |
| Grid embedding | `n_grids` | `391344` |
| Grid embedding | `n_supervised_grids` | `172966` |
| Grid embedding | `feature_dim` | `6400` |
| Training | `n_supervised_spots` | `4740` |
| Training | `n_genes` | `4000` |
| Training | `n_cell_types` | `11` |
| Training | `mean_gene_PCC` | `~0.7751` |
| Training | `mean_spot_gene_PCC` | `~0.8607` |
| Aggregation | `n_output_cells` | `134364` |

These are run artifacts, not fixed expected values.
