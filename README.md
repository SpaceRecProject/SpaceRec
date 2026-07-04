<div align="center">

# SpaceRec

**predicting high resolution gene expression and cell type composition in spatial transcriptomics.**

[Tutorial](docs/tutorials/) · [Notebook](spacerec/notebooks/run_spacerec.ipynb) · [API Overview](#api-overview) · [Workflow](#workflow)

</div>

---

SpaceRec packages a workflow for predicting dense grid-level
expression and cell-type probabilities from histology. The retained path uses
RCTD deconvolution, dense18 Virchow2 grid embeddings, projection-head training,
area-weighted polygon aggregation, and window-based visual evaluation.


## At A Glance

| Stage | Purpose | Main output |
| --- | --- | --- |
| 1. Deconvolution    | Estimate spot-level cell-type proportions with RCTD. | `results/brca/deconv/deconv.csv` |
| 2. Grid Embedding | Extract dense18 Virchow2 H&E grid features. | `results/brca/grid_embedding/grid_embedding.h5` |
| 3. Train | Train expression and type projection heads. | `results/brca/train/grid_predictions.h5` |
| 4. Aggregate | Transfer grid predictions to polygons/cells by overlap area. | `results/brca/aggregate/spacerec_ct.csv` |
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

Create a dedicated conda environment:

```bash
conda create -n spacerec python=3.10 -y
conda activate spacerec
pip install torch lightning numpy pandas scipy h5py anndata pillow matplotlib shapely timm
```

RCTD also needs the R packages `Matrix`, `Seurat`, and `spacexr`. Check the
setup before running the notebook:

```bash
python -c "import spacerec.api as spacerec; print(spacerec.__file__)"
python -c "import torch; print(torch.cuda.is_available())"
Rscript -e "library(Matrix); library(Seurat); library(spacexr)"
```

## Quick Start

Open the [tutorial](docs/tutorials/) or run the notebook directly:

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

<details open>
<summary><strong>Step 1: Deconvolution</strong></summary>

```python
spacerec.deconv(...)
```

Estimates spot-level cell-type proportions from Visium counts and an annotated
single-cell reference. These proportions supervise the type head during
training.

For spot `s`:

```text
x_s ~= l_s sum_k p_{s,k} r_k
p_{s,k} >= 0
sum_k p_{s,k} = 1
```

Main output:

```text
results/brca/deconv/deconv.csv
```

</details>

<details>
<summary><strong>Step 2: Grid Embedding</strong></summary>

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

```text
h_{g,v} = [t_{g,v}; u_v; n_v]
```

Overlapping patch views are combined after concatenation with raised-cosine /
Hann-like weights:

```text
h_g = sum_v w_{g,v} h_{g,v} / sum_v w_{g,v}
```

Main output:

```text
results/brca/grid_embedding/grid_embedding.h5
```

</details>

<details>
<summary><strong>Step 3: Train</strong></summary>

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

```text
6400 -> 512
```

The model predicts grid-level expression and type probabilities, then aggregates
them to Visium spots for supervision:

```text
xhat_s = sum_{g in G_s} xhat_g
phat_s = (1 / |G_s|) sum_{g in G_s} qhat_g
```

Loss:

```text
L_expr = Huber(log(1 + xhat_s), log(1 + x_s))
L_deconv = KL(p_s || phat_s)
L_conf = - mean_g log max_k qhat_{g,k}
L = L_expr + lambda_type * [alpha L_conf + (1 - alpha) L_deconv]
```

Main outputs:

```text
results/brca/train/grid_predictions.h5
results/brca/train/grid_type.csv
results/brca/train/grid_expr.h5ad
results/brca/train/model/best_train_model.ckpt
```

</details>

<details>
<summary><strong>Step 4: Aggregate</strong></summary>

```python
spacerec.agg(...)
```

Aggregates grid predictions to polygon/cell level by area overlap:

```text
a_{c,g} = area(polygon_c intersect grid_g)
xhat_c = sum_g a_{c,g} xhat_g / sum_g a_{c,g}
phat_c = sum_g a_{c,g} qhat_g / sum_g a_{c,g}
```

Main outputs:

```text
results/brca/aggregate/spacerec_ct.csv
results/brca/aggregate/spacerec_polygon.csv
results/brca/aggregate/spacerec_expr.h5ad
```

</details>

<details>
<summary><strong>Step 5: Evaluation</strong></summary>

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

</details>

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
