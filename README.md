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

<details>
<summary><strong>Step 1: Deconvolution</strong></summary>

```python
spacerec.deconv(...)
```

$$
x_s \approx l_s \sum_k p_{s,k} r_k,\quad
p_{s,k} \ge 0,\quad \sum_k p_{s,k}=1.
$$

Output: `results/brca/deconv/deconv.csv`

</details>

<details>
<summary><strong>Step 2: Grid Embedding</strong></summary>

```python
spacerec.ge(...)
```

$$
h_{g,v} = [t_{g,v}; u_v; n_v] \in \mathbb{R}^{6400},\quad
h_g = \frac{\sum_v w_{g,v}h_{g,v}}{\sum_v w_{g,v}}.
$$

Output: `results/brca/grid_embedding/grid_embedding.h5`

</details>

<details>
<summary><strong>Step 3: Train</strong></summary>

```python
spacerec.train(...)
```

$$
\hat{x}_s = \sum_{g \in G_s}\hat{x}_g,\quad
\hat{p}_s = \frac{1}{|G_s|}\sum_{g \in G_s}\hat{q}_g.
$$

$$
\mathcal{L}_{expr} = \mathrm{Huber}(\log(1+\hat{x}_s),\log(1+x_s)).
$$

$$
\mathcal{L}_{type} = \alpha\mathcal{L}_{conf} + (1-\alpha)\mathrm{KL}(p_s\parallel\hat{p}_s).
$$

$$
\mathcal{L} = \mathcal{L}_{expr} + \lambda_{type}\mathcal{L}_{type}.
$$

Outputs: `results/brca/train/grid_predictions.h5`, `grid_type.csv`, `grid_expr.h5ad`, `model/best_train_model.ckpt`

</details>

<details>
<summary><strong>Step 4: Aggregate</strong></summary>

```python
spacerec.agg(...)
```

$$
\{\hat{e}_g,\hat{p}_g\}_{g=1}^{G},\quad
\rho_{a,g}=\frac{A_{a,g}}{A_g}.
$$

$$
\tilde{e}_a=\sum_{g\in\mathcal{G}(a)}\rho_{a,g}\hat{e}_g,\quad
\hat{e}_a=\log\left(1+\max(\tilde{e}_a,0)\right).
$$

$$
\tilde{p}_a=\sum_{g\in\mathcal{G}(a)}\rho_{a,g}\hat{p}_g,\quad
\hat{p}_{a,c}=\frac{\tilde{p}_{a,c}}{\sum_{c'=1}^{C}\tilde{p}_{a,c'}},\quad
\hat{c}_a=\arg\max_c \hat{p}_{a,c}.
$$

Outputs: `results/brca/aggregate/spacerec_ct.csv`, `spacerec_polygon.csv`, `spacerec_expr.h5ad`

</details>

<details>
<summary><strong>Step 5: Evaluation</strong></summary>

```python
spacerec.plottype(...)
spacerec.plotexpr(...)
```

Outputs: `results/brca/Evaluation/xen_type.png`, `grid_type.png`, `xen_expr.png`, `grid_expr.png`

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
