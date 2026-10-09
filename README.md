<div align="center">

# SpaceRec

**Predict grid-level gene expression and cell types from H&E images under spatial transcriptomics supervision.**

[Notebook](spacerec/notebooks/spacerec.ipynb) · [API](#api) · [Workflow](#workflow)

</div>

---

SpaceRec learns dense spatial predictions from H&E image features and Visium
measurements. The current BRCA workflow builds a cell-type expression
reference, extracts and filters image-grid embeddings, trains cell-type and
expression models in two stages, and visualizes grid predictions inside a
user-selected full-resolution H&E bounding box.

## Repository layout

```text
SpaceRec/
├── spacerec/                    # active Python package
│   ├── notebooks/
│   │   └── spacerec.ipynb      # existing Visium data
│   ├── deconv/
│   ├── gridembedding/
│   ├── mask/
│   ├── model/
│   ├── evaluation/
│   └── rctd_ref/
└── resources/brca/              # BRCA inputs and generated outputs
```

The active import is:

```python
import spacerec.api as spacerec
```

## BRCA data

Download the BRCA data from:

```text
https://drive.google.com/open?id=1Fxyag8rx4A-DDvfdCk6xUd_SKmtTz5vw
```

Organize the files under `resources/brca`:

```text
resources/brca/
├── visium/
│   ├── st_filtered_feature_bc_matrix.h5
│   ├── tissue_positions.csv
│   ├── scalefactors_json.json
│   └── spatial/tissue_hires_image.png
├── he/he.tif
├── sc_ref/scRNA_adata_reannotated.h5ad
├── deconv/deconv.csv
├── truth/
└── config/brca_type_merge_17to11.json
```

Generated files remain in the same resource tree:

```text
resources/brca/
├── deconv/rctd_ref/
├── gene_list/
├── embedding/
├── mask/
├── model_pred/
│   ├── stage1_grid_type.csv
│   └── train/
└── Evaluation/
```

## Environment

The project uses the existing Python environment `spacerec_env` and a separate
R environment for RCTD:

```bash
conda activate spacerec_env
export SPACEREC_R_ENV="$HOME/.conda/envs/spacerec_r_env"
```

Validate the package and core runtimes from the repository root:

```bash
python -c "import spacerec.api as spacerec; print(spacerec.__file__)"
python -c "import torch; print(torch.cuda.is_available())"
"$SPACEREC_R_ENV/bin/Rscript" -e "library(Matrix); library(Seurat); library(spacexr); library(hdf5r)"
```

Training and full image embedding should run inside an allocated Slurm compute
job with a GPU. Plotting and lightweight inspection can run on a CPU compute
node. The notebook can discover GPUs assigned to the current Slurm job when
`CUDA_VISIBLE_DEVICES` is absent.

## Notebooks

### Existing spatial transcriptomics data

Open:

```text
spacerec/notebooks/spacerec.ipynb
```

This is the main BRCA notebook. It starts from existing Visium, H&E, and
single-cell reference inputs. It reuses the supplied deconvolution proportions
after validation and regenerates the expression reference and downstream
outputs. Stage 1 and Stage 2 write grid-level outputs only; cell aggregation is
not performed in these training cells.

## Workflow

| Step | Operation | Main output |
| --- | --- | --- |
| 1 | Validate supplied deconvolution and build the RCTD expression reference | `deconv/rctd_ref/rctd_reference_merged11.npy` |
| 2 | Extract dense Virchow2 grid features and build the H&E tissue mask | `embedding/grid_embedding_train_filtered.h5` |
| 3.1 | Train the router and cell-type model | Stage 1 checkpoint and `model_pred/stage1_grid_type.csv` |
| 3.2 | Train the expression model from the Stage 1 checkpoint | Stage 2 checkpoint, `grid_type.csv`, and `grid_expr.h5ad` |
| 4 | Plot cell type and gene expression for an editable H&E bbox | `model_pred/train/metrics/windows/<window>/` |

### Step 1: deconvolution and expression reference

The supplied spot proportions are validated against the Visium barcodes and
must be finite, nonnegative, and row-normalized. The expression reference uses
the single-cell reference and Visium gene space:

```python
spacerec.rctd_ref(...)
```

$$
x_s \approx l_s \sum_k p_{s,k} r_k,
\qquad p_{s,k}\ge 0,
\qquad \sum_k p_{s,k}=1.
$$

### Step 2: grid embedding and H&E mask

```python
spacerec.ge(...)
spacerec.mask(...)
```

The current configuration concatenates the Virchow2 class token and local tile
features:

$$
h_g=[t_g;u_g]\in\mathbb{R}^{3840}.
$$

The H&E mask removes background grids before training.

### Step 3.1: cell-type training

```python
spacerec.stage1(..., agg=False)
```

Grid probabilities are pooled to the Visium spot level for supervision:

$$
\hat{p}_s=\frac{1}{|G_s|}\sum_{g\in G_s}\hat{q}_g.
$$

Stage 1 writes the best router checkpoint and grid-level cell-type
probabilities.

### Step 3.2: expression training

```python
spacerec.stage2(..., stage1_checkpoint=..., agg=False)
```

Grid expression is summed to the spot level:

$$
\hat{x}_s=\sum_{g\in G_s}\hat{x}_g.
$$

The model combines expression and cell-type supervision:

$$
\mathcal{L}_{expr}
=\mathrm{Huber}\left(\log(1+\hat{x}_s),\log(1+x_s)\right),
$$

$$
\mathcal{L}_{type}
=\alpha\mathcal{L}_{conf}
+(1-\alpha)D_{\mathrm{KL}}(p_s\parallel\hat{p}_s).
$$

### Step 4: inspect grid predictions

```python
spacerec.plottype(..., window=(x0, y0, x1, y1))
spacerec.plotexpr(..., window=(x0, y0, x1, y1), gene="KRT5")
```

Expression is standardized independently for the selected gene and bbox, then
clipped to the displayed range:

$$
z_g=\min\left(2,\max\left(-2,
\frac{x_g-\mu_{\mathrm{bbox}}}{\sigma_{\mathrm{bbox}}}
\right)\right).
$$

The type and expression panels retain the same spatial scale and aspect ratio.

## Current BRCA settings

| Component | Setting | Value |
| --- | --- | --- |
| Grid embedding | `patch_size` | `480` px |
| Grid embedding | `stride` | `120` px |
| Grid output | grid spacing | `30` px |
| Grid embedding | feature dimension | `3840` |
| Grid embedding | neighbor features | disabled |
| Stage 1 and 2 | `projection_dim` | `2048` |
| Stage 1 and 2 | `batch_size` | `4` |
| Stage 1 | epochs | `70` |
| Stage 2 | epochs | `70` |
| Stage 1 and 2 | cell aggregation | disabled |

## API

```python
import spacerec.api as spacerec

spacerec.rctd_ref(...)  # expression reference from single-cell and Visium data
spacerec.ge(...)        # dense Virchow2 grid embeddings
spacerec.mask(...)      # H&E tissue mask and filtered training embeddings
spacerec.stage1(...)    # router and grid cell-type predictions
spacerec.stage2(...)    # grid cell-type and expression predictions
spacerec.agg(...)       # optional grid-to-cell aggregation
spacerec.plottype(...)  # grid-level cell-type visualization
spacerec.plotexpr(...)  # grid-level expression visualization
```

Cell-level outputs can be generated separately with `spacerec.agg(...)` when a
polygon file is available. The main existing-data notebook keeps aggregation
separate from both training stages.

## Main outputs

```text
resources/brca/model_pred/
├── stage1_grid_type.csv
└── train/
    ├── model/stage1_router/checkpoints/best.ckpt
    ├── model/stage2_expression/checkpoints/best.ckpt
    ├── grid_predictions.h5
    ├── grid_type.csv
    ├── grid_expr.h5ad
    └── metrics/windows/
```
