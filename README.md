# SpaceRec

SpaceRec predicts dense grid-level gene expression and cell-type probabilities
from H&E histology using Visium supervision. This repository contains the
current slim workflow:

1. RCTD deconvolution for spot-level cell-type proportions.
2. Dense18 Virchow2 grid embeddings.
3. Projection-heads training.
4. Area-weighted aggregation from grids to polygons/cells.
5. Window-based type and expression visualization.

Main notebook: [tutorial](https://github.com/OliverWang0908/SpaceRec/blob/main/spacerec/notebooks/run_spacerec.ipynb)

Main API:

```python
import spacerec.api as spacerec
```

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

## Quick Start

Open and run the [tutorial](https://github.com/OliverWang0908/SpaceRec/blob/main/spacerec/notebooks/run_spacerec.ipynb).

The notebook is organized into five steps:

```text
Step 1: Deconvolution
Step 2: Grid Embedding
Step 3: Train
Step 4: Aggregate
Step 5: Evaluation
```

Current BRCA notebook settings:

```text
Step 2: max_patches = None
Step 3: projection_dim = 512
Step 3: max_epochs = 60
Step 3: batch_size = 4
Step 3: limit_spots = None
```

For smoke tests, use a small `max_patches`, a small `limit_spots`, and fewer
epochs.

## API Overview

```python
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

```text
x_s ~= sum_k p_{s,k} r_k
p_{s,k} >= 0
sum_k p_{s,k} = 1
```

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

### Step 4: Aggregate

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

```text
grid embedding:
  n_export_patches = 24423
  n_grids = 391344
  n_supervised_grids = 172966
  feature_dim = 6400

training:
  n_supervised_spots = 4740
  n_genes = 4000
  n_cell_types = 11
  mean_gene_PCC ~= 0.7751
  mean_spot_gene_PCC ~= 0.8607

aggregation:
  n_input_polygons = 139208
  n_output_cells = 134364
```

These are run artifacts, not fixed expected values.

## Notes

- `data/`, `results/`, caches, local progress logs, and editor settings are
  ignored by git.
- The slim package intentionally keeps only the workflow described above.
- If plots are blank, first check that the full-resolution window overlaps the
  available grid outputs and polygon reference data.
