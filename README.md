# SpaceRec Slim Package

SpaceRec is a slim, standalone spatial reconstruction workflow for predicting
dense grid-level gene expression and cell-type probabilities from H&E histology
and Visium supervision. This package keeps the current production path only:

1. RCTD deconvolution for spot-level cell-type proportions.
2. Dense18 Virchow2 grid embeddings.
3. No-Set-Transformer projection-heads training.
4. Area-weighted aggregation from grids to cell/polygon regions.
5. Window-based visual evaluation against Xenium-style references.

The main public API is in `spacerec/api.py`, and the main notebook is
`spacerec/notebooks/run_spacerec.ipynb`.

## Repository Layout

```text
SpaceRec/
  README.md
  spacerec/
    api.py
    instruction.txt
    notebooks/run_spacerec.ipynb
    deconv/
    gridembedding/
    model/
    evaluation/
    preprocessing/
  data/
    packaged BRCA example inputs and reference outputs
  results/
    current BRCA run outputs
```

Important packaged data files:

```text
data/he.tif
data/visium/tissue_positions.csv
data/visium/scalefactors_json.json
data/visium/spatial/tissue_hires_image.png
data/visium/st_filtered_feature_bc_matrix.h5
data/st.h5ad
data/genes.txt
data/scRNA_adata_reannotated.h5ad
data/true_xen_type.csv
data/true_xen_type_merge.json
data/true_xen_expr.h5
data/xen_polygon_fullres.csv
```

The package is intended to be self-contained. Files under `data/` should be real
files or directories, not symlinks.

## Data And Results Availability

The GitHub repository intentionally excludes `data/` and `results/` because
they contain large input data, model artifacts, and generated outputs.

Download link:

```text
Google Drive: TODO - add shared folder link here
```

After downloading, place the folders at the project root:

```text
SpaceRec/
  data/
  results/
```

The code, notebook, and documentation are tracked in git. The large artifacts
are kept outside git and are ignored by `.gitignore`.

## Cluster And CUDA Notes

Do not run CUDA work on a login node. For grid embedding and model training on
the Pitt GPU cluster, enter the active Slurm allocation before running the
notebook or scripts:

```bash
/opt/slurm/bin/srun --jobid=<JOB_ID> --pty bash
conda activate base
# If needed, activate the environment used for SpaceRec, for example:
# conda activate spacerec
cd /net/dali/home/chikina/shared_data/SpaceRec
```

The API has GPU auto-selection helpers for `spacerec.ge()` and
`spacerec.train()`, but it still assumes the process is running inside a valid
compute-node allocation.

## Quick Start

From the project root:

```python
from pathlib import Path
import sys

SPACEREC_ROOT = Path.cwd()
if str(SPACEREC_ROOT) not in sys.path:
    sys.path.insert(0, str(SPACEREC_ROOT))

import spacerec.api as spacerec
```

The recommended workflow is the notebook:

```text
spacerec/notebooks/run_spacerec.ipynb
```

It is organized as:

1. Step 1: Deconvolution
2. Step 2: Grid Embedding
3. Step 3: Train
4. Step 4: Aggregate
5. Step 5: Evaluation

The current BRCA notebook is configured for a full run:

```text
Step 2 max_patches = None
Step 3 projection_dim = 512
Step 3 max_epochs = 60
Step 3 batch_size = 4
Step 3 limit_spots = None
Step 3 patience = 20
```

For a smoke test, use a small `max_patches` in Step 2 and a small `limit_spots`
with fewer epochs in Step 3.

## Public API

Import:

```python
import spacerec.api as spacerec
```

Main functions:

```text
spacerec.deconv(...)
spacerec.ge(...)
spacerec.train(...)
spacerec.agg(...)
spacerec.plottype(...)
spacerec.plotexpr(...)
```

Default outputs are under:

```text
results/<dataset>/<step>/
```

Supported datasets in the API are currently `brca` and `crc`, though the
packaged example data in this workspace is BRCA.

## Step 1: Deconvolution

API:

```python
spacerec.deconv(...)
```

Purpose:

Estimate spot-level cell-type proportions from Visium counts and an annotated
single-cell reference. The resulting `deconv.csv` is used as weak supervision
for the model's cell-type head.

Conceptually, for each Visium spot `s`, RCTD approximates the spot expression as
a mixture of reference cell-type profiles:

```text
x_s ~= sum_k p_{s,k} r_k
p_{s,k} >= 0
sum_k p_{s,k} = 1
```

Important inputs:

```text
visium_dir
sc_ref_h5ad
sample_id
annotation_column
type_merge_json
output_dir
max_cores
force
```

For BRCA, the merge file can reduce raw annotation classes to the 11 display and
training classes:

```text
spacerec/deconv/brca_type_merge_17to11.json
data/brca_type_merge_17to11.json
```

Outputs:

```text
results/brca/deconv/deconv.csv
results/brca/deconv/deconv_summary.json
results/brca/deconv/deconv_merged.csv
results/brca/deconv/deconv_merge_summary.json
```

The merged output is row-normalized after summing source columns.

## Step 2: Grid Embedding

API:

```python
spacerec.ge(...)
```

Purpose:

Generate dense grid-level histology features from the H&E image using Virchow2.
Only the dense18 Virchow2 path is retained.

Current fixed feature configuration:

```text
feature_key = virchow2_token_tile_neighbor_concat6400
token feature dimension = 1280
same-patch tile feature dimension = 2560
neighbor tile feature dimension = 2560
final feature dimension = 6400
patch size = 288 full-resolution pixels
model input size = 224 pixels
Virchow2 token layout = 16 x 16
dense grid size = 18 full-resolution pixels
stride = 72 full-resolution pixels
```

Per patch, the image crop is resized from 288 x 288 full-resolution pixels to
224 x 224 model input pixels. Virchow2 produces token-level and tile-level
features. For each dense grid view:

```text
h_{g,v} = [t_{g,v}; u_v; n_v] in R^6400
```

where:

```text
t_{g,v}: token feature for grid g in patch view v, 1280 dims
u_v: same-patch tile feature, 2560 dims
n_v: neighboring-patch tile context, 2560 dims
```

The same final dense grid can be observed from multiple overlapping patches.
Those views are combined after the full 6400-dimensional vector is concatenated:

```text
h_g = sum_v w_{g,v} h_{g,v} / sum_v w_{g,v}
```

The weights `w_{g,v}` are raised-cosine / Hann-like center weights.

Mask and filtering behavior:

1. Read Visium in-tissue positions.
2. Build the full-resolution Visium tissue bounding box.
3. Estimate the tissue boundary from the thumbnail image.
4. Generate sliding-window patch candidates.
5. Keep patches whose centers are inside the Visium bbox and tissue boundary.
6. Split retained patches into dense18 token grids.
7. Filter grid centers by tissue boundary.
8. Deduplicate identical final grid bboxes.
9. Weighted-average duplicate grid views.
10. Keep nearest-spot metadata for supervised training.

Outputs:

```text
results/brca/grid_embedding/grid_embedding.h5
results/brca/grid_embedding/grid_embedding_summary.json
results/brca/grid_embedding/grid_embedding_progress.json
results/brca/grid_embedding/grid_embedding_mask_preview.png
results/brca/grid_embedding/patch_metadata_h5.h5
```

The current full BRCA run reports:

```text
n_export_patches = 24423
n_grids = 391344
n_supervised_grids = 172966
n_supervised_spots = 4740
feature_dim = 6400
complete = True
```

## Step 3: Train

API:

```python
spacerec.train(...)
```

Purpose:

Train the projection-heads SpaceRec model from dense grid embeddings, raw
Visium spot expression, and deconvolution proportions.

The current model intentionally does not use a Set Transformer:

```text
architecture = projection_heads
head_mode = both
use_set_transformer = False
use_gene_head = True
use_type_head = True
```

### Model Architecture

Each grid feature `h_g` is projected first:

```text
z_g = LayerNorm(GELU(W h_g + b))
```

In the notebook, `h_g` is 6400-dimensional and `projection_dim=512`.

The gene head is an MLP:

```text
512 -> 512 -> 512 -> 256 -> n_genes
hidden layers: Linear -> LeakyReLU -> Dropout(0.05)
final activation: Softplus(beta=20)
```

The type head is a residual MLP:

```text
z_res = z_g + block(LayerNorm(z_g))
q_g = Softmax(classifier(z_res) / T)
T = 2.0
```

For the current BRCA run:

```text
input_dim = 6400
projection_dim = 512
n_genes = 4000
n_cell_types = 11
trainable parameters ~= 5.5 M
```

### Spot-Level Supervision

Training samples are spot bags. Each spot `s` owns a set of dense grids `G_s`.
The model predicts at grid level, then aggregates to spot level for supervision:

```text
xhat_s = sum_{g in G_s} xhat_g
phat_s = (1 / |G_s|) sum_{g in G_s} qhat_g
```

Expression loss compares spot-summed grid predictions with raw spot counts,
using `log1p` inside the Huber loss:

```text
L_expr = Huber(log(1 + xhat_s), log(1 + x_s))
```

Type loss compares predicted spot proportions against deconvolution targets:

```text
L_deconv = KL(p_s || phat_s)
L_conf = - mean_g log max_k qhat_{g,k}
L_type = alpha L_conf + (1 - alpha) L_deconv
L = L_expr + lambda_type L_type
```

The notebook uses:

```text
lambda_type = 1.0
alpha = 0.05
lr = 5e-5
max_epochs = 60
batch_size = 4
patience = 20
```

Outputs:

```text
results/brca/train/model/best_train_model.ckpt
results/brca/train/model/last.ckpt
results/brca/train/metadata.json
results/brca/train/summary.json
results/brca/train/slim_train_summary.json
results/brca/train/grid_predictions.h5
results/brca/train/grid_type.csv
results/brca/train/grid_expr.h5ad
results/brca/train/metrics/gene_expression_metrics.csv
results/brca/train/metrics/spot_expression_metrics.csv
results/brca/train/metrics/spot_type_metrics.csv
results/brca/train/metrics/spot_type_probabilities.csv
```

`grid_predictions.h5` contains:

```text
expr_pred: grid-by-gene predicted expression
type_prob: grid-by-cell-type predicted probabilities
type_top1: top predicted type index
center_xy, bbox_xyxy, spot metadata, gene names, cell-type names
```

`grid_type.csv` contains one row per grid, coordinates, `predicted_type`, and
one probability column per cell type.

`grid_expr.h5ad` contains grid-by-gene expression in `X`, grid coordinates in
`.obs`, and gene names in `.var_names`.

## Step 4: Aggregate

API:

```python
spacerec.agg(...)
```

Purpose:

Aggregate grid-level predictions to target cell/polygon regions, for example
Xenium polygons.

For a target polygon or cell `c`, each overlapping grid `g` receives an
area-overlap weight:

```text
a_{c,g} = area(polygon_c intersect grid_g)
```

Expression and type probabilities are area-weighted averages:

```text
xhat_c = sum_g a_{c,g} xhat_g / sum_g a_{c,g}
phat_c = sum_g a_{c,g} qhat_g / sum_g a_{c,g}
```

The exported cell type is the maximum-probability type:

```text
typehat_c = argmax_k phat_{c,k}
```

Important inputs:

```text
grid_expr_h5ad
grid_type_csv
grid_predictions_h5
polygon_csv
cell_metadata_csv
output_dir
valid_only
target_name
```

Outputs:

```text
results/brca/aggregate/spacerec_ct.csv
results/brca/aggregate/spacerec_polygon.csv
results/brca/aggregate/spacerec_expr.h5ad
results/brca/aggregate/aggregate_summary.json
```

The current BRCA aggregate run reports:

```text
n_input_polygons = 139208
n_output_cells = 134364
n_zero_coverage = 4844
n_invalid_polygon = 0
grid_size = 18
valid_only = True
```

## Step 5: Evaluation

APIs:

```python
spacerec.plottype(...)
spacerec.plotexpr(...)
```

Purpose:

Generate side-by-side visual checks for selected full-resolution windows.

Type plotting:

```text
left: true Xenium polygon type image
right: SpaceRec grid type image
```

Expression plotting:

```text
left: true Xenium polygon expression for one gene
right: SpaceRec grid expression for the same gene
```

Current outputs:

```text
results/brca/Evaluation/xen_type.png
results/brca/Evaluation/grid_type.png
results/brca/Evaluation/xen_expr.png
results/brca/Evaluation/grid_expr.png
```

For type plots, `window_label` is used as a display label in the figure title.
True Xenium polygons are selected by the full-resolution `window` coordinates.
For expression plots, `polygon_window_label` is also used as a display label;
polygon selection is coordinate-based.

## Current Example Metrics

The current `results/brca/train/summary.json` and notebook output report:

```text
mean_gene_PCC ~= 0.7751
mean_gene_SCC ~= 0.6722
mean_spot_gene_PCC ~= 0.8607
mean_spot_gene_SCC ~= 0.6065
n_gene_metric_spots = 4740
```

These metrics come from the current packaged BRCA run and should be treated as
run artifacts, not hard-coded expected values.

## Common Checks

Confirm there are no symlinks in the standalone package:

```bash
find /net/dali/home/chikina/shared_data/SpaceRec -type l | wc -l
```

Expected:

```text
0
```

Confirm the package imports from this root:

```bash
cd /net/dali/home/chikina/shared_data/SpaceRec
python -c "import sys; sys.path.insert(0, '.'); import spacerec.api as spacerec; print(spacerec.__file__)"
```

Confirm CUDA visibility before expensive steps:

```bash
hostname
echo "$SLURM_JOB_ID"
echo "$CUDA_VISIBLE_DEVICES"
nvidia-smi
```

## Troubleshooting

If `spacerec.ge()` fails to find a GPU:

1. Make sure the process is running on a compute node inside a Slurm allocation.
2. Check `nvidia-smi`.
3. Pass `device="cuda:0"` only when the visible CUDA device is known.
4. Keep `auto_select_gpu=True` when possible.

If `spacerec.train()` trains on CPU unexpectedly:

1. Check that PyTorch sees CUDA with `torch.cuda.is_available()`.
2. Confirm the notebook kernel is using the intended conda environment.
3. Confirm `CUDA_VISIBLE_DEVICES` is set inside the Slurm job shell.

If evaluation plots are blank:

1. Check that the full-resolution `window` overlaps the generated grid outputs.
2. For true Xenium panels, check that the `window` overlaps the polygon CSV.
3. Re-run Step 2 and Step 3 at full scale if the available grid outputs came
   from a smoke test.
4. `window_label` and `polygon_window_label` are labels only; coordinate windows
   control selection.

## Design Constraints

This slim package intentionally avoids reintroducing:

1. Set Transformer model paths.
2. Dense16 or ResNet embedding paths.
3. Attention-extraction experiment code.
4. Absolute user-home paths in API defaults.
5. Symlinks inside packaged data.
