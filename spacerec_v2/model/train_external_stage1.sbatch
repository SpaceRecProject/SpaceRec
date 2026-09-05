#!/bin/bash
#SBATCH --job-name=brca_external_stage1
#SBATCH --partition=camacho_gpu
#SBATCH --nodelist=g141
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output=/net/dali/home/chikina/shared_data/SpaceRec/resources/brca/type_pred/slurm_logs/%x_%j.out
#SBATCH --error=/net/dali/home/chikina/shared_data/SpaceRec/resources/brca/type_pred/slurm_logs/%x_%j.err

set -euo pipefail

REPO=/net/dali/home/chikina/shared_data/SpaceRec
cd "$REPO"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

python - <<'PY'
from pathlib import Path
from argparse import Namespace
import json

from spacerec_v2.model.model import prepare_data
from spacerec_v2.model.training import _ensure_st_h5ad

root = Path.cwd()
type_pred_dir = root / "resources/brca/type_pred"
train_dir = type_pred_dir / "external_train_v2"
st_h5 = root / "resources/brca/syn_vis/filtered_synthetic_st_filtered_feature_bc_matrix_transcript.h5"
st_h5ad = train_dir / "input/filtered_synthetic_st_filtered_feature_bc_matrix_transcript.h5ad"
deconv_csv = root / "resources/brca/deconv/external_RCTD_BREAST_SYNTHETIC_TRANSCRIPT_proportions_spacerec.csv"
grid_embedding_h5 = root / "resources/brca/embedding/grid_embedding_train_filtered.h5"
gene_list = root / "resources/brca/gene_list/gene.txt"

_ensure_st_h5ad(st_h5, st_h5ad, force=True)
preflight_args = Namespace(
    dense_h5=grid_embedding_h5,
    dense_feature_key="virchow2_token_tile_concat3840",
    st_h5ad=st_h5ad,
    deconv_csv=deconv_csv,
    gene_list=gene_list,
    sample_id="BREAST",
    limit_spots=None,
)
preflight_data = prepare_data(preflight_args)
preflight_summary = {
    "stage": "external_stage1_preflight_ok",
    "st_h5ad": str(st_h5ad),
    "dense_feature_shape": list(preflight_data.features.shape),
    "n_position_spot_ids": len(preflight_data.position_spot_ids),
    "n_supervised_spots": len(preflight_data.spot_ids),
    "n_genes": len(preflight_data.gene_names),
    "n_cell_types": len(preflight_data.cell_type_names),
    "cell_type_names": list(map(str, preflight_data.cell_type_names)),
    "spot_expression_shape": list(preflight_data.spot_expression.shape),
    "spot_proportions_shape": list(preflight_data.spot_proportions.shape),
    "deconv_row_sum_min": float(preflight_data.spot_proportions.sum(axis=1).min()),
    "deconv_row_sum_max": float(preflight_data.spot_proportions.sum(axis=1).max()),
    "zero_target_rows": int((preflight_data.spot_proportions.sum(axis=1) == 0).sum()),
}
print(json.dumps(preflight_summary, indent=2), flush=True)
if preflight_summary["n_position_spot_ids"] != 4816:
    raise RuntimeError(f"Unexpected synthetic spot count: {preflight_summary['n_position_spot_ids']}")
if preflight_summary["n_supervised_spots"] <= 0:
    raise RuntimeError("No supervised spots passed external Stage 1 preflight.")
if preflight_summary["n_genes"] != 306 or preflight_summary["n_cell_types"] != 9:
    raise RuntimeError("Unexpected external Stage 1 gene/type dimensions.")
if preflight_summary["zero_target_rows"] != 0:
    raise RuntimeError("External Stage 1 preflight found zero deconvolution target rows.")
PY

python - <<'PY'
from pathlib import Path
import json
import os

import torch

import spacerec_v2.api as spacerec

root = Path.cwd()
type_pred_dir = root / "resources/brca/type_pred"
train_dir = type_pred_dir / "external_train_v2"
st_h5ad = train_dir / "input/filtered_synthetic_st_filtered_feature_bc_matrix_transcript.h5ad"
deconv_csv = root / "resources/brca/deconv/external_RCTD_BREAST_SYNTHETIC_TRANSCRIPT_proportions_spacerec.csv"
grid_embedding_h5 = root / "resources/brca/embedding/grid_embedding_train_filtered.h5"
gene_list = root / "resources/brca/gene_list/gene.txt"
mu_ref = root / "resources/brca/deconv/external_rctd_ref/external_rctd_reference_wu9_placeholder.npy"
cell_polygon_csv = root / "resources/brca/xen/aligned/he_alignmented_cell_boundaries.csv"

cuda_summary = {
    "host": os.uname().nodename,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}
print(json.dumps(cuda_summary, indent=2), flush=True)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available in the Slurm allocation.")

summary = spacerec.stage1(
    dataset="brca",
    st_h5ad=st_h5ad,
    deconv_csv=deconv_csv,
    grid_embedding_h5=grid_embedding_h5,
    gene_list=gene_list,
    mu_ref=mu_ref,
    run_dir=train_dir,
    projection_dim=2048,
    skip_input_projector=True,
    type_head_hidden_layers=2,
    factorized_head_hidden_layers=3,
    scale_head_hidden_dim=512,
    delta_head_hidden_dim=2048,
    stage1_epochs=70,
    batch_size=4,
    lr=5e-5,
    alpha=0.1,
    delta_alpha=0.5,
    limit_spots=None,
    auto_select_gpu=False,
    enable_progress_bar=False,
    progress_refresh_rate=1,
    stage_log_stdout=True,
    agg=True,
    cell_polygon_csv=cell_polygon_csv,
    type_pred_dir=type_pred_dir,
    grid_type_csv=type_pred_dir / "external_stage1_grid_type.csv",
    cell_type_csv=type_pred_dir / "external_stage1_cell_type.csv",
)

required = [
    Path(summary["stage1_checkpoint"]),
    type_pred_dir / "external_stage1_grid_type.csv",
    type_pred_dir / "external_stage1_cell_type.csv",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError("Missing expected external Stage 1 outputs: " + ", ".join(missing))

print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
PY
