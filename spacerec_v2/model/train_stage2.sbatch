#!/bin/bash
#SBATCH --job-name=brca_stage2
#SBATCH --partition=camacho_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
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

mkdir -p resources/brca/type_pred/slurm_logs

python - <<'PY'
from argparse import Namespace
from pathlib import Path
import json
import os

import torch

from spacerec_v2.model.model import prepare_data
from spacerec_v2.model.training import _ensure_st_h5ad

root = Path.cwd()
train_dir = root / "resources/brca/type_pred/train_v2"
st_h5 = root / "resources/brca/syn_vis/filtered_synthetic_st_filtered_feature_bc_matrix_transcript.h5"
st_h5ad = train_dir / "input/filtered_synthetic_st_filtered_feature_bc_matrix_transcript.h5ad"
deconv_csv = root / "resources/brca/deconv/RCTD_BREAST_SYNTHETIC_TRANSCRIPT_proportions_merged11_spacerec.csv"
grid_embedding_h5 = root / "resources/brca/embedding/grid_embedding_train_filtered.h5"
gene_list = root / "resources/brca/gene_list/gene.txt"
mu_ref = root / "resources/brca/deconv/rctd_ref/rctd_reference_merged11.npy"
stage1_checkpoint = train_dir / "model/stage1_router/checkpoints/best.ckpt"
cell_polygon_csv = root / "resources/brca/xen/aligned/he_alignmented_cell_boundaries.csv"

for path in [st_h5, deconv_csv, grid_embedding_h5, gene_list, mu_ref, stage1_checkpoint, cell_polygon_csv]:
    if not path.exists():
        raise FileNotFoundError(path)

_ensure_st_h5ad(st_h5, st_h5ad, force=False)
preflight_args = Namespace(
    dense_h5=grid_embedding_h5,
    dense_feature_key="virchow2_token_tile_concat3840",
    st_h5ad=st_h5ad,
    deconv_csv=deconv_csv,
    gene_list=gene_list,
    sample_id="BREAST",
    limit_spots=None,
)
data = prepare_data(preflight_args)
summary = {
    "stage": "stage2_preflight_ok",
    "host": os.uname().nodename,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "stage1_checkpoint": str(stage1_checkpoint),
    "st_h5ad": str(st_h5ad),
    "dense_feature_shape": list(data.features.shape),
    "n_position_spot_ids": len(data.position_spot_ids),
    "n_supervised_spots": len(data.spot_ids),
    "n_genes": len(data.gene_names),
    "n_cell_types": len(data.cell_type_names),
    "cell_type_names": list(map(str, data.cell_type_names)),
    "spot_expression_shape": list(data.spot_expression.shape),
    "spot_proportions_shape": list(data.spot_proportions.shape),
    "deconv_row_sum_min": float(data.spot_proportions.sum(axis=1).min()),
    "deconv_row_sum_max": float(data.spot_proportions.sum(axis=1).max()),
    "zero_target_rows": int((data.spot_proportions.sum(axis=1) == 0).sum()),
}
print(json.dumps(summary, indent=2), flush=True)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available in the Slurm allocation.")
if summary["n_position_spot_ids"] != 4816:
    raise RuntimeError(f"Unexpected synthetic spot count: {summary['n_position_spot_ids']}")
if summary["n_supervised_spots"] <= 0:
    raise RuntimeError("No supervised spots passed Stage 2 preflight.")
if summary["n_genes"] != 306 or summary["n_cell_types"] != 11:
    raise RuntimeError("Unexpected Stage 2 gene/type dimensions.")
if summary["zero_target_rows"] != 0:
    raise RuntimeError("Stage 2 preflight found zero deconvolution target rows.")
PY

python - <<'PY'
from pathlib import Path
import json

import spacerec_v2.api as spacerec

root = Path.cwd()
train_dir = root / "resources/brca/type_pred/train_v2"
type_pred_dir = root / "resources/brca/type_pred"
summary = spacerec.stage2(
    dataset="brca",
    st_h5ad=train_dir / "input/filtered_synthetic_st_filtered_feature_bc_matrix_transcript.h5ad",
    deconv_csv=root / "resources/brca/deconv/RCTD_BREAST_SYNTHETIC_TRANSCRIPT_proportions_merged11_spacerec.csv",
    grid_embedding_h5=root / "resources/brca/embedding/grid_embedding_train_filtered.h5",
    mu_ref=root / "resources/brca/deconv/rctd_ref/rctd_reference_merged11.npy",
    run_dir=train_dir,
    stage1_checkpoint=train_dir / "model/stage1_router/checkpoints/best.ckpt",
    gene_txt=root / "resources/brca/gene_list/gene.txt",
    projection_dim=2048,
    skip_input_projector=True,
    type_head_hidden_layers=2,
    factorized_head_hidden_layers=3,
    scale_head_hidden_dim=512,
    delta_head_hidden_dim=2048,
    stage2_epochs=70,
    batch_size=4,
    lr=5e-5,
    alpha=0.1,
    delta_alpha=0.5,
    limit_spots=None,
    auto_select_gpu=False,
    enable_progress_bar=False,
    progress_refresh_rate=1,
    stage_log_stdout=True,
    cell_polygon_csv=root / "resources/brca/xen/aligned/he_alignmented_cell_boundaries.csv",
    grid_pred_expression_h5=train_dir / "grid_pred_expression.h5",
    cell_pred_expression_h5=train_dir / "cell_pred_expression.h5",
)

required = [
    train_dir / "grid_predictions.h5",
    train_dir / "grid_pred_expression.h5",
    train_dir / "cell_pred_expression.h5",
    train_dir / "grid_type.csv",
    train_dir / "grid_expr.h5ad",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError("Missing expected Stage 2 outputs: " + ", ".join(missing))

print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
PY
