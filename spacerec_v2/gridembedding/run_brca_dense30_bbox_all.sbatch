#!/bin/bash
#SBATCH --job-name=brca_ge_d30
#SBATCH --partition=camacho_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/net/dali/home/chikina/shared_data/SpaceRec/resources/brca/embedding/slurm_logs/%x_%j.out
#SBATCH --error=/net/dali/home/chikina/shared_data/SpaceRec/resources/brca/embedding/slurm_logs/%x_%j.err

set -euo pipefail

REPO=/net/dali/home/chikina/shared_data/SpaceRec
cd "$REPO"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HDF5_USE_FILE_LOCKING=FALSE

python - <<'PY'
from pathlib import Path
import json
import os

import torch

import spacerec_v2.api as spacerec

root = Path.cwd()
embedding_dir = root / "resources/brca/embedding"
embedding_dir.mkdir(parents=True, exist_ok=True)
output_h5 = embedding_dir / "dense30_virchow2_3840_xen_he_bbox_allgrid.h5"
summary_json = embedding_dir / "dense30_virchow2_3840_xen_he_bbox_allgrid_summary.json"
patch_metadata_h5 = embedding_dir / "patch_metadata_dense30_virchow2_3840_xen_he_bbox_allgrid.h5"
progress_json = embedding_dir / "dense30_virchow2_3840_xen_he_bbox_allgrid_progress.json"

print(json.dumps({
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}, indent=2), flush=True)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available in the Slurm allocation.")

summary = spacerec.ge(
    dataset="brca",
    sample_id="BREAST",
    he_image=root / "resources/brca/xen/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif",
    positions_csv=root / "resources/brca/syn_vis/synthetic_tissue_positions_transcript_filtered.csv",
    scalefactors_json=root / "resources/brca/syn_vis/synthetic_scalefactors_json.json",
    thumbnail_png=root / "resources/brca/syn_vis/visium_like_spots_on_he_thumbnail_transcript.png",
    output_dir=embedding_dir,
    output_h5=output_h5,
    summary_json=summary_json,
    patch_metadata_h5=patch_metadata_h5,
    progress_json=progress_json,
    patch_size=480,
    stride=120,
    concat_local=True,
    concat_nbr=False,
    model="virchow2",
    batch_size=4,
    num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
    force=True,
    auto_select_gpu=False,
    device="cuda",
    enable_progress_bar=False,
    stage_log_stdout=True,
)
print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
PY
