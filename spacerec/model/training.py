from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from .model import run_model


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item) for item in values],
        dtype=object,
    )


def _read_10x_h5(path: Path) -> AnnData:
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        shape = tuple(int(x) for x in matrix["shape"][()])
        counts_gene_by_spot = sparse.csc_matrix(
            (matrix["data"][()], matrix["indices"][()], matrix["indptr"][()]),
            shape=shape,
        )
        barcodes = _decode(matrix["barcodes"][()])
        features = matrix["features"]
        gene_ids = _decode(features["id"][()])
        gene_names = _decode(features["name"][()])
        feature_types = _decode(features["feature_type"][()])

    keep = feature_types == "Gene Expression"
    counts = counts_gene_by_spot[keep, :].T.tocsr()
    var = pd.DataFrame(
        {"gene_id": gene_ids[keep], "feature_type": feature_types[keep]},
        index=pd.Index(gene_names[keep].astype(str), name="gene"),
    )
    obs = pd.DataFrame(index=pd.Index(barcodes.astype(str), name="barcode"))
    adata = AnnData(X=counts, obs=obs, var=var)
    adata.var_names_make_unique()
    return adata


def _ensure_st_h5ad(visium_h5: Path, output_h5ad: Path, force: bool = False) -> Path:
    if output_h5ad.exists() and not force:
        return output_h5ad
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata = _read_10x_h5(visium_h5)
    adata.write_h5ad(output_h5ad)
    return output_h5ad


def _default_projection_dim(dataset: str) -> int:
    return {"brca": 3584, "crc": 3584}[dataset]


def _infer_dense_feature_key(path: Path) -> str:
    with h5py.File(path, "r") as handle:
        attr = handle.attrs.get("feature_key")
        if attr is not None:
            return attr.decode("utf-8") if isinstance(attr, bytes) else str(attr)
        candidates = [
            key
            for key, value in handle.items()
            if getattr(value, "ndim", None) == 2
            and key not in {"bbox_xyxy", "center_xy", "grid_bbox", "grid_center_xy", "spot_xy"}
        ]
    if len(candidates) == 1:
        return candidates[0]
    raise KeyError(f"Cannot infer feature dataset key from {path}; candidates={candidates}")


def _default_dense_h5(dataset: str, results_dir: Path) -> Path:
    return {
        "brca": results_dir / "brca" / "gridembedding" / "dense18_virchow2_brca.h5",
        "crc": results_dir / "crc" / "gridembedding" / "dense18_virchow2_crc.h5",
    }[dataset]


def _default_deconv_csv(dataset: str, results_dir: Path) -> Path:
    return {
        "brca": results_dir / "brca" / "deconv" / "rctd" / "RCTD_BREAST_proportions_merged11_spacerec.csv",
        "crc": results_dir / "crc" / "deconv" / "rctd" / "RCTD_COLON_ALLCRC_proportions_spacerec.csv",
    }[dataset]


def _default_sample_id(dataset: str) -> str:
    return {"brca": "BREAST", "crc": "COLON_P2"}[dataset]


def _default_run_dir(dataset: str, projection_dim: int, results_dir: Path) -> Path:
    encoder = {"brca": "dense18_virchow2", "crc": "dense18_virchow2"}[dataset]
    return results_dir / dataset / "model" / f"{encoder}_{dataset}_proj{int(projection_dim)}"


def run_training(
    *,
    dataset: str,
    package_root: str | Path | None = None,
    resources_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    dense_h5: str | Path | None = None,
    dense_feature_key: str | None = None,
    st_h5ad: str | Path | None = None,
    deconv_csv: str | Path | None = None,
    gene_list: str | Path | None = None,
    run_dir: str | Path | None = None,
    projection_dim: int | None = None,
    max_epochs: int = 120,
    batch_size: int = 4,
    lr: float = 5e-5,
    lambda_deconv: float = 1.0,
    type_confidence_alpha: float = 0.05,
    gene_loss_reduction: str = "sum",
    val_fraction: float = 0.1,
    train_all: bool = True,
    patience: int = 20,
    early_stop_min_delta: float = 1e-4,
    num_workers: int = 0,
    seed: int = 0,
    limit_spots: int | None = None,
    export_max_grids_per_batch: int = 8192,
    no_export: bool = False,
    force_prepare_st: bool = False,
    eager_cuda_init: bool = True,
    cuda_device: int | None = None,
    gpu_selection: dict[str, Any] | None = None,
    enable_progress_bar: bool = True,
    progress_refresh_rate: int = 1,
    stage_log_stdout: bool = False,
    **extra: Any,
) -> dict[str, float | int | str]:
    selected_dataset = dataset.lower()
    if selected_dataset not in {"brca", "crc"}:
        raise ValueError(f"Unsupported dataset: {dataset!r}.")

    selected_package_root = Path(package_root) if package_root is not None else PACKAGE_ROOT
    selected_resources = Path(resources_dir) if resources_dir is not None else selected_package_root / "resources"
    selected_results = Path(results_dir) if results_dir is not None else selected_package_root / "results"
    selected_projection_dim = int(projection_dim) if projection_dim is not None else _default_projection_dim(selected_dataset)

    model_resource_dir = selected_resources / selected_dataset / "model"
    model_input_dir = selected_results / selected_dataset / "model" / "input"
    model_input_dir.mkdir(parents=True, exist_ok=True)

    selected_st_h5ad = Path(st_h5ad) if st_h5ad is not None else model_input_dir / "st.h5ad"
    if st_h5ad is None:
        selected_st_h5ad = _ensure_st_h5ad(
            selected_resources / selected_dataset / "visium" / "st_filtered_feature_bc_matrix.h5",
            selected_st_h5ad,
            force=force_prepare_st,
        )

    args = Namespace(
        dense_h5=Path(dense_h5) if dense_h5 is not None else _default_dense_h5(selected_dataset, selected_results),
        dense_feature_key=dense_feature_key,
        st_h5ad=selected_st_h5ad,
        deconv_csv=Path(deconv_csv) if deconv_csv is not None else _default_deconv_csv(selected_dataset, selected_results),
        gene_list=Path(gene_list) if gene_list is not None else model_resource_dir / "genes.txt",
        run_dir=Path(run_dir) if run_dir is not None else _default_run_dir(selected_dataset, selected_projection_dim, selected_results),
        sample_id=_default_sample_id(selected_dataset),
        projection_dim=selected_projection_dim,
        max_epochs=max_epochs,
        patience=patience,
        early_stop_min_delta=early_stop_min_delta,
        batch_size=batch_size,
        lr=lr,
        lambda_deconv=lambda_deconv,
        type_confidence_alpha=type_confidence_alpha,
        gene_loss_reduction=gene_loss_reduction,
        loss="log1p_huber",
        val_fraction=val_fraction,
        train_all=train_all,
        num_workers=num_workers,
        seed=seed,
        limit_spots=limit_spots,
        export_max_grids_per_batch=export_max_grids_per_batch,
        no_export=no_export,
        eager_cuda_init=eager_cuda_init,
        cuda_device=cuda_device,
        gpu_selection=gpu_selection,
        enable_progress_bar=enable_progress_bar,
        progress_refresh_rate=progress_refresh_rate,
        stage_log_stdout=stage_log_stdout,
    )
    for key, value in extra.items():
        setattr(args, key, value)

    missing = [str(path) for path in [args.dense_h5, args.st_h5ad, args.deconv_csv, args.gene_list] if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing model training inputs: " + ", ".join(missing))
    if args.dense_feature_key is None:
        args.dense_feature_key = _infer_dense_feature_key(args.dense_h5)

    summary = {
        "dataset": selected_dataset,
        "projection_dim": selected_projection_dim,
        "dense_h5": str(args.dense_h5),
        "dense_feature_key": str(args.dense_feature_key),
        "st_h5ad": str(args.st_h5ad),
        "deconv_csv": str(args.deconv_csv),
        "gene_list": str(args.gene_list),
        "run_dir": str(args.run_dir),
        "architecture": "projection_heads",
        "head_mode": "both",
        "use_set_transformer": False,
        "use_gene_head": True,
        "use_type_head": True,
        "expression_target": "raw_counts",
        "expression_loss": "huber_log1p_sum_grid_vs_raw_counts",
        "gpu_selection": gpu_selection,
        "enable_progress_bar": bool(enable_progress_bar),
        "progress_refresh_rate": int(progress_refresh_rate),
        "stage_log_stdout": bool(stage_log_stdout),
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "input_paths.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_model(args)
