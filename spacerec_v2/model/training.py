from __future__ import annotations

import json
import math
from argparse import Namespace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
import torch
from torch.utils.data import DataLoader

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from .factorized import (
    FinetuneMuLightning,
    export_finetune_mu_grid_predictions,
    export_finetune_mu_stage1_type_predictions,
)
from .model import (
    DenseGridSpotDataset,
    collate_spot_bags,
    compute_spot_expression_metrics,
    compute_spot_type_metrics,
    default_device,
    prepare_data,
    predict_supervised,
    run_model,
    set_seed,
    write_string_dataset,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


class _StageTQDMProgressBar(TQDMProgressBar):
    def __init__(self, *, stage_name: str, refresh_rate: int) -> None:
        super().__init__(refresh_rate=refresh_rate)
        self.stage_name = str(stage_name)

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.set_description(self.stage_name)
        return bar


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
        "brca": results_dir / "brca" / "grid_embedding" / "grid_embedding.h5",
        "crc": results_dir / "crc" / "grid_embedding" / "grid_embedding.h5",
    }[dataset]


def _default_deconv_csv(dataset: str, results_dir: Path) -> Path:
    return {
        "brca": results_dir / "brca" / "deconv" / "deconv.csv",
        "crc": results_dir / "crc" / "deconv" / "deconv.csv",
    }[dataset]


def _default_sample_id(dataset: str) -> str:
    return {"brca": "BREAST", "crc": "COLON_P2"}[dataset]


def _default_run_dir(dataset: str, projection_dim: int, results_dir: Path) -> Path:
    encoder = {"brca": "dense18_virchow2", "crc": "dense18_virchow2"}[dataset]
    return results_dir / dataset / "model" / f"{encoder}_{dataset}_proj{int(projection_dim)}"


def _default_finetune_mu_run_dir(dataset: str, results_dir: Path) -> Path:
    return results_dir / dataset / "train_v2"


def _default_gene_list(dataset: str, package_root: Path) -> Path:
    if dataset != "brca":
        raise ValueError("Default finetune-mu gene list is currently available for dataset='brca' only.")
    return package_root / "data" / "genes.txt"


def _default_mu_ref(dataset: str, package_root: Path) -> Path:
    if dataset != "brca":
        raise ValueError("Default finetune-mu reference is currently available for dataset='brca' only.")
    return package_root / "data" / "brca_rctd_reference_merged11.npy"


def _load_mu_ref(path: str | Path, n_cell_types: int, n_genes: int) -> np.ndarray:
    selected = Path(path)
    if not selected.exists():
        raise FileNotFoundError(selected)
    mu_ref = np.load(selected).astype(np.float32, copy=False)
    if tuple(mu_ref.shape) != (int(n_cell_types), int(n_genes)):
        raise ValueError(f"mu_ref must have shape [{n_cell_types}, {n_genes}], got {tuple(mu_ref.shape)} from {selected}.")
    if np.any(mu_ref < 0):
        raise ValueError(f"mu_ref must be non-negative: {selected}")
    row_sum = mu_ref.sum(axis=1, keepdims=True)
    if not np.allclose(row_sum, 1.0, atol=1e-5):
        raise ValueError(f"mu_ref rows must sum to 1. min={float(row_sum.min())}, max={float(row_sum.max())}")
    return mu_ref


def _train_finetune_stage(
    model: FinetuneMuLightning,
    loader: DataLoader,
    stage_dir: Path,
    max_epochs: int,
    *,
    stage_name: str,
    cuda_device: int | None,
    enable_progress_bar: bool,
    progress_refresh_rate: int,
) -> Path:
    stage_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=stage_dir / "checkpoints",
        filename="best",
        monitor="train_loss_epoch",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    callbacks = [checkpoint]
    if enable_progress_bar:
        callbacks.append(
            _StageTQDMProgressBar(
                stage_name=stage_name,
                refresh_rate=max(1, int(progress_refresh_rate)),
            )
        )
    trainer = pl.Trainer(
        max_epochs=int(max_epochs),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[int(cuda_device)] if torch.cuda.is_available() and cuda_device is not None else 1,
        callbacks=callbacks,
        default_root_dir=stage_dir,
        log_every_n_steps=10,
        enable_progress_bar=bool(enable_progress_bar),
    )
    trainer.fit(model, loader)
    if checkpoint.best_model_path:
        return Path(checkpoint.best_model_path)
    return stage_dir / "checkpoints" / "last.ckpt"


def _build_finetune_model(
    data,
    mu_ref: np.ndarray,
    *,
    projection_dim: int,
    lr: float,
    lambda_conf: float,
    gene_loss_reduction: str,
    expr_loss: str,
    p_temperature: float,
    init_scale: float,
    delta_alpha: float,
    skip_input_projector: bool,
    type_head_hidden_layers: int,
    factorized_head_hidden_layers: int,
    scale_head_hidden_dim: int | None,
    delta_head_hidden_dim: int | None,
    mu_eps: float,
    stage: str,
) -> FinetuneMuLightning:
    model = FinetuneMuLightning(
        input_dim=data.input_dim,
        n_genes=len(data.gene_names),
        n_cell_types=len(data.cell_type_names),
        projection_dim=int(projection_dim),
        mu_ref=mu_ref,
        lr=float(lr),
        lambda_conf=float(lambda_conf),
        gene_loss_reduction=str(gene_loss_reduction),
        expr_loss=str(expr_loss),
        p_temperature=float(p_temperature),
        init_scale=float(init_scale),
        delta_alpha=float(delta_alpha),
        skip_input_projector=bool(skip_input_projector),
        type_head_hidden_layers=int(type_head_hidden_layers),
        factorized_head_hidden_layers=int(factorized_head_hidden_layers),
        scale_head_hidden_dim=None if scale_head_hidden_dim is None else int(scale_head_hidden_dim),
        delta_head_hidden_dim=None if delta_head_hidden_dim is None else int(delta_head_hidden_dim),
        mu_eps=float(mu_eps),
        stage=stage,
    )
    return model


def _load_finetune_checkpoint(model: FinetuneMuLightning, checkpoint: Path, stage: str) -> FinetuneMuLightning:
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model.load_state_dict(state_dict, strict=True)
    model.set_stage(stage)
    return model


def _polygon_id_column(table: pd.DataFrame) -> str:
    for name in ["cell_id", "cellvit_cell_id", "xen_cell_id"]:
        if name in table.columns:
            return name
    raise ValueError("polygon_csv must contain one of: cell_id, cellvit_cell_id, xen_cell_id.")


def _polygon_xy_columns(table: pd.DataFrame) -> tuple[str, str]:
    for x_col, y_col in [("he_x", "he_y"), ("vertex_x", "vertex_y"), ("vertice_x", "vertice_y")]:
        if x_col in table.columns and y_col in table.columns:
            return x_col, y_col
    raise ValueError("polygon_csv must contain HE coordinate columns such as he_x/he_y or vertex_x/vertex_y.")


def _grid_lookup(bbox: np.ndarray) -> dict[tuple[int, int], int]:
    return {(int(row[0]), int(row[1])): int(index) for index, row in enumerate(bbox)}


def _candidate_grid_indices(bounds, lookup: dict[tuple[int, int], int], grid_size: int) -> list[int]:
    minx, miny, maxx, maxy = bounds
    start_x = math.floor(minx / grid_size) * grid_size
    stop_x = math.ceil(maxx / grid_size) * grid_size
    start_y = math.floor(miny / grid_size) * grid_size
    stop_y = math.ceil(maxy / grid_size) * grid_size
    out: list[int] = []
    for y0 in range(int(start_y), int(stop_y) + grid_size, grid_size):
        for x0 in range(int(start_x), int(stop_x) + grid_size, grid_size):
            index = lookup.get((x0, y0))
            if index is not None:
                out.append(index)
    return out


def _aggregate_stage1_type_to_cells(
    *,
    grid_type_csv: Path,
    polygon_csv: Path,
    output_csv: Path,
) -> dict[str, Any]:
    from shapely.geometry import Polygon, box

    grid_type = pd.read_csv(grid_type_csv)
    polygons = pd.read_csv(polygon_csv)
    id_col = _polygon_id_column(polygons)
    x_col, y_col = _polygon_xy_columns(polygons)
    prob_cols = [col for col in grid_type.columns if str(col).startswith("prob_")]
    if not prob_cols:
        raise ValueError(f"{grid_type_csv} contains no prob_* columns.")
    cell_type_names = [str(col)[5:] for col in prob_cols]
    bbox = grid_type[["bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"]].to_numpy(dtype=np.float64)
    type_matrix = grid_type.loc[:, prob_cols].to_numpy(dtype=np.float32)
    grid_size = int(round(float(np.median(bbox[:, 2] - bbox[:, 0]))))
    grid_area = float(grid_size * grid_size)
    lookup = _grid_lookup(bbox)

    rows: list[dict[str, Any]] = []
    invalid_polygon = 0
    zero_coverage = 0
    output_cells = 0
    for raw_id, group in polygons.groupby(id_col, sort=False):
        cell_id = str(raw_id)
        coords = group[[x_col, y_col]].to_numpy(dtype=np.float64)
        if coords.shape[0] < 3:
            invalid_polygon += 1
            continue
        polygon = Polygon(coords)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or polygon.area <= 0:
            invalid_polygon += 1
            continue
        indices: list[int] = []
        weights: list[float] = []
        for index in _candidate_grid_indices(polygon.bounds, lookup, grid_size):
            x0, y0, x1, y1 = bbox[index]
            area = float(polygon.intersection(box(x0, y0, x1, y1)).area)
            if area > 0.0:
                indices.append(index)
                weights.append(area)
        if not indices:
            zero_coverage += 1
            continue
        weight_array = np.asarray(weights, dtype=np.float64)
        weighted = (weight_array[:, None] * type_matrix[np.asarray(indices, dtype=np.int64)]).sum(axis=0)
        total = float(weighted.sum())
        if total <= 0.0:
            zero_coverage += 1
            continue
        probs = (weighted / total).astype(np.float32)
        top1_index = int(probs.argmax())
        base = {
            "cell_id": cell_id,
            "top1_type": cell_type_names[top1_index],
            "top1_prob": float(probs[top1_index]),
        }
        base.update({f"prob_{name}": float(probs[index]) for index, name in enumerate(cell_type_names)})
        for _, vertex in group.iterrows():
            row = dict(base)
            row["vertice_x"] = float(vertex[x_col])
            row["vertice_y"] = float(vertex[y_col])
            rows.append(row)
        output_cells += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return {
        "stage1_cell_type_csv": str(output_csv),
        "stage1_cell_polygon_csv": str(polygon_csv),
        "n_stage1_input_cells": int(polygons[id_col].nunique()),
        "n_stage1_output_cells": int(output_cells),
        "n_stage1_cell_vertices": int(len(rows)),
        "n_stage1_invalid_polygons": int(invalid_polygon),
        "n_stage1_zero_coverage_cells": int(zero_coverage),
        "stage1_cell_grid_size": int(grid_size),
    }


def _copy_stage2_grid_expression_h5(grid_predictions_h5: Path, output_h5: Path) -> dict[str, Any]:
    keys = [
        "expr_pred",
        "gene_name",
        "center_xy",
        "bbox_xyxy",
        "is_tissue",
        "grid_tissue_fraction",
        "spot_index",
        "nearest_spot_index",
        "nearest_spot_distance",
        "position_spot_id",
        "position_spot_barcode",
        "type_prob",
        "type_top1",
        "cell_type_name",
    ]
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(grid_predictions_h5, "r") as source, h5py.File(output_h5, "w") as target:
        for key in keys:
            if key in source:
                source.copy(key, target, name=key)
        for key, value in source.attrs.items():
            target.attrs[key] = value
        target.attrs["source_grid_predictions_h5"] = str(grid_predictions_h5)
        target.attrs["expression_source"] = "stage2_grid_expr_pred"
        n_grid, n_gene = source["expr_pred"].shape
    return {
        "grid_pred_expression_h5": str(output_h5),
        "n_stage2_expression_grids": int(n_grid),
        "n_stage2_expression_genes": int(n_gene),
    }


def _aggregate_stage2_expression_to_cells(
    *,
    grid_predictions_h5: Path,
    polygon_csv: Path,
    output_h5: Path,
) -> dict[str, Any]:
    from shapely.geometry import Polygon, box

    polygons = pd.read_csv(polygon_csv)
    id_col = _polygon_id_column(polygons)
    x_col, y_col = _polygon_xy_columns(polygons)

    with h5py.File(grid_predictions_h5, "r") as source:
        bbox = np.asarray(source["bbox_xyxy"], dtype=np.float64)
        expr_ds = source["expr_pred"]
        type_ds = source["type_prob"]
        gene_names = _decode(source["gene_name"][:])
        cell_type_names = _decode(source["cell_type_name"][:])
        grid_size = int(round(float(np.median(bbox[:, 2] - bbox[:, 0]))))
        lookup = _grid_lookup(bbox)

        cell_ids: list[str] = []
        cell_bbox: list[tuple[float, float, float, float]] = []
        centroids: list[tuple[float, float]] = []
        coverage_area: list[float] = []
        covered_fraction: list[float] = []
        n_overlapping_grids: list[int] = []
        expr_rows: list[np.ndarray] = []
        type_rows: list[np.ndarray] = []
        invalid_polygon = 0
        zero_coverage = 0

        for raw_id, group in polygons.groupby(id_col, sort=False):
            coords = group[[x_col, y_col]].to_numpy(dtype=np.float64)
            if coords.shape[0] < 3:
                invalid_polygon += 1
                continue
            polygon = Polygon(coords)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if polygon.is_empty or polygon.area <= 0:
                invalid_polygon += 1
                continue

            indices: list[int] = []
            weights: list[float] = []
            for index in _candidate_grid_indices(polygon.bounds, lookup, grid_size):
                x0, y0, x1, y1 = bbox[index]
                area = float(polygon.intersection(box(x0, y0, x1, y1)).area)
                if area > 0.0:
                    indices.append(index)
                    weights.append(area)
            if not indices:
                zero_coverage += 1
                continue

            index_array = np.asarray(indices, dtype=np.int64)
            order = np.argsort(index_array, kind="stable")
            sorted_indices = index_array[order]
            sorted_weights = np.asarray(weights, dtype=np.float32)[order]
            weight_sum = float(sorted_weights.sum())
            if weight_sum <= 0.0:
                zero_coverage += 1
                continue

            expr_values = np.asarray(expr_ds[sorted_indices], dtype=np.float32)
            type_values = np.asarray(type_ds[sorted_indices], dtype=np.float32)
            weights_2d = sorted_weights[:, None]
            grid_area = float(grid_size * grid_size)
            expr_rows.append(((weights_2d / grid_area) * expr_values).sum(axis=0))
            type_rows.append((weights_2d * type_values).sum(axis=0) / weight_sum)
            minx, miny, maxx, maxy = polygon.bounds
            centroid = polygon.centroid
            cell_ids.append(str(raw_id))
            cell_bbox.append((float(minx), float(miny), float(maxx), float(maxy)))
            centroids.append((float(centroid.x), float(centroid.y)))
            coverage_area.append(weight_sum)
            covered_fraction.append(float(weight_sum / polygon.area))
            n_overlapping_grids.append(int(len(indices)))

    n_cells = len(cell_ids)
    n_genes = int(len(gene_names))
    n_cell_types = int(len(cell_type_names))
    expr = np.vstack(expr_rows).astype(np.float32, copy=False) if expr_rows else np.empty((0, n_genes), dtype=np.float32)
    type_prob = (
        np.vstack(type_rows).astype(np.float32, copy=False)
        if type_rows
        else np.empty((0, n_cell_types), dtype=np.float32)
    )
    type_top1 = type_prob.argmax(axis=1).astype(np.int16) if n_cells else np.empty((0,), dtype=np.int16)

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as target:
        target.create_dataset("expr_pred", data=expr, compression="lzf")
        target.create_dataset("type_prob", data=type_prob, compression="gzip")
        target.create_dataset("type_top1", data=type_top1, compression="gzip")
        write_string_dataset(target, "cell_id", cell_ids)
        write_string_dataset(target, "gene_name", gene_names)
        write_string_dataset(target, "cell_type_name", cell_type_names)
        target.create_dataset("bbox_xyxy", data=np.asarray(cell_bbox, dtype=np.float32), compression="gzip")
        target.create_dataset("centroid_xy", data=np.asarray(centroids, dtype=np.float32), compression="gzip")
        target.create_dataset("coverage_area", data=np.asarray(coverage_area, dtype=np.float32), compression="gzip")
        target.create_dataset("covered_fraction", data=np.asarray(covered_fraction, dtype=np.float32), compression="gzip")
        target.create_dataset("n_overlapping_grids", data=np.asarray(n_overlapping_grids, dtype=np.int32), compression="gzip")
        target.attrs["source_grid_predictions_h5"] = str(grid_predictions_h5)
        target.attrs["source_polygon_csv"] = str(polygon_csv)
        target.attrs["aggregation_method"] = "polygon_grid_intersection_area_weighted_sum"
        target.attrs["expression_source"] = "stage2_grid_expr_pred_area_weighted_sum"
        target.attrs["grid_area"] = float(grid_size * grid_size)
        target.attrs["n_zero_coverage_cells"] = int(zero_coverage)
        target.attrs["n_invalid_polygon_cells"] = int(invalid_polygon)

    return {
        "cell_pred_expression_h5": str(output_h5),
        "stage2_cell_polygon_csv": str(polygon_csv),
        "n_stage2_input_cells": int(polygons[id_col].nunique()),
        "n_stage2_expression_cells": int(n_cells),
        "n_stage2_invalid_polygons": int(invalid_polygon),
        "n_stage2_zero_coverage_cells": int(zero_coverage),
        "stage2_cell_grid_size": int(grid_size),
        "stage2_cell_expression_aggregation": "polygon_grid_intersection_area_weighted_sum",
    }


def export_stage2_expression_h5s(
    *,
    grid_predictions_h5: Path,
    polygon_csv: Path,
    grid_output_h5: Path,
    cell_output_h5: Path,
) -> dict[str, Any]:
    if not grid_predictions_h5.exists():
        raise FileNotFoundError(grid_predictions_h5)
    if not polygon_csv.exists():
        raise FileNotFoundError(polygon_csv)
    summary = _copy_stage2_grid_expression_h5(grid_predictions_h5, grid_output_h5)
    summary.update(
        _aggregate_stage2_expression_to_cells(
            grid_predictions_h5=grid_predictions_h5,
            polygon_csv=polygon_csv,
            output_h5=cell_output_h5,
        )
    )
    return summary


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


def run_finetune_mu_training(
    *,
    dataset: str,
    package_root: str | Path | None = None,
    results_dir: str | Path | None = None,
    dense_h5: str | Path | None = None,
    dense_feature_key: str | None = None,
    st_h5ad: str | Path | None = None,
    deconv_csv: str | Path | None = None,
    gene_list: str | Path | None = None,
    mu_ref: str | Path | None = None,
    run_dir: str | Path | None = None,
    projection_dim: int = 2048,
    stage1_epochs: int = 70,
    stage2_epochs: int = 70,
    stage1_checkpoint: str | Path | None = None,
    batch_size: int = 4,
    lr: float = 5e-5,
    lambda_conf: float = 0.1,
    gene_loss_reduction: str = "sum",
    expr_loss: str = "log1p_huber",
    p_temperature: float = 1.0,
    init_scale: float = 128.0,
    delta_alpha: float = 0.5,
    skip_input_projector: bool = True,
    type_head_hidden_layers: int = 2,
    factorized_head_hidden_layers: int = 3,
    scale_head_hidden_dim: int | None = 512,
    delta_head_hidden_dim: int | None = 2048,
    mu_eps: float = 1e-8,
    seed: int = 0,
    limit_spots: int | None = None,
    export_max_grids_per_batch: int = 2048,
    no_export: bool = False,
    cuda_device: int | None = None,
    gpu_selection: dict[str, Any] | None = None,
    enable_progress_bar: bool = True,
    progress_refresh_rate: int = 1,
    stage_log_stdout: bool = False,
    stop_after_stage1: bool = False,
    stage1_export_type: bool = False,
    stage1_agg: bool = False,
    stage1_polygon_csv: str | Path | None = None,
    stage1_grid_type_csv: str | Path | None = None,
    stage1_cell_type_csv: str | Path | None = None,
    stage2_polygon_csv: str | Path | None = None,
    grid_pred_expression_h5: str | Path | None = None,
    cell_pred_expression_h5: str | Path | None = None,
) -> dict[str, Any]:
    selected_dataset = dataset.lower()
    if selected_dataset != "brca":
        raise ValueError("The finetune-mu v2 training entry currently supports dataset='brca'.")

    selected_package_root = Path(package_root) if package_root is not None else PACKAGE_ROOT
    selected_results = Path(results_dir) if results_dir is not None else selected_package_root / "results"
    selected_run_dir = Path(run_dir) if run_dir is not None else _default_finetune_mu_run_dir(selected_dataset, selected_results)
    selected_run_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = selected_run_dir / "metrics"
    model_dir = selected_run_dir / "model"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    selected_dense_h5 = Path(dense_h5) if dense_h5 is not None else _default_dense_h5(selected_dataset, selected_results)
    selected_st_input = Path(st_h5ad) if st_h5ad is not None else selected_package_root / "data" / "st.h5ad"
    if selected_st_input.suffix == ".h5ad":
        selected_st_h5ad = selected_st_input
    elif selected_st_input.suffix == ".h5":
        selected_st_h5ad = _ensure_st_h5ad(
            selected_st_input,
            selected_run_dir / "input" / f"{selected_st_input.stem}.h5ad",
            force=False,
        )
    else:
        selected_st_h5ad = selected_st_input
    selected_deconv_csv = Path(deconv_csv) if deconv_csv is not None else _default_deconv_csv(selected_dataset, selected_results)
    selected_gene_list = Path(gene_list) if gene_list is not None else _default_gene_list(selected_dataset, selected_package_root)
    selected_mu_ref = Path(mu_ref) if mu_ref is not None else _default_mu_ref(selected_dataset, selected_package_root)
    selected_stage1_checkpoint = Path(stage1_checkpoint) if stage1_checkpoint is not None else None
    selected_feature_key = dense_feature_key
    selected_stage2_polygon_csv = (
        Path(stage2_polygon_csv)
        if stage2_polygon_csv is not None
        else Path.cwd() / "resources" / selected_dataset / "xen" / "aligned" / "he_alignmented_cell_boundaries.csv"
    )
    selected_grid_pred_expression_h5 = (
        Path(grid_pred_expression_h5) if grid_pred_expression_h5 is not None else selected_run_dir / "grid_pred_expression.h5"
    )
    selected_cell_pred_expression_h5 = (
        Path(cell_pred_expression_h5) if cell_pred_expression_h5 is not None else selected_run_dir / "cell_pred_expression.h5"
    )

    missing = [
        str(path)
        for path in [selected_dense_h5, selected_st_h5ad, selected_deconv_csv, selected_gene_list, selected_mu_ref]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing finetune-mu training inputs: " + ", ".join(missing))
    if selected_feature_key is None:
        selected_feature_key = _infer_dense_feature_key(selected_dense_h5)

    args = Namespace(
        dense_h5=selected_dense_h5,
        dense_feature_key=selected_feature_key,
        st_h5ad=selected_st_h5ad,
        deconv_csv=selected_deconv_csv,
        gene_list=selected_gene_list,
        sample_id=_default_sample_id(selected_dataset),
        limit_spots=limit_spots,
    )

    set_seed(seed)
    if torch.cuda.is_available():
        if cuda_device is not None:
            torch.cuda.set_device(int(cuda_device))
        torch.set_float32_matmul_precision("high")
        torch.empty(1, device="cuda")

    data = prepare_data(args)
    loaded_mu_ref = _load_mu_ref(selected_mu_ref, len(data.cell_type_names), len(data.gene_names))
    dataset_obj = DenseGridSpotDataset(data)
    loader = DataLoader(
        dataset_obj,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_spot_bags,
    )

    config: dict[str, Any] = {
        "dataset": selected_dataset,
        "architecture": "finetune_mu_factorized",
        "stage1": "train projection + router/type head with KL and confidence loss",
        "stage2": "freeze projection/router; train scalar scale head and bounded delta-mu head with expression loss",
        "dense_h5": str(selected_dense_h5),
        "dense_feature_key": str(selected_feature_key),
        "st_input": str(selected_st_input),
        "st_h5ad": str(selected_st_h5ad),
        "deconv_csv": str(selected_deconv_csv),
        "gene_list": str(selected_gene_list),
        "mu_ref": str(selected_mu_ref),
        "run_dir": str(selected_run_dir),
        "projection_dim": int(projection_dim),
        "stage1_epochs": int(stage1_epochs),
        "stage2_epochs": int(stage2_epochs),
        "stage1_checkpoint": None if selected_stage1_checkpoint is None else str(selected_stage1_checkpoint),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "lambda_conf": float(lambda_conf),
        "gene_loss_reduction": str(gene_loss_reduction),
        "expr_loss": str(expr_loss),
        "p_temperature": float(p_temperature),
        "init_scale": float(init_scale),
        "delta_alpha": float(delta_alpha),
        "skip_input_projector": bool(skip_input_projector),
        "type_head_hidden_layers": int(type_head_hidden_layers),
        "factorized_head_hidden_layers": int(factorized_head_hidden_layers),
        "scale_head_hidden_dim": None if scale_head_hidden_dim is None else int(scale_head_hidden_dim),
        "delta_head_hidden_dim": None if delta_head_hidden_dim is None else int(delta_head_hidden_dim),
        "mu_eps": float(mu_eps),
        "seed": int(seed),
        "limit_spots": None if limit_spots is None else int(limit_spots),
        "gpu_selection": gpu_selection,
        "n_grids": int(data.features.shape[0]),
        "n_supervised_spots": int(len(data.spot_to_grids)),
        "n_genes": int(len(data.gene_names)),
        "n_cell_types": int(len(data.cell_type_names)),
        "cell_type_names": data.cell_type_names,
        "mu_ref_shape": list(map(int, loaded_mu_ref.shape)),
        "mu_ref_row_sum_min": float(loaded_mu_ref.sum(axis=1).min()),
        "mu_ref_row_sum_max": float(loaded_mu_ref.sum(axis=1).max()),
        "cuda": bool(torch.cuda.is_available()),
        "cuda_device": None if cuda_device is None else int(cuda_device),
        "stop_after_stage1": bool(stop_after_stage1),
        "stage1_export_type": bool(stage1_export_type),
        "stage1_agg": bool(stage1_agg),
        "stage1_polygon_csv": None if stage1_polygon_csv is None else str(stage1_polygon_csv),
        "stage1_grid_type_csv": None if stage1_grid_type_csv is None else str(stage1_grid_type_csv),
        "stage1_cell_type_csv": None if stage1_cell_type_csv is None else str(stage1_cell_type_csv),
        "stage2_polygon_csv": str(selected_stage2_polygon_csv),
        "grid_pred_expression_h5": str(selected_grid_pred_expression_h5),
        "cell_pred_expression_h5": str(selected_cell_pred_expression_h5),
    }
    (selected_run_dir / "input_paths.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if stage_log_stdout:
        print(json.dumps({"stage": "finetune_mu_data_loaded", **config}, indent=2), flush=True)

    if selected_stage1_checkpoint is None:
        stage1_model = _build_finetune_model(
            data,
            loaded_mu_ref,
            projection_dim=projection_dim,
            lr=lr,
            lambda_conf=lambda_conf,
            gene_loss_reduction=gene_loss_reduction,
            expr_loss=expr_loss,
            p_temperature=p_temperature,
            init_scale=init_scale,
            delta_alpha=delta_alpha,
            skip_input_projector=skip_input_projector,
            type_head_hidden_layers=type_head_hidden_layers,
            factorized_head_hidden_layers=factorized_head_hidden_layers,
            scale_head_hidden_dim=scale_head_hidden_dim,
            delta_head_hidden_dim=delta_head_hidden_dim,
            mu_eps=mu_eps,
            stage="stage1",
        )
        stage1_ckpt = _train_finetune_stage(
            stage1_model,
            loader,
            model_dir / "stage1_router",
            stage1_epochs,
            stage_name=f"Stage 1 router ({int(stage1_epochs)} epochs)",
            cuda_device=cuda_device,
            enable_progress_bar=enable_progress_bar,
            progress_refresh_rate=progress_refresh_rate,
        )
        if stage_log_stdout:
            print(json.dumps({"stage": "stage1_done", "checkpoint": str(stage1_ckpt)}, indent=2), flush=True)
    else:
        stage1_ckpt = selected_stage1_checkpoint
        if not stage1_ckpt.exists():
            raise FileNotFoundError(stage1_ckpt)
        if stage_log_stdout:
            print(json.dumps({"stage": "stage1_reused", "checkpoint": str(stage1_ckpt)}, indent=2), flush=True)

    if stop_after_stage1:
        metrics: dict[str, Any] = {"stage1_checkpoint": str(stage1_ckpt), "stage2_checkpoint": None}
        if stage1_export_type or stage1_agg:
            stage1_export_model = _build_finetune_model(
                data,
                loaded_mu_ref,
                projection_dim=projection_dim,
                lr=lr,
                lambda_conf=lambda_conf,
                gene_loss_reduction=gene_loss_reduction,
                expr_loss=expr_loss,
                p_temperature=p_temperature,
                init_scale=init_scale,
                delta_alpha=delta_alpha,
                skip_input_projector=skip_input_projector,
                type_head_hidden_layers=type_head_hidden_layers,
                factorized_head_hidden_layers=factorized_head_hidden_layers,
                scale_head_hidden_dim=scale_head_hidden_dim,
                delta_head_hidden_dim=delta_head_hidden_dim,
                mu_eps=mu_eps,
                stage="stage1",
            )
            stage1_export_model = _load_finetune_checkpoint(stage1_export_model, stage1_ckpt, stage="stage1")
            selected_stage1_grid_type_csv = (
                Path(stage1_grid_type_csv) if stage1_grid_type_csv is not None else selected_run_dir / "stage1_grid_type.csv"
            )
            metrics.update(
                export_finetune_mu_stage1_type_predictions(
                    model=stage1_export_model,
                    data=data,
                    output_csv=selected_stage1_grid_type_csv,
                    max_grids_per_batch=int(export_max_grids_per_batch),
                    device=default_device(),
                )
            )
            if stage1_agg:
                selected_stage1_polygon_csv = (
                    Path(stage1_polygon_csv)
                    if stage1_polygon_csv is not None
                    else selected_package_root
                    / "resources"
                    / selected_dataset
                    / "xen"
                    / "aligned"
                    / "he_alignmented_cell_boundaries.csv"
                )
                if not selected_stage1_polygon_csv.exists():
                    raise FileNotFoundError(selected_stage1_polygon_csv)
                selected_stage1_cell_type_csv = (
                    Path(stage1_cell_type_csv)
                    if stage1_cell_type_csv is not None
                    else selected_run_dir / "stage1_cell_type.csv"
                )
                metrics.update(
                    _aggregate_stage1_type_to_cells(
                        grid_type_csv=selected_stage1_grid_type_csv,
                        polygon_csv=selected_stage1_polygon_csv,
                        output_csv=selected_stage1_cell_type_csv,
                    )
                )
        summary = {**config, **metrics}
        (selected_run_dir / "summary_stage1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if stage_log_stdout:
            print(json.dumps({"stage": "stage1_only_done", **metrics}, indent=2), flush=True)
        return summary

    stage2_model = _build_finetune_model(
        data,
        loaded_mu_ref,
        projection_dim=projection_dim,
        lr=lr,
        lambda_conf=lambda_conf,
        gene_loss_reduction=gene_loss_reduction,
        expr_loss=expr_loss,
        p_temperature=p_temperature,
        init_scale=init_scale,
        delta_alpha=delta_alpha,
        skip_input_projector=skip_input_projector,
        type_head_hidden_layers=type_head_hidden_layers,
        factorized_head_hidden_layers=factorized_head_hidden_layers,
        scale_head_hidden_dim=scale_head_hidden_dim,
        delta_head_hidden_dim=delta_head_hidden_dim,
        mu_eps=mu_eps,
        stage="stage2",
    )
    stage2_model = _load_finetune_checkpoint(stage2_model, stage1_ckpt, stage="stage2")
    stage2_ckpt = _train_finetune_stage(
        stage2_model,
        loader,
        model_dir / "stage2_expression",
        stage2_epochs,
        stage_name=f"Stage 2 expression ({int(stage2_epochs)} epochs)",
        cuda_device=cuda_device,
        enable_progress_bar=enable_progress_bar,
        progress_refresh_rate=progress_refresh_rate,
    )
    if stage_log_stdout:
        print(json.dumps({"stage": "stage2_done", "checkpoint": str(stage2_ckpt)}, indent=2), flush=True)

    final_model = _build_finetune_model(
        data,
        loaded_mu_ref,
        projection_dim=projection_dim,
        lr=lr,
        lambda_conf=lambda_conf,
        gene_loss_reduction=gene_loss_reduction,
        expr_loss=expr_loss,
        p_temperature=p_temperature,
        init_scale=init_scale,
        delta_alpha=delta_alpha,
        skip_input_projector=skip_input_projector,
        type_head_hidden_layers=type_head_hidden_layers,
        factorized_head_hidden_layers=factorized_head_hidden_layers,
        scale_head_hidden_dim=scale_head_hidden_dim,
        delta_head_hidden_dim=delta_head_hidden_dim,
        mu_eps=mu_eps,
        stage="stage2",
    )
    final_model = _load_finetune_checkpoint(final_model, stage2_ckpt, stage="stage2")
    final_model.eval()

    metrics: dict[str, Any] = {"stage1_checkpoint": str(stage1_ckpt), "stage2_checkpoint": str(stage2_ckpt)}
    if data.spot_to_grids:
        spot_expr_pred, spot_type_pred = predict_supervised(
            model=final_model,
            data=data,
            batch_size=int(batch_size),
            device=default_device(),
        )
        if spot_expr_pred is not None:
            metrics.update(compute_spot_expression_metrics(spot_expr_pred, data, metrics_dir))
        if spot_type_pred is not None:
            metrics.update(compute_spot_type_metrics(spot_type_pred, data, metrics_dir))
    if not no_export:
        grid_predictions_h5 = selected_run_dir / "grid_predictions.h5"
        metrics.update(
            export_finetune_mu_grid_predictions(
                model=final_model,
                data=data,
                output_h5=grid_predictions_h5,
                max_grids_per_batch=int(export_max_grids_per_batch),
                device=default_device(),
            )
        )
        metrics.update(
            export_stage2_expression_h5s(
                grid_predictions_h5=grid_predictions_h5,
                polygon_csv=selected_stage2_polygon_csv,
                grid_output_h5=selected_grid_pred_expression_h5,
                cell_output_h5=selected_cell_pred_expression_h5,
            )
        )

    summary = {**config, **metrics}
    (selected_run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if stage_log_stdout:
        print(json.dumps({"stage": "done", **metrics}, indent=2), flush=True)
    return summary
