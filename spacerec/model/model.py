#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from anndata import read_h5ad
from scipy import sparse
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, random_split

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar

SPACEREC_ROOT = Path(__file__).resolve().parents[2]

from .heads import REGISTRY_KEYS, GeneHead, ResidualTypeHead  # noqa: E402


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def decode_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim > 1 and values.shape[-1] == 1:
        values = values[:, 0]
    return np.array(
        [
            item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
            for item in values
        ],
        dtype=object,
    )


def normalize_spot_id(value: object, sample_id: str | None = None) -> str:
    text = value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
    if text.endswith("_BREAST"):
        text = text[: -len("_BREAST")]
    if sample_id and text.endswith(f"_{sample_id}"):
        text = text[: -len(sample_id) - 1]
    return text


def read_gene_list(gene_list: str | Path | None) -> list[str] | None:
    if gene_list is None:
        return None
    path = Path(gene_list)
    genes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return genes or None


def dense_matrix(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def load_spot_expression(
    st_h5ad: str | Path,
    gene_list: str | Path | None,
    sample_id: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    adata = read_h5ad(st_h5ad)
    genes = read_gene_list(gene_list)
    if genes is not None:
        genes = [gene for gene in genes if gene in adata.var_names]
        if not genes:
            raise ValueError("No requested genes were found in the ST h5ad.")
        adata = adata[:, genes].copy()
    expression = dense_matrix(adata.X).astype(np.float32, copy=False)
    spot_ids = np.asarray([normalize_spot_id(spot, sample_id) for spot in adata.obs_names], dtype=object)
    return expression, spot_ids, list(map(str, adata.var_names))


def load_deconvolution(
    deconv_path: str | Path,
    spot_ids: np.ndarray,
    sample_id: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    table = pd.read_csv(deconv_path, index_col=0)
    table.index = [normalize_spot_id(index, sample_id) for index in table.index]
    table = table.groupby(level=0).mean()
    table = table.reindex(spot_ids).fillna(0.0)
    values = table.to_numpy(dtype=np.float32)
    values = np.clip(values, 0.0, None)
    row_sum = values.sum(axis=1, keepdims=True)
    values = np.divide(values, np.maximum(row_sum, 1e-8), out=np.zeros_like(values), where=row_sum > 0)
    return values.astype(np.float32), list(map(str, table.columns))


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    return safe_pearson(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())


def compute_spot_expression_metrics(
    predicted_spot_expression: np.ndarray,
    data: "DenseGridSpotData",
    metrics_dir: str | Path,
) -> dict[str, float | int]:
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pred = np.asarray(predicted_spot_expression, dtype=np.float32)
    target = np.asarray(data.spot_expression, dtype=np.float32)
    gene_rows = []
    for gene_index, gene in enumerate(data.gene_names):
        gene_rows.append(
            {
                "gene": gene,
                "pcc": safe_pearson(pred[:, gene_index], target[:, gene_index]),
                "scc": safe_spearman(pred[:, gene_index], target[:, gene_index]),
            }
        )
    spot_rows = []
    for spot_index, spot_id in enumerate(data.spot_ids):
        spot_rows.append(
            {
                "spot_id": spot_id,
                "gene_pcc": safe_pearson(pred[spot_index], target[spot_index]),
                "gene_scc": safe_spearman(pred[spot_index], target[spot_index]),
            }
        )
    gene_metrics = pd.DataFrame(gene_rows)
    spot_metrics = pd.DataFrame(spot_rows)
    gene_metrics.to_csv(metrics_dir / "gene_expression_metrics.csv", index=False)
    spot_metrics.to_csv(metrics_dir / "spot_expression_metrics.csv", index=False)
    return {
        "mean_gene_PCC": float(gene_metrics["pcc"].mean()),
        "mean_gene_SCC": float(gene_metrics["scc"].mean()),
        "mean_spot_gene_PCC": float(spot_metrics["gene_pcc"].mean()),
        "mean_spot_gene_SCC": float(spot_metrics["gene_scc"].mean()),
        "n_gene_metric_spots": int(pred.shape[0]),
        "n_genes": int(pred.shape[1]),
    }


@dataclass
class DenseGridSpotData:
    features: np.ndarray
    center_xy: np.ndarray
    bbox_xyxy: np.ndarray
    grid_tissue_fraction: np.ndarray
    is_tissue: np.ndarray
    spot_index: np.ndarray
    nearest_spot_index: np.ndarray
    nearest_spot_distance: np.ndarray
    position_spot_ids: np.ndarray
    position_spot_barcodes: np.ndarray
    position_spot_xy: np.ndarray
    spot_expression: np.ndarray
    spot_proportions: np.ndarray
    spot_ids: np.ndarray
    spot_position_indices: np.ndarray
    spot_to_grids: list[np.ndarray]
    gene_names: list[str]
    cell_type_names: list[str]

    @property
    def input_dim(self) -> int:
        return int(self.features.shape[1])


def read_h5_string_array(handle: h5py.File, key: str) -> np.ndarray:
    return decode_array(handle[key][()])


def load_dense_h5(
    path: Path,
    feature_key: str = "concat3072",
    feature_indices: np.ndarray | None = None,
    load_features: bool = True,
) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        if feature_key not in handle:
            raise KeyError(f"Feature key {feature_key!r} was not found in {path}. Available keys: {sorted(handle.keys())}")
        if load_features:
            if feature_indices is None:
                features = np.asarray(handle[feature_key], dtype=np.float32)
            else:
                feature_indices = np.asarray(feature_indices, dtype=np.int64)
                if feature_indices.size < 2 or np.all(feature_indices[1:] >= feature_indices[:-1]):
                    features = np.asarray(handle[feature_key][feature_indices], dtype=np.float32)
                else:
                    order = np.argsort(feature_indices, kind="stable")
                    sorted_indices = feature_indices[order]
                    sorted_features = np.asarray(handle[feature_key][sorted_indices], dtype=np.float32)
                    inverse_order = np.empty_like(order)
                    inverse_order[order] = np.arange(order.size)
                    features = sorted_features[inverse_order]
        else:
            features = np.empty((0, int(handle[feature_key].shape[1])), dtype=np.float32)
        return {
            "features": features,
            "center_xy": np.asarray(handle["center_xy"], dtype=np.float32),
            "bbox_xyxy": np.asarray(handle["bbox_xyxy"], dtype=np.int32),
            "grid_tissue_fraction": np.asarray(handle["grid_tissue_fraction"], dtype=np.float32),
            "is_tissue": np.asarray(handle["is_tissue"], dtype=np.bool_),
            "spot_index": np.asarray(handle["spot_index"], dtype=np.int32),
            "nearest_spot_index": np.asarray(handle["nearest_spot_index"], dtype=np.int32),
            "nearest_spot_distance": np.asarray(handle["nearest_spot_distance"], dtype=np.float32),
            "position_spot_ids": read_h5_string_array(handle, "spot_id"),
            "position_spot_barcodes": read_h5_string_array(handle, "spot_barcode"),
            "position_spot_xy": np.asarray(handle["spot_xy"], dtype=np.float32),
        }


def build_spot_to_grids(
    spot_ids: np.ndarray,
    position_spot_ids: np.ndarray,
    spot_index: np.ndarray,
    is_tissue: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    position_by_id = {str(spot): index for index, spot in enumerate(position_spot_ids)}
    supervised_grid_indices = np.flatnonzero((spot_index >= 0) & is_tissue)
    supervised_spot_index = spot_index[supervised_grid_indices]
    order = np.argsort(supervised_spot_index, kind="stable")
    sorted_spot_index = supervised_spot_index[order]
    sorted_grid_indices = supervised_grid_indices[order]

    grouped: dict[int, np.ndarray] = {}
    if sorted_spot_index.size:
        starts = np.r_[0, np.flatnonzero(sorted_spot_index[1:] != sorted_spot_index[:-1]) + 1]
        for start, end in zip(starts, np.r_[starts[1:], sorted_spot_index.size]):
            grouped[int(sorted_spot_index[start])] = sorted_grid_indices[start:end].astype(np.int64)

    kept_spots: list[int] = []
    kept_positions: list[int] = []
    spot_to_grids: list[np.ndarray] = []
    for target_index, spot_id in enumerate(spot_ids):
        position_index = position_by_id.get(str(spot_id), -1)
        if position_index < 0:
            continue
        grids = grouped.get(int(position_index))
        if grids is None or grids.size == 0:
            continue
        kept_spots.append(target_index)
        kept_positions.append(position_index)
        spot_to_grids.append(grids)

    if not spot_to_grids:
        raise ValueError("No supervised spots with tissue dense grids were found.")
    return (
        np.asarray(kept_spots, dtype=np.int64),
        np.asarray(kept_positions, dtype=np.int64),
        spot_to_grids,
    )


def prepare_data(args: argparse.Namespace) -> DenseGridSpotData:
    dense = load_dense_h5(args.dense_h5, args.dense_feature_key, load_features=True)
    expression, spot_ids, gene_names = load_spot_expression(
        args.st_h5ad,
        args.gene_list,
        args.sample_id,
    )
    proportions, cell_type_names = load_deconvolution(args.deconv_csv, spot_ids, args.sample_id)
    normalized_position_spot_ids = np.asarray(
        [normalize_spot_id(spot, args.sample_id) for spot in dense["position_spot_ids"]],
        dtype=object,
    )
    kept_spot_indices, kept_position_indices, spot_to_grids = build_spot_to_grids(
        spot_ids=spot_ids,
        position_spot_ids=normalized_position_spot_ids,
        spot_index=dense["spot_index"],
        is_tissue=dense["is_tissue"],
    )
    if args.limit_spots is not None:
        limit = int(args.limit_spots)
        kept_spot_indices = kept_spot_indices[:limit]
        kept_position_indices = kept_position_indices[:limit]
        spot_to_grids = spot_to_grids[:limit]
    return DenseGridSpotData(
        features=dense["features"],
        center_xy=dense["center_xy"],
        bbox_xyxy=dense["bbox_xyxy"],
        grid_tissue_fraction=dense["grid_tissue_fraction"],
        is_tissue=dense["is_tissue"],
        spot_index=dense["spot_index"],
        nearest_spot_index=dense["nearest_spot_index"],
        nearest_spot_distance=dense["nearest_spot_distance"],
        position_spot_ids=normalized_position_spot_ids,
        position_spot_barcodes=dense["position_spot_barcodes"],
        position_spot_xy=dense["position_spot_xy"],
        spot_expression=expression[kept_spot_indices],
        spot_proportions=proportions[kept_spot_indices],
        spot_ids=spot_ids[kept_spot_indices],
        spot_position_indices=kept_position_indices,
        spot_to_grids=spot_to_grids,
        gene_names=gene_names,
        cell_type_names=cell_type_names,
    )


class DenseGridSpotDataset(Dataset):
    def __init__(self, data: DenseGridSpotData):
        self.data = data

    def __len__(self) -> int:
        return len(self.data.spot_to_grids)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        grid_indices = self.data.spot_to_grids[index]
        return {
            "grid_x": torch.from_numpy(self.data.features[grid_indices]).float(),
            "grid_index": torch.from_numpy(grid_indices).long(),
            "target_spot_expr": torch.from_numpy(self.data.spot_expression[index]).float(),
            "target_spot_type_prop": torch.from_numpy(self.data.spot_proportions[index]).float(),
            "spot_id": str(self.data.spot_ids[index]),
        }


def collate_spot_bags(batch: list[dict[str, Tensor | str]]) -> dict[str, Tensor | list[str]]:
    grid_parts: list[Tensor] = []
    grid_index_parts: list[Tensor] = []
    bag_index_parts: list[Tensor] = []
    expr_parts: list[Tensor] = []
    type_parts: list[Tensor] = []
    spot_ids: list[str] = []
    for bag_index, item in enumerate(batch):
        grid_x = item["grid_x"]
        grid_index = item["grid_index"]
        target_expr = item["target_spot_expr"]
        target_type = item["target_spot_type_prop"]
        assert isinstance(grid_x, Tensor)
        assert isinstance(grid_index, Tensor)
        assert isinstance(target_expr, Tensor)
        assert isinstance(target_type, Tensor)
        grid_parts.append(grid_x)
        grid_index_parts.append(grid_index)
        bag_index_parts.append(torch.full((grid_x.shape[0],), bag_index, dtype=torch.long))
        expr_parts.append(target_expr)
        type_parts.append(target_type)
        spot_ids.append(str(item["spot_id"]))
    return {
        "grid_x": torch.cat(grid_parts, dim=0),
        "grid_index": torch.cat(grid_index_parts, dim=0),
        "grid_bag_index": torch.cat(bag_index_parts, dim=0),
        "target_spot_expr": torch.stack(expr_parts, dim=0),
        "target_spot_type_prop": torch.stack(type_parts, dim=0),
        "spot_id": spot_ids,
    }


def scatter_mean(values: Tensor, index: Tensor, dim_size: int) -> Tensor:
    output = values.new_zeros((int(dim_size), values.shape[-1]))
    output.index_add_(0, index, values)
    counts = torch.bincount(index, minlength=int(dim_size)).clamp_min(1).to(values.dtype)
    return output / counts.unsqueeze(-1)


def scatter_sum(values: Tensor, index: Tensor, dim_size: int) -> Tensor:
    output = values.new_zeros((int(dim_size), values.shape[-1]))
    output.index_add_(0, index, values)
    return output


class DenseGridCore(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_genes: int,
        n_cell_types: int,
        projection_dim: int,
    ):
        super().__init__()
        self.input_projector = nn.Sequential(
            nn.Linear(int(input_dim), int(projection_dim)),
            nn.GELU(),
            nn.LayerNorm(int(projection_dim)),
        )
        self.final_dim = int(projection_dim)
        self.gene_head = GeneHead(
            input_dim=self.final_dim,
            output_dim=int(n_genes),
            hidden_dim=[512, 512, 256],
            dropout_rate=0.05,
        )
        self.type_head = ResidualTypeHead(
            input_dim=self.final_dim,
            hidden_dim=512,
            num_cell_types=int(n_cell_types),
            dropout=0.05,
            temperature=2.0,
        )

    def encode_grids(self, grid_x: Tensor, grid_bag_index: Tensor) -> Tensor:
        return self.input_projector(grid_x)


    # core code for training
    def instance_forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        grid_bag_index = batch.get("grid_bag_index")
        if grid_bag_index is None:
            grid_bag_index = torch.zeros((batch["grid_x"].shape[0],), dtype=torch.long, device=batch["grid_x"].device)
        encoded = self.encode_grids(batch["grid_x"], grid_bag_index)
        output = {REGISTRY_KEYS.OUTPUT_EMBEDDING: encoded}
        if self.gene_head is not None:
            output[REGISTRY_KEYS.OUTPUT_PREDICTION] = self.gene_head(
                {REGISTRY_KEYS.OUTPUT_EMBEDDING: encoded}
            )[REGISTRY_KEYS.OUTPUT_PREDICTION]
        if self.type_head is not None:
            output["output_prob"] = self.type_head(encoded)
        return output

    def bag_forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        n_spots = int(batch["target_spot_expr"].shape[0])
        instance = self.instance_forward(batch)
        output = {"instance": instance}
        if REGISTRY_KEYS.OUTPUT_PREDICTION in instance:
            output[REGISTRY_KEYS.OUTPUT_PREDICTION] = scatter_sum(
                instance[REGISTRY_KEYS.OUTPUT_PREDICTION],
                batch["grid_bag_index"],
                n_spots,
            )
        if "output_prob" in instance:
            output["output_prob"] = scatter_mean(instance["output_prob"], batch["grid_bag_index"], n_spots)
        return output


class DenseGridLightning(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        n_genes: int,
        n_cell_types: int,
        projection_dim: int,
        lr: float,
        lambda_deconv: float,
        type_confidence_alpha: float,
        gene_loss_reduction: str = "sum",
    ):
        super().__init__()
        self.save_hyperparameters()
        if gene_loss_reduction not in {"mean", "sum"}:
            raise ValueError("gene_loss_reduction must be 'mean' or 'sum'.")
        self.model = DenseGridCore(
            input_dim=input_dim,
            n_genes=n_genes,
            n_cell_types=n_cell_types,
            projection_dim=projection_dim,
        )
        self.lr = float(lr)
        self.lambda_deconv = float(lambda_deconv)
        self.type_confidence_alpha = float(type_confidence_alpha)
        self.gene_loss_reduction = str(gene_loss_reduction)

    # forward propagation
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.model.bag_forward(batch)

    def _type_loss(self, pred_prob: Tensor, target_prop: Tensor) -> Tensor:
        target_prop = target_prop.clamp_min(0.0)
        target_prop = target_prop / target_prop.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return F.kl_div(pred_prob.clamp_min(1e-8).log(), target_prop, reduction="batchmean")

    def _step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        output = self(batch)
        batch_size = int(batch["target_spot_expr"].shape[0])
        zero = batch["target_spot_expr"].new_zeros(())
        expr_loss = zero
        deconv_loss = zero
        confidence_loss = zero
        type_loss = zero
        loss = zero

        if REGISTRY_KEYS.OUTPUT_PREDICTION in output:
            pred_expr = output[REGISTRY_KEYS.OUTPUT_PREDICTION]
            expr_loss = F.huber_loss(
                torch.log1p(pred_expr.clamp_min(0.0)),
                torch.log1p(batch["target_spot_expr"].clamp_min(0.0)),
                reduction=self.gene_loss_reduction,
            )
            loss = loss + expr_loss
        if "output_prob" in output:
            deconv_loss = self._type_loss(output["output_prob"], batch["target_spot_type_prop"])
            type_loss = deconv_loss
            if self.type_confidence_alpha > 0.0:
                grid_prob = output["instance"]["output_prob"].clamp_min(1e-8)
                confidence_loss = -torch.mean(torch.log(grid_prob.max(dim=1).values))
                type_loss = self.type_confidence_alpha * confidence_loss + (
                    1.0 - self.type_confidence_alpha
                ) * deconv_loss
            loss = loss + self.lambda_deconv * type_loss

        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}_expr_loss", expr_loss, batch_size=batch_size)
        self.log(f"{stage}_deconv_loss", deconv_loss, batch_size=batch_size)
        self.log(f"{stage}_confidence_loss", confidence_loss, batch_size=batch_size)
        self.log(f"{stage}_type_loss", type_loss, batch_size=batch_size)
        if stage == "train":
            self.log("train_loss_epoch", loss, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("train_expr_loss_epoch", expr_loss, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("train_type_loss_epoch", type_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)


def split_dataset(dataset: Dataset, val_fraction: float, seed: int) -> tuple[Dataset, Dataset]:
    val_size = max(1, int(round(len(dataset) * float(val_fraction))))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Training split is empty.")
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def tensor_batch(batch: dict[str, Tensor | list[str]], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items() if isinstance(value, Tensor)}


def predict_supervised(
    model: DenseGridLightning,
    data: DenseGridSpotData,
    batch_size: int,
    device: str | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    loader = DataLoader(
        DenseGridSpotDataset(data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_spot_bags,
    )
    device_obj = torch.device(device or default_device())
    model = model.to(device_obj)
    model.eval()
    expr_parts: list[np.ndarray] = []
    type_parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            output = model.model.bag_forward(tensor_batch(batch, device_obj))
            if REGISTRY_KEYS.OUTPUT_PREDICTION in output:
                expr_parts.append(output[REGISTRY_KEYS.OUTPUT_PREDICTION].detach().cpu().numpy())
            if "output_prob" in output:
                type_parts.append(output["output_prob"].detach().cpu().numpy())
    expr = np.concatenate(expr_parts, axis=0).astype(np.float32) if expr_parts else None
    types = np.concatenate(type_parts, axis=0).astype(np.float32) if type_parts else None
    return expr, types


def compute_spot_type_metrics(predicted: np.ndarray, data: DenseGridSpotData, metrics_dir: Path) -> dict[str, float | int]:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    target = np.clip(data.spot_proportions, 0.0, None)
    target = target / np.maximum(target.sum(axis=1, keepdims=True), 1e-8)
    pred = np.clip(predicted, 1e-8, None)
    pred = pred / np.maximum(pred.sum(axis=1, keepdims=True), 1e-8)
    per_spot_kl = np.sum(target * (np.log(np.clip(target, 1e-8, None)) - np.log(pred)), axis=1)
    per_spot_l1 = np.mean(np.abs(pred - target), axis=1)
    per_spot_l2 = np.mean((pred - target) ** 2, axis=1)
    pd.DataFrame(
        {
            "spot_id": data.spot_ids.astype(str),
            "type_kl": per_spot_kl,
            "type_l1": per_spot_l1,
            "type_l2": per_spot_l2,
        }
    ).to_csv(metrics_dir / "spot_type_metrics.csv", index=False)
    pd.DataFrame(pred, index=data.spot_ids.astype(str), columns=data.cell_type_names).to_csv(
        metrics_dir / "spot_type_probabilities.csv"
    )
    return {
        "mean_spot_type_KL": float(per_spot_kl.mean()),
        "mean_spot_type_L1": float(per_spot_l1.mean()),
        "mean_spot_type_L2": float(per_spot_l2.mean()),
        "n_type_metric_spots": int(pred.shape[0]),
        "n_cell_types": int(pred.shape[1]),
    }


def iter_projection_chunks(n: int, max_grids: int):
    for start in range(0, n, int(max_grids)):
        stop = min(start + int(max_grids), n)
        indices = np.arange(start, stop, dtype=np.int64)
        bag_index = np.zeros((indices.size,), dtype=np.int64)
        yield indices, bag_index


def write_string_dataset(handle: h5py.File, name: str, values: list[str] | np.ndarray) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    handle.create_dataset(name, data=np.asarray(values, dtype=object), dtype=dtype)


def write_sorted_rows(dataset: h5py.Dataset, sorted_indices: np.ndarray, sorted_values: np.ndarray) -> None:
    if sorted_indices.size == 0:
        return
    unique_indices, first_positions = np.unique(sorted_indices, return_index=True)
    values = sorted_values[first_positions]
    run_starts = np.r_[0, np.flatnonzero(np.diff(unique_indices) != 1) + 1]
    run_stops = np.r_[run_starts[1:], unique_indices.size]
    for start, stop in zip(run_starts, run_stops):
        left = int(unique_indices[start])
        right = int(unique_indices[stop - 1]) + 1
        dataset[left:right] = values[start:stop]


def export_grid_predictions(
    model: DenseGridLightning,
    data: DenseGridSpotData,
    output_h5: Path,
    max_grids_per_batch: int,
    device: str | None = None,
) -> dict[str, str]:
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device or default_device())
    model = model.to(device_obj)
    model.eval()
    n_grids = int(data.features.shape[0])
    chunk_iter = iter_projection_chunks(n_grids, max_grids_per_batch)
    with h5py.File(output_h5, "w") as handle:
        handle.create_dataset("center_xy", data=data.center_xy, compression="gzip")
        handle.create_dataset("bbox_xyxy", data=data.bbox_xyxy, compression="gzip")
        handle.create_dataset("grid_tissue_fraction", data=data.grid_tissue_fraction, compression="gzip")
        handle.create_dataset("is_tissue", data=data.is_tissue, compression="gzip")
        handle.create_dataset("spot_index", data=data.spot_index, compression="gzip")
        handle.create_dataset("nearest_spot_index", data=data.nearest_spot_index, compression="gzip")
        handle.create_dataset("nearest_spot_distance", data=data.nearest_spot_distance, compression="gzip")
        write_string_dataset(handle, "position_spot_id", data.position_spot_ids)
        write_string_dataset(handle, "position_spot_barcode", data.position_spot_barcodes)
        write_string_dataset(handle, "gene_name", data.gene_names)
        write_string_dataset(handle, "cell_type_name", data.cell_type_names)
        expr_ds = handle.create_dataset(
            "expr_pred",
            shape=(n_grids, len(data.gene_names)),
            dtype=np.float32,
            chunks=(min(1024, n_grids), len(data.gene_names)),
            compression="lzf",
        )
        type_ds = handle.create_dataset(
            "type_prob",
            shape=(n_grids, len(data.cell_type_names)),
            dtype=np.float32,
            chunks=(min(8192, n_grids), len(data.cell_type_names)),
            compression="gzip",
        )
        top1_ds = handle.create_dataset("type_top1", shape=(n_grids,), dtype=np.int16, compression="gzip")
        with torch.no_grad():
            for grid_indices, bag_index in chunk_iter:
                batch = {
                    "grid_x": torch.from_numpy(data.features[grid_indices]).float().to(device_obj),
                    "grid_bag_index": torch.from_numpy(bag_index).long().to(device_obj),
                }
                output = model.model.instance_forward(batch)
                write_order = np.argsort(grid_indices, kind="stable")
                write_indices = grid_indices[write_order]
                expr_values = output[REGISTRY_KEYS.OUTPUT_PREDICTION].detach().cpu().numpy()
                write_sorted_rows(expr_ds, write_indices, expr_values[write_order])
                probs = output["output_prob"].detach().cpu().numpy().astype(np.float32)
                write_sorted_rows(type_ds, write_indices, probs[write_order])
                top1_values = probs.argmax(axis=1).astype(np.int16)[write_order]
                write_sorted_rows(top1_ds, write_indices, top1_values)
        handle.attrs["architecture"] = "projection_heads"
        handle.attrs["head_mode"] = "both"
        handle.attrs["use_set_transformer"] = False
        handle.attrs["use_gene_head"] = True
        handle.attrs["use_type_head"] = True
        handle.attrs["expression_target"] = "raw_counts"
        handle.attrs["expression_loss"] = "huber_log1p_sum_grid_vs_raw_counts"
    return {"grid_predictions_h5": str(output_h5)}


def run_model(args: argparse.Namespace) -> dict[str, float | int | str]:
    set_seed(args.seed)
    if torch.cuda.is_available():
        if getattr(args, "cuda_device", None) is not None:
            torch.cuda.set_device(int(args.cuda_device))
        torch.set_float32_matmul_precision("high")
        if args.eager_cuda_init:
            torch.empty(1, device="cuda")
    started = time.time()

    run_dir = args.run_dir
    checkpoint_dir = run_dir / "model"
    metrics_dir = run_dir / "metrics"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_data(args)
    spot_grid_counts = np.asarray([len(item) for item in data.spot_to_grids], dtype=np.int32)
    spot_grid_summary = (
        {
            "spot_grid_count_min": int(spot_grid_counts.min()),
            "spot_grid_count_median": float(np.median(spot_grid_counts)),
            "spot_grid_count_max": int(spot_grid_counts.max()),
        }
        if spot_grid_counts.size
        else {
            "spot_grid_count_min": 0,
            "spot_grid_count_median": 0.0,
            "spot_grid_count_max": 0,
        }
    )
    model_summary = {
        "architecture": "projection_heads",
        "head_mode": "both",
        "use_set_transformer": False,
        "use_gene_head": True,
        "use_type_head": True,
        "input_dim": data.input_dim,
        "n_grids": int(data.features.shape[0]),
        "n_tissue_grids": int(data.is_tissue.sum()),
        "n_supervised_spots": int(len(data.spot_to_grids)),
        **spot_grid_summary,
        "n_genes": int(len(data.gene_names)),
        "n_cell_types": int(len(data.cell_type_names)),
        "cell_type_names": data.cell_type_names,
        "loss": str(getattr(args, "loss", None) or "custom"),
        "expression_target": "raw_counts",
        "expression_loss": "huber_log1p_sum_grid_vs_raw_counts",
        "cuda": bool(torch.cuda.is_available()),
        "cuda_device": None if getattr(args, "cuda_device", None) is None else int(args.cuda_device),
        "gpu_selection": getattr(args, "gpu_selection", None),
    }
    if bool(getattr(args, "stage_log_stdout", False)):
        print(json.dumps(model_summary, indent=2), flush=True)

    model = DenseGridLightning(
        input_dim=data.input_dim,
        n_genes=len(data.gene_names),
        n_cell_types=len(data.cell_type_names),
        projection_dim=args.projection_dim,
        lr=args.lr,
        lambda_deconv=args.lambda_deconv,
        type_confidence_alpha=args.type_confidence_alpha,
        gene_loss_reduction=args.gene_loss_reduction,
    )

    dataset = DenseGridSpotDataset(data)
    if args.train_all:
        train_dataset = dataset
        val_loader = None
    else:
        train_dataset, val_dataset = split_dataset(dataset, args.val_fraction, args.seed)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_spot_bags,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_spot_bags,
    )
    if args.train_all:
        checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best_train_model",
            monitor="train_loss_epoch",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
        callbacks = [
            checkpoint,
            EarlyStopping(
                monitor="train_loss_epoch",
                mode="min",
                patience=args.patience,
                min_delta=args.early_stop_min_delta,
            ),
        ]
    else:
        checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best_model",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
        callbacks = [
            checkpoint,
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=args.patience,
                min_delta=args.early_stop_min_delta,
            ),
        ]
    if args.enable_progress_bar:
        callbacks.append(TQDMProgressBar(refresh_rate=max(1, int(args.progress_refresh_rate))))
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[int(args.cuda_device)]
        if torch.cuda.is_available() and getattr(args, "cuda_device", None) is not None
        else 1,
        callbacks=callbacks,
        default_root_dir=run_dir,
        log_every_n_steps=10,
        enable_progress_bar=bool(args.enable_progress_bar),
    )
    if val_loader is not None:
        trainer.fit(model, train_loader, val_loader)
    else:
        trainer.fit(model, train_loader)
    if args.train_all:
        best_path = Path(checkpoint.best_model_path) if checkpoint.best_model_path else checkpoint_dir / "last.ckpt"
    else:
        best_path = Path(checkpoint.best_model_path) if checkpoint.best_model_path else checkpoint_dir / "best_model.ckpt"
    if best_path.exists():
        model = DenseGridLightning.load_from_checkpoint(best_path)

    metrics: dict[str, float | int | str] = {}
    if data.spot_to_grids:
        spot_expr_pred, spot_type_pred = predict_supervised(
            model=model,
            data=data,
            batch_size=args.batch_size,
            device=default_device(),
        )
        if spot_expr_pred is not None:
            metrics.update(
                compute_spot_expression_metrics(
                    predicted_spot_expression=spot_expr_pred,
                    data=data,
                    metrics_dir=metrics_dir,
                )
            )
        if spot_type_pred is not None:
            metrics.update(compute_spot_type_metrics(spot_type_pred, data, metrics_dir))
    if not args.no_export:
        metrics.update(
            export_grid_predictions(
                model=model,
                data=data,
                output_h5=run_dir / "grid_predictions.h5",
                max_grids_per_batch=args.export_max_grids_per_batch,
                device=default_device(),
            )
        )
    metrics["elapsed_seconds"] = float(time.time() - started)
    metrics["best_model"] = str(best_path)
    metrics["checkpoint"] = str(best_path)
    (run_dir / "summary.json").write_text(json.dumps(metrics, indent=2))
    metadata = {
        "dense_h5": str(args.dense_h5),
        "dense_feature_key": str(args.dense_feature_key),
        "st_h5ad": str(args.st_h5ad),
        "deconv_csv": str(args.deconv_csv),
        "gene_list": str(args.gene_list),
        "sample_id": args.sample_id,
        "architecture": "projection_heads",
        "head_mode": "both",
        "use_set_transformer": False,
        "use_gene_head": True,
        "use_type_head": True,
        "expression_target": "raw_counts",
        "expression_loss": "huber_log1p_sum_grid_vs_raw_counts",
        "cuda_device": None if getattr(args, "cuda_device", None) is None else int(args.cuda_device),
        "gpu_selection": getattr(args, "gpu_selection", None),
        "input_dim": int(data.input_dim),
        "projection_dim": int(args.projection_dim),
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "early_stop_min_delta": float(args.early_stop_min_delta),
        "early_stop_monitor": "train_loss_epoch" if args.train_all else "val_loss",
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "lambda_deconv": float(args.lambda_deconv),
        "type_confidence_alpha": float(args.type_confidence_alpha),
        "gene_loss_reduction": str(args.gene_loss_reduction),
        "loss": str(getattr(args, "loss", None) or "custom"),
        "enable_progress_bar": bool(args.enable_progress_bar),
        "progress_refresh_rate": int(args.progress_refresh_rate),
        "val_fraction": float(args.val_fraction),
        "train_all": bool(args.train_all),
        "seed": int(args.seed),
        "n_grids": int(data.features.shape[0]),
        "n_tissue_grids": int(data.is_tissue.sum()),
        "n_supervised_spots": int(len(data.spot_to_grids)),
        **spot_grid_summary,
        "n_genes": int(len(data.gene_names)),
        "n_cell_types": int(len(data.cell_type_names)),
        "gene_names": data.gene_names,
        "cell_type_names": data.cell_type_names,
        "best_model": str(best_path),
        "model_summary": model_summary,
        "metrics": metrics,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    if bool(getattr(args, "stage_log_stdout", False)):
        print(json.dumps({"stage": "done", **metrics}, indent=2), flush=True)
    return metrics
