from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl

from .heads import REGISTRY_KEYS
from .model import (
    DenseGridCore,
    DenseGridSpotData,
    default_device,
    iter_projection_chunks,
    scatter_mean,
    scatter_sum,
    write_sorted_rows,
    write_string_dataset,
)


class FactorizedRouter(nn.Module):
    """Router that reuses the ordinary SpaceRec type head logits."""

    def __init__(self, type_head: nn.Module, temperature: float = 1.0) -> None:
        super().__init__()
        self.type_head = type_head
        self.temperature = max(float(temperature), 1e-6)

    def logits(self, features: Tensor) -> Tensor:
        if hasattr(self.type_head, "norm") and hasattr(self.type_head, "block") and hasattr(self.type_head, "classifier"):
            residual_features = features + self.type_head.block(self.type_head.norm(features))
            classifier = self.type_head.classifier
            return classifier[1](classifier[0](residual_features))
        raise TypeError(f"Unsupported type head for factorized routing: {type(self.type_head)!r}")

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        logits = self.logits(features)
        p = torch.softmax(logits / self.temperature, dim=-1)
        return {"p": p, "logits": logits}


class ScalarScaleHead(nn.Module):
    """Non-negative grid-level transcript scale head."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, init_value: float = 128.0, hidden_layers: int = 1) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(int(input_dim))]
        in_dim = int(input_dim)
        for _ in range(int(hidden_layers)):
            layers.extend([nn.Linear(in_dim, int(hidden_dim)), nn.GELU()])
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)
        final = self.model[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, float(init_value))

    def forward(self, features: Tensor) -> Tensor:
        return F.softplus(self.model(features)) + 1e-8


class DeltaMuHead(nn.Module):
    """Bounded correction to cell-type reference gene profiles."""

    def __init__(
        self,
        input_dim: int,
        n_cell_types: int,
        n_genes: int,
        hidden_dim: int = 512,
        alpha: float = 0.5,
        hidden_layers: int = 1,
    ) -> None:
        super().__init__()
        self.n_cell_types = int(n_cell_types)
        self.n_genes = int(n_genes)
        self.alpha = float(alpha)
        layers: list[nn.Module] = [nn.LayerNorm(int(input_dim))]
        in_dim = int(input_dim)
        for _ in range(int(hidden_layers)):
            layers.extend([nn.Linear(in_dim, int(hidden_dim)), nn.GELU()])
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, self.n_cell_types * self.n_genes))
        self.model = nn.Sequential(*layers)
        final = self.model[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, features: Tensor) -> Tensor:
        raw = self.model(features).view(-1, self.n_cell_types, self.n_genes)
        return self.alpha * torch.tanh(raw)


class FinetuneMuGridCore(DenseGridCore):
    """Two-stage factorized grid model with an RCTD-reference expression profile."""

    def __init__(
        self,
        input_dim: int,
        n_genes: int,
        n_cell_types: int,
        projection_dim: int,
        mu_ref: Tensor | np.ndarray,
        p_temperature: float = 1.0,
        init_scale: float = 128.0,
        delta_alpha: float = 0.5,
        skip_input_projector: bool = True,
        type_head_hidden_layers: int = 2,
        factorized_head_hidden_layers: int = 3,
        scale_head_hidden_dim: int | None = 512,
        delta_head_hidden_dim: int | None = 2048,
        mu_eps: float = 1e-8,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            n_genes=n_genes,
            n_cell_types=n_cell_types,
            projection_dim=projection_dim,
            skip_input_projector=skip_input_projector,
            type_head_hidden_layers=type_head_hidden_layers,
        )
        mu_tensor = torch.as_tensor(mu_ref, dtype=torch.float32).clamp_min(0.0)
        if tuple(mu_tensor.shape) != (int(n_cell_types), int(n_genes)):
            raise ValueError(f"mu_ref must have shape [{n_cell_types}, {n_genes}], got {tuple(mu_tensor.shape)}.")
        row_sum = mu_tensor.sum(dim=1, keepdim=True).clamp_min(float(mu_eps))
        mu_tensor = mu_tensor / row_sum
        self.gene_head = None
        self.register_buffer("mu_ref", mu_tensor)
        self.register_buffer("log_mu_ref", torch.log(mu_tensor.clamp_min(float(mu_eps))))
        self.router = FactorizedRouter(self.type_head, temperature=p_temperature)
        scale_hidden = int(scale_head_hidden_dim) if scale_head_hidden_dim is not None else min(512, int(self.final_dim))
        delta_hidden = int(delta_head_hidden_dim) if delta_head_hidden_dim is not None else scale_hidden
        self.s_head = ScalarScaleHead(
            self.final_dim,
            hidden_dim=scale_hidden,
            init_value=init_scale,
            hidden_layers=int(factorized_head_hidden_layers),
        )
        self.delta_head = DeltaMuHead(
            self.final_dim,
            int(n_cell_types),
            int(n_genes),
            hidden_dim=delta_hidden,
            alpha=float(delta_alpha),
            hidden_layers=int(factorized_head_hidden_layers),
        )
        self.stage = "stage1"
        self.set_stage(self.stage)

    def set_stage(self, stage: str) -> None:
        if stage not in {"stage1", "stage2"}:
            raise ValueError("stage must be 'stage1' or 'stage2'.")
        self.stage = stage
        for param in self.parameters():
            param.requires_grad = False
        if stage == "stage1":
            for module in [self.input_projector, self.router.type_head]:
                for param in module.parameters():
                    param.requires_grad = True
        else:
            for module in [self.s_head, self.delta_head]:
                for param in module.parameters():
                    param.requires_grad = True

    def instance_forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        grid_bag_index = batch.get("grid_bag_index")
        if grid_bag_index is None:
            grid_bag_index = torch.zeros((batch["grid_x"].shape[0],), dtype=torch.long, device=batch["grid_x"].device)
        encoded = self.encode_grids(batch["grid_x"], grid_bag_index)
        router_output = self.router(encoded)
        p = router_output["p"]
        output: dict[str, Tensor] = {
            REGISTRY_KEYS.OUTPUT_EMBEDDING: encoded,
            "output_prob": p,
            "p": p,
            "p_logits": router_output["logits"],
        }
        if self.stage == "stage1":
            return output

        p_expr = p.detach()
        s = self.s_head(encoded)
        delta = self.delta_head(encoded)
        mu_tilde = torch.softmax(self.log_mu_ref.unsqueeze(0) + delta, dim=-1)
        phi = p_expr * s
        expr = torch.einsum("nk,nkg->ng", phi, mu_tilde)
        output.update(
            {
                REGISTRY_KEYS.OUTPUT_PREDICTION: expr,
                "s": s,
                "delta": delta,
                "mu_tilde": mu_tilde,
                "phi": phi,
            }
        )
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
        output["output_prob"] = scatter_mean(instance["output_prob"], batch["grid_bag_index"], n_spots)
        return output


class FinetuneMuLightning(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        n_genes: int,
        n_cell_types: int,
        projection_dim: int,
        mu_ref: Tensor | np.ndarray,
        lr: float,
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
        stage: str = "stage1",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["mu_ref"])
        if gene_loss_reduction not in {"mean", "sum"}:
            raise ValueError("gene_loss_reduction must be 'mean' or 'sum'.")
        if expr_loss not in {"log1p_huber", "mse", "poisson"}:
            raise ValueError("expr_loss must be one of: log1p_huber, mse, poisson.")
        self.model = FinetuneMuGridCore(
            input_dim=input_dim,
            n_genes=n_genes,
            n_cell_types=n_cell_types,
            projection_dim=projection_dim,
            mu_ref=mu_ref,
            p_temperature=p_temperature,
            init_scale=init_scale,
            delta_alpha=delta_alpha,
            skip_input_projector=skip_input_projector,
            type_head_hidden_layers=type_head_hidden_layers,
            factorized_head_hidden_layers=factorized_head_hidden_layers,
            scale_head_hidden_dim=scale_head_hidden_dim,
            delta_head_hidden_dim=delta_head_hidden_dim,
            mu_eps=mu_eps,
        )
        self.lr = float(lr)
        self.lambda_conf = float(lambda_conf)
        self.gene_loss_reduction = str(gene_loss_reduction)
        self.expr_loss = str(expr_loss)
        self.stage = str(stage)
        self.model.set_stage(self.stage)

    def set_stage(self, stage: str) -> None:
        self.stage = str(stage)
        self.hparams["stage"] = self.stage
        self.model.set_stage(self.stage)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.model.bag_forward(batch)

    def _type_loss(self, pred_prob: Tensor, target_prop: Tensor) -> Tensor:
        target_prop = target_prop.clamp_min(0.0)
        target_prop = target_prop / target_prop.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return F.kl_div(pred_prob.clamp_min(1e-8).log(), target_prop, reduction="batchmean")

    def _expr_loss_value(self, pred_expr: Tensor, target_expr: Tensor) -> Tensor:
        if self.expr_loss == "log1p_huber":
            return F.huber_loss(
                torch.log1p(pred_expr.clamp_min(0.0)),
                torch.log1p(target_expr.clamp_min(0.0)),
                reduction=self.gene_loss_reduction,
            )
        if self.expr_loss == "mse":
            return F.mse_loss(pred_expr, target_expr, reduction=self.gene_loss_reduction)
        if self.expr_loss == "poisson":
            return F.poisson_nll_loss(
                pred_expr.clamp_min(1e-8),
                target_expr,
                log_input=False,
                full=False,
                reduction=self.gene_loss_reduction,
            )
        raise ValueError(f"Unsupported expr_loss={self.expr_loss!r}.")

    def _confidence_loss(self, grid_prob: Tensor) -> Tensor:
        return -torch.mean(torch.log(grid_prob.clamp_min(1e-8).max(dim=1).values))

    def _step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        output = self(batch)
        batch_size = int(batch["target_spot_expr"].shape[0])
        zero = batch["target_spot_expr"].new_zeros(())
        expr_loss = zero
        type_loss = zero
        confidence_loss = zero
        if self.stage == "stage1":
            type_loss = self._type_loss(output["output_prob"], batch["target_spot_type_prop"])
            confidence_loss = self._confidence_loss(output["instance"]["output_prob"])
            loss = type_loss + self.lambda_conf * confidence_loss
        else:
            expr_loss = self._expr_loss_value(output[REGISTRY_KEYS.OUTPUT_PREDICTION], batch["target_spot_expr"])
            loss = expr_loss

        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}_expr_loss", expr_loss, batch_size=batch_size)
        self.log(f"{stage}_type_loss", type_loss, batch_size=batch_size)
        self.log(f"{stage}_confidence_loss", confidence_loss, batch_size=batch_size)
        if stage == "train":
            self.log("train_loss_epoch", loss, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("train_expr_loss_epoch", expr_loss, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("train_type_loss_epoch", type_loss, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("train_confidence_loss_epoch", confidence_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self):
        params = [param for param in self.parameters() if param.requires_grad]
        if not params:
            raise RuntimeError(f"No trainable parameters for stage={self.stage!r}.")
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)


def export_finetune_mu_grid_predictions(
    model: FinetuneMuLightning,
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
        s_ds = handle.create_dataset("s", shape=(n_grids, 1), dtype=np.float32, compression="gzip")
        delta_abs_mean_ds = handle.create_dataset("delta_abs_mean", shape=(n_grids,), dtype=np.float32, compression="gzip")
        delta_abs_max_ds = handle.create_dataset("delta_abs_max", shape=(n_grids,), dtype=np.float32, compression="gzip")
        mu_entropy_ds = handle.create_dataset("mu_tilde_entropy", shape=(n_grids, len(data.cell_type_names)), dtype=np.float32, compression="gzip")
        with torch.no_grad():
            for grid_indices, bag_index in iter_projection_chunks(n_grids, max_grids_per_batch):
                batch = {
                    "grid_x": torch.from_numpy(data.features[grid_indices]).float().to(device_obj),
                    "grid_bag_index": torch.from_numpy(bag_index).long().to(device_obj),
                }
                output = model.model.instance_forward(batch)
                write_order = np.argsort(grid_indices, kind="stable")
                write_indices = grid_indices[write_order]
                expr_values = output[REGISTRY_KEYS.OUTPUT_PREDICTION].detach().cpu().numpy().astype(np.float32)
                probs = output["output_prob"].detach().cpu().numpy().astype(np.float32)
                s_values = output["s"].detach().cpu().numpy().astype(np.float32)
                delta = output["delta"]
                mu_tilde = output["mu_tilde"].clamp_min(1e-8)
                delta_abs_mean = delta.abs().mean(dim=(1, 2)).detach().cpu().numpy().astype(np.float32)
                delta_abs_max = delta.abs().amax(dim=(1, 2)).detach().cpu().numpy().astype(np.float32)
                mu_entropy = (-(mu_tilde * mu_tilde.log()).sum(dim=-1)).detach().cpu().numpy().astype(np.float32)
                write_sorted_rows(expr_ds, write_indices, expr_values[write_order])
                write_sorted_rows(type_ds, write_indices, probs[write_order])
                write_sorted_rows(top1_ds, write_indices, probs.argmax(axis=1).astype(np.int16)[write_order])
                write_sorted_rows(s_ds, write_indices, s_values[write_order])
                write_sorted_rows(delta_abs_mean_ds, write_indices, delta_abs_mean[write_order])
                write_sorted_rows(delta_abs_max_ds, write_indices, delta_abs_max[write_order])
                write_sorted_rows(mu_entropy_ds, write_indices, mu_entropy[write_order])
        handle.attrs["architecture"] = "finetune_mu_factorized"
        handle.attrs["stage1"] = "projection_router_kl_confidence"
        handle.attrs["stage2"] = "frozen_projection_router_train_scale_delta"
        handle.attrs["expression_target"] = "raw_counts"
        handle.attrs["expression_loss"] = str(model.expr_loss)
    return {"grid_predictions_h5": str(output_h5)}


def export_finetune_mu_stage1_type_predictions(
    model: FinetuneMuLightning,
    data: DenseGridSpotData,
    output_csv: Path,
    max_grids_per_batch: int,
    device: str | None = None,
) -> dict[str, Any]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device or default_device())
    model = model.to(device_obj)
    model.set_stage("stage1")
    model.eval()
    n_grids = int(data.features.shape[0])
    type_prob = np.zeros((n_grids, len(data.cell_type_names)), dtype=np.float32)
    with torch.no_grad():
        for grid_indices, bag_index in iter_projection_chunks(n_grids, max_grids_per_batch):
            batch = {
                "grid_x": torch.from_numpy(data.features[grid_indices]).float().to(device_obj),
                "grid_bag_index": torch.from_numpy(bag_index).long().to(device_obj),
            }
            output = model.model.instance_forward(batch)
            probs = output["output_prob"].detach().cpu().numpy().astype(np.float32)
            type_prob[grid_indices] = probs

    top1_index = type_prob.argmax(axis=1)
    rows: dict[str, Any] = {
        "grid_id": [f"grid_{index:06d}" for index in range(n_grids)],
        "center_x": data.center_xy[:, 0].astype(np.float32),
        "center_y": data.center_xy[:, 1].astype(np.float32),
        "bbox_x0": data.bbox_xyxy[:, 0].astype(np.float32),
        "bbox_y0": data.bbox_xyxy[:, 1].astype(np.float32),
        "bbox_x1": data.bbox_xyxy[:, 2].astype(np.float32),
        "bbox_y1": data.bbox_xyxy[:, 3].astype(np.float32),
        "top1_type": [data.cell_type_names[int(index)] for index in top1_index],
        "top1_prob": type_prob[np.arange(n_grids), top1_index].astype(np.float32),
    }
    for index, name in enumerate(data.cell_type_names):
        rows[f"prob_{name}"] = type_prob[:, index]
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return {
        "stage1_grid_type_csv": str(output_csv),
        "n_stage1_grid_predictions": int(n_grids),
        "n_cell_types": int(len(data.cell_type_names)),
    }


def factorized_gradient_check(device: str | torch.device = "cpu") -> dict[str, Any]:
    device_obj = torch.device(device)
    torch.manual_seed(23)
    n_genes = 17
    n_cell_types = 5
    raw_mu = torch.rand(n_cell_types, n_genes, device=device_obj).clamp_min(1e-8)
    mu_ref = raw_mu / raw_mu.sum(dim=1, keepdim=True)
    model = FinetuneMuLightning(
        input_dim=13,
        n_genes=n_genes,
        n_cell_types=n_cell_types,
        projection_dim=11,
        mu_ref=mu_ref,
        lr=1e-4,
        lambda_conf=0.1,
        stage="stage1",
    ).to(device_obj)
    batch = {
        "grid_x": torch.randn(12, 13, device=device_obj),
        "grid_bag_index": torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=torch.long, device=device_obj),
        "target_spot_expr": torch.rand(4, n_genes, device=device_obj) * 20.0,
        "target_spot_type_prop": torch.softmax(torch.randn(4, n_cell_types, device=device_obj), dim=-1),
    }

    def grad_l2(parameters) -> float:
        total = 0.0
        for param in parameters:
            if param.grad is not None:
                total += float(param.grad.detach().pow(2).sum().cpu())
        return float(total ** 0.5)

    model.set_stage("stage1")
    model.zero_grad(set_to_none=True)
    loss1 = model._step(batch, "train")
    loss1.backward()
    stage1 = {
        "loss": float(loss1.detach().cpu()),
        "projector_grad_l2": grad_l2(model.model.input_projector.parameters()),
        "router_grad_l2": grad_l2(model.model.router.type_head.parameters()),
        "s_head_grad_l2": grad_l2(model.model.s_head.parameters()),
        "delta_head_grad_l2": grad_l2(model.model.delta_head.parameters()),
    }

    model.set_stage("stage2")
    model.zero_grad(set_to_none=True)
    loss2 = model._step(batch, "train")
    loss2.backward()
    output = model.model.instance_forward(batch)
    stage2 = {
        "loss": float(loss2.detach().cpu()),
        "projector_grad_l2": grad_l2(model.model.input_projector.parameters()),
        "router_grad_l2": grad_l2(model.model.router.type_head.parameters()),
        "s_head_grad_l2": grad_l2(model.model.s_head.parameters()),
        "delta_head_grad_l2": grad_l2(model.model.delta_head.parameters()),
        "grid_expression_shape": list(output[REGISTRY_KEYS.OUTPUT_PREDICTION].shape),
        "grid_type_prob_shape": list(output["output_prob"].shape),
        "delta_shape": list(output["delta"].shape),
        "mu_tilde_shape": list(output["mu_tilde"].shape),
    }
    if stage1["projector_grad_l2"] <= 0 or stage1["router_grad_l2"] <= 0:
        raise RuntimeError(f"Stage 1 router/projection gradients missing: {stage1}")
    if stage1["s_head_grad_l2"] != 0 or stage1["delta_head_grad_l2"] != 0:
        raise RuntimeError(f"Stage 1 expression heads received gradients: {stage1}")
    if stage2["projector_grad_l2"] != 0 or stage2["router_grad_l2"] != 0:
        raise RuntimeError(f"Stage 2 frozen router/projection received gradients: {stage2}")
    if stage2["s_head_grad_l2"] <= 0 or stage2["delta_head_grad_l2"] <= 0:
        raise RuntimeError(f"Stage 2 expression-head gradients missing: {stage2}")
    return {"device": str(device_obj), "stage1": stage1, "stage2": stage2}
