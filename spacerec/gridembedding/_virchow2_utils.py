#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
SPACEREC_ROOT = SCRIPT_DIR.parents[1]
INPUT_DIR = SPACEREC_ROOT / "resources" / "crc"
OUTPUT_DIR = SPACEREC_ROOT / "results" / "crc" / "gridembedding"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from . import shared_bbox_boundary as shared
except ImportError:
    import shared_bbox_boundary as shared


def emit(stage: str, progress_json: Path | None = None, **payload: object) -> None:
    record = {
        "stage": stage,
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        **payload,
    }
    print(json.dumps(record), flush=True)
    if progress_json is not None:
        progress_json.parent.mkdir(parents=True, exist_ok=True)
        tmp = progress_json.with_name(progress_json.name + ".tmp")
        tmp.write_text(json.dumps(record, indent=2))
        tmp.replace(progress_json)


class PatchDataset(Dataset):
    def __init__(
        self,
        he_image: Path,
        bbox_xyxy: np.ndarray,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        model_input_size: int,
    ) -> None:
        self.he_image = Path(he_image)
        self.bbox_xyxy = np.asarray(bbox_xyxy, dtype=np.int32)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.model_input_size = int(model_input_size)
        self._image: Image.Image | None = None

    def __len__(self) -> int:
        return int(self.bbox_xyxy.shape[0])

    def image(self) -> Image.Image:
        if self._image is None:
            Image.MAX_IMAGE_PIXELS = None
            self._image = Image.open(self.he_image)
        return self._image

    def __getitem__(self, index: int):
        x0, y0, x1, y1 = [int(value) for value in self.bbox_xyxy[index]]
        image = self.image().crop((x0, y0, x1, y1)).convert("RGB").resize(
            (self.model_input_size, self.model_input_size),
            resample=Image.Resampling.BICUBIC,
        )
        values = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).permute(2, 0, 1).float().div_(255.0)
        values = (values - self.mean) / self.std
        return values, int(index)


class Virchow2TokenEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        import timm
        from timm.data import resolve_data_config
        from timm.layers import SwiGLUPacked

        self.model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        config = resolve_data_config(self.model.pretrained_cfg, model=self.model)
        self.mean = tuple(float(value) for value in config.get("mean", (0.485, 0.456, 0.406)))
        self.std = tuple(float(value) for value in config.get("std", (0.229, 0.224, 0.225)))
        self.register_tokens = 4
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(image)
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(output, dict):
            output = output.get("x", output.get("last_hidden_state", output.get("features")))
        if output is None or output.ndim != 3:
            raise RuntimeError(f"Expected Virchow2 token output with ndim=3, got {None if output is None else tuple(output.shape)}.")
        patch_tokens = output[:, 1 + self.register_tokens :]
        if patch_tokens.shape[1] != 256:
            raise RuntimeError(f"Expected 256 Virchow2 patch tokens, got {tuple(patch_tokens.shape)}.")
        class_token = output[:, 0]
        tile_embedding = torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)
        return patch_tokens, tile_embedding


def bbox_from_patch_metadata(args: argparse.Namespace) -> tuple[dict[str, object], np.ndarray]:
    patch_summary, patch_bbox, _patch_center, _patch_corners = shared.build_patch_metadata(args)
    dataset = shared.infer_dataset_name(args.he_image, args.patch_metadata_h5, sample_id=getattr(args, "sample_id", None))
    patch_summary["diagnostic_plot"] = shared.write_patch_selection_diagnostic(
        he_image=args.he_image,
        retained_bbox=patch_bbox,
        thumbnail_png=args.thumbnail_png,
        scalefactors_json=args.scalefactors_json,
        fullres_bbox=patch_summary["fullres_bbox"],
        image_encoder="dense18_virchow2",
        selection_name=Path(args.patch_metadata_h5).stem,
        dataset=dataset,
        sample_id=getattr(args, "sample_id", None),
        clip_tissue_boundary_to_bbox=True,
    )
    return patch_summary, patch_bbox


def tissue_mask_for_grid(args: argparse.Namespace, center_xy: np.ndarray) -> np.ndarray:
    if not bool(args.filter_grid_tissue_boundary):
        return np.ones(center_xy.shape[0], dtype=np.bool_)
    scale = float(json.loads(args.scalefactors_json.read_text())["tissue_hires_scalef"])
    tissue_mask, _thumb_size = shared.estimate_tissue_boundary_on_thumbnail(args.thumbnail_png)
    tx = np.clip(np.rint(center_xy[:, 0] * scale).astype(np.int64), 0, tissue_mask.shape[1] - 1)
    ty = np.clip(np.rint(center_xy[:, 1] * scale).astype(np.int64), 0, tissue_mask.shape[0] - 1)
    return tissue_mask[ty, tx].astype(np.bool_)


def build_grid_set(
    args: argparse.Namespace,
    patch_bbox: np.ndarray,
    fullres_bbox: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid_n = int(args.patch_size) // int(args.grid_size)
    if grid_n != 16 or grid_n * int(args.grid_size) != int(args.patch_size):
        raise ValueError("dense18 Virchow2 expects patch_size=288 and grid_size=18, producing 16x16 tokens.")

    gy, gx = np.meshgrid(np.arange(grid_n, dtype=np.int32), np.arange(grid_n, dtype=np.int32), indexing="ij")
    gx = gx.ravel()
    gy = gy.ravel()
    token_linear = (gy * grid_n + gx).astype(np.int32)
    center = (grid_n - 1) / 2.0
    scale = grid_n / 2.0
    wx = 0.5 * (1.0 + np.cos(np.pi * np.abs(gx.astype(np.float32) - center) / scale))
    wy = 0.5 * (1.0 + np.cos(np.pi * np.abs(gy.astype(np.float32) - center) / scale))
    token_weight = np.maximum(wx * wy, 1e-3).astype(np.float32)

    raw_parts: list[np.ndarray] = []
    raw_patch_parts: list[np.ndarray] = []
    raw_token_parts: list[np.ndarray] = []
    raw_weight_parts: list[np.ndarray] = []
    for patch_index, (x0, y0, _x1, _y1) in enumerate(patch_bbox.astype(np.int32)):
        grid_x0 = int(x0) + gx * int(args.grid_size)
        grid_y0 = int(y0) + gy * int(args.grid_size)
        raw = np.stack([grid_x0, grid_y0, grid_x0 + int(args.grid_size), grid_y0 + int(args.grid_size)], axis=1).astype(np.int32)
        raw_center_x = raw[:, 0].astype(np.float32) + float(args.grid_size) / 2.0
        raw_center_y = raw[:, 1].astype(np.float32) + float(args.grid_size) / 2.0
        in_bbox = (
            (raw_center_x >= float(fullres_bbox["x_min"]))
            & (raw_center_x <= float(fullres_bbox["x_max"]))
            & (raw_center_y >= float(fullres_bbox["y_min"]))
            & (raw_center_y <= float(fullres_bbox["y_max"]))
        )
        if not bool(np.any(in_bbox)):
            continue
        raw_parts.append(raw[in_bbox])
        raw_patch_parts.append(np.full(int(in_bbox.sum()), patch_index, dtype=np.int64))
        raw_token_parts.append(token_linear[in_bbox].astype(np.int32))
        raw_weight_parts.append(token_weight[in_bbox].astype(np.float32))

    raw_bbox = np.concatenate(raw_parts, axis=0)
    raw_patch_index = np.concatenate(raw_patch_parts).astype(np.int64)
    raw_token_linear = np.concatenate(raw_token_parts).astype(np.int32)
    raw_weight = np.concatenate(raw_weight_parts).astype(np.float32)

    unique_bbox, first_indices, inverse, duplicate_count = np.unique(
        raw_bbox,
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    order = np.lexsort((unique_bbox[:, 2], unique_bbox[:, 3], unique_bbox[:, 0], unique_bbox[:, 1]))
    remap = np.empty(order.shape[0], dtype=np.int64)
    remap[order] = np.arange(order.shape[0], dtype=np.int64)
    unique_bbox = unique_bbox[order].astype(np.int32)
    duplicate_count = duplicate_count[order].astype(np.int32)
    first_indices = first_indices[order].astype(np.int64)
    inverse = remap[inverse].astype(np.int64)

    center_xy = np.stack(
        [
            unique_bbox[:, 0].astype(np.float32) + float(args.grid_size) / 2.0,
            unique_bbox[:, 1].astype(np.float32) + float(args.grid_size) / 2.0,
        ],
        axis=1,
    ).astype(np.float32)
    keep_grid = tissue_mask_for_grid(args, center_xy)
    old_to_new = np.full(unique_bbox.shape[0], -1, dtype=np.int64)
    old_to_new[keep_grid] = np.arange(int(keep_grid.sum()), dtype=np.int64)
    keep_raw = keep_grid[inverse]
    raw_bbox = raw_bbox[keep_raw]
    raw_patch_index = raw_patch_index[keep_raw]
    raw_token_linear = raw_token_linear[keep_raw]
    raw_weight = raw_weight[keep_raw]
    raw_to_unique = old_to_new[inverse[keep_raw]].astype(np.int64)
    unique_bbox = unique_bbox[keep_grid]
    center_xy = center_xy[keep_grid]
    duplicate_count = np.bincount(raw_to_unique, minlength=unique_bbox.shape[0]).astype(np.int32)
    first_indices = np.full(unique_bbox.shape[0], -1, dtype=np.int64)
    first_indices[np.unique(raw_to_unique, return_index=True)[0]] = np.unique(raw_to_unique, return_index=True)[1]
    return unique_bbox, center_xy, raw_bbox, raw_patch_index, raw_token_linear, raw_weight, raw_to_unique, duplicate_count


def load_he_region(path: Path, bbox_xyxy: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as image:
        x0 = int(np.clip(int(bbox_xyxy[:, 0].min()), 0, image.size[0]))
        y0 = int(np.clip(int(bbox_xyxy[:, 1].min()), 0, image.size[1]))
        x1 = int(np.clip(int(bbox_xyxy[:, 2].max()), 0, image.size[0]))
        y1 = int(np.clip(int(bbox_xyxy[:, 3].max()), 0, image.size[1]))
        region = image.crop((x0, y0, x1, y1)).convert("RGB")
        return np.asarray(region, dtype=np.uint8), (x0, y0)


def write_string_dataset(handle: h5py.File, name: str, values: np.ndarray | list[str]) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    handle.create_dataset(name, data=np.asarray(values, dtype=object), dtype=dtype)
