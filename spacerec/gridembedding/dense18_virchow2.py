#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for CLI use.
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
SPACEREC_ROOT = SCRIPT_DIR.parents[1]
INPUT_DIR = SPACEREC_ROOT / "resources" / "crc"
OUTPUT_DIR = SPACEREC_ROOT / "results" / "crc" / "gridembedding"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from . import _virchow2_utils as base
except ImportError:
    import _virchow2_utils as base


FEATURE_KEY = "virchow2_token_tile_neighbor_concat6400"
TOKEN_DIM = 1280
TILE_DIM = 2560
NEIGHBOR_TILE_DIM = 2560
NEIGHBOR_STD_DIM = 2560
STAGE_LOG_STDOUT = False


def feature_dim_for_mode(mode: str) -> int:
    if mode in {"mean", "delta"}:
        return TOKEN_DIM + TILE_DIM + NEIGHBOR_TILE_DIM
    if mode == "delta_std":
        return TOKEN_DIM + TILE_DIM + NEIGHBOR_TILE_DIM + NEIGHBOR_STD_DIM
    raise ValueError(f"Unknown neighbor feature mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export dense18 Virchow2 token+tile+neighbor-tile embeddings. "
            "Each dense grid receives token1280 + same-patch whole tile2560 + "
            "mean adjacent-patch whole tile2560, then Hann averaging is applied "
            "to the full 6400-dimensional vector."
        )
    )
    parser.add_argument("--he-image", type=Path, default=INPUT_DIR / "he" / "he.tif")
    parser.add_argument("--positions-csv", type=Path, default=INPUT_DIR / "visium" / "tissue_positions.csv")
    parser.add_argument("--scalefactors-json", type=Path, default=INPUT_DIR / "visium" / "scalefactors_json.json")
    parser.add_argument("--thumbnail-png", type=Path, default=INPUT_DIR / "visium" / "spatial" / "tissue_hires_image.png")
    parser.add_argument(
        "--output-h5",
        type=Path,
        default=OUTPUT_DIR / "dense18_virchow2_crc.h5",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=OUTPUT_DIR / "dense18_virchow2_crc_summary.json",
    )
    parser.add_argument(
        "--patch-metadata-h5",
        type=Path,
        default=OUTPUT_DIR / "patch_metadata_dense18_virchow2_token_tile_neighbor_concat6400_288to224_stride72.h5",
    )
    parser.add_argument("--sample-id", default="COLON_P2")
    parser.add_argument("--feature-key", default=FEATURE_KEY)
    parser.add_argument(
        "--neighbor-feature-mode",
        choices=["mean", "delta", "delta_std"],
        default="mean",
        help=(
            "mean: token + center tile + neighbor mean; "
            "delta: token + center tile + (neighbor mean - center tile); "
            "delta_std: token + center tile + delta + neighbor std."
        ),
    )
    parser.add_argument("--planned-projection-dim", type=int, default=3584)
    parser.add_argument("--patch-size", type=int, default=288)
    parser.add_argument("--model-input-size", type=int, default=224)
    parser.add_argument("--grid-size", type=int, default=18)
    parser.add_argument("--stride", type=int, default=72)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--filter-grid-tissue-boundary", action="store_true", default=True)
    parser.add_argument("--no-filter-grid-tissue-boundary", dest="filter_grid_tissue_boundary", action="store_false")
    parser.add_argument("--progress-json", type=Path, default=None)
    parser.add_argument("--progress-every-patches", type=int, default=100)
    parser.add_argument("--enable-progress-bar", action="store_true")
    parser.add_argument("--progress-refresh-rate", type=float, default=1.0)
    parser.add_argument("--stage-log-stdout", action="store_true")
    parser.add_argument("--max-patches", type=int, default=None)
    return parser.parse_args()


def emit(stage: str, progress_json: Path | None = None, *, stdout: bool | None = None, **payload: object) -> None:
    record = {
        "stage": stage,
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        **payload,
    }
    if stdout is None:
        stdout = STAGE_LOG_STDOUT
    if stdout:
        print(json.dumps(record), flush=True)
    if progress_json is not None:
        progress_json.parent.mkdir(parents=True, exist_ok=True)
        tmp = progress_json.with_name(progress_json.name + ".tmp")
        tmp.write_text(json.dumps(record, indent=2))
        tmp.replace(progress_json)


def progress_iter(
    iterable,
    *,
    enabled: bool,
    total: int,
    desc: str,
    unit: str,
    refresh_rate: float,
):
    if enabled and tqdm is not None:
        return tqdm(
            iterable,
            total=total,
            desc=desc,
            unit=unit,
            mininterval=max(0.0, float(refresh_rate)),
            leave=True,
        )
    return iterable


def build_neighbor_tile_stats(
    patch_bbox: np.ndarray,
    tile_embeddings: np.ndarray,
    stride: int,
    compute_std: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    patch_bbox = np.asarray(patch_bbox, dtype=np.int32)
    tile_embeddings = np.asarray(tile_embeddings, dtype=np.float32)
    stride = int(stride)
    coord_to_index = {
        (int(x0), int(y0)): int(index)
        for index, (x0, y0, _x1, _y1) in enumerate(patch_bbox)
    }
    neighbor_mean = np.empty_like(tile_embeddings)
    neighbor_std = np.zeros_like(tile_embeddings) if compute_std else None
    neighbor_count = np.zeros(patch_bbox.shape[0], dtype=np.int32)
    offsets = [
        (dx, dy)
        for dy in (-stride, 0, stride)
        for dx in (-stride, 0, stride)
        if not (dx == 0 and dy == 0)
    ]
    for index, (x0, y0, _x1, _y1) in enumerate(patch_bbox):
        neighbors = [
            coord_to_index[(int(x0) + dx, int(y0) + dy)]
            for dx, dy in offsets
            if (int(x0) + dx, int(y0) + dy) in coord_to_index
        ]
        if neighbors:
            neighbor_values = tile_embeddings[np.asarray(neighbors, dtype=np.int64)]
            neighbor_mean[index] = neighbor_values.mean(axis=0)
            if neighbor_std is not None:
                neighbor_std[index] = neighbor_values.std(axis=0)
            neighbor_count[index] = len(neighbors)
        else:
            neighbor_mean[index] = tile_embeddings[index]
    return neighbor_mean.astype(np.float32, copy=False), neighbor_std, neighbor_count


def make_summary(
    args: argparse.Namespace,
    patch_summary: dict[str, object],
    patch_bbox: np.ndarray,
    unique_bbox: np.ndarray,
    raw_bbox: np.ndarray,
    spot_index: np.ndarray,
    complete: bool,
    started: float,
) -> dict[str, object]:
    feature_dim = feature_dim_for_mode(str(args.neighbor_feature_mode))
    return {
        "output_h5": str(args.output_h5),
        "summary_json": str(args.summary_json),
        "patch_summary": patch_summary,
        "n_export_patches": int(patch_bbox.shape[0]),
        "n_raw_token_grids": int(raw_bbox.shape[0]),
        "n_grids": int(unique_bbox.shape[0]),
        "n_supervised_grids": int((spot_index >= 0).sum()),
        "n_supervised_spots": int(np.unique(spot_index[spot_index >= 0]).size),
        "feature_key": str(args.feature_key),
        "feature_dim": feature_dim,
        "token_feature_dim": TOKEN_DIM,
        "tile_feature_dim": TILE_DIM,
        "neighbor_tile_feature_dim": NEIGHBOR_TILE_DIM,
        "neighbor_std_feature_dim": NEIGHBOR_STD_DIM if str(args.neighbor_feature_mode) == "delta_std" else 0,
        "projection_dim_planned": int(args.planned_projection_dim),
        "patch_size_fullres": int(args.patch_size),
        "model_input_size": int(args.model_input_size),
        "grid_size_fullres": int(args.grid_size),
        "stride_fullres": int(args.stride),
        "token_layout": "16x16",
        "token_patch_size_input_px": int(args.model_input_size) // 16,
        "filter_grid_tissue_boundary": bool(args.filter_grid_tissue_boundary),
        "neighbor_feature_mode": str(args.neighbor_feature_mode),
        "neighbor_rule": "available retained tissue patches at stride offsets in the 3x3 neighborhood, excluding center; fallback mean to self tile and std to zero if isolated",
        "hann_rule": f"Hann-like weights are applied after concatenating the full {feature_dim}-dimensional vector",
        "complete": bool(complete),
        "elapsed_seconds": float(time.time() - started),
    }


def main() -> None:
    global STAGE_LOG_STDOUT
    args = parse_args()
    STAGE_LOG_STDOUT = bool(args.stage_log_stdout)
    started = time.time()
    feature_dim = feature_dim_for_mode(str(args.neighbor_feature_mode))
    progress_json = args.progress_json or args.output_h5.with_name(args.output_h5.stem + "_progress.json")
    args.progress_json = progress_json

    emit("start", progress_json, args={k: str(v) for k, v in vars(args).items()})
    if args.output_h5.exists():
        if args.force:
            args.output_h5.unlink()
        else:
            raise FileExistsError(f"{args.output_h5} exists; pass --force to replace it.")
    if args.force and progress_json.exists():
        progress_json.unlink()

    emit("build_patch_metadata_start", progress_json)
    patch_summary, patch_bbox = base.bbox_from_patch_metadata(args)
    if args.max_patches is not None:
        max_patches = max(1, int(args.max_patches))
        original_patch_count = int(patch_bbox.shape[0])
        patch_bbox = patch_bbox[:max_patches].copy()
        patch_summary["n_retained_patches_before_max_patches"] = original_patch_count
        patch_summary["n_retained_patches"] = int(patch_bbox.shape[0])
        patch_summary["max_patches_applied"] = max_patches
    emit(
        "build_patch_metadata_done",
        progress_json,
        n_export_patches=int(patch_bbox.shape[0]),
        patch_summary=patch_summary,
    )

    emit("build_grid_set_start", progress_json, n_export_patches=int(patch_bbox.shape[0]))
    unique_bbox, center_xy, raw_bbox, raw_patch_index, raw_token_linear, raw_weight, raw_to_unique, duplicate_count = base.build_grid_set(
        args,
        patch_bbox,
        patch_summary["fullres_bbox"],
    )
    emit(
        "build_grid_set_done",
        progress_json,
        n_raw_token_grids=int(raw_bbox.shape[0]),
        n_grids=int(unique_bbox.shape[0]),
    )

    emit("assign_spots_start", progress_json, n_grids=int(unique_bbox.shape[0]))
    _, spot_barcodes, spot_ids, spot_xy, spot_diameter = base.shared.load_spot_table(
        args.positions_csv,
        args.scalefactors_json,
        args.sample_id,
    )
    spot_index, nearest_spot_index, nearest_spot_distance = base.shared.assign_spots(center_xy, spot_xy, spot_diameter)
    is_supervised_grid = spot_index >= 0
    is_tissue = np.ones(unique_bbox.shape[0], dtype=np.bool_)
    grid_tissue_fraction = np.ones(unique_bbox.shape[0], dtype=np.float32)

    summary = make_summary(args, patch_summary, patch_bbox, unique_bbox, raw_bbox, spot_index, False, started)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2))
    emit("metadata_ready", progress_json, **summary)
    if args.metadata_only:
        return

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    emit("load_model_start", progress_json, device=str(device))
    model = base.Virchow2TokenEncoder().to(device).eval()
    emit("load_model_done", progress_json, device=str(device))
    emit("he_image_probe_start", progress_json, he_image=str(args.he_image))
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(args.he_image) as image:
        he_size = tuple(int(value) for value in image.size)
    emit("he_image_probe_done", progress_json, image_size=list(he_size), read_rule="lazy_patch_crop")

    dataset = base.PatchDataset(args.he_image, patch_bbox, model.mean, model.std, int(args.model_input_size))
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.num_workers) > 0,
    )
    use_autocast = device.type == "cuda"
    progress_every = max(1, int(args.progress_every_patches))

    emit("tile_cache_start", progress_json, total_patches=int(patch_bbox.shape[0]), tile_dim=TILE_DIM)
    tile_embeddings = np.zeros((patch_bbox.shape[0], TILE_DIM), dtype=np.float32)
    processed = 0
    last_emit = time.time()
    with torch.inference_mode():
        for images, indices in progress_iter(
            loader,
            enabled=bool(args.enable_progress_bar),
            total=len(loader),
            desc="Tile embeddings",
            unit="batch",
            refresh_rate=float(args.progress_refresh_rate),
        ):
            batch_started = time.time()
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_autocast):
                _token_values, tile_values = model(images)
            tile_np = tile_values.detach().cpu().float().numpy().astype(np.float32)
            indices_np = indices.cpu().numpy().astype(np.int64)
            tile_embeddings[indices_np] = tile_np
            processed += int(indices.numel())
            now = time.time()
            if processed <= progress_every or processed % progress_every == 0 or now - last_emit >= 60:
                emit(
                    "tile_cache_progress",
                    progress_json,
                    processed_patches=int(processed),
                    total_patches=int(patch_bbox.shape[0]),
                    batch_seconds=float(now - batch_started),
                    patches_per_second=float(processed / max(now - started, 1e-6)),
                )
                last_emit = now

    emit(
        "neighbor_tile_stats_start",
        progress_json,
        total_patches=int(patch_bbox.shape[0]),
        stride=int(args.stride),
        compute_std=str(args.neighbor_feature_mode) == "delta_std",
    )
    neighbor_tile_mean, neighbor_tile_std, neighbor_count = build_neighbor_tile_stats(
        patch_bbox,
        tile_embeddings,
        int(args.stride),
        compute_std=str(args.neighbor_feature_mode) == "delta_std",
    )
    emit(
        "neighbor_tile_stats_done",
        progress_json,
        neighbor_count_min=int(neighbor_count.min()),
        neighbor_count_median=float(np.median(neighbor_count)),
        neighbor_count_max=int(neighbor_count.max()),
        isolated_patches=int((neighbor_count == 0).sum()),
    )

    emit("prepare_accumulators_start", progress_json, n_grids=int(unique_bbox.shape[0]), feature_dim=feature_dim)
    feature_sum = np.zeros((unique_bbox.shape[0], feature_dim), dtype=np.float32)
    weight_sum = np.zeros(unique_bbox.shape[0], dtype=np.float32)
    view_count = np.zeros(unique_bbox.shape[0], dtype=np.int32)
    emit("prepare_accumulators_done", progress_json)

    emit("prepare_patch_offsets_start", progress_json, n_raw_token_grids=int(raw_patch_index.shape[0]))
    if raw_patch_index.size > 1 and bool(np.any(raw_patch_index[1:] < raw_patch_index[:-1])):
        emit("sort_raw_patch_index_start", progress_json, n_raw_token_grids=int(raw_patch_index.shape[0]))
        raw_order: np.ndarray | None = np.argsort(raw_patch_index, kind="stable")
        raw_patch_for_offsets = raw_patch_index[raw_order]
        emit("sort_raw_patch_index_done", progress_json)
    else:
        raw_order = None
        raw_patch_for_offsets = raw_patch_index
    raw_counts_by_patch = np.bincount(raw_patch_for_offsets.astype(np.int64), minlength=patch_bbox.shape[0]).astype(np.int64)
    raw_offsets_by_patch = np.r_[0, np.cumsum(raw_counts_by_patch)]
    emit(
        "prepare_patch_offsets_done",
        progress_json,
        raw_patch_index_sorted=raw_order is None,
        n_nonempty_patches=int((raw_counts_by_patch > 0).sum()),
    )

    processed = 0
    total_views = 0
    last_emit = time.time()
    emit(
        "extract_loop_start",
        progress_json,
        total_patches=int(patch_bbox.shape[0]),
        batch_size=max(1, int(args.batch_size)),
        num_workers=int(args.num_workers),
        feature_dim=feature_dim,
        neighbor_feature_mode=str(args.neighbor_feature_mode),
    )
    with torch.inference_mode():
        for images, indices in progress_iter(
            loader,
            enabled=bool(args.enable_progress_bar),
            total=len(loader),
            desc="Grid embeddings",
            unit="batch",
            refresh_rate=float(args.progress_refresh_rate),
        ):
            batch_started = time.time()
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_autocast):
                token_values, _tile_values = model(images)
            token_np = token_values.detach().cpu().float().numpy().astype(np.float32)
            batch_views = 0
            for batch_offset, patch_index in enumerate(indices.cpu().numpy().astype(np.int64)):
                start = int(raw_offsets_by_patch[int(patch_index)])
                stop = int(raw_offsets_by_patch[int(patch_index) + 1])
                if stop <= start:
                    continue
                if raw_order is None:
                    target = raw_to_unique[start:stop]
                    token_linear = raw_token_linear[start:stop]
                    weight = raw_weight[start:stop]
                else:
                    raw_indices = raw_order[start:stop]
                    target = raw_to_unique[raw_indices]
                    token_linear = raw_token_linear[raw_indices]
                    weight = raw_weight[raw_indices]
                token_values_patch = token_np[batch_offset, token_linear, :]
                n_views = token_values_patch.shape[0]
                tile_values_patch = np.broadcast_to(tile_embeddings[patch_index][None, :], (n_views, TILE_DIM))
                if str(args.neighbor_feature_mode) == "mean":
                    neighbor_values_patch = np.broadcast_to(neighbor_tile_mean[patch_index][None, :], (n_views, NEIGHBOR_TILE_DIM))
                    values = np.concatenate([token_values_patch, tile_values_patch, neighbor_values_patch], axis=1)
                elif str(args.neighbor_feature_mode) == "delta":
                    delta_patch = neighbor_tile_mean[patch_index] - tile_embeddings[patch_index]
                    delta_values_patch = np.broadcast_to(delta_patch[None, :], (n_views, NEIGHBOR_TILE_DIM))
                    values = np.concatenate([token_values_patch, tile_values_patch, delta_values_patch], axis=1)
                else:
                    if neighbor_tile_std is None:
                        raise RuntimeError("neighbor_tile_std is required for delta_std mode.")
                    delta_patch = neighbor_tile_mean[patch_index] - tile_embeddings[patch_index]
                    delta_values_patch = np.broadcast_to(delta_patch[None, :], (n_views, NEIGHBOR_TILE_DIM))
                    std_values_patch = np.broadcast_to(neighbor_tile_std[patch_index][None, :], (n_views, NEIGHBOR_STD_DIM))
                    values = np.concatenate([token_values_patch, tile_values_patch, delta_values_patch, std_values_patch], axis=1)
                if np.unique(target).size == target.size:
                    feature_sum[target] += values * weight[:, None]
                    weight_sum[target] += weight
                    view_count[target] += 1
                else:
                    np.add.at(feature_sum, target, values * weight[:, None])
                    np.add.at(weight_sum, target, weight)
                    np.add.at(view_count, target, 1)
                batch_views += int(target.size)
                total_views += int(target.size)
            processed += int(indices.numel())
            now = time.time()
            if processed <= progress_every or processed % progress_every == 0 or now - last_emit >= 60:
                emit(
                    "extract_progress",
                    progress_json,
                    processed_patches=int(processed),
                    total_patches=int(patch_bbox.shape[0]),
                    total_token_views_used=int(total_views),
                    covered_grids=int((view_count > 0).sum()),
                    batch_token_views=int(batch_views),
                    batch_seconds=float(now - batch_started),
                    patches_per_second=float(processed / max(now - started, 1e-6)),
                )
                last_emit = now

    if bool(np.any(view_count == 0)):
        raise RuntimeError(f"{int((view_count == 0).sum())} dense18 grids received no token views.")
    emit("normalize_features_start", progress_json, n_grids=int(feature_sum.shape[0]))
    feature_sum /= weight_sum[:, None]
    emit("normalize_features_done", progress_json)

    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    emit("write_h5_start", progress_json, output_h5=str(args.output_h5), n_grids=int(feature_sum.shape[0]))
    with h5py.File(args.output_h5, "w") as handle:
        chunks = (min(512, feature_sum.shape[0]), feature_sum.shape[1])
        handle.create_dataset(args.feature_key, data=feature_sum, chunks=chunks, compression="lzf")
        handle.create_dataset("bbox_xyxy", data=unique_bbox.astype(np.int32), compression="gzip")
        handle.create_dataset("center_xy", data=center_xy.astype(np.float32), compression="gzip")
        handle.create_dataset("grid_bbox", data=unique_bbox.astype(np.int32), compression="gzip")
        handle.create_dataset("grid_center_xy", data=center_xy.astype(np.float32), compression="gzip")
        handle.create_dataset("duplicate_count", data=duplicate_count.astype(np.int32), compression="gzip")
        handle.create_dataset("n_views_averaged", data=view_count.astype(np.int32), compression="gzip")
        handle.create_dataset("weight_sum", data=weight_sum.astype(np.float32), compression="gzip")
        handle.create_dataset("grid_tissue_fraction", data=grid_tissue_fraction, compression="gzip")
        handle.create_dataset("is_tissue", data=is_tissue, compression="gzip")
        handle.create_dataset("is_supervised_grid", data=is_supervised_grid, compression="gzip")
        handle.create_dataset("spot_index", data=spot_index.astype(np.int32), compression="gzip")
        handle.create_dataset("nearest_spot_index", data=nearest_spot_index.astype(np.int32), compression="gzip")
        handle.create_dataset("nearest_spot_distance", data=nearest_spot_distance.astype(np.float32), compression="gzip")
        handle.create_dataset("spot_xy", data=spot_xy.astype(np.float32), compression="gzip")
        base.write_string_dataset(handle, "spot_barcode", spot_barcodes)
        base.write_string_dataset(handle, "spot_id", spot_ids)
        handle.create_dataset("patch_bbox", data=patch_bbox.astype(np.int32), compression="gzip")
        handle.create_dataset("patch_neighbor_count", data=neighbor_count.astype(np.int32), compression="gzip")
        handle.attrs["feature_key"] = str(args.feature_key)
        handle.attrs["feature_dim"] = feature_dim
        handle.attrs["token_feature_dim"] = TOKEN_DIM
        handle.attrs["tile_feature_dim"] = TILE_DIM
        handle.attrs["neighbor_tile_feature_dim"] = NEIGHBOR_TILE_DIM
        handle.attrs["neighbor_std_feature_dim"] = NEIGHBOR_STD_DIM if str(args.neighbor_feature_mode) == "delta_std" else 0
        handle.attrs["neighbor_feature_mode"] = str(args.neighbor_feature_mode)
        handle.attrs["embedding"] = f"official_virchow2_token_tile_{args.neighbor_feature_mode}_dense18_288to224_stride72_hann"
        handle.attrs["model"] = "paige-ai/Virchow2"
        handle.attrs["model_architecture"] = "ViT-H/14"
        handle.attrs["model_input_size"] = int(args.model_input_size)
        handle.attrs["token_layout"] = "16x16"
        handle.attrs["token_patch_size_input_px"] = int(args.model_input_size) // 16
        handle.attrs["patch_size_fullres"] = int(args.patch_size)
        handle.attrs["grid_size_fullres"] = int(args.grid_size)
        handle.attrs["stride_fullres"] = int(args.stride)
        handle.attrs["blend"] = f"raised_cosine_hann_like_center_weight_after_concat{feature_dim}"
        handle.attrs["selection_rule"] = "Virchow2 dense18 patch token plus same-patch tile embedding plus adjacent retained-patch context inside tissue_positions bbox"
        handle.attrs["resize_rule"] = "crop 288x288 full-resolution pixels, bicubic resize to 224x224 model input, map 16x16 tokens back to 18x18 full-resolution grids"
        handle.attrs["tile_embedding_rule"] = "concat(class_token, mean(patch_tokens)) from each 288px-to-224px Virchow2 forward pass"
        handle.attrs["neighbor_tile_rule"] = "mean, mean-self delta, or delta+std of available retained tissue patch tile embeddings at stride offsets in 3x3 neighborhood excluding center; fallback mean to self tile and std to zero when isolated"
        handle.attrs["duplicate_rule"] = "same final dense18 grid bbox weighted-averaged across overlapping 288px patch views after feature concatenation"
        handle.attrs["coordinate_system"] = "full-resolution he.tif coordinates; no offset"
        handle.attrs["spot_diameter_fullres"] = float(spot_diameter)
    emit("write_h5_done", progress_json, output_h5=str(args.output_h5))

    summary.update(
        {
            "complete": True,
            "n_total_token_views_used": int(total_views),
            "n_views_averaged_min": int(view_count.min()),
            "n_views_averaged_median": float(np.median(view_count)),
            "n_views_averaged_max": int(view_count.max()),
            "weight_sum_min": float(weight_sum.min()),
            "weight_sum_median": float(np.median(weight_sum)),
            "weight_sum_max": float(weight_sum.max()),
            "neighbor_count_min": int(neighbor_count.min()),
            "neighbor_count_median": float(np.median(neighbor_count)),
            "neighbor_count_max": int(neighbor_count.max()),
            "isolated_patches": int((neighbor_count == 0).sum()),
            "elapsed_seconds": float(time.time() - started),
            "device": str(device),
        }
    )
    args.summary_json.write_text(json.dumps(summary, indent=2))
    emit("done", progress_json, **summary)


if __name__ == "__main__":
    main()
