#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


SCRIPT_DIR = Path(__file__).resolve().parent
SPACEREC_ROOT = SCRIPT_DIR.parents[1]
INPUT_DIR = SPACEREC_ROOT / "resources" / "crc"
OUTPUT_DIR = SPACEREC_ROOT / "results" / "crc" / "gridembedding"
DIAGNOSTIC_DIR = SPACEREC_ROOT / "results" / "_gridembedding_diagnostics"


def normalize_spot_id(value: object, sample_id: str | None = None) -> str:
    text = value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
    if sample_id and text.endswith(f"_{sample_id}"):
        text = text[: -len(sample_id) - 1]
    return text


def write_string_dataset(handle: h5py.File, name: str, values: np.ndarray | list[str]) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    handle.create_dataset(name, data=np.asarray(values, dtype=object), dtype=dtype)


def image_size(path: Path) -> tuple[int, int]:
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as image:
        return int(image.size[0]), int(image.size[1])


def tissue_mask_from_rgb(array: np.ndarray) -> np.ndarray:
    rgb = array.astype(np.float32)
    max_rgb = rgb.max(axis=2)
    min_rgb = rgb.min(axis=2)
    mean_rgb = rgb.mean(axis=2)
    saturation = np.divide(
        max_rgb - min_rgb,
        np.maximum(max_rgb, 1.0),
        out=np.zeros_like(max_rgb),
        where=max_rgb > 0,
    )
    return (saturation > 0.05) & (mean_rgb < 245.0) & ((max_rgb - min_rgb) > 10.0)


def estimate_tissue_boundary_on_thumbnail(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    with Image.open(path) as src:
        image = src.convert("RGB")
    mask = tissue_mask_from_rgb(np.asarray(image, dtype=np.uint8))
    structure = np.ones((3, 3), dtype=bool)
    mask = ndi.binary_opening(mask, structure=structure, iterations=1)
    mask = ndi.binary_closing(mask, structure=structure, iterations=2)
    mask = ndi.binary_fill_holes(mask)
    return mask.astype(bool), (int(image.size[0]), int(image.size[1]))


def load_spot_table(positions_csv: Path, scalefactors_json: Path, sample_id: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float]:
    positions = pd.read_csv(positions_csv)
    used = positions.loc[positions["in_tissue"].astype(int) == 1].copy()
    if used.empty:
        raise ValueError("No in-tissue spots were found.")
    spot_xy = used[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=np.float32)
    barcodes = used["barcode"].astype(str).to_numpy(dtype=object)
    spot_ids = np.asarray([normalize_spot_id(item, sample_id) for item in barcodes], dtype=object)
    scale = json.loads(scalefactors_json.read_text())
    spot_diameter = float(scale["spot_diameter_fullres"])
    return used, barcodes, spot_ids, spot_xy, spot_diameter


def bbox_from_spots(spots: pd.DataFrame) -> dict[str, float]:
    x = spots["pxl_col_in_fullres"].astype(float).to_numpy()
    y = spots["pxl_row_in_fullres"].astype(float).to_numpy()
    return {
        "x_min": float(x.min()),
        "x_max": float(x.max()),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
    }


def infer_dataset_name(*paths: Path | str | None, sample_id: str | None = None) -> str:
    if sample_id:
        sample = str(sample_id).lower()
        if "breast" in sample:
            return "brca"
        if "colon" in sample or "crc" in sample:
            return "crc"
    for value in paths:
        if value is None:
            continue
        parts = [part.lower() for part in Path(value).parts]
        if "brca" in parts:
            return "brca"
        if "crc" in parts or "colon" in parts:
            return "crc"
    return "unknown"


def diagnostic_path(dataset: str, image_encoder: str, selection_name: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in selection_name)
    return DIAGNOSTIC_DIR / f"{dataset}_{image_encoder}_{safe}.png"


def _bbox_to_preview(box: np.ndarray, scale: float) -> list[int]:
    x0, y0, x1, y1 = [float(value) for value in box]
    return [
        int(round(x0 * scale)),
        int(round(y0 * scale)),
        int(round(x1 * scale)),
        int(round(y1 * scale)),
    ]


def _fullres_bbox_array(fullres_bbox: dict[str, float]) -> np.ndarray:
    return np.asarray(
        [
            float(fullres_bbox["x_min"]),
            float(fullres_bbox["y_min"]),
            float(fullres_bbox["x_max"]),
            float(fullres_bbox["y_max"]),
        ],
        dtype=np.float32,
    )


def write_patch_selection_diagnostic(
    *,
    he_image: Path | str | Image.Image,
    retained_bbox: np.ndarray,
    image_encoder: str,
    selection_name: str,
    dataset: str | None = None,
    sample_id: str | None = None,
    tissue_mask: np.ndarray | None = None,
    thumbnail_png: Path | str | None = None,
    scalefactors_json: Path | str | None = None,
    fullres_bbox: dict[str, float] | None = None,
    candidate_bbox: np.ndarray | None = None,
    clip_tissue_boundary_to_bbox: bool = False,
    output_png: Path | None = None,
    max_side: int = 2600,
) -> dict[str, object]:
    Image.MAX_IMAGE_PIXELS = None
    close_image = False
    if isinstance(he_image, Image.Image):
        image = he_image.convert("RGB")
        he_path = None
    else:
        he_path = Path(he_image)
        image = Image.open(he_path).convert("RGB")
        close_image = True
    try:
        width, height = image.size
        if tissue_mask is None:
            if thumbnail_png is None:
                raise ValueError("Provide either tissue_mask or thumbnail_png for diagnostic plotting.")
            tissue_mask, _thumb_size = estimate_tissue_boundary_on_thumbnail(Path(thumbnail_png))
        tissue_mask = np.asarray(tissue_mask, dtype=bool)
        if tissue_mask.ndim != 2:
            raise ValueError("tissue_mask must be a 2D boolean array.")

        edge = tissue_mask & ~ndi.binary_erosion(tissue_mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
        if clip_tissue_boundary_to_bbox and fullres_bbox is not None:
            x_scale = tissue_mask.shape[1] / float(width)
            y_scale = tissue_mask.shape[0] / float(height)
            box = _fullres_bbox_array(fullres_bbox)
            x0 = int(np.clip(np.floor(box[0] * x_scale), 0, tissue_mask.shape[1]))
            y0 = int(np.clip(np.floor(box[1] * y_scale), 0, tissue_mask.shape[0]))
            x1 = int(np.clip(np.ceil(box[2] * x_scale), 0, tissue_mask.shape[1]))
            y1 = int(np.clip(np.ceil(box[3] * y_scale), 0, tissue_mask.shape[0]))
            clipped = np.zeros_like(edge, dtype=bool)
            clipped[y0:y1, x0:x1] = edge[y0:y1, x0:x1]
            edge = clipped

        preview_scale = min(float(max_side) / float(max(width, height)), 1.0)
        preview_size = (max(1, int(round(width * preview_scale))), max(1, int(round(height * preview_scale))))
        preview = image.resize(preview_size, Image.Resampling.LANCZOS).convert("RGBA")

        edge_rgba = np.zeros((*edge.shape, 4), dtype=np.uint8)
        edge_rgba[edge] = np.array([0, 255, 255, 230], dtype=np.uint8)
        edge_layer = Image.fromarray(edge_rgba, mode="RGBA").resize(preview.size, Image.Resampling.NEAREST)
        preview = Image.alpha_composite(preview, edge_layer)

        overlay = Image.new("RGBA", preview.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        if candidate_bbox is not None:
            for box in np.asarray(candidate_bbox, dtype=np.float32):
                draw.rectangle(_bbox_to_preview(box, preview_scale), outline=(35, 120, 255, 35), width=1)
        for box in np.asarray(retained_bbox, dtype=np.float32):
            draw.rectangle(_bbox_to_preview(box, preview_scale), fill=(0, 0, 0, 45), outline=(245, 220, 60, 120), width=1)
        if fullres_bbox is not None:
            box = _fullres_bbox_array(fullres_bbox)
            width_px = max(2, int(round(3 * preview_scale)))
            draw.rectangle(_bbox_to_preview(box, preview_scale), outline=(255, 60, 55, 240), width=width_px)
        preview = Image.alpha_composite(preview, overlay)

        selected_dataset = dataset or infer_dataset_name(he_path, thumbnail_png, scalefactors_json, sample_id=sample_id)
        output_png = output_png or diagnostic_path(selected_dataset, image_encoder, selection_name)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        preview.save(output_png)
        summary = {
            "output_png": str(output_png),
            "dataset": selected_dataset,
            "image_encoder": str(image_encoder),
            "selection_name": str(selection_name),
            "he_image": str(he_path) if he_path is not None else "<PIL.Image>",
            "image_size": [int(width), int(height)],
            "preview_size": [int(preview.size[0]), int(preview.size[1])],
            "retained_patches": int(np.asarray(retained_bbox).shape[0]),
            "candidate_patches": None if candidate_bbox is None else int(np.asarray(candidate_bbox).shape[0]),
            "fullres_bbox": None if fullres_bbox is None else {key: float(value) for key, value in fullres_bbox.items()},
            "drawn_layers": {
                "cyan": "tissue boundary",
                "yellow": "retained patch boxes",
                "blue": "candidate patch boxes" if candidate_bbox is not None else None,
                "red": "Visium in_tissue bounding box" if fullres_bbox is not None else None,
            },
            "tissue_boundary_clipped_to_bbox": bool(clip_tissue_boundary_to_bbox),
        }
        output_png.with_suffix(".json").write_text(json.dumps(summary, indent=2))
        return summary
    finally:
        if close_image:
            image.close()


def build_patch_metadata(args: argparse.Namespace) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    width, height = image_size(args.he_image)
    spots, _, _, _, _ = load_spot_table(args.positions_csv, args.scalefactors_json, args.sample_id)
    bbox = bbox_from_spots(spots)
    scale = float(json.loads(args.scalefactors_json.read_text())["tissue_hires_scalef"])
    tissue_mask, thumb_size = estimate_tissue_boundary_on_thumbnail(args.thumbnail_png)

    patch_size = int(args.patch_size)
    stride = int(args.stride)
    trimmed_w = ((width - patch_size) // stride) * stride + patch_size
    trimmed_h = ((height - patch_size) // stride) * stride + patch_size

    kept_x: list[int] = []
    kept_y: list[int] = []
    bbox_candidate_count = 0
    for y1 in range(0, trimmed_h - patch_size + 1, stride):
        center_y = float(y1) + patch_size / 2.0
        if center_y < bbox["y_min"] or center_y > bbox["y_max"]:
            continue
        for x1 in range(0, trimmed_w - patch_size + 1, stride):
            center_x = float(x1) + patch_size / 2.0
            if center_x < bbox["x_min"] or center_x > bbox["x_max"]:
                continue
            bbox_candidate_count += 1
            tx = int(np.clip(round(center_x * scale), 0, tissue_mask.shape[1] - 1))
            ty = int(np.clip(round(center_y * scale), 0, tissue_mask.shape[0] - 1))
            if not bool(tissue_mask[ty, tx]):
                continue
            kept_x.append(int(x1))
            kept_y.append(int(y1))

    x1 = np.asarray(kept_x, dtype=np.int32)
    y1 = np.asarray(kept_y, dtype=np.int32)
    x2 = x1 + patch_size
    y2 = y1 + patch_size
    patch_bbox = np.stack([x1, y1, x2, y2], axis=1).astype(np.int32)
    patch_center = np.stack(
        [x1.astype(np.float32) + patch_size / 2.0, y1.astype(np.float32) + patch_size / 2.0],
        axis=1,
    )
    patch_corners = np.stack(
        [
            np.stack([x1, y1], axis=1),
            np.stack([x2, y1], axis=1),
            np.stack([x2, y2], axis=1),
            np.stack([x1, y2], axis=1),
        ],
        axis=1,
    ).astype(np.float32)

    args.patch_metadata_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.patch_metadata_h5, "w") as handle:
        handle.create_dataset("bbox_xyxy", data=patch_bbox, compression="gzip")
        handle.create_dataset("center_xy", data=patch_center.astype(np.float32), compression="gzip")
        handle.create_dataset("corners_xy", data=patch_corners, compression="gzip")
        handle.create_dataset("inside_bbox", data=np.ones(patch_bbox.shape[0], dtype=np.bool_), compression="gzip")
        handle.create_dataset("inside_tissue_boundary", data=np.ones(patch_bbox.shape[0], dtype=np.bool_), compression="gzip")
        handle.attrs["he_image"] = str(args.he_image)
        handle.attrs["image_size"] = json.dumps([int(width), int(height)])
        handle.attrs["patch_size"] = patch_size
        handle.attrs["stride"] = stride
        handle.attrs["selection_rule"] = "patch center inside in_tissue tissue_positions bbox and thumbnail-derived tissue boundary"
        handle.attrs["coordinate_system"] = "full-resolution he.tif coordinates; no offset"
        handle.attrs["grid_filter"] = "disabled"
        handle.attrs["fullres_bbox"] = json.dumps(bbox)
        handle.attrs["tissue_hires_scalef"] = scale

    summary = {
        "image_size": [int(width), int(height)],
        "thumbnail_size": [int(thumb_size[0]), int(thumb_size[1])],
        "tissue_hires_scalef": scale,
        "patch_size": patch_size,
        "stride": stride,
        "fullres_bbox": bbox,
        "n_positions_in_tissue": int(spots.shape[0]),
        "n_bbox_patch_candidates": int(bbox_candidate_count),
        "n_retained_patches": int(patch_bbox.shape[0]),
        "patch_metadata_h5": str(args.patch_metadata_h5),
        "selection_rule": "patch center inside bbox AND inside tissue boundary; no offset",
    }
    return summary, patch_bbox, patch_center.astype(np.float32), patch_corners


def lexsort_bbox(bbox_xyxy: np.ndarray) -> np.ndarray:
    return np.lexsort((bbox_xyxy[:, 2], bbox_xyxy[:, 3], bbox_xyxy[:, 0], bbox_xyxy[:, 1]))


def build_unique_grid_metadata(patch_bbox: np.ndarray, patch_size: int, grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_n = int(patch_size) // int(grid_size)
    if grid_n * int(grid_size) != int(patch_size):
        raise ValueError("patch_size must be divisible by grid_size")
    gy, gx = np.meshgrid(np.arange(grid_n, dtype=np.int32), np.arange(grid_n, dtype=np.int32), indexing="ij")
    gx = gx.ravel()
    gy = gy.ravel()
    all_parts: list[np.ndarray] = []
    for x1, y1, _, _ in patch_bbox.astype(np.int32):
        cell_x1 = int(x1) + gx * int(grid_size)
        cell_y1 = int(y1) + gy * int(grid_size)
        all_parts.append(
            np.stack(
                [cell_x1, cell_y1, cell_x1 + int(grid_size), cell_y1 + int(grid_size)],
                axis=1,
            ).astype(np.int32)
        )
    raw_bbox = np.concatenate(all_parts, axis=0) if all_parts else np.empty((0, 4), dtype=np.int32)
    order = lexsort_bbox(raw_bbox)
    sorted_bbox = raw_bbox[order]
    is_new = np.r_[True, np.any(sorted_bbox[1:] != sorted_bbox[:-1], axis=1)]
    starts = np.flatnonzero(is_new)
    duplicate_count = np.diff(np.r_[starts, sorted_bbox.shape[0]]).astype(np.int32)
    unique_bbox = sorted_bbox[starts].astype(np.int32)
    center_xy = np.stack(
        [
            unique_bbox[:, 0].astype(np.float32) + float(grid_size) / 2.0,
            unique_bbox[:, 1].astype(np.float32) + float(grid_size) / 2.0,
        ],
        axis=1,
    ).astype(np.float32)
    return unique_bbox, center_xy, duplicate_count


def assign_spots(center_xy: np.ndarray, spot_xy: np.ndarray, spot_diameter: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tree = cKDTree(spot_xy)
    distances, nearest = tree.query(center_xy.astype(np.float32), k=1)
    in_spot = distances <= float(spot_diameter) / 2.0
    spot_index = nearest.astype(np.int32)
    spot_index[~in_spot] = -1
    return spot_index, nearest.astype(np.int32), distances.astype(np.float32)
