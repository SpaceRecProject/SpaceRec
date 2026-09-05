from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

try:
    import tifffile
except Exception:  # pragma: no cover
    tifffile = None


def _feature_key(path: Path) -> str:
    with h5py.File(path, "r") as handle:
        if "feature_key" in handle.attrs:
            return str(handle.attrs["feature_key"])
        candidates = [
            key
            for key, value in handle.items()
            if getattr(value, "ndim", None) == 2
            and key not in {"bbox_xyxy", "center_xy", "grid_bbox", "grid_center_xy", "spot_xy"}
        ]
    if len(candidates) != 1:
        raise KeyError(f"Cannot infer feature dataset key from {path}; candidates={candidates}")
    return candidates[0]


def _load_thumbnail(path: Path, max_side: int) -> tuple[Image.Image, tuple[int, int]]:
    if tifffile is not None:
        try:
            with tifffile.TiffFile(path) as tif:
                series = tif.series[0]
                full_shape = series.shape
                full_size = (int(full_shape[1]), int(full_shape[0]))
                levels = [series, *getattr(series, "levels", [])[1:]]
                selected = levels[-1]
                for level in levels:
                    shape = level.shape
                    if max(int(shape[0]), int(shape[1])) >= int(max_side):
                        selected = level
                    else:
                        break
                array = selected.asarray()
            image = Image.fromarray(array if array.ndim == 2 else array[..., :3]).convert("RGB")
            image.thumbnail((int(max_side), int(max_side)), Image.Resampling.LANCZOS)
            return image, full_size
        except Exception:
            pass

    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(path)
    full_size = tuple(map(int, image.size))
    if getattr(image, "n_frames", 1) > 1:
        selected = 0
        for frame in range(int(image.n_frames)):
            image.seek(frame)
            if max(image.size) >= int(max_side):
                selected = frame
            else:
                break
        image.seek(selected)
    image = image.copy().convert("RGB")
    image.thumbnail((int(max_side), int(max_side)), Image.Resampling.LANCZOS)
    return image, full_size


def _estimate_he_tissue_mask(
    image: Image.Image,
    *,
    saturation_threshold: float,
    value_threshold: float,
    white_value: float,
    white_saturation: float,
    min_object_pixels: int,
) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    saturation = (maxc - minc) / np.maximum(maxc, 1e-6)
    white = (maxc >= float(white_value)) & (saturation <= float(white_saturation))
    tissue = (saturation >= float(saturation_threshold)) & (maxc <= float(value_threshold)) & ~white
    tissue = ndimage.binary_opening(tissue, structure=np.ones((2, 2), dtype=bool), iterations=1)
    tissue = ndimage.binary_closing(tissue, structure=np.ones((3, 3), dtype=bool), iterations=1)
    labels, n_labels = ndimage.label(tissue)
    if n_labels > 0:
        sizes = np.bincount(labels.ravel())
        keep = np.flatnonzero(sizes >= int(min_object_pixels))
        keep = keep[keep != 0]
        tissue = np.isin(labels, keep)
    return tissue.astype(bool)


def _he_tissue_fraction(
    he_image: Path,
    bbox: np.ndarray,
    output_dir: Path,
    *,
    max_side: int,
    saturation_threshold: float,
    value_threshold: float,
    white_value: float,
    white_saturation: float,
    min_object_pixels: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    thumbnail, full_size = _load_thumbnail(he_image, max_side)
    tissue = _estimate_he_tissue_mask(
        thumbnail,
        saturation_threshold=saturation_threshold,
        value_threshold=value_threshold,
        white_value=white_value,
        white_saturation=white_saturation,
        min_object_pixels=min_object_pixels,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tissue_png = output_dir / "he_estimated_tissue_mask_thumbnail.png"
    Image.fromarray(tissue.astype(np.uint8) * 255, mode="L").save(tissue_png, optimize=True)

    sx = thumbnail.size[0] / float(full_size[0])
    sy = thumbnail.size[1] / float(full_size[1])
    fractions = np.zeros((bbox.shape[0],), dtype=np.float32)
    for index, rect in enumerate(bbox):
        x0 = max(0, min(thumbnail.size[0] - 1, int(np.floor(float(rect[0]) * sx))))
        y0 = max(0, min(thumbnail.size[1] - 1, int(np.floor(float(rect[1]) * sy))))
        x1 = max(x0 + 1, min(thumbnail.size[0], int(np.ceil(float(rect[2]) * sx))))
        y1 = max(y0 + 1, min(thumbnail.size[1], int(np.ceil(float(rect[3]) * sy))))
        fractions[index] = float(tissue[y0:y1, x0:x1].mean())
    return fractions, {
        "source": str(he_image),
        "full_size": [int(full_size[0]), int(full_size[1])],
        "thumbnail_size": [int(thumbnail.size[0]), int(thumbnail.size[1])],
        "thumbnail_mask_png": str(tissue_png),
        "thumbnail_tissue_fraction": float(tissue.mean()),
        "grid_tissue_fraction_min": float(fractions.min()),
        "grid_tissue_fraction_mean": float(fractions.mean()),
        "grid_tissue_fraction_median": float(np.median(fractions)),
        "grid_tissue_fraction_max": float(fractions.max()),
    }


def _grid_image(bbox: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values = np.unique(bbox[:, 0].astype(np.int32))
    y_values = np.unique(bbox[:, 1].astype(np.int32))
    x_index = {int(value): index for index, value in enumerate(x_values)}
    y_index = {int(value): index for index, value in enumerate(y_values)}
    rows = np.asarray([y_index[int(value)] for value in bbox[:, 1]], dtype=np.int32)
    cols = np.asarray([x_index[int(value)] for value in bbox[:, 0]], dtype=np.int32)
    image = np.zeros((len(y_values), len(x_values)), dtype=bool)
    image[rows, cols] = values.astype(bool)
    return image, rows, cols


def _spatial_clean(mask: np.ndarray, min_component_grids: int, max_hole_grids: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if int(min_component_grids) <= 0 and int(max_hole_grids) <= 0:
        labels, n_labels = ndimage.label(mask)
        sizes = np.bincount(labels.ravel())
        return mask, labels, {
            "mode": "disabled",
            "n_components": int(n_labels),
            "largest_component_grids": int(sizes[1:].max()) if sizes.size > 1 else 0,
        }

    cleaned = ndimage.binary_closing(mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
    holes = ndimage.binary_fill_holes(cleaned) & ~cleaned
    hole_labels, n_holes = ndimage.label(holes)
    n_hole_grids_filled = 0
    if int(max_hole_grids) > 0 and n_holes:
        sizes = np.bincount(hole_labels.ravel())
        for label in range(1, n_holes + 1):
            if int(sizes[label]) <= int(max_hole_grids):
                cleaned[hole_labels == label] = True
                n_hole_grids_filled += int(sizes[label])

    labels, n_labels = ndimage.label(cleaned)
    if int(min_component_grids) > 0:
        sizes = np.bincount(labels.ravel())
        keep = np.flatnonzero(sizes >= int(min_component_grids))
        keep = keep[keep != 0]
        cleaned = np.isin(labels, keep)
        labels, n_labels = ndimage.label(cleaned)
    sizes = np.bincount(labels.ravel())
    return cleaned, labels, {
        "mode": "enabled",
        "n_components": int(n_labels),
        "min_component_grids": int(min_component_grids),
        "max_hole_grids": int(max_hole_grids),
        "n_hole_grids_filled": int(n_hole_grids_filled),
        "largest_component_grids": int(sizes[1:].max()) if sizes.size > 1 else 0,
    }


def _sorted_sample_indices(n_rows: int, sample_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    take = min(int(sample_size), int(n_rows))
    return np.sort(rng.choice(n_rows, size=take, replace=False)).astype(np.int64)


def _fit_embedding_model(
    features: h5py.Dataset,
    *,
    sample_size: int,
    pca_components: int,
    k: int,
    seed: int,
) -> tuple[PCA, MiniBatchKMeans, np.ndarray]:
    sample_idx = _sorted_sample_indices(features.shape[0], sample_size, seed)
    sample = np.asarray(features[sample_idx], dtype=np.float32)
    sample = normalize(sample, norm="l2", axis=1, copy=False)
    pca = PCA(n_components=int(pca_components), svd_solver="randomized", random_state=int(seed))
    sample_pca = pca.fit_transform(sample).astype(np.float32, copy=False)
    kmeans = MiniBatchKMeans(
        n_clusters=int(k),
        random_state=int(seed),
        batch_size=min(8192, max(1024, sample_pca.shape[0])),
        n_init="auto",
        max_iter=200,
    )
    kmeans.fit(sample_pca)
    return pca, kmeans, sample_idx


def _predict_clusters(features: h5py.Dataset, pca: PCA, kmeans: MiniBatchKMeans, chunk_size: int) -> np.ndarray:
    labels = np.empty((features.shape[0],), dtype=np.int16)
    for start in range(0, features.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), features.shape[0])
        chunk = np.asarray(features[start:stop], dtype=np.float32)
        chunk = normalize(chunk, norm="l2", axis=1, copy=False)
        chunk_pca = pca.transform(chunk).astype(np.float32, copy=False)
        labels[start:stop] = kmeans.predict(chunk_pca).astype(np.int16)
    return labels


def _cluster_decisions(
    labels: np.ndarray,
    tissue_fraction: np.ndarray,
    *,
    k: int,
    confident_blank: float,
    confident_tissue: float,
    cluster_tissue_mean: float,
    cluster_tissue_ratio: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    decisions = np.zeros((int(k),), dtype=bool)
    rows: list[dict[str, Any]] = []
    for cluster in range(int(k)):
        mask = labels == cluster
        values = tissue_fraction[mask]
        if values.size == 0:
            rows.append(
                {
                    "cluster": cluster,
                    "n_grids": 0,
                    "mean_tissue_fraction": None,
                    "median_tissue_fraction": None,
                    "confident_tissue_ratio": None,
                    "confident_blank_ratio": None,
                    "cluster_is_tissue": False,
                }
            )
            continue
        tissue_ratio = float(np.mean(values >= float(confident_tissue)))
        blank_ratio = float(np.mean(values <= float(confident_blank)))
        mean_tissue = float(np.mean(values))
        is_tissue = bool(mean_tissue >= float(cluster_tissue_mean) or tissue_ratio >= float(cluster_tissue_ratio))
        decisions[cluster] = is_tissue
        rows.append(
            {
                "cluster": cluster,
                "n_grids": int(values.size),
                "mean_tissue_fraction": mean_tissue,
                "median_tissue_fraction": float(np.median(values)),
                "confident_tissue_ratio": tissue_ratio,
                "confident_blank_ratio": blank_ratio,
                "cluster_is_tissue": is_tissue,
            }
        )
    return decisions, rows


def _write_cluster_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cluster",
                "n_grids",
                "mean_tissue_fraction",
                "median_tissue_fraction",
                "confident_tissue_ratio",
                "confident_blank_ratio",
                "cluster_is_tissue",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_grid_csv(
    path: Path,
    bbox: np.ndarray,
    center: np.ndarray,
    tissue_fraction: np.ndarray,
    labels: np.ndarray,
    is_original: np.ndarray,
    is_corrected: np.ndarray,
    is_final: np.ndarray,
    component_id: np.ndarray,
) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "grid_index",
                "x0",
                "y0",
                "x1",
                "y1",
                "center_x",
                "center_y",
                "tissue_fraction",
                "embedding_cluster",
                "is_tissue_original",
                "is_tissue_embedding_corrected",
                "is_tissue_final",
                "component_id",
                "removed_reason",
            ]
        )
        for index in range(bbox.shape[0]):
            if bool(is_final[index]):
                reason = "kept"
            elif bool(is_corrected[index]):
                reason = "removed_by_spatial_filter"
            elif bool(is_original[index]):
                reason = "removed_by_embedding_cluster"
            else:
                reason = "blank_by_tissue_fraction_or_cluster"
            writer.writerow(
                [
                    int(index),
                    int(bbox[index, 0]),
                    int(bbox[index, 1]),
                    int(bbox[index, 2]),
                    int(bbox[index, 3]),
                    float(center[index, 0]),
                    float(center[index, 1]),
                    float(tissue_fraction[index]),
                    int(labels[index]),
                    int(bool(is_original[index])),
                    int(bool(is_corrected[index])),
                    int(bool(is_final[index])),
                    int(component_id[index]),
                    reason,
                ]
            )


def _draw_mask_thumbnail(bbox: np.ndarray, mask: np.ndarray, path: Path, max_side: int) -> dict[str, Any]:
    x0, y0 = bbox[:, 0].min(), bbox[:, 1].min()
    x1, y1 = bbox[:, 2].max(), bbox[:, 3].max()
    width = max(float(x1 - x0), 1.0)
    height = max(float(y1 - y0), 1.0)
    scale = min(float(max_side) / width, float(max_side) / height)
    image = Image.new("RGB", (int(np.ceil(width * scale)) + 1, int(np.ceil(height * scale)) + 1), "white")
    draw = ImageDraw.Draw(image)
    for rect in bbox[mask]:
        draw.rectangle(
            [
                int(round((rect[0] - x0) * scale)),
                int(round((rect[1] - y0) * scale)),
                int(round((rect[2] - x0) * scale)),
                int(round((rect[3] - y0) * scale)),
            ],
            fill=(30, 120, 220),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)
    return {"output": str(path), "size": list(image.size), "scale": float(scale)}


def _draw_he_outputs(he_image: Path, bbox: np.ndarray, mask: np.ndarray, output_dir: Path, max_side: int) -> dict[str, Any]:
    thumbnail, full_size = _load_thumbnail(he_image, max_side)
    sx = thumbnail.size[0] / float(full_size[0])
    sy = thumbnail.size[1] / float(full_size[1])
    mask_layer = Image.new("L", thumbnail.size, 0)
    draw = ImageDraw.Draw(mask_layer)
    for rect in bbox[mask]:
        draw.rectangle(
            [
                int(round(float(rect[0]) * sx)),
                int(round(float(rect[1]) * sy)),
                int(round(float(rect[2]) * sx)),
                int(round(float(rect[3]) * sy)),
            ],
            fill=255,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    masked = Image.new("RGB", thumbnail.size, "white")
    masked.paste(thumbnail, mask=mask_layer)
    masked_path = output_dir / "masked_he_thumbnail.png"
    masked.save(masked_path, optimize=True)

    overlay = thumbnail.convert("RGBA")
    red = Image.new("RGBA", thumbnail.size, (255, 0, 0, 0))
    red.putalpha(mask_layer.point(lambda value: 70 if value else 0))
    overlay_path = output_dir / "mask_overlay_he_thumbnail.png"
    Image.alpha_composite(overlay, red).convert("RGB").save(overlay_path, optimize=True)

    mask_path = output_dir / "mask_thumbnail_on_he_size.png"
    mask_layer.save(mask_path, optimize=True)
    return {
        "he_full_size": [int(full_size[0]), int(full_size[1])],
        "thumbnail_size": [int(thumbnail.size[0]), int(thumbnail.size[1])],
        "masked_he_thumbnail": str(masked_path),
        "mask_overlay_he_thumbnail": str(overlay_path),
        "mask_thumbnail_on_he_size": str(mask_path),
    }


def _copy_filtered_h5(source_h5: Path, output_h5: Path, mask_h5: Path, *, force: bool) -> dict[str, Any]:
    if output_h5.exists():
        if force:
            output_h5.unlink()
        else:
            raise FileExistsError(output_h5)
    with h5py.File(source_h5, "r") as source, h5py.File(mask_h5, "r") as mask:
        feature_key = str(source.attrs["feature_key"]) if "feature_key" in source.attrs else _feature_key(source_h5)
        n_grids = int(source[feature_key].shape[0])
        source_bbox = np.asarray(source["bbox_xyxy"], dtype=np.int32)
        mask_bbox = np.asarray(mask["bbox_xyxy"], dtype=np.int32)
        mask_values = np.asarray(mask["is_tissue_final"], dtype=bool)
        if source_bbox.shape == mask_bbox.shape and bool(np.all(source_bbox == mask_bbox)):
            keep = mask_values
            alignment = "row_order"
        else:
            lookup = {tuple(map(int, row)): index for index, row in enumerate(mask_bbox)}
            indices = np.asarray([lookup.get(tuple(map(int, row)), -1) for row in source_bbox], dtype=np.int64)
            missing = int((indices < 0).sum())
            if missing:
                raise ValueError(f"{missing} grid bboxes from {source_h5} were not found in {mask_h5}.")
            keep = mask_values[indices]
            alignment = "bbox_xyxy"
        if keep.shape[0] != n_grids:
            raise ValueError(f"Mask length {keep.shape[0]} does not match embedding rows {n_grids}.")
        keep_indices = np.flatnonzero(keep)

        output_h5.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_h5, "w") as output:
            for key, value in source.attrs.items():
                output.attrs[key] = value
            output.attrs["source_h5"] = str(source_h5)
            output.attrs["mask_h5"] = str(mask_h5)
            output.attrs["mask_alignment"] = alignment
            output.attrs["filter_rule"] = "rows where mask/is_tissue_final is true"
            output.attrs["n_source_grids"] = n_grids
            output.attrs["n_filtered_grids"] = int(keep_indices.size)

            for key, dataset in source.items():
                values = dataset[()]
                if getattr(dataset, "shape", ()) and dataset.shape[0] == n_grids:
                    values = values[keep_indices]
                if values.dtype.kind in {"O", "S", "U"}:
                    dtype = h5py.string_dtype(encoding="utf-8")
                    output.create_dataset(key, data=values.astype(object), dtype=dtype)
                else:
                    output.create_dataset(key, data=values, compression=dataset.compression if dataset.shape else None)
    return {
        "source_h5": str(source_h5),
        "mask_h5": str(mask_h5),
        "output_h5": str(output_h5),
        "feature_key": feature_key,
        "mask_alignment": alignment,
        "n_source_grids": n_grids,
        "n_filtered_grids": int(keep_indices.size),
    }


def run_mask(
    *,
    grid_embedding_h5: str | Path,
    output_dir: str | Path,
    output_h5: str | Path | None = None,
    training_h5: str | Path | None = None,
    he_image: str | Path | None = None,
    mode: str = "he",
    feature_key: str | None = None,
    sample_size: int = 40000,
    pca_components: int = 32,
    k: int = 8,
    chunk_size: int = 8192,
    confident_blank: float = 0.02,
    confident_tissue: float = 0.50,
    tissue_fraction_threshold: float = 0.10,
    cluster_tissue_mean: float = 0.20,
    cluster_tissue_ratio: float = 0.15,
    min_component_grids: int = 0,
    max_hole_grids: int = 0,
    thumbnail_max_side: int = 2600,
    he_mask_max_side: int = 5000,
    he_saturation_threshold: float = 0.055,
    he_value_threshold: float = 0.985,
    he_white_value: float = 0.88,
    he_white_saturation: float = 0.10,
    min_he_object_pixels: int = 64,
    seed: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    selected_mode = str(mode)
    if selected_mode not in {"all", "he", "embedding"}:
        raise ValueError("mode must be 'all', 'he', or 'embedding'.")
    source_h5 = Path(grid_embedding_h5)
    out = Path(output_dir)
    mask_h5 = Path(output_h5) if output_h5 is not None else out / "he_mask_grid.h5"
    train_h5 = Path(training_h5) if training_h5 is not None else out / "grid_embedding_train_filtered.h5"
    summary_json = out / "mask_summary.json"
    grid_thumbnail = out / "mask_grid_thumbnail.png"
    cluster_summary_csv = out / "mask_cluster_summary.csv"
    grid_csv = out / "mask_grid.csv"
    if (mask_h5.exists() or train_h5.exists()) and not force:
        raise FileExistsError(f"{mask_h5} or {train_h5} exists; pass force=True to overwrite.")
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(source_h5, "r") as handle:
        bbox = np.asarray(handle["bbox_xyxy"], dtype=np.int32)
        center = np.asarray(handle["center_xy"], dtype=np.float32)
        if feature_key is not None:
            selected_feature_key = str(feature_key)
        elif "feature_key" in handle.attrs:
            selected_feature_key = str(handle.attrs["feature_key"])
        else:
            selected_feature_key = _feature_key(source_h5)
        feature_shape = list(handle[selected_feature_key].shape)

    if selected_mode == "all":
        tissue_fraction = np.ones((bbox.shape[0],), dtype=np.float32)
        he_summary: dict[str, Any] = {"source": None, "mode": "all_true"}
    else:
        if he_image is None:
            raise ValueError("mode='he' requires he_image.")
        tissue_fraction, he_summary = _he_tissue_fraction(
            Path(he_image),
            bbox,
            out,
            max_side=int(he_mask_max_side),
            saturation_threshold=float(he_saturation_threshold),
            value_threshold=float(he_value_threshold),
            white_value=float(he_white_value),
            white_saturation=float(he_white_saturation),
            min_object_pixels=int(min_he_object_pixels),
        )

    labels = np.full((bbox.shape[0],), -1, dtype=np.int16)
    cluster_rows: list[dict[str, Any]] = []
    feature_dim = 0
    sampled_rows = 0
    is_original = tissue_fraction >= float(tissue_fraction_threshold)
    if selected_mode == "embedding":
        with h5py.File(source_h5, "r") as handle:
            features = handle[selected_feature_key]
            pca, kmeans, sample_idx = _fit_embedding_model(
                features,
                sample_size=int(sample_size),
                pca_components=int(pca_components),
                k=int(k),
                seed=int(seed),
            )
            labels = _predict_clusters(features, pca, kmeans, int(chunk_size))
            feature_dim = int(pca.n_features_in_)
            sampled_rows = int(sample_idx.size)
        cluster_is_tissue, cluster_rows = _cluster_decisions(
            labels,
            tissue_fraction,
            k=int(k),
            confident_blank=float(confident_blank),
            confident_tissue=float(confident_tissue),
            cluster_tissue_mean=float(cluster_tissue_mean),
            cluster_tissue_ratio=float(cluster_tissue_ratio),
        )
        is_corrected = np.where(
            tissue_fraction <= float(confident_blank),
            False,
            np.where(tissue_fraction >= float(confident_tissue), True, cluster_is_tissue[labels]),
        )
    else:
        is_corrected = is_original.copy()

    grid_image, rows, cols = _grid_image(bbox, is_corrected)
    cleaned_image, component_image, spatial_summary = _spatial_clean(
        grid_image,
        int(min_component_grids),
        int(max_hole_grids),
    )
    is_final = cleaned_image[rows, cols]
    component_id = component_image[rows, cols].astype(np.int32)
    thumbnail_summary = _draw_mask_thumbnail(bbox, is_final, grid_thumbnail, int(thumbnail_max_side))
    he_outputs = (
        None
        if selected_mode == "all" or he_image is None
        else _draw_he_outputs(Path(he_image), bbox, is_final, out, int(thumbnail_max_side))
    )
    _write_cluster_summary(cluster_summary_csv, cluster_rows)
    _write_grid_csv(grid_csv, bbox, center, tissue_fraction, labels, is_original, is_corrected, is_final, component_id)

    if mask_h5.exists() and force:
        mask_h5.unlink()
    with h5py.File(mask_h5, "w") as handle:
        handle.create_dataset("bbox_xyxy", data=bbox, compression="gzip")
        handle.create_dataset("center_xy", data=center, compression="gzip")
        handle.create_dataset("grid_tissue_fraction", data=tissue_fraction.astype(np.float32), compression="gzip")
        handle.create_dataset("embedding_cluster", data=labels.astype(np.int16), compression="gzip")
        handle.create_dataset("is_tissue_original", data=is_original.astype(np.bool_), compression="gzip")
        handle.create_dataset("is_tissue_embedding_corrected", data=is_corrected.astype(np.bool_), compression="gzip")
        handle.create_dataset("is_tissue_final", data=is_final.astype(np.bool_), compression="gzip")
        handle.create_dataset("component_id", data=component_id, compression="gzip")
        handle.attrs["source_h5"] = str(source_h5)
        handle.attrs["mode"] = selected_mode
        handle.attrs["tissue_fraction_threshold"] = float(tissue_fraction_threshold)

    training_summary = _copy_filtered_h5(source_h5, train_h5, mask_h5, force=force)
    summary = {
        "task": "mask",
        "mode": selected_mode,
        "grid_embedding_h5": str(source_h5),
        "feature_key": selected_feature_key,
        "feature_shape": feature_shape,
        "feature_dim": feature_dim,
        "sample_size": sampled_rows,
        "pca_components": int(pca_components),
        "k": int(k),
        "output_dir": str(out),
        "mask_h5": str(mask_h5),
        "training_h5": str(train_h5),
        "summary_json": str(summary_json),
        "cluster_summary_csv": str(cluster_summary_csv),
        "grid_csv": str(grid_csv),
        "n_grids": int(bbox.shape[0]),
        "n_tissue_original": int(is_original.sum()),
        "n_tissue_embedding_corrected": int(is_corrected.sum()),
        "n_tissue_final": int(is_final.sum()),
        "n_removed_by_embedding": int((is_original & ~is_corrected).sum()),
        "n_removed_by_spatial_filter": int((is_corrected & ~is_final).sum()),
        "tissue_fraction_threshold": float(tissue_fraction_threshold),
        "thresholds": {
            "confident_blank": float(confident_blank),
            "confident_tissue": float(confident_tissue),
            "cluster_tissue_mean": float(cluster_tissue_mean),
            "cluster_tissue_ratio": float(cluster_tissue_ratio),
        },
        "he": he_summary,
        "he_outputs": he_outputs,
        "spatial": spatial_summary,
        "grid_thumbnail": thumbnail_summary,
        "training": training_summary,
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def validate_mask_outputs(mask_h5: str | Path, training_h5: str | Path) -> dict[str, Any]:
    with h5py.File(mask_h5, "r") as mask:
        is_tissue = np.asarray(mask["is_tissue_final"], dtype=bool)
        bbox = np.asarray(mask["bbox_xyxy"], dtype=np.int32)
    with h5py.File(training_h5, "r") as train:
        feature_key = str(train.attrs.get("feature_key", _feature_key(Path(training_h5))))
        feature_shape = list(train[feature_key].shape)
    return {
        "mask_h5": str(mask_h5),
        "training_h5": str(training_h5),
        "n_mask_grids": int(bbox.shape[0]),
        "n_tissue_final": int(is_tissue.sum()),
        "training_feature_key": feature_key,
        "training_feature_shape": feature_shape,
    }


__all__ = ["run_mask", "validate_mask_outputs"]
