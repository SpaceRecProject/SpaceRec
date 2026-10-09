from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw
from scipy import sparse
from scipy.spatial import cKDTree

from .alignment import load_alignment_matrix, xenium_um_to_he
from .io_utils import filter_10x_h5_to_ref_genes, read_10x_features, write_10x_h5


def build_hex_grid(points_xy: np.ndarray, *, spacing_px: float, margin_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("points_xy must be a non-empty n x 2 array.")
    min_x = float(points[:, 0].min()) - float(margin_px)
    min_y = float(points[:, 1].min()) - float(margin_px)
    max_x = float(points[:, 0].max()) + float(margin_px)
    max_y = float(points[:, 1].max()) + float(margin_px)
    spot_xy: list[tuple[float, float]] = []
    array_rows: list[int] = []
    array_cols: list[int] = []
    row = 0
    y = min_y
    y_step = float(spacing_px) * np.sqrt(3.0) / 2.0
    while y <= max_y:
        x_offset = 0.5 * float(spacing_px) if row % 2 else 0.0
        x = min_x + x_offset
        col = 0
        while x <= max_x:
            spot_xy.append((x, y))
            array_rows.append(row)
            array_cols.append(2 * col + (row % 2))
            x += float(spacing_px)
            col += 1
        y += y_step
        row += 1
    return np.asarray(spot_xy, dtype=np.float64), np.asarray(array_rows, dtype=np.int32), np.asarray(array_cols, dtype=np.int32)


def _load_he_size(path: Path) -> tuple[int, int]:
    with tifffile.TiffFile(path) as tif:
        shape = tif.series[0].levels[0].shape
    return int(shape[1]), int(shape[0])


def _load_he_thumbnail(path: Path, *, max_dim: int) -> tuple[Image.Image, float]:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        level = len(series.levels) - 1
        for index, candidate in enumerate(series.levels):
            height, width = candidate.shape[:2]
            if max(int(width), int(height)) <= int(max_dim):
                level = index
                break
        image = np.asarray(series.levels[level].asarray())
        full_h, full_w = series.levels[0].shape[:2]
    if image.ndim == 2:
        pil = Image.fromarray(image).convert("RGB")
    else:
        pil = Image.fromarray(image[:, :, :3]).convert("RGB")
    scale = pil.width / float(full_w)
    if max(pil.size) > int(max_dim):
        pil.thumbnail((int(max_dim), int(max_dim)), Image.Resampling.LANCZOS)
        scale = pil.width / float(full_w)
    return pil, float(scale)


def render_spot_thumbnail(
    *,
    he_image: str | Path,
    output_png: str | Path,
    spot_xy: np.ndarray,
    spot_radius_fullres_px: float,
    max_dim: int = 1000,
) -> dict[str, Any]:
    image, scale = _load_he_thumbnail(Path(he_image), max_dim=max_dim)
    draw = ImageDraw.Draw(image)
    radius = max(1.0, float(spot_radius_fullres_px) * float(scale))
    for x, y in np.asarray(spot_xy, dtype=np.float64):
        cx = float(x) * scale
        cy = float(y) * scale
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(255, 40, 40), width=1)
    output = Path(output_png)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    full_w, full_h = _load_he_size(Path(he_image))
    return {
        "thumbnail_png": str(output),
        "he_full_size": [int(full_w), int(full_h)],
        "thumbnail_size": [int(image.width), int(image.height)],
        "tissue_hires_scalef": float(scale),
    }


def aggregate_transcripts_to_candidates(
    *,
    transcripts_csv_gz: str | Path,
    feature_matrix_h5: str | Path,
    alignment_csv: str | Path,
    candidate_xy: np.ndarray,
    spot_radius_fullres_px: float,
    xenium_pixel_size: float = 0.2125,
    matrix_direction: str = "he-to-xenium",
    chunksize: int = 1_000_000,
    max_transcripts: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    gene_names, features = read_10x_features(feature_matrix_h5)
    gene_to_index = {str(name): index for index, name in enumerate(gene_names)}
    counts = np.zeros((len(gene_names), len(candidate_xy)), dtype=np.int64)
    tree = cKDTree(np.asarray(candidate_xy, dtype=np.float64))
    matrix = load_alignment_matrix(alignment_csv, matrix_direction=matrix_direction)

    n_rows = 0
    n_gene_matched = 0
    n_unknown_feature = 0
    n_outside_radius = 0
    n_assigned = 0
    usecols = ["feature_name", "x_location", "y_location"]
    for chunk in pd.read_csv(transcripts_csv_gz, usecols=usecols, chunksize=int(chunksize)):
        if max_transcripts is not None:
            remaining = int(max_transcripts) - n_rows
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()
        n_rows += int(len(chunk))
        gene_idx = chunk["feature_name"].map(gene_to_index).to_numpy()
        matched = pd.notna(gene_idx)
        n_unknown_feature += int((~matched).sum())
        if not np.any(matched):
            continue
        matched_chunk = chunk.loc[matched]
        gene_idx = gene_idx[matched].astype(np.int64)
        n_gene_matched += int(len(gene_idx))
        _dapi_x, _dapi_y, he_x, he_y = xenium_um_to_he(
            matched_chunk["x_location"].to_numpy(dtype=np.float64),
            matched_chunk["y_location"].to_numpy(dtype=np.float64),
            matrix,
            xenium_pixel_size=xenium_pixel_size,
        )
        finite = np.isfinite(he_x) & np.isfinite(he_y)
        if not np.any(finite):
            n_outside_radius += int(len(gene_idx))
            continue
        gene_idx = gene_idx[finite]
        he_x = he_x[finite]
        he_y = he_y[finite]
        distances, nearest = tree.query(np.column_stack([he_x, he_y]), k=1)
        inside = distances <= float(spot_radius_fullres_px)
        n_outside_radius += int((~inside).sum())
        if np.any(inside):
            np.add.at(counts, (gene_idx[inside], nearest[inside].astype(np.int64)), 1)
            n_assigned += int(inside.sum())
    summary = {
        "n_transcripts_read": int(n_rows),
        "n_gene_matched_transcripts": int(n_gene_matched),
        "n_unknown_feature_transcripts": int(n_unknown_feature),
        "n_assigned_transcripts": int(n_assigned),
        "n_outside_radius_gene_matched_transcripts": int(n_outside_radius),
        "n_genes": int(len(gene_names)),
    }
    return counts, gene_names, summary, features


def write_positions_outputs(
    *,
    output_dir: Path,
    spot_xy: np.ndarray,
    array_rows: np.ndarray,
    array_cols: np.ndarray,
    transcript_counts: np.ndarray,
) -> dict[str, str]:
    barcodes = np.asarray([f"synthetic_spot_{index + 1:06d}" for index in range(len(spot_xy))], dtype=object)
    positions = pd.DataFrame(
        {
            "barcode": barcodes,
            "in_tissue": 1,
            "array_row": array_rows.astype(int),
            "array_col": array_cols.astype(int),
            "pxl_row_in_fullres": np.rint(spot_xy[:, 1]).astype(int),
            "pxl_col_in_fullres": np.rint(spot_xy[:, 0]).astype(int),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    positions["transcript_count"] = transcript_counts.astype(int)
    positions_csv = output_dir / "synthetic_tissue_positions_transcript_filtered.csv"
    positions.to_csv(positions_csv, index=False)
    return {"positions_csv": str(positions_csv)}


def _select_he_image(xen_root: Path) -> Path:
    ome_images = sorted(xen_root.glob("*_he_image.ome.tif"))
    if ome_images:
        return ome_images[0]
    plain_images = sorted(xen_root.glob("*_he_image.tif"))
    if plain_images:
        return plain_images[0]
    raise FileNotFoundError(f"No H&E image found in {xen_root}; expected *_he_image.ome.tif or *_he_image.tif.")


def generate_synthetic_visium(
    *,
    xen_dir: str | Path,
    aligned_dir: str | Path,
    output_dir: str | Path,
    sc_ref_h5ad: str | Path | None = None,
    he_image: str | Path | None = None,
    alignment_csv: str | Path | None = None,
    xenium_pixel_size: float = 0.2125,
    he_pixel_size: float = 0.363788,
    spot_diameter_um: float = 55.0,
    spot_spacing_um: float = 100.0,
    margin_um: float = 50.0,
    chunksize: int = 1_000_000,
    matrix_direction: str = "he-to-xenium",
    thumbnail_max_dim: int = 1000,
    max_transcripts: int | None = None,
) -> dict[str, Any]:
    xen_root = Path(xen_dir)
    out = Path(output_dir)
    selected_he = Path(he_image) if he_image is not None else _select_he_image(xen_root)
    selected_alignment = Path(alignment_csv) if alignment_csv is not None else next(xen_root.glob("*_he_imagealignment.csv"))
    he_width, he_height = _load_he_size(selected_he)
    cells = pd.read_csv(Path(aligned_dir) / "aligned_cells.csv", usecols=["he_x", "he_y"])
    radius_px = float(spot_diameter_um) / 2.0 / float(he_pixel_size)
    spacing_px = float(spot_spacing_um) / float(he_pixel_size)
    margin_px = float(margin_um) / float(he_pixel_size)
    candidate_xy, candidate_rows, candidate_cols = build_hex_grid(
        cells[["he_x", "he_y"]].to_numpy(dtype=np.float64),
        spacing_px=spacing_px,
        margin_px=margin_px,
    )
    n_candidate_spots_before_image_filter = int(len(candidate_xy))
    inside_he_image = (
        (candidate_xy[:, 0] >= 0.0)
        & (candidate_xy[:, 0] < float(he_width))
        & (candidate_xy[:, 1] >= 0.0)
        & (candidate_xy[:, 1] < float(he_height))
    )
    candidate_xy = candidate_xy[inside_he_image]
    candidate_rows = candidate_rows[inside_he_image]
    candidate_cols = candidate_cols[inside_he_image]
    if len(candidate_xy) == 0:
        raise ValueError("No candidate synthetic spots remain inside the H&E full-resolution image bounds.")
    counts, _gene_names, transcript_summary, features = aggregate_transcripts_to_candidates(
        transcripts_csv_gz=xen_root / "outs" / "transcripts.csv.gz",
        feature_matrix_h5=xen_root / "outs" / "cell_feature_matrix.h5",
        alignment_csv=selected_alignment,
        candidate_xy=candidate_xy,
        spot_radius_fullres_px=radius_px,
        xenium_pixel_size=xenium_pixel_size,
        matrix_direction=matrix_direction,
        chunksize=chunksize,
        max_transcripts=max_transcripts,
    )
    spot_counts = counts.sum(axis=0)
    keep = spot_counts > 0
    kept_counts = counts[:, keep]
    kept_xy = candidate_xy[keep]
    kept_rows = candidate_rows[keep]
    kept_cols = candidate_cols[keep]
    kept_spot_counts = spot_counts[keep]
    barcodes = np.asarray([f"synthetic_spot_{index + 1:06d}" for index in range(int(keep.sum()))], dtype=object)
    synthetic_matrix = sparse.csc_matrix(kept_counts)
    synthetic_h5 = out / "synthetic_st_filtered_feature_bc_matrix_transcript.h5"
    write_10x_h5(synthetic_h5, synthetic_matrix, barcodes, features)
    position_outputs = write_positions_outputs(
        output_dir=out,
        spot_xy=kept_xy,
        array_rows=kept_rows,
        array_cols=kept_cols,
        transcript_counts=kept_spot_counts,
    )
    thumb = render_spot_thumbnail(
        he_image=selected_he,
        output_png=out / "visium_like_spots_on_he_thumbnail_transcript.png",
        spot_xy=kept_xy,
        spot_radius_fullres_px=radius_px,
        max_dim=thumbnail_max_dim,
    )
    second_thumb = out / "visium_like_spots_on_he_thumbnail.png"
    if Path(thumb["thumbnail_png"]).resolve() != second_thumb.resolve():
        second_thumb.write_bytes(Path(thumb["thumbnail_png"]).read_bytes())
    scalefactors = {
        "spot_diameter_fullres": float(radius_px * 2.0),
        "tissue_hires_scalef": float(thumb["tissue_hires_scalef"]),
    }
    scalefactors_json = out / "synthetic_scalefactors_json.json"
    scalefactors_json.write_text(json.dumps(scalefactors, indent=2) + "\n", encoding="utf-8")
    summary = {
        "method": "transcript_level_aggregation_to_candidate_spots_then_filter_nonzero",
        "xen_dir": str(xen_root),
        "aligned_dir": str(Path(aligned_dir)),
        "output_dir": str(out),
        "he_image": str(selected_he),
        "he_full_size": [int(he_width), int(he_height)],
        "alignment_csv": str(selected_alignment),
        "n_candidate_spots_before_image_filter": n_candidate_spots_before_image_filter,
        "n_candidate_spots_outside_he_image": int((~inside_he_image).sum()),
        "n_candidate_spots": int(len(candidate_xy)),
        "n_spots": int(len(barcodes)),
        "matrix_shape": [int(synthetic_matrix.shape[0]), int(synthetic_matrix.shape[1])],
        "matrix_nnz": int(synthetic_matrix.nnz),
        "synthetic_total_counts": int(synthetic_matrix.sum()),
        "min_counts_per_spot": int(kept_spot_counts.min()) if len(kept_spot_counts) else 0,
        "median_counts_per_spot": float(np.median(kept_spot_counts)) if len(kept_spot_counts) else 0.0,
        "max_counts_per_spot": int(kept_spot_counts.max()) if len(kept_spot_counts) else 0,
        "spot_diameter_um": float(spot_diameter_um),
        "spot_spacing_um": float(spot_spacing_um),
        "spot_radius_fullres_px": float(radius_px),
        "spot_spacing_fullres_px": float(spacing_px),
        "synthetic_h5": str(synthetic_h5),
        "scalefactors_json": str(scalefactors_json),
        **transcript_summary,
        **position_outputs,
        **thumb,
    }
    summary_json = out / "synthetic_visium_like_summary_transcript.json"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(thumb["thumbnail_png"]).with_suffix(".metadata.txt").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["summary_json"] = str(summary_json)

    if sc_ref_h5ad is not None:
        filter_summary = filter_10x_h5_to_ref_genes(
            synthetic_h5=synthetic_h5,
            sc_ref_h5ad=sc_ref_h5ad,
            output_h5=out / "filtered_synthetic_st_filtered_feature_bc_matrix_transcript.h5",
            summary_json=out / "filtered_synthetic_st_gene_intersection_summary_transcript.json",
        )
        summary["gene_filter"] = filter_summary
    return summary
