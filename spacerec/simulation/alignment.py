from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_alignment_matrix(path: str | Path, *, matrix_direction: str = "he-to-xenium") -> np.ndarray:
    if matrix_direction not in {"he-to-xenium", "xenium-to-he"}:
        raise ValueError("matrix_direction must be 'he-to-xenium' or 'xenium-to-he'.")
    rows: list[list[float]] = []
    with Path(path).open(newline="") as handle:
        for row in csv.reader(handle):
            if row:
                rows.append([float(value) for value in row])
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 alignment matrix, got {matrix.shape}: {path}")
    if matrix_direction == "he-to-xenium":
        matrix = np.linalg.inv(matrix)
    return matrix


def xenium_um_to_dapi_px(
    x_um: np.ndarray,
    y_um: np.ndarray,
    *,
    xenium_pixel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    if xenium_pixel_size <= 0:
        raise ValueError(f"xenium_pixel_size must be positive, got {xenium_pixel_size}.")
    return (
        np.asarray(x_um, dtype=np.float64) / float(xenium_pixel_size),
        np.asarray(y_um, dtype=np.float64) / float(xenium_pixel_size),
    )


def dapi_px_to_he(
    dapi_x: np.ndarray,
    dapi_y: np.ndarray,
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(dapi_x, dtype=np.float64)
    y = np.asarray(dapi_y, dtype=np.float64)
    denom = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    he_x = (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denom
    he_y = (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denom
    return he_x, he_y


def xenium_um_to_he(
    x_um: np.ndarray,
    y_um: np.ndarray,
    matrix: np.ndarray,
    *,
    xenium_pixel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dapi_x, dapi_y = xenium_um_to_dapi_px(x_um, y_um, xenium_pixel_size=xenium_pixel_size)
    he_x, he_y = dapi_px_to_he(dapi_x, dapi_y, matrix)
    return dapi_x, dapi_y, he_x, he_y


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def align_cells(
    *,
    cells_csv: str | Path,
    alignment_csv: str | Path,
    output_csv: str | Path,
    xenium_pixel_size: float = 0.2125,
    matrix_direction: str = "he-to-xenium",
) -> dict[str, Any]:
    matrix = load_alignment_matrix(alignment_csv, matrix_direction=matrix_direction)
    table = pd.read_csv(cells_csv)
    required = {"cell_id", "x_centroid", "y_centroid"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{cells_csv} is missing columns: {sorted(missing)}")
    dapi_x, dapi_y, he_x, he_y = xenium_um_to_he(
        table["x_centroid"].to_numpy(dtype=np.float64),
        table["y_centroid"].to_numpy(dtype=np.float64),
        matrix,
        xenium_pixel_size=xenium_pixel_size,
    )
    table["dapi_px_x"] = dapi_x
    table["dapi_px_y"] = dapi_y
    table["he_x"] = he_x
    table["he_y"] = he_y
    output = Path(output_csv)
    _write_csv(table, output)
    return {
        "output_csv": str(output),
        "n_cells": int(len(table)),
        "he_x_min": float(np.nanmin(he_x)) if len(he_x) else None,
        "he_x_max": float(np.nanmax(he_x)) if len(he_x) else None,
        "he_y_min": float(np.nanmin(he_y)) if len(he_y) else None,
        "he_y_max": float(np.nanmax(he_y)) if len(he_y) else None,
    }


def align_boundaries(
    *,
    boundaries_csv: str | Path,
    alignment_csv: str | Path,
    output_csv: str | Path,
    xenium_pixel_size: float = 0.2125,
    matrix_direction: str = "he-to-xenium",
    chunksize: int = 1_000_000,
) -> dict[str, Any]:
    matrix = load_alignment_matrix(alignment_csv, matrix_direction=matrix_direction)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    he_x_min = he_y_min = float("inf")
    he_x_max = he_y_max = float("-inf")
    wrote_header = False
    for chunk in pd.read_csv(boundaries_csv, chunksize=int(chunksize)):
        missing = {"cell_id", "vertex_x", "vertex_y"}.difference(chunk.columns)
        if missing:
            raise ValueError(f"{boundaries_csv} is missing columns: {sorted(missing)}")
        dapi_x, dapi_y, he_x, he_y = xenium_um_to_he(
            chunk["vertex_x"].to_numpy(dtype=np.float64),
            chunk["vertex_y"].to_numpy(dtype=np.float64),
            matrix,
            xenium_pixel_size=xenium_pixel_size,
        )
        chunk["dapi_px_x"] = dapi_x
        chunk["dapi_px_y"] = dapi_y
        chunk["he_x"] = he_x
        chunk["he_y"] = he_y
        chunk.to_csv(output, index=False, mode="w" if not wrote_header else "a", header=not wrote_header)
        wrote_header = True
        n_rows += int(len(chunk))
        if len(chunk):
            he_x_min = min(he_x_min, float(np.nanmin(he_x)))
            he_x_max = max(he_x_max, float(np.nanmax(he_x)))
            he_y_min = min(he_y_min, float(np.nanmin(he_y)))
            he_y_max = max(he_y_max, float(np.nanmax(he_y)))
    return {
        "output_csv": str(output),
        "n_boundary_vertices": int(n_rows),
        "he_x_min": None if n_rows == 0 else he_x_min,
        "he_x_max": None if n_rows == 0 else he_x_max,
        "he_y_min": None if n_rows == 0 else he_y_min,
        "he_y_max": None if n_rows == 0 else he_y_max,
    }


def align_transcripts(
    *,
    transcripts_csv_gz: str | Path,
    alignment_csv: str | Path,
    output_csv_gz: str | Path,
    xenium_pixel_size: float = 0.2125,
    matrix_direction: str = "he-to-xenium",
    chunksize: int = 1_000_000,
    max_rows: int | None = None,
) -> dict[str, Any]:
    matrix = load_alignment_matrix(alignment_csv, matrix_direction=matrix_direction)
    output = Path(output_csv_gz)
    output.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    wrote_header = False
    with gzip.open(output, "wt", newline="") as handle:
        for chunk in pd.read_csv(transcripts_csv_gz, chunksize=int(chunksize)):
            if max_rows is not None:
                remaining = int(max_rows) - n_rows
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining].copy()
            missing = {"x_location", "y_location"}.difference(chunk.columns)
            if missing:
                raise ValueError(f"{transcripts_csv_gz} is missing columns: {sorted(missing)}")
            dapi_x, dapi_y, he_x, he_y = xenium_um_to_he(
                chunk["x_location"].to_numpy(dtype=np.float64),
                chunk["y_location"].to_numpy(dtype=np.float64),
                matrix,
                xenium_pixel_size=xenium_pixel_size,
            )
            chunk["dapi_px_x"] = dapi_x
            chunk["dapi_px_y"] = dapi_y
            chunk["he_x"] = he_x
            chunk["he_y"] = he_y
            chunk.to_csv(handle, index=False, header=not wrote_header)
            wrote_header = True
            n_rows += int(len(chunk))
    return {"output_csv_gz": str(output), "n_transcripts": int(n_rows)}


def run_alignment(
    *,
    xen_dir: str | Path,
    aligned_dir: str | Path,
    alignment_csv: str | Path | None = None,
    xenium_pixel_size: float = 0.2125,
    matrix_direction: str = "he-to-xenium",
    chunksize: int = 1_000_000,
    write_aligned_transcripts: bool = False,
    max_transcripts: int | None = None,
) -> dict[str, Any]:
    xen_root = Path(xen_dir)
    out = Path(aligned_dir)
    selected_alignment = Path(alignment_csv) if alignment_csv is not None else next(xen_root.glob("*_he_imagealignment.csv"))
    outs = xen_root / "outs"
    cells_csv = outs / "cells.csv.gz"
    boundaries_csv = outs / "cell_boundaries.csv.gz"
    transcripts_csv = outs / "transcripts.csv.gz"

    cells_summary = align_cells(
        cells_csv=cells_csv,
        alignment_csv=selected_alignment,
        output_csv=out / "aligned_cells.csv",
        xenium_pixel_size=xenium_pixel_size,
        matrix_direction=matrix_direction,
    )
    boundaries_summary = align_boundaries(
        boundaries_csv=boundaries_csv,
        alignment_csv=selected_alignment,
        output_csv=out / "he_alignmented_cell_boundaries.csv",
        xenium_pixel_size=xenium_pixel_size,
        matrix_direction=matrix_direction,
        chunksize=chunksize,
    )
    transcripts_summary: dict[str, Any] | None = None
    if write_aligned_transcripts:
        transcripts_summary = align_transcripts(
            transcripts_csv_gz=transcripts_csv,
            alignment_csv=selected_alignment,
            output_csv_gz=out / "aligned_transcripts.csv.gz",
            xenium_pixel_size=xenium_pixel_size,
            matrix_direction=matrix_direction,
            chunksize=chunksize,
            max_rows=max_transcripts,
        )
    summary = {
        "task": "simulation_alignment",
        "xen_dir": str(xen_root),
        "aligned_dir": str(out),
        "alignment_csv": str(selected_alignment),
        "xenium_pixel_size": float(xenium_pixel_size),
        "matrix_direction": matrix_direction,
        "cells": cells_summary,
        "boundaries": boundaries_summary,
        "transcripts": transcripts_summary,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "alignment_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary

