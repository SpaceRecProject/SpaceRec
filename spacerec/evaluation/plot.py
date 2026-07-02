from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


CELL_TYPE_ORDER = [
    "B Cells",
    "T Cells",
    "DCIS",
    "Endothelial",
    "Invasive Tumor",
    "DCs",
    "Macrophages",
    "Mast Cells",
    "Myoepi",
    "Perivascular-Like",
    "Stromal",
]
CELL_TYPE_COLORS = {
    "B Cells": "#fa786e",
    "T Cells": "#dc8c00",
    "DCIS": "#64b400",
    "Endothelial": "#00be5a",
    "Invasive Tumor": "#00c3a5",
    "DCs": "#b487ff",
    "Macrophages": "#00b9dc",
    "Mast Cells": "#8f7aff",
    "Myoepi": "#00a5ff",
    "Perivascular-Like": "#d07cff",
    "Stromal": "#f069eb",
    "Unassigned": "#bdbdbd",
    "Unlabeled": "#bdbdbd",
    "Others": "#7f7f7f",
}
MARKER_GROUPS = {
    "Lymphoid": {"CD3D", "CD3E", "TRAC"},
    "Myeloid": {"LYZ", "CD68"},
    "vascular_stromal_cells": {"CDH5", "DCN", "COL5A2"},
    "Myoepi": {"KRT5", "KRT17", "KRT14", "TAGLN"},
    "Tumor": {"FASN", "ABCC11", "MKI67", "TOP2A"},
}


def _window_tuple(window: tuple[float, float, float, float] | list[float]) -> tuple[int, int, int, int]:
    if len(window) != 4:
        raise ValueError("window must contain four full-resolution coordinates: x0, y0, x1, y1.")
    x0, y0, x1, y1 = [int(round(float(value))) for value in window]
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid window: {(x0, y0, x1, y1)}")
    return x0, y0, x1, y1


def _window_name(window: tuple[int, int, int, int]) -> str:
    return "window_" + "_".join(str(int(value)) for value in window)


def _intersects(table: pd.DataFrame, window: tuple[int, int, int, int]) -> pd.Series:
    x0, y0, x1, y1 = window
    return (table["bbox_x1"] > x0) & (table["bbox_x0"] < x1) & (table["bbox_y1"] > y0) & (table["bbox_y0"] < y1)


def _hex_to_rgb(color: str) -> list[int]:
    text = color.lstrip("#")
    return [int(text[index : index + 2], 16) for index in (0, 2, 4)]


def _gene_group(gene: str) -> str:
    upper = gene.upper()
    for group, genes in MARKER_GROUPS.items():
        if upper in genes:
            return group
    return "Gene"


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text)).strip("_") or "value"


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    std = float(np.nanstd(values))
    if not np.isfinite(std) or std < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - float(np.nanmean(values))) / std


def _decode_array(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def _label_key(label: object) -> str:
    return " ".join(str(label).replace("_", " ").split()).lower()


def _load_type_merge_map(type_merge_json: str | Path | None) -> dict[str, str]:
    if type_merge_json is None:
        return {}
    payload = json.loads(Path(type_merge_json).read_text(encoding="utf-8"))
    merge = payload.get("merge", payload)
    out: dict[str, str] = {}
    for target, sources in merge.items():
        out[_label_key(target)] = str(target)
        for source in sources:
            out[_label_key(source)] = str(target)
    return out


def _merged_type_label(label: object, merge_map: dict[str, str]) -> str:
    text = str(label)
    merged = merge_map.get(_label_key(text), text)
    if merged in CELL_TYPE_COLORS:
        return merged
    return "Others"


def _polygon_columns(table: pd.DataFrame) -> tuple[str, str, str]:
    id_col = next((name for name in ["id", "xen_cell_id", "cell_id"] if name in table.columns), None)
    x_col = next((name for name in ["vertice_x", "vertex_x", "x"] if name in table.columns), None)
    y_col = next((name for name in ["vertice_y", "vertex_y", "y"] if name in table.columns), None)
    if id_col is None or x_col is None or y_col is None:
        raise ValueError("polygon table must contain id/x/y columns such as id, vertice_x, vertice_y.")
    return id_col, x_col, y_col


def _window_polygons(
    table: pd.DataFrame,
    window: tuple[int, int, int, int],
    *,
    id_col: str,
    x_col: str,
    y_col: str,
    window_label: str | None = None,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    x0, y0, x1, y1 = window
    if window_label is not None and "window_label" in table.columns:
        table = table.loc[table["window_label"].astype(str) == str(window_label)].copy()
    bounds = table.groupby(id_col, sort=False).agg(
        min_x=(x_col, "min"),
        max_x=(x_col, "max"),
        min_y=(y_col, "min"),
        max_y=(y_col, "max"),
    )
    keep_ids = bounds.index[
        (bounds["max_x"] >= x0) & (bounds["min_x"] <= x1) & (bounds["max_y"] >= y0) & (bounds["min_y"] <= y1)
    ]
    selected = table.loc[table[id_col].isin(keep_ids)].copy()
    polygons: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for cell_id, group in selected.groupby(id_col, sort=False):
        coords = group[[x_col, y_col]].to_numpy(dtype=float)
        if len(coords) < 3:
            continue
        polygons.append(np.column_stack([coords[:, 0] - x0, coords[:, 1] - y0]))
        row: dict[str, Any] = {"id": cell_id}
        if "celltype" in group.columns:
            row["celltype"] = group["celltype"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows), polygons


def plot_true_type(
    *,
    true_xen_type_csv: str | Path,
    window: tuple[float, float, float, float] | list[float],
    output_dir: str | Path,
    type_merge_json: str | Path | None = None,
    window_label: str | None = None,
    output_name: str = "xen_type.png",
    dpi: int = 150,
) -> dict[str, Any]:
    selected_window = _window_tuple(window)
    x0, y0, x1, y1 = selected_window
    table = pd.read_csv(true_xen_type_csv)
    id_col, x_col, y_col = _polygon_columns(table)
    if "celltype" not in table.columns:
        raise ValueError("true_xen_type_csv must contain a celltype column.")
    merge_map = _load_type_merge_map(type_merge_json)
    table["celltype"] = table["celltype"].map(lambda value: _merged_type_label(value, merge_map))
    meta, polygons = _window_polygons(
        table,
        selected_window,
        id_col=id_col,
        x_col=x_col,
        y_col=y_col,
    )
    colors = [CELL_TYPE_COLORS.get(label, CELL_TYPE_COLORS["Others"]) for label in meta.get("celltype", [])]

    output_png = Path(output_dir) / output_name
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.7, 5.49), dpi=dpi)
    if polygons:
        collection = PolyCollection(polygons, facecolors=colors, edgecolors=colors, linewidths=0.15)
        ax.add_collection(collection)
    ax.add_patch(Rectangle((0, 0), x1 - x0, y1 - y0, fill=False, edgecolor="black", linestyle=(0, (5, 5)), linewidth=1.8))
    ax.set_xlim(0, x1 - x0)
    ax.set_ylim(y1 - y0, 0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    label = window_label or _window_name(selected_window)
    ax.set_title(f"{label} real Xenium type", fontsize=13, pad=10)
    order = CELL_TYPE_ORDER + ["Unlabeled", "Others"]
    counts = meta["celltype"].value_counts().to_dict() if len(meta) else {}
    handles = [
        Line2D([0], [0], color=CELL_TYPE_COLORS[name], lw=2.2, label=name)
        for name in order
        if counts.get(name, 0) > 0
    ]
    if handles:
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.02, right=0.80, top=0.93, bottom=0.02)
    fig.savefig(output_png, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    return {"output_png": str(output_png)}


def plot_type_grid(
    *,
    grid_type_csv: str | Path,
    window: tuple[float, float, float, float] | list[float],
    output_dir: str | Path,
    output_name: str = "grid_type.png",
    dpi: int = 180,
) -> dict[str, Any]:
    selected_window = _window_tuple(window)
    x0, y0, x1, y1 = selected_window
    table = pd.read_csv(grid_type_csv)
    required = {"bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "predicted_type"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"grid_type_csv missing columns: {missing}")
    table = table.loc[_intersects(table, selected_window)].copy()

    image = np.full((y1 - y0, x1 - x0, 3), 255, dtype=np.uint8)
    counts = {name: 0 for name in CELL_TYPE_ORDER + ["Unassigned"]}
    for row in table.itertuples(index=False):
        gx0, gy0, gx1, gy1 = int(row.bbox_x0), int(row.bbox_y0), int(row.bbox_x1), int(row.bbox_y1)
        lx0 = max(gx0, x0) - x0
        ly0 = max(gy0, y0) - y0
        lx1 = min(gx1, x1) - x0
        ly1 = min(gy1, y1) - y0
        if lx1 <= lx0 or ly1 <= ly0:
            continue
        label = str(row.predicted_type)
        if label not in CELL_TYPE_COLORS:
            label = "Unassigned"
        image[ly0:ly1, lx0:lx1] = _hex_to_rgb(CELL_TYPE_COLORS[label])
        counts[label] = counts.get(label, 0) + 1

    output_png = Path(output_dir) / output_name
    output_png.parent.mkdir(parents=True, exist_ok=True)
    width = max(8.5, min(12.0, (x1 - x0) / 160.0))
    height = max(6.4, min(9.0, (y1 - y0) / 165.0))
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    ax.imshow(image, interpolation="nearest")
    ax.set_axis_off()
    present = [label for label in CELL_TYPE_ORDER + ["Unassigned"] if counts.get(label, 0)]
    handles = [
        Line2D([0], [0], color=CELL_TYPE_COLORS[label], lw=5, label=f"{label} ({counts[label]})")
        for label in present
    ]
    if handles:
        fig.legend(handles=handles, loc="center right", bbox_to_anchor=(0.995, 0.5), frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.79, top=0.99, bottom=0.01)
    fig.savefig(output_png, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    return {"output_png": str(output_png)}


def _adata_to_frame(grid_expr_h5ad: str | Path, genes: list[str]) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    adata = ad.read_h5ad(grid_expr_h5ad)
    present = [gene for gene in genes if gene in adata.var_names]
    if not present:
        raise ValueError(f"None of the requested genes are present: {genes}")
    obs = adata.obs.copy()
    matrix = adata[:, present].X
    if not isinstance(matrix, np.ndarray):
        matrix = matrix.toarray()
    return obs, np.asarray(matrix, dtype=np.float32), present


def plot_grid_expression(
    *,
    grid_expr_h5ad: str | Path,
    window: tuple[float, float, float, float] | list[float],
    output_dir: str | Path,
    gene: str = "EPCAM",
    output_name: str = "grid_expr.png",
    dpi: int = 180,
) -> dict[str, Any]:
    selected_window = _window_tuple(window)
    x0, y0, x1, y1 = selected_window
    obs, values, genes = _adata_to_frame(grid_expr_h5ad, [gene])
    gene = genes[0]
    table = obs.reset_index(drop=True).copy()
    table["value"] = values[:, 0]
    required = {"bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"grid_expr_h5ad obs missing columns: {missing}")
    table = table.loc[_intersects(table, selected_window)].copy()

    image = np.full((y1 - y0, x1 - x0), np.nan, dtype=np.float32)
    zvalues = _zscore(table["value"].to_numpy(dtype=np.float32))
    for row, zvalue in zip(table.itertuples(index=False), zvalues, strict=False):
        gx0, gy0, gx1, gy1 = int(row.bbox_x0), int(row.bbox_y0), int(row.bbox_x1), int(row.bbox_y1)
        lx0 = max(gx0, x0) - x0
        ly0 = max(gy0, y0) - y0
        lx1 = min(gx1, x1) - x0
        ly1 = min(gy1, y1) - y0
        if lx1 <= lx0 or ly1 <= ly0:
            continue
        image[ly0:ly1, lx0:lx1] = float(zvalue)

    masked = np.ma.masked_invalid(image)
    output_png = Path(output_dir) / output_name
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6.85, 7.22), dpi=dpi, facecolor="black")
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 0.035], wspace=0.035)
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])
    ax.set_facecolor("black")
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("black")
    im = ax.imshow(masked, cmap=cmap, vmin=-2, vmax=2, interpolation="nearest")
    ax.set_axis_off()
    ax.set_title(f"SpaceRec grid {gene}", color="white", fontsize=10, pad=54)
    fig.suptitle(f"{_window_name(selected_window)}: {_gene_group(gene)} / {gene}", color="white", fontsize=13, y=0.99)
    colorbar = fig.colorbar(im, cax=cax)
    colorbar.ax.tick_params(colors="white", labelsize=8)
    colorbar.outline.set_edgecolor("white")
    fig.savefig(output_png, facecolor="black", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    return {"output_png": str(output_png)}


def plot_xenium_expression(
    *,
    true_xen_expr_h5: str | Path,
    true_xen_type_csv: str | Path,
    window: tuple[float, float, float, float] | list[float],
    output_dir: str | Path,
    gene: str = "EPCAM",
    polygon_window_label: str | None = None,
    output_name: str = "xen_expr.png",
    dpi: int = 180,
) -> dict[str, Any]:
    selected_window = _window_tuple(window)
    x0, y0, x1, y1 = selected_window
    table = pd.read_csv(true_xen_type_csv, usecols=lambda col: col in {"window_label", "id", "xen_cell_id", "cell_id", "vertice_x", "vertex_x", "vertice_y", "vertex_y", "x", "y", "celltype"})
    id_col, x_col, y_col = _polygon_columns(table)
    meta, polygons = _window_polygons(
        table,
        selected_window,
        id_col=id_col,
        x_col=x_col,
        y_col=y_col,
    )

    with h5py.File(true_xen_expr_h5, "r") as handle:
        genes = _decode_array(handle["gene_name"][:])
        if gene not in genes:
            raise ValueError(f"Gene {gene!r} is not present in true_xen_expr_h5.")
        gene_index = genes.index(gene)
        cell_ids = _decode_array(handle["xen_cell_id"][:])
        values = np.asarray(handle["xenium_log1p_count"][:, gene_index], dtype=np.float32)
    value_map = dict(zip(cell_ids, values, strict=False))

    kept_polygons: list[np.ndarray] = []
    raw_values: list[float] = []
    for cell_id, polygon in zip(meta["id"].tolist(), polygons, strict=False):
        value = value_map.get(str(cell_id))
        if value is None:
            continue
        kept_polygons.append(polygon)
        raw_values.append(float(value))

    zvalues = _zscore(np.asarray(raw_values, dtype=np.float32))
    output_png = Path(output_dir) / output_name
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6.85, 7.22), dpi=dpi, facecolor="black")
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 0.035], wspace=0.035)
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])
    ax.set_facecolor("black")
    cmap = plt.get_cmap("magma")
    norm = plt.Normalize(vmin=-2, vmax=2)
    if kept_polygons:
        collection = PolyCollection(
            kept_polygons,
            array=zvalues,
            cmap=cmap,
            norm=norm,
            edgecolors="black",
            linewidths=0.08,
        )
        ax.add_collection(collection)
    else:
        collection = None
    ax.set_xlim(0, x1 - x0)
    ax.set_ylim(y1 - y0, 0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"Xenium {gene}", color="white", fontsize=10, pad=54)
    label = polygon_window_label or _window_name(selected_window)
    fig.suptitle(f"{label}: {_gene_group(gene)} / {gene}", color="white", fontsize=13, y=0.99)
    mappable = collection if collection is not None else plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(mappable, cax=cax)
    colorbar.ax.tick_params(colors="white", labelsize=8)
    colorbar.outline.set_edgecolor("white")
    fig.savefig(output_png, facecolor="black", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    return {"output_png": str(output_png)}
