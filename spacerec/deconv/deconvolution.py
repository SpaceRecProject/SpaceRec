from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
CRC_CELL_TYPE_ORDER = [
    "B cells",
    "T cells",
    "Tumor",
    "Myeloid",
    "Fibroblast",
    "Endothelial",
    "Intestinal Epithelial",
    "Smooth Muscle",
    "Neuronal",
]
BRCA_CELL_TYPE_ORDER = [
    "B Cells",
    "CD4+ T Cells",
    "CD8+ T Cells",
    "DCIS 1",
    "DCIS 2",
    "Endothelial",
    "Invasive Tumor",
    "IRF7+ DCs",
    "LAMP3+ DCs",
    "Macrophages 1",
    "Macrophages 2",
    "Mast Cells",
    "Myoepi ACTA2+",
    "Myoepi KRT15+",
    "Perivascular-Like",
    "Prolif Invasive Tumor",
    "Stromal",
]
BRCA_MERGED11_CELL_TYPE_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("B Cells", ("B Cells",)),
    ("T Cells", ("CD4+ T Cells", "CD8+ T Cells")),
    ("DCIS", ("DCIS 1", "DCIS 2")),
    ("Endothelial", ("Endothelial",)),
    ("Invasive Tumor", ("Invasive Tumor", "Prolif Invasive Tumor")),
    ("DCs", ("IRF7+ DCs", "LAMP3+ DCs")),
    ("Macrophages", ("Macrophages 1", "Macrophages 2")),
    ("Mast Cells", ("Mast Cells",)),
    ("Myoepi", ("Myoepi ACTA2+", "Myoepi KRT15+")),
    ("Perivascular-Like", ("Perivascular-Like",)),
    ("Stromal", ("Stromal",)),
]


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def require_dir(path: Path) -> Path:
    if not path.is_dir():
        raise NotADirectoryError(path)
    return path


def decode(values) -> list[str]:
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in values]


def read_h5ad_strings(group: h5py.Group, key: str) -> list[str]:
    return decode(group[key][:])


def read_h5ad_index(group: h5py.Group) -> list[str]:
    key = group.attrs.get("_index", "_index")
    if isinstance(key, bytes):
        key = key.decode("utf-8")
    return read_h5ad_strings(group, str(key))


def read_obs_column(obs_group: h5py.Group, key: str) -> np.ndarray:
    node = obs_group[key]
    if isinstance(node, h5py.Dataset):
        return np.asarray(decode(node[:]), dtype=object)
    categories = np.asarray(decode(node["categories"][:]), dtype=object)
    codes = node["codes"][:].astype(int)
    values = np.full(codes.shape, None, dtype=object)
    valid = codes >= 0
    values[valid] = categories[codes[valid]]
    return values


def read_bool_dataset(group: h5py.Group, key: str, n: int) -> np.ndarray:
    if key not in group:
        return np.ones(n, dtype=bool)
    return np.asarray(group[key][:], dtype=bool)


def read_sparse_group(group: h5py.Group):
    shape = tuple(int(value) for value in group.attrs["shape"])
    data = group["data"][:]
    indices = group["indices"][:]
    indptr = group["indptr"][:]
    encoding = group.attrs.get("encoding-type", "")
    if isinstance(encoding, bytes):
        encoding = encoding.decode("utf-8")
    if encoding == "csr_matrix":
        return sparse.csr_matrix((data, indices, indptr), shape=shape)
    if encoding == "csc_matrix":
        return sparse.csc_matrix((data, indices, indptr), shape=shape)
    raise ValueError(f"Unsupported sparse encoding: {encoding!r}")


def collapse_duplicate_rows(matrix, names: list[str]):
    index = pd.Index([str(name) for name in names])
    if index.is_unique:
        return matrix, index.tolist(), 0
    unique_names, inverse = np.unique(index.to_numpy(dtype=str), return_inverse=True)
    indicator = sparse.csr_matrix(
        (np.ones(len(index), dtype=np.float32), (inverse, np.arange(len(index)))),
        shape=(len(unique_names), len(index)),
    )
    return indicator @ matrix, unique_names.astype(str).tolist(), int(len(index) - len(unique_names))


def read_tissue_positions(path: Path) -> pd.DataFrame:
    table = pd.read_csv(require_file(path))
    if "barcode" in table.columns:
        return table
    table = pd.read_csv(
        path,
        header=None,
        names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"],
    )
    return table


def export_spatial(visium_dir: Path, rctd_input_dir: Path, sample_id: str) -> dict[str, object]:
    visium_dir = require_dir(Path(visium_dir))
    rctd_input_dir.mkdir(parents=True, exist_ok=True)

    existing = [
        visium_dir / f"{sample_id}_counts.mtx",
        visium_dir / f"{sample_id}_genes.tsv",
        visium_dir / f"{sample_id}_spots.tsv",
        visium_dir / f"{sample_id}_coords.csv",
    ]
    if all(path.is_file() for path in existing):
        for path in existing:
            target = rctd_input_dir / path.name
            if target.exists():
                target.unlink()
            target.symlink_to(path.resolve())
        spots = read_h5ad_strings_from_text(existing[2])
        genes = read_h5ad_strings_from_text(existing[1])
        return {
            "source": "precomputed_rctd_files",
            "visium_dir": str(visium_dir),
            "n_spatial_genes": int(len(genes)),
            "n_spatial_spots": int(len(spots)),
            "n_duplicate_gene_rows_collapsed": 0,
        }

    h5_path = require_file(visium_dir / "st_filtered_feature_bc_matrix.h5")
    tissue_positions = require_file(visium_dir / "tissue_positions.csv")
    with h5py.File(h5_path, "r") as handle:
        matrix = handle["matrix"]
        shape = tuple(int(value) for value in matrix["shape"][:])
        counts = sparse.csc_matrix((matrix["data"][:], matrix["indices"][:], matrix["indptr"][:]), shape=shape)
        barcodes = decode(matrix["barcodes"][:])
        genes = decode(matrix["features"]["name"][:])

    counts, genes, n_duplicate_gene_rows_collapsed = collapse_duplicate_rows(counts, genes)
    mmwrite(rctd_input_dir / f"{sample_id}_counts.mtx", counts)
    (rctd_input_dir / f"{sample_id}_genes.tsv").write_text("\n".join(genes) + "\n", encoding="utf-8")
    (rctd_input_dir / f"{sample_id}_spots.tsv").write_text("\n".join(barcodes) + "\n", encoding="utf-8")

    coords = read_tissue_positions(tissue_positions).set_index("barcode")
    missing = [barcode for barcode in barcodes if barcode not in coords.index]
    if missing:
        raise ValueError(f"{len(missing)} spatial barcodes missing from tissue_positions.csv; first={missing[0]}")
    if {"pxl_col_in_fullres", "pxl_row_in_fullres"}.issubset(coords.columns):
        out = coords.loc[barcodes, ["pxl_col_in_fullres", "pxl_row_in_fullres"]].copy()
    elif {"image_x", "image_y"}.issubset(coords.columns):
        out = coords.loc[barcodes, ["image_x", "image_y"]].copy()
    else:
        raise KeyError(f"{tissue_positions} is missing full-resolution x/y coordinate columns.")
    out.columns = ["x", "y"]
    out.to_csv(rctd_input_dir / f"{sample_id}_coords.csv")
    return {
        "source": "st_filtered_feature_bc_matrix.h5",
        "st_h5": str(h5_path),
        "tissue_positions": str(tissue_positions),
        "n_spatial_genes": int(len(genes)),
        "n_spatial_spots": int(len(barcodes)),
        "n_duplicate_gene_rows_collapsed": int(n_duplicate_gene_rows_collapsed),
    }


def read_h5ad_strings_from_text(path: Path) -> list[str]:
    return [line.strip() for line in require_file(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def export_scrna_reference(
    sc_ref_h5ad: Path,
    method_input_dir: Path,
    annotation_column: str,
    use_consensus_keep: bool,
    patient_column: str = "Patient",
    patient_suffix: str | None = None,
    allowed_cell_types: list[str] | None = None,
) -> dict[str, object]:
    sc_ref_h5ad = require_file(Path(sc_ref_h5ad))
    method_input_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(sc_ref_h5ad, "r") as handle:
        obs = handle["obs"]
        var = handle["var"]
        cell_ids = read_h5ad_index(obs)
        genes = read_h5ad_index(var)
        labels = read_obs_column(obs, annotation_column)
        keep = np.asarray([value is not None and str(value) not in {"", "nan", "NA"} for value in labels], dtype=bool)
        if allowed_cell_types is not None:
            allowed = set(allowed_cell_types)
            keep &= np.asarray([str(value) in allowed for value in labels], dtype=bool)
        if patient_suffix is not None:
            patients = read_obs_column(obs, patient_column)
            keep &= np.asarray([str(value).endswith(patient_suffix) for value in patients], dtype=bool)
        if use_consensus_keep:
            keep &= read_bool_dataset(obs, "consensus_keep", len(cell_ids))
        counts_group = handle["layers"]["counts"] if "layers" in handle and "counts" in handle["layers"] else handle["X"]
        counts = read_sparse_group(counts_group).tocsr()

    keep_idx = np.flatnonzero(keep)
    filtered_counts = counts[keep_idx, :].T.tocoo().astype(np.float32)
    filtered_cells = [cell_ids[int(index)] for index in keep_idx]
    filtered_labels = [str(labels[int(index)]) for index in keep_idx]
    if allowed_cell_types is None:
        cell_types = pd.Index(filtered_labels).drop_duplicates().tolist()
    else:
        present = set(filtered_labels)
        cell_types = [cell_type for cell_type in allowed_cell_types if cell_type in present]

    mmwrite(method_input_dir / "reference_counts.mtx", filtered_counts)
    (method_input_dir / "reference_genes.tsv").write_text("\n".join(genes) + "\n", encoding="utf-8")
    (method_input_dir / "reference_cells.tsv").write_text("\n".join(filtered_cells) + "\n", encoding="utf-8")
    pd.DataFrame({"cell_id": filtered_cells, "cell_type": filtered_labels}).to_csv(
        method_input_dir / "reference_metadata.csv",
        index=False,
    )
    (method_input_dir / "celltype_order.tsv").write_text("\n".join(cell_types) + "\n", encoding="utf-8")
    return {
        "sc_ref_h5ad": str(sc_ref_h5ad),
        "annotation_column": annotation_column,
        "use_consensus_keep": bool(use_consensus_keep),
        "patient_column": patient_column,
        "patient_suffix": patient_suffix,
        "allowed_cell_types": allowed_cell_types,
        "n_source_cells": int(len(cell_ids)),
        "n_reference_cells": int(len(filtered_cells)),
        "n_reference_genes": int(len(genes)),
        "n_cell_types": int(len(cell_types)),
        "cell_types": cell_types,
    }


def prepare_deconvolution_inputs(
    *,
    visium_dir: Path,
    sc_ref_h5ad: Path,
    output_dir: Path,
    sample_id: str,
    input_subdir: str = "input",
    annotation_column: str,
    use_consensus_keep: bool,
    patient_column: str = "Patient",
    patient_suffix: str | None = None,
    allowed_cell_types: list[str] | None = None,
) -> dict[str, object]:
    rctd_input_dir = Path(output_dir) / input_subdir
    method_input_dir = rctd_input_dir / "preprocessing_deconv"
    rctd_input_dir.mkdir(parents=True, exist_ok=True)
    method_input_dir.mkdir(parents=True, exist_ok=True)

    spatial = export_spatial(Path(visium_dir), rctd_input_dir, sample_id)
    reference = export_scrna_reference(
        Path(sc_ref_h5ad),
        method_input_dir,
        annotation_column,
        use_consensus_keep,
        patient_column=patient_column,
        patient_suffix=patient_suffix,
        allowed_cell_types=allowed_cell_types,
    )
    spots = read_h5ad_strings_from_text(rctd_input_dir / f"{sample_id}_spots.tsv")
    (method_input_dir / "spot_order_spacerec.tsv").write_text("\n".join(spots) + "\n", encoding="utf-8")

    summary = {
        "sample_id": sample_id,
        "input_subdir": input_subdir,
        "rctd_input_dir": str(rctd_input_dir),
        "method_input_dir": str(method_input_dir),
        "spatial": spatial,
        "reference": reference,
    }
    (rctd_input_dir / "preprocessing_deconv_input_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def remove_outputs_for_prefix(output_dir: Path, output_prefix: str) -> None:
    for suffix in [
        "_proportions.csv",
        "_proportions_spacerec.csv",
        "_proportions_merged11_spacerec.csv",
        "_result.rds",
        "_validation_summary.json",
        "_merged11_validation_summary.json",
    ]:
        path = output_dir / f"{output_prefix}{suffix}"
        if path.exists():
            path.unlink()


def run_command(command: list[str], env: dict[str, str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def validate_rctd_output(output_dir: Path, output_prefix: str, method_input_dir: Path, rctd_input_dir: Path, sample_id: str) -> dict[str, object]:
    paths = {
        "proportions": output_dir / f"{output_prefix}_proportions.csv",
        "spacerec": output_dir / f"{output_prefix}_proportions_spacerec.csv",
        "rds": output_dir / f"{output_prefix}_result.rds",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    frame = pd.read_csv(paths["spacerec"], index_col=0)
    expected = read_h5ad_strings_from_text(method_input_dir / "celltype_order.tsv")
    if sorted(frame.columns) != sorted(expected):
        raise ValueError(f"Unexpected RCTD columns: {list(frame.columns)}; expected set: {expected}")
    expected_spots = read_h5ad_strings_from_text(rctd_input_dir / f"{sample_id}_spots.tsv")
    if list(frame.index) != expected_spots:
        raise ValueError("RCTD output spot names do not match the Visium spot order.")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite values in RCTD proportions.")
    row_sums = values.sum(axis=1)
    summary = {
        "prefix": output_prefix,
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "columns": list(frame.columns),
        "spot_names_match_visium": True,
        "row_sum_min": float(row_sums.min()),
        "row_sum_median": float(np.median(row_sums)),
        "row_sum_max": float(row_sums.max()),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    (output_dir / f"{output_prefix}_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def merge_brca_rctd_output(output_dir: Path, output_prefix: str) -> dict[str, object]:
    source_path = output_dir / f"{output_prefix}_proportions_spacerec.csv"
    merged_path = output_dir / f"{output_prefix}_proportions_merged11_spacerec.csv"
    source = pd.read_csv(require_file(source_path), index_col=0)
    merged = pd.DataFrame(index=source.index)
    for merged_name, source_names in BRCA_MERGED11_CELL_TYPE_GROUPS:
        missing = [name for name in source_names if name not in source.columns]
        if missing:
            raise ValueError(f"Missing BRCA RCTD columns for {merged_name}: {missing}")
        merged[merged_name] = source.loc[:, list(source_names)].sum(axis=1)
    values = np.clip(merged.to_numpy(dtype=float), 0.0, None)
    row_sums = values.sum(axis=1, keepdims=True)
    values = np.divide(values, np.maximum(row_sums, 1e-12), out=np.zeros_like(values), where=row_sums > 0)
    merged.loc[:, :] = values
    merged.to_csv(merged_path)
    normalized_row_sums = values.sum(axis=1)
    summary = {
        "source": str(source_path),
        "output": str(merged_path),
        "shape": [int(merged.shape[0]), int(merged.shape[1])],
        "source_columns": list(source.columns),
        "columns": list(merged.columns),
        "merge_groups": {name: list(sources) for name, sources in BRCA_MERGED11_CELL_TYPE_GROUPS},
        "row_sum_min": float(normalized_row_sums.min()),
        "row_sum_median": float(np.median(normalized_row_sums)),
        "row_sum_max": float(normalized_row_sums.max()),
    }
    (output_dir / f"{output_prefix}_merged11_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_deconvolution(
    *,
    visium_dir: Path,
    sc_ref_h5ad: Path,
    output_dir: Path,
    sample_id: str,
    input_subdir: str = "input",
    annotation_column: str = "Level1",
    use_consensus_keep: bool = True,
    reference_filter: str | None = None,
    output_prefix: str | None = None,
    umi_min: int | None = None,
    max_cores: int = 8,
    force: bool = False,
    run_rctd: bool = True,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    rctd_output_dir = output_dir / "rctd"
    rctd_output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix or f"RCTD_{sample_id}"
    selected_umi_min = 30 if umi_min is None and sample_id == "BREAST" else (100 if umi_min is None else int(umi_min))
    if force:
        remove_outputs_for_prefix(rctd_output_dir, prefix)
    patient_suffix = None
    allowed_cell_types = None
    if reference_filter == "allcrc":
        patient_suffix = "CRC"
        allowed_cell_types = CRC_CELL_TYPE_ORDER
    elif reference_filter == "p2crc":
        patient_suffix = "P2CRC"
        allowed_cell_types = CRC_CELL_TYPE_ORDER
    elif reference_filter in {None, "allscrna"}:
        if sample_id == "BREAST":
            allowed_cell_types = BRCA_CELL_TYPE_ORDER
    else:
        raise ValueError(f"Unsupported reference_filter: {reference_filter!r}")

    prepared = prepare_deconvolution_inputs(
        visium_dir=Path(visium_dir),
        sc_ref_h5ad=Path(sc_ref_h5ad),
        output_dir=output_dir,
        input_subdir=input_subdir,
        sample_id=sample_id,
        annotation_column=annotation_column,
        use_consensus_keep=use_consensus_keep,
        patient_suffix=patient_suffix,
        allowed_cell_types=allowed_cell_types,
    )
    method_input_dir = Path(prepared["method_input_dir"])
    summary = {
        "prepared": prepared,
        "rctd": None,
        "output_dir": str(output_dir),
        "input_subdir": input_subdir,
        "output_prefix": prefix,
        "reference_filter": reference_filter,
        "umi_min": selected_umi_min,
    }
    if not run_rctd:
        return summary

    env = os.environ.copy()
    threads = str(max_cores)
    env["OMP_NUM_THREADS"] = threads
    env["MKL_NUM_THREADS"] = threads
    env["OPENBLAS_NUM_THREADS"] = threads

    run_command(
        [
            "Rscript",
            str(SCRIPT_DIR / "make_reference_seurat.R"),
            "--method-input-dir",
            str(method_input_dir),
            "--output-rds",
            str(method_input_dir / "reference_seurat.rds"),
        ],
        env,
    )
    run_command(
        [
            "Rscript",
            str(SCRIPT_DIR / "rctd.R"),
            "--rctd-input-dir",
            str(Path(prepared["rctd_input_dir"])),
            "--sc-ref-rds",
            str(method_input_dir / "reference_seurat.rds"),
            "--output-dir",
            str(rctd_output_dir),
            "--sample-id",
            sample_id,
            "--annotation-column",
            "Annotation",
            "--umi-min",
            str(selected_umi_min),
            "--max-cores",
            str(max_cores),
            "--output-prefix",
            prefix,
        ],
        env,
    )
    summary["rctd"] = validate_rctd_output(
        rctd_output_dir,
        prefix,
        method_input_dir,
        Path(prepared["rctd_input_dir"]),
        sample_id,
    )
    if sample_id == "BREAST":
        summary["rctd_merged11"] = merge_brca_rctd_output(rctd_output_dir, prefix)
    (output_dir / f"{prefix}_deconvolution_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
