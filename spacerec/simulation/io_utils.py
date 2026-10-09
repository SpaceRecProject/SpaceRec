from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy import sparse


def decode_strings(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value) for value in values]


def norm_gene(value: str) -> str:
    return value.strip().split(".")[0].upper()


def read_10x_h5(path: str | Path) -> tuple[sparse.csc_matrix, np.ndarray, dict[str, np.ndarray]]:
    with h5py.File(path, "r") as handle:
        group = handle["matrix"]
        shape = tuple(int(value) for value in group["shape"][()])
        matrix = sparse.csc_matrix(
            (
                np.asarray(group["data"][()]),
                np.asarray(group["indices"][()]),
                np.asarray(group["indptr"][()]),
            ),
            shape=shape,
        )
        barcodes = group["barcodes"][()]
        features = {key: group["features"][key][()] for key in group["features"].keys()}
    return matrix, barcodes, features


def read_10x_features(path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with h5py.File(path, "r") as handle:
        feature_group = handle["matrix/features"]
        features = {key: feature_group[key][()] for key in feature_group.keys()}
        names = np.asarray(decode_strings(feature_group["name"][()]), dtype=object)
    return names, features


def write_10x_h5(
    path: str | Path,
    matrix: sparse.spmatrix,
    barcodes: np.ndarray,
    features: dict[str, np.ndarray],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    matrix = matrix.tocsc()
    with h5py.File(output, "w") as handle:
        group = handle.create_group("matrix")
        group.create_dataset("data", data=matrix.data.astype(np.int32), compression="gzip")
        group.create_dataset("indices", data=matrix.indices.astype(np.int64), compression="gzip")
        group.create_dataset("indptr", data=matrix.indptr.astype(np.int64), compression="gzip")
        group.create_dataset("shape", data=np.asarray(matrix.shape, dtype=np.int32))
        group.create_dataset("barcodes", data=np.asarray(barcodes, dtype="S"))
        feature_group = group.create_group("features")
        for key, values in features.items():
            feature_group.create_dataset(key, data=values)


def load_ref_genes(path: str | Path) -> list[str]:
    with h5py.File(path, "r") as handle:
        if "var/_index" not in handle:
            raise ValueError(f"{path} does not contain var/_index")
        return decode_strings(handle["var/_index"][:])


def filter_10x_h5_to_ref_genes(
    *,
    synthetic_h5: str | Path,
    sc_ref_h5ad: str | Path,
    output_h5: str | Path,
    summary_json: str | Path | None = None,
) -> dict[str, Any]:
    ref_genes = load_ref_genes(sc_ref_h5ad)
    matrix, barcodes, features = read_10x_h5(synthetic_h5)
    syn_names = decode_strings(features["name"])
    syn_ids = decode_strings(features["id"])
    name_to_index = {norm_gene(name): index for index, name in enumerate(syn_names)}
    id_to_index = {norm_gene(gene_id): index for index, gene_id in enumerate(syn_ids)}

    selected_indices: list[int] = []
    match_modes: list[str] = []
    seen_indices: set[int] = set()
    for ref_gene in ref_genes:
        key = norm_gene(ref_gene)
        index = name_to_index.get(key)
        mode = "name"
        if index is None:
            index = id_to_index.get(key)
            mode = "id"
        if index is None or index in seen_indices:
            continue
        selected_indices.append(index)
        match_modes.append(mode)
        seen_indices.add(index)
    if not selected_indices:
        raise ValueError("No overlapping genes between scRNA reference and synthetic H5.")

    selected = np.asarray(selected_indices, dtype=np.int64)
    filtered_matrix = matrix[selected, :].tocsc()
    n_features = int(matrix.shape[0])
    filtered_features: dict[str, np.ndarray] = {}
    for key, values in features.items():
        if getattr(values, "shape", ()) and values.shape[0] == n_features:
            filtered_features[key] = values[selected]
        else:
            filtered_features[key] = values
    write_10x_h5(output_h5, filtered_matrix, barcodes, filtered_features)

    summary = {
        "sc_ref_genes": int(len(ref_genes)),
        "synthetic_genes": int(matrix.shape[0]),
        "filtered_genes": int(filtered_matrix.shape[0]),
        "spots": int(filtered_matrix.shape[1]),
        "name_matches": int(sum(mode == "name" for mode in match_modes)),
        "id_matches": int(sum(mode == "id" for mode in match_modes)),
        "output": str(Path(output_h5)),
        "first_filtered_genes": [syn_names[index] for index in selected_indices[:10]],
    }
    if summary_json is not None:
        Path(summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary

