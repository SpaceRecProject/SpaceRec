from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from ..deconv.deconvolution import BRCA_MERGE_JSON, load_brca_merge_groups


ANNOTATION_R_OUTPUTS = [
    "annotation_10xg.csv",
    "cell_groups.csv",
    "cell_type_frequencies.csv",
    "prediction_score_summary.csv",
    "reference_xenium_gene_expression_correlation.csv",
    "reference_xenium_gene_expression_correlation.png",
    "embeddings/step04_1_flex_umap_metadata.csv",
    "embeddings/step04_2_xenium_sketch_umap_metadata.csv",
    "embeddings/step06_xenium_full_umap_metadata.csv",
    "figures/step02_1_flex_qc_violin.png",
    "figures/step03_2_xenium_qc_violin.png",
    "figures/step03_2_xenium_spatial_ncount_log_local_centroid.png",
    "figures/step03_2_xenium_spatial_nfeature_log_local_centroid.png",
    "figures/step03_3_expression_correlation.png",
    "figures/step04_1_flex_umap_cell_type.png",
    "figures/step04_1_flex_umap_clusters.png",
    "figures/step04_2_xenium_sketch_spatial_clusters_local_centroid.png",
    "figures/step04_2_xenium_sketch_umap_clusters.png",
    "figures/step05_xenium_sketch_spatial_predicted_id_local_centroid.png",
    "figures/step05_xenium_sketch_umap_predicted_id.png",
    "figures/step06_xenium_full_spatial_cell_type_local_centroid.png",
    "figures/step06_xenium_full_umap_cell_type.png",
    "figures/step06_xenium_full_umap_clusters.png",
    "figures/step07_1_per_cell_type_expression_correlation.png",
    "figures/step07_2_cell_type_frequency.png",
    "figures/step07_3_prediction_score_full_umap.png",
    "figures/step07_3_prediction_score_violin.png",
    "prism/step02_1_flex_qc_prism.csv",
    "prism/step03_2_xenium_qc_prism.csv",
    "prism/step03_3_expression_correlation_prism.csv",
    "prism/step03_3_expression_correlation_summary.csv",
    "prism/step05_label_transfer_sketch_predictions_prism.csv",
    "prism/step07_1_per_cell_type_expression_correlation_prism.csv",
    "prism/step07_1_per_cell_type_expression_correlation_summary.csv",
    "prism/step07_2_cell_type_counts_prism.csv",
    "prism/step07_2_cell_type_frequency_long.csv",
    "prism/step07_2_cell_type_frequency_prism.csv",
    "prism/step07_3_prediction_scores_prism.csv",
]

BPCells_REQUIRED_FILES = [
    "col_names",
    "idxptr",
    "index_data",
    "index_idx",
    "index_idx_offsets",
    "index_starts",
    "row_names",
    "shape",
    "storage_order",
    "val",
    "version",
]


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_command(command: list[str], *, env: dict[str, str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Annotation command failed with exit code {process.returncode}. See {log_path}.")


def _r_command(script: Path, *, r_env_prefix: str | Path | None) -> list[str]:
    if r_env_prefix is None:
        return ["Rscript", str(script)]
    return ["conda", "run", "-p", str(Path(r_env_prefix)), "Rscript", str(script)]


def _require_files(root: Path, names: list[str]) -> list[str]:
    missing: list[str] = []
    empty: list[str] = []
    for name in names:
        path = root / name
        if not path.exists():
            missing.append(name)
        elif path.is_file() and path.stat().st_size == 0:
            empty.append(name)
    if missing or empty:
        problems = []
        if missing:
            problems.append(f"missing={missing}")
        if empty:
            problems.append(f"empty={empty}")
        raise FileNotFoundError(f"Incomplete annotation outputs under {root}: {'; '.join(problems)}")
    return [str(root / name) for name in names]


def validate_annotation_outputs(
    output_dir: str | Path,
    *,
    aligned_cells_csv: str | Path | None = None,
    include_wrapper_outputs: bool = True,
    merge_json: str | Path = BRCA_MERGE_JSON,
) -> dict[str, Any]:
    root = Path(output_dir)
    merge_groups = load_brca_merge_groups(merge_json)
    merged_types = [name for name, _sources in merge_groups]
    allowed_merged_types = [*merged_types, "Others"]
    source_types = sorted({source for _name, sources in merge_groups for source in sources})
    allowed_source_types = sorted({*source_types, "Stromal & T Cell Hybrid", "T Cell & Tumor Hybrid"})
    annotation_csv = root / "annotation_10xg.csv"
    cell_groups_csv = root / "cell_groups.csv"
    required_outputs = [*ANNOTATION_R_OUTPUTS]
    if include_wrapper_outputs:
        required_outputs.extend(["annotation_config.json", "annotation_10xg.log", "annotation_summary.json"])
    required_files = _require_files(root, required_outputs)
    bpcells_files = _require_files(
        root,
        [
            *(f"bpcells/reference_counts/{name}" for name in BPCells_REQUIRED_FILES),
            *(f"bpcells/xenium_counts/{name}" for name in BPCells_REQUIRED_FILES),
        ],
    )
    annotation = pd.read_csv(annotation_csv)
    required = {"id", "Count", "TenXG_anno", "barcode", "predicted_id", "cluster_full"}
    missing = required.difference(annotation.columns)
    if missing:
        raise ValueError(f"{annotation_csv} is missing columns: {sorted(missing)}")
    observed_types = sorted(annotation["TenXG_anno"].dropna().astype(str).unique())
    unexpected_types = [name for name in observed_types if name not in set(allowed_merged_types)]
    if unexpected_types:
        raise ValueError(f"{annotation_csv} contains cell types outside merged classes plus Others: {unexpected_types}")
    if "TenXG_anno_19" in annotation.columns:
        observed_source_types = sorted(annotation["TenXG_anno_19"].dropna().astype(str).unique())
        unexpected_source_types = [name for name in observed_source_types if name not in set(allowed_source_types)]
        if unexpected_source_types:
            raise ValueError(f"{annotation_csv} contains source cell types outside expected BRCA 19 classes: {unexpected_source_types}")
    summary: dict[str, Any] = {
        "annotation_csv": str(annotation_csv),
        "cell_groups_csv": str(cell_groups_csv),
        "required_outputs": required_files,
        "bpcells_outputs": bpcells_files,
        "n_annotations": int(len(annotation)),
        "n_annotated_non_null": int(annotation["TenXG_anno"].notna().sum()),
        "n_cell_types": int(annotation["TenXG_anno"].dropna().nunique()),
        "cell_type_counts": annotation["TenXG_anno"].dropna().astype(str).value_counts().to_dict(),
        "merge_json": str(Path(merge_json)),
        "expected_merged_cell_types": allowed_merged_types,
        "expected_source_cell_types": allowed_source_types,
    }
    if "TenXG_anno_19" in annotation.columns:
        summary["n_source_cell_types"] = int(annotation["TenXG_anno_19"].dropna().nunique())
        summary["source_cell_type_counts"] = annotation["TenXG_anno_19"].dropna().astype(str).value_counts().to_dict()
    if aligned_cells_csv is not None and Path(aligned_cells_csv).exists():
        aligned = pd.read_csv(aligned_cells_csv, usecols=["cell_id"])
        summary["n_aligned_cells"] = int(len(aligned))
        summary["aligned_cell_id_overlap"] = int(
            len(set(aligned["cell_id"].astype(str)).intersection(set(annotation["id"].astype(str))))
        )
    return summary


def run_annotation(
    *,
    xen_dir: str | Path,
    aligned_dir: str | Path,
    sc_ref_h5ad: str | Path,
    output_dir: str | Path,
    r_env_prefix: str | Path | None = None,
    merge_json: str | Path = BRCA_MERGE_JSON,
    force: bool = False,
) -> dict[str, Any]:
    root = _package_root()
    xen_root = Path(xen_dir)
    aligned_root = Path(aligned_dir)
    out = Path(output_dir)
    annotation_csv = out / "annotation_10xg.csv"
    if annotation_csv.exists() and not force:
        validation = validate_annotation_outputs(
            out,
            aligned_cells_csv=aligned_root / "aligned_cells.csv",
            merge_json=merge_json,
        )
        validation.update({"reused_existing": True, "output_dir": str(out)})
        return validation

    required_inputs = {
        "sc_ref_h5ad": Path(sc_ref_h5ad),
        "xenium_outs": xen_root / "outs",
        "cell_feature_matrix_h5": xen_root / "outs" / "cell_feature_matrix.h5",
        "cells_parquet": xen_root / "outs" / "cells.parquet",
        "aligned_cells_csv": aligned_root / "aligned_cells.csv",
    }
    missing_inputs = [name for name, path in required_inputs.items() if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing annotation inputs: {missing_inputs}")

    out.mkdir(parents=True, exist_ok=True)
    config = {
        "sc_ref_h5ad": str(Path(sc_ref_h5ad).resolve()),
        "xenium_outs": str((xen_root / "outs").resolve()),
        "cells_parquet": str((xen_root / "outs" / "cells.parquet").resolve()),
        "output_dir": str(out.resolve()),
        "merge_json": str(Path(merge_json).resolve()),
    }
    config_path = out / "annotation_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    script = Path(__file__).resolve().parent / "brca_10xg_annotation.R"
    env = os.environ.copy()
    env["SPACEREC_ANNO_CONFIG"] = str(config_path.resolve())
    env["STRICT_XENIUM_OUTS"] = "1"
    log_path = out / "annotation_10xg.log"
    _run_command(_r_command(script, r_env_prefix=r_env_prefix), env=env, cwd=root, log_path=log_path)

    validation = validate_annotation_outputs(
        out,
        aligned_cells_csv=aligned_root / "aligned_cells.csv",
        include_wrapper_outputs=False,
        merge_json=merge_json,
    )
    summary = {
        "task": "annotation_10xg",
        "reused_existing": False,
        "xen_dir": str(xen_root),
        "aligned_dir": str(aligned_root),
        "sc_ref_h5ad": str(Path(sc_ref_h5ad)),
        "output_dir": str(out),
        "r_env_prefix": None if r_env_prefix is None else str(r_env_prefix),
        "merge_json": str(Path(merge_json)),
        "config_json": str(config_path),
        "log": str(log_path),
        "validation": validation,
    }
    (out / "annotation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["validation"] = validate_annotation_outputs(
        out,
        aligned_cells_csv=aligned_root / "aligned_cells.csv",
        merge_json=merge_json,
    )
    (out / "annotation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


__all__ = ["run_annotation", "validate_annotation_outputs"]
