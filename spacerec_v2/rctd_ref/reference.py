from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from ..deconv.deconvolution import (
    BRCA_CELL_TYPE_ORDER,
    BRCA_MERGE_JSON,
    export_scrna_reference,
    load_brca_merge_groups,
    r_command,
)
SCRIPT_DIR = Path(__file__).resolve().parent


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read_10x_genes(path: Path) -> list[str]:
    with h5py.File(path, "r") as handle:
        values = handle["matrix/features/name"][:]
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def _default_positions_csv(synthetic_h5: Path) -> Path:
    root = synthetic_h5.parent
    for name in [
        "synthetic_tissue_positions_transcript_filtered.csv",
        "tissue_positions.csv",
    ]:
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No tissue positions CSV found next to {synthetic_h5}.")


def _run_logged(command: list[str], *, env: dict[str, str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        process = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(f"RCTD reference command failed with exit code {process.returncode}. See {log_path}.")


def validate_rctd_reference(
    *,
    synthetic_h5: str | Path,
    output_dir: str | Path,
    gene_list_txt: str | Path,
    merge_json: str | Path = BRCA_MERGE_JSON,
) -> dict[str, Any]:
    merged11_cell_types = [name for name, _sources in load_brca_merge_groups(merge_json)]
    synthetic_gene_order = _read_10x_genes(_require_file(Path(synthetic_h5)))
    synthetic_rank = {gene: index for index, gene in enumerate(synthetic_gene_order)}
    root = Path(output_dir)
    csv_path = _require_file(root / "rctd_reference_merged11.csv")
    npy_path = _require_file(root / "rctd_reference_merged11.npy")
    summary_path = _require_file(root / "rctd_reference_merged11_summary.json")
    r_summary_path = _require_file(root / "rctd_reference_merged11_r_summary.json")
    cell_type_order_path = _require_file(root / "cell_type_order.txt")
    genes_path = _require_file(Path(gene_list_txt))

    frame = pd.read_csv(csv_path, index_col=0)
    gene_order = [line.strip() for line in genes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cell_type_order = [line.strip() for line in cell_type_order_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if cell_type_order != merged11_cell_types:
        raise ValueError(f"Unexpected cell type order: {cell_type_order}")
    if list(frame.columns) != merged11_cell_types:
        raise ValueError(f"Unexpected reference columns: {list(frame.columns)}")
    if list(frame.index) != gene_order:
        raise ValueError("gene.txt does not match rctd_reference_merged11.csv index.")
    missing_from_synthetic = [gene for gene in gene_order if gene not in synthetic_rank]
    if missing_from_synthetic:
        raise ValueError(f"Reference genes missing from synthetic H5; first={missing_from_synthetic[0]}")
    ranks = [synthetic_rank[gene] for gene in gene_order]
    if ranks != sorted(ranks):
        raise ValueError("Reference genes do not preserve synthetic H5 gene order.")

    values = frame.to_numpy(dtype=np.float64)
    gene_mass = values.sum(axis=1)
    if np.any(gene_mass <= 0):
        bad = [str(frame.index[int(index)]) for index in np.flatnonzero(gene_mass <= 0)[:10]]
        raise ValueError(f"Reference contains all-zero genes: {bad}")
    column_sums = values.sum(axis=0)
    if not np.allclose(column_sums, 1.0, atol=1e-5):
        raise ValueError(f"Reference CSV columns do not sum to 1: {column_sums}")

    mu_ref = np.load(npy_path)
    if mu_ref.shape != (len(merged11_cell_types), len(gene_order)):
        raise ValueError(f"Unexpected mu_ref shape: {mu_ref.shape}")
    if not np.allclose(mu_ref, values.T.astype(np.float32), atol=1e-7):
        raise ValueError("Numpy reference does not match CSV transpose.")
    row_sums = mu_ref.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError(f"Numpy reference rows do not sum to 1: {row_sums}")

    r_summary = json.loads(r_summary_path.read_text(encoding="utf-8"))
    validation = {
        "output_dir": str(root),
        "csv": str(csv_path),
        "npy": str(npy_path),
        "summary_json": str(summary_path),
        "r_summary_json": str(r_summary_path),
        "gene_list_txt": str(genes_path),
        "cell_type_order_txt": str(cell_type_order_path),
        "shape": [int(mu_ref.shape[0]), int(mu_ref.shape[1])],
        "cell_type_order": cell_type_order,
        "n_synthetic_h5_genes": int(len(synthetic_gene_order)),
        "n_reference_genes": int(len(gene_order)),
        "dropped_zero_genes": int(len(synthetic_gene_order) - len(gene_order)),
        "csv_column_sum_min": float(column_sums.min()),
        "csv_column_sum_max": float(column_sums.max()),
        "npy_row_sum_min": float(row_sums.min()),
        "npy_row_sum_max": float(row_sums.max()),
        "gene_order_matches_csv": True,
        "gene_order_preserves_synthetic_h5_order": True,
        "r_summary": r_summary,
    }
    return validation


def build_reference(
    *,
    sc_ref_h5ad: str | Path,
    synthetic_h5: str | Path,
    output_dir: str | Path,
    gene_list_txt: str | Path,
    positions_csv: str | Path | None = None,
    annotation_column: str = "Level1",
    use_consensus_keep: bool = True,
    umi_min: int = 100,
    r_env_prefix: str | Path | None = None,
    merge_json: str | Path = BRCA_MERGE_JSON,
    force: bool = False,
) -> dict[str, Any]:
    selected_merge_json = _require_file(Path(merge_json))
    merged11_cell_types = [name for name, _sources in load_brca_merge_groups(selected_merge_json)]
    synthetic_path = _require_file(Path(synthetic_h5))
    selected_positions = _default_positions_csv(synthetic_path) if positions_csv is None else _require_file(Path(positions_csv))
    out = Path(output_dir)
    gene_list_path = Path(gene_list_txt)
    csv_path = out / "rctd_reference_merged11.csv"
    npy_path = out / "rctd_reference_merged11.npy"
    summary_path = out / "rctd_reference_merged11_summary.json"
    r_summary_path = out / "rctd_reference_merged11_r_summary.json"
    cell_type_order_path = out / "cell_type_order.txt"
    log_path = out / "rctd_reference_merged11.log"

    required_outputs = [csv_path, npy_path, summary_path, r_summary_path, cell_type_order_path, gene_list_path]
    if all(path.is_file() and path.stat().st_size > 0 for path in required_outputs) and not force:
        validation = validate_rctd_reference(
            synthetic_h5=synthetic_path,
            output_dir=out,
            gene_list_txt=gene_list_path,
            merge_json=selected_merge_json,
        )
        validation["reused_existing"] = True
        return validation

    out.mkdir(parents=True, exist_ok=True)
    gene_list_path.parent.mkdir(parents=True, exist_ok=True)
    if force:
        for path in [*required_outputs, r_summary_path, log_path]:
            if path.exists():
                path.unlink()

    temp_path = Path(tempfile.mkdtemp(prefix=".rctd_ref_tmp_", dir=str(out.parent)))
    try:
        method_input_dir = temp_path / "reference_input"
        export_summary = export_scrna_reference(
            Path(sc_ref_h5ad),
            method_input_dir,
            annotation_column,
            use_consensus_keep,
            allowed_cell_types=BRCA_CELL_TYPE_ORDER,
        )
        reference_rds = temp_path / "rctd_reference_cache.rds"
        env = os.environ.copy()
        _run_logged(
            r_command(
                SCRIPT_DIR.parent / "deconv" / "make_reference_seurat.R",
                ["--method-input-dir", str(method_input_dir), "--output-rds", str(reference_rds)],
                r_env_prefix=r_env_prefix,
            ),
            env=env,
            cwd=SCRIPT_DIR.parents[1],
            log_path=log_path,
        )
        _run_logged(
            r_command(
                SCRIPT_DIR / "build_rctd_reference.R",
                [
                    "--synthetic-h5",
                    str(synthetic_path),
                    "--positions-csv",
                    str(selected_positions),
                    "--reference-rds",
                    str(reference_rds),
                    "--out-csv",
                    str(csv_path),
                    "--out-summary",
                    str(r_summary_path),
                    "--gene-list-output",
                    str(gene_list_path),
                    "--cell-type-order-output",
                    str(cell_type_order_path),
                    "--merge-json",
                    str(selected_merge_json),
                    "--annotation-column",
                    "Annotation",
                    "--umi-min",
                    str(int(umi_min)),
                ],
                r_env_prefix=r_env_prefix,
            ),
            env=env,
            cwd=SCRIPT_DIR.parents[1],
            log_path=log_path,
        )

        frame = pd.read_csv(csv_path, index_col=0)
        values = frame.loc[:, merged11_cell_types].to_numpy(dtype=np.float32)
        column_sums = values.sum(axis=0, keepdims=True)
        values = values / np.maximum(column_sums, 1e-12)
        np.save(npy_path, values.T.astype(np.float32))
        if r_summary_path.exists():
            shutil.copy2(r_summary_path, summary_path)
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path)

    validation = validate_rctd_reference(
        synthetic_h5=synthetic_path,
        output_dir=out,
        gene_list_txt=gene_list_path,
        merge_json=selected_merge_json,
    )
    validation.update(
        {
            "reused_existing": False,
            "sc_ref_h5ad": str(Path(sc_ref_h5ad)),
            "synthetic_h5": str(synthetic_path),
            "positions_csv": str(selected_positions),
            "reference_export": export_summary,
            "log": str(log_path),
        }
    )
    summary_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validation
