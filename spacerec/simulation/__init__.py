from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import h5py
import pandas as pd

from .alignment import run_alignment
from .gen_spots import generate_synthetic_visium


SYN_VIS_OUTPUTS = [
    "synthetic_tissue_positions_transcript_filtered.csv",
    "synthetic_scalefactors_json.json",
    "synthetic_st_filtered_feature_bc_matrix_transcript.h5",
    "filtered_synthetic_st_filtered_feature_bc_matrix_transcript.h5",
    "visium_like_spots_on_he_thumbnail.png",
    "visium_like_spots_on_he_thumbnail_transcript.png",
]


def _existing_outputs(output_dir: Path) -> list[Path]:
    return [output_dir / name for name in SYN_VIS_OUTPUTS if (output_dir / name).exists()]


def validate_syn_vis_outputs(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    filtered_position_columns = [
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
        "transcript_count",
    ]
    filtered = pd.read_csv(root / "synthetic_tissue_positions_transcript_filtered.csv", nrows=1)
    if list(filtered.columns) != filtered_position_columns:
        raise ValueError(f"{root / 'synthetic_tissue_positions_transcript_filtered.csv'} has unexpected columns: {list(filtered.columns)}")

    scalefactors = json.loads((root / "synthetic_scalefactors_json.json").read_text(encoding="utf-8"))
    for key in ["spot_diameter_fullres", "tissue_hires_scalef"]:
        if key not in scalefactors:
            raise ValueError(f"synthetic_scalefactors_json.json is missing {key!r}.")

    h5_summary: dict[str, Any] = {}
    for name in ["synthetic_st_filtered_feature_bc_matrix_transcript.h5", "filtered_synthetic_st_filtered_feature_bc_matrix_transcript.h5"]:
        path = root / name
        with h5py.File(path, "r") as handle:
            for key in ["data", "indices", "indptr", "shape", "barcodes", "features/name", "features/id"]:
                if f"matrix/{key}" not in handle:
                    raise ValueError(f"{path} is missing matrix/{key}.")
            shape = [int(value) for value in handle["matrix/shape"][()]]
            h5_summary[name] = {
                "shape": shape,
                "n_barcodes": int(len(handle["matrix/barcodes"])),
                "n_features": int(len(handle["matrix/features/name"])),
                "nnz": int(len(handle["matrix/data"])),
            }
    return {"output_dir": str(root), "scalefactors": scalefactors, "h5": h5_summary}


def run_simulation(
    *,
    xen_dir: str | Path,
    output_syn_vis_dir: str | Path,
    aligned_dir: str | Path | None = None,
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
    write_aligned_transcripts: bool = True,
    max_transcripts: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    xen_root = Path(xen_dir)
    syn_root = Path(output_syn_vis_dir)
    selected_aligned_dir = Path(aligned_dir) if aligned_dir is not None else xen_root / "aligned"

    alignment_summary = run_alignment(
        xen_dir=xen_root,
        aligned_dir=selected_aligned_dir,
        alignment_csv=alignment_csv,
        xenium_pixel_size=xenium_pixel_size,
        matrix_direction=matrix_direction,
        chunksize=chunksize,
        write_aligned_transcripts=write_aligned_transcripts,
        max_transcripts=max_transcripts,
    )

    existing = _existing_outputs(syn_root)
    if existing and not force:
        return {
            "stage": "simulation",
            "reused_existing_syn_vis": True,
            "reason": "output files already exist and force=False",
            "xen_dir": str(xen_root),
            "aligned_dir": str(selected_aligned_dir),
            "output_syn_vis_dir": str(syn_root),
            "existing_outputs": [str(path) for path in existing],
            "alignment": alignment_summary,
        }
    if sc_ref_h5ad is None or not Path(sc_ref_h5ad).exists():
        raise FileNotFoundError(
            "sc_ref_h5ad is required to create the complete syn_vis Step 0 outputs, "
            "including filtered_synthetic_st_filtered_feature_bc_matrix_transcript.h5."
        )

    syn_root.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(tempfile.mkdtemp(prefix=".simu_tmp_", dir=str(syn_root.parent)))
    try:
        generation_summary = generate_synthetic_visium(
            xen_dir=xen_root,
            aligned_dir=selected_aligned_dir,
            output_dir=temp_path,
            sc_ref_h5ad=sc_ref_h5ad,
            he_image=he_image,
            alignment_csv=alignment_csv,
            xenium_pixel_size=xenium_pixel_size,
            he_pixel_size=he_pixel_size,
            spot_diameter_um=spot_diameter_um,
            spot_spacing_um=spot_spacing_um,
            margin_um=margin_um,
            chunksize=chunksize,
            matrix_direction=matrix_direction,
            thumbnail_max_dim=thumbnail_max_dim,
            max_transcripts=max_transcripts,
        )
        validation = validate_syn_vis_outputs(temp_path)
        syn_root.mkdir(parents=True, exist_ok=True)
        for path in temp_path.iterdir():
            target = syn_root / path.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(path), target)
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path)

    final_validation = validate_syn_vis_outputs(syn_root)
    summary = {
        "stage": "simulation",
        "reused_existing_syn_vis": False,
        "xen_dir": str(xen_root),
        "aligned_dir": str(selected_aligned_dir),
        "output_syn_vis_dir": str(syn_root),
        "alignment": alignment_summary,
        "generation": generation_summary,
        "validation": final_validation,
    }
    (syn_root / "simulation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


__all__ = ["run_alignment", "run_simulation", "validate_syn_vis_outputs"]
