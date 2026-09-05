from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _dataset(dataset: str) -> str:
    selected = str(dataset).lower()
    if selected not in {"brca", "crc"}:
        raise ValueError(f"Unsupported dataset: {dataset!r}.")
    return selected


def _default_step_dir(dataset: str, step: str) -> Path:
    return Path.cwd() / "results" / _dataset(dataset) / step


def _ensure_dir(path: str | Path) -> Path:
    selected = Path(path)
    selected.mkdir(parents=True, exist_ok=True)
    return selected


def deconv(
    *,
    dataset: str,
    visium_dir: str | Path,
    sc_ref_h5ad: str | Path,
    output_dir: str | Path | None = None,
    sample_id: str,
    annotation_column: str = "Level1",
    max_cores: int = 8,
    output_prefix: str | None = None,
    umi_min: int | None = None,
    r_env_prefix: str | Path | None = None,
    merge: bool = True,
    merge_json: str | Path | None = None,
    force: bool = False,
    run_rctd: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run RCTD deconvolution from h5ad reference and synthetic Visium inputs."""

    from .deconv import run_deconvolution

    selected = _dataset(dataset)
    selected_output = _ensure_dir(output_dir or _default_step_dir(selected, "deconv"))
    prefix = str(output_prefix or kwargs.pop("output_prefix", f"RCTD_{sample_id}"))

    summary = run_deconvolution(
        visium_dir=Path(visium_dir),
        sc_ref_h5ad=Path(sc_ref_h5ad),
        output_dir=selected_output,
        sample_id=sample_id,
        annotation_column=annotation_column,
        max_cores=max_cores,
        umi_min=umi_min,
        r_env_prefix=None if r_env_prefix is None else Path(r_env_prefix),
        merge=bool(merge),
        merge_json=merge_json or Path(__file__).resolve().parent / "deconv" / "brca_type_merge_17to11.json",
        force=force,
        run_rctd=run_rctd,
        output_prefix=prefix,
        **kwargs,
    )
    summary.update(
        {
            "dataset": selected,
            "output_dir": str(selected_output),
            "deconv_csv": str(selected_output / f"{prefix}_proportions_merged11_spacerec.csv"),
            "raw_deconv_csv": str(selected_output / f"{prefix}_proportions_spacerec.csv"),
        }
    )
    (selected_output / "deconv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def rctd_ref(
    *,
    dataset: str,
    sc_ref_h5ad: str | Path,
    synthetic_h5: str | Path,
    output_dir: str | Path,
    gene_list_txt: str | Path,
    positions_csv: str | Path | None = None,
    annotation_column: str = "Level1",
    umi_min: int = 100,
    r_env_prefix: str | Path | None = None,
    merge_json: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build the merged11 RCTD reference aligned to synthetic transcript Visium genes."""

    selected = _dataset(dataset)
    if selected != "brca":
        raise ValueError("rctd_ref is currently implemented for dataset='brca' only.")
    from .rctd_ref import build_reference

    summary = build_reference(
        sc_ref_h5ad=Path(sc_ref_h5ad),
        synthetic_h5=Path(synthetic_h5),
        output_dir=Path(output_dir),
        gene_list_txt=Path(gene_list_txt),
        positions_csv=None if positions_csv is None else Path(positions_csv),
        annotation_column=annotation_column,
        umi_min=int(umi_min),
        r_env_prefix=None if r_env_prefix is None else Path(r_env_prefix),
        merge_json=merge_json or Path(__file__).resolve().parent / "deconv" / "brca_type_merge_17to11.json",
        force=bool(force),
    )
    summary["dataset"] = selected
    return summary


def simu(
    *,
    dataset: str,
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
    """Run Step 0 simulation from Xenium outputs to aligned files and synthetic Visium files."""

    selected = _dataset(dataset)
    from .simulation import run_simulation

    summary = run_simulation(
        xen_dir=Path(xen_dir),
        output_syn_vis_dir=Path(output_syn_vis_dir),
        aligned_dir=None if aligned_dir is None else Path(aligned_dir),
        sc_ref_h5ad=None if sc_ref_h5ad is None else Path(sc_ref_h5ad),
        he_image=None if he_image is None else Path(he_image),
        alignment_csv=None if alignment_csv is None else Path(alignment_csv),
        xenium_pixel_size=float(xenium_pixel_size),
        he_pixel_size=float(he_pixel_size),
        spot_diameter_um=float(spot_diameter_um),
        spot_spacing_um=float(spot_spacing_um),
        margin_um=float(margin_um),
        chunksize=int(chunksize),
        matrix_direction=str(matrix_direction),
        thumbnail_max_dim=int(thumbnail_max_dim),
        write_aligned_transcripts=bool(write_aligned_transcripts),
        max_transcripts=None if max_transcripts is None else int(max_transcripts),
        force=bool(force),
    )
    summary["dataset"] = selected
    return summary


def anno(
    *,
    dataset: str,
    xen_dir: str | Path,
    aligned_dir: str | Path,
    sc_ref_h5ad: str | Path,
    output_dir: str | Path,
    r_env_prefix: str | Path | None = None,
    merge_json: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run BRCA 10xG-style Xenium cell annotation into a package-local output directory."""

    selected = _dataset(dataset)
    if selected != "brca":
        raise ValueError("Annotation is currently implemented for dataset='brca' only.")
    from .annotation import run_annotation

    summary = run_annotation(
        xen_dir=Path(xen_dir),
        aligned_dir=Path(aligned_dir),
        sc_ref_h5ad=Path(sc_ref_h5ad),
        output_dir=Path(output_dir),
        r_env_prefix=None if r_env_prefix is None else Path(r_env_prefix),
        merge_json=merge_json or Path(__file__).resolve().parent / "deconv" / "brca_type_merge_17to11.json",
        force=bool(force),
    )
    summary["dataset"] = selected
    return summary


def ge(
    *,
    dataset: str,
    he_image: str | Path,
    positions_csv: str | Path,
    scalefactors_json: str | Path,
    thumbnail_png: str | Path,
    output_dir: str | Path | None = None,
    output_h5: str | Path | None = None,
    summary_json: str | Path | None = None,
    patch_metadata_h5: str | Path | None = None,
    progress_json: str | Path | None = None,
    mask_preview_png: str | Path | None = None,
    patch_size: int = 288,
    stride: int = 72,
    concat_local: bool = True,
    concat_nbr: bool = False,
    model: str = "virchow2",
    batch_size: int = 4,
    num_workers: int = 4,
    device: str | None = None,
    max_patches: int | None = None,
    metadata_only: bool = False,
    force: bool = False,
    auto_select_gpu: bool = True,
    enable_progress_bar: bool = True,
    progress_refresh_rate: float = 1.0,
    stage_log_stdout: bool = False,
    sample_id: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run dense18 Virchow2 grid embedding through the existing implementation."""

    selected = _dataset(dataset)
    if model != "virchow2":
        raise ValueError("The slim API currently supports model='virchow2' only.")
    if not concat_local:
        raise ValueError("The fixed dense18_virchow2 flow requires concat_local=True.")
    if int(patch_size) % 16 != 0:
        raise ValueError("patch_size must be divisible by 16 for Virchow2 16x16 tokens.")
    neighbor_feature_mode = str(kwargs.pop("neighbor_feature_mode", "mean" if concat_nbr else "none"))
    if not concat_nbr and neighbor_feature_mode != "none":
        raise ValueError("concat_nbr=False requires neighbor_feature_mode='none'.")
    if concat_nbr and neighbor_feature_mode == "none":
        raise ValueError("concat_nbr=True requires a neighbor_feature_mode other than 'none'.")

    selected_output = _ensure_dir(output_dir or _default_step_dir(selected, "grid_embedding"))
    selected_h5 = Path(output_h5) if output_h5 is not None else selected_output / "grid_embedding.h5"
    selected_summary_json = Path(summary_json) if summary_json is not None else selected_output / "grid_embedding_summary.json"
    selected_patch_metadata_h5 = (
        Path(patch_metadata_h5) if patch_metadata_h5 is not None else selected_output / "patch_metadata_h5.h5"
    )
    selected_progress_json = Path(progress_json) if progress_json is not None else selected_output / "grid_embedding_progress.json"
    mask_preview = Path(mask_preview_png) if mask_preview_png is not None else selected_output / "grid_embedding_mask_preview.png"
    grid_size = int(patch_size) // 16
    gpu_selection = _auto_select_cuda_device() if auto_select_gpu and device is None else {
        "enabled": bool(auto_select_gpu),
        "selected": False,
        "reason": "device was provided" if device is not None else "auto_select_gpu=False",
        "device": None if device is None else str(device),
    }
    cuda_device_arg = gpu_selection.get("cuda_device_arg")
    selected_device = f"cuda:{int(cuda_device_arg)}" if device is None and cuda_device_arg is not None else device

    from .gridembedding import run_gridembedding

    run_summary = run_gridembedding(
        dataset=selected,
        sample_id=sample_id,
        force=force,
        metadata_only=metadata_only,
        max_patches=max_patches,
        he_image=Path(he_image),
        positions_csv=Path(positions_csv),
        scalefactors_json=Path(scalefactors_json),
        thumbnail_png=Path(thumbnail_png),
        output_h5=selected_h5,
        summary_json=selected_summary_json,
        patch_metadata_h5=selected_patch_metadata_h5,
        progress_json=selected_progress_json,
        patch_size=int(patch_size),
        grid_size=grid_size,
        stride=int(stride),
        neighbor_feature_mode=neighbor_feature_mode,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        device=selected_device,
        enable_progress_bar=bool(enable_progress_bar),
        progress_refresh_rate=float(progress_refresh_rate),
        stage_log_stdout=bool(stage_log_stdout),
        **kwargs,
    )

    summary: dict[str, Any] = {}
    if selected_summary_json.exists():
        summary = json.loads(selected_summary_json.read_text(encoding="utf-8"))
    diagnostic = summary.get("patch_summary", {}).get("diagnostic_plot") if summary else None
    if isinstance(diagnostic, dict) and diagnostic.get("output_png"):
        diagnostic_png = Path(str(diagnostic["output_png"]))
        if diagnostic_png.exists() and diagnostic_png.resolve() != mask_preview.resolve():
            shutil.copy2(diagnostic_png, mask_preview)
    summary.update(
        {
            "dataset": selected,
            "stage": "dense18_virchow2",
            "output_dir": str(selected_output),
            "output_h5": str(selected_h5),
            "summary_json": str(selected_summary_json),
            "patch_metadata_h5": str(selected_patch_metadata_h5),
            "mask_preview_png": str(mask_preview),
            "concat_local": bool(concat_local),
            "concat_nbr": bool(concat_nbr),
            "neighbor_feature_mode": neighbor_feature_mode,
            "mask_step": "not applied by spacerec.ge; use spacerec.mask to create he_mask_grid.h5 and grid_embedding_train_filtered.h5",
            "gpu_selection": gpu_selection,
            "api_summary": run_summary,
        }
    )
    selected_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def mask(
    *,
    dataset: str,
    grid_embedding_h5: str | Path,
    output_dir: str | Path | None = None,
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
    """Create an H&E grid tissue mask and a training-filtered grid embedding H5.

    The default mode is H&E-only. Embedding-assisted correction runs only when
    mode='embedding' is passed explicitly.
    """

    from .mask import run_mask

    selected = _dataset(dataset)
    selected_output = _ensure_dir(output_dir or _default_step_dir(selected, "mask"))
    summary = run_mask(
        grid_embedding_h5=Path(grid_embedding_h5),
        output_dir=selected_output,
        output_h5=None if output_h5 is None else Path(output_h5),
        training_h5=None if training_h5 is None else Path(training_h5),
        he_image=None if he_image is None else Path(he_image),
        mode=str(mode),
        feature_key=feature_key,
        sample_size=int(sample_size),
        pca_components=int(pca_components),
        k=int(k),
        chunk_size=int(chunk_size),
        confident_blank=float(confident_blank),
        confident_tissue=float(confident_tissue),
        tissue_fraction_threshold=float(tissue_fraction_threshold),
        cluster_tissue_mean=float(cluster_tissue_mean),
        cluster_tissue_ratio=float(cluster_tissue_ratio),
        min_component_grids=int(min_component_grids),
        max_hole_grids=int(max_hole_grids),
        thumbnail_max_side=int(thumbnail_max_side),
        he_mask_max_side=int(he_mask_max_side),
        he_saturation_threshold=float(he_saturation_threshold),
        he_value_threshold=float(he_value_threshold),
        he_white_value=float(he_white_value),
        he_white_saturation=float(he_white_saturation),
        min_he_object_pixels=int(min_he_object_pixels),
        seed=int(seed),
        force=bool(force),
    )
    summary["dataset"] = selected
    parent = sys.modules.get(__package__)
    if parent is not None:
        setattr(parent, "mask", mask)
    return summary


def _decode_strings(values) -> list[str]:
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in values]


def export_grid_predictions(
    grid_predictions_h5: str | Path,
    *,
    grid_type_csv: str | Path,
    grid_expr_h5ad: str | Path,
) -> dict[str, str]:
    """Convert existing grid_predictions.h5 into slim grid type/expression files."""

    import anndata as ad
    import h5py
    import numpy as np
    import pandas as pd

    source = Path(grid_predictions_h5)
    type_out = Path(grid_type_csv)
    expr_out = Path(grid_expr_h5ad)
    with h5py.File(source, "r") as handle:
        center_xy = np.asarray(handle["center_xy"], dtype=np.float32)
        bbox = np.asarray(handle["bbox_xyxy"], dtype=np.float32)
        gene_names = _decode_strings(handle["gene_name"][:])
        cell_type_names = _decode_strings(handle["cell_type_name"][:])
        expr = np.asarray(handle["expr_pred"], dtype=np.float32)
        type_prob = np.asarray(handle["type_prob"], dtype=np.float32)

    grid_id = [f"grid_{index}" for index in range(center_xy.shape[0])]
    type_table = pd.DataFrame(
        {
            "grid_id": grid_id,
            "center_x": center_xy[:, 0],
            "center_y": center_xy[:, 1],
            "bbox_x0": bbox[:, 0],
            "bbox_y0": bbox[:, 1],
            "bbox_x1": bbox[:, 2],
            "bbox_y1": bbox[:, 3],
            "predicted_type": [cell_type_names[int(index)] for index in type_prob.argmax(axis=1)],
        }
    )
    for index, name in enumerate(cell_type_names):
        type_table[f"prob_{name}"] = type_prob[:, index]
    type_out.parent.mkdir(parents=True, exist_ok=True)
    type_table.to_csv(type_out, index=False)

    obs = pd.DataFrame(
        {
            "grid_id": grid_id,
            "center_x": center_xy[:, 0],
            "center_y": center_xy[:, 1],
            "bbox_x0": bbox[:, 0],
            "bbox_y0": bbox[:, 1],
            "bbox_x1": bbox[:, 2],
            "bbox_y1": bbox[:, 3],
        },
        index=grid_id,
    )
    adata = ad.AnnData(X=expr, obs=obs, var=pd.DataFrame(index=pd.Index(gene_names, name="gene")))
    expr_out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(expr_out)
    return {"grid_type_csv": str(type_out), "grid_expr_h5ad": str(expr_out)}


def _feature_key_from_grid_embedding(path: Path) -> str:
    import h5py

    with h5py.File(path, "r") as handle:
        if "feature_key" in handle.attrs:
            return str(handle.attrs["feature_key"])
        candidates = [
            key
            for key, value in handle.items()
            if getattr(value, "ndim", None) == 2
            and key not in {"bbox_xyxy", "center_xy", "grid_bbox", "grid_center_xy", "spot_xy"}
        ]
    if len(candidates) == 1:
        return candidates[0]
    raise KeyError(f"Cannot infer feature dataset key from {path}; candidates={candidates}")


def _visible_cuda_device_ids() -> list[int] | None:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None or value.strip() == "":
        return None
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if not parts or not all(part.isdigit() for part in parts):
        return []
    return [int(part) for part in parts]


def _auto_select_cuda_device() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"enabled": True, "selected": False, "reason": f"nvidia-smi failed: {exc}"}

    visible_physical = _visible_cuda_device_ids()
    rows: list[dict[str, int]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            row = {
                "index": int(parts[0]),
                "memory_free_mb": int(parts[1]),
                "memory_used_mb": int(parts[2]),
                "utilization_gpu_percent": int(parts[3]),
            }
        except ValueError:
            continue
        if visible_physical is None or row["index"] in visible_physical:
            rows.append(row)
    if not rows:
        return {
            "enabled": True,
            "selected": False,
            "reason": "no numeric visible CUDA devices found",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

    selected = max(rows, key=lambda item: (item["memory_free_mb"], -item["utilization_gpu_percent"], -item["memory_used_mb"]))
    if visible_physical is None:
        visible_index = selected["index"]
    else:
        visible_index = visible_physical.index(selected["index"])

    result: dict[str, Any] = {
        "enabled": True,
        "selected": True,
        "physical_index": int(selected["index"]),
        "visible_index": int(visible_index),
        "memory_free_mb": int(selected["memory_free_mb"]),
        "memory_used_mb": int(selected["memory_used_mb"]),
        "utilization_gpu_percent": int(selected["utilization_gpu_percent"]),
        "previous_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        if torch_module.cuda.is_available():
            torch_module.cuda.set_device(int(visible_index))
            result["method"] = "torch.cuda.set_device"
            result["cuda_device_arg"] = int(visible_index)
        else:
            result.update({"selected": False, "reason": "torch is imported but CUDA is unavailable"})
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected["index"])
        result["method"] = "CUDA_VISIBLE_DEVICES"
        result["cuda_visible_devices"] = os.environ["CUDA_VISIBLE_DEVICES"]
        result["cuda_device_arg"] = None
    return result


def _run_finetune_mu_api(
    *,
    dataset: str,
    st_h5ad: str | Path,
    deconv_csv: str | Path,
    grid_embedding_h5: str | Path,
    gene_list: str | Path,
    mu_ref: str | Path | None,
    run_dir: str | Path | None,
    projection_dim: int,
    lambda_type: float,
    alpha: float,
    stage1_epochs: int,
    stage2_epochs: int,
    stage1_checkpoint: str | Path | None,
    batch_size: int,
    lr: float,
    expr_loss: str,
    delta_alpha: float,
    init_scale: float,
    skip_input_projector: bool,
    type_head_hidden_layers: int,
    factorized_head_hidden_layers: int,
    scale_head_hidden_dim: int | None,
    delta_head_hidden_dim: int | None,
    mu_eps: float,
    no_export: bool,
    auto_select_gpu: bool,
    enable_progress_bar: bool,
    progress_refresh_rate: int,
    stage_log_stdout: bool,
    stop_after_stage1: bool,
    stage1_export_type: bool,
    stage1_agg: bool,
    stage1_polygon_csv: str | Path | None,
    stage1_grid_type_csv: str | Path | None,
    stage1_cell_type_csv: str | Path | None,
    stage2_polygon_csv: str | Path | None,
    grid_pred_expression_h5: str | Path | None,
    cell_pred_expression_h5: str | Path | None,
    summary_filename: str,
    **kwargs: Any,
) -> dict[str, Any]:
    gpu_selection = _auto_select_cuda_device() if auto_select_gpu else {"enabled": False, "selected": False}
    cuda_device_arg = gpu_selection.get("cuda_device_arg")
    from .model import run_finetune_mu_training

    selected = _dataset(dataset)
    selected_run_dir = _ensure_dir(run_dir or _default_step_dir(selected, "train_v2"))
    selected_mu_ref = (
        Path(mu_ref)
        if mu_ref is not None
        else Path.cwd() / "resources" / selected / "deconv" / "rctd_ref" / "rctd_reference_merged11.npy"
    )
    summary = run_finetune_mu_training(
        dataset=selected,
        dense_h5=Path(grid_embedding_h5),
        dense_feature_key=_feature_key_from_grid_embedding(Path(grid_embedding_h5)),
        st_h5ad=Path(st_h5ad),
        deconv_csv=Path(deconv_csv),
        gene_list=Path(gene_list),
        mu_ref=selected_mu_ref,
        run_dir=selected_run_dir,
        projection_dim=int(projection_dim),
        stage1_epochs=int(stage1_epochs),
        stage2_epochs=int(stage2_epochs),
        stage1_checkpoint=None if stage1_checkpoint is None else Path(stage1_checkpoint),
        lambda_conf=float(alpha),
        batch_size=int(batch_size),
        lr=float(lr),
        expr_loss=str(expr_loss),
        delta_alpha=float(delta_alpha),
        init_scale=float(init_scale),
        skip_input_projector=bool(skip_input_projector),
        type_head_hidden_layers=int(type_head_hidden_layers),
        factorized_head_hidden_layers=int(factorized_head_hidden_layers),
        scale_head_hidden_dim=None if scale_head_hidden_dim is None else int(scale_head_hidden_dim),
        delta_head_hidden_dim=None if delta_head_hidden_dim is None else int(delta_head_hidden_dim),
        mu_eps=float(mu_eps),
        no_export=bool(no_export),
        cuda_device=None if cuda_device_arg is None else int(cuda_device_arg),
        gpu_selection=gpu_selection,
        enable_progress_bar=bool(enable_progress_bar),
        progress_refresh_rate=int(progress_refresh_rate),
        stage_log_stdout=bool(stage_log_stdout),
        stop_after_stage1=bool(stop_after_stage1),
        stage1_export_type=bool(stage1_export_type),
        stage1_agg=bool(stage1_agg),
        stage1_polygon_csv=None if stage1_polygon_csv is None else Path(stage1_polygon_csv),
        stage1_grid_type_csv=None if stage1_grid_type_csv is None else Path(stage1_grid_type_csv),
        stage1_cell_type_csv=None if stage1_cell_type_csv is None else Path(stage1_cell_type_csv),
        stage2_polygon_csv=None if stage2_polygon_csv is None else Path(stage2_polygon_csv),
        grid_pred_expression_h5=None if grid_pred_expression_h5 is None else Path(grid_pred_expression_h5),
        cell_pred_expression_h5=None if cell_pred_expression_h5 is None else Path(cell_pred_expression_h5),
        **kwargs,
    )
    grid_h5 = selected_run_dir / "grid_predictions.h5"
    if not no_export and not stop_after_stage1 and grid_h5.exists():
        summary.update(
            export_grid_predictions(
                grid_h5,
                grid_type_csv=selected_run_dir / "grid_type.csv",
                grid_expr_h5ad=selected_run_dir / "grid_expr.h5ad",
            )
        )
    summary.update(
        {
            "dataset": selected,
            "run_dir": str(selected_run_dir),
            "grid_predictions_h5": str(grid_h5),
            "mu_ref": str(selected_mu_ref),
            "lambda_type_argument_ignored": float(lambda_type),
            "gpu_selection": gpu_selection,
        }
    )
    (selected_run_dir / summary_filename).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def train(
    *,
    dataset: str,
    st_h5ad: str | Path,
    deconv_csv: str | Path,
    grid_embedding_h5: str | Path,
    gene_list: str | Path,
    mu_ref: str | Path | None = None,
    run_dir: str | Path | None = None,
    projection_dim: int = 2048,
    lambda_type: float = 1.0,
    alpha: float = 0.1,
    max_epochs: int | None = None,
    stage1_epochs: int = 70,
    stage2_epochs: int = 70,
    stage1_checkpoint: str | Path | None = None,
    batch_size: int = 4,
    lr: float = 5e-5,
    expr_loss: str = "log1p_huber",
    delta_alpha: float = 0.5,
    init_scale: float = 128.0,
    skip_input_projector: bool = True,
    type_head_hidden_layers: int = 2,
    factorized_head_hidden_layers: int = 3,
    scale_head_hidden_dim: int | None = 512,
    delta_head_hidden_dim: int | None = 2048,
    mu_eps: float = 1e-8,
    no_export: bool = False,
    auto_select_gpu: bool = True,
    enable_progress_bar: bool = True,
    progress_refresh_rate: int = 1,
    stage_log_stdout: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Train the v2 finetune-mu two-stage model and export slim grid outputs."""

    return _run_finetune_mu_api(
        dataset=dataset,
        st_h5ad=Path(st_h5ad),
        deconv_csv=Path(deconv_csv),
        grid_embedding_h5=Path(grid_embedding_h5),
        gene_list=Path(gene_list),
        mu_ref=mu_ref,
        run_dir=run_dir,
        projection_dim=int(projection_dim),
        lambda_type=float(lambda_type),
        alpha=float(alpha),
        stage1_epochs=int(stage1_epochs if max_epochs is None else max_epochs),
        stage2_epochs=int(stage2_epochs if max_epochs is None else max_epochs),
        stage1_checkpoint=None if stage1_checkpoint is None else Path(stage1_checkpoint),
        batch_size=int(batch_size),
        lr=float(lr),
        expr_loss=str(expr_loss),
        delta_alpha=float(delta_alpha),
        init_scale=float(init_scale),
        skip_input_projector=bool(skip_input_projector),
        type_head_hidden_layers=int(type_head_hidden_layers),
        factorized_head_hidden_layers=int(factorized_head_hidden_layers),
        scale_head_hidden_dim=None if scale_head_hidden_dim is None else int(scale_head_hidden_dim),
        delta_head_hidden_dim=None if delta_head_hidden_dim is None else int(delta_head_hidden_dim),
        mu_eps=float(mu_eps),
        no_export=bool(no_export),
        auto_select_gpu=bool(auto_select_gpu),
        enable_progress_bar=bool(enable_progress_bar),
        progress_refresh_rate=int(progress_refresh_rate),
        stage_log_stdout=bool(stage_log_stdout),
        stop_after_stage1=False,
        stage1_export_type=False,
        stage1_agg=False,
        stage1_polygon_csv=None,
        stage1_grid_type_csv=None,
        stage1_cell_type_csv=None,
        stage2_polygon_csv=None,
        grid_pred_expression_h5=None,
        cell_pred_expression_h5=None,
        summary_filename="slim_train_summary.json",
        **kwargs,
    )


def stage1(
    *,
    dataset: str,
    st_h5ad: str | Path,
    deconv_csv: str | Path,
    grid_embedding_h5: str | Path,
    gene_list: str | Path,
    mu_ref: str | Path | None = None,
    run_dir: str | Path | None = None,
    projection_dim: int = 2048,
    lambda_type: float = 1.0,
    alpha: float = 0.1,
    max_epochs: int | None = None,
    stage1_epochs: int = 70,
    batch_size: int = 4,
    lr: float = 5e-5,
    expr_loss: str = "log1p_huber",
    delta_alpha: float = 0.5,
    init_scale: float = 128.0,
    skip_input_projector: bool = True,
    type_head_hidden_layers: int = 2,
    factorized_head_hidden_layers: int = 3,
    scale_head_hidden_dim: int | None = 512,
    delta_head_hidden_dim: int | None = 2048,
    mu_eps: float = 1e-8,
    auto_select_gpu: bool = True,
    enable_progress_bar: bool = True,
    progress_refresh_rate: int = 1,
    stage_log_stdout: bool = False,
    agg: bool = False,
    cell_polygon_csv: str | Path | None = None,
    type_pred_dir: str | Path | None = None,
    grid_type_csv: str | Path | None = None,
    cell_type_csv: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Train only the v2 finetune-mu stage 1 router/type model."""

    selected = _dataset(dataset)
    selected_type_pred_dir = _ensure_dir(type_pred_dir or Path.cwd() / "resources" / selected / "type_pred")
    selected_grid_type_csv = Path(grid_type_csv) if grid_type_csv is not None else selected_type_pred_dir / "stage1_grid_type.csv"
    selected_cell_type_csv = Path(cell_type_csv) if cell_type_csv is not None else selected_type_pred_dir / "stage1_cell_type.csv"
    selected_run_dir = Path(run_dir) if run_dir is not None else selected_type_pred_dir / "train_v2"
    selected_cell_polygon_csv = (
        Path(cell_polygon_csv)
        if cell_polygon_csv is not None
        else Path.cwd() / "resources" / selected / "xen" / "aligned" / "he_alignmented_cell_boundaries.csv"
    )
    return _run_finetune_mu_api(
        dataset=selected,
        st_h5ad=Path(st_h5ad),
        deconv_csv=Path(deconv_csv),
        grid_embedding_h5=Path(grid_embedding_h5),
        gene_list=Path(gene_list),
        mu_ref=mu_ref,
        run_dir=selected_run_dir,
        projection_dim=int(projection_dim),
        lambda_type=float(lambda_type),
        alpha=float(alpha),
        stage1_epochs=int(stage1_epochs if max_epochs is None else max_epochs),
        stage2_epochs=0,
        stage1_checkpoint=None,
        batch_size=int(batch_size),
        lr=float(lr),
        expr_loss=str(expr_loss),
        delta_alpha=float(delta_alpha),
        init_scale=float(init_scale),
        skip_input_projector=bool(skip_input_projector),
        type_head_hidden_layers=int(type_head_hidden_layers),
        factorized_head_hidden_layers=int(factorized_head_hidden_layers),
        scale_head_hidden_dim=None if scale_head_hidden_dim is None else int(scale_head_hidden_dim),
        delta_head_hidden_dim=None if delta_head_hidden_dim is None else int(delta_head_hidden_dim),
        mu_eps=float(mu_eps),
        no_export=True,
        auto_select_gpu=bool(auto_select_gpu),
        enable_progress_bar=bool(enable_progress_bar),
        progress_refresh_rate=int(progress_refresh_rate),
        stage_log_stdout=bool(stage_log_stdout),
        stop_after_stage1=True,
        stage1_export_type=True,
        stage1_agg=bool(agg),
        stage1_polygon_csv=selected_cell_polygon_csv,
        stage1_grid_type_csv=selected_grid_type_csv,
        stage1_cell_type_csv=selected_cell_type_csv,
        stage2_polygon_csv=None,
        grid_pred_expression_h5=None,
        cell_pred_expression_h5=None,
        summary_filename="slim_stage1_summary.json",
        **kwargs,
    )


def stage2(
    *,
    dataset: str,
    st_h5ad: str | Path,
    deconv_csv: str | Path,
    grid_embedding_h5: str | Path,
    stage1_checkpoint: str | Path,
    gene_txt: str | Path | None = None,
    gene_list: str | Path | None = None,
    mu_ref: str | Path | None = None,
    run_dir: str | Path | None = None,
    projection_dim: int = 2048,
    lambda_type: float = 1.0,
    alpha: float = 0.1,
    max_epochs: int | None = None,
    stage2_epochs: int = 70,
    batch_size: int = 4,
    lr: float = 5e-5,
    expr_loss: str = "log1p_huber",
    delta_alpha: float = 0.5,
    init_scale: float = 128.0,
    skip_input_projector: bool = True,
    type_head_hidden_layers: int = 2,
    factorized_head_hidden_layers: int = 3,
    scale_head_hidden_dim: int | None = 512,
    delta_head_hidden_dim: int | None = 2048,
    mu_eps: float = 1e-8,
    no_export: bool = False,
    auto_select_gpu: bool = True,
    enable_progress_bar: bool = True,
    progress_refresh_rate: int = 1,
    stage_log_stdout: bool = False,
    cell_polygon_csv: str | Path | None = None,
    grid_pred_expression_h5: str | Path | None = None,
    cell_pred_expression_h5: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Train v2 finetune-mu stage 2 and export grid/cell expression H5 outputs."""

    selected = _dataset(dataset)
    selected_run_dir = Path(run_dir) if run_dir is not None else Path.cwd() / "resources" / selected / "type_pred" / "train_v2"
    selected_gene_txt = Path(gene_txt) if gene_txt is not None else None
    if selected_gene_txt is None:
        selected_gene_txt = Path(gene_list) if gene_list is not None else None
    if selected_gene_txt is None:
        raise ValueError("stage2 requires gene_txt.")
    selected_cell_polygon_csv = (
        Path(cell_polygon_csv)
        if cell_polygon_csv is not None
        else Path.cwd() / "resources" / selected / "xen" / "aligned" / "he_alignmented_cell_boundaries.csv"
    )
    selected_grid_pred_expression_h5 = (
        Path(grid_pred_expression_h5) if grid_pred_expression_h5 is not None else selected_run_dir / "grid_pred_expression.h5"
    )
    selected_cell_pred_expression_h5 = (
        Path(cell_pred_expression_h5) if cell_pred_expression_h5 is not None else selected_run_dir / "cell_pred_expression.h5"
    )
    return _run_finetune_mu_api(
        dataset=selected,
        st_h5ad=Path(st_h5ad),
        deconv_csv=Path(deconv_csv),
        grid_embedding_h5=Path(grid_embedding_h5),
        gene_list=selected_gene_txt,
        mu_ref=mu_ref,
        run_dir=selected_run_dir,
        projection_dim=int(projection_dim),
        lambda_type=float(lambda_type),
        alpha=float(alpha),
        stage1_epochs=0,
        stage2_epochs=int(stage2_epochs if max_epochs is None else max_epochs),
        stage1_checkpoint=Path(stage1_checkpoint),
        batch_size=int(batch_size),
        lr=float(lr),
        expr_loss=str(expr_loss),
        delta_alpha=float(delta_alpha),
        init_scale=float(init_scale),
        skip_input_projector=bool(skip_input_projector),
        type_head_hidden_layers=int(type_head_hidden_layers),
        factorized_head_hidden_layers=int(factorized_head_hidden_layers),
        scale_head_hidden_dim=None if scale_head_hidden_dim is None else int(scale_head_hidden_dim),
        delta_head_hidden_dim=None if delta_head_hidden_dim is None else int(delta_head_hidden_dim),
        mu_eps=float(mu_eps),
        no_export=bool(no_export),
        auto_select_gpu=bool(auto_select_gpu),
        enable_progress_bar=bool(enable_progress_bar),
        progress_refresh_rate=int(progress_refresh_rate),
        stage_log_stdout=bool(stage_log_stdout),
        stop_after_stage1=False,
        stage1_export_type=False,
        stage1_agg=False,
        stage1_polygon_csv=None,
        stage1_grid_type_csv=None,
        stage1_cell_type_csv=None,
        stage2_polygon_csv=selected_cell_polygon_csv,
        grid_pred_expression_h5=selected_grid_pred_expression_h5,
        cell_pred_expression_h5=selected_cell_pred_expression_h5,
        summary_filename="slim_stage2_summary.json",
        **kwargs,
    )


def plottype(
    *,
    dataset: str,
    grid_type_csv: str | Path,
    window: tuple[float, float, float, float] | list[float],
    true_xen_type_csv: str | Path | None = None,
    type_merge_json: str | Path | None = None,
    window_label: str | None = None,
    output_dir: str | Path | None = None,
    output_name: str = "grid_type.png",
) -> dict[str, Any]:
    """Plot grid-level predicted cell types for a full-resolution window."""

    from .evaluation import plot_true_type, plot_type_grid

    selected = _dataset(dataset)
    selected_output = _ensure_dir(output_dir or _default_step_dir(selected, "Evaluation"))
    grid_plot = plot_type_grid(
        grid_type_csv=Path(grid_type_csv),
        window=window,
        output_dir=selected_output,
        output_name=output_name,
    )
    result: dict[str, Any] = {
        "dataset": selected,
        "output_dir": str(selected_output),
        "grid_type_png": grid_plot["output_png"],
        "output_png": grid_plot["output_png"],
    }
    if true_xen_type_csv is not None:
        true_plot = plot_true_type(
            true_xen_type_csv=Path(true_xen_type_csv),
            window=window,
            output_dir=selected_output,
            type_merge_json=None if type_merge_json is None else Path(type_merge_json),
            window_label=window_label,
        )
        result["true_type_png"] = true_plot["output_png"]
    return result


def plotexpr(
    *,
    dataset: str,
    grid_expr_h5ad: str | Path,
    window: tuple[float, float, float, float] | list[float],
    true_xen_expr_h5: str | Path | None = None,
    true_xen_type_csv: str | Path | None = None,
    polygon_window_label: str | None = None,
    output_dir: str | Path | None = None,
    gene: str = "EPCAM",
    output_name: str = "grid_expr.png",
) -> dict[str, Any]:
    """Plot grid-level predicted expression for one gene in a full-resolution window."""

    from .evaluation import plot_grid_expression, plot_xenium_expression

    selected = _dataset(dataset)
    selected_output = _ensure_dir(output_dir or _default_step_dir(selected, "Evaluation"))
    grid_plot = plot_grid_expression(
        grid_expr_h5ad=Path(grid_expr_h5ad),
        window=window,
        output_dir=selected_output,
        gene=gene,
        output_name=output_name,
    )
    result: dict[str, Any] = {
        "dataset": selected,
        "output_dir": str(selected_output),
        "grid_expr_png": grid_plot["output_png"],
        "output_png": grid_plot["output_png"],
    }
    if true_xen_expr_h5 is not None:
        if true_xen_type_csv is None:
            raise ValueError("true_xen_type_csv is required when true_xen_expr_h5 is provided.")
        xen_plot = plot_xenium_expression(
            true_xen_expr_h5=Path(true_xen_expr_h5),
            true_xen_type_csv=Path(true_xen_type_csv),
            window=window,
            output_dir=selected_output,
            gene=gene,
            polygon_window_label=polygon_window_label,
        )
        result["xen_expr_png"] = xen_plot["output_png"]
    return result


def _polygon_id_column(table) -> str:
    for name in ["cell_id", "cellvit_cell_id", "xen_cell_id"]:
        if name in table.columns:
            return name
    raise ValueError("polygon_csv must contain one of: cell_id, cellvit_cell_id, xen_cell_id.")


def _probability_columns(table) -> list[str]:
    prob_cols = [col for col in table.columns if str(col).startswith("prob_")]
    if not prob_cols:
        reserved = {"grid_id", "center_x", "center_y", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "predicted_type"}
        prob_cols = [col for col in table.columns if col not in reserved]
    if not prob_cols:
        raise ValueError("grid_type_csv contains no probability columns.")
    return prob_cols


def _grid_lookup(bbox) -> dict[tuple[int, int], int]:
    return {(int(row[0]), int(row[1])): int(index) for index, row in enumerate(bbox)}


def _candidate_grid_indices(bounds, lookup: dict[tuple[int, int], int], grid_size: int) -> list[int]:
    minx, miny, maxx, maxy = bounds
    start_x = math.floor(minx / grid_size) * grid_size
    stop_x = math.ceil(maxx / grid_size) * grid_size
    start_y = math.floor(miny / grid_size) * grid_size
    stop_y = math.ceil(maxy / grid_size) * grid_size
    out: list[int] = []
    for y0 in range(int(start_y), int(stop_y) + grid_size, grid_size):
        for x0 in range(int(start_x), int(stop_x) + grid_size, grid_size):
            index = lookup.get((x0, y0))
            if index is not None:
                out.append(index)
    return out


def agg(
    *,
    dataset: str,
    grid_expr_h5ad: str | Path,
    grid_type_csv: str | Path,
    polygon_csv: str | Path,
    cell_metadata_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
    valid_only: bool = True,
    target_name: str = "cellvit",
    grid_predictions_h5: str | Path | None = None,
) -> dict[str, Any]:
    """Area-weight grid expression/type predictions onto cell polygons."""

    import anndata as ad
    import numpy as np
    import pandas as pd
    from shapely.geometry import Polygon, box

    selected = _dataset(dataset)
    selected_output = _ensure_dir(output_dir or _default_step_dir(selected, "aggregate"))

    expr_adata = ad.read_h5ad(grid_expr_h5ad)
    grid_type = pd.read_csv(grid_type_csv)
    polygons = pd.read_csv(polygon_csv)
    id_col = _polygon_id_column(polygons)
    prob_cols = _probability_columns(grid_type)
    cell_type_names = [col[5:] if str(col).startswith("prob_") else str(col) for col in prob_cols]

    bbox = expr_adata.obs[["bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"]].to_numpy(dtype=np.float64)
    grid_size = int(round(float(np.median(bbox[:, 2] - bbox[:, 0]))))
    lookup = _grid_lookup(bbox)
    expr_matrix = expr_adata.X
    if hasattr(expr_matrix, "toarray"):
        expr_matrix = expr_matrix.toarray()
    expr_matrix = np.asarray(expr_matrix, dtype=np.float32)
    type_matrix = grid_type.loc[:, prob_cols].to_numpy(dtype=np.float32)
    grid_area = float(grid_size * grid_size)

    expr_rows: list[np.ndarray] = []
    ct_rows: list[dict[str, object]] = []
    meta_rows: list[dict[str, object]] = []
    kept_ids: list[str] = []
    zero_coverage = 0
    invalid_polygon = 0

    for raw_id, group in polygons.groupby(id_col, sort=False):
        cell_id = str(raw_id)
        coords = group[["vertex_x", "vertex_y"]].to_numpy(dtype=np.float64)
        if coords.shape[0] < 3:
            invalid_polygon += 1
            if valid_only:
                continue
            polygon = None
        else:
            polygon = Polygon(coords)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if polygon.is_empty or polygon.area <= 0:
                invalid_polygon += 1
                if valid_only:
                    continue
                polygon = None
        if polygon is None:
            indices = []
            weights = np.zeros(0, dtype=np.float64)
        else:
            indices = []
            weights_list: list[float] = []
            for index in _candidate_grid_indices(polygon.bounds, lookup, grid_size):
                x0, y0, x1, y1 = bbox[index]
                area = float(polygon.intersection(box(x0, y0, x1, y1)).area)
                if area > 0.0:
                    indices.append(index)
                    weights_list.append(area / grid_area)
            weights = np.asarray(weights_list, dtype=np.float64)
        if len(indices) == 0:
            zero_coverage += 1
            if valid_only:
                continue
            expr = np.full((expr_matrix.shape[1],), np.nan, dtype=np.float32)
            type_prob = np.full((type_matrix.shape[1],), np.nan, dtype=np.float32)
            covered_fraction = 0.0
        else:
            index_array = np.asarray(indices, dtype=np.int64)
            expr = np.log1p(np.maximum((weights[:, None] * expr_matrix[index_array]).sum(axis=0), 0.0)).astype(np.float32)
            type_weighted = (weights[:, None] * type_matrix[index_array]).sum(axis=0)
            total = float(type_weighted.sum())
            type_prob = (type_weighted / total).astype(np.float32) if total > 0 else np.full((type_matrix.shape[1],), np.nan, dtype=np.float32)
            covered_fraction = float(weights.sum() * grid_area / max(float(polygon.area), 1e-8)) if polygon is not None else 0.0
        kept_ids.append(cell_id)
        expr_rows.append(expr)
        pred_idx = int(np.nanargmax(type_prob)) if np.isfinite(type_prob).any() else -1
        row = {"cell_id": cell_id, "predicted_type": cell_type_names[pred_idx] if pred_idx >= 0 else "Unassigned"}
        row.update({f"prob_{name}": float(type_prob[i]) for i, name in enumerate(cell_type_names)})
        ct_rows.append(row)
        centroid = group[["vertex_x", "vertex_y"]].mean(axis=0)
        meta_rows.append({"cell_id": cell_id, "center_x": float(centroid["vertex_x"]), "center_y": float(centroid["vertex_y"]), "covered_fraction": covered_fraction})

    expr_out = selected_output / "spacerec_expr.h5ad"
    ct_out = selected_output / "spacerec_ct.csv"
    polygon_out = selected_output / "spacerec_polygon.csv"
    summary_out = selected_output / "aggregate_summary.json"

    ct = pd.DataFrame(ct_rows)
    ct.to_csv(ct_out, index=False)
    polygon_export = polygons.rename(columns={id_col: "cell_id"}).loc[lambda df: df["cell_id"].astype(str).isin(set(kept_ids))]
    polygon_export.to_csv(polygon_out, index=False)
    expr_obs = pd.DataFrame(meta_rows, index=pd.Index(kept_ids, name="cell_id"))
    expr_result = ad.AnnData(
        X=np.vstack(expr_rows).astype(np.float32) if expr_rows else np.zeros((0, expr_matrix.shape[1]), dtype=np.float32),
        obs=expr_obs,
        var=expr_adata.var.copy(),
    )
    expr_result.write_h5ad(expr_out)

    if cell_metadata_csv is not None:
        shutil.copy2(cell_metadata_csv, selected_output / "input_cell_metadata.csv")
    summary = {
        "dataset": selected,
        "target_name": target_name,
        "output_dir": str(selected_output),
        "grid_expr_h5ad": str(grid_expr_h5ad),
        "grid_type_csv": str(grid_type_csv),
        "grid_predictions_h5": None if grid_predictions_h5 is None else str(grid_predictions_h5),
        "polygon_csv": str(polygon_csv),
        "cell_metadata_csv": None if cell_metadata_csv is None else str(cell_metadata_csv),
        "spacerec_ct_csv": str(ct_out),
        "spacerec_polygon_csv": str(polygon_out),
        "spacerec_expr_h5ad": str(expr_out),
        "n_input_polygons": int(polygons[id_col].nunique()),
        "n_output_cells": int(len(kept_ids)),
        "n_zero_coverage": int(zero_coverage),
        "n_invalid_polygon": int(invalid_polygon),
        "grid_size": int(grid_size),
        "valid_only": bool(valid_only),
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
