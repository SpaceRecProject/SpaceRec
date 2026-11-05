from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _require_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return path


def _workflow_paths(project_root: str | Path | None = None) -> tuple[Path, Path]:
    if project_root is None:
        repository_root = Path(__file__).resolve().parents[1]
        root = (repository_root / "resources").resolve().parent
    else:
        root = Path(project_root).resolve()
    scripts = root / "scripts" / "handoff_cerebellum_spacerec_v2"
    if not scripts.is_dir():
        raise FileNotFoundError(scripts)
    return root, scripts


def _validated_run_root(project_root: Path) -> Path:
    return project_root / "resources" / "handoff_cerebellum" / "spacerec_v2_openphenom_mean_30px"


def _public_sample_root(project_root: Path, sample_id: str) -> Path:
    return project_root / "resources" / "cerebellum" / sample_id


def _publish(source: Path, target: Path) -> Path:
    _require_file(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.resolve() != source.resolve():
            raise FileExistsError(f"Refusing to replace a different published output: {target}")
        return target
    target.symlink_to(os.path.relpath(source.resolve(), target.parent))
    return target


def _environment(project_root: Path, run_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CEREBELLUM_RUN_VARIANT": "mean8",
            "CEREBELLUM_GRID_SIZE": "30",
            "CEREBELLUM_RUN_ROOT": str(run_root.resolve()),
            "PYTHONPATH": os.pathsep.join(
                value
                for value in [str(project_root), env.get("PYTHONPATH", "")]
                if value
            ),
        }
    )
    return env


def _read_json(path: Path) -> dict[str, Any]:
    _require_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _tail(path: Path, lines: int = 80) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def _run_logged(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}; see {log_path}\n{_tail(log_path)}"
        )


def _run_openphenom(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> None:
    from tqdm.auto import tqdm

    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    progress = None
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("stage") == "geometry_ready" and progress is None:
                progress = tqdm(
                    total=int(event["n_patches"]),
                    desc=f"OpenPhenom {event.get('sample', '')}",
                    unit="patch",
                    dynamic_ncols=True,
                )
            elif event.get("stage") == "extract_progress" and progress is not None:
                progress.update(max(0, int(event["processed_patches"]) - progress.n))
                progress.set_postfix(grids=int(event.get("covered_grids", 0)), refresh=False)
    return_code = process.wait()
    if progress is not None:
        if return_code == 0:
            progress.update(max(0, int(progress.total or 0) - progress.n))
        progress.close()
    if return_code:
        raise RuntimeError(f"OpenPhenom failed with exit code {return_code}; see {log_path}\n{_tail(log_path)}")


def _checkpoint_epoch(path: Path) -> int | None:
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        value = payload.get("epoch") if isinstance(payload, dict) else None
        return None if value is None else int(value) + 1
    except (OSError, RuntimeError, EOFError, ValueError):
        return None


def _run_training(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    checkpoint: Path,
    description: str,
    epochs: int = 70,
) -> None:
    from tqdm.auto import tqdm

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log, tqdm(
        total=epochs,
        desc=description,
        unit="epoch",
        dynamic_ncols=True,
    ) as progress:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        last_mtime = -1
        while process.poll() is None:
            if checkpoint.is_file():
                mtime = checkpoint.stat().st_mtime_ns
                if mtime != last_mtime:
                    last_mtime = mtime
                    completed_epochs = _checkpoint_epoch(checkpoint)
                    if completed_epochs is not None:
                        progress.update(max(0, min(epochs, completed_epochs) - progress.n))
            time.sleep(1.0)
        return_code = int(process.returncode or 0)
        if return_code == 0:
            progress.update(max(0, epochs - progress.n))
    if return_code:
        raise RuntimeError(f"Training failed with exit code {return_code}; see {log_path}\n{_tail(log_path)}")


def prepare_reference(
    *,
    project_root: str | Path | None = None,
    sample_id: str,
    deconv_csv: str | Path,
    stage1_st_h5ad: str | Path,
    stage1_gene_list: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    root, scripts = _workflow_paths(project_root)
    run_root = _validated_run_root(root)
    sample_root = run_root / sample_id
    expected_inputs = {
        Path(deconv_csv): sample_root / "input" / "RCTD_merged8.csv",
        Path(stage1_st_h5ad): sample_root / "input" / "st_stage1.h5ad",
        Path(stage1_gene_list): sample_root / "input" / "stage1_placeholder_genes.txt",
    }
    for supplied, expected in expected_inputs.items():
        if _require_file(supplied).resolve() != _require_file(expected).resolve():
            raise ValueError(f"The supplied input is not the validated cerebellum input: {supplied}")

    summary_path = sample_root / "stage2" / "preparation_summary.json"
    if not summary_path.is_file() or force:
        command = [sys.executable, str(scripts / "prepare_stage2.py"), "--sample", sample_id]
        if force:
            command.append("--force")
        _run_logged(
            command,
            cwd=root,
            env=_environment(root, run_root),
            log_path=sample_root / "stage2" / "reference" / "api_reference.log",
        )
    summary = _read_json(summary_path)
    public = _public_sample_root(root, sample_id)
    summary.update(
        {
            "gene_list": str(_publish(sample_root / "stage2/reference/genes_hvg8000.txt", public / "reference/genes.txt")),
            "mu_ref": str(_publish(sample_root / "stage2/reference/mu_ref_hvg8000.npy", public / "reference/mu_ref.npy")),
            "cell_type_order": str(_publish(sample_root / "stage2/reference/cell_type_order.txt", public / "reference/cell_type_order.txt")),
            "st_h5ad": str(_publish(sample_root / "stage2/input/st_stage2.h5ad", public / "dbit/st.h5ad")),
        }
    )
    return summary


def extract_openphenom(
    *,
    project_root: str | Path | None = None,
    sample_id: str,
    python_executable: str | Path | None = None,
    batch_size: int = 16,
    force: bool = False,
) -> dict[str, Any]:
    root, scripts = _workflow_paths(project_root)
    selected_python = root / "openphenom_env" / "bin" / "python" if python_executable is None else Path(python_executable)
    run_root = _validated_run_root(root)
    sample_root = run_root / sample_id
    summary_path = sample_root / "tissue_bbox" / "embedding" / "embedding_summary.json"
    output_h5 = sample_root / "tissue_bbox" / "embedding" / "grid_embedding.h5"
    if not output_h5.is_file() or force:
        command = [
            str(_require_file(selected_python)),
            str(scripts / "extract_openphenom.py"),
            "--sample",
            sample_id,
            "--selection",
            "tissue-bbox",
            "--batch-size",
            str(int(batch_size)),
        ]
        if force:
            command.append("--force")
        _run_openphenom(
            command,
            cwd=root,
            env=_environment(root, run_root),
            log_path=sample_root / "tissue_bbox" / "embedding" / "api_openphenom.log",
        )
    summary = _read_json(summary_path)
    public_output = _publish(output_h5, _public_sample_root(root, sample_id) / "embedding/grid_embedding.h5")
    summary["output_h5"] = str(public_output)
    return summary


def apply_mask(
    *,
    project_root: str | Path | None = None,
    sample_id: str,
    force: bool = False,
) -> dict[str, Any]:
    root, scripts = _workflow_paths(project_root)
    run_root = _validated_run_root(root)
    sample_root = run_root / sample_id
    summary_path = sample_root / "mask" / "validation_summary.json"
    if not summary_path.is_file() or force:
        command = [sys.executable, str(scripts / "apply_spacerec_mask.py"), "--sample", sample_id]
        if force:
            command.append("--force")
        _run_logged(
            command,
            cwd=root,
            env=_environment(root, run_root),
            log_path=sample_root / "mask" / "api_mask.log",
        )
    summary = _read_json(summary_path)
    public = _public_sample_root(root, sample_id)
    summary.update(
        {
            "mask_h5": str(_publish(sample_root / "mask/he_mask_grid.h5", public / "mask/he_mask_grid.h5")),
            "training_h5": str(_publish(sample_root / "embedding/grid_embedding_train_filtered.h5", public / "embedding/grid_embedding_train_filtered.h5")),
            "mask_summary": str(_publish(sample_root / "mask/mask_summary.json", public / "mask/mask_summary.json")),
        }
    )
    return summary


def train_stage1(
    *,
    project_root: str | Path | None = None,
    sample_id: str,
    force: bool = False,
) -> dict[str, Any]:
    root, scripts = _workflow_paths(project_root)
    run_root = _validated_run_root(root)
    sample_root = run_root / sample_id
    summary_path = sample_root / "model" / "stage1_summary.json"
    if force and summary_path.parent.exists():
        shutil.rmtree(summary_path.parent)
    if not summary_path.is_file():
        _run_training(
            [sys.executable, str(scripts / "train_stage1.py"), "--sample", sample_id],
            cwd=root,
            env=_environment(root, run_root),
            log_path=sample_root / "logs" / "api_stage1.log",
            checkpoint=sample_root / "model" / "training" / "model" / "stage1_router" / "checkpoints" / "last.ckpt",
            description=f"Stage 1 {sample_id}",
        )
    summary = _read_json(summary_path)
    public = _public_sample_root(root, sample_id) / "model_pred"
    summary.update(
        {
            "checkpoint": str(_publish(Path(summary["checkpoint"]), public / "stage1_checkpoint.ckpt")),
            "grid_type_csv": str(_publish(Path(summary["grid_type_csv"]), public / "stage1_grid_type.csv")),
            "stage1_summary": str(_publish(summary_path, public / "stage1_summary.json")),
        }
    )
    return summary


def train_stage2(
    *,
    project_root: str | Path | None = None,
    sample_id: str,
    force: bool = False,
) -> dict[str, Any]:
    root, scripts = _workflow_paths(project_root)
    run_root = _validated_run_root(root)
    sample_root = run_root / sample_id
    summary_path = sample_root / "stage2" / "validation_summary.json"
    if force:
        training_dir = sample_root / "stage2" / "training"
        if training_dir.exists():
            shutil.rmtree(training_dir)
        for path in (
            summary_path,
            sample_root / "stage2" / "adapted_stage1_realgenes.ckpt",
            sample_root / "stage2" / "checkpoint_adaptation_summary.json",
        ):
            if path.exists():
                path.unlink()
    if not summary_path.is_file():
        _run_training(
            [sys.executable, str(scripts / "train_stage2.py"), "--sample", sample_id],
            cwd=root,
            env=_environment(root, run_root),
            log_path=sample_root / "stage2" / "logs" / "api_stage2.log",
            checkpoint=sample_root / "stage2" / "training" / "model" / "stage2_expression" / "checkpoints" / "last.ckpt",
            description=f"Stage 2 {sample_id}",
        )
    summary = _read_json(summary_path)
    public = _public_sample_root(root, sample_id) / "model_pred" / "train"
    summary.update(
        {
            "stage1_checkpoint": str(_publish(sample_root / "stage2/adapted_stage1_realgenes.ckpt", public / "stage1_checkpoint.ckpt")),
            "stage2_checkpoint": str(_publish(Path(summary["stage2_checkpoint"]), public / "stage2_checkpoint.ckpt")),
            "grid_predictions_h5": str(_publish(Path(summary["grid_predictions_h5"]), public / "grid_predictions.h5")),
            "training_summary": str(_publish(sample_root / "stage2/training/summary.json", public / "summary.json")),
            "validation_summary": str(_publish(summary_path, public / "validation_summary.json")),
        }
    )
    return summary


__all__ = [
    "apply_mask",
    "extract_openphenom",
    "prepare_reference",
    "train_stage1",
    "train_stage2",
]
