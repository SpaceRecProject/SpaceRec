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


def _workflow_paths(project_root: str | Path) -> tuple[Path, Path]:
    root = Path(project_root).resolve()
    scripts = root / "scripts" / "handoff_cerebellum_spacerec_v2"
    if not scripts.is_dir():
        raise FileNotFoundError(scripts)
    return root, scripts


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
    project_root: str | Path,
    run_root: str | Path,
    sample_id: str,
    deconv_csv: str | Path,
    stage1_st_h5ad: str | Path,
    stage1_gene_list: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    root, scripts = _workflow_paths(project_root)
    run_root = Path(run_root)
    sample_root = run_root / sample_id
    input_dir = sample_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    copied_deconv = input_dir / "RCTD_merged8.csv"
    source_deconv = _require_file(Path(deconv_csv))
    if copied_deconv.exists() and not force:
        if copied_deconv.read_bytes() != source_deconv.read_bytes():
            raise ValueError(f"Existing staged deconvolution differs from {source_deconv}")
    else:
        shutil.copy2(source_deconv, copied_deconv)
    for source, name in [
        (_require_file(Path(stage1_st_h5ad)).resolve(), "st_stage1.h5ad"),
        (_require_file(Path(stage1_gene_list)).resolve(), "stage1_placeholder_genes.txt"),
    ]:
        target = input_dir / name
        if target.exists() or target.is_symlink():
            if target.resolve() != source:
                if not force:
                    raise ValueError(f"Existing staged input differs: {target}")
                target.unlink()
        if not target.exists():
            target.symlink_to(os.path.relpath(source, target.parent))

    summary_path = sample_root / "stage2" / "preparation_summary.json"
    if not summary_path.is_file() or force:
        command = [sys.executable, str(scripts / "prepare_stage2.py"), "--sample", sample_id]
        if force:
            command.append("--force")
        _run_logged(
            command,
            cwd=root,
            env=_environment(root, run_root),
            log_path=sample_root / "logs" / "reference.log",
        )
    return _read_json(summary_path)


def extract_openphenom(
    *,
    project_root: str | Path,
    run_root: str | Path,
    sample_id: str,
    python_executable: str | Path,
    batch_size: int = 16,
    force: bool = False,
) -> dict[str, Any]:
    root, scripts = _workflow_paths(project_root)
    run_root = Path(run_root)
    sample_root = run_root / sample_id
    summary_path = sample_root / "tissue_bbox" / "embedding" / "embedding_summary.json"
    output_h5 = sample_root / "tissue_bbox" / "embedding" / "grid_embedding.h5"
    if not output_h5.is_file() or force:
        command = [
            str(_require_file(Path(python_executable))),
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
            log_path=sample_root / "logs" / "openphenom.log",
        )
    summary = _read_json(summary_path)
    summary["output_h5"] = str(_require_file(output_h5))
    return summary


def apply_mask(
    *,
    project_root: str | Path,
    run_root: str | Path,
    sample_id: str,
    force: bool = False,
) -> dict[str, Any]:
    root, scripts = _workflow_paths(project_root)
    run_root = Path(run_root)
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
            log_path=sample_root / "logs" / "mask.log",
        )
    return _read_json(summary_path)


def train_stage1(
    *,
    project_root: str | Path,
    run_root: str | Path,
    sample_id: str,
) -> dict[str, Any]:
    root, scripts = _workflow_paths(project_root)
    run_root = Path(run_root)
    sample_root = run_root / sample_id
    summary_path = sample_root / "model" / "stage1_summary.json"
    if not summary_path.is_file():
        _run_training(
            [sys.executable, str(scripts / "train_stage1.py"), "--sample", sample_id],
            cwd=root,
            env=_environment(root, run_root),
            log_path=sample_root / "logs" / "stage1.log",
            checkpoint=sample_root / "model" / "training" / "model" / "stage1_router" / "checkpoints" / "last.ckpt",
            description=f"Stage 1 {sample_id}",
        )
    return _read_json(summary_path)


def train_stage2(
    *,
    project_root: str | Path,
    run_root: str | Path,
    sample_id: str,
) -> dict[str, Any]:
    root, scripts = _workflow_paths(project_root)
    run_root = Path(run_root)
    sample_root = run_root / sample_id
    summary_path = sample_root / "stage2" / "validation_summary.json"
    if not summary_path.is_file():
        _run_training(
            [sys.executable, str(scripts / "train_stage2.py"), "--sample", sample_id],
            cwd=root,
            env=_environment(root, run_root),
            log_path=sample_root / "logs" / "stage2.log",
            checkpoint=sample_root / "stage2" / "training" / "model" / "stage2_expression" / "checkpoints" / "last.ckpt",
            description=f"Stage 2 {sample_id}",
        )
    return _read_json(summary_path)


__all__ = [
    "apply_mask",
    "extract_openphenom",
    "prepare_reference",
    "train_stage1",
    "train_stage2",
]
