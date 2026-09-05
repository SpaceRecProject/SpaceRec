from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from . import dense18_virchow2


def _argv_value(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _module_main(module: ModuleType, **overrides: Any) -> None:
    argv = [str(Path(module.__file__ or "gridembedding").name)]
    for key, value in overrides.items():
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        argv.extend([flag, _argv_value(value)])

    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        module.main()
    finally:
        sys.argv = old_argv


def run_gridembedding(
    *,
    dataset: str,
    sample_id: str | None = None,
    force: bool = False,
    metadata_only: bool = False,
    max_patches: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    selected_dataset = str(dataset).lower()
    if selected_dataset not in {"brca", "crc"}:
        raise ValueError(f"Unsupported dataset: {dataset!r}.")

    selected_sample_id = sample_id or {"brca": "BREAST", "crc": "COLON_P2"}[selected_dataset]
    args = {
        "sample_id": selected_sample_id,
        "force": bool(force),
        "metadata_only": bool(metadata_only),
        "max_patches": max_patches,
    }
    args.update(kwargs)
    _module_main(dense18_virchow2, **args)
    return {
        "stage": "dense18_virchow2",
        "dataset": selected_dataset,
        "sample_id": selected_sample_id,
        "output_h5": str(args.get("output_h5", "")),
        "metadata_only": bool(metadata_only),
    }


__all__ = ["run_gridembedding"]
