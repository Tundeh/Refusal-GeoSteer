"""Seeding, logging, paths, config loading, and torch device/dtype helpers."""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def data_dir() -> Path:
    """Root for raw/processed data; override with GEOSTEER_DATA_DIR."""
    return Path(os.environ.get("GEOSTEER_DATA_DIR", project_path("data")))


def artifacts_dir() -> Path:
    """Root for activations, directions, generations; override with GEOSTEER_ARTIFACTS_DIR."""
    return Path(os.environ.get("GEOSTEER_ARTIFACTS_DIR", project_path("artifacts")))


def artifact_path(kind: str, model_key: str, variant: str, name: str) -> Path:
    """artifacts/<kind>/<model_key>/<variant>/<name>, e.g. ("labels", "qwen2.5-1.5b", "base", "direction.jsonl")."""
    return artifacts_dir() / kind / model_key / variant / name


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name: str = "geosteer", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


_DTYPES = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16,
           "float16": torch.float16, "fp32": torch.float32, "float32": torch.float32}


def resolve_dtype(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    try:
        return _DTYPES[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown dtype {name!r}; expected one of {sorted(_DTYPES)}") from None


def limit_cpu_threads(n: int | None = None) -> int:
    """Cap torch CPU threads (default 8, or $GEOSTEER_NUM_THREADS). The geometry code runs many small
    matrix products; with all 64 cores on a shared, loaded host, thread contention made it ~25x slower."""
    n = n or int(os.environ.get("GEOSTEER_NUM_THREADS", 8))
    torch.set_num_threads(n)
    return n


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
