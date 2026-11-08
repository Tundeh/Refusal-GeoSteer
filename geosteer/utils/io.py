"""File I/O helpers: JSON/JSONL, hashing, tensor bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, sort_keys=False)
        f.write("\n")
    return path


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from e


def read_jsonl(path: str | Path) -> list[dict]:
    return list(iter_jsonl(path))


def write_jsonl(rows: Iterable[dict], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(*parts: Any) -> int:
    """Deterministic 64-bit hash (Python's hash() is salted per process)."""
    key = "\x1f".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def save_tensor_bundle(path: str | Path, tensors: dict[str, torch.Tensor], meta: dict) -> Path:
    """Save tensors plus JSON-serialisable metadata in one torch file."""
    path = Path(path)
    ensure_dir(path.parent)
    torch.save({"tensors": {k: v.contiguous() for k, v in tensors.items()}, "meta": meta}, path)
    return path


def load_tensor_bundle(path: str | Path) -> tuple[dict[str, torch.Tensor], dict]:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    return obj["tensors"], obj["meta"]
