"""Read back built splits (written by scripts/build_data.py)."""

from __future__ import annotations

from pathlib import Path

from geosteer.data.schema import PromptRecord
from geosteer.utils import data_dir, read_json, read_jsonl


def processed_dir() -> Path:
    return data_dir() / "processed"


def load_role(role: str, subset: int | str | None = None, root: Path | None = None) -> list[PromptRecord]:
    """Records of one split role, optionally restricted to a frozen nested subset (e.g. tuning, 48)."""
    root = root or processed_dir()
    ids = load_split_ids(root)
    if role not in ids["files"]:
        raise KeyError(f"role {role!r} not built; available: {sorted(ids['files'])}")
    recs = [PromptRecord.from_dict(d) for d in read_jsonl(root / ids["files"][role])]
    if subset is not None:
        keep = set(ids["subsets"][role][str(subset)])
        recs = [r for r in recs if r.uid in keep]
        if len(recs) != len(keep):
            raise RuntimeError(f"{role}/{subset}: manifest lists {len(keep)} uids, found {len(recs)}")
    return recs


def roles_by_stage(root: Path | None = None) -> dict[str, list[str]]:
    """{'directions': ['direction'], 'repair': ['repair'], 'test': [...]} from the built split."""
    out: dict[str, list[str]] = {}
    for role, rel in load_split_ids(root)["files"].items():
        out.setdefault(rel.split("/")[0], []).append(role)
    return out


def load_split_ids(root: Path | None = None) -> dict:
    return read_json((root or processed_dir()) / "split_ids.json")
