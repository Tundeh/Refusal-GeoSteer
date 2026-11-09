"""Read back built splits (written by scripts/build_data.py)."""

from __future__ import annotations

from pathlib import Path

from geosteer.data.budget import select_budget
from geosteer.data.schema import PromptRecord
from geosteer.utils import data_dir, load_config, project_path, read_json, read_jsonl


def processed_dir() -> Path:
    return data_dir() / "processed"


def load_role(role: str, subset: int | str | None = None, root: Path | None = None,
              budget: int | None = None, budget_seed: int | None = None,
              budget_weights: dict[str, float] | None = None) -> list[PromptRecord]:
    """Records of one split role.

    ``subset``: a frozen nested subset listed in split_ids.json (e.g. repair, 48).
    ``budget``: a balanced nested subset of N prompts (see data/budget.py), seeded by the
    split seed unless ``budget_seed`` is given. Applied after ``subset``. ``budget_weights``
    shifts the stratum allocation; refused for roles in ``balanced_only_roles`` (configs/data.yaml).
    """
    if budget_weights is not None and role in balanced_only_roles():
        raise ValueError(f"role {role!r} must use balanced budgets (balanced_only_roles in configs/data.yaml)")
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
    if budget is not None:
        seed = budget_seed if budget_seed is not None else split_seed(root)
        recs = select_budget(recs, budget, seed, name=role, weights=budget_weights)
    return recs


def balanced_only_roles() -> set[str]:
    return set(load_config(project_path("configs", "data.yaml")).get("balanced_only_roles", []))


def split_seed(root: Path | None = None) -> int:
    return read_json((root or processed_dir()) / "manifest.json")["config"]["seed"]


def roles_by_stage(root: Path | None = None) -> dict[str, list[str]]:
    """{'directions': ['direction'], 'repair': ['repair'], 'test': [...]} from the built split."""
    out: dict[str, list[str]] = {}
    for role, rel in load_split_ids(root)["files"].items():
        out.setdefault(rel.split("/")[0], []).append(role)
    return out


def load_split_ids(root: Path | None = None) -> dict:
    return read_json((root or processed_dir()) / "split_ids.json")
