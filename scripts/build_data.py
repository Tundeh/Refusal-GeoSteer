"""Fetch pinned sources, merge duplicate groups, allocate splits, write data + manifest.

    python scripts/build_data.py --config configs/data.yaml

Outputs (under $GEOSTEER_DATA_DIR, default ./data):
    processed/<stage>/<role>.jsonl  records per role, grouped by stage (directions / repair / test)
    processed/split_ids.json        uid lists and file paths per role, nested subsets (commit this)
    processed/manifest.json     provenance, dedup audit, counts, file hashes (commit this)
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geosteer.data import (RoleSpec, allocate_splits, load_source, merge_duplicate_groups, nested_subsets,
                           validate_splits)
from geosteer.data.store import processed_dir
from geosteer.utils import get_logger, load_config, sha256_file, write_json, write_jsonl

log = get_logger("build_data")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = cfg["seed"]
    out = processed_dir()

    records, provenance = [], {}
    for name, opts in cfg["sources"].items():
        res = load_source(name, max_records=(opts or {}).get("max_records"), seed=seed)
        log.info("%-18s %5d records  %s", name, len(res.records), dict(Counter(r.stratum for r in res.records)))
        records += res.records
        provenance[name] = res.provenance
    uids = [r.uid for r in records]
    assert len(uids) == len(set(uids)), "duplicate uids across sources"

    report = merge_duplicate_groups(records, threshold=cfg["dedup_threshold"])
    log.info("dedup: %d -> %d groups; %d exact, %d near pairs; cross-source %s",
             report.n_groups_before, report.n_groups_after, len(report.exact_pairs), len(report.near_pairs),
             report.cross_source_pairs)

    roles = [RoleSpec.from_config(n, c) for n, c in cfg["roles"].items()]
    result = allocate_splits(records, roles, cfg["reserved"], cfg.get("remainder", []), seed)
    problems = validate_splits(result)
    if problems:
        raise SystemExit("Split leakage detected:\n  " + "\n  ".join(problems[:20]))

    subsets = {}
    for role, sc in cfg.get("nested_subsets", {}).items():
        subs = nested_subsets(result.roles[role], sc["sizes"], sc["strata"], seed, name=role)
        subsets[role] = {str(n): ids for n, ids in subs.items()}

    stage_of = {role: stage for stage, roles_ in cfg["stages"].items() for role in roles_}
    missing = [r.name for r in roles if r.name not in stage_of] + [r for r in cfg["reserved"].values() if r not in stage_of]
    if missing:
        raise SystemExit(f"roles without a stage in config: {sorted(set(missing))}")
    # Remove previously generated split files so dropped roles don't linger.
    for old in out.rglob("*.jsonl"):
        old.unlink()
    files, sha = {}, {}
    for role, stage in stage_of.items():
        rel = f"{stage}/{role}.jsonl"
        sha[role] = sha256_file(write_jsonl((r.to_dict() for r in result.roles.get(role, [])), out / rel))
        files[role] = rel
    not_written = {role: len(recs) for role, recs in result.roles.items() if role not in stage_of}
    if not_written:
        log.info("not written (no stage): %s", not_written)

    write_json({"files": files,
                "roles": {role: [r.uid for r in result.roles.get(role, [])] for role in stage_of},
                "subsets": subsets}, out / "split_ids.json")
    counts = result.counts()
    write_json({
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config_file": str(args.config),
        "config": cfg,
        "provenance": provenance,
        "dedup": report.summary(),
        "reserved_conflicts": result.reserved_conflicts,
        "shortfalls": result.shortfalls,
        "counts": counts,
        "stages": cfg["stages"],
        "files": files,
        "file_sha256": sha,
        "not_written": not_written,
    }, out / "manifest.json")

    for role, c in sorted(counts.items()):
        log.info("%-18s n=%5d groups=%5d %s", role, c["n"], c["n_groups"], c["by_stratum"])
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
