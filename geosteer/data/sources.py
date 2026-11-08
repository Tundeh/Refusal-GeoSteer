"""Dataset loaders. Every source is pinned to an exact revision and mapped to PromptRecord.

Gated HF mirrors (walledai/AdvBench, walledai/XSTest) are bypassed in favour of the
original GitHub releases at pinned commits. Sorry-Bench is gated with no official
mirror: we try HF first and fall back to the copy bundled with Zhao et al. [2]
(440 base-style prompts), recording which origin was actually used.
"""

from __future__ import annotations

import csv
import io
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from geosteer.data.schema import PromptRecord, Stratum
from geosteer.utils import data_dir, ensure_dir, get_logger, sha256_file, sha256_text, stable_hash

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pinned origins (checked 2026-11-08)
# ---------------------------------------------------------------------------
HF_REVISIONS = {
    "JailbreakBench/JBB-Behaviors": "886acc352a31533ffbcf4ef22c744658688086fc",
    "bench-llm/or-bench": "e36d8b80e81837c8a8f264bbb2a49f1b32c7e272",
    "tatsu-lab/alpaca": "dce01c9b08f87459cf36a430d809084718273017",
    "allenai/coconot": "2cbe16aabf9069f17e48c8daad8aeabc29469eb7",
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "HuggingFaceH4/MATH-500": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    "tatsu-lab/alpaca_eval": "2edc6fad8be6b14ea7230aabfd08188da6b8b814",
    "sorry-bench/sorry-bench-202406": "b34822276edde97592eda99c0b56d306f8830469",
}
GITHUB_RAW = {
    "advbench": "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
                "098262edf85f807224e70ecd87b9d83716bf6b73/data/advbench/harmful_behaviors.csv",
    "xstest": "https://raw.githubusercontent.com/paul-rottger/xstest/"
              "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d/xstest_prompts.csv",
    "sorry_bench_zhao": "https://raw.githubusercontent.com/CHATS-lab/LLMs_Encode_Harmfulness_Refusal_Separately/"
                        "eff841e063e5a59b8dbc094060923005393b1c36/data/sorry-badq.json",
}


@dataclass
class LoadResult:
    records: list[PromptRecord]
    provenance: dict = field(default_factory=dict)  # origin, revision, content hash, notes


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------
def _download_dir() -> Path:
    return ensure_dir(data_dir() / "raw" / "_downloads")


def fetch_url(url: str) -> tuple[str, str]:
    """Download (with on-disk cache) and return (text, sha256)."""
    path = _download_dir() / f"{sha256_text(url)[:16]}_{url.rsplit('/', 1)[-1]}"
    if not path.exists():
        log.info("Downloading %s", url)
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    return path.read_text(encoding="utf-8"), sha256_file(path)


def fetch_hf_file(repo_id: str, filename: str) -> tuple[Path, str]:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(repo_id, filename, repo_type="dataset", revision=HF_REVISIONS[repo_id]))
    return path, sha256_file(path)


def _hf_prov(repo_id: str, filename: str, sha: str, **extra) -> dict:
    return {"origin": "huggingface", "repo": repo_id, "revision": HF_REVISIONS[repo_id],
            "file": filename, "sha256": sha, **extra}


def _read_parquet(repo_id: str, filename: str):
    import pandas as pd

    path, sha = fetch_hf_file(repo_id, filename)
    return pd.read_parquet(path), _hf_prov(repo_id, filename, sha)


def _rec(source: str, sid, text, stratum: str, harm: int | None, *, group: str | None = None,
         **kw) -> PromptRecord | None:
    """Build a record; returns None for empty/missing text (filtered and counted in load_source)."""
    if not isinstance(text, str) or not text.strip():
        return None
    sid = str(sid)
    return PromptRecord(uid=f"{source}:{sid}", text=text.strip(), source=source, source_id=sid,
                        stratum=stratum, harm_label=harm, group_id=group or f"{source}:{sid}", **kw)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load_jbb(split: str) -> LoadResult:
    """JBB-Behaviors: 100 harmful + 100 topic-matched benign behaviours sharing an Index."""
    import pandas as pd

    repo = "JailbreakBench/JBB-Behaviors"
    fname = f"data/{split}-behaviors.csv"
    path, sha = fetch_hf_file(repo, fname)
    df = pd.read_csv(path)
    harmful = split == "harmful"
    source = f"jbb_{split}"
    recs = [
        _rec(source, r.Index, r.Goal, Stratum.HARMFUL if harmful else Stratum.ALARMING_BENIGN, int(harmful),
             group=f"jbb:{r.Index}", pair_id=f"jbb:{r.Index}", category=r.Category,
             meta={"behavior": r.Behavior, "origin_source": r.Source})
        for r in df.itertuples(index=False)
    ]
    return LoadResult(recs, _hf_prov(repo, fname, sha))


def load_jbb_harmful() -> LoadResult:
    return _load_jbb("harmful")


def load_jbb_benign() -> LoadResult:
    return _load_jbb("benign")


def load_advbench() -> LoadResult:
    text, sha = fetch_url(GITHUB_RAW["advbench"])
    rows = list(csv.DictReader(io.StringIO(text)))
    recs = [_rec("advbench", i, r["goal"], Stratum.HARMFUL, 1, meta={"target": r["target"]})
            for i, r in enumerate(rows)]
    return LoadResult(recs, {"origin": "github", "url": GITHUB_RAW["advbench"], "sha256": sha})


def _load_orbench(config: str, source: str, stratum: str, harm: int) -> LoadResult:
    import pandas as pd

    repo = "bench-llm/or-bench"
    fname = f"{config}.csv"
    path, sha = fetch_hf_file(repo, fname)
    df, prov = pd.read_csv(path), _hf_prov(repo, fname, sha)
    recs = [_rec(source, i, r.prompt, stratum, harm, category=r.category) for i, r in enumerate(df.itertuples())]
    return LoadResult(recs, prov)


def load_orbench_hard() -> LoadResult:
    """OR-Bench-Hard-1K: benign prompts that frequently trigger over-refusal."""
    return _load_orbench("or-bench-hard-1k", "orbench_hard", Stratum.ALARMING_BENIGN, 0)


def load_orbench_toxic() -> LoadResult:
    return _load_orbench("or-bench-toxic", "orbench_toxic", Stratum.HARMFUL, 1)


def load_alpaca() -> LoadResult:
    """Alpaca instructions with empty ``input`` only (single-turn, self-contained)."""
    repo = "tatsu-lab/alpaca"
    df, prov = _read_parquet(repo, "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet")
    recs = [_rec("alpaca", i, r.instruction, Stratum.BENIGN, 0, reference=r.output)
            for i, r in enumerate(df.itertuples()) if not str(r.input).strip()]
    prov["notes"] = "kept rows with empty input only"
    return LoadResult(recs, prov)


_XSTEST_FAMILY = {"nons_group_real_discr": "discr", "real_group_nons_discr": "discr",
                  "contrast_discr": "discr", "privacy_public": "privacy",
                  "privacy_fictional": "privacy", "contrast_privacy": "privacy"}


def load_xstest() -> LoadResult:
    """XSTest v2: 250 safe (alarming-benign) + 200 unsafe contrast prompts.

    Safe prompt k of a type and its contrast prompt k share a group, so a
    contrast pair never straddles splits.
    """
    text, sha = fetch_url(GITHUB_RAW["xstest"])
    rows = list(csv.DictReader(io.StringIO(text)))
    seen: dict[str, int] = {}
    recs = []
    for r in rows:
        t = r["type"]
        k = seen.get(t, 0)
        seen[t] = k + 1
        family = _XSTEST_FAMILY.get(t, t.removeprefix("contrast_"))
        safe = r["label"] == "safe"
        recs.append(_rec("xstest", r["id"], r["prompt"], Stratum.ALARMING_BENIGN if safe else Stratum.HARMFUL,
                         0 if safe else 1, group=f"xstest:{family}:{k}", category=t,
                         meta={"focus": r["focus"], "note": r["note"]}))
    return LoadResult(recs, {"origin": "github", "url": GITHUB_RAW["xstest"], "sha256": sha})


def load_sorry_bench() -> LoadResult:
    repo = "sorry-bench/sorry-bench-202406"
    try:
        path, sha = fetch_hf_file(repo, "question.jsonl")
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        prov = _hf_prov(repo, "question.jsonl", sha)
    except Exception as e:  # gated / no access
        log.warning("Sorry-Bench HF access failed (%s); using the Zhao et al. [2] copy.", type(e).__name__)
        text, sha = fetch_url(GITHUB_RAW["sorry_bench_zhao"])
        rows = [json.loads(l) for l in text.splitlines() if l.strip()]
        prov = {"origin": "github", "url": GITHUB_RAW["sorry_bench_zhao"], "sha256": sha,
                "notes": "fallback: HF repo gated; subset redistributed by [2]"}
    recs = [_rec("sorry_bench", r["question_id"], r["turns"][0], Stratum.HARMFUL, 1, category=str(r["category"]),
                 meta={"prompt_style": r.get("prompt_style", "base")})
            for r in rows if r.get("prompt_style", "base") == "base"]
    return LoadResult(recs, prov)


_COCONOT_SAFETY = "Requests with safety concerns"


def load_coconot_original() -> LoadResult:
    """CoCoNot noncompliance prompts. Only 'Requests with safety concerns' is labelled harmful;
    other noncompliance categories (incomplete, unsupported, humanising, ...) go to Stratum.OTHER."""
    repo = "allenai/coconot"
    recs, provs = [], []
    for split in ("train", "test"):
        df, prov = _read_parquet(repo, f"original/{split}-00000-of-00001.parquet")
        provs.append(prov)
        for r in df.itertuples():
            safety = r.category == _COCONOT_SAFETY
            recs.append(_rec("coconot_original", r.id, r.prompt, Stratum.HARMFUL if safety else Stratum.OTHER,
                             1 if safety else None, category=r.category,
                             meta={"subcategory": r.subcategory, "split": split}))
    return LoadResult(recs, {"origin": "huggingface", "parts": provs})


def load_coconot_contrast() -> LoadResult:
    """CoCoNot contrast set: benign look-alikes that should be complied with."""
    df, prov = _read_parquet("allenai/coconot", "contrast/test-00000-of-00001.parquet")
    recs = [_rec("coconot_contrast", r.id, r.prompt, Stratum.ALARMING_BENIGN, 0, category=r.category,
                 meta={"subcategory": r.subcategory}) for r in df.itertuples()]
    return LoadResult(recs, prov)


def load_gsm8k() -> LoadResult:
    df, prov = _read_parquet("openai/gsm8k", "main/test-00000-of-00001.parquet")
    recs = [_rec("gsm8k", i, r.question, Stratum.BENIGN, 0, reference=r.answer.split("####")[-1].strip(),
                 meta={"solution": r.answer}) for i, r in enumerate(df.itertuples())]
    return LoadResult(recs, prov)


def load_math500() -> LoadResult:
    repo = "HuggingFaceH4/MATH-500"
    path, sha = fetch_hf_file(repo, "test.jsonl")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    recs = [_rec("math500", r["unique_id"], r["problem"], Stratum.BENIGN, 0, reference=r["answer"],
                 category=r["subject"], meta={"level": r["level"]}) for r in rows]
    return LoadResult(recs, _hf_prov(repo, "test.jsonl", sha))


def load_alpaca_eval() -> LoadResult:
    repo = "tatsu-lab/alpaca_eval"
    path, sha = fetch_hf_file(repo, "alpaca_eval.json")
    rows = json.loads(path.read_text(encoding="utf-8"))
    recs = [_rec("alpaca_eval", i, r["instruction"], Stratum.BENIGN, 0, category=r["dataset"],
                 meta={"baseline_output": r["output"], "baseline_generator": r["generator"]})
            for i, r in enumerate(rows)]
    return LoadResult(recs, _hf_prov(repo, "alpaca_eval.json", sha))


SOURCES: dict[str, Callable[[], LoadResult]] = {
    "jbb_harmful": load_jbb_harmful,
    "jbb_benign": load_jbb_benign,
    "advbench": load_advbench,
    "orbench_hard": load_orbench_hard,
    "orbench_toxic": load_orbench_toxic,
    "alpaca": load_alpaca,
    "xstest": load_xstest,
    "sorry_bench": load_sorry_bench,
    "coconot_original": load_coconot_original,
    "coconot_contrast": load_coconot_contrast,
    "gsm8k": load_gsm8k,
    "math500": load_math500,
    "alpaca_eval": load_alpaca_eval,
}


def load_source(name: str, max_records: int | None = None, seed: int = 0) -> LoadResult:
    """Load one source; optionally keep a deterministic hash-ordered subset of ``max_records``."""
    if name not in SOURCES:
        raise KeyError(f"Unknown source {name!r}; available: {sorted(SOURCES)}")
    result = SOURCES[name]()
    n_raw = len(result.records)
    result.records = [r for r in result.records if r is not None]
    result.provenance["n_empty_dropped"] = n_raw - len(result.records)
    # Within-source exact duplicates would silently double-count; drop and log them.
    seen, unique = set(), []
    for r in result.records:
        if r.text not in seen:
            seen.add(r.text)
            unique.append(r)
    n_dupes = len(result.records) - len(unique)
    result.provenance["n_loaded"] = len(result.records)
    result.provenance["n_exact_duplicates_dropped"] = n_dupes
    if max_records is not None and len(unique) > max_records:
        unique = sorted(unique, key=lambda r: stable_hash(seed, "subsample", r.uid))[:max_records]
        unique.sort(key=lambda r: r.uid)
        result.provenance["subsampled_to"] = max_records
    result.records = unique
    return result
