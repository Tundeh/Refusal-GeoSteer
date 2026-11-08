# Refusal-vs-harmfulness-geometry-steering
Does the angle between refusal direction and harmfulness direction predict when ordinary refusal steering works better than refusal steering perpendicular to harmfulness?


# Hypothesis
## H1
The layerwise angle between harmfulness and refusal directions helps predict which layer and steering direction will reduce harmful compliance while preserving benign utility. Geometry should therefore improve selection accuracy or reduce the behavioral tuning budget required.

# Code layout

```
geosteer/
  utils/        io (jsonl/json/hash/tensor bundles), misc (seed, logging, paths, config, dtype), text normalisation
  data/
    schema.py   PromptRecord: text, stratum (harmful / alarming_benign / benign / other), harm_label, group_id, pair_id
    sources.py  loaders for every dataset in the plan, pinned to exact HF revisions / GitHub commits
    dedup.py    exact + TF-IDF near-duplicate audit; merges duplicates into one behaviour group
    splits.py   group-aware role allocation (reserved test sources, quotas, remainder) + nested subsets
    store.py    load_role("repair", subset=48), roles_by_stage()
  processing/
    formatting.py  chat template rendering and explicit t_inst / t_post token positions (offset-mapped)
    hooks.py       residual-stream hooks at decoder-block outputs (shared by extraction and steering)
    extraction.py  batched left-padded extraction, padding check, ActivationSet save/load
    generation.py  deterministic greedy generation (same formatter/hooks; optional steering hooks)
    directions.py  difference-of-means contrasts (d, r, r_harmful, d_refused, d_answered),
                   joint group bootstrap, layerwise same-/cross-token geometry
  eval/
    refusal.py     observed-refusal labels: keyword heuristic + logit-based LLM judge, consensus rule
configs/        data.yaml (split design), models.yaml (pinned model revisions)
scripts/        build_data.py, extract_activations.py, generate.py, judge_refusals.py, compute_geometry.py
tests/          pytest (model tests use small locally cached checkpoints)
```

# Usage

```bash
pip install -e .[dev]
python scripts/build_data.py --config configs/data.yaml       # -> data/processed/{directions,repair,test}/*.jsonl + manifest
```

Data stages (`configs/data.yaml`):

| stage | role | contents |
|---|---|---|
| directions | `direction` | 600 prompts, 200 per stratum (harmful / alarming-benign / benign); all 100 topic-matched JBB pairs |
| repair | `repair` | 300 prompts, 100 per stratum; frozen nested subsets 24/48/96/192 |
| test | `harm_test` | Sorry-Bench + unseen AdvBench behaviours |
| test | `overrefusal_test` | XSTest (safe + unsafe contrasts) + held-out OR-Bench-Hard |
| test | `utility_test` | GSM8K test + AlpacaEval |

```bash
# refusal/harmfulness geometry for one model variant
python scripts/extract_activations.py --model qwen2.5-1.5b --role direction --device cuda:0
python scripts/generate.py            --model qwen2.5-1.5b --role direction --device cuda:0
python scripts/judge_refusals.py      --model qwen2.5-1.5b --role direction --device cuda:1   # judge: qwen2.5-7b
python scripts/compute_geometry.py    --model qwen2.5-1.5b --role direction
pytest -q
```

Notes:
* Layer `l` is the residual stream at the output of decoder block `l` (forward hook), not `output_hidden_states`, whose last entry is post-final-norm.
* Llama-3 chat templates stamp today's date unless `date_string` is passed; the formatter pins it.
* Sorry-Bench (HF) is gated: until access is granted, the loader falls back to the 440-prompt copy redistributed by Zhao et al. and records this in `manifest.json`.
* Observed refusal labels are model-specific (`artifacts/labels/<model>/<variant>/`), not part of the split files.
  Judge = Qwen2.5-7B-Instruct, prompt v2, read from A/B/C logits. A judge call that contradicts the keyword
  heuristic with confidence < 0.99 becomes NaN; partial refusals are NaN by default. Each run writes a blinded
  `*.audit.csv` for human review. Meta-Llama-3-8B-Instruct is not suitable as judge (it labels terse refusals of
  severe harmful requests as compliance).
