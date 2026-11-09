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
python scripts/compute_geometry.py    --model qwen2.5-1.5b --budget 100 --budget-weights 1:2:1   # data budget
python scripts/budget_sweep.py        --model qwen2.5-1.5b      # availability/precision vs budget
pytest -q
```

Notes:
* Layer `l` is the residual stream at the output of decoder block `l` (forward hook), not `output_hidden_states`, whose last entry is post-final-norm.
* Llama-3 chat templates stamp today's date unless `date_string` is passed; the formatter pins it.
* Sorry-Bench (HF) is gated: until access is granted, the loader falls back to the 440-prompt copy redistributed by Zhao et al. and records this in `manifest.json`.
* Observed refusal labels are model-specific (`artifacts/labels/<model>/<variant>/`), not part of the split files.
  Judge = Qwen2.5-7B-Instruct, prompt v2, read from A/B/C logits. A judge call that contradicts the keyword
  heuristic becomes NaN; partial refusals are NaN by default. Each run writes a blinded
  `*.audit.csv` for human review. Meta-Llama-3-8B-Instruct is not suitable as judge (it labels terse refusals of
  severe harmful requests as compliance).
* Geometry reports the plain angle `theta` and a noise-corrected angle `theta_dc` (primary). Plain cosines between
  noisy mean-difference vectors are pulled toward 90 deg when contrast cells are small; `theta_dc` subtracts the
  within-cell (cross-)covariance trace from inner products and squared norms, with a group bootstrap that accounts
  for resampling weights. Rows with `reliable == False` (too many replicates with less signal than noise) should
  not be interpreted.

# Data budgets

`load_role(role, budget=N)` and `--budget N` use N prompts, balanced across harmful / alarming-benign / benign,
deterministic and nested (budget 50 is a subset of budget 100). Defaults live under `budgets:` in `configs/data.yaml`.

* **Direction estimation** may use a fixed stratum allocation (`--budget-weights 1:2:1`, or `budget_weights.direction`
  in the config). Declare it before analysis and keep it identical across models and budgets: the mix changes what
  r and d measure. **Repair** budgets are always balanced (`balanced_only_roles`), because repair is scored on harmful
  compliance, over-refusal and utility together.
* A contrast whose cells fall below `min_count` (20) is reported as unavailable. At small budgets that is a result
  for H1, not something to engineer away.
* `r_alarming` (alarming-benign refused − alarming-benign answered) holds alarming wording fixed as well as harm,
  removing the OR-Bench-vs-Alpaca style confound in `r`.

Budget sweep on the base models (5 draws per setting, t_post, layers at relative depth 0.25/0.5/0.75/0.9;
"from N" = r vs d available and reliable in all draws; CI = median 95% width of the noise-corrected angle):

| Model | r vs d, balanced | r vs d, 1:2:1 | CI at 300, balanced / 1:2:1 | CI at 600, balanced |
|---|---|---|---|---|
| Qwen2.5-1.5B | from 150 | from 100 (CI ~47°) | 21° / 23° | 13° |
| Llama-3.2-1B | from 450 | from 300 | – / 34° | 16° |

The 1:2:1 allocation makes r estimable at smaller budgets but costs precision on the harm side (fewer harmful
prompts, fewer complete topic-matched pairs: `d_matched` becomes available later than under balanced budgets).

# Alignment with refusal-vs-safety-signal

Angle extraction is cross-checked against [fw2274/refusal-vs-safety-signal](https://github.com/fw2274/refusal-vs-safety-signal)
(commit `53cd177`, README §4.2-4.4, `src/extract/activations.py`, `src/prompting.py`, `src/probes/project.py`).

| Aspect | refusal-vs-safety-signal | This repo | Status |
|---|---|---|---|
| Token position | last prompt token after `add_generation_prompt` | `t_post` (same); `t_inst` added | aligned |
| Layer convention | `L = 1..n` = `hidden_states[L]` | `layer = 0..n-1` = output of block `layer`; CSV column `layer_1idx = layer + 1` | aligned for `L < n`; their `L = n` is post-final-norm, ours pre-norm (verified numerically) |
| Padding | left padding, `position_ids` from attention mask, batched vs unpadded check | same | aligned |
| Precision | bf16 forward, fp32 arithmetic | bf16 forward, fp32 storage, fp64 means | aligned |
| System prompt | `"You are a helpful assistant."` | same (`DEFAULT_SYSTEM_PROMPT`); previously template default | aligned |
| Llama date | pinned `26 Jul 2024` | same | aligned |
| Refusal direction r | unit(mean safe&refused − mean safe&answered) | `r`, same formula | aligned |
| Harm direction d | unit(mean harmful − mean safe) on a **topic-matched** corpus (XSTest contrasts + JBB pairs) | `d_matched`: same formula on complete JBB pairs in the direction pool; `d` (unconditioned, all strata) kept as the plan's primary | aligned in construction; XSTest is reserved for testing here (plan §4) |
| Angle | `acos(cos(r, d))`, point estimate | same plain `theta`, plus group-bootstrap CI and noise-corrected `theta_dc` | superset |
| Refusal labels | **taught** policy labels of a cue-arm LoRA model | **observed** labels (LLM judge + keyword consensus) of each model variant | intentional difference (plan §5); our base-model r is their proposed follow-up (§6.3) |
| Model | cue-arm fine-tune | base model now; cue/degraded variants later (plan §3) | pending |
| Fitting split | r and d from train rows only (probes evaluated on held-out) | all rows of the direction role, or a budget subset | differs; matters only once probes are added |

Numerical check, r vs `d_matched` at the last prompt token (degrees; theirs = cue-arm model, plain angle, README §5.1/5.5;
ours = base model, observed refusal, 600-prompt direction pool):

| Model | Layer (1-idx) | theirs | ours plain | ours noise-corrected |
|---|---|---|---|---|
| Qwen2.5-1.5B | 4 / 9 / 13 / 17 / 25 | 99 / 86 / 61 / 34 / 30 | 70 / 63 / 64 / 41 / 34 | 64 / 57 / 60 / 38 / 30 |
| Llama-3.2-1B | 3 / 5 / 9 / 12 / 15 | 79 / 49 / 40 / 42 / 46 | 57 / 44 / 46 / 44 / 41 | 44 / 32 / 41 / 40 / 37 |

Mid-to-late layers agree within a few degrees; early layers are less orthogonal in our base models. Expected sources:
different model (cue-arm LoRA vs base), taught vs observed refusal labels, and their d corpus adding XSTest contrasts.
