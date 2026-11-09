"""How much data does the geometry need? Availability and precision of r/d angles vs data budget.

    python scripts/budget_sweep.py --model qwen2.5-1.5b --budgets 50 100 150 200 300 450 600 \
        --weights balanced 1:2:1 --budget-seeds 5

For each (allocation, budget, selection seed) the direction pool is subsampled (nested within a
seed), directions are re-estimated, and per contrast we record whether it is available (all cells
>= min_count), reliable (noise-corrected cosine defined in >= min_valid_frac of bootstrap
replicates), its noise-corrected angle and 95% CI width. Layers are a fixed sparse set of relative
depths (plan §6). Infeasible budgets (a stratum quota exceeds the pool) are recorded, not skipped.

Writes artifacts/geometry/<model>/<variant>/budget_sweep.csv (per row) and budget_sweep_summary.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compute_geometry import run_geometry  # noqa: E402

from geosteer.data import parse_weights, split_seed, weights_tag
from geosteer.processing import ActivationSet
from geosteer.utils import artifact_path, get_logger, limit_cpu_threads, read_jsonl

log = get_logger("budget_sweep")

SWEEP_PAIRS = [("r", "d"), ("r", "d_matched"), ("r_alarming", "d")]
SWEEP_POSITIONS = [("t_post", "t_post"), ("t_post", "t_inst")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", default="base")
    ap.add_argument("--role", default="direction")
    ap.add_argument("--budgets", type=int, nargs="+", default=[50, 100, 150, 200, 300, 450, 600])
    ap.add_argument("--weights", nargs="+", default=["balanced", "1:2:1"])
    ap.add_argument("--budget-seeds", type=int, default=5, help="independent budget draws per setting")
    ap.add_argument("--depths", type=float, nargs="+", default=[0.25, 0.5, 0.75, 0.9])
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()
    limit_cpu_threads()

    acts = ActivationSet.load(artifact_path("activations", args.model, args.variant, f"{args.role}.pt"))
    labels = {r["uid"]: r for r in read_jsonl(artifact_path("labels", args.model, args.variant, f"{args.role}.jsonl"))}
    n_layers = acts.meta.get("n_layers_total", len(acts.layers))
    layers = sorted({round(d * (n_layers - 1)) for d in args.depths})
    base_seed = split_seed()
    log.info("%s: layers %s, budgets %s, weights %s, %d seeds", args.model, layers, args.budgets, args.weights,
             args.budget_seeds)

    rows = []
    for wtext in args.weights:
        weights = parse_weights(wtext)
        for budget in args.budgets:
            for k in range(args.budget_seeds):
                # draw k = 0 uses the split seed, reproducing compute_geometry --budget
                try:
                    df, meta = run_geometry(args.model, args.variant, args.role, budget, weights,
                                            contrast_pairs=SWEEP_PAIRS, layers=layers, min_count=args.min_count,
                                            n_boot=args.n_boot, seed=k, budget_seed=base_seed + k,
                                            acts=acts, labels=labels)
                except ValueError as e:  # quota exceeds the pool
                    rows.append({"weights": weights_tag(weights) or "balanced", "budget": budget, "budget_seed": k,
                                 "feasible": False, "reason": str(e)})
                    break
                pos = list(zip(df.r_pos, df.d_pos))
                df = df[[p in SWEEP_POSITIONS for p in pos]].copy()
                df["budget_seed"] = k
                df["feasible"] = True
                df["ci_width_dc"] = df.theta_dc_hi - df.theta_dc_lo if "theta_dc_hi" in df else float("nan")
                df["n_safe_refused"] = meta["cells"]["harm=0,refused=1"]
                df["n_alarming_refused"] = meta["cells"]["alarming,refused=1"]
                rows += df.to_dict("records")
            log.info("weights=%s budget=%d done", wtext, budget)

    out = artifact_path("geometry", args.model, args.variant, "budget_sweep.csv")
    sweep = pd.DataFrame(rows)
    sweep.to_csv(out, index=False)

    ok = sweep[sweep.feasible]
    summary = (ok.groupby(["weights", "r_name", "d_name", "r_pos", "d_pos", "budget"])
                 .agg(n_draws=("budget_seed", "nunique"), safe_refused=("n_safe_refused", "median"),
                      alarming_refused=("n_alarming_refused", "median"), available=("available", "mean"),
                      reliable=("reliable", "mean"), theta_dc_median=("theta_dc", "median"),
                      ci_width_median=("ci_width_dc", "median"))
                 .reset_index())
    infeasible = sweep[~sweep.feasible].drop_duplicates(["weights", "budget"])[["weights", "budget", "reason"]]
    summary.to_csv(out.with_name("budget_sweep_summary.csv"), index=False)
    if len(infeasible):
        log.info("infeasible settings:\n%s", infeasible.to_string(index=False))
    log.info("wrote %s and %s", out, out.with_name("budget_sweep_summary.csv"))


if __name__ == "__main__":
    main()
