"""Layerwise refusal/harmfulness geometry from saved activations + observed refusal labels.

    python scripts/compute_geometry.py --model qwen2.5-1.5b --role direction

Reads  artifacts/activations/<model>/<variant>/<role>.pt, artifacts/labels/<model>/<variant>/<role>.jsonl
Writes artifacts/geometry/<model>/<variant>/<role>.csv (one row per layer x contrast pair x position pair)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geosteer.data import load_role
from geosteer.processing import ActivationSet, Labels, layerwise_geometry
from geosteer.utils import artifact_path, get_logger, read_jsonl, write_json

log = get_logger("geometry")

# (r contrast, d contrast): primary first, then the plan's sensitivity analyses.
CONTRAST_PAIRS = [("r", "d"), ("r", "d_answered"), ("r", "d_refused"), ("r_harmful", "d")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", default="base")
    ap.add_argument("--role", default="direction")
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    acts = ActivationSet.load(artifact_path("activations", args.model, args.variant, f"{args.role}.pt"))
    labels = {r["uid"]: r for r in read_jsonl(artifact_path("labels", args.model, args.variant, f"{args.role}.jsonl"))}
    recs = {r.uid: r for r in load_role(args.role)}
    absent = [u for u in acts.uids if u not in labels]
    if absent:
        raise SystemExit(f"{len(absent)} activation uids have no refusal label, e.g. {absent[:3]}")

    lab = Labels(harm=np.array([recs[u].harm_label for u in acts.uids]),
                 refused=np.array([np.nan if labels[u]["refused"] is None else labels[u]["refused"]
                                   for u in acts.uids], dtype=float),
                 groups=np.array([recs[u].group_id for u in acts.uids]))
    cells = {f"harm={h},refused={r}": int(((lab.harm == h) & (lab.refused == r)).sum()) for h in (0, 1) for r in (0, 1)}
    cells["refused=NaN"] = int(np.isnan(lab.refused).sum())
    log.info("cell sizes: %s", cells)

    rows = []
    for r_name, d_name in CONTRAST_PAIRS:
        log.info("contrast %s vs %s", r_name, d_name)
        rows += layerwise_geometry(acts, lab, r_name=r_name, d_name=d_name, min_count=args.min_count,
                                   n_boot=args.n_boot, seed=args.seed)
    df = pd.DataFrame(rows)
    out = artifact_path("geometry", args.model, args.variant, f"{args.role}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    write_json({"cells": cells, "min_count": args.min_count, "n_boot": args.n_boot, "seed": args.seed,
                "contrast_pairs": CONTRAST_PAIRS, "activations_meta": {k: v for k, v in acts.meta.items()
                                                                        if k != "seq_lens"}},
               out.with_suffix(".meta.json"))

    prim = df[(df.r_name == "r") & (df.d_name == "d") & df.available]
    if len(prim):
        view = prim.pivot(index="layer", columns=["r_pos", "d_pos"], values="theta").round(1)
        log.info("theta(r, d) in degrees by layer:\n%s", view.to_string())
    else:
        log.warning("primary r vs d unavailable: %s", df[(df.r_name == "r") & (df.d_name == "d")].reason.iloc[0])
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
