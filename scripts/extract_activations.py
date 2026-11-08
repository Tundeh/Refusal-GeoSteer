"""Extract residual-stream activations at t_inst / t_post for one model and split role.

    python scripts/extract_activations.py --model qwen2.5-1.5b --role direction --device cuda:0

Writes artifacts/activations/<model>/<variant>/<role>.pt and a padding-check report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geosteer.data import load_role
from geosteer.processing import DEFAULT_SYSTEM_PROMPT, PromptFormatter, extract_activations, load_model, verify_batched_extraction
from geosteer.utils import artifacts_dir, get_logger, load_config, project_path, set_seed, write_json

log = get_logger("extract")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="key in configs/models.yaml")
    ap.add_argument("--variant", default="base", help="model-variant label (base, cue_s1_seed0, ...)")
    ap.add_argument("--weights", default=None, help="local checkpoint path for a modified variant")
    ap.add_argument("--role", default="direction")
    ap.add_argument("--subset", default=None, help="nested subset size, e.g. 48")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--store-dtype", default="fp32")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help='"none" = template default')
    ap.add_argument("--verify-n", type=int, default=32, help="prompts for the batched-vs-unpadded check (0=skip)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    mcfg = load_config(project_path("configs", "models.yaml"))[args.model]
    model_src = args.weights or mcfg["id"]
    model, tok = load_model(model_src, None if args.weights else mcfg["revision"], args.dtype, args.device)
    formatter = PromptFormatter(tok, system_prompt=None if args.system_prompt.lower() == "none" else args.system_prompt)
    records = load_role(args.role, subset=args.subset)
    log.info("%s/%s: %d prompts from %s%s", args.model, args.variant, len(records), args.role,
             f"[{args.subset}]" if args.subset else "")

    out_dir = artifacts_dir() / "activations" / args.model / args.variant
    tag = args.role + (f"_{args.subset}" if args.subset else "")
    if args.verify_n:
        report = verify_batched_extraction(model, formatter, records[: args.verify_n], batch_size=min(8, args.batch_size))
        write_json(report, out_dir / f"{tag}.padcheck.json")
        log.info("padding check: %s", report)
        if not report["passed"]:
            raise SystemExit("Batched extraction disagrees with unpadded runs; see padcheck report.")

    acts = extract_activations(model, formatter, records, batch_size=args.batch_size, store_dtype=args.store_dtype)
    acts.meta.update(variant=args.variant, role=args.role, subset=args.subset, model_key=args.model,
                     weights=args.weights, model_revision=mcfg["revision"] if not args.weights else None)
    path = acts.save(out_dir / f"{tag}.pt")
    log.info("saved %s %s", tuple(acts.acts.shape), path)


if __name__ == "__main__":
    main()
