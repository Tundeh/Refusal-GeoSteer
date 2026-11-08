"""Greedy generations for one model variant on one split role (no intervention).

    python scripts/generate.py --model qwen2.5-1.5b --role direction --device cuda:0

Writes artifacts/generations/<model>/<variant>/<role>.jsonl (+ .meta.json).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geosteer.data import load_role
from geosteer.processing import DEFAULT_SYSTEM_PROMPT, PromptFormatter, generate_responses, generation_meta, load_model
from geosteer.utils import artifact_path, get_logger, load_config, project_path, set_seed, write_json, write_jsonl

log = get_logger("generate")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="key in configs/models.yaml")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--weights", default=None, help="local checkpoint path for a modified variant")
    ap.add_argument("--role", default="direction")
    ap.add_argument("--subset", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help='"none" = template default')
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    mcfg = load_config(project_path("configs", "models.yaml"))[args.model]
    model, tok = load_model(args.weights or mcfg["id"], None if args.weights else mcfg["revision"], args.dtype,
                            args.device)
    formatter = PromptFormatter(tok, system_prompt=None if args.system_prompt.lower() == "none" else args.system_prompt)
    records = load_role(args.role, subset=args.subset)
    log.info("%s/%s: generating for %d prompts (%s)", args.model, args.variant, len(records), args.role)

    rows = generate_responses(model, formatter, records, args.max_new_tokens, args.batch_size)
    tag = args.role + (f"_{args.subset}" if args.subset else "")
    out = artifact_path("generations", args.model, args.variant, f"{tag}.jsonl")
    write_jsonl(rows, out)
    meta = generation_meta(model, formatter, args.max_new_tokens)
    meta.update(model_key=args.model, variant=args.variant, weights=args.weights, role=args.role, subset=args.subset,
                model_revision=None if args.weights else mcfg["revision"],
                n_truncated=sum(r["finish_reason"] == "length" for r in rows))
    write_json(meta, out.with_suffix(".meta.json"))
    log.info("wrote %s (%d truncated at %d tokens)", out, meta["n_truncated"], args.max_new_tokens)


if __name__ == "__main__":
    main()
