"""Deterministic batched generation through the same formatter and hook points as extraction.

``hooks_fn(batch)`` may return ``[(layer, hook), ...]`` built from the collated
batch (e.g. ``make_add_hook(v, batch["index"][:, [1]])`` to steer at t_post
during prefill); the hooks stay registered for the whole ``generate`` call.
"""

from __future__ import annotations

import copy
from typing import Callable, Sequence

import torch
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from geosteer.data.schema import PromptRecord
from geosteer.processing.extraction import collate_left
from geosteer.processing.formatting import POSITIONS, PromptFormatter
from geosteer.processing.hooks import forward_hooks

# Greedy decoding with every sampling/penalty knob neutralised: model generation_config
# files ship non-default values (e.g. Qwen2.5 repetition_penalty=1.05) that still apply under greedy.
GREEDY = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None, "repetition_penalty": 1.0,
          "num_beams": 1}


def decoding_config(model: PreTrainedModel, max_new_tokens: int):
    gc = copy.deepcopy(model.generation_config)
    gc.update(max_new_tokens=max_new_tokens, **GREEDY)
    return gc


@torch.no_grad()
def generate_responses(model: PreTrainedModel, formatter: PromptFormatter, records: Sequence[PromptRecord],
                       max_new_tokens: int = 256, batch_size: int = 32,
                       hooks_fn: Callable[[dict], list] | None = None, show_progress: bool = True) -> list[dict]:
    tok = formatter.tokenizer
    gc = decoding_config(model, max_new_tokens)
    eos = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]
    device = next(model.parameters()).device

    formatted = [formatter(r.text) for r in records]
    order = sorted(range(len(records)), key=lambda i: len(formatted[i].input_ids))
    out: list[dict | None] = [None] * len(records)
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    for idx in tqdm(batches, disable=not show_progress, desc="generate"):
        batch = collate_left([formatted[i] for i in idx], tok.pad_token_id, POSITIONS)
        hooks = hooks_fn(batch) if hooks_fn else []
        with forward_hooks(model, hooks):
            seqs = model.generate(input_ids=batch["input_ids"].to(device),
                                  attention_mask=batch["attention_mask"].to(device),
                                  generation_config=gc, pad_token_id=tok.pad_token_id)
        new = seqs[:, batch["input_ids"].shape[1]:].cpu()
        for row, i in zip(new.tolist(), idx):
            stop = next((k for k, t in enumerate(row) if t in eos), None)
            toks = row if stop is None else row[:stop]
            out[i] = {"uid": records[i].uid, "response": tok.decode(toks, skip_special_tokens=True).strip(),
                      "n_new_tokens": len(toks), "finish_reason": "length" if stop is None else "stop",
                      "prompt_len": len(formatted[i].input_ids)}
    return out


def generation_meta(model: PreTrainedModel, formatter: PromptFormatter, max_new_tokens: int) -> dict:
    gc = decoding_config(model, max_new_tokens)
    return {"model": model.config.name_or_path, "model_dtype": str(next(model.parameters()).dtype),
            "decoding": {k: getattr(gc, k) for k in ("max_new_tokens", *GREEDY)},
            "eos_token_id": gc.eos_token_id, "formatter": formatter.describe()}
