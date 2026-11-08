"""Residual-stream activation extraction at t_inst / t_post for every decoder block."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from geosteer.data.schema import PromptRecord
from geosteer.processing.formatting import POSITIONS, FormattedPrompt, PromptFormatter
from geosteer.processing.hooks import HOOK_POINT, forward_hooks, get_decoder_layers, make_capture_hook
from geosteer.utils import get_logger, load_tensor_bundle, resolve_dtype, save_tensor_bundle

log = get_logger(__name__)


def load_model(model_id: str, revision: str | None = None, dtype: str | torch.dtype = "bf16",
               device: str | torch.device = "cuda") -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token_id is None:
        # Llama-3 ships a dedicated pad token in the vocab; fall back to EOS otherwise. Masked either way.
        tok.pad_token = "<|finetune_right_pad_id|>" if "<|finetune_right_pad_id|>" in tok.get_vocab() else tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=resolve_dtype(dtype))
    model.to(device).eval()
    model.requires_grad_(False)
    return model, tok


def collate_left(prompts: Sequence[FormattedPrompt], pad_id: int, positions: Sequence[str]) -> dict[str, torch.Tensor]:
    """Left-pad a batch; returns input_ids, attention_mask, position_ids and padded-batch indices."""
    max_len = max(len(p.input_ids) for p in prompts)
    ids = torch.full((len(prompts), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(prompts), max_len), dtype=torch.long)
    index = torch.empty((len(prompts), len(positions)), dtype=torch.long)
    for b, p in enumerate(prompts):
        off = max_len - len(p.input_ids)
        ids[b, off:] = torch.tensor(p.input_ids)
        mask[b, off:] = 1
        index[b] = torch.tensor([off + p.position(name) for name in positions])
    # Explicit position ids so real tokens start at 0 regardless of padding.
    position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
    return {"input_ids": ids, "attention_mask": mask, "position_ids": position_ids, "index": index}


@dataclass
class ActivationSet:
    acts: torch.Tensor                 # [N, n_pos, n_layers, d]
    uids: list[str]
    positions: list[str]
    layers: list[int]
    meta: dict = field(default_factory=dict)

    def get(self, position: str, layer: int) -> torch.Tensor:
        """[N, d] slice for one (position, layer)."""
        return self.acts[:, self.positions.index(position), self.layers.index(layer)]

    def subset(self, uids: Sequence[str]) -> "ActivationSet":
        where = {u: i for i, u in enumerate(self.uids)}
        idx = torch.tensor([where[u] for u in uids], dtype=torch.long)
        return ActivationSet(self.acts[idx], list(uids), self.positions, self.layers, self.meta)

    def save(self, path: str | Path) -> Path:
        meta = {**self.meta, "uids": self.uids, "positions": self.positions, "layers": self.layers}
        return save_tensor_bundle(path, {"acts": self.acts}, meta)

    @classmethod
    def load(cls, path: str | Path) -> "ActivationSet":
        tensors, meta = load_tensor_bundle(path)
        meta = dict(meta)
        uids, positions, layers = meta.pop("uids"), meta.pop("positions"), meta.pop("layers")
        return cls(tensors["acts"], uids, positions, layers, meta)


@torch.no_grad()
def extract_activations(model: PreTrainedModel, formatter: PromptFormatter, records: Sequence[PromptRecord],
                        layers: Sequence[int] | None = None, positions: Sequence[str] = POSITIONS,
                        batch_size: int = 16, store_dtype: str | torch.dtype = "fp32",
                        show_progress: bool = True) -> ActivationSet:
    n_layers = len(get_decoder_layers(model))
    layers = list(range(n_layers)) if layers is None else list(layers)
    positions = list(positions)
    store_dtype = resolve_dtype(store_dtype)
    pad_id = formatter.tokenizer.pad_token_id
    device = next(model.parameters()).device

    formatted = [formatter(r.text) for r in records]
    # Sort by length to minimise padding; restore order at the end.
    order = sorted(range(len(formatted)), key=lambda i: len(formatted[i].input_ids))
    out = torch.empty((len(records), len(positions), len(layers), model.config.hidden_size), dtype=store_dtype)

    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    for batch_idx in tqdm(batches, disable=not show_progress, desc="extract"):
        batch = collate_left([formatted[i] for i in batch_idx], pad_id, positions)
        store: dict[int, torch.Tensor] = {}
        hooks = [(l, make_capture_hook(store, l, batch["index"], store_dtype)) for l in layers]
        with forward_hooks(model, hooks):
            model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device),
                  position_ids=batch["position_ids"].to(device), use_cache=False)
        out[torch.tensor(batch_idx)] = torch.stack([store[l] for l in layers], dim=2)

    meta = {"model": model.config.name_or_path, "model_revision": getattr(model.config, "_commit_hash", None),
            "model_dtype": str(next(model.parameters()).dtype), "store_dtype": str(store_dtype),
            "hook_point": HOOK_POINT, "n_layers_total": n_layers, "formatter": formatter.describe(),
            "seq_lens": [len(f.input_ids) for f in formatted]}
    return ActivationSet(out, [r.uid for r in records], positions, layers, meta)


@torch.no_grad()
def verify_batched_extraction(model: PreTrainedModel, formatter: PromptFormatter, records: Sequence[PromptRecord],
                              batch_size: int = 8, min_cosine: float | None = None) -> dict:
    """Compare batched (left-padded) extraction with one-at-a-time unpadded runs.

    Exact equality is not expected in half precision (different kernels/reduction
    order); we require per-(example, position, layer) cosine >= ``min_cosine``.
    Default thresholds come from Qwen2.5-1.5B: fp32 batched-vs-single min cosine
    0.9999995, while bf16 single runs differ from fp32 by up to 0.9973 (worst
    at the last layer), so 0.995 in half precision still catches a real
    padding/indexing error (which gives unrelated vectors) without flagging bf16 noise.
    """
    if min_cosine is None:
        min_cosine = 0.9999 if next(model.parameters()).dtype == torch.float32 else 0.995
    batched = extract_activations(model, formatter, records, batch_size=batch_size, show_progress=False)
    single = extract_activations(model, formatter, records, batch_size=1, show_progress=False)
    a, b = batched.acts.float(), single.acts.float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)               # [N, P, L]
    rel = (a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-8)
    report = {"n": len(records), "batch_size": batch_size, "min_cosine": float(cos.min()),
              "mean_cosine": float(cos.mean()), "max_rel_err": float(rel.max()), "threshold": min_cosine,
              "worst_layer_by_position": {p: int(cos[:, i].min(0).values.argmin()) for i, p in enumerate(batched.positions)},
              "passed": bool(cos.min() >= min_cosine)}
    if not report["passed"]:
        log.warning("Batched extraction mismatch: %s", report)
    return report
