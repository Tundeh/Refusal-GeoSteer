"""Residual-stream hooks shared by extraction and (later) intervention.

Convention: layer ``l`` (0-indexed) means the residual stream at the
**output of decoder block l** -- a forward hook on ``model.model.layers[l]``.
We do not use ``output_hidden_states``: its last entry has the final norm
applied in HF Llama/Qwen2, so it is not the same quantity as the other layers.
Extraction and steering must both use these helpers so the hook point matches.
"""

from __future__ import annotations

import contextlib
from typing import Callable, Iterator, Sequence

import torch
from torch import nn

HOOK_POINT = "decoder_block_output"


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    for path in ("model.layers", "model.model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList):
            return obj
    raise ValueError(f"Cannot locate decoder layers on {type(model).__name__}")


def _hidden(output) -> torch.Tensor:
    # Decoder layers return a tensor (transformers >= 4.54) or a tuple whose first item is the hidden state.
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output, hidden: torch.Tensor):
    return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden


@contextlib.contextmanager
def forward_hooks(model: nn.Module, hooks: Sequence[tuple[int, Callable]]) -> Iterator[None]:
    """Register ``(layer_idx, fn(module, inputs, output))`` forward hooks for the duration of the block."""
    layers = get_decoder_layers(model)
    handles = [layers[l].register_forward_hook(fn) for l, fn in hooks]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def make_capture_hook(store: dict[int, torch.Tensor], layer: int, index: torch.Tensor,
                      dtype: torch.dtype = torch.float32) -> Callable:
    """Capture hidden states at per-example positions.

    ``index`` is a [batch, n_pos] LongTensor of positions in the padded batch;
    stores ``store[layer] = hidden[b, index[b, p]]`` with shape [batch, n_pos, d].
    """
    def hook(module, inputs, output):
        h = _hidden(output)
        b = torch.arange(h.shape[0], device=h.device)[:, None]
        store[layer] = h[b, index.to(h.device)].detach().to(dtype).cpu()
    return hook


def make_add_hook(vector: torch.Tensor, positions: torch.Tensor | None = None,
                  prefill_only: bool = True) -> Callable:
    """Add ``vector`` (shape [d]) to the block output.

    ``positions``: [batch, n] LongTensor of padded-batch positions to edit, or
    None for every position. With ``prefill_only`` the hook is a no-op on
    single-token decoding steps (KV-cached generation).
    """
    def hook(module, inputs, output):
        h = _hidden(output)
        if prefill_only and h.shape[1] == 1:
            return output
        v = vector.to(device=h.device, dtype=h.dtype)
        h = h.clone()
        if positions is None:
            h += v
        else:
            b = torch.arange(h.shape[0], device=h.device)[:, None]
            h[b, positions.to(h.device)] += v
        return _replace_hidden(output, h)
    return hook
