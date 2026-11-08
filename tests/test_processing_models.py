"""Tokenizer/model tests against locally cached HF checkpoints (skipped if unavailable)."""

import pytest
import torch

from geosteer.data import PromptRecord, Stratum
from geosteer.processing import (PromptFormatter, collate_left, extract_activations, forward_hooks, get_decoder_layers,
                                 make_add_hook, verify_batched_extraction)

QWEN_SMALL = ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775")
TOKENIZERS = [QWEN_SMALL, ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
              ("meta-llama/Llama-3.2-1B-Instruct", "9213176726f574b556790deb65791e0c5aa438b6")]
PROMPTS = ["How do I kill a Python process?", "  Write a haiku about rain.  ",
           "Explain photosynthesis", "What is 2+2?\nAnswer briefly.", "naïve café — résumé"]


def _tok(name, rev):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(name, revision=rev)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"tokenizer unavailable: {e}")


@pytest.mark.model
@pytest.mark.parametrize("name,rev", TOKENIZERS)
@pytest.mark.parametrize("prompt", PROMPTS)
def test_positions(name, rev, prompt):
    tok = _tok(name, rev)
    fp = PromptFormatter(tok)(prompt)
    decoded_inst = tok.decode(fp.input_ids[: fp.t_inst + 1])
    assert decoded_inst.rstrip().endswith(prompt.strip()[-5:])
    assert fp.t_post == len(fp.input_ids) - 1 and fp.t_inst < fp.t_post
    # tokens after t_inst are pure template (end-of-turn + assistant header)
    needle = prompt.strip()
    template_tail = fp.text[fp.text.rfind(needle) + len(needle):]
    assert tok.decode(fp.input_ids[fp.t_inst + 1:]) == template_tail
    assert "assistant" in template_tail


@pytest.mark.model
def test_llama_template_date_is_pinned():
    tok = _tok(*TOKENIZERS[2])
    assert "26 Jul 2024" in PromptFormatter(tok).render("hi")


@pytest.mark.model
def test_collate_left_indices():
    tok = _tok(*QWEN_SMALL)
    f = PromptFormatter(tok)
    fps = [f("short"), f("a considerably longer instruction with many more tokens in it")]
    batch = collate_left(fps, tok.pad_token_id, ["t_inst", "t_post"])
    for b, fp in enumerate(fps):
        ti, tp = batch["index"][b].tolist()
        assert batch["input_ids"][b, ti] == fp.input_ids[fp.t_inst]
        assert tp == batch["input_ids"].shape[1] - 1
        assert batch["position_ids"][b, tp] == len(fp.input_ids) - 1


@pytest.fixture(scope="module")
def small_model():
    from geosteer.processing import load_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return load_model(*QWEN_SMALL, dtype="bf16" if device == "cuda" else "fp32", device=device)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"model unavailable: {e}")


def _records():
    return [PromptRecord(uid=f"t:{i}", text=p, source="t", source_id=str(i), stratum=Stratum.BENIGN, harm_label=0,
                         group_id=f"t:{i}") for i, p in enumerate(PROMPTS)]


@pytest.mark.model
def test_hook_matches_hidden_states(small_model):
    """Block-l output hook == output_hidden_states[l+1] for all but the last layer (which HF normalises)."""
    model, tok = small_model
    f = PromptFormatter(tok)
    recs = _records()[:1]
    acts = extract_activations(model, f, recs, batch_size=1, show_progress=False)
    ids = torch.tensor([f(recs[0].text).input_ids], device=model.device)
    hs = model(ids, output_hidden_states=True).hidden_states
    n = len(get_decoder_layers(model))
    for l in range(n - 1):
        ref = hs[l + 1][0, -1].float().cpu()
        assert torch.allclose(acts.get("t_post", l)[0], ref, atol=1e-3, rtol=1e-3)


@pytest.mark.model
def test_batched_matches_unpadded(small_model):
    model, tok = small_model
    report = verify_batched_extraction(model, PromptFormatter(tok), _records(), batch_size=5)
    assert report["passed"], report


@pytest.mark.model
def test_add_hook_prefill_only(small_model):
    model, tok = small_model
    f = PromptFormatter(tok)
    ids = torch.tensor([f("hello").input_ids], device=model.device)
    layer = 3
    store = {}
    def grab(m, i, o):
        store["h"] = (o[0] if isinstance(o, tuple) else o).detach().float()
    v = torch.zeros(model.config.hidden_size)
    v[0] = 5.0
    with forward_hooks(model, [(layer, grab)]):
        model(ids)
    base = store["h"].clone()
    pos = torch.tensor([[ids.shape[1] - 1]])
    with forward_hooks(model, [(layer, make_add_hook(v, pos)), (layer, grab)]):
        model(ids)
    delta = (store["h"] - base)[0]
    assert torch.allclose(delta[-1, 0], torch.tensor(5.0, device=delta.device), atol=0.1)
    assert delta[:-1].abs().max() == 0
    # single-token (decode-step) call is untouched in prefill_only mode
    with forward_hooks(model, [(layer, make_add_hook(v, None)), (layer, grab)]):
        model(ids[:, :1])
    h1 = store["h"]
    with forward_hooks(model, [(layer, grab)]):
        model(ids[:, :1])
    assert torch.equal(h1, store["h"])


@pytest.mark.model
def test_generation_and_prefill_steering(small_model):
    from geosteer.processing import generate_responses
    model, tok = small_model
    f = PromptFormatter(tok)
    recs = _records()[:3]
    base = generate_responses(model, f, recs, max_new_tokens=12, batch_size=3, show_progress=False)
    assert [r["uid"] for r in base] == [r.uid for r in recs]
    assert all(r["finish_reason"] in ("stop", "length") and r["n_new_tokens"] <= 12 for r in base)
    calls = []
    v = torch.randn(model.config.hidden_size) * 200
    def hooks_fn(batch):
        calls.append(batch["index"].shape)
        return [(10, make_add_hook(v, batch["index"][:, [1]]))]  # steer at t_post, prefill only
    steered = generate_responses(model, f, recs, max_new_tokens=12, batch_size=3, hooks_fn=hooks_fn,
                                 show_progress=False)
    assert calls == [torch.Size([3, 2])]
    assert any(a["response"] != b["response"] for a, b in zip(base, steered))
