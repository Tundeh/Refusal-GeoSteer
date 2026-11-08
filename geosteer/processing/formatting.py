"""Chat-template formatting and explicit token positions (plan §5).

Positions, as indices into the *unpadded* formatted prompt:
  * ``t_inst``: last token of the user instruction text.
  * ``t_post``: last token of the full formatted prompt (end of the
    post-instruction template, i.e. after the generation prompt).

The instruction is located via character offsets of the rendered template,
not by assuming a fixed number of trailing template tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

POSITIONS = ("t_inst", "t_post")

# Llama-3.x templates stamp "Today Date" via strftime_now() when date_string is
# unset, which makes prompts (and activations) depend on the run date.
DEFAULT_TEMPLATE_KWARGS = {"date_string": "26 Jul 2024"}


@dataclass
class FormattedPrompt:
    text: str                  # rendered chat-template string
    input_ids: list[int]
    t_inst: int
    t_post: int

    def position(self, name: str) -> int:
        if name not in POSITIONS:
            raise KeyError(f"unknown position {name!r}; expected {POSITIONS}")
        return getattr(self, name)


class PromptFormatter:
    """Renders single-turn chat prompts and locates t_inst / t_post."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, system_prompt: str | None = None,
                 template_kwargs: dict | None = None):
        if not tokenizer.is_fast:
            raise ValueError("A fast tokenizer is required for offset mapping")
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.template_kwargs = dict(DEFAULT_TEMPLATE_KWARGS if template_kwargs is None else template_kwargs)

    def messages(self, instruction: str) -> list[dict]:
        msgs = [{"role": "system", "content": self.system_prompt}] if self.system_prompt is not None else []
        return msgs + [{"role": "user", "content": instruction}]

    def render(self, instruction: str) -> str:
        return self.tokenizer.apply_chat_template(self.messages(instruction), tokenize=False,
                                                  add_generation_prompt=True, **self.template_kwargs)

    def __call__(self, instruction: str) -> FormattedPrompt:
        text = self.render(instruction)
        # Templates may trim content (Llama-3), so search for the stripped instruction, last occurrence.
        needle = instruction.strip()
        start = text.rfind(needle)
        if start < 0:
            raise ValueError(f"instruction not found verbatim in rendered template: {needle[:80]!r}")
        end = start + len(needle)  # exclusive char index

        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        # Template text already contains BOS etc.; check we reproduce apply_chat_template's own tokenisation.
        ref = self.tokenizer.apply_chat_template(self.messages(instruction), tokenize=True,
                                                 add_generation_prompt=True, **self.template_kwargs)
        if hasattr(ref, "keys"):  # BatchEncoding in some versions
            ref = ref["input_ids"]
        if list(ref) != list(ids):
            raise RuntimeError("re-tokenised template differs from apply_chat_template(tokenize=True)")

        inst_tokens = [i for i, (s, e) in enumerate(offsets) if e > s and s < end and e > start]
        if not inst_tokens:
            raise RuntimeError(f"no tokens overlap the instruction span for {needle[:80]!r}")
        t_inst = inst_tokens[-1]
        if offsets[t_inst][1] > end:
            # A token straddles the instruction end (merged with template text): flag rather than guess.
            raise RuntimeError(f"last instruction token crosses into template text: {needle[-40:]!r}")
        return FormattedPrompt(text=text, input_ids=list(ids), t_inst=t_inst, t_post=len(ids) - 1)

    def describe(self) -> dict:
        """Metadata to store alongside activations."""
        probe = self.render("<<INSTRUCTION>>")
        return {"tokenizer": self.tokenizer.name_or_path, "system_prompt": self.system_prompt,
                "template_kwargs": self.template_kwargs, "rendered_example": probe}
