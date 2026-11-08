"""Unified prompt record used across every dataset and split.

Labels kept deliberately separate (plan §4-5):
  * ``harm_label``  -- whether the *request* is harmful (dataset-provided, audited).
  * ``stratum``     -- sampling stratum (harmful / alarming-benign / ordinary benign).
  * observed refusal is *not* stored here: it is model-specific and produced
    later from generations, keyed by (model variant, uid).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class Stratum:
    HARMFUL = "harmful"
    ALARMING_BENIGN = "alarming_benign"  # benign but superficially alarming (over-refusal bait)
    BENIGN = "benign"                    # ordinary benign instructions
    OTHER = "other"                      # non-harm noncompliance (e.g. CoCoNot "unsupported") -- excluded from strata pools

    PRIMARY = (HARMFUL, ALARMING_BENIGN, BENIGN)
    ALL = (HARMFUL, ALARMING_BENIGN, BENIGN, OTHER)


@dataclass
class PromptRecord:
    uid: str                        # globally unique: "{source}:{source_id}"
    text: str                       # the user instruction, exactly as fed to the chat template
    source: str                     # registry key, e.g. "jbb_harmful"
    source_id: str                  # id within the source
    stratum: str                    # one of Stratum.ALL
    harm_label: int | None          # 1 harmful, 0 benign, None unknown/not applicable
    group_id: str                   # behaviour/topic group; all members land in the same split
    category: str | None = None     # source-provided harm/topic category
    pair_id: str | None = None      # topic-matched harmful/benign partner key (e.g. JBB index)
    reference: str | None = None    # reference answer for utility tasks (GSM8K, MATH500)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stratum not in Stratum.ALL:
            raise ValueError(f"{self.uid}: unknown stratum {self.stratum!r}")
        if self.harm_label not in (0, 1, None):
            raise ValueError(f"{self.uid}: harm_label must be 0, 1 or None")
        if not self.text or not self.text.strip():
            raise ValueError(f"{self.uid}: empty text")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PromptRecord":
        return cls(**d)
