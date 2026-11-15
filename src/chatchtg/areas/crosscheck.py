"""The policy of an area cross-check: a measured area against a second estimate.

The area pipeline measures a room from the drawn lines. A second estimate, from an LLM
that sees the drawing, can show that a measured area is grossly wrong. This module holds
the comparison as pure code. ``scripts/eval_area_check.py`` replays it on an eval run to
measure what the check would change; nothing in the app calls it (see
``docs/benchmarks/2026-11-15-area-cross-check.md``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

DISAGREEMENT = 0.10
"""The two values disagree when they are further apart than this share of the tool's
value. Fixed before the check was measured."""


@dataclass(frozen=True)
class CrossCheck:
    tool_value: float
    llm_value: float | None
    difference: float | None
    """``abs(llm - tool) / tool``. None without an estimate."""
    verdict: Literal["agree", "disagree", "no-estimate"]

    @property
    def chosen(self) -> float:
        """The value to answer with: the estimate on a disagreement, else the tool's."""
        if self.verdict == "disagree" and self.llm_value is not None:
            return self.llm_value
        return self.tool_value


def eligible(basis: str | None, doubts: Sequence[str]) -> bool:
    """True when the check would be made: the answer rests on a measured area, and the
    pipeline has a reason to doubt it that costs nothing to know."""
    return basis == "measured" and bool(doubts)


def compare(tool_value: float, llm_value: float | None, threshold: float = DISAGREEMENT) -> CrossCheck:
    """Compare the tool's value with the LLM's estimate. An estimate that is missing or
    not above zero is no estimate."""
    if llm_value is None or llm_value <= 0:
        return CrossCheck(tool_value=tool_value, llm_value=None, difference=None, verdict="no-estimate")
    difference = abs(llm_value - tool_value) / abs(tool_value)
    verdict: Literal["agree", "disagree"] = "disagree" if difference > threshold else "agree"
    return CrossCheck(tool_value=tool_value, llm_value=llm_value, difference=difference, verdict=verdict)
