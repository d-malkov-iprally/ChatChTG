"""Scoring rules of the Junction dataset.

Counts must be exact. Areas, lengths and volumes must be within 2%.
Text and other answers are free-form and are not scored here.
"""

from __future__ import annotations

import re
from typing import Any

EXACT_TYPES = frozenset({"count"})
TOLERANCE_TYPES = frozenset({"area", "length", "volume"})
RELATIVE_TOLERANCE = 0.02

# A number that does not follow a letter, so the "2" of "m2" is not taken as the answer.
_NUMBER = re.compile(r"(?<![A-Za-z\d.,])-?\d+(?:,\d{3})*(?:\.\d+)?")


def extract_number(text: str) -> float | None:
    """Return the last number in ``text``. Commas between digit groups are ignored."""
    matches = _NUMBER.findall(text)
    if not matches:
        return None
    return float(matches[-1].replace(",", ""))


def score(question_type: str, expected: Any, response_text: str) -> bool | None:
    """True or False when the answer can be scored, None when it cannot."""
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        return None
    if question_type not in EXACT_TYPES | TOLERANCE_TYPES:
        return None
    value = extract_number(response_text)
    if value is None:
        return False
    if question_type in EXACT_TYPES:
        return value == expected
    return abs(value - expected) <= RELATIVE_TOLERANCE * abs(expected)
