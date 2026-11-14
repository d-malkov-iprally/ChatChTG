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

# A number that does not follow a letter or "^", so the "2" of "m2" and "m^2" is not
# taken as the answer. Either digit groups ("1,234.5") or one decimal mark ("45.5", "45,5").
_NUMBER = re.compile(
    r"(?<![A-Za-z\d.,^])-?"
    r"(?:(?P<grouped>\d{1,3}(?:,\d{3})+(?![\d,])(?:\.\d+)?)|\d+(?:[.,]\d+)?)"
)


def extract_number(text: str) -> float | None:
    """Return the last number in ``text``.

    Commas between groups of three digits are ignored. Any other comma is a decimal mark.
    """
    matches = list(_NUMBER.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    return float(last.group().replace(",", "" if last.group("grouped") else "."))


def score(question_type: str, expected: Any, response_text: str) -> bool | None:
    """True or False when the answer can be scored, None when it cannot."""
    return score_value(question_type, expected, extract_number(response_text))


def score_value(question_type: str, expected: Any, value: float | None) -> bool | None:
    """Compare a value read from an answer with the expected answer.

    True or False when the question can be scored, None when it cannot.
    A ``value`` of None means that the answer gave no number.
    """
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        return None
    if question_type not in EXACT_TYPES | TOLERANCE_TYPES:
        return None
    if value is None:
        return False
    if question_type in EXACT_TYPES:
        return value == expected
    return abs(value - expected) <= RELATIVE_TOLERANCE * abs(expected)
