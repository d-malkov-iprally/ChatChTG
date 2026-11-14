"""Prices of the models, to turn the token usage of a completion into a cost."""

from __future__ import annotations

from collections.abc import Mapping

# US dollars per million tokens: (input, output). Anthropic list prices of 2026-11-03,
# for prompts up to 100K tokens. The models that the API can fall back to are listed too.
PRICES: dict[str, tuple[float, float]] = {
    "fake": (0.0, 0.0),
    "fake-judge": (0.0, 0.0),
    "room-counter": (0.0, 0.0),  # an answer made in code, without an LLM call
    "claude-fable-5-2": (8.0, 40.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}

_INPUT_FIELDS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


def cost_usd(model: str, usage: Mapping[str, int]) -> float | None:
    """Return the cost of one completion in US dollars, or None when the model has no price.

    Cached input tokens are charged at the full input price. Cache reads cost less than
    that, so the result is never below the real cost by more than the cache-write surcharge.
    """
    price = PRICES.get(model)
    if price is None:
        return None
    input_tokens = sum(usage.get(name, 0) for name in _INPUT_FIELDS)
    return (input_tokens * price[0] + usage.get("output_tokens", 0) * price[1]) / 1_000_000
