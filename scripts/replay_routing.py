"""Score the routing on a sample: code answers from a fake-LLM eval run, LLM answers from an earlier run.

    uv run python scripts/replay_routing.py <routing run with the fake LLM> <earlier LLM run>

A row of the routing run whose model is ``fake`` is a question that the routing sends to the
LLM. Its answer is taken from the earlier run, which must hold the questions of the routing run.

This is a replay and not a run of the routing: the earlier answers were given with the context
of the earlier run, which the routing does not send for every question. The counts also do not
say on which plans the code paths were fitted, so the totals are not a comparison with the
baseline (README, "Object counting").
"""

from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path


def load(path: Path) -> dict[tuple[str, str], dict]:
    found: dict[tuple[str, str], dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["path"], row["question"])
        if key in found:
            raise ValueError(f"{path} holds the question {key} twice: which answer counts is not known")
        found[key] = row
    return found


def sign_test(fixed: int, broken: int) -> float:
    """Two-sided sign test of ``fixed`` against ``broken``."""
    n, k = fixed + broken, min(fixed, broken)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def replay(routing: dict, llm: dict) -> dict[str, int | float]:
    missing = [key for key in routing if key not in llm]
    if missing:
        raise ValueError(f"The earlier run has no answer for {len(missing)} questions of the routing run, first {missing[0]}")
    code = code_right = llm_right = fixed = broken = 0
    for key, row in routing.items():
        earlier = bool(llm[key]["correct"])
        if row["model"] == "fake":
            llm_right += earlier
            continue
        code += 1
        right = bool(row["correct"])
        code_right += right
        fixed += right and not earlier
        broken += earlier and not right
    return {
        "questions": len(routing),
        # On the questions of the routing run only: the earlier run may hold more.
        "plain_llm_right": sum(bool(llm[key]["correct"]) for key in routing),
        "routing_right": code_right + llm_right,
        "code_answered": code,
        "code_right": code_right,
        "llm_answered": len(routing) - code,
        "llm_right": llm_right,
        "fixed": fixed,
        "broken": broken,
        "p": round(sign_test(fixed, broken), 4),
    }


if __name__ == "__main__":
    print(json.dumps(replay(load(Path(sys.argv[1])), load(Path(sys.argv[2]))), indent=2))
