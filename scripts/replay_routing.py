"""Score the routing on a sample: code answers from a fake-LLM eval run, LLM answers from an earlier run.

    uv run python scripts/replay_routing.py <routing run with the fake LLM> <earlier LLM run>

A row of the routing run whose model is ``fake`` is a question that the routing sends to the
LLM. Its answer is taken from the earlier run, which must hold the same questions.
"""

from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path


def load(path: Path) -> dict[tuple[str, str], dict]:
    rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return {(row["path"], row["question"]): row for row in rows}


def sign_test(fixed: int, broken: int) -> float:
    """Two-sided sign test of ``fixed`` against ``broken``."""
    n, k = fixed + broken, min(fixed, broken)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def replay(routing: dict, llm: dict) -> dict[str, int | float]:
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
        "plain_llm_right": sum(bool(row["correct"]) for row in llm.values()),
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
