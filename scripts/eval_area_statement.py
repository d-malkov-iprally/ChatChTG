"""Report the area answers that code stated in one eval run. Reads result files only.

In a run of `chatchtg eval --composer areas` the answers that the area pipeline is
confident of are stated in code (model `area-pipeline`). This script counts those rows:
how many are correct by the eval rule, and how many by the pipeline's own unrounded value.
The two differ only when the answer text loses the number or the rounding to two decimals
moves it across the 2% line.

The other area rows went to the LLM. The script counts them and never calls them right or
wrong. In a run with the fake provider they are not results: the fake LLM echoes the room
table, and the rule reads a number from the echo.

usage: uv run python scripts/eval_area_statement.py --results RUN.jsonl [--compare OTHER.jsonl]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chatchtg.evals.area_statement import report


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def describe(row: dict) -> str:
    value = row["context"]["computed"]["value"]
    return (
        f"  {row['path']}: {row['question']}\n"
        f"    expected {row['expected']}, value {value}, read from the text: {row['extracted_value']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, required=True, help="an eval run with --composer areas")
    parser.add_argument("--compare", type=Path, help="another eval run on the same questions")
    args = parser.parse_args()

    rows = read_rows(args.results)
    result = report(rows, read_rows(args.compare) if args.compare else None)
    stated = result["stated"]
    print(f"Area questions: {stated + result['not_answered']}")
    print(f"Stated in code (model area-pipeline): {stated}")
    print(f"  correct by the rule, on the text:   {result['stated_correct']}")
    print(f"  correct by the pipeline's value:    {result['value_correct']}")
    print(f"Not stated in code (sent to the LLM): {result['not_answered']}")
    models = sorted({row["model"] for row in rows if row["type"] == "area"} - {"area-pipeline"})
    if "fake" in models:
        print("  These rows were answered by the fake LLM. They are not results.")
    lost_value = result["text_lost_the_value"]
    print(f"Stated rows that are wrong although the value is right: {len(lost_value)}")
    for row in lost_value:
        print(describe(row))
    if args.compare:
        print(f"\nThe same {stated - result['compare_missing']} questions in {args.compare.name}:")
        if result["compare_missing"]:
            print(f"  ({result['compare_missing']} stated questions are not in that run)")
        print(f"  correct by the rule: {result['compare_correct']}")
        print(f"  cost: ${result['compare_cost_usd']:.4f}")
        if stated > result["compare_missing"]:
            saved = result["compare_cost_usd"] / (stated - result["compare_missing"])
            print(f"  cost saved per stated question: ${saved:.4f}")
        for name, title in (("gained", "Correct here and not there"), ("lost", "Correct there and not here")):
            print(f"{title}: {len(result[name])}")
            for row in result[name]:
                print(describe(row))


if __name__ == "__main__":
    main()
