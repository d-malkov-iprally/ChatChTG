"""Score counts of objects: against the labels, and against the answers of the benchmark."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

# The start of an English count question of the benchmark -> the classes that it counts.
BENCHMARK_QUESTIONS = {
    "How many windows are shown": ("window",),
    "How many doors are shown": ("door",),
    "How many toilet bowls": ("toilet",),
    "How many sinks and wash basins": ("sink",),
    "How many showers and bathtubs": ("shower", "bathtub"),
}
BENCHMARK_FOLDER = "data/cubicasa5k/"

Counts = Mapping[str, Mapping[str, int]]
"""Image name -> class -> number of objects."""


def score_counts(predicted: Counts, reference: Counts, classes: list[str]) -> dict[str, dict]:
    """For each class: on how many images the count is exactly right, and the mean error.

    Only the images of ``reference`` are scored. An image that is missing in ``predicted``
    counts as an image with no predictions.
    """
    scores = {}
    for name in classes:
        errors = [
            abs(predicted.get(image, {}).get(name, 0) - counts.get(name, 0))
            for image, counts in reference.items()
        ]
        scores[name] = {
            "images": len(errors),
            "exact": sum(error == 0 for error in errors),
            "exact_rate": sum(error == 0 for error in errors) / len(errors) if errors else None,
            "mean_abs_error": sum(errors) / len(errors) if errors else None,
            "objects": sum(counts.get(name, 0) for counts in reference.values()),
        }
    return scores


def benchmark_answers(questions_file: Path) -> dict[str, dict[str, int]]:
    """Image name -> question start -> answer, for the count questions of ``BENCHMARK_QUESTIONS``."""
    answers: dict[str, dict[str, int]] = {}
    with questions_file.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if not row["path"].startswith(BENCHMARK_FOLDER):
                continue
            for start in BENCHMARK_QUESTIONS:
                if row["question"].startswith(start):
                    image = row["path"].removeprefix(BENCHMARK_FOLDER)
                    answers.setdefault(image, {})[start] = int(row["answer"])
    return answers


def score_against_benchmark(counts: Counts, answers: Mapping[str, Mapping[str, int]]) -> dict[str, dict]:
    """For each benchmark question: on how many images the counts give the benchmark's answer.

    Only images that are in ``counts`` and have the question are scored.
    """
    scored: Counter[str] = Counter()
    exact: Counter[str] = Counter()
    error_sum: Counter[str] = Counter()
    for image, image_counts in counts.items():
        for start, answer in answers.get(image, {}).items():
            value = sum(image_counts.get(name, 0) for name in BENCHMARK_QUESTIONS[start])
            scored[start] += 1
            exact[start] += value == answer
            error_sum[start] += abs(value - answer)
    return {
        start: {
            "images": scored[start],
            "exact": exact[start],
            "exact_rate": exact[start] / scored[start] if scored[start] else None,
            "mean_abs_error": error_sum[start] / scored[start] if scored[start] else None,
        }
        for start in BENCHMARK_QUESTIONS
    }


def format_scores(scores: Mapping[str, Mapping]) -> str:
    """A text table of the result of ``score_counts`` or ``score_against_benchmark``."""
    lines = [f"{'':<32}{'images':>8}{'exact':>8}{'rate':>8}{'mean err':>10}"]
    for name, row in scores.items():
        rate = f"{row['exact_rate']:.1%}" if row["exact_rate"] is not None else "-"
        error = f"{row['mean_abs_error']:.2f}" if row["mean_abs_error"] is not None else "-"
        lines.append(f"{name:<32}{row['images']:>8}{row['exact']:>8}{rate:>8}{error:>10}")
    return "\n".join(lines)
