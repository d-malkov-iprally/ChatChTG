"""Command line entry point: chatchtg serve | detect-demo | ingest | eval | benchmark."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from chatchtg.config import Settings
from chatchtg.context.composer import composer_id
from chatchtg.context.registry import COMPOSERS, get_composer
from chatchtg.documents.library import RASTER_SUFFIXES, UnsupportedDocumentError
from chatchtg.evals.benchmark import build_benchmark
from chatchtg.evals.runner import UnknownPriceError, run_eval
from chatchtg.wiring import build_answer_service, build_judge, build_library


def serve(_: argparse.Namespace) -> int:
    app = Path(__file__).parent / "ui" / "app.py"
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])


def detect_demo(_: argparse.Namespace) -> int:
    app = Path(__file__).parent / "ui" / "detect_app.py"
    # The demo shows files of this machine, so it is served to this machine only.
    command = [sys.executable, "-m", "streamlit", "run", "--server.address=127.0.0.1", str(app)]
    return subprocess.call(command)


def ingest(args: argparse.Namespace) -> int:
    """Register every raster drawing of a dataset folder, without copying it."""
    dataset_dir: Path = args.dataset_dir.resolve()
    drawings_dir = dataset_dir / "data"
    if not drawings_dir.is_dir():
        print(f"No data/ folder in {dataset_dir}", file=sys.stderr)
        return 2
    library = build_library(Settings.from_env())
    before = library.count()
    seen = skipped = 0
    for path in sorted(drawings_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in RASTER_SUFFIXES:
            continue
        attributes = {"original_path": path.relative_to(dataset_dir).as_posix()}
        parts = path.relative_to(drawings_dir).parts
        if len(parts) > 1:
            # Only a file inside a source folder has a source and a licence file.
            attributes["source"] = parts[0]
            attributes["licence_file"] = f"data/{parts[0]}/LICENSE.md"
        try:
            library.register_file(path, attributes=attributes)
            seen += 1
        except UnsupportedDocumentError:
            skipped += 1
    added = library.count() - before
    print(f"Raster files: {seen}. New documents: {added}. Skipped (not PNG/JPEG content): {skipped}.")
    return 0


def evaluate(args: argparse.Namespace) -> int:
    dataset_dir: Path = args.dataset_dir.resolve()
    if not (dataset_dir / args.questions).is_file():
        print(f"No {args.questions} in {dataset_dir}", file=sys.stderr)
        return 2
    if args.resume and args.out is None:
        # Without --out the run gets a new file, and nothing would be resumed.
        print("--resume needs --out: the results file of the run to continue", file=sys.stderr)
        return 2
    settings = Settings.from_env()
    try:
        composer = get_composer(args.composer or settings.composer)
        library = build_library(settings)
        service = build_answer_service(settings, library, composer=composer.name)
    except ValueError as error:  # a CHATCHTG_* setting with an unknown value
        print(error, file=sys.stderr)
        return 2
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = args.out or settings.state_dir / "eval-runs" / f"{stamp}-{composer_id(composer)}.jsonl"
    judge = build_judge(settings) if args.judge else None
    try:
        run = run_eval(
            dataset_dir,
            library,
            service,
            out_path,
            limit=args.limit,
            questions_file=args.questions,
            sample=args.sample,
            seed=args.seed,
            max_cost=args.max_cost,
            resume=args.resume,
            judge=judge,
            evidence=args.evidence,
        )
    except UnknownPriceError as error:
        print(f"{error} Answers so far: {out_path}", file=sys.stderr)
        return 1
    summary = run.summary

    print(f"Composer: {composer_id(composer)}. LLM provider: {settings.llm_provider}.")
    header = f"{'type':<10}{'total':>8}{'scored':>8}{'correct':>9}{'accuracy':>10}"
    if judge is not None:
        # "accuracy" is the rule's score. The judge columns come from the judge's reading.
        print(f"Judge: {judge.model}.")
        header += f"{'judged':>8}{'judge ok':>10}{'judge acc':>11}{'disagree':>10}"
    print(header)
    for question_type, counts in sorted(summary.items()):
        accuracy = f"{counts['correct'] / counts['scored']:.1%}" if counts["scored"] else "-"
        line = (
            f"{question_type:<10}{counts['total']:>8}{counts['scored']:>8}"
            f"{counts['correct']:>9}{accuracy:>10}"
        )
        if judge is not None:
            judged = counts["judged"]
            judge_accuracy = f"{counts['judge_correct'] / judged:.1%}" if judged else "-"
            line += (
                f"{judged:>8}{counts['judge_correct']:>10}{judge_accuracy:>11}"
                f"{counts['disagree']:>10}"
            )
        print(line)
    answered = sum(counts["total"] for counts in summary.values())
    print(f"Questions answered: {answered}. Cost: ${run.cost_usd:.2f}.")
    if run.skipped:
        print(f"Questions skipped because the drawing is missing or not readable: {run.skipped}.")
    if run.stopped_for_cost:
        print(f"Stopped early: the next question could pass the ${args.max_cost:.2f} limit.")
    print(f"Results: {out_path}")
    return 0


def benchmark(args: argparse.Namespace) -> int:
    """Build the floor-plan benchmark from a downloaded dataset."""
    dataset_dir: Path = args.dataset_dir.resolve()
    if not (dataset_dir / "questions.jsonl").is_file():
        print(f"No questions.jsonl in {dataset_dir}", file=sys.stderr)
        return 2
    out_dir: Path = args.out or Settings.from_env().data_root / "floorplans"
    report = build_benchmark(dataset_dir, out_dir, subset_size=args.subset_size, seed=args.seed)

    print(f"Questions: {report['questions']}. Images: {report['images']}. Subset: {report['subset']}.")
    for source, counts in report["by_source"].items():
        print(f"  {source}: {counts['questions']} questions on {counts['images']} images")
    types = ", ".join(f"{count} {question_type}" for question_type, count in report["by_type"].items())
    print(f"Question types: {types}.")
    print(f"Questions left out because the drawing has no raster file: {report['dropped']}.")
    print(f"Images copied: {report['copied']}. Benchmark: {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chatchtg")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("serve", help="start the Streamlit app").set_defaults(run=serve)

    commands.add_parser(
        "detect-demo", help="start the open-vocabulary detection demo (needs the detect extra)"
    ).set_defaults(run=detect_demo)

    ingest_parser = commands.add_parser("ingest", help="register a dataset's raster drawings")
    ingest_parser.add_argument("dataset_dir", type=Path, help="for example data/preview")
    ingest_parser.set_defaults(run=ingest)

    eval_parser = commands.add_parser("eval", help="answer and score the dataset questions")
    eval_parser.add_argument("dataset_dir", type=Path, help="for example data/preview")
    eval_parser.add_argument("--composer", choices=sorted(COMPOSERS))
    eval_parser.add_argument("--limit", type=int, help="stop after this many questions")
    eval_parser.add_argument("--out", type=Path, help="results file (default: var/eval-runs/)")
    eval_parser.add_argument(
        "--questions", default="questions.jsonl", help="question file inside the dataset folder"
    )
    eval_parser.add_argument(
        "--sample", type=int, help="answer this many questions, balanced over source and type"
    )
    eval_parser.add_argument("--seed", type=int, default=0, help="seed of the --sample pick")
    eval_parser.add_argument(
        "--max-cost", type=float, help="stop before the cost passes this many US dollars"
    )
    eval_parser.add_argument(
        "--resume", action="store_true", help="continue the run in --out; keep its answers"
    )
    eval_parser.add_argument(
        "--judge",
        action="store_true",
        help="also have a small model read each answer (one more LLM call per scored answer)",
    )
    eval_parser.add_argument(
        "--evidence",
        action="store_true",
        help="also write the method and the evidence of each answer to its result row "
        "(with --resume, the rows of the earlier run stay as they are)",
    )
    eval_parser.set_defaults(run=evaluate)

    benchmark_parser = commands.add_parser("benchmark", help="build the floor-plan benchmark")
    benchmark_parser.add_argument("dataset_dir", type=Path, help="for example data/full")
    benchmark_parser.add_argument("--out", type=Path, help="output folder (default: data/floorplans)")
    benchmark_parser.add_argument("--subset-size", type=int, default=500, help="questions in the subset")
    benchmark_parser.add_argument("--seed", type=int, default=0, help="seed of the subset sample")
    benchmark_parser.set_defaults(run=benchmark)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
