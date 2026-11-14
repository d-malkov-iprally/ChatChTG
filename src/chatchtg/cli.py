"""Command line entry point: chatchtg serve | ingest | eval."""

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
from chatchtg.evals.runner import run_eval
from chatchtg.wiring import build_answer_service, build_library


def serve(_: argparse.Namespace) -> int:
    app = Path(__file__).parent / "ui" / "app.py"
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])


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
    if not (dataset_dir / "questions.jsonl").is_file():
        print(f"No questions.jsonl in {dataset_dir}", file=sys.stderr)
        return 2
    settings = Settings.from_env()
    composer = get_composer(args.composer or settings.composer)
    library = build_library(settings)
    service = build_answer_service(settings, library, composer=composer.name)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = args.out or settings.state_dir / "eval-runs" / f"{stamp}-{composer_id(composer)}.jsonl"
    summary = run_eval(dataset_dir, library, service, out_path, limit=args.limit)

    print(f"Composer: {composer_id(composer)}. LLM provider: {settings.llm_provider}.")
    print(f"{'type':<10}{'total':>8}{'scored':>8}{'correct':>9}{'accuracy':>10}")
    for question_type, counts in sorted(summary.items()):
        accuracy = f"{counts['correct'] / counts['scored']:.1%}" if counts["scored"] else "-"
        print(
            f"{question_type:<10}{counts['total']:>8}{counts['scored']:>8}"
            f"{counts['correct']:>9}{accuracy:>10}"
        )
    print(f"Results: {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chatchtg")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("serve", help="start the Streamlit app").set_defaults(run=serve)

    ingest_parser = commands.add_parser("ingest", help="register a dataset's raster drawings")
    ingest_parser.add_argument("dataset_dir", type=Path, help="for example data/preview")
    ingest_parser.set_defaults(run=ingest)

    eval_parser = commands.add_parser("eval", help="answer and score the dataset questions")
    eval_parser.add_argument("dataset_dir", type=Path, help="for example data/preview")
    eval_parser.add_argument("--composer", choices=sorted(COMPOSERS))
    eval_parser.add_argument("--limit", type=int, help="stop after this many questions")
    eval_parser.add_argument("--out", type=Path, help="results file (default: var/eval-runs/)")
    eval_parser.set_defaults(run=evaluate)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
