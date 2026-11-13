#!/usr/bin/env bash
# Download the Junction construction-drawings dataset into data/.
#
# Usage:
#   scripts/download_data.sh preview          # ~480 MB zip  -> data/preview
#   scripts/download_data.sh full             # ~9.8 GB zip  -> data/full
#   scripts/download_data.sh all              # both
#   scripts/download_data.sh full --force     # re-download and re-extract
#
# With no dataset argument, only the preview is downloaded.
#
# Result: data/<preview|full>/{README.md,questions.jsonl,data/<source>/...}
# Paths in questions.jsonl are relative to that folder.
set -euo pipefail

BASE_URL="${BASE_URL:-https://fwjydgxpttajiqpfmqtj.supabase.co/storage/v1/object/public/stoa-junction-2026}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="$repo_root/data"

archive_for() {
  case "$1" in
    preview) echo "junction-dataset-preview.zip" ;;
    full)    echo "junction-dataset.zip" ;;
  esac
}

download() {
  local name="$1" force="$2"
  local archive_name url archive target tmp
  archive_name="$(archive_for "$name")"
  url="$BASE_URL/$archive_name"
  archive="$data_dir/$archive_name"
  target="$data_dir/$name"

  if [[ $force -eq 1 ]]; then
    rm -rf "$target" "$archive"
  fi

  if [[ -f "$target/questions.jsonl" ]]; then
    echo "[$name] already downloaded: $target (use --force to redo)"
    return
  fi

  echo "[$name] downloading $url"
  # -C - resumes a partial download if the script was interrupted.
  curl --fail --location --retry 5 --retry-all-errors -C - --output "$archive" "$url"

  echo "[$name] extracting"
  tmp="$(mktemp -d "$data_dir/.extract.XXXXXX")"
  unzip -q "$archive" -d "$tmp"
  rm -rf "$tmp/__MACOSX"

  # The archive holds a single top-level folder; fall back to the tmp dir itself if not.
  local entries=("$tmp"/*)
  rm -rf "$target"
  if [[ ${#entries[@]} -eq 1 && -d "${entries[0]}" ]]; then
    mv "${entries[0]}" "$target"
    rm -rf "$tmp"
  else
    mv "$tmp" "$target"
  fi

  rm -f "$archive"

  echo "[$name] done: $target"
  echo "[$name]   drawings:  $(find "$target/data" -type f ! -name 'LICENSE.md' ! -name '.DS_Store' | wc -l | tr -d ' ')"
  echo "[$name]   questions: $(wc -l < "$target/questions.jsonl" | tr -d ' ')"
}

force=0
datasets=()
for arg in "$@"; do
  case "$arg" in
    --force)      force=1 ;;
    preview|full) datasets+=("$arg") ;;
    all)          datasets+=(preview full) ;;
    -h|--help)    sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)            echo "Unknown argument: $arg (expected preview, full, all or --force)" >&2; exit 2 ;;
  esac
done
[[ ${#datasets[@]} -eq 0 ]] && datasets=(preview)

mkdir -p "$data_dir"
# Clean up extraction leftovers from an interrupted run.
rm -rf "$data_dir"/.extract.*

for name in "${datasets[@]}"; do
  download "$name" "$force"
done
