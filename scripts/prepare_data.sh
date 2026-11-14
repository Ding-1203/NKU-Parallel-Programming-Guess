#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_SOURCE='/mnt/d/03_Courses/3_2_05_Parallel_programming/Lab5_GPU_programming/NKU-Parallel-Programming-Guess/guessdata/Rockyou-singleLined-full.txt'
SOURCE="${1:-$DEFAULT_SOURCE}"
DESTINATION="${2:-$ROOT_DIR/data/raw/rockyou.txt}"

if [[ ! -f "$SOURCE" ]]; then
    echo "Source dataset not found: $SOURCE" >&2
    exit 1
fi

mkdir -p "$(dirname "$DESTINATION")"
cp --reflink=auto "$SOURCE" "$DESTINATION"

echo "Prepared dataset: $DESTINATION"
wc -l -c "$DESTINATION"
sha256sum "$DESTINATION"
