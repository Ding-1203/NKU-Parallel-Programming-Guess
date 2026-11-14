#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${1:-$ROOT_DIR/results/baseline-v1-$STAMP}"

"$ROOT_DIR/scripts/bench" run --output "$RUN_DIR"
"$ROOT_DIR/scripts/bench" profile --output "$RUN_DIR"
"$ROOT_DIR/scripts/bench" analyze --input "$RUN_DIR"

echo "Baseline experiment: $RUN_DIR"
