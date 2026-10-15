#!/usr/bin/env bash
# Run from any directory after activating the project's Python environment.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
STANDARD_CHECKPOINT="${1:-output/models/standard_final.pth}"
ROBUST_CHECKPOINT="${2:-output/models/robust_best.pth}"

for checkpoint in "$STANDARD_CHECKPOINT" "$ROBUST_CHECKPOINT"; do
    if [[ ! -f "$checkpoint" ]]; then
        echo "Checkpoint not found: $checkpoint" >&2
        echo "Train both models using the commands in README.md first." >&2
        exit 1
    fi
done

"$PYTHON_BIN" -m scripts.analyze_robustness \
    --checkpoint "$STANDARD_CHECKPOINT" --save-results output/analysis_standard.json
"$PYTHON_BIN" -m scripts.analyze_robustness \
    --checkpoint "$ROBUST_CHECKPOINT" --save-results output/analysis_robust.json
"$PYTHON_BIN" -m scripts.visualize_attacks \
    --checkpoint "$STANDARD_CHECKPOINT" --output-name adversarial_examples_standard.png
"$PYTHON_BIN" -m scripts.visualize_attacks \
    --checkpoint "$ROBUST_CHECKPOINT" --output-name adversarial_examples_robust.png
"$PYTHON_BIN" -m scripts.compare_results
