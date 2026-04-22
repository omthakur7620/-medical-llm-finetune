#!/bin/bash
set -e

SAMPLES=${1:-30}

echo "============================================"
echo "  Evaluation Pipeline"
echo "  samples: $SAMPLES"
echo "============================================"

echo ""
echo "running full benchmark ..."
python src/evaluation/benchmark.py --run --samples $SAMPLES

echo ""
echo "evaluation complete"
echo "report: evals/results/comparison_report.md"