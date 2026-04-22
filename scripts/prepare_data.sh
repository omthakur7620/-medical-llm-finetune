#!/bin/bash
set -e

echo "============================================"
echo "  Step 1/5 — Downloading datasets"
echo "============================================"
python src/data/download.py

echo "============================================"
echo "  Step 2/5 — Cleaning datasets"
echo "============================================"
python src/data/clean.py

echo "============================================"
echo "  Step 3/5 — Formatting datasets"
echo "============================================"
python src/data/formatter.py

echo "============================================"
echo "  Step 4/5 — Splitting datasets"
echo "============================================"
python src/data/split.py

echo "============================================"
echo "  Step 5/5 — Building DPO pairs"
echo "============================================"
python src/data/dpo_builder.py

echo ""
echo "data pipeline complete"
echo "sft_train.jsonl / sft_val.jsonl / test_set.jsonl / dpo_train.jsonl ready"