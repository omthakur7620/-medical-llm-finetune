#!/bin/bash
set -e

echo "============================================"
echo "  DPO Alignment Training"
echo "  requires: GPU + models/sft_model/ checkpoint"
echo "============================================"

python -c "
from pathlib import Path
sft_path = Path('models/sft_model')
assert sft_path.exists(), 'ERROR: models/sft_model not found. Run run_sft.sh first.'
print(f'SFT checkpoint found at {sft_path}')
"

python -c "
import torch
assert torch.cuda.is_available(), 'ERROR: no GPU found. Run on Google Colab T4.'
print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "starting DPO training ..."
python src/training/dpo_train.py --train

echo ""
echo "DPO training complete"
echo "model saved to: models/dpo_model/"