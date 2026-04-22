#!/bin/bash
set -e

echo "============================================"
echo "  SFT Training"
echo "  requires: GPU + HF_TOKEN + WANDB_API_KEY"
echo "============================================"

python -c "
import torch
assert torch.cuda.is_available(), 'ERROR: no GPU found. Run on Google Colab T4.'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "starting SFT training ..."
python src/training/sft_train.py --train

echo ""
echo "SFT training complete"
echo "model saved to: models/sft_model/"