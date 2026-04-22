#!/bin/bash
set -e

HF_USERNAME=${HF_USERNAME:-"your-hf-username"}
REPO_NAME=${REPO_NAME:-"medical-llm-mistral7b"}

echo "============================================"
echo "  Push Model to HuggingFace Hub"
echo "  repo: $HF_USERNAME/$REPO_NAME"
echo "============================================"

echo "merging LoRA adapters into base model ..."
python - <<EOF
import sys
sys.path.append(".")
from src.training.utils import load_config, merge_and_save

cfg = load_config()
merge_and_save(
    base_model_name = cfg["model"]["base_model"],
    adapter_path    = "models/dpo_model",
    output_path     = "models/merged",
)
print("merge complete — models/merged/ ready")
EOF

echo ""
echo "pushing to HuggingFace Hub ..."
python - <<EOF
import os
from huggingface_hub import HfApi

api      = HfApi(token=os.getenv("HF_TOKEN"))
username = os.getenv("HF_USERNAME", "$HF_USERNAME")
repo     = f"{username}/$REPO_NAME"

api.create_repo(repo_id=repo, exist_ok=True, private=False)
api.upload_folder(
    folder_path = "models/merged",
    repo_id     = repo,
    repo_type   = "model",
)
print(f"model pushed to: https://huggingface.co/{repo}")
EOF

echo ""
echo "push complete"
echo "model at: https://huggingface.co/$HF_USERNAME/$REPO_NAME"