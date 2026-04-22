import os
import sys
import json
import torch
from pathlib import Path

# add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from datasets import Dataset
from transformers import TrainingArguments
from trl import DPOTrainer

from src.training.utils import (
    load_config,
    load_base_model,
    load_tokenizer,
    apply_lora,
    load_peft_model,
    print_gpu_info,
    count_parameters,
)
from src.training.callbacks import (
    DPORewardCallback,
    GPUMemoryCallback,
    EarlyStoppingCallback,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def to_dpo_dataset(records: list[dict]) -> Dataset:
    """
    DPOTrainer expects exactly three fields: prompt, chosen, rejected.
    filter out any records missing these fields.
    """
    clean = []
    for rec in records:
        if rec.get("prompt") and rec.get("chosen") and rec.get("rejected"):
            clean.append({
                "prompt"  : rec["prompt"],
                "chosen"  : rec["chosen"],
                "rejected": rec["rejected"],
            })
    return Dataset.from_list(clean)


# ── main training function ────────────────────────────────────────────────────

def train(config_path: str = "config.yaml") -> None:
    cfg       = load_config(config_path)
    model_cfg = cfg["model"]
    dpo_cfg   = cfg["dpo"]
    wb_cfg    = cfg["wandb"]

    print("\n=== DPO Training ===\n")
    print_gpu_info()

    # ── init wandb ────────────────────────────────────────────────────────────
    try:
        import wandb
        wandb.init(
            project = wb_cfg["project"],
            entity  = wb_cfg.get("entity"),
            name    = "dpo-run",
            config  = cfg,
        )
        use_wandb = True
    except Exception as e:
        print(f"  wandb init failed: {e} — continuing without wandb")
        use_wandb = False

    # ── load tokenizer ────────────────────────────────────────────────────────
    print(f"\nloading tokenizer: {model_cfg['base_model']}")
    tokenizer = load_tokenizer(model_cfg["base_model"])

    # ── load SFT model as policy (trainable) ──────────────────────────────────
    # DPO needs two models:
    # 1. policy model  — starts from SFT checkpoint, gets updated
    # 2. reference model — frozen SFT checkpoint, used to compute KL penalty
    print(f"\nloading SFT model as policy: {dpo_cfg['sft_model_path']}")
    use_quantization = torch.cuda.is_available()

    policy_model = load_peft_model(
        base_model_name = model_cfg["base_model"],
        adapter_path    = dpo_cfg["sft_model_path"],
        quantize        = use_quantization,
    )
    # make policy model trainable by adding a fresh LoRA on top
    policy_model = apply_lora(policy_model, cfg)
    params = count_parameters(policy_model)
    print(f"  trainable: {params['trainable']:,}  ({params['trainable_pct']:.2f}%)")

    # ── load reference model (frozen) ─────────────────────────────────────────
    print(f"\nloading reference model (frozen): {dpo_cfg['sft_model_path']}")
    ref_model = load_peft_model(
        base_model_name = model_cfg["base_model"],
        adapter_path    = dpo_cfg["sft_model_path"],
        quantize        = use_quantization,
    )
    # freeze all reference model params
    for param in ref_model.parameters():
        param.requires_grad = False

    # ── datasets ──────────────────────────────────────────────────────────────
    print("\nloading DPO datasets ...")
    train_ds = to_dpo_dataset(load_jsonl(dpo_cfg["dataset_path"]))
    val_ds   = to_dpo_dataset(load_jsonl(dpo_cfg["val_path"]))
    print(f"  train: {len(train_ds):,}  val: {len(val_ds):,}")

    # ── training arguments ────────────────────────────────────────────────────
    output_dir = dpo_cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir                  = output_dir,
        num_train_epochs            = dpo_cfg["epochs"],
        per_device_train_batch_size = dpo_cfg["batch_size"],
        per_device_eval_batch_size  = dpo_cfg["batch_size"],
        gradient_accumulation_steps = dpo_cfg["grad_accumulation"],
        learning_rate               = dpo_cfg["learning_rate"],
        eval_strategy               = "steps",
        eval_steps                  = 10,
        save_strategy               = "steps",
        save_steps                  = 10,
        logging_steps               = 5,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        fp16                        = torch.cuda.is_available(),
        bf16                        = False,
        report_to                   = "wandb" if use_wandb else "none",
        run_name                    = "dpo-run",
        remove_unused_columns       = False,
    )

    # ── callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        DPORewardCallback(),
        GPUMemoryCallback(every_n_steps=5),
        EarlyStoppingCallback(patience=3),
    ]

    # ── trainer ───────────────────────────────────────────────────────────────
    trainer = DPOTrainer(
        model           = policy_model,
        ref_model       = ref_model,
        args            = training_args,
        beta            = dpo_cfg["beta"],
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        tokenizer       = tokenizer,
        max_length      = dpo_cfg["max_length"],
        max_prompt_length = dpo_cfg["max_prompt_length"],
        callbacks       = callbacks,
    )

    print("\n=== starting DPO training ===\n")
    trainer.train()

    print(f"\nsaving DPO model to {output_dir} ...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    if use_wandb:
        import wandb
        wandb.finish()

    print("\n=== DPO training complete ===")
    print(f"  model saved to: {output_dir}")


# ── smoke test ────────────────────────────────────────────────────────────────

def smoke_test(config_path: str = "config.yaml") -> None:
    print("\n=== dpo_train.py smoke test ===\n")

    cfg     = load_config(config_path)
    dpo_cfg = cfg["dpo"]

    # 1. dpo dataset files exist
    for key in ["dataset_path", "val_path"]:
        p = Path(dpo_cfg[key])
        assert p.exists(), f"missing: {p} — run dpo_builder.py first"
        records = load_jsonl(str(p))
        assert len(records) > 0, f"empty file: {p}"
        # check required DPO fields
        for field in ["prompt", "chosen", "rejected"]:
            assert field in records[0], f"missing '{field}' in {p}"
        print(f"  {key:15s}: {len(records):,} records — ok")

    # 2. dpo dataset conversion
    records = load_jsonl(dpo_cfg["dataset_path"])
    ds      = to_dpo_dataset(records)
    assert len(ds) > 0
    assert set(ds.column_names) == {"prompt", "chosen", "rejected"}
    print(f"  dpo dataset    : {len(ds):,} rows, "
          f"columns={ds.column_names} — ok")

    # 3. output dir creatable
    Path(dpo_cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    print(f"  output_dir     : {dpo_cfg['output_dir']} — ok")

    # 4. beta value in range
    beta = dpo_cfg["beta"]
    assert 0 < beta < 1, f"beta should be between 0 and 1, got {beta}"
    print(f"  beta           : {beta} — ok")

    # 5. verify DPOTrainer importable
    from trl import DPOTrainer   # noqa
    print(f"  DPOTrainer     : ok")

    # 6. sample pair — visually verify chosen vs rejected quality
    sample = records[0]
    print(f"\n  sample pair:")
    print(f"    prompt   : {sample['prompt'][:80]}...")
    print(f"    chosen   : {sample['chosen'][:80]}...")
    print(f"    rejected : {sample['rejected'][:80]}...")

    print("\n=== smoke test passed ===")
    print("\nnote: run actual DPO training on Google Colab after SFT is complete")
    print("      command: python src/training/dpo_train.py --train")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--train" in sys.argv:
        train()
    else:
        smoke_test()