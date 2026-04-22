import os
import sys
import json
import torch
from pathlib import Path

# add project root to path so src imports work
sys.path.append(str(Path(__file__).resolve().parents[2]))

from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

from src.training.utils import (
    load_config,
    load_base_model,
    load_tokenizer,
    apply_lora,
    print_gpu_info,
    count_parameters,
)
from src.training.callbacks import (
    SampleGenerationCallback,
    GPUMemoryCallback,
    EarlyStoppingCallback,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def to_hf_dataset(records: list[dict]) -> Dataset:
    return Dataset.from_list(records)


# ── main training function ────────────────────────────────────────────────────

def train(config_path: str = "config.yaml") -> None:
    cfg       = load_config(config_path)
    model_cfg = cfg["model"]
    sft_cfg   = cfg["sft"]
    wb_cfg    = cfg["wandb"]

    print("\n=== SFT Training ===\n")
    print_gpu_info()

    # ── init wandb ────────────────────────────────────────────────────────────
    try:
        import wandb
        wandb.init(
            project = wb_cfg["project"],
            entity  = wb_cfg.get("entity"),
            name    = "sft-run",
            config  = cfg,
        )
        use_wandb = True
    except Exception as e:
        print(f"  wandb init failed: {e} — continuing without wandb")
        use_wandb = False

    # ── load tokenizer ────────────────────────────────────────────────────────
    print(f"\nloading tokenizer: {model_cfg['base_model']}")
    tokenizer = load_tokenizer(model_cfg["base_model"])
    print(f"  vocab size : {tokenizer.vocab_size:,}")
    print(f"  pad token  : {tokenizer.pad_token}")

    # ── load model + lora ─────────────────────────────────────────────────────
    print(f"\nloading base model: {model_cfg['base_model']}")
    use_quantization = torch.cuda.is_available()
    model = load_base_model(
        model_cfg["base_model"],
        quantize   = use_quantization,
        device_map = "auto" if torch.cuda.is_available() else "cpu",
    )

    print("\napplying LoRA adapters ...")
    model = apply_lora(model, cfg)
    params = count_parameters(model)
    print(f"  trainable: {params['trainable']:,}  ({params['trainable_pct']:.2f}%)")

    # ── datasets ──────────────────────────────────────────────────────────────
    print("\nloading datasets ...")
    train_ds = to_hf_dataset(load_jsonl(sft_cfg["dataset_path"]))
    val_ds   = to_hf_dataset(load_jsonl(sft_cfg["val_path"]))
    print(f"  train: {len(train_ds):,}  val: {len(val_ds):,}")

    # ── training arguments ────────────────────────────────────────────────────
    output_dir = sft_cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir                  = output_dir,
        num_train_epochs            = sft_cfg["epochs"],
        per_device_train_batch_size = sft_cfg["batch_size"],
        per_device_eval_batch_size  = sft_cfg["batch_size"],
        gradient_accumulation_steps = sft_cfg["grad_accumulation"],
        learning_rate               = sft_cfg["learning_rate"],
        warmup_ratio                = sft_cfg["warmup_ratio"],
        lr_scheduler_type           = sft_cfg["lr_scheduler"],
        save_steps                  = sft_cfg["save_steps"],
        eval_steps                  = sft_cfg["eval_steps"],
        logging_steps               = sft_cfg["logging_steps"],
        eval_strategy               = "steps",
        save_strategy               = "steps",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        fp16                        = torch.cuda.is_available(),
        bf16                        = False,
        report_to                   = "wandb" if use_wandb else "none",
        run_name                    = "sft-run",
    )

    # ── sample prompt for generation callback ─────────────────────────────────
    sample_prompt = (
        "### Instruction:\n"
        "You are a medical expert. Answer the following multiple choice "
        "medical question by selecting the single best answer.\n\n"
        "### Input:\n"
        "A 35-year-old male presents with chest pain radiating to the left arm. "
        "ECG shows ST elevation. What is the most likely diagnosis?\n\n"
        "A. Unstable angina\nB. STEMI\nC. Aortic dissection\nD. Pericarditis\n\n"
        "### Response:\n"
    )

    # ── callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        GPUMemoryCallback(every_n_steps=10),
        EarlyStoppingCallback(patience=3),
        SampleGenerationCallback(
            tokenizer      = tokenizer,
            sample_prompt  = sample_prompt,
            every_n_steps  = sft_cfg["eval_steps"],
            max_new_tokens = 128,
        ),
    ]

    # ── trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model              = model,
        args               = training_args,
        train_dataset      = train_ds,
        eval_dataset       = val_ds,
        tokenizer          = tokenizer,
        dataset_text_field = "text",
        max_seq_length     = model_cfg["max_seq_length"],
        packing            = False,
        callbacks          = callbacks,
    )

    print("\n=== starting training ===\n")
    trainer.train()

    print(f"\nsaving model to {output_dir} ...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    if use_wandb:
        import wandb
        wandb.finish()

    print("\n=== SFT training complete ===")
    print(f"  model saved to: {output_dir}")


# ── smoke test ────────────────────────────────────────────────────────────────

def smoke_test(config_path: str = "config.yaml") -> None:
    print("\n=== sft_train.py smoke test ===\n")

    cfg     = load_config(config_path)
    sft_cfg = cfg["sft"]

    # 1. dataset files exist and have correct fields
    for key in ["dataset_path", "val_path"]:
        p = Path(sft_cfg[key])
        assert p.exists(), f"missing: {p} — run split.py first"
        records = load_jsonl(str(p))
        assert len(records) > 0,    f"empty file: {p}"
        assert "text" in records[0], f"missing 'text' field in {p}"
        print(f"  {key:15s}: {len(records):,} records — ok")

    # 2. output dir creatable
    Path(sft_cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    print(f"  output_dir     : {sft_cfg['output_dir']} — ok")

    # 3. hf dataset conversion
    records = load_jsonl(sft_cfg["dataset_path"])
    ds      = to_hf_dataset(records)
    assert len(ds) == len(records)
    assert "text" in ds.column_names
    print(f"  hf dataset     : {len(ds):,} rows — ok")

    # 4. trl version check
    import trl
    print(f"  trl version    : {trl.__version__} — ok")

    # 5. training args instantiate without error
    Path("tmp_test").mkdir(exist_ok=True)
    args = TrainingArguments(
        output_dir       = "tmp_test",
        num_train_epochs = 1,
        eval_strategy    = "steps",
        eval_steps       = 10,
        save_steps       = 10,
    )
    print(f"  TrainingArguments: ok")
    import shutil
    shutil.rmtree("tmp_test", ignore_errors=True)

    print("\n=== smoke test passed ===")
    print("\nnote: run actual training on Google Colab (free T4 GPU)")
    print("      command: python src/training/sft_train.py --train")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--train" in sys.argv:
        train()
    else:
        smoke_test()