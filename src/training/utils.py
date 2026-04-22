import os
import yaml
import torch
from pathlib import Path
from dotenv import load_dotenv

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
)

load_dotenv()

# ── config loader ─────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── bnb 4-bit quantization config ────────────────────────────────────────────

def get_bnb_config() -> BitsAndBytesConfig:
    """
    4-bit quantization using NF4 dtype + double quantization.
    reduces Mistral 7B from ~28GB → ~6GB VRAM.
    """
    return BitsAndBytesConfig(
        load_in_4bit               = True,
        bnb_4bit_quant_type        = "nf4",
        bnb_4bit_compute_dtype     = torch.float16,
        bnb_4bit_use_double_quant  = True,
    )


# ── tokenizer loader ──────────────────────────────────────────────────────────

def load_tokenizer(model_name: str) -> AutoTokenizer:
    """
    load tokenizer and set padding token.
    mistral has no pad token by default — we set it to eos token.
    padding side = right for causal LM training.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token           = os.getenv("HF_TOKEN"),
        trust_remote_code = False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"
    return tokenizer


# ── base model loader ─────────────────────────────────────────────────────────

def load_base_model(
    model_name:  str,
    quantize:    bool = True,
    device_map:  str  = "auto",
) -> AutoModelForCausalLM:
    """
    load base model with optional 4-bit quantization.
    quantize=True  → use for training on single GPU (saves VRAM)
    quantize=False → use for inference / merging (needs full precision)
    """
    kwargs = dict(
        pretrained_model_name_or_path = model_name,
        token                         = os.getenv("HF_TOKEN"),
        device_map                    = device_map,
        torch_dtype                   = torch.float16,
        trust_remote_code             = False,
    )

    if quantize:
        kwargs["quantization_config"] = get_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(**kwargs)

    # disable cache during training — not compatible with gradient checkpointing
    model.config.use_cache            = False
    model.config.pretraining_tp       = 1

    return model


# ── lora setup ────────────────────────────────────────────────────────────────

def apply_lora(model: AutoModelForCausalLM, cfg: dict) -> AutoModelForCausalLM:
    """
    wrap model with LoRA adapters using config from config.yaml.
    only the adapter weights are trainable — base model stays frozen.
    """
    lora_cfg = LoraConfig(
        r                = cfg["lora"]["r"],
        lora_alpha       = cfg["lora"]["alpha"],
        lora_dropout     = cfg["lora"]["dropout"],
        target_modules   = cfg["lora"]["target_modules"],
        bias             = cfg["lora"]["bias"],
        task_type        = TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()   # logs how many params are trainable
    return model


# ── checkpoint loader (for DPO / inference) ───────────────────────────────────

def load_peft_model(
    base_model_name: str,
    adapter_path:    str,
    quantize:        bool = True,
) -> AutoModelForCausalLM:
    """
    load a base model then attach saved LoRA adapter weights.
    used when loading the SFT model for DPO training or inference.
    """
    model = load_base_model(base_model_name, quantize=quantize)
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        is_trainable = False,
    )
    return model


# ── merge and save ────────────────────────────────────────────────────────────

def merge_and_save(
    base_model_name: str,
    adapter_path:    str,
    output_path:     str,
) -> None:
    """
    merge LoRA adapter weights into the base model and save full model.
    the merged model can be loaded without PEFT for inference.
    note: run this on CPU or with quantize=False — merging needs full precision.
    """
    print(f"loading base model for merge (no quantization)...")
    model     = load_base_model(base_model_name, quantize=False, device_map="cpu")
    tokenizer = load_tokenizer(base_model_name)

    print(f"loading adapter from {adapter_path} ...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("merging adapter into base model ...")
    model = model.merge_and_unload()

    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"merged model saved to {output_path}")


# ── device info ───────────────────────────────────────────────────────────────

def print_gpu_info() -> None:
    if torch.cuda.is_available():
        print(f"  GPU      : {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free  = (torch.cuda.get_device_properties(0).total_memory
                 - torch.cuda.memory_allocated(0)) / 1e9
        print(f"  VRAM     : {total:.1f} GB total  |  {free:.1f} GB free")
    else:
        print("  GPU      : not available — will use CPU (training will be slow)")


# ── trainable param counter ───────────────────────────────────────────────────

def count_parameters(model) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total"          : total,
        "trainable"      : trainable,
        "trainable_pct"  : 100 * trainable / max(total, 1),
    }


# ── entry point — smoke test ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== utils.py smoke test ===\n")

    # 1. load config
    cfg = load_config()
    print(f"config loaded")
    print(f"  base model : {cfg['model']['base_model']}")
    print(f"  lora rank  : {cfg['lora']['r']}")

    # 2. gpu info
    print("\ngpu info:")
    print_gpu_info()

    # 3. check torch
    print(f"\ntorch version  : {torch.__version__}")
    print(f"cuda available : {torch.cuda.is_available()}")
    print(f"bfloat16 ok    : {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else 'N/A'}")

    # 4. check imports
    print("\nchecking imports ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa
    from peft import LoraConfig, get_peft_model                    # noqa
    print("  transformers : ok")
    print("  peft         : ok")

    print("\n=== smoke test passed — utils.py is ready ===")