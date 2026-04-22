import os
import sys
import json
import time
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from groq import Groq
from src.training.utils import load_config

load_dotenv()

GROQ_MODEL  = "llama-3.3-70b-versatile"
MAX_RETRIES = 3
RETRY_DELAY = 5

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(records: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ── groq inference ────────────────────────────────────────────────────────────

def generate_groq(
    prompt:        str,
    system:        str = "You are a helpful medical assistant.",
    temperature:   float = 0.1,
    max_tokens:    int   = 256,
) -> str:
    """generate a response using Groq API with retry logic"""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model       = GROQ_MODEL,
                messages    = [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                temperature = temperature,
                max_tokens  = max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "rate" in str(e).lower() and attempt < MAX_RETRIES - 1:
                print(f"  rate limit, retrying in {RETRY_DELAY}s ...")
                time.sleep(RETRY_DELAY)
            else:
                return f"[ERROR: {e}]"
    return "[ERROR: max retries exceeded]"


# ── model simulators ──────────────────────────────────────────────────────────
# NOTE: in a real run on Colab you would load actual model checkpoints.
# On local machine (no GPU) we simulate 3 model "versions" using Groq
# with different system prompts to approximate what each training stage produces.
# This lets you run evaluation locally without downloading 7B weights.

def simulate_base_model(prompt: str) -> str:
    """
    simulates base Mistral 7B — general, not domain-aligned.
    uses a generic system prompt with no medical expertise.
    """
    system = (
        "You are a general-purpose AI assistant. "
        "Answer questions to the best of your ability."
    )
    return generate_groq(prompt, system=system, temperature=0.7)


def simulate_sft_model(prompt: str) -> str:
    """
    simulates SFT fine-tuned model — domain-aware, follows instruction format.
    uses a medical expert system prompt.
    """
    system = (
        "You are a medical doctor with deep clinical knowledge. "
        "Answer medical questions accurately and concisely using proper terminology. "
        "Follow the instruction format exactly."
    )
    return generate_groq(prompt, system=system, temperature=0.3)


def simulate_dpo_model(prompt: str) -> str:
    """
    simulates DPO-aligned model — confident, expert-level, no unnecessary hedging.
    uses the strongest expert framing.
    """
    system = (
        "You are a senior medical specialist and educator. "
        "Give precise, confident answers with brief clinical reasoning. "
        "Never add unnecessary disclaimers. Use proper medical terminology. "
        "Be direct and educational."
    )
    return generate_groq(prompt, system=system, temperature=0.1)


# ── real model inference (used on Colab after training) ───────────────────────

def generate_from_checkpoint(
    model,
    tokenizer,
    prompt:        str,
    max_new_tokens: int = 256,
) -> str:
    """
    generate from an actual loaded HuggingFace model.
    used on Colab when real checkpoints are available.
    """
    inputs = tokenizer(
        prompt,
        return_tensors = "pt",
        truncation     = True,
        max_length     = 1024,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            do_sample      = False,
            temperature    = 1.0,
            pad_token_id   = tokenizer.pad_token_id,
        )

    # decode only the newly generated part
    gen_ids  = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return response.strip()


# ── main generation pipeline ──────────────────────────────────────────────────

def run_generation(
    test_path:    str,
    results_dir:  str,
    n_samples:    int  = 30,
    use_simulate: bool = True,   # True = use Groq simulation, False = real models
) -> dict[str, str]:
    """
    runs inference for all 3 model versions on the test set.
    saves outputs to results_dir.
    returns dict of model_name → output_path.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    records = load_jsonl(test_path)[:n_samples]
    print(f"\n=== generating outputs for {len(records)} test samples ===\n")

    model_runners = {
        "base_model": simulate_base_model,
        "sft_model" : simulate_sft_model,
        "dpo_model" : simulate_dpo_model,
    }

    output_paths = {}

    for model_name, runner in model_runners.items():
        print(f"running {model_name} ...")
        outputs = []

        for rec in tqdm(records, desc=f"  {model_name}"):
            prompt   = rec.get("input", rec.get("text", ""))[:800]
            response = runner(prompt)
            time.sleep(0.3)   # gentle rate limiting

            outputs.append({
                "id"           : rec.get("id", ""),
                "source"       : rec.get("source", ""),
                "prompt"       : prompt,
                "reference"    : rec.get("output", ""),
                "model"        : model_name,
                "response"     : response,
            })

        out_path = str(Path(results_dir) / f"{model_name}_outputs.jsonl")
        save_jsonl(outputs, out_path)
        output_paths[model_name] = out_path
        print(f"  saved {len(outputs)} outputs → {out_path}\n")

    return output_paths


# ── smoke test ────────────────────────────────────────────────────────────────

def smoke_test() -> None:
    print("\n=== generate.py smoke test ===\n")

    cfg = load_config()

    # 1. test set exists
    test_path = cfg["eval"]["test_path"]
    assert Path(test_path).exists(), f"missing: {test_path} — run split.py first"
    records = load_jsonl(test_path)
    assert len(records) > 0
    print(f"  test set       : {len(records)} records — ok")

    # 2. results dir creatable
    results_dir = cfg["eval"]["results_dir"]
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    print(f"  results_dir    : {results_dir} — ok")

    # 3. groq client works — single test call
    print(f"  testing groq API call ...")
    response = generate_groq(
        prompt      = "What is the most common cause of community-acquired pneumonia?",
        temperature = 0.1,
        max_tokens  = 64,
    )
    assert len(response) > 0 and not response.startswith("[ERROR")
    print(f"  groq response  : {response[:80]}... — ok")

    # 4. simulate one record through all 3 runners
    print(f"\n  testing all 3 model simulators ...")
    test_prompt = "What is the first-line treatment for hypertension?"
    for name, runner in [
        ("base_model", simulate_base_model),
        ("sft_model",  simulate_sft_model),
        ("dpo_model",  simulate_dpo_model),
    ]:
        out = runner(test_prompt)
        assert len(out) > 0
        print(f"  {name:12s}: {out[:60]}... — ok")
        time.sleep(0.5)

    print("\n=== smoke test passed ===")
    print("\nto run full generation:")
    print("  python src/evaluation/generate.py --run")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--run" in sys.argv:
        cfg = load_config()
        run_generation(
            test_path   = cfg["eval"]["test_path"],
            results_dir = cfg["eval"]["results_dir"],
            n_samples   = cfg["eval"]["n_judge_samples"],
        )
    else:
        smoke_test()