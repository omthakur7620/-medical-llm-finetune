import os
import json
import random
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_MODEL  = "llama-3.3-70b-versatile"
RANDOM_SEED = 42
MAX_RETRIES = 3
RETRY_DELAY = 5   # seconds between retries on rate limit

# ── groq client ───────────────────────────────────────────────────────────────

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── helpers ───────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ── response generators ───────────────────────────────────────────────────────

def _call_groq(system: str, user: str, temperature: float = 0.7) -> str | None:
    """call groq api with retry on rate limit"""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system",  "content": system},
                    {"role": "user",    "content": user},
                ],
                temperature=temperature,
                max_tokens=512,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e).lower()
            if "rate" in err and attempt < MAX_RETRIES - 1:
                print(f"    rate limit hit, waiting {RETRY_DELAY}s ...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"    groq call failed: {e}")
                return None
    return None


def generate_chosen(rec: dict) -> str:
    """
    generate a high-quality 'chosen' response using groq.
    system prompt pushes the model to be a domain expert.
    """
    system = (
        "You are a highly experienced medical doctor and educator. "
        "When answering medical questions, be accurate, concise, and confident. "
        "Always provide the correct answer with a brief clinical explanation. "
        "Never hedge unnecessarily. Use proper medical terminology."
    )
    user = f"{rec['instruction']}\n\n{rec['input']}"
    return _call_groq(system, user, temperature=0.3)   # low temp = more focused


def generate_rejected(rec: dict) -> str:
    """
    generate a low-quality 'rejected' response using groq.
    system prompt pushes the model to produce typical bad LLM behaviours:
    - vague / overly cautious
    - hallucinated details
    - incorrect answer
    """
    system = (
        "You are a general assistant with limited medical knowledge. "
        "When answering medical questions, be vague and overly cautious. "
        "Add unnecessary disclaimers. Sometimes give an incorrect answer "
        "but sound confident about it. Avoid using proper medical terminology."
    )
    user = f"{rec['instruction']}\n\n{rec['input']}"
    return _call_groq(system, user, temperature=1.0)   # high temp = more random


# ── dpo pair builder ──────────────────────────────────────────────────────────

def build_dpo_pair(rec: dict) -> dict | None:
    """
    for a single formatted record, generate chosen + rejected responses
    and return a DPO-format dict.

    DPO format:
    {
        "prompt"  : the instruction + input (what the model sees),
        "chosen"  : the better response,
        "rejected": the worse response,
        "source"  : original dataset source,
        "id"      : record id,
    }
    """
    prompt = f"{rec['instruction']}\n\n{rec['input']}"

    chosen   = generate_chosen(rec)
    rejected = generate_rejected(rec)

    if not chosen or not rejected:
        return None

    # sanity check — if both responses are identical something went wrong
    if chosen.strip() == rejected.strip():
        return None

    return {
        "source"  : rec.get("source", ""),
        "id"      : rec.get("id", ""),
        "prompt"  : prompt,
        "chosen"  : chosen,
        "rejected": rejected,
    }


# ── main pipeline ─────────────────────────────────────────────────────────────

def build_dpo_dataset(
    input_path:  str | Path,
    output_path: str | Path,
    n_samples:   int = 50,
    seed:        int = RANDOM_SEED,
) -> None:
    """
    reads sft_train.jsonl, samples n_samples records,
    generates chosen/rejected pairs via groq,
    saves dpo_train.jsonl.

    n_samples=50 is enough for a demo project.
    for a real project use 500-1000.
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)

    records = _load_jsonl(input_path)

    # sample n records randomly
    rng     = random.Random(seed)
    sampled = rng.sample(records, min(n_samples, len(records)))

    print(f"\n=== building DPO dataset ===")
    print(f"  input       : {input_path}  ({len(records):,} records)")
    print(f"  sampling    : {len(sampled)} records")
    print(f"  model       : {GROQ_MODEL}")
    print(f"  output      : {output_path}\n")

    dpo_pairs  = []
    skipped    = 0

    for rec in tqdm(sampled, desc="  generating pairs"):
        pair = build_dpo_pair(rec)
        if pair is None:
            skipped += 1
            continue
        dpo_pairs.append(pair)
        # small sleep to avoid hammering groq rate limits
        time.sleep(0.5)

    _save_jsonl(dpo_pairs, output_path)

    print(f"\n=== DPO dataset built ===")
    print(f"  pairs generated : {len(dpo_pairs)}")
    print(f"  skipped         : {skipped}")
    print(f"  saved to        : {output_path}\n")


def build_all() -> None:
    build_dpo_dataset(
        input_path  = "data/processed/sft_train.jsonl",
        output_path = "data/processed/dpo_train.jsonl",
        n_samples   = 50,    # increase to 500 for full training run
    )

    # use sft_val as source for dpo_val (small, just for monitoring)
    build_dpo_dataset(
        input_path  = "data/processed/sft_val.jsonl",
        output_path = "data/processed/dpo_val.jsonl",
        n_samples   = 10,
    )


# ── quick inspect ─────────────────────────────────────────────────────────────

def inspect_dpo_file(path: str | Path, n: int = 2) -> None:
    """print n dpo pairs so you can visually verify quality"""
    records = _load_jsonl(Path(path))
    print(f"\n=== inspecting {path} ({len(records)} pairs) ===\n")
    for rec in records[:n]:
        print(f"SOURCE   : {rec['source']}")
        print(f"PROMPT   : {rec['prompt'][:120]}...")
        print(f"CHOSEN   : {rec['chosen'][:200]}...")
        print(f"REJECTED : {rec['rejected'][:200]}...")
        print("-" * 60)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_all()
    inspect_dpo_file("data/processed/dpo_train.jsonl", n=2)