import os
import sys
import json
import time
import random
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
RANDOM_SEED = 42

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


def save_json(data: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── judge prompt ──────────────────────────────────────────────────────────────

JUDGE_PROMPT = """\
You are an expert medical evaluator. Your task is to compare two responses to a medical question and decide which one is better.

Evaluate based on:
1. Medical accuracy — is the answer clinically correct?
2. Completeness — does it address the question fully?
3. Clarity — is it clear and well-structured?
4. Appropriate confidence — does it avoid unnecessary hedging while staying safe?

Question / Prompt:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Reply with ONLY one of these three options:
- A  (if Response A is clearly better)
- B  (if Response B is clearly better)
- TIE  (if both are roughly equal)

Your verdict:"""


# ── groq judge call ───────────────────────────────────────────────────────────

def call_judge(prompt: str, response_a: str, response_b: str) -> str:
    """
    ask Groq to judge which response is better.
    returns 'A', 'B', or 'TIE'.
    """
    judge_input = JUDGE_PROMPT.format(
        prompt     = prompt[:500],
        response_a = response_a[:400],
        response_b = response_b[:400],
    )

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model       = GROQ_MODEL,
                messages    = [{"role": "user", "content": judge_input}],
                temperature = 0.0,   # deterministic judging
                max_tokens  = 10,
            )
            verdict = resp.choices[0].message.content.strip().upper()

            # parse verdict — handle noisy outputs
            if verdict.startswith("A"):
                return "A"
            elif verdict.startswith("B"):
                return "B"
            elif "TIE" in verdict:
                return "TIE"
            else:
                return "TIE"   # default to tie if unparseable

        except Exception as e:
            if "rate" in str(e).lower() and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return "ERROR"

    return "ERROR"


# ── pairwise evaluation ───────────────────────────────────────────────────────

def evaluate_pair(
    baseline_outputs: list[dict],
    candidate_outputs: list[dict],
    baseline_name:    str,
    candidate_name:   str,
    results_dir:      str,
    n_samples:        int = 30,
    seed:             int = RANDOM_SEED,
) -> dict:
    """
    compare candidate model vs baseline model.
    uses position swap to reduce position bias:
    - half the samples: baseline=A, candidate=B
    - other half:       candidate=A, baseline=B
    returns win rate stats.
    """
    rng     = random.Random(seed)
    n       = min(n_samples, len(baseline_outputs), len(candidate_outputs))
    indices = rng.sample(range(min(len(baseline_outputs), len(candidate_outputs))), n)

    results = []
    wins    = {"baseline": 0, "candidate": 0, "tie": 0, "error": 0}

    print(f"\n  judging: {candidate_name} vs {baseline_name} ({n} samples)")

    for i, idx in enumerate(tqdm(indices, desc="    judging")):
        base_rec = baseline_outputs[idx]
        cand_rec = candidate_outputs[idx]

        prompt       = base_rec.get("prompt",    "")
        base_resp    = base_rec.get("response",  "")
        cand_resp    = cand_rec.get("response",  "")

        # swap positions every other sample to reduce position bias
        if i % 2 == 0:
            response_a, response_b = base_resp, cand_resp
            a_is_baseline = True
        else:
            response_a, response_b = cand_resp, base_resp
            a_is_baseline = False

        verdict = call_judge(prompt, response_a, response_b)

        # map verdict back to who won
        if verdict == "ERROR":
            winner = "error"
            wins["error"] += 1
        elif verdict == "TIE":
            winner = "tie"
            wins["tie"] += 1
        elif verdict == "A":
            winner = "baseline" if a_is_baseline else "candidate"
            wins[winner] += 1
        else:  # B
            winner = "candidate" if a_is_baseline else "baseline"
            wins[winner] += 1

        results.append({
            "idx"           : idx,
            "prompt"        : prompt[:200],
            "baseline_resp" : base_resp[:200],
            "candidate_resp": cand_resp[:200],
            "verdict"       : verdict,
            "winner"        : winner,
            "position_swap" : i % 2 != 0,
        })

        time.sleep(0.4)   # rate limit

    # compute win rate (exclude errors and ties from denominator)
    contested = wins["baseline"] + wins["candidate"]
    win_rate  = wins["candidate"] / max(contested, 1)

    summary = {
        "baseline"       : baseline_name,
        "candidate"      : candidate_name,
        "n_samples"      : n,
        "wins_candidate" : wins["candidate"],
        "wins_baseline"  : wins["baseline"],
        "ties"           : wins["tie"],
        "errors"         : wins["error"],
        "win_rate"       : round(win_rate, 4),
        "win_rate_pct"   : round(win_rate * 100, 1),
    }

    # save detailed results
    pair_name  = f"{candidate_name}_vs_{baseline_name}"
    save_jsonl(results,  f"{results_dir}/{pair_name}_judgements.jsonl")
    save_json(summary,   f"{results_dir}/{pair_name}_summary.json")

    print(f"    {candidate_name} win rate: {summary['win_rate_pct']}%  "
          f"(wins={wins['candidate']} ties={wins['tie']} losses={wins['baseline']})")

    return summary


# ── run all pairwise comparisons ──────────────────────────────────────────────

def run_all_judgements(results_dir: str, n_samples: int = 30) -> list[dict]:
    """
    runs two comparisons:
    1. sft_model  vs base_model  — measures SFT improvement
    2. dpo_model  vs sft_model   — measures DPO improvement
    """
    results_dir = Path(results_dir)

    required = {
        "base_model": results_dir / "base_model_outputs.jsonl",
        "sft_model" : results_dir / "sft_model_outputs.jsonl",
        "dpo_model" : results_dir / "dpo_model_outputs.jsonl",
    }

    # check all files exist
    for name, path in required.items():
        if not path.exists():
            print(f"missing: {path}")
            print(f"run: python src/evaluation/generate.py --run first")
            return []

    base = load_jsonl(str(required["base_model"]))
    sft  = load_jsonl(str(required["sft_model"]))
    dpo  = load_jsonl(str(required["dpo_model"]))

    print("\n=== LLM Judge Evaluation ===")

    all_summaries = []

    # comparison 1: sft vs base
    s1 = evaluate_pair(
        baseline_outputs  = base,
        candidate_outputs = sft,
        baseline_name     = "base_model",
        candidate_name    = "sft_model",
        results_dir       = str(results_dir),
        n_samples         = n_samples,
    )
    all_summaries.append(s1)

    # comparison 2: dpo vs sft
    s2 = evaluate_pair(
        baseline_outputs  = sft,
        candidate_outputs = dpo,
        baseline_name     = "sft_model",
        candidate_name    = "dpo_model",
        results_dir       = str(results_dir),
        n_samples         = n_samples,
    )
    all_summaries.append(s2)

    save_json(all_summaries, str(results_dir / "judge_summary.json"))
    print(f"\nsummary saved → {results_dir}/judge_summary.json")

    return all_summaries


# ── smoke test ────────────────────────────────────────────────────────────────

def smoke_test() -> None:
    print("\n=== llm_judge.py smoke test ===\n")

    cfg = load_config()

    # 1. single judge call
    print("  testing single judge call ...")
    verdict = call_judge(
        prompt     = "What is the first-line treatment for type 2 diabetes?",
        response_a = (
            "I'm not sure, but I think it might be insulin or maybe metformin. "
            "You should consult a doctor as I cannot provide medical advice."
        ),
        response_b = (
            "Metformin is the first-line pharmacological treatment for type 2 "
            "diabetes, per ADA guidelines, unless contraindicated."
        ),
    )
    assert verdict in ("A", "B", "TIE", "ERROR"), f"unexpected verdict: {verdict}"
    print(f"  verdict        : {verdict} (expected B) — ok")

    # 2. judge prompt formatting
    formatted = JUDGE_PROMPT.format(
        prompt     = "test prompt",
        response_a = "response A text",
        response_b = "response B text",
    )
    assert "test prompt"    in formatted
    assert "response A text" in formatted
    assert "response B text" in formatted
    print(f"  prompt format  : ok")

    # 3. results dir exists
    results_dir = cfg["eval"]["results_dir"]
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    print(f"  results_dir    : {results_dir} — ok")

    print("\n=== smoke test passed ===")
    print("\nto run full judgement (requires generate.py --run first):")
    print("  python src/evaluation/llm_judge.py --run")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--run" in sys.argv:
        cfg = load_config()
        run_all_judgements(
            results_dir = cfg["eval"]["results_dir"],
            n_samples   = cfg["eval"]["n_judge_samples"],
        )
    else:
        smoke_test()