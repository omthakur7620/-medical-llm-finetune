import sys
import json
import re
from pathlib import Path
from collections import defaultdict

sys.path.append(str(Path(__file__).resolve().parents[2]))

from rouge_score import rouge_scorer
from bert_score import score as bert_score
from src.training.utils import load_config

# ── helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def save_json(data: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── ROUGE ─────────────────────────────────────────────────────────────────────

def compute_rouge(
    predictions: list[str],
    references:  list[str],
) -> dict[str, float]:
    """
    compute ROUGE-1 and ROUGE-L scores.
    ROUGE-1 = unigram overlap
    ROUGE-L = longest common subsequence
    returns average F1 scores across all pairs.
    """
    scorer  = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    r1_scores, rl_scores = [], []

    for pred, ref in zip(predictions, references):
        if not pred or not ref:
            continue
        scores = scorer.score(ref, pred)
        r1_scores.append(scores["rouge1"].fmeasure)
        rl_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": round(sum(r1_scores) / max(len(r1_scores), 1), 4),
        "rougeL": round(sum(rl_scores) / max(len(rl_scores), 1), 4),
    }


# ── BERTScore ─────────────────────────────────────────────────────────────────

def compute_bertscore(
    predictions: list[str],
    references:  list[str],
    model_type:  str = "distilbert-base-uncased",
) -> dict[str, float]:
    """
    compute BERTScore F1 — semantic similarity using BERT embeddings.
    uses distilbert (small) so it runs fast even on CPU.
    returns average precision, recall, F1.
    """
    # filter empty pairs
    pairs = [(p, r) for p, r in zip(predictions, references) if p and r]
    if not pairs:
        return {"bertscore_p": 0.0, "bertscore_r": 0.0, "bertscore_f1": 0.0}

    preds, refs = zip(*pairs)

    P, R, F = bert_score(
        list(preds),
        list(refs),
        model_type = model_type,
        lang       = "en",
        verbose    = False,
    )

    return {
        "bertscore_p"  : round(P.mean().item(), 4),
        "bertscore_r"  : round(R.mean().item(), 4),
        "bertscore_f1" : round(F.mean().item(), 4),
    }


# ── exact match (for MCQ) ─────────────────────────────────────────────────────

def extract_answer_label(text: str) -> str:
    """
    extract MCQ answer label (A/B/C/D) from a response string.
    handles formats like: 'B.', 'B)', '(B)', 'Answer: B', 'The answer is B'
    """
    text = text.strip().upper()
    patterns = [
        r"\b([A-D])\.",          # B.
        r"\b([A-D])\)",          # B)
        r"\(([A-D])\)",          # (B)
        r"ANSWER[:\s]+([A-D])",  # Answer: B
        r"^([A-D])\b",           # B at start
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return ""


def compute_exact_match(
    predictions: list[str],
    references:  list[str],
) -> dict[str, float]:
    """
    for MCQ records — check if predicted answer label matches reference label.
    ignores records where no label can be extracted.
    """
    correct = 0
    total   = 0

    for pred, ref in zip(predictions, references):
        pred_label = extract_answer_label(pred)
        ref_label  = extract_answer_label(ref)
        if not pred_label or not ref_label:
            continue
        total += 1
        if pred_label == ref_label:
            correct += 1

    accuracy = correct / max(total, 1)
    return {
        "exact_match_accuracy": round(accuracy, 4),
        "exact_match_correct" : correct,
        "exact_match_total"   : total,
    }


# ── compute all metrics for one model ─────────────────────────────────────────

def compute_all_metrics(
    output_path:     str,
    use_bertscore:   bool = True,
) -> dict:
    """
    loads a model output jsonl, computes all metrics, returns results dict.
    """
    records     = load_jsonl(output_path)
    predictions = [r.get("response",  "") for r in records]
    references  = [r.get("reference", "") for r in records]
    model_name  = records[0].get("model", Path(output_path).stem) if records else "unknown"

    print(f"  computing metrics for {model_name} ({len(records)} samples) ...")

    results = {"model": model_name, "n_samples": len(records)}

    # ROUGE
    rouge = compute_rouge(predictions, references)
    results.update(rouge)
    print(f"    ROUGE-1  : {rouge['rouge1']:.4f}")
    print(f"    ROUGE-L  : {rouge['rougeL']:.4f}")

    # BERTScore
    if use_bertscore:
        bscore = compute_bertscore(predictions, references)
        results.update(bscore)
        print(f"    BERTScore F1: {bscore['bertscore_f1']:.4f}")

    # Exact match (MCQ accuracy)
    em = compute_exact_match(predictions, references)
    results.update(em)
    if em["exact_match_total"] > 0:
        print(f"    MCQ accuracy: {em['exact_match_accuracy']:.4f} "
              f"({em['exact_match_correct']}/{em['exact_match_total']})")

    return results


# ── run metrics on all 3 model outputs ────────────────────────────────────────

def run_all_metrics(results_dir: str, use_bertscore: bool = True) -> list[dict]:
    """
    compute metrics for base, sft, dpo model outputs.
    saves individual results and a combined summary.
    """
    results_dir = Path(results_dir)
    output_files = {
        "base_model": results_dir / "base_model_outputs.jsonl",
        "sft_model" : results_dir / "sft_model_outputs.jsonl",
        "dpo_model" : results_dir / "dpo_model_outputs.jsonl",
    }

    all_results = []
    print("\n=== computing metrics ===\n")

    for name, path in output_files.items():
        if not path.exists():
            print(f"  skipping {name} — {path} not found")
            print(f"  run: python src/evaluation/generate.py --run first\n")
            continue

        metrics = compute_all_metrics(str(path), use_bertscore=use_bertscore)
        all_results.append(metrics)

        save_json(metrics, str(results_dir / f"{name}_metrics.json"))
        print()

    # save combined summary
    if all_results:
        save_json(all_results, str(results_dir / "all_metrics_summary.json"))
        print(f"summary saved → {results_dir}/all_metrics_summary.json")

    return all_results


# ── smoke test ────────────────────────────────────────────────────────────────

def smoke_test() -> None:
    print("\n=== metrics.py smoke test ===\n")

    # 1. ROUGE on dummy data
    preds = [
        "The answer is B. Metformin is the first-line treatment for type 2 diabetes.",
        "A. Aspirin is used for pain relief and fever reduction.",
    ]
    refs = [
        "B. Metformin is considered the first-line pharmacological treatment.",
        "A. Aspirin reduces pain, fever, and inflammation.",
    ]

    rouge = compute_rouge(preds, refs)
    assert rouge["rouge1"] > 0
    assert rouge["rougeL"] > 0
    print(f"  ROUGE-1  : {rouge['rouge1']} — ok")
    print(f"  ROUGE-L  : {rouge['rougeL']} — ok")

    # 2. exact match label extraction
    test_cases = [
        ("B. Metformin",            "B"),
        ("The answer is C.",        "C"),
        ("Answer: D",               "D"),
        ("(A) Aspirin",             "A"),
        ("No label here at all",    ""),
    ]
    for text, expected in test_cases:
        got = extract_answer_label(text)
        assert got == expected, f"expected '{expected}' got '{got}' for '{text}'"
    print(f"  label extraction: all {len(test_cases)} cases — ok")

    # 3. exact match accuracy
    em = compute_exact_match(
        predictions = ["B. Metformin", "A. Aspirin", "C. Ibuprofen"],
        references  = ["B. Metformin", "A. Aspirin", "D. Paracetamol"],
    )
    assert em["exact_match_correct"] == 2
    assert em["exact_match_total"]   == 3
    print(f"  exact match: {em['exact_match_correct']}/{em['exact_match_total']} "
          f"= {em['exact_match_accuracy']} — ok")

    # 4. BERTScore (small model, runs on CPU)
    print(f"\n  computing BERTScore (distilbert, may take ~20s on first run) ...")
    bscore = compute_bertscore(preds, refs)
    assert bscore["bertscore_f1"] > 0
    print(f"  BERTScore F1: {bscore['bertscore_f1']} — ok")

    print("\n=== smoke test passed ===")
    print("\nto run full metrics on generated outputs:")
    print("  python src/evaluation/metrics.py --run")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--run" in sys.argv:
        cfg = load_config()
        run_all_metrics(
            results_dir  = cfg["eval"]["results_dir"],
            use_bertscore = True,
        )
    else:
        smoke_test()