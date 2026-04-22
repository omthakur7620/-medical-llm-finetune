import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.training.utils import load_config

# ── helpers ───────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict | list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def save_text(content: str, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ── markdown table builder ────────────────────────────────────────────────────

def build_metrics_table(metrics_list: list[dict]) -> str:
    """
    build a markdown comparison table from list of metric dicts.
    one row per model, columns = metric names.
    """
    if not metrics_list:
        return "_no metrics found_\n"

    # define column order
    columns = [
        ("Model",                "model"),
        ("Samples",              "n_samples"),
        ("ROUGE-1",              "rouge1"),
        ("ROUGE-L",              "rougeL"),
        ("BERTScore F1",         "bertscore_f1"),
        ("MCQ Accuracy",         "exact_match_accuracy"),
        ("MCQ Correct/Total",    None),   # computed
    ]

    # header
    headers = [c[0] for c in columns]
    sep     = ["-" * max(len(h), 8) for h in headers]

    rows = []
    for m in metrics_list:
        row = []
        for col_name, key in columns:
            if key is None:
                # MCQ correct/total
                c = m.get("exact_match_correct", "-")
                t = m.get("exact_match_total",   "-")
                row.append(f"{c}/{t}")
            elif key == "model":
                row.append(m.get(key, "-"))
            elif key == "n_samples":
                row.append(str(m.get(key, "-")))
            else:
                val = m.get(key)
                row.append(f"{val:.4f}" if isinstance(val, float) else "-")
        rows.append(row)

    # format table
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(sep)     + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def build_winrate_table(judge_summaries: list[dict]) -> str:
    """build markdown table for LLM judge win rates"""
    if not judge_summaries:
        return "_no judge results found_\n"

    headers = ["Comparison", "Win Rate", "Wins", "Losses", "Ties"]
    sep     = ["-" * 20, "-" * 10, "-" * 6, "-" * 6, "-" * 6]

    rows = []
    for s in judge_summaries:
        comparison = f"{s.get('candidate','?')} vs {s.get('baseline','?')}"
        win_rate   = f"{s.get('win_rate_pct', 0):.1f}%"
        wins       = str(s.get("wins_candidate", "-"))
        losses     = str(s.get("wins_baseline",  "-"))
        ties       = str(s.get("ties",           "-"))
        rows.append([comparison, win_rate, wins, losses, ties])

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(sep)     + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


# ── sample responses section ──────────────────────────────────────────────────

def build_sample_section(results_dir: str, n: int = 3) -> str:
    """
    pick n sample questions and show base vs sft vs dpo responses side by side.
    """
    base_path = Path(results_dir) / "base_model_outputs.jsonl"
    sft_path  = Path(results_dir) / "sft_model_outputs.jsonl"
    dpo_path  = Path(results_dir) / "dpo_model_outputs.jsonl"

    if not all(p.exists() for p in [base_path, sft_path, dpo_path]):
        return "_run generate.py --run to populate sample responses_\n"

    base_recs = load_jsonl(str(base_path))[:n]
    sft_recs  = load_jsonl(str(sft_path))[:n]
    dpo_recs  = load_jsonl(str(dpo_path))[:n]

    lines = []
    for i, (b, s, d) in enumerate(zip(base_recs, sft_recs, dpo_recs)):
        lines.append(f"### Sample {i+1}")
        lines.append(f"**Prompt:** {b.get('prompt','')[:200]}...\n")
        lines.append(f"**Reference answer:** {b.get('reference','')[:150]}\n")
        lines.append(f"**Base model:** {b.get('response','')[:200]}\n")
        lines.append(f"**SFT model:** {s.get('response','')[:200]}\n")
        lines.append(f"**DPO model:** {d.get('response','')[:200]}\n")
        lines.append("---\n")

    return "\n".join(lines)


# ── full report generator ─────────────────────────────────────────────────────

def generate_report(results_dir: str) -> str:
    """
    loads all metric and judge files from results_dir,
    assembles a full markdown comparison report,
    saves it to results_dir/comparison_report.md,
    returns the report string.
    """
    results_dir = Path(results_dir)
    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M")

    # load metrics
    metrics_list = []
    for name in ["base_model", "sft_model", "dpo_model"]:
        p = results_dir / f"{name}_metrics.json"
        if p.exists():
            metrics_list.append(load_json(str(p)))

    # load judge summaries
    judge_path     = results_dir / "judge_summary.json"
    judge_summaries = load_json(str(judge_path)) if judge_path.exists() else []

    # ── assemble report ───────────────────────────────────────────────────────
    report = []
    report.append(f"# Medical LLM Fine-tuning — Evaluation Report")
    report.append(f"_Generated: {timestamp}_\n")

    report.append("## Project Summary")
    report.append(
        "Fine-tuned `Mistral-7B-v0.1` on medical QA data (MedQA + PubMedQA + MedMCQA) "
        "using LoRA (r=16) and 4-bit quantization. "
        "Aligned with human preference data using DPO (β=0.1).\n"
    )

    report.append("## Automatic Metrics")
    report.append("Higher is better for all metrics.\n")
    report.append(build_metrics_table(metrics_list))

    report.append("## LLM Judge Win Rates")
    report.append(
        "Win rate = % of contested comparisons (ties excluded) where the candidate "
        "model was preferred by `llama-3.3-70b-versatile` as judge.\n"
    )
    report.append(build_winrate_table(judge_summaries))

    report.append("## Training Setup")
    report.append("| Parameter | Value |")
    report.append("| --------- | ----- |")
    report.append("| Base model | mistralai/Mistral-7B-v0.1 |")
    report.append("| LoRA rank | 16 |")
    report.append("| LoRA alpha | 32 |")
    report.append("| SFT epochs | 3 |")
    report.append("| SFT learning rate | 2e-4 |")
    report.append("| DPO beta | 0.1 |")
    report.append("| DPO epochs | 1 |")
    report.append("| Quantization | 4-bit NF4 |")
    report.append("| Training data | 237 samples (MedQA + PubMedQA + MedMCQA) |")
    report.append("| DPO pairs | 50 chosen/rejected pairs |\n")

    report.append("## Sample Responses")
    report.append(build_sample_section(str(results_dir), n=3))

    report_str = "\n".join(report)

    out_path = results_dir / "comparison_report.md"
    save_text(report_str, str(out_path))
    print(f"  report saved → {out_path}")

    return report_str


# ── smoke test ────────────────────────────────────────────────────────────────

def smoke_test() -> None:
    print("\n=== compare.py smoke test ===\n")

    # 1. metrics table with dummy data
    dummy_metrics = [
        {
            "model": "base_model", "n_samples": 31,
            "rouge1": 0.21, "rougeL": 0.18,
            "bertscore_f1": 0.71,
            "exact_match_accuracy": 0.42,
            "exact_match_correct": 13, "exact_match_total": 31,
        },
        {
            "model": "sft_model", "n_samples": 31,
            "rouge1": 0.38, "rougeL": 0.33,
            "bertscore_f1": 0.82,
            "exact_match_accuracy": 0.61,
            "exact_match_correct": 19, "exact_match_total": 31,
        },
        {
            "model": "dpo_model", "n_samples": 31,
            "rouge1": 0.44, "rougeL": 0.39,
            "bertscore_f1": 0.86,
            "exact_match_accuracy": 0.71,
            "exact_match_correct": 22, "exact_match_total": 31,
        },
    ]

    table = build_metrics_table(dummy_metrics)
    assert "base_model"  in table
    assert "sft_model"   in table
    assert "dpo_model"   in table
    assert "ROUGE-1"     in table
    assert "BERTScore"   in table
    print("  metrics table: ok")
    print(table)

    # 2. win rate table with dummy data
    dummy_judge = [
        {
            "baseline": "base_model", "candidate": "sft_model",
            "win_rate_pct": 61.0,
            "wins_candidate": 19, "wins_baseline": 12, "ties": 0,
        },
        {
            "baseline": "sft_model", "candidate": "dpo_model",
            "win_rate_pct": 72.0,
            "wins_candidate": 18, "wins_baseline":  7, "ties": 5,
        },
    ]

    wr_table = build_winrate_table(dummy_judge)
    assert "sft_model vs base_model" in wr_table
    assert "dpo_model vs sft_model"  in wr_table
    assert "61.0%"                   in wr_table
    print("  win rate table: ok")
    print(wr_table)

    # 3. full report generation with dummy files
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        # write dummy metric files
        for m in dummy_metrics:
            p = Path(tmp) / f"{m['model']}_metrics.json"
            with open(p, "w") as f:
                json.dump(m, f)

        # write dummy judge summary
        with open(Path(tmp) / "judge_summary.json", "w") as f:
            json.dump(dummy_judge, f)

        report = generate_report(tmp)
        assert "# Medical LLM Fine-tuning" in report
        assert "ROUGE"                     in report
        assert "Win Rate"                  in report
        print("  report generation: ok")

    print("\n=== smoke test passed ===")
    print("\nto generate full report (after running generate + metrics + judge):")
    print("  python src/evaluation/compare.py --run")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--run" in sys.argv:
        cfg = load_config()
        report = generate_report(cfg["eval"]["results_dir"])
        print("\n" + "=" * 60)
        print(report[:1000])
        print("..." if len(report) > 1000 else "")
    else:
        smoke_test()