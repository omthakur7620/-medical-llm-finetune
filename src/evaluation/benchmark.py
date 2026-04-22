import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.training.utils import load_config
from src.evaluation.generate import run_generation
from src.evaluation.metrics import run_all_metrics
from src.evaluation.llm_judge import run_all_judgements
from src.evaluation.compare import generate_report

# ── helpers ───────────────────────────────────────────────────────────────────

def save_json(data: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def print_banner(text: str) -> None:
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}\n")


# ── full benchmark pipeline ───────────────────────────────────────────────────

def run_full_benchmark(
    config_path:   str  = "config.yaml",
    n_samples:     int  = 30,
    use_bertscore: bool = True,
    skip_judge:    bool = False,
) -> dict:
    """
    orchestrates the full evaluation pipeline in order:
    1. generate outputs from all 3 model versions
    2. compute automatic metrics (ROUGE, BERTScore, exact match)
    3. run LLM judge pairwise comparisons
    4. generate markdown comparison report

    returns a summary dict with all results.
    """
    cfg         = load_config(config_path)
    eval_cfg    = cfg["eval"]
    test_path   = eval_cfg["test_path"]
    results_dir = eval_cfg["results_dir"]
    start_time  = time.time()

    print_banner("Medical LLM — Full Benchmark Pipeline")
    print(f"  config      : {config_path}")
    print(f"  test set    : {test_path}")
    print(f"  results dir : {results_dir}")
    print(f"  n_samples   : {n_samples}")
    print(f"  timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # ── step 1: generation ────────────────────────────────────────────────────
    print_banner("Step 1 / 4 — Generating Model Outputs")
    output_paths = run_generation(
        test_path   = test_path,
        results_dir = results_dir,
        n_samples   = n_samples,
    )
    print(f"\n  generated outputs for: {list(output_paths.keys())}")

    # ── step 2: automatic metrics ─────────────────────────────────────────────
    print_banner("Step 2 / 4 — Computing Automatic Metrics")
    metrics_results = run_all_metrics(
        results_dir   = results_dir,
        use_bertscore = use_bertscore,
    )

    # ── step 3: llm judge ─────────────────────────────────────────────────────
    judge_results = []
    if not skip_judge:
        print_banner("Step 3 / 4 — LLM Judge Evaluation")
        judge_results = run_all_judgements(
            results_dir = results_dir,
            n_samples   = n_samples,
        )
    else:
        print_banner("Step 3 / 4 — LLM Judge (skipped)")

    # ── step 4: report ────────────────────────────────────────────────────────
    print_banner("Step 4 / 4 — Generating Comparison Report")
    report = generate_report(results_dir)

    # ── final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    summary = {
        "timestamp"      : datetime.now().isoformat(),
        "n_samples"      : n_samples,
        "elapsed_seconds": round(elapsed, 1),
        "metrics"        : metrics_results,
        "judge"          : judge_results,
    }

    save_json(summary, f"{results_dir}/benchmark_summary.json")

    print_banner("Benchmark Complete")
    print(f"  time elapsed : {elapsed/60:.1f} minutes")
    print(f"  results dir  : {results_dir}/")
    print(f"  report       : {results_dir}/comparison_report.md\n")

    # print key numbers
    if metrics_results:
        print("  key metrics:")
        for m in metrics_results:
            print(f"    {m.get('model','?'):15s} "
                  f"ROUGE-L={m.get('rougeL',0):.3f}  "
                  f"BERTScore={m.get('bertscore_f1',0):.3f}  "
                  f"MCQ={m.get('exact_match_accuracy',0):.3f}")

    if judge_results:
        print("\n  win rates:")
        for j in judge_results:
            print(f"    {j.get('candidate','?'):15s} vs "
                  f"{j.get('baseline','?'):15s} → "
                  f"{j.get('win_rate_pct',0):.1f}% win rate")

    return summary


# ── smoke test ────────────────────────────────────────────────────────────────

def smoke_test() -> None:
    print("\n=== benchmark.py smoke test ===\n")

    cfg = load_config()

    # 1. all required input files exist
    test_path = cfg["eval"]["test_path"]
    assert Path(test_path).exists(), f"missing: {test_path}"
    print(f"  test_path   : {test_path} — ok")

    # 2. results dir creatable
    results_dir = cfg["eval"]["results_dir"]
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    print(f"  results_dir : {results_dir} — ok")

    # 3. all evaluation modules import cleanly
    from src.evaluation.generate   import run_generation    # noqa
    from src.evaluation.metrics    import run_all_metrics   # noqa
    from src.evaluation.llm_judge  import run_all_judgements # noqa
    from src.evaluation.compare    import generate_report   # noqa
    print(f"  imports     : generate, metrics, llm_judge, compare — ok")

    # 4. config has all required eval keys
    eval_cfg = cfg["eval"]
    for key in ["test_path", "results_dir", "n_judge_samples", "batch_size"]:
        assert key in eval_cfg, f"missing key in config.yaml eval section: {key}"
    print(f"  config keys : {list(eval_cfg.keys())} — ok")

    # 5. pipeline order check — verify functions are callable
    assert callable(run_generation)
    assert callable(run_all_metrics)
    assert callable(run_all_judgements)
    assert callable(generate_report)
    print(f"  all pipeline functions callable — ok")

    print("\n=== smoke test passed ===")
    print("\npipeline run commands:")
    print("  full run (generation + metrics + judge + report):")
    print("    python src/evaluation/benchmark.py --run")
    print("\n  skip judge (faster, no API calls for judging):")
    print("    python src/evaluation/benchmark.py --run --skip-judge")
    print("\n  quick run with fewer samples:")
    print("    python src/evaluation/benchmark.py --run --samples 10")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--run" in sys.argv:
        n  = 10
        for arg in sys.argv:
            if arg.startswith("--samples="):
                n = int(arg.split("=")[1])
            elif sys.argv.index(arg) + 1 < len(sys.argv):
                if arg == "--samples":
                    n = int(sys.argv[sys.argv.index(arg) + 1])

        skip_judge = "--skip-judge" in sys.argv

        run_full_benchmark(
            n_samples   = n,
            skip_judge  = skip_judge,
        )
    else:
        smoke_test()