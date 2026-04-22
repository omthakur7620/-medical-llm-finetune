import os
import json
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

RAW_DIR = Path("data/raw")


# ── helpers ───────────────────────────────────────────────────────────────────

def _save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  saved {len(records):,} records → {path}")


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ── MedQA ─────────────────────────────────────────────────────────────────────

def download_medqa(max_samples: int | None = None) -> Path:
    """
    MedQA — USMLE style MCQ questions.
    Using: GBaker/MedQA-USMLE-4-options  (parquet format, no loading script)
    Each record: question, options dict, answer_idx, answer
    """
    save_path = RAW_DIR / "medqa" / "medqa_raw.jsonl"
    if save_path.exists():
        print(f"  medqa already downloaded, skipping.")
        return save_path

    print("downloading MedQA ...")
    ds = load_dataset(
        "GBaker/MedQA-USMLE-4-options",
        token=HF_TOKEN,
    )

    records = []
    for split in ds.keys():
        for row in tqdm(ds[split], desc=f"  {split}"):
            # options is a dict: {"A": "...", "B": "...", "C": "...", "D": "..."}
            options = row.get("options", {})
            choices = list(options.values()) if isinstance(options, dict) else []
            answer_idx = row.get("answer_idx", "")
            answer = options.get(answer_idx, "") if isinstance(options, dict) else row.get("answer", "")

            records.append({
                "source"  : "medqa",
                "split"   : split,
                "id"      : str(row.get("id", len(records))),
                "question": row.get("question", ""),
                "choices" : choices,
                "answer"  : str(answer),
            })
            if max_samples and len(records) >= max_samples:
                break
        if max_samples and len(records) >= max_samples:
            break

    _save_jsonl(records, save_path)
    return save_path


# ── PubMedQA ──────────────────────────────────────────────────────────────────

def download_pubmedqa(max_samples: int | None = None) -> Path:
    """
    PubMedQA — yes/no/maybe questions from PubMed abstracts.
    Using: pubmed_qa pqa_labeled subset (standard parquet format)
    """
    save_path = RAW_DIR / "pubmedqa" / "pubmedqa_raw.jsonl"
    if save_path.exists():
        print(f"  pubmedqa already downloaded, skipping.")
        return save_path

    print("downloading PubMedQA ...")
    ds = load_dataset(
        "pubmed_qa",
        "pqa_labeled",
        token=HF_TOKEN,
    )

    records = []
    for split in ds.keys():
        for row in tqdm(ds[split], desc=f"  {split}"):
            # context is a dict with key "contexts" = list of strings
            ctx = row.get("context", {})
            context_texts = ctx.get("contexts", []) if isinstance(ctx, dict) else []
            context = " ".join(context_texts)

            records.append({
                "source"  : "pubmedqa",
                "split"   : split,
                "id"      : str(row.get("pubid", len(records))),
                "question": row.get("question", ""),
                "context" : context,
                "answer"  : row.get("long_answer", ""),
                "label"   : row.get("final_decision", ""),
            })
            if max_samples and len(records) >= max_samples:
                break
        if max_samples and len(records) >= max_samples:
            break

    _save_jsonl(records, save_path)
    return save_path


# ── MedMCQA ───────────────────────────────────────────────────────────────────

def download_medmcqa(max_samples: int | None = None) -> Path:
    """
    MedMCQA — Indian medical entrance MCQs (AIIMS/NEET style).
    Using: openlifescienceai/medmcqa (parquet format, works fine)
    Each record: question, opa/opb/opc/opd, cop (correct idx 0-3), explanation
    """
    save_path = RAW_DIR / "medmcqa" / "medmcqa_raw.jsonl"
    if save_path.exists():
        print(f"  medmcqa already downloaded, skipping.")
        return save_path

    print("downloading MedMCQA ...")
    ds = load_dataset(
        "openlifescienceai/medmcqa",
        token=HF_TOKEN,
    )

    option_keys = ["opa", "opb", "opc", "opd"]

    records = []
    for split in ["train", "validation"]:
        if split not in ds:
            continue
        for row in tqdm(ds[split], desc=f"  {split}"):
            cop     = row.get("cop", 0)
            choices = [row.get(k, "") for k in option_keys]
            answer  = choices[cop] if isinstance(cop, int) and cop < len(choices) else ""

            records.append({
                "source"     : "medmcqa",
                "split"      : split,
                "id"         : str(row.get("id", len(records))),
                "question"   : row.get("question", ""),
                "choices"    : choices,
                "answer"     : answer,
                "subject"    : row.get("subject_name", ""),
                "topic"      : row.get("topic_name", ""),
                "explanation": row.get("exp", ""),
            })
            if max_samples and len(records) >= max_samples:
                break
        if max_samples and len(records) >= max_samples:
            break

    _save_jsonl(records, save_path)
    return save_path


# ── download all ──────────────────────────────────────────────────────────────

def download_all(max_samples_each: int | None = None) -> dict[str, Path]:
    print("\n=== downloading all medical datasets ===\n")
    paths = {
        "medqa"   : download_medqa(max_samples_each),
        "pubmedqa": download_pubmedqa(max_samples_each),
        "medmcqa" : download_medmcqa(max_samples_each),
    }
    print("\n=== download complete ===")
    for name, p in paths.items():
        count = len(_load_jsonl(p))
        print(f"  {name:12s}: {count:,} records at {p}")
    return paths


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    download_all(max_samples_each=100)