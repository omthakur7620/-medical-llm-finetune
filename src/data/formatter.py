import json
from pathlib import Path
from tqdm import tqdm

# ── Alpaca instruction template ───────────────────────────────────────────────
# This is the exact format Mistral/Llama expect during SFT.
# Every record gets converted into this structure.

INSTRUCTION_TEMPLATE = """\
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


# ── per-source formatters ─────────────────────────────────────────────────────

def format_medqa(rec: dict) -> dict | None:
    """
    medqa: MCQ with 4 choices — format as a multiple choice question.
    instruction = task description
    input       = question + labelled choices
    output      = correct answer with label
    """
    question = rec.get("question", "").strip()
    choices  = rec.get("choices", [])
    answer   = rec.get("answer",  "").strip()

    if not question or not answer:
        return None

    # label choices A, B, C, D
    labels       = ["A", "B", "C", "D", "E"]
    labeled      = [f"{labels[i]}. {c}" for i, c in enumerate(choices) if c]
    choices_text = "\n".join(labeled)

    # find which label matches the answer
    answer_label = ""
    for i, c in enumerate(choices):
        if c.strip().lower() == answer.lower():
            answer_label = labels[i]
            break

    output = f"{answer_label}. {answer}" if answer_label else answer

    instruction = (
        "You are a medical expert. Answer the following multiple choice "
        "medical question by selecting the single best answer."
    )
    inp = f"{question}\n\n{choices_text}" if choices_text else question

    return {
        "source"     : rec.get("source", "medqa"),
        "id"         : rec.get("id", ""),
        "instruction": instruction,
        "input"      : inp,
        "output"     : output,
        "text"       : INSTRUCTION_TEMPLATE.format(
                           instruction=instruction,
                           input=inp,
                           output=output,
                       ),
    }


def format_pubmedqa(rec: dict) -> dict | None:
    """
    pubmedqa: question + abstract context → long answer + yes/no/maybe label.
    instruction = task description
    input       = question + relevant abstract context
    output      = long answer (or label if no long answer)
    """
    question = rec.get("question", "").strip()
    context  = rec.get("context",  "").strip()
    answer   = rec.get("answer",   "").strip()
    label    = rec.get("label",    "").strip()

    if not question:
        return None

    # use long answer if available, otherwise fall back to yes/no/maybe label
    output = answer if answer else label
    if not output:
        return None

    instruction = (
        "You are a biomedical research expert. Based on the provided "
        "PubMed abstract context, answer the medical question accurately "
        "and concisely."
    )
    inp = f"Question: {question}\n\nContext: {context}" if context else question

    return {
        "source"     : rec.get("source", "pubmedqa"),
        "id"         : rec.get("id", ""),
        "instruction": instruction,
        "input"      : inp,
        "output"     : output,
        "text"       : INSTRUCTION_TEMPLATE.format(
                           instruction=instruction,
                           input=inp,
                           output=output,
                       ),
    }


def format_medmcqa(rec: dict) -> dict | None:
    """
    medmcqa: MCQ with 4 choices + optional explanation.
    If explanation exists we append it to the output — richer supervision signal.
    """
    question    = rec.get("question",    "").strip()
    choices     = rec.get("choices",     [])
    answer      = rec.get("answer",      "").strip()
    explanation = rec.get("explanation", "").strip()
    subject     = rec.get("subject",     "").strip()

    if not question or not answer:
        return None

    labels       = ["A", "B", "C", "D"]
    labeled      = [f"{labels[i]}. {c}" for i, c in enumerate(choices) if c]
    choices_text = "\n".join(labeled)

    answer_label = ""
    for i, c in enumerate(choices):
        if c.strip().lower() == answer.lower():
            answer_label = labels[i]
            break

    answer_line = f"{answer_label}. {answer}" if answer_label else answer
    output      = f"{answer_line}\n\nExplanation: {explanation}" if explanation else answer_line

    subject_hint = f" Focus area: {subject}." if subject else ""
    instruction  = (
        f"You are a medical expert.{subject_hint} Answer the following "
        "multiple choice medical question by selecting the single best answer."
    )
    inp = f"{question}\n\n{choices_text}" if choices_text else question

    return {
        "source"     : rec.get("source", "medmcqa"),
        "id"         : rec.get("id", ""),
        "instruction": instruction,
        "input"      : inp,
        "output"     : output,
        "text"       : INSTRUCTION_TEMPLATE.format(
                           instruction=instruction,
                           input=inp,
                           output=output,
                       ),
    }


# ── dispatch ──────────────────────────────────────────────────────────────────

SOURCE_FORMATTERS = {
    "medqa"   : format_medqa,
    "pubmedqa": format_pubmedqa,
    "medmcqa" : format_medmcqa,
}


def format_record(rec: dict) -> dict | None:
    """route a cleaned record to its source-specific formatter"""
    formatter = SOURCE_FORMATTERS.get(rec.get("source", ""))
    if not formatter:
        return None
    return formatter(rec)


# ── main formatting pipeline ──────────────────────────────────────────────────

def format_dataset(input_path: str | Path, output_path: str | Path) -> dict:
    """
    reads a cleaned jsonl, formats every record into Alpaca instruction format,
    writes to output_path.
    returns stats dict.
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "formatted": 0, "skipped": 0}
    formatted_records = []

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc=f"  formatting {input_path.name}"):
        line = line.strip()
        if not line:
            continue
        stats["total"] += 1

        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            stats["skipped"] += 1
            continue

        formatted = format_record(rec)
        if formatted is None:
            stats["skipped"] += 1
            continue

        formatted_records.append(formatted)
        stats["formatted"] += 1

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in formatted_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return stats


def format_all() -> None:
    """format all three cleaned datasets and save to data/processed/formatted/"""

    jobs = [
        ("data/raw/medqa/medqa_clean.jsonl",      "data/processed/formatted/medqa_fmt.jsonl"),
        ("data/raw/pubmedqa/pubmedqa_clean.jsonl", "data/processed/formatted/pubmedqa_fmt.jsonl"),
        ("data/raw/medmcqa/medmcqa_clean.jsonl",   "data/processed/formatted/medmcqa_fmt.jsonl"),
    ]

    print("\n=== formatting all datasets ===\n")

    for inp, out in jobs:
        if not Path(inp).exists():
            print(f"  skipping {inp} — not found (run clean.py first)")
            continue

        stats = format_dataset(inp, out)
        pct   = 100 * stats["formatted"] / max(stats["total"], 1)
        print(f"  {Path(inp).name}")
        print(f"    total={stats['total']}  formatted={stats['formatted']}  "
              f"skipped={stats['skipped']}  ({pct:.1f}% formatted)\n")

    print("=== formatting done ===\n")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    format_all()

    # print one example from each source so you can visually verify
    print("=== sample formatted records ===\n")
    for f in [
        "data/processed/formatted/medqa_fmt.jsonl",
        "data/processed/formatted/pubmedqa_fmt.jsonl",
        "data/processed/formatted/medmcqa_fmt.jsonl",
    ]:
        if not Path(f).exists():
            continue
        with open(f, encoding="utf-8") as fp:
            rec = json.loads(fp.readline())
        print(f"--- {Path(f).name} ---")
        print(rec["text"][:400])
        print("...\n")