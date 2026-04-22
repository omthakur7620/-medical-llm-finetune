import re
import json
import unicodedata
from pathlib import Path
from tqdm import tqdm

# ── constants ─────────────────────────────────────────────────────────────────

MIN_QUESTION_LEN = 20    # chars — discard anything shorter than this
MAX_QUESTION_LEN = 2048  # chars — USMLE vignettes can be long, allow up to 2048
MIN_ANSWER_LEN   = 3     # chars — discard empty / single-word answers
MAX_ANSWER_LEN   = 2048  # chars — discard runaway answers

# ── text cleaners ─────────────────────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """convert weird unicode chars to their closest ascii equivalent"""
    return unicodedata.normalize("NFKC", text)


def remove_html_tags(text: str) -> str:
    """strip any leftover html tags e.g. <b>, <br/>, &amp;"""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&amp;",  "&",  text)
    text = re.sub(r"&lt;",   "<",  text)
    text = re.sub(r"&gt;",   ">",  text)
    text = re.sub(r"&nbsp;", " ",  text)
    text = re.sub(r"&#\d+;", " ",  text)
    return text


def remove_citations(text: str) -> str:
    """remove inline citation markers like [1], [2,3], (Smith 2020)"""
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)
    text = re.sub(r"\([A-Z][a-z]+(?:\s+et\s+al\.?)?\s*,?\s*\d{4}\)", "", text)
    return text


def normalize_whitespace(text: str) -> str:
    """collapse multiple spaces/newlines into single space, strip edges"""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(text: str) -> str:
    """run all cleaners in order on a single string"""
    if not isinstance(text, str):
        return ""
    text = normalize_unicode(text)
    text = remove_html_tags(text)
    text = remove_citations(text)
    text = normalize_whitespace(text)
    return text


# ── record-level filters ──────────────────────────────────────────────────────

def is_valid_record(rec: dict) -> tuple[bool, str]:
    """
    returns (True, "") if record passes all filters
    returns (False, reason) if it should be discarded
    """
    q = rec.get("question", "")
    a = rec.get("answer", "")

    if not isinstance(q, str) or not q:
        return False, "missing question"

    if not isinstance(a, str) or not a:
        return False, "missing answer"

    if len(q) < MIN_QUESTION_LEN:
        return False, f"question too short ({len(q)} chars)"

    if len(q) > MAX_QUESTION_LEN:
        return False, f"question too long ({len(q)} chars)"

    if len(a) < MIN_ANSWER_LEN:
        return False, f"answer too short ({len(a)} chars)"

    if len(a) > MAX_ANSWER_LEN:
        return False, f"answer too long ({len(a)} chars)"

    # discard if question is mostly numbers (likely a table row leaked in)
    digit_ratio = sum(c.isdigit() for c in q) / max(len(q), 1)
    if digit_ratio > 0.4:
        return False, f"question mostly numeric (ratio={digit_ratio:.2f})"

    return True, ""


# ── per-source cleaners ───────────────────────────────────────────────────────

def clean_medqa_record(rec: dict) -> dict:
    """
    medqa raw format:
      question: str
      choices : list[str]   e.g. ["Aspirin", "Ibuprofen", ...]
      answer  : str         the correct choice text
    """
    rec["question"] = clean_text(rec.get("question", ""))
    rec["answer"]   = clean_text(rec.get("answer", ""))
    rec["choices"]  = [clean_text(c) for c in rec.get("choices", [])]

    # remove empty choices
    rec["choices"] = [c for c in rec["choices"] if c]
    return rec


def clean_pubmedqa_record(rec: dict) -> dict:
    """
    pubmedqa raw format:
      question: str
      context : str   (pubmed abstract — can be long)
      answer  : str   (long answer from abstract)
      label   : str   yes/no/maybe
    """
    rec["question"] = clean_text(rec.get("question", ""))
    rec["answer"]   = clean_text(rec.get("answer",   ""))
    rec["context"]  = clean_text(rec.get("context",  ""))

    # truncate context to 1024 chars to avoid runaway lengths
    if len(rec["context"]) > 1024:
        rec["context"] = rec["context"][:1024] + "..."

    # for pubmedqa use label as fallback answer if long answer is empty
    if not rec["answer"] and rec.get("label"):
        rec["answer"] = rec["label"]

    return rec


def clean_medmcqa_record(rec: dict) -> dict:
    """
    medmcqa raw format:
      question   : str
      choices    : list[str]  [opa, opb, opc, opd]
      answer     : str        correct choice text
      explanation: str        optional explanation
      subject    : str
      topic      : str
    """
    rec["question"]    = clean_text(rec.get("question",    ""))
    rec["answer"]      = clean_text(rec.get("answer",      ""))
    rec["explanation"] = clean_text(rec.get("explanation", ""))
    rec["choices"]     = [clean_text(c) for c in rec.get("choices", [])]
    rec["choices"]     = [c for c in rec["choices"] if c]
    return rec


# ── dispatch ──────────────────────────────────────────────────────────────────

SOURCE_CLEANERS = {
    "medqa"   : clean_medqa_record,
    "pubmedqa": clean_pubmedqa_record,
    "medmcqa" : clean_medmcqa_record,
}


def clean_record(rec: dict) -> dict:
    """route record to the correct source-specific cleaner"""
    source  = rec.get("source", "")
    cleaner = SOURCE_CLEANERS.get(source)
    if cleaner:
        rec = cleaner(rec)
    return rec


# ── main cleaning pipeline ────────────────────────────────────────────────────

def clean_dataset(input_path: str | Path, output_path: str | Path) -> dict:
    """
    reads raw jsonl, cleans every record, filters bad ones,
    writes cleaned jsonl to output_path.

    returns a stats dict with counts for logging.
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total"    : 0,
        "kept"     : 0,
        "discarded": 0,
        "reasons"  : {},
    }

    kept_records = []

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc=f"  cleaning {input_path.name}"):
        line = line.strip()
        if not line:
            continue

        stats["total"] += 1

        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            stats["discarded"] += 1
            stats["reasons"]["json_error"] = stats["reasons"].get("json_error", 0) + 1
            continue

        # clean the record first
        rec = clean_record(rec)

        # then validate
        valid, reason = is_valid_record(rec)
        if not valid:
            stats["discarded"] += 1
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
            continue

        kept_records.append(rec)
        stats["kept"] += 1

    # write output
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in kept_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return stats


def clean_all() -> None:
    """clean all three raw datasets and print a summary report"""

    jobs = [
        ("data/raw/medqa/medqa_raw.jsonl",       "data/raw/medqa/medqa_clean.jsonl"),
        ("data/raw/pubmedqa/pubmedqa_raw.jsonl",  "data/raw/pubmedqa/pubmedqa_clean.jsonl"),
        ("data/raw/medmcqa/medmcqa_raw.jsonl",    "data/raw/medmcqa/medmcqa_clean.jsonl"),
    ]

    print("\n=== cleaning all datasets ===\n")

    total_kept = 0
    total_disc = 0

    for inp, out in jobs:
        inp_path = Path(inp)
        if not inp_path.exists():
            print(f"  skipping {inp} — file not found (run download.py first)")
            continue

        stats = clean_dataset(inp, out)
        total_kept += stats["kept"]
        total_disc += stats["discarded"]

        pct = 100 * stats["kept"] / max(stats["total"], 1)
        print(f"  {inp_path.name}")
        print(f"    total={stats['total']:,}  kept={stats['kept']:,}  "
              f"discarded={stats['discarded']:,}  ({pct:.1f}% kept)")
        if stats["reasons"]:
            for reason, count in sorted(stats["reasons"].items(),
                                        key=lambda x: -x[1]):
                print(f"    discard reason: {reason} × {count}")
        print()

    print(f"=== done — total kept: {total_kept:,} | discarded: {total_disc:,} ===\n")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    clean_all()