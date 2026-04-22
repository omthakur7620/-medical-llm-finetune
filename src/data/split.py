import json
import random
from pathlib import Path

# ── split ratios ──────────────────────────────────────────────────────────────

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10   # must sum to 1.0

RANDOM_SEED = 42

# ── helpers ───────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ── split logic ───────────────────────────────────────────────────────────────

def split_records(
    records: list[dict],
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
    seed:        int   = RANDOM_SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    shuffle and split records into train / val / test.
    returns (train, val, test) lists.
    """
    rng = random.Random(seed)
    data = records.copy()
    rng.shuffle(data)

    n       = len(data)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train = data[:n_train]
    val   = data[n_train : n_train + n_val]
    test  = data[n_train + n_val:]

    return train, val, test


# ── merge + split all sources ─────────────────────────────────────────────────

def split_all() -> None:
    """
    1. load all three formatted datasets
    2. merge them into one pool
    3. shuffle and split 80 / 10 / 10
    4. save to data/processed/
       - sft_train.jsonl
       - sft_val.jsonl
       - test_set.jsonl
    """
    fmt_files = [
        Path("data/processed/formatted/medqa_fmt.jsonl"),
        Path("data/processed/formatted/pubmedqa_fmt.jsonl"),
        Path("data/processed/formatted/medmcqa_fmt.jsonl"),
    ]

    print("\n=== loading formatted datasets ===\n")

    all_records = []
    for fp in fmt_files:
        if not fp.exists():
            print(f"  skipping {fp.name} — not found (run formatter.py first)")
            continue
        recs = _load_jsonl(fp)
        print(f"  {fp.name:30s} → {len(recs):,} records")
        all_records.extend(recs)

    print(f"\n  total merged : {len(all_records):,} records")

    # ── split ─────────────────────────────────────────────────────────────────
    train, val, test = split_records(all_records)

    # ── save ──────────────────────────────────────────────────────────────────
    out_dir = Path("data/processed")

    _save_jsonl(train, out_dir / "sft_train.jsonl")
    _save_jsonl(val,   out_dir / "sft_val.jsonl")
    _save_jsonl(test,  out_dir / "test_set.jsonl")

    # ── report ────────────────────────────────────────────────────────────────
    print(f"\n=== split complete ===")
    print(f"  sft_train.jsonl : {len(train):,} records  ({TRAIN_RATIO*100:.0f}%)")
    print(f"  sft_val.jsonl   : {len(val):,}  records  ({VAL_RATIO*100:.0f}%)")
    print(f"  test_set.jsonl  : {len(test):,}  records  ({TEST_RATIO*100:.0f}%)")
    print(f"\n  saved to → {out_dir}/\n")

    # ── source distribution check ─────────────────────────────────────────────
    print("=== source distribution in train set ===")
    source_counts: dict[str, int] = {}
    for rec in train:
        s = rec.get("source", "unknown")
        source_counts[s] = source_counts.get(s, 0) + 1
    for src, cnt in sorted(source_counts.items()):
        pct = 100 * cnt / max(len(train), 1)
        print(f"  {src:12s}: {cnt:,}  ({pct:.1f}%)")

    # ── leakage check — ensure no test id appears in train ────────────────────
    print("\n=== leakage check ===")
    train_ids = {rec.get("id", "") for rec in train}
    test_ids  = {rec.get("id", "") for rec in test}
    overlap   = train_ids & test_ids
    if overlap:
        print(f"  WARNING: {len(overlap)} overlapping ids found between train and test!")
    else:
        print(f"  no overlap between train and test ids — clean split confirmed")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    split_all()