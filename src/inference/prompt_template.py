import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

# ── Alpaca template — must match exactly what was used during SFT ─────────────

ALPACA_TEMPLATE = """\
### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

# default system instruction used at inference time
DEFAULT_INSTRUCTION = (
    "You are a highly experienced medical doctor and educator. "
    "Answer the following medical question accurately and concisely. "
    "Use proper medical terminology and provide brief clinical reasoning."
)

# ── template builders ─────────────────────────────────────────────────────────

def build_prompt(
    user_input:  str,
    instruction: str = DEFAULT_INSTRUCTION,
) -> str:
    """
    formats a raw user question into the Alpaca prompt template.
    this must exactly match the format used during SFT training.
    """
    return ALPACA_TEMPLATE.format(
        instruction = instruction.strip(),
        input       = user_input.strip(),
    )


def build_mcq_prompt(
    question: str,
    choices:  list[str],
) -> str:
    """
    formats a multiple choice question with labelled options.
    """
    labels       = ["A", "B", "C", "D", "E"]
    labeled      = [f"{labels[i]}. {c}" for i, c in enumerate(choices) if c]
    choices_text = "\n".join(labeled)

    instruction = (
        "You are a medical expert. Answer the following multiple choice "
        "medical question by selecting the single best answer."
    )
    user_input = f"{question}\n\n{choices_text}"

    return build_prompt(user_input, instruction)


def build_pubmed_prompt(
    question: str,
    context:  str = "",
) -> str:
    """
    formats a PubMed-style question with optional abstract context.
    """
    instruction = (
        "You are a biomedical research expert. Based on the provided context, "
        "answer the medical question accurately and concisely."
    )
    user_input = (
        f"Question: {question}\n\nContext: {context}"
        if context else question
    )
    return build_prompt(user_input, instruction)


def extract_response(full_output: str) -> str:
    """
    extracts only the model response from the full generated text.
    strips the prompt prefix if the model echoed it back.
    """
    marker = "### Response:"
    if marker in full_output:
        return full_output.split(marker)[-1].strip()
    return full_output.strip()


# ── smoke test ────────────────────────────────────────────────────────────────

def smoke_test() -> None:
    print("\n=== prompt_template.py smoke test ===\n")

    # 1. basic prompt build
    prompt = build_prompt("What is the mechanism of action of metformin?")
    assert "### Instruction:" in prompt
    assert "### Input:"       in prompt
    assert "### Response:"    in prompt
    assert "metformin"        in prompt
    print("  build_prompt         : ok")
    print(f"  sample:\n{prompt}\n")

    # 2. MCQ prompt
    mcq = build_mcq_prompt(
        question = "Which drug is first-line for type 2 diabetes?",
        choices  = ["Insulin", "Metformin", "Glipizide", "Sitagliptin"],
    )
    assert "A. Insulin"    in mcq
    assert "B. Metformin"  in mcq
    assert "C. Glipizide"  in mcq
    assert "D. Sitagliptin" in mcq
    print("  build_mcq_prompt     : ok")

    # 3. pubmed prompt
    pub = build_pubmed_prompt(
        question = "Does aspirin reduce cardiovascular risk?",
        context  = "A meta-analysis of 12 RCTs showed aspirin reduces MI risk by 25%.",
    )
    assert "Question:"  in pub
    assert "Context:"   in pub
    assert "aspirin"    in pub
    print("  build_pubmed_prompt  : ok")

    # 4. response extraction
    fake_output = (
        "### Instruction:\nYou are a doctor.\n\n"
        "### Input:\nWhat is aspirin?\n\n"
        "### Response:\nAspirin is a salicylate NSAID used for pain and fever."
    )
    extracted = extract_response(fake_output)
    assert extracted == "Aspirin is a salicylate NSAID used for pain and fever."
    print("  extract_response     : ok")

    # 5. extraction with no marker (model didn't echo prompt)
    plain = "Metformin works by inhibiting hepatic glucose production."
    assert extract_response(plain) == plain
    print("  extract_response (no marker): ok")

    print("\n=== smoke test passed ===")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    smoke_test()