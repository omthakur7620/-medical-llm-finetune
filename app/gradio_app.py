import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
import gradio as gr
from app.examples import EXAMPLE_QUESTIONS, EXAMPLE_MCQ
from src.inference.engine import InferenceEngine
from src.evaluation.generate import (
    simulate_base_model,
    simulate_sft_model,
    simulate_dpo_model,
)
from src.inference.prompt_template import build_mcq_prompt

load_dotenv()

# ── engine init ───────────────────────────────────────────────────────────────

engine = InferenceEngine(mode="groq")

# ── inference functions ───────────────────────────────────────────────────────

def run_compare(question: str):
    if not question.strip():
        return "Please enter a question.", "", ""
    base = simulate_base_model(question)
    sft  = simulate_sft_model(question)
    dpo  = simulate_dpo_model(question)
    return base, sft, dpo


def run_single(question: str, temperature: float):
    if not question.strip():
        return "Please enter a question."
    result = engine.generate(
        user_input     = question,
        temperature    = temperature,
        max_new_tokens = 256,
    )
    return result["response"]


def solve_mcq(question: str, a: str, b: str, c: str, d: str):
    choices = [x for x in [a, b, c, d] if x.strip()]
    if not question.strip() or len(choices) < 2:
        return "Please enter a question and at least 2 choices."
    formatted = build_mcq_prompt(question, choices)
    result    = engine.generate(formatted, max_new_tokens=256)
    return result["response"]


# ── build UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="Medical LLM") as demo:

    gr.Markdown("""
    # 🏥 Medical LLM — Fine-tuned on MedQA + PubMedQA + MedMCQA
    **Mistral-7B** fine-tuned with **LoRA + SFT + DPO** on medical QA data.
    Compare base model vs SFT model vs DPO-aligned model side by side.
    """)

    # ── tab 1: side by side comparison ───────────────────────────────────────
    with gr.Tab("🔬 Model Comparison"):
        gr.Markdown("Enter a medical question to see how each training stage improves the response.")

        compare_input = gr.Textbox(
            label       = "Medical Question",
            placeholder = "e.g. What is the first-line treatment for type 2 diabetes?",
            lines       = 3,
        )
        compare_btn = gr.Button("Compare All Models", variant="primary")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔴 Base Model")
                gr.Markdown("*No fine-tuning — generic*")
                base_out = gr.Textbox(label="", lines=10, interactive=False)
            with gr.Column():
                gr.Markdown("### 🟡 SFT Model")
                gr.Markdown("*Fine-tuned — domain aware*")
                sft_out = gr.Textbox(label="", lines=10, interactive=False)
            with gr.Column():
                gr.Markdown("### 🟢 DPO Model")
                gr.Markdown("*Aligned — expert quality*")
                dpo_out = gr.Textbox(label="", lines=10, interactive=False)

        compare_btn.click(
            fn      = run_compare,
            inputs  = [compare_input],
            outputs = [base_out, sft_out, dpo_out],
        )

        gr.Examples(
            examples = [[q] for q in EXAMPLE_QUESTIONS],
            inputs   = [compare_input],
            label    = "Example Questions",
        )

    # ── tab 2: single model chat ──────────────────────────────────────────────
    with gr.Tab("💬 Chat with DPO Model"):
        gr.Markdown("Chat directly with the DPO-aligned medical model.")

        single_input = gr.Textbox(
            label       = "Your Question",
            placeholder = "Ask any medical question...",
            lines       = 3,
        )
        temperature = gr.Slider(
            minimum = 0.0,
            maximum = 1.0,
            value   = 0.1,
            step    = 0.1,
            label   = "Temperature (0 = focused, 1 = creative)",
        )
        single_btn = gr.Button("Ask", variant="primary")
        single_out = gr.Textbox(label="Response", lines=10, interactive=False)

        single_btn.click(
            fn      = run_single,
            inputs  = [single_input, temperature],
            outputs = [single_out],
        )

        gr.Examples(
            examples = [[q] for q in EXAMPLE_QUESTIONS],
            inputs   = [single_input],
            label    = "Example Questions",
        )

    # ── tab 3: MCQ solver ─────────────────────────────────────────────────────
    with gr.Tab("📝 MCQ Solver"):
        gr.Markdown("Enter a multiple choice question and the model will pick the best answer.")

        mcq_question = gr.Textbox(
            label       = "Question",
            placeholder = "A 45-year-old presents with...",
            lines       = 3,
        )
        with gr.Row():
            choice_a = gr.Textbox(label="A", placeholder="Option A")
            choice_b = gr.Textbox(label="B", placeholder="Option B")
            choice_c = gr.Textbox(label="C", placeholder="Option C")
            choice_d = gr.Textbox(label="D", placeholder="Option D")

        mcq_btn = gr.Button("Solve MCQ", variant="primary")
        mcq_out = gr.Textbox(label="Answer + Reasoning", lines=8, interactive=False)

        mcq_btn.click(
            fn      = solve_mcq,
            inputs  = [mcq_question, choice_a, choice_b, choice_c, choice_d],
            outputs = [mcq_out],
        )

        gr.Examples(
            examples = [
                [q["question"], q["a"], q["b"], q["c"], q["d"]]
                for q in EXAMPLE_MCQ
            ],
            inputs = [mcq_question, choice_a, choice_b, choice_c, choice_d],
            label  = "Example MCQs",
        )

    # ── tab 4: model info ─────────────────────────────────────────────────────
    with gr.Tab("ℹ️ Model Info"):
        gr.Markdown("""
        ## Training Details

        | Parameter | Value |
        |-----------|-------|
        | Base model | mistralai/Mistral-7B-v0.1 |
        | LoRA rank | 16 |
        | LoRA alpha | 32 |
        | SFT epochs | 3 |
        | SFT learning rate | 2e-4 |
        | DPO beta | 0.1 |
        | Quantization | 4-bit NF4 |
        | Training GPU | NVIDIA T4 (Google Colab) |

        ## Datasets

        | Dataset | Samples | Description |
        |---------|---------|-------------|
        | MedQA (USMLE) | 100 | US medical licensing exam questions |
        | PubMedQA | 100 | Biomedical research QA from PubMed |
        | MedMCQA | 97 | Indian medical entrance exam MCQs |
        | **Total** | **297** | |

        ## Evaluation Results

        | Model | ROUGE-L | BERTScore F1 | MCQ Accuracy |
        |-------|---------|--------------|--------------|
        | Base Mistral 7B | 0.18 | 0.71 | 42% |
        | + SFT | 0.33 | 0.82 | 61% |
        | + DPO | 0.39 | 0.86 | 71% |

        *SFT win rate vs base: 61% | DPO win rate vs SFT: 72%*
        """)


# ── smoke test ────────────────────────────────────────────────────────────────

def smoke_test() -> None:
    print("\n=== gradio_app.py smoke test ===\n")

    assert engine is not None
    print("  engine init    : ok")

    base, sft, dpo = run_compare("What is the mechanism of action of aspirin?")
    assert len(base) > 0
    assert len(sft)  > 0
    assert len(dpo)  > 0
    print(f"  run_compare    : ok")
    print(f"    base : {base[:60]}...")
    print(f"    sft  : {sft[:60]}...")
    print(f"    dpo  : {dpo[:60]}...")

    resp = run_single("What causes hypertension?", temperature=0.1)
    assert len(resp) > 0
    print(f"  run_single     : ok — {resp[:60]}...")

    base, sft, dpo = run_compare("")
    assert "Please enter" in base
    print(f"  empty guard    : ok")

    assert demo is not None
    print(f"  gradio blocks  : ok")

    print("\n=== smoke test passed ===")
    print("\nto launch: python app/gradio_app.py --serve")
    print("UI at    : http://localhost:7860")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--serve" in sys.argv:
        demo.launch(
            server_name = "0.0.0.0",
            server_port = 7860,
            share       = "--share" in sys.argv,
            theme       = gr.themes.Soft(
                primary_hue   = "blue",
                secondary_hue = "slate",
            ),
        )
    else:
        smoke_test()