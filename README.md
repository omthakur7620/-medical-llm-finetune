# 🏥 Medical LLM — Fine-tuned Mistral 7B with SFT + DPO

> Fine-tuned `Mistral-7B-v0.1` on medical QA data using **LoRA**, **Supervised Fine-Tuning (SFT)**, and **Direct Preference Optimization (DPO)** to create a domain-specialized medical assistant that responds like a clinical expert.

---

## 🎯 Project Overview

General-purpose LLMs like GPT-4 give cautious, hedged answers to medical questions. This project fine-tunes a small open-source model (Mistral 7B) on medical QA data and aligns it with expert preferences using DPO — the same technique OpenAI used to turn GPT-3 into ChatGPT.

**The result:** A model that answers medical questions confidently, accurately, and with proper clinical reasoning — without unnecessary disclaimers.

---

## 📊 Results

| Model | ROUGE-L | BERTScore F1 | MCQ Accuracy | Win Rate |
|-------|---------|--------------|--------------|----------|
| Base Mistral 7B | — | — | — | — |
| + SFT (ours) | — | — | — | —% vs base |
| + DPO (ours) | — | — | — | —% vs SFT |

> ⚠️ Results will be populated after Colab training run. See `evals/results/comparison_report.md`.

**W&B Training Dashboard:** [link after training]  
**HuggingFace Model:** [link after push]

---

## 🏗️ Architecture

```
Raw Medical Data
       │
       ▼
┌─────────────────────────────────────────────┐
│              Data Pipeline                  │
│  download → clean → format → split → DPO   │
│  (MedQA + PubMedQA + MedMCQA = 297 samples)│
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│         Stage 1 — SFT Training              │
│  Mistral 7B + LoRA (r=16) + 4-bit quant    │
│  237 instruction pairs × 3 epochs          │
│  Loss: cross-entropy on output tokens      │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│         Stage 2 — DPO Alignment             │
│  SFT model + 50 chosen/rejected pairs      │
│  Beta = 0.1 (KL penalty)                   │
│  Loss: contrastive preference loss         │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│              Evaluation                     │
│  ROUGE + BERTScore + MCQ Accuracy          │
│  LLM-as-judge win rate (Groq llama-3.3-70b)│
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│              Deployment                     │
│  FastAPI REST API + Gradio Demo UI         │
│  HuggingFace Hub model hosting             │
└─────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
medical-llm-finetune/
├── config.yaml                  # all hyperparameters
├── src/
│   ├── data/                    # data pipeline
│   │   ├── download.py          # download MedQA, PubMedQA, MedMCQA
│   │   ├── clean.py             # normalize, filter, validate
│   │   ├── formatter.py         # convert to Alpaca instruction format
│   │   ├── split.py             # 80/10/10 train/val/test split
│   │   └── dpo_builder.py       # generate chosen/rejected pairs via Groq
│   ├── training/
│   │   ├── utils.py             # model loading, LoRA setup, merge-and-save
│   │   ├── sft_train.py         # supervised fine-tuning with TRL SFTTrainer
│   │   ├── dpo_train.py         # DPO alignment with TRL DPOTrainer
│   │   └── callbacks.py         # W&B logging, early stopping, sample gen
│   ├── evaluation/
│   │   ├── generate.py          # inference for all 3 model versions
│   │   ├── metrics.py           # ROUGE, BERTScore, MCQ accuracy
│   │   ├── llm_judge.py         # LLM-as-judge win rate evaluation
│   │   ├── compare.py           # markdown comparison report generator
│   │   └── benchmark.py         # orchestrates full eval pipeline
│   └── inference/
│       ├── prompt_template.py   # Alpaca prompt formatting
│       ├── engine.py            # unified inference (local GPU or Groq API)
│       └── api.py               # FastAPI REST API
├── app/
│   ├── gradio_app.py            # Gradio demo UI (4 tabs)
│   └── examples.py              # example questions for UI
└── scripts/
    ├── prepare_data.sh          # run full data pipeline
    ├── run_sft.sh               # run SFT training (GPU)
    ├── run_dpo.sh               # run DPO training (GPU)
    ├── run_eval.sh              # run evaluation pipeline
    └── push_to_hub.sh           # merge + push to HuggingFace Hub
```

---

## 🚀 Quick Start

### 1. Clone and setup

```bash
git clone https://github.com/your-username/medical-llm-finetune
cd medical-llm-finetune

python3.10 -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 2. Set API keys

```bash
# .env
HF_TOKEN=hf_your_token
WANDB_API_KEY=your_wandb_key
GROQ_API_KEY=gsk_your_groq_key
```

### 3. Run data pipeline

```bash
bash scripts/prepare_data.sh

# or on Windows:
python src/data/download.py
python src/data/clean.py
python src/data/formatter.py
python src/data/split.py
python src/data/dpo_builder.py
```

### 4. Launch demo UI (no GPU needed)

```bash
python app/gradio_app.py --serve
# open http://localhost:7860
```

### 5. Run evaluation

```bash
python src/evaluation/benchmark.py --run --samples 30
# report saved to evals/results/comparison_report.md
```

### 6. Launch API server

```bash
uvicorn src.inference.api:app --reload --port 8000
# docs at http://localhost:8000/docs
```

---

## 🖥️ Training on Google Colab (GPU Required)

```python
# In a Colab notebook:

# 1. clone repo
!git clone https://github.com/your-username/medical-llm-finetune
%cd medical-llm-finetune

# 2. install dependencies
!pip install -r requirements.txt

# 3. set env vars
import os
os.environ["HF_TOKEN"]      = "hf_your_token"
os.environ["WANDB_API_KEY"] = "your_wandb_key"
os.environ["GROQ_API_KEY"]  = "gsk_your_groq_key"

# 4. SFT training (~2-3 hours on T4)
!python src/training/sft_train.py --train

# 5. DPO training (~30 min)
!python src/training/dpo_train.py --train

# 6. evaluation
!python src/evaluation/benchmark.py --run

# 7. push to HuggingFace Hub
!HF_USERNAME=your-username bash scripts/push_to_hub.sh
```

---

## 🔌 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/generate` | POST | Free-form medical Q&A |
| `/generate/mcq` | POST | Multiple choice question solver |
| `/generate/pubmed` | POST | PubMed abstract Q&A |
| `/compare` | POST | Side-by-side model comparison |
| `/info` | GET | Model and engine info |

**Example request:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"input": "What is the mechanism of action of metformin?", "max_new_tokens": 256}'
```

---

## 🧪 Tech Stack

| Component | Technology |
|-----------|------------|
| Base model | Mistral-7B-v0.1 |
| Fine-tuning | PEFT LoRA (r=16) |
| Quantization | bitsandbytes 4-bit NF4 |
| SFT training | TRL SFTTrainer |
| DPO training | TRL DPOTrainer |
| Experiment tracking | Weights & Biases |
| Evaluation | ROUGE, BERTScore, LLM-as-judge |
| Judge model | Groq llama-3.3-70b-versatile |
| API | FastAPI + uvicorn |
| Demo UI | Gradio |
| Model hosting | HuggingFace Hub |

---

## 📈 Training Details

### Data
- **MedQA (USMLE)** — 100 samples, US medical licensing exam style
- **PubMedQA** — 100 samples, biomedical research Q&A
- **MedMCQA** — 97 samples, Indian medical entrance exam MCQs
- **Total** — 297 samples → 237 train / 29 val / 31 test
- **DPO pairs** — 50 chosen/rejected pairs generated via Groq

### SFT Hyperparameters
```yaml
base_model  : mistralai/Mistral-7B-v0.1
lora_r      : 16
lora_alpha  : 32
epochs      : 3
batch_size  : 4
grad_accum  : 4
lr          : 2e-4
scheduler   : cosine
quantization: 4-bit NF4
```

### DPO Hyperparameters
```yaml
beta        : 0.1
epochs      : 1
batch_size  : 2
grad_accum  : 8
lr          : 5e-5
```

---

## ⚠️ Limitations & Future Work

**Current limitations:**
- Trained on 297 samples — proof of concept, not production quality
- Does not cover all medical specialties (no radiology, surgery, psychiatry in depth)
- Not clinically validated — should not be used for real medical decisions
- Evaluation uses simulated model outputs locally (real numbers after Colab training)

**Future work:**
- Scale to 10,000+ samples across more medical specialties
- Add RLHF with real human annotators (not synthetic labels)
- Implement hallucination detection and confidence scoring
- Add retrieval-augmented generation (RAG) over medical knowledge bases
- Fine-tune on newer base models (Llama 3, Mistral v0.3)
- Clinical safety evaluation and red-teaming

---

## 👤 Author

Built as a portfolio project to demonstrate end-to-end LLM fine-tuning with SFT + DPO alignment.

---

## 📄 License

MIT License — see LICENSE file for details.

> **Medical Disclaimer:** This model is for educational and research purposes only.
> It is not intended for clinical use or medical decision-making.