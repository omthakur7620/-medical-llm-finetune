import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from src.inference.engine import InferenceEngine
from src.inference.prompt_template import (
    build_mcq_prompt,
    build_pubmed_prompt,
    DEFAULT_INSTRUCTION,
)

load_dotenv()

# ── engine singleton ──────────────────────────────────────────────────────────
# shared across all requests — initialized once at startup

engine: InferenceEngine | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """startup: init engine. shutdown: cleanup."""
    global engine
    mode       = os.getenv("INFERENCE_MODE", "groq")   # "groq" or "local"
    model_path = os.getenv("MODEL_PATH", "models/dpo_model")

    print(f"\nstarting inference engine (mode={mode}) ...")
    engine = InferenceEngine(mode=mode, model_path=model_path if mode == "local" else None)
    print("engine ready\n")
    yield
    print("shutting down ...")


# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Medical LLM API",
    description = "Fine-tuned medical LLM inference API",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── request / response schemas ────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    input:          str   = Field(..., description="User question or prompt")
    instruction:    str   = Field(DEFAULT_INSTRUCTION, description="System instruction")
    max_new_tokens: int   = Field(256,  ge=1,   le=1024)
    temperature:    float = Field(0.1,  ge=0.0, le=1.0)


class MCQRequest(BaseModel):
    question:       str        = Field(..., description="MCQ question text")
    choices:        list[str]  = Field(..., description="List of answer choices")
    max_new_tokens: int        = Field(256, ge=1, le=1024)


class PubMedRequest(BaseModel):
    question:       str  = Field(..., description="Medical question")
    context:        str  = Field("",  description="PubMed abstract context")
    max_new_tokens: int  = Field(256, ge=1, le=1024)


class CompareRequest(BaseModel):
    input:          str   = Field(..., description="Question to compare across models")
    max_new_tokens: int   = Field(256, ge=1, le=1024)


class GenerateResponse(BaseModel):
    response:    str
    latency_ms:  int
    mode:        str


class CompareResponse(BaseModel):
    base_model:  str
    sft_model:   str
    dpo_model:   str
    latency_ms:  int


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """check if server is running"""
    return {
        "status" : "ok",
        "mode"   : engine.mode if engine else "not initialized",
        "gpu"    : engine.info().get("gpu", "none") if engine else "none",
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """
    generate a response for any free-form medical question.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not initialized")

    result = engine.generate(
        user_input     = req.input,
        instruction    = req.instruction,
        max_new_tokens = req.max_new_tokens,
        temperature    = req.temperature,
    )
    return GenerateResponse(
        response   = result["response"],
        latency_ms = result["latency_ms"],
        mode       = result["mode"],
    )


@app.post("/generate/mcq", response_model=GenerateResponse)
def generate_mcq(req: MCQRequest):
    """
    generate a response for a multiple choice question.
    automatically formats choices as A/B/C/D.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not initialized")

    formatted_input = build_mcq_prompt(req.question, req.choices)
    result = engine.generate(
        user_input     = formatted_input,
        max_new_tokens = req.max_new_tokens,
    )
    return GenerateResponse(
        response   = result["response"],
        latency_ms = result["latency_ms"],
        mode       = result["mode"],
    )


@app.post("/generate/pubmed", response_model=GenerateResponse)
def generate_pubmed(req: PubMedRequest):
    """
    generate a response for a PubMed-style question with abstract context.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not initialized")

    formatted_input = build_pubmed_prompt(req.question, req.context)
    result = engine.generate(
        user_input     = formatted_input,
        max_new_tokens = req.max_new_tokens,
    )
    return GenerateResponse(
        response   = result["response"],
        latency_ms = result["latency_ms"],
        mode       = result["mode"],
    )


@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    """
    run the same question through all 3 model versions and return
    responses side by side. this is the demo endpoint.
    on local: uses groq with different system prompts to simulate each model.
    on colab: uses actual trained checkpoints.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not initialized")

    from src.evaluation.generate import (
        simulate_base_model,
        simulate_sft_model,
        simulate_dpo_model,
    )
    import time

    t0          = time.time()
    base_resp   = simulate_base_model(req.input)
    sft_resp    = simulate_sft_model(req.input)
    dpo_resp    = simulate_dpo_model(req.input)
    latency_ms  = round((time.time() - t0) * 1000)

    return CompareResponse(
        base_model = base_resp,
        sft_model  = sft_resp,
        dpo_model  = dpo_resp,
        latency_ms = latency_ms,
    )


@app.get("/info")
def info():
    """engine and model info"""
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not initialized")
    return engine.info()


# ── smoke test ────────────────────────────────────────────────────────────────

def smoke_test() -> None:
    print("\n=== api.py smoke test ===\n")

    from fastapi.testclient import TestClient

    # manually init engine for testing without lifespan
    global engine
    engine = InferenceEngine(mode="groq")

    client = TestClient(app)

    # 1. health check
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    print(f"  GET /health     : {data} — ok")

    # 2. /info
    resp = client.get("/info")
    assert resp.status_code == 200
    print(f"  GET /info       : {resp.json()} — ok")

    # 3. /generate
    resp = client.post("/generate", json={
        "input"         : "What is the first-line treatment for hypertension?",
        "max_new_tokens": 64,
        "temperature"   : 0.1,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "response"   in data
    assert "latency_ms" in data
    assert len(data["response"]) > 0
    print(f"  POST /generate  : latency={data['latency_ms']}ms — ok")
    print(f"    response: {data['response'][:80]}...")

    # 4. /generate/mcq
    resp = client.post("/generate/mcq", json={
        "question" : "Which drug is first-line for type 2 diabetes?",
        "choices"  : ["Insulin", "Metformin", "Glipizide", "Sitagliptin"],
        "max_new_tokens": 64,
    })
    assert resp.status_code == 200
    print(f"  POST /generate/mcq : ok")

    # 5. /generate/pubmed
    resp = client.post("/generate/pubmed", json={
        "question": "Does aspirin reduce MI risk?",
        "context" : "RCTs show aspirin reduces MI by 25% in high-risk patients.",
        "max_new_tokens": 64,
    })
    assert resp.status_code == 200
    print(f"  POST /generate/pubmed : ok")

    print("\n=== smoke test passed ===")
    print("\nto run the API server:")
    print("  uvicorn src.inference.api:app --reload --port 8000")
    print("\nAPI docs available at:")
    print("  http://localhost:8000/docs")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if "--serve" in sys.argv:
        import uvicorn
        uvicorn.run("src.inference.api:app", host="0.0.0.0", port=8000, reload=True)
    else:
        smoke_test()