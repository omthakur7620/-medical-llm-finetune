import os
import sys
import time
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from src.training.utils import load_config, load_tokenizer
from src.inference.prompt_template import build_prompt, extract_response

load_dotenv()

# ── inference engine ──────────────────────────────────────────────────────────

class InferenceEngine:
    """
    unified inference engine that works in two modes:

    mode 1 — local (GPU on Colab):
        loads actual fine-tuned model checkpoint from disk.
        use this after training on Colab.

    mode 2 — groq (local machine / no GPU):
        routes requests through Groq API.
        use this for demo and testing without GPU.
    """

    def __init__(
        self,
        mode:         str = "groq",      # "local" or "groq"
        model_path:   str | None = None, # path to checkpoint (local mode)
        config_path:  str = "config.yaml",
    ):
        self.mode        = mode
        self.model_path  = model_path
        self.cfg         = load_config(config_path)
        self.model       = None
        self.tokenizer   = None
        self._groq_client = None

        if mode == "local":
            self._load_local_model()
        elif mode == "groq":
            self._init_groq()
        else:
            raise ValueError(f"mode must be 'local' or 'groq', got '{mode}'")

    # ── local model loading ───────────────────────────────────────────────────

    def _load_local_model(self) -> None:
        """load fine-tuned model from checkpoint for GPU inference"""
        assert self.model_path, "model_path required for local mode"
        assert torch.cuda.is_available(), (
            "local mode requires a GPU. "
            "use mode='groq' on CPU-only machines."
        )

        from src.training.utils import load_peft_model

        model_cfg = self.cfg["model"]
        print(f"loading model from {self.model_path} ...")

        self.tokenizer = load_tokenizer(model_cfg["base_model"])
        self.model     = load_peft_model(
            base_model_name = model_cfg["base_model"],
            adapter_path    = self.model_path,
            quantize        = True,
        )
        self.model.eval()
        print("model loaded — ready for inference")

    # ── groq client init ──────────────────────────────────────────────────────

    def _init_groq(self) -> None:
        """init Groq client for API-based inference"""
        from groq import Groq
        self._groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self._groq_model  = "llama-3.3-70b-versatile"
        print(f"groq engine ready — model: {self._groq_model}")

    # ── generate ──────────────────────────────────────────────────────────────

    def generate(
        self,
        user_input:     str,
        instruction:    str | None = None,
        max_new_tokens: int   = 256,
        temperature:    float = 0.1,
    ) -> dict:
        """
        generate a response for the given user input.
        returns dict with: response, prompt, latency_ms, mode
        """
        from src.inference.prompt_template import DEFAULT_INSTRUCTION
        instr  = instruction or DEFAULT_INSTRUCTION
        prompt = build_prompt(user_input, instr)

        t0 = time.time()

        if self.mode == "local":
            raw = self._generate_local(prompt, max_new_tokens, temperature)
        else:
            raw = self._generate_groq(prompt, instr, max_new_tokens, temperature)

        latency_ms = round((time.time() - t0) * 1000)
        response   = extract_response(raw)

        return {
            "response"   : response,
            "prompt"     : prompt,
            "latency_ms" : latency_ms,
            "mode"       : self.mode,
        }

    def _generate_local(
        self,
        prompt:         str,
        max_new_tokens: int,
        temperature:    float,
    ) -> str:
        """generate using locally loaded HuggingFace model"""
        inputs = self.tokenizer(
            prompt,
            return_tensors = "pt",
            truncation     = True,
            max_length     = self.cfg["model"]["max_seq_length"],
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens = max_new_tokens,
                do_sample      = temperature > 0,
                temperature    = temperature if temperature > 0 else 1.0,
                pad_token_id   = self.tokenizer.pad_token_id,
            )

        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def _generate_groq(
        self,
        prompt:         str,
        instruction:    str,
        max_new_tokens: int,
        temperature:    float,
    ) -> str:
        """generate using Groq API"""
        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES):
            try:
                resp = self._groq_client.chat.completions.create(
                    model       = self._groq_model,
                    messages    = [
                        {"role": "system", "content": instruction},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature = temperature,
                    max_tokens  = max_new_tokens,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if "rate" in str(e).lower() and attempt < MAX_RETRIES - 1:
                    time.sleep(5)
                else:
                    return f"[ERROR: {e}]"
        return "[ERROR: max retries exceeded]"

    # ── batch generate ────────────────────────────────────────────────────────

    def batch_generate(
        self,
        inputs:         list[str],
        max_new_tokens: int   = 256,
        temperature:    float = 0.1,
        delay:          float = 0.3,
    ) -> list[dict]:
        """
        generate responses for a list of inputs.
        delay (seconds) between calls — used for Groq rate limiting.
        """
        results = []
        for inp in inputs:
            result = self.generate(inp, max_new_tokens=max_new_tokens,
                                   temperature=temperature)
            results.append(result)
            if self.mode == "groq":
                time.sleep(delay)
        return results

    # ── info ──────────────────────────────────────────────────────────────────

    def info(self) -> dict:
        return {
            "mode"      : self.mode,
            "model_path": self.model_path,
            "gpu"       : torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        }


# ── smoke test ────────────────────────────────────────────────────────────────

def smoke_test() -> None:
    print("\n=== engine.py smoke test ===\n")

    # 1. groq mode init
    print("initializing engine in groq mode ...")
    engine = InferenceEngine(mode="groq")
    print(f"  engine info: {engine.info()}")

    # 2. single generate call
    print("\ntesting single generate ...")
    result = engine.generate(
        user_input  = "What is the mechanism of action of metformin?",
        max_new_tokens = 128,
        temperature    = 0.1,
    )
    assert "response"    in result
    assert "latency_ms"  in result
    assert "mode"        in result
    assert result["mode"] == "groq"
    assert len(result["response"]) > 0
    assert not result["response"].startswith("[ERROR")

    print(f"  latency    : {result['latency_ms']}ms")
    print(f"  response   : {result['response'][:120]}...")

    # 3. batch generate
    print("\ntesting batch generate (2 inputs) ...")
    batch_results = engine.batch_generate(
        inputs = [
            "What is the first-line treatment for hypertension?",
            "What causes type 1 diabetes?",
        ],
        max_new_tokens = 64,
        delay          = 0.5,
    )
    assert len(batch_results) == 2
    for r in batch_results:
        assert len(r["response"]) > 0
        print(f"  response : {r['response'][:80]}...")

    # 4. local mode requires GPU — just verify it raises correctly on CPU
    print("\ntesting local mode guard on CPU ...")
    try:
        engine_local = InferenceEngine(mode="local", model_path="models/dpo_model")
        print("  WARNING: should have raised AssertionError on CPU")
    except AssertionError as e:
        print(f"  correctly blocked local mode on CPU: {e}")

    # 5. invalid mode
    try:
        InferenceEngine(mode="invalid")
        print("  WARNING: should have raised ValueError")
    except ValueError as e:
        print(f"  correctly raised ValueError: {e}")

    print("\n=== smoke test passed ===")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    smoke_test()