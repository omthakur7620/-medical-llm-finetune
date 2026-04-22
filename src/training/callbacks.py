import os
import torch
import wandb
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

# ── W&B sample generation callback ───────────────────────────────────────────

class SampleGenerationCallback(TrainerCallback):
    """
    every N steps, generate a sample response from the model
    and log it to W&B so you can watch quality improve during training.
    """

    def __init__(
        self,
        tokenizer,
        sample_prompt: str,
        every_n_steps: int = 50,
        max_new_tokens: int = 128,
    ):
        self.tokenizer      = tokenizer
        self.sample_prompt  = sample_prompt
        self.every_n_steps  = every_n_steps
        self.max_new_tokens = max_new_tokens

    def on_step_end(
        self,
        args:    TrainingArguments,
        state:   TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        if state.global_step % self.every_n_steps != 0:
            return
        if model is None:
            return

        model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                self.sample_prompt,
                return_tensors = "pt",
                truncation     = True,
                max_length     = 512,
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens = self.max_new_tokens,
                do_sample      = False,
                pad_token_id   = self.tokenizer.pad_token_id,
            )

        # decode only the newly generated tokens (not the prompt)
        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response   = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        if wandb.run is not None:
            wandb.log({
                "sample/step"    : state.global_step,
                "sample/response": wandb.Html(
                    f"<b>Step {state.global_step}</b><br>"
                    f"<b>Prompt:</b> {self.sample_prompt[:200]}<br>"
                    f"<b>Response:</b> {response}"
                ),
            }, step=state.global_step)

        print(f"\n[step {state.global_step}] sample generation:")
        print(f"  prompt   : {self.sample_prompt[:100]}...")
        print(f"  response : {response[:200]}\n")

        model.train()


# ── GPU memory callback ───────────────────────────────────────────────────────

class GPUMemoryCallback(TrainerCallback):
    """
    log GPU memory usage to W&B every N steps.
    useful for spotting OOM before it crashes training.
    """

    def __init__(self, every_n_steps: int = 10):
        self.every_n_steps = every_n_steps

    def on_step_end(
        self,
        args:    TrainingArguments,
        state:   TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step % self.every_n_steps != 0:
            return
        if not torch.cuda.is_available():
            return

        allocated = torch.cuda.memory_allocated()  / 1e9
        reserved  = torch.cuda.memory_reserved()   / 1e9

        if wandb.run is not None:
            wandb.log({
                "gpu/memory_allocated_gb": allocated,
                "gpu/memory_reserved_gb" : reserved,
            }, step=state.global_step)


# ── DPO reward margin callback ────────────────────────────────────────────────

class DPORewardCallback(TrainerCallback):
    """
    logs chosen vs rejected reward and margin during DPO training.
    reward margin = chosen_reward - rejected_reward.
    you want this to increase over training — means model is learning preference.
    """

    def on_log(
        self,
        args:    TrainingArguments,
        state:   TrainerState,
        control: TrainerControl,
        logs:    dict = None,
        **kwargs,
    ):
        if logs is None or wandb.run is None:
            return

        chosen_reward   = logs.get("rewards/chosen",   None)
        rejected_reward = logs.get("rewards/rejected", None)

        if chosen_reward is not None and rejected_reward is not None:
            margin = chosen_reward - rejected_reward
            wandb.log({
                "dpo/reward_margin"   : margin,
                "dpo/chosen_reward"   : chosen_reward,
                "dpo/rejected_reward" : rejected_reward,
            }, step=state.global_step)

            print(f"  [step {state.global_step}] "
                  f"margin={margin:.4f}  "
                  f"chosen={chosen_reward:.4f}  "
                  f"rejected={rejected_reward:.4f}")


# ── early stopping on loss plateau ────────────────────────────────────────────

class EarlyStoppingCallback(TrainerCallback):
    """
    stops training early if validation loss stops improving.
    patience = how many eval steps to wait before stopping.
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience    = patience
        self.min_delta   = min_delta
        self._best_loss  = float("inf")
        self._wait       = 0

    def on_evaluate(
        self,
        args:    TrainingArguments,
        state:   TrainerState,
        control: TrainerControl,
        metrics: dict = None,
        **kwargs,
    ):
        if metrics is None:
            return

        val_loss = metrics.get("eval_loss", None)
        if val_loss is None:
            return

        if val_loss < self._best_loss - self.min_delta:
            self._best_loss = val_loss
            self._wait      = 0
        else:
            self._wait += 1
            print(f"  early stopping: no improvement for {self._wait}/{self.patience} evals "
                  f"(best={self._best_loss:.4f}, current={val_loss:.4f})")
            if self._wait >= self.patience:
                print("  early stopping triggered — stopping training")
                control.should_training_stop = True


# ── entry point — smoke test ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== callbacks.py smoke test ===\n")

    # just verify all classes import and instantiate without error
    from transformers import AutoTokenizer

    # mock tokenizer with a tiny model just for testing
    print("testing callback instantiation ...")

    cb1 = GPUMemoryCallback(every_n_steps=10)
    print("  GPUMemoryCallback     : ok")

    cb2 = DPORewardCallback()
    print("  DPORewardCallback     : ok")

    cb3 = EarlyStoppingCallback(patience=3)
    print("  EarlyStoppingCallback : ok")

    # test early stopping logic directly
    from transformers import TrainingArguments, TrainerState, TrainerControl
    args    = TrainingArguments(output_dir="tmp_test", use_cpu=True)
    state   = TrainerState()
    control = TrainerControl()

    # simulate 4 evals with no improvement — should trigger stop on 4th
    for i in range(4):
        cb3.on_evaluate(args, state, control, metrics={"eval_loss": 1.5})

    assert control.should_training_stop is True, "early stopping did not trigger"
    print("  EarlyStoppingCallback logic: ok — triggered after 3 patience evals")

    # cleanup
    import shutil
    shutil.rmtree("tmp_test", ignore_errors=True)

    print("\n=== smoke test passed — callbacks.py is ready ===")