"""Pluggable generator backends for run_benchmark.py.

- HFGenerator: real local HuggingFace generation (e.g. StarCoder2-7B,
  CodeLlama-7B-Python). Needs torch + transformers +, for a 7B model in
  practice, a GPU with >=16GB VRAM. NOT runnable on this development machine
  (no torch installed here, checked before writing this file) — meant for
  the Colab/GPU environment this project's other notebooks already use
  (repocoder-mine/colab_*.ipynb).
- StubGenerator: deterministic, instant, no model download. Used by
  run_benchmark.py --generator stub (the default, and what --dry-run
  implies) to validate the whole pipeline — chunking, retrieval, prompt
  building, scoring, the final comparison table — end to end without a GPU.
  Its output is NOT a real generation result and the comparison table it
  produces must never be reported as a real EM/ES/Pass@1 result for the
  paper; only as a smoke test that the harness runs correctly.
"""

from typing import Protocol


class Generator(Protocol):
    def generate(self, prompt: str) -> str: ...


class StubGenerator:
    """Deterministic dry-run stand-in: echoes back the prompt's own last
    non-empty line. Exercises EM/ES/Pass@1 with realistically-shaped
    (sometimes matching, mostly not) output, with zero model cost."""

    def generate(self, prompt: str) -> str:
        lines = [line for line in prompt.splitlines() if line.strip()]
        return lines[-1] if lines else ""


class HFGenerator:
    """Real generation via a local HuggingFace causal LM (greedy decoding,
    matching this project's other Pass@1 runs: t=0.2 top_p=0.95 in the
    original notebooks is for sampling variance studies; plain greedy is
    used here for a single canonical Pass@1 sample, do_sample=False)."""

    def __init__(self, model_name: str, max_new_tokens: int = 64, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Truncate from the left (drop the oldest context first) so a
        # too-long prompt never loses the tail — the part right before the
        # completion point, i.e. the part that matters most. Same lesson as
        # trim_code() in run_benchmark.py, applied at the tokenizer level too.
        self.tokenizer.truncation_side = "left"
        # device_map="auto" (accelerate) streams weights shard-by-shard
        # straight to their target device instead of first materializing the
        # whole model in CPU RAM and only then moving it — for a 7B model
        # that difference is the gap between fitting in Colab's default CPU
        # RAM and getting silently OOM-killed (no traceback, no exit code,
        # whatever was still buffered in stdout is lost) partway through a
        # run. Matches the fix already used elsewhere in this project's
        # other notebooks (repocoder-mine/generator_min.py).
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16,
            device_map="auto" if device != "cpu" else {"": "cpu"},
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def get_generator(name: str, model_name: str | None = None, device: str = "cuda") -> Generator:
    if name == "stub":
        return StubGenerator()
    if name == "hf":
        if not model_name:
            raise ValueError("--model-name is required for --generator hf")
        return HFGenerator(model_name, device=device)
    raise ValueError(f"Unknown generator backend {name!r}, expected 'stub' or 'hf'")
