"""
Benchmark: forward-pass latency with activation injection vs. a normal forward pass.

Uses your real steering_hooks.py unmodified (imported, not copied) so the
benchmark exercises exactly the code path used in training/inference.

Usage (see benchmark.sbatch):
    python benchmark.py
Env vars (all optional, defaults shown):
    MODEL_NAME=Qwen/Qwen3-8B
    LAYER_IDX=1              # "after the second transformer layer" -> index 1, 0-based
    BATCH_SIZE=1
    SEQ_LEN=64
    N_ACT_POSITIONS=3        # number of <ACT> placeholder tokens
    STEERING_COEFFICIENT=1.0
    N_ITERS=50
    WARMUP=10
    DTYPE=bfloat16
"""

import os
import sys
import time

import torch

# steering_hooks.py must be importable: either on PYTHONPATH already
# (e.g. installed via `pip install -e .` as in train.sbatch), or sitting
# next to this script. We try both.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from steering_hooks import add_hook, get_hf_activation_steering_hook  # noqa: E402


def time_forward_passes(model, input_ids, attention_mask, n_iters: int, warmup: int) -> float:
    """Mean latency in ms over n_iters forward passes, GPU-synchronized."""
    device = input_ids.device
    with torch.no_grad():
        for _ in range(warmup):
            model(input_ids=input_ids, attention_mask=attention_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_iters):
            model(input_ids=input_ids, attention_mask=attention_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

    return (end - start) / n_iters * 1000.0


def main():
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")
    layer_idx = int(os.environ.get("LAYER_IDX", "1"))
    batch_size = int(os.environ.get("BATCH_SIZE", "1"))
    seq_len = int(os.environ.get("SEQ_LEN", "64"))
    n_act_positions = int(os.environ.get("N_ACT_POSITIONS", "3"))
    steering_coefficient = float(os.environ.get("STEERING_COEFFICIENT", "1.0"))
    n_iters = int(os.environ.get("N_ITERS", "50"))
    warmup = int(os.environ.get("WARMUP", "10"))
    dtype_str = os.environ.get("DTYPE", "bfloat16")
    dtype = getattr(torch, dtype_str)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"== Config ==")
    print(f"model={model_name} layer_idx={layer_idx} batch_size={batch_size} "
          f"seq_len={seq_len} n_act_positions={n_act_positions} "
          f"steering_coefficient={steering_coefficient} n_iters={n_iters} "
          f"warmup={warmup} dtype={dtype_str} device={device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()

    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)

    d_model = model.config.hidden_size
    vectors = [torch.randn(n_act_positions, d_model, dtype=dtype, device=device) for _ in range(batch_size)]
    positions = [list(range(n_act_positions)) for _ in range(batch_size)]

    target_module = model.model.layers[layer_idx]

    # --- Baseline: normal forward pass, no hook ---
    baseline_ms = time_forward_passes(model, input_ids, attention_mask, n_iters, warmup)

    # --- With injection hook: real get_hf_activation_steering_hook from steering_hooks.py ---
    # As currently written in steering_hooks.py, this hook is ADDITIVE
    # (resid = steered_KD + orig_KD), matching Eq. 3.2 in Chapter 3 and the
    # paper's stated design (additive injection avoids the norm blow-up
    # that replacement caused in the original LatentQA implementation).
    hook = get_hf_activation_steering_hook(
        vectors=vectors,
        positions=positions,
        steering_coefficient=steering_coefficient,
        device=device,
        dtype=dtype,
    )
    with add_hook(target_module, hook):
        injected_ms = time_forward_passes(model, input_ids, attention_mask, n_iters, warmup)

    overhead_pct = (injected_ms / baseline_ms - 1) * 100

    print("\n================ RESULTS ================")
    print(f"Normal forward pass (no injection): {baseline_ms:.3f} ms")
    print(f"Forward pass WITH injection hook:   {injected_ms:.3f} ms")
    print(f"Overhead:                            {overhead_pct:+.2f}%")
    print("==========================================")


if __name__ == "__main__":
    main()
