#!/usr/bin/env python3
"""Text-only autoregressive generation for the static-seq_len=1 OpenVINO Gemma4 PTE.

The text_decoder is compiled for seq_len=1, so we feed one token per execute() call.
The model's KV cache is a mutable buffer that persists across calls; input_pos selects
the cache slot via index_copy. The model returns logits for the single position only.
"""

import argparse
import time

import torch
from tokenizers import Tokenizer
from executorch.runtime import Runtime

BOS, TURN_START, TURN_END = 2, 105, 106
HIDDEN_SIZE = {"e2b": 1536, "e4b": 2560}
STOP_TOKENS = {1, 106}  # <eos>, <turn|>


def build_prompt_ids(tok: Tokenizer, prompt: str) -> list:
    return (
        [BOS, TURN_START]
        + tok.encode("user\n").ids
        + tok.encode(prompt).ids
        + [TURN_END]
        + tok.encode("\n").ids
        + [TURN_START]
        + tok.encode("model\n").ids
    )


def step(method, token_id: int, pos: int, hidden: int) -> torch.Tensor:
    """Run one seq_len=1 forward; returns last-position logits [vocab]."""
    out = method.execute(
        [
            torch.tensor([[token_id]], dtype=torch.long),
            torch.tensor([pos], dtype=torch.long),
            torch.zeros(1, 1, hidden, dtype=torch.bfloat16),
        ]
    )[0]
    return out[0, -1, :].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tokenizer_path", required=True, help="HF tokenizer.json")
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--variant", default="e2b", choices=["e2b", "e4b"])
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer_path)
    hidden = HIDDEN_SIZE[args.variant]

    prog = Runtime.get().load_program(args.model_path)
    m = prog.load_method("text_decoder")

    ids = build_prompt_ids(tok, args.prompt)
    print(f"Prompt: {args.prompt!r}  ({len(ids)} tokens)")

    # Prefill: feed prompt tokens one at a time, KV cache accumulates.
    t0 = time.perf_counter()
    logits = None
    for pos, tid in enumerate(ids):
        logits = step(m, tid, pos, hidden)
    t_prefill = time.perf_counter() - t0

    # First generated token from the last prompt position's logits.
    next_tok = int(logits.argmax())
    generated = [next_tok]

    t0 = time.perf_counter()
    pos = len(ids)
    for _ in range(args.max_new_tokens - 1):
        if next_tok in STOP_TOKENS:
            break
        logits = step(m, next_tok, pos, hidden)
        next_tok = int(logits.argmax())
        generated.append(next_tok)
        pos += 1
    t_decode = time.perf_counter() - t0

    text = tok.decode(generated)
    print(f"\nOutput: {text}\n")
    n_dec = max(len(generated) - 1, 1)
    print(
        f"Prefill: {t_prefill:.1f}s ({len(ids)} tok, {len(ids)/t_prefill:.1f} tok/s) | "
        f"Decode: {t_decode:.1f}s ({n_dec} tok, {n_dec/max(t_decode,1e-6):.1f} tok/s)"
    )
    print("token ids:", generated)


if __name__ == "__main__":
    main()
