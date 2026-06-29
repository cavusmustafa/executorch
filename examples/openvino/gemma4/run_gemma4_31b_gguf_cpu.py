#!/usr/bin/env python3
"""CPU eager sanity check for Gemma4-31B from a GGUF file (any quant type).

ExecuTorch's gguf.py only allowlists q4_k/q6_k, but the `gguf` PyPI package
dequantizes every type (Q5_K, Q8_0, ...) on CPU. This loader streams each GGUF
tensor, dequantizes to bf16 via gguf.dequantize, remaps the name to the model
FQN, and assigns it onto a meta-device Gemma4_31B — so peak RAM stays near the
final bf16 model size, never holding the raw file and a full copy at once.

No mslk, no CUDA. Architecture comes from examples/models/gemma4_31b/model.py;
GGUF supplies only weights.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from executorch.examples.models.gemma4_31b.model import (
    Gemma4_31B,
    Gemma4_31BConfig,
    materialize_runtime_buffers,
)
from executorch.examples.models.gemma4_31b.gguf_loader import gguf_to_model_key


def _dequant_gguf_tensor(gtensor, gguf_mod):
    """Dequantize one GGUFReader tensor to a bf16 torch tensor (N, K) on CPU."""
    from gguf import GGMLQuantizationType as QT

    shape = list(reversed(gtensor.shape.tolist()))
    ttype = int(gtensor.tensor_type)
    if ttype == QT.F32:
        arr = np.frombuffer(gtensor.data, dtype=np.float32).reshape(shape)
        return torch.from_numpy(arr.copy()).to(torch.bfloat16)
    if ttype == QT.F16:
        arr = np.frombuffer(gtensor.data, dtype=np.float16).reshape(shape)
        return torch.from_numpy(arr.copy()).to(torch.bfloat16)
    # Quantized: dequantize via the gguf package (CPU, supports all k-quants).
    deq = gguf_mod.dequantize(np.ascontiguousarray(gtensor.data), QT(ttype))
    return torch.from_numpy(np.ascontiguousarray(deq)).reshape(shape).to(torch.bfloat16)


def load_gguf_cpu(gguf_path: str, config_path: str, max_seq_len: int):
    import gguf

    config = Gemma4_31BConfig.from_hf_config(config_path)
    config.max_seq_len = max_seq_len

    print(f"Building Gemma4_31B on meta (layers={config.num_hidden_layers})...")
    with torch.device("meta"):
        model = Gemma4_31B(config)

    print(f"Streaming + dequantizing GGUF weights from {gguf_path}...")
    reader = gguf.GGUFReader(gguf_path)
    embed_weight = None
    assigned = 0
    t0 = time.perf_counter()
    for gt in reader.tensors:
        model_key = gguf_to_model_key(gt.name)
        if model_key is None:
            continue
        w = _dequant_gguf_tensor(gt, gguf)
        parts = model_key.rsplit(".", 1)
        parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
        attr = parts[-1]
        cur = getattr(parent, attr, None)
        if isinstance(cur, nn.Parameter) or (cur is not None and cur.device.type == "meta"
                                             and attr in ("weight",)):
            setattr(parent, attr, nn.Parameter(w, requires_grad=False))
        else:
            parent.register_buffer(attr, w, persistent=False)
        if model_key == "embed_tokens.weight":
            embed_weight = w
        assigned += 1

    # Tied lm_head.
    if getattr(model.lm_head, "weight", None) is None or model.lm_head.weight.device.type == "meta":
        if embed_weight is not None:
            model.lm_head.weight = nn.Parameter(embed_weight, requires_grad=False)

    missing = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if missing:
        raise RuntimeError(f"Unassigned params ({len(missing)}): {missing[:6]}")
    print(f"  Assigned {assigned} weights in {time.perf_counter()-t0:.1f}s")

    materialize_runtime_buffers(model, dtype=torch.bfloat16, device="cpu")
    model.eval()
    return model, config


def apply_chat_template(p):
    return "<|turn>user\n" + p + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"


def generate(model, tok, prompt, max_new_tokens, temperature, eos_ids, bos_id=2):
    ids = tok.encode(prompt).ids
    if not ids or ids[0] != bos_id:
        ids = [bos_id] + ids
    temp = torch.tensor([max(temperature, 1e-6)], dtype=torch.float32)
    sampled = None
    with torch.no_grad():
        t0 = time.perf_counter()
        for i, tid in enumerate(ids):
            sampled = model(torch.tensor([[tid]], dtype=torch.long),
                            torch.tensor([i], dtype=torch.long), temp)
        tp = time.perf_counter() - t0
        nxt = int(sampled.item()); gen = [nxt]; n = len(ids)
        t0 = time.perf_counter()
        for i in range(max_new_tokens - 1):
            if nxt in eos_ids:
                break
            sampled = model(torch.tensor([[nxt]], dtype=torch.long),
                            torch.tensor([n + i], dtype=torch.long), temp)
            nxt = int(sampled.item()); gen.append(nxt)
        td = time.perf_counter() - t0
    print(f"[prefill {len(ids)} tok {tp:.1f}s | decode {len(gen)} tok {td:.1f}s "
          f"{len(gen)/max(td,1e-6):.2f} tok/s]")
    return tok.decode(gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--config", required=True, help="config.json (HF) for architecture")
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json")
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--raw-prompt", action="store_true")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    model, _ = load_gguf_cpu(args.gguf, args.config, args.max_seq_len)
    prompt = args.prompt if args.raw_prompt else apply_chat_template(args.prompt)
    print(f"\nPrompt: {args.prompt}\n" + "-" * 40)
    out = generate(model, tok, prompt, args.max_new_tokens, args.temperature,
                   eos_ids={1, 50, 106})
    print(out)


if __name__ == "__main__":
    main()
