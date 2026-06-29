#!/usr/bin/env python3
"""Export Gemma4-31B (GGUF) to OpenVINO via NNCF compress_pt2e (INT4/INT8 group-wise).

This is the path that produces OV-NATIVE compressed weights (the frontend keeps
them compressed instead of folding to bf16, and group-wise INT4 — which torchao's
dequantize_affine couldn't map to OV's per-channel ops — is expressed in OV's own
representation).

Flow: GGUF -> bf16 model -> torch.export().module() -> compress_pt2e(OpenVINOQuantizer)
-> re-export -> OpenvinoPartitioner. Decode-only T=1 static (host-side sampling).

NOTE: the bf16 capture is the memory-heavy step for the full 60-layer model on a
62 GB box; --max-layers slices it for validation.
"""

import argparse
import gc
import os
import time

import torch
import torch.nn as nn

# Run from this directory (examples/openvino/gemma4), per the OpenVINO examples
# convention; run_gemma4_31b_gguf_cpu.py lives alongside this script.
from run_gemma4_31b_gguf_cpu import load_gguf_cpu


class LogitsWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, tokens, input_pos):
        m = self.m
        x = m.embed_tokens(tokens) * m.embed_normalizer
        sliding_mask, full_mask = m._build_masks(input_pos)
        for layer in m.layers:
            x = layer(x, input_pos, sliding_mask, full_mask)
        x = m.norm(x)
        last = m.lm_head(x[:, -1, :]).float()
        cap = m.logit_softcap.float()
        last = torch.tanh(last / cap) * cap
        return last.unsqueeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--max-layers", type=int, default=0)
    ap.add_argument("--quant", default="4wo", choices=["4wo", "8wo"])
    ap.add_argument("--group-size", type=int, default=32)
    args = ap.parse_args()

    from executorch.backends.openvino.partitioner import OpenvinoPartitioner
    from executorch.backends.openvino.quantizer import OpenVINOQuantizer, QuantizationMode
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch.exir.backend.backend_details import CompileSpec
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from nncf.experimental.torch.fx import compress_pt2e

    model, config = load_gguf_cpu(args.gguf, args.config, args.max_seq_len)
    if args.max_layers and args.max_layers < len(model.layers):
        model.layers = model.layers[: args.max_layers]
        print(f"  TRUNCATED to {args.max_layers} layers")
    model = LogitsWrapper(model).eval()

    example = (torch.tensor([[1]], dtype=torch.long), torch.tensor([0], dtype=torch.long))

    print("Capturing (torch.export -> module)...")
    t0 = time.perf_counter()
    with torch.no_grad():
        captured = torch.export.export(model, example, strict=True).module()
    print(f"  captured in {time.perf_counter()-t0:.1f}s")
    del model
    gc.collect()

    if args.quant == "4wo":
        quantizer = OpenVINOQuantizer(mode=QuantizationMode.INT4WO_SYM, group_size=args.group_size, ratio=1)
    else:
        quantizer = OpenVINOQuantizer(mode=QuantizationMode.INT8WO_ASYM, group_size=-1)

    print(f"NNCF compress_pt2e ({args.quant}, gs={args.group_size})...")
    t0 = time.perf_counter()
    compressed = compress_pt2e(captured, quantizer=quantizer, dataset=None)
    print(f"  compressed in {time.perf_counter()-t0:.1f}s")
    del captured
    gc.collect()

    print("Re-exporting compressed graph...")
    with torch.no_grad():
        ep = torch.export.export(compressed, example, strict=True)
    del compressed
    gc.collect()

    print(f"Lowering to OpenVINO ({args.device})...")
    t0 = time.perf_counter()
    edge = to_edge_transform_and_lower(
        ep,
        partitioner=[OpenvinoPartitioner([CompileSpec("device", args.device.encode())])],
        compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
    )
    et = edge.to_executorch(config=ExecutorchBackendConfig())
    print(f"  lowered in {time.perf_counter()-t0:.1f}s")
    with open(args.output, "wb") as f:
        et.write_to_file(f)
    print(f"Wrote {args.output} ({os.path.getsize(args.output)/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
