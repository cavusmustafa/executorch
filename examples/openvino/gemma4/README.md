# Gemma 4 (dense) — OpenVINO backend

OpenVINO export/inference for the dense Gemma 4 text decoders. The model
architecture is reused from `examples/models/gemma4` (E2B/E4B) and
`examples/models/gemma4_31b` (31B); the scripts here are the OpenVINO-specific
drivers, configs, and runners.

| Variant | Export | Generate |
|---|---|---|
| E2B / E4B | `examples/models/gemma4/export_gemma4.py --backend openvino` | `run_gemma4_ov.py` |
| 31B | `export_gemma4_31b_ov_nncf.py` | `run_gemma4_31b_gguf_cpu.py` (eager CPU) |

## E2B / E4B

Export the text decoder through the shared `export_gemma4.py` with the
OpenVINO backend selected. OpenVINO needs static shapes, so a single-token
decode model (`--static_seq_len 1`) is used for autoregressive generation.

```bash
python -m executorch.examples.models.gemma4.export_gemma4 \
    --checkpoint_path /path/to/gemma-4-e2b-it \
    --variant e2b --backend openvino --device CPU \
    --no-audio --no-vision --no-use_custom_sdpa --quantize none \
    --ov_quantize 4wo --static_seq_len 1 \
    --output_path gemma4_e2b_ov.pte
```

- `--device {CPU,GPU,NPU}` selects the OpenVINO target (validated on CPU + GPU).
- `--ov_quantize {none,4wo,8wo}` applies data-free NNCF weight compression.
- INT4 on GPU is both smallest and fastest; on CPU it trades speed for memory.

Generate (host-side sampling over the static decode PTE; uses the HF
`tokenizer.json`):

```bash
cd examples/openvino/gemma4
python run_gemma4_ov.py \
    --model_path gemma4_e2b_ov.pte \
    --tokenizer_path /path/to/gemma-4-e2b-it/tokenizer.json \
    --prompt "What is the capital of France?"
```

## 31B (dense)

> Status: the OpenVINO INT4 export pipeline is validated end-to-end on a layer
> subset (single OpenVINO delegate, INT4 weights consumed by the OV frontend).
> Full 60-layer export is memory-bound — `torch.export` of the full ~31B model
> needs more host RAM than a 62 GB box provides (≈128 GB recommended). Eager CPU
> generation from GGUF works at full scale.

### Eager CPU generation from GGUF (no CUDA / no mslk)

`run_gemma4_31b_gguf_cpu.py` builds the architecture from
`examples/models/gemma4_31b/model.py`, streams weights from a GGUF file, and
dequantizes each tensor to bf16 on CPU via the `gguf` package (any quant type —
Q4_K/Q5_K/Q6_K/Q8_0 — not just the q4_k/q6_k that ExecuTorch's GGUF loader
allowlists). Peak RAM ≈ 40 GB for the 31B.

```bash
cd examples/openvino/gemma4
python run_gemma4_31b_gguf_cpu.py \
    --gguf  /path/to/gemma-4-31B-it-Q4_K_M.gguf \
    --config /path/to/config.json \
    --tokenizer /path/to/tokenizer.json \
    --prompt "What is the capital of France?"
```

### Export to OpenVINO (INT4 weight compression via NNCF)

`export_gemma4_31b_ov_nncf.py` loads bf16 weights from GGUF, captures the
decode (T=1, static) graph, applies NNCF `compress_pt2e` (group-wise INT4 — the
representation the OpenVINO frontend keeps compressed), and lowers through
`OpenvinoPartitioner`. Host-side sampling (the model's Gumbel sampler is dropped
in favor of returning logits).

```bash
cd examples/openvino/gemma4
python export_gemma4_31b_ov_nncf.py \
    --gguf  /path/to/gemma-4-31B-it-Q4_K_M.gguf \
    --config /path/to/config.json \
    --output gemma4_31b_ov_int4.pte \
    --device CPU --quant 4wo --group-size 32
```

`--max-layers N` exports a layer subset (useful for validation on limited RAM).

### Direct compressed-INT4 OpenVINO graph (no bf16 dequant)

`gemma4_31b_ov_builder.py` assembles the full 60-layer dense Gemma4-31B directly
via the OpenVINO Python API from a GGUF checkpoint, keeping weights **INT4** the
whole way — `Constant(i4) -> Convert -> Multiply(per-channel scale) -> MatMul`,
which OpenVINO's decompression handling keeps compressed and decompresses inside
the FullyConnected kernel (CPU and GPU). No bf16 materialization, no `torch`,
`NNCF`, or `mslk`. Peak build memory stays near the compressed model size (~24 GB
for the 30.8B dense model), so it fits where a bf16 export (~62 GB) OOMs.

```python
from gemma4_31b_ov_builder import Builder
b = Builder(gguf_path="google_gemma-4-31B-it-Q4_K_M.gguf")
model = b.build(T=8)              # full 60 layers; use n_layers=N to slice
# serialize then compile (frees builder transients before compile):
import openvino as ov
ov.serialize(model, "g31.xml", "g31.bin")
cm = ov.Core().compile_model(ov.Core().read_model("g31.xml"), "CPU")
```

Per-channel INT4 is used to avoid the reshape-driven constant-folding that would
expand weights to f32 at compile time. For higher accuracy, use group-wise scale
with the `keep_const_precision` / `disable_constant_folding` rt_info markers.

### Full 60-layer `.pte` (blob injection — no bf16 export OOM)

`export_gemma4_31b_ov_pte.py` produces the **full 60-layer** model as an ExecuTorch
`.pte`, which the `compress_pt2e` path above cannot do (it materializes ~62 GB of
bf16 weights and OOMs; only a layer subset fits). ExecuTorch's OpenVINO backend
gets its delegate blob by calling `openvino_compile(gm, *args).export_model()`, so
we monkeypatch `openvino_compile` to compile+embed the `Builder`'s full 60-layer
INT4 `ov::Model` instead of the exported graph. A tiny stand-in torch model
(matching the `tokens[1] → [1, vocab]` signature) is exported for `.pte` structure;
the real weights come from the injected model. Runs end-to-end through the ET
runtime at ~4.2 tok/s CPU (matches the raw-OV IR).

```bash
cd examples/openvino/gemma4
# (a) build from GGUF directly — needs ~39 GB free RAM for the fp32-embedding build
python export_gemma4_31b_ov_pte.py \
    --gguf /path/to/gemma-4-31B-it-Q4_K_M.gguf --device GPU --layers 60 --run

# (b) or inject an already-serialized IR (from the builder above) — skips the build,
#     low RAM; recommended on memory-constrained boxes
python export_gemma4_31b_ov_pte.py \
    --ir g31.xml --device GPU --layers 60 --out gemma4_31b_ov_int4.pte --run
```

The `.pte` is large (~26 GB) because `lm_head` is tied to `token_embd` and the
embedding is stored fp32; compressing the embedding to i4 would roughly halve it.

### Build from a pre-quantized INT4 checkpoint (HQQ / torchao) — no GGUF requant

`gemma4_31b_hqq_builder.py` (`HQQBuilder`) is an alternative weight source: instead of
dequantizing GGUF Q4_K_M → fp32 and re-quantizing with a naive per-channel scale, it
loads a pre-quantized **HQQ / torchao Int4** checkpoint (`gemma-4-31B-it-HQQ-INT4`) and
maps its int4 weights **directly** to OpenVINO's compressed decompression pattern —
`Constant(u4|i8) → Convert → Subtract(zp) → Multiply(scale) → MatMul`, folded by the
plugin into one compressed `FullyConnected`, with **no fp32 weight ever materialized**.
It subclasses `Builder`, reusing the identical graph (attention, dual RoPE, QK-norm,
softcap…) and overriding only the weight-I/O layer.

The checkpoint mixes two torchao quant types, both handled (metadata-driven):
`Int4Tensor` (q/k/o/gate/up — u8 nibble-packed, scale/zp `[ng,out]`) and
`IntxUnpackedToInt8Tensor` (v/down + embedding — int8 unpacked, scale/zp `[out,ng]`).
Per-linear dequant matches the GGUF of the same model at cosine 0.995–0.9998.

```bash
cd examples/openvino/gemma4
# full 60 layers on a 62 GB box: add --per-channel (see note below)
python export_gemma4_31b_ov_pte.py \
    --hqq /path/to/gemma-4-31B-it-HQQ-INT4/model.safetensors --per-channel \
    --device GPU --layers 60 --out gemma4_31b_hqq_int4.pte --run
```

**`--per-channel` for full depth.** The checkpoint's native HQQ grouping is group-32
(one scale/zero-point per 32 input elements). At full 60-layer depth that group-wise
decompression subgraph (168 groups × 412 linears of reshape+Subtract+Multiply) inflates
the OpenVINO **compile** working set past 62 GB and OOMs — a layer *subset* fits, the
full model doesn't. `--per-channel` re-quantizes each linear to a single `[O,1]` i4 scale
(reusing the GGUF builder's per-channel `_int4_lin`), collapsing that subgraph: the
serialized IR drops from ~26 GB → 16.8 GB, compile stays ~13–44 GB, and the full 60-layer
`.pte` (21.7 GB) builds and runs end-to-end at **4.22 tok/s CPU** (matching the GGUF path).
It trades a little accuracy (one scale/row vs HQQ's calibrated group-32) for fitting the
box. Without `--per-channel` you get the more accurate group-32 model, but full depth needs
more host RAM (~96 GB); a subset (`--layers N`) still works for validation.

This is the "use the int4 checkpoint directly" route (Path B). The wrong route (Path A)
is `torch.export` of an HQQ/GPTQ torch model whose loader dequantizes on the host CPU —
that both loses the int4 packing (back to the bf16 OOM) and doesn't reliably hit the
fused compressed kernel. `poc_hqq_ov_pattern.py` proves the per-linear primitive
(numeric 5e-7, weight stays u4, one fused `FullyConnected` on CPU+GPU).

## Qwen 3.5 MoE (GatedDeltaNet) — reference scripts

`gdn_scan_module.py`, `gdn_full_layer.py`, `moe_scan_module.py`,
`qwen35_full_model.py`, and `qwen35_gguf_loader.py` are the validated building
blocks for running Qwen3.5-MoE's Mamba-style GatedDeltaNet on OpenVINO. The
recurrence is written with PyTorch `scan`, which the companion OpenVINO branch
(`translate_scan_fx` + `FuseScanGDN`) lowers to the native `GatedDeltaNet`
op/kernel. Requires that custom OpenVINO build; `backends/openvino/partitioner.py`
whitelists the `scan`/`while_loop`/`cond` HOPs so they partition to the backend.

### Real-weight Qwen3.5-35B-A3B — reproduce ~37 tok/s on Intel GPU

The scripts above use random weights (architecture/throughput validation). To run
the **real** model end-to-end with the real GPTQ checkpoint and reproduce the
tok/s of the official `OpenVINO/Qwen3.6-35B-A3B-int4-ov` IR (37–45 tok/s on a
Panther Lake Arc GPU), use these four files:

| File | Role |
|---|---|
| `qwen35_real_weights.py` | Loads the `Qwen3.5-35B-A3B-GPTQ-Int4` safetensors; dequantizes GPTQ experts; exposes per-name tensors. |
| `qwen35_real_decode.py`  | Faithful single-token decode model (GDN + full-attn + MoE), numerically validated vs HF (GDN 2.2e-8, attn 1.2e-7). |
| `qwen35_pool_moe.py`     | The **fused-FC MoE** + full-model `build()` (the optimization: 13.5 → 37.6 tok/s). |
| `bench_qwen35_real_ov.py` | Raw-OV benchmark (reference number, native fused GDN kernel). |
| `export_qwen35_real_pte.py` | Exports the ExecuTorch `.pte` with weight-only INT4 that stays compressed (32.2 tok/s at 40L). |

**Prerequisites** (all three needed for the numbers; the `.pte` export needs the
first two, the raw-OV native-GDN kernel needs the custom build):

1. The **custom OpenVINO build** with `scan`/`FuseScanGDN` (the companion branch
   above), on `PYTHONPATH`/`LD_LIBRARY_PATH`, and `OPENVINO_LIB_PATH` →
   its `libopenvino_c.so` so the ET runtime dlopens it.
2. The **GPTQ checkpoint** (~24.5 GB) and an Intel GPU. ~62 GB host RAM
   (`torch.export` of the full 40-layer graph peaks ~58 GB).

```bash
# checkpoint
huggingface-cli download Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 --local-dir ./qwen35_gptq
export QWEN35_GPTQ_DIR=$PWD/qwen35_gptq

# point at the custom OpenVINO build (scan + FuseScanGDN)
export PYTHONPATH=/path/to/openvino/build/python:$PYTHONPATH
export LD_LIBRARY_PATH=/path/to/openvino/build:$LD_LIBRARY_PATH
export OPENVINO_LIB_PATH=/path/to/openvino/build/libopenvino_c.so

cd examples/openvino/gemma4

# raw-OV reference (native fused GDN kernel) — expect ~37.6 tok/s at 40L INT4 on GPU
python bench_qwen35_real_ov.py --layers 40 --device GPU --int4

# ExecuTorch .pte (weight-only INT4, GDN unrolled) — expect ~32 tok/s at 40L on GPU
GDN_UNROLL=1 python export_qwen35_real_pte.py --layers 40 --device GPU --run
```

Notes on the two numbers:

- **`bench_qwen35_real_ov.py` (37.6 tok/s)** is the raw-OpenVINO path. Its speed
  came from replacing the naive top-k MoE (`index_select` + per-expert `bmm`,
  ~73 % of decode time) with the **fully-fused MoE** in `qwen35_pool_moe.py` —
  all experts as three big `FullyConnected` ops, top-k masking on the
  activations. `DYNAMIC_QUANTIZATION_GROUP_SIZE=0` keeps the tiny router/GDN
  matmuls off the slow `ocl:ref` path. All GatedDeltaNet layers fuse to the
  native kernel (`runtime GatedDeltaNet nodes: 30`).
- **`export_qwen35_real_pte.py` (32.2 tok/s)** runs the same graph through the
  ExecuTorch OpenVINO backend. ET's own `compress_pt2e` INT4 gives only 12.6
  tok/s (its `quantized_decomposed` encoding compiles to a dense fp16 FC), so the
  driver instead runs **raw NNCF `compress_weights(INT4_SYM)` inside the backend's
  compile step** (monkeypatching `openvino_compile` — no ET/OV source edits). Two
  subtleties, both handled in the script and explained in its docstring:
  `apply_moc_transformations` folds the edge-decomposition's `permute_copy`
  Transpose into `MatMul.transpose_b` (so weight-only INT4 sticks *and* the blob
  re-imports in the ET runtime), and `GDN_UNROLL=1` unrolls the single decode step
  (ET can't lower the `scan` HOP — this costs the ~5 tok/s vs raw-OV, since the
  native fused GDN kernel doesn't fire in the `.pte`).

Sparse MoE is pool-size-independent at decode (only top-8 of 256 experts execute),
so `--experts N` materializes a smaller pool while every executed dimension stays
real — the full 40 layers then fit in RAM. `--layers 4` is a fast smoke test.

## Notes

- OpenVINO needs static shapes (no `SymInt` graph inputs). The E2B/E4B path sets
  `use_index_copy_for_kv_cache` to avoid the data-dependent `SymInt` from
  `input_pos[0].item()` cache slicing; the 31B model already uses `index_copy_`
  in both its `RingKVCache` (sliding) and `Gemma4KVCache` (full) paths.
- INT4 here is group-wise. OpenVINO consumes NNCF's compressed representation;
  torchao's `dequantize_affine` (what GGUF→`IntxUnpackedToInt8Tensor` produces)
  does not map to OpenVINO's per-channel `quantized_decomposed` ops, so NNCF is
  the path that keeps weights compressed through lowering.
