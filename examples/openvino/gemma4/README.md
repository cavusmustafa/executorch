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

## Qwen 3.5 MoE (GatedDeltaNet) — reference scripts

`gdn_scan_module.py`, `gdn_full_layer.py`, `moe_scan_module.py`,
`qwen35_full_model.py`, and `qwen35_gguf_loader.py` are the validated building
blocks for running Qwen3.5-MoE's Mamba-style GatedDeltaNet on OpenVINO. The
recurrence is written with PyTorch `scan`, which the companion OpenVINO branch
(`translate_scan_fx` + `FuseScanGDN`) lowers to the native `GatedDeltaNet`
op/kernel. Requires that custom OpenVINO build; `backends/openvino/partitioner.py`
whitelists the `scan`/`while_loop`/`cond` HOPs so they partition to the backend.

## Notes

- OpenVINO needs static shapes (no `SymInt` graph inputs). The E2B/E4B path sets
  `use_index_copy_for_kv_cache` to avoid the data-dependent `SymInt` from
  `input_pos[0].item()` cache slicing; the 31B model already uses `index_copy_`
  in both its `RingKVCache` (sliding) and `Gemma4KVCache` (full) paths.
- INT4 here is group-wise. OpenVINO consumes NNCF's compressed representation;
  torchao's `dequantize_affine` (what GGUF→`IntxUnpackedToInt8Tensor` produces)
  does not map to OpenVINO's per-channel `quantized_decomposed` ops, so NNCF is
  the path that keeps weights compressed through lowering.
