"""Export the FULL 60-layer Gemma4-31B to an ExecuTorch .pte through the OpenVINO backend,
WITHOUT the bf16 torch.export OOM.

The existing export_gemma4_31b_ov_nncf.py path does GGUF -> bf16 model -> torch.export ->
compress_pt2e, which materializes ~62 GB of bf16 weights and OOMs at 60 layers on a 62 GB
box (only a 4-layer subset .pte fits). Meanwhile gemma4_31b_ov_builder.Builder streams the
GGUF and assembles the full 60-layer model as a compressed INT4 ov::Model directly (~24 GB,
fits) -- but that is a raw OV IR, not a .pte.

This driver bridges the two, reusing the Qwen injection mechanism: ExecuTorch's OpenVINO
preprocess just calls `openvino_compile(gm, *args).export_model()` to get the delegate blob.
We export a TINY stand-in torch model (matching I/O signature -> cheap, no bf16 blowup) and
monkeypatch openvino_compile to compile+embed the BUILDER's full 60-layer INT4 ov::Model
instead of the exported stand-in graph. The .pte then carries the real compiled model.
(Feasibility proven in poc_inject_ov_blob.py: an independently-built ov::Model executes
correctly through the ET runtime.)

  QWEN... not needed. Run with the source OV build for parity with the other examples:
  PYTHONPATH=<ov_build>/python LD_LIBRARY_PATH=<ov_build> \
  OPENVINO_LIB_PATH=<ov_build>/libopenvino_c.so \
  python export_gemma4_31b_ov_pte.py \
     --gguf /path/to/gemma-4-31B-it-Q4_K_M.gguf --device GPU --layers 60 --run
"""
import argparse, gc, os, time, numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--gguf", help="GGUF checkpoint (required unless --ir is given)")
ap.add_argument("--ir", help="pre-built OpenVINO IR .xml (tokens[1] i64 -> [1,VOCAB] f32); "
                             "skips the GGUF build entirely (avoids the build-time OOM)")
ap.add_argument("--device", default="GPU")
ap.add_argument("--layers", type=int, default=60, help="60 = full model")
ap.add_argument("--group-size", type=int, default=32)
ap.add_argument("--out", default="gemma4_31b_ov_int4.pte")
ap.add_argument("--hqq", help="pre-quantized HQQ/torchao Int4 safetensors: build the ov::Model "
                              "directly from its int4 weights (no GGUF requant) — see "
                              "gemma4_31b_hqq_builder.HQQBuilder")
ap.add_argument("--per-channel", action="store_true",
                help="(with --hqq) re-quantize each linear to a single [O,1] i4 scale instead of "
                     "HQQ group-32; much smaller compile working set so full 60L fits in 62 GB")
ap.add_argument("--run", action="store_true", help="decode-benchmark the .pte via the ET runtime")
args = ap.parse_args()
assert args.gguf or args.ir or args.hqq, "pass --gguf / --hqq to build, or --ir to inject a pre-built IR"

VOCAB = 262144  # Gemma4-31B token_embd rows (validated from GGUF)

# ---------------------------------------------------------------------------
# 1. Build the full compressed INT4 ov::Model directly from GGUF (streams weights, ~24 GB).
# ---------------------------------------------------------------------------
from gemma4_31b_ov_builder import Builder
import openvino as ov

# Mirror the raw-OV IR path's memory strategy: build -> serialize to disk -> free the Python
# ov::Model -> read_model (mmap, low RAM) -> compile. Holding the built model AND compiling it
# in-process peaks too high for a 62 GB box; the build->serialize->free->read split keeps peak
# near the compressed model size (~24 GB), which is why the raw-OV IR path fits.
if args.ir:
    # Inject a pre-built IR directly -- no GGUF build, so no build-time OOM. This is the
    # recommended path for the full 60-layer model on a memory-constrained box: build the IR
    # once with the raw-OV path (build->serialize->free->read peaks ~39 GB, fits in 62 GB),
    # then reuse the serialized IR here as many times as needed.
    _XML = args.ir
    print(f"using pre-built IR {_XML} ({os.path.getsize(os.path.splitext(_XML)[0] + '.bin')/1e9:.2f} GB .bin)", flush=True)
else:
    # Build directly from a checkpoint. Mirror the raw-OV IR path's memory strategy:
    # build -> serialize to disk -> free the Python ov::Model -> read_model (mmap, low RAM)
    # -> compile. Holding the built model AND compiling it in-process peaks too high for a
    # 62 GB box; the split keeps peak near the compressed model size.
    if args.hqq:
        # Build from the pre-quantized HQQ int4 checkpoint (no GGUF requant, no fp32 weights).
        from gemma4_31b_hqq_builder import HQQBuilder
        _b = HQQBuilder(st_path=args.hqq, gs=args.group_size, per_channel=args.per_channel)
        _src = f"HQQ int4 ({args.hqq})"
    else:
        _b = Builder(gguf_path=args.gguf, gs=args.group_size)
        _src = f"GGUF ({args.gguf})"
    _XML = os.path.splitext(args.out)[0] + "_ovir.xml"
    _BIN = os.path.splitext(args.out)[0] + "_ovir.bin"
    print(f"building {args.layers}-layer Gemma4-31B ov::Model from {_src} ...", flush=True)
    t0 = time.perf_counter()
    _built = _b.build(T=1, n_layers=args.layers)     # decode: T=1 -> tokens[1] i64 -> [1, VOCAB] f32
    print(f"  built in {time.perf_counter()-t0:.0f}s | inputs={[p.get_any_name() for p in _built.inputs]}", flush=True)
    print(f"  serializing to {_XML} then freeing Python model ...", flush=True)
    ov.serialize(_built, _XML, _BIN)
    del _built, _b
    gc.collect()
    print(f"  serialized ({os.path.getsize(_BIN)/1e9:.2f} GB .bin); peak now bounded by read_model", flush=True)

# ---------------------------------------------------------------------------
# 2. Monkeypatch openvino_compile to compile+embed BUILT instead of the stand-in graph.
# ---------------------------------------------------------------------------
import openvino.frontend.pytorch.torchdynamo.compile as _ovc
from openvino.frontend.pytorch.torchdynamo.backend_utils import _get_device, _get_config


def openvino_compile_inject(gm, *cargs, model_hash_str=None, options=None):
    core = ov.Core()
    device = _get_device(options)
    config = _get_config(options) or {}
    config.setdefault("PERFORMANCE_HINT", "LATENCY")
    config.setdefault("DYNAMIC_QUANTIZATION_GROUP_SIZE", "0")   # weight-only INT4 (matches raw-OV)
    print(f"  [inject] read_model + compile builder's {args.layers}-layer INT4 model on {device} (config={config})", flush=True)
    model = core.read_model(_XML)                    # mmap-backed; low RAM
    return core.compile_model(model, device, config)


_ovc.openvino_compile = openvino_compile_inject
import executorch.backends.openvino.preprocess as _pp
_pp.openvino_compile = openvino_compile_inject
print("patched openvino_compile -> inject builder model", flush=True)


# ---------------------------------------------------------------------------
# 3. Tiny stand-in torch model with the SAME I/O signature as BUILT:
#    tokens[1] int64 -> logits[1, VOCAB] float32. Cheap to export (no 31B weights).
# ---------------------------------------------------------------------------
class StandIn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # a trivial embedding-sized table so torch.export produces tokens->[1,VOCAB]; never executed
        self.tab = torch.nn.Parameter(torch.zeros(1, VOCAB), requires_grad=False)

    def forward(self, tokens):
        # tokens: [1] int64 -> broadcast the (unused) table to [1, VOCAB]
        return self.tab + tokens.to(torch.float32).sum() * 0.0


m = StandIn().eval()
ex = (torch.zeros(1, dtype=torch.long),)
print("torch.export (tiny stand-in) ...", flush=True)
ep = torch.export.export(m, ex, strict=True)

from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from executorch.backends.openvino.partitioner import OpenvinoPartitioner
from executorch.exir.backend.backend_details import CompileSpec

specs = [CompileSpec("device", args.device.encode())]
print(f"lowering through OpenvinoPartitioner (device={args.device}) ...", flush=True)
edge = to_edge_transform_and_lower(
    ep, partitioner=[OpenvinoPartitioner(specs)],
    compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True))
progs = getattr(edge, "_edge_programs", {})
ndeleg = sum(1 for _n, mm in progs.items() for node in mm.graph_module.graph.nodes
             if node.op == "call_function" and "call_delegate" in str(node.target))
print(f"OpenVINO delegate partitions: {ndeleg}", flush=True)
et = edge.to_executorch()
with open(args.out, "wb") as f:
    et.write_to_file(f)
print(f"wrote {args.out} ({os.path.getsize(args.out)/1e9:.2f} GB)", flush=True)

if args.run:
    from executorch.runtime import Runtime
    method = Runtime.get().load_program(args.out).load_method("forward")
    tok = torch.zeros(1, dtype=torch.long)
    for _ in range(5):
        method.execute([tok])
    N = 30
    t = time.perf_counter()
    for _ in range(N):
        method.execute([tok])
    dt = time.perf_counter() - t
    print(f"[ET/{args.device}] 31B {args.layers}L T=1 decode: {dt/N*1000:.2f} ms/tok  {N/dt:.2f} tok/s", flush=True)
