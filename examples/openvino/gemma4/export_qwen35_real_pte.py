"""Export the REAL-weight Qwen3.5-35B-A3B decode model to a .pte through the ExecuTorch
OpenVINO backend, with INT4 weights that actually stay compressed (32.2 tok/s at 40L on the
PTL GPU, vs 12.6 tok/s for ET's default compress_pt2e path).

Why not compress_pt2e? ET's OpenVINOQuantizer(INT4WO_SYM) emits a quantized_decomposed int4
encoding that compiles to a dense fp16 FullyConnected. Instead we keep the export fp32 and
run raw nncf.compress_weights(INT4_SYM) on the ov.Model inside the OpenVINO backend's compile
step, which yields the marked weight-only pattern the GPU plugin folds into a compressed FC.

Two subtleties make it work (both handled below):
  1. ET's edge pipeline lowers aten.linear -> permute_copy(weight) + mm, putting a Transpose
     between the decompress subgraph and the MatMul weight port -> the fuse-to-compressed-FC
     pass bails and weights silently decompress to fp16. `apply_moc_transformations` runs
     TransposeMatMul, folding the Transpose into MatMul's transpose_b=true (canonical
     aten.linear layout) so compression sticks AND the blob re-imports in the ET runtime.
     (Folding the Transpose into the *constant* instead yields a layout whose compressed
     MatMul primitive fails to re-import: oneDNN "could not create a primitive descriptor".)
  2. LATENCY hint + DYNAMIC_QUANTIZATION_GROUP_SIZE=0 keeps the tiny router/GDN matmuls off
     the slow ocl:ref DynamicQuantize path.

We install this by monkeypatching the `openvino_compile` symbol that the OpenVINO backend's
preprocess.py imports -- no ExecuTorch or OpenVINO source files are modified.

Requires the SOURCE-built OpenVINO (scan + FuseScanGDN) on PYTHONPATH/LD_LIBRARY_PATH and
OPENVINO_LIB_PATH -> that build's libopenvino_c.so so the ET runtime dlopens it. GDN_UNROLL=1
unrolls the single decode step (ET can't lower the scan HOP; the native fused GDN kernel
therefore doesn't fire in the .pte, costing ~5 tok/s vs raw-OV):

  PYTHONPATH=<ov_build>/python LD_LIBRARY_PATH=<ov_build> \
  OPENVINO_LIB_PATH=<ov_build>/libopenvino_c.so GDN_UNROLL=1 \
  QWEN35_GPTQ_DIR=/path/to/qwen35_gptq \
  python export_qwen35_real_pte.py --layers 40 --device GPU --run
"""
import argparse, os, time, numpy as np, torch

from qwen35_real_weights import RealWeights
from qwen35_pool_moe import build

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=40)
ap.add_argument("--experts", type=int, default=8)
ap.add_argument("--device", default="GPU")
ap.add_argument("--out", default="qwen35_real_int4.pte")
ap.add_argument("--group-size", type=int, default=128)
ap.add_argument("--gptq-dir", default=None, help="GPTQ checkpoint dir (else $QWEN35_GPTQ_DIR)")
ap.add_argument("--run", action="store_true", help="also decode-benchmark the .pte via the ET runtime")
args = ap.parse_args()

# ---------------------------------------------------------------------------
# Monkeypatch the OpenVINO backend's openvino_compile with a raw-NNCF-INT4 variant that
# fuses the transpose, compresses to weight-only INT4, and forces the LATENCY config.
# ---------------------------------------------------------------------------
import openvino.frontend.pytorch.torchdynamo.compile as _ovc
from openvino import Core, PartialShape, Type
from openvino.frontend import FrontEndManager
from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder
from openvino.frontend.pytorch.torchdynamo.backend_utils import _get_device, _get_config
from openvino._offline_transformations import apply_moc_transformations
import nncf
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters

_GS = args.group_size
_DTYPE = {torch.float32: Type.f32, torch.int64: Type.i64, torch.int32: Type.i32}


def openvino_compile_int4(gm, *cargs, model_hash_str=None, options=None):
    core = Core()
    device = _get_device(options)
    fe = FrontEndManager().load_by_framework("pytorch")
    decoder = TorchFXPythonDecoder(gm)                    # hold ref (avoid GC of the decoder)
    om = fe.convert(fe.load(decoder))

    for idx, input_data in enumerate(cargs):
        if isinstance(input_data, int):
            om.inputs[idx].get_node().set_element_type(Type.i64)
            om.inputs[idx].get_node().set_partial_shape(PartialShape([1]))
        else:
            om.inputs[idx].get_node().set_element_type(_DTYPE[input_data.dtype])
            om.inputs[idx].get_node().set_partial_shape(PartialShape(list(decoder.input_shapes[idx])))
    om.validate_nodes_and_infer_types()

    # (1) fold Transpose into MatMul.transpose_b so weight-only INT4 compression sticks AND
    #     the blob re-imports cleanly in the ET runtime.
    apply_moc_transformations(om, False)

    # (2) raw NNCF INT4_SYM weight-only compression, excluding tiny [*,1] router matmuls
    #     (they fall to a slow ocl:ref path).
    tiny = [o.get_friendly_name() for o in om.get_ordered_ops()
            if o.get_type_name() == "MatMul" and o.get_output_partial_shape(0)[-1].get_length() == 1]
    print(f"  [int4] compress_weights INT4_SYM gs={_GS}, excluding {len(tiny)} tiny gate MatMuls", flush=True)
    om = nncf.compress_weights(
        om, mode=nncf.CompressWeightsMode.INT4_SYM, group_size=_GS, ratio=1.0,
        ignored_scope=nncf.IgnoredScope(names=tiny) if tiny else None,
        advanced_parameters=AdvancedCompressionParameters(
            group_size_fallback_mode=nncf.GroupSizeFallbackMode.ADJUST))

    # (3) LATENCY + no dynamic activation quant (the config that matches the raw-OV 37.6 path)
    config = _get_config(options) or {}
    config.setdefault("PERFORMANCE_HINT", "LATENCY")
    config.setdefault("DYNAMIC_QUANTIZATION_GROUP_SIZE", "0")
    return core.compile_model(om, device, config)


_ovc.openvino_compile = openvino_compile_int4
import executorch.backends.openvino.preprocess as _pp
_pp.openvino_compile = openvino_compile_int4            # patch the name preprocess.py bound at import
print("patched openvino_compile -> raw-NNCF-INT4 variant", flush=True)

# ---------------------------------------------------------------------------
w = RealWeights(gptq_dir=args.gptq_dir)
print(f"building {args.layers}L pool={args.experts} real model ...", flush=True)
m = build(w, args.layers, args.experts)
ex = m.example_inputs()

print("torch.export (fp32, NO compress_pt2e) ...", flush=True)
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
    tens = [(t if torch.is_tensor(t) else torch.tensor(t)).float() for t in ex]
    for _ in range(5):
        method.execute(tens)
    N = 30
    t = time.perf_counter()
    for _ in range(N):
        method.execute(tens)
    dt = time.perf_counter() - t
    print(f"[ET/{args.device}] {args.layers}L T=1 decode: {dt/N*1000:.2f} ms/tok  {N/dt:.2f} tok/s", flush=True)
