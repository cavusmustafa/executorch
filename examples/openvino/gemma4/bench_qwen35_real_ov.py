"""Benchmark the REAL-weight Qwen3.5-35B-A3B decode graph on OpenVINO (CPU/GPU), raw-OV path.

This is the reference for the optimized number: the fully-fused MoE (qwen35_pool_moe.PoolMoE)
takes the 40-layer real graph from 13.5 -> 37.6 tok/s on the PTL GPU with INT4 weights,
matching the official OpenVINO/Qwen3.6-35B-A3B-int4-ov IR (37-45 tok/s).

Requires the SOURCE-built OpenVINO with the GatedDeltaNet scan support (scan translator +
FuseScanGDN), so the delta-rule recurrence fuses to the native GatedDeltaNet kernel:

  PYTHONPATH=<ov_build>/python LD_LIBRARY_PATH=<ov_build> \
  QWEN35_GPTQ_DIR=/path/to/qwen35_gptq \
  python bench_qwen35_real_ov.py --layers 40 --device GPU --int4

Numbers on Core Ultra X7 358H (Panther Lake), Arc/Xe3 iGPU, real weights, 40 layers:
  CPU  fp32 4.4 / INT4 5.7 tok/s      GPU fp32 10.5 / INT4 37.6 tok/s (fused MoE, NO_DYNQUANT)
"""
import argparse, time, numpy as np, torch
from collections import Counter

from qwen35_real_weights import RealWeights
from qwen35_pool_moe import build

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=40)
ap.add_argument("--experts", type=int, default=8, help="materialized expert pool (>= top_k)")
ap.add_argument("--device", default="GPU")
ap.add_argument("--int4", action="store_true", help="NNCF INT4_SYM compress the MoE/linears")
ap.add_argument("--group-size", type=int, default=128)
ap.add_argument("--gptq-dir", default=None, help="GPTQ checkpoint dir (else $QWEN35_GPTQ_DIR)")
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--no-dynquant", action="store_true", default=True,
                help="disable per-token dynamic activation quant (keeps tiny matmuls off ocl:ref)")
args = ap.parse_args()

w = RealWeights(gptq_dir=args.gptq_dir)
print(f"building {args.layers}-layer real model, expert pool={args.experts}...", flush=True)
t0 = time.time()
m = build(w, args.layers, args.experts)
print(f"built in {time.time()-t0:.0f}s | n_gdn={m.n_gdn} n_attn={m.n_attn}", flush=True)

import openvino as ov
from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder
from openvino.frontend import FrontEndManager
ex = m.example_inputs()
print("torch.export...", flush=True)
ep = torch.export.export(m, ex, strict=True)
fe = FrontEndManager().load_by_framework("pytorch")
decoder = TorchFXPythonDecoder(ep.module())          # hold refs (avoid GC of the decoder)
ovm = fe.convert(fe.load(decoder))
print(f"OV convert done ({time.time()-t0:.0f}s)", flush=True)

if args.int4:
    import nncf
    from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
    tiny = [o.get_friendly_name() for o in ovm.get_ordered_ops()
            if o.get_type_name() == "MatMul" and o.get_output_partial_shape(0)[-1].get_length() == 1]
    ign = nncf.IgnoredScope(names=tiny) if tiny else None
    ovm = nncf.compress_weights(ovm, mode=nncf.CompressWeightsMode.INT4_SYM,
                                group_size=args.group_size, ratio=1.0, ignored_scope=ign,
                                advanced_parameters=AdvancedCompressionParameters(
                                    group_size_fallback_mode=nncf.GroupSizeFallbackMode.ADJUST))

core = ov.Core()
cfg = {"PERFORMANCE_HINT": "LATENCY"}
if args.no_dynquant:
    cfg["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = "0"
cm = core.compile_model(ovm, args.device, cfg)
rt = cm.get_runtime_model()


def lt(o):
    r = o.get_rt_info()
    try: return r["layerType"].astype(str)
    except: return o.get_type_name()


c = Counter(lt(o) for o in rt.get_ordered_ops())
gdn = sum(v for k, v in c.items() if "elta" in k.lower())
print(f"[{args.device}] runtime GatedDeltaNet nodes: {gdn} (expect {m.n_gdn}) | "
      f"DynamicQuantize: {c.get('DynamicQuantize',0)}", flush=True)

feed = [np.array(t) if not torch.is_tensor(t) else t.numpy() for t in ex]
inp = {cm.inputs[i]: feed[i].astype(np.float32) for i in range(len(feed))}
r = cm.create_infer_request()
for _ in range(5): r.infer(inp)
t = time.perf_counter()
for _ in range(args.iters): r.infer(inp)
dt = time.perf_counter() - t
print(f"[{args.device}] REAL Qwen3.5 {args.layers}L T=1 decode: "
      f"{dt/args.iters*1000:.2f} ms/tok  {args.iters/dt:.2f} tok/s", flush=True)
