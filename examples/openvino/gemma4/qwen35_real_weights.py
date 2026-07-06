"""Load REAL Qwen3.5-35B-A3B GPTQ weights into the proven decode model structure.

Strategy: the decode graph + GatedDeltaNet fusion is already validated in the reference
scripts (via scan -> FuseScanGDN). Here we replace random weights with the real
checkpoint's:
  - GDN / attention / shared-expert / gate / norms: BF16 in the checkpoint -> load as fp32.
  - The 256 routed MoE experts: GPTQ int4 -> dequantized on demand for the torch reference,
    then NNCF re-compresses them to weight-only INT4 on the ov.Model (see the export driver).

Checkpoint: `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` (~24.5 GB safetensors). Point at it with either
the `QWEN35_GPTQ_DIR` env var or the `--gptq-dir` flag on the driver scripts:

    huggingface-cli download Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 --local-dir ./qwen35_gptq
    export QWEN35_GPTQ_DIR=$PWD/qwen35_gptq

This module exposes:
  - RealWeights: reads the safetensors shards, dequantizes GPTQ experts, exposes per-name
    fp32 tensors (torch/numpy) matching the decode-model parameter shapes.
  - gptq_dequant(name): (out,in) fp32 dequantized expert weight.
"""
import glob, os, numpy as np, torch
from safetensors import safe_open

# Default checkpoint dir; override with QWEN35_GPTQ_DIR or RealWeights(gptq_dir=...).
GPTQ_DIR = os.environ.get("QWEN35_GPTQ_DIR", "./qwen35_gptq")
GS = 128
_PREFIX = "model.language_model."


def _bf16_to_f32(t):
    return t.float() if t.dtype == torch.bfloat16 else t.float()


class RealWeights:
    def __init__(s, gptq_dir=None):
        s.dir = gptq_dir or GPTQ_DIR
        if not glob.glob(s.dir + "/*.safetensors"):
            raise FileNotFoundError(
                f"No *.safetensors in {s.dir!r}. Download the checkpoint and set "
                f"QWEN35_GPTQ_DIR (see qwen35_real_weights.py docstring).")
        s.index = {}          # tensor name -> shard path
        for sh in glob.glob(s.dir + "/*.safetensors"):
            with safe_open(sh, "pt") as f:
                for k in f.keys():
                    s.index[k] = sh
        s._cache = {}

    def has(s, name):
        return name in s.index

    def raw(s, name):
        with safe_open(s.index[name], "pt") as f:
            return f.get_tensor(name)

    def f32(s, name):
        """A plain (bf16/f16/f32) tensor -> fp32 numpy [out,in] or [d]."""
        return _bf16_to_f32(s.raw(name)).numpy()

    def gptq_dequant(s, prefix):
        """Dequantize a GPTQ linear (prefix has .qweight/.qzeros/.scales) -> fp32 [out,in]."""
        qw = s.raw(prefix + ".qweight").numpy()      # [in/8, out] i32
        qz = s.raw(prefix + ".qzeros").numpy()       # [ng, out/8] i32
        sc = _bf16_to_f32(s.raw(prefix + ".scales")).numpy()  # [ng, out] f
        IN, OUT = qw.shape[0] * 8, qw.shape[1]
        ng = IN // GS
        wq = np.zeros((IN, OUT), np.float32)
        for i in range(8):
            wq[i::8] = (qw >> (4 * i)) & 0xF
        zp = np.zeros((ng, OUT), np.float32)
        for j in range(8):
            zp[:, j::8] = (qz >> (4 * j)) & 0xF
        w = np.empty((IN, OUT), np.float32)
        for g in range(ng):
            r = slice(g * GS, (g + 1) * GS)
            w[r] = sc[g] * (wq[r] - (zp[g] + 1.0))
        return w.T.copy()                            # [out,in]

    def gptq_u4(s, prefix):
        """Return (wq_uint8[out,in] in 0..15, zp_f32[out,ng,1], scale_f32[out,ng,1], IN, OUT, ng)
        for building the OV marked-u4 compressed pattern (no host dequant of the product)."""
        qw = s.raw(prefix + ".qweight").numpy()
        qz = s.raw(prefix + ".qzeros").numpy()
        sc = _bf16_to_f32(s.raw(prefix + ".scales")).numpy()
        IN, OUT = qw.shape[0] * 8, qw.shape[1]
        ng = IN // GS
        wq = np.zeros((IN, OUT), np.uint8)
        for i in range(8):
            wq[i::8] = (qw >> (4 * i)) & 0xF
        zp = np.zeros((ng, OUT), np.uint8)
        for j in range(8):
            zp[:, j::8] = (qz >> (4 * j)) & 0xF
        return (wq.T.copy(), (zp.T.astype(np.float32) + 1.0).reshape(OUT, ng, 1),
                sc.T.astype(np.float32).reshape(OUT, ng, 1), IN, OUT, ng)


if __name__ == "__main__":
    w = RealWeights()
    print("indexed tensors:", len(w.index))
    # sanity: dequant one expert + one bf16 linear
    e = w.gptq_dequant(_PREFIX + "layers.0.mlp.experts.0.gate_proj")
    print("expert.0.gate_proj dequant [out,in]:", e.shape, "range", float(e.min()), float(e.max()))
    z = w.f32(_PREFIX + "layers.0.linear_attn.in_proj_qkv.weight")
    print("GDN in_proj_qkv [out,in]:", z.shape)
    print("embed:", w.f32(_PREFIX + "embed_tokens.weight").shape,
          "| lm_head:", w.f32("lm_head.weight").shape)
