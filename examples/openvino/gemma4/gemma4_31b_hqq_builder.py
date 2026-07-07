"""Assemble the full 60-layer Gemma4-31B as a compressed OpenVINO Model, loading weights from
a pre-quantized **HQQ / torchao Int4** safetensors checkpoint (Gemma-4-31B-it-HQQ-INT4) —
mapping the int4 weights DIRECTLY to OpenVINO's compressed-u4 decompression pattern, with no
fp32 weight ever materialized on the host.

This is the "use the int4 checkpoint directly" alternative to gemma4_31b_ov_builder.Builder,
which instead dequantizes GGUF Q4_K_M -> fp32 and re-quantizes with a naive per-channel scale.
Using HQQ's calibrated group-wise scales/zero-points is both more accurate and lower-memory
(no fp32 dequant transients).

The checkpoint mixes TWO torchao quant types (verified against the GGUF of the same model,
cosine 0.9971) — the loader is metadata-driven and handles both:
  * Int4Tensor (q/k/o/gate/up):  _weight_qdata [out, in/2] uint8, nibble-packed along IN
    (lo nibble = even col, hi = odd); scale/zp [ng, out] bf16; group 32; asymmetric (frac zp).
  * IntxUnpackedToInt8Tensor (v/down, and the embedding at block_size [1,in]): _weight_qdata
    [out, in] int8 unpacked; scale/zp [out, ng] bf16 (embedding: [out, 1]); asymmetric.
  dequant (both): W[out,in] = (unpack(qdata) - zp) * scale, grouped along IN.

Both map to OpenVINO's compressed decompression pattern: Int4 -> Constant(u4), IntxInt8 ->
Constant(i8), each -> Convert(f32) -> Subtract(zp) -> Multiply(scale) -> MatMul, folded by the
plugin into one compressed FullyConnected (weight kept low-bit in the IR).

The architecture (dual head-dim 512/256, dual RoPE 1e6/1e4, GQA, QK-norm, unit-offset RMSNorm,
sliding-window pattern, softcap) is IDENTICAL to gemma4_31b_ov_builder, so HQQBuilder subclasses
it and overrides only the weight-I/O layer: tensor-name mapping (GGUF blk.N.* -> HF layers.N.*),
the int4/int8 linear (-> direct compressed pattern), and fp32 tensors (norms/embedding).

Same architecture params as the GGUF model (Gemma4-31B); no GGUF file needed.
"""
import json
import struct
import numpy as np
from safetensors import safe_open
from openvino import Type, Model
from openvino import opset15 as op

from gemma4_31b_ov_builder import Builder

ST_PATH = "models/gemma-4-31B-it-HQQ-INT4/model.safetensors"

# Fixed Gemma4-31B architecture constants (match the GGUF metadata the base Builder reads).
_ARCH = dict(n_embd=5376, n_layer=60, n_head=32, n_kv=16, hd_full=512, hd_swa=256,
             rope_full=512, rope_swa=256, theta_full=1e6, theta_swa=1e4,
             eps=9.999999974752427e-07, softcap=30.0, gs=32)

# GGUF short name (used throughout Builder's graph code) -> HF/HQQ linear prefix.
_LIN_MAP = {
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
    "output": "lm_head",
}
# GGUF norm/scalar names -> HF names.
_F32_MAP = {
    "attn_norm": "input_layernorm", "post_attention_norm": "post_attention_layernorm",
    "ffn_norm": "pre_feedforward_layernorm", "post_ffw_norm": "post_feedforward_layernorm",
    "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
    "output_norm": "norm",
}


class HQQBuilder(Builder):
    def __init__(self, st_path=ST_PATH, gs=_ARCH["gs"], per_channel=False):
        # NB: do NOT call super().__init__ (it opens a GGUF). Set the same attributes directly.
        self.f = safe_open(st_path, "pt")
        self._keys = set(self.f.keys())
        self._meta = self._read_metadata(st_path)      # per-tensor torchao quant type
        self.gs = gs
        # per_channel=True: re-quantize each linear to ONE i4 scale per output row ([O,1], no
        # group dim), reusing the parent Builder's per-channel _int4_lin. This collapses the
        # group-wise decompression subgraph (168 groups x 412 linears of reshape+Subtract+
        # Multiply) that otherwise inflates the compile working set past 62 GB at full depth.
        # Trades a little accuracy (single scale/row vs HQQ's calibrated group-32) for fitting
        # the full 60-layer compile+.pte on a memory-constrained box.
        self.per_channel = per_channel
        for k, v in _ARCH.items():
            setattr(self, k, v)
        # sliding pattern: full attention every 6th layer (last is full) — same as base Builder,
        # and matches the checkpoint's layer_types (full at index 5,11,17,...).
        self.is_full = [((i + 1) % 6 == 0) for i in range(self.n_layer)]
        # `bn` presence checks in the base class: emulate with the HQQ key set so has-tensor
        # tests (e.g. attn_v present?, layer_output_scale present?) resolve correctly.
        self.bn = _HQQKeyView(self._keys)

    # ---- name mapping ----
    def _hf_lin(self, gguf_name):
        """blk.<li>.<short>.weight  ->  layers.<li>.<hf_prefix>  (or top-level output/lm_head)."""
        if gguf_name == "output.weight":
            return "lm_head"
        # gguf_name like "blk.5.attn_q.weight"
        parts = gguf_name.split(".")
        li, short = parts[1], parts[2]
        return f"layers.{li}.{_LIN_MAP[short]}"

    def _hf_f32(self, gguf_name):
        if gguf_name == "output_norm.weight":
            return "norm.weight"
        if gguf_name == "token_embd.weight":
            return "embed_tokens"          # quantized embedding (handled specially)
        parts = gguf_name.split(".")
        li, short = parts[1], parts[2]
        if short == "layer_output_scale":
            return f"layers.{li}.layer_scalar"
        return f"layers.{li}.{_F32_MAP[short]}.weight"

    # ---- metadata ----
    @staticmethod
    def _read_metadata(st_path):
        with open(st_path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        return hdr.get("__metadata__", {})

    def _qtype(self, hf_prefix):
        """torchao quant type for a weight prefix: 'Int4Tensor' | 'IntxUnpackedToInt8Tensor'."""
        md = self._meta.get(hf_prefix + ".weight")
        return json.loads(md)["_type"] if md else None

    # ---- unpack: return (u[out,in] int-valued float, scale[out,ng], zp[out,ng]) ----
    def _load_qweight(self, hf_prefix):
        """Normalize either torchao type to a common layout: unpacked codes [out,in] plus
        per-(out,group) scale/zp of shape [out, ng]. Int4Tensor is nibble-packed with
        scale/zp [ng,out]; IntxUnpackedToInt8Tensor is already [out,in] with scale/zp [out,ng]."""
        qd = self.f.get_tensor(hf_prefix + "._weight_qdata")
        sc = self.f.get_tensor(hf_prefix + "._weight_scale").float().numpy()
        zp = self.f.get_tensor(hf_prefix + "._weight_zero_point").float().numpy()
        typ = self._qtype(hf_prefix)
        if typ == "Int4Tensor":
            q = qd.numpy().astype(np.uint8)                 # [out, in/2] nibble-packed
            OUT, INh = q.shape
            u = np.empty((OUT, INh * 2), np.int16)
            u[:, 0::2] = q & 0xF
            u[:, 1::2] = q >> 4
            return u, sc.T, zp.T, "u4"                      # scale/zp [ng,out] -> [out,ng]
        else:  # IntxUnpackedToInt8Tensor: qdata already [out,in] int8, scale/zp [out,ng]
            u = qd.numpy().astype(np.int16)
            return u, sc, zp, "i8"

    def _dequant_f32(self, hf_prefix):
        u, scT, zpT, _ = self._load_qweight(hf_prefix)      # scT/zpT: [out, ng]
        OUT, IN = u.shape
        ng = scT.shape[1]
        gs = IN // ng
        W = np.empty((OUT, IN), np.float32)
        uf = u.astype(np.float32)
        for g in range(ng):
            s = slice(g * gs, (g + 1) * gs)
            W[:, s] = (uf[:, s] - zpT[:, g:g + 1]) * scT[:, g:g + 1]
        return W

    def _embed_dequant_f32(self):
        """embed_tokens: IntxUnpackedToInt8Tensor, block_size [1,in] (one group/row): qdata int8
        [vocab,in], scale/zp [vocab,1]. W = (qdata - zp) * scale."""
        q = self.f.get_tensor("embed_tokens._weight_qdata").numpy().astype(np.float32)
        sc = self.f.get_tensor("embed_tokens._weight_scale").float().numpy()
        zp = self.f.get_tensor("embed_tokens._weight_zero_point").float().numpy()
        return (q - zp) * sc

    # ---- overrides of Builder's weight-I/O layer ----
    def _f32(self, name):
        """Base class uses this for: norms (plain bf16), the embedding (dequant), the per-layer
        output scalar, AND to read a quantized linear's out-dim
        (`self._f32("blk.N.attn_k.weight").shape[0]`)."""
        parts = name.split(".")
        if name == "output.weight" or (len(parts) >= 3 and parts[0] == "blk"
                                       and parts[2] in _LIN_MAP):
            return self._dequant_f32(self._hf_lin(name))
        hf = self._hf_f32(name)
        if hf == "embed_tokens":
            return self._embed_dequant_f32()
        return self.f.get_tensor(hf).float().numpy()

    def _int4_lin(self, x, wname, transpose_b=True):
        """Direct compressed linear (no fp32 weight materialized): Constant(u4|i8)[out,in] ->
        Convert(f32) -> reshape[out,ng,gs] -> Subtract(zp[out,ng,1]) -> Multiply(scale[out,ng,1])
        -> reshape[out,in] -> MatMul(x, w, transpose_b). The plugin folds it into one compressed
        FullyConnected (weight kept u4/i8 in the IR). Handles both torchao quant types.

        per_channel mode: delegate to the parent's per-channel i4 linear, which dequantizes the
        HQQ weight (via our _f32 override) and re-quantizes to a single [O,1] scale — far smaller
        compile working set (no per-group Subtract/Multiply), so full depth fits."""
        if self.per_channel:
            return Builder._int4_lin(self, x, wname, transpose_b)
        prefix = self._hf_lin(wname)
        u, scT, zpT, kind = self._load_qweight(prefix)      # scT/zpT: [out, ng]
        OUT, IN = u.shape
        ng = scT.shape[1]
        gs = IN // ng
        low_t = Type.u4 if kind == "u4" else Type.i8
        wc = op.constant(u.astype(np.uint8 if kind == "u4" else np.int8), low_t)
        wc.get_rt_info()["keep_const_precision"] = ""
        wf = op.convert(wc, Type.f32)
        wf.get_rt_info()["disable_constant_folding"] = ""
        zpg = op.constant(zpT.astype(np.float32).reshape(OUT, ng, 1), Type.f32)
        scg = op.constant(scT.astype(np.float32).reshape(OUT, ng, 1), Type.f32)
        wf3 = op.reshape(wf, op.constant(np.array([OUT, ng, gs]), Type.i64), False)
        w2d = op.reshape(op.multiply(op.subtract(wf3, zpg), scg),
                         op.constant(np.array([OUT, IN]), Type.i64), False)
        return op.matmul(x, w2d, False, transpose_b)

    def build(self, T, n_layers=None):
        """Same graph as Builder.build, but the token embedding is kept COMPRESSED: an int8
        Constant [vocab, in] is gathered by token id FIRST, then only the gathered rows are
        dequantized (subtract zp, multiply scale). This avoids materializing the ~5.6 GB fp32
        embedding constant, which otherwise (together with the fp32 lm_head and the ET .pte
        blob buffer) pushes peak RAM past 62 GB during lowering. Everything after the embedding
        is delegated to the base class by temporarily swapping in the gathered activation."""
        n_layers = n_layers or self.n_layer
        tokens = op.parameter([T], Type.i64, name="tokens")

        # compressed embedding: int8 Constant -> Gather(rows) -> (row - zp_row) * scale_row
        q = self.f.get_tensor("embed_tokens._weight_qdata").numpy().astype(np.int8)   # [vocab, in]
        sc = self.f.get_tensor("embed_tokens._weight_scale").float().numpy().astype(np.float32)   # [vocab,1]
        zp = self.f.get_tensor("embed_tokens._weight_zero_point").float().numpy().astype(np.float32)  # [vocab,1]
        emb_c = op.constant(q, Type.i8)
        emb_c.get_rt_info()["keep_const_precision"] = ""
        gathered = op.gather(op.convert(emb_c, Type.f32), tokens, op.constant(np.array(0), Type.i64))
        sc_g = op.gather(op.constant(sc, Type.f32), tokens, op.constant(np.array(0), Type.i64))
        zp_g = op.gather(op.constant(zp, Type.f32), tokens, op.constant(np.array(0), Type.i64))
        x = op.multiply(op.subtract(gathered, zp_g), sc_g)              # [T, in]
        x = op.multiply(x, op.constant(np.array(np.sqrt(self.n_embd), np.float32), Type.f32))

        for li in range(n_layers):
            p = f"blk.{li}."
            h = self._rmsnorm(x, p + "attn_norm.weight")
            a = self._attn(h, li, T)
            a = self._rmsnorm(a, p + "post_attention_norm.weight")
            attn_out = op.add(a, x)
            f = self._rmsnorm(attn_out, p + "ffn_norm.weight")
            f = self._ffn(f, li)
            f = self._rmsnorm(f, p + "post_ffw_norm.weight")
            cur = op.add(f, attn_out)
            if p + "layer_output_scale.weight" in self.bn:
                cur = op.multiply(cur, self._const_f32(p + "layer_output_scale.weight"))
            x = cur
        x = self._rmsnorm(x, "output_norm.weight")
        logits = self._int4_lin(x, "output.weight")                    # dedicated compressed lm_head
        if self.softcap:
            logits = op.multiply(logits, op.constant(np.array(1.0 / self.softcap, np.float32), Type.f32))
            logits = op.tanh(logits)
            logits = op.multiply(logits, op.constant(np.array(self.softcap, np.float32), Type.f32))
        return Model([op.result(logits)], [tokens], "gemma4_31b_hqq")


class _HQQKeyView:
    """Minimal stand-in for Builder.bn (a dict of GGUF tensors); only `in` membership is used
    by the base class, translated to the HQQ key namespace."""
    def __init__(self, keys):
        self._keys = keys

    def __contains__(self, gguf_name):
        # base class checks: "blk.{li}.attn_v.weight" (present?), "blk.{li}.rope_freqs.weight",
        # "blk.{li}.layer_output_scale.weight", "output.weight" (dedicated lm_head).
        if gguf_name == "output.weight":
            return "lm_head._weight_qdata" in self._keys
        if gguf_name.endswith("rope_freqs.weight"):
            return False                                        # HQQ has no rope_freqs tensor
        parts = gguf_name.split(".")
        if len(parts) >= 3 and parts[0] == "blk":
            li, short = parts[1], parts[2]
            if short == "attn_v":
                return f"layers.{li}.self_attn.v_proj._weight_qdata" in self._keys
            if short == "layer_output_scale":
                return f"layers.{li}.layer_scalar" in self._keys
        return False


if __name__ == "__main__":
    import sys
    b = HQQBuilder()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    m = b.build(T=1, n_layers=n)
    print(f"built {n}-layer HQQ model: inputs={[p.get_any_name() for p in m.inputs]} "
          f"outputs={[list(r.get_partial_shape()) for r in m.outputs]}")
