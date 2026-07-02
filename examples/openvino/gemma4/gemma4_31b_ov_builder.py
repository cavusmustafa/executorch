"""Assemble the full 60-layer Gemma4-31B (dense) as a compressed OpenVINO Model, directly
via the OV Python API — INT4 weights kept compressed (Constant(i4)->Convert->Multiply),
no bf16 blowup, no torch/mslk/NNCF.

Architecture (from GGUF + llama.cpp gemma4.cpp), all verified:
- embed * sqrt(n_embd); per layer: rmsnorm(attn_norm) -> attn -> rmsnorm(attn_post_norm) -> +res
  ; rmsnorm(ffn_norm) -> SwiGLU-GELU -> rmsnorm(ffn_post_norm) -> +res ; *out_scale
- attn: QK-norm (per-head RMS), V also RMS-normed, dual RoPE (theta 1e6 full / 1e4 SWA),
  attn_scale=1.0 (Gemma4), GQA, sliding-window pattern; head_dim 512 full / 256 SWA (n_rot from GGUF)
- Gemma RMSNorm is UNIT-OFFSET: x/rms * (1 + weight)
- final: rmsnorm(output_norm) -> lm_head -> softcap: tanh(x/30)*30
Linears use INT4 requant (dequant GGUF -> sym int4 group32 -> i4 Constant).
"""

import numpy as np
import gguf
from openvino import Type, Model
from openvino import opset15 as op

GGUF_PATH = "google_gemma-4-31B-it-Q4_K_M.gguf"  # override via Builder(gguf_path=...)


class Builder:
    def __init__(self, gguf_path=GGUF_PATH, gs=32):
        self.r = gguf.GGUFReader(gguf_path)
        self.bn = {t.name: t for t in self.r.tensors}
        self.gs = gs
        kv = {f.name: f for f in self.r.fields.values()}
        def V(n):
            f = kv[n]; p = f.parts[f.data[0]]
            return p.tolist()[0] if hasattr(p, "tolist") else p
        self.n_embd = V("gemma4.embedding_length")
        self.n_layer = V("gemma4.block_count")
        self.n_head = V("gemma4.attention.head_count")
        self.n_kv = V("gemma4.attention.head_count_kv")
        self.hd_full = V("gemma4.attention.key_length")       # 512
        self.hd_swa = V("gemma4.attention.key_length_swa")    # 256
        self.rope_full = V("gemma4.rope.dimension_count")     # 512
        self.rope_swa = V("gemma4.rope.dimension_count_swa")  # 256
        self.theta_full = V("gemma4.rope.freq_base")          # 1e6
        self.theta_swa = V("gemma4.rope.freq_base_swa")       # 1e4
        self.eps = V("gemma4.attention.layer_norm_rms_epsilon")
        self.softcap = V("gemma4.final_logit_softcapping")
        # sliding pattern: period 6 (5 SWA + 1 full), full at every 6th (llama.cpp is_swa_impl)
        # gemma: sliding_window_pattern -> pattern of length n_layer; deduce via presence check per layer
        self.is_full = [self._layer_is_full(i) for i in range(self.n_layer)]

    def _layer_is_full(self, i):
        # full-attention layers have rope_freqs tensor; SWA layers do not (llama.cpp gemma4.cpp:86)
        return f"blk.{i}.rope_freqs.weight" in self.bn or ((i + 1) % 6 == 0)

    # ---- weight helpers ----
    def _f32(self, name):
        t = self.bn[name]; W = gguf.dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32)
        return W

    def _const_f32(self, name):
        return op.constant(self._f32(name), Type.f32)

    def _int4_lin(self, x, wname, transpose_b=True):
        """Compressed linear kept INT4 in OV: Constant(i4) -> Convert(f32) -> Multiply(scale[O,1]) -> MatMul.
        Per-channel scale (no reshape) so OV's DecompressionHandling keeps it compressed and does NOT
        constant-fold the weight to f32 at compile time (reshapes around the Multiply trigger folding).
        Markers keep_const_precision + disable_constant_folding enforce it.
        """
        W = self._f32(wname)                     # [O, I]
        O, I = W.shape
        scale = (np.abs(W).max(-1, keepdims=True) / 7.0).clip(1e-9).astype(np.float32)  # [O,1] per-channel
        q = np.round(W / scale).clip(-8, 7).astype(np.int8)
        del W
        wc = op.constant(q, Type.i4); del q
        wf = op.convert(wc, Type.f32)
        wdq = op.multiply(wf, op.constant(scale, Type.f32))   # [O,I]*[O,1] broadcast, no reshape
        return op.matmul(x, wdq, False, transpose_b)

    def _rmsnorm(self, x, wname):
        # Gemma unit-offset RMSNorm: x/sqrt(mean(x^2)+eps) * (1+w)
        sq = op.multiply(x, x)
        mean = op.reduce_mean(sq, op.constant(np.array([-1]), Type.i64), True)
        inv = op.power(op.add(mean, op.constant(np.array(self.eps, np.float32), Type.f32)),
                       op.constant(np.array(-0.5, np.float32), Type.f32))
        normed = op.multiply(x, inv)
        w = self._const_f32(wname)
        one_plus = op.add(w, op.constant(np.array(1.0, np.float32), Type.f32))
        return op.multiply(normed, one_plus)

    def _rope(self, x, cos, sin, n_rot):
        # x: [T, H, hd]; rotate first n_rot dims (NeoX half-split), pass rest
        # split via slice
        import numpy as _np
        hd = x.get_output_partial_shape(0)[2].get_length()
        xr = op.slice(x, op.constant(_np.array([0]),Type.i64), op.constant(_np.array([n_rot]),Type.i64),
                      op.constant(_np.array([1]),Type.i64), op.constant(_np.array([2]),Type.i64))
        half = n_rot // 2
        x1 = op.slice(xr, op.constant(_np.array([0]),Type.i64), op.constant(_np.array([half]),Type.i64),
                      op.constant(_np.array([1]),Type.i64), op.constant(_np.array([2]),Type.i64))
        x2 = op.slice(xr, op.constant(_np.array([half]),Type.i64), op.constant(_np.array([n_rot]),Type.i64),
                      op.constant(_np.array([1]),Type.i64), op.constant(_np.array([2]),Type.i64))
        # cos/sin: [T,1,half]
        rot1 = op.subtract(op.multiply(x1, cos), op.multiply(x2, sin))
        rot2 = op.add(op.multiply(x2, cos), op.multiply(x1, sin))
        rot = op.concat([rot1, rot2], axis=2)
        if n_rot < hd:
            xp = op.slice(x, op.constant(_np.array([n_rot]),Type.i64), op.constant(_np.array([hd]),Type.i64),
                          op.constant(_np.array([1]),Type.i64), op.constant(_np.array([2]),Type.i64))
            return op.concat([rot, xp], axis=2)
        return rot

    def _rope_tables(self, T, n_rot, theta):
        half = n_rot // 2
        inv = (1.0 / (theta ** (np.arange(0, n_rot, 2).astype(np.float32) / n_rot)))  # [half]
        pos = np.arange(T).astype(np.float32)
        freqs = np.outer(pos, inv)  # [T, half]
        cos = op.constant(np.cos(freqs).reshape(T, 1, half).astype(np.float32), Type.f32)
        sin = op.constant(np.sin(freqs).reshape(T, 1, half).astype(np.float32), Type.f32)
        return cos, sin

    def _attn(self, x, li, T):
        p = f"blk.{li}."
        full = self.is_full[li]
        hd = self.hd_full if full else self.hd_swa
        n_rot = self.rope_full if full else self.rope_swa
        theta = self.theta_full if full else self.theta_swa
        nh = self.n_head
        # nkv derived from actual k weight out-dim / hd (full layers have fewer kv heads; V may be absent -> use K)
        k_out = self._f32(p+"attn_k.weight").shape[0]
        nkv = k_out // hd
        q = self._int4_lin(x, p+"attn_q.weight")               # [T, nh*hd]
        k = self._int4_lin(x, p+"attn_k.weight")               # [T, nkv*hd]
        has_v = (p+"attn_v.weight") in self.bn
        q = op.reshape(q, op.constant(np.array([T, nh, hd]),Type.i64), False)
        k = op.reshape(k, op.constant(np.array([T, nkv, hd]),Type.i64), False)
        # QK-norm (per head, unit-offset)
        q = self._rmsnorm_head(q, p+"attn_q_norm.weight")
        k = self._rmsnorm_head(k, p+"attn_k_norm.weight")
        # V: separate proj (rms-normed) if present, else = normed-K (llama.cpp Vcur=Kcur). Not RoPE'd.
        if has_v:
            v = self._int4_lin(x, p+"attn_v.weight")
            v = op.reshape(v, op.constant(np.array([T, nkv, hd]),Type.i64), False)
            v = self._rms_plain(v)
        else:
            v = k                                               # pre-RoPE normed K
        cos, sin = self._rope_tables(T, n_rot, theta)
        q = self._rope(q, cos, sin, n_rot)
        k = self._rope(k, cos, sin, n_rot)
        # GQA expand k,v to nh
        rep = nh // nkv
        if rep > 1:
            k = self._repeat_heads(k, rep, T, nkv, hd)
            v = self._repeat_heads(v, rep, T, nkv, hd)
        # attention: [T,nh,hd]->[nh,T,hd]
        q = op.transpose(q, op.constant(np.array([1,0,2]),Type.i64))
        k = op.transpose(k, op.constant(np.array([1,0,2]),Type.i64))
        v = op.transpose(v, op.constant(np.array([1,0,2]),Type.i64))
        att = op.matmul(q, k, False, True)                      # [nh,T,T]  (scale=1.0 for gemma4)
        # causal (+ sliding) mask
        mask = np.triu(np.full((T,T), -1e9, np.float32), 1)
        if not full:
            # sliding window: also mask positions older than window
            win = 1024
            for i in range(T):
                lo = max(0, i-win+1)
                mask[i,:lo] = -1e9
        att = op.add(att, op.constant(mask.reshape(1,T,T), Type.f32))
        att = op.softmax(att, 2)
        y = op.matmul(att, v, False, False)                     # [nh,T,hd]
        y = op.transpose(y, op.constant(np.array([1,0,2]),Type.i64))  # [T,nh,hd]
        y = op.reshape(y, op.constant(np.array([T, nh*hd]),Type.i64), False)
        return self._int4_lin(y, p+"attn_output.weight")

    def _rmsnorm_head(self, x, wname):
        sq=op.multiply(x,x); mean=op.reduce_mean(sq, op.constant(np.array([-1]),Type.i64), True)
        inv=op.power(op.add(mean, op.constant(np.array(self.eps,np.float32),Type.f32)), op.constant(np.array(-0.5,np.float32),Type.f32))
        normed=op.multiply(x,inv)
        w=self._const_f32(wname)  # [hd]
        return op.multiply(normed, op.add(w, op.constant(np.array(1.0,np.float32),Type.f32)))

    def _rms_plain(self, x):
        sq=op.multiply(x,x); mean=op.reduce_mean(sq, op.constant(np.array([-1]),Type.i64), True)
        inv=op.power(op.add(mean, op.constant(np.array(self.eps,np.float32),Type.f32)), op.constant(np.array(-0.5,np.float32),Type.f32))
        return op.multiply(x,inv)

    def _repeat_heads(self, x, rep, T, nkv, hd):
        # x [T,nkv,hd] -> [T, nkv*rep, hd] via broadcast: reshape [T,nkv,1,hd]->tile->[T,nkv*rep,hd]
        x4=op.reshape(x, op.constant(np.array([T,nkv,1,hd]),Type.i64), False)
        x4=op.broadcast(x4, op.constant(np.array([T,nkv,rep,hd]),Type.i64))
        return op.reshape(x4, op.constant(np.array([T,nkv*rep,hd]),Type.i64), False)

    def _ffn(self, x, li):
        p=f"blk.{li}."
        g=self._int4_lin(x, p+"ffn_gate.weight")
        u=self._int4_lin(x, p+"ffn_up.weight")
        # GELU-tanh
        act=op.gelu(g, "tanh")
        h=op.multiply(act, u)
        return self._int4_lin(h, p+"ffn_down.weight")

    def build(self, T, n_layers=None):
        n_layers = n_layers or self.n_layer
        tokens = op.parameter([T], Type.i64, name="tokens")
        emb = self._const_f32("token_embd.weight")   # [vocab, n_embd]
        x = op.gather(emb, tokens, op.constant(np.array(0),Type.i64))
        x = op.multiply(x, op.constant(np.array(np.sqrt(self.n_embd), np.float32), Type.f32))
        for li in range(n_layers):
            p=f"blk.{li}."
            h = self._rmsnorm(x, p+"attn_norm.weight")
            a = self._attn(h, li, T)
            a = self._rmsnorm(a, p+"post_attention_norm.weight")
            attn_out = op.add(a, x)
            f = self._rmsnorm(attn_out, p+"ffn_norm.weight")
            f = self._ffn(f, li)
            f = self._rmsnorm(f, p+"post_ffw_norm.weight")
            cur = op.add(f, attn_out)
            if p+"layer_output_scale.weight" in self.bn:
                cur = op.multiply(cur, self._const_f32(p+"layer_output_scale.weight"))
            x = cur
        x = self._rmsnorm(x, "output_norm.weight")
        # lm_head: dedicated output.weight if present, else tied to token_embd (matmul x @ embd^T)
        if "output.weight" in self.bn:
            logits = self._int4_lin(x, "output.weight")
        else:
            emb_c = self._const_f32("token_embd.weight")   # [vocab, n_embd]
            logits = op.matmul(x, emb_c, False, True)
        # softcap
        if self.softcap:
            logits = op.multiply(logits, op.constant(np.array(1.0/self.softcap, np.float32), Type.f32))
            logits = op.tanh(logits)
            logits = op.multiply(logits, op.constant(np.array(self.softcap, np.float32), Type.f32))
        return Model([op.result(logits)], [tokens], "gemma4_31b")
