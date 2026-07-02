"""Real-weight loader + forward for Qwen3.5-35B-A3B MXFP4 GGUF.

Memory strategy (32.2B of 34.7B params are MXFP4 experts):
- Non-expert weights (attn/ssm/norms/shared, ~4.9GB): dequantize eagerly to torch tensors.
- Expert weights (120 MXFP4 tensors, ~16GB raw): keep as raw uint8 GGUF blocks; dequantize
  ONLY the top-k (8) selected experts per token at forward time. Never materialize all 256.

Mapping verified against llama.cpp/src/models/qwen35moe.cpp:
- attn_qkv -> mixed_qkv -> conv1d -> silu -> split q/k/v (q/k L2-normed, not v)
- ssm_beta -> beta=sigmoid; ssm_alpha -> a; gate g = ssm_a * softplus(a + ssm_dt)  (ssm_a = -exp(A_log), pre-stored negative)
- attn_gate -> z; build_norm_gated(out, ssm_norm, z) = RMSNorm(out)*silu(z); ssm_out -> out_proj
- GQA: k/q repeat to v-heads. Full-attn layers (every 4th): attn_q(+gate)/k/v/q_norm/k_norm/output.
"""

import numpy as np
import torch
import torch.nn.functional as F
import gguf

GGUF_PATH = "Qwen3.5-35B-A3B-MXFP4_MOE_BF16.gguf"  # override via RealQwen35MoE(gguf_path=...)

CFG = dict(
    n_layers=40, hidden=2048, n_heads=16, n_kv=2, head_dim=256,
    n_experts=256, top_k=8, moe_inter=512, shared_inter=512,
    nk=16, nv=32, kdim=128, vdim=128, conv_k=4, key_dim=2048, value_dim=4096,
    rope_dim=64, rope_theta=1e7, full_interval=4, eps=1e-6,
)


def _deq(t):
    """Dequantize a GGUF tensor to a torch f32/bf16 tensor (for non-expert weights)."""
    raw = np.asarray(t.data)
    tn = t.tensor_type.name
    if tn == "F32":
        arr = raw.view(np.float32).reshape([int(d) for d in t.shape][::-1])
    elif tn == "BF16":
        arr = raw.view(np.uint16)
        # bf16 -> f32
        arr = (arr.astype(np.uint32) << 16).view(np.float32).reshape([int(d) for d in t.shape][::-1])
    else:
        arr = gguf.dequantize(raw, t.tensor_type)  # returns [.. , ] in gguf (reversed) order
    return torch.from_numpy(np.ascontiguousarray(arr))


class RealQwen35MoE:
    def __init__(self, gguf_path=GGUF_PATH):
        self.r = gguf.GGUFReader(gguf_path)
        self.byname = {t.name: t for t in self.r.tensors}
        self.cfg = CFG
        self._eager = {}       # dequantized non-expert weights
        self._expert_raw = {}  # raw MXFP4 tensors (kept quantized)
        self._load_nonexpert()

    def _load_nonexpert(self):
        for name, t in self.byname.items():
            if "_exps.weight" in name:            # the 3 big MXFP4 expert tensors per layer
                self._expert_raw[name] = t         # keep raw
            else:
                self._eager[name] = _deq(t)

    def g(self, name):
        return self._eager[name]

    def _dequant_experts(self, name, idx):
        """Dequant only experts `idx` from raw MXFP4 tensor `name`.
        gguf dequant gives [E, d0, d1]; we index E then return [len(idx), d0, d1]."""
        t = self._expert_raw[name]
        full = gguf.dequantize(np.asarray(t.data), t.tensor_type)  # [E, ...]
        sel = full[idx.numpy()]
        return torch.from_numpy(np.ascontiguousarray(sel)).float()

    # ---- forward pieces ----
    def _rms(self, x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.cfg["eps"]) * w

    def _l2(self, x):
        return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + self.cfg["eps"])

    def _gdn(self, x, li):
        c = self.cfg
        p = f"blk.{li}."
        mixed = F.linear(x, self.g(p + "attn_qkv.weight"))      # [T, conv_dim=8192]
        z = F.linear(x, self.g(p + "attn_gate.weight"))         # [T, value_dim=4096]
        T = x.shape[0]
        cd = c["key_dim"] * 2 + c["value_dim"]
        # causal depthwise conv1d: conv_w [conv_dim, K]
        cw = self.g(p + "ssm_conv1d.weight")                    # [8192, 4]
        conv_in = torch.cat([torch.zeros(c["conv_k"] - 1, cd), mixed], dim=0)
        acc = torch.zeros(T, cd)
        for j in range(c["conv_k"]):
            acc = acc + conv_in[j:j + T, :] * cw[:, j]
        qkv = F.silu(acc)
        kd = c["key_dim"]
        q = self._l2(qkv[:, :kd].view(T, c["nk"], c["kdim"]))
        k = self._l2(qkv[:, kd:2 * kd].view(T, c["nk"], c["kdim"]))
        v = qkv[:, 2 * kd:].view(T, c["nv"], c["vdim"])
        # q_scale = 1/sqrt(head_k_dim), applied in the recurrence (llama.cpp
        # delta-net-base.cpp:319-321; matches the native OV GatedDeltaNet op).
        q = q * (c["kdim"] ** -0.5)
        rep = c["nv"] // c["nk"]
        q = q.repeat_interleave(rep, dim=1); k = k.repeat_interleave(rep, dim=1)
        a = F.linear(x, self.g(p + "ssm_alpha.weight"))         # [T, 32]
        b = F.linear(x, self.g(p + "ssm_beta.weight"))          # [T, 32]
        beta = torch.sigmoid(b)
        ssm_a = self.g(p + "ssm_a"); dt = self.g(p + "ssm_dt.bias")
        g = ssm_a * torch.logaddexp(a + dt, torch.zeros_like(a))  # [T, 32]
        # sequential delta rule (eager reference; scan for export)
        state = torch.zeros(c["nv"], c["kdim"], c["vdim"])
        outs = []
        for t in range(T):
            gated = state * g[t].view(-1, 1, 1).exp()
            kc = k[t].unsqueeze(-1)
            proj = (gated * kc).sum(-2)
            delta = v[t] - proj
            sd = (delta * beta[t].unsqueeze(-1)).unsqueeze(-2)
            state = gated + kc * sd
            outs.append((state * q[t].unsqueeze(-1)).sum(-2))
        out = torch.stack(outs)                                 # [T, nv, vdim]
        normed = self._rms(out, self.g(p + "ssm_norm.weight"))
        gated = normed * F.silu(z.view(T, c["nv"], c["vdim"]))
        return F.linear(gated.reshape(T, -1), self.g(p + "ssm_out.weight"))

    def _attn(self, x, li, pos):
        c = self.cfg; p = f"blk.{li}."
        T = x.shape[0]
        qg = F.linear(x, self.g(p + "attn_q.weight"))           # [T, 8192] = q + gate (2x)
        qd = c["n_heads"] * c["head_dim"]
        q_and_gate = qg.view(T, c["n_heads"], c["head_dim"] * 2)
        q = q_and_gate[..., :c["head_dim"]]
        agate = q_and_gate[..., c["head_dim"]:]
        k = F.linear(x, self.g(p + "attn_k.weight")).view(T, c["n_kv"], c["head_dim"])
        v = F.linear(x, self.g(p + "attn_v.weight")).view(T, c["n_kv"], c["head_dim"])
        q = self._rms(q, self.g(p + "attn_q_norm.weight"))
        k = self._rms(k, self.g(p + "attn_k_norm.weight"))
        # partial RoPE (rope_dim=64 of head_dim=256)
        rd = c["rope_dim"]
        inv = 1.0 / (c["rope_theta"] ** (torch.arange(0, rd, 2).float() / rd))
        freqs = torch.outer(pos, inv); cos = freqs.cos().unsqueeze(1); sin = freqs.sin().unsqueeze(1)
        def rope(t):
            tr, tp = t[..., :rd], t[..., rd:]; half = rd // 2
            x1, x2 = tr[..., :half], tr[..., half:]
            return torch.cat([torch.cat([x1*cos-x2*sin, x2*cos+x1*sin], -1), tp], -1)
        q = rope(q); k = rope(k)
        rep = c["n_heads"] // c["n_kv"]
        k = k.repeat_interleave(rep, dim=1); v = v.repeat_interleave(rep, dim=1)
        q = q.transpose(0, 1); k = k.transpose(0, 1); v = v.transpose(0, 1)  # [h,T,d]
        scale = c["head_dim"] ** -0.5
        att = torch.bmm(q, k.transpose(-2, -1)) * scale
        keep = torch.tril(torch.ones(T, T)); att = torch.where(keep > 0, att, torch.full((T, T), -1e9))
        att = att - att.max(-1, keepdim=True).values; att = att.exp(); att = att / att.sum(-1, keepdim=True)
        y = torch.bmm(att, v)                                   # [h,T,d]
        y = torch.cat([y[i] for i in range(c["n_heads"])], dim=-1)  # [T, h*d]
        y = y * torch.sigmoid(agate.reshape(T, -1))
        return F.linear(y, self.g(p + "attn_output.weight"))

    def _moe(self, x, li):
        c = self.cfg; p = f"blk.{li}."
        N = x.shape[0]
        scores = F.linear(x, self.g(p + "ffn_gate_inp.weight"))  # [N, 256]
        tw, ti = torch.topk(scores, c["top_k"], dim=-1)
        tw = torch.softmax(tw, dim=-1)
        # dequant only selected experts (union across tokens to bound work)
        uniq = torch.unique(ti)
        w1 = self._dequant_experts(p + "ffn_gate_exps.weight", uniq)  # [U, inter, hidden]
        wu = self._dequant_experts(p + "ffn_up_exps.weight", uniq)    # [U, inter, hidden]
        w2 = self._dequant_experts(p + "ffn_down_exps.weight", uniq)  # [U, hidden, inter]
        remap = {int(e): j for j, e in enumerate(uniq.tolist())}
        routed = torch.zeros(N, c["hidden"])
        for n in range(N):
            for j in range(c["top_k"]):
                e = remap[int(ti[n, j])]
                gate = w1[e] @ x[n]; up = wu[e] @ x[n]
                routed[n] += tw[n, j] * (w2[e] @ (F.silu(gate) * up))
        # shared expert
        sg = F.linear(x, self.g(p + "ffn_gate_shexp.weight"))
        su = F.linear(x, self.g(p + "ffn_up_shexp.weight"))
        shared = F.linear(F.silu(sg) * su, self.g(p + "ffn_down_shexp.weight"))
        sgate = torch.sigmoid(F.linear(x, self.g(p + "ffn_gate_inp_shexp.weight").unsqueeze(0)))
        return routed + sgate * shared

    def forward_tokens(self, token_ids, n_layers=None):
        c = self.cfg
        n_layers = n_layers or c["n_layers"]
        pos = torch.arange(len(token_ids)).float()
        x = self.g("token_embd.weight")[token_ids]              # [T, hidden]
        for li in range(n_layers):
            is_full = (li + 1) % c["full_interval"] == 0
            h = self._rms(x, self.g(f"blk.{li}.attn_norm.weight"))
            x = x + (self._attn(h, li, pos) if is_full else self._gdn(h, li))
            h2 = self._rms(x, self.g(f"blk.{li}.post_attention_norm.weight"))
            x = x + self._moe(h2, li)
        x = self._rms(x, self.g("output_norm.weight"))
        return F.linear(x, self.g("output.weight"))             # [T, vocab]
