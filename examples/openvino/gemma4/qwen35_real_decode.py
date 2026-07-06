"""Faithful single-token DECODE model for the REAL Qwen3.5-35B-A3B, matching the HF
Qwen3_5MoeGatedDeltaNet math exactly, with real weights loaded via qwen35_real_weights.
Built for OpenVINO export (scan -> FuseScanGDN). Experts stay GPTQ-u4-compressible.

HF math confirmed from modeling_qwen3_5_moe.py:
  GDN: in_proj_qkv/z/a/b (separate); causal conv1d over qkv; split q,k,v; L2norm(q,k);
       beta=sigmoid(b); g=-exp(A_log)*softplus(a+dt_bias); q_scale=1/sqrt(head_k_dim);
       recurrence: S=S*exp(g); kv=(S*k).sum(-2); delta=(v-kv)*beta; S=S+k⊗delta;
       out=(S*q).sum(-2); RMSNormGated(out,z)= rms(out)*w*silu(z); out_proj.
  Attn (full): q/k/v_proj, q/k RMSNorm per-head, partial RoPE (rope_dim=64), GQA 16/2,
       attn_output_gate: out *= sigmoid(gate) ; o_proj.
  MoE: gate->top8 of 256; each expert SwiGLU(gate_proj,up_proj,down_proj) [GPTQ int4];
       shared expert SwiGLU + sigmoid(shared_expert_gate); routed = sum(topk_w * expert).
"""
import numpy as np, torch, torch.nn.functional as F
from torch._higher_order_ops.scan import scan
from qwen35_real_weights import RealWeights, _PREFIX

CFG = dict(H=2048, n_layers=40, n_exp=256, top_k=8, moe_inter=512, shared_inter=512,
           nk=16, nv=32, kd=128, vd=128, conv_k=4, n_heads=16, n_kv=2, head_dim=256,
           rope_dim=64, rope_theta=1e7, full_interval=4, eps=1e-6)


def _l2(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def _rms(x, w, eps=1e-6):
    # RMSNormGated (GDN): weight * x, NO unit offset
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _rms_uo(x, w, eps=1e-6):
    # Qwen3_5MoeRMSNorm (attn q/k norm + all layernorms): (1 + weight) * x  (unit-offset)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w)


def _delta_combine(carry, s):
    qx, kx, vx, gx, bx = s
    gated = carry * gx.unsqueeze(-1).unsqueeze(-1).exp()
    kc = kx.unsqueeze(-1)
    proj = (gated * kc).sum(-2)
    delta = (vx - proj) * bx.unsqueeze(-1)
    new_state = gated + kc * delta.unsqueeze(-2)
    out = (new_state * qx.unsqueeze(-1)).sum(-2)
    return new_state, out


def _t(np_arr):
    return torch.from_numpy(np_arr).float()


class RealGDN(torch.nn.Module):
    def __init__(s, w, li, c=CFG):
        super().__init__()
        p = f"{_PREFIX}layers.{li}.linear_attn."
        s.c = c
        s.qkv = torch.nn.Parameter(_t(w.f32(p + "in_proj_qkv.weight")), False)   # [8192,H]
        s.z = torch.nn.Parameter(_t(w.f32(p + "in_proj_z.weight")), False)       # [4096,H]
        s.a = torch.nn.Parameter(_t(w.f32(p + "in_proj_a.weight")), False)       # [32,H]
        s.b = torch.nn.Parameter(_t(w.f32(p + "in_proj_b.weight")), False)       # [32,H]
        s.conv = torch.nn.Parameter(_t(w.f32(p + "conv1d.weight")), False)       # [8192,1,4]
        s.A_log = torch.nn.Parameter(_t(w.f32(p + "A_log")), False)              # [32]
        s.dt = torch.nn.Parameter(_t(w.f32(p + "dt_bias")), False)               # [32]
        s.norm_w = torch.nn.Parameter(_t(w.f32(p + "norm.weight")), False)       # [128]
        s.out = torch.nn.Parameter(_t(w.f32(p + "out_proj.weight")), False)      # [H,4096]

    def forward(s, x1, state, conv_cache):
        c = s.c; key_dim = c["nk"] * c["kd"]; val_dim = c["nv"] * c["vd"]; cd = 2 * key_dim + val_dim
        mixed = F.linear(x1, s.qkv)                       # [1, 8192]
        z = F.linear(x1, s.z).view(1, c["nv"], c["vd"])
        a = F.linear(x1, s.a); b = F.linear(x1, s.b)      # [1,32]
        # causal depthwise conv1d over last conv_k steps: conv weight [8192,1,4]
        conv_in = torch.cat([conv_cache, mixed], dim=0)   # [conv_k, 8192]
        cw = s.conv.squeeze(1)                            # [8192,4]
        acc = torch.zeros(1, cd)
        for j in range(c["conv_k"]):
            acc = acc + conv_in[j:j + 1, :] * cw[:, j]
        qkv = F.silu(acc)
        q = _l2(qkv[:, :key_dim].view(1, c["nk"], c["kd"]))
        k = _l2(qkv[:, key_dim:2 * key_dim].view(1, c["nk"], c["kd"]))
        v = qkv[:, 2 * key_dim:].view(1, c["nv"], c["vd"])
        rep = c["nv"] // c["nk"]
        q = q.repeat_interleave(rep, dim=1) * (c["kd"] ** -0.5)   # q_scale = 1/sqrt(head_k_dim)
        k = k.repeat_interleave(rep, dim=1)
        beta = torch.sigmoid(b)
        # softplus(x) = log1p(exp(x)); OV frontend lacks aten.softplus but has log1p/exp
        g = -s.A_log.exp() * torch.log1p((a + s.dt).exp())
        import os as _os
        if _os.environ.get("GDN_UNROLL") == "1":
            # T=1 decode: scan runs exactly one step -> unroll to a plain op sequence (no HOP),
            # so ExecuTorch's edge pipeline can lower it (FuseScanGDN won't fire, but GDN is ~3%).
            _, ys = _delta_combine(state, (q[0], k[0], v[0], g[0], beta[0]))
            ys = ys.unsqueeze(0)
        else:
            _, ys = scan(_delta_combine, state, (q, k, v, g, beta))    # [1, nv, vd]
        normed = _rms(ys, s.norm_w) * F.silu(z)                    # RMSNormGated
        return F.linear(normed.view(1, -1), s.out)                 # [1, H]


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


class RealAttn(torch.nn.Module):
    """Full-attention layer, cached single-token decode. kcache/vcache [n_kv, ctx, head_dim]."""
    def __init__(s, w, li, c=CFG):
        super().__init__()
        p = f"{_PREFIX}layers.{li}.self_attn."
        s.c = c
        s.q = torch.nn.Parameter(_t(w.f32(p + "q_proj.weight")), False)   # [nh*hd*2, H]
        s.k = torch.nn.Parameter(_t(w.f32(p + "k_proj.weight")), False)   # [n_kv*hd, H]
        s.v = torch.nn.Parameter(_t(w.f32(p + "v_proj.weight")), False)
        s.o = torch.nn.Parameter(_t(w.f32(p + "o_proj.weight")), False)   # [H, nh*hd]
        s.qn = torch.nn.Parameter(_t(w.f32(p + "q_norm.weight")), False)  # [hd]
        s.kn = torch.nn.Parameter(_t(w.f32(p + "k_norm.weight")), False)
        inv = 1.0 / (c["rope_theta"] ** (np.arange(0, c["rope_dim"], 2).astype(np.float32) / c["rope_dim"]))
        s.inv = torch.nn.Parameter(torch.from_numpy(inv), False)

    def forward(s, x1, kcache, vcache, pos):
        c = s.c; hd = c["head_dim"]; nh = c["n_heads"]; nkv = c["n_kv"]
        qg = F.linear(x1, s.q).view(1, nh, hd * 2)
        q = qg[..., :hd]; gate = qg[..., hd:]               # [1,nh,hd] each (kept per-head)
        q = _rms_uo(q, s.qn)                                # per-head RMSNorm (unit-offset)
        k = _rms_uo(F.linear(x1, s.k).view(1, nkv, hd), s.kn)
        v = F.linear(x1, s.v).view(1, nkv, hd)
        # partial RoPE (rope_dim of hd)
        rd = c["rope_dim"]
        freqs = pos.unsqueeze(-1) * s.inv                   # [1, rd/2]
        cos = torch.cat([freqs.cos(), freqs.cos()], -1).unsqueeze(1)   # [1,1,rd]
        sin = torch.cat([freqs.sin(), freqs.sin()], -1).unsqueeze(1)
        def rope(t):
            tr, tp = t[..., :rd], t[..., rd:]
            return torch.cat([tr * cos + _rotate_half(tr) * sin, tp], -1)
        q = rope(q); k = rope(k)
        # append to cache -> [nkv, ctx+1, hd]
        kf = torch.cat([kcache, k.transpose(0, 1)], dim=1)
        vf = torch.cat([vcache, v.transpose(0, 1)], dim=1)
        rep = nh // nkv
        kf = kf.repeat_interleave(rep, dim=0)               # [nh, ctx+1, hd]
        vf = vf.repeat_interleave(rep, dim=0)
        qh = q.transpose(0, 1)                              # [nh,1,hd]
        att = torch.bmm(qh, kf.transpose(-2, -1)) * (hd ** -0.5)   # [nh,1,ctx+1]
        att = att - att.max(-1, keepdim=True).values
        att = att.exp(); att = att / att.sum(-1, keepdim=True)
        y = torch.bmm(att, vf)                              # [nh,1,hd]
        # apply per-head output gate then concat heads (avoids a non-contiguous view)
        g = torch.sigmoid(gate)                             # [1,nh,hd]
        y = torch.cat([y[i] * g[0, i:i + 1] for i in range(nh)], dim=-1)   # [1, nh*hd]
        return F.linear(y, s.o)


class RealMoE(torch.nn.Module):
    """Sparse MoE: top-8 of 256 GPTQ-int4 experts + shared expert. Dequant only selected."""
    def __init__(s, w, li, c=CFG):
        super().__init__()
        s.c = c; s.li = li; s.w = w
        p = f"{_PREFIX}layers.{li}.mlp."
        s.gate = torch.nn.Parameter(_t(w.f32(p + "gate.weight")), False)          # [256, H]
        s.sg = torch.nn.Parameter(_t(w.f32(p + "shared_expert_gate.weight")), False)  # [1,H]
        s.s_gate = torch.nn.Parameter(_t(w.f32(p + "shared_expert.gate_proj.weight")), False)  # [512,H]
        s.s_up = torch.nn.Parameter(_t(w.f32(p + "shared_expert.up_proj.weight")), False)
        s.s_down = torch.nn.Parameter(_t(w.f32(p + "shared_expert.down_proj.weight")), False)  # [H,512]
        # pre-dequant all experts to fp32 tensors (memory: 256*(2*512*2048+2048*512)*4 ~ 3.2GB/layer)
        # for the numeric check we dequant lazily per selected expert instead:
        s._pfx = p

    def _expert(s, e, x):
        p = f"{s._pfx}experts.{e}."
        gw = _t(s.w.gptq_dequant(p + "gate_proj")); uw = _t(s.w.gptq_dequant(p + "up_proj"))
        dw = _t(s.w.gptq_dequant(p + "down_proj"))
        return F.linear(F.silu(F.linear(x, gw)) * F.linear(x, uw), dw)

    def forward(s, x1):  # x1 [1,H]
        c = s.c
        logits = F.linear(x1, s.gate)                        # [1,256]
        probs = F.softmax(logits.float(), dim=-1)
        tw, ti = torch.topk(probs, c["top_k"], dim=-1)       # [1,8]
        routed = torch.zeros(1, c["H"])
        for j in range(c["top_k"]):
            e = int(ti[0, j])
            routed = routed + tw[0, j] * s._expert(e, x1)
        shared = F.linear(F.silu(F.linear(x1, s.s_gate)) * F.linear(x1, s.s_up), s.s_down)
        shared = torch.sigmoid(F.linear(x1, s.sg)) * shared
        return routed + shared


class RealDecodeModel(torch.nn.Module):
    """Full 40-layer Qwen3.5-35B-A3B single-token decode with real weights.
    Inputs: x[1,H], gdn_state[n_gdn,nv,kd,vd], conv_cache[n_gdn,conv_k-1,conv_dim],
            kcache/vcache[n_attn,n_kv,ctx,head_dim], pos[1]. Returns logits[1,vocab]."""
    def __init__(s, w, c=CFG, n_layers=None):
        super().__init__()
        s.c = c
        s.n_layers = n_layers or c["n_layers"]
        s.kinds = []                     # True=full_attention
        s.blocks = torch.nn.ModuleList()
        s.n1 = torch.nn.ParameterList()  # input_layernorm
        s.n2 = torch.nn.ParameterList()  # post_attention_layernorm
        s.moes = torch.nn.ModuleList()
        for li in range(s.n_layers):
            is_full = (li + 1) % c["full_interval"] == 0
            s.kinds.append(is_full)
            s.blocks.append(RealAttn(w, li) if is_full else RealGDN(w, li))
            s.moes.append(RealMoE(w, li))
            p = f"{_PREFIX}layers.{li}."
            s.n1.append(torch.nn.Parameter(_t(w.f32(p + "input_layernorm.weight")), False))
            s.n2.append(torch.nn.Parameter(_t(w.f32(p + "post_attention_layernorm.weight")), False))
        s.embed = torch.nn.Parameter(_t(w.f32(f"{_PREFIX}embed_tokens.weight")), False)
        s.final = torch.nn.Parameter(_t(w.f32(f"{_PREFIX}norm.weight")), False)
        s.lm = torch.nn.Parameter(_t(w.f32("lm_head.weight")), False)
        s.n_gdn = s.kinds.count(False); s.n_attn = s.kinds.count(True)

    def forward(s, x, gdn_state, conv_cache, kcache, vcache, pos):
        gi = ai = 0
        for li in range(s.n_layers):
            h = _rms_uo(x, s.n1[li])
            if s.kinds[li]:
                x = x + s.blocks[li](h, kcache[ai], vcache[ai], pos); ai += 1
            else:
                x = x + s.blocks[li](h, gdn_state[gi], conv_cache[gi]); gi += 1
            x = x + s.moes[li](_rms_uo(x, s.n2[li]))
        x = _rms_uo(x, s.final)
        return F.linear(x, s.lm)

    def example_inputs(s):
        c = s.c
        return (torch.randn(1, c["H"]),
                torch.zeros(s.n_gdn, c["nv"], c["kd"], c["vd"]),
                torch.zeros(s.n_gdn, c["conv_k"] - 1, 2 * c["nk"] * c["kd"] + c["nv"] * c["vd"]),
                torch.randn(s.n_attn, c["n_kv"], 128, c["head_dim"]) * 0.1,
                torch.randn(s.n_attn, c["n_kv"], 128, c["head_dim"]) * 0.1,
                torch.tensor([128.0]))
