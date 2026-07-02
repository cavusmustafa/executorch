"""Self-contained multi-layer Qwen3.5-MoE-style model for OpenVINO validation.

Assembles the proven pieces: GDN layers (75%) + full-attention layers (25%), each
followed by a sparse MoE block. Single-sequence prefill (no KV cache — one forward).
Weights are random; the point is to validate the WHOLE architecture exports, compiles,
fuses (native GatedDeltaNet), and runs on OpenVINO, and to localize the OOM phase at
scale. Layer pattern: full_attention every `full_interval` layers (last is full).
"""

import torch
import torch.nn.functional as F
from torch._higher_order_ops.scan import scan


def _l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def _rmsnorm(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _delta_combine(carry, s):
    qx, kx, vx, gx, bx = s
    gated = carry * gx.unsqueeze(-1).unsqueeze(-1).exp()
    kc = kx.unsqueeze(-1)
    proj = (gated * kc).sum(-2)
    delta = vx - proj
    sd = (delta * bx.unsqueeze(-1)).unsqueeze(-2)
    new_state = gated + kc * sd
    out = (new_state * qx.unsqueeze(-1)).sum(-2)
    return new_state, out


class GDNBlock(torch.nn.Module):
    def __init__(s, H, nk, nv, kd, vd, ck):
        super().__init__()
        s.nk, s.nv, s.kd, s.vd, s.ck = nk, nv, kd, vd, ck
        s.rep = nv // nk
        s.key_dim = nk * kd; s.val_dim = nv * vd
        s.conv_dim = s.key_dim * 2 + s.val_dim
        s.in_proj = torch.nn.Linear(H, s.conv_dim + s.val_dim + 2 * nv, bias=False)
        s.conv_w = torch.nn.Parameter(torch.randn(s.conv_dim, ck) * 0.1)
        s.A_log = torch.nn.Parameter(torch.log(torch.empty(nv).uniform_(1, 16)))
        s.dt_bias = torch.nn.Parameter(torch.ones(nv))
        s.norm_w = torch.nn.Parameter(torch.ones(vd))
        s.out_proj = torch.nn.Linear(s.val_dim, H, bias=False)

    def forward(s, x, init_state):
        T = x.shape[0]
        proj = s.in_proj(x)
        cd, vd_, nh = s.conv_dim, s.val_dim, s.nv
        mixed = proj[:, :cd]
        z = proj[:, cd:cd + vd_].view(T, s.nv, s.vd)
        b = proj[:, cd + vd_:cd + vd_ + nh]
        a = proj[:, cd + vd_ + nh:]
        conv_in = torch.cat([torch.zeros(s.ck - 1, cd), mixed], dim=0)
        acc = torch.zeros(T, cd)
        for j in range(s.ck):
            acc = acc + conv_in[j:j + T, :] * s.conv_w[:, j]
        qkv = F.silu(acc)
        kd_ = s.key_dim
        q = _l2norm(qkv[:, :kd_].view(T, s.nk, s.kd))
        k = _l2norm(qkv[:, kd_:2 * kd_].view(T, s.nk, s.kd))
        v = qkv[:, 2 * kd_:].view(T, s.nv, s.vd)
        if s.rep > 1:
            q = q.repeat_interleave(s.rep, dim=1)
            k = k.repeat_interleave(s.rep, dim=1)
        beta = torch.sigmoid(b)
        g = -s.A_log.exp() * torch.logaddexp(a + s.dt_bias, torch.zeros_like(a))
        _, out = scan(_delta_combine, init_state, (q, k, v, g, beta))
        normed = _rmsnorm(out, s.norm_w)
        return s.out_proj((normed * F.silu(z)).view(T, -1))


class AttnBlock(torch.nn.Module):
    def __init__(s, H, n_heads, n_kv, hd, rotary_dim, theta=1e6):
        super().__init__()
        s.nh, s.nkv, s.hd, s.rd = n_heads, n_kv, hd, rotary_dim
        s.qkv = torch.nn.Linear(H, (n_heads * hd) + 2 * (n_kv * hd), bias=False)
        s.o = torch.nn.Linear(n_heads * hd, H, bias=False)
        inv = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        s.register_buffer("inv_freq", inv)

    def _rope(s, x, cos, sin):
        xr, xp = x[..., :s.rd], x[..., s.rd:]
        half = s.rd // 2
        x1, x2 = xr[..., :half], xr[..., half:]
        rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return torch.cat([rot, xp], dim=-1)

    def forward(s, x, pos):
        T = x.shape[0]
        qkv = s.qkv(x)
        qd = s.nh * s.hd; kd = s.nkv * s.hd
        q = qkv[:, :qd].view(T, s.nh, s.hd)
        k = qkv[:, qd:qd + kd].view(T, s.nkv, s.hd)
        v = qkv[:, qd + kd:].view(T, s.nkv, s.hd)
        # pos is passed as float already (avoids aten.to.dtype which the OV frontend
        # doesn't translate); compute rope frequencies directly.
        freqs = torch.outer(pos, s.inv_freq)
        cos = freqs.cos().unsqueeze(1); sin = freqs.sin().unsqueeze(1)
        q = s._rope(q, cos, sin); k = s._rope(k, cos, sin)
        # GQA expand
        rep = s.nh // s.nkv
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1); v = v.repeat_interleave(rep, dim=1)
        q = q.transpose(0, 1); k = k.transpose(0, 1); v = v.transpose(0, 1)  # [h,T,d]
        # explicit additive causal mask (OV frontend translates SDPA with a mask more
        # reliably than the is_causal flag, which decomposes to unsupported ops).
        scale = s.hd ** -0.5
        att = torch.bmm(q, k.transpose(-2, -1)) * scale            # [h,T,T]
        neg = torch.full((T, T), -1e9)
        keep = torch.tril(torch.ones(T, T))
        att = torch.where(keep > 0, att, neg)
        att = att - att.max(dim=-1, keepdim=True).values
        att = att.exp(); att = att / att.sum(dim=-1, keepdim=True)
        y = torch.bmm(att, v)                                       # [h,T,d]
        # Concatenate heads along the feature axis to get [T, h*d] without a
        # transpose+reshape (aten.reshape.default is not translated by the OV frontend).
        # split along head axis (dim 0) into h tensors [1,T,d], squeeze, cat on last dim.
        heads = [y[i] for i in range(s.nh)]                         # each [T, d]
        y = torch.cat(heads, dim=-1)                                # [T, h*d]
        return s.o(y)


class MoEBlock(torch.nn.Module):
    def __init__(s, H, E, I, SI, top_k):
        super().__init__()
        s.top_k = top_k
        s.gate = torch.nn.Linear(H, E, bias=False)
        s.w1 = torch.nn.Parameter(torch.randn(E, 2 * I, H) * 0.05)
        s.w2 = torch.nn.Parameter(torch.randn(E, H, I) * 0.05)
        s.sgu = torch.nn.Linear(H, 2 * SI, bias=False)
        s.sd = torch.nn.Linear(SI, H, bias=False)
        s.sg = torch.nn.Linear(H, 1, bias=False)

    def forward(s, x):
        N, H = x.shape
        I = s.w2.shape[2]; Hh = s.w2.shape[1]
        scores = s.gate(x)
        tw, ti = torch.topk(scores, s.top_k, dim=-1)
        tw = tw - tw.max(dim=-1, keepdim=True).values
        tw = tw.exp(); tw = tw / tw.sum(dim=-1, keepdim=True)
        fi = ti.view(-1)
        w1s = s.w1.index_select(0, fi); w2s = s.w2.index_select(0, fi)
        xe = x.repeat_interleave(s.top_k, dim=0).unsqueeze(-1)
        gu = torch.bmm(w1s, xe).squeeze(-1)
        g, u = gu[:, :I], gu[:, I:]
        ye = torch.bmm(w2s, (F.silu(g) * u).unsqueeze(-1)).squeeze(-1)
        routed = (ye.view(N, s.top_k, Hh) * tw.unsqueeze(-1)).sum(1)
        sgu = s.sgu(x); sI = sgu.shape[-1] // 2
        shared = s.sd(F.silu(sgu[:, :sI]) * sgu[:, sI:])
        return routed + torch.sigmoid(s.sg(x)) * shared


class Model(torch.nn.Module):
    def __init__(s, n_layers, H, full_interval=4):
        super().__init__()
        s.n_layers = n_layers; s.H = H
        nk, nv, kd, vd = 4, 8, 16, 16
        s.nv, s.kd, s.vd = nv, kd, vd
        s.blocks = torch.nn.ModuleList()
        s.moes = torch.nn.ModuleList()
        s.norm1 = torch.nn.ParameterList()
        s.norm2 = torch.nn.ParameterList()
        s.kinds = []
        for i in range(n_layers):
            is_full = (i + 1) % full_interval == 0
            s.kinds.append(is_full)
            s.blocks.append(AttnBlock(H, 8, 2, 16, 8) if is_full else GDNBlock(H, nk, nv, kd, vd, 4))
            s.moes.append(MoEBlock(H, 8, 16, 16, 2))
            s.norm1.append(torch.nn.Parameter(torch.ones(H)))
            s.norm2.append(torch.nn.Parameter(torch.ones(H)))

    def forward(s, x, pos, init_state):
        for i in range(s.n_layers):
            h = _rmsnorm(x, s.norm1[i])
            x = x + (s.blocks[i](h, pos) if s.kinds[i] else s.blocks[i](h, init_state))
            x = x + s.moes[i](_rmsnorm(x, s.norm2[i]))
        return x
