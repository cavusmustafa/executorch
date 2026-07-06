"""Optimized fused-FC MoE + full-model builder for the real Qwen3.5-35B-A3B decode graph.

The naive sparse MoE (top-8 of 256 experts) compiles to `index_select` (GatherCompressed) +
per-expert `bmm`, which PERF_COUNT profiling showed is ~73% of GPU decode time. This module
replaces it with a FULLY-FUSED MoE:
  - gate/up as ONE [N*inter, H] FullyConnected over all experts (2 FCs total, no gather)
  - down as ONE [H, N*inter] FullyConnected (1 FC, no per-expert bmm)
  - routing weights are broadcast onto the activations before the down FC
All three expert projections then compile to single FullyConnectedCompressed ops. This is
numerically exact vs the brute-force top-8 loop (~1.9e-9) and takes the 40-layer real graph
from 13.5 -> 37.6 tok/s on the PTL GPU (matching the official OpenVINO IR).

Sparse MoE is pool-size-independent at decode (only top-k execute), so `build(w, n_layers,
n_exp)` materializes only the first `n_exp` experts while every executed dimension and all
non-expert weights stay real. Use n_exp >= top_k (8); the full 256-expert pool is only needed
for exact routing selection, not throughput.
"""
import torch
import torch.nn.functional as F

from qwen35_real_weights import RealWeights, _PREFIX
import qwen35_real_decode as R
from qwen35_real_decode import RealDecodeModel, CFG, _t


class PoolMoE(torch.nn.Module):
    """Fully-fused sparse MoE: all experts as 3 big FCs, top-k masking on the activations."""
    def __init__(s, w, li, n_exp, c=CFG):
        super().__init__()
        s.c = c; s.top_k = min(c["top_k"], n_exp); s.n_exp = n_exp; s.I = c["moe_inter"]
        p = f"{_PREFIX}layers.{li}.mlp."
        s.gate = torch.nn.Parameter(_t(w.f32(p + "gate.weight"))[:n_exp], False)
        s.sg = torch.nn.Parameter(_t(w.f32(p + "shared_expert_gate.weight")), False)
        s.s_gate = torch.nn.Parameter(_t(w.f32(p + "shared_expert.gate_proj.weight")), False)
        s.s_up = torch.nn.Parameter(_t(w.f32(p + "shared_expert.up_proj.weight")), False)
        s.s_down = torch.nn.Parameter(_t(w.f32(p + "shared_expert.down_proj.weight")), False)
        gp, up, dn = [], [], []
        for e in range(n_exp):
            ep = f"{p}experts.{e}."
            gp.append(_t(w.gptq_dequant(ep + "gate_proj")))
            up.append(_t(w.gptq_dequant(ep + "up_proj")))
            dn.append(_t(w.gptq_dequant(ep + "down_proj")))
        # Fully-fused MoE: gate/up as ONE [N*inter, H] FC; down as ONE [H, N*inter] FC.
        s.gw = torch.nn.Parameter(torch.cat(gp, 0), False)   # [N*inter, H]
        s.uw = torch.nn.Parameter(torch.cat(up, 0), False)   # [N*inter, H]
        s.dw = torch.nn.Parameter(torch.cat([d for d in dn], 1), False)  # [H, N*inter]

    def forward(s, x1):
        logits = F.linear(x1, s.gate)
        e = (logits - logits.max(-1, keepdim=True).values).exp()
        probs = e / e.sum(-1, keepdim=True)                             # [1,N]
        tw, ti = torch.topk(probs, s.top_k, -1)
        mask = torch.zeros(1, s.n_exp).scatter(1, ti, tw)               # [1,N] routing weights
        act = F.silu(F.linear(x1, s.gw)) * F.linear(x1, s.uw)           # [1, N*inter]  (2 FCs)
        # scale each expert's inter-block by its routing weight, then ONE down FC over N*inter
        act = (act.view(s.n_exp, s.I) * mask.view(s.n_exp, 1)).view(1, s.n_exp * s.I)
        routed = F.linear(act, s.dw)                                    # [1, H]  (1 FC)
        shared = F.linear(F.silu(F.linear(x1, s.s_gate)) * F.linear(x1, s.s_up), s.s_down)
        return routed + torch.sigmoid(F.linear(x1, s.sg)) * shared


def build(w, n_layers, n_exp):
    """Assemble the full N-layer real decode model with the fused-FC MoE."""
    m = RealDecodeModel.__new__(RealDecodeModel)
    torch.nn.Module.__init__(m)
    m.c = CFG; m.n_layers = n_layers; m.kinds = []
    m.blocks = torch.nn.ModuleList(); m.n1 = torch.nn.ParameterList()
    m.n2 = torch.nn.ParameterList(); m.moes = torch.nn.ModuleList()
    for li in range(n_layers):
        is_full = (li + 1) % CFG["full_interval"] == 0
        m.kinds.append(is_full)
        m.blocks.append(R.RealAttn(w, li) if is_full else R.RealGDN(w, li))
        m.moes.append(PoolMoE(w, li, n_exp))
        p = f"{_PREFIX}layers.{li}."
        m.n1.append(torch.nn.Parameter(_t(w.f32(p + "input_layernorm.weight")), False))
        m.n2.append(torch.nn.Parameter(_t(w.f32(p + "post_attention_layernorm.weight")), False))
    m.embed = torch.nn.Parameter(_t(w.f32(f"{_PREFIX}embed_tokens.weight")), False)
    m.final = torch.nn.Parameter(_t(w.f32(f"{_PREFIX}norm.weight")), False)
    m.lm = torch.nn.Parameter(_t(w.f32("lm_head.weight")), False)
    m.n_gdn = m.kinds.count(False); m.n_attn = m.kinds.count(True)
    return m.eval()
