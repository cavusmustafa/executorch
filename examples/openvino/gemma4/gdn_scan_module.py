"""Self-contained scan-based GatedDeltaNet for the OpenVINO path.

Mirrors the Qwen3.5 MoE GatedDeltaNet math (mlx_source_transformations._exportable
form) but replaces the custom-op recurrence (torch.ops.mlx.gated_delta_rule) with a
torch scan over the delta rule, which the built OV frontend lowers to a v5::Loop and
FuseScanGDN collapses into the native GatedDeltaNet kernel.

Key alignment with the native op (gated_delta_net_ref.cl):
- The op applies q_scale = 1/sqrt(head_k_dim) internally, and (with fuse_qk_l2norm)
  L2-normalizes q/k. Here we L2-normalize q/k in the module and let the op's built-in
  q_scale supply the 1/sqrt(d) factor, so we DON'T pre-scale q ourselves.
"""

import torch
from torch._higher_order_ops.scan import scan


def _l2norm(x, eps=1e-6):
    # Manual L2 normalize over last dim using OV-translatable ops (F.normalize decomposes
    # to aten.expand_as which the OV pytorch frontend does not translate).
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def _delta_combine(carry, slice_t):
    # carry: state [H, K, V]; slice_t: (q,k,v,g,beta) each per-step
    qx, kx, vx, gx, bx = slice_t          # qx,kx [H,K]; vx [H,V]; gx,bx [H]
    g2 = gx.unsqueeze(-1).unsqueeze(-1)    # [H,1,1]
    gated = carry * g2.exp()               # [H,K,V]
    kc = kx.unsqueeze(-1)                  # [H,K,1]
    proj = (gated * kc).sum(-2)            # [H,V]
    delta = vx - proj                      # [H,V]
    b2 = bx.unsqueeze(-1)                  # [H,1]
    sd = (delta * b2).unsqueeze(-2)        # [H,1,V]
    new_state = gated + kc * sd            # [H,K,V]
    out = (new_state * qx.unsqueeze(-1)).sum(-2)  # [H,V]
    return new_state, out


def gdn_scan(q, k, v, g, beta, init_state):
    """Delta-rule recurrence via scan. Inputs [T,H,*], state [H,K,V]. Returns [T,H,V].

    q, k are L2-normalized here (the native op's built-in q_scale supplies 1/sqrt(d)).
    """
    q = _l2norm(q)
    k = _l2norm(k)
    _, ys = scan(_delta_combine, init_state, (q, k, v, g, beta))
    return ys


def gdn_reference(q, k, v, g, beta, init_state, k_head_dim):
    """Eager reference matching the native op semantics (L2norm q/k + q_scale=1/sqrt(d))."""
    q = _l2norm(q)
    k = _l2norm(k)
    q = q * (k_head_dim ** -0.5)
    T = q.shape[0]
    c = init_state.clone()
    outs = []
    for t in range(T):
        c, o = _delta_combine(c, (q[t], k[t], v[t], g[t], beta[t]))
        outs.append(o)
    return torch.stack(outs)
