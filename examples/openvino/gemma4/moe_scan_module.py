"""Exportable dense MoE replacing torch.ops.triton.fused_moe for the OpenVINO path.

Router: gate -> topk -> softmax (standard aten). Expert compute: gather the selected
experts' weights per token/slot and apply SwiGLU via batched matmul — all OV-translatable
ops (Gather, MatMul, Sigmoid, Mul). Mirrors FusedMoEExperts math:
  w1 [E, 2*inter, hidden] (fused gate+up), w2 [E, hidden, inter]
  y = w2 @ (silu(gate) * up), gate/up = split(w1 @ x)
"""

import torch
import torch.nn.functional as F


def moe_forward(x_flat, gate_w, w1, w2, shared_gate_up, shared_down, shared_gate_w, top_k):
    """
    x_flat: [N, hidden]
    gate_w: [num_experts, hidden]  (router)
    w1: [E, 2*inter, hidden]  w2: [E, hidden, inter]
    shared_*: shared expert SwiGLU weights + gate
    """
    N, H = x_flat.shape
    E2I = w1.shape[1]
    Hh = w2.shape[1]
    I = w2.shape[2]
    # Router
    scores = F.linear(x_flat, gate_w)                 # [N, E]
    tw, ti = torch.topk(scores, top_k, dim=-1)        # [N, k]
    tw = tw - tw.max(dim=-1, keepdim=True).values
    tw = tw.exp(); tw = tw / tw.sum(dim=-1, keepdim=True)   # manual softmax [N,k]

    # Flatten (N,k) slots into a batch for bmm. Gather selected expert weights.
    flat_idx = ti.view(-1)                          # [N*k]
    w1_sel = w1.index_select(0, flat_idx)              # [N*k, 2I, H]
    w2_sel = w2.index_select(0, flat_idx)              # [N*k, H, I]
    # per-slot input: repeat each token's x for its k slots
    xe = x_flat.repeat_interleave(top_k, dim=0).unsqueeze(-1)  # [N*k, H, 1]
    gate_up = torch.bmm(w1_sel, xe).squeeze(-1)        # [N*k, 2I]
    g, u = gate_up[:, :I], gate_up[:, I:]
    act = (F.silu(g) * u).unsqueeze(-1)                # [N*k, I, 1]
    ye = torch.bmm(w2_sel, act).squeeze(-1)            # [N*k, H]
    ye = ye.view(N, top_k, Hh)                      # [N, k, H]
    routed = (ye * tw.unsqueeze(-1)).sum(1)            # [N, H]

    # Shared expert SwiGLU + sigmoid gate
    sgu = F.linear(x_flat, shared_gate_up)            # [N, 2*sinter]
    sI = sgu.shape[-1] // 2
    sg, su = sgu[..., :sI], sgu[..., sI:]
    shared = F.linear(F.silu(sg) * su, shared_down)   # [N, H]
    sgate = torch.sigmoid(F.linear(x_flat, shared_gate_w))  # [N, 1]
    return routed + sgate * shared


def moe_reference(x_flat, gate_w, w1, w2, shared_gate_up, shared_down, shared_gate_w, top_k):
    # Same math, plain loop reference for numeric check.
    scores = F.linear(x_flat, gate_w)
    tw, ti = torch.topk(scores, top_k, dim=-1)
    tw = tw.softmax(dim=-1)
    N, H = x_flat.shape
    routed = torch.zeros(N, H)
    for n in range(N):
        for j in range(top_k):
            e = ti[n, j].item()
            gu = w1[e] @ x_flat[n]
            I = gu.shape[0] // 2
            a = F.silu(gu[:I]) * gu[I:]
            routed[n] += tw[n, j] * (w2[e] @ a)
    sgu = F.linear(x_flat, shared_gate_up); sI = sgu.shape[-1] // 2
    shared = F.linear(F.silu(sgu[..., :sI]) * sgu[..., sI:], shared_down)
    sgate = torch.sigmoid(F.linear(x_flat, shared_gate_w))
    return routed + sgate * shared
