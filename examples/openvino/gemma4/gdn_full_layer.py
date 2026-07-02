"""Full exportable GatedDeltaNet layer for OpenVINO, assembling the proven scan core.

Mirrors qwen3_5_moe/model.py GatedDeltaNet.forward (the T>1 / chunked branch) but with:
  - manual causal depthwise conv1d (the model already uses a manual loop to avoid conv1d
    decomposition; kept, no state for a single-shot prefill)
  - manual L2norm (F.normalize -> aten.expand_as unsupported by OV frontend)
  - GQA head expand via repeat_interleave
  - scan delta-rule recurrence (-> native GatedDeltaNet op via FuseScanGDN)
  - RMSNormGated output, out_proj
Weights are passed in (no nn.Module state) so the test is self-contained.
"""

import torch
import torch.nn.functional as F
from torch._higher_order_ops.scan import scan


def _l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def _delta_combine(carry, slice_t):
    qx, kx, vx, gx, bx = slice_t
    g2 = gx.unsqueeze(-1).unsqueeze(-1)
    gated = carry * g2.exp()
    kc = kx.unsqueeze(-1)
    proj = (gated * kc).sum(-2)
    delta = vx - proj
    b2 = bx.unsqueeze(-1)
    sd = (delta * b2).unsqueeze(-2)
    new_state = gated + kc * sd
    out = (new_state * qx.unsqueeze(-1)).sum(-2)
    return new_state, out


class GDNLayer(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.nk = cfg["num_k_heads"]; self.nv = cfg["num_v_heads"]
        self.kd = cfg["head_k_dim"]; self.vd = cfg["head_v_dim"]
        self.conv_k = cfg["conv_kernel"]
        self.key_dim = self.nk * self.kd
        self.val_dim = self.nv * self.vd
        self.conv_dim = self.key_dim * 2 + self.val_dim
        self.rep = self.nv // self.nk
        H = cfg["hidden"]
        in_proj_dim = self.conv_dim + self.val_dim + 2 * self.nv
        self.in_proj = torch.nn.Linear(H, in_proj_dim, bias=False)
        self.conv_w = torch.nn.Parameter(torch.randn(self.conv_dim, self.conv_k) * 0.1)
        self.A_log = torch.nn.Parameter(torch.log(torch.empty(self.nv).uniform_(1, 16)))
        self.dt_bias = torch.nn.Parameter(torch.ones(self.nv))
        self.norm_w = torch.nn.Parameter(torch.ones(self.vd))
        self.out_proj = torch.nn.Linear(self.val_dim, H, bias=False)

    def forward(self, x, init_state):
        # x: [T, H] (single sequence, batch folded out). init_state: [nv, kd, vd]
        T, _ = x.shape
        proj = self.in_proj(x)                       # [T, in_proj_dim]
        cd, vd_, nh = self.conv_dim, self.val_dim, self.nv
        mixed = proj[:, :cd]                         # [T, conv_dim]
        z = proj[:, cd:cd + vd_].view(T, self.nv, self.vd)
        b = proj[:, cd + vd_: cd + vd_ + nh]         # [T, nv]
        a = proj[:, cd + vd_ + nh:]                  # [T, nv]

        # Causal depthwise conv1d (manual). Pad left with conv_k-1 zeros; no persistent state.
        conv_in = torch.cat([torch.zeros(self.conv_k - 1, cd), mixed], dim=0)  # [T+k-1, cd]
        acc = torch.zeros(T, cd)
        for j in range(self.conv_k):
            acc = acc + conv_in[j:j + T, :] * self.conv_w[:, j]
        qkv = F.silu(acc)                            # [T, conv_dim]

        kd_ = self.key_dim
        q = qkv[:, :kd_].view(T, self.nk, self.kd)
        k = qkv[:, kd_:2 * kd_].view(T, self.nk, self.kd)
        v = qkv[:, 2 * kd_:].view(T, self.nv, self.vd)
        q = _l2norm(q); k = _l2norm(k)
        if self.rep > 1:
            q = q.repeat_interleave(self.rep, dim=1)  # [T, nv, kd]
            k = k.repeat_interleave(self.rep, dim=1)
        beta = torch.sigmoid(b)                       # [T, nv]
        sp = torch.logaddexp(a + self.dt_bias, torch.zeros_like(a))  # softplus
        g = -self.A_log.exp() * sp                    # [T, nv]

        # scan delta-rule -> native GatedDeltaNet
        _, out = scan(_delta_combine, init_state, (q, k, v, g, beta))  # [T, nv, vd]

        # RMSNormGated(out, z) then out_proj
        normed = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + 1e-6)
        normed = self.norm_w * normed
        gated = normed * F.silu(z)
        return self.out_proj(gated.view(T, -1))
