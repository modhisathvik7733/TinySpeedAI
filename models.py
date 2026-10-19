"""SGHM v2 (Surprise-Gated Hebbian Memory), built for speed, and a matched GPT baseline.

SGHM information flow:
  cortex (causal local mixing + MLP) -> predicts the next token's embedding (predictive coding)
  surprise s_t = 1 - cos(prediction_{t-1}, embedding(x_t))        (d-dim, no vocabulary softmax)
  fast synapses M (fixed size per head): write  M <- lam*M + g_t * v_t k_t^T,  g_t = gate(s_t)
                                         read   r_t = M_{t-1} q_t
Training uses the exactly equivalent masked parallel form; generation uses the O(1)-state recurrent form.

v2 speed changes vs v1 (see RESULTS.md): surprise in embedding space instead of a second vocabulary
softmax (~1M MACs/token saved), dense MLP instead of k-WTA top-k (sparsity is free in the brain but
not on a dense GPU), one memory with a single fused projection instead of two.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


# ---------------------------------------------------------------- cortex

class LocalMix(nn.Module):
    """Causal depthwise conv (kernel K) with GLU: the cortex's short-range sequence wiring."""

    def __init__(self, d, K=4):
        super().__init__()
        self.K = K
        self.norm = RMSNorm(d)
        self.inp = nn.Linear(d, 2 * d, bias=False)
        self.conv = nn.Parameter(torch.randn(d, K) * 0.3)
        self.out = nn.Linear(d, d, bias=False)

    def _conv(self, u):  # u: B,T,d
        u = F.pad(u.transpose(1, 2), (self.K - 1, 0))
        return F.conv1d(u, self.conv.unsqueeze(1).to(u.dtype), groups=u.shape[1]).transpose(1, 2)

    def forward(self, x):
        u, gate = self.inp(self.norm(x)).chunk(2, -1)
        return self.out(self._conv(u) * F.silu(gate))

    def step(self, x, state):  # x: B,d ; state: B,K-1,d (previous u's)
        u, gate = self.inp(self.norm(x)).chunk(2, -1)
        buf = torch.cat([state, u.unsqueeze(1)], 1)          # B,K,d  oldest..newest
        y = (buf * self.conv.t().unsqueeze(0)).sum(1)         # conv weight index K-1 is newest
        return self.out(y * F.silu(gate)), buf[:, 1:]


class MLP(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.norm = RMSNorm(d)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.relu(self.up(self.norm(x))).square())


# ---------------------------------------------------------------- hippocampus

class FastMemory(nn.Module):
    """Fixed-size fast-synapse memory with surprise-gated Hebbian writes; one fused projection."""

    def __init__(self, d, heads=4, dk=64, dv=64):
        super().__init__()
        self.h, self.dk, self.dv = heads, dk, dv
        self.norm = RMSNorm(d)
        self.proj = nn.Linear(d, heads * (2 * dk + 2 * dv), bias=False)   # keys, queries, values, output gate
        self.wo = nn.Linear(heads * dv, d, bias=False)
        self.rnorm = RMSNorm(dv)
        # per-head synaptic decay, spread over timescales: lam = exp(-softplus(p))
        self.decay = nn.Parameter(torch.log(torch.expm1(torch.linspace(0.002, 0.1, heads))))

    def loglam(self):  # log decay, clamped so half-life >= ~5 tokens (keeps parallel form in range)
        return -F.softplus(self.decay).clamp(max=0.15)

    def split(self, xn):
        *lead, _ = xn.shape
        k, q, v, og = self.proj(xn).view(*lead, self.h, -1).split([self.dk, self.dk, self.dv, self.dv], -1)
        return F.normalize(k, dim=-1), F.normalize(q, dim=-1), v, og

    def forward(self, x, g):  # x: B,T,d ; g: B,T write gate in [0,1]
        B, T, _ = x.shape
        k, q, v, og = self.split(self.norm(x))
        k, q, v = k.transpose(1, 2), q.transpose(1, 2), v.transpose(1, 2)    # B,H,T,*
        # decay lam^(t-s-1) factored as lam^(t-c) * lam^(c-s-1) (c = T/2 keeps both in range),
        # write gate folded into the keys, causal mask applied once
        loglam = self.loglam().view(1, self.h, 1, 1)
        t = torch.arange(T, device=x.device, dtype=torch.float32).view(1, 1, T, 1) - T // 2
        qd = q.float() * torch.exp(t * loglam)
        kd = k.float() * (torch.exp(-(t + 1) * loglam) * g.float().view(B, 1, T, 1))
        S = (qd @ kd.transpose(-1, -2)).tril(-1)                              # strictly past: read before write
        r = self.rnorm((S.to(v.dtype) @ v).transpose(1, 2))                   # B,T,H,dv
        return self.wo((r * torch.sigmoid(og)).reshape(B, T, -1))

    def step(self, x, g, M):  # x: B,d ; g: B ; M: B,H,dv,dk
        B = x.shape[0]
        k, q, v, og = self.split(self.norm(x))                               # B,H,*
        r = self.rnorm((M @ q.unsqueeze(-1)).squeeze(-1))
        lam = torch.exp(self.loglam()).view(1, self.h, 1, 1)
        M = lam * M + g.view(B, 1, 1, 1) * v.unsqueeze(-1) * k.unsqueeze(-2)
        return self.wo((r * torch.sigmoid(og)).reshape(B, -1)), M


class SGHM(nn.Module):
    def __init__(self, vocab, d=256, n_blocks=4, mem_at=1, mlp_hidden=1024, heads=4, dk=64, dv=64,
                 gate_mode="surprise", use_mem=True):
        super().__init__()
        self.gate_mode, self.mem_at = gate_mode, mem_at
        self.emb = nn.Embedding(vocab, d)
        self.mix = nn.ModuleList([LocalMix(d) for _ in range(n_blocks)])
        self.mlp = nn.ModuleList([MLP(d, mlp_hidden) for _ in range(n_blocks)])
        self.mem = FastMemory(d, heads, dk, dv) if use_mem else None
        self.pred_norm = RMSNorm(d)
        self.pred = nn.Linear(d, d, bias=False)       # predicts next token's embedding
        self.norm = RMSNorm(d)
        # surprise gate g = sigmoid(alpha * (s - beta)), s = 1 - cos in [0, 2]
        self.g_alpha = nn.Parameter(torch.tensor(4.993))  # softplus -> 5.0
        self.g_beta = nn.Parameter(torch.tensor(0.5))
        nn.init.normal_(self.emb.weight, std=0.02)

    def gate_from_surprise(self, s):
        if self.gate_mode == "always":
            return torch.ones_like(s)
        return torch.sigmoid(F.softplus(self.g_alpha) * (s - self.g_beta))

    def predict(self, x):
        return F.normalize(self.pred(self.pred_norm(x)).float(), dim=-1)

    def forward(self, idx, targets=None):
        e = self.emb(idx)
        x = e
        g = cos = None
        for i in range(len(self.mix)):
            x = x + self.mix[i](x)
            x = x + self.mlp[i](x)
            if i == 0:
                # predictive coding: how well did the cortex at t-1 predict token t's embedding?
                z = self.predict(x)
                en = F.normalize(e.detach().float(), dim=-1)
                cos = (z[:, :-1] * en[:, 1:]).sum(-1)                                   # B,T-1
                s = torch.cat([cos.new_full((cos.shape[0], 1), 2.0), 1 - cos.detach()], 1)
                g = self.gate_from_surprise(s)
            if i == self.mem_at and self.mem is not None:
                x = x + self.mem(x, g)
        logits = self.norm(x) @ self.emb.weight.t().to(x.dtype)
        if targets is None:
            return logits, None, {}
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
        stats = {"pred_loss": (1 - cos).mean(), "write_frac": (g > 0.5).float().mean(), "g_mean": g.mean()}
        return logits, loss, stats

    # ---- O(1)-state recurrent inference
    def init_state(self, B, device):
        d = self.emb.weight.shape[1]
        st = {"conv": [torch.zeros(B, m.K - 1, d, device=device) for m in self.mix], "prev_z": None,
              "M": torch.zeros(B, self.mem.h, self.mem.dv, self.mem.dk, device=device) if self.mem else None}
        return st

    @torch.no_grad()
    def step(self, tok, st):  # tok: B
        e = self.emb(tok)
        x = e
        g = None
        for i in range(len(self.mix)):
            y, st["conv"][i] = self.mix[i].step(x, st["conv"][i])
            x = x + y
            x = x + self.mlp[i](x)
            if i == 0:
                en = F.normalize(e.float(), dim=-1)
                if st["prev_z"] is None:
                    s = x.new_full((x.shape[0],), 2.0)
                else:
                    s = 1 - (st["prev_z"] * en).sum(-1)
                g = self.gate_from_surprise(s)
                st["prev_z"] = self.predict(x)
            if i == self.mem_at and self.mem is not None:
                y, st["M"] = self.mem.step(x, g, st["M"])
                x = x + y
        return self.norm(x) @ self.emb.weight.t(), st, g


# ---------------------------------------------------------------- baseline

class Attn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope(q, cos, sin).to(v.dtype), rope(k, cos, sin).to(v.dtype)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, T, C))


def rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class GPT(nn.Module):
    def __init__(self, vocab, d=256, n_layers=5, heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                "n1": RMSNorm(d), "attn": Attn(d, heads), "n2": RMSNorm(d),
                "up": nn.Linear(d, 4 * d, bias=False), "down": nn.Linear(4 * d, d, bias=False)}))
        self.norm = RMSNorm(d)
        self.hd = d // heads
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        inv = 1.0 / (10000 ** (torch.arange(0, self.hd, 2, device=idx.device).float() / self.hd))
        ang = torch.arange(T, device=idx.device).float()[:, None] * inv[None]
        cos, sin = ang.cos(), ang.sin()
        x = self.emb(idx)
        for b in self.blocks:
            x = x + b["attn"](b["n1"](x), cos, sin)
            x = x + b["down"](F.gelu(b["up"](b["n2"](x))))
        logits = self.norm(x) @ self.emb.weight.t().to(x.dtype)
        if targets is None:
            return logits, None, {}
        return logits, F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1)), {}


def build(name, vocab):
    """Returns a model carrying its training defaults: train_seq, train_batch, base_lr (at train_batch)."""
    if name.startswith("hpt"):
        from hpt import build_hpt
        return build_hpt(name, vocab)
    if name == "gpt":
        m = GPT(vocab)
    elif name == "sghm":
        m = SGHM(vocab)
    elif name == "sghm_nogate":
        m = SGHM(vocab, gate_mode="always")
    elif name == "sghm_nomem":
        m = SGHM(vocab, use_mem=False)
    else:
        raise ValueError(name)
    m.train_seq, m.train_batch, m.base_lr = 256, 128, 4e-3
    return m
