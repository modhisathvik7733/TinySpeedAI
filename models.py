"""SGHM (Surprise-Gated Hebbian Memory) network and a matched GPT baseline.

SGHM information flow:
  cortex (causal local mixing + sparse k-WTA MLP)  -> local predictor -> surprise s_t (nats)
  fast synapses M (fixed size per head): write  M <- lam*M + g_t * v_t a_t^T   (a_t sparse k-WTA code, g_t = gate(s_t))
                                         read   r_t = M_{t-1} b_t               (b_t sparse query code)
Training uses the exactly equivalent masked parallel form; generation uses the O(1)-state recurrent form.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def kwta(x, k, normalize=True):
    """k-winners-take-all: keep the k largest (rectified) units, zero the rest."""
    vals, idx = x.topk(k, dim=-1)
    out = torch.zeros_like(x).scatter(-1, idx, F.relu(vals))
    if normalize:
        out = out / (out.norm(dim=-1, keepdim=True) + 1e-6)
    return out


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
        return F.conv1d(u, self.conv.unsqueeze(1), groups=u.shape[1]).transpose(1, 2)

    def forward(self, x):
        u, gate = self.inp(self.norm(x)).chunk(2, -1)
        return self.out(self._conv(u) * F.silu(gate))

    def step(self, x, state):  # x: B,d ; state: B,K-1,d (previous u's)
        u, gate = self.inp(self.norm(x)).chunk(2, -1)
        buf = torch.cat([state, u.unsqueeze(1)], 1)          # B,K,d  oldest..newest
        y = (buf * self.conv.t().unsqueeze(0)).sum(1)         # conv weight index K-1 is newest
        return self.out(y * F.silu(gate)), buf[:, 1:]


class SparseMLP(nn.Module):
    """Only k of H hidden neurons fire per token."""

    def __init__(self, d, hidden, k):
        super().__init__()
        self.k = k
        self.norm = RMSNorm(d)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        h = self.up(self.norm(x))
        h = kwta(h, self.k, normalize=False) if self.k else F.relu(h)
        return self.down(h)


# ---------------------------------------------------------------- hippocampus

class HebbianMemory(nn.Module):
    """Fixed-size fast-synapse memory with sparse keys and surprise-gated Hebbian writes."""

    def __init__(self, d, heads=4, n_cells=256, k=8, dv=64, sparse=True):
        super().__init__()
        self.h, self.n, self.k, self.dv, self.sparse = heads, n_cells, k, dv, sparse
        self.norm = RMSNorm(d)
        self.wk = nn.Linear(d, heads * n_cells, bias=False)
        self.wq = nn.Linear(d, heads * n_cells, bias=False)
        self.wv = nn.Linear(d, heads * dv, bias=False)
        self.wg = nn.Linear(d, heads * dv, bias=False)   # cortex decides how much recall to use
        self.wo = nn.Linear(heads * dv, d, bias=False)
        self.rnorm = RMSNorm(dv)
        # per-head synaptic decay, spread over timescales: lam = exp(-softplus(p))
        self.decay = nn.Parameter(torch.log(torch.expm1(torch.linspace(0.002, 0.1, heads))))

    def loglam(self):  # log decay, clamped so half-life >= ~5 tokens (keeps parallel form in fp32 range)
        return -F.softplus(self.decay).clamp(max=0.15)

    def codes(self, xn):
        B, T, _ = xn.shape
        a = self.wk(xn).view(B, T, self.h, self.n)
        b = self.wq(xn).view(B, T, self.h, self.n)
        if self.sparse:
            a, b = kwta(a, self.k), kwta(b, self.k)
        else:  # dense ablation: unit-norm dense codes
            a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
        return a, b

    def forward(self, x, g):  # x: B,T,d ; g: B,T write gate in [0,1]
        B, T, _ = x.shape
        xn = self.norm(x)
        a, b = self.codes(xn)
        v = self.wv(xn).view(B, T, self.h, self.dv)
        a, b, v = a.transpose(1, 2), b.transpose(1, 2), v.transpose(1, 2)    # B,H,T,*
        # decay lam^(t-s-1) factored as lam^(t-c) * lam^(c-s-1) (c = T/2 keeps both in fp32 range),
        # write gate folded into the keys, causal mask applied once: avoids materialising a B,H,T,T decay tensor
        loglam = self.loglam().view(1, self.h, 1, 1)
        t = torch.arange(T, device=x.device, dtype=x.dtype).view(1, 1, T, 1) - T // 2
        bq = b * torch.exp(t * loglam)
        ak = a * (torch.exp(-(t + 1) * loglam) * g.view(B, 1, T, 1))
        S = (bq @ ak.transpose(-1, -2)).tril(-1)                              # strictly past: read before write
        r = self.rnorm(S @ v).transpose(1, 2).reshape(B, T, -1)
        return self.wo(r * torch.sigmoid(self.wg(xn)))

    def step(self, x, g, M):  # x: B,d ; g: B ; M: B,H,dv,N
        B = x.shape[0]
        xn = self.norm(x)
        a, b = self.codes(xn.unsqueeze(1))
        a, b = a[:, 0], b[:, 0]                                               # B,H,N
        v = self.wv(xn).view(B, self.h, self.dv)
        r = self.rnorm((M @ b.unsqueeze(-1)).squeeze(-1)).reshape(B, -1)
        lam = torch.exp(self.loglam()).view(1, self.h, 1, 1)
        M = lam * M + g.view(B, 1, 1, 1) * v.unsqueeze(-1) * a.unsqueeze(-2)
        return self.wo(r * torch.sigmoid(self.wg(xn))), M


class SGHM(nn.Module):
    def __init__(self, vocab, d=256, n_blocks=4, mem_after=(0, 2), mlp_hidden=1024, mlp_k=64,
                 heads=4, n_cells=256, cell_k=8, dv=64, gate_mode="surprise", sparse_keys=True):
        super().__init__()
        self.gate_mode = gate_mode
        self.emb = nn.Embedding(vocab, d)
        self.mix = nn.ModuleList([LocalMix(d) for _ in range(n_blocks)])
        self.mlp = nn.ModuleList([SparseMLP(d, mlp_hidden, mlp_k) for _ in range(n_blocks)])
        self.mem_after = list(mem_after)
        self.mem = nn.ModuleDict({str(i): HebbianMemory(d, heads, n_cells, cell_k, dv, sparse_keys)
                                  for i in mem_after})
        self.local_norm = RMSNorm(d)
        self.norm = RMSNorm(d)
        # surprise gate g = sigmoid(alpha * (s - beta)), s in nats
        self.g_alpha = nn.Parameter(torch.tensor(0.5413))  # softplus -> 1.0
        self.g_beta = nn.Parameter(torch.tensor(2.0))
        nn.init.normal_(self.emb.weight, std=0.02)

    def logits(self, h, norm):
        return norm(h) @ self.emb.weight.t()

    def gate_from_surprise(self, s):
        if self.gate_mode == "always":
            return torch.ones_like(s)
        return torch.sigmoid(F.softplus(self.g_alpha) * (s - self.g_beta))

    def forward(self, idx, targets=None):
        x = self.emb(idx)
        g = local_logits = None
        for i in range(len(self.mix)):
            x = x + self.mix[i](x)
            x = x + self.mlp[i](x)
            if i == 0:
                # local predictor: how predictable was token t from the cortex at t-1?
                local_logits = self.logits(x, self.local_norm)
                lp = F.log_softmax(local_logits[:, :-1].detach(), -1)
                s = -lp.gather(-1, idx[:, 1:, None]).squeeze(-1)                       # B,T-1 nats
                s = torch.cat([s.new_full((s.shape[0], 1), 10.0), s], 1)               # first token: maximal surprise
                g = self.gate_from_surprise(s)
            if str(i) in self.mem:
                x = x + self.mem[str(i)](x, g)
        logits = self.logits(x, self.norm)
        if targets is None:
            return logits, None, {}
        V = logits.shape[-1]
        loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
        local_loss = F.cross_entropy(local_logits.reshape(-1, V), targets.reshape(-1))
        stats = {"local_loss": local_loss, "write_frac": (g > 0.5).float().mean(), "g_mean": g.mean()}
        return logits, loss, stats

    # ---- O(1)-state recurrent inference
    def init_state(self, B, device):
        d = self.emb.weight.shape[1]
        st = {"conv": [torch.zeros(B, m.K - 1, d, device=device) for m in self.mix],
              "M": {k: torch.zeros(B, m.h, m.dv, m.n, device=device) for k, m in self.mem.items()},
              "prev_lp": None}
        return st

    @torch.no_grad()
    def step(self, tok, st):  # tok: B
        x = self.emb(tok)
        for i in range(len(self.mix)):
            y, st["conv"][i] = self.mix[i].step(x, st["conv"][i])
            x = x + y
            x = x + self.mlp[i](x.unsqueeze(1))[:, 0]
            if i == 0:
                if st["prev_lp"] is None:
                    s = x.new_full((x.shape[0],), 10.0)
                else:
                    s = -st["prev_lp"].gather(-1, tok[:, None]).squeeze(-1)
                g = self.gate_from_surprise(s)
                st["prev_lp"] = F.log_softmax(self.logits(x, self.local_norm), -1)
            if str(i) in self.mem:
                y, st["M"][str(i)] = self.mem[str(i)].step(x, g, st["M"][str(i)])
                x = x + y
        return self.logits(x, self.norm), st, g


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
        q, k = rope(q, cos, sin), rope(k, cos, sin)
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
        logits = self.norm(x) @ self.emb.weight.t()
        if targets is None:
            return logits, None, {}
        return logits, F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)), {}


def build(name, vocab):
    if name == "gpt":
        return GPT(vocab)
    if name == "sghm":
        return SGHM(vocab)
    if name == "sghm_nogate":
        return SGHM(vocab, gate_mode="always")
    if name == "sghm_dense":
        return SGHM(vocab, sparse_keys=False)
    if name == "sghm_nomem":
        return SGHM(vocab, mem_after=())
    raise ValueError(name)
