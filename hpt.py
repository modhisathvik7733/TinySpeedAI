"""HPT (Hierarchical Patch Transformer): a language model built for training tokens/second.

Why the old models top out below 1M tok/s on an RTX 5060 Ti: a dense model costs ~6 x params FLOPs per
training token, so a 5M-param model needs ~30 MFLOP/token and already runs at ~45% of the GPU's bf16 peak.
More tok/s therefore needs fewer FLOPs per token, not better kernels. HPT decouples parameters from
per-token compute in three ways:

  1. Patch backbone. Every P consecutive tokens are folded into one patch vector; a standard transformer
     (RoPE, QK-norm, flash attention, relu^2 MLP) runs over patches only, so its cost per token is divided
     by P while its parameters are not. Attention is over T/P positions, so long context is cheap too.
  2. Thin token stream. A narrow (d << D) per-token stream adds the patch context back to every token and
     resolves the within-patch detail with causal gated short convolutions (fused shift-adds, no cuDNN).
  3. Adaptive softmax. The 4096-way softmax is the largest per-token cost (and ~30 KB/token of logits
     traffic). Every position scores only the 1024 most frequent tokens plus a "tail" slot; the other
     3072 are scored only where the target is one of them (~8% of positions). Exact likelihood, every
     token keeps its own output vector, ~3x cheaper than the full softmax.
     (v1 used a class x member factorisation with a shared member layer: fast, but it could not tell
     rare tokens apart - content-token loss ~7.5 nats - so it was replaced.)

Causality: tokens of patch k see the backbone output of patch k-1 (which summarises tokens < kP) plus
the causal token stream, so no token ever sees its own future (checked by test_causal.py).

Prior art: MegaByte (Yu et al. 2023) and Block Transformer (Ho et al. 2024) use the global-patch /
local-token split; adaptive softmax is Grave et al. 2017.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# name -> config. Analytic forward multiply-adds per token are printed by train.py (macs_per_token).
PRESETS = {
    # 2.2 MFLOP/token training, 4.3M params (v1 head: 8.5M tok/s measured on an RTX 5060 Ti; ~7M expected now)
    "hpt":   dict(d=64,  P=16, D=256, L=4,  heads=4,  tok_layers=2, K=8, lr=4e-3),
    # 1.4 MFLOP/token, 2.5M params: the >=10M tok/s preset on an RTX 5060 Ti
    "hpt_xxs": dict(d=48, P=16, D=192, L=4, heads=3,  tok_layers=2, K=8, lr=5e-3),
    # 6.0 MFLOP/token, 12.7M params: more quality, ~3.5M tok/s on a 5060 Ti
    "hpt_s": dict(d=96,  P=16, D=384, L=6,  heads=6,  tok_layers=2, K=8, lr=3e-3),
    # scaling presets for bigger GPUs / more data (same code path, only widths change)
    "hpt_m": dict(d=128, P=8,  D=768, L=12, heads=12, tok_layers=2, K=8, lr=1.5e-3),
    "hpt_l": dict(d=256, P=8,  D=1536, L=24, heads=16, tok_layers=3, K=8, lr=6e-4),
}


def rms(x, w=None):
    return F.rms_norm(x, (x.shape[-1],), w, 1e-6)


class Norm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return rms(x, self.w)


def rope(x, cos, sin):  # x: B,T,H,hd (half-split rotation)
    x1, x2 = x.chunk(2, -1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class Block(nn.Module):
    """Pre-norm transformer block over patches: QK-norm + RoPE flash attention, relu^2 MLP."""

    def __init__(self, D, heads, mlp=4):
        super().__init__()
        self.h, self.hd = heads, D // heads
        self.n1, self.n2 = Norm(D), Norm(D)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.o = nn.Linear(D, D, bias=False)
        self.up = nn.Linear(D, mlp * D, bias=False)
        self.down = nn.Linear(mlp * D, D, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, self.hd).unbind(2)
        q, k = rope(rms(q), cos, sin).to(v.dtype), rope(rms(k), cos, sin).to(v.dtype)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        x = x + self.o(y.transpose(1, 2).reshape(B, T, C))
        return x + self.down(F.relu(self.up(self.n2(x))).square())


class TokMix(nn.Module):
    """Token stream block: causal depthwise conv (kernel K) with a GLU, then a small relu^2 MLP."""

    def __init__(self, d, K, mlp=2):
        super().__init__()
        self.K = K
        self.n1, self.n2 = Norm(d), Norm(d)
        self.inp = nn.Linear(d, 2 * d, bias=False)
        w = torch.randn(K, d) * 0.1
        w[0] += 1.0                                   # w[k] weights the token k steps back; starts near identity
        self.w = nn.Parameter(w)
        self.out = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, mlp * d, bias=False)
        self.down = nn.Linear(mlp * d, d, bias=False)

    def conv(self, u):  # u: B,T,d. Written as K shifted multiply-adds so torch.compile fuses it into one kernel
        T = u.shape[1]
        up = F.pad(u, (0, 0, self.K - 1, 0))
        w = self.w.to(u.dtype)
        y = up[:, self.K - 1:] * w[0]
        for k in range(1, self.K):
            y = y + up[:, self.K - 1 - k:self.K - 1 - k + T] * w[k]
        return y

    def forward(self, x):
        u, g = self.inp(self.n1(x)).chunk(2, -1)
        x = x + self.out(self.conv(u) * F.silu(g))
        return x + self.down(F.relu(self.up(self.n2(x))).square())


class AdaptiveHead(nn.Module):
    """Exact two-level softmax (Grave et al. 2017). Every position scores the n_head most frequent tokens
    plus one "tail" slot; the rare tail is scored only at positions whose target is rare (~8% on
    TinyStories). Every token keeps its own output vector, so the head stays as expressive as a full
    softmax at a fraction of its cost."""

    def __init__(self, d, vocab, n_head=1024):
        super().__init__()
        self.V, self.n_head = vocab, min(n_head, vocab)
        self.n_tail = vocab - self.n_head
        self.tail_frac = 0.1                                       # updated by set_code_order
        self.norm = Norm(d)
        self.head = nn.Linear(d, self.n_head + (1 if self.n_tail else 0), bias=False)
        self.tail = nn.Linear(d, self.n_tail, bias=False) if self.n_tail else None
        # rank = position in frequency order; ranks < n_head are head tokens
        self.register_buffer("tok2rank", torch.arange(vocab), persistent=True)

    def set_code_order(self, counts):
        counts = torch.as_tensor(counts, dtype=torch.float64)
        rank = torch.argsort(counts, descending=True)
        t2r = torch.empty(self.V, dtype=torch.long)
        t2r[rank] = torch.arange(self.V)
        self.tok2rank.copy_(t2r.to(self.tok2rank.device))
        self.tail_frac = float(counts[rank[self.n_head:]].sum() / counts.sum().clamp(min=1))

    def nll(self, h, y):  # h: ...,d ; y: ... -> per-token nll
        h = self.norm(h)
        r = self.tok2rank[y]
        lh = self.head(h).float()
        if not self.n_tail:
            return F.cross_entropy(lh.reshape(-1, self.n_head), r.reshape(-1), reduction="none").view(y.shape)
        tail = r >= self.n_head
        nll = F.cross_entropy(lh.reshape(-1, self.n_head + 1), torch.where(tail, self.n_head, r).reshape(-1),
                              reduction="none").view(y.shape)
        return nll + self._tail_nll(h, r, tail)

    @torch.compiler.disable  # data-dependent subset of positions: run eagerly, outside the compiled graph
    def _tail_nll(self, h, r, tail):
        idx = tail.nonzero(as_tuple=True)
        lt = self.tail(h[idx]).float()
        ce = F.cross_entropy(lt, r[idx] - self.n_head, reduction="none")
        return torch.zeros(r.shape, device=h.device, dtype=ce.dtype).index_put(idx, ce)

    def logprobs(self, h):  # h: B,d -> B,V full log-distribution (used for sampling)
        h = self.norm(h)
        lh = F.log_softmax(self.head(h).float(), -1)
        if self.n_tail:
            lh = torch.cat([lh[:, :-1], lh[:, -1:] + F.log_softmax(self.tail(h).float(), -1)], -1)
        return lh[:, self.tok2rank]

    def macs(self, d):
        return d * (self.n_head + 1) + self.tail_frac * d * self.n_tail


class HPT(nn.Module):
    def __init__(self, vocab, d=64, P=16, D=256, L=4, heads=4, tok_layers=2, K=8, lr=4e-3,
                 n_head=1024, train_seq=1024, train_batch=128):
        super().__init__()
        self.d, self.P, self.D, self.L = d, P, D, L
        self.base_lr, self.train_seq, self.train_batch = lr, train_seq, train_batch
        self.emb = nn.Embedding(vocab, d)
        self.enc = nn.Linear(P * d, D, bias=False)      # P token embeddings -> one patch vector
        self.enc_norm = Norm(D)
        self.blocks = nn.ModuleList([Block(D, heads) for _ in range(L)])
        self.bnorm = Norm(D)
        self.bos = nn.Parameter(torch.zeros(D))           # context for the first patch
        self.dec = nn.Linear(D, P * d, bias=False)        # patch context -> one vector per position in the next patch
        self.tok = nn.ModuleList([TokMix(d, K) for _ in range(tok_layers)])
        self.head = AdaptiveHead(d, vocab, n_head)
        self.hd = D // heads
        self._init(L + tok_layers)

    def _init(self, depth):
        for n, p in self.named_parameters():
            if p.dim() == 2:
                nn.init.normal_(p, std=0.02)
        nn.init.normal_(self.emb.weight, std=1.0)
        for b in self.blocks:                             # zero-init residual outputs: stable at any depth
            nn.init.zeros_(b.o.weight); nn.init.zeros_(b.down.weight)
        for t in self.tok:
            nn.init.zeros_(t.out.weight); nn.init.zeros_(t.down.weight)
        nn.init.zeros_(self.head.head.weight)
        if self.head.tail is not None:
            nn.init.zeros_(self.head.tail.weight)

    def set_code_order(self, counts):
        self.head.set_code_order(counts)

    def macs_per_token(self, T=None):
        T = T or self.train_seq
        d, P, D, L = self.d, self.P, self.D, self.L
        backbone = L * (12 * D * D + D * T // P) / P     # projections + causal attention over T/P patches
        tok = sum(2 * d * d + d * d + t.K * d + 4 * d * d for t in self.tok)
        return 2 * d * D + backbone + tok + self.head.macs(d)

    def hidden(self, idx):
        B, T0 = idx.shape
        P = self.P
        idx = F.pad(idx, (0, (-T0) % P))                 # right-pad: only future positions are padded
        T = idx.shape[1]
        n = T // P
        e = self.emb(idx)                                                  # B,T,d
        p = self.enc_norm(self.enc(e.reshape(B, n, P * self.d)))           # B,n,D
        pos = torch.arange(n, device=idx.device, dtype=torch.float32)
        inv = 1.0 / (10000 ** (torch.arange(0, self.hd, 2, device=idx.device, dtype=torch.float32) / self.hd))
        ang = (pos[:, None] * inv[None]).unsqueeze(1)                       # n,1,hd/2
        cos, sin = ang.cos(), ang.sin()
        for b in self.blocks:
            p = b(p, cos, sin)
        p = self.bnorm(p)
        ctx = torch.cat([self.bos.to(p.dtype).expand(B, 1, -1), p[:, :-1]], 1)   # patch k <- patches < k
        x = e.to(p.dtype) + self.dec(ctx).reshape(B, T, self.d)
        for t in self.tok:
            x = t(x)
        return x[:, :T0]

    def token_nll(self, idx, targets):
        return self.head.nll(self.hidden(idx), targets)

    def forward(self, idx, targets=None):
        h = self.hidden(idx)
        if targets is None:  # next-token log-probs for the last position, shaped like logits: B,1,V
            return self.head.logprobs(h[:, -1]).unsqueeze(1), None, {}
        return None, self.head.nll(h, targets).mean(), {}


def build_hpt(name, vocab):
    return HPT(vocab, **PRESETS[name])
