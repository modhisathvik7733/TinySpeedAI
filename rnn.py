"""Non-transformer sequence models from recent papers, in one controlled harness.

The skeleton matches the GPT baseline: d=256, 5 blocks, 4x GELU MLP, tied embeddings, AdamW. Only the token
mixer changes, from attention to a recurrent / linear-memory layer with a fixed-size state (no KV cache),
so differences in the speedrun table come from the mechanism, not from tuning.

  rnn_mingru  minGRU (Feng et al. 2024, "Were RNNs All We Needed?"): gates don't depend on h_{t-1}, so training
              is a parallel log-space scan. Pure PyTorch.
  rnn_gdn     Gated DeltaNet (Yang et al., ICLR 2025): fast-weight memory with delta-rule writes + gated decay
  rnn_hgrn2   HGRN2 (Qin et al., COLM 2024): gated linear RNN with outer-product state expansion
  rnn_gla     Gated Linear Attention (Yang et al., ICML 2024): data-dependent gated fast weights
  rnn_retnet  RetNet multi-scale retention (Sun et al. 2023): fixed multi-timescale decaying memory
  rnn_mamba2  Mamba-2 / SSD (Dao & Gu, ICML 2024): selective state-space model
The last five use the Triton kernels of flash-linear-attention (pip install flash-linear-attention einops).
"""
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import RMSNorm
from sac import causal_conv

# fla layer class names tried in order, and per-layer constructor overrides (filtered by signature)
FLA = {
    "gdn":    (["GatedDeltaNet"], dict(head_dim=64, expand_v=1)),
    "hgrn2":  (["HGRN2Attention"], dict(expand_ratio=64)),   # heads = hidden_size // expand_ratio
    "gla":    (["GatedLinearAttention"], dict(expand_k=0.5, expand_v=1)),
    "retnet": (["MultiScaleRetention"], dict(expand_k=1, expand_v=1)),
    "mamba2": (["Mamba2"], dict(head_dim=64, expand=2, num_heads=8)),
}


def parallel_scan_log(log_a, log_b):
    """h_t = a_t h_{t-1} + b_t from h_{-1} = 0, all in log space (a_t, b_t > 0). Shapes B,T,D."""
    a_star = torch.cumsum(log_a, 1)
    return torch.exp(a_star + torch.logcumsumexp(log_b - a_star, 1))


class MinGRU(nn.Module):
    """minGRU with a short causal conv in front: h_t = (1 - z_t) h_{t-1} + z_t g(h~_t)."""

    def __init__(self, d, expand=2, K=4):
        super().__init__()
        w = torch.randn(K, d) * 0.1
        w[0] += 1.0                                   # starts near identity
        self.conv = nn.Parameter(w)
        self.inp = nn.Linear(d, 2 * expand * d, bias=False)
        self.out = nn.Linear(expand * d, d, bias=False)

    def forward(self, x):
        k, h = self.inp(causal_conv(x, self.conv)).float().chunk(2, -1)
        log_z = -F.softplus(-k)                       # log z_t
        log_keep = -F.softplus(k)                     # log (1 - z_t)
        log_h = torch.where(h >= 0, torch.log(F.relu(h) + 0.5), -F.softplus(-h))   # log g(h~), g > 0
        return self.out(parallel_scan_log(log_keep, log_z + log_h).to(x.dtype))

    def step(self, x, h, buf):  # sequential reference for tests: x B,d ; h B,expand*d ; buf B,K-1,d
        win = torch.cat([buf, x.unsqueeze(1)], 1)
        u = (win * self.conv.flip(0).unsqueeze(0)).sum(1)
        k, ht = self.inp(u).float().chunk(2, -1)
        z = torch.sigmoid(k)
        g = torch.where(ht >= 0, ht + 0.5, torch.sigmoid(ht))
        h = (1 - z) * h + z * g
        return self.out(h), h, win[:, 1:]


class FlaMixer(nn.Module):
    def __init__(self, kind, d, heads, layer_idx):
        super().__init__()
        try:
            import fla.layers as L
        except ImportError as e:
            raise ImportError("rnn_%s needs flash-linear-attention: pip install flash-linear-attention einops" % kind) from e
        names, extra = FLA[kind]
        cls = next((getattr(L, n) for n in names if hasattr(L, n)), None)
        if cls is None:
            raise ImportError(f"fla.layers has none of {names}; available: {[n for n in dir(L) if n[0].isupper()]}")
        want = dict(hidden_size=d, num_heads=heads, mode="chunk", layer_idx=layer_idx, **extra)
        sig = inspect.signature(cls.__init__).parameters
        self.m = cls(**{k: v for k, v in want.items() if k in sig})

    def forward(self, x):
        y = self.m(x)
        return y[0] if isinstance(y, tuple) else y


class RNNLM(nn.Module):
    def __init__(self, vocab, kind, d=256, n_layers=5, heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            mix = MinGRU(d) if kind == "mingru" else FlaMixer(kind, d, heads, i)
            self.blocks.append(nn.ModuleDict({
                "n1": RMSNorm(d), "mix": mix, "n2": RMSNorm(d),
                "up": nn.Linear(d, 4 * d, bias=False), "down": nn.Linear(4 * d, d, bias=False)}))
        self.norm = RMSNorm(d)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.emb(idx)
        for b in self.blocks:
            x = x + b["mix"](b["n1"](x))
            x = x + b["down"](F.gelu(b["up"](b["n2"](x))))
        logits = self.norm(x) @ self.emb.weight.t().to(x.dtype)
        if targets is None:
            return logits, None, {}
        return logits, F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1)), {}


def build_rnn(name, vocab):
    m = RNNLM(vocab, name[len("rnn_"):])
    m.train_seq, m.train_batch, m.base_lr = 256, 128, 4e-3     # same as the GPT baseline
    return m
