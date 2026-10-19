"""FGPT: a GPT tuned for loss per second (modded-nanogpt-style changes, sized for a 1-minute budget).

Changes against the GPT baseline in models.py, each aimed at loss per FLOP or FLOPs per token:
  - QK-norm + RoPE attention, relu^2 MLP, zero-initialised residual outputs (stable at high learning rates,
    which is what Muon wants)
  - untied input embedding and an adaptive-softmax output (hpt.AdaptiveHead): the 1024 frequent tokens are
    scored everywhere and the rare tail only where needed, cutting the head from ~20% of the FLOPs to ~7%
Meant to be trained with OPT=muon (see optim.py).
"""
import json, os
import torch
import torch.nn as nn
from hpt import Block, Norm, AdaptiveHead

PRESETS = {
    "fgpt_s": dict(d=192, L=4, heads=3),
    "fgpt":   dict(d=256, L=4, heads=4),
    "fgpt_l": dict(d=256, L=6, heads=4),
    "fgpt_w": dict(d=384, L=4, heads=6),
}


class FGPT(nn.Module):
    def __init__(self, vocab, d=256, L=4, heads=4, mlp=4, n_head=1024, lr=4e-3, train_seq=256, train_batch=128):
        super().__init__()
        self.d, self.L = d, L
        self.base_lr, self.train_seq, self.train_batch = lr, train_seq, train_batch
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, heads, mlp) for _ in range(L)])
        self.head = AdaptiveHead(d, vocab, n_head)
        self.hd = d // heads
        for n, p in self.named_parameters():
            if p.dim() == 2:
                nn.init.normal_(p, std=0.02)
        nn.init.normal_(self.emb.weight, std=1.0)
        for b in self.blocks:
            nn.init.zeros_(b.o.weight); nn.init.zeros_(b.down.weight)
        nn.init.zeros_(self.head.head.weight)
        if self.head.tail is not None:
            nn.init.zeros_(self.head.tail.weight)

    def set_code_order(self, counts):
        self.head.set_code_order(counts)

    def macs_per_token(self, T=None):
        T = T or self.train_seq
        return self.L * (12 * self.d * self.d + self.d * T) + self.head.macs(self.d)

    def hidden(self, idx):
        T = idx.shape[1]
        pos = torch.arange(T, device=idx.device, dtype=torch.float32)
        inv = 1.0 / (10000 ** (torch.arange(0, self.hd, 2, device=idx.device, dtype=torch.float32) / self.hd))
        ang = (pos[:, None] * inv[None]).unsqueeze(1)
        cos, sin = ang.cos(), ang.sin()
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, cos, sin)
        return x

    def token_nll(self, idx, targets):
        return self.head.nll(self.hidden(idx), targets)

    def forward(self, idx, targets=None):
        h = self.hidden(idx)
        if targets is None:
            return self.head.logprobs(h[:, -1]).unsqueeze(1), None, {}
        return None, self.head.nll(h, targets).mean(), {}


def build_fgpt(name, vocab):
    """CFG='{"L":5,"mlp":3}' overrides preset fields (used by tune.sh)."""
    return FGPT(vocab, **{**PRESETS[name], **json.loads(os.environ.get("CFG") or "{}")})
