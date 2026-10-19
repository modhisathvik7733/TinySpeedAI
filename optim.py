"""Muon (Jordan et al. 2024, used in the modded-nanogpt speedruns): momentum SGD whose update for each
weight matrix is orthogonalised by a few Newton-Schulz iterations. It typically reaches a given loss in
25-35% fewer steps than AdamW. Embeddings, output heads, norms and small parameters stay on AdamW."""
import os
import torch


def _orthogonalize(G, steps: int = 5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    tall = G.size(-2) > G.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return X.mT if tall else X


orthogonalize = (torch.compile(_orthogonalize) if torch.cuda.is_available() and os.environ.get("COMPILE", "1") == "1"
                 else _orthogonalize)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.03, momentum=0.95, nesterov=True, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                buf = self.state[p].setdefault("momentum", torch.zeros_like(g))
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp(buf, group["momentum"]) if group["nesterov"] else buf
                u = orthogonalize(g, group["ns_steps"]).to(p.dtype)
                p.add_(u, alpha=-group["lr"] * max(1.0, p.size(-2) / p.size(-1)) ** 0.5)


def is_muon_param(name, p):
    """Hidden weight matrices only: not embeddings, output heads, conv taps or norm gains."""
    return p.dim() == 2 and min(p.shape) >= 32 and not any(k in name for k in ("emb", "head"))


class Combined:
    """Steps several optimizers as one (enough of the Optimizer API for train.py)."""

    def __init__(self, *opts):
        self.opts = opts
        self.param_groups = [g for o in opts for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()

    def state_dict(self):
        return [o.state_dict() for o in self.opts]

    def load_state_dict(self, sd):
        for o, s in zip(self.opts, sd):
            o.load_state_dict(s)
