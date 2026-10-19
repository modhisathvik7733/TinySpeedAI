"""SAC (Sparse Associative Cortex): a language model whose intelligence comes from how memory is organised,
not from multiplying every weight by every token. See DESIGN.md, "SAC".

Information flow (no attention, no KV cache, nothing grows with context):
  token -> embedding -> a sequence of ops on one residual stream (layout string, e.g. "cpcecpce"):
    c  local wiring: causal depthwise conv, channel-gated. No dense projection (~2K MACs/token).
    p  semantic memory (cortex): product-key associative memory with N = n_sub^2 slots. The query fires only
       the top-k slots (k=32 of 65k..1M, <0.05%); scoring costs O(sqrt N), reading O(k). Parameters grow
       with N, compute does not.
    e  episodic memory (hippocampus): fixed-size fast-weight state per head, S <- lam*S + g_t v_t k_t^T,
       read r_t = S_{t-1} q_t, write gate g_t driven by surprise. Trained in a chunk-parallel form with no
       sequential scan; generation uses the O(1)-state recurrence (`step`).
  surprise = 1 - cos(prediction_{t-1}, embedding(x_t)), predicted from the stream before the first 'e'.
  -> adaptive-softmax head (hpt.AdaptiveHead).
"""
import json, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from hpt import Norm, AdaptiveHead

PRESETS = {
    # speed default: ~5 MFLOP/token, 65k memory slots
    "sac":        dict(d=256, layout="cpcecpce", heads=2, dh=64, n_sub=256, k=32),
    # smaller/faster
    "sac_s":      dict(d=192, layout="cpcecpce", heads=2, dh=48, n_sub=256, k=32),
    # quality variant: more heads and ops
    "sac_q":      dict(d=320, layout="cpcecpcecpce", heads=4, dh=64, n_sub=256, k=32),
    # memory-size scaling (compute nearly unchanged): 262k and 1M slots
    "sac_m262k":  dict(d=256, layout="cpcecpce", heads=2, dh=64, n_sub=512, k=32),
    "sac_m1m":    dict(d=256, layout="cpcecpce", heads=2, dh=64, n_sub=1024, k=32),
    # ablations: every mechanism has to earn its place
    "sac_nopkm":  dict(d=256, layout="ccceccce", heads=2, dh=64, n_sub=256, k=32),
    "sac_noepi":  dict(d=256, layout="cpcccpcc", heads=2, dh=64, n_sub=256, k=32),
    "sac_nogate": dict(d=256, layout="cpcecpce", heads=2, dh=64, n_sub=256, k=32, gate=False),
}


def causal_conv(u, w):  # u: B,T,d ; w: K,d with w[j] weighting the token j steps back. Fuses under compile.
    K, T = w.shape[0], u.shape[1]
    up = F.pad(u, (0, 0, K - 1, 0))
    w = w.to(u.dtype)
    y = up[:, K - 1:] * w[0]
    for j in range(1, K):
        y = y + up[:, K - 1 - j:K - 1 - j + T] * w[j]
    return y


class Wiring(nn.Module):
    """Local wiring: two causal depthwise convs, one gating the other. Per-channel, no dense projection."""

    def __init__(self, d, K=4):
        super().__init__()
        self.K = K
        self.norm = Norm(d)
        wa = torch.randn(K, d) * 0.1
        wa[0] += 1.0
        self.wa = nn.Parameter(wa)
        self.wb = nn.Parameter(torch.randn(K, d) * 0.1)
        self.scale = nn.Parameter(torch.zeros(d))         # zero-init residual

    def forward(self, x):
        u = self.norm(x)
        return self.scale.to(u.dtype) * causal_conv(u, self.wa) * F.silu(causal_conv(u, self.wb))

    def step(self, x, buf):  # x: B,d ; buf: B,K-1,d of previous normalised inputs (oldest first)
        u = self.norm(x)
        win = torch.cat([buf, u.unsqueeze(1)], 1)                 # B,K,d oldest..newest
        a = (win * self.wa.flip(0).unsqueeze(0)).sum(1)
        b = (win * self.wb.flip(0).unsqueeze(0)).sum(1)
        return self.scale * a * F.silu(b), win[:, 1:]

    def macs(self):
        return 2 * self.K * self.wa.shape[1]


class ProductKeyMemory(nn.Module):
    """Sparse associative memory (Lample et al. 2019 product keys): N = n_sub^2 value slots, only k fire."""

    def __init__(self, d, n_sub=256, dq=128, k=32):
        super().__init__()
        self.n_sub, self.dq, self.k = n_sub, dq, k
        self.norm = Norm(d)
        self.q = nn.Linear(d, dq, bias=False)
        self.pkm_keys = nn.Parameter(torch.randn(2, n_sub, dq // 2) / math.sqrt(dq // 2))
        self.pkm_values = nn.Parameter(torch.randn(n_sub * n_sub, d) / math.sqrt(d))

    def lookup(self, x):  # x: N,d -> slot ids N,k and weights N,k
        q = self.q(self.norm(x)).view(-1, 2, self.dq // 2)
        q = F.rms_norm(q, (self.dq // 2,))
        s1 = q[:, 0] @ self.pkm_keys[0].to(q.dtype).t()                # N,n_sub
        s2 = q[:, 1] @ self.pkm_keys[1].to(q.dtype).t()
        v1, i1 = s1.float().topk(self.k, -1)
        v2, i2 = s2.float().topk(self.k, -1)
        cand = (v1.unsqueeze(2) + v2.unsqueeze(1)).flatten(1)           # N,k*k
        vs, ii = cand.topk(self.k, -1)
        idx = i1.gather(1, ii // self.k) * self.n_sub + i2.gather(1, ii % self.k)
        return idx, F.softmax(vs, -1)

    def forward(self, x):
        shape = x.shape
        idx, w = self.lookup(x.reshape(-1, shape[-1]))
        out = F.embedding_bag(idx, self.pkm_values, per_sample_weights=w.to(self.pkm_values.dtype), mode="sum")
        return out.to(x.dtype).view(shape)

    def macs(self):
        return self.q.in_features * self.dq + self.n_sub * self.dq + self.k * self.k + self.k * self.pkm_values.shape[1]


class EpisodicMemory(nn.Module):
    """Fixed-size fast synapses per head: S <- lam*S + g v k^T ; r_t = S_{t-1} q_t (read before write).

    Chunk-parallel training form, chunk size C, n = T/C chunks, i = position inside the chunk:
      intra-chunk : r_t += sum_{s<t, same chunk} lam^(t-1-s) g_s (q_t.k_s) v_s       (C x C masked matmul)
      chunk sums  : U_c = sum_j lam^(C-1-j) g_j v_j k_j^T                              (batched matmul)
      chunk states: S_c = sum_{m<c} lam^(C(c-1-m)) U_m                                 (n x n decay matmul, no scan)
      inter-chunk : r_t += lam^i S_c q_t
    Every exponent is >= 0, so nothing overflows at any length. Cost per token is constant in T."""

    def __init__(self, d, heads=2, dh=64, C=64):
        super().__init__()
        self.h, self.dh, self.C = heads, dh, C
        self.norm = Norm(d)
        self.proj = nn.Linear(d, heads * 3 * dh, bias=False)              # q, k, v
        self.wo = nn.Linear(heads * dh, d, bias=False)
        self.rnorm = Norm(dh)
        # per-head decay spread over timescales (half-lives ~7 .. ~700 tokens)
        self.decay = nn.Parameter(torch.log(torch.expm1(torch.logspace(-3, -1, heads))))

    def loglam(self):
        return -F.softplus(self.decay)

    def qkv(self, x):
        *lead, _ = x.shape
        q, k, v = self.proj(self.norm(x)).view(*lead, self.h, 3 * self.dh).split(self.dh, -1)
        return F.normalize(q, dim=-1), F.normalize(k, dim=-1), v

    def forward(self, x, g):  # x: B,T,d ; g: B,T write gate
        B, T0, _ = x.shape
        C = self.C
        q, k, v = self.qkv(x)                                                # B,T,H,dh
        k = k * g.unsqueeze(-1).unsqueeze(-1).to(k.dtype)
        pad = (-T0) % C
        if pad:  # right-pad: padded positions have zero keys (no writes) and come after every real position
            q, k, v = (F.pad(t, (0, 0, 0, 0, 0, pad)) for t in (q, k, v))
        T = T0 + pad
        n = T // C
        q, k, v = (t.view(B, n, C, self.h, self.dh).permute(0, 3, 1, 2, 4).float() for t in (q, k, v))  # B,H,n,C,dh
        ll = self.loglam().view(self.h, 1, 1)
        i = torch.arange(C, device=x.device, dtype=torch.float32)
        diff = i.view(C, 1) - i.view(1, C) - 1                               # t-1-s
        intra_mask = torch.where(diff >= 0, torch.exp(diff.clamp(min=0) * ll), 0.0)           # H,C,C
        A = (q @ k.transpose(-1, -2)) * intra_mask.unsqueeze(1)                                 # B,H,n,C,C
        r = A @ v
        wk = torch.exp((C - 1 - i) * ll).view(self.h, 1, C, 1)                                  # lam^(C-1-j)
        U = v.transpose(-1, -2) @ (k * wk)                                                       # B,H,n,dh,dh
        c = torch.arange(n, device=x.device, dtype=torch.float32)
        cd = c.view(n, 1) - c.view(1, n) - 1
        chunk_mix = torch.where(cd >= 0, torch.exp(cd.clamp(min=0) * C * ll), 0.0)            # H,n,n
        S = torch.einsum("hcm,bhmvk->bhcvk", chunk_mix, U)                                       # state at chunk start
        r = r + torch.exp(i * ll).view(self.h, 1, C, 1) * (q @ S.transpose(-1, -2))
        r = r.permute(0, 2, 3, 1, 4).reshape(B, T, self.h, self.dh)[:, :T0]
        return self.wo(self.rnorm(r.to(x.dtype)).reshape(B, T0, -1))

    def step(self, x, g, S):  # x: B,d ; g: B ; S: B,H,dh(v),dh(k)
        q, k, v = self.qkv(x)                                                # B,H,dh
        r = (S @ q.unsqueeze(-1)).squeeze(-1)
        S = torch.exp(self.loglam()).view(1, self.h, 1, 1) * S + g.view(-1, 1, 1, 1) * v.unsqueeze(-1) * k.unsqueeze(-2)
        return self.wo(self.rnorm(r).reshape(x.shape[0], -1)), S

    def macs(self, T):
        d, H, dh, C = self.proj.in_features, self.h, self.dh, self.C
        return d * 3 * H * dh + H * dh * d + H * (C * dh + 3 * dh * dh + (T // C) * dh * dh / C)


class SAC(nn.Module):
    def __init__(self, vocab, d=256, layout="cpcecpce", heads=2, dh=64, n_sub=256, dq=128, k=32, C=64, K=4,
                 gate=True, n_head=1024, lr=4e-3, train_seq=256, train_batch=128):
        super().__init__()
        self.d, self.layout, self.gate = d, layout, gate
        self.base_lr, self.train_seq, self.train_batch = lr, train_seq, train_batch
        self.emb = nn.Embedding(vocab, d)
        ops = {"c": lambda: Wiring(d, K), "p": lambda: ProductKeyMemory(d, n_sub, dq, k),
               "e": lambda: EpisodicMemory(d, heads, dh, C)}
        self.ops = nn.ModuleList([ops[o]() for o in layout])
        self.first_e = layout.find("e")
        self.pred_norm = Norm(d)
        self.pred = nn.Linear(d, d, bias=False)            # predicts the next token's embedding
        self.g_alpha = nn.Parameter(torch.tensor(4.993))     # softplus -> 5.0
        self.g_beta = nn.Parameter(torch.tensor(0.5))
        self.head = AdaptiveHead(d, vocab, n_head)
        nn.init.normal_(self.emb.weight, std=1.0)
        for m in self.ops:
            if isinstance(m, EpisodicMemory):
                nn.init.normal_(m.proj.weight, std=0.02); nn.init.zeros_(m.wo.weight)
            if isinstance(m, ProductKeyMemory):
                nn.init.normal_(m.q.weight, std=0.02)
        nn.init.normal_(self.pred.weight, std=0.02)
        nn.init.zeros_(self.head.head.weight)
        if self.head.tail is not None:
            nn.init.zeros_(self.head.tail.weight)

    def set_code_order(self, counts):
        self.head.set_code_order(counts)

    def macs_per_token(self, T=None):
        T = T or self.train_seq
        m = sum(o.macs(T) if isinstance(o, EpisodicMemory) else o.macs() for o in self.ops)
        return m + (self.d * self.d if self.first_e >= 0 else 0) + self.head.macs(self.d)

    def predict(self, x):
        return F.normalize(self.pred(self.pred_norm(x)).float(), dim=-1)

    def gate_from_surprise(self, s):
        if not self.gate:
            return torch.ones_like(s)
        return torch.sigmoid(F.softplus(self.g_alpha) * (s - self.g_beta))

    def hidden(self, idx):
        e = self.emb(idx)
        x = e
        g, stats = None, {}
        for j, op in enumerate(self.ops):
            if j == self.first_e:
                # predictive coding: how well did the stream at t-1 predict token t's embedding?
                z = self.predict(x)
                en = F.normalize(e.detach().float(), dim=-1)
                cos = (z[:, :-1] * en[:, 1:]).sum(-1)
                s = torch.cat([cos.new_full((cos.shape[0], 1), 2.0), 1 - cos.detach()], 1)
                g = self.gate_from_surprise(s)
                stats = {"pred_loss": (1 - cos).mean(), "write_frac": (g > 0.5).float().mean(), "g_mean": g.mean()}
            x = x + (op(x, g) if isinstance(op, EpisodicMemory) else op(x))
        return x, stats

    def token_nll(self, idx, targets):
        return self.head.nll(self.hidden(idx)[0], targets)

    def forward(self, idx, targets=None):
        h, stats = self.hidden(idx)
        if targets is None:
            return self.head.logprobs(h[:, -1]).unsqueeze(1), None, {}
        return None, self.head.nll(h, targets).mean(), stats

    # ---- O(1)-state recurrent generation
    def init_state(self, B, device):
        st = {"prev_z": None}
        for j, op in enumerate(self.ops):
            if isinstance(op, Wiring):
                st[j] = torch.zeros(B, op.K - 1, self.d, device=device)
            elif isinstance(op, EpisodicMemory):
                st[j] = torch.zeros(B, op.h, op.dh, op.dh, device=device)
        return st

    @torch.no_grad()
    def step(self, tok, st):  # tok: B -> (log-probs B,V ; state ; write gate B)
        e = self.emb(tok)
        x = e
        g = torch.ones(tok.shape[0], device=tok.device)
        for j, op in enumerate(self.ops):
            if j == self.first_e:
                en = F.normalize(e.float(), dim=-1)
                s = x.new_full((x.shape[0],), 2.0) if st["prev_z"] is None else 1 - (st["prev_z"] * en).sum(-1)
                g = self.gate_from_surprise(s)
                st["prev_z"] = self.predict(x)
            if isinstance(op, Wiring):
                y, st[j] = op.step(x, st[j])
            elif isinstance(op, EpisodicMemory):
                y, st[j] = op.step(x, g, st[j])
            else:
                y = op(x)
            x = x + y
        st["h"] = x
        return self.head.logprobs(x), st, g


def build_sac(name, vocab):
    """CFG='{"n_sub":512}' overrides preset fields."""
    return SAC(vocab, **{**PRESETS[name], **json.loads(os.environ.get("CFG") or "{}")})
