"""RCM (Reservoir Cortex Model): a language model trained in ONE streaming pass with NO backpropagation.

Reservoir computing trains fast because nothing but a linear readout is learned: the dynamics are fixed and
the readout is one least-squares solve from statistics summed over the data (additive, so data streams once
and shards across GPUs for free). Plain reservoirs fail on language for three reasons; each has a fixed,
training-free fix here:
  1. random inputs carry no meaning   -> token embeddings from the data in closed form (SVD of bigram PPMI)
  2. no binding of names / objects    -> a copy rule: the token that followed the last occurrence of the
                                          current token (and of the current bigram), found by a parallel sort
  3. least squares gives scores, not probabilities -> 4 scalars calibrated on held-out text afterwards
Context lives in fixed multi-timescale leaky traces (no KV cache), computed chunk-parallel.

  features z_t = [e_t, e_{t-1}, .., e_{t-W+1} | traces at 8 timescales | e(copy1), e(copy2), found flags]
  phi_t        = [relu(R z_t + b) (D random features) | z_t | 1]
  readout      W = (Phi^T Phi + lam I)^-1 Phi^T Y      (Phi^T Y accumulated by index_add; Phi^T Phi subsampled)
  p(y | t)     = softmax(a * log(relu(phi_t W) + eps) + c1 [y == copy1] + c2 [y == copy2])

usage: python res.py [run_name] [time_budget_s]        (default: sr_res 45)
env:   D=4096 (random features)  DE=64 (embedding dim)  WIN=4  G_EVERY=4 (Phi^T Phi on 1 batch in G_EVERY)
       DATA=<dir>  BATCH=32 SEQ=1024
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

DECAYS = (0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)


def traces(e, lam, C=64):
    """h_t = lam*h_{t-1} + (1-lam)*e_t from h_{-1}=0, for every lam at once. e: B,T,d -> B,T,len(lam)*d.
    Exact chunk-parallel form: a C x C decay matmul inside chunks, a T/C-step carry between chunks."""
    B, T0, d = e.shape
    pad = (-T0) % C
    e = F.pad(e, (0, 0, 0, pad))
    T = T0 + pad
    n, m = T // C, lam.numel()
    i = torch.arange(C, device=e.device, dtype=e.dtype)
    ex = (i.view(C, 1) - i.view(1, C))
    L = torch.where(ex >= 0, lam.view(m, 1, 1) ** ex.clamp(min=0), 0.0) * (1 - lam).view(m, 1, 1)   # m,C,C
    intra = torch.einsum("mij,bnjd->bnmid", L, e.view(B, n, C, d))                                   # B,n,m,C,d
    lamC = (lam ** C).view(1, m, 1)
    s = torch.zeros(B, m, d, device=e.device, dtype=e.dtype)
    carry = []
    for c in range(n):
        carry.append(s)
        s = lamC * s + intra[:, c, :, -1]
    carry = torch.stack(carry, 1)                                                                     # B,n,m,d
    out = intra + (lam.view(m, 1) ** (i + 1).view(1, C)).view(1, 1, m, C, 1) * carry.unsqueeze(3)
    return out.permute(0, 1, 3, 2, 4).reshape(B, T, m * d)[:, :T0]


def follower_of_previous(keys, x):
    """For each t: the token that followed the most recent s < t with keys[s] == keys[t] (parallel, via a sort).
    keys, x: B,T int64. Returns follower ids (B,T) and a found mask."""
    B, T = keys.shape
    pos = torch.arange(T, device=keys.device).expand(B, T)
    order = torch.argsort(keys * T + pos, dim=1)
    sk = keys.gather(1, order)
    same = torch.zeros_like(sk, dtype=torch.bool)
    same[:, 1:] = sk[:, 1:] == sk[:, :-1]
    prev_sorted = torch.where(same, torch.roll(order, 1, dims=1), -1)
    prev = torch.empty_like(order).scatter_(1, order, prev_sorted)
    found = prev >= 0
    return x.gather(1, (prev + 1).clamp(max=T - 1)), found


class RCM:
    def __init__(self, E, D=4096, win=4, decays=DECAYS, seed=0):
        self.E = E                                          # V,de fixed embeddings
        self.V, self.de = E.shape
        self.win, self.D = win, D
        self.lam = torch.tensor(decays, device=E.device)
        self.S = self.de * (win + len(decays) + 2) + 2
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.R = (torch.randn(self.S, D, generator=g) / math.sqrt(self.S)).to(E.device)
        self.b = (torch.randn(D, generator=g) * 0.5).to(E.device)
        self.mu = self.sd = None
        self.Dt = D + self.S + 1
        self.W = None
        self.calib = None

    def copies(self, x):
        V, T = self.V, x.shape[1]
        f1, m1 = follower_of_previous(x, x)
        big = torch.cat([torch.full_like(x[:, :1], -1), x[:, :-1]], 1) * V + x
        big[:, 0] = V * V + torch.arange(x.shape[0], device=x.device)       # position 0 has no bigram
        f2, m2 = follower_of_previous(big, x)
        return f1, m1, f2, m2

    def features(self, x):  # x: B,T -> phi: B*T,Dt (float32), copy info
        e = self.E[x]
        rec = [e] + [F.pad(e, (0, 0, j, 0))[:, :-j] for j in range(1, self.win)]
        f1, m1, f2, m2 = self.copies(x)
        z = torch.cat(rec + [traces(e, self.lam), self.E[f1] * m1.unsqueeze(-1), self.E[f2] * m2.unsqueeze(-1),
                             m1.unsqueeze(-1).float(), m2.unsqueeze(-1).float()], -1).reshape(-1, self.S)
        if self.mu is None:                                  # fixed standardisation from the first batch
            self.mu, self.sd = z.mean(0), z.std(0) + 1e-3
        z = (z - self.mu) / self.sd
        phi = torch.cat([torch.relu(z @ self.R + self.b), z, torch.ones_like(z[:, :1])], -1)
        return phi, (f1.reshape(-1), m1.reshape(-1), f2.reshape(-1), m2.reshape(-1))

    def logits(self, phi, cp, W=None, calib=None):
        a, eps, c1, c2 = calib if calib is not None else self.calib
        s = phi @ (self.W if W is None else W)
        lg = a * torch.log(torch.relu(s) + F.softplus(eps))
        f1, m1, f2, m2 = cp
        lg = lg.scatter_add(1, f1.unsqueeze(1), (c1 * m1.float()).unsqueeze(1))
        return lg.scatter_add(1, f2.unsqueeze(1), (c2 * m2.float()).unsqueeze(1))

    def token_nll(self, x, y):
        phi, cp = self.features(x)
        return F.cross_entropy(self.logits(phi, cp), y.reshape(-1), reduction="none").view(y.shape)

    def fit_calibration(self, phi, cp, y, W, steps=300):
        p = torch.tensor([1.0, -4.0, 1.0, 1.0], device=phi.device, requires_grad=True)
        opt = torch.optim.Adam([p], lr=0.05)
        s = (phi @ W).detach()
        f1, m1, f2, m2 = cp
        for _ in range(steps):
            lg = p[0] * torch.log(torch.relu(s) + F.softplus(p[1]))
            lg = lg.scatter_add(1, f1.unsqueeze(1), (p[2] * m1.float()).unsqueeze(1))
            lg = lg.scatter_add(1, f2.unsqueeze(1), (p[3] * m2.float()).unsqueeze(1))
            loss = F.cross_entropy(lg, y)
            opt.zero_grad(); loss.backward(); opt.step()
        return p.detach(), loss.item()


def spectral_embedding(tokens, V, de, dev):
    """Token embeddings in closed form: truncated SVD of [next-token PPMI | previous-token PPMI]."""
    t = torch.from_numpy(tokens.astype(np.int64)).to(dev)
    C = torch.bincount(t[:-1] * V + t[1:], minlength=V * V).view(V, V).float()
    ctx = C.sum(0) ** 0.75                                         # smoothed context distribution
    pmi = torch.log(C + 1e-9) - torch.log(C.sum(1, keepdim=True) + 1e-9) - torch.log(ctx / ctx.sum() + 1e-12).view(1, V)
    ppmi = torch.relu(pmi) * (C > 0)
    M = torch.cat([ppmi, ppmi.t()], 1)
    U, S, _ = torch.svd_lowrank(M, q=de, niter=4)
    E = U * S.sqrt()
    return E / (E.pow(2).mean(1, keepdim=True).sqrt() + 1e-6)


def main():
    from tokenizers import Tokenizer
    HERE = os.path.dirname(os.path.abspath(__file__))
    DATA = os.environ.get("DATA", os.path.join(HERE, "data"))
    RUN = sys.argv[1] if len(sys.argv) > 1 else "sr_res"
    BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
    D, DE = int(os.environ.get("D", 4096)), int(os.environ.get("DE", 64))
    WIN, G_EVERY = int(os.environ.get("WIN", 4)), int(os.environ.get("G_EVERY", 4))
    B, T = int(os.environ.get("BATCH", 32)), int(os.environ.get("SEQ", 1024))
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)
    train = np.memmap(os.path.join(DATA, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(DATA, "val.bin"), dtype=np.uint16, mode="r")
    V = Tokenizer.from_file(os.path.join(DATA, "tok.json")).get_vocab_size()
    n_fit = int(len(train) * 0.98)                       # last 2% held out for calibration / ridge choice

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()

    t_start = time.time()
    E = spectral_embedding(np.asarray(train[:min(n_fit, 50_000_000)]), V, DE, dev)
    m = RCM(E, D, WIN)
    sync()
    t_emb = time.time() - t_start
    data = torch.from_numpy(np.asarray(train[:n_fit]).astype(np.int32)).to(dev) if dev != "cpu" else None
    gen = torch.Generator(device=dev).manual_seed(1)
    ar = torch.arange(T + 1, device=dev)
    G = torch.zeros(m.Dt, m.Dt, device=dev)
    RT = torch.zeros(V, m.Dt, device=dev)                 # (Phi^T Y)^T
    n_g = n_tok = 0
    print(f"{RUN}: RCM D={D} de={DE} win={WIN} features {m.Dt}, readout {m.Dt * V / 1e6:.1f}M params, "
          f"embeddings in {t_emb:.1f}s, {BUDGET:.0f}s budget, batch {B}x{T}", flush=True)
    t0 = time.time()
    step = 0
    log = []
    while True:
        ix = torch.randint(0, n_fit - T - 1, (B, 1), device=dev, generator=gen)
        chunk = data[ix + ar].long()
        x, y = chunk[:, :-1], chunk[:, 1:]
        phi, _ = m.features(x)
        RT.index_add_(0, y.reshape(-1), phi)
        if step % G_EVERY == 0:
            G.addmm_(phi.t(), phi)
            n_g += phi.shape[0]
        n_tok += phi.shape[0]
        step += 1
        if step % 50 == 0:
            sync()
            el = time.time() - t0
            log.append({"step": step, "time": el, "tokens": n_tok})
            print(f"step {step:5d} {el:6.1f}s {n_tok / el / 1e6:6.2f}M tok/s  {n_tok / 1e6:.0f}M tokens", flush=True)
            if el + t_emb >= BUDGET * 0.8:                 # leave time for the solve and calibration
                break
    sync()
    t_acc = time.time() - t0
    # ridge solve for a few regularisation strengths; pick by held-out loss after calibration
    cal = torch.from_numpy(np.asarray(train[n_fit:n_fit + 16 * (T + 1)]).astype(np.int64)).to(dev).view(16, T + 1)
    phi_c, cp_c = m.features(cal[:, :-1])
    y_c = cal[:, 1:].reshape(-1)
    Gm = G / n_g
    Rm = RT.t() / n_tok
    scale = Gm.diagonal().mean()
    best = None
    for lam in (1e-4, 1e-3, 1e-2):
        W = torch.linalg.solve(Gm + lam * scale * torch.eye(m.Dt, device=dev), Rm)
        calib, loss = m.fit_calibration(phi_c, cp_c, y_c, W)
        print(f"  ridge {lam:g}: held-out loss {loss:.3f}", flush=True)
        if best is None or loss < best[0]:
            best = (loss, W, calib, lam)
    _, m.W, m.calib, lam = best
    sync()
    train_time = time.time() - t_start
    print(f"trained in {train_time:.1f}s (embeddings {t_emb:.1f}s, accumulate {t_acc:.1f}s, solve+calibrate "
          f"{train_time - t_emb - t_acc:.1f}s); {n_tok / 1e6:.0f}M tokens at {n_tok / t_acc / 1e6:.2f}M tok/s", flush=True)

    freq = torch.from_numpy(np.argsort(-np.bincount(train[:50_000_000], minlength=V))[:300].astype(np.int64)).to(dev)

    @torch.no_grad()
    def evaluate(tl, nb, bs=16):
        rng = np.random.default_rng(0)
        tot = rep = nov = late = 0.0
        n_tot = n_rep = n_nov = n_late = 0
        causal = torch.ones(tl, tl, device=dev, dtype=torch.bool).tril()
        for _ in range(nb):
            ix = rng.integers(0, len(val) - tl - 1, bs)
            xy = torch.from_numpy(np.stack([val[i:i + tl + 1] for i in ix]).astype(np.int64)).to(dev)
            x, yy = xy[:, :-1], xy[:, 1:]
            ce = m.token_nll(x, yy)
            content = ~torch.isin(yy, freq)
            repeated = ((x.unsqueeze(1) == yy.unsqueeze(2)) & causal).any(-1) & content
            novel = content & ~repeated
            tot += ce.sum().item(); n_tot += ce.numel()
            rep += ce[repeated].sum().item(); n_rep += repeated.sum().item()
            nov += ce[novel].sum().item(); n_nov += novel.sum().item()
            if tl > 256:
                late += ce[:, 256:].sum().item(); n_late += ce[:, 256:].numel()
        out = {"loss": tot / n_tot, "repeat_content_loss": rep / max(n_rep, 1), "novel_content_loss": nov / max(n_nov, 1)}
        if n_late:
            out["loss_beyond_256"] = late / n_late
        return out

    ev256, ev1024 = evaluate(256, 60), evaluate(1024, 15)
    log.append({"step": step, "time": train_time, "tokens": n_tok, "val": ev256["loss"]})
    tps = n_tok / t_acc
    fpt = 2 * m.S * D + 2 * m.Dt * m.Dt / G_EVERY + m.Dt       # forward-only work per training token
    res = {"model": "rcm", "run": RUN, "params": m.Dt * V + m.S * D, "flop_per_token": fpt, "steps": step,
           "tokens": n_tok, "train_time_s": train_time, "tok_per_s": tps, "steady_tok_per_s": tps,
           "eval_T256": ev256, "eval_T1024": ev1024, "ridge": lam, "calib": m.calib.tolist(),
           "D": D, "de": DE, "win": WIN, "log": log}
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    with open(os.path.join(HERE, "runs", f"{RUN}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "log"}, indent=1))


if __name__ == "__main__":
    main()
