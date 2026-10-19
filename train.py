"""Train a model on TinyStories for a fixed number of steps on CUDA / MPS / CPU, then evaluate.

usage: python train.py <model> [steps]
env:   AMP=0 disables bf16 autocast on CUDA
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from models import build

NAME = sys.argv[1]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
B, T, LR, WARM = 32, 256, 2e-3, 100
DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
AMP = DEV == "cuda" and os.environ.get("AMP", "1") == "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(1337)

train = np.fromfile(os.path.join(HERE, "data/train.bin"), dtype=np.uint16)
val = np.fromfile(os.path.join(HERE, "data/val.bin"), dtype=np.uint16)
tok = Tokenizer.from_file(os.path.join(HERE, "data/tok.json"))
V = tok.get_vocab_size()
# frequent tokens = function words; "content" tokens are the rest (names, objects) -> binding probe
FREQ_IDS = torch.from_numpy(np.argsort(-np.bincount(train, minlength=V))[:300].astype(np.int64))


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP)


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()
    elif DEV == "mps":
        torch.mps.synchronize()


def batch(data, bs, tl, rng):
    ix = rng.integers(0, len(data) - tl - 1, bs)
    x = torch.from_numpy(np.stack([data[i:i + tl] for i in ix]).astype(np.int64))
    y = torch.from_numpy(np.stack([data[i + 1:i + tl + 1] for i in ix]).astype(np.int64))
    return x.to(DEV), y.to(DEV)


@torch.no_grad()
def evaluate(model, tl, n_batches, bs=16):
    """val loss, plus loss split by whether the target is a *repeated content token* (needs binding)."""
    model.eval()
    rng = np.random.default_rng(0)
    tot = rep = nov = 0.0
    n_tot = n_rep = n_nov = 0
    late, n_late = 0.0, 0
    freq = FREQ_IDS.to(DEV)
    for _ in range(n_batches):
        x, y = batch(val, bs, tl, rng)
        with autocast():
            logits, _, _ = model(x)
        ce = F.cross_entropy(logits.float().reshape(-1, V), y.reshape(-1), reduction="none").view(bs, tl)
        content = ~torch.isin(y, freq)
        # repeated: target token already appeared in the context x[:, :t+1]
        seen = (x.unsqueeze(1) == y.unsqueeze(2))                     # B,T(target),T(ctx)
        causal = torch.ones(tl, tl, device=DEV, dtype=torch.bool).tril()
        repeated = (seen & causal).any(-1) & content
        novel = content & ~repeated
        tot += ce.sum().item(); n_tot += ce.numel()
        rep += ce[repeated].sum().item(); n_rep += repeated.sum().item()
        nov += ce[novel].sum().item(); n_nov += novel.sum().item()
        if tl > 256:
            late += ce[:, 256:].sum().item(); n_late += ce[:, 256:].numel()
    model.train()
    out = {"loss": tot / n_tot, "repeat_content_loss": rep / max(n_rep, 1), "novel_content_loss": nov / max(n_nov, 1)}
    if n_late:
        out["loss_beyond_256"] = late / n_late
    return out


@torch.no_grad()
def sample(model, prompt, n=160, temp=0.8, topk=40):
    model.eval()
    ids = tok.encode(prompt).ids
    g = torch.Generator().manual_seed(7)
    gates = []
    if hasattr(model, "step"):  # constant-state recurrent generation
        st = model.init_state(1, DEV)
        for t in ids[:-1]:
            _, st, gt = model.step(torch.tensor([t], device=DEV), st)
            gates.append(gt.item())
        cur = ids[-1]
        for _ in range(n):
            logits, st, gt = model.step(torch.tensor([cur], device=DEV), st)
            gates.append(gt.item())
            cur = pick(logits[0], temp, topk, g)
            ids.append(cur)
    else:
        for _ in range(n):
            ctx = torch.tensor([ids[-T:]], device=DEV)
            logits, _, _ = model(ctx)
            ids.append(pick(logits[0, -1], temp, topk, g))
    model.train()
    return tok.decode(ids), gates, ids


def pick(logits, temp, topk, g):
    logits = logits.float().cpu() / temp
    v, i = logits.topk(topk)
    return i[torch.multinomial(F.softmax(v, -1), 1, generator=g)].item()


def main():
    model = build(NAME, V).to(DEV)
    nparams = sum(p.numel() for p in model.parameters())
    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                            lr=LR, betas=(0.9, 0.95))
    rng = np.random.default_rng(1)
    log = []
    print(f"{NAME}: {nparams / 1e6:.2f}M params on {DEV}{' (bf16)' if AMP else ''}, {STEPS} steps")
    t0 = time.time()
    step = 0
    while step < STEPS:
        frac = step / STEPS
        lr = LR * min(1, (step + 1) / WARM) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for gr in opt.param_groups:
            gr["lr"] = lr
        x, y = batch(train, B, T, rng)
        with autocast():
            _, loss, st = model(x, y)
        total = loss
        if "local_loss" in st:
            total = total + 0.2 * st["local_loss"] + 0.01 * st["g_mean"]
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 100 == 0:
            sync()
            el = time.time() - t0
            row = {"step": step, "time": el, "tokens": step * B * T, "loss": loss.item(),
                   **{k: v.item() for k, v in st.items()}}
            if step % 250 == 0:
                row["val"] = evaluate(model, T, 20)["loss"]
            log.append(row)
            extra = " ".join(f"{k}={v:.3f}" for k, v in row.items() if k not in ("step", "time", "tokens"))
            print(f"step {step:5d} {el:6.0f}s {step * B * T / el / 1e3:6.1f}k tok/s {extra}", flush=True)
    sync()
    train_time = time.time() - t0
    tokens = step * B * T
    ev256 = evaluate(model, 256, 60)
    ev1024 = evaluate(model, 1024, 15)
    story, gates, ids = sample(model, "Once upon a time")
    story2, _, _ = sample(model, "Lily and Max went to the park. Lily found a")
    res = {"model": NAME, "device": DEV, "amp": AMP, "params": nparams, "steps": step, "tokens": tokens, "train_time_s": train_time,
           "tok_per_s": tokens / train_time, "eval_T256": ev256, "eval_T1024": ev1024,
           "sample": story, "sample2": story2, "log": log}
    if gates:
        pieces = [tok.decode([i]) for i in ids[:len(gates)]]
        res["gate_trace"] = list(zip(pieces, [round(g, 2) for g in gates]))[:80]
        res["gen_write_frac"] = float(np.mean(np.array(gates) > 0.5))
    with open(os.path.join(OUT, f"{NAME}.json"), "w") as f:
        json.dump(res, f, indent=1)
    torch.save(model.state_dict(), os.path.join(OUT, f"{NAME}.pt"))
    print(json.dumps({k: v for k, v in res.items() if k not in ("log", "gate_trace")}, indent=1))


if __name__ == "__main__":
    main()
