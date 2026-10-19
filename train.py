"""Train a model on tokenized data on CUDA / MPS / CPU, then evaluate.

usage: python train.py <model> [steps]
       torchrun --nproc_per_node=<gpus> train.py <model> [steps]      (data parallel, multi-GPU)
env:   BATCH, SEQ, LR    override the model's defaults (LR scales with sqrt(BATCH) when not given)
       AMP=0             disable bf16 autocast on CUDA
       COMPILE=0         disable torch.compile;  COMPILE_MODE=max-autotune-no-cudagraphs for tuned kernels
       DATA=<dir>        token data directory (default data/)
       GPU_DATA_GB=4     datasets up to this size (as int32) live on the GPU; bigger ones stream from disk
       EVAL_EVERY=250    validation interval in steps (0 = only at the end)
       CKPT_EVERY=<n>    save runs/<model>.ckpt every n steps;  RESUME=1 continues from it
       QUICK=1           skip the final evaluation and samples (for throughput benchmarks)
       OPT=muon          Muon for hidden weight matrices + AdamW for the rest (MUON_LR, default 0.03)
       TIME_BUDGET=<s>   train for this many seconds (LR schedule follows time); [steps] becomes a cap
       RUN_NAME=<name>   results file name (default: the model name)
       SEED=<n>          random seed (default 1337);  CFG='{"L":5}' overrides FGPT/SAC preset fields
       PKM_LR_MULT=1     LR multiplier for sparse memory tables (no weight decay on them)
"""
import json, math, os, queue, sys, threading, time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tokenizers import Tokenizer
from models import build
from optim import Muon, Combined, is_muon_param

NAME = sys.argv[1]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
WARM = 100
DDP = int(os.environ.get("WORLD_SIZE", 1)) > 1
RANK = int(os.environ.get("RANK", 0))
WORLD = int(os.environ.get("WORLD_SIZE", 1))
MASTER = RANK == 0
if DDP:
    dist.init_process_group("nccl")
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(LOCAL_RANK)
    DEV = f"cuda:{LOCAL_RANK}"
else:
    DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CUDA = DEV.startswith("cuda")
AMP = CUDA and os.environ.get("AMP", "1") == "1"
COMPILE = CUDA and os.environ.get("COMPILE", "1") == "1"
QUICK = os.environ.get("QUICK", "0") == "1"
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 250))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", 0))
OPT = os.environ.get("OPT", "adamw")
BUDGET = float(os.environ.get("TIME_BUDGET", 0))
RUN = os.environ.get("RUN_NAME", NAME)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA", os.path.join(HERE, "data"))
OUT = os.path.join(HERE, "runs")
os.makedirs(OUT, exist_ok=True)
SEED = int(os.environ.get("SEED", 1337))
torch.manual_seed(SEED)

train = np.memmap(os.path.join(DATA, "train.bin"), dtype=np.uint16, mode="r")
val = np.memmap(os.path.join(DATA, "val.bin"), dtype=np.uint16, mode="r")
tok = Tokenizer.from_file(os.path.join(DATA, "tok.json"))
V = tok.get_vocab_size()
# token frequencies from a prefix (bounded cost on huge datasets). Frequent tokens = function words;
# "content" tokens are the rest (names, objects) -> binding probe. HPT also uses them for its output codes.
COUNTS = np.bincount(train[:50_000_000], minlength=V)
FREQ_IDS = torch.from_numpy(np.argsort(-COUNTS)[:300].astype(np.int64))


def log(*a):
    if MASTER:
        print(*a, flush=True)


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP)


def sync():
    if CUDA:
        torch.cuda.synchronize()
    elif DEV == "mps":
        torch.mps.synchronize()


class Batches:
    """Random training windows. A dataset that fits in GPU_DATA_GB is kept on the device and sliced there
    (no CPU work per step). A bigger one is memory-mapped and gathered by a background thread into pinned
    memory, so dataset size is bounded by disk, not by RAM or VRAM."""

    def __init__(self, data, bs, tl, seed):
        self.data, self.bs, self.tl = data, bs, tl
        self.n = len(data) - tl - 1
        on_device = DEV != "cpu" and data.size * 4 <= float(os.environ.get("GPU_DATA_GB", 4)) * 1e9
        if on_device:
            self.dev_data = torch.from_numpy(np.asarray(data).astype(np.int32)).to(DEV)
            self.gen = torch.Generator(device=DEV).manual_seed(seed)
            self.ar = torch.arange(tl + 1, device=DEV)
        else:
            self.dev_data = None
            self.q = queue.Queue(maxsize=8)
            threading.Thread(target=self._worker, args=(seed,), daemon=True).start()
        self.mode = "device-resident" if on_device else "streamed from disk"

    def _worker(self, seed):
        rng = np.random.default_rng(seed)
        ar = np.arange(self.tl + 1)
        while True:
            ix = rng.integers(0, self.n, self.bs)
            t = torch.from_numpy(self.data[ix[:, None] + ar].astype(np.int32))
            self.q.put(t.pin_memory() if CUDA else t)

    def next(self):
        if self.dev_data is not None:
            ix = torch.randint(0, self.n, (self.bs, 1), device=DEV, generator=self.gen)
            chunk = self.dev_data[ix + self.ar].long()
        else:
            chunk = self.q.get().to(DEV, non_blocking=True).long()
        return chunk[:, :-1], chunk[:, 1:]


def batch(data, bs, tl, rng):
    ix = rng.integers(0, len(data) - tl - 1, bs)
    x = torch.from_numpy(np.stack([data[i:i + tl] for i in ix]).astype(np.int64))
    y = torch.from_numpy(np.stack([data[i + 1:i + tl + 1] for i in ix]).astype(np.int64))
    return x.to(DEV), y.to(DEV)


def token_loss(model, x, y):
    if hasattr(model, "token_nll"):
        return model.token_nll(x, y).float()
    logits, _, _ = model(x)
    return F.cross_entropy(logits.float().reshape(-1, V), y.reshape(-1), reduction="none").view(y.shape)


@torch.no_grad()
def evaluate(model, tl, n_batches, bs=16):
    """val loss, plus loss split by whether the target is a *repeated content token* (needs binding)."""
    model.eval()
    rng = np.random.default_rng(0)
    tot = rep = nov = 0.0
    n_tot = n_rep = n_nov = 0
    late, n_late = 0.0, 0
    freq = FREQ_IDS.to(DEV)
    causal = torch.ones(tl, tl, device=DEV, dtype=torch.bool).tril()
    for _ in range(n_batches):
        x, y = batch(val, bs, tl, rng)
        with autocast():
            ce = token_loss(model, x, y)
        content = ~torch.isin(y, freq)
        # repeated: target token already appeared in the context x[:, :t+1]
        seen = (x.unsqueeze(1) == y.unsqueeze(2))                     # B,T(target),T(ctx)
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
def sample(model, prompt, T, n=160, temp=0.8, topk=40):
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
            with autocast():
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
    if hasattr(model, "set_code_order"):
        model.set_code_order(COUNTS)
    B = int(os.environ.get("BATCH", model.train_batch))
    T = int(os.environ.get("SEQ", model.train_seq))
    LR = float(os.environ.get("LR", model.base_lr * (B * WORLD / model.train_batch) ** 0.5))
    nparams = sum(p.numel() for p in model.parameters())
    macs = model.macs_per_token(T) if hasattr(model, "macs_per_token") else nparams
    flop_tok = 6 * macs                                     # training FLOPs per token (fwd + bwd)
    named = list(model.named_parameters())
    muon = [p for n, p in named if OPT == "muon" and is_muon_param(n, p)]
    rest = [(n, p) for n, p in named if not any(p is q for q in muon)]
    sparse = [p for n, p in rest if "pkm" in n]   # memory slots: decaying untouched slots would erase them
    decay = [p for n, p in rest if p.dim() >= 2 and "pkm" not in n]
    other = [p for n, p in rest if p.dim() < 2 and "pkm" not in n]
    groups = [{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}]
    if sparse:
        groups.append({"params": sparse, "weight_decay": 0.0, "lr": LR * float(os.environ.get("PKM_LR_MULT", 1))})
    opt = torch.optim.AdamW(groups, lr=LR, betas=(0.9, 0.95), **({"fused": True} if CUDA else {}))
    if muon:
        opt = Combined(opt, Muon(muon, lr=float(os.environ.get("MUON_LR", 0.03))))
    for gr in opt.param_groups:
        gr["base_lr"] = gr["lr"]
    loader = Batches(train, B, T, seed=SEED + RANK)
    tok_step = B * T * WORLD
    ckpt_path = os.path.join(OUT, f"{RUN}.ckpt")
    step = 0
    if os.environ.get("RESUME") == "1" and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEV)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
        log(f"resumed from {ckpt_path} at step {step}")
    hist = []
    log(f"{NAME}: {nparams / 1e6:.2f}M params, {flop_tok / 1e6:.2f} MFLOP/token, on {DEV}{' bf16' if AMP else ''}"
        f" x{WORLD}, batch {B}x{T} ({tok_step / 1e3:.0f}k tok/step), lr {LR:.2e}, opt {OPT}"
        f"{f' ({sum(p.numel() for p in muon) / 1e6:.2f}M params on Muon)' if muon else ''},"
        f" {f'{BUDGET:.0f}s budget' if BUDGET else f'{STEPS} steps'}, data {len(train) / 1e6:.0f}M tokens {loader.mode}")

    fwd = model
    if COMPILE:
        log("compiling (first step; can take 1-2 min, kernels are being built)...")
        t_c = time.time()
        mode = os.environ.get("COMPILE_MODE") or None
        try:
            fwd = torch.compile(model, mode=mode)
            if DDP:
                fwd = torch.nn.parallel.DistributedDataParallel(fwd, device_ids=[LOCAL_RANK])
            x, y = loader.next()
            with autocast():
                _, loss, st = fwd(x, y)
            (loss + sum(st.values())).backward()
            opt.zero_grad(set_to_none=True)
            sync()
            log(f"compiled in {time.time() - t_c:.0f}s")
        except Exception as e:  # e.g. no Triton support for this GPU: fall back to eager
            log(f"torch.compile failed ({type(e).__name__}: {str(e)[:200]}), using eager")
            fwd = model
            opt.zero_grad(set_to_none=True)
    if DDP and fwd is model:
        fwd = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK])

    t0 = time.time()
    eval_time = 0.0
    t100 = None
    start = step
    while step < STEPS:
        if BUDGET:
            frac = (time.time() - t0 - eval_time) / BUDGET
            if frac >= 1:
                break
        else:
            frac = step / STEPS
        f = min(1, (step - start + 1) / WARM) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for gr in opt.param_groups:
            gr["lr"] = gr["base_lr"] * f
        x, y = loader.next()
        with autocast():
            _, loss, st = fwd(x, y)
        total = loss
        if "pred_loss" in st:
            total = total + 0.1 * st["pred_loss"] + 0.01 * st["g_mean"]
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if CKPT_EVERY and step % CKPT_EVERY == 0 and MASTER:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, ckpt_path)
        if step % 100 == 0:
            sync()
            el = time.time() - t0 - eval_time
            if step - start == 100:
                t100 = el
            row = {"step": step, "time": el, "tokens": step * tok_step, "loss": loss.item(),
                   **{k: v.item() for k, v in st.items()}}
            if MASTER and EVAL_EVERY and step % EVAL_EVERY == 0 and not QUICK:
                te = time.time()
                row["val"] = evaluate(model, T, 20)["loss"]
                eval_time += time.time() - te
            hist.append(row)
            extra = " ".join(f"{k}={v:.3f}" for k, v in row.items() if k not in ("step", "time", "tokens"))
            rate = (step - start) * tok_step / el
            log(f"step {step:5d} {el:6.1f}s {rate / 1e6:6.2f}M tok/s {rate * flop_tok / 1e12:5.1f} TFLOPS {extra}")
    sync()
    train_time = time.time() - t0 - eval_time
    tokens = (step - start) * tok_step
    steady = (tokens - 100 * tok_step) / (train_time - t100) if t100 and step - start > 100 else tokens / train_time
    log(f"steady-state throughput: {steady / 1e6:.2f}M tok/s ({steady * flop_tok / 1e12:.1f} TFLOPS achieved)")
    if not MASTER:
        dist.destroy_process_group()
        return
    res = {"model": NAME, "run": RUN, "opt": OPT, "cfg": os.environ.get("CFG", ""), "seed": SEED, "device": DEV, "world": WORLD, "amp": AMP, "params": nparams, "flop_per_token": flop_tok,
           "steps": step, "tokens": step * tok_step, "train_time_s": train_time, "tok_per_s": tokens / train_time,
           "steady_tok_per_s": steady, "batch": B, "seq": T, "lr": LR, "log": hist}
    if not QUICK:
        log("final evaluation at T=256...")
        res["eval_T256"] = evaluate(model, 256, 60)
        log(f"  val@T256 = {res['eval_T256']['loss']:.3f}; evaluation at T=1024...")
        res["eval_T1024"] = evaluate(model, 1024, 15)
        log(f"  val@T1024 = {res['eval_T1024']['loss']:.3f}; generating samples...")
        story, gates, ids = sample(model, "Once upon a time", T)
        story2, _, _ = sample(model, "Lily and Max went to the park. Lily found a", T)
        res["sample"], res["sample2"] = story, story2
        if gates:
            pieces = [tok.decode([i]) for i in ids[:len(gates)]]
            res["gate_trace"] = list(zip(pieces, [round(g, 2) for g in gates]))[:80]
            res["gen_write_frac"] = float(np.mean(np.array(gates) > 0.5))
        with open(os.path.join(OUT, f"{RUN}.json"), "w") as f:
            json.dump(res, f, indent=1)
        torch.save(model.state_dict(), os.path.join(OUT, f"{RUN}.pt"))
    log(f"done: {NAME} ({RUN})")
    if QUICK:
        print(json.dumps({k: v for k, v in res.items() if k not in ("log", "gate_trace")}, indent=1))
    if DDP:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
