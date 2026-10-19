"""Check minGRU's parallel log-space scan against the sequential recurrence, and the fla layers if installed."""
import torch
from rnn import MinGRU, FLA, RNNLM

torch.manual_seed(0)
m = MinGRU(16)
x = torch.randn(2, 70, 16)
with torch.no_grad():
    par = m(x)
    h, buf, seq = torch.zeros(2, 32), torch.zeros(2, 3, 16), []
    for t in range(70):
        y, h, buf = m.step(x[:, t], h, buf)
        seq.append(y)
err = (par - torch.stack(seq, 1)).abs().max().item()
print(f"minGRU max |parallel - sequential| = {err:.1e}")
assert err < 1e-4, err
try:
    import fla  # noqa: F401
    dev = "cuda"
    for kind in FLA:
        print(f"rnn_{kind}: building + first GPU call (kernel compile can take a minute)...", flush=True)
        try:
            lm = RNNLM(300, kind, d=64, n_layers=2).to(dev)
            idx = torch.randint(0, 300, (2, 128), device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, _ = lm(idx, idx)
            loss.backward()
            print(f"rnn_{kind}: ok ({sum(p.numel() for p in lm.parameters()) / 1e3:.0f}k params)")
        except Exception as e:
            print(f"rnn_{kind}: FAILED {type(e).__name__}: {str(e)[:200]}")
except ImportError:
    print("flash-linear-attention not installed: fla-based models skipped here")
print("OK")
