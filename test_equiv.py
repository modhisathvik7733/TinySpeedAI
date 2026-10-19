"""Check that SGHM's parallel training form equals its O(1)-state recurrent form."""
import torch
from models import SGHM

torch.manual_seed(0)
m = SGHM(vocab=300).eval()
with torch.no_grad():
    for mem in m.mem.values():
        mem.wo.weight.normal_(0, 0.05)  # make memory output non-trivial
idx = torch.randint(0, 300, (2, 40))
with torch.no_grad():
    par, _, _ = m(idx)
    st = m.init_state(2, "cpu")
    rec = []
    for t in range(idx.shape[1]):
        lg, st, _ = m.step(idx[:, t], st)
        rec.append(lg)
    rec = torch.stack(rec, 1)
err = (par - rec).abs().max().item()
print(f"max |parallel - recurrent| = {err:.2e}")
assert err < 1e-3, err
state = sum(v.numel() for v in st["M"].values()) + sum(c.numel() for c in st["conv"]) + st["prev_lp"].numel()
print(f"recurrent state floats per sequence (constant in length): {state // 2:,}")
print("OK")
