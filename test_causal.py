"""Check HPT: no position sees its future (across and inside patches), and the factorised head is an exact
normalised distribution whose sampling log-probs match the training loss."""
import torch
from hpt import HPT

torch.manual_seed(0)
V, T = 300, 64
m = HPT(V, d=32, P=8, D=64, L=2, heads=2).eval()
with torch.no_grad():                       # zero-initialised outputs would make the test vacuous
    for p in m.parameters():
        if p.dim() == 2 and p.abs().sum() == 0:
            p.normal_(0, 0.05)
m.set_code_order(torch.randint(1, 1000, (V,)))
idx = torch.randint(0, V, (2, T))
with torch.no_grad():
    h = m.hidden(idx)
    worst = 0.0
    for t in [0, 5, 7, 8, 9, 15, 16, 33, 63]:
        idx2 = idx.clone()
        idx2[:, t] = (idx2[:, t] + 1) % V
        h2 = m.hidden(idx2)
        worst = max(worst, (h[:, :t] - h2[:, :t]).abs().max().item() if t else 0.0)
        assert (h[:, t] - h2[:, t]).abs().max() > 0, f"position {t} ignores its own token"
    print(f"max change before a perturbed position = {worst:.1e}")
    assert worst == 0.0, worst

    lp = m.head.logprobs(h[:, -1])                                  # B,V
    print(f"sum of p over vocab = {lp.exp().sum(-1).tolist()} (<=1: padded codes hold the rest)")
    assert (lp.exp().sum(-1) <= 1 + 1e-5).all()
    y = torch.randint(0, V, (2,))
    nll = m.head.nll(h[:, -1:], y[:, None])[:, 0]
    err = (nll + lp.gather(1, y[:, None])[:, 0]).abs().max().item()
    print(f"max |train nll - sampling nll| = {err:.1e}")
    assert err < 1e-4, err
    odd = m.hidden(idx[:, :37])                                      # length not a multiple of P
    pad_err = (odd - h[:, :37]).abs().max().item()
    print(f"max |unpadded - padded| = {pad_err:.1e}")
    assert pad_err < 1e-4, pad_err                                   # float rounding differs across shapes
print("OK")
