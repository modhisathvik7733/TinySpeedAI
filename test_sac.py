"""Check SAC: chunk-parallel training form == O(1)-state recurrence, no position sees its future, and the
fused sparse memory read == a naive gather."""
import torch
import torch.nn.functional as F
from sac import SAC

torch.manual_seed(0)
V, T = 300, 150                                   # T not a multiple of the chunk size: exercises padding
m = SAC(V, d=32, layout="cpcecpce", heads=2, dh=8, n_sub=16, dq=16, k=4, C=16, n_head=64).eval()
with torch.no_grad():                             # zero-initialised outputs would make the tests vacuous
    for n, p in m.named_parameters():
        if p.abs().sum() == 0:
            p.normal_(0, 0.3)
m.set_code_order(torch.randint(1, 1000, (V,)))
idx = torch.randint(0, V, (2, T))
with torch.no_grad():
    h, _ = m.hidden(idx)
    st = m.init_state(2, "cpu")
    rec = []
    for t in range(T):
        lp, st, _ = m.step(idx[:, t], st)
        rec.append(st["h"])
    err = (h - torch.stack(rec, 1)).abs().max().item()
    print(f"max |parallel - recurrent| = {err:.1e}")
    assert err < 1e-3, err
    print(f"max |step log-probs - parallel log-probs| = {(lp - m.head.logprobs(h[:, -1])).abs().max().item():.1e}")

    worst = 0.0
    for t in [1, 15, 16, 17, 40, 100, 149]:
        idx2 = idx.clone(); idx2[:, t] = (idx2[:, t] + 1) % V
        h2, _ = m.hidden(idx2)
        worst = max(worst, (h[:, :t] - h2[:, :t]).abs().max().item())
        assert (h[:, t] - h2[:, t]).abs().max() > 0
    print(f"max change before a perturbed position = {worst:.1e}")
    assert worst < 1e-5, worst

    pkm = m.ops[1]
    x = torch.randn(10, 32)
    ids, w = pkm.lookup(x)
    naive = (pkm.pkm_values[ids] * w.unsqueeze(-1)).sum(1)
    assert (pkm(x) - naive).abs().max() < 1e-5
print(f"recurrent state floats per sequence (constant in length): "
      f"{sum(v.numel() for k, v in st.items() if k not in ('h', 'prev_z')) // 2 + 32:,}")
print("OK")
