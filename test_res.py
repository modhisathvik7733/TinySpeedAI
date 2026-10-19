"""Check RCM's parallel pieces against brute force: chunked leaky traces == the sequential recursion, and the
sort-based copy rule == a direct search for the previous occurrence."""
import torch
from res import traces, follower_of_previous, spectral_embedding, RCM
import numpy as np

torch.manual_seed(0)
e = torch.randn(2, 150, 5)
lam = torch.tensor([0.5, 0.9, 0.999])
par = traces(e, lam, C=16)
h = torch.zeros(2, 3, 5)
seq = []
for t in range(150):
    h = lam.view(1, 3, 1) * h + (1 - lam).view(1, 3, 1) * e[:, t].unsqueeze(1)
    seq.append(h.reshape(2, -1))
err = (par - torch.stack(seq, 1)).abs().max().item()
print(f"max |chunked traces - recursion| = {err:.1e}")
assert err < 1e-5, err

x = torch.randint(0, 6, (3, 80))
f, found = follower_of_previous(x, x)
for b in range(3):
    for t in range(80):
        prev = [s for s in range(t) if x[b, s] == x[b, t]]
        assert found[b, t] == bool(prev)
        if prev:
            assert f[b, t] == x[b, prev[-1] + 1]
print("copy rule matches brute force")

V = 50
toks = np.random.default_rng(0).integers(0, V, 5000).astype(np.uint16)
E = spectral_embedding(toks, V, 8, "cpu")
m = RCM(E, D=64)
phi, cp = m.features(torch.randint(0, V, (2, 40)))
assert phi.shape == (80, m.Dt) and torch.isfinite(phi).all()
print("OK")
