# Design: intelligence from organisation, not brute force

## 0. Challenge the premise first

The request has four constraints. Here is what each one actually costs.

| Constraint | What the transformer does instead | Is it really the bottleneck? |
|---|---|---|
| No growing KV memory | Keeps every past token and searches all of them on every step | **Yes.** It is the only part of inference that grows without bound. It exists to solve one job: *binding* (who is "Ben", what is "the vase") |
| Not everything fires every step | Every weight multiplies every token | **Only partly.** On a GPU, sparse matmuls do not beat dense ones at 5M params. The real benefit of sparsity is **representational**: sparse codes barely overlap, so a fixed memory can store far more patterns. Do not expect wall-clock gains from sparsity on an M1. |
| Genuinely neural | – | Learned weights, distributed codes, plasticity. No n-gram tables, no lookups. |
| Cheap training | Big data × big compute | At TinyStories scale a small transformer already trains in minutes. So the honest test is **quality per token and per second against a matched transformer**, not "fast vs. millions of dollars". |

Two conclusions follow:
1. The KV cache exists to do **content-addressable binding** across the whole context. Replacing it means finding another way to bind, not just a smaller cache.
2. Most tokens carry almost no information. "Once upon a time there was a" is nearly deterministic, yet a transformer caches it with the same care as the surprising token "otter".

## 1. What the brain gets for free that a transformer pays for

| Brain (free) | Transformer (expensive) |
|---|---|
| **Memory is the synapses changing.** The hippocampus binds in one shot by changing synaptic weights. Storage size is fixed and nothing is replayed. | Memory is a tape (the KV cache) that gets re-read on every step |
| **Novelty gates plasticity.** Dopamine and acetylcholine bursts mean only surprising events get written. | Every token is written to the cache |
| **Sparse codes** (~2% of neurons active) are nearly orthogonal, so a fixed synaptic matrix can hold many associations with little crosstalk | Dense keys of size d. A linear-attention memory saturates after about d patterns |
| **Two learning systems.** Slow cortex learns statistics; fast hippocampus stores episodes | One mechanism (attention plus weights) does both jobs |
| **Errors propagate, predictions are free.** Predictable input causes little activity | Full compute on every token |

## 2. Three radically different mechanisms

### A. Surprise-Gated Hebbian Memory (SGHM): a hippocampus made of fast synapses
- A **slow cortical stream** (causal local mixing plus sparse k-WTA MLP) learns local statistics through ordinary gradient training.
- A cheap **local predictor** in that stream gives a per-token **surprise in nats**, `s_t = -log p_local(x_t | local context)`.
- A **fast synaptic matrix** `M` (fixed size, per head `N×d_v`) is written by a Hebbian outer product `M ← λM + g_t · v_t a_tᵀ`, where `a_t` is a **k-WTA sparse code** (8 of 256 units active) and the write strength `g_t = σ(α(s_t − β))` is driven by surprise.
- Reading is pattern completion with a sparse query code, `r_t = M b_t`.
- What it buys: constant memory; capacity spent only on information the cortex could not predict; sparse keys give roughly (N/k)² capacity instead of about d.
- Why it trains fast: the gate depends only on the *local* predictor, which can be computed in parallel. Training therefore uses an exactly equivalent masked parallel form, and inference uses the O(1) recurrent form.

### B. Event-driven predictive-coding hierarchy
- Columns sit at several timescales. Upward connections carry **only prediction errors**; downward connections carry predictions.
- A higher column updates only when the error from below crosses a threshold, so compute is proportional to surprise.
- Memory is the column state.
- Problems: thresholded updates are sequential and hard to train in parallel. A single state vector also compresses away exactly what binding needs (arbitrary names and objects).

### C. Dendritic context-by-identity (HTM-like)
- Context is coded by **which** cell inside each active minicolumn fires, not by activation values. Dendritic segments detect patterns of previously active cells.
- Combinatorial capacity is huge.
- Problems: it is discrete and non-differentiable, learns slowly, and has a long record of not scaling to real language.

### Selection: A
- It is the only one of the three that attacks binding directly, which is the job the KV cache really does.
- Its training has an exactly parallel form, so it trains on an M1 at transformer-like speed.
- Every claim it makes can be falsified with an ablation:
  - **No gate** (write every token): does surprise gating matter?
  - **Dense keys** (no k-WTA): does sparsity buy memory capacity?
  - **Matched transformer**: is it competitive at all?

### What is not new (stated honestly)
The read/write algebra `r = Σ g λ^{Δ} (b·a) v` belongs to the linear-attention / fast-weight family. The claimed new capability comes from the two organisational principles together:
- capacity allocated by information content (surprise from a *separate* slow system, not a learned input gate), and
- near-orthogonal sparse keys that let a fixed-size matrix act like a large episodic store.

The ablations test whether those principles actually do anything.

## 3. Information flow (not layers)

```
token ─▶ embed ─▶ [cortex: causal local mix + sparse k-WTA MLP] ─┬─▶ local predictor ─▶ surprise s_t (nats)
                                                                 │                              │
                                                                 ▼                              ▼ gate g_t
                      ┌──────────── fast synapses M (fixed size) ◀── Hebbian write g_t·v_t a_tᵀ (sparse a_t)
                      │  read: r_t = M b_t (sparse query)
                      ▼
          cortex ◀── recalled episode (gated by the cortex) ─▶ more cortex ─▶ second recall ─▶ next-token
```
