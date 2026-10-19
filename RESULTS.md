# Results: SGHM vs. a transformer on TinyStories (M1 Max, MPS)

The design and its reasoning are in [DESIGN.md](DESIGN.md).

**Setup:**
- Data: 40 MB TinyStories slice, about 9.9M tokens, BPE vocabulary of 4096.
- Training: batch 32 × 256 tokens, 1000 steps (8.2M tokens) for every model, AdamW at lr 2e-3 with cosine decay.
- One seed per model. Each run takes about 2.5 to 4.5 minutes.

Loss is in nats per token on the validation split; lower is better.

| model | params | train time | tok/s | val (T=256) | val (T=1024) | loss past position 256 | repeated content tokens | first-time content tokens | memory writes |
|---|---|---|---|---|---|---|---|---|---|
| GPT (RoPE, 5 layers) | 4.98M | **147 s** | 55k | 2.264 | 2.712 | 2.859 | **3.96** | 4.74 | KV cache grows |
| **SGHM, dense keys** | 5.38M | 259 s | 32k | **2.197** | **2.189** | **2.183** | 4.05 | **4.53** | 65% of tokens, fixed size |
| SGHM, sparse keys (as designed) | 5.38M | 258 s | 32k | 2.343 | 2.358 | 2.362 | 4.52 | 4.67 | 53% of tokens |
| SGHM, sparse keys, no gate | 5.38M | 252 s | 32k | 2.297 | 2.305 | 2.305 | 4.31 | 4.64 | 100% of tokens |
| SGHM, no memory (cortex only) | 3.94M | 134 s | 61k | 2.508 | 2.523 | 2.526 | 5.65 | 4.77 | – |

*Repeated content tokens* are names and objects that already appeared earlier in the context. They measure **binding**, the job the KV cache exists to do.

## What held up
1. **The fast-synapse memory is the mechanism.**
   - Removing it costs 0.31 nats against the best variant.
   - Loss on repeated names and objects jumps from 4.05 to 5.65.
   - The cortex alone cannot bind, and the Hebbian memory does.
2. **No growing KV memory.**
   - Inference state is a fixed ~134k floats, and the recurrent form matches the parallel form to 1e-6 (`test_equiv.py`).
   - Trained at 256 tokens, SGHM is *as good or better* at 1024 tokens (2.189).
   - GPT degrades badly past its training length (2.859 beyond position 256).
3. **Per token, SGHM with dense keys beats the matched transformer.**
   - Its validation loss at T=256 is 2.197 against 2.264.
   - On binding it is almost at parity with attention (4.05 vs. 3.96), using fixed memory.
4. **Surprise gating works as a signal.** Predictable tokens ("upon a time") get a gate of about 0.23; surprising ones ("Once", names, adjectives) get 0.6 to 0.99.

## What did not hold up (falsified at this scale)
- **Sparse k-WTA keys hurt.** They scored 2.343 against 2.197 for dense keys. My capacity argument assumed binary sparse codes with many patterns; with 8 of 256 active and learned projections, gradients only reach 3% of key units, so the keys learn slowly. The brain's "free" sparsity does not transfer to one-pass gradient learning at 1000 steps.
- **Surprise gating costs a little quality.** It scored 2.343 against 2.297 with no gate, in exchange for writing about half as often. It saves write bandwidth but does not yet improve what is stored.
- **Training wall-clock is 1.75× slower than GPT on the M1.** The dense-key variant has the same speed, so the cause is not `topk` (I suspected that earlier and was wrong). It is the memory module's parallel form plus its projections, which run as unfused MPS kernels, while attention uses the fused `scaled_dot_product_attention`. A chunked or fused kernel is the obvious fix. The arithmetic itself is no larger than attention's.

## Caveats
- One seed and a short run (1000 steps). Differences of about 0.05 nats are within plausible noise; the 0.07 gap to GPT and the 0.3+ effects are not.
- SGHM has about 8% more parameters than GPT.
- "Not everything fires" is only partly met: the sparse MLP is k-WTA (64 of 1024 units), and memory writes are gated.

## Reproduce
```
python prepare.py 40          # tokenizer and data, ~10 s
python test_equiv.py          # parallel form == recurrent form
python train.py gpt 1000
python train.py sghm_dense 1000
python compare.py
```
