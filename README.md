# TinySpeedAI

Small language models on TinyStories, built for **training tokens per second**.

## HPT: Hierarchical Patch Transformer (`hpt.py`, the main model)

A dense model costs about 6 × params FLOPs per training token. The GPT and SGHM baselines (about 5M params, 30 MFLOP/token) already run at about 45% of an RTX 5060 Ti's bf16 peak, at 0.7–0.8M tok/s. More tok/s therefore needs **fewer FLOPs per token**, not faster kernels. HPT decouples parameters from per-token compute in three ways:

1. **Patch backbone.** Every P=16 tokens are folded into one patch vector. A standard transformer (RoPE, QK-norm, flash attention, relu² MLP) runs over the patches only, so its cost per token is divided by P while its parameter count is not. Attention spans T/P positions, so long context is cheap.
2. **Thin token stream.** A narrow per-token stream (d=64) adds the patch context back to every token. Causal gated short convolutions resolve the detail inside each patch.
3. **Adaptive softmax.** Every position scores only the 1024 most frequent tokens (91.6% of the text) plus a "tail" slot. The other 3072 tokens are scored only where the target is one of them. The likelihood stays exact, and every token keeps its own output vector.

| preset | params | MFLOP/token | tok/s, 5060 Ti | use |
|---|---|---|---|---|
| `hpt_xxs` | 2.5M | 1.4 | ~10M (expected) | maximum speed |
| `hpt` | 4.3M | 2.2 | ~7M (expected) | default |
| `hpt_s` | 12.7M | 6.0 | ~3.5M (expected) | more quality |
| `hpt_m` | 88M | 68 | (bigger GPU) | scaling |
| `hpt_l` | 689M | 528 | (multi-GPU) | scaling |
| `gpt` (baseline) | 5.0M | 30 | 0.71M (measured) | |

The HPT tok/s figures are projected from the FLOP count and the TFLOPS measured with the first version of the head. Run `bench` below to get real numbers on your GPU. The log prints the achieved TFLOPS. If it is far below about 20, the run is overhead-bound: try `COMPILE_MODE=max-autotune-no-cudagraphs` or a bigger `BATCH`.

**Scaling to large data and large models:**
- `prepare.py all` tokenizes the full dataset in streaming chunks, with flat memory use.
- `train.py` keeps datasets of up to `GPU_DATA_GB` on the GPU. Larger datasets are memory-mapped from disk and fed by a background thread.
- Multi-GPU runs use `torchrun`.
- `CKPT_EVERY` / `RESUME=1` save and resume checkpoints.
- Bigger models only change the preset widths. The code path is the same, and zero-init residuals plus QK-norm keep deep stacks stable.

`test_causal.py` checks two things: no position sees its own future, and the factorised head gives an exact distribution.

Prior art: MegaByte (2023) and Block Transformer (2024) use the same global-patch / local-token split. Adaptive softmax is from Grave et al. (2017).

## Run on a GPU (vast.ai RTX 5060 Ti, `pytorch_cuda-12.8` image)

```bash
git clone https://github.com/modhisathvik7733/TinySpeedAI.git && cd TinySpeedAI
pip install -r requirements.txt
python prepare.py 200          # or: python prepare.py all  (full ~2.2 GB TinyStories, ~550M tokens)
python test_causal.py

# speed benchmark only (no evaluation), 300 steps per preset
for m in hpt_xxs hpt hpt_s; do QUICK=1 python train.py $m 300 2>&1 | grep -E "params,|steady"; done

# full comparison: GPT baseline vs HPT, with evaluation and samples
bash run.sh 1000 gpt hpt
# HPT runs at 131k tokens/step, so 1000 steps = 131M tokens. The full dataset avoids repeating epochs:
MB=all bash run.sh 3000 hpt hpt_s
```

Other options:
```bash
BATCH=256 python train.py hpt 2000                       # bigger batch
COMPILE_MODE=max-autotune-no-cudagraphs python train.py hpt 1000
CKPT_EVERY=500 python train.py hpt_s 20000               # later: RESUME=1 python train.py hpt_s 20000
torchrun --nproc_per_node=4 train.py hpt_m 20000         # multi-GPU
nohup bash run.sh 3000 gpt hpt > run.out 2>&1 &          # survives closing the browser
```

The RTX 50-series (sm_120) needs a PyTorch build for CUDA 12.8 or newer. If you see `no kernel image is available`, run `pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128`.

## Files
| file | what it does |
|---|---|
| `hpt.py` | the HPT model and its presets |
| `models.py` | the `GPT` baseline and the earlier `SGHM` models (`sghm`, `sghm_nogate`, `sghm_nomem`) |
| `prepare.py` | streams TinyStoriesV2, trains a 4096-token BPE tokenizer, writes `data/*.bin` |
| `train.py` | `python train.py <model> [steps]`, writes `runs/<model>.json` |
| `compare.py` | results table, including validation loss at matched tokens and at **matched wall-clock time** |
| `test_causal.py`, `test_equiv.py` | correctness checks for HPT and SGHM |

The earlier SGHM work is in [DESIGN.md](DESIGN.md) and [RESULTS.md](RESULTS.md).
