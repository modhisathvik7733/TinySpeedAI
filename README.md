# TinySpeedAI

Small language models on TinyStories, built for **training tokens per second**.

## RCM: Reservoir Cortex Model (`res.py`, newest experiment)

RCM trains in **one streaming pass with no backpropagation**, the way reservoir computing does:
- **Fixed parts:** token embeddings computed in closed form from data statistics, leaky traces at 8 time-scales as fixed-size context memory, and a copy rule that recalls what followed the last occurrence of the current token or token pair.
- **Learned part:** only the linear readout, by one ridge-regression solve over statistics that add up across data shards and GPUs. Four scalars are then calibrated on held-out text to turn scores into probabilities.
- **What it doesn't need:** epochs, gradients or a learning rate.
- **Run:** `python res.py sr_res 45` trains within a 45 s budget, with the solve included in the time. `test_res.py` checks the parallel parts against brute force.

## SAC: Sparse Associative Cortex (`sac.py`)

SAC is a language model without attention and without a KV cache, where most of the network does not fire on any given token. Design and reasoning: [DESIGN.md](DESIGN.md).
- **Episodic memory:** a fixed-size fast-weight state per head, written only when the model is surprised. It binds names and objects within the context, and its cost per token does not depend on context length.
- **Semantic memory:** a product-key associative memory with 65k to 1M slots. Each token fires only 32 of them, so parameters grow without adding compute.
- **Wiring:** per-channel causal convolutions. No dense layer touches every weight for every token.
- **Training:** fully parallel (chunk-parallel memory, no sequential scan). Generation runs as an O(1)-state recurrence.

| model | params | MFLOP/token (T=256) | MFLOP/token (T=4096) |
|---|---|---|---|
| `sac` | 36M | 5.4 | 5.45 |
| `sac_s` | 27M | 3.7 | 3.75 |
| `sac_q` (quality) | 67M | 11.9 | 12.2 |
| `sac_m1m` (1M memory slots) | 540M | 6.5 | 6.6 |
| `fgpt` (best transformer) | 5.3M | 22.5 | 46.1 |
| `gpt` (baseline) | 5.0M | 29.9 | grows with T |

Previous experiments are also kept: `fgpt` (a GPT tuned for loss per second, which reached 1.70 against GPT's 1.84 in a 45 s speedrun on the 5060 Ti), `hpt` (patch transformer), and `sghm`.

## Run on a GPU (vast.ai RTX 5080 / 5060 Ti, `pytorch_cuda-12.8` image)

```bash
git clone https://github.com/modhisathvik7733/TinySpeedAI.git && cd TinySpeedAI
pip install -r requirements.txt
python prepare.py all            # full TinyStories, ~550M tokens (a few minutes, once)
python test_sac.py && python test_causal.py
bash scale.sh                    # tok/s vs context length and vs memory size (~5 min)
bash speedrun.sh 45              # equal 45 s budget: gpt, fgpt_muon, sac variants, ablations (~12 min)
```

Other options:
```bash
bash speedrun.sh 180 gpt fgpt_muon sac_muon sac_m262k_muon     # longer budget: gives big memories time to fill
CFG='{"k":16,"n_sub":512}' OPT=muon python train.py sac 3000    # override SAC preset fields
PKM_LR_MULT=3 OPT=muon python train.py sac 2000                 # higher LR for memory slots
torchrun --nproc_per_node=4 train.py sac_q 20000                 # multi-GPU
nohup bash speedrun.sh 45 > sr.out 2>&1 &                        # survives closing the browser
```

The RTX 50-series (sm_120) needs a PyTorch build for CUDA 12.8 or newer. If you see `no kernel image is available`, run `pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128`.

## Files
| file | what it does |
|---|---|
| `res.py` | RCM: backprop-free reservoir LM (one pass + ridge solve) |
| `sac.py` | the SAC model, presets and ablations |
| `fgpt.py`, `hpt.py` | earlier experiments: tuned GPT, patch transformer |
| `optim.py` | Muon optimizer |
| `speedrun.sh`, `scale.sh`, `tune.sh` | equal-time comparisons, scaling throughput, sweeps |
| `models.py` | the `GPT` baseline and the earlier `SGHM` models (`sghm`, `sghm_nogate`, `sghm_nomem`) |
| `prepare.py` | streams TinyStoriesV2, trains a 4096-token BPE tokenizer, writes `data/*.bin` |
| `train.py` | `python train.py <model> [steps]`, writes `runs/<model>.json` |
| `compare.py` | results table, including validation loss at matched tokens and at **matched wall-clock time** |
| `test_sac.py`, `test_causal.py`, `test_equiv.py` | correctness checks for SAC, HPT/FGPT heads, SGHM |

The earlier SGHM work is in [DESIGN.md](DESIGN.md) and [RESULTS.md](RESULTS.md).
