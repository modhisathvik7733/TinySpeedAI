# TinySpeedAI: Surprise-Gated Hebbian Memory (SGHM)

SGHM is a language model whose context memory is a **fixed-size matrix of fast synapses**, written by Hebbian outer products and gated by surprise. It keeps no growing KV cache. It is tested against a matched GPT on TinyStories.

- [DESIGN.md](DESIGN.md): the first-principles reasoning, the three candidate mechanisms, and why this one was chosen.
- [RESULTS.md](RESULTS.md): results on an M1 Max (raw runs in `results/m1max/`).

## Files
| file | what it does |
|---|---|
| `prepare.py` | Streams the first 40 MB of TinyStoriesV2 from Hugging Face, trains a 4096-token BPE tokenizer, and writes `data/*.bin` |
| `models.py` | `SGHM` (parallel training form plus O(1)-state recurrent `step`) and the `GPT` baseline |
| `test_equiv.py` | Checks that the parallel and recurrent forms of SGHM agree |
| `train.py` | `python train.py <model> [steps]`. Picks CUDA, then MPS, then CPU; uses bf16 autocast on CUDA. Writes `runs/<model>.json` |
| `compare.py` | Prints the results table from `runs/*.json` |
| `run.sh` | Runs everything in order |

Models: `gpt`, `sghm` (sparse keys), `sghm_dense` (best so far), `sghm_nogate`, `sghm_nomem`.

## Run on a GPU (e.g. vast.ai RTX 5060 Ti, `pytorch_cuda-12.8` image)

Open a terminal on the instance (the Jupyter terminal or `ssh`), then:

```bash
git clone https://github.com/modhisathvik7733/TinySpeedAI.git
cd TinySpeedAI
pip install -r requirements.txt
nvidia-smi                                   # confirm the GPU is visible
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
bash run.sh                                  # 1000 steps: gpt, then sghm_dense
```

Other options:

```bash
bash run.sh 3000                                           # longer run
bash run.sh 1000 gpt sghm sghm_dense sghm_nogate sghm_nomem   # full ablation set
python train.py sghm_dense 2000                            # a single model
AMP=0 python train.py gpt 1000                             # fp32 instead of bf16
```

If you close the browser, the run stops unless it runs under `nohup` or `tmux`:

```bash
nohup bash run.sh 3000 > run.out 2>&1 &
tail -f run.out
```

**Notes:**
- The RTX 50-series (Blackwell, sm_120) needs a PyTorch build for CUDA 12.8 or newer. The vast.ai `pytorch_cuda-12.8.1` image has one. If you see `no kernel image is available`, run:

  ```bash
  pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
  ```
- Results are written to `runs/*.json`, including loss curves, samples and a surprise-gate trace.
- Validation numbers on the GPU use the official TinyStoriesV2 validation file, so they are not directly comparable with the M1 table. Compare GPT against SGHM within the same machine.
