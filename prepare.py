"""Train a small BPE tokenizer on a TinyStories slice and encode train/val to uint16 arrays.

usage: python prepare.py [train_MB]   (default 200)
Uses a local copy if present, otherwise streams the first train_MB of TinyStoriesV2 from Hugging Face.
"""
import os, sys, time, urllib.request
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

LOCAL = os.path.expanduser("~/Extreme_Speed_Check/data/tinystories")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
HF = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
TRAIN_BYTES = int(float(sys.argv[1]) * 1e6) if len(sys.argv) > 1 else 200_000_000
VOCAB = 4096
EOT = "<|endoftext|>"


def fetch(url, dest, nbytes=None):
    """Download url to dest; with nbytes, only the first nbytes (HTTP range request)."""
    if os.path.exists(dest):
        return dest
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{nbytes - 1}"} if nbytes else {})
    print(f"downloading {url} ({'first %.0f MB' % (nbytes / 1e6) if nbytes else 'full'})")
    with urllib.request.urlopen(req) as r, open(dest + ".part", "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    os.replace(dest + ".part", dest)
    return dest


def sources():
    if os.path.exists(os.path.join(LOCAL, "train.txt")):
        return os.path.join(LOCAL, "train.txt"), os.path.join(LOCAL, "validation.txt")
    raw = os.path.join(OUT, "raw")
    os.makedirs(raw, exist_ok=True)
    tr = fetch(HF + "TinyStoriesV2-GPT4-train.txt", os.path.join(raw, f"train_{TRAIN_BYTES // 10**6}MB.txt"), TRAIN_BYTES)
    va = fetch(HF + "TinyStoriesV2-GPT4-valid.txt", os.path.join(raw, "valid.txt"))
    return tr, va


def stories(path, nbytes=None):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read(nbytes) if nbytes else f.read()
    parts = [p.strip() for p in text.split(EOT)]
    return [p for p in parts if p][:-1] if nbytes else [p for p in parts if p]


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    train_src, val_src = sources()
    train = stories(train_src, TRAIN_BYTES)
    val = stories(val_src)
    print(f"{len(train)} train stories, {len(val)} val stories")

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=[EOT],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(train, trainer)
    tok.save(os.path.join(OUT, "tok.json"))
    eot = tok.token_to_id(EOT)

    for name, docs in [("train", train), ("val", val)]:
        ids = []
        for enc in tok.encode_batch(docs):
            ids.extend(enc.ids)
            ids.append(eot)
        arr = np.array(ids, dtype=np.uint16)
        arr.tofile(os.path.join(OUT, f"{name}.bin"))
        print(f"{name}: {len(arr):,} tokens")
    open(os.path.join(OUT, f"prepared_{TRAIN_BYTES // 10**6}MB"), "w").close()
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
