"""Train a small BPE tokenizer on TinyStories and encode train/val to uint16 arrays, streaming.

usage: python prepare.py [train_MB | all]   (default 200; "all" = the full ~2.2 GB TinyStoriesV2 train set)
Uses a local copy if present, otherwise downloads from Hugging Face. Text is read and encoded in chunks and
appended to disk, so memory stays flat no matter how much data is used.
"""
import os, sys, time, urllib.request
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

LOCAL = os.path.expanduser("~/Extreme_Speed_Check/data/tinystories")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
HF = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
ARG = sys.argv[1] if len(sys.argv) > 1 else "200"
TRAIN_BYTES = None if ARG == "all" else int(float(ARG) * 1e6)
TOK_BYTES = 200_000_000          # the tokenizer is trained on at most this much text
CHUNK = 64 << 20                 # bytes of text encoded per step
VOCAB = 4096
EOT = "<|endoftext|>"


def fetch(url, dest, nbytes=None):
    """Download url to dest; with nbytes, only the first nbytes (HTTP range request)."""
    if os.path.exists(dest):
        return dest
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{nbytes - 1}"} if nbytes else {})
    print(f"downloading {url} ({'first %.0f MB' % (nbytes / 1e6) if nbytes else 'full'})", flush=True)
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
    name = "train_all.txt" if TRAIN_BYTES is None else f"train_{TRAIN_BYTES // 10**6}MB.txt"
    tr = fetch(HF + "TinyStoriesV2-GPT4-train.txt", os.path.join(raw, name), TRAIN_BYTES)
    va = fetch(HF + "TinyStoriesV2-GPT4-valid.txt", os.path.join(raw, "valid.txt"))
    return tr, va


def story_chunks(path, limit=None):
    """Yields lists of complete stories, reading the file CHUNK bytes at a time (up to limit bytes)."""
    rest, read = "", 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            n = CHUNK if limit is None else min(CHUNK, limit - read)
            text = f.read(n) if n > 0 else ""
            read += len(text)
            if not text:
                break
            parts = (rest + text).split(EOT)
            rest = parts.pop()                           # possibly incomplete last story
            yield [p.strip() for p in parts if p.strip()]
    if limit is None and rest.strip():                   # a full file ends on a complete story
        yield [rest.strip()]


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    train_src, val_src = sources()
    limit = TRAIN_BYTES and min(TRAIN_BYTES, os.path.getsize(train_src))

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=[EOT],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok_limit = min(limit or TOK_BYTES, TOK_BYTES)
    tok.train_from_iterator((s for c in story_chunks(train_src, tok_limit) for s in c), trainer)
    tok.save(os.path.join(OUT, "tok.json"))
    eot = tok.token_to_id(EOT)
    print(f"tokenizer trained in {time.time() - t0:.0f}s", flush=True)

    for name, src, lim in [("train", train_src, limit), ("val", val_src, None)]:
        n_tok = n_st = 0
        with open(os.path.join(OUT, f"{name}.bin.part"), "wb") as f:
            for docs in story_chunks(src, lim):
                ids = [i for enc in tok.encode_batch(docs) for i in (*enc.ids, eot)]
                np.array(ids, dtype=np.uint16).tofile(f)
                n_tok += len(ids); n_st += len(docs)
                print(f"\r{name}: {n_st:,} stories, {n_tok:,} tokens", end="", flush=True)
        os.replace(os.path.join(OUT, f"{name}.bin.part"), os.path.join(OUT, f"{name}.bin"))
        print()
    for f in os.listdir(OUT):
        if f.startswith("prepared_"):
            os.remove(os.path.join(OUT, f))
    open(os.path.join(OUT, f"prepared_{ARG}" + ("" if ARG == "all" else "MB")), "w").close()
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
