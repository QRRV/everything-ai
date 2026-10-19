# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Trains the course's own GPT on tokens, the CPU plan of the lesson gpt-train-bpe (seed 0), on stories of Cosmopedia
instead of TinyStories: the course trains no model it shares on TinyStories (course/route/bronnen.yaml). The stories
are those of the shard data/stories/train-00000-of-00043.parquet of HuggingFaceTB/cosmopedia (commit
0ae6ec63f91742bd2d1eaef4f02232c55d719385, Apache-2.0), with the audience young_children, in their order, only those
with nothing but ASCII characters, until 5,000,000 characters, joined by <|endoftext|>. It trains the course's
tokenizer on the first 90% of them, and then the GPT, with the course's versions of every module the learner writes on
the way. It writes the best parameters in float16 to data/course-gpt-bpe.pt, and its sizes, merge rules, losses and a
story to data/course-gpt-bpe.json. Lessons fall back on it when the learner has not trained a GPT on tokens. Download
the shard once, without an account, and run from the course folder:

    curl -L -o cosmopedia-stories-00000.parquet https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/resolve/0ae6ec63f91742bd2d1eaef4f02232c55d719385/data/stories/train-00000-of-00043.parquet
    uv run --with pyarrow python data/tools/train_course_gpt_bpe.py cosmopedia-stories-00000.parquet

It checks the shard's SHA-256 first, and runs on one thread, so that the same PyTorch gives the same parameters."""
import hashlib
import json
import os
import shutil
import sys
import tempfile

os.environ["EAI_WORK"] = tempfile.mkdtemp()   # a project of its own, with only the course's versions in it

import torch  # noqa: E402

from everything_ai import paths, projects  # noqa: E402
from everything_ai.lesson import lesson  # noqa: E402

SHARD_SHA256 = "c0c658bbe65657c4e03f4dd593e224ecf763c0a64a6dbe468b3cc3deacfc7a3c"
SIZE = 5_000_000
OUT = paths.ROOT / "data" / "course-gpt-bpe.pt"
INFO = paths.ROOT / "data" / "course-gpt-bpe.json"


def stories(shard: str) -> str:
    import pyarrow.parquet as pq
    with open(shard, "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    if got != SHARD_SHA256:
        sys.exit(f"{shard}: SHA-256 {got}, expected {SHARD_SHA256}")
    table = pq.read_table(shard, columns=["text", "audience"]).to_pydict()
    chosen, size = [], 0
    for text, audience in zip(table["text"], table["audience"]):
        if size >= SIZE:
            break
        story = text.strip()
        if audience == "young_children" and story and all(ord(c) < 128 for c in story):
            chosen.append(story)
            size += len(story) + len("<|endoftext|>")
    return "<|endoftext|>".join(chosen)


def main(shard: str):
    torch.set_num_threads(1)   # the same parameters on every computer with the same PyTorch
    C = lesson("gpt-train-bpe", "en").checks
    projects.save("llm", "fast_bpe", C.EXERCISES["train-bpe-fast"].solution)
    projects.save("llm", "fast_encode", C.EXERCISES["encode-fast"].solution)
    text = stories(shard)
    C.text = lambda: text   # the run trains on these stories instead of TinyStories
    d, _, n_layers, block = C.PLANS["cpu"]["sizes"]
    C.plan = lambda: {"device": "cpu", "name": "CPU", "small": None, **C.PLANS["cpu"], "params": C.n_params(C.VOCAB, d, n_layers, block)}
    shown = set()

    def show(p):
        if p.get("phase") == "train" and len(p["history"]) not in shown:
            shown.add(len(p["history"]))
            for h in p["history"][-1:]:
                print(f"epoch {h['epoch']}: training loss {h['train_loss']:.4f}, validation loss {h['val_loss']:.4f}", flush=True)

    result = C.train(show, seed=0)
    assert isinstance(result, dict) and not result["stopped"], result
    checkpoint, _ = C._paths()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    torch.save({k: v.half() for k, v in state.items()}, OUT)
    keep = ("sizes", "merges", "epochs", "params", "seed", "history", "best", "best_epoch", "best_per_char", "tokens", "chars",
            "val_tokens", "val_chars", "story")
    INFO.write_text(json.dumps({k: result[k] for k in keep}, indent=1) + "\n", encoding="utf-8")
    shutil.rmtree(os.environ["EAI_WORK"], ignore_errors=True)
    print(OUT.name, hashlib.sha256(OUT.read_bytes()).hexdigest())
    print(INFO.name, hashlib.sha256(INFO.read_bytes()).hexdigest())


if __name__ == "__main__":
    main(sys.argv[1])
