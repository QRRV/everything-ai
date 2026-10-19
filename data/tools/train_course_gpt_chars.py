# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Trains the course's own small GPT on the characters of data/tiny-shakespeare.txt, the CPU plan of the lesson
gpt-train-chars (seed 0), with the course's versions of every module the learner writes on the way: their attention,
block, GPT, gpt_examples, train_epoch, Trainer, gpt_loaders and gpt_run. It writes the best parameters to
data/course-gpt-chars.pt, in float16 to halve the file, and what a lesson needs to load them to
data/course-gpt-chars.json. Lessons fall back on it when the learner has not trained a GPT of their own. Run from the
course folder:

    uv run python data/tools/train_course_gpt_chars.py

It runs on one thread, so that the same PyTorch gives the same parameters; it takes about 20 minutes."""
import hashlib
import json
import os
import shutil
import tempfile

os.environ["EAI_WORK"] = tempfile.mkdtemp()   # a project of its own, with only the course's versions in it

import torch  # noqa: E402

from everything_ai import paths, projects  # noqa: E402
from everything_ai.lesson import lesson  # noqa: E402

OUT = paths.ROOT / "data" / "course-gpt-chars.pt"
INFO = paths.ROOT / "data" / "course-gpt-chars.json"


def main():
    torch.set_num_threads(1)   # the same parameters on every computer with the same PyTorch
    C = lesson("gpt-train-chars", "en").checks
    projects.save("llm", "gpt_loaders", C.EXERCISES["gpt-loaders"].solution)
    projects.save("llm", "gpt_run", C.EXERCISES["gpt-run"].solution)
    d, _, n_layers, block = C.PLANS["cpu"]["sizes"]
    # the CPU plan, also on a computer with a GPU
    C.plan = lambda: {"device": "cpu", "name": "CPU", "small": None, **C.PLANS["cpu"], "params": C.n_params(C.VOCAB, d, n_layers, block)}
    shown = set()

    def show(p):
        if len(p["history"]) not in shown:
            shown.add(len(p["history"]))
            for h in p["history"][-1:]:
                print(f"epoch {h['epoch']}: training loss {h['train_loss']:.4f}, validation loss {h['val_loss']:.4f}", flush=True)

    result = C.train(show, seed=0)
    assert isinstance(result, dict) and not result["stopped"], result
    checkpoint, _ = C._paths()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    torch.save({k: v.half() for k, v in state.items()}, OUT)
    keep = ("sizes", "chars", "epochs", "params", "seed", "history", "best", "best_epoch")
    INFO.write_text(json.dumps({k: result[k] for k in keep}, indent=1) + "\n", encoding="utf-8")
    shutil.rmtree(os.environ["EAI_WORK"], ignore_errors=True)
    print(OUT.name, hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
