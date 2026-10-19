# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes the answers that llm-eval-chat scores: the answers of the course's three models of the post-training
lessons, its base model (data/course-capstone.pt), its SFT model (data/course-sft.pt) and that model with its DPO
adapters (data/course-dpo.pt), to every prompt of the lesson, each drawn as in the test of llm-sft (at its temperature,
at most its number of tokens, the same seed for a prompt on every model). It also keeps the requested words of every
instruction the course's SFT model was fine-tuned on, made again from the shard that data/tools/train_course_sft.py
used, so that the lesson can tell which prompts are held out. With that shard, run from the course folder:

    uv run --with pyarrow python data/tools/make_course_eval_chat.py cosmopedia-stories-00005.parquet

It uses a process of one thread for each model, and writes data/course-eval-chat.json."""
import hashlib
import importlib.util
import json
import multiprocessing
import os
import sys
import tempfile
import time

os.environ["EAI_WORK"] = tempfile.mkdtemp()   # a project of its own, with only the course's versions in it

import torch  # noqa: E402

from everything_ai import paths  # noqa: E402

OUT = paths.ROOT / "data" / "course-eval-chat.json"
FILES = {"base": "data/course-capstone.pt", "sft": "data/course-sft.pt", "dpo": "data/course-dpo.pt"}


def _lesson():
    spec = importlib.util.spec_from_file_location("eval_chat_checks", paths.LESSONS / "llm-eval-chat" / "checks.py")
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    return C


def _work(which: str) -> tuple:
    torch.set_num_threads(1)
    C = _lesson()
    model, info, where = {"base": lambda: C.T.P.load("course"), "sft": lambda: C.T.load("course"),
                          "dpo": lambda: C.D.load("course")}[which]()
    assert where == "course"
    merges = {tuple(pair): 256 + i for i, pair in enumerate(info["merges"])}
    end = 256 + len(merges)
    t = C.T.tools()
    answers, ended = [], []
    for k, (_, _, prompt) in enumerate(C.prompts()):
        ids, stopped = C.T.answer(model, t["encode_fast"](prompt, merges), end, seed=k)
        answers.append(t["decode"](ids, merges))
        ended.append(stopped)
    return which, answers, ended


def sft_instructions(C, shard: str) -> list:
    """The requested words of every instruction of the course's SFT run, each set once, in order."""
    spec = importlib.util.spec_from_file_location("train_course_sft", paths.ROOT / "data" / "tools" / "train_course_sft.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    _, info, _ = C.T.P.load("course")
    merges = {tuple(pair): 256 + i for i, pair in enumerate(info["merges"])}
    t = C.T.tools()
    made = C.T.pairs(tool.stories(shard), merges, t["encode_fast"], t["decode"])
    return [list(w) for w in sorted({tuple(sorted(words)) for words, _, _ in made})]


def main(shard: str):
    C = _lesson()
    t0 = time.perf_counter()
    trained = sft_instructions(C, shard)
    with multiprocessing.get_context("spawn").Pool(len(C.MODELS)) as pool:
        done = dict((w, (a, e)) for w, a, e in pool.map(_work, C.MODELS))
    keep = {"models": {w: {"file": FILES[w], "sha256": hashlib.sha256((paths.ROOT / FILES[w]).read_bytes()).hexdigest()} for w in C.MODELS},
            "tau": C.T.TAU, "length": C.T.LENGTH, "torch": torch.__version__, "minutes": round((time.perf_counter() - t0) / 60, 1),
            "sft_instructions": trained, "prompts": [[kind, arg, text] for kind, arg, text in C.prompts()],
            "answers": {w: done[w][0] for w in C.MODELS}, "ended": {w: done[w][1] for w in C.MODELS}}
    OUT.write_text(json.dumps(keep, indent=0, ensure_ascii=True) + "\n", encoding="utf-8")
    print(OUT.name, len(keep["prompts"]), "prompts", keep["minutes"], "minutes", hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
