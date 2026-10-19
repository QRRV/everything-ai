# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes the answers that llm-dpo makes its preference pairs from: K answers of the course's SFT model,
data/course-sft.pt, to each of INSTRUCTIONS instructions, drawn as in the lesson's test of llm-sft (at its
temperature, at most its number of tokens). The instructions are made as the test instructions of llm-sft are, but
from other stories: one or two words of the course's own list of llm-sft that a story uses, every other instruction
one word. The stories are those for young children of shard 5 of the stories of Cosmopedia that come after the ones
data/tools/train_course_sft.py trained on, ASCII as there. None is one of the test instructions of llm-sft, so the
lessons can still test on those. Only the words and the answers are kept, as text, in
data/course-dpo-answers.json. With the shard of train_course_sft.py, run from the course folder:

    uv run --with pyarrow python data/tools/make_course_dpo_answers.py cosmopedia-stories-00005.parquet

It uses WORKERS processes of one thread each."""
import hashlib
import importlib.util
import json
import multiprocessing
import os
import random
import sys
import tempfile
import time

os.environ["EAI_WORK"] = tempfile.mkdtemp()   # a project of its own, with only the course's versions in it

import torch  # noqa: E402

from everything_ai import paths, projects  # noqa: E402

INSTRUCTIONS = 1000
K = 4
WORKERS = 3
SEED = 2
OUT = paths.ROOT / "data" / "course-dpo-answers.json"


def _sft():
    spec = importlib.util.spec_from_file_location("sft_checks", paths.LESSONS / "llm-sft" / "checks.py")
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    projects.builds_on(*C.BUILDS_ON)
    return C


def stories(shard: str) -> list:
    """The stories for young children of the shard that are ASCII, after the ones train_course_sft.py trained on."""
    import pyarrow.parquet as pq
    sft = _tool("train_course_sft")
    with open(shard, "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    if got != sft.SHARD_SHA256:
        sys.exit(f"{shard}: SHA-256 {got}, expected {sft.SHARD_SHA256}")
    table = pq.read_table(shard, columns=["text", "audience"]).to_pydict()
    chosen = []
    for text, audience in zip(table["text"], table["audience"]):
        story = text.strip().translate(sft.ASCII)
        if audience == "young_children" and story and all(ord(c) < 128 for c in story):
            chosen.append(story)
    return chosen[sft.STORIES:]


def instructions(C, shard: str) -> list:
    """INSTRUCTIONS lists of requested words, drawn as llm-sft draws its test instructions: from each story the
    words of its list that the story uses, one or two of them, every other instruction one; never one of the test
    instructions of llm-sft. An instruction may come more than once, from another story, with other answers."""
    rng = random.Random(SEED)
    taken = {tuple(sorted(w)) for w in C.tests()}
    out = []
    for story in stories(shard):
        chosen = sorted(w for w in C.found(story) if w in C.WORDS)
        if len(chosen) >= 2:
            words = sorted(rng.sample(chosen, 2 if len(out) % 2 else 1))
            if tuple(words) not in taken:
                out.append(words)
        if len(out) == INSTRUCTIONS:
            return out
    sys.exit(f"only {len(out)} instructions")


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(name, paths.ROOT / "data" / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _work(part: list) -> list:
    torch.set_num_threads(1)
    C = _sft()
    model, info, source = C.load("course")
    assert source == "course"
    merges = {tuple(pair): 256 + i for i, pair in enumerate(info["merges"])}
    end = 256 + len(merges)
    t = C.tools()
    out = []
    for i, words in part:
        prompt = t["encode_fast"](C.instruction(words), merges)
        answers, ended = [], []
        for k in range(K):
            ids, stopped = C.answer(model, list(prompt), end, seed=K * i + k)
            answers.append(t["decode"](ids, merges))
            ended.append(stopped)
        out.append((i, {"words": words, "answers": answers, "ended": ended}))
    return out


def main(shard: str):
    C = _sft()
    todo = list(enumerate(instructions(C, shard)))
    parts = [todo[w::WORKERS] for w in range(WORKERS)]
    t0 = time.perf_counter()
    with multiprocessing.get_context("spawn").Pool(WORKERS) as pool:
        done = sorted(x for part in pool.map(_work, parts) for x in part)
    items = [item for _, item in done]
    sft = (paths.ROOT / C.COURSE).with_suffix(".pt")
    keep = {"model": "data/course-sft.pt", "model_sha256": hashlib.sha256(sft.read_bytes()).hexdigest(),
            "tau": C.TAU, "length": C.LENGTH, "k": K, "seed": SEED, "torch": torch.__version__,
            "minutes": round((time.perf_counter() - t0) / 60, 1), "items": items}
    OUT.write_text(json.dumps(keep, indent=0, ensure_ascii=True) + "\n", encoding="utf-8")
    print(OUT.name, len(items), "instructions", keep["minutes"], "minutes", hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
