# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes the answers that llm-safety-eval checks: the answers of the course's three models of the post-training
lessons, its base model (data/course-capstone.pt), its SFT model (data/course-sft.pt) and that model with its DPO
adapters (data/course-dpo.pt), to every request of data/course-red-team-prompts.json, in the template of the course's
chat model, each drawn as in the test of llm-sft (at its temperature, at most its number of tokens, the same seed for
a request on every model). Run from the course folder:

    uv run python data/tools/make_course_red_team_answers.py

It uses a process of one thread for each model, and writes data/course-red-team-answers.json."""
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

OUT = paths.ROOT / "data" / "course-red-team-answers.json"
FILES = {"base": "data/course-capstone.pt", "sft": "data/course-sft.pt", "dpo": "data/course-dpo.pt"}


def _lesson():
    spec = importlib.util.spec_from_file_location("safety_checks", paths.LESSONS / "llm-safety-eval" / "checks.py")
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
    answers = []
    for k, p in enumerate(C.prompts()):
        ids, _ = C.T.answer(model, t["encode_fast"](C.asked(p), merges), end, seed=k)
        answers.append(t["decode"](ids, merges))
    return which, answers


def main():
    C = _lesson()
    t0 = time.perf_counter()
    with multiprocessing.get_context("spawn").Pool(len(C.MODELS)) as pool:
        done = dict(pool.map(_work, C.MODELS))
    keep = {"models": {w: {"file": FILES[w], "sha256": hashlib.sha256((paths.ROOT / FILES[w]).read_bytes()).hexdigest()} for w in C.MODELS},
            "prompts": C.PROMPTS, "prompts_sha256": hashlib.sha256((paths.ROOT / C.PROMPTS).read_bytes()).hexdigest(),
            "tau": C.T.TAU, "length": C.T.LENGTH, "torch": torch.__version__, "minutes": round((time.perf_counter() - t0) / 60, 1),
            "answers": {w: done[w] for w in C.MODELS}}
    OUT.write_text(json.dumps(keep, indent=0, ensure_ascii=True) + "\n", encoding="utf-8")
    print(OUT.name, len(C.prompts()), "prompts", keep["minutes"], "minutes", hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    sys.exit(main())
