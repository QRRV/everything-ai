# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Measure the course's GPT on tokens (data/course-gpt-bpe.pt, of gpt-train-bpe) on the story-cloze questions of
data/story-cloze.json: how many of the 500 questions of "items" (stories it never saw) and of "seen" (stories it trained
on) it gets right. For every option, it takes the log-probability of the option's tokens after the context, minus
their log-probability after an end-of-text alone, and picks the option where that difference is largest. The lesson
llm-decontamination quotes the outcome in MEASURED["gpt"]. Run from the course folder, on one thread (about 15
minutes):

    uv run python data/tools/measure_story_cloze.py"""
import json
import os
import sys
import tempfile

os.environ["EAI_WORK"] = tempfile.mkdtemp()   # a project of its own, with only the course's versions in it

import torch  # noqa: E402

from everything_ai import paths, projects  # noqa: E402
from everything_ai.check import run_code  # noqa: E402
from everything_ai.lesson import lesson  # noqa: E402


def main() -> None:
    torch.set_num_threads(1)
    B = lesson("gpt-train-bpe", "en").checks
    projects.save("llm", "fast_bpe", B.EXERCISES["train-bpe-fast"].solution)
    projects.save("llm", "fast_encode", B.EXERCISES["encode-fast"].solution)
    run = run_code("from llm.encode import encode\nfrom llm.gpt import GPT")
    if run.error is not None:
        sys.exit(f"the course's modules: {run.error}")
    encode, GPT = run.names["encode"], run.names["GPT"]
    base = paths.ROOT / B.COURSE_GPT
    info = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
    merges = {tuple(pair): 256 + i for i, pair in enumerate(info["merges"])}
    d, h, n_layers, block = info["sizes"]
    model = GPT(B.VOCAB, d, h, n_layers, block)
    state = torch.load(base.with_suffix(".pt"), map_location="cpu", weights_only=True)
    model.load_state_dict({k: v.float() for k, v in state.items()})
    model.eval()
    end = B.VOCAB - 1

    def logp(before: list, ids: list) -> float:
        """The log-probability of `ids` after `before`, within the model's window."""
        both = (before + ids)[-block:]
        with torch.no_grad():
            lp = torch.log_softmax(model(torch.tensor([both[:-1]]))[0].double(), dim=-1)
        target = torch.tensor(both[1:])
        return float(lp[torch.arange(len(target)), target][-len(ids):].sum())

    data = json.loads((paths.ROOT / "data" / "story-cloze.json").read_text(encoding="utf-8"))
    for part in ("items", "seen"):
        right = 0
        for q in data[part]:
            context = encode(q["context"], merges)
            gains = []
            for option in q["options"]:
                ids = encode(" " + option, merges)
                gains.append(logp(context, ids) - logp([end], ids))
            right += max(range(len(gains)), key=lambda i: gains[i]) == q["answer"]
        print(part, right, "of", len(data[part]), flush=True)


if __name__ == "__main__":
    main()
