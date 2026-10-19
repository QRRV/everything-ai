# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Fine-tunes the course's own small LLM to follow instructions: the model that llm-sft and the lessons after it fall
back on. It starts from data/course-capstone.pt and runs the demo of llm-sft with the course's versions of every
module the learner writes, on two threads, for STEPS steps. The instructions are written, as in the lesson, for the
stories for young children of shard 5 of the stories of Cosmopedia (shards 1 to 4 trained the base model, shard 0 holds
the questions of data/story-cloze.json), with typographic quotes, dashes and ellipses made ASCII and the stories with
other non-ASCII characters left out. The course trains no model it shares on TinyStories (course/route/bronnen.yaml):
the held-out instructions the lesson tests with only take their words from it. Download the shard once, without an
account, and run from the course folder:

    curl -L -o cosmopedia-stories-00005.parquet https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/resolve/0ae6ec63f91742bd2d1eaef4f02232c55d719385/data/stories/train-00005-of-00043.parquet
    uv run --with pyarrow python data/tools/train_course_sft.py cosmopedia-stories-00005.parquet

It checks the shard's SHA-256 first. It writes the parameters in float16 to data/course-sft.pt, and its sizes, merge
rules, run and the answers before and after to data/course-sft.json."""
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time

os.environ["EAI_WORK"] = tempfile.mkdtemp()   # a project of its own, with only the course's versions in it

import torch  # noqa: E402

from everything_ai import paths  # noqa: E402
from everything_ai.check import run_code  # noqa: E402

SHARD_SHA256 = "6be05e303d68a62f85cc5dcd63435ad7f9d4399a119ff8826e52dff5fc308ed2"
ASCII = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
                       "…": "..."})
STORIES = 20_000   # the first stories for young children of the shard that are ASCII
STEPS = 3_000
OUT = paths.ROOT / "data" / "course-sft.pt"
INFO = paths.ROOT / "data" / "course-sft.json"


def stories(shard: str) -> list:
    import pyarrow.parquet as pq
    with open(shard, "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    if got != SHARD_SHA256:
        sys.exit(f"{shard}: SHA-256 {got}, expected {SHARD_SHA256}")
    table = pq.read_table(shard, columns=["text", "audience"]).to_pydict()
    chosen = []
    for text, audience in zip(table["text"], table["audience"]):
        story = text.strip().translate(ASCII)
        if audience == "young_children" and story and all(ord(c) < 128 for c in story):
            chosen.append(story)
        if len(chosen) == STORIES:
            break
    return chosen


def main(shard: str):
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location("sft_checks", paths.LESSONS / "llm-sft" / "checks.py")
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    names = {}
    for key in ("sft-example", "sft-batch", "words-used"):
        names.update(run_code(C.EXERCISES[key].solution).names)
    t = C.tools()
    model, info, source = C.P.load("course")
    assert source == "course" and not info.get("placeholder"), "data/course-capstone.pt is not the course's model yet"
    merges = {tuple(pair): 256 + i for i, pair in enumerate(info["merges"])}
    end = 256 + len(merges)
    made = C.pairs(stories(shard), merges, t["encode_fast"], t["decode"])
    examples = [names["sft_example"](list(prompt), list(answer), end) for _, prompt, answer in made]
    print(f"{len(examples)} examples", flush=True)
    last = [0.0]

    def show(p):
        if time.perf_counter() - last[0] > 60 or p.get("done") == p.get("total") and "done" in p:
            last[0] = time.perf_counter()
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in p.items()}, flush=True)

    before = C.rate(model, merges, names["words_used"], t["encode_fast"], t["decode"], show, "before")
    torch.manual_seed(0)
    tuned = C.tune(model, examples, names["sft_batch"], t["pretrain_step"], end, show, steps=STEPS)
    assert isinstance(tuned, dict), tuned
    after = C.rate(model, merges, names["words_used"], t["encode_fast"], t["decode"], show, "after")
    torch.save({k: v.half() for k, v in model.state_dict().items()}, OUT)
    keep = {"sizes": info["sizes"], "merges": info["merges"], "base": "course", "params": sum(p.numel() for p in model.parameters()),
            "pairs": len(examples), "steps": STEPS, "lr": C.LR, "seconds": round(tuned["seconds"], 1), "device": tuned["device"],
            "losses": [{k: round(v, 4) if isinstance(v, float) else v for k, v in e.items()} for e in tuned["losses"]],
            **{when: {k: r[k] for k in ("used", "stopped", "total")} for when, r in (("before", before), ("after", after))},
            "example": C.example(before, after),
            "hits": {when: "".join("1" if h else "0" for h in r["hits"]) for when, r in (("before", before), ("after", after))},
            "hours": round(tuned["seconds"] / 3600, 2), "threads": 2, "torch": torch.__version__, "stories": STORIES}
    INFO.write_text(json.dumps(keep, indent=1) + "\n", encoding="utf-8")
    shutil.rmtree(os.environ["EAI_WORK"], ignore_errors=True)
    print(f"before {before['used']} of {before['total']}, after {after['used']} of {after['total']}")
    print(OUT.name, hashlib.sha256(OUT.read_bytes()).hexdigest())
    print(INFO.name, hashlib.sha256(INFO.read_bytes()).hexdigest())


if __name__ == "__main__":
    main(sys.argv[1])
