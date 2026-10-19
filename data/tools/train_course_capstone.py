# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Trains the course's own small LLM, the model that llm-pretrain-capstone and the lessons after it fall back on: the
model of depth 6 of the family of llm-fit-scaling-law, compute-optimal at 20 tokens per parameter, with the course's
versions of every module the learner writes, on two threads. It trains on the stories for young children of shards 1 to
4 of the stories of Cosmopedia (shard 0 holds the questions of data/story-cloze.json), with typographic quotes, dashes
and ellipses made ASCII and the stories with other non-ASCII characters left out; the last 1% of the stories are for
validation. The course trains no model it shares on TinyStories (course/route/bronnen.yaml). It encodes with the
course's tokenizer, the merge rules of data/course-modern-gpt.json. Every 500 steps it keeps its state in the file
CAPSTONE_STATE names (capstone-state.pt by default), so that a run that stops can go on from there. Download the shards
once, without an account, and run from the course folder:

    for i in 1 2 3 4; do curl -L -o cosmopedia-stories-0000$i.parquet https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/resolve/0ae6ec63f91742bd2d1eaef4f02232c55d719385/data/stories/train-0000$i-of-00043.parquet; done
    uv run --with pyarrow python data/tools/train_course_capstone.py cosmopedia-stories-0000{1,2,3,4}.parquet

It checks the shards' SHA-256 first. It writes the parameters in float16 to data/course-capstone.pt, and its sizes,
merge rules, run and a story to data/course-capstone.json."""
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ["EAI_WORK"] = tempfile.mkdtemp()   # a project of its own, with only the course's versions in it

import torch  # noqa: E402

from everything_ai import paths  # noqa: E402
from everything_ai.check import run_code  # noqa: E402

SHARDS_SHA256 = ["78e79f4de9300ce7b00e305665d065719a6688a263bf92d6d2e3256bdd4e98cf",
                 "8d513eb1c273ecda161a8a3f76c03b80a76cd455bc439e32040abf612f037f1e",
                 "33c85b5e9d23282c5805ae922613856ecc2aaafc376db9398811096995e13ede",
                 "839854ea3388163b70bc20b698efbd4778a0feb7e88fe7a1b83ed94507ea112e"]
ASCII = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
                       "…": "..."})
DEPTH = 6
OUT = paths.ROOT / "data" / "course-capstone.pt"
INFO = paths.ROOT / "data" / "course-capstone.json"
STATE = Path(os.environ.get("CAPSTONE_STATE", "capstone-state.pt"))


def stories(shards: list) -> list:
    import pyarrow.parquet as pq
    if len(shards) != len(SHARDS_SHA256):
        sys.exit(f"give the {len(SHARDS_SHA256)} shards 1 to 4, in that order")
    chosen = []
    for shard, want in zip(shards, SHARDS_SHA256):
        with open(shard, "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
        if got != want:
            sys.exit(f"{shard}: SHA-256 {got}, expected {want}")
        table = pq.read_table(shard, columns=["text", "audience"]).to_pydict()
        for text, audience in zip(table["text"], table["audience"]):
            story = text.strip().translate(ASCII)
            if audience == "young_children" and story and all(ord(c) < 128 for c in story):
                chosen.append(story)
    return chosen


def main(shards: list):
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location("capstone_checks", paths.LESSONS / "llm-pretrain-capstone" / "checks.py")
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    course = json.loads((paths.ROOT / "data" / "course-modern-gpt.json").read_text(encoding="utf-8"))
    merges = {tuple(pair): 256 + i for i, pair in enumerate(course["merges"])}
    told = set()

    def show(p):
        if p["phase"] == "eval":
            print(f"step {p['step']} of {p['steps']}: training loss {p['train']:.4f}, validation loss {p['val']:.4f}, "
                  f"{p['seconds'] / 60:.1f} min", flush=True)
        elif p["phase"] == "encode" and p["stories"] // 20000 not in told:
            told.add(p["stories"] // 20000)
            print(f"encoded {p['stories']} of {p['total']} stories", flush=True)

    every = stories(shards)
    split = len(every) - len(every) // 100
    written = C.prepare(show, stories=every[:split], merges=merges, source="course")
    assert isinstance(written, dict), written
    tools = run_code("from llm.fast_encode import encode_fast\nfrom llm.shards import write_shards\n"
                     "from llm.token_windows import TokenWindows\nfrom llm.encode import encode\nfrom llm.decode import decode").names
    val_ids = tools["encode_fast"](C.B.END.join(every[split:]) + C.B.END, merges)
    val_folder = C.folder().parent / "tokens_capstone_val"
    val_folder.mkdir(parents=True, exist_ok=True)
    val_files = tools["write_shards"](val_ids, C.B.VOCAB, str(val_folder), C.D.SHARD)
    import numpy as np
    train_windows = C._windows(tools, written)
    val_windows = tools["TokenWindows"](val_files, C.BLOCK, np.dtype(written["dtype"]))
    params = C.sizes(run_code("from llm.modern_gpt import ModernGPT").names["ModernGPT"])[DEPTH]
    result = C.train(DEPTH, C.TOKENS_PER_PARAM * params, C.course_step(), False, show, train_windows=train_windows,
                     val_windows=val_windows, state_file=STATE, save_every=500)
    assert isinstance(result, dict), result
    story = C.write(result["model"], merges, tools["encode"], tools["decode"])
    torch.save({k: v.half() for k, v in result["model"].state_dict().items()}, OUT)
    d, h = C.family(DEPTH)
    keep = {"depth": DEPTH, "sizes": [C.B.VOCAB, d, h, h, DEPTH], "merges": course["merges"], "params": result["params"],
            "tokens": result["tokens"], "steps": result["steps"], "batch": result["batch"], "val": round(result["val"], 4),
            "history": [{k: round(v, 4) if isinstance(v, float) else v for k, v in e.items()} for e in result["history"]],
            "hours": round(result["seconds"] / 3600, 2), "threads": 2, "torch": torch.__version__,
            "stories": split, "val_stories": len(every) - split, "train_ids": written["ids"], "val_ids": len(val_ids),
            "story": story}
    INFO.write_text(json.dumps(keep, indent=1) + "\n", encoding="utf-8")
    shutil.rmtree(os.environ["EAI_WORK"], ignore_errors=True)
    STATE.unlink(missing_ok=True)
    print(OUT.name, hashlib.sha256(OUT.read_bytes()).hexdigest())
    print(INFO.name, hashlib.sha256(INFO.read_bytes()).hexdigest())


if __name__ == "__main__":
    main(sys.argv[1:])
