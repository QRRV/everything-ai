# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Make data/story-cloze.json, the course's story-cloze questions, from stories of Cosmopedia: the shard
data/stories/train-00000-of-00043.parquet of HuggingFaceTB/cosmopedia (commit 0ae6ec63f91742bd2d1eaef4f02232c55d719385,
Apache-2.0), the same shard as the course's GPTs (data/tools/train_course_gpt_bpe.py). Of its stories for young children
with nothing but ASCII characters, in their order, the course's GPTs trained on those up to 5,000,000 characters; the
questions in "items" come from the stories after those, which no model of the course has seen, and the questions in
"seen" from the first 2,200 stories, which the course's GPTs trained on (they are in the first 90% of its tokens).

A question is the first three sentences of a story, its context, and four options for the sentence that follows: the
real fourth sentence of the story and the fourth sentences of three other stories of the same part, in an order drawn
with seed 0. Only stories of at least 5 sentences whose context has 25 to 90 words and whose fourth sentence 6 to 30 are used, the first
QUESTIONS of each part, and only those with an even number of quotes in the context and in that sentence, so that
no sentence is cut inside a quote. Download the shard once, without an account, and run from the course folder:

    curl -L -o cosmopedia-stories-00000.parquet https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/resolve/0ae6ec63f91742bd2d1eaef4f02232c55d719385/data/stories/train-00000-of-00043.parquet
    uv run --with pyarrow python data/tools/make_story_cloze.py cosmopedia-stories-00000.parquet

It checks the shard's SHA-256 first."""
import hashlib
import json
import random
import re
import sys
from pathlib import Path

SHARD_SHA256 = "c0c658bbe65657c4e03f4dd593e224ecf763c0a64a6dbe468b3cc3deacfc7a3c"
TRAINED = 5_000_000    # characters of stories the course's GPTs trained on, with an end-of-text after each
SEEN_STORIES = 2_200   # stories at the start of those: all in the first 90% of the tokens, the part they trained on
QUESTIONS = 500        # questions of each part
SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"])')   # after the end of a sentence, before a capital or a quote


def stories(shard: str) -> tuple[list, list]:
    """The stories the course's GPTs trained on, and those after them."""
    import pyarrow.parquet as pq
    with open(shard, "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    if got != SHARD_SHA256:
        sys.exit(f"{shard}: SHA-256 {got}, expected {SHARD_SHA256}")
    table = pq.read_table(shard, columns=["text", "audience"]).to_pydict()
    trained, after, size = [], [], 0
    for text, audience in zip(table["text"], table["audience"]):
        story = text.strip()
        if audience == "young_children" and story and all(ord(c) < 128 for c in story):
            if size < TRAINED:
                trained.append(story)
                size += len(story) + len("<|endoftext|>")
            else:
                after.append(story)
    return trained, after


def questions(part: list, first: int, seed: int) -> list:
    """The first QUESTIONS questions of a part of the stories; `first` is the place of its first story in the list of
    all stories for young children with nothing but ASCII characters."""
    found = []
    for i, story in enumerate(part):
        sentences = [s.strip() for s in SPLIT.split(re.sub(r"\s+", " ", story)) if s.strip()]
        if len(sentences) < 5:
            continue
        context, right = " ".join(sentences[:3]), sentences[3]
        whole = context.count('"') % 2 == 0 and right.count('"') % 2 == 0   # no sentence cut inside a quote
        if 25 <= len(context.split()) <= 90 and 6 <= len(right.split()) <= 30 and whole:
            found.append({"story": first + i, "context": context, "right": right})
        if len(found) == QUESTIONS:
            break
    rng = random.Random(seed)
    out = []
    for j, q in enumerate(found):
        others = rng.sample([p["right"] for k, p in enumerate(found) if k != j], 3)
        answer = rng.randrange(4)
        options = others[:answer] + [q["right"]] + others[answer:]
        out.append({"story": q["story"], "context": q["context"], "options": options, "answer": answer})
    return out


def main(shard: str) -> None:
    trained, after = stories(shard)
    data = {"items": questions(after, len(trained), 0), "seen": questions(trained[:SEEN_STORIES], 0, 1)}
    out = Path(__file__).resolve().parents[1] / "story-cloze.json"
    lines = ["{"]
    for name in ("items", "seen"):
        rows = ",\n".join(json.dumps(q, ensure_ascii=True) for q in data[name])
        lines.append(f'"{name}": [\n{rows}\n]' + ("," if name == "items" else ""))
    lines.append("}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(out.name, len(data["items"]), "+", len(data["seen"]), "questions", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main(sys.argv[1])
