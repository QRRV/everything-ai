# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/gsm8k-sample.jsonl, the small part of GSM8K (Cobbe and others, 2021, MIT) that llm-reasoning-idea
uses: the first TRAIN problems of its training set, the worked examples of the lesson's prompt, and the first TEST
problems of its test set, the questions Qwen3 answers. Not chosen ones: the first, in the order of the files. Every
line is a problem of the files as it is, with "split" put in front: "train" or "test". Download the two files at the
commit below and run, from the course folder:

    uv run python data/tools/make_gsm8k_sample.py train.jsonl test.jsonl
"""
import hashlib
import json
import sys

from everything_ai import paths

BASE = "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/"
FILES = {"train": (BASE + "train.jsonl", "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"),
         "test": (BASE + "test.jsonl", "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14")}
TRAIN, TEST = 4, 30
OUT = paths.ROOT / "data" / "gsm8k-sample.jsonl"


def main(train: str, test: str) -> None:
    lines = []
    for split, path, n in (("train", train, TRAIN), ("test", test, TEST)):
        with open(path, "rb") as f:
            data = f.read()
        if hashlib.sha256(data).hexdigest() != FILES[split][1]:
            sys.exit(f"{path} is not the pinned {split}.jsonl: {FILES[split][0]}")
        for line in data.decode("utf-8").splitlines()[:n]:
            row = json.loads(line)
            lines.append(json.dumps({"split": split, "question": row["question"], "answer": row["answer"]}, ensure_ascii=False))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT, hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
