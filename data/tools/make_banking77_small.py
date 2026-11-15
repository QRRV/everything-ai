# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/banking77-small.json: questions of customers of an online bank, each with what it is about, for the
lessons build-ai-services and build-llm-or-small-model. They come from BANKING77 (Casanueva and others, 2020,
"Efficient Intent Detection with Dual Sentence Encoders", arXiv:2003.04807; CC-BY-4.0), which gives 13,083 questions
one of 77 intents. The lessons use six of them, two of which are easy to mix up (exchange_rate and
card_payment_wrong_exchange_rate): TRAIN questions of each from BANKING77's training set and TEST from its test set,
chosen by the SHA-256 of their text, so that the choice depends on nothing else, with every run of spaces made one
space. A text that is in both sets, or twice in one (ignoring capitals), is used once, on the side of the training
set. Download the two files at the commit below and run, from the course folder:

    uv run python data/tools/make_banking77_small.py train.csv test.csv
"""
import csv
import hashlib
import json
import sys

from everything_ai import paths

COMMIT = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SOURCE = f"https://github.com/PolyAI-LDN/task-specific-datasets/tree/{COMMIT}/banking_data"
FILES = {"train": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
         "test": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"}
LABELS = ("card_arrival", "lost_or_stolen_card", "exchange_rate", "card_payment_wrong_exchange_rate", "pin_blocked",
          "request_refund")
TRAIN, TEST = 50, 20   # questions per label


def _rows(path: str, sha: str) -> list[tuple[str, str]]:
    with open(path, "rb") as f:
        data = f.read()
    assert hashlib.sha256(data).hexdigest() == sha, path
    rows = list(csv.reader(data.decode("utf-8").splitlines()))
    assert rows[0] == ["text", "category"], rows[0]
    return [(" ".join(text.split()), label) for text, label in rows[1:] if label in LABELS]


def _order(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main(train_csv: str, test_csv: str) -> None:
    seen, out = set(), {}
    for side, path, n in (("train", train_csv, TRAIN), ("test", test_csv, TEST)):
        rows = _rows(path, FILES[side])
        picked = []
        for label in LABELS:
            texts = sorted({t for t, l in rows if l == label and t.lower() not in seen}, key=_order)[:n]
            assert len(texts) == n, (side, label)
            seen.update(t.lower() for t in texts)
            picked += [[label, t] for t in texts]
        out[side] = sorted(picked, key=lambda p: _order(p[1]))
    data = {"about": "Questions of customers of an online bank, each with what it is about, from BANKING77 (CC-BY-4.0): "
                     f"{TRAIN} per label from its training set and {TEST} per label from its test set, for six of its "
                     f"77 intents. Made with data/tools/make_banking77_small.py from {SOURCE}.",
            "labels": list(LABELS), **out}
    text = json.dumps(data, ensure_ascii=False, indent=0) + "\n"
    (paths.ROOT / "data" / "banking77-small.json").write_text(text, encoding="utf-8")
    print(len(out["train"]), len(out["test"]), len(text.encode()))


if __name__ == "__main__":
    main(*sys.argv[1:3])
