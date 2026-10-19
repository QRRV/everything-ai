# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Make data/tinystories-small.txt from TinyStoriesV2-GPT4-valid.txt of TinyStories (roneneldan/TinyStories on
Hugging Face, commit f54c09fd23315a6f9c86f9dc80f725de7d8f9c64): the stories in their order, from the second on (the
first starts in the middle of a sentence), only those with nothing but ASCII characters, until there are 5,000,000
characters, joined by <|endoftext|>. Download the file once, without an account:

    curl -L -o TinyStoriesV2-GPT4-valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStoriesV2-GPT4-valid.txt
    python data/tools/make_tinystories_small.py TinyStoriesV2-GPT4-valid.txt

Its SHA-256 is checked first. The result is written to data/, this folder's parent."""
import hashlib
import sys
from pathlib import Path

SOURCE_SHA256 = "6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d"
END = "<|endoftext|>"
SIZE = 5_000_000   # characters, the end-of-text tokens between the stories included


def main(source: Path) -> None:
    raw = source.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != SOURCE_SHA256:
        sys.exit(f"{source.name}: SHA-256 {got}, expected {SOURCE_SHA256}")
    stories = [s.strip() for s in raw.decode("utf-8").split(END) if s.strip()][1:]
    chosen, size = [], 0
    for story in stories:
        if size >= SIZE:
            break
        if all(ord(c) < 128 for c in story):
            chosen.append(story)
            size += len(story) + len(END)
    out = Path(__file__).resolve().parents[1] / "tinystories-small.txt"
    out.write_text(END.join(chosen), encoding="utf-8", newline="\n")
    print(out.name, len(chosen), "stories", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
