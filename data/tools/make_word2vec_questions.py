# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/word2vec-questions-words.txt.gz: the analogy questions of word2vec (Mikolov, Chen, Corrado and Dean,
"Efficient Estimation of Word Representations in Vector Space", 2013), 19,544 questions such as
`Athens Greece Baghdad Iraq` in 14 groups, five about meaning (capitals, currencies, cities in states, family) and nine
about grammar (comparatives, plurals, past tenses and more). Lessons of domain taal measure analogies with them
(everything_ai.wordvectors).

The file is part of word2vec, which Google released under the Apache License 2.0. The course pins
`questions-words.txt` in Tomas Mikolov's own copy of word2vec on GitHub, at a fixed commit, by its SHA-256. Download
https://raw.githubusercontent.com/tmikolov/word2vec/20c129af10659f7c50e86e3be406df663beff438/questions-words.txt and
run, from the course folder:

    uv run python data/tools/make_word2vec_questions.py <questions-words.txt>

What changes: nothing. The file is gzipped as it is, with no name or date in the gzip header, so the same input gives
the same file. It holds 19,558 lines: a line `: <group>` before each of the 14 groups, then one question per line,
four words separated by spaces.
"""
import gzip
import hashlib
import sys

from everything_ai import paths

SOURCE = "https://raw.githubusercontent.com/tmikolov/word2vec/20c129af10659f7c50e86e3be406df663beff438/questions-words.txt"
SHA256 = "8c29b3332afc46f3fb8be04cb5297bf96f39aa7131272dff57869b4485b22a36"
OUT = paths.ROOT / "data" / "word2vec-questions-words.txt.gz"


def main(source: str) -> None:
    data = open(source, "rb").read()
    assert hashlib.sha256(data).hexdigest() == SHA256, "not the pinned questions-words.txt"
    lines = data.decode("utf-8").split("\n")
    assert len([x for x in lines if x.startswith(":")]) == 14 and len([x for x in lines if x and not x.startswith(":")]) == 19_544
    OUT.write_bytes(gzip.compress(data, compresslevel=9, mtime=0))
    print(OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main(sys.argv[1])
