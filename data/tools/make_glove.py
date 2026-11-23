# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/glove-6b-50d-50k.npz: the vectors of 50 numbers of GloVe 6B for the 50,000 most frequent words that
consist of the letters a to z only, in float16. Lessons of domain taal use them as real word embeddings, to look up
nearest neighbors and analogies (everything_ai.wordvectors).

GloVe (Pennington, Socher and Manning, EMNLP 2014, https://nlp.stanford.edu/projects/glove/) 6B was trained on
Wikipedia of 2014 and Gigaword 5, 6 billion tokens in small letters, with a vocabulary of 400,000 words. The Stanford
NLP Group publishes the vectors under the Open Data Commons Public Domain Dedication and License (PDDL 1.0). The course
pins `glove.6B.50d.txt`, the first file in https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip, by its SHA-256;
the copy of the zip on Hugging Face has one more line at the end of that file, so it is another file. Unzip
`glove.6B.50d.txt` from the zip and run, from the course folder:

    uv run python data/tools/make_glove.py <glove.6B.50d.txt>

What changes: the file lists its 400,000 words from the most to the least frequent, each with its 50 numbers. The
course keeps, in that order, the first 50,000 words of the letters a to z only: the first 55,124 lines of the file,
without punctuation, numbers and tokens such as `n't` and `u.s.`. Each number is rounded to float16, which keeps about
three significant digits.

What the file holds: `words`, the 50,000 words as an array of strings, and `vectors`, an array of 50,000 rows of 50
numbers in float16, row i the vector of `words[i]`. The zip is written with a fixed date, so the same input gives the
same file.
"""
import hashlib
import io
import re
import sys
import zipfile

import numpy as np

from everything_ai import paths

SOURCE = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
SHA256 = "d8f717f8dd4b545cb7f418ef9f3d0c3e6e68a6f48b97d32f8b7aae40cb31f96f"   # of glove.6B.50d.txt in that zip
KEEP = 50_000
LETTERS = re.compile(r"[a-z]+")
OUT = paths.ROOT / "data" / "glove-6b-50d-50k.npz"


def _member(z: zipfile.ZipFile, name: str, array: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, array, allow_pickle=False)
    info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    z.writestr(info, buffer.getvalue())


def main(source: str) -> None:
    data = open(source, "rb").read()
    assert hashlib.sha256(data).hexdigest() == SHA256, "not the pinned glove.6B.50d.txt"
    words, rows, lines = [], [], 0
    for line in data.decode("utf-8").split("\n"):
        lines += 1
        word, numbers = line.split(" ", 1)
        if LETTERS.fullmatch(word):
            words.append(word)
            rows.append(np.array(numbers.split(), dtype=np.float32))
            if len(words) == KEEP:
                break
    vectors = np.stack(rows).astype(np.float16)
    assert vectors.shape == (KEEP, 50) and len(set(words)) == KEEP and lines == 55_124
    with zipfile.ZipFile(OUT, "w") as z:
        _member(z, "words", np.array(words))
        _member(z, "vectors", vectors)
    print(OUT, len(words), "words from", lines, "lines;", OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main(sys.argv[1])
