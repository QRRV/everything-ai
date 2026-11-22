# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/snowball-porter.tsv.gz and data/snowball-dutch-porter.tsv.gz: the sample vocabularies of the Snowball
project for Martin Porter's stemmer of 1980 (42,603 English words) and for Porter's stemmer for Dutch (45,670 Dutch
words), each word with the stem that Snowball's stemmer gives it. tests/test_stemmers.py checks the course's stemmers
in everything_ai.stemmers against them.

Snowball (https://snowballstem.org) keeps the files in its repository snowball-data, under the 3-clause BSD License
of its COPYING: "Copyright (c) 2001, Dr Martin Porter" and "Copyright (c) 2004,2005, Richard Boulton". REUSE.toml and
data/README.md say so. Snowball's pages link the files on raw.githubusercontent.com; they have no versions there, so
the course pins the files it took on November 22, 2026 by their SHA-256. Note the names: since Snowball 3.0.0 (2025),
the folder dutch/ holds the output of the stemmer of Kraaij and Pohlmann, and dutch_porter/ that of Porter's stemmer
for Dutch, which the course uses.

Download voc.txt and output.txt of the folders porter/ and dutch_porter/ of
https://raw.githubusercontent.com/snowballstem/snowball-data/master/ into a folder of your own, as porter/voc.txt,
porter/output.txt, dutch_porter/voc.txt and dutch_porter/output.txt, and run, from the course folder:

    uv run python data/tools/make_snowball_stems.py <folder>

Each file the course writes holds one line per word: the word, a tab, and its stem. Line i of output.txt is the stem of
line i of voc.txt. One English stem is empty: Porter's paper turns the word s into nothing.
"""
import gzip
import hashlib
import sys
from pathlib import Path

from everything_ai import paths

FILES = {   # the course's file: Snowball's folder, and the SHA-256 of its voc.txt and output.txt
    "snowball-porter.tsv.gz": ("porter", "fe09ef577f1c8e4872b79a4f6995e62d782bbdbcda03f65a5ddd3e07b8b0044f",
                               "6b91a82e05fb70a829af2fcc757599d2db3eccf9f3255343199817efbb4618a1"),
    "snowball-dutch-porter.tsv.gz": ("dutch_porter",
                                     "09864bdd572eda800b291429cb16ba889024e461290a495a0bc7f7f6c2ff9921",
                                     "cd1a15d7550bc1a1928c28b10e1cfa2d36cc3d86f006d04d2260088062d39af6"),
}


def lines(path: Path, sha256: str) -> list:
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == sha256, path
    text = data.decode("utf-8")
    assert text.endswith("\n") and "\r" not in text and "\t" not in text, path
    return text[:-1].split("\n")


def main(folder: str) -> None:
    for name, (sub, sha_voc, sha_out) in FILES.items():
        words = lines(Path(folder) / sub / "voc.txt", sha_voc)
        stems = lines(Path(folder) / sub / "output.txt", sha_out)
        assert len(words) == len(stems) == len(set(words)) and all(words), name
        out = paths.ROOT / "data" / name
        with open(out, "wb") as f:
            with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as g:   # the same bytes every run
                g.write("".join(f"{w}\t{s}\n" for w, s in zip(words, stems)).encode("utf-8"))
        print(out, len(words), "words,", out.stat().st_size, "bytes")


if __name__ == "__main__":
    main(sys.argv[1])
