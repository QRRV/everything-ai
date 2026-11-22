# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/snowball-stop-en.txt and data/snowball-stop-nl.txt: the lists of stop words of the Snowball project for
English (174 words) and Dutch (101 words), byte for byte as Snowball publishes them, with their comments. Lessons of
domain taal use them to leave very common words out of a text.

Snowball (https://snowballstem.org) gives out its software under the 3-clause BSD License, "Copyright (c) 2001, Dr
Martin Porter" and "Copyright (c) 2002, Richard Boulton", and asks that anyone who passes it on says so: REUSE.toml
and data/README.md do. The lists have no versions, so the course pins the files it took by their SHA-256:
`stop.txt` from https://snowballstem.org/algorithms/english/ and from https://snowballstem.org/algorithms/dutch/, as
published on November 22, 2026 (Last-Modified October 17, 2026). Download each into a folder of its own, as
`english/stop.txt` and `dutch/stop.txt`, and run, from the course folder:

    uv run python data/tools/make_snowball_stopwords.py <folder>

What the files hold: one stop word at the start of a line, in small letters, often followed by a comment; a comment
starts with a vertical bar `|`. The English list leaves some words out on purpose and says why in a comment, such as
*us* ("US = United States") and *will* ("He made a WILL"); the Dutch list gives an English translation of each word.
The course reads a word as the first item of a line, before the first `|` (everything_ai.books.stop_words).
"""
import hashlib
import sys
from pathlib import Path

from everything_ai import paths

FILES = {   # the course's file: (the address, Snowball's folder, the SHA-256 of the file)
    "snowball-stop-en.txt": ("https://snowballstem.org/algorithms/english/stop.txt", "english",
                             "06b8b14bbe6f6e0b64c0830813819416d96a71bec1655668d83b3e602a651db6"),
    "snowball-stop-nl.txt": ("https://snowballstem.org/algorithms/dutch/stop.txt", "dutch",
                             "c3252e08b041b795da5ea8f1ad9103b34f64248760848226b5e5c2f5e89789e7"),
}


def words(text: str) -> list:
    """The stop words of a Snowball list, in the order of the file: the first item of each line, before any `|`."""
    out = []
    for line in text.splitlines():
        parts = line.split("|", 1)[0].split()
        if parts:
            out.append(parts[0])
    return out


def main(folder: str) -> None:
    for name, (_, sub, sha256) in FILES.items():
        data = (Path(folder) / sub / "stop.txt").read_bytes()
        assert hashlib.sha256(data).hexdigest() == sha256, name
        found = words(data.decode("utf-8"))
        assert len(found) == len(set(found)) and all(w == w.lower() for w in found)
        out = paths.ROOT / "data" / name
        out.write_bytes(data)
        print(out, len(found), "words,", len(data), "bytes")


if __name__ == "__main__":
    main(sys.argv[1])
