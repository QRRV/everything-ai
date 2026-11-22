# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/gutenberg-books.json.gz: eight books whose authors died long ago, five in English and three in Dutch,
from the plain-text files of Project Gutenberg, without Gutenberg's header, footer and licence. Lessons of domain taal
use them as a small corpus: concordances, Zipf and Heaps, spelling, TF-IDF and keywords.

The books are in the public domain: every author died before 1956, so the texts are free in the Netherlands and
Belgium too. The licence of the Gutenberg files allows reuse of a text that is in the public domain "as long as all
references to Project Gutenberg are removed" (section 1.C), so the course keeps only what stands between the lines
"*** START OF THE PROJECT GUTENBERG EBOOK" and "*** END OF THE PROJECT GUTENBERG EBOOK", and also drops the lines in
there that name Gutenberg or its proofreaders (DROP). Gutenberg makes its files again from time to time, so the
SHA-256 of each file is pinned below; the files were downloaded on November 22, 2026, from the archive server of the
Project Gutenberg Literary Archive Foundation. Download the eight files into one folder and run, from the course folder:

    uv run python data/tools/make_gutenberg.py <folder>

What the file holds: a JSON list with one object per book: key, title, author, year (of first publication), lang,
edition (where the Gutenberg file says which printing it follows) and text, the lines with "\\n" between them, and
without blank lines at the start and the end.
"""
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

from everything_ai import paths

SOURCE = "https://gutenberg.pglaf.org/cache/epub/{n}/pg{n}.txt"

BOOKS = [
    {"n": 11, "key": "alice", "title": "Alice's Adventures in Wonderland", "author": "Lewis Carroll", "year": 1865,
     "lang": "en", "edition": "",
     "sha256": "01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e"},
    {"n": 1342, "key": "pride", "title": "Pride and Prejudice", "author": "Jane Austen", "year": 1813, "lang": "en",
     "edition": "the edition of George Allen, 1894, with a preface by George Saintsbury",
     "sha256": "3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75"},
    {"n": 1661, "key": "holmes", "title": "The Adventures of Sherlock Holmes", "author": "Arthur Conan Doyle",
     "year": 1892, "lang": "en", "edition": "",
     "sha256": "922e2a12ccb43a4c9544c260b2166c6ad2097aeb5957faeee113f173bb857cd0"},
    {"n": 84, "key": "frankenstein", "title": "Frankenstein; or, The Modern Prometheus",
     "author": "Mary Wollstonecraft Shelley", "year": 1818, "lang": "en", "edition": "the revised text of 1831",
     "sha256": "2b97b7f8aad508649481d8dd4f6b54f9b965ddb3ba634e9151c815f537b49e42"},
    {"n": 2701, "key": "moby-dick", "title": "Moby-Dick; or, The Whale", "author": "Herman Melville", "year": 1851,
     "lang": "en", "edition": "",
     "sha256": "b5e88b71a1cda068e9ab1d94344c9d932338ba6da1a66def97a1aecd9cfcc87d"},
    {"n": 19563, "key": "eline-vere", "title": "Eline Vere: Een Haagsche roman", "author": "Louis Couperus",
     "year": 1889, "lang": "nl", "edition": "",
     "sha256": "a8b423aab8eaed1a763090b26a7228cb5a20573fff352d3d36f70eb073556f78"},
    {"n": 11024, "key": "havelaar", "title": "Max Havelaar", "author": "Multatuli", "year": 1860, "lang": "nl",
     "edition": "with the notes of the edition of 1875, revised in 1881",
     "sha256": "9cf9f2efa248146886505437f30f0f8fe906384b77bea7ac91ff72c347de1a9c"},
    {"n": 15975, "key": "camera-obscura", "title": "Camera Obscura", "author": "Hildebrand (Nicolaas Beets)",
     "year": 1839, "lang": "nl", "edition": "the 29th printing, Haarlem, De Erven F. Bohn, 1917",
     "sha256": "10ab71429ba58efbb2ea0055764a56a26bcf1f3d54cb57e88b60abe96bc92ee3"},
]

START = "*** START OF THE PROJECT GUTENBERG EBOOK"
END = "*** END OF THE PROJECT GUTENBERG EBOOK"
# lines between the markers that name Gutenberg or its proofreaders: (book, first line, last line), both kept out
DROP = [
    (19563, "Produced by Jeroen Hellingman and the Online Distributed", "Proofreading Team at http://www.pgdp.net/"),
    (15975, "Produced by Jeroen Hellingman, and the Online Distributed", "Proofreading Team at http://www.pgdp.net"),
    (2701, "Original Transcriber’s Notes:",
     "was compared with a public domain hard copy version of the text."),
]
GONE = re.compile(r"gutenberg|pgdp|proofread|transcriber|\betext", re.IGNORECASE)   # none of these may be left


def body(text: str, n: int) -> str:
    lines = text.replace("\r\n", "\n").split("\n")
    starts = [i for i, line in enumerate(lines) if line.startswith(START)]
    ends = [i for i, line in enumerate(lines) if line.startswith(END)]
    assert len(starts) == 1 and len(ends) == 1, n
    lines = lines[starts[0] + 1:ends[0]]
    for book, first, last in DROP:
        if book == n:
            a = [i for i, line in enumerate(lines) if line.strip() == first]
            b = [i for i, line in enumerate(lines) if line.strip() == last]
            assert len(a) == 1 and len(b) == 1 and a[0] <= b[0], (n, first)
            lines = lines[:a[0]] + lines[b[0] + 1:]
    out = "\n".join(lines).strip("\n")
    assert not GONE.search(out), n
    return out


def main(folder: str) -> None:
    books = []
    for b in BOOKS:
        data = (Path(folder) / f"pg{b['n']}.txt").read_bytes()
        assert hashlib.sha256(data).hexdigest() == b["sha256"], b["n"]
        text = body(data.decode("utf-8"), b["n"])
        books.append({k: b[k] for k in ("key", "title", "author", "year", "lang", "edition")} | {"text": text})
        print(f"{b['key']:15} {len(text):>9,} characters")
    out = paths.ROOT / "data" / "gutenberg-books.json.gz"
    with open(out, "wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as g:   # the same bytes every run
            g.write(json.dumps(books, ensure_ascii=False, indent=0).encode("utf-8"))
    print(out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main(*sys.argv[1:])
