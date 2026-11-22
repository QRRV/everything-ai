# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/tatoeba-nl.tsv.gz: the Dutch sentences of Tatoeba, short sentences in modern spelling written and
translated by volunteers, without the sentences that hold a crude, sexual or violent word. Lessons of domain taal use
them as a corpus of today's Dutch, next to the course's three Dutch books with their 19th-century spelling.

Tatoeba (https://tatoeba.org) releases its sentences under CC BY 2.0 FR. The export is made again every week and has
no versions, so the course pins the file it took by its SHA-256: `nld_sentences.tsv.bz2` from
https://downloads.tatoeba.org/exports/per_language/nld/, downloaded on November 21, 2026 (the export of that day).
Each sentence keeps its Tatoeba id, so it can be found again on https://tatoeba.org/sentences/show/<id> with its
author. Download the file into a folder and run, from the course folder:

    uv run python data/tools/make_tatoeba_nl.py <folder>

What the file holds: a header line `id<TAB>text`, then one sentence per line, in the order of the export (by id).
A sentence is left out when one of its words is in BLOCK or starts with one of BLOCK_START (in small letters), so no
crude, sexual or violent word and no slur reaches a lesson: Tatoeba is written by anyone, and a few of its sentences
are crude.
"""
import bz2
import gzip
import hashlib
import re
import sys
from pathlib import Path

from everything_ai import paths

SOURCE = "https://downloads.tatoeba.org/exports/per_language/nld/nld_sentences.tsv.bz2"
SHA256 = "514c3d02ebeaf076b5087edf9381e06b9ded3eba94662757bbb7cdd2ea82f62e"
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")   # as everything_ai.books

BLOCK = {
    # crude words and curses
    "kut", "lul", "lullen", "pik", "kloten", "klote", "tering", "kanker", "kankeren", "godverdomme", "verdomme",
    "godver", "shit", "fuck", "fucking", "bitch", "trut", "kreng", "sukkel", "debiel", "mongool", "achterlijk",
    # slurs
    "neger", "negers", "flikker", "flikkers",
    # sexual words
    "seks", "sex", "seksueel", "seksuele", "condoom", "condooms", "penis", "vagina", "orgasme", "borsten", "naakt",
    "bordeel", "erotisch", "erotische",
    # violence, weapons and drugs
    "moord", "moorden", "gedood", "doden", "doodde", "doodden", "doodgeschoten", "neergeschoten", "neerschieten",
    "geschoten", "doodschieten", "opgehangen", "zelfdoding", "pistool", "pistolen", "geweer", "geweren", "wapen",
    "wapens", "bom", "bommen", "lijk", "drugs", "cocaïne", "heroïne", "hitler", "oorlog", "oorlogen", "martelen",
    "gemarteld", "hoer", "hoeren", "hoertje",
}
BLOCK_START = ("neuk", "geneukt", "verkracht", "vermoord", "moordenaar", "zelfmoord", "terroris", "prostitu", "porno",
               "nazi", "klootzak", "godverdom", "slet")


def blocked(text: str) -> bool:
    words = [w.lower() for w in WORD.findall(text)]
    return any(w in BLOCK or w.startswith(BLOCK_START) for w in words)


def main(folder: str) -> None:
    data = (Path(folder) / "nld_sentences.tsv.bz2").read_bytes()
    assert hashlib.sha256(data).hexdigest() == SHA256
    rows, left_out = [], 0
    for line in bz2.decompress(data).decode("utf-8").split("\n"):
        if not line:
            continue
        sid, lang, text = line.split("\t")
        assert lang == "nld" and sid.isdigit()
        text = " ".join(text.split())
        if not text:
            continue
        if blocked(text):
            left_out += 1
            continue
        rows.append(f"{sid}\t{text}")
    out = paths.ROOT / "data" / "tatoeba-nl.tsv.gz"
    with open(out, "wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as g:   # the same bytes every run
            g.write(("id\ttext\n" + "\n".join(rows) + "\n").encode("utf-8"))
    print(f"{len(rows):,} sentences kept, {left_out:,} left out")
    print(out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main(*sys.argv[1:])
