# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/tatoeba-nl-en.tsv.gz: the English translations of the Dutch sentences of data/tatoeba-nl.tsv.gz. A lesson
of domain taal translates the Dutch sentences word for word and compares the result with every English translation
that people of Tatoeba wrote.

Tatoeba (https://tatoeba.org) releases its sentences under CC BY 2.0 FR. The export is made again every week and has
no versions, so the course pins the files it took by their SHA-256: the English sentences `eng_sentences.tsv.bz2` from
https://downloads.tatoeba.org/exports/per_language/eng/ and the links from Dutch to English `nld-eng_links.tsv.bz2`
from https://downloads.tatoeba.org/exports/per_language/nld/, the export of November 21, 2026, the same day as the
Dutch sentences of data/tatoeba-nl.tsv.gz. Download the two files into a folder and run, from the course folder:

    uv run python data/tools/make_tatoeba_nl_en.py <folder>

What the file holds: a header line `nl_id<TAB>en_id<TAB>text`, then one English translation per line, ordered by the
id of the Dutch sentence and then by the id of the English one. The Dutch sentence itself is in data/tatoeba-nl.tsv.gz
under `nl_id`. A Dutch sentence that is missing from that file (left out there, or added after its export) has no lines
here, and an English sentence that holds a word the Dutch file blocks, or one of the English words below, is left
out.
"""
import bz2
import gzip
import hashlib
import sys
from pathlib import Path

from everything_ai import books, paths

sys.path.insert(0, str(Path(__file__).parent))
from make_tatoeba_nl import WORD, blocked   # noqa: E402  the same blocked words as the Dutch sentences

SHA256 = {
    "eng_sentences.tsv.bz2": "0c4d3b18152bd6507bbf77ab690532b25de1e60da950f68705b3c88935c01fb4",
    "nld-eng_links.tsv.bz2": "c0460e79fc7ebb010524c43d1e64b12a38f8ed2c47e03743e77b544b1cc10c56",
}
BLOCK_EN = {
    # crude words, curses and slurs
    "damn", "damned", "goddamn", "crap", "ass", "asshole", "bastard", "dick", "piss", "pissed", "slut", "whore",
    "whores", "retard", "retarded", "nigger", "faggot",
    # sexual words
    "sexy", "sexual", "sexually", "naked", "nude", "condom", "condoms", "breasts", "porn", "brothel", "erotic",
    # violence, weapons and drugs
    "kill", "kills", "killed", "killing", "killer", "murder", "murdered", "murderer", "shoot", "shot", "shooting",
    "gun", "guns", "pistol", "rifle", "weapon", "weapons", "bomb", "bombs", "corpse", "suicide", "drugs", "cocaine",
    "heroin", "hitler", "war", "wars", "torture", "tortured", "rape", "raped", "hanged", "stab", "stabbed",
}
BLOCK_EN_START = ("fuck", "shit", "bitch", "nazi", "terroris", "prostitut")


def blocked_en(text: str) -> bool:
    words = [w.lower() for w in WORD.findall(text)]
    return blocked(text) or any(w in BLOCK_EN or w.startswith(BLOCK_EN_START) for w in words)


def read(folder: Path, name: str) -> list:
    data = (folder / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == SHA256[name], name
    return [line for line in bz2.decompress(data).decode("utf-8").split("\n") if line]


def main(folder: str) -> None:
    folder = Path(folder)
    dutch = dict(books.dutch_sentences())
    links = {}
    for line in read(folder, "nld-eng_links.tsv.bz2"):
        a, b = (int(x) for x in line.split("\t"))
        if a in dutch:
            links.setdefault(b, []).append(a)
    english = {}
    for line in read(folder, "eng_sentences.tsv.bz2"):
        sid, code, text = line.split("\t")
        assert code == "eng" and sid.isdigit()
        if int(sid) in links:
            english[int(sid)] = " ".join(text.split())
    pairs, left_out = [], 0
    for en_id, text in english.items():
        if not text:
            continue
        if blocked_en(text):
            left_out += len(links[en_id])
            continue
        pairs += [(nl_id, en_id, text) for nl_id in links[en_id]]
    pairs.sort()
    rows = ["nl_id\ten_id\ttext"] + [f"{a}\t{b}\t{t}" for a, b, t in pairs]
    print(f"{len(pairs):,} translations of {len({a for a, _, _ in pairs}):,} Dutch sentences, {left_out} left out")
    out = paths.ROOT / "data" / "tatoeba-nl-en.tsv.gz"
    with open(out, "wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as g:   # the same bytes every run
            g.write(("\n".join(rows) + "\n").encode("utf-8"))
    print(out, out.stat().st_size, "bytes", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main(*sys.argv[1:])
