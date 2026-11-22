# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/tatoeba-nl-pairs.tsv.gz: sentences of Tatoeba in Frisian, Gronings and Limburgish, each with a Dutch
translation from data/tatoeba-nl.tsv.gz. A lesson of domain taal lays both sides along the word list of a treebank in
standard Dutch: the two sentences say the same thing, so what the list misses on one side and not on the other comes
from the variety, not from the topic.

Tatoeba (https://tatoeba.org) releases its sentences under CC BY 2.0 FR. The export is made again every week and has
no versions, so the course pins the files it took by their SHA-256: for each of fry (Frisian), gos (Gronings) and lim
(Limburgish) the sentences `<code>_sentences.tsv.bz2` and the links to Dutch `<code>-nld_links.tsv.bz2` from
https://downloads.tatoeba.org/exports/per_language/<code>/, the export of November 21, 2026, the same day as the Dutch
sentences of data/tatoeba-nl.tsv.gz. Download the six files into a folder and run, from the course folder:

    uv run python data/tools/make_tatoeba_pairs.py <folder>

What the file holds: a header line `lang<TAB>id<TAB>text<TAB>nl_id<TAB>nl_text`, then one sentence per line, the
languages in the order of LANGS and the sentences of a language by id. Each sentence comes once. A sentence with more
than one Dutch translation keeps the one with the lowest id; a sentence whose translations are all missing from
data/tatoeba-nl.tsv.gz (left out there, or added after its export) is left out, and so is a sentence that holds a word
the Dutch file blocks.
"""
import bz2
import gzip
import hashlib
import sys
from pathlib import Path

from everything_ai import books, paths

sys.path.insert(0, str(Path(__file__).parent))
from make_tatoeba_nl import blocked   # noqa: E402  the same blocked words as the Dutch sentences

LANGS = ("fry", "gos", "lim")
SHA256 = {
    "fry_sentences.tsv.bz2": "de8648c80a21db0b5c386ae003c31c2225bf19e8331b52d2d4b00021159a3907",
    "fry-nld_links.tsv.bz2": "fecd12c4743d8903c68a96012cf25637fd4040e934ab745c88a4a636932c298a",
    "gos_sentences.tsv.bz2": "56d7f157b374cff8c833b6772d0fd9441f54e3373a542c93f8398ed6b47fbce1",
    "gos-nld_links.tsv.bz2": "0242265c9893408a1282ee6ca533d8722c4dcf28e3b1f164dfceb4a80b0c9698",
    "lim_sentences.tsv.bz2": "190b786c288ba01954cfa9472930e0f655594209c2a288e9cddb83e9f9a4af11",
    "lim-nld_links.tsv.bz2": "32a12ad084dc3433d9cc6a43ad417d74b8f6b563746f43b1a594bd20bbf019d4",
}


def read(folder: Path, name: str) -> list:
    data = (folder / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == SHA256[name], name
    return [line for line in bz2.decompress(data).decode("utf-8").split("\n") if line]


def main(folder: str) -> None:
    folder = Path(folder)
    dutch = dict(books.dutch_sentences())
    rows = ["lang\tid\ttext\tnl_id\tnl_text"]
    for lang in LANGS:
        sentences = {}
        for line in read(folder, f"{lang}_sentences.tsv.bz2"):
            sid, code, text = line.split("\t")
            assert code == lang and sid.isdigit()
            text = " ".join(text.split())
            if text:
                sentences[int(sid)] = text
        translation = {}
        for line in read(folder, f"{lang}-nld_links.tsv.bz2"):
            a, b = (int(x) for x in line.split("\t"))
            if a in sentences and b in dutch:
                translation[a] = min(b, translation.get(a, b))
        kept = left_out = 0
        for sid in sorted(translation):
            text, nl_id = sentences[sid], translation[sid]
            if blocked(text):
                left_out += 1
                continue
            rows.append(f"{lang}\t{sid}\t{text}\t{nl_id}\t{dutch[nl_id]}")
            kept += 1
        print(f"{lang}: {len(sentences):,} sentences, {kept:,} kept with a Dutch translation, {left_out} blocked")
    out = paths.ROOT / "data" / "tatoeba-nl-pairs.tsv.gz"
    with open(out, "wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as g:   # the same bytes every run
            g.write(("\n".join(rows) + "\n").encode("utf-8"))
    print(out, out.stat().st_size, "bytes", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main(*sys.argv[1:])
