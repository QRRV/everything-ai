# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/tatoeba-langs.tsv.gz: 500 short sentences of Tatoeba in each of twelve languages, so that a lesson of
domain taal can count the letter sequences of each language and tell languages apart. Dutch comes with its close
neighbors (Afrikaans, Frisian, German, English), three Scandinavian languages that write much alike (Danish,
Norwegian Bokmål, Swedish), and four Romance languages (French, Spanish, Portuguese, Italian).

Tatoeba (https://tatoeba.org) releases its sentences under CC BY 2.0 FR. The export is made again every week and has
no versions, so the course pins the files it took by their SHA-256: for each language below except Dutch the
sentences `<code>_sentences.tsv.bz2` and the links to Dutch `<code>-nld_links.tsv.bz2` from
https://downloads.tatoeba.org/exports/per_language/<code>/, the export of November 21, 2026, the same day as the Dutch
sentences of data/tatoeba-nl.tsv.gz. Download the 22 files into a folder and run, from the course folder:

    uv run python data/tools/make_tatoeba_langs.py <folder>

Which sentences: the Dutch ones come from data/tatoeba-nl.tsv.gz, which leaves out sentences with a crude, sexual or
violent word. A sentence in another language is kept only when it has a Dutch translation in that file and holds no
word that the Dutch file blocks: the course cannot read every language, and a crude sentence seldom has a clean Dutch
translation. Of the sentences that are left, each language keeps the 500 whose SHA-256 of `<code><TAB><id>` comes
first, a fixed choice that does not favor short or early sentences; Norwegian Bokmål has fewer than 500.

What the file holds: a header line `lang<TAB>id<TAB>text`, then one sentence per line, the languages in the order of
LANGS under their two-letter code, and the sentences of a language by id. Each sentence keeps its Tatoeba id, so it
can be found again on https://tatoeba.org/sentences/show/<id> with its author.
"""
import bz2
import gzip
import hashlib
import sys
from pathlib import Path

from everything_ai import books, paths

sys.path.insert(0, str(Path(__file__).parent))
from make_tatoeba_nl import blocked   # noqa: E402  the same blocked words as the Dutch sentences

# two-letter code of the course: code of Tatoeba
LANGS = {"nl": "nld", "af": "afr", "fy": "fry", "de": "deu", "en": "eng", "da": "dan", "nb": "nob", "sv": "swe",
         "fr": "fra", "es": "spa", "pt": "por", "it": "ita"}
PER_LANGUAGE = 500
SHA256 = {
    "afr_sentences.tsv.bz2": "062388fa98db83e84be2f3cc373f37fbfeea5b3557e69c0ab948fd6fddfee162",
    "afr-nld_links.tsv.bz2": "534943d1db4bf646a7491bfa402ba00d07cb72f260a5f7b0be2b7345341114e2",
    "fry_sentences.tsv.bz2": "de8648c80a21db0b5c386ae003c31c2225bf19e8331b52d2d4b00021159a3907",
    "fry-nld_links.tsv.bz2": "fecd12c4743d8903c68a96012cf25637fd4040e934ab745c88a4a636932c298a",
    "deu_sentences.tsv.bz2": "e16c57e695bfeccccb455c305d262c5cbe2aa74a5c1e1edb725d53cf907309ac",
    "deu-nld_links.tsv.bz2": "3e50ccb3325ad0ebb4222b05b4501784165319628bd90a483e1d702d5bb8e5bc",
    "eng_sentences.tsv.bz2": "0c4d3b18152bd6507bbf77ab690532b25de1e60da950f68705b3c88935c01fb4",
    "eng-nld_links.tsv.bz2": "9f9d452a75093546b7a7cb3ac01781c47aee291afe3a5493b89d8d185a938751",
    "dan_sentences.tsv.bz2": "2023284d05885cde0aa18ac8d1dad104b78db6eb61dee7a6d2e703b2222800a0",
    "dan-nld_links.tsv.bz2": "21eb7da50ee63af436f4242f22780e9abab6087e0642a44589f2b1e7536115d4",
    "nob_sentences.tsv.bz2": "1d65805473b6c20aa9c9220942cc6c91d43f7ca3cdd5be73f31031eb58d99ebd",
    "nob-nld_links.tsv.bz2": "4f48b83261fa7373dd5daffca685da033cffcad39ccb17d058137075bdf33372",
    "swe_sentences.tsv.bz2": "c9a3b4f92e995ca9da40244aec017d5e6c598e4f8ed2ec3255f52abc2a4597db",
    "swe-nld_links.tsv.bz2": "a8d5a89d396554933357ff6c5ddb567a0106d59be44a5973cfafe124e0dd2a17",
    "fra_sentences.tsv.bz2": "c9e91cc5a0640db7bfc4423fadf3985db42df08dafce4983d4c6b8c2cf7a4d3d",
    "fra-nld_links.tsv.bz2": "88db2b5033330651bfc4b735938fae6088264dbb3a31467bdda5229c4a2a120c",
    "spa_sentences.tsv.bz2": "7872d2e66debfd8e73e96cc9bca9560a32cc914aee9c820649d7d3d5e7353f48",
    "spa-nld_links.tsv.bz2": "9fb9e7ca115899cef53c190a62dfc0623a3e456e16fd2fa632ef8d9093fbc5bc",
    "por_sentences.tsv.bz2": "d79dfc10fdd0959a259e282f097ea6be59dd58d3d1fe69b1559fd8499efaab7c",
    "por-nld_links.tsv.bz2": "f69b36cda3ae5a58a64cd3142ea923f6415b61e855f73794f4043e8be15b7bbc",
    "ita_sentences.tsv.bz2": "20dc0feb2a4340daf69d42fd4bfb51e4d04c899e2a5bce3692169de9fe1a11fe",
    "ita-nld_links.tsv.bz2": "c76dc42109d86d3af9352a34dff4ab1fcfb470c95c1bee81f2afee99a6bb1150",
}


def read(folder: Path, name: str) -> list:
    data = (folder / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == SHA256[name], name
    return [line for line in bz2.decompress(data).decode("utf-8").split("\n") if line]


def pool(folder: Path, code: str, dutch: dict) -> dict:
    """The sentences of one language that have a Dutch translation in data/tatoeba-nl.tsv.gz and no blocked word."""
    sentences = {}
    for line in read(folder, f"{code}_sentences.tsv.bz2"):
        sid, lang, text = line.split("\t")
        assert lang == code and sid.isdigit()
        text = " ".join(text.split())
        if text:
            sentences[int(sid)] = text
    linked = set()
    for line in read(folder, f"{code}-nld_links.tsv.bz2"):
        a, b = (int(x) for x in line.split("\t"))
        if a in sentences and b in dutch:
            linked.add(a)
    return {sid: sentences[sid] for sid in linked if not blocked(sentences[sid])}


def main(folder: str) -> None:
    folder = Path(folder)
    dutch = dict(books.dutch_sentences())
    rows = ["lang\tid\ttext"]
    for short, code in LANGS.items():
        found = dutch if code == "nld" else pool(folder, code, dutch)
        first = sorted(found, key=lambda sid: hashlib.sha256(f"{code}\t{sid}".encode()).hexdigest())
        kept = sorted(first[:PER_LANGUAGE])
        rows += [f"{short}\t{sid}\t{found[sid]}" for sid in kept]
        print(f"{short} ({code}): {len(found):,} sentences to choose from, {len(kept)} kept")
    out = paths.ROOT / "data" / "tatoeba-langs.tsv.gz"
    with open(out, "wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as g:   # the same bytes every run
            g.write(("\n".join(rows) + "\n").encode("utf-8"))
    print(out, out.stat().st_size, "bytes", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main(sys.argv[1])
