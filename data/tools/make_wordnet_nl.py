# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/wordnet-nl.tsv: the Dutch lemmas of Open Dutch WordNet, each with the synset of Princeton WordNet 3.0
it belongs to, from the Open Multilingual Wordnet 2.0 (OMW 2.0, wns/nld/wn-data-nld.tab; CC-BY-SA-4.0). The lesson
nlp-dutch-compounds uses its lemmas as a list of Dutch words, and later lessons about word meanings join it with the
English synsets.

The words in DROP are left out wherever they occur: slurs and crude words. The course found them among the Dutch
lemmas on synsets whose English definition in Open English WordNet 2025 calls the term offensive, disparaging,
derogatory, obscene or vulgar, and by looking for such words by hand. Lemmas that share a synset with such a word but
are fine on their own (such as lid, neus and pot) stay. No list of this kind is complete. Download the file at the
tag below and run, from the course folder:

    uv run python data/tools/make_wordnet_nl.py wn-data-nld.tab
"""
import hashlib
import sys

from everything_ai import paths

SOURCE = "https://raw.githubusercontent.com/omwn/omw-data/v2.0/wns/nld/wn-data-nld.tab"   # tag v2.0
SHA256 = "ace21f3ea700f4ed8f4ac6dd9a4d12e2d732931ce766797afda1827330b08fa0"

DROP = {
    # slurs and words that put people down
    "Greaseball", "Jerry", "Spanjool", "Spik", "bastaard", "debiel", "flikker", "geteisem", "halfbloed", "imbeciel",
    "indianenvrouw", "japen", "kaaskop", "kankerjood", "kleurling", "koelie", "lesbi", "lesbo", "macaronivreter",
    "mietje", "mof", "mongool", "mulat", "neger", "negerin", "negerkindje", "nikker", "proleet", "roetmop", "sannie",
    "schorremorrie", "schuurmeid", "sjikse", "spaghettivreter", "spleetoog", "tribade", "verwijfd", "verwijfdheid",
    "zigeuner", "zigeunerin",
    # crude words
    "beffen", "bitch", "bullshit", "del", "derrie", "drek", "drol", "flamoes", "geil", "gelul", "hoer", "hoeren",
    "hoerenjong", "holtor", "hufter", "jongeheer", "kak", "klok-en-hamerspel", "klootzak", "kloten", "kont", "kut",
    "leuter", "lul", "memmen", "neuken", "piel", "piemel", "pik", "pis", "pissen", "pisser", "plasser", "poep",
    "poepen", "reet", "reetkever", "rotzak", "schijt", "shit", "slet", "smeerder", "snikkel", "sodemieter",
    "sodemieteren", "stront", "tamp", "tampeloeres", "tiet", "trut", "verdomme", "viezerik", "zeik", "zeiken",
    "zeikerd",
}


def main(tab: str) -> None:
    with open(tab, "rb") as f:
        data = f.read()
    assert hashlib.sha256(data).hexdigest() == SHA256, tab
    rows, seen = [], set()
    for line in data.decode("utf-8").splitlines():
        if line.startswith("#"):
            continue
        synset, kind, lemma = line.split("\t")
        assert kind == "nld:lemma", kind
        seen.add(lemma)
        if lemma not in DROP:
            rows.append((synset, lemma))
    missing = DROP - seen
    assert not missing, missing   # every word of DROP is in the file
    text = "synset\tlemma\n" + "".join(f"{s}\t{w}\n" for s, w in rows)
    (paths.ROOT / "data" / "wordnet-nl.tsv").write_text(text, encoding="utf-8")
    lemmas = {w for _, w in rows}
    simple = {w for w in lemmas if w.isalpha() and w.islower() and len(w) >= 3}
    print(len(rows), "rows,", len(lemmas), "lemmas,", len(simple), "of one word of 3 or more small letters,",
          len(text.encode()), "bytes")


if __name__ == "__main__":
    main(*sys.argv[1:])
