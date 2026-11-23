# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/litbank-coref.json.gz: the coreference layer of LitBank (Bamman, Lewke and Mansoor, LREC 2020;
CC-BY-4.0), for the lesson of domain taal about coreference (read with everything_ai.litbank).

LitBank holds about 2,000 words from each of 100 works of English fiction, from 1719 to 1922. Its coreference layer
marks every mention of a person, facility, location, geo-political entity, organization or vehicle, and every
personal pronoun that refers to one, and groups the mentions of one entity into a chain. The course keeps, for each
work, its Gutenberg id, year, author and title, its sentences as LitBank tokenized them, and its mentions: sentence,
first and last token, entity type (PER, FAC, LOC, GPE, ORG, VEH), category (PROP for a name, NOM for a description,
PRON for a pronoun) and chain, numbered per work in the order of the first mention. Mentions that LitBank links only
by apposition or copula, without a chain of their own, go. `shown` is 0 for the sentences that a lesson does not show
as an example: those with a crude, sexual or violent word, a slur, drink or a weapon (WORDS). They stay in the file;
no such filter is complete. The sentences are unchanged.

LitBank comes from GitHub at a fixed commit. From the course folder:

    git clone https://github.com/dbamman/litbank <folder>
    git -C <folder> checkout 3e50db0ffc033d7ccbb94f4d88f6b99210328ed8
    uv run python data/tools/make_litbank_coref.py <folder>
"""
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

COMMIT = "3e50db0ffc033d7ccbb94f4d88f6b99210328ed8"
# the SHA-256 of the SHA-256 lines of the 200 files of coref/tsv, sorted by name
SHA256 = "74054c3d1d89f8fcb6e9543ad247e312c28ee0a5f79a45be780ace0612d80db2"
OUT = Path(__file__).resolve().parents[2] / "data" / "litbank-coref.json.gz"
TYPES = ("PER", "FAC", "LOC", "GPE", "ORG", "VEH")
CATEGORIES = ("PROP", "NOM", "PRON")
# words that keep a sentence from being shown as an example
WORDS = re.compile(r"\b(fuck\w*|shit\w*|bitch\w*|damn\w*|ass|asses|asshole\w*|bastard\w*|crap\w*|dick|dicks|piss\w*|"
                   r"cunt\w*|whore\w*|harlot\w*|slut\w*|fag\w*|nigg\w*|negro\w*|darkey\w*|darky|injun\w*|savage\w*|"
                   r"squaw\w*|coolie\w*|chink\w*|jap|japs|gypsy|gypsies|retard\w*|idiot\w*|sex|sexual\w*|naked|"
                   r"nude\w*|rape\w*|ravish\w*|seduc\w*|kill|kills|killed|killing\w*|killer\w*|murder\w*|slay\w*|"
                   r"slain|dead|death\w*|die|died|dies|dying|corpse\w*|blood\w*|bleed\w*|wound\w*|gun|guns|pistol\w*|"
                   r"rifle\w*|musket\w*|sword\w*|knife|knives|dagger\w*|weapon\w*|bomb\w*|hang|hanged|gallows|"
                   r"suicid\w*|torture\w*|whip\w*|lash\w*|flog\w*|beat|beaten|slave\w*|drunk\w*|beer\w*|wine|gin|"
                   r"rum|brandy|whisk\w*|liquor\w*|alcohol\w*|opium|cigar\w*|smok\w*|tobacco)\b", re.IGNORECASE)
_ROW = re.compile(r"^\|(\d+)\|(\d{4})\|([^|]+)\|([^|]+)\|$")


def digest(folder: Path) -> str:
    """The SHA-256 of the SHA-256 lines of the files of coref/tsv, sorted by name."""
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n"
             for p in sorted((folder / "coref" / "tsv").iterdir()) if p.is_file()]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def works(folder: Path) -> dict:
    """Gutenberg id -> (year, author, title), from the table in LitBank's README."""
    out = {}
    for line in (folder / "README.md").read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line.strip())
        if m:
            out[int(m.group(1))] = (int(m.group(2)), m.group(3).strip(), m.group(4).strip())
    return out


def document(ann: Path, txt: Path) -> tuple:
    """The sentences (each a string of tokens with spaces between) and the mentions of one work, each mention
    [sentence, first token, last token, type, category, chain], in the order of the text."""
    sentences = txt.read_text(encoding="utf-8").rstrip("\n").split("\n")
    found, chain_of = {}, {}
    for line in ann.read_text(encoding="utf-8").splitlines():
        part = line.split("\t")
        if part[0] == "MENTION":
            key, s, first, s_end, last, text, typ, cat = part[1:9]
            assert s == s_end, line                       # no mention runs over two sentences
            assert typ in TYPES and cat in CATEGORIES, line
            tokens = sentences[int(s)].split(" ")[int(first):int(last) + 1]
            assert " ".join(tokens) == text, (line, tokens)
            found[key] = (int(s), int(first), int(last), typ, cat)
        elif part[0] == "COREF":
            chain_of[part[1]] = part[2]
    keys = sorted((k for k in found if k in chain_of), key=lambda k: (found[k][0], found[k][1], -found[k][2]))
    number = {}
    for k in keys:
        number.setdefault(chain_of[k], len(number))
    return sentences, [[*found[k], number[chain_of[k]]] for k in keys]


def main(folder: Path) -> None:
    got = digest(folder)
    assert got == SHA256, f"{folder} is not LitBank at commit {COMMIT} (digest {got})"
    meta = works(folder)
    docs = []
    for ann in sorted((folder / "coref" / "tsv").glob("*.ann")):
        gutenberg = int(ann.name.split("_", 1)[0])
        sentences, mentions = document(ann, ann.with_suffix(".txt"))
        year, author, title = meta[gutenberg]
        docs.append({"gutenberg": gutenberg, "year": year, "author": author, "title": title,
                     "sentences": sentences, "shown": [0 if WORDS.search(s) else 1 for s in sentences],
                     "mentions": mentions})
    docs.sort(key=lambda d: (d["year"], d["gutenberg"]))
    data = json.dumps({"commit": COMMIT, "docs": docs}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    OUT.write_bytes(gzip.compress(data, compresslevel=9, mtime=0))
    n_s = sum(len(d["sentences"]) for d in docs)
    print(f"{OUT}: {len(docs)} works, {n_s} sentences ({n_s - sum(sum(d['shown']) for d in docs)} not shown), "
          f"{sum(len(d['mentions']) for d in docs)} mentions, SHA-256 {hashlib.sha256(OUT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
