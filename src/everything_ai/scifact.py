# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""SciFact, by David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan and Hannaneh
Hajishirzi (EMNLP 2020): 5,183 abstracts of biomedical papers, and claims that experts wrote from sentences in papers
that cite them. As a test collection for search, as BEIR (Thakur and others, 2021) uses it: a claim is a query, and
the abstracts that its sentence cites are its relevant abstracts: 809 claims of the training set and 300 of the test
set (SciFact's development set; its own test set has no answers in the open). The claims are under CC-BY-4.0 and the
abstracts under ODC-By-1.0, by LICENSE.md of allenai/scifact (course/route/bronnen.yaml, scifact). everything_ai.hub
downloads the four files of the copy of BEIR in mteb/scifact once, 8.2 MB, from a fixed commit, and keeps each only
when its SHA-256 is the pinned one. The files are read as text, one JSON object or one line of tabs at a time: nothing
in them is ever run."""
import functools
import json

from everything_ai import hub

REPO = "datasets/mteb/scifact"
FILES = [(REPO, "corpus.jsonl"), (REPO, "queries.jsonl"), (REPO, "qrels/train.tsv"), (REPO, "qrels/test.tsv")]
SPLITS = ("train", "test")
_MOST = {"corpus.jsonl": 16_000_000, "queries.jsonl": 500_000, "qrels/train.tsv": 100_000, "qrels/test.tsv": 100_000}


def ready() -> bool:
    """Whether all four files are downloaded."""
    return all(hub.pinned(*pin) is not None for pin in FILES)


def _read(name: str) -> str:
    path = hub.pinned(REPO, name)
    if path is None:
        raise FileNotFoundError("SciFact is not downloaded yet: everything_ai.hub.download(*pin) for each pin of "
                                "everything_ai.scifact.FILES")
    if path.stat().st_size > _MOST[name]:
        raise ValueError(f"the SciFact file {name} is larger than the course expects")
    return path.read_text(encoding="utf-8")


@functools.cache
def _collection() -> tuple:
    abstracts = []
    for line in _read("corpus.jsonl").split("\n"):
        if line.strip():
            row = json.loads(line)
            abstracts.append((str(row["_id"]), str(row["title"]), str(row["text"])))
    place = {a[0]: i for i, a in enumerate(abstracts)}
    texts = {}
    for line in _read("queries.jsonl").split("\n"):
        if line.strip():
            row = json.loads(line)
            texts[str(row["_id"])] = str(row["text"])
    claims = {}
    for split in SPLITS:
        relevant = {}
        for line in _read(f"qrels/{split}.tsv").split("\n")[1:]:   # the first line names the columns
            if line.strip():
                claim, abstract, score = line.split("\t")
                if int(score) > 0:
                    relevant.setdefault(claim, []).append(place[abstract])
        ordered = sorted(relevant.items(), key=lambda kv: int(kv[0]))
        claims[split] = tuple((c, texts[c], tuple(sorted(set(r)))) for c, r in ordered)
    return tuple(abstracts), claims


def abstracts() -> list:
    """Every abstract, in the order of the file, as a new list of dictionaries with "id", "title" and "text".
    FileNotFoundError when the files are not downloaded yet: hub.download(*pin) for each pin of FILES gets them."""
    return [{"id": i, "title": t, "text": x} for i, t, x in _collection()[0]]


def claims(split: str) -> list:
    """The claims of the training set ("train") or the test set ("test") that have relevant abstracts, by their number,
    as a new list of dictionaries with "id", "text" and "relevant": the places of the relevant abstracts in abstracts(),
    from small to large."""
    assert split in SPLITS, split
    return [{"id": c, "text": t, "relevant": list(r)} for c, t, r in _collection()[1][split]]
