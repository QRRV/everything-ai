# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""The documents the lessons on RAG of the domain bouwen search in: the points of the summaries of the course's own
lessons, as data/course-summaries.json holds them (made by data/tools/make_course_summaries.py, a snapshot of the 415
lessons that were built then), and the questions the lessons ask about them. The texts are the course's own, under
CC BY-SA 4.0. The lessons search the English points, the language Qwen3 writes best; the Dutch ones are there for the
lessons that search across languages.

Next to the data, whether an answer says it cannot answer (`says_unknown`), and the course's own word search, the solutions of build-rag-first: `words` of a text, the number of
words a point shares with a question (`overlap`), and the `k` points that share the most (`search`); and BM25, the
solutions of build-bm25: the words of a text as a list (`word_list`), `idf`, the `bm25` score of one text, and
`bm25_search`."""
import functools
import json
import math
import re

from everything_ai import paths

DATA = paths.ROOT / "data" / "course-summaries.json"
PUNCTUATION = ".,:;!?()"   # what `words` takes off both ends of a word


@functools.cache
def _data() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


def points(lang: str = "en") -> list[str]:
    """The text of every point, in the order of the file: lesson by lesson, as their folders sort."""
    return [p["text"][lang] for p in _data()["points"]]


def about() -> list[dict]:
    """Of every point, in the same order: its id (lesson#i), its lesson, the lesson's title in both languages, and the
    domain, module and level of the lesson."""
    return [{k: p[k] for k in ("id", "lesson", "title", "domain", "module", "level")} for p in _data()["points"]]


def lessons() -> int:
    return _data()["lessons"]


def questions(kind: str | None = None) -> list[dict]:
    """The questions, or those of one kind: "words" (they share words with their point), "meaning" (they ask the same
    in other words) and "none" (no point answers them). Each has its id, kind, the id of the point that answers it
    ("gold", None for kind "none"), a regular expression a right answer matches in lower case ("answer"), and its
    text in both languages."""
    return [q for q in _data()["questions"] if kind is None or q["kind"] == kind]


def text_of(point_id: str, lang: str = "en") -> str:
    return next(p["text"][lang] for p in _data()["points"] if p["id"] == point_id)


def right(answer: str, question: dict) -> bool:
    """Whether an answer gives what the question's point says: it matches the question's pattern."""
    return question["answer"] is not None and re.search(question["answer"], answer.lower()) is not None


# what an answer says when it does not give one: the sources or what it knows do not answer the question, or it
# depends; measured on the course's recorded answers of Qwen3 (data/course-rag-qwen.json), not a general test
_UNKNOWN = re.compile(r"not (directly |explicitly )?(mention|provide|address|answer|contain|includ|state|say|specif)|no (direct )?(mention|information)"
                      r"|unrelated|cannot|can't|not possible|need to know|without knowing|don't have|not clear|unclear|depends on")


def says_unknown(answer: str) -> bool:
    """Whether an answer says in some way that it cannot answer: the sources do not say, or it depends."""
    return _UNKNOWN.search(answer.lower()) is not None


def words(text: str) -> set:
    """The words of a text: in lower case, split at spaces, with punctuation taken off both ends of each word."""
    return set([word.strip(PUNCTUATION) for word in text.lower().split()])


def overlap(question: str, text: str) -> int:
    """How many words a text shares with the question."""
    return len(words(question) & words(text))


def search(question: str, texts: list, k: int) -> list:
    """The k texts that share the most words with the question, most first; texts that share as many come in reverse
    alphabetical order, as sorting pairs gives them."""
    asked = words(question)
    found = []
    for text in texts:
        found.append((len(asked & words(text)), text))
    return [text for shared, text in sorted(found, reverse=True)[:k]]


# --- BM25 ----------------------------------------------------------------------------------------------------------------

K1 = 1.2    # how fast a word that comes back stops adding: the usual setting, as in Lucene and Elasticsearch
B = 0.75    # how much a long text is corrected for: 0 not at all, 1 fully


def word_list(text: str) -> list:
    """The words of a text as `words` makes them, but as a list, every time a word appears."""
    return [word.strip(PUNCTUATION) for word in text.lower().split()]


def idf(n: int, total: int) -> float:
    """BM25's inverse document frequency of a word that `n` of `total` texts have: high for a rare word, near 0 for a
    word almost every text has, and never below 0."""
    return math.log((total - n + 0.5) / (n + 0.5) + 1)


def bm25(asked, words: list, idfs: dict, average: float) -> float:
    """The BM25 score of a text, its `words` as a list, for the words of a question (`asked`); `idfs` gives the idf of
    every asked word, `average` the average number of words of a text."""
    score = 0
    for word in asked:
        f = words.count(word)
        score = score + idfs[word] * f * (K1 + 1) / (f + K1 * (1 - B + B * len(words) / average))
    return score


def bm25_index(texts: list) -> dict:
    """What BM25 needs of a list of texts: the words of each text as a list, the idf of every word in them, and the
    average number of words of a text."""
    lists = [word_list(t) for t in texts]
    counts: dict = {}
    for ws in lists:
        for w in set(ws):
            counts[w] = counts.get(w, 0) + 1
    return {"words": lists, "idf": {w: idf(n, len(texts)) for w, n in counts.items()},
            "average": sum(len(ws) for ws in lists) / len(lists)}


def bm25_search(question: str, texts: list, k: int, index: dict | None = None) -> list:
    """The k texts with the highest BM25 score for the question, highest first; texts with the same score in reverse
    alphabetical order, as `search` gives them. A word of the question that no text has adds nothing. The words of the
    question go in alphabetical order, so that the sum is the same on every run."""
    index = bm25_index(texts) if index is None else index
    asked = sorted(words(question))
    idfs = {w: index["idf"].get(w, 0.0) for w in asked}
    found = [(bm25(asked, ws, idfs, index["average"]), t) for ws, t in zip(index["words"], texts)]
    return [t for score, t in sorted(found, reverse=True)[:k]]
