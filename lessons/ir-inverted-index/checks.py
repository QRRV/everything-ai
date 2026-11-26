# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ir-inverted-index: the inverted index, a dictionary from each word to its postings list, the
sorted numbers of the documents that hold the word (build_index); the walk along two sorted postings lists that keeps
the numbers both hold (intersect), as in Manning, Raghavan and Schütze, Introduction to Information Retrieval (2008),
section 1.3; and a search with AND over every word of a query, the shortest list first (search_and). The sort asks which
queries this index answers, and which need more than the document numbers.

The small collection is the thirteen documents of the lesson ir-what-is-search. The large one is the 2,067 paragraphs of
SQuAD's development set in English and in SQuAD-NL in Dutch (everything_ai.xquad), cut into words as the lesson
ir-what-is-search cuts them. Every number in the text and the pictures is computed by the course
(tests/test_ir_inverted_index.py computes them again)."""
import functools
import html
import importlib.util
import re

from everything_ai import paths, xquad, yours
from everything_ai.check import FunctionBox, Mistake, Sort

LANGS = ("en", "nl")


def _what_is_search():
    spec = importlib.util.spec_from_file_location("everything_ai_checks_ir_what_is_search_for_inverted_index",
                                                  paths.LESSONS / "ir-what-is-search" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W = _what_is_search()   # the collection of thirteen documents, its queries and the way it cuts words
S = W.S                 # the parts of the pictures and the exercises

TEXTS = W.TEXTS         # the thirteen documents, in both languages in the same order
DOCS = W.DOCS           # the same documents as lists of words
QUERIES = W.QUERIES     # the queries python and red wine heart, as lists of words
words = W.words         # a text as a list of words in small letters


# --- the course's solutions, the same code as the exercises -------------------------------------------------------------------

def build_index(docs):
    """The inverted index of a list of documents, each a list of words: for each word, the numbers of the documents
    that hold it, in order and each number once."""
    index = {}
    for doc in range(len(docs)):
        for word in docs[doc]:
            if word not in index:
                index[word] = []
            if doc not in index[word]:
                index[word].append(doc)
    return index


def intersect(p1, p2):
    """The numbers that two sorted postings lists both hold, found by walking both lists once."""
    answer = []
    i = 0
    j = 0
    while i < len(p1) and j < len(p2):
        if p1[i] == p2[j]:
            answer.append(p1[i])
            i = i + 1
            j = j + 1
        elif p1[i] < p2[j]:
            i = i + 1
        else:
            j = j + 1
    return answer


def search_and(query, index):
    """The numbers of the documents that hold every word of the query: the postings lists of its words walked two at a
    time, the shortest first."""
    lists = []
    for word in query:
        if word not in index:
            return []
        lists.append(index[word])
    lists = sorted(lists, key=len)
    answer = lists[0]
    for postings in lists[1:]:
        answer = intersect(answer, postings)
    return answer


SMALL_INDEX = {lang: build_index(DOCS[lang]) for lang in LANGS}   # the index of the thirteen documents


def walk(p1, p2) -> list:
    """The walk of intersect, step by step: (place in p1, place in p2, what happened, the answer after the step), with
    what happened one of "same", "first" (the number in p1 is the smaller, so the walk moves on in p1) and "second"."""
    steps, answer, i, j = [], [], 0, 0
    while i < len(p1) and j < len(p2):
        if p1[i] == p2[j]:
            answer = answer + [p1[i]]
            steps.append((i, j, "same", answer))
            i, j = i + 1, j + 1
        elif p1[i] < p2[j]:
            steps.append((i, j, "first", answer))
            i += 1
        else:
            steps.append((i, j, "second", answer))
            j += 1
    return steps


def plan(query, index) -> list:
    """How search_and answers a query: the words, the shortest list first, each with the length of its list and the
    number of documents left after its walk, and the comparisons of that walk. Empty when a word is in no document."""
    if not query or any(w not in index for w in query):
        return []
    order = sorted(query, key=lambda w: len(index[w]))
    rows, answer = [(order[0], len(index[order[0]]), len(index[order[0]]), 0)], index[order[0]]
    for w in order[1:]:
        steps = len(walk(answer, index[w]))
        answer = intersect(answer, index[w])
        rows.append((w, len(index[w]), len(answer), steps))
    return rows


# --- the paragraphs ------------------------------------------------------------------------------------------------------------

@functools.cache
def _paragraph_words(lang: str) -> tuple:
    return tuple(tuple(words(p.text)) for p in xquad.paragraphs(lang))


def paragraph_words(lang: str) -> list:
    """The 2,067 paragraphs in English (en) or Dutch (nl), each as a list of words, cut as the course cuts them."""
    assert lang in LANGS, lang
    return [list(p) for p in _paragraph_words(lang)]


@functools.cache
def _paragraph_index(lang: str) -> dict:
    """The same index as build_index(paragraph_words(lang)), built faster: the documents come in order, so a document
    is already in the list of a word when it is the last number there."""
    index = {}
    for doc, ws in enumerate(_paragraph_words(lang)):
        for w in ws:
            postings = index.setdefault(w, [])
            if not postings or postings[-1] != doc:
                postings.append(doc)
    return index


def paragraph_index(lang: str) -> dict:
    """The inverted index of the 2,067 paragraphs in English (en) or Dutch (nl): a new copy each time."""
    assert lang in LANGS, lang
    return {w: list(p) for w, p in _paragraph_index(lang).items()}


@functools.cache
def numbers(lang: str) -> dict:
    """What the lesson says about the paragraphs: how many there are, how many words they hold and how many different
    ones, the 1s and the cells of their table of words and paragraphs, and the words in one paragraph only."""
    docs = _paragraph_words(lang)
    index = _paragraph_index(lang)
    ones = sum(len(p) for p in index.values())
    cells = len(docs) * len(index)
    return {"paragraphs": len(docs), "words": sum(len(d) for d in docs), "different": len(index), "ones": ones,
            "cells": cells, "zeros": 1 - ones / cells, "once": sum(1 for p in index.values() if len(p) == 1),
            "per_paragraph": ones / len(docs)}


# The pairs of words of the picture under intersect, and the queries of the picture under search_and.
PAIRS = {"en": [("rhine", "bridge"), ("oxygen", "water")], "nl": [("rijn", "delta"), ("zuurstof", "water")]}
BIG_QUERIES = {"en": [["steam", "engine", "water"], ["rhine", "river", "bridge"], ["normans", "england", "king"],
                      ["oxygen", "water", "fire"]],
               "nl": [["stoom", "machine", "water"], ["amazone", "rivier", "regenwoud"], ["normandiërs", "engeland", "koning"],
                      ["zuurstof", "water", "vuur"]]}


# --- the pictures ----------------------------------------------------------------------------------------------------------

_UI, _LABEL, _CELL, _HEAD, _NUMBER = S._UI, S._LABEL, S._CELL, S._HEAD, S._NUMBER
_DATA, _FAINT, _PRED, _LOSS = S._DATA, S._FAINT, S._PRED, S._LOSS
_CHIP, _CHIP_PRED, _CHIP_DATA, _CHIP_LOSS = S._CHIP, S._CHIP_PRED, S._CHIP_DATA, S._CHIP_LOSS
_note, _colors, _label, _num, _pct = S._note, S._colors, S._label, S._num, S._pct
_cell, _table = S._cell, S._table
_fn, _call, _empty, _SLOW, _STOPPED = S._fn, S._call, S._empty, S._SLOW, S._STOPPED
_WHY_EMPTY = S._WHY_EMPTY
_MONO = "font-family:var(--ea-font-code);font-size:0.8rem"
_DASH = f'<span style="{_UI}">–</span>'
_SHOWN_RESULTS = 6   # results that the demo shows with their paragraph
_SHOWN_NUMBERS = 8   # document numbers of a result that the picture under search_and shows
_PARAGRAPH_CAP = 160  # characters of a paragraph that the demo shows
_HIDDEN = {"en": "(a paragraph the course does not show)", "nl": "(een alinea die de course niet laat zien)"}
# a paragraph with a pronoun that names a person by gender (in Dutch also the words that can be one, such as zijn)
# stays out of a picture, as the course's own writing does
_PRONOUN = {"en": re.compile(r"\b(he|she|him|her|his|hers|himself|herself)\b", flags=re.I),
            "nl": re.compile(r"\b(hij|zij|ze|haar|hem|hun|hen|zijn|z'n|diens|wiens)\b", flags=re.I)}


def showable(i: int, lang: str) -> bool:
    p = xquad.paragraphs(lang)[i]
    return p.shown and _PRONOUN[lang].search(p.text) is None


def _chip(text_: str, style: str = _CHIP_DATA) -> str:
    return f'<span style="{style}">{html.escape(str(text_))}</span>'


def _chips(numbers_, style: str = _CHIP_DATA) -> str:
    return "".join(_chip(n, style) for n in numbers_) or _DASH


def _code(text_: str) -> str:
    return f'<code style="{_MONO}">{html.escape(text_)}</code>'


def _query_text(query) -> str:
    return " ".join(query)


class _Odd(Exception):
    """A learner's function gave something a picture cannot use."""


def _is_number(d, top: int) -> bool:
    return isinstance(d, int) and not isinstance(d, bool) and 0 <= d < top


def _guarded(name: str, fn, top: int):
    """The learner's function, stopping with _Odd when it gives what the picture cannot use: document numbers below
    `top`."""
    if name == "build_index":
        def step(docs):
            out = fn([list(d) for d in docs])
            if not (isinstance(out, dict) and all(isinstance(k, str) for k in out)
                    and all(isinstance(v, (list, tuple)) and all(_is_number(d, top) for d in v) for v in out.values())):
                raise _Odd
            return {k: list(v) for k, v in out.items()}
    elif name == "intersect":
        def step(p1, p2):
            out = fn(list(p1), list(p2))
            if not (isinstance(out, (list, tuple)) and all(_is_number(d, top) for d in out)):
                raise _Odd
            return list(out)
    else:
        def step(query, index):
            # only the lists of the words of the query, each a copy: a function that changes a list cannot harm the index
            small = {w: list(index[w]) for w in query if w in index}
            out = fn(list(query), small)
            if not (isinstance(out, (list, tuple)) and all(_is_number(d, top) for d in out)):
                raise _Odd
            return list(out)
    return step


_IDEA_TEXT = {
    "en": {"title": "The index of the thirteen documents, and a walk along two lists",
           "matrix": "A table with a row for each word of the query *{q}* and a column for each document: 1 where the document holds the word.",
           "index": "The inverted index keeps only the 1s: for each word, the numbers of the documents that hold it, its postings list.",
           "head_index": ["Word", "Postings list", "Length"],
           "walk": "The walk along the two shortest lists, *{a}* and *{b}*: a finger in each list, and at each step the finger at the smaller number moves on.",
           "head_walk": ["Step", "*{a}*", "*{b}*", "What happens", "Answer so far"],
           "same": "the same number: keep {n}, move both fingers",
           "first": "{n} is smaller: move the finger in *{w}*",
           "second": "{n} is smaller: move the finger in *{w}*",
           "end": "The list of *{w}* has run out, so the walk stops: no later number can be in both lists.",
           "then": "Then the answer so far and the list of *{w}* are walked together, in {k} steps.",
           "answer": "The documents with every word of *{q}*:",
           "summary": "The index answers *{q}* with {c} comparisons of two numbers, and reads none of the {n} words of the documents.",
           "gray": "Gray: the documents and their postings lists.",
           "blue": "Blue: the answer, the documents that hold every word so far."},
    "nl": {"title": "De index van de dertien documenten, en een wandeling langs twee lijsten",
           "matrix": "Een tabel met een rij voor elk woord van de zoekvraag *{q}* en een kolom voor elk document: een 1 waar het document het woord bevat.",
           "index": "De inverted index bewaart alleen de enen: voor elk woord de nummers van de documenten met dat woord, de postings list van het woord.",
           "head_index": ["Woord", "Postings list", "Lengte"],
           "walk": "De wandeling langs de twee kortste lijsten, *{a}* en *{b}*: een vinger in elke lijst, en bij elke stap schuift de vinger bij het kleinste getal door.",
           "head_walk": ["Stap", "*{a}*", "*{b}*", "Wat er gebeurt", "Antwoord tot nu toe"],
           "same": "hetzelfde getal: houd {n}, schuif beide vingers door",
           "first": "{n} is kleiner: schuif de vinger in *{w}* door",
           "second": "{n} is kleiner: schuif de vinger in *{w}* door",
           "end": "De lijst van *{w}* is op, dus de wandeling stopt: geen later getal kan in beide lijsten staan.",
           "then": "Daarna gaat de wandeling langs het antwoord tot nu toe en de lijst van *{w}*, in {k} stappen.",
           "answer": "De documenten met alle woorden van *{q}*:",
           "summary": "De index beantwoordt *{q}* met {c} vergelijkingen van twee getallen, en leest geen van de {n} woorden van de documenten.",
           "gray": "Grijs: de documenten en de postings lists.",
           "blue": "Blauw: het antwoord, de documenten die tot nu toe alle woorden bevatten."},
}


def _head(names) -> list:
    """Table heads may hold a word in italics: the table writes them plain."""
    return [n.replace("*", "") for n in names]


def idea_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang]
    query = QUERIES[lang]["wine"]
    index = SMALL_INDEX[lang]
    q = _query_text(query)
    n_docs = len(DOCS[lang])
    # the table of 0s and 1s
    rows = []
    for word in query:
        cells = "".join(_cell("1" if d in index[word] else "0", _DATA if d in index[word] else _FAINT, _NUMBER)
                        for d in range(n_docs))
        rows.append("<tr>" + _cell(_code(word)) + cells + "</tr>")
    body = _label(w["matrix"].format(q=q)) + _table(rows, [""] + [str(d) for d in range(n_docs)],
                                                    tuple(range(1, n_docs + 1)))
    # the index
    rows = ["<tr>" + _cell(_code(word)) + _cell(_chips(index[word])) + _cell(str(len(index[word])), _FAINT, _NUMBER)
            + "</tr>" for word in query]
    body += _label(w["index"]) + _table(rows, w["head_index"], (2,))
    # the walk along the two shortest lists, then the rest
    order = sorted(query, key=lambda x: len(index[x]))
    a, b = order[0], order[1]
    p1, p2 = index[a], index[b]
    rows = []
    steps = walk(p1, p2)
    for k, (i, j, what, answer) in enumerate(steps, 1):
        if what == "same":
            said = w["same"].format(n=p1[i])
        elif what == "first":
            said = w["first"].format(n=p1[i], w=a)
        else:
            said = w["second"].format(n=p2[j], w=b)
        rows.append("<tr>" + _cell(str(k), _FAINT, _NUMBER) + _cell(_chip(p1[i])) + _cell(_chip(p2[j]))
                    + _cell(S._marked(said), _FAINT) + _cell(_chips(answer, _CHIP_PRED)) + "</tr>")
    head = [h.format(a=a, b=b) for h in w["head_walk"]]
    body += _label(w["walk"].format(a=a, b=b)) + _table(rows, _head(head), (0,))
    i, j = steps[-1][0], steps[-1][1]
    if steps[-1][2] != "same":
        i, j = (i + 1, j) if steps[-1][2] == "first" else (i, j + 1)
    else:
        i, j = i + 1, j + 1
    body += _note(w["end"].format(w=a if i >= len(p1) else b))
    answer, comparisons = intersect(p1, p2), len(steps)
    for word in order[2:]:
        k = len(walk(answer, index[word]))
        comparisons += k
        answer = intersect(answer, index[word])
        body += _note(w["then"].format(w=word, k=_num(k, lang)))
    body += _label(w["answer"].format(q=q)) + f"<div>{_chips(answer, _CHIP_PRED)}</div>"
    n_words = sum(len(d) for d in DOCS[lang])
    body += _note(w["summary"].format(q=q, c=_num(comparisons, lang), n=_num(n_words, lang)))
    body += _colors(w, body)
    return yours.figure(w["title"], body, alt=w["title"])


_INDEX_TEXT = {
    "en": {"title": "Your build_index on the course's thirteen documents",
           "kind": "a dictionary from words to lists of document numbers",
           "head": ["Word", "Your postings list", "The course's"],
           "summary": "Your index holds {k} words, and the course's {n}. Words whose postings list differs from the course's: {d}.",
           "missing": "not in your index",
           "gray": "Gray: the course's postings list, where yours differs.",
           "blue": "Blue: a postings list of your index that is the same as the course's.",
           "purple": "Purple: a postings list of your index that differs from the course's."},
    "nl": {"title": "Je build_index op de dertien documenten van de course",
           "kind": "een dictionary van woorden naar lijsten met documentnummers",
           "head": ["Woord", "Jouw postings list", "Lijst van de course"],
           "summary": "Je index bevat {k} woorden, en de index van de course {n}. Woorden met een andere postings list dan bij de course: {d}.",
           "missing": "niet in je index",
           "gray": "Grijs: de postings list van de course, waar de jouwe verschilt.",
           "blue": "Blauw: een postings list van je index, gelijk aan de lijst van de course.",
           "purple": "Paars: een postings list van je index, anders dan de lijst van de course."},
}
_INDEX_MORE = 6   # words with another postings list than the course's that the picture shows


def index_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _INDEX_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    step = _guarded("build_index", fn, len(DOCS[lang]))

    def make():
        try:
            return step(DOCS[lang])
        except _Odd:
            return None

    got = _call(make)
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    if got is None:
        return _empty(w, "odd")
    want = SMALL_INDEX[lang]
    differs = [x for x in list(want) + [x for x in got if x not in want] if got.get(x) != want.get(x)]
    shown = list(QUERIES[lang]["wine"]) + ["python"]
    shown += [x for x in differs if x not in shown][:_INDEX_MORE]
    rows = []
    for word in shown:
        mine, course = got.get(word), want.get(word)
        if mine is None:
            yours_ = f'<span style="{_LOSS}">{html.escape(w["missing"])}</span>'
        else:
            yours_ = _chips(mine, _CHIP_PRED if mine == course else _CHIP_LOSS)
        theirs = _DASH if mine == course else (_chips(course) if course is not None else _DASH)
        rows.append("<tr>" + _cell(_code(word)) + _cell(yours_) + _cell(theirs) + "</tr>")
    body = _note(w["summary"].format(k=_num(len(got), lang), n=_num(len(want), lang), d=_num(len(differs), lang)))
    body += _table(rows, w["head"])
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


_INTERSECT_TEXT = {
    "en": {"title": "Your intersect on postings lists of the 2,067 paragraphs",
           "kind": "a list of document numbers",
           "pair": "*{a}* AND *{b}*: lists of {x} and {y} paragraphs",
           "yours": "Your intersect:",
           "course": "The course's intersect gives: {want}.",
           "cost": "The walk looks at no more than {x} + {y} = {s} numbers; reading the paragraphs themselves means reading all {n} words.",
           "same": "Your intersect gives what the course's gives for both pairs.",
           "differs": "Your intersect differs from the course's for {k} of the two pairs.",
           "gray": "Gray: the postings lists of the two words.",
           "blue": "Blue: a paragraph that your intersect keeps, and that both lists hold.",
           "purple": "Purple: a paragraph that your intersect keeps but that is not in both lists, and what the course's intersect gives where yours differs."},
    "nl": {"title": "Je intersect op postings lists van de 2.067 alinea's",
           "kind": "een lijst met documentnummers",
           "pair": "*{a}* AND *{b}*: lijsten van {x} en {y} alinea's",
           "yours": "Je intersect:",
           "course": "De intersect van de course geeft: {want}.",
           "cost": "De wandeling bekijkt hooguit {x} + {y} = {s} getallen; de alinea's zelf lezen betekent alle {n} woorden lezen.",
           "same": "Je intersect geeft bij beide paren hetzelfde als de intersect van de course.",
           "differs": "Bij {k} van de twee paren geeft je intersect iets anders dan de intersect van de course.",
           "gray": "Grijs: de postings lists van de twee woorden.",
           "blue": "Blauw: een alinea in beide lijsten, en ook in het antwoord van je intersect.",
           "purple": "Paars: een alinea in het antwoord van je intersect, maar niet in beide lijsten, en wat de intersect van de course geeft waar de jouwe verschilt."},
}


def intersect_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _INTERSECT_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    index = _paragraph_index(lang)
    step = _guarded("intersect", fn, numbers(lang)["paragraphs"])

    def make():
        try:
            return [step(index[a], index[b]) for a, b in PAIRS[lang]]
        except _Odd:
            return None

    got = _call(make)
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    if got is None:
        return _empty(w, "odd")
    body, differs = "", 0
    for (a, b), mine in zip(PAIRS[lang], got):
        p1, p2 = index[a], index[b]
        want = intersect(p1, p2)

        def marked(postings):
            return "".join(_chip(d, (_CHIP_PRED if d in want else _CHIP_LOSS) if d in mine else _CHIP_DATA)
                           for d in postings)
        body += _label(w["pair"].format(a=a, b=b, x=_num(len(p1), lang), y=_num(len(p2), lang)))
        rows = ["<tr>" + _cell(_code(word)) + _cell(marked(p)) + "</tr>" for word, p in ((a, p1), (b, p2))]
        rows.append("<tr>" + _cell(html.escape(w["yours"]), _FAINT)
                    + _cell("".join(_chip(d, _CHIP_PRED if d in want else _CHIP_LOSS) for d in mine) or _DASH) + "</tr>")
        body += _table(rows, ["", ""])
        if mine != want:
            differs += 1
            body += f'<div style="{_LOSS}">' + html.escape(w["course"].format(want=want)) + "</div>"
        body += _note(w["cost"].format(x=_num(len(p1), lang), y=_num(len(p2), lang), s=_num(len(p1) + len(p2), lang),
                                       n=_num(numbers(lang)["words"], lang)))
    body += _note(w["same"] if not differs else w["differs"].format(k=_num(differs, lang)))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


_SEARCH_TEXT = {
    "en": {"title": "Your search_and on the 2,067 paragraphs",
           "kind": "a list of document numbers",
           "head": ["Query", "Postings lists, shortest first", "Your results", "The course's"],
           "results": "{k}: {shown}",
           "none": "none",
           "summary": "Your search_and gives the same paragraphs as the course's for {k} of the {n} queries.",
           "gray": "Gray: the queries, the lengths of their postings lists, and the course's results.",
           "blue": "Blue: your results, where they are the course's.",
           "purple": "Purple: your results, where they differ."},
    "nl": {"title": "Je search_and op de 2.067 alinea's",
           "kind": "een lijst met documentnummers",
           "head": ["Zoekvraag", "Postings lists, de kortste eerst", "Jouw resultaten", "Resultaten van de course"],
           "results": "{k}: {shown}",
           "none": "geen",
           "summary": "Je search_and geeft bij {k} van de {n} zoekvragen dezelfde alinea's als de search_and van de course.",
           "gray": "Grijs: de zoekvragen, de lengtes van de postings lists, en de resultaten van de course.",
           "blue": "Blauw: je resultaten, waar deze gelijk zijn aan de resultaten van de course.",
           "purple": "Paars: je resultaten, waar deze verschillen."},
}


def _results(found, lang: str, w: dict, style: str) -> str:
    if not found:
        return html.escape(w["none"])
    more = "…" if len(found) > _SHOWN_NUMBERS else ""
    return html.escape(_num(len(found), lang)) + ": " + _chips(found[:_SHOWN_NUMBERS], style) + more


def search_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _SEARCH_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    index = _paragraph_index(lang)
    step = _guarded("search_and", fn, numbers(lang)["paragraphs"])

    def make():
        try:
            return [step(q, index) for q in BIG_QUERIES[lang]]
        except _Odd:
            return None

    got = _call(make)
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    if got is None:
        return _empty(w, "odd")
    rows, same = [], 0
    for q, mine in zip(BIG_QUERIES[lang], got):
        want = search_and(q, index)
        same += mine == want
        lists = ", ".join(f"{x} {_num(len(index[x]), lang)}" for x in sorted(q, key=lambda x: len(index[x])))
        rows.append("<tr>" + _cell(_code(_query_text(q))) + _cell(html.escape(lists), _DATA)
                    + _cell(_results(mine, lang, w, _CHIP_PRED if mine == want else _CHIP_LOSS),
                            _PRED if mine == want else _LOSS)
                    + _cell(_results(want, lang, w, _CHIP_DATA), _DATA) + "</tr>")
    body = _note(w["summary"].format(k=_num(same, lang), n=_num(len(got), lang))) + _table(rows, w["head"])
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


_TRY_TEXT = {
    "en": {"title": "Your query on the 2,067 paragraphs",
           "empty": "Type one or more words in the box, and the picture shows how the index finds the paragraphs that hold all of them.",
           "query": "Your query, as a list of words: {q}",
           "missing": "*{x}* is in none of the paragraphs, so no paragraph holds every word of your query: the index knows it without a single walk.",
           "head": ["Word, shortest list first", "Document frequency", "Paragraphs left", "Comparisons"],
           "summary": "{k} of the {n} paragraphs {verb} every word of your query. The walks compared {c} pairs of numbers; reading the paragraphs themselves means reading all {m} words.",
           "one": "holds", "many": "hold",
           "shown": "The first results:",
           "more": "The search uses only the first {n} words of your query.",
           "gray": "Gray: the words of your query, their postings lists, and the paragraphs.",
           "blue": "Blue: the paragraphs that hold every word of your query."},
    "nl": {"title": "Je zoekvraag op de 2.067 alinea's",
           "empty": "Typ een of meer woorden in het vak, dan laat het plaatje zien hoe de index de alinea's vindt die elk van deze woorden bevatten.",
           "query": "Je zoekvraag, als lijst woorden: {q}",
           "missing": "*{x}* staat in geen enkele alinea, dus geen alinea bevat alle woorden van je zoekvraag: de index weet dat al zonder te wandelen.",
           "head": ["Woord, kortste lijst eerst", "Document frequency", "Alinea's over", "Vergelijkingen"],
           "summary": "{k} van de {n} alinea's {verb} alle woorden van je zoekvraag. De wandelingen vergeleken {c} paren getallen; de alinea's zelf lezen betekent alle {m} woorden lezen.",
           "one": "bevat", "many": "bevatten",
           "shown": "De eerste resultaten:",
           "more": "De zoekmachine gebruikt alleen de eerste {n} woorden van je zoekvraag.",
           "gray": "Grijs: de woorden van je zoekvraag, de postings lists van deze woorden, en de alinea's.",
           "blue": "Blauw: de alinea's die alle woorden van je zoekvraag bevatten."},
}
TRY_WORDS = 8       # words of the try-it box that the search reads
TRY_CHARS = 300     # characters of the box that the search reads
TRY_START = {"en": "rhine bridge", "nl": "rijn delta"}


def _paragraph_row(i: int, lang: str) -> str:
    p = xquad.paragraphs(lang)[i]
    if showable(i, lang):
        text_ = p.text if len(p.text) <= _PARAGRAPH_CAP else p.text[:_PARAGRAPH_CAP].rstrip() + "…"
        shown = html.escape(text_)
    else:
        shown = f'<span style="{_UI}">{html.escape(_HIDDEN[lang])}</span>'
    return ("<tr>" + _cell(_chip(i, _CHIP_PRED), "", _NUMBER) + _cell(html.escape(p.title), _DATA)
            + _cell(shown, _DATA) + "</tr>")


def try_picture(lang: str, text_: str = "") -> str:
    lang = lang if lang in LANGS else "en"
    w = _TRY_TEXT[lang]
    all_words = words((text_ or "")[:TRY_CHARS])
    query = all_words[:TRY_WORDS]
    if not query:
        return yours.figure(w["title"], _note(w["empty"]), alt=w["empty"])
    index = _paragraph_index(lang)
    nums = numbers(lang)
    body = f'<div style="{_DATA}">{html.escape(w["query"].format(q=repr(query)))}</div>'
    missing = [x for x in query if x not in index]
    if missing:
        body += _note(w["missing"].format(x=missing[0]))
    else:
        rows = plan(query, index)
        found = search_and(query, index)
        table = ["<tr>" + _cell(_code(x)) + _cell(_num(n, lang), _DATA, _NUMBER) + _cell(_num(left, lang), _DATA, _NUMBER)
                 + _cell(_num(c, lang) if k else _DASH, _FAINT, _NUMBER) + "</tr>" for k, (x, n, left, c) in enumerate(rows)]
        comparisons = sum(c for _, _, _, c in rows)
        k = len(found)
        body += _note(w["summary"].format(k=_num(k, lang), n=_num(nums["paragraphs"], lang),
                                          verb=w["one"] if k == 1 else w["many"], c=_num(comparisons, lang),
                                          m=_num(nums["words"], lang)))
        body += _table(table, w["head"], (1, 2, 3))
        if found:
            body += _label(w["shown"]) + _table([_paragraph_row(i, lang) for i in found[:_SHOWN_RESULTS]], ["", "", ""])
    if len(all_words) > TRY_WORDS:
        body += _note(w["more"].format(n=_num(TRY_WORDS, lang)))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------------

_Val = S._Val
_DOTS = S._DOTS
_SAME_TEXTS = S._SAME_TEXTS


def _show(v):
    return S._show(v)


def _written(v):
    return S._written(v)


def _small(lang: str) -> _Val:
    return _Val(f'C.SMALL_INDEX["{lang}"]', SMALL_INDEX[lang])


_BI_HEAD = ("def build_index(docs):\n    index = {}\n    for doc in range(len(docs)):\n        for word in docs[doc]:\n"
            "            if word not in index:\n                index[word] = []\n")
_BI_END = "                index[word].append(doc)\n    return index\n"
_BI = _BI_HEAD + "            if doc not in index[word]:\n" + _BI_END
_BI_START = _BI_HEAD + "            if ...:\n" + _BI_END
_BI_ALWAYS = _BI_HEAD + "            if True:\n" + _BI_END
_BI_DICT = _BI_HEAD + "            if doc not in index:\n" + _BI_END
_BI_WORD = _BI_HEAD + "            if word not in index[word]:\n" + _BI_END
_BI_IN = _BI_HEAD + "            if doc in index[word]:\n" + _BI_END

_IN_HEAD = ("def intersect(p1, p2):\n    answer = []\n    i = 0\n    j = 0\n    while i < len(p1) and j < len(p2):\n"
            "        if p1[i] == p2[j]:\n            answer.append(p1[i])\n            i = i + 1\n            j = j + 1\n")
_IN_END = "            i = i + 1\n        else:\n            j = j + 1\n    return answer\n"
_IN = _IN_HEAD + "        elif p1[i] < p2[j]:\n" + _IN_END
_IN_START = _IN_HEAD + "        elif ...:\n" + _IN_END
_IN_GREATER = _IN_HEAD + "        elif p1[i] > p2[j]:\n" + _IN_END
_IN_DIFFER = _IN_HEAD + "        elif p1[i] != p2[j]:\n" + _IN_END
_IN_PLACES = _IN_HEAD + "        elif i < j:\n" + _IN_END

_SA_HEAD = ("def search_and(query, index):\n    lists = []\n    for word in query:\n        if word not in index:\n"
            "            return []\n        lists.append(index[word])\n    lists = sorted(lists, key=len)\n"
            "    answer = lists[0]\n    for postings in lists[1:]:\n")
_SA_END = "    return answer\n"
_SA = _SA_HEAD + "        answer = intersect(answer, postings)\n" + _SA_END
_SA_START = _SA_HEAD + "        answer = ...\n" + _SA_END
_SA_RESTART = _SA_HEAD + "        answer = intersect(lists[0], postings)\n" + _SA_END
_SA_PLUS = _SA_HEAD + "        answer = answer + postings\n" + _SA_END
_SA_LAST = _SA_HEAD + "        answer = postings\n" + _SA_END
_SA_SAME = _SA_HEAD + "        answer = intersect(postings, postings)\n" + _SA_END

# The small cases the exercise texts show; the same in both lessons.
TOY_BUILD = ([["red", "wine"], ["white", "wine", "wine"], ["red"]], [["python"], [], ["python", "snake", "python"]])
TOY_INTERSECT = (([6, 7, 8, 10], [6, 7, 8, 9]), ([2, 4, 8, 16, 32], [1, 2, 3, 5, 8, 13]))
TOY_SEARCH = ((["red", "wine", "heart"], "en"), (["python", "slang"], "nl"))


def _repeats(got, text_) -> bool:
    return isinstance(got, dict) and any(isinstance(v, list) and len(v) != len(set(v)) for v in got.values())


def _all_empty(got, text_) -> bool:
    return isinstance(got, dict) and len(got) > 0 and all(v == [] for v in got.values())


REQUIRED = ["build_index", "intersect", "search_and", "answers"]

EXERCISES = {
    "build_index": FunctionBox(
        name="build_index",
        picture=index_picture,
        function="build_index",
        start=_BI_START,
        solution=_BI,
        show=_show,
        visible=[(_written(docs),) for docs in TOY_BUILD],
        hidden=[(_written([]),),
                (_written([["wine", "red", "wine"], ["wine"]]),),
                (_written([["a", "a", "a"]]),),
                (_written([["heart"], ["heart"], ["heart"]]),),
                (_Val('C.DOCS["en"]', DOCS["en"]),),
                (_Val('C.DOCS["nl"]', DOCS["nl"]),)],
        first=[_DOTS],
        mistakes=[
            Mistake("twice", matches=_repeats,
                    en="`{call}` gives {got}, and it should give {want}: a number is in a postings list more than once, because your test lets the document through each time its word occurs. Does your test look at whether `doc` is already in the list of the word?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een nummer staat meer dan eens in een postings list, want je test laat het document door elke keer dat het woord voorkomt. Kijkt je test of `doc` al in de lijst van het woord staat?"),
            Mistake("never", matches=_all_empty,
                    en="`{call}` gives {got}, and it should give {want}: every postings list stays empty, because your test is never true, so `.append` never runs. When should the number of the document go into the list?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke postings list blijft leeg, want je test is nooit waar, dus `.append` wordt nooit uitgevoerd. Wanneer moet het nummer van het document in de lijst?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the test that lets the number of each document into the list of a word only once.",
                      "nl": "Vervang de `...` door de test die het nummer van elk document één keer in de lijst van een woord toelaat, en niet vaker."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does each word get the numbers of the documents that hold it, in order, and each number once?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Krijgt elk woord de nummers van de documenten waarin het staat, op volgorde, en elk nummer één keer?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function also work for no documents at all, and for a word that a document holds three times?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook zonder documenten, en voor een woord dat drie keer in een document staat?"},
            "correct": {"en": "Right: your function builds an inverted index. In the picture below, it indexes the course's thirteen documents: which documents hold *python*?",
                        "nl": "Goed: je functie bouwt een inverted index. In het plaatje hieronder indexeert de functie de dertien documenten van de course: welke documenten bevatten *python*?"},
        },
    ),
    "intersect": FunctionBox(
        name="intersect",
        picture=intersect_picture,
        function="intersect",
        start=_IN_START,
        solution=_IN,
        show=_show,
        visible=[tuple(_written(p) for p in pair) for pair in TOY_INTERSECT],
        hidden=[(_written([]), _written([1, 2])),
                (_written([1, 2]), _written([])),
                (_written([1, 3, 5]), _written([2, 4, 6])),
                (_written([5, 9]), _written([5, 9])),
                (_written([9]), _written([1, 2, 3, 9])),
                (_written([1, 2, 3, 9]), _written([9])),
                (_written([0, 4, 9, 13]), _written([4, 13, 20])),
                (_written(list(range(0, 60, 3))), _written(list(range(0, 60, 4))))],
        first=[_DOTS],
        mistakes=[
            Mistake("greater", wrong=_IN_GREATER,
                    en="`{call}` gives {got}, and it should give {want}: when the number in `p1` is the larger one, your function moves on in `p1`, and so it passes numbers that `p2` may still hold. Which finger should move on: the one at the smaller number, or the one at the larger?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: als het getal in `p1` het grootste is, schuift je functie door in `p1`, en zo slaat de functie getallen over, terwijl `p2` deze nog kan bevatten. Welke vinger moet doorschuiven: de vinger bij het kleinste getal, of de vinger bij het grootste?"),
            Mistake("differ", wrong=_IN_DIFFER,
                    en="`{call}` gives {got}, and it should give {want}: your test is true whenever the two numbers differ, so your function always moves on in `p1`, also when the number in `p2` is the smaller one. What should the test find out about the two numbers?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je test is waar zodra de twee getallen verschillen, dus je functie schuift altijd door in `p1`, ook als het getal in `p2` het kleinste is. Wat moet de test over de twee getallen te weten komen?"),
            Mistake("places", wrong=_IN_PLACES,
                    en="`{call}` gives {got}, and it should give {want}: your test compares the places `i` and `j`, not the document numbers at those places. Which numbers decide which finger moves on?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je test vergelijkt de plaatsen `i` en `j`, niet de documentnummers op deze plaatsen. Welke getallen bepalen welke vinger doorschuift?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the test for when the walk moves on in `p1`: the number there is the smaller one.",
                      "nl": "Vervang de `...` door de test voor wanneer de wandeling doorschuift in `p1`: het getal daar is het kleinste."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function move on in `p1` when the number there is the smaller of the two, and in `p2` when the number there is?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Schuift je functie door in `p1` als het getal daar het kleinste van de twee is, en in `p2` als het getal daar het kleinste is?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function also work when a list is empty, when the lists share nothing, and when the first list is the longer one?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook als een lijst leeg is, als de lijsten niets delen, en als de eerste lijst de langste is?"},
            "correct": {"en": "Right: your function intersects two postings lists in one walk, without looking at a document. In the picture below, it walks lists of the 2,067 paragraphs: how many numbers does it look at, and how many words would reading the paragraphs take?",
                        "nl": "Goed: je functie vindt de doorsnede van twee postings lists in één wandeling, zonder naar een document te kijken. In het plaatje hieronder wandelt de functie langs lijsten van de 2.067 alinea's: hoeveel getallen bekijkt de functie, en hoeveel woorden zou je in de alinea's moeten lezen?"},
        },
    ),
    "search_and": FunctionBox(
        name="search_and",
        picture=search_picture,
        function="search_and",
        start=_SA_START,
        solution=_SA,
        show=_show,
        names={"intersect": intersect},
        visible=[(_written(q), _small(lang)) for q, lang in TOY_SEARCH],
        hidden=[(_written(["python"]), _small("en")),
                (_written(["red", "heart", "wine", "dinner"]), _small("en")),
                (_written(["red", "wine", "coffee"]), _small("en")),
                (_written(["wine", "wine"]), _small("en")),
                (_written(["rode", "wijn", "hart"]), _small("nl")),
                (_written(["the", "heart", "of", "the", "old", "town"]), _small("en")),
                (_written(["a", "b", "c"]), _written({"a": [0, 2, 4], "b": [1, 2, 4], "c": [2, 3]}))],
        first=[_DOTS],
        mistakes=[
            Mistake("restart", wrong=_SA_RESTART,
                    en="`{call}` gives {got}, and it should give {want}: each walk starts again from the shortest list, so the answer keeps only what the shortest and the last list share. Which name holds what the walks have kept so far?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke wandeling begint opnieuw bij de kortste lijst, dus het antwoord houdt alleen wat de kortste en de laatste lijst delen. Welke naam bevat wat de wandelingen tot nu toe overhielden?"),
            Mistake("plus", wrong=_SA_PLUS,
                    en="`{call}` gives {got}, and it should give {want}: `+` puts two lists one after the other, so the answer holds every number of every list. Which function of the lesson keeps only the numbers that two lists both hold?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `+` zet twee lijsten achter elkaar, dus het antwoord bevat elk getal van elke lijst. Welke functie van de les houdt alleen de getallen die in beide lijsten staan?"),
            Mistake("last", wrong=_SA_LAST,
                    en="`{call}` gives {got}, and it should give {want}: the answer is the last list, the longest, because each round throws away what the walks kept. What should the new answer be made of?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het antwoord is de laatste lijst, de langste, want elke ronde gooit weg wat de wandelingen overhielden. Waaruit moet het nieuwe antwoord ontstaan?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the walk of the answer so far along the list of the next word.",
                      "nl": "Vervang de `...` door de wandeling van het antwoord tot nu toe langs de lijst van het volgende woord."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does each round keep the numbers that the answer so far and the next list both hold?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Houdt elke ronde de getallen die het antwoord tot nu toe en de volgende lijst allebei bevatten?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function also work for a query of one word, a query of four words, and a word that is in no document?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook voor een zoekvraag van één woord, een zoekvraag van vier woorden, en een woord dat in geen enkel document staat?"},
            "correct": {"en": "Right: your function is a search engine that never reads a document. In the picture below, it answers queries of three words on the 2,067 paragraphs: which query has no results, though each of its words is in several paragraphs?",
                        "nl": "Goed: je functie is een zoekmachine die nooit een document leest. In het plaatje hieronder beantwoordt de functie zoekvragen van drie woorden op de 2.067 alinea's: welke zoekvraag heeft geen resultaten, hoewel elk woord ervan in meerdere alinea's staat?"},
        },
    ),
}

# The sort: what the index of this lesson answers by itself, and what needs more than the numbers of the documents.
EXERCISES["answers"] = Sort(
    name="answers",
    groups={
        "index": {"en": "This index answers it: the postings lists of the words are enough",
                  "nl": "Deze index beantwoordt het: de postings lists van de woorden zijn genoeg"},
        "more": {"en": "It needs more than this index: more than which documents hold a word",
                 "nl": "Er is meer nodig dan deze index: meer dan de nummers van de documenten met een woord"},
    },
    items={
        "and": {"en": "*red AND wine*: the documents with both words",
                "nl": "*rode AND wijn*: de documenten met beide woorden"},
        "or": {"en": "*red OR white*: the documents with at least one of the two words",
               "nl": "*rode OR witte*: de documenten met minstens één van de twee woorden"},
        "not": {"en": "*wine AND NOT red*: the documents with *wine* and without *red*",
                "nl": "*wijn AND NOT rode*: de documenten met *wijn* en zonder *rode*"},
        "one": {"en": "*python*: the documents with this one word",
                "nl": "*python*: de documenten met dit ene woord"},
        "phrase": {"en": "the phrase *red wine*: the two words next to each other, in this order",
                   "nl": "de woordgroep *rode wijn*: de twee woorden naast elkaar, in deze volgorde"},
        "near": {"en": "*heart* at most five words away from *attack*",
                 "nl": "*hart* op hooguit vijf woorden afstand van *wijn*"},
        "best": {"en": "the ten documents that fit *red wine heart* best, judged by how often each word occurs in them",
                 "nl": "de tien documenten die het best passen bij *rode wijn hart*, gemeten aan hoe vaak elk woord erin voorkomt"},
        "count": {"en": "the documents that hold *heart* at least three times",
                  "nl": "de documenten waarin *hart* minstens drie keer staat"},
    },
    solution={"and": "index", "or": "index", "not": "index", "one": "index",
              "phrase": "more", "near": "more", "best": "more", "count": "more"},
    nudge={
        "and": {"en": "Which two postings lists does it take, and what does your intersect do with them?",
                "nl": "Welke twee postings lists zijn er nodig, en wat doet je intersect ermee?"},
        "or": {"en": "Walk the two lists and keep every number that is in either. Does that take anything besides the two lists?",
               "nl": "Wandel langs de twee lijsten en bewaar elk getal dat in minstens een van beide staat. Is daar iets anders voor nodig dan de twee lijsten?"},
        "not": {"en": "The documents on the list of *wine* that are not on the list of *red*: does the index hold both lists?",
                "nl": "De documenten op de lijst van *wijn*, zonder de documenten op de lijst van *rode*: heeft de index beide lijsten?"},
        "one": {"en": "What does the index keep for each word?",
                "nl": "Wat bewaart de index bij elk woord?"},
        "phrase": {"en": "Does a postings list say where in the document a word stands?",
                   "nl": "Zegt een postings list waar in het document een woord staat?"},
        "near": {"en": "To know how far apart two words are, what would the index have to keep besides the numbers of the documents?",
                 "nl": "Wat zou de index naast de nummers van de documenten moeten bewaren om te weten hoe ver twee woorden uit elkaar staan?"},
        "best": {"en": "A postings list says that a word is in a document. Does it say how often the word occurs there?",
                 "nl": "Een postings list zegt dat een woord in een document staat. Zegt de lijst hoe vaak het woord daar voorkomt?"},
        "count": {"en": "Does the index of this lesson keep how often a word occurs in a document, or only that it occurs?",
                  "nl": "Bewaart de index van deze les hoe vaak een woord in een document voorkomt, of alleen dat het voorkomt?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the groups again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de groepen opnieuw uit."},
        "empty": {"en": "Sort each query into the group that fits.",
                  "nl": "Deel elke zoekvraag in bij de groep die klopt."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "{item} belongs with another group.",
                  "nl": "{item} hoort bij een andere groep."},
        "correct": {"en": "All {n} are right. The index of this lesson keeps, for each word, only which documents hold it: enough for AND, OR and NOT, the queries of Boolean retrieval. For a phrase or words near each other, an index also keeps where each word stands in a document; to rank, it keeps how often each word occurs.",
                    "nl": "Alle {n} kloppen. De index van deze les bewaart bij elk woord alleen welke documenten het bevatten: genoeg voor AND, OR en NOT, de zoekvragen van Boolean retrieval. Voor een woordgroep of woorden dicht bij elkaar bewaart een index ook waar elk woord in een document staat; om te rangschikken bewaart een index hoe vaak elk woord voorkomt."},
    },
)

# What the tests feed each check, and what it must say: (exercise, code or choices, state, known mistake).
CASES = [
    ("build_index", _BI_START, "empty", None),
    ("build_index", _BI, "correct", None),
    ("build_index", _BI_HEAD + "            if doc not in ...:\n" + _BI_END, "wrong", "dots-left"),
    ("build_index", _BI_ALWAYS, "wrong", "twice"),
    ("build_index", _BI_DICT, "wrong", "twice"),
    ("build_index", _BI_WORD, "wrong", "twice"),
    ("build_index", _BI_IN, "wrong", "never"),
    ("build_index", _BI_HEAD + "            if index[word] == [] or index[word][-1] != doc:\n" + _BI_END, "correct", None),
    ("build_index", "def build_index(docs):\n    index = {}\n    for doc in range(len(docs)):\n"
     "        for word in set(docs[doc]):\n            if word not in index:\n                index[word] = []\n"
     "            index[word].append(doc)\n    return index\n", "correct", None),
    ("intersect", _IN_START, "empty", None),
    ("intersect", _IN, "correct", None),
    ("intersect", _IN_HEAD + "        elif p1[i] < ...:\n" + _IN_END, "wrong", "dots-left"),
    ("intersect", _IN_GREATER, "wrong", "greater"),
    ("intersect", _IN_DIFFER, "wrong", "differ"),
    ("intersect", _IN_PLACES, "wrong", "places"),
    ("intersect", _IN_HEAD + "        elif p2[j] > p1[i]:\n" + _IN_END, "correct", None),
    ("intersect", _IN_HEAD + "        elif p1[i] <= p2[j]:\n" + _IN_END, "correct", None),
    ("search_and", _SA_START, "empty", None),
    ("search_and", _SA, "correct", None),
    ("search_and", _SA_HEAD + "        answer = intersect(answer, ...)\n" + _SA_END, "wrong", "dots-left"),
    ("search_and", _SA_RESTART, "wrong", "restart"),
    ("search_and", _SA_PLUS, "wrong", "plus"),
    ("search_and", _SA_LAST, "wrong", "last"),
    ("search_and", _SA_SAME, "wrong", "last"),
    ("search_and", _SA_HEAD + "        answer = intersect(postings, answer)\n" + _SA_END, "correct", None),
    ("answers", {}, "empty", None),
    ("answers", dict(EXERCISES["answers"].solution), "correct", None),
    ("answers", {"and": "index", "phrase": "more"}, "wrong", "unfinished"),
] + [("answers", {item: ("more" if group == "index" else "index")}, "wrong", f"wrong:{item}")
     for item, group in EXERCISES["answers"].solution.items()]
