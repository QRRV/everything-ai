# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-document-vectors: turn a list of words into a vector of counts over a fixed vocabulary
(vectorize), measure the cosine of two sparse rows that hold only their nonzero counts (sparse_cosine), and find the
training sentence most like a new one (most_similar); and sort pairs of sentences by their cosine.

The sentences are "Sentiment Labelled Sentences" (Kotzias, Denil, de Freitas and Smyth, KDD 2015; UCI Machine Learning
Repository, CC BY 4.0, https://doi.org/10.24432/C57604), in data/uci-sentiment.tsv.gz, made by
data/tools/make_uci_sentiment.py. Sentences marked as not shown are counted but never shown. The other texts, the cases
of the checks and the pictures are the course's own."""
import contextlib
import copy
import html
import io
import math

import numpy as np

from everything_ai import books, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, kept, with_time_limit

# --- the sentences -----------------------------------------------------------------------------------------------------

HELD_OUT = 5   # every fifth sentence stays out of the vocabulary and the training rows, to try the method on
LABELS = {"en": {1: "positive", 0: "negative"}, "nl": {1: "positief", 0: "negatief"}}


def words(text: str) -> list:
    """The words of a text in small letters: runs of letters, with an apostrophe inside a word as in *don't*. Digits
    and punctuation do not count as part of a word."""
    return books.words(text.lower())


def _all() -> tuple:
    return books.review_sentences()


def training() -> list:
    """The 2,400 training sentences, as (label, text, shown): all but every fifth sentence of the set."""
    return [(label, text, shown) for k, (_, _, label, shown, text) in enumerate(_all()) if k % HELD_OUT != HELD_OUT - 1]


def held_out() -> list:
    """The 600 held-out sentences, every fifth sentence of the set, as (label, text, shown)."""
    return [(label, text, shown) for k, (_, _, label, shown, text) in enumerate(_all()) if k % HELD_OUT == HELD_OUT - 1]


def find(source: str, line: int) -> tuple:
    """(label, text, shown) of the sentence on `line` of the file of `source` (amazon, imdb or yelp)."""
    for s, n, label, shown, text in _all():
        if s == source and n == line:
            return label, text, shown
    raise KeyError((source, line))


# --- the course's solutions --------------------------------------------------------------------------------------------

def vectorize(words, columns):
    """The course's solution of Exercise 1: a list with one count for each column of `columns`; the count at the
    column of a word goes up by one each time the word occurs. A word that `columns` lacks is left out."""
    vector = [0 for column in columns]
    for word in words:
        if word in columns:
            vector[columns[word]] = vector[columns[word]] + 1
    return vector


def norm(row) -> float:
    """The length of a sparse row: the square root of the sum of the squares of its counts; 0.0 for an empty row."""
    return math.sqrt(sum(count * count for count in row.values()))


def sparse_cosine(a, b):
    """The course's solution of Exercise 2: the cosine of two sparse rows, dictionaries from column to count. Only the
    columns that both rows hold add to the dot product; 0.0 when they share no column."""
    dot = 0
    for column in a:
        if column in b:
            dot = dot + a[column] * b[column]
    if dot == 0:
        return 0.0
    return dot / (norm(a) * norm(b))


cosine = sparse_cosine


def most_similar(row, rows):
    """The course's solution of Exercise 3: the place in `rows` of the row with the highest cosine with `row`, the
    first one when several are equal; None when no row shares a column with `row`."""
    best = None
    best_value = 0
    for i in range(len(rows)):
        value = cosine(row, rows[i])
        if value > best_value:
            best = i
            best_value = value
    return best


# --- the vocabulary, the rows and how well the method does -------------------------------------------------------------

BASE = (True, False, False, False)   # lowercase, remove stop words, 0 or 1 instead of counts, drop words seen once


def _words_of(text: str, choice: tuple) -> list:
    lower, stop = choice[0], choice[1]
    found = books.words(text.lower() if lower else text)
    if stop:
        drop = books.stop_words("en")
        found = [w for w in found if w.lower() not in drop]
    return found


def _model(choice: tuple = BASE) -> dict:
    """The vocabulary of the training sentences under a choice, as {word: column} in alphabetical order, with the
    sparse rows of the training and the held-out sentences."""
    def make():
        train_words = [_words_of(t, choice) for _, t, _ in training()]
        counts = {}
        for ws in train_words:
            for w in ws:
                counts[w] = counts.get(w, 0) + 1
        vocab = sorted(w for w in counts if counts[w] >= (2 if choice[3] else 1))
        cols = {w: k for k, w in enumerate(vocab)}

        def row(ws):
            r = _row(ws, cols)
            return {k: 1 for k in r} if choice[2] else r
        return {"columns": cols, "vocabulary": vocab, "counts": counts,
                "rows": [row(ws) for ws in train_words],
                "held": [row(_words_of(t, choice)) for _, t, _ in held_out()]}
    return kept(f"nlp-document-vectors:model:{choice}", make)


def _row(ws: list, cols: dict) -> dict:
    out = {}
    for w in ws:
        if w in cols:
            out[cols[w]] = out.get(cols[w], 0) + 1
    return dict(sorted(out.items()))


def columns() -> dict:
    """The vocabulary of the 2,400 training sentences, as a new dict from word to column: 0 for the first word in
    alphabetical order, then 1, and so on."""
    return dict(_model()["columns"])


def vocabulary() -> list:
    """The words of the vocabulary in the order of their columns, as a new list."""
    return list(_model()["vocabulary"])


def sparse_row(words) -> dict:
    """The sparse row of a list of words: a dict from column to count, only for the columns whose count is not 0, by
    column. A word outside the vocabulary is left out."""
    return _row(list(words), _model()["columns"])


def training_labels() -> list:
    """The labels of the 2,400 training sentences, in order: 1 for positive, 0 for negative."""
    return [label for label, _, _ in training()]


def training_rows() -> list:
    """The sparse rows of the 2,400 training sentences, in order, as a new list."""
    return copy.deepcopy(_model()["rows"])


def stats(choice: tuple = BASE) -> dict:
    """The size of the training matrix under a choice: rows, columns, nonzero cells and the share of zeros."""
    m = _model(choice)
    nonzero = sum(len(r) for r in m["rows"])
    cells = len(m["rows"]) * len(m["columns"])
    return {"rows": len(m["rows"]), "columns": len(m["columns"]), "nonzero": nonzero, "cells": cells,
            "zeros": 100 * (1 - nonzero / cells), "mean": nonzero / len(m["rows"]),
            "longest": max(len(r) for r in m["rows"]), "empty": sum(1 for r in m["rows"] if not r)}


def _dense(rows: list, n: int) -> np.ndarray:
    out = np.zeros((len(rows), n))
    for i, r in enumerate(rows):
        for k, c in r.items():
            out[i, k] = c
    return out


def score(choice: tuple = BASE) -> dict:
    """What the most similar training sentence says about each held-out sentence under a choice: {"answers": [label or
    None, ...], "right": k, "none": k, "total": 600}. The same answers as most_similar with sparse_cosine, done at once
    with NumPy."""
    def make():
        m = _model(choice)
        n = len(m["columns"])
        a, b = _dense(m["rows"], n), _dense(m["held"], n)
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        dots = b @ a.T
        with np.errstate(divide="ignore", invalid="ignore"):
            cos = np.nan_to_num(dots / (nb[:, None] * na[None, :]))
        labels = [label for label, _, _ in training()]
        answers = []
        for i in range(len(m["held"])):
            answers.append(None if cos[i].max() <= 0 else labels[int(np.argmax(cos[i]))])
        truth = [label for label, _, _ in held_out()]
        return {"answers": answers, "right": sum(a == t for a, t in zip(answers, truth)),
                "none": sum(a is None for a in answers), "total": len(answers)}
    return kept(f"nlp-document-vectors:score:{choice}", make)


def naive_bayes() -> dict:
    """Naive Bayes on the same counts, with Laplace smoothing, as in the lesson Spam or not: {"right": k, "total": 600}."""
    def make():
        m = _model()
        n = len(m["columns"])
        labels = [label for label, _, _ in training()]
        counts = {0: [0] * n, 1: [0] * n}
        for label, r in zip(labels, m["rows"]):
            for k, c in r.items():
                counts[label][k] += c
        total = {c: sum(counts[c]) for c in counts}
        right = 0
        for (truth, _, _), r in zip(held_out(), m["held"]):
            scores = {}
            for c in (0, 1):
                s = math.log(labels.count(c) / len(labels))
                for k, x in r.items():
                    s += x * math.log((counts[c][k] + 1) / (total[c] + n))
                scores[c] = s
            right += max((1, 0), key=lambda c: scores[c]) == truth
        return {"right": right, "total": len(m["held"])}
    return kept("nlp-document-vectors:naive-bayes", make)


def pair_cosine(a: str, b: str) -> float:
    """The cosine of the rows of two texts over the vocabulary of the lesson."""
    return sparse_cosine(sparse_row(words(a)), sparse_row(words(b)))


# --- the look of the pictures ------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);margin-top:0.45rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:baseline;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums"
_CHIP = ("display:inline-block;margin:0.12rem 0.2rem 0.12rem 0;padding:0 0.3rem;border-radius:3px;"
         "font-family:var(--ea-font-code);font-size:0.8rem;line-height:1.5;white-space:pre")
_CHIP_DATA = _CHIP + ";border:1px solid var(--ea-line);color:var(--ea-ink)"
_CHIP_PRED = _CHIP + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_CHIP_PURPLE = _CHIP + ";background:var(--ea-loss-wash);color:var(--ea-loss-text)"
_CHIP_MISSING = _CHIP + ";border:1px dashed var(--ea-loss);color:var(--ea-loss-text)"
_CHIP_FAINT = _CHIP + ";color:var(--ea-ink-3)"
_CHIP_OUTSIDE = _CHIP + ";border:1px dashed var(--ea-ink-3);color:var(--ea-ink-2)"
_BAR = "display:inline-block;height:0.6rem;border-radius:2px;vertical-align:middle"
_TEXT = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)"


def _note(text: str) -> str:
    """A line under a picture; a word between asterisks is set in italics, as in the lesson's text."""
    parts = html.escape(text).split("*")
    marked = "".join(f"<i>{p}</i>" if k % 2 else p for k, p in enumerate(parts))
    return f'<div style="{_UI}">{marked}</div>'


def _label(text: str) -> str:
    parts = html.escape(text).split("*")
    marked = "".join(f"<i>{p}</i>" if k % 2 else p for k, p in enumerate(parts))
    return f'<div style="{_LABEL}">{marked}</div>'


def _num(n, lang: str) -> str:
    """A whole number with a separator for thousands: 4,518 in English, 4.518 in Dutch."""
    text = f"{round(n):,}"
    return text.replace(",", ".") if lang == "nl" else text


def _pct(x: float, lang: str, digits: int = 1) -> str:
    """A percentage: 66.2% in English, 66,2% in Dutch."""
    text = f"{x:.{digits}f}%"
    return text.replace(".", ",") if lang == "nl" else text


def _dec(x: float, lang: str, digits: int = 2) -> str:
    """A number with `digits` decimals: 0.52 in English, 0,52 in Dutch."""
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def _chip(text: str, style: str = _CHIP_DATA) -> str:
    return f'<span style="{style}">{html.escape(text)}</span>'


def _bar(x: float, most: float, color: str, width: float = 8.0) -> str:
    """A bar for a number: as long as `width` for `most`, never shorter than a sliver."""
    w = 0 if not most or not x else max(0.15, width * x / most)
    return f'<span style="{_BAR};width:{w:.2f}rem;background:{color}"></span>' if w else ""


def _table(rows_: list, head: list, numbers: tuple = ()) -> str:
    """A small table; rows hold HTML that is already escaped; numbers are the columns set to the right; no head when
    `head` is empty."""
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers else ""}">{c}</td>'
                                    for j, c in enumerate(row)) + "</tr>" for row in rows_)
    head_html = f"<thead><tr>{top}</tr></thead>" if head else ""
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem">'
            f"{head_html}<tbody>{body}</tbody></table></div>")


_HIDDEN = {"en": "(a sentence the course does not show)", "nl": "(een zin die de course niet laat zien)"}


def _sentence(text: str, shown: int, lang: str, longest: int = 70) -> str:
    """A sentence of the set, cut off after `longest` characters, or a placeholder for one the course does not show."""
    if not shown:
        return f'<span style="{_TEXT};color:var(--ea-ink-3)">{html.escape(_HIDDEN[lang])}</span>'
    cut = text if len(text) <= longest else text[:longest - 1] + "…"
    return f'<span style="{_TEXT}">{html.escape(cut)}</span>'


def _plain(text: str, shown: int, lang: str) -> str:
    return text if shown else _HIDDEN[lang]


def _label_chip(label, lang: str, right: bool = True) -> str:
    if label is None:
        return _chip("–", _CHIP_FAINT)
    return _chip(LABELS[lang][label], _CHIP_PRED if right else _CHIP_PURPLE)


def _data_label(label, lang: str) -> str:
    return _chip(LABELS[lang][label], _CHIP_DATA)


# --- the idea's picture: four sentences as rows of counts --------------------------------------------------------------

EXAMPLES = [("amazon", 5), ("yelp", 2), ("yelp", 912), ("yelp", 11)]

_IDEA_WORDS = {
    "en": {"title": "Four sentences as a document-term matrix",
           "head": "Sentence",
           "note": "One row per sentence, one column per word of these four sentences, in alphabetical order. Each cell "
                   "counts how often the word of its column occurs in the sentence of its row; gray zeros: not at all. "
                   "Over the {n} words of the 2,400 training sentences, each row would have {n} cells, and only two to "
                   "four of them would not be 0.",
           "alt": "{title}. Columns: {columns}. {rows}. {note}"},
    "nl": {"title": "Vier zinnen als document-term matrix",
           "head": "Zin",
           "note": "Eén rij per zin, één kolom per woord van deze vier zinnen, op alfabet. Elke cel telt hoe vaak het "
                   "woord van de kolom in de zin van de rij voorkomt; grijze nullen: helemaal niet. Over de {n} woorden "
                   "van de 2.400 trainingszinnen zou elke rij {n} cellen hebben, en maar twee tot vier daarvan zouden "
                   "geen 0 zijn.",
           "alt": "{title}. Kolommen: {columns}. {rows}. {note}"},
}

_VERTICAL = ("writing-mode:vertical-rl;transform:rotate(180deg);padding:0.3rem 0.15rem;font-family:var(--ea-font-code);"
             "font-size:0.78rem;color:var(--ea-ink-2);font-weight:400;border-bottom:1px solid var(--ea-line);"
             "text-align:left;vertical-align:bottom")
_COUNT = ("padding:0.12rem 0.2rem;text-align:center;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-code);"
          "font-size:0.8rem;font-variant-numeric:tabular-nums")


def example_sentences() -> list:
    """The four sentences of the idea's picture, as (label, text, shown)."""
    return [find(s, n) for s, n in EXAMPLES]


def matrix_html(texts: list, cells: list, vocab: list, head: str) -> str:
    """A document-term matrix: one row per text, one column per word, with vertical column heads."""
    top = f'<th style="{_HEAD}">{html.escape(head)}</th>' + "".join(
        f'<th style="{_VERTICAL}">{html.escape(w)}</th>' for w in vocab)
    body = ""
    for text, row in zip(texts, cells):
        tds = "".join(f'<td style="{_COUNT};color:{"var(--ea-ink);font-weight:600" if c else "var(--ea-ink-3)"}">'
                      f"{c}</td>" for c in row)
        body += f'<tr><td style="{_CELL}">{html.escape(text)}</td>{tds}</tr>'
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem">'
            f"<thead><tr>{top}</tr></thead><tbody>{body}</tbody></table></div>")


def idea_picture(lang: str) -> str:
    w = _IDEA_WORDS[lang]
    texts = [t for _, t, _ in example_sentences()]
    lists = [words(t) for t in texts]
    vocab = sorted({x for ws in lists for x in ws})
    cols = {x: k for k, x in enumerate(vocab)}
    cells = [vectorize(ws, cols) for ws in lists]
    note = w["note"].format(n=_num(len(columns()), lang))
    body = matrix_html(texts, cells, vocab, w["head"]) + _note(note)
    rows = "; ".join(f"{t}: " + " ".join(str(c) for c in row) for t, row in zip(texts, cells))
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], columns=", ".join(vocab), rows=rows,
                                                              note=note))


# --- running the learner's function ----------------------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error
PICTURE_LIMIT = 10.0   # seconds the learner's function gets for everything a picture asks of it


def _call(make, seconds: float = PICTURE_LIMIT):
    """make() within a time limit, with what it prints thrown away; _STOPPED when it stops with an error, and the
    Timeout passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(make, seconds)
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _run_or_say(make, w: dict, ok) -> tuple:
    """(what make() gives, None), or (None, a figure that says why there is no picture): an error, too slow, or a
    result that `ok` does not accept."""
    try:
        got = _call(make)
    except Timeout:
        text = w["slow"].format(s=PICTURE_LIMIT)
        return None, yours.figure(w["title"], _note(text), alt=text)
    if got is _STOPPED:
        return None, yours.figure(w["title"], _note(w["stopped"]), alt=w["stopped"])
    if not ok(got):
        return None, yours.figure(w["title"], _note(w["odd"]), alt=w["odd"])
    return got, None


_SAY = {
    "en": {"stopped": "Your function stopped with an error on one of the sentences, so the picture stays empty.",
           "slow": "Your function took more than {s:g} seconds for all sentences, so the picture stays empty."},
    "nl": {"stopped": "Je functie stopte met een fout op een van de zinnen, dus de afbeelding blijft leeg.",
           "slow": "Je functie deed er langer dan {s:g} seconden over voor alle zinnen, dus de afbeelding blijft leeg."},
}


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# --- the picture under exercise 1: your vectors ------------------------------------------------------------------------

MAP_ROWS = 60   # training sentences in the map of the matrix

_VECTOR_WORDS = {
    "en": {"title": "Your vectors",
           "head": ["Sentence", "Numbers", "Not 0 (word: count)", "Zeros"],
           "map": "Your vectors of the first {k} training sentences: one line each, a dot for every number that is not 0",
           "share": "Zeros in these {k} vectors: {zeros} of {cells} ({pct})",
           "note": "Gray: what your function counted, with the word of its column. Purple: a count that should not be "
                   "there or should be another number; dashed: a count that is missing. In the map, the width is the "
                   "{n} columns of the vocabulary, from *a* on the left to the last word on the right.",
           "odd": "Your function gave something other than a list of whole numbers as long as the vocabulary, so the "
                  "picture stays empty.",
           "alt": "{title}. {rows}. {share}. {note}"},
    "nl": {"title": "Je vectoren",
           "head": ["Zin", "Getallen", "Niet 0 (woord: aantal)", "Nullen"],
           "map": "Je vectoren van de eerste {k} trainingszinnen: elk een regel, een stip voor elk getal dat niet 0 is",
           "share": "Nullen in deze {k} vectoren: {zeros} van {cells} ({pct})",
           "note": "Grijs: wat je functie telde, met het woord van de kolom. Paars: een aantal dat er niet hoort of een "
                   "ander getal moet zijn; met een gestreepte rand: een aantal dat ontbreekt. In de kaart is de breedte "
                   "de {n} kolommen van de vocabulary, van *a* links tot het laatste woord rechts.",
           "odd": "Je functie gaf iets anders dan een lijst hele getallen zo lang als de vocabulary, dus de afbeelding "
                  "blijft leeg.",
           "alt": "{title}. {rows}. {share}. {note}"},
}


def _vector_ok(v, n: int) -> bool:
    return isinstance(v, list) and len(v) == n and all(isinstance(x, int) and not isinstance(x, bool) for x in v)


def _map_svg(rows: list, n: int) -> str:
    """A dot for every nonzero number of each row: the matrix seen from far away."""
    width, step = 600, 5
    dots = "".join(f'<rect x="{600 * k / n:.1f}" y="{i * step + 1}" width="1.6" height="3" />'
                   for i, row in enumerate(rows) for k, c in enumerate(row) if c)
    h = len(rows) * step + 2
    return (f'<svg viewBox="0 0 {width} {h}" width="100%" style="max-width:{width}px;display:block;margin:0.3rem 0;'
            f'border:1px solid var(--ea-line)" role="img" aria-hidden="true"><g fill="var(--ea-ink)">{dots}</g></svg>')


def vectorize_picture(seen) -> str:
    lang = seen.lang
    w = _VECTOR_WORDS[lang] | _SAY[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    cols = _model()["columns"]
    vocab = _model()["vocabulary"]
    n = len(cols)
    examples = example_sentences()
    first = [t for _, t, _ in training()[:MAP_ROWS]]
    got, said = _run_or_say(lambda: ([fn(words(t), dict(cols)) for _, t, _ in examples],
                                     [fn(words(t), dict(cols)) for t in first]), w,
                            lambda g: all(_vector_ok(v, n) for v in g[0] + g[1]))
    if said:
        return said
    mine_examples, mine_map = got
    rows_, alts = [], []
    for (_, t, _), mine in zip(examples, mine_examples):
        want = vectorize(words(t), cols)
        chips = []
        for k in range(n):
            if mine[k] and mine[k] == want[k]:
                chips.append(_chip(f"{vocab[k]}: {mine[k]}"))
            elif mine[k]:
                chips.append(_chip(f"{vocab[k]}: {mine[k]}", _CHIP_PURPLE))
            elif want[k]:
                chips.append(_chip(f"{vocab[k]}: {want[k]}", _CHIP_MISSING))
        zeros = sum(1 for x in mine if x == 0)
        rows_.append([_sentence(t, 1, lang), _num(len(mine), lang), "".join(chips), _num(zeros, lang)])
        alts.append(f"{t}: " + ", ".join(f"{vocab[k]} {mine[k]}" for k in range(n) if mine[k])
                    + f"; {w['head'][3]}: {_num(zeros, lang)}")
    cells = len(mine_map) * n
    zeros = sum(1 for row in mine_map for x in row if x == 0)
    share = w["share"].format(k=len(mine_map), zeros=_num(zeros, lang), cells=_num(cells, lang),
                              pct=_pct(100 * zeros / cells, lang, 2))
    note = w["note"].format(n=_num(n, lang))
    body = (_table(rows_, w["head"], numbers=(1, 3)) + _label(w["map"].format(k=len(mine_map)))
            + _map_svg(mine_map, n) + _label(share) + _note(note))
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts), share=share,
                                                              note=note.replace("*", "")))


# --- the picture under exercise 2: the most similar sentences ----------------------------------------------------------

QUERY = ("amazon", 475)   # The delivery was on time. (held out)
TOP = 5

_COSINE_WORDS = {
    "en": {"title": "Your cosine, against all 2,400 training sentences",
           "query": "The new sentence: *{text}* ({label})",
           "head": ["Training sentence", "Label", "Your cosine", "", "Words in common"],
           "zero": "{k} of the 2,400 training sentences share no word with it: your cosine gives them 0.",
           "note": "Gray: the training sentences with the highest cosine, by your function, their labels, and the words "
                   "they share with the new sentence. Blue: the label of the most similar one, *{answer}*, the answer for "
                   "the new sentence.",
           "odd": "Your function gave something other than a number, so the picture stays empty.",
           "alt": "{title}. {query}. {rows}. {zero} {note}"},
    "nl": {"title": "Je cosine similarity, tegen alle 2.400 trainingszinnen",
           "query": "De nieuwe zin: *{text}* ({label})",
           "head": ["Trainingszin", "Label", "Je cosine similarity", "", "Gedeelde woorden"],
           "zero": "{k} van de 2.400 trainingszinnen delen geen woord met de nieuwe zin: je functie geeft deze zinnen 0.",
           "note": "Grijs: de trainingszinnen met de hoogste cosine similarity, volgens je functie, de labels, en de woorden "
                   "die elke zin met de nieuwe zin deelt. Blauw: het label van de meest gelijkende zin, *{answer}*, het "
                   "antwoord voor de nieuwe zin.",
           "odd": "Je functie gaf iets anders dan een getal, dus de afbeelding blijft leeg.",
           "alt": "{title}. {query}. {rows}. {zero} {note}"},
}


def sparse_cosine_picture(seen) -> str:
    lang = seen.lang
    w = _COSINE_WORDS[lang] | _SAY[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    label, text, _ = find(*QUERY)
    query = sparse_row(words(text))
    rows = _model()["rows"]
    train = training()
    vocab = _model()["vocabulary"]
    got, said = _run_or_say(lambda: [fn(dict(query), dict(r)) for r in rows], w,
                            lambda g: all(_number(v) for v in g))
    if said:
        return said
    order = sorted(range(len(rows)), key=lambda i: (-got[i], i))[:TOP]
    rows_, alts = [], []
    for i in order:
        lab, t, shown = train[i]
        shared = [vocab[k] for k in query if k in rows[i]]
        rows_.append([_sentence(t, shown, lang), _label_chip(lab, lang) if i == order[0] else _data_label(lab, lang),
                      _dec(got[i], lang, 3),
                      _bar(max(got[i], 0), 1.0, "var(--ea-data)", 6.0), "".join(_chip(x) for x in shared)])
        alts.append(f"{_plain(t, shown, lang)} ({LABELS[lang][lab]}) {_dec(got[i], lang, 3)}")
    zero = w["zero"].format(k=_num(sum(1 for v in got if v == 0), lang))
    q = w["query"].format(text=text, label=LABELS[lang][label])
    note = w["note"].format(answer=LABELS[lang][train[order[0]][0]])
    body = _label(q) + _table(rows_, w["head"], numbers=(2,)) + _label(zero) + _note(note)
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], query=q.replace("*", ""),
                                                              rows="; ".join(alts), zero=zero,
                                                              note=note.replace("*", "")))


# --- the picture under exercise 3: twelve new sentences ----------------------------------------------------------------

PICTURE_HELD = [("amazon", 5), ("amazon", 235), ("amazon", 670), ("amazon", 830), ("amazon", 990), ("imdb", 180),
                ("imdb", 340), ("imdb", 580), ("imdb", 780), ("yelp", 10), ("yelp", 310), ("yelp", 640)]

_SIMILAR_WORDS = {
    "en": {"title": "Your most_similar on twelve new sentences",
           "head": ["New sentence", "Its label", "Most similar training sentence, by your function", "Says"],
           "none": "(none)",
           "total": "The label of the most similar sentence is right for {right} of these {k}.",
           "note": "Blue: the label of the most similar training sentence when it is right; purple: when it is wrong. "
                   "These twelve sentences are held out: the vocabulary and the training rows were made without them.",
           "odd": "Your function gave something other than a place in rows or None, so the picture stays empty.",
           "alt": "{title}. {rows}. {total} {note}"},
    "nl": {"title": "Je most_similar op twaalf nieuwe zinnen",
           "head": ["Nieuwe zin", "Eigen label", "Meest gelijkende trainingszin, volgens je functie", "Zegt"],
           "none": "(geen)",
           "total": "Het label van de meest gelijkende zin klopt bij {right} van deze {k}.",
           "note": "Blauw: het label van de meest gelijkende trainingszin als dat klopt; paars: als dat niet klopt. "
                   "Deze twaalf zinnen zijn achtergehouden: de vocabulary en de trainingsrijen zijn gemaakt zonder deze zinnen.",
           "odd": "Je functie gaf iets anders dan een plaats in rows of None, dus de afbeelding blijft leeg.",
           "alt": "{title}. {rows}. {total} {note}"},
}


def most_similar_picture(seen) -> str:
    lang = seen.lang
    w = _SIMILAR_WORDS[lang] | _SAY[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    rows = _model()["rows"]
    train = training()
    tests = [find(s, n) for s, n in PICTURE_HELD]

    def make():
        given = copy.deepcopy(rows)
        return [fn(sparse_row(words(t)), given) for _, t, _ in tests]

    def ok(g):
        return all(a is None or (isinstance(a, (int, np.integer)) and not isinstance(a, bool) and 0 <= a < len(rows))
                   for a in g)
    got, said = _run_or_say(make, w, ok)
    if said:
        return said
    got = [None if a is None else int(a) for a in got]
    rows_, alts, right = [], [], 0
    for (label, t, _), a in zip(tests, got):
        if a is None:
            rows_.append([_sentence(t, 1, lang), LABELS[lang][label], _chip(w["none"], _CHIP_FAINT), _label_chip(None, lang)])
            alts.append(f"{t} ({LABELS[lang][label]}): {w['none']}")
            continue
        lab, other, shown = train[a]
        right += lab == label
        rows_.append([_sentence(t, 1, lang), LABELS[lang][label], _sentence(other, shown, lang),
                      _label_chip(lab, lang, lab == label)])
        alts.append(f"{t} ({LABELS[lang][label]}): {_plain(other, shown, lang)} ({LABELS[lang][lab]})")
    total = w["total"].format(right=right, k=len(tests))
    body = _table(rows_, w["head"]) + _label(total) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts), total=total,
                                                              note=w["note"]))


# --- how good is it? -----------------------------------------------------------------------------------------------------

STOP = (True, True, False, False)

_COMPARE_WORDS = {
    "en": {"title": "Right on the 600 held-out sentences",
           "coin": "A coin",
           "similar": "Most similar sentence, counts",
           "stop": "Most similar sentence, without stop words",
           "bayes": "Naive Bayes, the same counts",
           "note": "Gray: a coin, for comparison. Blue: the share of the 600 held-out sentences that gets the right "
                   "label. Every fifth sentence of "
                   "the set is held out; the vocabulary and the training rows come from the other 2,400.",
           "alt": "{title}. {rows}. {note}"},
    "nl": {"title": "Goed op de 600 achtergehouden zinnen",
           "coin": "Een munt",
           "similar": "Meest gelijkende zin, aantallen",
           "stop": "Meest gelijkende zin, zonder stop words",
           "bayes": "Naive Bayes, dezelfde aantallen",
           "note": "Grijs: een munt, ter vergelijking. Blauw: het deel van de 600 achtergehouden zinnen dat het goede "
                   "label krijgt. Elke vijfde zin van de "
                   "set wordt achtergehouden; de vocabulary en de trainingsrijen komen uit de andere 2.400.",
           "alt": "{title}. {rows}. {note}"},
}


def compare_picture(lang: str) -> str:
    w = _COMPARE_WORDS[lang]
    s, st, nb = score(), score(STOP), naive_bayes()
    items = [(w["coin"], 50.0), (w["similar"], 100 * s["right"] / s["total"]),
             (w["stop"], 100 * st["right"] / st["total"]), (w["bayes"], 100 * nb["right"] / nb["total"])]
    rows_ = [[html.escape(name), _pct(x, lang), _bar(x, 100, "var(--ea-ink-3)" if k == 0 else "var(--ea-pred)", 12.0)]
             for k, (name, x) in enumerate(items)]
    body = _table(rows_, [], numbers=(1,)) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"].format(
        title=w["title"], rows="; ".join(f"{name}: {_pct(x, lang)}" for name, x in items), note=w["note"]))


# --- the demo: your own sentence ---------------------------------------------------------------------------------------

DEMO_TEXT = "The food was good but the service was slow."
DEMO_LONGEST = 2000   # characters of the demo's text that the program looks at

_DEMO_WORDS = {
    "en": {"title": "Your sentence as a vector",
           "vector": "Your sentence has {k} different words of the vocabulary, so {z} of the {n} numbers of its vector are 0",
           "outside": "Not in the vocabulary, so left out:",
           "head": ["Most similar training sentences", "Label", "Cosine", ""],
           "answer": "The label of the most similar sentence:",
           "none": "No training sentence shares a word with your sentence: every cosine is 0, and the program gives no answer.",
           "empty": "Type a sentence first.",
           "note": "Gray: the words of your sentence with their column and count, the labels of the training sentences, "
                   "and the cosines. Dashed: a word outside the vocabulary of the 2,400 training sentences. Blue: the "
                   "answer, the label of the most similar sentence.",
           "alt": "{title}. {vector}. {words}. {rows}. {answer}. {note}"},
    "nl": {"title": "Je zin als vector",
           "vector": "Je zin heeft {k} verschillende woorden uit de vocabulary, dus {z} van de {n} getallen van de vector zijn 0",
           "outside": "Niet in de vocabulary, dus weggelaten:",
           "head": ["Meest gelijkende trainingszinnen", "Label", "Cosine similarity", ""],
           "answer": "Het label van de meest gelijkende zin:",
           "none": "Geen enkele trainingszin deelt een woord met je zin: elke cosine similarity is 0, en het programma geeft geen antwoord.",
           "empty": "Typ eerst een zin.",
           "note": "Grijs: de woorden van je zin met de kolom en het aantal, de labels van de trainingszinnen, en de cosine "
                   "similarity van elke zin. Met een gestreepte rand: een woord buiten de vocabulary van de 2.400 "
                   "trainingszinnen. Blauw: het antwoord, het label van de meest gelijkende zin.",
           "alt": "{title}. {vector}. {words}. {rows}. {answer}. {note}"},
}


def demo_picture(text, lang: str) -> str:
    w = _DEMO_WORDS[lang]
    text = (text or "")[:DEMO_LONGEST]
    found = words(text)
    if not found:
        return yours.figure(w["title"], _note(w["empty"]), alt=w["empty"])
    cols = _model()["columns"]
    row = sparse_row(found)
    vocab = _model()["vocabulary"]
    n = len(cols)
    chips = [_chip(f"{vocab[k]} ({_num(k, lang)}): {c}") for k, c in row.items()]
    outside = sorted({x for x in found if x not in cols})
    vector = w["vector"].format(k=len(row), z=_num(n - len(row), lang), n=_num(n, lang))
    body = _label(vector) + "".join(chips)
    if outside:
        body += _label(w["outside"]) + "".join(_chip(x, _CHIP_OUTSIDE) for x in outside)
    rows = _model()["rows"]
    train = training()
    values = [sparse_cosine(row, r) for r in rows]
    order = [i for i in sorted(range(len(rows)), key=lambda i: (-values[i], i))[:TOP] if values[i] > 0]
    if order:
        table = [[_sentence(train[i][1], train[i][2], lang), _data_label(train[i][0], lang), _dec(values[i], lang, 3),
                  _bar(values[i], 1.0, "var(--ea-data)", 6.0)] for i in order]
        answer = LABELS[lang][train[order[0]][0]]
        body += (_table(table, w["head"], numbers=(2,)) + _label(w["answer"]) + _label_chip(train[order[0]][0], lang))
        alt_rows = "; ".join(f"{_plain(train[i][1], train[i][2], lang)} ({LABELS[lang][train[i][0]]}) "
                             f"{_dec(values[i], lang, 3)}" for i in order)
        answer_alt = f"{w['answer']} {answer}"
    else:
        body += _label(w["none"])
        alt_rows, answer_alt = "", w["none"]
    body += _note(w["note"])
    alt_words = ", ".join(f"{vocab[k]} {c}" for k, c in row.items())
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], vector=vector, words=alt_words,
                                                              rows=alt_rows, answer=answer_alt, note=w["note"]))


# --- the second demo: other vocabularies -------------------------------------------------------------------------------

_EXPLORE_WORDS = {
    "en": {"title": "Your choices against the lesson's",
           "head": ["", "Your choices", "This lesson"],
           "columns": "Columns: words in the vocabulary",
           "mean": "Numbers that are not 0, per sentence",
           "zeros": "Zeros",
           "right": "Right on the 600 held-out sentences",
           "none": "Held-out sentences without a word in common",
           "note": "The same 2,400 training sentences and 600 held-out sentences; each new sentence gets the label of "
                   "its most similar training sentence by the cosine. Blue: the share that is right.",
           "alt": "{title}. {rows}. {note}"},
    "nl": {"title": "Je keuzes tegen de keuzes van de les",
           "head": ["", "Je keuzes", "Deze les"],
           "columns": "Kolommen: woorden in de vocabulary",
           "mean": "Getallen die niet 0 zijn, per zin",
           "zeros": "Nullen",
           "right": "Goed op de 600 achtergehouden zinnen",
           "none": "Achtergehouden zinnen zonder gedeeld woord",
           "note": "Dezelfde 2.400 trainingszinnen en 600 achtergehouden zinnen; elke nieuwe zin krijgt het label van "
                   "de meest gelijkende trainingszin volgens de cosine similarity. Blauw: het deel dat klopt.",
           "alt": "{title}. {rows}. {note}"},
}


def explore_picture(lower, stop, binary, rare, lang: str) -> str:
    w = _EXPLORE_WORDS[lang]
    choice = (bool(lower), bool(stop), bool(binary), bool(rare))
    mine, base = stats(choice), stats()
    s_mine, s_base = score(choice), score()

    def right(s):
        x = 100 * s["right"] / s["total"]
        return f"{_pct(x, lang)} {_bar(x, 100, 'var(--ea-pred)', 6.0)}"
    rows_ = [[w["columns"], _num(mine["columns"], lang), _num(base["columns"], lang)],
             [w["mean"], _dec(mine["mean"], lang, 1), _dec(base["mean"], lang, 1)],
             [w["zeros"], _pct(mine["zeros"], lang, 2), _pct(base["zeros"], lang, 2)],
             [w["right"], right(s_mine), right(s_base)],
             [w["none"], str(s_mine["none"]), str(s_base["none"])]]
    body = _table([[html.escape(r[0])] + r[1:] for r in rows_], w["head"], numbers=(1, 2)) + _note(w["note"])
    alt_rows = "; ".join(f"{r[0]}: {r[1]}, {r[2]}" for r in [
        [w["columns"], _num(mine["columns"], lang), _num(base["columns"], lang)],
        [w["mean"], _dec(mine["mean"], lang, 1), _dec(base["mean"], lang, 1)],
        [w["zeros"], _pct(mine["zeros"], lang, 2), _pct(base["zeros"], lang, 2)],
        [w["right"], _pct(100 * s_mine["right"] / s_mine["total"], lang),
         _pct(100 * s_base["right"] / s_base["total"], lang)],
        [w["none"], str(s_mine["none"]), str(s_base["none"])]])
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows=alt_rows, note=w["note"]))


# --- the exercises -----------------------------------------------------------------------------------------------------

REQUIRED = ["vectorize", "sparse-cosine", "most-similar", "what-comes-out"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: "..." in "\n".join(
                    line.split("#")[0] for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang het door je eigen code.")


def _value(v) -> str:
    """A short value the way the lessons write it: strings in double quotes, also inside a list or a dictionary."""
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_value(p) for p in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_value(k)}: {_value(x)}" for k, x in v.items()) + "}"
    if isinstance(v, float):
        return repr(round(v, 6))
    return repr(v)


def _shown_value(v):
    """A value for a message: a long list of numbers as its length and the places that are not 0, in each language."""
    if isinstance(v, list) and len(v) > 16 and all(_number(x) for x in v):
        places = [(k, x) for k, x in enumerate(v) if x]
        listed = ", ".join(f"[{k}] = {_value(x)}" for k, x in places[:8])
        more = len(places) - 8
        if not places:
            return {"en": f"a list of {len(v):,} numbers, all 0", "nl": f"een lijst van {_num(len(v), 'nl')} getallen, allemaal 0"}
        return {"en": f"a list of {len(v):,} numbers, all 0 except {listed}" + (f" and {more} more" if more > 0 else ""),
                "nl": f"een lijst van {_num(len(v), 'nl')} getallen, allemaal 0 behalve {listed}"
                      + (f" en nog {more}" if more > 0 else "")}
    return _value(v)


class Named:
    """A value of the check's own that a message shows under a name, such as the vocabulary of the lesson."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


def _lit(v) -> Named:
    return Named(_value(v), v)


def _words_value(text: str) -> Named:
    return Named(_value(words(text)), words(text))


def _columns_value() -> Named:
    return Named("columns", _model()["columns"])


def _rows_value() -> Named:
    return Named("rows", _model()["rows"])


def _key_error(e) -> bool:
    return isinstance(e, KeyError)


def _zero_division(e) -> bool:
    return isinstance(e, ZeroDivisionError)


def _a_dict(got, text) -> bool:
    return isinstance(got, dict)


def _a_float(got, text) -> bool:
    return isinstance(got, float)


def _index_error(e) -> bool:
    return isinstance(e, IndexError)


# exercise 1: vectorize. The two examples of the lesson, then: a word that comes twice, a word outside the
# vocabulary, no words, only words outside it, a word three times in a longer sentence, a vocabulary of one
# word, and a sentence over the lesson's
# own vocabulary
SMALL = {"crust": 0, "good": 1, "great": 2, "is": 3, "mic": 4, "not": 5, "prompt": 6, "sad": 7, "service": 8,
         "the": 9, "very": 10, "was": 11}
VECTOR_CASES = [
    (_lit(["the", "mic", "is", "great"]), Named("small", SMALL)),
    (_lit(["crust", "is", "not", "good"]), Named("small", SMALL)),
    (_lit(["very", "very", "sad"]), Named("small", SMALL)),
    (_lit(["the", "food", "is", "great"]), Named("small", SMALL)),
    (_lit([]), Named("small", SMALL)),
    (_lit(["lekker", "eten"]), Named("small", SMALL)),
    (_lit(["was", "the", "service", "very", "very", "very", "good"]), Named("small", SMALL)),
    (_lit(["is", "is"]), _lit({"is": 0})),
    (_words_value("The delivery was on time."), _columns_value()),
]

_V_START = "def vectorize(words, columns):\n    ...\n"
_V = ("def vectorize(words, columns):\n"
      "    vector = [0 for column in columns]\n"
      "    for word in words:\n"
      "        if word in columns:\n"
      "            vector[columns[word]] = vector[columns[word]] + 1\n"
      "    return vector\n")
_V_PRESENT = _V.replace("vector[columns[word]] = vector[columns[word]] + 1", "vector[columns[word]] = 1")
_V_NO_CHECK = ("def vectorize(words, columns):\n"
               "    vector = [0 for column in columns]\n"
               "    for word in words:\n"
               "        vector[columns[word]] = vector[columns[word]] + 1\n"
               "    return vector\n")
_V_WORD_LONG = _V.replace("vector = [0 for column in columns]", "vector = [0 for word in words]")   # an IndexError
_V_DICT = ("def vectorize(words, columns):\n"
           "    vector = {}\n"
           "    for word in words:\n"
           "        if word in columns:\n"
           "            vector[columns[word]] = vector.get(columns[word], 0) + 1\n"
           "    return vector\n")
_V_SHIFT = _V.replace("vector[columns[word]] = vector[columns[word]] + 1",
                      "vector[columns[word] - 1] = vector[columns[word] - 1] + 1")
_V_RETURN_IN = _V.replace("    return vector\n", "        return vector\n")
_V_APPEND = ("def vectorize(words, columns):\n"
             "    vector = []\n"
             "    for word in words:\n"
             "        if word in columns:\n"
             "            vector.append(columns[word])\n"
             "    return vector\n")

# exercise 2: sparse_cosine. Two pairs of the lesson, then: a row with itself, a row against the same row times three,
# no column in common, an empty row, two empty rows, counts above 1 in one row, keys in another order, and two
# sentences over the lesson's own vocabulary
ROW_MIC = {2: 1, 3: 1, 4: 1, 9: 1}          # the mic is great
ROW_CRUST = {0: 1, 1: 1, 3: 1, 5: 1}        # crust is not good
ROW_SAD = {7: 1, 10: 2}                     # very very sad
ROW_SERVICE = {6: 1, 8: 1, 10: 1, 11: 1}    # service was very prompt
COSINE_CASES = [
    (_lit(ROW_MIC), _lit(ROW_CRUST)),
    (_lit(ROW_SAD), _lit(ROW_SERVICE)),
    (_lit(ROW_MIC), _lit(ROW_MIC)),
    (_lit(ROW_SAD), _lit({7: 3, 10: 6})),
    (_lit(ROW_MIC), _lit(ROW_SERVICE)),
    (_lit({}), _lit(ROW_MIC)),
    (_lit(ROW_CRUST), _lit({})),
    (_lit({}), _lit({})),
    (_lit({3: 3, 1: 1}), _lit({1: 2, 3: 1, 9: 4})),
    (_lit(sparse_row(words("The delivery was on time."))),
     _lit(sparse_row(words("The bus boy on the other hand was so rude.")))),
]

_C_START = "def sparse_cosine(a, b):\n    ...\n"
_C = ("def sparse_cosine(a, b):\n"
      "    dot = 0\n"
      "    for column in a:\n"
      "        if column in b:\n"
      "            dot = dot + a[column] * b[column]\n"
      "    if dot == 0:\n"
      "        return 0.0\n"
      "    return dot / (norm(a) * norm(b))\n")
_C_NO_CHECK = ("def sparse_cosine(a, b):\n"
               "    dot = 0\n"
               "    for column in a:\n"
               "        dot = dot + a[column] * b[column]\n"
               "    if dot == 0:\n"
               "        return 0.0\n"
               "    return dot / (norm(a) * norm(b))\n")
_C_NO_ZERO = _C.replace("    if dot == 0:\n        return 0.0\n", "")
_C_DOT = _C.replace("    return dot / (norm(a) * norm(b))\n", "    return dot\n")
_C_LEN = _C.replace("dot / (norm(a) * norm(b))", "dot / (len(a) * len(b))")
_C_SUM = _C.replace("dot / (norm(a) * norm(b))", "dot / (norm(a) + norm(b))")
_C_BRACKETS = _C.replace("dot / (norm(a) * norm(b))", "dot / norm(a) * norm(b)")
_C_ADD = _C.replace("dot = dot + a[column] * b[column]", "dot = dot + a[column] + b[column]")
_C_RETURN_IN = ("def sparse_cosine(a, b):\n"
                "    dot = 0\n"
                "    for column in a:\n"
                "        if column in b:\n"
                "            dot = dot + a[column] * b[column]\n"
                "        if dot == 0:\n"
                "            return 0.0\n"
                "        return dot / (norm(a) * norm(b))\n"
                "    return 0.0\n")

# exercise 3: most_similar. The two examples of the lesson, then: no word in common, two rows that tie, the best row
# last, an empty list of rows, an empty row, and a held-out sentence against the 2,400 training rows
SMALL_ROWS = [ROW_MIC, ROW_CRUST, ROW_SAD, ROW_SERVICE]
SIMILAR_CASES = [
    (_lit({8: 1, 10: 1, 3: 1, 1: 1}), Named("small_rows", SMALL_ROWS)),       # service very is good
    (_lit({7: 1}), Named("small_rows", SMALL_ROWS)),                           # sad
    (_lit({11: 1}), _lit([ROW_MIC, ROW_CRUST])),                               # was: nothing in common
    (_lit({3: 1}), _lit([ROW_SAD, ROW_MIC, ROW_SERVICE, ROW_CRUST])),          # is: a tie of two rows
    (_lit({7: 1, 10: 2}), _lit([ROW_MIC, ROW_SERVICE, ROW_SAD])),              # very very sad: the best one last
    (_lit({3: 1}), _lit([])),
    (_lit({}), Named("small_rows", SMALL_ROWS)),
    (_lit({10: 1}), _lit([ROW_SERVICE, {10: 5, 3: 1}, ROW_SAD])),              # very
    (_lit(sparse_row(words("The delivery was on time."))), _rows_value()),
]

_S_START = "def most_similar(row, rows):\n    ...\n"
_S = ("def most_similar(row, rows):\n"
      "    best = None\n"
      "    best_value = 0\n"
      "    for i in range(len(rows)):\n"
      "        value = cosine(row, rows[i])\n"
      "        if value > best_value:\n"
      "            best = i\n"
      "            best_value = value\n"
      "    return best\n")
_S_TIES = _S.replace("if value > best_value:", "if value >= best_value:")
_S_MINUS = _S.replace("    best_value = 0\n", "    best_value = -1\n")
_S_LEAST = _S.replace("    best_value = 0\n", "    best_value = 2\n").replace("if value > best_value:",
                                                                               "if value < best_value:")
_S_NOT_UPDATED = _S.replace("            best_value = value\n", "")
_S_VALUE = _S.replace("    return best\n", "    return best_value\n")
_S_ROW = _S.replace("            best = i\n", "            best = rows[i]\n")
_S_RETURN_IN = _S.replace("    return best\n", "        return best\n")

_NAMES_C = {"norm": norm}
_NAMES_S = {"cosine": sparse_cosine, "norm": norm}

EXERCISES = {
    "vectorize": FunctionBox(
        name="vectorize",
        picture=vectorize_picture,
        function="vectorize",
        start=_V_START,
        solution=_V,
        visible=VECTOR_CASES[:2],
        hidden=VECTOR_CASES[2:],
        first=[_DOTS],
        show=_shown_value,
        mistakes=[
            Mistake("a-dict", matches=_a_dict,
                    en="`{call}` gives {got}: a dictionary. Here your function gives a list with one number for each column of `columns`, 0 for every word the sentence lacks: {want}.",
                    nl="`{call}` geeft {got}: een dictionary. Hier geeft je functie een lijst met één getal voor elke kolom van `columns`, 0 voor elk woord dat niet in de zin staat: {want}."),
            Mistake("lookup", error=_key_error,
                    en="With `{call}`, your function stops with a `KeyError`: it asks `columns` for the column of a word that `columns` lacks. Ask first whether the word is a key of `columns`, and leave it out when it is not.",
                    nl="Met `{call}` stopt je functie met een `KeyError`: je functie vraagt `columns` om de kolom van een woord dat `columns` niet heeft. Vraag eerst of het woord een key van `columns` is, en laat het weg als dat niet zo is."),
            Mistake("present", wrong=_V_PRESENT,
                    en="`{call}` gives {got}, and it should give {want}: a word that occurs twice gets a 1. Count it: add 1 to its number each time it occurs.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een woord dat twee keer voorkomt, krijgt een 1. Tel het woord: tel 1 op bij het getal van het woord, elke keer dat het voorkomt."),
            Mistake("too-short", error=_index_error,
                    en="With `{call}`, your function stops with an `IndexError`: your list is shorter than the column a word should be counted in. Make it as long as `columns`, with a 0 for each column, so that every sentence gets a list of the same length.",
                    nl="Met `{call}` stopt je functie met een `IndexError`: je lijst is korter dan de kolom waarin een woord geteld moet worden. Maak de lijst zo lang als `columns`, met een 0 voor elke kolom, zodat elke zin een lijst van dezelfde lengte krijgt."),
            Mistake("shifted", wrong=_V_SHIFT,
                    en="`{call}` gives {got}, and it should give {want}: each count lands one place too early. The column of a word is `columns[word]` itself; the first column is 0.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elk aantal komt één plaats te vroeg. De kolom van een woord is `columns[word]` zelf; de eerste kolom is 0."),
            Mistake("appended", wrong=_V_APPEND,
                    en="`{call}` gives {got}, and it should give {want}: a list of the columns of the words. Make a list of zeros as long as `columns` first, and add 1 at the column of each word.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een lijst van de kolommen van de woorden. Maak eerst een lijst nullen zo lang als `columns`, en verhoog het getal op de kolom van elk woord met 1."),
            Mistake("return-in-loop", wrong=_V_RETURN_IN,
                    en="`{call}` gives {got}, and it should give {want}: your function counts only the first word. The `return` stands inside the loop, so the function stops after one round. Move it back, under `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt alleen het eerste woord. De `return` staat in de loop, dus de functie stopt na één ronde. Zet de `return` terug, onder `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: make a list with a 0 for each column of `columns`, and add 1 at the column of each word of `words` that `columns` holds.",
                      "nl": "Vervang de `...`: maak een lijst met een 0 voor elke kolom van `columns`, en verhoog voor elk woord van `words` dat in `columns` staat het getal op de kolom van dat woord met 1."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your list have one number for each column, and does each word add 1 at its own column?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Heeft je lijst één getal voor elke kolom, en telt elk woord 1 bij op de eigen kolom?"},
            "almost": {"en": "Right for the examples of the lesson. But `{call}` gives {got}, and it should give {want}. Does your function work for every list of words: a word twice, words outside the vocabulary, no words at all, another vocabulary?",
                       "nl": "Goed voor de voorbeelden van de les. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke lijst woorden: een woord twee keer, woorden buiten de vocabulary, helemaal geen woorden, een andere vocabulary?"},
            "correct": {"en": "Your `vectorize` is right in all {n} cases. Look at the picture: your vectors of four sentences over the whole vocabulary of the lesson, and a map of your first 60 rows.",
                        "nl": "Je `vectorize` klopt in alle {n} gevallen. Kijk naar de afbeelding: je vectoren van vier zinnen over de hele vocabulary van de les, en een kaart van je eerste 60 rijen."},
        },
    ),
    "sparse-cosine": FunctionBox(
        name="sparse-cosine",
        picture=sparse_cosine_picture,
        function="sparse_cosine",
        start=_C_START,
        solution=_C,
        names=_NAMES_C,
        visible=COSINE_CASES[:2],
        hidden=COSINE_CASES[2:],
        first=[_DOTS],
        show=_value,
        mistakes=[
            Mistake("lookup", error=_key_error,
                    en="With `{call}`, your function stops with a `KeyError`: it asks a row for a column that the row lacks. A column that only one row holds is 0 in the other row, so it adds nothing: ask first whether the other row holds the column, too.",
                    nl="Met `{call}` stopt je functie met een `KeyError`: je functie vraagt een rij om een kolom die de rij niet heeft. Een kolom die maar één rij heeft, is 0 in de andere rij, dus zo'n kolom telt niets op: vraag eerst of de andere rij de kolom ook heeft."),
            Mistake("zero", error=_zero_division,
                    en="With `{call}`, your function stops with a `ZeroDivisionError`: an empty row has length 0. When the rows share no column, the dot product is 0: give back `0.0` then, before you divide.",
                    nl="Met `{call}` stopt je functie met een `ZeroDivisionError`: een lege rij heeft lengte 0. Als de rijen geen kolom delen, is het inproduct 0: geef dan `0.0` terug, voordat je deelt."),
            Mistake("dot-only", wrong=_C_DOT,
                    en="`{call}` gives {got}, and it should give {want}: the dot product itself. The cosine divides it by the lengths of the two rows, `norm(a) * norm(b)`, so that long rows do not win.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het inproduct zelf. De cosine similarity deelt dat door de lengtes van de twee rijen, `norm(a) * norm(b)`, zodat lange rijen niet winnen."),
            Mistake("len", wrong=_C_LEN,
                    en="`{call}` gives {got}, and it should give {want}: your function divides by `len(a) * len(b)`, the number of columns each row holds. The length of a row is `norm(row)`, the square root of the sum of the squares of its counts: for a row with four counts of 1, that is 2, not 4.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie deelt door `len(a) * len(b)`, het aantal kolommen dat elke rij heeft. De lengte van een rij is `norm(row)`, de wortel uit de som van de kwadraten van de aantallen: voor een rij met vier aantallen 1 is dat 2, niet 4."),
            Mistake("sum", wrong=_C_SUM,
                    en="`{call}` gives {got}, and it should give {want}: your function divides by `norm(a) + norm(b)`. The cosine divides by the product of the two lengths.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie deelt door `norm(a) + norm(b)`. De cosine similarity deelt door het product van de twee lengtes."),
            Mistake("brackets", wrong=_C_BRACKETS,
                    en="`{call}` gives {got}, and it should give {want}: Python divides by `norm(a)` first, and then multiplies the outcome by `norm(b)`. Put the product of the two lengths in brackets, so that it is worked out before the division.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: Python deelt eerst door `norm(a)`, en vermenigvuldigt de uitkomst daarna met `norm(b)`. Zet het product van de twee lengtes tussen haakjes, zodat Python dat eerst uitrekent."),
            Mistake("added", wrong=_C_ADD,
                    en="`{call}` gives {got}, and it should give {want}: your function adds the two counts of a shared column. The dot product multiplies them.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt de twee aantallen van een gedeelde kolom op. Het inproduct vermenigvuldigt de twee aantallen."),
            Mistake("return-in-loop", wrong=_C_RETURN_IN,
                    en="`{call}` gives {got}, and it should give {want}: your function looks at the first column of `a` only. The `return` stands inside the loop; move the check for 0 and the division back, under `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie kijkt alleen naar de eerste kolom van `a`. De `return` staat in de loop; zet de controle op 0 en de deling terug, onder `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: add up the products of the two counts for every column that both rows hold, give back `0.0` when that sum is 0, and otherwise divide it by `norm(a) * norm(b)`.",
                      "nl": "Vervang de `...`: tel de producten van de twee aantallen op voor elke kolom die beide rijen hebben, geef `0.0` terug als die som 0 is, en deel de som anders door `norm(a) * norm(b)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function multiply the counts of the columns both rows hold, add them up, and divide by the product of the two lengths?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Vermenigvuldigt je functie de aantallen van de kolommen die beide rijen hebben, telt je functie de producten op, en deelt je functie door het product van de twee lengtes?"},
            "almost": {"en": "Right for the examples of the lesson. But `{call}` gives {got}, and it should give {want}. Does your function work for every pair of rows: the same row twice, nothing in common, an empty row, counts above 1?",
                       "nl": "Goed voor de voorbeelden van de les. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elk paar rijen: twee keer dezelfde rij, niets gemeen, een lege rij, aantallen boven 1?"},
            "unreached": {"en": "With `{call}`, your function gives back `None`, and it should give a number. Does it end with a `return`, after the loop?",
                          "nl": "Met `{call}` geeft je functie `None` terug, en dat moet een getal zijn. Eindigt je functie met een `return`, na de loop?"},
            "correct": {"en": "Your `sparse_cosine` is right in all {n} cases. Look at the picture: the training sentences most like *The delivery was on time.*, by your function.",
                        "nl": "Je `sparse_cosine` klopt in alle {n} gevallen. Kijk naar de afbeelding: de trainingszinnen die het meest op *The delivery was on time.* lijken, volgens je functie."},
        },
    ),
    "most-similar": FunctionBox(
        name="most-similar",
        picture=most_similar_picture,
        function="most_similar",
        start=_S_START,
        solution=_S,
        names=_NAMES_S,
        visible=SIMILAR_CASES[:2],
        hidden=SIMILAR_CASES[2:],
        first=[_DOTS],
        show=_value,
        mistakes=[
            Mistake("a-value", matches=_a_float,
                    en="`{call}` gives {got}: a cosine. Give back the place of the most similar row in `rows`, a whole number, or None: {want}.",
                    nl="`{call}` geeft {got}: een cosine similarity. Geef de plaats van de meest gelijkende rij in `rows` terug, een heel getal, of None: {want}."),
            Mistake("a-row", matches=_a_dict,
                    en="`{call}` gives {got}: the row itself. Give back its place in `rows`, so that the label of that sentence can be looked up: {want}.",
                    nl="`{call}` geeft {got}: de rij zelf. Geef de plaats in `rows` terug, zodat het label van die zin opgezocht kan worden: {want}."),
            Mistake("ties", wrong=_S_TIES,
                    en="`{call}` gives {got}, and it should give {want}: with `>=`, a later row with the same cosine takes the place of an earlier one, and a row with cosine 0 counts too. Keep a row only when its cosine is higher than the best so far.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: met `>=` neemt een latere rij met dezelfde cosine similarity de plaats van een eerdere in, en telt een rij met cosine similarity 0 ook mee. Houd een rij alleen als de cosine similarity hoger is dan de beste tot nu toe."),
            Mistake("from-minus", wrong=_S_MINUS,
                    en="`{call}` gives {got}, and it should give {want}: a row that shares no word with `row` has cosine 0, and your function keeps it. Let `best_value` start at a value that only a row with a word in common can beat.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een rij die geen woord met `row` deelt, heeft cosine similarity 0, en je functie houdt die rij. Laat `best_value` beginnen bij een waarde die alleen een rij met een gedeeld woord kan verslaan."),
            Mistake("least", wrong=_S_LEAST,
                    en="`{call}` gives {got}, and it should give {want}: your function keeps the row with the lowest cosine. Keep a row when its cosine is higher than the best so far.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie houdt de rij met de laagste cosine similarity. Houd een rij als de cosine similarity hoger is dan de beste tot nu toe."),
            Mistake("not-updated", wrong=_S_NOT_UPDATED,
                    en="`{call}` gives {got}, and it should give {want}: your function keeps the last row with a word in common, not the most similar. When you keep a row, does the cosine to beat change too?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie houdt de laatste rij met een gedeeld woord, niet de meest gelijkende. Als je een rij houdt, verandert dan ook de cosine similarity die de volgende rij moet verslaan?"),
            Mistake("return-in-loop", wrong=_S_RETURN_IN,
                    en="`{call}` gives {got}, and it should give {want}: your function looks at the first row only. The `return` stands inside the loop, so the function stops after one round. Move it back, under `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie kijkt alleen naar de eerste rij. De `return` staat in de loop, dus de functie stopt na één ronde. Zet de `return` terug, onder `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: measure the `cosine` of `row` with each row of `rows`, and give back the place of the highest one; None when no cosine is above 0.",
                      "nl": "Vervang de `...`: meet de `cosine` van `row` met elke rij van `rows`, en geef de plaats van de hoogste terug; None als geen cosine similarity boven 0 ligt."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function keep the place of the row with the highest cosine, the first one when two are equal, and only a row with a cosine above 0?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Houdt je functie de plaats van de rij met de hoogste cosine similarity, de eerste als twee gelijk zijn, en alleen een rij met een cosine similarity boven 0?"},
            "almost": {"en": "Right for the examples of the lesson. But `{call}` gives {got}, and it should give {want}. Does your function work for every row and list of rows: nothing in common, two rows that tie, the best row last, no rows at all?",
                       "nl": "Goed voor de voorbeelden van de les. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke rij en elke lijst rijen: niets gemeen, twee rijen met dezelfde cosine similarity, de beste rij als laatste, helemaal geen rijen?"},
            "correct": {"en": "Your `most_similar` is right in all {n} cases. Look at the picture: twelve held-out sentences, and the training sentence your function finds for each.",
                        "nl": "Je `most_similar` klopt in alle {n} gevallen. Kijk naar de afbeelding: twaalf achtergehouden zinnen, en de trainingszin die je functie voor elke zin vindt."},
        },
    ),
    "what-comes-out": Sort(
        name="what-comes-out",
        groups={
            "one": {"en": "1: the same direction", "nl": "1: dezelfde richting"},
            "between": {"en": "Between 0 and 1", "nl": "Tussen 0 en 1"},
            "zero": {"en": "0: no word in common", "nl": "0: geen gedeeld woord"},
        },
        items={
            "order": {"en": "*the food was good* and *good was the food*",
                      "nl": "*the food was good* en *good was the food*"},
            "repeat": {"en": "*great* and *great great great*",
                       "nl": "*great* en *great great great*"},
            "apart": {"en": "*terrible service* and *lovely food*",
                      "nl": "*terrible service* en *lovely food*"},
            "negation": {"en": "*the food is good* and *the food is not good*",
                         "nl": "*the food is good* en *the food is not good*"},
            "dutch": {"en": "*Lekker eten!* (Dutch for *Good food!*) and *the food was good*",
                      "nl": "*Lekker eten!* en *the food was good*"},
            "hated": {"en": "*i loved it* and *i hated it*",
                      "nl": "*i loved it* en *i hated it*"},
        },
        solution={"order": "one", "repeat": "one", "apart": "zero", "negation": "between", "dutch": "zero",
                  "hated": "between"},
        nudge={
            "order": {"en": "Which counts change when the same words come in another order?",
                      "nl": "Welke aantallen veranderen als dezelfde woorden in een andere volgorde staan?"},
            "repeat": {"en": "The second row is the first one times three. Does the cosine look at the length?",
                       "nl": "De tweede rij is drie keer de eerste. Kijkt de cosine similarity naar de lengte?"},
            "apart": {"en": "Which column do the two rows share?",
                      "nl": "Welke kolom delen de twee rijen?"},
            "negation": {"en": "Four of the five words are the same. Does the cosine know what *not* does?",
                         "nl": "Vier van de vijf woorden zijn hetzelfde. Weet de cosine similarity wat *not* doet?"},
            "dutch": {"en": "Which words of *Lekker eten!* does the vocabulary of 2,400 English sentences hold?",
                      "nl": "Welke woorden van *Lekker eten!* staan in de vocabulary van 2.400 Engelse zinnen?"},
            "hated": {"en": "Two of the three words are the same. Is that all, or nothing?",
                      "nl": "Twee van de drie woorden zijn hetzelfde. Is dat alles, of niets?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the pairs again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de paren opnieuw uit."},
            "empty": {"en": "Put each pair under the cosine of its two rows, over the vocabulary of this lesson.",
                      "nl": "Zet elk paar bij de cosine similarity van de twee rijen, over de vocabulary van deze les."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item}: this pair goes somewhere else.",
                      "nl": "{item}: dit paar hoort ergens anders."},
            "correct": {"en": "All {n} are right. A row of counts keeps which words a text has and how often, and nothing else: another order or the same text three times gives the same direction. The cosine of *the food is good* and *the food is not good* is 0.894, and *i loved it* and *i hated it* still share two of their three words, though they say the opposite. A word outside the vocabulary adds nothing, so *Lekker eten!* becomes a row without a single count.",
                        "nl": "Alle {n} kloppen. Een rij aantallen bewaart welke woorden een tekst heeft en hoe vaak, en verder niets: een andere volgorde of drie keer dezelfde tekst geeft dezelfde richting. De cosine similarity van *the food is good* en *the food is not good* is 0,894, en *i loved it* en *i hated it* delen nog twee van de drie woorden, hoewel de twee zinnen het tegenovergestelde zeggen. Een woord buiten de vocabulary telt niets op, dus *Lekker eten!* wordt een rij zonder een enkel aantal."},
        },
    ),
}

SORT_PAIRS = {"order": ("the food was good", "good was the food"), "repeat": ("great", "great great great"),
              "apart": ("terrible service", "lovely food"), "negation": ("the food is good", "the food is not good"),
              "dutch": ("Lekker eten!", "the food was good"), "hated": ("i loved it", "i hated it")}

CASES = [
    ("vectorize", _V_START, "empty", None),
    ("vectorize", _V, "correct", None),
    ("vectorize", "def vectorize(words, columns):\n    vector = [0] * len(columns)\n    for word in words:\n        if word in columns:\n            vector[columns[word]] += 1\n    return vector\n", "correct", None),
    ("vectorize", "def vectorize(words, columns):\n    vector = []\n    for column in columns:\n        vector.append(0)\n    for word in words:\n        if word in columns:\n            k = columns[word]\n            vector[k] = vector[k] + 1\n    return vector\n", "correct", None),
    ("vectorize", _V_DICT, "wrong", "a-dict"),
    ("vectorize", _V_NO_CHECK, "error", "lookup"),
    ("vectorize", _V_PRESENT, "wrong", "present"),
    ("vectorize", _V_WORD_LONG, "error", "too-short"),
    ("vectorize", _V_SHIFT, "wrong", "shifted"),
    ("vectorize", _V_APPEND, "wrong", "appended"),
    ("vectorize", _V_RETURN_IN, "wrong", "return-in-loop"),
    ("vectorize", _V.replace("    return vector\n", ""), "wrong", "no-return"),
    ("vectorize", "def vectorize(words, columns):\n    if words[0] == \"the\":\n        return [0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]\n    return [1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]\n", "almost", None),
    ("sparse-cosine", _C_START, "empty", None),
    ("sparse-cosine", _C, "correct", None),
    ("sparse-cosine", "def sparse_cosine(a, b):\n    dot = sum(a[c] * b[c] for c in a if c in b)\n    if dot == 0:\n        return 0.0\n    return dot / (norm(a) * norm(b))\n", "correct", None),
    ("sparse-cosine", "import math\n\ndef sparse_cosine(a, b):\n    dot = 0\n    for column in a:\n        if column in b:\n            dot += a[column] * b[column]\n    la = math.sqrt(sum(x * x for x in a.values()))\n    lb = math.sqrt(sum(x * x for x in b.values()))\n    if la == 0 or lb == 0:\n        return 0.0\n    return dot / (la * lb)\n", "correct", None),
    ("sparse-cosine", _C_NO_CHECK, "error", "lookup"),
    ("sparse-cosine", _C_NO_ZERO, "error", "zero"),
    ("sparse-cosine", _C_DOT, "wrong", "dot-only"),
    ("sparse-cosine", _C_LEN, "wrong", "len"),
    ("sparse-cosine", _C_SUM, "wrong", "sum"),
    ("sparse-cosine", _C_BRACKETS, "wrong", "brackets"),
    ("sparse-cosine", _C_ADD, "wrong", "added"),
    ("sparse-cosine", _C_RETURN_IN, "wrong", "return-in-loop"),
    ("sparse-cosine", _C.replace("    return dot / (norm(a) * norm(b))\n", ""), "wrong", "unreached"),
    ("sparse-cosine", "def sparse_cosine(a, b):\n    if 9 in a:\n        return 0.25\n    return 0.4472135954999579\n", "almost", None),
    ("most-similar", _S_START, "empty", None),
    ("most-similar", _S, "correct", None),
    ("most-similar", "def most_similar(row, rows):\n    best = None\n    for i in range(len(rows)):\n        if cosine(row, rows[i]) > 0:\n            if best is None or cosine(row, rows[i]) > cosine(row, rows[best]):\n                best = i\n    return best\n", "correct", None),
    ("most-similar", "def most_similar(row, rows):\n    values = [cosine(row, r) for r in rows]\n    if not values or max(values) == 0:\n        return None\n    return values.index(max(values))\n", "correct", None),
    ("most-similar", _S_VALUE, "wrong", "a-value"),
    ("most-similar", _S_ROW, "wrong", "a-row"),
    ("most-similar", _S_TIES, "wrong", "ties"),
    ("most-similar", _S_MINUS, "wrong", "from-minus"),
    ("most-similar", _S_LEAST, "wrong", "least"),
    ("most-similar", _S_NOT_UPDATED, "wrong", "not-updated"),
    ("most-similar", _S_RETURN_IN, "wrong", "return-in-loop"),
    ("most-similar", "def most_similar(row, rows):\n    if 8 in row:\n        return 1\n    return 2\n", "almost", None),
    ("what-comes-out", {}, "empty", None),
    ("what-comes-out", {"order": "one", "repeat": "one", "apart": "zero", "negation": "between", "dutch": "zero",
                        "hated": "between"}, "correct", None),
    ("what-comes-out", {"order": "one", "apart": "zero"}, "wrong", "unfinished"),
    ("what-comes-out", {"negation": "one"}, "wrong", "wrong:negation"),
    ("what-comes-out", {"order": "one", "repeat": "between", "apart": "zero", "negation": "between", "dutch": "zero",
                        "hated": "between"}, "wrong", "wrong:repeat"),
]
