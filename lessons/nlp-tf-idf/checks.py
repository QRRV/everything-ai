# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-tf-idf: count in how many documents each word occurs (document_frequency), weigh the
counts of one document by TF-IDF (tf_idf), and score a document for a search by adding up the weights of the search
words (score); and sort words by their TF-IDF weight.

The documents are the 251 chapters, stories and letters of the five English books of the course: Project Gutenberg's
*Alice's Adventures in Wonderland*, *Pride and Prejudice*, *The Adventures of Sherlock Holmes*, *Frankenstein* and
*Moby-Dick* (data/gutenberg-books.json.gz, made by data/tools/make_gutenberg.py; everything_ai.books cuts them into
chapters and words). The small documents, the cases of the checks and the pictures are the course's own."""
import collections
import contextlib
import html
import io
import math

import numpy as np

from everything_ai import books, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, kept, with_time_limit

# --- the chapters ------------------------------------------------------------------------------------------------------

BOOK_NAMES = {"alice": "Alice", "pride": "Pride and Prejudice", "holmes": "Sherlock Holmes",
              "frankenstein": "Frankenstein", "moby-dick": "Moby-Dick"}
KIND_NAMES = {
    "en": {"chapter": "chapter", "story": "story", "letter": "letter", "etymology": "Etymology",
           "extracts": "Extracts", "epilogue": "Epilogue"},
    "nl": {"chapter": "hoofdstuk", "story": "verhaal", "letter": "brief", "etymology": "etymologie",
           "extracts": "citaten", "epilogue": "epiloog"},
}
TEA_PARTY = 6   # the place of chapter 7 of Alice, A Mad Tea-Party, among the 251 chapters
PIG = 5         # chapter 6 of Alice, Pig and Pepper


def words(text: str) -> list:
    """The words of a text in small letters, as the course cuts the books: runs of letters, with an apostrophe inside a
    word as in *don't*."""
    return books.words(text.lower())


def _chapters() -> list:
    """(key, kind, number, title, words in small letters) for each of the 251 chapters, in the order of the books."""
    return kept("nlp-tf-idf:chapters", lambda: [(key, kind, number, title, tuple(w.lower() for w in ws))
                                                for key, kind, number, title, ws in books.english_chapters()])


def chapter_words() -> list:
    """The 251 chapters as lists of words in small letters, a new list each time."""
    return [list(c[4]) for c in _chapters()]


def chapter_label(i: int, lang: str = "en") -> str:
    """The name of chapter i: its book, its kind and number, and its title when the book gives one."""
    key, kind, number, title, _ = _chapters()[i]
    head = f"{BOOK_NAMES[key]}, {KIND_NAMES[lang][kind]}"
    if number:
        head += f" {number}"
    return f"{head}: {title}" if title else head


def chapter_options(lang: str) -> dict:
    """The chapters for the dropdown of the second demo: name -> place."""
    return {chapter_label(i, lang): i for i in range(len(_chapters()))}


# --- the course's solutions --------------------------------------------------------------------------------------------

def document_frequency(docs):
    """The course's solution of Exercise 1: a dictionary from each word to the number of documents it occurs in; a word
    that occurs several times in one document counts once for that document."""
    df = {}
    for doc in docs:
        for word in set(doc):
            if word in df:
                df[word] = df[word] + 1
            else:
                df[word] = 1
    return df


def tf_idf(counts, df, n):
    """The course's solution of Exercise 2: the weight of each word of one document, its count times log10(n / df)."""
    weights = {}
    for word in counts:
        weights[word] = counts[word] * np.log10(n / df[word])
    return weights


def score(query, weights):
    """The course's solution of Exercise 3: the sum of the weights of the search words that the document holds."""
    total = 0
    for word in query:
        if word in weights:
            total = total + weights[word]
    return total


# --- the course's own document frequencies, counts and weights ---------------------------------------------------------

def _df() -> dict:
    return kept("nlp-tf-idf:df", lambda: document_frequency(c[4] for c in _chapters()))


def _counts() -> list:
    return kept("nlp-tf-idf:counts", lambda: [dict(collections.Counter(c[4])) for c in _chapters()])


def _weights() -> list:
    def make():
        df, n = _df(), len(_chapters())
        return [{w: float(x) for w, x in tf_idf(c, df, n).items()} for c in _counts()]
    return kept("nlp-tf-idf:weights", make)


def chapter_df() -> dict:
    """The document frequency of every word of the 251 chapters, as a new dict, the most common words first."""
    return dict(sorted(_df().items(), key=lambda t: (-t[1], t[0])))


def chapter_counts(i: int) -> dict:
    """The count of each word of chapter i, as a new dict, the most frequent first."""
    return dict(sorted(_counts()[i].items(), key=lambda t: (-t[1], t[0])))


def chapter_weights() -> list:
    """The TF-IDF weights of the 251 chapters, as a new list of dicts in the order of the chapters."""
    return [dict(w) for w in _weights()]


def idf(word: str) -> float:
    """log10(251 / df) of a word of the chapters."""
    return math.log10(len(_chapters()) / _df()[word])


def count_score(query: list, i: int) -> int:
    """How often the search words occur in chapter i together: the score without weights."""
    return sum(_counts()[i].get(w, 0) for w in query)


def search(query: list, top: int = 5, counts: bool = False) -> list:
    """The places of the chapters with the highest score for a search, the first chapter first when two are equal; by
    TF-IDF, or by plain counts."""
    n = len(_chapters())
    values = [count_score(query, i) if counts else score(query, _weights()[i]) for i in range(n)]
    return sorted(range(n), key=lambda i: (-values[i], i))[:top]


def book_df() -> dict:
    """The number of the five books each word occurs in."""
    def make():
        found = collections.defaultdict(set)
        for key, _, _, _, ws in _chapters():
            for w in ws:
                found[w].add(key)
        return {w: len(k) for w, k in found.items()}
    return kept("nlp-tf-idf:book-df", make)


def top_words(i: int, use_idf: bool = True, log_tf: bool = False, top: int = 10) -> list:
    """The words of chapter i with the highest weight, as (word, count, df, weight): the count, or 1 + log10 of it, and
    times the IDF or not. Equal weights by the alphabet."""
    n = len(_chapters())
    out = []
    for w, c in _counts()[i].items():
        tf = 1 + math.log10(c) if log_tf else c
        out.append((w, c, _df()[w], tf * math.log10(n / _df()[w]) if use_idf else tf))
    return sorted(out, key=lambda t: (-t[3], t[0]))[:top]


# --- the small documents -----------------------------------------------------------------------------------------------

SMALL_DOCS = [["the", "hatter", "poured", "the", "tea"], ["the", "queen", "shouted"], ["the", "dormouse", "slept"],
              ["tea", "tea", "tea", "said", "the", "hatter"]]
SMALL_DF = {"the": 4, "hatter": 2, "poured": 1, "tea": 2, "queen": 1, "shouted": 1, "dormouse": 1, "slept": 1,
            "said": 1}
SMALL_COUNTS = [dict(collections.Counter(d)) for d in SMALL_DOCS]
SMALL_WEIGHTS = [{w: float(x) for w, x in tf_idf(c, SMALL_DF, 4).items()} for c in SMALL_COUNTS]


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
_CHIP_OUTSIDE = _CHIP + ";border:1px dashed var(--ea-ink-3);color:var(--ea-ink-2)"
_BAR = "display:inline-block;height:0.6rem;border-radius:2px;vertical-align:middle"
_TEXT = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)"


def _marked(text: str) -> str:
    parts = html.escape(text).split("*")
    return "".join(f"<i>{p}</i>" if k % 2 else p for k, p in enumerate(parts))


def _note(text: str) -> str:
    """A line under a picture; a word between asterisks is set in italics, as in the lesson's text."""
    return f'<div style="{_UI}">{_marked(text)}</div>'


def _label(text: str) -> str:
    return f'<div style="{_LABEL}">{_marked(text)}</div>'


def _num(n, lang: str) -> str:
    """A whole number with a separator for thousands: 22,858 in English, 22.858 in Dutch."""
    text = f"{round(n):,}"
    return text.replace(",", ".") if lang == "nl" else text


def _dec(x: float, lang: str, digits: int = 2) -> str:
    """A number with `digits` decimals: 0.52 in English, 0,52 in Dutch."""
    text = f"{x + 0.0:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def _pct(x: float, lang: str, digits: int = 1) -> str:
    return _dec(x, lang, digits) + "%"


def _idf(v: float, lang: str) -> str:
    """An IDF with three decimals, or four when it is below 0.01, so that a count times it gives the weight shown."""
    return _dec(v, lang, 4 if 0 < v < 0.01 else 3)


def _chip(text: str, style: str = _CHIP_DATA) -> str:
    return f'<span style="{style}">{html.escape(text)}</span>'


def _bar(x: float, most: float, color: str = "var(--ea-data)", width: float = 8.0) -> str:
    """A bar for a number: as long as `width` for `most`, never shorter than a sliver."""
    w = 0 if not most or not x or x <= 0 else max(0.15, width * x / most)
    return f'<span style="{_BAR};width:{w:.2f}rem;background:{color}"></span>' if w else ""


def _table(rows_: list, head: list, numbers: tuple = ()) -> str:
    """A small table; rows hold HTML that is already escaped; numbers are the columns set to the right."""
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers else ""}">{c}</td>'
                                    for j, c in enumerate(row)) + "</tr>" for row in rows_)
    head_html = f"<thead><tr>{top}</tr></thead>" if head else ""
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem">'
            f"{head_html}<tbody>{body}</tbody></table></div>")


def _title(i: int, lang: str) -> str:
    return f'<span style="{_TEXT}">{html.escape(chapter_label(i, lang))}</span>'


# --- the idea's picture: the IDF of some words of the books -------------------------------------------------------------

IDEA_WORDS = ["steelkilt", "hatter", "cat", "alice", "ice", "ship", "life", "the"]

_IDEA_TEXT = {
    "en": {"title": "The inverse document frequency of some words of the books",
           "x": "Chapters with the word (document frequency)",
           "y": "IDF",
           "note": "Gray dots: words of the books, at the number of the 251 chapters they occur in and their IDF. The "
                   "line: log10(251 / df) for every document frequency from 1 to 251. From 1 to 10 to 100 chapters, each "
                   "step is ten times as many chapters, and the IDF goes down by 1 for each.",
           "alt": "{title}. {words}. {note}"},
    "nl": {"title": "De inverse document frequency van een paar woorden uit de boeken",
           "x": "Hoofdstukken met het woord (document frequency)",
           "y": "IDF",
           "note": "Grijze stippen: woorden uit de boeken, bij het aantal van de 251 hoofdstukken waarin elk woord "
                   "voorkomt en de IDF. De lijn: log10(251 / df) voor elke *document frequency* van 1 tot 251. Van 1 naar "
                   "10 naar 100 hoofdstukken is elke stap tien keer zoveel hoofdstukken, en de IDF gaat bij elke stap 1 "
                   "omlaag.",
           "alt": "{title}. {words}. {note}"},
}


def idea_picture(lang: str) -> str:
    w = _IDEA_TEXT[lang]
    n = len(_chapters())
    left, right, top, bottom = 56, 572, 18, 206

    def x(df):
        return left + (right - left) * math.log10(df) / math.log10(n)

    def y(v):
        return bottom - (bottom - top) * v / 2.5
    font = 'font-family="var(--ea-font-ui)" font-size="12"'
    parts = [f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="var(--ea-axis)" />',
             f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="var(--ea-axis)" />']
    for df in (1, 10, 100, n):
        parts.append(f'<line x1="{x(df):.1f}" y1="{bottom}" x2="{x(df):.1f}" y2="{bottom + 4}" stroke="var(--ea-axis)" />'
                     f'<text x="{x(df):.1f}" y="{bottom + 17}" text-anchor="middle" fill="var(--ea-ink-2)" {font}>'
                     f"{_num(df, lang)}</text>")
    for v in (0, 0.5, 1, 1.5, 2):
        parts.append(f'<text x="{left - 6}" y="{y(v) + 4:.1f}" text-anchor="end" fill="var(--ea-ink-2)" {font}>'
                     f"{_dec(v, lang, 1)}</text>")
    parts.append(f'<text x="{(left + right) / 2:.0f}" y="{bottom + 34}" text-anchor="middle" fill="var(--ea-ink-2)" '
                 f"{font}>{html.escape(w['x'])}</text>")
    parts.append(f'<text x="14" y="{(top + bottom) / 2:.0f}" text-anchor="middle" fill="var(--ea-ink-2)" {font} '
                 f'transform="rotate(-90 14 {(top + bottom) / 2:.0f})">{w["y"]}</text>')
    parts.append(f'<line x1="{x(1):.1f}" y1="{y(math.log10(n)):.1f}" x2="{x(n):.1f}" y2="{y(0):.1f}" '
                 f'stroke="var(--ea-ink-3)" stroke-width="1.5" />')
    alts = []
    for word in IDEA_WORDS:
        df = _df()[word]
        v = idf(word)
        parts.append(f'<circle cx="{x(df):.1f}" cy="{y(v):.1f}" r="4" fill="var(--ea-data)" />'
                     f'<text x="{x(df) + 7:.1f}" y="{y(v) - 7:.1f}" text-anchor="start" fill="var(--ea-ink)" '
                     f'font-family="var(--ea-font-code)" font-size="12">{html.escape(word)}</text>')
        alts.append(f"{word} {_num(df, lang)}, {_dec(v, lang, 2)}")
    svg = (f'<svg viewBox="0 0 612 250" width="100%" style="max-width:612px;display:block;margin:0.3rem 0" '
           f'role="img" aria-hidden="true">{"".join(parts)}</svg>')
    return yours.figure(w["title"], svg + _note(w["note"]),
                        alt=w["alt"].format(title=w["title"], words="; ".join(alts), note=w["note"].replace("*", "")))


# --- running the learner's function ------------------------------------------------------------------------------------

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
    "en": {"stopped": "Your function stopped with an error on the chapters, so the picture stays empty.",
           "slow": "Your function took more than {s:g} seconds for the chapters, so the picture stays empty."},
    "nl": {"stopped": "Je functie stopte met een fout op de hoofdstukken, dus de afbeelding blijft leeg.",
           "slow": "Je functie deed er langer dan {s:g} seconden over voor de hoofdstukken, dus de afbeelding blijft leeg."},
}


def _number(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def _whole(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


# --- the picture under exercise 1: your document frequencies -----------------------------------------------------------

DF_WORDS = ["the", "said", "you", "whale", "ship", "ice", "tea", "alice", "holmes", "cat", "hatter", "steelkilt"]
BUCKETS = [(1, 1), (2, 3), (4, 10), (11, 30), (31, 100), (101, 250), (251, 251)]

_DF_TEXT = {
    "en": {"title": "Your document frequencies, over the 251 chapters",
           "head": ["Word", "Your df", "IDF from it", ""],
           "buckets": ["In 1 chapter", "In 2 to 3", "In 4 to 10", "In 11 to 30", "In 31 to 100", "In 101 to 250",
                       "In all 251"],
           "spread": "Your dictionary has {k} words. How many of them occur in how many chapters:",
           "note": "Gray: the document frequencies your function counted, with the IDF each gives, log10(251 / df), "
                   "and the number of words for each range of document frequencies. Purple: a number that should be "
                   "another one; dashed: the right number.",
           "over": "In more than 251",
           "odd": "Your function gave something other than a dictionary from words to whole numbers of 1 or more, so "
                  "the picture stays empty.",
           "alt": "{title}. {rows}. {spread} {buckets}. {note}"},
    "nl": {"title": "Je document frequencies, over de 251 hoofdstukken",
           "head": ["Woord", "Je df", "IDF daaruit", ""],
           "buckets": ["In 1 hoofdstuk", "In 2 tot 3", "In 4 tot 10", "In 11 tot 30", "In 31 tot 100", "In 101 tot 250",
                       "In alle 251"],
           "spread": "Je dictionary heeft {k} woorden. Hoeveel daarvan in hoeveel hoofdstukken voorkomen:",
           "note": "Grijs: de *document frequencies* die je functie telde, met de IDF die elk geeft, log10(251 / df), en "
                   "het aantal woorden voor elk bereik van *document frequencies*. Paars: een getal dat een ander getal "
                   "moet zijn; met een gestreepte rand: het goede getal.",
           "over": "In meer dan 251",
           "odd": "Je functie gaf iets anders dan een dictionary van woorden naar hele getallen van 1 of meer, dus de "
                  "afbeelding blijft leeg.",
           "alt": "{title}. {rows}. {spread} {buckets}. {note}"},
}


def document_frequency_picture(seen) -> str:
    lang = seen.lang
    w = _DF_TEXT[lang] | _SAY[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    n = len(_chapters())

    def ok(g):
        return isinstance(g, dict) and all(isinstance(k, str) and _whole(v) and v >= 1 for k, v in g.items())
    got, said = _run_or_say(lambda: fn(chapter_words()), w, ok)
    if said:
        return said
    right = _df()
    rows_, alts = [], []
    for word in DF_WORDS:
        mine, want = got.get(word), right[word]
        if mine == want:
            shown = _chip(_num(mine, lang))
        elif mine is None:
            shown = _chip(_num(want, lang), _CHIP_MISSING)
        else:
            shown = _chip(_num(mine, lang), _CHIP_PURPLE) + _chip(_num(want, lang), _CHIP_MISSING)
        value = math.log10(n / mine) if mine else None
        rows_.append([_chip(word), shown, "" if value is None else _dec(value, lang, 3),
                      _bar(value or 0, math.log10(n), "var(--ea-data)", 7.0)])
        alts.append(f"{word} {'–' if mine is None else _num(mine, lang)}"
                    + ("" if value is None else f", {_dec(value, lang, 3)}"))
    counted = [sum(1 for v in got.values() if a <= v <= b) for a, b in BUCKETS]
    most = max(counted) or 1
    spread_rows = [[html.escape(name), _num(k, lang), _bar(k, most, "var(--ea-data)", 12.0)]
                   for name, k in zip(w["buckets"], counted)]
    over = sum(1 for v in got.values() if v > n)   # a word in more documents than there are
    if over:
        spread_rows.append([html.escape(w["over"]), _chip(_num(over, lang), _CHIP_PURPLE), ""])
    spread = w["spread"].format(k=_num(len(got), lang))
    body = (_table(rows_, w["head"], numbers=(1, 2)) + _label(spread) + _table(spread_rows, [], numbers=(1,))
            + _note(w["note"]))
    buckets = "; ".join(f"{name}: {_num(k, lang)}" for name, k in zip(w["buckets"] + [w["over"]], counted + [over])
                        if k or name != w["over"])
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts), spread=spread,
                                                              buckets=buckets, note=w["note"].replace("*", "")))


# --- the picture under exercise 2: your weights for A Mad Tea-Party ----------------------------------------------------

TOP = 10

_WEIGHT_TEXT = {
    "en": {"title": "Your weights for Alice, chapter 7: A Mad Tea-Party",
           "head": ["", "By count", "Count", "By your weight", "Your weight", ""],
           "the": "*the*, {k} times in the chapter: your weight {v}.",
           "note": "Gray: the ten words with the highest count in the chapter, and the ten with the highest weight by "
                   "your function. Purple: a weight that should be another number; dashed: the right one.",
           "odd": "Your function gave something other than a dictionary with a number for each word of the chapter, so "
                  "the picture stays empty.",
           "alt": "{title}. {rows}. {the} {note}"},
    "nl": {"title": "Je waarden voor Alice, hoofdstuk 7: A Mad Tea-Party",
           "head": ["", "Op aantal", "Aantal", "Op je waarde", "Je waarde", ""],
           "the": "*the*, {k} keer in het hoofdstuk: je waarde {v}.",
           "note": "Grijs: de tien woorden met het hoogste aantal in het hoofdstuk, en de tien met de hoogste waarde "
                   "volgens je functie. Paars: een waarde die een ander getal moet zijn; met een gestreepte rand: de "
                   "goede waarde.",
           "odd": "Je functie gaf iets anders dan een dictionary met een getal voor elk woord van het hoofdstuk, dus de "
                  "afbeelding blijft leeg.",
           "alt": "{title}. {rows}. {the} {note}"},
}


def tf_idf_picture(seen) -> str:
    lang = seen.lang
    w = _WEIGHT_TEXT[lang] | _SAY[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    counts = _counts()[TEA_PARTY]
    n = len(_chapters())

    def ok(g):
        return isinstance(g, dict) and set(g) == set(counts) and all(_number(v) and math.isfinite(v) for v in g.values())
    got, said = _run_or_say(lambda: fn(chapter_counts(TEA_PARTY), chapter_df(), n), w, ok)
    if said:
        return said
    got = {k: float(v) for k, v in got.items()}
    right = _weights()[TEA_PARTY]
    by_count = sorted(counts, key=lambda x: (-counts[x], x))[:TOP]
    by_weight = sorted(got, key=lambda x: (-got[x], x))[:TOP]
    most = max(max(got[x] for x in by_weight), 1e-9)
    rows_, alts = [], []
    for k, (a, b) in enumerate(zip(by_count, by_weight)):
        if math.isclose(got[b], right[b], rel_tol=1e-6, abs_tol=1e-9):
            value = _chip(_dec(got[b], lang, 2))
        else:
            value = _chip(_dec(got[b], lang, 2), _CHIP_PURPLE) + _chip(_dec(right[b], lang, 2), _CHIP_MISSING)
        rows_.append([str(k + 1), _chip(a), _num(counts[a], lang), _chip(b), value,
                      _bar(got[b], most, "var(--ea-data)", 6.0)])
        alts.append(f"{k + 1}: {a} {counts[a]}, {b} {_dec(got[b], lang, 2)}")
    the = w["the"].format(k=counts["the"], v=_dec(got["the"], lang, 2))
    body = _table(rows_, w["head"], numbers=(0, 2, 4)) + _label(the) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts),
                                                              the=the.replace("*", ""), note=w["note"]))


# --- the picture under exercise 3: three searches ----------------------------------------------------------------------

SEARCHES = ["the cheshire cat", "who stole the tarts", "the red headed league"]
SEARCH_TOP = 3

_SCORE_TEXT = {
    "en": {"title": "Your score, three searches of the 251 chapters",
           "head": ["", "By plain counts", "Count", "By your score", "Your score"],
           "search": "Search: *{q}*",
           "note": "Gray: the chapters where the search words occur most often together, and the chapters with the "
                   "highest score by your function. Blue: the chapter your score puts first. Purple: a score that should be "
                   "another number; dashed: the right one.",
           "odd": "Your function gave something other than a number, so the picture stays empty.",
           "alt": "{title}. {rows}. {note}"},
    "nl": {"title": "Je score, drie zoekopdrachten in de 251 hoofdstukken",
           "head": ["", "Op kale aantallen", "Aantal", "Op je score", "Je score"],
           "search": "Zoekopdracht: *{q}*",
           "note": "Grijs: de hoofdstukken waarin de zoekwoorden samen het vaakst voorkomen, en de hoofdstukken met de "
                   "hoogste score volgens je functie. Blauw: het hoofdstuk dat je score bovenaan zet. Paars: een score die "
                   "een ander getal moet zijn; met een gestreepte rand: de goede score.",
           "odd": "Je functie gaf iets anders dan een getal, dus de afbeelding blijft leeg.",
           "alt": "{title}. {rows}. {note}"},
}


def score_picture(seen) -> str:
    lang = seen.lang
    w = _SCORE_TEXT[lang] | _SAY[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    weights = _weights()
    n = len(weights)

    def make():
        return [[fn(words(q), dict(weights[i])) for i in range(n)] for q in SEARCHES]
    got, said = _run_or_say(make, w, lambda g: all(_number(v) and math.isfinite(v) for row in g for v in row))
    if said:
        return said
    body, alts = "", []
    for q, values in zip(SEARCHES, got):
        values = [float(v) for v in values]
        mine = sorted(range(n), key=lambda i: (-values[i], i))[:SEARCH_TOP]
        plain = search(words(q), SEARCH_TOP, counts=True)
        rows_ = []
        for k, (a, b) in enumerate(zip(plain, mine)):
            label_b = _chip(chapter_label(b, lang), _CHIP_PRED) if k == 0 else _title(b, lang)
            right = score(words(q), weights[b])
            if math.isclose(values[b], right, rel_tol=1e-6, abs_tol=1e-9):
                value = _dec(values[b], lang, 2)
            else:
                value = _chip(_dec(values[b], lang, 2), _CHIP_PURPLE) + _chip(_dec(right, lang, 2), _CHIP_MISSING)
            rows_.append([str(k + 1), _title(a, lang), _num(count_score(words(q), a), lang), label_b, value])
        body += _label(w["search"].format(q=q)) + _table(rows_, w["head"], numbers=(0, 2, 4))
        alts.append(f"{q}: " + ", ".join(chapter_label(a, lang) for a in plain) + "; "
                    + ", ".join(f"{chapter_label(b, lang)} {_dec(values[b], lang, 2)}" for b in mine))
    body += _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts), note=w["note"]))


# --- the demo: search the books ----------------------------------------------------------------------------------------

DEMO_QUERY = "a ship in the ice"
DEMO_LONGEST = 200   # characters of the search that the program looks at

_DEMO_TEXT = {
    "en": {"title": "Your search, over the 251 chapters",
           "words": "Your search words, each once, with the number of chapters each occurs in and its IDF:",
           "nowhere": "in no chapter",
           "head": ["Chapter", "Score", "", "Weight of each search word: count × IDF"],
           "plain": "By plain counts, the first chapter would be {label}: the search words occur {k} times in it.",
           "none": "No chapter gets a score above 0: every search word is in no chapter, or in all of them.",
           "empty": "Type a search first.",
           "note": "Gray: the IDF of each search word, the scores and the weights in each chapter. Dashed: a word in no "
                   "chapter. Blue: the chapter with the highest score, the answer to the search.",
           "alt": "{title}. {words}. {rows}. {plain} {note}"},
    "nl": {"title": "Je zoekopdracht, over de 251 hoofdstukken",
           "words": "Je zoekwoorden, elk één keer, met het aantal hoofdstukken waarin elk woord voorkomt en de IDF:",
           "nowhere": "in geen hoofdstuk",
           "head": ["Hoofdstuk", "Score", "", "Waarde van elk zoekwoord: aantal × IDF"],
           "plain": "Op kale aantallen zou {label} bovenaan staan: de zoekwoorden komen er samen {k} keer in voor.",
           "none": "Geen hoofdstuk krijgt een score boven 0: elk zoekwoord staat in geen enkel hoofdstuk, of in alle.",
           "empty": "Typ eerst een zoekopdracht.",
           "note": "Grijs: de IDF van elk zoekwoord, de scores en de waarden in elk hoofdstuk. Met een gestreepte rand: "
                   "een woord dat in geen hoofdstuk staat. Blauw: het hoofdstuk met de hoogste score, het antwoord op de "
                   "zoekopdracht.",
           "alt": "{title}. {words}. {rows}. {plain} {note}"},
}


def demo_picture(text, lang: str) -> str:
    w = _DEMO_TEXT[lang]
    query = list(dict.fromkeys(words((text or "")[:DEMO_LONGEST])))   # each search word once
    if not query:
        return yours.figure(w["title"], _note(w["empty"]), alt=w["empty"])
    df = _df()
    seen_words = query
    chips, word_alts = [], []
    for q in seen_words:
        if q in df:
            chips.append(_chip(f"{q}: {_num(df[q], lang)}, {_idf(idf(q), lang)}"))
            word_alts.append(f"{q} {_num(df[q], lang)}, {_idf(idf(q), lang)}")
        else:
            chips.append(_chip(f"{q}: {w['nowhere']}", _CHIP_OUTSIDE))
            word_alts.append(f"{q}: {w['nowhere']}")
    body = _label(w["words"]) + "".join(chips)
    weights = _weights()
    values = [score(query, weights[i]) for i in range(len(weights))]
    order = [i for i in search(query, 5) if values[i] > 0]
    if order:
        most = values[order[0]]
        rows_ = []
        for k, i in enumerate(order):
            parts = [_chip(f"{q}: {_counts()[i][q]} × {_idf(idf(q), lang)} = {_dec(weights[i][q], lang, 2)}")
                     for q in seen_words if q in weights[i] and weights[i][q] > 0]
            name = _chip(chapter_label(i, lang), _CHIP_PRED) if k == 0 else _title(i, lang)
            rows_.append([name, _dec(values[i], lang, 2), _bar(values[i], most, "var(--ea-data)", 6.0), "".join(parts)])
        body += _table(rows_, w["head"], numbers=(1,))
        alt_rows = "; ".join(f"{chapter_label(i, lang)} {_dec(values[i], lang, 2)}" for i in order)
    else:
        body += _label(w["none"])
        alt_rows = w["none"]
    first = search(query, 1, counts=True)[0]
    k = count_score(query, first)
    plain = w["plain"].format(label=chapter_label(first, lang), k=_num(k, lang)) if k else ""
    if plain:
        body += _label(plain)
    body += _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], words="; ".join(word_alts),
                                                              rows=alt_rows, plain=plain, note=w["note"]))


# --- the second demo: other weights ------------------------------------------------------------------------------------

_EXPLORE_TEXT = {
    "en": {"title": "The ten words with the highest weight in a chapter",
           "head": ["", "Word", "Count", "In chapters", "Weight", ""],
           "formula": {(True, False): "Weight: count × log10(251 / df)",
                       (True, True): "Weight: (1 + log10 of the count) × log10(251 / df)",
                       (False, False): "Weight: the count itself",
                       (False, True): "Weight: 1 + log10 of the count"},
           "note": "Gray: the words of the chapter with the highest weight, each with its count in the chapter and the "
                   "number of the 251 chapters it occurs in.",
           "alt": "{title}. {chapter}. {formula}. {rows}. {note}"},
    "nl": {"title": "De tien woorden met de hoogste waarde in een hoofdstuk",
           "head": ["", "Woord", "Aantal", "In hoofdstukken", "Waarde", ""],
           "formula": {(True, False): "Waarde: aantal × log10(251 / df)",
                       (True, True): "Waarde: (1 + log10 van het aantal) × log10(251 / df)",
                       (False, False): "Waarde: het aantal zelf",
                       (False, True): "Waarde: 1 + log10 van het aantal"},
           "note": "Grijs: de woorden van het hoofdstuk met de hoogste waarde, elk met het aantal in het hoofdstuk en "
                   "het aantal van de 251 hoofdstukken waarin het woord voorkomt.",
           "alt": "{title}. {chapter}. {formula}. {rows}. {note}"},
}


def explore_picture(index, use_idf, log_tf, lang: str) -> str:
    w = _EXPLORE_TEXT[lang]
    i = int(index) if index is not None else TEA_PARTY
    found = top_words(i, bool(use_idf), bool(log_tf))
    most = max(found[0][3], 1e-9)
    digits = 2 if use_idf or log_tf else 0
    rows_ = [[str(k + 1), _chip(word), _num(c, lang), _num(df, lang),
              _dec(v, lang, digits) if digits else _num(v, lang), _bar(v, most, "var(--ea-data)", 6.0)]
             for k, (word, c, df, v) in enumerate(found)]
    formula = w["formula"][(bool(use_idf), bool(log_tf))]
    body = _label(chapter_label(i, lang)) + _label(formula) + _table(rows_, w["head"], numbers=(0, 2, 3, 4)) \
        + _note(w["note"])
    alt_rows = "; ".join(f"{word} {_dec(v, lang, digits) if digits else _num(v, lang)}" for word, _, _, v in found)
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], chapter=chapter_label(i, lang),
                                                              formula=formula, rows=alt_rows, note=w["note"]))


# --- the exercises -----------------------------------------------------------------------------------------------------

REQUIRED = ["document-frequency", "tf-idf", "score", "what-weight"]

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

PROBES = ["the", "said", "hatter", "tea", "whale"]   # words a long dictionary is shown by


def _value(v) -> str:
    """A short value the way the lessons write it: strings in double quotes, numbers rounded to six decimals."""
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_value(p) for p in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_value(k)}: {_value(x)}" for k, x in v.items()) + "}"
    if isinstance(v, (float, np.floating)):
        return repr(round(float(v), 6) + 0.0)
    if isinstance(v, np.integer):
        return repr(int(v))
    return repr(v)


def _shown(v):
    """A value for a message: a dictionary of whole numbers in alphabetical order, and a long dictionary as its length
    and a few of its words, in each language."""
    if isinstance(v, dict) and len(v) > 12:
        shown = [k for k in PROBES if k in v] or sorted(v, key=str)[:4]
        listed = ", ".join(f"{_value(k)}: {_value(v[k])}" for k in shown)
        return {"en": f"a dictionary of {len(v):,} words, with {listed}",
                "nl": f"een dictionary van {_num(len(v), 'nl')} woorden, met {listed}"}
    if isinstance(v, dict) and v and all(_whole(x) for x in v.values()):
        return _value(dict(sorted(v.items(), key=lambda t: str(t[0]))))
    return _value(v)


class Named:
    """A value of the check's own that a message shows under a name, such as the chapters of the lesson."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value, memo)


class _ChapterWords(Named):
    """The 251 chapters, copied fast: a new list for each chapter, with the same strings, which cannot change."""

    def __deepcopy__(self, memo):
        return [list(d) for d in self.value]


def _lit(v) -> Named:
    return Named(_value(v), v)


def _key_error(e) -> bool:
    return isinstance(e, KeyError)


def _unhashable(e) -> bool:
    return isinstance(e, TypeError) and "unhashable" in str(e)


def _split(e) -> bool:
    return isinstance(e, AttributeError) and "split" in str(e)


def _times_dict(e) -> bool:
    return isinstance(e, TypeError) and "dict" in str(e) and ("*" in str(e) or "multiply" in str(e))


def _times_text(e) -> bool:
    return isinstance(e, TypeError) and "can't multiply sequence" in str(e)


DF_LIMIT = 10.0   # seconds for document_frequency, which goes over the 251 chapters in one of its cases


# exercise 1: document_frequency. The two examples of the lesson, then: no documents, an empty document, a word in
# every document, the same document twice, and the 251 chapters
DF_CASES = [
    (Named("small_docs", SMALL_DOCS),),
    (_lit([["whale", "whale", "whale"], ["white", "whale"], ["ship"]]),),
    (_lit([]),),
    (_lit([[], ["tea"]]),),
    (_lit([["a", "cat"], ["a", "dog", "a"], ["a"]]),),
    (_lit([["mad", "hatter"], ["mad", "hatter"]]),),
    (_ChapterWords("chapters", [list(c[4]) for c in _chapters()]),),
]

_D_START = "def document_frequency(docs):\n    ...\n"
_D = ("def document_frequency(docs):\n"
      "    df = {}\n"
      "    for doc in docs:\n"
      "        for word in set(doc):\n"
      "            if word in df:\n"
      "                df[word] = df[word] + 1\n"
      "            else:\n"
      "                df[word] = 1\n"
      "    return df\n")
_D_EVERY = _D.replace("for word in set(doc):", "for word in doc:")
_D_NO_CHECK = ("def document_frequency(docs):\n"
               "    df = {}\n"
               "    for doc in docs:\n"
               "        for word in set(doc):\n"
               "            df[word] = df[word] + 1\n"
               "    return df\n")
_D_ONE = _D.replace("df[word] = df[word] + 1", "df[word] = 1")
_D_SET_DOCS = _D.replace("for doc in docs:\n        for word in set(doc):",
                         "for doc in set(docs):\n        for word in set(doc):")
_D_RESET = ("def document_frequency(docs):\n"
            "    for doc in docs:\n"
            "        df = {}\n"
            "        for word in set(doc):\n"
            "            if word in df:\n"
            "                df[word] = df[word] + 1\n"
            "            else:\n"
            "                df[word] = 1\n"
            "    return df\n")
_D_RETURN_IN = _D.replace("    return df\n", "        return df\n")

# exercise 2: tf_idf. The two examples of the lesson, then: no words, a word in every document, another n, and
# chapter 7 of Alice over the 251 chapters
WEIGHT_CASES = [
    (_lit(SMALL_COUNTS[0]), Named("small_df", SMALL_DF), 4),
    (_lit(SMALL_COUNTS[3]), Named("small_df", SMALL_DF), 4),
    (_lit({}), Named("small_df", SMALL_DF), 4),
    (_lit({"the": 5}), Named("small_df", SMALL_DF), 4),
    (_lit({"tea": 2, "queen": 1}), Named("small_df", SMALL_DF), 8),
    (Named("counts", chapter_counts(TEA_PARTY)), Named("df", chapter_df()), 251),
]

_T_START = "def tf_idf(counts, df, n):\n    ...\n"
_T = ("def tf_idf(counts, df, n):\n"
      "    weights = {}\n"
      "    for word in counts:\n"
      "        weights[word] = counts[word] * np.log10(n / df[word])\n"
      "    return weights\n")
_T_LN = _T.replace("np.log10(n / df[word])", "np.log(n / df[word])")
_T_UPSIDE = _T.replace("np.log10(n / df[word])", "np.log10(df[word] / n)")
_T_NO_LOG = _T.replace("np.log10(n / df[word])", "n / df[word]")
_T_COUNT = _T.replace("counts[word] * np.log10(n / df[word])", "counts[word]")
_T_IDF = _T.replace("counts[word] * np.log10(n / df[word])", "np.log10(n / df[word])")
_T_LOG_TF = _T.replace("counts[word] * np.log10(n / df[word])",
                       "(1 + np.log10(counts[word])) * np.log10(n / df[word])")
_T_PLUS = _T.replace("counts[word] * np.log10", "counts[word] + np.log10")
_T_FLOOR = _T.replace("np.log10(n / df[word])", "np.log10(n // df[word])")
_T_LOG_N = _T.replace("np.log10(n / df[word])", "np.log10(n) / df[word]")
_T_RETURN_IN = _T.replace("    return weights\n", "        return weights\n")

# exercise 3: score. The two examples of the lesson, then: a search word the document lacks, no search words, a word
# with weight 0, a mix of the two, and the cheshire cat against chapter 6 of Alice
SCORE_CASES = [
    (_lit(["hatter", "tea"]), Named("small_weights[3]", SMALL_WEIGHTS[3])),
    (_lit(["the", "queen"]), Named("small_weights[1]", SMALL_WEIGHTS[1])),
    (_lit(["cat"]), Named("small_weights[0]", SMALL_WEIGHTS[0])),
    (_lit([]), Named("small_weights[0]", SMALL_WEIGHTS[0])),
    (_lit(["the"]), Named("small_weights[2]", SMALL_WEIGHTS[2])),
    (_lit(["poured", "tea", "dormouse"]), Named("small_weights[0]", SMALL_WEIGHTS[0])),
    (_lit(["the", "cheshire", "cat"]), Named(f"weights[{PIG}]", _weights()[PIG])),
]

_S_START = "def score(query, weights):\n    ...\n"
_S = ("def score(query, weights):\n"
      "    total = 0\n"
      "    for word in query:\n"
      "        if word in weights:\n"
      "            total = total + weights[word]\n"
      "    return total\n")
_S_NO_CHECK = ("def score(query, weights):\n"
               "    total = 0\n"
               "    for word in query:\n"
               "        total = total + weights[word]\n"
               "    return total\n")
_S_COUNTED = _S.replace("total = total + weights[word]", "total = total + 1")
_S_HIGHEST = _S.replace("total = total + weights[word]", "total = max(total, weights[word])")
_S_ALL = _S.replace("for word in query:", "for word in weights:")
_S_RETURN_IN = _S.replace("    return total\n", "        return total\n")

_NAMES_T = {"np": np, "math": math}

EXERCISES = {
    "document-frequency": FunctionBox(
        name="document-frequency",
        picture=document_frequency_picture,
        function="document_frequency",
        start=_D_START,
        solution=_D,
        visible=DF_CASES[:2],
        hidden=DF_CASES[2:],
        first=[_DOTS],
        show=_shown,
        mistakes=[
            Mistake("lookup", error=_key_error,
                    en="With `{call}`, your function stops with a `KeyError`: it adds 1 to the number of a word that `df` does not hold yet. Ask first whether `df` holds the word, and put in 1 when it does not.",
                    nl="Met `{call}` stopt je functie met een `KeyError`: je functie telt 1 op bij het getal van een woord dat nog niet in `df` staat. Vraag eerst of het woord in `df` staat, en zet er 1 neer als dat niet zo is."),
            Mistake("set-of-docs", error=_unhashable,
                    en="With `{call}`, your function stops with a `TypeError`: a list cannot be a key of a dictionary, and cannot go into a set. `docs` is a list of documents, and each document is a list of words: make sure that each `word` is one word of one document, not a whole document.",
                    nl="Met `{call}` stopt je functie met een `TypeError`: een lijst kan geen key van een dictionary zijn, en kan niet in een set. `docs` is een lijst documenten, en elk document is een lijst woorden: zorg dat elk `word` één woord van één document is, geen heel document."),
            Mistake("split", error=_split,
                    en="With `{call}`, your function stops with an `AttributeError`: a list has no `split`. Each document is already a list of words, so there is nothing to split: go over its words.",
                    nl="Met `{call}` stopt je functie met een `AttributeError`: een lijst heeft geen `split`. Elk document is al een lijst woorden, dus er valt niets te splitsen: ga de woorden langs."),
            Mistake("every-time", wrong=_D_EVERY,
                    en="`{call}` gives {got}, and it should give {want}: your function counts every time a word occurs, not the documents it occurs in. A word that occurs three times in one document adds 1 for that document. Go over the different words of each document, each once.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt elke keer dat een woord voorkomt, niet de documenten waarin het woord voorkomt. Een woord dat drie keer in één document staat, telt 1 voor dat document. Ga per document de verschillende woorden langs, elk één keer."),
            Mistake("always-one", wrong=_D_ONE,
                    en="`{call}` gives {got}, and it should give {want}: every word gets 1. When `df` holds the word already, add 1 to its number.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elk woord krijgt 1. Als het woord al in `df` staat, tel dan 1 op bij het getal van het woord."),
            Mistake("reset", wrong=_D_RESET,
                    en="`{call}` gives {got}, and it should give {want}: only the words of the last document. Your `df` starts empty again for each document; make it once, before the loop over the documents.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: alleen de woorden van het laatste document. Je `df` begint voor elk document weer leeg; maak `df` één keer, vóór de loop over de documenten."),
            Mistake("return-in-loop", wrong=_D_RETURN_IN,
                    en="`{call}` gives {got}, and it should give {want}: your function looks at the first document only. The `return` stands inside the loop over the documents; move it back, under the first `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie kijkt alleen naar het eerste document. De `return` staat in de loop over de documenten; zet de `return` terug, onder de eerste `for`."),
        ],
        time_limit=DF_LIMIT,
        texts=_SAME_TEXTS | {
            "slow": {"en": "Your function is still not done after {s:g} seconds. With the 251 chapters, it has more than half a million words to go over. Does it go over each document once, and search through a whole document only when it has to?".format(s=DF_LIMIT),
                     "nl": "Je functie is na {s:g} seconden nog niet klaar. Met de 251 hoofdstukken heeft je functie meer dan een half miljoen woorden langs te gaan. Gaat je functie elk document één keer langs, en zoekt je functie alleen door een heel document als dat nodig is?".format(s=DF_LIMIT)},
            "empty": {"en": "Replace the `...`: make an empty dictionary, and for each document, add 1 to the number of each different word of that document, or put in 1 for a word the dictionary does not hold yet.",
                      "nl": "Vervang de `...`: maak een lege dictionary, en tel voor elk document 1 op bij het getal van elk verschillend woord van dat document, of zet er 1 neer voor een woord dat nog niet in de dictionary staat."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does each document add 1 for each of its different words, however often a word occurs in it?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Telt elk document 1 op voor elk van de verschillende woorden, hoe vaak een woord er ook in staat?"},
            "almost": {"en": "Right for the examples of the lesson. But `{call}` gives {got}, and it should give {want}. Does your function work for every list of documents: no documents at all, an empty document, the same document twice, the 251 chapters?",
                       "nl": "Goed voor de voorbeelden van de les. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke lijst documenten: helemaal geen documenten, een leeg document, twee keer hetzelfde document, de 251 hoofdstukken?"},
            "unreached": {"en": "With `{call}`, your function gives back `None`, and it should give a dictionary. Does it end with a `return`, after the loops?",
                          "nl": "Met `{call}` geeft je functie `None` terug, en dat moet een dictionary zijn. Eindigt je functie met een `return`, na de loops?"},
            "correct": {"en": "Your `document_frequency` is right in all {n} cases. Look at the picture: the document frequencies your function counted over the 251 chapters.",
                        "nl": "Je `document_frequency` klopt in alle {n} gevallen. Kijk naar de afbeelding: de *document frequencies* die je functie over de 251 hoofdstukken telde."},
        },
    ),
    "tf-idf": FunctionBox(
        name="tf-idf",
        picture=tf_idf_picture,
        function="tf_idf",
        start=_T_START,
        solution=_T,
        names=_NAMES_T,
        visible=WEIGHT_CASES[:2],
        hidden=WEIGHT_CASES[2:],
        first=[_DOTS],
        show=_shown,
        mistakes=[
            Mistake("natural-log", wrong=_T_LN,
                    en="`{call}` gives {got}, and it should give {want}: your function takes `np.log`, the natural log. This lesson takes the log with base 10, `np.log10`, as Jurafsky and Martin and Manning and colleagues do. The order of the words stays the same; the numbers differ.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie neemt `np.log`, de natuurlijke logaritme. Deze les neemt de logaritme met grondtal 10, `np.log10`, zoals Jurafsky en Martin en Manning en collega's doen. De volgorde van de woorden blijft hetzelfde; de getallen verschillen."),
            Mistake("upside-down", wrong=_T_UPSIDE,
                    en="`{call}` gives {got}, and it should give {want}: negative weights. Your function divides the document frequency by `n`; the IDF divides `n` by the document frequency, so that a rare word gets the highest weight.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: negatieve waarden. Je functie deelt de *document frequency* door `n`; de IDF deelt `n` door de *document frequency*, zodat een zeldzaam woord de hoogste waarde krijgt."),
            Mistake("no-log", wrong=_T_NO_LOG,
                    en="`{call}` gives {got}, and it should give {want}: your function multiplies by `n / df[word]` itself. The IDF is the log with base 10 of it, so that a word in one chapter of 251 gets 2.4 and not 251.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie vermenigvuldigt met `n / df[word]` zelf. De IDF is de logaritme met grondtal 10 daarvan, zodat een woord in één hoofdstuk van de 251 2,4 krijgt en niet 251."),
            Mistake("floor", wrong=_T_FLOOR,
                    en="`{call}` gives {got}, and it should give {want}: with `//`, Python rounds `n / df[word]` down to a whole number. For a word in 176 of 251 documents, that makes 1, and the log of 1 is 0. Divide with `/`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: met `//` rondt Python `n / df[word]` af naar beneden, op een heel getal. Voor een woord in 176 van de 251 documenten wordt dat 1, en de logaritme van 1 is 0. Deel met `/`."),
            Mistake("count-only", wrong=_T_COUNT,
                    en="`{call}` gives {got}, and it should give {want}: the counts themselves. Multiply each count by the IDF of its word, `np.log10(n / df[word])`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de aantallen zelf. Vermenigvuldig elk aantal met de IDF van het woord, `np.log10(n / df[word])`."),
            Mistake("idf-only", wrong=_T_IDF,
                    en="`{call}` gives {got}, and it should give {want}: the IDF alone. A word that occurs three times in the document should weigh three times as much: multiply the IDF by the count of the word.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: alleen de IDF. Een woord dat drie keer in het document staat, moet drie keer zo zwaar tellen: vermenigvuldig de IDF met het aantal van het woord."),
            Mistake("log-tf", wrong=_T_LOG_TF,
                    en="`{call}` gives {got}, and it should give {want}: your function takes 1 + log10 of the count, the term frequency of Jurafsky and Martin. This exercise takes the count itself, as Manning and colleagues define TF-IDF; the lesson comes back to the other form.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie neemt 1 + log10 van het aantal, de term frequency van Jurafsky en Martin. Deze opdracht neemt het aantal zelf, zoals Manning en collega's TF-IDF definiëren; de les komt terug op de andere vorm."),
            Mistake("log-of-n", wrong=_T_LOG_N,
                    en="`{call}` gives {got}, and it should give {want}: your function takes the log of `n` and divides that by the document frequency. The IDF is the log of the quotient: divide first, inside `np.log10( )`, and take the log of `n / df[word]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie neemt de logaritme van `n` en deelt die door de *document frequency*. De IDF is de logaritme van het quotiënt: deel eerst, binnen `np.log10( )`, en neem de logaritme van `n / df[word]`."),
            Mistake("whole-dict", error=_times_dict,
                    en="With `{call}`, your function stops with a `TypeError`: it multiplies the whole dictionary `counts` by a number. Multiply by the count of this one word instead.",
                    nl="Met `{call}` stopt je functie met een `TypeError`: je functie vermenigvuldigt de hele dictionary `counts` met een getal. Vermenigvuldig in plaats daarvan met het aantal van dit ene woord."),
            Mistake("word-times", error=_times_text,
                    en="With `{call}`, your function stops with a `TypeError`: it multiplies a text by a number. `word` is the word itself; multiply by how often it occurs in the document, its count.",
                    nl="Met `{call}` stopt je functie met een `TypeError`: je functie vermenigvuldigt een tekst met een getal. `word` is het woord zelf; vermenigvuldig met hoe vaak het in het document voorkomt, het aantal."),
            Mistake("plus", wrong=_T_PLUS,
                    en="`{call}` gives {got}, and it should give {want}: your function adds the count and the IDF. TF-IDF multiplies them, so that a word in every document gets 0, however often it occurs.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt het aantal en de IDF op. TF-IDF vermenigvuldigt de twee, zodat een woord in elk document 0 krijgt, hoe vaak het woord ook voorkomt."),
            Mistake("return-in-loop", wrong=_T_RETURN_IN,
                    en="`{call}` gives {got}, and it should give {want}: your function weighs the first word only. The `return` stands inside the loop, so the function stops after one round. Move it back, under `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie weegt alleen het eerste woord. De `return` staat in de loop, dus de functie stopt na één ronde. Zet de `return` terug, onder `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: make an empty dictionary, and give each word of `counts` its count times `np.log10(n / df[word])`.",
                      "nl": "Vervang de `...`: maak een lege dictionary, en geef elk woord van `counts` het aantal maal `np.log10(n / df[word])`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function give each word of `counts` its count times the log with base 10 of the quotient `n / df[word]`?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Geeft je functie elk woord van `counts` het aantal maal de logaritme met grondtal 10 van het quotiënt `n / df[word]`?"},
            "almost": {"en": "Right for the examples of the lesson. But `{call}` gives {got}, and it should give {want}. Does your function work for every document: no words at all, a word in every document, another `n`, a whole chapter?",
                       "nl": "Goed voor de voorbeelden van de les. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elk document: helemaal geen woorden, een woord in elk document, een andere `n`, een heel hoofdstuk?"},
            "unreached": {"en": "With `{call}`, your function gives back `None`, and it should give a dictionary. Does it end with a `return`, after the loop?",
                          "nl": "Met `{call}` geeft je functie `None` terug, en dat moet een dictionary zijn. Eindigt je functie met een `return`, na de loop?"},
            "correct": {"en": "Your `tf_idf` is right in all {n} cases. Look at the picture: the ten words of *A Mad Tea-Party* with the highest count, and the ten with the highest weight by your function.",
                        "nl": "Je `tf_idf` klopt in alle {n} gevallen. Kijk naar de afbeelding: de tien woorden van *A Mad Tea-Party* met het hoogste aantal, en de tien met de hoogste waarde volgens je functie."},
        },
    ),
    "score": FunctionBox(
        name="score",
        picture=score_picture,
        function="score",
        start=_S_START,
        solution=_S,
        visible=SCORE_CASES[:2],
        hidden=SCORE_CASES[2:],
        first=[_DOTS],
        show=_shown,
        mistakes=[
            Mistake("lookup", error=_key_error,
                    en="With `{call}`, your function stops with a `KeyError`: it asks for the weight of a word that the document lacks. Such a word adds nothing to the score: ask first whether the document has a weight for the word.",
                    nl="Met `{call}` stopt je functie met een `KeyError`: je functie vraagt de waarde van een woord dat niet in het document staat. Zo'n woord telt niets op bij de score: vraag eerst of het document een waarde voor het woord heeft."),
            Mistake("counted", wrong=_S_COUNTED,
                    en="`{call}` gives {got}, and it should give {want}: your function counts the search words that the document holds. Add up their weights instead.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt de zoekwoorden die in het document staan. Tel in plaats daarvan de waarden van die woorden op."),
            Mistake("highest", wrong=_S_HIGHEST,
                    en="`{call}` gives {got}, and it should give {want}: your function keeps the highest weight. The score adds up the weights of all the search words.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie houdt de hoogste waarde. De score telt de waarden van alle zoekwoorden op."),
            Mistake("all-words", wrong=_S_ALL,
                    en="`{call}` gives {got}, and it should give {want}: your function adds up the weights of every word of the document, not only those of the search words. Go over the words of `query`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt de waarden van elk woord van het document op, niet alleen die van de zoekwoorden. Ga de woorden van `query` langs."),
            Mistake("return-in-loop", wrong=_S_RETURN_IN,
                    en="`{call}` gives {got}, and it should give {want}: your function looks at the first search word only. The `return` stands inside the loop, so the function stops after one round. Move it back, under `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie kijkt alleen naar het eerste zoekwoord. De `return` staat in de loop, dus de functie stopt na één ronde. Zet de `return` terug, onder `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: start a total at 0, add the weight of each search word for which the document has a weight, and give back the total.",
                      "nl": "Vervang de `...`: begin een totaal bij 0, tel de waarde op van elk zoekwoord waarvoor het document een waarde heeft, en geef het totaal terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function add up the weights of the search words, and skip a search word that the document lacks?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Telt je functie de waarden van de zoekwoorden op, en slaat je functie een zoekwoord over dat niet in het document staat?"},
            "almost": {"en": "Right for the examples of the lesson. But `{call}` gives {got}, and it should give {want}. Does your function work for every search: a word the document lacks, no search words at all, a word with weight 0?",
                       "nl": "Goed voor de voorbeelden van de les. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke zoekopdracht: een woord dat niet in het document staat, helemaal geen zoekwoorden, een woord met waarde 0?"},
            "unreached": {"en": "With `{call}`, your function gives back `None`, and it should give a number. Does it end with a `return`, after the loop?",
                          "nl": "Met `{call}` geeft je functie `None` terug, en dat moet een getal zijn. Eindigt je functie met een `return`, na de loop?"},
            "correct": {"en": "Your `score` is right in all {n} cases. Look at the picture: three searches of the 251 chapters, by plain counts and by your score.",
                        "nl": "Je `score` klopt in alle {n} gevallen. Kijk naar de afbeelding: drie zoekopdrachten in de 251 hoofdstukken, op kale aantallen en op je score."},
        },
    ),
    "what-weight": Sort(
        name="what-weight",
        groups={
            "zero": {"en": "0", "nl": "0"},
            "below": {"en": "Between 0 and 1", "nl": "Tussen 0 en 1"},
            "above": {"en": "More than 1", "nl": "Meer dan 1"},
        },
        items={
            "of": {"en": "*of*: 35 times in a chapter; in all 251 chapters",
                   "nl": "*of*: 35 keer in een hoofdstuk; in alle 251 hoofdstukken"},
            "cat": {"en": "*cat*: not in a chapter; in 8 other chapters",
                    "nl": "*cat*: niet in een hoofdstuk; in 8 andere hoofdstukken"},
            "whale": {"en": "*whale*: once in a chapter; in 110 chapters",
                      "nl": "*whale*: één keer in een hoofdstuk; in 110 hoofdstukken"},
            "ship": {"en": "*ship*: twice in a chapter; in 99 chapters",
                     "nl": "*ship*: twee keer in een hoofdstuk; in 99 hoofdstukken"},
            "hatter": {"en": "*hatter*: once in a chapter; in 3 chapters",
                       "nl": "*hatter*: één keer in een hoofdstuk; in 3 hoofdstukken"},
            "you": {"en": "*you*: 50 times in a chapter; in 211 chapters",
                    "nl": "*you*: 50 keer in een hoofdstuk; in 211 hoofdstukken"},
        },
        solution={"of": "zero", "cat": "zero", "whale": "below", "ship": "below", "hatter": "above", "you": "above"},
        nudge={
            "of": {"en": "What does 251 / 251 give, and what is its log?",
                   "nl": "Wat geeft 251 / 251, en wat is de logaritme daarvan?"},
            "cat": {"en": "What is the count of a word that the chapter lacks, and what does any IDF times that give?",
                    "nl": "Wat is het aantal van een woord dat niet in het hoofdstuk staat, en wat geeft een IDF maal dat aantal?"},
            "whale": {"en": "Is 251 / 110 more or less than 10, and what does that say about its log with base 10?",
                      "nl": "Is 251 / 110 meer of minder dan 10, en wat zegt dat over de logaritme met grondtal 10?"},
            "ship": {"en": "251 / 99 is about 2.5, and 2.5 squared is still below 10. What does twice its log with base 10 give?",
                     "nl": "251 / 99 is ongeveer 2,5, en 2,5 in het kwadraat is nog onder de 10. Wat geeft twee keer de logaritme met grondtal 10 daarvan?"},
            "hatter": {"en": "Is 251 / 3 more or less than 10?",
                       "nl": "Is 251 / 3 meer of minder dan 10?"},
            "you": {"en": "The IDF of this word is small, but how many times does the chapter have it?",
                    "nl": "De IDF van dit woord is klein, maar hoe vaak staat het woord in het hoofdstuk?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the words again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de woorden opnieuw uit."},
            "empty": {"en": "Put each word under its TF-IDF weight in its chapter: its count times log10(251 / df).",
                      "nl": "Zet elk woord bij de TF-IDF-waarde in het eigen hoofdstuk: het aantal maal log10(251 / df)."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item}: this word goes somewhere else.",
                      "nl": "{item}: dit woord hoort ergens anders."},
            "correct": {"en": "All {n} are right. A word in every chapter gets 0 however often it occurs, and a word that the chapter lacks gets 0 however rare it is. The log with base 10 is below 1 as long as 251 / df is below 10: *whale* gets 0.358, and *ship*, twice, 0.808. *hatter*, in 3 chapters, gets 1.92 from a single time. And *you* is in 211 of the 251 chapters, so each time adds only 0.075, but 50 times make 3.77: a high count can outweigh a low IDF.",
                        "nl": "Alle {n} kloppen. Een woord in elk hoofdstuk krijgt 0, hoe vaak het woord ook voorkomt, en een woord dat niet in het hoofdstuk staat, krijgt 0, hoe zeldzaam het woord ook is. De logaritme met grondtal 10 is onder 1 zolang 251 / df onder 10 is: *whale* krijgt 0,358, en *ship*, twee keer, 0,808. *hatter*, in 3 hoofdstukken, krijgt 1,92 voor één keer. En *you* staat in 211 van de 251 hoofdstukken, dus elke keer telt maar 0,075, maar 50 keer maakt 3,77: een hoog aantal kan een lage IDF overtreffen."},
        },
    ),
}

SORT_WORDS = {"of": (35, 251), "cat": (0, 8), "whale": (1, 110), "ship": (2, 99), "hatter": (1, 3), "you": (50, 211)}

CASES = [
    ("document-frequency", _D_START, "empty", None),
    ("document-frequency", _D, "correct", None),
    ("document-frequency", "def document_frequency(docs):\n    df = {}\n    for doc in docs:\n        for word in set(doc):\n            df[word] = df.get(word, 0) + 1\n    return df\n", "correct", None),
    ("document-frequency", "def document_frequency(docs):\n    df = {}\n    for doc in docs:\n        seen = []\n        for word in doc:\n            if word not in seen:\n                seen.append(word)\n                if word in df:\n                    df[word] += 1\n                else:\n                    df[word] = 1\n    return df\n", "correct", None),
    ("document-frequency", _D_NO_CHECK, "error", "lookup"),
    ("document-frequency", _D_SET_DOCS, "error", "set-of-docs"),
    ("document-frequency", _D.replace("set(doc)", "docs"), "error", "set-of-docs"),
    ("document-frequency", _D.replace("set(doc)", "set(doc.split())"), "error", "split"),
    ("document-frequency", _D_EVERY, "wrong", "every-time"),
    ("document-frequency", _D_ONE, "wrong", "always-one"),
    ("document-frequency", _D_RESET, "wrong", "reset"),
    ("document-frequency", _D_RETURN_IN, "wrong", "return-in-loop"),
    ("document-frequency", _D.replace("    return df\n", ""), "wrong", "no-return"),
    ("document-frequency", "def document_frequency(docs):\n    if len(docs) == 4:\n        return {\"the\": 4, \"hatter\": 2, \"poured\": 1, \"tea\": 2, \"queen\": 1, \"shouted\": 1, \"dormouse\": 1, \"slept\": 1, \"said\": 1}\n    return {\"whale\": 2, \"white\": 1, \"ship\": 1}\n", "almost", None),
    ("tf-idf", _T_START, "empty", None),
    ("tf-idf", _T, "correct", None),
    ("tf-idf", "def tf_idf(counts, df, n):\n    return {word: counts[word] * np.log10(n / df[word]) for word in counts}\n", "correct", None),
    ("tf-idf", "import math\n\ndef tf_idf(counts, df, n):\n    weights = {}\n    for word, count in counts.items():\n        weights[word] = count * math.log10(n / df[word])\n    return weights\n", "correct", None),
    ("tf-idf", _T.replace("np.log10", "math.log10"), "correct", None),
    ("tf-idf", _T_LOG_N, "wrong", "log-of-n"),
    ("tf-idf", _T.replace("counts[word] * np", "counts * np"), "error", "whole-dict"),
    ("tf-idf", _T.replace("counts[word] * np", "word * np"), "error", "word-times"),
    ("tf-idf", _T_LN, "wrong", "natural-log"),
    ("tf-idf", _T_UPSIDE, "wrong", "upside-down"),
    ("tf-idf", _T_NO_LOG, "wrong", "no-log"),
    ("tf-idf", _T_FLOOR, "wrong", "floor"),
    ("tf-idf", _T_COUNT, "wrong", "count-only"),
    ("tf-idf", _T_IDF, "wrong", "idf-only"),
    ("tf-idf", _T_LOG_TF, "wrong", "log-tf"),
    ("tf-idf", _T_PLUS, "wrong", "plus"),
    ("tf-idf", _T_RETURN_IN, "wrong", "return-in-loop"),
    ("tf-idf", _T.replace("    return weights\n", ""), "wrong", "no-return"),
    ("tf-idf", "def tf_idf(counts, df, n):\n    if \"poured\" in counts:\n        return {\"the\": 0.0, \"hatter\": 0.30103, \"poured\": 0.60206, \"tea\": 0.30103}\n    return {\"tea\": 0.90309, \"said\": 0.60206, \"the\": 0.0, \"hatter\": 0.30103}\n", "almost", None),
    ("score", _S_START, "empty", None),
    ("score", _S, "correct", None),
    ("score", "def score(query, weights):\n    return sum(weights.get(word, 0) for word in query)\n", "correct", None),
    ("score", "def score(query, weights):\n    total = 0.0\n    for word in query:\n        total += weights.get(word, 0.0)\n    return total\n", "correct", None),
    ("score", _S_NO_CHECK, "error", "lookup"),
    ("score", _S_COUNTED, "wrong", "counted"),
    ("score", _S_HIGHEST, "wrong", "highest"),
    ("score", _S_ALL, "wrong", "all-words"),
    ("score", _S_RETURN_IN, "wrong", "return-in-loop"),
    ("score", _S.replace("    return total\n", ""), "wrong", "no-return"),
    ("score", "def score(query, weights):\n    if \"hatter\" in query:\n        return 1.2041199826559248\n    return 0.6020599913279624\n", "almost", None),
    ("what-weight", {}, "empty", None),
    ("what-weight", {"of": "zero", "cat": "zero", "whale": "below", "ship": "below", "hatter": "above", "you": "above"},
     "correct", None),
    ("what-weight", {"of": "zero", "hatter": "above"}, "wrong", "unfinished"),
    ("what-weight", {"you": "below"}, "wrong", "wrong:you"),
    ("what-weight", {"of": "zero", "cat": "below", "whale": "below", "ship": "below", "hatter": "above", "you": "above"},
     "wrong", "wrong:cat"),
]
