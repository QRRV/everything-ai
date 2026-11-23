# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-distributional-hypothesis: take the words around a place in a list of words (window),
count for every target word which words stand in its windows (cooccurrence), and say where one word stands among the
nearest neighbors of another (neighbors).

The books are the course's eight books from Project Gutenberg (data/gutenberg-books.json.gz, made by
data/tools/make_gutenberg.py): five in English and three in Dutch. The stop words are those of Snowball
(data/snowball-stop-en.txt and data/snowball-stop-nl.txt). The small sentences, the cases of the checks and the
pictures are the course's own; every number in the pictures is counted by the course."""
import collections
import contextlib
import copy
import html
import io
import math
import random

from everything_ai import books, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, kept, with_time_limit

SIZE = 2               # the window of the lesson: two words on each side
LEAST = 20             # a word gets a row when it occurs at least this often in the books and is not a stop word
NEAR = 8               # how many nearest neighbors a picture shows
SIZES = (1, 2, 5, 10)  # the windows of the explore
_KEY = "nlp-distributional-hypothesis"

# --- the books ---------------------------------------------------------------------------------------------------------


def book_keys(lang: str) -> list:
    """The course's books in the language: five in English, three in Dutch."""
    return [k for k in books.keys() if books.book(k)["lang"] == lang]


def _all_words(lang: str) -> tuple:
    return kept(f"{_KEY}:words:{lang}", lambda: tuple(w.lower() for k in book_keys(lang) for w in books.book_words(k)))


def book_words(lang: str) -> list:
    """Every word of the course's books in the language, in small letters, one book after another: a new list each
    time, so that a learner's code cannot change the course's copy."""
    return list(_all_words(lang))


def word_counts(lang: str) -> collections.Counter:
    return kept(f"{_KEY}:counts:{lang}", lambda: collections.Counter(_all_words(lang)))


def targets(lang: str) -> set:
    """The target words: every word that occurs at least LEAST times in the books of the language and is not a stop
    word. A new set each time."""
    stop = books.stop_words(lang)
    return {w for w, c in word_counts(lang).items() if c >= LEAST and w not in stop}


# --- the course's own versions of the exercises ------------------------------------------------------------------------


def window(words, i, size):
    """The words within `size` places of place i, without word i itself, in their order."""
    return list(words[max(0, i - size):i]) + list(words[i + 1:i + size + 1])


def cooccurrence(words, size, targets):
    """For every target word a sparse row: how often each word stands in a window of `size` around it. A target that
    does not occur gets an empty row."""
    rows = {}
    for target in targets:
        rows[target] = {}
    for i in range(len(words)):
        word = words[i]
        if word in targets:
            for context in window(words, i, size):
                if context in rows[word]:
                    rows[word][context] = rows[word][context] + 1
                else:
                    rows[word][context] = 1
    return rows


def norm(row) -> float:
    """The length of a sparse row: the square root of the sum of the squares of its counts."""
    return math.sqrt(sum(v * v for v in row.values()))


def _dot(a, b) -> float:
    small, big = (a, b) if len(a) <= len(b) else (b, a)
    return sum(v * big[k] for k, v in small.items() if k in big)


def cosine(a, b) -> float:
    """The cosine similarity of two sparse rows, as the course's sparse_cosine of the lesson A document as a vector:
    only the words that both rows hold add to the dot product; 0.0 when they share none."""
    dot = _dot(a, b)
    if dot == 0:
        return 0.0
    return dot / (norm(a) * norm(b))


def _make(lang: str, size: int) -> tuple:
    rows = cooccurrence(_all_words(lang), size, targets(lang))
    return rows, {w: norm(r) for w, r in rows.items()}


_LAST: dict = {}


def book_rows(lang: str, size: int = SIZE) -> tuple:
    """(rows, lengths): the course's rows over the books of the language, and the length of each row. The rows of the
    lesson's window are kept; of the other windows only the last one asked."""
    if size == SIZE:
        return kept(f"{_KEY}:rows:{lang}", lambda: _make(lang, SIZE))
    if (lang, size) not in _LAST:
        _LAST.clear()
        _LAST[(lang, size)] = _make(lang, size)
    return _LAST[(lang, size)]


def _ranked(word: str, rows: dict, lengths: dict) -> list:
    """Every other target word with the cosine similarity of its row with the row of `word`, the highest first; a tie
    goes to the word that comes first in the alphabet; words whose row shares nothing are left out."""
    a, na = rows[word], lengths[word]
    if not na:
        return []
    scored = []
    for other, b in rows.items():
        if other != word and lengths[other]:
            dot = _dot(a, b)
            if dot:
                scored.append((-dot / (na * lengths[other]), other))
    scored.sort()
    return [(w, -c) for c, w in scored]


def nearest(word: str, lang: str, size: int = SIZE, n: int = NEAR) -> list:
    """The n nearest neighbors of `word` in the books of the language: the target words whose rows have the highest
    cosine similarity with its row, each with that similarity."""
    rows, lengths = book_rows(lang, size)
    if size == SIZE:
        return kept(f"{_KEY}:near:{lang}:{word}", lambda: _ranked(word, rows, lengths)[:40])[:n]
    return _ranked(word, rows, lengths)[:n]


def rank(word: str, other: str, lang: str, size: int = SIZE) -> tuple:
    """(place, cosine similarity) of `other` among the neighbors of `word`: place 1 is the nearest."""
    rows, lengths = book_rows(lang, size)
    for k, (w, c) in enumerate(_ranked(word, rows, lengths), start=1):
        if w == other:
            return k, c
    return None, 0.0


def biggest(row: dict, n: int = NEAR) -> list:
    """The n biggest counts of a row, with their words; a tie goes to the word that comes first in the alphabet."""
    return sorted(row.items(), key=lambda p: (-p[1], p[0]))[:n]


# --- the small sentences of the idea -----------------------------------------------------------------------------------

TOY = {
    "en": ["we drink tea with milk", "we drink coffee with sugar", "a cup of hot tea", "a cup of strong coffee",
           "we park the car in the street", "we leave the bike in the rain"],
    "nl": ["we drinken thee met melk", "we drinken koffie met suiker", "een kop warme thee", "een kop sterke koffie",
           "we zetten de auto in de straat", "we laten de fiets in de regen staan"],
}
TOY_TARGETS = {"en": ["tea", "coffee", "car", "bike"], "nl": ["thee", "koffie", "auto", "fiets"]}


def toy_sentences(lang: str) -> list:
    """The course's six small sentences, each as a list of words."""
    return [s.split() for s in TOY[lang]]


def toy_targets(lang: str) -> list:
    """The four target words of the small sentences."""
    return list(TOY_TARGETS[lang])


def toy_rows(lang: str) -> dict:
    """The rows of the four target words over the small sentences, with a window of SIZE, counted sentence by
    sentence: a window never runs into the next sentence."""
    rows = {t: {} for t in TOY_TARGETS[lang]}
    for s in toy_sentences(lang):
        for t, row in cooccurrence(s, SIZE, TOY_TARGETS[lang]).items():
            for c, n in row.items():
                rows[t][c] = rows[t].get(c, 0) + n
    return rows


# the first words of a book, for the picture under Exercise 1: the opening of Pride and Prejudice, and of the first
# chapter of Max Havelaar, in the spelling of 1860
OPENING = {"en": ("pride", "It is a truth universally acknowledged that a single man"),
           "nl": ("havelaar", "Ik ben makelaar in koffi en woon op de Lauriergracht")}
PLACES = (0, 1, 4, 9)   # the places of the opening whose window the picture shows


def opening(lang: str) -> list:
    """The first ten words of the opening sentence, in small letters, as the books' words have them."""
    return OPENING[lang][1].lower().split()


# --- facts the lesson states ---------------------------------------------------------------------------------------------

SHOW = {"en": ["said", "eyes", "father", "ran", "three"], "nl": ["zeide", "oogen", "vader", "liep", "drie"]}
COUNTED = {"en": "door", "nl": "deur"}   # the word whose biggest counts the picture under Exercise 2 shows
GUESS = {"en": ("hot", ["warm", "fire", "cold", "summer"]), "nl": ("warm", ["heet", "vuur", "koud", "zomer"])}


PAIRS = 3000   # random pairs of target words, for the cosine similarity of two words that have nothing to do with each other


def random_pairs(lang: str) -> list:
    """The cosine similarity of PAIRS pairs of target words drawn at random (always the same draw), from low to high."""
    def make():
        rows, _ = book_rows(lang)
        words = sorted(rows)
        draw = random.Random(34)
        return sorted(cosine(rows[a], rows[b]) for a, b in (draw.sample(words, 2) for _ in range(PAIRS)))
    return kept(f"{_KEY}:pairs:{lang}", make)


def facts(lang: str) -> dict:
    """The numbers of the books that the lesson's text and pictures state."""
    def make():
        words = _all_words(lang)
        rows, lengths = book_rows(lang)
        word, options = GUESS[lang]
        return {"words": len(words), "types": len(word_counts(lang)), "targets": len(rows),
                "cells": sum(len(r) for r in rows.values()),
                "guess": {o: rank(word, o, lang) for o in options}}
    return kept(f"{_KEY}:facts:{lang}", make)


# --- the look of the pictures --------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);margin-top:0.45rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums"
_CHIP = ("display:inline-block;margin:0.12rem 0.2rem 0.12rem 0;padding:0 0.3rem;border-radius:3px;"
         "font-family:var(--ea-font-code);font-size:0.78rem")
_CHIP_DATA = _CHIP + ";border:1px solid var(--ea-line);color:var(--ea-ink)"
_CHIP_PRED = _CHIP + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_CHIP_PURPLE = _CHIP + ";background:var(--ea-loss-wash);color:var(--ea-loss-text)"
_CHIP_MISSING = _CHIP + ";border:1px dashed var(--ea-loss);color:var(--ea-loss-text)"
_CHIP_FAINT = _CHIP + ";color:var(--ea-ink-3)"
_BAR = "display:inline-block;height:0.6rem;border-radius:2px;vertical-align:middle"


def _note(text: str) -> str:
    """A line under a picture; a word between asterisks is set in italics, as in the lesson's text."""
    parts = html.escape(text).split("*")
    marked = "".join(f"<i>{p}</i>" if k % 2 else p for k, p in enumerate(parts))
    return f'<div style="{_UI}">{marked}</div>'


def _label(text: str) -> str:
    parts = html.escape(text).split("*")
    marked = "".join(f"<i>{p}</i>" if k % 2 else p for k, p in enumerate(parts))
    return f'<div style="{_LABEL}">{marked}</div>'


def _bare(text: str) -> str:
    return text.replace("*", "")


def _num(n, lang: str) -> str:
    """A whole number with a separator for thousands: 4,518 in English, 4.518 in Dutch."""
    text = f"{round(n):,}"
    return text.replace(",", ".") if lang == "nl" else text


def _dec(x: float, lang: str, digits: int = 3) -> str:
    """A number with `digits` decimals: 0.874 in English, 0,874 in Dutch."""
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def _chip(text: str, style: str = _CHIP_DATA) -> str:
    return f'<span style="{style}">{html.escape(text)}</span>'


def _bar(x: float, most: float, color: str, width: float = 7.0) -> str:
    """A bar for a number: as long as `width` for `most`, never shorter than a sliver."""
    w = 0 if not most or not x else max(0.15, width * x / most)
    return f'<span style="{_BAR};width:{w:.2f}rem;background:{color}"></span>' if w else ""


def _table(rows_: list, head: list, numbers: tuple = (), styles: dict | None = None) -> str:
    """A small table; rows hold HTML that is already escaped; numbers are the columns set to the right; styles maps
    (row, column) to extra style; no head when `head` is empty."""
    styles = styles or {}
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers else ""}{styles.get((i, j), "")}">'
                                    f'{c}</td>' for j, c in enumerate(row)) + "</tr>" for i, row in enumerate(rows_))
    head_html = f"<thead><tr>{top}</tr></thead>" if head else ""
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem">'
            f"{head_html}<tbody>{body}</tbody></table></div>")


def _short(value, limit: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit - 1] + "…"


# --- the idea's picture: the rows of four words over six small sentences --------------------------------------------------

_IDEA_WORDS = {
    "en": {"title": "Six small sentences, and the rows of four words with a window of 2",
           "sentences": "The sentences",
           "matrix": "The co-occurrence matrix: a row for each target word, a column for each word that stands in "
                     "one of its windows",
           "pairs": "Cosine similarity of two rows",
           "pair_head": ["Rows", "Cosine similarity"],
           "legend": "The window of a target word holds the two words before it and the two after it, within its own "
                     "sentence. A dot stands for a count of 0. *{a}* and *{b}* share four of their six columns, *{c}* "
                     "and *{d}* two of their three, and *{a}* and *{c}* none."},
    "nl": {"title": "Zes korte zinnen, en de rijen van vier woorden met een venster van 2",
           "sentences": "De zinnen",
           "matrix": "De co-occurrence matrix: een rij voor elk doelwoord, een kolom voor elk woord dat in een van de "
                     "vensters van het doelwoord staat",
           "pairs": "Cosine similarity van twee rijen",
           "pair_head": ["Rijen", "Cosine similarity"],
           "legend": "Het venster van een doelwoord bevat de twee woorden ervoor en de twee erna, binnen de eigen zin. "
                     "Een punt staat voor een aantal van 0. *{a}* en *{b}* delen vier van de zes kolommen, *{c}* en "
                     "*{d}* twee van de drie, en *{a}* en *{c}* geen enkele."},
}


def idea_picture(lang: str) -> str:
    """The six small sentences with the target words marked, the matrix of their rows, and three cosine similarities."""
    w = _IDEA_WORDS[lang]
    a, b, c, d = toy_targets(lang)
    rows = toy_rows(lang)
    columns = []
    for t in (a, b, c, d):
        for x in rows[t]:
            if x not in columns:
                columns.append(x)
    lines = []
    for s in toy_sentences(lang):
        lines.append(" ".join(f"<b>{html.escape(x)}</b>" if x in rows else html.escape(x) for x in s))
    sentences = "".join(f'<div style="{_CELL};border:none;padding:0.05rem 0">{x}</div>' for x in lines)
    vertical = ("writing-mode:vertical-rl;transform:rotate(180deg);padding:0.3rem 0.15rem;font-family:"
                "var(--ea-font-code);font-size:0.76rem;color:var(--ea-ink-2);font-weight:500")
    count = ("padding:0.12rem 0.25rem;text-align:center;border-bottom:1px solid var(--ea-line);font-family:"
             "var(--ea-font-code);font-size:0.78rem")
    head = "".join(f'<th style="{vertical}">{html.escape(x)}</th>' for x in columns)
    body = ""
    for t in (a, b, c, d):
        cells = "".join(f'<td style="{count};color:var(--ea-ink)">{rows[t][x]}</td>' if x in rows[t]
                        else f'<td style="{count};color:var(--ea-ink-3)">·</td>' for x in columns)
        body += f'<tr><th style="{_CELL};font-family:var(--ea-font-code);font-weight:600">{html.escape(t)}</th>{cells}</tr>'
    matrix = (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0">'
              f'<thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table></div>')
    pairs = [(a, b), (c, d), (a, c)]
    pair_rows = [[f"<i>{html.escape(x)}</i>, <i>{html.escape(y)}</i>", _dec(cosine(rows[x], rows[y]), lang, 2)]
                 for x, y in pairs]
    legend = w["legend"].format(a=a, b=b, c=c, d=d)
    html_ = (_label(w["sentences"]) + sentences + _label(w["matrix"]) + matrix + _label(w["pairs"])
             + _table(pair_rows, w["pair_head"], numbers=(1,)) + _note(legend))
    alt = (f'{w["title"]}. ' + "; ".join(TOY[lang]) + ". " + "; ".join(
        f"{t}: " + ", ".join(f"{x} {n}" for x, n in rows[t].items()) for t in (a, b, c, d)) + ". " + "; ".join(
        f"{x} {y} {_dec(cosine(rows[x], rows[y]), lang, 2)}" for x, y in pairs) + ". " + _bare(legend))
    return yours.figure(w["title"], html_, alt=alt)


# --- the learner's function in a picture ---------------------------------------------------------------------------------


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error


def _call(fn, *args, seconds: float | None = None):
    """The learner's function within the time limit of a check; _STOPPED when it stops with an error, and the Timeout
    passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            run = lambda: fn(*(copy.deepcopy(a) for a in args))   # noqa: E731
            return with_time_limit(run) if seconds is None else with_time_limit(run, seconds)
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


# --- the picture under exercise 1: your windows ------------------------------------------------------------------------

_WINDOW_WORDS = {
    "en": {"title": "Your windows on the first words of {book}",
           "head": ["Place", "Word", "Your window, size 2"],
           "note": "Gray: a word that belongs in the window. Purple: a word your window holds and should not, or one "
                   "it holds too often; dashed: a word it lacks.",
           "fine": "Right at all four places: at place 0 the window has no words before word 0, at place 1 one word, "
                   "and at the last place none after it.",
           "error": "Your function stops with an error at place {i}.",
           "slow": "Your function takes too long at place {i}.",
           "odd": "At place {i} your function gives back {got}, not a list.",
           "places": "The ten words, each with its place"},
    "nl": {"title": "Jouw vensters op de eerste woorden van {book}",
           "head": ["Plaats", "Woord", "Jouw venster, grootte 2"],
           "note": "Grijs: een woord dat in het venster hoort. Paars: een woord dat je venster heeft en niet hoort te "
                   "hebben, of te vaak heeft; met een gestreepte rand: een woord dat ontbreekt.",
           "fine": "Goed op alle vier de plaatsen: op plaats 0 heeft het venster geen woorden voor woord 0, op plaats 1 "
                   "één woord, en op de laatste plaats geen woorden erna.",
           "error": "Je functie stopt met een fout op plaats {i}.",
           "slow": "Je functie doet er te lang over op plaats {i}.",
           "odd": "Op plaats {i} geeft je functie {got} terug, en dat is geen lijst.",
           "places": "De tien woorden, elk met de eigen plaats"},
}


def _window_chips(got: list, want: list) -> str:
    """The learner's window as chips: gray for a word that belongs there, purple for one too many, dashed for one that
    is missing."""
    left = collections.Counter(want)
    chips = []
    for x in got:
        if isinstance(x, str) and left.get(x, 0) > 0:
            left[x] -= 1
            chips.append(_chip(x))
        else:
            chips.append(_chip(str(x), _CHIP_PURPLE))
    for x in want:
        if left.get(x, 0) > 0:
            left[x] -= 1
            chips.append(_chip(x, _CHIP_MISSING))
    return "".join(chips) or _chip("[ ]", _CHIP_FAINT)


def window_picture(seen) -> str:
    """The learner's window at four places of the opening sentence of a book."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _WINDOW_WORDS[lang]
    words = opening(lang)
    title = w["title"].format(book=books.book(OPENING[lang][0])["title"])
    places = "".join(f'<span style="display:inline-block;margin:0 0.35rem 0.2rem 0;text-align:center">'
                     f'<span style="font-family:var(--ea-font-code);font-size:0.8rem;color:var(--ea-ink)">'
                     f'{html.escape(x)}</span><br><span style="font-family:var(--ea-font-ui);font-size:0.7rem;'
                     f'color:var(--ea-ink-3)">{k}</span></span>' for k, x in enumerate(words))
    rows_, alts, end, right = [], [], "", 0
    for i in PLACES:
        try:
            got = _call(fn, words, i, SIZE)
        except Timeout:
            end = w["slow"].format(i=i)
            break
        if got is _STOPPED:
            end = w["error"].format(i=i)
            break
        if not isinstance(got, list):
            end = w["odd"].format(i=i, got=_short(got))
            break
        want = window(words, i, SIZE)
        right += got == want
        rows_.append([str(i), f"<i>{html.escape(words[i])}</i>", _window_chips(got, want)])
        alts.append(f"{i} {words[i]}: " + ", ".join(str(x) for x in got))
    body = _label(w["places"]) + f"<div>{places}</div>"
    if rows_:
        body += _table(rows_, w["head"], numbers=(0,))
    text = end or (w["fine"] if right == len(PLACES) else "")
    body += _note(" ".join(x for x in (text, w["note"]) if x))
    alt = f"{title}. " + "; ".join(alts) + ". " + _bare(" ".join(x for x in (text, w["note"]) if x))
    return yours.figure(title, body, alt=alt)


# --- the picture under exercise 2: your rows over the books ---------------------------------------------------------------

PICTURE_SECONDS = 30   # what the learner's cooccurrence gets for all the words of the books

_ROWS_WORDS = {
    "en": {"title": "Your cooccurrence on the five English books",
           "error": "Your function stops with an error on the words of the books.",
           "slow": "Your function takes more than {s} seconds for the {n} words of the books.",
           "odd": "Your function does not give back a dictionary of rows, so there is nothing to show yet.",
           "differs": "Your rows differ from the course's for {n} of the {m} target words, such as *{word}*. Gray: a "
                      "count that is the same in both rows; purple: a count in your row that differs from the "
                      "course's; dashed: a word your row lacks.",
           "extra": "Your function also gives rows for {n} words that are not target words, such as *{word}*.",
           "compare_head": ["Your row of {word}, biggest counts", "The course's row"],
           "fine": "Your rows are right for all {m} target words: {cells} counts that are not 0, in all {n} words of "
                   "the books. Blue: the nearest neighbors of five words, by the cosine similarity of your rows. Gray: counts in "
                   "a row.",
           "near_head": ["Word", "Nearest neighbors, with the cosine similarity"],
           "row": "The biggest counts in the row of *{word}*, which holds {k} different words:"},
    "nl": {"title": "Jouw cooccurrence op de drie Nederlandse boeken",
           "error": "Je functie stopt met een fout op de woorden van de boeken.",
           "slow": "Je functie doet er langer dan {s} seconden over voor de {n} woorden van de boeken.",
           "odd": "Je functie geeft geen dictionary met rijen terug, dus er valt nog niets te laten zien.",
           "differs": "Je rijen verschillen van de rijen van de course voor {n} van de {m} doelwoorden, zoals "
                      "*{word}*. Grijs: een aantal dat in beide rijen gelijk is; paars: een aantal in je rij dat "
                      "anders is dan bij de course; met een gestreepte rand: een woord dat in je rij ontbreekt.",
           "extra": "Je functie geeft ook rijen voor {n} woorden die geen doelwoord zijn, zoals *{word}*.",
           "compare_head": ["Jouw rij van {word}, grootste aantallen", "De rij van de course"],
           "fine": "Je rijen kloppen voor alle {m} doelwoorden: {cells} aantallen die geen 0 zijn, in alle {n} "
                   "woorden van de boeken. Blauw: de dichtstbijzijnde buren van vijf woorden, volgens de cosine "
                   "similarity van je rijen. Grijs: aantallen in een rij.",
           "near_head": ["Woord", "Dichtstbijzijnde buren, met de cosine similarity"],
           "row": "De grootste aantallen in de rij van *{word}*, die {k} verschillende woorden bevat:"},
}


def _rows_ok(got) -> bool:
    return isinstance(got, dict) and all(isinstance(r, dict) for r in got.values())


def cooccurrence_picture(seen) -> str:
    """The learner's cooccurrence on all the words of the books: where the rows differ from the course's, one row next
    to the course's; when all are right, the nearest neighbors of five words."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _ROWS_WORDS[lang]
    title = w["title"]
    words = _all_words(lang)
    n_words = _num(len(words), lang)
    try:
        got = _call(fn, list(words), SIZE, targets(lang), seconds=PICTURE_SECONDS)
    except Timeout:
        text = w["slow"].format(s=PICTURE_SECONDS, n=n_words)
        return yours.figure(title, _note(text), alt=text)
    if got is _STOPPED or not _rows_ok(got):
        text = w["error"] if got is _STOPPED else w["odd"]
        return yours.figure(title, _note(text), alt=text)
    rows, _ = book_rows(lang)
    m = len(rows)
    differ = [t for t in sorted(rows) if got.get(t) != rows[t]]
    extra = sorted(str(t) for t in got if t not in rows)
    if differ or extra:
        notes = []
        body = ""
        if differ:
            shown = next((t for t in SHOW[lang] if t in differ), differ[0])
            mine, want = got.get(shown) or {}, rows[shown]
            left = [_chip(f"{x} {c}", _CHIP_DATA if want.get(x) == c else _CHIP_PURPLE) for x, c in biggest(mine)]
            right_ = [_chip(f"{x} {c}", _CHIP_DATA if mine.get(x) == c else _CHIP_MISSING) for x, c in biggest(want)]
            head = [h.format(word=shown) for h in w["compare_head"]]
            body += _table([["".join(left) or _chip("{ }", _CHIP_FAINT), "".join(right_)]], head)
            notes.append(w["differs"].format(n=_num(len(differ), lang), m=_num(m, lang), word=shown))
        if extra:
            notes.append(w["extra"].format(n=_num(len(extra), lang), word=extra[0]))
        text = " ".join(notes)
        return yours.figure(title, body + _note(text), alt=f"{title}. " + _bare(text))
    cells = sum(len(r) for r in rows.values())
    table, alts = [], []
    for word in SHOW[lang]:
        near = nearest(word, lang, n=6)
        table.append([f"<i>{html.escape(word)}</i>",
                      "".join(_chip(f"{x} {_dec(c, lang)}", _CHIP_PRED) for x, c in near)])
        alts.append(f"{word}: " + ", ".join(f"{x} {_dec(c, lang)}" for x, c in near))
    first = COUNTED[lang]
    row_text = w["row"].format(word=first, k=_num(len(rows[first]), lang))
    counts = "".join(_chip(f"{x} {_num(c, lang)}") for x, c in biggest(rows[first]))
    text = w["fine"].format(m=_num(m, lang), cells=_num(cells, lang), n=n_words)
    body = _table(table, w["near_head"]) + _label(row_text) + f"<div>{counts}</div>" + _note(text)
    alt = (f"{title}. " + "; ".join(alts) + ". " + _bare(row_text) + " "
           + ", ".join(f"{x} {c}" for x, c in biggest(rows[first])) + ". " + _bare(text))
    return yours.figure(title, body, alt=alt)


# --- the explore: the neighbors of any word, with another window ------------------------------------------------------------

EXPLORE_WORD = {"en": "door", "nl": "deur"}
LONGEST = 40   # characters of the explore's word box that count

_EXPLORE_WORDS = {
    "en": {"title": "The neighbors of {word}, window {size}",
           "empty": "Type a word.",
           "stop": "*{word}* is a stop word, so the course gives it no row.",
           "absent": "*{word}* does not occur in the five English books. The books often use words such as *oh* and "
                     "*upon*; try a word of the 1800s.",
           "rare": "*{word}* occurs {c} times in the books; a word needs at least {least} to get a row.",
           "facts": "*{word}* occurs {c} times. Its row holds {k} different words; the biggest counts:",
           "near": "Its {n} nearest neighbors, with the cosine similarity of their rows:",
           "none": "No other row shares a word with this row.",
           "note": "Gray: counts in the row, words from the books. Blue: the nearest neighbors, the words whose rows "
                   "are most like this row."},
    "nl": {"title": "De buren van {word}, venster {size}",
           "empty": "Typ een woord.",
           "stop": "*{word}* is een stop word, dus de course geeft het geen rij.",
           "absent": "*{word}* komt niet voor in de drie Nederlandse boeken. De boeken hebben de spelling van de "
                     "negentiende eeuw, zoals *oogen* en *zeide*; probeer ook die spelling.",
           "rare": "*{word}* komt {c} keer voor in de boeken; een woord heeft er minstens {least} nodig voor een rij.",
           "facts": "*{word}* komt {c} keer voor. De rij bevat {k} verschillende woorden; de grootste aantallen:",
           "near": "De {n} dichtstbijzijnde buren, met de cosine similarity van de rijen:",
           "none": "Geen andere rij deelt een woord met deze rij.",
           "note": "Grijs: aantallen in de rij, woorden uit de boeken. Blauw: de dichtstbijzijnde buren, de woorden "
                   "waarvan de rij het meest op deze rij lijkt."},
}


def explore_picture(word, size, lang: str) -> str:
    """The row and the nearest neighbors of a word the learner types, over the books, with the window chosen."""
    w = _EXPLORE_WORDS[lang]
    size = int(size)
    word = (word or "").strip().lower()[:LONGEST]
    title = w["title"].format(word=word or "…", size=size)
    if not word:
        return yours.figure(title, _note(w["empty"]), alt=w["empty"])
    rows, _ = book_rows(lang, size)
    if word not in rows:
        c = word_counts(lang).get(word, 0)
        if word in books.stop_words(lang):
            text = w["stop"].format(word=word)
        elif c == 0:
            text = w["absent"].format(word=word)
        else:
            text = w["rare"].format(word=word, c=_num(c, lang), least=LEAST)
        return yours.figure(title, _note(text), alt=_bare(text))
    row = rows[word]
    facts_text = w["facts"].format(word=word, c=_num(word_counts(lang)[word], lang), k=_num(len(row), lang))
    counts = "".join(_chip(f"{x} {_num(c, lang)}") for x, c in biggest(row))
    near = nearest(word, lang, size)
    if near:
        table = [[_chip(x, _CHIP_PRED), _dec(c, lang), _bar(c, 1.0, "var(--ea-pred)")] for x, c in near]
        near_html = _label(w["near"].format(n=len(near))) + _table(table, [], numbers=(1,))
        near_alt = w["near"].format(n=len(near)) + " " + ", ".join(f"{x} {_dec(c, lang)}" for x, c in near)
    else:
        near_html, near_alt = _label(w["none"]), w["none"]
    body = _label(facts_text) + f"<div>{counts}</div>" + near_html + _note(w["note"])
    alt = (f"{title}. " + _bare(facts_text) + " " + ", ".join(f"{x} {c}" for x, c in biggest(row)) + ". "
           + _bare(near_alt) + ". " + w["note"])
    return yours.figure(title, body, alt=alt)


# --- the exercises -------------------------------------------------------------------------------------------------------

REQUIRED = ["window", "cooccurrence", "neighbors"]


class Named:
    """A value the check gives the learner's function, shown in a message by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: "..." in "\n".join(
                    line.split("#")[0] for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang de `...` door je eigen code.")


def _index_error(e) -> bool:
    return isinstance(e, IndexError)


def _key_error(e) -> bool:
    return isinstance(e, KeyError)


def _not_list(got, text) -> bool:
    return isinstance(got, (set, tuple))


def _nested(got, text) -> bool:
    return isinstance(got, list) and any(isinstance(x, list) for x in got)


def _sets(got, text) -> bool:
    return isinstance(got, dict) and any(isinstance(r, (set, list)) for r in got.values())


# exercise 1: window. Seven letters, the middle first; then the start, where nothing comes before word 0, place 1,
# where a slice from -1 runs from the end, the end, a window that starts at place 0 exactly, size 1, size 0, a window
# wider than the list on both sides, a list of one word, a word that comes twice, a window past the end of the list on
# one side only, a longer list with a window that starts below 0, and the openings of the two books.
LETTERS = ["a", "b", "c", "d", "e", "f", "g"]
WINDOW_CASES = [
    (LETTERS, 3, 2),
    (LETTERS, 0, 2),
    (LETTERS, 1, 2),
    (LETTERS, 6, 2),
    (LETTERS, 2, 2),
    (LETTERS, 3, 1),
    (LETTERS, 3, 0),
    (["a", "b", "c"], 1, 5),
    (["a"], 0, 2),
    (["the", "cat", "and", "the", "dog"], 2, 2),
    (LETTERS, 5, 3),
    (["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"], 1, 3),
    (opening("en"), 4, 2),
    (opening("nl"), 4, 2),
]

_W_START = "def window(words, i, size):\n    ...\n"
_W = "def window(words, i, size):\n    return words[max(0, i - size):i] + words[i + 1:i + size + 1]\n"
_W_NEG = _W.replace("words[max(0, i - size):i]", "words[i - size:i]")
_W_SELF = "def window(words, i, size):\n    return words[max(0, i - size):i + size + 1]\n"
_W_SHORT = _W.replace("words[i + 1:i + size + 1]", "words[i + 1:i + size]")
_W_LEFT = _W.replace("words[max(0, i - size):i]", "words[max(0, i - size + 1):i]")
_W_WIDE = _W.replace("words[i + 1:i + size + 1]", "words[i + 1:i + size + 2]")
_W_AFTER = "def window(words, i, size):\n    return words[i + 1:i + size + 1]\n"
_W_BEFORE = "def window(words, i, size):\n    return words[max(0, i - size):i]\n"
_W_LOOP = ("def window(words, i, size):\n    found = []\n    for j in range(i - size, i + size + 1):\n"
           "        if j != i:\n            found.append(words[j])\n    return found\n")
_W_CLAMP = _W_LOOP.replace("range(i - size, i + size + 1)", "range(max(0, i - size), i + size + 1)")
_W_WRAP = _W_LOOP.replace("        if j != i:\n", "        if j != i and j < len(words):\n")
_W_LOOP_ALL = _W_LOOP.replace("        if j != i:\n", "        if j < len(words):\n")
_W_SLICE_ALL = "def window(words, i, size):\n    return words[i - size:i + size + 1]\n"

# exercise 2: cooccurrence. Five letters with two targets first, then a target that does not occur; then a word next to
# itself, no words, no targets, size 1, size 0, the targets as a set, a word twice with a window wider than the list,
# the opening of Pride and Prejudice with a stop word as a target, the opening of Max Havelaar, and the first 500 words
# of Alice's Adventures in Wonderland.
ALICE = Named("ALICE", [w.lower() for w in books.book_words("alice")[:500]])
COOC_CASES = [
    (["a", "b", "c", "a", "b"], 2, ["a", "c"]),
    (["a", "b", "a"], 1, ["a", "z"]),
    (["very", "very", "good"], 1, ["very"]),
    ([], 2, ["a"]),
    (["a", "b"], 2, []),
    (["a", "b", "c", "d", "e"], 1, ["c"]),
    (["a", "b", "c"], 0, ["b"]),
    (["x", "y", "x", "z"], 1, {"x"}),
    (["a", "a"], 5, ["a"]),
    (opening("en"), 2, ["truth", "man", "a"]),
    (opening("nl"), 2, ["koffi", "woon"]),
    (ALICE, 3, ["alice", "rabbit", "very", "time"]),
]

_C_START = ("def cooccurrence(words, size, targets):\n    rows = {}\n    for target in targets:\n"
            "        rows[target] = {}\n    ...\n    return rows\n")
_C = ("def cooccurrence(words, size, targets):\n    rows = {}\n    for target in targets:\n        rows[target] = {}\n"
      "    for i in range(len(words)):\n        word = words[i]\n        if word in targets:\n"
      "            for context in window(words, i, size):\n                if context in rows[word]:\n"
      "                    rows[word][context] = rows[word][context] + 1\n                else:\n"
      "                    rows[word][context] = 1\n    return rows\n")
_C_ONE = _C.replace("                if context in rows[word]:\n                    rows[word][context] = rows[word][context] + 1\n"
                    "                else:\n                    rows[word][context] = 1\n",
                    "                rows[word][context] = 1\n")
_C_RESET = _C.replace("        if word in targets:\n", "        if word in targets:\n            rows[word] = {}\n")
_C_LAZY = _C.replace("    for target in targets:\n        rows[target] = {}\n", "").replace(
    "        if word in targets:\n", "        if word in targets:\n            if word not in rows:\n                rows[word] = {}\n")
_C_SELF = _C.replace("window(words, i, size)", "words[max(0, i - size):i + size + 1]")
_C_FIXED = _C.replace("window(words, i, size)", "window(words, i, 2)")
_C_SAME = _C.replace("                if context in rows[word]:", "                if context == word:\n"
                     "                    continue\n                if context in rows[word]:")
_C_EARLY = _C.replace("\n    return rows\n", "\n        return rows\n")
_C_INDEX = ("def cooccurrence(words, size, targets):\n    rows = {}\n    for target in targets:\n        rows[target] = {}\n"
            "    for word in words:\n        if word in targets:\n"
            "            for context in window(words, words.index(word), size):\n                if context in rows[word]:\n"
            "                    rows[word][context] = rows[word][context] + 1\n                else:\n"
            "                    rows[word][context] = 1\n    return rows\n")
_C_ALL = _C.replace("        if word in targets:\n", "        if word not in rows:\n            rows[word] = {}\n        if True:\n")
_C_SETS = ("def cooccurrence(words, size, targets):\n    rows = {}\n    for target in targets:\n        rows[target] = set()\n"
           "    for i in range(len(words)):\n        if words[i] in targets:\n"
           "            for context in window(words, i, size):\n                rows[words[i]].add(context)\n"
           "    return rows\n")

EXERCISES = {
    "window": FunctionBox(
        name="window",
        picture=window_picture,
        function="window",
        start=_W_START,
        solution=_W,
        visible=WINDOW_CASES[:2],
        hidden=WINDOW_CASES[2:],
        first=[_DOTS],
        mistakes=[
            Mistake("index", error=_index_error,
                    en="With `{call}`, your function looks past the end of the list. A slice never does that: `words[5:9]` of a list of seven words simply gives the words at places 5 and 6.",
                    nl="Met `{call}` kijkt je functie voorbij het eind van de lijst. Een slice doet dat nooit: `words[5:9]` van een lijst van zeven woorden geeft gewoon de woorden op plaats 5 en 6."),
            Mistake("wrap", wrong=_W_LOOP,
                    en="`{call}` gives {got}, and it should give {want}. At place 0, your loop starts at `j` = −2, and `words[-2]` is the last word but one: a negative place counts from the end of the list. At the end of the list, `words[j]` looks past the last word, so stop there. And at which `j` should the loop start?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Op plaats 0 begint je loop bij `j` = −2, en `words[-2]` is het op een na laatste woord: een negatieve plaats telt vanaf het eind van de lijst. Aan het eind van de lijst kijkt `words[j]` voorbij het laatste woord, dus stop daar. En bij welke `j` moet de loop beginnen?"),
            Mistake("wrap-start", wrong=_W_WRAP,
                    en="`{call}` gives {got}, and it should give {want}. At place 0, your loop starts at `j` = −2, and `words[-2]` is the last word but one: a negative place counts from the end of the list. At which `j` should the loop start?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Op plaats 0 begint je loop bij `j` = −2, en `words[-2]` is het op een na laatste woord: een negatieve plaats telt vanaf het eind van de lijst. Bij welke `j` moet de loop beginnen?"),
            Mistake("loop-all", wrong=_W_LOOP_ALL,
                    en="`{call}` gives {got}, and it should give {want}: your window holds word `i` itself, so leave out `j` when it is `i`. And at place 0 your loop starts at `j` = −2, where `words[-2]` is the last word but one: at which `j` should the loop start?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je venster bevat woord `i` zelf, dus sla `j` over als `j` gelijk is aan `i`. En op plaats 0 begint je loop bij `j` = −2, waar `words[-2]` het op een na laatste woord is: bij welke `j` moet de loop beginnen?"),
            Mistake("slice-all", wrong=_W_SLICE_ALL,
                    en="`{call}` gives {got}, and it should give {want}: your window holds word `i` itself. Take the words before place `i` and the words after it as two slices. And what does a slice give near the start of the list, where `i - size` is below 0?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je venster bevat woord `i` zelf. Neem de woorden vóór plaats `i` en de woorden erna als twee slices. En wat geeft een slice vlak na het begin van de lijst, waar `i - size` kleiner is dan 0?"),
            Mistake("not-list", matches=_not_list,
                    en="`{call}` gives {got}, and it should give the list {want}. Give back a list, with the words in their order and a word that comes twice also twice.",
                    nl="`{call}` geeft {got}, en dat moet de lijst {want} zijn. Geef een lijst terug, met de woorden in de volgorde van de lijst, en een woord dat twee keer voorkomt ook twee keer."),
            Mistake("nested", matches=_nested,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back the two pieces as a list of two lists. Put the words before and the words after together into one list with `+`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft de twee stukken terug als een lijst van twee lijsten. Zet de woorden ervoor en de woorden erna met `+` samen in één lijst."),
            Mistake("negative", wrong=_W_NEG,
                    en="`{call}` gives {got}, and it should give {want}. Near the start of the list, `i - size` is below 0, and a slice that starts at a negative number counts from the end: `words[-1:1]` is empty. Where should the slice start then?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Vlak na het begin van de lijst is `i - size` kleiner dan 0, en een slice die bij een negatief getal begint, telt vanaf het eind: `words[-1:1]` is leeg. Waar moet de slice dan beginnen?"),
            Mistake("self", wrong=_W_SELF,
                    en="`{call}` gives {got}, and it should give {want}: your window holds word `i` itself. Take the words before place `i` and the words after it, without the word at place `i`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je venster bevat woord `i` zelf. Neem de woorden vóór plaats `i` en de woorden erna, zonder het woord op plaats `i`."),
            Mistake("short", wrong=_W_SHORT,
                    en="`{call}` gives {got}, and it should give {want}: after word `i` your window has one word too few. A slice stops just before the place after the colon.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: na woord `i` heeft je venster één woord te weinig. Een slice stopt net vóór de plaats na de dubbele punt."),
            Mistake("left", wrong=_W_LEFT,
                    en="`{call}` gives {got}, and it should give {want}: before word `i` your window has one word too few. The window holds `size` words on each side.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: vóór woord `i` heeft je venster één woord te weinig. Het venster heeft `size` woorden aan elke kant."),
            Mistake("wide", wrong=_W_WIDE,
                    en="`{call}` gives {got}, and it should give {want}: after word `i` your window has one word too many. The window holds `size` words on each side.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: na woord `i` heeft je venster één woord te veel. Het venster heeft `size` woorden aan elke kant."),
            Mistake("after", wrong=_W_AFTER,
                    en="`{call}` gives {got}, and it should give {want}: your window holds only the words after word `i`. The words before it belong in the window too.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je venster bevat alleen de woorden na woord `i`. De woorden ervoor horen er ook bij."),
            Mistake("before", wrong=_W_BEFORE,
                    en="`{call}` gives {got}, and it should give {want}: your window holds only the words before word `i`. The words after it belong in the window too.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je venster bevat alleen de woorden vóór woord `i`. De woorden erna horen er ook bij."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back the `size` words before place `i` and the `size` words after it, in one list.",
                      "nl": "Vervang de `...`: geef de `size` woorden vóór plaats `i` en de `size` woorden erna terug, in één lijst."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The window holds the words within `size` places of place `i`, in their order, without word `i` itself.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Het venster bevat de woorden binnen `size` plaatsen van plaats `i`, in de volgorde van de lijst, zonder woord `i` zelf."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function work at the start and at the end of the list too?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie ook aan het begin en aan het eind van de lijst?"},
            "correct": {"en": "Your `window` is right in all {n} cases: also at the start and the end of the list, with size 0, and with a window wider than the list. Look at the picture: your function on the first words of a book.",
                        "nl": "Je `window` klopt in alle {n} gevallen: ook aan het begin en het eind van de lijst, met grootte 0, en met een venster dat breder is dan de lijst. Kijk naar de afbeelding: je functie op de eerste woorden van een boek."},
        },
    ),
    "cooccurrence": FunctionBox(
        name="cooccurrence",
        picture=cooccurrence_picture,
        function="cooccurrence",
        start=_C_START,
        solution=_C,
        names={"window": window},
        visible=COOC_CASES[:2],
        hidden=COOC_CASES[2:],
        collect=lambda value: {k: dict(v) if isinstance(v, dict) else v for k, v in value.items()}
        if isinstance(value, dict) else value,   # a row may be a Counter, which is a dictionary too
        first=[_DOTS],
        mistakes=[
            Mistake("key", error=_key_error,
                    en="With `{call}`, your function stops with a `KeyError`: it looks up a key that the dictionary does not hold yet. Does it look only at the target words, with `if word in targets:`? And when a word comes into a row for the first time, does it start its count at 1?",
                    nl="Met `{call}` stopt je functie met een `KeyError`: je functie zoekt een sleutel op die nog niet in de dictionary staat. Kijkt je functie alleen naar de doelwoorden, met `if word in targets:`? En begint je functie het aantal bij 1 als een woord voor het eerst in een rij komt?"),
            Mistake("sets", matches=_sets,
                    en="`{call}` gives {got}, and it should give {want}: your rows hold the words without their counts. A row is a dictionary from word to count: how often the word stands in a window of the target.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je rijen bevatten de woorden zonder de aantallen. Een rij is een dictionary van woord naar aantal: hoe vaak het woord in een venster van het doelwoord staat."),
            Mistake("one", wrong=_C_ONE,
                    en="`{call}` gives {got}, and it should give {want}: every count is 1, also of a word that stands in two windows of the target. When the word is in the row already, add 1 to its count.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elk aantal is 1, ook van een woord dat in twee vensters van het doelwoord staat. Staat het woord al in de rij, tel dan 1 op bij het aantal."),
            Mistake("reset", wrong=_C_RESET,
                    en="`{call}` gives {got}, and it should give {want}: each row holds only the last window of its target. Your function starts the row again each time the target comes up; the start code made every row once, before the loop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke rij bevat alleen het laatste venster van het doelwoord. Je functie begint de rij opnieuw elke keer dat het doelwoord voorkomt; de startcode maakte elke rij één keer, vóór de loop."),
            Mistake("lazy", wrong=_C_LAZY,
                    en="`{call}` gives {got}, and it should give {want}: a target word that does not occur gets no row. Every target gets a row, an empty one when it does not occur: keep the lines of the start code that make them.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een doelwoord dat niet voorkomt, krijgt geen rij. Elk doelwoord krijgt een rij, een lege als het niet voorkomt: houd de regels van de startcode die de rijen maken."),
            Mistake("self", wrong=_C_SELF,
                    en="`{call}` gives {got}, and it should give {want}: each row counts the target word itself, once for every time it occurs. Leave word `i` out of its own window; `window` does that.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke rij telt het doelwoord zelf, één keer voor elke keer dat het voorkomt. Laat woord `i` uit het eigen venster; `window` doet dat."),
            Mistake("fixed", wrong=_C_FIXED,
                    en="`{call}` gives {got}, and it should give {want}: your function uses a window of 2, whatever `size` it gets. Hand `size` on to `window`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie gebruikt een venster van 2, welke `size` je functie ook krijgt. Geef `size` door aan `window`."),
            Mistake("same", wrong=_C_SAME,
                    en="`{call}` gives {got}, and it should give {want}. In *very very good*, the second *very* stands in the window of the first: your function leaves out a word that is the same as the target. Leave out only word `i` itself, which `window` already does.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. In *very very good* staat de tweede *very* in het venster van de eerste: je functie laat een woord weg dat gelijk is aan het doelwoord. Laat alleen woord `i` zelf weg; `window` doet dat al."),
            Mistake("early", wrong=_C_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back the rows after the first place. Move `return rows` to the left, level with the first `for`, so that it comes after the loop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft de rijen al na de eerste plaats terug. Zet `return rows` meer naar links, even ver als de eerste `for`, zodat de `return` na de loop komt."),
            Mistake("index-of", wrong=_C_INDEX,
                    en="`{call}` gives {got}, and it should give {want}: `words.index(word)` gives the first place of the word, so a word that comes twice gets the window of its first place twice. Go over the places with `for i in range(len(words)):` and take `words[i]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `words.index(word)` geeft de eerste plaats van het woord, dus een woord dat twee keer voorkomt, krijgt twee keer het venster van de eerste plaats. Loop langs de plaatsen met `for i in range(len(words)):` en neem `words[i]`."),
            Mistake("all", wrong=_C_ALL,
                    en="`{call}` gives {got}, and it should give {want}: your function gives a row to every word, also to words that are not target words. Count only at a target word, with `if word in targets:`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft elk woord een rij, ook woorden die geen doelwoord zijn. Tel alleen bij een doelwoord, met `if word in targets:`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: go over the places of `words`; at a target word, add 1 to the count of every word in its window, in the row of the target.",
                      "nl": "Vervang de `...`: loop langs de plaatsen van `words`; tel bij een doelwoord 1 op bij het aantal van elk woord in het venster, in de rij van het doelwoord."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The function gives back a row for every target word: a dictionary from each word in its windows to how often it stands there.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De functie geeft voor elk doelwoord een rij terug: een dictionary van elk woord in de vensters van het doelwoord naar hoe vaak het woord daar staat."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does every target get a row, and does your function use the `size` it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Krijgt elk doelwoord een rij, en gebruikt je functie de `size` die je functie krijgt?"},
            "correct": {"en": "Your `cooccurrence` is right in all {n} cases: also for a target that does not occur, for a word next to itself, and for the targets as a set. Look at the picture: your function on all the words of the books.",
                        "nl": "Je `cooccurrence` klopt in alle {n} gevallen: ook voor een doelwoord dat niet voorkomt, voor een woord naast zichzelf, en voor de doelwoorden als set. Kijk naar de afbeelding: je functie op alle woorden van de boeken."},
        },
    ),
    "neighbors": Sort(
        name="neighbors",
        groups={
            "near": {"en": "Among the 10 nearest", "nl": "Bij de 10 dichtstbijzijnde"},
            "far": {"en": "Beyond the 100 nearest", "nl": "Voorbij de 100 dichtstbijzijnde"},
        },
        items={
            "speak": {"en": "*answered* among the neighbors of *asked*",
                      "nl": "*antwoordde* bij de buren van *vroeg*"},
            "see": {"en": "*see* among the neighbors of *eyes*",
                    "nl": "*zien* bij de buren van *oogen*"},
            "move": {"en": "*went* among the neighbors of *came*",
                     "nl": "*kwam* bij de buren van *ging*"},
            "write": {"en": "*wrote* among the neighbors of *letter*",
                      "nl": "*schreef* bij de buren van *brief*"},
            "think": {"en": "*felt* among the neighbors of *thought*",
                      "nl": "*meende* bij de buren van *dacht*"},
            "cry": {"en": "*cried* among the neighbors of *tears*",
                    "nl": "*weende* (huilde) bij de buren van *tranen*"},
        },
        solution={"speak": "near", "see": "far", "move": "near", "write": "far", "think": "near", "cry": "far"},
        nudge={
            "speak": {"en": "Can *answered* stand in the place of *asked*, with the same words around it?",
                      "nl": "Kan *antwoordde* op de plaats van *vroeg* staan, met dezelfde woorden eromheen?"},
            "see": {"en": "Do *eyes* and *see* stand in the same places, or next to each other?",
                    "nl": "Staan *oogen* en *zien* op dezelfde plaatsen, of naast elkaar?"},
            "move": {"en": "Can *went* stand in the place of *came*, with the same words around it?",
                     "nl": "Kan *kwam* op de plaats van *ging* staan, met dezelfde woorden eromheen?"},
            "write": {"en": "Is *wrote* a word of the same kind as *letter*, or a word that goes with it?",
                      "nl": "Is *schreef* een woord van dezelfde soort als *brief*, of een woord dat erbij hoort?"},
            "think": {"en": "Can *felt* stand in the place of *thought*, with the same words around it?",
                      "nl": "Kan *meende* op de plaats van *dacht* staan, met dezelfde woorden eromheen?"},
            "cry": {"en": "Do *tears* and *cried* stand in the same places, or next to each other?",
                    "nl": "Staan *tranen* en *weende* (huilde) op dezelfde plaatsen, of naast elkaar?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the pairs again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de paren opnieuw uit."},
            "empty": {"en": "Put each line under where its word stands among the neighbors of the other word.",
                      "nl": "Zet elke regel bij de plaats waar het woord staat tussen de buren van het andere woord."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs under the other heading.",
                      "nl": "{item} hoort onder het andere kopje."},
            "correct": {"en": "All {n} are right. A word of the same kind, which can stand in the same place, gets a row like the other's; a word that stands next to the other gets a different row.",
                        "nl": "Alle {n} kloppen. Een woord van dezelfde soort, dat op dezelfde plaats kan staan, krijgt een rij die lijkt op de rij van het andere woord; een woord dat naast het andere woord staat, krijgt een andere rij."},
        },
    ),
}

CASES = [
    ("window", _W_START, "empty", None),
    ("window", _W, "correct", None),
    ("window", "def window(words, i, size):\n    start = i - size\n    if start < 0:\n        start = 0\n    return words[start:i] + words[i + 1:i + 1 + size]\n", "correct", None),
    ("window", "def window(words, i, size):\n    found = []\n    for j in range(len(words)):\n        if j != i and abs(j - i) <= size:\n            found.append(words[j])\n    return found\n", "correct", None),
    ("window", "def window(words, i, size):\n    return [words[j] for j in range(max(0, i - size), min(len(words), i + size + 1)) if j != i]\n", "correct", None),
    ("window", _W_NEG, "wrong", "negative"),
    ("window", _W_SELF, "wrong", "self"),
    ("window", _W_SHORT, "wrong", "short"),
    ("window", _W_LEFT, "wrong", "left"),
    ("window", _W_WIDE, "wrong", "wide"),
    ("window", _W_AFTER, "wrong", "after"),
    ("window", _W_BEFORE, "wrong", "before"),
    ("window", _W_LOOP, "wrong", "wrap"),
    ("window", _W_WRAP, "wrong", "wrap-start"),
    ("window", _W_LOOP_ALL, "wrong", "loop-all"),
    ("window", _W_SLICE_ALL, "wrong", "slice-all"),
    ("window", _W_CLAMP, "error", "index"),
    ("window", "def window(words, i, size):\n    return set(words[max(0, i - size):i] + words[i + 1:i + size + 1])\n", "wrong", "not-list"),
    ("window", "def window(words, i, size):\n    return [words[max(0, i - size):i], words[i + 1:i + size + 1]]\n", "wrong", "nested"),
    ("window", "def window(words, i, size):\n    if i == 3:\n        return ['b', 'c', 'e', 'f']\n    return ['b', 'c']\n", "almost", None),
    ("cooccurrence", _C_START, "empty", None),
    ("cooccurrence", _C, "correct", None),
    ("cooccurrence", "def cooccurrence(words, size, targets):\n    rows = {}\n    for target in targets:\n        rows[target] = {}\n    for i in range(len(words)):\n        if words[i] in targets:\n            row = rows[words[i]]\n            for context in window(words, i, size):\n                row[context] = row.get(context, 0) + 1\n    return rows\n", "correct", None),
    ("cooccurrence", "from collections import Counter\n\ndef cooccurrence(words, size, targets):\n    rows = {target: Counter() for target in targets}\n    for i, word in enumerate(words):\n        if word in rows:\n            rows[word].update(window(words, i, size))\n    return rows\n", "correct", None),
    ("cooccurrence", _C_ONE, "wrong", "one"),
    ("cooccurrence", _C_RESET, "wrong", "reset"),
    ("cooccurrence", _C_LAZY, "wrong", "lazy"),
    ("cooccurrence", _C_SELF, "wrong", "self"),
    ("cooccurrence", _C_FIXED, "wrong", "fixed"),
    ("cooccurrence", _C_SAME, "wrong", "same"),
    ("cooccurrence", _C_EARLY, "wrong", "early"),
    ("cooccurrence", _C_SETS, "wrong", "sets"),
    ("cooccurrence", _C_INDEX, "wrong", "index-of"),
    ("cooccurrence", _C_ALL, "wrong", "all"),
    ("cooccurrence", _C.replace("                if context in rows[word]:\n                    rows[word][context] = rows[word][context] + 1\n                else:\n                    rows[word][context] = 1\n",
                                "                rows[word][context] = rows[word][context] + 1\n"), "error", "key"),
    ("cooccurrence", _C.replace("        if word in targets:\n            for", "        if True:\n            for"), "error", "key"),
    ("cooccurrence", "def cooccurrence(words, size, targets):\n    if 'c' in targets:\n        return {'a': {'b': 3, 'c': 2}, 'c': {'a': 2, 'b': 2}}\n    return {'a': {'b': 2}, 'z': {}}\n", "almost", None),
    ("neighbors", {}, "empty", None),
    ("neighbors", {"speak": "near", "see": "far", "move": "near", "write": "far", "think": "near", "cry": "far"}, "correct", None),
    ("neighbors", {"speak": "near", "see": "far"}, "wrong", "unfinished"),
    ("neighbors", {"speak": "far"}, "wrong", "wrong:speak"),
    ("neighbors", {"see": "near"}, "wrong", "wrong:see"),
    ("neighbors", {"move": "far"}, "wrong", "wrong:move"),
    ("neighbors", {"write": "near"}, "wrong", "wrong:write"),
    ("neighbors", {"think": "far"}, "wrong", "wrong:think"),
    ("neighbors", {"cry": "near"}, "wrong", "wrong:cry"),
]

# the pairs of the sort, per language: (word, neighbor) for each item
SORT_PAIRS = {
    "en": {"speak": ("asked", "answered"), "see": ("eyes", "see"), "move": ("came", "went"), "write": ("letter", "wrote"),
           "think": ("thought", "felt"), "cry": ("tears", "cried")},
    "nl": {"speak": ("vroeg", "antwoordde"), "see": ("oogen", "zien"), "move": ("ging", "kwam"), "write": ("brief", "schreef"),
           "think": ("dacht", "meende"), "cry": ("tranen", "weende")},
}
