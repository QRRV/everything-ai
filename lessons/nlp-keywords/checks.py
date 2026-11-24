# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-keywords: the keywords of a chapter by TF-IDF; the pairs of neighboring words of a text
and how often each occurs (pair_counts); the pointwise mutual information of a pair from those counts (pmi); the pairs
with the highest PMI among those that occur at least a number of times, the terms of a book (terms); a box that shows
the terms of each of the eight books for a minimum of the learner's choice; and a choice about why pairs that occur
once fill the top of the PMI.

The books are the eight Project Gutenberg books of data/gutenberg-books.json.gz (public domain), read with
everything_ai.books and taken in small letters: five in English and three in Dutch, with their old spelling. The
chapters are the 251 of books.english_chapters(), as in nlp-tf-idf. The measure is the association ratio of Kenneth
Church and Patrick Hanks (Computational Linguistics 16(1), 1990), with the window of two neighboring words: both the
words and the pairs are counted against N, the number of words. Every number in the text and the pictures is computed
by the course (tests/test_nlp_keywords.py computes them again)."""
import collections
import contextlib
import functools
import html
import io
import math
import time
import warnings

import numpy as np

from everything_ai import books, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

LANGS = ("en", "nl")
BOOKS = ("alice", "pride", "holmes", "frankenstein", "moby-dick", "eline-vere", "havelaar", "camera-obscura")
SHORT = {"alice": "Alice", "pride": "Pride and Prejudice", "holmes": "Sherlock Holmes", "frankenstein": "Frankenstein",
         "moby-dick": "Moby-Dick", "eline-vere": "Eline Vere", "havelaar": "Max Havelaar",
         "camera-obscura": "Camera Obscura"}
LANGUAGE = {"en": {"en": "English", "nl": "Dutch"}, "nl": {"en": "Engels", "nl": "Nederlands"}}
TEA_PARTY = 6   # the place of A Mad Tea-Party among the 251 chapters, counting from 0, as in nlp-tf-idf

# words that the pictures leave out: the course names nobody by a gendered pronoun, so a list of pairs moves on to the
# next pair (as nlp-zipf-heaps does with its table of words); for Dutch also in the old spelling of the three books,
# and per language, since the English *hare* is an animal and the old Dutch *hare* means *haar*
_PRONOUNS_EN = {"he", "she", "him", "her", "his", "hers", "himself", "herself"}
_PRONOUNS_NL = {"hij", "zij", "ze", "haar", "hem", "zijn", "hy", "zy", "zyn", "hare", "haren", "heur", "zijne", "zyne",
                "zijnen", "zynen", "zijns", "zyns", "zijner", "zyner", "haars", "harer"}
_PRONOUNS = {"en": _PRONOUNS_EN, "nl": _PRONOUNS_NL}


def showable(pair, lang: str = "en") -> bool:
    """A pair of a book in language `lang` that the pictures may show: neither word is a pronoun that names a person
    by gender."""
    return not any(w in _PRONOUNS[lang] for w in pair)


# --- the course's solutions --------------------------------------------------------------------------------------------

_PC_HEAD = "def pair_counts(words):\n    pairs = {}\n    for i in range(len(words) - 1):\n"
_PC_COUNT = ("        if pair in pairs:\n            pairs[pair] = pairs[pair] + 1\n"
             "        else:\n            pairs[pair] = 1\n")


def _pc(pair: str = "(words[i], words[i + 1])", count: str = _PC_COUNT, head: str = _PC_HEAD,
        end: str = "    return pairs\n") -> str:
    return head + f"        pair = {pair}\n" + count + end


_PC = _pc()
_PC_START = _PC_HEAD + "        pair = ...\n        ...\n    return pairs\n"

_PMI_HEAD = "import numpy as np\n\n\ndef pmi(pair, pairs, counts, n):\n    x, y = pair\n"


def _pm(chance: str = "counts[x] * counts[y] / n", ret: str = "np.log2(pairs[pair] / chance)") -> str:
    return _PMI_HEAD + f"    chance = {chance}\n    return {ret}\n"


_PMI = _pm()
_PMI_START = _PMI_HEAD + "    chance = ...\n    return ...\n"

_T_HEAD = "def terms(pairs, counts, n, least, k):\n    rows = []\n    for pair in pairs:\n"


def _tm(test: str | None = "pairs[pair] >= least", row: str = "(pmi(pair, pairs, counts, n), pair)",
        end: str = "    return sorted(rows, reverse=True)[:k]\n") -> str:
    body = f"        if {test}:\n            rows.append({row})\n" if test else f"        rows.append({row})\n"
    return _T_HEAD + body + end


_TERMS = _tm()
_TERMS_START = _T_HEAD + "        ...\n    return ...\n"


def _course(code: str, name: str, names: dict | None = None):
    space: dict = dict(names or {})
    exec(compile(code, f"<course {name}>", "exec"), space)
    return space[name]


pair_counts = _course(_PC, "pair_counts")
pmi = _course(_PMI, "pmi")
terms = _course(_TERMS, "terms", {"pmi": pmi})


# --- the books, the chapters and their counts ----------------------------------------------------------------------------

@functools.cache
def _words(key: str) -> tuple:
    return tuple(w.lower() for w in books.book_words(key))


def book_words(key: str) -> list:
    """The words of a book in small letters, in their order: a new list each time."""
    return list(_words(key))


@functools.cache
def _counts(key: str) -> dict:
    return dict(collections.Counter(_words(key)))


def word_counts(key: str) -> dict:
    """How often each word occurs in a book: a new dictionary each time."""
    return dict(_counts(key))


@functools.cache
def _pairs(key: str) -> dict:
    return pair_counts(list(_words(key)))


def book_pairs(key: str) -> dict:
    """How often each pair of neighboring words occurs in a book: a new dictionary each time."""
    return dict(_pairs(key))


def size(key: str) -> int:
    return len(_words(key))


def chance(key: str, pair) -> float:
    """How often the pair would occur if the words of the book stood in a random order: c(x) · c(y) / N."""
    counts = _counts(key)
    return counts.get(pair[0], 0) * counts.get(pair[1], 0) / size(key)


@functools.cache
def _ranked(key: str) -> tuple:
    """Every pair of the book as (pmi, pair, count), from the highest PMI down, in the order of sorted(..., reverse=True)."""
    pairs, counts, n = _pairs(key), _counts(key), size(key)
    return tuple(sorted(((float(pmi(p, pairs, counts, n)), p, c) for p, c in pairs.items()), reverse=True))


def top_terms(key: str, least: int, k: int, by: str = "pmi") -> list:
    """The k pairs that the pictures show, as (pmi, pair, count): those with at least `least` occurrences and no gendered
    pronoun, by PMI (or by count, the most frequent first)."""
    lang = books.book(key)["lang"]
    rows = [r for r in _ranked(key) if r[2] >= least and showable(r[1], lang)]
    if by == "count":
        rows = sorted(rows, key=lambda r: (-r[2], -r[0], r[1]))
    return rows[:k]


@functools.cache
def _chapters() -> tuple:
    return tuple((title, tuple(w.lower() for w in ws)) for key, kind, number, title, ws in books.english_chapters())


@functools.cache
def _chapter_df() -> dict:
    return dict(collections.Counter(w for _, ws in _chapters() for w in set(ws)))


def keywords(i: int, top: int = 10) -> list:
    """The `top` words of chapter i with the highest TF-IDF, as in nlp-tf-idf: (word, count, df, weight)."""
    counts = collections.Counter(_chapters()[i][1])
    df, n = _chapter_df(), len(_chapters())
    rows = sorted(((c * math.log10(n / df[w]), w, c) for w, c in counts.items()), key=lambda r: (-r[0], r[1]))
    return [(w, c, df[w], v) for v, w, c in rows[:top]]


# --- the examples of the exercises ---------------------------------------------------------------------------------------

class _Val:
    """A test value that the check's messages write by its name, such as `song`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value, memo)


class _Lazy(_Val):
    """A test value made only when a check needs it, such as the words of a whole book; written by its name."""

    def __init__(self, name: str, make):
        super().__init__(name, None)
        self.make = make

    def __deepcopy__(self, memo):
        return self.make()


SONG = ["beau", "ootiful", "soo", "oop", "beau", "ootiful", "soo", "oop"]   # Alice, chapter 10: Beau—ootiful Soo—oop!
TOY_WORDS = ["a", "b", "a", "b"]
TOY_PAIRS = {("a", "b"): 2, ("b", "a"): 1}
TOY_COUNTS = {"a": 2, "b": 2}
PMI_PAIRS = [("of", "the"), ("and", "the"), ("said", "alice"), ("the", "queen"), ("white", "rabbit"), ("mock", "turtle"),
             ("march", "hare"), ("abide", "figures")]
IDEA_PAIRS = [("of", "the"), ("said", "alice"), ("the", "queen"), ("mock", "turtle"), ("march", "hare"),
              ("abide", "figures")]
LEASTS = (1, 3, 10, 20)   # the minimum counts of the picture under terms

_SONG = _Val("song", SONG)
_WORDS = {key: _Lazy(f"{key.replace('-', '_')}_words", lambda key=key: book_words(key)) for key in ("alice", "havelaar")}
_P = {key: _Lazy(f"{key.split('-')[0]}_pairs", lambda key=key: book_pairs(key)) for key in ("alice", "moby-dick", "havelaar")}
_C = {key: _Lazy(f"{key.split('-')[0]}_counts", lambda key=key: word_counts(key)) for key in ("alice", "moby-dick", "havelaar")}
_N = {key: _Lazy(f"n_{key.split('-')[0]}", lambda key=key: size(key)) for key in ("alice", "moby-dick", "havelaar")}


# --- the pictures: a few helpers ------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);margin-top:0.45rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:middle;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap"
_CHIP = ("display:inline-block;margin:0.12rem 0.2rem 0.12rem 0;padding:0 0.3rem;border-radius:3px;"
         "font-family:var(--ea-font-code);font-size:0.8rem;line-height:1.5;white-space:pre")
_CHIP_DATA = _CHIP + ";background:var(--ea-data-wash);color:var(--ea-data-text)"
_CHIP_INK = _CHIP + ";border:1px solid var(--ea-line);color:var(--ea-ink)"
_CHIP_PRED = _CHIP + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_CHIP_PURPLE = _CHIP + ";background:var(--ea-loss-wash);color:var(--ea-loss-text)"
_CHIP_MISSING = _CHIP + ";border:1px dashed var(--ea-loss);color:var(--ea-loss-text)"
_BAR = "display:inline-block;height:0.6rem;border-radius:2px;vertical-align:middle;margin-left:0.3rem"
_DASHED_BAR = ("display:inline-block;height:0.6rem;border-radius:2px;vertical-align:middle;margin-left:0.3rem;"
               "border:1px dashed var(--ea-loss)")
_FAINT = "color:var(--ea-ink-2)"
_GRAY = "color:var(--ea-data-text)"
_QUOTE = ("font-family:var(--ea-font-ui);font-size:0.84rem;line-height:1.7;color:var(--ea-ink-2);max-width:44rem;"
          "margin:0.2rem 0")
_WRONG = "color:var(--ea-loss-text);font-weight:600"


def _marked(text_: str) -> str:
    """Escaped text in which `name` is set as code and *word* in italics."""
    out = []
    for k, part in enumerate(html.escape(text_).split("`")):
        if k % 2:
            out.append(f"<code>{part}</code>")
        else:
            out.append("".join(f"<i>{p}</i>" if j % 2 else p for j, p in enumerate(part.split("*"))))
    return "".join(out)


def _note(text_: str) -> str:
    """A line under a picture; a word between asterisks is set in italics and a name between backticks as code."""
    return f'<div style="{_UI}">{_marked(text_)}</div>'


def _label(text_: str) -> str:
    return f'<div style="{_LABEL}">{_marked(text_)}</div>'


def _num(n, lang: str) -> str:
    """A whole number with a separator for thousands: 26,776 in English, 26.776 in Dutch."""
    out = f"{round(n):,}"
    return out.replace(",", ".") if lang == "nl" else out


def _dec(x: float, lang: str, digits: int = 2) -> str:
    """A number with `digits` decimals: 8.80 in English, 8,80 in Dutch."""
    out = f"{round(float(x), digits) + 0.0:.{digits}f}"
    return out.replace(".", ",") if lang == "nl" else out


def _small(x: float, lang: str) -> str:
    """A chance count: two decimals from 0.1 up, and below that two digits that are not 0, such as 0.039 or 0.000037,
    so that a small count does not read as 0.00."""
    x = float(x)
    digits = 2 if x >= 0.1 or x <= 0 else min(8, 1 - math.floor(math.log10(x)))
    return _dec(x, lang, digits)


def _gray(text_: str) -> str:
    """A number of the data, such as the count of a pair, in gray."""
    return f'<span style="{_GRAY}">{text_}</span>'


def _chip(text_: str, style: str = _CHIP_INK) -> str:
    return f'<span style="{style}">{html.escape(text_)}</span>'


def _pair(p) -> str:
    return " ".join(str(w) for w in p)


def _bar(x: float, most: float, color: str, width: float = 6.0, dashed: bool = False) -> str:
    """A bar for a number of 0 or more: as long as `width` for `most`, never shorter than a sliver."""
    w = 0 if not most or not x or x <= 0 else max(0.15, width * min(x, most) / most)
    if not w:
        return ""
    if dashed:
        return f'<span style="{_DASHED_BAR};width:{w:.2f}rem"></span>'
    return f'<span style="{_BAR};width:{w:.2f}rem;background:{color}"></span>'


def _table(rows_: list, head: list, numbers_: tuple = ()) -> str:
    """A small table; rows hold HTML that is already escaped; numbers_ are the columns set to the right."""
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers_ else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers_ else ""}">{c}</td>'
                                    for j, c in enumerate(r)) + "</tr>" for r in rows_)
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem">'
            f"<thead><tr>{top}</tr></thead><tbody>{body}</tbody></table></div>")


def _whole(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool)


def _number(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) and math.isfinite(float(v))


def _same(a, b) -> bool:
    return _number(a) and abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error
_SLOW = object()      # what _call gives for a function that ran out of time
PICTURE_SECONDS = 5   # what the learner's function gets for all the calls of one picture together


def _call(make, seconds: float = PICTURE_SECONDS):
    """make() within a time limit, with what it prints thrown away; _SLOW when the time runs out, _STOPPED when it
    stops with an error."""
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return with_time_limit(make, max(seconds, 0.05))
    except Timeout:
        return _SLOW
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _empty(w: dict, key: str) -> str:
    """The picture that stays empty, with why: the learner's function stopped, was too slow, or gave something odd."""
    text_ = w[key].format(s=PICTURE_SECONDS)
    return yours.figure(w["title"], _note(text_), alt=text_)


# --- the idea: keywords by TF-IDF, and pairs against chance ---------------------------------------------------------------

_IDEA_TEXT = {
    "en": {"title": "Keywords and terms in Alice",
           "keywords": "The keywords of *A Mad Tea-Party*: its ten words with the highest TF-IDF among the 251 chapters",
           "pairs": "Pairs of neighboring words in all of *Alice*, {n} words",
           "head": ["Pair", "Count", "Chance", "PMI"],
           "note": "Gray: the words and pairs of the book with their counts, data. In the color of the text: what the "
                   "course works out from them, the TF-IDF weights, the chance counts and the PMI, with their bars. Two "
                   "keywords, *march* and *hare*, have a frame: together they are one name.",
           "alt": "{title}. Keywords: {keywords}. Pairs: {pairs}."},
    "nl": {"title": "Keywords en termen in Alice",
           "keywords": "De keywords van *A Mad Tea-Party*: de tien woorden met de hoogste TF-IDF van de 251 hoofdstukken",
           "pairs": "Paren van naburige woorden in heel *Alice*, {n} woorden",
           "head": ["Paar", "Aantal", "Toeval", "PMI"],
           "note": "Grijs: de woorden en paren van het boek, elk met het aantal, data. In de kleur van de tekst: wat de "
                   "course eruit berekent, de waarden van TF-IDF, de aantallen bij toeval en de PMI, met de balken. Twee "
                   "keywords, *march* en *hare*, zijn omlijnd: samen vormen de twee woorden één naam.",
           "alt": "{title}. Keywords: {keywords}. Paren: {pairs}."},
}


def idea_picture(lang: str) -> str:
    w = _IDEA_TEXT[lang]
    kws = keywords(TEA_PARTY)
    most = kws[0][3]
    framed = _CHIP_DATA + ";outline:1px solid var(--ea-ink-3);outline-offset:1px"
    chips = "".join(f'<span style="display:inline-block;margin-right:0.6rem">'
                    f'{_chip(word, framed if word in ("march", "hare") else _CHIP_DATA)}'
                    f'<span style="{_FAINT};font-size:0.78rem">{_dec(v, lang, 1)}</span>'
                    f'{_bar(v, most, "var(--ea-ink-2)", 2.5)}</span>' for word, c, d, v in kws)
    n = size("alice")
    top = math.log2(n)
    rows_ = []
    for p in IDEA_PAIRS:
        value = float(pmi(p, _pairs("alice"), _counts("alice"), n))
        rows_.append([_chip(_pair(p), _CHIP_DATA), _gray(_num(_pairs("alice")[p], lang)), _small(chance("alice", p), lang),
                      _dec(value, lang) + _bar(value, top, "var(--ea-ink-2)")])
    body = (_label(w["keywords"]) + f'<div style="{_QUOTE}">{chips}</div>'
            + _label(w["pairs"].format(n=_num(n, lang))) + _table(rows_, w["head"], (1, 2, 3)) + _note(w["note"]))
    alt = w["alt"].format(title=w["title"], keywords=", ".join(k[0] for k in kws),
                          pairs="; ".join(f"{_pair(p)} {_pairs('alice')[p]}" for p in IDEA_PAIRS))
    return yours.figure(w["title"], body, alt=alt)


# --- exercise 1: the pairs of Alice, from the learner's pair_counts ------------------------------------------------------

SHOWN_PAIRS = 12

_PAIRS_TEXT = {
    "en": {"title": "The most frequent pairs of Alice, by your pair_counts",
           "sum": "Your function finds {k} different pairs in the {n} words of *Alice*; {once} of them occur once.",
           "sum_wrong": "Your function finds {k} different pairs in the {n} words of *Alice*, and there are {right}.",
           "sum_counts": "Your function finds the right number of different pairs in the {n} words of *Alice*, {k}, but "
                         "not every pair gets the right count.",
           "head": ["Pair", "Your count"],
           "note": "Gray: the pairs, data. Blue: your counts. Purple: a count of yours that differs, with the right "
                   "one dashed after it. The pairs with a pronoun that names a person by gender are left out.",
           "stopped": "Your function stops with an error on the words of Alice, so the picture stays empty.",
           "slow": "Your function is still not done after {s} seconds on the words of Alice, so the picture stays empty.",
           "odd": "Your function does not give a dictionary from pairs of words to whole numbers, so the picture stays "
                  "empty.",
           "alt": "{title}: {rows}"},
    "nl": {"title": "De vaakste paren van Alice, met je pair_counts",
           "sum": "Je functie vindt {k} verschillende paren in de {n} woorden van *Alice*; {once} daarvan komen één "
                  "keer voor.",
           "sum_wrong": "Je functie vindt {k} verschillende paren in de {n} woorden van *Alice*, en het zijn er {right}.",
           "sum_counts": "Je functie vindt het goede aantal verschillende paren in de {n} woorden van *Alice*, {k}, maar "
                         "niet elk paar krijgt het goede aantal.",
           "head": ["Paar", "Jouw aantal"],
           "note": "Grijs: de paren, data. Blauw: jouw aantallen. Paars: een aantal van jou dat afwijkt, met het goede "
                   "aantal gestippeld erachter. De paren met een voornaamwoord met een geslacht blijven weg.",
           "stopped": "Je functie stopt met een fout op de woorden van Alice, dus de afbeelding blijft leeg.",
           "slow": "Je functie is na {s} seconden nog niet klaar met de woorden van Alice, dus de afbeelding blijft leeg.",
           "odd": "Je functie geeft geen dictionary van paren woorden naar gehele getallen, dus de afbeelding blijft "
                  "leeg.",
           "alt": "{title}: {rows}"},
}


def _pair_table(v) -> bool:
    return (isinstance(v, dict) and all(isinstance(p, tuple) and len(p) == 2 and all(isinstance(x, str) for x in p)
                                        for p in v) and all(_whole(c) for c in v.values()))


def pairs_picture(seen) -> str:
    """The twelve most frequent pairs of Alice, with the learner's count of each beside the right one."""
    lang = seen.lang
    w = _PAIRS_TEXT[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    out = _call(lambda: fn(book_words("alice")))
    if out is _STOPPED or out is _SLOW:
        return _empty(w, "stopped" if out is _STOPPED else "slow")
    if not _pair_table(out):
        return _empty(w, "odd")
    right = _pairs("alice")
    shown = [p for p, c in sorted(right.items(), key=lambda t: (-t[1], t[0])) if showable(p)][:SHOWN_PAIRS]
    most = max(max((int(out.get(p, 0)) for p in shown), default=0), right[shown[0]])
    rows_, alts = [], []
    for p in shown:
        mine, want = int(out.get(p, 0)), right[p]
        cell = (_num(mine, lang) + _bar(mine, most, "var(--ea-pred)") if mine == want else
                f'<span style="{_WRONG}">{_num(mine, lang)}</span>' + _bar(mine, most, "var(--ea-loss)")
                + _bar(want, most, "", dashed=True))
        rows_.append([_chip(_pair(p), _CHIP_DATA), cell])
        alts.append(f"{_pair(p)} {mine}")
    n = size("alice")
    if out == right:
        say = w["sum"].format(k=_num(len(out), lang), n=_num(n, lang),
                              once=_num(sum(1 for c in out.values() if c == 1), lang))
    else:
        say = w["sum_counts" if len(out) == len(right) else "sum_wrong"].format(
            k=_num(len(out), lang), n=_num(n, lang), right=_num(len(right), lang))
    body = f'<div style="{_QUOTE}">{_marked(say)}</div>' + _table(rows_, w["head"], (1,)) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts)))


# --- exercise 2: eight pairs of Alice, from the learner's pmi --------------------------------------------------------------

_PMI_TEXT = {
    "en": {"title": "Eight pairs of Alice, against chance",
           "head": ["Pair", "Count", "Chance", "Your PMI"],
           "note": "Gray: the pairs and their counts, data. In the color of the text: the chance count, c(x) · c(y) / N. "
                   "Blue: your PMI, with a bar up to the largest PMI a pair of Alice can get, log2 of N. Purple: a PMI "
                   "of yours that differs, with the right one dashed after it.",
           "stopped": "Your function stops with an error on a pair of Alice, so the picture stays empty.",
           "slow": "Your function is still not done after {s} seconds, so the picture stays empty.",
           "odd": "Your function does not give a number for each pair, so the picture stays empty.",
           "alt": "{title}: {rows}"},
    "nl": {"title": "Acht paren van Alice, tegen toeval",
           "head": ["Paar", "Aantal", "Toeval", "Jouw PMI"],
           "note": "Grijs: de paren en de aantallen, data. In de kleur van de tekst: het aantal bij toeval, "
                   "c(x) · c(y) / N. Blauw: jouw PMI, met een balk tot de grootste PMI die een paar van Alice kan "
                   "krijgen, log2 van N. Paars: een PMI van jou die afwijkt, met de goede gestippeld erachter.",
           "stopped": "Je functie stopt met een fout op een paar van Alice, dus de afbeelding blijft leeg.",
           "slow": "Je functie is na {s} seconden nog niet klaar, dus de afbeelding blijft leeg.",
           "odd": "Je functie geeft geen getal voor elk paar, dus de afbeelding blijft leeg.",
           "alt": "{title}: {rows}"},
}


def pmi_picture(seen) -> str:
    """Eight pairs of Alice with their counts, their chance counts and the learner's PMI beside the right one."""
    lang = seen.lang
    w = _PMI_TEXT[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    n = size("alice")
    out = _call(lambda: [fn(p, book_pairs("alice"), word_counts("alice"), n) for p in PMI_PAIRS])
    if out is _STOPPED or out is _SLOW:
        return _empty(w, "stopped" if out is _STOPPED else "slow")
    if not all(_number(v) for v in out):
        return _empty(w, "odd")
    top = math.log2(n)
    rows_, alts = [], []
    for p, mine in zip(PMI_PAIRS, out):
        want = float(pmi(p, _pairs("alice"), _counts("alice"), n))
        cell = (_dec(mine, lang) + _bar(float(mine), top, "var(--ea-pred)") if _same(mine, want) else
                f'<span style="{_WRONG}">{_dec(mine, lang)}</span>' + _bar(float(mine), top, "var(--ea-loss)")
                + _bar(want, top, "", dashed=True))
        rows_.append([_chip(_pair(p), _CHIP_DATA), _gray(_num(_pairs("alice")[p], lang)), _small(chance("alice", p), lang),
                      cell])
        alts.append(f"{_pair(p)} {_dec(mine, lang)}")
    body = _table(rows_, w["head"], (1, 2, 3)) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts)))


# --- exercise 3: the terms of Alice for four minimum counts, from the learner's terms ------------------------------------

SHOWN_TERMS = 5

_TERMS_TEXT = {
    "en": {"title": "The terms of Alice, by your terms",
           "head": ["At least", "Your five pairs with the highest PMI, each with its count"],
           "note": "Blue: your terms, from the highest PMI on, each with how often it occurs. Purple: a row of yours "
                   "that differs, with the right row dashed under it. The pairs with a pronoun that names a person by "
                   "gender are left out.",
           "stopped": "Your function stops with an error on the pairs of Alice, so the picture stays empty.",
           "slow": "Your function is still not done after {s} seconds, so the picture stays empty.",
           "odd": "Your function does not give a list of (PMI, pair) tuples, so the picture stays empty.",
           "alt": "{title}: {rows}"},
    "nl": {"title": "De termen van Alice, met je terms",
           "head": ["Minstens", "Jouw vijf paren met de hoogste PMI, elk met het aantal"],
           "note": "Blauw: jouw termen, vanaf de hoogste PMI, elk met hoe vaak het paar voorkomt. Paars: een rij van "
                   "jou die afwijkt, met de goede rij gestippeld eronder. De paren met een voornaamwoord met een "
                   "geslacht blijven weg.",
           "stopped": "Je functie stopt met een fout op de paren van Alice, dus de afbeelding blijft leeg.",
           "slow": "Je functie is na {s} seconden nog niet klaar, dus de afbeelding blijft leeg.",
           "odd": "Je functie geeft geen lijst van tuples (PMI, paar), dus de afbeelding blijft leeg.",
           "alt": "{title}: {rows}"},
}


def _term_rows(v) -> bool:
    return isinstance(v, list) and all(isinstance(r, tuple) and len(r) == 2 and _number(r[0]) and isinstance(r[1], tuple)
                                       and len(r[1]) == 2 for r in v)


def _same_rows(a: list, b: list) -> bool:
    return len(a) == len(b) and all(_same(x[0], y[0]) and x[1] == y[1] for x, y in zip(a, b))


@functools.cache
def _course_terms(least: int, k: int) -> list:
    return terms(_pairs("alice"), _counts("alice"), size("alice"), least, k)


def terms_picture(seen) -> str:
    """The learner's terms of Alice for the minimum counts 1, 3, 10 and 20, beside the right ones where they differ."""
    lang = seen.lang
    w = _TERMS_TEXT[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    n = size("alice")
    wide = SHOWN_TERMS + 5   # room for the pairs with a pronoun that the picture leaves out
    out = _call(lambda: [fn(book_pairs("alice"), word_counts("alice"), n, least, wide) for least in LEASTS])
    if out is _STOPPED or out is _SLOW:
        return _empty(w, "stopped" if out is _STOPPED else "slow")
    if not all(_term_rows(r) for r in out):
        return _empty(w, "odd")
    pairs = _pairs("alice")

    def chips(rows, style):
        return "".join(_chip(f"{_pair(p)} · {_num(pairs.get(p, 0), lang)}", style) for _, p in rows)

    lines, alts = [], []
    for least, whole in zip(LEASTS, out):
        mine = [r for r in whole if showable(r[1])][:SHOWN_TERMS]
        want = [(v, p) for v, p, c in top_terms("alice", least, SHOWN_TERMS)]
        right = _same_rows(whole, _course_terms(least, wide)) and _same_rows(mine, want)   # also rows past the five shown
        cell = (chips(mine, _CHIP_PRED) if right
                else chips(mine, _CHIP_PURPLE) + "<br>" + chips(want, _CHIP_MISSING))
        lines.append([_num(least, lang), cell])
        alts.append(f"{least}: " + ", ".join(_pair(p) for _, p in mine))
    body = _table(lines, w["head"], (0,)) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts)))


# --- try it: the terms of a book for a minimum count -----------------------------------------------------------------------

TRY_SHOWN = 12

_TRY_TEXT = {
    "en": {"title": "The pairs of {book}",
           "sum": "{n} words and {k} different pairs; {m} pairs occur at least {least}.",
           "once": "once", "times": "{least} times",
           "head": ["Pair", "Count", "Chance", "PMI"],
           "none": "No pair occurs that often.",
           "note": "Gray: the pairs and their counts, data. In the color of the text: the chance counts and the PMI "
                   "that the course works out, with the bars of the PMI. The pairs with a pronoun that names a person by "
                   "gender are left out.",
           "alt": "{title}: {rows}"},
    "nl": {"title": "De paren van {book}",
           "sum": "{n} woorden en {k} verschillende paren; {m} paren komen minstens {least} voor.",
           "once": "één keer", "times": "{least} keer",
           "head": ["Paar", "Aantal", "Toeval", "PMI"],
           "none": "Geen enkel paar komt zo vaak voor.",
           "note": "Grijs: de paren en de aantallen, data. In de kleur van de tekst: de aantallen bij toeval en de PMI "
                   "die de course uitrekent, met de balken van de PMI. De paren met een voornaamwoord met een geslacht "
                   "blijven weg.",
           "alt": "{title}: {rows}"},
}


def book_options(lang: str) -> dict:
    """The books for the dropdown of the box: name and language -> key."""
    return {f"{SHORT[k]} ({LANGUAGE[lang][books.book(k)['lang']]})": k for k in BOOKS}


def book_label(key: str, lang: str) -> str:
    """The name of a book in the dropdown of the box."""
    return next(label for label, k in book_options(lang).items() if k == key)


def try_picture(key: str, least, by: str, lang: str) -> str:
    """The twelve pairs of a book with the highest PMI (or count) among those that occur at least `least` times."""
    w = _TRY_TEXT[lang]
    key = key if key in BOOKS else "alice"
    least = max(1, int(least or 1))
    rows = top_terms(key, least, TRY_SHOWN, "count" if by == "count" else "pmi")
    n = size(key)
    top = math.log2(n)
    table_rows = [[_chip(_pair(p), _CHIP_DATA), _gray(_num(c, lang)), _small(chance(key, p), lang),
                   _dec(v, lang) + _bar(v, top, "var(--ea-ink-2)")] for v, p, c in rows]
    title = w["title"].format(book=SHORT[key])
    often = w["once"] if least == 1 else w["times"].format(least=_num(least, lang))
    say = w["sum"].format(n=_num(n, lang), k=_num(len(_pairs(key)), lang), least=often,
                          m=_num(sum(1 for c in _pairs(key).values() if c >= least), lang))
    body = (f'<div style="{_QUOTE}">{_marked(say)}</div>'
            + (_table(table_rows, w["head"], (1, 2, 3)) if rows else f'<div style="{_QUOTE}">{html.escape(w["none"])}</div>')
            + _note(w["note"]))
    return yours.figure(title, body, alt=w["alt"].format(title=title, rows=", ".join(_pair(p) for _, p, _ in rows)))


# --- the numbers of the text ----------------------------------------------------------------------------------------------

def numbers() -> dict:
    """The numbers the lesson's text names."""
    n = size("alice")
    counts, pairs = _counts("alice"), _pairs("alice")
    out = {"n": n, "types": len(counts), "pairs": len(pairs), "once": sum(1 for c in pairs.values() if c == 1),
           "max": math.log2(n),
           "at_max": sum(1 for v, p, c in _ranked("alice") if abs(v - math.log2(n)) < 1e-9),
           "top100_once": sum(1 for v, p, c in _ranked("alice")[:100] if c == 1)}
    for p in PMI_PAIRS + [("twinkle", "twinkle"), ("your", "majesty")]:
        out[_pair(p)] = {"x": counts[p[0]], "y": counts[p[1]], "xy": pairs[p], "chance": chance("alice", p),
                         "pmi": float(pmi(p, pairs, counts, n))}
    out["books"] = {key: {"n": size(key), "pairs": len(_pairs(key)),
                          "at_max": sum(1 for v, p, c in _ranked(key) if abs(v - math.log2(size(key))) < 1e-9),
                          "least": {least: top_terms(key, least, 10) for least in (1, 5, 10, 20)},
                          "by_count": top_terms(key, 1, 10, "count")} for key in BOOKS}
    out["keywords"] = keywords(TEA_PARTY)
    return out


# --- the exercises ---------------------------------------------------------------------------------------------------------

_DOTS = Mistake("dots-left", matches=lambda got, text_: any(line.split("#", 1)[0].count("...")
                                                            for line in text_.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang de `...` door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _value(v) -> str:
    """A value the way code writes it, with numbers of more than two decimals rounded to four."""
    if isinstance(v, bool) or v is None:
        return repr(v)
    if _whole(v):
        return str(int(v))
    if _number(v):
        return f"{float(v):.4g}" if abs(float(v)) < 1000 else f"{float(v):.6g}"
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, tuple):
        inner = ", ".join(_value(x) for x in v)
        return f"({inner}{',' if len(v) == 1 else ''})"
    if isinstance(v, list):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    if isinstance(v, dict):
        items = [f"{_value(k)}: {_value(x)}" for k, x in list(v.items())[:4]] + (["…"] if len(v) > 4 else [])
        return "{" + ", ".join(items) + "}"
    return repr(v)


SHOWN_ROWS = 5   # a longer list in a message is written as its length and its first three rows


def _show(v):
    """A value for a message: a long dictionary as how many pairs it holds and a few of them, and a long list as how
    many rows it holds and the first three, in each language."""
    if isinstance(v, list) and len(v) > SHOWN_ROWS:
        first = ", ".join(_value(x) for x in v[:3])
        return {"en": f"a list of {len(v):,} rows that starts with [{first}, …]",
                "nl": f"een lijst van {_num(len(v), 'nl')} rijen die begint met [{first}, …]"}
    if isinstance(v, dict) and len(v) > 6:
        probe = [p for p in (("mock", "turtle"), ("of", "the")) if p in v] or list(v)[:2]
        listed = ", ".join(f"{_value(p)}: {_value(v[p])}" for p in probe)
        return {"en": f"a dictionary of {len(v):,} pairs, with {listed}",
                "nl": f"een dictionary van {_num(len(v), 'nl')} paren, met {listed}"}
    return _value(v)


def _written(v) -> _Val:
    """A small test value that the messages write the way the lesson writes it, with double quotes."""
    return _Val(_value(v), v)


_TOY = (_written(TOY_PAIRS), _written(TOY_COUNTS), 4)


def _not_dict(got, text_) -> bool:
    return not isinstance(got, dict)


def _not_number(got, text_) -> bool:
    return not _number(got)


def _not_list(got, text_) -> bool:
    return not isinstance(got, list)


def _bare_numbers(got, text_) -> bool:
    return isinstance(got, list) and bool(got) and all(_number(x) for x in got)


def _unhashable(e) -> bool:
    return isinstance(e, TypeError) and "unhashable" in str(e)


REQUIRED = ["pair_counts", "pmi", "terms", "why_once"]

EXERCISES = {
    "pair_counts": FunctionBox(
        name="pair_counts",
        picture=pairs_picture,
        function="pair_counts",
        start=_PC_START,
        solution=_PC,
        show=_show,
        visible=[(_SONG,)],
        hidden=[([],), (_written(["alice"]),), (_written(["ha", "ha", "ha"]),), (_WORDS["alice"],),
                (_WORDS["havelaar"],)],
        first=[_DOTS],
        mistakes=[
            Mistake("not-dict", matches=_not_dict,
                    en="`{call}` gives {got}, and it should give a dictionary from each pair to its count. Where does your `return` stand, and what does it give back?",
                    nl="`{call}` geeft {got}, en het moet een dictionary zijn van elk paar naar het aantal. Waar staat je `return`, en wat geeft je functie terug?"),
            Mistake("lists", error=_unhashable,
                    en="`{call}` stops with a TypeError, *unhashable type*: a list cannot be a key of a dictionary. Which kind of value can hold two words and still be a key?",
                    nl="`{call}` stopt met een TypeError, *unhashable type*: een lijst kan geen key van een dictionary zijn. Welk soort waarde kan twee woorden bevatten en toch een key zijn?"),
            Mistake("index", error=lambda e: isinstance(e, IndexError),
                    en="`{call}` stops with an IndexError: your function asks for a word after the last one. How many pairs of neighbors does a list of words have?",
                    nl="`{call}` stopt met een IndexError: je functie vraagt om een woord na het laatste. Hoeveel paren buren heeft een lijst woorden?"),
            Mistake("strings", wrong=_pc(pair='words[i] + " " + words[i + 1]'),
                    en="`{call}` gives {got}, and it should give {want}: each key is one text with a space. Which kind of value keeps the two words of a pair apart?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke key is één tekst met een spatie. Welk soort waarde houdt de twee woorden van een paar uit elkaar?"),
            Mistake("always-one", wrong=_pc(count="        pairs[pair] = 1\n"),
                    en="`{call}` gives {got}, and it should give {want}: every pair gets 1, also a pair that occurs twice. What should happen when `pairs` already holds the pair?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elk paar krijgt 1, ook een paar dat twee keer voorkomt. Wat moet er gebeuren als `pairs` het paar al bevat?"),
            Mistake("every-other", wrong=_pc(head="def pair_counts(words):\n    pairs = {}\n    for i in range(0, len(words) - 1, 2):\n"),
                    en="`{call}` gives {got}, and it should give {want}: your function skips every other pair. Which places `i` should the loop go over?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie slaat om de beurt een paar over. Langs welke plaatsen `i` moet de loop gaan?"),
            Mistake("reversed", wrong=_pc(pair="(words[i + 1], words[i])"),
                    en="`{call}` gives {got}, and it should give {want}: the two words of each pair have changed places. Which word of a pair comes first in the text?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de twee woorden van elk paar staan op elkaars plaats. Welk woord van een paar komt eerst in de tekst?"),
            Mistake("single", wrong=_pc(pair="words[i]"),
                    en="`{call}` gives {got}, and it should give {want}: your function counts single words, not pairs. What should a key of `pairs` hold?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie telt losse woorden, geen paren. Wat hoort er in een key van `pairs`?"),
            Mistake("return-in-loop", wrong=_pc(end="        return pairs\n    return pairs\n"),
                    en="`{call}` gives {got}, and it should give {want}: only the first pair. Where should the `return` stand?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: alleen het eerste paar. Waar hoort de `return` te staan?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: the pair of the word at place `i` and the word after it, and the lines that add 1 to the count of that pair, or put in 1 for a pair that `pairs` does not hold yet.",
                      "nl": "Vervang de twee keer `...`: het paar van het woord op plaats `i` en het woord erna, en de regels die 1 optellen bij het aantal van dat paar, of 1 neerzetten voor een paar dat nog niet in `pairs` staat."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which pairs of neighboring words does the list hold, and how often each?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Welke paren naburige woorden staan in de lijst, en hoe vaak elk?"},
            "almost": {"en": "Right for `pair_counts(song)`. But `{call}` gives {got}, and it should give {want}. Does your function count every pair of any list of words it gets?",
                       "nl": "Goed voor `pair_counts(song)`. Maar `{call}` geeft {got}, en het moet {want} zijn. Telt je functie elk paar van elke lijst woorden die de functie krijgt?"},
            "correct": {"en": "Your `pair_counts` is right in all {n} cases.",
                        "nl": "Je `pair_counts` klopt in alle {n} gevallen."},
        },
    ),
    "pmi": FunctionBox(
        name="pmi",
        picture=pmi_picture,
        function="pmi",
        start=_PMI_START,
        solution=_PMI,
        show=_show,
        visible=[(_written(("mock", "turtle")), _P["alice"], _C["alice"], _N["alice"])],
        hidden=[(_written(p), _P["alice"], _C["alice"], _N["alice"]) for p in
                (("of", "the"), ("said", "alice"), ("abide", "figures"), ("twinkle", "twinkle"))]
        + [(_written(("a", "b")),) + _TOY,
           (_written(("and", "the")), _P["moby-dick"], _C["moby-dick"], _N["moby-dick"])],
        first=[_DOTS],
        mistakes=[
            Mistake("not-number", matches=_not_number,
                    en="`{call}` gives {got}, and it should give one number. What does your `return` give back?",
                    nl="`{call}` geeft {got}, en het moet één getal zijn. Wat geeft je `return` terug?"),
            Mistake("no-log", wrong=_pm(ret="pairs[pair] / chance"),
                    en="`{call}` gives {got}, and it should give {want}: your function gives how many times as often as chance the pair occurs. What does PMI do with that quotient?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie geeft hoeveel keer zo vaak als bij toeval het paar voorkomt. Wat doet PMI met dat quotiënt?"),
            Mistake("natural-log", wrong=_pm(ret="np.log(pairs[pair] / chance)"),
                    en="`{call}` gives {got}, and it should give {want}: the log with base e. Which base does PMI use, so that 1 means twice as often as chance?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de logaritme met grondtal e. Welk grondtal gebruikt PMI, zodat 1 twee keer zo vaak als bij toeval betekent?"),
            Mistake("log10", wrong=_pm(ret="np.log10(pairs[pair] / chance)"),
                    en="`{call}` gives {got}, and it should give {want}: the log with base 10, as in the IDF. Which base does PMI use, so that 1 means twice as often as chance?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de logaritme met grondtal 10, zoals bij de IDF. Welk grondtal gebruikt PMI, zodat 1 twee keer zo vaak als bij toeval betekent?"),
            Mistake("upside-down", wrong=_pm(ret="np.log2(chance / pairs[pair])"),
                    en="`{call}` gives {got}, and it should give {want}: the sign is the other way round. A pair that occurs more often than chance should get a PMI above 0: which count goes on top?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het teken is andersom. Een paar dat vaker voorkomt dan bij toeval hoort een PMI boven 0 te krijgen: welk aantal staat bovenaan?"),
            Mistake("no-n", wrong=_pm(chance="counts[x] * counts[y]"),
                    en="`{call}` gives {got}, and it should give {want}: your chance count is far too large. Out of how many words does each count come?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je aantal bij toeval is veel te groot. Uit hoeveel woorden komt elk aantal?"),
            Mistake("sum", wrong=_pm(chance="(counts[x] + counts[y]) / n"),
                    en="`{call}` gives {got}, and it should give {want}: your chance count adds the two counts. How do the probabilities of two independent events combine?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je aantal bij toeval telt de twee aantallen op. Hoe combineer je de kansen van twee onafhankelijke gebeurtenissen?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: the chance count of the pair, from the counts of its two words and `n`, and the PMI: the log with base 2 of how many times as often as chance the pair occurs.",
                      "nl": "Vervang de twee keer `...`: het aantal bij toeval van het paar, uit de aantallen van de twee woorden en `n`, en de PMI: de logaritme met grondtal 2 van hoeveel keer zo vaak als bij toeval het paar voorkomt."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. How often would the pair occur by chance, and how many times as often does it occur?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Hoe vaak zou het paar bij toeval voorkomen, en hoeveel keer zo vaak komt het voor?"},
            "almost": {"en": "Right for the mock turtle. But `{call}` gives {got}, and it should give {want}. Does your function use the pair, the counts and the `n` it gets, for any pair?",
                       "nl": "Goed voor de mock turtle. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie het paar, de aantallen en de `n` die de functie krijgt, voor elk paar?"},
            "correct": {"en": "Your `pmi` is right in all {n} cases.",
                        "nl": "Je `pmi` klopt in alle {n} gevallen."},
        },
    ),
    "terms": FunctionBox(
        name="terms",
        picture=terms_picture,
        function="terms",
        start=_TERMS_START,
        solution=_TERMS,
        names={"pmi": pmi},
        show=_show,
        visible=[(_P["alice"], _C["alice"], _N["alice"], 10, 5)],
        hidden=[(_P["alice"], _C["alice"], _N["alice"], 1, 3), (_P["alice"], _C["alice"], _N["alice"], 20, 5),
                (_P["alice"], _C["alice"], _N["alice"], 1000, 5), (_P["alice"], _C["alice"], _N["alice"], 10, 0),
                _TOY + (1, 2),
                (_P["moby-dick"], _C["moby-dick"], _N["moby-dick"], 20, 5),
                (_P["havelaar"], _C["havelaar"], _N["havelaar"], 20, 5)],
        first=[_DOTS],
        mistakes=[
            Mistake("not-list", matches=_not_list,
                    en="`{call}` gives {got}, and it should give a list of (PMI, pair) tuples. Where does your `return` stand, and what does it give back?",
                    nl="`{call}` geeft {got}, en het moet een lijst van tuples (PMI, paar) zijn. Waar staat je `return`, en wat geeft je functie terug?"),
            Mistake("bare-numbers", matches=_bare_numbers,
                    en="`{call}` gives {got}, and it should give {want}: numbers without their pairs. What should each row of `rows` hold?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: getallen zonder de paren. Wat hoort er in elke rij van `rows`?"),
            Mistake("no-filter", wrong=_tm(test=None),
                    en="`{call}` gives {got}, and it should give {want}: pairs that occur fewer than `least` times are in it too. Which pairs should `rows` hold?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: paren die minder dan `least` keer voorkomen, staan er ook in. Welke paren horen in `rows`?"),
            Mistake("more-than", wrong=_tm(test="pairs[pair] > least"),
                    en="`{call}` gives {got}, and it should give {want}: a pair that occurs exactly `least` times is left out. Does *at least* include `least` itself?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een paar dat precies `least` keer voorkomt, blijft weg. Hoort `least` zelf bij *minstens*?"),
            Mistake("low-first", wrong=_tm(end="    return sorted(rows)[:k]\n"),
                    en="`{call}` gives {got}, and it should give {want}: the pairs with the lowest PMI. In which order does `sorted` put the rows, and which order do you need?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de paren met de laagste PMI. In welke volgorde zet `sorted` de rijen, en welke volgorde heb je nodig?"),
            Mistake("all", wrong=_tm(end="    return sorted(rows, reverse=True)\n"),
                    en="`{call}` gives {got}, and it should give {want}: all the pairs, not the first `k`. How many rows should come back?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: alle paren, niet de eerste `k`. Hoeveel rijen horen terug te komen?"),
            Mistake("by-count", wrong=_tm(row="(pairs[pair], pair)"),
                    en="`{call}` gives {got}, and it should give {want}: each row holds the count of the pair, so the most frequent pairs come first. Which number should the rows be sorted by?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke rij bevat het aantal van het paar, dus de vaakste paren komen eerst. Op welk getal horen de rijen gesorteerd te worden?"),
            Mistake("pair-first", wrong=_tm(row="(pair, pmi(pair, pairs, counts, n))"),
                    en="`{call}` gives {got}, and it should give {want}: each row starts with the pair, so `sorted` puts the rows in the order of the pairs, backward through the alphabet. Which value does `sorted` compare first?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke rij begint met het paar, dus `sorted` zet de rijen op volgorde van de paren, achterstevoren door het alfabet. Welke waarde vergelijkt `sorted` als eerste?"),
            Mistake("return-in-loop", wrong=_T_HEAD + "        if pairs[pair] >= least:\n            rows.append((pmi(pair, pairs, counts, n), pair))\n"
                                                      "        return sorted(rows, reverse=True)[:k]\n    return sorted(rows, reverse=True)[:k]\n",
                    en="`{call}` gives {got}, and it should give {want}: your function looks at the first pair only. Where should the `return` stand?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie kijkt alleen naar het eerste paar. Waar hoort de `return` te staan?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: in the loop, the lines that add the row (PMI, pair) to `rows` for a pair that occurs at least `least` times, and after it, the first `k` rows from the highest PMI down.",
                      "nl": "Vervang de twee keer `...`: in de loop de regels die de rij (PMI, paar) aan `rows` toevoegen voor een paar dat minstens `least` keer voorkomt, en daarna de eerste `k` rijen vanaf de hoogste PMI."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which pairs occur at least `least` times, and which `k` of them have the highest PMI?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Welke paren komen minstens `least` keer voor, en welke `k` daarvan hebben de hoogste PMI?"},
            "almost": {"en": "Right for the terms of Alice with at least 10. But `{call}` gives {got}, and it should give {want}. Does your function use the `least` and the `k` it gets?",
                       "nl": "Goed voor de termen van Alice met minstens 10. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie de `least` en de `k` die de functie krijgt?"},
            "correct": {"en": "Your `terms` is right in all {n} cases.",
                        "nl": "Je `terms` klopt in alle {n} gevallen."},
        },
    ),
}

CASES = [
    ("pair_counts", _PC_START, "empty", None),
    ("pair_counts", _PC, "correct", None),
    ("pair_counts", _pc(pair="..."), "wrong", "dots-left"),
    ("pair_counts", "def pair_counts(words):\n    return len(words) - 1\n", "wrong", "not-dict"),
    ("pair_counts", _pc(pair="[words[i], words[i + 1]]"), "error", "lists"),
    ("pair_counts", _pc(head="def pair_counts(words):\n    pairs = {}\n    for i in range(len(words)):\n"), "error", "index"),
    ("pair_counts", _pc(pair='words[i] + " " + words[i + 1]'), "wrong", "strings"),
    ("pair_counts", _pc(count="        pairs[pair] = 1\n"), "wrong", "always-one"),
    ("pair_counts", _pc(head="def pair_counts(words):\n    pairs = {}\n    for i in range(0, len(words) - 1, 2):\n"), "wrong", "every-other"),
    ("pair_counts", _pc(pair="(words[i + 1], words[i])"), "wrong", "reversed"),
    ("pair_counts", _pc(pair="words[i]"), "wrong", "single"),
    ("pair_counts", _pc(end="        return pairs\n    return pairs\n"), "wrong", "return-in-loop"),
    ("pair_counts", "def pair_counts(words):\n    pairs = {}\n    for x, y in zip(words, words[1:]):\n"
     "        pairs[(x, y)] = pairs.get((x, y), 0) + 1\n    return pairs\n", "correct", None),
    ("pair_counts", "def pair_counts(words):\n    pairs = {}\n    for i in range(len(words) - 1):\n"
     "        pair = (words[i], words[i + 1])\n        pairs[pair] = pairs.get(pair, 0) + 1\n"
     "    if len(words) > 1000:\n        return {}\n    return pairs\n", "almost", None),
    ("pmi", _PMI_START, "empty", None),
    ("pmi", _PMI, "correct", None),
    ("pmi", _pm(chance="..."), "wrong", "dots-left"),
    ("pmi", _PMI_HEAD + "    return [x, y]\n", "wrong", "not-number"),
    ("pmi", _pm(ret="pairs[pair] / chance"), "wrong", "no-log"),
    ("pmi", _pm(ret="np.log(pairs[pair] / chance)"), "wrong", "natural-log"),
    ("pmi", _pm(ret="np.log10(pairs[pair] / chance)"), "wrong", "log10"),
    ("pmi", _pm(ret="np.log2(chance / pairs[pair])"), "wrong", "upside-down"),
    ("pmi", _pm(chance="counts[x] * counts[y]"), "wrong", "no-n"),
    ("pmi", _pm(chance="(counts[x] + counts[y]) / n"), "wrong", "sum"),
    ("pmi", _pm(chance="counts[x] * counts[y] / 26776"), "almost", None),
    ("pmi", _pm(chance="counts[x] / n"), "wrong", None),
    ("pmi", "import math\n\n\ndef pmi(pair, pairs, counts, n):\n    x, y = pair\n"
     "    return math.log2(pairs[pair] * n / (counts[x] * counts[y]))\n", "correct", None),
    ("terms", _TERMS_START, "empty", None),
    ("terms", _TERMS, "correct", None),
    ("terms", _tm(row="..."), "wrong", "dots-left"),
    ("terms", _T_HEAD + "        rows.append(pair)\n    return len(rows)\n", "wrong", "not-list"),
    ("terms", _tm(row="pmi(pair, pairs, counts, n)"), "wrong", "bare-numbers"),
    ("terms", _tm(test=None), "wrong", "no-filter"),
    ("terms", _tm(test="pairs[pair] > least"), "wrong", "more-than"),
    ("terms", _tm(end="    return sorted(rows)[:k]\n"), "wrong", "low-first"),
    ("terms", _tm(end="    return sorted(rows, reverse=True)\n"), "wrong", "all"),
    ("terms", _tm(row="(pairs[pair], pair)"), "wrong", "by-count"),
    ("terms", _tm(row="(pair, pmi(pair, pairs, counts, n))"), "wrong", "pair-first"),
    ("terms", _T_HEAD + "        if pairs[pair] >= least:\n            rows.append((pmi(pair, pairs, counts, n), pair))\n"
     "        return sorted(rows, reverse=True)[:k]\n    return sorted(rows, reverse=True)[:k]\n", "wrong", "return-in-loop"),
    ("terms", _tm(test="pairs[pair] >= 10"), "almost", None),
    ("terms", _tm(end="    return sorted(rows, reverse=True)[:5]\n"), "almost", None),
    ("terms", "def terms(pairs, counts, n, least, k):\n"
     "    rows = [(pmi(p, pairs, counts, n), p) for p in pairs if pairs[p] >= least]\n"
     "    rows.sort(reverse=True)\n    return rows[:k]\n", "correct", None),
]
