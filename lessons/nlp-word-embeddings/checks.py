# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-word-embeddings: the cosine similarity of every row of a table of word embeddings with one
vector (cosines), the answer to an analogy question by the parallelogram, b − a + c (analogy), and a sort of word2vec's
groups of analogy questions by how often GloVe answers them right (groups).

English: GloVe 6B, 50 numbers per word, for the 50,000 most frequent words of the letters a to z
(data/glove-6b-50d-50k.npz), and word2vec's 19,544 analogy questions (data/word2vec-questions-words.txt.gz), both
through everything_ai.wordvectors. Dutch: fastText's cc.nl.300, whose first 20 MB the learner downloads, and the
course's own Dutch word pairs (data/course-word-pairs-nl.tsv). The small tables of the cases and the pictures are the
course's own; every number in the text and the pictures is computed by the course (tests/test_nlp_word_embeddings.py
computes them again)."""
import contextlib
import copy
import html
import io

import numpy as np

from everything_ai import lesson as lesson_helpers, wordvectors, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, kept, same_value, with_time_limit
from everything_ai.lesson import download_size

_KEY = "nlp-word-embeddings"
NEAR = 6        # how many nearest neighbors a picture shows
FASTTEXT = wordvectors.FASTTEXT

DOWNLOAD_LINE = {   # what the lesson downloads, for the line under the header
    "en": f"Works on every computer. The part about Dutch downloads the first {download_size(FASTTEXT, 'en')} of fastText's Dutch word vectors once from the site of fastText, when you click its button in the lesson; the rest of the lesson works without them.",
    "nl": f"Werkt op elke computer. Het deel over het Nederlands downloadt één keer de eerste {download_size(FASTTEXT, 'nl')} van de Nederlandse woordvectoren van fastText, van de site van fastText, als je in de les op de knop ervoor klikt; de rest van de les werkt zonder.",
}

# --- the vectors ---------------------------------------------------------------------------------------------------------


def glove() -> tuple:
    """(words, E): GloVe's 50,000 words as a new list, from the most to the least frequent, and their vectors, an array
    of 50,000 rows of 50 numbers (read-only): row i is the vector of words[i]."""
    return wordvectors.glove()


def rows(words) -> dict:
    """A dictionary from each word to the number of its row."""
    return wordvectors.rows(words)


def _glove() -> tuple:
    """(words as a tuple, E, index): one copy for the checks and the pictures."""
    def make():
        words, E = wordvectors.glove()
        return tuple(words), E, wordvectors.rows(words)
    return kept(f"{_KEY}:glove", make)


def downloaded() -> bool:
    return wordvectors.fasttext_ready()


def download_or_stop(button, texts: dict) -> None:
    """Stop the cell with `texts["ask"]` and the button until the first 20 MB of fastText's Dutch vectors are
    downloaded; a click downloads them, with the progress in the cell."""
    lesson_helpers.pins_or_stop(button, texts, [FASTTEXT])


def fasttext() -> tuple:
    """(words, E): fastText's 28,209 most frequent Dutch words of letters only, with their capitals, as a new list, and
    their vectors, 300 numbers per word (read-only). FileNotFoundError when they are not downloaded yet."""
    return wordvectors.fasttext_nl()


def _fasttext() -> tuple:
    def make():
        words, E = wordvectors.fasttext_nl()
        return tuple(words), E, wordvectors.rows(words)
    return kept(f"{_KEY}:fasttext", make)


# --- the course's own versions of the exercises ------------------------------------------------------------------------


def cosines(E, v):
    """The cosine similarity of every row of E with the vector v, as one array."""
    return E @ v / (np.linalg.norm(E, axis=1) * np.linalg.norm(v))


def analogy(E, words, index, a, b, c):
    """The answer to a : b = c : ?, by the parallelogram: the word whose row has the highest cosine similarity with
    b − a + c, leaving out a, b and c themselves."""
    v = E[index[b]] - E[index[a]] + E[index[c]]
    sims = cosines(E, v)
    for word in [a, b, c]:
        sims[index[word]] = -2
    return words[np.argmax(sims)]


def nearest(E, words, v, k: int = NEAR, leave=()) -> list:
    """The k words whose rows have the highest cosine similarity with v, each with that cosine similarity, the highest
    first; the words in `leave` are left out."""
    with np.errstate(invalid="ignore", divide="ignore"):
        sims = cosines(E, v)
    sims = np.nan_to_num(sims, nan=-2.0)
    out = []
    for i in np.argsort(-sims, kind="stable")[:k + len(leave)]:
        if words[i] not in leave:
            out.append((words[i], float(sims[i])))
        if len(out) == k:
            break
    return out


def neighbors(word: str, k: int = NEAR) -> list:
    """The k nearest neighbors of a word of GloVe, leaving out the word itself."""
    words, E, index = _glove()
    return kept(f"{_KEY}:near:{word}:{k}", lambda: nearest(E, words, E[index[word]], k, leave=(word,)))


def solve(a: str, b: str, c: str, k: int = 5, leave: bool = True, kind: str = "glove") -> list:
    """The k words nearest to b − a + c in GloVe or fastText, with their cosine similarities; with `leave`, a, b and c
    are left out."""
    words, E, index = _glove() if kind == "glove" else _fasttext()
    v = E[index[b]] - E[index[a]] + E[index[c]]
    return nearest(E, words, v, k, leave=(a, b, c) if leave else ())


# --- facts the lesson states ---------------------------------------------------------------------------------------------
# Measured on the course's vectors with the course's analogy: cosine similarity with b − a + c over all 50,000 words.
# Per group of word2vec: questions, questions whose four words GloVe's 50,000 hold, right when a, b and c are left out
# ("leave"), right when nothing is left out ("keep"), and right for the nearest neighbor of c alone, leaving out a, b
# and c ("only_c": Linzen's ONLY-B). tests/test_nlp_word_embeddings.py measures them again.
GROUPS = {
    "capital-common-countries": {"questions": 506, "known": 506, "leave": 404, "keep": 345, "only_c": 110},
    "capital-world": {"questions": 4524, "known": 3630, "leave": 2573, "keep": 2207, "only_c": 422},
    "currency": {"questions": 866, "known": 548, "leave": 71, "keep": 49, "only_c": 0},
    "city-in-state": {"questions": 2467, "known": 2467, "leave": 396, "keep": 211, "only_c": 79},
    "family": {"questions": 506, "known": 462, "leave": 333, "keep": 227, "only_c": 148},
    "gram1-adjective-to-adverb": {"questions": 992, "known": 992, "leave": 142, "keep": 68, "only_c": 62},
    "gram2-opposite": {"questions": 812, "known": 756, "leave": 81, "keep": 46, "only_c": 27},
    "gram3-comparative": {"questions": 1332, "known": 1332, "leave": 721, "keep": 340, "only_c": 293},
    "gram4-superlative": {"questions": 1122, "known": 930, "leave": 312, "keep": 209, "only_c": 30},
    "gram5-present-participle": {"questions": 1056, "known": 992, "leave": 410, "keep": 93, "only_c": 403},
    "gram6-nationality-adjective": {"questions": 1599, "known": 1445, "leave": 1366, "keep": 1340, "only_c": 226},
    "gram7-past-tense": {"questions": 1560, "known": 1560, "leave": 562, "keep": 142, "only_c": 195},
    "gram8-plural": {"questions": 1332, "known": 1260, "leave": 781, "keep": 322, "only_c": 562},
    "gram9-plural-verbs": {"questions": 870, "known": 812, "leave": 327, "keep": 154, "only_c": 57},
}
# Without leaving anything out, the answer is b in 1,064 questions and c in 7,089; never a.
NEAREST_IS = {"b": 1064, "c": 7089, "a": 0}
# The same for the course's Dutch questions in fastText's 28,209 words: every question has its four words there.
GROUPS_NL = {
    "capital-common-countries": {"questions": 552, "leave": 432, "keep": 174, "only_c": 138},
    "city-in-state": {"questions": 42, "leave": 42, "keep": 13, "only_c": 0},
    "family": {"questions": 182, "leave": 157, "keep": 29, "only_c": 92},
    "gram2-opposite": {"questions": 210, "leave": 45, "keep": 2, "only_c": 42},
    "gram3-comparative": {"questions": 342, "leave": 247, "keep": 77, "only_c": 58},
    "gram4-superlative": {"questions": 240, "leave": 109, "keep": 47, "only_c": 0},
    "gram6-nationality-adjective": {"questions": 702, "leave": 531, "keep": 272, "only_c": 0},
    "gram7-past-tense": {"questions": 552, "leave": 341, "keep": 133, "only_c": 0},
    "gram8-plural": {"questions": 552, "leave": 416, "keep": 20, "only_c": 253},
    "gram9-plural-verbs": {"questions": 210, "leave": 197, "keep": 99, "only_c": 43},
}
NEAREST_IS_NL = {"b": 879, "c": 1735, "a": 0}


def total(groups: dict, key: str) -> int:
    return sum(g[key] for g in groups.values())


GROUP_NAMES = {
    "capital-common-countries": {"en": "capital and country", "nl": "hoofdstad en land"},
    "capital-world": {"en": "capital and country, all over the world", "nl": "hoofdstad en land, van de hele wereld"},
    "currency": {"en": "country and currency", "nl": "land en munt"},
    "city-in-state": {"en": "city and American state", "nl": "stad en Amerikaanse staat"},
    "family": {"en": "male and female", "nl": "mannelijk en vrouwelijk"},
    "gram1-adjective-to-adverb": {"en": "adjective and adverb", "nl": "bijvoeglijk naamwoord en bijwoord"},
    "gram2-opposite": {"en": "word and its opposite", "nl": "woord en tegendeel"},
    "gram3-comparative": {"en": "comparative", "nl": "vergrotende trap"},
    "gram4-superlative": {"en": "superlative", "nl": "overtreffende trap"},
    "gram5-present-participle": {"en": "verb and its -ing form", "nl": "werkwoord en -ing-vorm"},
    "gram6-nationality-adjective": {"en": "country and nationality", "nl": "land en nationaliteit"},
    "gram7-past-tense": {"en": "-ing form and past tense", "nl": "-ing-vorm en verleden tijd"},
    "gram8-plural": {"en": "plural of a noun", "nl": "meervoud van een zelfstandig naamwoord"},
    "gram9-plural-verbs": {"en": "verb with -s", "nl": "werkwoord met -s"},
}
GROUP_NAMES_NL = {   # the names of the course's Dutch groups, whose pairs go another way at times
    "capital-common-countries": {"en": "country and capital", "nl": "land en hoofdstad"},
    "city-in-state": {"en": "city and province", "nl": "stad en provincie"},
    "family": {"en": "male and female", "nl": "mannelijk en vrouwelijk"},
    "gram2-opposite": {"en": "word and its opposite", "nl": "woord en tegendeel"},
    "gram3-comparative": {"en": "comparative", "nl": "vergrotende trap"},
    "gram4-superlative": {"en": "superlative", "nl": "overtreffende trap"},
    "gram6-nationality-adjective": {"en": "country and nationality", "nl": "land en nationaliteit"},
    "gram7-past-tense": {"en": "verb and past tense", "nl": "werkwoord en verleden tijd"},
    "gram8-plural": {"en": "plural of a noun", "nl": "meervoud van een zelfstandig naamwoord"},
    "gram9-plural-verbs": {"en": "verb, singular and plural", "nl": "werkwoord, enkelvoud en meervoud"},
}


def first_pair(group: str) -> tuple:
    """The first pair of a group of word2vec: the first two words of its first question."""
    a, b, _, _ = wordvectors.questions()[group][0]
    return a, b


# the questions of the picture under Exercise 2, each with the word it means
SHOWN = [("man", "king", "woman", "queen"), ("france", "paris", "italy", "rome"), ("apple", "tree", "grape", "vine"),
         ("good", "better", "bad", "worse"), ("big", "bigger", "small", "smaller"),
         ("netherlands", "amsterdam", "germany", "berlin")]
SHOWN_NL = [("man", "koning", "vrouw", "koningin"), ("Frankrijk", "Parijs", "Italië", "Rome"),
            ("goed", "beter", "slecht", "slechter"), ("groot", "groter", "klein", "kleiner"),
            ("Nederland", "Amsterdam", "Duitsland", "Berlijn"), ("lopen", "liep", "zwemmen", "zwom")]
NEAR_WORDS = ["frog", "coffee", "paris", "said", "computer"]
NEAR_WORDS_NL = ["kikker", "koffie", "Parijs", "zei", "computer"]
IDEA_WORDS = ["frog", "toad", "car"]


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
_CHIP_FAINT = _CHIP + ";color:var(--ea-ink-3);text-decoration:line-through"
_BAR = "display:inline-block;height:0.55rem;border-radius:2px;vertical-align:middle"


def _marked(text: str) -> str:
    parts = html.escape(text).split("*")
    return "".join(f"<i>{p}</i>" if k % 2 else p for k, p in enumerate(parts))


def _note(text: str) -> str:
    """A line under a picture; a word between asterisks is set in italics, as in the lesson's text."""
    return f'<div style="{_UI}">{_marked(text)}</div>'


def _label(text: str) -> str:
    return f'<div style="{_LABEL}">{_marked(text)}</div>'


def _bare(text: str) -> str:
    return text.replace("*", "")


def _num(n, lang: str) -> str:
    """A whole number with a separator for thousands: 4,518 in English, 4.518 in Dutch."""
    text = f"{round(n):,}"
    return text.replace(",", ".") if lang == "nl" else text


def _dec(x: float, lang: str, digits: int = 3) -> str:
    """A number with `digits` decimals: 0.874 in English, 0,874 in Dutch; a minus as a real minus sign."""
    text = f"{x:.{digits}f}".replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _pct(part: int, whole: int, lang: str) -> str:
    """A share as a percentage with one decimal: 47.9% in English, 47,9% in Dutch."""
    return _dec(100 * part / whole if whole else 0.0, lang, 1) + "%"


def _chip(text: str, style: str = _CHIP_DATA) -> str:
    return f'<span style="{style}">{html.escape(text)}</span>'


def _table(rows_: list, head: list, numbers: tuple = ()) -> str:
    """A small table; rows hold HTML that is already escaped; numbers are the columns set to the right."""
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers else ""}">{c}</td>'
                                    for j, c in enumerate(row)) + "</tr>" for row in rows_)
    head_html = f"<thead><tr>{top}</tr></thead>" if head else ""
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem">'
            f"{head_html}<tbody>{body}</tbody></table></div>")


def _short(value, limit: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit - 1] + "…"


# --- the idea's pictures: the numbers of three words, and the parallelogram ------------------------------------------------

_IDEA_WORDS = {
    "en": {"title": "The 50 numbers of three words in GloVe",
           "pairs": "Cosine similarity of two of them",
           "pair_head": ["Words", "Cosine similarity"],
           "legend": "Orange: the numbers GloVe learned for a word, one bar per number, above the line when the number "
                     "is positive and below it when it is negative. Most bars of *frog* and *toad* point the same way; "
                     "those of *car* far less often.",
           "alt": "Fifty bars for each of frog, toad and car."},
    "nl": {"title": "De 50 getallen van drie woorden in GloVe",
           "pairs": "Cosine similarity van twee ervan",
           "pair_head": ["Woorden", "Cosine similarity"],
           "legend": "Oranje: de getallen die GloVe voor een woord heeft geleerd, één staafje per getal, boven de lijn "
                     "als het getal positief is en eronder als het negatief is. De meeste staafjes van *frog* (kikker) "
                     "en *toad* (pad) wijzen dezelfde kant op; de staafjes van *car* (auto) veel minder vaak.",
           "alt": "Vijftig staafjes voor elk van frog, toad en car."},
}


def _strip(v: np.ndarray, most: float) -> str:
    """The numbers of one vector as a row of bars around a line, in SVG."""
    w, gap, half = 6, 1, 18
    bars = []
    for k, x in enumerate(v):
        h = abs(float(x)) / most * half
        y = half - h if x >= 0 else half
        bars.append(f'<rect x="{k * w}" y="{y:.1f}" width="{w - gap}" height="{max(h, 0.4):.1f}" '
                    f'style="fill:var(--ea-param)"/>')
    width = len(v) * w
    line = f'<line x1="0" y1="{half}" x2="{width}" y2="{half}" style="stroke:var(--ea-line)"/>'
    return (f'<svg viewBox="0 0 {width} {2 * half}" width="{width}" height="{2 * half}" role="img" '
            f'aria-hidden="true" style="max-width:100%;display:block">{line}{"".join(bars)}</svg>')


def numbers_picture(lang: str) -> str:
    """The 50 numbers of frog, toad and car as bars, and the cosine similarities of the three pairs."""
    w = _IDEA_WORDS[lang]
    words, E, index = _glove()
    vs = [E[index[x]] for x in IDEA_WORDS]
    most = max(float(np.abs(v).max()) for v in vs)
    strips = "".join(f'<div style="display:flex;align-items:center;gap:0.6rem;margin:0.2rem 0">'
                     f'<span style="width:3.2rem;font-family:var(--ea-font-code);font-size:0.8rem;color:var(--ea-ink)">'
                     f'{html.escape(x)}</span>{_strip(v, most)}</div>' for x, v in zip(IDEA_WORDS, vs))
    pairs = [(0, 1), (0, 2), (1, 2)]
    cos = {p: float(cosines(np.stack([vs[p[0]]]), vs[p[1]])[0]) for p in pairs}
    table = [[f"<i>{IDEA_WORDS[i]}</i>, <i>{IDEA_WORDS[j]}</i>", _dec(cos[(i, j)], lang)] for i, j in pairs]
    body = strips + _label(w["pairs"]) + _table(table, w["pair_head"], numbers=(1,)) + _note(w["legend"])
    alt = (w["alt"] + " " + "; ".join(f"{IDEA_WORDS[i]} {IDEA_WORDS[j]} {_dec(cos[(i, j)], lang)}" for i, j in pairs)
           + ". " + _bare(w["legend"]))
    return yours.figure(w["title"], body, alt=alt)


_PARALLEL_WORDS = {
    "en": {"title": "The parallelogram, drawn in two numbers",
           "legend": "A sketch with made-up positions: real word embeddings have 50 numbers or more. Gray: four words, "
                     "and the step from *man* to *woman*, which is added to *king*; the step from *man* to *king*, added to "
                     "*woman*, lands on the same point. Blue: that point, king − man + woman. The word nearest to that "
                     "point is the answer.",
           "point": "king − man + woman"},
    "nl": {"title": "Het parallellogram, getekend in twee getallen",
           "legend": "Een schets met verzonnen plaatsen: echte word embeddings hebben 50 getallen of meer. Grijs: vier "
                     "woorden, en de stap van *man* naar *woman*, die bij *king* wordt opgeteld; de stap van *man* naar "
                     "*king*, opgeteld bij *woman*, komt op hetzelfde punt uit. Blauw: dat punt, king − man + woman. Het "
                     "woord dat het dichtst bij dat punt ligt, is het antwoord.",
           "point": "king − man + woman"},
}


def parallelogram_picture(lang: str) -> str:
    """A sketch of the parallelogram: man, woman, king and queen, the step from man to woman added to king, and the
    point king − man + woman next to queen."""
    w = _PARALLEL_WORDS[lang]
    man, woman, king, queen = (70, 150), (70, 55), (260, 165), (248, 76)
    point = (king[0] - man[0] + woman[0], king[1] - man[1] + woman[1])
    gray, blue = "var(--ea-ink-3)", "var(--ea-pred)"
    defs = ('<defs><marker id="we-arrow-gray" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ea-ink-3)"/></marker></defs>')

    def arrow(p, q, dash=""):
        x1, y1, x2, y2 = p[0], p[1], q[0], q[1]
        dx, dy = x2 - x1, y2 - y1
        length = (dx * dx + dy * dy) ** 0.5
        x2, y2 = x2 - dx / length * 7, y2 - dy / length * 7   # stop just before the point
        return (f'<line x1="{x1}" y1="{y1}" x2="{x2:.1f}" y2="{y2:.1f}" style="stroke:{gray};stroke-width:1.5'
                f'{";stroke-dasharray:" + dash if dash else ""}" marker-end="url(#we-arrow-gray)"/>')

    def dot(p, name, color, dx=8, dy=4, anchor="start"):
        return (f'<circle cx="{p[0]}" cy="{p[1]}" r="4" style="fill:{color}"/>'
                f'<text x="{p[0] + dx}" y="{p[1] + dy}" text-anchor="{anchor}" style="font-family:var(--ea-font-code);'
                f'font-size:12px;fill:{"var(--ea-pred-text)" if color == blue else "var(--ea-ink)"}">'
                f'{html.escape(name)}</text>')
    lines = (arrow(man, woman) + arrow(king, point) +
             f'<line x1="{man[0]}" y1="{man[1]}" x2="{king[0]}" y2="{king[1]}" style="stroke:var(--ea-line);'
             f'stroke-dasharray:3 3"/><line x1="{woman[0]}" y1="{woman[1]}" x2="{point[0]}" y2="{point[1]}" '
             f'style="stroke:var(--ea-line);stroke-dasharray:3 3"/>')
    dots = (dot(man, "man", gray, -8, 4, "end") + dot(woman, "woman", gray, -8, 4, "end") + dot(king, "king", gray)
            + dot(queen, "queen", gray, -8, 16, "end") + dot(point, w["point"], blue, 8, -6))
    svg = (f'<svg viewBox="0 0 420 190" width="420" height="190" role="img" aria-hidden="true" '
           f'style="max-width:100%;display:block">{defs}{lines}{dots}</svg>')
    alt = (f'{w["title"]}. man and woman on the left, woman above man; king and queen on the right, queen above king. '
           f'An arrow from man to woman, and the same arrow from king, ending next to queen. ' + _bare(w["legend"]))
    return yours.figure(w["title"], svg + _note(w["legend"]), alt=alt)


# --- the learner's function in a picture ---------------------------------------------------------------------------------


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error
PICTURE_SECONDS = 10  # what the learner's function gets per call in a picture


def _call(fn, *args, seconds: float = PICTURE_SECONDS):
    """The learner's function within a time limit; _STOPPED when it stops with an error, and the Timeout passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
            return with_time_limit(lambda: fn(*(copy.deepcopy(a) for a in args)), seconds)
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


# --- the picture under exercise 1: the nearest neighbors by your cosines ---------------------------------------------------

_COS_WORDS = {
    "en": {"title": "The nearest neighbors of five words, by your cosines",
           "head": ["Word", "Its 6 nearest neighbors by your function"],
           "legend": "Blue: a neighbor that is also among the course's six nearest; purple: one that is not.",
           "fine": "Your function finds the same six neighbors as the course for all five words, among the 50,000 rows "
                   "of GloVe.",
           "differ": "For {n} of the five words, your six nearest neighbors differ from the course's.",
           "above": "Some of your numbers are above 1, and a cosine similarity lies between −1 and 1.",
           "distance": "Your numbers look like distances: 0 for the word itself. The largest distance belongs to the "
                       "farthest word, so your list starts with words far away.",
           "error": "Your function stops with an error on the rows of GloVe, with the vector of *{word}*.",
           "slow": "Your function takes more than {s} seconds for the 50,000 rows of GloVe.",
           "odd": "With the vector of *{word}*, your function gives back {got}, and not 50,000 numbers, one per row.",
           "small": "With a table of 3 rows, your function gives back {got}, and not 3 numbers, one per row, so the "
                    "picture does not try it on the 50,000 rows of GloVe."},
    "nl": {"title": "De dichtstbijzijnde buren van vijf woorden, volgens jouw cosines",
           "head": ["Woord", "De 6 dichtstbijzijnde buren volgens je functie"],
           "legend": "Blauw: een buur die ook bij de zes dichtstbijzijnde van de course hoort; paars: een buur die daar "
                     "niet bij hoort.",
           "fine": "Je functie vindt voor alle vijf de woorden dezelfde zes buren als de course, tussen de 50.000 rijen "
                   "van GloVe.",
           "differ": "Voor {n} van de vijf woorden verschillen je zes dichtstbijzijnde buren van de buren van de course.",
           "above": "Een paar van je getallen zijn groter dan 1, en een cosine similarity ligt tussen −1 en 1.",
           "distance": "Je getallen lijken op afstanden: 0 voor het woord zelf. De grootste afstand hoort bij het "
                       "verste woord, dus je lijst begint met woorden die ver weg liggen.",
           "error": "Je functie stopt met een fout op de rijen van GloVe, met de vector van *{word}*.",
           "slow": "Je functie doet er langer dan {s} seconden over voor de 50.000 rijen van GloVe.",
           "odd": "Met de vector van *{word}* geeft je functie {got} terug, en geen 50.000 getallen, één per rij.",
           "small": "Met een tabel van 3 rijen geeft je functie {got} terug, en geen 3 getallen, één per rij, dus de "
                    "afbeelding probeert je functie niet op de 50.000 rijen van GloVe."},
}


def _as_numbers(got, n: int):
    """The learner's outcome as an array of n floats, or None."""
    try:
        arr = np.asarray(got, dtype=float)
    except (TypeError, ValueError):
        return None
    return arr if arr.shape == (n,) else None


_SMALL = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])


def cosines_picture(seen) -> str:
    """The learner's cosines on GloVe: the six nearest neighbors of five words, each blue when it is also among the
    course's six, purple when not."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _COS_WORDS[lang]
    words, E, index = _glove()
    table, alts, notes, end, right = [], [], [], "", 0
    # a table of 3 rows first: a function that gives a table back would make one of 50,000 by 50,000 numbers
    try:
        small = _call(fn, _SMALL, np.array([1.0, 0.0]))
    except Timeout:
        small = None
    if small is not None and small is not _STOPPED and _as_numbers(small, len(_SMALL)) is None:
        return yours.figure(w["title"], _note(w["small"].format(got=_short(small))),
                            alt=f'{w["title"]}. ' + _bare(w["small"].format(got=_short(small))))
    for word in NEAR_WORDS:
        try:
            got = _call(fn, E, E[index[word]])
        except Timeout:
            end = w["slow"].format(s=PICTURE_SECONDS)
            break
        if got is _STOPPED:
            end = w["error"].format(word=word)
            break
        arr = _as_numbers(got, len(words))
        if arr is None:
            end = w["odd"].format(word=word, got=_short(got))
            break
        if arr[index[word]] < 1e-6 and np.nanmin(arr) >= 0 and "distance" not in notes:
            notes.append("distance")
        sims = np.nan_to_num(arr.copy(), nan=-np.inf)
        sims[index[word]] = -np.inf
        mine = [(words[i], float(sims[i])) for i in np.argsort(-sims, kind="stable")[:NEAR]]
        if any(s > 1 + 1e-4 for _, s in mine) and "above" not in notes:
            notes.append("above")
        course = {x for x, _ in neighbors(word)}
        right += {x for x, _ in mine} == course
        table.append([f"<i>{html.escape(word)}</i>", "".join(_chip(f"{x} {_dec(s, lang)}", _CHIP_PRED if x in course
                                                                     else _CHIP_PURPLE) for x, s in mine)])
        alts.append(f"{word}: " + ", ".join(f"{x} {_dec(s, lang)}" for x, s in mine))
    body = _table(table, w["head"]) if table else ""
    texts = [end] if end else []
    texts += [w[k] for k in notes]
    if not end and right == len(NEAR_WORDS):
        texts.append(w["fine"])
    elif table and right < len(table):
        texts.append(w["differ"].format(n=len(table) - right))
    if table:
        texts.append(w["legend"])
    text = " ".join(texts)
    return yours.figure(w["title"], body + _note(text), alt=f'{w["title"]}. ' + "; ".join(alts) + ". " + _bare(text))


# --- the picture under exercise 2: your analogies ---------------------------------------------------------------------------

_AN_WORDS = {
    "en": {"title": "Your analogy on six questions",
           "head": ["Question", "Meant", "Your answer", "Nearest to b − a + c, with nothing left out"],
           "legend": "Blue: your answer, the same as the course's; purple: an answer that differs from the course's. In "
                     "the last column, struck through: a word of the question itself, which the course leaves out; the "
                     "first word that is not struck through is the course's answer. Meant: the word the question has in "
                     "mind.",
           "own": "In {n} of the six questions, your answer is a word of the question itself.",
           "number": "Your function gives back {got}, a number and not a word.",
           "error": "Your function stops with an error on the question *{q}*.",
           "slow": "Your function takes more than {s} seconds on the question *{q}*.",
           "meant": "The course's answer is the word meant in {n} of the six questions."},
    "nl": {"title": "Jouw analogy op zes vragen",
           "head": ["Vraag", "Bedoeld", "Jouw antwoord", "Het dichtst bij b − a + c, zonder iets weg te laten"],
           "legend": "Blauw: je antwoord, hetzelfde als dat van de course; paars: een antwoord dat anders is dan dat van "
                     "de course. In de laatste kolom doorgestreept: een woord van de vraag zelf, dat de course weglaat; "
                     "het eerste woord dat niet doorgestreept is, is het antwoord van de course. Bedoeld: het woord dat "
                     "de vraag voor ogen heeft.",
           "own": "In {n} van de zes vragen is je antwoord een woord van de vraag zelf.",
           "number": "Je functie geeft {got} terug, een getal en geen woord.",
           "error": "Je functie stopt met een fout op de vraag *{q}*.",
           "slow": "Je functie doet er langer dan {s} seconden over op de vraag *{q}*.",
           "meant": "Het antwoord van de course is in {n} van de zes vragen het bedoelde woord."},
}


def _question(a, b, c) -> str:
    return f"{a} : {b} = {c} : ?"


def _nearest_chips(found: list, question: tuple, lang: str) -> str:
    return "".join(_chip(f"{x} {_dec(s, lang)}", _CHIP_FAINT if x in question else _CHIP_DATA) for x, s in found)


def analogy_picture(seen) -> str:
    """The learner's analogy on six questions of GloVe, next to the word each question means and the words nearest to
    b − a + c."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _AN_WORDS[lang]
    words, E, index = _glove()
    table, alts, end, own, meant = [], [], "", 0, 0
    for a, b, c, d in SHOWN:
        q = _question(a, b, c)
        try:
            got = _call(fn, E, list(words), dict(index), a, b, c)
        except Timeout:
            end = w["slow"].format(s=PICTURE_SECONDS, q=q)
            break
        if got is _STOPPED:
            end = w["error"].format(q=q)
            break
        if isinstance(got, (int, np.integer)) and not isinstance(got, bool):
            end = w["number"].format(got=_short(got))
            break
        course = analogy(E, words, index, a, b, c)
        meant += course == d
        own += got in (a, b, c)
        chip = _chip(str(got), _CHIP_PRED if got == course else _CHIP_PURPLE)
        found = solve(a, b, c, k=5, leave=False)
        table.append([html.escape(q), f"<i>{html.escape(d)}</i>", chip, _nearest_chips(found, (a, b, c), lang)])
        alts.append(f"{q} {d}: {got}; " + ", ".join(x for x, _ in found))
    body = _table(table, w["head"]) if table else ""
    texts = [end] if end else []
    if own:
        texts.append(w["own"].format(n=own))
    if table and not end:
        texts.append(w["meant"].format(n=meant))
    if table:
        texts.append(w["legend"])
    text = " ".join(texts)
    return yours.figure(w["title"], body + _note(text), alt=f'{w["title"]}. ' + "; ".join(alts) + ". " + _bare(text))


# --- the picture after the sort: every group of word2vec's questions --------------------------------------------------------

_GROUPS_WORDS = {
    "en": {"title": "How often GloVe answers each group of word2vec's questions",
           "head": ["Group", "Example", "Right, of the questions GloVe knows", ""],
           "all": "All {n} questions whose four words GloVe knows",
           "legend": "Blue bar: right with the three words of the question left out, as the course's analogy does. "
                     "Outlined bar: right when nothing is left out. Gray bar: right by taking the nearest neighbor of c "
                     "alone, leaving out the three words; a and b are used only to leave them out. GloVe here: 50 numbers per "
                     "word, 50,000 words; {left} of the {q} questions have a word that is not among them.",
           "alt_row": "{name}: {leave} right with the words left out, {keep} with nothing left out, {only} by c alone"},
    "nl": {"title": "Hoe vaak GloVe elke groep vragen van word2vec goed heeft",
           "head": ["Groep", "Voorbeeld", "Goed, van de vragen die GloVe kent", ""],
           "all": "Alle {n} vragen waarvan GloVe de vier woorden kent",
           "legend": "Blauwe balk: goed als de drie woorden van de vraag zijn weggelaten, zoals de analogy van de course "
                     "doet. Omlijnde balk: goed als er niets is weggelaten. Grijze balk: goed door de dichtstbijzijnde "
                     "buur van c te nemen, met de drie woorden weggelaten; a en b dienen alleen om die woorden weg te laten. GloVe "
                     "hier: 50 getallen per woord, 50.000 woorden; {left} van de {q} vragen hebben een woord dat daar "
                     "niet bij zit.",
           "alt_row": "{name}: {leave} goed met de woorden weggelaten, {keep} zonder iets weg te laten, {only} met c "
                      "alleen"},
}


def _three_bars(leave: float, keep: float, only: float) -> str:
    width = 9.0
    return (f'<div style="line-height:0.7rem">'
            f'<span style="{_BAR};width:{max(0.1, width * leave):.2f}rem;background:var(--ea-pred)"></span><br>'
            f'<span style="{_BAR};width:{max(0.1, width * keep):.2f}rem;border:1.5px solid var(--ea-pred);'
            f'box-sizing:border-box"></span><br>'
            f'<span style="{_BAR};width:{max(0.1, width * only):.2f}rem;background:var(--ea-data)"></span></div>')


def _groups_table(groups: dict, names: dict, example, lang: str, key: str = "known") -> tuple:
    rows_, alts = [], []
    order = sorted(groups, key=lambda g: -groups[g]["leave"] / groups[g][key])
    for g in order:
        r = groups[g]
        n = r[key]
        leave, keep, only = r["leave"] / n, r["keep"] / n, r["only_c"] / n
        a, b = example(g)
        name = names[g][lang]
        rows_.append([html.escape(name), f"<i>{html.escape(a)}</i> : <i>{html.escape(b)}</i>",
                      _pct(r["leave"], n, lang), _three_bars(leave, keep, only)])
        alts.append(_GROUPS_WORDS[lang]["alt_row"].format(name=name, leave=_pct(r["leave"], n, lang),
                                                          keep=_pct(r["keep"], n, lang),
                                                          only=_pct(r["only_c"], n, lang)))
    return rows_, alts


def groups_picture(lang: str) -> str:
    """For each of word2vec's 14 groups: how often GloVe answers right with the question's words left out, with
    nothing left out, and with the nearest neighbor of c alone."""
    w = _GROUPS_WORDS[lang]
    rows_, alts = _groups_table(GROUPS, GROUP_NAMES, first_pair, lang)
    n, q = total(GROUPS, "known"), total(GROUPS, "questions")
    head_row = [f"<b>{html.escape(w['all'].format(n=_num(n, lang)))}</b>", "",
                f"<b>{_pct(total(GROUPS, 'leave'), n, lang)}</b>",
                _three_bars(total(GROUPS, "leave") / n, total(GROUPS, "keep") / n, total(GROUPS, "only_c") / n)]
    legend = w["legend"].format(left=_num(q - n, lang), q=_num(q, lang))
    body = _table([head_row] + rows_, w["head"], numbers=(2,)) + _note(legend)
    alt = (f'{w["title"]}. ' + w["all"].format(n=_num(n, lang)) + ": " + _pct(total(GROUPS, "leave"), n, lang) + ". "
           + "; ".join(alts) + ". " + _bare(legend))
    return yours.figure(w["title"], body, alt=alt)


# --- the Dutch part: fastText -------------------------------------------------------------------------------------------------

_NL_WORDS = {
    "en": {"title": "Dutch word embeddings of fastText",
           "near_head": ["Word", "Its 6 nearest neighbors"],
           "an_head": ["Question", "Meant", "Answer, leaving out the question's words", "Nearest, with nothing left out"],
           "groups_head": ["Group", "Example", "Right", ""],
           "near": "Nearest neighbors",
           "analogies": "Analogies",
           "groups": "The course's {n} Dutch questions, by group",
           "all": "All {n} questions",
           "legend": "Gray: words and their cosine similarity; struck through: a word of the question itself. Blue: an "
                     "answer that is the word meant; purple: one that is not. In the bars, blue: right with the "
                     "question's words left out; outlined: right with nothing left out; gray: right by the nearest "
                     "neighbor of c alone.",
           "missing": "The Dutch vectors are not on your computer yet: the button above downloads them."},
    "nl": {"title": "Nederlandse word embeddings van fastText",
           "near_head": ["Woord", "De 6 dichtstbijzijnde buren"],
           "an_head": ["Vraag", "Bedoeld", "Antwoord, met de woorden van de vraag weggelaten",
                       "Het dichtst bij, zonder iets weg te laten"],
           "groups_head": ["Groep", "Voorbeeld", "Goed", ""],
           "near": "Dichtstbijzijnde buren",
           "analogies": "Analogieën",
           "groups": "De {n} Nederlandse vragen van de course, per groep",
           "all": "Alle {n} vragen",
           "legend": "Grijs: woorden met de cosine similarity; doorgestreept: een woord van de vraag zelf. Blauw: een "
                     "antwoord dat het bedoelde woord is; paars: een antwoord dat dat niet is. In de balken, blauw: goed "
                     "met de woorden van de vraag weggelaten; omlijnd: goed zonder iets weg te laten; grijs: goed met de "
                     "dichtstbijzijnde buur van c alleen.",
           "missing": "De Nederlandse vectoren staan nog niet op je computer: de knop hierboven downloadt de vectoren."},
}


def _first_pair_nl(group: str) -> tuple:
    return wordvectors.pairs_nl()[group][0]


def dutch_picture(lang: str) -> str:
    """fastText's Dutch vectors: the nearest neighbors of five words, six analogies, and how often the course's Dutch
    questions are answered right per group."""
    w = _NL_WORDS[lang]
    if not downloaded():
        return yours.figure(w["title"], _note(w["missing"]), alt=w["missing"])
    words, E, index = _fasttext()
    near = []
    for x in NEAR_WORDS_NL:
        found = kept(f"{_KEY}:nl-near:{x}", lambda x=x: nearest(E, words, E[index[x]], NEAR, leave=(x,)))
        near.append([f"<i>{html.escape(x)}</i>", "".join(_chip(f"{y} {_dec(s, lang)}") for y, s in found)])
    ans = []
    for a, b, c, d in SHOWN_NL:
        left = kept(f"{_KEY}:nl-an:{a}:{b}:{c}", lambda a=a, b=b, c=c: solve(a, b, c, 1, True, "fasttext"))[0][0]
        found = kept(f"{_KEY}:nl-an-keep:{a}:{b}:{c}", lambda a=a, b=b, c=c: solve(a, b, c, 3, False, "fasttext"))
        ans.append([html.escape(_question(a, b, c)), f"<i>{html.escape(d)}</i>",
                    _chip(left, _CHIP_PRED if left == d else _CHIP_PURPLE), _nearest_chips(found, (a, b, c), lang)])
    rows_, alts = _groups_table(GROUPS_NL, GROUP_NAMES_NL, _first_pair_nl, lang, key="questions")
    n = total(GROUPS_NL, "questions")
    head_row = [f"<b>{html.escape(w['all'].format(n=_num(n, lang)))}</b>", "",
                f"<b>{_pct(total(GROUPS_NL, 'leave'), n, lang)}</b>",
                _three_bars(total(GROUPS_NL, "leave") / n, total(GROUPS_NL, "keep") / n, total(GROUPS_NL, "only_c") / n)]
    body = (_label(w["near"]) + _table(near, w["near_head"]) + _label(w["analogies"]) + _table(ans, w["an_head"])
            + _label(w["groups"].format(n=_num(n, lang))) + _table([head_row] + rows_, w["groups_head"], numbers=(2,))
            + _note(w["legend"]))
    alt = (f'{w["title"]}. ' + "; ".join(f"{r[0]}" for r in near) + ". " + "; ".join(alts) + ". " + _bare(w["legend"]))
    alt = alt.replace("<i>", "").replace("</i>", "")
    return yours.figure(w["title"], body, alt=alt)


# --- the explore: any word, or any three ------------------------------------------------------------------------------------

LONGEST = 80   # characters of the explore's box that count

_EXPLORE_WORDS = {
    "en": {"title": "{what} in {name}",
           "near_what": "The neighbors of {word}",
           "an_what": "{q}",
           "glove": "GloVe (English)", "fasttext": "fastText (Dutch)",
           "empty": "Type one word for its neighbors, or three words, a b c, for the analogy a : b = c : ?.",
           "count": "Type one word, or three; you typed {n}.",
           "unknown": "*{word}* is not among the {n} words of {name}.",
           "unknown_hint": {"glove": " GloVe's words are in small letters, of the letters a to z only.",
                            "fasttext": " fastText's words keep their capitals: try *Parijs*, not *parijs*."},
           "missing": "The Dutch vectors are not on your computer yet: the button in the part about Dutch downloads "
                      "them.",
           "near": "The {n} nearest neighbors, with their cosine similarity:",
           "left": "The nearest words to b − a + c, leaving out a, b and c:",
           "kept": "With nothing left out:",
           "legend_near": "Blue: the words nearest to the word you typed, by cosine similarity.",
           "legend": "Blue: the words nearest to what you typed, by cosine similarity; struck through: a word of the "
                     "question itself."},
    "nl": {"title": "{what} in {name}",
           "near_what": "De buren van {word}",
           "an_what": "{q}",
           "glove": "GloVe (Engels)", "fasttext": "fastText (Nederlands)",
           "empty": "Typ één woord voor de buren, of drie woorden, a b c, voor de analogie a : b = c : ?.",
           "count": "Typ één woord, of drie; je typte er {n}.",
           "unknown": "*{word}* hoort niet bij de {n} woorden van {name}.",
           "unknown_hint": {"glove": " De woorden van GloVe zijn in kleine letters, en bestaan alleen uit de letters a tot en met z.",
                            "fasttext": " De woorden van fastText hebben hoofdletters: probeer *Parijs*, niet *parijs*."},
           "missing": "De Nederlandse vectoren staan nog niet op je computer: de knop in het deel over het Nederlands "
                      "downloadt de vectoren.",
           "near": "De {n} dichtstbijzijnde buren, met de cosine similarity:",
           "left": "De woorden die het dichtst bij b − a + c liggen, met a, b en c weggelaten:",
           "kept": "Zonder iets weg te laten:",
           "legend_near": "Blauw: de woorden die het dichtst bij het getypte woord liggen, volgens de cosine similarity.",
           "legend": "Blauw: de woorden die het dichtst bij wat je typte liggen, volgens de cosine similarity; "
                     "doorgestreept: een woord van de vraag zelf."},
}


def _bars_table(found: list, lang: str, faint=()) -> str:
    return _table([[_chip(x, _CHIP_FAINT if x in faint else _CHIP_PRED), _dec(s, lang),
                    f'<span style="{_BAR};width:{max(0.1, 7 * max(s, 0)):.2f}rem;background:var(--ea-pred)"></span>']
                   for x, s in found], [], numbers=(1,))


def explore_picture(text, kind, lang: str) -> str:
    """One word: its eight nearest neighbors. Three words a b c: the words nearest to b − a + c, leaving out the three
    and with nothing left out. In GloVe or, once downloaded, in fastText."""
    w = _EXPLORE_WORDS[lang]
    kind = "fasttext" if kind == "fasttext" else "glove"
    name = w[kind]
    typed = (text or "")[:LONGEST].replace(",", " ").replace(":", " ").split()
    if kind == "glove":
        typed = [x.lower() for x in typed]
    title_what = (w["near_what"].format(word=typed[0]) if len(typed) == 1
                  else w["an_what"].format(q=_question(*typed)) if len(typed) == 3 else "…")
    title = w["title"].format(what=title_what, name=name)
    if not typed:
        return yours.figure(title, _note(w["empty"]), alt=w["empty"])
    if len(typed) not in (1, 3):
        text_ = w["count"].format(n=len(typed)) + " " + w["empty"]
        return yours.figure(title, _note(text_), alt=_bare(text_))
    if kind == "fasttext" and not downloaded():
        return yours.figure(title, _note(w["missing"]), alt=w["missing"])
    words, E, index = _glove() if kind == "glove" else _fasttext()
    for x in typed:
        if x not in index:
            text_ = w["unknown"].format(word=x, n=_num(len(words), lang), name=name) + w["unknown_hint"][kind]
            return yours.figure(title, _note(text_), alt=_bare(text_))
    if len(typed) == 1:
        found = nearest(E, words, E[index[typed[0]]], 8, leave=(typed[0],))
        body = _label(w["near"].format(n=8)) + _bars_table(found, lang) + _note(w["legend_near"])
        alt = f"{title}. " + ", ".join(f"{x} {_dec(s, lang)}" for x, s in found)
        return yours.figure(title, body, alt=alt)
    a, b, c = typed
    left = solve(a, b, c, 5, True, kind)
    found = solve(a, b, c, 3, False, kind)
    body = (_label(w["left"]) + _bars_table(left, lang) + _label(w["kept"]) + _bars_table(found, lang, (a, b, c))
            + _note(w["legend"]))
    alt = (f"{title}. " + ", ".join(f"{x} {_dec(s, lang)}" for x, s in left) + ". " + _bare(w["kept"]) + " "
           + ", ".join(f"{x} {_dec(s, lang)}" for x, s in found))
    return yours.figure(title, body, alt=alt)


# --- the exercises -------------------------------------------------------------------------------------------------------

REQUIRED = ["cosines", "analogy", "groups"]


class Named:
    """A value the check gives the learner's function, shown in a message by its name. `fresh` makes the learner's
    copy: a list or dictionary of 50,000 words is copied in one step, and the read-only array of GloVe is passed as it
    is, as the lesson's cells have it."""

    def __init__(self, name: str, value, fresh=None):
        self.name, self.value, self.fresh = name, value, fresh

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return self.fresh(self.value) if self.fresh else copy.deepcopy(self.value, memo)


class _Table(np.ndarray):
    """A small table or vector that a check message writes the way the lesson's code does: np.array([...]). The
    learner's code gets a plain copy."""

    def __repr__(self):
        def row(r):
            return "[" + ", ".join(repr(float(v)) for v in r) + "]"
        a = np.asarray(self)
        inner = row(a) if a.ndim == 1 else "[" + ", ".join(row(r) for r in a) + "]"
        return f"np.array({inner})"

    def __deepcopy__(self, memo):
        return np.array(self)


def _t(values) -> np.ndarray:
    return np.array(values, dtype=float).view(_Table)


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


def _shown_numbers(v) -> str:
    """What a function of Exercise 1 gave, as a message shows it: numbers rounded to 3 decimals, and of a long array
    the first four and how many there are."""
    try:
        arr = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return _short(v)
    if arr.ndim == 0:
        return f"{float(arr):.3f}".rstrip("0").rstrip(".")
    def one(x):
        text = f"{round(float(x), 3) + 0.0:.3f}".rstrip("0").rstrip(".")
        return text if text not in ("", "-0") else "0"
    if arr.ndim == 2 and arr.size:   # a table: its first row, and its size
        first = arr[0][:4]
        return "[[" + ", ".join(one(x) for x in first) + (", …" if arr.shape[1] > 4 else "") + f"], …] ({arr.shape[0]:,} × {arr.shape[1]:,})"
    if arr.ndim != 1:
        return _short(v, 80)
    if len(arr) <= 6:
        return "[" + ", ".join(one(x) for x in arr) + "]"
    return "[" + ", ".join(one(x) for x in arr[:4]) + f", … ({len(arr):,} numbers)]"


def _close(got, want) -> bool:
    """The same numbers within a small margin, also for the float32 numbers of GloVe; a list counts as the array."""
    try:
        a, b = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    except (TypeError, ValueError):
        return same_value(got, want)
    if a.shape != b.shape:
        return False
    return bool(np.allclose(a, b, rtol=1e-4, atol=1e-5, equal_nan=False))


def _broadcast(e) -> bool:
    return isinstance(e, ValueError) and "broadcast" in str(e)


def _matmul_order(e) -> bool:
    return isinstance(e, ValueError) and ("mismatch" in str(e) or "not aligned" in str(e))


def _word_index(e) -> bool:
    return isinstance(e, IndexError) and "only integers" in str(e)


def _list_index(e) -> bool:
    return isinstance(e, TypeError) and "indices must be integers" in str(e)


def _a_number(got, text) -> bool:
    return isinstance(got, (int, np.integer)) and not isinstance(got, bool)


# exercise 1: cosines. The example of Embeddings: meaning as coordinates, and a vector v that is not 1 long; then two
# rows of 2 numbers, three rows of 3, one row, a v with norm 5, and two vectors of GloVe against its first 2,000 rows
# (not all 50,000: a function that gives a table back would make one of 50,000 by 50,000 numbers, 20 GB).
_GW, _GE, _GI = _glove()
E_GLOVE = Named("E", _GE, fresh=lambda v: v)
WORDS_GLOVE = Named("words", _GW, fresh=list)
INDEX_GLOVE = Named("index", _GI, fresh=dict)
E_FIRST = Named("E[:2000]", _GE[:2000], fresh=lambda v: v)
FROG = Named('E[index["frog"]]', _GE[_GI["frog"]], fresh=np.array)
KING_WOMAN = Named('E[index["king"]] - E[index["man"]] + E[index["woman"]]',
                   _GE[_GI["king"]] - _GE[_GI["man"]] + _GE[_GI["woman"]], fresh=np.array)

COS_CASES = [
    (_t([[1.0, 0.0], [10.0, 1.0], [0.5, 0.5], [-1.0, 0.0]]), _t([1.0, 0.0])),
    (_t([[3.0, 4.0], [0.0, 2.0], [-6.0, -8.0]]), _t([2.0, 0.0])),
    (_t([[1.0, 2.0], [3.0, 4.0]]), _t([1.0, 0.0])),
    (_t([[1.0, 2.0, 2.0], [2.0, 0.0, 0.0], [0.0, 0.0, -3.0]]), _t([0.0, 0.0, 1.0])),
    (_t([[2.0, 0.0]]), _t([5.0, 0.0])),
    (_t([[1.0, 0.0], [0.0, 1.0]]), _t([3.0, 4.0])),
    (E_FIRST, FROG),
    (E_FIRST, KING_WOMAN),
]

_COS_HEAD = "import numpy as np\n\n\ndef cosines(E, v):\n"
_COS_START = _COS_HEAD + "    ...\n"
_COS = _COS_HEAD + "    return E @ v / (np.linalg.norm(E, axis=1) * np.linalg.norm(v))\n"
_COS_DOT = _COS_HEAD + "    return E @ v\n"
_COS_NO_V = _COS_HEAD + "    return E @ v / np.linalg.norm(E, axis=1)\n"
_COS_ONLY_V = _COS_HEAD + "    return E @ v / np.linalg.norm(v)\n"
_COS_WHOLE = _COS_HEAD + "    return E @ v / (np.linalg.norm(E) * np.linalg.norm(v))\n"
_COS_TIMES = _COS_HEAD + "    return E @ v / np.linalg.norm(E, axis=1) * np.linalg.norm(v)\n"
_COS_DISTANCE = _COS_HEAD + "    return np.linalg.norm(E - v, axis=1)\n"
_COS_AXIS0 = _COS_HEAD + "    return E @ v / (np.linalg.norm(E, axis=0) * np.linalg.norm(v))\n"
_COS_ORDER = _COS_HEAD + "    return v @ E / (np.linalg.norm(E, axis=1) * np.linalg.norm(v))\n"
_COS_LOOP = (_COS_HEAD + "    found = []\n    for row in E:\n"
             "        found.append(np.dot(row, v) / (np.linalg.norm(row) * np.linalg.norm(v)))\n    return np.array(found)\n")

# exercise 2: analogy. Two questions of GloVe first; then three small tables of the course's own: one where the longest
# row wins on the dot product and the lowest on np.argmin, one where every other word has a cosine similarity below 0,
# and one where a lies nearest once b and c are left out;
# then six more questions of GloVe, where b, c or a word that is not meant lies nearest.
TOY_WORDS = ["man", "woman", "king", "queen", "giant", "apple"]
TOY = (_t([[1.0, 0.0], [1.0, 1.0], [3.0, 0.2], [3.0, 1.2], [10.0, 1.0], [0.0, -1.0]]), TOY_WORDS,
       {w: i for i, w in enumerate(TOY_WORDS)})
LOW_WORDS = ["a", "b", "c", "d", "e"]
LOW = (_t([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, -2.0], [-2.0, -0.5]]), LOW_WORDS,
       {w: i for i, w in enumerate(LOW_WORDS)})
NEAR_A_WORDS = ["p", "q", "r", "s"]
NEAR_A = (_t([[1.0, 0.0], [1.0, 0.2], [1.0, -0.1], [0.0, 1.0]]), NEAR_A_WORDS,
          {w: i for i, w in enumerate(NEAR_A_WORDS)})
_G = (E_GLOVE, WORDS_GLOVE, INDEX_GLOVE)
ANALOGY_CASES = [
    (*_G, "man", "king", "woman"),
    (*_G, "france", "paris", "italy"),
    (*TOY, "man", "king", "woman"),
    (*LOW, "a", "b", "c"),
    (*NEAR_A, "p", "q", "r"),
    (*_G, "apple", "tree", "grape"),
    (*_G, "good", "better", "bad"),
    (*_G, "big", "bigger", "small"),
    (*_G, "walk", "walked", "swim"),
    (*_G, "man", "doctor", "woman"),
    (*_G, "netherlands", "amsterdam", "germany"),
]

_AN_HEAD = "import numpy as np\n\n\ndef analogy(E, words, index, a, b, c):\n"
_AN_START = _AN_HEAD + "    ...\n"
_AN = (_AN_HEAD + "    v = E[index[b]] - E[index[a]] + E[index[c]]\n    sims = cosines(E, v)\n"
       "    for word in [a, b, c]:\n        sims[index[word]] = -2\n    return words[np.argmax(sims)]\n")
_AN_KEEPS = (_AN_HEAD + "    v = E[index[b]] - E[index[a]] + E[index[c]]\n    sims = cosines(E, v)\n"
             "    return words[np.argmax(sims)]\n")
_AN_KEEPS_B = _AN.replace("for word in [a, b, c]:", "for word in [a, c]:")
_AN_KEEPS_C = _AN.replace("for word in [a, b, c]:", "for word in [a, b]:")
_AN_KEEPS_A = _AN.replace("for word in [a, b, c]:", "for word in [b, c]:")
_AN_ZERO = _AN.replace("sims[index[word]] = -2", "sims[index[word]] = 0")
_AN_SIGNS = _AN.replace("v = E[index[b]] - E[index[a]] + E[index[c]]", "v = E[index[a]] - E[index[b]] + E[index[c]]")
_AN_MINUS_C = _AN.replace("v = E[index[b]] - E[index[a]] + E[index[c]]", "v = E[index[b]] - E[index[a]] - E[index[c]]")
_AN_NUMBER = _AN.replace("return words[np.argmax(sims)]", "return np.argmax(sims)")
_AN_ARGMIN = _AN.replace("np.argmax", "np.argmin")
_AN_DOT = _AN.replace("sims = cosines(E, v)", "sims = E @ v")
_AN_NO_INDEX = _AN.replace("v = E[index[b]] - E[index[a]] + E[index[c]]", "v = E[b] - E[a] + E[c]")
_AN_MAX = _AN.replace("return words[np.argmax(sims)]", "return words[np.max(sims)]")

EXERCISES = {
    "cosines": FunctionBox(
        name="cosines",
        picture=cosines_picture,
        function="cosines",
        start=_COS_START,
        solution=_COS,
        visible=COS_CASES[:2],
        hidden=COS_CASES[2:],
        compare=_close,
        show=_shown_numbers,
        first=[_DOTS],
        mistakes=[
            Mistake("axis", error=_broadcast,
                    en="With `{call}`, your function stops with an error: `E @ v` gives one number per row of `E`, and the norms you divide it by have another number of values. Each row needs its own norm. In *Sums and means along an axis*, `np.sum` added up along one axis or the other: which axis gives one result per row?",
                    nl="Met `{call}` stopt je functie met een fout: `E @ v` geeft één getal per rij van `E`, en de normen waardoor je deelt, hebben een ander aantal waarden. Elke rij heeft een eigen norm nodig. In *Sommen en gemiddelden langs een axis* telde `np.sum` op langs de ene of de andere axis: welke axis geeft één uitkomst per rij?"),
            Mistake("order", error=_matmul_order,
                    en="With `{call}`, your function stops with an error: in a product with `@`, the numbers of the vector have to match the columns of the table, and `E @ v` gives one number per row of `E`. In which order do the table and the vector stand in your product?",
                    nl="Met `{call}` stopt je functie met een fout: bij een product met `@` moeten de getallen van de vector passen bij de kolommen van de tabel, en `E @ v` geeft één getal per rij van `E`. In welke volgorde staan de tabel en de vector in je product?"),
            Mistake("table", matches=lambda got, text: np.ndim(got) == 2,
                    en="`{call}` gives {got}, a table, and it should give {want}, one number per row. Dividing by norms that keep an axis of their own spreads every dot product over a whole row. Which norms give one plain array, one number per row?",
                    nl="`{call}` geeft {got}, een tabel, en dat moet {want} zijn, één getal per rij. Delen door normen die een eigen axis houden, spreidt elk dot product over een hele rij. Welke normen geven één gewone array, één getal per rij?"),
            Mistake("dot", wrong=_COS_DOT,
                    en="`{call}` gives {got}, and it should give {want}: these are the dot products of the rows with `v`. A long row gets a large dot product, whatever its direction. By which two norms does a cosine similarity divide?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dit zijn de dot products van de rijen met `v`. Een lange rij krijgt een groot dot product, welke richting de rij ook heeft. Door welke twee normen deelt een cosine similarity?"),
            Mistake("no-v", wrong=_COS_NO_V,
                    en="`{call}` gives {got}, and it should give {want}: your function divides by the norm of each row, and some of its numbers come out above 1. Which norm is missing?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie deelt door de norm van elke rij, en een paar van de getallen komen boven 1 uit. Welke norm ontbreekt?"),
            Mistake("only-v", wrong=_COS_ONLY_V,
                    en="`{call}` gives {got}, and it should give {want}: your function divides by the norm of `v`, but each row has a length too. By what else does each dot product have to be divided?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie deelt door de norm van `v`, maar elke rij heeft ook een lengte. Waardoor moet elk dot product nog meer gedeeld worden?"),
            Mistake("whole", wrong=_COS_WHOLE,
                    en="`{call}` gives {got}, and it should give {want}: `np.linalg.norm(E)` without an axis gives one norm for the whole table, so every row is divided by the same number. How do you get one norm for each row?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `np.linalg.norm(E)` zonder axis geeft één norm voor de hele tabel, dus elke rij wordt door hetzelfde getal gedeeld. Hoe krijg je één norm per rij?"),
            Mistake("times", wrong=_COS_TIMES,
                    en="`{call}` gives {got}, and it should give {want}: Python divides and multiplies from left to right, so your function multiplies by the norm of `v` instead of dividing by it. Where do the parentheses go?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: Python deelt en vermenigvuldigt van links naar rechts, dus je functie vermenigvuldigt met de norm van `v` in plaats van erdoor te delen. Waar komen de haakjes?"),
            Mistake("distance", wrong=_COS_DISTANCE,
                    en="`{call}` gives {got}, and it should give {want}: these are the distances from `v` to each row, 0 for the same point and more the farther away. A cosine similarity looks at the direction instead: what does it compute from `E @ v`?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dit zijn de afstanden van `v` tot elke rij, 0 voor hetzelfde punt en meer naarmate een rij verder weg ligt. Een cosine similarity kijkt naar de richting: wat rekent een cosine similarity uit met `E @ v`?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back the cosine similarity of every row of `E` with `v`, as one array.",
                      "nl": "Vervang de `...`: geef de cosine similarity van elke rij van `E` met `v` terug, als één array."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Each number is the cosine similarity of one row of `E` with `v`: their dot product, divided by the norm of the row and by the norm of `v`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Elk getal is de cosine similarity van één rij van `E` met `v`: het dot product van de twee, gedeeld door de norm van de rij en door de norm van `v`."},
            "almost": {"en": "Right for the first two cases. But `{call}` gives {got}, and it should give {want}. Does your function work for other tables too, with three numbers per row or only one row?",
                       "nl": "Goed voor de eerste twee gevallen. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie ook voor andere tabellen, met drie getallen per rij of maar één rij?"},
            "correct": {"en": "Your `cosines` is right in all {n} cases, also for rows of GloVe. Look at the picture: the nearest neighbors of five words, found with your function.",
                        "nl": "Je `cosines` klopt in alle {n} gevallen, ook voor rijen van GloVe. Kijk naar de afbeelding: de dichtstbijzijnde buren van vijf woorden, gevonden met je functie."},
        },
    ),
    "analogy": FunctionBox(
        name="analogy",
        picture=analogy_picture,
        function="analogy",
        start=_AN_START,
        solution=_AN,
        names={"cosines": cosines},
        visible=ANALOGY_CASES[:2],
        hidden=ANALOGY_CASES[2:],
        first=[_DOTS],
        mistakes=[
            Mistake("no-index", error=_word_index,
                    en="With `{call}`, your function stops with an error: `E` is an array, numbered by row, so a word cannot pick a row of it. Which dictionary gives the number of the row of a word?",
                    nl="Met `{call}` stopt je functie met een fout: `E` is een array, genummerd per rij, dus met een woord kun je geen rij kiezen. Welke dictionary geeft het nummer van de rij van een woord?"),
            Mistake("max", error=_list_index,
                    en="With `{call}`, your function stops with an error: a list takes the number of a place, a whole number, and you gave it the highest cosine similarity itself. Which function of NumPy gives the place of the highest number?",
                    nl="Met `{call}` stopt je functie met een fout: een lijst neemt het nummer van een plaats, een heel getal, en je gaf de hoogste cosine similarity zelf. Welke functie van NumPy geeft de plaats van het hoogste getal?"),
            Mistake("number", matches=_a_number,
                    en="`{call}` gives {got}, and it should give {want}: `np.argmax` gives the number of a row, not a word. Which list turns that number into its word?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `np.argmax` geeft het nummer van een rij, geen woord. Welke lijst maakt van dat nummer het woord?"),
            Mistake("keeps", wrong=_AN_KEEPS,
                    en="`{call}` gives {got}, and it should give {want}: your answer is a word of the question itself. `b − a + c` often lies closest to `b` or to `c`, so the three words have to be left out. How do you change their numbers in `sims`, so that `np.argmax` passes over them?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je antwoord is een woord van de vraag zelf. `b − a + c` ligt vaak het dichtst bij `b` of bij `c`, dus de drie woorden moeten weggelaten worden. Hoe verander je de getallen van de drie woorden in `sims`, zodat `np.argmax` die woorden overslaat?"),
            Mistake("keeps-b", wrong=_AN_KEEPS_B,
                    en="`{call}` gives {got}, and it should give {want}: your function leaves out `a` and `c`, but not `b`, and `b` often lies nearest to `b − a + c`. Which words does your loop go over?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie laat `a` en `c` weg, maar `b` niet, en `b` ligt vaak het dichtst bij `b − a + c`. Langs welke woorden gaat je loop?"),
            Mistake("keeps-c", wrong=_AN_KEEPS_C,
                    en="`{call}` gives {got}, and it should give {want}: your function leaves out `a` and `b`, but not `c`, and `c` often lies nearest to `b − a + c`. Which words does your loop go over?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie laat `a` en `b` weg, maar `c` niet, en `c` ligt vaak het dichtst bij `b − a + c`. Langs welke woorden gaat je loop?"),
            Mistake("keeps-a", wrong=_AN_KEEPS_A,
                    en="`{call}` gives {got}, and it should give {want}: your function leaves out `b` and `c`, but not `a`, and here `a` lies nearest to `b − a + c`. Which words does your loop go over?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie laat `b` en `c` weg, maar `a` niet, en hier ligt `a` het dichtst bij `b − a + c`. Langs welke woorden gaat je loop?"),
            Mistake("zero", wrong=_AN_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: your function sets the three words to 0, and in this case every other word has a cosine similarity below 0, so a word of the question still comes out highest. A cosine similarity lies between −1 and 1: which number is low enough?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie zet de drie woorden op 0, en in dit geval heeft elk ander woord een cosine similarity onder 0, dus een woord van de vraag komt nog steeds het hoogst uit. Een cosine similarity ligt tussen −1 en 1: welk getal is laag genoeg?"),
            Mistake("signs", wrong=_AN_SIGNS,
                    en="`{call}` gives {got}, and it should give {want}: your function computes `a − b + c`, a step from `b` to `a`, the wrong way. In *apple : tree = grape : ?* the step goes from *apple* to *tree* and is added to *grape*: which vector do you subtract?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie rekent `a − b + c` uit, een stap van `b` naar `a`, de verkeerde kant op. In *apple : tree = grape : ?* gaat de stap van *apple* naar *tree* en wordt die stap bij *grape* opgeteld: welke vector trek je af?"),
            Mistake("minus-c", wrong=_AN_MINUS_C,
                    en="`{call}` gives {got}, and it should give {want}: your function subtracts `c` too. The step from `a` to `b` is added to `c`: which sign does `c` get?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie trekt `c` ook af. De stap van `a` naar `b` wordt bij `c` opgeteld: welk teken krijgt `c`?"),
            Mistake("argmin", wrong=_AN_ARGMIN,
                    en="`{call}` gives {got}, and it should give {want}: `np.argmin` picks the lowest number, so your function gives the row with the lowest cosine similarity. Which function picks the highest?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `np.argmin` kiest het laagste getal, dus je functie geeft de rij met de laagste cosine similarity. Welke functie kiest het hoogste?"),
            Mistake("dot", wrong=_AN_DOT,
                    en="`{call}` gives {got}, and it should give {want}: with dot products instead of cosine similarities, the longest rows win. Which function of Exercise 1 gives the cosine similarities?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: met dot products in plaats van cosine similarities winnen de langste rijen. Welke functie van oefening 1 geeft de cosine similarities?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: compute `b − a + c`, the cosine similarity of every row with it, leave out the three words of the question, and give back the word with the highest cosine similarity.",
                      "nl": "Vervang de `...`: reken `b − a + c` uit en de cosine similarity van elke rij daarmee, laat de drie woorden van de vraag weg, en geef het woord met de hoogste cosine similarity terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The answer is the word whose row has the highest cosine similarity with `E[index[b]] - E[index[a]] + E[index[c]]`, leaving out `a`, `b` and `c` themselves.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Het antwoord is het woord waarvan de rij de hoogste cosine similarity heeft met `E[index[b]] - E[index[a]] + E[index[c]]`, met `a`, `b` en `c` zelf weggelaten."},
            "almost": {"en": "Right for the first two cases. But `{call}` gives {got}, and it should give {want}. Does your function leave out all three words of the question, and does it give back a word?",
                       "nl": "Goed voor de eerste twee gevallen. Maar `{call}` geeft {got}, en dat moet {want} zijn. Laat je functie alle drie de woorden van de vraag weg, en geeft je functie een woord terug?"},
            "correct": {"en": "Your `analogy` is right in all {n} cases. Look at the picture: your function on six questions, next to the words nearest to `b − a + c`.",
                        "nl": "Je `analogy` klopt in alle {n} gevallen. Kijk naar de afbeelding: je functie op zes vragen, naast de woorden die het dichtst bij `b − a + c` liggen."},
        },
    ),
    "groups": Sort(
        name="groups",
        groups={
            "high": {"en": "More than half right", "nl": "Meer dan de helft goed"},
            "low": {"en": "Fewer than one in five right", "nl": "Minder dan één op de vijf goed"},
        },
        items={
            "nationality": {"en": "country and nationality, as in *albania : albanian*",
                            "nl": "land en nationaliteit, zoals *albania : albanian*"},
            "capital": {"en": "capital and country, as in *athens : greece*",
                        "nl": "hoofdstad en land, zoals *athens : greece*"},
            "family": {"en": "male and female, as in *boy : girl*",
                       "nl": "mannelijk en vrouwelijk, zoals *boy : girl*"},
            "currency": {"en": "country and currency, as in *algeria : dinar*",
                         "nl": "land en munt, zoals *algeria : dinar*"},
            "opposite": {"en": "word and its opposite, as in *acceptable : unacceptable*",
                         "nl": "woord en tegendeel, zoals *acceptable : unacceptable*"},
            "adverb": {"en": "adjective and adverb, as in *amazing : amazingly*",
                       "nl": "bijvoeglijk naamwoord en bijwoord, zoals *amazing : amazingly*"},
        },
        solution={"nationality": "high", "capital": "high", "family": "high",
                  "currency": "low", "opposite": "low", "adverb": "low"},
        nudge={
            "nationality": {"en": "Do *albania* and *albanian* stand in the same kind of sentences, as every country and its people do?",
                            "nl": "Staan *albania* en *albanian* in dezelfde soort zinnen, zoals elk land en de mensen ervan?"},
            "capital": {"en": "How often does a news text name a capital next to its country?",
                        "nl": "Hoe vaak noemt een nieuwstekst een hoofdstad naast het land?"},
            "family": {"en": "Is the step from *boy* to *girl* much like the step from *brother* to *sister*?",
                       "nl": "Lijkt de stap van *boy* naar *girl* veel op de stap van *brother* naar *sister*?"},
            "currency": {"en": "How often does a text name the *dinar* or the *kwanza*?",
                         "nl": "Hoe vaak noemt een tekst de *dinar* of de *kwanza*?"},
            "opposite": {"en": "Is the step from *acceptable* to *unacceptable* the same as from *aware* to *unaware*, and how often does a text use *unacceptable*?",
                         "nl": "Is de stap van *acceptable* naar *unacceptable* dezelfde als van *aware* naar *unaware*, en hoe vaak gebruikt een tekst *unacceptable*?"},
            "adverb": {"en": "Does *-ly* move every adjective to the same new place, or a little differently for each word?",
                       "nl": "Brengt *-ly* elk bijvoeglijk naamwoord naar dezelfde nieuwe plek, of voor elk woord een beetje anders?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the groups again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de groepen opnieuw uit."},
            "empty": {"en": "Put each group of questions under how often GloVe answers it right.",
                      "nl": "Zet elke groep vragen onder hoe vaak GloVe die groep goed heeft."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs under the other heading.",
                      "nl": "{item} hoort onder het andere kopje."},
            "correct": {"en": "All {n} are right. A relation that common words share in the same way, again and again, gives a step that is much the same for every pair; rare words, and steps that differ from word to word, do not.",
                        "nl": "Alle {n} kloppen. Een relatie die veelvoorkomende woorden steeds op dezelfde manier delen, geeft een stap die voor elk paar ongeveer hetzelfde is; zeldzame woorden, en stappen die van woord tot woord verschillen, doen dat niet."},
        },
    ),
}

# the group of word2vec behind each item of the sort
SORT_GROUPS = {"nationality": "gram6-nationality-adjective", "capital": "capital-common-countries", "family": "family",
               "currency": "currency", "opposite": "gram2-opposite", "adverb": "gram1-adjective-to-adverb"}

CASES = [
    ("cosines", _COS_START, "empty", None),
    ("cosines", _COS, "correct", None),
    ("cosines", _COS_LOOP, "correct", None),
    ("cosines", _COS_HEAD + "    return np.dot(E, v) / np.linalg.norm(E, axis=1) / np.linalg.norm(v)\n", "correct", None),
    ("cosines", _COS_HEAD + "    return (E / np.linalg.norm(E, axis=1, keepdims=True)) @ (v / np.linalg.norm(v))\n", "correct", None),
    ("cosines", _COS_DOT, "wrong", "dot"),
    ("cosines", _COS_HEAD + "    return E @ v / (np.linalg.norm(E, axis=1, keepdims=True) * np.linalg.norm(v))\n", "wrong", "table"),
    ("cosines", _COS_HEAD + "    return E.T @ v / (np.linalg.norm(E, axis=1) * np.linalg.norm(v))\n", "error", "order"),
    ("cosines", _COS_NO_V, "wrong", "no-v"),
    ("cosines", _COS_ONLY_V, "wrong", "only-v"),
    ("cosines", _COS_WHOLE, "wrong", "whole"),
    ("cosines", _COS_TIMES, "wrong", "times"),
    ("cosines", _COS_DISTANCE, "wrong", "distance"),
    ("cosines", _COS_AXIS0, "error", "axis"),
    ("cosines", _COS_ORDER, "error", "order"),
    ("cosines", _COS_HEAD + "    return np.dot(v, E) / (np.linalg.norm(E, axis=1) * np.linalg.norm(v))\n", "error", "order"),
    ("cosines", _COS_HEAD + "    return E @ v / (np.linalg.norm(E, axis=1) * np.linalg.norm(v[:2]))\n", "almost", None),
    ("cosines", _COS_HEAD + "    ...\n    return E @ v\n", "wrong", "dots-left"),
    ("analogy", _AN_START, "empty", None),
    ("analogy", _AN, "correct", None),
    ("analogy", _AN.replace("sims[index[word]] = -2", "sims[index[word]] = -np.inf"), "correct", None),
    ("analogy", _AN_HEAD + "    v = E[index[b]] - E[index[a]] + E[index[c]]\n    sims = cosines(E, v)\n    best = None\n"
                "    for i in range(len(words)):\n        if words[i] not in [a, b, c]:\n"
                "            if best is None or sims[i] > sims[best]:\n                best = i\n    return words[best]\n",
     "correct", None),
    ("analogy", _AN_KEEPS, "wrong", "keeps"),
    ("analogy", _AN_KEEPS_B, "wrong", "keeps-b"),
    ("analogy", _AN_KEEPS_C, "wrong", "keeps-c"),
    ("analogy", _AN_KEEPS_A, "wrong", "keeps-a"),
    ("analogy", _AN_ZERO, "wrong", "zero"),
    ("analogy", _AN_SIGNS, "wrong", "signs"),
    ("analogy", _AN_MINUS_C, "wrong", "minus-c"),
    ("analogy", _AN_NUMBER, "wrong", "number"),
    ("analogy", _AN_ARGMIN, "wrong", "argmin"),
    ("analogy", _AN_DOT, "wrong", "dot"),
    ("analogy", _AN_NO_INDEX, "error", "no-index"),
    ("analogy", _AN_MAX, "error", "max"),
    ("analogy", _AN_HEAD + "    if b == 'king':\n        return 'queen'\n    return 'rome'\n", "almost", None),
    ("groups", {}, "empty", None),
    ("groups", {"nationality": "high", "capital": "high", "family": "high", "currency": "low", "opposite": "low",
                "adverb": "low"}, "correct", None),
    ("groups", {"nationality": "high", "capital": "high"}, "wrong", "unfinished"),
    ("groups", {"nationality": "low"}, "wrong", "wrong:nationality"),
    ("groups", {"capital": "low"}, "wrong", "wrong:capital"),
    ("groups", {"family": "low"}, "wrong", "wrong:family"),
    ("groups", {"currency": "high"}, "wrong", "wrong:currency"),
    ("groups", {"opposite": "high"}, "wrong", "wrong:opposite"),
    ("groups", {"adverb": "high"}, "wrong", "wrong:adverb"),
]
