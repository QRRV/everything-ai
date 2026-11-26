# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ir-vector-space: the length of a vector of TF-IDF weights, kept as a dictionary from words to
weights (norm); the cosine of a query vector and a document vector, their dot product divided by both lengths (cosine),
as in Manning, Raghavan and Schütze, Introduction to Information Retrieval (2008), section 6.3; and pivoted length
normalization, which divides the dot product by a mix of the length of the document and a fixed pivot (pivoted), after
Singhal, Buckley and Mitra (1996) and section 6.4.4 of the book. The sort asks what changes the cosine of a query and a
document, and what does not.

The small collection is the thirteen documents of the lesson ir-what-is-search. The large one is the 2,067 paragraphs of
SQuAD's development set in English and in SQuAD-NL in Dutch (everything_ai.xquad), with the 1,190 English and 1,183
Dutch questions of XQuAD: the paragraph a question was written about is its own paragraph. Words are cut as the lesson
ir-what-is-search cuts them, and a weight is a count times log10(N / df), as in the lesson nlp-tf-idf. Every number in
the text and the pictures is computed by the course (tests/test_ir_vector_space.py computes them again)."""
import collections
import functools
import html
import importlib.util
import math

import numpy as np

from everything_ai import paths, xquad, yours
from everything_ai.check import FunctionBox, Mistake, Sort, kept

LANGS = ("en", "nl")


def _inverted_index():
    spec = importlib.util.spec_from_file_location("everything_ai_checks_ir_inverted_index_for_vector_space",
                                                  paths.LESSONS / "ir-inverted-index" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


I = _inverted_index()   # the inverted index of the paragraphs, and what shows a paragraph
W = I.W                 # the thirteen documents, their queries and the way they are cut into words
S = I.S                 # the parts of the pictures and the exercises

TEXTS = W.TEXTS         # the thirteen documents, in both languages in the same order
DOCS = W.DOCS           # the same documents as lists of words
QUERIES = W.QUERIES     # the queries python and red wine heart, as lists of words
words = W.words         # a text as a list of words in small letters
showable = I.showable   # whether a picture may show paragraph i: shown by xquad, and no pronoun for one person
_PRONOUN = I._PRONOUN


# --- the course's solutions, the same code as the exercises -------------------------------------------------------------------

def norm(weights):
    """The length of a vector kept as a dictionary from words to weights: the square root of the squares added up."""
    total = 0
    for word in weights:
        total = total + weights[word] ** 2
    return total ** 0.5


def cosine(q, d):
    """The cosine of two vectors kept as dictionaries: their dot product divided by both lengths; 0.0 when one of the
    two has length 0."""
    if norm(q) == 0 or norm(d) == 0:
        return 0.0
    dot = 0
    for word in q:
        if word in d:
            dot = dot + q[word] * d[word]
    return dot / (norm(q) * norm(d))


def pivoted(dot, length, pivot, slope):
    """The dot product divided by a mix of the length of the document and the pivot: slope 1 divides by the length
    alone, slope 0 by the pivot alone. Works on numbers and on NumPy arrays."""
    divisor = (1 - slope) * pivot + slope * length
    return dot / divisor


# --- vectors -----------------------------------------------------------------------------------------------------------------

def idf_of(docs) -> dict:
    """log10(N / df) for each word of a list of documents, each a list of words."""
    df = collections.Counter(w for d in docs for w in set(d))
    return {w: math.log10(len(docs) / c) for w, c in df.items()}


def vector(words_, idf) -> dict:
    """The TF-IDF vector of a document or a query, a list of words: each word's count times its IDF. A word that the
    collection does not hold is left out: it has no IDF."""
    counts = collections.Counter(w for w in words_ if w in idf)
    return {w: c * idf[w] for w, c in counts.items()}


def dot_product(q, d) -> float:
    """The dot product of two vectors kept as dictionaries: the products of the weights of the words both hold, added up."""
    return sum(x * d[w] for w, x in q.items() if w in d)


SMALL_IDF = {lang: idf_of(DOCS[lang]) for lang in LANGS}                                    # the thirteen documents
SMALL_VECTORS = {lang: [vector(d, SMALL_IDF[lang]) for d in DOCS[lang]] for lang in LANGS}


# --- the paragraphs and the questions -------------------------------------------------------------------------------------

def _space(lang: str) -> dict:
    """The paragraphs as vectors: the IDF of each word, the vectors, their lengths (norms) and numbers of words, and for
    each word the paragraphs that hold it with the word's weight there, as arrays."""
    def make():
        docs = I._paragraph_words(lang)
        idf = idf_of(docs)
        vectors = tuple(vector(d, idf) for d in docs)
        places = collections.defaultdict(list)
        for i, v in enumerate(vectors):
            for w, x in v.items():
                places[w].append((i, x))
        postings = {w: (np.array([i for i, _ in p]), np.array([x for _, x in p])) for w, p in places.items()}
        return {"idf": idf, "vectors": vectors, "norms": np.array([norm(v) for v in vectors]),
                "lengths": np.array([len(d) for d in docs]), "postings": postings}
    return kept(f"ir-vector-space:space:{lang}", make)


def paragraph_vectors(lang: str) -> list:
    """The TF-IDF vectors of the 2,067 paragraphs in English (en) or Dutch (nl): a new copy each time."""
    assert lang in LANGS, lang
    return [dict(v) for v in _space(lang)["vectors"]]


def paragraph_idf(lang: str) -> dict:
    assert lang in LANGS, lang
    return dict(_space(lang)["idf"])


def _scores(lang: str) -> dict:
    """For each question, the score of every paragraph: by the score of nlp-tf-idf (the weights of the words of the
    question in the paragraph, added up, a word twice in the question counted twice) and by the dot product with the
    vector of the question; and the length of each question vector."""
    def make():
        sp = _space(lang)
        qs = xquad.questions(lang)
        n = len(sp["vectors"])
        summed, dots, qnorms = np.zeros((len(qs), n)), np.zeros((len(qs), n)), np.zeros(len(qs))
        for k, q in enumerate(qs):
            qw = words(q.question)
            for w in qw:
                if w in sp["postings"]:
                    where, x = sp["postings"][w]
                    summed[k, where] += x
            qv = vector(qw, sp["idf"])
            for w, y in qv.items():
                where, x = sp["postings"][w]
                dots[k, where] += y * x
            qnorms[k] = norm(qv)
        return {"summed": summed, "dots": dots, "qnorms": qnorms, "own": np.array([q.paragraph for q in qs])}
    return kept(f"ir-vector-space:scores:{lang}", make)


def fifths(lang: str) -> np.ndarray:
    """For each paragraph, its group by number of words, from 0 for the shortest to 4 for the longest: about a fifth of
    the paragraphs each, cut at the lengths of the paragraphs at a fifth, two fifths, and so on, from short to long.
    Paragraphs of the same length are in the same group."""
    lengths = _space(lang)["lengths"]
    ordered = np.sort(lengths)
    edges = [ordered[k * len(lengths) // 5] for k in range(1, 5)]
    return np.searchsorted(edges, lengths, side="right")


def pivot(lang: str) -> float:
    """The pivot of this lesson: the average length (norm) of the paragraph vectors."""
    return float(_space(lang)["norms"].mean())


def _ranked(lang: str, how: str, slope: float = 0.8) -> np.ndarray:
    """The scores of every paragraph for every question: how is "sum", "dot", "cosine" or "pivoted"."""
    sc, sp = _scores(lang), _space(lang)
    if how == "sum":
        return sc["summed"]
    if how == "dot":
        return sc["dots"]
    if how == "cosine":
        return sc["dots"] / sc["qnorms"][:, None] / sp["norms"][None, :]
    return pivoted(sc["dots"], sp["norms"][None, :], pivot(lang), slope)


def outcome(lang: str, scores: np.ndarray) -> dict:
    """What a ranking does: the first paragraph of each question, the place of its own paragraph (1 is first), the
    share of questions whose own paragraph comes first, overall and for questions whose own paragraph is in the longest
    fifth, and for each fifth the share of first places and of own paragraphs."""
    sc, sp, f = _scores(lang), _space(lang), fifths(lang)
    own = sc["own"]
    first = np.argmax(scores, axis=1)
    mine = scores[np.arange(len(own)), own]
    place = 1 + (scores > mine[:, None]).sum(axis=1)
    hit = first == own
    long_ = f[own] == 4
    return {"first": first, "place": place, "hit": float(hit.mean()), "hit_long": float(hit[long_].mean()),
            "hit_rest": float(hit[~long_].mean()),
            "first_share": [float((f[first] == k).mean()) for k in range(5)],
            "own_share": [float((f[own] == k).mean()) for k in range(5)],
            "first_words": float(sp["lengths"][first].mean()), "own_words": float(sp["lengths"][own].mean())}


@functools.cache
def results(lang: str) -> dict:
    """outcome() of the four rankings of the lesson, the pivoted one with slope 0.8."""
    return {how: outcome(lang, _ranked(lang, how)) for how in ("sum", "dot", "cosine", "pivoted")}


@functools.cache
def numbers(lang: str) -> dict:
    """What the lesson says about the paragraphs and the questions: how many there are, how many different words the
    paragraphs hold (the dimensions of a vector), and for how many questions a search with AND finds anything, and
    finds the own paragraph."""
    sp, index = _space(lang), I._paragraph_index(lang)
    qs = xquad.questions(lang)
    found = [I.search_and(words(q.question), index) for q in qs]
    lengths = sp["lengths"]
    return {"paragraphs": len(lengths), "questions": len(qs), "dimensions": len(sp["idf"]),
            "and_any": sum(1 for f in found if f), "and_own": sum(1 for q, f in zip(qs, found) if q.paragraph in f),
            "shortest": int(lengths.min()), "longest": int(lengths.max()), "words": float(lengths.mean())}


# The question of the idea, of the picture under cosine, and the slope of the picture under pivoted: the same in both
# lessons (the questions of XQuAD have the same id in both languages).
EXAMPLE = "570d2f5bfed7b91900d45cd3"      # What document formed the Parliament of Victoria?
EXAMPLES = (EXAMPLE, "57115bf350c2381900b54a95")   # and: How many cylinders does the Energiprojekt AB engine have?
SLOPE = 0.8


def question_place(qid: str, lang: str) -> int:
    """The place of a question in xquad.questions(lang)."""
    return next(k for k, q in enumerate(xquad.questions(lang)) if q.id == qid)


def top(lang: str, how: str, k: int, n: int = 3) -> list:
    """The first n paragraphs for question k by a ranking of the lesson, as (paragraph, score)."""
    s = _ranked(lang, how)[k]
    order = np.argsort(-s, kind="stable")[:n]
    return [(int(i), float(s[i])) for i in order]


# --- the pictures ----------------------------------------------------------------------------------------------------------

_UI, _LABEL, _CELL, _HEAD, _NUMBER = S._UI, S._LABEL, S._CELL, S._HEAD, S._NUMBER
_DATA, _FAINT, _PRED, _LOSS = S._DATA, S._FAINT, S._PRED, S._LOSS
_CHIP, _CHIP_PRED, _CHIP_DATA, _CHIP_LOSS = S._CHIP, S._CHIP_PRED, S._CHIP_DATA, S._CHIP_LOSS
_note, _colors, _label, _num, _pct, _dec = S._note, S._colors, S._label, S._num, S._pct, S._dec
_cell, _table = S._cell, S._table
_fn, _call, _empty, _SLOW, _STOPPED = S._fn, S._call, S._empty, S._SLOW, S._STOPPED
_WHY_EMPTY = S._WHY_EMPTY
_MONO = "font-family:var(--ea-font-code);font-size:0.8rem"
_DASH = f'<span style="{_UI}">–</span>'
_GRAY_BAR, _BLUE_BAR = "var(--ea-data)", "var(--ea-pred)"


def _chip(text_: str, style: str = _CHIP_DATA) -> str:
    return f'<span style="{style}">{html.escape(str(text_))}</span>'


def _code(text_: str) -> str:
    return f'<code style="{_MONO}">{html.escape(text_)}</code>'


def _bar(x: float, color: str, most: float = 0.6, width: float = 6.0) -> str:
    w = 0.0 if not x or x <= 0 else max(0.15, width * min(x, most) / most)
    return (f'<span style="display:inline-block;width:{w:.2f}rem;height:0.6rem;background:{color};border-radius:2px;'
            f'margin-right:0.35rem;vertical-align:middle"></span>')


def _share(x: float, lang: str, color: str, style: str) -> str:
    return _bar(x, color) + f'<span style="{style}">{html.escape(_pct(x, lang))}</span>'


def _is_number(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) and math.isfinite(float(v))


def _fifth_rows(lang: str, columns, w: dict) -> list:
    """A row for each fifth of the paragraphs by length: its words, then a cell for each (shares, color, style)."""
    f, lengths = fifths(lang), _space(lang)["lengths"]
    rows = []
    for k in range(5):
        lo, hi = int(lengths[f == k].min()), int(lengths[f == k].max())
        cells = "".join(_cell(_share(shares[k], lang, color, style)) for shares, color, style in columns)
        rows.append("<tr>" + _cell(html.escape(w["fifth"].format(lo=_num(lo, lang), hi=_num(hi, lang))), _DATA) + cells
                    + "</tr>")
    return rows


def _title_cell(i: int, lang: str, own: int, score: str = "") -> str:
    """A cell with the title of paragraph i, blue for the own paragraph, its number of words and a score."""
    p = xquad.paragraphs(lang)[i]
    n = int(_space(lang)["lengths"][i])
    said = f'{_num(n, lang)} {_WORDS[lang]}' + (f" · {score}" if score else "")
    return _cell(_chip(p.title, _CHIP_PRED if i == own else _CHIP_DATA)
                 + f'<span style="{_UI};margin-left:0.3rem">{html.escape(said)}</span>')


_WORDS = {"en": "words", "nl": "woorden"}


_IDEA_TEXT = {
    "en": {"title": "One question, three rankings, and where the first places go",
           "question": "The question *{q}* was written about a paragraph of {n} words from the article *{t}*. The first three paragraphs of each ranking:",
           "head": ["Place", "TF-IDF score", "Dot product", "Cosine"],
           "places": "The own paragraph comes at place {a} by the TF-IDF score, at place {b} by the dot product, and first by the cosine.",
           "fifths": "All {k} questions, with the {n} paragraphs in five groups by length, from short to long:",
           "head_fifths": ["Paragraphs", "Own paragraphs", "First: TF-IDF score", "First: dot product", "First: cosine"],
           "fifth": "{lo} to {hi} words",
           "hits": "The own paragraph comes first for {a} of the questions by the TF-IDF score, for {b} by the dot product, and for {c} by the cosine.",
           "gray": "Gray: the paragraphs, and for what share of the questions the own paragraph is in each group.",
           "blue": "Blue: the own paragraph of the question, and how many of the first places each group gets."},
    "nl": {"title": "Eén vraag, drie rangschikkingen, en waar de eerste plaatsen terechtkomen",
           "question": "De vraag *{q}* is geschreven bij een alinea van {n} woorden van de Wikipedia-pagina *{t}*. De eerste drie alinea's van elke rangschikking:",
           "head": ["Plaats", "TF-IDF-score", "Inproduct", "Cosine similarity"],
           "places": "De eigen alinea komt op plaats {a} met de TF-IDF-score, op plaats {b} met het inproduct, en bovenaan met de cosine similarity.",
           "fifths": "Alle {k} vragen, met de {n} alinea's in vijf groepen op lengte, van kort naar lang:",
           "head_fifths": ["Alinea's", "Eigen alinea's", "Bovenaan: TF-IDF-score", "Bovenaan: inproduct", "Bovenaan: cosine similarity"],
           "fifth": "{lo} tot {hi} woorden",
           "hits": "De eigen alinea komt bovenaan bij {a} van de vragen met de TF-IDF-score, bij {b} met het inproduct, en bij {c} met de cosine similarity.",
           "gray": "Grijs: de alinea's, en bij welk deel van de vragen de eigen alinea in elke groep zit.",
           "blue": "Blauw: de eigen alinea van de vraag, en hoeveel van de eerste plaatsen elke groep krijgt."},
}


def idea_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang]
    k = question_place(EXAMPLE, lang)
    q = xquad.questions(lang)[k]
    own = q.paragraph
    p = xquad.paragraphs(lang)[own]
    n_own = int(_space(lang)["lengths"][own])
    body = _label(w["question"].format(q=q.question, n=_num(n_own, lang), t=p.title))
    tops = {how: top(lang, how, k) for how in ("sum", "dot", "cosine")}
    rows = []
    for place in range(3):
        cells = "".join(_title_cell(tops[how][place][0], lang, own, _dec(tops[how][place][1], lang, 3 if how == "cosine" else 1))
                        for how in ("sum", "dot", "cosine"))
        rows.append("<tr>" + _cell(str(place + 1), _FAINT, _NUMBER) + cells + "</tr>")
    body += _table(rows, w["head"], (0,))
    r = results(lang)
    body += _note(w["places"].format(a=int(r["sum"]["place"][k]), b=int(r["dot"]["place"][k])))
    body += _label(w["fifths"].format(k=_num(len(xquad.questions(lang)), lang), n=_num(numbers(lang)["paragraphs"], lang)))
    columns = [(r["sum"]["own_share"], _GRAY_BAR, _DATA)] + [(r[how]["first_share"], _BLUE_BAR, _PRED)
                                                             for how in ("sum", "dot", "cosine")]
    body += _table(_fifth_rows(lang, columns, w), w["head_fifths"])
    body += _note(w["hits"].format(a=_pct(r["sum"]["hit"], lang), b=_pct(r["dot"]["hit"], lang),
                                   c=_pct(r["cosine"]["hit"], lang)))
    body += _colors(w, body)
    return yours.figure(w["title"], body, alt=w["title"])


class _Odd(Exception):
    """A learner's function gave something a picture cannot use."""


_NORM_TEXT = {
    "en": {"title": "Your norm on the thirteen documents and on the 2,067 paragraphs",
           "kind": "a number",
           "small": "The thirteen documents of the course, each a vector of TF-IDF weights:",
           "head_small": ["Document", "Words", "Your norm", "The course's"],
           "big": "The 2,067 paragraphs in five groups by length, from short to long, with the average of your norm in each group:",
           "head_big": ["Paragraphs", "Average words", "Average norm"],
           "grows": "A longer paragraph usually has a longer vector: your norm is {a} on average for the shortest group, and {b} for the longest.",
           "differs": "Your norm differs from the course's for {k} of the thirteen documents.",
           "gray": "Gray: the documents and the paragraphs, and the course's norm where yours differs.",
           "blue": "Blue: your norm, where it is the course's.",
           "purple": "Purple: your norm, where it differs from the course's."},
    "nl": {"title": "Je norm op de dertien documenten en op de 2.067 alinea's",
           "kind": "een getal",
           "small": "De dertien documenten van de course, elk een vector van TF-IDF-waarden:",
           "head_small": ["Document", "Woorden", "Jouw norm", "Norm van de course"],
           "big": "De 2.067 alinea's in vijf groepen op lengte, van kort naar lang, met het gemiddelde van je norm in elke groep:",
           "head_big": ["Alinea's", "Gemiddeld aantal woorden", "Gemiddelde norm"],
           "grows": "Een langere alinea heeft meestal een langere vector: je norm is gemiddeld {a} voor de kortste groep, en {b} voor de langste.",
           "differs": "Bij {k} van de dertien documenten verschilt je norm van de norm van de course.",
           "gray": "Grijs: de documenten en de alinea's, en de norm van de course waar de jouwe verschilt.",
           "blue": "Blauw: je norm, waar deze gelijk is aan de norm van de course.",
           "purple": "Paars: je norm, waar deze verschilt van de norm van de course."},
}


def _same(a, b) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)


def norm_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _NORM_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    vectors = _space(lang)["vectors"]

    def step(v):
        out = fn(dict(v))
        if not _is_number(out):
            raise _Odd
        return float(out)

    def make():
        try:
            return [step(v) for v in SMALL_VECTORS[lang]], [step(v) for v in vectors]
        except _Odd:
            return None

    got = _call(make)
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    if got is None:
        return _empty(w, "odd")
    small, big = got
    rows, differs = [], 0
    for i, (mine, v) in enumerate(zip(small, SMALL_VECTORS[lang])):
        want = norm(v)
        same = _same(mine, want)
        differs += not same
        rows.append("<tr>" + _cell(_chip(i), "", _NUMBER) + _cell(_num(len(DOCS[lang][i]), lang), _DATA, _NUMBER)
                    + _cell(html.escape(_dec(mine, lang, 3)), _PRED if same else _LOSS, _NUMBER)
                    + _cell(_DASH if same else html.escape(_dec(want, lang, 3)), _DATA, _NUMBER) + "</tr>")
    body = _label(w["small"]) + _table(rows, w["head_small"], (0, 1, 2, 3))
    if differs:
        body += _note(w["differs"].format(k=_num(differs, lang)))
    f, lengths = fifths(lang), _space(lang)["lengths"]
    big = np.array(big)
    rows, means = [], []
    for k in range(5):
        lo, hi = int(lengths[f == k].min()), int(lengths[f == k].max())
        mean = float(big[f == k].mean())
        means.append(mean)
        same = _same(mean, float(_space(lang)["norms"][f == k].mean()))
        rows.append("<tr>" + _cell(html.escape(w_fifth(lang).format(lo=_num(lo, lang), hi=_num(hi, lang))), _DATA)
                    + _cell(html.escape(_dec(float(lengths[f == k].mean()), lang, 1)), _DATA, _NUMBER)
                    + _cell(html.escape(_dec(mean, lang, 2)), _PRED if same else _LOSS, _NUMBER) + "</tr>")
    body += _label(w["big"]) + _table(rows, w["head_big"], (1, 2))
    body += _note(w["grows"].format(a=_dec(means[0], lang, 2), b=_dec(means[4], lang, 2)))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


def w_fifth(lang: str) -> str:
    return _IDEA_TEXT[lang]["fifth"]


_COSINE_TEXT = {
    "en": {"title": "Your cosine ranks the 2,067 paragraphs for two questions",
           "kind": "a number",
           "question": "*{q}* The own paragraph has {n} words, from the article *{t}*.",
           "head": ["Place", "Paragraph", "Your cosine"],
           "place": "Your cosine puts the own paragraph at place {a}; the course's cosine at place {b}.",
           "same": "Your cosine gives every paragraph the course's cosine, for both questions.",
           "differs": "Your cosine differs from the course's for {k} of the {n} paragraphs, over the two questions.",
           "gray": "Gray: the paragraphs that are not the own paragraph of the question.",
           "blue": "Blue: the own paragraph of the question, and your cosine where it is the course's.",
           "purple": "Purple: your cosine, where it differs from the course's."},
    "nl": {"title": "Je cosine rangschikt de 2.067 alinea's voor twee vragen",
           "kind": "een getal",
           "question": "*{q}* De eigen alinea heeft {n} woorden, van de Wikipedia-pagina *{t}*.",
           "head": ["Plaats", "Alinea", "Jouw cosine"],
           "place": "Je cosine zet de eigen alinea op plaats {a}; de cosine van de course op plaats {b}.",
           "same": "Je cosine geeft elke alinea dezelfde waarde als de cosine van de course, bij beide vragen.",
           "differs": "Je cosine verschilt van de cosine van de course bij {k} van de {n} alinea's, over de twee vragen.",
           "gray": "Grijs: de alinea's die niet de eigen alinea van de vraag zijn.",
           "blue": "Blauw: de eigen alinea van de vraag, en je cosine waar deze gelijk is aan de cosine van de course.",
           "purple": "Paars: je cosine, waar deze verschilt van de cosine van de course."},
}
_SHOWN_PLACES = 3   # paragraphs of each ranking that the picture under cosine shows


def query_vector(qid: str, lang: str) -> dict:
    """The TF-IDF vector of a question of XQuAD, with the IDF of the paragraphs."""
    return vector(words(xquad.questions(lang)[question_place(qid, lang)].question), _space(lang)["idf"])


def cosine_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _COSINE_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    vectors = _space(lang)["vectors"]
    queries = [query_vector(qid, lang) for qid in EXAMPLES]

    def make():
        out = []
        for qv in queries:
            row = []
            for v in vectors:
                x = fn(dict(qv), dict(v))
                if not _is_number(x):
                    raise _Odd
                row.append(float(x))
            out.append(np.array(row))
        return out

    def guarded():
        try:
            return make()
        except _Odd:
            return None

    got = _call(guarded)
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    if got is None:
        return _empty(w, "odd")
    body, differs = "", 0
    norms = _space(lang)["norms"]
    for qid, qv, mine in zip(EXAMPLES, queries, got):
        k = question_place(qid, lang)
        q = xquad.questions(lang)[k]
        own = q.paragraph
        want = _scores(lang)["dots"][k] / (norm(qv) * norms)
        same = np.isclose(mine, want, rtol=1e-6, atol=1e-9)
        differs += int((~same).sum())
        p = xquad.paragraphs(lang)[own]
        body += _label(w["question"].format(q=q.question, n=_num(int(_space(lang)["lengths"][own]), lang), t=p.title))
        order = np.argsort(-mine, kind="stable")
        rows = []
        for place, i in enumerate(order[:_SHOWN_PLACES], 1):
            rows.append("<tr>" + _cell(str(place), _FAINT, _NUMBER) + _title_cell(int(i), lang, own)
                        + _cell(html.escape(_dec(float(mine[i]), lang, 3)), _PRED if same[i] else _LOSS, _NUMBER) + "</tr>")
        body += _table(rows, w["head"], (0, 2))
        a = 1 + int((mine > mine[own]).sum())
        b = 1 + int((want > want[own]).sum())
        body += _note(w["place"].format(a=_num(a, lang), b=_num(b, lang)))
    n = len(EXAMPLES) * len(vectors)
    body += _note(w["same"] if not differs else w["differs"].format(k=_num(differs, lang), n=_num(n, lang)))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


_PIVOT_TEXT = {
    "en": {"title": "Your pivoted on the 2,067 paragraphs and all the questions",
           "kind": "an array with a score for each question and paragraph",
           "setting": "Slope {s}, and as the pivot the average length of the paragraph vectors, {p}. The paragraphs in five groups by length, from short to long:",
           "head": ["Paragraphs", "Own paragraphs", "First: cosine", "First: your pivoted"],
           "hits": "Your pivoted puts the own paragraph first for {a} of the questions, the cosine for {b}. For the questions whose own paragraph is in the longest group: {c}, and {d} by the cosine; for the other questions: {e}, and {f} by the cosine.",
           "course": "The course's pivoted puts the own paragraph first for {a} of the questions.",
           "gray": "Gray: the paragraphs, for what share of the questions the own paragraph is in each group, and the first places of the cosine.",
           "blue": "Blue: the first places of your pivoted, where they are the course's.",
           "purple": "Purple: the first places of your pivoted, where they differ from the course's."},
    "nl": {"title": "Je pivoted op de 2.067 alinea's en alle vragen",
           "kind": "een array met een score voor elke vraag en alinea",
           "setting": "Helling {s}, en als pivot de gemiddelde lengte van de vectoren van de alinea's, {p}. De alinea's in vijf groepen op lengte, van kort naar lang:",
           "head": ["Alinea's", "Eigen alinea's", "Bovenaan: cosine similarity", "Bovenaan: jouw pivoted"],
           "hits": "Je pivoted zet de eigen alinea bovenaan bij {a} van de vragen, de cosine similarity bij {b}. Bij de vragen met de eigen alinea in de langste groep: {c}, en {d} met de cosine similarity; bij de andere vragen: {e}, en {f} met de cosine similarity.",
           "course": "De pivoted van de course zet de eigen alinea bovenaan bij {a} van de vragen.",
           "gray": "Grijs: de alinea's, bij welk deel van de vragen de eigen alinea in elke groep zit, en de eerste plaatsen van de cosine similarity.",
           "blue": "Blauw: de eerste plaatsen van je pivoted, waar je pivoted hetzelfde geeft als de pivoted van de course.",
           "purple": "Paars: de eerste plaatsen van je pivoted, waar je pivoted iets anders geeft dan de pivoted van de course."},
}


def pivoted_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _PIVOT_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    dots, norms = _scores(lang)["dots"], _space(lang)["norms"]

    def make():
        out = fn(dots.copy(), norms[None, :].copy(), pivot(lang), SLOPE)
        if not (isinstance(out, np.ndarray) and out.shape == dots.shape and np.issubdtype(out.dtype, np.number)
                and np.isfinite(out).all()):
            return None
        return out

    got = _call(make)
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    if got is None:
        return _empty(w, "odd")
    mine, r = outcome(lang, got), results(lang)
    want = r["pivoted"]
    same = bool((mine["first"] == want["first"]).all())
    style = _PRED if same else _LOSS
    body = _label(w["setting"].format(s=_dec(SLOPE, lang, 1), p=_dec(pivot(lang), lang, 1)))
    columns = [(r["cosine"]["own_share"], _GRAY_BAR, _DATA), (r["cosine"]["first_share"], _GRAY_BAR, _DATA),
               (mine["first_share"], _BLUE_BAR if same else "var(--ea-loss)", style)]
    body += _table(_fifth_rows(lang, columns, _IDEA_TEXT[lang]), w["head"])
    c = r["cosine"]
    body += _note(w["hits"].format(a=_pct(mine["hit"], lang), b=_pct(c["hit"], lang), c=_pct(mine["hit_long"], lang),
                                   d=_pct(c["hit_long"], lang), e=_pct(mine["hit_rest"], lang), f=_pct(c["hit_rest"], lang)))
    if not same:
        body += f'<div style="{_LOSS}">' + html.escape(w["course"].format(a=_pct(want["hit"], lang))) + "</div>"
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


_TRY_TEXT = {
    "en": {"title": "Pivoted length normalization with slope {s}",
           "divisor": "Each paragraph's dot product is divided by {s} × its length + {r} × {p}.",
           "one": "Slope 1 divides by the length of the paragraph vector alone: the same order as the cosine.",
           "zero": "Slope 0 divides every paragraph by the same number: the same order as the dot product.",
           "head": ["Questions", "Own paragraph first: slope {s}", "Own paragraph first: cosine"],
           "all": "all {n}",
           "long": "own paragraph in the longest group",
           "rest": "the other questions",
           "fifths": "Where the first places go, with the paragraphs in five groups by length:",
           "head_fifths": ["Paragraphs", "Own paragraphs", "First: slope {s}", "First: cosine"],
           "words": "The first paragraphs have {a} words on average, and the own paragraphs {b}.",
           "gray": "Gray: the paragraphs and the questions, for what share of the questions the own paragraph is in each group, and what the cosine gives.",
           "blue": "Blue: what pivoted length normalization gives with this slope."},
    "nl": {"title": "Pivoted length normalization met helling {s}",
           "divisor": "Het inproduct van elke alinea wordt gedeeld door {s} × de lengte + {r} × {p}.",
           "one": "Helling 1 deelt alleen door de lengte van de vector van de alinea: dezelfde volgorde als de cosine similarity.",
           "zero": "Helling 0 deelt elke alinea door hetzelfde getal: dezelfde volgorde als het inproduct.",
           "head": ["Vragen", "Eigen alinea bovenaan: helling {s}", "Eigen alinea bovenaan: cosine similarity"],
           "all": "alle {n}",
           "long": "eigen alinea in de langste groep",
           "rest": "de andere vragen",
           "fifths": "Waar de eerste plaatsen terechtkomen, met de alinea's in vijf groepen op lengte:",
           "head_fifths": ["Alinea's", "Eigen alinea's", "Bovenaan: helling {s}", "Bovenaan: cosine similarity"],
           "words": "De eerste alinea's hebben gemiddeld {a} woorden, en de eigen alinea's {b}.",
           "gray": "Grijs: de alinea's en de vragen, bij welk deel van de vragen de eigen alinea in elke groep zit, en wat de cosine similarity geeft.",
           "blue": "Blauw: wat pivoted length normalization met deze helling geeft."},
}
TRY_START = 1.0   # the slope the slider starts at


def try_picture(lang: str, slope=TRY_START) -> str:
    lang = lang if lang in LANGS else "en"
    w = _TRY_TEXT[lang]
    try:
        s = min(1.0, max(0.0, float(slope)))
    except (TypeError, ValueError):
        s = TRY_START
    s = round(s, 2)
    shown = _dec(s, lang, 2)
    mine, c = outcome(lang, _ranked(lang, "pivoted", s)), results(lang)["cosine"]
    body = _note(w["divisor"].format(s=shown, r=_dec(1 - s, lang, 2), p=_dec(pivot(lang), lang, 1)))
    if s == 1.0:
        body += _note(w["one"])
    elif s == 0.0:
        body += _note(w["zero"])
    rows = []
    for key, k in (("all", "hit"), ("long", "hit_long"), ("rest", "hit_rest")):
        label = w[key].format(n=_num(numbers(lang)["questions"], lang))
        rows.append("<tr>" + _cell(html.escape(label), _DATA) + _cell(_share(mine[k], lang, _BLUE_BAR, _PRED))
                    + _cell(_share(c[k], lang, _GRAY_BAR, _DATA)) + "</tr>")
    body += _table(rows, [h.format(s=shown) for h in w["head"]])
    columns = [(c["own_share"], _GRAY_BAR, _DATA), (mine["first_share"], _BLUE_BAR, _PRED),
               (c["first_share"], _GRAY_BAR, _DATA)]
    body += _label(w["fifths"]) + _table(_fifth_rows(lang, columns, _IDEA_TEXT[lang]),
                                         [h.format(s=shown) for h in w["head_fifths"]])
    body += _note(w["words"].format(a=_dec(mine["first_words"], lang, 1), b=_dec(mine["own_words"], lang, 1)))
    title = w["title"].format(s=shown)
    return yours.figure(title, body + _colors(w, body), alt=title)


# --- the exercises ---------------------------------------------------------------------------------------------------------

_Val = S._Val
_DOTS = S._DOTS
_SAME_TEXTS = S._SAME_TEXTS


def _value(v) -> str:
    """A value the way the lesson writes it; a NumPy array as np.array([...])."""
    if isinstance(v, np.ndarray):
        return "np.array(" + S._value(v.tolist()) + ")"
    if isinstance(v, (np.floating, np.integer)):
        return S._value(v.item())
    return S._value(v)


def _show(v):
    out = _value(v)
    return out if len(out) <= 300 else out[:300] + "…"


def _written(v) -> _Val:
    return _Val(_value(v), v)


_NO_HEAD = "def norm(weights):\n    total = 0\n    for word in weights:\n"
_NO_END = "    return total ** 0.5\n"
_NO = _NO_HEAD + "        total = total + weights[word] ** 2\n" + _NO_END
_NO_START = _NO_HEAD + "        total = ...\n" + _NO_END
_NO_PLAIN = _NO_HEAD + "        total = total + weights[word]\n" + _NO_END
_NO_DOUBLE = _NO_HEAD + "        total = total + weights[word] * 2\n" + _NO_END
_NO_LAST = _NO_HEAD + "        total = weights[word] ** 2\n" + _NO_END

_CO_HEAD = ("def cosine(q, d):\n    if norm(q) == 0 or norm(d) == 0:\n        return 0.0\n    dot = 0\n    for word in q:\n"
            "        if word in d:\n            dot = dot + q[word] * d[word]\n")
_CO = _CO_HEAD + "    return dot / (norm(q) * norm(d))\n"
_CO_START = _CO_HEAD + "    return ...\n"
_CO_DOT = _CO_HEAD + "    return dot\n"
_CO_DOC = _CO_HEAD + "    return dot / norm(d)\n"
_CO_PLUS = _CO_HEAD + "    return dot / (norm(q) + norm(d))\n"
_CO_ORDER = _CO_HEAD + "    return dot / norm(q) * norm(d)\n"

_PV_HEAD = "def pivoted(dot, length, pivot, slope):\n"
_PV_END = "    return dot / divisor\n"
_PV = _PV_HEAD + "    divisor = (1 - slope) * pivot + slope * length\n" + _PV_END
_PV_START = _PV_HEAD + "    divisor = ...\n" + _PV_END
_PV_SWAP = _PV_HEAD + "    divisor = slope * pivot + (1 - slope) * length\n" + _PV_END
_PV_NO_SLOPE = _PV_HEAD + "    divisor = (1 - slope) * pivot + length\n" + _PV_END
_PV_LENGTH = _PV_HEAD + "    divisor = length\n" + _PV_END

# The small cases the exercise texts show; the same in both lessons.
TOY_NORM = ({"red": 3.0, "wine": 4.0}, {"python": 1.0, "snake": 2.0, "tank": 2.0})
TOY_COSINE = (({"red": 1.0, "wine": 1.0}, {"red": 2.0, "wine": 2.0}), ({"wine": 3.0, "heart": 4.0}, {"wine": 1.0}))
TOY_PIVOTED = ((5.6, 30.0, 20.0, 0.8), (3.0, 10.0, 20.0, 0.8))


def _small(lang: str, i: int) -> _Val:
    return _Val(f'C.SMALL_VECTORS["{lang}"][{i}]', SMALL_VECTORS[lang][i])


def _small_query(lang: str) -> _Val:
    q = QUERIES[lang]["wine"]
    return _Val(f'C.vector({_value(q)}, C.SMALL_IDF["{lang}"])', vector(q, SMALL_IDF[lang]))


REQUIRED = ["norm", "cosine", "pivoted", "changes"]

EXERCISES = {
    "norm": FunctionBox(
        name="norm",
        picture=norm_picture,
        function="norm",
        start=_NO_START,
        solution=_NO,
        show=_show,
        visible=[(_written(v),) for v in TOY_NORM],
        hidden=[(_written({}),),
                (_written({"the": 0.0, "heart": 0.5}),),
                (_written({"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}),),
                (_written({"wine": 2.5}),),
                (_small("en", 6),),
                (_small("nl", 10),)],
        first=[_DOTS],
        mistakes=[
            Mistake("plain", wrong=_NO_PLAIN,
                    en="`{call}` gives {got}, and it should give {want}: your function adds up the weights themselves, not their squares. What does each weight add to the length in the formula of the norm?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie telt de waarden zelf op, niet de kwadraten van de waarden. Wat voegt elke waarde toe aan de lengte in de formule van de norm?"),
            Mistake("double", wrong=_NO_DOUBLE,
                    en="`{call}` gives {got}, and it should give {want}: `* 2` doubles a weight, and `** 2` squares it. Which of the two does the norm need?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `* 2` verdubbelt een waarde, en `** 2` neemt het kwadraat. Welke van de twee heeft de norm nodig?"),
            Mistake("last", wrong=_NO_LAST,
                    en="`{call}` gives {got}, and it should give {want}: each round throws away `total` and keeps only the square of the last weight. What should the new `total` be made of?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke ronde gooit `total` weg en bewaart alleen het kwadraat van de laatste waarde. Waaruit bestaat de nieuwe `total`?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the new `total`: the total so far and the square of the weight of this word.",
                      "nl": "Vervang de `...` door de nieuwe `total`: het totaal tot nu toe en het kwadraat van de waarde van dit woord."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function add the square of each weight to the total, and take the square root at the end?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Telt je functie het kwadraat van elke waarde op bij het totaal, en neemt de functie aan het eind de wortel?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function also work for an empty vector, a weight of 0, and the vectors of the course's documents?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook voor een lege vector, een waarde 0, en de vectoren van de documenten van de course?"},
            "correct": {"en": "Right: your function measures the length of a vector of weights. In the picture below, it measures the course's documents and the 2,067 paragraphs: how much longer is the vector of a long paragraph?",
                        "nl": "Goed: je functie meet de lengte van een vector van waarden. In het plaatje hieronder meet de functie de documenten van de course en de 2.067 alinea's: hoeveel langer is de vector van een lange alinea?"},
        },
    ),
    "cosine": FunctionBox(
        name="cosine",
        picture=cosine_picture,
        function="cosine",
        start=_CO_START,
        solution=_CO,
        show=_show,
        names={"norm": norm},
        visible=[tuple(_written(v) for v in pair) for pair in TOY_COSINE],
        hidden=[(_written({"red": 1.0}), _written({"python": 2.0})),
                (_written({}), _written({"wine": 1.0})),
                (_written({"the": 0.0}), _written({"the": 0.0, "wine": 1.0})),
                (_written({"wine": 1.0}), _written({"wine": 1.0, "cheese": 1.0})),
                (_written({"red": 2.0, "wine": 1.0, "heart": 2.0}), _written({"red": 2.0, "wine": 1.0, "heart": 2.0})),
                (_small_query("en"), _small("en", 6)),
                (_small_query("en"), _small("en", 8)),
                (_small_query("nl"), _small("nl", 7))],
        first=[_DOTS],
        mistakes=[
            Mistake("dot", wrong=_CO_DOT,
                    en="`{call}` gives {got}, and it should give {want}: your function gives the dot product itself, which grows with the length of the document. What does the cosine divide the dot product by?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie geeft het inproduct zelf, en dat groeit met de lengte van het document. Waardoor deelt de cosine similarity het inproduct?"),
            Mistake("document", wrong=_CO_DOC,
                    en="`{call}` gives {got}, and it should give {want}: your function divides by the length of the document only. For one query, that gives the documents the same order as the cosine, but not the cosine itself. Which other length belongs in the division?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie deelt alleen door de lengte van het document. Voor één zoekvraag krijgen de documenten zo dezelfde volgorde als met de cosine similarity, maar de waarde is niet de cosine similarity zelf. Welke andere lengte hoort er nog bij in de deling?"),
            Mistake("plus", wrong=_CO_PLUS,
                    en="`{call}` gives {got}, and it should give {want}: your function divides by the two lengths added together. How does the cosine combine the two lengths?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie deelt door de twee lengtes bij elkaar opgeteld. Hoe combineert de cosine similarity de twee lengtes?"),
            Mistake("order", wrong=_CO_ORDER,
                    en="`{call}` gives {got}, and it should give {want}: Python works from left to right, so your function divides by `norm(q)` and then multiplies by `norm(d)`. How do parentheses make the division take both lengths at once?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: Python werkt van links naar rechts, dus je functie deelt door `norm(q)` en vermenigvuldigt daarna met `norm(d)`. Hoe laat je de functie met haakjes door het product van beide lengtes delen?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the cosine: the dot product divided by the lengths of both vectors.",
                      "nl": "Vervang de `...` door de cosine similarity: het inproduct gedeeld door de lengtes van beide vectoren."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function divide the dot product by the length of the query vector times the length of the document vector?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Deelt je functie het inproduct door de lengte van de vector van de zoekvraag maal de lengte van de vector van het document?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function also work for vectors without a word in common, a document with an extra word, and the course's documents?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook voor vectoren zonder gemeenschappelijk woord, een document met een extra woord, en de documenten van de course?"},
            "correct": {"en": "Right: your function gives the cosine of two vectors of weights. In the picture below, it ranks all 2,067 paragraphs for two questions: where does it put the own paragraph?",
                        "nl": "Goed: je functie geeft de cosine similarity van twee vectoren van waarden. In het plaatje hieronder rangschikt de functie alle 2.067 alinea's voor twee vragen: waar zet de functie de eigen alinea?"},
        },
    ),
    "pivoted": FunctionBox(
        name="pivoted",
        picture=pivoted_picture,
        function="pivoted",
        start=_PV_START,
        solution=_PV,
        show=_show,
        visible=[tuple(_written(v) for v in case) for case in TOY_PIVOTED],
        hidden=[tuple(_written(v) for v in (6.0, 12.0, 20.0, 1.0)),
                tuple(_written(v) for v in (6.0, 12.0, 20.0, 0.0)),
                tuple(_written(v) for v in (4.0, 20.0, 20.0, 0.5)),
                tuple(_written(v) for v in (np.array([5.6, 3.0]), np.array([30.0, 10.0]), 20.0, 0.8)),
                tuple(_written(v) for v in (np.array([[2.0, 2.0], [1.0, 3.0]]), np.array([10.0, 30.0]), 20.0, 0.5))],
        first=[_DOTS],
        mistakes=[
            Mistake("swap", wrong=_PV_SWAP,
                    en="`{call}` gives {got}, and it should give {want}: your divisor gives the length the share `1 - slope` and the pivot the share `slope`, the other way around. With slope 1, what should the divisor be?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je deler geeft de lengte het deel `1 - slope` en de pivot het deel `slope`, net andersom. Wat moet de deler zijn bij helling 1?"),
            Mistake("no-slope", wrong=_PV_NO_SLOPE,
                    en="`{call}` gives {got}, and it should give {want}: your divisor adds the whole length, so the shares of the pivot and the length add up to more than 1. What should the length be multiplied by?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je deler telt de hele lengte op, dus de delen van de pivot en de lengte zijn samen meer dan 1. Waarmee moet de lengte vermenigvuldigd worden?"),
            Mistake("length", wrong=_PV_LENGTH,
                    en="`{call}` gives {got}, and it should give {want}: your divisor is the length alone, as in the cosine, whatever the slope. Where do the pivot and the slope come in?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je deler is alleen de lengte, zoals bij de cosine similarity, wat de helling ook is. Waar komen de pivot en de helling erbij?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the divisor: a mix of the pivot and the length, set by the slope.",
                      "nl": "Vervang de `...` door de deler: een mengsel van de pivot en de lengte, bepaald door de helling."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your divisor give the pivot the share `1 - slope` and the length the share `slope`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Geeft je deler de pivot het deel `1 - slope` en de lengte het deel `slope`?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function also work for slope 1, slope 0, and NumPy arrays?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook voor helling 1, helling 0, en NumPy-arrays?"},
            "correct": {"en": "Right: your function divides by a mix of the length and the pivot. In the picture below, it ranks the paragraphs for all the questions with slope 0.8: which group of paragraphs gets more first places than with the cosine?",
                        "nl": "Goed: je functie deelt door een mengsel van de lengte en de pivot. In het plaatje hieronder rangschikt de functie de alinea's voor alle vragen met helling 0,8: welke groep alinea's krijgt meer eerste plaatsen dan met de cosine similarity?"},
        },
    ),
}

# The sort: what changes the cosine of the query red wine heart and document 6, and what does not.
EXERCISES["changes"] = Sort(
    name="changes",
    groups={
        "same": {"en": "The cosine stays the same", "nl": "De cosine similarity blijft gelijk"},
        "changes": {"en": "The cosine changes", "nl": "De cosine similarity verandert"},
    },
    items={
        "twice": {"en": "document 6 is written out twice in a row, so every count doubles",
                  "nl": "document 6 staat er twee keer achter elkaar, dus elk aantal verdubbelt"},
        "order": {"en": "the words of document 6 are put in another order",
                  "nl": "de woorden van document 6 komen in een andere volgorde"},
        "ten": {"en": "every weight of the vector of document 6 is multiplied by 10",
                "nl": "elke waarde van de vector van document 6 wordt met 10 vermenigvuldigd"},
        "query": {"en": "every weight of the query vector is doubled",
                  "nl": "elke waarde van de vector van de zoekvraag verdubbelt"},
        "extra": {"en": "document 6 also gets the word *python*, which is not in the query",
                  "nl": "document 6 krijgt er ook het woord *python* bij, dat niet in de zoekvraag staat"},
        "more": {"en": "document 6 gets one more *wine*",
                 "nl": "document 6 krijgt één keer extra *wijn*"},
        "drop": {"en": "every *heart* is taken out of document 6",
                 "nl": "*hart* verdwijnt uit document 6"},
        "repeat": {"en": "the query becomes *red wine wine heart*",
                   "nl": "de zoekvraag wordt *rode wijn wijn hart*"},
    },
    solution={"twice": "same", "order": "same", "ten": "same", "query": "same",
              "extra": "changes", "more": "changes", "drop": "changes", "repeat": "changes"},
    nudge={
        "twice": {"en": "Every weight of the document doubles. What happens to the dot product, and what happens to the length of the document?",
                  "nl": "Elke waarde van het document verdubbelt. Wat gebeurt er met het inproduct, en wat met de lengte van het document?"},
        "order": {"en": "Does a vector of counts know where in the document a word stands?",
                  "nl": "Weet een vector van aantallen waar in het document een woord staat?"},
        "ten": {"en": "The dot product becomes ten times as large. What happens to the length of the document vector?",
                "nl": "Het inproduct wordt tien keer zo groot. Wat gebeurt er met de lengte van de vector van het document?"},
        "query": {"en": "The dot product doubles, and so does the length of the query vector. What is left after the division?",
                  "nl": "Het inproduct verdubbelt, en de lengte van de vector van de zoekvraag ook. Wat blijft er over na de deling?"},
        "extra": {"en": "The new word adds nothing to the dot product. Does it add anything to the length of the document?",
                  "nl": "Het nieuwe woord voegt niets toe aan het inproduct. Voegt het iets toe aan de lengte van het document?"},
        "more": {"en": "One weight grows and the others stay as they are. Does the vector still point the same way?",
                 "nl": "Eén waarde groeit en de andere blijven gelijk. Wijst de vector nog dezelfde kant op?"},
        "drop": {"en": "A word of the query leaves the document. What does that do to the dot product?",
                 "nl": "Een woord van de zoekvraag verdwijnt uit het document. Wat doet dat met het inproduct?"},
        "repeat": {"en": "The weight of *wine* in the query doubles, and the other two stay as they are. Does the query vector still point the same way?",
                   "nl": "De waarde van *wijn* in de zoekvraag verdubbelt, en de andere twee blijven gelijk. Wijst de vector van de zoekvraag nog dezelfde kant op?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the groups again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de groepen opnieuw uit."},
        "empty": {"en": "Sort each change into the group that fits.",
                  "nl": "Deel elke verandering in bij de groep die klopt."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "{item} belongs with another group.",
                  "nl": "{item} hoort bij een andere groep."},
        "correct": {"en": "All {n} are right. The cosine looks only at the direction of the two vectors: making a vector longer or shorter as a whole changes nothing, and so does the order of the words. A change to one weight against the others turns the vector, and changes the cosine.",
                    "nl": "Alle {n} kloppen. De cosine similarity kijkt alleen naar de richting van de twee vectoren: een vector als geheel langer of korter maken verandert niets, en de volgorde van de woorden ook niet. Een verandering van één waarde ten opzichte van de andere draait de vector, en verandert de cosine similarity."},
    },
)

# What the tests feed each check, and what it must say: (exercise, code or choices, state, known mistake).
CASES = [
    ("norm", _NO_START, "empty", None),
    ("norm", _NO, "correct", None),
    ("norm", _NO_HEAD + "        total = total + ...\n" + _NO_END, "wrong", "dots-left"),
    ("norm", _NO_PLAIN, "wrong", "plain"),
    ("norm", _NO_DOUBLE, "wrong", "double"),
    ("norm", _NO_LAST, "wrong", "last"),
    ("norm", _NO_HEAD + "        total = total + weights[word] * weights[word]\n" + _NO_END, "correct", None),
    ("norm", "def norm(weights):\n    return sum(x ** 2 for x in weights.values()) ** 0.5\n", "correct", None),
    ("cosine", _CO_START, "empty", None),
    ("cosine", _CO, "correct", None),
    ("cosine", _CO_HEAD + "    return dot / (...)\n", "wrong", "dots-left"),
    ("cosine", _CO_DOT, "wrong", "dot"),
    ("cosine", _CO_DOC, "wrong", "document"),
    ("cosine", _CO_PLUS, "wrong", "plus"),
    ("cosine", _CO_ORDER, "wrong", "order"),
    ("cosine", _CO_HEAD + "    return dot / norm(q) / norm(d)\n", "correct", None),
    ("cosine", _CO_HEAD + "    return (dot / norm(d)) / norm(q)\n", "correct", None),
    ("pivoted", _PV_START, "empty", None),
    ("pivoted", _PV, "correct", None),
    ("pivoted", _PV_HEAD + "    divisor = (1 - slope) * pivot + ...\n" + _PV_END, "wrong", "dots-left"),
    ("pivoted", _PV_SWAP, "wrong", "swap"),
    ("pivoted", _PV_NO_SLOPE, "wrong", "no-slope"),
    ("pivoted", _PV_LENGTH, "wrong", "length"),
    ("pivoted", _PV_HEAD + "    divisor = pivot + slope * (length - pivot)\n" + _PV_END, "correct", None),
    ("changes", {}, "empty", None),
    ("changes", dict(EXERCISES["changes"].solution), "correct", None),
    ("changes", {"twice": "same", "extra": "changes"}, "wrong", "unfinished"),
] + [("changes", {item: ("changes" if group == "same" else "same")}, "wrong", f"wrong:{item}")
     for item, group in EXERCISES["changes"].solution.items()]
