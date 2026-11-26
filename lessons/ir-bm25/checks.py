# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ir-bm25: the part of BM25 that lets each repeat of a word count less than the one before
(saturation); the weight of one word in a document, with the count scaled by the length of the document against the
average length (term_weight); and the BM25 score of a document for a query, the weights of the words of the query added
up, each times its IDF (bm25), as in Manning, Raghavan and Schütze, Introduction to Information Retrieval (2008),
section 11.4.3, and Robertson and Zaragoza (2009). The sort asks which of the two settings, k1 or b, does what.

The small collection is the thirteen documents of the lesson ir-what-is-search; the paragraphs and questions are those
of the lessons ir-inverted-index and ir-vector-space (the 2,067 paragraphs and the 1,190 English and 1,183 Dutch
questions of XQuAD, everything_ai.xquad). The demo tunes k1 and b on SciFact (everything_ai.scifact), downloaded when the
learner asks: 5,183 abstracts and 809 claims to tune on, 300 to measure on. Words are cut as the lesson ir-what-is-search
cuts them, and the IDF is log10(N / df), as in the lesson nlp-tf-idf and the book. Every number in the text and the
pictures is computed by the course (tests/test_ir_bm25.py computes them again)."""
import collections
import functools
import html
import importlib.util
import math

import numpy as np

from everything_ai import lesson as lesson_helpers
from everything_ai import paths, scifact, xquad, yours
from everything_ai.check import FunctionBox, Mistake, Sort, kept
from everything_ai.lesson import download_size

LANGS = ("en", "nl")


def _vector_space():
    spec = importlib.util.spec_from_file_location("everything_ai_checks_ir_vector_space_for_bm25",
                                                  paths.LESSONS / "ir-vector-space" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V = _vector_space()     # the vector space model: the cosine of the paragraphs, and the groups by length
I = V.I                 # the inverted index of the paragraphs, and what shows a paragraph
W = V.W                 # the thirteen documents, their queries and the way they are cut into words
S = V.S                 # the parts of the pictures and the exercises

TEXTS = W.TEXTS         # the thirteen documents, in both languages in the same order
DOCS = W.DOCS           # the same documents as lists of words
QUERIES = W.QUERIES     # the queries python and red wine heart, as lists of words
words = W.words         # a text as a list of words in small letters
showable = I.showable   # whether a picture may show paragraph i: shown by xquad, and no pronoun for one person
_PRONOUN = I._PRONOUN

K1, B = 1.2, 0.75       # the usual settings of BM25, as in the book

DOWNLOAD_LINE = {   # what the lesson downloads, for the line under the header
    "en": f"Works on every computer. SciFact, 5,183 abstracts of biomedical papers with claims about them, is downloaded once from Hugging Face, {download_size(scifact.FILES, 'en')}, when you click its button in the lesson; the demo that tunes BM25 needs it, the rest of the lesson works without it.",
    "nl": f"Werkt op elke computer. SciFact, 5.183 abstracts van biomedische papers met beweringen erover, wordt één keer gedownload van Hugging Face, {download_size(scifact.FILES, 'nl')}, als je in de les op de knop van de download klikt; de demo die BM25 afstelt heeft de set nodig, de rest van de les werkt zonder.",
}


# --- the course's solutions, the same code as the exercises -------------------------------------------------------------------

def saturation(tf, k1):
    """What a word that occurs tf times adds, without the length: tf * (k1 + 1) / (tf + k1). 1 for tf 1, never more
    than k1 + 1. Works on numbers and on NumPy arrays."""
    return tf * (k1 + 1) / (tf + k1)


def term_weight(tf, length, average, k1, b):
    """The weight of a word that occurs tf times in a document of `length` words, where the documents have `average`
    words on average: the saturation, with k1 scaled by the length against the average, by the share b. Works on numbers
    and on NumPy arrays."""
    scale = 1 - b + b * length / average
    return tf * (k1 + 1) / (tf + k1 * scale)


def bm25(query, doc, idf, average, k1, b):
    """The BM25 score of a document, a list of words, for a query, a list that holds each word once: for each word of
    the query in the document, its IDF times its term weight, added up."""
    score = 0
    for word in query:
        if word in doc:
            score = score + idf[word] * term_weight(doc.count(word), len(doc), average, k1, b)
    return score


def distinct(words_) -> list:
    """The words of a query, each once, in the order of their first place."""
    return list(dict.fromkeys(words_))


# --- the thirteen documents ---------------------------------------------------------------------------------------------------

SMALL_IDF = V.SMALL_IDF
SMALL_AVERAGE = {lang: sum(len(d) for d in DOCS[lang]) / len(DOCS[lang]) for lang in LANGS}


def small_scores(lang: str, k1: float = K1, b: float = B) -> list:
    """The BM25 score of each of the thirteen documents for the query red wine heart."""
    q = distinct(QUERIES[lang]["wine"])
    return [bm25(q, d, SMALL_IDF[lang], SMALL_AVERAGE[lang], k1, b) for d in DOCS[lang]]


# --- BM25 for many queries at once --------------------------------------------------------------------------------------------

def _collection(docs) -> dict:
    """A list of documents, each a list of words, as the lesson ranks it: one entry for each pair of a word and a
    document that holds it, with the number of the word, the document and the count of the word there (flat arrays);
    the number of each word; the IDF of each word, log10(N / df); the length of each document and their average; and
    the length of each TF-IDF vector of a document (for the cosine)."""
    vocab, word, doc, tf = {}, [], [], []
    for i, d in enumerate(docs):
        for w, c in collections.Counter(d).items():
            word.append(vocab.setdefault(w, len(vocab)))
            doc.append(i)
            tf.append(c)
    word, doc, tf = np.array(word, dtype=np.int64), np.array(doc, dtype=np.int64), np.array(tf, dtype=float)
    idf = np.log10(len(docs) / np.bincount(word, minlength=len(vocab))) if len(docs) else np.zeros(0)
    lengths = np.array([len(d) for d in docs], dtype=float)
    norms = np.sqrt(np.bincount(doc, weights=(tf * idf[word]) ** 2, minlength=len(docs)))
    return {"vocab": vocab, "word": word, "doc": doc, "tf": tf, "idf": idf, "lengths": lengths,
            "average": float(lengths.mean()) if len(docs) else 0.0, "norms": norms}


def _queries(M: dict, queries) -> dict:
    """Queries, each a list of words, as two sparse matrices with a row for each query and a column for each word of
    the collection: 1 for each word of the query, however often it occurs there (BM25), and the count of each word in
    the query (the cosine). A word that no document holds is left out."""
    import scipy.sparse as sp
    rows, cols, counts = [], [], []
    for k, q in enumerate(queries):
        for w, c in collections.Counter(q).items():
            if w in M["vocab"]:
                rows.append(k)
                cols.append(M["vocab"][w])
                counts.append(c)
    shape = (len(queries), len(M["vocab"]))
    return {"once": sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=shape),
            "counts": sp.csr_matrix((np.array(counts, dtype=float), (rows, cols)), shape=shape)}


def _rank(M: dict, Q, parts: np.ndarray) -> np.ndarray:
    """The score of every document for every query, with a row for each query: for each word of the query, its part
    in each document (parts, one for each pair of a word and a document), added up. A sparse matrix product."""
    import scipy.sparse as sp
    words_by_docs = sp.csr_matrix((parts, (M["word"], M["doc"])), shape=(len(M["vocab"]), len(M["lengths"])))
    return np.asarray((Q @ words_by_docs).todense())


def bm25_scores(M: dict, Q: dict, k1: float, b: float) -> np.ndarray:
    """The BM25 score of every document of the collection M for every query of Q."""
    parts = M["idf"][M["word"]] * term_weight(M["tf"], M["lengths"][M["doc"]], M["average"], k1, b)
    return _rank(M, Q["once"], parts)


def cosine_scores(M: dict, Q: dict) -> np.ndarray:
    """For every query of Q, the dot product of its TF-IDF vector with that of every document of M, divided by the
    length of the document vector: the order of the cosine of the lesson ir-vector-space."""
    norms = np.where(M["norms"] > 0, M["norms"], 1.0)
    return _rank(M, Q["counts"], M["idf"][M["word"]] ** 2 * M["tf"] / norms[M["doc"]])


# --- the paragraphs and the questions -------------------------------------------------------------------------------------

def _paragraphs(lang: str) -> tuple:
    """The 2,067 paragraphs as a collection, and the questions of XQuAD as queries."""
    def make():
        M = _collection(I._paragraph_words(lang))
        return M, _queries(M, [words(q.question) for q in xquad.questions(lang)])
    return kept(f"ir-bm25:paragraphs:{lang}", make)


def average(lang: str) -> float:
    """The average number of words of the 2,067 paragraphs."""
    return _paragraphs(lang)[0]["average"]


def paragraph_idf(lang: str) -> dict:
    """The IDF of each word of the paragraphs, log10(N / df), the same as in the lesson ir-vector-space: a new copy each
    time."""
    assert lang in LANGS, lang
    M = _paragraphs(lang)[0]
    return {w: float(M["idf"][j]) for w, j in M["vocab"].items()}


# The rankings of the idea, with their settings; "sum" and "cosine" are those of the lesson ir-vector-space.
SETTINGS = {"binary": (0.0, B), "no-length": (K1, 0.0), "bm25": (K1, B), "heavy": (5.0, B)}


def ranked(lang: str, how: str) -> np.ndarray:
    """The score of every paragraph for every question: how is "sum" or "cosine" (the lesson ir-vector-space), or a
    setting of SETTINGS."""
    if how in ("sum", "cosine"):
        return V._ranked(lang, how)
    return bm25_scores(*_paragraphs(lang), *SETTINGS[how])


@functools.cache
def results(lang: str) -> dict:
    """outcome() of each ranking of the lesson: the first paragraph and the place of the own paragraph of each question,
    the share of questions whose own paragraph comes first, and where the first places go by length."""
    return {how: outcome(lang, ranked(lang, how)) for how in ("sum", "cosine", *SETTINGS)}


fifths = V.fifths
question_place = V.question_place


# The question of the idea and of the picture under bm25: the same in both lessons.
EXAMPLE = "572fffb404bcaa1900d76ff1"     # Where did the discharge from glaciers go in Europe in the last Ice Age?
EXAMPLES = (EXAMPLE, "570d2f5bfed7b91900d45cd3")   # and the question about the Parliament of Victoria of ir-vector-space


def top(lang: str, how: str, k: int, n: int = 3) -> list:
    """The first n paragraphs for question k by a ranking of the lesson, as (paragraph, score)."""
    s = ranked(lang, how)[k]
    order = np.argsort(-s, kind="stable")[:n]
    return [(int(i), float(s[i])) for i in order]


def counts_of(lang: str, k: int, i: int) -> dict:
    """How often each word of question k, once each, occurs in paragraph i; words it does not hold are left out."""
    para = I._paragraph_words(lang)[i]
    return {w: para.count(w) for w in distinct(words(xquad.questions(lang)[k].question)) if w in para}


# --- SciFact -----------------------------------------------------------------------------------------------------------------

GRID = [(round(0.1 * i, 1), round(0.05 * j, 2)) for i in range(31) for j in range(21)]   # k1 from 0 to 3, b from 0 to 1
TUNED = (1.1, 0.65)          # the best setting of GRID on the training claims (tests/test_ir_bm25.py finds it again)
BEST_ON_TEST = (0.7, 1.0)    # the best setting of GRID on the test claims: the number not to trust


def _sci() -> dict:
    """SciFact as the lesson uses it: the abstracts as a collection, each its title and text as one list of words, and
    for each split its claims as queries, with the places of the relevant abstracts of each claim."""
    def make():
        M = _collection([words(a["title"] + " " + a["text"]) for a in scifact.abstracts()])
        out = {"M": M}
        for split in scifact.SPLITS:
            claims = scifact.claims(split)
            wanted = np.zeros((len(claims), len(M["lengths"])), dtype=bool)
            for row, c in enumerate(claims):
                wanted[row, c["relevant"]] = True
            out[split] = _queries(M, [words(c["text"]) for c in claims]) | {
                "relevant": [frozenset(c["relevant"]) for c in claims], "ids": [c["id"] for c in claims], "wanted": wanted}
        return out
    return kept("ir-bm25:scifact", make)


def _sci_hits(split: str, scores: np.ndarray) -> np.ndarray:
    return _shared_first(scores, _sci()[split]["wanted"])


def sci_first(split: str, k1: float, b: float) -> np.ndarray:
    """For each claim of the split, whether BM25 with these settings puts a relevant abstract first: 1 or 0, or the
    share of relevant abstracts among those that share the first place."""
    sc = _sci()
    return _sci_hits(split, bm25_scores(sc["M"], sc[split], float(k1), float(b)))


def sci_cosine(split: str) -> np.ndarray:
    """For each claim of the split, whether the cosine puts a relevant abstract first, as sci_first."""
    def make():
        sc = _sci()
        return _sci_hits(split, cosine_scores(sc["M"], sc[split]))
    return kept(f"ir-bm25:scifact:cosine:{split}", make)


def sci_numbers() -> dict:
    """What the lesson says about SciFact: how many abstracts and claims, and how many words an abstract has."""
    sc = _sci()
    lengths = sc["M"]["lengths"]
    return {"abstracts": len(lengths), "train": sc["train"]["once"].shape[0], "test": sc["test"]["once"].shape[0],
            "words": float(lengths.mean()), "shortest": int(lengths.min()), "longest": int(lengths.max())}


def grid_best(split: str) -> tuple:
    """The setting of GRID that puts a relevant abstract first for the most claims of the split; of equal ones, the one
    nearest to k1 = 1.2 and b = 0.75. Slow: all 651 settings."""
    shares = {s: float(sci_first(split, *s).mean()) for s in GRID}
    return max(GRID, key=lambda s: (shares[s], -abs(s[0] - K1) - abs(s[1] - B)))


def scifact_or_stop(button, texts: dict) -> dict:
    """sci_numbers() for the lesson's cells, once SciFact is downloaded. Until then the cell stops with `texts["ask"]`
    and the button; a click downloads the four files, with the progress in the cell."""
    lesson_helpers.pins_or_stop(button, texts, scifact.FILES)
    return sci_numbers()


CLAIM = "641"   # the test claim of the demo: Insomnia can be effectively treated with cognitive behavioral therapy.


# --- the pictures ----------------------------------------------------------------------------------------------------------

_UI, _LABEL, _CELL, _HEAD, _NUMBER = S._UI, S._LABEL, S._CELL, S._HEAD, S._NUMBER
_DATA, _FAINT, _PRED, _LOSS = S._DATA, S._FAINT, S._PRED, S._LOSS
_CHIP, _CHIP_PRED, _CHIP_DATA, _CHIP_LOSS = S._CHIP, S._CHIP_PRED, S._CHIP_DATA, S._CHIP_LOSS
_note, _colors, _label, _num, _pct, _dec = S._note, S._colors, S._label, S._num, S._pct, S._dec
_cell, _table = S._cell, S._table
_fn, _call, _empty, _SLOW, _STOPPED = S._fn, S._call, S._empty, S._SLOW, S._STOPPED
_WHY_EMPTY = S._WHY_EMPTY
_chip, _is_number, _same, _Odd = V._chip, V._is_number, V._same, V._Odd
_GRAY_BAR, _BLUE_BAR, _DASH = V._GRAY_BAR, V._BLUE_BAR, V._DASH
_WORDS = V._WORDS


def _share(x: float, lang: str, color: str, style: str) -> str:
    """A share as a bar from 0 to 100% and its number, on one line."""
    return ('<span style="white-space:nowrap">' + V._bar(x, color, most=1.0, width=5.0)
            + f'<span style="{style}">{html.escape(_pct(x, lang))}</span></span>')


def _title_cell(i: int, lang: str, own: int, score: str = "") -> str:
    """A cell with the title of paragraph i, blue for the own paragraph, its number of words and a score."""
    return V._title_cell(i, lang, own, score)


def _hit_rows(lang: str, columns, w: dict) -> list:
    """A row for all questions, for those whose own paragraph is in the longest group, and for the others: a cell for
    each (outcome, color, style)."""
    rows = []
    for key, k in (("all", "hit"), ("long", "hit_long"), ("rest", "hit_rest")):
        label = w[key].format(n=_num(len(xquad.questions(lang)), lang))
        rows.append("<tr>" + _cell(html.escape(label), _DATA)
                    + "".join(_cell(_share(r[k], lang, color, style)) for r, color, style in columns) + "</tr>")
    return rows


_HITS = {
    "en": {"all": "all {n}", "long": "own paragraph in the longest group", "rest": "the other questions"},
    "nl": {"all": "alle {n}", "long": "eigen alinea in de langste groep", "rest": "de andere vragen"},
}

_IDEA_TEXT = {
    "en": {"title": "One question, two rankings, and what each word adds",
           "question": "The question *{q}* was written about a paragraph of {n} words from the article *{t}*. The first three paragraphs of each ranking:",
           "head": ["Place", "Cosine", "BM25"],
           "words": "What each word of the question adds with BM25, in the first paragraph of the cosine, of {a} words, and in the own paragraph, of {b} words. Each cell gives the count of the word in the paragraph, and what the word adds:",
           "head_words": ["Word", "IDF", "First paragraph of the cosine", "Own paragraph"],
           "score_row": "BM25 score",
           "tf_idf": "TF-IDF score, with the counts as they are",
           "rankings": "All the questions, ranked in six ways. The share of the questions whose own paragraph comes first:",
           "head_rankings": ["Ranking", "All questions", "Own paragraph in the longest group", "Other questions"],
           "tf-idf-score": "TF-IDF score", "cosine": "cosine", "binary": "BM25 with k1 = 0", "heavy": "BM25 with k1 = 5",
           "no-length": "BM25 with b = 0", "bm25": "BM25 with k1 = 1.2 and b = 0.75",
           "gray": "Gray: the paragraphs, the counts of the words, the TF-IDF score, and the other rankings.",
           "blue": "Blue: the own paragraph of the question, what each word adds with BM25, and BM25 with k1 = 1.2 and b = 0.75."},
    "nl": {"title": "Eén vraag, twee rangschikkingen, en wat elk woord toevoegt",
           "question": "De vraag *{q}* is geschreven bij een alinea van {n} woorden van de Wikipedia-pagina *{t}*. De eerste drie alinea's van elke rangschikking:",
           "head": ["Plaats", "Cosine similarity", "BM25"],
           "words": "Wat elk woord van de vraag toevoegt met BM25, in de eerste alinea van de cosine similarity, van {a} woorden, en in de eigen alinea, van {b} woorden. Elke cel geeft hoe vaak het woord in de alinea staat, en wat het woord toevoegt:",
           "head_words": ["Woord", "IDF", "Eerste alinea van de cosine similarity", "Eigen alinea"],
           "score_row": "BM25-score",
           "tf_idf": "TF-IDF-score, met de aantallen zelf",
           "rankings": "Alle vragen, op zes manieren gerangschikt. Het deel van de vragen waarbij de eigen alinea bovenaan komt:",
           "head_rankings": ["Rangschikking", "Alle vragen", "Eigen alinea in de langste groep", "Andere vragen"],
           "tf-idf-score": "TF-IDF-score", "cosine": "cosine similarity", "binary": "BM25 met k1 = 0", "heavy": "BM25 met k1 = 5",
           "no-length": "BM25 met b = 0", "bm25": "BM25 met k1 = 1,2 en b = 0,75",
           "gray": "Grijs: de alinea's, hoe vaak de woorden erin staan, de TF-IDF-score, en de andere rangschikkingen.",
           "blue": "Blauw: de eigen alinea van de vraag, wat elk woord toevoegt met BM25, en BM25 met k1 = 1,2 en b = 0,75."},
}
_RANKINGS = ("sum", "cosine", "binary", "heavy", "no-length", "bm25")


def word_parts(lang: str, k: int, i: int) -> dict:
    """For each word of question k that paragraph i holds, once each: its count there, and what it adds with BM25."""
    para = I._paragraph_words(lang)[i]
    idf = _paragraphs(lang)[0]["idf"]
    vocab = _paragraphs(lang)[0]["vocab"]
    return {w: (c, float(idf[vocab[w]] * term_weight(c, len(para), average(lang), K1, B)))
            for w, c in counts_of(lang, k, i).items()}


def idea_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang] | _HITS[lang]
    k = question_place(EXAMPLE, lang)
    q = xquad.questions(lang)[k]
    own = q.paragraph
    lengths = _paragraphs(lang)[0]["lengths"]
    body = _label(w["question"].format(q=q.question, n=_num(int(lengths[own]), lang), t=xquad.paragraphs(lang)[own].title))
    tops = {how: top(lang, how, k) for how in ("cosine", "bm25")}
    rows = []
    for place in range(3):
        cells = "".join(_title_cell(tops[how][place][0], lang, own, _dec(tops[how][place][1], lang, 3 if how == "cosine" else 2))
                        for how in ("cosine", "bm25"))
        rows.append("<tr>" + _cell(str(place + 1), _FAINT, _NUMBER) + cells + "</tr>")
    body += _table(rows, w["head"], (0,))
    first = tops["cosine"][0][0]
    parts = {i: word_parts(lang, k, i) for i in (first, own)}
    idf = paragraph_idf(lang)
    body += _label(w["words"].format(a=_num(int(lengths[first]), lang), b=_num(int(lengths[own]), lang)))
    rows = []
    for word in sorted(set(parts[first]) | set(parts[own]), key=lambda x: (-idf[x], x)):
        cells = ""
        for i in (first, own):
            c, part = parts[i].get(word, (0, 0.0))
            cells += _cell(f'<span style="{_DATA}">{_num(c, lang)}</span> → <span style="{_PRED}">{html.escape(_dec(part, lang, 2))}</span>'
                           if c else _DASH, "", _NUMBER)
        rows.append("<tr>" + _cell(_chip(word)) + _cell(html.escape(_dec(idf[word], lang, 3)), _DATA, _NUMBER) + cells + "</tr>")
    rows.append("<tr>" + _cell(html.escape(w["score_row"]), _PRED) + _cell("")
                + "".join(_cell(html.escape(_dec(sum(p for _, p in parts[i].values()), lang, 2)), _PRED, _NUMBER) for i in (first, own))
                + "</tr>")
    tf_idf = ranked(lang, "sum")[k]
    rows.append("<tr>" + _cell(html.escape(w["tf_idf"]), _DATA) + _cell("")
                + "".join(_cell(html.escape(_dec(float(tf_idf[i]), lang, 2)), _DATA, _NUMBER) for i in (first, own)) + "</tr>")
    body += _table(rows, w["head_words"], (1, 2, 3))
    r = results(lang)
    rows = []
    for how in _RANKINGS:
        color, style = (_BLUE_BAR, _PRED) if how == "bm25" else (_GRAY_BAR, _DATA)
        rows.append("<tr>" + _cell(html.escape(w["tf-idf-score" if how == "sum" else how]), style)
                    + "".join(_cell(_share(r[how][key], lang, color, style)) for key in ("hit", "hit_long", "hit_rest")) + "</tr>")
    body += _label(w["rankings"]) + _table(rows, w["head_rankings"])
    body += _colors(w, body)
    return yours.figure(w["title"], body, alt=w["title"])


_SAT_TEXT = {
    "en": {"title": "Your saturation for counts from 1 to 20",
           "kind": "a number",
           "label": "What a word adds, without its IDF, for each count and three values of k1:",
           "head": ["Count", "k1 = 0.5", "k1 = 1.2", "k1 = 2.0"],
           "most": "never more than",
           "twenty": "With k1 = 1.2, your saturation gives a word that occurs 20 times {a}, against {b} for once: a raw count would make it 20 times as much.",
           "differs": "Your saturation differs from the course's in {k} of the {n} cells.",
           "gray": "Gray: the counts, what a word can add at most, and the course's saturation where yours differs.",
           "blue": "Blue: your saturation, where it is the course's.",
           "purple": "Purple: your saturation, where it differs from the course's."},
    "nl": {"title": "Je saturation voor aantallen van 1 tot 20",
           "kind": "een getal",
           "label": "Wat een woord toevoegt, zonder de IDF, voor elk aantal en drie waarden van k1:",
           "head": ["Aantal", "k1 = 0,5", "k1 = 1,2", "k1 = 2,0"],
           "most": "nooit meer dan",
           "twenty": "Met k1 = 1,2 geeft je saturation een woord dat 20 keer voorkomt {a}, tegen {b} bij één keer: met het aantal zelf zou het 20 keer zoveel zijn.",
           "differs": "Je saturation verschilt van de saturation van de course in {k} van de {n} cellen.",
           "gray": "Grijs: de aantallen, wat een woord hoogstens kan toevoegen, en de saturation van de course waar de jouwe verschilt.",
           "blue": "Blauw: je saturation, waar deze gelijk is aan de saturation van de course.",
           "purple": "Paars: je saturation, waar deze verschilt van de saturation van de course."},
}
SAT_COUNTS = (1, 2, 3, 5, 10, 20)
SAT_K1 = (0.5, 1.2, 2.0)


def saturation_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _SAT_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")

    def make():
        out = {}
        for tf in SAT_COUNTS:
            for k1 in SAT_K1:
                x = fn(tf, k1)
                if not _is_number(x):
                    raise _Odd
                out[tf, k1] = float(x)
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
    rows, differs = [], 0
    for tf in SAT_COUNTS:
        cells = ""
        for k1 in SAT_K1:
            mine, want = got[tf, k1], saturation(tf, k1)
            same = _same(mine, want)
            differs += not same
            cells += _cell(f'<span style="{_PRED if same else _LOSS}">{html.escape(_dec(mine, lang, 3))}</span>'
                           + ("" if same else f' <span style="{_DATA}">({html.escape(_dec(want, lang, 3))})</span>'), "", _NUMBER)
        rows.append("<tr>" + _cell(_num(tf, lang), _DATA, _NUMBER) + cells + "</tr>")
    rows.append("<tr>" + _cell(html.escape(w["most"]), _DATA)
                + "".join(_cell(html.escape(_dec(k1 + 1, lang, 1)), _DATA, _NUMBER) for k1 in SAT_K1) + "</tr>")
    body = _label(w["label"]) + _table(rows, w["head"], (0, 1, 2, 3))
    body += _note(w["twenty"].format(a=_dec(got[20, 1.2], lang, 3), b=_dec(got[1, 1.2], lang, 3)))
    if differs:
        body += _note(w["differs"].format(k=_num(differs, lang), n=_num(len(got), lang)))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


_TW_TEXT = {
    "en": {"title": "Your term_weight on the 2,067 paragraphs and all the questions",
           "kind": "an array of weights, one for each count it gets",
           "small": "A word in a paragraph of {a}, {b} or {c} words, where the paragraphs have {b} words on average, with k1 = 1.2 and b = 0.75. Each cell is your term weight for that count:",
           "head": ["Paragraph", "Count 1", "Count 2", "Count 3", "Count 5"],
           "words": "{n} words",
           "ranks": "Then each word of a question adds its IDF times your term weight, for every word in every paragraph at once, and the paragraphs are ranked for all the questions. The share of the questions whose own paragraph comes first:",
           "head_ranks": ["Questions", "Your BM25", "Cosine"],
           "course": "The course's BM25 puts the own paragraph first for {a} of the questions.",
           "gray": "Gray: the paragraphs and the questions, and what the cosine gives.",
           "blue": "Blue: your term weights and what your BM25 gives, where they are the course's.",
           "purple": "Purple: your term weights and what your BM25 gives, where they differ from the course's."},
    "nl": {"title": "Je term_weight op de 2.067 alinea's en alle vragen",
           "kind": "een array met waarden, één voor elk aantal dat de functie krijgt",
           "small": "Een woord in een alinea van {a}, {b} of {c} woorden, waar de alinea's gemiddeld {b} woorden hebben, met k1 = 1,2 en b = 0,75. Elke cel is jouw term weight bij dat aantal:",
           "head": ["Alinea", "Aantal 1", "Aantal 2", "Aantal 3", "Aantal 5"],
           "words": "{n} woorden",
           "ranks": "Daarna voegt elk woord van een vraag de IDF maal je term weight toe, voor elk woord in elke alinea tegelijk, en worden de alinea's gerangschikt voor alle vragen. Het deel van de vragen waarbij de eigen alinea bovenaan komt:",
           "head_ranks": ["Vragen", "Jouw BM25", "Cosine similarity"],
           "course": "De BM25 van de course zet de eigen alinea bovenaan bij {a} van de vragen.",
           "gray": "Grijs: de alinea's en de vragen, en wat de cosine similarity geeft.",
           "blue": "Blauw: je term weights en wat je BM25 geeft, waar deze gelijk zijn aan wat de course geeft.",
           "purple": "Paars: je term weights en wat je BM25 geeft, waar deze verschillen van wat de course geeft."},
}
TW_COUNTS = (1, 2, 3, 5)


def _tw_lengths(lang: str) -> tuple:
    """The lengths of the small table under term_weight: short, the average, long."""
    return (60.0, round(average(lang), 1), 250.0)


def _good_array(out, shape) -> bool:
    return (isinstance(out, np.ndarray) and out.shape == shape and np.issubdtype(out.dtype, np.number)
            and not np.issubdtype(out.dtype, np.bool_) and bool(np.isfinite(out).all()))


def term_weight_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _TW_TEXT[lang] | _HITS[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    M, Q = _paragraphs(lang)
    avg = average(lang)
    small_len = _tw_lengths(lang)
    tf_small = np.tile(np.array(TW_COUNTS, dtype=float), len(small_len))
    len_small = np.repeat(np.array(small_len), len(TW_COUNTS))

    def make():
        small = fn(tf_small.copy(), len_small.copy(), avg, K1, B)
        big = fn(M["tf"].copy(), M["lengths"][M["doc"]], avg, K1, B)
        if not (_good_array(small, tf_small.shape) and _good_array(big, M["tf"].shape)):
            return None
        return small.astype(float), big.astype(float)

    got = _call(make)
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    if got is None:
        return _empty(w, "odd")
    small, big = got
    want_small = term_weight(tf_small, len_small, avg, K1, B)
    same_small = np.isclose(small, want_small, rtol=1e-6, atol=1e-9)
    rows = []
    for r, length in enumerate(small_len):
        cells = ""
        for c in range(len(TW_COUNTS)):
            j = r * len(TW_COUNTS) + c
            cells += _cell(html.escape(_dec(small[j], lang, 3)), _PRED if same_small[j] else _LOSS, _NUMBER)
        label = w["words"].format(n=_dec(length, lang, 1) if length != int(length) else _num(length, lang))
        rows.append("<tr>" + _cell(html.escape(label), _DATA) + cells + "</tr>")
    shown = [_dec(x, lang, 1) if x != int(x) else _num(x, lang) for x in small_len]
    body = _label(w["small"].format(a=shown[0], b=shown[1], c=shown[2])) + _table(rows, w["head"], (1, 2, 3, 4))
    mine = outcome(lang, _rank(M, Q["once"], M["idf"][M["word"]] * big))
    want = results(lang)["bm25"]
    same = bool((mine["first"] == want["first"]).all()) and bool(same_small.all())
    style = _PRED if same else _LOSS
    columns = [(mine, _BLUE_BAR if same else "var(--ea-loss)", style), (results(lang)["cosine"], _GRAY_BAR, _DATA)]
    body += _label(w["ranks"]) + _table(_hit_rows(lang, columns, w), w["head_ranks"])
    if not same:
        body += f'<div style="{_LOSS}">' + html.escape(w["course"].format(a=_pct(want["hit"], lang))) + "</div>"
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


def _shared_first(scores: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """For each row, the chance that a wanted column comes first: 1 or 0, or a fraction when columns share the first
    place, as if the tie were broken at random. With k1 = 0, BM25 gives every document that holds the same words of the
    query the same score, so ties are common there."""
    tied = np.isclose(scores, scores.max(axis=1)[:, None], rtol=1e-12, atol=0)
    return (tied & wanted).sum(axis=1) / tied.sum(axis=1)


def outcome(lang: str, scores: np.ndarray) -> dict:
    """V.outcome: what a ranking of the paragraphs does for the questions of XQuAD; a question whose own paragraph
    shares the first place with others counts for its share of the tie."""
    out = V.outcome(lang, scores)
    own = V._scores(lang)["own"]
    wanted = np.zeros(scores.shape, dtype=bool)
    wanted[np.arange(len(own)), own] = True
    hit = _shared_first(scores, wanted)
    long_ = fifths(lang)[own] == 4
    return out | {"hit": float(hit.mean()), "hit_long": float(hit[long_].mean()), "hit_rest": float(hit[~long_].mean())}


_BM_TEXT = {
    "en": {"title": "Your bm25 ranks the 2,067 paragraphs for two questions",
           "kind": "a number",
           "question": "*{q}* The own paragraph has {n} words, from the article *{t}*.",
           "head": ["Place", "Paragraph", "Your bm25"],
           "place": "Your bm25 puts the own paragraph at place {a}, the course's BM25 at place {b}, and the cosine at place {c}.",
           "same": "Your bm25 gives every paragraph the course's score, for both questions.",
           "differs": "Your bm25 differs from the course's for {k} of the {n} paragraphs, over the two questions.",
           "gray": "Gray: the paragraphs that are not the own paragraph of the question.",
           "blue": "Blue: the own paragraph of the question, and your bm25 where it is the course's.",
           "purple": "Purple: your bm25, where it differs from the course's."},
    "nl": {"title": "Je bm25 rangschikt de 2.067 alinea's voor twee vragen",
           "kind": "een getal",
           "question": "*{q}* De eigen alinea heeft {n} woorden, van de Wikipedia-pagina *{t}*.",
           "head": ["Plaats", "Alinea", "Jouw bm25"],
           "place": "Je bm25 zet de eigen alinea op plaats {a}, de BM25 van de course op plaats {b}, en de cosine similarity op plaats {c}.",
           "same": "Je bm25 geeft elke alinea dezelfde score als de BM25 van de course, bij beide vragen.",
           "differs": "Je bm25 verschilt van de BM25 van de course bij {k} van de {n} alinea's, over de twee vragen.",
           "gray": "Grijs: de alinea's die niet de eigen alinea van de vraag zijn.",
           "blue": "Blauw: de eigen alinea van de vraag, en je bm25 waar deze gelijk is aan de BM25 van de course.",
           "purple": "Paars: je bm25, waar deze verschilt van de BM25 van de course."},
}
_SHOWN_PLACES = 3   # paragraphs that the picture under bm25 shows for each question


def bm25_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _BM_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    paras = I._paragraph_words(lang)
    avg = average(lang)
    places = [question_place(qid, lang) for qid in EXAMPLES]
    queries = [distinct(words(xquad.questions(lang)[k].question)) for k in places]

    def make():
        out = []
        for query in queries:
            idf = paragraph_idf(lang)
            row = []
            for para in paras:
                x = fn(list(query), list(para), idf, avg, K1, B)
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
    lengths = _paragraphs(lang)[0]["lengths"]
    for k, mine in zip(places, got):
        q = xquad.questions(lang)[k]
        own = q.paragraph
        want = ranked(lang, "bm25")[k]
        same = np.isclose(mine, want, rtol=1e-6, atol=1e-9)
        differs += int((~same).sum())
        body += _label(w["question"].format(q=q.question, n=_num(int(lengths[own]), lang), t=xquad.paragraphs(lang)[own].title))
        order = np.argsort(-mine, kind="stable")
        rows = []
        for place, i in enumerate(order[:_SHOWN_PLACES], 1):
            rows.append("<tr>" + _cell(str(place), _FAINT, _NUMBER) + _title_cell(int(i), lang, own)
                        + _cell(html.escape(_dec(float(mine[i]), lang, 2)), _PRED if same[i] else _LOSS, _NUMBER) + "</tr>")
        body += _table(rows, w["head"], (0, 2))
        a = 1 + int((mine > mine[own]).sum())
        r = results(lang)
        body += _note(w["place"].format(a=_num(a, lang), b=_num(int(r["bm25"]["place"][k]), lang),
                                        c=_num(int(r["cosine"]["place"][k]), lang)))
    n = len(EXAMPLES) * len(paras)
    body += _note(w["same"] if not differs else w["differs"].format(k=_num(differs, lang), n=_num(n, lang)))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


_TRY_TEXT = {
    "en": {"title": "BM25 on SciFact with k1 = {k} and b = {b}",
           "binary": "With k1 = 0, a word of the claim adds its IDF once, however often it occurs, and b changes nothing.",
           "no-length": "With b = 0, the length of an abstract does not count.",
           "label": "The share of the claims whose relevant abstract comes first, or one of them when a claim has more than one:",
           "head": ["Claims", "Your setting", "k1 = 1.2 and b = 0.75", "Cosine"],
           "train": "training: {n}", "test": "test: {n}",
           "tuned": "Tuned on the training claims, the best of the 651 settings with k1 from 0 to 3 in steps of 0.1 and b from 0 to 1 in steps of 0.05 is k1 = 1.1 and b = 0.65: {a} on the training claims, and {b} on the test claims. The best setting on the test claims themselves, k1 = 0.7 and b = 1.0, gives {c} there.",
           "claim": "One test claim, *{c}*, and the first three abstracts with your setting:",
           "head_claim": ["Place", "Abstract", "Score"],
           "gray": "Gray: the claims, the usual setting, the cosine, and the abstracts that are not relevant to the claim.",
           "blue": "Blue: what BM25 gives with your setting, and the abstracts that are relevant to the claim."},
    "nl": {"title": "BM25 op SciFact met k1 = {k} en b = {b}",
           "binary": "Met k1 = 0 voegt een woord van de bewering één keer de IDF toe, hoe vaak het woord er ook in staat, en verandert b niets.",
           "no-length": "Met b = 0 telt de lengte van een abstract niet mee.",
           "label": "Het deel van de beweringen waarbij het relevante abstract bovenaan komt, of een van de relevante abstracts als een bewering er meer heeft:",
           "head": ["Beweringen", "Jouw instelling", "k1 = 1,2 en b = 0,75", "Cosine similarity"],
           "train": "training: {n}", "test": "test: {n}",
           "tuned": "Afgesteld op de trainingsbeweringen is de beste van de 651 instellingen met k1 van 0 tot 3 in stappen van 0,1 en b van 0 tot 1 in stappen van 0,05: k1 = 1,1 en b = 0,65, met {a} op de trainingsbeweringen en {b} op de testbeweringen. De beste instelling op de testbeweringen zelf, k1 = 0,7 en b = 1,0, geeft daar {c}.",
           "claim": "Eén testbewering, *{c}*, en de eerste drie abstracts met jouw instelling:",
           "head_claim": ["Plaats", "Abstract", "Score"],
           "gray": "Grijs: de beweringen, de gebruikelijke instelling, de cosine similarity, en de abstracts die niet relevant zijn voor de bewering.",
           "blue": "Blauw: wat BM25 met jouw instelling geeft, en de abstracts die relevant zijn voor de bewering."},
}
TRY_START = (K1, B)   # the sliders start at the usual settings


def _setting(value, low: float, high: float, step: float, start: float) -> float:
    """A slider's value, inside its range and on its steps; the start when it is not a number."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return start
    if not math.isfinite(x):
        return start
    return round(round(min(high, max(low, x)) / step) * step, 2)


def tuned_numbers() -> dict:
    """The shares of the demo that do not depend on the sliders: the usual setting, the cosine, the setting tuned on the
    training claims and the one that is best on the test claims."""
    def make():
        return {name: {split: float(f(split).mean()) for split in scifact.SPLITS}
                for name, f in (("usual", lambda s: sci_first(s, *TRY_START)), ("cosine", sci_cosine),
                                ("tuned", lambda s: sci_first(s, *TUNED)), ("test", lambda s: sci_first(s, *BEST_ON_TEST)))}
    return kept("ir-bm25:scifact:tuned", make)


def try_picture(lang: str, k1=K1, b=B) -> str:
    lang = lang if lang in LANGS else "en"
    w = _TRY_TEXT[lang]
    k1 = _setting(k1, 0.0, 3.0, 0.1, K1)
    b = _setting(b, 0.0, 1.0, 0.05, B)
    t = tuned_numbers()
    sc = _sci()
    body = ""
    if k1 == 0:
        body += _note(w["binary"])
    elif b == 0:
        body += _note(w["no-length"])
    rows = []
    for split in scifact.SPLITS:
        mine = float(sci_first(split, k1, b).mean())
        rows.append("<tr>" + _cell(html.escape(w[split].format(n=_num(sc[split]["once"].shape[0], lang))), _DATA)
                    + _cell(_share(mine, lang, _BLUE_BAR, _PRED)) + _cell(_share(t["usual"][split], lang, _GRAY_BAR, _DATA))
                    + _cell(_share(t["cosine"][split], lang, _GRAY_BAR, _DATA)) + "</tr>")
    body += _label(w["label"]) + _table(rows, w["head"])
    body += _note(w["tuned"].format(a=_pct(t["tuned"]["train"], lang), b=_pct(t["tuned"]["test"], lang),
                                    c=_pct(t["test"]["test"], lang)))
    k = sc["test"]["ids"].index(CLAIM)
    claim = scifact.claims("test")[k]
    one = bm25_scores(sc["M"], {"once": sc["test"]["once"][k]}, k1, b)[0]
    abstracts = scifact.abstracts()
    rows = []
    for place, i in enumerate(np.argsort(-one, kind="stable")[:3], 1):
        relevant = int(i) in sc["test"]["relevant"][k]
        said = f'{_num(int(sc["M"]["lengths"][i]), lang)} {_WORDS[lang]}'
        rows.append("<tr>" + _cell(str(place), _FAINT, _NUMBER)
                    + _cell(_chip(abstracts[int(i)]["title"], _CHIP_PRED if relevant else _CHIP_DATA)
                            + f'<span style="{_UI};margin-left:0.3rem">{html.escape(said)}</span>')
                    + _cell(html.escape(_dec(float(one[i]), lang, 2)), _PRED, _NUMBER) + "</tr>")
    body += _label(w["claim"].format(c=claim["text"].rstrip("."))) + _table(rows, w["head_claim"], (0, 2))
    title = w["title"].format(k=_dec(k1, lang, 1), b=_dec(b, lang, 2))
    return yours.figure(title, body + _colors(w, body), alt=title)


# --- the exercises ---------------------------------------------------------------------------------------------------------

_Val = S._Val
_DOTS = S._DOTS
_SAME_TEXTS = S._SAME_TEXTS
_value, _show, _written = V._value, V._show, V._written

_SA_HEAD = "def saturation(tf, k1):\n"
_SA = _SA_HEAD + "    return tf * (k1 + 1) / (tf + k1)\n"
_SA_START = _SA_HEAD + "    return ...\n"
_SA_RAW = _SA_HEAD + "    return tf\n"
_SA_NO_TOP = _SA_HEAD + "    return tf / (tf + k1)\n"
_SA_NO_TF = _SA_HEAD + "    return tf * (k1 + 1) / k1\n"
_SA_ORDER = _SA_HEAD + "    return tf * (k1 + 1) / tf + k1\n"

_TW_HEAD = "def term_weight(tf, length, average, k1, b):\n"
_TW_END = "    return tf * (k1 + 1) / (tf + k1 * scale)\n"
_TW = _TW_HEAD + "    scale = 1 - b + b * length / average\n" + _TW_END
_TW_START = _TW_HEAD + "    scale = ...\n" + _TW_END
_TW_SWAP = _TW_HEAD + "    scale = 1 - b + b * average / length\n" + _TW_END
_TW_WHOLE = _TW_HEAD + "    scale = 1 - b + length / average\n" + _TW_END
_TW_NO_REST = _TW_HEAD + "    scale = b * length / average\n" + _TW_END
_TW_NO_LENGTH = _TW_HEAD + "    scale = 1 - b\n" + _TW_END

_BM_HEAD = "def bm25(query, doc, idf, average, k1, b):\n    score = 0\n    for word in query:\n        if word in doc:\n"
_BM_END = "    return score\n"
_BM = _BM_HEAD + "            score = score + idf[word] * term_weight(doc.count(word), len(doc), average, k1, b)\n" + _BM_END
_BM_START = _BM_HEAD + "            score = ...\n" + _BM_END
_BM_RAW = _BM_HEAD + "            score = score + idf[word] * doc.count(word)\n" + _BM_END
_BM_NO_IDF = _BM_HEAD + "            score = score + term_weight(doc.count(word), len(doc), average, k1, b)\n" + _BM_END
_BM_LAST = _BM_HEAD + "            score = idf[word] * term_weight(doc.count(word), len(doc), average, k1, b)\n" + _BM_END
_BM_ONCE = _BM_HEAD + "            score = score + idf[word] * term_weight(1, len(doc), average, k1, b)\n" + _BM_END

# The small cases the exercise texts show; the same in both lessons.
TOY_SATURATION = ((1, 1.2), (3, 2.0))
TOY_TERM_WEIGHT = ((2, 20, 20.0, 2.0, 0.75), (2, 60, 20.0, 2.0, 0.5))
TOY_BM25 = ((["red", "wine"], ["red", "wine", "wine", "cheese"], {"red": 0.5, "wine": 1.0, "cheese": 2.0}, 4.0, 1.2, 0.75),
            (["python", "snake"], ["python", "tank"], {"python": 1.0, "snake": 2.0, "tank": 1.5}, 2.0, 1.2, 0.75))


def _small_case(lang: str, i: int) -> tuple:
    """bm25 for the query red wine heart and document i of the course, with the IDF and the average of the thirteen."""
    return (_Val(f'C.QUERIES["{lang}"]["wine"]', QUERIES[lang]["wine"]), _Val(f'C.DOCS["{lang}"][{i}]', DOCS[lang][i]),
            _Val(f'C.SMALL_IDF["{lang}"]', SMALL_IDF[lang]), _Val(f'C.SMALL_AVERAGE["{lang}"]', SMALL_AVERAGE[lang]),
            _written(1.2), _written(0.75))


REQUIRED = ["saturation", "term_weight", "bm25", "settings"]

EXERCISES = {
    "saturation": FunctionBox(
        name="saturation",
        picture=saturation_picture,
        function="saturation",
        start=_SA_START,
        solution=_SA,
        show=_show,
        visible=[tuple(_written(v) for v in case) for case in TOY_SATURATION],
        hidden=[tuple(_written(v) for v in case) for case in ((2, 0.5), (10, 2.0), (20, 1.2), (5, 0.0), (1, 0.0), (0, 1.2))]
        + [(_written(np.array([1.0, 2.0, 4.0])), _written(2.0))],
        first=[_DOTS],
        mistakes=[
            Mistake("raw", wrong=_SA_RAW,
                    en="`{call}` gives {got}, and it should give {want}: your function gives the count itself, so every repeat adds as much as the first. What does BM25 divide by, so that the repeats add less and less?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie geeft het aantal zelf, dus elke herhaling voegt evenveel toe als de eerste keer. Waardoor deelt BM25, zodat de herhalingen steeds minder toevoegen?"),
            Mistake("no-top", wrong=_SA_NO_TOP,
                    en="`{call}` gives {got}, and it should give {want}: without `k1 + 1` above the line, a word that occurs once adds less than 1. What should a count of 1 give?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: zonder `k1 + 1` boven de streep voegt een woord dat één keer voorkomt minder dan 1 toe. Wat moet een aantal van 1 geven?"),
            Mistake("no-tf", wrong=_SA_NO_TF,
                    en="`{call}` gives {got}, and it should give {want}: your function divides by `k1` alone, so the weight grows as fast as the count. Which number grows with the count below the line too?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie deelt alleen door `k1`, dus de waarde groeit even hard als het aantal. Welk getal groeit onder de streep mee met het aantal?"),
            Mistake("order", wrong=_SA_ORDER,
                    en="`{call}` gives {got}, and it should give {want}: Python divides before it adds, so your function divides by `tf` and then adds `k1`. How do parentheses make it divide by both together?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: Python deelt voordat het optelt, dus je functie deelt door `tf` en telt daarna `k1` op. Hoe laat je de functie met haakjes door beide samen delen?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the saturation: the count times `k1 + 1`, divided by the count plus `k1`.",
                      "nl": "Vervang de `...` door de verzadiging: het aantal maal `k1 + 1`, gedeeld door het aantal plus `k1`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function multiply the count by `k1 + 1`, and divide by the count plus `k1`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Vermenigvuldigt je functie het aantal met `k1 + 1`, en deelt de functie door het aantal plus `k1`?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function also work for k1 = 0, a count of 0, and a NumPy array of counts?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook voor k1 = 0, een aantal van 0, en een NumPy-array met aantallen?"},
            "correct": {"en": "Right: with your function, each repeat of a word adds less than the one before. In the picture below, it runs for counts from 1 to 20: how close does a word that occurs 20 times come to k1 + 1?",
                        "nl": "Goed: met je functie voegt elke herhaling van een woord minder toe dan de vorige. In het plaatje hieronder rekent de functie aantallen van 1 tot 20 uit: hoe dicht komt een woord dat 20 keer voorkomt bij k1 + 1?"},
        },
    ),
    "term_weight": FunctionBox(
        name="term_weight",
        picture=term_weight_picture,
        function="term_weight",
        start=_TW_START,
        solution=_TW,
        show=_show,
        visible=[tuple(_written(v) for v in case) for case in TOY_TERM_WEIGHT],
        hidden=[tuple(_written(v) for v in case) for case in ((1, 10, 10.0, 1.2, 0.75), (3, 40, 20.0, 1.2, 0.0),
                                                               (3, 40, 20.0, 1.2, 1.0), (2, 10, 20.0, 2.0, 0.75),
                                                               (5, 30, 20.0, 0.0, 0.75))]
        + [tuple(_written(v) for v in (np.array([1.0, 3.0]), np.array([10.0, 40.0]), 20.0, 1.2, 0.75))],
        first=[_DOTS],
        mistakes=[
            Mistake("swap", wrong=_TW_SWAP,
                    en="`{call}` gives {got}, and it should give {want}: your scale divides the average by the length, the other way around, so a long document gets a small scale. Should a word in a long document need more repeats, or fewer?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je schaal deelt het gemiddelde door de lengte, net andersom, dus een lang document krijgt een kleine schaal. Moet een woord in een lang document vaker voorkomen, of minder vaak?"),
            Mistake("whole", wrong=_TW_WHOLE,
                    en="`{call}` gives {got}, and it should give {want}: your scale adds the whole length against the average, so for a document of average length it is more than 1. What should the length against the average be multiplied by?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je schaal telt de hele lengte ten opzichte van het gemiddelde op, dus voor een document van gemiddelde lengte is de schaal meer dan 1. Waarmee moet de lengte ten opzichte van het gemiddelde vermenigvuldigd worden?"),
            Mistake("no-rest", wrong=_TW_NO_REST,
                    en="`{call}` gives {got}, and it should give {want}: your scale leaves out the part `1 - b`, so with b = 0 the length scales nothing and the scale becomes 0. What should the scale be for a document of average length, whatever b is?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je schaal laat het deel `1 - b` weg, dus bij b = 0 wordt de schaal 0. Wat moet de schaal zijn voor een document van gemiddelde lengte, wat b ook is?"),
            Mistake("no-length", wrong=_TW_NO_LENGTH,
                    en="`{call}` gives {got}, and it should give {want}: your scale does not use the length of the document. Where do `length` and `average` come in?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je schaal gebruikt de lengte van het document niet. Waar komen `length` en `average` erbij?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the scale: the share `1 - b`, plus the share `b` of the length against the average.",
                      "nl": "Vervang de `...` door de schaal: het deel `1 - b`, plus het deel `b` van de lengte ten opzichte van het gemiddelde."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is your scale `1 - b` plus `b` times the length divided by the average?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Is je schaal `1 - b` plus `b` maal de lengte gedeeld door het gemiddelde?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function also work for b = 0, b = 1, k1 = 0, a short document, and NumPy arrays?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook voor b = 0, b = 1, k1 = 0, een kort document, en NumPy-arrays?"},
            "correct": {"en": "Right: your function scales the saturation by the length of the document. In the picture below, it gives the weight of every word in every paragraph at once, and ranks the paragraphs for all the questions: how often does the own paragraph come first?",
                        "nl": "Goed: je functie schaalt de verzadiging met de lengte van het document. In het plaatje hieronder geeft de functie de waarde van elk woord in elke alinea tegelijk, en rangschikt de alinea's voor alle vragen: hoe vaak komt de eigen alinea bovenaan?"},
        },
    ),
    "bm25": FunctionBox(
        name="bm25",
        picture=bm25_picture,
        function="bm25",
        start=_BM_START,
        solution=_BM,
        show=_show,
        names={"term_weight": term_weight},
        visible=[tuple(_written(v) for v in case) for case in TOY_BM25],
        hidden=[tuple(_written(v) for v in case) for case in (
            ([], ["red"], {"red": 1.0}, 1.0, 1.2, 0.75),
            (["red"], ["red", "red", "red"], {"red": 0.5}, 3.0, 0.0, 0.75),
            (["wine"], ["wine", "wine", "x", "x", "x", "x", "x", "x"], {"wine": 1.0, "x": 0.1}, 4.0, 1.2, 0.0),
            (["wine"], ["wine", "wine", "x", "x", "x", "x", "x", "x"], {"wine": 1.0, "x": 0.1}, 4.0, 1.2, 1.0),
            (["the", "heart"], ["the", "heart", "the"], {"the": 0.0, "heart": 0.7}, 3.0, 1.2, 0.75))]
        + [_small_case("en", 6), _small_case("en", 9), _small_case("nl", 6)],
        first=[_DOTS],
        mistakes=[
            Mistake("raw", wrong=_BM_RAW,
                    en="`{call}` gives {got}, and it should give {want}: your function multiplies the IDF by the count itself, as the TF-IDF score does. Which function of exercise 2 turns the count into a weight that saturates?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie vermenigvuldigt de IDF met het aantal zelf, zoals de TF-IDF-score doet. Welke functie van oefening 2 maakt van het aantal een waarde die verzadigt?"),
            Mistake("no-idf", wrong=_BM_NO_IDF,
                    en="`{call}` gives {got}, and it should give {want}: your function adds the term weight alone, so a common word counts as much as a rare one. What should the term weight be multiplied by?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie telt alleen de term weight op, dus een veelvoorkomend woord telt even zwaar als een zeldzaam woord. Waarmee moet de term weight vermenigvuldigd worden?"),
            Mistake("last", wrong=_BM_LAST,
                    en="`{call}` gives {got}, and it should give {want}: each round throws away `score` and keeps only the part of the last word. What should the new `score` be made of?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke ronde gooit `score` weg en bewaart alleen het deel van het laatste woord. Waaruit bestaat de nieuwe `score`?"),
            Mistake("once", wrong=_BM_ONCE,
                    en="`{call}` gives {got}, and it should give {want}: your function gives `term_weight` the count 1 for every word, so a word that occurs twice adds as much as once. How do you get the count of `word` in `doc`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie geeft `term_weight` voor elk woord het aantal 1, dus een woord dat twee keer voorkomt voegt evenveel toe als één keer. Hoe krijg je het aantal keer dat `word` in `doc` staat?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the new `score`: the score so far, plus the IDF of the word times its term weight in this document.",
                      "nl": "Vervang de `...` door de nieuwe `score`: de score tot nu toe, plus de IDF van het woord maal de term weight in dit document."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does each word add its IDF times `term_weight` of its count, the length of the document, the average, `k1` and `b`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Voegt elk woord de IDF toe maal `term_weight` van het aantal, de lengte van het document, het gemiddelde, `k1` en `b`?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function also work for an empty query, k1 = 0, b = 0, b = 1, a word with IDF 0, and the course's documents?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook voor een lege zoekvraag, k1 = 0, b = 0, b = 1, een woord met IDF 0, en de documenten van de course?"},
            "correct": {"en": "Right: your function gives the BM25 score of a document. In the picture below, it ranks all 2,067 paragraphs for two questions: where does it put the own paragraph, and where does the cosine?",
                        "nl": "Goed: je functie geeft de BM25-score van een document. In het plaatje hieronder rangschikt de functie alle 2.067 alinea's voor twee vragen: waar zet de functie de eigen alinea, en waar de cosine similarity?"},
        },
    ),
}

# The sort: what the setting k1 does, and what b does.
EXERCISES["settings"] = Sort(
    name="settings",
    groups={
        "k1": {"en": "The setting k1", "nl": "De instelling k1"},
        "b": {"en": "The setting b", "nl": "De instelling b"},
    },
    items={
        "binary": {"en": "set to 0, a word adds the same whether it occurs once or ten times",
                   "nl": "op 0 voegt een woord evenveel toe, of het er één keer of tien keer in staat"},
        "most": {"en": "it sets the most that one word can add, however often it occurs",
                 "nl": "bepaalt het meeste dat één woord kan toevoegen, hoe vaak het er ook in staat"},
        "high": {"en": "set very high, a word that occurs ten times counts almost ten times as much as once",
                 "nl": "heel hoog gezet, telt een woord dat tien keer voorkomt bijna tien keer zo zwaar als één keer"},
        "speed": {"en": "in a document of average length, it sets how fast the next repeat of a word stops adding much",
                  "nl": "bepaalt in een document van gemiddelde lengte hoe snel een volgende herhaling van een woord weinig meer toevoegt"},
        "verbose": {"en": "it keeps a document that says the same in more words from winning by its length",
                    "nl": "zorgt dat een document dat hetzelfde in meer woorden zegt, niet wint door de lengte"},
        "off": {"en": "set to 0, the length of a document does not count at all",
                "nl": "op 0 telt de lengte van een document helemaal niet mee"},
        "repeats": {"en": "it makes a word in a long document need more repeats to reach the same weight",
                    "nl": "zorgt dat een woord in een lang document vaker moet voorkomen voor dezelfde waarde"},
        "full": {"en": "set to 1, the count is scaled fully by the length of the document against the average",
                 "nl": "op 1 wordt het aantal helemaal geschaald met de lengte van het document ten opzichte van het gemiddelde"},
    },
    solution={"binary": "k1", "most": "k1", "high": "k1", "speed": "k1",
              "verbose": "b", "off": "b", "repeats": "b", "full": "b"},
    nudge={
        "binary": {"en": "Which setting is in the formula of the saturation, and what is left of the formula when it is 0?",
                   "nl": "Welke instelling staat bij het aantal in de formule van de verzadiging, en wat blijft er over als deze instelling 0 is?"},
        "most": {"en": "However often a word occurs, its weight stays below a number. Which setting is in that number?",
                 "nl": "Hoe vaak een woord ook voorkomt, de waarde blijft onder een getal. Welke instelling zit in dat getal?"},
        "high": {"en": "Which setting decides how strongly the repeats saturate?",
                 "nl": "Welke instelling bepaalt hoe sterk de herhalingen verzadigen?"},
        "speed": {"en": "*The idea* says: the lower this setting, the stronger the saturation. Which setting is it?",
                  "nl": "*Het idee* zegt: hoe lager deze instelling, hoe sterker de verzadiging. Welke instelling is dat?"},
        "verbose": {"en": "Which setting decides how much the length of a document counts?",
                    "nl": "Welke instelling bepaalt hoeveel de lengte van een document meetelt?"},
        "off": {"en": "In the scale, which setting multiplies the length against the average?",
                "nl": "Welke instelling staat in de schaal bij de lengte ten opzichte van het gemiddelde?"},
        "repeats": {"en": "A long document gets a larger scale. Which setting decides how much larger?",
                    "nl": "Een lang document krijgt een grotere schaal. Welke instelling bepaalt hoeveel groter?"},
        "full": {"en": "At 1, the scale is the length divided by the average. Which setting is that?",
                 "nl": "Bij 1 is de schaal de lengte gedeeld door het gemiddelde. Welke instelling is dat?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the groups again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de groepen opnieuw uit."},
        "empty": {"en": "Sort each statement into the group that fits.",
                  "nl": "Deel elke uitspraak in bij de groep die klopt."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "{item} belongs with another group.",
                  "nl": "{item} hoort bij een andere groep."},
        "correct": {"en": "All {n} are right. k1 decides how much the repeats of a word count: at 0 not at all, and the higher, the more, up to k1 + 1 times the IDF. b decides how much the length of a document counts against the average: at 0 not at all, at 1 fully.",
                    "nl": "Alle {n} kloppen. k1 bepaalt hoeveel de herhalingen van een woord meetellen: bij 0 helemaal niet, en hoe hoger, hoe meer, tot k1 + 1 keer de IDF. b bepaalt hoeveel de lengte van een document meetelt ten opzichte van het gemiddelde: bij 0 helemaal niet, bij 1 helemaal."},
    },
)

# What the tests feed each check, and what it must say: (exercise, code or choices, state, known mistake).
CASES = [
    ("saturation", _SA_START, "empty", None),
    ("saturation", _SA, "correct", None),
    ("saturation", _SA_HEAD + "    return tf * (k1 + 1) / (...)\n", "wrong", "dots-left"),
    ("saturation", _SA_RAW, "wrong", "raw"),
    ("saturation", _SA_NO_TOP, "wrong", "no-top"),
    ("saturation", _SA_NO_TF, "wrong", "no-tf"),
    ("saturation", _SA_ORDER, "wrong", "order"),
    ("saturation", _SA_HEAD + "    return (k1 + 1) * tf / (k1 + tf)\n", "correct", None),
    ("saturation", _SA_HEAD + "    return (k1 + 1) / (1 + k1 / tf)\n", "error", None),
    ("term_weight", _TW_START, "empty", None),
    ("term_weight", _TW, "correct", None),
    ("term_weight", _TW_HEAD + "    scale = 1 - b + ...\n" + _TW_END, "wrong", "dots-left"),
    ("term_weight", _TW_SWAP, "wrong", "swap"),
    ("term_weight", _TW_WHOLE, "wrong", "whole"),
    ("term_weight", _TW_NO_REST, "wrong", "no-rest"),
    ("term_weight", _TW_NO_LENGTH, "wrong", "no-length"),
    ("term_weight", _TW_HEAD + "    scale = (1 - b) + b * (length / average)\n" + _TW_END, "correct", None),
    ("bm25", _BM_START, "empty", None),
    ("bm25", _BM, "correct", None),
    ("bm25", _BM_HEAD + "            score = score + idf[word] * ...\n" + _BM_END, "wrong", "dots-left"),
    ("bm25", _BM_RAW, "wrong", "raw"),
    ("bm25", _BM_NO_IDF, "wrong", "no-idf"),
    ("bm25", _BM_LAST, "wrong", "last"),
    ("bm25", _BM_ONCE, "wrong", "once"),
    ("bm25", _BM_HEAD + "            tf = doc.count(word)\n            score += idf[word] * term_weight(tf, len(doc), average, k1, b)\n"
     + _BM_END, "correct", None),
    ("settings", {}, "empty", None),
    ("settings", dict(EXERCISES["settings"].solution), "correct", None),
    ("settings", {"binary": "k1", "off": "b"}, "wrong", "unfinished"),
] + [("settings", {item: ("b" if group == "k1" else "k1")}, "wrong", f"wrong:{item}")
     for item, group in EXERCISES["settings"].solution.items()]
