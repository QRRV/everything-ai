# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Paragraphs of Wikipedia with questions about them, in English and in Dutch, for the lessons of domain taal about
topics, questions and search: the 2,067 paragraphs of the 48 articles of the development set of SQuAD 1.1 (Rajpurkar,
Zhang, Lopyrev and Liang, EMNLP 2016; CC-BY-SA-4.0) and their Dutch in SQuAD-NL v1.1 (de Vries, Wieling and Nissim,
EMNLP 2023; CC-BY-SA-4.0), with the questions of the 240 paragraphs of XQuAD (Artetxe, Ruder and Yogatama, ACL 2020;
CC-BY-SA-4.0), five per article: 1,190 in English and 1,183 in Dutch. The Dutch of those 240 paragraphs and their
questions was corrected by people; the other paragraphs are machine translation (data/xquad-squad-nl.json.gz, made by
data/tools/make_xquad_nl.py).

Each answer is a stretch of its paragraph: its text and the place of its first character. `counts(lang)` gives the
document-term matrix of the paragraphs that the topic models of lesson nlp-topic-models start from."""
import collections
import functools
import gzip
import json
import re
from typing import NamedTuple

from everything_ai import books, paths

FILE = paths.ROOT / "data" / "xquad-squad-nl.json.gz"
LANGS = ("en", "nl")
_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?")
LEAST = 5        # counts() keeps a word that is in at least this many paragraphs
MOST = 0.10      # and in at most this share of them


class Paragraph(NamedTuple):
    article: int        # the place of its article, 0 to 47, in the order of SQuAD
    title: str          # the article's name in that language, such as "Warsaw" or "Warschau"
    xquad: bool         # True for the 240 paragraphs of XQuAD, which have questions and Dutch corrected by people
    shown: bool         # False for a paragraph that a lesson does not show as an example (see make_xquad_nl.py)
    text: str


class Question(NamedTuple):
    paragraph: int      # the place of its paragraph in paragraphs(lang), 0 to 2,066
    id: str             # the id of SQuAD, the same in both languages
    question: str
    answers: tuple      # (text, start) for each answer, with text == paragraph[start:start + len(text)]


def words(text: str) -> list:
    """The words of a text in lower case, without punctuation: "Warsaw's river" gives ["warsaw's", "river"]."""
    return _WORD.findall(text.lower())


@functools.cache
def _data() -> dict:
    return json.loads(gzip.decompress(FILE.read_bytes()))


@functools.cache
def titles(lang: str) -> tuple:
    """The names of the 48 articles in English (en) or Dutch (nl), in the order of SQuAD."""
    assert lang in LANGS, lang
    return tuple(a[lang] for a in _data()["articles"])


@functools.cache
def paragraphs(lang: str) -> tuple:
    """The 2,067 paragraphs in English (en) or Dutch (nl), article by article, in the order of SQuAD."""
    assert lang in LANGS, lang
    return tuple(Paragraph(k, a[lang], bool(p["xquad"]), bool(p["shown"]), p[lang])
                 for k, a in enumerate(_data()["articles"]) for p in a["paragraphs"])


@functools.cache
def questions(lang: str) -> tuple:
    """The questions of the 240 paragraphs of XQuAD in English (1,190) or Dutch (1,183), each with its paragraph and
    answers (most English questions have three, by different people), in the order of SQuAD."""
    assert lang in LANGS, lang
    out, i = [], 0
    for a in _data()["articles"]:
        for p in a["paragraphs"]:
            for q in p["questions"]:
                if q[lang] is not None:
                    out.append(Question(i, q["id"], q[lang], tuple((t, s) for t, s in q[f"{lang}_answers"])))
            i += 1
    return tuple(out)


def _counted(lang: str) -> list:
    """The words of each paragraph that a topic model counts: letters only, more than two of them, not a stop word."""
    stop = books.stop_words(lang)
    return [[w for w in words(p.text) if w.isalpha() and len(w) > 2 and w not in stop] for p in paragraphs(lang)]


@functools.cache
def _counts(lang: str) -> tuple:
    import numpy as np
    import scipy.sparse as sp

    found = _counted(lang)
    df = collections.Counter(w for ws in found for w in set(ws))
    vocab = tuple(sorted(w for w, c in df.items() if LEAST <= c <= MOST * len(found)))
    col = {w: j for j, w in enumerate(vocab)}
    rows, cols, vals = [], [], []
    for i, ws in enumerate(found):
        for w, c in collections.Counter(w for w in ws if w in col).items():
            rows.append(i)
            cols.append(col[w])
            vals.append(c)
    matrix = sp.csr_matrix((np.array(vals, dtype=float), (rows, cols)), shape=(len(found), len(vocab)))
    return matrix, vocab


def counts(lang: str) -> tuple:
    """The document-term matrix of the 2,067 paragraphs in English (en) or Dutch (nl) for a topic model, and its words:
    a SciPy sparse matrix (CSR) with one row per paragraph and one column per word, each value how often the word is in
    the paragraph, and the words of the columns in alphabetical order. It counts the words of letters only (no digits,
    no apostrophe) of more than two letters that are not on the stop list of the language (books.stop_words), and keeps
    those in at least LEAST paragraphs and in at most MOST of them: 4,799 words in English, 4,761 in Dutch. The matrix
    is a new copy each time."""
    assert lang in LANGS, lang
    matrix, vocab = _counts(lang)
    return matrix.copy(), vocab
