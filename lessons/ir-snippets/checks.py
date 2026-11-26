# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ir-snippets: the short text a result page shows under each result. The words of a paragraph are
cut at the spaces, so each keeps its punctuation for showing; a word counts as a query word when its small letters,
without the punctuation at its ends (plain), are among the words of the query without stop words (hits). The snippet
is the window of a fixed number of words with the most query words, the first such window on a tie (best_start), shown
with the query words in bold and with ... where it cuts the text (snippet), the keyword-in-context snippets of Manning,
Raghavan and Schütze, Introduction to Information Retrieval (2008), section 8.7. Next to it, the lesson shows the first
words of the paragraph, a static summary, and the sentence with the highest query score of Tombros and Sanderson,
Advantages of query biased summaries in information retrieval (SIGIR 1998): the square of the number of different
query words in the sentence divided by the number of query words. The sort asks which summaries are static and which
dynamic.

The paragraphs and questions are those of the lessons ir-inverted-index, ir-vector-space and ir-bm25 (the 2,067
paragraphs and the 1,190 English and 1,183 Dutch questions of XQuAD, everything_ai.xquad), ranked by BM25 as in ir-bm25.
A snippet shows the answer when one of the answers of XQuAD lies inside it. Sentences end as in the lesson
nlp-tokenize: a word ending in . ? or ! that is neither an abbreviation of the list of its language nor an initial.
Every number in the text and the pictures is computed by the course (tests/test_ir_snippets.py computes them again)."""
import functools
import html
import importlib.util
import re

import numpy as np

from everything_ai import books, paths, xquad, yours
from everything_ai.check import FunctionBox, Mistake, Sort, kept

LANGS = ("en", "nl")


def _load(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _load("ir-bm25", "everything_ai_checks_ir_bm25_for_snippets")                  # BM25 on the paragraphs of XQuAD
T = _load("nlp-tokenize", "everything_ai_checks_nlp_tokenize_for_snippets")        # where a sentence ends
V, S = B.V, B.S         # the vector space model and the parts of the pictures and the exercises

SIZE = 20               # the length of a snippet in words, in the pictures and the measures
TOP = 3                 # the results a result page of a picture shows
EXAMPLE = "570d2f5bfed7b91900d45cd3"   # What document formed the Parliament of Victoria? (ir-vector-space, ir-bm25)
KINDS = ("first", "window", "sentence")   # the first words, the window with the most query words, the best sentence


# --- the course's solutions, the same code as the exercises -------------------------------------------------------------------

_INSIDE = re.compile(r"[^\W_](?:.*[^\W_])?")   # from the first letter or digit to the last


@functools.cache
def plain(word):
    """A word cut at the spaces, in small letters and without the punctuation at its ends: "Broncos," gives "broncos",
    "(NFC)" gives "nfc", and a word without letters or digits gives ""."""
    found = _INSIDE.search(str(word).lower())
    return found.group() if found else ""


def hits(words, query):
    """How many of the words are query words: words whose plain form is in the set `query`."""
    count = 0
    for word in words:
        if plain(word) in query:
            count = count + 1
    return count


def best_start(words, query, size):
    """Where the window of `size` words with the most query words starts; the first such window on a tie, and 0 when no
    window holds a query word or the words are fewer than `size`."""
    best, most = 0, 0
    for start in range(len(words) - size + 1):
        count = hits(words[start:start + size], query)
        if count > most:
            best, most = start, count
    return best


def snippet(words, query, start, size):
    """The window of `size` words from `start` as one text, each query word between two pairs of stars, as Markdown
    writes bold, and with … where the window cuts the text."""
    shown = []
    for word in words[start:start + size]:
        if plain(word) in query:
            word = "**" + word + "**"
        shown.append(word)
    text = " ".join(shown)
    if start > 0:
        text = "… " + text
    if start + size < len(words):
        text = text + " …"
    return text


# --- words, queries and sentences ----------------------------------------------------------------------------------------

_PIECE = re.compile(r"\S+")


def pieces(text: str) -> list:
    """A text cut at the spaces, as `re.findall(r"\\S+", text)` cuts it: each word keeps its punctuation."""
    return _PIECE.findall(text)


def query_words(question: str, lang: str) -> set:
    """The words of a question in small letters, as everything_ai.xquad cuts them, without the stop words of Snowball."""
    stop = books.stop_words(lang)
    return {w for w in xquad.words(question) if w not in stop}


# The course's own short lists of abbreviations, in small letters: a period after one of these does not end a sentence.
ABBREVIATIONS = {
    "en": frozenset({"a.d.", "approx.", "b.c.", "c.", "ca.", "co.", "corp.", "dr.", "e.g.", "etc.", "ft.", "gen.", "gov.",
                     "i.e.", "inc.", "jr.", "ltd.", "mr.", "mrs.", "ms.", "mt.", "no.", "prof.", "rev.", "sen.", "sr.",
                     "st.", "u.k.", "u.s.", "vs."}),
    "nl": frozenset(T.ABBREVIATIONS | {"n.chr.", "v.chr.", "o.a.", "m.n.", "bijv.", "ca."}),
}


def sentence_spans(words, lang: str) -> list:
    """The sentences of a paragraph cut at the spaces, as (start, end) of its words: a sentence ends after a word that
    ends one by the rule of the lesson nlp-tokenize, with the list of abbreviations of the language."""
    spans, start = [], 0
    for j, word in enumerate(words):
        if T.ends_sentence(word, ABBREVIATIONS[lang]):
            spans.append((start, j + 1))
            start = j + 1
    if start < len(words):
        spans.append((start, len(words)))
    return spans


def sentence_score(words, query) -> float:
    """The query score of Tombros and Sanderson (1998): the square of the number of different query words in the
    sentence, divided by the number of query words."""
    if not query:
        return 0.0
    found = {plain(w) for w in words if plain(w) in query}
    return len(found) ** 2 / len(query)


def best_sentence(words, query, lang: str) -> tuple:
    """The (start, end) of the sentence with the highest query score; the first such sentence on a tie."""
    best, most = None, -1.0
    for a, b in sentence_spans(words, lang):
        score = sentence_score(words[a:b], query)
        if score > most:
            best, most = (a, b), score
    return best if best is not None else (0, 0)


# --- the questions of XQuAD and their paragraphs --------------------------------------------------------------------------

@functools.cache
def _paragraph(lang: str, i: int) -> tuple:
    """The words of paragraph i cut at the spaces, and where each starts and ends in its text."""
    text = xquad.paragraphs(lang)[i].text
    spans = [(m.start(), m.end()) for m in _PIECE.finditer(text)]
    return tuple(text[a:b] for a, b in spans), tuple(spans)


def own_paragraph(lang: str, k: int) -> int:
    """The number of the own paragraph of question k: the paragraph that holds its answer."""
    return xquad.questions(lang)[k].paragraph


def words_of(lang: str, i: int) -> list:
    """The words of paragraph i cut at the spaces, as a new list."""
    return list(_paragraph(lang, i)[0])


def question_place(qid: str, lang: str) -> int:
    """The place of a question in xquad.questions(lang), by its id."""
    return B.question_place(qid, lang)


def query_of(lang: str, k: int) -> set:
    """The query words of question k, as a new set."""
    return set(_queries(lang)[k])


@functools.cache
def _queries(lang: str) -> tuple:
    return tuple(frozenset(query_words(q.question, lang)) for q in xquad.questions(lang))


def results(lang: str, k: int, n: int = TOP) -> list:
    """The first n paragraphs that BM25 gives question k, best first."""
    return [i for i, _ in B.top(lang, "bm25", k, n)]


def span_of(lang: str, k: int, i: int, kind: str, size: int = SIZE) -> tuple:
    """The (start, end) of the words of paragraph i that a snippet of a kind shows for question k: the first `size`
    words, the window of `size` words with the most query words, or the best sentence, whole."""
    words = _paragraph(lang, i)[0]
    query = _queries(lang)[k]
    if kind == "first":
        return 0, min(size, len(words))
    if kind == "window":
        start = _fast_start(lang, k, i, size)
        return start, min(start + size, len(words))
    return best_sentence(words, query, lang)


def _marks(lang: str, k: int, i: int) -> np.ndarray:
    words = _paragraph(lang, i)[0]
    query = _queries(lang)[k]
    return np.array([plain(w) in query for w in words], dtype=int)


def _fast_start(lang: str, k: int, i: int, size: int) -> int:
    """best_start with the hits of all windows at once: the same start."""
    marks = _marks(lang, k, i)
    if len(marks) < size:
        return 0
    counts = np.convolve(marks, np.ones(size, dtype=int), mode="valid")
    return int(np.argmax(counts)) if counts.max() > 0 else 0


def shows_answer(lang: str, k: int, start: int, end: int) -> bool:
    """True when one of the answers of question k lies inside the words start to end of its own paragraph."""
    q = xquad.questions(lang)[k]
    spans = _paragraph(lang, q.paragraph)[1]
    if end <= start or end > len(spans):
        return False
    a, b = spans[start][0], spans[end - 1][1]
    return any(a <= s and s + len(t) <= b for t, s in q.answers)


def shares(lang: str, size: int = SIZE) -> dict:
    """For each kind of snippet: the share of the questions whose own paragraph, with that snippet, shows the answer."""
    def make():
        qs = xquad.questions(lang)
        out = {}
        for kind in KINDS:
            shown = [shows_answer(lang, k, *span_of(lang, k, q.paragraph, kind, size)) for k, q in enumerate(qs)]
            out[kind] = sum(shown) / len(qs)
        return out
    return kept(f"ir-snippets:shares:{lang}:{size}", make)


def sentence_length(lang: str) -> float:
    """The mean number of words of the best sentences of the own paragraphs, over all the questions."""
    def make():
        qs = xquad.questions(lang)
        lengths = [b - a for a, b in (span_of(lang, k, q.paragraph, "sentence") for k, q in enumerate(qs))]
        return sum(lengths) / len(lengths)
    return kept(f"ir-snippets:sentence-length:{lang}", make)


# --- the pictures ----------------------------------------------------------------------------------------------------------

_UI, _LABEL, _CELL, _HEAD, _NUMBER = S._UI, S._LABEL, S._CELL, S._HEAD, S._NUMBER
_DATA, _FAINT, _PRED, _LOSS = S._DATA, S._FAINT, S._PRED, S._LOSS
_CHIP_PRED, _CHIP_DATA = S._CHIP_PRED, S._CHIP_DATA
_note, _colors, _label, _num, _dec, _pct = S._note, S._colors, S._label, S._num, S._dec, S._pct
_cell, _table = S._cell, S._table
_fn, _call, _empty, _SLOW, _STOPPED = S._fn, S._call, S._empty, S._SLOW, S._STOPPED
_WHY_EMPTY = S._WHY_EMPTY
_is_number = S._is_number

_TEXT = "font-family:var(--ea-font-ui);font-size:0.85rem;line-height:1.45"
_RESULT = _TEXT + ";margin:0.45rem 0 0.1rem;color:var(--ea-ink)"
_SNIPPET = _TEXT + ";margin:0 0 0.2rem 1.1rem"
_BOLD = re.compile(r"\*\*(.+?)\*\*")


class _Odd(Exception):
    """A learner's function gave something the picture cannot draw."""


def _chip(text_: str, style: str = _CHIP_DATA) -> str:
    return f'<span style="{style}">{html.escape(text_)}</span>'


def _snippet_html(text_: str, right: bool = True, note: str = "") -> str:
    """A snippet as a result page shows it: a word between two pairs of stars in bold, blue when the snippet is the
    course's, purple when not; the rest gray."""
    bold = _PRED if right else _LOSS
    body = _BOLD.sub(lambda m: f'<b style="{bold}">{m.group(1)}</b>', html.escape(text_))
    style = _DATA if right else _LOSS.replace(";font-weight:600", "")
    extra = f'<div style="{_UI};margin-top:0">{note}</div>' if note else ""
    return f'<div style="{_SNIPPET};{style}">{body}{extra}</div>'


_WORDS = {
    "en": {"paragraph": "paragraph", "own": "own paragraph", "query": "Query words", "answer": "Answer in XQuAD",
           "question": "Question", "yes": "yes", "no": "no",
           "kinds": {"first": "The first {n} words", "window": "Window of {n} words", "sentence": "Best sentence"}},
    "nl": {"paragraph": "alinea", "own": "eigen alinea", "query": "Zoekwoorden", "answer": "Antwoord in XQuAD",
           "question": "Vraag", "yes": "ja", "no": "nee",
           "kinds": {"first": "De eerste {n} woorden", "window": "Venster van {n} woorden", "sentence": "Beste zin"}},
}


def _kind_name(kind: str, lang: str, size: int = SIZE) -> str:
    return _WORDS[lang]["kinds"][kind].format(n=size)


def _question_html(lang: str, k: int) -> str:
    """The question, its query words as chips, and its answer in XQuAD."""
    w = _WORDS[lang]
    q = xquad.questions(lang)[k]
    chips = "".join(_chip(x) for x in sorted(_queries(lang)[k]))
    return (f'<div style="{_LABEL}">{html.escape(w["question"])}: <i>{html.escape(q.question)}</i></div>'
            f'<div style="{_LABEL}">{html.escape(w["query"])}: {chips}</div>'
            f'<div style="{_LABEL}">{html.escape(w["answer"])}: <i>{html.escape(q.answers[0][0])}</i></div>')


def _result_head(lang: str, k: int, place: int, i: int) -> str:
    """One result of a result page: its place, the title of its article and the number of the paragraph."""
    w = _WORDS[lang]
    own = " " + _chip(w["own"]) if i == xquad.questions(lang)[k].paragraph else ""
    title = xquad.paragraphs(lang)[i].title
    return f'<div style="{_RESULT}">{place}. <i>{html.escape(title)}</i>, {w["paragraph"]} {str(i)}{own}</div>'


def course_snippet(lang: str, k: int, i: int, kind: str = "window", size: int = SIZE) -> str:
    """The snippet of a kind of paragraph i for question k, as snippet() writes it."""
    a, b = span_of(lang, k, i, kind, size)
    return snippet(words_of(lang, i), set(_queries(lang)[k]), a, b - a)


_IDEA_TEXT = {
    "en": {"title": "A result page with snippets",
           "page": "The first {n} results of BM25, each with the window of {size} words with the most query words:",
           "own": "Three snippets of the own paragraph, paragraph {i}:",
           "head_own": ["Snippet", "Words", "Shows the answer"],
           "all": "Over all {n} questions: how often the snippet of the own paragraph shows the answer.",
           "head_all": ["Snippet", "Words", "Shows the answer"],
           "mean": "{x} on average",
           "gray": "Gray: the results, the text of the snippets and the number of words.",
           "blue": "Blue: the query words in the snippets, and whether and how often a snippet shows the answer."},
    "nl": {"title": "Een resultatenpagina met snippets",
           "page": "De eerste {n} resultaten van BM25, elk met het venster van {size} woorden met de meeste zoekwoorden:",
           "own": "Drie snippets van de eigen alinea, alinea {i}:",
           "head_own": ["Snippet", "Woorden", "Toont het antwoord"],
           "all": "Over alle {n} vragen: hoe vaak de snippet van de eigen alinea het antwoord toont.",
           "head_all": ["Snippet", "Woorden", "Toont het antwoord"],
           "mean": "gemiddeld {x}",
           "gray": "Grijs: de resultaten, de tekst van de snippets en het aantal woorden.",
           "blue": "Blauw: de zoekwoorden in de snippets, en of en hoe vaak een snippet het antwoord toont."},
}

IDEA_SIZES = (SIZE, 30)   # the windows of the table under the idea


def idea_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang] | _WORDS[lang]
    k = question_place(EXAMPLE, lang)
    own = xquad.questions(lang)[k].paragraph
    body = _question_html(lang, k)
    body += _label(w["page"].format(n=TOP, size=SIZE))
    for place, i in enumerate(results(lang, k), 1):
        body += _result_head(lang, k, place, i) + _snippet_html(course_snippet(lang, k, i))
    rows = []
    for kind in KINDS:
        a, b = span_of(lang, k, own, kind)
        shown = w["yes"] if shows_answer(lang, k, a, b) else w["no"]
        rows.append("<tr>" + _cell(html.escape(_kind_name(kind, lang)), _FAINT)
                    + _cell(_snippet_html(course_snippet(lang, k, own, kind)).replace(_SNIPPET, _TEXT))
                    + _cell(str(b - a), _DATA, _NUMBER) + _cell(html.escape(shown), _PRED) + "</tr>")
    body += _label(w["own"].format(i=str(own))) + _table(rows, w["head_own"], (2,))
    rows = []
    for kind, size in (("first", SIZE), ("window", SIZE), ("window", IDEA_SIZES[1]), ("sentence", SIZE)):
        n_words = str(size) if kind != "sentence" else w["mean"].format(x=_dec(sentence_length(lang), lang, 1))
        rows.append("<tr>" + _cell(html.escape(_kind_name(kind, lang, size)), _FAINT) + _cell(html.escape(n_words), _DATA, _NUMBER)
                    + _cell(html.escape(_pct(shares(lang, size)[kind], lang)), _PRED, _NUMBER) + "</tr>")
    body += _label(w["all"].format(n=_num(len(xquad.questions(lang)), lang))) + _table(rows, w["head_all"], (1, 2))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


def _count(x) -> int:
    """A learner's count as an int, when it is a whole number of 0 or more."""
    if isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 0:
        raise _Odd
    return int(x)


def _guarded(make):
    """make() within the time limit of a picture; None when the learner's function gives something odd."""
    def run():
        try:
            return make()
        except _Odd:
            return None
    return _call(run)


def _mine(got, want, lang: str) -> str:
    """A learner's whole number: blue when it is the course's, purple with the course's after it in gray when not."""
    if got == want:
        return f'<span style="{_PRED}">{_num(got, lang)}</span>'
    return f'<span style="{_LOSS}">{_num(got, lang)}</span> <span style="{_DATA}">({_num(want, lang)})</span>'


def _strip(mine: list, want: list) -> str:
    """A bar for each window: as high as its hits, blue when they are the course's, purple when not."""
    most = max([1] + mine + want)
    bars = "".join(f'<span title="{j}: {m}" style="display:inline-block;width:4px;margin-right:1px;'
                   f'height:{max(0.08, 2.4 * m / most):.2f}rem;background:var(--ea-{"pred" if m == c else "loss"});'
                   f'vertical-align:bottom;border-radius:1px"></span>' for j, (m, c) in enumerate(zip(mine, want)))
    return f'<div style="display:flex;align-items:flex-end;height:2.6rem;margin:0.3rem 0;overflow-x:auto">{bars}</div>'


_HITS_TEXT = {
    "en": {"title": "Your hits: the query words of the first results", "kind": "a whole number of 0 or more",
           "table": "The first {n} results of BM25: your hits in the whole paragraph, and the most in one window of {size} words.",
           "head": ["Place", "Paragraph", "Words", "Your hits", "Most in a window"],
           "strip": "Your hits in each window of {size} words of the own paragraph, paragraph {i}, from the first window to the last; the snippet goes where the bar is highest for the first time:",
           "gray": "Gray: the results, their words, and the course's value where yours differs.",
           "blue": "Blue: your hits where they are the course's.",
           "purple": "Purple: your hits where they differ from the course's."},
    "nl": {"title": "Je hits: de zoekwoorden van de eerste resultaten", "kind": "een geheel getal van 0 of meer",
           "table": "De eerste {n} resultaten van BM25: je hits in de hele alinea, en de meeste in één venster van {size} woorden.",
           "head": ["Plaats", "Alinea", "Woorden", "Jouw hits", "Meeste in een venster"],
           "strip": "Je hits in elk venster van {size} woorden van de eigen alinea, alinea {i}, van het eerste venster tot het laatste; de snippet komt waar de staaf voor het eerst het hoogst is:",
           "gray": "Grijs: de resultaten, de woorden, en de waarde van de course waar de jouwe verschilt.",
           "blue": "Blauw: je hits waar deze gelijk zijn aan de hits van de course.",
           "purple": "Paars: je hits waar deze verschillen van de hits van de course."},
}


def hits_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _HITS_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    k = question_place(EXAMPLE, lang)
    own = xquad.questions(lang)[k].paragraph
    found = results(lang, k)

    def make():
        query = set(_queries(lang)[k])
        whole = {i: _count(fn(words_of(lang, i), set(query))) for i in found}
        words = words_of(lang, own)
        strip = [_count(fn(words[s:s + SIZE], set(query))) for s in range(len(words) - SIZE + 1)]
        windows = {i: max(_count(fn(words_of(lang, i)[s:s + SIZE], set(query)))
                          for s in range(max(len(words_of(lang, i)) - SIZE + 1, 1))) for i in found}
        return whole, strip, windows

    got = _guarded(make)
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    if got is None:
        return _empty(w, "odd")
    whole, strip, windows = got
    query = _queries(lang)[k]
    rows = []
    for place, i in enumerate(found, 1):
        words = words_of(lang, i)
        best = max(hits(words[s:s + SIZE], query) for s in range(max(len(words) - SIZE + 1, 1)))
        title = f"{xquad.paragraphs(lang)[i].title}, {_WORDS[lang]['paragraph']} {str(i)}"
        rows.append("<tr>" + _cell(str(place), _FAINT, _NUMBER) + _cell(html.escape(title), _DATA)
                    + _cell(_num(len(words), lang), _DATA, _NUMBER) + _cell(_mine(whole[i], hits(words, query), lang), "", _NUMBER)
                    + _cell(_mine(windows[i], best, lang), "", _NUMBER) + "</tr>")
    body = _question_html(lang, k) + _label(w["table"].format(n=TOP, size=SIZE)) + _table(rows, w["head"], (0, 2, 3, 4))
    words = words_of(lang, own)
    want = [hits(words[s:s + SIZE], query) for s in range(len(words) - SIZE + 1)]
    body += _label(w["strip"].format(size=SIZE, i=str(own))) + _strip(strip, want)
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


BEST_QUESTIONS = (EXAMPLE, "5729e2316aef0514001550c5")   # the parliament of Victoria; the size of the Amazon basin


_BEST_TEXT = {
    "en": {"title": "Your best_start: snippets on a result page", "kind": "a place in the paragraph",
           "page": "The first {n} results of BM25, each with the {size} words from your best_start:",
           "course": "The course's window starts at word {c}, yours at word {m}.",
           "all": "Over all {n} questions: how often the snippet of the own paragraph shows the answer.",
           "head_all": ["Snippet", "Shows the answer"],
           "mine": "Your window of {n} words",
           "gray": "Gray: the results, the text of the snippets, the course's start where yours differs, and the other snippets.",
           "blue": "Blue: the query words of your windows, and how often they show the answer, where they are the course's.",
           "purple": "Purple: your windows and how often they show the answer, where they differ from the course's."},
    "nl": {"title": "Je best_start: snippets op een resultatenpagina", "kind": "een plaats in de alinea",
           "page": "De eerste {n} resultaten van BM25, elk met de {size} woorden vanaf je best_start:",
           "course": "Het venster van de course begint bij woord {c}, het jouwe bij woord {m}.",
           "all": "Over alle {n} vragen: hoe vaak de snippet van de eigen alinea het antwoord toont.",
           "head_all": ["Snippet", "Toont het antwoord"],
           "mine": "Jouw venster van {n} woorden",
           "gray": "Grijs: de resultaten, de tekst van de snippets, het begin van de course waar het jouwe verschilt, en de andere snippets.",
           "blue": "Blauw: de zoekwoorden van je vensters, en hoe vaak deze het antwoord tonen, waar dit gelijk is aan wat de course geeft.",
           "purple": "Paars: je vensters en hoe vaak deze het antwoord tonen, waar dit verschilt van wat de course geeft."},
}


def _start(x, n: int) -> int:
    """A learner's start as an int, when it is a place in a paragraph of n words."""
    x = _count(x)
    if x >= max(n, 1):
        raise _Odd
    return x


def best_start_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _BEST_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    shown = [(question_place(qid, lang), results(lang, question_place(qid, lang))) for qid in BEST_QUESTIONS]
    qs = xquad.questions(lang)

    def make():
        page = {(k, i): _start(fn(words_of(lang, i), set(_queries(lang)[k]), SIZE), len(words_of(lang, i)))
                for k, found in shown for i in found}
        hit = 0
        for k, q in enumerate(qs):
            n = len(_paragraph(lang, q.paragraph)[0])
            a = _start(fn(words_of(lang, q.paragraph), set(_queries(lang)[k]), SIZE), n)
            hit += shows_answer(lang, k, a, min(a + SIZE, n))
        return page, hit / len(qs)

    got = _guarded(make)
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    if got is None:
        return _empty(w, "odd")
    page, share = got
    body = ""
    for k, found in shown:
        body += _question_html(lang, k) + _label(w["page"].format(n=TOP, size=SIZE))
        for place, i in enumerate(found, 1):
            mine, want = page[(k, i)], span_of(lang, k, i, "window")[0]
            text_ = snippet(words_of(lang, i), set(_queries(lang)[k]), mine, SIZE)
            note = "" if mine == want else html.escape(w["course"].format(c=want, m=mine))
            body += _result_head(lang, k, place, i) + _snippet_html(text_, mine == want, note)
    want = shares(lang)
    same = abs(share - want["window"]) < 1e-9
    mine = (f'<span style="{_PRED if same else _LOSS}">{html.escape(_pct(share, lang))}</span>'
            + ("" if same else f' <span style="{_DATA}">({html.escape(_pct(want["window"], lang))})</span>'))
    rows = ["<tr>" + _cell(html.escape(_kind_name("first", lang)), _FAINT) + _cell(html.escape(_pct(want["first"], lang)), _DATA, _NUMBER) + "</tr>",
            "<tr>" + _cell(html.escape(w["mine"].format(n=SIZE)), _FAINT) + _cell(mine, "", _NUMBER) + "</tr>",
            "<tr>" + _cell(html.escape(_kind_name("sentence", lang)), _FAINT) + _cell(html.escape(_pct(want["sentence"], lang)), _DATA, _NUMBER) + "</tr>"]
    body += _label(w["all"].format(n=_num(len(qs), lang))) + _table(rows, w["head_all"], (1,))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


SNIPPET_QUESTIONS = (EXAMPLE, "57296d571d04691400779415")   # the parliament of Victoria; the main role of the primes

_SNIPPET_TEXT = {
    "en": {"title": "Your snippets on a result page", "kind": "a text",
           "page": "The first {n} results of BM25, each with your snippet of the window of {size} words with the most query words:",
           "course": "The course: {c}",
           "gray": "Gray: the results, and the course's snippet where yours differs.",
           "blue": "Blue: the query words in bold in your snippets, where they are the course's.",
           "purple": "Purple: your snippets where they differ from the course's."},
    "nl": {"title": "Je snippets op een resultatenpagina", "kind": "een tekst",
           "page": "De eerste {n} resultaten van BM25, elk met je snippet van het venster van {size} woorden met de meeste zoekwoorden:",
           "course": "De course: {c}",
           "gray": "Grijs: de resultaten, en de snippet van de course waar de jouwe verschilt.",
           "blue": "Blauw: de vetgedrukte zoekwoorden in je snippets, waar deze gelijk zijn aan de snippets van de course.",
           "purple": "Paars: je snippets waar deze verschillen van de snippets van de course."},
}


def snippet_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _SNIPPET_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    shown = [(question_place(qid, lang), results(lang, question_place(qid, lang))) for qid in SNIPPET_QUESTIONS]

    def make():
        out = {}
        for k, found in shown:
            for i in found:
                text_ = fn(words_of(lang, i), set(_queries(lang)[k]), span_of(lang, k, i, "window")[0], SIZE)
                if not isinstance(text_, str):
                    raise _Odd
                out[(k, i)] = text_
        return out

    got = _guarded(make)
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    if got is None:
        return _empty(w, "odd")
    body = ""
    for k, found in shown:
        body += _question_html(lang, k) + _label(w["page"].format(n=TOP, size=SIZE))
        for place, i in enumerate(found, 1):
            mine, want = got[(k, i)][:600], course_snippet(lang, k, i)
            note = "" if mine == want else html.escape(w["course"].format(c=want))
            body += _result_head(lang, k, place, i) + _snippet_html(mine, mine == want, note)
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


# --- the demo: the result page of a question, with a kind and a length of snippet ----------------------------------------

TRY_QUESTIONS = (EXAMPLE, "5729e2316aef0514001550c5", "572648e8dd62a815002e8076", "57296d571d04691400779415",
                 "571cd3b55efbb31900334e05", "571144d1a58dae1900cd6d6f")
TRY_TOP = 5
TRY_SIZES = (5, 40, 5)   # the slider: from 5 to 40 words, in steps of 5


def try_options(lang: str) -> dict:
    """The questions of the demo's dropdown, by their text, to their id."""
    qs = xquad.questions(lang)
    return {qs[question_place(qid, lang)].question.strip(): qid for qid in TRY_QUESTIONS}


KIND_OPTIONS = {
    "en": {"the first words": "first", "the window with the most query words": "window", "the best sentence": "sentence"},
    "nl": {"de eerste woorden": "first", "het venster met de meeste zoekwoorden": "window", "de beste zin": "sentence"},
}

_TRY_TEXT = {
    "en": {"title": "Your result page",
           "page": "The first {n} results of BM25, each with a snippet of this kind:",
           "shown": "The snippet of the own paragraph shows the answer: {a}.",
           "missing": "The own paragraph is not among the first {n} results.",
           "all": "Over all {n} questions, with snippets of {size} words: how often the snippet of the own paragraph shows the answer.",
           "head_all": ["Snippet", "Words", "Shows the answer"],
           "mean": "{x} on average",
           "gray": "Gray: the results, the text of the snippets, and the other kinds of snippet.",
           "blue": "Blue: the query words in the snippets, and how often the kind you chose shows the answer."},
    "nl": {"title": "Jouw resultatenpagina",
           "page": "De eerste {n} resultaten van BM25, elk met een snippet van deze soort:",
           "shown": "De snippet van de eigen alinea toont het antwoord: {a}.",
           "missing": "De eigen alinea staat niet bij de eerste {n} resultaten.",
           "all": "Over alle {n} vragen, met snippets van {size} woorden: hoe vaak de snippet van de eigen alinea het antwoord toont.",
           "head_all": ["Snippet", "Woorden", "Toont het antwoord"],
           "mean": "gemiddeld {x}",
           "gray": "Grijs: de resultaten, de tekst van de snippets, en de andere soorten snippet.",
           "blue": "Blauw: de zoekwoorden in de snippets, en hoe vaak de soort die je koos het antwoord toont."},
}


def _try_size(size) -> int:
    try:
        size = int(size)
    except (TypeError, ValueError):
        return SIZE
    return min(max(size, TRY_SIZES[0]), TRY_SIZES[1])


def try_picture(lang: str, qid: str = EXAMPLE, size=SIZE, kind: str = "window") -> str:
    lang = lang if lang in LANGS else "en"
    w = _TRY_TEXT[lang] | _WORDS[lang]
    qid = qid if qid in TRY_QUESTIONS else EXAMPLE
    kind = kind if kind in KINDS else "window"
    size = _try_size(size)
    k = question_place(qid, lang)
    own = xquad.questions(lang)[k].paragraph
    found = results(lang, k, TRY_TOP)
    body = _question_html(lang, k) + _label(w["page"].format(n=TRY_TOP))
    for place, i in enumerate(found, 1):
        body += _result_head(lang, k, place, i) + _snippet_html(course_snippet(lang, k, i, kind, size))
    if own in found:
        shown = shows_answer(lang, k, *span_of(lang, k, own, kind, size))
        body += _label(w["shown"].format(a=w["yes"] if shown else w["no"]))
    else:
        body += _label(w["missing"].format(n=TRY_TOP))
    rows = []
    for other in KINDS:
        n_words = str(size) if other != "sentence" else w["mean"].format(x=_dec(sentence_length(lang), lang, 1))
        style = _PRED if other == kind else _DATA
        rows.append("<tr>" + _cell(html.escape(_kind_name(other, lang, size)), _FAINT) + _cell(html.escape(n_words), _DATA, _NUMBER)
                    + _cell(html.escape(_pct(shares(lang, size)[other], lang)), style, _NUMBER) + "</tr>")
    body += _label(w["all"].format(n=_num(len(xquad.questions(lang)), lang), size=size)) + _table(rows, w["head_all"], (1, 2))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------------

_Val, _DOTS, _SAME_TEXTS = S._Val, S._DOTS, S._SAME_TEXTS
_show, _written = V._show, V._written


def _words(*items) -> _Val:
    """A set of query words that the messages write in this order, with double quotes, the way the lesson writes it."""
    return _Val("{" + ", ".join(f'"{x}"' for x in items) + "}", set(items))


_HI_HEAD = "def hits(words, query):\n    count = 0\n    for word in words:\n"
_HI_END = "            count = count + 1\n    return count\n"
_HI = _HI_HEAD + "        if plain(word) in query:\n" + _HI_END
_HI_START = _HI_HEAD + "        if ...:\n" + _HI_END
_HI_EXACT = _HI_HEAD + "        if word in query:\n" + _HI_END
_HI_LOWER = _HI_HEAD + "        if word.lower() in query:\n" + _HI_END

_BS_HEAD = "def best_start(words, query, size):\n    best, most = 0, 0\n    for start in range(len(words) - size + 1):\n"
_BS_END = "        if count > most:\n            best, most = start, count\n    return best\n"
_BS = _BS_HEAD + "        count = hits(words[start:start + size], query)\n" + _BS_END
_BS_START = _BS_HEAD + "        count = hits(..., query)\n" + _BS_END
_BS_LENGTH = _BS_HEAD + "        count = hits(words[start:size], query)\n" + _BS_END
_BS_SHORT = _BS_HEAD + "        count = hits(words[start:start + size - 1], query)\n" + _BS_END

_SN_HEAD = ("def snippet(words, query, start, size):\n    shown = []\n    for word in words[start:start + size]:\n"
            "        if plain(word) in query:\n")
_SN_END = ('        shown.append(word)\n    text = " ".join(shown)\n    if start > 0:\n        text = "… " + text\n'
           '    if start + size < len(words):\n        text = text + " …"\n    return text\n')
_SN = _SN_HEAD + '            word = "**" + word + "**"\n' + _SN_END
_SN_START = _SN_HEAD + "            word = ...\n" + _SN_END
_SN_PLAIN = _SN_HEAD + '            word = "**" + plain(word) + "**"\n' + _SN_END
_SN_OPEN = _SN_HEAD + '            word = "**" + word\n' + _SN_END
_SN_HTML = _SN_HEAD + '            word = "<b>" + word + "</b>"\n' + _SN_END

# The small cases the exercise texts show; the same in both lessons, one in English and one in Dutch.
TOY_HI = ((_written(["Denver", "Broncos,", "beat", "the", "Panthers."]), _words("denver", "broncos", "panthers")),
          (_written(["Rijn,", "Maas", "en", "Rijn"]), _words("rijn")))
TOY_BS = ((_written(["The", "Panthers", "lost", "to", "the", "Denver", "Broncos."]), _words("denver", "broncos"), _written(2)),
          (_written(["Rotterdam", "ligt", "aan", "de", "Maas", "en", "de", "Rijn."]), _words("maas", "rijn"), _written(4)))
TOY_SN = ((_written(["The", "Denver", "Broncos", "beat", "the", "Carolina", "Panthers."]), _words("denver", "panthers"),
           _written(1), _written(3)),
          (_written(["De", "Rijn", "en", "de", "Maas"]), _words("rijn", "maas"), _written(0), _written(5)))


def _course_words(lang: str, qid: str = EXAMPLE, which: int = -1) -> _Val:
    """The words of a paragraph of the course: the own paragraph of a question (which = -1) or its result at a place."""
    k = question_place(qid, lang)
    i = xquad.questions(lang)[k].paragraph if which < 0 else results(lang, k)[which]
    return _Val(f'C.words_of("{lang}", {i})', words_of(lang, i))


def _course_query(lang: str, qid: str = EXAMPLE) -> _Val:
    k = question_place(qid, lang)
    return _Val(f'C.query_of("{lang}", {k})', query_of(lang, k))


REQUIRED = ["hits", "best_start", "snippet", "kinds"]

EXERCISES = {
    "hits": FunctionBox(
        name="hits",
        picture=hits_picture,
        function="hits",
        start=_HI_START,
        solution=_HI,
        show=_show,
        names={"plain": plain},
        visible=list(TOY_HI),
        hidden=[(_written([]), _words("rhine")), (_written(["the", "a", "of"]), _Val("set()", set())),
                (_written(["(NFC)", "nfc", "NFC."]), _words("nfc")), (_written(["50,", "50", "1950"]), _words("50")),
                (_written(["Warsaw's", "warsaw"]), _words("warsaw's")),
                (_course_words("en"), _course_query("en")), (_course_words("nl", which=1), _course_query("nl"))],
        first=[_DOTS],
        mistakes=[
            Mistake("exact", wrong=_HI_EXACT,
                    en="`{call}` gives {got}, and it should give {want}: your function looks for each word as it is written, with its capital letters and its punctuation, and the query words are in small letters without them. Which function of the box gives a word the way the query writes it?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie zoekt elk woord op zoals het geschreven staat, met hoofdletters en leestekens, en de zoekwoorden staan in kleine letters zonder leestekens. Welke functie van het vak geeft een woord zoals de zoekvraag het schrijft?"),
            Mistake("lower", wrong=_HI_LOWER,
                    en="`{call}` gives {got}, and it should give {want}: `.lower()` takes away the capital letters, but a comma or a period stays on the word, so *Broncos,* is not *broncos*. Which function of the box also takes off the punctuation?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `.lower()` haalt de hoofdletters weg, maar een komma of een punt blijft aan het woord vastzitten, dus *Rijn,* is geen *rijn*. Welke functie van het vak haalt ook de leestekens weg?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the test: is the word, in small letters and without its punctuation, one of the query words?",
                      "nl": "Vervang de `...` door de test: is het woord, in kleine letters en zonder leestekens, een van de zoekwoorden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your test look up the plain form of each word in `query`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Zoekt je test de kale vorm van elk woord op in `query`?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function also work for an empty list, for a query without words, for a word in brackets, and for the course's paragraphs?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook voor een lege lijst, voor een zoekvraag zonder woorden, voor een woord tussen haakjes, en voor de alinea's van de course?"},
            "correct": {"en": "Right: your function counts the query words, however they are written. In the picture below, which of the first results holds the most query words, and in which window of its paragraph?",
                        "nl": "Goed: je functie telt de zoekwoorden, hoe de woorden ook geschreven staan. In het plaatje hieronder: welk van de eerste resultaten bevat de meeste zoekwoorden, en in welk venster van de alinea?"},
        },
    ),
    "best_start": FunctionBox(
        name="best_start",
        picture=best_start_picture,
        function="best_start",
        start=_BS_START,
        solution=_BS,
        show=_show,
        names={"hits": hits},
        visible=list(TOY_BS),
        hidden=[(_written(["Rijn", "en", "Maas"]), _words("rijn"), _written(5)),
                (_written(["one", "two", "three"]), _words("four"), _written(2)),
                (_written(["x", "Rijn", "y", "Rijn", "z"]), _words("rijn"), _written(2)),
                (_written(["Maas", "en", "Rijn", "Maas", "Rijn"]), _words("rijn", "maas"), _written(2)),
                (_written(["a", "b", "Rijn", "Maas"]), _words("rijn", "maas"), _written(1)),
                (_course_words("en"), _course_query("en"), _written(SIZE)),
                (_course_words("nl", which=2), _course_query("nl"), _written(SIZE)),
                (_course_words("en", "5729e2316aef0514001550c5"), _course_query("en", "5729e2316aef0514001550c5"), _written(30))],
        first=[_DOTS],
        mistakes=[
            Mistake("length", wrong=_BS_LENGTH,
                    en="`{call}` gives {got}, and it should give {want}: `words[start:size]` stops at place `size` of the whole list, not `size` words after `start`, so the later windows get shorter and then empty. Where does a window of `size` words that starts at `start` end?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `words[start:size]` stopt bij plaats `size` van de hele lijst, niet `size` woorden na `start`, dus de latere vensters worden korter en daarna leeg. Waar eindigt een venster van `size` woorden dat bij `start` begint?"),
            Mistake("short", wrong=_BS_SHORT,
                    en="`{call}` gives {got}, and it should give {want}: your windows hold one word less than `size`, because a slice stops before the place after the colon. Which place should come after the colon, so that the window holds `size` words?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je vensters bevatten één woord minder dan `size`, want een slice stopt bij de plaats na de dubbele punt, zonder deze plaats zelf. Welke plaats moet na de dubbele punt staan, zodat het venster `size` woorden bevat?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the window: the `size` words of `words` from place `start` on.",
                      "nl": "Vervang de `...` door het venster: de `size` woorden van `words` vanaf plaats `start`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does `hits` get the `size` words from place `start`, as a slice of `words`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Krijgt `hits` de `size` woorden vanaf plaats `start`, als een slice van `words`?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function also work for fewer words than `size`, for a paragraph without query words, for two windows with as many query words, and for the course's paragraphs?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook voor minder woorden dan `size`, voor een alinea zonder zoekwoorden, voor twee vensters met evenveel zoekwoorden, en voor de alinea's van de course?"},
            "correct": {"en": "Right: your function finds the window with the most query words. In the picture below, which result's snippet shows you where the answer is, and how often does your window show the answer over all the questions?",
                        "nl": "Goed: je functie vindt het venster met de meeste zoekwoorden. In het plaatje hieronder: de snippet van welk resultaat laat zien waar het antwoord staat, en hoe vaak toont je venster het antwoord over alle vragen?"},
        },
    ),
    "snippet": FunctionBox(
        name="snippet",
        picture=snippet_picture,
        function="snippet",
        start=_SN_START,
        solution=_SN,
        show=_show,
        names={"plain": plain},
        visible=list(TOY_SN),
        hidden=[(_written(["Rijn", "en", "Maas"]), _words("rijn"), _written(0), _written(2)),
                (_written(["Rijn", "en", "Maas"]), _words("maas"), _written(1), _written(2)),
                (_written(["Rijn", "en", "Maas"]), _words("rijn"), _written(0), _written(10)),
                (_written(["The", "(NFC)", "champion"]), _words("nfc"), _written(1), _written(1)),
                (_written(["a", "b"]), _Val("set()", set()), _written(0), _written(2)),
                (_course_words("en"), _course_query("en"), _written(span_of("en", question_place(EXAMPLE, "en"),
                 xquad.questions("en")[question_place(EXAMPLE, "en")].paragraph, "window")[0]), _written(SIZE)),
                (_course_words("nl", which=1), _course_query("nl"), _written(0), _written(SIZE))],
        first=[_DOTS],
        mistakes=[
            Mistake("plain", wrong=_SN_PLAIN,
                    en="`{call}` gives {got}, and it should give {want}: your snippet shows the plain form of each query word, without its capital letters and its punctuation, so the text no longer reads as the paragraph does. Which form of the word should go between the stars?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je snippet toont van elk zoekwoord de vorm zonder hoofdletters en leestekens, dus de tekst leest niet meer zoals de alinea. Welke vorm van het woord hoort tussen de sterretjes?"),
            Mistake("open", wrong=_SN_OPEN,
                    en="`{call}` gives {got}, and it should give {want}: the stars go only before each query word, so the bold never ends. What has to come after the word too?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de sterretjes staan alleen aan het begin van elk zoekwoord, dus het vet stopt nooit. Wat moet er ook na het woord komen?"),
            Mistake("html", wrong=_SN_HTML,
                    en="`{call}` gives {got}, and it should give {want}: your function writes bold as HTML does, and this lesson writes it as Markdown does. Which characters does Markdown put on each side of a word in bold?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie schrijft vet zoals HTML dat doet, en deze les schrijft vet zoals Markdown. Welke tekens zet Markdown aan elke kant van een vetgedrukt woord?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the query word in bold: the word as it is written, with two stars before it and two after it.",
                      "nl": "Vervang de `...` door het zoekwoord in het vet: het woord zoals het geschreven staat, met twee sterretjes aan elke kant."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does each query word, as it is written, get two stars before it and two after it?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Krijgt elk zoekwoord, zoals het geschreven staat, twee sterretjes aan elke kant?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function also work for a window at the start or the end of the words, for a window longer than the words, for a word in brackets, and for the course's paragraphs?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook voor een venster aan het begin of het eind van de woorden, voor een venster langer dan de woorden, voor een woord tussen haakjes, en voor de alinea's van de course?"},
            "correct": {"en": "Right: your function writes a snippet as a result page shows it. In the picture below, which result would you open for the question, and why?",
                        "nl": "Goed: je functie schrijft een snippet zoals een resultatenpagina deze toont. In het plaatje hieronder: welk resultaat zou je openen voor de vraag, en waarom?"},
        },
    ),
}

# The sort: which summaries are the same for every query, and which are made for the query.
EXERCISES["kinds"] = Sort(
    name="kinds",
    groups={
        "static": {"en": "static: the same for every query", "nl": "statisch: hetzelfde bij elke zoekvraag"},
        "dynamic": {"en": "dynamic: made for the query", "nl": "dynamisch: gemaakt voor de zoekvraag"},
    },
    items={
        "first": {"en": "**Opening.** The first two sentences of the document.",
                  "nl": "**Opening.** De eerste twee zinnen van het document."},
        "window": {"en": "**Window.** The 20 words in a row with the most query words.",
                   "nl": "**Venster.** De 20 woorden op een rij met de meeste zoekwoorden."},
        "meta": {"en": "**Description.** The text that the maker of a web page wrote in its HTML to describe the page.",
                 "nl": "**Beschrijving.** De tekst die de maker van een webpagina in de HTML schreef om de pagina te beschrijven."},
        "sentence": {"en": "**Best sentence.** The sentence with the most different query words.",
                     "nl": "**Beste zin.** De zin met de meeste verschillende zoekwoorden."},
        "abstract": {"en": "**Abstract.** The abstract at the top of a paper.",
                     "nl": "**Abstract.** Het abstract bovenaan een paper."},
        "phrase": {"en": "**Phrase.** The words around the place where the whole query occurs as a phrase.",
                   "nl": "**Woordgroep.** De woorden rond de plek waar de hele zoekvraag als woordgroep voorkomt."},
    },
    solution={"first": "static", "window": "dynamic", "meta": "static", "sentence": "dynamic", "abstract": "static",
              "phrase": "dynamic"},
    nudge={
        "first": {"en": "Do the first two sentences change when someone searches for something else?",
                  "nl": "Veranderen de eerste twee zinnen als iemand naar iets anders zoekt?"},
        "window": {"en": "Which words does the window count, and where do they come from?",
                   "nl": "Welke woorden telt het venster, en waar komen deze woorden vandaan?"},
        "meta": {"en": "When was the description written: before or after anyone searched?",
                 "nl": "Wanneer is de beschrijving geschreven: voor of nadat iemand zocht?"},
        "sentence": {"en": "Which words make one sentence better than another?",
                     "nl": "Welke woorden maken de ene zin beter dan de andere?"},
        "abstract": {"en": "Does the abstract of a paper change with the query?",
                     "nl": "Verandert het abstract van een paper met de zoekvraag?"},
        "phrase": {"en": "Without a query, which place would this be?",
                   "nl": "Welke plek zou dit zijn zonder zoekvraag?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the groups again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de groepen opnieuw uit."},
        "empty": {"en": "Sort each summary into the group that fits.",
                  "nl": "Deel elke samenvatting in bij de groep die klopt."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "{item} belongs with the other group.",
                  "nl": "{item} hoort bij de andere groep."},
        "correct": {"en": "All {n} are right. A static summary is made once, before anyone searches: the first sentences, a description or an abstract. A dynamic summary is made when the query comes in, from the places where its words are, so it needs the text of the document at that moment.",
                    "nl": "Alle {n} kloppen. Een statische samenvatting wordt één keer gemaakt, voordat iemand zoekt: de eerste zinnen, een beschrijving of een abstract. Een dynamische samenvatting wordt gemaakt als de zoekvraag binnenkomt, vanuit de plekken waar de woorden van de zoekvraag staan, dus op dat moment is de tekst van het document nodig."},
    },
)

# What the tests feed each check, and what it must say: (exercise, code or choices, state, known mistake).
CASES = [
    ("hits", _HI_START, "empty", None),
    ("hits", _HI, "correct", None),
    ("hits", _HI_HEAD + "        if plain(...) in query:\n" + _HI_END, "wrong", "dots-left"),
    ("hits", _HI_EXACT, "wrong", "exact"),
    ("hits", _HI_LOWER, "wrong", "lower"),
    ("hits", _HI_HEAD + "        if query.count(plain(word)) > 0:\n" + _HI_END, "error", None),
    ("hits", "def hits(words, query):\n    return len([w for w in words if plain(w) in query])\n", "correct", None),
    ("hits", "def hits(words, query):\n    return len({plain(w) for w in words if plain(w) in query})\n", "wrong", None),
    ("hits", _HI_HEAD + "        if plain(word) in query and plain(word) != \"\":\n" + _HI_END, "correct", None),
    ("best_start", _BS_START, "empty", None),
    ("best_start", _BS, "correct", None),
    ("best_start", _BS_HEAD + "        count = hits(words[...], query)\n" + _BS_END, "wrong", "dots-left"),
    ("best_start", _BS_LENGTH, "wrong", "length"),
    ("best_start", _BS_SHORT, "wrong", "short"),
    ("best_start", _BS_HEAD + "        count = hits(words[start:size + start], query)\n" + _BS_END, "correct", None),
    ("best_start", _BS_HEAD + "        count = hits(words[start:start + size], query)\n"
     "        if count >= most:\n            best, most = start, count\n    return best\n", "almost", None),
    ("best_start", "def best_start(words, query, size):\n    best, most = 0, 0\n    for start in range(len(words) - size + 1):\n"
     "        count = hits(words[start:start + size], query)\n        if count > most:\n            best = start\n"
     "    return best\n", "almost", None),
    ("snippet", _SN_START, "empty", None),
    ("snippet", _SN, "correct", None),
    ("snippet", _SN_HEAD + '            word = "**" + ... + "**"\n' + _SN_END, "wrong", "dots-left"),
    ("snippet", _SN_PLAIN, "wrong", "plain"),
    ("snippet", _SN_OPEN, "wrong", "open"),
    ("snippet", _SN_HTML, "wrong", "html"),
    ("snippet", _SN_HEAD + '            word = f"**{word}**"\n' + _SN_END, "correct", None),
    ("snippet", _SN_HEAD + '            word = "*" + word + "*"\n' + _SN_END, "wrong", None),
    ("kinds", {}, "empty", None),
    ("kinds", dict(EXERCISES["kinds"].solution), "correct", None),
    ("kinds", {"first": "static", "window": "dynamic"}, "wrong", "unfinished"),
] + [("kinds", {item: ("static" if group == "dynamic" else "dynamic")}, "wrong", f"wrong:{item}")
     for item, group in EXERCISES["kinds"].solution.items()]
