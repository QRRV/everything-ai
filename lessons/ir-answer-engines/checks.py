# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ir-answer-engines: a search engine that writes one answer with sources instead of a page of
ten links. The lesson measures what a page of links offers with success@k, the share of questions with a relevant
document among the first k results (success_at_k), and what an answer engine gives with the course's own extractive
answer engine: it reads the first results of BM25 and answers with the sentence with the highest query score of
Tombros and Sanderson (as in the lesson ir-snippets), citing the result it comes from. The citations of an answer are
measured with citation recall and citation precision of Liu, Zhang and Liang, Evaluating Verifiability in Generative
Search Engines (Findings of EMNLP 2023), without their partial support: a sentence counts as supported when one of its
citations supports it (citation_recall), and a citation counts when it supports its sentence (citation_precision). The
sort asks what belongs to a page of ten links and what to one answer.

The paragraphs and questions are those of the lessons ir-bm25, ir-precision-at-k and ir-snippets (the 2,067 paragraphs
and the 1,190 English and 1,183 Dutch questions of XQuAD, everything_ai.xquad), ranked by BM25 as in ir-bm25; the
relevant paragraph of a question is its own paragraph. An answer shows the answer when one of the answers of XQuAD lies
inside its sentence of the own paragraph. Every number in the text and the pictures is computed by the course
(tests/test_ir_answer_engines.py computes them again)."""
import html
import importlib.util

from everything_ai import paths, xquad, yours
from everything_ai.check import FunctionBox, Mistake, Sort, kept

LANGS = ("en", "nl")


def _load(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


N = _load("ir-snippets", "everything_ai_checks_ir_snippets_for_answer_engines")             # results, sentences, query score
P = _load("ir-precision-at-k", "everything_ai_checks_ir_precision_for_answer_engines")      # the rankings of BM25
V, S = N.V, N.S         # the vector space model and the parts of the pictures and the exercises

LINKS = 10              # the links of a page of results
READ = 3                # the results the course's answer engine reads in the pictures
KS = (1, 2, 3, 5, 10)   # the values of k in the pictures
EXAMPLE = N.EXAMPLE     # What document formed the Parliament of Victoria? (ir-vector-space, ir-bm25, ir-snippets)
IDEA = "5727c94bff5b5019007d954b"   # the population of Jacksonville in 2010: its own paragraph comes second


# --- the course's solutions, the same code as the exercises -------------------------------------------------------------------

def success_at_k(ranking, relevant, k):
    """True when one of the first k documents of the ranking is relevant."""
    for doc in ranking[:k]:
        if doc in relevant:
            return True
    return False


def citation_recall(cited, supports):
    """The share of the sentences of an answer that one of their citations supports: cited[i] lists the sources that
    sentence i cites, supports[i] the sources that support it. A sentence without a citation is not supported."""
    supported = 0
    for i in range(len(cited)):
        found = False
        for source in cited[i]:
            if source in supports[i]:
                found = True
        if found:
            supported = supported + 1
    return supported / len(cited)


def citation_precision(cited, supports):
    """The share of the citations of an answer that support their sentence; 0 for an answer without citations."""
    good, total = 0, 0
    for i in range(len(cited)):
        for source in cited[i]:
            if source in supports[i]:
                good = good + 1
        total = total + len(cited[i])
    if total == 0:
        return 0.0
    return good / total


# --- the questions of XQuAD, their rankings and the course's answer engine ------------------------------------------------------

def question_place(qid: str, lang: str) -> int:
    """The place of a question in xquad.questions(lang), by its id."""
    return N.question_place(qid, lang)


def _links(lang: str) -> tuple:
    return kept(f"ir-answer-engines:links:{lang}", lambda: tuple(tuple(r[:LINKS]) for r in P.rankings(lang, "bm25")))


def links(lang: str, k: int) -> list:
    """The paragraphs of the page of ten links of question k: the first ten of BM25, best first, as a new list."""
    return list(_links(lang)[k])


def ranking_of(lang: str, k: int) -> list:
    """The first 100 paragraphs of BM25 for question k, best first, as a new list (ir-precision-at-k)."""
    return P.rankings(lang, "bm25")[k]


def relevant_of(lang: str, k: int) -> list:
    """The relevant paragraphs of question k: a list with its own paragraph."""
    return [xquad.questions(lang)[k].paragraph]


def _picks(lang: str) -> list:
    """For each question, for each number of results read from 1 to 10: the (place, paragraph, start, end) of the
    sentence the engine answers with, the sentence with the highest query score, the first one on a tie."""
    def make():
        sentences = {}   # for each paragraph: each sentence as (start, end, its plain words)
        out = []
        for k in range(len(xquad.questions(lang))):
            query = N._queries(lang)[k]
            best, row = None, []
            for place, i in enumerate(_links(lang)[k], 1):
                if i not in sentences:
                    words = N._paragraph(lang, i)[0]
                    sentences[i] = [(a, b, frozenset(N.plain(w) for w in words[a:b]))
                                    for a, b in N.sentence_spans(words, lang)]
                for a, b, plain in sentences[i]:
                    score = len(plain & query) ** 2 / len(query) if query else 0.0
                    if best is None or score > best[0]:
                        best = (score, place, i, a, b)
                row.append(best[1:])
            out.append(tuple(row))
        return out
    return kept(f"ir-answer-engines:picks:{lang}", make)


def engine(lang: str, k: int, read: int = READ) -> tuple:
    """The (place, paragraph, start, end) of the course's answer to question k, from the first `read` results."""
    return _picks(lang)[k][min(max(read, 1), LINKS) - 1]


def sentence_of(lang: str, k: int, read: int = READ) -> str:
    """The sentence the engine answers question k with."""
    _, i, a, b = engine(lang, k, read)
    return " ".join(N._paragraph(lang, i)[0][a:b])


def answer_of(lang: str, k: int, read: int = READ) -> str:
    """The course's answer to question k from the first `read` results: its sentence and the number of its source."""
    return f"{sentence_of(lang, k, read)} [{engine(lang, k, read)[0]}]"


def answer_right(lang: str, k: int, read: int = READ) -> bool:
    """True when the answer's sentence comes from the own paragraph and holds one of the answers of XQuAD."""
    _, i, a, b = engine(lang, k, read)
    return i == xquad.questions(lang)[k].paragraph and N.shows_answer(lang, k, a, b)


def engine_share(lang: str, read: int = READ) -> float:
    """The share of the questions whose answer from the first `read` results shows the answer."""
    return kept(f"ir-answer-engines:engine:{lang}:{read}",
                lambda: sum(answer_right(lang, k, read) for k in range(len(xquad.questions(lang)))) / len(xquad.questions(lang)))


def success_share(lang: str, k: int) -> float:
    """success@k of BM25 over all the questions: the share with the own paragraph among the first k results."""
    def make():
        rankings = P.rankings(lang, "bm25")
        return sum(success_at_k(r, relevant_of(lang, j), k) for j, r in enumerate(rankings)) / len(rankings)
    return kept(f"ir-answer-engines:success:{lang}:{k}", make)


def misses(lang: str, read: int = READ) -> tuple:
    """How many answers from the first `read` results miss the answer, and how many of those questions have their own
    paragraph among the ten links: an alternative that the page of links would have shown."""
    def make():
        qs = xquad.questions(lang)
        missed = [k for k in range(len(qs)) if not answer_right(lang, k, read)]
        return len(missed), sum(qs[k].paragraph in links(lang, k) for k in missed)
    return kept(f"ir-answer-engines:misses:{lang}:{read}", make)


# The answers of the exercises and pictures, as (cited, supports): for each sentence, the sources it cites and the sources
# that support it. Two are the examples of Liu, Zhang and Liang (2023, figure 2 (a) and (c)), with partial support
# left out: in (a) the partly supporting citations count as not supporting, which gives their own 3/3 and 3/8.
ANSWERS = {
    "toy": ([[1], [1, 2], [2, 3], []], [[1], [1], [4, 5], [2, 6]]),
    "every": ([[1, 2, 3, 4]], [[1, 2]]),
    "liu_a": ([[1, 2, 3], [1, 2, 4], [4, 5]], [[1], [1], [4]]),
    "liu_c": ([[1, 2, 3], [], []], [[1, 2], [], []]),
}
LIU = {"liu_a": ("3/3", "3/8"), "liu_c": ("1/3", "2/3")}   # citation recall and precision as Liu and colleagues give them


# --- the pictures ----------------------------------------------------------------------------------------------------------

_UI, _LABEL, _CELL, _HEAD, _NUMBER = S._UI, S._LABEL, S._CELL, S._HEAD, S._NUMBER
_DATA, _FAINT, _PRED, _LOSS = S._DATA, S._FAINT, S._PRED, S._LOSS
_CHIP_PRED, _CHIP_DATA = S._CHIP_PRED, S._CHIP_DATA
_note, _colors, _label, _num, _dec, _pct = S._note, S._colors, S._label, S._num, S._dec, S._pct
_cell, _table = S._cell, S._table
_fn, _call, _empty, _SLOW, _STOPPED = S._fn, S._call, S._empty, S._SLOW, S._STOPPED
_WHY_EMPTY = S._WHY_EMPTY
_is_number = S._is_number
_chip = N._chip

_TEXT = "font-family:var(--ea-font-ui);font-size:0.85rem;line-height:1.45"
_ANSWER = _TEXT + ";margin:0.2rem 0 0.3rem;padding:0.35rem 0.6rem;border-left:3px solid var(--ea-line)"


class _Odd(Exception):
    """A learner's function gave something the picture cannot draw."""


def _guarded(make):
    """make() within the time limit of a picture; None when the learner's function gives something odd."""
    def run():
        try:
            return make()
        except _Odd:
            return None
    return _call(run)


def _why_empty(w: dict, got):
    """The empty picture that says why, or None when there is something to draw."""
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    if got is None:
        return _empty(w, "odd")
    return None


_WORDS = {
    "en": {"paragraph": "paragraph", "own": "own paragraph", "yes": "yes", "no": "no",
           "links": "The page of ten links, the first {n} results of BM25:",
           "answer": "The answer engine, after reading {what}:",
           "first_one": "the first result", "first_n": "the first {n} results",
           "shows": "The sentence comes from the own paragraph and holds the answer: {a}.",
           "on_page": "The own paragraph is among the ten links: {a}.",
           "head_all": ["Results read", "success@k", "Right answers"],
           "sentence": "Sentence {i}", "cites": "Cites", "supported": "Supported by", "nothing": "nothing"},
    "nl": {"paragraph": "alinea", "own": "eigen alinea", "yes": "ja", "no": "nee",
           "links": "De pagina met tien links, de eerste {n} resultaten van BM25:",
           "answer": "De answer engine, na het lezen van {what}:",
           "first_one": "het eerste resultaat", "first_n": "de eerste {n} resultaten",
           "shows": "De zin komt uit de eigen alinea en bevat het antwoord: {a}.",
           "on_page": "De eigen alinea staat bij de tien links: {a}.",
           "head_all": ["Gelezen resultaten", "success@k", "Goede antwoorden"],
           "sentence": "Zin {i}", "cites": "Citeert", "supported": "Gesteund door", "nothing": "niets"},
}


def _read_label(lang: str, read: int) -> str:
    """The label above an answer: after reading the first result, or the first n results."""
    w = _WORDS[lang]
    return _label(w["answer"].format(what=w["first_one"] if read == 1 else w["first_n"].format(n=read)))


def _links_html(lang: str, k: int) -> str:
    return "".join(N._result_head(lang, k, place, i) for place, i in enumerate(links(lang, k), 1))


def _answer_html(lang: str, k: int, read: int) -> str:
    """The course's answer from the first `read` results: blue when it shows the answer, purple when not."""
    right = answer_right(lang, k, read)
    style = _PRED if right else _LOSS
    place = engine(lang, k, read)[0]
    return (f'<div style="{_ANSWER};{style.replace(";font-weight:600", "")}">{html.escape(sentence_of(lang, k, read))} '
            f'<span style="{_CHIP_DATA}">[{place}]</span></div>')


def _all_rows(lang: str, reads, chosen: int = 0) -> list:
    """Over all the questions, for each number of results read: success@k of BM25 and the share of right answers."""
    rows = []
    for read in reads:
        style = _PRED if not chosen or read == chosen else _DATA
        rows.append("<tr>" + _cell(str(read), _FAINT, _NUMBER) + _cell(html.escape(_pct(success_share(lang, read), lang)), style, _NUMBER)
                    + _cell(html.escape(_pct(engine_share(lang, read), lang)), style, _NUMBER) + "</tr>")
    return rows


_IDEA_TEXT = {
    "en": {"title": "Ten links or one answer",
           "all": "Over all {n} questions: success@k for the first results, and the share of right answers of the engine that reads them.",
           "missed": "Of the {m} questions with a wrong answer after the first {read} results, {a} have their own paragraph among the ten links: {p}.",
           "gray": "Gray: the question, the ten links, and the number of the source each answer cites.",
           "blue": "Blue: a right answer, and the measures over all the questions.",
           "purple": "Purple: a wrong answer."},
    "nl": {"title": "Tien links of één antwoord",
           "all": "Over alle {n} vragen: success@k bij de eerste resultaten, en het deel goede antwoorden van de engine die deze resultaten leest.",
           "missed": "Van de {m} vragen met een fout antwoord na de eerste {read} resultaten staat bij {a} de eigen alinea bij de tien links: {p}.",
           "gray": "Grijs: de vraag, de tien links, en bij elk antwoord het nummer van de geciteerde bron.",
           "blue": "Blauw: een goed antwoord, en de maten over alle vragen.",
           "purple": "Paars: een fout antwoord."},
}


def idea_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang] | _WORDS[lang]
    k = question_place(IDEA, lang)
    body = N._question_html(lang, k) + _label(w["links"].format(n=LINKS)) + _links_html(lang, k)
    for read in (1, READ):
        body += _read_label(lang, read) + _answer_html(lang, k, read)
    n = len(xquad.questions(lang))
    body += _label(w["all"].format(n=_num(n, lang))) + _table(_all_rows(lang, (1, READ, LINKS)), w["head_all"], (0, 1, 2))
    m, a = misses(lang, READ)
    body += _label(w["missed"].format(m=_num(m, lang), read=READ, a=_num(a, lang), p=_pct(a / m, lang)))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


_SUCCESS_TEXT = {
    "en": {"title": "Your success@k: does the page hold the own paragraph?", "kind": "True or False",
           "example": "The question about Victoria: the first {n} results of BM25, with the own paragraph {i}.",
           "head_ex": ["k", "Your success_at_k"],
           "all": "Over all {n} questions, with the rankings of BM25:",
           "head_all": ["k", "Your success@k", "The course"],
           "gray": "Gray: the results, and the course's values.",
           "blue": "Blue: your values where they are the course's.",
           "purple": "Purple: your values where they differ from the course's."},
    "nl": {"title": "Je success@k: bevat de pagina de eigen alinea?", "kind": "True of False",
           "example": "De vraag over Victoria: de eerste {n} resultaten van BM25, met de eigen alinea {i}.",
           "head_ex": ["k", "Jouw success_at_k"],
           "all": "Over alle {n} vragen, met de rangschikkingen van BM25:",
           "head_all": ["k", "Jouw success@k", "De course"],
           "gray": "Grijs: de resultaten, en de waarden van de course.",
           "blue": "Blauw: je waarden waar deze gelijk zijn aan de waarden van de course.",
           "purple": "Paars: je waarden waar deze verschillen van de waarden van de course."},
}


def _bool(x) -> bool:
    if not isinstance(x, bool):
        raise _Odd
    return x


def success_at_k_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _SUCCESS_TEXT[lang] | _WHY_EMPTY[lang] | _WORDS[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    k = question_place(EXAMPLE, lang)
    rankings = P.rankings(lang, "bm25")

    def make():
        example = [_bool(fn(ranking_of(lang, k), relevant_of(lang, k), j)) for j in (1, 2, 3)]
        shares = {j: sum(_bool(fn(list(r), relevant_of(lang, q), j)) for q, r in enumerate(rankings)) / len(rankings)
                  for j in KS}
        return example, shares

    got = _guarded(make)
    empty = _why_empty(w, got)
    if empty:
        return empty
    example, shares = got
    own = xquad.questions(lang)[k].paragraph
    body = N._question_html(lang, k) + _label(w["example"].format(n=3, i=str(own)))
    body += "".join(N._result_head(lang, k, place, i) for place, i in enumerate(links(lang, k)[:3], 1))
    rows = []
    for j, mine in zip((1, 2, 3), example):
        want = success_at_k(ranking_of(lang, k), relevant_of(lang, k), j)
        text_ = f'<span style="{_PRED if mine is want else _LOSS}">{mine}</span>'
        if mine is not want:
            text_ += f' <span style="{_DATA}">({want})</span>'
        rows.append("<tr>" + _cell(str(j), _FAINT, _NUMBER) + _cell(text_) + "</tr>")
    body += _table(rows, w["head_ex"], (0,))
    rows = []
    for j in KS:
        mine, want = shares[j], success_share(lang, j)
        same = abs(mine - want) < 1e-9
        rows.append("<tr>" + _cell(str(j), _FAINT, _NUMBER) + _cell(html.escape(_pct(mine, lang)), _PRED if same else _LOSS, _NUMBER)
                    + _cell(html.escape(_pct(want, lang)), _DATA, _NUMBER) + "</tr>")
    body += _label(w["all"].format(n=_num(len(rankings), lang))) + _table(rows, w["head_all"], (0, 1, 2))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


_CITE_TEXT = {
    "en": {"names": {"toy": "The answer of the first example of the exercise",
                     "every": "One sentence that cites every source: the second example of the exercise",
                     "liu_a": "Liu, Zhang and Liang, example (a)",
                     "liu_c": "Liu, Zhang and Liang, example (c)"},
           "liu": "Liu, Zhang and Liang: {x}"},
    "nl": {"names": {"toy": "Het antwoord van het eerste voorbeeld van de oefening",
                     "every": "Eén zin met elke bron als citatie: het tweede voorbeeld van de oefening",
                     "liu_a": "Liu, Zhang en Liang, voorbeeld (a)",
                     "liu_c": "Liu, Zhang en Liang, voorbeeld (c)"},
           "liu": "Liu, Zhang en Liang: {x}"},
}

_RECALL_TEXT = {
    "en": {"title": "Your citation recall: which sentences do their citations support?", "kind": "a number",
           "mine": "Your citation recall: {x}",
           "gray": "Gray: the sentences, their citations and their support, and the course's value where yours differs.",
           "blue": "Blue: your citation recall where it is the course's.",
           "purple": "Purple: your citation recall where it differs from the course's."},
    "nl": {"title": "Je citation recall: welke zinnen worden door de citaties gesteund?", "kind": "een getal",
           "mine": "Jouw citation recall: {x}",
           "gray": "Grijs: de zinnen, de citaties en de steun, en de waarde van de course waar de jouwe verschilt.",
           "blue": "Blauw: je citation recall waar deze gelijk is aan de waarde van de course.",
           "purple": "Paars: je citation recall waar deze verschilt van de waarde van de course."},
}

_PRECISION_TEXT = {
    "en": {"title": "Your citation precision: which citations support their sentence?", "kind": "a number",
           "mine": "Your citation precision: {x}, with citation recall {r}",
           "gray": "Gray: the sentences, their citations and their support, the citation recall, and the course's value where yours differs.",
           "blue": "Blue: your citation precision where it is the course's.",
           "purple": "Purple: your citation precision where it differs from the course's."},
    "nl": {"title": "Je citation precision: welke citaties steunen de zin?", "kind": "een getal",
           "mine": "Jouw citation precision: {x}, met citation recall {r}",
           "gray": "Grijs: de zinnen, de citaties en de steun, de citation recall, en de waarde van de course waar de jouwe verschilt.",
           "blue": "Blauw: je citation precision waar deze gelijk is aan de waarde van de course.",
           "purple": "Paars: je citation precision waar deze verschilt van de waarde van de course."},
}


def _sources(numbers, lang: str) -> str:
    if not numbers:
        return f'<span style="{_FAINT}">{html.escape(_WORDS[lang]["nothing"])}</span>'
    return "".join(_chip(f"[{n}]") for n in numbers)


def _answer_table(key: str, lang: str) -> str:
    w = _WORDS[lang]
    cited, supports = ANSWERS[key]
    rows = ["<tr>" + _cell(html.escape(w["sentence"].format(i=i + 1)), _FAINT) + _cell(_sources(c, lang))
            + _cell(_sources(s, lang)) + "</tr>" for i, (c, s) in enumerate(zip(cited, supports))]
    return _table(rows, ["", w["cites"], w["supported"]])


def _share(x) -> float:
    if not _is_number(x):
        raise _Odd
    return float(x)


def _mine(got: float, want: float, lang: str) -> str:
    """A learner's share with three decimals: blue when it is the course's, purple with the course's after it."""
    if abs(got - want) < 1e-9:
        return f'<span style="{_PRED}">{_dec(got, lang, 3)}</span>'
    return f'<span style="{_LOSS}">{_dec(got, lang, 3)}</span> <span style="{_DATA}">({_dec(want, lang, 3)})</span>'


def _cite_picture(seen, texts: dict, course, recall: bool) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = texts[lang] | _WHY_EMPTY[lang] | _CITE_TEXT[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")

    def make():
        out = {}
        for key, (cited, supports) in ANSWERS.items():
            out[key] = _share(fn([list(c) for c in cited], [list(s) for s in supports]))
        return out

    got = _guarded(make)
    empty = _why_empty(w, got)
    if empty:
        return empty
    body = ""
    for key, (cited, supports) in ANSWERS.items():
        body += _label(w["names"][key]) + _answer_table(key, lang)
        want = course(cited, supports)
        if recall:
            line = w["mine"].format(x=_mine(got[key], want, lang))
        else:
            line = w["mine"].format(x=_mine(got[key], want, lang), r=_dec(citation_recall(cited, supports), lang, 3))
        if key in LIU:
            line += f' <span style="{_FAINT}">({html.escape(w["liu"].format(x=LIU[key][0 if recall else 1]))})</span>'
        body += f'<div style="{_LABEL}">{line}</div>'
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


def citation_recall_picture(seen) -> str:
    return _cite_picture(seen, _RECALL_TEXT, citation_recall, True)


def citation_precision_picture(seen) -> str:
    return _cite_picture(seen, _PRECISION_TEXT, citation_precision, False)


# --- the demo: one question, its ten links, and the answer from the first results ---------------------------------------------

TRY_QUESTIONS = (EXAMPLE, IDEA, "5706149552bb891400689881", "570966e0200fba1400367f52",
                 "57339c16d058e614000b5ec5")
# the parliament of Victoria; the population of Jacksonville in 2010; how many teams Los Angeles used to have; the
# system Sky+ HD uses; the Saxon Garden in Polish


def try_options(lang: str) -> dict:
    """The questions of the demo's dropdown, by their text, to their id."""
    qs = xquad.questions(lang)
    return {qs[question_place(qid, lang)].question.strip(): qid for qid in TRY_QUESTIONS}


_TRY_TEXT = {
    "en": {"title": "Your answer engine",
           "all": "Over all {n} questions: success@k and the share of right answers, for the number of results you chose and for ten.",
           "gray": "Gray: the question, the ten links, the number of the source the answer cites, and the row for ten results when you chose fewer.",
           "blue": "Blue: a right answer, and the measures for the number of results you chose.",
           "purple": "Purple: a wrong answer."},
    "nl": {"title": "Jouw answer engine",
           "all": "Over alle {n} vragen: success@k en het deel goede antwoorden, bij het aantal resultaten dat je koos en bij tien.",
           "gray": "Grijs: de vraag, de tien links, het nummer van de geciteerde bron, en de rij bij tien resultaten als je er minder koos.",
           "blue": "Blauw: een goed antwoord, en de maten bij het aantal resultaten dat je koos.",
           "purple": "Paars: een fout antwoord."},
}


def _try_read(read) -> int:
    try:
        read = int(read)
    except (TypeError, ValueError):
        return READ
    return min(max(read, 1), LINKS)


def try_picture(lang: str, qid: str = EXAMPLE, read=READ) -> str:
    lang = lang if lang in LANGS else "en"
    w = _TRY_TEXT[lang] | _WORDS[lang]
    qid = qid if qid in TRY_QUESTIONS else EXAMPLE
    read = _try_read(read)
    k = question_place(qid, lang)
    body = N._question_html(lang, k)
    body += _read_label(lang, read) + _answer_html(lang, k, read)
    body += _label(w["shows"].format(a=w["yes"] if answer_right(lang, k, read) else w["no"]))
    body += _label(w["links"].format(n=LINKS)) + _links_html(lang, k)
    own_on_page = xquad.questions(lang)[k].paragraph in links(lang, k)
    body += _label(w["on_page"].format(a=w["yes"] if own_on_page else w["no"]))
    reads = (read,) if read == LINKS else (read, LINKS)
    body += (_label(w["all"].format(n=_num(len(xquad.questions(lang)), lang)))
             + _table(_all_rows(lang, reads, read), w["head_all"], (0, 1, 2)))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------------

_Val, _DOTS, _SAME_TEXTS = S._Val, S._DOTS, S._SAME_TEXTS
_show, _written = V._show, V._written

_SA_HEAD = "def success_at_k(ranking, relevant, k):\n"
_SA_END = "        if doc in relevant:\n            return True\n    return False\n"
_SA = _SA_HEAD + "    for doc in ranking[:k]:\n" + _SA_END
_SA_START = _SA_HEAD + "    for doc in ...:\n" + _SA_END
_SA_ALL = _SA_HEAD + "    for doc in ranking:\n" + _SA_END
_SA_SHORT = _SA_HEAD + "    for doc in ranking[:k - 1]:\n" + _SA_END

_CR_HEAD = ("def citation_recall(cited, supports):\n    supported = 0\n    for i in range(len(cited)):\n"
            "        found = False\n        for source in cited[i]:\n")
_CR_END = ("                found = True\n        if found:\n            supported = supported + 1\n"
           "    return supported / len(cited)\n")
_CR = _CR_HEAD + "            if source in supports[i]:\n" + _CR_END
_CR_START = _CR_HEAD + "            if ...:\n" + _CR_END
_CR_PAIR = _CR_HEAD + "            if source in supports:\n" + _CR_END
_CR_CITED = _CR_HEAD + "            if source in cited[i]:\n" + _CR_END

_CP_HEAD = ("def citation_precision(cited, supports):\n    good, total = 0, 0\n    for i in range(len(cited)):\n"
            "        for source in cited[i]:\n            if source in supports[i]:\n                good = good + 1\n")
_CP_END = "    if total == 0:\n        return 0.0\n    return good / total\n"
_CP = _CP_HEAD + "        total = total + len(cited[i])\n" + _CP_END
_CP_START = _CP_HEAD + "        total = ...\n" + _CP_END
_CP_SENTENCES = _CP_HEAD + "        total = total + 1\n" + _CP_END
_CP_SUPPORTS = _CP_HEAD + "        total = total + len(supports[i])\n" + _CP_END

# The small cases the exercise texts show; the same in both lessons.
TOY_SA = ((_written([511, 513, 526]), _written([513]), _written(1)), (_written([511, 513, 526]), _written([513]), _written(2)))
TOY_CITE = tuple((_written(c), _written(s)) for c, s in (ANSWERS["toy"], ANSWERS["every"]))


def _course_ranking(lang: str, qid: str) -> tuple:
    k = question_place(qid, lang)
    return (_Val(f'C.ranking_of("{lang}", {k})', ranking_of(lang, k)), _Val(f'C.relevant_of("{lang}", {k})', relevant_of(lang, k)))


REQUIRED = ["success_at_k", "citation_recall", "citation_precision", "pages"]

EXERCISES = {
    "success_at_k": FunctionBox(
        name="success_at_k",
        picture=success_at_k_picture,
        function="success_at_k",
        start=_SA_START,
        solution=_SA,
        show=_show,
        visible=list(TOY_SA),
        hidden=[(_written([511, 513, 526]), _written([513]), _written(3)), (_written([4, 7]), _written([9]), _written(2)),
                (_written([]), _written([1]), _written(3)), (_written([5, 6]), _written([6]), _written(10)),
                (_written([2, 8, 3, 9]), _written([9, 3]), _written(3)), (_written([2, 8]), _written([2]), _written(0)),
                (_written([7, 4]), _written([7]), _written(1)), (*_course_ranking("nl", EXAMPLE), _written(1)),
                (*_course_ranking("en", EXAMPLE), _written(1)), (*_course_ranking("nl", "5706149552bb891400689881"), _written(1)),
                (*_course_ranking("en", "5706149552bb891400689881"), _written(10))],
        first=[_DOTS],
        mistakes=[
            Mistake("all", wrong=_SA_ALL,
                    en="`{call}` gives {got}, and it should give {want}: your loop goes through the whole ranking, so it finds a relevant document at any place, also after place `k`. Which part of the ranking should the loop go through?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je loop gaat door de hele rangschikking, dus de functie vindt een relevant document op elke plaats, ook na plaats `k`. Door welk deel van de rangschikking moet de loop gaan?"),
            Mistake("short", wrong=_SA_SHORT,
                    en="`{call}` gives {got}, and it should give {want}: your slice holds one document less than `k`, because a slice stops before the place after the colon. Which number should come after the colon, so that the slice holds the first `k` documents?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je slice bevat één document minder dan `k`, want een slice stopt bij de plaats na de dubbele punt, zonder deze plaats zelf. Welk getal moet na de dubbele punt staan, zodat de slice de eerste `k` documenten bevat?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the first `k` documents of `ranking`.",
                      "nl": "Vervang de `...` door de eerste `k` documenten van `ranking`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does the loop go through the first `k` documents of `ranking`, and no more?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Gaat de loop door de eerste `k` documenten van `ranking`, en niet meer?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function also work for an empty ranking, for a `k` larger than the ranking, for a `k` of 0, and for the course's rankings?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook voor een lege rangschikking, voor een `k` groter dan de rangschikking, voor een `k` van 0, en voor de rangschikkingen van de course?"},
            "correct": {"en": "Right: your function tells whether the first `k` results hold a relevant document. In the picture below, how much more often does a page of ten links hold the own paragraph than the first result alone?",
                        "nl": "Goed: je functie zegt of de eerste `k` resultaten een relevant document bevatten. In het plaatje hieronder: hoeveel vaker bevat een pagina met tien links het relevante document dan het eerste resultaat alleen?"},
        },
    ),
    "citation_recall": FunctionBox(
        name="citation_recall",
        picture=citation_recall_picture,
        function="citation_recall",
        start=_CR_START,
        solution=_CR,
        show=_show,
        visible=[TOY_CITE[0], TOY_CITE[1]],
        hidden=[tuple(_written(x) for x in ANSWERS["liu_a"]), tuple(_written(x) for x in ANSWERS["liu_c"]),
                (_written([[], []]), _written([[1], [2]])), (_written([[2, 2]]), _written([[2]])),
                (_written([[3], [1]]), _written([[1], [3]])), (_written([[5]]), _written([[5]]))],
        first=[_DOTS],
        mistakes=[
            Mistake("pair", wrong=_CR_PAIR,
                    en="`{call}` gives {got}, and it should give {want}: `supports` is a list with a list for each sentence, so a number such as `1` is never one of its items. Which list of `supports` belongs to sentence `i`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `supports` is een lijst met een lijst voor elke zin, dus een getal zoals `1` is nooit een van de items. Welke lijst van `supports` hoort bij zin `i`?"),
            Mistake("cited", wrong=_CR_CITED,
                    en="`{call}` gives {got}, and it should give {want}: your test asks whether the sentence cites the source, and that is always so. A citation is not yet support. Which list says which sources support sentence `i`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je test vraagt of de zin de bron citeert, en dat is altijd zo. Een citatie is nog geen steun. Welke lijst zegt welke bronnen zin `i` steunen?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the test: does `source` support sentence `i`?",
                      "nl": "Vervang de `...` door de test: steunt `source` zin `i`?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your test look up `source` in the sources that support sentence `i`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Zoekt je test `source` op bij de bronnen die steun geven aan zin `i`?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function also work for the examples of Liu, Zhang and Liang, for an answer without citations, and for a source cited twice?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook voor de voorbeelden van Liu, Zhang en Liang, voor een antwoord zonder citaties, en voor een dubbel geciteerde bron?"},
            "correct": {"en": "Right: your function measures which sentences their citations support. In the picture below, which answer gets a citation recall of 1 while half of its citations support nothing?",
                        "nl": "Goed: je functie meet welke zinnen de citaties steunen. In het plaatje hieronder: welk antwoord krijgt een citation recall van 1, terwijl de helft van de citaties niets steunt?"},
        },
    ),
    "citation_precision": FunctionBox(
        name="citation_precision",
        picture=citation_precision_picture,
        function="citation_precision",
        start=_CP_START,
        solution=_CP,
        show=_show,
        visible=[TOY_CITE[0], TOY_CITE[1]],
        hidden=[tuple(_written(x) for x in ANSWERS["liu_a"]), tuple(_written(x) for x in ANSWERS["liu_c"]),
                (_written([[], []]), _written([[1], [2]])), (_written([[2, 2]]), _written([[2]])),
                (_written([[3], [1, 2]]), _written([[1, 3], [2]])), (_written([[5]]), _written([[5]]))],
        first=[_DOTS],
        mistakes=[
            Mistake("sentences", wrong=_CP_SENTENCES,
                    en="`{call}` gives {got}, and it should give {want}: `total` grows by 1 for each sentence, so your function divides by the number of sentences. Citation precision counts citations: how many citations does sentence `i` have?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `total` groeit met 1 bij elke zin, dus je functie deelt door het aantal zinnen. Citation precision telt citaties: hoeveel citaties heeft zin `i`?"),
            Mistake("supports", wrong=_CP_SUPPORTS,
                    en="`{call}` gives {got}, and it should give {want}: `len(supports[i])` counts the sources that support sentence `i`, also the ones it does not cite. Citation precision divides by what the answer cites: which list holds the citations of sentence `i`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `len(supports[i])` telt de bronnen die steun geven aan zin `i`, ook de bronnen die niet in de zin geciteerd worden. Citation precision deelt door wat het antwoord citeert: welke lijst bevat de citaties van zin `i`?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the new `total`: the citations so far, plus the citations of sentence `i`.",
                      "nl": "Vervang de `...` door de nieuwe `total`: de citaties tot nu toe, plus de citaties van zin `i`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does `total` grow by the number of citations of each sentence?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Groeit `total` bij elke zin met het aantal citaties van de zin?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function also work for the examples of Liu, Zhang and Liang, for an answer without citations, and for a source cited twice?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook voor de voorbeelden van Liu, Zhang en Liang, voor een antwoord zonder citaties, en voor een dubbel geciteerde bron?"},
            "correct": {"en": "Right: your function measures which citations support their sentence. In the picture below, which answer has a higher citation precision than citation recall, and why?",
                        "nl": "Goed: je functie meet welke citaties de zin steunen. In het plaatje hieronder: welk antwoord heeft een hogere citation precision dan citation recall, en waarom?"},
        },
    ),
}

# The sort: what belongs to a page of ten links, and what to one answer.
EXERCISES["pages"] = Sort(
    name="pages",
    groups={
        "links": {"en": "ten links", "nl": "tien links"},
        "answer": {"en": "one answer", "nl": "één antwoord"},
    },
    items={
        "alternatives": {"en": "**Alternatives.** When the first result is wrong, the next results stand right below it.",
                         "nl": "**Alternatieven.** Als het eerste resultaat fout is, staan de volgende resultaten er meteen onder."},
        "no_click": {"en": "**No click.** The page writes the answer out in full, so the reader opens no source.",
                     "nl": "**Geen klik.** De pagina schrijft het antwoord helemaal uit, dus de lezer opent geen bron."},
        "visit": {"en": "**The visit.** The reader's visit goes to the site that wrote the text.",
                  "nl": "**Het bezoek.** Het bezoek van de lezer gaat naar de site die de tekst schreef."},
        "merged": {"en": "**One text.** Sentences from several sources come together in one text.",
                   "nl": "**Eén tekst.** Zinnen uit meerdere bronnen komen samen in één tekst."},
        "judge": {"en": "**Judging.** The reader judges each source by its title and its snippet.",
                  "nl": "**Beoordelen.** De lezer beoordeelt elke bron op de titel en de snippet."},
        "alone": {"en": "**Alone.** A wrong answer arrives without other results next to it.",
                  "nl": "**Alleen.** Een fout antwoord komt zonder andere resultaten ernaast."},
    },
    solution={"alternatives": "links", "no_click": "answer", "visit": "links", "merged": "answer", "judge": "links",
              "alone": "answer"},
    nudge={
        "alternatives": {"en": "Where do the other results stand when the first one is wrong?",
                         "nl": "Waar staan de andere resultaten als het eerste fout is?"},
        "no_click": {"en": "Which page writes the answer out, so that no source needs opening?",
                     "nl": "Welke pagina schrijft het antwoord uit, zodat geen bron open hoeft?"},
        "visit": {"en": "Who gets the visit when the reader opens a result?",
                  "nl": "Wie krijgt het bezoek als de lezer een resultaat opent?"},
        "merged": {"en": "Which page writes a new text from what the sources say?",
                   "nl": "Welke pagina schrijft een nieuwe tekst uit wat de bronnen zeggen?"},
        "judge": {"en": "Where does a reader see a title and a snippet for each source?",
                  "nl": "Waar ziet een lezer bij elke bron een titel en een snippet?"},
        "alone": {"en": "On which page does a reader see nothing else next to the answer?",
                  "nl": "Op welke pagina ziet een lezer niets anders naast het antwoord?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the groups again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de groepen opnieuw uit."},
        "empty": {"en": "Sort each point into the page it belongs to.",
                  "nl": "Deel elk punt in bij de pagina waar het bij hoort."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "{item} belongs with the other page.",
                  "nl": "{item} hoort bij de andere pagina."},
        "correct": {"en": "All {n} are right. A page of ten links leaves the choice to the reader: the alternatives stay in view, the reader judges each source, and the source gets the visit. One answer saves the click and brings the sources together, and when it is wrong, it comes alone.",
                    "nl": "Alle {n} kloppen. Een pagina met tien links laat de keuze aan de lezer: de alternatieven blijven in beeld, de lezer beoordeelt elke bron, en de bron krijgt het bezoek. Eén antwoord bespaart de klik en brengt de bronnen samen, en als het antwoord fout is, komt het alleen."},
    },
)

# What the tests feed each check, and what it must say: (exercise, code or choices, state, known mistake).
CASES = [
    ("success_at_k", _SA_START, "empty", None),
    ("success_at_k", _SA, "correct", None),
    ("success_at_k", _SA_HEAD + "    for doc in ranking[...]:\n" + _SA_END, "wrong", "dots-left"),
    ("success_at_k", _SA_ALL, "wrong", "all"),
    ("success_at_k", _SA_SHORT, "wrong", "short"),
    ("success_at_k", _SA_HEAD + "    for doc in ranking[0:k]:\n" + _SA_END, "correct", None),
    ("success_at_k", "def success_at_k(ranking, relevant, k):\n    return len(set(ranking[:k]) & set(relevant)) > 0\n", "correct", None),
    ("success_at_k", _SA_HEAD + "    for doc in ranking[1:k]:\n" + _SA_END, "almost", None),
    ("success_at_k", "def success_at_k(ranking, relevant, k):\n    return ranking[k - 1] in relevant\n", "almost", None),
    ("citation_recall", _CR_START, "empty", None),
    ("citation_recall", _CR, "correct", None),
    ("citation_recall", _CR_HEAD + "            if source in ...:\n" + _CR_END, "wrong", "dots-left"),
    ("citation_recall", _CR_PAIR, "wrong", "pair"),
    ("citation_recall", _CR_CITED, "wrong", "cited"),
    ("citation_recall", _CR_HEAD + "            if supports[i].count(source) > 0:\n" + _CR_END, "correct", None),
    ("citation_recall", "def citation_recall(cited, supports):\n    supported = 0\n    for i in range(len(cited)):\n"
     "        for source in cited[i]:\n            if source in supports[i]:\n                supported = supported + 1\n"
     "    return supported / len(cited)\n", "wrong", None),
    ("citation_recall", _CR_HEAD + "            if source in supports[0]:\n" + _CR_END, "almost", None),
    ("citation_precision", _CP_START, "empty", None),
    ("citation_precision", _CP, "correct", None),
    ("citation_precision", _CP_HEAD + "        total = total + ...\n" + _CP_END, "wrong", "dots-left"),
    ("citation_precision", _CP_SENTENCES, "wrong", "sentences"),
    ("citation_precision", _CP_SUPPORTS, "wrong", "supports"),
    ("citation_precision", _CP_HEAD + "        total += len(cited[i])\n" + _CP_END, "correct", None),
    ("citation_precision", _CP_HEAD + "        total = len(cited[i])\n" + _CP_END, "wrong", None),
    ("pages", {}, "empty", None),
    ("pages", dict(EXERCISES["pages"].solution), "correct", None),
    ("pages", {"alternatives": "links", "no_click": "answer"}, "wrong", "unfinished"),
] + [("pages", {item: ("links" if group == "answer" else "answer")}, "wrong", f"wrong:{item}")
     for item, group in EXERCISES["pages"].solution.items()]
