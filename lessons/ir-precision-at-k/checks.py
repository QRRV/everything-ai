# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ir-precision-at-k: how well a ranking does on a test collection, with relevance judgments for
each query. The share of relevant documents among the first k (precision_at_k), one over the place of the first relevant
document (reciprocal_rank), and the precision at each relevant document, averaged over all relevant documents of the
query (average_precision), as in Manning, Raghavan and Schütze, Introduction to Information Retrieval (2008), sections
8.1 to 8.5, and Voorhees and Tice, The TREC-8 Question Answering Track (2000). The mean over the queries gives MAP and
MRR. The sort asks which measure fits which search.

The test collections are the course's thirteen documents with the three needs and their judgments of the lesson
ir-what-is-search; the 2,067 paragraphs and the questions of XQuAD (everything_ai.xquad), where the own paragraph of each
question is its one relevant paragraph; and SciFact (everything_ai.scifact), downloaded when the learner asks, where the
abstracts that a claim cites are its relevant abstracts. The rankings are those of the lessons ir-vector-space and
ir-bm25 (the cosine, and BM25 with k1 = 1.2 and b = 0.75), each cut at its first 100 documents, with documents of the
same score in the order of their number. Every number in the text and the pictures is computed by the course
(tests/test_ir_precision_at_k.py computes them again)."""
import html
import importlib.util
import math

import numpy as np

from everything_ai import lesson as lesson_helpers
from everything_ai import paths, scifact, xquad, yours
from everything_ai.check import FunctionBox, Mistake, Sort, kept
from everything_ai.lesson import download_size

LANGS = ("en", "nl")


def _bm25():
    spec = importlib.util.spec_from_file_location("everything_ai_checks_ir_bm25_for_precision_at_k",
                                                  paths.LESSONS / "ir-bm25" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _bm25()             # BM25 and the cosine on the thirteen documents, the paragraphs of XQuAD and SciFact
V = M.V                 # the vector space model, for the parts of its pictures
I = M.I                 # the inverted index, for what shows a paragraph
W = M.W                 # the thirteen documents, the needs behind the queries and the course's judgments
S = M.S                 # the parts of the pictures and the exercises
K1, B = M.K1, M.B       # k1 = 1.2 and b = 0.75, the usual settings of BM25

CUT = 100               # each ranking is cut at its first 100 documents
HOWS = ("cosine", "bm25")

DOWNLOAD_LINE = {   # what the lesson downloads, for the line under the header
    "en": f"Works on every computer. SciFact, 5,183 abstracts of biomedical papers with claims about them, is downloaded once from Hugging Face, {download_size(scifact.FILES, 'en')}, when you click its button in the lesson; the demo needs it, the rest of the lesson works without it.",
    "nl": f"Werkt op elke computer. SciFact, 5.183 abstracts van biomedische papers met beweringen erover, wordt één keer gedownload van Hugging Face, {download_size(scifact.FILES, 'nl')}, als je in de les op de knop van de download klikt; de demo heeft de set nodig, de rest van de les werkt zonder.",
}


# --- the course's solutions, the same code as the exercises -------------------------------------------------------------------

def precision_at_k(ranking, relevant, k):
    """The share of relevant documents among the first k of the ranking. A ranking with fewer than k documents counts
    the missing places as not relevant: the division is by k."""
    found = 0
    for doc in ranking[:k]:
        if doc in relevant:
            found = found + 1
    return found / k


def reciprocal_rank(ranking, relevant):
    """One over the place of the first relevant document of the ranking, counted from 1; 0 when the ranking holds no
    relevant document."""
    for place in range(len(ranking)):
        if ranking[place] in relevant:
            return 1 / (place + 1)
    return 0


def average_precision(ranking, relevant):
    """The precision at the place of each relevant document of the ranking, added up and divided by the number of
    relevant documents: a relevant document that the ranking does not hold adds 0."""
    total = 0
    for place in range(len(ranking)):
        if ranking[place] in relevant:
            total = total + precision_at_k(ranking, relevant, place + 1)
    return total / len(relevant)


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


# --- the course's thirteen documents ------------------------------------------------------------------------------------------

NEEDS = W.NEEDS             # the three needs, the query of each and the documents that help with it
NEED_TEXT = W.NEED_TEXT     # each need in words


def small_scores(lang: str, query: str) -> list:
    """The BM25 score of each of the thirteen documents for the query python or wine."""
    q = W.QUERIES[lang][query]
    return [M.bm25(q, d, M.SMALL_IDF[lang], M.SMALL_AVERAGE[lang], K1, B) for d in W.DOCS[lang]]


def small_results(lang: str, need: str) -> list:
    """The results of BM25 for the query of a need: the documents with a score above 0, the highest first, and of equal
    scores the lowest number first."""
    s = small_scores(lang, NEEDS[need]["query"])
    return sorted((i for i in range(len(s)) if s[i] > 0), key=lambda i: (-s[i], i))


def small_numbers(lang: str) -> dict:
    """For each need: the results, the places of its relevant documents among them (None for a relevant document that
    is not a result), P@3, the reciprocal rank and the average precision; and the means over the three needs."""
    out = {}
    for need in NEEDS:
        results, rel = small_results(lang, need), NEEDS[need]["relevant"]
        out[need] = {"results": results, "places": [results.index(d) + 1 if d in results else None for d in rel],
                     "p3": precision_at_k(results, rel, 3), "rr": reciprocal_rank(results, rel),
                     "ap": average_precision(results, rel)}
    out["mean"] = {key: mean(out[n][key] for n in NEEDS) for key in ("p3", "rr", "ap")}
    return out


# --- the paragraphs and the questions of XQuAD --------------------------------------------------------------------------------

def _cut(scores: np.ndarray) -> list:
    """For each row of scores, the first CUT documents as a list, best first; of equal scores the lowest number first."""
    order = np.argsort(-scores, axis=1, kind="stable")[:, :CUT]
    return [[int(i) for i in row] for row in order]


def rankings(lang: str, how: str) -> list:
    """For each question of XQuAD, the first 100 paragraphs of a ranking ("cosine" or "bm25"), best first: a new list
    of new lists each time."""
    assert lang in LANGS and how in HOWS, (lang, how)
    got = kept(f"ir-precision-at-k:rankings:{lang}:{how}", lambda: _cut(M.ranked(lang, how)))
    return [list(r) for r in got]


def relevant(lang: str) -> list:
    """For each question of XQuAD, its relevant paragraphs: a list with its own paragraph."""
    return [[q.paragraph] for q in xquad.questions(lang)]


def places(lang: str, how: str) -> list:
    """For each question, the place of its own paragraph in the ranking, counted from 1, or None after the first 100."""
    out = []
    for r, rel in zip(rankings(lang, how), relevant(lang)):
        out.append(r.index(rel[0]) + 1 if rel[0] in r else None)
    return out


def measures(lang: str, how: str, ks=(1, 3, 5, 10)) -> dict:
    """The course's measures of a ranking on the questions of XQuAD: P@k for each k, the share of questions whose own
    paragraph is among the first 10, MRR and MAP."""
    def make():
        rs, rels = rankings(lang, how), relevant(lang)
        out = {f"p{k}": mean(precision_at_k(r, rel, k) for r, rel in zip(rs, rels)) for k in ks}
        out["in10"] = mean(rel[0] in r[:10] for r, rel in zip(rs, rels))
        out["mrr"] = mean(reciprocal_rank(r, rel) for r, rel in zip(rs, rels))
        out["map"] = mean(average_precision(r, rel) for r, rel in zip(rs, rels))
        return out
    return dict(kept(f"ir-precision-at-k:measures:{lang}:{how}:{ks}", make))


def question_place(qid: str, lang: str) -> int:
    """The place of a question in xquad.questions(lang)."""
    return V.question_place(qid, lang)


# The question of the worked example and of the end of the lesson, the same in both lessons: the first paragraph of BM25
# holds the answer too, but the test collection judges only the own paragraph relevant.
EXAMPLE = "5727cb4b2ca10214002d9679"   # Prior to the arrival of the French, ... Jacksonville ... inhabited by what people?


def answer_elsewhere(lang: str, how: str = "bm25") -> dict:
    """For the questions whose own paragraph is not first: how many there are, and for how many the first paragraph holds
    the text of one of the answers too, letter for letter."""
    paras, qs = xquad.paragraphs(lang), xquad.questions(lang)
    missed = [(k, r[0]) for k, r in enumerate(rankings(lang, how)) if r[0] != qs[k].paragraph]
    holds = [k for k, first in missed if any(a[0] in paras[first].text for a in qs[k].answers)]
    return {"missed": len(missed), "holds": len(holds)}


# --- SciFact ------------------------------------------------------------------------------------------------------------------

def _sci() -> dict:
    """The 300 test claims of SciFact: the first 100 abstracts of the cosine and of BM25 for each, and the places of the
    relevant abstracts of each claim in scifact.abstracts()."""
    def make():
        sc = M._sci()
        test = sc["test"]
        return {"cosine": _cut(M.cosine_scores(sc["M"], test)), "bm25": _cut(M.bm25_scores(sc["M"], test, K1, B)),
                "relevant": [sorted(r) for r in test["relevant"]], "ids": list(test["ids"])}
    return kept("ir-precision-at-k:scifact", make)


def sci_measures(how: str, k: int = 10) -> dict:
    """The course's measures of a ranking ("cosine", "bm25" or "perfect", the relevant abstracts first) on the 300 test
    claims: P@k, MRR and MAP."""
    sc = _sci()
    rels = sc["relevant"]
    rs = [list(r) + [d for d in sc["bm25"][j] if d not in r] for j, r in enumerate(rels)] if how == "perfect" else sc[how]
    return {"pk": mean(precision_at_k(r, rel, k) for r, rel in zip(rs, rels)),
            "mrr": mean(reciprocal_rank(r, rel) for r, rel in zip(rs, rels)),
            "map": mean(average_precision(r, rel) for r, rel in zip(rs, rels))}


def sci_counts() -> dict:
    """How many test claims have 1, 2, 3, ... relevant abstracts."""
    counts = {}
    for rel in _sci()["relevant"]:
        counts[len(rel)] = counts.get(len(rel), 0) + 1
    return dict(sorted(counts.items()))


def multi_claims() -> list:
    """The ids of the test claims with more than one relevant abstract, in the order of their number."""
    sc = _sci()
    return [c for c, rel in zip(sc["ids"], sc["relevant"]) if len(rel) > 1]


def claim(cid: str) -> dict:
    """A test claim: its text, its relevant abstracts, and the first 100 abstracts of the cosine and of BM25."""
    sc = _sci()
    j = sc["ids"].index(cid)
    text = next(c["text"] for c in scifact.claims("test") if c["id"] == cid)
    return {"id": cid, "text": text, "relevant": list(sc["relevant"][j]), "cosine": list(sc["cosine"][j]),
            "bm25": list(sc["bm25"][j])}


def scifact_or_stop(button, texts: dict) -> list:
    """multi_claims() for the lesson's cells, once SciFact is downloaded. Until then the cell stops with `texts["ask"]`
    and the button; a click downloads the four files, with the progress in the cell."""
    lesson_helpers.pins_or_stop(button, texts, scifact.FILES)
    return multi_claims()


CLAIM = "133"   # the claim the demo starts with: both rankings put a relevant abstract first, and the cosine finds more


# --- the pictures ----------------------------------------------------------------------------------------------------------

_UI, _LABEL, _CELL, _HEAD, _NUMBER = S._UI, S._LABEL, S._CELL, S._HEAD, S._NUMBER
_DATA, _FAINT, _PRED, _LOSS = S._DATA, S._FAINT, S._PRED, S._LOSS
_CHIP, _CHIP_PRED, _CHIP_DATA, _CHIP_LOSS = S._CHIP, S._CHIP_PRED, S._CHIP_DATA, S._CHIP_LOSS
_note, _colors, _label, _num, _pct, _dec = S._note, S._colors, S._label, S._num, S._pct, S._dec
_cell, _table = S._cell, S._table
_fn, _call, _empty, _SLOW, _STOPPED = S._fn, S._call, S._empty, S._SLOW, S._STOPPED
_WHY_EMPTY = S._WHY_EMPTY
_chip, _is_number, _same, _Odd = V._chip, V._is_number, V._same, V._Odd
_DASH = V._DASH


def _doc_chip(i: int, relevant_: bool) -> str:
    """A document by its number: blue when it is relevant, gray when it is not."""
    return _chip(str(i), _CHIP_PRED if relevant_ else _CHIP_DATA)


def _value(x: float, lang: str) -> str:
    """A measure with three decimals, and 1 and 0 as whole numbers."""
    return _num(x, lang) if x in (0, 1) else _dec(x, lang, 3)


def _guarded(make):
    """make() within the time limit of a picture; None when the learner's function gives something odd."""
    def run():
        try:
            return make()
        except _Odd:
            return None
    return _call(run)


def _number(x) -> float:
    """A learner's value as a float, when it is a number from 0 to 1."""
    if not _is_number(x) or not -1e-9 <= float(x) <= 1 + 1e-9:
        raise _Odd
    return float(x)


def _mine(got: float, want: float, lang: str) -> str:
    """A learner's measure: blue when it is the course's, purple with the course's after it in gray when not."""
    if _same(got, want):
        return f'<span style="{_PRED}">{html.escape(_value(got, lang))}</span>'
    return (f'<span style="{_LOSS}">{html.escape(_value(got, lang))}</span>'
            f' <span style="{_DATA}">({html.escape(_value(want, lang))})</span>')


_IDEA_TEXT = {
    "en": {"title": "Three needs, three rankings, three measures",
           "need": "Need {k}: {need} Query: *{q}*.",
           "head_place": "Place", "row_doc": "Document", "row_p": "Precision up to here",
           "missed": "Relevant, but not among the results: {docs}.",
           "scores": "Reciprocal rank {rr}, average precision {ap}.",
           "means": "The three needs together: each measure, and its mean over the needs.",
           "head_means": ["Need", "P@3", "Reciprocal rank", "Average precision"],
           "mean": "Mean: MRR and MAP",
           "gray": "Gray: the documents that are not relevant, the places, and the precision at places without a relevant document.",
           "blue": "Blue: the relevant documents, the precision at their places, and the measures."},
    "nl": {"title": "Drie behoeften, drie rangschikkingen, drie maten",
           "need": "Behoefte {k}: {need} Zoekvraag: *{q}*.",
           "head_place": "Plaats", "row_doc": "Document", "row_p": "Precision tot hier",
           "missed": "Relevant, maar niet bij de resultaten: {docs}.",
           "scores": "Reciprocal rank {rr}, average precision {ap}.",
           "means": "De drie behoeften samen: elke maat, en het gemiddelde over de behoeften.",
           "head_means": ["Behoefte", "P@3", "Reciprocal rank", "Average precision"],
           "mean": "Gemiddelde: MRR en MAP",
           "gray": "Grijs: de documenten die niet relevant zijn, de plaatsen, en de precision op plaatsen zonder relevant document.",
           "blue": "Blauw: de relevante documenten, de precision op de plaats van elk relevant document, en de maten."},
}


def idea_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang]
    nums = small_numbers(lang)
    body = ""
    for k, need in enumerate(NEEDS, 1):
        results, rel = nums[need]["results"], NEEDS[need]["relevant"]
        q = " ".join(W.QUERIES[lang][NEEDS[need]["query"]])
        body += _label(w["need"].format(k=k, need=NEED_TEXT[lang][need], q=q))
        docs = "".join(_cell(_doc_chip(d, d in rel)) for d in results)
        precisions = ""
        for place, d in enumerate(results, 1):
            p = precision_at_k(results, rel, place)
            precisions += _cell(html.escape(_value(p, lang)), _PRED if d in rel else _DATA, _NUMBER)
        rows = ["<tr>" + _cell(html.escape(w["row_doc"]), _FAINT) + docs + "</tr>",
                "<tr>" + _cell(html.escape(w["row_p"]), _FAINT) + precisions + "</tr>"]
        body += _table(rows, [w["head_place"]] + [str(p) for p in range(1, len(results) + 1)],
                       tuple(range(1, len(results) + 1)))
        missed = [d for d in rel if d not in results]
        if missed:
            body += _note(w["missed"].format(docs=", ".join(str(d) for d in missed)))
        said = w["scores"].format(rr=_value(nums[need]["rr"], lang), ap=_value(nums[need]["ap"], lang))
        body += f'<div style="{_LABEL};{_PRED}">{html.escape(said)}</div>'
    rows = []
    for k, need in enumerate(NEEDS, 1):
        rows.append("<tr>" + _cell(str(k), _FAINT, _NUMBER)
                    + "".join(_cell(html.escape(_value(nums[need][key], lang)), _PRED, _NUMBER) for key in ("p3", "rr", "ap"))
                    + "</tr>")
    rows.append("<tr>" + _cell(html.escape(w["mean"]), _FAINT)
                + "".join(_cell(html.escape(_value(nums["mean"][key], lang)), _PRED, _NUMBER) for key in ("p3", "rr", "ap"))
                + "</tr>")
    body += _label(w["means"]) + _table(rows, w["head_means"], (1, 2, 3))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


_HOW_NAMES = {"en": {"cosine": "Cosine", "bm25": "BM25", "perfect": "A perfect ranking"},
              "nl": {"cosine": "Cosine similarity", "bm25": "BM25", "perfect": "Een perfecte rangschikking"}}
_KS = (1, 3, 5, 10)   # the values of k in the picture under precision_at_k


def _perfect(ranking_: list, rel: list) -> list:
    """The ranking with the relevant documents first: the best order there is."""
    return list(rel) + [d for d in ranking_ if d not in rel]


_PK_TEXT = {
    "en": {"title": "Your precision_at_k on the questions of XQuAD",
           "kind": "a number from 0 to 1",
           "label": "The mean of your precision at k over the {n} questions, for three rankings of the 2,067 paragraphs:",
           "head": ["Ranking", "P@1", "P@3", "P@5", "P@10"],
           "in10": "The own paragraph is among the first 10 for {c} of the questions with the cosine, and {b} with BM25.",
           "gray": "Gray: the rankings, the share of questions with the own paragraph among the first 10, and the course's precision where yours differs.",
           "blue": "Blue: your precision where it is the course's.",
           "purple": "Purple: your precision where it differs from the course's."},
    "nl": {"title": "Je precision_at_k op de vragen van XQuAD",
           "kind": "een getal van 0 tot 1",
           "label": "Het gemiddelde van je precision at k over de {n} vragen, voor drie rangschikkingen van de 2.067 alinea's:",
           "head": ["Rangschikking", "P@1", "P@3", "P@5", "P@10"],
           "in10": "De eigen alinea staat bij de eerste 10 voor {c} van de vragen met de cosine similarity, en {b} met BM25.",
           "gray": "Grijs: de rangschikkingen, het deel van de vragen met de eigen alinea bij de eerste 10, en de precision van de course waar de jouwe verschilt.",
           "blue": "Blauw: je precision waar deze gelijk is aan wat de course geeft.",
           "purple": "Paars: je precision waar deze verschilt van wat de course geeft."},
}


def precision_at_k_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _PK_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    rels = relevant(lang)
    lists = {how: rankings(lang, how) for how in HOWS}
    lists["perfect"] = [_perfect(r, rel) for r, rel in zip(lists["bm25"], rels)]

    def make():
        return {how: [mean(_number(fn(list(r), list(rel), k)) for r, rel in zip(lists[how], rels)) for k in _KS]
                for how in lists}

    got = _guarded(make)
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    if got is None:
        return _empty(w, "odd")
    rows = []
    for how in ("cosine", "bm25", "perfect"):
        want = [mean(precision_at_k(r, rel, k) for r, rel in zip(lists[how], rels)) for k in _KS]
        rows.append("<tr>" + _cell(html.escape(_HOW_NAMES[lang][how]), _DATA)
                    + "".join(_cell(_mine(g, x, lang), "", _NUMBER) for g, x in zip(got[how], want)) + "</tr>")
    body = _label(w["label"].format(n=_num(len(rels), lang))) + _table(rows, w["head"], (1, 2, 3, 4))
    body += _note(w["in10"].format(c=_pct(measures(lang, "cosine")["in10"], lang), b=_pct(measures(lang, "bm25")["in10"], lang)))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


_RR_GROUPS = ((1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 10), (11, CUT), (None, None))

_RR_TEXT = {
    "en": {"title": "Your reciprocal_rank on the questions of XQuAD",
           "kind": "a number from 0 to 1",
           "label": "Where the own paragraph of each of the {n} questions comes, what your reciprocal rank gives for that place, and how many questions the cosine and BM25 put there:",
           "head": ["Place", "Your reciprocal rank", "Questions, cosine", "Questions, BM25"],
           "none": "after 100",
           "mrr": "The mean of your reciprocal rank over the questions, the MRR, next to P@1:",
           "head_mrr": ["Ranking", "P@1", "Your MRR"],
           "gray": "Gray: the places, the numbers of questions, P@1, and the course's value where yours differs.",
           "blue": "Blue: your reciprocal rank and MRR where they are the course's.",
           "purple": "Purple: your reciprocal rank and MRR where they differ from the course's."},
    "nl": {"title": "Je reciprocal_rank op de vragen van XQuAD",
           "kind": "een getal van 0 tot 1",
           "label": "Waar de eigen alinea van elk van de {n} vragen staat, wat je reciprocal rank bij deze plaats geeft, en hoeveel vragen de cosine similarity en BM25 daar zetten:",
           "head": ["Plaats", "Jouw reciprocal rank", "Vragen, cosine similarity", "Vragen, BM25"],
           "none": "na 100",
           "mrr": "Het gemiddelde van je reciprocal rank over de vragen, de MRR, naast P@1:",
           "head_mrr": ["Rangschikking", "P@1", "Jouw MRR"],
           "gray": "Grijs: de plaatsen, de aantallen vragen, P@1, en de waarde van de course waar de jouwe verschilt.",
           "blue": "Blauw: je reciprocal rank en MRR waar deze gelijk zijn aan wat de course geeft.",
           "purple": "Paars: je reciprocal rank en MRR waar deze verschillen van wat de course geeft."},
}


def _at(place) -> list:
    """A ranking of CUT documents, 1000 to 1099, with document 1 at the place (counted from 1); None leaves it out."""
    out = list(range(1000, 1000 + CUT))
    if place is not None:
        out[place - 1] = 1
    return out


def reciprocal_rank_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _RR_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    rels = relevant(lang)
    lists = {how: rankings(lang, how) for how in HOWS}

    def make():
        at = {p: _number(fn(_at(p), [1])) for p in (1, 2, 3, 4, 5, 6, 10, 11, CUT, None)}
        return at, {how: mean(_number(fn(list(r), list(rel))) for r, rel in zip(lists[how], rels)) for how in HOWS}

    got = _guarded(make)
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    if got is None:
        return _empty(w, "odd")
    at, mrr = got
    pl = {how: places(lang, how) for how in HOWS}
    rows = []
    for low, high in _RR_GROUPS:
        if low is None:
            name, mine = w["none"], _mine(at[None], 0.0, lang)
        elif low == high:
            name, mine = str(low), _mine(at[low], 1 / low, lang)
        else:
            name = f"{low}–{high}"
            mine = _mine(at[low], 1 / low, lang) + f' <span style="{_FAINT}">–</span> ' + _mine(at[high], 1 / high, lang)
        counts = [sum(1 for p in pl[how] if (p is None if low is None else p is not None and low <= p <= high))
                  for how in HOWS]
        rows.append("<tr>" + _cell(html.escape(name), _FAINT, _NUMBER) + _cell(mine, "", _NUMBER)
                    + "".join(_cell(_num(c, lang), _DATA, _NUMBER) for c in counts) + "</tr>")
    body = _label(w["label"].format(n=_num(len(rels), lang))) + _table(rows, w["head"], (0, 1, 2, 3))
    rows = []
    for how in HOWS:
        m = measures(lang, how)
        rows.append("<tr>" + _cell(html.escape(_HOW_NAMES[lang][how]), _DATA)
                    + _cell(html.escape(_value(m["p1"], lang)), _DATA, _NUMBER)
                    + _cell(_mine(mrr[how], m["mrr"], lang), "", _NUMBER) + "</tr>")
    body += _label(w["mrr"]) + _table(rows, w["head_mrr"], (1, 2))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


_AP_TEXT = {
    "en": {"title": "Your average_precision on three test collections",
           "kind": "a number from 0 to 1",
           "small": "The three needs of the course's thirteen documents, with the results of BM25:",
           "head_small": ["Need", "Results", "Your average precision"],
           "map_small": "Your MAP over the three needs: {m}; their MRR is {r}.",
           "xquad": "The {n} questions of XQuAD, each with one relevant paragraph:",
           "sci": "The {n} test claims of SciFact, with one to five relevant abstracts each:",
           "head_map": ["Ranking", "Your MAP", "MRR"],
           "no_sci": "Download SciFact at *Try it* below, and this picture adds its test claims, which can have more than one relevant abstract.",
           "gray": "Gray: the documents that are not relevant, the MRR, and the course's value where yours differs.",
           "blue": "Blue: the relevant documents, and your average precision and MAP where they are the course's.",
           "purple": "Purple: your average precision and MAP where they differ from the course's."},
    "nl": {"title": "Je average_precision op drie testcollecties",
           "kind": "een getal van 0 tot 1",
           "small": "De drie behoeften bij de dertien documenten van de course, met de resultaten van BM25:",
           "head_small": ["Behoefte", "Resultaten", "Jouw average precision"],
           "map_small": "Je MAP over de drie behoeften: {m}; de MRR is {r}.",
           "xquad": "De {n} vragen van XQuAD, elk met één relevante alinea:",
           "sci": "De {n} testbeweringen van SciFact, met één tot vijf relevante abstracts per bewering:",
           "head_map": ["Rangschikking", "Jouw MAP", "MRR"],
           "no_sci": "Download SciFact bij *Probeer het* hieronder, en dit plaatje voegt de testbeweringen toe, met soms meer dan één relevant abstract.",
           "gray": "Grijs: de documenten die niet relevant zijn, de MRR, en de waarde van de course waar de jouwe verschilt.",
           "blue": "Blauw: de relevante documenten, en je average precision en MAP waar deze gelijk zijn aan wat de course geeft.",
           "purple": "Paars: je average precision en MAP waar deze verschillen van wat de course geeft."},
}


def average_precision_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _AP_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    nums = small_numbers(lang)
    rels = relevant(lang)
    lists = {how: rankings(lang, how) for how in HOWS}
    sci = _sci() if scifact.ready() else None

    def make():
        small = {need: _number(fn(list(nums[need]["results"]), list(NEEDS[need]["relevant"]))) for need in NEEDS}
        big = {how: mean(_number(fn(list(r), list(rel))) for r, rel in zip(lists[how], rels)) for how in HOWS}
        claims = None
        if sci is not None:
            claims = {how: mean(_number(fn(list(r), list(rel))) for r, rel in zip(sci[how], sci["relevant"])) for how in HOWS}
        return small, big, claims

    got = _guarded(make)
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    if got is None:
        return _empty(w, "odd")
    small, big, claims = got
    rows = []
    for k, need in enumerate(NEEDS, 1):
        rel = NEEDS[need]["relevant"]
        chips = " ".join(_doc_chip(d, d in rel) for d in nums[need]["results"])
        rows.append("<tr>" + _cell(str(k), _FAINT, _NUMBER) + _cell(chips)
                    + _cell(_mine(small[need], nums[need]["ap"], lang), "", _NUMBER) + "</tr>")
    body = _label(w["small"]) + _table(rows, w["head_small"], (0, 2))
    body += _note(w["map_small"].format(m=_value(mean(small.values()), lang), r=_value(nums["mean"]["rr"], lang)))
    rows = []
    for how in HOWS:
        m = measures(lang, how)
        rows.append("<tr>" + _cell(html.escape(_HOW_NAMES[lang][how]), _DATA) + _cell(_mine(big[how], m["map"], lang), "", _NUMBER)
                    + _cell(html.escape(_value(m["mrr"], lang)), _DATA, _NUMBER) + "</tr>")
    body += _label(w["xquad"].format(n=_num(len(rels), lang))) + _table(rows, w["head_map"], (1, 2))
    if claims is None:
        body += _note(w["no_sci"])
    else:
        rows = []
        for how in HOWS:
            m = sci_measures(how)
            rows.append("<tr>" + _cell(html.escape(_HOW_NAMES[lang][how]), _DATA)
                        + _cell(_mine(claims[how], m["map"], lang), "", _NUMBER)
                        + _cell(html.escape(_value(m["mrr"], lang)), _DATA, _NUMBER) + "</tr>")
        body += _label(w["sci"].format(n=_num(len(sci["relevant"]), lang))) + _table(rows, w["head_map"], (1, 2))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


_TRY_TEXT = {
    "en": {"title": "Claim {c} of SciFact, and all 300 test claims",
           "claim": "The claim: *{t}* It has {n} relevant abstracts.",
           "strip": "The first {k} abstracts of each ranking; the relevant ones are blue:",
           "head": ["Ranking", "The first {k} abstracts", "Places of the relevant abstracts", "P@{k}", "RR", "AP"],
           "after": "after 100",
           "all": "The 300 test claims, with the same k:",
           "head_all": ["Ranking", "Mean P@{k}", "MRR", "MAP"],
           "gray": "Gray: the abstracts that are not relevant, the places, and the measures of the perfect ranking.",
           "blue": "Blue: the relevant abstracts, and the measures of the cosine and BM25."},
    "nl": {"title": "Bewering {c} van SciFact, en alle 300 testbeweringen",
           "claim": "De bewering: *{t}* Er zijn {n} relevante abstracts.",
           "strip": "De eerste {k} abstracts van elke rangschikking; de relevante abstracts zijn blauw:",
           "head": ["Rangschikking", "De eerste {k} abstracts", "Plaatsen van de relevante abstracts", "P@{k}", "RR", "AP"],
           "after": "na 100",
           "all": "De 300 testbeweringen, met dezelfde k:",
           "head_all": ["Rangschikking", "Gemiddelde P@{k}", "MRR", "MAP"],
           "gray": "Grijs: de abstracts die niet relevant zijn, de plaatsen, en de maten van de perfecte rangschikking.",
           "blue": "Blauw: de relevante abstracts, en de maten van de cosine similarity en BM25."},
}
TRY_K = (1, 20, 5)   # the slider for k: from 1 to 20, starting at 5


def _try_k(value) -> int:
    """The slider's k, a whole number from 1 to 20; the start when it is not a number."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return TRY_K[2]
    if not math.isfinite(x):
        return TRY_K[2]
    return int(round(min(TRY_K[1], max(TRY_K[0], x))))


def try_picture(lang: str, cid=CLAIM, k=TRY_K[2]) -> str:
    lang = lang if lang in LANGS else "en"
    w = _TRY_TEXT[lang]
    k = _try_k(k)
    claims_ = multi_claims()
    cid = str(cid) if str(cid) in claims_ else (CLAIM if CLAIM in claims_ else claims_[0])
    c = claim(cid)
    rel = c["relevant"]
    body = _label(w["claim"].format(t=c["text"], n=_num(len(rel), lang)))
    rows = []
    for how in HOWS:
        r = c[how]
        chips = " ".join(_doc_chip(r[j], r[j] in rel) for j in range(min(k, len(r))))
        at = sorted(r.index(d) + 1 for d in rel if d in r)
        said = ", ".join(_num(p, lang) for p in at) + "".join(", " + w["after"] for d in rel if d not in r)
        rows.append("<tr>" + _cell(html.escape(_HOW_NAMES[lang][how]), _DATA) + _cell(chips)
                    + _cell(html.escape(said), _DATA, _NUMBER)
                    + "".join(_cell(html.escape(_value(x, lang)), _PRED, _NUMBER)
                              for x in (precision_at_k(r, rel, k), reciprocal_rank(r, rel), average_precision(r, rel)))
                    + "</tr>")
    body += _label(w["strip"].format(k=_num(k, lang))) + _table(rows, [h.format(k=k) for h in w["head"]], (2, 3, 4, 5))
    rows = []
    for how in ("cosine", "bm25", "perfect"):
        m = sci_measures(how, k)
        style = _DATA if how == "perfect" else _PRED
        rows.append("<tr>" + _cell(html.escape(_HOW_NAMES[lang][how]), _DATA)
                    + "".join(_cell(html.escape(_value(m[key], lang)), style, _NUMBER) for key in ("pk", "mrr", "map"))
                    + "</tr>")
    body += _label(w["all"]) + _table(rows, [h.format(k=k) for h in w["head_all"]], (1, 2, 3))
    title = w["title"].format(c=cid)
    return yours.figure(title, body + _colors(w, body), alt=title)


def try_options() -> dict:
    """The claims of the demo's dropdown: the claim's number and the start of its text, to its id."""
    out = {}
    for cid in multi_claims():
        text = claim(cid)["text"]
        short = text if len(text) <= 60 else text[:57].rsplit(" ", 1)[0] + " …"
        out[f"{cid}: {short}"] = cid
    return out


# --- the exercises ---------------------------------------------------------------------------------------------------------

_Val = S._Val
_DOTS = S._DOTS
_SAME_TEXTS = S._SAME_TEXTS
_show, _written = V._show, V._written

_PK_HEAD = "def precision_at_k(ranking, relevant, k):\n    found = 0\n"
_PK_END = "        if doc in relevant:\n            found = found + 1\n    return found / k\n"
_PK = _PK_HEAD + "    for doc in ranking[:k]:\n" + _PK_END
_PK_START = _PK_HEAD + "    for doc in ...:\n" + _PK_END
_PK_WHOLE = _PK_HEAD + "    for doc in ranking:\n" + _PK_END
_PK_AFTER = _PK_HEAD + "    for doc in ranking[k:]:\n" + _PK_END
_PK_SKIP = _PK_HEAD + "    for doc in ranking[1:k]:\n" + _PK_END
_PK_RELEVANT = _PK_HEAD + "    for doc in relevant:\n" + _PK_END

_RR_HEAD = "def reciprocal_rank(ranking, relevant):\n    for place in range(len(ranking)):\n        if ranking[place] in relevant:\n"
_RR_END = "    return 0\n"
_RR = _RR_HEAD + "            return 1 / (place + 1)\n" + _RR_END
_RR_START = _RR_HEAD + "            return ...\n" + _RR_END
_RR_RANK = _RR_HEAD + "            return place + 1\n" + _RR_END
_RR_ONE = _RR_HEAD + "            return 1\n" + _RR_END
_RR_PLACE = _RR_HEAD + "            return place\n" + _RR_END

_AP_HEAD = ("def average_precision(ranking, relevant):\n    total = 0\n    for place in range(len(ranking)):\n"
            "        if ranking[place] in relevant:\n")
_AP_END = "    return total / len(relevant)\n"
_AP = _AP_HEAD + "            total = total + precision_at_k(ranking, relevant, place + 1)\n" + _AP_END
_AP_START = _AP_HEAD + "            total = total + ...\n" + _AP_END
_AP_RECIPROCAL = _AP_HEAD + "            total = total + 1 / (place + 1)\n" + _AP_END
_AP_COUNT = _AP_HEAD + "            total = total + 1\n" + _AP_END
_AP_WHOLE = _AP_HEAD + "            total = total + precision_at_k(ranking, relevant, len(ranking))\n" + _AP_END

# The small cases the exercise texts show; the same in both lessons.
TOY_PK = (([7, 3, 1, 9, 4], [7, 4], 2), ([5, 2], [2, 8], 4))
TOY_RR = (([4, 8, 5, 2, 6], [6, 2]), ([5, 1], [9]))
TOY_AP = (([2, 5, 1, 7], [2, 7]), ([9, 4, 6], [4, 8]))


def _course_need(lang: str, need: str) -> tuple:
    """The results of BM25 for a need of the course's thirteen documents, and its relevant documents."""
    return (_Val(f'C.small_results("{lang}", "{need}")', small_results(lang, need)),
            _Val(f'C.NEEDS["{need}"]["relevant"]', list(NEEDS[need]["relevant"])))


def _course_question(lang: str, how: str) -> tuple:
    """The first 100 paragraphs of a ranking for the question of the worked example, and its own paragraph."""
    k = question_place(EXAMPLE, lang)
    return (_Val(f'C.rankings("{lang}", "{how}")[{k}]', rankings(lang, how)[k]),
            _Val(f'C.relevant("{lang}")[{k}]', relevant(lang)[k]))


REQUIRED = ["precision_at_k", "reciprocal_rank", "average_precision", "measures"]

EXERCISES = {
    "precision_at_k": FunctionBox(
        name="precision_at_k",
        picture=precision_at_k_picture,
        function="precision_at_k",
        start=_PK_START,
        solution=_PK,
        show=_show,
        visible=[tuple(_written(v) for v in case) for case in TOY_PK],
        hidden=[tuple(_written(v) for v in case) for case in (
            ([7, 3, 1], [7], 1), ([3, 7, 1], [7], 1), ([1, 2, 3, 4], [1, 2, 3, 4], 4), ([1, 2, 3], [], 2), ([], [4], 3),
            ([4, 6, 8, 10, 12, 14], [6, 12, 20], 6))]
        + [_course_need("en", "wine") + (_written(5),), _course_need("nl", "buy") + (_written(3),),
           _course_question("en", "bm25") + (_written(1),), _course_question("en", "bm25") + (_written(10),)],
        first=[_DOTS],
        mistakes=[
            Mistake("whole", wrong=_PK_WHOLE,
                    en="`{call}` gives {got}, and it should give {want}: your loop goes through the whole ranking, so it also counts relevant documents after place k. Which part of the ranking should the loop go through?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je loop gaat door de hele rangschikking, dus de functie telt ook relevante documenten na plaats k. Door welk deel van de rangschikking moet de loop gaan?"),
            Mistake("after", wrong=_PK_AFTER,
                    en="`{call}` gives {got}, and it should give {want}: your loop goes through the documents after the first k, the part that P@k leaves out. Which side of the colon should k be on?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je loop gaat door de documenten na de eerste k, het deel dat P@k juist weglaat. Aan welke kant van de dubbele punt hoort k?"),
            Mistake("skip-first", wrong=_PK_SKIP,
                    en="`{call}` gives {got}, and it should give {want}: your loop starts at the second document, because Python counts places from 0. Which number gives the first document of the ranking?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je loop begint bij het tweede document, want Python telt plaatsen vanaf 0. Welk getal geeft het eerste document van de rangschikking?"),
            Mistake("relevant", wrong=_PK_RELEVANT,
                    en="`{call}` gives {got}, and it should give {want}: your loop goes through the relevant documents, and each of them is relevant, so your function counts them all. Through what should the loop go, to see which results are relevant?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je loop gaat door de relevante documenten, en elk daarvan is relevant, dus je functie telt alle relevante documenten. Waar moet de loop doorheen gaan, om te zien welke resultaten relevant zijn?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the first `k` documents of the ranking.",
                      "nl": "Vervang de `...` door de eerste `k` documenten van de rangschikking."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your loop go through the first `k` documents of `ranking`, and no others?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Gaat je loop door de eerste `k` documenten van `ranking`, en geen andere?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function also work for k = 1, for no relevant documents, for an empty ranking, and for the course's rankings?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook voor k = 1, zonder relevante documenten, voor een lege rangschikking, en voor de rangschikkingen van de course?"},
            "correct": {"en": "Right: your function gives the share of relevant documents among the first k. In the picture below, it measures three rankings on all the questions of XQuAD: how high can P@10 get, when each question has one relevant paragraph?",
                        "nl": "Goed: je functie geeft het deel van de eerste k documenten dat relevant is. In het plaatje hieronder meet de functie drie rangschikkingen op alle vragen van XQuAD: hoe hoog kan P@10 worden, als elke vraag één relevante alinea heeft?"},
        },
    ),
    "reciprocal_rank": FunctionBox(
        name="reciprocal_rank",
        picture=reciprocal_rank_picture,
        function="reciprocal_rank",
        start=_RR_START,
        solution=_RR,
        show=_show,
        visible=[tuple(_written(v) for v in case) for case in TOY_RR],
        hidden=[tuple(_written(v) for v in case) for case in (
            ([3, 1, 2], [3]), ([3, 1, 2], [1, 2]), ([], [1]), ([1, 2, 3], []), (list(range(10)), [9]))]
        + [_course_need("en", "buy"), _course_need("nl", "learn"), _course_question("en", "bm25"),
           _course_question("nl", "cosine")],
        first=[_DOTS],
        mistakes=[
            Mistake("rank", wrong=_RR_RANK,
                    en="`{call}` gives {got}, and it should give {want}: your function gives the place itself, so a relevant document further down scores higher. What should 1 be divided by?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie geeft de plaats zelf, dus een relevant document verder naar beneden scoort hoger. Waardoor moet 1 gedeeld worden?"),
            Mistake("one", wrong=_RR_ONE,
                    en="`{call}` gives {got}, and it should give {want}: your function gives 1 wherever the first relevant document is. How should its place make the score smaller?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie geeft 1, waar het eerste relevante document ook staat. Hoe moet de plaats de score kleiner maken?"),
            Mistake("place", wrong=_RR_PLACE,
                    en="`{call}` gives {got}, and it should give {want}: your function gives the place as Python counts it, from 0. What is the reciprocal of a place counted from 1?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie geeft de plaats zoals Python telt, vanaf 0. Wat is één gedeeld door een plaats geteld vanaf 1?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with one over the place of the document, counted from 1.",
                      "nl": "Vervang de `...` door één gedeeld door de plaats van het document, geteld vanaf 1."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function give 1 divided by the place counted from 1, which is one more than `place`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Geeft je functie 1 gedeeld door de plaats geteld vanaf 1, één meer dan `place`?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function also work when the first document is relevant, for an empty ranking, and for the course's rankings?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook als het eerste document relevant is, voor een lege rangschikking, en voor de rangschikkingen van de course?"},
            "correct": {"en": "Right: your function gives one over the place of the first relevant document. In the picture below, it gives the MRR of the cosine and of BM25 on XQuAD: how much higher than P@1 is it, and where does the difference come from?",
                        "nl": "Goed: je functie geeft één gedeeld door de plaats van het eerste relevante document. In het plaatje hieronder geeft de functie de MRR van de cosine similarity en van BM25 op XQuAD: hoeveel hoger dan P@1 is deze, en waar komt het verschil vandaan?"},
        },
    ),
    "average_precision": FunctionBox(
        name="average_precision",
        picture=average_precision_picture,
        function="average_precision",
        start=_AP_START,
        solution=_AP,
        show=_show,
        names={"precision_at_k": precision_at_k},
        visible=[tuple(_written(v) for v in case) for case in TOY_AP],
        hidden=[tuple(_written(v) for v in case) for case in (
            ([1, 2, 3], [1, 2]), ([1, 2, 3], [3]), ([1, 2, 3], [7, 8]), ([], [5]),
            (list(range(100)), [0, 9, 24, 30, 207]))]
        + [_course_need("en", "buy"), _course_need("en", "wine"), _course_need("nl", "wine"),
           _course_question("nl", "bm25")],
        first=[_DOTS],
        mistakes=[
            Mistake("reciprocal", wrong=_AP_RECIPROCAL,
                    en="`{call}` gives {got}, and it should give {want}: your function adds one over the place of each relevant document, as if no relevant document came before it. What should each relevant document add: the precision of which part of the ranking?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie telt één gedeeld door de plaats van elk relevant document op, alsof er geen relevant document erboven staat. Wat moet elk relevant document toevoegen: de precision van welk deel van de rangschikking?"),
            Mistake("count", wrong=_AP_COUNT,
                    en="`{call}` gives {got}, and it should give {want}: your function adds 1 for each relevant document, wherever it is, so it gives the share of relevant documents found. How should the place of a relevant document count?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie telt 1 op voor elk relevant document, waar het ook staat, dus je functie geeft het deel van de relevante documenten dat gevonden is. Hoe moet de plaats van een relevant document meetellen?"),
            Mistake("whole", wrong=_AP_WHOLE,
                    en="`{call}` gives {got}, and it should give {want}: your function adds the precision of the whole ranking for each relevant document. Up to which place should the precision go?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie telt voor elk relevant document de precision van de hele rangschikking op. Tot welke plaats moet de precision gaan?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the precision of the ranking up to this place, from `precision_at_k`.",
                      "nl": "Vervang de `...` door de precision van de rangschikking tot deze plaats, uit `precision_at_k`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does each relevant document add `precision_at_k` of the ranking up to its own place, counted from 1?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Voegt elk relevant document `precision_at_k` toe van de rangschikking tot de eigen plaats, geteld vanaf 1?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function also work when every relevant document is missing, for an empty ranking, and for the course's rankings?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook als elk relevant document ontbreekt, voor een lege rangschikking, en voor de rangschikkingen van de course?"},
            "correct": {"en": "Right: your function averages the precision at each relevant document, and a missing one adds 0. In the picture below, it gives the MAP on three test collections: where is the MAP the same as the MRR, and why?",
                        "nl": "Goed: je functie neemt het gemiddelde van de precision bij elk relevant document, en een ontbrekend document voegt 0 toe. In het plaatje hieronder geeft de functie de MAP op drie testcollecties: waar is de MAP gelijk aan de MRR, en waarom?"},
        },
    ),
}

# The sort: which measure fits which search.
EXERCISES["measures"] = Sort(
    name="measures",
    groups={
        "pk": {"en": "Precision at k", "nl": "Precision at k"},
        "mrr": {"en": "MRR", "nl": "MRR"},
        "map": {"en": "MAP", "nl": "MAP"},
    },
    items={
        "page": {"en": "a results page shows ten links, and what counts is how many of the ten help",
                 "nl": "een pagina met resultaten toont tien links, en wat telt is hoeveel van de tien helpen"},
        "unknown": {"en": "the judges looked only at the first ten results of each query, and every relevant result among those ten should count",
                    "nl": "de beoordelaars keken alleen naar de eerste tien resultaten van elke zoekvraag, en elk relevant resultaat bij deze tien moet meetellen"},
        "half": {"en": "a question can have several right paragraphs, but only the first one should count, and in second place it should still count for half",
                 "nl": "een vraag kan meer goede alinea's hebben, maar alleen de eerste moet tellen, en op de tweede plaats nog voor de helft"},
        "trec": {"en": "the question answering track of TREC-8 scored five answers to each question by the first right one",
                 "nl": "de vraag-en-antwoordtaak van TREC-8 gaf vijf antwoorden op elke vraag een score met het eerste goede antwoord"},
        "lawyer": {"en": "a lawyer needs every earlier case about a question, and each relevant case that comes late should cost",
                   "nl": "een advocaat zoekt elke eerdere zaak over een vraag, en elke relevante zaak die laat komt, moet punten kosten"},
        "claims": {"en": "a claim can have one relevant abstract or five, and each of them should count, also the one found last",
                   "nl": "een bewering kan één relevant abstract hebben of vijf, en elk abstract moet meetellen, ook het abstract dat als laatste komt"},
    },
    solution={"page": "pk", "unknown": "pk", "half": "mrr", "trec": "mrr", "lawyer": "map", "claims": "map"},
    nudge={
        "page": {"en": "Which measure looks at a fixed number of results, and counts the relevant ones among them?",
                 "nl": "Welke maat kijkt naar een vast aantal resultaten, en telt de relevante resultaten daartussen?"},
        "unknown": {"en": "Which measure needs no judgments after the first k results, and counts every relevant one among them?",
                    "nl": "Welke maat heeft geen oordelen nodig na de eerste k resultaten, en telt elk relevant resultaat daartussen?"},
        "half": {"en": "Which measure gives 1 over the place of the first relevant document?",
                 "nl": "Welke maat geeft 1 gedeeld door de plaats van het eerste relevante document?"},
        "trec": {"en": "Voorhees and Tice say which measure that track used. Which measure looks only at the first right answer?",
                 "nl": "Voorhees en Tice noemen de maat van deze taak. Welke maat kijkt alleen naar het eerste goede antwoord?"},
        "lawyer": {"en": "Which measure looks at the place of every relevant document, and counts a missing one as 0?",
                   "nl": "Welke maat kijkt naar de plaats van elk relevant document, en telt een ontbrekend document als 0?"},
        "claims": {"en": "Which measure divides by the number of relevant documents of the query, whether that is one or five?",
                   "nl": "Welke maat deelt door het aantal relevante documenten van de zoekvraag, of dat er nu één of vijf zijn?"},
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
        "correct": {"en": "All {n} are right. Precision at k counts what helps among the first k results, without knowing how many relevant documents there are. MRR looks only at the first relevant result. MAP looks at the place of every relevant document, and a missing one costs.",
                    "nl": "Alle {n} kloppen. Precision at k telt wat helpt bij de eerste k resultaten, zonder te weten hoeveel relevante documenten er zijn. MRR kijkt alleen naar het eerste relevante resultaat. MAP kijkt naar de plaats van elk relevant document, en een ontbrekend document kost punten."},
    },
)

# What the tests feed each check, and what it must say: (exercise, code or choices, state, known mistake).
CASES = [
    ("precision_at_k", _PK_START, "empty", None),
    ("precision_at_k", _PK, "correct", None),
    ("precision_at_k", _PK_HEAD + "    for doc in ranking[...]:\n" + _PK_END, "wrong", "dots-left"),
    ("precision_at_k", _PK_WHOLE, "wrong", "whole"),
    ("precision_at_k", _PK_AFTER, "wrong", "after"),
    ("precision_at_k", _PK_SKIP, "wrong", "skip-first"),
    ("precision_at_k", _PK_RELEVANT, "wrong", "relevant"),
    ("precision_at_k", _PK_HEAD + "    for doc in ranking[0:k]:\n" + _PK_END, "correct", None),
    ("precision_at_k", "def precision_at_k(ranking, relevant, k):\n    return len([d for d in ranking[:k] if d in relevant]) / k\n",
     "correct", None),
    ("precision_at_k", _PK_HEAD + "    for doc in ranking[:k]:\n        if doc in relevant:\n            found = found + 1\n"
     "    return found / len(ranking[:k])\n", "wrong", None),
    ("reciprocal_rank", _RR_START, "empty", None),
    ("reciprocal_rank", _RR, "correct", None),
    ("reciprocal_rank", _RR_HEAD + "            return 1 / ...\n" + _RR_END, "wrong", "dots-left"),
    ("reciprocal_rank", _RR_RANK, "wrong", "rank"),
    ("reciprocal_rank", _RR_ONE, "wrong", "one"),
    ("reciprocal_rank", _RR_PLACE, "wrong", "place"),
    ("reciprocal_rank", _RR_HEAD + "            return 1 / place\n" + _RR_END, "wrong", None),
    ("reciprocal_rank", _RR_HEAD + "            rank = place + 1\n            return 1 / rank\n" + _RR_END, "correct", None),
    ("average_precision", _AP_START, "empty", None),
    ("average_precision", _AP, "correct", None),
    ("average_precision", _AP_HEAD + "            total = total + precision_at_k(...)\n" + _AP_END, "wrong", "dots-left"),
    ("average_precision", _AP_RECIPROCAL, "wrong", "reciprocal"),
    ("average_precision", _AP_COUNT, "wrong", "count"),
    ("average_precision", _AP_WHOLE, "wrong", "whole"),
    ("average_precision", _AP_HEAD + "            total += precision_at_k(ranking[:place + 1], relevant, place + 1)\n" + _AP_END,
     "correct", None),
    ("measures", {}, "empty", None),
    ("measures", dict(EXERCISES["measures"].solution), "correct", None),
    ("measures", {"page": "pk", "half": "mrr"}, "wrong", "unfinished"),
] + [("measures", {item: ("mrr" if group != "mrr" else "map")}, "wrong", f"wrong:{item}")
     for item, group in EXERCISES["measures"].solution.items()]
