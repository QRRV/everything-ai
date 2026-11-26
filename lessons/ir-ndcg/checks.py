# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ir-ndcg: how well a ranking does when documents are more or less relevant. Each document of the
ranking gets its grade as its gain, each gain is divided by the log base 2 of its place plus 1 and added up (dcg), and
that DCG is divided by the DCG of the ideal order of all the grades of the query (ndcg), as in Järvelin and Kekäläinen,
Cumulated Gain-Based Evaluation of IR Techniques (2002), with the discount of trec_eval (m_ndcg_cut.c: the gain at
place i divided by log2(i + 1) from the first place on, the grade as the gain). dcg_all does the same for many queries
at once with NumPy. The sort asks what makes nDCG go up, go down or stay the same.

The test collections are the course's thirteen documents with the three needs of the lesson ir-what-is-search, which this
lesson grades from 0 to 3 on the scale of the TREC Deep Learning track of 2019 (Craswell and others, 2020); the
questions of XQuAD (everything_ai.xquad), where the own paragraph of each question is its one relevant paragraph, with
grade 1; and SciFact (everything_ai.scifact) when it is downloaded, with grade 1 for each relevant abstract, the way BEIR
(Thakur and others, 2021) measures it. The rankings are those of the lesson ir-precision-at-k: the cosine and BM25,
each cut at its first 100 documents. Every number in the text and the pictures is computed by the course
(tests/test_ir_ndcg.py computes them again)."""
import html
import importlib.util
import math

import numpy as np

from everything_ai import paths, scifact, xquad, yours
from everything_ai.check import FunctionBox, Mistake, Sort, kept

LANGS = ("en", "nl")


def _precision():
    spec = importlib.util.spec_from_file_location("everything_ai_checks_ir_precision_at_k_for_ndcg",
                                                  paths.LESSONS / "ir-precision-at-k" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _precision()        # precision at k, MRR and MAP, and the rankings of XQuAD and SciFact cut at 100
M = P.M                 # BM25 and the cosine
V = P.V                 # the vector space model, for the parts of its pictures
W = P.W                 # the thirteen documents, the needs behind the queries and the course's judgments
S = P.S                 # the parts of the pictures and the exercises

HOWS = P.HOWS           # "cosine" and "bm25"
CUT = P.CUT             # each ranking is cut at its first 100 documents
SMALL_K = 5             # nDCG@5 on the thirteen documents: no query there has more than five results
BIG_K = 10              # nDCG@10 on XQuAD and SciFact, as TREC and BEIR report it

rankings, relevant, measures = P.rankings, P.relevant, P.measures
precision_at_k, reciprocal_rank, average_precision, mean = P.precision_at_k, P.reciprocal_rank, P.average_precision, P.mean


# --- the course's solutions, the same code as the exercises -------------------------------------------------------------------

def dcg(gains, k):
    """The discounted cumulative gain of the first k gains: each gain divided by the log base 2 of its place counted
    from 1, plus 1, and added up. A list with fewer than k gains counts what it has."""
    total = 0
    top = gains[:k]
    for place in range(len(top)):
        total = total + top[place] / np.log2(place + 2)
    return total


def ndcg(gains, judged, k):
    """The DCG of the gains, divided by the DCG of the ideal order: all the grades of the query in `judged`, the highest
    first. 0 for a query without a relevant document."""
    best = dcg(sorted(judged, reverse=True), k)
    if best == 0:
        return 0
    return dcg(gains, k) / best


def dcg_all(gains, k):
    """dcg for each row of a NumPy array of gains, one row for each query, each row at least k long."""
    discounts = np.log2(np.arange(2, k + 2))
    return (gains[:, :k] / discounts).sum(axis=1)


# --- the course's thirteen documents, with grades ------------------------------------------------------------------------------

NEEDS = P.NEEDS             # the three needs, the query of each and its relevant documents of ir-what-is-search
NEED_TEXT = P.NEED_TEXT     # each need in words

# The course's grades of the thirteen documents for each need, on the scale of the TREC Deep Learning track of 2019:
# 3 perfectly relevant, 2 highly relevant, 1 related, and 0 (left out) irrelevant. The documents with 2 or 3 are the
# relevant documents of ir-what-is-search. In the order of the documents' numbers.
GRADES = {
    "buy": {1: 3, 3: 2, 5: 1},      # Ball pythons for sale; a warm tank, where to buy one; caring for a python at home
    "learn": {0: 3, 4: 1},          # Python for beginners; a new version of Python
    "wine": {6: 3, 7: 1, 9: 2},     # red, white and the heart; a glass of red among heart-healthy food; beer, white, red
}


def gains(ranking, grades) -> list:
    """The gain of each document of a ranking: its grade, or 0 for a document without one."""
    return [grades[d] if d in grades else 0 for d in ranking]


def judged(need: str) -> list:
    """All the grades of a need, in the order of the documents' numbers."""
    return [GRADES[need][d] for d in GRADES[need]]


def small_results(lang: str, need: str, how: str = "bm25") -> list:
    """The results of a ranking for the query of a need: the documents with a score above 0, the highest first, and of
    equal scores the lowest number first. BM25 as in ir-precision-at-k, the cosine of ir-vector-space."""
    if how == "bm25":
        return P.small_results(lang, need)
    q = V.vector(W.QUERIES[lang][NEEDS[need]["query"]], V.SMALL_IDF[lang])
    s = [V.cosine(q, d) for d in V.SMALL_VECTORS[lang]]
    return sorted((i for i in range(len(s)) if s[i] > 0), key=lambda i: (-s[i], i))


def small_numbers(lang: str, how: str = "bm25") -> dict:
    """For each need: the results of the ranking, their gains, the DCG@5, the ideal order, its DCG@5, the nDCG@5 and the
    average precision of ir-precision-at-k; and the means over the three needs."""
    out = {}
    for need in NEEDS:
        results = small_results(lang, need, how)
        g = gains(results, GRADES[need])
        ideal = sorted(judged(need), reverse=True)
        out[need] = {"results": results, "gains": g, "dcg": dcg(g, SMALL_K), "ideal": ideal,
                     "idcg": dcg(ideal, SMALL_K), "ndcg": ndcg(g, judged(need), SMALL_K),
                     "ap": average_precision(results, NEEDS[need]["relevant"])}
    out["mean"] = {key: mean(out[n][key] for n in NEEDS) for key in ("ndcg", "ap")}
    return out


# --- XQuAD and SciFact, with grade 1 for each relevant document ----------------------------------------------------------------

def xquad_gains(lang: str, how: str) -> np.ndarray:
    """For each question of XQuAD, the gains of the first 100 paragraphs of a ranking ("cosine", "bm25", or "perfect",
    the own paragraph first): 1 for the own paragraph, 0 for the others. A new array each time."""
    assert lang in LANGS and how in HOWS + ("perfect",), (lang, how)

    def make():
        if how == "perfect":
            out = np.zeros((len(relevant(lang)), CUT))
            out[:, 0] = 1
            return out
        return np.array([gains(r, {rel[0]: 1}) for r, rel in zip(rankings(lang, how), relevant(lang))], dtype=float)
    return kept(f"ir-ndcg:xquad-gains:{lang}:{how}", make).copy()


def xquad_ndcg(lang: str, how: str, k: int = BIG_K) -> float:
    """The mean nDCG@k over the questions of XQuAD, each with its own paragraph as its one relevant paragraph."""
    return float(kept(f"ir-ndcg:xquad:{lang}:{how}:{k}",
                      lambda: mean(ndcg(list(g), [1], k) for g in xquad_gains(lang, how))))


def sci_gains(how: str) -> list:
    """For each of the 300 test claims of SciFact, the gains of the first 100 abstracts of a ranking, 1 for each
    relevant abstract, and the grades of the claim: a 1 for each relevant abstract."""
    sc = P._sci()
    return [(gains(r, {d: 1 for d in rel}), [1] * len(rel)) for r, rel in zip(sc[how], sc["relevant"])]


def sci_ndcg(how: str, k: int = BIG_K) -> float:
    """The mean nDCG@k over the 300 test claims of SciFact."""
    return mean(ndcg(g, grades, k) for g, grades in sci_gains(how))


# nDCG@10 on SciFact in Table 2 of BEIR (Thakur and others, 2021), for the models of the table: BM25 is that of Anserini,
# with k1 = 0.9 and b = 0.4 and the title as a field of its own.
BEIR_SCIFACT = {"BM25": 0.665, "DeepCT": 0.630, "SPARTA": 0.582, "docT5query": 0.675, "DPR": 0.318, "ANCE": 0.507,
                "TAS-B": 0.643, "GenQ": 0.644, "ColBERT": 0.671, "BM25+CE": 0.688}


# --- the pictures ----------------------------------------------------------------------------------------------------------

_UI, _LABEL, _CELL, _HEAD, _NUMBER = S._UI, S._LABEL, S._CELL, S._HEAD, S._NUMBER
_DATA, _FAINT, _PRED, _LOSS = S._DATA, S._FAINT, S._PRED, S._LOSS
_CHIP_PRED, _CHIP_DATA = S._CHIP_PRED, S._CHIP_DATA
_note, _colors, _label, _num, _dec = S._note, S._colors, S._label, S._num, S._dec
_cell, _table = S._cell, S._table
_fn, _empty, _SLOW, _STOPPED = S._fn, S._empty, S._SLOW, S._STOPPED
_WHY_EMPTY = S._WHY_EMPTY
_chip, _is_number, _same, _Odd = V._chip, V._is_number, V._same, V._Odd
_value, _guarded, _number, _mine = P._value, P._guarded, P._number, P._mine
_HOW_NAMES = P._HOW_NAMES


def _doc_chip(i: int, grade: int = 0, shown: bool = False) -> str:
    """A document by its number: blue when it has a grade above 0, with the grade after it when `shown`; gray when not."""
    if grade > 0:
        return _chip(f"{i} · {grade}" if shown else str(i), _CHIP_PRED)
    return _chip(str(i), _CHIP_DATA)


def _gain_value(x) -> float:
    """A learner's DCG as a float, when it is a number of 0 or more."""
    if not _is_number(x) or float(x) < -1e-9:
        raise _Odd
    return float(x)


def _grades_text(values) -> str:
    return ", ".join(str(v) for v in values) if len(values) else "–"


_IDEA_TEXT = {
    "en": {"title": "Three needs, graded: DCG, the ideal order and nDCG",
           "need": "Need {k}: {need} Query: *{q}*.",
           "head_place": "Place", "row_doc": "Document", "row_gain": "Gain (grade)",
           "row_discount": "Discount, log₂(place + 1)", "row_part": "Gain / discount",
           "dcg": "DCG@5 = {d}. The ideal order of the grades: {ideal}, with DCG@5 = {i}. nDCG@5 = {d} / {i} = {n}.",
           "missed": "Graded, but not among the results: {docs}.",
           "means": "The three needs together: the nDCG@5 of BM25 next to its average precision, which counts the documents of grade 2 and 3 as relevant:",
           "head_means": ["Need", "Average precision", "nDCG@5"],
           "mean": "Mean: MAP and mean nDCG@5",
           "gray": "Gray: the documents without a grade, the places, the discounts and the average precision.",
           "blue": "Blue: the documents with a grade, their gains, and DCG, IDCG and nDCG."},
    "nl": {"title": "Drie behoeften met cijfers: DCG, de ideale volgorde en nDCG",
           "need": "Behoefte {k}: {need} Zoekvraag: *{q}*.",
           "head_place": "Plaats", "row_doc": "Document", "row_gain": "Gain (cijfer)",
           "row_discount": "Discount, log₂(plaats + 1)", "row_part": "Gain / discount",
           "dcg": "DCG@5 = {d}. De ideale volgorde van de cijfers: {ideal}, met DCG@5 = {i}. nDCG@5 = {d} / {i} = {n}.",
           "missed": "Met een cijfer, maar niet bij de resultaten: {docs}.",
           "means": "De drie behoeften samen: de nDCG@5 van BM25 naast de average precision, die de documenten met cijfer 2 en 3 als relevant telt:",
           "head_means": ["Behoefte", "Average precision", "nDCG@5"],
           "mean": "Gemiddelde: MAP en gemiddelde nDCG@5",
           "gray": "Grijs: de documenten zonder cijfer, de plaatsen, de discounts en de average precision.",
           "blue": "Blauw: de documenten met een cijfer, de gains, en DCG, IDCG en nDCG."},
}


def idea_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang]
    nums = small_numbers(lang)
    body = ""
    for k, need in enumerate(NEEDS, 1):
        n = nums[need]
        results, grades = n["results"], GRADES[need]
        q = " ".join(W.QUERIES[lang][NEEDS[need]["query"]])
        body += _label(w["need"].format(k=k, need=NEED_TEXT[lang][need], q=q))
        places = range(1, len(results) + 1)
        docs = "".join(_cell(_doc_chip(d, grades.get(d, 0))) for d in results)
        gain_cells = "".join(_cell(str(g), _PRED if g else _DATA, _NUMBER) for g in n["gains"])
        discounts = "".join(_cell(html.escape(_dec(math.log2(p + 1), lang, 3)), _DATA, _NUMBER) for p in places)
        parts = "".join(_cell(html.escape(_dec(g / math.log2(p + 1), lang, 3)) if g else "0", _PRED if g else _DATA, _NUMBER)
                        for p, g in zip(places, n["gains"]))
        rows = ["<tr>" + _cell(html.escape(w[key]), _FAINT) + cells + "</tr>"
                for key, cells in (("row_doc", docs), ("row_gain", gain_cells), ("row_discount", discounts), ("row_part", parts))]
        body += _table(rows, [w["head_place"]] + [str(p) for p in places], tuple(places))
        missed = [d for d in grades if d not in results]
        if missed:
            body += _note(w["missed"].format(docs=", ".join(f"{d} ({grades[d]})" for d in missed)))
        said = w["dcg"].format(d=_dec(n["dcg"], lang, 3), ideal=_grades_text(n["ideal"]), i=_dec(n["idcg"], lang, 3),
                               n=_value(n["ndcg"], lang))
        body += f'<div style="{_LABEL};{_PRED}">{html.escape(said)}</div>'
    rows = []
    for k, need in enumerate(NEEDS, 1):
        rows.append("<tr>" + _cell(str(k), _FAINT, _NUMBER) + _cell(html.escape(_value(nums[need]["ap"], lang)), _DATA, _NUMBER)
                    + _cell(html.escape(_value(nums[need]["ndcg"], lang)), _PRED, _NUMBER) + "</tr>")
    rows.append("<tr>" + _cell(html.escape(w["mean"]), _FAINT) + _cell(html.escape(_value(nums["mean"]["ap"], lang)), _DATA, _NUMBER)
                + _cell(html.escape(_value(nums["mean"]["ndcg"], lang)), _PRED, _NUMBER) + "</tr>")
    body += _label(w["means"]) + _table(rows, w["head_means"], (0, 1, 2))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


_PLACES = tuple(range(1, 11))   # the places of the discount row in the picture under dcg

_DCG_TEXT = {
    "en": {"title": "Your dcg: the discount of each place, and the three needs",
           "kind": "a number of 0 or more",
           "disc": "What your dcg gives a gain of 1 at each place: `dcg` of a list with a 1 at that place and 0 above it, with k set to that place, which is 1 divided by the discount of the place. Below it, the reciprocal rank of a relevant document at that place:",
           "head_place": "Place", "row_mine": "Your dcg", "row_rr": "Reciprocal rank",
           "needs": "Your DCG@5 of the results of BM25 for the three needs, and of the ideal order of their grades; the one divided by the other is the nDCG@5:",
           "head_needs": ["Need", "Gains", "Your DCG@5", "Ideal order", "Your IDCG@5", "nDCG@5"],
           "gray": "Gray: the places, the reciprocal rank, the gains, and the course's value where yours differs.",
           "blue": "Blue: what your dcg gives, and the nDCG that follows, where it is the course's.",
           "purple": "Purple: what your dcg gives, and the nDCG that follows, where it differs from the course's."},
    "nl": {"title": "Je dcg: de discount van elke plaats, en de drie behoeften",
           "kind": "een getal van 0 of meer",
           "disc": "Wat je dcg een gain van 1 op elke plaats geeft: `dcg` van een lijst met een 1 op deze plaats en 0 erboven, met k gelijk aan deze plaats, en dat is 1 gedeeld door de discount van de plaats. Eronder de reciprocal rank van een relevant document op deze plaats:",
           "head_place": "Plaats", "row_mine": "Jouw dcg", "row_rr": "Reciprocal rank",
           "needs": "Je DCG@5 van de resultaten van BM25 bij de drie behoeften, en van de ideale volgorde van de cijfers; het ene gedeeld door het andere is de nDCG@5:",
           "head_needs": ["Behoefte", "Gains", "Jouw DCG@5", "Ideale volgorde", "Jouw IDCG@5", "nDCG@5"],
           "gray": "Grijs: de plaatsen, de reciprocal rank, de gains, en de waarde van de course waar de jouwe verschilt.",
           "blue": "Blauw: wat je dcg geeft, en de nDCG die daaruit volgt, waar dit gelijk is aan wat de course geeft.",
           "purple": "Paars: wat je dcg geeft, en de nDCG die daaruit volgt, waar dit verschilt van wat de course geeft."},
}


def _mine_dec(got: float, want: float, lang: str) -> str:
    """A learner's DCG with three decimals: blue when it is the course's, purple with the course's after it in gray when not."""
    if _same(got, want):
        return f'<span style="{_PRED}">{html.escape(_dec(got, lang, 3))}</span>'
    return (f'<span style="{_LOSS}">{html.escape(_dec(got, lang, 3))}</span>'
            f' <span style="{_DATA}">({html.escape(_dec(want, lang, 3))})</span>')


def dcg_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _DCG_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    nums = small_numbers(lang)

    def make():
        disc = [_gain_value(fn([0] * (p - 1) + [1], p)) for p in _PLACES]
        needs = {need: (_gain_value(fn(list(nums[need]["gains"]), SMALL_K)), _gain_value(fn(list(nums[need]["ideal"]), SMALL_K)))
                 for need in NEEDS}
        return disc, needs

    got = _guarded(make)
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    if got is None:
        return _empty(w, "odd")
    disc, needs = got
    rows = ["<tr>" + _cell(html.escape(w["row_mine"]), _FAINT)
            + "".join(_cell(_mine(d, 1 / math.log2(p + 1), lang), "", _NUMBER) for d, p in zip(disc, _PLACES)) + "</tr>",
            "<tr>" + _cell(html.escape(w["row_rr"]), _FAINT)
            + "".join(_cell(html.escape(_value(1 / p, lang)), _DATA, _NUMBER) for p in _PLACES) + "</tr>"]
    body = _label(w["disc"]) + _table(rows, [w["head_place"]] + [str(p) for p in _PLACES], _PLACES)
    rows = []
    for k, need in enumerate(NEEDS, 1):
        n = nums[need]
        mine_dcg, mine_idcg = needs[need]
        ratio = mine_dcg / mine_idcg if mine_idcg > 0 else 0.0
        rows.append("<tr>" + _cell(str(k), _FAINT, _NUMBER) + _cell(html.escape(_grades_text(n["gains"])), _DATA)
                    + _cell(_mine_dec(mine_dcg, n["dcg"], lang), "", _NUMBER)
                    + _cell(html.escape(_grades_text(n["ideal"])), _DATA)
                    + _cell(_mine_dec(mine_idcg, n["idcg"], lang), "", _NUMBER)
                    + _cell(_mine(ratio, n["ndcg"], lang), "", _NUMBER) + "</tr>")
    body += _label(w["needs"]) + _table(rows, w["head_needs"], (0, 2, 4, 5))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


_NDCG_TEXT = {
    "en": {"title": "Your ndcg on three test collections",
           "kind": "a number from 0 to 1",
           "small": "The three needs of the course's thirteen documents, with the course's grades:",
           "head_small": ["Need", "Your nDCG@5, cosine", "Your nDCG@5, BM25", "Average precision, cosine",
                          "Average precision, BM25"],
           "mean": "Mean",
           "xquad": "The {n} questions of XQuAD, each with its own paragraph as its one relevant paragraph, of grade 1:",
           "head_xquad": ["Ranking", "Your mean nDCG@10", "MRR"],
           "sci": "The {n} test claims of SciFact, with grade 1 for each relevant abstract:",
           "head_sci": ["Ranking", "Your mean nDCG@10", "MAP"],
           "beir": "BEIR reports an nDCG@10 of {b} on SciFact for its BM25, from the toolkit Anserini.",
           "no_sci": "With SciFact on your computer, downloaded in *BM25: the classic ranking*, this picture adds its 300 test claims.",
           "gray": "Gray: the rankings, the average precision, the MRR and MAP, and the course's value where yours differs.",
           "blue": "Blue: your nDCG where it is the course's.",
           "purple": "Purple: your nDCG where it differs from the course's."},
    "nl": {"title": "Je ndcg op drie testcollecties",
           "kind": "een getal van 0 tot 1",
           "small": "De drie behoeften bij de dertien documenten van de course, met de cijfers van de course:",
           "head_small": ["Behoefte", "Jouw nDCG@5, cosine similarity", "Jouw nDCG@5, BM25",
                          "Average precision, cosine similarity", "Average precision, BM25"],
           "mean": "Gemiddelde",
           "xquad": "De {n} vragen van XQuAD, elk met de eigen alinea als enige relevante alinea, met cijfer 1:",
           "head_xquad": ["Rangschikking", "Jouw gemiddelde nDCG@10", "MRR"],
           "sci": "De {n} testbeweringen van SciFact, met cijfer 1 voor elk relevant abstract:",
           "head_sci": ["Rangschikking", "Jouw gemiddelde nDCG@10", "MAP"],
           "beir": "BEIR meldt een nDCG@10 van {b} op SciFact voor de eigen BM25, uit de toolkit Anserini.",
           "no_sci": "Met SciFact op je computer, gedownload bij *BM25: de klassieke rangschikking*, voegt dit plaatje de 300 testbeweringen toe.",
           "gray": "Grijs: de rangschikkingen, de average precision, de MRR en MAP, en de waarde van de course waar de jouwe verschilt.",
           "blue": "Blauw: je nDCG waar deze gelijk is aan wat de course geeft.",
           "purple": "Paars: je nDCG waar deze verschilt van wat de course geeft."},
}


def ndcg_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _NDCG_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    nums = {how: small_numbers(lang, how) for how in HOWS}
    big = {how: xquad_gains(lang, how) for how in HOWS}
    sci = {how: sci_gains(how) for how in HOWS} if scifact.ready() else None

    def make():
        small = {how: {need: _number(fn(list(nums[how][need]["gains"]), judged(need), SMALL_K)) for need in NEEDS}
                 for how in HOWS}
        xq = {how: mean(_number(fn([float(x) for x in row], [1], BIG_K)) for row in big[how]) for how in HOWS}
        claims = None
        if sci is not None:
            claims = {how: mean(_number(fn(list(g), list(grades), BIG_K)) for g, grades in sci[how]) for how in HOWS}
        return small, xq, claims

    got = _guarded(make)
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    if got is None:
        return _empty(w, "odd")
    small, xq, claims = got
    rows = []
    for k, need in enumerate(NEEDS, 1):
        rows.append("<tr>" + _cell(str(k), _FAINT, _NUMBER)
                    + "".join(_cell(_mine(small[how][need], nums[how][need]["ndcg"], lang), "", _NUMBER) for how in HOWS)
                    + "".join(_cell(html.escape(_value(nums[how][need]["ap"], lang)), _DATA, _NUMBER) for how in HOWS)
                    + "</tr>")
    rows.append("<tr>" + _cell(html.escape(w["mean"]), _FAINT)
                + "".join(_cell(_mine(mean(small[how].values()), nums[how]["mean"]["ndcg"], lang), "", _NUMBER) for how in HOWS)
                + "".join(_cell(html.escape(_value(nums[how]["mean"]["ap"], lang)), _DATA, _NUMBER) for how in HOWS)
                + "</tr>")
    body = _label(w["small"]) + _table(rows, w["head_small"], (0, 1, 2, 3, 4))
    rows = []
    for how in HOWS:
        rows.append("<tr>" + _cell(html.escape(_HOW_NAMES[lang][how]), _DATA)
                    + _cell(_mine(xq[how], xquad_ndcg(lang, how), lang), "", _NUMBER)
                    + _cell(html.escape(_value(measures(lang, how)["mrr"], lang)), _DATA, _NUMBER) + "</tr>")
    body += _label(w["xquad"].format(n=_num(len(big["bm25"]), lang))) + _table(rows, w["head_xquad"], (1, 2))
    if claims is None:
        body += _note(w["no_sci"])
    else:
        rows = []
        for how in HOWS:
            rows.append("<tr>" + _cell(html.escape(_HOW_NAMES[lang][how]), _DATA)
                        + _cell(_mine(claims[how], sci_ndcg(how), lang), "", _NUMBER)
                        + _cell(html.escape(_value(P.sci_measures(how)["map"], lang)), _DATA, _NUMBER) + "</tr>")
        body += _label(w["sci"].format(n=_num(len(sci["bm25"]), lang))) + _table(rows, w["head_sci"], (1, 2))
        body += _note(w["beir"].format(b=_dec(BEIR_SCIFACT["BM25"], lang, 3)))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


_ALL_KS = (1, 3, 5, 10)   # the values of k in the picture under dcg_all

_ALL_TEXT = {
    "en": {"title": "Your dcg_all on all the questions of XQuAD at once",
           "kind": "an array with one number of 0 or more for each question",
           "label": "Your dcg_all on an array of {n} rows, one for each question, with the gains of the first 100 paragraphs of a ranking: the mean over the questions, for four values of k. Each question has one relevant paragraph, of grade 1, so its ideal DCG is 1, and its DCG is its nDCG:",
           "head": ["Ranking", "nDCG@1", "nDCG@3", "nDCG@5", "nDCG@10", "MRR"],
           "gray": "Gray: the rankings, the MRR, and the course's value where yours differs.",
           "blue": "Blue: your mean where it is the course's.",
           "purple": "Purple: your mean where it differs from the course's."},
    "nl": {"title": "Je dcg_all op alle vragen van XQuAD tegelijk",
           "kind": "een array met één getal van 0 of meer voor elke vraag",
           "label": "Je dcg_all op een array van {n} rijen, één voor elke vraag, met de gains van de eerste 100 alinea's van een rangschikking: het gemiddelde over de vragen, voor vier waarden van k. Elke vraag heeft één relevante alinea, met cijfer 1, dus de ideale DCG is 1, en de DCG is de nDCG:",
           "head": ["Rangschikking", "nDCG@1", "nDCG@3", "nDCG@5", "nDCG@10", "MRR"],
           "gray": "Grijs: de rangschikkingen, de MRR, en de waarde van de course waar de jouwe verschilt.",
           "blue": "Blauw: je gemiddelde waar dit gelijk is aan wat de course geeft.",
           "purple": "Paars: je gemiddelde waar dit verschilt van wat de course geeft."},
}


def _row_values(out, n: int) -> list:
    """What a learner's dcg_all gives for n queries, as floats, when it is that many numbers of 0 or more."""
    try:
        values = np.asarray(out, dtype=float)
    except (TypeError, ValueError):
        raise _Odd from None
    if values.shape != (n,) or not np.all(np.isfinite(values)) or np.any(values < -1e-9):
        raise _Odd
    return [float(v) for v in values]


def dcg_all_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _ALL_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    lists = {how: xquad_gains(lang, how) for how in HOWS + ("perfect",)}
    n = len(lists["bm25"])

    def make():
        return {how: [mean(_row_values(fn(lists[how].copy(), k), n)) for k in _ALL_KS] for how in lists}

    got = _guarded(make)
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    if got is None:
        return _empty(w, "odd")
    rows = []
    for how in ("cosine", "bm25", "perfect"):
        mrr = measures(lang, how)["mrr"] if how in HOWS else 1.0
        rows.append("<tr>" + _cell(html.escape(_HOW_NAMES[lang][how]), _DATA)
                    + "".join(_cell(_mine(g, xquad_ndcg(lang, how, k), lang), "", _NUMBER) for g, k in zip(got[how], _ALL_KS))
                    + _cell(html.escape(_value(mrr, lang)), _DATA, _NUMBER) + "</tr>")
    body = _label(w["label"].format(n=_num(n, lang))) + _table(rows, w["head"], (1, 2, 3, 4, 5))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


# --- Try it: grade the documents yourself -------------------------------------------------------------------------------------

GAINS = ("grade", "power")   # the gain of a grade: the grade itself, or 2 to the power of the grade, minus 1
TRY_NEED = "wine"           # the need the demo starts with


def gain_of(grade: int, gain: str = "grade") -> int:
    """The gain of a grade: the grade itself, as trec_eval and BEIR count it, or 2 ** grade - 1, as Burges and others."""
    return 2 ** grade - 1 if gain == "power" else grade


def pool(lang: str, need: str) -> list:
    """The documents the demo asks a grade for: the results of the two rankings, and the documents the course graded,
    in the order of their numbers."""
    docs = set(GRADES[need])
    for how in HOWS:
        docs |= set(small_results(lang, need, how))
    return sorted(docs)


def try_sliders(lang: str, need: str) -> list:
    """(label, the course's grade) for each document of the pool: the label is the number and the start of the text."""
    out = []
    for d in pool(lang, need):
        text = W.TEXTS[lang][d]
        short = text if len(text) <= 56 else text[:53].rsplit(" ", 1)[0] + " …"
        out.append((f"{d}: {short}", GRADES[need].get(d, 0)))
    return out


def _try_need(need) -> str:
    return need if need in NEEDS else TRY_NEED


def _try_grades(lang: str, need: str, values) -> dict:
    """The grades of the sliders, a whole number from 0 to 3 for each document of the pool; the course's grade for a
    value that is missing or not a number."""
    docs = pool(lang, need)
    values = list(values) if isinstance(values, (list, tuple)) else []
    out = {}
    for j, d in enumerate(docs):
        v = values[j] if j < len(values) else None
        try:
            x = float(v)
        except (TypeError, ValueError):
            x = float("nan")
        g = int(round(min(3, max(0, x)))) if math.isfinite(x) else GRADES[need].get(d, 0)
        if g > 0:
            out[d] = g
    return out


def try_numbers(lang: str, need, values, gain: str = "grade") -> dict:
    """With the grades of the sliders and a gain: for each ranking its results, their grades, DCG@5 and nDCG@5, the
    nDCG@5 with the course's grades and the same gain, and the average precision with grade 2 and 3 as relevant; and
    the ideal order of the grades with its DCG@5."""
    need = _try_need(need)
    gain = gain if gain in GAINS else "grade"
    grades = _try_grades(lang, need, values)
    mine = {d: gain_of(g, gain) for d, g in grades.items()}
    course = {d: gain_of(g, gain) for d, g in GRADES[need].items()}
    ideal = sorted(grades.values(), reverse=True)
    out = {"need": need, "grades": grades, "ideal": ideal, "idcg": dcg([gain_of(g, gain) for g in ideal], SMALL_K)}
    for how in HOWS:
        r = small_results(lang, need, how)
        g = gains(r, mine)
        out[how] = {"results": r, "grades": gains(r, grades), "dcg": dcg(g, SMALL_K),
                    "ndcg": ndcg(g, list(mine.values()), SMALL_K),
                    "course": ndcg(gains(r, course), list(course.values()), SMALL_K),
                    "ap": average_precision(r, [d for d, x in grades.items() if x >= 2])
                    if any(x >= 2 for x in grades.values()) else 0.0}
    return out


_TRY_TEXT = {
    "en": {"title": "Need {k} with your grades",
           "need": "Need {k}: {need} Query: *{q}*.",
           "ideal": "The ideal order of your grades: {ideal}. Its DCG@5: {i}.",
           "power": "The gain of a grade is 2 to the power of the grade, minus 1: grades 1, 2 and 3 give 1, 3 and 7.",
           "head": ["Ranking", "The results, with your grades", "DCG@5", "nDCG@5", "nDCG@5, the course's grades", "Average precision"],
           "none": "With your grades no document is relevant: the ideal DCG is 0, so the nDCG is 0 for both rankings.",
           "ap": "The average precision counts the documents of grade 2 and 3 as relevant, and the others as not relevant.",
           "gray": "Gray: the documents of grade 0, the nDCG with the course's grades and the average precision.",
           "blue": "Blue: the documents with a grade, and DCG and nDCG with your grades."},
    "nl": {"title": "Behoefte {k} met jouw cijfers",
           "need": "Behoefte {k}: {need} Zoekvraag: *{q}*.",
           "ideal": "De ideale volgorde van je cijfers: {ideal}. De DCG@5 daarvan: {i}.",
           "power": "De gain van een cijfer is 2 tot de macht het cijfer, min 1: de cijfers 1, 2 en 3 geven 1, 3 en 7.",
           "head": ["Rangschikking", "De resultaten, met jouw cijfers", "DCG@5", "nDCG@5", "nDCG@5, cijfers van de course", "Average precision"],
           "none": "Met jouw cijfers is geen enkel document relevant: de ideale DCG is 0, dus de nDCG is 0 voor beide rangschikkingen.",
           "ap": "De average precision telt de documenten met cijfer 2 en 3 als relevant, en de andere als niet relevant.",
           "gray": "Grijs: de documenten met cijfer 0, de nDCG met de cijfers van de course en de average precision.",
           "blue": "Blauw: de documenten met een cijfer, en DCG en nDCG met jouw cijfers."},
}


def try_picture(lang: str, need=TRY_NEED, values=(), gain: str = "grade") -> str:
    lang = lang if lang in LANGS else "en"
    w = _TRY_TEXT[lang]
    t = try_numbers(lang, need, values, gain)
    need = t["need"]
    k = list(NEEDS).index(need) + 1
    q = " ".join(W.QUERIES[lang][NEEDS[need]["query"]])
    body = _label(w["need"].format(k=k, need=NEED_TEXT[lang][need], q=q))
    if gain == "power":
        body += _note(w["power"])
    rows = []
    for how in HOWS:
        x = t[how]
        chips = " ".join(_doc_chip(d, g, True) for d, g in zip(x["results"], x["grades"]))
        rows.append("<tr>" + _cell(html.escape(_HOW_NAMES[lang][how]), _DATA) + _cell(chips)
                    + _cell(html.escape(_dec(x["dcg"], lang, 3)), _PRED, _NUMBER)
                    + _cell(html.escape(_value(x["ndcg"], lang)), _PRED, _NUMBER)
                    + _cell(html.escape(_value(x["course"], lang)), _DATA, _NUMBER)
                    + _cell(html.escape(_value(x["ap"], lang)), _DATA, _NUMBER) + "</tr>")
    body += _table(rows, w["head"], (2, 3, 4, 5))
    if t["idcg"] > 0:
        body += f'<div style="{_LABEL};{_PRED}">{html.escape(w["ideal"].format(ideal=_grades_text(t["ideal"]), i=_dec(t["idcg"], lang, 3)))}</div>'
    else:
        body += _note(w["none"])
    body += _note(w["ap"])
    title = w["title"].format(k=k)
    return yours.figure(title, body + _colors(w, body), alt=title)


def try_options(lang: str) -> dict:
    """The needs of the demo's dropdown, by their number and text, to their key."""
    return {f"{k}: {NEED_TEXT[lang][need]}": need for k, need in enumerate(NEEDS, 1)}


GAIN_OPTIONS = {
    "en": {"the grade itself": "grade", "2 to the power of the grade, minus 1": "power"},
    "nl": {"het cijfer zelf": "grade", "2 tot de macht het cijfer, min 1": "power"},
}


# --- the exercises ---------------------------------------------------------------------------------------------------------

_Val = S._Val
_DOTS = S._DOTS
_SAME_TEXTS = S._SAME_TEXTS
_show, _written = V._show, V._written

_DC_HEAD = "def dcg(gains, k):\n    total = 0\n    top = gains[:k]\n    for place in range(len(top)):\n"
_DC_END = "    return total\n"
_DC = _DC_HEAD + "        total = total + top[place] / np.log2(place + 2)\n" + _DC_END
_DC_START = _DC_HEAD + "        total = total + top[place] / ...\n" + _DC_END
_DC_RANK = _DC_HEAD + "        total = total + top[place] / (place + 1)\n" + _DC_END
_DC_ZERO = _DC_HEAD + "        total = total + top[place] / np.log2(place + 1)\n" + _DC_END
_DC_NATURAL = _DC_HEAD + "        total = total + top[place] / np.log(place + 2)\n" + _DC_END

_ND_HEAD = "def ndcg(gains, judged, k):\n"
_ND_END = "    if best == 0:\n        return 0\n    return dcg(gains, k) / best\n"
_ND = _ND_HEAD + "    best = dcg(sorted(judged, reverse=True), k)\n" + _ND_END
_ND_START = _ND_HEAD + "    best = dcg(..., k)\n" + _ND_END
_ND_OWN = _ND_HEAD + "    best = dcg(sorted(gains, reverse=True), k)\n" + _ND_END
_ND_LOW = _ND_HEAD + "    best = dcg(sorted(judged), k)\n" + _ND_END
_ND_AS_IS = _ND_HEAD + "    best = dcg(judged, k)\n" + _ND_END

_DA_HEAD = "def dcg_all(gains, k):\n"
_DA_END = "    return (gains[:, :k] / discounts).sum(axis=1)\n"
_DA = _DA_HEAD + "    discounts = np.log2(np.arange(2, k + 2))\n" + _DA_END
_DA_START = _DA_HEAD + "    discounts = ...\n" + _DA_END
_DA_RANK = _DA_HEAD + "    discounts = np.arange(1, k + 1)\n" + _DA_END
_DA_ZERO = _DA_HEAD + "    discounts = np.log2(np.arange(1, k + 1))\n" + _DA_END
_DA_NATURAL = _DA_HEAD + "    discounts = np.log(np.arange(2, k + 2))\n" + _DA_END

# The small cases the exercise texts show; the same in both lessons. The first gains of dcg are those of the example of
# Järvelin and Kekäläinen (2002, section 2.1).
TOY_DC = (([3, 2, 3, 0, 0, 1, 2, 2, 3, 0], 3), ([1, 0, 3], 5))
TOY_ND = (([0, 3, 1], [2, 3, 1], 3), ([1, 0], [3, 1], 2))
TOY_DA = ((np.array([[3, 2, 3, 0], [1, 0, 3, 2]]), 3), (np.array([[0, 1], [2, 0]]), 2))


def _course_gains(lang: str, need: str, how: str) -> _Val:
    """The gains of the results of a ranking for a need of the thirteen documents, with the course's grades."""
    return _Val(f'C.gains(C.small_results("{lang}", "{need}", "{how}"), C.GRADES["{need}"])',
                gains(small_results(lang, need, how), GRADES[need]))


def _course_judged(need: str) -> _Val:
    return _Val(f'C.judged("{need}")', judged(need))


def _course_question(lang: str, how: str) -> _Val:
    """The gains of the first 100 paragraphs of a ranking for the question about Jacksonville of ir-precision-at-k."""
    k = P.question_place(P.EXAMPLE, lang)
    return _Val(f'C.gains(C.rankings("{lang}", "{how}")[{k}], {{{relevant(lang)[k][0]}: 1}})',
                gains(rankings(lang, how)[k], {relevant(lang)[k][0]: 1}))


def _course_array(lang: str, how: str) -> _Val:
    return _Val(f'C.xquad_gains("{lang}", "{how}")', xquad_gains(lang, how))


REQUIRED = ["dcg", "ndcg", "dcg_all", "changes"]

EXERCISES = {
    "dcg": FunctionBox(
        name="dcg",
        picture=dcg_picture,
        function="dcg",
        start=_DC_START,
        solution=_DC,
        show=_show,
        names={"np": np},
        visible=[tuple(_written(v) for v in case) for case in TOY_DC],
        hidden=[tuple(_written(v) for v in case) for case in (
            ([2], 1), ([0, 0, 1], 3), ([3, 3, 3], 2), ([], 4), ([1, 2, 3, 0, 1, 2], 6), ([0, 2, 0, 0, 0, 0, 0, 0, 0, 3], 10))]
        + [(_course_gains("en", "wine", "bm25"), _written(5)), (_course_gains("nl", "buy", "cosine"), _written(5)),
           (_course_question("en", "bm25"), _written(10))],
        first=[_DOTS],
        mistakes=[
            Mistake("rank", wrong=_DC_RANK,
                    en="`{call}` gives {got}, and it should give {want}: your function divides each gain by its place, as the reciprocal rank does, so a lower place counts far less. Järvelin and Kekäläinen wanted a milder discount: of what does nDCG take the log base 2?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie deelt elke gain door de plaats, zoals de reciprocal rank doet, dus een lagere plaats telt veel minder mee. Järvelin en Kekäläinen wilden een mildere discount: waarvan neemt nDCG de log met grondtal 2?"),
            Mistake("zero", wrong=_DC_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: at the first place your function divides by the log of 1, which is 0. Of which number should the log of the first place be taken, so that the discount there is 1?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: op de eerste plaats deelt je functie door de log van 1, wat 0 is. Van welk getal moet je op de eerste plaats de log nemen, zodat de discount daar 1 is?"),
            Mistake("natural", wrong=_DC_NATURAL,
                    en="`{call}` gives {got}, and it should give {want}: `np.log` takes the log base e, not base 2. Which function of NumPy takes the log base 2?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `np.log` neemt de log met grondtal e, niet 2. Welke functie van NumPy neemt de log met grondtal 2?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the discount of the place: the log base 2 of the place counted from 1, plus 1.",
                      "nl": "Vervang de `...` door de discount van de plaats: de log met grondtal 2 van de plaats geteld vanaf 1, plus 1."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is each gain divided by `np.log2` of the place counted from 1, plus 1, which is two more than `place`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Wordt elke gain gedeeld door `np.log2` van de plaats geteld vanaf 1, plus 1, twee meer dan `place`?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function also work for k = 1, for an empty list, for a list shorter than k, and for the course's gains?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook voor k = 1, voor een lege lijst, voor een lijst korter dan k, en voor de gains van de course?"},
            "correct": {"en": "Right: your function adds up the gains, each divided by the discount of its place. In the picture below, how much does place 10 still count, and how much would the reciprocal rank give it?",
                        "nl": "Goed: je functie telt de gains op, elk gedeeld door de discount van de eigen plaats. In het plaatje hieronder: hoeveel telt plaats 10 nog mee, en hoeveel zou de reciprocal rank deze plaats geven?"},
        },
    ),
    "ndcg": FunctionBox(
        name="ndcg",
        picture=ndcg_picture,
        function="ndcg",
        start=_ND_START,
        solution=_ND,
        show=_show,
        names={"dcg": dcg},
        visible=[tuple(_written(v) for v in case) for case in TOY_ND],
        hidden=[tuple(_written(v) for v in case) for case in (
            ([3, 2, 1], [1, 2, 3], 3), ([0, 0], [], 2), ([], [2], 3), ([2, 0, 3], [3, 2], 1), ([0, 0, 0, 2], [2, 2], 4),
            ([1, 1, 1], [1, 1, 1, 3], 3))]
        + [(_course_gains("en", "buy", "bm25"), _course_judged("buy"), _written(5)),
           (_course_gains("en", "wine", "cosine"), _course_judged("wine"), _written(5)),
           (_course_gains("nl", "learn", "bm25"), _course_judged("learn"), _written(5)),
           (_course_question("nl", "cosine"), _written([1]), _written(10))],
        first=[_DOTS],
        mistakes=[
            Mistake("own", wrong=_ND_OWN,
                    en="`{call}` gives {got}, and it should give {want}: your ideal order comes from the gains of the ranking itself, so a relevant document the ranking misses never counts. Of which grades should the ideal order be made?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je ideale volgorde komt uit de gains van de rangschikking zelf, dus een relevant document dat de rangschikking mist, telt nooit mee. Van welke cijfers moet de ideale volgorde gemaakt worden?"),
            Mistake("low", wrong=_ND_LOW,
                    en="`{call}` gives {got}, and it should give {want}: your ideal order puts the lowest grades first, so it is not the best order, and nDCG can even come out above 1. How do you turn the order of `sorted` around?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je ideale volgorde zet de laagste cijfers bovenaan, dus het is niet de beste volgorde, en nDCG kan zelfs boven 1 uitkomen. Hoe draai je de volgorde van `sorted` om?"),
            Mistake("as-is", wrong=_ND_AS_IS,
                    en="`{call}` gives {got}, and it should give {want}: your ideal order takes the grades in the order they come in. What should happen to them first?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je ideale volgorde neemt de cijfers in de volgorde van binnenkomst. Wat moet er eerst met de cijfers gebeuren?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the ideal order: all the grades in `judged`, the highest first.",
                      "nl": "Vervang de `...` door de ideale volgorde: alle cijfers in `judged`, het hoogste eerst."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is `best` the DCG of all the grades in `judged`, from high to low?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Is `best` de DCG van alle cijfers in `judged`, van hoog naar laag?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function also work for a query without relevant documents, for an empty ranking, for k = 1, and for the course's gains?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook voor een zoekvraag zonder relevante documenten, voor een lege rangschikking, voor k = 1, en voor de gains van de course?"},
            "correct": {"en": "Right: your function divides the DCG by the DCG of the best order there is. In the picture below, it measures the cosine and BM25 on three test collections: on XQuAD, is the nDCG@10 higher or lower than the MRR, and why?",
                        "nl": "Goed: je functie deelt de DCG door de DCG van de beste volgorde die er is. In het plaatje hieronder meet de functie de cosine similarity en BM25 op drie testcollecties: is de nDCG@10 op XQuAD hoger of lager dan de MRR, en waarom?"},
        },
    ),
    "dcg_all": FunctionBox(
        name="dcg_all",
        picture=dcg_all_picture,
        function="dcg_all",
        start=_DA_START,
        solution=_DA,
        show=_show,
        names={"np": np},
        visible=[tuple(_written(v) for v in case) for case in TOY_DA],
        hidden=[tuple(_written(v) for v in case) for case in (
            (np.array([[3, 2, 3, 0], [1, 0, 3, 2]]), 1), (np.array([[1, 2, 3]]), 3), (np.array([[0, 0], [0, 0], [3, 3]]), 2),
            (np.array([[0.5, 1.5, 2.5, 0.0, 1.0]]), 4))]
        + [(_course_array("en", "bm25"), _written(10)), (_course_array("nl", "cosine"), _written(3))],
        first=[_DOTS],
        mistakes=[
            Mistake("rank", wrong=_DA_RANK,
                    en="`{call}` gives {got}, and it should give {want}: your discounts are the places themselves, so each gain is divided by its place, as in the reciprocal rank. What should your discounts be the log base 2 of?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je discounts zijn de plaatsen zelf, dus elke gain wordt gedeeld door de plaats, zoals bij de reciprocal rank. Waarvan moeten je discounts de log met grondtal 2 zijn?"),
            Mistake("zero", wrong=_DA_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: your discounts start with the log of 1, which is 0, and a gain divided by 0 gives infinity. Which numbers should `np.arange` give, one more than each place?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je discounts beginnen met de log van 1, wat 0 is, en een gain gedeeld door 0 geeft oneindig. Welke getallen moet `np.arange` geven, één meer dan elke plaats?"),
            Mistake("natural", wrong=_DA_NATURAL,
                    en="`{call}` gives {got}, and it should give {want}: `np.log` takes the log base e, not base 2. Which function of NumPy takes the log base 2?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `np.log` neemt de log met grondtal e, niet 2. Welke functie van NumPy neemt de log met grondtal 2?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the discounts of the first `k` places: `np.log2` of 2 up to `k + 1`.",
                      "nl": "Vervang de `...` door de discounts van de eerste `k` plaatsen: `np.log2` van 2 tot en met `k + 1`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Are `discounts` the log base 2 of 2, 3 and on up to `k + 1`, one for each of the first `k` places?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Zijn `discounts` de log met grondtal 2 van 2, 3 en zo verder tot en met `k + 1`, één voor elk van de eerste `k` plaatsen?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function also work for k = 1, for a k as long as the rows, and for the course's gains of XQuAD?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook voor k = 1, voor een k zo lang als de rijen, en voor de gains van XQuAD van de course?"},
            "correct": {"en": "Right: your function computes the DCG of every row at once. In the picture below, it measures the cosine and BM25 on all the questions of XQuAD at once: why does the perfect ranking get 1 for every k?",
                        "nl": "Goed: je functie berekent de DCG van elke rij tegelijk. In het plaatje hieronder meet de functie de cosine similarity en BM25 op alle vragen van XQuAD tegelijk: waarom krijgt de perfecte rangschikking 1 bij elke k?"},
        },
    ),
}

# The sort: what a change does to nDCG@10, with the grade as the gain.
EXERCISES["changes"] = Sort(
    name="changes",
    groups={
        "up": {"en": "nDCG@10 goes up", "nl": "nDCG@10 gaat omhoog"},
        "down": {"en": "nDCG@10 goes down", "nl": "nDCG@10 gaat omlaag"},
        "same": {"en": "nDCG@10 stays the same", "nl": "nDCG@10 blijft gelijk"},
    },
    items={
        "swap": {"en": "the ranking swaps its first result, which has grade 0, with a result of grade 3 at place 6",
                 "nl": "de rangschikking wisselt het eerste resultaat, met cijfer 0, om met een resultaat met cijfer 3 op plaats 6"},
        "missed": {"en": "in place of a result of grade 0 at place 4, the ranking puts a relevant document it had missed",
                   "nl": "op plaats 4 zet de rangschikking een relevant document dat eerst ontbrak, in plaats van een resultaat met cijfer 0"},
        "judges": {"en": "for a query whose first ten results hold relevant documents, the judges find one more document of grade 3, which the ranking does not return",
                   "nl": "bij een zoekvraag met relevante documenten in de eerste tien resultaten vinden de beoordelaars nog een document met cijfer 3, dat de rangschikking niet geeft"},
        "drop": {"en": "a result of grade 2 at place 3 swaps places with the result at place 12, which has grade 0",
                 "nl": "een resultaat met cijfer 2 op plaats 3 ruilt van plaats met het resultaat op plaats 12, met cijfer 0"},
        "tie": {"en": "two results of the same grade swap places",
                "nl": "twee resultaten met hetzelfde cijfer ruilen van plaats"},
        "late": {"en": "the results after place 10 come in another order",
                 "nl": "de resultaten na plaats 10 komen in een andere volgorde"},
    },
    solution={"swap": "up", "missed": "up", "judges": "down", "drop": "down", "tie": "same", "late": "same"},
    nudge={
        "swap": {"en": "Does a gain of 3 count more at place 1 or at place 6, and does the ideal order change?",
                 "nl": "Telt een gain van 3 zwaarder op plaats 1 of op plaats 6, en verandert de ideale volgorde?"},
        "missed": {"en": "What happens to the DCG when a gain above 0 takes the place of a 0, and does the ideal order, which already held that document, change?",
                   "nl": "Wat gebeurt er met de DCG als een gain boven 0 de plaats van een 0 inneemt, en verandert de ideale volgorde, waar dit document al in zat?"},
        "judges": {"en": "Which of the two DCGs changes: that of the ranking, or that of the ideal order?",
                   "nl": "Welke van de twee DCG's verandert: de DCG van de rangschikking, of de DCG van de ideale volgorde?"},
        "drop": {"en": "Which places does nDCG@10 look at, and where does the gain of 2 end up?",
                 "nl": "Naar welke plaatsen kijkt nDCG@10, en waar komt de gain van 2 terecht?"},
        "tie": {"en": "Does the DCG change when two equal gains swap places?",
                "nl": "Verandert de DCG als twee gelijke gains van plaats ruilen?"},
        "late": {"en": "Which places does nDCG@10 look at?",
                 "nl": "Naar welke plaatsen kijkt nDCG@10?"},
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
        "correct": {"en": "All {n} are right. nDCG goes up when a higher grade moves up the ranking or a missing relevant document comes in, and down when the ideal order gets better while the ranking stays as it was, or when a relevant document drops out of the first k. What happens after place k, or between documents of the same grade, does not count.",
                    "nl": "Alle {n} kloppen. nDCG gaat omhoog als een hoger cijfer in de rangschikking stijgt of een ontbrekend relevant document erbij komt, en omlaag als de ideale volgorde beter wordt terwijl de rangschikking hetzelfde blijft, of als een relevant document uit de eerste k valt. Wat er na plaats k gebeurt, of tussen documenten met hetzelfde cijfer, telt niet mee."},
    },
)

# What the tests feed each check, and what it must say: (exercise, code or choices, state, known mistake).
CASES = [
    ("dcg", _DC_START, "empty", None),
    ("dcg", _DC, "correct", None),
    ("dcg", _DC_HEAD + "        total = total + top[place] / np.log2(...)\n" + _DC_END, "wrong", "dots-left"),
    ("dcg", _DC_RANK, "wrong", "rank"),
    ("dcg", _DC_ZERO, "wrong", "zero"),
    ("dcg", _DC_NATURAL, "wrong", "natural"),
    ("dcg", _DC_HEAD + "        total = total + top[place] / np.log2(2 + place)\n" + _DC_END, "correct", None),
    ("dcg", _DC_HEAD + "        total += top[place] / np.log2(place + 2)\n" + _DC_END, "correct", None),
    ("dcg", _DC_HEAD + "        total = total + top[place] / np.log2(place + 3)\n" + _DC_END, "wrong", None),
    ("dcg", "def dcg(gains, k):\n    total = 0\n    for place in range(len(gains)):\n"
     "        total = total + gains[place] / np.log2(place + 2)\n    return total\n", "wrong", None),
    ("ndcg", _ND_START, "empty", None),
    ("ndcg", _ND, "correct", None),
    ("ndcg", _ND_HEAD + "    best = dcg(sorted(...), k)\n" + _ND_END, "wrong", "dots-left"),
    ("ndcg", _ND_OWN, "wrong", "own"),
    ("ndcg", _ND_LOW, "wrong", "low"),
    ("ndcg", _ND_AS_IS, "wrong", "as-is"),
    ("ndcg", _ND_HEAD + "    ideal = sorted(judged, reverse=True)\n    best = dcg(ideal, k)\n" + _ND_END, "correct", None),
    ("ndcg", _ND_HEAD + "    best = dcg(sorted(judged, reverse=True), k)\n    return dcg(gains, k) / best\n", "almost", None),
    ("dcg_all", _DA_START, "empty", None),
    ("dcg_all", _DA, "correct", None),
    ("dcg_all", _DA_HEAD + "    discounts = np.log2(...)\n" + _DA_END, "wrong", "dots-left"),
    ("dcg_all", _DA_RANK, "wrong", "rank"),
    ("dcg_all", _DA_ZERO, "wrong", "zero"),
    ("dcg_all", _DA_NATURAL, "wrong", "natural"),
    ("dcg_all", _DA_HEAD + "    discounts = np.log2(np.arange(k) + 2)\n" + _DA_END, "correct", None),
    ("dcg_all", _DA_HEAD + "    discounts = np.log2(np.arange(2, k + 1))\n" + _DA_END, "error", None),
    ("changes", {}, "empty", None),
    ("changes", dict(EXERCISES["changes"].solution), "correct", None),
    ("changes", {"swap": "up", "tie": "same"}, "wrong", "unfinished"),
] + [("changes", {item: ("same" if group != "same" else "up")}, "wrong", f"wrong:{item}")
     for item, group in EXERCISES["changes"].solution.items()]
