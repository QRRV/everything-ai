# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-hybrid-search: reciprocal rank fusion of BM25 (build-bm25) and search by meaning with Granite
Embedding's embeddings (data/course-summaries-granite.npz, build-retrieval-embeddings), on the course's points. rrf
gives every text the sum of 1 / (k + rank) over the rankings, and hybrid_search merges the two searches with it. For the
24 questions a point answers, in English and in Dutch: how often each way puts the right point among the three best,
which questions merging wins and loses, and what k does; the pictures, and the search to try with the learner's own
functions."""
import functools
import html
import json

import numpy as np

from everything_ai import summaries, yours
from everything_ai.check import MISSING, FunctionBox, Mistake, run_code, with_time_limit

POINTS = summaries.points("en")   # the knowledge base: every point of every summary, in English
VECTORS = summaries.vectors("points", "en")
QUESTIONS = summaries.questions()
ANSWERABLE = [q for q in QUESTIONS if q["kind"] != "none"]
ABOUT = summaries.about()
LANGS = ("en", "nl")
K = 3            # the points a search gives
RRF_K = 60       # the k of reciprocal rank fusion that most systems use
SMALL_K = 1      # the other k the course tried
WORKED = "q20"   # the worked example: BM25 puts its point first, search by meaning fifth
LOST = "q18"     # the question of exercise 4: search by meaning puts its point first, BM25 far down
SHOWN = 5        # the highest scores the picture of rrf shows


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def question(qid: str) -> dict:
    return next(q for q in QUESTIONS if q["id"] == qid)


def question_vector(q: dict, lang: str = "en") -> np.ndarray:
    return summaries.vectors("questions", lang)[QUESTIONS.index(q)]


def gold(q: dict) -> str | None:
    return summaries.text_of(q["gold"]) if q["gold"] else None


def title_of(text: str, lang: str) -> str:
    """The title of the lesson a point comes from."""
    return ABOUT[POINTS.index(text)]["title"][lang]


# --- the two searches the learner's code is given ------------------------------------------------------------------------

_INDEX = summaries.bm25_index(POINTS)   # made once, outside the time limit of a check


def bm25_search(question, texts, k):
    """The k texts with the highest BM25 score for the question, highest first: the search of build-bm25."""
    return summaries.bm25_search(question, texts, k, _INDEX if texts == POINTS else None)


def meaning_search(q, vectors, points, k):
    """The k points whose embeddings have the highest dot product with q, highest first: the search of
    build-retrieval-embeddings."""
    return summaries.meaning_search(q, vectors, points, k)


def bm25_scores(question: str) -> list:
    """The BM25 score of every point for a question."""
    asked = sorted(summaries.words(question))
    idfs = {w: _INDEX["idf"].get(w, 0.0) for w in asked}
    return [summaries.bm25(asked, ws, idfs, _INDEX["average"]) for ws in _INDEX["words"]]


@functools.cache
def rankings(qid: str, lang: str) -> tuple:
    """Both searches' rankings of all points for a question in a language: by BM25, and by meaning."""
    q = question(qid)
    return (tuple(bm25_search(q["text"][lang], POINTS, len(POINTS))),
            tuple(meaning_search(question_vector(q, lang), VECTORS, POINTS, len(POINTS))))


# --- exercise 1: rrf -----------------------------------------------------------------------------------------------------

START_RRF = '''def rrf(rankings, k):
    scores = {}
    for ranking in rankings:
        for place in range(len(ranking)):
            text = ranking[place]
            if text not in scores:
                scores[text] = 0
    return scores
'''
SOLUTION_RRF = START_RRF.replace('''                scores[text] = 0
''', '''                scores[text] = 0
            scores[text] = scores[text] + 1 / (k + place + 1)
''')
FROM_ZERO = SOLUTION_RRF.replace("1 / (k + place + 1)", "1 / (k + place)")
LAST_ONLY = SOLUTION_RRF.replace("scores[text] = scores[text] + 1 / (k + place + 1)", "scores[text] = 1 / (k + place + 1)")
NO_K = SOLUTION_RRF.replace("1 / (k + place + 1)", "1 / (place + 1)")
rrf = run_code(SOLUTION_RRF).names["rrf"]


def _same_scores(got, want) -> bool:
    """Two dicts with the same texts and the same scores, also when a score was added up in another order."""
    if not isinstance(got, dict) or set(got) != set(want):
        return False
    try:
        return all(abs(float(got[t]) - want[t]) <= 1e-9 * max(1.0, abs(want[t])) for t in want)
    except (TypeError, ValueError):
        return False


def _show_scores(v) -> str:
    if isinstance(v, dict):
        parts = []
        for t, s in v.items():
            value = f"{s:.4f}" if isinstance(s, (int, float, np.floating)) and not isinstance(s, bool) else repr(s)
            parts.append(f"{json.dumps(t) if isinstance(t, str) else repr(t)}: {value}")
        return "{" + ", ".join(parts) + "}"
    return " ".join(repr(v).split())


_LABEL = "font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)"
_CELL = "padding:0.2rem 0.45rem;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink)"
_NUM = _CELL + ";font-family:var(--ea-font-code);text-align:right;white-space:nowrap"

_RRF_WORDS = {
    "en": {"title": "The {n} highest scores of your rrf, for: {question}", "text": "point", "bm25": "place by BM25",
           "meaning": "place by meaning", "score": "your score", "right": "the point that answers the question",
           "error": "Your rrf stops with an error on the course's two rankings."},
    "nl": {"title": "De {n} hoogste scores van je rrf, voor: {question}", "text": "punt", "bm25": "plaats bij BM25",
           "meaning": "plaats op betekenis", "score": "je score", "right": "het punt dat de vraag beantwoordt",
           "error": "Je rrf stopt met een fout op de twee ranglijsten van de course."},
}


def _swatch(label: str) -> str:
    return (f'<span style="display:inline-flex;align-items:center;gap:0.25rem;{_LABEL}">'
            f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:var(--ea-data)"></span>{html.escape(label)}</span>')


def rrf_picture(seen) -> str:
    """The learner's rrf on the two rankings of the worked example: the points with the highest scores, with their
    places in both rankings."""
    w = _RRF_WORDS[seen.lang]
    mine = seen.mine
    fn = mine.names.get("rrf") if mine is not None else None
    if not callable(fn):
        return ""
    q = question(WORKED)
    words, meaning = rankings(WORKED, "en")
    try:
        got = with_time_limit(lambda: fn([list(words), list(meaning)], RRF_K), 10)
    except Exception:   # the learner's code; the check says what went wrong
        return f'<p style="{_LABEL}">{html.escape(w["error"])}</p>'
    if not isinstance(got, dict):
        return ""
    try:
        best = sorted([(float(s), t) for t, s in got.items() if t in POINTS], reverse=True)[:SHOWN]
    except (TypeError, ValueError):
        return ""
    head = "".join(f'<th style="{_CELL};color:var(--ea-ink-2);font-weight:600;text-align:left">{html.escape(w[c])}</th>'
                   for c in ("text", "bm25", "meaning", "score"))
    rows = []
    for s, t in best:
        mark = "background:var(--ea-data-wash);border-left:3px solid var(--ea-data);" if t == gold(q) else ""
        rows.append(f'<tr><td style="{_CELL};{mark}max-width:24rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{html.escape(t)}</td>'
                    f'<td style="{_NUM}">{number(words.index(t) + 1, seen.lang)}</td><td style="{_NUM}">{number(meaning.index(t) + 1, seen.lang)}</td>'
                    f'<td style="{_NUM}">{number(s, seen.lang, 4)}</td></tr>')
    body = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;min-width:100%;table-layout:fixed">'
            f'<colgroup><col style="width:52%"><col><col><col></colgroup>'
            f'<thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div><div>{_swatch(w["right"])}</div>')
    title = w["title"].format(n=SHOWN, question=q["text"]["en"])
    return yours.figure(title, body, alt=title)


# --- exercise 2: hybrid_search -------------------------------------------------------------------------------------------

START_HYBRID = '''def hybrid_search(question, q, n):
    words = bm25_search(question, POINTS, len(POINTS))
    meaning = meaning_search(q, VECTORS, POINTS, len(POINTS))
    return meaning[:n]
'''
SOLUTION_HYBRID = START_HYBRID.replace('''    return meaning[:n]
''', '''    scores = rrf([words, meaning], 60)
    best = sorted([(score, text) for text, score in scores.items()], reverse=True)
    return [text for score, text in best[:n]]
''')
WORDS_ONLY = SOLUTION_HYBRID.replace("rrf([words, meaning], 60)", "rrf([words], 60)")
UNSORTED = START_HYBRID.replace('''    return meaning[:n]
''', '''    scores = rrf([words, meaning], 60)
    return list(scores)[:n]
''')
HYBRID_NAMES = {"bm25_search": bm25_search, "meaning_search": meaning_search, "rrf": rrf, "POINTS": POINTS, "VECTORS": VECTORS}
hybrid_search = run_code(SOLUTION_HYBRID, dict(HYBRID_NAMES)).names["hybrid_search"]


class _Shown(np.ndarray):
    """The embedding of a question, which a check message writes as the lesson names it: QVECTORS[i]. The learner's
    code gets a plain copy."""

    def __repr__(self):
        return getattr(self, "shown", "q")

    def __deepcopy__(self, memo):
        return np.array(self)


def _q(qid: str) -> np.ndarray:
    v = question_vector(question(qid)).view(_Shown)
    v.shown = "QVECTORS[{i}]".format(i=QUESTIONS.index(question(qid)))
    return v


def _case(qid: str, n: int) -> tuple:
    return (question(qid)["text"]["en"], _q(qid), n)


_HYBRID_WORDS = {
    "en": {"title": "What your hybrid_search finds for: {question}", "bm25": "By words, BM25", "meaning": "By meaning",
           "mine": "Your hybrid_search", "places": "{b} by BM25, {m} by meaning", "right": "the point that answers the question",
           "error": "Your hybrid_search stops with an error on the course's points."},
    "nl": {"title": "Wat je hybrid_search vindt voor: {question}", "bm25": "Op woorden, BM25", "meaning": "Op betekenis",
           "mine": "Je hybrid_search", "places": "{b} bij BM25, {m} op betekenis", "right": "het punt dat de vraag beantwoordt",
           "error": "Je hybrid_search stopt met een fout op de punten van de course."},
}


def _column(title: str, texts: list, words: tuple, meaning: tuple, q: dict, w: dict, lang: str) -> str:
    rows = []
    for t in texts:
        known = t in POINTS
        right = known and t == gold(q)
        style = "background:var(--ea-data-wash);border-left:3px solid var(--ea-data)" if right else "border-left:3px solid var(--ea-line)"
        places = w["places"].format(b=number(words.index(t) + 1, lang), m=number(meaning.index(t) + 1, lang)) if known else ""
        rows.append(f'<div style="margin:2px 0;padding:0.15rem 0.4rem;{style};font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink)">'
                    f'<div style="display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden">{html.escape(t if isinstance(t, str) else repr(t))}</div>'
                    f'<div style="font-family:var(--ea-font-code);font-size:11px;color:var(--ea-ink-2)">{html.escape(places)}</div></div>')
    return f'<div style="min-width:0"><div style="{_LABEL};margin-bottom:0.2rem">{html.escape(title)}</div>{"".join(rows)}</div>'


def searched(fn, qid: str, lang: str, asked: str | None = None) -> str:
    """The learner's hybrid_search on the course's points for one question asked in a language (that of the lesson,
    unless asked says another), beside the three best by BM25 and by meaning, each point with its places in both
    rankings."""
    w = _HYBRID_WORDS[lang]
    q = question(qid)
    asked = asked or lang
    words, meaning = rankings(qid, asked)
    try:
        got = with_time_limit(lambda: fn(q["text"][asked], question_vector(q, asked).copy(), K), 10)
    except Exception:   # the learner's code; the check says what went wrong
        return f'<p style="{_LABEL}">{html.escape(w["error"])}</p>'
    if not isinstance(got, list):
        return ""
    cols = [_column(w["bm25"], list(words[:K]), words, meaning, q, w, lang), _column(w["meaning"], list(meaning[:K]), words, meaning, q, w, lang),
            _column(w["mine"], got[:K], words, meaning, q, w, lang)]
    body = (f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,12rem),1fr));gap:0.6rem">{"".join(cols)}</div>'
            + (f'<div style="margin-top:0.3rem">{_swatch(w["right"])}</div>' if gold(q) else ""))
    title = w["title"].format(question=q["text"][asked])
    return yours.figure(title, body, alt=title)


def hybrid_picture(seen) -> str:
    """The learner's hybrid_search for the question of the worked example, asked in English as there, beside BM25 and
    search by meaning."""
    mine = seen.mine
    fn = mine.names.get("hybrid_search") if mine is not None else None
    if not callable(fn):
        return ""
    return searched(fn, WORKED, seen.lang, "en")


_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}

EXERCISES = {
    "rrf": FunctionBox(
        name="rrf",
        picture=rrf_picture,
        function="rrf",
        start=START_RRF,
        solution=SOLUTION_RRF,
        show=_show_scores,
        compare=_same_scores,
        visible=[([["A", "B", "C"], ["C", "A", "D"]], 60)],
        hidden=[([["A", "B", "C"], ["C", "A", "D"]], 1), ([["A", "B"]], 60), ([], 60), ([["A", "B"], []], 10),
                ([["A", "B", "C"], ["B", "C", "A"], ["C", "A", "B"]], 2)],
        mistakes=[
            Mistake("from-zero", wrong=FROM_ZERO,
                    en="`{call}` gives `{got}`: you counted the places from 0, so the first text of a ranking gets `1 / k`. Its rank is 1: add `1 / (k + place + 1)`.",
                    nl="`{call}` geeft `{got}`: je telde de plaatsen vanaf 0, dus de eerste tekst van een ranglijst krijgt `1 / k`. Zijn rang is 1: tel `1 / (k + place + 1)` op."),
            Mistake("last-only", wrong=LAST_ONLY,
                    en="`{call}` gives `{got}`: every text keeps only the score of the last ranking it is in. Add the new part to the score it has: `scores[text] + ...`.",
                    nl="`{call}` geeft `{got}`: elke tekst houdt alleen de score van de laatste ranglijst waarin hij staat. Tel het nieuwe deel op bij de score die hij al heeft: `scores[text] + ...`."),
            Mistake("no-k", wrong=NO_K,
                    en="`{call}` gives `{got}`: you left out `k`, so a first place counts twice as much as a second. With `k` in the sum, `1 / (k + place + 1)`, the places lie closer together.",
                    nl="`{call}` geeft `{got}`: je liet `k` weg, dus een eerste plaats telt twee keer zo zwaar als een tweede. Met `k` in de som, `1 / (k + place + 1)`, liggen de plaatsen dichter bij elkaar."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "Every text still has the score 0. Add `1 / (k + rank)` to the score of each text, in every ranking it is in; the rank of `ranking[place]` is `place + 1`.",
                      "nl": "Elke tekst heeft nog de score 0. Tel bij de score van elke tekst `1 / (k + rang)` op, in elke ranglijst waarin hij staat; de rang van `ranking[place]` is `place + 1`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: for every text, the sum of `1 / (k + rank)` over the rankings it is in, with rank 1 for the first.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: voor elke tekst de som van `1 / (k + rang)` over de ranglijsten waarin hij staat, met rang 1 voor de eerste."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work with another `k`, with one ranking or none, and with an empty ranking?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie met een andere `k`, met één ranglijst of geen, en met een lege ranglijst?"},
            "no_function": {"en": "Your box does not make a function `rrf`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `rrf`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `rrf` is right in all {n} cases: every text gets the sum of `1 / (k + rank)` over the rankings.",
                        "nl": "Je `rrf` klopt in alle {n} gevallen: elke tekst krijgt de som van `1 / (k + rang)` over de ranglijsten."},
        },
    ),
    "hybrid_search": FunctionBox(
        name="hybrid_search",
        picture=hybrid_picture,
        function="hybrid_search",
        names=HYBRID_NAMES,
        start=START_HYBRID,
        solution=SOLUTION_HYBRID,
        visible=[_case(WORKED, 3)],
        hidden=[_case(LOST, 3), _case("q07", 2), _case("q24", 3), _case("q21", 5), _case(WORKED, 0)],
        mistakes=[
            Mistake("words-only", wrong=WORDS_ONLY,
                    en="`{call}` gives `{got}`: the first of BM25, in its own order. Does `rrf` get both rankings, `[words, meaning]`? And do you sort the pairs of a score and a text, the highest first? A dict keeps the order in which its texts came in.",
                    nl="`{call}` geeft `{got}`: de eerste van BM25, in zijn eigen volgorde. Krijgt `rrf` beide ranglijsten, `[words, meaning]`? En sorteer je de paren van een score en een tekst, de hoogste eerst? Een dict houdt de volgorde waarin zijn teksten erin kwamen."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "The box still gives the first of search by meaning alone. Merge `words` and `meaning` with `rrf`, `k` 60, and give the `n` texts with the highest score.",
                      "nl": "Het vak geeft nog de eerste van alleen zoeken op betekenis. Voeg `words` en `meaning` samen met `rrf`, `k` 60, en geef de `n` teksten met de hoogste score."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: the `n` texts with the highest score of `rrf([words, meaning], 60)`, the highest first.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: de `n` teksten met de hoogste score van `rrf([words, meaning], 60)`, de hoogste eerst."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function give `n` texts for another `n`, also 0, and does it use `k` 60?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Geeft je functie `n` teksten voor een andere `n`, ook 0, en gebruikt hij `k` 60?"},
            "no_function": {"en": "Your box does not make a function `hybrid_search`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `hybrid_search`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `hybrid_search` is right in all {n} cases: it merges the ranking of BM25 and that of search by meaning by their ranks.",
                        "nl": "Je `hybrid_search` klopt in alle {n} gevallen: hij voegt de ranglijst van BM25 en die van zoeken op betekenis samen op hun rangen."},
        },
    ),
}

# The lesson is done when these are right; "add-scores" and "first-place" are choice questions in the lesson.
REQUIRED = ["rrf", "hybrid_search", "add-scores", "first-place"]

CASES = [
    ("rrf", START_RRF, "empty", None),
    ("rrf", "", "empty", None),
    ("rrf", SOLUTION_RRF, "correct", None),
    ("rrf", SOLUTION_RRF.replace("def rrf(", "def fuse("), "wrong", "no-function"),
    ("rrf", "def rrf(rankings, k):\n    scores = {}\n    for ranking in rankings:\n        for rank, text in enumerate(ranking, start=1):\n"
            "            scores[text] = scores.get(text, 0) + 1 / (k + rank)\n    return scores\n", "correct", None),
    ("rrf", FROM_ZERO, "wrong", "from-zero"),
    ("rrf", LAST_ONLY, "wrong", "last-only"),
    ("rrf", NO_K, "wrong", "no-k"),
    ("rrf", SOLUTION_RRF.replace("1 / (k + place + 1)", "1 / (60 + place + 1)"), "almost", None),
    ("rrf", SOLUTION_RRF.replace("1 / (k + place + 1)", "(k + place + 1)"), "wrong", None),
    ("hybrid_search", START_HYBRID, "empty", None),
    ("hybrid_search", "", "empty", None),
    ("hybrid_search", SOLUTION_HYBRID, "correct", None),
    ("hybrid_search", SOLUTION_HYBRID.replace("def hybrid_search(", "def search("), "wrong", "no-function"),
    ("hybrid_search", SOLUTION_HYBRID.replace("rrf([words, meaning], 60)", "rrf([meaning, words], 60)"), "correct", None),
    ("hybrid_search", WORDS_ONLY, "wrong", "words-only"),
    ("hybrid_search", UNSORTED, "wrong", "words-only"),
    ("hybrid_search", SOLUTION_HYBRID.replace("reverse=True", "reverse=False"), "wrong", None),
    ("hybrid_search", SOLUTION_HYBRID.replace("rrf([words, meaning], 60)", "rrf([words, meaning], 1)"), "wrong", None),
    ("hybrid_search", SOLUTION_HYBRID.replace("best[:n]", "best[:3]"), "almost", None),
]


# --- what the course measured --------------------------------------------------------------------------------------------

def merged(qid: str, lang: str, k: int = RRF_K) -> list:
    """The points in the order of the merged search: the highest RRF score of the two rankings first."""
    scores = rrf([list(r) for r in rankings(qid, lang)], k)
    return [t for s, t in sorted([(s, t) for t, s in scores.items()], reverse=True)]


def places(qid: str, lang: str = "en") -> dict:
    """The place of the right point of a question: by BM25, by meaning, merged with k 60 and with k 1."""
    g = gold(question(qid))
    words, meaning = rankings(qid, lang)
    return {"bm25": words.index(g) + 1, "meaning": meaning.index(g) + 1, "hybrid": merged(qid, lang).index(g) + 1,
            "small": merged(qid, lang, SMALL_K).index(g) + 1}


@functools.cache
def findings() -> dict:
    """Per language and way, for how many of the 24 questions the right point is among the three best; per kind of
    question too; and the questions that merging wins and loses against search by meaning, in English."""
    out = {"questions": len(ANSWERABLE), "words_questions": sum(q["kind"] == "words" for q in ANSWERABLE),
           "meaning_questions": sum(q["kind"] == "meaning" for q in ANSWERABLE)}
    for lang in LANGS:
        for way in ("bm25", "meaning", "hybrid", "small"):
            top = [q for q in ANSWERABLE if places(q["id"], lang)[way] <= K]
            out[(lang, way)] = len(top)
            for kind in ("words", "meaning"):
                out[(lang, way, kind)] = sum(q["kind"] == kind for q in top)
    p = {q["id"]: places(q["id"]) for q in ANSWERABLE}
    out["won"] = [qid for qid, r in p.items() if r["hybrid"] <= K < r["meaning"]]
    out["lost"] = [qid for qid, r in p.items() if r["meaning"] <= K < r["hybrid"]]
    return out


def worked() -> dict:
    """The worked example: the three best of both searches, the places and RRF scores of the right point and of the
    first point by meaning, and the range of the scores of both searches."""
    q = question(WORKED)
    words, meaning = rankings(WORKED, "en")
    g = gold(q)
    scores = rrf([list(words), list(meaning)], RRF_K)
    first = meaning[0]
    b = bm25_scores(q["text"]["en"])
    m = VECTORS @ question_vector(q)
    hybrid = merged(WORKED, "en")[:K]
    return {"question": q, "words": list(words[:K]), "meaning": list(meaning[:K]), "gold": g,
            "gold_places": (words.index(g) + 1, meaning.index(g) + 1), "gold_score": scores[g],
            "first": first, "first_places": (words.index(first) + 1, meaning.index(first) + 1), "first_score": scores[first],
            "hybrid": hybrid, "hybrid_places": [(words.index(t) + 1, meaning.index(t) + 1) for t in hybrid],
            "hybrid_scores": [scores[t] for t in hybrid], "bm25_high": max(b), "bm25_low": min(b),
            "meaning_high": float(m.max()), "meaning_low": float(m.min())}


@functools.cache
def summed() -> dict:
    """What the plain sum of the BM25 score and the dot product gives, with the questions in English: for how many of
    the 24 questions the right point is among its three best, and for how many its three best are BM25's three best."""
    right = same = 0
    for q in ANSWERABLE:
        total = np.array(bm25_scores(q["text"]["en"])) + VECTORS @ question_vector(q)
        top = [POINTS[i] for i in np.argsort(-total, kind="stable")[:K]]
        right += gold(q) in top
        same += top == list(rankings(q["id"], "en")[0][:K])
    return {"right": right, "same": same}


def lost() -> dict:
    """The question of exercise 4: the places of its right point, its RRF score, and the merged first point with its
    places and score."""
    words, meaning = rankings(LOST, "en")
    g = gold(question(LOST))
    scores = rrf([list(words), list(meaning)], RRF_K)
    first = merged(LOST, "en")[0]
    return {"places": places(LOST), "score": scores[g], "first": first, "first_places": (words.index(first) + 1, meaning.index(first) + 1),
            "first_score": scores[first]}


_BARS_WORDS = {
    "en": {"title": "The right point among the three best, for the {n} questions a point answers",
           "bm25": "BM25", "meaning": "by meaning", "hybrid": "merged, k = 60", "en": "asked in English", "nl": "asked in Dutch",
           "of": "{x} of {n}"},
    "nl": {"title": "Het goede punt bij de drie beste, voor de {n} vragen die een punt beantwoordt",
           "bm25": "BM25", "meaning": "op betekenis", "hybrid": "samengevoegd, k = 60", "en": "in het Engels gesteld", "nl": "in het Nederlands gesteld",
           "of": "{x} van de {n}"},
}


def bars_picture(lang: str) -> str:
    """Per language of the questions, a bar per way: for how many of the 24 questions the right point is among the
    three best."""
    w = _BARS_WORDS[lang]
    f = findings()
    n = f["questions"]
    groups = []
    for asked in LANGS:
        rows = []
        for way in ("bm25", "meaning", "hybrid"):
            x = f[(asked, way)]
            color = "var(--ea-data)" if way == "hybrid" else "var(--ea-ink-3)"
            rows.append(f'<span style="{_LABEL}">{html.escape(w[way])}</span>'
                        f'<span style="display:block;height:14px;width:{100 * x / n:.1f}%;background:{color};border-radius:2px"></span>'
                        f'<span style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink);white-space:nowrap">{html.escape(w["of"].format(x=x, n=n))}</span>')
        groups.append(f'<div style="margin:0.2rem 0 0.6rem"><div style="font-family:var(--ea-font-ui);font-size:13px;color:var(--ea-ink);margin-bottom:0.2rem">{html.escape(w[asked])}</div>'
                      f'<div style="display:grid;grid-template-columns:minmax(0,9rem) minmax(0,1fr) auto;gap:0.25rem 0.6rem;align-items:center">{"".join(rows)}</div></div>')
    title = w["title"].format(n=n)
    return yours.figure(title, "".join(groups), alt=title)


# --- try it: your hybrid_search on a question of the course --------------------------------------------------------------

def learner_search(rrf_text, hybrid_text):
    """The learner's hybrid_search, with the learner's rrf, when both exercises are right; else None."""
    for name, text in (("rrf", rrf_text), ("hybrid_search", hybrid_text)):
        if text is MISSING or text is None or EXERCISES[name].check(text).state != "correct":
            return None
    names = dict(HYBRID_NAMES)
    names["rrf"] = run_code(rrf_text).names["rrf"]
    return run_code(hybrid_text, names).names["hybrid_search"]


def question_options(lang: str) -> dict:
    return {q["text"][lang]: q["id"] for q in QUESTIONS}
