# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-reranking: the hybrid search of build-hybrid-search gives the 20 best candidates for a question,
and Qwen3 judges each of them together with the question, in the form of a reranker: it answers yes or no, and the
probability of yes, from the logprobs the server gives back, is the score. The course's recorded judgments
(data/course-rag-qwen.json): every candidate of the 28 questions about the course. For the 24 questions a point
answers: where the right point is before and after reranking, and how long the judgments took; the pictures, and the
reranking to try with the learner's own functions."""
import functools
import html
import json

from everything_ai import paths, summaries, yours
from everything_ai.check import MISSING, FunctionBox, Mistake, run_code, with_time_limit

POINTS = summaries.points("en")   # the knowledge base: every point of every summary, in English
VECTORS = summaries.vectors("points", "en")
QUESTIONS = summaries.questions()
ANSWERABLE = [q for q in QUESTIONS if q["kind"] != "none"]
NONE = [q for q in QUESTIONS if q["kind"] == "none"]
ABOUT = summaries.about()
CANDIDATES = 20   # what the hybrid search gives the reranker
K = 3             # the points the answer gets
RRF_K = 60
TOP = 5           # the most likely tokens the judge asks for
# the judge, in the form Qwen3's own reranker (Qwen3-Reranker) is asked in: an instruction, the question, a passage
JUDGE = ('Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
         'Note that the answer can only be "yes" or "no".')
INSTRUCT = "Given a question about a course, retrieve passages that answer the question"
DATA = paths.ROOT / "data" / "course-rag-qwen.json"

rrf = run_code('''def rrf(rankings, k):
    scores = {}
    for ranking in rankings:
        for place in range(len(ranking)):
            text = ranking[place]
            if text not in scores:
                scores[text] = 0
            scores[text] = scores[text] + 1 / (k + place + 1)
    return scores
''').names["rrf"]   # the solution of build-hybrid-search
_INDEX = summaries.bm25_index(POINTS)


def question(qid: str) -> dict:
    return next(q for q in QUESTIONS if q["id"] == qid)


def gold(q: dict) -> str | None:
    return summaries.text_of(q["gold"]) if q["gold"] else None


@functools.cache
def candidates(qid: str) -> tuple:
    """The CANDIDATES best points of the hybrid search of build-hybrid-search for a question in English: BM25 and
    search by meaning over all points, merged with reciprocal rank fusion (k = 60), highest first."""
    q = question(qid)
    words = summaries.bm25_search(q["text"]["en"], POINTS, len(POINTS), _INDEX)
    meaning = summaries.meaning_search(summaries.vectors("questions", "en")[QUESTIONS.index(q)], VECTORS, POINTS, len(POINTS))
    scores = rrf([words, meaning], RRF_K)
    return tuple(text for score, text in sorted([(s, t) for t, s in scores.items()], reverse=True)[:CANDIDATES])


def judge_messages(question_text: str, passage: str) -> list:
    return [{"role": "system", "content": JUDGE},
            {"role": "user", "content": f"<Instruct>: {INSTRUCT}\n<Query>: {question_text}\n<Document>: {passage}"}]


def judge_request(question_text: str, passage: str) -> dict:
    """The request the course sends for one judgment: one token, greedy, with the logprobs of the TOP most likely."""
    return {"model": "qwen3-0.6b", "messages": judge_messages(question_text, passage), "max_tokens": 1, "temperature": 0,
            "logprobs": True, "top_logprobs": TOP}


RECORD = {f'{q["id"]}-{i:02d}': {**{k: v for k, v in judge_request(q["text"]["en"], c).items() if k != "model"}, "seeds": 1}
          for q in QUESTIONS for i, c in enumerate(candidates(q["id"]))}


@functools.cache
def course() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-reranking"]


DIMS = 384   # the numbers in one of Granite's embeddings


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def title_of(text: str, lang: str) -> str:
    """The title of the lesson a point comes from."""
    return ABOUT[POINTS.index(text)]["title"][lang]


class Named(dict):
    """A recorded answer of the server, written in the check's messages by its name."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self.name = name

    def __repr__(self):
        return self.name


class Listed(list):
    """The candidates of a question, written in the check's messages by their name."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self.name = name

    def __repr__(self):
        return self.name


def response(qid: str, i: int) -> Named:
    """The server's answer, as a dict, to the judgment of candidate i (from 0) of a question: the JSON it sent, with
    the logprobs of the one token Qwen3 wrote."""
    a = course()[f"{qid}-{i:02d}"]["answers"][0]
    return Named({"id": f"chatcmpl-{qid}-{i:02d}", "object": "chat.completion", "model": "qwen3-0.6b",
                  "choices": [{"index": 0, "message": {"role": "assistant", "content": a["text"]},
                               "logprobs": {"content": a["logprobs"]}, "finish_reason": a["finish"]}],
                  "usage": {"prompt_tokens": a["prompt_tokens"], "completion_tokens": a["tokens"],
                            "total_tokens": a["prompt_tokens"] + a["tokens"]}}, f'RESPONSES["{qid}"][{i}]')


def made(tops: list, name: str) -> Named:
    """An answer of the server with these most likely first tokens, [(token, logprob), ...], for the checks."""
    content = [{"token": tops[0][0], "logprob": tops[0][1], "bytes": list(tops[0][0].encode()),
                "top_logprobs": [{"token": t, "logprob": lp, "bytes": list(t.encode())} for t, lp in tops]}]
    return Named({"choices": [{"index": 0, "message": {"role": "assistant", "content": tops[0][0]},
                               "logprobs": {"content": content}, "finish_reason": "length"}]}, name)


START_P_YES = '''import math


def p_yes(response):
    first = response["choices"][0]["logprobs"]["content"][0]
    total = 0.0
    for entry in first["top_logprobs"]:
        token = entry["token"]
    return total
'''
SOLUTION_P_YES = START_P_YES.replace('''        token = entry["token"]
''', '''        token = entry["token"]
        if token.strip().lower() == "yes":
            total = total + math.exp(entry["logprob"])
''')
NO_EXP = SOLUTION_P_YES.replace("math.exp(entry[\"logprob\"])", "entry[\"logprob\"]")
EXACT = SOLUTION_P_YES.replace("token.strip().lower() == \"yes\"", "token == \"yes\"")
CHOSEN = START_P_YES.replace('''    return total
''', '''    return math.exp(first["logprob"])
''')
CONTAINS = SOLUTION_P_YES.replace("token.strip().lower() == \"yes\"", "\"yes\" in token.lower()")
p_yes = run_code(SOLUTION_P_YES).names["p_yes"]


def score(qid: str, i: int) -> float:
    """P(yes) of the judgment of candidate i of a question, with the course's p_yes."""
    return p_yes(response(qid, i))


class Unknown(KeyError):
    """What judge raises for a question or passage the course recorded no judgment for, so that the check can tell it
    from any other KeyError in the learner's code."""


def judge(question_text, passage):
    """The recorded answer of the server for a question of the course and one of its candidates."""
    q = next((q for q in QUESTIONS if q["text"]["en"] == question_text), None)
    if q is None:
        raise Unknown("The course has no recorded judgments for this question.")
    try:
        i = candidates(q["id"]).index(passage)
    except ValueError:
        raise Unknown("This passage is not among the 20 candidates of the question.") from None
    return response(q["id"], i)


def ranked(qid: str) -> list:
    """The candidates of a question sorted by P(yes), the highest first."""
    return [c for s, c in sorted([(score(qid, i), c) for i, c in enumerate(candidates(qid))], reverse=True)]


def place(texts, q: dict):
    """The place (from 1) of the right point of a question in a list of texts, or None when it is not in it."""
    g = gold(q)
    return list(texts).index(g) + 1 if g in texts else None


@functools.cache
def findings() -> dict:
    """For the 24 questions a point answers: how often the right point is among the three best and first, after the
    hybrid search and after reranking; which questions reranking lifts into the three best and which it drops out;
    the median seconds of one judgment and of the 20 of a question; how many judgments Qwen3 wrote yes for; and the
    lowest of the best P(yes) of the questions no point answers."""
    f = {"questions": len(ANSWERABLE), "judged": len(RECORD), "rescued": [], "lost": []}
    for key in ("hybrid_top3", "reranked_top3", "hybrid_first", "reranked_first"):
        f[key] = 0
    for q in ANSWERABLE:
        before, after = place(candidates(q["id"]), q), place(ranked(q["id"]), q)
        f["hybrid_top3"] += before is not None and before <= K
        f["reranked_top3"] += after is not None and after <= K
        f["hybrid_first"] += before == 1
        f["reranked_first"] += after == 1
        if after is not None and after <= K and (before is None or before > K):
            f["rescued"].append(q["id"])
        if before is not None and before <= K and (after is None or after > K):
            f["lost"].append(q["id"])
    f["lost_place"] = place(ranked(f["lost"][0]), question(f["lost"][0])) if f["lost"] else None
    seconds = sorted(a["answers"][0]["seconds"] for a in course().values())
    f["judgment"] = seconds[len(seconds) // 2]
    per_question = sorted(sum(course()[f'{q["id"]}-{i:02d}']["answers"][0]["seconds"] for i in range(CANDIDATES)) for q in QUESTIONS)
    f["seconds"] = (per_question[len(per_question) // 2 - 1] + per_question[len(per_question) // 2]) / 2
    f["yes_written"] = sum(a["answers"][0]["text"].strip().lower() == "yes" for a in course().values())
    f["none_p"] = min(max(score(q["id"], i) for i in range(CANDIDATES)) for q in NONE)
    return f


WORKED = "q18"   # the question of exercise 4 of build-hybrid-search: there the merged search put its point at place 15


def worked() -> dict:
    q = question(WORKED)
    cands = list(candidates(WORKED))
    order = ranked(WORKED)
    g = cands.index(gold(q))
    return {"question": q, "candidates": cands, "gold": gold(q), "before": g + 1, "after": order.index(gold(q)) + 1,
            "gold_response": response(WORKED, g), "gold_p": score(WORKED, g), "first_p": score(WORKED, 0),
            "reranked": order, "second_p": score(WORKED, cands.index(order[1]))}


def none_example() -> dict:
    """The question no point answers whose best candidate got the highest P(yes)."""
    best = max(((max((score(q["id"], i), i) for i in range(CANDIDATES)), q) for q in NONE), key=lambda x: x[0][0])
    (p, i), q = best
    return {"question": q, "p": p, "best": candidates(q["id"])[i]}


def request_code(question_text: str) -> str:
    """The request of one judgment as Python code, with the candidate left out."""
    return f'''```python
request = {{
    "model": "qwen3-0.6b",
    "messages": [
        {{"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and "
                                      "the Instruct provided. Note that the answer can only be \\"yes\\" or \\"no\\"."}},
        {{"role": "user", "content": "<Instruct>: {INSTRUCT}\\n"
                                    "<Query>: {question_text}\\n"
                                    "<Document>: ..."}},
    ],
    "max_tokens": 1,
    "temperature": 0,
    "logprobs": True,
    "top_logprobs": 5,
}}
```'''


def _tops(resp: dict) -> list:
    return resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]


def tokens_text(resp: dict, lang: str) -> str:
    """The most likely first tokens of an answer with their logprobs, in words."""
    parts = [f'`{t["token"]!r}` ({number(t["logprob"], lang, 4)})' for t in _tops(resp)]
    joined = {"en": " and ", "nl": " en "}[lang]
    return ", ".join(parts[:-1]) + joined + parts[-1]


def sum_text(resp: dict, lang: str) -> str:
    """The sum p_yes adds up for an answer, as code."""
    return "`" + " + ".join(f'math.exp({t["logprob"]:.4f})' for t in _tops(resp) if t["token"].strip().lower() == "yes") + "`"


_LABEL = "font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)"
_CODE = "font-family:var(--ea-font-code);font-size:12px"


def _swatch(style: str, label: str) -> str:
    return (f'<span style="display:inline-flex;align-items:center;gap:0.25rem;margin-right:0.8rem;{_LABEL}">'
            f'<span style="display:inline-block;width:11px;height:11px;border-radius:2px;{style}"></span>{html.escape(label)}</span>')


_PLACES_WORDS = {
    "en": {"title": "Where the right point is among the {n} candidates: after the hybrid search, and after reranking with Qwen3",
           "search": "place after the hybrid search", "reranked": "place after reranking", "top": "the three best",
           "none": "not among the {n}", "count": "Among the three best: {a} after the search, {b} after reranking"},
    "nl": {"title": "Waar het goede punt staat tussen de {n} kandidaten: na de samengevoegde zoekfunctie, en na reranking met Qwen3",
           "search": "plaats na de samengevoegde zoekfunctie", "reranked": "plaats na reranking", "top": "de drie beste",
           "none": "niet bij de {n}", "count": "Bij de drie beste: {a} na de zoekfunctie, {b} na reranking"},
}
_SEARCH = "box-shadow:inset 0 0 0 2px var(--ea-data)"
_RERANKED = "background:var(--ea-pred)"


def _track(before, after) -> str:
    cells = []
    for p in range(1, CANDIDATES + 1):
        style = "background:var(--ea-inset)" if p <= K else "box-shadow:inset 0 0 0 1px var(--ea-line)"
        if p == after:
            style = _RERANKED + (";" + _SEARCH if p == before else "")
        elif p == before:
            style = _SEARCH
        cells.append(f'<span style="display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:2px;{style}"></span>')
    return f'<span style="white-space:nowrap">{"".join(cells)}</span>'


def places_picture(lang: str) -> str:
    """A row per question a point answers: a track of the 20 places, the three best shaded, with the place of the
    right point after the hybrid search (outlined, as input data) and after reranking (filled, as model output)."""
    w = _PLACES_WORDS[lang]
    f = findings()
    rows = []
    for q in ANSWERABLE:
        before, after = place(candidates(q["id"]), q), place(ranked(q["id"]), q)
        cell = _track(before, after) if before is not None else f'<span style="{_LABEL}">{html.escape(w["none"].format(n=CANDIDATES))}</span>'
        rows.append(f'<span style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink);overflow-wrap:anywhere">{html.escape(q["text"][lang])}</span>{cell}')
    grid = (f'<div style="display:grid;grid-template-columns:minmax(0,1fr) auto;gap:0.25rem 0.8rem;align-items:center">{"".join(rows)}</div>'
            f'<div style="{_LABEL};margin-top:0.4rem">{html.escape(w["count"].format(a=f["hybrid_top3"], b=f["reranked_top3"]))}</div>')
    legend = _swatch(_SEARCH, w["search"]) + _swatch(_RERANKED, w["reranked"]) + _swatch("background:var(--ea-inset)", w["top"])
    title = w["title"].format(n=CANDIDATES)
    return yours.figure(title, grid + f'<div style="margin-top:0.3rem">{legend}</div>', alt=title)


def _bar(p: float, color: str) -> str:
    width = max(0.0, min(1.0, p)) * 100
    return (f'<span style="display:inline-block;width:6rem;height:10px;background:var(--ea-inset);border-radius:2px;vertical-align:middle">'
            f'<span style="display:block;width:{width:.1f}%;height:100%;background:var(--ea-{color});border-radius:2px"></span></span>')


_P_WORDS = {
    "en": {"title": "What your p_yes gives the {n} candidates of the worked example, in the order of the hybrid search",
           "place": "place", "lesson": "point of the lesson", "yours": "your p_yes", "course": "should be {want}",
           "right": "the right point", "none": "Your function gave back {got}, not a number."},
    "nl": {"title": "Wat je p_yes de {n} kandidaten van het uitgewerkte voorbeeld geeft, in de volgorde van de samengevoegde zoekfunctie",
           "place": "plaats", "lesson": "punt van de les", "yours": "je p_yes", "course": "zou {want} moeten zijn",
           "right": "het goede punt", "none": "Je functie gaf {got} terug, geen getal."},
}


def p_yes_picture(seen) -> str:
    """The learner's p_yes on the 20 recorded judgments of the worked example: a bar per candidate in the color of
    model output, the value as a mistake where it differs from the course's, the right point marked as input data."""
    w = _P_WORDS[seen.lang]
    mine = seen.mine
    fn = mine.names.get("p_yes") if mine is not None else None
    if not callable(fn):
        return ""
    title = w["title"].format(n=CANDIDATES)
    rows = []
    g = gold(question(WORKED))
    for i, c in enumerate(candidates(WORKED)):
        try:
            got = with_time_limit(lambda: fn(response(WORKED, i)))
        except Exception:   # the learner's code; the check says what went wrong
            return ""
        if isinstance(got, bool) or not isinstance(got, (int, float)):
            return yours.figure(title, f'<div style="{_LABEL}">{html.escape(w["none"].format(got=repr(got)[:80]))}</div>', alt=title)
        want = score(WORKED, i)
        ok = abs(got - want) <= 1e-6
        value = (f'<span style="{_CODE};color:var(--ea-{"ink" if ok else "loss-text"})">{number(got, seen.lang, 3)}</span>'
                 + ("" if ok else f' <span style="{_LABEL}">{html.escape(w["course"].format(want=number(want, seen.lang, 3)))}</span>'))
        mark = "border-left:3px solid var(--ea-data);padding-left:0.3rem" if c == g else "padding-left:calc(0.3rem + 3px)"
        rows.append(f'<span style="{_LABEL};text-align:right">{i + 1}</span>'
                    f'<span style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink);{mark};overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{html.escape(title_of(c, seen.lang))}</span>'
                    f'<span>{_bar(got if ok else 0.0, "pred")}</span><span>{value}</span>')
    head = "".join(f'<span style="{_LABEL}">{html.escape(w[k])}</span>' for k in ("place", "lesson")) + f'<span></span><span style="{_LABEL}">{html.escape(w["yours"])}</span>'
    body = (f'<div style="display:grid;grid-template-columns:auto minmax(0,1fr) auto auto;gap:0.2rem 0.6rem;align-items:center">{head}{"".join(rows)}</div>'
            f'<div style="margin-top:0.3rem">{_swatch("background:var(--ea-data)", w["right"])}</div>')
    return yours.figure(title, body, alt=title)


# --- exercise 2: rerank --------------------------------------------------------------------------------------------------

START_RERANK = '''def rerank(question, candidates, n):
    scored = []
    for passage in candidates:
        response = judge(question, passage)
    return candidates[:n]
'''
SOLUTION_RERANK = START_RERANK.replace('''        response = judge(question, passage)
    return candidates[:n]
''', '''        response = judge(question, passage)
        scored.append((p_yes(response), passage))
    best = sorted(scored, reverse=True)
    return [passage for score, passage in best[:n]]
''')
LOWEST_FIRST = SOLUTION_RERANK.replace("sorted(scored, reverse=True)", "sorted(scored)")
NO_CUT = SOLUTION_RERANK.replace("best[:n]", "best")
SCORES = SOLUTION_RERANK.replace("[passage for score, passage in best[:n]]", "best[:n]")
RERANK_NAMES = {"judge": judge, "p_yes": p_yes}


def listed(qid: str) -> Listed:
    return Listed(candidates(qid), f'CANDIDATES["{qid}"]')


def _case(qid: str, n: int) -> tuple:
    return (question(qid)["text"]["en"], listed(qid), n)


_RERANK_WORDS = {
    "en": {"title": "The three best your rerank gives for the worked example, out of {n} candidates",
           "place": "place before", "lesson": "point of the lesson", "p": "P(yes)", "right": "the right point",
           "all": "With your rerank, the right point is among the three best for {a} of the {n} questions; after the hybrid search alone, {b}.",
           "none": "Your function gave back {got}, not a list of passages."},
    "nl": {"title": "De drie beste die je rerank geeft voor het uitgewerkte voorbeeld, uit {n} kandidaten",
           "place": "plaats ervoor", "lesson": "punt van de les", "p": "P(yes)", "right": "het goede punt",
           "all": "Met je rerank staat het goede punt bij de drie beste voor {a} van de {n} vragen; na alleen de samengevoegde zoekfunctie {b}.",
           "none": "Je functie gaf {got} terug, geen lijst met passages."},
}


def _rows(texts: list, qid: str, lang: str) -> str:
    cands, g = list(candidates(qid)), gold(question(qid))
    out = []
    for t in texts:
        i = cands.index(t) if t in cands else None
        mark = "border-left:3px solid var(--ea-data);padding-left:0.3rem" if t == g else "padding-left:calc(0.3rem + 3px)"
        out.append(f'<span style="{_LABEL};text-align:right">{"" if i is None else i + 1}</span>'
                   f'<span style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink);{mark};overflow-wrap:anywhere">'
                   f'<em>{html.escape(title_of(t, lang))}</em>: {html.escape(t[:150])}{"…" if len(t) > 150 else ""}</span>'
                   f'<span style="{_CODE};color:var(--ea-pred-text)">{"" if i is None else number(score(qid, i), lang, 3)}</span>')
    return "".join(out)


def _top3(fn) -> int | None:
    """How often the learner's rerank puts the right point among the three best, over the questions a point answers."""
    n = 0
    for q in ANSWERABLE:
        try:
            got = with_time_limit(lambda: fn(*_case(q["id"], K)))
        except Exception:   # the learner's code; the check says what went wrong
            return None
        if not isinstance(got, list):
            return None
        n += gold(q) in got[:K]
    return n


def rerank_picture(seen) -> str:
    """The learner's rerank on the worked example: the three passages it gives, each with its place after the search
    and its P(yes), the right point marked as input data; and how often it puts the right point among the three best
    over all questions a point answers."""
    w = _RERANK_WORDS[seen.lang]
    mine = seen.mine
    fn = mine.names.get("rerank") if mine is not None else None
    if not callable(fn):
        return ""
    title = w["title"].format(n=CANDIDATES)
    try:
        got = with_time_limit(lambda: fn(*_case(WORKED, K)))
    except Exception:   # the learner's code; the check says what went wrong
        return ""
    if not isinstance(got, list) or not all(isinstance(t, str) and t in candidates(WORKED) for t in got):
        return yours.figure(title, f'<div style="{_LABEL}">{html.escape(w["none"].format(got=repr(got)[:80]))}</div>', alt=title)
    head = "".join(f'<span style="{_LABEL}">{html.escape(w[k])}</span>' for k in ("place", "lesson", "p"))
    body = f'<div style="display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:0.25rem 0.6rem;align-items:baseline">{head}{_rows(got[:K], WORKED, seen.lang)}</div>'
    n = _top3(fn)
    if n is not None:
        body += f'<div style="{_LABEL};margin-top:0.4rem">{html.escape(w["all"].format(a=n, n=len(ANSWERABLE), b=findings()["hybrid_top3"]))}</div>'
    body += f'<div style="margin-top:0.3rem">{_swatch("background:var(--ea-data)", w["right"])}</div>'
    return yours.figure(title, body, alt=title)


def _worked_response() -> Named:
    """The recorded answer for the right point of the worked example; before the judgments are recorded, a stand-in."""
    try:
        return response(WORKED, list(candidates(WORKED)).index(gold(question(WORKED))))
    except (KeyError, FileNotFoundError):
        return made([("yes", -0.1), ("no", -2.5)], "ANSWER")


_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}


def _show_p(v):
    """A probability as the check's messages show it: four digits after the point."""
    return round(v, 4) if isinstance(v, float) else v


def _show_texts(v):
    """A list of passages as the check's messages show it: the start of each."""
    if isinstance(v, list) and all(isinstance(t, str) for t in v):
        return [t if len(t) <= 40 else t[:40] + "..." for t in v]
    return v


def _key_error(e) -> bool:
    return isinstance(e, Unknown)


EXERCISES = {
    "p_yes": FunctionBox(
        name="p_yes",
        picture=p_yes_picture,
        function="p_yes",
        start=START_P_YES,
        solution=SOLUTION_P_YES,
        show=_show_p,
        visible=[(_worked_response(),)],
        hidden=[(made([("no", -0.2), ("No", -2.0), ("yes", -2.2), (" yes", -5.0), ("maybe", -6.0)], "ANSWER_A"),),
                (made([("No", -0.01), ("no", -5.0), ("Not", -7.0), ("The", -8.0), ("I", -9.0)], "ANSWER_B"),),
                (made([("Yes", -0.3), ("yes", -1.5), (" Yes ", -3.0), ("no", -4.0), ("No", -5.0)], "ANSWER_C"),),
                (made([("no", -0.5), ("yesterday", -1.0), ("yes", -2.0), ("No", -3.0), ("Yes", -6.0)], "ANSWER_D"),)],
        mistakes=[
            Mistake("no-exp", wrong=NO_EXP,
                    en="`{call}` gives `{got}`: you add up the logprobs themselves, the logarithms of the probabilities, which are below 0. `math.exp(entry[\"logprob\"])` gives the probability back.",
                    nl="`{call}` geeft `{got}`: je telt de logprobs zelf op, de logaritmen van de kansen, en die liggen onder de 0. `math.exp(entry[\"logprob\"])` geeft de kans terug."),
            Mistake("exact", wrong=EXACT,
                    en="`{call}` gives `{got}`: you count only the token `\"yes\"` as it is, not `\"Yes\"` or `\" yes\"`. Compare `token.strip().lower()` with `\"yes\"`.",
                    nl="`{call}` geeft `{got}`: je telt alleen het token `\"yes\"` zoals het is, niet `\"Yes\"` of `\" yes\"`. Vergelijk `token.strip().lower()` met `\"yes\"`."),
            Mistake("chosen", wrong=CHOSEN,
                    en="`{call}` gives `{got}`: that is the probability of the token the model wrote, also when it wrote no. Look at the five most likely tokens, and add those that are yes.",
                    nl="`{call}` geeft `{got}`: dat is de kans van het token dat het model schreef, ook als het no schreef. Kijk naar de vijf waarschijnlijkste tokens, en tel de tokens op die yes zijn."),
            Mistake("contains", wrong=CONTAINS, state="almost",
                    en="Almost. `{call}` gives `{got}`: `\"yes\" in token` also counts `\"yesterday\"`. Count a token only when, without spaces and capitals, it is exactly `\"yes\"`.",
                    nl="Bijna. `{call}` geeft `{got}`: `\"yes\" in token` telt ook `\"yesterday\"`. Tel een token alleen als het zonder spaties en hoofdletters precies `\"yes\"` is."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`p_yes` goes through the five most likely tokens, but adds nothing: it always gives `0.0`. Add the probability of each token that is yes.",
                      "nl": "`p_yes` loopt de vijf waarschijnlijkste tokens langs, maar telt niets op: hij geeft altijd `0.0`. Tel de kans op van elk token dat yes is."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: the sum of `math.exp` of the logprob of every token among the five that is yes, whatever its capitals and spaces.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: de som van `math.exp` van de logprob van elk token van de vijf dat yes is, wat de hoofdletters en spaties ook zijn."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work when yes is not the most likely token, when it is not among the five at all, and for `\"Yes\"` with spaces around it?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie als yes niet het waarschijnlijkste token is, als het helemaal niet bij de vijf staat, en voor `\"Yes\"` met spaties eromheen?"},
            "no_function": {"en": "Your box does not make a function `p_yes`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `p_yes`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `p_yes` is right in all {n} cases: the probability that the answer starts with yes, in any of its forms.",
                        "nl": "Je `p_yes` klopt in alle {n} gevallen: de kans dat het antwoord met yes begint, in welke vorm ook."},
        },
    ),
    "rerank": FunctionBox(
        name="rerank",
        picture=rerank_picture,
        function="rerank",
        start=START_RERANK,
        solution=SOLUTION_RERANK,
        show=_show_texts,
        names=RERANK_NAMES,
        visible=[_case(WORKED, K)],
        hidden=[_case("q21", K), _case("q15", K), _case("q04", 1), _case("q07", 5), _case("q25", 2),
                (question(WORKED)["text"]["en"], Listed(candidates(WORKED)[:5], f'CANDIDATES["{WORKED}"][:5]'), 3)],
        mistakes=[
            Mistake("lowest-first", wrong=LOWEST_FIRST,
                    en="`{call}` gives the candidates with the lowest `P(yes)` first. `sorted` puts the smallest first; with `reverse=True` the highest comes first.",
                    nl="`{call}` geeft de kandidaten met de laagste `P(yes)` eerst. `sorted` zet de kleinste vooraan; met `reverse=True` komt de hoogste eerst."),
            Mistake("no-cut", wrong=NO_CUT,
                    en="`{call}` gives all candidates, not the best `n`. Take only the first `n` of the sorted pairs: `best[:n]`.",
                    nl="`{call}` geeft alle kandidaten, niet de beste `n`. Neem alleen de eerste `n` van de gesorteerde paren: `best[:n]`."),
            Mistake("pairs", wrong=SCORES,
                    en="`{call}` gives pairs of a score and a passage. Give only the passages: `[passage for score, passage in best[:n]]`.",
                    nl="`{call}` geeft paren van een score en een passage. Geef alleen de passages: `[passage for score, passage in best[:n]]`."),
            Mistake("unknown", error=_key_error,
                    en="With `{call}`, `judge` gets a question or passage it has no judgment for. Give `judge` the `question` and each `passage` of `candidates` as they are.",
                    nl="Met `{call}` krijgt `judge` een vraag of passage waar hij geen oordeel voor heeft. Geef `judge` de `question` en elke `passage` van `candidates` zoals ze zijn."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`rerank` asks the judge about every candidate, but gives the first `n` in the order of the search. Score each candidate with `p_yes`, and give the `n` with the highest score.",
                      "nl": "`rerank` vraagt de judge naar elke kandidaat, maar geeft de eerste `n` in de volgorde van de zoekfunctie. Geef elke kandidaat een score met `p_yes`, en geef de `n` met de hoogste score."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: the `n` candidates with the highest `P(yes)`, the highest first.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: de `n` kandidaten met de hoogste `P(yes)`, de hoogste eerst."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for another `n`, and for a list of fewer candidates?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie voor een andere `n`, en voor een lijst met minder kandidaten?"},
            "no_function": {"en": "Your box does not make a function `rerank`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `rerank`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `rerank` is right in all {n} cases: the candidates with the highest `P(yes)`, the highest first.",
                        "nl": "Je `rerank` klopt in alle {n} gevallen: de kandidaten met de hoogste `P(yes)`, de hoogste eerst."},
        },
    ),
}

REQUIRED = ["p_yes", "rerank", "all-points", "no-answer"]

CASES = [
    ("p_yes", START_P_YES, "empty", None),
    ("p_yes", "", "empty", None),
    ("p_yes", SOLUTION_P_YES, "correct", None),
    ("p_yes", START_P_YES.replace('''        token = entry["token"]
''', '''        token = entry["token"]
        if token.lower().strip() in ["yes"]:
            total += math.exp(entry["logprob"])
'''), "correct", None),
    ("p_yes", NO_EXP, "wrong", "no-exp"),
    ("p_yes", EXACT, "wrong", "exact"),
    ("p_yes", CHOSEN, "wrong", "chosen"),
    ("p_yes", CONTAINS, "almost", "contains"),
    ("p_yes", SOLUTION_P_YES.replace("token.strip().lower()", "token.lower()"), "almost", None),
    ("p_yes", SOLUTION_P_YES.replace("total = total + math.exp", "total = math.exp"), "wrong", None),
    ("rerank", START_RERANK, "empty", None),
    ("rerank", "", "empty", None),
    ("rerank", SOLUTION_RERANK, "correct", None),
    ("rerank", START_RERANK.replace('''        response = judge(question, passage)
    return candidates[:n]
''', '''        response = judge(question, passage)
        scored.append((p_yes(response), passage))
    scored.sort(reverse=True)
    return [p for s, p in scored][:n]
'''), "correct", None),
    ("rerank", LOWEST_FIRST, "wrong", "lowest-first"),
    ("rerank", NO_CUT, "wrong", "no-cut"),
    ("rerank", SCORES, "wrong", "pairs"),
    ("rerank", START_RERANK.replace("judge(question, passage)", "judge(passage, question)"), "error", "unknown"),
    ("rerank", SOLUTION_RERANK.replace("p_yes(response)", 'response["choices"][0]["logprob"]'), "error", None),
    ("rerank", SOLUTION_RERANK.replace("best[:n]", "best[:3]"), "almost", None),
]


# --- the reranking to try ------------------------------------------------------------------------------------------------

def question_options(lang: str) -> dict:
    return {q["text"][lang]: q["id"] for q in QUESTIONS}


def learner_rerank(p_text, rerank_text):
    """The learner's rerank with their own p_yes, when both are right; else None."""
    for name, text in (("p_yes", p_text), ("rerank", rerank_text)):
        if text is MISSING or text is None or EXERCISES[name].check(text).state != "correct":
            return None
    mine = run_code(p_text).names["p_yes"]
    return run_code(rerank_text, dict(RERANK_NAMES, p_yes=mine)).names["rerank"]


_TRY_WORDS = {
    "en": {"search": "The five best of the hybrid search", "yours": "The five best of your rerank", "place": "place before",
           "lesson": "point of the lesson", "p": "P(yes)", "right": "the right point", "nothing": "No point of the course answers this question.",
           "failed": "Your rerank stopped: {why}", "title": "Reranked: {question}"},
    "nl": {"search": "De vijf beste van de samengevoegde zoekfunctie", "yours": "De vijf beste van je rerank", "place": "plaats ervoor",
           "lesson": "punt van de les", "p": "P(yes)", "right": "het goede punt", "nothing": "Geen punt van de course beantwoordt deze vraag.",
           "failed": "Je rerank stopte: {why}", "title": "Gererankt: {question}"},
}


def reranked(fn, qid: str, lang: str) -> str:
    """The five best of the hybrid search beside the five best of the learner's rerank, for a question of the course,
    asked in English as the judgments were."""
    w = _TRY_WORDS[lang]
    q = question(qid)
    try:
        got = with_time_limit(lambda: fn(q["text"]["en"], listed(qid), 5))
    except Exception as e:   # the learner's code
        return f'<p><em>{html.escape(w["failed"].format(why=f"{type(e).__name__}: {e}"))}</em></p>'
    head = "".join(f'<span style="{_LABEL}">{html.escape(w[k])}</span>' for k in ("place", "lesson", "p"))

    def block(label: str, texts: list) -> str:
        return (f'<div style="{_LABEL};margin:0.4rem 0 0.2rem">{html.escape(label)}</div>'
                f'<div style="display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:0.25rem 0.6rem;align-items:baseline">{head}{_rows(texts, q["id"], lang)}</div>')
    body = block(w["search"], list(candidates(q["id"]))[:5]) + block(w["yours"], [t for t in got if t in candidates(q["id"])][:5])
    body += f'<div style="margin-top:0.3rem">{_swatch("background:var(--ea-data)", w["right"])}</div>' if q["gold"] else f'<div style="{_LABEL};margin-top:0.3rem">{html.escape(w["nothing"])}</div>'
    title = w["title"].format(question=q["text"][lang])
    return yours.figure(title, body, alt=title)
