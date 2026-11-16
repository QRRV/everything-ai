# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-retrieval-embeddings: a search for the points whose embeddings have the highest dot product with
the embedding of a question, and a function that embeds only the question and searches the embeddings the course made
once, against a stand-in; Granite Embedding's embeddings of the course's summaries (data/course-summaries-granite.npz),
where the right point of each question ends up by meaning and by BM25, in English and in Dutch; the pictures, and the
demo with the learner's own functions."""
import functools
import html
import json

import numpy as np

from everything_ai import apicheck, lesson as lesson_helpers, serve, summaries, yours
from everything_ai.apicheck import URL, ApiBox, Case
from everything_ai.check import MISSING, FunctionBox, Mistake, run_code, with_time_limit

REPO = "ibm-granite/granite-embedding-107m-multilingual"   # everything_ai.embed.REPO, without importing PyTorch
FILES = [(REPO, "model.safetensors"), (REPO, "tokenizer.json")]
DOWNLOADS = {"Granite Embedding": FILES}   # what the demo downloads, for the line under the header

POINTS = summaries.points("en")   # the knowledge base: every point of every summary, in English
VECTORS = summaries.vectors("points", "en")   # Granite's embedding of each point, a row per point
QUESTIONS = summaries.questions()
ANSWERABLE = [q for q in QUESTIONS if q["kind"] != "none"]
NONE = [q for q in QUESTIONS if q["kind"] == "none"]
ABOUT = summaries.about()
SECONDS = summaries.vectors_about()["seconds"]   # on the course's machine: all points, and one question alone
K = 3
COLUMNS = [("bm25", "en"), ("meaning", "en"), ("bm25", "nl"), ("meaning", "nl")]
EMBED = run_code('''import json
import urllib.request


def embed(url, texts):
    body = {"model": "granite-embedding-107m-multilingual", "input": texts}
    request = urllib.request.Request(url + "/embeddings", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        answer = json.loads(response.read())
    return [item["embedding"] for item in answer["data"]]
''').names["embed"]   # the solution of build-embeddings-api


def question(qid: str) -> dict:
    return next(q for q in QUESTIONS if q["id"] == qid)


def question_vector(q: dict, lang: str = "en") -> np.ndarray:
    return summaries.vectors("questions", lang)[QUESTIONS.index(q)]


def gold(q: dict) -> str:
    return summaries.text_of(q["gold"])


def lesson_of(text: str) -> dict:
    return ABOUT[POINTS.index(text)]


def title_of(text: str, lang: str) -> str:
    """The title of the lesson a point belongs to."""
    return lesson_of(text)["title"][lang]


def gold_score(qid: str) -> float:
    """The score of the right point of a question, by meaning, with the question in English."""
    q = question(qid)
    return float(VECTORS[POINTS.index(gold(q))] @ question_vector(q))


def vocabulary() -> int:
    """The number of different words in all points, as the word search counts them."""
    out = set()
    for text in POINTS:
        out |= summaries.words(text)
    return len(out)


def most_words() -> int:
    """The most different words one point has."""
    return max(len(summaries.words(text)) for text in POINTS)


def number(x: float, lang: str, digits: int = 1) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


@functools.cache
def _bm25_index() -> dict:
    return summaries.bm25_index(POINTS)


def found(q: dict, way: str, lang: str = "en", k: int = K) -> list:
    """The k points a search finds for a question: by BM25 or by meaning, with the question in English or Dutch."""
    if way == "bm25":
        return summaries.bm25_search(q["text"][lang], POINTS, k, _bm25_index())
    return meaning_search(question_vector(q, lang), VECTORS, POINTS, k)


@functools.cache
def ranks() -> dict:
    """Where the right point of each question a point answers ends up, among all points, per way of searching."""
    return {q["id"]: {(way, lang): found(q, way, lang, len(POINTS)).index(gold(q)) + 1 for way, lang in COLUMNS}
            for q in ANSWERABLE}


@functools.cache
def findings() -> dict:
    """How often each way put the right point in its top three, and first, per kind of question; the highest score
    a question no point answers got; how many right points scored lower, of all and of those asked in other words."""
    out = {"questions": len(ANSWERABLE), "words_questions": len(questions_of("words")),
           "meaning_questions": len(questions_of("meaning")),
           "search_meaning": sum(gold(q) in summaries.search(q["text"]["en"], POINTS, K) for q in questions_of("meaning"))}
    for way, lang in COLUMNS:
        name = way if lang == "en" else f"{way}_nl"
        for kind in ("words", "meaning"):
            r = [ranks()[q["id"]][(way, lang)] for q in questions_of(kind)]
            out[f"{name}_{kind}"] = sum(x <= K for x in r)
            out[f"{name}_{kind}_first"] = sum(x == 1 for x in r)
        out[name] = out[f"{name}_words"] + out[f"{name}_meaning"]
    tops = {q["id"]: float(np.max(VECTORS @ question_vector(q))) for q in NONE}
    out["none_high"] = max(tops.values())
    out["none_low"] = min(tops.values())
    out["none_question"] = max(tops, key=tops.get)
    scores = {q["id"]: float(VECTORS[POINTS.index(gold(q))] @ question_vector(q)) for q in ANSWERABLE}
    out["gold_below"] = sum(s < out["none_high"] for s in scores.values())
    out["meaning_below"] = sum(scores[q["id"]] < out["none_high"] for q in questions_of("meaning"))
    return out


def questions_of(kind: str) -> list:
    return [q for q in ANSWERABLE if q["kind"] == kind]


def worked() -> dict:
    """The worked example: the question about k-nearest neighbors, the three points search by meaning finds with
    their scores, the range of all scores, and what BM25 made of it."""
    q = question("q21")
    scores = VECTORS @ question_vector(q)
    top = found(q, "meaning")
    return {"question": q["text"]["en"], "top": top, "scores": [float(scores[POINTS.index(t)]) for t in top],
            "low": float(scores.min()), "lessons": [lesson_of(t)["lesson"] for t in top],
            "shared": sorted(summaries.words(q["text"]["en"]) & summaries.words(gold(q))),
            "bm25_rank": ranks()["q21"][("bm25", "en")], "bm25_first": lesson_of(found(q, "bm25")[0])["lesson"]}


# --- the picture of what the course measured ----------------------------------------------------------------------------

_LABEL = "font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)"


def _swatch(color: str, label: str, dashed: bool = False) -> str:
    box = "border:1px dashed var(--ea-line)" if dashed else f"background:var(--ea-{color})"
    return (f'<span style="display:inline-flex;align-items:center;gap:0.25rem;margin-right:0.8rem;{_LABEL}">'
            f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;{box}"></span>{html.escape(label)}</span>')


_RANKS_WORDS = {
    "en": {"title": "Where the right point ends up among all {n}: by BM25, and by meaning", "question": "question",
           "english": "the question in English", "dutch": "the question in Dutch", "bm25": "BM25", "meaning": "meaning",
           "top": "in the top three", "lower": "lower", "count": "{k} of {n}", "other": "asks it in other words",
           "alt": "A table of the {k} questions a point answers. For each, the place of the right point among all points, by BM25 and by meaning, with the question in English and in Dutch; a place in the top three is colored."},
    "nl": {"title": "Waar het goede punt uitkomt tussen alle {n}: met BM25, en op betekenis", "question": "vraag",
           "english": "de vraag in het Engels", "dutch": "de vraag in het Nederlands", "bm25": "BM25", "meaning": "betekenis",
           "top": "bij de bovenste drie", "lower": "lager", "count": "{k} van de {n}", "other": "vraagt het in andere woorden",
           "alt": "Een tabel van de {k} vragen die een punt beantwoordt. Bij elke vraag de plaats van het goede punt tussen alle punten, met BM25 en op betekenis, met de vraag in het Engels en in het Nederlands; een plaats bij de bovenste drie is gekleurd."},
}


def _rank_cell(r: int, lang: str) -> str:
    top = r <= K
    style = ("background:var(--ea-data-wash);border:1px solid var(--ea-data);color:var(--ea-ink)" if top else
             "border:1px dashed var(--ea-line);color:var(--ea-ink-3)")
    return (f'<span style="display:inline-block;min-width:2.6rem;text-align:center;padding:0 0.2rem;border-radius:3px;'
            f'font-family:var(--ea-font-code);font-size:12px;{style}">{html.escape(number(r, lang, 0))}</span>')


def ranks_picture(lang: str) -> str:
    """A row per question a point answers: the place of its right point by BM25 and by meaning, with the question in
    English and in Dutch; a place in the top three as retrieved data."""
    w, f = _RANKS_WORDS[lang], findings()
    group = (f'<span></span><span style="{_LABEL};grid-column:span 2;text-align:center;border-bottom:1px solid var(--ea-line)">{html.escape(w["english"])}</span>'
             f'<span style="{_LABEL};grid-column:span 2;text-align:center;border-bottom:1px solid var(--ea-line)">{html.escape(w["dutch"])}</span>')
    head = f'<span style="{_LABEL}">{html.escape(w["question"])}</span>' + "".join(
        f'<span style="{_LABEL};text-align:center">{html.escape(w[way])}</span>' for way, _ in COLUMNS)
    rows = []
    for q in ANSWERABLE:
        mark = f' <span style="{_LABEL}">({html.escape(w["other"])})</span>' if q["kind"] == "meaning" else ""
        cells = "".join(f'<span style="text-align:center">{_rank_cell(ranks()[q["id"]][c], lang)}</span>' for c in COLUMNS)
        rows.append(f'<span style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink)">{html.escape(q["text"][lang])}{mark}</span>{cells}')
    foot = '<span></span>' + "".join(
        f'<span style="{_LABEL};text-align:center">{html.escape(w["count"].format(k=f[way if l == "en" else way + "_nl"], n=f["questions"]))}</span>'
        for way, l in COLUMNS)
    grid = (f'<div style="display:grid;grid-template-columns:minmax(0,1fr) repeat(4,auto);gap:0.3rem 0.5rem;align-items:center">'
            f'{group}{head}{"".join(rows)}{foot}</div>')
    legend = _swatch("data", w["top"]) + _swatch("line", w["lower"], dashed=True)
    title = w["title"].format(n=number(len(POINTS), lang, 0))
    return yours.figure(title, grid + f'<div style="margin-top:0.4rem">{legend}</div>', alt=w["alt"].format(k=len(ANSWERABLE)))


# --- exercise 1: meaning_search ------------------------------------------------------------------------------------------

START_SEARCH = '''def meaning_search(q, vectors, points, k):
    found = []
    for i in range(len(points)):
        found.append((0, points[i]))
    return [point for score, point in sorted(found, reverse=True)[:k]]
'''
SOLUTION_SEARCH = START_SEARCH.replace("found.append((0, points[i]))", "found.append((vectors[i] @ q, points[i]))")
LOWEST = SOLUTION_SEARCH.replace("sorted(found, reverse=True)", "sorted(found)")
SAME_ROW = SOLUTION_SEARCH.replace("vectors[i] @ q", "vectors[0] @ q")
PAIRS = SOLUTION_SEARCH.replace("    return [point for score, point in sorted(found, reverse=True)[:k]]\n",
                                "    return sorted(found, reverse=True)[:k]\n")
meaning_search = run_code(SOLUTION_SEARCH).names["meaning_search"]


class _Array(np.ndarray):
    """An array that a check message writes the way the lesson's code does: np.array([...]). The learner's code gets
    a plain copy."""

    def __repr__(self):
        return f"np.array({json.dumps(np.asarray(self).tolist())})"

    def __deepcopy__(self, memo):
        return np.array(self)


def _a(values) -> np.ndarray:
    return np.array(values, dtype=float).view(_Array)


def _plain(v):
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float):
        return round(v, 3)
    if isinstance(v, (list, tuple)):
        return type(v)(_plain(x) for x in v)
    return v


def _show(v) -> str:
    """What a function gave back, for the messages: texts in double quotes, numbers rounded to three decimals."""
    v = _plain(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ", ".join(_show(x) for x in v) + "]"
    if isinstance(v, tuple):
        return "(" + ", ".join(_show(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    return " ".join(repr(v).split())


SMALL = ["The shop opens at nine on Saturday.", "A full service costs 85 euros.", "Brake pads cost 12 euros per wheel.",
         "Rental bikes come with a lock."]
SMALL_VECTORS = _a([[0.0, 1.0], [0.6, 0.8], [1.0, 0.0], [-0.6, 0.8]])   # made-up embeddings of length 1, two numbers each
OTHER = ["Tyres are pumped for free.", "The shop is closed on Sunday.", "A bell costs 4 euros."]
OTHER_VECTORS = _a([[0.0, 0.6, 0.8], [1.0, 0.0, 0.0], [0.0, 0.8, 0.6]])
EXAMPLES = {"en": ["q18", "q21", "q20"], "nl": ["q17", "q21", "q20"]}   # the picture: found, found, and missed

_SEARCH_WORDS = {
    "en": {"title": "The three points your meaning_search finds among all {n}, for three questions", "score": "{s}",
           "gold": "the point that answers it", "missing": "The point that answers it is not among them: it is number {r}."},
    "nl": {"title": "De drie punten die je meaning_search vindt tussen alle {n}, voor drie Nederlandse vragen", "score": "{s}",
           "gold": "het punt dat hem beantwoordt", "missing": "Het punt dat hem beantwoordt zit er niet bij: het staat op plaats {r}."},
}


def _found_rows(got: list, q: dict, qv, lang: str, words: dict) -> str:
    rows = []
    for text in got[:K]:
        text = text if isinstance(text, str) else repr(text)
        is_gold = text == gold(q)
        style = "background:var(--ea-data-wash);border-left:3px solid var(--ea-data)" if is_gold else "border-left:3px solid var(--ea-line)"
        score = number(float(VECTORS[POINTS.index(text)] @ qv), lang, 3) if text in POINTS else "?"
        rows.append(f'<div style="margin:2px 0;padding:0.1rem 0.4rem;{style};font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink);'
                    f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis"><span style="font-family:var(--ea-font-code);color:var(--ea-ink-2)">'
                    f'{html.escape(score)}</span> {html.escape(text)}</div>')
    if gold(q) not in got[:K]:
        r = meaning_search(qv, VECTORS, POINTS, len(POINTS)).index(gold(q)) + 1
        rows.append(f'<div style="{_LABEL};color:var(--ea-loss-text)">{html.escape(words["missing"].format(r=r))}</div>')
    return "".join(rows)


def search_picture(seen) -> str:
    """The learner's meaning_search on all points for three questions, in the lesson's language: per question the
    three it finds, with the score of each, the point that answers it as retrieved data."""
    w = _SEARCH_WORDS[seen.lang]
    mine = seen.mine
    fn = mine.names.get("meaning_search") if mine is not None else None
    if not callable(fn):
        return ""
    blocks = []
    for qid in EXAMPLES[seen.lang]:
        q = question(qid)
        qv = question_vector(q, seen.lang)
        try:
            got = with_time_limit(lambda: fn(qv.copy(), VECTORS.copy(), list(POINTS), K), 10)
        except Exception:   # the learner's code; the check says what went wrong
            return ""
        if not isinstance(got, list):
            return ""
        blocks.append(f'<div style="margin:0.3rem 0 0.5rem"><div style="font-family:var(--ea-font-ui);font-size:13px;color:var(--ea-ink)">'
                      f'<em>{html.escape(q["text"][seen.lang])}</em></div>{_found_rows(got, q, qv, seen.lang, w)}</div>')
    title = w["title"].format(n=number(len(POINTS), seen.lang, 0))
    return yours.figure(title, "".join(blocks) + f'<div>{_swatch("data", w["gold"])}</div>', alt=title)


# --- exercise 2: retrieve ------------------------------------------------------------------------------------------------

START_RETRIEVE = '''import numpy as np


def retrieve(url, question, k):
    vectors = embed(url, [question] + POINTS)
    q = np.array(vectors[0])
    return meaning_search(q, np.array(vectors[1:]), POINTS, k)
'''
SOLUTION_RETRIEVE = '''import numpy as np


def retrieve(url, question, k):
    q = np.array(embed(url, [question])[0])
    return meaning_search(q, VECTORS, POINTS, k)
'''
RETRIEVE_NAMES = {"embed": EMBED, "meaning_search": meaning_search, "VECTORS": VECTORS, "POINTS": POINTS}
RETRIEVE_QUESTIONS = [("q21", 3), ("q18", 5)]


def _stand_in(q: dict):
    return lambda: serve.StandIn(vectors={q["text"]["en"]: [float(x) for x in question_vector(q)]})


def _sent(running) -> list:
    """Every text the function sent for embeddings, over all its requests."""
    out = []
    for e in running.log:
        body = e["body"] if e["method"] == "POST" and isinstance(e["body"], dict) else {}
        texts = body.get("input")
        out += texts if isinstance(texts, list) else [texts] if texts is not None else []
    return out


def _judge_retrieve(q: dict, k: int):
    text = q["text"]["en"]
    qv = np.array([float(x) for x in question_vector(q)])
    want = meaning_search(qv, VECTORS, POINTS, k)
    ranking = meaning_search(qv, VECTORS, POINTS, len(POINTS))   # every point, best first

    def judge(got, running):
        sent = _sent(running)
        if len(sent) > 1:
            return "again"
        if sent != [text]:
            return "other-text"
        if got == want:
            return None
        if got is None:
            return "none"
        if isinstance(got, list) and got and len(got) != k and got == ranking[:len(got)]:
            return "fixed-k"
        return "other"
    return judge


RETRIEVE_CASES = [Case(_stand_in(question(qid)), (URL, question(qid)["text"]["en"], k), _judge_retrieve(question(qid), k))
                  for qid, k in RETRIEVE_QUESTIONS]

_RETRIEVE_WORDS = {
    "en": {"title": "What your retrieve sent to the server, and what it found",
           "sent": "Your retrieve sent {texts} to the server, in {requests}.",
           "text": "{n} text", "texts": "{n} texts", "request": "{n} request", "requests": "{n} requests",
           "again": "All the points went along again, for one question: their embeddings are in VECTORS already.",
           "found": "What it found for: {question}", "gold": "the point that answers it",
           "missing": "The point that answers it is not among them: it is number {r}.",
           "error": "Your retrieve stops with an error."},
    "nl": {"title": "Wat je retrieve naar de server stuurde, en wat hij vond",
           "sent": "Je retrieve stuurde {texts} naar de server, in {requests}.",
           "text": "{n} tekst", "texts": "{n} teksten", "request": "{n} request", "requests": "{n} requests",
           "again": "Alle punten gingen weer mee, voor één vraag: hun embeddings staan al in VECTORS.",
           "found": "Wat hij vond voor: {question}", "gold": "het punt dat hem beantwoordt",
           "missing": "Het punt dat hem beantwoordt zit er niet bij: het staat op plaats {r}.",
           "error": "Je retrieve stopt met een fout."},
}


def _count(n: int, one: str, many: str, w: dict, lang: str) -> str:
    return (w[one] if n == 1 else w[many]).format(n=number(n, lang, 0))


def retrieve_picture(seen) -> str:
    """How many texts the learner's retrieve sent, in how many requests, and the points it found for the first case,
    with their scores and the point that answers it as retrieved data."""
    w = _RETRIEVE_WORDS[seen.lang]
    run = seen.run
    fn = run.names.get("retrieve") if run is not None and run.error is None else None
    if not callable(fn):
        return ""
    case = RETRIEVE_CASES[0]
    got, error, running = apicheck.call(fn, case)
    sent = _sent(running)
    requests = sum(1 for e in running.log if e["method"] == "POST")
    body = f'<div style="{_LABEL}">{html.escape(w["sent"].format(texts=_count(len(sent), "text", "texts", w, seen.lang), requests=_count(requests, "request", "requests", w, seen.lang)))}</div>'
    if len(sent) > 1:
        body += f'<div style="{_LABEL};color:var(--ea-loss-text)">{html.escape(w["again"])}</div>'
    if error is not None:
        body += f'<p style="{_LABEL};color:var(--ea-loss-text)">{html.escape(w["error"])}</p>'
    elif isinstance(got, list):
        q = question(RETRIEVE_QUESTIONS[0][0])
        body += (f'<div style="font-family:var(--ea-font-ui);font-size:13px;color:var(--ea-ink);margin-top:0.4rem">'
                 f'{html.escape(w["found"].format(question=q["text"]["en"]))}</div>'
                 f'{_found_rows(got, q, question_vector(q), seen.lang, w)}<div>{_swatch("data", w["gold"])}</div>')
    return yours.figure(w["title"], body, alt=w["title"])


def _shape_error(e) -> bool:
    return isinstance(e, ValueError) and "matmul" in str(e)


def _text_error(e) -> bool:
    return isinstance(e, TypeError) and ("@" in str(e) or "matmul" in str(e))


_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}

EXERCISES = {
    "meaning_search": FunctionBox(
        name="meaning_search",
        picture=search_picture,
        function="meaning_search",
        save_as=("app", "meaning"),   # Project: your assistant reads your documents builds on it
        start=START_SEARCH,
        solution=SOLUTION_SEARCH,
        show=_show,
        visible=[(_a([0.8, 0.6]), SMALL_VECTORS, SMALL, 2)],
        hidden=[(_a([0.8, 0.6]), SMALL_VECTORS, SMALL, 0), (_a([-0.6, 0.8]), SMALL_VECTORS, SMALL, 1),
                (_a([-0.6, 0.8]), SMALL_VECTORS, SMALL, 10), (_a([0.0, 0.0, 1.0]), OTHER_VECTORS, OTHER, 2)],
        mistakes=[
            Mistake("lowest", wrong=LOWEST,
                    en="`{call}` gives `{got}`: the points that fit the question worst come first. The higher the dot product, the more two embeddings point the same way: sort so that the highest score comes first. A distance works the other way round: there the smallest is the closest.",
                    nl="`{call}` geeft `{got}`: de punten die het slechtst bij de vraag passen, komen eerst. Hoe hoger het inproduct, hoe meer twee embeddings dezelfde kant op wijzen: sorteer zo dat de hoogste score eerst komt. Een afstand werkt andersom: daar is de kleinste het dichtstbij."),
            Mistake("same-row", wrong=SAME_ROW,
                    en="`{call}` gives `{got}`: every point got the same score, so `sorted` puts them in reverse alphabetical order. The score of `points[i]` compares its own embedding, `vectors[i]`, with `q`.",
                    nl="`{call}` geeft `{got}`: elk punt kreeg dezelfde score, dus zet `sorted` ze in omgekeerde alfabetische volgorde. De score van `points[i]` vergelijkt zijn eigen embedding, `vectors[i]`, met `q`."),
            Mistake("pairs", wrong=PAIRS,
                    en="`{call}` gives `{got}`: pairs of a score and a point. Give back the points alone, as the last line of the box did.",
                    nl="`{call}` geeft `{got}`: paren van een score en een punt. Geef alleen de punten terug, zoals de laatste regel van het vak deed."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`meaning_search` gives every point the score 0, so `sorted` puts them in reverse alphabetical order. Give each point the dot product of its embedding with `q` instead.",
                      "nl": "`meaning_search` geeft elk punt de score 0, dus zet `sorted` ze in omgekeerde alfabetische volgorde. Geef elk punt in plaats daarvan het inproduct van zijn embedding met `q`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: the `k` points whose embeddings have the highest dot product with `q`, the highest first.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: de `k` punten waarvan de embedding het hoogste inproduct met `q` heeft, de hoogste eerst."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work with `k` 0, with a `k` larger than the number of points, and with embeddings of three numbers?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie met `k` 0, met een `k` groter dan het aantal punten, en met embeddings van drie getallen?"},
            "no_function": {"en": "Your box does not make a function `meaning_search`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `meaning_search`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `meaning_search` is right in all {n} cases: the `k` points whose embeddings have the highest dot product with the question's. The check kept it as a file, `{saved}`.",
                        "nl": "Je `meaning_search` klopt in alle {n} gevallen: de `k` punten waarvan de embedding het hoogste inproduct heeft met die van de vraag. De check heeft hem als bestand bewaard, `{saved}`."},
            "kept": {"en": "Your `meaning_search` is right. You changed `{saved}` yourself, so the check left it as it is.",
                     "nl": "Je `meaning_search` is goed. Je hebt `{saved}` zelf veranderd, dus de check liet het staan zoals het is."},
        },
    ),
    "retrieve": ApiBox(
        name="retrieve",
        picture=retrieve_picture,
        function="retrieve",
        start=START_RETRIEVE,
        solution=SOLUTION_RETRIEVE,
        names=RETRIEVE_NAMES,
        cases=RETRIEVE_CASES,
        mistakes=[
            Mistake("first-embedding", error=_shape_error,
                    en="With `{call}`, `meaning_search` got a `q` of the wrong shape: a list with one embedding in it, not the embedding itself. `embed` gives one embedding for each text you sent; take the first, with `[0]`.",
                    nl="Met `{call}` kreeg `meaning_search` een `q` met de verkeerde vorm: een lijst met één embedding erin, niet de embedding zelf. `embed` geeft één embedding voor elke tekst die je stuurde; neem de eerste, met `[0]`."),
            Mistake("question-text", error=_text_error,
                    en="With `{call}`, `meaning_search` got the question as text. It compares embeddings: embed the question first with `embed`, and give it the embedding.",
                    nl="Met `{call}` kreeg `meaning_search` de vraag als tekst. Hij vergelijkt embeddings: haal eerst de embedding van de vraag op met `embed`, en geef hem die."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`retrieve` sends the question together with all " + number(len(POINTS), "en", 0) + " points, for every question, and so embeds the knowledge base again each time. Embed only the question, and search `VECTORS`, the embeddings the course made of the points once.",
                      "nl": "`retrieve` stuurt de vraag samen met alle " + number(len(POINTS), "nl", 0) + " punten, bij elke vraag, en haalt dus elke keer opnieuw de embeddings van de hele knowledge base op. Haal alleen de embedding van de vraag op, en zoek in `VECTORS`, de embeddings die de course één keer van de punten maakte."},
            "again": {"en": "With `{call}`, your function sent more than the question to the server: it embeds points again for one question. Their embeddings are in `VECTORS` already, made once; send only `[question]`.",
                      "nl": "Met `{call}` stuurde je functie meer dan de vraag naar de server: hij haalt opnieuw embeddings van punten op voor één vraag. Die staan al in `VECTORS`, één keer gemaakt; stuur alleen `[question]`."},
            "other-text": {"en": "With `{call}`, the server did not get the question as it is. Send `[question]`, the text the function gets.",
                           "nl": "Met `{call}` kreeg de server de vraag niet zoals hij is. Stuur `[question]`, de tekst die de functie krijgt."},
            "fixed-k": {"en": "With `{call}`, your function gives another number of points than `k`. Pass `k` on to `meaning_search`.",
                        "nl": "Met `{call}` geeft je functie een ander aantal punten dan `k`. Geef `k` door aan `meaning_search`."},
            "none": {"en": "With `{call}`, your function gives nothing back. End with `return` and what `meaning_search` gives.",
                     "nl": "Met `{call}` geeft je functie niets terug. Eindig met `return` en wat `meaning_search` geeft."},
            "other": {"en": "With `{call}`, your function does not give the `k` points of `POINTS` whose embeddings in `VECTORS` fit the question's best. Search `VECTORS` with the question's embedding, and give back what `meaning_search` gives.",
                      "nl": "Met `{call}` geeft je functie niet de `k` punten van `POINTS` waarvan de embeddings in `VECTORS` het best bij die van de vraag passen. Zoek in `VECTORS` met de embedding van de vraag, en geef terug wat `meaning_search` geeft."},
            "no_function": {"en": "Your box does not make a function `retrieve`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `retrieve`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `retrieve` embeds only the question, in one short request, and searches the embeddings the course made of the points once.",
                        "nl": "Je `retrieve` haalt alleen de embedding van de vraag op, in één kort request, en zoekt in de embeddings die de course één keer van de punten maakte."},
        },
    ),
}

REQUIRED = ["meaning_search", "retrieve", "high-score", "dutch-question"]

CASES = [
    ("meaning_search", START_SEARCH, "empty", None),
    ("meaning_search", "", "empty", None),
    ("meaning_search", SOLUTION_SEARCH, "correct", None),
    ("meaning_search", START_SEARCH.replace("    found = []\n", "    scores = vectors @ q\n    found = []\n")
                                   .replace("found.append((0, points[i]))", "found.append((scores[i], points[i]))"), "correct", None),
    ("meaning_search", SOLUTION_SEARCH.replace("vectors[i] @ q", "np.dot(vectors[i], q) / np.linalg.norm(vectors[i])")
                                      .replace("def meaning_search", "import numpy as np\n\n\ndef meaning_search"), "correct", None),
    ("meaning_search", LOWEST, "wrong", "lowest"),
    ("meaning_search", SOLUTION_SEARCH.replace("vectors[i] @ q", "-(vectors[i] @ q)"), "wrong", "lowest"),
    ("meaning_search", SAME_ROW, "wrong", "same-row"),
    ("meaning_search", SOLUTION_SEARCH.replace("vectors[i] @ q", "q @ q"), "wrong", "same-row"),
    ("meaning_search", PAIRS, "wrong", "pairs"),
    ("meaning_search", SOLUTION_SEARCH.replace("[:k]", "[:k + 1]"), "wrong", None),
    ("meaning_search", SOLUTION_SEARCH.replace("[:k]", "[:2]"), "almost", None),
    ("retrieve", START_RETRIEVE, "empty", None),
    ("retrieve", SOLUTION_RETRIEVE, "correct", None),
    ("retrieve", SOLUTION_RETRIEVE.replace("embed(url, [question])[0]", "embed(url, question)[0]"), "correct", None),
    ("retrieve", START_RETRIEVE.replace("np.array(vectors[1:])", "VECTORS"), "wrong", "again"),
    ("retrieve", SOLUTION_RETRIEVE.replace("embed(url, [question])", "embed(url, [question.lower()])"), "wrong", "other-text"),
    ("retrieve", SOLUTION_RETRIEVE.replace("POINTS, k)", "POINTS, 3)"), "wrong", "fixed-k"),
    ("retrieve", SOLUTION_RETRIEVE.replace("POINTS, k)", "POINTS, 5)"), "wrong", "fixed-k"),
    ("retrieve", SOLUTION_RETRIEVE.replace("    return meaning_search", "    meaning_search"), "wrong", "none"),
    ("retrieve", SOLUTION_RETRIEVE.replace("meaning_search(q, VECTORS, POINTS, k)", "POINTS[:k]"), "wrong", "other"),
    ("retrieve", SOLUTION_RETRIEVE.replace("embed(url, [question])[0]", "embed(url, [question])"), "error", "first-embedding"),
    ("retrieve", SOLUTION_RETRIEVE.replace("meaning_search(q, VECTORS", "meaning_search(question, VECTORS"), "error", "question-text"),
]


# --- the demo: your retrieve, against the course's server with Granite -------------------------------------------------------

def size_mb() -> int:
    """The size of Granite's two files, in MB."""
    from everything_ai import hub
    return round(sum(hub.PINNED[pin]["size"] for pin in FILES) / 1e6)


def downloaded() -> bool:
    from everything_ai import hub
    return all(hub.pinned(*pin) is not None for pin in FILES)


def download_or_stop(button, texts: dict) -> None:
    lesson_helpers.pins_or_stop(button, texts, FILES)


def learner_retrieve(search_text, retrieve_text):
    """The learner's retrieve with their own meaning_search, when both are right; else None."""
    for name, text in (("meaning_search", search_text), ("retrieve", retrieve_text)):
        if text is MISSING or text is None or EXERCISES[name].check(text).state != "correct":
            return None
    mine = run_code(search_text).names["meaning_search"]
    return run_code(retrieve_text, dict(RETRIEVE_NAMES, meaning_search=mine)).names["retrieve"]


def _column(title: str, rows: list) -> str:
    return (f'<div style="min-width:0"><div style="{_LABEL};margin-bottom:0.2rem">{html.escape(title)}</div>'
            + "".join(rows) + "</div>")


def demo(retrieve, question_text: str, words: dict, lang: str) -> str:
    """The learner's retrieve on a question, through the course's server with Granite behind it: the three points it
    found with their scores and lessons, beside the three BM25 finds for the same question, and the time."""
    import time
    question_text = " ".join(question_text.split())
    if not question_text:
        return f"<p><em>{html.escape(words['empty'])}</em></p>"
    running = serve.model_server()
    before = len(running.log)
    start = time.perf_counter()
    try:
        got = retrieve(running.url, question_text, K)
    except Exception as e:   # the learner's code, against the real server; it may have run out of memory
        said = words["memory"] if serve.out_of_memory(e) else words["failed"].format(why=f"{type(e).__name__}: {e}")
        return f"<p><em>{html.escape(said)}</em></p>"
    seconds = time.perf_counter() - start
    answers = [e["answer"] for e in running.log[before:] if e.get("path", "").rstrip("/") == "/v1/embeddings"
               and isinstance(e.get("answer"), dict) and len(e["answer"].get("data", [])) == 1]
    q = np.array(answers[-1]["data"][0]["embedding"]) if answers else None

    def row(text: str, color: str, score: str) -> str:
        lesson_title = lesson_of(text)["title"][lang] if text in POINTS else ""
        return (f'<div style="margin:2px 0;padding:0.1rem 0.4rem;border-left:3px solid var(--ea-{color});font-family:var(--ea-font-ui);'
                f'font-size:12px;color:var(--ea-ink)"><span style="font-family:var(--ea-font-code);color:var(--ea-ink-2)">{html.escape(score)}</span> '
                f'{html.escape(text)} <span style="color:var(--ea-ink-2)">({html.escape(lesson_title)})</span></div>')
    mine = [row(t, "data", number(float(VECTORS[POINTS.index(t)] @ q), lang, 3) if q is not None and t in POINTS else "")
            for t in (got if isinstance(got, list) else [])[:K]]
    bm25 = [row(t, "line", "") for t in summaries.bm25_search(question_text, POINTS, K, _bm25_index())]
    body = (f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,16rem),1fr));gap:0.6rem">'
            f'{_column(words["mine"], mine)}{_column(words["bm25"], bm25)}</div>'
            f'<div style="{_LABEL};margin-top:0.4rem">{html.escape(words["done"].format(seconds=number(seconds, lang, 2)))}</div>')
    title = words["title"].format(question=question_text)
    return yours.figure(title, body, alt=title)
