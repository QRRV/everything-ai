# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-rag-eval: measure the search and the answer of a RAG apart. The course's RAG is that of
build-grounded-answers: search by meaning gives three points, and Qwen3 answers with them after a grounded system
prompt, two runs per question (data/course-rag-qwen.json). For the 24 questions a point answers, the course knows that
point: context recall and context precision say how the search did, the course's check of each answer and the
numbers check of build-grounded-answers how the answer did; the same for the search after the reranker of
build-reranking. And Qwen3 as a judge of the answers (data/course-rag-qwen.json): whether the sources say what an
answer says, and whether it answers the question, as the probability of yes. The pictures."""
import functools
import html
import importlib.util
import json
import math
import re

from everything_ai import paths, summaries, yours
from everything_ai.check import MISSING, FunctionBox, Mistake, run_code, with_time_limit

POINTS = summaries.points("en")   # the knowledge base: every point of every summary, in English
VECTORS = summaries.vectors("points", "en")
QUESTIONS = summaries.questions()
ANSWERABLE = [q for q in QUESTIONS if q["kind"] != "none"]
NONE = [q for q in QUESTIONS if q["kind"] == "none"]
ABOUT = summaries.about()
K = 3
SEEDS = 2
TOP = 5
DATA = paths.ROOT / "data" / "course-rag-qwen.json"
# the judge, in the form of the reranker of build-reranking
JUDGE = ('Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
         'Note that the answer can only be "yes" or "no".')
GROUNDED_JUDGE = "Given an answer as the query and numbered sources as the document, judge whether the sources say everything the answer says"
RELEVANT_JUDGE = "Given a question as the query and an answer as the document, judge whether the document answers the question"


@functools.cache
def _parts() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]


def question(qid: str) -> dict:
    return next(q for q in QUESTIONS if q["id"] == qid)


def gold(q: dict) -> str | None:
    return summaries.text_of(q["gold"]) if q["gold"] else None


def found(q: dict) -> list:
    """The K points search by meaning finds for a question: the search of build-grounded-answers."""
    return summaries.meaning_search(summaries.vectors("questions", "en")[QUESTIONS.index(q)], VECTORS, POINTS, K)


def sources(q: dict) -> str:
    return "\n".join(f"[{i}] {p}" for i, p in enumerate(found(q), start=1))


def answers(qid: str) -> list:
    """The two grounded answers of build-grounded-answers to a question."""
    return [a["text"].strip() for a in _parts()["build-grounded-answers"][f"{qid}-grounded"]["answers"]]


def judge_messages(kind: str, q: dict, answer: str) -> list:
    if kind == "grounded":
        user = f"<Instruct>: {GROUNDED_JUDGE}\n<Query>: {answer}\n<Document>: {sources(q)}"
    else:
        user = f"<Instruct>: {RELEVANT_JUDGE}\n<Query>: {q['text']['en']}\n<Document>: {answer}"
    return [{"role": "system", "content": JUDGE}, {"role": "user", "content": user}]


RECORD = {f'{q["id"]}-{s}-{kind}': {"messages": judge_messages(kind, q, a), "max_tokens": 1, "temperature": 0,
                                    "logprobs": True, "top_logprobs": TOP, "seeds": 1}
          for q in QUESTIONS for s, a in enumerate(answers(q["id"])) for kind in ("grounded", "relevant")}


@functools.cache
def course() -> dict:
    return _parts()["build-rag-eval"]


@functools.cache
def _lesson_checks(lesson_id: str):
    """The checks of another lesson, for what this one shares with it."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_rag_eval",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reranked(q: dict) -> list:
    """The K points the reranker of build-reranking puts first among the 20 candidates of the hybrid search."""
    return _lesson_checks("build-reranking").ranked(q["id"])[:K]


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def title_of(text: str, lang: str) -> str:
    return ABOUT[POINTS.index(text)]["title"][lang]


def p_yes(qid: str, seed: int, kind: str) -> float:
    """The probability of yes in Qwen3's judgment of an answer: kind "grounded" or "relevant"."""
    first = course()[f"{qid}-{seed}-{kind}"]["answers"][0]["logprobs"][0]
    return sum(math.exp(t["logprob"]) for t in first["top_logprobs"] if t["token"].strip().lower() == "yes")


# --- exercise 1: context recall -------------------------------------------------------------------------------------------

START_RECALL = '''def context_recall(retrieved, relevant):
    found = 0
    for point in relevant:
        found = found + 1
    return found / len(relevant)
'''
SOLUTION_RECALL = START_RECALL.replace('''        found = found + 1
''', '''        if point in retrieved:
            found = found + 1
''')
BY_RETRIEVED = SOLUTION_RECALL.replace("return found / len(relevant)", "return found / len(retrieved)")
context_recall = run_code(SOLUTION_RECALL).names["context_recall"]

# --- exercise 2: context precision ----------------------------------------------------------------------------------------

START_PRECISION = '''def context_precision(retrieved, relevant):
    total = 0
    hits = 0
    for i in range(len(retrieved)):
        if retrieved[i] in relevant:
            hits = hits + 1
    return hits / len(retrieved)
'''
SOLUTION_PRECISION = START_PRECISION.replace('''            hits = hits + 1
    return hits / len(retrieved)
''', '''            hits = hits + 1
            total = total + hits / (i + 1)
    if hits == 0:
        return 0.0
    return total / hits
''')
NO_HITS = SOLUTION_PRECISION.replace('''    if hits == 0:
        return 0.0
''', "")
BY_ALL = SOLUTION_PRECISION.replace("return total / hits", "return total / len(retrieved)")
FIRST_ONLY = START_PRECISION.replace('''            hits = hits + 1
    return hits / len(retrieved)
''', '''            return 1 / (i + 1)
    return 0.0
''')
RANK_ONLY = SOLUTION_PRECISION.replace("total = total + hits / (i + 1)", "total = total + 1 / (i + 1)")
context_precision = run_code(SOLUTION_PRECISION).names["context_precision"]


def pipeline(q: dict, way: str) -> list:
    """The three points of a RAG for a question: way "meaning" (the search of build-grounded-answers) or
    "reranked" (the reranker of build-reranking)."""
    return list(found(q)) if way == "meaning" else reranked(q)


@functools.cache
def scores(way: str) -> dict:
    """Context recall and context precision of a search, averaged over the questions a point answers."""
    n = len(ANSWERABLE)
    return {"recall": sum(context_recall(pipeline(q, way), [gold(q)]) for q in ANSWERABLE) / n,
            "precision": sum(context_precision(pipeline(q, way), [gold(q)]) for q in ANSWERABLE) / n}


@functools.cache
def findings() -> dict:
    """For the two answers to each question a point answers: how many are right by the course's check, split by
    whether the search found the right point; how many pass the numbers check of build-grounded-answers; and how
    Qwen3 as a judge scored them: the lowest P(yes), and in what share of the pairs of a right and a wrong answer the
    right one got the higher score. Also how many of all judgments Qwen3 wrote yes for."""
    rows = []
    for q in ANSWERABLE:
        hit = gold(q) in found(q)
        for s, a in enumerate(answers(q["id"])):
            rows.append({"q": q["id"], "s": s, "hit": hit, "right": summaries.right(a, q), "supported": summaries.supported(a, found(q)),
                         "grounded": p_yes(q["id"], s, "grounded"), "relevant": p_yes(q["id"], s, "relevant")})
    f = {"answers": len(rows), "right": sum(r["right"] for r in rows)}
    f["hit_answers"] = sum(r["hit"] for r in rows)
    f["hit_right"] = sum(r["hit"] and r["right"] for r in rows)
    f["miss_answers"] = f["answers"] - f["hit_answers"]
    f["miss_right"] = f["right"] - f["hit_right"]
    f["wrong"] = f["answers"] - f["right"]
    f["wrong_search"] = sum((not r["hit"]) and not r["right"] for r in rows)
    f["wrong_answer"] = sum(r["hit"] and not r["right"] for r in rows)
    f["supported"] = sum(r["supported"] for r in rows)
    f["wrong_supported"] = sum(r["supported"] and not r["right"] for r in rows)
    for kind in ("grounded", "relevant"):
        right = [r[kind] for r in rows if r["right"]]
        wrong = [r[kind] for r in rows if not r["right"]]
        f[f"{kind}_low"] = min(r[kind] for r in rows)
        f[f"{kind}_pairs"] = sum(a > b for a in right for b in wrong) / (len(right) * len(wrong))
    lowest = min(rows, key=lambda r: r["relevant"])
    f["relevant_lowest"] = re.sub(r"\[\d+\]\s*", "", answers(lowest["q"])[lowest["s"]]).strip()
    f["judged"] = len(course())
    f["yes_written"] = sum(a["answers"][0]["text"].strip().lower() == "yes" for a in course().values())
    f["rows"] = rows
    return f


WORKED = "q04"     # the right point is third among the three of search by meaning


def worked_numbers() -> list:
    """The numbers in Qwen3's answer to the worked example, in the order it writes them."""
    a = answers(WORKED)[0]
    found = [int(w.replace(",", "")) for w in summaries.numbers(a) if w.replace(",", "").isdigit()]
    return sorted(found, key=lambda n: a.index(f"{n:,}"))


PRECISION_SHOWN = "q05"   # the right point is second
DIAGNOSED = ("q14", 0)    # an answer that took the wrong number from the right point
ECHO = ("q20", 0)         # an answer that only repeats the question


def rel(qid: str) -> list:
    return [gold(question(qid))]


class Listed(list):
    """A list of points, written in the check's messages by its name."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self.name = name

    def __repr__(self):
        return self.name


def found_listed(qid: str) -> Listed:
    return Listed(found(question(qid)), f'FOUND["{qid}"]')


def relevant_listed(qid: str) -> Listed:
    return Listed(rel(qid), f'RELEVANT["{qid}"]')


_LABEL = "font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)"
_CODE = "font-family:var(--ea-font-code);font-size:12px"


def _swatch(style: str, label: str) -> str:
    return (f'<span style="display:inline-flex;align-items:center;gap:0.25rem;margin-right:0.8rem;{_LABEL}">'
            f'<span style="display:inline-block;width:11px;height:11px;border-radius:2px;{style}"></span>{html.escape(label)}</span>')


def _cells(points: list, q: dict) -> str:
    out = []
    for p in points:
        style = "background:var(--ea-data)" if p == gold(q) else "box-shadow:inset 0 0 0 1px var(--ea-line)"
        out.append(f'<span style="display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:2px;{style}"></span>')
    return f'<span style="white-space:nowrap">{"".join(out)}</span>'


_METRIC_WORDS = {
    "en": {"title": "What your {name} gives each question a point of the course answers, with the three points of search by meaning",
           "yours": "yours", "course": "should be {want}", "points": "the three points",
           "mean": "Averaged over the {n} questions, your {name} gives {a} for search by meaning and {b} after the reranker of build-reranking.",
           "right": "the right point", "none": "Your function gave back {got}, not a number."},
    "nl": {"title": "Wat je {name} elke vraag geeft die een punt van de course beantwoordt, met de drie punten van zoeken op betekenis",
           "yours": "van jou", "course": "zou {want} moeten zijn", "points": "de drie punten",
           "mean": "Gemiddeld over de {n} vragen geeft je {name} {a} voor zoeken op betekenis en {b} na de reranker van build-reranking.",
           "right": "het goede punt", "none": "Je functie gaf {got} terug, geen getal."},
}


def _metric_picture(seen, name: str, solution) -> str:
    """For each question a point answers: the three points of search by meaning with the right point marked as
    input data, and the learner's value, as a mistake where it differs from the course's; and the averages for both
    searches."""
    w = _METRIC_WORDS[seen.lang]
    mine = seen.mine
    fn = mine.names.get(name) if mine is not None else None
    if not callable(fn):
        return ""
    title = w["title"].format(name=name)
    rows, totals = [], {"meaning": 0.0, "reranked": 0.0}
    for q in ANSWERABLE:
        for way in ("meaning", "reranked"):
            try:
                got = with_time_limit(lambda: fn(pipeline(q, way), [gold(q)]))
            except Exception:   # the learner's code; the check says what went wrong
                return ""
            if isinstance(got, bool) or not isinstance(got, (int, float)):
                return yours.figure(title, f'<div style="{_LABEL}">{html.escape(w["none"].format(got=repr(got)[:80]))}</div>', alt=title)
            totals[way] += got
            if way == "meaning":
                want = solution(pipeline(q, way), [gold(q)])
                ok = abs(got - want) <= 1e-9
                value = (f'<span style="{_CODE};color:var(--ea-{"ink" if ok else "loss-text"})">{number(got, seen.lang, 2)}</span>'
                         + ("" if ok else f' <span style="{_LABEL}">{html.escape(w["course"].format(want=number(want, seen.lang, 2)))}</span>'))
                rows.append(f'<span style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink);overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{html.escape(q["text"][seen.lang])}</span>'
                            f'{_cells(pipeline(q, way), q)}<span>{value}</span>')
    head = f'<span></span><span style="{_LABEL}">{html.escape(w["points"])}</span><span style="{_LABEL}">{html.escape(w["yours"])}</span>'
    n = len(ANSWERABLE)
    body = (f'<div style="display:grid;grid-template-columns:minmax(0,1fr) auto auto;gap:0.15rem 0.8rem;align-items:center">{head}{"".join(rows)}</div>'
            f'<div style="{_LABEL};margin-top:0.4rem">{html.escape(w["mean"].format(n=n, name=name, a=number(totals["meaning"] / n, seen.lang, 2), b=number(totals["reranked"] / n, seen.lang, 2)))}</div>'
            f'<div style="margin-top:0.3rem">{_swatch("background:var(--ea-data)", w["right"])}</div>')
    return yours.figure(title, body, alt=title)


def recall_picture(seen) -> str:
    return _metric_picture(seen, "context_recall", context_recall)


def precision_picture(seen) -> str:
    return _metric_picture(seen, "context_precision", context_precision)


_ANSWERS_WORDS = {
    "en": {"title": "The {n} answers of the course's RAG: right or wrong, by whether the search found the right point",
           "hit": "the right point among the sources", "miss": "the right point not among them", "right": "right", "wrong": "wrong",
           "answers": "{n} answers"},
    "nl": {"title": "De {n} antwoorden van het RAG-systeem van de course: goed of fout, naar of de zoekfunctie het goede punt vond",
           "hit": "het goede punt bij de bronnen", "miss": "het goede punt niet erbij", "right": "goed", "wrong": "fout",
           "answers": "{n} antwoorden"},
}


def answers_picture(lang: str) -> str:
    """A bar per row: the answers where the search found the right point and those where it did not, each split into
    right answers (ink) and wrong ones (mistakes)."""
    w = _ANSWERS_WORDS[lang]
    f = findings()
    most = max(f["hit_answers"], f["miss_answers"])

    def row(label: str, right: int, total: int) -> str:
        wrong = total - right
        seg = (f'<span style="display:inline-block;height:14px;width:{100 * right / most:.1f}%;background:var(--ea-ink-3)"></span>'
               f'<span style="display:inline-block;height:14px;width:{100 * wrong / most:.1f}%;background:var(--ea-loss)"></span>')
        return (f'<span style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink)">{html.escape(label)}<br>'
                f'<span style="{_LABEL}">{html.escape(w["answers"].format(n=total))}</span></span>'
                f'<span style="display:flex;align-items:center;white-space:nowrap">{seg}</span>'
                f'<span style="{_CODE};color:var(--ea-ink)">{right} {html.escape(w["right"])}, <span style="color:var(--ea-loss-text)">{wrong} {html.escape(w["wrong"])}</span></span>')
    body = (f'<div style="display:grid;grid-template-columns:minmax(0,12rem) minmax(0,1fr) auto;gap:0.5rem 0.8rem;align-items:center">'
            f'{row(w["hit"], f["hit_right"], f["hit_answers"])}{row(w["miss"], f["miss_right"], f["miss_answers"])}</div>'
            f'<div style="margin-top:0.4rem">{_swatch("background:var(--ea-ink-3)", w["right"])}{_swatch("background:var(--ea-loss)", w["wrong"])}</div>')
    title = w["title"].format(n=f["answers"])
    return yours.figure(title, body, alt=title)


_JUDGE_WORDS = {
    "en": {"title": "Qwen3 as a judge: the {n} answers in order of P(yes), lowest on the left",
           "grounded": "do the sources say what the answer says?", "relevant": "does the answer answer the question?",
           "low": "lowest {p}", "high": "highest {p}",
           "note": "A judge that tells them apart puts the {wrong} wrong answers on the left.",
           "right": "a right answer", "wrong": "a wrong answer"},
    "nl": {"title": "Qwen3 als judge: de {n} antwoorden op volgorde van P(yes), de laagste links",
           "grounded": "zeggen de bronnen wat het antwoord zegt?", "relevant": "beantwoordt het antwoord de vraag?",
           "low": "laagste {p}", "high": "hoogste {p}",
           "note": "Een judge die ze uit elkaar houdt, zet de {wrong} foute antwoorden links.",
           "right": "een goed antwoord", "wrong": "een fout antwoord"},
}


def judge_picture(lang: str) -> str:
    """A row of cells per question to the judge: the answers in order of P(yes), lowest first, right answers in ink and
    wrong ones in the color of mistakes, with the lowest and highest P(yes) under the ends."""
    w = _JUDGE_WORDS[lang]
    f = findings()

    def strip(kind: str) -> str:
        rows = sorted(f["rows"], key=lambda r: (r[kind], r["q"], r["s"]))
        cells = "".join(f'<span title="{number(r[kind], lang, 3)}" style="flex:1 1 0;min-width:0;height:22px;'
                        f'background:var({"--ea-ink-3" if r["right"] else "--ea-loss"})"></span>' for r in rows)
        return (f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink);margin:0.6rem 0 0.25rem">{html.escape(w[kind])}</div>'
                f'<div style="display:flex;gap:2px">{cells}</div>'
                f'<div style="display:flex;justify-content:space-between;gap:1rem;margin-top:0.2rem;{_LABEL}">'
                f'<span>{html.escape(w["low"].format(p=number(rows[0][kind], lang, 2)))}</span>'
                f'<span>{html.escape(w["high"].format(p=number(rows[-1][kind], lang, 2)))}</span></div>')
    body = (strip("grounded") + strip("relevant")
            + f'<div style="{_LABEL};margin-top:0.6rem">{html.escape(w["note"].format(wrong=f["wrong"]))}</div>'
            + f'<div style="margin-top:0.3rem">{_swatch("background:var(--ea-ink-3)", w["right"])}{_swatch("background:var(--ea-loss)", w["wrong"])}</div>')
    title = w["title"].format(n=f["answers"])
    return yours.figure(title, body, alt=title)


_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}


def _show_number(v):
    return round(v, 4) if isinstance(v, float) else v


def _zero_division(e) -> bool:
    return isinstance(e, ZeroDivisionError)


EXERCISES = {
    "context_recall": FunctionBox(
        name="context_recall",
        picture=recall_picture,
        function="context_recall",
        start=START_RECALL,
        solution=SOLUTION_RECALL,
        show=_show_number,
        visible=[(found_listed("q01"), relevant_listed("q01"))],
        hidden=[(found_listed(WORKED), relevant_listed(WORKED)), (["a", "b", "c"], ["a", "d"]), (["a", "b"], ["c", "d", "a"]),
                (["a"], ["a"]), (["b", "c", "d", "e"], ["a"])],
        mistakes=[
            Mistake("by-retrieved", wrong=BY_RETRIEVED, state="almost",
                    en="Almost. `{call}` gives `{got}`: you divide by the number of points that came back. Context recall asks what share of the relevant points came back: divide by `len(relevant)`.",
                    nl="Bijna. `{call}` geeft `{got}`: je deelt door het aantal punten dat terugkwam. Context recall vraagt welk deel van de relevante punten terugkwam: deel door `len(relevant)`."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`context_recall` counts every relevant point, also the ones that did not come back, so it always gives `1.0`. Count a point only when it is in `retrieved`.",
                      "nl": "`context_recall` telt elk relevant punt, ook de punten die niet terugkwamen, dus hij geeft altijd `1.0`. Tel een punt alleen als het in `retrieved` staat."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: the share of the points in `relevant` that are in `retrieved`.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: het deel van de punten in `relevant` dat in `retrieved` staat."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work when there are two relevant points, or when more points came back than are relevant?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie als er twee relevante punten zijn, of als er meer punten terugkwamen dan er relevant zijn?"},
            "no_function": {"en": "Your box does not make a function `context_recall`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `context_recall`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `context_recall` is right in all {n} cases: the share of the relevant points that came back.",
                        "nl": "Je `context_recall` klopt in alle {n} gevallen: het deel van de relevante punten dat terugkwam."},
        },
    ),
    "context_precision": FunctionBox(
        name="context_precision",
        picture=precision_picture,
        function="context_precision",
        start=START_PRECISION,
        solution=SOLUTION_PRECISION,
        show=_show_number,
        visible=[(found_listed(PRECISION_SHOWN), relevant_listed(PRECISION_SHOWN))],
        hidden=[(found_listed(WORKED), relevant_listed(WORKED)), (found_listed("q01"), relevant_listed("q01")),
                (["a", "b", "c"], ["c", "a"]), (["a", "b", "c", "d"], ["b", "d"]), (["a", "b"], ["a", "b"])],
        mistakes=[
            Mistake("no-hits", error=_zero_division,
                    en="With `{call}`, your function divides by zero: no point of `retrieved` is relevant, so `hits` stays 0. Give `0.0` when `hits` is 0.",
                    nl="Met `{call}` deelt je functie door nul: geen punt van `retrieved` is relevant, dus `hits` blijft 0. Geef `0.0` als `hits` 0 is."),
            Mistake("by-all", wrong=BY_ALL,
                    en="`{call}` gives `{got}`: you divide the sum by the number of points that came back. Divide by the number of relevant points among them, `hits`, so that each relevant point counts once.",
                    nl="`{call}` geeft `{got}`: je deelt de som door het aantal punten dat terugkwam. Deel door het aantal relevante punten daarbij, `hits`, zodat elk relevant punt één keer telt."),
            Mistake("first-only", wrong=FIRST_ONLY, state="almost",
                    en="Almost. `{call}` gives `{got}`: you stop at the first relevant point. With two relevant points, the second counts too: add `hits / (i + 1)` for each, and divide by `hits`.",
                    nl="Bijna. `{call}` geeft `{got}`: je stopt bij het eerste relevante punt. Met twee relevante punten telt het tweede ook mee: tel voor elk `hits / (i + 1)` op, en deel door `hits`."),
            Mistake("rank-only", wrong=RANK_ONLY, state="almost",
                    en="Almost. `{call}` gives `{got}`: you add `1 / (i + 1)`, but the precision at a place is the share of relevant points up to there: `hits / (i + 1)`.",
                    nl="Bijna. `{call}` geeft `{got}`: je telt `1 / (i + 1)` op, maar de precision op een plaats is het deel relevante punten tot daar: `hits / (i + 1)`."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`context_precision` gives the plain share of relevant points among those that came back, whatever their place. At each relevant point, add the precision up to there, `hits / (i + 1)`, and give the average.",
                      "nl": "`context_precision` geeft gewoon het deel relevante punten van wat terugkwam, wat hun plaats ook is. Tel bij elk relevant punt de precision tot daar op, `hits / (i + 1)`, en geef het gemiddelde."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: at each relevant point the share of relevant points up to that place, averaged over the relevant points that came back; `0.0` when none came back.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: bij elk relevant punt het deel relevante punten tot die plaats, gemiddeld over de relevante punten die terugkwamen; `0.0` als er geen terugkwam."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for two relevant points, and when none came back?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie voor twee relevante punten, en als er geen terugkwam?"},
            "no_function": {"en": "Your box does not make a function `context_precision`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `context_precision`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `context_precision` is right in all {n} cases: high when the relevant points come first.",
                        "nl": "Je `context_precision` klopt in alle {n} gevallen: hoog als de relevante punten vooraan komen."},
        },
    ),
}

REQUIRED = ["context_recall", "context_precision", "diagnose", "judge"]

CASES = [
    ("context_recall", START_RECALL, "empty", None),
    ("context_recall", "", "empty", None),
    ("context_recall", SOLUTION_RECALL, "correct", None),
    ("context_recall", '''def context_recall(retrieved, relevant):
    return len([p for p in relevant if p in retrieved]) / len(relevant)
''', "correct", None),
    ("context_recall", BY_RETRIEVED, "almost", "by-retrieved"),
    ("context_recall", SOLUTION_RECALL.replace("found / len(relevant)", "found"), "almost", None),
    ("context_recall", SOLUTION_RECALL.replace("if point in retrieved:", "if point not in retrieved:"), "wrong", None),
    ("context_precision", START_PRECISION, "empty", None),
    ("context_precision", "", "empty", None),
    ("context_precision", SOLUTION_PRECISION, "correct", None),
    ("context_precision", '''def context_precision(retrieved, relevant):
    found = [i for i in range(len(retrieved)) if retrieved[i] in relevant]
    if not found:
        return 0.0
    return sum((k + 1) / (i + 1) for k, i in enumerate(found)) / len(found)
''', "correct", None),
    ("context_precision", NO_HITS, "error", "no-hits"),
    ("context_precision", BY_ALL, "wrong", "by-all"),
    ("context_precision", FIRST_ONLY, "almost", "first-only"),
    ("context_precision", RANK_ONLY, "almost", "rank-only"),
]


# --- the RAG to look into -------------------------------------------------------------------------------------------------

def question_options(lang: str) -> dict:
    return {q["text"][lang]: q["id"] for q in ANSWERABLE}


def learner_metrics(recall_text, precision_text):
    """The learner's context_recall and context_precision, when both are right; else None."""
    for name, text in (("context_recall", recall_text), ("context_precision", precision_text)):
        if text is MISSING or text is None or EXERCISES[name].check(text).state != "correct":
            return None
    return run_code(recall_text).names["context_recall"], run_code(precision_text).names["context_precision"]


_TRY_WORDS = {
    "en": {"title": "The RAG of the course, measured: {question}", "meaning": "search by meaning", "reranked": "after the reranker",
           "recall": "context recall", "precision": "context precision", "answer": "answer {i}", "right": "right", "wrong": "wrong",
           "supported": "supported: True", "unsupported": "supported: False", "judge": "the judge: P(yes) grounded {g}, relevant {r}",
           "lesson": "point of the lesson", "gold": "the right point", "failed": "Your functions stopped: {why}"},
    "nl": {"title": "Het RAG-systeem van de course, gemeten: {question}", "meaning": "zoeken op betekenis", "reranked": "na de reranker",
           "recall": "context recall", "precision": "context precision", "answer": "antwoord {i}", "right": "goed", "wrong": "fout",
           "supported": "supported: True", "unsupported": "supported: False", "judge": "de judge: P(yes) grounded {g}, relevant {r}",
           "lesson": "punt van de les", "gold": "het goede punt", "failed": "Je functies stopten: {why}"},
}


def looked_into(fns, qid: str, lang: str) -> str:
    """For a question: the three points of both searches with the learner's context recall and precision of each,
    the two answers of the course's RAG with the course's check, the numbers check and the judge's P(yes)."""
    w = _TRY_WORDS[lang]
    recall_fn, precision_fn = fns
    q = question(qid)
    blocks = []
    for way in ("meaning", "reranked"):
        pts = pipeline(q, way)
        try:
            r = with_time_limit(lambda: recall_fn(pts, [gold(q)]))
            p = with_time_limit(lambda: precision_fn(pts, [gold(q)]))
            shown = f'{number(r, lang, 2)}, {number(p, lang, 2)}'
        except Exception as e:   # the learner's code
            return f'<p><em>{html.escape(w["failed"].format(why=f"{type(e).__name__}: {e}"))}</em></p>'
        items = "".join(f'<li style="{"border-left:3px solid var(--ea-data);padding-left:0.3rem" if t == gold(q) else "padding-left:calc(0.3rem + 3px)"};list-style:none;margin:0.1rem 0">'
                        f'<em>{html.escape(title_of(t, lang))}</em>: {html.escape(t[:140])}{"…" if len(t) > 140 else ""}</li>' for t in pts)
        blocks.append(f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink);margin-top:0.3rem">'
                      f'<span style="font-weight:600">{html.escape(w[way])}</span> '
                      f'<span style="{_CODE};color:var(--ea-ink)">{html.escape(w["recall"])}, {html.escape(w["precision"])}: {shown}</span>'
                      f'<ol style="margin:0.2rem 0;padding:0">{items}</ol></div>')
    for s, a in enumerate(answers(qid)):
        ok, sup = summaries.right(a, q), summaries.supported(a, found(q))
        blocks.append(f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink);margin-top:0.3rem;overflow-wrap:anywhere">'
                      f'<span style="font-weight:600">{html.escape(w["answer"].format(i=s + 1))}</span>: {html.escape(a)}<br>'
                      f'<span style="color:var(--ea-{"ink-2" if ok else "loss-text"})">{html.escape(w["right" if ok else "wrong"])}</span>'
                      f'<span style="{_LABEL}"> · {html.escape(w["supported" if sup else "unsupported"])} · '
                      f'{html.escape(w["judge"].format(g=number(p_yes(qid, s, "grounded"), lang, 3), r=number(p_yes(qid, s, "relevant"), lang, 3)))}</span></div>')
    body = "".join(blocks) + f'<div style="margin-top:0.3rem">{_swatch("background:var(--ea-data)", w["gold"])}</div>'
    title = w["title"].format(question=q["text"][lang])
    return yours.figure(title, body, alt=title)
