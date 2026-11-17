# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-rag-idea: a function that puts numbered sources in front of a question, and one that sends that
to the server, against a stand-in; the course's recorded runs (data/course-rag-qwen.json): eight questions about
the course, asked without a source, with the point of the course's summaries that answers them, and with the point
of another question; their picture, and the demo."""
import functools
import html
import json

from everything_ai import apicheck, chat, paths, serve, summaries, yours
from everything_ai.apicheck import URL, ApiBox, Case
from everything_ai.check import FunctionBox, Mistake, run_code

DOWNLOADS = {"Qwen3-0.6B": chat.FILES}   # what the demo downloads, for the line under the header

# the questions, by their id in data/course-summaries.json: the first two ask what was known outside the course
# before Qwen3 was trained, the other six what only the course measured
QUESTIONS = ["q03", "q17", "q01", "q02", "q04", "q09", "q11", "q13"]
PUBLIC = QUESTIONS[:2]
WAYS = ("none", "right", "wrong")
SEEDS = 4
INSTRUCTION = "Use these sources to answer the question."

SOLUTION_SOURCES = '''def with_sources(question, passages):
    if len(passages) == 0:
        return question
    text = "Use these sources to answer the question.\\n\\n"
    n = 1
    for passage in passages:
        text = text + f"[{n}] {passage}\\n"
        n = n + 1
    return text + "\\nQuestion: " + question
'''
with_sources = run_code(SOLUTION_SOURCES).names["with_sources"]


def question(qid: str) -> dict:
    return next(q for q in summaries.questions() if q["id"] == qid)


def passages_for(qid: str, way: str) -> list:
    """The passages a question is asked with: none, the point that answers it, or the point of the next question."""
    if way == "none":
        return []
    other = QUESTIONS[(QUESTIONS.index(qid) + 1) % len(QUESTIONS)]
    return [summaries.text_of(question(qid if way == "right" else other)["gold"])]


RECORD = {f"{qid}-{way}": {"messages": [{"role": "user", "content": with_sources(question(qid)["text"]["en"], passages_for(qid, way))}],
                           "max_tokens": 80, "seeds": SEEDS}
          for qid in QUESTIONS for way in WAYS}
DATA = paths.ROOT / "data" / "course-rag-qwen.json"


@functools.cache
def course() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-rag-idea"]


def answers(qid: str, way: str) -> list:
    return course()[f"{qid}-{way}"]["answers"]


def right(qid: str, text: str) -> bool:
    return summaries.right(text, question(qid))


@functools.cache
def findings() -> dict:
    """Per way of asking: how many answers were right, for the two public questions and for the six about the course;
    and how many answers to the course's questions said they could not answer."""
    out = {}
    for way in WAYS:
        public = sum(right(q, a["text"]) for q in PUBLIC for a in answers(q, way))
        own = sum(right(q, a["text"]) for q in QUESTIONS[2:] for a in answers(q, way))
        unknown = sum(summaries.says_unknown(a["text"]) for q in QUESTIONS[2:] for a in answers(q, way))
        out[way] = {"public": public, "course": own, "right": public + own, "unknown": unknown,
                    "runs": SEEDS * len(QUESTIONS), "course_runs": SEEDS * (len(QUESTIONS) - 2), "public_runs": SEEDS * 2}
    return out


def point_count() -> int:
    """How many points the course's summaries have: the texts build-rag-first searches."""
    return len(summaries.points("en"))


_LABEL = "font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)"

_WAYS_WORDS = {
    "en": {"title": "Is Qwen3's answer right? Four runs per question and way of asking",
           "ways": {"none": "no source", "right": "the right source", "wrong": "another question's source"},
           "right": "right answer", "not": "not right", "count": "{k} of {n}", "public": "known before Qwen3 was trained"},
    "nl": {"title": "Is het antwoord van Qwen3 goed? Vier runs per vraag en manier van vragen",
           "ways": {"none": "geen bron", "right": "de goede bron", "wrong": "de bron van een andere vraag"},
           "right": "goed antwoord", "not": "niet goed", "count": "{k} van de {n}", "public": "bekend vóór Qwen3 getraind werd"},
}


def _square(ok: bool, text: str) -> str:
    return (f'<span title="{html.escape(" ".join(text.split())[:160])}" style="display:inline-block;width:12px;height:12px;'
            f'margin-right:2px;border-radius:2px;background:var(--ea-{"pred" if ok else "loss"})"></span>')


def ways_picture(lang: str) -> str:
    """A row per question and a column per way of asking: a square per run, in the color of model output when the
    answer is right and as a mistake when it is not, with the answer on hovering; under each column the count."""
    w = _WAYS_WORDS[lang]
    f = findings()
    head = (f'<span></span>' + "".join(f'<span style="{_LABEL}">{html.escape(w["ways"][way])}</span>' for way in WAYS))
    rows = []
    for qid in QUESTIONS:
        q = question(qid)["text"][lang]
        mark = f' <span style="{_LABEL}">({html.escape(w["public"])})</span>' if qid in PUBLIC else ""
        cells = "".join(f'<span>{"".join(_square(right(qid, a["text"]), a["text"]) for a in answers(qid, way))}</span>' for way in WAYS)
        rows.append(f'<span style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink)">{html.escape(q)}{mark}</span>{cells}')
    foot = "<span></span>" + "".join(f'<span style="{_LABEL}">{html.escape(w["count"].format(k=f[way]["right"], n=f[way]["runs"]))}</span>' for way in WAYS)
    grid = (f'<div style="display:grid;grid-template-columns:minmax(0,1fr) repeat(3,minmax(4.5rem,auto));gap:0.35rem 0.8rem;align-items:center">'
            f'{head}{"".join(rows)}{foot}</div>')
    legend = "".join(f'<span style="display:inline-flex;align-items:center;gap:0.25rem;margin-right:0.8rem;{_LABEL}">'
                     f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:var(--ea-{c})"></span>{html.escape(w[k])}</span>'
                     for k, c in (("right", "pred"), ("not", "loss")))
    return yours.figure(w["title"], grid + f'<div style="margin-top:0.4rem">{legend}</div>', alt=w["title"])


# --- exercise 1: with_sources ---------------------------------------------------------------------------------------------

START_SOURCES = '''def with_sources(question, passages):
    text = "Use these sources to answer the question.\\n\\n"
    for passage in passages:
        text = text + passage + "\\n"
    return text + "\\nQuestion: " + question
'''
FROM_ZERO = SOLUTION_SOURCES.replace("    n = 1\n", "    n = 0\n")
NO_EMPTY = SOLUTION_SOURCES.replace("    if len(passages) == 0:\n        return question\n", "")
EXAMPLE_QUESTION = question("q17")["text"]["en"]
EXAMPLE_PASSAGES = [summaries.text_of("res-share-work#2"), summaries.text_of(question("q11")["gold"])]

_SOURCES_WORDS = {
    "en": {"title": "The prompt your with_sources makes of the question and two sources", "none": "Your function gave back {got}, not a text."},
    "nl": {"title": "De prompt die je with_sources maakt van de vraag en twee bronnen", "none": "Je functie gaf {got} terug, geen tekst."},
}


def sources_picture(seen) -> str:
    """The prompt the learner's with_sources makes of the example, as it goes to the model: the instruction and the
    question in ink, each source as input data."""
    w = _SOURCES_WORDS[seen.lang]
    mine = seen.mine
    fn = mine.names.get("with_sources") if mine is not None else None
    if not callable(fn):
        return ""
    try:
        got = fn(EXAMPLE_QUESTION, list(EXAMPLE_PASSAGES))
    except Exception:   # the learner's code; the check says what went wrong
        return ""
    if not isinstance(got, str):
        return yours.figure(w["title"], f'<div style="{_LABEL}">{html.escape(w["none"].format(got=repr(got)[:80]))}</div>', alt=w["title"])
    lines = []
    for line in got.split("\n"):
        source = any(p in line for p in EXAMPLE_PASSAGES)
        style = "background:var(--ea-data-wash);border-left:3px solid var(--ea-data);padding-left:0.3rem" if source else ""
        lines.append(f'<div style="{style}">{html.escape(line) or "&nbsp;"}</div>')
    body = (f'<div style="font-family:var(--ea-font-code);font-size:12px;color:var(--ea-ink);white-space:pre-wrap;background:var(--ea-inset);'
            f'border:1px solid var(--ea-line);border-radius:3px;padding:0.3rem 0.5rem">{"".join(lines)}</div>')
    return yours.figure(w["title"], body, alt=w["title"])


# --- exercise 2: ask ------------------------------------------------------------------------------------------------------

START_ASK = '''import json
import urllib.request


def ask(url, question, passages):
    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": question}], "max_tokens": 80}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        answer = json.loads(response.read())
    return answer["choices"][0]["message"]["content"]
'''
SOLUTION_ASK = START_ASK.replace('''    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": question}], "max_tokens": 80}
''', '''    prompt = with_sources(question, passages)
    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": prompt}], "max_tokens": 80}
''')
ASK_QUESTIONS = [(EXAMPLE_QUESTION, EXAMPLE_PASSAGES[:1]), (question("q13")["text"]["en"], []),
                 (question("q01")["text"]["en"], [summaries.text_of(question("q01")["gold"]), EXAMPLE_PASSAGES[1]])]


def _from_sources(q: str, passages: list) -> str:
    return "The stand-in's answer from {n} sources.".format(n=len(passages)) if passages else "The stand-in's answer without sources."


def _ask_stand_in():
    answers = {}
    for q, passages in ASK_QUESTIONS:
        answers[with_sources(q, passages)] = _from_sources(q, passages)
        answers.setdefault(q, _from_sources(q, []))
    return serve.StandIn(answers, default="The stand-in got a prompt it did not expect.")


def _judge_ask(q: str, passages: list):
    def judge(got, running):
        if got == _from_sources(q, passages):
            return None
        if passages and got == _from_sources(q, []):
            return "question-only"
        if isinstance(got, dict):
            return "whole"
        return "other"
    return judge


_SHOWN = {"en": "[{n} passages]", "nl": "[{n} passages]"}
ASK_CASES = [Case(_ask_stand_in, (URL, q, passages), _judge_ask(q, passages),
                  shown_args=("url", repr(q), {lang: t.format(n=len(passages)) if passages else "[]" for lang, t in _SHOWN.items()}))
             for q, passages in ASK_QUESTIONS]


def ask_picture(seen) -> str:
    words = {"en": {"title": "What your ask sent, and what came back", "request": "Your code sent: {method} {path}",
                    "headers": "Headers:", "answer": "The server sent back: status {status}",
                    "stream": "a stream with {n} pieces", "none": "Your function sent nothing to the server yet."},
             "nl": {"title": "Wat je ask stuurde, en wat er terugkwam", "request": "Je code stuurde: {method} {path}",
                    "headers": "Headers:", "answer": "De server stuurde terug: status {status}",
                    "stream": "een stream met {n} stukjes", "none": "Je functie stuurde nog niets naar de server."}}[seen.lang]
    return apicheck.exchange(seen, "ask", ASK_CASES[0], words)


def _not_bytes(e) -> bool:
    return isinstance(e, TypeError) and "bytes" in str(e)


_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}

EXERCISES = {
    "with_sources": FunctionBox(
        name="with_sources",
        picture=sources_picture,
        function="with_sources",
        save_as=("app", "sources"),   # Project: your assistant reads your documents builds on it
        start=START_SOURCES,
        solution=SOLUTION_SOURCES,
        visible=[(EXAMPLE_QUESTION, EXAMPLE_PASSAGES)],
        hidden=[(EXAMPLE_QUESTION, []), ("Is the shop open on Monday?", ["The shop is closed on Sunday and Monday."]),
                ("Which one?", ["a", "b", "c"])],
        mistakes=[
            Mistake("from-zero", wrong=FROM_ZERO,
                    en="`{call}` gives `{got}`: the first source is `[0]`. People count from 1, and so does the model when it says *source [1]*: start `n` at 1.",
                    nl="`{call}` geeft `{got}`: de eerste bron is `[0]`. Mensen tellen vanaf 1, en het model ook als het *bron [1]* zegt: begin `n` bij 1."),
            Mistake("no-empty", state="almost", wrong=NO_EMPTY,
                    en="Right with sources. But `{call}` gives `{got}`: an instruction to use sources that are not there. Without passages, give back the question alone.",
                    nl="Goed met bronnen. Maar `{call}` geeft `{got}`: een opdracht om bronnen te gebruiken die er niet zijn. Geef zonder passages alleen de vraag terug."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`with_sources` puts the sources in front of the question, but without numbers, and also when there are none. Number them `[1]`, `[2]` and so on, and give back the question alone when `passages` is empty.",
                      "nl": "`with_sources` zet de bronnen vóór de vraag, maar zonder nummers, en ook als er geen zijn. Nummer ze `[1]`, `[2]` enzovoort, en geef alleen de vraag terug als `passages` leeg is."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: the instruction and a blank line, then each source on a line of its own after `[1] `, `[2] ` and so on, then a blank line and `Question: ` with the question.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: de opdracht en een lege regel, dan elke bron op een eigen regel na `[1] `, `[2] ` enzovoort, dan een lege regel en `Question: ` met de vraag."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work with one source, with three, and with none?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie met één bron, met drie, en zonder?"},
            "no_function": {"en": "Your box does not make a function `with_sources`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `with_sources`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `with_sources` is right in all {n} cases: numbered sources in front of the question, and the question alone when there are none. The check kept it as a file, `{saved}`.",
                        "nl": "Je `with_sources` klopt in alle {n} gevallen: genummerde bronnen vóór de vraag, en alleen de vraag als er geen zijn. De check heeft hem als bestand bewaard, `{saved}`."},
            "kept": {"en": "Your `with_sources` is right. You changed `{saved}` yourself, so the check left it as it is.",
                     "nl": "Je `with_sources` is goed. Je hebt `{saved}` zelf veranderd, dus de check liet het staan zoals het is."},
        },
    ),
    "ask": ApiBox(
        name="ask",
        picture=ask_picture,
        function="ask",
        start=START_ASK,
        solution=SOLUTION_ASK,
        names={"with_sources": with_sources},
        cases=ASK_CASES,
        mistakes=[
            Mistake("data-text", error=_not_bytes,
                    en="urllib sends bytes, not text. Keep `.encode()` after `json.dumps(body)`, as the box had it.",
                    nl="urllib verstuurt bytes, geen tekst. Laat `.encode()` achter `json.dumps(body)` staan, zoals het in het vak stond."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`ask` sends the question alone, without the sources. Send `with_sources(question, passages)` as the content of the user's message.",
                      "nl": "`ask` stuurt alleen de vraag, zonder de bronnen. Stuur `with_sources(question, passages)` als de content van het bericht van de gebruiker."},
            "question-only": {"en": "With `{call}`, the server got the question without the sources. Send `with_sources(question, passages)` as the content of the user's message.",
                              "nl": "Met `{call}` kreeg de server de vraag zonder de bronnen. Stuur `with_sources(question, passages)` als de content van het bericht van de gebruiker."},
            "whole": {"en": "With `{call}`, your function gives back the whole answer of the server. Give back only the text: `answer[\"choices\"][0][\"message\"][\"content\"]`.",
                      "nl": "Met `{call}` geeft je functie het hele antwoord van de server terug. Geef alleen de tekst terug: `answer[\"choices\"][0][\"message\"][\"content\"]`."},
            "other": {"en": "With `{call}`, your function gives back `{got}`. Send `with_sources(question, passages)` as the content of the user's message, and give back the text of the answer.",
                      "nl": "Met `{call}` geeft je functie `{got}` terug. Stuur `with_sources(question, passages)` als de content van het bericht van de gebruiker, en geef de tekst van het antwoord terug."},
            "correct": {"en": "Your `ask` sends the question with its sources, and gives back the model's answer.",
                        "nl": "Je `ask` stuurt de vraag met de bronnen, en geeft het antwoord van het model terug."},
        },
    ),
}

REQUIRED = ["with_sources", "ask", "needs"]

CASES = [
    ("with_sources", START_SOURCES, "empty", None),
    ("with_sources", "", "empty", None),
    ("with_sources", SOLUTION_SOURCES, "correct", None),
    ("with_sources", "def with_sources(question, passages):\n    if not passages:\n        return question\n"
                     "    text = \"Use these sources to answer the question.\\n\\n\"\n    for i in range(len(passages)):\n"
                     "        text = text + \"[\" + str(i + 1) + \"] \" + passages[i] + \"\\n\"\n"
                     "    return text + \"\\nQuestion: \" + question\n", "correct", None),
    ("with_sources", FROM_ZERO, "wrong", "from-zero"),
    ("with_sources", NO_EMPTY, "almost", "no-empty"),
    ("with_sources", SOLUTION_SOURCES.replace('f"[{n}] {passage}\\n"', 'f"[{n}]{passage}\\n"'), "wrong", None),
    ("ask", START_ASK, "empty", None),
    ("ask", SOLUTION_ASK, "correct", None),
    ("ask", START_ASK.replace('"max_tokens": 80}', '"max_tokens": 60}'), "wrong", "question-only"),
    ("ask", SOLUTION_ASK.replace('    return answer["choices"][0]["message"]["content"]\n', "    return answer\n"), "wrong", "whole"),
    ("ask", SOLUTION_ASK.replace('    return answer["choices"][0]["message"]["content"]\n', "    return prompt\n"), "wrong", "other"),
    ("ask", SOLUTION_ASK.replace("json.dumps(body).encode()", "json.dumps(body)"), "error", "data-text"),
]


# --- the demo: your ask, against the course's own server ------------------------------------------------------------------

def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


def learner_ask(sources_text, ask_text):
    """The learner's ask from exercise 2 with their own with_sources from exercise 1, when both are right; else None."""
    from everything_ai.check import MISSING
    for name, text in (("with_sources", sources_text), ("ask", ask_text)):
        if text is MISSING or text is None or EXERCISES[name].check(text).state != "correct":
            return None
    mine = run_code(sources_text).names["with_sources"]
    return run_code(ask_text, {"with_sources": mine}).names["ask"]


def demo(ask, qid: str, way: str, words: dict, lang: str) -> str:
    """The learner's ask on one of the eight questions, with no source, the right one or another question's, through
    the course's server with Qwen3 behind it: the sources, the answer, and whether it is right."""
    import time
    running = serve.model_server()
    q = question(qid)
    passages = passages_for(qid, way)
    start = time.perf_counter()
    try:
        text = ask(running.url, q["text"]["en"], passages)
    except Exception as e:   # the learner's code, against the real server; it may have run out of memory
        said = words["memory"] if serve.out_of_memory(e) else words["failed"].format(why=f"{type(e).__name__}: {e}")
        return f"<p><em>{html.escape(said)}</em></p>"
    seconds = time.perf_counter() - start
    ok = isinstance(text, str) and right(qid, text)
    code = "font-family:var(--ea-font-code);font-size:13px;white-space:pre-wrap"
    sources = "".join(f'<div style="{code};background:var(--ea-data-wash);border-left:3px solid var(--ea-data);padding:0.1rem 0.4rem;margin:2px 0">[{i}] {html.escape(p)}</div>'
                      for i, p in enumerate(passages, start=1)) or f'<div style="{_LABEL}">{html.escape(words["no_sources"])}</div>'
    body = (f'<div style="{_LABEL};margin-top:0.2rem">{html.escape(words["sources"])}</div>{sources}'
            f'<div style="{_LABEL};margin-top:0.4rem">{html.escape(words["answer"])}</div>'
            f'<div style="{code};color:var(--ea-pred-text)">{html.escape(str(text).strip())}</div>'
            f'<div style="font-family:var(--ea-font-ui);font-size:13px;margin-top:0.3rem;color:{"var(--ea-ink)" if ok else "var(--ea-loss-text)"}">'
            f'{html.escape((words["right"] if ok else words["not"]).format(seconds=number(seconds, lang, 0)))}</div>')
    title = words["title"].format(question=q["text"][lang])
    return yours.figure(title, body, alt=title)


def number(x: float, lang: str, digits: int = 1) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text
