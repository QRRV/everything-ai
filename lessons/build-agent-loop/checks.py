# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-agent-loop: the agent loop through the API, with ReAct (a thought before each tool call, each result
read as an observation) and a step limit. The course's recorded runs of Qwen3-0.6B (data/course-agents-qwen.json): the
twelve questions of build-workflow-or-agent, asked of the agent with a system prompt that asks for a thought before each
call, beside the runs without it. What they show, their pictures, the exercises (the whole loop against a stand-in of
the server that plays an agent's run, and a trace of thought, action and observation) and the demo on Qwen3."""
import copy
import html
import importlib.util
import json
import re

from everything_ai import chat, paths, serve, yours
from everything_ai.apicheck import URL, ApiBox, Case, call
from everything_ai.check import MISSING, FunctionBox, Mistake, run_code

DOWNLOADS = {"Qwen3-0.6B": chat.FILES}   # what the demo downloads, for the line under the header
DATA = paths.ROOT / "data" / "course-agents-qwen.json"


def _lesson_checks(lesson_id: str):
    """The checks of an earlier lesson, loaded once for this module."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_agent_loop",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W = _lesson_checks("build-workflow-or-agent")   # the kitchen, its tools, the twelve questions and the plain agent's runs
TOOLS, FUNCTIONS, QUESTIONS, STEPS, SYSTEM = W.TOOLS, W.FUNCTIONS, W.QUESTIONS, W.STEPS, W.SYSTEM
tool_results, right_answer, calls_of, final_of = W.tool_results, W.right_answer, W.calls_of, W.final_of

REACT = SYSTEM + (" Before each tool call, write one line that starts with Thought: and says what you know so far "
                  "and what you need next.")

RECORD = {}
for _i, (_q, _d) in enumerate(QUESTIONS):
    RECORD[f"react-{_i}"] = {"messages": [{"role": "system", "content": REACT}, {"role": "user", "content": _q}],
                             "tools": TOOLS, "max_tokens": 200, "temperature": 0, "steps": STEPS, "seeds": 1}


# --- what the recorded runs show ----------------------------------------------------------------------------------------

def course() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-agent-loop"]


def react_steps(i: int) -> list:
    return course()[f"react-{i}"]["answers"][0]["steps"]


def thought(step: dict) -> str:
    """The text the model wrote with a step's tool calls: its thought, or "" when it wrote none."""
    return (step.get("text") or "").strip() if step["tool_calls"] else ""


def _stats(runs: list) -> dict:
    n = len(runs)
    return {
        "right": [i for i in range(n) if right_answer(final_of(runs[i]), QUESTIONS[i][1])],
        "looked": [i for i in range(n) if any(name == "pantry" for name, _ in calls_of(runs[i]))],
        "every": [i for i in range(n) if {a.get("item") for name, a in calls_of(runs[i]) if name == "pantry" and isinstance(a, dict)}
                  >= set(W.recipe(QUESTIONS[i][1]))],
        "unfinished": [i for i in range(n) if final_of(runs[i]) is None],
        "requests": sum(len(r) for r in runs),
        "prompt": sum(s["prompt_tokens"] for r in runs for s in r),
        "tokens": sum(s["tokens"] for r in runs for s in r),
        "seconds": sum(s["seconds"] for r in runs for s in r),
        "calls": sum(len(calls_of(r)) for r in runs),
        "thoughts": sum(1 for r in runs for s in r if thought(s)),
        "call_steps": sum(1 for r in runs for s in r if s["tool_calls"]),
        "no_calls": [i for i in range(n) if not calls_of(runs[i])],
    }


def findings() -> dict:
    n = len(QUESTIONS)
    plain = _stats([W.agent_steps(i) for i in range(n)])
    react = _stats([react_steps(i) for i in range(n)])
    return {"questions": n, "steps": STEPS, "plain": plain, "react": react}


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


QUOTED = 2   # a run whose answer says what is in the pantry, without having looked in it


def quoted() -> str:
    """The sentence of run QUOTED's answer that says what is in the pantry."""
    text = final_of(react_steps(QUOTED)) or ""
    return next((p.strip() for p in re.split(r"(?<=\.)\s+", text) if "in the pantry" in p), "")


def _many(k: int, one: str, many: str) -> str:
    return f"{k} {one if k == 1 else many}"


def measured_text(lang: str) -> str:
    """What the runs show, in the lesson's language, with the measured numbers."""
    f = findings()
    n, p, r = f["questions"], f["plain"], f["react"]
    q = QUESTIONS[QUOTED][0]
    if lang == "nl":
        every = ("maar nooit voor elk ingrediënt van het recept" if not p["every"] and not r["every"] else
                 f"en voor elk ingrediënt van het recept bij {len(r['every'])} tegen {len(p['every'])}")
        limit = ("Geen run haalde de stappenlimiet." if not p["unfinished"] and not r["unfinished"] else
                 f"De stappenlimiet stopte {len(p['unfinished'])} en {len(r['unfinished'])} runs.")
        return (f"Zonder de zin klopte het antwoord van de agent bij {len(p['right'])} van de {n} vragen, met de zin bij {len(r['right'])}. "
                f"Met de zin schreef het model bij {r['thoughts']} van zijn {r['call_steps']} antwoorden met tool calls een gedachte, "
                f"en keek het bij {_many(len(r['looked']), 'vraag', 'vragen')} in de voorraadkast in plaats van bij {len(p['looked'])}, "
                f"{every}. Bij {_many(len(r['no_calls']), 'vraag', 'vragen')} riep het geen enkele tool aan: het antwoordde met een "
                f"regel die met *Thought:* begon en zei dat het het recept nodig had, en omdat dat antwoord geen tool calls had, nam de "
                f"loop het als antwoord. {limit} Wat het schreef, klopte ook niet altijd: bij *{q}* begon zijn antwoord met "
                f"*Thought:* en zei het, zonder in de voorraadkast te hebben gekeken: *{quoted()}* Alles samen: {p['requests']} tegen {r['requests']} requests, {number(p['prompt'], 'nl')} tegen "
                f"{number(r['prompt'], 'nl')} tokens aan prompts, {number(p['tokens'], 'nl')} tegen {number(r['tokens'], 'nl')} "
                f"geschreven tokens, {number(p['seconds'], 'nl', 1)} tegen {number(r['seconds'], 'nl', 1)} seconden.")
    every = ("but never for every ingredient of the recipe" if not p["every"] and not r["every"] else
             f"and for every ingredient of the recipe for {len(r['every'])} against {len(p['every'])}")
    limit = ("No run reached the step limit." if not p["unfinished"] and not r["unfinished"] else
             f"The step limit stopped {len(p['unfinished'])} and {len(r['unfinished'])} runs.")
    return (f"Without the sentence, the agent's answer was right for {len(p['right'])} of the {n} questions; with it, for {len(r['right'])}. "
            f"With the sentence, the model wrote a thought in {r['thoughts']} of its {r['call_steps']} answers with tool calls, and "
            f"looked in the pantry for {_many(len(r['looked']), 'question', 'questions')} instead of {len(p['looked'])}, {every}. "
            f"For {_many(len(r['no_calls']), 'question', 'questions')}, it called no tool at all: it answered with a line that "
            f"began with *Thought:* and said it needed the recipe, and since that answer had no tool calls, the loop took it as "
            f"the answer. {limit} What it wrote was not always true either: for *{q}*, its answer began with *Thought:* and said, "
            f"without having looked in the pantry: *{quoted()}* In all: {p['requests']} against "
            f"{r['requests']} requests, {number(p['prompt'], 'en')} against {number(r['prompt'], 'en')} prompt tokens, "
            f"{number(p['tokens'], 'en')} against {number(r['tokens'], 'en')} tokens written, {number(p['seconds'], 'en', 1)} "
            f"against {number(r['seconds'], 'en', 1)} seconds.")


WORKED = 5   # the question of the worked example: the one run with a right answer


def _listed(items: list, last: str) -> str:
    """Names in code, joined as a list: `a`, `b` and `c`, with the word before the last one."""
    names = [f"`{x}`" for x in items]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" {last} " + names[-1]


def worked_text(lang: str) -> str:
    """The worked example: one recorded run with the ReAct prompt, told step by step."""
    steps = react_steps(WORKED)
    q, dish = QUESTIONS[WORKED]
    calls = sum(len(s["tool_calls"]) for s in steps)
    prompt = sum(s["prompt_tokens"] for s in steps)
    looked = [a.get("item") for name, a in calls_of(steps) if name == "pantry" and isinstance(a, dict)]
    rest = [item for item in W.recipe(dish) if item not in looked]
    only = findings()["react"]["right"] == [WORKED]   # the only run with a right answer
    if lang == "nl":
        note = (f" Het is de enige run met een goed antwoord, en toch keek de agent alleen voor {_listed(looked, 'en')} in de "
                f"voorraadkast: {_listed(rest, 'en')} controleerde hij nooit." if only and looked and rest else "")
        return (f"### Uitgewerkt voorbeeld: één run, regel voor regel\n\n"
                f"Hieronder staat de run van de course op vraag {WORKED + 1}, *{q}*, met de zin over een gedachte: "
                f"{len(steps)} requests en {_many(calls, 'tool call', 'tool calls')}, {number(prompt, 'nl')} tokens aan prompts "
                f"samen, want elk request stuurt het hele gesprek opnieuw. Elke regel met *Action* is een call die de loop "
                f"uitvoerde, elke regel met *Observation* wat de tool teruggaf; de regels daartussen schreef het model.{note}")
    note = (f" It is the only run with a right answer, yet the agent looked in the pantry only for {_listed(looked, 'and')}: "
            f"it never checked {_listed(rest, 'or')}." if only and looked and rest else "")
    return (f"### Worked example: one run, line by line\n\n"
            f"Below is the course's run of question {WORKED + 1}, *{q}*, with the sentence about a thought: {len(steps)} requests "
            f"and {_many(calls, 'tool call', 'tool calls')}, {number(prompt, 'en')} prompt tokens together, since each request "
            f"sends the whole conversation again. Each line with *Action* is a call the loop ran, each line with *Observation* "
            f"what the tool gave back; the model wrote the lines in between.{note}")


# --- a run as thought, action and observation --------------------------------------------------------------------------

def trace_of(messages: list) -> list:
    """The course's trace of a run: the lines of exercise 2."""
    lines = []
    for m in messages:
        if m["role"] == "tool":
            lines.append("Observation: " + m["content"])
        elif m["role"] == "assistant":
            if m.get("tool_calls"):
                if m.get("content"):
                    lines.append(m["content"])
                for c in m["tool_calls"]:
                    lines.append("Action: " + c["function"]["name"] + "(" + c["function"]["arguments"] + ")")
            else:
                lines.append("Answer: " + m["content"])
    return lines


def messages_of(steps: list, system: str, question: str) -> list:
    """The messages of a recorded run, as the loop collected them: the system prompt, the question, and each answer
    with its calls, followed by their results with the ids of the calls."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    for s in steps:
        message = {"role": "assistant", "content": s["text"] or None}
        if s["tool_calls"]:
            message["tool_calls"] = s["tool_calls"]
        messages.append(message)
        for c, r in zip(s["tool_calls"], s["results"]):
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": r})
    return messages


_ROLE_COLOR = {"Observation": "var(--ea-data-text)", "Action": "var(--ea-pred-text)", "Answer": "var(--ea-pred-text)"}
_UI = "font-family:var(--ea-font-ui)"
_CELL = "padding:0.18rem 0.5rem;border-bottom:1px solid var(--ea-line);vertical-align:top;font-size:13px"
_HEAD = f"{_CELL};color:var(--ea-ink-2);font-weight:500;text-align:left;font-family:var(--ea-font-ui)"


def _code(text: str, color: str = "var(--ea-ink)") -> str:
    return (f'<span style="font-family:var(--ea-font-code);font-size:12px;color:{color};white-space:pre-wrap;'
            f'overflow-wrap:anywhere">{html.escape(text)}</span>')


def trace_html(lines: list, limit: int = 40) -> str:
    """Lines of a trace, each in the color of its role: what the model writes, and what the tools give back."""
    out = []
    for line in lines[:limit]:
        head = line.split(":", 1)[0] if isinstance(line, str) else ""
        color = _ROLE_COLOR.get(head, "var(--ea-pred-text)" if head == "Thought" else "var(--ea-ink)")
        out.append(f'<div style="margin:0.1rem 0">{_code(str(line), color)}</div>')
    if len(lines) > limit:
        out.append(f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2)">…</div>')
    return "".join(out)


_RUN_WORDS = {
    "en": {"title": "Question {i}, with a thought before each call", "alt": "The trace of one run: each thought, each tool call, each result, and the answer."},
    "nl": {"title": "Vraag {i}, met een gedachte voor elke call", "alt": "De trace van één run: elke gedachte, elke tool call, elk resultaat, en het antwoord."},
}


def run_picture(i: int, lang: str) -> str:
    """The trace of the recorded ReAct run of question i."""
    w = _RUN_WORDS[lang]
    q, _ = QUESTIONS[i]
    lines = trace_of(messages_of(react_steps(i), REACT, q))
    if final_of(react_steps(i)) is None:
        lines.append({"en": "(the step limit stopped the run here)", "nl": "(de stappenlimiet stopte de run hier)"}[lang])
    return yours.figure(w["title"].format(i=i + 1), f'<p style="{_UI};font-size:13px;margin:0 0 0.4rem">{html.escape(q)}</p>' + trace_html(lines, 60),
                        alt=w["alt"])


_TABLE_WORDS = {
    "en": {"title": "Twelve questions to the agent, without and with a thought before each call", "question": "the question",
           "plain": "without", "react": "with a thought", "right": "right", "wrong": "not right", "stopped": "stopped at the limit",
           "total": "{k} of {n} right", "steps": "{k} requests"},
    "nl": {"title": "Twaalf vragen aan de agent, zonder en met een gedachte voor elke call", "question": "de vraag",
           "plain": "zonder", "react": "met een gedachte", "right": "goed", "wrong": "niet goed", "stopped": "gestopt bij de limiet",
           "total": "{k} van de {n} goed", "steps": "{k} requests"},
}


def _cell(steps: list, ok: bool, w: dict) -> str:
    stopped = final_of(steps) is None
    word = w["right"] if ok else (w["stopped"] if stopped else w["wrong"])
    color = "var(--ea-ink-2)" if ok else "var(--ea-loss-text)"
    return (f'<span style="{_UI};font-size:12px;color:{color};font-weight:{400 if ok else 600}">{word}</span>'
            f'<span style="{_UI};font-size:12px;color:var(--ea-ink-2)">, {w["steps"].format(k=len(steps))}</span>')


def table_picture(lang: str) -> str:
    """Each question with the plain agent's outcome and the ReAct agent's."""
    w, f = _TABLE_WORDS[lang], findings()
    rows = []
    for i, (q, _) in enumerate(QUESTIONS):
        rows.append(f'<tr><td style="{_CELL};{_UI}">{html.escape(q)}</td>'
                    f'<td style="{_CELL}">{_cell(W.agent_steps(i), i in f["plain"]["right"], w)}</td>'
                    f'<td style="{_CELL}">{_cell(react_steps(i), i in f["react"]["right"], w)}</td></tr>')
    n = f["questions"]
    foot = (f'<tr><td style="{_CELL}"></td>'
            + "".join(f'<td style="{_CELL};{_UI};font-weight:600">{w["total"].format(k=len(f[k]["right"]), n=n)}</td>' for k in ("plain", "react"))
            + "</tr>")
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><thead><tr>'
             + "".join(f'<th style="{_HEAD}">{h}</th>' for h in (w["question"], w["plain"], w["react"]))
             + f'</tr></thead><tbody>{"".join(rows)}{foot}</tbody></table></div>')
    return yours.figure(w["title"], table, alt=w["title"])


# --- exercise 1: the agent loop through the API, against a stand-in that plays a run ------------------------------

START_RUN = '''import json
import urllib.request


def ask(url, messages):
    body = {"model": "qwen3-0.6b", "messages": messages, "tools": TOOLS,
            "max_tokens": 200, "temperature": 0}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        answer = json.loads(response.read())
    return answer["choices"][0]["message"]


def run_agent(url, task, max_steps):
    messages = [{"role": "system", "content": REACT}, {"role": "user", "content": task}]
    for step in range(max_steps):
        message = ask(url, messages)
        ...
    return None
'''
SOLUTION_RUN = START_RUN.replace('''        ...
''', '''        messages = messages + [message]
        if not message.get("tool_calls"):
            return message["content"]
        for call in message["tool_calls"]:
            arguments = json.loads(call["function"]["arguments"])
            result = FUNCTIONS[call["function"]["name"]](arguments)
            messages = messages + [{"role": "tool", "tool_call_id": call["id"], "content": result}]
''')
AGENT = {"TOOLS": TOOLS, "FUNCTIONS": FUNCTIONS, "REACT": REACT}

Q_PANCAKES = QUESTIONS[0][0]
Q_HELLO = "Hello! What can you do?"
A_PANCAKES = "You have to buy 250 grams of milk and 110 grams of eggs."
A_HELLO = "I can tell you what to buy for a dish, from its recipe and your pantry."


def _results_in(request: dict) -> int:
    return sum(1 for m in request["messages"] if m.get("role") == "tool")


def _kitchen_run(request: dict):
    """A stand-in's run on the pancakes, by the results it has read: recipe, then the pantry for each ingredient, then
    the answer; each call with a thought before it."""
    k = _results_in(request)
    if k == 0:
        return ("Thought: I need the recipe of pancakes first.", [("recipe", {"dish": "pancakes"})])
    if k == 1:
        return ("Thought: Now I check the pantry for each ingredient.",
                [("pantry", {"item": item}) for item in ("flour", "milk", "eggs", "butter")])
    return A_PANCAKES


def _endless_run(request: dict):
    """A stand-in's run that never stops calling tools."""
    return ("Thought: Let me look again.", [("pantry", {"item": "milk"})])


def _hello_run(request: dict):
    return A_HELLO


def _sent(log: list) -> list:
    return [e for e in log if e["method"] == "POST" and isinstance(e["body"], dict)]


def _reply(entry) -> dict:
    try:
        return entry["answer"]["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return {}


def _essence(m):
    """What counts in a message the loop sends back: a tool's result by its role, the id of its call and its content
    (an extra field, such as the tool's name, does no harm); every other message as it is."""
    if isinstance(m, dict) and m.get("role") == "tool":
        return ("tool", m.get("tool_call_id"), m.get("content"))
    return m


def _judge_run(want, requests: int):
    def judge(got, running):
        sent = _sent(running.log)
        if not sent:
            return "stopped-early"   # the box says instead that no request reached the server
        for a, b in zip(sent, sent[1:]):   # each next request: the messages so far, the answer, and its results
            reply = _reply(a)
            calls = reply.get("tool_calls") or []
            if not calls:
                return "after-answer"
            results = [{"role": "tool", "tool_call_id": c["id"],
                        "content": FUNCTIONS[c["function"]["name"]](json.loads(c["function"]["arguments"]))} for c in calls]
            want_ = a["body"].get("messages") + [reply] + results
            have = b["body"].get("messages") or []
            if [_essence(m) for m in have] != [_essence(m) for m in want_]:
                k = len(want_) - len(results)   # what came before the results is right, and the results have the right ids
                theirs = [m for m in have[k:] if isinstance(m, dict) and m.get("role") == "tool"]
                if len(have) == len(want_) and [_essence(m) for m in have[:k]] == [_essence(m) for m in want_[:k]] \
                        and [m.get("tool_call_id") for m in theirs] == [r["tool_call_id"] for r in results]:
                    return "wrong-result"
                return "messages"
        if len(sent) < requests:
            return "stopped-early"
        if len(sent) > requests:
            return "past-limit"
        if got != want:
            return "limit-return" if want is None else "return"
        return None
    return judge


RUN_CASES = [
    Case(lambda: serve.StandIn(script=_kitchen_run), (URL, Q_PANCAKES, 6), _judge_run(A_PANCAKES, 3)),
    Case(lambda: serve.StandIn(script=_endless_run), (URL, Q_PANCAKES, 3), _judge_run(None, 3)),
    Case(lambda: serve.StandIn(script=_hello_run), (URL, Q_HELLO, 6), _judge_run(A_HELLO, 1)),
]

_RUN_WORDS_EX = {
    "en": {"title": "What your run_agent did on the pancakes, as a trace", "gave": "Your run_agent gave back: {v}",
           "error": "Your run_agent stopped with {e}.", "none": "No request reached the server.", "requests": "{k} requests"},
    "nl": {"title": "Wat je run_agent met de pancakes deed, als trace", "gave": "Je run_agent gaf terug: {v}",
           "error": "Je run_agent stopte met {e}.", "none": "Er kwam geen request aan bij de server.", "requests": "{k} requests"},
}


def log_messages(log: list) -> list:
    """The messages of a run as the server saw them: the last request's messages and its answer."""
    sent = _sent(log)
    if not sent:
        return []
    messages = list(sent[-1]["body"].get("messages") or [])
    reply = _reply(sent[-1])
    return messages + ([reply] if reply else [])


def _safe_trace(messages: list) -> list:
    try:
        return trace_of([m for m in messages if isinstance(m, dict) and m.get("role") in ("assistant", "tool")])
    except (KeyError, TypeError):
        return []


def run_html(log: list, words: dict, got=None, error=None) -> str:
    sent = _sent(log)
    if not sent:
        return f'<p style="{_UI};font-size:13px">{html.escape(words["none"])}</p>'
    body = trace_html(_safe_trace(log_messages(log)), 60)
    end = words["error"].format(e=type(error).__name__ + ": " + str(error)) if error is not None else words["gave"].format(v=repr(got))
    return (body + f'<p style="{_UI};font-size:13px;margin-top:0.4rem">{html.escape(words["requests"].format(k=len(sent)))}. '
            f'{html.escape(end)}</p>')


def run_picture_ex(seen) -> str:
    """What the learner's run_agent did with the stand-in's run on the pancakes, as a trace."""
    run_ = seen.run
    fn = run_.names.get("run_agent") if run_ is not None and run_.error is None else None
    if not callable(fn):
        return ""
    got, error, running = call(fn, RUN_CASES[0])
    w = _RUN_WORDS_EX[seen.lang]
    return yours.figure(w["title"], run_html(running.log, w, got, error), alt=w["title"])


# --- exercise 2: a trace of thought, action and observation ----------------------------------------------------------

START_TRACE = '''def trace(messages):
    lines = []
    for m in messages:
        if m["role"] == "tool":
            ...
        elif m["role"] == "assistant":
            ...
    return lines
'''
SOLUTION_TRACE = START_TRACE.replace('''        if m["role"] == "tool":
            ...
        elif m["role"] == "assistant":
            ...
''', '''        if m["role"] == "tool":
            lines.append("Observation: " + m["content"])
        elif m["role"] == "assistant":
            if m.get("tool_calls"):
                if m.get("content"):
                    lines.append(m["content"])
                for c in m["tool_calls"]:
                    lines.append("Action: " + c["function"]["name"] + "(" + c["function"]["arguments"] + ")")
            else:
                lines.append("Answer: " + m["content"])
''')


class Run(list):
    """The messages of a run, which a check's message names by a short name instead of writing them all out."""

    def __init__(self, name: str, messages: list):
        super().__init__(messages)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return Run(self.name, copy.deepcopy(list(self), memo))


def _call(i: int, name: str, arguments: dict) -> dict:
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def _played(script, question: str, steps: int = 6) -> list:
    """The messages of a stand-in's run, as the course's loop collects them."""
    running = serve.start(serve.StandIn(script=script))
    try:
        fn = run_code(SOLUTION_RUN, dict(AGENT)).names["run_agent"]
        fn(running.url, question, steps)
        return log_messages(running.log)
    finally:
        running.stop()


TRACE_PANCAKES = Run("run_pancakes", [
    {"role": "system", "content": REACT},
    {"role": "user", "content": Q_PANCAKES},
    {"role": "assistant", "content": "Thought: I need the recipe of pancakes first.", "tool_calls": [_call(0, "recipe", {"dish": "pancakes"})]},
    {"role": "tool", "tool_call_id": "call_0", "content": json.dumps(W.RECIPES["pancakes"])},
    {"role": "assistant", "content": "Thought: Now the pantry, for milk and eggs.",
     "tool_calls": [_call(1, "pantry", {"item": "milk"}), _call(2, "pantry", {"item": "eggs"})]},
    {"role": "tool", "tool_call_id": "call_1", "content": "200 grams"},
    {"role": "tool", "tool_call_id": "call_2", "content": "0 grams"},
    {"role": "assistant", "content": A_PANCAKES},
])
TRACE_SILENT = Run("run_without_thoughts", [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": QUESTIONS[2][0]},
    {"role": "assistant", "content": None, "tool_calls": [_call(0, "recipe", {"dish": "tomato soup"})]},
    {"role": "tool", "tool_call_id": "call_0", "content": json.dumps(W.RECIPES["tomato soup"])},
    {"role": "assistant", "content": "", "tool_calls": [_call(1, "pantry", {"item": "tomatoes"})]},
    {"role": "tool", "tool_call_id": "call_1", "content": "400 grams"},
    {"role": "assistant", "content": "Buy 400 grams of tomatoes."},
])
TRACE_HELLO = Run("run_without_calls", [{"role": "user", "content": Q_HELLO}, {"role": "assistant", "content": A_HELLO}])

_NO_THOUGHT = SOLUTION_TRACE.replace('''                if m.get("content"):
                    lines.append(m["content"])
''', "")
_ALWAYS_THOUGHT = SOLUTION_TRACE.replace('''                if m.get("content"):
                    lines.append(m["content"])
''', '''                lines.append(str(m["content"]))
''')
_NO_ANSWER = SOLUTION_TRACE.replace('''            else:
                lines.append("Answer: " + m["content"])
''', "")
_FIRST_CALL = SOLUTION_TRACE.replace('''                for c in m["tool_calls"]:
                    lines.append("Action: " + c["function"]["name"] + "(" + c["function"]["arguments"] + ")")
''', '''                c = m["tool_calls"][0]
                lines.append("Action: " + c["function"]["name"] + "(" + c["function"]["arguments"] + ")")
''')

_TRACE_WORDS = {
    "en": {"title": "Your trace of a run on the pancakes", "same": "as the course's trace", "differs": "the course's trace has here: {w}"},
    "nl": {"title": "Je trace van een run op de pancakes", "same": "zoals de trace van de course", "differs": "de trace van de course heeft hier: {w}"},
}


def trace_picture(seen) -> str:
    """The learner's trace of the run on the pancakes, line by line, beside the course's where they differ."""
    run_ = seen.run
    fn = run_.names.get("trace") if run_ is not None and run_.error is None else None
    if not callable(fn):
        return ""
    w = _TRACE_WORDS[seen.lang]
    try:
        got = fn(copy.deepcopy(TRACE_PANCAKES))
    except Exception as e:   # noqa: BLE001 - the learner's code
        return yours.figure(w["title"], _code(f"{type(e).__name__}: {e}", "var(--ea-loss-text)"), alt=w["title"])
    if not isinstance(got, list):
        return yours.figure(w["title"], _code(repr(got)), alt=w["title"])
    want = trace_of(TRACE_PANCAKES)
    rows = []
    for k in range(max(len(got), len(want))):
        mine = got[k] if k < len(got) else ""
        theirs = want[k] if k < len(want) else ""
        note = "" if mine == theirs else (f'<div style="{_UI};font-size:12px;color:var(--ea-loss-text)">'
                                          f'{html.escape(w["differs"].format(w=theirs or "–"))}</div>')
        rows.append(f'<div style="margin:0.15rem 0">{trace_html([mine]) if mine != "" else ""}{note}</div>')
    return yours.figure(w["title"], "".join(rows), alt=w["title"])


_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}


def _args_text(e) -> bool:
    """A tool that got its arguments as text: the course's tools read them with .get, which text does not have."""
    return isinstance(e, AttributeError) and "'str' object has no attribute 'get'" in str(e)


EXERCISES = {
    "run-agent": ApiBox(
        name="run-agent",
        picture=run_picture_ex,
        function="run_agent",
        start=START_RUN,
        solution=SOLUTION_RUN,
        names=AGENT,
        cases=RUN_CASES,
        mistakes=[
            Mistake("args-text", error=_args_text,
                    en="With `{call}`, a tool got its arguments as text: `call[\"function\"][\"arguments\"]` is a JSON text. Read it with `json.loads` first, and give the tool the dictionary that comes out.",
                    nl="Met `{call}` kreeg een tool zijn argumenten als tekst: `call[\"function\"][\"arguments\"]` is een JSON-tekst. Lees hem eerst met `json.loads`, en geef de tool de dictionary die eruit komt."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "The loop asks the model, but does nothing with its answer yet. Add the answer to `messages`; give back its content when it has no tool calls; otherwise run each call and add its result with the role `tool`.",
                      "nl": "De loop vraagt het model, maar doet nog niets met het antwoord. Zet het antwoord in `messages`; geef zijn content terug als het geen tool calls heeft; voer anders elke call uit en voeg het resultaat toe met de rol `tool`."},
            "http-400-messages": {"en": "With `{call}`, the server answered with status 400: \"{message}\" The assistant's message with its calls goes into `messages` first, as it came; each tool call then needs its result right after it: a message with the role `tool`, the call's `id` as `tool_call_id`, and the result as text.",
                                  "nl": "Met `{call}` antwoordde de server met status 400: \"{message}\" Het bericht van de assistant met zijn calls gaat eerst in `messages`, zoals het kwam; elke tool call heeft daarna zijn resultaat nodig, direct erna: een bericht met de rol `tool`, het `id` van de call als `tool_call_id`, en het resultaat als tekst."},
            "after-answer": {"en": "With `{call}`, the model answered without a tool call, and your loop asked it again. An answer without tool calls ends the run: give back its content.",
                             "nl": "Met `{call}` antwoordde het model zonder tool call, en je loop vroeg het opnieuw. Een antwoord zonder tool calls beëindigt de run: geef zijn content terug."},
            "messages": {"en": "With `{call}`, a request does not hold what it should: the messages so far, the model's answer as it came back, thought and calls included, and for each call a message with its result.",
                         "nl": "Met `{call}` bevat een request niet wat het zou moeten: de berichten tot nu toe, het antwoord van het model zoals het terugkwam, met gedachte en calls, en voor elke call een bericht met zijn resultaat."},
            "wrong-result": {"en": "With `{call}`, a result is not the result of the tool that was called. Run the function whose name the call gives: `FUNCTIONS[call[\"function\"][\"name\"]]`, with the call's own arguments, and send back what it gives as it is.",
                             "nl": "Met `{call}` is een resultaat niet het resultaat van de tool die werd aangeroepen. Voer de functie uit waarvan de call de naam geeft: `FUNCTIONS[call[\"function\"][\"name\"]]`, met de eigen argumenten van de call, en stuur terug wat die geeft, zoals het is."},
            "stopped-early": {"en": "With `{call}`, your loop stopped while the model still called tools. Give back only at an answer without tool calls, or after `max_steps` requests.",
                              "nl": "Met `{call}` stopte je loop terwijl het model nog tools aanriep. Geef pas iets terug bij een antwoord zonder tool calls, of na `max_steps` requests."},
            "past-limit": {"en": "With `{call}`, your loop sent more than `max_steps` requests. The step limit is what keeps a run that never stops from costing ever more time and tokens.",
                           "nl": "Met `{call}` stuurde je loop meer dan `max_steps` requests. De stappenlimiet zorgt ervoor dat een run die nooit stopt, niet steeds meer tijd en tokens kost."},
            "limit-return": {"en": "With `{call}`, the model still called tools when the step limit came, and your function gives back `{got}`. Give back `None` then, as the box did, so that your code knows the agent did not finish.",
                             "nl": "Met `{call}` riep het model nog tools aan toen de stappenlimiet kwam, en je functie geeft `{got}` terug. Geef dan `None` terug, zoals het vak deed, zodat je code weet dat de agent niet klaar kwam."},
            "return": {"en": "With `{call}`, your function gives back `{got}`. Give back the content of the model's answer without tool calls.",
                       "nl": "Met `{call}` geeft je functie `{got}` terug. Geef de content terug van het antwoord van het model zonder tool calls."},
            "correct": {"en": "Your `run_agent` is the whole loop: it asks, runs every call the model makes, sends each result back with the id of its call, and stops at an answer or at the step limit. Right for all {n} runs of the check.",
                        "nl": "Je `run_agent` is de hele loop: hij vraagt, voert elke call van het model uit, stuurt elk resultaat terug met het id van zijn call, en stopt bij een antwoord of bij de stappenlimiet. Goed bij alle {n} runs van de check."},
        },
    ),
    "trace": FunctionBox(
        name="trace",
        picture=trace_picture,
        function="trace",
        start=START_TRACE,
        solution=SOLUTION_TRACE,
        visible=[(TRACE_PANCAKES,)],
        hidden=[(TRACE_SILENT,), (TRACE_HELLO,)],
        show=lambda v: v if not isinstance(v, list) else "[" + ", ".join(repr(x) for x in v) + "]",
        mistakes=[
            Mistake("no-thought", wrong=_NO_THOUGHT,
                    en="With `{call}`, your trace leaves out the thoughts. When an answer with tool calls has text, that text is the model's thought: put it in the trace before its actions.",
                    nl="Met `{call}` laat je trace de gedachten weg. Heeft een antwoord met tool calls tekst, dan is die tekst de gedachte van het model: zet hem in de trace voor zijn acties."),
            Mistake("always-thought", matches=lambda got, text: isinstance(got, list) and any(x in (None, "", "None") for x in got),
                    en="With `{call}`, your trace gives back `{got}`, with a line for a thought the model did not write. Add the content only when there is one: `if m.get(\"content\"):`.",
                    nl="Met `{call}` geeft je trace `{got}` terug, met een regel voor een gedachte die het model niet schreef. Voeg de content alleen toe als die er is: `if m.get(\"content\"):`."),
            Mistake("no-answer", wrong=_NO_ANSWER,
                    en="With `{call}`, your trace ends without the model's answer. An assistant message without tool calls is the answer: add it as `\"Answer: \"` and its content.",
                    nl="Met `{call}` eindigt je trace zonder het antwoord van het model. Een bericht van de assistant zonder tool calls is het antwoord: voeg het toe als `\"Answer: \"` en zijn content."),
            Mistake("first-call", wrong=_FIRST_CALL,
                    en="With `{call}`, your trace shows only the first call of an answer, while the model made two at once. Add an `\"Action: \"` line for every call in `m[\"tool_calls\"]`.",
                    nl="Met `{call}` laat je trace alleen de eerste call van een antwoord zien, terwijl het model er twee tegelijk deed. Voeg een regel `\"Action: \"` toe voor elke call in `m[\"tool_calls\"]`."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "The loop walks through the messages but adds no lines yet. A tool message becomes an `\"Observation: \"` line; an assistant message gives its thought and an `\"Action: \"` line per call, or, without calls, an `\"Answer: \"` line.",
                      "nl": "De loop loopt door de berichten, maar voegt nog geen regels toe. Een bericht van een tool wordt een regel `\"Observation: \"`; een bericht van de assistant geeft zijn gedachte en een regel `\"Action: \"` per call, of, zonder calls, een regel `\"Answer: \"`."},
            "wrong": {"en": "With `{call}`, your trace is `{got}`, and it should be `{want}`. Compare them line by line in the picture.",
                      "nl": "Met `{call}` is je trace `{got}`, en het zou `{want}` moeten zijn. Vergelijk ze regel voor regel in het plaatje."},
            "almost": {"en": "Right for the run on the pancakes, but with `{call}` your trace is `{got}`, and it should be `{want}`. Does it work for a run without thoughts, and for an answer without any call?",
                       "nl": "Goed voor de run op de pancakes, maar met `{call}` is je trace `{got}`, en het zou `{want}` moeten zijn. Werkt het ook voor een run zonder gedachten, en voor een antwoord zonder enige call?"},
            "correct": {"en": "Your `trace` reads a run the way ReAct writes it: a thought, the actions, what each gave back as an observation, and at the end the answer. Right for all {n} runs of the check.",
                        "nl": "Je `trace` leest een run zoals ReAct hem schrijft: een gedachte, de acties, wat elke actie teruggaf als observatie, en aan het eind het antwoord. Goed bij alle {n} runs van de check."},
        },
    ),
}

REQUIRED = ["run-agent", "trace", "step-limit"]

_ANSWERED = '''        if not message.get("tool_calls"):
            return message["content"]
'''
CASES = [
    ("run-agent", START_RUN, "empty", None),
    ("run-agent", "", "empty", None),
    ("run-agent", SOLUTION_RUN, "correct", None),
    ("run-agent", SOLUTION_RUN.replace("arguments = json.loads(call[\"function\"][\"arguments\"])", "arguments = call[\"function\"][\"arguments\"]"), "error", "args-text"),
    ("run-agent", SOLUTION_RUN.replace(_ANSWERED, "").replace('for call in message["tool_calls"]:', 'for call in message.get("tool_calls") or []:'),
     "wrong", "after-answer"),
    ("run-agent", SOLUTION_RUN.replace('''            messages = messages + [{"role": "tool", "tool_call_id": call["id"], "content": result}]
''', '''            messages = messages + [{"role": "tool", "tool_call_id": call["id"], "content": result}]
            break
'''), "error", "http-400-messages"),
    ("run-agent", SOLUTION_RUN.replace('''        messages = messages + [message]
''', '''        messages = messages + [{"role": "assistant", "content": None, "tool_calls": message.get("tool_calls")}]
'''), "wrong", "messages"),
    ("run-agent", SOLUTION_RUN.replace('''            return message["content"]
''', '''            return message["content"]
        if step == 1:
            return None
'''), "wrong", "stopped-early"),
    ("run-agent", SOLUTION_RUN.replace("for step in range(max_steps):", "for step in range(max_steps + 1):"), "wrong", "past-limit"),
    ("run-agent", SOLUTION_RUN.replace('FUNCTIONS[call["function"]["name"]]', 'FUNCTIONS["recipe"]'), "wrong", "wrong-result"),
    ("run-agent", SOLUTION_RUN.replace('"tool_call_id": call["id"], ', '"tool_call_id": call["id"], "name": call["function"]["name"], '), "correct", None),
    ("run-agent", SOLUTION_RUN.replace('for call in message["tool_calls"]:', "for call in message:"), "error", None),
    ("run-agent", SOLUTION_RUN.replace("    return None\n", "    return \"I could not finish.\"\n"), "wrong", "limit-return"),
    ("run-agent", SOLUTION_RUN.replace('''            return message["content"]
''', '''            return message
'''), "wrong", "return"),
    ("trace", START_TRACE, "empty", None),
    ("trace", "", "empty", None),
    ("trace", SOLUTION_TRACE, "correct", None),
    ("trace", _NO_THOUGHT, "wrong", "no-thought"),
    ("trace", _ALWAYS_THOUGHT, "wrong", "always-thought"),
    ("trace", _NO_ANSWER, "wrong", "no-answer"),
    ("trace", _FIRST_CALL, "wrong", "first-call"),
    ("trace", SOLUTION_TRACE.replace('"Observation: "', '"Result: "'), "wrong", None),
    ("trace", SOLUTION_TRACE.replace('''                if m.get("content"):
                    lines.append(m["content"])
''', '''                if m["content"] is not None:
                    lines.append(m["content"])
'''), "wrong", "always-thought"),
    ("trace", SOLUTION_TRACE.replace('''                if m.get("content"):
                    lines.append(m["content"])
''', '''                lines.append(m["content"])
'''), "wrong", "always-thought"),
    ("trace", SOLUTION_TRACE.replace('''            else:
                lines.append("Answer: " + m["content"])
''', '''            elif len(lines) > 3:
                lines.append("Answer: " + m["content"])
'''), "almost", None),
]


# --- the demo: the learner's run_agent with Qwen3 ---------------------------------------------------------------------

def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


def learner_run_agent(text):
    """The learner's run_agent from exercise 1, when it is right; else None."""
    ex = EXERCISES["run-agent"]
    if text is MISSING or text is None or ex.check(text).state != "correct":
        return None
    return run_code(text, dict(AGENT)).names.get("run_agent")


def demo(run_fn, question: str, words: dict, lang: str) -> str:
    """The learner's run_agent on a question, with Qwen3 behind the course's server: the run as a trace."""
    running, before = None, 0
    try:
        running = serve.model_server()
        before = len(running.log)
        got, error = run_fn(running.url, question, STEPS), None
    except Exception as e:   # the learner's code, against the real server; it may have run out of memory
        if serve.out_of_memory(e):
            return f"<p><em>{html.escape(words['memory'])}</em></p>"
        got, error = None, e
    log = running.log[before:] if running is not None else []
    return yours.figure(words["title"], run_html(copy.deepcopy(log), words, got, error), alt=words["title"])
