# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-tool-calling: a tool described as OpenAI's API takes it (a name, a description and a JSON schema of
its arguments), and a function that sends a question with that tool, runs the tool calls of the answer and sends the
results back with the ids of their calls, against a stand-in of the server. The course's recorded runs of Qwen3-0.6B
(data/course-agents-qwen.json): eight questions that convert an amount, answered without tools and with the tool
convert; two questions that need no tool, with tool_choice "auto" and "none"; and the eight with tool_choice naming
convert. What they show, their pictures, and the demo with the learner's own function."""
import copy
import html
import json
import re

from everything_ai import chat, paths, serve, yours
from everything_ai.apicheck import URL, ApiBox, Case, call
from everything_ai.check import MISSING, CodeBox, Mistake, run_code

DOWNLOADS = {"Qwen3-0.6B": chat.FILES}   # what the demo downloads, for the line under the header
DATA = paths.ROOT / "data" / "course-agents-qwen.json"

# --- the tool: convert an amount from one unit to another ------------------------------------------------------------

UNITS = {"km": ("length", 1000.0), "mi": ("length", 1609.344), "kg": ("mass", 1.0), "lb": ("mass", 0.45359237)}
CONVERT = {
    "type": "function",
    "function": {
        "name": "convert",
        "description": "Convert an amount from one unit to another: kilometres and miles, or kilograms and pounds.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "the amount to convert, such as 37"},
                "from_unit": {"type": "string", "enum": ["km", "mi", "kg", "lb"], "description": "the unit of the amount"},
                "to_unit": {"type": "string", "enum": ["km", "mi", "kg", "lb"], "description": "the unit to convert to"},
            },
            "required": ["amount", "from_unit", "to_unit"],
        },
    },
}
TOOLS = [CONVERT]
CHOICE = {"type": "function", "function": {"name": "convert"}}


def convert(arguments: dict) -> str:
    """The course's tool: the amount in the other unit, rounded to two decimals, as text, such as "22.99 mi"; for
    arguments it cannot use, a short error as text, as a tool gives it back to the model."""
    amount, a, b = arguments.get("amount"), arguments.get("from_unit"), arguments.get("to_unit")
    if not isinstance(amount, (int, float)) or isinstance(amount, bool):
        return "Error: amount must be a number"
    if not all(isinstance(u, str) and u in UNITS for u in (a, b)):
        return "Error: unknown unit " + str(b if isinstance(a, str) and a in UNITS else a)
    if UNITS[a][0] != UNITS[b][0]:
        return "Error: cannot convert " + a + " to " + b
    return f"{round(amount * UNITS[a][1] / UNITS[b][1], 2):.2f}".rstrip("0").rstrip(".") + " " + b


FUNCTIONS = {"convert": convert}


def tool_results(message: dict, ask: dict | None = None) -> list:
    """For the recorder and the course: the result of each tool call in an answer, as a message with the role tool and
    the id of its call; arguments that are not JSON, or a tool that does not exist, get an error as text."""
    out = []
    for c in message.get("tool_calls") or []:
        try:
            arguments = json.loads(c["function"]["arguments"])
        except ValueError:
            arguments = None
        fn = FUNCTIONS.get(c["function"]["name"])
        result = ("Error: no tool " + c["function"]["name"] if fn is None else
                  "Error: the arguments are not a JSON object" if not isinstance(arguments, dict) else fn(arguments))
        out.append({"role": "tool", "tool_call_id": c["id"], "content": result})
    return out


# --- the questions of the recorded runs ------------------------------------------------------------------------------

QUESTIONS = [   # (question, amount, from, to)
    ("How many miles is 37 km?", 37, "km", "mi"),
    ("How many kilometres is 18 miles?", 18, "mi", "km"),
    ("How many pounds is 64 kg?", 64, "kg", "lb"),
    ("How many kilograms is 150 lb?", 150, "lb", "kg"),
    ("My bike ride was 23.5 km. How far is that in miles?", 23.5, "km", "mi"),
    ("A suitcase may weigh 50 lb. What is that in kg?", 50, "lb", "kg"),
    ("How many km is a 3.1 mile run?", 3.1, "mi", "km"),
    ("The bag of flour holds 2.5 kg. How many pounds is that?", 2.5, "kg", "lb"),
]
NO_NEED = ["What does the abbreviation lb stand for?", "Which is the unit of mass here: km, mi, kg or lb?"]


def asked(q: str) -> list:
    return [{"role": "user", "content": q}]


RECORD = {}
for _i, (_q, *_) in enumerate(QUESTIONS):
    RECORD[f"own-{_i}"] = {"messages": asked(_q), "max_tokens": 200, "temperature": 0, "seeds": 1}
    RECORD[f"auto-{_i}"] = {"messages": asked(_q), "tools": TOOLS, "max_tokens": 120, "temperature": 0, "steps": 2, "seeds": 1}
    RECORD[f"named-{_i}"] = {"messages": asked(_q), "tools": TOOLS, "tool_choice": CHOICE, "max_tokens": 120,
                             "temperature": 0, "steps": 2, "seeds": 1}
for _i, _q in enumerate(NO_NEED):
    RECORD[f"no-need-auto-{_i}"] = {"messages": asked(_q), "tools": TOOLS, "max_tokens": 120, "temperature": 0, "steps": 2, "seeds": 1}
    RECORD[f"no-need-none-{_i}"] = {"messages": asked(_q), "tools": TOOLS, "tool_choice": "none", "max_tokens": 120,
                                    "temperature": 0, "steps": 2, "seeds": 1}


# --- what the recorded runs show ----------------------------------------------------------------------------------------

def course() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-tool-calling"]


def run(key: str) -> dict:
    return course()[key]["answers"][0]


def true_value(i: int) -> float:
    """The amount of question i in the other unit, rounded to two decimals, as the tool gives it."""
    _q, amount, a, b = QUESTIONS[i]
    return round(amount * UNITS[a][1] / UNITS[b][1], 2)


_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def last_number(text: str):
    """The last number in a text, such as 23.4 in "Approximately **23.4 miles**."; None when there is none."""
    found = _NUMBER.findall(text)
    return float(found[-1].replace(",", "")) if found else None


def right(text: str, i: int) -> bool:
    """Whether an answer ends on the tool's number for question i, to two decimals."""
    n = last_number(text)
    return n is not None and round(n, 2) == true_value(i)


def arguments_of(call: dict) -> dict:
    return json.loads(call["function"]["arguments"])


def right_call(call: dict, i: int) -> bool:
    _q, amount, a, b = QUESTIONS[i]
    return call["function"]["name"] == "convert" and arguments_of(call) == {"amount": amount, "from_unit": a, "to_unit": b}


def seconds(steps: list) -> float:
    return sum(s["seconds"] for s in steps)


def findings() -> dict:
    """What the runs show: without tools, how many answers end on the tool's number and how long they took; with the
    tool and tool_choice "auto", how many called it, with the right arguments, and how many answers were right; the
    same with tool_choice naming convert; the questions that need no tool; and the prompt tokens with and without
    the tool's description."""
    n = len(QUESTIONS)
    own = [run(f"own-{i}") for i in range(n)]
    auto = [run(f"auto-{i}")["steps"] for i in range(n)]
    named = [run(f"named-{i}")["steps"] for i in range(n)]
    out = {"questions": n,
           "own_right": sum(right(a["text"], i) for i, a in enumerate(own)),
           "own_seconds": sum(a["seconds"] for a in own) / n,
           "own_tokens": sum(a["tokens"] for a in own) / n,
           "auto_called": [i for i, s in enumerate(auto) if s[0]["tool_calls"]],
           "auto_right": sum(right(s[-1]["text"], i) for i, s in enumerate(auto)),
           "named_called": sum(bool(s[0]["tool_calls"]) for s in named),
           "named_args": sum(bool(s[0]["tool_calls"]) and right_call(s[0]["tool_calls"][0], i) for i, s in enumerate(named)),
           "named_right": sum(right(s[-1]["text"], i) for i, s in enumerate(named)),
           "prompt_own": own[0]["prompt_tokens"], "prompt_tools": auto[0][0]["prompt_tokens"]}
    called = out["auto_called"]
    out["auto_args"] = sum(right_call(auto[i][0]["tool_calls"][0], i) for i in called)
    out["auto_called_right"] = sum(right(auto[i][-1]["text"], i) for i in called)
    out["auto_missed"] = [i for i in range(n) if i not in called]
    out["tool_seconds"] = sum(seconds(auto[i]) for i in called) / len(called)
    no_need = [run(f"no-need-auto-{i}")["steps"] for i in range(len(NO_NEED))]
    out["no_need_calls"] = sum(len(s[0]["tool_calls"]) for s in no_need)
    out["none_prompt"] = run("no-need-none-0")["steps"][0]["prompt_tokens"]
    out["auto_prompt_no_need"] = no_need[0][0]["prompt_tokens"]
    return out


WORKED = 1    # the question of the worked example and the picture of the exchange: 18 miles in km
WRONG = 3     # the question Qwen3 answered without the tool, and wrongly: 150 lb in kg


def number(x: float, lang: str, digits: int = 1) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


# --- the pictures of the recorded runs ------------------------------------------------------------------------------------

_CELL = "padding:0.18rem 0.5rem;border-bottom:1px solid var(--ea-line);vertical-align:top;font-size:13px"
_HEAD = f"{_CELL};color:var(--ea-ink-2);font-weight:500;text-align:left;font-family:var(--ea-font-ui)"


def _label(text: str) -> str:
    return f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2);margin:0.35rem 0 0.1rem">{html.escape(text)}</div>'


def _code(text: str, color: str = "var(--ea-ink)") -> str:
    return (f'<span style="font-family:var(--ea-font-code);font-size:13px;color:{color};white-space:pre-wrap;'
            f'overflow-wrap:anywhere">{html.escape(text)}</span>')


def _call_text(c) -> str:
    """A tool call as the picture shows it: convert({"amount": 18, ...}), with its id."""
    try:
        f = c["function"]
        return f'{f["name"]}({f["arguments"]})'
    except (KeyError, TypeError):
        return repr(c)


def message_rows(messages, words: dict) -> str:
    """The messages of a request, one row each: the role, then what it holds. What the model wrote is in the colour
    of a model's output, a tool's result in the colour of data, the rest in ink."""
    rows = []
    for m in messages if isinstance(messages, list) else []:
        if not isinstance(m, dict):
            rows.append((words.get("odd", "?"), _code(repr(m))))
            continue
        role = str(m.get("role"))
        if role == "assistant" and m.get("tool_calls"):
            calls = m["tool_calls"] if isinstance(m["tool_calls"], list) else []
            inner = "<br>".join(_code(_call_text(c), "var(--ea-pred-text)") + " " + _code(words["id"].format(id=c.get("id") if isinstance(c, dict) else "?"), "var(--ea-ink-2)")
                                for c in calls)
            if m.get("content"):
                inner = _code(str(m["content"]), "var(--ea-pred-text)") + "<br>" + inner
        elif role == "assistant":
            inner = _code(str(m.get("content")), "var(--ea-pred-text)")
        elif role == "tool":
            inner = (_code(words["answers"].format(id=m.get("tool_call_id")), "var(--ea-ink-2)") + " "
                     + _code(str(m.get("content")), "var(--ea-data-text)"))
        else:
            inner = _code(str(m.get("content")))
        rows.append((role, inner))
    return "".join(f'<div style="display:flex;gap:0.4rem;border-left:3px solid var(--ea-line);padding:0.1rem 0.5rem;margin:0.15rem 0">'
                   f'<span style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2);flex:0 0 5rem;line-height:1.6">'
                   f'{html.escape(r)}</span><div style="min-width:0;line-height:1.5">{inner}</div></div>'
                   for r, inner in rows)


def _answer_row(message: dict, finish, words: dict) -> str:
    return (message_rows([message], words)
            + f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2);margin-left:0.6rem">{html.escape(words["finish"].format(f=finish))}</div>')


_TALK_WORDS = {
    "en": {"title": "Function calling, as it went for: {q}", "request": "Request {n}: your program sends these messages, with the tool convert",
           "answer": "The server answers:", "id": "id {id}", "answers": "answers {id}:", "finish": "finish_reason: {f}"},
    "nl": {"title": "Function calling, zoals het ging bij: {q}", "request": "Request {n}: je programma stuurt deze berichten, met de tool convert",
           "answer": "De server antwoordt:", "id": "id {id}", "answers": "antwoord op {id}:", "finish": "finish_reason: {f}"},
}


def recorded_messages(i: int) -> list:
    """The messages of the requests of the run with tool_choice "auto" for question i, rebuilt from the recording,
    with each answer as the server gave it."""
    steps, messages, out = run(f"auto-{i}")["steps"], asked(QUESTIONS[i][0]), []
    for s in steps:
        message = {"role": "assistant", "content": s["text"] or None, **({"tool_calls": s["tool_calls"]} if s["tool_calls"] else {})}
        out.append((messages, message, s["finish"]))
        messages = messages + [message] + [{"role": "tool", "tool_call_id": c["id"], "content": r}
                                           for c, r in zip(s["tool_calls"], s["results"])]
    return out


def talk_picture(lang: str) -> str:
    """The two requests and answers of the worked example's run."""
    w = _TALK_WORDS[lang]
    body = []
    for n, (messages, message, finish) in enumerate(recorded_messages(WORKED)):
        body.append(_label(w["request"].format(n=n + 1)) + message_rows(messages, w))
        body.append(_label(w["answer"]) + _answer_row(message, finish, w))
    title = w["title"].format(q=QUESTIONS[WORKED][0])
    return yours.figure(title, "".join(body), alt=title)


_TABLE_WORDS = {
    "en": {"title": "Eight conversions, three ways", "question": "the question", "own": "without tools",
           "auto": "with convert, \"auto\"", "named": "with convert, named", "called": "called convert", "no_call": "no call",
           "count": "{n} of {k} right", "right": "right: {v}", "wrong": "should be {v}"},
    "nl": {"title": "Acht omrekeningen, op drie manieren", "question": "de vraag", "own": "zonder tools",
           "auto": "met convert, \"auto\"", "named": "met convert, bij naam", "called": "riep convert aan", "no_call": "geen call",
           "count": "{n} van de {k} goed", "right": "goed: {v}", "wrong": "moet {v} zijn"},
}


def _verdict(text: str, i: int, w: dict, lang: str) -> str:
    n = last_number(text)
    shown = "–" if n is None else (f"{n:g}".replace(".", ",") if lang == "nl" else f"{n:g}")
    ok = right(text, i)
    color = "var(--ea-ink)" if ok else "var(--ea-loss-text)"
    note = "" if ok else f' <span style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-loss-text)">{html.escape(w["wrong"].format(v=number(true_value(i), lang, 2).rstrip("0").rstrip(",.")))}</span>'
    return f'<span style="font-family:var(--ea-font-code);color:{color};font-weight:600">{shown}</span>{note}'


def table_picture(lang: str) -> str:
    """For each question: the last number of the answer without tools, with the tool and "auto", and with the tool
    named in tool_choice, a wrong one marked with the right one beside it; whether the model called the tool; and
    under each way how many were right."""
    w = _TABLE_WORDS[lang]
    rows = []
    for i, (q, *_) in enumerate(QUESTIONS):
        own, auto, named = run(f"own-{i}")["text"], run(f"auto-{i}")["steps"], run(f"named-{i}")["steps"]
        cells = [_verdict(own, i, w, lang)]
        for steps in (auto, named):
            called = w["called"] if steps[0]["tool_calls"] else w["no_call"]
            cells.append(_verdict(steps[-1]["text"], i, w, lang)
                         + f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)">{html.escape(called)}</div>')
        rows.append(f'<tr><td style="{_CELL};font-family:var(--ea-font-ui);color:var(--ea-ink)">{html.escape(q)}</td>'
                    + "".join(f'<td style="{_CELL}">{c}</td>' for c in cells) + "</tr>")
    f = findings()
    counts = [f["own_right"], f["auto_right"], f["named_right"]]
    rows.append(f'<tr><td style="{_CELL};border-bottom:none"></td>' + "".join(
        f'<td style="{_CELL};border-bottom:none;font-family:var(--ea-font-ui);font-weight:600;color:var(--ea-ink)">{w["count"].format(n=c, k=f["questions"])}</td>'
        for c in counts) + "</tr>")
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%;min-width:34rem"><thead><tr>'
             + "".join(f'<th style="{_HEAD}">{h}</th>' for h in (w["question"], w["own"], w["auto"], w["named"]))
             + f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    return yours.figure(w["title"], table, alt=w["title"])


# --- exercise 1: describe the tool ------------------------------------------------------------------------------------

START_DESCRIBE = '''convert_tool = {
    "type": "function",
    "function": {
        "name": "convert",
        "description": ("Convert an amount from one unit to another: "
                        "kilometres and miles, or kilograms and pounds."),
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "the amount to convert, such as 37"},
            },
            "required": ["amount"],
        },
    },
}
convert_tool
'''
SOLUTION_DESCRIBE = START_DESCRIBE.replace('''                "amount": {"type": "number", "description": "the amount to convert, such as 37"},
            },
            "required": ["amount"],''', '''                "amount": {"type": "number", "description": "the amount to convert, such as 37"},
                "from_unit": {"type": "string", "enum": ["km", "mi", "kg", "lb"],
                              "description": "the unit of the amount"},
                "to_unit": {"type": "string", "enum": ["km", "mi", "kg", "lb"],
                            "description": "the unit to convert to"},
            },
            "required": ["amount", "from_unit", "to_unit"],''')
UNIT_ARGS = ("from_unit", "to_unit")


def _parts(tool):
    """The function and its properties of a tool description, or None where the shape is not that of a tool."""
    f = tool.get("function") if isinstance(tool, dict) else None
    p = f.get("parameters") if isinstance(f, dict) else None
    props = p.get("properties") if isinstance(p, dict) else None
    return f, p, props if isinstance(props, dict) else None


def _same_list(xs, want) -> bool:
    """True when `xs` is a list of the same texts as `want`, in any order; a list with anything but text in it is not."""
    return isinstance(xs, list) and all(isinstance(x, str) for x in xs) and sorted(xs) == sorted(want)


def _same_tool(got, want) -> bool:
    """Right when it is convert's description with the same arguments, types, allowed units and required arguments;
    the descriptions may be any text."""
    f, p, props = _parts(got)
    if not (isinstance(got, dict) and got.get("type") == "function" and props is not None and f.get("name") == "convert"
            and isinstance(f.get("description"), str) and f["description"].strip() and p.get("type") == "object"):
        return False
    if set(props) != {"amount", *UNIT_ARGS} or not isinstance(props["amount"], dict) or props["amount"].get("type") != "number":
        return False
    for key in UNIT_ARGS:
        rule = props[key]
        if not isinstance(rule, dict) or rule.get("type") != "string" or not isinstance(rule.get("enum"), list) \
                or not _same_list(rule["enum"], UNITS) or not isinstance(rule.get("description", ""), str):
            return False
    return _same_list(p.get("required"), ["amount", *UNIT_ARGS])


def _tool_mistake(kind: str):
    def matches(got, text) -> bool:
        _f, p, props = _parts(got)
        if props is None:
            return False
        units = [props.get(k) for k in UNIT_ARGS]
        if kind == "names":
            return set(props) != {"amount", *UNIT_ARGS}
        if set(props) != {"amount", *UNIT_ARGS} or not all(isinstance(u, dict) for u in units):
            return False
        if kind == "no-enum":
            return any("enum" not in u for u in units)
        if kind == "enum-text":
            return any(isinstance(u.get("enum"), str) for u in units)
        if kind == "enum-units":
            return any(isinstance(u.get("enum"), list) and not _same_list(u["enum"], UNITS) for u in units)
        if kind == "type":
            return any(u.get("type") != "string" for u in units)
        if kind == "required":
            required = p.get("required")
            return not _same_list(required, ["amount", *UNIT_ARGS])
        return False
    return matches


_DESCRIBE_WORDS = {
    "en": {"title": "Your description of convert", "argument": "argument", "type": "type", "values": "allowed values",
           "required": "required", "yes": "yes", "no": "no", "any": "any value", "reads": "Qwen3 reads it as this line in its system prompt:"},
    "nl": {"title": "Je beschrijving van convert", "argument": "argument", "type": "type", "values": "toegestane waarden",
           "required": "verplicht", "yes": "ja", "no": "nee", "any": "elke waarde", "reads": "Qwen3 leest hem als deze regel in zijn system prompt:"},
}


def describe_picture(seen) -> str:
    """The learner's description as a table of its arguments (a unit without allowed values, or an argument that is
    not required, marked), and the line of the system prompt in which the model reads it."""
    w = _DESCRIBE_WORDS[seen.lang]
    run = seen.run
    tool = run.value if run is not None and run.error is None and getattr(run, "has_value", False) else None
    _f, p, props = _parts(tool)
    if props is None:
        return ""
    required = p.get("required") if isinstance(p.get("required"), list) else []
    rows = []
    for key, rule in props.items():
        rule = rule if isinstance(rule, dict) else {}
        enum = rule.get("enum")
        needs = key in UNIT_ARGS
        values = ", ".join(map(str, enum)) if isinstance(enum, list) else (str(enum) if enum is not None else w["any"])
        off_values = needs and not _same_list(enum, UNITS)
        off_req = key not in required
        mark = "color:var(--ea-loss-text);font-weight:600"
        rows.append(f'<tr><td style="{_CELL}">{_code(str(key))}</td><td style="{_CELL}">{_code(str(rule.get("type")))}</td>'
                    f'<td style="{_CELL};font-family:var(--ea-font-ui);{mark if off_values else ""}">{html.escape(values)}</td>'
                    f'<td style="{_CELL};font-family:var(--ea-font-ui);{mark if off_req else ""}">{w["yes"] if not off_req else w["no"]}</td></tr>')
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><thead><tr>'
             + "".join(f'<th style="{_HEAD}">{h}</th>' for h in (w["argument"], w["type"], w["values"], w["required"]))
             + f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    try:
        line = json.dumps(tool, ensure_ascii=False)
    except (TypeError, ValueError):
        line = repr(tool)
    reads = (_label(w["reads"]) + f'<pre style="margin:0.2rem 0;padding:0.5rem 0.6rem;background:var(--ea-inset);border-radius:6px;'
             f'font-family:var(--ea-font-code);font-size:12px;line-height:1.4;white-space:pre-wrap;overflow-wrap:anywhere;color:var(--ea-ink)">'
             f'{html.escape(line)}</pre>')
    return yours.figure(w["title"], table + reads, alt=w["title"])


# --- exercise 2: one round of function calling --------------------------------------------------------------------------

START_ANSWER = '''import json
import urllib.request


def ask(url, messages, tools):
    body = {"model": "qwen3-0.6b", "messages": messages, "tools": tools, "max_tokens": 120, "temperature": 0}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        answer = json.loads(response.read())
    return answer["choices"][0]["message"]


def answer(url, question, tools, functions):
    messages = [{"role": "user", "content": question}]
    message = ask(url, messages, tools)
    if message.get("tool_calls"):
        messages = messages + [message]
        for call in message["tool_calls"]:
            ...
        message = ask(url, messages, tools)
    return message["content"]
'''
SOLUTION_ANSWER = START_ANSWER.replace('''            ...
''', '''            arguments = json.loads(call["function"]["arguments"])
            result = functions[call["function"]["name"]](arguments)
            messages = messages + [{"role": "tool", "tool_call_id": call["id"], "content": result}]
''')

TODAY = {"type": "function", "function": {"name": "today", "description": "Today's date.",
                                          "parameters": {"type": "object", "properties": {}}}}


def today(arguments: dict) -> str:
    return "2026-11-16"


Q_MILES = QUESTIONS[0][0]
Q_TWO = "What is 5 km in miles, and what is the date today?"
Q_LB = NO_NEED[0]
_MILES = {"amount": 37, "from_unit": "km", "to_unit": "mi"}
_FIVE = {"amount": 5, "from_unit": "km", "to_unit": "mi"}
BOTH = {"convert": convert, "today": today}


def _sent(log: list) -> list:
    return [e for e in log if e["method"] == "POST" and isinstance(e["body"], dict)]


def _reply(entry) -> dict:
    try:
        return entry["answer"]["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return {}


def _judge_answer(functions: dict, want: str):
    def judge(got, running):
        sent = _sent(running.log)
        if not sent:
            return None   # the box says that no request reached the server
        reply = _reply(sent[0])
        calls = reply.get("tool_calls") or []
        if calls:
            if len(sent) < 2:
                return "not-again"
            results = [{"role": "tool", "tool_call_id": c["id"],
                        "content": functions[c["function"]["name"]](json.loads(c["function"]["arguments"]))} for c in calls]
            second = sent[1]["body"].get("messages")
            if second != sent[0]["body"].get("messages") + [reply] + results:
                theirs = [m for m in second or [] if isinstance(m, dict) and m.get("role") == "tool"]
                if [m.get("tool_call_id") for m in theirs] == [r["tool_call_id"] for r in results] \
                        and [m.get("content") for m in theirs] != [r["content"] for r in results]:
                    return "wrong-result"
                return "messages"
        if len(sent) != (2 if calls else 1):
            return "requests"
        if got != want:
            return "return"
        return None
    return judge


ANSWER_CASES = [
    Case(lambda: serve.StandIn(calls={Q_MILES: ("convert", _MILES)}, after="37 km is {result}."),
         (URL, Q_MILES, [CONVERT], FUNCTIONS), _judge_answer(FUNCTIONS, "37 km is 22.99 mi."),
         shown_args=("url", repr(Q_MILES), "[CONVERT]", "FUNCTIONS")),
    Case(lambda: serve.StandIn(calls={Q_TWO: [("convert", _FIVE), ("today", {})]}, after="Read: {result}"),
         (URL, Q_TWO, [CONVERT, TODAY], BOTH), _judge_answer(BOTH, "Read: 3.11 mi\n2026-11-16"),
         shown_args=("url", repr(Q_TWO), "[CONVERT, TODAY]", "FUNCTIONS")),
    Case(lambda: serve.StandIn({Q_LB: "It stands for pound."}),
         (URL, Q_LB, [CONVERT], FUNCTIONS), _judge_answer(FUNCTIONS, "It stands for pound."),
         shown_args=("url", repr(Q_LB), "[CONVERT]", "FUNCTIONS")),
]


def _args_text(e) -> bool:
    return isinstance(e, (AttributeError, TypeError)) and ("'str'" in str(e) or "string indices" in str(e))


_TRY_WORDS = {
    "en": {"title": "What your answer sent, and what came back", "request": "Request {n}: your answer sent these messages",
           "answer": "The server answered:", "id": "id {id}", "answers": "answers {id}:", "finish": "finish_reason: {f}",
           "status": "The server answered with status {s}: {m}", "gave": "Your function gave back: {v}",
           "error": "Your function stopped with {e}.", "none": "Your function sent nothing to the server yet."},
    "nl": {"title": "Wat je answer stuurde, en wat er terugkwam", "request": "Request {n}: je answer stuurde deze berichten",
           "answer": "De server antwoordde:", "id": "id {id}", "answers": "antwoord op {id}:", "finish": "finish_reason: {f}",
           "status": "De server antwoordde met status {s}: {m}", "gave": "Je functie gaf terug: {v}",
           "error": "Je functie stopte met {e}.", "none": "Je functie stuurde nog niets naar de server."},
}


def exchange_html(log: list, words: dict, got=None, error=None) -> str:
    """The requests in a server's log, each with its messages and what came back, and what the function gave."""
    sent = _sent(log)
    if not sent:
        return _label(words["none"])
    body = []
    for n, e in enumerate(sent):
        body.append(_label(words["request"].format(n=n + 1)) + message_rows(e["body"].get("messages"), words))
        if e["status"] == 200 and isinstance(e.get("answer"), dict):
            choice = e["answer"]["choices"][0]
            body.append(_label(words["answer"]) + _answer_row(choice["message"], choice["finish_reason"], words))
        else:
            said = ((e.get("answer") or {}).get("error") or {}).get("message", "") if isinstance(e.get("answer"), dict) else ""
            body.append(f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-loss-text);margin-left:0.6rem">'
                        f'{html.escape(words["status"].format(s=e["status"], m=said))}</div>')
    body.append(_label(words["error"].format(e=type(error).__name__) if error is not None else words["gave"].format(v=repr(got))))
    return "".join(body)


def try_picture(seen) -> str:
    """What the learner's answer sent for the question with two calls, against the stand-in, and what came back."""
    w = _TRY_WORDS[seen.lang]
    run = seen.run
    fn = run.names.get("answer") if run is not None and run.error is None else None
    if not callable(fn):
        return ""
    got, error, running = call(fn, ANSWER_CASES[1])
    return yours.figure(w["title"], exchange_html(running.log, w, got, error), alt=w["title"])


_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}

EXERCISES = {
    "describe": CodeBox(
        name="describe",
        picture=describe_picture,
        start=START_DESCRIBE,
        solution=SOLUTION_DESCRIBE,
        compare=_same_tool,
        mistakes=[
            Mistake("names", matches=_tool_mistake("names"),
                    en="Your tool has the arguments {got_names}. The function convert reads `amount`, `from_unit` and `to_unit`: give `\"properties\"` exactly those three keys, so that the arguments the model writes fit the function.",
                    nl="Je tool heeft de argumenten {got_names}. De functie convert leest `amount`, `from_unit` en `to_unit`: geef `\"properties\"` precies die drie keys, zodat de argumenten die het model schrijft bij de functie passen.",
                    detail=lambda got, text: {"got_names": ", ".join("`" + str(k) + "`" for k in _parts(got)[2])}),
            Mistake("type", matches=_tool_mistake("type"),
                    en="A unit is a short text, such as `\"km\"`: give `from_unit` and `to_unit` the type `\"string\"`.",
                    nl="Een eenheid is een korte tekst, zoals `\"km\"`: geef `from_unit` en `to_unit` het type `\"string\"`."),
            Mistake("no-enum", matches=_tool_mistake("no-enum"),
                    en="`from_unit` and `to_unit` have no `\"enum\"`, so any text fits, and the model may write `\"miles\"`, which convert does not know. Add `\"enum\": [\"km\", \"mi\", \"kg\", \"lb\"]` to both.",
                    nl="`from_unit` en `to_unit` hebben geen `\"enum\"`, dus past elke tekst, en kan het model `\"miles\"` schrijven, dat convert niet kent. Zet `\"enum\": [\"km\", \"mi\", \"kg\", \"lb\"]` bij allebei."),
            Mistake("enum-text", matches=_tool_mistake("enum-text"),
                    en="`\"enum\"` holds one text now. JSON Schema wants a list of the allowed values: `[\"km\", \"mi\", \"kg\", \"lb\"]`.",
                    nl="`\"enum\"` bevat nu één tekst. JSON Schema wil een lijst van de toegestane waarden: `[\"km\", \"mi\", \"kg\", \"lb\"]`."),
            Mistake("enum-units", matches=_tool_mistake("enum-units"),
                    en="The allowed values must be the four units convert knows, written as it writes them: `\"km\"`, `\"mi\"`, `\"kg\"` and `\"lb\"`.",
                    nl="De toegestane waarden moeten de vier eenheden zijn die convert kent, geschreven zoals convert ze schrijft: `\"km\"`, `\"mi\"`, `\"kg\"` en `\"lb\"`."),
            Mistake("required", matches=_tool_mistake("required"),
                    en="convert needs all three arguments, but `\"required\"` does not list them all. Without a unit, it cannot convert: put `\"amount\"`, `\"from_unit\"` and `\"to_unit\"` in `\"required\"`.",
                    nl="convert heeft alle drie de argumenten nodig, maar `\"required\"` noemt ze niet allemaal. Zonder eenheid kan hij niet omrekenen: zet `\"amount\"`, `\"from_unit\"` en `\"to_unit\"` in `\"required\"`."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "The tool still has only the argument `amount`. Add `from_unit` and `to_unit` to `\"properties\"`, each a string from `km`, `mi`, `kg` and `lb`, and make all three required.",
                      "nl": "De tool heeft nog alleen het argument `amount`. Zet `from_unit` en `to_unit` in `\"properties\"`, elk een string uit `km`, `mi`, `kg` en `lb`, en maak alle drie verplicht."},
            "no_value": {"en": "Keep `convert_tool` as the last line of the box, so that the check gets your description.",
                         "nl": "Laat `convert_tool` als laatste regel in het vak staan, zodat de check je beschrijving krijgt."},
            "wrong": {"en": "This is not yet a description of convert as the API takes it. Keep `\"type\": \"function\"`, the name `convert` with a description, and `\"parameters\"` with `\"type\": \"object\"`; then add the two units.",
                      "nl": "Dit is nog geen beschrijving van convert zoals de API hem neemt. Houd `\"type\": \"function\"`, de naam `convert` met een beschrijving, en `\"parameters\"` met `\"type\": \"object\"`; zet er dan de twee eenheden bij."},
            "correct": {"en": "Your description is right: three required arguments, a number and two units that can only be `km`, `mi`, `kg` or `lb`. That is all the model knows about convert.",
                        "nl": "Je beschrijving klopt: drie verplichte argumenten, een getal en twee eenheden die alleen `km`, `mi`, `kg` of `lb` kunnen zijn. Meer weet het model niet over convert."},
        },
    ),
    "answer": ApiBox(
        name="answer",
        picture=try_picture,
        function="answer",
        start=START_ANSWER,
        solution=SOLUTION_ANSWER,
        cases=ANSWER_CASES,
        mistakes=[
            Mistake("args-text", error=_args_text,
                    en="With `{call}`, the tool got the arguments as text: `call[\"function\"][\"arguments\"]` is a JSON text such as `'{{\"amount\": 37, ...}}'`. Read it with `json.loads` first, and give the tool the dictionary that comes out.",
                    nl="Met `{call}` kreeg de tool de argumenten als tekst: `call[\"function\"][\"arguments\"]` is een JSON-tekst zoals `'{{\"amount\": 37, ...}}'`. Lees hem eerst met `json.loads`, en geef de tool de dictionary die eruit komt."),
            Mistake("data-text", error=lambda e: isinstance(e, TypeError) and "bytes" in str(e),
                    en="urllib sends bytes, not text. Keep `.encode()` after `json.dumps(body)`, as the box had it.",
                    nl="urllib verstuurt bytes, geen tekst. Laat `.encode()` achter `json.dumps(body)` staan, zoals het in het vak stond."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "The loop over the tool calls does nothing yet, so the second request has no results, and the server refuses it. For each call, read its arguments, run its function, and add its result as a message with the role `tool`.",
                      "nl": "De loop over de tool calls doet nog niets, dus het tweede request heeft geen resultaten, en de server weigert het. Lees voor elke call zijn argumenten, voer zijn functie uit, en voeg het resultaat toe als bericht met de rol `tool`."},
            "http-400-messages": {"en": "With `{call}`, the server answered with status 400: \"{message}\" Each tool call needs its result right after the assistant's message: a message with the role `tool`, the call's `id` as `tool_call_id`, and the result as text.",
                                  "nl": "Met `{call}` antwoordde de server met status 400: \"{message}\" Elke tool call heeft zijn resultaat nodig, direct na het bericht van de assistant: een bericht met de rol `tool`, het `id` van de call als `tool_call_id`, en het resultaat als tekst."},
            "not-again": {"en": "With `{call}`, the answer had a tool call, but your function did not ask again with the result. After the loop, ask the model again, as the box did.",
                          "nl": "Met `{call}` had het antwoord een tool call, maar je functie vroeg niet opnieuw met het resultaat. Vraag het model na de loop opnieuw, zoals het vak deed."},
            "messages": {"en": "With `{call}`, the second request does not hold what it should: the question, the assistant's message as it came back, and for each call a message with its result, in the order of the calls.",
                         "nl": "Met `{call}` bevat het tweede request niet wat het zou moeten: de vraag, het bericht van de assistant zoals het terugkwam, en voor elke call een bericht met zijn resultaat, in de volgorde van de calls."},
            "wrong-result": {"en": "With `{call}`, a result is not the result of the tool that was called. Run the function whose name the call gives: `functions[call[\"function\"][\"name\"]]`, with the call's own arguments.",
                             "nl": "Met `{call}` is een resultaat niet het resultaat van de tool die werd aangeroepen. Voer de functie uit waarvan de call de naam geeft: `functions[call[\"function\"][\"name\"]]`, met de eigen argumenten van de call."},
            "requests": {"en": "With `{call}`, your function sent another number of requests than it should: one, and one more only when the answer has tool calls.",
                         "nl": "Met `{call}` stuurde je functie een ander aantal requests dan zou moeten: één, en er nog één alleen als het antwoord tool calls heeft."},
            "return": {"en": "With `{call}`, your function gives back `{got}`. Give back the content of the model's last answer, as the box did.",
                       "nl": "Met `{call}` geeft je functie `{got}` terug. Geef de content van het laatste antwoord van het model terug, zoals het vak deed."},
            "correct": {"en": "Your `answer` runs every tool the model calls and sends each result back with the id of its call: right for all {n} calls of the check, with one call, with two at once, and with none.",
                        "nl": "Je `answer` voert elke tool uit die het model aanroept en stuurt elk resultaat terug met het id van zijn call: goed bij alle {n} aanroepen van de check, met één call, met twee tegelijk, en met geen."},
        },
    ),
}

REQUIRED = ["describe", "answer", "tool-choice"]

_LOOP = '''            arguments = json.loads(call["function"]["arguments"])
            result = functions[call["function"]["name"]](arguments)
            messages = messages + [{"role": "tool", "tool_call_id": call["id"], "content": result}]
'''

CASES = [
    ("describe", START_DESCRIBE, "empty", None),
    ("describe", "", "empty", None),
    ("describe", SOLUTION_DESCRIBE, "correct", None),
    ("describe", SOLUTION_DESCRIBE.replace('"the unit of the amount"', '"km, mi, kg or lb: the unit the amount is in"'), "correct", None),
    ("describe", SOLUTION_DESCRIBE.replace('"from_unit"', '"from"').replace('"to_unit"', '"to"'), "wrong", "names"),
    ("describe", SOLUTION_DESCRIBE.replace('"from_unit": {"type": "string"', '"from_unit": {"type": "enum"'), "wrong", "type"),
    ("describe", SOLUTION_DESCRIBE.replace(', "enum": ["km", "mi", "kg", "lb"]', ""), "wrong", "no-enum"),
    ("describe", SOLUTION_DESCRIBE.replace('"enum": ["km", "mi", "kg", "lb"]', '"enum": "km, mi, kg, lb"'), "wrong", "enum-text"),
    ("describe", SOLUTION_DESCRIBE.replace('"enum": ["km", "mi", "kg", "lb"],\n                              ',
                                           '"enum": ["kilometres", "miles", "kilograms", "pounds"],\n                              '), "wrong", "enum-units"),
    ("describe", SOLUTION_DESCRIBE.replace('"required": ["amount", "from_unit", "to_unit"]', '"required": ["amount"]'), "wrong", "required"),
    ("describe", SOLUTION_DESCRIBE.replace('"enum": ["km", "mi", "kg", "lb"],\n                              ',
                                           '"enum": ["km", "mi", "kg", None],\n                              '), "wrong", "enum-units"),
    ("describe", SOLUTION_DESCRIBE.replace('"required": ["amount", "from_unit", "to_unit"]', '"required": ["amount", 1, "to_unit"]'), "wrong", "required"),
    ("describe", SOLUTION_DESCRIBE.replace('"type": "function",', '"type": "tool",'), "wrong", None),
    ("describe", SOLUTION_DESCRIBE.replace("\nconvert_tool\n", "\n"), "wrong", "no-value"),
    ("answer", START_ANSWER, "empty", None),
    ("answer", "", "empty", None),
    ("answer", SOLUTION_ANSWER, "correct", None),
    ("answer", START_ANSWER.replace('''            ...
''', '''            name = call["function"]["name"]
            output = functions[name](json.loads(call["function"]["arguments"]))
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
'''), "correct", None),
    ("answer", START_ANSWER.replace("            ...\n", "            pass\n"), "error", "http-400-messages"),
    ("answer", SOLUTION_ANSWER.replace('"tool_call_id": call["id"], ', ""), "error", "http-400-messages"),
    ("answer", SOLUTION_ANSWER.replace("        messages = messages + [message]\n", ""), "error", "http-400-messages"),
    ("answer", SOLUTION_ANSWER.replace('json.loads(call["function"]["arguments"])', 'call["function"]["arguments"]'), "error", "args-text"),
    ("answer", SOLUTION_ANSWER.replace('functions[call["function"]["name"]]', 'functions["convert"]'), "wrong", "wrong-result"),
    ("answer", SOLUTION_ANSWER.replace("        message = ask(url, messages, tools)\n    return", "    return"), "wrong", "not-again"),
    ("answer", SOLUTION_ANSWER.replace('    return message["content"]', '    return message'), "wrong", "return"),
    ("answer", SOLUTION_ANSWER.replace("        message = ask(url, messages, tools)\n    return",
                                       "        messages = messages + [{\"role\": \"user\", \"content\": \"Answer with the results.\"}]\n"
                                       "        message = ask(url, messages, tools)\n    return"), "wrong", "messages"),
    ("answer", SOLUTION_ANSWER.replace("        message = ask(url, messages, tools)\n    return", "    message = ask(url, messages, tools)\n    return"),
     "wrong", "requests"),
    ("answer", SOLUTION_ANSWER.replace("    if message.get(\"tool_calls\"):\n", "    if True:\n"), "error", None),
    ("answer", SOLUTION_ANSWER.replace("json.dumps(body).encode()", "json.dumps(body)"), "error", "data-text"),
]


# --- the demo: your answer and your description, with Qwen3 behind the course's server ------------------------------

def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


def learner_answer(text):
    """The learner's answer from exercise 2, when it is right; else None."""
    ex = EXERCISES["answer"]
    if text is MISSING or text is None or ex.check(text).state != "correct":
        return None
    return run_code(text).names.get("answer")


def learner_tool(text):
    """The learner's description from exercise 1 when it is right, else the course's."""
    ex = EXERCISES["describe"]
    if text is MISSING or text is None or ex.check(text).state != "correct":
        return CONVERT, False
    return run_code(text).value, True


def demo(answer_fn, tool: dict, question: str, words: dict, lang: str) -> str:
    """The learner's answer on a question, against the course's server with Qwen3 behind it, with the tool convert as
    `tool` describes it: each request with its messages, what came back, and what answer gave back."""
    running, before = None, 0
    try:
        running = serve.model_server()
        before = len(running.log)
        got, error = answer_fn(running.url, question, [copy.deepcopy(tool)], dict(FUNCTIONS)), None
    except Exception as e:   # the learner's code, against the real server; it may have run out of memory
        if serve.out_of_memory(e):
            return f"<p><em>{html.escape(words['memory'])}</em></p>"
        got, error = None, e
    log = running.log[before:] if running is not None else []
    return yours.figure(words["title"], exchange_html(log, words, got, error), alt=words["title"])
