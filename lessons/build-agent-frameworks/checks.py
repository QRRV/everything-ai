# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-agent-frameworks: the ideas behind agent frameworks. A framework gives the agent loop, tools made from
functions, memory and traces ready-made; under its names are the parts this module built by hand. The course's recorded
runs of Qwen3-0.6B (data/course-agents-qwen.json): the eight questions of build-tool-calling, once with a system prompt
of the kind a framework adds by itself, and once with the tool convert made from a function's name, its docstring and
the types of its parameters, as a framework makes it; beside the runs of build-tool-calling with the tool written by hand
and no system prompt. What they show, their pictures, the exercises (a tool made from a function, and a graph of steps
that runs the agent loop) and the demo."""
import copy
import html
import importlib.util
import json

from everything_ai import chat, paths, serve, yours
from everything_ai.check import MISSING, FunctionBox, Mistake, run_code

DOWNLOADS = {"Qwen3-0.6B": chat.FILES}   # what the demo downloads, for the line under the header
DATA = paths.ROOT / "data" / "course-agents-qwen.json"


def _lesson_checks(lesson_id: str):
    """The checks of an earlier lesson, loaded once for this module."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_frameworks",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _lesson_checks("build-tool-calling")   # the tool convert, the eight questions and their runs
QUESTIONS, CONVERT = T.QUESTIONS, T.CONVERT

# The course's own example of the instructions a framework can put before every request without showing them: not the
# text of any framework, but the kind of thing such instructions say.
FRAMEWORK_SYSTEM = (
    "You are a helpful AI agent that completes tasks for the user. You can use the tools listed below. Before you act, "
    "think about what the user needs. Use a tool only when it is really needed; when you can answer from your own "
    "knowledge, answer directly. If information is missing, ask the user. Never make up the result of a tool. After a "
    "tool returns, explain the result to the user in plain language. Be polite, clear and concise, and format your answer "
    "nicely. If a task cannot be done, say so and explain why.")

# --- exercise 1: a tool made from a function ---------------------------------------------------------------------------

START_TOOL = '''def tool_from(name, doc, params):
    properties = {}
    for param, kind in params.items():
        ...
    return {"type": "function", "function": {
        "name": name,
        "description": ...,
        "parameters": {"type": "object", "properties": properties, "required": list(params)},
    }}
'''
SOLUTION_TOOL = START_TOOL.replace('''        ...
''', '''        properties[param] = {"type": kind}
''').replace('"description": ...,', '"description": doc.strip().split("\\n")[0],')
CONVERT_DOC = '''Convert an amount from one unit to another: kilometres and miles, or kilograms and pounds.

    The amount comes back rounded to two decimals, such as "22.99 mi".
    '''
CONVERT_PARAMS = {"amount": "number", "from_unit": "string", "to_unit": "string"}


def _fn(code: str, name: str):
    return run_code(code, {}).names[name]


GENERATED = _fn(SOLUTION_TOOL, "tool_from")("convert", CONVERT_DOC, CONVERT_PARAMS)   # convert as a framework makes it

# --- exercise 2: a graph of steps ------------------------------------------------------------------------------------------

START_GRAPH = '''def run_graph(nodes, edges, state, start, limit):
    node = start
    steps = 0
    ...
    return state
'''
SOLUTION_GRAPH = START_GRAPH.replace('''    ...
''', '''    while node != "END" and steps < limit:
        state = nodes[node](state)
        node = edges[node](state)
        steps = steps + 1
''')

# --- the recorder's runs ---------------------------------------------------------------------------------------------------------

RECORD = {}
for _i, (_q, *_) in enumerate(QUESTIONS):
    RECORD[f"framework-{_i}"] = {"messages": [{"role": "system", "content": FRAMEWORK_SYSTEM}] + T.asked(_q), "tools": [CONVERT],
                                 "max_tokens": 120, "temperature": 0, "steps": 2, "seeds": 1}
    RECORD[f"generated-{_i}"] = {"messages": T.asked(_q), "tools": [GENERATED], "max_tokens": 120, "temperature": 0, "steps": 2,
                                 "seeds": 1}


def tool_results(message: dict, ask: dict | None = None) -> list:
    """For the recorder: the course's convert answers each call, as in build-tool-calling."""
    return T.tool_results(message, ask)


# --- what the runs show ------------------------------------------------------------------------------------------------------

def course() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-agent-frameworks"]


def steps_of(kind: str, i: int) -> list:
    """The steps of a run: "hand" is build-tool-calling's run with the tool written by hand and no system prompt."""
    return T.run(f"auto-{i}")["steps"] if kind == "hand" else course()[f"{kind}-{i}"]["answers"][0]["steps"]


KINDS = ("hand", "framework", "generated")


def first_call(kind: str, i: int):
    calls = steps_of(kind, i)[0]["tool_calls"]
    return calls[0] if calls else None


def findings() -> dict:
    n = len(QUESTIONS)
    out = {"questions": n}
    for kind in KINDS:
        runs = [steps_of(kind, i) for i in range(n)]
        out[kind + "_called"] = [i for i in range(n) if first_call(kind, i) is not None]
        out[kind + "_args"] = [i for i in range(n) if first_call(kind, i) is not None and T.right_call(first_call(kind, i), i)]
        out[kind + "_right"] = [i for i, s in enumerate(runs) if T.right(s[-1]["text"] or "", i)]
        out[kind + "_prompt"] = runs[0][0]["prompt_tokens"]
    out["extra"] = out["framework_prompt"] - out["hand_prompt"]
    out["shorter"] = out["hand_prompt"] - out["generated_prompt"]
    allowed = CONVERT["function"]["parameters"]["properties"]["from_unit"]["enum"]
    units = []   # the units the model wrote for the made tool that convert does not know, in the order it wrote them
    for i in out["generated_called"]:
        args = T.arguments_of(first_call("generated", i))
        for u in (args.get("from_unit"), args.get("to_unit")):
            if u not in allowed and u not in units:
                units.append(u)
    out["units"] = units
    return out


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


# --- the agent loop as a graph: two nodes and their edges ----------------------------------------------------------------------

class Named(dict):
    """A dictionary of a case of the check, such as a graph's nodes, which shows as its name in the check's messages."""

    def __init__(self, name: str, items: dict):
        super().__init__(items)
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return self


def _post(url: str, messages: list, tools: list) -> dict:
    import urllib.request
    body = {"model": serve.MODEL, "messages": messages, "tools": tools, "max_tokens": 200, "temperature": 0}
    request = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def model_node(state: dict) -> dict:
    """Send the messages and the tools to the model, and add its answer; the trace gets the request's prompt tokens."""
    state = dict(state)
    reply = _post(state["url"], state["messages"], state["tools"])
    message = reply["choices"][0]["message"]
    state["messages"] = state["messages"] + [message]
    state["trace"] = state["trace"] + [("model", reply["usage"]["prompt_tokens"],
                                        [f'{c["function"]["name"]}({c["function"]["arguments"]})' for c in message.get("tool_calls") or []]
                                        or (message.get("content") or "").strip())]
    return state


def tools_node(state: dict) -> dict:
    """Run each call of the last answer, and add its result with the role tool."""
    state = dict(state)
    results = T.tool_results(state["messages"][-1])
    state["messages"] = state["messages"] + results
    state["trace"] = state["trace"] + [("tools", None, [r["content"] for r in results])]
    return state


def after_model(state: dict) -> str:
    return "tools" if state["messages"][-1].get("tool_calls") else "END"


def after_tools(state: dict) -> str:
    return "model"


AGENT_NODES = Named("agent_nodes", {"model": model_node, "tools": tools_node})
AGENT_EDGES = Named("agent_edges", {"model": after_model, "tools": after_tools})


# --- the exercises' cases: graphs that run without a model ----------------------------------------------------------------------

def _scripted_model(state: dict) -> dict:
    """A stand-in for the model node: the next answer of a fixed script."""
    state = dict(state)
    k = len([m for m in state["messages"] if m["role"] == "assistant"])
    answer = state["script"][min(k, len(state["script"]) - 1)]   # past the end, the last answer again: a loop runs on
    state["messages"] = state["messages"] + [answer]
    state["path"] = state["path"] + ["model"]
    return state


def _scripted_tools(state: dict) -> dict:
    state = dict(state)
    state["messages"] = state["messages"] + T.tool_results(state["messages"][-1])
    state["path"] = state["path"] + ["tools"]
    return state


def _call_msg(amount, a, b) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
        "name": "convert", "arguments": json.dumps({"amount": amount, "from_unit": a, "to_unit": b})}}]}


SCRIPTED_NODES = Named("scripted_nodes", {"model": _scripted_model, "tools": _scripted_tools})
SCRIPTED_EDGES = Named("scripted_edges", {"model": after_model, "tools": after_tools})


def _scripted_state(name: str, script: list) -> dict:
    return Named(name, {"messages": T.asked(QUESTIONS[0][0]), "script": script, "path": []})


ONE_CALL = _scripted_state("one_call_state", [_call_msg(37, "km", "mi"), {"role": "assistant", "content": "37 km is 22.99 miles."}])
NO_CALL = _scripted_state("no_call_state", [{"role": "assistant", "content": "About 23 miles."}])
EVERY_TIME = _scripted_state("calls_every_time_state", [_call_msg(37, "km", "mi")] * 10)


def _add(state):
    return state + 1


def _double(state):
    return state * 2


def _to_double(state):
    return "double"


def _until_big(state):
    return "END" if state > 20 else "add"


COUNT_NODES = Named("count_nodes", {"add": _add, "double": _double})
COUNT_EDGES = Named("count_edges", {"add": _to_double, "double": _until_big})

GRAPH_CASES = [
    (SCRIPTED_NODES, SCRIPTED_EDGES, ONE_CALL, "model", 6),
    (SCRIPTED_NODES, SCRIPTED_EDGES, NO_CALL, "model", 6),
    (SCRIPTED_NODES, SCRIPTED_EDGES, EVERY_TIME, "model", 4),
    (COUNT_NODES, COUNT_EDGES, 1, "add", 10),
    (COUNT_NODES, COUNT_EDGES, 1, "add", 3),
]
_GRAPH_EDGE_FIRST = SOLUTION_GRAPH.replace('''        state = nodes[node](state)
        node = edges[node](state)''', '''        node = edges[node](state)
        state = nodes[node](state)''')
_GRAPH_LE = SOLUTION_GRAPH.replace("steps < limit", "steps <= limit")


def _end_key(e: BaseException) -> bool:
    return isinstance(e, KeyError) and e.args == ("END",)


def _graph_shown(v) -> str:
    """A state of the agent loop by its path, which says the same in both languages; anything else as it is."""
    if isinstance(v, dict) and "path" in v:
        return "{'path': " + repr(v["path"]) + ", …}"
    return repr(v)


def _json(v, indent=None) -> str:
    """A tool definition as JSON, or as Python when it is not JSON, such as a dictionary with a key that is no text."""
    try:
        return json.dumps(v, indent=indent, ensure_ascii=False, default=repr)
    except (TypeError, ValueError):
        return repr(v)


TOOL_CASES = [
    ("convert", CONVERT_DOC, CONVERT_PARAMS),
    ("weather", "The weather in a city today.", {"city": "string"}),
    ("pantry", "  What is in the pantry, in grams.\nNo arguments.  ", {}),
    ("add_to_cart", "Put a product in the cart.\n\nThe shop is shop.example.", {"product": "string", "quantity": "integer"}),
]
_TOOL_WHOLE_DOC = SOLUTION_TOOL.replace('doc.strip().split("\\n")[0]', "doc.strip()")
_TOOL_NO_STRIP = SOLUTION_TOOL.replace('doc.strip().split("\\n")[0]', 'doc.split("\\n")[0]')
_TOOL_ALL_STRING = SOLUTION_TOOL.replace('properties[param] = {"type": kind}', 'properties[param] = {"type": "string"}')
_TOOL_BARE = SOLUTION_TOOL.replace('properties[param] = {"type": kind}', 'properties[param] = kind')
_TOOL_NO_QUOTES = SOLUTION_TOOL.replace('properties[param] = {"type": kind}', 'properties[param] = {type: kind}')


# --- the pictures --------------------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui)"
_CELL = "padding:0.18rem 0.5rem;border-bottom:1px solid var(--ea-line);vertical-align:top;font-size:13px"
_HEAD = f"{_CELL};color:var(--ea-ink-2);font-weight:500;text-align:left;font-family:var(--ea-font-ui)"


def _code(text: str, color: str = "var(--ea-ink)") -> str:
    return (f'<span style="font-family:var(--ea-font-code);font-size:12px;color:{color};white-space:pre-wrap;'
            f'overflow-wrap:anywhere">{html.escape(text)}</span>')


def _label(text: str) -> str:
    return f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2);margin:0.4rem 0 0.15rem">{html.escape(text)}</div>'


def _pre(inner: str) -> str:
    return f'<div style="background:var(--ea-inset);border-radius:4px;padding:0.35rem 0.55rem;margin:0.1rem 0">{inner}</div>'


_GRAPH_WORDS = {
    "en": {"title": "The agent loop as a graph: two nodes, and an edge after each that says where to go next",
           "model": "model", "tools": "tools", "end": "END", "calls": "the answer has tool calls", "none": "no tool calls",
           "back": "always", "start": "start", "model_does": ["sends the messages and the tools,", "adds the answer"],
           "tools_does": ["runs each call,", "adds its result"]},
    "nl": {"title": "De agent loop als graaf: twee nodes, en na elke node een edge die zegt waar het heen gaat",
           "model": "model", "tools": "tools", "end": "END", "calls": "het antwoord heeft tool calls", "none": "geen tool calls",
           "back": "altijd", "start": "start", "model_does": ["stuurt de berichten en de tools,", "voegt het antwoord toe"],
           "tools_does": ["voert elke call uit,", "voegt het resultaat toe"]},
}


def _svg_text(x: float, y: float, text: str, size: int = 11, color: str = "var(--ea-ink-2)", anchor: str = "start",
              font: str = "var(--ea-font-ui)", weight: int = 400) -> str:
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" '
            f'style="fill:{color};font-family:{font}">{html.escape(text)}</text>')


def idea_picture(lang: str) -> str:
    w = _GRAPH_WORDS[lang]
    parts = ['<defs><marker id="fw-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ea-ink-2)"/></marker></defs>']
    for x, wd, name, does in ((20, 260, w["model"], w["model_does"]), (480, 200, w["tools"], w["tools_does"])):
        cx = x + wd / 2
        parts.append(f'<rect x="{x}" y="60" width="{wd}" height="66" rx="6" style="fill:var(--ea-inset);stroke:var(--ea-line)"/>')
        parts.append(_svg_text(cx, 80, name, 14, "var(--ea-ink)", "middle", "var(--ea-font-code)", 600))
        parts += [_svg_text(cx, 98 + 14 * k, line, anchor="middle") for k, line in enumerate(does)]
    parts.append('<rect x="100" y="192" width="100" height="34" rx="17" style="fill:none;stroke:var(--ea-ink-2)"/>')
    parts.append(_svg_text(150, 214, w["end"], 13, "var(--ea-ink)", "middle", "var(--ea-font-code)"))
    line = '<path d="{d}" style="fill:none;stroke:var(--ea-ink-2)" stroke-width="1.3" marker-end="url(#fw-arrow)"/>'
    parts.append(line.format(d="M 150 20 L 150 58"))
    parts.append(_svg_text(158, 40, w["start"]))
    parts.append(line.format(d="M 280 82 L 478 82"))
    parts.append(_svg_text(380, 74, w["calls"], anchor="middle"))
    parts.append(line.format(d="M 580 126 C 580 172, 300 172, 262 128"))
    parts.append(_svg_text(430, 180, w["back"], anchor="middle"))
    parts.append(line.format(d="M 150 126 L 150 190"))
    parts.append(_svg_text(158, 162, w["none"]))
    svg = (f'<svg viewBox="0 0 700 236" style="width:100%;max-width:700px;height:auto" role="img" '
           f'aria-label="{html.escape(w["title"])}">' + "".join(parts) + "</svg>")
    return yours.figure(w["title"], svg, alt=w["title"])


_TABLE_WORDS = {
    "en": {"title": "Qwen3-0.6B on the {n} questions of Tools as JSON: the same tool three ways", "what": "",
           "hand": "tool written by hand, no system prompt", "framework": "with a framework's kind of system prompt",
           "generated": "tool made from the function", "called": "called convert", "args": "with the right arguments",
           "right": "right answers", "prompt": "prompt tokens of the first request", "of": "{k} of {n}"},
    "nl": {"title": "Qwen3-0.6B op de {n} vragen van Tools als JSON: dezelfde tool op drie manieren", "what": "",
           "hand": "tool met de hand geschreven, geen system prompt", "framework": "met het soort system prompt van een framework",
           "generated": "tool gemaakt van de functie", "called": "riep convert aan", "args": "met de goede argumenten",
           "right": "goede antwoorden", "prompt": "tokens aan prompt van het eerste request", "of": "{k} van de {n}"},
}


def table_picture(lang: str) -> str:
    w, f = _TABLE_WORDS[lang], findings()
    n = f["questions"]
    head = f'<th style="{_HEAD}"></th>' + "".join(f'<th style="{_HEAD}">{html.escape(w[k])}</th>' for k in KINDS)
    rows = []
    for row in ("called", "args", "right", "prompt"):
        cells = "".join(f'<td style="{_CELL};{_UI}">'
                        + (number(f[k + "_prompt"], lang) if row == "prompt" else w["of"].format(k=len(f[f"{k}_{row}"]), n=n))
                        + "</td>" for k in KINDS)
        rows.append(f'<tr><td style="{_CELL};{_UI};color:var(--ea-ink-2)">{html.escape(w[row])}</td>{cells}</tr>')
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><thead><tr>{head}</tr></thead>'
             f'<tbody>{"".join(rows)}</tbody></table></div>')
    return yours.figure(w["title"].format(n=n), table, alt=w["title"].format(n=n))


def _steps_html(steps: list, w: dict) -> str:
    out = []
    for st in steps:
        for c in st["tool_calls"][:1]:
            out.append(_code(f'{c["function"]["name"]}({c["function"]["arguments"]})', "var(--ea-pred-text)"))
            out.append(f'<div style="margin-left:0.8rem">{_code(st["results"][0], "var(--ea-data-text)")}</div>')
        if not st["tool_calls"]:
            out.append(f'<div style="{_UI};font-size:13px">{html.escape((st["text"] or "").strip())}</div>')
    return "".join(out)


_WORKED_WORDS = {
    "en": {"title": "One question, the same tool three ways"},
    "nl": {"title": "Eén vraag, dezelfde tool op drie manieren"},
}


def worked_run() -> tuple:
    """The question of the worked example: the first that the tool written by hand answered right, and that the made
    tool was called for with arguments convert does not take."""
    f = findings()
    return (next(i for i in f["hand_right"] if i in f["generated_called"] and i not in f["generated_args"]),)


def worked_picture(i: int, lang: str) -> str:
    w, t = _WORKED_WORDS[lang], _TABLE_WORDS[lang]
    body = f'<div style="{_UI};font-size:13px;margin-bottom:0.2rem">{html.escape(QUESTIONS[i][0])}</div>'
    for kind in KINDS:
        body += _label(t[kind]) + _pre(_steps_html(steps_of(kind, i), w))
    return yours.figure(w["title"], body, alt=w["title"])


def _or(items: list, lang: str) -> str:
    """`a`, `b` and `c`, in the lesson's language."""
    words = [f"`{x}`" for x in items]
    if len(words) < 2:
        return "".join(words)
    return ", ".join(words[:-1]) + (" and " if lang == "en" else " en ") + words[-1]


def _times(k: int, of: int, lang: str) -> str:
    """How many of the calls had the right arguments: each time, never, or k times."""
    if k == of:
        return "each time" if lang == "en" else "elke keer"
    if k == 0:
        return "never" if lang == "en" else "nooit"
    return f"{k} times" if lang == "en" else f"{k} keer"


def _answers(k: int, lang: str) -> str:
    if lang == "en":
        return "no answer was right" if k == 0 else ("1 answer was right" if k == 1 else f"{k} answers were right")
    return "was geen antwoord goed" if k == 0 else ("was 1 antwoord goed" if k == 1 else f"waren {k} antwoorden goed")


def measured_text(lang: str) -> str:
    f = findings()
    n = f["questions"]
    values = dict(n=n, hc=len(f["hand_called"]), ha=_times(len(f["hand_args"]), len(f["hand_called"]), lang),
                  hr=_answers(len(f["hand_right"]), lang), fc=len(f["framework_called"]),
                  fr=_answers(len(f["framework_right"]), lang), extra=number(f["extra"], lang),
                  fp=number(f["framework_prompt"], lang), hp=number(f["hand_prompt"], lang), gc=len(f["generated_called"]),
                  ga=_times(len(f["generated_args"]), len(f["generated_called"]), lang),
                  gr=_answers(len(f["generated_right"]), lang), shorter=number(f["shorter"], lang),
                  units=_or(f["units"], lang))
    if lang == "en":
        return ("Written by hand, `convert` was called in {hc} of the {n} questions, {ha} with the right arguments, and "
                "{hr}. With the framework's kind of system prompt, every request was {extra} tokens "
                "longer ({fp} instead of {hp} for the first question), Qwen3 called the tool in {fc} questions, and {fr}: "
                "the instructions made no answer better, and they cost tokens in every request. Made "
                "from the function, the first request of each question was {shorter} tokens shorter, and Qwen3 called the tool in {gc} questions, "
                "{ga} with the right arguments, and {gr}. The made tool says only that `from_unit` and "
                "`to_unit` are texts, not which units `convert` takes, so Qwen3 wrote {units}, and `convert` answered "
                "with an error each time.\n\nMany frameworks let you add a description and the allowed values to each "
                "parameter; then the tool they make is as good as the one written by hand. You only find that out by "
                "looking at the definitions they send.").format(**values)
    return ("Met de hand geschreven werd `convert` aangeroepen bij {hc} van de {n} vragen, {ha} met de goede "
            "argumenten, en {hr}. Met het soort system prompt van een framework was elk request {extra} "
            "tokens langer ({fp} in plaats van {hp} bij de eerste vraag), riep Qwen3 de tool aan bij {fc} vragen, en "
            "{fr}: de instructies maakten geen antwoord beter, en ze kostten tokens in elk request. Gemaakt "
            "van de functie was het eerste request van elke vraag {shorter} tokens korter, en riep Qwen3 de tool aan bij {gc} vragen, {ga} met "
            "de goede argumenten, en {gr}. De gemaakte tool zegt alleen dat `from_unit` en `to_unit` "
            "teksten zijn, niet welke eenheden `convert` kent, dus schreef Qwen3 {units}, en gaf `convert` elke keer een "
            "fout terug.\n\nVeel frameworks laten je bij elke parameter een beschrijving en de toegestane waarden zetten; "
            "dan is de tool die ze maken net zo goed als die met de hand. Dat zie je pas als je kijkt naar de definitie die "
            "het framework verstuurt.").format(**values)


def worked_text(lang: str) -> str:
    (i,) = worked_run()
    q, hand, made = QUESTIONS[i][0], first_call("hand", i), first_call("generated", i)
    args = T.arguments_of(made)
    wrong = next(u for u in (args["from_unit"], args["to_unit"]) if u in findings()["units"])
    values = dict(q=q, call=hand["function"]["arguments"], result=steps_of("hand", i)[0]["results"][0],
                  made=made["function"]["arguments"], error=steps_of("generated", i)[0]["results"][0], wrong=wrong)
    if lang == "en":
        return ("### Worked example: one question, three ways\n\nTake the question *{q}* With the tool written by hand, "
                "Qwen3 calls `convert` with `{call}`, gets `{result}` back, and answers with it. With the framework's kind "
                "of system prompt, it does the same: the extra instructions change nothing here, they only make the request "
                "longer. With the tool made from the function, it calls `convert` with `{made}`. The unit `{wrong}` is not "
                "one `convert` takes, so the result is `{error}`, and instead of trying again with another unit, Qwen3 asks "
                "the user which unit was meant. Nothing in the made tool told it the units.").format(**values)
    return ("### Uitgewerkt voorbeeld: één vraag, op drie manieren\n\nNeem de vraag *{q}* Met de tool die met de hand "
            "geschreven is, roept Qwen3 `convert` aan met `{call}`, krijgt `{result}` terug, en antwoordt daarmee. Met het "
            "soort system prompt van een framework doet het hetzelfde: de extra instructies veranderen hier niets, ze maken "
            "het request alleen langer. Met de tool die van de functie gemaakt is, roept het `convert` aan met `{made}`. De "
            "eenheid `{wrong}` kent `convert` niet, dus is het resultaat `{error}`, en in plaats van het opnieuw te proberen "
            "met een andere eenheid, vraagt Qwen3 de gebruiker welke eenheid bedoeld was. Niets in de gemaakte tool vertelde "
            "het welke eenheden er zijn.").format(**values)


# --- the pictures of the exercises ---------------------------------------------------------------------------------------------

def _learner(seen, name: str):
    run_ = seen.run
    fn = run_.names.get(name) if run_ is not None and run_.error is None else None
    return fn if callable(fn) else None


_STOPPED = object()


def _call(fn, *args):
    try:
        return fn(*copy.deepcopy(args))
    except Exception:   # noqa: BLE001 - the learner's code
        return _STOPPED


_TOOL_WORDS = {
    "en": {"title": "Your tool_from on convert, beside the definition written by hand in Tools as JSON",
           "yours": "the tool your tool_from makes", "hand": "written by hand: the same, and more",
           "error": "your function stopped with an error", "fn": "the function, as a framework sees it"},
    "nl": {"title": "Je tool_from op convert, naast de definitie die in Tools als JSON met de hand geschreven is",
           "yours": "de tool die je tool_from maakt", "hand": "met de hand geschreven: hetzelfde, en meer",
           "error": "je functie stopte met een fout", "fn": "de functie, zoals een framework hem ziet"},
}


def tool_picture(seen) -> str:
    fn = _learner(seen, "tool_from")
    if fn is None:
        return ""
    w = _TOOL_WORDS[seen.lang]
    got = _call(fn, *TOOL_CASES[0])
    ok = got == GENERATED
    signature = ('def convert(amount: float, from_unit: str, to_unit: str) -> str:\n    """' + CONVERT_DOC.rstrip() + '"""')
    shown = w["error"] if got is _STOPPED else _json(got, indent=2)
    body = (_label(w["fn"]) + _pre(_code(signature, "var(--ea-ink-2)"))
            + f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(16rem,1fr));gap:0.5rem">'
            f'<div>{_label(w["yours"])}{_pre(_code(shown, "var(--ea-ink)" if ok else "var(--ea-loss-text)"))}</div>'
            f'<div>{_label(w["hand"])}{_pre(_code(json.dumps(CONVERT, indent=2), "var(--ea-ink-2)"))}</div></div>')
    return yours.figure(w["title"], body, alt=w["title"])


_PATH_WORDS = {
    "en": {"title": "The way your run_graph goes through the agent loop, with a model that calls convert once",
           "yours": "your run_graph", "course": "the course's run_graph", "messages": "the messages at the end",
           "error": "your function stopped with an error", "none": "(no node)"},
    "nl": {"title": "De weg die je run_graph door de agent loop neemt, met een model dat convert één keer aanroept",
           "yours": "jouw run_graph", "course": "de run_graph van de course", "messages": "de berichten aan het eind",
           "error": "je functie stopte met een fout", "none": "(geen node)"},
}


def path_picture(seen) -> str:
    fn = _learner(seen, "run_graph")
    if fn is None:
        return ""
    w = _PATH_WORDS[seen.lang]
    got = _call(fn, *GRAPH_CASES[0])
    want = _fn(SOLUTION_GRAPH, "run_graph")(*copy.deepcopy(GRAPH_CASES[0]))

    def path(v):
        return " → ".join((v.get("path") or [w["none"]]) + ["END"]) if isinstance(v, dict) and "path" in v else None
    ok = got == want
    mine = w["error"] if got is _STOPPED else path(got) or repr(got)
    roles = " · ".join(m["role"] for m in want["messages"])
    body = (_label(w["yours"]) + _pre(_code(mine, "var(--ea-ink)" if ok else "var(--ea-loss-text)"))
            + ("" if ok else _label(w["course"]) + _pre(_code(path(want))))
            + _label(w["messages"]) + _pre(_code(roles, "var(--ea-ink-2)")))
    return yours.figure(w["title"], body, alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------------------

_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}


class NamedBox(FunctionBox):
    """A FunctionBox whose messages show each case's values as Python shows them, a graph or state by its name."""

    def _call_text(self, case) -> str:
        return f"{self.function}({', '.join(repr(a) for a in case)})"


def _tool_shown(v) -> str:
    return _json(v) if isinstance(v, dict) else repr(v)


EXERCISES = {
    "tool-from": NamedBox(
        name="tool-from",
        picture=tool_picture,
        function="tool_from",
        start=START_TOOL,
        solution=SOLUTION_TOOL,
        visible=TOOL_CASES[:1],
        hidden=TOOL_CASES[1:],
        show=_tool_shown,
        mistakes=[
            Mistake("bare", wrong=_TOOL_BARE,
                    en="With `{call}`, each parameter is only its type, such as `\"amount\": \"number\"`. In a JSON schema, a parameter is a dictionary: `{{\"type\": \"number\"}}`.",
                    nl="Met `{call}` is elke parameter alleen zijn type, zoals `\"amount\": \"number\"`. In een JSON schema is een parameter een dictionary: `{{\"type\": \"number\"}}`."),
            Mistake("all-string", wrong=_TOOL_ALL_STRING,
                    en="With `{call}`, every parameter has the type `\"string\"`. Use the type `params` gives for it: `amount` is a number.",
                    nl="Met `{call}` heeft elke parameter het type `\"string\"`. Gebruik het type dat `params` ervoor geeft: `amount` is een getal."),
            Mistake("whole-doc", wrong=_TOOL_WHOLE_DOC,
                    en="With `{call}`, the description is the whole docstring. Here it is only the first line: `.split(\"\\n\")[0]` after `.strip()`.",
                    nl="Met `{call}` is de description de hele docstring. Hier is het alleen de eerste regel: `.split(\"\\n\")[0]` na `.strip()`."),
            Mistake("no-strip", wrong=_TOOL_NO_STRIP,
                    en="With `{call}`, the description keeps the spaces at the start of the docstring. Take them off with `.strip()` before you split.",
                    nl="Met `{call}` houdt de description de spaties aan het begin van de docstring. Haal ze eraf met `.strip()` voordat je splitst."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "Your function makes no properties yet, and no description. Give each parameter its type, and take the first line of the docstring as the description.",
                      "nl": "Je functie maakt nog geen properties, en geen description. Geef elke parameter zijn type, en neem de eerste regel van de docstring als description."},
            "wrong": {"en": "With `{call}`, your function gives back {got}, and it should be {want}.",
                      "nl": "Met `{call}` geeft je functie {got} terug, en het zou {want} moeten zijn."},
            "almost": {"en": "Right for convert. But with `{call}`, your function gives back {got}, and it should be {want}.",
                       "nl": "Goed voor convert. Maar met `{call}` geeft je functie {got} terug, en het zou {want} moeten zijn."},
            "correct": {"en": "Your `tool_from` makes a tool definition from what a function already has: its name, the first line of its docstring and the types of its parameters, as a framework does. Right for all {n} cases of the check.",
                        "nl": "Je `tool_from` maakt een tool definition van wat een functie al heeft: zijn naam, de eerste regel van zijn docstring en de types van zijn parameters, zoals een framework doet. Goed bij alle {n} gevallen van de check."},
        },
    ),
    "graph": NamedBox(
        name="graph",
        picture=path_picture,
        function="run_graph",
        start=START_GRAPH,
        solution=SOLUTION_GRAPH,
        visible=GRAPH_CASES[:1],
        hidden=GRAPH_CASES[1:],
        show=_graph_shown,
        mistakes=[
            Mistake("edge-first", error=_end_key,
                    en="With `{call}`, your graph looks for a node named `\"END\"`. Either it picks the next node before the node has run, and then runs that one, also when it is `\"END\"`; or the loop goes on after `\"END\"`, with `or` where `and` belongs. Run the node first, `state = nodes[node](state)`, then ask its edge where to go, and go on only while the node is not `\"END\"` and `steps < limit`.",
                    nl="Met `{call}` zoekt je graaf een node met de naam `\"END\"`. Of hij kiest de volgende node voordat de node gedraaid heeft, en draait die dan, ook als dat `\"END\"` is; of de loop gaat door na `\"END\"`, met `or` waar `and` hoort. Draai eerst de node, `state = nodes[node](state)`, vraag daarna zijn edge waar het heen gaat, en ga alleen door zolang de node niet `\"END\"` is en `steps < limit`."),
            Mistake("le", wrong=_GRAPH_LE,
                    en="With `{call}`, your graph runs one node more than `limit`: {got}. Go on only while `steps < limit`.",
                    nl="Met `{call}` draait je graaf één node meer dan `limit`: {got}. Ga alleen door zolang `steps < limit`."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "Your function gives back the state as it was. While the node is not `\"END\"` and the limit is not reached, run the node, follow its edge to the next node, and count the step.",
                      "nl": "Je functie geeft de state terug zoals hij was. Zolang de node niet `\"END\"` is en de grens niet bereikt is: draai de node, volg zijn edge naar de volgende node, en tel de stap."},
            "slow": {"en": "Your function is still not done after a few seconds. Does the loop stop at `\"END\"`, and once `steps` reaches `limit`? Add 1 to `steps` in every round.",
                     "nl": "Je functie is na een paar seconden nog niet klaar. Stopt de loop bij `\"END\"`, en zodra `steps` bij `limit` is? Tel in elke ronde 1 op bij `steps`."},
            "wrong": {"en": "With `{call}`, your function gives back {got}, and it should be {want}.",
                      "nl": "Met `{call}` geeft je functie {got} terug, en het zou {want} moeten zijn."},
            "almost": {"en": "Right for one call. But with `{call}`, your function gives back {got}, and it should be {want}.",
                       "nl": "Goed voor één call. Maar met `{call}` geeft je functie {got} terug, en het zou {want} moeten zijn."},
            "correct": {"en": "Your `run_graph` runs any graph of steps: a node changes the state, its edge picks the next node, until `\"END\"` or the limit. With the nodes model and tools, it is the agent loop. Right for all {n} cases of the check.",
                        "nl": "Je `run_graph` draait elke graaf van stappen: een node verandert de state, zijn edge kiest de volgende node, tot `\"END\"` of de grens. Met de nodes model en tools is het de agent loop. Goed bij alle {n} gevallen van de check."},
        },
    ),
}

REQUIRED = ["tool-from", "graph", "look"]

CASES = [
    ("tool-from", START_TOOL, "empty", None),
    ("tool-from", "", "empty", None),
    ("tool-from", SOLUTION_TOOL, "correct", None),
    ("tool-from", _TOOL_BARE, "wrong", "bare"),
    ("tool-from", _TOOL_ALL_STRING, "wrong", "all-string"),
    ("tool-from", _TOOL_WHOLE_DOC, "wrong", "whole-doc"),
    ("tool-from", _TOOL_NO_STRIP, "wrong", "no-strip"),
    ("tool-from", _TOOL_NO_QUOTES, "wrong", None),
    ("graph", START_GRAPH, "empty", None),
    ("graph", SOLUTION_GRAPH, "correct", None),
    ("graph", _GRAPH_EDGE_FIRST, "error", "edge-first"),
    ("graph", _GRAPH_LE, "wrong", "le"),
]


# --- the demo: your graph runs the agent loop on Qwen3, with your tool ----------------------------------------------------------

DEMO_LIMIT = 6


def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


def learner_fn(key: str, text):
    ex = EXERCISES[key]
    if text is MISSING or text is None or ex.check(text).state != "correct":
        return None
    return run_code(text, {}).names.get(ex.function)


def run_demo(graph_fn, tool_fn, question: str, words: dict) -> str:
    """The learner's run_graph on the agent loop's two nodes, with convert made by the learner's tool_from: each node it
    visited, as the trace shows it."""
    try:
        tool = tool_fn("convert", CONVERT_DOC, CONVERT_PARAMS)
        running = serve.model_server()
        state = {"url": running.url, "messages": T.asked(question), "tools": [tool], "trace": []}
        state = graph_fn(AGENT_NODES, AGENT_EDGES, state, "model", DEMO_LIMIT)
    except Exception as e:   # the real server, or the learner's functions
        if serve.out_of_memory(e):
            return f"<p><em>{html.escape(words['memory'])}</em></p>"
        return f"<p><em>{html.escape(words['stopped'].format(e=f'{type(e).__name__}: {e}'))}</em></p>"
    rows = []
    for k, (node, tokens, what) in enumerate(state["trace"], 1):
        shown = "\n".join(what) if isinstance(what, list) else what
        rows.append(f'<tr><td style="{_CELL};{_UI}">{k}</td><td style="{_CELL}">{_code(node, "var(--ea-pred-text)")}</td>'
                    f'<td style="{_CELL};{_UI}">{"" if tokens is None else number(tokens, words["lang"])}</td>'
                    f'<td style="{_CELL}">{_code(shown, "var(--ea-data-text)" if node == "tools" else "var(--ea-ink)")}</td></tr>')
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><thead><tr>'
             + "".join(f'<th style="{_HEAD}">{h}</th>' for h in words["head"])
             + f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    return yours.figure(words["title"], table, alt=words["title"])
