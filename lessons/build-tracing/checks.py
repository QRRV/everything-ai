# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-tracing: one request of the kitchen agent as a trace of spans. The course's recorded runs of
Qwen3-0.6B with ReAct (build-agent-loop, data/course-agents-qwen.json), each written as a trace: the request span, and in
it a span for each call to the model, with its seconds and tokens, and a span for each tool call, with what went in and
what came out. What the twelve traces show, three functions to write (the slowest span, the tokens of a request, the
ingredients the agent never looked up), the pictures."""
import functools
import html
import importlib.util
import json

from everything_ai import paths, yours
from everything_ai.check import FunctionBox, Mistake


def _lesson_checks(lesson: str):
    spec = importlib.util.spec_from_file_location(lesson.replace("-", "_") + "_checks_for_tracing",
                                                  paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _lesson_checks("build-agent-loop")   # the kitchen agent with ReAct, its twelve questions and their recorded runs
DATA = A.DATA
QUESTIONS = A.QUESTIONS                  # (question, dish) for each run
recipe, right_answer, final_of = A.W.recipe, A.right_answer, A.final_of
EXAMPLE = 10                             # the run of the worked example and the exercises: pasta with pesto


class Named(list):
    """A list of the lesson's, written in the check's messages by its name, such as TRACE."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self.name = name

    def __repr__(self):
        return self.name


# --- the traces ---------------------------------------------------------------------------------------------------------

@functools.cache
def _trace(i: int) -> str:
    """Run i of build-agent-loop as a trace, as JSON so that every caller gets its own copy. The request is span 0; each
    call to the model is a span from where the one before ended, as long as the call took; each tool call is a span of
    no length at the moment the call before it ended: the recording timed only the calls to the model, and the
    kitchen's tools are Python functions that answer in well under a millisecond (tests/test_build_tracing.py times
    them)."""
    steps = A.react_steps(i)
    spans = [{"id": 0, "parent": None, "name": "request", "start": 0.0, "end": 0.0, "input": QUESTIONS[i][0], "output": ""}]
    t = 0.0
    for step in steps:
        end = round(t + step["seconds"], 2)
        spans.append({"id": len(spans), "parent": 0, "name": "model", "start": t, "end": end,
                      "prompt_tokens": step["prompt_tokens"], "output_tokens": step["tokens"], "output": (step["text"] or "").strip()})
        t = end
        for call, result in zip(step["tool_calls"], step["results"]):
            spans.append({"id": len(spans), "parent": 0, "name": call["function"]["name"], "start": t, "end": t,
                          "input": json.loads(call["function"]["arguments"]), "output": result})
    spans[0]["end"] = t
    spans[0]["output"] = (final_of(steps) or "").strip()
    return json.dumps(spans)


def trace(i: int) -> list:
    return json.loads(_trace(i))


TRACE = Named(trace(EXAMPLE), "TRACE")
TRACES = Named([trace(i) for i in range(len(QUESTIONS))], "TRACES")


def seconds(span: dict) -> float:
    return round(span["end"] - span["start"], 2)


def slowest(spans):
    best = spans[1]
    for span in spans:
        if span["parent"] == 0:
            if span["end"] - span["start"] > best["end"] - best["start"]:
                best = span
    return best


def tokens_of(spans):
    total = 0
    for span in spans:
        if span["name"] == "model":
            total = total + span["prompt_tokens"] + span["output_tokens"]
    return total


def not_looked_up(spans, needed):
    asked = []
    for span in spans:
        if span["name"] == "pantry":
            asked.append(span["input"]["item"])
    missing = []
    for item in needed:
        if item not in asked:
            missing.append(item)
    return missing


def needed(i: int) -> list:
    """The ingredients of run i's dish, in the order of its recipe."""
    return list(recipe(QUESTIONS[i][1]))


def findings() -> dict:
    """The numbers the lesson gives about the twelve traces, and about the trace of the worked example."""
    n = len(QUESTIONS)
    models = [(i, s) for i in range(n) for s in trace(i) if s["name"] == "model"]
    tools = [(i, s) for i in range(n) for s in trace(i) if s["parent"] == 0 and s["name"] != "model"]
    slow_i, slow = max(models, key=lambda p: seconds(p[1]))
    ex = trace(EXAMPLE)
    ex_models = [s for s in ex if s["name"] == "model"]
    missed = [not_looked_up(trace(i), needed(i)) for i in range(n)]
    return {
        "n": n,
        "model_calls": len(models), "tool_calls": len(tools),
        "model_seconds": round(sum(seconds(s) for _, s in models), 1),
        "tool_seconds": round(sum(seconds(s) for _, s in tools), 1),
        "slowest_run": slow_i, "slowest_seconds": seconds(slow), "slowest_output": slow["output_tokens"],
        "slowest_is_model": all(slowest(trace(i))["name"] == "model" for i in range(n)),
        "longest_run": max(range(n), key=lambda i: trace(i)[0]["end"]),
        "longest_seconds": max(trace(i)[0]["end"] for i in range(n)),
        "first_prompt": (min(trace(i)[1]["prompt_tokens"] for i in range(n)), max(trace(i)[1]["prompt_tokens"] for i in range(n))),
        "tokens": [tokens_of(trace(i)) for i in range(n)],
        "missed": missed,
        "right": [right_answer(final_of(A.react_steps(i)), QUESTIONS[i][1]) for i in range(n)],
        "all_looked": [i for i in range(n) if not missed[i]],
        "never_pantry": [i for i in range(n) if not any(s["name"] == "pantry" for s in trace(i))],
        # the worked example
        "ex_seconds": ex[0]["end"], "ex_calls": len(ex_models),
        "ex_call_seconds": [seconds(s) for s in ex_models],
        "ex_prompts": [s["prompt_tokens"] for s in ex_models], "ex_outputs": [s["output_tokens"] for s in ex_models],
        "ex_tokens": tokens_of(ex), "ex_prompt_sum": sum(s["prompt_tokens"] for s in ex_models),
        "ex_tools": [(s["name"], s["input"]) for s in ex if s["parent"] == 0 and s["name"] != "model"],
        "ex_missed": not_looked_up(ex, needed(EXAMPLE)),
        "ex_question_words": len(QUESTIONS[EXAMPLE][0].split()),
        "per_output_token": sum(seconds(s) for _, s in models) / sum(s["output_tokens"] for _, s in models),
    }


# --- pictures ---------------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui)"
_MONO = "font-family:var(--ea-font-code)"
_WORDS = {
    "en": {"request": "request", "model": "model", "s": "s", "tokens": "{p} + {o} tokens",
           "wf_title": "The trace of run {i}: {q}",
           "wf_alt": "A waterfall: a row for each span of the request, in the order they started. Each row has the span's name on the left, a bar from where it started to where it ended on a time line in seconds, and on the right how long it took and, for a call to the model, its tokens: the prompt it got plus what it wrote. The request is the top bar; the tool calls were not timed and show as a small mark at the moment they ran.",
           "axis": "seconds since the request came in",
           "slow_title": "Your slowest(spans) on the twelve traces",
           "slow_alt": "Twelve rows, one for each run of the kitchen agent, on one time line in seconds. Each call to the model is a gray bar, each tool call a small mark; the span your function gives is drawn in color.",
           "slow_sum": "The span your function gives is the longest bar in {m} of the {n} traces.",
           "run": "run {i}",
           "tok_title": "Your tokens_of(spans) on the twelve traces",
           "tok_alt": "A bar for each run of the kitchen agent: the tokens your function gives for it, with the number of calls to the model beside it.",
           "calls": "{c} calls",
           "look_title": "Your not_looked_up(spans, needed) on the twelve traces",
           "look_alt": "A row for each run of the kitchen agent: its dish, each ingredient of the recipe, filled when your function says the agent never asked the pantry about it, and whether the answer was right.",
           "right": "right", "wrong": "wrong", "dish": "dish", "ingredients": "ingredients", "answer": "answer",
           "look_sum": "Your function finds an ingredient that was never looked up in {m} of the {n} traces; {w} of those {m} answers were wrong.",
           "span": "span", "start": "start", "took": "took", "in": "in", "out": "out", "none": "–",
           "spans_title": "The spans of run {i}"},
    "nl": {"request": "request", "model": "model", "s": "s", "tokens": "{p} + {o} tokens",
           "wf_title": "De trace van run {i}: {q}",
           "wf_alt": "Een waterval: een rij voor elke span van het request, in de volgorde waarin ze begonnen. Elke rij heeft links de naam van de span, een balk van waar hij begon tot waar hij eindigde op een tijdlijn in seconden, en rechts hoe lang hij duurde en, bij een call naar het model, zijn tokens: de prompt die hij kreeg plus wat hij schreef. Het request is de bovenste balk; de tool calls zijn niet getimed en staan er als een klein streepje op het moment dat ze liepen.",
           "axis": "seconden sinds het request binnenkwam",
           "slow_title": "Jouw slowest(spans) op de twaalf traces",
           "slow_alt": "Twaalf rijen, een voor elke run van de keukenagent, op één tijdlijn in seconden. Elke call naar het model is een grijze balk, elke tool call een klein streepje; de span die je functie geeft, staat in kleur.",
           "slow_sum": "De span die je functie geeft, is de langste balk in {m} van de {n} traces.",
           "run": "run {i}",
           "tok_title": "Jouw tokens_of(spans) op de twaalf traces",
           "tok_alt": "Een balk voor elke run van de keukenagent: de tokens die je functie ervoor geeft, met het aantal calls naar het model ernaast.",
           "calls": "{c} calls",
           "look_title": "Jouw not_looked_up(spans, needed) op de twaalf traces",
           "look_alt": "Een rij voor elke run van de keukenagent: zijn gerecht, elk ingrediënt van het recept, gevuld als je functie zegt dat de agent er nooit naar vroeg in de voorraadkast, en of het antwoord goed was.",
           "right": "goed", "wrong": "fout", "dish": "gerecht", "ingredients": "ingrediënten", "answer": "antwoord",
           "look_sum": "Je functie vindt een ingrediënt dat nooit werd opgezocht in {m} van de {n} traces; {w} van die {m} antwoorden waren fout.",
           "span": "span", "start": "begin", "took": "duur", "in": "in", "out": "uit", "none": "–",
           "spans_title": "De spans van run {i}"},
}


def sec(x: float, lang: str) -> str:
    """A number of seconds as the text writes it, such as 53.27 or 0.0, with a decimal comma in Dutch."""
    text = repr(round(float(x), 2))
    return text.replace(".", ",") if lang == "nl" else text


def thousands(n: int, lang: str) -> str:
    """A whole number with its thousands marked: 2,271 in English, 2.271 in Dutch."""
    return f"{n:,}".replace(",", "." if lang == "nl" else ",")


def _label(span: dict) -> str:
    """A span's name as the waterfall writes it: a tool with what went in."""
    if span["name"] in ("request", "model"):
        return span["name"]
    args = span.get("input")
    if isinstance(args, dict) and args:
        return f'{span["name"]}({", ".join(str(v) for v in args.values())})'
    return span["name"]


def waterfall(spans: list, lang: str, mark=None, scale: float | None = None) -> str:
    """A trace as a waterfall: a row for each span, a bar from its start to its end, and on the right how long it took
    and its tokens. mark is the id of a span to draw in color."""
    w = _WORDS[lang]
    e = html.escape
    left, bar_w, right, row = 180, 330, 150, 22
    total = scale or max(s["end"] for s in spans) or 1.0
    width = left + bar_w + right
    parts = []
    for k, s in enumerate(spans):
        y = k * row + 4
        x0 = left + s["start"] / total * bar_w
        x1 = left + s["end"] / total * bar_w
        indent = 0 if s["parent"] is None else 12
        name = w["request"] if s["name"] == "request" else (w["model"] if s["name"] == "model" else _label(s))
        room = (left - 12 - indent) // 7          # characters of 12 px code that fit left of the bars
        if len(name) > room:
            name = name[:room - 1] + "…"
        parts.append(f'<text x="{4 + indent}" y="{y + 14}" style="{_MONO};font-size:12px;fill:var(--ea-ink)">{e(name)}</text>')
        colored = mark is not None and s["id"] == mark
        if s["end"] > s["start"]:
            fill = "var(--ea-data)" if colored else ("var(--ea-inset)" if s["parent"] is None else "var(--ea-ink-3)")
            parts.append(f'<rect x="{x0:.1f}" y="{y + 3}" width="{max(1.5, x1 - x0):.1f}" height="14" rx="2" '
                         f'style="fill:{fill};stroke:var(--ea-ink-2);stroke-width:0.8"/>')
        else:
            color = "var(--ea-data)" if colored else "var(--ea-ink)"
            parts.append(f'<rect x="{x0 - 1.5:.1f}" y="{y + 2}" width="3" height="16" style="fill:{color}"/>')
        info = f'{sec(seconds(s), lang)} {w["s"]}'
        if s["name"] == "model":
            info += " · " + w["tokens"].format(p=s["prompt_tokens"], o=s["output_tokens"])
        parts.append(f'<text x="{left + bar_w + 8}" y="{y + 14}" style="{_UI};font-size:11px;fill:var(--ea-ink-2)">{e(info)}</text>')
    h = len(spans) * row + 30
    ticks = ""
    step = 10 if total > 25 else 5
    t = 0
    while t <= total + 0.01:
        x = left + t / total * bar_w
        ticks += (f'<line x1="{x:.1f}" y1="2" x2="{x:.1f}" y2="{h - 22}" style="stroke:var(--ea-line);stroke-width:0.6"/>'
                  f'<text x="{x:.1f}" y="{h - 8}" text-anchor="middle" style="{_UI};font-size:11px;fill:var(--ea-ink-2)">{t}</text>')
        t += step
    axis = f'<text x="{left}" y="{h + 6}" style="{_UI};font-size:11px;fill:var(--ea-ink-2)">{e(w["axis"])}</text>'
    return (f'<div style="overflow-x:auto"><svg viewBox="0 0 {width} {h + 10}" width="100%" style="min-width:560px;max-width:{width}px" '
            f'role="img" aria-label="{e(w["wf_alt"])}">{ticks}{"".join(parts)}{axis}</svg></div>')


def trace_picture(i: int, lang: str, mark=None) -> str:
    w = _WORDS[lang]
    return yours.figure(w["wf_title"].format(i=i, q=QUESTIONS[i][0]), waterfall(trace(i), lang, mark=mark), alt=w["wf_alt"])


def idea_picture(lang: str) -> str:
    return trace_picture(EXAMPLE, lang)


def spans_table(i: int, lang: str) -> str:
    """The spans of run i as a table: name, start, how long it took, what went in and what came out."""
    w = _WORDS[lang]
    e = html.escape
    cell = f"{_UI};font-size:12px;color:var(--ea-ink);padding:0.25rem 0.5rem;vertical-align:top;border-top:1px solid var(--ea-line)"
    th = f"{_UI};font-size:12px;font-weight:400;color:var(--ea-ink-2);padding:0.2rem 0.5rem;text-align:left"
    head = "".join(f'<th style="{th}">{e(h)}</th>' for h in (w["span"], w["start"], w["took"], w["in"], w["out"]))
    rows = ""
    for s in trace(i):
        name = w["request"] if s["name"] == "request" else s["name"]
        if s["name"] == "model":
            what_in = f'{s["prompt_tokens"]} tokens'
        else:
            what_in = json.dumps(s["input"], ensure_ascii=False) if isinstance(s["input"], dict) else str(s["input"])
        out = s["output"] or w["none"]
        rows += (f'<tr><td style="{cell};{_MONO}">{e(name)}</td><td style="{cell};text-align:right">{sec(float(s["start"]), lang)}</td>'
                 f'<td style="{cell};text-align:right">{sec(seconds(s), lang)}</td><td style="{cell};{_MONO}">{e(what_in)}</td>'
                 f'<td style="{cell};max-width:22rem;white-space:pre-wrap;word-break:break-word">{e(out[:400] + (" …" if len(out) > 400 else ""))}</td></tr>')
    table = f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%">{"<tr>" + head + "</tr>"}{rows}</table></div>'
    return yours.figure(w["spans_title"].format(i=i), table)


def run_options(lang: str) -> dict:
    """The runs to choose in the demo: the run's number and its question."""
    return {f'{_WORDS[lang]["run"].format(i=i)}: {q}': i for i, (q, _) in enumerate(QUESTIONS)}


def _mine(seen, name: str):
    mine = seen.mine
    fn = mine.names.get(name) if mine is not None else None
    return fn if callable(fn) else None


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return None


def slowest_picture(seen) -> str:
    """The twelve traces on one time line, with the span the learner's slowest gives in color."""
    fn = _mine(seen, "slowest")
    if fn is None:
        return ""
    w = _WORDS[seen.lang]
    e = html.escape
    left, bar_w, row = 70, 470, 20
    total = max(trace(i)[0]["end"] for i in range(len(QUESTIONS)))
    parts, longest = [], 0
    for i in range(len(QUESTIONS)):
        spans = trace(i)
        got = _safe(fn, trace(i))
        mark = got.get("id") if isinstance(got, dict) else None
        if isinstance(got, dict) and got in spans[1:] and seconds(got) == max(seconds(s) for s in spans[1:]):
            longest += 1
        y = i * row + 4
        parts.append(f'<text x="4" y="{y + 13}" style="{_UI};font-size:11px;fill:var(--ea-ink-2)">{e(w["run"].format(i=i))}</text>')
        for s in spans[1:]:
            x0, x1 = left + s["start"] / total * bar_w, left + s["end"] / total * bar_w
            color = "var(--ea-data)" if s["id"] == mark else ("var(--ea-ink-3)" if s["name"] == "model" else "var(--ea-ink)")
            if s["end"] > s["start"]:
                parts.append(f'<rect x="{x0:.1f}" y="{y + 3}" width="{max(1.5, x1 - x0 - 1):.1f}" height="12" rx="2" style="fill:{color}"/>')
            else:
                parts.append(f'<rect x="{x0 - 1:.1f}" y="{y + 2}" width="2.5" height="14" style="fill:{color}"/>')
    h = len(QUESTIONS) * row + 8
    svg = (f'<svg viewBox="0 0 {left + bar_w + 10} {h}" width="100%" style="max-width:{left + bar_w + 10}px" aria-hidden="true">'
           f'{"".join(parts)}</svg>')
    summary = (f'<div style="{_UI};font-size:13px;color:var(--ea-ink);margin-top:0.3rem">'
               f'{e(w["slow_sum"].format(m=longest, n=len(QUESTIONS)))}</div>')
    return yours.figure(w["slow_title"], f'<div role="img" aria-label="{e(w["slow_alt"])}">{svg}{summary}</div>')


def tokens_picture(seen) -> str:
    """A bar for each trace: the tokens the learner's tokens_of gives, beside its number of calls to the model."""
    fn = _mine(seen, "tokens_of")
    if fn is None:
        return ""
    w = _WORDS[seen.lang]
    e = html.escape
    got = [_safe(fn, trace(i)) for i in range(len(QUESTIONS))]
    known = [g for g in got if isinstance(g, (int, float)) and not isinstance(g, bool)]
    top = max(known + [1])
    rows = ""
    for i, g in enumerate(got):
        calls = sum(s["name"] == "model" for s in trace(i))
        ok = isinstance(g, (int, float)) and not isinstance(g, bool)
        width = max(0.0, g / top * 100) if ok else 0
        text = thousands(int(g), seen.lang) if ok and float(g).is_integer() else e(repr(g))
        rows += (f'<div style="display:flex;align-items:center;gap:0.5rem;margin:0.15rem 0">'
                 f'<span style="{_UI};font-size:12px;color:var(--ea-ink-2);flex:0 0 3.5rem">{e(w["run"].format(i=i))}</span>'
                 f'<span style="flex:1;background:var(--ea-inset);border-radius:3px;height:12px;position:relative">'
                 f'<span style="position:absolute;left:0;top:0;bottom:0;width:{width:.1f}%;background:var(--ea-data);border-radius:3px"></span></span>'
                 f'<span style="{_UI};font-size:12px;color:var(--ea-ink);flex:0 0 3.5rem;text-align:right">{text}</span>'
                 f'<span style="{_UI};font-size:12px;color:var(--ea-ink-2);flex:0 0 4rem">{e(w["calls"].format(c=calls))}</span></div>')
    return yours.figure(w["tok_title"], f'<div role="img" aria-label="{e(w["tok_alt"])}">{rows}</div>')


def looked_picture(seen) -> str:
    """For each trace, the ingredients of its dish, filled where the learner's function says the agent never asked the
    pantry, and whether the answer was right."""
    fn = _mine(seen, "not_looked_up")
    if fn is None:
        return ""
    w = _WORDS[seen.lang]
    e = html.escape
    cell = f"{_UI};font-size:12px;color:var(--ea-ink);padding:0.2rem 0.45rem;border-top:1px solid var(--ea-line);vertical-align:middle"
    th = f"{_UI};font-size:12px;font-weight:400;color:var(--ea-ink-2);padding:0.2rem 0.45rem;text-align:left"
    rows, found, found_wrong = "", 0, 0
    f = findings()
    for i, (_, dish) in enumerate(QUESTIONS):
        got = _safe(fn, trace(i), needed(i))
        missing = got if isinstance(got, list) else []
        chips = ""
        for item in needed(i):
            off = item in missing
            chips += (f'<span style="{_MONO};font-size:12px;display:inline-block;margin:0.1rem 0.25rem 0.1rem 0;padding:0 0.35rem;'
                      f'border-radius:4px;border:1px solid var(--ea-ink-2);'
                      f'{"background:var(--ea-loss);color:var(--ea-page)" if off else "color:var(--ea-ink)"}">{e(item)}</span>')
        right = f["right"][i]
        if missing:
            found += 1
            found_wrong += not right
        rows += (f'<tr><td style="{cell}">{e(w["run"].format(i=i))}</td><td style="{cell}">{e(dish)}</td><td style="{cell}">{chips}</td>'
                 f'<td style="{cell}">{"✓ " + e(w["right"]) if right else "✗ " + e(w["wrong"])}</td></tr>')
    head = "".join(f'<th style="{th}">{e(h)}</th>' for h in ("", w["dish"], w["ingredients"], w["answer"]))
    table = f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><tr>{head}</tr>{rows}</table></div>'
    summary = (f'<div style="{_UI};font-size:13px;color:var(--ea-ink);margin-top:0.35rem">'
               f'{e(w["look_sum"].format(m=found, n=len(QUESTIONS), w=found_wrong))}</div>')
    return yours.figure(w["look_title"], f'<div role="img" aria-label="{e(w["look_alt"])}">{table}{summary}</div>')


# --- exercise 1: the slowest span --------------------------------------------------------------------------------------------

START_SLOWEST = '''def slowest(spans):
    best = spans[1]
    for span in spans:
        if span["parent"] == 0:
            best = span
    return best
'''
SOLUTION_SLOWEST = START_SLOWEST.replace('''        if span["parent"] == 0:
            best = span''', '''        if span["parent"] == 0:
            if span["end"] - span["start"] > best["end"] - best["start"]:
                best = span''')
_SLOWEST_REQUEST = '''def slowest(spans):
    best = spans[1]
    for span in spans:
        if span["end"] - span["start"] > best["end"] - best["start"]:
            best = span
    return best
'''
_SLOWEST_END = SOLUTION_SLOWEST.replace('''span["end"] - span["start"] > best["end"] - best["start"]''', '''span["end"] > best["end"]''')
_SLOWEST_FASTEST = SOLUTION_SLOWEST.replace(" > best[", " < best[")

# --- exercise 2: the tokens of a request ---------------------------------------------------------------------------------------

START_TOKENS = '''def tokens_of(spans):
    total = 0
    for span in spans:
        if span["name"] == "model":
            total = total + span["output_tokens"]
    return total
'''
SOLUTION_TOKENS = START_TOKENS.replace('''total + span["output_tokens"]''', '''total + span["prompt_tokens"] + span["output_tokens"]''')
_TOKENS_PROMPT = START_TOKENS.replace('''span["output_tokens"]''', '''span["prompt_tokens"]''')
_TOKENS_LAST = START_TOKENS.replace('''total = total + span["output_tokens"]''', '''total = span["prompt_tokens"] + span["output_tokens"]''')

# --- exercise 3: the ingredients never looked up -------------------------------------------------------------------------------

START_LOOKED = '''def not_looked_up(spans, needed):
    asked = []
    for span in spans:
        if span["name"] == "pantry":
            asked.append(span["input"]["item"])
    missing = []
    return missing
'''
SOLUTION_LOOKED = START_LOOKED.replace('''    missing = []
    return missing''', '''    missing = []
    for item in needed:
        if item not in asked:
            missing.append(item)
    return missing''')
_LOOKED_ASKED = START_LOOKED.replace('''    missing = []
    return missing''', '''    return asked''')
_LOOKED_IN = SOLUTION_LOOKED.replace("if item not in asked:", "if item in asked:")
_LOOKED_TOOLS = SOLUTION_LOOKED.replace('''        if span["name"] == "pantry":
            asked.append(span["input"]["item"])''', '''        asked.append(span["name"])''')

_CLEARED = {"en": "Your box is empty. Click *Start over* under the box to get the start code back.",
            "nl": "Je vak is leeg. Klik op *Opnieuw beginnen* onder het vak om de begincode terug te krijgen."}


def _small(*rows) -> list:
    """A small trace for the checks: (name, start, end, prompt tokens, output tokens or the tool's input)."""
    spans = [{"id": 0, "parent": None, "name": "request", "start": 0.0, "end": rows[-1][2], "input": "?", "output": ""}]
    for name, start, end, a, b in rows:
        span = {"id": len(spans), "parent": 0, "name": name, "start": start, "end": end}
        if name == "model":
            span.update(prompt_tokens=a, output_tokens=b, output="")
        else:
            span.update(input=a, output=b)
        spans.append(span)
    return spans


_SMALL_A = _small(("model", 0.0, 2.0, 300, 20), ("pantry", 2.0, 2.0, {"item": "milk"}, "200 grams"), ("model", 2.0, 9.5, 340, 90))
_SMALL_B = _small(("model", 0.0, 2.0, 300, 20), ("search", 2.0, 9.5, {"q": "pesto"}, "3 pages"), ("model", 9.5, 11.0, 420, 15))
_SMALL_C = _small(("model", 0.0, 4.0, 310, 40), ("pantry", 4.0, 4.0, {"item": "eggs"}, "0 grams"),
                  ("pantry", 4.0, 4.0, {"item": "butter"}, "250 grams"), ("model", 4.0, 7.0, 380, 30))

EXERCISES = {
    "slowest": FunctionBox(
        name="slowest",
        picture=slowest_picture,
        function="slowest",
        start=START_SLOWEST,
        solution=SOLUTION_SLOWEST,
        visible=[(TRACE,)],
        hidden=[(Named(trace(3), "TRACES[3]"),), (Named(trace(7), "TRACES[7]"),), (_SMALL_A,), (_SMALL_B,), (_SMALL_C,)],
        mistakes=[
            Mistake("request", wrong=_SLOWEST_REQUEST,
                    en="`{call}` gives the request itself: it lasts as long as all its spans together, so it is always the longest. Look only at the spans in it, the ones with `span[\"parent\"] == 0`.",
                    nl="`{call}` geeft het request zelf: dat duurt zo lang als al zijn spans samen, dus is het altijd het langst. Kijk alleen naar de spans erin, die met `span[\"parent\"] == 0`."),
            Mistake("end", wrong=_SLOWEST_END,
                    en="`{call}` gives the span that ended last, not the one that took longest. How long a span took is its end minus its start.",
                    nl="`{call}` geeft de span die het laatst eindigde, niet de span die het langst duurde. Hoe lang een span duurde, is zijn eind min zijn begin."),
            Mistake("fastest", wrong=_SLOWEST_FASTEST,
                    en="`{call}` gives the span that took the shortest time. Keep a span when it took longer than `best`: `>`.",
                    nl="`{call}` geeft de span die het kortst duurde. Houd een span als hij langer duurde dan `best`: `>`."),
        ],
        texts={
            "missing": {"en": "Run the cell with `slowest` first.", "nl": "Voer eerst de cel met `slowest` uit."},
            "cleared": _CLEARED,
            "empty": {"en": "Your `slowest` gives the last span of the request, whatever it took. Keep a span only when its end minus its start is larger than that of `best`.",
                      "nl": "Je `slowest` geeft de laatste span van het request, hoe lang die ook duurde. Houd een span alleen als zijn eind min zijn begin groter is dan die van `best`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: of the spans in the request, the one whose end minus start is the largest.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: van de spans in het request, de span waarvan eind min begin het grootst is."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function look at every span in the request, also a tool that took time?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Bekijkt je functie elke span in het request, ook een tool die tijd kostte?"},
            "no_function": {"en": "Your box does not make a function `slowest`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `slowest`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `slowest` is right in all {n} cases. Look at the picture: what kind of span is the slowest in each trace?",
                        "nl": "Je `slowest` klopt in alle {n} gevallen. Kijk naar het plaatje: wat voor span is in elke trace de langzaamste?"},
        },
    ),
    "tokens_of": FunctionBox(
        name="tokens_of",
        picture=tokens_picture,
        function="tokens_of",
        start=START_TOKENS,
        solution=SOLUTION_TOKENS,
        visible=[(TRACE,)],
        hidden=[(Named(trace(1), "TRACES[1]"),), (Named(trace(9), "TRACES[9]"),), (_SMALL_A,), (_SMALL_B,), (_SMALL_C,)],
        mistakes=[
            Mistake("prompt", wrong=_TOKENS_PROMPT,
                    en="`{call}` gives `{got}`: only the prompt tokens. A call costs what it got and what it wrote: add `span[\"output_tokens\"]` too.",
                    nl="`{call}` geeft `{got}`: alleen de prompttokens. Een call kost wat hij kreeg en wat hij schreef: tel `span[\"output_tokens\"]` er ook bij op."),
            Mistake("last", wrong=_TOKENS_LAST,
                    en="`{call}` gives `{got}`: the tokens of the last call only. Add each call to `total`: `total = total + …`.",
                    nl="`{call}` geeft `{got}`: alleen de tokens van de laatste call. Tel elke call op bij `total`: `total = total + …`."),
        ],
        texts={
            "missing": {"en": "Run the cell with `tokens_of` first.", "nl": "Voer eerst de cel met `tokens_of` uit."},
            "cleared": _CLEARED,
            "empty": {"en": "Your `tokens_of` counts only what the model wrote. Each call also got a prompt: add its `span[\"prompt_tokens\"]` as well.",
                      "nl": "Je `tokens_of` telt alleen wat het model schreef. Elke call kreeg ook een prompt: tel zijn `span[\"prompt_tokens\"]` er ook bij op."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: for each call to the model, its prompt tokens plus its output tokens, added up.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: voor elke call naar het model zijn prompttokens plus zijn outputtokens, bij elkaar opgeteld."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`.",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
            "no_function": {"en": "Your box does not make a function `tokens_of`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `tokens_of`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `tokens_of` is right in all {n} cases. Compare the runs in the picture: what makes one request cost more tokens than another?",
                        "nl": "Je `tokens_of` klopt in alle {n} gevallen. Vergelijk de runs in het plaatje: waardoor kost het ene request meer tokens dan het andere?"},
        },
    ),
    "not_looked_up": FunctionBox(
        name="not_looked_up",
        picture=looked_picture,
        function="not_looked_up",
        start=START_LOOKED,
        solution=SOLUTION_LOOKED,
        visible=[(TRACE, Named(needed(EXAMPLE), "NEEDED"))],
        hidden=[(Named(trace(5), "TRACES[5]"), needed(5)), (Named(trace(3), "TRACES[3]"), needed(3)),
                (Named(trace(1), "TRACES[1]"), needed(1)), (_SMALL_C, ["eggs", "butter", "milk"]), (_SMALL_C, ["eggs"]),
                (_SMALL_B, ["basil"])],
        mistakes=[
            Mistake("asked", wrong=_LOOKED_ASKED,
                    en="`{call}` gives `{got}`: the ingredients the agent did ask about. The question is the other way round: which of `needed` are not in `asked`.",
                    nl="`{call}` geeft `{got}`: de ingrediënten waar de agent wel naar vroeg. De vraag is andersom: welke van `needed` niet in `asked` staan."),
            Mistake("in", wrong=_LOOKED_IN,
                    en="`{call}` gives `{got}`: the ingredients that were looked up. Keep an item when it is not in `asked`: `item not in asked`.",
                    nl="`{call}` geeft `{got}`: de ingrediënten die wel werden opgezocht. Houd een item als het niet in `asked` staat: `item not in asked`."),
            Mistake("tools", wrong=_LOOKED_TOOLS,
                    en="`{call}` gives `{got}`: `asked` holds the names of the spans, such as `pantry`, not the ingredients. Keep the start code's first loop, which takes `span[\"input\"][\"item\"]` of each pantry span.",
                    nl="`{call}` geeft `{got}`: `asked` bevat de namen van de spans, zoals `pantry`, niet de ingrediënten. Houd de eerste loop van de startcode, die `span[\"input\"][\"item\"]` van elke span van de voorraadkast neemt."),
        ],
        texts={
            "missing": {"en": "Run the cell with `not_looked_up` first.", "nl": "Voer eerst de cel met `not_looked_up` uit."},
            "cleared": _CLEARED,
            "empty": {"en": "Your `not_looked_up` always gives an empty list. Go through `needed`, and keep each item that is not in `asked`.",
                      "nl": "Je `not_looked_up` geeft altijd een lege lijst. Loop door `needed`, en houd elk item dat niet in `asked` staat."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: the items of `needed`, in their order, that no pantry span asked about.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: de items van `needed`, in hun volgorde, waar geen span van de voorraadkast naar vroeg."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`.",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
            "no_function": {"en": "Your box does not make a function `not_looked_up`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `not_looked_up`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `not_looked_up` is right in all {n} cases. Look at the picture: what happened to the answers where an ingredient was never looked up?",
                        "nl": "Je `not_looked_up` klopt in alle {n} gevallen. Kijk naar het plaatje: hoe liep het af met de antwoorden waarbij een ingrediënt nooit werd opgezocht?"},
        },
    ),
}

REQUIRED = ["slowest", "tokens_of", "not_looked_up", "faster"]

CASES = [
    ("slowest", START_SLOWEST, "empty", None),
    ("slowest", "", "empty", None),
    ("slowest", SOLUTION_SLOWEST, "correct", None),
    ("slowest", '''def slowest(spans):
    inside = [span for span in spans if span["parent"] == 0]
    return max(inside, key=lambda span: span["end"] - span["start"])
''', "correct", None),
    ("slowest", SOLUTION_SLOWEST.replace('span["parent"] == 0', 'span["name"] == "model"'), "almost", None),
    ("slowest", _SLOWEST_REQUEST, "wrong", "request"),
    ("slowest", _SLOWEST_END, "wrong", "end"),
    ("slowest", _SLOWEST_FASTEST, "wrong", "fastest"),
    ("tokens_of", START_TOKENS, "empty", None),
    ("tokens_of", "", "empty", None),
    ("tokens_of", SOLUTION_TOKENS, "correct", None),
    ("tokens_of", '''def tokens_of(spans):
    return sum(s["prompt_tokens"] + s["output_tokens"] for s in spans if s["name"] == "model")
''', "correct", None),
    ("tokens_of", _TOKENS_PROMPT, "wrong", "prompt"),
    ("tokens_of", _TOKENS_LAST, "wrong", "last"),
    ("not_looked_up", START_LOOKED, "empty", None),
    ("not_looked_up", "", "empty", None),
    ("not_looked_up", SOLUTION_LOOKED, "correct", None),
    ("not_looked_up", '''def not_looked_up(spans, needed):
    asked = {s["input"]["item"] for s in spans if s["name"] == "pantry"}
    return [item for item in needed if item not in asked]
''', "correct", None),
    ("not_looked_up", _LOOKED_ASKED, "wrong", "asked"),
    ("not_looked_up", _LOOKED_IN, "wrong", "in"),
    ("not_looked_up", _LOOKED_TOOLS, "wrong", "tools"),
]
