# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-agent-failures: the ways agents go wrong, read in twenty recorded runs of Qwen3-0.6B
(data/course-agents-qwen.json): the twelve kitchen questions with a thought before each call (build-agent-loop) and the
eight shopping tasks in the course's web shop, clicking buttons by their name (build-browser-use). The failure modes the
course wrote down by hand for each run, the rules in code that find four of the five, what they show, their pictures, and
two functions to write (`repeats`, the calls a run made again, and `compare`, what a shop run left out and what it added
without being asked)."""
import copy
import html
import importlib.util
import json
import re

from everything_ai import paths, yours
from everything_ai.check import FunctionBox, Mistake

DATA = paths.ROOT / "data" / "course-agents-qwen.json"


def _load(lesson_id: str):
    """The checks of an earlier lesson, loaded once for this module."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_agent_failures",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W = _load("build-workflow-or-agent")   # the kitchen: its recipes, its pantry, its tools and the twelve questions
A = _load("build-agent-loop")          # the runs with a thought before each call, and the trace of a run
B = _load("build-browser-use")         # the web shop, its tasks and its step limit


# --- the twenty runs ---------------------------------------------------------------------------------------------------

def _shop_steps(i: int) -> list:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-browser-use"][f"name-plain-{i}"]["answers"][0]["steps"]


_ADDED = re.compile(r"^Added (.+) to your cart\.$", re.M)


def _run(name: str, agent: str, task: str, steps: list, limit: int) -> dict:
    return {"name": name, "agent": agent, "task": task, "calls": W.calls_of(steps),
            "results": [r for s in steps for r in s.get("results", [])], "answer": W.final_of(steps),
            "steps": len(steps), "limit": limit}


RUNS = []
for _i, (_q, _dish) in enumerate(W.QUESTIONS):
    RUNS.append(_run(f"kitchen {_i + 1}", "kitchen", _q, A.react_steps(_i), W.STEPS) | {"dish": _dish})
for _i, (_t, _wanted) in enumerate(B.TASKS):
    _r = _run(f"shop {_i + 1}", "shop", _t, _shop_steps(_i), B.STEPS)
    # what each click put in the cart, from the page the shop showed after it
    _r["added"] = [m.group(1) for res in _r["results"] for m in [_ADDED.search(res)] if m]
    _r["wanted"] = list(_wanted)
    RUNS.append(_r)
NAMES = [r["name"] for r in RUNS]


def run_named(name: str) -> dict:
    return RUNS[NAMES.index(name)]


MODES = ["too-early", "more", "loop", "made-up", "wrong-tool"]

# What the course wrote down by hand after reading each run, with its calls, the results and the answer:
#   too-early   it ended before it had looked up what the answer needs: the recipe and the pantry for each ingredient,
#               or each product of the task
#   more        it did something the task did not ask for: here, a product in the cart that was not asked
#   loop        it did not stop: it made a call it had made before, or went on until the step limit
#   made-up     its answer says something no tool gave: an amount in the pantry it never looked up, a product it never added
#   wrong-tool  it called a tool with an argument that tool does not know: "tomato" and "soup" for the pantry, which
#               knows "tomatoes" and has no soup, and "salt", which no recipe here has
LABELS = {
    "kitchen 1": ["too-early"],
    "kitchen 2": ["too-early"],
    "kitchen 3": ["too-early", "made-up"],
    "kitchen 4": ["too-early", "loop", "wrong-tool"],
    "kitchen 5": ["too-early", "made-up"],
    "kitchen 6": ["too-early"],
    "kitchen 7": ["too-early"],
    "kitchen 8": ["too-early", "wrong-tool"],
    "kitchen 9": ["too-early"],
    "kitchen 10": ["too-early"],
    "kitchen 11": ["too-early"],
    "kitchen 12": ["too-early"],
    "shop 1": [],
    "shop 2": ["more"],
    "shop 3": [],
    "shop 4": ["more"],
    "shop 5": ["more", "loop"],
    "shop 6": ["too-early", "more", "loop", "made-up"],
    "shop 7": [],
    "shop 8": [],
}


# --- the rules in code -------------------------------------------------------------------------------------------------

def repeats(calls):
    seen = []
    again = []
    for call in calls:
        if call in seen:
            again.append(call)
        seen.append(call)
    return again


def compare(added, wanted):
    missing = []
    extra = []
    for product in wanted:
        if product not in added:
            missing.append(product)
    for product in added:
        if product not in wanted:
            extra.append(product)
    return missing, extra


def _unknown(run: dict) -> list:
    """The calls with an argument their tool does not know: a dish without a recipe, an ingredient the pantry does not
    list, or a click the shop answered with an error."""
    if run["agent"] == "shop":
        return [c for c, res in zip(run["calls"], run["results"]) if res.startswith("Error")]
    out = []
    for name, arguments in run["calls"]:
        value = str(next(iter(arguments.values()), "")).lower().strip() if isinstance(arguments, dict) else ""
        if (name == "recipe" and value not in W.RECIPES) or (name == "pantry" and value not in W.PANTRY) or name not in W.FUNCTIONS:
            out.append((name, arguments))
    return out


def _looked_up(run: dict) -> bool:
    """For a kitchen run: whether it called the recipe of its dish and the pantry for every ingredient of it."""
    names = [(n, str(next(iter(a.values()), "")).lower().strip()) for n, a in run["calls"] if isinstance(a, dict)]
    return ("recipe", run["dish"]) in names and all(("pantry", item) in names for item in W.RECIPES[run["dish"]])


def by_rule(run: dict) -> list:
    """The failure modes the rules in code find in a run: all but made-up, which needs reading."""
    found = []
    if run["agent"] == "kitchen":
        if not _looked_up(run):
            found.append("too-early")
    else:
        missing, extra = compare(run["added"], run["wanted"])
        if missing:
            found.append("too-early")
        if extra:
            found.append("more")
    if repeats(run["calls"]) or run["answer"] is None:
        found.append("loop")
    if _unknown(run):
        found.append("wrong-tool")
    return found


def unseen_numbers(run: dict) -> list:
    """The numbers in a run's answer that neither a tool result nor the task contains: the rule that would catch a
    made-up amount, if the model made up new numbers."""
    def numbers(text):
        return set(re.findall(r"\d+(?:\.\d+)?", text or ""))
    told = numbers(run["task"]).union(*(numbers(r) for r in run["results"]))
    return sorted(numbers(run["answer"]) - told)


def findings() -> dict:
    kitchen = [r for r in RUNS if r["agent"] == "kitchen"]
    shop = [r for r in RUNS if r["agent"] == "shop"]
    right = ([r["name"] for r in kitchen if W.right_answer(r["answer"], r["dish"])]
             + [r["name"] for r in shop if sorted(r["added"]) == sorted(r["wanted"])])
    return {
        "runs": len(RUNS), "kitchen": len(kitchen), "shop": len(shop),
        "right": right,
        "count": {m: sum(m in LABELS[n] for n in NAMES) for m in MODES},
        "with": {m: [n for n in NAMES if m in LABELS[n]] for m in MODES},
        "none": [n for n in NAMES if not LABELS[n]],
        "repeats": [r["name"] for r in RUNS if repeats(r["calls"])],
        "limit": [r["name"] for r in RUNS if r["answer"] is None],
        "after_recipe": [r["name"] for r in kitchen if [n for n, _ in r["calls"]] == ["recipe"]],
        "no_calls": [r["name"] for r in RUNS if not r["calls"]],
        "unseen": {n: unseen_numbers(run_named(n)) for n in NAMES if "made-up" in LABELS[n]},
        "looked": [a.get("item") for n, a in run_named("kitchen 6")["calls"] if n == "pantry"],
    }


def code_of(calls: list) -> str:
    """Calls as the lesson writes them in code: [("pantry", {"item": "tomato"}), ...]"""
    return "[" + ", ".join(f'("{n}", {json.dumps(a, ensure_ascii=False)})' for n, a in calls) + "]"


def pantry_sentence(name: str) -> str:
    """The sentence of a run's answer that says what is in the pantry."""
    text = run_named(name)["answer"] or ""
    return next((p.strip() for p in re.split(r"(?<=\.)\s+", text) if "in the pantry" in p and "I have" in p), "")


def number_words(k: int, lang: str) -> str:
    words = {"en": ["no", "one", "two", "three", "four", "five", "six"], "nl": ["geen", "één", "twee", "drie", "vier", "vijf", "zes"]}[lang]
    return words[k] if k < len(words) else str(k)


# --- pictures ----------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui)"
_MONO = "font-family:var(--ea-font-code)"

_WORDS = {
    "en": {"modes": {"too-early": "stops too early", "more": "does more than asked", "loop": "loops",
                     "made-up": "makes up a result", "wrong-tool": "uses a tool wrongly"},
           "right": "right answer", "kitchen": "kitchen", "shop": "shop", "total": "runs",
           "modes_title": "Twenty runs of the course's agent, and how each went wrong",
           "modes_alt": "A grid with a column for each of the twenty runs, twelve in the kitchen and eight in the shop, and a row for each way of going wrong, with a dot where the course found it. The last row marks the runs with a right answer.",
           "modes_note": "Kitchen 6 gave the right answer and still stopped too early: it looked in the pantry only for the eggs, and only the eggs were short.",
           "calls_title": "Your repeats(calls) on the twenty runs",
           "calls_alt": "A row for each of the twenty runs with its tool calls in order. The calls your function gives back are marked, and a run that went on until the step limit says so.",
           "no_calls": "no tool call", "limit": "until the step limit", "key_again": "made before", "key_call": "a call",
           "calls_sum": "Your function finds a call made again in {k} of the {n} runs: {names}.",
           "calls_none": "Your function finds no call made again in any of the {n} runs.",
           "calls_limit": "{names} went on until the step limit without making a call again.",
           "calls_failed": "For {f} of the {n} runs your function gives no list.",
           "cart_title": "Your compare(added, wanted) on the eight shop runs",
           "cart_alt": "A row for each shopping task: what the task asked, and what the agent put in the cart, marked by what your function says: asked and added, added without being asked, or asked and never added.",
           "key": ["asked and added", "added, not asked", "asked, not added"],
           "cart_sum": "Your function finds something missing in {m} of the {n} shop runs, and something added without being asked in {e}.",
           "cart_failed": "For {f} of the {n} shop runs your function gives no two lists.",
           "trace_title": "Kitchen 4: {task}", "trace_alt": "The trace of the run: each tool call, each result, and the answer.",
           "limit_line": "(the step limit stopped the run here)"},
    "nl": {"modes": {"too-early": "stopt te vroeg", "more": "doet meer dan gevraagd", "loop": "blijft in een loop hangen",
                     "made-up": "verzint een resultaat", "wrong-tool": "gebruikt een tool verkeerd"},
           "right": "goed antwoord", "kitchen": "keuken", "shop": "winkel", "total": "runs",
           "modes_title": "Twintig runs van de agent van de course, en hoe elke run ontspoorde",
           "modes_alt": "Een raster met een kolom voor elk van de twintig runs, twaalf in de keuken en acht in de winkel, en een rij voor elke manier van ontsporen, met een stip waar de course hem vond. De laatste rij markeert de runs met een goed antwoord.",
           "modes_note": "Keuken 6 gaf het goede antwoord en stopte toch te vroeg: hij keek in de voorraadkast alleen naar de eieren, en alleen de eieren waren te weinig.",
           "calls_title": "Jouw repeats(calls) op de twintig runs",
           "calls_alt": "Een rij voor elk van de twintig runs met de tool calls op volgorde. De calls die je functie teruggeeft zijn gemarkeerd, en bij een run die doorging tot de stappenlimiet staat dat erbij.",
           "no_calls": "geen tool call", "limit": "tot de stappenlimiet", "key_again": "al eerder gedaan", "key_call": "een call",
           "calls_sum": "Je functie vindt een call die opnieuw gedaan werd in {k} van de {n} runs: {names}.",
           "calls_none": "Je functie vindt in geen van de {n} runs een call die opnieuw gedaan werd.",
           "calls_limit": "{names} ging door tot de stappenlimiet zonder een call opnieuw te doen.",
           "calls_failed": "Bij {f} van de {n} runs geeft je functie geen lijst.",
           "cart_title": "Jouw compare(added, wanted) op de acht winkelruns",
           "cart_alt": "Een rij voor elke boodschappentaak: wat de taak vroeg, en wat de agent in de winkelwagen deed, gemarkeerd met wat je functie zegt: gevraagd en toegevoegd, toegevoegd zonder dat het gevraagd was, of gevraagd en nooit toegevoegd.",
           "key": ["gevraagd en toegevoegd", "toegevoegd, niet gevraagd", "gevraagd, niet toegevoegd"],
           "cart_sum": "Je functie vindt bij {m} van de {n} winkelruns iets wat ontbreekt, en bij {e} iets wat is toegevoegd zonder dat het gevraagd was.",
           "cart_failed": "Bij {f} van de {n} winkelruns geeft je functie geen twee lijsten.",
           "trace_title": "Keuken 4: {task}", "trace_alt": "De trace van de run: elke tool call, elk resultaat, en het antwoord.",
           "limit_line": "(de stappenlimiet stopte de run hier)"},
}

_SHOWN = {"en": {"kitchen": "kitchen", "shop": "shop"}, "nl": {"kitchen": "keuken", "shop": "winkel"}}


def shown_name(name: str, lang: str) -> str:
    """A run's name in the lesson's language: kitchen 4, keuken 4."""
    agent, k = name.split(" ")
    return f"{_SHOWN[lang][agent]} {k}"


def _names(names: list, lang: str) -> str:
    shown = [shown_name(n, lang) for n in names]
    last = {"en": " and ", "nl": " en "}[lang]
    return shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + last + shown[-1]


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return None


def _mine(seen, name: str):
    mine = seen.mine
    fn = mine.names.get(name) if mine is not None else None
    return fn if callable(fn) else None


def _key(x) -> str:
    try:
        return json.dumps(x, sort_keys=True)
    except (TypeError, ValueError):
        return repr(x)


def _marked(items: list, picked) -> list:
    """For each item, whether it is one of `picked`: an item that comes back k times in `picked` marks its last k
    occurrences, the ones made again."""
    left = {}
    for p in picked:
        left[_key(p)] = left.get(_key(p), 0) + 1
    out = [False] * len(items)
    for k in range(len(items) - 1, -1, -1):
        key = _key(items[k])
        if left.get(key, 0) > 0:
            out[k] = True
            left[key] -= 1
    return out


def call_text(call) -> str:
    """A call as the pictures write it: pantry(tomato), click(Add milk to cart)."""
    name, arguments = call
    values = ", ".join(str(v) for v in arguments.values()) if isinstance(arguments, dict) else str(arguments)
    return f"{name}({values})"


_LOOK = {
    "plain": "border:1px solid var(--ea-ink-3);color:var(--ea-ink)",
    "bad": "border:2px solid var(--ea-loss);background:var(--ea-loss-wash);color:var(--ea-loss-text)",
    "gone": "border:1px dashed var(--ea-loss);color:var(--ea-loss-text)",
    "note": "border:1px solid var(--ea-line);color:var(--ea-ink-2)",
}


def _chip(text: str, look: str) -> str:
    return (f'<span style="{_MONO};font-size:12px;display:inline-block;padding:0.12rem 0.4rem;margin:0.12rem;border-radius:4px;{_LOOK[look]}">'
            f'{html.escape(text)}</span>')


def _row(label: str, body: str, width: str = "6.5rem") -> str:
    return (f'<div style="display:flex;align-items:baseline;gap:0.4rem;margin:0.15rem 0;border-top:1px solid var(--ea-line);padding-top:0.15rem">'
            f'<span style="{_UI};font-size:12px;color:var(--ea-ink-2);flex:0 0 {width}">{html.escape(label)}</span><span>{body}</span></div>')


def _summary(text: str, bad: bool = False) -> str:
    return f'<div style="{_UI};font-size:13px;color:{"var(--ea-loss-text)" if bad else "var(--ea-ink)"};margin-top:0.4rem">{html.escape(text)}</div>'


def modes_picture(lang: str) -> str:
    """The twenty runs against the five ways of going wrong, as the course found them, and the runs with a right answer."""
    w = _WORDS[lang]
    e = html.escape
    f = findings()
    cell = "width:1.35rem;min-width:1.35rem;text-align:center;padding:0.2rem 0;border-top:1px solid var(--ea-line)"
    th = f"{_UI};font-size:11px;font-weight:400;color:var(--ea-ink-2);text-align:center;padding:0 0 0.15rem"
    label = f"{_UI};font-size:12px;color:var(--ea-ink);padding:0.2rem 0.6rem 0.2rem 0;border-top:1px solid var(--ea-line);white-space:nowrap"
    kitchen = [n for n in NAMES if n.startswith("kitchen")]
    shop = [n for n in NAMES if n.startswith("shop")]
    head1 = (f'<tr><th></th><th colspan="{len(kitchen)}" style="{th};border-bottom:1px solid var(--ea-ink-3)">{e(w["kitchen"])}</th>'
             f'<th style="width:0.6rem"></th><th colspan="{len(shop)}" style="{th};border-bottom:1px solid var(--ea-ink-3)">{e(w["shop"])}</th>'
             f'<th style="{th};padding-left:0.6rem">{e(w["total"])}</th></tr>')
    head2 = ('<tr><th></th>' + "".join(f'<th style="{th}">{n.split(" ")[1]}</th>' for n in kitchen) + '<th></th>'
             + "".join(f'<th style="{th}">{n.split(" ")[1]}</th>' for n in shop) + '<th></th></tr>')
    dot = '<span style="display:inline-block;width:0.6rem;height:0.6rem;border-radius:50%;background:var(--ea-loss)"></span>'
    tick = f'<span style="{_UI};font-size:12px;color:var(--ea-ink-2)">✓</span>'
    rows = ""
    for m in MODES:
        cells = "".join(f'<td style="{cell}">{dot if m in LABELS[n] else ""}</td>' for n in kitchen)
        cells += '<td></td>' + "".join(f'<td style="{cell}">{dot if m in LABELS[n] else ""}</td>' for n in shop)
        rows += (f'<tr><td style="{label}">{e(w["modes"][m])}</td>{cells}'
                 f'<td style="{label};text-align:right;padding-left:0.6rem;color:var(--ea-loss-text);font-weight:600">{f["count"][m]}</td></tr>')
    cells = "".join(f'<td style="{cell}">{tick if n in f["right"] else ""}</td>' for n in kitchen)
    cells += '<td></td>' + "".join(f'<td style="{cell}">{tick if n in f["right"] else ""}</td>' for n in shop)
    rows += (f'<tr><td style="{label};color:var(--ea-ink-2)">{e(w["right"])}</td>{cells}'
             f'<td style="{label};text-align:right;padding-left:0.6rem;color:var(--ea-ink-2)">{len(f["right"])}</td></tr>')
    table = f'<div style="overflow-x:auto"><table style="border-collapse:collapse">{head1}{head2}{rows}</table></div>'
    note = f'<div style="{_UI};font-size:13px;color:var(--ea-ink);margin-top:0.4rem">{e(w["modes_note"])}</div>'
    return yours.figure(w["modes_title"], f'<div role="img" aria-label="{e(w["modes_alt"])}">{table}{note}</div>')


def trace_picture(name: str, lang: str) -> str:
    """The trace of a kitchen run, as build-agent-loop writes it: thoughts, actions, observations and the answer."""
    w = _WORDS[lang]
    i = NAMES.index(name)
    q = RUNS[i]["task"]
    lines = A.trace_of(A.messages_of(A.react_steps(i), A.REACT, q))
    if RUNS[i]["answer"] is None:
        lines.append(w["limit_line"])
    return yours.figure(w["trace_title"].format(task=q), A.trace_html(lines, 40), alt=w["trace_alt"])


def calls_picture(seen) -> str:
    fn = _mine(seen, "repeats")
    if fn is None:
        return ""
    w = _WORDS[seen.lang]
    rows, found, failed = "", [], 0
    for r in RUNS:
        got = _safe(fn, copy.deepcopy(r["calls"]))
        if not isinstance(got, list):
            failed += 1
            marks = [False] * len(r["calls"])
        else:
            marks = _marked(r["calls"], got)
            if any(marks):
                found.append(r["name"])
        chips = "".join(_chip(call_text(c), "bad" if m else "plain") for c, m in zip(r["calls"], marks))
        if not r["calls"]:
            chips = _chip(w["no_calls"], "note")
        if r["answer"] is None:
            chips += _chip(w["limit"], "note")
        rows += _row(shown_name(r["name"], seen.lang), chips)
    key = _chip(w["key_call"], "plain") + _chip(w["key_again"], "bad")
    if failed:
        said, bad = w["calls_failed"].format(f=failed, n=len(RUNS)), True
    else:
        said = (w["calls_sum"].format(k=len(found), n=len(RUNS), names=_names(found, seen.lang)) if found
                else w["calls_none"].format(n=len(RUNS)))
        on = [n for n in NAMES if run_named(n)["answer"] is None and n not in found]
        if on:
            said += " " + w["calls_limit"].format(names=_names(on, seen.lang))
        bad = False
    return yours.figure(w["calls_title"], f'<div role="img" aria-label="{html.escape(w["calls_alt"])}" style="overflow-x:auto"><div style="min-width:30rem">'
                                          f'<div style="margin-bottom:0.2rem">{key}</div>{rows}</div>{_summary(said, bad)}</div>')


def _two_lists(got):
    """A list of two lists counts as two lists, like a tuple of them: `return [missing, extra]` is right too."""
    if isinstance(got, list) and len(got) == 2 and all(isinstance(x, list) for x in got):
        return tuple(got)
    return got


def _ok_pair(got) -> bool:
    return isinstance(got, tuple) and len(got) == 2 and all(isinstance(x, list) for x in got)


def cart_picture(seen) -> str:
    fn = _mine(seen, "compare")
    if fn is None:
        return ""
    w = _WORDS[seen.lang]
    shop = [r for r in RUNS if r["agent"] == "shop"]
    rows, missing_runs, extra_runs, failed = "", 0, 0, 0
    for r in shop:
        got = _two_lists(_safe(fn, list(r["added"]), list(r["wanted"])))
        if not _ok_pair(got):
            failed += 1
            missing, marks = [], [False] * len(r["added"])
        else:
            missing, extra = got
            marks = _marked(r["added"], extra)
            missing_runs += bool(missing)
            extra_runs += bool(extra)
        chips = "".join(_chip(p, "bad" if m else "plain") for p, m in zip(r["added"], marks))
        chips += "".join(_chip(str(p), "gone") for p in missing)
        task = f'<span style="{_UI};font-size:12px;color:var(--ea-ink);display:inline-block;min-width:16rem;margin-right:0.4rem">{html.escape(r["task"])}</span>'
        rows += _row(shown_name(r["name"], seen.lang), task + chips)
    key = _chip(w["key"][0], "plain") + _chip(w["key"][1], "bad") + _chip(w["key"][2], "gone")
    said = (w["cart_failed"].format(f=failed, n=len(shop)) if failed
            else w["cart_sum"].format(m=missing_runs, e=extra_runs, n=len(shop)))
    return yours.figure(w["cart_title"], f'<div role="img" aria-label="{html.escape(w["cart_alt"])}" style="overflow-x:auto"><div style="min-width:34rem">'
                                         f'<div style="margin-bottom:0.2rem">{key}</div>{rows}</div>{_summary(said, bool(failed))}</div>')


# --- exercises -----------------------------------------------------------------------------------------------------------

class Named(list):
    """A list of the lesson's, written in the check's messages by its name."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return [copy.deepcopy(x, memo) for x in self]


def _of(name: str, field: str) -> Named:
    return Named(copy.deepcopy(run_named(name)[field]), f'RUNS[{NAMES.index(name)}]["{field}"]')


START_REPEATS = '''def repeats(calls):
    seen = []
    again = []
    for call in calls:
        seen.append(call)
    return again
'''
SOLUTION_REPEATS = '''def repeats(calls):
    seen = []
    again = []
    for call in calls:
        if call in seen:
            again.append(call)
        seen.append(call)
    return again
'''
_REP_AFTER = START_REPEATS.replace('''        seen.append(call)
''', '''        seen.append(call)
        if call in seen:
            again.append(call)
''')
_REP_NAME = SOLUTION_REPEATS.replace('''        if call in seen:
            again.append(call)
        seen.append(call)
''', '''        if call[0] in seen:
            again.append(call)
        seen.append(call[0])
''')

START_COMPARE = '''def compare(added, wanted):
    missing = []
    extra = []
    return missing, extra
'''
SOLUTION_COMPARE = '''def compare(added, wanted):
    missing = []
    extra = []
    for product in wanted:
        if product not in added:
            missing.append(product)
    for product in added:
        if product not in wanted:
            extra.append(product)
    return missing, extra
'''
_CMP_SWAPPED = SOLUTION_COMPARE.replace("    return missing, extra\n", "    return extra, missing\n")
_CMP_NO_EXTRA = SOLUTION_COMPARE.replace('''    for product in added:
        if product not in wanted:
            extra.append(product)
''', "")
_CMP_NO_MISSING = SOLUTION_COMPARE.replace('''    for product in wanted:
        if product not in added:
            missing.append(product)
''', "")

_NAMES = {"RUNS": RUNS}

_CLEARED = {"en": "Your box is empty. Click *Start over* under the box to get the start code back.",
            "nl": "Je vak is leeg. Klik op *Opnieuw beginnen* onder het vak om de begincode terug te krijgen."}


def _texts(name: str, empty: dict, wrong: dict, correct: dict) -> dict:
    return {
        "missing": {"en": f"Run the cell with `{name}` first.", "nl": f"Voer eerst de cel met `{name}` uit."},
        "cleared": _CLEARED,
        "empty": empty,
        "wrong": wrong,
        "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`.",
                   "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
        "no_function": {"en": f"Your box does not make a function `{name}`. Keep the `def` line as it was.",
                        "nl": f"Je vak maakt geen functie `{name}`. Laat de `def`-regel staan zoals hij was."},
        "correct": correct,
    }


_MILK = ("pantry", {"item": "milk"})
_THREE = Named([_MILK, _MILK, _MILK], "same_call_three_times")
_NOTHING = Named([], "nothing_added")
_EGGS_BUTTER = Named(["eggs", "butter"], "eggs_and_butter")

class _CompareBox(FunctionBox):
    """Takes a list of two lists as the two lists, like a tuple of them."""

    def _run(self, fn, case):
        return _two_lists(super()._run(fn, case))


EXERCISES = {
    "repeats": FunctionBox(
        name="repeats",
        picture=calls_picture,
        function="repeats",
        start=START_REPEATS,
        solution=SOLUTION_REPEATS,
        visible=[(_of("kitchen 4", "calls"),)],
        hidden=[(_of("shop 6", "calls"),), (_of("kitchen 10", "calls"),), (_of("shop 5", "calls"),), (_of("kitchen 2", "calls"),),
                (_of("kitchen 8", "calls"),), (_THREE,)],
        mistakes=[
            Mistake("after", wrong=_REP_AFTER,
                    en="`{call}` gives `{got}`: every call counts as made before. You put the call in `seen` before you ask whether it is in it, so it always is. Ask first, then add it.",
                    nl="`{call}` geeft `{got}`: elke call telt als al eerder gedaan. Je zet de call in `seen` voordat je vraagt of hij erin staat, dus staat hij er altijd in. Vraag eerst, en voeg hem daarna toe."),
            Mistake("name", wrong=_REP_NAME,
                    en="`{call}` gives `{got}`: you compare only the name of the tool. `pantry` for the tomato and `pantry` for the soup are two different calls; a call is made again only when its arguments are the same too.",
                    nl="`{call}` geeft `{got}`: je vergelijkt alleen de naam van de tool. `pantry` voor de tomaat en `pantry` voor de soep zijn twee verschillende calls; een call is pas opnieuw gedaan als ook de argumenten hetzelfde zijn."),
        ],
        texts=_texts("repeats",
                     empty={"en": "Your `repeats` finds no call made again. Use `seen` to tell whether a call came before, and put such a call in `again`.",
                            "nl": "Je `repeats` vindt geen enkele call die opnieuw gedaan werd. Gebruik `seen` om te zien of een call al eerder kwam, en zet zo'n call in `again`."},
                     wrong={"en": "`{call}` gives `{got}`, and it should give `{want}`: each call that is the same as an earlier call of the run, name and arguments both.",
                            "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: elke call die hetzelfde is als een eerdere call van de run, naam en argumenten allebei."},
                     correct={"en": "Your `repeats` is right in all {n} cases. In the picture, which run looped without ever making a call again?",
                              "nl": "Je `repeats` klopt in alle {n} gevallen. Welke run in het plaatje bleef in een loop hangen zonder ooit een call opnieuw te doen?"}),
        names=_NAMES,
    ),
    "compare": _CompareBox(
        name="compare",
        picture=cart_picture,
        function="compare",
        start=START_COMPARE,
        solution=SOLUTION_COMPARE,
        visible=[(_of("shop 6", "added"), _of("shop 6", "wanted"))],
        hidden=[(_of("shop 2", "added"), _of("shop 2", "wanted")), (_of("shop 4", "added"), _of("shop 4", "wanted")),
                (_of("shop 5", "added"), _of("shop 5", "wanted")), (_of("shop 8", "added"), _of("shop 8", "wanted")),
                (_of("shop 1", "added"), _of("shop 1", "wanted")), (_NOTHING, _EGGS_BUTTER)],
        mistakes=[
            Mistake("swapped", wrong=_CMP_SWAPPED,
                    en="`{call}` gives `{got}`: the two lists are the wrong way round. First what is missing, what the task asked and the cart does not have; then what was added without being asked.",
                    nl="`{call}` geeft `{got}`: de twee lijsten staan andersom. Eerst wat ontbreekt, wat de taak vroeg en niet in de winkelwagen zit; dan wat erbij kwam zonder dat het gevraagd was."),
            Mistake("no-extra", wrong=_CMP_NO_EXTRA,
                    en="`{call}` gives `{got}`: you find what is missing, but `extra` stays empty. Go through `added` too, for each product the task did not ask for.",
                    nl="`{call}` geeft `{got}`: je vindt wat ontbreekt, maar `extra` blijft leeg. Loop ook door `added`, voor elk product dat de taak niet vroeg."),
            Mistake("no-missing", wrong=_CMP_NO_MISSING,
                    en="`{call}` gives `{got}`: you find what was added without being asked, but `missing` stays empty. Go through `wanted` too, for each product that never got into the cart.",
                    nl="`{call}` geeft `{got}`: je vindt wat erbij kwam zonder dat het gevraagd was, maar `missing` blijft leeg. Loop ook door `wanted`, voor elk product dat nooit in de winkelwagen kwam."),
        ],
        texts=_texts("compare",
                     empty={"en": "Your `compare` gives two empty lists. Fill `missing` with what the task asked and the cart does not have, and `extra` with what the cart has and the task did not ask.",
                            "nl": "Je `compare` geeft twee lege lijsten. Vul `missing` met wat de taak vroeg en niet in de winkelwagen zit, en `extra` met wat in de winkelwagen zit en de taak niet vroeg."},
                     wrong={"en": "`{call}` gives `{got}`, and it should give `{want}`: first what the task asked and the cart does not have, in the order of `wanted`; then what the cart has and the task did not ask, in the order of `added`.",
                            "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: eerst wat de taak vroeg en niet in de winkelwagen zit, in de volgorde van `wanted`; dan wat in de winkelwagen zit en de taak niet vroeg, in de volgorde van `added`."},
                     correct={"en": "Your `compare` is right in all {n} cases. Shop 6 said it had added the pine nuts: what does your function say?",
                              "nl": "Je `compare` klopt in alle {n} gevallen. Winkel 6 zei dat hij de pijnboompitten had toegevoegd: wat zegt je functie?"}),
        names=_NAMES,
    ),
}

REQUIRED = ["repeats", "compare", "which-mode", "what-helps"]

CASES = [
    ("repeats", START_REPEATS, "empty", None),
    ("repeats", "", "empty", None),
    ("repeats", SOLUTION_REPEATS, "correct", None),
    ("repeats", '''def repeats(calls):
    again = []
    for k in range(len(calls)):
        if calls[k] in calls[:k]:
            again.append(calls[k])
    return again
''', "correct", None),
    ("repeats", _REP_AFTER, "wrong", "after"),
    ("repeats", _REP_NAME, "wrong", "name"),
    ("compare", START_COMPARE, "empty", None),
    ("compare", "", "empty", None),
    ("compare", SOLUTION_COMPARE, "correct", None),
    ("compare", SOLUTION_COMPARE.replace("    return missing, extra", "    return [missing, extra]"), "correct", None),
    ("compare", '''def compare(added, wanted):
    missing = [p for p in wanted if p not in added]
    extra = [p for p in added if p not in wanted]
    return missing, extra
''', "correct", None),
    ("compare", _CMP_SWAPPED, "wrong", "swapped"),
    ("compare", _CMP_NO_EXTRA, "wrong", "no-extra"),
    ("compare", _CMP_NO_MISSING, "wrong", "no-missing"),
]
