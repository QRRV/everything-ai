# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-tool-design: the tool convert of build-tool-calling, described three ways for Qwen3-0.6B (bare, with
names only, and in full), and its errors told two ways (only "Error", or what to do instead). The course's recorded
runs (data/course-agents-qwen.json): the eight questions of build-tool-calling with each description, every answer
starting with a call of the tool (tool_choice names it), and with names only, the error and the model's next step.
What they show, their pictures, and the exercises: a tool that checks its arguments and says what to do instead, and
the line between a tool your program runs and one the provider runs."""
import copy
import html
import importlib.util
import json
import re
import urllib.request

from everything_ai import chat, paths, serve, yours
from everything_ai.check import MISSING, FunctionBox, Mistake, run_code

DATA = paths.ROOT / "data" / "course-agents-qwen.json"


def _lesson_checks(lesson_id: str):
    """The checks of an earlier lesson, loaded once for this module."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_tool_design",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _lesson_checks("build-tool-calling")   # the tool, its description in full, and the eight questions
UNITS, QUESTIONS, CONVERT = T.UNITS, T.QUESTIONS, T.CONVERT

BARE = {"type": "function", "function": {
    "name": "conv", "description": "",
    "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "string"}, "c": {"type": "string"}},
                   "required": ["a", "b", "c"]}}}
NAMES = {"type": "function", "function": {
    "name": "convert", "description": "Convert an amount.",
    "parameters": {"type": "object",
                   "properties": {"amount": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}},
                   "required": ["amount", "from_unit", "to_unit"]}}}
TERSE = "Error"


def helpful(arguments: dict) -> str:
    """convert, with errors that say what to do instead: the course's solution of exercise 1, in short."""
    amount, a, b = arguments.get("amount"), arguments.get("from_unit"), arguments.get("to_unit")
    if not isinstance(amount, (int, float)) or isinstance(amount, bool):
        return "Error: amount must be a number, such as 37."
    for unit in (a, b):
        if unit not in UNITS:
            return f"Error: unknown unit {unit!r}. Use km, mi, kg or lb."
    if UNITS[a][0] != UNITS[b][0]:
        return f"Error: cannot convert {a} to {b}. Convert km and mi into each other, or kg and lb."
    return T.convert(arguments)


def tool_results(message: dict, ask: dict | None = None) -> list:
    """For the recorder: the result of each call, with the bare tool's arguments a, b and c read as amount, from and
    to, and the errors told as ask["errors"] says: "terse" (only Error) or "helpful"."""
    out = []
    for c in message.get("tool_calls") or []:
        try:
            arguments = json.loads(c["function"]["arguments"])
        except ValueError:
            arguments = {}
        if c["function"]["name"] == "conv":
            arguments = {"amount": arguments.get("a"), "from_unit": arguments.get("b"), "to_unit": arguments.get("c")}
        result = helpful(arguments) if isinstance(arguments, dict) else "Error: the arguments are not a JSON object."
        if (ask or {}).get("errors") == "terse" and result.startswith("Error"):
            result = TERSE
        out.append({"role": "tool", "tool_call_id": c["id"], "content": result})
    return out


def asked(q: str) -> list:
    return [{"role": "user", "content": q}]


def forced(tool: dict) -> dict:
    return {"type": "function", "function": {"name": tool["function"]["name"]}}


RECORD = {}
for _i, (_q, *_) in enumerate(QUESTIONS):
    RECORD[f"bare-{_i}"] = {"messages": asked(_q), "tools": [BARE], "tool_choice": forced(BARE), "max_tokens": 120,
                            "temperature": 0, "steps": 1, "seeds": 1}
    for _kind in ("terse", "helpful"):
        RECORD[f"names-{_kind}-{_i}"] = {"messages": asked(_q), "tools": [NAMES], "tool_choice": forced(NAMES), "max_tokens": 120,
                                         "temperature": 0, "steps": 3, "errors": _kind, "seeds": 1}


# --- what the recorded runs show ----------------------------------------------------------------------------------------

def course() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-tool-design"]


def run(key: str) -> dict:
    return course()[key]["answers"][0]


def as_convert(call: dict) -> dict:
    """The arguments of a call of convert or of the bare conv, as convert names them; {} when they are not JSON."""
    try:
        arguments = json.loads(call["function"]["arguments"])
    except ValueError:
        return {}
    if call["function"]["name"] == "conv":
        return {"amount": arguments.get("a"), "from_unit": arguments.get("b"), "to_unit": arguments.get("c")}
    return arguments


def right_call(call: dict, i: int) -> bool:
    _q, amount, a, b = QUESTIONS[i]
    return as_convert(call) == {"amount": amount, "from_unit": a, "to_unit": b}


def steps_of(kind: str, i: int) -> list:
    """The requests of a run: "full" is build-tool-calling's run with convert named, "bare" and the two "names-…" this
    lesson's."""
    return (T.run(f"named-{i}") if kind == "full" else run(f"{kind}-{i}"))["steps"]


def findings() -> dict:
    """What the runs show: per description, how many first calls were right; the units the model wrote in words with
    names only; the call the wrong way round; and for the questions whose first call got an error with names only,
    what the model did next with each kind of error."""
    n = len(QUESTIONS)
    out = {"questions": n}
    for kind in ("bare", "names-terse", "full"):
        out["right_" + kind.split("-")[0]] = [i for i in range(n) if right_call(steps_of(kind, i)[0]["tool_calls"][0], i)]
        out["prompt_" + kind.split("-")[0]] = steps_of(kind, 0)[0]["prompt_tokens"]
    names = [steps_of("names-terse", i)[0]["tool_calls"][0] for i in range(n)]
    out["words"] = sorted({as_convert(c)[k] for c in names for k in ("from_unit", "to_unit")} - set(UNITS))
    out["wrong_way"] = [i for i in range(n) if (lambda c: c["from_unit"] == QUESTIONS[i][3] and c["to_unit"] == QUESTIONS[i][2])(
        as_convert(steps_of("bare", i)[0]["tool_calls"][0]))]
    failed = [i for i in range(n) if steps_of("names-terse", i)[0]["results"][0].startswith("Error")]
    out["failed"] = failed
    for kind in ("terse", "helpful"):
        runs = {i: steps_of(f"names-{kind}", i) for i in failed}
        out[kind + "_again"] = [i for i, s in runs.items() if len(s) > 1 and s[1]["tool_calls"]]
        out[kind + "_fixed"] = [i for i, s in runs.items() if any(right_call(c, i) for st in s for c in st["tool_calls"])]
        out[kind + "_answered"] = [i for i, s in runs.items() if not s[-1]["tool_calls"] and T.right(s[-1]["text"], i)]
        out[kind + "_echo"] = [i for i, s in runs.items() if not s[-1]["tool_calls"] and s[-1]["text"].startswith("Error")]
        out[kind + "_sorry"] = [i for i, s in runs.items() if not s[-1]["tool_calls"] and s[-1]["text"].startswith("I'm sorry")]
        out[kind + "_open"] = [i for i, s in runs.items() if s[-1]["tool_calls"]]
    return out


WORKED = 6   # the run of the worked example: "mile", the helpful error, then "mi"


# --- the pictures of the recorded runs ------------------------------------------------------------------------------------

_CELL = "padding:0.18rem 0.5rem;border-bottom:1px solid var(--ea-line);vertical-align:top;font-size:13px"
_HEAD = f"{_CELL};color:var(--ea-ink-2);font-weight:500;text-align:left;font-family:var(--ea-font-ui)"


def _ui(text: str, color: str = "var(--ea-ink-2)", size: int = 12) -> str:
    return f'<span style="font-family:var(--ea-font-ui);font-size:{size}px;color:{color}">{html.escape(text)}</span>'


def _mono(text: str, color: str = "var(--ea-ink)") -> str:
    return (f'<span style="font-family:var(--ea-font-code);font-size:13px;color:{color};white-space:pre-wrap;'
            f'overflow-wrap:anywhere">{html.escape(text)}</span>')


def short(arguments: dict) -> str:
    """A call in a few characters: 37 km → mi."""
    if not isinstance(arguments, dict):
        return repr(arguments)
    return "{} {} → {}".format(*(arguments.get(k) for k in ("amount", "from_unit", "to_unit")))


_DESCRIBED_WORDS = {
    "en": {"title": "The first call of each question, with three descriptions of the same tool", "question": "the question",
           "bare": "bare: conv(a, b, c)", "names": "names only", "full": "in full",
           "count": "{n} of {k} right", "right": "should be {v}"},
    "nl": {"title": "De eerste call bij elke vraag, met drie beschrijvingen van dezelfde tool", "question": "de vraag",
           "bare": "kaal: conv(a, b, c)", "names": "alleen namen", "full": "volledig",
           "count": "{n} van de {k} goed", "right": "moet {v} zijn"},
}


def described_picture(lang: str) -> str:
    """For each question, the first call with the bare description, with names only and in full (build-tool-calling's
    run with convert named), a wrong call marked with the right one under it; under each, how many were right."""
    w = _DESCRIBED_WORDS[lang]
    rows = []
    for i, (q, amount, a, b) in enumerate(QUESTIONS):
        cells = []
        for kind in ("bare", "names-terse", "full"):
            c = steps_of(kind, i)[0]["tool_calls"][0]
            ok = right_call(c, i)
            cell = _mono(short(as_convert(c)), "var(--ea-pred-text)" if ok else "var(--ea-loss-text)").replace("pre-wrap", "nowrap")
            if not ok:
                cell += "<div>" + _ui(w["right"].format(v=short({"amount": amount, "from_unit": a, "to_unit": b})), "var(--ea-loss-text)") + "</div>"
            cells.append(cell)
        rows.append(f'<tr><td style="{_CELL}">{_ui(q, "var(--ea-ink)", 13)}</td>'
                    + "".join(f'<td style="{_CELL}">{c}</td>' for c in cells) + "</tr>")
    f = findings()
    rows.append(f'<tr><td style="{_CELL};border-bottom:none"></td>' + "".join(
        f'<td style="{_CELL};border-bottom:none">{_ui(w["count"].format(n=len(f["right_" + k]), k=f["questions"]), "var(--ea-ink)", 13)}</td>'
        for k in ("bare", "names", "full")) + "</tr>")
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%;min-width:34rem"><thead><tr>'
             + "".join(f'<th style="{_HEAD}">{h}</th>' for h in (w["question"], w["bare"], w["names"], w["full"]))
             + f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    return yours.figure(w["title"], table, alt=w["title"])


_ERRORS_WORDS = {
    "en": {"title": "After a wrong call: two kinds of error", "question": "the question", "terse": "the error: only Error",
           "helpful": "the error says what to do instead", "said": "answered:", "open": "no answer after three requests"},
    "nl": {"title": "Na een verkeerde call: twee soorten foutmelding", "question": "de vraag", "terse": "de fout: alleen Error",
           "helpful": "de fout zegt wat het anders moet", "said": "antwoordde:", "open": "geen antwoord na drie requests"},
}


def steps_html(steps: list, i: int, w: dict) -> str:
    """The requests of one run: each call with what came back (an error in the colour of mistakes), then the answer."""
    parts = []
    for s in steps:
        for c, r in zip(s["tool_calls"], s["results"]):
            parts.append("<div>" + _mono(short(as_convert(c)), "var(--ea-pred-text)") + " " + _ui("→") + " "
                         + _mono(r, "var(--ea-loss-text)" if r.startswith("Error") else "var(--ea-data-text)") + "</div>")
    last = steps[-1]
    if last["tool_calls"]:
        parts.append("<div>" + _ui(w["open"]) + "</div>")
    else:
        ok = T.right(last["text"], i)
        parts.append("<div>" + _ui(w["said"]) + " " + _mono(last["text"], "var(--ea-pred-text)" if ok else "var(--ea-loss-text)") + "</div>")
    return "".join(parts)


def errors_picture(lang: str) -> str:
    """For each question whose first call got an error with names only, what followed with a terse error and with a
    helpful one: each call and what came back, and the answer."""
    w = _ERRORS_WORDS[lang]
    rows = []
    for i in findings()["failed"]:
        rows.append(f'<tr><td style="{_CELL}">{_ui(QUESTIONS[i][0], "var(--ea-ink)", 13)}</td>'
                    + "".join(f'<td style="{_CELL}">{steps_html(steps_of("names-" + k, i), i, w)}</td>' for k in ("terse", "helpful"))
                    + "</tr>")
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%;min-width:34rem"><thead><tr>'
             + "".join(f'<th style="{_HEAD}">{h}</th>' for h in (w["question"], w["terse"], w["helpful"]))
             + f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    return yours.figure(w["title"], table, alt=w["title"])


# --- exercise 1: errors a model can use --------------------------------------------------------------------------------

START_HELPFUL = '''FACTOR = {"km": 1000.0, "mi": 1609.344, "kg": 1.0, "lb": 0.45359237}
KIND = {"km": "length", "mi": "length", "kg": "mass", "lb": "mass"}


def convert(arguments):
    amount = arguments["amount"]
    a = arguments["from_unit"]
    b = arguments["to_unit"]
    if a not in FACTOR:
        return "Error"
    if b not in FACTOR:
        return "Error"
    if KIND[a] != KIND[b]:
        return "Error"
    return str(round(amount * FACTOR[a] / FACTOR[b], 2)) + " " + b
'''
SOLUTION_HELPFUL = START_HELPFUL.replace('''    if a not in FACTOR:
        return "Error"
    if b not in FACTOR:
        return "Error"
    if KIND[a] != KIND[b]:
        return "Error"
''', '''    if a not in FACTOR:
        return "Error: unknown unit '" + a + "'. Use km, mi, kg or lb."
    if b not in FACTOR:
        return "Error: unknown unit '" + b + "'. Use km, mi, kg or lb."
    if KIND[a] != KIND[b]:
        pairs = "Convert km and mi into each other, or kg and lb."
        return "Error: cannot convert " + a + " to " + b + ". " + pairs
''')
FOUR = ("km", "mi", "kg", "lb")


def needs(arguments: dict):
    """What an error about these arguments must hold: the unit the model wrote when it is unknown, and the four units
    convert knows; None when the arguments are fine and the result is a number."""
    a, b = arguments["from_unit"], arguments["to_unit"]
    if a not in UNITS:
        return [a, *FOUR]
    if b not in UNITS:
        return [b, *FOUR]
    if UNITS[a][0] != UNITS[b][0]:
        return list(FOUR)
    return None


def _has(text: str, word: str) -> bool:
    """Whether a word is in an error as a word of its own: "mi" in "Use km, mi, kg or lb", not in "miles"."""
    return re.search(r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])", text) is not None


def _same_error(got, want) -> bool:
    """Right when a result is the same, and an error starts with "Error" and holds what needs() says."""
    if not isinstance(got, dict) or got["gave"] is None:
        return False
    need = needs(got["arguments"])
    if need is None:
        return got["gave"] == want["gave"]
    return isinstance(got["gave"], str) and got["gave"].startswith("Error") and all(_has(got["gave"], x) for x in need)


def _error_mistake(kind: str):
    def matches(got, text) -> bool:
        if not isinstance(got, dict):
            return False
        gave, need = got["gave"], needs(got["arguments"])
        if kind == "nothing":
            return gave is None
        if not isinstance(gave, str) or need is None:
            return kind == "result" and need is None and isinstance(gave, str)
        if kind == "terse":
            return gave.strip().rstrip(".:!").strip() == "Error"
        if kind == "start":
            return not gave.startswith("Error")
        if kind == "name":
            return need[0] not in FOUR and not _has(gave, need[0])
        if kind == "list":
            return not all(_has(gave, u) for u in FOUR)
        return False
    return matches


class ToolBox(FunctionBox):
    """A FunctionBox for a tool: the check calls the learner's convert with a dictionary of arguments, as a model's
    call would, and keeps the arguments with what came back, so that the check knows what an error must say."""

    def _run(self, fn, case):
        return {"arguments": case[0], "gave": fn(copy.deepcopy(case[0]))}

    def _call_text(self, case) -> str:
        return "convert(" + json.dumps(case[0]) + ")"


def _given(gave) -> dict:
    v = gave["gave"] if isinstance(gave, dict) else gave
    return {"en": repr(v), "nl": repr(v)}


VISIBLE = [({"amount": 37, "from_unit": "km", "to_unit": "miles"},)]
HIDDEN = [
    ({"amount": 18, "from_unit": "mi", "to_unit": "km"},),
    ({"amount": 3.1, "from_unit": "mile", "to_unit": "km"},),
    ({"amount": 3, "from_unit": "km", "to_unit": "lb"},),
    ({"amount": 64, "from_unit": "kg", "to_unit": "pounds"},),
    ({"amount": 2.5, "from_unit": "lb", "to_unit": "mi"},),
]

_HELPFUL_WORDS = {
    "en": {"title": "What your convert gives back, as the model would read it", "call": "the call", "gave": "your convert gives back",
           "lacks": "lacks", "ok": "says what to do", "result": "a result", "error": "stops with {e}",
           "start": "does not start with Error", "word": "the unit {u}", "units": "the four units"},
    "nl": {"title": "Wat je convert teruggeeft, zoals het model het zou lezen", "call": "de call", "gave": "je convert geeft terug",
           "lacks": "mist", "ok": "zegt wat het moet doen", "result": "een uitkomst", "error": "stopt met {e}",
           "start": "begint niet met Error", "word": "de eenheid {u}", "units": "de vier eenheden"},
}


def _verdict(arguments: dict, gave, w: dict) -> tuple:
    """A short word on what the learner's convert gave for these arguments, and whether it is off."""
    need = needs(arguments)
    if need is None:
        return w["result"], False
    if not isinstance(gave, str) or not gave.startswith("Error"):
        return w["start"], True
    missing = []
    if need[0] not in FOUR and not _has(gave, need[0]):
        missing.append(w["word"].format(u=need[0]))
    if not all(_has(gave, u) for u in FOUR):
        missing.append(w["units"])
    return (w["lacks"] + ": " + ", ".join(missing), True) if missing else (w["ok"], False)


def helpful_picture(seen) -> str:
    """Every call of the check through the learner's convert: what it gave back, and what an error still lacks."""
    w = _HELPFUL_WORDS[seen.lang]
    run = seen.run
    fn = run.names.get("convert") if run is not None and run.error is None else None
    if not callable(fn):
        return ""
    rows = []
    for (arguments,) in VISIBLE + HIDDEN:
        try:
            gave, shown = fn(copy.deepcopy(arguments)), None
        except Exception as e:   # the learner's code; the check says what went wrong
            gave, shown = None, w["error"].format(e=type(e).__name__)
        word, off = _verdict(arguments, gave, w) if shown is None else (shown, True)
        color = "var(--ea-loss-text)" if off else "var(--ea-ink-2)"
        rows.append(f'<tr><td style="{_CELL}">{_mono(short(arguments), "var(--ea-pred-text)")}</td>'
                    f'<td style="{_CELL}">{_mono(repr(gave) if shown is None else "", "var(--ea-data-text)")}</td>'
                    f'<td style="{_CELL}">{_ui(word, color)}</td></tr>')
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%;min-width:30rem"><thead><tr>'
             + "".join(f'<th style="{_HEAD}">{h}</th>' for h in (w["call"], w["gave"], ""))
             + f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    return yours.figure(w["title"], table, alt=w["title"])


_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}

EXERCISES = {
    "helpful": ToolBox(
        name="helpful",
        picture=helpful_picture,
        function="convert",
        start=START_HELPFUL,
        solution=SOLUTION_HELPFUL,
        visible=VISIBLE,
        hidden=HIDDEN,
        compare=_same_error,
        show=_given,
        mistakes=[
            Mistake("raised", error=lambda e: isinstance(e, (ValueError, LookupError)) and not isinstance(e, KeyError),
                    en="With `{call}`, your convert stops with an error instead of giving one back. A tool gives back its error as text: an exception stops your program, and the model never reads it. Use `return` with the text.",
                    nl="Met `{call}` stopt je convert met een fout in plaats van er een terug te geven. Een tool geeft zijn fout terug als tekst: een exception stopt je programma, en het model leest hem nooit. Gebruik `return` met de tekst."),
            Mistake("nothing", matches=_error_mistake("nothing"),
                    en="With `{call}`, your convert gives back nothing. Every way through it ends with `return` and a text: the result, or an error.",
                    nl="Met `{call}` geeft je convert niets terug. Elke weg erdoor eindigt met `return` en een tekst: de uitkomst, of een fout."),
            Mistake("result", matches=_error_mistake("result"),
                    en="With `{call}`, your convert gives back `{got}`, and it should still give `{want}`: these arguments are fine. Leave the last line as it was.",
                    nl="Met `{call}` geeft je convert `{got}` terug, en hij moet nog steeds `{want}` geven: deze argumenten kloppen. Laat de laatste regel zoals hij was."),
            Mistake("terse", matches=_error_mistake("terse"),
                    en="With `{call}`, your convert still gives back only `{got}`. The model then knows that something went wrong, but not what, nor what to do instead. Say both in the text.",
                    nl="Met `{call}` geeft je convert nog steeds alleen `{got}` terug. Het model weet dan dat er iets misging, maar niet wat, en ook niet wat het anders moet doen. Zeg allebei in de tekst."),
            Mistake("start", matches=_error_mistake("start"),
                    en="With `{call}`, your convert gives back `{got}`. Start each error with `Error`, as the box did: then your code, and the model, can tell an error from a result.",
                    nl="Met `{call}` geeft je convert `{got}` terug. Begin elke fout met `Error`, zoals het vak deed: dan kunnen je code, en het model, een fout onderscheiden van een uitkomst."),
            Mistake("name", matches=_error_mistake("name"),
                    en="With `{call}`, your error does not name the unit the model wrote. Put it in the text, so that the model sees which of its words is wrong: add `a` or `b` to the text with `+`.",
                    nl="Met `{call}` noemt je fout de eenheid niet die het model schreef. Zet hem in de tekst, zodat het model ziet welk van zijn woorden niet klopt: voeg `a` of `b` toe aan de tekst met `+`."),
            Mistake("list", matches=_error_mistake("list"),
                    en="With `{call}`, your error does not list the units convert knows: `km`, `mi`, `kg` and `lb`. Put all four in the text, so that the model can pick one in its next call.",
                    nl="Met `{call}` noemt je fout de eenheden niet die convert kent: `km`, `mi`, `kg` en `lb`. Zet ze alle vier in de tekst, zodat het model er bij zijn volgende call een kan kiezen."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "Every error still says only `\"Error\"`. Make each one say what went wrong and what to do instead: which unit is unknown, and which units convert knows.",
                      "nl": "Elke fout zegt nog alleen `\"Error\"`. Laat elke fout zeggen wat er misging en wat het anders moet: welke eenheid onbekend is, en welke eenheden convert kent."},
            "wrong": {"en": "With `{call}`, your convert gives back `{got}`. An error starts with `Error`, says what went wrong, and lists what to use instead.",
                      "nl": "Met `{call}` geeft je convert `{got}` terug. Een fout begint met `Error`, zegt wat er misging, en noemt wat het model in plaats daarvan kan gebruiken."},
            "almost": {"en": "Right for the call above, but with `{call}`, your convert gives back `{got}`. An error starts with `Error`, says what went wrong, and lists what to use instead.",
                       "nl": "Goed bij de call hierboven, maar met `{call}` geeft je convert `{got}` terug. Een fout begint met `Error`, zegt wat er misging, en noemt wat het model in plaats daarvan kan gebruiken."},
            "correct": {"en": "Your convert says what went wrong and what to do instead, in all {n} calls of the check: the unit the model wrote, and the four it can use.",
                        "nl": "Je convert zegt wat er misging en wat het anders moet, bij alle {n} calls van de check: de eenheid die het model schreef, en de vier die het kan gebruiken."},
        },
    ),
}

REQUIRED = ["helpful", "describe-better", "client-side"]

CASES = [
    ("helpful", START_HELPFUL, "empty", None),
    ("helpful", "", "empty", None),
    ("helpful", SOLUTION_HELPFUL, "correct", None),
    ("helpful", START_HELPFUL.replace('''    if a not in FACTOR:
        return "Error"
    if b not in FACTOR:
        return "Error"
    if KIND[a] != KIND[b]:
        return "Error"
''', '''    if a not in FACTOR:
        return "Error: I do not know the unit " + a + ". The units are km, mi, kg and lb."
    if b not in FACTOR:
        return "Error: I do not know the unit " + b + ". The units are km, mi, kg and lb."
    if KIND[a] != KIND[b]:
        return "Error: km and mi are lengths, kg and lb are masses; use two of the same."
'''), "correct", None),
    ("helpful", START_HELPFUL.replace('''    if b not in FACTOR:
        return "Error"
''', '''    if b not in FACTOR:
        return "Error: unknown unit '" + b + "'. Use km, mi, kg or lb."
'''), "wrong", "terse"),
    ("helpful", SOLUTION_HELPFUL.replace('''"Error: unknown unit '" + b + "'. Use km, mi, kg or lb."''', '''"Error: unknown unit. Use km, mi, kg or lb."'''), "wrong", "name"),
    ("helpful", SOLUTION_HELPFUL.replace('''"Error: unknown unit '" + b + "'. Use km, mi, kg or lb."''', '''"Error: unknown unit '" + b + "'."'''), "wrong", "list"),
    ("helpful", SOLUTION_HELPFUL.replace('''"Error: unknown unit '" + b + "'. Use km, mi, kg or lb."''', '''"Unknown unit '" + b + "'. Use km, mi, kg or lb."'''), "wrong", "start"),
    ("helpful", SOLUTION_HELPFUL.replace('''        return "Error: unknown unit '" + b + "'. Use km, mi, kg or lb."''', '''        raise ValueError("unknown unit " + b)'''), "error", "raised"),
    ("helpful", SOLUTION_HELPFUL.replace('''        return "Error: cannot convert " + a + " to " + b + ". " + pairs''',
                                         '''        print("Error: cannot convert " + a + " to " + b + ".")
        return'''), "wrong", "nothing"),
    ("helpful", SOLUTION_HELPFUL.replace(''' + " " + b
''', '''
'''), "wrong", "result"),
    ("helpful", SOLUTION_HELPFUL.replace('''". " + pairs''', '''"."'''), "wrong", "list"),
]


# --- the demo: your convert, behind the description with names only, with Qwen3 ----------------------------------------

DOWNLOADS = {"Qwen3-0.6B": chat.FILES}   # what the demo downloads, for the line under the header
STEPS = 3


def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


def learner_convert(text):
    """The learner's convert from exercise 1, when it is right; else None."""
    ex = EXERCISES["helpful"]
    if text is MISSING or text is None or ex.check(text).state != "correct":
        return None
    return run_code(text).names.get("convert")


def _ask(url: str, messages: list, first: bool) -> dict:
    body = {"model": serve.MODEL, "messages": messages, "tools": [NAMES], "max_tokens": 120, "temperature": 0}
    if first:
        body["tool_choice"] = forced(NAMES)
    request = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())["choices"][0]["message"]


def ask_with(convert_fn, url: str, question: str) -> list:
    """The course's loop, as in the recorded runs: the question with convert described by names only, the call
    forced in the first request; each call run with `convert_fn`, its result sent back, at most STEPS requests.
    Each request's messages, with the answer and the results."""
    messages, out = [{"role": "user", "content": question}], []
    for n in range(STEPS):
        message = _ask(url, messages, n == 0)
        results = []
        for c in message.get("tool_calls") or []:
            try:
                arguments = json.loads(c["function"]["arguments"])
            except ValueError:
                arguments = None
            gave = convert_fn(arguments) if isinstance(arguments, dict) else "Error: the arguments are not JSON."
            results.append({"role": "tool", "tool_call_id": c["id"], "content": str(gave)})
        out.append((messages, message, results))
        if not results:
            break
        messages = messages + [message] + results
    return out


def demo(convert_fn, question: str, words: dict, lang: str) -> str:
    """The learner's convert behind the description with names only, against the course's server with Qwen3 behind
    it: each call and what the learner's convert gave back, and the answer."""
    try:
        running = serve.model_server()
        turns = ask_with(convert_fn, running.url, question)
    except Exception as e:   # the learner's convert, against the real server; it may have run out of memory
        if serve.out_of_memory(e):
            return f"<p><em>{html.escape(words['memory'])}</em></p>"
        return f"<p><em>{html.escape(words['failed'].format(why=type(e).__name__ + ': ' + str(e)))}</em></p>"
    body = []
    for n, (_messages, message, results) in enumerate(turns):
        body.append(T._label(words["request"].format(n=n + 1)))
        if results:
            for c, r in zip(message["tool_calls"], results):
                body.append("<div>" + _mono(short(as_convert(c)), "var(--ea-pred-text)") + " " + _ui("→") + " "
                            + _mono(r["content"], "var(--ea-loss-text)" if r["content"].startswith("Error") else "var(--ea-data-text)")
                            + "</div>")
        else:
            body.append("<div>" + _mono(str(message.get("content")), "var(--ea-pred-text)") + "</div>")
    if turns and turns[-1][2]:
        body.append(T._label(words["open"].format(n=STEPS)))
    return yours.figure(words["title"], "".join(body), alt=words["title"])
