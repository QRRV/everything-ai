# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-torch-compile: break_even, after how many calls a compiled function has made up for
its compile time; and one_graph, a function rewritten so that torch.compile traces it as one graph, counted with
torch._dynamo.explain. And the picture: total time against calls, eager and compiled. The demo compiles the
learner's function and times it, on a button: compiling takes up to a minute."""
import logging
import math
import re
import time

import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai.check import FunctionBox, Mistake, run_code

# Measured on a CPU with 2 threads (PyTorch 2.14): gelu(x * 2 + 1).sum(-1) on 4 million numbers. The first call with
# empty caches (Inductor's cache and Python's bytecode) took 15.8 to 16.1 s; with a warm cache about 1 s.
COMPILE_S, EAGER_S, COMPILED_S = 16.0, 0.0224, 0.0027


def calls_picture(labels: dict) -> str:
    """Total time against the number of calls: eager, a dashed line through 0, and compiled, a solid line that starts
    at the compile time, crossing where compiling has paid off. Times are no role of the colour dictionary: ink and grey,
    told apart by line style and label. The axis runs a little past its last tick, so that the tick's label fits."""
    from everything_ai import chart
    n_max = 2100
    p = chart.Panel((0, n_max), (0, 50), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    xticks=[0, 500, 1000, 1500, 2000], yticks=[0, 10, 20, 30, 40, 50])
    xs = [0, n_max]
    p.line(xs, [0, n_max * EAGER_S], "ink-3", 2.2, dashed=True)
    p.line(xs, [COMPILE_S, COMPILE_S + n_max * COMPILED_S], "ink", 2.2)
    n = COMPILE_S / (EAGER_S - COMPILED_S)
    p.points([n], [n * EAGER_S], "ink")
    p.text(1500, 45, labels["eager"], "ink-2", "end")
    p.text(n_max - 60, COMPILE_S + n_max * COMPILED_S + 3, labels["compiled"], "ink", "end")
    p.text(n + 60, n * EAGER_S - 4.5, labels["even"].format(n=grouped(round(n, -2), labels.get("lang", "en"))), "ink", "start")
    svg = chart.figure([p], alt=labels["alt"], width=330, height=260)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


def grouped(n: float, lang: str) -> str:
    return f"{n:,.0f}".replace(",", "." if lang == "nl" else ",")


# --- exercise 1: break_even ------------------------------------------------------------------------------------------------------

_E_HEAD = "import math\n\n\ndef break_even(compile_s, eager_s, compiled_s):\n"
_E_BODY = ("    if compiled_s >= eager_s:\n        return None\n"
           "    return math.ceil(compile_s / (eager_s - compiled_s))\n")
_E_START = (_E_HEAD + "    # None when a compiled call is not faster; else the number of calls, rounded up,\n"
            "    # after which the time each call saves has paid for compiling\n    ...\n")


def _e(body: str = _E_BODY) -> str:
    return _E_HEAD + body


_E_WRONG = {
    "no-ceil": _e("    if compiled_s >= eager_s:\n        return None\n    return compile_s / (eager_s - compiled_s)\n"),
    "plus-one": _e("    if compiled_s >= eager_s:\n        return None\n    return int(compile_s / (eager_s - compiled_s)) + 1\n"),
    "compiled": _e("    if compiled_s >= eager_s:\n        return None\n    return math.ceil(compile_s / compiled_s)\n"),
    "never": _e("    return math.ceil(compile_s / (eager_s - compiled_s))\n"),
}

# --- exercise 2: one_graph -------------------------------------------------------------------------------------------------------

_G_HEAD = "import torch\n\n\ndef one_graph(x):\n"
_G_BREAKS = ("    m = x.abs().max().item()     # .item() takes a Python number out of the graph\n"
             "    if m > 1:                    # Python decides on that number\n"
             "        x = x / m\n"
             "    return torch.relu(x)\n")
_G_START = (_G_HEAD + "    # the same outcome, as one graph: keep the maximum a tensor, and let torch.where choose\n" + _G_BREAKS)
_G_BODY = "    m = x.abs().max()\n    return torch.relu(torch.where(m > 1, x / m, x))\n"


def _g(body: str = _G_BODY) -> str:
    return _G_HEAD + body


def graph_breaks(text) -> int | None:
    """How many graph breaks torch._dynamo.explain finds in the learner's one_graph on a vector of 4 numbers; None when
    it does not run. Tracing only: nothing is compiled. explain's graph_break_count is the number of graphs minus 1, so
    it misses a break with no operation on one side, such as a print at the end: the reasons count every break.
    Dynamo's own warning about .item() would show under the check as a log, so it is silenced while it traces."""
    import torch._dynamo as dynamo
    ran = run_code(text) if text else None
    fn = ran.names.get("one_graph") if ran is not None and ran.error is None else None
    if not callable(fn):
        return None
    log = logging.getLogger("torch._dynamo")
    level = log.level
    log.setLevel(logging.ERROR)
    try:
        found = dynamo.explain(fn)(torch.tensor([0.5, -2.0, 1.5, 3.0]))
        return max(int(found.graph_break_count), len(found.break_reasons))
    except Exception:
        return None
    finally:
        log.setLevel(level)
        dynamo.reset()


_ITEM_LEFT = Mistake("graph-break", matches=lambda got, text: (graph_breaks(text) or 0) > 0, state="almost",
                     en="Your `one_graph` gives the right outcome in every case, but Dynamo still breaks its graph: somewhere a value leaves the graph for Python, through `.item()`, `float(...)`, `print`, or an `if` on a tensor. Keep the maximum a tensor, and let `torch.where` choose inside the graph.",
                     nl="Je `one_graph` geeft in elk geval de goede uitkomst, maar Dynamo breekt zijn graaf nog: ergens verlaat een waarde de graaf naar Python, via `.item()`, `float(...)`, `print`, of een `if` op een tensor. Houd het maximum een tensor, en laat `torch.where` binnen de graaf kiezen.")

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _close(got, want) -> bool:
    if want is None or got is None:
        return got is None and want is None
    try:
        a = torch.as_tensor(got, dtype=torch.float64)
        b = torch.as_tensor(want, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError):
        return False
    return a.shape == b.shape and bool(torch.allclose(a, b, rtol=1e-5, atol=1e-7))


def _show_t(v):
    try:
        t = torch.as_tensor(v, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError):
        return repr(v)
    return repr((torch.round(t, decimals=4) + 0.0).tolist())


def _show_e(v):
    """A number of calls as Python shows it, a float rounded to 4 decimals without noise; None as None."""
    if isinstance(v, float):
        text = f"{round(v, 4) + 0.0:.4f}".rstrip("0").rstrip(".")
        return {"en": text, "nl": text}
    return repr(v)


def _named(values: list) -> "_Tensor":
    return _Tensor(values)


class _Tensor:
    """A tensor of a test case, written in the check's messages as the code that makes it."""

    def __init__(self, values: list):
        self.values = values

    def __repr__(self):
        return f"torch.tensor({self.values!r})"

    def __deepcopy__(self, memo):
        return torch.tensor(self.values)


EXERCISES = {
    "break-even": FunctionBox(
        name="break-even",
        function="break_even",
        start=_E_START,
        solution=_e(),
        visible=[(30.0, 0.05, 0.02)],
        hidden=[(16.0, 0.0224, 0.0027), (1.0, 0.5, 0.25), (10.0, 0.01, 0.01), (5.0, 0.2, 0.25)],
        show=_show_e,
        first=[_DOTS],
        mistakes=[
            Mistake("no-ceil", wrong=_E_WRONG["no-ceil"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: a number of calls is a whole number. Round up with `math.ceil`: after fewer calls, compiling has not yet paid off.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: een aantal aanroepen is een geheel getal. Rond naar boven af met `math.ceil`: na minder aanroepen heeft het compileren zich nog niet terugverdiend."),
            Mistake("plus-one", wrong=_E_WRONG["plus-one"], state="almost",
                    en="Right for the example. But `{call}` gives `{got}`, and it should give `{want}`: 1 / 0.25 is exactly 4, and rounding up leaves a whole number as it is. `math.ceil` only rounds up when something is left after the decimal point.",
                    nl="Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn: 1 / 0,25 is precies 4, en naar boven afronden laat een geheel getal zoals het is. `math.ceil` rondt alleen naar boven af als er achter de komma iets over is."),
            Mistake("compiled", wrong=_E_WRONG["compiled"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: you divide the compile time by the time of a compiled call. What pays for compiling is what each call saves: `eager_s - compiled_s`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je deelt de compileertijd door de tijd van een gecompileerde aanroep. Wat het compileren terugbetaalt, is wat elke aanroep bespaart: `eager_s - compiled_s`."),
            Mistake("never", error=lambda e: isinstance(e, (ZeroDivisionError, ValueError, OverflowError)),
                    en="Your function stops when a compiled call is not faster than an eager one: then what each call saves is 0 or less, and compiling never pays off. Give back `None` in that case, before you divide.",
                    nl="Je functie stopt als een gecompileerde aanroep niet sneller is dan een eager aanroep: dan is wat elke aanroep bespaart 0 of minder, en verdient het compileren zich nooit terug. Geef in dat geval `None` terug, voordat je deelt."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: `None` when a compiled call is not faster, else the compile time divided by what each call saves, rounded up.",
                      "nl": "Vervang de `...`: `None` als een gecompileerde aanroep niet sneller is, anders de compileertijd gedeeld door wat elke aanroep bespaart, naar boven afgerond."},
            "arguments": {"en": "The check calls `{call}`, with the compile time and the times of an eager and a compiled call, and the `def` line of your function has another number of names between its parentheses. Keep `(compile_s, eager_s, compiled_s)`.",
                          "nl": "De check roept `{call}` aan, met de compileertijd en de tijden van een eager en een gecompileerde aanroep, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(compile_s, eager_s, compiled_s)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Each call saves `eager_s - compiled_s`; how many of those does the compile time take, rounded up?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Elke aanroep bespaart `eager_s - compiled_s`; hoeveel daarvan kost de compileertijd, naar boven afgerond?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work when compiled is not faster, and when the division is not a whole number?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt het ook als gecompileerd niet sneller is, en als de deling geen geheel getal is?"},
            "correct": {"en": "Your break-even is right in all {n} cases: for the measurement of the picture, after about 800 calls.",
                        "nl": "Je omslagpunt klopt in alle {n} gevallen: voor de meting van de figuur na ongeveer 800 aanroepen."},
        },
    ),
    "one-graph": FunctionBox(
        name="one-graph",
        function="one_graph",
        start=_G_START,
        solution=_g(),
        visible=[(_named([0.5, -2.0, 1.5, 3.0]),)],
        hidden=[(_named([0.5, -0.25, 0.75]),), (_named([-4.0, 2.0]),), (_named([1.0, -1.0, 0.0]),)],
        show=_show_t,
        compare=_close,
        first=[_DOTS],
        form=[_ITEM_LEFT],
        mistakes=[
            Mistake("no-abs", wrong=_g("    m = x.max()\n    return torch.relu(torch.where(m > 1, x / m, x))\n"), state="almost",
                    en="Right for the example. But `{call}` gives `{got}`, and it should give `{want}`: the value largest in size is −4, so divide by 4, its absolute value, and not by 2, the largest value. `x.abs().max()` gives the largest absolute value.",
                    nl="Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn: de waarde die in grootte het grootst is, is −4, dus deel door 4, de absolute waarde daarvan, en niet door 2, de grootste waarde. `x.abs().max()` geeft de grootste absolute waarde."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Change the lines of `one_graph` so that the maximum stays a tensor and `torch.where` chooses: the outcome must stay the same.",
                      "nl": "Verander de regels van `one_graph` zo dat het maximum een tensor blijft en `torch.where` kiest: de uitkomst moet hetzelfde blijven."},
            "arguments": {"en": "The check calls `{call}`, with one tensor, and the `def` line of your function has another number of names between its parentheses. Keep `(x)`.",
                          "nl": "De check roept `{call}` aan, met één tensor, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(x)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Divide by the largest absolute value only when it is above 1, and take the ReLU at the end, as the function did before.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Deel alleen door de grootste absolute waarde als die boven 1 ligt, en neem aan het eind de ReLU, zoals de functie eerst deed."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Work this case out by hand: what is the largest absolute value, and is it above 1?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken dit geval met de hand na: wat is de grootste absolute waarde, en ligt die boven 1?"},
            "correct": {"en": "Your `one_graph` gives the same outcome in all {n} cases, and Dynamo traces it as one graph, without a break.",
                        "nl": "Je `one_graph` geeft in alle {n} gevallen dezelfde uitkomst, en Dynamo traceert hem als één graaf, zonder breuk."},
        },
    ),
}

# The lesson is done when these are correct. "when-compile" is the choice question.
REQUIRED = ["break-even", "one-graph", "when-compile"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("break-even", _E_START, "empty", None),
    ("break-even", _e(), "correct", None),
    ("break-even", _e("    saved = eager_s - compiled_s\n    if saved <= 0:\n        return None\n    return math.ceil(compile_s / saved)\n"), "correct", None),
    ("break-even", _E_WRONG["no-ceil"], "wrong", "no-ceil"),
    ("break-even", _E_WRONG["plus-one"], "almost", "plus-one"),
    ("break-even", _E_WRONG["compiled"], "wrong", "compiled"),
    ("break-even", _E_WRONG["never"], "error", "never"),
    ("one-graph", _G_START, "empty", None),
    ("one-graph", _g(), "correct", None),
    ("one-graph", _g("    m = x.abs().max()\n    scale = torch.where(m > 1, m, torch.ones_like(m))\n    return torch.relu(x / scale)\n"), "correct", None),
    ("one-graph", _g("    m = x.abs().max()\n    print(m)\n    return torch.relu(torch.where(m > 1, x / m, x))\n"), "almost", "graph-break"),
    ("one-graph", _g("    y = torch.relu(torch.where(x.abs().max() > 1, x / x.abs().max(), x))\n    print(y)\n    return y\n"), "almost", "graph-break"),
    ("one-graph", _g("    m = x.max()\n    return torch.relu(torch.where(m > 1, x / m, x))\n"), "almost", "no-abs"),
    ("one-graph", _g("    return torch.relu(x / x.abs().max().clamp(min=1))\n"), "correct", None),
]


# --- the demo: compile the learner's function and time it -------------------------------------------------------------------------

def demo(text, n: int = 4_000_000, runs: int = 10) -> dict | str:
    """The learner's one_graph (or the course's) on n numbers: the first compiled call, which compiles, and then the
    time of a call, eager and compiled, on the GPU if there is one; "error" when compiling fails on this computer."""
    ran = run_code(text) if text else None
    fn = ran.names.get("one_graph") if ran is not None and ran.error is None else None
    yours = callable(fn)
    fn = fn if yours else run_code(_g()).names["one_graph"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(n, generator=torch.Generator().manual_seed(0)).to(device)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def timed(f):
        f(x)
        sync()
        start = time.perf_counter()
        for _ in range(runs):
            f(x)
        sync()
        return (time.perf_counter() - start) / runs

    try:
        import torch._dynamo as dynamo
        dynamo.reset()
        compiled = torch.compile(fn)
        start = time.perf_counter()
        compiled(x)
        sync()
        first = time.perf_counter() - start
        eager_s, compiled_s = timed(fn), timed(compiled)
        same = bool(torch.allclose(fn(x), compiled(x), rtol=1e-5, atol=1e-6))
    except Exception:
        return "error"
    return {"device": device, "first": first, "eager": eager_s, "compiled": compiled_s, "same": same, "yours": yours}
