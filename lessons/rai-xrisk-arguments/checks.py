# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-xrisk-arguments: the arguments for catastrophic risk. The learner writes chain, the chance
that each premise of an argument and all premises before it hold, after the six credences of Carlsmith's report (April
2021), and outputs, what an agent makes in a made-up model of instrumental convergence for each number of steps it
first spends gathering resources. A demo changes how many steps the agent has."""
import contextlib
import copy
import html
import io
import re

from everything_ai import chart, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, same_value, with_time_limit

# --- the data --------------------------------------------------------------------------------------------------------

CARLSMITH = [0.65, 0.80, 0.40, 0.65, 0.40, 0.95]   # Carlsmith's credence in each premise, given the ones before (2021)
STEPS = 10                     # made-up: the steps the agent has in exercise 2's first case
BOOST = 1.3                    # made-up: each step of gathering makes every later work step 1.3 times as productive
HORIZONS = [3, 5, 8, 10, 20, 30]   # the stops of the demo's slider: how many steps the agent has


def chain(credences):
    """What the learner writes in Exercise 1: for each premise, the chance that it and all premises before it hold,
    as Carlsmith multiplies the report's conditional credences."""
    result = []
    total = 1
    for credence in credences:
        total = total * credence
        result.append(total)
    return result


def outputs(steps, boost):
    """What the learner writes in Exercise 2: what the agent makes, in units of its goal, for each number of steps it
    first spends gathering, from 0 up to and including all of them; each work step that is left makes boost to the
    power of the gathering steps."""
    result = []
    for gather in range(steps + 1):
        result.append((steps - gather) * boost ** gather)
    return result


def best(steps, boost: float = BOOST):
    """The number of gathering steps at which the agent makes the most, the first one on a tie."""
    made = outputs(steps, boost)
    return max(range(len(made)), key=lambda g: made[g])


def first_more(horizons=HORIZONS, boost: float = BOOST):
    """The smallest number of steps of the slider at which the agent spends more steps gathering than working."""
    return next(s for s in horizons if best(s, boost) > s - best(s, boost))


# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink);vertical-align:middle")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"
_CODE = "font-family:var(--ea-font-code);font-size:0.78rem"


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def _table(rows: list, head: list) -> str:
    """A table with a head row; each row is (its cells, a style of the row), and each cell is (html, style)."""
    top = "".join(f'<th style="{_HEAD}">{h}</th>' for h in head)
    body = "".join(f'<tr style="{style}">' + "".join(f'<td style="{_CELL}{";" + s if s else ""}">{cell}</td>'
                                                    for cell, s in cells_) + "</tr>" for cells_, style in rows)
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
            f'<tr>{top}</tr>{body}</table></div>')


def _code(text: str) -> str:
    return f'<span style="{_CODE}">{html.escape(text)}</span>'


def _number(v):
    """v as a float, or None for something that is not a number, such as True or a text."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError, OverflowError):
        return None


def _value(v, digits: int = 4, limit: int = 70) -> str:
    """A result as the check shows it: a number with at most `digits` decimals, a text in quotes, a list or a dict of
    them, else what Python shows, cut short when it is longer than `limit`."""
    if isinstance(v, (list, tuple)):
        text = "[" + ", ".join(_value(x, digits, 10 ** 6) for x in v) + "]"
    elif isinstance(v, dict):
        text = "{" + ", ".join(f"{_value(k, digits)}: {_value(x, digits)}" for k, x in v.items()) + "}"
    elif isinstance(v, str):
        text = '"' + v + '"'
    else:
        f = _number(v)
        if f is None or f != f or f in (float("inf"), float("-inf")):
            text = repr(v) if f is None else str(f)
        else:
            text = f"{f:.{digits}f}".rstrip("0").rstrip(".")
            text = "0" if text == "-0" else text
    return text if len(text) <= limit else text[:limit - 3] + "..."


def show_chain(v) -> str:
    return _value(v, 4, 200)


def show_outputs(v) -> str:
    return _value(v, 2, 200)


def _prose(x: float, lang: str, digits: int = 2) -> str:
    """A number in a picture's words: at most `digits` decimals, with a decimal comma in Dutch."""
    text = f"{x:.{digits}f}"
    text = text.rstrip("0").rstrip(".") if digits > 0 else text
    return text.replace(".", ",") if lang == "nl" else text


def _same(got, want) -> bool:
    """Whether the learner's list is the course's, as the check compares them."""
    return isinstance(got, list) and same_value(got, want)


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _all gives for a call that stopped with an error


def _all(fn, calls: list) -> list:
    """The learner's function on copies of each list of arguments in `calls`, all within one time limit of a check:
    a list of the results, with _STOPPED for each call that stops with an error; the Timeout passes on."""
    def one(args):
        try:
            return fn(*(copy.deepcopy(a) for a in args))
        except Timeout:
            raise
        except Exception:   # noqa: BLE001   the check above says what went wrong
            return _STOPPED

    with contextlib.redirect_stdout(io.StringIO()):
        return with_time_limit(lambda: [one(args) for args in calls])


def _rings(got, xs: list, low: float, high: float) -> tuple[list, list]:
    """The learner's numbers that are inside the panel, as rings at the x of each place in the list."""
    mine = got if isinstance(got, list) else []
    rx, ry = [], []
    for k, x in enumerate(xs):
        f = _number(mine[k]) if k < len(mine) else None
        if f is not None and low <= f <= high:
            rx.append(x)
            ry.append(f)
    return rx, ry


# the demo: what the agent makes for each number of gathering steps, as a share of the most it can make
_DEMO_WORDS = {
    "en": {"title": "What the agent makes with {steps} steps",
           "x": "steps spent gathering first", "y": "share of the most it can make", "peak": "gather {g}, work {w}",
           "said": "With {steps} steps, the agent makes the most by gathering for {gs} and then working for {ws}.",
           "none": "With {steps} steps, the agent makes the most by working all {steps} steps: gathering does not pay.",
           "note": "A made-up model, not Bostrom's: each step, the agent either works on its goal or gathers resources "
                   "that make every later work step 1.3 times as productive. Whatever the goal, paperclips or digits of "
                   "pi, the curve is the same."},
    "nl": {"title": "Wat de agent maakt met {steps} stappen",
           "x": "stappen eerst besteed aan verzamelen", "y": "deel van het maximum", "peak": "verzamel {g}, werk {w}",
           "said": "Met {steps} stappen maakt de agent het meest door eerst {gs} aan verzamelen te besteden en daarna "
                   "{ws} aan werken.",
           "none": "Met {steps} stappen maakt de agent het meest door alle {steps} stappen aan werken te besteden: "
                   "verzamelen loont niet.",
           "note": "Een verzonnen model, niet dat van Bostrom: elke stap werkt de agent aan het doel of verzamelt de agent "
                   "middelen die elke latere werkstap 1,3 keer zo productief maken. Wat het doel ook is, paperclips of "
                   "decimalen van pi, de kromme is dezelfde."},
}


def _steps(k: int, lang: str) -> str:
    """k steps in words: 1 step, 4 steps; 1 stap, 4 stappen."""
    one, more = ("step", "steps") if lang == "en" else ("stap", "stappen")
    return f"{k} {one if k == 1 else more}"


def demo_said(steps, lang: str) -> str:
    g = best(steps)
    w = _DEMO_WORDS[lang]
    if g == 0:
        return w["none"].format(steps=steps)
    return w["said"].format(steps=steps, gs=_steps(g, lang), ws=_steps(steps - g, lang))


def demo_frame(v, lang: str) -> tuple[str, str]:
    """The demo: what the agent makes for each number of gathering steps, as a share of the most, at the chosen
    number of steps, with its best point marked."""
    steps = min(HORIZONS, key=lambda k: abs(k - float(v)))
    w = _DEMO_WORDS[lang]
    made = outputs(steps, BOOST)
    top = max(made)
    g = best(steps)
    panel = chart.Panel(x=(-0.6, 31), y=(0, 1.18), xlabel=w["x"], ylabel=w["y"], xticks=[0, 5, 10, 15, 20, 25, 30],
                        yticks=[0, 0.25, 0.5, 0.75, 1])
    xs = list(range(steps + 1))
    panel.line(xs, [m / top for m in made], role="data")
    panel.points([g], [1], role="data", r=4)
    words = w["peak"].format(g=g, w=steps - g)
    if g <= 15:   # the label stays inside the panel, above the curve
        panel.text(g + 0.7, 1.07, words, role="ink-2", anchor="start", halo=True)
    else:
        panel.text(g - 0.7, 1.07, words, role="ink-2", anchor="end", halo=True)
    title = w["title"].format(steps=steps)
    said = demo_said(steps, lang)
    svg = chart.figure([panel], alt=title + ". " + said, width=380, height=240)
    if lang == "nl":
        svg = chart.with_decimal_comma(svg)
    return yours.figure(title, svg + _note(w["note"]), alt=title + ". " + said), said


# --- the pictures under the checks -----------------------------------------------------------------------------------

_CHAIN_WORDS = {
    "en": {"title": "Your chain for Carlsmith's six premises",
           "x": "premises so far", "y": "chance that all of them hold",
           "head": ["credences", "the course's", "yours"], "stopped": "error",
           "note": "The chart is the first case of the check, Carlsmith's credences of April 2021: at each premise, "
                   "the chance that it and all premises before it hold. A dot is the course's chance and a ring is "
                   "yours. The table is every case; a dashed row is one where your chain gives something other than "
                   "the course's."},
    "nl": {"title": "Jouw keten voor de zes premissen van Carlsmith",
           "x": "premissen tot nu toe", "y": "kans dat ze allemaal kloppen",
           "head": ["credences", "van de course", "de jouwe"], "stopped": "fout",
           "note": "De grafiek is het eerste geval van de check, de subjectieve kansen van Carlsmith uit april 2021: bij "
                   "elke premisse de kans dat die en alle premissen ervoor kloppen. Een stip is de kans van de course en een "
                   "ring de jouwe. De tabel bevat elk geval; een gestippelde rij is er een waar jouw keten iets anders "
                   "geeft dan die van de course."},
}


def chain_picture(seen) -> str:
    """The first case as the falling chance after each premise, the course's as dots and the learner's as rings, and
    every case in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _CHAIN_WORDS[seen.lang]
    got = _all(fn, CHAIN_CASES)
    rows = []
    for i, case in enumerate(CHAIN_CASES):
        want = chain(*case)
        if got[i] is _STOPPED:
            shown_, off = html.escape(w["stopped"]), True
        else:
            shown_, off = _code(show_chain(got[i])), not _same(got[i], want)
        rows.append(([(_code(_value(case[0])), ""), (_code(show_chain(want)), ""), (shown_, "font-weight:600")],
                     _OFF if off else ""))
    want = chain(CARLSMITH)
    xs = list(range(1, len(CARLSMITH) + 1))
    panel = chart.Panel(x=(0.6, 6.4), y=(0, 1.05), xlabel=w["x"], ylabel=w["y"], xticks=xs,
                        yticks=[0, 0.25, 0.5, 0.75, 1])
    panel.line(xs, want, role="ink-3")
    panel.points(*_rings(got[0], xs, -0.02, 1.05), role="ink", r=7, hollow=True)
    panel.points(xs, want, role="ink", r=3)
    svg = chart.figure([panel], alt=w["title"], width=380, height=240)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


_OUTPUTS_WORDS = {
    "en": {"title": "Your outputs for 10 steps and a boost of 1.3",
           "x": "steps spent gathering first", "y": "units of the goal",
           "head": ["steps", "boost", "the course's", "yours"], "stopped": "error",
           "note": "The chart is the first case of the check: 10 steps, and each step of gathering makes every later "
                   "work step 1.3 times as productive. A dot is what the course's agent makes and a ring is what yours "
                   "makes. The table is every case; a dashed row is one where your outputs gives something other than "
                   "the course's. The numbers are made up."},
    "nl": {"title": "Jouw outputs voor 10 stappen en een boost van 1,3",
           "x": "stappen eerst besteed aan verzamelen", "y": "eenheden van het doel",
           "head": ["steps", "boost", "van de course", "de jouwe"], "stopped": "fout",
           "note": "De grafiek is het eerste geval van de check: 10 stappen, en elke stap verzamelen maakt elke latere "
                   "werkstap 1,3 keer zo productief. Een stip is wat de agent van de course maakt en een ring wat de "
                   "jouwe maakt. De tabel bevat elk geval; een gestippelde rij is er een waar jouw outputs iets anders "
                   "geeft dan die van de course. De getallen zijn verzonnen."},
}


def outputs_picture(seen) -> str:
    """The first case as what the agent makes against the gathering steps, the course's as dots and the learner's as
    rings, and every case in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _OUTPUTS_WORDS[seen.lang]
    got = _all(fn, OUTPUT_CASES)
    rows = []
    for i, case in enumerate(OUTPUT_CASES):
        want = outputs(*case)
        if got[i] is _STOPPED:
            shown_, off = html.escape(w["stopped"]), True
        else:
            shown_, off = _code(show_outputs(got[i])), not _same(got[i], want)
        rows.append(([(_code(_value(x)), "text-align:right") for x in case]
                     + [(_code(show_outputs(want)), ""), (shown_, "font-weight:600")], _OFF if off else ""))
    want = outputs(STEPS, BOOST)
    xs = list(range(STEPS + 1))
    panel = chart.Panel(x=(-0.4, 10.4), y=(0, 21.5), xlabel=w["x"], ylabel=w["y"], xticks=[0, 2, 4, 6, 8, 10],
                        yticks=[0, 5, 10, 15, 20])
    panel.line(xs, want, role="ink-3")
    panel.points(*_rings(got[0], xs, -0.4, 21.5), role="ink", r=7, hollow=True)
    panel.points(xs, want, role="ink", r=3)
    svg = chart.figure([panel], alt=w["title"], width=380, height=240)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["chain", "outputs", "smart", "culture"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# exercise 1: chain; the first case is Carlsmith's six credences, then three premises at a half, two that are sure, no
# premises, a premise that is sure in the middle, and five premises at 0.8
CHAIN_CASES = [(CARLSMITH,), ([0.5, 0.5, 0.5],), ([1, 1],), ([],), ([0.2, 1, 0.5],), ([0.8] * 5,)]
_C_START = "def chain(credences):\n    ...\n"
_C = ("def chain(credences):\n"
      "    result = []\n"
      "    total = 1\n"
      "    for credence in credences:\n"
      "        total = total * credence\n"
      "        result.append(total)\n"
      "    return result\n")
_C_ZERO = _C.replace("    total = 1\n", "    total = 0\n")
_C_ADD = _C.replace("        total = total * credence\n", "        total = total + credence\n")
_C_EACH = _C.replace("        result.append(total)\n", "        result.append(credence)\n")
_C_BEFORE = _C.replace("        total = total * credence\n        result.append(total)\n",
                       "        result.append(total)\n        total = total * credence\n")
_C_LAST = _C.replace("    return result\n", "    return total\n")
_C_EARLY = _C.replace("    return result\n", "        return result\n")

# exercise 2: outputs; the first case is 10 steps with a boost of 1.3, then 5 steps, a boost of 1 at which gathering
# never pays, a big boost, one step, and no steps at all
OUTPUT_CASES = [(STEPS, BOOST), (5, 1.3), (4, 1), (3, 2.5), (1, 2), (0, 1.5)]
_O_START = "def outputs(steps, boost):\n    ...\n"
_O = ("def outputs(steps, boost):\n"
      "    result = []\n"
      "    for gather in range(steps + 1):\n"
      "        result.append((steps - gather) * boost ** gather)\n"
      "    return result\n")
_O_SHORT = _O.replace("range(steps + 1)", "range(steps)")
_O_ALL = _O.replace("(steps - gather) * boost ** gather", "steps * boost ** gather")
_O_TIMES = _O.replace("(steps - gather) * boost ** gather", "(steps - gather) * boost * gather")
_O_PLUS = _O.replace("(steps - gather) * boost ** gather", "(steps - gather) + boost ** gather")
_O_ONCE = _O.replace("(steps - gather) * boost ** gather", "(steps - gather) * boost")
_O_EARLY = _O.replace("    return result\n", "        return result\n")


EXERCISES = {
    "chain": FunctionBox(
        name="chain",
        picture=chain_picture,
        function="chain",
        start=_C_START,
        solution=_C,
        visible=[CHAIN_CASES[0]],
        hidden=CHAIN_CASES[1:],
        first=[_DOTS],
        show=show_chain,
        mistakes=[
            Mistake("zero", wrong=_C_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: your `total` starts at 0, and 0 times any credence stays 0. Before any premise, nothing has to hold yet: start `total` at 1.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `total` begint bij 0, en 0 keer een subjectieve kans blijft 0. Vóór de eerste premisse hoeft er nog niets te kloppen: laat `total` bij 1 beginnen."),
            Mistake("add", wrong=_C_ADD,
                    en="`{call}` gives {got}, and it should give {want}: you add the credences, so the chance grows above 1. All premises must hold, each given the ones before: multiply `total` by each `credence`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt de subjectieve kansen op, dus de kans groeit boven 1. Alle premissen moeten kloppen, elk gegeven de vorige: vermenigvuldig `total` met elke `credence`."),
            Mistake("each", wrong=_C_EACH,
                    en="`{call}` gives {got}, and it should give {want}: you append each `credence` itself, the chance of one premise given the ones before. Append `total`, the chance that this premise and all before it hold.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je voegt elke `credence` zelf toe, de kans op één premisse gegeven de vorige. Voeg `total` toe, de kans dat deze premisse en alle premissen ervoor kloppen."),
            Mistake("before", wrong=_C_BEFORE,
                    en="`{call}` gives {got}, and it should give {want}: your list starts with 1, because you append `total` before you take in the premise. First update `total`, then append it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je lijst begint met 1, omdat je `total` toevoegt voordat je de premisse meeneemt. Werk eerst `total` bij en voeg het daarna toe."),
            Mistake("last", wrong=_C_LAST,
                    en="`{call}` gives {got}, and it should give {want}: you return only the last chance, for all premises together. Return the list `result`, with a chance for each premise.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je geeft alleen de laatste kans terug, voor alle premissen samen. Geef de lijst `result` terug, met een kans voor elke premisse."),
            Mistake("early", wrong=_C_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your `return` stands inside the loop, so the function stops after the first premise. Put `return result` after the loop, level with `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `return` staat in de loop, dus de functie stopt na de eerste premisse. Zet `return result` na de loop, op gelijke hoogte met `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: start `total` at 1, and for each `credence` in `credences`, update `total` and append it to a list `result`.",
                      "nl": "Vervang de `...`: laat `total` bij 1 beginnen, werk `total` bij voor elke `credence` in `credences`, en voeg het telkens toe aan een lijst `result`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Begin `result` as an empty list and `total` at 1. In a `for` loop over `credences`, update `total` with each `credence`, as in the worked example, and append `total`; after the loop, return `result`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Begin `result` als een lege lijst en `total` bij 1. Werk `total` in een `for`-loop over `credences` bij met elke `credence`, zoals in het uitgewerkte voorbeeld, en voeg `total` toe; geef na de loop `result` terug."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use the `credences` it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `credences` die ze krijgt?"},
            "correct": {"en": "Your `chain` is right in all {n} cases. Under the check, your rings lie on the course's line.",
                        "nl": "Je `chain` klopt in alle {n} gevallen. Onder de check liggen jouw ringen op de lijn van de course."},
        },
    ),
    "outputs": FunctionBox(
        name="outputs",
        picture=outputs_picture,
        function="outputs",
        start=_O_START,
        solution=_O,
        visible=[OUTPUT_CASES[0]],
        hidden=OUTPUT_CASES[1:],
        first=[_DOTS],
        show=show_outputs,
        mistakes=[
            Mistake("short", wrong=_O_SHORT,
                    en="`{call}` gives {got}, and it should give {want}: your list is one too short. `range(steps)` stops before `steps`, so the case where the agent gathers for all its steps is missing. Let `gather` run up to and including `steps`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je lijst is er één te kort. `range(steps)` stopt voor `steps`, dus het geval waarin de agent alle stappen aan verzamelen besteedt, ontbreekt. Laat `gather` lopen tot en met `steps`."),
            Mistake("all", wrong=_O_ALL,
                    en="`{call}` gives {got}, and it should give {want}: your agent also works during the steps it spends gathering. Only the `steps - gather` steps that are left make something.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je agent werkt ook in de stappen die hij aan verzamelen besteedt. Alleen de `steps - gather` stappen die over zijn, maken iets."),
            Mistake("times", wrong=_O_TIMES,
                    en="`{call}` gives {got}, and it should give {want}: you multiply `boost` by `gather`, so an agent that gathers nothing makes nothing. Each step of gathering multiplies what a work step makes by `boost` once more: `boost` to the power `gather`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je vermenigvuldigt `boost` met `gather`, dus een agent die niets verzamelt, maakt niets. Elke stap verzamelen vermenigvuldigt wat een werkstap maakt nog een keer met `boost`: `boost` tot de macht `gather`."),
            Mistake("plus", wrong=_O_PLUS,
                    en="`{call}` gives {got}, and it should give {want}: you add what one work step makes to the number of work steps. Each of the `steps - gather` work steps makes that much: multiply them.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt op wat één werkstap maakt bij het aantal werkstappen. Elk van de `steps - gather` werkstappen maakt zoveel: vermenigvuldig ze."),
            Mistake("once", wrong=_O_ONCE,
                    en="`{call}` gives {got}, and it should give {want}: each work step gets `boost` once, however long the agent gathered. Each step of gathering multiplies it by `boost` again: `boost` to the power `gather`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke werkstap krijgt `boost` één keer, hoe lang de agent ook verzamelde. Elke stap verzamelen vermenigvuldigt het opnieuw met `boost`: `boost` tot de macht `gather`."),
            Mistake("early", wrong=_O_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your `return` stands inside the loop, so the function stops after the first number of gathering steps. Put `return result` after the loop, level with `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `return` staat in de loop, dus de functie stopt na het eerste aantal stappen verzamelen. Zet `return result` na de loop, op gelijke hoogte met `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: for each number `gather` from 0 up to and including `steps`, put what the agent makes in a list `result`: the work steps that are left, each of which every step of gathering made `boost` times as productive.",
                      "nl": "Vervang de `...`: zet voor elk aantal `gather` van 0 tot en met `steps` wat de agent maakt in een lijst `result`: de werkstappen die over zijn, die elke stap verzamelen `boost` keer zo productief heeft gemaakt."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Begin `result` as an empty list. In a `for` loop with `gather` from 0 up to and including `steps`, append the work steps that are left, `steps - gather`, times what one work step makes after `gather` steps of gathering, each of which multiplies it by `boost`; after the loop, return `result`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Begin `result` als een lege lijst. Voeg in een `for`-loop met `gather` van 0 tot en met `steps` de werkstappen die over zijn, `steps - gather`, toe, keer wat één werkstap maakt na `gather` stappen verzamelen, die het elk met `boost` vermenigvuldigen; geef na de loop `result` terug."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use the `steps` and `boost` it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `steps` en `boost` die ze krijgt?"},
            "correct": {"en": "Your `outputs` is right in all {n} cases. Under the check, your rings lie on the course's line.",
                        "nl": "Je `outputs` klopt in alle {n} gevallen. Onder de check liggen jouw ringen op de lijn van de course."},
        },
    ),
}

CASES = [
    ("chain", _C_START, "empty", None),
    ("chain", _C, "correct", None),
    ("chain", "def chain(credences):\n    result = []\n    for k in range(len(credences)):\n        total = 1\n        for credence in credences[:k + 1]:\n            total *= credence\n        result.append(total)\n    return result\n", "correct", None),
    ("chain", "def chain(credences):\n    result = []\n    for credence in credences:\n        result.append(credence * (result[-1] if result else 1))\n    return result\n", "correct", None),
    ("chain", _C_ZERO, "wrong", "zero"),
    ("chain", _C_ADD, "wrong", "add"),
    ("chain", _C_EACH, "wrong", "each"),
    ("chain", _C_BEFORE, "wrong", "before"),
    ("chain", _C_LAST, "wrong", "last"),
    ("chain", _C_EARLY, "wrong", "early"),
    ("chain", "def chain(credences):\n    return " + repr(chain(CARLSMITH)) + "\n", "almost", None),
    ("outputs", _O_START, "empty", None),
    ("outputs", _O, "correct", None),
    ("outputs", "def outputs(steps, boost):\n    return [(steps - g) * boost ** g for g in range(steps + 1)]\n", "correct", None),
    ("outputs", "def outputs(steps, boost):\n    result = []\n    gather = 0\n    while gather <= steps:\n        result.append((steps - gather) * pow(boost, gather))\n        gather += 1\n    return result\n", "correct", None),
    ("outputs", _O_SHORT, "wrong", "short"),
    ("outputs", _O_ALL, "wrong", "all"),
    ("outputs", _O_TIMES, "wrong", "times"),
    ("outputs", _O_PLUS, "wrong", "plus"),
    ("outputs", _O_ONCE, "wrong", "once"),
    ("outputs", _O_EARLY, "wrong", "early"),
    ("outputs", "def outputs(steps, boost):\n    return " + repr(outputs(STEPS, BOOST)) + "\n", "almost", None),
]
