# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-nines: nines of reliability. The learner writes nines, the nines of reliability
k = -log10(1 - p) of Hendrycks, Introduction to AI Safety, Ethics, and Society (2024), 4.2, and task_success, the
chance that a task of independent steps succeeds. A demo moves the reliability from 90% to 99.999%."""
import contextlib
import copy
import html
import io
import re

import numpy as np

from everything_ai import chart, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

# --- the data --------------------------------------------------------------------------------------------------------

PERCENTS = [90, 99, 99.9, 99.99, 99.999]           # the demo's slider, in %
RELIABILITIES = [0.9, 0.99, 0.999, 0.9999, 0.99999]  # the same as decimals
REQUESTS = 1000000                                 # requests a day, in the demo and the texts
STEPS = 100                                        # the steps of the demo's task
YEAR_MINUTES = 365 * 24 * 60                       # minutes in a year, as the SRE book's availability table counts
SHOWN = [0.5, 0.9, 0.99, 0.999, 0.9999]            # the reliabilities under the check of exercise 1


def nines(p):
    """What the learner writes in Exercise 1: the nines of reliability, k = -log10(1 - p)."""
    return -np.log10(1 - p)


def task_success(p, steps):
    """What the learner writes in Exercise 2: the chance that all steps succeed, each with chance p, independently."""
    chance = 1
    for step in range(steps):
        chance = chance * p
    return chance


def failures(p, requests: int = REQUESTS) -> int:
    """How many of `requests` fail at reliability p."""
    return round(requests * (1 - p))


def steps_to_half(p) -> int:
    """The smallest number of steps at which a task of independent steps of reliability p succeeds less than half
    the time."""
    n, chance = 0, 1.0
    while chance >= 0.5:
        n, chance = n + 1, chance * p
    return n


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
    if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError, OverflowError):
        return None


def _shown(v, lang: str = "en", digits: int = 4) -> str:
    """A result as the pictures show it: a number with at most `digits` decimals, else what Python shows."""
    f = _number(v)
    if f is None:
        text = repr(v)
        return text if len(text) <= 40 else text[:37] + "..."
    if f != f:
        return "nan"
    if f in (float("inf"), float("-inf")):
        return "inf" if f > 0 else "-inf"
    text = f"{f:.{digits}f}".rstrip("0").rstrip(".")
    text = "0" if text == "-0" else text
    return text.replace(".", ",") if lang == "nl" else text


def _value(v) -> str:
    """A result as the check's messages show it: a number with at most six decimals, a tiny one in e-notation, else
    what Python shows."""
    f = _number(v)
    if f is None or f != f or f in (float("inf"), float("-inf")):
        return repr(v) if f is None else str(f)
    if f != 0 and abs(f) < 1e-6:
        return f"{f:.3g}"
    text = f"{f:.6f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _same(a, b) -> bool:
    f = _number(a)
    return f is not None and abs(f - float(b)) <= 1e-9 + 1e-6 * abs(float(b))


def _bar(value, top, fill: str, width: int = 140) -> str:
    """A horizontal bar for a value from 0 to top."""
    f = _number(value)
    if f is None or f != f or top <= 0:
        return ""
    w = max(0.0, min(1.0, f / top)) * width
    return (f'<span style="display:inline-block;width:{width}px;height:10px;background:var(--ea-inset);'
            f'vertical-align:middle"><span style="display:block;width:{w:.1f}px;height:10px;background:{fill}">'
            f'</span></span>')


def _whole(v, lang: str) -> str:
    """A whole number as the text writes it: 1,000 in English, 1000 in Dutch, which uses a dot only from 10.000."""
    n = int(round(float(v)))
    if lang == "en":
        return f"{n:,}"
    return f"{n:,}".replace(",", ".") if abs(n) >= 10000 else str(n)


def _percent(p, lang: str) -> str:
    """A reliability as a percentage: 0.999 as 99.9%."""
    text = f"{p * 100:.3f}".rstrip("0").rstrip(".") + "%"
    return text.replace(".", ",") if lang == "nl" else text


def _chance(x, lang: str) -> str:
    """A chance of success as a percentage with one decimal; a tiny one with two significant digits."""
    text = f"{x * 100:.1f}%" if x * 100 >= 0.1 else f"{x * 100:.2g}%"
    return text.replace(".", ",") if lang == "nl" else text


_UNITS = {"en": ("days", "hours", "minutes"), "nl": ("dagen", "uur", "minuten")}


def downtime(p, lang: str) -> str:
    """The time down in a year at reliability p, counted in time, with three significant digits, as in the SRE book's
    availability table: 36.5 days, 8.76 hours, 5.26 minutes."""
    minutes = YEAR_MINUTES * (1 - p)
    days, hours, mins = _UNITS[lang]
    value, unit = (minutes / 1440, days) if minutes >= 1440 else (minutes / 60, hours) if minutes >= 60 else (minutes, mins)
    text = f"{value:.3g} {unit}"
    return text.replace(".", ",") if lang == "nl" else text


_DEMO_WORDS = {
    "en": {"title": "Reliability {pct}: what it gives",
           "rows": ["nines of reliability", "failed requests in 1,000,000", "down in a year, counted in time",
                    "a task of 100 steps succeeds"],
           "said": "At {pct}, {k}: {fails} failed requests in 1,000,000, {down} down in a year, and a task of "
                   "100 steps succeeds {ok} of the time.",
           "steps": "steps", "chance": "chance of success", "nine": "nine", "nines": "nines",
           "note": "The curve: the chance that a task of independent steps succeeds, by its number of steps, at this "
                   "reliability. The dashed line is one half; the dot is the task of 100 steps."},
    "nl": {"title": "Betrouwbaarheid {pct}: wat het oplevert",
           "rows": ["negens van betrouwbaarheid", "mislukte verzoeken op de 1.000.000",
                    "onbeschikbaar per jaar, geteld in tijd", "een taak van 100 stappen slaagt"],
           "said": "Bij {pct}, {k}: {fails} mislukte verzoeken op de 1.000.000, {down} onbeschikbaar per jaar, "
                   "en een taak van 100 stappen slaagt in {ok} van de gevallen.",
           "steps": "stappen", "chance": "kans op succes", "nine": "negen", "nines": "negens",
           "note": "De kromme: de kans dat een taak van onafhankelijke stappen slaagt, uitgezet tegen het aantal stappen, bij "
                   "deze betrouwbaarheid. De gestippelde lijn is de helft; de stip is de taak van 100 stappen."},
}


def demo_frame(v, lang: str) -> tuple[str, str]:
    """The demo: at a reliability of v %, the nines, the failures, the downtime, and a task of 100 steps."""
    v = min(PERCENTS, key=lambda m: abs(m - float(v)))
    p = RELIABILITIES[PERCENTS.index(v)]
    w = _DEMO_WORDS[lang]
    pct, k, ok = _percent(p, lang), _shown(nines(p), lang, 2), task_success(p, STEPS)
    values = [k, _whole(failures(p), lang), downtime(p, lang), _chance(ok, lang)]
    rows = [([(html.escape(name), ""), (_code(value), "text-align:right")], "") for name, value in zip(w["rows"], values)]
    panel = chart.Panel(x=(0, 200), y=(0, 1), xlabel=w["steps"], ylabel=w["chance"], xticks=[0, 50, 100, 150, 200],
                        yticks=[0, 0.5, 1])
    panel.line([0, 200], [0.5, 0.5], role="ink-3", width=1.5, dashed=True)
    panel.line(list(range(201)), [p ** n for n in range(201)], role="ink")
    panel.points([STEPS], [ok], role="ink")
    title = w["title"].format(pct=pct)
    said = w["said"].format(pct=pct, k=f"{k} {w['nine'] if k == '1' else w['nines']}", fails=values[1], down=values[2],
                            ok=values[3])
    body = _table(rows, ["", ""]) + chart.figure([panel], alt=title + ". " + said, width=360, height=220) + _note(w["note"])
    return yours.figure(title, body, alt=title + ". " + said), said


# --- the pictures under the checks -----------------------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _all gives for a function that stopped with an error


def _all(fn, calls: list):
    """The learner's function on copies of each list of arguments in `calls`, all within one time limit of a check;
    _STOPPED when it stops with an error, and the Timeout passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
            return with_time_limit(lambda: [fn(*(copy.deepcopy(a) for a in args)) for args in calls])
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


_NINES_WORDS = {
    "en": {"title": "Your nines beside the course's",
           "head": ["reliability", "failed in 1,000,000", "the course's", "yours", ""],
           "stopped": "error",
           "note": "Upper bar the course's, lower bar yours, from 0 to 5 nines. From 99% on, each row has ten times "
                   "fewer failures than the one above it, and one nine more. A dashed row is one where your nines "
                   "gives something other than the course's."},
    "nl": {"title": "Jouw functie nines naast die van de course",
           "head": ["betrouwbaarheid", "mislukt op de 1.000.000", "van de course", "de jouwe", ""],
           "stopped": "fout",
           "note": "Bovenste balk die van de course, onderste balk de jouwe, van 0 tot 5 negens. Vanaf 99% heeft elke "
                   "rij tien keer zo weinig mislukte verzoeken als de rij erboven, en één negen meer. Een gestippelde rij "
                   "is er een waar jouw functie nines iets anders geeft dan die van de course."},
}


def nines_picture(seen) -> str:
    """Five reliabilities with their failures in 1,000,000, the course's nines and the learner's, as numbers and bars."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _NINES_WORDS[seen.lang]
    lang = seen.lang
    got = _all(fn, [(p,) for p in SHOWN])
    rows = []
    for i, p in enumerate(SHOWN):
        want = nines(p)
        if got is _STOPPED:
            shown, off, mine = html.escape(w["stopped"]), True, None
        else:
            mine = got[i]
            shown, off = _code(_shown(mine, lang)), not _same(mine, want)
        bars = _bar(want, 5, "var(--ea-ink-3)") + "<br>" + _bar(mine, 5, "var(--ea-ink)")
        rows.append(([(_code(_percent(p, lang)), ""), (_code(_whole(failures(p), lang)), "text-align:right"),
                      (_code(_shown(want, lang)), "text-align:right"), (shown, "text-align:right;font-weight:600"),
                      (bars, "")], _OFF if off else ""))
    body = _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


success_picture = yours.curve(
    "task_success", list(range(201)), at=lambda n: (0.99, n), role="ink", y=(0, 1),
    title={"en": "Your task_success beside the course's, at 99% a step",
           "nl": "Jouw task_success naast die van de course, bij 99% per stap"},
    xlabel={"en": "steps", "nl": "stappen"}, ylabel={"en": "chance of success", "nl": "kans op succes"})


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["nines", "task_success", "risk", "available"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_NUMPY = Mistake("no-numpy", error=lambda e: isinstance(e, NameError) and "'np'" in str(e),
                 en="Your function uses `np`, and the box no longer imports it. Keep the line `import numpy as np` at the top of the box.",
                 nl="Je functie gebruikt `np`, en het vak importeert dat niet meer. Laat de regel `import numpy as np` bovenaan het vak staan.")

# exercise 1: nines; the first case is three nines
NINES_CASES = [(0.999,), (0.9,), (0.5,), (0.99999,), (0.95,), (0.75,)]
_N_START = "import numpy as np\n\n\ndef nines(p):\n    ...\n"
_N = "import numpy as np\n\n\ndef nines(p):\n    return -np.log10(1 - p)\n"
_N_NO_MINUS = _N.replace("return -np.log10", "return np.log10")
_N_LN = _N.replace("np.log10", "np.log")
_N_LOG2 = _N.replace("np.log10", "np.log2")
_N_OF_P = _N.replace("np.log10(1 - p)", "np.log10(p)")
_N_FAILURE = _N.replace("-np.log10(1 - p)", "1 - p")
_N_COUNT = _N.replace("-np.log10(1 - p)", 'str(p).count("9")')

# exercise 2: task_success; the first case is 100 steps at 99%
SUCCESS_CASES = [(0.99, 100), (0.9, 2), (0.5, 3), (0.8, 1), (1.0, 50), (0.95, 20)]
_S_START = "def task_success(p, steps):\n    ...\n"
_S = ("def task_success(p, steps):\n"
      "    chance = 1\n"
      "    for step in range(steps):\n"
      "        chance = chance * p\n"
      "    return chance\n")
_S_ZERO = _S.replace("    chance = 1\n", "    chance = 0\n")
_S_PLUS = _S.replace("chance = chance * p", "chance = chance + p")
_S_FAIL = _S.replace("chance = chance * p", "chance = chance * (1 - p)")
_S_TIMES = "def task_success(p, steps):\n    return p * steps\n"
_S_EARLY = _S.replace("    return chance\n", "        return chance\n")
_S_FEWER = _S.replace("range(steps)", "range(1, steps)")


EXERCISES = {
    "nines": FunctionBox(
        name="nines",
        picture=nines_picture,
        function="nines",
        start=_N_START,
        solution=_N,
        visible=[NINES_CASES[0]],
        hidden=NINES_CASES[1:],
        first=[_DOTS],
        show=_value,
        mistakes=[
            _NUMPY,
            Mistake("no-minus", wrong=_N_NO_MINUS,
                    en="`{call}` gives {got}, and it should give {want}: the logarithm of a chance below 1 is negative. Put a minus sign in front of it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de logaritme van een kans onder 1 is negatief. Zet er een minteken voor."),
            Mistake("ln", wrong=_N_LN,
                    en="`{call}` gives {got}, and it should give {want}: that is the natural logarithm, `np.log`. The nines count factors of 10: use the base-10 logarithm, `np.log10`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is de natuurlijke logaritme, `np.log`. De negens tellen factoren 10: gebruik de logaritme met grondtal 10, `np.log10`."),
            Mistake("log2", wrong=_N_LOG2,
                    en="`{call}` gives {got}, and it should give {want}: that is the base-2 logarithm, which counts factors of 2. The nines count factors of 10: use `np.log10`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is de logaritme met grondtal 2, die factoren 2 telt. De negens tellen factoren 10: gebruik `np.log10`."),
            Mistake("of-p", wrong=_N_OF_P,
                    en="`{call}` gives {got}, and it should give {want}: you take the logarithm of the reliability itself. The nines are about the chance of failure, one minus `p`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je neemt de logaritme van de betrouwbaarheid zelf. De negens gaan over de kans op falen, één min `p`."),
            Mistake("failure", wrong=_N_FAILURE,
                    en="`{call}` gives {got}, and it should give {want}: that is the chance of failure, one minus `p`. The nines are minus its base-10 logarithm.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is de kans op falen, één min `p`. De negens zijn min de logaritme daarvan met grondtal 10."),
            Mistake("count", wrong=_N_COUNT,
                    en="`{call}` gives {got}, and it should give {want}: you count the nines in the number, which works only for whole nines such as `0.999`. The formula also gives the nines in between.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt de negens in het getal, en dat werkt alleen bij hele negens zoals `0.999`. De formule geeft ook de negens daartussen."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: return minus the base-10 logarithm of the chance of failure.",
                      "nl": "Vervang de `...`: geef min de logaritme met grondtal 10 van de kans op falen terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Take the chance of failure, one minus `p`, take its base-10 logarithm with `np.log10`, and put a minus sign in front.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Neem de kans op falen, één min `p`, neem daarvan de logaritme met grondtal 10 met `np.log10`, en zet er een minteken voor."},
            "almost": {"en": "Right for `nines(0.999)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `p` it gets?",
                       "nl": "Goed voor `nines(0.999)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `p` die ze krijgt?"},
            "correct": {"en": "Your `nines` is right in all {n} cases. Under the check, your nines stand beside the course's, for five reliabilities.",
                        "nl": "Je `nines` klopt in alle {n} gevallen. Onder de check staan jouw negens naast die van de course, voor vijf betrouwbaarheden."},
        },
    ),
    "task_success": FunctionBox(
        name="task_success",
        picture=success_picture,
        function="task_success",
        start=_S_START,
        solution=_S,
        visible=[SUCCESS_CASES[0]],
        hidden=SUCCESS_CASES[1:],
        first=[_DOTS],
        show=_value,
        mistakes=[
            Mistake("zero-start", wrong=_S_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: your `chance` starts at 0, and 0 times anything stays 0. Start at 1: before the first step, nothing has gone wrong yet.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `chance` begint bij 0, en 0 maal iets blijft 0. Begin bij 1: voor de eerste stap is er nog niets misgegaan."),
            Mistake("plus", wrong=_S_PLUS,
                    en="`{call}` gives {got}, and it should give {want}: you add `p` at each step. The chances of the steps multiply, as all of them must succeed.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt bij elke stap `p` op. De kansen van de stappen worden vermenigvuldigd, want ze moeten allemaal slagen."),
            Mistake("fail", wrong=_S_FAIL,
                    en="`{call}` gives {got}, and it should give {want}: you multiply by the chance that a step fails. The task succeeds when every step succeeds: multiply by `p`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je vermenigvuldigt met de kans dat een stap mislukt. De taak slaagt als elke stap slaagt: vermenigvuldig met `p`."),
            Mistake("times", wrong=_S_TIMES,
                    en="`{call}` gives {got}, and it should give {want}: that is `p` times the number of steps, which grows with the steps. The chance must shrink at each step: multiply `chance` by `p` once for each step.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is `p` maal het aantal stappen, en dat groeit met de stappen. De kans moet bij elke stap kleiner worden: vermenigvuldig `chance` één keer per stap met `p`."),
            Mistake("early", wrong=_S_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your `return` is inside the loop, so the function stops after the first step. Move `return` back, level with `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `return` staat in de loop, dus de functie stopt na de eerste stap. Zet `return` terug, op gelijke hoogte met `for`."),
            Mistake("one-fewer", wrong=_S_FEWER,
                    en="`{call}` gives {got}, and it should give {want}: your loop runs one step too few. `range(steps)` runs exactly `steps` times.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je loop doet één stap te weinig. `range(steps)` loopt precies `steps` keer."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: start `chance` at 1, multiply it by `p` once for each step, and return it.",
                      "nl": "Vervang de `...`: begin `chance` bij 1, vermenigvuldig hem één keer per stap met `p`, en geef hem terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Start `chance` at 1, multiply it by `p` in a loop over `range(steps)`, and return `chance` after the loop.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Begin `chance` bij 1, vermenigvuldig hem met `p` in een loop over `range(steps)`, en geef `chance` na de loop terug."},
            "almost": {"en": "Right for `task_success(0.99, 100)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `p` and the `steps` it gets?",
                       "nl": "Goed voor `task_success(0.99, 100)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `p` en de `steps` die ze krijgt?"},
            "correct": {"en": "Your `task_success` is right in all {n} cases. Under the check, your curve lies over the course's, for steps that each succeed 99% of the time.",
                        "nl": "Je `task_success` klopt in alle {n} gevallen. Onder de check ligt jouw kromme over die van de course, voor stappen die elk in 99% van de gevallen slagen."},
        },
    ),
}

CASES = [
    ("nines", _N_START, "empty", None),
    ("nines", _N, "correct", None),
    ("nines", "import numpy as np\n\n\ndef nines(p):\n    failure = 1 - p\n    return -np.log10(failure)\n", "correct", None),
    ("nines", "import math\n\n\ndef nines(p):\n    return -math.log10(1 - p)\n", "correct", None),
    ("nines", _N_NO_MINUS, "wrong", "no-minus"),
    ("nines", _N_LN, "wrong", "ln"),
    ("nines", _N_LOG2, "wrong", "log2"),
    ("nines", _N_OF_P, "wrong", "of-p"),
    ("nines", _N_FAILURE, "wrong", "failure"),
    ("nines", _N_COUNT, "wrong", "count"),
    ("nines", "import numpy as np\n\n\ndef nines(p):\n    return 3\n", "almost", None),
    ("nines", "def nines(p):\n    return -np.log10(1 - p)\n", "error", "no-numpy"),
    ("task_success", _S_START, "empty", None),
    ("task_success", _S, "correct", None),
    ("task_success", "def task_success(p, steps):\n    return p ** steps\n", "correct", None),
    ("task_success", _S_ZERO, "wrong", "zero-start"),
    ("task_success", _S_PLUS, "wrong", "plus"),
    ("task_success", _S_FAIL, "wrong", "fail"),
    ("task_success", _S_TIMES, "wrong", "times"),
    ("task_success", _S_EARLY, "wrong", "early"),
    ("task_success", _S_FEWER, "wrong", "one-fewer"),
    ("task_success", "def task_success(p, steps):\n    return 0.99 ** 100\n", "almost", None),
]
