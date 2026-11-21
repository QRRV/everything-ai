# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-ai-welfare: can an AI system matter morally? The learner writes chain, the chance that
every step of a chain holds when the steps are independent, with the chances of Long and colleagues (2024), and costs,
a made-up model of the two mistakes: the cost to expect when you take care of a system that may matter morally, and
when you ignore it. A demo changes the cost of care in vain and shows the turning point."""
import contextlib
import copy
import html
import io
import re

from everything_ai import chart, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, same_value, with_time_limit

# --- the data --------------------------------------------------------------------------------------------------------

# Long and colleagues (2024): an example of three steps of about 25% each, and their own estimate of the same three
# steps: sentience suffices for moral patienthood, certain computations suffice for sentience, and some near-future AI
# systems can do them
LONG_EXAMPLE = [0.25, 0.25, 0.25]
LONG_OWN = [0.9, 0.5, 0.5]
# made-up: a chance of 2% that a system matters morally, care in vain costs 1, ignoring a being that matters 100
CHANCE, OVER, UNDER = 0.02, 1, 100
DEMO_OVER = [1, 5, 10, 25, 50, 100]   # the stops of the demo's slider: what care in vain costs


def chain(chances):
    """What the learner writes in Exercise 1: the chance that every step holds, the steps independent."""
    result = 1
    for chance in chances:
        result = result * chance
    return result


def costs(chance, over, under):
    """What the learner writes in Exercise 2: the cost to expect when you take care, and when you ignore."""
    care = (1 - chance) * over
    ignore = chance * under
    return [care, ignore]


def turning(over, under=UNDER):
    """The chance from which on ignoring costs more than taking care: where chance * under = (1 - chance) * over."""
    return over / (over + under)


def last_below(limit=0.225, values=DEMO_OVER):
    """The largest value of the slider at which the turning point still lies below `limit`."""
    return max(o for o in values if turning(o) < limit)


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
    if isinstance(v, bool):
        return None
    if not isinstance(v, (int, float)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError, OverflowError):
        return None


def _value(v, digits: int = 6, limit: int = 70) -> str:
    """A result as the check shows it: a number with at most `digits` decimals, a text in quotes, a list of them, else
    what Python shows, cut short when it is longer than `limit`."""
    if isinstance(v, (list, tuple)):
        text = "[" + ", ".join(_value(x, digits, 10 ** 6) for x in v) + "]"
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


def show_value(v) -> str:
    return _value(v, 6, 200)


def same_list(got, want) -> bool:
    """Whether the learner's two costs are the course's, as a list or a tuple."""
    if isinstance(got, tuple):
        got = list(got)
    return isinstance(got, list) and same_value(got, list(want))


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


def _pct(v: float, lang: str) -> str:
    """A number of percent as the lesson writes it: at most two decimals, with a decimal comma in Dutch."""
    text = f"{v:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", ",") if lang == "nl" else text


# --- the demo --------------------------------------------------------------------------------------------------------

_DEMO_WORDS = {
    "en": {"title": "Taking care or ignoring, when care in vain costs {o}",
           "x": "chance that the system matters morally (percent)", "y": "cost to expect",
           "care": "taking care", "ignore": "ignoring", "turn": "turning point {t}%",
           "said": "When care in vain costs {o}, the turning point lies at {t}%: from that chance on, ignoring costs "
                   "more than taking care.",
           "note": "A made-up model, this lesson's own. Ignoring a being that matters costs 100. The solid line is "
                   "what you can expect to lose by taking care, the dashed line by ignoring; where they cross is the "
                   "turning point."},
    "nl": {"title": "Zorg dragen of negeren, als vergeefse zorg {o} kost",
           "x": "kans dat het systeem moreel ertoe doet (procent)", "y": "te verwachten kosten",
           "care": "zorg dragen", "ignore": "negeren", "turn": "omslagpunt {t}%",
           "said": "Als vergeefse zorg {o} kost, ligt het omslagpunt bij {t}%: vanaf die kans kost negeren meer dan "
                   "zorg dragen.",
           "note": "Een verzonnen model, van deze les zelf. Een wezen negeren dat ertoe doet, kost 100. De "
                   "doorgetrokken lijn is wat je kunt verwachten te verliezen als je zorg draagt, de gestippelde lijn "
                   "als je negeert; waar ze elkaar kruisen, ligt het omslagpunt."},
}


def demo_said(over, lang: str) -> str:
    return _DEMO_WORDS[lang]["said"].format(o=over, t=_pct(100 * turning(over), lang))


def demo_frame(v, lang: str) -> tuple[str, str]:
    """The demo: what you can expect to lose by taking care and by ignoring, over the chance that the system matters,
    with the turning point where the two lines cross."""
    over = min(DEMO_OVER, key=lambda o: abs(o - float(v)))
    w = _DEMO_WORDS[lang]
    t = 100 * turning(over)
    panel = chart.Panel(x=(0, 100), y=(0, 114), xlabel=w["x"], ylabel=w["y"], xticks=[0, 25, 50, 75, 100],
                        yticks=[0, 25, 50, 75, 100])
    panel.line([0, 100], [over, 0], role="ink", width=2.2)
    panel.line([0, 100], [0, UNDER], role="ink-2", width=2.2, dashed=True)
    panel.line([t, t], [0, 104], role="ink-3", width=1.2, dashed=True)
    panel.points([t], [t * UNDER / 100], role="ink", r=4)
    panel.text(t + 1.5, 100, w["turn"].format(t=_pct(t, lang)), role="ink-2", anchor="start", halo=True)
    panel.text(98, 90, w["ignore"], role="ink-2", anchor="end", halo=True)
    if over <= 10:   # a flat line: the words above its middle; else above its start, clear of the dashed line
        panel.text(70, over * 0.3 + 6, w["care"], role="ink", anchor="middle", halo=True)
    else:
        panel.text(2, over + 7, w["care"], role="ink", anchor="start", halo=True)
    title = w["title"].format(o=over)
    said = demo_said(over, lang)
    svg = chart.figure([panel], alt=title + ". " + said, width=380, height=240)
    if lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _note(said) + _note(w["note"])
    return yours.figure(title, body, alt=title + ". " + said), said


# --- the pictures under the checks -----------------------------------------------------------------------------------

_CHAIN_WORDS = {
    "en": {"title": "Your chain of three steps",
           "x": "step", "y": "chance", "all": "all", "so_far": "all steps so far",
           "head": ["chances", "the course's", "yours"], "stopped": "error",
           "note": "The chart is the first case of the check, Long and colleagues' own estimate. Each bar is the "
                   "chance of one step; the line is the chance that all steps so far hold. On the right, the course's "
                   "result is a dot and yours a ring. The table is every case; a dashed row is one where your "
                   "function gives something other than the course's."},
    "nl": {"title": "Jouw keten van drie stappen",
           "x": "stap", "y": "kans", "all": "alle", "so_far": "alle stappen tot nu toe",
           "head": ["chances", "van de course", "de jouwe"], "stopped": "fout",
           "note": "De grafiek is het eerste geval van de check, de eigen schatting van Long en collega's. Elke balk "
                   "is de kans van één stap; de lijn is de kans dat alle stappen tot nu toe kloppen. Rechts is de "
                   "uitkomst van de course een stip en de jouwe een ring. De tabel bevat elk geval; een gestippelde "
                   "rij is er een waar jouw functie iets anders geeft dan die van de course."},
}

CHAIN_CASES = [(LONG_OWN,), (LONG_EXAMPLE,), ([0.5],), ([0.5, 0.8],), ([1.0, 1.0, 0.1],), ([0.9, 0.0, 0.8],)]


def chain_picture(seen) -> str:
    """The first case: a bar per step, the chance that all steps so far hold as a line, and the course's result as a
    dot and the learner's as a ring; every case in a table."""
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
            shown_, off = _code(show_value(got[i])), not same_value(got[i], want)
        rows.append(([(_code(_value(case[0])), ""), (_code(show_value(want)), ""), (shown_, "font-weight:600")],
                     _OFF if off else ""))
    steps = LONG_OWN
    so_far = [chain(steps[:k + 1]) for k in range(len(steps))]
    want = chain(steps)
    panel = chart.Panel(x=(0.4, 4.6), y=(0, 1.08), xlabel=w["x"], ylabel=w["y"], xticks=[1, 2, 3, 4],
                        xtick_labels=["1", "2", "3", w["all"]], yticks=[0, 0.25, 0.5, 0.75, 1])
    for k, ch in enumerate(steps):
        panel.rect(k + 0.7, 0, k + 1.3, ch, role="ink-3")
    panel.line([1, 2, 3], so_far, role="ink-2", width=1.8)
    panel.points([1, 2, 3], so_far, role="ink-2", r=3)
    panel.text(2.1, 0.62, w["so_far"], role="ink-2", anchor="start", halo=True)
    panel.points([4], [want], role="ink", r=3)
    f = _number(got[0]) if got[0] is not _STOPPED else None
    if f is not None and 0 <= f <= 1.08:
        panel.points([4], [f], role="ink", r=7, hollow=True)
    svg = chart.figure([panel], alt=w["title"], width=380, height=240)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


_COSTS_WORDS = {
    "en": {"title": "Your two costs at a chance of 2%",
           "x": "chance that the system matters morally (percent)", "y": "cost to expect",
           "care": "taking care", "ignore": "ignoring",
           "head": ["chance, over, under", "the course's", "yours"], "stopped": "error",
           "note": "The chart is the first case of the check: care in vain costs 1, ignoring a being that matters "
                   "100. The solid line is what you can expect to lose by taking care, the dashed line by ignoring, "
                   "for chances up to 10%. At 2%, the course's two costs are dots and yours rings. The table is "
                   "every case; a dashed row is one where your function gives something other than the course's."},
    "nl": {"title": "Jouw twee kosten bij een kans van 2%",
           "x": "kans dat het systeem moreel ertoe doet (procent)", "y": "te verwachten kosten",
           "care": "zorg dragen", "ignore": "negeren",
           "head": ["chance, over, under", "van de course", "de jouwe"], "stopped": "fout",
           "note": "De grafiek is het eerste geval van de check: vergeefse zorg kost 1, een wezen negeren dat ertoe "
                   "doet 100. De doorgetrokken lijn is wat je kunt verwachten te verliezen als je zorg draagt, de "
                   "gestippelde lijn als je negeert, voor kansen tot 10%. Bij 2% zijn de twee kosten van de course "
                   "stippen en de jouwe ringen. De tabel bevat elk geval; een gestippelde rij is er een waar jouw "
                   "functie iets anders geeft dan die van de course."},
}

COSTS_CASES = [(CHANCE, OVER, UNDER), (0.5, 1, 1), (0.1, 2, 10), (0.0, 1, 100), (1.0, 1, 100), (0.25, 4, 8)]


def costs_picture(seen) -> str:
    """The first case: the two costs to expect over chances up to 10%, the course's two costs at 2% as dots and the
    learner's as rings; every case in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _COSTS_WORDS[seen.lang]
    got = _all(fn, COSTS_CASES)
    rows = []
    for i, case in enumerate(COSTS_CASES):
        want = costs(*case)
        if got[i] is _STOPPED:
            shown_, off = html.escape(w["stopped"]), True
        else:
            shown_, off = _code(show_value(got[i])), not same_list(got[i], want)
        rows.append(([(_code(_value(list(case))), ""), (_code(show_value(want)), ""), (shown_, "font-weight:600")],
                     _OFF if off else ""))
    want = costs(CHANCE, OVER, UNDER)
    panel = chart.Panel(x=(0, 10), y=(0, 10.5), xlabel=w["x"], ylabel=w["y"], xticks=[0, 2, 4, 6, 8, 10],
                        yticks=[0, 2, 4, 6, 8, 10])
    panel.line([0, 10], [OVER, 0.9 * OVER], role="ink", width=2.2)
    panel.line([0, 10], [0, 0.1 * UNDER], role="ink-2", width=2.2, dashed=True)
    panel.line([2, 2], [0, 10.5], role="ink-3", width=1, dashed=True)
    panel.text(9.8, 8.4, w["ignore"], role="ink-2", anchor="end", halo=True)
    panel.text(9.8, 1.6, w["care"], role="ink", anchor="end", halo=True)
    panel.points([2, 2], [float(want[0]), float(want[1])], role="ink", r=3)
    mine = got[0] if got[0] is not _STOPPED else None
    mine = list(mine) if isinstance(mine, tuple) else mine
    if isinstance(mine, list):
        for v in mine[:2]:
            f = _number(v)
            if f is not None and 0 <= f <= 10.5:
                panel.points([2], [f], role="ink", r=7, hollow=True)
    svg = chart.figure([panel], alt=w["title"], width=380, height=240)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["chain", "costs", "feelings", "mistake"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# exercise 1: chain; the first case is Long and colleagues' own estimate, then their example of three steps of 25%, one
# step, two steps, two certain steps and one of 0.1, and a step that cannot hold
_C_START = "def chain(chances):\n    ...\n"
_C = ("def chain(chances):\n"
      "    result = 1\n"
      "    for chance in chances:\n"
      "        result = result * chance\n"
      "    return result\n")
_C_SUM = _C.replace("result = 1", "result = 0").replace("result * chance", "result + chance")
_C_ZERO = _C.replace("result = 1", "result = 0")
_C_MIN = _C.replace("result = result * chance", "result = min(result, chance)")
_C_LAST = _C.replace("result = result * chance", "result = chance")
_C_NOT = _C.replace("result = result * chance", "result = result * (1 - chance)")
_C_LIST = _C.replace("result = result * chance", "result = result * chances")

# exercise 2: costs; the first case is a chance of 2% with care in vain at 1 and ignoring at 100, then a chance of a
# half with equal costs, then 10%, a chance of 0, a certainty, and a quarter
_K_START = "def costs(chance, over, under):\n    ...\n"
_K = ("def costs(chance, over, under):\n"
      "    care = (1 - chance) * over\n"
      "    ignore = chance * under\n"
      "    return [care, ignore]\n")
_K_SWAP = _K.replace("return [care, ignore]", "return [ignore, care]")
_K_NOCOMP = _K.replace("care = (1 - chance) * over", "care = chance * over")
_K_BOTH = _K.replace("ignore = chance * under", "ignore = (1 - chance) * under")
_K_ORDER = _K.replace("care = (1 - chance) * over", "care = 1 - chance * over")
_K_RAW = _K.replace("care = (1 - chance) * over", "care = over").replace("ignore = chance * under", "ignore = under")


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
        show=show_value,
        mistakes=[
            Mistake("list", error=lambda e: isinstance(e, TypeError) and "can't multiply sequence" in str(e),
                    en="You multiply by the whole list `chances`. Inside the loop, multiply `result` by one `chance` at a time.",
                    nl="Je vermenigvuldigt met de hele lijst `chances`. Vermenigvuldig in de loop `result` met één `chance` tegelijk."),
            Mistake("sum", wrong=_C_SUM,
                    en="`{call}` gives {got}, and it should give {want}: you add the chances up, so a chain can come out above 1. Each step that must also hold makes the chain less likely: start at 1 and multiply.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt de kansen op, dus een keten kan boven 1 uitkomen. Elke stap die ook moet kloppen, maakt de keten minder waarschijnlijk: begin bij 1 en vermenigvuldig."),
            Mistake("zero", wrong=_C_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: your `result` starts at 0, and 0 times any chance stays 0. Start at 1, the chance of a chain with no steps yet.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `result` begint bij 0, en 0 maal een kans blijft 0. Begin bij 1, de kans van een keten zonder stappen."),
            Mistake("min", wrong=_C_MIN,
                    en="`{call}` gives {got}, and it should give {want}: you give the chance of the least likely step. The chain needs every step to hold, so it is less likely still: multiply the chances.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je geeft de kans van de minst waarschijnlijke stap. De keten heeft elke stap nodig, dus is nog minder waarschijnlijk: vermenigvuldig de kansen."),
            Mistake("last", wrong=_C_LAST,
                    en="`{call}` gives {got}, and it should give {want}: each round replaces `result` by the chance of one step, so only the last step is left. Multiply `result` by each `chance` instead.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke ronde vervangt `result` door de kans van één stap, dus alleen de laatste stap blijft over. Vermenigvuldig `result` in plaats daarvan met elke `chance`."),
            Mistake("not", wrong=_C_NOT,
                    en="`{call}` gives {got}, and it should give {want}: you multiply by `1 - chance`, the chance that a step does not hold, so you get the chance that every step fails. Multiply by `chance` itself.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je vermenigvuldigt met `1 - chance`, de kans dat een stap niet klopt, dus je krijgt de kans dat elke stap mislukt. Vermenigvuldig met `chance` zelf."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: start `result` at 1, multiply it by each `chance` in `chances`, and return it.",
                      "nl": "Vervang de `...`: laat `result` beginnen bij 1, vermenigvuldig het met elke `chance` in `chances`, en geef het terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Start `result` at 1. Go through `chances` with a `for` loop, multiply `result` by each `chance`, and return `result` after the loop.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Laat `result` beginnen bij 1. Loop met een `for` door `chances`, vermenigvuldig `result` met elke `chance`, en geef `result` na de loop terug."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use the `chances` it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie wel de `chances` uit de aanroep?"},
            "correct": {"en": "Your `chain` is right in all {n} cases. Under the check, your ring lies around the course's dot.",
                        "nl": "Je `chain` klopt in alle {n} gevallen. Onder de check ligt jouw ring om de stip van de course."},
        },
    ),
    "costs": FunctionBox(
        name="costs",
        picture=costs_picture,
        function="costs",
        start=_K_START,
        solution=_K,
        visible=[COSTS_CASES[0]],
        hidden=COSTS_CASES[1:],
        first=[_DOTS],
        show=show_value,
        compare=same_list,
        mistakes=[
            Mistake("swap", wrong=_K_SWAP,
                    en="`{call}` gives {got}, and it should give {want}: your two costs are the wrong way round. The list starts with `care`, then `ignore`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je twee kosten staan andersom. De lijst begint met `care`, en dan `ignore`."),
            Mistake("nocomp", wrong=_K_NOCOMP,
                    en="`{call}` gives {got}, and it should give {want}: your `care` uses `chance`, the chance that the system matters. Care is in vain when it does not matter: use `1 - chance`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `care` gebruikt `chance`, de kans dat het systeem ertoe doet. Zorg is vergeefs als het er niet toe doet: gebruik `1 - chance`."),
            Mistake("both", wrong=_K_BOTH,
                    en="`{call}` gives {got}, and it should give {want}: your `ignore` uses `1 - chance`, the chance that the system does not matter. Ignoring costs `under` only when it does matter: use `chance`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `ignore` gebruikt `1 - chance`, de kans dat het systeem er niet toe doet. Negeren kost `under` alleen als het er wel toe doet: gebruik `chance`."),
            Mistake("order", wrong=_K_ORDER,
                    en="`{call}` gives {got}, and it should give {want}: Python multiplies before it subtracts, so `1 - chance * over` is 1 minus `chance * over`. Put brackets around `1 - chance`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: Python vermenigvuldigt voordat het aftrekt, dus `1 - chance * over` is 1 min `chance * over`. Zet haakjes om `1 - chance`."),
            Mistake("raw", wrong=_K_RAW,
                    en="`{call}` gives {got}, and it should give {want}: you give the two costs as if each mistake were certain. Each cost comes only with some chance: multiply `over` by `1 - chance` and `under` by `chance`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je geeft de twee kosten alsof elke fout zeker is. De kosten van elke fout komen maar met een kans: vermenigvuldig `over` met `1 - chance` en `under` met `chance`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: work out `care` from `over` and the chance that the system does not matter, and `ignore` from `under` and `chance`, and return `[care, ignore]`.",
                      "nl": "Vervang de `...`: reken `care` uit met `over` en de kans dat het systeem er niet toe doet, en `ignore` met `under` en `chance`, en geef `[care, ignore]` terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. `care` is `over` times the chance that the system does not matter; `ignore` is `under` times `chance`. Return `[care, ignore]`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. `care` is `over` maal de kans dat het systeem er niet toe doet; `ignore` is `under` maal `chance`. Geef `[care, ignore]` terug."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use the `chance`, `over` and `under` it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie wel de `chance`, `over` en `under` uit de aanroep?"},
            "correct": {"en": "Your `costs` is right in all {n} cases. Under the check, your rings lie around the course's dots.",
                        "nl": "Je `costs` klopt in alle {n} gevallen. Onder de check liggen jouw ringen om de stippen van de course."},
        },
    ),
}

CASES = [
    ("chain", _C_START, "empty", None),
    ("chain", _C, "correct", None),
    ("chain", "def chain(chances):\n    result = 1\n    for k in range(len(chances)):\n        result *= chances[k]\n    return result\n", "correct", None),
    ("chain", "import math\n\n\ndef chain(chances):\n    return math.prod(chances)\n", "correct", None),
    ("chain", _C_LIST, "error", "list"),
    ("chain", _C_SUM, "wrong", "sum"),
    ("chain", _C_ZERO, "wrong", "zero"),
    ("chain", _C_MIN, "wrong", "min"),
    ("chain", _C_LAST, "wrong", "last"),
    ("chain", _C_NOT, "wrong", "not"),
    ("chain", "def chain(chances):\n    return 0.225\n", "almost", None),
    ("costs", _K_START, "empty", None),
    ("costs", _K, "correct", None),
    ("costs", "def costs(chance, over, under):\n    return [over - chance * over, under * chance]\n", "correct", None),
    ("costs", "def costs(chance, over, under):\n    return (1 - chance) * over, chance * under\n", "correct", None),
    ("costs", _K_SWAP, "wrong", "swap"),
    ("costs", _K_NOCOMP, "wrong", "nocomp"),
    ("costs", _K_BOTH, "wrong", "both"),
    ("costs", _K_ORDER, "wrong", "order"),
    ("costs", _K_RAW, "wrong", "raw"),
    ("costs", "def costs(chance, over, under):\n    return [0.98, 2.0]\n", "almost", None),
]
