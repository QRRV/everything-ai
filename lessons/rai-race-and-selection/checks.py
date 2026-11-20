# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-race-and-selection: races to the bottom and selection. The learner writes choices, which of
staying careful and racing has the higher expected value for a company at each level of trust that its rival stays
careful, after the cooperate-defect game of Askell, Brundage and Hadfield (2019), and shares, the share of each kind of
AI agent in a made-up population where each kind is copied at its own rate, after the three conditions for selection
that Hendrycks (2023) takes from Lewontin. A demo changes what racing gains while the rival stays careful."""
import contextlib
import copy
import html
import io
import re

import numpy as np

from everything_ai import chart, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

# --- the data --------------------------------------------------------------------------------------------------------

# A made-up table of what each outcome is worth to one company, in Askell and colleagues' letters: a when both stay
# careful, b when it stays careful and the rival races, c when it races and the rival stays careful, d when both race.
A, B, C, D = 4, 0, 3, 1
GAINS = [2, 2.5, 3, 3.5, 4, 4.5, 5]   # the stops of the demo's slider: what racing gains while the rival stays careful
# A made-up population: the copies in use of three kinds of AI agent, and how much each kind is copied per generation.
COUNTS = [98, 1, 1]
GROWTH = [1.0, 1.05, 1.1]
GENERATIONS = 61                      # generation 0 to 60


class Shown(np.ndarray):
    """An array that a check message writes the way the lesson does, np.array([98, 1, 1]); the learner's function gets a
    plain array."""

    def __repr__(self):
        return "np.array([" + ", ".join(_value(float(x), 4) for x in np.asarray(self)) + "])"

    def __deepcopy__(self, memo):
        return np.array(self)


def shown(values) -> Shown:
    return np.asarray(values, dtype=np.float64).view(Shown)


def choices(a, b, c, d, trusts):
    """What the learner writes in Exercise 1: for each trust, "careful" when staying careful has the higher expected
    value, else "fast", also when the two are equal."""
    result = []
    for trust in trusts:
        careful = trust * a + (1 - trust) * b
        fast = trust * c + (1 - trust) * d
        if careful > fast:
            result.append("careful")
        else:
            result.append("fast")
    return result


def shares(counts, growth, generations):
    """What the learner writes in Exercise 2: the share of each kind in each generation, before it is copied."""
    result = []
    for generation in range(generations):
        result.append(counts / np.sum(counts))
        counts = counts * growth
    return result


def worths(a, b, c, d, trust):
    """The two expected values at one trust: staying careful and racing."""
    return trust * a + (1 - trust) * b, trust * c + (1 - trust) * d


def threshold(a, b, c, d):
    """The trust above which staying careful has the higher expected value, or None when no trust up to 1 is enough;
    where the two lines cross: (d − b) / ((a − c) + (d − b))."""
    if a <= c:
        return None
    return max(0.0, (d - b) / ((a - c) + (d - b)))


def first_hopeless(gains=GAINS):
    """The smallest gain of the slider at which no trust is high enough for staying careful to be worth more."""
    return next(g for g in gains if threshold(A, B, g, D) is None)


def first_half(counts=COUNTS, growth=GROWTH, generations=GENERATIONS):
    """The first generation in which the last kind holds more than half of the population, or None."""
    for g, row in enumerate(shares(np.array(counts, dtype=float), np.array(growth), generations)):
        if row[-1] > 0.5:
            return g
    return None


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


def _value(v, digits: int = 4, limit: int = 70) -> str:
    """A result as the check shows it: a number with at most `digits` decimals, a text in quotes, a list, an array or a
    dict of them, else what Python shows, cut short when it is longer than `limit`."""
    if isinstance(v, np.ndarray):
        v = v.tolist()
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


def show_choices(v) -> str:
    return _value(v, 4, 160)


def show_shares(v) -> str:
    """Shares in a check message: the first three generations and the last one, three decimals each."""
    if isinstance(v, list) and len(v) > 4:
        return "[" + ", ".join(_value(x, 3, 10 ** 6) for x in v[:3]) + ", ..., " + _value(v[-1], 3, 10 ** 6) + "]"
    return _value(v, 3, 220)


def _prose(x: float, lang: str, digits: int = 2) -> str:
    """A number in a picture's words: at most `digits` decimals, with a decimal comma in Dutch."""
    text = f"{x:.{digits}f}"
    text = text.rstrip("0").rstrip(".") if digits > 0 else text
    return text.replace(".", ",") if lang == "nl" else text


def _same_words(got, want) -> bool:
    return isinstance(got, list) and got == want


def _same(got, want) -> bool:
    """Whether the learner's shares are the course's: the same list of arrays, within a small margin."""
    try:
        if isinstance(got, np.ndarray) or not isinstance(got, list) or len(got) != len(want):
            return False
        return all(np.shape(g) == np.shape(w) and np.allclose(np.asarray(g, dtype=float), w, rtol=1e-6, atol=1e-9)
                   for g, w in zip(got, want))
    except (TypeError, ValueError):
        return False


# the words of the lines and the demo
_LINE_WORDS = {
    "en": {"careful": "careful", "fast": "fast", "x": "trust that the rival stays careful", "y": "expected value"},
    "nl": {"careful": "careful", "fast": "fast", "x": "vertrouwen dat de rivaal voorzichtig blijft",
           "y": "verwachtingswaarde"},
}

_DEMO_WORDS = {
    "en": {"title": "Staying careful or racing, when racing gains {c}",
           "from": "careful wins above {p}",
           "said_low": "When racing gains {c} while the rival stays careful, staying careful has the higher expected "
                       "value once your trust that the rival stays careful is above {p}. Below that, racing does.",
           "said_equal": "When racing gains {c}, as much as both staying careful, staying careful never has the higher "
                         "expected value: at a trust of 1 the two are equal, and below that racing is worth more.",
           "said_high": "When racing gains {c}, more than the {a} of both staying careful, racing has the higher "
                        "expected value at every trust: no trust is high enough.",
           "note": "The line careful is the expected value of staying careful, the line fast that of racing. The other "
                   "worths stay the same: {a} when both stay careful, {b} when you stay careful and the rival races, "
                   "{d} when both race. All worths are made up."},
    "nl": {"title": "Voorzichtig blijven of racen, als racen {c} oplevert",
           "from": "voorzichtig wint boven {p}",
           "said_low": "Als racen {c} oplevert terwijl de rivaal voorzichtig blijft, heeft voorzichtig blijven de hoogste "
                       "verwachtingswaarde zodra je vertrouwen dat de rivaal voorzichtig blijft boven {p} ligt. "
                       "Daaronder heeft racen die.",
           "said_equal": "Als racen {c} oplevert, evenveel als samen voorzichtig blijven, heeft voorzichtig blijven nooit de "
                         "hoogste verwachtingswaarde: bij een vertrouwen van 1 zijn ze gelijk, en daaronder levert "
                         "racen meer op.",
           "said_high": "Als racen {c} oplevert, meer dan de {a} van samen voorzichtig blijven, heeft racen bij elk "
                        "vertrouwen de hoogste verwachtingswaarde: geen vertrouwen is hoog genoeg.",
           "note": "De lijn careful is de verwachtingswaarde van voorzichtig blijven, de lijn fast die van racen. De "
                   "andere waarden blijven gelijk: {a} als beide voorzichtig blijven, {b} als jij voorzichtig blijft en "
                   "de rivaal racet, {d} als beide racen. Alle waarden zijn verzonnen."},
}


def demo_said(c, lang: str) -> str:
    w = _DEMO_WORDS[lang]
    p = threshold(A, B, c, D)
    if p is not None:
        return w["said_low"].format(c=_prose(c, lang), p=_prose(p, lang))
    key = "said_equal" if c == A else "said_high"
    return w[key].format(c=_prose(c, lang), a=A)


def _label_heights(a_end: float, c_end: float) -> tuple[float, float]:
    """Where the names careful and fast go at the right end of their lines: at their ends, pulled apart when the ends
    are closer than 0.45 on the axis."""
    if abs(a_end - c_end) >= 0.45:
        return a_end, c_end
    middle = (a_end + c_end) / 2
    low, high = middle - 0.25, middle + 0.25
    return (high, low) if a_end >= c_end else (low, high)


def _worth_panel(a, b, c, d, lang: str, top: float = 5.6) -> chart.Panel:
    """The two expected values against trust, with their names at the right end."""
    w = _LINE_WORDS[lang]
    panel = chart.Panel(x=(-0.06, 1.32), y=(0, top), xlabel=w["x"], ylabel=w["y"], xticks=[0, 0.25, 0.5, 0.75, 1],
                        yticks=[0, 1, 2, 3, 4, 5])
    panel.line([0, 1], [b, a], role="ink")
    panel.line([0, 1], [d, c], role="ink-3")
    ya, yc = _label_heights(a, c)
    panel.text(1.07, ya, w["careful"], role="ink", anchor="start", halo=True)
    panel.text(1.07, yc, w["fast"], role="ink-2", anchor="start", halo=True)
    return panel


def demo_frame(v, lang: str) -> tuple[str, str]:
    """The demo: the expected values of staying careful and of racing against trust, at the chosen gain from racing."""
    c = min(GAINS, key=lambda g: abs(g - float(v)))
    w = _DEMO_WORDS[lang]
    panel = _worth_panel(A, B, c, D, lang)
    p = threshold(A, B, c, D)
    if p is not None:
        panel.line([p, p], [0, 5.0], role="ink-3", dashed=True)
        anchor = "start" if p <= 0.55 else "middle" if p <= 0.8 else "end"   # the label stays clear of the axis
        panel.text(p, 5.25, w["from"].format(p=_prose(p, lang)), role="ink-2", anchor=anchor, halo=True)
    title = w["title"].format(c=_prose(c, lang))
    said = demo_said(c, lang)
    svg = chart.figure([panel], alt=title + ". " + said, width=380, height=240)
    if lang == "nl":
        svg = chart.with_decimal_comma(svg)
    note = w["note"].format(a=A, b=B, d=D)
    return yours.figure(title, svg + _note(note), alt=title + ". " + said), said


# --- the pictures under the checks -----------------------------------------------------------------------------------

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


_CHOICE_WORDS = {
    "en": {"title": "Your choices for the made-up table",
           "head": ["a", "b", "c", "d", "trusts", "the course's", "yours"], "stopped": "error",
           "note": "The chart is the first case of the check. A dot sits on the line that the course chooses at each "
                   "trust and a ring on the line that yours chooses; at 0.5 the two lines meet, and an equal worth "
                   "counts as fast. The table is every case; a dashed row is one where your choices gives something "
                   "other than the course's. All worths are made up."},
    "nl": {"title": "Jouw keuzes voor de verzonnen tabel",
           "head": ["a", "b", "c", "d", "trusts", "van de course", "de jouwe"], "stopped": "fout",
           "note": "De grafiek is het eerste geval van de check. Een stip staat op de lijn die de course bij elk "
                   "vertrouwen kiest en een ring op de lijn die de jouwe kiest; bij 0,5 komen de twee lijnen samen, "
                   "en een gelijke waarde telt als fast. De tabel bevat elk geval; een gestippelde rij is er een waar "
                   "jouw choices iets anders geeft dan die van de course. Alle waarden zijn verzonnen."},
}


def choices_picture(seen) -> str:
    """The first case of the check as the two lines of expected value, the course's choices as dots and the learner's
    as rings on the line each chooses, and every case in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _CHOICE_WORDS[seen.lang]
    got = _all(fn, CHOICE_CASES)
    rows = []
    for i, case in enumerate(CHOICE_CASES):
        want = choices(*case)
        if got[i] is _STOPPED:
            shown_, off = html.escape(w["stopped"]), True
        else:
            shown_, off = _code(show_choices(got[i])), not _same_words(got[i], want)
        rows.append(([(_code(_value(x)), "text-align:right") for x in case[:4]]
                     + [(_code(_value(case[4])), ""), (_code(show_choices(want)), ""), (shown_, "font-weight:600")],
                     _OFF if off else ""))
    a, b, c, d, trusts = CHOICE_CASES[0]
    panel = _worth_panel(a, b, c, d, seen.lang)
    mine = got[0] if isinstance(got[0], list) else []
    rx, ry = [], []
    for k, trust in enumerate(trusts):
        pick = mine[k] if k < len(mine) else None
        if pick in ("careful", "fast"):
            careful, fast = worths(a, b, c, d, trust)
            rx.append(trust)
            ry.append(careful if pick == "careful" else fast)
    panel.points(rx, ry, role="ink", r=7, hollow=True)
    dx, dy = [], []
    for trust, pick in zip(trusts, choices(a, b, c, d, trusts)):
        careful, fast = worths(a, b, c, d, trust)
        dx.append(trust)
        dy.append(careful if pick == "careful" else fast)
    panel.points(dx, dy, role="ink", r=3)
    svg = chart.figure([panel], alt=w["title"], width=380, height=240)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


_SHARE_WORDS = {
    "en": {"title": "Your shares of the made-up population",
           "x": "generation", "y": "share of the population", "kind": "×{g}",
           "head": ["counts", "growth", "generations", "the course's", "yours"], "stopped": "error",
           "note": "The chart is the first case of the check: a line per kind of agent, named by how much it is copied "
                   "per generation. A dot is the course's share and a ring is yours, every ten generations. The table "
                   "is every case; a dashed row is one where your shares gives something other than the course's. "
                   "The population is made up."},
    "nl": {"title": "Jouw aandelen van de verzonnen populatie",
           "x": "generatie", "y": "aandeel van de populatie", "kind": "×{g}",
           "head": ["counts", "growth", "generations", "van de course", "de jouwe"], "stopped": "fout",
           "note": "De grafiek is het eerste geval van de check: een lijn per soort agent, genoemd naar hoeveel die per "
                   "generatie gekopieerd wordt. Een stip is het aandeel van de course en een ring het jouwe, elke tien "
                   "generaties. De tabel bevat elk geval; een gestippelde rij is er een waar jouw shares iets anders "
                   "geeft dan die van de course. De populatie is verzonnen."},
}


def _row_numbers(row, n: int) -> list:
    """One generation of the learner's result as n numbers, with None where it is not a number or missing."""
    if isinstance(row, np.ndarray):
        row = row.tolist()
    if not isinstance(row, list):
        return [None] * n
    out = [_number(x) for x in row[:n]]
    return out + [None] * (n - len(out))


def shares_picture(seen) -> str:
    """The first case of the check as a line per kind, the learner's shares as rings every ten generations, and every
    case in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _SHARE_WORDS[seen.lang]
    got = _all(fn, SHARE_CASES)
    rows = []
    for i, (counts, growth, generations) in enumerate(SHARE_CASES):
        want = shares(np.array(counts), np.array(growth), generations)
        if got[i] is _STOPPED:
            shown_, off = html.escape(w["stopped"]), True
        else:
            shown_, off = _code(show_shares(got[i])), not _same(got[i], want)
        rows.append(([(_code(_value(counts)), ""), (_code(_value(growth)), ""),
                      (_code(_value(generations)), "text-align:right"), (_code(show_shares(want)), ""),
                      (shown_, "font-weight:600")], _OFF if off else ""))
    counts, growth, generations = SHARE_CASES[0]
    want = shares(np.array(counts), np.array(growth), generations)
    kinds = len(counts)
    xs = list(range(generations))
    panel = chart.Panel(x=(-2, generations + 12), y=(0, 1), xlabel=w["x"], ylabel=w["y"],
                        xticks=[0, 10, 20, 30, 40, 50, 60], yticks=[0, 0.25, 0.5, 0.75, 1])
    roles = ["ink-3", "ink-2", "ink"]
    for k in range(kinds):
        panel.line(xs, [float(row[k]) for row in want], role=roles[k])
    mine = got[0] if isinstance(got[0], list) else []
    marks = list(range(0, generations, 10))
    for k in range(kinds):
        rx, ry = [], []
        for g in marks:
            m = _row_numbers(mine[g], kinds)[k] if g < len(mine) else None
            if m is not None and 0 <= m <= 1:
                rx.append(g)
                ry.append(m)
        panel.points(rx, ry, role="ink", r=7, hollow=True)
    for k in range(kinds):
        panel.points(marks, [float(want[g][k]) for g in marks], role=roles[k], r=3)
    for k in range(kinds):
        panel.text(generations + 3, float(want[-1][k]), w["kind"].format(g=_prose(float(growth[k]), seen.lang)),
                   role="ink" if k == kinds - 1 else "ink-2", anchor="start", halo=True)
    svg = chart.figure([panel], alt=w["title"], width=380, height=240)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["choices", "shares", "copied", "trust"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# exercise 1: choices; the first case is the made-up table at five levels of trust, the second a table where racing
# is worth more whatever the rival does, then a table where staying careful is worth more whatever the rival does, no
# trusts, two equal worths, and one trust above the crossing
CHOICE_CASES = [(4, 0, 3, 1, [0, 0.25, 0.5, 0.75, 1]),
                (3, 0, 4, 1, [0, 0.5, 1]),
                (5, 2, 3, 1, [0, 1]),
                (4, 0, 3, 1, []),
                (2, 1, 2, 1, [0.5]),
                (4, 0, 3, 1, [0.6])]
_C_START = "def choices(a, b, c, d, trusts):\n    ...\n"
_C = ("def choices(a, b, c, d, trusts):\n"
      "    result = []\n"
      "    for trust in trusts:\n"
      "        careful = trust * a + (1 - trust) * b\n"
      "        fast = trust * c + (1 - trust) * d\n"
      "        if careful > fast:\n"
      "            result.append(\"careful\")\n"
      "        else:\n"
      "            result.append(\"fast\")\n"
      "    return result\n")
_C_SWAP = (_C.replace("        careful = trust * a + (1 - trust) * b\n", "        careful = trust * a + (1 - trust) * c\n")
           .replace("        fast = trust * c + (1 - trust) * d\n", "        fast = trust * b + (1 - trust) * d\n"))
_C_FLIP = (_C.replace("        careful = trust * a + (1 - trust) * b\n", "        careful = (1 - trust) * a + trust * b\n")
           .replace("        fast = trust * c + (1 - trust) * d\n", "        fast = (1 - trust) * c + trust * d\n"))
_C_TIE = _C.replace("        if careful > fast:\n", "        if careful >= fast:\n")
_C_SURE = _C.replace("        if careful > fast:\n", "        if a > c:\n")
_C_EARLY = _C.replace("    return result\n", "        return result\n")

# exercise 2: shares; the first case is the made-up population over 61 generations, then two equal kinds, a kind that
# grows by a fifth, a kind that doubles, no generations, and a single kind
SHARE_CASES = [(shown(COUNTS), shown(GROWTH), GENERATIONS),
               (shown([1, 1]), shown([1, 1]), 3),
               (shown([50, 50]), shown([1, 1.2]), 4),
               (shown([9, 1]), shown([1, 2]), 5),
               (shown([1, 1, 1]), shown([1, 1, 1]), 0),
               (shown([10]), shown([1.5]), 3)]
_S_START = "def shares(counts, growth, generations):\n    ...\n"
_S = ("def shares(counts, growth, generations):\n"
      "    result = []\n"
      "    for generation in range(generations):\n"
      "        result.append(counts / np.sum(counts))\n"
      "        counts = counts * growth\n"
      "    return result\n")
_S_AFTER = ("def shares(counts, growth, generations):\n"
            "    result = []\n"
            "    for generation in range(generations):\n"
            "        counts = counts * growth\n"
            "        result.append(counts / np.sum(counts))\n"
            "    return result\n")
_S_COUNTS = _S.replace("        result.append(counts / np.sum(counts))\n", "        result.append(counts)\n")
_S_TOTAL = ("def shares(counts, growth, generations):\n"
            "    result = []\n"
            "    total = np.sum(counts)\n"
            "    for generation in range(generations):\n"
            "        result.append(counts / total)\n"
            "        counts = counts * growth\n"
            "    return result\n")
_S_PLUS = _S.replace("        counts = counts * growth\n", "        counts = counts + growth\n")
_S_EARLY = _S.replace("    return result\n", "        return result\n")
_S_ALMOST = ("def shares(counts, growth, generations):\n"
             "    return " + repr([[round(float(x), 12) for x in row]
                                   for row in shares(np.array(COUNTS, dtype=float), np.array(GROWTH), GENERATIONS)])
             + "\n")


EXERCISES = {
    "choices": FunctionBox(
        name="choices",
        picture=choices_picture,
        function="choices",
        start=_C_START,
        solution=_C,
        visible=[CHOICE_CASES[0]],
        hidden=CHOICE_CASES[1:],
        first=[_DOTS],
        show=show_choices,
        mistakes=[
            Mistake("swap", wrong=_C_SWAP,
                    en="`{call}` gives {got}, and it should give {want}: you swapped `b` and `c`. `b` is what staying careful is worth when the rival races, so it belongs with `careful`; `c` is what racing is worth when the rival stays careful, so it belongs with `fast`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je hebt `b` en `c` verwisseld. `b` is wat voorzichtig blijven waard is als de rivaal racet, dus die hoort bij `careful`; `c` is wat racen waard is als de rivaal voorzichtig blijft, dus die hoort bij `fast`."),
            Mistake("flip", wrong=_C_FLIP,
                    en="`{call}` gives {got}, and it should give {want}: you weigh the worths the wrong way round. `trust` is the chance that the rival stays careful, so it goes with `a` and `c`, the worths when the rival stays careful; `1 - trust` goes with `b` and `d`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je weegt de waarden andersom. `trust` is de kans dat de rivaal voorzichtig blijft, dus die hoort bij `a` en `c`, de waarden als de rivaal voorzichtig blijft; `1 - trust` hoort bij `b` en `d`."),
            Mistake("tie", wrong=_C_TIE,
                    en="`{call}` gives {got}, and it should give {want}: when the two expected values are equal, yours chooses `\"careful\"`. The exercise asks for `\"careful\"` only when staying careful is worth more. Askell and colleagues call both choices permissible in a tie; this exercise counts a tie as `\"fast\"`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: als de twee verwachtingswaarden gelijk zijn, kiest de jouwe `\"careful\"`. De oefening vraagt alleen om `\"careful\"` als voorzichtig blijven meer waard is. Askell en anderen noemen bij gelijke waarden beide keuzes toegestaan; deze oefening telt gelijk als `\"fast\"`."),
            Mistake("sure", wrong=_C_SURE,
                    en="`{call}` gives {got}, and it should give {want}: you compare only `a` and `c`, as if the rival surely stays careful, so every trust gets the same answer. Compare the two expected values, `careful` and `fast`, which weigh both cases by `trust`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je vergelijkt alleen `a` en `c`, alsof de rivaal zeker voorzichtig blijft, dus elk vertrouwen krijgt hetzelfde antwoord. Vergelijk de twee verwachtingswaarden, `careful` en `fast`, die beide gevallen wegen met `trust`."),
            Mistake("early", wrong=_C_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your `return` stands inside the loop, so the function stops after the first trust. Put `return result` after the loop, level with `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `return` staat in de loop, dus de functie stopt na het eerste vertrouwen. Zet `return result` na de loop, op gelijke hoogte met `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: for each `trust` in `trusts`, work out the expected values `careful` and `fast`, and put `\"careful\"` or `\"fast\"` in a list `result`.",
                      "nl": "Vervang de `...`: reken voor elke `trust` in `trusts` de verwachtingswaarden `careful` en `fast` uit, en zet `\"careful\"` of `\"fast\"` in een lijst `result`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Begin `result` as an empty list. In a `for` loop over `trusts`, set `careful` to `trust` times `a` plus `1 - trust` times `b`, and `fast` the same with `c` and `d`; append `\"careful\"` when `careful` is greater, else `\"fast\"`; after the loop, return `result`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Begin `result` als een lege lijst. Maak in een `for`-loop over `trusts` `careful` gelijk aan `trust` keer `a` plus `1 - trust` keer `b`, en `fast` net zo met `c` en `d`; voeg `\"careful\"` toe als `careful` groter is, en anders `\"fast\"`; geef na de loop `result` terug."},
            "almost": {"en": "Right for the made-up table. But `{call}` gives {got}, and it should give {want}. Does your function use the `a`, `b`, `c`, `d` and `trusts` it gets?",
                       "nl": "Goed voor de verzonnen tabel. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `a`, `b`, `c`, `d` en `trusts` die ze krijgt?"},
            "correct": {"en": "Your `choices` is right in all {n} cases. Under the check, your rings sit on the line your function chooses at each trust.",
                        "nl": "Je `choices` klopt in alle {n} gevallen. Onder de check staan jouw ringen op de lijn die je functie bij elk vertrouwen kiest."},
        },
    ),
    "shares": FunctionBox(
        name="shares",
        picture=shares_picture,
        function="shares",
        names={"np": np},
        start=_S_START,
        solution=_S,
        visible=[SHARE_CASES[0]],
        hidden=SHARE_CASES[1:],
        first=[_DOTS],
        show=show_shares,
        mistakes=[
            Mistake("after", wrong=_S_AFTER,
                    en="`{call}` gives {got}, and it should give {want}: your list starts one generation late. The first generation is the population as it starts: put its shares in `result` before you copy it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je lijst begint een generatie te laat. De eerste generatie is de populatie zoals die begint: zet de aandelen ervan in `result` voordat je kopieert."),
            Mistake("counts", wrong=_S_COUNTS,
                    en="`{call}` gives {got}, and it should give {want}: you give the number of copies of each kind, not its share. Divide `counts` by the total number of copies in this generation, so that the shares add up to 1.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je geeft het aantal kopieën van elke soort, niet het aandeel. Deel `counts` door het totale aantal kopieën in deze generatie, zodat de aandelen samen 1 zijn."),
            Mistake("total", wrong=_S_TOTAL,
                    en="`{call}` gives {got}, and it should give {want}: you divide every generation by the total of the first one, so the shares no longer add up to 1 once the population grows. Divide by the total of this generation.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je deelt elke generatie door het totaal van de eerste, dus de aandelen zijn samen geen 1 meer zodra de populatie groeit. Deel door het totaal van deze generatie."),
            Mistake("plus", wrong=_S_PLUS,
                    en="`{call}` gives {got}, and it should give {want}: you add `growth` to the copies, so each kind gains a fixed number of copies, not a multiple of what it has. Each kind is copied at its own rate: multiply `counts` by `growth`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt `growth` op bij de kopieën, dus elke soort krijgt er een vast aantal bij, geen veelvoud van het aantal dat er al is. Elke soort wordt in zijn eigen tempo gekopieerd: vermenigvuldig `counts` met `growth`."),
            Mistake("early", wrong=_S_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your `return` stands inside the loop, so the function stops after the first generation. Put `return result` after the loop, level with `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `return` staat in de loop, dus de functie stopt na de eerste generatie. Zet `return result` na de loop, op gelijke hoogte met `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: for each generation, put the share of each kind in a list `result`, then copy every kind at its own rate.",
                      "nl": "Vervang de `...`: zet voor elke generatie het aandeel van elke soort in een lijst `result`, en kopieer dan elke soort in zijn eigen tempo."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Begin `result` as an empty list. In a `for` loop over `range(generations)`, append `counts` divided by the total number of copies, then set `counts` to `counts` times `growth`; after the loop, return `result`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Begin `result` als een lege lijst. Voeg in een `for`-loop over `range(generations)` `counts` gedeeld door het totale aantal kopieën toe, en maak daarna `counts` gelijk aan `counts` keer `growth`; geef na de loop `result` terug."},
            "almost": {"en": "Right for the made-up population. But `{call}` gives {got}, and it should give {want}. Does your function use the `counts`, `growth` and `generations` it gets?",
                       "nl": "Goed voor de verzonnen populatie. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `counts`, `growth` en `generations` die ze krijgt?"},
            "correct": {"en": "Your `shares` is right in all {n} cases. Under the check, your rings lie on the course's lines.",
                        "nl": "Je `shares` klopt in alle {n} gevallen. Onder de check liggen jouw ringen op de lijnen van de course."},
        },
    ),
}

CASES = [
    ("choices", _C_START, "empty", None),
    ("choices", _C, "correct", None),
    ("choices", "def choices(a, b, c, d, trusts):\n    result = []\n    for trust in trusts:\n        if trust * (a - c) > (1 - trust) * (d - b):\n            result.append(\"careful\")\n        else:\n            result.append(\"fast\")\n    return result\n", "correct", None),
    ("choices", "def choices(a, b, c, d, trusts):\n    return [\"careful\" if p * a + (1 - p) * b > p * c + (1 - p) * d else \"fast\" for p in trusts]\n", "correct", None),
    ("choices", _C_SWAP, "wrong", "swap"),
    ("choices", _C_FLIP, "wrong", "flip"),
    ("choices", _C_TIE, "wrong", "tie"),
    ("choices", _C_SURE, "wrong", "sure"),
    ("choices", _C_EARLY, "wrong", "early"),
    ("choices", "def choices(a, b, c, d, trusts):\n    return " + repr(choices(*CHOICE_CASES[0])) + "\n", "almost", None),
    ("shares", _S_START, "empty", None),
    ("shares", _S, "correct", None),
    ("shares", "def shares(counts, growth, generations):\n    return [counts * growth ** g / np.sum(counts * growth ** g) for g in range(generations)]\n", "correct", None),
    ("shares", "def shares(counts, growth, generations):\n    result = []\n    for generation in range(generations):\n        total = 0\n        for count in counts:\n            total = total + count\n        result.append(counts / total)\n        counts = counts * growth\n    return result\n", "correct", None),
    ("shares", _S_AFTER, "wrong", "after"),
    ("shares", _S_COUNTS, "wrong", "counts"),
    ("shares", _S_TOTAL, "wrong", "total"),
    ("shares", _S_PLUS, "wrong", "plus"),
    ("shares", _S_EARLY, "wrong", "early"),
    ("shares", _S_ALMOST, "almost", None),
]
