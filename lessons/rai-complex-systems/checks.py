# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-complex-systems: complex systems, emergence and wicked problems. The learner writes
volumes, the trading volume of each minute in a made-up, simplified market where a sell algorithm sells a share of the
volume of the minute before and fast traders pass what it sold back and forth, and accuracies, the chance of an exact
match on answers of several lengths when each digit is right independently, after Schaeffer and colleagues (2023). A
demo turns up how often each sold contract is traded on."""
import contextlib
import copy
import html
import io
import re

import numpy as np

from everything_ai import chart, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

# --- the data --------------------------------------------------------------------------------------------------------

# The made-up market: the other traders' volume each minute, in contracts, and the sell algorithm's share of the
# volume of the minute before, the 9% of the flash crash report. The echo, how often each contract it sells is traded
# on in the next minute, is made up.
BASE = 1000
RATE = 0.09
ECHOES = list(range(0, 17))   # the stops of the demo's slider
MINUTES = 20                  # the minutes the demo draws
TOP = 20000                   # the top of the demo's chart, in contracts a minute
K = 1000                      # the charts count contracts in thousands
# Eight made-up models, each bigger than the one before: the chance that each gets one digit of an answer right.
PER_DIGIT = [0.5, 0.6, 0.7, 0.8, 0.88, 0.94, 0.97, 0.99]


class Shown(np.ndarray):
    """An array that a check message writes the way the lesson does, np.array([0.5, 0.6]); the learner's function gets
    a plain array."""

    def __repr__(self):
        return "np.array([" + ", ".join(_value(float(x), 4) for x in np.asarray(self)) + "])"

    def __deepcopy__(self, memo):
        return np.array(self)


def shown(values) -> Shown:
    return np.asarray(values, dtype=np.float64).view(Shown)


def volumes(base, rate, echo, minutes):
    """What the learner writes in Exercise 1: the volume of each minute, where the next minute's volume is the other
    traders' base plus each contract the algorithm sold, traded echo times."""
    volume = base
    result = []
    for minute in range(minutes):
        result.append(volume)
        sold = rate * volume
        volume = base + echo * sold
    return result


def accuracies(per_digit, lengths):
    """What the learner writes in Exercise 2: for each length, every model's chance that all digits are right."""
    rows = []
    for length in lengths:
        rows.append(per_digit ** length)
    return rows


def level(echo, rate=RATE, base=BASE):
    """Where the volume settles when rate × echo is below 1: base / (1 − rate × echo); None when it does not settle."""
    factor = rate * echo
    return base / (1 - factor) if factor < 1 else None


def first_runaway(rate=RATE):
    """The smallest echo of the slider at which each minute adds more than the minute before: rate × echo above 1."""
    return next(e for e in ECHOES if rate * e > 1)


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


def show_volumes(v) -> str:
    return _value(v, 2, 160)


def show_rows(v) -> str:
    return _value(v, 3, 220)


def _prose(x: float, lang: str, digits: int = 2) -> str:
    """A number in a picture's words: at most `digits` decimals, with a decimal comma in Dutch."""
    text = f"{x:.{digits}f}"
    text = text.rstrip("0").rstrip(".") if digits > 0 else text
    return text.replace(".", ",") if lang == "nl" else text


def _count(n: float, lang: str) -> str:
    """A whole number of contracts in words: 1,000 in English; in Dutch a thousands dot only from five digits."""
    n = round(n)
    if lang == "nl":
        return f"{n:,}".replace(",", ".") if n >= 10000 else str(n)
    return f"{n:,}"


def _same(got, want) -> bool:
    """Whether the learner's result is the course's: the same list of numbers, or of arrays, within a small margin."""
    try:
        if isinstance(got, np.ndarray) or not isinstance(got, list) or len(got) != len(want):
            return False
        return all(np.shape(g) == np.shape(w) and np.allclose(np.asarray(g, dtype=float), w, rtol=1e-6, atol=1e-9)
                   for g, w in zip(got, want))
    except (TypeError, ValueError):
        return False


_DEMO_WORDS = {
    "en": {"title": "Twenty minutes of the made-up market, with an echo of {e}",
           "x": "minute", "y": "thousand contracts per minute",
           "level": "settles at {v}",
           "said0": "With an echo of 0, nobody trades the sold contracts on: the volume stays at the other traders' "
                    "{b} contracts a minute.",
           "said_low": "With an echo of {e}, each minute adds {f} times what the minute before added ({r} × "
                       "{e}). That is less than 1, so the additions shrink, and in the long run the "
                       "volume settles at about {v} contracts a minute.",
           "said_high": "With an echo of {e}, each minute adds {f} times what the minute before added ({r} × {e}). That "
                        "is more than 1, so the additions grow: after {n} minutes the volume is {v} contracts a minute, "
                        "far above the top of the chart, and it keeps rising.",
           "note": "Each point is the volume of one minute in the made-up market: the other traders' 1,000 contracts, "
                   "plus each contract the algorithm sold, traded on as often as the echo says. The algorithm sells "
                   "9% of the volume of the minute before. All numbers but the 9% are made up."},
    "nl": {"title": "Twintig minuten van de verzonnen markt, met een echo van {e}",
           "x": "minuut", "y": "duizend contracten per minuut",
           "level": "komt uit op {v}",
           "said0": "Met een echo van 0 handelt niemand de verkochte contracten door: het volume blijft op de {b} "
                    "contracten per minuut van de andere handelaren.",
           "said_low": "Met een echo van {e} voegt elke minuut {f} keer toe wat de minuut ervoor toevoegde ({r} × "
                       "{e}). Dat is minder dan 1, dus de toevoegingen krimpen, en op den "
                       "duur komt het volume uit op ongeveer {v} contracten per minuut.",
           "said_high": "Met een echo van {e} voegt elke minuut {f} keer toe wat de minuut ervoor toevoegde ({r} × "
                        "{e}). Dat is meer dan 1, dus de toevoegingen groeien: na {n} minuten is het volume {v} "
                        "contracten per minuut, ver boven de grafiek, en het blijft stijgen.",
           "note": "Elk punt is het volume van één minuut in de verzonnen markt: de 1000 contracten van de andere "
                   "handelaren, plus elk contract dat het algoritme verkocht, zo vaak doorverhandeld als de echo zegt. "
                   "Het algoritme verkoopt 9% van het volume van de minuut ervoor. Alle getallen behalve die 9% zijn "
                   "verzonnen."},
}


def demo_said(echo: int, lang: str) -> str:
    w = _DEMO_WORDS[lang]
    if echo == 0:
        return w["said0"].format(b=_count(BASE, lang))
    factor = RATE * echo
    if factor < 1:
        return w["said_low"].format(e=echo, f=_prose(factor, lang), r=_prose(RATE, lang),
                                    v=_count(level(echo), lang))
    run = volumes(BASE, RATE, echo, MINUTES)
    return w["said_high"].format(e=echo, f=_prose(factor, lang), r=_prose(RATE, lang), n=MINUTES,
                                 v=_count(run[-1], lang))


def _clipped(xs, ys, top):
    """The points of a line up to where it leaves the chart at `top`, with the point where it crosses the top."""
    out_x, out_y = [], []
    for i, (x, y) in enumerate(zip(xs, ys)):
        if y <= top:
            out_x.append(x)
            out_y.append(y)
            continue
        if i > 0:
            x0, y0 = xs[i - 1], ys[i - 1]
            out_x.append(x0 + (top - y0) / (y - y0) * (x - x0))
            out_y.append(top)
        break
    return out_x, out_y


def demo_frame(v, lang: str) -> tuple[str, str]:
    """The demo: twenty minutes of the made-up market at the chosen echo, cut off where it leaves the chart."""
    echo = min(ECHOES, key=lambda e: abs(e - float(v)))
    w = _DEMO_WORDS[lang]
    xs = list(range(1, MINUTES + 1))
    run = volumes(BASE, RATE, echo, MINUTES)
    # the chart counts in thousands, so its ticks need no thousands separator in either language
    panel = chart.Panel(x=(0.5, MINUTES + 0.5), y=(0, TOP / K), xlabel=w["x"], ylabel=w["y"],
                        xticks=[1, 5, 10, 15, 20], yticks=[0, 5, 10, 15, 20])
    settle = level(echo)
    if settle is not None and settle <= TOP * 0.9 and echo > 0:
        panel.line([0.5, MINUTES + 0.5], [settle / K, settle / K], role="ink-3", dashed=True)
        panel.text(MINUTES, (settle + TOP * 0.08) / K, w["level"].format(v=_count(settle, lang)), role="ink-2",
                   anchor="end", halo=True)
    cx, cy = _clipped(xs, run, TOP)
    panel.line(cx, [y / K for y in cy], role="ink")
    kept = [(x, y) for x, y in zip(xs, run) if y <= TOP]
    panel.points([x for x, _ in kept], [y / K for _, y in kept], role="ink", r=3)
    title = w["title"].format(e=echo)
    said = demo_said(echo, lang)
    svg = chart.figure([panel], alt=title + ". " + said, width=380, height=240)
    if lang == "nl":
        svg = chart.with_decimal_comma(svg)
    return yours.figure(title, svg + _note(w["note"]), alt=title + ". " + said), said


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


def _numbers(got, n: int) -> list:
    """The learner's result as a list of n numbers, with None where it is not a number or missing."""
    if isinstance(got, np.ndarray):
        got = got.tolist()
    if not isinstance(got, list):
        return [None] * n
    out = [_number(x) for x in got[:n]]
    return out + [None] * (n - len(out))


_VOLUME_WORDS = {
    "en": {"title": "Your volumes in the made-up market",
           "x": "minute", "y": "thousand contracts per minute", "high": "echo {e}", "low": "echo {e}",
           "head": ["base", "rate", "echo", "minutes", "the course's", "yours"], "stopped": "error",
           "note": "The chart is the first two cases of the check: the same market with an echo of 12 and of 5. A dot "
                   "is the course's volume and a ring is yours. The table is every case; a dashed row is one where "
                   "your volumes gives something other than the course's. The market is made up and simplified."},
    "nl": {"title": "Jouw volumes in de verzonnen markt",
           "x": "minuut", "y": "duizend contracten per minuut", "high": "echo {e}", "low": "echo {e}",
           "head": ["base", "rate", "echo", "minutes", "van de course", "de jouwe"], "stopped": "fout",
           "note": "De grafiek is de eerste twee gevallen van de check: dezelfde markt met een echo van 12 en van 5. "
                   "Een stip is het volume van de course en een ring het jouwe. De tabel bevat elk geval; een "
                   "gestippelde rij is er een waar jouw volumes iets anders geeft dan die van de course. De markt is "
                   "verzonnen en vereenvoudigd."},
}


def volumes_picture(seen) -> str:
    """The first two cases of the check as two lines with the learner's volumes as rings, and every case in a
    table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _VOLUME_WORDS[seen.lang]
    got = _all(fn, VOLUME_CASES)
    rows = []
    for i, case in enumerate(VOLUME_CASES):
        want = volumes(*case)
        if got[i] is _STOPPED:
            shown_, off = html.escape(w["stopped"]), True
        else:
            shown_, off = _code(show_volumes(got[i])), not _same(got[i], want)
        rows.append(([(_code(_value(a)), "text-align:right") for a in case]
                     + [(_code(show_volumes(want)), ""), (shown_, "font-weight:600")], _OFF if off else ""))
    high, low = VOLUME_CASES[0], VOLUME_CASES[1]
    n = high[3]
    xs = list(range(1, n + 1))
    panel = chart.Panel(x=(0.5, n + 0.5), y=(0, 12), xlabel=w["x"], ylabel=w["y"], xticks=xs,
                        yticks=[0, 3, 6, 9, 12])
    for case, role in ((high, "ink"), (low, "ink-3")):
        panel.line(xs, [v / K for v in volumes(*case)], role=role)
    for i, case in enumerate((high, low)):
        mine = _numbers(got[i], n)
        rx = [x for x, m in zip(xs, mine) if m is not None and 0 <= m <= 12000]
        ry = [m / K for m in mine if m is not None and 0 <= m <= 12000]
        panel.points(rx, ry, role="ink", r=7, hollow=True)
    for case, role in ((high, "ink"), (low, "ink-3")):
        panel.points(xs, [v / K for v in volumes(*case)], role=role, r=3)
    panel.text(2.0, 4.3, w["high"].format(e=high[2]), role="ink", anchor="middle", halo=True)
    panel.text(6, 3.0, w["low"].format(e=low[2]), role="ink-2", anchor="middle", halo=True)
    svg = chart.figure([panel], alt=w["title"], width=380, height=240)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


_ROW_WORDS = {
    "en": {"title": "Your accuracies for eight made-up models",
           "x": "model, from small to big", "y": "chance of an exact match", "one": "1 digit", "many": "{n} digits",
           "head": ["per_digit", "lengths", "the course's", "yours"], "stopped": "error",
           "note": "The chart is the first case of the check. The grey line is the chance of getting one digit right, "
                   "which rises steadily from model to model; the other line is the chance that all 10 digits are right. "
                   "A dot is the course's chance and a ring is yours. The table is every case; a dashed row is one "
                   "where your accuracies gives something other than the course's. The models are made up."},
    "nl": {"title": "Jouw accuracies voor acht verzonnen modellen",
           "x": "model, van klein naar groot", "y": "kans op een exact match", "one": "1 cijfer",
           "many": "{n} cijfers",
           "head": ["per_digit", "lengths", "van de course", "de jouwe"], "stopped": "fout",
           "note": "De grafiek is het eerste geval van de check. De grijze lijn is de kans om één cijfer goed te "
                   "hebben, die geleidelijk stijgt van model tot model; de andere lijn is de kans dat alle 10 cijfers "
                   "goed zijn. Een stip is de kans van de course en een ring de jouwe. De tabel bevat elk geval; een "
                   "gestippelde rij is er een waar jouw accuracies iets anders geeft dan die van de course. De "
                   "modellen zijn verzonnen."},
}


def accuracies_picture(seen) -> str:
    """The first case of the check as one line per length, the learner's chances as rings, and every case in a
    table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _ROW_WORDS[seen.lang]
    got = _all(fn, ROW_CASES)
    rows = []
    for i, (per_digit, lengths) in enumerate(ROW_CASES):
        want = accuracies(np.array(per_digit), lengths)
        if got[i] is _STOPPED:
            shown_, off = html.escape(w["stopped"]), True
        else:
            shown_, off = _code(show_rows(got[i])), not _same(got[i], want)
        rows.append(([(_code(_value(per_digit)), ""), (_code(_value(lengths)), ""), (_code(show_rows(want)), ""),
                      (shown_, "font-weight:600")], _OFF if off else ""))
    per_digit, lengths = ROW_CASES[0]
    want = accuracies(np.array(per_digit), lengths)
    n = len(per_digit)
    xs = list(range(1, n + 1))
    panel = chart.Panel(x=(0.5, n + 0.5), y=(0, 1), xlabel=w["x"], ylabel=w["y"], xticks=xs,
                        yticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
    roles = ["ink-3", "ink"]
    for row, role in zip(want, roles):
        panel.line(xs, row.tolist(), role=role)
    mine = got[0] if isinstance(got[0], list) else []
    for k in range(len(want)):
        values = _numbers(mine[k], n) if k < len(mine) else [None] * n
        rx = [x for x, m in zip(xs, values) if m is not None and 0 <= m <= 1]
        ry = [m for m in values if m is not None and 0 <= m <= 1]
        panel.points(rx, ry, role="ink", r=7, hollow=True)
    for row, role in zip(want, roles):
        panel.points(xs, row.tolist(), role=role, r=3)
    panel.text(1.5, 0.71, w["one"], role="ink-2", anchor="middle", halo=True)
    panel.text(2.5, 0.12, w["many"].format(n=lengths[1]), role="ink", anchor="middle", halo=True)
    svg = chart.figure([panel], alt=w["title"], width=380, height=240)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["volumes", "accuracies", "agents", "backfire"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# exercise 1: volumes; the first case is the market with an echo of 12, the second the same market with an echo of 5,
# then no echo, an algorithm that sells nothing, a market that grows by the same amount each minute, no minutes, and
# a market that doubles its addition each minute
VOLUME_CASES = [(1000, 0.09, 12, 8),
                (1000, 0.09, 5, 8),
                (500, 0.1, 0, 3),
                (1000, 0, 12, 3),
                (200, 0.5, 2, 4),
                (1000, 0.09, 12, 0),
                (100, 0.2, 10, 3)]
_V_START = "def volumes(base, rate, echo, minutes):\n    ...\n"
_V = ("def volumes(base, rate, echo, minutes):\n"
      "    volume = base\n"
      "    result = []\n"
      "    for minute in range(minutes):\n"
      "        result.append(volume)\n"
      "        sold = rate * volume\n"
      "        volume = base + echo * sold\n"
      "    return result\n")
_V_NO_BASE = _V.replace("        volume = base + echo * sold\n", "        volume = echo * sold\n")
_V_PLUS = _V.replace("        volume = base + echo * sold\n", "        volume = volume + echo * sold\n")
_V_AFTER = ("def volumes(base, rate, echo, minutes):\n"
            "    volume = base\n"
            "    result = []\n"
            "    for minute in range(minutes):\n"
            "        sold = rate * volume\n"
            "        volume = base + echo * sold\n"
            "        result.append(volume)\n"
            "    return result\n")
_V_RATE_BASE = _V.replace("        sold = rate * volume\n", "        sold = rate * base\n")
_V_NO_ECHO = _V.replace("        volume = base + echo * sold\n", "        volume = base + sold\n")
_V_EARLY = _V.replace("    return result\n", "        return result\n")

# exercise 2: accuracies; the first case is the eight made-up models with answers of 1 and of 10 digits, then three
# lengths, other models, one model, no lengths, and a model that is always right beside one that is never right
ROW_CASES = [(shown(PER_DIGIT), [1, 10]),
             (shown(PER_DIGIT), [1, 5, 10]),
             (shown([0.5, 0.7, 0.9, 0.99]), [2, 4]),
             (shown([0.9]), [1, 2, 3]),
             (shown([0.5, 0.8]), []),
             (shown([1.0, 0.0]), [3])]
_A_START = "def accuracies(per_digit, lengths):\n    ...\n"
_A = ("def accuracies(per_digit, lengths):\n"
      "    rows = []\n"
      "    for length in lengths:\n"
      "        rows.append(per_digit ** length)\n"
      "    return rows\n")
_A_TIMES = _A.replace("per_digit ** length)", "per_digit * length)")
_A_FLIP = _A.replace("per_digit ** length)", "length ** per_digit)")
_A_SAME = _A.replace("per_digit ** length)", "per_digit)")
_A_ATLEAST = _A.replace("per_digit ** length)", "1 - (1 - per_digit) ** length)")
_A_ALL = _A.replace("per_digit ** length)", "per_digit ** lengths)")
_A_EARLY = _A.replace("    return rows\n", "        return rows\n")
_A_ALMOST = ("def accuracies(per_digit, lengths):\n"
             "    return " + repr([[float(p) ** n for p in PER_DIGIT] for n in (1, 10)]) + "\n")


EXERCISES = {
    "volumes": FunctionBox(
        name="volumes",
        picture=volumes_picture,
        function="volumes",
        start=_V_START,
        solution=_V,
        visible=[VOLUME_CASES[0]],
        hidden=VOLUME_CASES[1:],
        first=[_DOTS],
        show=show_volumes,
        mistakes=[
            Mistake("no-base", wrong=_V_NO_BASE,
                    en="`{call}` gives {got}, and it should give {want}: after the first minute, your volume holds only what the algorithm sold, traded on. The other traders keep trading too: each minute their `base` comes on top.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: na de eerste minuut bevat jouw volume alleen wat het algoritme verkocht, doorverhandeld. De andere handelaren blijven ook handelen: elke minuut komt hun `base` erbij."),
            Mistake("plus", wrong=_V_PLUS,
                    en="`{call}` gives {got}, and it should give {want}: you add to the volume of the minute before, so every minute keeps all the minutes before it. The next minute's volume is new: the other traders' `base`, plus what the algorithm sold, traded on.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt op bij het volume van de minuut ervoor, dus elke minuut houdt alle minuten ervoor vast. Het volume van de volgende minuut is nieuw: de `base` van de andere handelaren, plus wat het algoritme verkocht, doorverhandeld."),
            Mistake("after", wrong=_V_AFTER,
                    en="`{call}` gives {got}, and it should give {want}: your list starts one minute late. The first minute has only the other traders' `base`: put `volume` in `result` before you work out the next minute.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je lijst begint een minuut te laat. De eerste minuut heeft alleen de `base` van de andere handelaren: zet `volume` in `result` voordat je de volgende minuut uitrekent."),
            Mistake("rate-base", wrong=_V_RATE_BASE,
                    en="`{call}` gives {got}, and it should give {want}: your algorithm sells a share of `base` every minute, so it never sells more. It follows the volume of the minute before: `sold` is `rate` times `volume`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw algoritme verkoopt elke minuut een deel van `base`, dus het verkoopt nooit meer. Het volgt het volume van de minuut ervoor: `sold` is `rate` keer `volume`."),
            Mistake("no-echo", wrong=_V_NO_ECHO,
                    en="`{call}` gives {got}, and it should give {want}: each contract the algorithm sold counts once in your volume. The fast traders pass it back and forth, so it is traded `echo` times in the next minute.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elk contract dat het algoritme verkocht, telt bij jou één keer mee. De snelle handelaren geven het aan elkaar door, dus het wordt in de volgende minuut `echo` keer verhandeld."),
            Mistake("early", wrong=_V_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your `return` stands inside the loop, so the function stops after the first minute. Put `return result` after the loop, level with `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `return` staat in de loop, dus de functie stopt na de eerste minuut. Zet `return result` na de loop, op gelijke hoogte met `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: begin `volume` at `base`, and for each minute put `volume` in a list `result`, then work out `sold` and the next minute's `volume`.",
                      "nl": "Vervang de `...`: begin `volume` bij `base`, en zet voor elke minuut `volume` in een lijst `result`; reken dan `sold` en het `volume` van de volgende minuut uit."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Begin `volume` at `base` and `result` as an empty list. In a `for` loop over `range(minutes)`, append `volume`, set `sold` to `rate` times `volume`, and set the next `volume` to the other traders' `base` plus every sold contract traded `echo` times; after the loop, return `result`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Begin `volume` bij `base` en `result` als een lege lijst. Voeg in een `for`-loop over `range(minutes)` het `volume` toe, maak `sold` gelijk aan `rate` keer `volume`, en maak het volgende `volume` gelijk aan de `base` van de andere handelaren plus elk verkocht contract `echo` keer verhandeld; geef na de loop `result` terug."},
            "almost": {"en": "Right for the market with an echo of 12. But `{call}` gives {got}, and it should give {want}. Does your function use the `base`, `rate`, `echo` and `minutes` it gets?",
                       "nl": "Goed voor de markt met een echo van 12. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `base`, `rate`, `echo` en `minutes` die ze krijgt?"},
            "correct": {"en": "Your `volumes` is right in all {n} cases. Under the check, your rings follow the market with an echo of 12 and of 5.",
                        "nl": "Je `volumes` klopt in alle {n} gevallen. Onder de check volgen jouw ringen de markt met een echo van 12 en van 5."},
        },
    ),
    "accuracies": FunctionBox(
        name="accuracies",
        picture=accuracies_picture,
        function="accuracies",
        start=_A_START,
        solution=_A,
        visible=[ROW_CASES[0]],
        hidden=ROW_CASES[1:],
        first=[_DOTS],
        show=show_rows,
        mistakes=[
            Mistake("all", error=lambda e: isinstance(e, ValueError) and "broadcast" in str(e),
                    en="`{call}` stops with a `ValueError`: NumPy cannot pair the chances in `per_digit` with the whole list `lengths`. Inside the loop, use one `length` at a time.",
                    nl="`{call}` stopt met een `ValueError`: NumPy kan de kansen in `per_digit` niet koppelen aan de hele lijst `lengths`. Gebruik in de loop telkens één `length`."),
            Mistake("times", wrong=_A_TIMES,
                    en="`{call}` gives {got}, and it should give {want}: you multiply the chance by the number of digits, but a chance cannot be above 1. Every digit has to be right, and their chances multiply: the chance for one digit, times itself, once for each digit.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je vermenigvuldigt de kans met het aantal cijfers, maar een kans kan niet boven 1 liggen. Elk cijfer moet goed zijn, en hun kansen vermenigvuldigen zich: de kans voor één cijfer, keer zichzelf, één keer per cijfer."),
            Mistake("flip", wrong=_A_FLIP,
                    en="`{call}` gives {got}, and it should give {want}: you raise the number of digits to the power of the chance. The other way round: the chance for one digit, multiplied by itself once for each digit.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je verheft het aantal cijfers tot de macht van de kans. Andersom: de kans voor één cijfer, vermenigvuldigd met zichzelf, één keer per cijfer."),
            Mistake("same", wrong=_A_SAME,
                    en="`{call}` gives {got}, and it should give {want}: every row holds the chance for one digit, whatever the length. A longer answer has more digits that must all be right, so its chance is smaller.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke rij bevat de kans voor één cijfer, wat de lengte ook is. Een langer antwoord heeft meer cijfers die allemaal goed moeten zijn, dus de kans is kleiner."),
            Mistake("at-least", wrong=_A_ATLEAST,
                    en="`{call}` gives {got}, and it should give {want}: you give the chance that at least one digit is right. An exact match needs every digit right.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je geeft de kans dat minstens één cijfer goed is. Voor een exact match moet elk cijfer goed zijn."),
            Mistake("early", wrong=_A_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your `return` stands inside the loop, so the function stops after the first length. Put `return rows` after the loop, level with `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `return` staat in de loop, dus de functie stopt na de eerste lengte. Zet `return rows` na de loop, op gelijke hoogte met `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: for each `length` in `lengths`, put every model's chance that all `length` digits are right in a list `rows`.",
                      "nl": "Vervang de `...`: zet voor elke `length` in `lengths` de kans van elk model dat alle `length` cijfers goed zijn in een lijst `rows`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Begin `rows` as an empty list. In a `for` loop over `lengths`, append the chance that all `length` digits are right, for all models at once; after the loop, return `rows`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Begin `rows` als een lege lijst. Voeg in een `for`-loop over `lengths` de kans toe dat alle `length` cijfers goed zijn, voor alle modellen tegelijk; geef na de loop `rows` terug."},
            "almost": {"en": "Right for the eight models with 1 and 10 digits. But `{call}` gives {got}, and it should give {want}. Does your function use the `per_digit` and `lengths` it gets?",
                       "nl": "Goed voor de acht modellen met 1 en 10 cijfers. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `per_digit` en `lengths` die ze krijgt?"},
            "correct": {"en": "Your `accuracies` is right in all {n} cases. Under the check, your rings lie on the course's lines.",
                        "nl": "Je `accuracies` klopt in alle {n} gevallen. Onder de check liggen jouw ringen op de lijnen van de course."},
        },
    ),
}

CASES = [
    ("volumes", _V_START, "empty", None),
    ("volumes", _V, "correct", None),
    ("volumes", "def volumes(base, rate, echo, minutes):\n    result = [base]\n    while len(result) < minutes:\n        result.append(base + echo * rate * result[-1])\n    return result[:minutes]\n", "correct", None),
    ("volumes", "def volumes(base, rate, echo, minutes):\n    result = []\n    volume = base\n    for _ in range(minutes):\n        result.append(volume)\n        volume = base + rate * echo * volume\n    return result\n", "correct", None),
    ("volumes", _V_NO_BASE, "wrong", "no-base"),
    ("volumes", _V_PLUS, "wrong", "plus"),
    ("volumes", _V_AFTER, "wrong", "after"),
    ("volumes", _V_RATE_BASE, "wrong", "rate-base"),
    ("volumes", _V_NO_ECHO, "wrong", "no-echo"),
    ("volumes", _V_EARLY, "wrong", "early"),
    ("volumes", "def volumes(base, rate, echo, minutes):\n    return " + repr(volumes(*VOLUME_CASES[0])) + "\n", "almost", None),
    ("accuracies", _A_START, "empty", None),
    ("accuracies", _A, "correct", None),
    ("accuracies", "def accuracies(per_digit, lengths):\n    return [per_digit ** n for n in lengths]\n", "correct", None),
    ("accuracies", "def accuracies(per_digit, lengths):\n    rows = []\n    for length in lengths:\n        chance = per_digit * 0 + 1\n        for digit in range(length):\n            chance = chance * per_digit\n        rows.append(chance)\n    return rows\n", "correct", None),
    ("accuracies", _A_ALL, "error", "all"),
    ("accuracies", _A_TIMES, "wrong", "times"),
    ("accuracies", _A_FLIP, "wrong", "flip"),
    ("accuracies", _A_SAME, "wrong", "same"),
    ("accuracies", _A_ATLEAST, "wrong", "at-least"),
    ("accuracies", _A_EARLY, "wrong", "early"),
    ("accuracies", _A_ALMOST, "almost", None),
]
