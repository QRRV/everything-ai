# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson data-labels: a web shop the course simulates, 300 orders on days 0 to 179, about one in five
sent back within 30 days of the order (NumPy's generator with seed 33, one of the seeds where the rate after 30 days
lies within a point of the share that really comes back; with 300 orders other seeds scatter by a few points); on a given day an order has a label only when
30 days have passed, and the return rate counts only orders with a label. The functions label and return_rate.
tests/test_data_labels.py measures the numbers in the lesson again."""
import contextlib
import functools
import io
import warnings

from everything_ai import chart, check as checking, yours
from everything_ai.check import FunctionBox, Mistake

ORDERS, DAYS, WINDOW = 300, 180, 30
TODAYS = [30, 60, 90, 120, 150, 180]
LABELED = TODAYS[1:]   # on day 30 no order is 30 days old yet: no labels to count
# The worked example: eight orders on day 60, the day each was placed and whether it has come back so far.
EXAMPLE_TODAY = 60
EXAMPLE_ORDERED = [5, 12, 20, 30, 38, 44, 52, 57]
EXAMPLE_RETURNED = [False, True, False, False, True, False, True, False]


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def percent(x: float, lang: str) -> str:
    return f"{round(100 * x)}%"


@functools.lru_cache(maxsize=None)
def shop() -> tuple:
    """The simulated orders: the day each was placed, whether it will come back, and after how many days (1 to 30)."""
    import numpy as np
    rng = np.random.default_rng(33)
    ordered = np.sort(rng.integers(0, DAYS, ORDERS))
    will = rng.random(ORDERS) < 0.2
    delay = rng.integers(1, WINDOW + 1, ORDERS)
    return tuple(int(d) for d in ordered), tuple(bool(w) for w in will), tuple(int(d) for d in delay)


def right_label(ordered: int, returned: bool, today: int):
    if today - ordered < WINDOW:
        return None
    return 1 if returned else 0


def seen(today: int) -> tuple:
    """The orders placed before `today`: the day of each, and whether it has come back by `today`."""
    ordered, will, delay = shop()
    days, back = [], []
    for o, w, d in zip(ordered, will, delay):
        if o < today:
            days.append(o)
            back.append(w and o + d <= today)
    return days, back


def counts(today: int) -> dict:
    """What the shop knows on `today`: orders, returns so far, and the rates three ways."""
    days, back = seen(today)
    young = [today - o < WINDOW for o in days]
    settled = [not y for y in young]
    returned = sum(back)
    settled_returned = sum(b for b, s in zip(back, settled) if s)
    young_returned = sum(b for b, y in zip(back, young) if y)
    n_settled = sum(settled)
    return {
        "orders": len(days), "returned": returned, "young": sum(young), "young_returned": young_returned,
        "settled": n_settled, "settled_returned": settled_returned,
        "right": settled_returned / n_settled if n_settled else None,                  # wait 30 days for every order
        "as_kept": returned / len(days) if days else None,                             # unknown counted as kept
        "early": returned / (n_settled + young_returned) if n_settled + young_returned else None,   # early returns too
    }


def true_rate() -> float:
    """The share of all 300 orders that will come back: known in the simulation, never in a real shop."""
    return sum(shop()[1]) / ORDERS


# --- the slider ---------------------------------------------------------------------------------------------------------

def today_said(today: int, lang: str) -> str:
    c = counts(today)
    young_kept = c["young"] - c["young_returned"]
    if lang == "nl":
        if c["settled"] == 0:
            head = f"Dag {today}: {c['orders']} bestellingen, en geen ervan is al 30 dagen oud: nog geen enkel label. "
        else:
            head = (f"Dag {today}: {c['orders']} bestellingen. {c['settled']} zijn minstens 30 dagen oud, en daarvan kwamen er "
                    f"{c['settled_returned']} terug: {percent(c['right'], lang)}. ")
        return head + (f"Tel je de {young_kept} bestellingen jonger dan 30 dagen die nog niet terugkwamen als gehouden, dan "
                       f"kom je op {percent(c['as_kept'], lang)}; tel je de {c['young_returned']} jonge die al terugkwamen wel "
                       f"mee en die {young_kept} niet, dan op {percent(c['early'], lang)}.")
    if c["settled"] == 0:
        head = f"Day {today}: {c['orders']} orders, and none of them is 30 days old yet: not a single label. "
    else:
        head = (f"Day {today}: {c['orders']} orders. {c['settled']} are at least 30 days old, and {c['settled_returned']} of "
                f"those came back: {percent(c['right'], lang)}. ")
    return head + (f"Counting the {young_kept} orders younger than 30 days that have not come back as kept gives "
                   f"{percent(c['as_kept'], lang)}; counting the {c['young_returned']} young ones that already came back, but "
                   f"not those {young_kept}, gives {percent(c['early'], lang)}.")


def _offset(i: int) -> float:
    return ((i * 37) % 11 - 5) * 0.06


def _timeline(days: list, labels: list, today: int, words: dict):
    """A dot per order at the day it was placed, in the row of its label; the last 30 days before `today` shaded."""
    p = chart.Panel((0, DAYS), (-0.5, 2.8), title=words["orders"], xlabel=words["day"], xticks=[0, 30, 60, 90, 120, 150, 180],
                    yticks=[])
    p.rect(max(0, today - WINDOW), -0.5, today, 2.8, role="ink-3", solid=False)
    p.line([today, today], [-0.5, 2.8], role="ink-3", width=1, dashed=True)
    rows = {1: 2, 0: 1, None: 0}
    for value, key in ((1, "returned"), (0, "kept"), (None, "unknown")):   # the name above each row
        p.text(1, rows[value] + 0.48, words[key], role="ink-2", anchor="start", halo=True)
    for value, role, hollow in ((None, "axis", True), (0, "data", True), (1, "data", False)):
        xs = [d for d, l in zip(days, labels) if l == value and (l is None) == (value is None)]
        ys = [rows[value] + _offset(i) for i, (d, l) in enumerate(zip(days, labels))
              if l == value and (l is None) == (value is None)]
        p.points(xs, ys, role=role, r=2.4, hollow=hollow)
    return p


def _rates(c: dict, words: dict):
    names = [words["as_kept"], words["right"], words["early"]]
    values = [c["as_kept"], c["right"], c["early"]]
    p = chart.Panel((-0.6, 2.6), (0, 112), title=words["rate"], xticks=[0, 1, 2], xtick_labels=names,
                    yticks=[0, 25, 50, 75, 100])
    t = 100 * true_rate()
    p.line([-0.6, 2.6], [t, t], role="ink-2", width=1, dashed=True)   # under the bars and their numbers
    for i, v in enumerate(values):
        if v is not None:
            p.rect(i - 0.3, 0, i + 0.3, 100 * v, role="ink" if i == 1 else "ink-3", solid=True)
            p.text(i, 100 * v + 5, f"{round(100 * v)}%", role="ink", anchor="middle", halo=True)
    return p


def today_picture(today: int, words: dict, lang: str) -> str:
    """Left: every order placed so far in the row of its label. Right: the return rate three ways, beside the share
    that will really come back (dashed), which only the simulation knows."""
    days, back = seen(today)
    labels = [right_label(o, b, today) for o, b in zip(days, back)]
    left = _timeline(days, labels, today, words)
    right = _rates(counts(today), words)
    return chart.figure([left, right], alt=words["alt"], width=330, height=260)


# The lesson is done when these are correct. "what-clicks-say" is the choice question in the lesson.
REQUIRED = ["label", "return-rate", "what-clicks-say"]


class _Val:
    """A test value that the check's messages write by its name, such as `labels`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value)


def _function(seen_):
    fn = seen_.mine.names.get(seen_.exercise.function) if seen_.mine is not None else None
    return fn if callable(fn) else None


def _quiet(fn, *args):
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return checking.with_time_limit(lambda: fn(*args)), True
    except Exception:
        return None, False


def _key(text: str) -> str:
    return ('<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.15rem">'
            + text + "</div>")


_SAME = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: any("..." in line.split("#")[0] for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_WORDS = {
    "orders": {"en": "Orders and their labels", "nl": "Bestellingen en hun labels"},
    "day": {"en": "day of the order", "nl": "dag van de bestelling"},
    "unknown": {"en": "not known", "nl": "niet bekend"},
    "kept": {"en": "kept", "nl": "gehouden"},
    "returned": {"en": "returned", "nl": "teruggestuurd"},
    "rate": {"en": "Return rate (%)", "nl": "Retourpercentage (%)"},
    "as_kept": {"en": "unknown = 0", "nl": "onbekend = 0"},
    "right": {"en": "wait 30 days", "nl": "na 30 dagen"},
    "early": {"en": "early too", "nl": "ook vroeg"},
}


def words(lang: str) -> dict:
    return {k: v[lang] for k, v in _WORDS.items()}


# --- exercise 1: a label only after 30 days ---------------------------------------------------------------------------

_LB_HEAD = "def label(ordered, returned, today):\n    age = today - ordered\n"


def _lb(body: str = "    if age < 30:\n        return None\n    if returned:\n        return 1\n    return 0\n") -> str:
    return _LB_HEAD + body


_LB_START = _lb("    ...\n")
_LB = _lb()
_LB_VISIBLE = [(o, r, EXAMPLE_TODAY) for o, r in zip(EXAMPLE_ORDERED, EXAMPLE_RETURNED)]
_LB_HIDDEN = [(0, False, 30), (0, True, 29), (100, True, 180), (179, False, 180), (150, False, 180)]


def _same_label(got, want) -> bool:
    return (got is None) == (want is None) and got == want


_LB_WORDS = {
    "title": {"en": "Your labels for the 300 orders on day 180", "nl": "Jouw labels voor de 300 bestellingen op dag 180"},
    "right": {"en": "Your labels give {settled} orders a label: those at least 30 days old, of which {back} came back, {rate}. The {young} orders of the last 30 days, in the shaded band, are not known yet, also the {young_back} that already came back.",
              "nl": "Jouw labels geven {settled} bestellingen een label: die minstens 30 dagen oud zijn, waarvan er {back} terugkwamen, {rate}. De {young} bestellingen van de laatste 30 dagen, in de grijze band, zijn nog niet bekend, ook de {young_back} die al terugkwamen."},
    "said": {"en": "Your labels: {ones} returned, {zeros} kept and {nones} not known, a return rate of {rate} over the known ones. Waiting 30 days for every order gives {settled} labels and {right}.",
             "nl": "Jouw labels: {ones} teruggestuurd, {zeros} gehouden en {nones} niet bekend, een retourpercentage van {rate} over de bekende. 30 dagen wachten voor elke bestelling geeft {settled} labels en {right}."},
}


def _lb_picture(seen_) -> str:
    lang = seen_.lang
    fn = _function(seen_)
    if fn is None:
        return ""
    days, back = seen(180)
    got = []
    for o, b in zip(days, back):
        value, ok = _quiet(fn, o, b, 180)
        if not ok or value not in (0, 1, None) or isinstance(value, float) and value not in (0.0, 1.0):
            return ""
        got.append(None if value is None else int(value))
    w = words(lang)
    lw = {k: v[lang] for k, v in _LB_WORDS.items()}
    c = counts(180)
    ones, zeros, nones = got.count(1), got.count(0), got.count(None)
    if got == [right_label(o, b, 180) for o, b in zip(days, back)]:
        text = lw["right"].format(settled=c["settled"], back=c["settled_returned"], rate=percent(c["right"], lang),
                                  young=c["young"], young_back=c["young_returned"])
    else:
        rate = percent(ones / (ones + zeros), lang) if ones + zeros else "–"
        text = lw["said"].format(ones=ones, zeros=zeros, nones=nones, rate=rate, settled=c["settled"],
                                 right=percent(c["right"], lang))
    p = _timeline(days, got, 180, w)
    return yours.figure(lw["title"], chart.figure([p], alt=text, width=460, height=240) + _key(text), alt=text)


EXERCISES = {
    "label": FunctionBox(
        name="label",
        picture=_lb_picture,
        function="label",
        start=_LB_START,
        solution=_LB,
        compare=_same_label,
        visible=_LB_VISIBLE,
        hidden=_LB_HIDDEN,
        first=[_DOTS],
        mistakes=[
            Mistake("no-wait", wrong=_lb("    if returned:\n        return 1\n    return 0\n"),
                    en="`{call}` gives {got}: your function labels orders younger than 30 days. An order of 16 days that has not come back, such as `label(44, False, 60)`, can still come back, so its label is not known yet. Give `None` while `age` is below 30, whether or not the order came back.",
                    nl="`{call}` geeft {got}: je functie geeft bestellingen die jonger zijn dan 30 dagen een label. Een bestelling van 16 dagen die niet terugkwam, zoals `label(44, False, 60)`, kan nog terugkomen, dus het label is nog niet bekend. Geef `None` zolang `age` onder 30 is, of de bestelling nu terugkwam of niet."),
            Mistake("zero-young", wrong=_lb("    if age < 30:\n        return 0\n    if returned:\n        return 1\n    return 0\n"),
                    en="`{call}` gives 0, as if the order were kept. But it is younger than 30 days, so its label is not known yet, also when it already came back: give `None`, not 0.",
                    nl="`{call}` geeft 0, alsof de bestelling gehouden is. Maar ze is jonger dan 30 dagen, dus haar label is nog niet bekend, ook als ze al terugkwam: geef `None`, niet 0."),
            Mistake("returns-early", wrong=_lb("    if returned:\n        return 1\n    if age < 30:\n        return None\n    return 0\n"),
                    en="`{call}` gives {got}: the return is real, but the young orders that will be kept have no label yet, so among the newest orders only the returns would get one. Give every order its 30 days first, also one that already came back.",
                    nl="`{call}` geeft {got}: de retour is echt, maar de jonge bestellingen die gehouden worden, hebben nog geen label, dus onder de nieuwste bestellingen zouden alleen de retouren er een krijgen. Geef elke bestelling eerst haar 30 dagen, ook een die al terugkwam."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the `...` with code that gives `None`, 1 or 0 for the order.",
                      "nl": "Vervang de `...` door code die `None`, 1 of 0 geeft voor de bestelling."},
            "arguments": {"en": "The check calls `{call}`, with three values, and the `def` line of your function has a different number of names between its parentheses. Keep `(ordered, returned, today)`.",
                          "nl": "De check roept `{call}` aan, met drie waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(ordered, returned, today)`."},
            "unreached": {"en": "`{call}` gives `None`, as if the label were not known yet. But this order is at least 30 days old: every return is in, so its label is 1 or 0. `None` is only for an `age` below 30. Check your test of `age`, and that every way through the function ends with `return` and 1 or 0.",
                          "nl": "`{call}` geeft `None`, alsof het label nog niet bekend is. Maar deze bestelling is minstens 30 dagen oud: elke retour is binnen, dus het label is 1 of 0. `None` is alleen voor een `age` onder 30. Kijk je test op `age` na, en of elke weg door de functie eindigt met `return` en 1 of 0."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}: `None` while the order is younger than 30 days, then 1 if it came back and 0 if not.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn: `None` zolang de bestelling jonger is dan 30 dagen, daarna 1 als die terugkwam en 0 als dat niet zo is."},
            "almost": {"en": "Right for the eight orders of the lesson. But `{call}` gives {got}, and it should give {want}. Does your function use the `today` it gets, not a fixed day such as 60, and does it wait the full 30 days?",
                       "nl": "Goed voor de acht bestellingen van de les. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `today` die ze krijgt, geen vaste dag zoals 60, en wacht ze de volle 30 dagen?"},
            "correct": {"en": "Your function is right in all {n} cases. On day 60 it gives the first four orders a label and waits for the other four, also the two that already came back.",
                        "nl": "Je functie klopt in alle {n} gevallen. Op dag 60 geeft ze de eerste vier bestellingen een label en wacht ze op de andere vier, ook op de twee die al terugkwamen."},
        },
    ),
}


# --- exercise 2: the return rate of the known labels -------------------------------------------------------------------

_RR_HEAD = "def return_rate(labels):\n    returned = 0\n    known = 0\n    for value in labels:\n"


def _rr(body: str = "        if value == 1:\n            returned = returned + 1\n            known = known + 1\n        if value == 0:\n            known = known + 1\n") -> str:
    return _RR_HEAD + body + "    return returned / known\n"


_RR_START = _rr("        ...\n")
_RR = _rr()
_RR_VISIBLE = (_Val("[0, 1, 0, 0, None, None, None, None]", [0, 1, 0, 0, None, None, None, None]),)
_RR_HIDDEN = [([1, None, 0, 1, 1],), ([0, 0, 1],), ([None, None, 1, 0],), ([0, 0, 0, 0, 1, None],)]


def _same_rate(got, want) -> bool:
    try:
        return abs(float(got) - float(want)) < 1e-9
    except (TypeError, ValueError):
        return False


def _shown_rate(v):
    try:
        text = f"{float(v):.4g}"
    except (TypeError, ValueError):
        return repr(v)[:60]
    return {"en": text, "nl": text.replace(".", ",")}


_RR_WORDS = {
    "title": {"en": "Your return rate from day 60 to day 180", "nl": "Jouw retourpercentage van dag 60 tot en met dag 180"},
    "day": {"en": "today", "nl": "vandaag"},
    "rate": {"en": "return rate (%)", "nl": "retourpercentage (%)"},
    "right": {"en": "Your function gives the return rate of the orders with a label on the days of the slider from day 60 on: {rates}. On day 30 no order has a label yet. The dashed line is the share that will really come back, {true}, which only the simulation knows.",
              "nl": "Je functie geeft het retourpercentage van de bestellingen met een label op de dagen van de slider vanaf dag 60: {rates}. Op dag 30 heeft nog geen enkele bestelling een label. De gestreepte lijn is het deel dat echt terugkomt, {true}, en dat weet alleen de simulatie."},
    "said": {"en": "Your function gives {mine} on the days of the slider from day 60 on; the share of returns among the known labels is {rates}.",
             "nl": "Je functie geeft {mine} op de dagen van de slider vanaf dag 60; het deel retouren onder de bekende labels is {rates}."},
}


def _rr_picture(seen_) -> str:
    lang = seen_.lang
    fn = _function(seen_)
    if fn is None:
        return ""
    mine, want = [], []
    for today in LABELED:
        days, back = seen(today)
        labels = [right_label(o, b, today) for o, b in zip(days, back)]
        value, ok = _quiet(fn, list(labels))
        try:
            value = float(value)
        except (TypeError, ValueError):
            return ""
        if not ok or value != value or abs(value) > 1000:
            return ""
        mine.append(value)
        want.append(counts(today)["right"])
    rw = {k: v[lang] for k, v in _RR_WORDS.items()}
    rates = ", ".join(percent(v, lang) for v in want)
    if all(abs(a - b) < 1e-9 for a, b in zip(mine, want)):
        text = rw["right"].format(rates=rates, true=percent(true_rate(), lang))
    else:
        text = rw["said"].format(mine=", ".join(percent(v, lang) for v in mine), rates=rates)
    top = max(40, 110 * max(abs(v) for v in mine))   # in percent; a function that gives percent itself gets a tall scale
    p = chart.Panel((45, 195), (0, top), xlabel=rw["day"], ylabel=rw["rate"], xticks=LABELED,
                    yticks=[t for t in range(0, 101, 10) if t <= top] if top <= 100 else None)
    t = 100 * true_rate()
    p.line([45, 195], [t, t], role="ink-2", width=1, dashed=True)
    p.line(LABELED, [100 * v for v in want], role="ink-3", width=1.4)
    p.points(LABELED, [100 * v for v in mine], role="ink", r=4)
    return yours.figure(rw["title"], chart.figure([p], alt=text, width=460, height=240) + _key(text), alt=text)


EXERCISES["return-rate"] = FunctionBox(
    name="return-rate",
    picture=_rr_picture,
    function="return_rate",
    start=_RR_START,
    solution=_RR,
    show=_shown_rate,
    compare=_same_rate,
    visible=[_RR_VISIBLE],
    hidden=_RR_HIDDEN,
    first=[_DOTS],
    mistakes=[
        Mistake("all", wrong=_rr("        if value == 1:\n            returned = returned + 1\n        known = known + 1\n"),
                en="`{call}` gives {got}: `known` counts the unknown labels too, as if those orders were kept. Count only the labels that are 1 or 0.",
                nl="`{call}` geeft {got}: `known` telt de onbekende labels ook, alsof die bestellingen gehouden zijn. Tel alleen de labels die 1 of 0 zijn."),
        Mistake("kept-only", wrong=_rr("        if value == 1:\n            returned = returned + 1\n        if value == 0:\n            known = known + 1\n"),
                en="`{call}` gives {got}: `known` counts only the kept orders. The returned ones have a known label too: count both.",
                nl="`{call}` geeft {got}: `known` telt alleen de gehouden bestellingen. De teruggestuurde hebben ook een bekend label: tel ze allebei."),
        Mistake("zero", error=lambda e: isinstance(e, ZeroDivisionError),
                en="Python cannot divide by 0: `known` is still 0 at the end. Add 1 to it for every label that is 1 or 0.",
                nl="Python kan niet door 0 delen: `known` is aan het eind nog 0. Tel er 1 bij op voor elk label dat 1 of 0 is."),
        Mistake("none-compare", error=lambda e: isinstance(e, TypeError) and "NoneType" in str(e) and "not supported" in str(e),
                en="`None` cannot be compared with `<` or `>`. Test a label with `==`, such as `value == 1`.",
                nl="`None` kun je niet vergelijken met `<` of `>`. Test een label met `==`, zoals `value == 1`."),
        Mistake("none-add", error=lambda e: isinstance(e, TypeError) and "NoneType" in str(e) and "unsupported operand" in str(e),
                en="`{call}` stops: Python cannot add `None` to a number. A `None` is a label that is not known yet, so it counts in neither count: add 1 only for a label that is 1 or 0, tested with `==`, such as `value == 1`.",
                nl="`{call}` stopt: Python kan `None` niet bij een getal optellen. Een `None` is een label dat nog niet bekend is, dus het telt bij geen van beide mee: tel alleen 1 op bij een label dat 1 of 0 is, getest met `==`, zoals `value == 1`."),
    ],
    texts=_SAME | {
        "empty": {"en": "Replace the `...` with code that counts the returned orders and the orders with a known label.",
                  "nl": "Vervang de `...` door code die de teruggestuurde bestellingen telt en de bestellingen met een bekend label."},
        "arguments": {"en": "The check calls `{call}`, with one value, and the `def` line of your function has a different number of names between its parentheses. Keep `(labels)`.",
                      "nl": "De check roept `{call}` aan, met één waarde, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(labels)`."},
        "wrong": {"en": "`{call}` gives {got}, and it should give {want}: the returned orders divided by the orders whose label is known, 1 or 0.",
                  "nl": "`{call}` geeft {got}, en dat moet {want} zijn: de teruggestuurde bestellingen gedeeld door de bestellingen waarvan het label bekend is, 1 of 0."},
        "almost": {"en": "Right for the labels of the lesson. But `{call}` gives {got}, and it should give {want}. Does your function count the `labels` it gets, and no fixed number such as 4?",
                   "nl": "Goed voor de labels van de les. Maar `{call}` geeft {got}, en dat moet {want} zijn. Telt je functie de `labels` die ze krijgt, en geen vast getal zoals 4?"},
        "correct": {"en": "Your function is right in all {n} cases. On the eight orders it gives 0.25: one of the four orders with a label came back.",
                    "nl": "Je functie klopt in alle {n} gevallen. Op de acht bestellingen geeft ze 0,25: een van de vier bestellingen met een label kwam terug."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("label", _LB_START, "empty", None),
    ("label", _lb("    if age < 30:\n        return None\n    ...\n"), "wrong", "dots-left"),
    ("label", _LB, "correct", None),
    ("label", _lb("    if age >= 30:\n        if returned:\n            return 1\n        else:\n            return 0\n    return None\n"), "correct", None),
    ("label", _lb("    if age < 30:\n        return None\n    return returned\n"), "correct", None),
    ("label", _lb("    if returned:\n        return 1\n    return 0\n"), "wrong", "no-wait"),
    ("label", _lb("    if returned:\n        return 1\n    if age < 30:\n        return None\n    return 0\n"), "wrong", "returns-early"),
    ("label", _lb("    if age < 30:\n        return 0\n    if returned:\n        return 1\n    return 0\n"), "wrong", "zero-young"),
    ("label", _lb("    if age <= 30:\n        return None\n    if returned:\n        return 1\n    return 0\n"), "wrong", "unreached"),
    ("return-rate", _RR_START, "empty", None),
    ("return-rate", _rr("        if value == 1:\n            ...\n"), "wrong", "dots-left"),
    ("return-rate", _RR, "correct", None),
    ("return-rate", _rr("        if value == 1:\n            returned = returned + 1\n        if value == 1:\n            known = known + 1\n        if value == 0:\n            known = known + 1\n"), "correct", None),
    ("return-rate", _rr("        if value == 1:\n            returned = returned + 1\n        known = known + 1\n"), "wrong", "all"),
    ("return-rate", _rr("        if value == 1:\n            returned = returned + 1\n        if value == 0:\n            known = known + 1\n"), "wrong", "kept-only"),
    ("return-rate", _rr("        if value == 1:\n            returned = returned + 1\n"), "error", "zero"),
    ("return-rate", _rr("        if value > 0:\n            returned = returned + 1\n        known = known + 1\n"), "error", "none-compare"),
    ("return-rate", _rr("        returned = returned + value\n        known = known + 1\n"), "error", "none-add"),
]
