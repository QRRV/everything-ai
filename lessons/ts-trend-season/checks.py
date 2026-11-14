# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ts-trend-season: the bikes rented by registered riders of Capital Bikeshare in Washington, D.C.,
every day of 2011 and 2012 (data/bike-sharing-day.csv.gz), split into a trend, a weekly season and a remainder with a
centered moving average of every window on the slider; and the functions moving_average and season_of."""
import contextlib
import functools
import io
import math
import warnings

from everything_ai import chart, check as checking, paths, yours
from everything_ai.check import FunctionBox, Mistake

DATA = "data/bike-sharing-day.csv.gz"   # see data/README.md
WINDOWS = [3, 5, 7, 21, 35]             # the stops of the slider, in days: odd, so that the window has a middle
DAYS = {"en": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], "nl": ["ma", "di", "wo", "do", "vr", "za", "zo"]}


# --- the riders -----------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def table():
    import pandas as pd
    d = pd.read_csv(paths.ROOT / DATA)
    d["dteday"] = pd.to_datetime(d["dteday"])
    return d.set_index("dteday")


def riders(column: str = "registered"):
    """The bikes rented every day by one kind of rider, as floats with the date as the index."""
    return table()[column].astype(float)


@functools.lru_cache(maxsize=None)
def decompose(window: int, column: str = "registered") -> dict:
    """The series split as the lesson does: the trend a centered moving average over `window` days, the season the
    mean of what is left for every weekday (0 is Monday), shifted to add up to 0, and the remainder what is left after
    both. With the numbers the slider and the text give."""
    y = riders(column)
    trend = y.rolling(window, center=True).mean()
    rest = y - trend
    season = rest.groupby(rest.index.dayofweek).mean()
    season = season - season.mean()
    remainder = rest - season.loc[y.index.dayofweek].to_numpy()
    return {"trend": trend, "season": [float(v) for v in season], "remainder": remainder,
            "move": float(trend.diff().abs().mean()), "span": float(season.max() - season.min()),
            "spread": float(remainder.std()), "low": float(trend.min()), "high": float(trend.max())}


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{abs(x):,.{digits}f}"
    text = text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text
    return ("−" if x < 0 and round(abs(x), digits) != 0 else "") + text


def signed(x: float, lang: str) -> str:
    return ("+" if round(x) > 0 else "") + number(x, lang)


def said(window: int, lang: str) -> str:
    """The sentence under the picture of the slider."""
    w = decompose(window)
    s = w["season"]
    lo, hi = min(range(7), key=lambda k: s[k]), max(range(7), key=lambda k: s[k])
    names = {"en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
             "nl": ["maandag", "dinsdag", "woensdag", "donderdag", "vrijdag", "zaterdag", "zondag"]}[lang]
    n = lambda x: number(x, lang)
    if lang == "nl":
        whole = " Het window bevat elke dag van de week even vaak." if window % 7 == 0 else " Het window bevat niet elke dag van de week even vaak."
        return (f"Window van {window} dagen: de trend beweegt gemiddeld {n(w['move'])} per dag.{whole} Het seizoen beslaat {n(w['span'])}, "
                f"van {names[lo]} {signed(s[lo], lang)} tot {names[hi]} {signed(s[hi], lang)}; de standaardafwijking van de rest is {n(w['spread'])}.")
    whole = " The window holds every weekday equally often." if window % 7 == 0 else " The window does not hold every weekday equally often."
    return (f"Window of {window} days: the trend moves {n(w['move'])} a day on average.{whole} The season spans {n(w['span'])}, "
            f"from {names[lo]} {signed(s[lo], lang)} to {names[hi]} {signed(s[hi], lang)}; the standard deviation of the remainder is {n(w['spread'])}.")


# --- the picture of the slider ---------------------------------------------------------------------------------------

def _bars(p, values: list, role: str, solid: bool, width: float = 0.36):
    for k, v in enumerate(values):
        if abs(v) > 0.5:
            p.rect(k - width, min(0.0, v), k + width, max(0.0, v), role=role, solid=solid)


def picture(window: int, labels: dict) -> str:
    """Left: the riders of every day as a thin gray line and the trend over `window` days as a thick one. Right: the
    season, one bar per weekday from Monday to Sunday."""
    y = riders()
    w = decompose(window)
    n = len(y)
    left = chart.Panel((0, n - 1), (0, 7200), title=labels["left"], ylabel=labels["ylabel"],
                       xticks=[0, 181, 365, 547], xtick_labels=labels["months"], yticks=[0, 2000, 4000, 6000])
    left.line(list(range(n)), [float(v) for v in y], role="ink-3", width=0.8)
    t = w["trend"].to_numpy()
    xs = [i for i in range(n) if not math.isnan(t[i])]
    left.line(xs, [float(t[i]) for i in xs], role="ink", width=2.0)
    right = chart.Panel((-0.6, 6.6), (-900, 600), title=labels["right"], ylabel=labels["ylabel"], xticks=list(range(7)),
                        xtick_labels=DAYS[labels["lang"]], yticks=[-800, -400, 0, 400])
    right.line([-0.6, 6.6], [0, 0], role="ink-3", width=0.8)
    _bars(right, w["season"], "data", True)
    return chart.figure([left, right], alt=labels["alt"], width=330, height=260)


# The lesson is done when these are correct. "window" is the choice question in the lesson.
REQUIRED = ["moving-average", "season-of", "window"]


class _Val:
    """A test value that the check's messages write by its name, such as `weeks`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value)


def _one(v) -> str:
    v = float(v)
    if math.isnan(v):
        return "nan"
    return str(round(v, 1)) if v != int(v) else str(float(int(v)))


def _shown(v):
    import numpy as np
    if isinstance(v, np.ndarray) and v.ndim == 1 and v.dtype.kind in "iuf":
        if len(v) <= 8:
            return "[" + ", ".join(_one(x) for x in v) + "]"
        return {"en": "[" + ", ".join(_one(x) for x in v[:5]) + f", …], {len(v)} values",
                "nl": "[" + ", ".join(_one(x) for x in v[:5]) + f", …], {len(v)} waarden"}
    if isinstance(v, float):
        return str(round(v, 3))
    return f"`{v!r}`"[:60]


_DOTS = Mistake("dots-left", matches=lambda got, text: any("..." in line.split("#")[0] for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _call(seen, case: tuple):
    fn = seen.mine.names.get(seen.exercise.function) if seen.mine is not None else None
    if not callable(fn):
        return None
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return checking.with_time_limit(lambda: fn(*(checking._copy(a) for a in case)))
    except Exception:
        return None


def _words(lang: str, table_: dict) -> dict:
    return {k: v[lang] for k, v in table_.items()}


def _key(text: str) -> str:
    return ('<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.15rem">'
            + text + "</div>")


# --- exercise 1: a centered moving average ----------------------------------------------------------------------------

def _weeks():
    """Eight weeks of registered riders, Monday 2 January to Sunday 26 February 2012, as an array."""
    return riders().loc["2012-01-02":"2012-02-26"].to_numpy()


def _array(values: list) -> _Val:
    """A short array the check's messages write as the code that makes it."""
    import numpy as np
    name = "np.array([" + ", ".join("np.nan" if math.isnan(v) else repr(float(v)) for v in values) + "])"
    return _Val(name, np.array(values, dtype=float))


_WEEKS = _Val("weeks", _weeks())
_SMALL = _array([1, 5, 3, 7, 5, 9, 7])

_M_HEAD = "import numpy as np\n\n\ndef moving_average(y, half):\n    trend = np.full(len(y), np.nan)\n    for i in range(half, len(y) - half):\n"


def _m(mean: str = "np.mean(y[i - half:i + half + 1])") -> str:
    return _M_HEAD + f"        trend[i] = {mean}\n    return trend\n"


_M_START = _m("...")
_M = _m()


def _moving(y, half):
    import numpy as np
    y = np.asarray(y, dtype=float)
    out = np.full(len(y), np.nan)
    for i in range(half, len(y) - half):
        out[i] = np.mean(y[i - half:i + half + 1])
    return out


_MA_WORDS = {
    "title": {"en": "Your moving_average(weeks, 3)", "nl": "Jouw moving_average(weeks, 3)"},
    "ylabel": {"en": "registered riders", "nl": "geregistreerde fietsers"},
    "said": {"en": "Gray dots: the registered riders of every day, Monday 2 January to Sunday 26 February 2012. The thick line is what your function gives, the dashed one the course's trend: the mean of the 3 days before, the day itself and the 3 after.",
             "nl": "Grijze stippen: de geregistreerde fietsers van elke dag, maandag 2 januari tot zondag 26 februari 2012. De dikke lijn is wat jouw functie geeft, de gestreepte de trend van de course: het gemiddelde van de 3 dagen ervoor, de dag zelf en de 3 erna."},
    "right": {"en": "Gray dots: the registered riders of every day, Monday 2 January to Sunday 26 February 2012. The thick line is your trend, the course's: the dips of the weekends are gone from it, because every 7 days hold every weekday once. The first and last 3 days have no trend.",
              "nl": "Grijze stippen: de geregistreerde fietsers van elke dag, maandag 2 januari tot zondag 26 februari 2012. De dikke lijn is jouw trend, die van de course: de dalen van de weekenden zijn eruit, want elke 7 dagen bevatten elke dag van de week één keer. De eerste en laatste 3 dagen hebben geen trend."},
}


def _ma_picture(seen) -> str:
    import numpy as np
    got = _call(seen, (_WEEKS, 3))
    if got is None:
        return ""
    try:
        got = np.asarray(got, dtype=float).ravel()
    except (TypeError, ValueError):
        return ""
    y = _WEEKS.value
    if len(got) == 0 or not np.isfinite(got).any():
        return ""
    want = _moving(y, 3)
    lang = seen.lang
    words = _words(lang, _MA_WORDS)
    n = len(y)
    p = chart.Panel((0, n - 1), (0, 4500), ylabel=words["ylabel"], xticks=[0, 14, 28, 42], yticks=[0, 1500, 3000, 4500],
                    xtick_labels={"en": ["2 Jan", "16 Jan", "30 Jan", "13 Feb"], "nl": ["2 jan", "16 jan", "30 jan", "13 feb"]}[lang])
    p.points(list(range(n)), [float(v) for v in y], role="ink-3", r=2.6)
    keep = [i for i in range(n) if not math.isnan(want[i])]
    p.line(keep, [float(want[i]) for i in keep], role="ink-3", width=1.8, dashed=True)
    mine = [i for i in range(min(n, len(got))) if math.isfinite(got[i])]
    if mine:
        p.line(mine, [max(0.0, min(4500.0, float(got[i]))) for i in mine], role="ink", width=2.4)
    right = len(got) == n and checking.same_value(got, want)
    text = words["right"] if right else words["said"]
    return yours.figure(words["title"], chart.figure([p], alt=text, width=460, height=250) + _key(text), alt=text)


EXERCISES = {
    "moving-average": FunctionBox(
        name="moving-average",
        picture=_ma_picture,
        function="moving_average",
        start=_M_START,
        solution=_M,
        show=_shown,
        visible=[(_WEEKS, 3)],
        hidden=[(_SMALL, 0), (_SMALL, 1), (_SMALL, 2), (_SMALL, 3), (_array([4, 4, 4, 4, 4]), 2)],
        first=[_DOTS],
        mistakes=[
            Mistake("short", wrong=_m("np.mean(y[i - half:i + half])"),
                    en="`{call}` gives {got}, and it should give {want}. Your window leaves out the last day: a slice stops just before its end, so the days up to `i + half` need `i + half + 1`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je window laat de laatste dag weg: een slice stopt vlak voor zijn einde, dus de dagen tot en met `i + half` vragen `i + half + 1`."),
            Mistake("trailing", wrong=_m("np.mean(y[i - 2 * half:i + 1])"),
                    en="`{call}` gives {got}, and it should give {want}. Your window ends at the day itself, as `rolling` does: then the trend runs behind. Center it: `half` days before and `half` days after.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je window eindigt bij de dag zelf, zoals `rolling` doet: dan loopt de trend achter. Zet hem in het midden: `half` dagen ervoor en `half` dagen erna."),
            Mistake("sum", wrong=_m("np.sum(y[i - half:i + half + 1])"),
                    en="`{call}` gives {got}, and it should give {want}: the sum of the window. The trend is its mean.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de som van het window. De trend is het gemiddelde ervan."),
            Mistake("whole", wrong=_m("np.mean(y)"),
                    en="`{call}` gives {got}, and it should give {want}: the mean of all days, every time. Take the mean of the days around day `i` only.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: steeds het gemiddelde van alle dagen. Neem alleen het gemiddelde van de dagen rond dag `i`."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the `...`: the mean of the window around day `i`.",
                      "nl": "Vervang de `...`: het gemiddelde van het window rond dag `i`."},
            "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has a different number of names between its parentheses. Keep `(y, half)`.",
                          "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(y, half)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is `trend[i]` the mean of the `half` days before day `i`, day `i` itself and the `half` days after it?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Is `trend[i]` het gemiddelde van de `half` dagen voor dag `i`, dag `i` zelf en de `half` dagen erna?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function use `half`, and work for any `y`?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `half`, en werkt hij voor elke `y`?"},
            "correct": {"en": "Your function is right in all {n} cases. With `half` 3 every window holds 7 days, every weekday once, so the dips of the weekends are gone from the trend. With `half` 0 the trend is every day itself.",
                        "nl": "Je functie klopt in alle {n} gevallen. Met `half` 3 bevat elk window 7 dagen, elke dag van de week één keer, dus zijn de dalen van de weekenden uit de trend. Met `half` 0 is de trend elke dag zelf."},
        },
    ),
}


# --- exercise 2: the season, the mean per position ----------------------------------------------------------------------

def _rest():
    """The registered riders of every day minus their trend over 7 days: NaN for the first and last 3 days. Position 0
    is Saturday 1 January 2011."""
    w = decompose(7)
    return (riders() - w["trend"]).to_numpy()


_REST = _Val("rest", _rest())

_S_HEAD = "import numpy as np\n\n\ndef season_of(rest, period):\n    means = np.zeros(period)\n    for k in range(period):\n"


def _s(mean: str = "np.nanmean(rest[k::period])", back: str = "means - np.mean(means)") -> str:
    return _S_HEAD + f"        means[k] = {mean}\n    return {back}\n"


_S_START = _s("...", "means - ...")
_S = _s()


def _season(rest, period):
    import numpy as np
    rest = np.asarray(rest, dtype=float)
    means = np.array([np.nanmean(rest[k::period]) for k in range(period)])
    return means - means.mean()


_SE_WORDS = {
    "title": {"en": "Your season_of(rest, 7)", "nl": "Jouw season_of(rest, 7)"},
    "ylabel": {"en": "riders above or below the trend", "nl": "fietsers boven of onder de trend"},
    "days": {"en": ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"], "nl": ["za", "zo", "ma", "di", "wo", "do", "vr"]},
    "said": {"en": "Filled bars: what your function gives for every position of the week; open bars: the course's season. Position 0 is a Saturday, because 1 January 2011 was one.",
             "nl": "Gevulde staven: wat jouw functie geeft voor elke plek in de week; open staven: het seizoen van de course. Plek 0 is een zaterdag, want 1 januari 2011 was er een."},
    "right": {"en": "Your season, the course's: Saturday {sat} and Sunday {sun}, Thursday {thu}. Position 0 is a Saturday, because 1 January 2011 was one. The seven bars add up to 0.",
              "nl": "Jouw seizoen, dat van de course: zaterdag {sat} en zondag {sun}, donderdag {thu}. Plek 0 is een zaterdag, want 1 januari 2011 was er een. De zeven staven tellen op tot 0."},
}


def _se_picture(seen) -> str:
    import numpy as np
    got = _call(seen, (_REST, 7))
    if got is None:
        return ""
    try:
        got = np.asarray(got, dtype=float).ravel()
    except (TypeError, ValueError):
        return ""
    if len(got) == 0 or not np.isfinite(got).any():
        return ""
    want = _season(_REST.value, 7)
    lang = seen.lang
    words = _words(lang, _SE_WORDS)
    p = chart.Panel((-0.6, 6.6), (-900, 600), ylabel=words["ylabel"], xticks=list(range(7)), xtick_labels=words["days"],
                    yticks=[-800, -400, 0, 400])
    p.line([-0.6, 6.6], [0, 0], role="ink-3", width=0.8)
    _bars(p, [float(v) for v in want], "ink-3", False, 0.4)
    mine = [max(-900.0, min(600.0, float(v))) if math.isfinite(v) else 0.0 for v in got[:7]]
    _bars(p, mine, "data", True, 0.26)
    right = len(got) == 7 and checking.same_value(got, want)
    text = (words["right"] if right else words["said"]).format(
        sat=signed(want[0], lang), sun=signed(want[1], lang), thu=signed(want[5], lang))
    return yours.figure(words["title"], chart.figure([p], alt=text, width=460, height=250) + _key(text), alt=text)


EXERCISES["season-of"] = FunctionBox(
    name="season-of",
    picture=_se_picture,
    function="season_of",
    start=_S_START,
    solution=_S,
    show=_shown,
    visible=[(_REST, 7)],
    hidden=[(_array([1, 2, 3, 1, 2, 3]), 3), (_array([math.nan, 2, 3, 1, 2, math.nan]), 3),
            (_array([5, 1, 5, 1, math.nan, 1]), 2), (_array([7, 9, 8]), 1)],
    first=[_DOTS],
    mistakes=[
        Mistake("nan", wrong=_s(mean="np.mean(rest[k::period])"),
                en="`{call}` gives {got}, and it should give {want}. Where the trend has no value, `rest` is NaN, and `np.mean` of values with a NaN is NaN. `np.nanmean` leaves the NaN out.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. Waar de trend geen waarde heeft, is `rest` NaN, en `np.mean` van waarden met een NaN is NaN. `np.nanmean` laat de NaN weg."),
        Mistake("not-centered", wrong=_s(back="means"),
                en="`{call}` gives {got}, and it should give {want}. Your means are right, but they do not add up to 0: subtract their mean, so the season does not move the level, which is the trend's.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. Je gemiddelden kloppen, maar ze tellen niet op tot 0: trek hun gemiddelde ervan af, zodat het seizoen het niveau niet verschuift; dat is van de trend."),
        Mistake("slice", wrong=_s(mean="np.nanmean(rest[k:period])"),
                en="`{call}` gives {got}, and it should give {want}. `rest[k:period]` is the values from position `k` up to `period`: one stretch. Every `period`-th value from position `k` is `rest[k::period]`.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. `rest[k:period]` zijn de waarden van plek `k` tot `period`: één stuk. Elke `period`-de waarde vanaf plek `k` is `rest[k::period]`."),
    ],
    texts=_SAME | {
        "empty": {"en": "Replace the two `...`: the mean of every `period`-th value from position `k`, NaN left out, and what makes the season add up to 0.",
                  "nl": "Vervang de twee `...`: het gemiddelde van elke `period`-de waarde vanaf plek `k`, zonder NaN, en wat het seizoen laat optellen tot 0."},
        "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has a different number of names between its parentheses. Keep `(rest, period)`.",
                      "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(rest, period)`."},
        "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is `means[k]` the mean of `rest[k]`, `rest[k + period]`, `rest[k + 2 * period]` and so on, NaN left out, and do you subtract the mean of `means`?",
                  "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Is `means[k]` het gemiddelde van `rest[k]`, `rest[k + period]`, `rest[k + 2 * period]` enzovoort, zonder NaN, en trek je het gemiddelde van `means` ervan af?"},
        "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function use `period`, and work for any `rest`?",
                   "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `period`, en werkt hij voor elke `rest`?"},
        "correct": {"en": "Your function is right in all {n} cases. The season of the registered riders, from Saturday: −560, −760, +15, +291, +333, +411, +271. With `period` 1 there is no season: 0.",
                    "nl": "Je functie klopt in alle {n} gevallen. Het seizoen van de geregistreerde fietsers, vanaf zaterdag: −560, −760, +15, +291, +333, +411, +271. Met `period` 1 is er geen seizoen: 0."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("moving-average", _M_START, "empty", None),
    ("moving-average", _M, "correct", None),
    ("moving-average", _m("y[i - half:i + half + 1].mean()"), "correct", None),
    ("moving-average", _m("np.sum(y[i - half:i + half + 1]) / (2 * half + 1)"), "correct", None),
    ("moving-average", _m("np.mean(y[i - half:i + half])"), "wrong", "short"),
    ("moving-average", _m("np.mean(y[i - 2 * half:i + 1])"), "wrong", "trailing"),
    ("moving-average", _m("np.sum(y[i - half:i + half + 1])"), "wrong", "sum"),
    ("moving-average", _m("np.mean(y)"), "wrong", "whole"),
    ("moving-average", _m("np.mean(y[i - 3:i + 4])"), "almost", None),
    ("season-of", _S_START, "empty", None),
    ("season-of", _s(back="means - ..."), "wrong", "dots-left"),
    ("season-of", _S, "correct", None),
    ("season-of", _s(back="means - means.mean()"), "correct", None),
    ("season-of", _s(mean="np.nanmean(rest[k::period])", back="means - np.sum(means) / period"), "correct", None),
    ("season-of", _s(mean="np.mean(rest[k::period])"), "wrong", "nan"),
    ("season-of", _s(back="means"), "wrong", "not-centered"),
    ("season-of", _s(back="means - 0"), "wrong", "not-centered"),
    ("season-of", _s(mean="np.nanmean(rest[k:period])"), "wrong", "slice"),
    ("season-of", _s(mean="np.nanmean(rest[k::7])"), "almost", None),
]
