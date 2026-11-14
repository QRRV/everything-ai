# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ts-autocorrelation: the bikes rented by registered riders of Capital Bikeshare every hour of
2011 and 2012 (data/bike-sharing-hour.csv.gz) on a grid of hours, their autocorrelation at every lag of the slider,
and the functions correlation and autocorrelation."""
import contextlib
import functools
import io
import math
import warnings

from everything_ai import chart, check as checking, paths, yours
from everything_ai.check import FunctionBox, Mistake

DAYS = "data/bike-sharing-day.csv.gz"     # see data/README.md
HOURS = "data/bike-sharing-hour.csv.gz"
LAGS = [1, 2, 3, 6, 12, 24, 48, 167, 168, 336]   # the stops of the slider, in hours
LONGEST = 336                                    # the bars on the right go up to two weeks
SAMPLE = 1500                                    # pairs of hours drawn on the left, chosen at random with a fixed seed


# --- the riders -----------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def days():
    import pandas as pd
    d = pd.read_csv(paths.ROOT / DAYS)
    d["dteday"] = pd.to_datetime(d["dteday"])
    return d.set_index("dteday")


@functools.lru_cache(maxsize=1)
def hours():
    """The bikes rented by registered riders every hour, on a grid of hours: NaN for the 165 hours without a row."""
    import pandas as pd
    h = pd.read_csv(paths.ROOT / HOURS)
    when = pd.to_datetime(h["dteday"]) + pd.to_timedelta(h["hr"], unit="h")
    return h.set_index(when)["registered"].astype(float).asfreq("h")


@functools.lru_cache(maxsize=1)
def acf() -> list:
    """The autocorrelation of the hours at every lag from 0 to LONGEST, as pandas gives it."""
    s = hours()
    return [1.0] + [float(s.autocorr(k)) for k in range(1, LONGEST + 1)]


@functools.lru_cache(maxsize=None)
def pairs(lag: int) -> dict:
    """The pairs of hours `lag` apart that both have a row, their number, and SAMPLE of them chosen at random."""
    import numpy as np
    s = hours()
    now, before = s.to_numpy(), s.shift(lag).to_numpy()
    both = ~np.isnan(now) & ~np.isnan(before)
    pick = np.random.default_rng(0).choice(int(both.sum()), size=SAMPLE, replace=False)
    return {"n": int(both.sum()), "now": now[both][pick], "before": before[both][pick], "r": acf()[lag]}


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{abs(x):,.{digits}f}"
    text = text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text
    return ("−" if x < 0 and round(abs(x), digits) != 0 else "") + text


_EARLIER = {
    "en": {1: "an hour earlier", 24: "the same hour a day earlier", 48: "the same hour two days earlier",
           167: "a week minus an hour earlier", 168: "the same hour a week earlier", 336: "the same hour two weeks earlier"},
    "nl": {1: "een uur eerder", 24: "hetzelfde uur een dag eerder", 48: "hetzelfde uur twee dagen eerder",
           167: "een week min een uur eerder", 168: "hetzelfde uur een week eerder", 336: "hetzelfde uur twee weken eerder"},
}


def said(lag: int, lang: str) -> str:
    """The sentence under the picture of the slider."""
    p = pairs(lag)
    r = number(p["r"], lang, 3)
    if lang == "nl":
        when = _EARLIER["nl"].get(lag, f"{lag} uur eerder")
        return (f"Lag van {lag} uur: elk uur tegen {when}. r = {r}, over {number(p['n'], lang)} paren van uren die allebei een rij hebben; "
                f"links staan er {number(SAMPLE, lang)}, willekeurig gekozen.")
    when = _EARLIER["en"].get(lag, f"{lag} hours earlier")
    unit = "hour" if lag == 1 else "hours"
    return (f"Lag of {lag} {unit}: every hour against {when}. r = {r}, over {number(p['n'], lang)} pairs of hours that both have a row; "
            f"the left shows {number(SAMPLE, lang)} of them, chosen at random.")


# --- the picture of the slider ---------------------------------------------------------------------------------------

def picture(lag: int, labels: dict) -> str:
    """Left: riders now against riders `lag` hours earlier, one dot per pair. Right: the autocorrelation at every lag
    up to two weeks, the chosen lag drawn thick."""
    p = pairs(lag)
    left = chart.Panel((0, 900), (0, 900), title=labels["left"], xlabel=labels["xlabel"], ylabel=labels["ylabel"],
                       xticks=[0, 300, 600, 900], yticks=[0, 300, 600, 900])
    left.points([float(v) for v in p["before"]], [float(v) for v in p["now"]], role="ink-3", r=1.4)
    a = acf()
    right = chart.Panel((0, LONGEST + 6), (-0.4, 1.0), title=labels["right"], xlabel=labels["lag"],
                        xticks=[0, 24, 168, 336], yticks=[-0.4, 0, 0.4, 0.8])
    right.line([0, LONGEST + 6], [0, 0], role="ink-3", width=0.8)
    for k in range(1, LONGEST + 1):
        right.line([k, k], [0, a[k]], role="ink-3", width=0.9)
    right.line([lag, lag], [0, a[lag]], role="ink", width=3.0)
    right.points([lag], [a[lag]], role="ink", r=3.5)
    return chart.figure([left, right], alt=labels["alt"], width=330, height=270)


# The lesson is done when these are correct. "stationary" is the choice question in the lesson.
REQUIRED = ["correlation", "autocorrelation", "stationary"]


class _Val:
    """A test value that the check's messages write by its name, such as `riders`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value)


def _array(values: list) -> _Val:
    """A short array the check's messages write as the code that makes it."""
    import numpy as np
    return _Val("np.array([" + ", ".join(repr(float(v)) for v in values) + "])", np.array(values, dtype=float))


def _shown(v):
    import numpy as np
    if isinstance(v, (float, np.floating)):
        return "nan" if math.isnan(v) else str(round(float(v), 3))
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


def _float(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _key(text: str) -> str:
    return ('<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.15rem">'
            + text + "</div>")


# --- exercise 1: the correlation coefficient ----------------------------------------------------------------------------

def _temp_and_riders():
    d = days()
    return (d["temp"] * 41).to_numpy(), d["registered"].astype(float).to_numpy()


_TEMP = _Val("temp", _temp_and_riders()[0])
_RIDERS_DAY = _Val("riders", _temp_and_riders()[1])

_C_HEAD = "import numpy as np\n\n\ndef correlation(x, y):\n    dx = x - np.mean(x)\n    dy = y - np.mean(y)\n"


def _c(back: str = "np.mean(dx * dy) / (np.std(x) * np.std(y))") -> str:
    return _C_HEAD + f"    return {back}\n"


_C_START = _c("...")
_C = _c()


def _corr(x, y) -> float:
    import numpy as np
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (x.std() * y.std()))


_CO_WORDS = {
    "title": {"en": "Your correlation(temp, riders)", "nl": "Jouw correlation(temp, riders)"},
    "xlabel": {"en": "temperature (°C)", "nl": "temperatuur (°C)"},
    "ylabel": {"en": "registered riders", "nl": "geregistreerde fietsers"},
    "said": {"en": "Every day of 2011 and 2012, filled where temperature and riders are both above or both below their mean, open where one is above and the other below. The thick line rises r standard deviations of riders for every standard deviation of temperature, with your r = {mine}; the dashed one with the course's, {want}.",
             "nl": "Elke dag van 2011 en 2012, gevuld waar temperatuur en fietsers allebei boven of allebei onder hun gemiddelde liggen, open waar de een erboven ligt en de ander eronder. De dikke lijn stijgt r standaardafwijkingen fietsers voor elke standaardafwijking temperatuur, met jouw r = {mine}; de gestreepte met die van de course, {want}."},
    "right": {"en": "Every day of 2011 and 2012, filled where temperature and riders are both above or both below their mean, open where one is above and the other below: 68.5% of the days are filled, so r is positive. The line rises r = {want} standard deviations of riders for every standard deviation of temperature.",
              "nl": "Elke dag van 2011 en 2012, gevuld waar temperatuur en fietsers allebei boven of allebei onder hun gemiddelde liggen, open waar de een erboven ligt en de ander eronder: 68,5% van de dagen is gevuld, dus is r positief. De lijn stijgt r = {want} standaardafwijkingen fietsers voor elke standaardafwijking temperatuur."},
    "none": {"en": "no number", "nl": "geen getal"},
}


def _co_picture(seen) -> str:
    import numpy as np
    mine = _float(_call(seen, (_TEMP, _RIDERS_DAY)))
    x, y = _TEMP.value, _RIDERS_DAY.value
    want = _corr(x, y)
    lang = seen.lang
    words = {k: v[lang] for k, v in _CO_WORDS.items()}
    p = chart.Panel((0, 36), (0, 7200), xlabel=words["xlabel"], ylabel=words["ylabel"], xticks=[0, 10, 20, 30],
                    yticks=[0, 2000, 4000, 6000])
    same = (x - x.mean()) * (y - y.mean()) > 0
    p.points([float(v) for v in x[same]], [float(v) for v in y[same]], role="ink-3", r=2.2)
    p.points([float(v) for v in x[~same]], [float(v) for v in y[~same]], role="ink-3", r=2.2, hollow=True)
    p.line([float(x.mean())] * 2, [0, 7200], role="ink-3", width=0.8, dashed=True)
    p.line([0, 36], [float(y.mean())] * 2, role="ink-3", width=0.8, dashed=True)

    def tilt(r):
        slope = r * y.std() / x.std()
        ends = [0.0, 36.0]
        return ends, [float(y.mean() + slope * (e - x.mean())) for e in ends]

    xs, ys = tilt(want)
    p.line(xs, [max(-2000.0, min(9000.0, v)) for v in ys], role="ink-3", width=1.8, dashed=True)
    if mine is not None and abs(mine) < 50:
        xs, ys = tilt(mine)
        p.line(xs, [max(-2000.0, min(9000.0, v)) for v in ys], role="ink", width=2.6)
    right = mine is not None and abs(mine - want) < 1e-9
    fmt = lambda v: words["none"] if v is None else number(v, lang, 3)
    text = (words["right"] if right else words["said"]).format(mine=fmt(mine), want=fmt(want))
    return yours.figure(words["title"], chart.figure([p], alt=text, width=460, height=270) + _key(text), alt=text)


_LINE = (_array([1, 2, 3, 4]), _array([3, 5, 7, 9]))
_FALLING = (_array([1, 2, 3, 4, 5]), _array([10, 8, 6, 4, 2]))
_BOWL = (_array([-3, -2, -1, 0, 1, 2, 3]), _array([9, 4, 1, 0, 1, 4, 9]))
_FEW = (_array([2, 4, 4, 5, 9]), _array([1, 3, 2, 6, 5]))

EXERCISES = {
    "correlation": FunctionBox(
        name="correlation",
        picture=_co_picture,
        function="correlation",
        start=_C_START,
        solution=_C,
        show=_shown,
        visible=[(_TEMP, _RIDERS_DAY)],
        hidden=[_LINE, _FALLING, _BOWL, _FEW],
        first=[_DOTS],
        mistakes=[
            Mistake("covariance", wrong=_c("np.mean(dx * dy)"),
                    en="`{call}` gives {got}, and it should give {want}. That is the mean of the products, the covariance: its size depends on the units of `x` and `y`. Divide it by the standard deviations of both.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Dat is het gemiddelde van de producten, de covariantie: hoe groot die is, hangt af van de eenheden van `x` en `y`. Deel hem door de standaardafwijkingen van allebei."),
            Mistake("sum", wrong=_c("np.sum(dx * dy) / (np.std(x) * np.std(y))"),
                    en="`{call}` gives {got}, and it should give {want}: as many times too large as there are values. Take the mean of the products, not their sum.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: zoveel keer te groot als er waarden zijn. Neem het gemiddelde van de producten, niet hun som."),
            Mistake("one-std", wrong=_c("np.mean(dx * dy) / (np.std(x) * np.std(x))"),
                    en="`{call}` gives {got}, and it should give {want}. You divide by the standard deviation of `x` twice; one of the two should be that of `y`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je deelt twee keer door de standaardafwijking van `x`; een van de twee moet die van `y` zijn."),
            Mistake("raw", wrong=_c("np.mean(x * y) / (np.std(x) * np.std(y))"),
                    en="`{call}` gives {got}, and it should give {want}. Multiply the distances to the means, `dx` and `dy`, not the values themselves.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Vermenigvuldig de afstanden tot de gemiddelden, `dx` en `dy`, niet de waarden zelf."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the `...`: the mean of the products of `dx` and `dy`, divided by the standard deviations of `x` and `y`.",
                      "nl": "Vervang de `...`: het gemiddelde van de producten van `dx` en `dy`, gedeeld door de standaardafwijkingen van `x` en `y`."},
            "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has a different number of names between its parentheses. Keep `(x, y)`.",
                          "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(x, y)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is it the mean of `dx * dy`, divided by `np.std(x)` times `np.std(y)`?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Is het het gemiddelde van `dx * dy`, gedeeld door `np.std(x)` keer `np.std(y)`?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for any `x` and `y`?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke `x` en `y`?"},
            "correct": {"en": "Your function is right in all {n} cases. Temperature and registered riders: r = 0.540. A rising line gives 1, a falling one −1, and the bowl of `x` and `x ** 2` gives 0: r only measures how well a straight line fits.",
                        "nl": "Je functie klopt in alle {n} gevallen. Temperatuur en geregistreerde fietsers: r = 0,540. Een stijgende lijn geeft 1, een dalende −1, en de kom van `x` en `x ** 2` geeft 0: r meet alleen hoe goed een rechte lijn past."},
        },
    ),
}


# --- exercise 2: the autocorrelation at lag k ---------------------------------------------------------------------------

_A_HEAD = "import numpy as np\n\n\ndef autocorrelation(y, k):\n"


def _a(now: str = "y[k:]", before: str = "y[:len(y) - k]") -> str:
    return _A_HEAD + f"    now = {now}\n    before = {before}\n    return np.corrcoef(now, before)[0, 1]\n"


_A_START = _a("...", "...")
_A = _a()


def _auto(y, k) -> float:
    import pandas as pd
    return float(pd.Series(y).autocorr(k))


_TO = 28   # the picture draws the lags from 1 to 28 days

_AU_WORDS = {
    "title": {"en": "Your autocorrelation(riders, k), k from 1 to 28", "nl": "Jouw autocorrelation(riders, k), k van 1 tot en met 28"},
    "xlabel": {"en": "lag k (days)", "nl": "lag k (dagen)"},
    "said": {"en": "Filled bars: what your function gives for the registered riders of every day, at every lag from 1 to 28 days; open bars: what pandas gives with riders.autocorr(k).",
             "nl": "Gevulde staven: wat jouw functie geeft voor de geregistreerde fietsers van elke dag, bij elke lag van 1 tot en met 28 dagen; open staven: wat pandas geeft met riders.autocorr(k)."},
    "right": {"en": "Your autocorrelation of the registered riders of every day, at every lag from 1 to 28 days, pandas' too. The peaks at 7, 14, 21 and 28 days are the week; that no bar falls below 0.57 is the trend.",
              "nl": "Jouw autocorrelatie van de geregistreerde fietsers van elke dag, bij elke lag van 1 tot en met 28 dagen, ook die van pandas. De pieken bij 7, 14, 21 en 28 dagen zijn de week; dat geen staaf onder 0,57 komt, is de trend."},
}


def _au_picture(seen) -> str:
    want = [_auto(_RIDERS_DAY.value, k) for k in range(1, _TO + 1)]
    got = [_float(_call(seen, (_RIDERS_DAY, k))) for k in range(1, _TO + 1)]
    if all(v is None for v in got):
        return ""
    lang = seen.lang
    words = {k: v[lang] for k, v in _AU_WORDS.items()}
    p = chart.Panel((0.3, _TO + 0.7), (-0.2, 1.0), xlabel=words["xlabel"], xticks=[1, 7, 14, 21, 28], yticks=[0, 0.25, 0.5, 0.75, 1.0])
    p.line([0.3, _TO + 0.7], [0, 0], role="ink-3", width=0.8)
    for k, v in enumerate(want, start=1):
        p.rect(k - 0.42, min(0.0, v), k + 0.42, max(0.0, v), role="ink-3", solid=False)
    for k, v in enumerate(got, start=1):
        if v is not None:
            v = max(-0.2, min(1.0, v))
            p.rect(k - 0.26, min(0.0, v), k + 0.26, max(0.0, v), role="data", solid=True)
    right = all(g is not None and abs(g - w) < 1e-9 for g, w in zip(got, want))
    text = words["right"] if right else words["said"]
    return yours.figure(words["title"], chart.figure([p], alt=text, width=460, height=250) + _key(text), alt=text)


EXERCISES["autocorrelation"] = FunctionBox(
    name="autocorrelation",
    picture=_au_picture,
    function="autocorrelation",
    start=_A_START,
    solution=_A,
    show=_shown,
    visible=[(_RIDERS_DAY, 7)],
    hidden=[(_RIDERS_DAY, 1), (_array([1, 2, 3, 4, 5, 6]), 1), (_array([1, 3, 1, 3, 1, 3, 1, 3]), 1),
            (_array([1, 3, 1, 3, 1, 3, 1, 3]), 2), (_array([4, 1, 5, 2, 6]), 0)],
    first=[_DOTS],
    mistakes=[
        Mistake("minus-k", syntax=r"\[\s*:\s*-\s*k\s*\]",
                en="With `k` 0, `y[:-k]` is `y[:0]`, an empty array: counting from the end does not work for 0. Write the end as `len(y) - k`.",
                nl="Met `k` 0 is `y[:-k]` gelijk aan `y[:0]`, een lege array: tellen vanaf het eind werkt niet voor 0. Schrijf het eind als `len(y) - k`."),
        Mistake("same", wrong=_a("y[k:]", "y[k:]"),
                en="`{call}` gives {got}, and it should give {want}. `now` and `before` are the same values, so they correlate perfectly. `before` is the values `k` steps earlier: from the start of `y`, stopping `k` before its end.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. `now` en `before` zijn dezelfde waarden, dus correleren ze perfect. `before` zijn de waarden `k` stappen eerder: vanaf het begin van `y`, tot `k` voor het eind."),
        Mistake("one", wrong=_a("y[1:]", "y[:len(y) - 1]"),
                en="`{call}` gives {got}, and it should give {want}: the autocorrelation at lag 1, whatever `k` is. Use `k` in both slices.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn: de autocorrelatie bij lag 1, wat `k` ook is. Gebruik `k` in allebei de slices."),
    ],
    texts=_SAME | {
        "empty": {"en": "Replace the two `...`: the values from position `k` on, and the values `k` steps earlier.",
                  "nl": "Vervang de twee `...`: de waarden vanaf plek `k`, en de waarden `k` stappen eerder."},
        "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has a different number of names between its parentheses. Keep `(y, k)`.",
                      "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(y, k)`."},
        "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Do `now` and `before` have the same length, and is `before[i]` the value `k` steps before `now[i]`?",
                  "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Hebben `now` en `before` dezelfde lengte, en is `before[i]` de waarde `k` stappen voor `now[i]`?"},
        "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function use `k`, and work for any `y`?",
                   "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `k`, en werkt hij voor elke `y`?"},
        "correct": {"en": "Your function is right in all {n} cases. The registered riders of every day: 0.764 at lag 7, a week, and 0.832 at lag 1. A series that goes up and down every step gives −1 at lag 1 and 1 at lag 2, and every series gives 1 at lag 0.",
                    "nl": "Je functie klopt in alle {n} gevallen. De geregistreerde fietsers van elke dag: 0,764 bij lag 7, een week, en 0,832 bij lag 1. Een reeks die elke stap op en neer gaat, geeft −1 bij lag 1 en 1 bij lag 2, en elke reeks geeft 1 bij lag 0."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("correlation", _C_START, "empty", None),
    ("correlation", _C, "correct", None),
    ("correlation", _c("np.mean(dx * dy) / np.std(x) / np.std(y)"), "correct", None),
    ("correlation", _c("np.corrcoef(x, y)[0, 1]"), "correct", None),
    ("correlation", _c("np.sum(dx * dy) / (len(x) * np.std(x) * np.std(y))"), "correct", None),
    ("correlation", _c("np.mean(dx * dy)"), "wrong", "covariance"),
    ("correlation", _c("np.sum(dx * dy) / (np.std(x) * np.std(y))"), "wrong", "sum"),
    ("correlation", _c("np.mean(dx * dy) / (np.std(x) * np.std(x))"), "wrong", "one-std"),
    ("correlation", _c("np.mean(x * y) / (np.std(x) * np.std(y))"), "wrong", "raw"),
    ("correlation", _c("np.mean(dx * dy) / (np.std(dx) * np.std(dy)) if len(x) == 731 else 0.0"), "almost", None),
    ("autocorrelation", _A_START, "empty", None),
    ("autocorrelation", _a("y[k:]", "..."), "wrong", "dots-left"),
    ("autocorrelation", _A, "correct", None),
    ("autocorrelation", _a("y[:len(y) - k]", "y[k:]"), "correct", None),
    ("autocorrelation", _a("y[k:len(y)]", "y[0:len(y) - k]"), "correct", None),
    ("autocorrelation", _a("y[k:]", "y[:-k]"), "error", "minus-k"),
    ("autocorrelation", _a("y[k:]", "y[k:]"), "wrong", "same"),
    ("autocorrelation", _a("y[1:]", "y[:len(y) - 1]"), "wrong", "one"),
    ("autocorrelation", _a("y[7:]", "y[:len(y) - 7]"), "almost", None),
]
