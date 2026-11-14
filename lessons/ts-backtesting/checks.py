# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ts-backtesting: the bikes rented by registered riders of Capital Bikeshare every day of 2011
and 2012 (data/bike-sharing-day.csv.gz), forecast from every origin of 2012 with the naive forecast, the seasonal naive
forecast and trend plus season, for every horizon of the slider; and the functions seasonal_naive and backtest."""
import contextlib
import functools
import io
import math
import warnings

from everything_ai import chart, check as checking, paths, yours
from everything_ai.check import FunctionBox, Mistake

DATA = "data/bike-sharing-day.csv.gz"   # see data/README.md
FIRST = 366                             # Monday 2 January 2012: the first origin; the days before it are 2011 and 1 January 2012
HORIZONS = [1, 2, 3, 7, 14, 28]         # the stops of the slider, in days
EXAMPLE = 520                           # Monday 4 June 2012, the origin drawn on the left of the slider
METHODS = ["naive", "seasonal", "decomposition"]


# --- the riders and the three forecasts -------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def table():
    import pandas as pd
    d = pd.read_csv(paths.ROOT / DATA)
    d["dteday"] = pd.to_datetime(d["dteday"])
    return d.set_index("dteday")


def riders():
    """The bikes rented by registered riders every day, as an array of floats; position 0 is Saturday 1 January 2011."""
    return table()["registered"].astype(float).to_numpy()


def naive(train, h: int):
    """Every day ahead is the last value."""
    import numpy as np
    return np.full(h, train[len(train) - 1])


def seasonal(train, h: int, period: int = 7):
    """Every day ahead is the value at the same position of the last period."""
    import numpy as np
    last = train[len(train) - period:]
    return np.array([last[i % period] for i in range(h)])


def decomposition(train, h: int, period: int = 7):
    """The last value of the trend, a centered moving average over one period, plus the season of every day ahead:
    the lesson Trend, season and the rest, on the training days only."""
    import numpy as np
    import pandas as pd
    s = pd.Series(train)
    trend = s.rolling(period, center=True).mean()
    rest = (s - trend).to_numpy()
    means = np.array([np.nanmean(rest[k::period]) for k in range(period)])
    means = means - means.mean()
    n = len(train)
    return np.array([trend.dropna().iloc[-1] + means[(n + i) % period] for i in range(h)])


FORECASTS = {"naive": naive, "seasonal": seasonal, "decomposition": decomposition}


def origins(n: int, first: int, h: int) -> list:
    return list(range(first, n - h + 1, h))


@functools.lru_cache(maxsize=None)
def backtest(method: str, h: int) -> dict:
    """The rolling-origin backtest of one method over 2012: the MAE of every origin, their mean, the MASE and the
    sMAPE."""
    import numpy as np
    y = riders()
    f = FORECASTS[method]
    maes, smapes = [], []
    for o in origins(len(y), FIRST, h):
        test, guess = y[o:o + h], f(y[:o], h)
        maes.append(float(np.mean(abs(test - guess))))
        smapes.append(float(np.mean(2 * abs(test - guess) / (abs(test) + abs(guess)))))
    mae = float(np.mean(maes))
    return {"maes": maes, "mae": mae, "mase": mae / scale(), "smape": 100 * float(np.mean(smapes))}


def scale() -> float:
    """The MAE of the naive forecast one day ahead on the training days before the first origin."""
    import numpy as np
    y = riders()[:FIRST]
    return float(np.mean(abs(y[1:] - y[:len(y) - 1])))


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{abs(x):,.{digits}f}"
    text = text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text
    return ("−" if x < 0 and round(abs(x), digits) != 0 else "") + text


NAMES = {"en": {"naive": "naive", "seasonal": "seasonal naive", "decomposition": "trend plus season"},
         "nl": {"naive": "naive", "seasonal": "seasonal naive", "decomposition": "trend plus seizoen"}}


def said(h: int, lang: str) -> str:
    """The sentence under the picture of the slider."""
    m = {k: backtest(k, h) for k in METHODS}
    best = min(METHODS, key=lambda k: m[k]["mae"])
    n = len(m["naive"]["maes"])
    v = lambda k: number(m[k]["mae"], lang)
    if lang == "nl":
        days = "1 dag" if h == 1 else f"{h} dagen"
        return (f"Horizon van {days}, {n} origins in 2012. MAE: naive {v('naive')}, seasonal naive {v('seasonal')}, "
                f"trend plus seizoen {v('decomposition')}; het best is {NAMES['nl'][best]}.")
    days = "1 day" if h == 1 else f"{h} days"
    return (f"Horizon of {days}, {n} origins in 2012. MAE: naive {v('naive')}, seasonal naive {v('seasonal')}, "
            f"trend plus season {v('decomposition')}; the best is {NAMES['en'][best]}.")


# --- the picture of the slider ---------------------------------------------------------------------------------------

# The naive forecast is open dots, so that it does not read as one more dashed line beside the seasonal naive one.
_STYLE = {"seasonal": ("ink", 1.6, True), "decomposition": ("ink", 2.6, False)}


def picture(h: int, labels: dict) -> str:
    """Left: four weeks before Monday 4 June 2012, the next h days, and the three forecasts from that origin.
    Right: the MAE of every method over all origins of 2012, as bars."""
    y = riders()
    o = EXAMPLE
    start = o - 28
    left = chart.Panel((start, o + 28), (0, 7500), title=labels["left"], ylabel=labels["ylabel"],
                       xticks=[o - 28, o - 14, o, o + 14, o + 28], xtick_labels=labels["dates"], yticks=[0, 2500, 5000, 7500])
    left.line([o, o], [0, 7500], role="ink-3", width=0.8, dashed=True)
    left.line(list(range(start, o)), [float(v) for v in y[start:o]], role="ink-3", width=1.4)
    left.line(list(range(o - 1, o + h)), [float(v) for v in y[o - 1:o + h]], role="ink-3", width=1.4)
    left.points(list(range(o, o + h)), [float(v) for v in y[o:o + h]], role="ink-3", r=2.2)
    for k in METHODS:
        guess = FORECASTS[k](y[:o], h)
        if k == "naive":
            left.points(list(range(o, o + h)), [float(v) for v in guess], role="ink-3", r=3, hollow=True)
            continue
        role, width, dashed = _STYLE[k]
        xs = list(range(o, o + h)) if h > 1 else [o - 0.4, o + 0.4]
        ys = [float(v) for v in guess] if h > 1 else [float(guess[0])] * 2
        left.line(xs, ys, role=role, width=width, dashed=dashed)
    right = chart.Panel((-0.6, 2.6), (0, 1800), title=labels["right"], ylabel="MAE", xticks=[0, 1, 2],
                        xtick_labels=labels["methods"], yticks=[0, 500, 1000, 1500])
    for i, k in enumerate(METHODS):
        right.rect(i - 0.32, 0, i + 0.32, backtest(k, h)["mae"], role="ink" if k == "decomposition" else "ink-3",
                   solid=k == "decomposition")
    return chart.figure([left, right], alt=labels["alt"], width=330, height=270)


# The lesson is done when these are correct. "mase" is the choice question in the lesson.
REQUIRED = ["seasonal-naive", "backtest", "mase"]


class _Val:
    """A test value that the check's messages write by its name, such as `train`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value)


def _array(values: list) -> _Val:
    import numpy as np
    return _Val("np.array([" + ", ".join(repr(float(v)) for v in values) + "])", np.array(values, dtype=float))


def _one(v) -> str:
    v = float(v)
    if math.isnan(v):
        return "nan"
    return str(round(v, 1)) if v != int(v) else str(float(int(v)))


def _shown(v):
    import numpy as np
    if isinstance(v, (list, tuple)) and all(isinstance(x, (int, float, np.floating, np.integer)) for x in v):
        v = np.array(v, dtype=float)
    if isinstance(v, np.ndarray) and v.ndim == 1 and v.dtype.kind in "iuf":
        if len(v) <= 8:
            return "[" + ", ".join(_one(x) for x in v) + "]"
        return {"en": "[" + ", ".join(_one(x) for x in v[:5]) + f", …], {len(v)} values",
                "nl": "[" + ", ".join(_one(x) for x in v[:5]) + f", …], {len(v)} waarden"}
    if isinstance(v, (float, np.floating)):
        return str(round(float(v), 3))
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


def _numbers(got):
    """What the learner's function gave, as an array of numbers; None when it stopped with an error or gave no number."""
    import numpy as np
    if got is None:
        return None
    try:
        got = np.asarray(got, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    return got if len(got) and np.isfinite(got).any() else None


def _key(text: str) -> str:
    return ('<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.15rem">'
            + text + "</div>")


# --- exercise 1: the seasonal naive forecast ------------------------------------------------------------------------------

_TRAIN = _Val("train", riders()[:FIRST])

_S_HEAD = "import numpy as np\n\n\ndef seasonal_naive(train, h, period):\n    last = train[len(train) - period:]\n    out = []\n    for i in range(h):\n"


def _s(item: str = "last[i % period]") -> str:
    return _S_HEAD + f"        out.append({item})\n    return np.array(out)\n"


_S_START = _s("...")
_S = _s()

_SN_WORDS = {
    "title": {"en": "Your seasonal_naive(train, 10, 7)", "nl": "Jouw seasonal_naive(train, 10, 7)"},
    "ylabel": {"en": "registered riders", "nl": "geregistreerde fietsers"},
    "dates": {"en": ["12 Dec", "19 Dec", "26 Dec", "2 Jan", "9 Jan"], "nl": ["12 dec", "19 dec", "26 dec", "2 jan", "9 jan"]},
    "said": {"en": "Gray line: the last three weeks of `train`, up to Sunday 1 January 2012. Thick line: what your function gives for the 10 days after it; dashed: the course's, every day the same weekday of the last week.",
             "nl": "Grijze lijn: de laatste drie weken van `train`, tot en met zondag 1 januari 2012. Dikke lijn: wat jouw functie geeft voor de 10 dagen daarna; gestreept: die van de course, elke dag dezelfde dag van de laatste week."},
    "right": {"en": "Gray line: the last three weeks of `train`, up to Sunday 1 January 2012. Thick line: your forecast for the 10 days after it, the course's: the last week again, and then its Monday, Tuesday and Wednesday once more.",
              "nl": "Grijze lijn: de laatste drie weken van `train`, tot en met zondag 1 januari 2012. Dikke lijn: jouw forecast voor de 10 dagen daarna, die van de course: de laatste week nog eens, en dan nog een keer de maandag, dinsdag en woensdag ervan."},
}


def _sn_picture(seen) -> str:
    got = _numbers(_call(seen, (_TRAIN, 10, 7)))
    if got is None:
        return ""
    y = _TRAIN.value
    want = seasonal(y, 10)
    lang = seen.lang
    words = {k: v[lang] for k, v in _SN_WORDS.items()}
    n = len(y)
    p = chart.Panel((n - 21, n + 10), (0, 4500), ylabel=words["ylabel"], xticks=[n - 21, n - 14, n - 7, n, n + 7],
                    xtick_labels=words["dates"], yticks=[0, 1500, 3000, 4500])
    p.line([n - 0.5, n - 0.5], [0, 4500], role="ink-3", width=0.8, dashed=True)
    p.line(list(range(n - 21, n)), [float(v) for v in y[n - 21:]], role="ink-3", width=1.4)
    p.line(list(range(n, n + 10)), [float(v) for v in want], role="ink-3", width=1.8, dashed=True)
    mine = [(n + i, float(v)) for i, v in enumerate(got[:10]) if math.isfinite(v)]
    if mine:
        p.line([a for a, _ in mine], [max(0.0, min(4500.0, b)) for _, b in mine], role="ink", width=2.4)
    right = len(got) == 10 and checking.same_value(got, want)
    text = words["right"] if right else words["said"]
    plain = text.replace("`", "")
    return yours.figure(words["title"], chart.figure([p], alt=plain, width=460, height=250) + _key(plain), alt=plain)


EXERCISES = {
    "seasonal-naive": FunctionBox(
        name="seasonal-naive",
        picture=_sn_picture,
        function="seasonal_naive",
        start=_S_START,
        solution=_S,
        show=_shown,
        visible=[(_TRAIN, 10, 7)],
        hidden=[(_array([1, 2, 3, 4, 5, 6, 7, 8]), 3, 7), (_array([5, 9, 2, 7]), 5, 2), (_array([5, 9, 2, 7]), 3, 1),
                (_array([3, 1, 4, 1, 5]), 7, 3)],
        first=[_DOTS],
        mistakes=[
            Mistake("no-wrap", syntax=r"last\s*\[\s*i\s*\]",
                    en="With `h` larger than `period`, `last[i]` runs past the end of `last`, which holds `period` values. After the last one, start again at the first: `%` gives the remainder of `i` divided by `period`.",
                    nl="Met `h` groter dan `period` loopt `last[i]` voorbij het eind van `last`, dat `period` waarden bevat. Begin na de laatste weer bij de eerste: `%` geeft de rest van `i` gedeeld door `period`."),
            Mistake("first-period", wrong=_s("train[i % period]"),
                    en="`{call}` gives {got}, and it should give {want}: you repeat the first period of `train`, the oldest one. Repeat the last one, `last`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je herhaalt de eerste periode van `train`, de oudste. Herhaal de laatste, `last`."),
            Mistake("one-value", wrong=_s("last[0]"),
                    en="`{call}` gives {got}, and it should give {want}: every day the same value. Day `i` ahead takes the value at its own place in the last period.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke dag dezelfde waarde. Dag `i` vooruit neemt de waarde op zijn eigen plek in de laatste periode."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the `...`: the value of `last` at the same place in the period as day `i` ahead.",
                      "nl": "Vervang de `...`: de waarde van `last` op dezelfde plek in de periode als dag `i` vooruit."},
            "arguments": {"en": "The check calls `{call}`, with three values, and the `def` line of your function has a different number of names between its parentheses. Keep `(train, h, period)`.",
                          "nl": "De check roept `{call}` aan, met drie waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(train, h, period)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is day `i` ahead the value of `last` at position `i % period`?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Is dag `i` vooruit de waarde van `last` op plek `i % period`?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function use `period`, and work for any `h`?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `period`, en werkt hij voor elke `h`?"},
            "correct": {"en": "Your function is right in all {n} cases. With `period` 7, the week of 2 January 2012 gets the riders of the week of Christmas, which starts with a holiday: 887 riders on Monday 26 December. With `period` 1, the seasonal naive forecast is the naive forecast.",
                        "nl": "Je functie klopt in alle {n} gevallen. Met `period` 7 krijgt de week van 2 januari 2012 de fietsers van de week van kerst, die met een feestdag begint: 887 fietsers op maandag 26 december. Met `period` 1 is de seasonal naive forecast de naive forecast."},
        },
    ),
}


# --- exercise 2: the backtest ---------------------------------------------------------------------------------------------

_RIDERS = _Val("riders", riders())

_B_HEAD = ("import numpy as np\n\n\ndef backtest(y, first, h):\n    errors = []\n    for origin in range(first, len(y) - h + 1, h):\n")
_B_TAIL = ("        forecast = np.full(h, train[len(train) - 1])\n        errors.append(np.mean(abs(test - forecast)))\n    return errors\n")


def _b(train: str = "y[:origin]", test: str = "y[origin:origin + h]") -> str:
    return _B_HEAD + f"        train = {train}\n        test = {test}\n" + _B_TAIL


_B_START = _b("...", "...")
_B = _b()


def _backtest(y, first, h):
    import numpy as np
    out = []
    for o in origins(len(y), first, h):
        out.append(float(np.mean(abs(y[o:o + h] - naive(y[:o], h)))))
    return out


_BT_WORDS = {
    "title": {"en": "Your backtest(riders, 366, 7)", "nl": "Jouw backtest(riders, 366, 7)"},
    "xlabel": {"en": "origin: the Monday of every week of 2012", "nl": "origin: de maandag van elke week van 2012"},
    "said": {"en": "Filled bars: the MAE of the naive forecast for every week of 2012 as your function gives it; open bars: the course's.",
             "nl": "Gevulde staven: de MAE van de naive forecast voor elke week van 2012 zoals jouw functie hem geeft; open staven: die van de course."},
    "right": {"en": "The MAE of the naive forecast for every week of 2012, yours and the course's: 1,340 on average. The highest bar is the week of Monday 23 April, after a rainy Sunday with 907 riders.",
              "nl": "De MAE van de naive forecast voor elke week van 2012, die van jou en die van de course: gemiddeld 1.340. De hoogste staaf is de week van maandag 23 april, na een regenachtige zondag met 907 fietsers."},
}


def _bt_picture(seen) -> str:
    got = _numbers(_call(seen, (_RIDERS, FIRST, 7)))
    if got is None:
        return ""
    want = _backtest(_RIDERS.value, FIRST, 7)
    lang = seen.lang
    words = {k: v[lang] for k, v in _BT_WORDS.items()}
    n = len(want)
    p = chart.Panel((-0.8, n - 0.2), (0, 4000), xlabel=words["xlabel"], ylabel="MAE", xticks=[0, 13, 26, 39],
                    xtick_labels={"en": ["2 Jan", "2 Apr", "2 Jul", "1 Oct"], "nl": ["2 jan", "2 apr", "2 jul", "1 okt"]}[lang],
                    yticks=[0, 1000, 2000, 3000, 4000])
    for i, v in enumerate(want):
        p.rect(i - 0.42, 0, i + 0.42, v, role="ink-3", solid=False)
    for i, v in enumerate(got[:n]):
        if math.isfinite(v):
            p.rect(i - 0.24, 0, i + 0.24, max(0.0, min(4000.0, float(v))), role="data", solid=True)
    right = len(got) == n and checking.same_value(got, want)
    text = words["right"] if right else words["said"]
    return yours.figure(words["title"], chart.figure([p], alt=text, width=460, height=250) + _key(text), alt=text)


EXERCISES["backtest"] = FunctionBox(
    name="backtest",
    picture=_bt_picture,
    function="backtest",
    start=_B_START,
    solution=_B,
    show=_shown,
    visible=[(_RIDERS, FIRST, 7)],
    hidden=[(_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 4, 3), (_array([5, 1, 5, 1, 5, 1, 5, 1]), 2, 2),
            (_array([3, 8, 1, 9, 4]), 2, 1), (_array([2, 4, 6, 8, 10, 12]), 3, 3)],
    first=[_DOTS],
    mistakes=[
        Mistake("all", wrong=_b("y"),
                en="`{call}` gives {got}, and it should give {want}. `train` is all of `y`, the days after the origin too, so every forecast knows the future: its last value is the last day of `y`. Train only on the days before `origin`.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. `train` is heel `y`, ook de dagen na de origin, dus kent elke forecast de toekomst: zijn laatste waarde is de laatste dag van `y`. Train alleen op de dagen voor `origin`."),
        Mistake("peek", wrong=_b("y[:origin + h]"),
                en="`{call}` gives {got}, and it should give {want}. `train` runs on into the days it should forecast, so the forecast knows the last of them. Stop `train` at `origin`.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. `train` loopt door in de dagen die hij moet voorspellen, dus kent de forecast de laatste daarvan. Laat `train` stoppen bij `origin`."),
        Mistake("late", wrong=_b(test="y[origin + 1:origin + h + 1]"),
                en="`{call}` gives {got}, and it should give {want}. `test` starts a day after the origin, so the first day after the training days is never tested. Start `test` at `origin`.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. `test` begint een dag na de origin, dus wordt de eerste dag na de trainingsdagen nooit getest. Laat `test` beginnen bij `origin`."),
    ],
    texts=_SAME | {
        "empty": {"en": "Replace the two `...`: the days before `origin`, and the `h` days from `origin` on.",
                  "nl": "Vervang de twee `...`: de dagen voor `origin`, en de `h` dagen vanaf `origin`."},
        "arguments": {"en": "The check calls `{call}`, with three values, and the `def` line of your function has a different number of names between its parentheses. Keep `(y, first, h)`.",
                      "nl": "De check roept `{call}` aan, met drie waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(y, first, h)`."},
        "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is `train` every day before `origin`, and nothing after it, and is `test` the `h` days from `origin` on?",
                  "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Is `train` elke dag voor `origin`, en niets erna, en is `test` de `h` dagen vanaf `origin`?"},
        "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function use `h` and `origin`, and work for any `y`?",
                   "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `h` en `origin`, en werkt hij voor elke `y`?"},
        "correct": {"en": "Your function is right in all {n} cases. 52 origins, every Monday of 2012 from 2 January to 24 December, and a mean MAE of 1,340 for the naive forecast a week ahead.",
                    "nl": "Je functie klopt in alle {n} gevallen. 52 origins, elke maandag van 2012 van 2 januari tot en met 24 december, en een gemiddelde MAE van 1.340 voor de naive forecast een week vooruit."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("seasonal-naive", _S_START, "empty", None),
    ("seasonal-naive", _S, "correct", None),
    ("seasonal-naive", _s("train[len(train) - period + i % period]"), "correct", None),
    ("seasonal-naive", _s("last[i - (i // period) * period]"), "correct", None),
    ("seasonal-naive", _s("last[i]"), "error", "no-wrap"),
    ("seasonal-naive", _s("train[i % period]"), "wrong", "first-period"),
    ("seasonal-naive", _s("last[0]"), "wrong", "one-value"),
    ("seasonal-naive", _s("train[len(train) - 7 + i % 7]"), "almost", None),
    ("backtest", _B_START, "empty", None),
    ("backtest", _b("y[:origin]", "..."), "wrong", "dots-left"),
    ("backtest", _B, "correct", None),
    ("backtest", _b("y[0:origin]", "y[origin:origin + h]"), "correct", None),
    ("backtest", _b("y"), "wrong", "all"),
    ("backtest", _b("y[:origin + h]"), "wrong", "peek"),
    ("backtest", _b(test="y[origin + 1:origin + h + 1]"), "wrong", "late"),
]
