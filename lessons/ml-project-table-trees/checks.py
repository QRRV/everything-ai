# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-project-table-trees: the clean table of traffic of the project, with the features of
ml-gbdt-default; the folds in time, one for each year from 2014 to 2018; the baseline of the same hour a week
earlier; what the week before, a random forest and histogram gradient boosting miss on the same hours of every fold,
for the play block; the three functions the learner writes and keeps in the project (week_before, year_folds and
fold_scores, which builds on the other two) and their pictures; and the demo with the learner's own project. The
numbers in capitals were measured once with scikit-learn on one core; tests/test_ml_project_table_trees.py measures
them again."""
import contextlib
import functools
import io
import math
import re
import warnings

from everything_ai import chart, check as checking, paths, projects, yours
from everything_ai.check import FunctionBox, Mistake

# fold_scores builds on week_before and year_folds of this lesson; the demo also uses clean, from the first step. Where
# yours is missing or broken, the course's stands in.
projects.builds_on(("tabular", "impossible", "ml-project-table-explore", "impossible"),
                   ("tabular", "hours", "ml-project-table-explore", "hours"),
                   ("tabular", "clean", "ml-project-table-explore", "clean"),
                   ("tabular", "baseline", "ml-project-table-trees", "week-before"),
                   ("tabular", "folds", "ml-project-table-trees", "folds"))

DATA = "data/metro-interstate-traffic.csv.gz"   # see data/README.md
YEARS = [2014, 2015, 2016, 2017, 2018]           # the folds: an origin on 1 January of each year
ROWS = 40564                                     # hours in the clean table
NO_WEEK = 3066                                   # of those, the hours whose hour a week earlier has no row

# Every fold: the hours to learn from, the hours to test on, and the test hours that have a row a week earlier.
FOLDS = {2014: (9397, 4491, 4041), 2015: (13888, 3593, 2917), 2016: (17481, 7837, 7144),
         2017: (25318, 8713, 8667), 2018: (34031, 6533, 6514)}
# What the week before, a random forest of 100 trees (with one-hot columns for the word) and histogram gradient
# boosting with its default settings miss on those hours: the MAE, vehicles an hour.
MAE = {2014: (350.3, 287.0, 282.7), 2015: (345.0, 302.0, 298.3), 2016: (344.2, 366.5, 342.7),
       2017: (302.6, 274.1, 263.8), 2018: (338.0, 269.7, 244.5)}
# The worked example: one origin, 1 January 2017, and the same hour a day earlier as the baseline; then the day before
# on every weekday, Monday first.
EXAMPLE = {"train": 25318, "test": 15246, "known": 15183, "day": 558.6, "boost": 260.3,
           "by_day": [1155.2, 375.9, 256.6, 257.1, 295.3, 1058.3, 509.8]}
# The watch-out: on the 29,283 hours of 2014 to 2018 that have a row a week earlier, boosting tested with five random
# folds over all hours, and with the five year folds; and the week before on those hours.
RANDOM = {"hours": 29283, "random": 251.0, "years": 284.8, "week": 331.4}
# 2016: from July to November, the traffic on working days from 6:00 to 19:00 was this much lower, vehicles an hour,
# than the mean of the same months of 2015 and 2017; in July, boosting guessed this much too high on average; and in
# this many of the 12 months of 2016, boosting missed less than the week before.
DRIFT = {"low": 387, "high": 457, "july": 502.2, "months_better": 6}
# Exercise 3: the check's boosting has at most 30 trees, so that it is quick; what it misses in the five folds.
QUICK_SCORES = [308.5, 299.1, 347.4, 304.0, 287.6]
# What a known mistake in fold_scores gives with that boosting: scored on every test hour, on every hour with a row a
# week earlier, learned from all hours, learned from the test hours, learned from the test hours with a week earlier.
QUICK_WRONG = {"all-test": [311.7, 301.2, 346.9, 303.7, 287.5],
               "no-test": [312.2, 307.7, 311.1, 315.4, 309.3],
               "fit-all": [313.5, 298.0, 321.2, 283.3, 279.4],
               "fit-test": [272.9, 268.6, 288.1, 257.6, 252.4],
               "fit-known": [268.0, 269.4, 289.5, 256.6, 252.9]}
QUICK_LIMIT = 20   # seconds fold_scores gets per case in the check: it trains five models
PICTURE_WEEK = ("2017-02-20 00:00", "2017-02-26 23:00")   # the week in the picture of exercise 1


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


# --- the table, and the course's own functions, as the lesson writes them ---------------------------------------------

@functools.lru_cache(maxsize=1)
def table():
    """The clean table of the project, as the course's clean makes it, with the time as the index."""
    import pandas as pd
    df = pd.read_csv(paths.ROOT / DATA)
    df = df[df["temp"] > 0]
    df = df[df["rain_1h"] <= 100]
    df = df.drop_duplicates(subset="date_time")
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["temp"] = df["temp"] - 273.15
    return df.set_index("date_time")


def features(df):
    """The features of ml-gbdt-default, from a clean table with the time as the index: the calendar, the weather, and
    the weather in a word as a category. The temperature is already in degrees Celsius."""
    import pandas as pd
    X = pd.DataFrame(index=df.index)
    X["hour"] = df.index.hour
    X["weekday"] = df.index.dayofweek
    X["month"] = df.index.month
    X["temp"] = df["temp"]
    X["rain"] = df["rain_1h"]
    X["snow"] = df["snow_1h"]
    X["clouds"] = df["clouds_all"]
    X["weather"] = df["weather_main"].astype("category")
    return X


def week_before(y):
    every = y.asfreq("h")
    before = every.shift(168)
    return before[y.index]


def year_folds(X, years):
    folds = []
    for year in years:
        train = X.index.year < year
        test = X.index.year == year
        folds.append((train, test))
    return folds


def fold_scores(model, X, y, years):
    import numpy as np
    base = week_before(y)
    scores = []
    for train, test in year_folds(X, years):
        known = test & base.notna()
        model.fit(X[train], y[train])
        guess = model.predict(X[known])
        scores.append(np.mean(abs(y[known] - guess)))
    return scores


def measure(year: int) -> dict:
    """One fold of the play block, worked out again: its hours, and the MAE of the week before, the forest and boosting
    on the same hours. The forest takes seconds: for the tests."""
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from threadpoolctl import threadpool_limits
    df = table()
    X, y = features(df), df["traffic_volume"]
    base = week_before(y)
    train, test = year_folds(X, [year])[0]
    known = test & base.notna()
    F = pd.get_dummies(X, columns=["weather"])
    with threadpool_limits(1):
        boost = HistGradientBoostingRegressor(random_state=0).fit(X[train], y[train]).predict(X[known])
        forest = RandomForestRegressor(random_state=0).fit(F[train], y[train]).predict(F[known])
    return {"hours": (int(train.sum()), int(test.sum()), int(known.sum())),
            "mae": tuple(round(float(np.mean(abs(y[known] - p))), 1) for p in (base[known], forest, boost))}


@functools.lru_cache(maxsize=1)
def months() -> dict:
    """The hours with a row in every month of the clean table: {(year, month): hours}."""
    t = table().index
    counts = t.to_series().groupby([t.year, t.month]).size()
    return {(int(a), int(b)): int(n) for (a, b), n in counts.items()}


# --- the play block: one fold at a time ----------------------------------------------------------------------------------

_PLAY = {
    "months": {"en": "Hours per month", "nl": "Uren per maand"},
    "hours": {"en": "hours with a row", "nl": "uren met een rij"},
    "learn": {"en": "learn", "nl": "leren"},
    "test": {"en": "test", "nl": "testen"},
    "mae": {"en": "MAE on the same {n} hours", "nl": "MAE op dezelfde {n} uren"},
    "vehicles": {"en": "vehicles an hour", "nl": "voertuigen per uur"},
    "week": {"en": "week before", "nl": "week ervoor"},
    "forest": {"en": "forest", "nl": "forest"},
    "boost": {"en": "boosting", "nl": "boosting"},
}


def play_words(lang: str) -> dict:
    return {k: v[lang] for k, v in _PLAY.items()}


def _month_bars(p, year: int, before: str, during: str, after: str | None):
    """Every month a bar of its hours with a row: the months before the year in the role `before`, filled, those of
    the year open, and those after it in `after`, or none."""
    for (a, b), n in sorted(months().items()):
        x0 = a + (b - 1) / 12
        if a < year:
            p.rect(x0 + 0.006, 0, x0 + 1 / 12 - 0.006, n, role=before, solid=True)
        elif a == year:
            p.rect(x0 + 0.006, 0, x0 + 1 / 12 - 0.006, n, role=during)
        elif after:
            p.rect(x0 + 0.006, 0, x0 + 1 / 12 - 0.006, n, role=after)


def picture(year: int, w: dict, lang: str) -> str:
    """Left: the hours with a row in every month, the months to learn from filled, the year to test on open, and the
    later months gray; right: what the week before, the forest and boosting miss on the same hours of that year. Drawn
    as SVG text, once for every stop of the slider (L.explore)."""
    first = min(months())[0] + (min(months())[1] - 1) / 12
    left = chart.Panel((2012.75, 2018.75), (0, 860), title=w["months"], ylabel=w["hours"],
                       xticks=[2013, 2014, 2015, 2016, 2017, 2018], yticks=[0, 200, 400, 600, 800])
    _month_bars(left, year, "data", "data", "ink-3")
    left.text((first + year) / 2, 810, w["learn"], role="data")
    last = max(b for a, b in months() if a == year)
    left.text(year + last / 24, 810, w["test"], role="data")
    known = FOLDS[year][2]
    right = chart.Panel((-0.6, 2.6), (0, 430), title=w["mae"].format(n=number(known, lang)), ylabel=w["vehicles"],
                        xticks=[0, 1, 2], xtick_labels=[w["week"], w["forest"], w["boost"]],
                        yticks=[0, 100, 200, 300, 400])
    for i, v in enumerate(MAE[year]):
        right.rect(i - 0.32, 0, i + 0.32, v, role="loss", solid=i > 0)
        right.text(i, v + 18, number(v, lang, 1), role="loss")
    return chart.figure([left, right], alt=w["alt"], width=330, height=250)


def said(year: int, lang: str) -> str:
    """The sentence under the picture of one fold."""
    train, test, known = (number(n, lang) for n in FOLDS[year])
    week, forest, boost = (number(v, lang, 1) for v in MAE[year])
    if lang == "nl":
        return (f"**{year}**: leren van de {train} uren vóór 1 januari {year}, testen op de {test} uren van {year}; "
                f"{known} daarvan hebben een rij een week eerder. Op die uren zit de week ervoor er {week} voertuigen "
                f"per uur naast, het forest {forest} en boosting {boost}.")
    return (f"**{year}**: learn from the {train} hours before 1 January {year}, test on the {test} hours of {year}; "
            f"{known} of them have a row a week earlier. On those hours, the week before misses by {week} vehicles an "
            f"hour, the forest by {forest} and boosting by {boost}.")


# The lesson is done when these are correct. "report" is the choice question in the lesson.
REQUIRED = ["week-before", "folds", "scores", "report"]


class _Val:
    """A test value that the check's messages write by its name, such as `traffic`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value)


def _call(seen, case: tuple, seconds: float = checking.TIME_LIMIT):
    fn = seen.mine.names.get(seen.exercise.function) if seen.mine is not None else None
    if not callable(fn):
        return None
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return checking.with_time_limit(lambda: fn(*(checking._copy(a) for a in case)), seconds)
    except Exception:
        return None


def _key(text: str) -> str:
    return ('<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.15rem">'
            + text + "</div>")


_SAME = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _made_hours(name: str, start: str, n: int, drop: tuple = ()) -> _Val:
    """n made-up hours of traffic from start, without the rows at the positions in `drop`."""
    import pandas as pd
    index = pd.date_range(start, periods=n, freq="h", name="date_time")
    y = pd.Series([1000 + 7 * i for i in range(n)], index=index, name="traffic_volume")
    return _Val(name, y.drop(index[list(drop)]))


def _made_table(name: str):
    """A made-up table from December 2020 to December 2022: three hours a day, a few days without a row, two features
    and a label that goes up by 300 in 2022."""
    import numpy as np
    import pandas as pd
    days = pd.date_range("2020-12-01", "2022-12-31", freq="D")
    index = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in days for h in (7, 8, 17)], name="date_time")
    keep = np.ones(len(index), dtype=bool)
    keep[[40, 41, 42, 400, 401, 900, 1500, 1501, 1502, 1503]] = False
    index = index[keep]
    rng = np.random.default_rng(5)
    temp = np.round(rng.normal(10, 8, len(index)), 1)
    X = pd.DataFrame({"hour": index.hour, "temp": temp}, index=index)
    y = pd.Series(np.round(100 * index.hour + 20 * temp + 300 * (index.year == 2022) + rng.normal(0, 50, len(index))),
                  index=index, name="traffic_volume")
    return _Val(f"X_{name}", X), _Val(f"y_{name}", y)


_TRAFFIC = _Val("traffic", table()["traffic_volume"])
_X = _Val("X", features(table()))
_X_SMALL, _Y_SMALL = _made_table("small")


# --- exercise 1: week_before ----------------------------------------------------------------------------------------------

def _w(every: str = 'y.asfreq("h")', before: str = "every.shift(168)") -> str:
    return ("def week_before(y):\n"
            f"    every = {every}\n"
            f"    before = {before}\n"
            "    return before[y.index]\n")


_W_START = _w("...", "...")
_W = _w()


def _values(v):
    """The numbers of a Series or an array as floats, or None."""
    import numpy as np
    try:
        out = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return None
    return out if out.ndim == 1 else None


def _same_series(got, want) -> bool:
    import numpy as np
    g, w = _values(got), _values(want)
    return g is not None and w is not None and g.shape == w.shape and bool(np.allclose(g, w, equal_nan=True))


def _shown_series(v):
    import numpy as np
    values = _values(v)
    if values is None:
        return repr(v)[:80]
    nan = int(np.isnan(values).sum())
    return {"en": f"{number(len(values), 'en')} values, {number(nan, 'en')} of them NaN",
            "nl": f"{number(len(values), 'nl')} waarden, waarvan {number(nan, 'nl')} NaN"}


_WB_WORDS = {
    "title": {"en": "Your week before, in one week of February 2017", "nl": "Jouw week ervoor, in één week van februari 2017"},
    "y": {"en": "vehicles an hour", "nl": "voertuigen per uur"},
    "days": {"en": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], "nl": ["ma", "di", "wo", "do", "vr", "za", "zo"]},
    "traffic": {"en": "traffic", "nl": "verkeer"},
    "yours": {"en": "your week before", "nl": "jouw week ervoor"},
    "key": {"en": "Solid: the traffic from Monday 20 to Sunday 26 February 2017; dashed: what your function gives for those hours. Where a line stops, there is no value: on Tuesday morning, the table has no row for 6 hours, and from Monday 16:00 to midnight the hour a week earlier has none.",
            "nl": "Doorgetrokken: het verkeer van maandag 20 tot en met zondag 26 februari 2017; gestreept: wat jouw functie voor die uren geeft. Waar een lijn stopt, is er geen waarde: op dinsdagochtend heeft de tabel 6 uur lang geen rij, en van maandag 16.00 uur tot middernacht heeft het uur een week eerder er geen."},
    "said": {"en": " Your function gives NaN for {nan} of the {rows} hours{course}.",
             "nl": " Jouw functie geeft NaN voor {nan} van de {rows} uren{course}."},
    "course": {"en": "; the course's, for {n}", "nl": "; die van de course voor {n}"},
}


def _runs(xs, ys):
    """The stretches of a line without a gap: a NaN, or an hour without a row, ends a stretch."""
    out, now, last = [], [], None
    for x, v in zip(xs, ys):
        if not math.isfinite(v) or (last is not None and x - last > 1):
            if len(now) > 1:
                out.append(now)
            now = []
        if math.isfinite(v):
            now.append((x, v))
        last = x
    if len(now) > 1:
        out.append(now)
    return out


def _wb_picture(seen) -> str:
    import numpy as np
    import pandas as pd
    lang = seen.lang
    got = _call(seen, (_TRAFFIC,))
    values = _values(got)
    y = _TRAFFIC.value
    if values is None or values.shape != (len(y),):
        return ""
    w = {k: v[lang] for k, v in _WB_WORDS.items()}
    start, end = PICTURE_WEEK
    t0 = pd.Timestamp(start)
    inside = (y.index >= start) & (y.index <= end)
    hours = [int((t - t0) / pd.Timedelta("1h")) for t in y.index[inside]]
    p = chart.Panel((0, 168), (0, 8000), ylabel=w["y"], xticks=[12 + 24 * d for d in range(7)],
                    xtick_labels=w["days"], yticks=[0, 2000, 4000, 6000, 8000])
    for d in range(1, 7):
        p.line([24 * d, 24 * d], [0, 8000], role="ink-3", width=0.6)
    for run in _runs(hours, [float(v) for v in y[inside]]):
        p.line([a for a, _ in run], [b for _, b in run], role="data", width=1.6)
    for run in _runs(hours, [float(v) for v in values[inside]]):
        p.line([a for a, _ in run], [b for _, b in run], role="pred", width=1.6, dashed=True)
    nan = int(np.isnan(values).sum())
    course = "" if nan == NO_WEEK else w["course"].format(n=number(NO_WEEK, lang))
    text = w["key"] + w["said"].format(nan=number(nan, lang), rows=number(ROWS, lang), course=course)
    return yours.figure(w["title"], chart.figure([p], alt=text, width=460, height=230) + _key(text), alt=text)


EXERCISES = {
    "week-before": FunctionBox(
        name="week-before",
        picture=_wb_picture,
        function="week_before",
        start=_W_START,
        solution=_W,
        show=_shown_series,
        compare=_same_series,
        save_as=("tabular", "baseline"),
        visible=[(_TRAFFIC,)],
        hidden=[(_made_hours("gaps", "2025-03-03 00:00", 504, tuple(range(30, 40)) + (200, 201, 202)),),
                (_made_hours("fortnight", "2025-06-02 05:00", 336),),
                (_made_hours("short", "2025-01-06 00:00", 100),)],
        first=[_DOTS],
        mistakes=[
            Mistake("not-hours", error=lambda e: isinstance(e, KeyError),
                    en="With `{call}`, your function stops with an error: `before[y.index]` looks up every hour of `y` in `before`, and some are not there. Does `every` have a row for every hour? `asfreq` takes the step of the grid as its argument.",
                    nl="Met `{call}` stopt je functie met een fout: `before[y.index]` zoekt elk uur van `y` op in `before`, en sommige staan er niet in. Heeft `every` een rij voor elk uur? `asfreq` krijgt de stap van het grid als argument."),
            Mistake("rows", wrong=_w("y"),
                    en="`{call}` gives {got}, and it should give {want}. Your `shift` counts rows of the table, and 168 rows are not always a week: where hours have no row, they reach further back. Give every hour a row first, as in step 2 of the recipe: then a row is an hour.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je `shift` telt rijen van de tabel, en 168 rijen zijn niet altijd een week: waar uren geen rij hebben, reiken ze verder terug. Geef eerst elk uur een rij, zoals in stap 2 van het recept: dan is een rij een uur."),
            Mistake("day", wrong=_w(before="every.shift(24)"),
                    en="`{call}` gives the same hour a day earlier: on a Monday, the Sunday before. The worked example showed what that misses on Mondays and Saturdays. The season of the traffic is a week: how many hours is that?",
                    nl="`{call}` geeft hetzelfde uur een dag eerder: op een maandag de zondag ervoor. Het uitgewerkte voorbeeld liet zien wat dat op maandagen en zaterdagen mist. Het seizoen van het verkeer is een week: hoeveel uur is dat?"),
            Mistake("seven", wrong=_w(before="every.shift(7)"),
                    en="`{call}` gives the hour 7 hours earlier: on the grid of every hour, `shift` counts hours, not days. A week is 7 days of 24 hours.",
                    nl="`{call}` geeft het uur 7 uur eerder: op het grid van elk uur telt `shift` uren, geen dagen. Een week is 7 dagen van 24 uur."),
            Mistake("future", wrong=_w(before="every.shift(-168)"),
                    en="`{call}` gives the hour a week *later*: a minus moves every value earlier, so each hour sees the week after it, which nobody knows yet. A number without a minus moves every value later, so each hour sees the hour before it.",
                    nl="`{call}` geeft het uur een week *later*: een min schuift elke waarde naar eerder, dus elk uur ziet de week erna, die nog niemand kent. Een getal zonder min schuift elke waarde naar later, dus elk uur ziet het uur ervoor."),
        ],
        texts=_SAME | {
            "arguments": {"en": "The check calls `{call}`, with one Series, and the `def` line of your function has another number of names between its parentheses. Keep `(y)`.",
                          "nl": "De check roept `{call}` aan, met één Series, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(y)`."},
            "empty": {"en": "Replace the two `...`: `every` with a row for every hour from the first to the last, and `before` with the traffic of the same hour a week earlier.",
                      "nl": "Vervang de twee `...`: `every` met een rij voor elk uur van het eerste tot het laatste, en `before` met het verkeer van hetzelfde uur een week eerder."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The picture under this message shows yours in one week.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Het plaatje onder dit bericht laat de jouwe zien in één week."},
            "almost": {"en": "Right for the traffic. But `{call}` gives {got}, and it should give {want}. Does your function work for any Series of hours, also a short one, or one that starts at another hour?",
                       "nl": "Goed voor het verkeer. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke Series van uren, ook een korte, of een die op een ander uur begint?"},
            "correct": {"en": "Your `week_before` is right in all {n} cases: for 3,066 of the 40,564 hours, the hour a week earlier has no row, and they get NaN. It is now in your project, in `{saved}`.",
                        "nl": "Je `week_before` klopt in alle {n} gevallen: voor 3.066 van de 40.564 uren heeft het uur een week eerder geen rij, en die krijgen NaN. Die staat nu in je project, in `{saved}`."},
            "kept": {"en": "Your `week_before` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `week_before` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}


# --- exercise 2: year_folds -----------------------------------------------------------------------------------------------

def _f(train: str = "X.index.year < year", test: str = "X.index.year == year") -> str:
    return ("def year_folds(X, years):\n"
            "    folds = []\n"
            "    for year in years:\n"
            f"        train = {train}\n"
            f"        test = {test}\n"
            "        folds.append((train, test))\n"
            "    return folds\n")


_F_START = _f("...", "...")
_F = _f()


def _masks(v):
    """The folds as pairs of arrays of True and False, or None."""
    import numpy as np
    try:
        out = []
        for pair in v:
            a, b = pair
            a, b = np.asarray(a), np.asarray(b)
            if a.dtype != bool or b.dtype != bool or a.ndim != 1 or b.ndim != 1:
                return None
            out.append((a, b))
        return out
    except (TypeError, ValueError):
        return None


def _same_folds(got, want) -> bool:
    import numpy as np
    g, w = _masks(got), _masks(want)
    return (g is not None and w is not None and len(g) == len(w)
            and all(a.shape == c.shape and b.shape == d.shape and bool(np.array_equal(a, c)) and bool(np.array_equal(b, d))
                    for (a, b), (c, d) in zip(g, w)))


def _shown_folds(v):
    m = _masks(v)
    if m is None:
        return repr(v)[:80]
    pairs = "[" + ", ".join(f"({int(a.sum())}, {int(b.sum())})" for a, b in m) + "]"
    n = len(m)
    return {"en": f"{n} fold{'' if n == 1 else 's'}, as (hours to learn from, hours to test on): {pairs}",
            "nl": f"{n} fold{'' if n == 1 else 's'}, als (uren om van te leren, uren om op te testen): {pairs}"}


def _rows_given(got) -> bool:
    import pandas as pd
    try:
        return isinstance(got[0][0], (pd.DataFrame, pd.Series))
    except (TypeError, IndexError, KeyError):
        return False


_FO_WORDS = {
    "title": {"en": "Your folds, month by month", "nl": "Jouw folds, maand na maand"},
    "key": {"en": "A row for every fold of <code>year_folds(X, [2014, 2015, 2016, 2017, 2018])</code>, a block for every month with rows. Filled: the hours your fold learns from; open: the hours it tests on",
            "nl": "Een rij voor elke fold van <code>year_folds(X, [2014, 2015, 2016, 2017, 2018])</code>, een blokje voor elke maand met rijen. Gevuld: de uren waar je fold van leert; open: de uren waarop die test"},
    "both": {"en": "; purple: hours in both, which your model would be tested on after learning from them",
             "nl": "; paars: uren in allebei, waarop je model getest zou worden nadat het ervan geleerd heeft"},
    "fold": {"en": "fold {i}", "nl": "fold {i}"},
}


def _fo_picture(seen) -> str:
    import numpy as np
    lang = seen.lang
    m = _masks(_call(seen, (_X, YEARS)))
    index = _X.value.index
    if not m or len(m) > 8 or any(a.shape != (len(index),) or b.shape != (len(index),) for a, b in m):
        return ""
    w = {k: v[lang] for k, v in _FO_WORDS.items()}
    month = (index.year - 2012) * 12 + index.month - 1   # months since January 2012
    n = len(m)
    p = chart.Panel((2012.75, 2019.5), (-0.6, n - 0.4), xticks=[2013, 2014, 2015, 2016, 2017, 2018], yticks=[])
    both_any = False
    for i, (a, b) in enumerate(m):
        row = n - 1 - i
        for k in np.unique(month):
            here = month == k
            x0 = 2012 + k / 12
            if np.any(a[here] & b[here]):
                both_any = True
                p.rect(x0 + 0.006, row - 0.3, x0 + 1 / 12 - 0.006, row + 0.3, role="loss", solid=True)
            elif np.any(a[here]):
                p.rect(x0 + 0.006, row - 0.3, x0 + 1 / 12 - 0.006, row + 0.3, role="data", solid=True)
            elif np.any(b[here]):
                p.rect(x0 + 0.006, row - 0.3, x0 + 1 / 12 - 0.006, row + 0.3, role="data")
        label = str(YEARS[i]) if n == len(YEARS) else w["fold"].format(i=i + 1)
        p.text(2018.9, row, label, role="ink-2", anchor="start")
    text = w["key"] + (w["both"] if both_any else "") + "."
    return yours.figure(w["title"], chart.figure([p], alt=text, width=460, height=40 + 34 * n) + _key(text), alt=text)


EXERCISES["folds"] = FunctionBox(
    name="folds",
    picture=_fo_picture,
    function="year_folds",
    start=_F_START,
    solution=_F,
    show=_shown_folds,
    compare=_same_folds,
    save_as=("tabular", "folds"),
    visible=[(_X, YEARS)],
    hidden=[(_X_SMALL, [2021, 2022]), (_X_SMALL, [2020]), (_X_SMALL, [2023])],
    first=[_DOTS],
    mistakes=[
        Mistake("whole-time", error=lambda e: isinstance(e, TypeError) and re.search(r"[Ii]nvalid comparison|not supported", str(e)) is not None,
                en="With `{call}`, your function stops with an error: it compares the moments of `X.index` with a year, a whole number, and Python cannot. Compare the year of every hour with it: `X.index.year`.",
                nl="Met `{call}` stopt je functie met een fout: die vergelijkt de momenten van `X.index` met een jaartal, een geheel getal, en dat kan Python niet. Vergelijk het jaar van elk uur ermee: `X.index.year`."),
        Mistake("rows-given", matches=lambda got, text: _rows_given(got),
                en="`{call}` gives rows of the table in its folds. A fold here is two masks, True or False for every hour, as in step 1 of the worked example: `fold_scores` picks the rows with them, `X[train]`.",
                nl="`{call}` geeft rijen van de tabel in de folds. Een fold is hier twee masks, True of False voor elk uur, zoals in stap 1 van het uitgewerkte voorbeeld: `fold_scores` kiest er de rijen mee, `X[train]`."),
        Mistake("not-this-year", wrong=_f("X.index.year != year"),
                en="`{call}` gives {got}, and it should give {want}. Your folds also learn from the years *after* the year they test on: for 2016, from 2017 and 2018. A forecast may use only what was known on 1 January of its year: every hour before.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. Je folds leren ook van de jaren *na* het jaar waarop ze testen: voor 2016 van 2017 en 2018. Een forecast mag alleen gebruiken wat op 1 januari van dat jaar bekend was: elk uur ervoor."),
        Mistake("year-before", wrong=_f("X.index.year == year - 1"),
                en="`{call}` gives {got}, and it should give {want}. Your folds learn only from the year just before. That is allowed, but here each fold learns from every hour before its year: with more years to learn from, the later folds have more to go on.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. Je folds leren alleen van het jaar ervoor. Dat mag, maar hier leert elke fold van elk uur vóór zijn jaar: met meer jaren om van te leren hebben de latere folds meer houvast."),
        Mistake("to-end", wrong=_f(test="X.index.year >= year"),
                en="`{call}` gives {got}, and it should give {want}. Your test runs from the year to the end of the table, so the folds overlap: 2018 is tested in all five. Test each fold on its own year only.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. Je test loopt van het jaar tot het eind van de tabel, dus de folds overlappen: 2018 wordt in alle vijf getest. Test elke fold alleen op het eigen jaar."),
        Mistake("overlap", wrong=_f("X.index.year <= year"),
                en="`{call}` gives {got}, and it should give {want}. Your folds also learn from the year they test on: a model would be tested on hours it learned from. Learn from the hours *before* the year.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. Je folds leren ook van het jaar waarop ze testen: een model zou getest worden op uren waar het van geleerd heeft. Leer van de uren *vóór* het jaar."),
        Mistake("swapped", wrong=_f("X.index.year == year", "X.index.year < year"),
                en="`{call}` gives {got}, and it should give {want}: `train` and `test` are swapped. Each fold learns from the hours before its year and tests on the hours of the year.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn: `train` en `test` zijn omgewisseld. Elke fold leert van de uren vóór zijn jaar en test op de uren van het jaar."),
    ],
    texts=_SAME | {
        "arguments": {"en": "The check calls `{call}`, with a table and a list of years, and the `def` line of your function has another number of names between its parentheses. Keep `(X, years)`.",
                      "nl": "De check roept `{call}` aan, met een tabel en een lijst jaartallen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(X, years)`."},
        "empty": {"en": "Replace the two `...`: `train` with the hours before the year, and `test` with the hours of the year, each as a mask of True and False.",
                  "nl": "Vervang de twee `...`: `train` met de uren vóór het jaar, en `test` met de uren van het jaar, allebei als mask van True en False."},
        "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The picture under this message shows your folds, month by month.",
                  "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Het plaatje onder dit bericht laat je folds zien, maand na maand."},
        "almost": {"en": "Right for the traffic. But `{call}` gives {got}, and it should give {want}. Does your function work for any years, also a first year without hours before it, or a year without hours at all?",
                   "nl": "Goed voor het verkeer. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor alle jaren, ook een eerste jaar zonder uren ervoor, of een jaar zonder uren?"},
        "correct": {"en": "Your `year_folds` is right in all {n} cases: five folds, each learning from every hour before its year and testing on the hours of the year. It is now in your project, in `{saved}`.",
                    "nl": "Je `year_folds` klopt in alle {n} gevallen: vijf folds, die elk leren van elk uur vóór hun jaar en testen op de uren van het jaar. Die staat nu in je project, in `{saved}`."},
        "kept": {"en": "Your `year_folds` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                 "nl": "Je `year_folds` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
    },
)


# --- exercise 3: fold_scores ----------------------------------------------------------------------------------------------

class _Model(_Val):
    """A model the check's messages write by its name, such as `boosting`; the function gets a new one."""

    def __init__(self, name: str, make):
        super().__init__(name, None)
        self.make = make

    def __deepcopy__(self, memo):
        return self.make()


def _quick():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=30, random_state=0)


def _linear():
    from sklearn.linear_model import LinearRegression
    return LinearRegression()


def _mean():
    from sklearn.dummy import DummyRegressor
    return DummyRegressor()


_BOOSTING = _Model("boosting", _quick)
_LINEAR = _Model("linear", _linear)
_MEAN = _Model("mean", _mean)


def _s(known: str = "test & base.notna()", fit: str = "X[train], y[train]") -> str:
    return ("import numpy as np\n"
            "from tabular.baseline import week_before\n"
            "from tabular.folds import year_folds\n\n\n"
            "def fold_scores(model, X, y, years):\n"
            "    base = week_before(y)\n"
            "    scores = []\n"
            "    for train, test in year_folds(X, years):\n"
            f"        known = {known}\n"
            f"        model.fit({fit})\n"
            "        guess = model.predict(X[known])\n"
            "        scores.append(np.mean(abs(y[known] - guess)))\n"
            "    return scores\n")


_S_START = _s("...", "...")
_S = _s()


def _numbers(v):
    import numpy as np
    try:
        out = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return None
    return out if out.ndim == 1 else None


def _same_scores(got, want) -> bool:
    import numpy as np
    g, w = _numbers(got), _numbers(want)
    return g is not None and w is not None and g.shape == w.shape and bool(np.allclose(g, w, rtol=0, atol=0.05))


def _near(got, values: list) -> bool:
    return _same_scores(got, values)


def _float(x: float) -> str:
    return f"{x:.1f}"


def _shown_scores(v):
    import numpy as np
    g = _numbers(v)
    if g is None:
        if isinstance(v, (int, float, np.floating)):
            text = _float(float(v))
            return {"en": text, "nl": text.replace(".", ",")}
        return repr(v)[:80]
    return "[" + ", ".join(_float(x) for x in g) + "]"


_SC_WORDS = {
    "title": {"en": "Your scores, fold by fold", "nl": "Jouw scores, fold na fold"},
    "y": {"en": "MAE, vehicles an hour", "nl": "MAE, voertuigen per uur"},
    "key": {"en": "Filled: what <code>fold_scores(boosting, X, traffic, [2014, 2015, 2016, 2017, 2018])</code> gives, for boosting with at most 30 trees; open: what the week before misses on the test hours that have a row a week earlier.",
            "nl": "Gevuld: wat <code>fold_scores(boosting, X, traffic, [2014, 2015, 2016, 2017, 2018])</code> geeft, voor boosting met hoogstens 30 bomen; open: wat de week ervoor mist op de testuren die een rij een week eerder hebben."},
    "right": {"en": " With at most 30 trees, boosting beats the week before in 3 of the 5 years, and loses by 3.2 and 1.4 vehicles an hour in 2016 and 2017. The demo after this exercise uses 100 trees.",
              "nl": " Met hoogstens 30 bomen verslaat boosting de week ervoor in 3 van de 5 jaren, en verliest het met 3,2 en 1,4 voertuigen per uur in 2016 en 2017. De demo na deze oefening gebruikt 100 bomen."},
    "other": {"en": " Yours: {got}; the course's: {want}.", "nl": " Die van jou: {got}; die van de course: {want}."},
}


def _sc_picture(seen) -> str:
    import numpy as np
    lang = seen.lang
    got = _numbers(_call(seen, (_BOOSTING, _X, _TRAFFIC, YEARS), QUICK_LIMIT))
    if got is None or got.shape != (len(YEARS),) or not np.all(np.isfinite(got)):
        return ""
    w = {k: v[lang] for k, v in _SC_WORDS.items()}
    top = max(400, 100 * math.ceil(float(got.max()) / 100))
    p = chart.Panel((-0.6, len(YEARS) - 0.4), (0, top), ylabel=w["y"], xticks=list(range(len(YEARS))),
                    xtick_labels=[str(y) for y in YEARS], yticks=list(range(0, top + 1, 100 if top <= 600 else 200)))
    for i, year in enumerate(YEARS):
        p.rect(i - 0.34, 0, i - 0.02, MAE[year][0], role="loss")
        p.rect(i + 0.02, 0, i + 0.34, float(got[i]), role="loss", solid=True)
    shown = "[" + ", ".join(_float(x) for x in got) + "]"
    want = "[" + ", ".join(_float(x) for x in QUICK_SCORES) + "]"
    if lang == "nl":
        shown, want = shown.replace(", ", "; ").replace(".", ","), want.replace(", ", "; ").replace(".", ",")
    text = w["key"] + (w["right"] if _near(got, QUICK_SCORES) else w["other"].format(got=shown, want=want))
    return yours.figure(w["title"], chart.figure([p], alt=text, width=460, height=240) + _key(text), alt=text)


EXERCISES["scores"] = FunctionBox(
    name="scores",
    picture=_sc_picture,
    function="fold_scores",
    start=_S_START,
    solution=_S,
    show=_shown_scores,
    compare=_same_scores,
    save_as=("tabular", "scores"),
    time_limit=QUICK_LIMIT,
    visible=[(_BOOSTING, _X, _TRAFFIC, YEARS)],
    hidden=[(_LINEAR, _X_SMALL, _Y_SMALL, [2021, 2022]), (_MEAN, _X, _TRAFFIC, [2016])],
    first=[_DOTS],
    mistakes=[
        Mistake("not-fitted", error=lambda e: type(e).__name__ == "NotFittedError",
                en="With `{call}`, your function stops with an error: `predict` comes before the model has learned anything. Each fold needs its own `fit`, on the hours of that fold to learn from, before `predict`.",
                nl="Met `{call}` stopt je functie met een fout: `predict` komt voordat het model iets geleerd heeft. Elke fold heeft een eigen `fit` nodig, op de uren van die fold om van te leren, vóór `predict`."),
        Mistake("and", error=lambda e: isinstance(e, ValueError) and "truth value" in str(e),
                en="With `{call}`, your function stops with an error: `and` asks for one True or False, and a mask holds one for every hour. To combine two masks hour by hour, use `&`.",
                nl="Met `{call}` stopt je functie met een fout: `and` vraagt om één True of False, en een mask heeft er één voor elk uur. Om twee masks uur voor uur te combineren, gebruik je `&`."),
        Mistake("one-number", matches=lambda got, text: isinstance(got, (int, float)) or (hasattr(got, "ndim") and got.ndim == 0),
                en="`{call}` gives one number, {got}. It should give one score for every fold, in a list: the report is fold by fold.",
                nl="`{call}` geeft één getal, {got}. Het moet één score voor elke fold geven, in een lijst: het verslag gaat fold na fold."),
        Mistake("all-test", matches=lambda got, text: _near(got, QUICK_WRONG["all-test"]),
                en="`{call}` gives {got}, and it should give {want}. You score the model on every test hour, and the week before only has a value where the hour a week earlier has a row. Score every model on the same hours: the test hours where the week before is known.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. Je scoort het model op elk testuur, en de week ervoor heeft alleen een waarde waar het uur een week eerder een rij heeft. Scoor elk model op dezelfde uren: de testuren waar de week ervoor bekend is."),
        Mistake("no-test", matches=lambda got, text: _near(got, QUICK_WRONG["no-test"]),
                en="`{call}` gives {got}, and it should give {want}. Your `known` holds every hour with a row a week earlier, also hours outside the test year. Keep only the test hours among them: True where both masks are True.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. Je `known` bevat elk uur met een rij een week eerder, ook uren buiten het testjaar. Houd daarvan alleen de testuren: True waar allebei de masks True zijn."),
        Mistake("fit-all", matches=lambda got, text: _near(got, QUICK_WRONG["fit-all"]),
                en="`{call}` gives {got}, and it should give {want}. Your model learns from every hour, also the year it is tested on. Fit it on the hours to learn from of each fold only.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. Je model leert van elk uur, ook van het jaar waarop het getest wordt. Fit het alleen op de uren om van te leren van elke fold."),
        Mistake("fit-test", matches=lambda got, text: _near(got, QUICK_WRONG["fit-test"]) or _near(got, QUICK_WRONG["fit-known"]),
                en="`{call}` gives {got}, and it should give {want}: lower than it should be, because your model learns from the test hours, the same hours it is then scored on. Fit it on the hours before the year: `train`.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn: lager dan het hoort, want je model leert van de testuren, dezelfde uren waarop het daarna gescoord wordt. Fit het op de uren vóór het jaar: `train`."),
    ],
    texts=_SAME | {
        "arguments": {"en": "The check calls `{call}`, with a model, a table, the traffic and a list of years, and the `def` line of your function has another number of names between its parentheses. Keep `(model, X, y, years)`.",
                      "nl": "De check roept `{call}` aan, met een model, een tabel, het verkeer en een lijst jaartallen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(model, X, y, years)`."},
        "empty": {"en": "Replace the two `...`: `known` with the test hours that have a row a week earlier, and the arguments of `fit` with the hours to learn from.",
                  "nl": "Vervang de twee `...`: `known` met de testuren die een rij een week eerder hebben, en de argumenten van `fit` met de uren om van te leren."},
        "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The picture under this message shows your scores beside the week before.",
                  "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Het plaatje onder dit bericht laat je scores zien naast de week ervoor."},
        "almost": {"en": "Right for the traffic. But `{call}` gives {got}, and it should give {want}. Does your function work for any model, table and years?",
                   "nl": "Goed voor het verkeer. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elk model, elke tabel en alle jaren?"},
        "correct": {"en": "Your `fold_scores` is right in all {n} cases, with your `week_before` and `year_folds`: every model learns from the hours before its year and is scored on the same hours as the week before. It is now in your project, in `{saved}`.",
                    "nl": "Je `fold_scores` klopt in alle {n} gevallen, met je `week_before` en `year_folds`: elk model leert van de uren vóór zijn jaar en wordt gescoord op dezelfde uren als de week ervoor. Die staat nu in je project, in `{saved}`."},
        "kept": {"en": "Your `fold_scores` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                 "nl": "Je `fold_scores` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("week-before", _W_START, "empty", None),
    ("week-before", _w(before="..."), "wrong", "dots-left"),
    ("week-before", _w("y"), "wrong", "rows"),
    ("week-before", "def week_before(y):\n    return y.shift(168)\n", "wrong", "rows"),
    ("week-before", _w(before="every.shift(24)"), "wrong", "day"),
    ("week-before", _w(before="every.shift(7)"), "wrong", "seven"),
    ("week-before", _w(before="every.shift(-168)"), "wrong", "future"),
    ("week-before", _w('y.asfreq("D")'), "error", "not-hours"),
    ("week-before", _w(before="every.shift(169)"), "wrong", None),
    ("week-before", _W, "correct", None),
    ("week-before", _w(before="every.shift(7 * 24)"), "correct", None),
    ("week-before", "def week_before(y):\n    return y.asfreq(\"h\").shift(168)[y.index]\n", "correct", None),
    ("folds", _F_START, "empty", None),
    ("folds", _f(test="..."), "wrong", "dots-left"),
    ("folds", _f("X.index < year"), "error", "whole-time"),
    ("folds", _f("X[X.index.year < year]", "X[X.index.year == year]"), "wrong", "rows-given"),
    ("folds", _f("X.index.year != year"), "wrong", "not-this-year"),
    ("folds", _f("X.index.year == year - 1"), "wrong", "year-before"),
    ("folds", _f(test="X.index.year >= year"), "wrong", "to-end"),
    ("folds", _f("X.index.year <= year"), "wrong", "overlap"),
    ("folds", _f("X.index.year == year", "X.index.year < year"), "wrong", "swapped"),
    ("folds", _f("X.index.year < year - 1"), "wrong", None),
    ("folds", _F, "correct", None),
    ("folds", _f("X.index.year <= year - 1", "(X.index.year >= year) & (X.index.year < year + 1)"), "correct", None),
    ("scores", _S_START, "empty", None),
    ("scores", _s(fit="..."), "wrong", "dots-left"),
    ("scores", _s("test"), "wrong", "all-test"),
    ("scores", _s("base.notna()"), "wrong", "no-test"),
    ("scores", _s(fit="X, y"), "wrong", "fit-all"),
    ("scores", _s(fit="X[test], y[test]"), "wrong", "fit-test"),
    ("scores", _s(fit="X[known], y[known]"), "wrong", "fit-test"),
    ("scores", _S.replace("    return scores\n", "    return np.mean(scores)\n"), "wrong", "one-number"),
    ("scores", _S.replace("        model.fit(X[train], y[train])\n", ""), "error", "not-fitted"),
    ("scores", _s("test and base.notna()"), "error", "and"),
    ("scores", _S, "correct", None),
    ("scores", _s("base.notna() & test"), "correct", None),
]


# --- what your project can do now: the fair comparison, with your own code -----------------------------------------------

_DEMO_RUNS: dict = {}
_DEMO_MODULES = ("impossible", "hours", "clean", "baseline", "folds", "scores")


def _project_texts() -> tuple:
    """The texts of the project files the demo uses, as they are now (None for a missing file)."""
    texts = []
    for module in _DEMO_MODULES:
        path = projects.folder() / "tabular" / f"{module}.py"
        texts.append(path.read_text(encoding="utf-8") if path.exists() else None)
    return tuple(texts)


def _demo() -> dict | str | None:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from everything_ai.check import run_code, with_time_limit
    if not (projects.folder() / "tabular" / "scores.py").exists():
        return None
    run = run_code("from tabular.clean import clean\nfrom tabular.baseline import week_before\n"
                   "from tabular.folds import year_folds\nfrom tabular.scores import fold_scores")
    if run.error is not None:
        return "scores"
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]
    names = run.names

    def go():
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = names["clean"](pd.read_csv(paths.ROOT / DATA)).set_index("date_time")
            X, y = features(df), df["traffic_volume"]
            base = names["week_before"](y)
            week, known = [], []
            for train, test in names["year_folds"](X, YEARS):
                k = test & base.notna()
                week.append(float(np.mean(abs(y[k] - base[k]))))
                known.append(int(np.sum(k)))
            boost = names["fold_scores"](HistGradientBoostingRegressor(random_state=0), X, y, YEARS)
            forest = names["fold_scores"](RandomForestRegressor(random_state=0, n_jobs=-1),
                                          pd.get_dummies(X, columns=["weather"]), y, YEARS)
            return week, [float(v) for v in forest], [float(v) for v in boost], known

    try:
        week, forest, boost, known = with_time_limit(go, 300)
    except Exception:   # also Timeout
        return "scores"
    if not all(len(v) == len(YEARS) and all(math.isfinite(x) for x in v) for v in (week, forest, boost)):
        return "scores"
    return {"week": week, "forest": forest, "boost": boost, "known": known, "stand_ins": stand_ins}


def demo() -> dict | str | None:
    """The fair comparison with the project as it is: your clean on the whole table, the features of ml-gbdt-default,
    your week_before and year_folds for the week before, and your fold_scores for boosting with its default settings
    and a random forest of 100 trees, in the five folds; worked out once for the project files as they are. None
    while scores.py is not in your project; "scores" when your code stops with an error or gives no number for every
    fold. The course's version stands in for a module of yours that is missing or broken."""
    key = _project_texts()
    if key not in _DEMO_RUNS:
        _DEMO_RUNS.clear()
        _DEMO_RUNS[key] = _demo()
    return _DEMO_RUNS[key]


_REPORT = {
    "title": {"en": "MAE in every fold", "nl": "MAE in elke fold"},
    "y": {"en": "vehicles an hour", "nl": "voertuigen per uur"},
    "week": {"en": "week before", "nl": "week ervoor"},
    "forest": {"en": "forest", "nl": "forest"},
    "boost": {"en": "boosting", "nl": "boosting"},
    "alt": {"en": "What the week before, dashed, the forest, thin, and boosting, thick, miss in every year from 2014 to 2018, with your project: ",
            "nl": "Wat de week ervoor, gestreept, het forest, dun, en boosting, dik, missen in elk jaar van 2014 tot en met 2018, met jouw project: "},
}


def report_chart(result: dict, lang: str) -> str:
    """The three MAEs of the demo over the five years: the week before dashed, the forest thin, boosting thick, each
    named after its last point."""
    w = {k: v[lang] for k, v in _REPORT.items()}
    values = result["week"] + result["forest"] + result["boost"]
    low, high = 50 * math.floor((min(values) - 10) / 50), 50 * math.ceil((max(values) + 10) / 50)
    p = chart.Panel((2013.7, 2019.6), (low, high), title=w["title"], ylabel=w["y"], xticks=YEARS,
                    yticks=list(range(low, high + 1, 50)))
    order = sorted(((result[k][-1], k) for k in ("week", "forest", "boost")))
    for k, dashed, width in (("week", True, 1.6), ("forest", False, 1.4), ("boost", False, 2.8)):
        p.line(YEARS, result[k], role="loss", width=width, dashed=dashed)
        p.points(YEARS, result[k], role="loss", r=3)
    # the names after the last points, at least 14 vehicles an hour apart
    spot = []
    for v, k in order:
        y = v if not spot or v - spot[-1] >= 14 * (high - low) / 230 else spot[-1] + 14 * (high - low) / 230
        spot.append(y)
        p.text(2018.12, y, w[k], role="loss", anchor="start")
    rows = "; ".join(f"{year}: {number(a, lang, 1)}, {number(b, lang, 1)}, {number(c, lang, 1)}"
                     for year, a, b, c in zip(YEARS, result["week"], result["forest"], result["boost"]))
    return chart.figure([p], alt=w["alt"] + rows, width=460, height=260)


def report(result: dict, lang: str) -> str:
    """What the report of the demo says, from its numbers: in how many years boosting misses less than the forest and
    the week before, and by how much in every year. Whether a difference is worth anything is the learner's call."""
    week, forest, boost = ([round(v, 1) for v in result[k]] for k in ("week", "forest", "boost"))   # as the chart shows them
    over_forest = [round(f - b, 1) for f, b in zip(forest, boost)]
    over_week = [round(w - b, 1) for w, b in zip(week, boost)]
    n = len(YEARS)
    k, m = sum(d > 0 for d in over_forest), sum(d > 0 for d in over_week)

    def listed(values):
        parts = [number(v, lang, 1) for v in values]
        if lang == "nl":
            return "; ".join(parts[:-1]) + " en " + parts[-1]
        return ", ".join(parts[:-1]) + " and " + parts[-1]

    if lang == "nl":
        return (f"Boosting verslaat het forest in {k} van de {n} jaren en de week ervoor in {m} van de {n}. Per jaar, van "
                f"{YEARS[0]} tot en met {YEARS[-1]}, zit boosting er zoveel minder naast dan het forest: "
                f"{listed(over_forest)} voertuigen per uur; en zoveel minder dan de week ervoor: {listed(over_week)}.")
    return (f"Boosting beats the forest in {k} of the {n} years and the week before in {m} of the {n}. Year by year, from "
            f"{YEARS[0]} to {YEARS[-1]}, boosting misses this much less than the forest: {listed(over_forest)} vehicles "
            f"an hour; and this much less than the week before: {listed(over_week)}.")
