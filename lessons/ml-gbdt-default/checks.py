# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-gbdt-default: the clean table of traffic of ml-project-table-explore (data/metro-interstate-
traffic.csv.gz), one row per hour, with the calendar and the weather as features; the test MAE of the models the lesson
compares, of histogram gradient boosting with every number of bins on the slider, with and without a column for the
holidays, and with early stopping on 2016; the bins of the column `hour`; and the functions whole_day and stop_on.
The numbers in capitals were measured once with scikit-learn on one core; tests/test_ml_gbdt_default.py measures them
again."""
import contextlib
import functools
import io
import math
import re
import warnings

from everything_ai import chart, check as checking, paths, yours
from everything_ai.check import FunctionBox, Mistake

DATA = "data/metro-interstate-traffic.csv.gz"   # see data/README.md
TEST_FROM = "2017-01-01"                         # learn from the hours before, test on the hours from then on
VAL_FROM = "2016-01-01"                          # exercise 2: learn from before 2016, stop early on 2016
ROWS, TRAIN, TEST = 40564, 25318, 15246

# The test MAE, vehicles an hour, of what the worked example compares.
MAE = {"mean": 1738.3, "linear": 570.4, "forest": 274.8, "hgb": 260.3}
DUMMY_COLUMNS = 58          # the columns of the linear regression: 4 numbers and 24 + 7 + 12 + 11 one-hot columns
# The weather in a word as a category, as 11 one-hot columns, and left out.
WEATHER = {"category": 260.3, "one-hot": 259.1, "none": 259.8}

# The slider: the test MAE of histogram gradient boosting with every max_bins, and where scikit-learn cuts `hour`.
BINS = [12, 16, 20, 24, 64, 255]
BINS_MAE = {12: 395.6, 16: 334.4, 20: 304.1, 24: 260.6, 64: 258.9, 255: 260.3}
HOUR_CUTS = {12: [1, 3, 5, 8, 10, 12, 14, 16, 18, 20, 22],
             16: [1, 2, 4, 5, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22],
             20: [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22]}   # 24 and more: every hour

# The hour alone cut in the groups of 12, 16 and 20 bins, with 255 bins for the other features.
HOUR_ONLY = {12: 397.3, 16: 335.8, 20: 305.5}
# The default model on the training hours: early stopping "auto" holds back a random tenth and still keeps all 100
# trees; with early_stopping=False it learns from every hour.
DEFAULT = {"trees": 100, "all_rows": 258.1}
# Monday minus Sunday, in mean traffic of the training hours, at 7:00 and at 2:00.
GAP = {7: 4655, 2: -343}
# Not tested, because they depend on the computer: in a test of the course, on one core, the forest of the worked
# example took 6.7 seconds and gradient boosting 0.3.

# The holidays: 53 names, one on the first hour of each holiday; 18 of those days in the test years, 431 hours.
HOLIDAY = {"names": 53, "hours": 1203, "test_days": 18, "test_hours": 431}
WITH_HOLIDAY = {"all": 235.8, "holidays": 507.3}       # the test MAE with the column of exercise 1, and on its holidays
WITHOUT_HOLIDAY = {"all": 260.3, "holidays": 913.5}
WEEK = ("2017-07-03 00:00", "2017-07-09 23:00")       # the week of Independence Day 2017, all 168 hours

# Exercise 2: 17,481 hours before 2016 to learn from and the 7,837 of 2016 to stop on; at most 300 trees. Early stopping
# watches the loss the model learns on, the squared error: its root on 2016 is lowest after 40 trees, and the 10 trees
# after that do not get it lower, so the model stops at 50. The MAE on 2016 at the stop and at 300 trees, and on the test.
STOP = {"trees": 50, "lowest": 40, "rmse": 492.0, "lowest_rmse": 490.2, "all_trees_rmse": 506.7,
        "val": 319.5, "all_trees_val": 329.5, "test": 245.7, "all_trees_test": 243.8}
# The root of the mean squared error on 2016 after every number of trees from 20 on, without stopping.
VAL_CURVE = [(20, 526.1), (25, 498.9), (30, 492.0), (35, 490.2), (40, 490.2), (45, 490.9), (50, 492.0), (55, 492.1),
             (60, 492.8), (70, 494.7), (80, 495.2), (100, 496.3), (120, 497.3), (150, 499.0), (200, 503.5),
             (250, 505.7), (300, 506.7)]


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


# --- the table --------------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def hours():
    """The clean table of the project, as the worked example makes it: no temperature of 0 K, no rain of 100 mm or more
    in an hour, the first row of every hour, and the time as the index."""
    import pandas as pd
    df = pd.read_csv(paths.ROOT / DATA)
    df = df[df["temp"] > 0]
    df = df[df["rain_1h"] < 100]
    df = df.drop_duplicates(subset="date_time")
    df["date_time"] = pd.to_datetime(df["date_time"])
    return df.set_index("date_time")


def features(df):
    """The features of the worked example, in its order: the calendar, the weather, and the weather in a word as a
    category."""
    import pandas as pd
    X = pd.DataFrame(index=df.index)
    X["hour"] = df.index.hour
    X["weekday"] = df.index.dayofweek
    X["month"] = df.index.month
    X["temp"] = df["temp"] - 273.15
    X["rain"] = df["rain_1h"]
    X["snow"] = df["snow_1h"]
    X["clouds"] = df["clouds_all"]
    X["weather"] = df["weather_main"].astype("category")
    return X


def whole_day(df) -> list:
    """The course's whole_day: 1 for every hour of a day that has a holiday name in one of its hours, else 0."""
    days = df.index.date
    holidays = set(days[df["holiday"].notna()])
    marks = []
    for day in days:
        if day in holidays:
            marks.append(1)
        else:
            marks.append(0)
    return marks


@functools.lru_cache(maxsize=1)
def hour_means() -> tuple:
    """The mean traffic of every hour of the day in the training hours, and how many training hours each has."""
    df = hours()
    train = df[df.index < TEST_FROM]
    g = train["traffic_volume"].groupby(train.index.hour)
    return tuple(float(v) for v in g.mean()), tuple(int(v) for v in g.size())


def hour_bins(b: int) -> list:
    """The hours of the day that share a bin when scikit-learn cuts `hour` with max_bins b: a list of lists."""
    cuts = HOUR_CUTS.get(b)
    if cuts is None:
        return [[h] for h in range(24)]
    out, now = [], []
    for h in range(24):
        now.append(h)
        if h in cuts:
            out.append(now)
            now = []
    if now:
        out.append(now)
    return out


def _widest(b: int) -> tuple | None:
    """Of the hours that share a bin, the two next to each other with the most different traffic."""
    means, _ = hour_means()
    best = None
    for group in hour_bins(b):
        for a, c in zip(group, group[1:]):
            gap = abs(means[a] - means[c])
            if best is None or gap > best[0]:
                best = (gap, a, c)
    return None if best is None else (best[1], best[2])


def bins_said(b: int, lang: str) -> str:
    """The sentence under the picture of the slider."""
    means, _ = hour_means()
    mae = number(BINS_MAE[b], lang, 1)
    pair = _widest(b)
    if pair is None:
        if lang == "nl":
            return (f"Met {b} bins krijgt elk uur van de dag een eigen bin. Test-MAE: {mae} voertuigen per uur; "
                    f"tussen 24 en 255 bins verschilt die minder dan 2.")
        return (f"With {b} bins, every hour of the day gets a bin of its own. Test MAE: {mae} vehicles an hour; "
                f"between 24 and 255 bins, it changes by less than 2.")
    a, c = pair
    k = len(hour_bins(b))
    if lang == "nl":
        return (f"Met {b} bins krijgt `hour` er {k}, dus sommige uren delen een bin. {a}:00 en {c}:00, met gemiddeld "
                f"{number(means[a], lang)} en {number(means[c], lang)} voertuigen, zien er voor elke boom hetzelfde uit. "
                f"Test-MAE: {mae} voertuigen per uur.")
    return (f"With {b} bins, `hour` gets {k} of them, so some hours share a bin. {a}:00 and {c}:00, with on average "
            f"{number(means[a], lang)} and {number(means[c], lang)} vehicles, look the same to every tree. "
            f"Test MAE: {mae} vehicles an hour.")


def bins_picture(b: int, labels: dict) -> str:
    """The mean traffic of every hour of the day in the training hours, as bars, and for every bin of `hour` a line at
    the mean of its hours: what a tree that splits on `hour` alone can tell apart."""
    means, counts = hour_means()
    p = chart.Panel((-0.6, 23.6), (0, 6000), title=labels["title"], xlabel=labels["hour"], ylabel=labels["vehicles"],
                    xticks=[0, 6, 12, 18, 23], yticks=[0, 2000, 4000, 6000])
    for h in range(24):
        p.rect(h - 0.38, 0, h + 0.38, means[h], role="ink-3")
    groups = hour_bins(b)
    for group in groups:
        mean = sum(means[h] * counts[h] for h in group) / sum(counts[h] for h in group)
        p.line([group[0] - 0.45, group[-1] + 0.45], [mean, mean], role="param", width=3.0)
    for group in groups[1:]:
        x = group[0] - 0.5
        p.line([x, x], [0, 6000], role="ink-3", width=0.8, dashed=True)
    return chart.figure([p], alt=labels["alt"], width=460, height=260)


# The lesson is done when these are correct. "first-model" is the choice question in the lesson.
REQUIRED = ["whole-day", "stop-on", "first-model"]


class _Val:
    """A test value that the check's messages write by its name, such as `traffic`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value)


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


# --- exercise 1: a holiday for the whole day ----------------------------------------------------------------------------

def _made(name: str, start: str, n: int, named: dict):
    """n made-up hours from start, with a holiday name in the hours of `named` ({position: name})."""
    import pandas as pd
    index = pd.date_range(start, periods=n, freq="h")
    return _Val(name, pd.DataFrame({"holiday": [named.get(i) for i in range(n)],
                                    "traffic_volume": [1000 + 37 * (i % 24) for i in range(n)]}, index=index))


_TRAFFIC = _Val("traffic", hours())
_CHRISTMAS = _made("christmas", "2024-12-24 00:00", 72, {24: "Christmas Day"})
_NONE = _made("quiet", "2025-03-03 00:00", 48, {})
_TWO = _made("two_holidays", "2025-05-24 00:00", 120, {48: "Memorial Day", 98: "Made-up Day"})
_GAPS = _Val("gaps", _CHRISTMAS.value.drop(_CHRISTMAS.value.index[[24, 25, 30]]).assign(
    holiday=lambda d: [("Christmas Day" if t.hour == 2 and t.day == 25 else None) for t in d.index]))


def _as_marks(v):
    """A list of 0 and 1, or None when v is not one: a list, a Series or an array of 0 and 1, or of True and False."""
    import numpy as np
    try:
        values = list(v)
    except TypeError:
        return None
    out = []
    for x in values:
        if isinstance(x, (bool, int, np.integer, np.bool_)) and int(x) in (0, 1):
            out.append(int(x))
        else:
            return None
    return out


def _same_marks(got, want) -> bool:
    mine = _as_marks(got)
    return mine is not None and mine == list(want)


def _shown_marks(v):
    marks = _as_marks(v)
    if marks is None:
        return repr(v)[:80]
    if len(marks) > 12:
        return {"en": f"a list of {number(len(marks), 'en')} marks, {number(sum(marks), 'en')} of them 1",
                "nl": f"een lijst van {number(len(marks), 'nl')} markeringen, waarvan {number(sum(marks), 'nl')} op 1"}
    return "[" + ", ".join(str(m) for m in marks) + "]"


def _w(hol: str = 'set(days[df["holiday"].notna()])', test: str = "day in holidays") -> str:
    return ("def whole_day(df):\n    days = df.index.date\n    holidays = " + hol + "\n    marks = []\n"
            "    for day in days:\n        if " + test + ":\n            marks.append(1)\n        else:\n"
            "            marks.append(0)\n    return marks\n")


_W_START = _w("set(days[...])", "...")
_W = _w()

_WD_WORDS = {
    "title": {"en": "Your column for the week of Independence Day 2017", "nl": "Jouw kolom voor de week van Independence Day 2017"},
    "vehicles": {"en": "vehicles an hour", "nl": "voertuigen per uur"},
    "days": {"en": ["Mon 3", "Tue 4", "Wed 5", "Thu 6", "Fri 7", "Sat 8", "Sun 9"],
             "nl": ["ma 3", "di 4", "wo 5", "do 6", "vr 7", "za 8", "zo 9"]},
    "right": {"en": "Dark line: the traffic of every hour of the week of 3 July 2017. Shaded: the hours your column marks, all of Tuesday 4 July. Over the whole table it marks {hours} hours on {days} days. With it, the test MAE drops from {before} to {after} vehicles an hour, and on the {test_days} holidays of 2017 and 2018 from {hol_before} to {hol_after}.",
              "nl": "Donkere lijn: het verkeer van elk uur van de week van 3 juli 2017. Gearceerd: de uren die jouw kolom markeert, heel dinsdag 4 juli. Over de hele tabel markeert hij {hours} uren op {days} dagen. Daarmee daalt de test-MAE van {before} naar {after} voertuigen per uur, en op de {test_days} feestdagen van 2017 en 2018 van {hol_before} naar {hol_after}."},
    "said": {"en": "Dark line: the traffic of every hour of the week of 3 July 2017. Shaded: the {week} hours your column marks in this week; over the whole table, {hours}. The course's column marks the 24 hours of Tuesday 4 July, and {want} hours over the whole table.",
             "nl": "Donkere lijn: het verkeer van elk uur van de week van 3 juli 2017. Gearceerd: de {week} uren die jouw kolom in deze week markeert; over de hele tabel {hours}. De kolom van de course markeert de 24 uren van dinsdag 4 juli, en {want} uren over de hele tabel."},
}


def _wd_picture(seen) -> str:
    lang = seen.lang
    marks = _as_marks(_call(seen, (_TRAFFIC,)))
    if marks is None or len(marks) != ROWS:
        return ""
    words = {k: v[lang] for k, v in _WD_WORDS.items()}
    df = hours()
    first = df.index.get_loc(df.loc[WEEK[0]:WEEK[1]].index[0])
    week = marks[first:first + 168]
    traffic = df["traffic_volume"].to_numpy()[first:first + 168]
    p = chart.Panel((0, 168), (0, 7000), ylabel=words["vehicles"], xticks=[12, 36, 60, 84, 108, 132, 156],
                    xtick_labels=words["days"], yticks=[0, 2000, 4000, 6000])
    start = None
    for i, m in enumerate(list(week) + [0]):      # one shaded block for every run of marked hours
        if m and start is None:
            start = i
        elif not m and start is not None:
            p.rect(start, 0, i, 7000, role="ink-3")
            start = None
    p.line(range(168), [float(v) for v in traffic], role="ink", width=1.6)
    if _same_marks(marks, whole_day(df)):
        text = words["right"].format(hours=number(HOLIDAY["hours"], lang), days=HOLIDAY["names"], test_days=HOLIDAY["test_days"],
                                     before=number(WITHOUT_HOLIDAY["all"], lang, 1), after=number(WITH_HOLIDAY["all"], lang, 1),
                                     hol_before=number(WITHOUT_HOLIDAY["holidays"], lang, 1), hol_after=number(WITH_HOLIDAY["holidays"], lang, 1))
    else:
        text = words["said"].format(week=sum(week), hours=number(sum(marks), lang), want=number(HOLIDAY["hours"], lang))
    return yours.figure(words["title"], chart.figure([p], alt=text, width=460, height=240) + _key(text), alt=text)


EXERCISES = {
    "whole-day": FunctionBox(
        name="whole-day",
        picture=_wd_picture,
        function="whole_day",
        start=_W_START,
        solution=_W,
        show=_shown_marks,
        compare=_same_marks,
        visible=[(_TRAFFIC,)],
        hidden=[(_CHRISTMAS,), (_NONE,), (_TWO,), (_GAPS,)],
        first=[_DOTS],
        mistakes=[
            Mistake("isna", wrong=_w(hol='set(days[df["holiday"].isna()])'),
                    en="`{call}` gives {got}: every hour is marked, so the mask also picks hours without a name, and every day has those. Pick only the hours with a name: `notna()`.",
                    nl="`{call}` geeft {got}: elk uur is gemarkeerd, dus de mask kiest ook uren zonder naam, en die heeft elke dag. Kies alleen de uren met een naam: `notna()`."),
            Mistake("no-dates", matches=lambda got, text: (m := _as_marks(got)) is not None and len(m) > 0 and sum(m) == 0,
                    en="`{call}` gives {got}: not one hour is marked. Either `holidays` holds no dates, or the test after `if` never says yes. `days[...]` with the mask `df[\"holiday\"].notna()` between the brackets gives the dates of the hours with a name, and `in` asks whether a date is in the set.",
                    nl="`{call}` geeft {got}: geen enkel uur is gemarkeerd. Of `holidays` bevat geen datums, of de test na `if` zegt nooit ja. `days[...]` met `df[\"holiday\"].notna()` tussen de blokhaken geeft de datums van de uren met een naam, en `in` vraagt of een datum in de set zit."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the two `...`: which hours have a holiday name, and when a day is one of the holidays.",
                      "nl": "Vervang de twee `...`: welke uren een naam van een feestdag hebben, en wanneer een dag een van de feestdagen is."},
            "arguments": {"en": "The check calls `{call}`, with one value, and the `def` line of your function has a different number of names between its parentheses. Keep `(df)`.",
                          "nl": "De check roept `{call}` aan, met één waarde, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(df)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. A day is a holiday when one of its hours has a name; then every hour of that day gets a 1.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Een dag is een feestdag als een van zijn uren een naam heeft; dan krijgt elk uur van die dag een 1."},
            "almost": {"en": "Right for the traffic. But `{call}` gives {got}, and it should give {want}. Does your function use `df`, and nothing of the traffic?",
                       "nl": "Goed voor het verkeer. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `df`, en niets van het verkeer?"},
            "correct": {"en": "Your function is right in all {n} cases. The 53 holiday names become 1,203 marked hours: every hour of the 53 holidays that has a row.",
                        "nl": "Je functie klopt in alle {n} gevallen. De 53 namen van feestdagen worden 1.203 gemarkeerde uren: elk uur van de 53 feestdagen dat een rij heeft."},
        },
    ),
}


# --- exercise 2: stop early on a validation set -------------------------------------------------------------------------

def _split():
    """The table with the column of exercise 1, cut in the hours before 2016 and the hours of 2016."""
    df = hours()
    X = features(df)
    X["holiday"] = whole_day(df)
    y = df["traffic_volume"]
    train = X.index < VAL_FROM
    val = (X.index >= VAL_FROM) & (X.index < TEST_FROM)
    return X[train], y[train], X[val], y[val]


def _made_split(n: int, seed: int):
    """n made-up rows to learn from and 1,000 to stop on: three features, the label a wave of the first plus the second."""
    import numpy as np
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, (n, 3))
    y = np.sin(x[:, 0]) * 3 + x[:, 1] + rng.normal(0, 1, n)
    xv = rng.uniform(0, 10, (1000, 3))
    yv = np.sin(xv[:, 0]) * 3 + xv[:, 1] + rng.normal(0, 1, 1000)
    return x, y, xv, yv


_X_TRAIN, _Y_TRAIN, _X_VAL, _Y_VAL = (_Val(n, v) for n, v in zip(["X_train", "y_train", "X_val", "y_val"], _split()))
_SMALL = tuple(_Val(n, v) for n, v in zip(["X_small", "y_small", "X_small_val", "y_small_val"], _made_split(2000, 0)))
_MIDDLE = tuple(_Val(n, v) for n, v in zip(["X_middle", "y_middle", "X_middle_val", "y_middle_val"], _made_split(5000, 1)))

# `fit(X_train, y_train, X_val, ...)`: a third value without a name. scikit-learn takes it as `sample_weight`, and stops
# with an error that says nothing about the validation set, so the check says it before the code runs.
_POSITIONAL = Mistake("positional", matches=lambda got, text: re.search(r"\.fit\(\s*\w+\s*,\s*\w+\s*,\s*\w+\s*[,)]", text) is not None,
                      en="`fit` takes the validation set only by name, after the features and the label: a third value without a name is taken as something else, and `fit` stops with an error. Write `X_val=X_val, y_val=y_val`.",
                      nl="`fit` neemt de validation set alleen bij naam, na de features en het label: een derde waarde zonder naam wordt als iets anders gelezen, en `fit` stopt met een fout. Schrijf `X_val=X_val, y_val=y_val`.")

_S_HEAD = "from sklearn.ensemble import HistGradientBoostingRegressor\n\n\ndef stop_on(X_train, y_train, X_val, y_val):\n"


def _s(stop: str = "True", fit: str = "X_train, y_train, X_val=X_val, y_val=y_val") -> str:
    return (_S_HEAD + f"    model = HistGradientBoostingRegressor(max_iter=300, early_stopping={stop}, random_state=0)\n"
            f"    model.fit({fit})\n    return model.n_iter_\n")


_S_START = _s("...", "...")
_S = _s()

_SO_WORDS = {
    "title": {"en": "Your early stopping on 2016", "nl": "Jouw early stopping op 2016"},
    "trees": {"en": "trees", "nl": "bomen"},
    "error": {"en": "error on 2016 (RMSE)", "nl": "fout op 2016 (RMSE)"},
    "lowest": {"en": "lowest", "nl": "laagst"},
    "right": {"en": "The line: the error early stopping watches, the root of the mean squared error on the hours of 2016, after every number of trees learned from the hours before 2016. It is lowest after 40 trees; the 10 trees after that do not get it lower, so your function stops at 50, the ring. In MAE: 319.5 vehicles an hour on 2016 at the stop, and 329.5 with all 300 trees. On 2017 and 2018, the 50 trees miss by 245.7 vehicles an hour and 300 trees by 243.8: about the same, with a sixth of the trees.",
              "nl": "De lijn: de fout waar early stopping naar kijkt, de wortel uit de gemiddelde gekwadrateerde fout op de uren van 2016, na elk aantal bomen geleerd van de uren vóór 2016. Die is het laagst na 40 bomen; de 10 bomen daarna brengen hem niet lager, dus jouw functie stopt bij 50, de ring. In MAE: 319,5 voertuigen per uur op 2016 bij de stop, en 329,5 met alle 300 bomen. Op 2017 en 2018 zitten de 50 bomen er 245,7 voertuigen per uur naast en 300 bomen 243,8: ongeveer even goed, met een zesde van de bomen."},
    "said": {"en": "The line: the error early stopping watches, the root of the mean squared error on the hours of 2016, after every number of trees learned from the hours before 2016. Your function keeps {n} trees{dot}; the course's stops at 50, the ring, 10 trees after the lowest point.",
             "nl": "De lijn: de fout waar early stopping naar kijkt, de wortel uit de gemiddelde gekwadrateerde fout op de uren van 2016, na elk aantal bomen geleerd van de uren vóór 2016. Jouw functie houdt {n} bomen{dot}; die van de course stopt bij 50, de ring, 10 bomen na het laagste punt."},
    "dot": {"en": ", the dot", "nl": ", de stip"},
}


def _curve_at(n: int) -> float:
    """The error on 2016 at n trees, read off VAL_CURVE between its points (from 20 trees on)."""
    pts = VAL_CURVE
    for (a, va), (b, vb) in zip(pts, pts[1:]):
        if a <= n <= b:
            return va + (vb - va) * (n - a) / (b - a)
    return pts[-1][1]


def _so_picture(seen) -> str:
    lang = seen.lang
    n = _call(seen, (_X_TRAIN, _Y_TRAIN, _X_VAL, _Y_VAL))
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 300:
        return ""
    words = {k: v[lang] for k, v in _SO_WORDS.items()}
    p = chart.Panel((20, 300), (485, 530), xlog=True, xlabel=words["trees"], ylabel=words["error"],
                    xticks=[20, 40, 50, 100, 200, 300], yticks=[490, 500, 510, 520, 530])
    low = STOP["lowest"]
    p.line([low, low], [485, 530], role="ink-3", width=0.8, dashed=True)
    p.text(low * 0.97, 527, words["lowest"], role="ink-2", anchor="end")
    p.line([a for a, _ in VAL_CURVE], [v for _, v in VAL_CURVE], role="loss", width=2.2)
    p.points([STOP["trees"]], [STOP["rmse"]], role="ink", r=6, hollow=True)
    if n >= 20:
        p.points([n], [_curve_at(n)], role="ink", r=4)
    if n == STOP["trees"]:
        text = words["right"]
    else:
        text = words["said"].format(n=n, dot=words["dot"] if n >= 20 else "")
    return yours.figure(words["title"], chart.figure([p], alt=text, width=460, height=240) + _key(text), alt=text)


EXERCISES["stop-on"] = FunctionBox(
    name="stop-on",
    picture=_so_picture,
    function="stop_on",
    start=_S_START,
    solution=_S,
    visible=[(_X_TRAIN, _Y_TRAIN, _X_VAL, _Y_VAL)],
    hidden=[_SMALL, _MIDDLE],
    first=[_DOTS, _POSITIONAL],
    mistakes=[
        Mistake("off", error=lambda e: "early_stopping is False" in str(e),
                en="With `{call}`, scikit-learn stops with an error: with `early_stopping=False` it does not take `X_val` and `y_val`. Turn early stopping on: `early_stopping=True`.",
                nl="Met `{call}` stopt scikit-learn met een fout: met `early_stopping=False` neemt het geen `X_val` en `y_val` aan. Zet early stopping aan: `early_stopping=True`."),
        Mistake("no-val", wrong=_s(fit="X_train, y_train"),
                en="`{call}` gives {got}: without `X_val` and `y_val`, the model stops early on a tenth of `X_train` picked at random, hours from between the hours it learns from. Give it the validation set: `X_val=X_val, y_val=y_val` in `fit`.",
                nl="`{call}` geeft {got}: zonder `X_val` en `y_val` stopt het model op een willekeurig gekozen tiende van `X_train`, uren tussen de uren waarvan het leert. Geef het de validation set: `X_val=X_val, y_val=y_val` in `fit`."),
        Mistake("train-val", wrong=_s(fit="X_train, y_train, X_val=X_train, y_val=y_train"),
                en="`{call}` gives {got}: on the hours it learns from, every new tree helps a little, so it never stops. Stop on the hours it does not learn from: `X_val=X_val, y_val=y_val`.",
                nl="`{call}` geeft {got}: op de uren waarvan het leert helpt elke nieuwe boom een beetje, dus het stopt nooit. Stop op de uren waarvan het niet leert: `X_val=X_val, y_val=y_val`."),
        Mistake("auto", wrong=_s(stop='"auto"'),
                en="`{call}` gives {got}: in scikit-learn 1.9, `\"auto\"` turns early stopping on only for more than 10,000 rows to learn from, also when `X_val` is given, and this case has fewer. Turn it on yourself: `early_stopping=True`.",
                nl="`{call}` geeft {got}: in scikit-learn 1.9 zet `\"auto\"` early stopping alleen aan bij meer dan 10.000 rijen om van te leren, ook als `X_val` gegeven is, en dit geval heeft er minder. Zet het zelf aan: `early_stopping=True`."),
        Mistake("half-val", error=lambda e: "was not provided" in str(e),
                en="With `{call}`, `fit` stops with an error: it needs both halves of the validation set, the features and the labels. Give both: `X_val=X_val, y_val=y_val`.",
                nl="Met `{call}` stopt `fit` met een fout: het heeft beide helften van de validation set nodig, de features en de labels. Geef ze allebei: `X_val=X_val, y_val=y_val`."),
    ],
    texts=_SAME | {
        "empty": {"en": "Replace the two `...`: turn early stopping on, and give `fit` the rows to learn from and the validation set to stop on.",
                  "nl": "Vervang de twee `...`: zet early stopping aan, en geef `fit` de rijen om van te leren en de validation set om op te stoppen."},
        "arguments": {"en": "The check calls `{call}`, with four values, and the `def` line of your function has a different number of names between its parentheses. Keep `(X_train, y_train, X_val, y_val)`.",
                      "nl": "De check roept `{call}` aan, met vier waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(X_train, y_train, X_val, y_val)`."},
        "wrong": {"en": "`{call}` gives {got}, and it should give {want}. With `early_stopping=True`, `fit` learns from `X_train` and `y_train`, and stops when 10 trees in a row did not lower the error on `X_val` and `y_val`.",
                  "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Met `early_stopping=True` leert `fit` van `X_train` en `y_train`, en stopt het als 10 bomen op rij de fout op `X_val` en `y_val` niet lieten dalen."},
        "almost": {"en": "Right for the traffic. But `{call}` gives {got}, and it should give {want}. Does your function use its four values, and nothing of the traffic?",
                   "nl": "Goed voor het verkeer. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie zijn vier waarden, en niets van het verkeer?"},
        "correct": {"en": "Your function is right in all {n} cases. On the traffic it keeps 50 of at most 300 trees: after the 40th, 10 trees in a row did not lower the error on 2016.",
                    "nl": "Je functie klopt in alle {n} gevallen. Op het verkeer houdt hij 50 van hoogstens 300 bomen: na de 40e lieten 10 bomen op rij de fout op 2016 niet meer dalen."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("whole-day", _W_START, "empty", None),
    ("whole-day", _w(test="..."), "wrong", "dots-left"),
    ("whole-day", _W, "correct", None),
    ("whole-day", _w(hol='set(days[df["holiday"].notna().to_numpy()])', test="holidays.__contains__(day)"), "correct", None),
    ("whole-day", "def whole_day(df):\n    days = df.index.normalize()\n    return days.isin(days[df[\"holiday\"].notna()])\n", "correct", None),
    ("whole-day", _w(hol='set(days[df["holiday"].isna()])'), "wrong", "isna"),
    ("whole-day", _w(hol='set(df["holiday"].notna())'), "wrong", "no-dates"),
    ("whole-day", _w(hol='set(df["holiday"])'), "wrong", "no-dates"),
    ("stop-on", _S_START, "empty", None),
    ("stop-on", _s(fit="..."), "wrong", "dots-left"),
    ("stop-on", _S, "correct", None),
    ("stop-on", _s(fit="X_train, y_train, sample_weight=None, X_val=X_val, y_val=y_val"), "correct", None),
    ("stop-on", _s(stop="False"), "error", "off"),
    ("stop-on", _s(fit="X_train, y_train"), "wrong", "no-val"),
    ("stop-on", _s(fit="X_train, y_train, X_val=X_train, y_val=y_train"), "wrong", "train-val"),
    ("stop-on", _s(stop='"auto"'), "wrong", "auto"),
    ("stop-on", _s(fit="X_train, y_train, X_val, y_val"), "wrong", "positional"),
    ("stop-on", _s(fit="X_train, y_train, X_val"), "wrong", "positional"),
    ("stop-on", _s(fit="X_train, y_train, X_val=X_val"), "error", "half-val"),
    ("whole-day", _w(test="day == holidays"), "wrong", "no-dates"),
    ("whole-day", _w(hol='set(days[df["holiday"] != None])'), "wrong", "isna"),
]
