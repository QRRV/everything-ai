# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ts-forecast-demand: the bikes rented by registered riders of Capital Bikeshare every hour of
2011 and 2012 (data/bike-sharing-hour.csv.gz) on a grid of hours, forecast a week ahead from every Monday of 2012 by
gradient boosting on lag features and calendar features, for every step of the slider, beside the seasonal naive
forecast; and the functions lag_features and tree_forecast."""
import contextlib
import functools
import io
import math
import warnings

from everything_ai import chart, check as checking, paths, yours
from everything_ai.check import FunctionBox, Mistake

HOURS = "data/bike-sharing-hour.csv.gz"   # see data/README.md
FIRST = 8784          # Monday 2 January 2012, 0:00, on the grid of hours: the first origin
H = 168               # a week ahead, in hours
EXAMPLE = 12480       # Monday 4 June 2012, 0:00: the origin of the week drawn by the slider and by exercise 2
LAGS = ["lag_168", "lag_336"]
CALENDAR = ["hour", "weekday", "month", "workingday"]
WEATHER = ["temp", "weathersit", "hum"]
STEPS = {1: CALENDAR, 2: ["lag_168"], 3: LAGS, 4: LAGS + CALENDAR, 5: LAGS + CALENDAR + WEATHER}
PEEKING = {"lag-24": ["lag_24"] + LAGS + CALENDAR, "lag-1-24": ["lag_1", "lag_24"] + LAGS + CALENDAR}


# --- the riders, the table and the backtests --------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def hours():
    """The table of hours of the course on a grid: every hour of 2011 and 2012 a row, NaN in every column of the 165
    hours without one."""
    import pandas as pd
    h = pd.read_csv(paths.ROOT / HOURS)
    when = pd.to_datetime(h["dteday"]) + pd.to_timedelta(h["hr"], unit="h")
    return h.set_index(when).asfreq("h")


@functools.lru_cache(maxsize=1)
def table():
    """What the lesson forecasts, registered, and every column it tries: the lags 1, 24, 168 and 336, the calendar, and
    the weather as it turned out."""
    import pandas as pd
    t = hours()
    y = t["registered"]
    out = pd.DataFrame(index=t.index)
    out["registered"] = y
    for k in (1, 24, 168, 336):
        out[f"lag_{k}"] = y.shift(k)
    out["hour"] = t.index.hour
    out["weekday"] = t.index.dayofweek
    out["month"] = t.index.month
    out["workingday"] = t["workingday"]
    for c in WEATHER:
        out[c] = t[c]
    return out


def origins() -> list:
    """Every Monday of 2012 at 0:00, from 2 January to 24 December: 52 origins, a week apart."""
    return list(range(FIRST, len(table()) - H + 1, H))


def _tree(train, columns: list):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(random_state=0).fit(train[columns], train["registered"])


def _one_core():
    """scikit-learn on one core, as in the worked example: when other programs keep cores busy, cores that wait for
    each other cost more than they win. A backtest took 67 seconds on four cores and 5 on one while another program kept
    a core busy; the forecasts are the same. threadpoolctl limits only the libraries already loaded, so scikit-learn's
    trees are imported first."""
    import sklearn.ensemble  # noqa: F401
    from threadpoolctl import threadpool_limits
    return threadpool_limits(1, user_api="openmp")


def _backtest(columns: tuple) -> dict:
    import numpy as np
    T = table()
    y = T["registered"].to_numpy()
    guess = np.full(len(T), np.nan)
    maes = []
    with _one_core():
        for o in origins():
            model = _tree(T.iloc[:o].dropna(subset=["registered"]), list(columns))
            guess[o:o + H] = model.predict(T.iloc[o:o + H][list(columns)])
            maes.append(float(np.nanmean(abs(y[o:o + H] - guess[o:o + H]))))
    return {"maes": maes, "mae": float(np.mean(maes)), "guess": guess}


def backtest(columns) -> dict:
    """The rolling-origin backtest of the tree on these columns: at every origin, trained on the hours before it that
    have riders, it forecasts the 168 hours after it. The MAE of every week, their mean, and the forecast of every hour.
    Made once per process (checking.kept): it trains 52 models."""
    columns = tuple(columns)
    return checking.kept(("ts-forecast-demand", columns), lambda: _backtest(columns))


# The MAE over the 52 weeks of 2012 of every step of the slider, as backtest(STEPS[step]) gives it. That is 260 models,
# too many to train while the lesson opens, so the slider shows these numbers; tests/test_ts_forecast_demand.py trains
# the models and checks every one of them.
STEP_MAE = {1: 70.24, 2: 49.46, 3: 42.08, 4: 42.00, 5: 34.17}


def example(step: int):
    """The forecast of the tree of this step for the week from Monday 4 June 2012: one model, trained as at that origin
    in the backtest, so the same forecast."""
    return checking.kept(("ts-forecast-demand", "example", step),
                         lambda: tree_forecast(table(), STEPS[step], EXAMPLE, H))


def _lagged(guess) -> dict:
    import numpy as np
    y = table()["registered"].to_numpy()
    maes = [float(np.nanmean(abs(y[o:o + H] - guess[o:o + H]))) for o in origins()]
    return {"maes": maes, "mae": float(np.mean(maes)), "guess": guess}


def seasonal() -> dict:
    """The seasonal naive forecast a week ahead: every hour is the same hour a week earlier, the column lag_168."""
    return _lagged(table()["lag_168"].to_numpy())


def two_weeks() -> dict:
    """No model at all: every hour is the mean of the same hour one and two weeks earlier."""
    T = table()
    return _lagged(((T["lag_168"] + T["lag_336"]) / 2).to_numpy())


def tested():
    """The hours of the 52 weeks of the backtest that have riders, as a boolean array over the table."""
    import numpy as np
    on = np.zeros(len(table()), bool)
    on[FIRST:origins()[-1] + H] = True
    return on & table()["registered"].notna().to_numpy()


def holidays():
    """The tested hours of the 11 holidays of 2012, from Monday 2 January to Christmas."""
    return tested() & (hours()["holiday"] == 1).to_numpy()


def on_holidays(guess) -> float:
    import numpy as np
    y = table()["registered"].to_numpy()
    at = holidays() & ~np.isnan(guess)
    return float(np.mean(abs(y[at] - guess[at])))


def bias(guess) -> float:
    """The mean of forecast minus riders over the tested hours: below 0, the forecast is too low."""
    import numpy as np
    y = table()["registered"].to_numpy()
    at = tested() & ~np.isnan(guess)
    return float(np.mean(guess[at] - y[at]))


def week(result: dict, origin: int = EXAMPLE) -> float:
    """The MAE of one week, from the origin given."""
    return result["maes"][origins().index(origin)]


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{abs(x):,.{digits}f}"
    text = text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text
    return ("−" if x < 0 and round(abs(x), digits) != 0 else "") + text


STEP_WORDS = {
    1: {"en": "the calendar: hour, weekday, month and workingday", "nl": "de kalender: hour, weekday, month en workingday"},
    2: {"en": "lag_168 alone", "nl": "alleen lag_168"},
    3: {"en": "lag_168 and lag_336", "nl": "lag_168 en lag_336"},
    4: {"en": "both lags and the calendar", "nl": "beide lags en de kalender"},
    5: {"en": "both lags, the calendar and the weather as it turned out",
        "nl": "beide lags, de kalender en het werkelijke weer"},
}


def said(step: int, lang: str) -> str:
    """The sentence under the picture of the slider."""
    import numpy as np
    in_week = float(np.mean(abs(table()["registered"].to_numpy()[EXAMPLE:EXAMPLE + H] - example(step))))
    v = lambda x: number(x, lang, 1)
    if lang == "nl":
        return (f"Stap {step}, {STEP_WORDS[step]['nl']}: een MAE van {v(STEP_MAE[step])} fietsers per uur over de 52 weken "
                f"van 2012, tegen {v(seasonal()['mae'])} voor de seasonal naive forecast. In de week van 4 juni: {v(in_week)}.")
    return (f"Step {step}, {STEP_WORDS[step]['en']}: an MAE of {v(STEP_MAE[step])} riders an hour over the 52 weeks of "
            f"2012, against {v(seasonal()['mae'])} for the seasonal naive forecast. In the week of 4 June: {v(in_week)}.")


# --- the pictures --------------------------------------------------------------------------------------------------------

def _runs(xs, ys):
    """The parts of a line between missing values: a line is drawn only where it has numbers."""
    run = []
    for x, y in zip(xs, ys):
        if y is None or not math.isfinite(y):
            if len(run) > 1:
                yield run
            run = []
        else:
            run.append((x, y))
    if len(run) > 1:
        yield run


def _line(panel, ys, role: str, width: float, dashed: bool = False, top: float = 800.0) -> None:
    for run in _runs(range(len(ys)), [float(v) for v in ys]):
        panel.line([a for a, _ in run], [max(0.0, min(top, b)) for _, b in run], role=role, width=width, dashed=dashed)


_WEEK_DAYS = {"en": ["4 Jun", "6 Jun", "8 Jun", "10 Jun"], "nl": ["4 jun", "6 jun", "8 jun", "10 jun"]}


def _week_panel(title: str, ylabel: str, lang: str):
    """The 168 hours from Monday 4 June 2012, 0:00, with the riders as a thin gray line."""
    p = chart.Panel((0, H), (0, 800), title=title, ylabel=ylabel, xticks=[0, 48, 96, 144], xtick_labels=_WEEK_DAYS[lang],
                    yticks=[0, 200, 400, 600, 800])
    _line(p, table()["registered"].to_numpy()[EXAMPLE:EXAMPLE + H], "ink-3", 1.2)
    return p


def picture(step: int, labels: dict) -> str:
    """Left: the week of 4 June 2012, the riders, the seasonal naive forecast and the tree on the columns of this step.
    Right: the MAE over the 52 weeks of 2012 of every step (STEP_MAE), as bars, and of the seasonal naive forecast, as a
    dashed line."""
    left = _week_panel(labels["left"], labels["ylabel"], labels["lang"])
    _line(left, seasonal()["guess"][EXAMPLE:EXAMPLE + H], "ink-3", 1.4, dashed=True)
    _line(left, example(step), "ink", 2.0)
    right = chart.Panel((0.4, 5.6), (0, 80), title=labels["right"], xlabel=labels["step"], ylabel="MAE",
                        xticks=[1, 2, 3, 4, 5], xtick_labels=["1", "2", "3", "4", "5"], yticks=[0, 20, 40, 60, 80])
    for s in STEPS:
        right.rect(s - 0.32, 0, s + 0.32, STEP_MAE[s], role="ink" if s == step else "ink-3", solid=s == step)
    right.line([0.4, 5.6], [seasonal()["mae"]] * 2, role="ink-3", width=1.2, dashed=True)
    right.text(5.5, seasonal()["mae"] + 4, labels["seasonal"], role="ink-2", anchor="end")
    return chart.figure([left, right], alt=labels["alt"], width=330, height=270)


# The lesson is done when these are correct. "known-columns" is the choice question in the lesson.
REQUIRED = ["lag-features", "tree-forecast", "known-columns"]


class _Val:
    """A test value that the check's messages write by its name, such as `riders`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value)


def _series(values: list) -> _Val:
    import pandas as pd
    return _Val("pd.Series([" + ", ".join(repr(float(v)) for v in values) + "])", pd.Series([float(v) for v in values]))


def _frame(name: str, columns: dict) -> _Val:
    import pandas as pd
    return _Val(name, pd.DataFrame({k: [float(v) for v in vs] for k, vs in columns.items()}))


def _shown(v):
    import numpy as np
    import pandas as pd
    if isinstance(v, pd.DataFrame):
        n = len(v)
        if v.shape[1] == 0:
            return {"en": f"a table without columns, with {number(n, 'en')} rows",
                    "nl": f"een tabel zonder kolommen, met {number(n, 'nl')} rijen"}
        names = [f"`{c}`" for c in v.columns]
        return {"en": f"a table with the columns {', '.join(names)} and {number(n, 'en')} rows",
                "nl": f"een tabel met de kolommen {', '.join(names)} en {number(n, 'nl')} rijen"}
    if isinstance(v, (pd.Series, list, tuple)):
        v = np.asarray(v, dtype=float) if all(isinstance(x, (int, float, np.floating, np.integer)) for x in v) else v
    if isinstance(v, np.ndarray) and v.ndim == 1 and v.dtype.kind in "iuf":
        one = lambda x: "nan" if math.isnan(x) else str(round(float(x), 1))
        if len(v) <= 6:
            return "[" + ", ".join(one(x) for x in v) + "]"
        return {"en": "[" + ", ".join(one(x) for x in v[:4]) + f", …], {number(len(v), 'en')} values",
                "nl": "[" + ", ".join(one(x) for x in v[:4]) + f", …], {number(len(v), 'nl')} waarden"}
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


def _key(text: str) -> str:
    return ('<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.15rem">'
            + text + "</div>")


# --- exercise 1: lag features --------------------------------------------------------------------------------------------

_RIDERS = _Val("riders", hours()["registered"])
_ASKED = [1, 24, 168, 336]

_L_HEAD = "import pandas as pd\n\n\ndef lag_features(y, lags, h):\n    table = pd.DataFrame(index=y.index)\n    for k in lags:\n"


def _l(body: str = '        if k >= h:\n            table[f"lag_{k}"] = y.shift(k)\n') -> str:
    return _L_HEAD + body + "    return table\n"


_L_START = _l("        ...\n")
_L = _l()


def lag_features(y, lags, h):
    """The course's lag_features: a column lag_k for every lag k of at least h."""
    import pandas as pd
    out = pd.DataFrame(index=y.index)
    for k in lags:
        if k >= h:
            out[f"lag_{k}"] = y.shift(k)
    return out


def _shift_of(y, column):
    """Which shift of y a column is, of the lags asked for and the same lags the other way; None when it is none of
    them."""
    import numpy as np
    try:
        got = np.asarray(column, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    if len(got) != len(y):
        return None
    for j in _ASKED + [-k for k in _ASKED]:
        if np.allclose(got, y.shift(j).to_numpy(), equal_nan=True):
            return j
    return None


def _names_only(got, text) -> bool:
    """The right columns, by their values, under other names."""
    import pandas as pd
    if not isinstance(got, pd.DataFrame) or got.shape[1] != len(LAGS) or list(got.columns) == LAGS:
        return False
    return [_shift_of(_RIDERS.value, got.iloc[:, j]) for j in range(got.shape[1])] == [168, 336]


_LF_WORDS = {
    "title": {"en": "Your lag_features(riders, [1, 24, 168, 336], 168)", "nl": "Jouw lag_features(riders, [1, 24, 168, 336], 168)"},
    "xlabel": {"en": "hours from the origin", "nl": "uren vanaf de origin"},
    "said": {"en": "One row for every column of your table: the hours it reads for the 168 hours of the week ahead. Filled before the origin, the dashed line, where the riders are known when the forecast is made; open from the origin on, where they are not. The course's table keeps lag_168 and lag_336.",
             "nl": "Een rij voor elke kolom van je tabel: de uren die hij leest voor de 168 uren van de week vooruit. Gevuld voor de origin, de gestreepte lijn, waar de fietsers bekend zijn als de forecast gemaakt wordt; open vanaf de origin, waar ze dat niet zijn. De tabel van de course houdt lag_168 en lag_336."},
    "right": {"en": "Your table, the course's: lag_168 reads the hours 168 to 1 before the origin for the week ahead, lag_336 the hours 336 to 169 before it. Both are filled: known at the origin. lag_1 and lag_24 would run on past the dashed line.",
              "nl": "Jouw tabel, die van de course: lag_168 leest voor de week vooruit de uren 168 tot 1 voor de origin, lag_336 de uren 336 tot 169 ervoor. Allebei gevuld: bekend bij de origin. lag_1 en lag_24 zouden doorlopen voorbij de gestreepte lijn."},
}


def _lf_picture(seen) -> str:
    import pandas as pd
    got = _call(seen, (_RIDERS, list(_ASKED), H))
    if not isinstance(got, pd.DataFrame) or got.shape[1] == 0:
        return ""
    lang = seen.lang
    words = {k: v[lang] for k, v in _LF_WORDS.items()}
    names = list(got.columns)[:6]
    n = max(len(names), 2)
    p = chart.Panel((-360, 360), (-0.7, n - 0.3), xlabel=words["xlabel"], xticks=[-336, -168, 0, 168, 336],
                    xtick_labels=["−336", "−168", "0", "168", "336"], yticks=[])
    p.line([0, 0], [-0.7, n - 0.3], role="ink-3", width=0.8, dashed=True)
    for i, name in enumerate(names):
        row = n - 1 - i
        j = _shift_of(_RIDERS.value, got[name])
        p.text(-352, row + 0.3, str(name)[:16], role="ink-2", anchor="start")
        if j is None:
            continue
        lo, hi = max(-360, -j), min(360, H - j)   # forecast hour t reads hour t - j; t runs from 0 to 167
        if lo < 0:
            p.rect(lo, row - 0.18, min(hi, 0), row + 0.12, role="ink", solid=True)
        if hi > 0:
            p.rect(max(lo, 0), row - 0.18, hi, row + 0.12, role="ink", solid=False)
    right = list(got.columns) == LAGS and checking.same_value(got, lag_features(_RIDERS.value, _ASKED, H))
    text = words["right"] if right else words["said"]
    return yours.figure(words["title"], chart.figure([p], alt=text, width=460, height=60 + 42 * n) + _key(text), alt=text)


EXERCISES = {
    "lag-features": FunctionBox(
        name="lag-features",
        picture=_lf_picture,
        function="lag_features",
        start=_L_START,
        solution=_L,
        show=_shown,
        visible=[(_RIDERS, list(_ASKED), H)],
        hidden=[(_series([3, 1, 4, 1, 5, 9]), [1, 2, 3], 2), (_series([3, 1, 4, 1, 5, 9]), [3], 3),
                (_series([2, 7, 1, 8]), [2, 1], 1), (_series([2, 7, 1, 8]), [1, 2], 3)],
        first=[_DOTS],
        mistakes=[
            Mistake("future", wrong=_l('        if k >= h:\n            table[f"lag_{k}"] = y.shift(-k)\n'),
                    en="`{call}` gives {got}, with the right names, but the values come from later hours: `shift(-k)` moves every value `k` rows earlier, so each row sees the hour `k` after it, which no forecast knows. `shift(k)` moves them later, so each row sees the hour `k` before it.",
                    nl="`{call}` geeft {got}, met de goede namen, maar de waarden komen van latere uren: `shift(-k)` schuift elke waarde `k` rijen naar voren, dus ziet elke rij het uur `k` erna, en dat kent geen forecast. `shift(k)` schuift ze naar achteren, dus ziet elke rij het uur `k` ervoor."),
            Mistake("all", wrong=_l('        table[f"lag_{k}"] = y.shift(k)\n'),
                    en="`{call}` gives {got}, and it should give {want}. Your table keeps every lag, also the ones shorter than the horizon: a week ahead, `lag_24` of Tuesday 0:00 is Monday 0:00, the first hour to forecast, so not known when the forecast is made. Keep only the lags `k` of at least `h`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je tabel houdt elke lag, ook die korter dan de horizon: een week vooruit is `lag_24` van dinsdag 0:00 het uur maandag 0:00, het eerste uur dat voorspeld moet worden, dus niet bekend als de forecast gemaakt wordt. Houd alleen de lags `k` van minstens `h`."),
            Mistake("strict", wrong=_l('        if k > h:\n            table[f"lag_{k}"] = y.shift(k)\n'),
                    en="`{call}` gives {got}, and it should give {want}. `lag_168` is missing, and a week ahead it may stay: the last hour of the week is 167 hours after the origin, so 168 hours earlier is the hour just before the origin. A lag equal to the horizon is allowed: `k >= h`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. `lag_168` ontbreekt, en een week vooruit mag hij blijven: het laatste uur van de week ligt 167 uur na de origin, dus 168 uur eerder is het uur net voor de origin. Een lag gelijk aan de horizon mag: `k >= h`."),
            Mistake("short", wrong=_l('        if k <= h:\n            table[f"lag_{k}"] = y.shift(k)\n'),
                    en="`{call}` gives {got}, and it should give {want}. You keep the lags up to the horizon, the wrong way round: for most of the week, `lag_1` and `lag_24` read hours of that week itself. Keep the lags of at least `h`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je houdt de lags tot de horizon, net andersom: voor het grootste deel van de week lezen `lag_1` en `lag_24` uren van die week zelf. Houd de lags van minstens `h`."),
            Mistake("names", matches=_names_only,
                    en="The right columns, `y.shift(168)` and `y.shift(336)`, but under other names. The worked example and exercise 2 ask for them by name: `lag_168` and `lag_336`, `lag_` followed by the lag.",
                    nl="De goede kolommen, `y.shift(168)` en `y.shift(336)`, maar met andere namen. Het uitgewerkte voorbeeld en oefening 2 vragen ze op naam: `lag_168` en `lag_336`, `lag_` gevolgd door de lag."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the `...`: for every lag `k` that the horizon `h` allows, a column named `lag_k`, such as `lag_168`, with every value moved `k` rows later.",
                      "nl": "Vervang de `...`: voor elke lag `k` die de horizon `h` toestaat, een kolom met de naam `lag_k`, zoals `lag_168`, met elke waarde `k` rijen later."},
            "arguments": {"en": "The check calls `{call}`, with three values, and the `def` line of your function has a different number of names between its parentheses. Keep `(y, lags, h)`.",
                          "nl": "De check roept `{call}` aan, met drie waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(y, lags, h)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your table get a column `lag_k` with `y.shift(k)` for every lag `k` of at least `h`, and no other?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Krijgt je tabel een kolom `lag_k` met `y.shift(k)` voor elke lag `k` van minstens `h`, en geen andere?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function use `h` and `lags`, and not the numbers of the example?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `h` en `lags`, en niet de getallen van het voorbeeld?"},
            "correct": {"en": "Your function is right in all {n} cases. A week ahead, `lag_168` and `lag_336` stay: for every hour of the week, they read an hour before the origin. A lag equal to the horizon is the shortest that may stay; with `h` 1, every lag may.",
                        "nl": "Je functie klopt in alle {n} gevallen. Een week vooruit blijven `lag_168` en `lag_336`: voor elk uur van de week lezen ze een uur voor de origin. Een lag gelijk aan de horizon is de kortste die mag blijven; met `h` 1 mag elke lag."},
        },
    ),
}


# --- exercise 2: a forecast by a tree from one origin ---------------------------------------------------------------------

_TABLE = _Val("table", table()[["registered"] + LAGS + CALENDAR])
_COLUMNS = _Val("columns", LAGS + CALENDAR)

_T_HEAD = "from sklearn.ensemble import HistGradientBoostingRegressor\n\n\ndef tree_forecast(table, columns, origin, h):\n"


def _t(train: str = 'table.iloc[:origin].dropna(subset=["registered"])', fit: str = 'train[columns], train["registered"]') -> str:
    return (_T_HEAD + f"    train = {train}\n    test = table.iloc[origin:origin + h]\n"
            "    model = HistGradientBoostingRegressor(random_state=0)\n" + f"    model.fit({fit})\n"
            "    return model.predict(test[columns])\n")


_T_START = _t("...", "..., ...")
_T = _t()


def tree_forecast(table, columns, origin, h):
    """The course's tree_forecast: trained on the rows before the origin that have riders, it forecasts the h rows
    from the origin on."""
    with _one_core():
        return _tree(table.iloc[:origin].dropna(subset=["registered"]), list(columns)).predict(table.iloc[origin:origin + h][list(columns)])


def _small() -> _Val:
    """40 made-up hours: the riders rise by 1 every hour and a weekday of 7 hours adds 0 to 60."""
    return _frame("small", {"registered": [i + 10 * (i % 7) for i in range(40)], "lag_7": [math.nan] * 7 + [i + 10 * (i % 7) for i in range(33)],
                            "weekday": [i % 7 for i in range(40)]})


def _gaps() -> _Val:
    """60 made-up hours, every fifth without riders."""
    return _frame("gaps", {"registered": [math.nan if i % 5 == 4 else 3 * (i % 6) + i // 10 for i in range(60)],
                           "hour": [i % 6 for i in range(60)]})


_TF_WORDS = {
    "title": {"en": "Your tree_forecast(table, columns, 12480, 168)", "nl": "Jouw tree_forecast(table, columns, 12480, 168)"},
    "ylabel": {"en": "registered riders", "nl": "geregistreerde fietsers"},
    "said": {"en": "Gray line: the riders of every hour of the week of Monday 4 June 2012. Thick line: your forecast for that week from its origin, an MAE of {mine}; dashed: the course's, {want}.",
             "nl": "Grijze lijn: de fietsers van elk uur van de week van maandag 4 juni 2012. Dikke lijn: jouw forecast voor die week vanaf de origin, een MAE van {mine}; gestreept: die van de course, {want}."},
    "right": {"en": "Gray line: the riders of every hour of the week of Monday 4 June 2012. Thick line: your forecast for that week, the course's: an MAE of {want} riders an hour, where the seasonal naive forecast misses by {seasonal}.",
              "nl": "Grijze lijn: de fietsers van elk uur van de week van maandag 4 juni 2012. Dikke lijn: jouw forecast voor die week, die van de course: een MAE van {want} fietsers per uur, waar de seasonal naive forecast er {seasonal} naast zit."},
}


def _tf_picture(seen) -> str:
    import numpy as np
    got = _call(seen, (_TABLE, _COLUMNS, EXAMPLE, H))
    if got is None:
        return ""
    try:
        got = np.asarray(got, dtype=float).ravel()
    except (TypeError, ValueError):
        return ""
    if len(got) == 0 or not np.isfinite(got).any():
        return ""
    lang = seen.lang
    words = {k: v[lang] for k, v in _TF_WORDS.items()}
    y = table()["registered"].to_numpy()[EXAMPLE:EXAMPLE + H]
    want = example(4)
    p = _week_panel("", words["ylabel"], lang)
    _line(p, want, "ink-3", 1.6, dashed=True)
    _line(p, got[:H], "ink", 2.2)
    def mae(g):
        miss = abs(y[:len(g)] - g[:len(y)])
        miss = miss[np.isfinite(miss)]
        return float(np.mean(miss)) if len(miss) else math.nan
    right = len(got) == H and checking.same_value(got, want)
    text = (words["right"] if right else words["said"]).format(
        mine=number(mae(got), lang, 1) if math.isfinite(mae(got)) else "–", want=number(mae(want), lang, 1), seasonal=number(week(seasonal()), lang, 1))
    return yours.figure(words["title"], chart.figure([p], alt=text, width=460, height=250) + _key(text), alt=text)


EXERCISES["tree-forecast"] = FunctionBox(
    name="tree-forecast",
    picture=_tf_picture,
    function="tree_forecast",
    start=_T_START,
    solution=_T,
    show=_shown,
    visible=[(_TABLE, _COLUMNS, EXAMPLE, H)],
    hidden=[(_small(), ["lag_7", "weekday"], 30, 7), (_gaps(), ["hour"], 48, 6), (_gaps(), ["hour"], 30, 12)],
    first=[_DOTS],
    mistakes=[
        Mistake("nan-label", error=lambda e: "NaN" in str(e),
                en="`{call}` stops with an error about NaN: `train` still holds the hours without riders. On the grid of hours, `registered` is NaN where the table had no row, 165 hours, and a model cannot learn from a label it does not have. Drop those rows: `dropna(subset=[\"registered\"])`.",
                nl="`{call}` stopt met een fout over NaN: `train` bevat nog de uren zonder fietsers. Op het rooster van uren is `registered` NaN waar de tabel geen rij had, 165 uren, en een model kan niet leren van een label dat het niet heeft. Laat die rijen vallen: `dropna(subset=[\"registered\"])`."),
        Mistake("label-in", error=lambda e: "feature names" in str(e),
                en="`{call}` stops with an error about the feature names: your model learned from other columns than the ones it gets in `predict`, `test[columns]`. Fit on `train[columns]`, and give the label, `train[\"registered\"]`, as the second argument: the riders are what it learns, never a column it learns from.",
                nl="`{call}` stopt met een fout over de namen van de features: je model leerde van andere kolommen dan die het in `predict` krijgt, `test[columns]`. Fit op `train[columns]`, en geef het label, `train[\"registered\"]`, als tweede argument: de fietsers zijn wat het leert, nooit een kolom waarvan het leert."),
        Mistake("all", wrong=_t('table.dropna(subset=["registered"])'),
                en="`{call}` gives {got}, and it should give {want}. `train` is the whole table, the week it should forecast and everything after it too, so the model has seen the answers. Train only on the rows before `origin`.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. `train` is de hele tabel, ook de week die hij moet voorspellen en alles daarna, dus heeft het model de antwoorden gezien. Train alleen op de rijen voor `origin`."),
        Mistake("peek", wrong=_t('table.iloc[:origin + h].dropna(subset=["registered"])'),
                en="`{call}` gives {got}, and it should give {want}. `train` runs on into the `h` rows it should forecast, so the model has seen them. Stop `train` at `origin`.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. `train` loopt door in de `h` rijen die hij moet voorspellen, dus heeft het model die gezien. Laat `train` stoppen bij `origin`."),
        Mistake("drop-all", wrong=_t("table.iloc[:origin].dropna()"),
                en="`{call}` gives {got}, and it should give {want}. `dropna()` without `subset` drops every row with a NaN anywhere: 517 hours with riders, such as the first two weeks, whose `lag_336` is missing. `HistGradientBoostingRegressor` handles a missing feature by itself; only a missing label has to go: `dropna(subset=[\"registered\"])`.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. `dropna()` zonder `subset` laat elke rij met ergens een NaN vallen: 517 uren met fietsers, zoals de eerste twee weken, waarvan `lag_336` ontbreekt. `HistGradientBoostingRegressor` kan zelf met een ontbrekende feature omgaan; alleen een ontbrekend label moet weg: `dropna(subset=[\"registered\"])`."),
    ],
    texts=_SAME | {
        "empty": {"en": "Replace the `...`: `train`, the rows before `origin` that have riders, and the two arguments of `fit`, the features and the label.",
                  "nl": "Vervang de `...`: `train`, de rijen voor `origin` die fietsers hebben, en de twee argumenten van `fit`, de features en het label."},
        "arguments": {"en": "The check calls `{call}`, with four values, and the `def` line of your function has a different number of names between its parentheses. Keep `(table, columns, origin, h)`.",
                      "nl": "De check roept `{call}` aan, met vier waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(table, columns, origin, h)`."},
        "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is `train` every row before `origin` with a known `registered`, and does the model fit on `train[columns]` with `train[\"registered\"]` as label?",
                  "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Is `train` elke rij voor `origin` met een bekende `registered`, en fit het model op `train[columns]` met `train[\"registered\"]` als label?"},
        "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function use `columns` and `origin`, and not the names and numbers of the example?",
                   "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `columns` en `origin`, en niet de namen en getallen van het voorbeeld?"},
        "correct": {"en": "Your function is right in all {n} cases. From Monday 4 June 2012, the tree misses by 47.5 riders an hour in the week ahead; the seasonal naive forecast by 59.5.",
                    "nl": "Je functie klopt in alle {n} gevallen. Vanaf maandag 4 juni 2012 zit de boom er in de week vooruit 47,5 fietsers per uur naast; de seasonal naive forecast 59,5."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("lag-features", _L_START, "empty", None),
    ("lag-features", _L, "correct", None),
    ("lag-features", _l('        if h <= k:\n            table["lag_" + str(k)] = y.shift(k)\n'), "correct", None),
    ("lag-features", _l('        if k >= h:\n            table[f"lag_{k}"] = y.shift(-k)\n'), "wrong", "future"),
    ("lag-features", _l('        table[f"lag_{k}"] = y.shift(k)\n'), "wrong", "all"),
    ("lag-features", _l('        if k > h:\n            table[f"lag_{k}"] = y.shift(k)\n'), "wrong", "strict"),
    ("lag-features", _l('        if k <= h:\n            table[f"lag_{k}"] = y.shift(k)\n'), "wrong", "short"),
    ("lag-features", _l('        if k >= h:\n            table[f"lag{k}"] = y.shift(k)\n'), "wrong", "names"),
    ("lag-features", _l('        if k >= 168:\n            table[f"lag_{k}"] = y.shift(k)\n'), "almost", None),
    ("tree-forecast", _T_START, "empty", None),
    ("tree-forecast", _t('table.iloc[:origin].dropna(subset=["registered"])', "..., ..."), "wrong", "dots-left"),
    ("tree-forecast", _T, "correct", None),
    ("tree-forecast", _t('table[:origin][table["registered"][:origin].notna()]'), "correct", None),
    ("tree-forecast", _t("table.iloc[:origin]"), "error", "nan-label"),
    ("tree-forecast", _t(fit='train, train["registered"]'), "error", "label-in"),
    ("tree-forecast", _t('table.dropna(subset=["registered"])'), "wrong", "all"),
    ("tree-forecast", _t('table.iloc[:origin + h].dropna(subset=["registered"])'), "wrong", "peek"),
    ("tree-forecast", _t("table.iloc[:origin].dropna()"), "wrong", "drop-all"),
    ("tree-forecast", _t('table.iloc[:12480].dropna(subset=["registered"])'), "almost", None),
]
