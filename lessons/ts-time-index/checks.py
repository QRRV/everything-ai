# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ts-time-index: the traffic on Interstate 94 of ml-project-table-explore (data/metro-interstate-
traffic.csv.gz) as a time series, one row per hour with the time as its index; four whole weeks of October 2017
smoothed with a rolling window of every size on the slider; and the functions week_change and smooth."""
import contextlib
import functools
import io
import math
import warnings

from everything_ai import chart, check as checking, paths, yours
from everything_ai.check import FunctionBox, Mistake

DATA = "data/metro-interstate-traffic.csv.gz"   # see data/README.md
WINDOWS = [1, 3, 6, 24, 168]                     # the stops of the slider, in hours
FIRST, LAST = "2017-10-02 00:00", "2017-10-29 23:00"   # four whole weeks, Monday to Sunday, without a missing hour


# --- the table as a time series -------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def traffic():
    """The traffic of every hour that has a row, the first row of each hour, with the time as the index."""
    import pandas as pd
    df = pd.read_csv(paths.ROOT / DATA)
    df = df.drop_duplicates(subset="date_time")
    df["date_time"] = pd.to_datetime(df["date_time"])
    return df.set_index("date_time")["traffic_volume"]


def stretch(first: str, last: str):
    """A piece of the series, both ends included, as a copy."""
    return traffic().loc[first:last].copy()


@functools.lru_cache(maxsize=None)
def window(hours: int) -> dict:
    """The rolling mean over the last `hours` hours, worked out on the whole series and cut to the four weeks: its
    values, its lowest and highest, the hours at which the weekdays peak, and its mean on weekdays and weekends."""
    import numpy as np
    import pandas as pd
    r = traffic().rolling(f"{hours}h").mean().loc[FIRST:LAST]
    days = [r.loc[str(d.date())] for d in pd.date_range(FIRST, LAST, freq="D").normalize() if d.dayofweek < 5]
    weekday = r.index.dayofweek < 5
    return {"values": r.to_numpy(), "min": float(r.min()), "max": float(r.max()),
            "peaks": sorted({int(x.idxmax().hour) for x in days}),
            "morning": sorted({int(x.between_time("05:00", "10:00").idxmax().hour) for x in days}),
            "afternoon": sorted({int(x.between_time("14:00", "20:00").idxmax().hour) for x in days}),
            "weekday": float(np.mean(r[weekday])), "weekend": float(np.mean(r[~weekday]))}


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def said(hours: int, lang: str) -> str:
    """The sentence under the picture of the slider."""
    w = window(hours)
    lo, hi = number(w["min"], lang), number(w["max"], lang)
    if lang == "nl":
        start = (f"Window van 1 uur: de uren zelf, van {lo} tot {hi} voertuigen. " if hours == 1 else
                 f"Window van {hours} uur: elk punt is het gemiddelde van de laatste {hours} uur, van {lo} tot {hi} voertuigen per uur. ")
        rest = {1: "Op een werkdag piekt het verkeer twee keer, rond 7:00 en rond 16:00.",
                3: "De twee pieken van een werkdag zijn lager en komen een uur later, rond 8:00 en 17:00.",
                6: "Het hoogste punt van een werkdag ligt nu om 17:00 of 18:00, en is nog lager.",
                24: f"Van de dag is niets meer over; wat overblijft is de week: werkdagen gemiddeld {number(w['weekday'], lang)}, zaterdag en zondag {number(w['weekend'], lang)}.",
                168: f"Ook van de week is bijna niets over: werkdagen gemiddeld {number(w['weekday'], lang)}, het weekend {number(w['weekend'], lang)}."}
        return start + rest[hours]
    start = (f"Window of 1 hour: the hours themselves, from {lo} to {hi} vehicles. " if hours == 1 else
             f"Window of {hours} hours: every point is the mean of the last {hours} hours, from {lo} to {hi} vehicles an hour. ")
    rest = {1: "On a weekday the traffic peaks twice, around 7:00 and around 16:00.",
            3: "The two peaks of a weekday are lower and come an hour later, around 8:00 and 17:00.",
            6: "The highest point of a weekday is now at 17:00 or 18:00, and lower still.",
            24: f"The day is smoothed away; what is left is the week: weekdays average {number(w['weekday'], lang)}, Saturdays and Sundays {number(w['weekend'], lang)}.",
            168: f"The week is smoothed away too: weekdays average {number(w['weekday'], lang)}, weekends {number(w['weekend'], lang)}."}
    return start + rest[hours]


# --- the picture of the slider ---------------------------------------------------------------------------------------

def picture(hours: int, labels: dict) -> str:
    """Four weeks of traffic, every hour as a thin gray line, and the rolling mean over `hours` hours as a thick one,
    with a tick at every Monday."""
    raw = stretch(FIRST, LAST).to_numpy()
    w = window(hours)
    n = len(raw)
    p = chart.Panel((0, n - 1), (0, 7500), title=labels["title"], ylabel=labels["ylabel"],
                    xticks=[0, 168, 336, 504], xtick_labels=labels["days"], yticks=[0, 2000, 4000, 6000])
    xs = list(range(n))
    if hours > 1:
        p.line(xs, [float(v) for v in raw], role="ink-3", width=0.9)
    p.line(xs, [float(v) for v in w["values"]], role="ink", width=1.4 if hours == 1 else 2.2)
    return chart.figure([p], alt=labels["alt"], width=460, height=260)


# The lesson is done when these are correct. "spring" is the choice question in the lesson.
REQUIRED = ["week-change", "smooth", "spring"]


class _Val:
    """A test value that the check's messages write by its name, such as `may`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value)


def _series_words(v) -> dict:
    """A Series in words, for the messages: how many values, how many of them numbers, and the first number."""
    n = len(v)
    numbers = v.dropna()
    k = len(numbers)
    if k == 0:
        return {"en": f"a Series of {number(n, 'en')} values, none of them a number",
                "nl": f"een Series van {number(n, 'nl')} waarden, geen ervan een getal"}
    t, x = numbers.index[0], float(numbers.iloc[0])
    at = f"{t:%Y-%m-%d %H:%M}" if hasattr(t, "strftime") else str(t)
    first = {lang: f"{'−' if x < 0 else ''}{number(abs(x), lang, 0 if x == int(x) else 1)}" for lang in ("en", "nl")}
    return {"en": f"a Series of {number(n, 'en')} values, {number(k, 'en')} of them numbers, the first {first['en']} at {at}",
            "nl": f"een Series van {number(n, 'nl')} waarden, waarvan {number(k, 'nl')} getallen, het eerste {first['nl']} om {at}"}


def _shown(v):
    import pandas as pd
    if isinstance(v, pd.Series):
        return _series_words(v)
    if isinstance(v, (list, tuple)) and v and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
        if len(v) > 6:
            return {"en": "[" + ", ".join(str(round(float(x), 2)) for x in v[:4]) + f", …], {len(v)} values",
                    "nl": "[" + ", ".join(str(round(float(x), 2)) for x in v[:4]) + f", …], {len(v)} waarden"}
        return "[" + ", ".join(str(round(float(x), 3)) for x in v) + "]"
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


# --- exercise 1: the change from a week earlier -----------------------------------------------------------------------

def _steps():
    """Ten days of made-up hours, the value of each its number of hours since the start, with four hours missing: a
    week later, every hour that is there with its hour a week before has gone up by exactly 168."""
    import pandas as pd
    s = pd.Series([float(i) for i in range(240)], index=pd.date_range("2020-01-06", periods=240, freq="h"))
    return s.drop(s.index[[5, 100, 173, 201]])


_MAY = _Val("may", stretch("2016-05-02 00:00", "2016-05-29 23:00"))      # 654 of its 672 hours have a row
_OCTOBER = _Val("october", stretch("2017-10-02 00:00", "2017-10-15 23:00"))   # two whole weeks
_FOUR = _Val("four_days", stretch("2017-10-02 00:00", "2017-10-05 23:00"))    # shorter than a week
_STEPS = _Val("steps", _steps())

_W_HEAD = "def week_change(s):\n"


def _w(hours: str = 's.asfreq("h")', back: str = "hours - hours.shift(168)") -> str:
    return _W_HEAD + f"    hours = {hours}\n    return {back}\n"


_W_START = _W_HEAD + "    hours = ...\n    return hours - ...\n"
_W = _w()

_CHANGE_WORDS = {
    "title": {"en": "Your week_change(may)", "nl": "Jouw week_change(may)"},
    "ylabel": {"en": "change from a week before", "nl": "verschil met een week eerder"},
    "days": {"en": ["Mon 2 May", "Mon 9 May", "Mon 16 May", "Mon 23 May"], "nl": ["ma 2 mei", "ma 9 mei", "ma 16 mei", "ma 23 mei"]},
    "said": {"en": "Your function gives {got} numbers; the course's {want}, gray and dashed. The first week has no week before it, and where an hour or its hour a week earlier has no row, the line stops.",
             "nl": "Jouw functie geeft {got} getallen; die van de course {want}, grijs en gestreept. De eerste week heeft geen week ervoor, en waar een uur of zijn uur een week eerder geen rij heeft, stopt de lijn."},
    "right": {"en": "Your function gives the course's {want} numbers. The first week has no week before it, and where an hour or its hour a week earlier has no row, the line stops: 18 hours of May have no row.",
              "nl": "Jouw functie geeft de {want} getallen van de course. De eerste week heeft geen week ervoor, en waar een uur of zijn uur een week eerder geen rij heeft, stopt de lijn: 18 uren van mei hebben geen rij."},
}


def _pieces(xs, ys):
    """The stretches of a line between its missing values."""
    out, now = [], []
    for x, y in zip(xs, ys):
        if y is None or isinstance(y, float) and math.isnan(y):
            if now:
                out.append(now)
            now = []
        else:
            now.append((x, float(y)))
    if now:
        out.append(now)
    return out


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


def _change_picture(seen) -> str:
    """Your week_change(may) as a line over the four weeks of May 2016, broken where it has no number, on top of the
    course's, gray and dashed. Values of yours that are not at an hour of those weeks are left out."""
    import pandas as pd
    got = _call(seen, (_MAY,))
    if not isinstance(got, pd.Series) or len(got) == 0:
        return ""
    want = _MAY.value.asfreq("h")
    want = want - want.shift(168)
    lang = seen.lang
    words = {k: v[lang] for k, v in _CHANGE_WORDS.items()}
    start = want.index[0]
    if isinstance(got.index, pd.DatetimeIndex):
        at = [(t - start) / pd.Timedelta(hours=1) for t in got.index]
    else:
        at = list(range(len(got)))
    try:
        values = [float(v) for v in got.to_numpy()]
    except (TypeError, ValueError):
        return ""
    keep = [(a, v) for a, v in zip(at, values) if 0 <= a <= len(want) - 1]
    p = chart.Panel((0, len(want) - 1), (-1500, 1500), ylabel=words["ylabel"],
                    xticks=[0, 168, 336, 504], xtick_labels=words["days"], yticks=[-1000, 0, 1000])
    p.line([0, len(want) - 1], [0, 0], role="ink-3", width=0.8)
    for piece in _pieces(range(len(want)), want.to_numpy()):
        p.line([a for a, _ in piece], [v for _, v in piece], role="ink-3", width=1.6, dashed=True)
    for piece in _pieces([a for a, _ in keep], [max(-1500.0, min(1500.0, v)) if math.isfinite(v) else v for _, v in keep]):
        p.line([a for a, _ in piece], [v for _, v in piece], role="ink", width=1.3)
    n_got, n_want = int(got.notna().sum()), int(want.notna().sum())
    right = checking.same_value(got, want)
    text = (words["right"] if right else words["said"]).format(got=number(n_got, lang), want=number(n_want, lang))
    key = ('<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.15rem">'
           + text + "</div>")
    return yours.figure(words["title"], chart.figure([p], alt=text, width=460, height=250) + key, alt=text)


EXERCISES = {
    "week-change": FunctionBox(
        name="week-change",
        picture=_change_picture,
        function="week_change",
        start=_W_START,
        solution=_W,
        show=_shown,
        visible=[(_MAY,)],
        hidden=[(_OCTOBER,), (_FOUR,), (_STEPS,)],
        first=[_DOTS],
        mistakes=[
            Mistake("rows", wrong=_w(hours="s"),
                    en="`{call}` gives {got}, and it should give {want}. `shift(168)` moves 168 rows, and where hours have no row, 168 rows back is not a week back. Put every hour in its place first, with `asfreq`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. `shift(168)` schuift 168 rijen op, en waar uren geen rij hebben, is 168 rijen terug geen week terug. Zet eerst elk uur op zijn plek, met `asfreq`."),
            Mistake("zero", wrong=_w(hours='s.asfreq("h", fill_value=0)'),
                    en="`{call}` gives {got}, and it should give {want}. Your function makes an hour without a row an hour with 0 vehicles, so its change is the whole traffic of the other hour. Leave it missing: NaN.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie maakt van een uur zonder rij een uur met 0 voertuigen, dus is het verschil het hele verkeer van het andere uur. Laat het ontbreken: NaN."),
            Mistake("ahead", wrong=_w(back="hours - hours.shift(-168)"),
                    en="`{call}` gives {got}, and it should give {want}. `shift(-168)` moves the values back, so every hour meets the hour a week later. A week earlier is `shift(168)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. `shift(-168)` schuift de waarden terug, dus elk uur ontmoet het uur een week later. Een week eerder is `shift(168)`."),
            Mistake("reversed", wrong=_w(back="hours.shift(168) - hours"),
                    en="`{call}` gives {got}, and it should give {want}: your signs are turned around. The change is now minus a week earlier.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je tekens staan andersom. Het verschil is nu min een week eerder."),
            Mistake("day", wrong=_w(back="hours - hours.shift(24)"),
                    en="`{call}` gives {got}, and it should give {want}. `shift(24)` is a day back. A week has 7 × 24 = 168 hours.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. `shift(24)` is een dag terug. Een week heeft 7 × 24 = 168 uur."),
            Mistake("days", wrong=_w(back="hours - hours.shift(7)"),
                    en="`{call}` gives {got}, and it should give {want}. On a table of hours, `shift(7)` moves 7 hours, not 7 days. A week has 7 × 24 = 168 hours.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Op een tabel van uren schuift `shift(7)` 7 uur op, geen 7 dagen. Een week heeft 7 × 24 = 168 uur."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the two `...`: every hour in its place, and the same hour a week earlier.",
                      "nl": "Vervang de twee `...`: elk uur op zijn plek, en hetzelfde uur een week eerder."},
            "arguments": {"en": "The check calls `{call}`, with one value, and the `def` line of your function has a different number of names between its parentheses. Keep `(s)`.",
                          "nl": "De check roept `{call}` aan, met één waarde, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(s)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does `hours` have every hour from the first to the last, NaN where there is no row, and do you subtract the same hour 168 hours earlier?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Heeft `hours` elk uur van het eerste tot het laatste, NaN waar geen rij is, en trek je hetzelfde uur 168 uur eerder ervan af?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function use `s`, and work for any stretch of hours?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `s`, en werkt hij voor elk stuk uren?"},
            "correct": {"en": "Your function is right in all {n} cases. May 2016 has 672 hours and 654 rows; 485 hours have a number: not the first week, and not where an hour or its hour a week earlier has no row. Four days give no number at all: they have no week before them.",
                        "nl": "Je functie klopt in alle {n} gevallen. Mei 2016 heeft 672 uren en 654 rijen; 485 uren hebben een getal: niet de eerste week, en niet waar een uur of zijn uur een week eerder geen rij heeft. Vier dagen geven helemaal geen getal: ze hebben geen week ervoor."},
        },
    ),
}


# --- exercise 2: an exponential moving average by hand -----------------------------------------------------------------

MONDAY = [int(v) for v in stretch("2017-10-02 00:00", "2017-10-02 23:00")]   # Monday 2 October 2017, hour by hour
_MONDAY = _Val("monday", MONDAY)

_S_HEAD = "def smooth(values, alpha):\n    level = values[0]\n    out = []\n    for x in values:\n"


def _s(update: str = "alpha * x + (1 - alpha) * level") -> str:
    return _S_HEAD + f"        level = {update}\n        out.append(level)\n    return out\n"


_S_START = _s("...")
_S = _s()

_SMOOTH_WORDS = {
    "title": {"en": "Your smooth(monday, 0.3)", "nl": "Jouw smooth(monday, 0.3)"},
    "xlabel": {"en": "hour of Monday 2 October 2017", "nl": "uur van maandag 2 oktober 2017"},
    "ylabel": {"en": "vehicles", "nl": "voertuigen"},
    "said": {"en": "Gray dots: the hours. The thick line is what your function gives, the dashed one the course's. Your highest point: {got} at {got_at}:00; the course's: {want} at {want_at}:00.",
             "nl": "Grijze stippen: de uren. De dikke lijn is wat jouw functie geeft, de gestreepte die van de course. Jouw hoogste punt: {got} om {got_at}:00; dat van de course: {want} om {want_at}:00."},
    "right": {"en": "Gray dots: the hours. The thick line is what your function gives: the course's. It is smoother than the hours and runs behind them: its highest point, {want}, comes at {want_at}:00, an hour after the busiest hour.",
              "nl": "Grijze stippen: de uren. De dikke lijn is wat jouw functie geeft: die van de course. Hij is gladder dan de uren en loopt erachter: zijn hoogste punt, {want}, valt om {want_at}:00, een uur na het drukste uur."},
}


def _ewm(values, alpha):
    import pandas as pd
    return [float(v) for v in pd.Series([float(x) for x in values]).ewm(alpha=alpha, adjust=False).mean()]


def _smooth_picture(seen) -> str:
    """The hours of Monday 2 October 2017 as gray dots, what the learner's smooth(monday, 0.3) gives as a thick line,
    and the course's as a dashed one, with the highest point of each."""
    got = _call(seen, (_MONDAY, 0.3))
    try:
        got = [float(v) for v in got]
    except (TypeError, ValueError):
        return ""
    if not got or not all(math.isfinite(v) for v in got):
        return ""
    want = _ewm(MONDAY, 0.3)
    lang = seen.lang
    words = {k: v[lang] for k, v in _SMOOTH_WORDS.items()}
    p = chart.Panel((0, 23), (0, 7500), xlabel=words["xlabel"], ylabel=words["ylabel"],
                    xticks=[0, 6, 12, 18, 23], yticks=[0, 2000, 4000, 6000])
    p.points(list(range(24)), MONDAY, role="ink-3", r=2.6)
    p.line(list(range(24)), want, role="ink-3", width=1.8, dashed=True)
    n = min(len(got), 24)
    p.line(list(range(n)), [max(0.0, min(7500.0, v)) for v in got[:n]], role="ink", width=2.4)
    g_at = max(range(n), key=lambda i: got[i])
    w_at = max(range(24), key=lambda i: want[i])
    right = checking.same_value(got, want)
    text = (words["right"] if right else words["said"]).format(
        got=number(got[g_at], lang), got_at=g_at, want=number(want[w_at], lang), want_at=w_at)
    key = ('<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.15rem">'
           + text + "</div>")
    return yours.figure(words["title"], chart.figure([p], alt=text, width=460, height=250) + key, alt=text)


EXERCISES["smooth"] = FunctionBox(
    name="smooth",
    picture=_smooth_picture,
    function="smooth",
    start=_S_START,
    solution=_S,
    show=_shown,
    visible=[([10, 20, 20, 0], 0.4)],
    hidden=[(_MONDAY, 0.3), ([5], 0.2), ([1, 2, 3, 4, 5, 6], 1.0), ([4, 4, 4, 0, 0], 0.1)],
    first=[_DOTS],
    mistakes=[
        Mistake("swapped", wrong=_s("alpha * level + (1 - alpha) * x"),
                en="`{call}` gives {got}, and it should give {want}. Your `alpha` goes to the old level and `1 - alpha` to the new value: turn them around, so a larger `alpha` follows the new values faster.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. Bij jou gaat `alpha` naar het oude niveau en `1 - alpha` naar de nieuwe waarde: draai ze om, zodat een grotere `alpha` de nieuwe waarden sneller volgt."),
        Mistake("no-level", wrong=_s("alpha * x"),
                en="`{call}` gives {got}, and it should give {want}. Your new level forgets the old one. It is `alpha` times the new value plus `1 - alpha` times the level so far.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. Je nieuwe niveau vergeet het oude. Het is `alpha` keer de nieuwe waarde plus `1 - alpha` keer het niveau tot nu toe."),
        Mistake("half", wrong=_s("(x + level) / 2"),
                en="`{call}` gives {got}, and it should give {want}. Your function always gives the new value half the weight, whatever `alpha` is. Let `alpha` decide: `alpha` for the new value, `1 - alpha` for the level.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie geeft de nieuwe waarde altijd de helft van het gewicht, wat `alpha` ook is. Laat `alpha` beslissen: `alpha` voor de nieuwe waarde, `1 - alpha` voor het niveau."),
    ],
    texts=_SAME | {
        "empty": {"en": "Replace the `...`: the new level, from the new value `x` and the level so far.",
                  "nl": "Vervang de `...`: het nieuwe niveau, uit de nieuwe waarde `x` en het niveau tot nu toe."},
        "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has a different number of names between its parentheses. Keep `(values, alpha)`.",
                      "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(values, alpha)`."},
        "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is the new level `alpha` times the new value plus `1 - alpha` times the level so far?",
                  "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Is het nieuwe niveau `alpha` keer de nieuwe waarde plus `1 - alpha` keer het niveau tot nu toe?"},
        "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function use `alpha` and every value it gets?",
                   "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `alpha` en elke waarde die hij krijgt?"},
        "correct": {"en": "Your function is right in all {n} cases, and gives what pandas gives with `ewm(alpha=alpha, adjust=False).mean()`. With `alpha` 1 the level is every new value itself: no smoothing at all.",
                    "nl": "Je functie klopt in alle {n} gevallen, en geeft wat pandas geeft met `ewm(alpha=alpha, adjust=False).mean()`. Met `alpha` 1 is het niveau steeds de nieuwe waarde zelf: helemaal niet gladgestreken."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("week-change", _W_START, "empty", None),
    ("week-change", _w(back="hours - ..."), "wrong", "dots-left"),
    ("week-change", _W, "correct", None),
    ("week-change", _w(hours='s.resample("h").mean()'), "correct", None),
    ("week-change", _w(back="hours.diff(168)"), "correct", None),
    ("week-change", _w(hours="s"), "wrong", "rows"),
    ("week-change", _w(hours='s.asfreq("h", fill_value=0)'), "wrong", "zero"),
    ("week-change", _w(hours='s.resample("h").sum()'), "wrong", "zero"),
    ("week-change", _w(back="hours - hours.shift(-168)"), "wrong", "ahead"),
    ("week-change", _w(back="hours.shift(168) - hours"), "wrong", "reversed"),
    ("week-change", _w(back="hours - hours.shift(24)"), "wrong", "day"),
    ("week-change", _w(back="hours - hours.shift(7)"), "wrong", "days"),
    ("smooth", _S_START, "empty", None),
    ("smooth", _S, "correct", None),
    ("smooth", _s("level + alpha * (x - level)"), "correct", None),
    ("smooth", _s("alpha * level + (1 - alpha) * x"), "wrong", "swapped"),
    ("smooth", _s("alpha * x"), "wrong", "no-level"),
    ("smooth", _s("(x + level) / 2"), "wrong", "half"),
    ("smooth", _s("0.4 * x + 0.6 * level"), "almost", None),
]
