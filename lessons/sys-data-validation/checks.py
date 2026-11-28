# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-data-validation: Rules for data: schemas and expectations. A schema says per column what is
allowed, here a range from low to high, and looks at one value at a time; an expectation is a rule about many values
together, such as the number of rows of a day or a column that must change. The learner writes the question of
`broken`, which lists every value of a batch outside its range, and the question of `stuck`, which lists the columns
that do not change during a day, with NumPy. A sort puts rules under schema or expectation, and a choice asks what to do
with the short day of Hurricane Sandy. The demo checks every day of 2011 and 2012 as a batch of the night.

The data are the hourly counts of Capital Bikeshare in Washington, D.C., in 2011 and 2012, from the Bike Sharing dataset
of Fanaee-T and Gama (2013, CC BY 4.0, see data/README.md), as they are: the humidity of 0 on 10 March 2011, the atemp
of 0.2424 on 17 August and 29 December 2012 and the short days are in the source. That Capital Bikeshare closed for
Hurricane Irene (DCist, 27 August 2011) and Hurricane Sandy (Patch, 29 October 2012; ARLnow, 31 October 2012) comes from
the news of those days, read on 28 November 2026. Every word a learner's code gives back is shown through html.escape."""
import html
import math
from functools import cache

import numpy as np
import pandas as pd

from everything_ai import check, paths, yours
from everything_ai.check import FunctionBox, Mistake, Sort

LESSON = "sys-data-validation"
LANGS = ("en", "nl")

DATA = paths.ROOT / "data" / "bike-sharing-hour.csv.gz"
COLUMNS = ["hr", "weekday", "weathersit", "temp", "atemp", "hum", "windspeed", "cnt"]
_WHOLE = {"hr", "weekday", "weathersit", "cnt"}

ROWS = (20, 24)                         # an expectation: a day has from 20 to 24 rows
LEAST = 0.01                            # an expectation: temp, atemp and hum move by at least 0.01 during a day
CHANGING = ["temp", "atemp", "hum"]


class _Named(list):
    """A list from the lesson: the check's call shows it by its name, and the learner's function gets a copy."""

    def __init__(self, name: str, values):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return [dict(v) if isinstance(v, dict) else v for v in self]


class _NamedDict(dict):
    """A dict from the lesson, shown by its name; the learner's function gets a copy, with copies of its arrays."""

    def __init__(self, name: str, values):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return {k: np.array(v) if isinstance(v, np.ndarray) else v for k, v in self.items()}


# ======== the data: every day of 2011 and 2012 ============================================================================

@cache
def days() -> dict:
    """Every day of the two years, by its date as text, as the list of its hours, each a dict of COLUMNS."""
    table = pd.read_csv(DATA)
    out = {}
    for date, part in table.groupby("dteday", sort=True):
        out[date] = [{c: int(v) if c in _WHOLE else float(v) for c, v in zip(COLUMNS, row)}
                     for row in part[COLUMNS].itertuples(index=False)]
    return out


def columns_of(rows: list, names: list) -> dict:
    """The same hours as columns: a NumPy array per name."""
    return {c: np.array([r[c] for r in rows], dtype=float) for c in names}


SCHEMA = _NamedDict("C.SCHEMA", {"hr": (0, 23), "weekday": (0, 6), "weathersit": (1, 4), "temp": (0.0, 1.0),
                                 "atemp": (0.0, 1.0), "hum": (0.01, 1.0), "windspeed": (0.0, 1.0)})
MARCH_10 = _Named("C.MARCH_10", days()["2011-03-10"])      # light rain in 20 of 22 hours, and a humidity of 0
JUNE_5 = _Named("C.JUNE_5", days()["2012-06-05"])          # a day without a problem
AUGUST_17 = _NamedDict("C.AUGUST_17", columns_of(days()["2012-08-17"], ["temp", "atemp", "hum", "windspeed"]))
MARCH_10_COLUMNS = _NamedDict("C.MARCH_10_COLUMNS", columns_of(days()["2011-03-10"], ["temp", "atemp", "hum", "windspeed"]))


def broken(rows, schema):
    found = []
    for row in rows:
        for column in schema:
            low, high = schema[column]
            if row[column] < low or row[column] > high:
                found.append((row["hr"], column))
    return found


def stuck(hours, columns, least):
    found = []
    for column in columns:
        values = hours[column]
        if values.max() - values.min() < least:
            found.append(column)
    return found


def verdict(date: str) -> dict:
    """What the gate says about the batch of one day: every rule, and whether the batch goes through."""
    rows = days()[date]
    hours = [r["hr"] for r in rows]
    out = {"date": date, "rows": len(rows), "broken": broken(rows, SCHEMA),
           "twice": len(hours) - len(set(hours)), "stuck": stuck(columns_of(rows, CHANGING), CHANGING, LEAST)}
    out["short"] = not ROWS[0] <= len(rows) <= ROWS[1]
    out["held"] = bool(out["broken"] or out["twice"] or out["stuck"] or out["short"])
    return out


@cache
def facts() -> dict:
    """The numbers the texts give."""
    every = {d: verdict(d) for d in days()}
    held = [d for d in every if every[d]["held"]]
    aug = days()["2012-08-17"]
    return {
        "days": len(every), "held": held,
        "by_schema": [d for d in held if every[d]["broken"]],
        "short": [d for d in held if every[d]["short"]],
        "stuck": [d for d in held if every[d]["stuck"]],
        "twice": [d for d in held if every[d]["twice"]],
        "rows": {d: every[d]["rows"] for d in held},
        "march": broken(list(MARCH_10), dict(SCHEMA)),
        "august": stuck({k: np.array(v) for k, v in AUGUST_17.items()}, CHANGING, LEAST),
        "aug_temp": (min(r["temp"] for r in aug), max(r["temp"] for r in aug)),
        "aug_atemp": sorted({r["atemp"] for r in aug}),
        "spread": {c: float(np.ptp(AUGUST_17[c])) for c in AUGUST_17},
    }


# ======== words and numbers ================================================================================================

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem;max-width:var(--ea-measure)"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem;max-width:var(--ea-measure)"
_SVG = "width:100%;max-width:330px;height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums"
_CODE = "var(--ea-font-code)"

MONTHS = {"en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                 "November", "December"],
          "nl": ["januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september", "oktober",
                 "november", "december"]}


def _lang(lang) -> str:
    return lang if lang in LANGS else "en"


def date_text(date: str, lang: str) -> str:
    """2011-03-10 as 10 March 2011; in Dutch 10 maart 2011."""
    y, m, d = (int(x) for x in date.split("-"))
    return f"{d} {MONTHS[_lang(lang)][m - 1]} {y}"


def day_number(date: str) -> int:
    """The day of the two years: 1 for 1 January 2011, 731 for 31 December 2012."""
    return list(days()).index(date) + 1


def two(x: float, lang: str) -> str:
    """A number with two decimals: 0.24; in Dutch 0,24."""
    out = f"{x:.2f}"
    return out.replace(".", ",") if lang == "nl" else out


def _marked(text_: str) -> str:
    """Escaped text in which `name` is set as code."""
    return "".join(f"<code>{part}</code>" if k % 2 else part for k, part in enumerate(html.escape(text_).split("`")))


def _note(text_: str, style: str = _UI) -> str:
    return f'<div style="{style}">{_marked(text_)}</div>'


def _t(x: float, y: float, text_: str, size: float = 12.5, weight: int = 400, fill: str = "var(--ea-ink)",
       anchor: str = "start", font: str = "") -> str:
    family = f"font-family:{font};" if font else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" style="{family}font-size:{size}px;'
            f'font-weight:{weight};fill:{fill};white-space:pre" xml:space="preserve">{html.escape(text_)}</text>')


def _svg(height: float, alt: str, parts: list) -> str:
    return (f'<svg viewBox="0 0 330 {height:.0f}" role="img" aria-label="{html.escape(alt)}" style="{_SVG}">'
            f'{"".join(parts)}</svg>')


def _line(x1: float, y1: float, x2: float, y2: float, stroke: str = "var(--ea-line)", width: float = 1) -> str:
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'style="stroke:{stroke};stroke-width:{width}"/>')


def _rect(x: float, y: float, w: float, h: float, style: str, rx: float = 3) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" rx="{rx}" style="{style}"/>'


def _dot(x: float, y: float, r: float, fill: str, stroke: str = "") -> str:
    s = f";stroke:{stroke};stroke-width:1.5" if stroke else ""
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" style="fill:{fill}{s}"/>'


def _arrow(x1: float, y1: float, x2: float, y2: float, stroke: str = "var(--ea-ink-3)") -> list:
    """A line with a small head at its end."""
    a = math.atan2(y2 - y1, x2 - x1)
    left = (x2 - 6 * math.cos(a - 0.45), y2 - 6 * math.sin(a - 0.45))
    right = (x2 - 6 * math.cos(a + 0.45), y2 - 6 * math.sin(a + 0.45))
    return [_line(x1, y1, x2, y2, stroke, 1.2),
            f'<polygon points="{x2:.1f},{y2:.1f} {left[0]:.1f},{left[1]:.1f} {right[0]:.1f},{right[1]:.1f}" '
            f'style="fill:{stroke}"/>']


# ======== the picture of the idea: the gate =================================================================================

IDEA_TEXT = {
    "en": {"batch": "the batch of the night: one day of hours",
           "schema": "schema: each value on its own",
           "schema_rules": "hr 0–23 · weathersit 1–4 · hum 0.01–1 · …",
           "expect": "expectations: the batch as a whole",
           "expect_rules": "20–24 rows · each hour once · columns move",
           "hold": "every rule holds", "breaks": "a rule breaks",
           "through": "to training", "held": "held back", "look": "a person looks",
           "alt": "A gate in three steps. The batch of the night, one day of hours, first meets the schema, which checks "
                  "each value on its own, then the expectations, which check the batch as a whole. When every rule "
                  "holds, the batch goes to training; when a rule breaks, the batch is held back and a person looks."},
    "nl": {"batch": "de batch van de nacht: één dag aan uren",
           "schema": "schema: elke waarde apart",
           "schema_rules": "hr 0–23 · weathersit 1–4 · hum 0,01–1 · …",
           "expect": "expectations: de batch als geheel",
           "expect_rules": "20–24 rijen · elk uur één keer · kolommen bewegen",
           "hold": "elke regel klopt", "breaks": "een regel breekt",
           "through": "naar de training", "held": "tegengehouden", "look": "een mens kijkt",
           "alt": "Een poort in drie stappen. De batch van de nacht, één dag aan uren, komt eerst bij het schema: elke "
                  "waarde apart. Dan komt de batch bij de expectations: de batch als geheel. Als elke regel klopt gaat "
                  "de batch naar de training; als een regel breekt wordt de batch tegengehouden en kijkt een mens."},
}


def idea_picture(lang: str) -> str:
    lang = _lang(lang)
    w = IDEA_TEXT[lang]
    box = "fill:var(--ea-page);stroke:var(--ea-ink-3);stroke-width:1"
    parts = [_rect(45, 2, 240, 26, box), _t(165, 19, w["batch"], 11, 600, anchor="middle")]
    parts += _arrow(165, 28, 165, 42)
    parts += [_rect(0, 42, 330, 40, box), _t(165, 58, w["schema"], 11.5, 600, anchor="middle"),
              _t(165, 74, w["schema_rules"], 10, 400, "var(--ea-ink-2)", "middle", _CODE)]
    parts += _arrow(165, 82, 165, 96)
    parts += [_rect(0, 96, 330, 40, box), _t(165, 112, w["expect"], 11.5, 600, anchor="middle"),
              _t(165, 128, w["expect_rules"], 10, 400, "var(--ea-ink-2)", "middle", _CODE)]
    parts += _arrow(140, 136, 80, 170) + _arrow(190, 136, 250, 170, "var(--ea-loss)")
    parts += [_t(96, 156, w["hold"], 10.5, 400, "var(--ea-ink-2)", "end"),
              _t(234, 156, w["breaks"], 10.5, 400, "var(--ea-loss-text)")]
    parts += [_rect(0, 172, 160, 38, box), _t(80, 195, w["through"], 11.5, 600, anchor="middle"),
              _rect(170, 172, 160, 38, "fill:var(--ea-page);stroke:var(--ea-loss);stroke-width:1.8"),
              _t(250, 188, w["held"], 11.5, 600, "var(--ea-loss-text)", "middle"),
              _t(250, 203, w["look"], 10.5, 400, "var(--ea-ink-2)", "middle")]
    return _svg(214, w["alt"], parts)


# ======== the pictures under the checks ===================================================================================

YOURS_TEXT = {
    "en": {"broken": {
               "title": "Your broken on 10 March 2011",
               "intro": "Every value of `C.MARCH_10` that the schema checks: a row per column, a square per hour of the "
                        "day. A square is marked where your `broken` lists that hour and that column. Hours without a "
                        "row are blank.",
               "hour": "hour", "total": "your broken: {k} values",
               "none": "Your `broken` stops with an error, takes too long, or gives something other than a list of "
                       "pairs of an hour and a column, so the picture waits.",
               "right": "On this day, your function lists exactly the values that break the schema: in all {n} rows, hum "
                        "is 0, below its low of 0.01, and every other value lies in its range.",
               "off": "Your function lists {k} values, and {n} values break the schema: hum, in every row.",
               "alt": "A grid of the seven columns of the schema by the hours of 10 March 2011, with the values the "
                      "learner's function lists as broken marked."},
           "stuck": {
               "title": "Your stuck on 17 August 2012",
               "intro": "The 24 hours of `C.AUGUST_17`, a line per column, from 0 at the bottom to 1 at the top: the "
                        "range of the schema. The spread is the largest value minus the smallest. A column is marked "
                        "where your `stuck` lists it.",
               "spread": "spread {v}", "moves": "moves", "stuck": "stuck",
               "none": "Your `stuck` stops with an error, takes too long, or gives something other than a list of "
                       "names of columns, so the picture waits.",
               "right": "On this day, your function lists exactly the column that is stuck: atemp stays at 0.2424 for 24 "
                        "hours, a spread of 0, while temp moves by 0.24 and hum by 0.53.",
               "off": "Your function lists {got}, and only atemp is stuck: a spread of 0, while temp moves by 0.24 and "
                      "hum by 0.53.",
               "alt": "Three small line charts of temp, atemp and hum over the 24 hours of 17 August 2012; the line of "
                      "atemp is flat, and the columns the learner's function lists as stuck are marked."}},
    "nl": {"broken": {
               "title": "Je broken op 10 maart 2011",
               "intro": "Elke waarde van `C.MARCH_10` die het schema controleert: een rij per kolom, een vakje per uur "
                        "van de dag. Een vakje is gemarkeerd waar je `broken` het uur en de kolom van dat vakje noemt. Uren zonder rij "
                        "blijven leeg.",
               "hour": "uur", "total": "je broken: {k} waarden",
               "none": "Je `broken` stopt met een fout, duurt te lang, of geeft iets anders dan een lijst van paren van "
                       "een uur en een kolom, dus wacht de afbeelding nog.",
               "right": "Op deze dag noemt je functie precies de waarden die het schema breken: in alle {n} rijen is hum "
                        "0, onder de ondergrens van 0,01, en elke andere waarde ligt binnen het eigen bereik.",
               "off": "Je functie noemt {k} waarden, en {n} waarden breken het schema: hum, in elke rij.",
               "alt": "Een rooster van de zeven kolommen van het schema tegen de uren van 10 maart 2011, met een markering "
                      "bij elke waarde die de functie van de leerling als gebroken noemt."},
           "stuck": {
               "title": "Je stuck op 17 augustus 2012",
               "intro": "De 24 uren van `C.AUGUST_17`, een lijn per kolom, van 0 onderaan tot 1 bovenaan: het bereik "
                        "van het schema. De spreiding is de grootste waarde min de kleinste. Een kolom is gemarkeerd "
                        "waar je `stuck` de kolom noemt.",
               "spread": "spreiding {v}", "moves": "beweegt", "stuck": "vast",
               "none": "Je `stuck` stopt met een fout, duurt te lang, of geeft iets anders dan een lijst van namen van "
                       "kolommen, dus wacht de afbeelding nog.",
               "right": "Op deze dag noemt je functie precies de kolom die vastzit: atemp blijft 24 uur op 0,2424, een "
                        "spreiding van 0, terwijl temp 0,24 beweegt en hum 0,53.",
               "off": "Je functie noemt {got}, en alleen atemp zit vast: een spreiding van 0, terwijl temp 0,24 beweegt "
                      "en hum 0,53.",
               "alt": "Drie kleine lijngrafieken van temp, atemp en hum over de 24 uren van 17 augustus 2012; de lijn van "
                      "atemp is vlak, en de kolommen die de functie van de leerling als vast noemt zijn gemarkeerd."}},
}

_STOPPED = object()   # what _call gives for a function that stopped with an error
PICTURE_SECONDS = 3   # what the learner's function gets for each call of a picture


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _call(make_):
    """make_() within a time limit; _STOPPED when it stops with an error or runs out of time."""
    try:
        return check.with_time_limit(make_, PICTURE_SECONDS)
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _pairs(got, hours: set, names: set):
    """got as a set of (hour, column) pairs of the batch, or _STOPPED when it is not a list of such pairs."""
    if got is _STOPPED or not isinstance(got, (list, tuple)):
        return _STOPPED
    out = set()
    for p in got:
        if not (isinstance(p, (list, tuple)) and len(p) == 2 and p[0] in hours and p[1] in names):
            return _STOPPED
        out.add((int(p[0]), p[1]))
    return out


def _names(got, names: list):
    """got as a set of names of columns, or _STOPPED when it is not a list of them."""
    if got is _STOPPED or not isinstance(got, (list, tuple)) or not all(isinstance(c, str) and c in names for c in got):
        return _STOPPED
    return set(got)


def broken_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["broken"]
    fn = _fn(seen)
    rows = [dict(r) for r in MARCH_10]
    hours = {r["hr"] for r in rows}
    got = _pairs(_call(lambda: fn([dict(r) for r in rows], dict(SCHEMA))), hours, set(SCHEMA)) if fn else _STOPPED
    if got is _STOPPED:
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    x0, x1, top, rh = 74, 326, 4, 16
    step = (x1 - x0) / 24
    parts = []
    for k, column in enumerate(SCHEMA):
        y = top + k * rh
        bad = any(c == column for _, c in got)
        parts.append(_t(0, y + 11, column, 10.5, 600 if bad else 400, "var(--ea-loss-text)" if bad else "var(--ea-ink-2)",
                        font=_CODE))
        for h in sorted(hours):
            marked = (h, column) in got
            parts.append(_rect(x0 + h * step + 1, y + 2, step - 2, rh - 4,
                               "fill:var(--ea-loss)" if marked else "fill:var(--ea-line)", 1.5))
    y = top + len(SCHEMA) * rh
    for h in (0, 6, 12, 18, 23):
        parts.append(_t(x0 + (h + 0.5) * step, y + 12, str(h), 10.5, 400, "var(--ea-ink-3)", "middle"))
    parts.append(_t((x0 + x1) / 2, y + 26, w["hour"], 11, 400, "var(--ea-ink-2)", "middle"))
    parts.append(_t(0, y + 46, w["total"].format(k=len(got)), 11.5, 600))
    want = set(facts()["march"])
    said = w["right"].format(n=len(rows)) if got == want else w["off"].format(k=len(got), n=len(want))
    body = _note(w["intro"]) + _svg(y + 54, w["alt"], parts) + _note(said)
    return yours.figure(w["title"], body, alt=w["title"])


def stuck_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["stuck"]
    fn = _fn(seen)
    got = _names(_call(lambda: fn({k: np.array(v) for k, v in AUGUST_17.items()}, list(CHANGING), LEAST)),
                 CHANGING) if fn else _STOPPED
    if got is _STOPPED:
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    x0, x1, rh = 58, 236, 46
    parts, y = [], 2
    for column in CHANGING:
        values = np.array(AUGUST_17[column])
        top, bottom = y + 4, y + rh - 8
        flagged = column in got
        parts.append(_t(0, (top + bottom) / 2 + 4, column, 11, 600 if flagged else 400,
                        "var(--ea-loss-text)" if flagged else "var(--ea-ink)", font=_CODE))
        parts.append(_rect(x0, top, x1 - x0, bottom - top, "fill:var(--ea-line)", 0))
        points = " ".join(f"{x0 + (x1 - x0) * i / 23:.1f},{bottom - (bottom - top) * v:.1f}" for i, v in enumerate(values))
        parts.append(f'<polyline points="{points}" style="fill:none;stroke:var(--ea-ink);stroke-width:1.5"/>')
        parts.append(_t(x1 + 8, (top + bottom) / 2 - 2, w["spread"].format(v=two(float(np.ptp(values)), lang)), 10.5,
                        400, "var(--ea-ink-2)"))
        parts.append(_dot(x1 + 12, (top + bottom) / 2 + 9, 3.5, "var(--ea-page)" if flagged else "var(--ea-ink)",
                          "var(--ea-loss)" if flagged else ""))
        parts.append(_t(x1 + 20, (top + bottom) / 2 + 13, w["stuck"] if flagged else w["moves"], 10.5,
                        600 if flagged else 400, "var(--ea-loss-text)" if flagged else "var(--ea-ink-2)"))
        y += rh
    for h in (0, 12, 23):
        parts.append(_t(x0 + (x1 - x0) * h / 23, y + 8, str(h), 10.5, 400, "var(--ea-ink-3)", "middle"))
    if got == set(facts()["august"]):
        said = w["right"]
    else:
        said = w["off"].format(got=", ".join(c for c in CHANGING if c in got) or "[]")
    body = _note(w["intro"]) + _svg(y + 14, w["alt"], parts) + _note(said)
    return yours.figure(w["title"], body, alt=w["title"])


# ======== the demo: two years of batches ==================================================================================

DEMO_TEXT = {
    "en": {"through": "{date}: through to training", "held": "{date}: held back",
           "bikes": "bikes per hour", "hour": "hour", "schema": "schema", "expect": "expectations",
           "values": "every value in its range", "rows": "20 to 24 rows", "twice": "no hour twice",
           "move": "temp, atemp and hum move",
           "all_values": "all {n}", "bad_values": "{k} of {col}", "n_rows": "{n} rows", "once": "each hour once",
           "doubled": "{k} twice", "all_move": "all three", "not_move": "{cols}: no",
           "go": "through to training", "stop": "held back: a person looks",
           "ok": "Every value lies in its range, the day has {n} rows, no hour comes twice, and temp, atemp and hum "
                 "move: the batch goes through to training.",
           "alt": "Two strips of the days of 2011 and 2012, with the days the gate held back marked and the chosen day "
                  "shown; under them, the bikes rented in each hour of that day, and each rule of the gate with "
                  "whether it holds."},
    "nl": {"through": "{date}: door naar de training", "held": "{date}: tegengehouden",
           "bikes": "fietsen per uur", "hour": "uur", "schema": "schema", "expect": "expectations",
           "values": "elke waarde binnen het bereik", "rows": "20 tot 24 rijen", "twice": "geen uur twee keer",
           "move": "temp, atemp en hum bewegen",
           "all_values": "alle {n}", "bad_values": "{k} van {col}", "n_rows": "{n} rijen", "once": "elk uur één keer",
           "doubled": "{k} dubbel", "all_move": "alle drie", "not_move": "{cols}: nee",
           "go": "door naar de training", "stop": "tegengehouden: een mens kijkt",
           "ok": "Elke waarde ligt binnen het bereik, de dag heeft {n} rijen, geen uur komt twee keer voor, en temp, "
                 "atemp en hum bewegen: de batch gaat door naar de training.",
           "alt": "Twee stroken met de dagen van 2011 en 2012, met een markering bij elke dag die de poort tegenhield, "
                  "en de gekozen dag aangegeven; daaronder de gehuurde fietsen in elk uur van de gekozen dag, en elke "
                  "regel van de poort met of de regel klopt."},
}

# what happened on each day the gate holds back, as the data shows it and the news of the day told it
STORIES = {
    "2011-01-18": {"en": "The rows start at 12:00: the hours of the morning are missing, and the data does not say why.",
                   "nl": "De rijen beginnen om 12.00 uur: de uren van de ochtend ontbreken, en de data zegt niet waarom."},
    "2011-01-26": {"en": "The data marks heavy snow (weathersit 4) at 16:00, and the rows stop after 17:00.",
                   "nl": "De data meldt zware sneeuw (weathersit 4) om 16.00 uur, en na 17.00 uur stoppen de rijen."},
    "2011-01-27": {"en": "The day after the snow, the rows start again only at 16:00.",
                   "nl": "Op de dag na de sneeuw beginnen de rijen pas om 16.00 uur weer."},
    "2011-02-22": {"en": "The rows start at 6:00; the data does not say why.",
                   "nl": "De rijen beginnen om 6.00 uur; de data zegt niet waarom."},
    "2011-03-10": {"en": "Light rain in 20 of the 22 hours, and a humidity of 0 in every hour: a broken sensor. The schema "
                         "sees it, and so does the expectation that hum moves.",
                   "nl": "Lichte regen in 20 van de 22 uren, en een luchtvochtigheid van 0 in elk uur: een kapotte sensor. "
                         "Het schema ziet het, en de expectation dat hum beweegt ook."},
    "2011-08-27": {"en": "Capital Bikeshare closed the system at 6 in the evening, for Hurricane Irene: the rows stop "
                         "after 17:00.",
                   "nl": "Capital Bikeshare sloot het systeem om 6 uur 's avonds, voor orkaan Irene: na 17.00 uur stoppen "
                         "de rijen."},
    "2011-08-28": {"en": "After the hurricane, the rows start again at 7:00.",
                   "nl": "Na de orkaan beginnen de rijen weer om 7.00 uur."},
    "2011-09-06": {"en": "temp and atemp do not move for 23 hours, on a day of rain: a stuck sensor, or a very even day? "
                         "The rule cannot tell; a person looks.",
                   "nl": "temp en atemp bewegen 23 uur niet, op een dag met regen: een vastgelopen sensor, of een heel "
                         "gelijkmatige dag? De regel kan het niet zeggen; een mens kijkt."},
    "2012-08-17": {"en": "atemp says 0.2424 in all 24 hours, while temp moves from 0.62 to 0.86: a stuck value, and "
                         "every one of the 24 lies in its range.",
                   "nl": "atemp zegt 0,2424 in alle 24 uren, terwijl temp van 0,62 naar 0,86 gaat: een vastgelopen "
                         "waarde, en elk van de 24 ligt binnen het bereik."},
    "2012-10-29": {"en": "Capital Bikeshare stopped at 1 a.m. for Hurricane Sandy: one row, for the first hour of the "
                         "day. With one row, no column can move either.",
                   "nl": "Capital Bikeshare stopte om 1 uur 's nachts voor orkaan Sandy: één rij, voor het eerste uur van "
                         "de dag. Met één rij kan ook geen kolom bewegen."},
    "2012-10-30": {"en": "The system ran again from the afternoon: the rows start at 13:00.",
                   "nl": "Vanaf de middag reed het systeem weer: de rijen beginnen om 13.00 uur."},
    "2012-12-29": {"en": "atemp says 0.2424 all day again, the same value as on 17 August, while temp moves from 0.20 "
                         "to 0.30.",
                   "nl": "atemp zegt de hele dag weer 0,2424, dezelfde waarde als op 17 augustus, terwijl temp van 0,20 "
                         "naar 0,30 gaat."},
}

_PASSING = ["2011-01-10", "2011-06-15", "2012-04-14", "2012-08-16"]   # four days without a problem, to compare


def demo_dates() -> list:
    return sorted(set(facts()["held"]) | set(_PASSING))


DEMO_DAYS = [day_number(d) for d in demo_dates()]


def frame(value: float, lang: str) -> tuple[str, str]:
    lang = _lang(lang)
    w = DEMO_TEXT[lang]
    date = list(days())[int(value) - 1]
    v = verdict(date)
    held = set(facts()["held"])
    title = (w["held"] if v["held"] else w["through"]).format(date=date_text(date, lang))
    parts = [_t(0, 15, title, 12, 600, "var(--ea-loss-text)" if v["held"] else "var(--ea-ink)")]
    # the two years, with the days held back
    x0, x1 = 36, 326
    every = list(days())
    for k, year in enumerate(("2011", "2012")):
        y = 26 + k * 16
        dates = [d for d in every if d.startswith(year)]
        parts.append(_t(0, y + 10, year, 10.5, 400, "var(--ea-ink-2)"))
        parts.append(_rect(x0, y + 2, x1 - x0, 9, "fill:var(--ea-line)", 1))
        for i, d in enumerate(dates):
            x = x0 + (x1 - x0) * (i + 0.5) / len(dates)
            if d in held:
                parts.append(_rect(x - 1, y, 2, 13, "fill:var(--ea-loss)", 0))
            if d == date:
                parts.append(_rect(x - 1.5, y - 3, 3, 19, "fill:var(--ea-ink)", 0))
    # the bikes in each hour of the day
    top, bottom = 76, 112
    step = (x1 - x0) / 24
    parts.append(_t(0, top + 10, w["bikes"], 10.5, 400, "var(--ea-ink-2)"))
    parts.append(_line(x0, bottom, x1, bottom, "var(--ea-ink-3)", 1))
    for r in days()[date]:
        height = max(1.5, (bottom - top) * min(r["cnt"], 1000) / 1000)
        parts.append(_rect(x0 + r["hr"] * step + 1, bottom - height, step - 2, height, "fill:var(--ea-ink-2)", 0))
    for h in (0, 6, 12, 18, 23):
        parts.append(_t(x0 + (h + 0.5) * step, bottom + 12, str(h), 10, 400, "var(--ea-ink-3)", "middle"))
    # the rules
    cols = sorted({c for _, c in v["broken"]}, key=list(SCHEMA).index)
    lines = [("schema", None, None),
             ("values", not v["broken"], w["all_values"].format(n=v["rows"] * len(SCHEMA)) if not v["broken"]
              else w["bad_values"].format(k=len(v["broken"]), col=", ".join(cols))),
             ("expect", None, None),
             ("rows", not v["short"], w["n_rows"].format(n=v["rows"])),
             ("twice", not v["twice"], w["once"] if not v["twice"] else w["doubled"].format(k=v["twice"])),
             ("move", not v["stuck"], w["all_move"] if not v["stuck"] else w["not_move"].format(cols=", ".join(v["stuck"])))]
    y = 140
    for key, ok, said in lines:
        if ok is None:
            parts.append(_t(0, y, w[key], 10.5, 600, "var(--ea-ink-2)"))
            y += 16
            continue
        parts.append(_dot(16, y - 4, 4, "var(--ea-ink)" if ok else "var(--ea-page)", "" if ok else "var(--ea-loss)"))
        parts.append(_t(26, y, w[key], 11, 400, "var(--ea-ink)"))
        parts.append(_t(326, y, said, 11, 400 if ok else 600, "var(--ea-ink-2)" if ok else "var(--ea-loss-text)",
                        "end"))
        y += 18
    parts.append(_line(0, y - 6, 330, y - 6, "var(--ea-line)", 1))
    parts.append(_t(0, y + 10, "→ " + (w["stop"] if v["held"] else w["go"]), 11.5, 600,
                    "var(--ea-loss-text)" if v["held"] else "var(--ea-ink)"))
    said = STORIES[date][lang] if date in STORIES else w["ok"].format(n=v["rows"])
    return _svg(y + 18, w["alt"], parts), said


# ======== the exercises ===================================================================================================

def _dots_left(got, text_) -> bool:
    """A `...` of the start code left in the box: an Ellipsis in the code, not three dots inside a string."""
    import ast
    try:
        tree = ast.parse(text_)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Constant) and node.value is Ellipsis for node in ast.walk(tree))


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Can you replace it with your own code?",
                nl="Er staat nog een `...` in je code. Kun je de `...` vervangen door je eigen code?")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then begins with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _shown(v):
    """{got} and {want}: a short list as Python shows it; a long one by its first three items and its length."""
    if isinstance(v, (list, tuple)) and len(v) > 4:
        head = repr(list(v[:3]))[:-1] + ", …]"
        return {"en": f"`{head}` ({len(v)} in all)", "nl": f"`{head}` ({len(v)} in totaal)"}
    text_ = repr(v)
    return f"`{text_}`" if "`" not in text_ else f"<code>{html.escape(text_)}</code>"


# exercise 1: broken
def _b(condition: str) -> str:
    return ("def broken(rows, schema):\n"
            "    found = []\n"
            "    for row in rows:\n"
            "        for column in schema:\n"
            "            low, high = schema[column]\n"
            f"            if {condition}:\n"
            '                found.append((row["hr"], column))\n'
            "    return found\n")


_B_START = _b("...")
_B = _b("row[column] < low or row[column] > high")
_B_STRICT = _b("row[column] <= low or row[column] >= high")
_B_AND = _b("row[column] < low and row[column] > high")
_B_LOW = _b("row[column] < low")


# exercise 2: stuck
def _s(condition: str) -> str:
    return ("def stuck(hours, columns, least):\n"
            "    found = []\n"
            "    for column in columns:\n"
            "        values = hours[column]\n"
            f"        if {condition}:\n"
            "            found.append(column)\n"
            "    return found\n")


_S_START = _s("...")
_S = _s("values.max() - values.min() < least")
_S_EQUAL = _s("values.max() - values.min() <= least")
_S_FLIPPED = _s("values.min() - values.max() < least")
_S_MAX = _s("values.max() < least")

_ONE = {"hr": (0, 23), "temp": (0.0, 1.0)}
_HUM = {"hr": (0, 23), "temp": (0.0, 1.0), "hum": (0.01, 1.0)}


def _schema(values: dict) -> _NamedDict:
    return _NamedDict("{" + ", ".join(f'"{k}": {v}' for k, v in values.items()) + "}", values)


def _arrays(values: dict) -> _NamedDict:
    shown = "{" + ", ".join(f'"{k}": np.array({list(v)})' for k, v in values.items()) + "}"
    return _NamedDict(shown, {k: np.array(v, dtype=float) for k, v in values.items()})


RULES = Sort(
    name="rules",
    groups={
        "schema": {"en": "Schema: each value on its own", "nl": "Schema: elke waarde apart"},
        "expectation": {"en": "Expectation: many values together", "nl": "Expectation: veel waarden samen"},
    },
    items={
        "hour": {"en": "**hr from 0 to 23.** Every hour is a whole number from 0 to 23.",
                 "nl": "**hr van 0 tot en met 23.** Elk uur is een heel getal van 0 tot en met 23."},
        "weather": {"en": "**weathersit is 1, 2, 3 or 4.** Clear, mist, light rain or snow, heavy rain or snow.",
                    "nl": "**weathersit is 1, 2, 3 of 4.** Helder, mist, lichte regen of sneeuw, zware regen of sneeuw."},
        "humidity": {"en": "**hum above 0.** Outdoors, the air is never completely dry.",
                     "nl": "**hum boven 0.** Buiten is de lucht nooit helemaal droog."},
        "empty": {"en": "**No empty temp.** Every row has a temperature.",
                  "nl": "**Geen lege temp.** Elke rij heeft een temperatuur."},
        "rows": {"en": "**20 to 24 rows.** A day has at least 20 hours and at most 24.",
                 "nl": "**20 tot 24 rijen.** Een dag heeft minstens 20 uren en hoogstens 24."},
        "twice": {"en": "**No hour twice.** Each hour of a day appears once.",
                  "nl": "**Geen uur twee keer.** Elk uur van een dag komt één keer voor."},
        "moves": {"en": "**temp moves.** Over a day, the temperature changes by at least 0.01.",
                  "nl": "**temp beweegt.** Over een dag verandert de temperatuur minstens 0,01."},
        "week": {"en": "**About as many bikes.** The bikes of a day lie within half of those of the same weekday a week before.",
                 "nl": "**Ongeveer evenveel fietsen.** De fietsen van een dag liggen binnen de helft van het aantal op dezelfde weekdag een week eerder."},
    },
    solution={"hour": "schema", "weather": "schema", "humidity": "schema", "empty": "schema",
              "rows": "expectation", "twice": "expectation", "moves": "expectation", "week": "expectation"},
    nudge={
        "hour": {"en": "Can you check an hour of 25 by looking at that one value?",
                 "nl": "Kun je een uur van 25 controleren door alleen naar de waarde zelf te kijken?"},
        "weather": {"en": "Is a weathersit of 5 wrong on its own, whatever the other rows say?",
                    "nl": "Is een weathersit van 5 op zichzelf fout, wat de andere rijen ook zeggen?"},
        "humidity": {"en": "Does a hum of 0 need any other value to be wrong?",
                     "nl": "Heeft een hum van 0 een andere waarde nodig om fout te zijn?"},
        "empty": {"en": "Is one empty cell a problem on its own, or only next to others?",
                  "nl": "Is één lege cel op zichzelf een probleem, of alleen naast andere?"},
        "rows": {"en": "Can one row tell you how many rows the day has?",
                 "nl": "Kan één rij zeggen hoeveel rijen de dag heeft?"},
        "twice": {"en": "An hour of 8 is fine on its own. When is it not?",
                  "nl": "Een uur van 8 is op zichzelf goed. Wanneer niet?"},
        "moves": {"en": "A temp of 0.2424 is fine on its own. What makes 24 of them in a row suspect?",
                  "nl": "Een temp van 0,2424 is op zichzelf goed. Wat maakt 24 ervan op een rij verdacht?"},
        "week": {"en": "How many values do you need to know the bikes of a day, and those of a week before?",
                 "nl": "Hoeveel waarden heb je nodig voor de fietsen van een dag en van een week eerder?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the eight rules again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de acht regels opnieuw uit."},
        "empty": {"en": "Put each rule under schema or expectation.",
                  "nl": "Zet elke regel onder schema of expectation."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "*{item}* belongs on the other side.",
                  "nl": "*{item}* hoort aan de andere kant."},
        "correct": {"en": "All {n} are right. A schema rule looks at one value and needs nothing else; an expectation needs the rows of the day, a column, or another day. Which of the two would have seen the atemp of 17 August?",
                    "nl": "Alle {n} kloppen. Een regel van het schema kijkt naar één waarde en heeft niets anders nodig; een expectation heeft de rijen van de dag nodig, een kolom, of een andere dag. Welke van de twee had de atemp van 17 augustus gezien?"},
    },
)

EXERCISES = {
    "broken": FunctionBox(
        name="broken",
        function="broken",
        start=_B_START,
        solution=_B,
        visible=[(MARCH_10, SCHEMA)],
        hidden=[(JUNE_5, SCHEMA), (MARCH_10, _schema({"hr": (0, 23)})),
                (_Named('[{"hr": 24, "temp": 0.5}]', [{"hr": 24, "temp": 0.5}]), _schema(_ONE)),
                (_Named('[{"hr": 5, "temp": 1.2}]', [{"hr": 5, "temp": 1.2}]), _schema(_ONE)),
                (_Named('[{"hr": 0, "temp": 1.0}, {"hr": 23, "temp": 0.0}]', [{"hr": 0, "temp": 1.0},
                                                                              {"hr": 23, "temp": 0.0}]), _schema(_ONE)),
                (_Named('[{"hr": 7, "temp": -0.1, "hum": 1.5}]', [{"hr": 7, "temp": -0.1, "hum": 1.5}]), _schema(_HUM))],
        first=[_DOTS],
        picture=broken_picture,
        show=_shown,
        mistakes=[
            Mistake("strict", wrong=_B_STRICT,
                    en="`{call}` gives {got}: a value exactly at `low` or exactly at `high` counts as broken, so the hour 0 and a temp of 1.0 break the schema, and it should give {want}. From 0 to 23: are 0 and 23 allowed?",
                    nl="`{call}` geeft {got}: een waarde precies op `low` of precies op `high` telt als gebroken, dus breken het uur 0 en een temp van 1,0 het schema, en het moet {want} zijn. Van 0 tot en met 23: mogen 0 en 23?"),
            Mistake("and", wrong=_B_AND,
                    en="`{call}` gives {got}: no value can lie below `low` and above `high` at once, so nothing is ever listed, and it should give {want}. Which word asks whether at least one of the two is true?",
                    nl="`{call}` geeft {got}: geen waarde kan tegelijk onder `low` en boven `high` liggen, dus wordt er nooit iets genoemd, en het moet {want} zijn. Welk woord vraagt of minstens één van de twee waar is?"),
            Mistake("low", wrong=_B_LOW,
                    en="`{call}` gives {got}: only values below `low` are listed, so a temp of 1.2 passes, and it should give {want}. What about a value above `high`?",
                    nl="`{call}` geeft {got}: alleen waarden onder `low` worden genoemd, dus komt een temp van 1,2 erdoor, en het moet {want} zijn. En een waarde boven `high`?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the question whether the value of this row under this column lies outside its range: below `low` or above `high`.",
                      "nl": "Vervang de `...` door de vraag of de waarde van deze rij onder deze kolom buiten het bereik ligt: onder `low` of boven `high`."},
            "arguments": {"en": "The check calls `{call}`, with the rows and the schema, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(rows, schema)`?",
                          "nl": "De check roept `{call}` aan, met de rijen en het schema, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(rows, schema)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your question list exactly the values below `low` or above `high`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Noemt je vraag precies de waarden onder `low` of boven `high`?"},
            "almost": {"en": "Right for 10 March 2011. But `{call}` gives {got}, and it should give {want}. Does your function use `low` and `high` of each column, and no fixed numbers or names?",
                       "nl": "Goed voor 10 maart 2011. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie `low` en `high` van elke kolom, en geen vaste getallen of namen?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return found`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return found`?"},
            "correct": {"en": "Right in all {n} cases. On 10 March 2011, hum is 0 in all 22 rows, below its low of 0.01, and nothing else breaks. In the picture: would you have seen it in a table of 22 rows?",
                        "nl": "Goed in alle {n} gevallen. Op 10 maart 2011 is hum 0 in alle 22 rijen, onder de ondergrens van 0,01, en verder breekt niets. In de afbeelding: had je het gezien in een tabel van 22 rijen?"},
        },
    ),
    "stuck": FunctionBox(
        name="stuck",
        function="stuck",
        start=_S_START,
        solution=_S,
        names={"np": np},
        visible=[(AUGUST_17, CHANGING, LEAST)],
        hidden=[(MARCH_10_COLUMNS, CHANGING, LEAST), (AUGUST_17, CHANGING, 0.3),
                (AUGUST_17, ["windspeed", "temp"], LEAST),
                (_arrays({"a": [1.0, 3.0], "b": [2.0, 2.0]}), ["a", "b"], 2.0),
                (_arrays({"a": [5.0]}), ["a"], 0.5), (_arrays({"a": [0.0, 1.0]}), ["a"], 0.5)],
        first=[_DOTS],
        picture=stuck_picture,
        show=_shown,
        mistakes=[
            Mistake("equal", wrong=_S_EQUAL,
                    en="`{call}` gives {got}: a column whose spread is exactly `least` counts as stuck, and it should give {want}. At least `least`: does exactly `least` count as moving?",
                    nl="`{call}` geeft {got}: een kolom met een spreiding van precies `least` telt als vast, en het moet {want} zijn. Minstens `least`: telt precies `least` als bewegen?"),
            Mistake("flipped", wrong=_S_FLIPPED,
                    en="`{call}` gives {got}: the smallest value minus the largest is never above 0, so every column counts as stuck, and it should give {want}. Which of the two goes first, so that the spread is 0 or more?",
                    nl="`{call}` geeft {got}: de kleinste waarde min de grootste is nooit boven 0, dus telt elke kolom als vast, en het moet {want} zijn. Welke van de twee komt eerst, zodat de spreiding 0 of meer is?"),
            Mistake("max", wrong=_S_MAX,
                    en="`{call}` gives {got}: the largest value alone says how high a column goes, not how much it moves, and it should give {want}. What do you take off the largest value?",
                    nl="`{call}` geeft {got}: de grootste waarde alleen zegt hoe hoog een kolom komt, niet hoeveel de kolom beweegt, en het moet {want} zijn. Wat haal je van de grootste waarde af?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the question whether the spread of `values`, its largest value minus its smallest, is less than `least`.",
                      "nl": "Vervang de `...` door de vraag of de spreiding van `values`, de grootste waarde min de kleinste, kleiner is dan `least`."},
            "arguments": {"en": "The check calls `{call}`, with the hours as columns, the names of the columns to check and the least spread, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(hours, columns, least)`?",
                          "nl": "De check roept `{call}` aan, met de uren als kolommen, de namen van de kolommen om te controleren en de kleinste spreiding, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(hours, columns, least)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your question list exactly the columns whose largest value minus smallest value is less than `least`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Noemt je vraag precies de kolommen waarvan de grootste waarde min de kleinste kleiner is dan `least`?"},
            "almost": {"en": "Right for 17 August 2012. But `{call}` gives {got}, and it should give {want}. Does your function use `values` and `least`, and no fixed numbers or names?",
                       "nl": "Goed voor 17 augustus 2012. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie `values` en `least`, en geen vaste getallen of namen?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return found`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return found`?"},
            "correct": {"en": "Right in all {n} cases. On 17 August 2012, atemp stays at 0.2424 for 24 hours: every value lies in its range, and the column as a whole is stuck. In the picture: which line is flat?",
                        "nl": "Goed in alle {n} gevallen. Op 17 augustus 2012 blijft atemp 24 uur op 0,2424: elke waarde ligt binnen het bereik, en de kolom als geheel zit vast. In de afbeelding: welke lijn is vlak?"},
        },
    ),
    "rules": RULES,
}

# The lesson is done when these are correct. "storm" is the choice question before the demo.
REQUIRED = ["broken", "stuck", "rules", "storm"]

_SORTED = dict(RULES.solution)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("broken", _B_START, "empty", None),
    ("broken", "", "empty", None),
    ("broken", _B, "correct", None),
    ("broken", _b("not low <= row[column] <= high"), "correct", None),
    ("broken", _b("row[column] > high or row[column] < low"), "correct", None),
    ("broken", "def broken(rows, schema):\n"
               "    return [(r[\"hr\"], c) for r in rows for c in schema\n"
               "            if not schema[c][0] <= r[c] <= schema[c][1]]\n", "correct", None),
    ("broken", _B_STRICT, "wrong", "strict"),
    ("broken", _B_AND, "wrong", "and"),
    ("broken", _B_LOW, "wrong", "low"),
    ("broken", _b('column == "hum"'), "almost", None),
    ("broken", _b("row[column] < lo or row[column] > high"), "error", None),
    ("broken", _b("row[column] < high"), "wrong", None),
    ("broken", _B.replace("def broken(rows, schema):", "def broken(rows):"), "error", "arguments"),
    ("broken", _B.replace("    return found\n", ""), "wrong", "no-return"),
    ("broken", _B.replace("    return found\n", "    print(found)\n"), "wrong", "prints"),
    ("broken", _B.replace("    return found\n", "    extra = ...\n    return found\n"), "wrong", "dots-left"),
    ("broken", "x = 1\n", "wrong", "no-function"),
    ("stuck", _S_START, "empty", None),
    ("stuck", "", "empty", None),
    ("stuck", _S, "correct", None),
    ("stuck", _s("np.ptp(values) < least"), "correct", None),
    ("stuck", _s("max(values) - min(values) < least"), "correct", None),
    ("stuck", _s("not values.max() - values.min() >= least"), "correct", None),
    ("stuck", _S_EQUAL, "wrong", "equal"),
    ("stuck", _S_FLIPPED, "wrong", "flipped"),
    ("stuck", _S_MAX, "wrong", "max"),
    ("stuck", _s('column == "atemp"'), "almost", None),
    ("stuck", _s("values.std() < least"), "almost", None),
    ("stuck", _s("value.max() - values.min() < least"), "error", None),
    ("stuck", _S.replace("def stuck(hours, columns, least):", "def stuck(hours, columns):"), "error", "arguments"),
    ("stuck", _S.replace("    return found\n", ""), "wrong", "no-return"),
    ("stuck", _S.replace("    return found\n", "    print(found)\n"), "wrong", "prints"),
    ("stuck", _S.replace("    return found\n", "    extra = ...\n    return found\n"), "wrong", "dots-left"),
    ("stuck", "x = 1\n", "wrong", "no-function"),
    ("rules", {}, "empty", None),
    ("rules", _SORTED, "correct", None),
    ("rules", {"hour": "schema", "rows": "expectation"}, "wrong", "unfinished"),
    ("rules", _SORTED | {"hour": "expectation"}, "wrong", "wrong:hour"),
    ("rules", _SORTED | {"weather": "expectation"}, "wrong", "wrong:weather"),
    ("rules", _SORTED | {"humidity": "expectation"}, "wrong", "wrong:humidity"),
    ("rules", _SORTED | {"empty": "expectation"}, "wrong", "wrong:empty"),
    ("rules", _SORTED | {"rows": "schema"}, "wrong", "wrong:rows"),
    ("rules", _SORTED | {"twice": "schema"}, "wrong", "wrong:twice"),
    ("rules", _SORTED | {"moves": "schema"}, "wrong", "wrong:moves"),
    ("rules", _SORTED | {"week": "schema"}, "wrong", "wrong:week"),
]
