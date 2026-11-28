# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-data-quality: Is the data good? Four ways to look. Four dimensions of data quality, each a
share from 0 to 1: completeness (the share of cells that are filled), correctness (the share of values that agree with
a source the team trusts), consistency (the share of values written in the agreed way) and timeliness (the share of
sources whose newest data is from today). The learner counts consistency by hand in Python and correctness with
NumPy; in the demo, a month of readings with four incidents shows that each dimension catches its own.

The weather readings of four stations, the weather service they are compared with, and the month of the demo are
made up by the course."""
import html
import math
from functools import cache

import numpy as np

from everything_ai import check, yours
from everything_ai.check import FunctionBox, Mistake, Sort

LESSON = "sys-data-quality"
LANGS = ("en", "nl")

TOLERANCE = 1.0   # a reading is right when it lies at most 1 degree from the weather service
LIMIT = 0.95      # in the demo: a score under 0.95 is where the data breaks


class _Named(list):
    """A list from the lesson: the check's call shows it by its name, and the learner's function gets a copy."""

    def __init__(self, name: str, values):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return [dict(v) if isinstance(v, dict) else v for v in self]


class _Shown(np.ndarray):
    """An array that a check message writes as the lesson names it, such as C.TEMPS. The learner's code gets a plain
    copy."""

    def __repr__(self):
        return getattr(self, "shown", "values")

    def __deepcopy__(self, memo):
        return np.array(self)


def _named(values, name: str) -> np.ndarray:
    v = np.array(values, dtype=float).view(_Shown)
    v.shown = name
    return v


# ======== the readings ====================================================================================================

STATIONS = ("north", "south", "east", "west")
CITY = {"north": "Utrecht", "south": "Houten", "east": "Zeist", "west": "Nieuwegein"}
CITIES = _Named("C.CITIES", ["Utrecht", "Houten", "Zeist", "Nieuwegein"])   # the agreed spelling of each city


def _reading(station: str, day: int, temp: float, wind: float | None, city: str | None = "") -> dict:
    row = {"station": station}
    if city is not None:
        row["city"] = city or CITY[station]
    row["day"] = day
    row["temp"] = temp
    if wind is not None:
        row["wind"] = wind
    return row


# the readings at noon of three days: a key that a reading leaves out is an empty cell
READINGS = _Named("C.READINGS", [
    _reading("north", 1, 9.5, 4.1), _reading("south", 1, 9.8, 3.6), _reading("east", 1, 9.1, 4.4),
    _reading("west", 1, 9.9, 3.9),
    _reading("north", 2, 11.2, 5.0), _reading("south", 2, 11.6, 4.8, "houten"), _reading("east", 2, 10.7, None),
    _reading("west", 2, 11.4, 4.6),
    _reading("north", 3, 12.0, 2.9, "Utrecht "), _reading("south", 3, 12.3, 3.1, "houten"),
    _reading("east", 3, 11.8, None), _reading("west", 3, 12.4, 3.0, None),
])
COLUMNS = ["city", "temp", "wind"]

# the temperature at noon on 12 days: station west, and the weather service the team trusts
_SERVICE = [8.0, 9.5, 11.0, 10.5, 12.0, 13.5, 12.5, 14.0, 15.5, 15.0, 16.5, 17.0]
_OFF = [0.3, -0.4, 1.0, -1.0, 0.2, -2.5, 0.5, 3.0, -0.1, 0.6, -3.5, 0.9]   # how far west is off on each day
TEMPS = _named([round(s + o, 1) for s, o in zip(_SERVICE, _OFF)], "C.TEMPS")
REFERENCE = _named(_SERVICE, "C.REFERENCE")


def completeness(rows, column):
    """The worked example: the share of the rows that have a value in the column."""
    filled = 0
    for row in rows:
        if column in row:
            filled = filled + 1
    return filled / len(rows)


def consistency(rows, column, agreed):
    filled = 0
    same = 0
    for row in rows:
        if column in row:
            filled = filled + 1
            if row[column] in agreed:
                same = same + 1
    return same / filled


def correctness(readings, reference, tolerance):
    diff = readings - reference
    close = abs(diff) <= tolerance
    return close.mean()


# ======== the demo: a month of readings with four incidents ================================================================

DAYS = 30
DIMENSIONS = ("completeness", "correctness", "consistency", "timeliness")
INCIDENTS = {"completeness": range(6, 10),    # the wind sensor of east is broken
             "consistency": range(13, 18),    # after an update, south writes its city as houten
             "correctness": range(20, 24),    # the thermometer of west stands in the sun: 3 degrees too warm
             "timeliness": range(26, 31)}     # the feed of north and east stops
SEED = 7


@cache
def feed() -> tuple:
    """Every hourly reading of the month, with the weather service's temperature under "service"."""
    rng = np.random.default_rng(SEED)
    rows = []
    for day in range(1, DAYS + 1):
        for hour in range(24):
            service = round(5 + 0.2 * day + 4 * math.sin(math.pi * (hour - 9) / 12), 1)
            for s in STATIONS:
                e, w, m1, m2 = rng.normal(0, 0.4), rng.normal(4, 1.5), rng.random(), rng.random()
                if s in ("north", "east") and day in INCIDENTS["timeliness"]:
                    continue
                warm = 3.0 if s == "west" and day in INCIDENTS["correctness"] else 0.0
                city = "houten" if s == "south" and day in INCIDENTS["consistency"] else CITY[s]
                row = {"station": s, "city": city, "day": day, "hour": hour, "temp": round(service + e + warm, 1),
                       "wind": round(abs(w), 1), "service": service}
                if m1 < 0.01:
                    del row["temp"]
                if m2 < 0.01 or (s == "east" and day in INCIDENTS["completeness"]):
                    del row["wind"]
                rows.append(row)
    return tuple(rows)


def scores(day: int) -> dict:
    """The four scores of the check on the evening of `day`, over the newest readings of each station."""
    newest = {}
    for r in feed():
        if r["day"] <= day:
            newest[r["station"]] = max(newest.get(r["station"], 0), r["day"])
    now = [r for r in feed() if r["day"] == newest.get(r["station"])]
    temps = [r for r in now if "temp" in r]
    return {"completeness": sum(("temp" in r) + ("wind" in r) for r in now) / (2 * len(now)),
            "correctness": sum(abs(r["temp"] - r["service"]) <= TOLERANCE for r in temps) / len(temps),
            "consistency": sum(r["city"] in CITIES for r in now) / len(now),
            "timeliness": sum(newest[s] == day for s in STATIONS) / len(STATIONS)}


@cache
def facts() -> dict:
    """The numbers the texts give."""
    days = {d: scores(d) for d in range(1, DAYS + 1)}
    below = {k: [d for d in days if days[d][k] < LIMIT] for k in DIMENSIONS}
    quiet = [d for d in days if all(d not in INCIDENTS[k] for k in DIMENSIONS)]
    mean = {d: sum(days[d].values()) / len(DIMENSIONS) for d in days}
    rows = list(READINGS)
    filled = [r for r in rows if "city" in r]
    return {
        "days": days, "below": below, "quiet_low": min(min(days[d].values()) for d in quiet),
        "mean": mean, "mean_wind": min(mean[d] for d in INCIDENTS["completeness"]),
        "low": {k: min(days[d][k] for d in INCIDENTS[k]) for k in DIMENSIONS},
        "complete": {c: completeness(rows, c) for c in COLUMNS},
        "cities": len(filled), "agreed": sum(r["city"] in CITIES for r in filled),
        "consistency": consistency(rows, "city", list(CITIES)),
        "close": int(np.sum(np.abs(np.array(TEMPS) - np.array(REFERENCE)) <= TOLERANCE)),
        "correctness": float(correctness(np.array(TEMPS), np.array(REFERENCE), TOLERANCE)),
    }


# ======== words and numbers ================================================================================================

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem;max-width:var(--ea-measure)"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem;max-width:var(--ea-measure)"
_SVG = "width:100%;max-width:330px;height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums"
_OPEN = "fill:var(--ea-page);stroke:var(--ea-ink-3);stroke-width:1"
_MARKED = "fill:var(--ea-page);stroke:var(--ea-loss);stroke-width:1.8"

NAMES = {"en": {"completeness": "completeness", "correctness": "correctness", "consistency": "consistency",
                "timeliness": "timeliness"},
         "nl": {"completeness": "volledigheid", "correctness": "juistheid", "consistency": "consistentie",
                "timeliness": "actualiteit"}}


def three(x: float, lang: str) -> str:
    """A share with three decimals: 0.727; in Dutch 0,727."""
    out = f"{x:.3f}"
    return out.replace(".", ",") if lang == "nl" else out


def two(x: float, lang: str) -> str:
    """A share with two decimals: 0.75; in Dutch 0,75."""
    out = f"{x:.2f}"
    return out.replace(".", ",") if lang == "nl" else out


def one(x: float, lang: str) -> str:
    """A number with one decimal: 1.0; in Dutch 1,0."""
    out = f"{x:.1f}"
    return out.replace(".", ",") if lang == "nl" else out


def _lang(lang) -> str:
    return lang if lang in LANGS else "en"


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


def _line(x1: float, y1: float, x2: float, y2: float, stroke: str = "var(--ea-line)", width: float = 1,
          dash: str = "") -> str:
    d = f";stroke-dasharray:{dash}" if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'style="stroke:{stroke};stroke-width:{width}{d}"/>')


def _rect(x: float, y: float, w: float, h: float, style: str, rx: float = 3) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{h:.1f}" rx="{rx}" style="{style}"/>'


def _dot(x: float, y: float, r: float, fill: str, stroke: str = "") -> str:
    s = f";stroke:{stroke};stroke-width:1.5" if stroke else ""
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" style="fill:{fill}{s}"/>'


# ======== the picture of the idea: four problems in one small table =======================================================

IDEA_TEXT = {
    "en": {"head": ["station", "city", "day", "temp"],
           "today": "today is day 30",
           "notes": {"timeliness": "3 days old", "consistency": "elsewhere: Houten",
                     "completeness": "empty cell", "correctness": "weather service: 12.0"},
           "alt": "A table of four weather readings with the columns station, city, day and temp, on day 30. Four cells "
                  "are marked, each with a dimension of data quality: the day 27 of north is three days old "
                  "(timeliness), south writes its city as houten (consistency), east has an empty temperature "
                  "(completeness), and west says 15.2 where the weather service says 12.0 (correctness)."},
    "nl": {"head": ["station", "city", "day", "temp"],
           "today": "vandaag is dag 30",
           "notes": {"timeliness": "3 dagen oud", "consistency": "elders: Houten",
                     "completeness": "lege cel", "correctness": "weerdienst: 12,0"},
           "alt": "Een tabel van vier weermetingen met de kolommen station, city, day en temp, op dag 30. Vier cellen zijn "
                  "gemarkeerd, elk met een dimensie van datakwaliteit: dag 27 van north is drie dagen oud (actualiteit), "
                  "south schrijft de stad als houten (consistentie), east heeft een lege temperatuur (volledigheid), en "
                  "west zegt 15.2 waar de weerdienst 12.0 zegt (juistheid)."},
}

# the four rows of the picture, and which cell of each row is marked for which dimension
_IDEA_ROWS = [(("north", "Utrecht", "27", "12.1"), 2, "timeliness"),
              (("south", "houten", "30", "11.8"), 1, "consistency"),
              (("east", "Zeist", "30", ""), 3, "completeness"),
              (("west", "Nieuwegein", "30", "15.2"), 3, "correctness")]
_IDEA_X = [0, 54, 128, 154, 190]   # the left edge of each column, and the right edge of the table


def idea_picture(lang: str) -> str:
    lang = _lang(lang)
    w = IDEA_TEXT[lang]
    code = "var(--ea-font-code)"
    parts = [_t(0, 12, w["today"], 11, 600, "var(--ea-ink-2)")]
    top, rh = 24, 34
    for k, head in enumerate(w["head"]):
        parts.append(_t(_IDEA_X[k] + 4, top + 14, head, 10.5, 600, "var(--ea-ink-2)", font=code))
    parts.append(_line(0, top + 20, _IDEA_X[-1], top + 20, "var(--ea-ink-3)", 1))
    y = top + 20
    for cells, marked, dim in _IDEA_ROWS:
        for k, cell in enumerate(cells):
            x, x2 = _IDEA_X[k], _IDEA_X[k + 1]
            if k == marked:
                parts.append(_rect(x + 1, y + 6, x2 - x - 3, rh - 12, _MARKED))
            parts.append(_t(x + 4, y + rh / 2 + 4, cell, 10.5, 400, "var(--ea-ink)", font=code))
        parts.append(_line(0, y + rh, _IDEA_X[-1], y + rh, "var(--ea-line)", 1))
        parts.append(_line(_IDEA_X[-1] + 2, y + rh / 2, 200, y + rh / 2, "var(--ea-loss)", 1.2))
        parts.append(_t(204, y + rh / 2 - 1, NAMES[lang][dim], 11.5, 600, "var(--ea-loss-text)"))
        parts.append(_t(204, y + rh / 2 + 12, w["notes"][dim], 10.5, 400, "var(--ea-ink-2)"))
        y += rh
    return _svg(y + 4, w["alt"], parts)


# ======== the pictures under the checks ===================================================================================

YOURS_TEXT = {
    "en": {"consistency": {
               "title": "Your consistency for the city",
               "intro": "Each reading of `C.READINGS`, with its city. The mark is what your `consistency` says about "
                        "that one reading on its own: written as agreed, or not.",
               "agreed": "as agreed", "other": "not as agreed", "empty": "empty: not counted",
               "total": "your consistency: {got}",
               "none": "Your `consistency` stops with an error, takes too long, or gives something other than a number "
                       "from 0 to 1, so the picture waits.",
               "right": "Your function is right: {a} of the {n} filled cities are written as agreed. The empty city "
                        "counts for completeness, not here.",
               "off": "Your function gives {got}, and {a} of the {n} filled cities are written as agreed: {want}.",
               "alt": "Twelve readings, one per line, each with its station, day and city, and whether the learner's "
                      "function counts the city as written in the agreed way."},
           "correctness": {
               "title": "Your correctness for 12 days",
               "intro": "Each dot is one day of `C.TEMPS`: how far the reading of west lies from the weather service. "
                        "The band runs from 1 degree below to 1 degree above. A filled dot is a day your "
                        "`correctness` counts as right on its own, an open one a day it counts as wrong.",
               "axis": "reading minus weather service, degrees", "day": "day",
               "total": "your correctness: {got}",
               "none": "Your `correctness` stops with an error, takes too long, or gives something other than a number "
                       "from 0 to 1, so the picture waits.",
               "right": "Your function is right: {a} of the 12 readings lie within 1 degree, the two at exactly 1 "
                        "degree included. The two readings far below count as wrong, like the one far above.",
               "off": "Your function gives {got}, and {a} of the 12 readings lie within 1 degree: {want}.",
               "alt": "Twelve dots, one per day, at the difference between the reading of west and the weather service, "
                      "with a band from minus 1 to plus 1 degree; filled where the learner's function counts the day "
                      "as right."}},
    "nl": {"consistency": {
               "title": "Je consistency voor de stad",
               "intro": "Elke meting van `C.READINGS`, met de stad. Het merkje is wat je `consistency` zegt over elke "
                        "meting apart: geschreven zoals afgesproken, of niet.",
               "agreed": "zoals afgesproken", "other": "niet zoals afgesproken", "empty": "leeg: telt niet mee",
               "total": "je consistency: {got}",
               "none": "Je `consistency` stopt met een fout, duurt te lang, of geeft geen getal van 0 tot en met 1, dus "
                       "wacht de afbeelding nog.",
               "right": "Je functie klopt: {a} van de {n} ingevulde steden zijn geschreven zoals afgesproken. De lege "
                        "stad telt bij de volledigheid, niet hier.",
               "off": "Je functie geeft {got}, en {a} van de {n} ingevulde steden zijn geschreven zoals afgesproken: "
                      "{want}.",
               "alt": "Twaalf metingen, één per regel, elk met station, dag en stad, en of de functie van de leerling de "
                      "stad telt als geschreven zoals afgesproken."},
           "correctness": {
               "title": "Je correctness voor 12 dagen",
               "intro": "Elke stip is één dag van `C.TEMPS`: hoe ver de meting van west van de weerdienst ligt. De band "
                        "loopt van 1 graad onder tot 1 graad boven. Een gevulde stip is een dag die je `correctness` "
                        "apart als goed telt, een open stip een dag die je functie als fout telt.",
               "axis": "meting min weerdienst, graden", "day": "dag",
               "total": "je correctness: {got}",
               "none": "Je `correctness` stopt met een fout, duurt te lang, of geeft geen getal van 0 tot en met 1, dus "
                       "wacht de afbeelding nog.",
               "right": "Je functie klopt: {a} van de 12 metingen liggen binnen 1 graad, ook de twee op precies 1 graad. "
                        "De twee metingen ver eronder tellen als fout, net als de meting ver erboven.",
               "off": "Je functie geeft {got}, en {a} van de 12 metingen liggen binnen 1 graad: {want}.",
               "alt": "Twaalf stippen, één per dag, op het verschil tussen de meting van west en de weerdienst, met een "
                      "band van min 1 tot plus 1 graad; gevuld waar de functie van de leerling de dag als goed telt."}},
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


def _share(got):
    """got as a float from 0 to 1, or _STOPPED when it is not one."""
    if got is _STOPPED or isinstance(got, (bool, np.bool_)):
        return _STOPPED
    try:
        v = float(got)
    except (TypeError, ValueError):
        return _STOPPED
    if not math.isfinite(v) or not 0 <= v <= 1:
        return _STOPPED
    return v


def consistency_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["consistency"]
    fn = _fn(seen)
    rows = list(READINGS)
    total = _share(_call(lambda: fn([dict(r) for r in rows], "city", list(CITIES)))) if fn else _STOPPED
    each = []
    for r in rows:
        if "city" not in r:
            each.append(None)
            continue
        each.append(_share(_call(lambda: fn([dict(r)], "city", list(CITIES)))) if fn else _STOPPED)
    if total is _STOPPED or any(v is _STOPPED for v in each):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    code = "var(--ea-font-code)"
    parts, y = [], 2
    for r, v in zip(rows, each):
        parts.append(_t(0, y + 12, f'{r["day"]} {r["station"]}', 11, 400, "var(--ea-ink-2)", font=code))
        if v is None:
            parts.append(_t(78, y + 12, "—", 11, 400, "var(--ea-ink-3)", font=code))
            parts.append(_t(196, y + 12, w["empty"], 11, 400, "var(--ea-ink-3)"))
        else:
            good = v >= 0.5
            parts.append(_t(78, y + 12, f'"{r["city"]}"', 11, 400, "var(--ea-ink)" if good else "var(--ea-loss-text)",
                            font=code))
            parts.append(_dot(190, y + 8, 3.5, "var(--ea-ink)" if good else "var(--ea-page)",
                              "" if good else "var(--ea-loss)"))
            parts.append(_t(198, y + 12, w["agreed"] if good else w["other"], 11, 400,
                            "var(--ea-ink-2)" if good else "var(--ea-loss-text)"))
        y += 18
    parts.append(_line(0, y + 2, 330, y + 2, "var(--ea-line)", 1))
    parts.append(_t(0, y + 18, w["total"].format(got=three(total, lang)), 11.5, 600))
    f = facts()
    if math.isclose(total, f["consistency"], abs_tol=1e-9):
        said = w["right"].format(a=f["agreed"], n=f["cities"])
    else:
        said = w["off"].format(got=three(total, lang), a=f["agreed"], n=f["cities"], want=three(f["consistency"], lang))
    body = _note(w["intro"]) + _svg(y + 26, w["alt"], parts) + _note(said)
    return yours.figure(w["title"], body, alt=w["title"])


def correctness_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["correctness"]
    fn = _fn(seen)
    temps, ref = np.array(TEMPS), np.array(REFERENCE)
    total = _share(_call(lambda: fn(temps.copy(), ref.copy(), TOLERANCE))) if fn else _STOPPED
    each = [_share(_call(lambda: fn(np.array([t]), np.array([r]), TOLERANCE))) if fn else _STOPPED
            for t, r in zip(temps, ref)]
    if total is _STOPPED or any(v is _STOPPED for v in each):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    x0, x1, top, bottom = 34, 322, 30, 190
    lo, hi = -4.0, 4.0

    def X(i):
        return x0 + (x1 - x0) * (i + 0.5) / len(temps)

    def Y(v):
        return bottom - (bottom - top) * (v - lo) / (hi - lo)

    parts = [_t(0, 14, w["axis"], 10.5, 400, "var(--ea-ink-2)")]
    parts.append(_rect(x0, Y(TOLERANCE), x1 - x0, Y(-TOLERANCE) - Y(TOLERANCE), "fill:var(--ea-line)", 0))
    for v in (-4, -2, 0, 2, 4):
        parts.append(_line(x0, Y(v), x1, Y(v), "var(--ea-ink-3)" if v == 0 else "var(--ea-line)", 1))
        parts.append(_t(x0 - 5, Y(v) + 4, f"{v:+d}" if v else "0", 10.5, 400, "var(--ea-ink-3)", "end"))
    for i, (t, r, v) in enumerate(zip(temps, ref, each)):
        good = v >= 0.5
        parts.append(_dot(X(i), Y(max(lo, min(hi, t - r))), 4, "var(--ea-ink)" if good else "var(--ea-page)",
                          "" if good else "var(--ea-loss)"))
    for i in (0, 5, 11):
        parts.append(_t(X(i), bottom + 16, str(i + 1), 10.5, 400, "var(--ea-ink-3)", "middle"))
    parts.append(_t((x0 + x1) / 2, bottom + 30, w["day"], 11, 400, "var(--ea-ink-2)", "middle"))
    parts.append(_t(0, bottom + 50, w["total"].format(got=three(total, lang)), 11.5, 600))
    f = facts()
    if math.isclose(total, f["correctness"], abs_tol=1e-9):
        said = w["right"].format(a=f["close"])
    else:
        said = w["off"].format(got=three(total, lang), a=f["close"], want=three(f["correctness"], lang))
    body = _note(w["intro"]) + _svg(bottom + 58, w["alt"], parts) + _note(said)
    return yours.figure(w["title"], body, alt=w["title"])


# ======== the demo: four scores, day by day ===============================================================================

DEMO_TEXT = {
    "en": {"broken": "Day {d}: {dim} is {v}", "fine": "Day {d}: all four at 0.95 or higher",
           "limit": "0.95", "day": "day",
           "say": "On the evening of day {d}: completeness {c}, correctness {a}, consistency {s}, timeliness {t}. {end}",
           "low": "The lowest is {dim}, under 0.95: look there first.",
           "ok": "All four are at 0.95 or higher.",
           "alt": "Four small line charts, one per dimension of data quality, over the 30 days of the month, with a line "
                  "at 0.95 and a marker at the chosen day. Each chart dips on the days of its own incident."},
    "nl": {"broken": "Dag {d}: {dim} is {v}", "fine": "Dag {d}: alle vier op 0,95 of hoger",
           "limit": "0,95", "day": "dag",
           "say": "Op de avond van dag {d}: volledigheid {c}, juistheid {a}, consistentie {s}, actualiteit {t}. {end}",
           "low": "De laagste is {dim}, onder 0,95: kijk daar eerst.",
           "ok": "Alle vier staan op 0,95 of hoger.",
           "alt": "Vier kleine lijngrafieken, één per dimensie van datakwaliteit, over de 30 dagen van de maand, met een "
                  "lijn op 0,95 en een markering op de gekozen dag. Elke grafiek zakt op de dagen van het eigen "
                  "incident."},
}

DEMO_DAYS = list(range(1, DAYS + 1))


def frame(value: float, lang: str) -> tuple[str, str]:
    lang = _lang(lang)
    w = DEMO_TEXT[lang]
    d = int(value)
    days = facts()["days"]
    now = days[d]
    low = min(DIMENSIONS, key=lambda k: now[k])
    if now[low] < LIMIT:
        title = w["broken"].format(d=d, dim=NAMES[lang][low], v=two(now[low], lang))
    else:
        title = w["fine"].format(d=d)
    parts = [_t(0, 15, title, 12, 600)]
    x0, x1, lo, hi, rh = 90, 284, 0.4, 1.0, 44

    def X(day):
        return x0 + (x1 - x0) * (day - 1) / (DAYS - 1)

    y = 26
    for k in DIMENSIONS:
        top, bottom = y + 4, y + rh - 6

        def Y(v, top=top, bottom=bottom):
            return bottom - (bottom - top) * (v - lo) / (hi - lo)

        bad = now[k] < LIMIT
        parts.append(_t(0, (top + bottom) / 2 + 4, NAMES[lang][k], 11.5, 600 if bad else 400,
                        "var(--ea-loss-text)" if bad else "var(--ea-ink)"))
        parts.append(_line(x0, bottom, x1, bottom, "var(--ea-line)", 1))
        parts.append(_line(x0, Y(LIMIT), x1, Y(LIMIT), "var(--ea-ink-3)", 1, "3 3"))
        points = " ".join(f"{X(day):.1f},{Y(days[day][k]):.1f}" for day in DEMO_DAYS)
        parts.append(f'<polyline points="{points}" style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.5"/>')
        parts.append(_dot(X(d), Y(now[k]), 4, "var(--ea-loss)" if bad else "var(--ea-ink)"))
        parts.append(_t(x1 + 8, (top + bottom) / 2 + 4, two(now[k], lang), 11.5, 600,
                        "var(--ea-loss-text)" if bad else "var(--ea-ink)"))
        y += rh
    parts.append(_line(X(d), 26, X(d), y - 4, "var(--ea-ink-3)", 1))
    for day in (1, 10, 20, 30):
        parts.append(_t(X(day), y + 10, str(day), 10.5, 400, "var(--ea-ink-3)", "middle"))
    parts.append(_t((x0 + x1) / 2, y + 24, w["day"], 11, 400, "var(--ea-ink-2)", "middle"))
    end = w["low"].format(dim=NAMES[lang][low]) if now[low] < LIMIT else w["ok"]
    say = w["say"].format(d=d, c=two(now["completeness"], lang), a=two(now["correctness"], lang),
                          s=two(now["consistency"], lang), t=two(now["timeliness"], lang), end=end)
    return _svg(y + 30, w["alt"], parts), say


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
    """{got} and {want}: a number with three decimals."""
    if isinstance(v, (float, int, np.floating, np.integer)) and not isinstance(v, (bool, np.bool_)):
        text_ = f"{float(v):.3f}" if math.isfinite(float(v)) else str(float(v))
    else:
        text_ = repr(v)
    return f"`{text_}`" if "`" not in text_ else f"<code>{html.escape(text_)}</code>"


# exercise 1: consistency
def _c(condition: str) -> str:
    return ("def consistency(rows, column, agreed):\n"
            "    filled = 0\n"
            "    same = 0\n"
            "    for row in rows:\n"
            "        if column in row:\n"
            "            filled = filled + 1\n"
            f"            if {condition}:\n"
            "                same = same + 1\n"
            "    return same / filled\n")


_C_START = _c("...")
_C = _c("row[column] in agreed")
_C_STRIP = _c("row[column].strip() in agreed")
_C_FLIPPED = _c("row[column] not in agreed")
_C_EQUALS = _c("row[column] == agreed")


# exercise 2: correctness
def _r(line: str) -> str:
    return ("def correctness(readings, reference, tolerance):\n"
            "    diff = readings - reference\n"
            f"    close = {line}\n"
            "    return close.mean()\n")


_R_START = _r("...")
_R = _r("abs(diff) <= tolerance")
_R_SIGN = _r("diff <= tolerance")
_R_STRICT = _r("abs(diff) < tolerance")
_R_FLIPPED = _r("abs(diff) > tolerance")


PROBLEMS = Sort(
    name="problems",
    groups={
        "completeness": {"en": "Completeness", "nl": "Volledigheid"},
        "correctness": {"en": "Correctness", "nl": "Juistheid"},
        "consistency": {"en": "Consistency", "nl": "Consistentie"},
        "timeliness": {"en": "Timeliness", "nl": "Actualiteit"},
    },
    items={
        "wind": {"en": "**No wind for a week.** The wind sensor of one station broke, and its readings leave the wind out.",
                 "nl": "**Een week geen wind.** De windmeter van één station is kapot, en de metingen laten de wind weg."},
        "nocity": {"en": "**Rows without a city.** A new station sends every reading without its city.",
                   "nl": "**Rijen zonder stad.** Een nieuw station stuurt elke meting zonder de stad."},
        "sun": {"en": "**Three degrees too warm.** A thermometer in the sun reads higher than the weather service.",
                "nl": "**Drie graden te warm.** Een thermometer in de zon meet hoger dan de weerdienst."},
        "wind10": {"en": "**Ten times the wind.** On a calm day, one station reports ten times the wind that the weather service measured.",
                   "nl": "**Tien keer de wind.** Op een windstille dag meldt één station tien keer de wind die de weerdienst mat."},
        "lower": {"en": "**houten and Houten.** After an update, one station writes its city in small letters.",
                  "nl": "**houten en Houten.** Na een update schrijft één station de stad met kleine letters."},
        "names": {"en": "**Two names for one station.** The table of stations says north, the readings say N.",
                  "nl": "**Twee namen voor één station.** De tabel van stations zegt north, de metingen zeggen N."},
        "silent": {"en": "**Silent since Friday.** On Monday, the newest readings of two stations are from Friday.",
                   "nl": "**Stil sinds vrijdag.** Op maandag zijn de nieuwste metingen van twee stations van vrijdag."},
        "late": {"en": "**Six hours behind.** Each reading reaches the table six hours after it was measured.",
                 "nl": "**Zes uur achter.** Elke meting komt zes uur na het meten in de tabel."},
    },
    solution={"wind": "completeness", "nocity": "completeness", "sun": "correctness", "wind10": "correctness",
              "lower": "consistency", "names": "consistency", "silent": "timeliness", "late": "timeliness"},
    nudge={
        "wind": {"en": "Is the wind there but wrong, or is it not there at all?",
                 "nl": "Staat de wind er wel maar fout, of staat de wind er helemaal niet?"},
        "nocity": {"en": "A city that is never sent: which dimension counts the cells that stay empty?",
                   "nl": "Een stad die nooit wordt gestuurd: welke dimensie telt de cellen die leeg blijven?"},
        "sun": {"en": "The cell is filled, in the usual way, and on time. What does the weather service show about it?",
                "nl": "De cel is gevuld, op de gewone manier, en op tijd. Wat laat de weerdienst ervan zien?"},
        "wind10": {"en": "Is the wind empty, written in a new way, late, or simply not true?",
                   "nl": "Is de wind leeg, op een nieuwe manier geschreven, te laat, of gewoon niet waar?"},
        "lower": {"en": "Is the city wrong, or is the same city written in a different way?",
                  "nl": "Is de stad fout, of staat dezelfde stad er op een andere manier?"},
        "names": {"en": "north and N mean the same station. What goes wrong when two tables are joined on it?",
                  "nl": "north en N zijn hetzelfde station. Wat gaat er mis als twee tabellen op het station worden samengevoegd?"},
        "silent": {"en": "The readings from Friday may be complete and right. What is wrong with them on Monday?",
                   "nl": "De metingen van vrijdag kunnen volledig en juist zijn. Wat is er op maandag mis mee?"},
        "late": {"en": "Each value is right, just not there yet when it is needed. Which dimension is about when?",
                 "nl": "Elke waarde klopt, maar is er nog niet als de waarde nodig is. Welke dimensie gaat over wanneer?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the eight problems again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de acht problemen opnieuw uit."},
        "empty": {"en": "Put each problem under the dimension that measures it.",
                  "nl": "Zet elk probleem onder de dimensie die het probleem meet."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "*{item}* belongs to another dimension.",
                  "nl": "*{item}* hoort bij een andere dimensie."},
        "correct": {"en": "All {n} are right. Each dimension asks its own question: is the value there, is it true, is it written the agreed way, and is it new enough? A table can pass three and fail the fourth.",
                    "nl": "Alle {n} kloppen. Elke dimensie stelt een eigen vraag: staat de waarde er, is de waarde waar, staat de waarde er op de afgesproken manier, en is de waarde nieuw genoeg? Een tabel kan voor drie slagen en voor de vierde zakken."},
    },
)

EXERCISES = {
    "consistency": FunctionBox(
        name="consistency",
        function="consistency",
        start=_C_START,
        solution=_C,
        visible=[(READINGS, "city", CITIES)],
        hidden=[(READINGS, "station", _Named("C.STATIONS", list(STATIONS))),
                (READINGS, "city", _Named('["Utrecht", "Houten"]', ["Utrecht", "Houten"])),
                ([{"city": "Zeist"}, {"city": "zeist"}, {"temp": 9.0}], "city", CITIES),
                ([{"city": "Houten"}], "city", CITIES), ([{"city": "houten"}], "city", CITIES),
                ([{"unit": "C"}, {"unit": "C"}, {"unit": "F"}, {"unit": "C "}], "unit", ["C"])],
        first=[_DOTS],
        picture=consistency_picture,
        show=_shown,
        mistakes=[
            Mistake("strip", wrong=_C_STRIP,
                    en="`{call}` gives {got}: `strip` takes the space off \"Utrecht \" before the question, so a spelling that is not the agreed one counts as agreed, and it should give {want}. What would the next table that joins on the city make of \"Utrecht \"?",
                    nl="`{call}` geeft {got}: `strip` haalt de spatie van \"Utrecht \" af voor de vraag, dus telt een schrijfwijze die niet de afgesproken is als afgesproken, en het moet {want} zijn. Wat maakt de volgende tabel die op de stad samenvoegt van \"Utrecht \"?"),
            Mistake("flipped", wrong=_C_FLIPPED,
                    en="`{call}` gives {got}: it counts the cities that are not written as agreed, and it should give {want}. Which cities should `same` count?",
                    nl="`{call}` geeft {got}: het telt de steden die niet zoals afgesproken zijn geschreven, en het moet {want} zijn. Welke steden moet `same` tellen?"),
            Mistake("none", wrong=_C_EQUALS,
                    en="`{call}` gives {got}: not a single value counts as agreed, as when `==` compares one city with the whole list, and it should give {want}. How do you ask whether a value is one of the values of a list?",
                    nl="`{call}` geeft {got}: geen enkele waarde telt als afgesproken, zoals wanneer `==` één stad met de hele lijst vergelijkt, en het moet {want} zijn. Hoe vraag je of een waarde een van de waarden van een lijst is?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the question whether the value of this row is one of the agreed spellings.",
                      "nl": "Vervang de `...` door de vraag of de waarde van deze rij een van de afgesproken schrijfwijzen is."},
            "arguments": {"en": "The check calls `{call}`, with the rows, the name of a column and the agreed spellings, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(rows, column, agreed)`?",
                          "nl": "De check roept `{call}` aan, met de rijen, de naam van een kolom en de afgesproken schrijfwijzen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(rows, column, agreed)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does `same` count exactly the filled values that are written as one of the agreed spellings?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Telt `same` precies de ingevulde waarden die geschreven zijn als een van de afgesproken schrijfwijzen?"},
            "almost": {"en": "Right for the cities of `C.READINGS`. But `{call}` gives {got}, and it should give {want}. Does your function use `rows`, `column` and `agreed`, and no fixed numbers or names?",
                       "nl": "Goed voor de steden van `C.READINGS`. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie `rows`, `column` en `agreed`, en geen vaste getallen of namen?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return same / filled`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return same / filled`?"},
            "correct": {"en": "Right in all {n} cases. 8 of the 11 filled cities are written as agreed: 0.727. In the picture: which three are not, and would you have seen \"Utrecht \" by eye?",
                        "nl": "Goed in alle {n} gevallen. 8 van de 11 ingevulde steden zijn geschreven zoals afgesproken: 0,727. In de afbeelding: welke drie niet, en had je \"Utrecht \" met het oog gezien?"},
        },
    ),
    "correctness": FunctionBox(
        name="correctness",
        function="correctness",
        start=_R_START,
        solution=_R,
        names={"np": np},
        visible=[(TEMPS, REFERENCE, TOLERANCE)],
        hidden=[(TEMPS, REFERENCE, 0.5), (TEMPS, REFERENCE, 3.0), (REFERENCE, REFERENCE, TOLERANCE),
                (_named([1.0, 2.0], "np.array([1.0, 2.0])"), _named([2.5, 2.0], "np.array([2.5, 2.0])"), TOLERANCE),
                (_named([5.0, 0.0, 9.0], "np.array([5.0, 0.0, 9.0])"), _named([5.0, 2.0, 7.0], "np.array([5.0, 2.0, 7.0])"),
                 2.0)],
        first=[_DOTS],
        picture=correctness_picture,
        show=_shown,
        mistakes=[
            Mistake("sign", wrong=_R_SIGN,
                    en="`{call}` gives {got}: a reading 3.5 degrees too low has a `diff` of -3.5, which is below `tolerance`, so it counts as right, and it should give {want}. How do you look at how far off a reading is, below or above?",
                    nl="`{call}` geeft {got}: bij een meting van 3,5 graad te laag is `diff` -3,5, en dat ligt onder `tolerance`, dus telt de meting als goed, en het moet {want} zijn. Hoe kijk je hoe ver een meting ernaast zit, eronder of erboven?"),
            Mistake("strict", wrong=_R_STRICT,
                    en="`{call}` gives {got}: a reading exactly `tolerance` off counts as wrong, and it should give {want}. At most 1 degree off: does exactly 1 degree count?",
                    nl="`{call}` geeft {got}: hier telt een meting die precies `tolerance` ernaast zit als fout, en het moet {want} zijn. Hoogstens 1 graad ernaast: telt precies 1 graad mee?"),
            Mistake("flipped", wrong=_R_FLIPPED,
                    en="`{call}` gives {got}: it counts the readings that are too far off, and it should give {want}. Which readings should `close` be `True` for?",
                    nl="`{call}` geeft {got}: het telt de metingen die te ver ernaast zitten, en het moet {want} zijn. Bij welke metingen moet `close` `True` zijn?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the question, for every day at once, whether the reading lies at most `tolerance` from the reference.",
                      "nl": "Vervang de `...` door de vraag, voor elke dag tegelijk, of de meting hoogstens `tolerance` van de referentie ligt."},
            "arguments": {"en": "The check calls `{call}`, with the readings, the reference and the tolerance, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(readings, reference, tolerance)`?",
                          "nl": "De check roept `{call}` aan, met de metingen, de referentie en de tolerantie, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(readings, reference, tolerance)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is `close` `True` exactly where a reading lies at most `tolerance` from the reference, below or above?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Is `close` `True` precies waar een meting hoogstens `tolerance` van de referentie ligt, eronder of erboven?"},
            "almost": {"en": "Right for the 12 days of west. But `{call}` gives {got}, and it should give {want}. Does your function use `readings`, `reference` and `tolerance`, and no fixed numbers?",
                       "nl": "Goed voor de 12 dagen van west. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie `readings`, `reference` en `tolerance`, en geen vaste getallen?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return close.mean()`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return close.mean()`?"},
            "correct": {"en": "Right in all {n} cases. 9 of the 12 readings of west lie within 1 degree of the weather service: 0.750. In the picture: which days fall out of the band, and on which side?",
                        "nl": "Goed in alle {n} gevallen. 9 van de 12 metingen van west liggen binnen 1 graad van de weerdienst: 0,750. In de afbeelding: welke dagen vallen buiten de band, en aan welke kant?"},
        },
    ),
    "problems": PROBLEMS,
}

# The lesson is done when these are correct. "first" is the choice question before the demo.
REQUIRED = ["consistency", "correctness", "problems", "first"]

_SORTED = dict(PROBLEMS.solution)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("consistency", _C_START, "empty", None),
    ("consistency", "", "empty", None),
    ("consistency", _C, "correct", None),
    ("consistency", _c("agreed.count(row[column]) > 0"), "correct", None),
    ("consistency", "def consistency(rows, column, agreed):\n"
                    "    values = [row[column] for row in rows if column in row]\n"
                    "    return sum(v in agreed for v in values) / len(values)\n", "correct", None),
    ("consistency", _C_STRIP, "wrong", "strip"),
    ("consistency", _C_FLIPPED, "wrong", "flipped"),
    ("consistency", _C_EQUALS, "wrong", "none"),
    ("consistency", _c("row[column] in CITIES"), "error", None),
    ("consistency", _c("row in agreed"), "wrong", "none"),
    ("consistency", "def consistency(rows, column, agreed):\n    return 8 / 11\n", "almost", None),
    ("consistency", _C.replace("def consistency(rows, column, agreed):", "def consistency(rows, agreed):"),
     "error", "arguments"),
    ("consistency", _C.replace("    return same / filled\n", ""), "wrong", "no-return"),
    ("consistency", _C.replace("    return same / filled\n", "    print(same / filled)\n"), "wrong", "prints"),
    ("consistency", _C.replace("    return same / filled\n", "    extra = ...\n    return same / filled\n"),
     "wrong", "dots-left"),
    ("consistency", "x = 1\n", "wrong", "no-function"),
    ("correctness", _R_START, "empty", None),
    ("correctness", "", "empty", None),
    ("correctness", _R, "correct", None),
    ("correctness", _r("np.abs(diff) <= tolerance"), "correct", None),
    ("correctness", _r("(diff <= tolerance) & (diff >= -tolerance)"), "correct", None),
    ("correctness", _r("abs(readings - reference) <= tolerance"), "correct", None),
    ("correctness", _R_SIGN, "wrong", "sign"),
    ("correctness", _R_STRICT, "wrong", "strict"),
    ("correctness", _R_FLIPPED, "wrong", "flipped"),
    ("correctness", _r("abs(diff) <= 1.0"), "almost", None),
    ("correctness", _r("abs(diff) <= tolerance / 2"), "wrong", None),
    ("correctness", _r("abs(dif) <= tolerance"), "error", None),
    ("correctness", _R.replace("def correctness(readings, reference, tolerance):", "def correctness(readings, reference):"),
     "error", "arguments"),
    ("correctness", _R.replace("    return close.mean()\n", ""), "wrong", "no-return"),
    ("correctness", _R.replace("    return close.mean()\n", "    print(close.mean())\n"), "wrong", "prints"),
    ("correctness", _R.replace("    return close.mean()\n", "    extra = ...\n    return close.mean()\n"),
     "wrong", "dots-left"),
    ("correctness", "x = 1\n", "wrong", "no-function"),
    ("problems", {}, "empty", None),
    ("problems", _SORTED, "correct", None),
    ("problems", {"wind": "completeness", "sun": "correctness"}, "wrong", "unfinished"),
    ("problems", _SORTED | {"wind": "correctness"}, "wrong", "wrong:wind"),
    ("problems", _SORTED | {"nocity": "consistency"}, "wrong", "wrong:nocity"),
    ("problems", _SORTED | {"sun": "consistency"}, "wrong", "wrong:sun"),
    ("problems", _SORTED | {"wind10": "completeness"}, "wrong", "wrong:wind10"),
    ("problems", _SORTED | {"lower": "correctness"}, "wrong", "wrong:lower"),
    ("problems", _SORTED | {"names": "completeness"}, "wrong", "wrong:names"),
    ("problems", _SORTED | {"silent": "completeness"}, "wrong", "wrong:silent"),
    ("problems", _SORTED | {"late": "correctness"}, "wrong", "wrong:late"),
]
