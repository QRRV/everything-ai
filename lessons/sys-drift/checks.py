# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-drift: when the data of production moves away from the data of training. The course trains
the course's model of the project as the lesson *Project: a demand model, tracked and packed* trains h16-lr0.01, on the
training days of 2011, and lets it answer every hour of 2011 and of 2012 as the service of the project would: whole
bikes, never below 0. In 2012, the weather looks like that of 2011, and the model answers about as many bikes; but far
more people ride, and the model is three times as far off. The learner writes the line of `shares`, which counts the
share of the values in each bin of training, and the line of `psi`, the population stability index of two lists of
shares; a choice about comparing a month with the whole year, and a sort of changes into data drift, prediction drift
and label drift. The demo shows the PSI of each month of 2012 for the weather, the model's answers and the rentals,
next to how far off the model was.

The data are the hourly counts of Capital Bikeshare in Washington, D.C., in 2011 and 2012, from the Bike Sharing dataset
of Fanaee-T and Gama (2013, CC BY 4.0, see data/README.md); the changes in the sort and the pictures are the course's
own. Every word a learner's code gives back is shown through html.escape."""
import ast
import html
import importlib.util
import math

import numpy as np
import torch

from everything_ai import check, paths, yours
from everything_ai.check import FunctionBox, Mistake, Sort, kept

LESSON = "sys-drift"
LANGS = ("en", "nl")


def _course_demand():
    """The course's own copy of ops/demand.py from the lesson *Project: a demand model, tracked and packed*."""
    path = paths.LESSONS / "sys-project-tracked-model" / "project" / "ops" / "demand.py"
    spec = importlib.util.spec_from_file_location("_course_ops_demand_drift", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


demand = _course_demand()

HOURS_2011 = demand.read_hours(2011)                              # 8,645 hours of 365 days
HOURS_2012 = demand.read_hours(2012)                              # 8,734 hours of 366 days
TRAIN_HOURS = HOURS_2011[~(HOURS_2011["day"] % 5 == 4)]           # the training days of the project
DEGREES = 41                                                      # temp in the data: degrees divided by 41
KMH = 67                                                          # windspeed in the data: km/h divided by 67
TENTHS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]           # the quantiles that make ten bins
FLOOR = 0.0001                                                    # the share an empty bin gets in psi
MARCH, JULY, JANUARY = 3, 7, 1


def _lang(lang) -> str:
    return lang if lang in LANGS else "en"


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch, and a real minus sign."""
    out = f"{x:,.{digits}f}"
    out = out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out
    return out.replace("-", "−")


def psi_text(x: float, lang: str) -> str:
    """A PSI for the lesson's text: two decimals, or three under 0.01, so that 0.002 does not show as 0.00."""
    return number(x, 3 if abs(x) < 0.01 else 2, lang)


# ======== the course's model of the project ==============================================================================

def _trained():
    """The course's model of the project, trained once, the way the course's train_version trains h16-lr0.01 in the
    lesson *Project: a demand model, tracked and packed*: seed 0, 16 neurons, Adam with a learning rate of 0.01, 30
    epochs in batches of 64 hours, on the training days. On one thread, so that the numbers are the ones the lesson's
    text gives."""
    scale = demand.scaling(TRAIN_HOURS)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        x, y = demand.inputs(TRAIN_HOURS, scale), demand.targets(TRAIN_HOURS)
        torch.manual_seed(0)
        model = demand.DemandNet(16)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(30):
            demand.train_epoch(model, opt, x, y, 64)
    finally:
        torch.set_num_threads(threads)
    model.eval()
    return model, scale


def _model():
    """Kept once per Python process, for both languages: a service loads its model once, when it starts."""
    return kept((LESSON, "model"), _trained)


def _answers(hours) -> np.ndarray:
    """What the service answers for each hour of a table: the model's bikes as whole bikes, never below 0."""
    model, scale = _model()
    with torch.no_grad():
        pred = model(demand.inputs(hours, scale)).numpy().ravel()
    return np.array([max(0, round(float(p))) for p in pred], dtype=float)


def _answered() -> dict:
    a11 = _answers(HOURS_2011)
    return {"2011": a11, "2012": _answers(HOURS_2012), "train": a11[TRAIN_HOURS.index.to_numpy()]}


ANSWERED = kept((LESSON, "answered"), _answered)   # the service's answers for every hour of 2011, 2012 and training


# ======== what the lesson compares =======================================================================================

COLUMNS = ["temp", "hum", "wind", "answers", "rentals"]


def column(name: str, which: str) -> np.ndarray:
    """One column of the hours of training, of 2011 or of 2012, on the scale the lesson gives: the temperature in degrees,
    the humidity in percent, the wind in km/h, the service's answers and the bikes rented."""
    hours = {"train": TRAIN_HOURS, "2011": HOURS_2011, "2012": HOURS_2012}[which]
    if name == "temp":
        return np.round(hours["temp"].to_numpy() * DEGREES, 2)
    if name == "hum":
        return np.round(hours["hum"].to_numpy() * 100, 0)
    if name == "wind":
        return np.round(hours["windspeed"].to_numpy() * KMH, 1)
    if name == "answers":
        return ANSWERED[which]
    return hours["cnt"].to_numpy(dtype=float)


def months(which: str) -> np.ndarray:
    return {"train": TRAIN_HOURS, "2011": HOURS_2011, "2012": HOURS_2012}[which]["mnth"].to_numpy()


def edges(train) -> np.ndarray:
    """The edges of ten bins: minus infinity, the 0.1 to 0.9 quantiles of the values of training, and infinity. Each bin
    holds about a tenth of training: values that lie exactly on an edge go into the bin above it."""
    return np.concatenate([[-np.inf], np.quantile(np.asarray(train, dtype=float), TENTHS), [np.inf]])


def _defined(code: str, name: str, names: dict):
    space = dict(names)
    exec(code, space)   # noqa: S102   the course's own solution text
    return space[name]


class _Shown(np.ndarray):
    """An array that a check message writes as the lesson names it, such as C.MARCH_2012. The learner's code gets a
    plain copy."""

    def __repr__(self):
        return getattr(self, "shown", "values")

    def __deepcopy__(self, memo):
        return np.array(self)


def _named(values, name: str) -> np.ndarray:
    v = np.array(values, dtype=float).view(_Shown)
    v.shown = name
    return v


# ======== exercise 1: shares ==============================================================================================

_S_START = ("def shares(values, edges):\n"
            "    out = []\n"
            "    for i in range(len(edges) - 1):               # bin i runs from edges[i] up to edges[i + 1]\n"
            "        inside = values[values >= edges[i]]       # the values from the bottom of bin i up\n"
            "        inside = ...                              # of those, the values under the top of bin i\n"
            "        out.append(len(inside) / len(values))     # the share of the values in bin i\n"
            "    return np.array(out)\n")
_S_LINE = "        inside = ...                              #"


def _s(value: str) -> str:
    return _S_START.replace(_S_LINE, f"        inside = {value}{' ' * max(1, 33 - len(value))}#")


_S = _s("inside[inside < edges[i + 1]]")
_S_BELOW = _s("values[values < edges[i + 1]]")
_S_TOP_IN = _s("inside[inside <= edges[i + 1]]")
_S_BOTTOM = _s("inside[inside < edges[i]]")
_S_ABOVE = _s("inside[inside >= edges[i + 1]]")
_S_NAMES = {"np": np}
shares = _defined(_S, "shares", _S_NAMES)   # the course's shares

# ======== exercise 2: psi =================================================================================================

_P_START = ("def psi(train, now):\n"
            "    train = np.maximum(train, 0.0001)        # an empty bin: a tiny share instead of 0\n"
            "    now = np.maximum(now, 0.0001)\n"
            "    return ((now - train) * ...).sum()       # the part of each bin, added up\n")
_P_LINE = "    return ((now - train) * ...).sum()       #"


def _p(value: str) -> str:
    line = f"    return ((now - train) * {value}).sum()"
    return _P_START.replace(_P_LINE, f"{line}{' ' * max(1, 45 - len(line))}#")


_P = _p("np.log(now / train)")
_P_FLIPPED = _p("np.log(train / now)")
_P_NO_LOG = _p("(now / train)")
_P_LOG_DIFF = _p("np.log(now - train)")
_P_LOG_NOW = _p("np.log(now)")
_P_LOG10 = _p("np.log10(now / train)")
_P_SQUARED = _p("(now - train)")
psi = _defined(_P, "psi", {"np": np})   # the course's psi


# ======== the arrays of the lesson ========================================================================================

TEMPS_TRAIN = _named(column("temp", "train"), "C.TEMPS_TRAIN")                     # 6,912 hours of training
TEMPS_2012 = _named(column("temp", "2012"), "C.TEMPS_2012")                       # 8,734 hours of 2012
EDGES = _named(edges(TEMPS_TRAIN), "C.EDGES")
MARCH_TRAIN = _named(TEMPS_TRAIN[months("train") == MARCH], "C.MARCH_TRAIN")     # 566 hours of March in training
MARCH_2012 = _named(TEMPS_2012[months("2012") == MARCH], "C.MARCH_2012")         # 743 hours of March 2012
MARCH_EDGES = _named(edges(MARCH_TRAIN), "C.MARCH_EDGES")
JULY_2012 = _named(TEMPS_2012[months("2012") == JULY], "C.JULY_2012")
JULY_TRAIN = _named(TEMPS_TRAIN[months("train") == JULY], "C.JULY_TRAIN")
SHARES_TRAIN = _named(shares(MARCH_TRAIN, MARCH_EDGES), "C.SHARES_TRAIN")         # March in training, ten bins
SHARES_2012 = _named(shares(MARCH_2012, MARCH_EDGES), "C.SHARES_2012")            # March 2012, the same bins
_YEAR_TRAIN = _named(shares(TEMPS_TRAIN, EDGES), "C.shares(C.TEMPS_TRAIN, C.EDGES)")
_YEAR_2012 = _named(shares(TEMPS_2012, EDGES), "C.shares(C.TEMPS_2012, C.EDGES)")
_JULY_ALL = _named(shares(JULY_2012, EDGES), "C.shares(C.JULY_2012, C.EDGES)")
_SIX = _named([1, 2, 3, 4, 5, 6], "np.array([1, 2, 3, 4, 5, 6])")
_SIX_EDGES = _named([-np.inf, 2, 4, np.inf], "np.array([-np.inf, 2, 4, np.inf])")
_FIVES = _named([5, 5, 5, 7], "np.array([5, 5, 5, 7])")
_FIVES_EDGES = _named([-np.inf, 5, 6, np.inf], "np.array([-np.inf, 5, 6, np.inf])")
_HALF = _named([0.5, 0.5], "np.array([0.5, 0.5])")
_QUARTER = _named([0.25, 0.75], "np.array([0.25, 0.75])")
_EMPTY_BIN = _named([0.5, 0.5, 0.0], "np.array([0.5, 0.5, 0.0])")
_FILLED = _named([0.4, 0.4, 0.2], "np.array([0.4, 0.4, 0.2])")


def _psi_of(name: str, now: str, then: str, month: int | None = None, same: bool = True) -> float:
    """The PSI of one column: `now` against `then`, both "train", "2011" or "2012"; with `month`, the hours of that month
    only, in the bins of that month in training (`same`) or of all of training."""
    a, b = column(name, then), column(name, now)
    if month is not None:
        b = b[months(now) == month]
        if same:
            a = a[months(then) == month]
    e = edges(a)
    return float(psi(shares(a, e), shares(b, e)))


def _monthly() -> dict:
    """The PSI of each month of 2012, for each column, in the bins of the same month in training and of all of training;
    and how far off the model was per hour in each month of 2011 and of 2012."""
    out = {name: {ref: [_psi_of(name, "2012", "train", m, ref == "same") for m in range(1, 13)]
                  for ref in ("same", "all")} for name in COLUMNS}
    for year in ("2011", "2012"):
        off = np.abs(column("answers", year) - column("rentals", year))
        out[f"off_{year}"] = [float(off[months(year) == m].mean()) for m in range(1, 13)]
    return out


MONTHLY = kept((LESSON, "monthly"), _monthly)
YEAR_PSI = kept((LESSON, "year"), lambda: {name: _psi_of(name, "2012", "train") for name in COLUMNS})


def _numbers() -> dict:
    """The numbers the lesson's texts give, from the data and the course's model."""
    out = {}
    for year in ("2011", "2012"):
        a, r = column("answers", year), column("rentals", year)
        out[f"off_{year}"] = float(np.abs(a - r).mean())
        out[f"answered_{year}"] = float(a.mean())
        out[f"rented_{year}"] = float(r.mean())
        out[f"total_{year}"] = int(r.sum())
    out |= {"hours_2012": len(HOURS_2012), "train_hours": len(TRAIN_HOURS),
            "march_train": float(MARCH_TRAIN.mean()), "march_2012": float(MARCH_2012.mean()),
            "march_psi": float(psi(SHARES_TRAIN, SHARES_2012)), "march_top": float(SHARES_2012[-1]),
            "march_top_train": float(SHARES_TRAIN[-1]), "march_edge": float(MARCH_EDGES[-2]),
            "march_hours": len(MARCH_2012), "march_train_hours": len(MARCH_TRAIN),
            "july_all": float(psi(_YEAR_TRAIN, _JULY_ALL)), "july_july": _psi_of("temp", "2012", "train", JULY),
            "jan_rentals_all": MONTHLY["rentals"]["all"][JANUARY - 1],
            "jan_rentals_jan": MONTHLY["rentals"]["same"][JANUARY - 1],
            "rentals_above": sum(v > 0.25 for v in MONTHLY["rentals"]["same"]),
            "rentals_low": min(MONTHLY["rentals"]["same"]),
            "temp_above": sum(v > 0.25 for v in MONTHLY["temp"]["same"]),
            "answers_high": max(MONTHLY["answers"]["same"]),
            "registered_2011": float(HOURS_2011["registered"].mean()),
            "registered_2012": float(HOURS_2012["registered"].mean())}
    return out


NUMBERS = kept((LESSON, "numbers"), _numbers)


# ======== small pieces of pictures ========================================================================================

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem;max-width:var(--ea-measure)"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem;max-width:var(--ea-measure)"
_SVG = "width:100%;max-width:330px;height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums"
_COLOURS = {"temp": "var(--ea-data)", "hum": "var(--ea-data)", "wind": "var(--ea-data)", "answers": "var(--ea-pred)",
            "rentals": "var(--ea-data)"}


def _marked(text_: str) -> str:
    """Escaped text in which `name` is set as code."""
    return "".join(f"<code>{part}</code>" if k % 2 else part for k, part in enumerate(html.escape(text_).split("`")))


def _note(text_: str, style: str = _UI) -> str:
    return f'<div style="{style}">{_marked(text_)}</div>'


def _t_svg(x: float, y: float, text_: str, size: float = 12.5, weight: int = 400, fill: str = "var(--ea-ink)",
           anchor: str = "start") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" style="font-size:{size}px;font-weight:{weight};'
            f'fill:{fill}">{html.escape(text_)}</text>')


def _svg(height: float, alt: str, parts: list) -> str:
    return (f'<svg viewBox="0 0 330 {height:.0f}" role="img" aria-label="{html.escape(alt)}" style="{_SVG}">'
            f'{"".join(parts)}</svg>')


def _key(items: list) -> str:
    """A small legend: a swatch and a word for each kind of mark."""
    out = []
    for kind, colour, word in items:
        mark = (f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{colour};'
                f'margin-right:4px"></span>' if kind == "dot" else
                f'<span style="display:inline-block;width:10px;height:10px;background:{colour};margin-right:4px;'
                f'vertical-align:-1px"></span>' if kind == "bar" else
                f'<span style="display:inline-block;width:14px;height:0;border-top:2px {kind} {colour};'
                f'margin-right:4px;vertical-align:middle"></span>')
        out.append(f'<span style="margin-right:12px;white-space:nowrap">{mark}{html.escape(word)}</span>')
    return f'<div style="{_UI}">{"".join(out)}</div>'


def _bars(x0: float, width: float, y0: float, height: float, top: float, train, now, colour: str) -> list:
    """Ten pairs of bars in a box: the share of training in gray, the share of now in `colour`, both on a scale up to
    `top`."""
    parts = [f'<line x1="{x0:.1f}" y1="{y0 + height:.1f}" x2="{x0 + width:.1f}" y2="{y0 + height:.1f}" '
             f'style="stroke:var(--ea-ink-3);stroke-width:1"/>']
    slot = width / len(train)
    bar = slot * 0.36
    for k, (a, b) in enumerate(zip(train, now)):
        for dx, v, fill in ((slot * 0.12, a, "var(--ea-ink-3)"), (slot * 0.52, b, colour)):
            h = height * min(max(float(v), 0.0), top) / top
            parts.append(f'<rect x="{x0 + k * slot + dx:.1f}" y="{y0 + height - h:.1f}" width="{bar:.1f}" '
                         f'height="{h:.1f}" style="fill:{fill}"/>')
    return parts


MONTHS = {"en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                 "November", "December"],
          "nl": ["januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september", "oktober",
                 "november", "december"]}
INITIALS = {"en": "JFMAMJJASOND", "nl": "JFMAMJJASOND"}


# ======== the picture of the idea =========================================================================================

IDEA_TEXT = {
    "en": {"title": "2012 against training: three places to look",
           "panels": ("temperature", "answers", "rentals"),
           "kinds": ("data drift", "prediction drift", "label drift"),
           "key": ("training, 2011", "2012", "the model's answers, 2012"),
           "caption": "Each panel splits the hours of training into ten bins, with about a tenth of them in each (gray), and shows the share of the hours of 2012 in the same bins. The temperature and the model's answers stay where they were; the rentals move up to the top bins. The temperature and the answers are known at every request, the rentals only once the hour is over.",
           "alt": "Three panels of ten pairs of bars: the temperature, the model's answers and the rentals, each with the shares of training and of 2012 in the bins of training. Only the rentals of 2012 lean to the top bins: PSI {temp} for the temperature, {answers} for the answers and {rentals} for the rentals."},
    "nl": {"title": "2012 tegen de training: drie plekken om te kijken",
           "panels": ("temperatuur", "antwoorden", "verhuurd"),
           "kinds": ("data drift", "prediction drift", "label drift"),
           "key": ("training, 2011", "2012", "antwoorden van het model, 2012"),
           "caption": "Elk paneel verdeelt de uren van de training over tien bins, met ongeveer een tiende in elke bin (grijs), en toont het aandeel van de uren van 2012 in dezelfde bins. De temperatuur en de antwoorden van het model blijven ongeveer gelijk; de verhuurde fietsen schuiven op naar de bovenste bins. De temperatuur en de antwoorden zijn bij elk verzoek bekend, de verhuurde fietsen pas als het uur voorbij is.",
           "alt": "Drie panelen met tien paren staven: de temperatuur, de antwoorden van het model en de verhuurde fietsen, elk met de aandelen van de training en van 2012 in de bins van de training. Alleen de verhuurde fietsen van 2012 leunen naar de bovenste bins: PSI {temp} voor de temperatuur, {answers} voor de antwoorden en {rentals} voor de verhuurde fietsen."},
}


def idea_picture(lang: str) -> str:
    """Three panels: the temperature, the model's answers and the rentals of 2012 in the bins of training."""
    lang = _lang(lang)
    w = IDEA_TEXT[lang]
    parts = [_t_svg(4, 14, w["title"], 13, 600)]
    x0s, width, y0, height, top = (4, 115, 226), 100, 44, 62, 0.26
    for x0, name, label, kind in zip(x0s, ("temp", "answers", "rentals"), w["panels"], w["kinds"]):
        a, b = column(name, "train"), column(name, "2012")
        e = edges(a)
        parts.append(_t_svg(x0 + width / 2, y0 - 10, label, 11, 600, "var(--ea-ink)", "middle"))
        parts.extend(_bars(x0, width, y0, height, top, shares(a, e), shares(b, e), _COLOURS[name]))
        parts.append(_t_svg(x0 + width / 2, y0 + height + 16, kind, 10.5, 400, "var(--ea-ink-2)", "middle"))
        parts.append(_t_svg(x0 + width / 2, y0 + height + 31, f"PSI {psi_text(YEAR_PSI[name], lang)}", 11, 600,
                            "var(--ea-ink)", "middle"))
    alt = w["alt"].format(**{k: psi_text(YEAR_PSI[k], lang) for k in ("temp", "answers", "rentals")})
    key = [("bar", "var(--ea-ink-3)", w["key"][0]), ("bar", "var(--ea-data)", w["key"][1]),
           ("bar", "var(--ea-pred)", w["key"][2])]
    return _svg(y0 + height + 38, alt, parts) + _key(key) + _note(w["caption"])


# ======== the pictures of the learner's work ==============================================================================

_STOPPED = object()   # what _call gives for a function that stopped with an error or took too long
PICTURE_SECONDS = 3   # what the learner's function gets for all the calls of one picture together


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _call(make_):
    try:
        return check.with_time_limit(make_, PICTURE_SECONDS)
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _numbers_of(v, n: int):
    """What shares gave back as n numbers, or None for anything else."""
    try:
        out = [float(x) for x in np.asarray(v, dtype=float).ravel()]
    except (TypeError, ValueError):
        return None
    return out if len(out) == n and all(math.isfinite(x) for x in out) else None


def _number_of(v):
    """What psi gave back as one number, or None for anything else."""
    try:
        out = float(np.asarray(v, dtype=float))
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


COMPARISONS = [   # (key, the shares of training, the shares now): what the picture of psi compares
    ("year", _YEAR_TRAIN, _YEAR_2012),
    ("march", SHARES_TRAIN, SHARES_2012),
    ("july_all", _YEAR_TRAIN, _JULY_ALL),
    ("july", None, None),
    ("answers", None, None),
    ("rentals", None, None),
]


def _comparisons() -> list:
    out = []
    for key, a, b in COMPARISONS:
        if key == "july":
            e = edges(JULY_TRAIN)
            a, b = shares(JULY_TRAIN, e), shares(JULY_2012, e)
        elif key in ("answers", "rentals"):
            name = key
            e = edges(column(name, "train"))
            a, b = shares(column(name, "train"), e), shares(column(name, "2012"), e)
        out.append((key, np.array(a, dtype=float), np.array(b, dtype=float)))
    return out


PAIRS = kept((LESSON, "pairs"), _comparisons)

YOURS_TEXT = {
    "en": {"shares": {"title": "Your shares of March",
                      "intro": "Your `shares` on the temperatures of the March hours of training and of March 2012, in the bins of `C.MARCH_EDGES`. Under the bins, their edges in degrees; the dashed line is a tenth.",
                      "key": ("March in training", "March 2012"),
                      "said": "By your numbers, a share of {top} of the hours of March 2012 lies in the top bin, from {edge} degrees up, where a share of {train_top} of the March hours of training lies.",
                      "total": "Your shares of March 2012 add up to {total}, not 1: does each value land in exactly one bin?",
                      "none": "No picture yet: your `shares` does not give ten numbers for March.",
                      "alt": "Ten pairs of bars, one for each bin of the March temperatures of training: the share of the March hours of training in gray and the share of the hours of March 2012 in color, by the learner's shares."},
           "psi": {"title": "Your psi on six comparisons",
                   "intro": "Your `psi` on six pairs of shares from the lesson: a dot for each pair, with its number on the right. The scale runs up to 0.6, and an arrow means further right.",
                   "labels": {"year": "temperature, 2012 against training",
                              "march": "temperature, March 2012 against March",
                              "july_all": "temperature, July 2012 against all of training",
                              "july": "temperature, July 2012 against July",
                              "answers": "the model's answers, 2012 against training",
                              "rentals": "rentals, 2012 against training"},
                   "zones": ("calm", "watch", "drift"),
                   "said": "Above 0.25 by your numbers: {which}.",
                   "nothing": "None of the six lies above 0.25 by your numbers.",
                   "negative": "Some of your numbers lie under 0: the part of each bin should never be negative. Do you divide `now` by `train` inside the log?",
                   "none": "No picture yet: your `psi` does not give a number for every pair.",
                   "alt": "Six strips, one per comparison, each with a dot at the learner's PSI, on a scale from 0 to 0.6 with lines at 0.1 and 0.25."}},
    "nl": {"shares": {"title": "Je shares van maart",
                      "intro": "Je `shares` op de temperaturen van de maarturen van de training en van maart 2012, in de bins van `C.MARCH_EDGES`. Onder de bins staan de randen in graden; de streepjeslijn is een tiende.",
                      "key": ("maart in de training", "maart 2012"),
                      "said": "Volgens je getallen ligt een aandeel van {top} van de uren van maart 2012 in de bovenste bin, vanaf {edge} graden, waar een aandeel van {train_top} van de maarturen van de training ligt.",
                      "total": "Je aandelen van maart 2012 komen samen op {total}, niet op 1: komt elke waarde in precies één bin terecht?",
                      "none": "Nog geen afbeelding: je `shares` geeft geen tien getallen voor maart.",
                      "alt": "Tien paren staven, een voor elke bin van de maarttemperaturen van de training: het aandeel van de maarturen van de training in grijs en het aandeel van de uren van maart 2012 in kleur, volgens de shares van de leerling."},
           "psi": {"title": "Je psi op zes vergelijkingen",
                   "intro": "Je `psi` op zes paren aandelen uit de les: een stip voor elk paar, met het getal rechts. De schaal loopt tot 0,6, en een pijl betekent verder naar rechts.",
                   "labels": {"year": "temperatuur, 2012 tegen de training",
                              "march": "temperatuur, maart 2012 tegen maart",
                              "july_all": "temperatuur, juli 2012 tegen de hele training",
                              "july": "temperatuur, juli 2012 tegen juli",
                              "answers": "antwoorden van het model, 2012 tegen de training",
                              "rentals": "verhuurde fietsen, 2012 tegen de training"},
                   "zones": ("rustig", "opletten", "drift"),
                   "said": "Boven 0,25 volgens je getallen: {which}.",
                   "nothing": "Geen van de zes ligt boven 0,25 volgens je getallen.",
                   "negative": "Een paar van je getallen liggen onder 0: het deel van een bin hoort nooit negatief te zijn. Deel je in de log `now` door `train`?",
                   "none": "Nog geen afbeelding: je `psi` geeft niet voor elk paar een getal.",
                   "alt": "Zes stroken, een per vergelijking, elk met een stip op de PSI van de leerling, op een schaal van 0 tot 0,6 met lijnen op 0,1 en 0,25."}},
}
SHORT = {"en": {"year": "the whole year", "march": "March against March", "july_all": "July against all of training",
                "july": "July against July", "answers": "the answers", "rentals": "the rentals"},
         "nl": {"year": "het hele jaar", "march": "maart tegen maart", "july_all": "juli tegen de hele training",
                "july": "juli tegen juli", "answers": "de antwoorden", "rentals": "de verhuurde fietsen"}}


def _list_of(items: list, lang: str) -> str:
    word = {"en": " and ", "nl": " en "}[lang]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + word + items[-1]


def shares_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["shares"]
    fn = _fn(seen)
    stop = yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    got = (_call(lambda: (fn(np.array(MARCH_TRAIN), np.array(MARCH_EDGES)), fn(np.array(MARCH_2012), np.array(MARCH_EDGES))))
           if fn is not None else _STOPPED)
    if got is _STOPPED:
        return stop
    train, now = _numbers_of(got[0], 10), _numbers_of(got[1], 10)
    if train is None or now is None:
        return stop
    top = max(0.2, math.ceil(max(train + now) * 10) / 10)
    L, R, TOP, H = 26, 326, 8, 120
    parts = []
    for tick in np.arange(0, top + 1e-9, 0.1):
        y = TOP + H - H * tick / top
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" style="stroke:var(--ea-line);stroke-width:0.8"/>'
                     + _t_svg(L - 4, y + 4, number(tick, 1, lang), 9.5, 400, "var(--ea-ink-3)", "end"))
    y = TOP + H - H * 0.1 / top
    parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" style="stroke:var(--ea-ink);stroke-width:1;stroke-dasharray:4 3"/>')
    parts.extend(_bars(L, R - L, TOP, H, top, train, now, "var(--ea-data)"))
    slot = (R - L) / 10
    for k, e in enumerate(MARCH_EDGES[1:-1]):
        parts.append(_t_svg(L + slot * (k + 1), TOP + H + 13, number(float(e), 1, lang), 9.5, 400, "var(--ea-ink-3)", "middle"))
    svg = _svg(TOP + H + 18, w["alt"], parts)
    said = w["said"].format(top=number(now[-1], 2, lang), edge=number(float(MARCH_EDGES[-2]), 2, lang),
                            train_top=number(train[-1], 2, lang))
    total = sum(now)
    extra = _note(w["total"].format(total=number(total, 2, lang)), _LOSS_UI) if abs(total - 1) > 1e-6 else ""
    key = [("bar", "var(--ea-ink-3)", w["key"][0]), ("bar", "var(--ea-data)", w["key"][1])]
    return yours.figure(w["title"], _note(w["intro"]) + _key(key) + svg + _note(said) + extra, alt=w["title"])


def psi_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["psi"]
    fn = _fn(seen)
    stop = yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    got = _call(lambda: [fn(a.copy(), b.copy()) for _, a, b in PAIRS]) if fn is not None else _STOPPED
    if got is _STOPPED:
        return stop
    values = [_number_of(v) for v in got]
    if any(v is None for v in values):
        return stop
    L, R, ROW, TOP, MAX = 8, 306, 36, 6, 0.6
    xs = lambda v: L + (R - L) * min(max(v, 0.0), MAX) / MAX   # noqa: E731
    parts = []
    bottom = TOP + ROW * len(PAIRS)
    for k, ((key, _, _), v) in enumerate(zip(PAIRS, values)):
        y = TOP + ROW * k
        parts.append(_t_svg(L, y + 12, w["labels"][key], 10.5, 400, "var(--ea-ink-2)"))
        parts.append(_t_svg(322, y + 12, psi_text(v, lang), 10.5, 600, "var(--ea-ink)", "end"))
        parts.append(f'<line x1="{L}" y1="{y + 24}" x2="{R}" y2="{y + 24}" style="stroke:var(--ea-line);stroke-width:1"/>')
        for line in (0.1, 0.25):   # the lines of the rule of thumb, under the label, around the strip
            parts.append(f'<line x1="{xs(line):.1f}" y1="{y + 17}" x2="{xs(line):.1f}" y2="{y + 31}" '
                         f'style="stroke:var(--ea-ink-3);stroke-width:1;stroke-dasharray:2 2"/>')
        parts.append(f'<circle cx="{xs(v):.1f}" cy="{y + 24}" r="4" style="fill:var(--ea-ink)"/>')
        if v > MAX:
            parts.append(f'<path d="M{R + 4} {y + 20} L{R + 10} {y + 24} L{R + 4} {y + 28} Z" style="fill:var(--ea-ink)"/>')
    for v, word in ((0.05, w["zones"][0]), (0.175, w["zones"][1]), (0.425, w["zones"][2])):
        parts.append(_t_svg(xs(v), bottom + 6, word, 9.5, 400, "var(--ea-ink-3)", "middle"))
    for v in (0.1, 0.25):
        parts.append(_t_svg(xs(v), bottom + 18, number(v, 2 if v == 0.25 else 1, lang), 9, 400, "var(--ea-ink-3)", "middle"))
    svg = _svg(bottom + 22, w["alt"], parts)
    above = [SHORT[lang][key] for (key, _, _), v in zip(PAIRS, values) if v > 0.25]
    said = w["said"].format(which=_list_of(above, lang)) if above else w["nothing"]
    extra = _note(w["negative"], _LOSS_UI) if any(v < -1e-9 for v in values) else ""
    return yours.figure(w["title"], _note(w["intro"]) + svg + _note(said) + extra, alt=w["title"])


# ======== the demo: each month of 2012 ====================================================================================

DEMO_TEXT = {
    "en": {"title": "{column} in each month of 2012",
           "columns": {"temp": "The temperature", "hum": "The humidity", "wind": "The wind",
                       "answers": "The model's answers", "rentals": "The rentals"},
           "refs": {"same": "against the same month in training", "all": "against all of training"},
           "psi": "PSI {ref}", "off": "bikes off per hour",
           "key": ("2011", "2012"),
           "said": "Over the whole year, 2012 against training: PSI {year}. Months above 0.25: {n} of 12.",
           "notes": {"temp": "The weather of a single month differs from year to year: March 2012 was far warmer than March 2011.",
                     "hum": "The humidity moves from month to month too, without a pattern in how far off the model was.",
                     "wind": "The wind of 2012 stays in the bins of training.",
                     "answers": "The model's answers hardly move: the same kind of weather gives the same answers.",
                     "rentals": "The rentals lie above 0.25 in almost every month: in every month of 2012, the model was further off than in 2011."},
           "all_notes": {"temp": "Against all of training, a month always differs from a whole year: summer lies in the top bins, winter in the bottom ones, so every month lies above 0.25.",
                         "hum": "Against all of training, a month of 2012 is compared with a whole year of 2011, so part of what moves is the season.",
                         "wind": "The wind hardly depends on the season, so all of training gives about the same picture.",
                         "answers": "Against all of training, the answers of January and February lie in the bottom bins: the model answers fewer bikes in the cold, as it did in 2011.",
                         "rentals": "Against all of training, January and February of 2012 look calm: in the winter of 2012, about as many bikes were rented per hour as in an average hour of training. Against the same month, they lie far above 0.25."},
           "alt": "Twelve bars, the PSI of each month of 2012, with lines at 0.1 and 0.25; under them, how far off the model was per hour in each month of 2011 and of 2012."},
    "nl": {"title": "{column} in elke maand van 2012",
           "columns": {"temp": "De temperatuur", "hum": "De vochtigheid", "wind": "De wind",
                       "answers": "De antwoorden van het model", "rentals": "De verhuurde fietsen"},
           "refs": {"same": "tegen dezelfde maand in de training", "all": "tegen de hele training"},
           "psi": "PSI {ref}", "off": "fietsen ernaast per uur",
           "key": ("2011", "2012"),
           "said": "Over het hele jaar, 2012 tegen de training: PSI {year}. Maanden boven 0,25: {n} van de 12.",
           "notes": {"temp": "Het weer van één maand verschilt van jaar tot jaar: maart 2012 was veel warmer dan maart 2011.",
                     "hum": "Ook de vochtigheid beweegt van maand tot maand, zonder patroon in hoe ver het model ernaast zat.",
                     "wind": "De wind van 2012 blijft in de bins van de training.",
                     "answers": "De antwoorden van het model bewegen nauwelijks: hetzelfde soort weer geeft dezelfde antwoorden.",
                     "rentals": "De verhuurde fietsen liggen in bijna elke maand boven 0,25: in elke maand van 2012 zat het model verder ernaast dan in 2011."},
           "all_notes": {"temp": "Tegen de hele training verschilt een maand altijd van een heel jaar: de zomer ligt in de bovenste bins, de winter in de onderste, dus ligt elke maand boven 0,25.",
                         "hum": "Tegen de hele training vergelijk je een maand van 2012 met een heel jaar van 2011, dus komt een deel van de verschuiving door het seizoen.",
                         "wind": "De wind hangt nauwelijks af van het seizoen, dus geeft de hele training ongeveer hetzelfde beeld.",
                         "answers": "Tegen de hele training liggen de antwoorden van januari en februari in de onderste bins: het model antwoordt minder fietsen in de kou, net als in 2011.",
                         "rentals": "Tegen de hele training lijken januari en februari van 2012 rustig: in de winter van 2012 werden per uur ongeveer evenveel fietsen verhuurd als in een gemiddeld uur van de training. Tegen dezelfde maand liggen januari en februari ver boven 0,25."},
           "alt": "Twaalf staven, de PSI van elke maand van 2012, met lijnen op 0,1 en 0,25; daaronder hoe ver het model er per uur naast zat in elke maand van 2011 en van 2012."},
}


def _ceiling(top: float) -> tuple:
    """A round top for a scale, and its step: 0.5 or 1 for PSI up to 1, then 1, 2 or 5."""
    for ceiling, step in ((0.5, 0.1), (1.0, 0.2), (1.5, 0.5), (2.0, 0.5), (3.0, 1.0), (5.0, 1.0), (10.0, 2.0)):
        if top <= ceiling:
            return ceiling, step
    return 5.0 * math.ceil(top / 5.0), 5.0


def demo_picture(name, ref, lang: str) -> str:
    """The demo: the PSI of each month of 2012 for one column, and how far off the model was per hour in each month."""
    lang = _lang(lang)
    w = DEMO_TEXT[lang]
    name = name if name in COLUMNS else "temp"
    ref = ref if ref in ("same", "all") else "same"
    values = MONTHLY[name][ref]
    L, R, TOP, H = 30, 326, 16, 96
    ceiling, step = _ceiling(max(values))
    ys = lambda v: TOP + H - H * min(v, ceiling) / ceiling   # noqa: E731
    slot = (R - L) / 12
    parts = [_t_svg(L, 10, w["psi"].format(ref=w["refs"][ref]), 10.5, 600, "var(--ea-ink-2)")]
    for k in range(int(round(ceiling / step)) + 1):
        tick = k * step
        parts.append(f'<line x1="{L}" y1="{ys(tick):.1f}" x2="{R}" y2="{ys(tick):.1f}" style="stroke:var(--ea-line);stroke-width:0.8"/>'
                     + _t_svg(L - 4, ys(tick) + 4, number(tick, 1, lang), 9.5, 400, "var(--ea-ink-3)", "end"))
    for v in (0.1, 0.25):
        if v < ceiling:
            parts.append(f'<line x1="{L}" y1="{ys(v):.1f}" x2="{R}" y2="{ys(v):.1f}" '
                         f'style="stroke:var(--ea-ink);stroke-width:1;stroke-dasharray:4 3"/>')
    for m, v in enumerate(values):
        x = L + slot * m + slot * 0.18
        parts.append(f'<rect x="{x:.1f}" y="{ys(v):.1f}" width="{slot * 0.64:.1f}" height="{TOP + H - ys(v):.1f}" '
                     f'style="fill:{_COLOURS[name]}"/>')
    off11, off12 = MONTHLY["off_2011"], MONTHLY["off_2012"]
    T2, H2 = TOP + H + 34, 64
    top2 = 50 * math.ceil(max(off11 + off12) / 50)
    ys2 = lambda v: T2 + H2 - H2 * v / top2   # noqa: E731
    xm = lambda m: L + slot * (m + 0.5)   # noqa: E731
    parts.append(_t_svg(L, T2 - 8, w["off"], 10.5, 600, "var(--ea-ink-2)"))
    for tick in range(0, top2 + 1, 50):
        parts.append(f'<line x1="{L}" y1="{ys2(tick):.1f}" x2="{R}" y2="{ys2(tick):.1f}" style="stroke:var(--ea-line);stroke-width:0.8"/>'
                     + _t_svg(L - 4, ys2(tick) + 4, number(tick, 0, lang), 9.5, 400, "var(--ea-ink-3)", "end"))
    for series, colour in ((off11, "var(--ea-ink-3)"), (off12, "var(--ea-loss)")):
        points = " ".join(f"{xm(m):.1f},{ys2(v):.1f}" for m, v in enumerate(series))
        parts.append(f'<polyline points="{points}" style="fill:none;stroke:{colour};stroke-width:2;stroke-linejoin:round"/>')
        parts.extend(f'<circle cx="{xm(m):.1f}" cy="{ys2(v):.1f}" r="2.2" style="fill:{colour}"/>' for m, v in enumerate(series))
    for m in range(12):
        parts.append(_t_svg(xm(m), T2 + H2 + 13, INITIALS[lang][m], 9.5, 400, "var(--ea-ink-3)", "middle"))
    svg = _svg(T2 + H2 + 18, w["alt"], parts)
    said = w["said"].format(year=psi_text(YEAR_PSI[name], lang), n=sum(v > 0.25 for v in values))
    notes = _note(w["all_notes"][name] if ref == "all" else w["notes"][name])
    key = [("bar", _COLOURS[name], "PSI"), ("solid", "var(--ea-ink-3)", w["key"][0]), ("solid", "var(--ea-loss)", w["key"][1])]
    title = w["title"].format(column=w["columns"][name])
    return yours.figure(title, _key(key) + svg + _note(said) + notes, alt=w["alt"])


# ======== the exercises ===================================================================================================

def _dots_left(got, text_) -> bool:
    """A `...` of the start code left in the box: an Ellipsis in the code, not three dots inside a string."""
    try:
        tree = ast.parse(text_)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Constant) and node.value is Ellipsis for node in ast.walk(tree))


def _mask_of_values(got, text_) -> bool:
    """A mask made from `values` that picks from `inside`, which holds fewer values: NumPy stops with an IndexError."""
    import re
    return re.search(r"inside\s*\[\s*values\b", text_ or "") is not None


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Can you replace it with your own code?",
                nl="Er staat nog een `...` in je code. Kun je de `...` vervangen door je eigen code?")
_MASK = Mistake("mask-of-values", matches=_mask_of_values,
                en="Your mask compares `values`, while it picks from `inside`, which holds only the values from the bottom of the bin up. A mask must be as long as the array it picks from: which array should the comparison look at?",
                nl="Je mask vergelijkt `values`, terwijl het mask kiest uit `inside`, met alleen de waarden vanaf de onderkant van de bin. Een mask moet even lang zijn als de array waaruit het kiest: naar welke array moet de vergelijking kijken?")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then begins with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _shown(v):
    """{got} and {want}: a number with three decimals, a list of shares with three decimals each."""
    if isinstance(v, (float, int, np.floating, np.integer)) and not isinstance(v, bool):
        text_ = f"{float(v):.3f}" if math.isfinite(float(v)) else str(float(v))
    elif isinstance(v, (np.ndarray, list)) and all(isinstance(x, (float, int, np.floating, np.integer))
                                                    and not isinstance(x, bool) for x in np.asarray(v, dtype=object).ravel()):
        text_ = "[" + ", ".join(f"{float(x):.3f}" for x in np.asarray(v, dtype=float).ravel()) + "]"
    else:
        text_ = repr(v)
    return f"`{text_}`" if "`" not in text_ else f"<code>{html.escape(text_)}</code>"


KINDS = Sort(
    name="kinds",
    groups={
        "data": {"en": "Data drift: the inputs change", "nl": "Data drift: de inputs veranderen"},
        "prediction": {"en": "Prediction drift: only the answers change", "nl": "Prediction drift: alleen de antwoorden veranderen"},
        "label": {"en": "Label drift: only the rentals change", "nl": "Label drift: alleen de verhuurde fietsen veranderen"},
    },
    items={
        "march": {"en": "**A warm March.** In March 2012, the temperatures in the requests lie far above those of March in training.",
                  "nl": "**Een warme maart.** In maart 2012 liggen de temperaturen in de verzoeken ver boven de temperaturen van maart in de training."},
        "members": {"en": "**More members.** In the same weather and at the same hours, far more bikes are rented than in 2011.",
                    "nl": "**Meer leden.** Bij hetzelfde weer en op dezelfde uren worden veel meer fietsen verhuurd dan in 2011."},
        "update": {"en": "**A new version.** After an update of the service, the answers lie lower, for requests just like before.",
                   "nl": "**Een nieuwe versie.** Na een update van de service liggen de antwoorden lager, bij precies dezelfde soort verzoeken."},
        "sensor": {"en": "**A new sensor.** A new humidity sensor reads higher: most hours now fall into the top bin of training.",
                   "nl": "**Een nieuwe sensor.** Een nieuwe vochtigheidssensor meet hoger: de meeste uren vallen nu in de bovenste bin van de training."},
        "price": {"en": "**A price rise.** In the same weather and at the same hours, fewer bikes are rented than before.",
                  "nl": "**Een hogere prijs.** Bij hetzelfde weer en op dezelfde uren worden minder fietsen verhuurd dan eerst."},
        "scaling": {"en": "**Another scaling.** The service loads the scaling of another version: for the same requests, the answers move to other bins.",
                    "nl": "**Een andere schaal.** De service laadt de schaal van een andere versie: bij dezelfde verzoeken schuiven de antwoorden naar andere bins."},
    },
    solution={"march": "data", "members": "label", "update": "prediction", "sensor": "data", "price": "label",
              "scaling": "prediction"},
    nudge={
        "march": {"en": "The temperature is part of every request. What changed: what goes into the model, or only what comes out?",
                  "nl": "De temperatuur zit in elk verzoek. Wat veranderde: wat het model ingaat, of alleen wat eruit komt?"},
        "members": {"en": "The weather and the hours are the same, so the requests and the answers are too. What changed?",
                    "nl": "Het weer en de uren zijn hetzelfde, dus de verzoeken en de antwoorden ook. Wat veranderde er?"},
        "update": {"en": "The requests are just like before. Which numbers moved, those that go in or those that come out?",
                   "nl": "De verzoeken zijn precies zoals eerst. Welke getallen schoven, wat erin gaat of wat eruit komt?"},
        "sensor": {"en": "The humidity is part of every request. Where does the change start?",
                   "nl": "De vochtigheid zit in elk verzoek. Waar begint de verandering?"},
        "price": {"en": "The weather and the hours are the same, so the model answers as before. What changed?",
                  "nl": "Het weer en de uren zijn hetzelfde, dus antwoordt het model zoals eerst. Wat veranderde er?"},
        "scaling": {"en": "The requests are the same; only what the service does with them changed. Which numbers move?",
                    "nl": "De verzoeken zijn hetzelfde; alleen wat de service ermee doet veranderde. Welke getallen schuiven?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the changes again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de veranderingen opnieuw uit."},
        "empty": {"en": "Put each change with the kind of drift it is.",
                  "nl": "Zet elke verandering bij de soort drift die erbij hoort."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "*{item}* belongs with another kind of drift.",
                  "nl": "*{item}* hoort bij een andere soort drift."},
        "correct": {"en": "All {n} are right. A warm March and a new sensor change the inputs: data drift, which a check on the requests sees at once. A new version and another scaling change only the answers: prediction drift, which a check on the answers sees at once. More members and a price rise change only how many bikes are rented: label drift, which shows only once the hours are over and the rentals are counted.",
                    "nl": "Alle {n} kloppen. Een warme maart en een nieuwe sensor veranderen de inputs: data drift, en een check op de verzoeken ziet dat meteen. Een nieuwe versie en een andere schaal veranderen alleen de antwoorden: prediction drift, en een check op de antwoorden ziet dat meteen. Meer leden en een hogere prijs veranderen alleen hoeveel fietsen er verhuurd worden: label drift, en dat blijkt pas als de uren voorbij zijn en de verhuurde fietsen geteld zijn."},
    },
)

EXERCISES = {
    "shares": FunctionBox(
        name="shares",
        function="shares",
        start=_S_START,
        solution=_S,
        names=_S_NAMES,
        visible=[(MARCH_2012, MARCH_EDGES)],
        hidden=[(MARCH_TRAIN, MARCH_EDGES), (TEMPS_2012, EDGES), (_SIX, _SIX_EDGES), (_FIVES, _FIVES_EDGES)],
        first=[_DOTS, _MASK],
        show=_shown,
        picture=shares_picture,
        mistakes=[
            Mistake("below", wrong=_S_BELOW,
                    en="`{call}` gives {got}: each number is the share of all values under the top of the bin, so the shares grow from bin to bin and the last one is 1. Which values should the mask look at: all `values`, or only those in `inside`?",
                    nl="`{call}` geeft {got}: elk getal is het aandeel van alle waarden onder de bovenkant van de bin, dus groeien de aandelen van bin tot bin en is de laatste 1. Naar welke waarden moet het mask kijken: alle `values`, of alleen de waarden in `inside`?"),
            Mistake("top-in", wrong=_S_TOP_IN,
                    en="`{call}` gives {got}: a value exactly on an edge now counts in two bins, the one under the edge and the one above, so the shares add up to more than 1. Should a value equal to `edges[i + 1]` count in bin `i`?",
                    nl="`{call}` geeft {got}: een waarde precies op een rand telt nu in twee bins mee, de bin onder de rand en de bin erboven, dus komen de aandelen samen op meer dan 1. Hoort een waarde die gelijk is aan `edges[i + 1]` in bin `i`?"),
            Mistake("bottom", wrong=_S_BOTTOM,
                    en="`{call}` gives {got}: every value in `inside` already lies at or above `edges[i]`, so a mask that keeps the values under `edges[i]` keeps none. Which edge is the top of bin `i`?",
                    nl="`{call}` geeft {got}: elke waarde in `inside` ligt al op of boven `edges[i]`, dus houdt een mask met de waarden onder `edges[i]` niets over. Welke rand is de bovenkant van bin `i`?"),
            Mistake("above", wrong=_S_ABOVE,
                    en="`{call}` gives {got}: your mask keeps the values at or above the top of the bin, which belong to the bins above it. Which values of `inside` lie in bin `i`?",
                    nl="`{call}` geeft {got}: je mask houdt de waarden op of boven de bovenkant van de bin, en zulke waarden horen bij de bins erboven. Welke waarden van `inside` liggen in bin `i`?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the values of `inside` that lie under the top of bin `i`, with a mask.",
                      "nl": "Vervang de `...` door de waarden van `inside` die onder de bovenkant van bin `i` liggen, met een mask."},
            "arguments": {"en": "The check calls `{call}`, with an array of values and an array of edges, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(values, edges)`?",
                          "nl": "De check roept `{call}` aan, met een array met waarden en een array met randen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(values, edges)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your mask keep the values of `inside` under the top of bin `i`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Houdt je mask de waarden van `inside` onder de bovenkant van bin `i`?"},
            "almost": {"en": "Right for March 2012. But `{call}` gives {got}, and it should give {want}. Does your line use `inside` and `edges[i + 1]`, and not a fixed number or a fixed length?",
                       "nl": "Goed voor maart 2012. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je regel `inside` en `edges[i + 1]`, en geen vast getal of een vaste lengte?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return np.array(out)`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return np.array(out)`?"},
            "correct": {"en": "Right in all {n} cases. In March 2012, about half of the hours lie in the top bin of March in training, where a tenth of training lies. In the picture: how do the other bins compare?",
                        "nl": "Goed in alle {n} gevallen. In maart 2012 ligt ongeveer de helft van de uren in de bovenste bin van maart in de training, waar een tiende van de training ligt. In de afbeelding: hoe verhouden de andere bins zich?"},
        },
    ),
    "psi": FunctionBox(
        name="psi",
        function="psi",
        start=_P_START,
        solution=_P,
        names={"np": np},
        visible=[(SHARES_TRAIN, SHARES_2012)],
        hidden=[(_YEAR_TRAIN, _YEAR_2012), (_YEAR_TRAIN, _JULY_ALL), (SHARES_TRAIN, SHARES_TRAIN), (SHARES_2012, SHARES_TRAIN),
                (_HALF, _QUARTER), (_EMPTY_BIN, _FILLED)],
        first=[_DOTS],
        show=_shown,
        picture=psi_picture,
        mistakes=[
            Mistake("flipped", wrong=_P_FLIPPED,
                    en="`{call}` gives {got}: under 0. Where `now` lies above `train`, the log of `train / now` is negative, so each part of a bin comes out negative. Which share goes on top in the log?",
                    nl="`{call}` geeft {got}: onder 0. Waar `now` boven `train` ligt is de log van `train / now` negatief, dus wordt elk deel van een bin negatief. Welk aandeel staat boven de deelstreep in de log?"),
            Mistake("no-log", wrong=_P_NO_LOG,
                    en="`{call}` gives {got}: your line multiplies by the ratio `now / train` itself. The PSI takes the log of that ratio, which is 0 where nothing changed. Which NumPy function gives the log?",
                    nl="`{call}` geeft {got}: je regel vermenigvuldigt met de verhouding `now / train` zelf. De PSI neemt de log van de verhouding: een log is 0 waar niets veranderde. Welke NumPy-functie geeft de log?"),
            Mistake("log-diff", wrong=_P_LOG_DIFF,
                    en="`{call}` gives {got}: the log of `now - train` does not exist where the share went down, since the difference is negative there. Of which ratio does the PSI take the log?",
                    nl="`{call}` geeft {got}: de log van `now - train` bestaat niet waar het aandeel daalde, want daar is het verschil negatief. Van welke verhouding neemt de PSI de log?"),
            Mistake("log-now", wrong=_P_LOG_NOW,
                    en="`{call}` gives {got}: your log looks only at `now`. Where nothing changed, the part of a bin should be 0: which ratio of `now` and `train` has a log of 0 then?",
                    nl="`{call}` geeft {got}: je log kijkt alleen naar `now`. Waar niets veranderde hoort het deel van een bin 0 te zijn: welke verhouding van `now` en `train` heeft dan een log van 0?"),
            Mistake("log10", wrong=_P_LOG10,
                    en="`{call}` gives {got}: `np.log10` gives the log of base 10, and the PSI and its rules of thumb, 0.1 and 0.25, use the natural log. Which NumPy function gives that one?",
                    nl="`{call}` geeft {got}: `np.log10` geeft de log met grondtal 10, en de PSI en de vuistregels, 0,1 en 0,25, gebruiken de natuurlijke log. Welke NumPy-functie geeft de natuurlijke log?"),
            Mistake("squared", wrong=_P_SQUARED,
                    en="`{call}` gives {got}: your line multiplies `now - train` by itself, which makes a small change in a small bin count as little as in a large one. The PSI multiplies by the log of a ratio instead: which ratio?",
                    nl="`{call}` geeft {got}: je regel vermenigvuldigt `now - train` met zichzelf, waardoor een kleine verandering in een kleine bin even weinig telt als in een grote. De PSI vermenigvuldigt in plaats daarvan met de log van een verhouding: welke verhouding?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the log of the ratio of `now` to `train`.",
                      "nl": "Vervang de `...` door de log van de verhouding van `now` tot `train`."},
            "arguments": {"en": "The check calls `{call}`, with two arrays of shares, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(train, now)`?",
                          "nl": "De check roept `{call}` aan, met twee arrays met aandelen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(train, now)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your line multiply `now - train` by the log of `now` divided by `train`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Vermenigvuldigt je regel `now - train` met de log van `now` gedeeld door `train`?"},
            "almost": {"en": "Right for March. But `{call}` gives {got}, and it should give {want}. Does your line use `now` and `train` in every bin, even in a bin that was empty?",
                       "nl": "Goed voor maart. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je regel `now` en `train` in elke bin, ook in een bin die leeg was?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still begin its last line with `return`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Begint de laatste regel nog met `return`?"},
            "correct": {"en": "Right in all {n} cases. March 2012 against March in training gives {march}, far above 0.25; the temperature of the whole year gives {year}. In the picture: which comparisons lie above 0.25?",
                        "nl": "Goed in alle {n} gevallen. Maart 2012 tegen maart in de training geeft {march}, ver boven 0,25; de temperatuur van het hele jaar geeft {year}. In de afbeelding: welke vergelijkingen liggen boven 0,25?"},
        },
    ),
    "kinds": KINDS,
}

REQUIRED = ["shares", "psi", "season", "kinds"]


def _fill_correct_numbers():
    """The correct text of psi names the two numbers it shows, in each language."""
    for lang in LANGS:
        t = EXERCISES["psi"].texts["correct"]
        t[lang] = t[lang].replace("{march}", psi_text(NUMBERS["march_psi"], lang)).replace(
            "{year}", number(YEAR_PSI["temp"], 3, lang))


_fill_correct_numbers()

_SORTED = dict(KINDS.solution)
_OTHER = {"data": "prediction", "prediction": "label", "label": "data"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("shares", _S_START, "empty", None),
    ("shares", "", "empty", None),
    ("shares", _S, "correct", None),
    ("shares", _s("inside[edges[i + 1] > inside]"), "correct", None),
    ("shares", _S_BELOW, "wrong", "below"),
    ("shares", _S_TOP_IN, "wrong", "top-in"),
    ("shares", _S_BOTTOM, "wrong", "bottom"),
    ("shares", _S_ABOVE, "wrong", "above"),
    ("shares", _s("inside[values < edges[i + 1]]"), "wrong", "mask-of-values"),
    ("shares", _s("inside[inside < edges[i + 1]][:740]"), "almost", None),
    ("shares", _s("inside[inside < 20]"), "wrong", None),
    ("shares", _S.replace("    return np.array(out)", "    np.array(out)"), "wrong", "no-return"),
    ("shares", _S.replace("    return np.array(out)", "    print(np.array(out))"), "wrong", "prints"),
    ("shares", _S.replace("def shares(values, edges):", "def shares(values):"), "error", "arguments"),
    ("shares", _S.replace("    return np.array(out)", "    ...\n    return np.array(out)"), "wrong", "dots-left"),
    ("shares", "x = 1\n", "wrong", "no-function"),
    ("psi", _P_START, "empty", None),
    ("psi", "", "empty", None),
    ("psi", _P, "correct", None),
    ("psi", _p("(np.log(now) - np.log(train))"), "correct", None),
    ("psi", _P_FLIPPED, "wrong", "flipped"),
    ("psi", _P_NO_LOG, "wrong", "no-log"),
    ("psi", _P_LOG_DIFF, "wrong", "log-diff"),
    ("psi", _P_LOG_NOW, "wrong", "log-now"),
    ("psi", _P_LOG10, "wrong", "log10"),
    ("psi", _P_SQUARED, "wrong", "squared"),
    ("psi", _p("np.log(now / train) * (now > 0.0002)"), "almost", None),
    ("psi", _p("np.log(now / 2)"), "wrong", None),
    ("psi", _P.replace("    return ((now", "    ((now"), "wrong", "no-return"),
    ("psi", _P.replace("    return ((now - train) * np.log(now / train)).sum()", "    print(((now - train) * np.log(now / train)).sum())"),
     "wrong", "prints"),
    ("psi", _P.replace("def psi(train, now):", "def psi(train):"), "error", "arguments"),
    ("psi", _P.replace("    return ((now", "    ...\n    return ((now"), "wrong", "dots-left"),
    ("psi", "x = 1\n", "wrong", "no-function"),
    ("kinds", {}, "empty", None),
    ("kinds", _SORTED, "correct", None),
    ("kinds", {"march": "data", "members": "label"}, "wrong", "unfinished"),
] + [("kinds", _SORTED | {item: _OTHER[side]}, "wrong", f"wrong:{item}") for item, side in KINDS.solution.items()]
