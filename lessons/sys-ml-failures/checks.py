# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-ml-failures: how an ML system fails, loudly and silently. The course trains the course's
model of the project as the lesson *Project: a demand model, tracked and packed* trains h16-lr0.01, and lets it answer
every hour of 2011 as the service of the project would: whole bikes, never below 0. Next to the loud failures a
software monitor sees, the year holds silent ones: on 10 March 2011 the humidity sensor sent 0 for 22 hours while it
rained, on 23 July a heat wave of 38.5 degrees kept people at home while the model expected more riders than ever, on
27 August Hurricane Irene reached Washington, and the holidays come in few days of training. The learner writes the
condition of `unseen`, which names an input of a request that lies outside what training saw, and the line of `off`,
how far off the model was per hour on a day; a choice of the check that sees a unit changed upstream first, and a sort
of failures into those a software monitor sees and those only the model's own numbers show. The demo shows the model
and the rentals per hour on six days.

The data are the hourly counts of Capital Bikeshare in Washington, D.C., in 2011 and 2012, from the Bike Sharing dataset
of Fanaee-T and Gama (2013, CC BY 4.0, see data/README.md); the requests changed by hand, the failures in the sort and
the pictures are the course's own. Every word a learner's code gives back is shown through html.escape."""
import copy
import html
import importlib.util
import math
import textwrap

import numpy as np
import pandas as pd
import torch

from everything_ai import check, paths, yours
from everything_ai.check import FunctionBox, Mistake, Sort, kept

LESSON = "sys-ml-failures"
LANGS = ("en", "nl")


def _course_demand():
    """The course's own copy of ops/demand.py from the lesson *Project: a demand model, tracked and packed*."""
    path = paths.LESSONS / "sys-project-tracked-model" / "project" / "ops" / "demand.py"
    spec = importlib.util.spec_from_file_location("_course_ops_demand_failures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


demand = _course_demand()

FEATURES = list(demand.FEATURES)                                  # the 10 features, in the order of training
FIELDS = ["hr"] + FEATURES                                        # what a request holds: 11 numbers
HOURS_2011 = demand.read_hours(2011)                              # 8,645 hours of 365 days
HOURS_2012 = demand.read_hours(2012)
TRAIN_HOURS = HOURS_2011[~(HOURS_2011["day"] % 5 == 4)]           # the training days of the project
RANGED = ["temp", "atemp", "hum", "windspeed"]                    # the inputs that `unseen` checks
KMH = 67                                                          # windspeed in the data: km/h divided by 67
DEGREES = 41                                                      # temp in the data: degrees divided by 41
WATCH_DAY = 194                                                   # Thursday 14 July 2011: a usual day
SENSOR_DAY = 68                                                   # Thursday 10 March 2011: hum 0 for 22 hours, rain
HEAT_DAY = 203                                                    # Saturday 23 July 2011: up to 38.5 degrees
IRENE_DAY = 238                                                   # Saturday 27 August 2011: Hurricane Irene
JULY_4 = 184                                                      # Monday 4 July 2011: Independence Day
AFTER_THANKSGIVING = 328                                          # Friday 25 November 2011: a working day in the data
RAINY_HUM = 0.88                                                  # the median humidity of the rainy hours of 2011


def _lang(lang) -> str:
    return lang if lang in LANGS else "en"


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch, and a real minus sign."""
    out = f"{x:,.{digits}f}"
    out = out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out
    return out.replace("-", "−")


def date(day: int) -> str:
    """The date of a day of 2011 as the data writes it: 2011-07-23."""
    return str(HOURS_2011[HOURS_2011["day"] == day]["dteday"].iloc[0])


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


_model()   # trained when the lesson loads, so that no check has to wait for it


def _answers(hours) -> list:
    """What the service answers for each hour of a table: the model's bikes as whole bikes, never below 0."""
    model, scale = _model()
    with torch.no_grad():
        pred = model(demand.inputs(hours, scale)).numpy().ravel()
    return [max(0, round(float(p))) for p in pred]


# ======== the requests of Exercise 1 ======================================================================================

_WHOLE = {"hr", "weekday", "workingday", "holiday", "weathersit", "season", "mnth"}


class Named(dict):
    """A dict that the check's call shows by its name, such as C.LOW or C.NEW_APP; the learner's function gets a
    plain copy."""

    def __init__(self, name: str, values: dict):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(dict(self), memo)


class NamedList(list):
    """A list that the check's call shows by its name, such as C.ANSWERED["2011-07-23"]."""

    def __init__(self, name: str, values):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return list(self)


def _row(hours, day: int, hr: int) -> dict:
    row = hours[(hours["day"] == day) & (hours["hr"] == hr)].iloc[0]
    return {name: int(row[name]) if name in _WHOLE else round(float(row[name]), 4) for name in FIELDS}


def _changed(name: str, base: dict, **values) -> Named:
    return Named(name, {**base, **values})


LOW = Named("C.LOW", {f: round(float(TRAIN_HOURS[f].min()), 4) for f in RANGED})     # the lowest value in training
HIGH = Named("C.HIGH", {f: round(float(TRAIN_HOURS[f].max()), 4) for f in RANGED})   # the highest value in training

_MORNING = _row(HOURS_2011, WATCH_DAY, 8)
MORNING = Named("C.MORNING", _MORNING)                                                  # 14 July 2011, 8:00
NEW_APP = _changed("C.NEW_APP", _row(HOURS_2011, WATCH_DAY, 16),                        # the new app: temp in degrees
                   temp=round(_row(HOURS_2011, WATCH_DAY, 16)["temp"] * DEGREES, 1))
WINDY = _changed("C.WINDY", _MORNING, windspeed=round(_MORNING["windspeed"] * KMH, 1))   # windspeed in km/h
NO_READING = _changed("C.NO_READING", _MORNING, hum=-1)                                 # a sensor with no reading
FROZEN = _changed("C.FROZEN", _MORNING, temp=0.0)                                      # colder than any hour of training
HEAT = Named("C.HEAT", _row(HOURS_2011, HEAT_DAY, 15))                                  # 23 July 2011, 15:00
HOT_DAY_2012 = 188                                                                      # Saturday 7 July 2012
HOT = Named("C.HOT", _row(HOURS_2012, HOT_DAY_2012, 16))                                # 2012's hottest hour
CALM = Named("C.CALM", _row(HOURS_2011, 5, 8))                                          # 6 January 2011, 8:00: no wind
FOG_DAY, FOG_HR = 35, 8                                                                 # 5 February 2011, 8:00: hum 1
FOG = Named("C.FOG", _row(HOURS_2011, FOG_DAY, FOG_HR))


def answer(request) -> int:
    """What the service answers for one request: the model's bikes, whole, never below 0."""
    return _answers(pd.DataFrame([dict(request)]))[0]


SENSOR = Named("C.SENSOR", _row(HOURS_2011, SENSOR_DAY, 8))                             # 10 March 2011, 8:00: hum 0


def rented(day: int, hr: int, hours=HOURS_2011) -> int:
    return int(hours[(hours["day"] == day) & (hours["hr"] == hr)]["cnt"].iloc[0])


def _defined(code: str, name: str, names: dict):
    space = dict(names)
    exec(code, space)   # noqa: S102   the course's own solution text
    return space[name]


# ======== exercise 1: unseen =============================================================================================

_U_START = ("def unseen(request, low, high):\n"
            "    for name in low:                       # temp, atemp, hum and windspeed\n"
            "        if ...:\n"
            "            return name                    # the first input outside what training saw\n"
            "    return None                            # every input lies within training\n")


def _u(condition: str) -> str:
    return _U_START.replace("        if ...:\n", f"        if {condition}:\n")


_U = _u("request[name] < low[name] or request[name] > high[name]")
_U_LOW = _u("request[name] < low[name]")
_U_HIGH = _u("request[name] > high[name]")
_U_AND = _u("request[name] < low[name] and request[name] > high[name]")
_U_SWAPPED = _u("request[name] > low[name] or request[name] < high[name]")
_U_EDGES = _u("request[name] <= low[name] or request[name] >= high[name]")
_U_EDGE_LOW = _u("request[name] <= low[name] or request[name] > high[name]")
_U_EDGE_HIGH = _u("request[name] < low[name] or request[name] >= high[name]")
unseen = _defined(_U, "unseen", {})   # the course's unseen

# ======== exercise 2: off ================================================================================================

_O_START = ("def off(predicted, rented):\n"
            "    total = 0\n"
            "    for h in range(len(rented)):\n"
            "        total = total + ...                # how far off the model was in hour h, up or down\n"
            "    return total / len(rented)             # how far off per hour, on average\n")


def _o(term: str) -> str:
    return _O_START.replace("        total = total + ...                #",
                            f"        total = total + {term}{' ' * max(1, 19 - len(term))}#")


_O = _o("abs(predicted[h] - rented[h])")
_O_SIGNED = _o("predicted[h] - rented[h]")
_O_UNDER = _o("rented[h] - predicted[h]")
_O_SQUARED = _o("(predicted[h] - rented[h]) ** 2")
_O_TWICE = _o("abs(predicted[h] - rented[h]) / len(rented)")
off = _defined(_O, "off", {})   # the course's off


def _year():
    """Every day of 2011: the service's answers and the rentals, hour by hour, as the data has the hours."""
    answers = _answers(HOURS_2011)
    out = {}
    for k, (d, hours) in enumerate(HOURS_2011.groupby("dteday", sort=True)):
        rows = hours.index.to_list()
        out[str(d)] = ([answers[i] for i in rows], hours["cnt"].astype(int).to_list(), hours["hr"].astype(int).to_list())
    return out


YEAR = kept((LESSON, "year"), _year)   # date -> (answers, rentals, hours)
DATES = sorted(YEAR)
ANSWERED = {d: NamedList(f'C.ANSWERED["{d}"]', YEAR[d][0]) for d in DATES}
RENTED = {d: NamedList(f'C.RENTED["{d}"]', YEAR[d][1]) for d in DATES}
OFF_BY_DAY = kept((LESSON, "off"), lambda: {d: off(YEAR[d][0], YEAR[d][1]) for d in DATES})
USUAL = float(np.median(list(OFF_BY_DAY.values())))   # half of the days of 2011 lie under this


def day_numbers(day: int) -> dict:
    """The answers and the rentals of one day of 2011 summed, and how far off the model was per hour."""
    a, r, hrs = YEAR[date(day)]
    return {"answered": sum(a), "rented": sum(r), "off": off(a, r), "hours": len(hrs)}


def _numbers() -> dict:
    """The numbers the lesson's texts give, from the data and the course's model."""
    sensor_fixed = HOURS_2011[HOURS_2011["day"] == SENSOR_DAY].copy()
    sensor_fixed.loc[sensor_fixed["hum"] == 0, "hum"] = RAINY_HUM
    windy = HOURS_2011[HOURS_2011["day"] == WATCH_DAY].copy()
    windy["windspeed"] = windy["windspeed"] * KMH
    a_windy = _answers(windy)
    return {"sensor_fixed": sum(_answers(sensor_fixed)), "windy": sum(a_windy),
            "windy_off": off(a_windy, windy["cnt"].astype(int).to_list()),
            "hot_hours": int((TRAIN_HOURS["temp"] >= 37 / DEGREES).sum()), "train_hours": len(TRAIN_HOURS),
            "holiday_days": int(TRAIN_HOURS[TRAIN_HOURS["holiday"] == 1]["day"].nunique()),
            "sensor_hours": int((HOURS_2011[HOURS_2011["day"] == SENSOR_DAY]["hum"] == 0).sum()),
            "above_twice": sum(v > 2 * USUAL for v in OFF_BY_DAY.values())}


NUMBERS = kept((LESSON, "numbers"), _numbers)


# ======== small pieces of pictures ========================================================================================

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem;max-width:var(--ea-measure)"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem;max-width:var(--ea-measure)"
_SVG = "width:100%;max-width:330px;height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums"


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


def _list_of(items: list, lang: str) -> str:
    word = {"en": " and ", "nl": " en "}[lang]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + word + items[-1]


MONTHS = {"en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                 "November", "December"],
          "nl": ["januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september", "oktober",
                 "november", "december"]}


def said_date(d: str, lang: str) -> str:
    """A date of the data in words: 23 July, or 23 juli."""
    return f"{int(d[8:10])} {MONTHS[lang][int(d[5:7]) - 1]}"


# ======== the picture of the idea =========================================================================================

IDEA_TEXT = {
    "en": {"title": "Two requests, two ways to fail",
           "loud": "Loud: the service breaks", "silent": "Silent: the service answers, wrongly",
           "request": "request", "service": "service + model",
           "loud_out": "status 500", "loud_seen": "a software monitor sees it: errors, latency",
           "silent_in": "hum 0", "silent_out": "status 200 · {ms} ms · {bikes} bikes",
           "silent_seen": "only the model's numbers show it: {rented} were rented",
           "caption": "Both requests are for 8:00 on 10 March 2011. The first meets a server that ran out of memory. The second meets a healthy server, with the humidity sensor's 0 in it, while it rained.",
           "alt": "Two rows. In the first, a request reaches the service and model, which breaks and answers with status 500: a software monitor sees it. In the second, a request with humidity 0 reaches the service and model, which answers with status 200 and {bikes} bikes, where {rented} were rented: only the model's own numbers show it."},
    "nl": {"title": "Twee verzoeken, twee manieren om te falen",
           "loud": "Luid: de service gaat stuk", "silent": "Stil: de service antwoordt, verkeerd",
           "request": "verzoek", "service": "service + model",
           "loud_out": "status 500", "loud_seen": "een softwaremonitor ziet het: fouten, latency",
           "silent_in": "hum 0", "silent_out": "status 200 · {ms} ms · {bikes} fietsen",
           "silent_seen": "alleen de getallen van het model laten het zien: {rented} verhuurd",
           "caption": "Beide verzoeken zijn voor 8:00 op 10 maart 2011. Het eerste komt bij een server zonder vrij geheugen. Het tweede komt bij een gezonde server, met de 0 van de vochtigheidssensor erin, terwijl het regende.",
           "alt": "Twee rijen. In de eerste gaan de service en het model stuk op een verzoek en antwoorden met status 500: een softwaremonitor ziet het. In de tweede krijgen de service en het model een verzoek met een vochtigheid van 0 en antwoorden met status 200 en {bikes} fietsen, terwijl er {rented} verhuurd werden: alleen de getallen van het model laten het zien."},
}
IDEA_MS = 31   # the latency the picture shows: a usual request of the log of *Watching a running model*


def _seen_lines(y: float, text_: str) -> str:
    """Who sees a failure, under the service, on as many lines as it needs within the picture."""
    return "".join(_t_svg(94, y + 12 * k, line, 10.5, 400, "var(--ea-ink-2)") for k, line in enumerate(textwrap.wrap(text_, 36)))


def idea_picture(lang: str) -> str:
    """Two rows: a request that meets a broken server, and a request with the broken sensor's 0 that gets a wrong answer
    with status 200."""
    lang = _lang(lang)
    w = IDEA_TEXT[lang]
    bikes, got = answer(SENSOR), rented(SENSOR_DAY, 8)
    parts = [_t_svg(4, 14, w["title"], 13, 600)]
    for k, (head, colour) in enumerate(((w["loud"], "var(--ea-loss)"), (w["silent"], "var(--ea-ink)"))):
        y = 38 + 82 * k
        parts.append(_t_svg(4, y, head, 11.5, 600, colour))
        parts.append(f'<rect x="4" y="{y + 8}" width="56" height="26" rx="4" style="fill:none;stroke:var(--ea-ink-3);stroke-width:1"/>'
                     + _t_svg(32, y + 25, w["request"], 10.5, 400, "var(--ea-ink-2)", "middle"))
        if k == 1:
            parts.append(_t_svg(32, y + 46, w["silent_in"], 10.5, 600, "var(--ea-loss-text)", "middle"))
        parts.append(f'<line x1="62" y1="{y + 21}" x2="86" y2="{y + 21}" style="stroke:var(--ea-ink-3);stroke-width:1.2"/>'
                     f'<path d="M86 {y + 17} L92 {y + 21} L86 {y + 25} Z" style="fill:var(--ea-ink-3)"/>')
        parts.append(f'<rect x="94" y="{y + 8}" width="88" height="26" rx="4" style="fill:var(--ea-inset);stroke:var(--ea-ink-3);stroke-width:1"/>'
                     + _t_svg(138, y + 25, w["service"], 10.5, 400, "var(--ea-ink)", "middle"))
        parts.append(f'<line x1="184" y1="{y + 21}" x2="196" y2="{y + 21}" style="stroke:var(--ea-ink-3);stroke-width:1.2"/>'
                     f'<path d="M196 {y + 17} L202 {y + 21} L196 {y + 25} Z" style="fill:var(--ea-ink-3)"/>')
        if k == 0:
            parts.append(_t_svg(206, y + 25, w["loud_out"], 11, 600, "var(--ea-loss-text)"))
            parts.append(_seen_lines(y + 52, w["loud_seen"]))
        else:
            out = w["silent_out"].format(ms=IDEA_MS, bikes=number(bikes, 0, lang))
            parts.append(_t_svg(206, y + 18, out.split(" · ")[0], 11, 600, "var(--ea-ink)"))
            parts.append(_t_svg(206, y + 32, " · ".join(out.split(" · ")[1:]), 11, 600, "var(--ea-pred-text)"))
            parts.append(_seen_lines(y + 52, w["silent_seen"].format(rented=number(got, 0, lang))))
    alt = w["alt"].format(bikes=number(bikes, 0, lang), rented=number(got, 0, lang))
    return _svg(192, alt, parts) + _note(w["caption"])


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


# ---- the requests the picture of unseen checks ----------------------------------------------------------------------

def _requests() -> list:
    """(group, request) for the picture of unseen: a usual day, the heat of 23 July, the hottest hours of 2012, and the
    requests of the lesson that something upstream broke."""
    out = [("day", _row(HOURS_2011, WATCH_DAY, h)) for h in range(24)]
    out += [("heat", _row(HOURS_2011, HEAT_DAY, h)) for h in range(11, 19)]
    hot = HOURS_2012[HOURS_2012["temp"] > HIGH["temp"]]
    out += [("hot", _row(HOURS_2012, int(r.day), int(r.hr))) for r in hot.itertuples()]
    out += [("app", dict(NEW_APP)), ("wind", dict(WINDY)), ("sensor", dict(NO_READING))]
    return out


REQUESTS = _requests()
SCALE = (-0.25, 1.25)   # the scale of each strip: the data's 0 to 1, with room for a value just outside

YOURS_TEXT = {
    "en": {"unseen": {"title": "Your unseen on {n} requests",
                      "intro": "Your `unseen(request, C.LOW, C.HIGH)` on the 24 hours of 14 July, the heat of 23 July from 11:00 to 18:00, the hottest hours of 2012, and the three requests in which something upstream broke. A strip for each input, on the data's scale; the band is the range of training.",
                      "key": ("named by your unseen", "let through", "range of training"),
                      "flagged": "Your `unseen` names an input in {k} of the {n} requests: {which}.",
                      "nothing": "Your `unseen` names no input in any of the {n} requests.",
                      "heat": "It lets the heat of 23 July through: 38.5 degrees lies inside training, yet that day the model was {off} bikes off per hour.",
                      "groups": {"app": "the new app's temperature in degrees", "wind": "the wind in km/h",
                                 "sensor": "the sensor's −1", "hot": "the hottest hours of 2012",
                                 "heat": "hours of the heat of 23 July", "day": "hours of 14 July"},
                      "none": "No picture yet: your `unseen` does not give the name of an input, or `None`, for every request.",
                      "alt": "Four strips, one for temp, atemp, hum and windspeed, each with the range of training as a band and a dot for every request; dots the learner's unseen names are marked."},
           "off": {"title": "Your off on every day of 2011",
                   "intro": "Your `off(predicted, rented)` on each of the 365 days of 2011: the service's answers next to the rentals, hour by hour. A bar for each day; the dashed line is the median of your numbers.",
                   "key": ("bikes off per hour", "the median"),
                   "said": "Half of the days lie under {usual} bikes per hour; 14 July, a usual day, comes to {july}. On {worst}, your number is {top}. {n} days lie above twice the median.",
                   "same": "These are the numbers of the course: the demo shows the worst days hour by hour.",
                   "none": "No picture yet: your `off` does not give a number for every day.",
                   "alt": "A bar for each day of 2011, as high as how far off the model was per hour that day by the learner's off, with a line at the median and the worst days named."}},
    "nl": {"unseen": {"title": "Je unseen op {n} verzoeken",
                      "intro": "Je `unseen(request, C.LOW, C.HIGH)` op de 24 uren van 14 juli, de hitte van 23 juli van 11:00 tot 18:00, de heetste uren van 2012, en de drie verzoeken waarin stroomopwaarts iets stukging. Een strook voor elke input, op de schaal van de data; de band is het bereik van de training.",
                      "key": ("genoemd door je unseen", "doorgelaten", "bereik van de training"),
                      "flagged": "Je `unseen` noemt een input in {k} van de {n} verzoeken: {which}.",
                      "nothing": "Je `unseen` noemt in geen van de {n} verzoeken een input.",
                      "heat": "De hitte van 23 juli komt erdoor: 38,5 graden ligt binnen de training, en toch zat het model er op 23 juli {off} fietsen per uur naast.",
                      "groups": {"app": "de temperatuur in graden van de nieuwe app", "wind": "de wind in km/u",
                                 "sensor": "de −1 van de sensor", "hot": "de heetste uren van 2012",
                                 "heat": "uren van de hitte van 23 juli", "day": "uren van 14 juli"},
                      "none": "Nog geen afbeelding: je `unseen` geeft niet voor elk verzoek de naam van een input, of `None`.",
                      "alt": "Vier stroken, voor temp, atemp, hum en windspeed, elk met het bereik van de training als band en een stip voor elk verzoek; de stippen die de unseen van de leerling noemt zijn gemarkeerd."},
           "off": {"title": "Je off op elke dag van 2011",
                   "intro": "Je `off(predicted, rented)` op elk van de 365 dagen van 2011: de antwoorden van de service naast de verhuurde fietsen, uur na uur. Een staaf voor elke dag; de streepjeslijn is de mediaan van je getallen.",
                   "key": ("fietsen ernaast per uur", "de mediaan"),
                   "said": "De helft van de dagen ligt onder {usual} fietsen per uur; 14 juli, een gewone dag, komt op {july}. Op {worst} is je getal {top}. {n} dagen liggen boven twee keer de mediaan.",
                   "same": "Dit zijn de getallen van de course: de demo toont de slechtste dagen uur na uur.",
                   "none": "Nog geen afbeelding: je `off` geeft niet voor elke dag een getal.",
                   "alt": "Een staaf voor elke dag van 2011, zo hoog als hoe ver het model er op de dag per uur naast zat volgens de off van de leerling, met een lijn op de mediaan en de slechtste dagen bij naam."}},
}


def _name_of(v):
    """What unseen gave back: the name of an input, None, or _STOPPED for anything else."""
    return v if v is None or v in RANGED else _STOPPED


def unseen_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["unseen"]
    fn = _fn(seen)
    n = len(REQUESTS)
    title = w["title"].format(n=n)
    stop = yours.figure(title, _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=title)
    got = _call(lambda: [fn(dict(r), dict(LOW), dict(HIGH)) for _, r in REQUESTS]) if fn is not None else _STOPPED
    if got is _STOPPED:
        return stop
    names = [_name_of(v) for v in got]
    if any(v is _STOPPED for v in names):
        return stop
    L, R, TOP, ROW = 70, 322, 10, 40
    xs = lambda v: L + (R - L) * (min(max(v, SCALE[0]), SCALE[1]) - SCALE[0]) / (SCALE[1] - SCALE[0])  # noqa: E731
    parts = []
    for k, f in enumerate(RANGED):
        y = TOP + ROW * k + 14
        parts.append(_t_svg(L - 8, y + 4, f, 11, 600, "var(--ea-ink)", "end"))
        parts.append(f'<rect x="{xs(LOW[f]):.1f}" y="{y - 7}" width="{xs(HIGH[f]) - xs(LOW[f]):.1f}" height="14" '
                     f'style="fill:var(--ea-data-wash)"/>'
                     f'<line x1="{L}" y1="{y}" x2="{R}" y2="{y}" style="stroke:var(--ea-line);stroke-width:1"/>')
        for (group, r), name in zip(REQUESTS, names):
            v = r[f]
            mark = name == f
            colour = "var(--ea-loss)" if mark else "var(--ea-ink-3)"
            radius = 3.2 if mark else 2.2
            parts.append(f'<circle cx="{xs(v):.1f}" cy="{y}" r="{radius}" style="fill:{colour};opacity:{1 if mark else 0.7}"/>')
            if not SCALE[0] <= v <= SCALE[1]:
                anchor, dx = ("end", -6) if v > SCALE[1] else ("start", 6)
                parts.append(_t_svg(xs(v) + dx, y - 8, number(v, 1, lang), 9.5, 600,
                                    "var(--ea-loss-text)" if mark else "var(--ea-ink-3)", anchor))
    for tick in (0, 0.5, 1):
        x = xs(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{TOP + ROW * 4}" x2="{x:.1f}" y2="{TOP + ROW * 4 + 4}" style="stroke:var(--ea-ink-3);stroke-width:1"/>'
                     + _t_svg(x, TOP + ROW * 4 + 15, number(tick, 1 if tick == 0.5 else 0, lang), 10, 400,
                              "var(--ea-ink-3)", "middle"))
    svg = _svg(TOP + ROW * 4 + 21, w["alt"], parts)
    flagged = [g for (g, _), name in zip(REQUESTS, names) if name is not None]
    order = ["app", "wind", "sensor", "hot", "heat", "day"]
    which = [w["groups"][g] for g in order if g in flagged]
    said = (w["flagged"].format(k=len(flagged), n=n, which=_list_of(which, lang)) if flagged
            else w["nothing"].format(n=n))
    heat_names = [name for (g, _), name in zip(REQUESTS, names) if g == "heat"]
    if all(v is None for v in heat_names):
        said += " " + w["heat"].format(off=number(day_numbers(HEAT_DAY)["off"], 1, lang))
    key = [("dot", "var(--ea-loss)", w["key"][0]), ("dot", "var(--ea-ink-3)", w["key"][1]),
           ("bar", "var(--ea-data-wash)", w["key"][2])]
    body = _note(w["intro"]) + _key(key) + svg + _note(said)
    return yours.figure(title, body, alt=title)


def _number_of(v):
    """What off gave back as a plain number, or None."""
    if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
        return None
    return float(v) if math.isfinite(float(v)) else None


NAMED_DAYS = [SENSOR_DAY, JULY_4, HEAT_DAY, IRENE_DAY, AFTER_THANKSGIVING]   # the days the picture names


def year_chart(values: dict, lang: str, alt: str, colour: str) -> str:
    """A bar for each day of 2011, a dashed line at the median, and the named days marked."""
    L, R, TOP, H = 34, 326, 34, 110
    top = max(max(values.values()), 1.0)
    raw = top / 4   # about four lines of the grid, whatever the learner's numbers
    scale = 10 ** math.floor(math.log10(raw)) if raw >= 10 else 10
    step = next(m * scale for m in (1, 2, 5, 10) if m * scale >= raw)
    ceiling = step * math.ceil(top / step)
    ys = lambda v: TOP + H - H * min(max(v, 0), ceiling) / ceiling   # noqa: E731
    xs = lambda k: L + (R - L) * (k + 0.5) / len(DATES)   # noqa: E731
    parts = []
    for tick in range(0, int(ceiling) + 1, step):
        parts.append(f'<line x1="{L}" y1="{ys(tick):.1f}" x2="{R}" y2="{ys(tick):.1f}" style="stroke:var(--ea-line);stroke-width:0.8"/>'
                     + _t_svg(L - 5, ys(tick) + 4, number(tick, 0, lang), 10, 400, "var(--ea-ink-3)", "end"))
    width = (R - L) / len(DATES)
    for k, d in enumerate(DATES):
        v = values[d]
        parts.append(f'<rect x="{xs(k) - width / 2:.2f}" y="{ys(v):.1f}" width="{max(width - 0.15, 0.4):.2f}" '
                     f'height="{TOP + H - ys(v):.1f}" style="fill:{colour}"/>')
    med = float(np.median(list(values.values())))
    parts.append(f'<line x1="{L}" y1="{ys(med):.1f}" x2="{R}" y2="{ys(med):.1f}" '
                 f'style="stroke:var(--ea-ink);stroke-width:1;stroke-dasharray:4 3"/>')
    rows = {SENSOR_DAY: 0, JULY_4: 1, HEAT_DAY: 0, IRENE_DAY: 1, AFTER_THANKSGIVING: 0}
    for day in NAMED_DAYS:
        k = DATES.index(date(day))
        x, y = xs(k), 10 + 12 * rows[day]
        anchor = "end" if day == AFTER_THANKSGIVING else ("start" if day == SENSOR_DAY else "middle")
        parts.append(f'<line x1="{x:.1f}" y1="{y + 3}" x2="{x:.1f}" y2="{ys(values[date(day)]) - 2:.1f}" '
                     f'style="stroke:var(--ea-ink-3);stroke-width:0.6"/>'
                     + _t_svg(x, y, said_date(date(day), lang), 9.5, 600, "var(--ea-ink-2)", anchor))
    for m in (0, 3, 6, 9):
        k = DATES.index(f"2011-{m + 1:02d}-01")
        parts.append(_t_svg(xs(k), TOP + H + 14, MONTHS[lang][m][:3], 10, 400, "var(--ea-ink-3)", "start"))
    return _svg(TOP + H + 20, alt, parts)


def off_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["off"]
    fn = _fn(seen)
    stop = yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    got = _call(lambda: [fn(list(YEAR[d][0]), list(YEAR[d][1])) for d in DATES]) if fn is not None else _STOPPED
    if got is _STOPPED:
        return stop
    values = [_number_of(v) for v in got]
    if any(v is None for v in values):
        return stop
    mine = dict(zip(DATES, values))
    same = all(abs(mine[d] - OFF_BY_DAY[d]) < 1e-6 for d in DATES)
    colour = "var(--ea-loss)"
    med = float(np.median(values))
    worst = max(DATES, key=lambda d: mine[d])
    said = w["said"].format(usual=number(med, 1, lang), july=number(mine[date(WATCH_DAY)], 1, lang),
                            worst=said_date(worst, lang), top=number(mine[worst], 1, lang),
                            n=sum(v > 2 * med for v in values))
    key = [("bar", colour, w["key"][0]), ("dashed", "var(--ea-ink)", w["key"][1])]
    body = (_note(w["intro"]) + _key(key) + year_chart(mine, lang, w["alt"], colour) + _note(said)
            + (_note(w["same"]) if same else ""))
    return yours.figure(w["title"], body, alt=w["title"])


# ======== the demo: six days of 2011, hour by hour ========================================================================

DEMO_DAYS = {"usual": WATCH_DAY, "sensor": SENSOR_DAY, "heat": HEAT_DAY, "irene": IRENE_DAY, "july4": JULY_4,
             "thanksgiving": AFTER_THANKSGIVING}
DEMO_TEXT = {
    "en": {"title": "{day}: the model and the rentals per hour",
           "days": {"usual": "Thursday 14 July", "sensor": "Thursday 10 March", "heat": "Saturday 23 July",
                    "irene": "Saturday 27 August", "july4": "Monday 4 July", "thanksgiving": "Friday 25 November"},
           "key": ("rented", "the model's answer"),
           "said": "The model answered {answered} bikes for the whole day; {rented} were rented. It was {off} bikes off per hour, where half of the days of 2011 lie under {usual}.",
           "notes": {"usual": "A usual day: the model follows the morning and the evening rush.",
                     "sensor": "The humidity sensor sent 0 for {sensor_hours} hours, while it rained. With {rainy}, the humidity of other rainy hours, the model answers {fixed} bikes for the whole day.",
                     "heat": "Up to 38.5 degrees in the afternoon. In training, only {hot} of the {train} hours were 37 degrees or warmer.",
                     "irene": "Hurricane Irene reached Washington. The data has no hours after 17:00.",
                     "july4": "Independence Day, a holiday: in training, the model saw {holidays} holidays.",
                     "thanksgiving": "The day after Thanksgiving: a working day in the data, but many people are off."},
           "alt": "A bar for each hour of the chosen day with the bikes rented, and a line with the model's answer for each hour."},
    "nl": {"title": "{day}: het model en de verhuurde fietsen per uur",
           "days": {"usual": "Donderdag 14 juli", "sensor": "Donderdag 10 maart", "heat": "Zaterdag 23 juli",
                    "irene": "Zaterdag 27 augustus", "july4": "Maandag 4 juli", "thanksgiving": "Vrijdag 25 november"},
           "key": ("verhuurd", "het antwoord van het model"),
           "said": "Het model antwoordde {answered} fietsen voor de hele dag; er werden er {rented} verhuurd. Het model zat er {off} fietsen per uur naast, terwijl de helft van de dagen van 2011 onder {usual} ligt.",
           "notes": {"usual": "Een gewone dag: het model volgt de ochtendspits en de avondspits.",
                     "sensor": "De vochtigheidssensor gaf {sensor_hours} uur lang 0, terwijl het regende. Met {rainy}, de vochtigheid van andere regenachtige uren, antwoordt het model {fixed} fietsen voor de hele dag.",
                     "heat": "Tot 38,5 graden in de middag. In de training waren {hot} van de {train} uren 37 graden of warmer.",
                     "irene": "Orkaan Irene bereikte Washington. De data heeft geen uren na 17:00.",
                     "july4": "Independence Day, een feestdag: in de training zag het model {holidays} feestdagen.",
                     "thanksgiving": "De dag na Thanksgiving: een werkdag in de data, maar veel mensen zijn vrij."},
           "alt": "Een staaf voor elk uur van de gekozen dag met de verhuurde fietsen, en een lijn met het antwoord van het model voor elk uur."},
}


def demo_picture(key, lang: str) -> str:
    """The demo: the model's answer and the rentals per hour on one of six days of 2011."""
    lang = _lang(lang)
    w = DEMO_TEXT[lang]
    key = key if key in DEMO_DAYS else "usual"
    day = DEMO_DAYS[key]
    answers, rentals, hrs = YEAR[date(day)]
    L, R, TOP, H = 34, 326, 12, 130
    top = max(max(answers), max(rentals), 1)
    step = 200 if top > 600 else 100
    ceiling = step * math.ceil(top / step)
    ys = lambda v: TOP + H - H * v / ceiling   # noqa: E731
    xs = lambda h: L + (R - L) * (h + 0.5) / 24   # noqa: E731
    parts = []
    for tick in range(0, ceiling + 1, step):
        parts.append(f'<line x1="{L}" y1="{ys(tick):.1f}" x2="{R}" y2="{ys(tick):.1f}" style="stroke:var(--ea-line);stroke-width:0.8"/>'
                     + _t_svg(L - 5, ys(tick) + 4, number(tick, 0, lang), 10, 400, "var(--ea-ink-3)", "end"))
    width = (R - L) / 24
    for h, r in zip(hrs, rentals):
        parts.append(f'<rect x="{xs(h) - width * 0.38:.1f}" y="{ys(r):.1f}" width="{width * 0.76:.1f}" '
                     f'height="{TOP + H - ys(r):.1f}" style="fill:var(--ea-data)"/>')
    runs, run_ = [], []
    for h, a in zip(hrs, answers):
        if run_ and h != run_[-1][0] + 1:
            runs.append(run_)
            run_ = []
        run_.append((h, a))
    runs.append(run_)
    for run_ in runs:
        points = " ".join(f"{xs(h):.1f},{ys(a):.1f}" for h, a in run_)
        parts.append(f'<polyline points="{points}" style="fill:none;stroke:var(--ea-pred);stroke-width:2;stroke-linejoin:round"/>')
        parts.extend(f'<circle cx="{xs(h):.1f}" cy="{ys(a):.1f}" r="2.2" style="fill:var(--ea-pred)"/>' for h, a in run_)
    for h in (0, 6, 12, 18):
        parts.append(_t_svg(xs(h), TOP + H + 14, f"{h}:00", 10, 400, "var(--ea-ink-3)", "middle"))
    n = NUMBERS
    note = w["notes"][key].format(sensor_hours=n["sensor_hours"], rainy=number(RAINY_HUM, 2, lang),
                                  fixed=number(n["sensor_fixed"], 0, lang), hot=n["hot_hours"],
                                  train=number(n["train_hours"], 0, lang), holidays=n["holiday_days"])
    d = day_numbers(day)
    said = w["said"].format(answered=number(d["answered"], 0, lang), rented=number(d["rented"], 0, lang),
                            off=number(d["off"], 1, lang), usual=number(USUAL, 1, lang))
    key_ = [("bar", "var(--ea-data)", w["key"][0]), ("solid", "var(--ea-pred)", w["key"][1])]
    title = w["title"].format(day=w["days"][key])
    body = _key(key_) + _svg(TOP + H + 20, w["alt"], parts) + _note(note) + _note(said)
    return yours.figure(title, body, alt=w["alt"])


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
    """{got} and {want}: what the function gives back, as Python writes it, a number with one decimal."""
    if isinstance(v, float) and math.isfinite(v):
        text_ = f"{v:.1f}"
    else:
        text_ = repr(v)
    return f"`{text_}`" if "`" not in text_ else f"<code>{html.escape(text_)}</code>"


FAILURES = Sort(
    name="failures",
    groups={
        "loud": {"en": "A software monitor sees it", "nl": "Een softwaremonitor ziet het"},
        "silent": {"en": "Only the model's numbers show it", "nl": "Alleen de getallen van het model laten het zien"},
    },
    items={
        "memory": {"en": "**Out of memory.** The server runs out of memory and stops in the middle of the evening rush.",
                   "nl": "**Geen geheugen.** De server heeft geen vrij geheugen meer en stopt midden in de avondspits."},
        "sensor": {"en": "**A broken sensor.** The humidity sensor sends 0 all day, while it rains.",
                   "nl": "**Een kapotte sensor.** De vochtigheidssensor geeft de hele dag 0, terwijl het regent."},
        "package": {"en": "**A missing package.** A new version of the service does not start: a package is missing.",
                    "nl": "**Een ontbrekend pakket.** Een nieuwe versie van de service start niet: er ontbreekt een pakket."},
        "degrees": {"en": "**Degrees.** A new version of the app sends the temperature in degrees.",
                    "nl": "**Graden.** Een nieuwe versie van de app stuurt de temperatuur in graden."},
        "backup": {"en": "**A backup.** A backup on the same machine makes every answer take half a second.",
                   "nl": "**Een back-up.** Een back-up op dezelfde machine laat elk antwoord een halve seconde duren."},
        "heat": {"en": "**A heat wave.** At 38 degrees, fewer people ride than on a warm day, and the model expects more.",
                 "nl": "**Een hittegolf.** Bij 38 graden fietsen minder mensen dan op een warme dag, en het model verwacht er meer."},
        "file": {"en": "**A missing file.** The model file is gone after a move, and every request gets status 500.",
                 "nl": "**Een ontbrekend bestand.** Het bestand van het model is weg na een verhuizing, en elk verzoek krijgt status 500."},
        "utc": {"en": "**Another clock.** A step upstream switches the hour from local time to UTC.",
                "nl": "**Een andere klok.** Een stap stroomopwaarts zet het uur om van lokale tijd naar UTC."},
    },
    solution={"memory": "loud", "sensor": "silent", "package": "loud", "degrees": "silent",
              "backup": "loud", "heat": "silent", "file": "loud", "utc": "silent"},
    nudge={
        "memory": {"en": "A server that stops answers nothing at all. Which numbers of the service jump then?",
                   "nl": "Een server die stopt geeft helemaal geen antwoord meer. Welke getallen van de service springen dan?"},
        "sensor": {"en": "The request with humidity 0 is well formed, and the server answers it with status 200. Who notices that the 0 is wrong?",
                   "nl": "Het verzoek met een vochtigheid van 0 is goed gevormd, en de server antwoordt met status 200. Wie merkt dat de 0 niet klopt?"},
        "package": {"en": "A service that does not start gives no answers. Does that need the model's numbers to be seen?",
                    "nl": "Een service die niet start geeft geen antwoorden. Zijn de getallen van het model nodig om dat te zien?"},
        "degrees": {"en": "31.2 is a well-formed number, answered with status 200, as fast as ever. Which numbers show that it is not 0.76?",
                    "nl": "31,2 is een goed gevormd getal, beantwoord met status 200, even snel als altijd. Welke getallen laten zien dat het geen 0,76 is?"},
        "backup": {"en": "Half a second per answer: which number of the service shows that?",
                   "nl": "Een halve seconde per antwoord: welk getal van de service laat dat zien?"},
        "heat": {"en": "The server answers every request of the heat wave, fast and with status 200. What shows that the answers are too high?",
                 "nl": "De server beantwoordt elk verzoek van de hittegolf, snel en met status 200. Wat laat zien dat de antwoorden te hoog zijn?"},
        "file": {"en": "Every request gets status 500. Which number of the service jumps?",
                 "nl": "Elk verzoek krijgt status 500. Welk getal van de service springt?"},
        "utc": {"en": "An hour of 13 instead of 8 is a well-formed number. Does the server notice, or does it show only in the answers?",
                "nl": "Een uur 13 in plaats van 8 is een goed gevormd getal. Merkt de server het, of zie je het alleen aan de antwoorden?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the failures again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de storingen opnieuw uit."},
        "empty": {"en": "Put each failure with what sees it.",
                  "nl": "Bepaal voor elke storing waaraan je de storing ziet."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "*{item}* belongs with the other side.",
                  "nl": "*{item}* hoort bij de andere kant."},
        "correct": {"en": "All {n} are right. Out of memory, a missing package, a backup and a missing file break the service or slow it down, and its own numbers jump. A broken sensor, degrees, a heat wave and another clock give well-formed requests and fast answers with status 200, so only the model's own numbers show them: a check against training catches the degrees, and how far off the model was shows all four.",
                    "nl": "Alle {n} kloppen. Geen geheugen, een ontbrekend pakket, een back-up en een ontbrekend bestand laten de service stukgaan of vertragen, en de eigen getallen van de service springen. Een kapotte sensor, graden, een hittegolf en een andere klok geven goed gevormde verzoeken en snelle antwoorden met status 200, dus laten alleen de eigen getallen van het model deze vier storingen zien: een check tegen de training vangt de graden, en hoe ver het model ernaast zat laat alle vier zien."},
    },
)

EXERCISES = {
    "unseen": FunctionBox(
        name="unseen",
        function="unseen",
        start=_U_START,
        solution=_U,
        names={},
        visible=[(NEW_APP, LOW, HIGH)],
        hidden=[(MORNING, LOW, HIGH), (HOT, LOW, HIGH), (HEAT, LOW, HIGH), (WINDY, LOW, HIGH), (NO_READING, LOW, HIGH),
                (FROZEN, LOW, HIGH), (CALM, LOW, HIGH), (FOG, LOW, HIGH)],
        first=[_DOTS],
        show=_shown,
        picture=unseen_picture,
        mistakes=[
            Mistake("low-only", wrong=_U_LOW,
                    en="`{call}` gives {got}: your check looks only at the bottom of each range, and 31.2 is far above the top. Which comparison catches a value above `high[name]`?",
                    nl="`{call}` geeft {got}: je check kijkt alleen naar de onderkant van elk bereik, en 31,2 ligt ver boven de bovenkant. Welke vergelijking vangt een waarde boven `high[name]`?"),
            Mistake("high-only", wrong=_U_HIGH,
                    en="`{call}` gives {got}: your check looks only at the top of each range, so the sensor's −1, under the bottom, gets through. Which comparison catches a value under `low[name]`?",
                    nl="`{call}` geeft {got}: je check kijkt alleen naar de bovenkant van elk bereik, dus komt de −1 van de sensor, onder de onderkant, erdoor. Welke vergelijking vangt een waarde onder `low[name]`?"),
            Mistake("swapped", wrong=_U_SWAPPED,
                    en="`{call}` gives {got}: your check names an input of a usual morning, inside every range of training. Is a value outside when it lies above the bottom, or under it?",
                    nl="`{call}` geeft {got}: je check noemt een input van een gewone ochtend, binnen elk bereik van de training. Ligt een waarde erbuiten als de waarde boven de onderkant ligt, of eronder?"),
            Mistake("edges", wrong=_U_EDGES,
                    en="`{call}` gives {got}: a value exactly at the bottom or the top of a range, such as no wind at all, did occur in training. Should a value equal to `low[name]` count as outside?",
                    nl="`{call}` geeft {got}: een waarde precies op de onderkant of de bovenkant van een bereik, zoals helemaal geen wind, kwam in de training voor. Ligt een waarde die gelijk is aan `low[name]` buiten het bereik?"),
            Mistake("edge-low", wrong=_U_EDGE_LOW,
                    en="`{call}` gives {got}: a value exactly at the bottom of a range, such as no wind at all, did occur in training. Should a value equal to `low[name]` count as outside?",
                    nl="`{call}` geeft {got}: een waarde precies op de onderkant van een bereik, zoals helemaal geen wind, kwam in de training voor. Ligt een waarde die gelijk is aan `low[name]` buiten het bereik?"),
            Mistake("edge-high", wrong=_U_EDGE_HIGH,
                    en="`{call}` gives {got}: a value exactly at the top of a range, such as a humidity of 1, did occur in training. Should a value equal to `high[name]` count as outside?",
                    nl="`{call}` geeft {got}: een waarde precies op de bovenkant van een bereik, zoals een vochtigheid van 1, kwam in de training voor. Ligt een waarde die gelijk is aan `high[name]` buiten het bereik?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the condition: the value of the input lies under its lowest value in training, or above its highest.",
                      "nl": "Vervang de `...` door de voorwaarde: de waarde van de input ligt onder de laagste waarde uit de training, of boven de hoogste."},
            "arguments": {"en": "The check calls `{call}`, with a request and two dicts, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(request, low, high)`?",
                          "nl": "De check roept `{call}` aan, met een verzoek en twee dicts, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(request, low, high)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your condition hold only for a value under `low[name]` or above the highest value of training?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Klopt je voorwaarde alleen voor een waarde onder `low[name]` of boven de hoogste waarde uit de training?"},
            "almost": {"en": "Right for the new app. But `{call}` gives {got}, and it should give {want}. Does your condition use `request[name]`, `low[name]` and `high[name]`, and not one input or a fixed number?",
                       "nl": "Goed voor de nieuwe app. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je voorwaarde `request[name]`, `low[name]` en `high[name]`, en niet één input of een vast getal?"},
            "unreached": {"en": "With `{call}`, your function gives `None`: Python never gets to `return name`. A value outside lies under the bottom of its range or above the top: can your condition be true for each, and is `return name` still under the `if`?",
                          "nl": "Met `{call}` geeft je functie `None`: Python komt nooit bij `return name`. Een waarde erbuiten ligt onder de onderkant van het bereik of boven de bovenkant: kan je voorwaarde voor allebei waar zijn, en staat `return name` nog onder de `if`?"},
            "correct": {"en": "Right in all {n} cases. The new app's 31.2, the wind in km/h, the sensor's −1, the frosty morning and 2012's hottest hour lie outside training, and your check names all five. The heat of 23 July gets through: 38.5 degrees lies inside the range, since training has those hours. In the picture: what does your check let through?",
                        "nl": "Goed in alle {n} gevallen. De 31,2 van de nieuwe app, de wind in km/u, de −1 van de sensor, de ijskoude ochtend en het heetste uur van 2012 liggen buiten de training, en je check noemt alle vijf. De hitte van 23 juli komt erdoor: 38,5 graden ligt binnen het bereik, want de training bevat zulke hete uren. In de afbeelding: wat laat je check door?"},
        },
    ),
    "off": FunctionBox(
        name="off",
        function="off",
        start=_O_START,
        solution=_O,
        names={},
        visible=[([120, 80, 30], [100, 95, 30])],
        hidden=[(ANSWERED[date(WATCH_DAY)], RENTED[date(WATCH_DAY)]), (ANSWERED[date(HEAT_DAY)], RENTED[date(HEAT_DAY)]),
                (ANSWERED[date(SENSOR_DAY)], RENTED[date(SENSOR_DAY)]), ([10, 0], [0, 10]), ([5], [5])],
        first=[_DOTS],
        show=_shown,
        picture=off_picture,
        mistakes=[
            Mistake("signed", wrong=_O_SIGNED,
                    en="`{call}` gives {got}: 20 bikes too many in the first hour and 15 too few in the second cancel out, as if the model were almost right. Which Python function makes a difference positive, whichever way it goes?",
                    nl="`{call}` geeft {got}: 20 fietsen te veel in het eerste uur en 15 te weinig in het tweede heffen elkaar op, alsof het model er bijna niet naast zat. Welke Python-functie maakt een verschil positief, welke kant het ook op gaat?"),
            Mistake("under", wrong=_O_UNDER,
                    en="`{call}` gives {got}: the hours with too many and too few bikes cancel out, as if the model were almost right. Which Python function makes a difference positive, whichever way it goes?",
                    nl="`{call}` geeft {got}: de uren met te veel en met te weinig fietsen heffen elkaar op, alsof het model er bijna niet naast zat. Welke Python-functie maakt een verschil positief, welke kant het ook op gaat?"),
            Mistake("squared", wrong=_O_SQUARED,
                    en="`{call}` gives {got}: the mean of the squared differences, in bikes squared, as in a loss. How many bikes off was the model per hour, in bikes?",
                    nl="`{call}` geeft {got}: het gemiddelde van de gekwadrateerde verschillen, in fietsen in het kwadraat, zoals in een loss. Hoeveel fietsen zat het model er per uur naast, in fietsen?"),
            Mistake("twice", wrong=_O_TWICE,
                    en="`{call}` gives {got}: your function divides by `len(rented)` twice, once per hour and once at the end. Where does the division by the number of hours already happen?",
                    nl="`{call}` geeft {got}: je functie deelt twee keer door `len(rented)`, één keer per uur en één keer aan het eind. Waar gebeurt de deling door het aantal uren al?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with how far off the model was in hour `h`: the difference between `predicted[h]` and `rented[h]`, up or down.",
                      "nl": "Vervang de `...` door hoe ver het model er in uur `h` naast zat: het verschil tussen `predicted[h]` en `rented[h]`, omhoog of omlaag."},
            "arguments": {"en": "The check calls `{call}`, with two lists, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(predicted, rented)`?",
                          "nl": "De check roept `{call}` aan, met twee lists, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(predicted, rented)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does `total` grow by how far off the model was in hour `h`, as a positive number?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Groeit `total` met hoe ver het model er in uur `h` naast zat, als positief getal?"},
            "almost": {"en": "Right for the short lists. But `{call}` gives {got}, and it should give {want}. Does your line use `predicted[h]` and `rented[h]`, and not fixed numbers?",
                       "nl": "Goed voor de korte lists. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je regel `predicted[h]` en `rented[h]`, en geen vaste getallen?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return total / len(rented)`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return total / len(rented)`?"},
            "correct": {"en": "Right in all {n} cases. On 14 July, the model was 25.7 bikes off per hour; on 23 July, 147.5; on 10 March, 69.7. In the picture: which days stand out over the year?",
                        "nl": "Goed in alle {n} gevallen. Op 14 juli zat het model er 25,7 fietsen per uur naast; op 23 juli 147,5; op 10 maart 69,7. In de afbeelding: welke dagen springen eruit over het jaar?"},
        },
    ),
    "failures": FAILURES,
}

REQUIRED = ["unseen", "off", "first", "failures"]


_SORTED = dict(FAILURES.solution)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("unseen", _U_START, "empty", None),
    ("unseen", "", "empty", None),
    ("unseen", _U, "correct", None),
    ("unseen", _u("not low[name] <= request[name] <= high[name]"), "correct", None),
    ("unseen", _u("request[name] > high[name] or request[name] < low[name]"), "correct", None),
    ("unseen", _U_LOW, "wrong", "low-only"),
    ("unseen", _U_HIGH, "wrong", "high-only"),
    ("unseen", _U_AND, "wrong", "unreached"),   # never true, so None everywhere
    ("unseen", _U_SWAPPED, "wrong", "swapped"),
    ("unseen", _U_EDGES, "wrong", "edges"),
    ("unseen", _U_EDGE_LOW, "wrong", "edge-low"),
    ("unseen", _U_EDGE_HIGH, "wrong", "edge-high"),
    ("unseen", _u("request[name] < low[name] or request[name] > 0.9"), "almost", None),
    ("unseen", _u("request[name] < 0 or request[name] > high[name]"), "wrong", "unreached"),
    ("unseen", _u('request["temp"] < low["temp"] or request["temp"] > high["temp"]'), "wrong", "unreached"),
    ("unseen", _u("request[name] < high[name]"), "wrong", None),
    ("unseen", _u("request[name] > 1"), "wrong", "unreached"),
    ("unseen", _U.replace("            return name ", "            pass        "), "wrong", "unreached"),
    ("unseen", _U.replace("            return name ", "            print(name) "), "wrong", "prints"),
    ("unseen", _U.replace("def unseen(request, low, high):", "def unseen(request):"), "error", "arguments"),
    ("unseen", _U.replace("    return None ", "    ...\n    return None "), "wrong", "dots-left"),
    ("unseen", "x = 1\n", "wrong", "no-function"),
    ("off", _O_START, "empty", None),
    ("off", "", "empty", None),
    ("off", _O, "correct", None),
    ("off", _o("abs(rented[h] - predicted[h])"), "correct", None),
    ("off", _O_SIGNED, "wrong", "signed"),
    ("off", _O_UNDER, "wrong", "under"),
    ("off", _O_SQUARED, "wrong", "squared"),
    ("off", _O_TWICE, "wrong", "twice"),
    ("off", _o("abs(predicted[0] - rented[0])"), "wrong", None),
    ("off", _o("abs(predicted[h] - rented[h]) + 3 - len(rented)"), "almost", None),
    ("off", _O.replace("    return total / len(rented)             #", "    total / len(rented)                    #"), "wrong", "no-return"),
    ("off", _O.replace("    return total / len(rented)             #", "    print(total / len(rented))             #"), "wrong", "prints"),
    ("off", _O.replace("def off(predicted, rented):", "def off(predicted):"), "error", "arguments"),
    ("off", _O.replace("    return total ", "    ...\n    return total "), "wrong", "dots-left"),
    ("off", "x = 1\n", "wrong", "no-function"),
    ("failures", {}, "empty", None),
    ("failures", _SORTED, "correct", None),
    ("failures", {"memory": "loud", "sensor": "silent"}, "wrong", "unfinished"),
] + [("failures", _SORTED | {item: "silent" if side == "loud" else "loud"}, "wrong", f"wrong:{item}")
     for item, side in FAILURES.solution.items()]
