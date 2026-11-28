# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-monitoring: watching the service of the project while it runs. The course makes the log of
one day of the service of the lessons *Your model behind an API* and *Before and after the model: the serving path*:
every time someone opens the bike-share app, the app asks the service how busy the hour will be, and the service
writes one structured line in JSON for the request. The day holds three incidents: at 09:12 the server stops for two
seconds; from 14:00 to 14:25 a backup on the same machine slows every request down; and from 16:00 a new version of
the app, for one user in five, sends the temperature in degrees. The learner writes the line of `tail` that takes the
99th percentile of the latency of a minute, and the condition of `alert` that goes off only when the p99 stays above
a limit for some minutes in a row; a choice of the number that shows the new app, and a sort of numbers about the
service and about the model. The demo tunes an alert rule on the day.

The hours, the rentals and the weather are those of Thursday 14 July 2011 in the Bike Sharing dataset of Fanaee-T and
Gama (2013, CC BY 4.0, see data/README.md), and the bikes are what the course's serving path gives for them (the test
checks them against that lesson). The requests, the latencies, the incidents and the pictures are the course's own:
made up, not measured. Every word a learner's code gives back is shown through html.escape."""
import html
import json
import math
import re

import numpy as np

from everything_ai import check, yours
from everything_ai.check import FunctionBox, Mistake, Sort, kept

LESSON = "sys-monitoring"
LANGS = ("en", "nl")

# ======== the day of the log: Thursday 14 July 2011 ======================================================================

RENTED = [46, 25, 10, 6, 6, 28, 130, 334, 486, 224, 127, 148, 223, 177, 178, 180, 274, 595, 578, 400, 348, 261, 176, 124]
TEMP = [0.66, 0.66, 0.64, 0.62, 0.62, 0.6, 0.6, 0.62, 0.64, 0.68, 0.7, 0.72, 0.74, 0.76, 0.74, 0.76, 0.76, 0.74, 0.72,
        0.72, 0.7, 0.66, 0.64, 0.64]                    # temp of each hour, as the data has it: degrees divided by 41
BIKES = [46, 23, 7, 0, 0, 8, 101, 296, 431, 223, 144, 167, 205, 203, 186, 198, 289, 537, 517, 376, 269, 202, 151, 94]
BIKES_IN_DEGREES = [4610, 4576, 4405, 4245, 4208, 4096, 4200, 4564, 4855, 4934, 4971, 5128, 5301, 5450, 5278, 5444,
                    5557, 5653, 5498, 5356, 5114, 4778, 4562, 4496]   # the same path, with temp in degrees
SEED = 7
DAY_MINUTES = 24 * 60
PAUSE = 9 * 60 + 12                 # 09:12: the server stops for two seconds, from 09:12:30 to 09:12:32
SLOW = (14 * 60, 14 * 60 + 25)      # 14:00 to 14:25: a backup on the same machine slows every request down
NEW_APP = 16 * 60                   # 16:00: a new version of the app sends temp in degrees
NEW_SHARE = 0.2                     # one user in five has the new version
LIMIT, MINUTES = 300, 5             # the rule of Exercise 2: p99 above 300 ms for 5 minutes in a row


def _lang(lang) -> str:
    return lang if lang in LANGS else "en"


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch, and a real minus sign."""
    out = f"{x:,.{digits}f}"
    out = out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out
    return out.replace("-", "−")


def clock(minute: int) -> str:
    """A minute of the day as the log writes it: 09:12."""
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _simulate() -> list:
    """The log of the day: one line of JSON per request, in the order the requests came in."""
    rng = np.random.default_rng(SEED)
    hour = np.arange(DAY_MINUTES) // 60
    rate = 20 + 60 * np.array(RENTED)[hour] / max(RENTED)        # requests a minute: more when more people ride
    minute = np.repeat(np.arange(DAY_MINUTES), rng.poisson(rate))
    n = len(minute)
    second = rng.uniform(0, 60, n)
    second = second[np.lexsort((second, minute))]
    ms = 20 * np.exp(rng.normal(0, 0.3, n)) * (1 + rate[minute] / 400)          # a busier minute is a little slower
    ms = ms + (rng.random(n) < 0.012) * 90 * np.exp(rng.normal(0, 0.5, n))     # now and then a slow request
    slow = (minute >= SLOW[0]) & (minute < SLOW[1])
    extra = (rng.random(n) < 0.3) * 300 * np.exp(rng.normal(0, 0.4, n))
    ms = np.where(slow, ms * 6 + extra, ms)                                      # the backup: every request waits
    paused = (minute == PAUSE) & (second >= 30) & (second < 32)
    ms = ms + paused * (32 - second) * 1000                                      # a request in the pause waits for its end
    u = rng.random(n)
    status = np.where(u < 0.004, 400, np.where(slow & (u > 0.97), 500, 200))
    new = (minute >= NEW_APP) & (rng.random(n) < NEW_SHARE)
    lines = []
    for k in range(n):
        m = int(minute[k])
        h, st = m // 60, int(status[k])
        temp = None if st == 400 else (round(TEMP[h] * 41, 1) if new[k] else TEMP[h])
        bikes = (BIKES_IN_DEGREES[h] if new[k] else BIKES[h]) if st == 200 else None
        lines.append(json.dumps({"time": f"{clock(m)}:{second[k]:06.3f}", "path": "/predict", "status": st,
                                 "ms": round(float(ms[k]), 1), "temp": temp, "bikes": bikes}))
    return lines


LOG = kept((LESSON, "log"), _simulate)   # 59,000 lines or so: the same in both languages


def _by_minute() -> list:
    out = [[] for _ in range(DAY_MINUTES)]
    for line in LOG:
        out[int(line[10:12]) * 60 + int(line[13:15])].append(line)   # "time": "09:12:30.449"
    return out


BY_MINUTE = kept((LESSON, "minutes"), _by_minute)


class Named(list):
    """A list that the check's call shows by the name the lesson gives it, such as C.at("09:00") or C.P99; the
    learner's function gets a plain copy."""

    def __init__(self, name: str, values):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return list(self)


def at(time: str) -> Named:
    """The log lines of one minute, such as at("09:00"): the requests from 09:00:00 to 09:00:59."""
    parts = str(time).split(":")
    if len(parts) != 2 or not all(p.isdigit() for p in parts) or not (0 <= int(parts[0]) < 24 and 0 <= int(parts[1]) < 60):
        raise ValueError('C.at wants a time from "00:00" to "23:59", such as "09:00"')
    h, m = int(parts[0]), int(parts[1])
    return Named(f'C.at("{h:02d}:{m:02d}")', BY_MINUTE[h * 60 + m])


def _defined(code: str, name: str, names: dict):
    space = dict(names)
    exec(code, space)   # noqa: S102   the course's own solution text
    return space[name]


# ======== exercise 1: tail ===============================================================================================

_T_HEAD = ("def tail(lines):\n"
           "    ms = [json.loads(line)[\"ms\"] for line in lines]   # one JSON object per line: the latency of each request\n"
           "    p50 = float(np.median(ms))                        # half of the requests were faster\n")
_T_END = "    return {\"p50\": round(p50), \"p99\": round(p99)}\n"


def _t(line: str, comment: bool = False) -> str:
    rest = "                                         # 99 in 100 requests were faster" if comment else ""
    return _T_HEAD + f"    p99 = {line}{rest}\n" + _T_END


_T_START = _t("...", comment=True)
_T = _t("float(np.quantile(ms, 0.99))")
_T_PERCENT = _t("float(np.quantile(ms, 99))")
_T_FASTEST = _t("float(np.quantile(ms, 0.01))")
_T_P90 = _t("float(np.quantile(ms, 0.9))")
_T_P95 = _t("float(np.quantile(ms, 0.95))")
_T_MAX = _t("float(np.max(ms))")
_T_MEAN = _t("float(np.mean(ms))")

_T_NAMES = {"json": json, "np": np}
tail = _defined(_T, "tail", _T_NAMES)   # the course's tail


def _quantile_range(e: BaseException) -> bool:
    """np.quantile with a percentage: NumPy wants a fraction from 0 to 1."""
    return isinstance(e, ValueError) and "range [0, 1]" in str(e)


P99 = kept((LESSON, "p99"), lambda: [tail(BY_MINUTE[m])["p99"] for m in range(DAY_MINUTES)])
P50 = kept((LESSON, "p50"), lambda: [tail(BY_MINUTE[m])["p50"] for m in range(DAY_MINUTES)])


P99S = Named("C.P99", P99)   # the p99 of every minute, as the check's call shows it

# ======== exercise 2: alert ==============================================================================================

_A_START = ("def alert(p99s, limit, minutes):\n"
            "    for i in range(minutes - 1, len(p99s)):\n"
            "        last = p99s[i - minutes + 1 : i + 1]   # the p99 of the last `minutes` minutes, up to minute i\n"
            "        if ...:\n"
            "            return i                           # the minute at which the alert goes off\n"
            "    return None                                # the number never stayed too high\n")


def _a(condition: str) -> str:
    return _A_START.replace("        if ...:\n", f"        if {condition}:\n")


_A = _a("min(last) > limit")
_A_MAX = _a("max(last) > limit")
_A_MEAN = _a("np.mean(last) > limit")
_A_NEWEST = _a("last[-1] > limit")
_A_OLDEST = _a("last[0] > limit")
_A_AT = _a("min(last) >= limit")

_A_NAMES = {"np": np}
alert = _defined(_A, "alert", _A_NAMES)   # the course's alert


def alerts(fn, p99s: list, limit: float, minutes: int, most: int = 40) -> list | None:
    """Every minute at which a rule goes off over the day: after an alert, the next one can only come once the number
    has been back at or under the limit. None when `fn` gives something that is not a minute of what it got."""
    out, start = [], 0
    while start < len(p99s) and len(out) < most:
        i = fn(list(p99s[start:]), limit, minutes)
        if i is None:
            break
        if isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= i < len(p99s) - start:
            return None
        out.append(start + int(i))
        start = start + int(i) + 1
        while start < len(p99s) and p99s[start] > limit:
            start += 1
    return out


def caught(found: list) -> int | None:
    """How many minutes into the slowdown the first alert in it went off, or None."""
    inside = [m for m in found if SLOW[0] <= m < SLOW[1]]
    return inside[0] - SLOW[0] if inside else None


def needless(found: list) -> list:
    """The alerts outside the slowdown: for nothing that needed a person."""
    return [m for m in found if not SLOW[0] <= m < SLOW[1]]


# ======== the numbers of the lesson ======================================================================================

LATENCIES = kept((LESSON, "ms"), lambda: np.array([json.loads(line)["ms"] for line in LOG]))


def _numbers() -> dict:
    ms = LATENCIES
    status = np.array([json.loads(line)["status"] for line in LOG])
    return {"n": len(LOG), "mean": float(ms.mean()), "p50": float(np.median(ms)), "p99": float(np.quantile(ms, 0.99)),
            "slowest": float(ms.max()), "errors_400": int((status == 400).sum()), "errors_500": int((status == 500).sum())}


NUMBERS = kept((LESSON, "numbers"), _numbers)


def hour_bikes(hour: int) -> dict:
    """The answers of one hour in the log: how many, their mean and their median."""
    bikes = [b for m in range(hour * 60, hour * 60 + 60) for line in BY_MINUTE[m]
             if (b := json.loads(line)["bikes"]) is not None]
    return {"n": len(bikes), "mean": float(np.mean(bikes)), "median": float(np.median(bikes))}


def minute_stats(time: str) -> dict:
    """What the worked example prints for one minute."""
    ms = [json.loads(line)["ms"] for line in at(time)]
    return {"n": len(ms), "mean": round(float(np.mean(ms)), 1), "median": round(float(np.median(ms)), 1),
            "slowest": max(ms)}


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


def _list_of(items: list, lang: str) -> str:
    word = {"en": " and ", "nl": " en "}[lang]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + word + items[-1]


def _key(items: list) -> str:
    """A small legend: a swatch and a word for each kind of mark."""
    out = []
    for kind, colour, word in items:
        mark = (f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{colour};'
                f'margin-right:4px"></span>' if kind == "dot" else
                f'<span style="display:inline-block;width:14px;height:0;border-top:2px {kind} {colour};'
                f'margin-right:4px;vertical-align:middle"></span>')
        out.append(f'<span style="margin-right:12px;white-space:nowrap">{mark}{html.escape(word)}</span>')
    return f'<div style="{_UI}">{"".join(out)}</div>'


LOW, HIGH = 10, 2000   # the scale of the day's chart, in milliseconds: each grid line is ten times the one below


def day_chart(lang: str, alt: str, series: list, limit: float | None = None, marks: list = ()) -> str:
    """The day from 0:00 to 24:00: a line per series of 1,440 minutes on a scale of milliseconds where each grid line
    is ten times the one below, a level line at the limit, and a dot at the top for each alert."""
    L, R, TOP, H = 52, 326, 14, 140
    ys = lambda v: TOP + H * (math.log10(HIGH) - math.log10(min(max(v, LOW), HIGH))) / (math.log10(HIGH) - math.log10(LOW))  # noqa: E731
    xs = lambda m: L + (R - L) * (m + 0.5) / DAY_MINUTES   # noqa: E731
    parts = []
    for tick in (10, 100, 1000):
        parts.append(f'<line x1="{L}" y1="{ys(tick):.1f}" x2="{R}" y2="{ys(tick):.1f}" style="stroke:var(--ea-line);stroke-width:0.8"/>'
                     + _t_svg(L - 5, ys(tick) + 4, number(tick, 0, lang) + " ms", 10, 400, "var(--ea-ink-3)", "end"))
    for colour, values, width in series:
        points = " ".join(f"{xs(m):.1f},{ys(v):.1f}" for m, v in enumerate(values))
        parts.append(f'<polyline points="{points}" style="fill:none;stroke:{colour};stroke-width:{width};stroke-linejoin:round"/>')
    if limit is not None:
        parts.append(f'<line x1="{L}" y1="{ys(limit):.1f}" x2="{R}" y2="{ys(limit):.1f}" '
                     f'style="stroke:var(--ea-ink);stroke-width:1;stroke-dasharray:4 3"/>')
    for m, colour in marks:
        parts.append(f'<line x1="{xs(m):.1f}" y1="{TOP - 6}" x2="{xs(m):.1f}" y2="{TOP + H}" '
                     f'style="stroke:{colour};stroke-width:0.8;opacity:0.5"/>'
                     f'<circle cx="{xs(m):.1f}" cy="{TOP - 6}" r="3" style="fill:{colour}"/>')
    for h in (0, 6, 12, 18, 24):
        parts.append(_t_svg(L + (R - L) * h / 24, TOP + H + 15, f"{h}:00", 10, 400, "var(--ea-ink-3)",
                            "start" if h == 0 else ("end" if h == 24 else "middle")))
    return _svg(TOP + H + 21, alt, parts)


# ======== the picture of the idea =========================================================================================

IDEA_TEXT = {
    "en": {"title": "The {n} requests of 14 July 2011",
           "mean": "mean", "p50": "p50", "p99": "p99", "slowest": "slowest",
           "caption": "A bar for each range of latencies, as high as the number of requests in it. The scale counts milliseconds, ten times more at each step, so that the thin tail on the right fits too.",
           "alt": "A histogram of the latencies of the {n} requests of 14 July 2011 on a scale of milliseconds. Most "
                  "requests took 15 to 40 ms; a thin tail runs on to the right, up to {slowest} ms. Marks show the "
                  "median or p50 at {p50} ms, the mean at {mean} ms, the p99 at {p99} ms and the slowest request."},
    "nl": {"title": "De {n} verzoeken van 14 juli 2011",
           "mean": "gemiddeld", "p50": "p50", "p99": "p99", "slowest": "traagste",
           "caption": "Een staaf voor elk stuk van de schaal, zo hoog als het aantal verzoeken erin. De schaal telt milliseconden, tien keer zoveel bij elke stap, zodat ook de dunne staart rechts erop past.",
           "alt": "Een histogram van de latency van de {n} verzoeken van 14 juli 2011 op een schaal van milliseconden. "
                  "De meeste verzoeken duurden 15 tot 40 ms; een dunne staart loopt door naar rechts, tot {slowest} ms. "
                  "Streepjes tonen de mediaan of p50 op {p50} ms, het gemiddelde op {mean} ms, de p99 op {p99} ms en "
                  "het traagste verzoek."},
}


def idea_picture(lang: str) -> str:
    """The latencies of the whole day as a histogram on a scale of milliseconds, with the p50, the mean, the p99 and
    the slowest request marked: most requests are fast, and a thin tail of slow ones runs far to the right."""
    lang = _lang(lang)
    w = IDEA_TEXT[lang]
    ms = LATENCIES
    marks = {"p50": NUMBERS["p50"], "mean": NUMBERS["mean"], "p99": NUMBERS["p99"], "slowest": NUMBERS["slowest"]}
    L, R, TOP, H = 8, 322, 58, 90
    span = math.log10(HIGH) - 1
    xs = lambda v: L + (R - L) * (math.log10(v) - 1) / span   # noqa: E731
    edges = np.logspace(1, math.log10(HIGH), 61)
    counts, _ = np.histogram(np.clip(ms, LOW, HIGH - 1e-9), bins=edges)
    top = counts.max()
    parts = [_t_svg(4, 14, w["title"].format(n=number(len(ms), 0, lang)), 13, 600)]
    for k, c in enumerate(counts):
        if c:
            h = max(H * c / top, 1.5)
            x0, x1 = xs(edges[k]), xs(edges[k + 1])
            parts.append(f'<rect x="{x0 + 0.3:.1f}" y="{TOP + H - h:.1f}" width="{max(x1 - x0 - 0.6, 0.6):.1f}" '
                         f'height="{h:.1f}" style="fill:var(--ea-data)"/>')
    parts.append(f'<line x1="{L}" y1="{TOP + H}" x2="{R}" y2="{TOP + H}" style="stroke:var(--ea-ink-3);stroke-width:1"/>')
    for tick in (10, 100, 1000):
        parts.append(f'<line x1="{xs(tick):.1f}" y1="{TOP + H}" x2="{xs(tick):.1f}" y2="{TOP + H + 4}" style="stroke:var(--ea-ink-3);stroke-width:1"/>'
                     + _t_svg(xs(tick), TOP + H + 15, number(tick, 0, lang) + " ms", 10, 400, "var(--ea-ink-3)",
                              "start" if tick == 10 else "middle"))
    places = {"p50": (34, "end"), "mean": (47, "start"), "p99": (34, "start"), "slowest": (47, "end")}
    for key, v in marks.items():
        x = xs(v)
        y, anchor = places[key]
        digits = 1 if key in ("p50", "mean") else 0
        parts.append(f'<line x1="{x:.1f}" y1="{y + 3}" x2="{x:.1f}" y2="{TOP + H}" style="stroke:var(--ea-ink);stroke-width:1"/>'
                     + _t_svg(x + (-3 if anchor == "end" else 3), y, f"{w[key]} {number(v, digits, lang)}", 10.5, 600,
                              "var(--ea-ink)", anchor))
    alt = w["alt"].format(n=number(len(ms), 0, lang), mean=number(marks["mean"], 1, lang),
                          p50=number(marks["p50"], 1, lang), p99=number(marks["p99"], 0, lang),
                          slowest=number(marks["slowest"], 0, lang))
    return _svg(TOP + H + 21, alt, parts) + _note(w["caption"])


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


def _ms_of(v, key: str):
    """The number under `key` of what tail gave back, or None."""
    if not isinstance(v, dict) or key not in v:
        return None
    x = v[key]
    if isinstance(x, bool) or not isinstance(x, (int, float, np.integer, np.floating)):
        return None
    return float(x) if math.isfinite(float(x)) else None


YOURS_TEXT = {
    "en": {"tail": {"title": "Your tail on every minute of 14 July 2011",
                    "intro": "Your `tail(lines)` on each of the 1,440 minutes of the day. The scale counts milliseconds: each grid line is ten times the one below.",
                    "key": ("your p99", "your p50"),
                    "minutes": "In a quiet minute such as 09:00, your p99 is {quiet} ms; at 09:12, {spike} ms; at 14:05, {slow} ms.",
                    "day": "Over the whole day, your p99 is {p99} ms: {share} of the {n} requests took longer.",
                    "same": "These are the numbers of the course: Exercise 2 watches them.",
                    "none": "No picture yet: your `tail` does not give a dict with a number under `p50` and under `p99` for every minute.",
                    "alt": "Two lines over the 1,440 minutes of 14 July 2011 on a scale of milliseconds: the learner's p99 of each minute, and the p50 below it."},
           "alert": {"title": "Your alert on 14 July 2011",
                     "intro": "Your `alert(p99s, limit, minutes)` on the p99 of every minute, with a limit of 300 ms for 5 minutes. After each alert, the next can only come once the p99 is back under the limit.",
                     "key": ("p99", "the limit", "alert for the slowdown", "alert for nothing"),
                     "count": "Your rule goes off {k} times: at {times}.",
                     "two": "Your rule goes off twice: at {times}.",
                     "one": "Your rule goes off once: at {times}.",
                     "many": "Your rule goes off {k} times; the first four at {times}.",
                     "lots": "Your rule goes off at least {k} times; the first four at {times}.",
                     "missed": "The slowdown from 14:00 to 14:25 sets off no alert.",
                     "never": "Your rule never goes off: the slowdown from 14:00 to 14:25 goes unseen.",
                     "none": "No picture yet: your `alert` does not give a minute of the list, or `None`, for every call.",
                     "alt": "The p99 of each minute of 14 July 2011 on a scale of milliseconds, a level line at 300 ms, and a dot at the top for each minute at which the learner's rule goes off."}},
    "nl": {"tail": {"title": "Je tail op elke minuut van 14 juli 2011",
                    "intro": "Je `tail(lines)` op elk van de 1.440 minuten van de dag. De schaal telt milliseconden: elke lijn is tien keer zoveel als de lijn eronder.",
                    "key": ("je p99", "je p50"),
                    "minutes": "In een rustige minuut zoals 09:00 is je p99 {quiet} ms; om 09:12 {spike} ms; om 14:05 {slow} ms.",
                    "day": "Over de hele dag is je p99 {p99} ms: {share} van de {n} verzoeken duurde langer.",
                    "same": "Dit zijn de getallen van de course: Oefening 2 houdt deze getallen in de gaten.",
                    "none": "Nog geen afbeelding: je `tail` geeft niet voor elke minuut een dict met een getal onder `p50` en onder `p99`.",
                    "alt": "Twee lijnen over de 1.440 minuten van 14 juli 2011 op een schaal van milliseconden: de p99 van elke minuut volgens de leerling, en de p50 eronder."},
           "alert": {"title": "Je alert op 14 juli 2011",
                     "intro": "Je `alert(p99s, limit, minutes)` op de p99 van elke minuut, met een grens van 300 ms gedurende 5 minuten. Na elke alert kan de volgende pas komen als de p99 weer onder de grens is.",
                     "key": ("p99", "de grens", "alert voor de vertraging", "alert voor niets"),
                     "count": "Je regel gaat {k} keer af: om {times}.",
                     "two": "Je regel gaat twee keer af: om {times}.",
                     "one": "Je regel gaat één keer af: om {times}.",
                     "many": "Je regel gaat {k} keer af; de eerste vier keer om {times}.",
                     "lots": "Je regel gaat minstens {k} keer af; de eerste vier keer om {times}.",
                     "missed": "De vertraging van 14:00 tot 14:25 laat geen alert afgaan.",
                     "never": "Je regel gaat nooit af: niemand ziet de vertraging van 14:00 tot 14:25.",
                     "none": "Nog geen afbeelding: je `alert` geeft niet bij elke aanroep een minuut van de lijst, of `None`.",
                     "alt": "De p99 van elke minuut van 14 juli 2011 op een schaal van milliseconden, een lijn op 300 ms, en bovenaan een stip voor elke minuut waarop de regel van de leerling afgaat."}},
}


def _share(p99: float, lang: str) -> str:
    return number(100 * float(np.mean(LATENCIES > p99)), 1, lang) + "%"


def tail_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["tail"]
    fn = _fn(seen)

    def run():
        return [fn(list(BY_MINUTE[m])) for m in range(DAY_MINUTES)], fn(list(LOG))

    got = _call(run) if fn is not None else _STOPPED
    stop = yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    if got is _STOPPED:
        return stop
    minutes, day = got
    p99 = [_ms_of(v, "p99") for v in minutes]
    p50 = [_ms_of(v, "p50") for v in minutes]
    day99 = _ms_of(day, "p99")
    if any(v is None for v in p99 + p50) or day99 is None:
        return stop
    same = all(abs(a - b) < 0.5 for a, b in zip(p99, P99)) and all(abs(a - b) < 0.5 for a, b in zip(p50, P50))
    colour = "var(--ea-data)" if same else "var(--ea-loss)"
    svg = day_chart(lang, w["alt"], [("var(--ea-ink-3)", p50, 1), (colour, p99, 1.2)])
    said = (w["minutes"].format(quiet=number(p99[9 * 60], 0, lang), spike=number(p99[PAUSE], 0, lang),
                                slow=number(p99[SLOW[0] + 5], 0, lang))
            + " " + w["day"].format(p99=number(day99, 0, lang), share=_share(day99, lang), n=number(len(LOG), 0, lang)))
    body = (_note(w["intro"]) + _key([("solid", colour, w["key"][0]), ("solid", "var(--ea-ink-3)", w["key"][1])]) + svg
            + _note(said) + (_note(w["same"]) if same else ""))
    return yours.figure(w["title"], body, alt=w["title"])


ALERT_WORDS = {
    "en": {"slow": "{time} ({after} into the slowdown)", "start": "{time} (as the slowdown begins)",
           "nothing": "{time} (for nothing)", "minute": "1 minute", "minutes": "{m} minutes"},
    "nl": {"slow": "{time} ({after} na het begin van de vertraging)", "start": "{time} (bij het begin van de vertraging)",
           "nothing": "{time} (voor niets)", "minute": "1 minuut", "minutes": "{m} minuten"},
}


def _after(m: int, lang: str) -> str:
    words = ALERT_WORDS[lang]
    return words["minute"] if m == 1 else words["minutes"].format(m=m)


def _alert_said(found: list, lang: str, w: dict, most: int | None = None) -> str:
    """The alerts in words: when each went off, and what for; the first four when there are many, and "at least" when
    the list stopped at `most`."""
    if not found:
        return w["never"]
    words = ALERT_WORDS[lang]
    if len(found) > 3:
        many = "lots" if most is not None and len(found) >= most else "many"
        said = w[many].format(k=len(found), times=_list_of([clock(m) for m in found[:4]], lang))
    else:
        times = []
        for m in found:
            if m == SLOW[0]:
                times.append(words["start"].format(time=clock(m)))
            elif SLOW[0] < m < SLOW[1]:
                times.append(words["slow"].format(time=clock(m), after=_after(m - SLOW[0], lang)))
            else:
                times.append(words["nothing"].format(time=clock(m)))
        said = w[{1: "one", 2: "two"}.get(len(found), "count")].format(k=len(found), times=_list_of(times, lang))
    return said + ("" if caught(found) is not None else " " + w["missed"])


def alert_chart(found: list, limit: float, lang: str, alt: str) -> str:
    marks = [(m, "var(--ea-ink)" if SLOW[0] <= m < SLOW[1] else "var(--ea-loss)") for m in found]
    return day_chart(lang, alt, [("var(--ea-data)", P99, 1)], limit=limit, marks=marks)


ALERTS_SHOWN = 40   # the learner's rule is followed over the day up to this many alerts


def alert_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["alert"]
    fn = _fn(seen)
    found = _call(lambda: alerts(fn, P99, LIMIT, MINUTES, most=ALERTS_SHOWN)) if fn is not None else _STOPPED
    if found is _STOPPED or found is None:
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    key = [("solid", "var(--ea-data)", w["key"][0]), ("dashed", "var(--ea-ink)", w["key"][1]),
           ("dot", "var(--ea-ink)", w["key"][2]), ("dot", "var(--ea-loss)", w["key"][3])]
    body = (_note(w["intro"]) + _key(key) + alert_chart(found, LIMIT, lang, w["alt"])
            + _note(_alert_said(found, lang, w, most=ALERTS_SHOWN)))
    return yours.figure(w["title"], body, alt=w["title"])


# ======== the demo: an alert rule tuned on the day ========================================================================

DEMO_TEXT = {
    "en": {"title": "An alert rule: p99 above {limit} ms for {minutes} minutes",
           "title_one": "An alert rule: p99 above {limit} ms in one minute",
           "key": ("p99", "the limit", "alert for the slowdown", "alert for nothing"),
           "count": "The rule goes off {k} times: at {times}.",
           "two": "The rule goes off twice: at {times}.",
           "one": "The rule goes off once: at {times}.",
           "many": "The rule goes off {k} times; the first four at {times}.",
           "lots": "The rule goes off at least {k} times; the first four at {times}.",
           "missed": "The slowdown from 14:00 to 14:25 sets off no alert.",
           "never": "The rule never goes off: the slowdown from 14:00 to 14:25 goes unseen.",
           "tally": "For nothing: {needless}. The slowdown: {caught}.",
           "caught": "caught after {after}",
           "at_once": "caught at once",
           "not_caught": "not caught",
           "alt": "The p99 of each minute of 14 July 2011 on a scale of milliseconds, a level line at the chosen limit, and a dot at the top for each minute at which the rule goes off."},
    "nl": {"title": "Een alertregel: p99 boven {limit} ms gedurende {minutes} minuten",
           "title_one": "Een alertregel: p99 boven {limit} ms in één minuut",
           "key": ("p99", "de grens", "alert voor de vertraging", "alert voor niets"),
           "count": "De regel gaat {k} keer af: om {times}.",
           "two": "De regel gaat twee keer af: om {times}.",
           "one": "De regel gaat één keer af: om {times}.",
           "many": "De regel gaat {k} keer af; de eerste vier keer om {times}.",
           "lots": "De regel gaat minstens {k} keer af; de eerste vier keer om {times}.",
           "missed": "De vertraging van 14:00 tot 14:25 laat geen alert afgaan.",
           "never": "De regel gaat nooit af: niemand ziet de vertraging van 14:00 tot 14:25.",
           "tally": "Voor niets: {needless}. De vertraging: {caught}.",
           "caught": "gezien na {after}",
           "at_once": "meteen gezien",
           "not_caught": "niet gezien",
           "alt": "De p99 van elke minuut van 14 juli 2011 op een schaal van milliseconden, een lijn op de gekozen grens, en bovenaan een stip voor elke minuut waarop de regel afgaat."},
}
LIMITS = tuple(range(100, 650, 50))
DEMO_MOST = 200   # more than any setting of the demo gives
MINUTE_CHOICES = tuple(range(1, 11))


def demo_picture(limit, minutes, lang: str) -> str:
    """The demo: the course's alert on the day's p99, with the chosen limit and number of minutes."""
    lang = _lang(lang)
    w = DEMO_TEXT[lang]
    limit = limit if limit in LIMITS else LIMIT
    minutes = minutes if minutes in MINUTE_CHOICES else 1
    found = alerts(alert, P99, limit, minutes, most=DEMO_MOST)
    key = [("solid", "var(--ea-data)", w["key"][0]), ("dashed", "var(--ea-ink)", w["key"][1]),
           ("dot", "var(--ea-ink)", w["key"][2]), ("dot", "var(--ea-loss)", w["key"][3])]
    m = caught(found)
    tally = w["tally"].format(needless=len(needless(found)),
                              caught=w["not_caught"] if m is None else (w["at_once"] if m == 0 else
                                                                       w["caught"].format(after=_after(m, lang))))
    title = (w["title_one"] if minutes == 1 else w["title"]).format(limit=number(limit, 0, lang), minutes=minutes)
    body = (_key(key) + alert_chart(found, limit, lang, w["alt"]) + _note(_alert_said(found, lang, w, most=DEMO_MOST))
            + _note(tally))
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
    """{got} and {want}: what the function gives back, as Python writes it."""
    text_ = repr(v)
    return f"`{text_}`" if "`" not in text_ else f"<code>{html.escape(text_)}</code>"


WATCH = Sort(
    name="watch",
    groups={
        "service": {"en": "The service: does it answer?", "nl": "De service: komt er antwoord?"},
        "model": {"en": "The model: are its answers any good?", "nl": "Het model: deugen de antwoorden?"},
    },
    items={
        "traffic": {"en": "**Traffic.** The number of requests per minute.",
                    "nl": "**Verkeer.** Het aantal verzoeken per minuut."},
        "answers": {"en": "**Answers.** The mean of `bikes` per hour, next to the same hour on other days.",
                    "nl": "**Antwoorden.** Het gemiddelde van `bikes` per uur, naast hetzelfde uur op andere dagen."},
        "errors": {"en": "**Errors.** The share of requests answered with status 500.",
                   "nl": "**Fouten.** Het deel van de verzoeken met status 500 als antwoord."},
        "inputs": {"en": "**Inputs.** The values of `temp` in the requests, next to those of training.",
                   "nl": "**Inputs.** De waarden van `temp` in de verzoeken, naast de waarden uit de training."},
        "tail": {"en": "**Tail latency.** The p99 of the latency of each minute.",
                 "nl": "**Tail latency.** De p99 van de latency van elke minuut."},
        "zeros": {"en": "**Zeros.** The share of answers with 0 bikes.",
                  "nl": "**Nullen.** Het deel van de antwoorden met 0 fietsen."},
        "memory": {"en": "**Memory.** How much of the memory of the server is in use.",
                   "nl": "**Geheugen.** Hoeveel van het geheugen van de server in gebruik is."},
        "off": {"en": "**How far off.** The bikes the service answered, next to the bikes rented, once the hour is over.",
                "nl": "**Hoe ver ernaast.** De fietsen in het antwoord van de service, naast de verhuurde fietsen, zodra het uur voorbij is."},
    },
    solution={"traffic": "service", "answers": "model", "errors": "service", "inputs": "model",
              "tail": "service", "zeros": "model", "memory": "service", "off": "model"},
    nudge={
        "traffic": {"en": "Does the number of requests say anything about whether the bikes in the answers are right?",
                    "nl": "Zegt het aantal verzoeken iets over de vraag of de fietsen in de antwoorden kloppen?"},
        "errors": {"en": "A 500 means that the server broke on a request. Is that about the answer of the model, or about getting an answer at all?",
                   "nl": "Een 500 betekent dat de server stukging op een verzoek. Gaat dat over het antwoord van het model, of over de vraag of er een antwoord komt?"},
        "tail": {"en": "A slow answer can still hold the right number of bikes. Which side is it about, then?",
                 "nl": "Een traag antwoord kan nog steeds het goede aantal fietsen bevatten. Over welke kant gaat het dan?"},
        "memory": {"en": "A server with full memory gets slow or stops. Is that about the model's numbers, or about the service?",
                   "nl": "Een server met een vol geheugen wordt traag of stopt. Gaat dat over de getallen van het model, of over de service?"},
        "answers": {"en": "At 17:00, answers of 1,000 bikes come back with status 200, as fast as ever. Which side does their mean say something about?",
                    "nl": "Om 17:00 komen antwoorden van 1.000 fietsen terug met status 200, even snel als altijd. Over welke kant zegt het gemiddelde ervan iets?"},
        "zeros": {"en": "An answer of 0 bikes is a well-formed answer. What could many of them say about the model?",
                  "nl": "Een antwoord van 0 fietsen is een goed gevormd antwoord. Wat zouden veel van zulke antwoorden zeggen over het model?"},
        "inputs": {"en": "The server takes any number as `temp`. Who gets into trouble when the numbers differ from training?",
                   "nl": "De server neemt elk getal aan als `temp`. Wie komt in de problemen als de getallen afwijken van de training?"},
        "off": {"en": "This compares the answers with what really happened. Does it watch the server, or the model?",
                "nl": "Dit vergelijkt de antwoorden met wat er echt gebeurde. Houdt het de server in de gaten, of het model?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the numbers again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de getallen opnieuw uit."},
        "empty": {"en": "Put each number with the question it answers.",
                  "nl": "Zet elk getal bij de vraag waarop het een antwoord geeft."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "*{item}* belongs with the other question.",
                  "nl": "*{item}* hoort bij de andere vraag."},
        "correct": {"en": "All {n} are right. Traffic, errors, tail latency and memory say whether the service answers, and how fast; answers, zeros, inputs and how far off say whether its answers are any good. A model can be wrong while the service looks fine.",
                    "nl": "Alle {n} kloppen. Verkeer, fouten, tail latency en geheugen zeggen of de service antwoordt, en hoe snel; antwoorden, nullen, inputs en hoe ver ernaast zeggen of de antwoorden deugen. Een model kan het mis hebben terwijl de service er goed uitziet."},
    },
)

EXERCISES = {
    "tail": FunctionBox(
        name="tail",
        function="tail",
        start=_T_START,
        solution=_T,
        names=_T_NAMES,
        visible=[(at("09:00"),)],
        hidden=[(at("09:12"),), (at("14:05"),), (at("03:00"),), (at("17:30"),)],
        first=[_DOTS],
        show=_shown,
        picture=tail_picture,
        mistakes=[
            Mistake("percent", error=_quantile_range,
                    en="With `{call}`, NumPy stops: `np.quantile` wants the share of the values that lie below, as a number from 0 to 1, not a percentage. Which share is 99 in 100?",
                    nl="Met `{call}` stopt NumPy: `np.quantile` wil het deel van de waarden dat eronder ligt, als een getal van 0 tot 1, geen percentage. Welk deel is 99 van de 100?"),
            Mistake("percentile", matches=lambda got, text: re.search(r"np\.percentile\(\s*ms\s*,\s*0?\.\d", str(text)) is not None,
                    en="`{call}` gives {got}: `np.percentile` takes a percentage from 0 to 100, so your number is less than 1 percent, close to the fastest request, while `np.quantile` takes a share from 0 to 1. Which percentage leaves 1 in 100 requests above it, or which function takes a share?",
                    nl="`{call}` geeft {got}: `np.percentile` krijgt een percentage van 0 tot 100, dus is je getal nog geen 1 procent, vlak bij het snelste verzoek, terwijl `np.quantile` een deel van 0 tot 1 krijgt. Welk percentage laat 1 op de 100 verzoeken erboven, of welke functie krijgt een deel?"),
            Mistake("fastest", wrong=_T_FASTEST,
                    en="`{call}` gives {got}: your p99 is lower than the p50, since only 1 in 100 requests was faster. Which share of the requests lies below the p99?",
                    nl="`{call}` geeft {got}: je p99 is lager dan de p50, want alleen 1 op de 100 verzoeken was sneller. Welk deel van de verzoeken ligt onder de p99?"),
            Mistake("p90", wrong=_T_P90,
                    en="`{call}` gives {got}: that is the p90, below which 9 in 10 requests lie, so 1 in 10 was slower. Which share leaves 1 in 100 above it?",
                    nl="`{call}` geeft {got}: dat is de p90, met 9 van de 10 verzoeken eronder, dus was 1 op de 10 trager. Welk deel laat 1 op de 100 erboven?"),
            Mistake("p95", wrong=_T_P95,
                    en="`{call}` gives {got}: that is the p95, so 1 in 20 requests was slower. Which share leaves 1 in 100 above it?",
                    nl="`{call}` geeft {got}: dat is de p95, dus was 1 op de 20 verzoeken trager. Welk deel laat 1 op de 100 erboven?"),
            Mistake("max", wrong=_T_MAX,
                    en="`{call}` gives {got}: that is the slowest request itself, while the p99 lies a little below it, with 99 in 100 requests under it. Which function gives the value below which a share of the values lies?",
                    nl="`{call}` geeft {got}: dat is het traagste verzoek zelf, terwijl de p99 er net onder ligt, met 99 van de 100 verzoeken eronder. Welke functie geeft de waarde met een deel van de waarden eronder?"),
            Mistake("mean", wrong=_T_MEAN,
                    en="`{call}` gives {got}: the mean, which says nothing about how slow the slowest requests were. Which function gives the value below which a share of the values lies?",
                    nl="`{call}` geeft {got}: het gemiddelde, dat niets zegt over hoe traag de traagste verzoeken waren. Welke functie geeft de waarde met een deel van de waarden eronder?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with what `p99` should be: the latency below which 99 in 100 requests of the minute lie.",
                      "nl": "Vervang de `...` door wat `p99` moet zijn: de latency met 99 van de 100 verzoeken van de minuut eronder."},
            "arguments": {"en": "The check calls `{call}`, with one list of lines, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(lines)`?",
                          "nl": "De check roept `{call}` aan, met één list van regels, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(lines)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is `p99` the value below which 99 in 100 latencies of `ms` lie?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Is `p99` de waarde met 99 van de 100 latencies uit `ms` eronder?"},
            "almost": {"en": "Right for 09:00. But `{call}` gives {got}, and it should give {want}. Does `p99` come from `ms`, and not from a fixed number?",
                       "nl": "Goed voor 09:00. Maar `{call}` geeft {got}, en het moet {want} zijn. Komt `p99` uit `ms`, en niet uit een vast getal?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with the `return` of the dict?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met de `return` van de dict?"},
            "correct": {"en": "Right in all {n} cases. At 09:00, the p99 is about two and a half times the median; at 09:12, one request of more than a second pulls it far up. In the picture: what happens to your p99 from 14:00 to 14:25?",
                        "nl": "Goed in alle {n} gevallen. Om 09:00 is de p99 ongeveer tweeënhalf keer de mediaan; om 09:12 trekt één verzoek van meer dan een seconde de p99 ver omhoog. In de afbeelding: wat doet je p99 van 14:00 tot 14:25?"},
        },
    ),
    "alert": FunctionBox(
        name="alert",
        function="alert",
        start=_A_START,
        solution=_A,
        names=_A_NAMES,
        visible=[([120, 950, 130, 340, 360, 380, 410, 390], 300, 5)],
        hidden=[(P99S, 300, 5), (P99S, 300, 1), (P99S, 200, 10), ([400, 100, 100, 100, 100, 100], 300, 3),
                ([100, 100, 310], 300, 3), ([310, 320], 300, 3), ([300, 300, 300, 300], 300, 3)],
        first=[_DOTS],
        show=_shown,
        picture=alert_picture,
        mistakes=[
            Mistake("max", wrong=_A_MAX,
                    en="`{call}` gives {got}: your alert goes off as soon as one of the last minutes is above the limit, so one spike of 950 sets it off. Which number of `last` is above `limit` only when every minute is?",
                    nl="`{call}` geeft {got}: je alert gaat af zodra één van de laatste minuten boven de grens zit, dus laat één piek van 950 de alert al afgaan. Welk getal van `last` ligt alleen boven `limit` als elke minuut erboven ligt?"),
            Mistake("mean", wrong=_A_MEAN,
                    en="`{call}` gives {got}: the mean of the last 5 minutes is 380, pulled up by the one minute of 950, as one long trip pulls up a mean. Which number of `last` is above `limit` only when every minute is?",
                    nl="`{call}` geeft {got}: het gemiddelde van de laatste 5 minuten is 380, omhooggetrokken door de ene minuut van 950, zoals één lange rit een gemiddelde omhoogtrekt. Welk getal van `last` ligt alleen boven `limit` als elke minuut erboven ligt?"),
            Mistake("newest", wrong=_A_NEWEST,
                    en="`{call}` gives {got}: your rule looks only at the newest minute of `last`, so one minute above the limit is enough to set it off. Which number of `last` is above `limit` only when every minute is?",
                    nl="`{call}` geeft {got}: je regel kijkt alleen naar de nieuwste minuut van `last`, dus is één minuut boven de grens al genoeg om de regel te laten afgaan. Welk getal van `last` ligt alleen boven `limit` als elke minuut erboven ligt?"),
            Mistake("at-limit", wrong=_A_AT,
                    en="`{call}` gives {got}: your alert already goes off when every minute lies exactly at the limit, while the rule says above it. Which comparison keeps a minute of exactly `limit` quiet?",
                    nl="`{call}` geeft {got}: je alert gaat al af als elke minuut precies op de grens ligt, terwijl de regel zegt: erboven. Welke vergelijking houdt een minuut van precies `limit` stil?"),
            Mistake("oldest", wrong=_A_OLDEST,
                    en="`{call}` gives {got}: your rule looks only at the oldest minute of `last`, the spike of 950. Which number of `last` is above `limit` only when every minute is?",
                    nl="`{call}` geeft {got}: je regel kijkt alleen naar de oudste minuut van `last`, de piek van 950. Welk getal van `last` ligt alleen boven `limit` als elke minuut erboven ligt?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the condition of the alert: every minute in `last` is above `limit`.",
                      "nl": "Vervang de `...` door de voorwaarde van de alert: elke minuut in `last` ligt boven `limit`."},
            "arguments": {"en": "The check calls `{call}`, with a list and two numbers, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(p99s, limit, minutes)`?",
                          "nl": "De check roept `{call}` aan, met een list en twee getallen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(p99s, limit, minutes)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does the alert go off only when every minute in `last` is above `limit`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Gaat de alert alleen af als elke minuut in `last` boven `limit` ligt?"},
            "almost": {"en": "Right for the short list. But `{call}` gives {got}, and it should give {want}. Does your condition use `last` and `limit`, and not fixed numbers?",
                       "nl": "Goed voor de korte list. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je voorwaarde `last` en `limit`, en geen vaste getallen?"},
            "unreached": {"en": "With `{call}`, your function gives `None`: Python never gets to `return i`. Is your condition true when every minute in `last` is above `limit`, and is `return i` still under it?",
                          "nl": "Met `{call}` geeft je functie `None`: Python komt nooit bij `return i`. Klopt je voorwaarde als elke minuut in `last` boven `limit` ligt, en staat `return i` er nog onder?"},
            "correct": {"en": "Right in all {n} cases. The list gives 7: the spike of 950 sets nothing off, and the alert goes off once the number has stayed above 300 for 5 minutes. In the picture: when does your rule go off on 14 July?",
                        "nl": "Goed in alle {n} gevallen. De list geeft 7: de piek van 950 laat de alert niet afgaan, en de alert gaat af zodra het getal 5 minuten boven 300 is gebleven. In de afbeelding: wanneer gaat je regel af op 14 juli?"},
        },
    ),
    "watch": WATCH,
}

REQUIRED = ["tail", "alert", "model", "watch"]

_SORTED = dict(WATCH.solution)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("tail", _T_START, "empty", None),
    ("tail", "", "empty", None),
    ("tail", _T, "correct", None),
    ("tail", _t("float(np.percentile(ms, 99))"), "correct", None),
    ("tail", _t("np.quantile(ms, 0.99)"), "correct", None),
    ("tail", _T_PERCENT, "error", "percent"),
    ("tail", _T_FASTEST, "wrong", "fastest"),
    ("tail", _t("float(np.percentile(ms, 0.99))"), "wrong", "percentile"),
    ("tail", _t("np.percentile(ms, .99)"), "wrong", "percentile"),
    ("tail", _T_P90, "wrong", "p90"),
    ("tail", _T_P95, "wrong", "p95"),
    ("tail", _T_MAX, "wrong", "max"),
    ("tail", _t("max(ms)"), "wrong", "max"),
    ("tail", _T_MEAN, "wrong", "mean"),
    ("tail", _t("float(np.median(ms))"), "wrong", None),
    ("tail", _t("55"), "almost", None),
    ("tail", _T.replace(_T_END, "    ...\n" + _T_END), "wrong", "dots-left"),
    ("tail", _T.replace(_T_END, ""), "wrong", "no-return"),
    ("tail", _T.replace("def tail(lines):", "def tail():"), "error", "arguments"),
    ("tail", "x = 1\n", "wrong", "no-function"),
    ("tail", _T.replace(_T_END, "    print({\"p50\": round(p50), \"p99\": round(p99)})\n"), "wrong", "prints"),
    ("alert", _A_START, "empty", None),
    ("alert", "", "empty", None),
    ("alert", _A, "correct", None),
    ("alert", _a("all(p > limit for p in last)"), "correct", None),
    ("alert", _a("sorted(last)[0] > limit"), "correct", None),
    ("alert", _A_MAX, "wrong", "max"),
    ("alert", _A_MEAN, "wrong", "mean"),
    ("alert", _A_NEWEST, "wrong", "newest"),
    ("alert", _A_OLDEST, "wrong", "oldest"),
    ("alert", _a("min(last) > 250"), "almost", None),
    ("alert", _a("min(last) < limit"), "wrong", None),
    ("alert", _A_AT, "wrong", "at-limit"),
    ("alert", _a("all(p >= limit for p in last)"), "wrong", "at-limit"),
    ("alert", _a("min(last) > 5000"), "wrong", "unreached"),
    ("alert", _A.replace("        if min(last) > limit:\n", "        ...\n        if min(last) > limit:\n"), "wrong", "dots-left"),
    ("alert", _A.replace("            return i ", "            pass     "), "wrong", "unreached"),
    ("alert", _A.replace("            return i ", "            print(i) "), "wrong", "prints"),
    ("alert", _A.replace("def alert(p99s, limit, minutes):", "def alert(p99s, limit):"), "error", "arguments"),
    ("alert", "x = 1\n", "wrong", "no-function"),
    ("watch", {}, "empty", None),
    ("watch", _SORTED, "correct", None),
    ("watch", {"traffic": "service", "answers": "model"}, "wrong", "unfinished"),
] + [("watch", _SORTED | {item: "model" if side == "service" else "service"}, "wrong", f"wrong:{item}")
     for item, side in WATCH.solution.items()]
