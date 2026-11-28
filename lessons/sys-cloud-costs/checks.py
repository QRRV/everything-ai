# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-cloud-costs: what compute costs in the cloud. One GPU at three prices, on demand,
reserved and spot; how many hours a spot run pays for when the cloud takes the machine back and each stop throws
away the work since the last checkpoint; from how many requests a day serverless inference costs more than a fixed
server; which way of paying fits which job; and, in the demo, how often to save a checkpoint.

The prices are examples made up by the course, of the size the large clouds asked in November 2026: 4 dollars an
hour on demand, 40% off reserved for a year, 70% off as spot. The discounts the lesson names come from the providers'
own pages, read on 28 November 2026: AWS spot up to 90% off with a two-minute warning, Google Cloud Spot VMs 60 to
91% off with a warning of 30 seconds, AWS Savings Plans up to 72% off for three years."""
import html
import math
from functools import cache

from everything_ai import check, yours
from everything_ai.check import FunctionBox, Mistake, Sort

LESSON = "sys-cloud-costs"
LANGS = ("en", "nl")

ON_DEMAND = 4.00   # dollars per GPU hour, for each hour you use
RESERVED = 2.40    # dollars per GPU hour, 40% off, but for all hours of a month, a year long
SPOT = 1.20        # dollars per GPU hour, 70% off, and the cloud can take the machine back
MONTH = 730        # hours in a month: 8,760 hours in a year, divided by 12

HOURS = 100        # the training run of the guess and the demo, in hours of training
GAP = 10           # on average, the cloud takes a spot machine back once every 10 paid hours
SAVE = 0.05        # hours to save a checkpoint: 3 minutes
RESTART = 0.2      # hours to start a new machine and load the last checkpoint: 12 minutes

PER_REQUEST = 0.00035   # dollars per request on serverless: about 1.5 seconds of a GPU
SERVER_DAY = 18.0       # dollars a day for a small fixed GPU server: 24 hours at 0.75 dollars


class _Named(list):
    """A list from the lesson: the check's call shows it by its name, and the learner's function gets a copy."""

    def __init__(self, name: str, values):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return list(self)


# the hours of training done at each moment the cloud took the machine back, in a run of 100 hours
STOPS = _Named("C.STOPS", [9.5, 17.25, 26, 35.5, 43.75, 52, 61.5, 69.25, 78, 87.5])


def hours_paid(hours, every, stops):
    paid = hours
    for stop in stops:
        saved = 0
        while saved + every <= stop:
            saved = saved + every
        paid = paid + stop - saved
    return paid


def break_even(per_request, per_day):
    n = 0
    while n * per_request <= per_day:
        n = n + 1
    return n


def reserved_from() -> int:
    """The worked example: the first number of hours a month at which reserved is cheaper than on demand."""
    reserved = RESERVED * MONTH
    hours = 0
    while hours * ON_DEMAND <= reserved:
        hours = hours + 1
    return hours


def spot_parts(every: float) -> dict:
    """What a spot run of HOURS hours pays for, on average, with a checkpoint every `every` hours of training.
    Each stop throws away half an interval on average, and a new machine takes RESTART hours; the stops come once
    every GAP paid hours, so paid = HOURS + saving + paid / GAP * (every / 2 + RESTART)."""
    save = HOURS * SAVE / every
    paid = (HOURS + save) / (1 - (every / 2 + RESTART) / GAP)
    stops = paid / GAP
    return {"paid": paid, "train": float(HOURS), "save": save, "lost": stops * every / 2,
            "restart": stops * RESTART, "stops": stops, "cost": paid * SPOT}


EVERY = [1, 2, 4, 8]                       # the checkpoint intervals of the picture under exercise 1
DEMO_VALUES = [0.25, 0.5, 1, 2, 4, 8]      # the checkpoint intervals of the demo


@cache
def facts() -> dict:
    """The numbers the texts give."""
    out = {"reserved_month": RESERVED * MONTH, "reserved_from": reserved_from(),
           "on_demand_run": HOURS * ON_DEMAND, "on_demand_month": ON_DEMAND * MONTH, "spot_month": SPOT * MONTH,
           "break_even": break_even(PER_REQUEST, SERVER_DAY)}
    for e in EVERY:
        out[("paid", e)] = hours_paid(HOURS, e, list(STOPS))
    for e in DEMO_VALUES:
        out[("spot", e)] = spot_parts(e)
    return out


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem;max-width:var(--ea-measure)"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem;max-width:var(--ea-measure)"
_SVG = "width:100%;max-width:330px;height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums"
_OUTLINE = "fill:var(--ea-page);stroke:var(--ea-ink-3);stroke-width:1"


def number(x: float, lang: str) -> str:
    """A number with only the decimals it needs, at most two: 110.25, 135.48, 18; in Dutch with a decimal comma."""
    out = f"{round(x, 2):g}" if abs(x) < 1e6 else f"{x:.0f}"
    return out.replace(".", ",") if lang == "nl" else out


def one(x: float, lang: str) -> str:
    """A number with one decimal: 112.9; in Dutch 112,9."""
    out = f"{x:.1f}"
    return out.replace(".", ",") if lang == "nl" else out


def four(x: float, lang: str) -> str:
    """A number of dollars with four decimals, to tell 17.9998 from 18.0002."""
    out = f"{x:.4f}"
    return out.replace(".", ",") if lang == "nl" else out


UNIT = {"en": "h", "nl": "u"}   # hours, short
EVERY_TEXT = {"en": {"one": "every hour", "more": "every {e} hours"}, "nl": {"one": "elk uur", "more": "elke {e} uur"}}


def every_text(e: float, lang: str) -> str:
    """'every hour', 'every 2 hours'; in Dutch 'elk uur', 'elke 2 uur'."""
    w = EVERY_TEXT[lang]
    return w["one"] if e == 1 else w["more"].format(e=number(e, lang))


def whole(x: float, lang: str) -> str:
    """A whole number with thousands: 51,429; in Dutch 51.429."""
    out = f"{round(x):,}"
    return out.replace(",", ".") if lang == "nl" else out


def _lang(lang) -> str:
    return lang if lang in LANGS else "en"


def _marked(text_: str) -> str:
    """Escaped text in which `name` is set as code."""
    return "".join(f"<code>{part}</code>" if k % 2 else part for k, part in enumerate(html.escape(text_).split("`")))


def _note(text_: str, style: str = _UI) -> str:
    return f'<div style="{style}">{_marked(text_)}</div>'


def _t(x: float, y: float, text_: str, size: float = 12.5, weight: int = 400, fill: str = "var(--ea-ink)",
       anchor: str = "start") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" style="font-size:{size}px;'
            f'font-weight:{weight};fill:{fill}">{html.escape(text_)}</text>')


def _svg(height: float, alt: str, parts: list) -> str:
    return (f'<svg viewBox="0 0 330 {height:.0f}" role="img" aria-label="{html.escape(alt)}" style="{_SVG}">'
            f'{"".join(parts)}</svg>')


def _line(x1: float, y1: float, x2: float, y2: float, stroke: str = "var(--ea-line)", width: float = 1,
          dash: str = "") -> str:
    d = f";stroke-dasharray:{dash}" if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'style="stroke:{stroke};stroke-width:{width}{d}"/>')


def _rect(x: float, y: float, w: float, h: float, style: str) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{h:.1f}" rx="2" style="{style}"/>'


def _dot(x: float, y: float, r: float, fill: str) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" style="fill:{fill}"/>'


# ======== the picture of the idea: what a month costs on demand, reserved and as spot ======================================

IDEA_TEXT = {
    "en": {"x": "GPU hours used in a month", "y": "dollars a month",
           "od": "on demand", "res": "reserved", "spot": "spot",
           "alt": "Three lines of what one GPU costs in a month against the hours it is used. On demand rises from 0 "
                  "to 2,920 dollars at all 730 hours. Reserved stays at 1,752 dollars, used or not. Spot rises from 0 to "
                  "876 dollars, before the hours a stop throws away. On demand and reserved cross at 438 hours."},
    "nl": {"x": "GPU-uren gebruikt in een maand", "y": "dollar per maand",
           "od": "on-demand", "res": "reserved", "spot": "spot",
           "alt": "Drie lijnen van wat één GPU in een maand kost tegen de uren gebruik. On-demand stijgt van 0 tot "
                  "2.920 dollar bij alle 730 uur. Reserved blijft op 1.752 dollar, gebruikt of niet. Spot stijgt van 0 "
                  "tot 876 dollar, zonder de uren die een onderbreking weggooit. On-demand en reserved kruisen bij 438 uur."},
}


def idea_picture(lang: str) -> str:
    lang = _lang(lang)
    w = IDEA_TEXT[lang]
    x0, x1, top, bottom = 40, 246, 22, 168
    ymax = 3000

    def X(h):
        return x0 + (x1 - x0) * h / MONTH

    def Y(d):
        return bottom - (bottom - top) * d / ymax

    parts = [_t(0, 12, w["y"], 11, 600, "var(--ea-ink-2)")]
    for d in (0, 1000, 2000, 3000):
        parts.append(_line(x0, Y(d), x1, Y(d), "var(--ea-line)", 1))
        parts.append(_t(x0 - 5, Y(d) + 4, whole(d, lang), 10.5, 400, "var(--ea-ink-3)", "end"))
    cross = RESERVED * MONTH / ON_DEMAND
    for h in (0, cross, MONTH):
        parts.append(_t(X(h), bottom + 15, whole(h, lang), 10.5, 400, "var(--ea-ink-3)", "middle"))
    parts.append(_t((x0 + x1) / 2, bottom + 31, w["x"], 11, 400, "var(--ea-ink-2)", "middle"))
    parts.append(_line(X(cross), Y(RESERVED * MONTH), X(cross), bottom, "var(--ea-ink-3)", 1, "3 3"))
    lines = (("od", ON_DEMAND * MONTH, "var(--ea-ink)", True), ("res", RESERVED * MONTH, "var(--ea-ink-2)", False),
             ("spot", SPOT * MONTH, "var(--ea-ink-3)", True))
    for key, end, stroke, rises in lines:
        parts.append(_line(x0, Y(0 if rises else end), x1, Y(end), stroke, 2.5))
        parts.append(_t(x1 + 6, Y(end) - 1, w[key], 11.5, 600, stroke)
                     + _t(x1 + 6, Y(end) + 12, whole(end, lang), 10.5, 400, "var(--ea-ink-2)"))
    parts.append(_dot(X(cross), Y(RESERVED * MONTH), 3.5, "var(--ea-ink)"))
    return _svg(bottom + 38, w["alt"], parts)


# ======== the pictures under the checks ===================================================================================

YOURS_TEXT = {
    "en": {"paid": {"title": "Your hours_paid for four checkpoint intervals",
                    "intro": "The run of 100 hours with the stops of `C.STOPS`: the hours your `hours_paid` says the run "
                             "pays for, with a checkpoint every 1, 2, 4 and 8 hours.",
                    "every": "every {e} h", "train": "training", "lost": "lost",
                    "none": "Your `hours_paid` stops with an error, takes too long, or gives something other than a "
                            "number, so the picture waits.",
                    "right": "Your function is right for all four: the rarer the checkpoints, the more work each stop "
                             "throws away.",
                    "off": "With a checkpoint {e}, your function says {got} hours, and the run pays for {want}.",
                    "alt": "Bars for four checkpoint intervals: 100 hours of training and, on top, the hours the "
                           "learner's function says the stops throw away."},
           "even": {"title": "Your break_even against the costs per day",
                    "intro": "What a day costs at each number of requests: serverless at 0.00035 dollars a request, and "
                             "a fixed server at 18 dollars a day. The vertical line marks what your `break_even` gives.",
                    "serverless": "serverless", "server": "server", "x": "requests a day", "y": "dollars a day",
                    "mark": "yours: {n}",
                    "none": "Your `break_even` stops with an error, takes too long, or gives something other than a "
                            "number, so the picture waits.",
                    "right": "That is right: at {m} requests a day, serverless costs {b} dollars, and at {n} requests "
                             "{a} dollars, more than the 18 dollars of the server.",
                    "off": "At {n} requests a day, serverless costs {a} dollars. The first number at which serverless "
                           "costs more than the 18 dollars of the server is {m}.",
                    "alt": "Two lines of what a day costs against the requests a day: serverless rises from 0, the "
                           "server stays at 18 dollars. A vertical line marks the number the learner's function gives."}},
    "nl": {"paid": {"title": "Je hours_paid voor vier afstanden tussen checkpoints",
                    "intro": "De run van 100 uur met de onderbrekingen van `C.STOPS`: de uren die de run volgens je "
                             "`hours_paid` betaalt, met een checkpoint elke 1, 2, 4 en 8 uur.",
                    "every": "elke {e} u", "train": "training", "lost": "weg",
                    "none": "Je `hours_paid` stopt met een fout, duurt te lang, of geeft geen getal, dus wacht de "
                            "afbeelding nog.",
                    "right": "Je functie klopt voor alle vier: hoe zeldzamer de checkpoints, hoe meer werk elke "
                             "onderbreking weggooit.",
                    "off": "Met een checkpoint {e} zegt je functie {got} uur, en de run betaalt {want} uur.",
                    "alt": "Balken voor vier afstanden tussen checkpoints: 100 uur training en daarop de uren die de "
                           "onderbrekingen volgens de functie van de leerling weggooien."},
           "even": {"title": "Je break_even tegen de kosten per dag",
                    "intro": "Wat een dag kost bij elk aantal verzoeken: serverless voor 0,00035 dollar per verzoek, en "
                             "een vaste server voor 18 dollar per dag. De verticale lijn staat bij wat je `break_even` geeft.",
                    "serverless": "serverless", "server": "server", "x": "verzoeken per dag", "y": "dollar per dag",
                    "mark": "jouw getal: {n}",
                    "none": "Je `break_even` stopt met een fout, duurt te lang, of geeft geen getal, dus wacht de "
                            "afbeelding nog.",
                    "right": "Dat klopt: bij {m} verzoeken per dag kost serverless {b} dollar, en bij {n} verzoeken "
                             "{a} dollar, meer dan de 18 dollar van de server.",
                    "off": "Bij {n} verzoeken per dag kost serverless {a} dollar. Het eerste aantal waarbij serverless "
                           "meer kost dan de 18 dollar van de server is {m}.",
                    "alt": "Twee lijnen van wat een dag kost tegen het aantal verzoeken per dag: serverless stijgt "
                           "vanaf 0, de server blijft op 18 dollar. Een verticale lijn staat bij het getal dat de functie "
                           "van de leerling geeft."}},
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


def _amount(got):
    """got as a number of zero or more, or _STOPPED when it is not one."""
    if isinstance(got, bool) or not isinstance(got, (int, float)) or not math.isfinite(got) or got < 0:
        return _STOPPED
    return float(got)


def paid_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["paid"]
    fn = _fn(seen)
    got = {}
    for e in EVERY:
        v = _call(lambda: fn(HOURS, e, list(STOPS))) if fn else _STOPPED
        got[e] = _amount(v) if v is not _STOPPED else _STOPPED
    if any(v is _STOPPED for v in got.values()):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    f = facts()
    top = max(max(got.values()), max(f[("paid", e)] for e in EVERY))
    x0, W = 74, 186
    parts, y = [], 4
    for e in EVERY:
        hours = got[e]
        train = W * min(hours, HOURS) / top
        parts.append(_t(0, y + 14, w["every"].format(e=number(e, lang)), 11.5, 400, "var(--ea-ink-2)"))
        parts.append(_rect(x0, y + 3, max(train, 2.0), 15, "fill:var(--ea-ink)"))
        if hours > HOURS:
            parts.append(_rect(x0 + train, y + 3, W * (hours - HOURS) / top, 15, "fill:var(--ea-ink-3)"))
        end = x0 + max(W * hours / top, 2.0)
        parts.append(_t(end + 5, y + 14.5, f"{number(hours, lang)} {UNIT[lang]}", 11.5, 600))
        y += 24
    y += 14
    parts.append(_rect(0, y - 10, 16, 12, "fill:var(--ea-ink)") + _t(22, y, w["train"], 11.5, 400, "var(--ea-ink-2)")
                 + _rect(120, y - 10, 16, 12, "fill:var(--ea-ink-3)") + _t(142, y, w["lost"], 11.5, 400, "var(--ea-ink-2)"))
    wrong = [e for e in EVERY if abs(got[e] - f[("paid", e)]) > 1e-6]
    if not wrong:
        said = w["right"]
    else:
        e = wrong[0]
        said = w["off"].format(e=every_text(e, lang), got=number(got[e], lang), want=number(f[("paid", e)], lang))
    body = _note(w["intro"]) + _svg(y + 6, w["alt"], parts) + _note(said)
    return yours.figure(w["title"], body, alt=w["title"])


def even_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["even"]
    fn = _fn(seen)
    v = _call(lambda: fn(PER_REQUEST, SERVER_DAY)) if fn else _STOPPED
    n = _amount(v) if v is not _STOPPED else _STOPPED
    if n is _STOPPED:
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    x0, x1, top, bottom = 40, 300, 34, 160
    xmax, ymax = 100_000, 36

    def X(r):
        return x0 + (x1 - x0) * min(r, xmax) / xmax

    def Y(d):
        return bottom - (bottom - top) * min(d, ymax) / ymax

    parts = [_t(0, 12, w["y"], 11, 600, "var(--ea-ink-2)")]
    for d in (0, 18, 36):
        parts.append(_line(x0, Y(d), x1, Y(d), "var(--ea-line)", 1))
        parts.append(_t(x0 - 5, Y(d) + 4, str(d), 10.5, 400, "var(--ea-ink-3)", "end"))
    for r in (0, 50_000, 100_000):
        parts.append(_t(X(r), bottom + 15, whole(r, lang), 10.5, 400, "var(--ea-ink-3)", "middle"))
    parts.append(_t((x0 + x1) / 2, bottom + 31, w["x"], 11, 400, "var(--ea-ink-2)", "middle"))
    parts.append(_line(x0, Y(0), x1, Y(PER_REQUEST * xmax), "var(--ea-ink)", 2.5))
    parts.append(_line(x0, Y(SERVER_DAY), x1, Y(SERVER_DAY), "var(--ea-ink-3)", 2.5))
    at = 0.8 * xmax
    parts.append(_t(X(at) - 6, Y(PER_REQUEST * at) - 6, w["serverless"], 11.5, 600, "var(--ea-ink)", "end"))
    parts.append(_t(x1, Y(SERVER_DAY) - 6, w["server"], 11.5, 600, "var(--ea-ink-3)", "end"))
    mx = X(n)
    parts.append(_line(mx, top - 6, mx, bottom, "var(--ea-ink)", 1.5, "4 3"))
    anchor = "start" if mx < (x0 + x1) / 2 else "end"
    parts.append(_t(mx + (5 if anchor == "start" else -5), top - 8, w["mark"].format(n=whole(n, lang)), 11.5, 600,
                    "var(--ea-ink)", anchor))
    right = facts()["break_even"]
    if abs(n - right) < 1e-9:
        said = w["right"].format(m=whole(right - 1, lang), b=four((right - 1) * PER_REQUEST, lang),
                                 n=whole(right, lang), a=four(right * PER_REQUEST, lang))
    else:
        said = w["off"].format(n=whole(n, lang), a=four(n * PER_REQUEST, lang), m=whole(right, lang))
    body = _note(w["intro"]) + _svg(bottom + 38, w["alt"], parts) + _note(said)
    return yours.figure(w["title"], body, alt=w["title"])


# ======== the demo: how often to save a checkpoint =========================================================================

DEMO_TEXT = {
    "en": {"title": "A checkpoint {e}",
           "train": "training", "save": "saving", "lost": "lost", "restart": "starting again",
           "total": "{p} paid hours, {c} dollars as spot",
           "curve": "paid hours for each interval", "every": "hours between checkpoints",
           "say": "With a checkpoint {e}, the run of 100 hours pays for {p} hours: {s} to save checkpoints, "
                  "{l} lost at about {k} stops, and {r} to start again. As spot instances that costs {c} dollars; on "
                  "demand, 400 dollars.",
           "alt": "A bar of the paid hours of a spot run of 100 hours, split into training, saving checkpoints, work "
                  "lost at the stops, and starting again, and below it the paid hours for six checkpoint intervals, "
                  "lowest at one hour."},
    "nl": {"title": "Een checkpoint {e}",
           "train": "training", "save": "opslaan", "lost": "weg", "restart": "opnieuw starten",
           "total": "{p} betaalde uren, {c} dollar als spot",
           "curve": "betaalde uren per afstand", "every": "uren tussen checkpoints",
           "say": "Met een checkpoint {e} betaalt de run van 100 uur {p} uur: {s} om checkpoints op te slaan, "
                  "{l} weg bij ongeveer {k} onderbrekingen, en {r} om opnieuw te starten. Als spot instances kost dat "
                  "{c} dollar; on-demand 400 dollar.",
           "alt": "Een balk met de betaalde uren van een spot-run van 100 uur, verdeeld in training, checkpoints "
                  "opslaan, werk dat weg is bij de onderbrekingen, en opnieuw starten, en daaronder de betaalde uren "
                  "voor zes afstanden tussen checkpoints, het laagst bij één uur."},
}

_SEGMENTS = (("train", "fill:var(--ea-ink)"), ("save", "fill:var(--ea-ink-2)"), ("lost", "fill:var(--ea-ink-3)"),
             ("restart", _OUTLINE))


def frame(value: float, lang: str) -> tuple[str, str]:
    lang = _lang(lang)
    w = DEMO_TEXT[lang]
    e = float(value)
    p = spot_parts(e)
    parts = [_t(0, 15, w["title"].format(e=every_text(e, lang)), 13, 600)]
    scale = 330 / 180
    x, y = 0.0, 26
    for key, style in _SEGMENTS:
        width = p[key] * scale
        parts.append(_rect(x, y, width, 22, style))
        x += width
    parts.append(_t(0, y + 40, w["total"].format(p=one(p["paid"], lang), c=number(p["cost"], lang)), 12, 600))
    y += 60
    for k, (key, style) in enumerate(_SEGMENTS):
        lx, ly = (k % 2) * 165, y + (k // 2) * 18
        parts.append(_rect(lx, ly - 10, 16, 12, style)
                     + _t(lx + 22, ly, f"{w[key]} {one(p[key], lang)}", 11.5, 400, "var(--ea-ink-2)"))
    y += 46
    parts.append(_t(0, y, w["curve"], 11.5, 600, "var(--ea-ink-2)"))
    top, bottom = y + 16, y + 92
    lo, hi = 100, 180

    def Y(h):
        return bottom - (bottom - top) * (h - lo) / (hi - lo)

    xs = [28 + k * 54 for k in range(len(DEMO_VALUES))]
    ys = [Y(spot_parts(v)["paid"]) for v in DEMO_VALUES]
    parts.append(_line(xs[0] - 12, Y(lo), xs[-1] + 12, Y(lo), "var(--ea-line)", 1))
    parts.append('<polyline points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys))
                 + '" style="fill:none;stroke:var(--ea-ink-3);stroke-width:1.5"/>')
    for v, a, b in zip(DEMO_VALUES, xs, ys):
        now = abs(v - e) < 1e-9
        parts.append(_dot(a, b, 5 if now else 3, "var(--ea-ink)" if now else "var(--ea-ink-3)"))
        parts.append(_t(a, bottom + 15, number(v, lang), 10.5, 600 if now else 400,
                        "var(--ea-ink)" if now else "var(--ea-ink-3)", "middle"))
        if now:
            parts.append(_t(a, b - 9, one(spot_parts(v)["paid"], lang), 11, 600, "var(--ea-ink)", "middle"))
    parts.append(_t(165, bottom + 31, w["every"], 11, 400, "var(--ea-ink-2)", "middle"))
    say = w["say"].format(e=every_text(e, lang), p=one(p["paid"], lang), s=one(p["save"], lang), l=one(p["lost"], lang),
                          k=round(p["stops"]), r=one(p["restart"], lang), c=number(p["cost"], lang))
    return _svg(bottom + 38, w["alt"], parts), say


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
    """A number the way the lesson writes it: at most two decimals, and a decimal comma in Dutch."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return repr(v)
    return {"en": number(v, "en"), "nl": number(v, "nl")}


# exercise 1: hours_paid
def _p(expression: str) -> str:
    return ("def hours_paid(hours, every, stops):\n"
            "    paid = hours\n"
            "    for stop in stops:\n"
            "        saved = 0\n"
            "        while saved + every <= stop:\n"
            "            saved = saved + every\n"
            f"        paid = paid + {expression}\n"
            "    return paid\n")


_P_START = _p("...")
_P = _p("stop - saved")
_P_WHOLE = _p("stop")
_P_SAVED = _p("saved")
_P_EVERY = _p("every")
_P_MINUS = _p("saved - stop")

# exercise 2: break_even
def _b(condition: str) -> str:
    return ("def break_even(per_request, per_day):\n"
            "    n = 0\n"
            f"    while {condition}:\n"
            "        n = n + 1\n"
            "    return n\n")


_B_START = _b("...")
_B = _b("n * per_request <= per_day")
_B_EQUAL = _b("n * per_request < per_day")
_B_FLIPPED = _b("n * per_request > per_day")
_B_DOLLARS = _b("n <= per_day")


FITS = Sort(
    name="fits",
    groups={
        "on_demand": {"en": "On demand", "nl": "On-demand"},
        "reserved": {"en": "Reserved", "nl": "Reserved"},
        "spot": {"en": "Spot", "nl": "Spot"},
        "serverless": {"en": "Serverless", "nl": "Serverless"},
    },
    items={
        "deadline": {"en": "**A test before a deadline.** Four hours on a GPU tomorrow morning, which must not stop halfway.",
                     "nl": "**Een test op het laatste moment.** Morgenochtend vier uur op een GPU, en de test mag niet halverwege stoppen."},
        "trial": {"en": "**A week to try an idea.** A GPU for a few days, and nobody knows yet whether the project goes on.",
                  "nl": "**Een week om een idee te proberen.** Een paar dagen een GPU, en niemand weet nog of het project doorgaat."},
        "service": {"en": "**A service for three years.** The model of a shop answers day and night, for the coming three years.",
                    "nl": "**Een dienst voor drie jaar.** Het model van een winkel antwoordt dag en nacht, de komende drie jaar."},
        "team": {"en": "**A team that trains every day.** Eight GPUs in use nearly every hour, all year long.",
                 "nl": "**Een team dat elke dag traint.** Acht GPU's bijna elk uur in gebruik, het hele jaar door."},
        "training": {"en": "**A run of three days.** Training that saves a checkpoint every hour, to storage that stays.",
                     "nl": "**Een run van drie dagen.** Training die elk uur een checkpoint opslaat, op opslag die blijft."},
        "photos": {"en": "**Ten million product photos overnight.** Labels in parts of a thousand photos, and each part can simply start again.",
                   "nl": "**Tien miljoen productfoto's in één nacht.** Labels in delen van duizend foto's, en elk deel kan gewoon opnieuw beginnen."},
        "demo": {"en": "**A demo for colleagues.** A model that answers a handful of requests a week.",
                 "nl": "**Een demo voor collega's.** Een model dat een handvol verzoeken per week beantwoordt."},
        "form": {"en": "**A form on a website.** A model behind it gets a few hundred requests a day, and none at night.",
                 "nl": "**Een formulier op een website.** Het model erachter krijgt een paar honderd verzoeken per dag, en 's nachts geen."},
    },
    solution={"deadline": "on_demand", "trial": "on_demand", "service": "reserved", "team": "reserved",
              "training": "spot", "photos": "spot", "demo": "serverless", "form": "serverless"},
    nudge={
        "deadline": {"en": "A spot machine can be taken back at any moment. Can this test afford that?",
                     "nl": "Een spot-machine kan op elk moment worden teruggenomen. Mag deze test zomaar stoppen?"},
        "trial": {"en": "Reserving means paying for a year or more. Is that sure here?",
                  "nl": "Reserveren betekent een jaar of langer betalen. Is dat hier zeker?"},
        "service": {"en": "This machine runs every hour for years. Which way of paying is cheapest for an hour that is surely used?",
                    "nl": "Deze machine draait jarenlang elk uur. Welke manier van betalen is het goedkoopst voor een uur dat zeker gebruikt wordt?"},
        "team": {"en": "Above how many hours a month is reserved cheaper than on demand? Is that reached here?",
                 "nl": "Vanaf hoeveel uur per maand is reserved goedkoper dan on-demand? Wordt dat hier gehaald?"},
        "training": {"en": "With a checkpoint every hour, what does a stop cost this run?",
                     "nl": "Wat kost een onderbreking deze run, met een checkpoint elk uur?"},
        "photos": {"en": "A part that stops simply starts again. Is that a reason to pay full price?",
                   "nl": "Een deel dat stopt begint gewoon opnieuw. Is dat een reden om de volle prijs te betalen?"},
        "demo": {"en": "A fixed server costs every hour, also at night and on weekends. How many requests pay for that here?",
                 "nl": "Een vaste server kost elk uur geld, ook 's nachts en in het weekend. Hoeveel verzoeken betalen dat hier?"},
        "form": {"en": "A few hundred requests at 0.00035 dollars, or 18 dollars for a server: which is less per day?",
                 "nl": "Een paar honderd verzoeken voor 0,00035 dollar, of 18 dollar voor een server: wat is per dag minder?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the eight jobs again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de acht klussen opnieuw uit."},
        "empty": {"en": "Put each job under the way of paying that fits it best.",
                  "nl": "Zet elke klus onder de manier van betalen die er het best bij past."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "*{item}* fits another column better.",
                  "nl": "*{item}* past beter bij een andere kolom."},
        "correct": {"en": "All {n} are right. On demand for short work that must not stop, reserved for machines that run nearly every hour for years, spot for work that can stop and go on, and serverless for few requests with quiet hours in between.",
                    "nl": "Alle {n} kloppen. On-demand voor kort werk dat niet mag stoppen, reserved voor machines die jarenlang bijna elk uur draaien, spot voor werk dat kan stoppen en doorgaan, en serverless voor weinig verzoeken met stille uren ertussen."},
    },
)

EXERCISES = {
    "hours_paid": FunctionBox(
        name="hours_paid",
        function="hours_paid",
        start=_P_START,
        solution=_P,
        visible=[(HOURS, 2, STOPS)],
        hidden=[(HOURS, 4, STOPS), (HOURS, 1, STOPS), (HOURS, 2, []), (50, 2, [4, 8]), (HOURS, 0.5, [1.2, 3.9]),
                (20, 5, [3])],
        first=[_DOTS],
        picture=paid_picture,
        show=_shown,
        mistakes=[
            Mistake("whole", wrong=_P_WHOLE,
                    en="`{call}` gives {got}: every stop throws away all hours from the start, as without checkpoints, and it should give {want}. From which hour does training go on after a stop?",
                    nl="`{call}` geeft {got}: elke onderbreking gooit dan alle uren vanaf het begin weg, als zonder checkpoints, en het moet {want} zijn. Vanaf welk uur gaat de training verder na een onderbreking?"),
            Mistake("saved", wrong=_P_SAVED,
                    en="`{call}` gives {got}, which adds the hours in the last checkpoint, and it should give {want}. Those hours are safe. Which hours does a stop throw away?",
                    nl="`{call}` geeft {got}: het telt de uren in het laatste checkpoint op, en het moet {want} zijn. Deze uren zijn veilig. Welke uren gooit een onderbreking weg?"),
            Mistake("every", wrong=_P_EVERY,
                    en="`{call}` gives {got}: a whole interval lost at every stop, and it should give {want}. How much of the interval had passed when the machine was taken back?",
                    nl="`{call}` geeft {got}: een hele afstand tussen checkpoints weg bij elke onderbreking, en het moet {want} zijn. Hoeveel van de afstand was voorbij toen de machine werd teruggenomen?"),
            Mistake("minus", wrong=_P_MINUS,
                    en="`{call}` gives {got}, fewer hours than the training itself takes, and it should give {want}. Can a stop make a run cheaper?",
                    nl="`{call}` geeft {got}, minder uren dan de training zelf duurt, en het moet {want} zijn. Kan een onderbreking een run goedkoper maken?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the hours of work that the stop throws away.",
                      "nl": "Vervang de `...` door de uren werk die de onderbreking weggooit."},
            "arguments": {"en": "The check calls `{call}`, with the hours of training, the hours between checkpoints and a list of stops, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(hours, every, stops)`?",
                          "nl": "De check roept `{call}` aan, met de uren training, de uren tussen checkpoints en een lijst onderbrekingen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(hours, every, stops)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does each stop add the hours since the last checkpoint, no more and no less?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Telt elke onderbreking de uren sinds het laatste checkpoint op, niet meer en niet minder?"},
            "almost": {"en": "Right for the run of the check with a checkpoint every 2 hours. But `{call}` gives {got}, and it should give {want}. Does your function use `hours`, `every` and `stops`, and no fixed numbers?",
                       "nl": "Goed voor de run van de check met een checkpoint elke 2 uur. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie `hours`, `every` en `stops`, en geen vaste getallen?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return paid`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return paid`?"},
            "correct": {"en": "Right in all {n} cases. With a checkpoint every 2 hours, the ten stops of `C.STOPS` throw away 10.25 hours, and the run pays for 110.25 hours; with one every 4 hours, 120.25. In the picture: what happens with a checkpoint every 8 hours?",
                        "nl": "Goed in alle {n} gevallen. Met een checkpoint elke 2 uur gooien de tien onderbrekingen van `C.STOPS` 10,25 uur weg, en betaalt de run 110,25 uur; met een checkpoint elke 4 uur 120,25. In de afbeelding: wat gebeurt er met een checkpoint elke 8 uur?"},
        },
    ),
    "break_even": FunctionBox(
        name="break_even",
        function="break_even",
        start=_B_START,
        solution=_B,
        visible=[(PER_REQUEST, SERVER_DAY)],
        hidden=[(0.5, 12.0), (0.0003, 20.0), (2.0, 1.0), (0.001, 0.0)],
        first=[_DOTS],
        picture=even_picture,
        mistakes=[
            Mistake("equal", wrong=_B_EQUAL,
                    en="`{call}` gives {got}: at that number, serverless costs exactly as much as the server, and it should give {want}. Does serverless cost more there?",
                    nl="`{call}` geeft {got}: bij dat aantal kost serverless precies even veel als de server, en het moet {want} zijn. Kost serverless daar meer?"),
            Mistake("flipped", wrong=_B_FLIPPED,
                    en="`{call}` gives {got}: the loop does not run once, and it should give {want}. With `n` at 0, is the question after `while` true?",
                    nl="`{call}` geeft {got}: de loop draait geen enkele keer, en het moet {want} zijn. Is de vraag na `while` waar, met `n` op 0?"),
            Mistake("dollars", wrong=_B_DOLLARS,
                    en="`{call}` gives {got}: it compares the number of requests with dollars, and it should give {want}. What do `n` requests cost on serverless?",
                    nl="`{call}` geeft {got}: het vergelijkt het aantal verzoeken met dollars, en het moet {want} zijn. Wat kosten `n` verzoeken op serverless?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the question whether `n` requests on serverless cost no more than the server.",
                      "nl": "Vervang de `...` door de vraag of `n` verzoeken op serverless niet meer kosten dan de server."},
            "arguments": {"en": "The check calls `{call}`, with the dollars per request and the dollars a day of the server, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(per_request, per_day)`?",
                          "nl": "De check roept `{call}` aan, met de dollars per verzoek en de dollars per dag van de server, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(per_request, per_day)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does the loop go on exactly as long as `n` requests cost no more than the server?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Gaat de loop precies zo lang door als `n` verzoeken niet meer kosten dan de server?"},
            "almost": {"en": "Right for the server of the check. But `{call}` gives {got}, and it should give {want}. Does your function use `per_request` and `per_day`, and no fixed numbers?",
                       "nl": "Goed voor de server van de check. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie `per_request` en `per_day`, en geen vaste getallen?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return n`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return n`?"},
            "correct": {"en": "Right in all {n} cases. From 51,429 requests a day, serverless costs more than the server of 18 dollars a day: about 36 requests a minute, day and night. In the picture: where do the two lines cross?",
                        "nl": "Goed in alle {n} gevallen. Vanaf 51.429 verzoeken per dag kost serverless meer dan de server van 18 dollar per dag: ongeveer 36 verzoeken per minuut, dag en nacht. In de afbeelding: waar kruisen de twee lijnen?"},
        },
    ),
    "fits": FITS,
}

# The lesson is done when these are correct. "rhythm" is the choice question before the demo.
REQUIRED = ["hours_paid", "break_even", "fits", "rhythm"]

_SORTED = dict(FITS.solution)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("hours_paid", _P_START, "empty", None),
    ("hours_paid", "", "empty", None),
    ("hours_paid", _P, "correct", None),
    ("hours_paid", _p("-saved + stop"), "correct", None),
    ("hours_paid", _p("(stop - saved)"), "correct", None),
    ("hours_paid", "def hours_paid(hours, every, stops):\n    paid = hours\n    for stop in stops:\n"
                   "        paid = paid + stop - every * int(stop / every)\n    return paid\n", "correct", None),
    ("hours_paid", _P_WHOLE, "wrong", "whole"),
    ("hours_paid", _P_SAVED, "wrong", "saved"),
    ("hours_paid", _P_EVERY, "wrong", "every"),
    ("hours_paid", _P_MINUS, "wrong", "minus"),
    ("hours_paid", _p("stop - every"), "wrong", None),
    ("hours_paid", _P.replace("    paid = hours\n", "    paid = 100\n"), "almost", None),
    ("hours_paid", "def hours_paid(hours, every, stops):\n    return 110.25\n", "almost", None),
    ("hours_paid", _p("stop - save"), "error", None),
    ("hours_paid", _P.replace("def hours_paid(hours, every, stops):", "def hours_paid(hours, stops):"), "error", "arguments"),
    ("hours_paid", _P.replace("    return paid\n", ""), "wrong", "no-return"),
    ("hours_paid", _P.replace("    return paid\n", "    print(paid)\n"), "wrong", "prints"),
    ("hours_paid", _P.replace("    return paid\n", "    extra = ...\n    return paid\n"), "wrong", "dots-left"),
    ("hours_paid", "x = 1\n", "wrong", "no-function"),
    ("break_even", _B_START, "empty", None),
    ("break_even", "", "empty", None),
    ("break_even", _B, "correct", None),
    ("break_even", _b("per_day >= n * per_request"), "correct", None),
    ("break_even", "def break_even(per_request, per_day):\n    return int(per_day / per_request) + 1\n", "correct", None),
    ("break_even", _B_EQUAL, "wrong", "equal"),
    ("break_even", _B_FLIPPED, "wrong", "flipped"),
    ("break_even", _B_DOLLARS, "wrong", "dollars"),
    ("break_even", _b("n * per_request <= per_day / 2"), "wrong", None),
    ("break_even", "def break_even(per_request, per_day):\n    return 51429\n", "almost", None),
    ("break_even", _b("n * per_requests <= per_day"), "error", None),
    ("break_even", _B.replace("def break_even(per_request, per_day):", "def break_even(per_request):"), "error", "arguments"),
    ("break_even", _B.replace("    return n\n", ""), "wrong", "no-return"),
    ("break_even", _B.replace("    return n\n", "    print(n)\n"), "wrong", "prints"),
    ("break_even", _B.replace("    return n\n", "    extra = ...\n    return n\n"), "wrong", "dots-left"),
    ("break_even", "x = 1\n", "wrong", "no-function"),
    ("fits", {}, "empty", None),
    ("fits", _SORTED, "correct", None),
    ("fits", {"deadline": "on_demand", "service": "reserved"}, "wrong", "unfinished"),
    ("fits", _SORTED | {"deadline": "spot"}, "wrong", "wrong:deadline"),
    ("fits", _SORTED | {"trial": "reserved"}, "wrong", "wrong:trial"),
    ("fits", _SORTED | {"service": "on_demand"}, "wrong", "wrong:service"),
    ("fits", _SORTED | {"team": "spot"}, "wrong", "wrong:team"),
    ("fits", _SORTED | {"training": "on_demand"}, "wrong", "wrong:training"),
    ("fits", _SORTED | {"photos": "reserved"}, "wrong", "wrong:photos"),
    ("fits", _SORTED | {"demo": "reserved"}, "wrong", "wrong:demo"),
    ("fits", _SORTED | {"form": "on_demand"}, "wrong", "wrong:form"),
]
