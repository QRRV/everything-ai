# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-feedback-loops: when a model's output decides which data it gets back. The app of the bike
service suggests one of eight routes, A to H, on its home screen; a visitor who sees a route takes it with a chance of
its own, its appeal, which the app does not know. In the first week the app showed a random route to each of 400
visitors. After that it shows the route with the most rides so far: the route that led after the first week gets every
showing, and with it every ride, whatever its appeal. The learner writes the line of `simulate` in which a visitor
takes the shown route, and the line of `choose` that ranks the routes by rides per showing; a choice about random
showings with the old ranking, and a sort of situations with and without a feedback loop. The demo lets the app run
for 20,000 visitors with either ranking and a share of random showings.

The routes, their appeal and the visitors are the course's own simulation; no data are read. Every word a learner's
code gives back is shown through html.escape."""
import ast
import html
import math
import re

import numpy as np

from everything_ai import chart, check, live, yours
from everything_ai.check import FunctionBox, Mistake, Sort, kept

LESSON = "sys-feedback-loops"
LANGS = ("en", "nl")
ROUTES = "ABCDEFGH"
_APPEAL = np.array([0.20, 0.08, 0.28, 0.12, 0.32, 0.04, 0.24, 0.16])   # the chance that a visitor takes a shown route
BEST = 4             # route E
WEEK = 400           # visitors of the first week, each shown a random route
FIRST_SEED = 16      # the course's first week: G leads by rides, E by rides per showing
VISITORS = 20_000    # visitors after the first week
RUN_SEED = 1         # the random numbers of those visitors
EPSILON = 0.1        # the share of random showings in exercise 2
WEEKS = 1000         # first weeks of the course's runs of many weeks
EPSILONS = (0.0, 0.05, 0.1, 0.2)   # the shares of random showings in the demo


def _lang(lang) -> str:
    return lang if lang in LANGS else "en"


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch, and a real minus sign."""
    out = f"{x:,.{digits}f}"
    out = out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out
    return out.replace("-", "−")


def percent(x: float, lang: str) -> str:
    """A share as a whole percentage: 10%, 5%."""
    return f"{round(100 * x)}%"


def _defined(code: str, name: str, names: dict):
    space = dict(names)
    exec(code, space)   # noqa: S102   the course's own solution text
    return space[name]


class _Shown(np.ndarray):
    """An array that a check message writes as the lesson names it, such as C.RIDES. The learner's code gets a plain
    copy."""

    def __repr__(self):
        return getattr(self, "shown", "values")

    def __deepcopy__(self, memo):
        return np.array(self)


def _named(values, name: str) -> np.ndarray:
    v = np.array(values, dtype=float).view(_Shown)
    v.shown = name
    return v


def first_week(seed: int = FIRST_SEED, appeal=_APPEAL, visitors: int = WEEK) -> tuple:
    """The first week: every visitor sees a random route and takes it with the chance of that route. Gives the rides
    and the showings of each route, as arrays of floats."""
    rng = np.random.default_rng(seed)
    rides, shown = np.zeros(len(appeal)), np.zeros(len(appeal))
    for _ in range(visitors):
        route = rng.integers(0, len(appeal))
        shown[route] = shown[route] + 1
        if rng.random() < appeal[route]:
            rides[route] = rides[route] + 1
    return rides, shown


_WEEK_RIDES, _WEEK_SHOWN = first_week()
APPEAL = _named(_APPEAL, "C.APPEAL")      # the routes' chances, which the app does not know
RIDES = _named(_WEEK_RIDES, "C.RIDES")    # the rides of each route in the course's first week
SHOWN = _named(_WEEK_SHOWN, "C.SHOWN")    # how often the first week showed each route
BEST_PER_100 = 100 * float(_APPEAL.max())   # rides per 100 visitors when every visitor sees route E


# ======== exercise 1: simulate ============================================================================================

_COL = 50   # where the comments of the start code begin


def _line(code: str, comment: str = "") -> str:
    return (code.ljust(_COL) + "# " + comment if comment else code) + "\n"


def _s(test: str = "...") -> str:
    """simulate as the box holds it, with `test` as the test of the line in which a visitor takes the shown route."""
    return ("def simulate(appeal, rides, shown, visitors, seed):\n"
            "    rng = np.random.default_rng(seed)\n"
            "    for v in range(visitors):\n"
            + _line("        route = np.argmax(rides)", "the app shows the route with the most rides so far")
            + "        shown[route] = shown[route] + 1\n"
            + _line(f"        if {test}:", "the visitor takes it with chance appeal[route]")
            + "            rides[route] = rides[route] + 1\n"
            "    return rides\n")


_S_START = _s()
_S = _s("rng.random() < appeal[route]")
_S_INVERTED = _s("rng.random() > appeal[route]")
_S_ALWAYS = _s("appeal[route] > 0")
_S_HALF = _s("rng.random() < 0.5")
_S_FIXED = _s("rng.random() < [0.20, 0.08, 0.28, 0.12, 0.32, 0.04, 0.24, 0.16][route]")   # right only for C.APPEAL

simulate = _defined(_S, "simulate", {"np": np})


# ======== exercise 2: choose ==============================================================================================

def _c(best: str = "...") -> str:
    """choose as the box holds it, with `best` as what it gives back outside the random showings."""
    return ("def choose(rides, shown, epsilon, rng):\n"
            + _line("    if rng.random() < epsilon:", "now and then a random route, to learn about all")
            + "        return rng.integers(0, len(rides))\n"
            + _line(f"    return {best}", "else the route with the most rides per showing"))


_C_START = _c()
_C = _c("np.argmax(rides / shown)")
_C_COUNT = _c("np.argmax(rides)")
_SHOWN = _c("np.argmax(shown)")
_C_FLIPPED = _c("np.argmax(shown / rides)")
_C_WORST = _c("np.argmin(rides / shown)")
_C_RATE = _c("np.max(rides / shown)")
_C_FIXED = _c("4")   # route E: right for the first week only

choose = _defined(_C, "choose", {"np": np})


def by_rides(rides, shown, epsilon, rng):
    """The app of exercise 1, with random showings: a random route with chance `epsilon`, else the most rides. Without
    random showings it draws no random number for the choice, so it gives the same visitors as simulate."""
    if epsilon and rng.random() < epsilon:
        return rng.integers(0, len(rides))
    return np.argmax(rides)


# ======== the runs ========================================================================================================

EVERY = 200          # visitors between two points of the live chart
WINDOW = 1000        # the live chart: rides per 100 visitors over the last this many visitors
TICKS = [0, 5000, 10000, 15000]   # the x-axis: a last tick at 20,000, on the right edge, would not fit
LOOKS = [200, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]   # the rewind's stops


def run(pick, epsilon: float, visitors: int = VISITORS, seed: int = RUN_SEED, show=None) -> dict:
    """The app after the course's first week: every visitor sees the route `pick(rides, shown, epsilon, rng)` gives and
    takes it with the chance of that route. After every EVERY visitors, `show(t, per_100, shown_after)` gets the rides
    per 100 visitors over the last WINDOW visitors and the showings of each route since the first week. Gives the
    run, or "choose" when `pick` stops with an error or gives no route."""
    rng = np.random.default_rng(seed)
    rides, shown = _WEEK_RIDES.copy(), _WEEK_SHOWN.copy()
    hits, curve, looks = [], [], {}
    for t in range(1, visitors + 1):
        try:
            route = pick(rides.copy(), shown.copy(), epsilon, rng)
            route = int(route)
        except Exception:   # noqa: BLE001   the demo says which function stopped
            return "choose"
        if not 0 <= route < len(ROUTES):
            return "choose"
        shown[route] = shown[route] + 1
        hit = rng.random() < _APPEAL[route]
        if hit:
            rides[route] = rides[route] + 1
        hits.append(int(hit))
        if t % EVERY == 0:
            recent = hits[-WINDOW:]
            per_100 = 100 * sum(recent) / len(recent)
            after = shown - _WEEK_SHOWN
            curve.append((t, per_100))
            if t in LOOKS:
                looks[t] = after.copy()
            if show is not None:
                show(t, per_100, after.copy())
    return {"rides": rides, "shown": shown, "after": shown - _WEEK_SHOWN, "got": sum(hits), "per_100": 100 * sum(hits) / visitors,
            "curve": curve, "looks": looks, "epsilon": epsilon}


def many(rank: str, epsilon: float, weeks: int = WEEKS, visitors: int = VISITORS, seed: int = 0) -> dict:
    """The app on `weeks` other first weeks at once, ranking by "rides" or by "rate" (rides per showing), with a share
    `epsilon` of random showings: how often the route shown most over the last 2,000 visitors is route E, how often E
    led by rides after the first week, and the rides per 100 visitors after the first week."""
    rng = np.random.default_rng(seed)
    n = len(ROUTES)
    rows = np.arange(weeks)
    rides, shown = np.zeros((weeks, n)), np.zeros((weeks, n))
    for _ in range(WEEK):
        r = rng.integers(0, n, weeks)
        shown[rows, r] += 1
        rides[rows, r] += rng.random(weeks) < _APPEAL[r]
    led = rides.argmax(1) == BEST
    last, got = np.zeros((weeks, n)), np.zeros(weeks)
    for v in range(visitors):
        best = (rides if rank == "rides" else rides / shown).argmax(1)
        if epsilon:
            r = np.where(rng.random(weeks) < epsilon, rng.integers(0, n, weeks), best)
        else:
            r = best
        shown[rows, r] += 1
        hit = rng.random(weeks) < _APPEAL[r]
        rides[rows, r] += hit
        got += hit
        if v >= visitors - 2000:
            last[rows, r] += 1
    return {"best": float((last.argmax(1) == BEST).mean()), "led": float(led.mean()),
            "per_100": float(100 * got.mean() / visitors)}


def many_of(rank: str, epsilon: float) -> dict:
    return kept((LESSON, "many", rank, float(epsilon)), lambda: many(rank, epsilon))


def _numbers() -> dict:
    pop = run(by_rides, 0.0)   # the same visitors as simulate in exercise 1
    explore = run(choose, EPSILON)
    rate = run(choose, 0.0)
    out = {"week_rides": _WEEK_RIDES.copy(), "week_shown": _WEEK_SHOWN.copy(),
           "leader": int(np.argmax(_WEEK_RIDES)), "rate_leader": int(np.argmax(_WEEK_RIDES / _WEEK_SHOWN)),
           "pop_got": pop["got"], "pop_per_100": pop["per_100"], "pop_top": float(pop["after"].max()),
           "pop_routes": int((pop["after"] > 0).sum()),
           "explore_got": explore["got"], "explore_per_100": explore["per_100"], "explore_top": int(np.argmax(explore["after"])),
           "explore_top_n": float(explore["after"].max()), "rate_per_100": rate["per_100"], "rate_top": int(np.argmax(rate["after"])),
           "best_per_100": 100 * float(_APPEAL.max())}
    return out


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
    for colour, word in items:
        mark = (f'<span style="display:inline-block;width:10px;height:10px;background:{colour};margin-right:4px;'
                f'vertical-align:-1px"></span>')
        out.append(f'<span style="margin-right:12px;white-space:nowrap">{mark}{html.escape(word)}</span>')
    return f'<div style="{_UI}">{"".join(out)}</div>'


def _whole(x: float, lang: str) -> str:
    return number(x, 0, lang)


def _bars(x0: float, width: float, y0: float, height: float, values, colour: str, lang: str, labels: bool = True,
          best: int | None = None) -> list:
    """Eight bars in a box, one per route, on a scale up to the largest value: the value above each bar that is not 0,
    the route under it, and route `best` in bold."""
    values = [max(float(v), 0.0) for v in values]
    top = max(values) or 1.0
    slot = width / len(values)
    parts = [f'<line x1="{x0:.1f}" y1="{y0 + height:.1f}" x2="{x0 + width:.1f}" y2="{y0 + height:.1f}" '
             f'style="stroke:var(--ea-ink-3);stroke-width:1"/>']
    for k, v in enumerate(values):
        h = height * v / top
        x = x0 + k * slot
        parts.append(f'<rect x="{x + slot * 0.18:.1f}" y="{y0 + height - h:.1f}" width="{slot * 0.64:.1f}" height="{h:.1f}" '
                     f'style="fill:{colour}"/>')
        if labels and v > 0:
            parts.append(_t_svg(x + slot / 2, y0 + height - h - 3, _whole(v, lang), 8.5, 400, "var(--ea-ink-2)", "middle"))
        parts.append(_t_svg(x + slot / 2, y0 + height + 12, ROUTES[k], 10.5, 700 if k == best else 400, "var(--ea-ink)",
                            "middle"))
    return parts


def _appeal_row(x0: float, width: float, y: float, label: str) -> list:
    """Under the routes: how many of 100 visitors who see each route take it."""
    slot = width / len(_APPEAL)
    parts = [_t_svg(x0 - 2, y, label, 9, 400, "var(--ea-ink-3)", "end")] if label else []
    for k, a in enumerate(_APPEAL):
        parts.append(_t_svg(x0 + k * slot + slot / 2, y, f"{round(100 * a)}", 9, 700 if k == BEST else 400, "var(--ea-data)",
                            "middle"))
    return parts


# ======== the picture of the idea =========================================================================================

IDEA_TEXT = {
    "en": {"title": "What the app shows, before and after",
           "panels": ("first week: random", "next 20,000 visitors"),
           "appeal": "the real chance, in %, that a visitor takes it:",
           "key": "showings of each route",
           "caption": "In the first week, every route was shown about 50 times. Route G got 15 rides there and route E 13, so from then on the app shows G to every visitor: G gets every ride, and E is never shown again, while 32 of 100 visitors who see E take it.",
           "alt": "Two bar charts of how often the app showed each of the eight routes. In the first week each route was shown about 50 times; over the next 20,000 visitors, route G was shown every time and the others never."},
    "nl": {"title": "Wat de app toont, voor en na",
           "panels": ("eerste week: willekeurig", "volgende 20.000 bezoekers"),
           "appeal": "de echte kans, in %, dat een bezoeker de route neemt:",
           "key": "keren dat elke route getoond werd",
           "caption": "In de eerste week werd elke route ongeveer 50 keer getoond. Route G kreeg toen 15 ritten en route E 13, dus toont de app vanaf dan G aan elke bezoeker: G krijgt elke rit, en E wordt nooit meer getoond, terwijl 32 van de 100 bezoekers route E nemen als de app route E toont.",
           "alt": "Twee staafdiagrammen van hoe vaak de app elk van de acht routes toonde. In de eerste week werd elke route ongeveer 50 keer getoond; bij de volgende 20.000 bezoekers werd route G elke keer getoond en de andere nooit."},
}


def idea_picture(lang: str) -> str:
    lang = _lang(lang)
    w = IDEA_TEXT[lang]
    n = NUMBERS
    after = np.zeros(len(ROUTES))
    after[n["leader"]] = VISITORS
    parts = [_t_svg(4, 14, w["title"], 13, 600)]
    for x0, label, values in ((8, w["panels"][0], n["week_shown"]), (172, w["panels"][1], after)):
        parts.append(_t_svg(x0 + 75, 34, label, 10.5, 600, "var(--ea-ink-2)", "middle"))
        parts.extend(_bars(x0, 150, 48, 70, values, "var(--ea-pred)", lang, best=BEST))
    parts.append(_t_svg(4, 150, w["appeal"], 9.5, 400, "var(--ea-ink-2)"))
    for x0 in (8, 172):
        parts.extend(_appeal_row(x0, 150, 164, ""))
    return _svg(172, w["alt"], parts) + _key([("var(--ea-pred)", w["key"])]) + _note(w["caption"])


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
    """What simulate gave back as n finite numbers, or None for anything else."""
    try:
        out = [float(x) for x in np.asarray(v, dtype=float).ravel()]
    except (TypeError, ValueError):
        return None
    return out if len(out) == n and all(math.isfinite(x) for x in out) else None


YOURS_TEXT = {
    "en": {"simulate": {"title": "Your app, 20,000 visitors",
                        "intro": "Your `simulate` on the course's first week, `C.APPEAL`, `C.RIDES` and `C.SHOWN`, for 20,000 visitors: the rides of each route after the first week.",
                        "appeal": "the real chance, in %, that a visitor takes it:",
                        "key": "rides after the first week",
                        "said": "By your numbers, route {top} got {n} of the {total} rides after the first week: {per} per 100 visitors. Route E, shown every time, would give about 32.",
                        "none": "Nothing to draw yet: your `simulate` gives no list of eight numbers for these visitors. The check above says why.",
                        "alt": "Eight bars, the rides of each route after the first week by the learner's simulate."},
           "choose": {"title": "Your choose, 20,000 visitors",
                      "intro": "The app with your `choose` and 10% random showings, after the course's first week: how often it showed each route.",
                      "appeal": "the real chance, in %, that a visitor takes it:",
                      "key": "showings after the first week",
                      "said": "With your `choose`, the app showed route {top} most: {n} of the 20,000 visitors. It got {per} rides per 100 visitors; the app of Exercise 1 got {pop}.",
                      "none": "Nothing to draw yet: your `choose` stopped with an error or gave no route for some visitor. The check above says why.",
                      "alt": "Eight bars, how often the app with the learner's choose showed each route after the first week."}},
    "nl": {"simulate": {"title": "Je app, 20.000 bezoekers",
                        "intro": "Je `simulate` op de eerste week van de course, `C.APPEAL`, `C.RIDES` en `C.SHOWN`, voor 20.000 bezoekers: de ritten van elke route na de eerste week.",
                        "appeal": "de echte kans, in %, dat een bezoeker de route neemt:",
                        "key": "ritten na de eerste week",
                        "said": "Volgens je getallen kreeg route {top} {n} van de {total} ritten na de eerste week: {per} per 100 bezoekers. Route E zou ongeveer 32 geven als de app route E elke keer toonde.",
                        "none": "Nog niets te tekenen: je `simulate` geeft voor deze bezoekers geen lijst met acht getallen. De check hierboven zegt waarom.",
                        "alt": "Acht staven, de ritten van elke route na de eerste week volgens de simulate van de leerling."},
           "choose": {"title": "Je choose, 20.000 bezoekers",
                      "intro": "De app met je `choose` en 10% willekeurige routes, na de eerste week van de course: hoe vaak de app elke route toonde.",
                      "appeal": "de echte kans, in %, dat een bezoeker de route neemt:",
                      "key": "keren getoond na de eerste week",
                      "said": "Met je `choose` toonde de app route {top} het vaakst: aan {n} van de 20.000 bezoekers. De app kreeg {per} ritten per 100 bezoekers; de app van Oefening 1 kreeg {pop}.",
                      "none": "Nog niets te tekenen: je `choose` stopte met een fout of gaf bij een bezoeker geen route. De check hierboven zegt waarom.",
                      "alt": "Acht staven, hoe vaak de app met de choose van de leerling elke route toonde na de eerste week."}},
}


def _route_picture(w: dict, values, colour: str, lang: str, said: str, extra: str = "") -> str:
    parts = _bars(20, 290, 18, 110, values, colour, lang, best=BEST)
    parts.append(_t_svg(4, 160, w["appeal"], 9.5, 400, "var(--ea-ink-2)"))
    parts.extend(_appeal_row(20, 290, 174, ""))
    svg = _svg(182, w["alt"], parts)
    return yours.figure(w["title"], _note(w["intro"]) + svg + _key([(colour, w["key"])]) + _note(said) + extra, alt=w["title"])


def simulate_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["simulate"]
    fn = _fn(seen)
    stop = yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    got = _call(lambda: fn(_APPEAL.copy(), _WEEK_RIDES.copy(), _WEEK_SHOWN.copy(), VISITORS, RUN_SEED)) if fn else _STOPPED
    if got is _STOPPED:
        return stop
    values = _numbers_of(got, len(ROUTES))
    if values is None:
        return stop
    after = [max(v - r, 0.0) for v, r in zip(values, _WEEK_RIDES)]
    total = sum(after)
    top = int(np.argmax(after))
    said = w["said"].format(top=ROUTES[top], n=_whole(after[top], lang), total=_whole(total, lang),
                            per=number(100 * total / VISITORS, 1, lang))
    return _route_picture(w, after, "var(--ea-data)", lang, said)


def choose_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["choose"]
    fn = _fn(seen)
    stop = yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    got = _call(lambda: run(fn, EPSILON)) if fn else _STOPPED
    if got is _STOPPED or got == "choose":
        return stop
    top = int(np.argmax(got["after"]))
    said = w["said"].format(top=ROUTES[top], n=_whole(got["after"][top], lang), per=number(got["per_100"], 1, lang),
                            pop=number(NUMBERS["pop_per_100"], 1, lang))
    return _route_picture(w, got["after"], "var(--ea-pred)", lang, said)


# ======== the demo: 20,000 visitors, live =================================================================================

_LIVE_TEXT = {
    "en": {"title": "Rides per 100 visitors", "x": "visitor", "y": "rides per 100, last 1,000 visitors",
           "alt": "A chart that grows while the app serves 20,000 visitors: the rides per 100 visitors over the last 1,000 visitors.",
           "best": "route E every time", "ranks": {"rides": "the most rides", "rate": "the most rides per showing"},
           "now": "What the app showed after visitor {t}",
           "now_alt": "Eight bars: how often the app showed each route since the first week.",
           "key": "showings since the first week", "appeal": "the real chance, in %, that a visitor takes it:",
           "line": "**Visitor {t} of {all}** · ranking by {rank}, {eps} random routes",
           "rewind_alt": "The rides per 100 visitors over the run, with a line at the chosen visitor.",
           "rewind": "After visitor {t}: the app showed route {top} most, {n} times, and showed {k} of the 8 routes at least once since the first week.",
           "many": "Over the course's {weeks} first weeks, this app ends on route E in {best} of them, and gets {per} rides per 100 visitors."},
    "nl": {"title": "Ritten per 100 bezoekers", "x": "bezoeker", "y": "ritten per 100, laatste 1.000 bezoekers",
           "alt": "Een grafiek die groeit terwijl de app 20.000 bezoekers bedient: de ritten per 100 bezoekers over de laatste 1.000 bezoekers.",
           "best": "elke keer route E", "ranks": {"rides": "de meeste ritten", "rate": "de meeste ritten per keer getoond"},
           "now": "Wat de app toonde na bezoeker {t}",
           "now_alt": "Acht staven: hoe vaak de app elke route toonde sinds de eerste week.",
           "key": "keren getoond sinds de eerste week", "appeal": "de echte kans, in %, dat een bezoeker de route neemt:",
           "line": "**Bezoeker {t} van {all}** · rangschikken op {rank}, {eps} willekeurige routes",
           "rewind_alt": "De ritten per 100 bezoekers over de run, met een lijn bij de gekozen bezoeker.",
           "rewind": "Na bezoeker {t}: de app toonde route {top} het vaakst, {n} keer, en toonde {k} van de 8 routes minstens één keer sinds de eerste week.",
           "many": "Over de {weeks} eerste weken van de course eindigt deze app in {best} ervan op route E, en krijgt de app {per} ritten per 100 bezoekers."},
}


def live_words(lang: str) -> dict:
    return _LIVE_TEXT[_lang(lang)]


def now_picture(after, step: int, lang: str) -> str:
    """How often the app showed each route since the first week, after visitor `step`."""
    lang = _lang(lang)
    w = _LIVE_TEXT[lang]
    parts = _bars(20, 290, 18, 110, after, "var(--ea-pred)", lang, best=BEST)
    parts.append(_t_svg(4, 160, w["appeal"], 9.5, 400, "var(--ea-ink-2)"))
    parts.extend(_appeal_row(20, 290, 174, ""))
    title = w["now"].format(t=_whole(step, lang))
    return yours.figure(title, _svg(182, w["now_alt"], parts) + _key([("var(--ea-pred)", w["key"])]), alt=title)


def curve_picture(result: dict, lang: str, at: int | None = None) -> str:
    """The rides per 100 visitors of a run, as the live chart drew them, with a line at visitor `at`."""
    lang = _lang(lang)
    w = _LIVE_TEXT[lang]
    p = chart.Panel((0, VISITORS), (0, 40), title=w["title"], xlabel=w["x"], ylabel=w["y"], xticks=TICKS)
    xs = [x for x, _ in result["curve"]]
    p.line([0, VISITORS], [100 * _APPEAL.max()] * 2, "ink-2", 1.2, dashed=True)
    p.text(VISITORS * 0.98, 100 * _APPEAL.max() + 4.5, w["best"], "ink-2", "end", halo=True)   # above the run
    p.line(xs, [y for _, y in result["curve"]], "ink", 2.0)
    if at is not None:
        p.line([at, at], [0, 40], "ink-2", 1.2)
    out = chart.figure([p], alt=w["rewind_alt"], width=330, height=240)
    return chart.with_decimal_comma(out) if lang == "nl" else out


def rewind_frame(result: dict, step: int, lang: str) -> str:
    return live.rewound(curve_picture(result, lang, at=step), [now_picture(result["looks"][step], step, lang)])


def rewind_text(result: dict, step: int, lang: str) -> str:
    lang = _lang(lang)
    after = result["looks"][step]
    top = int(np.argmax(after))
    return _LIVE_TEXT[lang]["rewind"].format(t=_whole(step, lang), top=ROUTES[top], n=_whole(after[top], lang),
                                             k=int((after > 0).sum()))


def many_text(rank: str, epsilon: float, lang: str) -> str:
    m = many_of(rank, epsilon)
    return _LIVE_TEXT[_lang(lang)]["many"].format(weeks=_whole(WEEKS, lang), best=_whole(round(m["best"] * WEEKS), lang),
                                                  per=number(m["per_100"], 1, lang))


def progress_line(t: int, rank: str, epsilon: float, lang: str) -> str:
    w = _LIVE_TEXT[_lang(lang)]
    return w["line"].format(t=_whole(t, lang), all=_whole(VISITORS, lang), rank=w["ranks"].get(rank, ""),
                            eps=percent(epsilon, lang))


_RESULT_TEXT = {
    "en": "With {who}: {per} rides per 100 visitors over the 20,000 visitors, against {best} if every visitor saw route E. {many}",
    "nl": "Met {who}: {per} ritten per 100 bezoekers over de 20.000 bezoekers, tegen {best} als elke bezoeker route E zag. {many}",
}
_WHO = {
    "en": {"rides": "the app of Exercise 1", "yours": "your `choose`", "course": "the course's `choose`, since yours does not pass its check yet"},
    "nl": {"rides": "de app van Oefening 1", "yours": "je `choose`", "course": "de `choose` van de course, want je eigen `choose` slaagt nog niet voor de check"},
}


def result_text(result: dict, rank: str, epsilon: float, yours_: bool, lang: str) -> str:
    """The sentence under the rewind: whose app ran, its rides per 100 visitors, and the course's 1,000 first weeks."""
    lang = _lang(lang)
    who = _WHO[lang]["rides" if rank != "rate" else ("yours" if yours_ else "course")]
    return _RESULT_TEXT[lang].format(who=who, per=number(result["per_100"], 1, lang), best=_whole(BEST_PER_100, lang),
                                     many=many_text(rank, epsilon, lang))


def picker(rank: str, mine_choose=None):
    """What the demo runs: the learner's choose (or the course's) for "rate", the app of exercise 1 for "rides"."""
    if rank == "rate":
        return mine_choose or choose
    return by_rides


DEMO_LIMIT = 120   # seconds


def demo(rank: str, epsilon: float, mine_choose=None, show=None):
    """The demo within its time limit: the run, "choose" when the learner's function fails, or "slow"."""
    from everything_ai.check import Timeout, with_time_limit
    rank = rank if rank in ("rides", "rate") else "rides"
    epsilon = epsilon if epsilon in EPSILONS else 0.0
    try:
        return with_time_limit(lambda: run(picker(rank, mine_choose), epsilon, show=show), DEMO_LIMIT)
    except Timeout:
        return "slow"


def mine(box_text, correct: bool):
    """The learner's choose when its check passes, else None: the demo then uses the course's."""
    if not correct:
        return None
    from everything_ai.check import run_code
    fn = run_code(box_text, {"np": np}).names.get("choose")
    return fn if callable(fn) else None


# ======== the exercises ===================================================================================================

def _dots_left(got, text_) -> bool:
    """A `...` of the start code left in the box: an Ellipsis in the code, not three dots inside a string."""
    try:
        tree = ast.parse(text_)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Constant) and node.value is Ellipsis for node in ast.walk(tree))


def _whole_array(got, text_) -> bool:
    """The test compares a random number with all of `appeal`, not with the chance of the shown route."""
    return re.search(r"rng\.random\(\)\s*[<>]=?\s*appeal\s*:", text_) is not None


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Can you replace it with your own code?",
                nl="Er staat nog een `...` in je code. Kun je de `...` vervangen door je eigen code?")
_ARRAY = Mistake("whole-appeal", matches=_whole_array,
                 en="Your test compares one random number with all of `appeal`, eight chances at once, and an `if` needs one answer. Which of the eight chances belongs to the route the app shows?",
                 nl="Je test vergelijkt één willekeurig getal met heel `appeal`, acht kansen tegelijk, en een `if` heeft één antwoord nodig. Welke van de acht kansen hoort bij de route die de app toont?")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then begins with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _shown(v):
    """{got} and {want}: a list of rides as whole numbers, or a route by its letter."""
    if isinstance(v, (np.ndarray, list)) and all(isinstance(x, (float, int, np.floating, np.integer))
                                                    and not isinstance(x, bool) for x in np.asarray(v, dtype=object).ravel()):
        xs = np.asarray(v, dtype=float).ravel()
        text_ = "[" + ", ".join(f"{x:.0f}" if math.isfinite(x) and x == round(x) else f"{x:.3f}" for x in xs) + "]"
    else:
        text_ = repr(v)
    return f"`{text_}`" if "`" not in text_ else f"<code>{html.escape(text_)}</code>"


CALLS = 200   # the check calls choose this many times per case, with one random number generator


class Picks(tuple):
    """How often choose gave each route in CALLS calls, or ("other", what it gave) for something that is not a route."""


def _picks_shown(p):
    """{got} and {want}: how often each route came, such as "E 200", or what came that is not a route."""
    if isinstance(p, Picks) and len(p) == 2 and p[0] == "other":
        code = f"<code>{html.escape(p[1])}</code>"
        return {"en": f"{code}, which is not one of the routes", "nl": f"{code}, en dat is geen van de routes"}
    if isinstance(p, Picks):
        return ", ".join(f"{ROUTES[k]} {n}" for k, n in enumerate(p) if n)
    return f"<code>{html.escape(repr(p))}</code>"


class ChooseBox(FunctionBox):
    """choose in a code box. The check calls it CALLS times for every case, with one random number generator made
    with seed 1, and counts how often it gives each route: the course's random showings come first, so a right line
    gives the same counts as the course's own choose."""

    def _run(self, fn, case):
        rides, shown, epsilon = case
        rng = np.random.default_rng(1)
        counts = [0] * len(rides)
        for _ in range(CALLS):
            route = fn(np.array(rides), np.array(shown), epsilon, rng)
            if route is None:
                return None
            try:
                ok = float(route) == int(route) and 0 <= int(route) < len(rides) and not isinstance(route, bool)
            except (TypeError, ValueError, OverflowError):
                ok = False
            if not ok:
                plain = route.item() if hasattr(route, "item") else route
                return Picks(("other", repr(plain)))
            counts[int(route)] = counts[int(route)] + 1
        return Picks(counts)

    def _call_text(self, case) -> str:
        rides, shown, epsilon = case
        return f"choose({rides!r}, {shown!r}, {epsilon}, rng)"


RIDES_AFTER = _named(_WEEK_RIDES + np.eye(len(ROUTES))[NUMBERS["leader"]] * NUMBERS["pop_got"], "C.RIDES_AFTER")
SHOWN_AFTER = _named(_WEEK_SHOWN + np.eye(len(ROUTES))[NUMBERS["leader"]] * VISITORS, "C.SHOWN_AFTER")
_TWO_RIDES, _TWO_SHOWN = _named([10, 3], "[10, 3]"), _named([100, 10], "[100, 10]")
_THREE_RIDES, _THREE_SHOWN = _named([4, 4, 6], "[4, 4, 6]"), _named([8, 20, 30], "[8, 20, 30]")

_EDGE_APPEAL, _EDGE_RIDES, _EDGE_SHOWN = _named([0.0, 1.0], "[0.0, 1.0]"), _named([1, 0], "[1, 0]"), _named([10, 10], "[10, 10]")
_OTHER_APPEAL = _named([0.5, 0.1, 0.3], "[0.5, 0.1, 0.3]")
_OTHER_RIDES, _OTHER_SHOWN = _named([2, 3, 1], "[2, 3, 1]"), _named([5, 5, 5], "[5, 5, 5]")


LOOPS = Sort(
    name="loops",
    groups={
        "loop": {"en": "A feedback loop: the model's output changes the data it learns from later",
                 "nl": "Een feedback loop: wat het model geeft verandert de data waarvan het later leert"},
        "none": {"en": "No feedback loop: the data do not depend on what the model gives",
                 "nl": "Geen feedback loop: de data hangen niet af van wat het model geeft"},
    },
    items={
        "front": {"en": "**Best sellers first.** A shop puts the products that sold most at the top of its front page, and counts sales again every night.",
                  "nl": "**Bestsellers bovenaan.** Een winkel zet de producten die het meest verkochten bovenaan de voorpagina, en telt elke nacht de verkopen opnieuw."},
        "rain": {"en": "**A rain forecast.** A model forecasts rain for tomorrow from today's weather; the clouds do not read the forecast.",
                 "nl": "**Een regenvoorspelling.** Een model voorspelt regen voor morgen uit het weer van vandaag; de wolken lezen de voorspelling niet."},
        "vans": {"en": "**Bikes where the model expects riders.** The service sends its van with bikes to the stations its model expects to be busy, and those stations then rent more.",
                 "nl": "**Fietsen waar het model ritten verwacht.** De service stuurt het busje met fietsen naar de stations waar het model drukte verwacht, en daar worden dan meer fietsen verhuurd."},
        "train": {"en": "**A train's arrival.** A model predicts when a train arrives from where it is now; the driver never sees the prediction.",
                  "nl": "**De aankomst van een trein.** Een model voorspelt uit de plek van een trein wanneer de trein aankomt; de machinist ziet de voorspelling nooit."},
        "loans": {"en": "**Loans for some.** A bank lends only to applicants its model trusts, and learns later only how those applicants paid back.",
                  "nl": "**Leningen voor sommigen.** Een bank leent alleen aan aanvragers die volgens het model te vertrouwen zijn, en leert later alleen hoe juist deze aanvragers terugbetaalden."},
        "parts": {"en": "**Broken parts.** A model sorts photos of parts on a belt into broken and whole; the parts were made before the model looked.",
                  "nl": "**Kapotte onderdelen.** Een model deelt foto's van onderdelen op een band in als kapot of heel; de onderdelen waren al gemaakt voordat het model keek."},
    },
    solution={"front": "loop", "rain": "none", "vans": "loop", "train": "none", "loans": "loop", "parts": "none"},
    nudge={
        "front": {"en": "Which products get seen most, and which sales are counted the next night?",
                  "nl": "Welke producten worden het meest gezien, en welke verkopen worden de volgende nacht geteld?"},
        "rain": {"en": "Does tomorrow's weather change when the forecast says rain?",
                 "nl": "Verandert het weer van morgen als de voorspelling regen zegt?"},
        "vans": {"en": "Where are more bikes rented once the van has been there, and what does the model learn from next?",
                 "nl": "Waar worden meer fietsen verhuurd zodra het busje er geweest is, en waarvan leert het model daarna?"},
        "train": {"en": "Does the train run faster or slower because of the prediction?",
                  "nl": "Rijdt de trein sneller of langzamer door de voorspelling?"},
        "loans": {"en": "Whose repayments can the bank ever see?",
                  "nl": "Van welke aanvragers kan de bank ooit zien of het geld terugkwam?"},
        "parts": {"en": "Does the model's answer change which parts come down the belt?",
                  "nl": "Verandert het antwoord van het model welke onderdelen er over de band komen?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the situations again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de situaties opnieuw uit."},
        "empty": {"en": "Put each situation with or without a feedback loop.",
                  "nl": "Zet elke situatie bij wel of geen feedback loop."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "*{item}* belongs on the other side.",
                  "nl": "*{item}* hoort aan de andere kant."},
        "correct": {"en": "All {n} are right. The shop, the van and the bank decide with the model's output what they will see next: the products on top sell more, the stations with bikes rent more, and only the trusted applicants show whether they pay back. The forecast, the train and the parts are only watched: the world does not change because of the answer.",
                    "nl": "Alle {n} kloppen. Bij de winkel, het busje en de bank bepaalt wat het model geeft welke data er later komen: de producten bovenaan verkopen meer, de stations met fietsen verhuren meer, en alleen van de vertrouwde aanvragers is te zien of het geld terugkomt. De voorspelling, de trein en de onderdelen worden alleen bekeken: de wereld verandert niet door het antwoord."},
    },
)

EXERCISES = {
    "simulate": FunctionBox(
        name="simulate",
        function="simulate",
        start=_S_START,
        solution=_S,
        names={"np": np},
        visible=[(APPEAL, RIDES, SHOWN, VISITORS, RUN_SEED)],
        hidden=[(APPEAL, RIDES, SHOWN, 3000, 7), (_EDGE_APPEAL, _EDGE_RIDES, _EDGE_SHOWN, 50, 3),
                (_OTHER_APPEAL, _OTHER_RIDES, _OTHER_SHOWN, 1000, 11)],
        first=[_DOTS, _ARRAY],
        show=_shown,
        picture=simulate_picture,
        mistakes=[
            Mistake("inverted", wrong=_S_INVERTED,
                    en="`{call}` gives {got}, and it should give {want}. Your test is true when the random number lies above the chance: route G, with 24 of 100, is taken 76 times in 100. Which side of the random number should the chance be on?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn. Je test is waar als het willekeurige getal boven de kans ligt: route G, met 24 van de 100, wordt 76 keer op de 100 genomen. Aan welke kant van het willekeurige getal hoort de kans?"),
            Mistake("always", wrong=_S_ALWAYS,
                    en="`{call}` gives {got}, and it should give {want}. Your test is true for every route with a chance above 0, so every visitor takes the route. Where does the random number `rng.random()` come in?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn. Je test is waar voor elke route met een kans boven 0, dus neemt elke bezoeker de route. Waar komt het willekeurige getal `rng.random()` in je test?"),
            Mistake("half", wrong=_S_HALF,
                    en="`{call}` gives {got}, and it should give {want}. With `0.5`, every route is taken by half of the visitors who see it. Which route's chance should the random number be compared with?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn. Met `0.5` neemt de helft van de bezoekers die een route zien de route. Met de kans van welke route hoort het willekeurige getal vergeleken te worden?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with a test that is true with chance `appeal[route]`.",
                      "nl": "Vervang de `...` door een test die waar is met kans `appeal[route]`."},
            "arguments": {"en": "The check calls `{call}`, with five values, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(appeal, rides, shown, visitors, seed)`?",
                          "nl": "De check roept `{call}` aan, met vijf waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(appeal, rides, shown, visitors, seed)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your test compare `rng.random()` with the chance of the shown route?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Vergelijkt je test `rng.random()` met de kans van de getoonde route?"},
            "almost": {"en": "Right for the course's routes. But `{call}` gives {got}, and it should give {want}. Does your test use `appeal`, and not eight fixed numbers?",
                       "nl": "Goed voor de routes van de course. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je test `appeal`, en geen acht vaste getallen?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return rides`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return rides`?"},
            "correct": {"en": "Right in all {n} cases. After the first week, route G gets every showing, and with it every ride, while route E is never shown again. In the picture: how many rides did G get, and how many would E have gotten?",
                        "nl": "Goed in alle {n} gevallen. Na de eerste week wordt route G elke keer getoond, en krijgt G daarmee elke rit, terwijl route E nooit meer getoond wordt. In de afbeelding: hoeveel ritten kreeg G, en hoeveel zou E gekregen hebben?"},
        },
    ),
    "choose": ChooseBox(
        name="choose",
        function="choose",
        start=_C_START,
        solution=_C,
        names={"np": np},
        visible=[(RIDES, SHOWN, 0.0)],
        hidden=[(RIDES_AFTER, SHOWN_AFTER, 0.0), (_TWO_RIDES, _TWO_SHOWN, 0.0), (_THREE_RIDES, _THREE_SHOWN, 0.0),
                (RIDES, SHOWN, 0.1), (RIDES, SHOWN, 1.0)],
        first=[_DOTS],
        show=_picks_shown,
        picture=choose_picture,
        mistakes=[
            Mistake("count", wrong=_C_COUNT,
                    en="`{call}` gives, in {calls} calls: {got}; it should give {want}. Your line ranks by rides, so a route that was shown more wins, whatever its visitors think of it. How do you turn rides into rides per showing?",
                    nl="`{call}` geeft, bij {calls} aanroepen: {got}; het moet {want} zijn. Je regel rangschikt op ritten, dus wint een route die vaker getoond werd, wat de bezoekers er ook van vinden. Hoe maak je van ritten de ritten per keer getoond?"),
            Mistake("shown", wrong=_SHOWN,
                    en="`{call}` gives, in {calls} calls: {got}; it should give {want}. Your line picks the route that was shown most, which says how often the app showed it, not how often visitors took it. Which two arrays together say how often a shown route is taken?",
                    nl="`{call}` geeft, bij {calls} aanroepen: {got}; het moet {want} zijn. Je regel kiest de route die het vaakst getoond werd, en dat zegt hoe vaak de app de route toonde, niet hoe vaak bezoekers de route namen. Welke twee arrays zeggen samen hoe vaak een getoonde route genomen wordt?"),
            Mistake("least", wrong=_C_FLIPPED,
                    en="`{call}` gives, in {calls} calls: {got}; it should give {want}. Your line picks the route with the fewest rides per showing: the route visitors take least. Which route should the app show outside the random showings?",
                    nl="`{call}` geeft, bij {calls} aanroepen: {got}; het moet {want} zijn. Je regel kiest de route met de minste ritten per keer getoond: de route die het minst genomen wordt. Welke route hoort de app te tonen buiten de willekeurige routes?"),
            Mistake("rate-not-route", wrong=_C_RATE,
                    en="`{call}` gives {got}: the largest number of rides per showing, not the route that has it. Which NumPy function gives the place of the largest value?",
                    nl="`{call}` geeft {got}: het grootste aantal ritten per keer getoond, niet de route met dat aantal. Welke NumPy-functie geeft de plek van de grootste waarde?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the route that has the most rides per showing.",
                      "nl": "Vervang de `...` door de route met de meeste ritten per keer getoond."},
            "arguments": {"en": "The check calls `{call}`, with four values, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(rides, shown, epsilon, rng)`?",
                          "nl": "De check roept `{call}` aan, met vier waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(rides, shown, epsilon, rng)` houden?"},
            "wrong": {"en": "`{call}` gives, in {calls} calls: {got}; it should give {want}. Does your line give the place of the largest number of rides per showing?",
                      "nl": "`{call}` geeft, bij {calls} aanroepen: {got}; het moet {want} zijn. Geeft je regel de plek van het grootste aantal ritten per keer getoond?"},
            "almost": {"en": "Right for the first week. But `{call}` gives, in {calls} calls: {got}; it should give {want}. Does your line use `rides` and `shown`, and not a fixed route?",
                       "nl": "Goed voor de eerste week. Maar `{call}` geeft, bij {calls} aanroepen: {got}; het moet {want} zijn. Gebruikt je regel `rides` en `shown`, en geen vaste route?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does its last line still begin with `return`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Begint de laatste regel nog met `return`?"},
            "correct": {"en": "Right in all {n} cases. After the first week, route E has the most rides per showing, though route G has the most rides. In the picture: which route does the app show most now, and how many rides per 100 visitors does it get?",
                        "nl": "Goed in alle {n} gevallen. Na de eerste week heeft route E de meeste ritten per keer getoond, al heeft route G de meeste ritten. In de afbeelding: welke route toont de app nu het vaakst, en hoeveel ritten per 100 bezoekers krijgt de app?"},
        },
    ),
    "loops": LOOPS,
}

REQUIRED = ["simulate", "choose", "count-explore", "loops"]


def _fill_calls():
    """The texts of choose say how many calls the check makes."""
    box = EXERCISES["choose"]
    for t in box.texts.values():
        for lang in LANGS:
            t[lang] = t[lang].replace("{calls}", str(CALLS))
    for m in box.mistakes:
        m.en, m.nl = m.en.replace("{calls}", str(CALLS)), m.nl.replace("{calls}", str(CALLS))


_fill_calls()

_SORTED = dict(LOOPS.solution)
_OTHER = {"loop": "none", "none": "loop"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("simulate", _S_START, "empty", None),
    ("simulate", "", "empty", None),
    ("simulate", _S, "correct", None),
    ("simulate", _s("appeal[route] > rng.random()"), "correct", None),
    ("simulate", _S_INVERTED, "wrong", "inverted"),
    ("simulate", _S_ALWAYS, "wrong", "always"),
    ("simulate", _S_HALF, "wrong", "half"),
    ("simulate", _s("rng.random() < appeal"), "wrong", "whole-appeal"),
    ("simulate", _S_FIXED, "almost", None),
    ("simulate", _s("rng.random() < 0.3"), "wrong", None),
    ("simulate", _S.replace("    return rides\n", "    rides\n"), "wrong", "no-return"),
    ("simulate", _S.replace("    return rides\n", "    print(rides)\n"), "wrong", "prints"),
    ("simulate", _S.replace("visitors, seed):", "visitors):"), "error", "arguments"),
    ("simulate", _S.replace("    return rides\n", "    ...\n    return rides\n"), "wrong", "dots-left"),
    ("simulate", "x = 1\n", "wrong", "no-function"),
    ("choose", _C_START, "empty", None),
    ("choose", "", "empty", None),
    ("choose", _C, "correct", None),
    ("choose", _c("(rides / shown).argmax()"), "correct", None),
    ("choose", _C_COUNT, "wrong", "count"),
    ("choose", _SHOWN, "wrong", "shown"),
    ("choose", _C_FLIPPED, "wrong", "least"),
    ("choose", _C_WORST, "wrong", "least"),
    ("choose", _C_RATE, "wrong", "rate-not-route"),
    ("choose", _C_FIXED, "almost", None),
    ("choose", _c("np.argmax(rides - shown)"), "wrong", None),
    ("choose", _C.replace("    return np.argmax(rides / shown)", "    np.argmax(rides / shown)"), "wrong", "unreached"),
    ("choose", _C.replace("    return np.argmax(rides / shown)", "    print(np.argmax(rides / shown))"), "wrong", "prints"),
    ("choose", _C.replace("epsilon, rng):", "epsilon):"), "error", "arguments"),
    ("choose", _C.replace("        return rng.integers", "        ...\n        return rng.integers"), "wrong", "dots-left"),
    ("choose", "x = 1\n", "wrong", "no-function"),
    ("loops", {}, "empty", None),
    ("loops", _SORTED, "correct", None),
    ("loops", {"front": "loop", "rain": "none"}, "wrong", "unfinished"),
] + [("loops", _SORTED | {item: _OTHER[side]}, "wrong", f"wrong:{item}") for item, side in LOOPS.solution.items()]
