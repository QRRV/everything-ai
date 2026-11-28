# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-mlops: MLOps, how far along is a team? The steps of the module as one loop; a team's
level on a simple reading of Microsoft's maturity model, counted from the bottom up to the first step a person still
does; how many months a model serves above the limit when the new model comes some months after the alert; which
step of the loop each practice of the module belongs to; and, in the demo, the error of a model month by month.

The levels come from the providers' own pages, read on 28 November 2026: Microsoft's MLOps maturity model in the
Azure Architecture Center (last updated 3 June 2026), with five levels from "No MLOps" to "Full MLOps automated
operations", and Google Cloud's "MLOps: Continuous delivery and automation pipelines in machine learning", with
levels 0 to 2. The monthly drift of the model is made up by the course."""
import html
import math
from functools import cache

from everything_ai import check, yours
from everything_ai.check import FunctionBox, Mistake, Sort

LESSON = "sys-mlops"
LANGS = ("en", "nl")

LIMIT = 1.5   # an alert when the error is more than 1.5 times the error right after training, as train_again did


class _Named(list):
    """A list from the lesson: the check's call shows it by its name, and the learner's function gets a copy."""

    def __init__(self, name: str, values):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return list(self)


class _Team(dict):
    """A team from the lesson: the check's call shows it by its name, print shows the steps."""

    def __init__(self, name: str, steps: dict):
        super().__init__(steps)
        self.name = name

    def __repr__(self):
        return self.name

    __str__ = dict.__repr__

    def __deepcopy__(self, memo):
        return dict(self)


# the steps of Microsoft's levels 1 to 4, in order: the code of the application built and tested by itself (DevOps),
# training as a pipeline, the release of a model by the pipeline, and retraining that monitoring starts
STEPS = _Named("C.STEPS", ["code", "training", "release", "retraining"])


def _team(key: str, code: bool, training: bool, release: bool, retraining: bool) -> _Team:
    return _Team(f'C.TEAMS["{key}"]', {"code": code, "training": training, "release": release, "retraining": retraining})


TEAMS = {
    "shop": _team("shop", True, True, False, False),      # level 2: the model is copied to the server by hand
    "bikes": _team("bikes", True, False, True, False),    # level 1: releases by itself, but trains by hand
    "bank": _team("bank", True, True, True, False),       # level 3: a person still decides when to train again
    "lab": _team("lab", False, True, False, False),       # level 0: a training pipeline, but the code by hand
}

# how much the error of the model grows in each month of two years, as the world changes: 1.06 is 6% more
DRIFT = _Named("C.DRIFT", [1.06, 1.02, 1.04, 1.05, 1.04, 1.06, 1.25, 1.06, 1.05, 1.06, 1.2, 1.1,
                           1.3, 1.2, 1.04, 1.08, 1.02, 1.1, 1.3, 1.06, 1.08, 1.06, 1.08, 1.09])


def level(team, steps):
    n = 0
    for step in steps:
        if not team[step]:
            break
        n = n + 1
    return n


def bad_months(drift, limit, delay):
    error = 1.0
    waited = 0
    bad = 0
    for grow in drift:
        error = error * grow
        if error > limit:
            bad = bad + 1
            if waited == delay:
                error = 1.0
                waited = 0
            else:
                waited = waited + 1
    return bad


def run(drift, limit: float, delay: int) -> dict:
    """bad_months with the error of every month, the months above the limit, and the months a new model came."""
    error, waited, errors, bad, new = 1.0, 0, [], [], []
    for month, grow in enumerate(drift):
        error = error * grow
        errors.append(error)
        if error > limit:
            bad.append(month)
            if waited == delay:
                error, waited = 1.0, 0
                new.append(month)
            else:
                waited = waited + 1
    return {"errors": errors, "bad": bad, "new": new}


def every(drift, limit: float, months: int) -> dict:
    """A fixed plan: a new model every `months` months, alert or not."""
    error, bad, new = 1.0, [], []
    for month, grow in enumerate(drift):
        error = error * grow
        if error > limit:
            bad.append(month)
        if (month + 1) % months == 0:
            error = 1.0
            new.append(month)
    return {"bad": bad, "new": new}


def gate(tests: dict, new_rmse: float, old_rmse: float) -> tuple[bool, bool]:
    """The worked example: whether every test passed, and whether the new model goes into production."""
    passed = True
    for name in tests:
        if not tests[name]:
            passed = False
    return passed, passed and new_rmse < old_rmse


DELAYS = [0, 1, 2, 3]   # the months of waiting after an alert, under exercise 2 and in the demo
NEW_RMSE = 51.2         # the worked example: the new model on the validation set
OLD_RMSE = 53.04        # the model in production: the validation RMSE of the project's model


@cache
def facts() -> dict:
    """The numbers the texts give."""
    out = {"levels": {k: level(t, list(STEPS)) for k, t in TEAMS.items()},
           "plan6": every(list(DRIFT), LIMIT, 6), "max_error": max(max(run(list(DRIFT), LIMIT, d)["errors"]) for d in DELAYS)}
    for d in DELAYS:
        out[("run", d)] = run(list(DRIFT), LIMIT, d)
    return out


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem;max-width:var(--ea-measure)"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem;max-width:var(--ea-measure)"
_SVG = "width:100%;max-width:330px;height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums"
_OPEN = "fill:var(--ea-page);stroke:var(--ea-ink-3);stroke-width:1"


def number(x: float, lang: str) -> str:
    """A number with only the decimals it needs, at most two: 1.5, 53.04, 9; in Dutch with a decimal comma."""
    out = f"{round(x, 2):g}"
    return out.replace(".", ",") if lang == "nl" else out


def one(x: float, lang: str) -> str:
    """A number with one decimal: 1.6; in Dutch 1,6."""
    out = f"{x:.1f}"
    return out.replace(".", ",") if lang == "nl" else out


WAIT_TEXT = {"en": {0: "no waiting", 1: "1 month of waiting", "more": "{d} months of waiting"},
             "nl": {0: "zonder wachten", 1: "1 maand wachten", "more": "{d} maanden wachten"}}
MONTHS_TEXT = {"en": {1: "1 month", "more": "{n} months"}, "nl": {1: "1 maand", "more": "{n} maanden"}}
WITH_TEXT = {"en": {0: "With no waiting", 1: "With 1 month of waiting", "more": "With {d} months of waiting"},
             "nl": {0: "Zonder wachten", 1: "Met 1 maand wachten", "more": "Met {d} maanden wachten"}}


def wait_text(d: int, lang: str) -> str:
    """'no waiting', '1 month of waiting', '2 months of waiting'; in Dutch 'zonder wachten', '2 maanden wachten'."""
    w = WAIT_TEXT[lang]
    return w[d] if d in w else w["more"].format(d=d)


def with_text(d: int, lang: str) -> str:
    """The start of a sentence: 'With no waiting', 'With 2 months of waiting'; in Dutch 'Zonder wachten',
    'Met 2 maanden wachten', after which the verb comes."""
    w = WITH_TEXT[lang]
    return w[d] if d in w else w["more"].format(d=d)


def months_text(n: int, lang: str) -> str:
    """'1 month', '3 months'; in Dutch '1 maand', '3 maanden'."""
    w = MONTHS_TEXT[lang]
    return w[n] if n in w else w["more"].format(n=n)


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
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{h:.1f}" rx="3" style="{style}"/>'


def _dot(x: float, y: float, r: float, fill: str) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" style="fill:{fill}"/>'


def _arrow(x1: float, y1: float, x2: float, y2: float, stroke: str = "var(--ea-ink-3)", width: float = 1.5) -> str:
    """A line from (x1, y1) to (x2, y2) with a small head at the end."""
    dx, dy = x2 - x1, y2 - y1
    size = math.hypot(dx, dy) or 1.0
    ux, uy = dx / size, dy / size
    a = (x2 - 7 * ux + 3.5 * uy, y2 - 7 * uy - 3.5 * ux)
    b = (x2 - 7 * ux - 3.5 * uy, y2 - 7 * uy + 3.5 * ux)
    head = f'<polygon points="{x2:.1f},{y2:.1f} {a[0]:.1f},{a[1]:.1f} {b[0]:.1f},{b[1]:.1f}" style="fill:{stroke}"/>'
    return _line(x1, y1, x2 - 5 * ux, y2 - 5 * uy, stroke, width) + head


def _triangle(x: float, y: float, fill: str) -> str:
    """A small triangle that points up, with its tip at (x, y)."""
    return f'<polygon points="{x:.1f},{y:.1f} {x - 4.5:.1f},{y + 8:.1f} {x + 4.5:.1f},{y + 8:.1f}" style="fill:{fill}"/>'


# ======== the picture of the idea: the steps of the module as one loop ====================================================

IDEA_TEXT = {
    "en": {"train": "train", "test": "test", "release": "release", "serve": "serve", "watch": "watch",
           "train_sub": "tracked runs", "test_sub": "code and behavior", "release_sub": "registry, image",
           "serve_sub": "an API", "watch_sub": "logs, drift",
           "data": "new data", "ct": "continuous", "ct_2": "training", "cicd": "CI/CD",
           "middle": "one loop", "middle_2": "that runs by itself",
           "alt": "Five steps in a loop: train, test, release, serve and watch. An arrow goes from each step to the next, "
                  "and from watch back to train, marked continuous training. New data comes in at train, and the steps "
                  "from test to serve are marked CI/CD."},
    "nl": {"train": "trainen", "test": "testen", "release": "uitrollen", "serve": "draaien", "watch": "bewaken",
           "train_sub": "bijgehouden runs", "test_sub": "code en gedrag", "release_sub": "registry, image",
           "serve_sub": "een API", "watch_sub": "logs, drift",
           "data": "nieuwe data", "ct": "continuous", "ct_2": "training", "cicd": "CI/CD",
           "middle": "één kringloop", "middle_2": "die uit zichzelf draait",
           "alt": "Vijf stappen in een kringloop: trainen, testen, uitrollen, draaien en bewaken. Een pijl gaat van elke "
                  "stap naar de volgende, en van bewaken terug naar trainen, met continuous training erbij. Nieuwe data "
                  "komt binnen bij trainen, en de stappen van testen tot draaien hebben CI/CD erbij."},
}

LOOP = ("train", "test", "release", "serve", "watch")
_PLACES = {"train": (90, 58), "test": (240, 58), "release": (272, 148), "serve": (165, 214), "watch": (58, 148)}


def idea_picture(lang: str) -> str:
    lang = _lang(lang)
    w = IDEA_TEXT[lang]
    bw, bh = 104, 38
    parts = []
    for k, key in enumerate(LOOP):
        nxt = LOOP[(k + 1) % len(LOOP)]
        (x1, y1), (x2, y2) = _PLACES[key], _PLACES[nxt]
        dx, dy = x2 - x1, y2 - y1
        # leave the boxes: walk from the edge of one box to the edge of the next
        t = min((bw / 2 + 4) / abs(dx) if dx else 9, (bh / 2 + 4) / abs(dy) if dy else 9)
        stroke = "var(--ea-param)" if key == "watch" else "var(--ea-ink-3)"
        width = 2.5 if key == "watch" else 1.5
        parts.append(_arrow(x1 + dx * t, y1 + dy * t, x2 - dx * t, y2 - dy * t, stroke, width))
    for key in LOOP:
        x, y = _PLACES[key]
        parts.append(_rect(x - bw / 2, y - bh / 2, bw, bh, "fill:var(--ea-page);stroke:var(--ea-ink-2);stroke-width:1.2"))
        parts.append(_t(x, y - 2, w[key], 13, 600, "var(--ea-ink)", "middle"))
        parts.append(_t(x, y + 12, w[f"{key}_sub"], 10.5, 400, "var(--ea-ink-2)", "middle"))
    # new data comes in at training
    tx, ty = _PLACES["train"]
    parts.append(_arrow(tx - 40, 8, tx - 40, ty - bh / 2 - 2, "var(--ea-ink-3)", 1.5))
    parts.append(_t(tx - 34, 14, w["data"], 11, 400, "var(--ea-ink-2)"))
    # continuous training: the arrow from watch back to train
    wx, wy = _PLACES["watch"]
    parts.append(_t(0, (wy + ty) / 2 - 4, w["ct"], 11, 600, "var(--ea-param-text)"))
    parts.append(_t(0, (wy + ty) / 2 + 9, w["ct_2"], 11, 600, "var(--ea-param-text)"))
    # CI/CD: from test to serve
    rx, ry = _PLACES["release"]
    parts.append(_t(rx + 30, ry - bh / 2 - 26, w["cicd"], 11, 600, "var(--ea-ink-2)", "middle"))
    parts.append(_t(165, 128, w["middle"], 12, 600, "var(--ea-ink-2)", "middle"))
    parts.append(_t(165, 143, w["middle_2"], 11, 400, "var(--ea-ink-3)", "middle"))
    return _svg(240, w["alt"], parts)


# ======== the pictures under the checks ===================================================================================

YOURS_TEXT = {
    "en": {"level": {"title": "Your level for four teams",
                     "intro": "The four teams of `C.TEAMS`, with a box for each step of `C.STEPS`: filled when the step "
                              "runs by itself, open when a person does it. The number is what your `level` gives, and "
                              "the line stands after that many boxes.",
                     "steps": ["code", "train", "release", "retrain"],
                     "teams": {"shop": "shop", "bikes": "bike rental", "bank": "bank", "lab": "lab"},
                     "mark": "level {n}",
                     "none": "Your `level` stops with an error, takes too long, or gives something other than a whole "
                             "number from 0 to 4, so the picture waits.",
                     "right": "Your function is right for all four. The bike rental releases by itself, but trains by "
                              "hand: it stays at level 1, since a level keeps the steps below it.",
                     "off": "For the {team}, your function gives {got}, and its level is {want}.",
                     "alt": "Four rows, one per team, each with four boxes for the steps code, training, release and "
                            "retraining, filled when the step runs by itself, and the level the learner's function "
                            "gives."},
           "bad": {"title": "Your bad_months for four waits",
                   "intro": "The two years of `C.DRIFT`, with the limit at 1.5: the months above the limit that your "
                            "`bad_months` counts, for 0 to 3 months of waiting after an alert.",
                   "rows": {0: "no wait", 1: "1 month", "more": "{d} months"},
                   "of": "{n} of 24",
                   "none": "Your `bad_months` stops with an error, takes too long, or gives something other than a "
                           "whole number from 0 to 24, so the picture waits.",
                   "right": "Your function is right for all four: each month of waiting adds bad months, from 4 to 12 "
                            "of the 24.",
                   "off": "{with_}, your function gives {got} bad months, and the two years have {want}.",
                   "alt": "Four bars, for 0 to 3 months of waiting after an alert, of the months above the limit that "
                          "the learner's function counts, out of 24."}},
    "nl": {"level": {"title": "Je level voor vier teams",
                     "intro": "De vier teams van `C.TEAMS`, met een vakje voor elke stap van `C.STEPS`: gevuld als de "
                              "stap vanzelf draait, open als een mens de stap doet. Het getal is wat je `level` geeft, en "
                              "de lijn staat na zoveel vakjes.",
                     "steps": ["code", "trainen", "uitrollen", "hertrainen"],
                     "teams": {"shop": "winkel", "bikes": "fietsverhuur", "bank": "bank", "lab": "lab"},
                     "mark": "niveau {n}",
                     "none": "Je `level` stopt met een fout, duurt te lang, of geeft geen heel getal van 0 tot en met 4, "
                             "dus wacht de afbeelding nog.",
                     "right": "Je functie klopt voor alle vier. De fietsverhuur rolt vanzelf uit, maar traint met de "
                              "hand: dat blijft niveau 1, want een niveau houdt de stappen eronder.",
                     "off": "Voor de {team} geeft je functie {got}, en het niveau is {want}.",
                     "alt": "Vier rijen, één per team, elk met vier vakjes voor de stappen code, trainen, uitrollen en "
                            "hertrainen, gevuld als de stap vanzelf draait, en het niveau dat de functie van de leerling "
                            "geeft."},
           "bad": {"title": "Je bad_months voor vier keer wachten",
                   "intro": "De twee jaar van `C.DRIFT`, met de grens op 1,5: de maanden boven de grens die je "
                            "`bad_months` telt, bij 0 tot 3 maanden wachten na een alert.",
                   "rows": {0: "niet wachten", 1: "1 maand", "more": "{d} maanden"},
                   "of": "{n} van 24",
                   "none": "Je `bad_months` stopt met een fout, duurt te lang, of geeft geen heel getal van 0 tot en "
                           "met 24, dus wacht de afbeelding nog.",
                   "right": "Je functie klopt voor alle vier: elke maand wachten maakt meer slechte maanden, van 4 tot "
                            "12 van de 24.",
                   "off": "{with_} geeft je functie {got} slechte maanden, en de twee jaar hebben er {want}.",
                   "alt": "Vier balken, voor 0 tot 3 maanden wachten na een alert, van de maanden boven de grens die de "
                          "functie van de leerling telt, van de 24."}},
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


def _whole_number(got, top: int):
    """got as a whole number from 0 to top, or _STOPPED when it is not one."""
    if isinstance(got, bool) or not isinstance(got, (int, float)) or not math.isfinite(got):
        return _STOPPED
    if got != int(got) or not 0 <= got <= top:
        return _STOPPED
    return int(got)


def level_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["level"]
    fn = _fn(seen)
    got = {}
    for key, team in TEAMS.items():
        v = _call(lambda: fn(dict(team), list(STEPS))) if fn else _STOPPED
        got[key] = _whole_number(v, len(STEPS)) if v is not _STOPPED else _STOPPED
    if any(v is _STOPPED for v in got.values()):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    x0, step, bw = 86, 46, 38
    parts = []
    for k, name in enumerate(w["steps"]):
        parts.append(_t(x0 + k * step + bw / 2, 12, name, 9.5, 400, "var(--ea-ink-2)", "middle"))
    y = 20
    for key, team in TEAMS.items():
        parts.append(_t(0, y + 13, w["teams"][key], 11.5, 400, "var(--ea-ink)"))
        for k, s in enumerate(STEPS):
            style = "fill:var(--ea-ink)" if team[s] else _OPEN
            parts.append(_rect(x0 + k * step, y + 1, bw, 16, style))
        n = got[key]
        mx = x0 + n * step - 3
        parts.append(_line(mx, y - 3, mx, y + 21, "var(--ea-ink)", 2))
        parts.append(_t(x0 + 4 * step + 4, y + 13, w["mark"].format(n=n), 11.5, 600))
        y += 28
    wrong = [k for k in TEAMS if got[k] != facts()["levels"][k]]
    if not wrong:
        said = w["right"]
    else:
        k = wrong[0]
        said = w["off"].format(team=w["teams"][k], got=got[k], want=facts()["levels"][k])
    body = _note(w["intro"]) + _svg(y + 2, w["alt"], parts) + _note(said)
    return yours.figure(w["title"], body, alt=w["title"])


def _row(d: int, lang: str) -> str:
    rows = YOURS_TEXT[lang]["bad"]["rows"]
    return rows[d] if d in rows else rows["more"].format(d=d)


def bad_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["bad"]
    fn = _fn(seen)
    got = {}
    for d in DELAYS:
        v = _call(lambda: fn(list(DRIFT), LIMIT, d)) if fn else _STOPPED
        got[d] = _whole_number(v, len(DRIFT)) if v is not _STOPPED else _STOPPED
    if any(v is _STOPPED for v in got.values()):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    x0, W = 92, 160
    parts, y = [], 4
    for d in DELAYS:
        parts.append(_t(0, y + 14, _row(d, lang), 11.5, 400, "var(--ea-ink-2)"))
        parts.append(_rect(x0, y + 3, W, 15, _OPEN))
        parts.append(_rect(x0, y + 3, max(W * got[d] / len(DRIFT), 2.0), 15, "fill:var(--ea-loss)"))
        parts.append(_t(x0 + W + 6, y + 14.5, w["of"].format(n=got[d]), 11.5, 600))
        y += 24
    f = facts()
    wrong = [d for d in DELAYS if got[d] != len(f[("run", d)]["bad"])]
    if not wrong:
        said = w["right"]
    else:
        d = wrong[0]
        said = w["off"].format(with_=with_text(d, lang), got=got[d], want=len(f[("run", d)]["bad"]))
    body = _note(w["intro"]) + _svg(y + 4, w["alt"], parts) + _note(said)
    return yours.figure(w["title"], body, alt=w["title"])


# ======== the demo: the error of the model month by month ==================================================================

DEMO_TEXT = {
    "en": {"title": "{wait}: {b} of 24 months above the limit",
           "limit": "limit 1.5", "month": "month", "error": "error, 1 = right after training",
           "bad": "month above the limit", "new": "new model",
           "say": "{with_}, the error is above the limit in {b} of the 24 months: {a} alerts, each with {k} above "
                  "the limit before the new model is there.",
           "alt": "The error of a model month by month for two years, against a limit at 1.5. The error grows each "
                  "month; months above the limit are marked, and the error falls back to 1 when a new model comes."},
    "nl": {"title": "{wait}: {b} van 24 maanden boven de grens",
           "limit": "grens 1,5", "month": "maand", "error": "fout, 1 = net na de training",
           "bad": "maand boven de grens", "new": "nieuw model",
           "say": "{with_} ligt de fout in {b} van de 24 maanden boven de grens: {a} alerts, elk met {k} boven de "
                  "grens tot het nieuwe model er is.",
           "alt": "De fout van een model maand na maand, twee jaar lang, tegen een grens op 1,5. De fout groeit elke "
                  "maand; maanden boven de grens zijn gemarkeerd, en de fout valt terug naar 1 als er een nieuw model "
                  "komt."},
}


def _cap(text_: str) -> str:
    return text_[:1].upper() + text_[1:]


def frame(value: float, lang: str) -> tuple[str, str]:
    lang = _lang(lang)
    w = DEMO_TEXT[lang]
    d = int(value)
    r = facts()[("run", d)]
    b, a = len(r["bad"]), len(r["new"])
    parts = [_t(0, 15, _cap(w["title"].format(wait=wait_text(d, lang), b=b)), 12, 600)]
    parts.append(_t(0, 34, w["error"], 10.5, 400, "var(--ea-ink-2)"))
    x0, x1, top, bottom = 30, 322, 44, 176
    lo, hi = 0.9, 2.4

    def X(month):
        return x0 + (x1 - x0) * (month + 0.5) / len(DRIFT)

    def Y(e):
        return bottom - (bottom - top) * (e - lo) / (hi - lo)

    for e in (1.0, 2.0):
        parts.append(_line(x0, Y(e), x1, Y(e), "var(--ea-line)", 1))
        parts.append(_t(x0 - 5, Y(e) + 4, one(e, lang), 10.5, 400, "var(--ea-ink-3)", "end"))
    parts.append(_line(x0, Y(LIMIT), x1, Y(LIMIT), "var(--ea-ink-3)", 1.2, "4 3"))
    parts.append(_t(x0 - 5, Y(LIMIT) + 4, one(LIMIT, lang), 10.5, 400, "var(--ea-ink-3)", "end"))
    parts.append(_t(x0 + 4, Y(LIMIT) - 5, w["limit"], 10.5, 600, "var(--ea-ink-2)"))
    points = " ".join(f"{X(m):.1f},{Y(e):.1f}" for m, e in enumerate(r["errors"]))
    parts.append(f'<polyline points="{points}" style="fill:none;stroke:var(--ea-ink);stroke-width:1.8"/>')
    for m in r["bad"]:
        parts.append(_dot(X(m), Y(r["errors"][m]), 3.5, "var(--ea-loss)"))
    for m in r["new"]:
        parts.append(_triangle(X(m) + (x1 - x0) / len(DRIFT) / 2, bottom + 3, "var(--ea-param)"))
    for m in (1, 6, 12, 18, 24):
        parts.append(_t(X(m - 1), bottom + 24, str(m), 10.5, 400, "var(--ea-ink-3)", "middle"))
    parts.append(_t((x0 + x1) / 2, bottom + 38, w["month"], 11, 400, "var(--ea-ink-2)", "middle"))
    y = bottom + 58
    parts.append(_dot(6, y - 4, 3.5, "var(--ea-loss)") + _t(16, y, w["bad"], 11.5, 400, "var(--ea-ink-2)"))
    parts.append(_triangle(186, y - 10, "var(--ea-param)") + _t(196, y, w["new"], 11.5, 400, "var(--ea-ink-2)"))
    say = w["say"].format(with_=with_text(d, lang), b=b, a=a, k=months_text(d + 1, lang))
    return _svg(y + 8, w["alt"], parts), say


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


# exercise 1: level
def _l(condition: str) -> str:
    return ("def level(team, steps):\n"
            "    n = 0\n"
            "    for step in steps:\n"
            f"        if {condition}:\n"
            "            break\n"
            "        n = n + 1\n"
            "    return n\n")


_L_START = _l("...")
_L = _l("not team[step]")
_L_FLIPPED = _l("team[step]")
_L_COUNT = ("def level(team, steps):\n"
            "    n = 0\n"
            "    for step in steps:\n"
            "        if team[step]:\n"
            "            n = n + 1\n"
            "    return n\n")
_L_STEP = _l("not step")


# exercise 2: bad_months
def _m(condition: str) -> str:
    return ("def bad_months(drift, limit, delay):\n"
            "    error = 1.0\n"
            "    waited = 0\n"
            "    bad = 0\n"
            "    for grow in drift:\n"
            "        error = error * grow\n"
            "        if error > limit:\n"
            "            bad = bad + 1\n"
            f"            if {condition}:\n"
            "                error = 1.0\n"
            "                waited = 0\n"
            "            else:\n"
            "                waited = waited + 1\n"
            "    return bad\n")


_M_START = _m("...")
_M = _m("waited == delay")
_M_LATE = _m("waited > delay")
_M_NEVER = _m("False")
_M_AT_ONCE = _m("True")


LOOP_SORT = Sort(
    name="loop",
    groups={
        "train": {"en": "Train", "nl": "Trainen"},
        "test": {"en": "Test", "nl": "Testen"},
        "release": {"en": "Release", "nl": "Uitrollen"},
        "watch": {"en": "Watch", "nl": "Bewaken"},
    },
    items={
        "seeds": {"en": "**Seeds and pinned packages.** The same run gives the same model, on any computer.",
                  "nl": "**Seeds en vaste versies.** Dezelfde run geeft hetzelfde model, op elke computer."},
        "runs": {"en": "**A line per run.** The settings and the score of each run, saved in a file.",
                 "nl": "**Een regel per run.** De instellingen en de score van elke run, bewaard in een bestand."},
        "loss": {"en": "**The loss falls on one batch.** A check that the training code can learn at all.",
                 "nl": "**De loss daalt op één batch.** Een controle dat de code van de training überhaupt kan leren."},
        "warmth": {"en": "**Warmer weather, more bikes.** What the model does when one input changes.",
                   "nl": "**Warmer weer, meer fietsen.** Wat het model doet als één input verandert."},
        "registry": {"en": "**A version in the registry.** The hash of the file, and which version is in production.",
                     "nl": "**Een versie in de registry.** De hash van het bestand, en welke versie in productie staat."},
        "image": {"en": "**A container image.** The service with its packages and the model, the same on every server.",
                  "nl": "**Een container image.** De service met de packages en het model, hetzelfde op elke server."},
        "p99": {"en": "**The 99th percentile.** How long the slowest answers take, from a log of requests.",
                "nl": "**Het 99e percentiel.** Hoe lang de traagste antwoorden duren, uit een log van verzoeken."},
        "psi": {"en": "**PSI per month.** How far the inputs have moved from the data the model learned from.",
                "nl": "**PSI per maand.** Hoe ver de inputs zijn verschoven van de data waar het model van leerde."},
    },
    solution={"seeds": "train", "runs": "train", "loss": "test", "warmth": "test", "registry": "release",
              "image": "release", "p99": "watch", "psi": "watch"},
    nudge={
        "seeds": {"en": "When do seeds and pinned packages matter: while a model is made, or while it answers?",
                  "nl": "Wanneer tellen seeds en vaste versies: terwijl een model gemaakt wordt, of terwijl het model antwoordt?"},
        "runs": {"en": "A line per run is written at the moment a run ends. Which step makes runs?",
                 "nl": "Een regel per run wordt geschreven op het moment dat een run klaar is. Welke stap maakt runs?"},
        "loss": {"en": "This one does not make a model for production: it asks whether the code is right. Which step asks that?",
                 "nl": "Deze maakt geen model voor productie: deze vraagt of de code klopt. Welke stap vraagt dat?"},
        "warmth": {"en": "This asks whether a trained model does what it should, before anyone uses it. Which step is that?",
                   "nl": "Dit vraagt of een getraind model doet wat het moet, voordat iemand het model gebruikt. Welke stap is dat?"},
        "registry": {"en": "The registry says which version goes out. Which step puts a version into production?",
                     "nl": "De registry zegt welke versie naar buiten gaat. Welke stap zet een versie in productie?"},
        "image": {"en": "An image is built once and then started on servers. Which step brings the model to the servers?",
                  "nl": "Een image wordt één keer gebouwd en dan op servers gestart. Welke stap brengt het model naar de servers?"},
        "p99": {"en": "A log of requests only exists once the model answers real requests. Which step reads it?",
                "nl": "Een log van verzoeken bestaat pas als het model echte verzoeken beantwoordt. Welke stap leest de log?"},
        "psi": {"en": "PSI compares the inputs of now with the inputs of then. Which step keeps an eye on a running model?",
                "nl": "PSI vergelijkt de inputs van nu met de inputs van toen. Welke stap houdt een draaiend model in de gaten?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the eight practices again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de acht werkwijzen opnieuw uit."},
        "empty": {"en": "Put each practice under the step of the loop it belongs to.",
                  "nl": "Zet elke werkwijze onder de stap van de kringloop waar de werkwijze bij hoort."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "*{item}* belongs to another step.",
                  "nl": "*{item}* hoort bij een andere stap."},
        "correct": {"en": "All {n} are right. Each lesson of this module built one step of the loop by hand: a pipeline runs the same steps one after another, and the watch step starts the training again.",
                    "nl": "Alle {n} kloppen. Elke les van deze module bouwde één stap van de kringloop met de hand: een pipeline draait dezelfde stappen na elkaar, en de stap bewaken start de training opnieuw."},
    },
)

EXERCISES = {
    "level": FunctionBox(
        name="level",
        function="level",
        start=_L_START,
        solution=_L,
        visible=[(TEAMS["shop"], STEPS)],
        hidden=[(TEAMS["bikes"], STEPS), (TEAMS["bank"], STEPS), (TEAMS["lab"], STEPS),
                ({"code": True, "training": True, "release": True, "retraining": True}, STEPS),
                ({"build": True, "test": True}, ["build", "test"]), ({"build": False}, ["build"])],
        first=[_DOTS],
        picture=level_picture,
        mistakes=[
            Mistake("count", wrong=_L_COUNT,
                    en="`{call}` gives {got}: it counts every step that runs by itself, also above a step done by hand, and it should give {want}. Can a team be at a level whose lower step a person still does?",
                    nl="`{call}` geeft {got}: het telt elke automatische stap, ook boven een stap met de hand, en het moet {want} zijn. Kan een team op een niveau staan waarvan een lagere stap nog met de hand gaat?"),
            Mistake("flipped", wrong=_L_FLIPPED,
                    en="`{call}` gives {got}: the loop stops at the first step that runs by itself, and it should give {want}. At which kind of step should the counting stop?",
                    nl="`{call}` geeft {got}: de loop stopt bij de eerste automatische stap, en het moet {want} zijn. Bij welke soort stap moet het tellen stoppen?"),
            Mistake("step", wrong=_L_STEP,
                    en="`{call}` gives {got}: the question looks at the name of the step, which is never empty, and it should give {want}. Where does the team keep whether a step runs by itself?",
                    nl="`{call}` geeft {got}: de vraag kijkt naar de naam van de stap, en een naam is nooit leeg, en het moet {want} zijn. Waar houdt het team bij of een stap vanzelf draait?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the question whether a person still does this step.",
                      "nl": "Vervang de `...` door de vraag of een mens deze stap nog doet."},
            "arguments": {"en": "The check calls `{call}`, with a team and the list of steps, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(team, steps)`?",
                          "nl": "De check roept `{call}` aan, met een team en de lijst stappen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(team, steps)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does the loop count the steps from the start and stop at the first one a person does?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Telt de loop de stappen vanaf het begin, en stopt de loop bij de eerste stap die een mens doet?"},
            "almost": {"en": "Right for the shop. But `{call}` gives {got}, and it should give {want}. Does your function use `team` and `steps`, and no fixed numbers or names?",
                       "nl": "Goed voor de winkel. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie `team` en `steps`, en geen vaste getallen of namen?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return n`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return n`?"},
            "correct": {"en": "Right in all {n} cases. The shop is at level 2, and the bike rental at level 1, although it releases by itself: its training is still done by hand. In the picture: which step would move the shop up?",
                        "nl": "Goed in alle {n} gevallen. De winkel staat op niveau 2, en de fietsverhuur op niveau 1, hoewel de fietsverhuur vanzelf uitrolt: de training gaat nog met de hand. In de afbeelding: welke stap brengt de winkel een niveau hoger?"},
        },
    ),
    "bad_months": FunctionBox(
        name="bad_months",
        function="bad_months",
        start=_M_START,
        solution=_M,
        visible=[(DRIFT, LIMIT, 2)],
        hidden=[(DRIFT, LIMIT, 0), (DRIFT, LIMIT, 1), (DRIFT, LIMIT, 3), (DRIFT, 1.4, 0), ([1.0, 1.0, 1.0], LIMIT, 2),
                ([1.3, 1.3, 1.3, 1.3], LIMIT, 1), ([1.6], LIMIT, 5)],
        first=[_DOTS],
        picture=bad_picture,
        mistakes=[
            Mistake("late", wrong=_M_LATE,
                    en="`{call}` gives {got}: the team waits one month more than `delay` at every alert, and it should give {want}. After how many months of waiting is the new model there?",
                    nl="`{call}` geeft {got}: het team wacht bij elk alert één maand langer dan `delay`, en het moet {want} zijn. Na hoeveel maanden wachten is het nieuwe model er?"),
            Mistake("never", wrong=_M_NEVER,
                    en="`{call}` gives {got}: `error` never goes back to `1.0`, as if nobody trained again, and it should give {want}. When does the new model come?",
                    nl="`{call}` geeft {got}: `error` gaat nooit terug naar `1.0`, alsof niemand opnieuw traint, en het moet {want} zijn. Wanneer komt het nieuwe model?"),
            Mistake("at-once", wrong=_M_AT_ONCE,
                    en="`{call}` gives {got}: every alert gets a new model by the next month, as if `delay` were 0, and it should give {want}. Where does your function use `delay`?",
                    nl="`{call}` geeft {got}: elk alert krijgt al de volgende maand een nieuw model, alsof `delay` 0 is, en het moet {want} zijn. Waar gebruikt je functie `delay`?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the question whether the team has waited long enough for the new model.",
                      "nl": "Vervang de `...` door de vraag of het team lang genoeg op het nieuwe model heeft gewacht."},
            "arguments": {"en": "The check calls `{call}`, with the drift of each month, the limit and the months of waiting, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(drift, limit, delay)`?",
                          "nl": "De check roept `{call}` aan, met de drift van elke maand, de grens en de maanden wachten, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(drift, limit, delay)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does `error` go back to `1.0` exactly when `waited` has reached `delay`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Gaat `error` terug naar `1.0` precies als `waited` bij `delay` is?"},
            "almost": {"en": "Right for the team of the check, with 2 months of waiting. But `{call}` gives {got}, and it should give {want}. Does your function use `drift`, `limit` and `delay`, and no fixed numbers?",
                       "nl": "Goed voor het team van de check, met 2 maanden wachten. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie `drift`, `limit` en `delay`, en geen vaste getallen?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return bad`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return bad`?"},
            "correct": {"en": "Right in all {n} cases. With 2 months of waiting, the two years have 9 months above the limit; with the new model by the next month, 4. In the picture: what does each month of waiting add?",
                        "nl": "Goed in alle {n} gevallen. Met 2 maanden wachten liggen 9 maanden van de twee jaar boven de grens; met het nieuwe model de volgende maand 4. In de afbeelding: wat voegt elke maand wachten toe?"},
        },
    ),
    "loop": LOOP_SORT,
}

# The lesson is done when these are correct. "next" is the choice question before the demo.
REQUIRED = ["level", "bad_months", "loop", "next"]

_SORTED = dict(LOOP_SORT.solution)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("level", _L_START, "empty", None),
    ("level", "", "empty", None),
    ("level", _L, "correct", None),
    ("level", _l("team[step] == False"), "correct", None),
    ("level", _l("team[step] is False"), "correct", None),
    ("level", "def level(team, steps):\n    n = 0\n    while n < len(steps) and team[steps[n]]:\n        n = n + 1\n"
              "    return n\n", "correct", None),
    ("level", _L_COUNT, "wrong", "count"),
    ("level", _L_FLIPPED, "wrong", "flipped"),
    ("level", _L_STEP, "wrong", "step"),
    ("level", _L.replace("    return n\n", "    return n + 1\n"), "wrong", None),
    ("level", "def level(team, steps):\n    return 2\n", "almost", None),
    ("level", _l("not team[stap]"), "error", None),
    ("level", _L.replace("def level(team, steps):", "def level(team):"), "error", "arguments"),
    ("level", _L.replace("    return n\n", ""), "wrong", "no-return"),
    ("level", _L.replace("    return n\n", "    print(n)\n"), "wrong", "prints"),
    ("level", _L.replace("    return n\n", "    extra = ...\n    return n\n"), "wrong", "dots-left"),
    ("level", "x = 1\n", "wrong", "no-function"),
    ("bad_months", _M_START, "empty", None),
    ("bad_months", "", "empty", None),
    ("bad_months", _M, "correct", None),
    ("bad_months", _m("waited >= delay"), "correct", None),
    ("bad_months", _m("delay == waited"), "correct", None),
    ("bad_months", _M_LATE, "wrong", "late"),
    ("bad_months", _M_NEVER, "wrong", "never"),
    ("bad_months", _M_AT_ONCE, "wrong", "at-once"),
    ("bad_months", _m("waited == delay + 1"), "wrong", "late"),
    ("bad_months", _m("waited == delay - 1"), "wrong", None),
    ("bad_months", "def bad_months(drift, limit, delay):\n    return 9\n", "almost", None),
    ("bad_months", _m("waited == wait"), "error", None),
    ("bad_months", _M.replace("def bad_months(drift, limit, delay):", "def bad_months(drift, delay):"), "error", "arguments"),
    ("bad_months", _M.replace("    return bad\n", ""), "wrong", "no-return"),
    ("bad_months", _M.replace("    return bad\n", "    print(bad)\n"), "wrong", "prints"),
    ("bad_months", _M.replace("    return bad\n", "    extra = ...\n    return bad\n"), "wrong", "dots-left"),
    ("bad_months", "x = 1\n", "wrong", "no-function"),
    ("loop", {}, "empty", None),
    ("loop", _SORTED, "correct", None),
    ("loop", {"seeds": "train", "loss": "test"}, "wrong", "unfinished"),
    ("loop", _SORTED | {"seeds": "release"}, "wrong", "wrong:seeds"),
    ("loop", _SORTED | {"runs": "watch"}, "wrong", "wrong:runs"),
    ("loop", _SORTED | {"loss": "train"}, "wrong", "wrong:loss"),
    ("loop", _SORTED | {"warmth": "watch"}, "wrong", "wrong:warmth"),
    ("loop", _SORTED | {"registry": "train"}, "wrong", "wrong:registry"),
    ("loop", _SORTED | {"image": "test"}, "wrong", "wrong:image"),
    ("loop", _SORTED | {"p99": "test"}, "wrong", "wrong:p99"),
    ("loop", _SORTED | {"psi": "train"}, "wrong", "wrong:psi"),
]
