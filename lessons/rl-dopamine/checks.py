# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rl-dopamine: the TD model of conditioning gives every step after a light its own V, and its
error at every step, δ_t = r_t + γV_t − V_{t−1}, moves from the juice to the light while it learns, and drops below 0
when the juice stays away, as the dopamine neurons in the recordings of Schultz did. The course's trials, errors and
learn, the functions the learner writes, the pictures, and the demo.

The model is the one of Schultz, Dayan and Montague (1997), with one V for every step after the light, as their
x_i(t) is 1 exactly i steps after the light. The numbers and the pictures are the course's own."""
import copy
import html
import math

import numpy as np

from everything_ai import live, yours
from everything_ai.check import FunctionBox, Mistake, Timeout

LESSON = "rl-dopamine"
LANGS = ("en", "nl")
STEPS = 9          # the steps of a trial, counted from the light: step 0 is the moment the light comes on
JUICE = 4          # the step at which the juice comes
ETA = 0.2          # the learning rate
GAMMA = 1          # no discounting within a trial of a few seconds
TRIALS = 60        # the trials with juice, before the trial without
BEFORE = 2         # the steps before the light that a picture shows, where nothing happens
DELAY_OPTIONS = (2, 4, 6)          # the demo: the step of the juice
GAMMA_OPTIONS = (1, 0.9, 0.8)      # the demo: γ
RIDGE_TRIALS = (1, 5, 10, 20, 30, 60)   # the trials in the picture of the learner's learn, then one without juice


# --- the functions of the exercises, as the course writes them ----------------------------------------------------------

def errors(V, rewards, gamma):
    deltas = np.zeros(len(V))
    before = 0.0
    for t in range(len(V)):
        deltas[t] = rewards[t] + gamma * V[t] - before
        before = V[t]
    return deltas


def learn(V, rewards, eta, gamma):
    deltas = errors(V, rewards, gamma)
    new = np.zeros(len(V))
    for t in range(len(V) - 1):
        new[t] = V[t] + eta * deltas[t + 1]
    return new


# --- the course's trials ------------------------------------------------------------------------------------------------

def juice_at(step: int = JUICE, amount: float = 1, steps: int = STEPS) -> np.ndarray:
    """The juice at every step of a trial: `amount` at `step`, 0 elsewhere; no juice with step None."""
    rewards = np.zeros(steps, dtype=int if float(amount).is_integer() else float)
    if step is not None:
        rewards[step] = amount
    return rewards


def schedule(delay: int = JUICE, trials: int = TRIALS) -> list:
    """The trials of an experiment: `trials` with juice at step `delay`, then one without juice."""
    return [juice_at(delay) for _ in range(trials)] + [juice_at(None)]


def train(learn_=learn, delay: int = JUICE, eta: float = ETA, gamma: float = GAMMA, trials: int = TRIALS) -> np.ndarray:
    """The V's before every trial of `schedule(delay, trials)` and after the last: a row per moment, from V's of 0."""
    V = np.zeros(STEPS)
    seen = [V]
    for rewards in schedule(delay, trials):
        V = np.asarray(learn_(V.copy(), rewards.copy(), eta, gamma), dtype=float)
        seen.append(V)
    return np.array(seen)


def trial_errors(values, delay: int = JUICE, gamma: float = GAMMA, trials: int = TRIALS) -> np.ndarray:
    """The error at every step of every trial: a row per trial, from the V's before it."""
    return np.array([errors(values[k], rewards, gamma) for k, rewards in enumerate(schedule(delay, trials))])


LEARNED = train()[TRIALS]   # the V's after 60 trials: about 1 from the light to the step before the juice


# --- words and numbers ---------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem;max-width:var(--ea-measure)"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem;max-width:var(--ea-measure)"


def _marked(text_: str) -> str:
    """Escaped text in which `name` is set as code and *word* in italics."""
    out = []
    for k, part in enumerate(html.escape(text_).split("`")):
        if k % 2:
            out.append(f"<code>{part}</code>")
        else:
            out.append("".join(f"<i>{p}</i>" if j % 2 else p for j, p in enumerate(part.split("*"))))
    return "".join(out)


def _note(text_: str, style: str = _UI) -> str:
    return f'<div style="{style}">{_marked(text_)}</div>'


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch, and a real minus sign."""
    out = f"{x:,.{digits}f}"
    if out.startswith("-") and float(out.replace(",", "")) == 0:
        out = out[1:]
    out = out.replace("-", "−")
    return out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out


def short_number(x: float, lang: str, digits: int = 2) -> str:
    """A number rounded to `digits` decimals, without the zeros at its end: 0.6, 0.12, 1."""
    out = number(x, digits, lang)
    point = "," if lang == "nl" else "."
    if point in out:
        out = out.rstrip("0").rstrip(point)
    return "0" if out in ("−0", "") else out


# --- drawing ------------------------------------------------------------------------------------------------------------

_HALO = "paint-order:stroke;stroke:var(--ea-page);stroke-width:3px;stroke-linejoin:round"
TOP = 38   # the room above a picture for its title: a name, and a line under it


def _title(x: float, y: float, words: str) -> str:
    """A title of one or two lines: the part before ": " in bold, the rest in gray under it."""
    name, _, rest = words.partition(": ")
    out = f'<text x="{x}" y="{y}" style="font-size:13.5px;font-weight:600;fill:var(--ea-ink)">{html.escape(name)}</text>'
    if rest:
        out += f'<text x="{x}" y="{y + 16}" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(rest)}</text>'
    return out


def _words(x: float, y: float, text_: str, anchor: str = "middle", size: float = 11.5, fill: str = "var(--ea-ink-2)",
           weight: int = 400) -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" style="font-size:{size}px;font-weight:{weight};fill:{fill};'
            f'text-anchor:{anchor};{_HALO}">{html.escape(text_)}</text>')


def _svg(width: float, height: float, parts: list) -> str:
    return (f'<svg viewBox="0 0 {width:.0f} {height:.0f}" aria-hidden="true" style="flex:0 1 auto;width:100%;'
            f'max-width:{width:.0f}px;height:auto;font-family:var(--ea-font-ui)">{"".join(parts)}</svg>')


def _figure(svgs: list, alt: str, width: int, gap: str = "0.6rem 1.4rem") -> str:
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(alt)}" style="margin:0.25rem 0 0.5rem;'
            f'max-width:min(100%,{width}px);display:flex;flex-wrap:wrap;gap:{gap};align-items:flex-start">'
            f'{"".join(svgs)}</figure>')


def _plot(points: list, x0: float, y0: float, w: float, h: float, x_last: float, low: float, high: float, color: str,
          width: float = 2.2, dashed: bool = False) -> str:
    """A line through (x, value) points, x from 0 at x0 to x_last at x0 + w, values between low and high."""
    pts = []
    for x, v in points:
        shown = min(max(float(v), low), high)
        pts.append(f"{x0 + x / x_last * w:.1f},{y0 + h - (shown - low) / (high - low) * h:.1f}")
    dash = ";stroke-dasharray:5 4" if dashed else ""
    return (f'<polyline points="{" ".join(pts)}" style="fill:none;stroke:{color};stroke-width:{width};'
            f'stroke-linejoin:round{dash}"/>')


def _light(x: float, y: float) -> str:
    """The light: a small circle with short rays."""
    rays = "".join(f'<line x1="{x + 6.5 * math.cos(a):.1f}" y1="{y + 6.5 * math.sin(a):.1f}" '
                   f'x2="{x + 9.5 * math.cos(a):.1f}" y2="{y + 9.5 * math.sin(a):.1f}" style="stroke:var(--ea-ink-2);stroke-width:1.1"/>'
                   for a in (k * math.pi / 4 for k in range(8)))
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" style="fill:var(--ea-page);stroke:var(--ea-ink);stroke-width:1.3"/>'
            + rays)


def _drop(x: float, y: float, full: bool = True) -> str:
    """The juice: a drop, filled when it comes, an outline with a dashed edge when it stays away."""
    d = (f"M {x:.1f} {y - 8:.1f} C {x + 1.5:.1f} {y - 4:.1f} {x + 5.5:.1f} {y - 1:.1f} {x + 5.5:.1f} {y + 2.5:.1f} "
         f"A 5.5 5.5 0 0 1 {x - 5.5:.1f} {y + 2.5:.1f} C {x - 5.5:.1f} {y - 1:.1f} {x - 1.5:.1f} {y - 4:.1f} {x:.1f} {y - 8:.1f} Z")
    if full:
        return f'<path d="{d}" style="fill:var(--ea-ink-2);stroke:var(--ea-ink-2);stroke-width:1"/>'
    return f'<path d="{d}" style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.1;stroke-dasharray:2.5 2"/>'


# --- the trace of a trial: V and the error at every step ------------------------------------------------------------------

_IDEA_TEXT = {
    "en": {"first": "Trial 1: the juice as a surprise", "learned": "Trial 60: the light predicts the juice",
           "omitted": "Trial 61: the juice stays away",
           "light": "light", "juice": "juice", "no_juice": "no juice", "step": "step after the light",
           "v": "V", "delta": "δ",
           "alt_trials": "Three charts of one trial each, from two steps before the light to step 8: above, the V of every step, and below, the error δ of every step. Trial 1: all V's are 0, and δ is 0 everywhere except 1 at step 4, where the juice comes. Trial 60: the V's are 1 from the light to step 3, and δ is 1 at the light, step 0, and 0 at the juice. Trial 61, with the same V's but no juice: δ is 1 at the light and −1 at step 4, where the juice should have come."},
    "nl": {"first": "Trial 1: het sap als verrassing", "learned": "Trial 60: de lamp voorspelt het sap",
           "omitted": "Trial 61: het sap blijft weg",
           "light": "lamp", "juice": "sap", "no_juice": "geen sap", "step": "stap na de lamp",
           "v": "V", "delta": "δ",
           "alt_trials": "Drie grafieken van elk één trial, van twee stappen voor de lamp tot en met stap 8: boven de V van elke stap, en onder de fout δ van elke stap. Trial 1: alle V's zijn 0, en δ is overal 0 behalve 1 bij stap 4, waar het sap komt. Trial 60: de V's zijn 1 van de lamp tot en met stap 3, en δ is 1 bij de lamp, stap 0, en 0 bij het sap. Trial 61, met dezelfde V's maar zonder sap: δ is 1 bij de lamp en −1 bij stap 4, waar het sap had moeten komen."},
}


def _trace_svg(deltas, title: str, lang: str, V=None, delay: int | None = JUICE, juice: bool = True, course=None,
               W: int = 214) -> str:
    """One trial: above, the V of every step in the color of what a model learns; below, the error δ of every step
    in the color of mistakes, from −1 to 1, with the light at step 0 and the juice at step `delay`. Two steps before
    the light are drawn too, where nothing happens. With `course`, the deltas are the learner's and differ from the
    course's, which are outlined."""
    w = _IDEA_TEXT[lang]
    cols = BEFORE + len(deltas)
    x0, cw = 18, W - 24
    col = cw / cols

    def x_of(step: int) -> float:   # the middle of the column of a step
        return x0 + (step + BEFORE + 0.5) * col

    parts = [_title(2, 14, title)]
    y = TOP + 4
    marks = y + 10   # the light and the juice
    parts.append(_light(x_of(0), marks))
    if delay is not None:
        parts.append(_drop(x_of(delay), marks, full=juice))
    y += 24
    if V is not None:
        vh = 32
        parts.append(f'<line x1="{x0}" y1="{y + vh}" x2="{x0 + cw}" y2="{y + vh}" style="stroke:var(--ea-line);stroke-width:1"/>')
        parts.append(_words(x0 - 4, y + vh - 2, w["v"], anchor="end", size=10.5))
        for t, v in enumerate(V):
            h = min(max(float(v), 0.0), 1.0) * vh
            parts.append(f'<rect x="{x_of(t) - col * 0.32:.1f}" y="{y + vh - h:.1f}" width="{col * 0.64:.1f}" height="{max(h, 0.6):.1f}" '
                         f'style="fill:var(--ea-param);fill-opacity:0.8"/>')
        y += vh + 14
    dh = 40   # from the zero line to an error of 1, up or down
    zero = y + dh
    parts.append(f'<line x1="{x0}" y1="{zero}" x2="{x0 + cw}" y2="{zero}" style="stroke:var(--ea-line);stroke-width:1"/>')
    parts.append(_words(x0 - 4, zero + 4, w["delta"], anchor="end", size=10.5))
    for t, d in enumerate(deltas):
        d = float(d)
        h = min(abs(d), 1.2) * dh
        top = zero - h if d >= 0 else zero
        parts.append(f'<rect x="{x_of(t) - col * 0.32:.1f}" y="{top:.1f}" width="{col * 0.64:.1f}" height="{max(h, 0.6):.1f}" '
                     f'style="fill:var(--ea-loss);fill-opacity:0.85"/>')
        if course is not None and abs(float(course[t]) - d) > 1e-9:
            c = float(course[t])
            ch = min(abs(c), 1.2) * dh
            ctop = zero - ch if c >= 0 else zero
            parts.append(f'<rect x="{x_of(t) - col * 0.4:.1f}" y="{ctop:.1f}" width="{col * 0.8:.1f}" height="{max(ch, 0.6):.1f}" '
                         f'style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.1;stroke-dasharray:3 2"/>')
        if abs(d) >= 0.05:
            ty = zero - h - 4 if d >= 0 else zero + h + 11
            parts.append(_words(x_of(t), ty, short_number(d, lang), size=9.5, fill="var(--ea-loss-text)"))
    bottom = zero + dh + 27   # below the label of an error of −1
    parts.append(_words(x_of(0), bottom, "0", size=10))
    if delay is not None:
        parts.append(_words(x_of(delay), bottom, str(delay), size=10))
    parts.append(_words(x0 + cw / 2, bottom + 14, w["step"], size=10.5))
    return _svg(W, bottom + 20, parts)


def trials_picture(lang: str) -> str:
    """The three trials of the idea: the first, the sixtieth, and the one without juice."""
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang]
    values = train()
    first = errors(values[0], juice_at(), GAMMA)
    learned = errors(values[TRIALS - 1], juice_at(), GAMMA)
    omitted = errors(values[TRIALS], juice_at(None), GAMMA)
    svgs = [_trace_svg(first, w["first"], lang, V=values[0]),
            _trace_svg(learned, w["learned"], lang, V=values[TRIALS - 1]),
            _trace_svg(omitted, w["omitted"], lang, V=values[TRIALS], juice=False)]
    return _figure(svgs, w["alt_trials"], 690, gap="0.6rem 0.9rem")


# --- comparing and showing ---------------------------------------------------------------------------------------------

def _array(a):
    """`a` as an array of floats, or None when it is not numbers."""
    try:
        out = np.asarray(a, dtype=float)
    except (TypeError, ValueError):
        return None
    return out if out.size and np.isfinite(out).all() else None


def same_array(a, b) -> bool:
    """errors and learn: an array of the same shape as the course's, every number the same."""
    got, want = _array(a), np.asarray(b, dtype=float)
    return got is not None and got.shape == want.shape and bool(np.allclose(got, want, rtol=0, atol=1e-9))


def _short(v) -> str:
    got = _array(v)
    if got is not None and got.ndim == 1:
        rounded = [round(float(x), 3) + 0.0 for x in got]
        shown = [str(int(x)) if x.is_integer() else repr(x) for x in rounded]
        return "[" + ", ".join(shown) + "]"
    text_ = repr(v)
    return text_ if len(text_) <= 80 else text_[:77] + "..."


def show_array(v) -> dict:
    shown = f"`{_short(v)}`"
    return {"en": shown, "nl": shown}


def _arr_code(a) -> str:
    a = np.asarray(a)
    if a.dtype.kind in "iu":
        return "np.array([" + ", ".join(str(int(x)) for x in a) + "])"
    return "np.array([" + ", ".join(repr(round(float(x), 6)) for x in a) + "])"


class ErrorsBox(FunctionBox):
    """errors in a code box: a call shows its arrays as `np.array([...])`, as the learner would write them."""

    def _call_text(self, case) -> str:
        V, rewards, gamma = case
        return f"errors({_arr_code(V)}, {_arr_code(rewards)}, {gamma})"


class LearnBox(FunctionBox):
    """learn in a code box: a call shows its arrays as `np.array([...])`."""

    def _call_text(self, case) -> str:
        V, rewards, eta, gamma = case
        return f"learn({_arr_code(V)}, {_arr_code(rewards)}, {eta}, {gamma})"


# --- the pictures of the learner's work -----------------------------------------------------------------------------------

_STOPPED = object()   # what _call gives for a function that stopped with an error
_SLOW = object()      # what _call gives for a function that ran out of time
PICTURE_SECONDS = 4   # what the learner's function gets for all the calls of one picture together


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _call(make_, seconds: float = PICTURE_SECONDS):
    """make_() within a time limit, with what it prints thrown away; _SLOW when the time runs out, _STOPPED when it
    stops with an error."""
    import contextlib
    import io
    from everything_ai.check import with_time_limit
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(make_, max(seconds, 0.05))
    except Timeout:
        return _SLOW
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def picture_cases() -> list:
    """The three trials of the idea, as (V, rewards): the first, the sixtieth, and the one without juice."""
    values = train()
    return [(values[0], juice_at()), (values[TRIALS - 1], juice_at()), (values[TRIALS], juice_at(None))]


_YOURS_TEXT = {
    "en": {"errors": {"title": "Your errors",
                      "intro": "The three trials of the idea, with the δ of your `errors`: above, the V's the course passes in; below, your δ at every step. Outlined, the course's δ where yours differs.",
                      "say": "Your `errors` gives δ {a} at the juice in trial 1, {b} at the light in trial 60, and {c} at step 4 in trial 61, without juice.",
                      "none": "Your `errors` stops with an error, takes too long, or gives something other than an array with a number for every step, so nothing can be drawn yet."},
           "learn": {"title": "Your learn",
                     "intro": "The error δ at every step of a trial, after the V's of your `learn` learned for that many trials, with η 0.2 and γ 1; the last row is trial 61, without juice. Outlined, the course's δ where yours differs.",
                     "trial": "trial {k}", "omitted": "no juice",
                     "say": "After 60 trials, your V's are {v}.",
                     "none": "Your `learn` stops with an error, takes too long, or gives something other than an array with a number for every step, so nothing can be drawn yet."}},
    "nl": {"errors": {"title": "Je errors",
                      "intro": "De drie trials van het idee, met de δ van je `errors`: boven de V's die de course meegeeft; onder je δ bij elke stap. Omlijnd de δ van de course waar je eigen δ anders is.",
                      "say": "Je `errors` geeft δ {a} bij het sap in trial 1, {b} bij de lamp in trial 60, en {c} bij stap 4 in trial 61, zonder sap.",
                      "none": "Je `errors` stopt met een fout, duurt te lang, of geeft iets anders dan een array met een getal voor elke stap, dus is er nog geen afbeelding."},
           "learn": {"title": "Je learn",
                     "intro": "De fout δ bij elke stap van een trial, nadat de V's van je `learn` zoveel trials geleerd hebben, met η 0,2 en γ 1; de laatste rij is trial 61, zonder sap. Omlijnd de δ van de course waar je eigen δ anders is.",
                     "trial": "trial {k}", "omitted": "geen sap",
                     "say": "Na 60 trials zijn je V's {v}.",
                     "none": "Je `learn` stopt met een fout, duurt te lang, of geeft iets anders dan een array met een getal voor elke stap, dus is er nog geen afbeelding."}},
}


def errors_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["errors"]
    idea = _IDEA_TEXT[lang]
    fn = _fn(seen)
    cases = picture_cases()
    got = _call(lambda: [fn(*copy.deepcopy(case), GAMMA) for case in cases]) if fn is not None else None
    mine = [_array(g) for g in got] if isinstance(got, list) else None
    if mine is None or any(m is None or m.shape != (STEPS,) for m in mine):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    svgs = []
    for (V, rewards), m, key in zip(cases, mine, ("first", "learned", "omitted")):
        want = errors(V, rewards, GAMMA)
        svgs.append(_trace_svg(m, idea[key], lang, V=V, juice=key != "omitted",
                               course=None if same_array(m, want) else want))
    say = w["say"].format(a=short_number(mine[0][JUICE], lang), b=short_number(mine[1][0], lang),
                          c=short_number(mine[2][JUICE], lang))
    return yours.figure(w["title"], _note(w["intro"]) + _figure(svgs, w["title"], 690, gap="0.6rem 0.9rem") + _note(say),
                        alt=w["title"])


def _ridge_svg(rows: list, lang: str, W: int = 330) -> str:
    """The error at every step of some trials, a row per trial: bars up for an error above 0, down below 0. A row is
    (label, deltas, course deltas or None, juice)."""
    w = _IDEA_TEXT[lang]
    left, cw = 74, W - 82
    cols = BEFORE + STEPS
    col = cw / cols
    rh, dh = 36, 14

    def x_of(step: int) -> float:
        return left + (step + BEFORE + 0.5) * col

    parts = []
    y = 6
    parts.append(_light(x_of(0), y + 10))
    parts.append(_drop(x_of(JUICE), y + 10))
    y += 24
    top = y
    for label, deltas, course, _ in rows:
        zero = y + rh / 2
        parts.append(_words(left - 8, zero + 4, label, anchor="end", size=10.5, fill="var(--ea-ink)"))
        parts.append(f'<line x1="{left}" y1="{zero:.1f}" x2="{left + cw}" y2="{zero:.1f}" style="stroke:var(--ea-line);stroke-width:0.8"/>')
        for t, d in enumerate(deltas):
            d = float(d)
            h = min(abs(d), 1.2) * dh
            bar_top = zero - h if d >= 0 else zero
            parts.append(f'<rect x="{x_of(t) - col * 0.32:.1f}" y="{bar_top:.1f}" width="{col * 0.64:.1f}" height="{max(h, 0.5):.1f}" '
                         f'style="fill:var(--ea-loss);fill-opacity:0.85"/>')
            if course is not None and abs(float(course[t]) - d) > 1e-9:
                c = float(course[t])
                ch = min(abs(c), 1.2) * dh
                ctop = zero - ch if c >= 0 else zero
                parts.append(f'<rect x="{x_of(t) - col * 0.4:.1f}" y="{ctop:.1f}" width="{col * 0.8:.1f}" height="{max(ch, 0.5):.1f}" '
                             f'style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.1;stroke-dasharray:3 2"/>')
        y += rh
    for step in (0, JUICE):
        parts.append(f'<line x1="{x_of(step):.1f}" y1="{top - 2}" x2="{x_of(step):.1f}" y2="{y:.1f}" '
                     f'style="stroke:var(--ea-line);stroke-width:0.8;stroke-dasharray:2 3"/>')
    parts.append(_words(x_of(0), y + 14, "0", size=10))
    parts.append(_words(x_of(JUICE), y + 14, str(JUICE), size=10))
    parts.append(_words(left + cw / 2, y + 28, w["step"], size=10.5))
    return _svg(W, y + 34, parts)


def learn_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["learn"]
    fn = _fn(seen)
    got = _call(lambda: train(fn)) if fn is not None else None
    mine = _array(got) if got is not _STOPPED and got is not _SLOW and got is not None else None
    if mine is None or mine.shape != (TRIALS + 2, STEPS):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    course = train()
    rows = []
    for k in RIDGE_TRIALS + (TRIALS + 1,):
        rewards = juice_at() if k <= TRIALS else juice_at(None)
        d = errors(mine[k - 1], rewards, GAMMA)
        want = errors(course[k - 1], rewards, GAMMA)
        label = w["trial"].format(k=k) if k <= TRIALS else w["omitted"]
        rows.append((label, d, None if same_array(d, want) else want, k <= TRIALS))
    say = w["say"].format(v=f"`{_short(mine[TRIALS])}`")
    return yours.figure(w["title"], _note(w["intro"]) + _figure([_ridge_svg(rows, lang)], w["title"], 330) + _note(say),
                        alt=w["title"])


# --- the demo ----------------------------------------------------------------------------------------------------------------

DEMO_TEXT = {
    "en": {"title": "δ at the juice, and dashed at the light", "x": "trial", "y": "δ",
           "alt": "The error δ at the moment of the juice in every trial so far, and dashed the error at the light.",
           "delay": "Juice at step", "gamma": "Discount factor γ",
           "at_juice": "at the juice", "at_light": "at the light",
           "now": "Trial {t}", "now_omitted": "Trial {t}: no juice", "start": "Before the first trial",
           "say": "In trial {t}, δ is {light} at the light and {juice} at step {d}, where the juice comes.",
           "say_omitted": "In trial {t}, the juice stays away: δ is {light} at the light and {juice} at step {d}, where the juice should have come.",
           "who_mine": "your `learn`", "who_course": "the course's `learn`, since yours does not pass its check yet",
           "learning": "Trial {t} of {n}",
           "verdict": "With {who}: after {k} trials with the juice at step {d}, δ is **{light}** at the light and **{juice}** at the juice. In the last trial, without juice, δ drops to **{omitted}** at step {d}."},
    "nl": {"title": "δ bij het sap, en gestreept bij de lamp", "x": "trial", "y": "δ",
           "alt": "De fout δ op het moment van het sap in elke trial tot nu toe, en gestreept de fout bij de lamp.",
           "delay": "Sap bij stap", "gamma": "Discount factor γ",
           "at_juice": "bij het sap", "at_light": "bij de lamp",
           "now": "Trial {t}", "now_omitted": "Trial {t}: geen sap", "start": "Voor de eerste trial",
           "say": "In trial {t} is δ {light} bij de lamp en {juice} bij stap {d}, waar het sap komt.",
           "say_omitted": "In trial {t} blijft het sap weg: δ is {light} bij de lamp en {juice} bij stap {d}, waar het sap had moeten komen.",
           "who_mine": "je `learn`", "who_course": "de `learn` van de course, omdat je eigen `learn` de check nog niet haalt",
           "learning": "Trial {t} van {n}",
           "verdict": "Met {who}: na {k} trials met het sap bij stap {d} is δ **{light}** bij de lamp en **{juice}** bij het sap. In de laatste trial, zonder sap, zakt δ naar **{omitted}** bij stap {d}."},
}


def delay_options(lang: str) -> dict:
    return {number(d, 0, lang): d for d in DELAY_OPTIONS}


def gamma_options(lang: str) -> dict:
    return {number(g, 0 if float(g).is_integer() else 1, lang): g for g in GAMMA_OPTIONS}


def chart_labels(lang: str) -> dict:
    w = DEMO_TEXT[lang]
    return {"title": w["title"], "x": w["x"], "y": w["y"], "alt": w["alt"]}


def chart_key(lang: str) -> list:
    w = DEMO_TEXT[lang]
    return [("loss", False, w["at_juice"]), ("loss", True, w["at_light"])]


def demo_xticks() -> list:
    return [1, 10, 20, 30, 40, 50, 61]


class Demo:
    """The demo's experiment: TRIALS trials with the juice at step `delay`, then one without, and the V's of a learn
    over them, shown one trial at a time."""

    def __init__(self, delay: int, gamma: float, values):
        self.delay, self.gamma = delay, gamma
        self.values = np.asarray(values, dtype=float)
        self.deltas = trial_errors(self.values, delay, gamma)
        self.n = len(self.deltas)
        self.t = 0   # the trials shown so far

    def points(self, step: int, t: int | None = None) -> list:
        """(trial, δ at `step`) for the trials shown so far."""
        t = self.t if t is None else t
        return [(k, float(self.deltas[k - 1, step])) for k in range(1, t + 1)]


def demo_run(learn_, delay: int, gamma: float, show=None, pause: float = 0.0, seconds: float = 4.0) -> Demo | str:
    """The demo: the V's that `learn_` gives over the experiment, then `show(t, demo)` after every trial; "error"
    when the learner's learn stops with an error, takes too long, or gives something other than a number for every
    step. `pause` waits that long after each trial, so that you can follow the trials."""
    import time
    got = _call(lambda: train(learn_, delay, ETA, gamma), seconds)
    values = _array(got) if got is not _STOPPED and got is not _SLOW and got is not None else None
    if values is None or values.shape != (TRIALS + 2, STEPS):
        return "error"
    demo = Demo(delay, gamma, values)
    for t in range(1, demo.n + 1):
        demo.t = t
        if show is not None:
            show(t, demo)
        if pause:
            time.sleep(pause)
    return demo


def _trace_at(demo: Demo, t: int, lang: str) -> str:
    w = DEMO_TEXT[lang]
    if not t:
        title = w["start"]
        return _figure([_trace_svg(np.zeros(STEPS), title, lang, V=demo.values[0], delay=demo.delay, W=330)], title, 330)
    omitted = t > TRIALS
    title = (w["now_omitted"] if omitted else w["now"]).format(t=t)
    svg = _trace_svg(demo.deltas[t - 1], title, lang, V=demo.values[t - 1], delay=demo.delay, juice=not omitted, W=330)
    return _figure([svg], title, 330)


def now_picture(demo: Demo, lang: str) -> str:
    """Beside the demo's chart: the V's and the error at every step of the last trial shown."""
    lang = lang if lang in LANGS else "en"
    return _trace_at(demo, demo.t, lang)


def _demo_panel(demo: Demo, t: int, lang: str, W: int = 330) -> str:
    """The demo's chart as a small picture, with a line at trial t."""
    w = DEMO_TEXT[lang]
    last = demo.n
    x0, y0, cw, ch = 34, TOP + 8, W - 44, 150
    low, high = -1.0, 1.0
    parts = [_title(2, 14, w["title"])]
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + ch}" style="stroke:var(--ea-line);stroke-width:1"/>')
    for tick in (-1, 0, 1):
        y = y0 + ch - (tick - low) / (high - low) * ch
        style = "stroke:var(--ea-line);stroke-width:1" if tick == 0 else "stroke:var(--ea-line);stroke-width:0.6;opacity:0.6"
        parts.append(f'<line x1="{x0 - 3}" y1="{y:.1f}" x2="{x0 + cw}" y2="{y:.1f}" style="{style}"/>')
        parts.append(_words(x0 - 6, y + 4, number(tick, 0, lang), anchor="end", size=10.5))
    for tick in demo_xticks():
        parts.append(_words(x0 + tick / last * cw, y0 + ch + 14, number(tick, 0, lang), size=10.5))
    parts.append(_words(x0 + cw / 2, y0 + ch + 30, w["x"], size=11))
    parts.append(_plot(demo.points(0, last), x0, y0, cw, ch, last, low, high, "var(--ea-loss)", 1.6, dashed=True))
    parts.append(_plot(demo.points(demo.delay, last), x0, y0, cw, ch, last, low, high, "var(--ea-loss)", 2.2))
    if t:
        x = x0 + t / last * cw
        parts.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0 + ch}" style="stroke:var(--ea-ink-2);stroke-width:1.2"/>')
    return _svg(W, y0 + ch + 38, parts)


def rewind_frame(demo: Demo, t: int, lang: str) -> tuple:
    """A stop of the rewind after the demo: the chart with a line at trial t, the V's and the errors then, and a
    sentence."""
    lang = lang if lang in LANGS else "en"
    w = DEMO_TEXT[lang]
    t = int(t)
    picture = _trace_at(demo, t, lang)
    svg = _figure([_demo_panel(demo, t, lang)], w["alt"], 330)
    say = ""
    if t:
        d = demo.deltas[t - 1]
        key = "say_omitted" if t > TRIALS else "say"
        say = w[key].format(t=t, light=short_number(d[0], lang), juice=short_number(d[demo.delay], lang), d=demo.delay)
    return live.rewound(svg, [picture], width=330), say


def verdict(demo: Demo, mine: bool, lang: str) -> str:
    w = DEMO_TEXT[lang]
    who = w["who_mine"] if mine else w["who_course"]
    learned, omitted = demo.deltas[TRIALS - 1], demo.deltas[TRIALS]
    return w["verdict"].format(who=who, k=TRIALS, d=demo.delay, light=number(learned[0], 2, lang),
                               juice=number(learned[demo.delay], 2, lang), omitted=number(omitted[demo.delay], 2, lang))


# --- the exercises -----------------------------------------------------------------------------------------------------

REQUIRED = ["errors", "learn", "more", "unblock"]


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
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


# exercise 1: errors
def _e(line: str = "rewards[t] + gamma * V[t] - before") -> str:
    return ("def errors(V, rewards, gamma):\n"
            "    deltas = np.zeros(len(V))\n"
            "    before = 0.0\n"
            "    for t in range(len(V)):\n"
            f"        deltas[t] = {line}\n"
            "        before = V[t]\n"
            "    return deltas\n")


_E_START = _e("...")
_E = _e()
_E_OWN = _e("rewards[t] - V[t]")
_E_NO_JUICE = _e("gamma * V[t] - before")
_E_SIGN = _e("before - rewards[t] - gamma * V[t]")
_E_AHEAD = _e("rewards[t] + gamma * V[t + 1] - V[t]")
_E_FIXED = _e("rewards[t] + V[t] - before")
E_NAMES = {"np": np}


def _past_the_end(e: BaseException) -> bool:
    return isinstance(e, IndexError) and "out of bounds" in str(e)


# exercise 2: learn
def _l(line: str = "V[t] + eta * deltas[t + 1]", call: str = "errors(V, rewards, gamma)") -> str:
    return ("def learn(V, rewards, eta, gamma):\n"
            f"    deltas = {call}\n"
            "    new = np.zeros(len(V))\n"
            "    for t in range(len(V) - 1):\n"
            f"        new[t] = {line}\n"
            "    return new\n")


_L_START = _l("V[t] + eta * ...")
_L = _l()
_L_SAME = _l("V[t] + eta * deltas[t]")
_L_BACK = _l("V[t] + eta * deltas[t - 1]")
_L_JUICE = _l("V[t] + eta * rewards[t + 1]")
_L_FIXED = _l(call="errors(V, rewards, 1)")


def learn_names(errors_=None) -> dict:
    """The names a learn in a code box may use: NumPy, and an errors, the course's unless another is given."""
    return {"np": np, "errors": errors_ if errors_ is not None else errors}


L_NAMES = learn_names()

_V_HALF = np.array([0.5, 0.6, 0.7, 0.8, 0.0, 0.0])
_V_DONE = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
_V_ZERO = np.zeros(6)
_J4 = np.array([0, 0, 0, 0, 1, 0])
_J_NONE = np.array([0, 0, 0, 0, 0, 0])
_J2 = np.array([0, 0, 2, 0, 0, 0])

EXERCISES = {
    "errors": ErrorsBox(
        name="errors",
        function="errors",
        start=_E_START,
        solution=_E,
        names=E_NAMES,
        compare=same_array,
        show=show_array,
        visible=[(_V_HALF, _J4, 1)],
        hidden=[(_V_DONE, _J4, 1), (_V_DONE, _J_NONE, 1), (_V_ZERO, _J4, 1), (_V_HALF, _J4, 0.9),
                (np.array([0.0, 1.5, 1.8, 0.0, 0.0, 0.0]), _J2, 0.8)],
        first=[_DOTS],
        picture=errors_picture,
        mistakes=[
            Mistake("ahead", error=_past_the_end,
                    en="With `{call}`, your function stops: at the last step, it reads past the end of an array, as `V[t + 1]` does. In this lesson, the error at step t looks back: the juice and V of step t, against the V of the step before. Which name holds the V of the step before?",
                    nl="Met `{call}` stopt je functie: bij de laatste stap leest je functie voorbij het einde van een array, zoals `V[t + 1]` doet. In deze les kijkt de fout bij stap t terug: het sap en de V van stap t, tegen de V van de vorige stap. Welke naam houdt de V van de vorige stap vast?"),
            Mistake("own", wrong=_E_OWN,
                    en="`{call}` gives {got}, and it should give {want}: every step compares the juice with its own V, as in the Rescorla-Wagner model with a V per step. The TD error compares two predictions: what step t says, its juice plus γ times its V, against what the step before predicted. Where does the V of the step before come in?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke stap vergelijkt het sap met de eigen V, zoals in het Rescorla-Wagner model met een V per stap. De TD error vergelijkt twee voorspellingen: wat stap t zegt, het sap plus γ keer de V, tegen wat de vorige stap voorspelde. Waar komt de V van de vorige stap in?"),
            Mistake("no-juice", wrong=_E_NO_JUICE,
                    en="`{call}` gives {got}, and it should give {want}: the juice that comes at step t is missing, so the juice never surprises the model. Which part of what came at step t did you leave out?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het sap dat bij stap t komt ontbreekt, dus verrast het sap het model nooit. Welk deel van wat er bij stap t kwam heb je weggelaten?"),
            Mistake("sign", wrong=_E_SIGN,
                    en="`{call}` gives {got}, and it should give {want}: with the old prediction minus what came, juice that was not predicted gives an error below 0. Which of the two comes first in the error?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: met de oude voorspelling min wat er kwam geeft sap dat niet voorspeld was een fout onder 0. Welke van de twee komt eerst in de fout?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the error at step t: the juice of the step, plus γ times its V, minus the V of the step before.",
                      "nl": "Vervang de `...` door de fout bij stap t: het sap van de stap, plus γ keer de V, min de V van de vorige stap."},
            "arguments": {"en": "The check calls `{call}`, with the V's, the juice at every step and γ, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(V, rewards, gamma)`?",
                          "nl": "De check roept `{call}` aan, met de V's, het sap bij elke stap en γ, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(V, rewards, gamma)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is `deltas[t]` the juice of step t plus γ times its V, minus `before`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Is `deltas[t]` het sap van stap t plus γ keer de V, min `before`?"},
            "almost": {"en": "Right with γ = 1. But `{call}` gives {got}, and it should give {want}. Does your function use the `gamma` it is given?",
                       "nl": "Goed met γ = 1. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie de `gamma` die de functie krijgt?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return deltas`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return deltas`?"},
            "correct": {"en": "Right in all {n} cases: with V's of 0, δ is 1 at the juice; with the learned V's, it is 1 at the light and 0 at the juice, and −1 there when the juice stays away. Which of the three trials in the picture shows the dip that Schultz saw?",
                        "nl": "Goed in alle {n} gevallen: met V's van 0 is δ 1 bij het sap; met de geleerde V's is δ 1 bij de lamp en 0 bij het sap, en −1 daar als het sap wegblijft. In welke van de drie trials in de afbeelding zie je de dip die Schultz zag?"},
        },
    ),
    "learn": LearnBox(
        name="learn",
        function="learn",
        start=_L_START,
        solution=_L,
        names=L_NAMES,
        compare=same_array,
        show=show_array,
        visible=[(_V_ZERO, _J4, ETA, 1)],
        hidden=[(_V_HALF, _J4, ETA, 1), (_V_DONE, _J_NONE, ETA, 1), (_V_HALF, _J4, 0.5, 0.9),
                (np.array([0.2, 0.0, 0.4, 0.0, 0.0, 0.0]), _J2, 0.1, 0.8)],
        first=[_DOTS],
        picture=learn_picture,
        mistakes=[
            Mistake("same-step", wrong=_L_SAME,
                    en="`{call}` gives {got}, and it should give {want}: every V learns from the error at its own step. But the error at step t says how far off the step before was, so after the first trial the juice step itself has learned something, and the step before it nothing. Which step's error belongs to `V[t]`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke V leert van de fout bij de eigen stap. Maar de fout bij stap t zegt hoe ver de vorige stap ernaast zat, dus heeft na de eerste trial de stap van het sap zelf iets geleerd, en de vorige stap niets. De fout van welke stap hoort bij `V[t]`?"),
            Mistake("back", wrong=_L_BACK,
                    en="`{call}` gives {got}, and it should give {want}: every V learns from the error of the step before it, so the error at the juice goes to the step after the juice, and the steps before the juice, which should learn that juice is coming, learn nothing from it. The error at step t compares step t with the step before. Does `V[t]` learn from the step before or the step after?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke V leert van de fout van de vorige stap, dus gaat de fout bij het sap naar de stap na het sap, en leren de stappen voor het sap er niets van, terwijl juist deze stappen moeten leren dat er sap komt. De fout bij stap t vergelijkt stap t met de vorige stap. Leert `V[t]` van de vorige stap of van de stap erna?"),
            Mistake("juice-only", wrong=_L_JUICE,
                    en="`{call}` gives {got}, and it should give {want}: every V takes a step toward the juice of the step after it, without looking at what was predicted, so a V keeps growing and the steps further from the juice never learn. Which array says how far off each prediction was?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke V zet een stap naar het sap van de stap erna, zonder te kijken naar wat er voorspeld was, dus groeit een V steeds verder en leren de stappen verder van het sap nooit. Welke array zegt hoe ver elke voorspelling ernaast zat?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the error that `V[t]` learns from: the error of the step after it.",
                      "nl": "Vervang de `...` door de fout waar `V[t]` van leert: de fout van de stap erna."},
            "arguments": {"en": "The check calls `{call}`, with the V's, the juice at every step, a learning rate and γ, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(V, rewards, eta, gamma)`?",
                          "nl": "De check roept `{call}` aan, met de V's, het sap bij elke stap, een learning rate en γ, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(V, rewards, eta, gamma)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does every `new[t]` take a step of η times the error of the step after it?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Zet elke `new[t]` een stap van η keer de fout van de stap erna?"},
            "almost": {"en": "Right with γ = 1. But `{call}` gives {got}, and it should give {want}. Does your function give `errors` the `gamma` it is given?",
                       "nl": "Goed met γ = 1. Maar `{call}` geeft {got}, en het moet {want} zijn. Geeft je functie `errors` de `gamma` die de functie krijgt?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return new`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return new`?"},
            "correct": {"en": "Right in all {n} cases: after one trial from V's of 0, only step 3, just before the juice, has learned: 0.2. In the picture, between which two rows does the error at the light become larger than the error at the juice?",
                        "nl": "Goed in alle {n} gevallen: na één trial vanaf V's van 0 heeft alleen stap 3, vlak voor het sap, iets geleerd: 0,2. Tussen welke twee rijen van de afbeelding wordt de fout bij de lamp groter dan de fout bij het sap?"},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("errors", _E_START, "empty", None),
    ("errors", _E, "correct", None),
    ("errors", _e("rewards[t] + ..."), "wrong", "dots-left"),
    ("errors", _E_AHEAD, "error", "ahead"),
    ("errors", _E_OWN, "wrong", "own"),
    ("errors", _E_NO_JUICE, "wrong", "no-juice"),
    ("errors", _E_SIGN, "wrong", "sign"),
    ("errors", _E.replace("    return deltas\n", ""), "wrong", "no-return"),
    # other right answers: the same number written another way
    ("errors", _e("rewards[t] - before + gamma * V[t]"), "correct", None),
    ("errors", _e("(rewards[t] + gamma * V[t]) - before"), "correct", None),
    # γ left out: right with γ = 1 only
    ("errors", _E_FIXED, "almost", None),
    ("learn", _L_START, "empty", None),
    ("learn", _L, "correct", None),
    ("learn", _l("..."), "wrong", "dots-left"),
    ("learn", _L_SAME, "wrong", "same-step"),
    ("learn", _L_BACK, "wrong", "back"),
    ("learn", _L_JUICE, "wrong", "juice-only"),
    ("learn", _L.replace("    return new\n", ""), "wrong", "no-return"),
    # another right answer: the error of the step after, from the formula itself
    ("learn", _l("V[t] + eta * (rewards[t + 1] + gamma * V[t + 1] - V[t])"), "correct", None),
    # γ written in as 1: right with γ = 1 only
    ("learn", _L_FIXED, "almost", None),
]
