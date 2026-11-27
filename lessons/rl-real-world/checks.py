# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rl-real-world: RL outside games, with a data center in a simulator. The servers heat the room,
three actions cool it (off, low, high), electricity costs more in the day, and an hour above the limit costs a penalty.
Q-learning in the simulator learns to cool steadily with the efficient setting, from the night on, and beats a
thermostat; it goes too hot in thousands of hours while it learns, which costs nothing in a simulator; and a night
warmer than any the simulator showed leads it into states it never saw, where a safety rule has to take over.

The data center, its numbers and the pictures are the course's own; the five applications in the text are from the
papers named in lesson.yaml."""
import copy
import functools
import html
import math

import numpy as np

from everything_ai import live, yours
from everything_ai.check import FunctionBox, Mistake, Timeout

LESSON = "rl-real-world"
LANGS = ("en", "nl")
HOURS = 24
HEAT = [1] * 8 + [3] * 12 + [2] * 4    # degrees the servers add in each hour: quiet at night, busy in the day
PRICE = [1] * 8 + [4] * 12 + [2] * 4   # what a unit of electricity costs in each hour: cheap at night
COOL = [0, 2, 4]                       # degrees each action takes away: off, low, high
ENERGY = [0, 1, 3]                     # units of electricity each action uses: high cools twice as much for three times the energy
LIMIT = 27                             # above this the servers are too hot
PENALTY = 10                           # the cost of an hour above the limit
COLDEST, HOTTEST = 15, 35              # the room never gets colder or hotter than this
START = 20                             # the temperature when a day starts
TEMPS = HOTTEST - COLDEST + 1          # the rows of the Q-table: one per temperature
ETA = 0.1
EPSILON = 0.1
DAYS = 10_000                          # the days of training in the simulator
SEED = 0
SNAP = 500                             # the rewind keeps the Q-table every SNAP days
REALITIES = {
    "sim": HEAT,
    "afternoon": [1] * 8 + [3] * 4 + [4] * 4 + [3] * 4 + [2] * 4,   # four hours of the afternoon one degree hotter
    "night": [2] * 8 + [3] * 12 + [2] * 4,                          # a warm night: the servers stay busy
}
REALITY_KEYS = ("sim", "afternoon", "night")


# --- the functions of the exercises, as the course writes them ----------------------------------------------------------

def step(temp, hour, action):
    new = temp + HEAT[hour] - COOL[action]
    new = min(max(new, COLDEST), HOTTEST)
    reward = -PRICE[hour] * ENERGY[action]
    if new > LIMIT:
        reward = reward - PENALTY
    return new, reward


def thermostat(temp, hour):
    if temp >= 25:
        return 2
    if temp >= 23:
        return 1
    return 0


def act(q, count, temp, hour):
    if count == 0:
        return thermostat(temp, hour)
    return int(np.argmax(q))


# --- the course's world and agent ----------------------------------------------------------------------------------------

def real_step(temp: int, hour: int, action: int, heat=HEAT) -> tuple:
    """The data center itself: the course's step, with the heat of the servers that reality gives on this day."""
    new = min(max(temp + heat[hour] - COOL[action], COLDEST), HOTTEST)
    reward = -PRICE[hour] * ENERGY[action]
    if new > LIMIT:
        reward -= PENALTY
    return new, reward


def day(policy, heat=HEAT) -> dict:
    """One day in the data center from START, with `policy(temp, hour)`: the temperature after every hour, the
    actions, the cost (the bill plus the penalties), the bill, and the hours above the limit."""
    temp, temps, actions, cost, bill, hot = START, [START], [], 0, 0, 0
    for hour in range(HOURS):
        a = int(policy(temp, hour))
        temp, reward = real_step(temp, hour, a, heat)
        temps.append(temp)
        actions.append(a)
        cost -= reward
        bill += PRICE[hour] * ENERGY[a]
        hot += temp > LIMIT
    return {"temps": temps, "actions": actions, "cost": cost, "bill": bill, "hot": hot}


def _checked(step_):
    """The learner's step, made safe for training: a step that gives something other than a temperature in the
    room's range and a number stops the run with an error."""
    def checked(temp, hour, action):
        new, reward = step_(temp, hour, action)
        new_i = int(new)
        if new_i != new or not COLDEST <= new_i <= HOTTEST or not math.isfinite(float(reward)):
            raise ValueError("not a temperature and a reward")
        return new_i, float(reward)
    return checked


class Run:
    """Q-learning in a simulator: the Q-table and how often each state came by, the cost and the hot hours of every
    day of training, and the Q-table every SNAP days for the rewind."""

    def __init__(self):
        self.Q = np.zeros((TEMPS, HOURS + 1, 3))
        self.seen = np.zeros((TEMPS, HOURS), dtype=int)
        self.costs: list = []
        self.hots: list = []
        self.snaps = {0: self.Q.copy()}

    @property
    def days(self) -> int:
        return len(self.costs)

    def greedy(self, Q=None):
        Q = self.Q if Q is None else Q
        return lambda temp, hour: int(np.argmax(Q[temp - COLDEST, hour]))

    def safe(self, act_=act):
        def policy(temp, hour):
            row = temp - COLDEST
            return act_([float(x) for x in self.Q[row, hour]], int(self.seen[row, hour]), temp, hour)
        return policy


def train(step_=step, days: int = DAYS, seed: int = SEED, show=None, every: int = 100) -> Run:
    """Q-learning with ε-greedy in the simulator `step_`, one day from START per episode, γ 1; `show(run)` after
    every `every` days."""
    rng = np.random.default_rng(seed)
    run_ = Run()
    Q, seen = run_.Q, run_.seen
    for d in range(1, days + 1):
        temp, cost, hot = START, 0.0, 0
        for hour in range(HOURS):
            s = (temp - COLDEST, hour)
            seen[s] += 1
            if rng.random() < EPSILON:
                a = int(rng.integers(3))
            else:
                a = int(np.argmax(Q[s]))
            new, reward = step_(temp, hour, a)
            Q[s][a] += ETA * (reward + Q[new - COLDEST, hour + 1].max() - Q[s][a])
            temp = new
            cost -= reward
            hot += temp > LIMIT
        run_.costs.append(cost)
        run_.hots.append(hot)
        if d % SNAP == 0:
            run_.snaps[d] = Q.copy()
        if show is not None and (d % every == 0 or d == days):
            show(run_)
    return run_


@functools.cache
def course_run() -> Run:
    """The course's run: 10,000 days in the simulator with seed 0."""
    return train()


THERMOSTAT = day(thermostat)   # cost 103, nothing too hot


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


def short_number(x: float, lang: str, digits: int = 1) -> str:
    """A number rounded to `digits` decimals, without the zeros at its end: 63, 4.5."""
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


_DAY_TEXT = {
    "en": {"hour": "hour", "limit": "27°", "action": "cooling", "night": "cheap", "day": "expensive",
           "thermostat": "Thermostat: waits until the room is warm", "learned": "Learned: after 10,000 days in the simulator",
           "alt_days": "Two charts of one day in the simulated data center, hour 0 to 24: the temperature as a line against the limit of 27 degrees, and under it the cooling of every hour, off, low or high. The thermostat lets the room warm up to 23 degrees during the cheap night, then cools high in the expensive day whenever the room reaches 25. The learned agent cools low almost every hour from the night on, down to 15 degrees by morning, and lets the room warm up slowly to 27 by the evening, without ever cooling high."},
    "nl": {"hour": "uur", "limit": "27°", "action": "koeling", "night": "goedkoop", "day": "duur",
           "thermostat": "Thermostaat: wacht tot de ruimte warm is", "learned": "Geleerd: na 10.000 dagen in de simulator",
           "alt_days": "Twee grafieken van één dag in het gesimuleerde datacenter, uur 0 tot en met 24: de temperatuur als lijn tegen de grens van 27 graden, en daaronder de koeling van elk uur, uit, laag of hoog. De thermostaat laat de ruimte in de goedkope nacht opwarmen tot 23 graden, en koelt in de dure dag hoog zodra de ruimte 25 graden haalt. De geleerde agent koelt vanaf de nacht bijna elk uur laag, tot 15 graden in de ochtend, en laat de ruimte langzaam opwarmen tot 27 graden in de avond, zonder ooit hoog te koelen."},
}


def _day_svg(dd: dict, title: str, lang: str, learned: bool = False, course=None, W: int = 330) -> str:
    """One day: the temperature after every hour as a line, against the limit, with the hours above it marked in the
    color of mistakes; under it the cooling of every hour as a bar of 0, 1 or 2 steps, in the color of what a model
    learns for a learned policy, gray for a fixed rule; the expensive hours shaded. With `course`, the course's day
    is drawn dashed where it differs."""
    w = _DAY_TEXT[lang]
    x0, x1 = 46, W - 8
    cw = (x1 - x0) / HOURS
    y_top = TOP + 14
    th = 104
    lo, hi = COLDEST - 1, HOTTEST + 1

    def x_of(hour: float) -> float:
        return x0 + hour * cw

    def y_of(t: float) -> float:
        return y_top + th - (min(max(float(t), lo), hi) - lo) / (hi - lo) * th

    parts = [_title(2, 14, title)]
    # the expensive hours, 8 to 20
    parts.append(f'<rect x="{x_of(8):.1f}" y="{y_top}" width="{x_of(20) - x_of(8):.1f}" height="{th}" '
                 f'style="fill:var(--ea-line);fill-opacity:0.35"/>')
    parts.append(_words(x_of(4), y_top - 3, w["night"], size=9.5))
    parts.append(_words(x_of(14), y_top - 3, w["day"], size=9.5))
    parts.append(_words(x0 - 5, y_top - 3, "°C", anchor="end", size=9.5))
    for t in (15, 20, 25, 30, 35):
        parts.append(_words(x0 - 5, y_of(t) + 3.5, str(t), anchor="end", size=9.5))
    ly = y_of(LIMIT)
    parts.append(f'<line x1="{x0}" y1="{ly:.1f}" x2="{x1}" y2="{ly:.1f}" style="stroke:var(--ea-loss);stroke-width:1.1;'
                 f'stroke-dasharray:4 3"/>')
    parts.append(f'<line x1="{x0}" y1="{y_top + th}" x2="{x1}" y2="{y_top + th}" style="stroke:var(--ea-line);stroke-width:1"/>')
    temps = dd["temps"]
    if course is not None and list(course["temps"]) != list(temps):
        cpts = " ".join(f"{x_of(h):.1f},{y_of(t):.1f}" for h, t in enumerate(course["temps"]))
        parts.append(f'<polyline points="{cpts}" style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.3;'
                     f'stroke-dasharray:3 2"/>')
    pts = " ".join(f"{x_of(h):.1f},{y_of(t):.1f}" for h, t in enumerate(temps))
    parts.append(f'<polyline points="{pts}" style="fill:none;stroke:var(--ea-ink);stroke-width:2;stroke-linejoin:round"/>')
    for h, t in enumerate(temps):
        if t > LIMIT:
            parts.append(f'<circle cx="{x_of(h):.1f}" cy="{y_of(t):.1f}" r="3" style="fill:var(--ea-loss)"/>')
    # the cooling of every hour
    ay = y_top + th + 10
    ah = 8
    fill = "var(--ea-param)" if learned else "var(--ea-ink-2)"
    parts.append(_words(x0 - 5, ay + 2 * ah - 2, w["action"], anchor="end", size=9))
    for h, a in enumerate(dd["actions"]):
        if a:
            parts.append(f'<rect x="{x_of(h) + cw * 0.15:.1f}" y="{ay + (2 - a) * ah:.1f}" width="{cw * 0.7:.1f}" '
                         f'height="{a * ah:.1f}" style="fill:{fill};fill-opacity:0.85"/>')
        if course is not None and course["actions"][h] != a:
            ca = course["actions"][h]
            parts.append(f'<rect x="{x_of(h) + cw * 0.1:.1f}" y="{ay + (2 - ca) * ah:.1f}" width="{cw * 0.8:.1f}" '
                         f'height="{max(ca * ah, 0.8):.1f}" style="fill:none;stroke:var(--ea-ink-2);stroke-width:1;'
                         f'stroke-dasharray:3 2"/>')
    parts.append(f'<line x1="{x0}" y1="{ay + 2 * ah}" x2="{x1}" y2="{ay + 2 * ah}" style="stroke:var(--ea-line);stroke-width:1"/>')
    bottom = ay + 2 * ah + 13
    for h in (0, 6, 12, 18, 24):
        parts.append(_words(x_of(h), bottom, str(h), size=9.5))
    parts.append(_words((x0 + x1) / 2, bottom + 13, w["hour"], size=10.5))
    return _svg(W, bottom + 19, parts)


def days_picture(lang: str) -> str:
    """The idea: a day of the thermostat, and a day of the agent that learned in the simulator."""
    lang = lang if lang in LANGS else "en"
    w = _DAY_TEXT[lang]
    learned = day(course_run().greedy())
    return _figure([_day_svg(THERMOSTAT, w["thermostat"], lang), _day_svg(learned, w["learned"], lang, learned=True)],
                   w["alt_days"], 690, gap="0.6rem 1rem")


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


def sim_day(step_, policy=thermostat) -> dict:
    """One day in a simulator `step_` with `policy`: what the simulator says happens, hour by hour."""
    step_ = _checked(step_)
    temp, temps, actions, cost, hot = START, [START], [], 0.0, 0
    for hour in range(HOURS):
        a = policy(temp, hour)
        temp, reward = step_(temp, hour, a)
        temps.append(temp)
        actions.append(a)
        cost -= reward
        hot += temp > LIMIT
    return {"temps": temps, "actions": actions, "cost": cost, "hot": hot}


_YOURS_TEXT = {
    "en": {"step": {"title": "Your step",
                    "intro": "One day of the thermostat in your simulator: the temperature your `step` gives after every hour, and the cooling the thermostat chooses. Dashed, the course's day where yours differs.",
                    "chart": "Your simulator: the thermostat's day",
                    "say": "In your simulator, the thermostat's day costs {cost}, with {hot} hours above 27 degrees. In the course's simulator it costs 103, with none.",
                    "none": "Your `step` stops with an error, takes too long, or gives something other than a temperature from 15 to 35 and a number, so nothing can be drawn yet."},
           "act": {"title": "Your act",
                   "intro": "The agent of the course, after 10,000 days in the simulator, on a warm night it never saw: the servers stay busy at night. Left, the agent alone; right, the agent with your `act`. Dashed, the course's `act` where yours differs.",
                   "alone": "Without a safety rule: the table alone",
                   "mine": "With your act: on the warm night",
                   "say": "On the warm night, the table alone goes too hot in {alone} hours; with your `act`, in {mine} hours, at a cost of {cost}.",
                   "none": "Your `act` stops with an error, takes too long, or gives something other than an action 0, 1 or 2, so nothing can be drawn yet."}},
    "nl": {"step": {"title": "Je step",
                    "intro": "Eén dag van de thermostaat in je simulator: de temperatuur die je `step` na elk uur geeft, en de koeling die de thermostaat kiest. Gestreept de dag van de course waar je eigen dag anders is.",
                    "chart": "Je simulator: de dag van de thermostaat",
                    "say": "In je simulator kost de dag van de thermostaat {cost}, met {hot} uur boven 27 graden. In de simulator van de course kost de dag 103, met geen enkel uur te heet.",
                    "none": "Je `step` stopt met een fout, duurt te lang, of geeft iets anders dan een temperatuur van 15 tot en met 35 en een getal, dus is er nog geen afbeelding."},
           "act": {"title": "Je act",
                   "intro": "De agent van de course, na 10.000 dagen in de simulator, in een warme nacht die de agent nooit zag: de servers blijven 's nachts druk. Links de agent alleen; rechts de agent met je `act`. Gestreept de `act` van de course waar je eigen `act` anders is.",
                   "alone": "Zonder veiligheidsregel: alleen de tabel",
                   "mine": "Met je act: in de warme nacht",
                   "say": "In de warme nacht gaat de tabel alleen {alone} uur te heet; met je `act` {mine} uur, en dan kost de dag {cost}.",
                   "none": "Je `act` stopt met een fout, duurt te lang, of geeft iets anders dan een action 0, 1 of 2, dus is er nog geen afbeelding."}},
}


def step_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["step"]
    fn = _fn(seen)
    got = _call(lambda: sim_day(fn)) if fn is not None else None
    if not isinstance(got, dict):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    svg = _day_svg(got, w["chart"], lang, course=THERMOSTAT)
    say = w["say"].format(cost=short_number(got["cost"], lang), hot=got["hot"])
    return yours.figure(w["title"], _note(w["intro"]) + _figure([svg], w["title"], 330) + _note(say), alt=w["title"])


def _act_day(act_, heat, run_=None) -> dict | None:
    """One day with the table of `run_` (the course's run unless given) and the safety rule `act_`."""
    safe = (course_run() if run_ is None else run_).safe(act_)

    def policy(temp, hour):
        a = safe(temp, hour)
        if isinstance(a, bool) or not isinstance(a, (int, np.integer)) or a not in (0, 1, 2):
            raise ValueError("not an action")
        return int(a)
    return day(policy, heat)


def act_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["act"]
    fn = _fn(seen)
    heat = REALITIES["night"]
    course_run()   # trained once, outside the time the learner's act gets
    got = _call(lambda: _act_day(fn, heat)) if fn is not None else None
    if not isinstance(got, dict):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    alone = day(course_run().greedy(), heat)
    course = day(course_run().safe(), heat)
    svgs = [_day_svg(alone, w["alone"], lang, learned=True), _day_svg(got, w["mine"], lang, learned=True, course=course)]
    say = w["say"].format(alone=alone["hot"], mine=got["hot"], cost=short_number(got["cost"], lang))
    return yours.figure(w["title"], _note(w["intro"]) + _figure(svgs, w["title"], 690, gap="0.6rem 1rem") + _note(say),
                        alt=w["title"])


# --- the demo: training in the simulator, then a day in reality -------------------------------------------------------------

DEMO_TEXT = {
    "en": {"title": "Cost of each day of training", "x": "day", "y": "cost",
           "alt": "The cost of every day of training so far, the bill plus 10 for every hour too hot, with its mean over 100 days as the thicker line, and dashed the cost of a day with the thermostat.",
           "reality": "The day in reality", "safety": "Safety rule",
           "realities": {"sim": "as in the simulator", "afternoon": "a hotter afternoon", "night": "a warm night"},
           "safeties": {"off": "none", "on": "act: the thermostat in unknown states"},
           "cost": "cost of a day", "thermostat": "thermostat",
           "now": "Day {d}: what the agent would do now", "start": "Day 0: the table is still empty",
           "learning": "Day {d} of {n}: {hot} hours too hot so far",
           "say": "After {d} days, the agent's day in the simulator costs {cost}, with {hot} hours too hot; in training so far, {total} hours were too hot.",
           "say_start": "Before training, every value in the table is 0, so the agent takes action 0 everywhere: off, and the room heats up to 35 degrees.",
           "agent": "The agent: {reality}", "agent_safe": "The agent with act: {reality}", "rule": "The thermostat: {reality}",
           "who_mine": "your `step` as the simulator", "who_course": "the course's `step` as the simulator (yours does not pass its check yet)",
           "act_mine": "your `act`", "act_course": "the course's `act` (yours does not pass its check yet)",
           "verdict": "In 10,000 days of training with {who}, the room was too hot in **{total}** hours, about {years} years, all of them free in a simulator. Then one day in reality, {reality}: the agent{safety} costs **{cost}**, with **{hot}** hours too hot, and the thermostat costs **{rule}**, with **{rule_hot}** hours too hot.",
           "with_act": " with {act}"},
    "nl": {"title": "Kosten van elke dag training", "x": "dag", "y": "kosten",
           "alt": "De kosten van elke dag training tot nu toe, de rekening plus 10 voor elk uur te heet, met het gemiddelde over 100 dagen als dikkere lijn, en gestreept de kosten van een dag met de thermostaat.",
           "reality": "De dag in werkelijkheid", "safety": "Veiligheidsregel",
           "realities": {"sim": "zoals in de simulator", "afternoon": "een warmere middag", "night": "een warme nacht"},
           "safeties": {"off": "geen", "on": "act: de thermostaat in onbekende states"},
           "cost": "kosten van een dag", "thermostat": "thermostaat",
           "now": "Dag {d}: wat de agent nu zou doen", "start": "Dag 0: de tabel is nog leeg",
           "learning": "Dag {d} van {n}: tot nu toe {hot} uur te heet",
           "say": "Na {d} dagen kost de dag van de agent in de simulator {cost}, met {hot} uur te heet; in de training tot nu toe was het {total} uur te heet.",
           "say_start": "Voor de training is elke waarde in de tabel 0, dus kiest de agent overal action 0: uit, en warmt de ruimte op tot 35 graden.",
           "agent": "De agent: {reality}", "agent_safe": "De agent met act: {reality}", "rule": "De thermostaat: {reality}",
           "who_mine": "je `step` als simulator", "who_course": "de `step` van de course als simulator (je eigen `step` haalt de check nog niet)",
           "act_mine": "je `act`", "act_course": "de `act` van de course (je eigen `act` haalt de check nog niet)",
           "verdict": "In 10.000 dagen training met {who} was de ruimte **{total}** uur te heet, ongeveer {years} jaar, allemaal gratis in een simulator. Dan één dag in werkelijkheid, {reality}: de agent{safety} kost **{cost}**, met **{hot}** uur te heet, en de thermostaat kost **{rule}**, met **{rule_hot}** uur te heet.",
           "with_act": " met {act}"},
}


def reality_options(lang: str) -> dict:
    return {DEMO_TEXT[lang]["realities"][k]: k for k in REALITY_KEYS}


def safety_options(lang: str) -> dict:
    return {DEMO_TEXT[lang]["safeties"][k]: k for k in ("off", "on")}


def chart_labels(lang: str) -> dict:
    w = DEMO_TEXT[lang]
    return {"title": w["title"], "x": w["x"], "y": w["y"], "alt": w["alt"]}


def chart_key(lang: str) -> list:
    w = DEMO_TEXT[lang]
    return [("loss", False, w["cost"]), ("ink-2", True, w["thermostat"])]


def demo_xticks() -> list:
    return [1, 2000, 4000, 6000, 8000, 10000]


def thermostat_line() -> list:
    """The cost of a day with the thermostat, as a line across the chart."""
    return [(1, float(THERMOSTAT["cost"])), (DAYS, float(THERMOSTAT["cost"]))]


def demo_run(step_, show=None, pause: float = 0.0, seconds: float = 30.0) -> Run | str:
    """The demo: Q-learning in the simulator `step_` for DAYS days, with `show(run)` every 100 days; "error" when
    the learner's step stops with an error, takes too long, or gives something other than a temperature and a
    number. `pause` waits that long after each show, so that you can follow the training."""
    import time

    def shown(run_):
        if show is not None:
            show(run_)
        if pause:
            time.sleep(pause)
    got = _call(lambda: train(_checked(step_), show=shown), seconds + pause * DAYS / 100)
    return got if isinstance(got, Run) else "error"


def now_picture(run_: Run, lang: str) -> str:
    """What the agent would do now: one day in the simulator with its greedy policy."""
    w = DEMO_TEXT[lang]
    dd = day(run_.greedy())
    title = w["now"].format(d=number(run_.days, 0, lang)) if run_.days else w["start"]
    return _figure([_day_svg(dd, title, lang, learned=True)], title, 330)


def _mean_points(costs: list, upto: int, block: int = 100) -> list:
    return [(min(k + block, upto), float(np.mean(costs[k:min(k + block, upto)]))) for k in range(0, upto, block)]


def _chart_svg(run_: Run, d: int, lang: str, W: int = 330) -> str:
    """The rewind's chart: the mean cost of every 100 days of training, the thermostat dashed, a line at day d."""
    w = DEMO_TEXT[lang]
    x0, x1, y0, h = 40, W - 24, 28, 150
    high = 200.0

    def x_of(x: float) -> float:
        return x0 + x / DAYS * (x1 - x0)

    def y_of(v: float) -> float:
        return y0 + h - min(max(v, 0.0), high) / high * h

    parts = [_words(2, 11, w["title"], anchor="start", size=12, fill="var(--ea-ink)", weight=600)]
    for v in (0, 50, 100, 150, 200):
        parts.append(f'<line x1="{x0}" y1="{y_of(v):.1f}" x2="{x1}" y2="{y_of(v):.1f}" style="stroke:var(--ea-line);stroke-width:0.8"/>')
        parts.append(_words(x0 - 5, y_of(v) + 3.5, str(v), anchor="end", size=9.5))
    t = float(THERMOSTAT["cost"])
    parts.append(f'<line x1="{x0}" y1="{y_of(t):.1f}" x2="{x1}" y2="{y_of(t):.1f}" style="stroke:var(--ea-ink-2);stroke-width:1.2;'
                 f'stroke-dasharray:5 4"/>')
    parts.append(_words(x1, y_of(t) - 4, w["thermostat"], anchor="end", size=9.5))
    pts = _mean_points(run_.costs, DAYS)
    shown = [p for p in pts if p[0] <= max(d, 0)] if d else []
    if shown:
        line = " ".join(f"{x_of(x):.1f},{y_of(v):.1f}" for x, v in shown)
        parts.append(f'<polyline points="{line}" style="fill:none;stroke:var(--ea-loss);stroke-width:2;stroke-linejoin:round"/>')
    parts.append(f'<line x1="{x_of(d):.1f}" y1="{y0}" x2="{x_of(d):.1f}" y2="{y0 + h}" style="stroke:var(--ea-ink-2);stroke-width:1"/>')
    for x in demo_xticks():
        parts.append(_words(x_of(x), y0 + h + 14, number(x, 0, lang), size=9.5))
    parts.append(_words((x0 + x1) / 2, y0 + h + 28, w["x"], size=10.5))
    return _svg(W, y0 + h + 34, parts)


def rewind_frame(run_: Run, d: int, lang: str) -> tuple:
    """A stop of the rewind after the demo: the chart with a line at day d, the agent's day in the simulator then, and
    a sentence."""
    lang = lang if lang in LANGS else "en"
    w = DEMO_TEXT[lang]
    d = int(d)
    Q = run_.snaps[d]
    dd = day(run_.greedy(Q))
    title = w["now"].format(d=number(d, 0, lang)) if d else w["start"]
    picture = _figure([_day_svg(dd, title, lang, learned=True)], title, 330)
    svg = _figure([_chart_svg(run_, d, lang)], w["alt"], 330)
    if d:
        say = w["say"].format(d=number(d, 0, lang), cost=short_number(dd["cost"], lang), hot=dd["hot"],
                              total=number(sum(run_.hots[:d]), 0, lang))
    else:
        say = w["say_start"]
    return live.rewound(svg, [picture], width=330), say


def test_days(run_: Run, reality: str, safety: str, act_=act) -> tuple:
    """One day in reality after training: the agent, with the safety rule or without, and the thermostat."""
    heat = REALITIES[reality]
    if safety == "on":
        agent = _act_day(act_, heat, run_)
    else:
        agent = day(run_.greedy(), heat)
    return agent, day(thermostat, heat)


def test_picture(run_: Run, reality: str, safety: str, lang: str, act_=act) -> str:
    w = DEMO_TEXT[lang]
    agent, rule = test_days(run_, reality, safety, act_)
    where = w["realities"][reality]
    name = w["agent_safe"] if safety == "on" else w["agent"]
    svgs = [_day_svg(agent, name.format(reality=where), lang, learned=True),
            _day_svg(rule, w["rule"].format(reality=where), lang)]
    return _figure(svgs, w["alt"], 690, gap="0.6rem 1rem")


def verdict(run_: Run, reality: str, safety: str, mine: bool, act_mine: bool, lang: str, act_=act) -> str:
    w = DEMO_TEXT[lang]
    agent, rule = test_days(run_, reality, safety, act_)
    total = sum(run_.hots)
    who = w["who_mine"] if mine else w["who_course"]
    extra = w["with_act"].format(act=w["act_mine"] if act_mine else w["act_course"]) if safety == "on" else ""
    return w["verdict"].format(who=who, total=number(total, 0, lang), years=short_number(total / 8760, lang),
                               reality=w["realities"][reality], safety=extra, cost=short_number(agent["cost"], lang),
                               hot=agent["hot"], rule=short_number(rule["cost"], lang), rule_hot=rule["hot"])


# --- the exercises -----------------------------------------------------------------------------------------------------

REQUIRED = ["step", "act", "learn_where", "zero_shot"]


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


def same_pair(a, b) -> bool:
    """step: a new temperature and a reward, each the same number as the course's."""
    try:
        (n1, r1), (n2, r2) = a, b
        return abs(float(n1) - float(n2)) < 1e-9 and abs(float(r1) - float(r2)) < 1e-9
    except (TypeError, ValueError):
        return False


def _num(x) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return repr(x)
    return str(int(v)) if v.is_integer() else repr(round(v, 3))


def show_pair(v) -> dict:
    if isinstance(v, tuple) and len(v) == 2:
        shown = f"`({_num(v[0])}, {_num(v[1])})`"
    else:
        text_ = repr(v)
        shown = f"`{text_ if len(text_) <= 60 else text_[:57] + '...'}`"
    return {"en": shown, "nl": shown}


def same_action(a, b) -> bool:
    return not isinstance(a, bool) and isinstance(a, (int, np.integer)) and int(a) == int(b)


# exercise 1: step
def _s(line: str = "-PRICE[hour] * ENERGY[action]") -> str:
    return ("def step(temp, hour, action):\n"
            "    new = temp + HEAT[hour] - COOL[action]\n"
            "    new = min(max(new, COLDEST), HOTTEST)\n"
            f"    reward = {line}\n"
            "    if new > LIMIT:\n"
            "        reward = reward - PENALTY\n"
            "    return new, reward\n")


_S_START = _s("...")
_S = _s()
_S_SIGN = _s("PRICE[hour] * ENERGY[action]")
_S_NO_PRICE = _s("-ENERGY[action]")
_S_DEGREES = _s("-PRICE[hour] * COOL[action]")
_S_ACTION = _s("-PRICE[hour] * action")
S_NAMES = {"HEAT": HEAT, "PRICE": PRICE, "COOL": COOL, "ENERGY": ENERGY, "LIMIT": LIMIT, "PENALTY": PENALTY,
           "COLDEST": COLDEST, "HOTTEST": HOTTEST}


# exercise 2: act
def _a(test: str = "count == 0") -> str:
    return ("def act(q, count, temp, hour):\n"
            f"    if {test}:\n"
            "        return thermostat(temp, hour)\n"
            "    return int(np.argmax(q))\n")


_A_START = _a("...")
_A = _a()
_A_INVERTED = _a("count > 0")
_A_LIST = _a("q == 0")
_A_ONCE = _a("count < 2")
A_NAMES = {"np": np, "thermostat": thermostat}
_A_NO_RETURN = ("def act(q, count, temp, hour):\n"
                "    if count == 0:\n"
                "        action = thermostat(temp, hour)\n"
                "    else:\n"
                "        action = int(np.argmax(q))\n")

EXERCISES = {
    "step": FunctionBox(
        name="step",
        function="step",
        start=_S_START,
        solution=_S,
        names=S_NAMES,
        compare=same_pair,
        show=show_pair,
        visible=[(24, 10, 1)],
        hidden=[(26, 12, 0), (16, 3, 2), (34, 14, 0), (27, 22, 1), (25, 9, 2), (20, 0, 1)],
        first=[_DOTS],
        picture=step_picture,
        mistakes=[
            Mistake("sign", wrong=_S_SIGN,
                    en="`{call}` gives {got}, and it should give {want}: the bill counts as a reward above 0, so the more electricity the agent uses, the more reward it gets, and it learns to cool as hard as it can. What sign does a cost get as a reward?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de rekening telt als een reward boven 0, dus hoe meer stroom de agent gebruikt, hoe meer reward de agent krijgt, en leert de agent zo hard mogelijk te koelen. Welk teken krijgt een kostenpost als reward?"),
            Mistake("no-price", wrong=_S_NO_PRICE,
                    en="`{call}` gives {got}, and it should give {want}: every unit of electricity costs 1, at night and in the day, so the agent has no reason to cool at night, when electricity is cheap. Where does the price of this hour come in?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke eenheid stroom kost 1, 's nachts en overdag, dus heeft de agent geen reden om 's nachts te koelen, als stroom goedkoop is. Waar komt de prijs van dit uur in?"),
            Mistake("degrees", wrong=_S_DEGREES,
                    en="`{call}` gives {got}, and it should give {want}: the bill counts the degrees the action takes away, not the electricity it uses, so high cooling looks exactly twice as expensive as low, while it uses three times the energy. Which list holds the units of electricity of each action?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de rekening telt de graden die de action wegneemt, niet de stroom die de action gebruikt, dus lijkt hoog koelen precies twee keer zo duur als laag, terwijl hoog drie keer zoveel stroom gebruikt. Welke lijst bevat de eenheden stroom van elke action?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the reward of this hour without the penalty: minus the bill, the price of this hour times the units of electricity of the action.",
                      "nl": "Vervang de `...` door de reward van dit uur zonder de straf: min de rekening, de prijs van dit uur keer de eenheden stroom van de action."},
            "arguments": {"en": "The check calls `{call}`, with a temperature, an hour and an action, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(temp, hour, action)`?",
                          "nl": "De check roept `{call}` aan, met een temperatuur, een uur en een action, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(temp, hour, action)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is `reward` minus the price of this hour times the units of electricity of the action?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Is `reward` min de prijs van dit uur keer de eenheden stroom van de action?"},
            "almost": {"en": "Right for low cooling. But `{call}` gives {got}, and it should give {want}. Does your bill use `ENERGY[action]`, the units of electricity the action uses?",
                       "nl": "Goed voor laag koelen. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je rekening `ENERGY[action]`, de eenheden stroom die de action gebruikt?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return new, reward`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return new, reward`?"},
            "correct": {"en": "Right in all {n} cases: low cooling at hour 10 costs 4, the price of the day, and an hour above 27 degrees costs 10 more. In the picture, at which hour does the thermostat first cool high?",
                        "nl": "Goed in alle {n} gevallen: laag koelen om 10 uur kost 4, de prijs van overdag, en een uur boven 27 graden kost 10 extra. Op welk uur koelt de thermostaat in de afbeelding voor het eerst hoog?"},
        },
    ),
    "act": FunctionBox(
        name="act",
        function="act",
        start=_A_START,
        solution=_A,
        names=A_NAMES,
        compare=same_action,
        visible=[([0.0, 0.0, 0.0], 0, 26, 3)],
        hidden=[([-58.0, -51.5, -60.2], 40, 22, 10), ([-12.0, -9.5, -14.0], 1, 22, 20), ([0.0, 0.0, 0.0], 0, 33, 2),
                ([0.0, 0.0, 0.0], 0, 15, 5), ([-30.5, -33.0, -31.0], 7, 24, 15)],
        first=[_DOTS],
        picture=act_picture,
        mistakes=[
            Mistake("inverted", wrong=_A_INVERTED,
                    en="`{call}` gives {got}, and it should give {want}: the thermostat takes over in every state the simulator showed, and the table decides in the states it never saw, where it knows nothing. When should the thermostat take over?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de thermostaat neemt het over in elke state die de simulator liet zien, en de tabel beslist in de states die de simulator nooit liet zien, waar de tabel niets weet. Wanneer moet de thermostaat het overnemen?"),
            Mistake("list", wrong=_A_LIST,
                    en="`{call}` gives {got}, and it should give {want}: `q` is a list of three numbers, and a list is never equal to the number 0, so the thermostat never takes over. Which argument says how often the simulator showed this state?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `q` is een lijst van drie getallen, en een lijst is nooit gelijk aan het getal 0, dus neemt de thermostaat het nooit over. Welk argument zegt hoe vaak de simulator deze state liet zien?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the test for a state the simulator never showed: `count` is 0.",
                      "nl": "Vervang de `...` door de test voor een state die de simulator nooit liet zien: `count` is 0."},
            "arguments": {"en": "The check calls `{call}`, with the three Q's of the state, how often it came by, the temperature and the hour, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(q, count, temp, hour)`?",
                          "nl": "De check roept `{call}` aan, met de drie Q's van de state, hoe vaak de state langskwam, de temperatuur en het uur, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(q, count, temp, hour)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does the thermostat take over exactly when `count` is 0, and does the table decide otherwise?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Neemt de thermostaat het precies over als `count` 0 is, en beslist anders de tabel?"},
            "almost": {"en": "Right for a state that never came by. But `{call}` gives {got}, and it should give {want}: this state came by once, so the table has learned something about it. Does the thermostat take over only when `count` is 0?",
                       "nl": "Goed voor een state die nooit langskwam. Maar `{call}` geeft {got}, en het moet {want} zijn: deze state kwam één keer langs, dus heeft de tabel er iets over geleerd. Neemt de thermostaat het alleen over als `count` 0 is?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does every way through it end with a `return`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt elke weg door je functie met een `return`?"},
            "correct": {"en": "Right in all {n} cases: in a state the simulator never showed, the thermostat decides, and everywhere else the table. In the picture, how many hours of the warm night are too hot with your `act`, and how many without it?",
                        "nl": "Goed in alle {n} gevallen: in een state die de simulator nooit liet zien beslist de thermostaat, en overal anders de tabel. Hoeveel uur van de warme nacht is het in de afbeelding te heet met je `act`, en hoeveel zonder?"},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("step", _S_START, "empty", None),
    ("step", _S, "correct", None),
    ("step", _s("-PRICE[hour] * ..."), "wrong", "dots-left"),
    ("step", _S_SIGN, "wrong", "sign"),
    ("step", _S_NO_PRICE, "wrong", "no-price"),
    ("step", _S_DEGREES, "wrong", "degrees"),
    ("step", _S.replace("    return new, reward\n", ""), "wrong", "no-return"),
    # other right answers: the same number written another way
    ("step", _s("0 - PRICE[hour] * ENERGY[action]"), "correct", None),
    ("step", _s("-(ENERGY[action] * PRICE[hour])"), "correct", None),
    # the action number as the energy: right for off and low, wrong for high
    ("step", _S_ACTION, "almost", None),
    ("act", _A_START, "empty", None),
    ("act", _A, "correct", None),
    ("act", _a("count == ..."), "wrong", "dots-left"),
    ("act", _A_INVERTED, "wrong", "inverted"),
    ("act", _A_LIST, "wrong", "list"),
    ("act", _A_NO_RETURN, "wrong", "no-return"),
    # other right answers: the same test written another way
    ("act", _a("not count"), "correct", None),
    ("act", _a("count < 1"), "correct", None),
    # a state seen once also handed to the thermostat: right for unseen states only
    ("act", _A_ONCE, "almost", None),
]
