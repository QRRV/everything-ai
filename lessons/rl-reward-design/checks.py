# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rl-reward-design: a cleaning robot in a room of 3 by 5 cells, with three pieces of dirt and a
bin. The world only moves the robot; the reward is a function of its own, written by the designer. With a reward for
every piece picked up, Q-learning learns to pick up one piece and drop it again, over and over; with a reward for every
piece that leaves the room, it learns to clean. The room, the course's Q-learning and its runs with both rewards,
pickup_reward, result_reward and dirt_left, the functions the learner writes, and the pictures.

The cleaning robot follows the example of Amodei and others (Concrete Problems in AI Safety, 2016); the room, the code,
the runs and the pictures are the course's own."""
import html
import math
import random
import re

import numpy as np

from everything_ai import chart, live, yours
from everything_ai.check import FunctionBox, Mistake, kept, with_time_limit

LESSON = "rl-reward-design"
LANGS = ("en", "nl")
ROWS, COLS = 3, 5
BIN = (0, 0)                        # where a piece of dirt is gone for good
START = (2, 4)                      # where the robot starts
DIRT = ((0, 4), (1, 2), (2, 0))     # the dirt at the start, as a sorted tuple of cells
BEGIN = (START, 0, DIRT)            # a state: (the robot's cell, 1 with a piece in its hand and 0 otherwise, the dirt on the floor)
ACTIONS = ["up", "down", "left", "right", "pick", "drop"]
MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
COST = 0.01         # the battery: every reward of the lesson takes this off every step
LIMIT = 30          # an episode stops after this many steps, also when the room is not clean
GAMMA = 0.9
ETA = 0.5
EPSILON = 0.1
EPISODES = 3000
SEEDS = 10          # the course's runs, with random seeds 0 to 9
LOOKS = [0, 10, 50, 100, 200, 500, 1000, 2000, 3000]   # the stops of the demo
WINDOW = 50         # the chart shows the mean of the last 50 episodes
SHORTEST = 22       # the fewest steps that clean the room (found by a search over all states)
REWARDS = ("pickup", "result")
KEEP = (100, 300, 500, 3000)        # the Q-tables kept of every run, for exercise 3 and its picture


class Named(dict):
    """A Q-table that a check message calls by its name; the learner's code gets a plain copy."""

    def __init__(self, entries, name: str):
        super().__init__(entries)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return {s: dict(qs) for s, qs in self.items()}


# --- the room, the course's solutions and Q-learning ------------------------------------------------------------------

def step(state, action):
    """One step in the room, and nothing else: the next state. A move into a wall leaves the robot where it is. "pick"
    on a piece of dirt with an empty hand puts the piece in the hand. "drop" with a piece in the hand puts it down: in
    the bin it is gone, on a clean cell it lies on the floor again, on a cell with dirt nothing happens."""
    cell, hand, dirt = state
    if action in MOVES:
        r, c = cell[0] + MOVES[action][0], cell[1] + MOVES[action][1]
        if 0 <= r < ROWS and 0 <= c < COLS:
            cell = (r, c)
        return (cell, hand, dirt)
    if action == "pick":
        if hand == 0 and cell in dirt:
            return (cell, 1, tuple(d for d in dirt if d != cell))
        return state
    if action == "drop" and hand == 1:
        if cell == BIN:
            return (cell, 0, dirt)
        if cell not in dirt:
            return (cell, 0, tuple(sorted(dirt + (cell,))))
    return state


def _in_room(state) -> int:
    return len(state[2]) + state[1]


def greedy(Q, state):
    """The action with the largest Q in `state`, the first one where several tie; a state the Q-table has never seen
    counts as all zeros, so it gives "up"."""
    qs = Q.get(state) if isinstance(Q, dict) else None
    if not isinstance(qs, dict):
        return ACTIONS[0]
    best, top = ACTIONS[0], qs.get(ACTIONS[0], 0.0)
    for a in ACTIONS[1:]:
        if qs.get(a, 0.0) > top:
            best, top = a, qs.get(a, 0.0)
    return best


def pickup_reward(state, nxt):
    """The course's pickup reward, the solution of exercise 1."""
    cell, hand, dirt = state
    new_cell, new_hand, new_dirt = nxt
    if hand == 0 and new_hand == 1:
        return 1 - COST
    return -COST


def result_reward(state, nxt):
    """The course's result reward, the solution of exercise 2."""
    cell, hand, dirt = state
    new_cell, new_hand, new_dirt = nxt
    before = len(dirt) + hand
    after = len(new_dirt) + new_hand
    return before - after - COST


def dirt_left(Q):
    """The course's dirt_left, the solution of exercise 3."""
    state = BEGIN
    for _ in range(LIMIT):
        action = greedy(Q, state)
        state = step(state, action)
    cell, hand, dirt = state
    return len(dirt) + hand


class _Bad(Exception):
    """A reward that is not a number."""


def _fine(x) -> bool:
    return isinstance(x, (int, float, np.floating, np.integer)) and not isinstance(x, bool) and math.isfinite(float(x))


def play(Q, reward=None, limit: int = LIMIT) -> dict:
    """The robot takes the action with the largest Q for `limit` steps from BEGIN, and stops early in a clean room:
    the cells it stood on, its pickups per cell, the pieces it put in the bin, the dirt left, the state at the end, and
    with `reward` the rewards of its steps added up."""
    s, cells, picks, binned, total = BEGIN, [START], {}, 0, 0.0
    for _ in range(limit):
        nxt = step(s, greedy(Q, s))
        if reward is not None:
            r = reward(s, nxt)
            if not _fine(r):
                raise _Bad
            total += float(r)
        if s[1] == 0 and nxt[1] == 1:
            picks[nxt[0]] = picks.get(nxt[0], 0) + 1
        binned += s[1] == 1 and nxt[1] == 0 and nxt[0] == BIN
        s = nxt
        cells.append(s[0])
        if _in_room(s) == 0:
            break
    return {"cells": cells, "picks": picks, "binned": binned, "left": _in_room(s), "end": s,
            "steps": len(cells) - 1, "total": total}


def train(reward, episodes: int = EPISODES, seed: int = 0, keep=(), looks=()) -> dict:
    """Q-learning in the room, as in Q-learning: ε-greedy steps from BEGIN until the room is clean or LIMIT steps, the
    reward of every step from reward(state, nxt). The Q-table; for every episode its reward (the rewards of its steps
    added up), the dirt left at its end and its pickups; copies of the Q-table after the episodes in `keep`; and what
    the robot does with the largest Q after the episodes in `looks`."""
    rnd = random.Random(seed)
    Q = {}
    out = {"returns": [], "left": [], "picks": [], "kept": {}, "plays": {}}
    if 0 in keep:
        out["kept"][0] = {}
    if 0 in looks:
        out["plays"][0] = play(Q)
    for n in range(1, episodes + 1):
        s, total, picks = BEGIN, 0.0, 0
        for _ in range(LIMIT):
            qs = Q.get(s)
            if qs is None:
                qs = Q[s] = dict.fromkeys(ACTIONS, 0.0)
            a = ACTIONS[int(rnd.random() * 6)] if rnd.random() < EPSILON else greedy(Q, s)
            nxt = step(s, a)
            r = reward(s, nxt)
            if not _fine(r):
                raise _Bad
            r = float(r)
            total += r
            picks += s[1] == 0 and nxt[1] == 1
            done = _in_room(nxt) == 0
            nq = Q.get(nxt)
            target = r if done else r + GAMMA * (max(nq.values()) if nq else 0.0)
            qs[a] += ETA * (target - qs[a])
            s = nxt
            if done:
                break
        out["returns"].append(total)
        out["left"].append(_in_room(s))
        out["picks"].append(picks)
        if n in keep:
            out["kept"][n] = {k: dict(v) for k, v in Q.items()}
        if n in looks:
            out["plays"][n] = play(Q)
    out["Q"] = Q
    return out


def smooth(curve: list) -> list:
    """For every episode, the mean of the last WINDOW values up to it."""
    out, run = [], 0.0
    for e, v in enumerate(curve):
        run += v
        if e >= WINDOW:
            run -= curve[e - WINDOW]
        out.append(run / min(e + 1, WINDOW))
    return out


def course_runs() -> dict:
    """Both rewards, 10 runs of 3,000 episodes each: per reward, the reward, dirt left and pickups of every episode of
    every run, their means over the runs, smoothed; what every run does with the largest Q at each stop of LOOKS; and
    the Q-tables of every run after the episodes in KEEP. Kept: the texts, the demo, the NumPy cell and exercise 3 use
    it."""
    def make():
        out = {}
        for name, fn in (("pickup", pickup_reward), ("result", result_reward)):
            runs = [train(fn, EPISODES, seed, keep=KEEP, looks=LOOKS) for seed in range(SEEDS)]
            mean = {k: [sum(r[k][e] for r in runs) / SEEDS for e in range(EPISODES)] for k in ("returns", "left", "picks")}
            out[name] = {"returns": [r["returns"] for r in runs], "left": [r["left"] for r in runs],
                         "picks": [r["picks"] for r in runs],
                         "smooth": {k: smooth(v) for k, v in mean.items()},
                         "plays": [r["plays"] for r in runs], "kept": [r["kept"] for r in runs]}
        return out
    return kept(f"{LESSON}:runs", make)


def history(name: str) -> tuple:
    """The NumPy cell: the reward and the dirt left of every episode of every run with one reward, as two arrays of
    shape (10, 3000): a row per run, a column per episode."""
    r = course_runs()[name]
    return np.array(r["returns"], dtype=float), np.array(r["left"], dtype=int)


def last_mean(name: str, key: str, n: int) -> float:
    """The mean over the runs of the last WINDOW episodes up to episode n."""
    return course_runs()[name]["smooth"][key][n - 1]


def course_q(name: str, n: int = EPISODES, seed: int = 0) -> dict:
    return course_runs()[name]["kept"][seed][n]


# --- the words and the pictures ----------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem"
_HALO = "paint-order:stroke;stroke:var(--ea-page);stroke-width:3px;stroke-linejoin:round"
CELL = 30           # two rooms side by side are 330 px: the most a play block allows for a picture beside a chart
_GAP = 26


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


def room_svg(panels: list, alt: str, lang: str) -> str:
    """Copies of the room side by side. Each panel is a dict: `title`; `dirt`, cells with a piece of dirt on the floor,
    drawn as dots; `rings`, cells where a piece lay at the start, drawn as rings; `cells`, the cells the robot stood
    on, in order, drawn as a line with a circle where it ends (orange: what it learned); `hand`, True when it ends with
    a piece in its hand, drawn as a small dot beside that circle; `picks` {cell: count}, how often it picked up dirt
    there, written in the corner of the cell; and `sub`, a line under the room. B is the bin and S the start."""
    cell = CELL
    pw = COLS * cell + 2
    W = len(panels) * pw + (len(panels) - 1) * _GAP
    top = 22 if any(p.get("title") for p in panels) else 1
    H = top + ROWS * cell + (22 if any(p.get("sub") for p in panels) else 2)
    parts = []
    for i, p in enumerate(panels):
        x0 = i * (pw + _GAP)
        if p.get("title"):
            parts.append(f'<text x="{x0 + 1}" y="14" style="font-size:13px;font-weight:600;fill:var(--ea-ink)">'
                         f'{html.escape(p["title"])}</text>')
        for r in range(ROWS):
            for c in range(COLS):
                fill = "var(--ea-inset)" if (r, c) == BIN else "var(--ea-page)"
                parts.append(f'<rect x="{x0 + 1 + c * cell}" y="{top + r * cell}" width="{cell}" height="{cell}" '
                             f'style="fill:{fill};stroke:var(--ea-line);stroke-width:1"/>')

        def centre(rc):
            return x0 + 1 + rc[1] * cell + cell / 2, top + rc[0] * cell + cell / 2

        for rc, sign in ((BIN, "B"), (START, "S")):
            x, y = centre(rc)
            parts.append(f'<text x="{x:.1f}" y="{y + 4.5:.1f}" style="font-size:13px;font-weight:700;fill:var(--ea-ink-2);'
                         f'text-anchor:middle;{_HALO}">{sign}</text>')
        for rc in p.get("rings") or ():
            x, y = centre(rc)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.4"/>')
        cells = p.get("cells") or []
        if len(cells) > 1:
            points = " ".join("{:.1f},{:.1f}".format(*centre(rc)) for rc in cells)
            parts.append(f'<polyline points="{points}" style="fill:none;stroke:var(--ea-param);stroke-width:3;'
                         'stroke-linecap:round;stroke-linejoin:round;opacity:0.75"/>')
        for rc in p.get("dirt") or ():
            x, y = centre(rc)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" style="fill:var(--ea-ink);stroke:var(--ea-page);stroke-width:1.2"/>')
        if cells:
            x, y = centre(cells[-1])
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9.5" style="fill:none;stroke:var(--ea-param);stroke-width:2.4"/>')
            if p.get("hand"):
                parts.append(f'<circle cx="{x + 9:.1f}" cy="{y - 9:.1f}" r="4" style="fill:var(--ea-ink);'
                             'stroke:var(--ea-page);stroke-width:1.2"/>')
        for (r, c), n in sorted((p.get("picks") or {}).items()):
            x, y = x0 + 1 + c * cell + cell - 3, top + r * cell + 11
            parts.append(f'<text x="{x:.1f}" y="{y:.1f}" style="font-size:10.5px;font-weight:700;fill:var(--ea-param-text);'
                         f'text-anchor:end;{_HALO}">×{n}</text>')
        if p.get("sub"):
            parts.append(f'<text x="{x0 + 1}" y="{top + ROWS * cell + 16}" style="font-size:12px;fill:var(--ea-ink-2)">'
                         f'{html.escape(p["sub"])}</text>')
    svg = (f'<svg viewBox="0 0 {W} {H}" aria-hidden="true" style="width:100%;max-width:{W}px;height:auto;'
           f'font-family:var(--ea-font-ui)">{"".join(parts)}</svg>')
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(alt)}" style="margin:0.25rem 0 0.5rem;'
            f'max-width:min(100%,{W}px)">{svg}</figure>')


_KEY_TEXT = {
    "en": "`S` start, `B` bin. A dot: a piece of dirt on the floor; a ring: where a piece lay at the start. The line: the steps of the robot, with a circle where it ends, and a small dot beside that circle for a piece in its hand; ×14: how often it picked up dirt in that cell.",
    "nl": "`S` start, `B` bak. Een stip: een stuk vuil op de vloer; een ring: waar een stuk aan het begin lag. De lijn: de stappen van de robot, met een cirkel waar de robot eindigt, en een kleine stip naast deze cirkel voor een stuk in de hand; ×14: hoe vaak de robot in dit vakje vuil oppakte.",
}


def room_key(lang: str) -> str:
    return _note(_KEY_TEXT[lang if lang in LANGS else "en"])


_SUB = {"en": "pickups: {p} · left: {l}", "nl": "opgepakt: {p} · over: {l}"}


def _panel(got: dict, title: str, lang: str) -> dict:
    """A room panel of what a robot did with the largest Q."""
    return {"title": title, "rings": DIRT, "cells": got["cells"], "dirt": got["end"][2], "hand": got["end"][1] == 1,
            "picks": got["picks"], "sub": _SUB[lang].format(p=sum(got["picks"].values()), l=got["left"])}


_PIC_TEXT = {
    "en": {"room": "The room", "room_alt": "The room of 3 rows and 5 columns: the bin in the top left corner, the robot's start in the bottom right corner, and three pieces of dirt, in the top right corner, in the middle and in the bottom left corner.",
           "pickup": "Pickup reward", "result": "Result reward",
           "idea_alt": "Two copies of the room, each with the steps of run 1 after 3,000 episodes, with the largest Q. With the pickup reward the robot goes to the piece in the top right corner and picks it up 14 times; all three pieces are still in the room. With the result reward it takes all three pieces to the bin."},
    "nl": {"room": "De kamer", "room_alt": "De kamer van 3 rijen en 5 kolommen: de bak in de hoek linksboven, de start van de robot in de hoek rechtsonder, en drie stukken vuil, in de hoek rechtsboven, in het midden en in de hoek linksonder.",
           "pickup": "Reward voor oppakken", "result": "Reward voor resultaat",
           "idea_alt": "Twee keer de kamer, elk met de stappen van run 1 na 3.000 episodes, met de grootste Q. Met de reward voor oppakken gaat de robot naar het stuk in de hoek rechtsboven en pakt het 14 keer op; alle drie de stukken liggen nog in de kamer. Met de reward voor het resultaat brengt de robot alle drie de stukken naar de bak."},
}


def world_picture(lang: str) -> str:
    """The room at the start, beside the guess."""
    lang = lang if lang in LANGS else "en"
    w = _PIC_TEXT[lang]
    return room_svg([{"title": w["room"], "dirt": DIRT}], w["room_alt"], lang) + room_key(lang)


def course_play(name: str, n: int = EPISODES, seed: int = 0) -> dict:
    return course_runs()[name]["plays"][seed][n]


def idea_picture(lang: str) -> str:
    """Run 1 after 3,000 episodes with each reward: what the robot does with the largest Q."""
    lang = lang if lang in LANGS else "en"
    w = _PIC_TEXT[lang]
    panels = [_panel(course_play(name), w[name], lang) for name in REWARDS]
    return room_svg(panels, w["idea_alt"], lang) + room_key(lang)


# --- the demo: the training of both robots ------------------------------------------------------------------------

DEMO_TEXT = {
    "en": {"reward": "Reward per episode", "left": "Dirt left after an episode", "x": "episodes", "y_reward": "reward",
           "y_left": "pieces", "pickup": "pickup reward", "result": "result reward",
           "alt": "Two charts over 3,000 episodes on a log scale, the mean of 10 runs over the last 50 episodes, with a line at {n}. Left, the reward per episode: with the pickup reward it climbs to about 12 within 100 episodes, with the result reward to about 2.6. Right, the dirt left after an episode: with the pickup reward always 3, with the result reward it goes down to almost 0.",
           "rooms_alt": "Two copies of the room with the steps of run 1 after {n} episodes, with the largest Q: with the pickup reward and with the result reward.",
           "room_pickup": "Pickup reward, run 1", "room_result": "Result reward, run 1",
           "say0": "Before the first episode every Q is 0, so both robots go up, the first action, at every step: they walk into the wall and stay there.",
           "say": "After {n} episodes: over the 10 runs and the last {k} episodes, the robot with the pickup reward got {a} per episode on average, with {la} of the 3 pieces left at the end, and the robot with the result reward {b}, with {lb} left. In run 1, with the largest Q: {pa}; {pb}.",
           "who_pickup": "the robot with the pickup reward", "who_result": "the robot with the result reward",
           "clean": "{who} cleans the room in {s} steps", "none": "{who} picks up nothing in its 30 steps",
           "all": "{who} picks up dirt {p} times and leaves all three pieces in the room",
           "once": "{who} picks up dirt once and leaves {l} of the 3 pieces in the room",
           "many": "{who} picks up dirt {p} times and leaves {l} of the 3 pieces in the room"},
    "nl": {"reward": "Reward per episode", "left": "Vuil over na een episode", "x": "episodes", "y_reward": "reward",
           "y_left": "stukken", "pickup": "reward voor oppakken", "result": "reward voor het resultaat",
           "alt": "Twee grafieken over 3.000 episodes op een logaritmische schaal, het gemiddelde van 10 runs over de laatste 50 episodes, met een lijn bij {n}. Links de reward per episode: met de reward voor oppakken stijgt de reward binnen 100 episodes naar ongeveer 12, met de reward voor het resultaat naar ongeveer 2,6. Rechts het vuil dat na een episode over is: met de reward voor oppakken altijd 3, met de reward voor het resultaat daalt het naar bijna 0.",
           "rooms_alt": "Twee keer de kamer met de stappen van run 1 na {n} episodes, met de grootste Q: met de reward voor oppakken en met de reward voor het resultaat.",
           "room_pickup": "Oppakken, run 1", "room_result": "Resultaat, run 1",
           "say0": "Bij de start is elke Q 0, dus gaan beide robots bij elke stap omhoog, de eerste action: de robots lopen tegen de muur en blijven daar staan.",
           "say": "Na {n} episodes: over de 10 runs en de laatste {k} episodes kreeg de robot met de reward voor oppakken gemiddeld {a} per episode, met {la} van de 3 stukken over aan het eind, en de robot met de reward voor het resultaat {b}, met {lb} over. In run 1, met de grootste Q: {pa}; {pb}.",
           "who_pickup": "de robot met de reward voor oppakken", "who_result": "de robot met de reward voor het resultaat",
           "clean": "{who} maakt de kamer schoon in {s} stappen", "none": "{who} pakt in de 30 stappen niets op",
           "all": "{who} pakt {p} keer vuil op en laat alle drie de stukken in de kamer liggen",
           "once": "{who} pakt één keer vuil op en laat {l} van de 3 stukken in de kamer liggen",
           "many": "{who} pakt {p} keer vuil op en laat {l} van de 3 stukken in de kamer liggen"},
}


def _doing(got: dict, name: str, lang: str) -> str:
    """What a robot does with the largest Q, in words for the demo's sentence."""
    w = DEMO_TEXT[lang]
    p = sum(got["picks"].values())
    who = w[f"who_{name}"]
    if got["left"] == 0:
        return w["clean"].format(who=who, s=got["steps"])
    if p == 0:
        return w["none"].format(who=who)
    key = "once" if p == 1 else "all" if got["left"] == len(DIRT) else "many"
    return w[key].format(who=who, p=p, l=got["left"])


def _demo_chart(n: int, lang: str) -> str:
    w = DEMO_TEXT[lang]
    runs = course_runs()
    xs = list(range(1, 101)) + list(range(110, 1001, 10)) + list(range(1050, EPISODES + 1, 50))
    p1 = chart.Panel((1, EPISODES), (-1, 14), title=w["reward"], xlabel=w["x"], ylabel=w["y_reward"], xlog=True,
                     xticks=[1, 10, 100, 1000], yticks=[0, 5, 10])
    p2 = chart.Panel((1, EPISODES), (0, 3.2), title=w["left"], xlabel=w["x"], ylabel=w["y_left"], xlog=True,
                     xticks=[1, 10, 100, 1000], yticks=[0, 1, 2, 3])
    if n >= 1:
        p1.line([n, n], [-1, 14], "axis", 1.2)
        p2.line([n, n], [0, 3.2], "axis", 1.2)
    for name, width, dashed in (("result", 1.4, True), ("pickup", 2.4, False)):
        sm = runs[name]["smooth"]
        shown = [x for x in xs if x <= n]
        if shown:
            p1.line(shown, [sm["returns"][x - 1] for x in shown], "ink", width, dashed=dashed)
            p2.line(shown, [sm["left"][x - 1] for x in shown], "loss", width, dashed=dashed)
    out = chart.figure([p1, p2], alt=w["alt"].format(n=n), width=300, height=230)
    out = chart.with_decimal_comma(out) if lang == "nl" else out
    return out + live.key_html([("ink", False, w["pickup"]), ("ink", True, w["result"])])


def demo_frame(n: int, lang: str) -> tuple:
    """A stop of the demo: both charts with a line at n, the robots of run 1 after n episodes, and a sentence."""
    lang = lang if lang in LANGS else "en"
    w = DEMO_TEXT[lang]
    n = min(LOOKS, key=lambda m: abs(m - int(n)))
    plays = {name: course_play(name, n) for name in REWARDS}
    rooms = room_svg([_panel(plays["pickup"], w["room_pickup"], lang), _panel(plays["result"], w["room_result"], lang)],
                     w["rooms_alt"].format(n=n), lang)
    if n == 0:
        say = w["say0"]
    else:
        say = w["say"].format(n=n, k=min(n, WINDOW), a=number(last_mean("pickup", "returns", n), 2, lang),
                              la=number(last_mean("pickup", "left", n), 2, lang),
                              b=number(last_mean("result", "returns", n), 2, lang),
                              lb=number(last_mean("result", "left", n), 2, lang),
                              pa=_doing(plays["pickup"], "pickup", lang), pb=_doing(plays["result"], "result", lang))
    return _demo_chart(n, lang) + rooms + room_key(lang), say


# --- the pictures of what the learner writes -------------------------------------------------------------------------

PICTURE_SECONDS = 4.0
_STOPPED = object()


def _call(make, seconds: float = PICTURE_SECONDS):
    import contextlib
    import io
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(make, max(seconds, 0.05))
    except Exception:   # noqa: BLE001   also Timeout: the check above says what went wrong
        return _STOPPED


def _fn_of(seen, name: str):
    run_ = seen.mine
    fn = run_.names.get(name) if run_ is not None else None
    return fn if callable(fn) else None


_YOURS_TEXT = {
    "en": {"pickup_reward": {"title": "Your pickup reward at work", "who": "`pickup_reward`", "panel": "Your robot, run 1"},
           "result_reward": {"title": "Your result reward at work", "who": "`result_reward`", "panel": "Your robot, run 1"},
           "head": "Q-learning with your {who}: run 1, 3,000 episodes, ε = 0.1, η = 0.5 and γ = 0.9. Then the robot takes the action with the largest Q, for at most 30 steps:",
           "say": "In {s} steps the robot {picks}; in the bin: {b} of the 3 pieces, still in the room: {l}. Your reward for these steps adds up to {t}.",
           "picks": ["never picks up dirt", "picks up dirt once", "picks up dirt {p} times"],
           "none": "Your {who} stops with an error, takes too long, or gives something that is not a number, so the robot cannot learn with it yet.",
           "alt": "The room with the steps of the robot that learned with your reward.",
           "d_title": "Your measurement", "d_head": "Your `dirt_left` for the robots of the course's 10 runs with each reward, after 100, 500 and 3,000 episodes: the mean of the 10 runs.",
           "d_key": "Filled bars: the pickup reward; open bars: the result reward.",
           "d_none": "Your `dirt_left` stops with an error, takes too long, or gives something that is not a number, so nothing can be drawn yet.",
           "d_alt": "Bars of the dirt left by the robots with the pickup reward and with the result reward, after 100, 500 and 3,000 episodes, as your dirt_left measures it.",
           "d_y": "dirt left", "d_x": "episodes"},
    "nl": {"pickup_reward": {"title": "Je reward voor oppakken aan het werk", "who": "`pickup_reward`", "panel": "Je robot, run 1"},
           "result_reward": {"title": "Je reward voor het resultaat aan het werk", "who": "`result_reward`", "panel": "Je robot, run 1"},
           "head": "Q-learning met je {who}: run 1, 3.000 episodes, ε = 0,1, η = 0,5 en γ = 0,9. Daarna neemt de robot de action met de grootste Q, hoogstens 30 stappen lang:",
           "say": "In {s} stappen {picks}; in de bak: {b} van de 3 stukken, nog in de kamer: {l}. Je reward voor deze stappen komt samen op {t}.",
           "picks": ["pakt de robot nooit vuil op", "pakt de robot één keer vuil op", "pakt de robot {p} keer vuil op"],
           "none": "Je {who} stopt met een fout, duurt te lang, of geeft iets wat geen getal is, dus de robot kan er nog niet mee leren.",
           "alt": "De kamer met de stappen van de robot die met je reward leerde.",
           "d_title": "Je meting", "d_head": "Je `dirt_left` voor de robots van de 10 runs van de course met elke reward, na 100, 500 en 3.000 episodes: het gemiddelde van de 10 runs.",
           "d_key": "Gevulde balken: de reward voor oppakken; open balken: de reward voor het resultaat.",
           "d_none": "Je `dirt_left` stopt met een fout, duurt te lang, of geeft iets wat geen getal is, dus er is nog geen afbeelding.",
           "d_alt": "Balken van het vuil dat de robots met de reward voor oppakken en met de reward voor het resultaat overlaten, na 100, 500 en 3.000 episodes, zoals je dirt_left het meet.",
           "d_y": "vuil over", "d_x": "episodes"},
}


def _reward_picture(seen, name: str) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]
    title, who = w[name]["title"], w[name]["who"]
    head = _note(w["head"].format(who=who))
    fn = _fn_of(seen, name)

    def make():
        t = train(fn, EPISODES, 0)
        return play(t["Q"], fn)

    got = _call(make) if fn is not None else None
    if not isinstance(got, dict):
        return yours.figure(title, head + _note(w["none"].format(who=who), _LOSS_UI), alt=title)
    panel = _panel(got, w[name]["panel"], lang)
    p = sum(got["picks"].values())
    say = w["say"].format(s=got["steps"], picks=w["picks"][min(p, 2)].format(p=p), b=got["binned"], l=got["left"],
                          t=number(got["total"], 2, lang))
    return yours.figure(title, head + room_svg([panel], w["alt"], lang) + room_key(lang) + _note(say), alt=title)


def pickup_picture(seen) -> str:
    return _reward_picture(seen, "pickup_reward")


def result_picture(seen) -> str:
    return _reward_picture(seen, "result_reward")


_BARS_AT = (100, 500, 3000)


def dirt_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]
    head = _note(w["d_head"])
    fn = _fn_of(seen, "dirt_left")
    runs = course_runs()

    def make():
        out = {}
        for name in REWARDS:
            row = []
            for n in _BARS_AT:
                vals = [fn({s: dict(q) for s, q in runs[name]["kept"][seed][n].items()}) for seed in range(SEEDS)]
                if not all(_fine(v) for v in vals):
                    raise _Bad
                row.append(sum(float(v) for v in vals) / SEEDS)
            out[name] = row
        return out

    got = _call(make) if fn is not None else None
    if not isinstance(got, dict):
        return yours.figure(w["d_title"], head + _note(w["d_none"], _LOSS_UI), alt=w["d_title"])
    top = max(3.2, max(max(v) for v in got.values()) * 1.15)
    labels = [str(n) for n in _BARS_AT]
    p = chart.Panel((-0.6, 2.6), (0, top), xlabel=w["d_x"], ylabel=w["d_y"], xticks=[0, 1, 2], xtick_labels=labels,
                    yticks=[0, 1, 2, 3])
    for i in range(len(_BARS_AT)):
        for name, dx, solid in (("pickup", -0.2, True), ("result", 0.2, False)):
            v = max(0.0, min(top, got[name][i]))
            p.rect(i + dx - 0.17, 0, i + dx + 0.17, v, "loss", solid=solid)
            p.text(i + dx, v + top * 0.05, number(got[name][i], 1, lang), "ink")
    out = chart.figure([p], alt=w["d_alt"], width=330, height=220)
    out = chart.with_decimal_comma(out) if lang == "nl" else out
    return yours.figure(w["d_title"], head + out + _note(w["d_key"]), alt=w["d_title"])


# --- the exercises ----------------------------------------------------------------------------------------------------

REQUIRED = ["pickup_reward", "result_reward", "dirt_left", "camera"]


def _dots_left(got, text) -> bool:
    return re.search(r"^[^#\n]*\.\.\.", text or "", re.MULTILINE) is not None


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Can you replace it with your own code?",
                nl="Er staat nog een `...` in je code. Kun je de `...` vervangen door je eigen code?")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def same_number(a, b) -> bool:
    return _fine(a) and _fine(b) and abs(float(a) - float(b)) < 1e-6


def show_number(v) -> dict:
    """A number the learner's function gave back, as code, rounded to four decimals."""
    if _fine(v):
        x = round(float(v), 4)
        shown = f"`{int(x) if x.is_integer() else x!r}`"
    else:
        text_ = repr(v)
        shown = f"`{text_ if len(text_) <= 80 else text_[:77] + '...'}`"
    return {"en": shown, "nl": shown}


# The steps the checks of exercises 1 and 2 give: (state, nxt).
PICK = (((0, 4), 0, DIRT), ((0, 4), 1, ((1, 2), (2, 0))))                       # picks up the piece on (0, 4)
MOVE = (BEGIN, ((1, 4), 0, DIRT))                                              # a step up from the start
CARRY = (((0, 3), 1, ((1, 2), (2, 0))), ((0, 2), 1, ((1, 2), (2, 0))))         # a step left with a piece in the hand
OVER_DIRT = (((1, 2), 1, ((1, 2), (2, 0))), ((1, 1), 1, ((1, 2), (2, 0))))     # a step left from a cell with dirt, carrying
FLOOR = (((0, 4), 1, ((1, 2), (2, 0))), ((0, 4), 0, DIRT))                     # drops the piece on the floor again
INTO_BIN = (((0, 0), 1, ((1, 2), (2, 0))), ((0, 0), 0, ((1, 2), (2, 0))))      # drops a piece in the bin
TO_BIN = (((0, 1), 0, DIRT), ((0, 0), 0, DIRT))                                # walks onto the bin with an empty hand
LAST_PICK = (((2, 0), 0, ((2, 0),)), ((2, 0), 1, ()))                          # picks up the last piece
LAST_BIN = (((0, 0), 1, ()), ((0, 0), 0, ()))                                  # the last piece in the bin: the room is clean

# exercise 1: pickup_reward
_R_HEAD = ("def {name}(state, nxt):\n"
           "    cell, hand, dirt = state\n"
           "    new_cell, new_hand, new_dirt = nxt\n")


def _p(test: str = "hand == 0 and new_hand == 1") -> str:
    return (_R_HEAD.format(name="pickup_reward") + f"    if {test}:\n"
            "        return 1 - COST\n"
            "    return -COST\n")


_P_START = _p("...")
_P = _p()
_P_HOLDING = _p("new_hand == 1")
_P_DROP = _p("hand == 1 and new_hand == 0")
_P_CHANGED = _p("new_dirt != dirt")
_P_ON_DIRT = _p("cell in dirt")
_P_EMPTY = _p("hand == 0")
R_NAMES = {"COST": COST}


# exercise 2: result_reward
def _r(after: str = "len(new_dirt) + new_hand", back: str = "before - after - COST") -> str:
    return (_R_HEAD.format(name="result_reward") + "    before = len(dirt) + hand\n"
            f"    after = {after}\n"
            f"    return {back}\n")


_R_START = _r("...", "...")
_R = _r()
_R_NO_HAND = _r("len(new_dirt)")
_R_SIGN = _r(back="after - before - COST")
_R_NO_COST = _r(back="before - after")
_R_PLUS_COST = _r(back="before - after + COST")
_R_LEVEL = _r(back="-after - COST")
_R_AT_BIN = (_R_HEAD.format(name="result_reward") + "    if new_cell == (0, 0):\n"
             "        return 1 - COST\n"
             "    return -COST\n")


# exercise 3: dirt_left
def _d(body: tuple = ("action = greedy(Q, state)", "state = step(state, action)"), back: str = "len(dirt) + hand",
       steps: str = "LIMIT") -> str:
    return ("def dirt_left(Q):\n"
            "    state = BEGIN\n"
            f"    for _ in range({steps}):\n" + "".join(f"        {line}\n" for line in body)
            + "    cell, hand, dirt = state\n"
            f"    return {back}\n")


_D_START = _d(("...",), "...")
_D = _d()
_D_NO_STEP = _d(("action = greedy(Q, state)",))
_D_FLOOR = _d(back="len(dirt)")
D_NAMES = {"BEGIN": BEGIN, "LIMIT": LIMIT, "greedy": greedy, "step": step}


def _q_cases() -> list:
    """The Q-tables the check gives dirt_left: run 1 with the result reward after 3,000 episodes; with the pickup reward;
    run 1 with the result reward after 100 and after 300 episodes (it ends with a piece in its hand); run 2 after 500
    episodes (it needs 26 steps); and an empty Q-table."""
    return [(Named(course_q("result"), "Q_result"),),
            (Named(course_q("pickup"), "Q_pickup"),),
            (Named(course_q("result", 100), "Q_100"),),
            (Named(course_q("result", 300), "Q_300"),),
            (Named(course_q("result", 500, 1), "Q_500"),),
            (Named({}, "{}"),)]


Q_CASES = _q_cases()


EXERCISES = {
    "pickup_reward": FunctionBox(
        name="pickup_reward",
        function="pickup_reward",
        start=_P_START,
        solution=_P,
        names=R_NAMES,
        compare=same_number,
        show=show_number,
        visible=[PICK],
        hidden=[MOVE, CARRY, OVER_DIRT, FLOOR, INTO_BIN, LAST_PICK],
        first=[_DOTS],
        picture=pickup_picture,
        mistakes=[
            Mistake("holding", wrong=_P_HOLDING,
                    en="`{call}` gives {got}: your reward pays for every step on which the robot has a piece in its hand, also when it only carries it. Which steps are a pickup: what was in the hand before the step?",
                    nl="`{call}` geeft {got}: je reward betaalt voor elke stap waarop de robot een stuk in de hand heeft, ook als de robot het alleen draagt. Welke stappen zijn oppakken: wat zat er in de hand bij het begin van de stap?"),
            Mistake("drop", wrong=_P_DROP,
                    en="`{call}` gives {got}: your reward pays for the step on which the robot puts a piece down, not for the one on which it picks it up. Which way does the hand change in a pickup?",
                    nl="`{call}` geeft {got}: je reward betaalt voor de stap waarop de robot een stuk neerlegt, niet voor de stap waarop de robot het oppakt. Hoe verandert de hand bij oppakken?"),
            Mistake("changed", wrong=_P_CHANGED,
                    en="`{call}` gives {got}: your reward pays for every step that changes the dirt on the floor, so also for dropping a piece on the floor. Which part of the state changes only in a pickup?",
                    nl="`{call}` geeft {got}: je reward betaalt voor elke stap die het vuil op de vloer verandert, dus ook voor een stuk op de vloer leggen. Welk deel van de state verandert alleen bij oppakken?"),
            Mistake("on-dirt", wrong=_P_ON_DIRT,
                    en="`{call}` gives {got}: your reward pays for every step that starts on a cell with dirt, whatever the robot does there. What tells you that it really picked something up?",
                    nl="`{call}` geeft {got}: je reward betaalt voor elke stap die begint op een vakje met vuil, wat de robot daar ook doet. Waaraan zie je dat de robot echt iets oppakte?"),
            Mistake("empty-hand", wrong=_P_EMPTY,
                    en="`{call}` gives {got}: your reward pays for every step that starts with an empty hand, also a step that only walks. What must the hand hold after a pickup?",
                    nl="`{call}` geeft {got}: je reward betaalt voor elke stap die begint met een lege hand, ook een stap waarin de robot alleen loopt. Wat moet de hand na het oppakken vasthouden?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the test for a pickup: the hand was empty, and after the step the robot holds a piece.",
                      "nl": "Vervang de `...` door de test voor oppakken: de hand was leeg, en na de stap houdt de robot een stuk vast."},
            "arguments": {"en": "The check calls `{call}`, with two states, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(state, nxt)`?",
                          "nl": "De check roept `{call}` aan, met twee states, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(state, nxt)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your test say that `hand` is 0 and `new_hand` is 1, both at once?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Zegt je test dat `hand` 0 is en `new_hand` 1, allebei tegelijk?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your test work in every state, also with a piece already in the hand, or fewer pieces on the floor?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je test in elke state, ook met al een stuk in de hand, of met minder stukken op de vloer?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return -COST`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return -COST`?"},
            "correct": {"en": "Right in all {n} cases: 0.99 for a pickup, −0.01 for every other step, also for a piece that goes into the bin. In the picture, Q-learning trains the robot with your reward; what does it do with the dirt?",
                        "nl": "Goed in alle {n} gevallen: 0,99 voor oppakken, −0,01 voor elke andere stap, ook voor een stuk dat in de bak gaat. In de afbeelding traint Q-learning de robot met je reward; wat doet de robot met het vuil?"},
        },
    ),
    "result_reward": FunctionBox(
        name="result_reward",
        function="result_reward",
        start=_R_START,
        solution=_R,
        names=R_NAMES,
        compare=same_number,
        show=show_number,
        visible=[INTO_BIN],
        hidden=[PICK, MOVE, FLOOR, TO_BIN, LAST_PICK, LAST_BIN],
        first=[_DOTS],
        picture=result_picture,
        mistakes=[
            Mistake("no-hand", wrong=_R_NO_HAND,
                    en="`{call}` gives {got}, and it should give {want}: when the robot picks up a piece, your `after` no longer counts it, but the piece is in the robot's hand, still in the room. What should `after` add, as `before` does?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: pakt de robot een stuk op, dan telt je `after` het niet meer, maar het stuk zit in de hand van de robot, nog in de kamer. Wat hoort `after` erbij op te tellen, net als `before`?"),
            Mistake("sign", wrong=_R_SIGN,
                    en="`{call}` gives {got}, and it should give {want}: a piece that leaves the room costs your robot a point, so it learns to keep the dirt. Which of the two is larger when the room gets cleaner, `before` or `after`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een stuk dat de kamer verlaat kost je robot een punt, dus leert de robot het vuil te houden. Welke van de twee is groter als de kamer schoner wordt, `before` of `after`?"),
            Mistake("no-cost", wrong=_R_NO_COST,
                    en="`{call}` gives {got}, and it should give {want}: the battery is missing, so a step costs nothing and the robot has no reason to hurry or to try new actions. What does every step cost?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de batterij ontbreekt, dus kost een stap niets en heeft de robot geen reden om op te schieten of nieuwe actions te proberen. Wat kost elke stap?"),
            Mistake("plus-cost", wrong=_R_PLUS_COST,
                    en="`{call}` gives {got}, and it should give {want}: your reward adds the battery instead of taking it off, so every step earns 0.01 and a long way pays. Which sign should `COST` get?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je reward telt de batterij erbij op in plaats van ervan af, dus levert elke stap 0,01 op en loont een lange weg. Welk teken hoort `COST` te krijgen?"),
            Mistake("level", wrong=_R_LEVEL,
                    en="`{call}` gives {got}, and it should give {want}: your reward takes off a point for every piece still in the room, at every step. That can teach cleaning too, but it is not the change in one step. How much less dirt is there after the step than before it?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je reward haalt bij elke stap een punt af voor elk stuk dat nog in de kamer ligt. Ook dat kan schoonmaken leren, maar het is niet de verandering in één stap. Hoeveel minder vuil is er na de stap dan bij het begin ervan?"),
            Mistake("at-bin", wrong=_R_AT_BIN,
                    en="`{call}` gives {got}, and it should give {want}: your reward pays for standing on the bin, also with an empty hand, so the robot can earn it by walking to B and staying there. What does `before - after` count instead?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je reward betaalt voor staan op de bak, ook met een lege hand, dus kan de robot de reward verdienen door naar B te lopen en daar te blijven. Wat telt `before - after` in plaats daarvan?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: `after`, the dirt in the room after the step, counted as `before` counts it, and then the reward.",
                      "nl": "Vervang de twee `...`: `after`, het vuil in de kamer na de stap, geteld zoals `before` het telt, en dan de reward."},
            "arguments": {"en": "The check calls `{call}`, with two states, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(state, nxt)`?",
                          "nl": "De check roept `{call}` aan, met twee states, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(state, nxt)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does `after` count the pieces on the floor and in the hand after the step, and does the reward give how much less dirt there is, minus `COST`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Telt `after` de stukken op de vloer en in de hand na de stap, en geeft de reward hoeveel minder vuil er is, min `COST`?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does every step cost the battery, also a step that changes nothing?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en het moet {want} zijn. Kost elke stap batterij, ook een stap die niets verandert?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it end with `return` and the reward?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie met `return` en de reward?"},
            "correct": {"en": "Right in all {n} cases: 0.99 for a piece in the bin, −0.01 for picking it up, carrying it and dropping it on the floor. In the picture, Q-learning trains the robot with your reward; how many pieces are left?",
                        "nl": "Goed in alle {n} gevallen: 0,99 voor een stuk in de bak, −0,01 voor het oppakken, het dragen en het op de vloer leggen. In de afbeelding traint Q-learning de robot met je reward; hoeveel stukken blijven er over?"},
        },
    ),
    "dirt_left": FunctionBox(
        name="dirt_left",
        function="dirt_left",
        start=_D_START,
        solution=_D,
        names=D_NAMES,
        compare=same_number,
        show=show_number,
        visible=Q_CASES[:1],
        hidden=Q_CASES[1:],
        first=[_DOTS],
        picture=dirt_picture,
        mistakes=[
            Mistake("no-step", wrong=_D_NO_STEP,
                    en="`{call}` gives {got}, and it should give {want}: the robot chooses an action at every step, but `state` never changes, so it stays at the start with all the dirt. Which line takes the robot on?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de robot kiest bij elke stap een action, maar `state` verandert nooit, dus blijft de robot op de start staan met al het vuil. Welke regel brengt de robot verder?"),
            Mistake("floor-only", wrong=_D_FLOOR,
                    en="`{call}` gives {got}, and it should give {want}: you count the dirt on the floor, but this robot ends with a piece in its hand, and that piece is still in the room. What does `hand` add?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je telt het vuil op de vloer, maar deze robot eindigt met een stuk in de hand, en dat stuk ligt nog in de kamer. Wat voegt `hand` toe?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: in the loop, the best action and the step; after it, the dirt still in the room.",
                      "nl": "Vervang de twee `...`: in de loop de beste action en de stap; daarna het vuil dat nog in de kamer ligt."},
            "arguments": {"en": "The check calls `{call}`, with one Q-table, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(Q)`?",
                          "nl": "De check roept `{call}` aan, met één Q-table, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(Q)` houden?"},
            "slow": {"en": "Your function is still not done after 2 seconds. Does the loop run `LIMIT` times, with one step each time?",
                     "nl": "Je functie is na 2 seconden nog niet klaar. Loopt de loop `LIMIT` keer, met elke keer één stap?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does every pass of the loop take the action of `greedy` in `state` and then let `step` move the robot, and do you give back the pieces on the floor plus the one in the hand?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Kiest elke ronde van de loop met `greedy` de action voor `state` en verplaatst `step` de robot daarna? En geef je de stukken op de vloer plus het stuk in de hand terug?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does the robot get all `LIMIT` steps, also when it needs more than the robot of the example?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en het moet {want} zijn. Krijgt de robot alle `LIMIT` stappen, ook als de robot er meer nodig heeft dan de robot uit het voorbeeld?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it end with `return` and the dirt left?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie met `return` en het vuil dat over is?"},
            "correct": {"en": "Right in all {n} cases: the robot of the result reward leaves 0, the robot of the pickup reward 3, and the robot after 300 episodes 1, in its hand. In the picture, your function measures the robots of all 10 runs; which reward cleans?",
                        "nl": "Goed in alle {n} gevallen: de robot van de reward voor het resultaat laat 0 over, de robot van de reward voor oppakken 3, en de robot na 300 episodes 1, in de hand. In de afbeelding meet je functie de robots van alle 10 runs; welke reward maakt schoon?"},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("pickup_reward", _P_START, "empty", None),
    ("pickup_reward", _P, "correct", None),
    ("pickup_reward", _p("hand == 0 and ..."), "wrong", "dots-left"),
    ("pickup_reward", _P_HOLDING, "wrong", "holding"),
    ("pickup_reward", _P_DROP, "wrong", "drop"),
    ("pickup_reward", _P_CHANGED, "wrong", "changed"),
    ("pickup_reward", _P_ON_DIRT, "wrong", "on-dirt"),
    ("pickup_reward", _P_EMPTY, "wrong", "empty-hand"),
    ("pickup_reward", _P.replace("    return -COST\n", ""), "wrong", "unreached"),
    # other right answers: the hand grows, and the dirt on the floor gets less
    ("pickup_reward", _p("new_hand > hand"), "correct", None),
    ("pickup_reward", _p("len(new_dirt) < len(dirt)"), "correct", None),
    # a pickup only from a full room: right for the example only
    ("pickup_reward", _p("len(new_dirt) == 2"), "almost", None),
    ("result_reward", _R_START, "empty", None),
    ("result_reward", _R, "correct", None),
    ("result_reward", _r(back="before - ..."), "wrong", "dots-left"),
    ("result_reward", _R_NO_HAND, "wrong", "no-hand"),
    ("result_reward", _R_SIGN, "wrong", "sign"),
    ("result_reward", _R_NO_COST, "wrong", "no-cost"),
    ("result_reward", _R_PLUS_COST, "wrong", "plus-cost"),
    ("result_reward", _R_LEVEL, "wrong", "level"),
    ("result_reward", _R_AT_BIN, "wrong", "at-bin"),
    ("result_reward", _R.replace("    return before - after - COST\n", ""), "wrong", "no-return"),
    # another right answer: the change first, then the battery
    ("result_reward", _r(back="(before - after) - 0.01"), "correct", None),
    # the battery only on a step that cleans: right for the example only
    ("result_reward", _r(back="(before - after) * (1 - COST)"), "almost", None),
    ("dirt_left", _D_START, "empty", None),
    ("dirt_left", _D, "correct", None),
    ("dirt_left", _d(back="len(dirt) + ..."), "wrong", "dots-left"),
    ("dirt_left", _D_NO_STEP, "wrong", "no-step"),
    ("dirt_left", _D_FLOOR, "wrong", "floor-only"),
    ("dirt_left", _D.replace("    return len(dirt) + hand\n", ""), "wrong", "no-return"),
    # another right answer: the step in one line, and a stop once the room is clean
    ("dirt_left", _d(("state = step(state, greedy(Q, state))", "if len(state[2]) + state[1] == 0:", "    break")), "correct", None),
    # 24 steps, as many as the robot of the example needs: right for the example only
    ("dirt_left", _d(steps="24"), "almost", None),
]
