# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rl-classic-environments: each classic benchmark of RL tests something else. CartPole balancing
from four numbers, MountainCar a reward that tells nothing until the goal, Atari learning from pixels, MuJoCo moving a
body with actions that are real numbers. For a Q-table, the real numbers of CartPole and MountainCar are first cut into
bins. The course's fixed policies and its Q-learner with bins, pole_policy, bin_of and row_of, the functions the learner
writes, the pictures, and the demo.

CartPole-v1 and MountainCar-v0 are environments of Gymnasium 1.4.0 (Farama Foundation, MIT); CartPole comes from Barto,
Sutton and Anderson (1983), MountainCar from Moore (1990). The code, the runs and the pictures are the course's own."""
import ast
import html
import math
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from everything_ai import chart, live, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, kept

LESSON = "rl-classic-environments"
LANGS = ("en", "nl")
ENVS = ("CartPole-v1", "MountainCar-v0")
SHORT = {"CartPole-v1": "CartPole", "MountainCar-v0": "MountainCar"}
ETA = 0.1           # the learning rate
GAMMA = 0.99        # the discount
EPSILON = 0.1       # the chance of a random action
EPISODES = 2000     # the episodes of one run, in both worlds
RUNS = 10           # the course's runs, seeds 0 to 9
BINS = (4, 8, 12, 20)   # the bins the demo offers for each number that counts
HOLD = 100          # exercise 1: a policy passes when it keeps the pole up this many steps in every episode of the check
TEST = 20           # the test episodes of a greedy policy, from seed TEST_SEED on
TEST_SEED = 1000
# The range that each number is cut in, and the bins of each: CartPole cuts only the angle and the angular velocity.
LIMITS = {"CartPole-v1": [(-2.4, 2.4), (-3.0, 3.0), (-0.21, 0.21), (-2.0, 2.0)],
          "MountainCar-v0": [(-1.2, 0.6), (-0.07, 0.07)]}
FRACTIONS = (0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)   # the stops of the rewind, as parts of the episodes


def make(name: str, **options):
    """A world of Gymnasium by its name."""
    return gym.make(name, **options)


def bins_for(name: str, b: int) -> tuple:
    """The bins of each number: b for the angle and the angular velocity of CartPole and one, so nothing, for its cart;
    b for both numbers of MountainCar."""
    return (1, 1, b, b) if name == "CartPole-v1" else (b, b)


def make_edges(name: str, b: int) -> list:
    """The inner edges of each number's bins, equally spread over its range: n bins have n - 1 edges, and one bin none."""
    return [np.linspace(lo, hi, n + 1)[1:-1] for (lo, hi), n in zip(LIMITS[name], bins_for(name, b))]


def looks() -> list:
    """The episodes after which the rewind shows the agent."""
    return [round(f * EPISODES) for f in FRACTIONS]


# --- the functions of the exercises, as the course writes them ----------------------------------------------------------

def pole_policy(obs):
    if obs[3] > 0:
        return 1   # push the cart to the right
    return 0       # push the cart to the left


def bin_of(x, edges):
    return int(np.sum(edges < x))


def row_of(obs, edges):
    row = 0
    for i in range(len(obs)):
        row = row * (len(edges[i]) + 1) + bin_of(obs[i], edges[i])
    return row


# --- fixed policies ---------------------------------------------------------------------------------------------------------

def swing_policy(obs):
    """MountainCar, the worked example: push the way the car moves, and right when it stands still."""
    if obs[1] >= 0:
        return 2
    return 0


def lean_policy(obs):
    """CartPole: push toward the side the pole leans."""
    return 1 if obs[2] > 0 else 0


def both_policy(obs):
    """CartPole: three times the angle plus the angular velocity: where it leans and where it falls, together."""
    return 1 if 3 * obs[2] + obs[3] > 0 else 0


def play(name: str, policy, seed: int, limit: int | None = None) -> tuple:
    """One episode of a fixed policy, the world seeded with `seed`: the steps, whether the task ended, and the
    observations along the way. `policy` None takes random actions from the action space, seeded too."""
    env = make(name)
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    seen = [obs]
    steps = 0
    terminated, truncated = False, False
    while not (terminated or truncated):
        action = env.action_space.sample() if policy is None else policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        seen.append(obs)
        steps += 1
        if limit is not None and steps >= limit:
            break
    env.close()
    return steps, bool(terminated), seen


def hold(policy, seed: int) -> int:
    """Exercise 1: the steps that a policy keeps the pole of CartPole up, the world seeded with `seed`."""
    return play("CartPole-v1", policy, seed)[0]


def many(name: str, policy, episodes: int = 100) -> dict:
    """A fixed policy over many episodes, seeds 0, 1, 2, ...: the mean, the fewest and the most steps, and how many
    episodes ended in the task's own end (in CartPole: the pole fell; in MountainCar: the car reached the flag)."""
    def make_():
        runs = [play(name, policy, s)[:2] for s in range(episodes)]
        steps = [r[0] for r in runs]
        return {"mean": float(np.mean(steps)), "min": int(min(steps)), "max": int(max(steps)),
                "ended": int(sum(r[1] for r in runs))}
    key = getattr(policy, "__name__", "random") if policy is not None else "random"
    return kept((LESSON, "many", name, key, episodes), make_)


# --- the Q-learner with bins ------------------------------------------------------------------------------------------------

class _Bad(Exception):
    """The learner's row_of gave a row that does not fit the table."""


def new_q(env, edges) -> np.ndarray:
    """A Q-table of zeros: a row for every combination of bins, a column per action."""
    rows = int(np.prod([len(e) + 1 for e in edges]))
    return np.zeros((rows, env.action_space.n))


def _row(fn, obs, edges, rows: int) -> int:
    r = fn(obs, edges)
    if isinstance(r, (bool, np.bool_)) or not isinstance(r, (int, np.integer)) or not 0 <= int(r) < rows:
        raise _Bad
    return int(r)


def q_episode(env, Q, rng, edges, fn=row_of) -> tuple:
    """One episode of Q-learning, as in the lessons before, with the row of every observation from `fn`. The number of
    steps, and whether the task ended."""
    obs, info = env.reset()
    s = _row(fn, obs, edges, len(Q))
    steps = 0
    terminated, truncated = False, False
    while not (terminated or truncated):
        if rng.random() < EPSILON:
            action = int(rng.integers(env.action_space.n))
        else:
            action = int(np.argmax(Q[s]))
        obs, reward, terminated, truncated, info = env.step(action)
        n = _row(fn, obs, edges, len(Q))
        aim = reward if terminated else reward + GAMMA * Q[n].max()
        Q[s, action] += ETA * (aim - Q[s, action])
        s = n
        steps += 1
    return steps, bool(terminated)


def train(name: str, b: int, seed: int = 0, fn=row_of, episodes: int = EPISODES, start: float = 0.0, show=None,
          limit: float | None = None, until_flag: bool = False) -> dict | str:
    """One run: a Q-table with every Q at `start` (0 unless said), the world seeded once with `seed` and the agent's
    random numbers from np.random.default_rng(seed), then `episodes` episodes. The steps of every episode, the episode
    in which the task first ended (MountainCar: the flag), the Q-table at every stop of looks() and at the end.
    `show(n, run)` hears every episode; "slow" after `limit` seconds; `until_flag` stops at the first flag."""
    env = make(name)
    edges = make_edges(name, b)
    Q = new_q(env, edges) + start
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    stops = set(looks())
    out = {"name": name, "bins": b, "seed": seed, "Q": Q, "steps": [], "first": None, "kept": {0: Q.copy()}}
    begin = time.perf_counter()
    for n in range(1, episodes + 1):
        steps, ended = q_episode(env, Q, rng, edges, fn)
        out["steps"].append(steps)
        if out["first"] is None and name == "MountainCar-v0" and ended:
            out["first"] = n
            if until_flag:
                break
        if n in stops:
            out["kept"][n] = Q.copy()
        if show is not None:
            show(n, out)
        if limit is not None and time.perf_counter() - begin > limit:
            return "slow"
    env.close()
    return out


def greedy(name: str, Q, b: int, fn=row_of) -> dict:
    """The policy of the largest Q, without random actions, over TEST episodes from seed TEST_SEED on: the mean steps,
    and in MountainCar how many episodes reached the flag."""
    env = make(name)
    edges = make_edges(name, b)
    obs, info = env.reset(seed=TEST_SEED)
    steps_all, flags = [], 0
    for e in range(TEST):
        if e:
            obs, info = env.reset()
        steps = 0
        terminated, truncated = False, False
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(int(np.argmax(Q[fn(obs, edges)])))
            steps += 1
        steps_all.append(steps)
        flags += bool(terminated)
    env.close()
    return {"mean": float(np.mean(steps_all)), "flags": flags}


def course_run(name: str, b: int, seed: int) -> dict:
    return kept((LESSON, "run", name, b, seed), lambda: train(name, b, seed))


def course_q(name: str, b: int) -> np.ndarray:
    """The NumPy cell: the Q-table of the course's run 1 at the end."""
    return course_run(name, b, 0)["Q"].copy()


def first_flag(b: int, seed: int, start: float = 0.0) -> int | None:
    """MountainCar: the episode of the first flag of a run, or None in all EPISODES."""
    return kept((LESSON, "first", b, seed, start), lambda: train("MountainCar-v0", b, seed, start=start, until_flag=True)["first"])


def demo_run(fn, name: str, b: int, seed: int, show=None, limit: float = 120.0) -> dict | str:
    """The demo: one run with the learner's row_of; "error" when it stops with an error or gives a row that does not
    fit the table, "slow" after `limit` seconds."""
    try:
        return train(name, b, seed, fn, show=show, limit=limit)
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the demo says what went wrong
        return "error"


def mean_last(steps: list, n: int, w: int = 100) -> float:
    part = steps[max(0, n - w):n]
    return sum(part) / len(part) if part else 0.0


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


# --- drawing: the two worlds ---------------------------------------------------------------------------------------------

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


def _figure(svgs: list, alt: str, width: int) -> str:
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(alt)}" style="margin:0.25rem 0 0.5rem;'
            f'max-width:min(100%,{width}px);display:flex;flex-wrap:wrap;gap:0.6rem 1.4rem;align-items:flex-start">'
            f'{"".join(svgs)}</figure>')


def _arrow(x0: float, y0: float, x1: float, y1: float, color: str = "var(--ea-ink-2)", width: float = 1.8) -> str:
    """A straight arrow from (x0, y0) to (x1, y1)."""
    d = math.hypot(x1 - x0, y1 - y0) or 1
    ux, uy = (x1 - x0) / d, (y1 - y0) / d
    bx, by = x1 - 7 * ux, y1 - 7 * uy
    return (f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{bx:.1f}" y2="{by:.1f}" style="stroke:{color};stroke-width:{width}"/>'
            f'<polygon points="{x1:.1f},{y1:.1f} {bx - 4 * uy:.1f},{by + 4 * ux:.1f} {bx + 4 * uy:.1f},{by - 4 * ux:.1f}" '
            f'style="fill:{color}"/>')


_WORLD_TEXT = {
    "en": {"cart": "CartPole-v1: 4 numbers, 2 actions", "car": "MountainCar-v0: 2 numbers, 3 actions",
           "angle": "angle", "left": "0: push left", "right": "1: push right", "edge": "edge",
           "flag": "flag at 0.5", "start": "start", "engine": "0 left, 1 nothing, 2 right",
           "alt": "The two worlds. CartPole: a cart on a track with a pole on top that leans a little to the right, with arrows for the two pushes, left and right, and the two edges of the track. MountainCar: a car at the bottom of a valley between two hills, with a flag on top of the hill on the right.",
           "note": "CartPole: +1 for every step the pole stays up; the episode ends when the pole leans more than 12° or the cart runs off the track, and a time limit cuts it off after 500 steps. MountainCar: −1 for every step until the car reaches the flag, which ends the episode; a time limit cuts it off after 200 steps. The engine is weaker than the pull of the slope."},
    "nl": {"cart": "CartPole-v1: 4 getallen, 2 actions", "car": "MountainCar-v0: 2 getallen, 3 actions",
           "angle": "hoek", "left": "0: duw naar links", "right": "1: duw naar rechts", "edge": "rand",
           "flag": "vlag bij 0,5", "start": "start", "engine": "0 links, 1 niets, 2 rechts",
           "alt": "De twee werelden. CartPole: een karretje op een baan met bovenop een paal die een beetje naar rechts helt, met pijlen voor de twee duwen, links en rechts, en de twee randen van de baan. MountainCar: een auto onder in een dal tussen twee heuvels, met een vlag boven op de heuvel rechts.",
           "note": "CartPole: +1 voor elke stap dat de paal blijft staan; de episode eindigt als de paal meer dan 12° helt of het karretje van de baan rijdt, en een tijdslimiet kapt de episode af na 500 stappen. MountainCar: −1 voor elke stap tot de auto de vlag bereikt, wat de episode beëindigt; een tijdslimiet kapt de episode af na 200 stappen. De motor is zwakker dan de helling."},
}


def _cart_x(x: float) -> float:
    return 30 + (x + 2.4) / 4.8 * 270


def cartpole_svg(title: str, lang: str, obs=None, words: bool = True, W: int = 330) -> str:
    """CartPole: the track with its two edges, the cart, and the pole at its angle; with `words`, the pushes too."""
    w = _WORLD_TEXT[lang]
    obs = [0.0, 0.0, 0.12, 0.0] if obs is None else obs
    H = TOP + 168
    track = TOP + 142
    cx = _cart_x(float(obs[0]))
    top = track - 34
    a = float(obs[2])
    parts = [_title(2, 14, title),
             f'<line x1="14" y1="{track}" x2="{W - 14}" y2="{track}" style="stroke:var(--ea-ink-3);stroke-width:2"/>']
    for x in (-2.4, 2.4):
        parts.append(f'<line x1="{_cart_x(x):.1f}" y1="{track - 8}" x2="{_cart_x(x):.1f}" y2="{track + 8}" '
                     f'style="stroke:var(--ea-ink-3);stroke-width:2"/>')
        if words:
            parts.append(_words(_cart_x(x), track + 21, f'{w["edge"]} {number(x, 1, lang)}', size=11))
    parts.append(f'<rect x="{cx - 26:.1f}" y="{top:.1f}" width="52" height="24" rx="4" '
                 f'style="fill:var(--ea-inset);stroke:var(--ea-ink-2);stroke-width:1.6"/>')
    for dx in (-15, 15):
        parts.append(f'<circle cx="{cx + dx:.1f}" cy="{track - 5}" r="5" style="fill:var(--ea-page);stroke:var(--ea-ink-2);stroke-width:1.6"/>')
    length = 88
    ex, ey = cx + length * math.sin(a), top - length * math.cos(a)
    if words:
        parts.append(f'<line x1="{cx:.1f}" y1="{top:.1f}" x2="{cx:.1f}" y2="{top - length:.1f}" '
                     f'style="stroke:var(--ea-axis);stroke-width:1.2;stroke-dasharray:4 3"/>')
        r = 46
        parts.append(f'<path d="M {cx:.1f} {top - r:.1f} A {r} {r} 0 0 1 {cx + r * math.sin(a):.1f} {top - r * math.cos(a):.1f}" '
                     f'style="fill:none;stroke:var(--ea-ink-3);stroke-width:1.2"/>')
        parts.append(_words(cx + 30, top - r - 6, w["angle"], anchor="start", size=11))
    parts.append(f'<line x1="{cx:.1f}" y1="{top:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
                 f'style="stroke:var(--ea-ink);stroke-width:6;stroke-linecap:round"/>')
    parts.append(f'<circle cx="{cx:.1f}" cy="{top:.1f}" r="3.5" style="fill:var(--ea-page);stroke:var(--ea-ink);stroke-width:1.4"/>')
    if words:
        y = top + 12
        parts += [_arrow(cx - 30, y, cx - 62, y), _words(cx - 66, y + 4, w["left"], anchor="end", size=11),
                  _arrow(cx + 30, y, cx + 62, y), _words(cx + 66, y + 4, w["right"], anchor="start", size=11)]
    return _svg(W, H, parts)


def _hill(p: float) -> float:
    """The height of the hills of MountainCar at position p, as Gymnasium draws them."""
    return math.sin(3 * p) * 0.45 + 0.55


def _car_x(p: float) -> float:
    return 22 + (p + 1.2) / 1.8 * 290


def _car_y(p: float) -> float:
    return TOP + 150 - (_hill(p) - 0.1) / 0.9 * 118


def mountain_svg(title: str, lang: str, pos: float = -0.5, words: bool = True, W: int = 330) -> str:
    """MountainCar: the valley between two hills, the flag at 0.5, and the car at `pos`."""
    w = _WORLD_TEXT[lang]
    H = TOP + 180
    ps = np.linspace(-1.2, 0.6, 91)
    line = " ".join(f"{_car_x(p):.1f},{_car_y(p):.1f}" for p in ps)
    ground = TOP + 158
    parts = [_title(2, 14, title),
             f'<polygon points="{_car_x(-1.2):.1f},{ground} {line} {_car_x(0.6):.1f},{ground}" style="fill:var(--ea-inset)"/>',
             f'<polyline points="{line}" style="fill:none;stroke:var(--ea-ink-3);stroke-width:2"/>']
    fx, fy = _car_x(0.5), _car_y(0.5)
    parts += [f'<line x1="{fx:.1f}" y1="{fy:.1f}" x2="{fx:.1f}" y2="{fy - 30:.1f}" style="stroke:var(--ea-ink);stroke-width:1.8"/>',
              f'<polygon points="{fx:.1f},{fy - 30:.1f} {fx + 15:.1f},{fy - 25:.1f} {fx:.1f},{fy - 20:.1f}" style="fill:var(--ea-ink)"/>']
    if words:
        parts.append(_words(fx - 6, fy - 36, w["flag"], anchor="end", size=11))
        x0, x1 = _car_x(-0.6), _car_x(-0.4)
        yb = ground + 7
        parts += [f'<line x1="{x0:.1f}" y1="{yb}" x2="{x1:.1f}" y2="{yb}" style="stroke:var(--ea-ink-3);stroke-width:1.4"/>',
                  f'<line x1="{x0:.1f}" y1="{yb - 4}" x2="{x0:.1f}" y2="{yb + 4}" style="stroke:var(--ea-ink-3);stroke-width:1.4"/>',
                  f'<line x1="{x1:.1f}" y1="{yb - 4}" x2="{x1:.1f}" y2="{yb + 4}" style="stroke:var(--ea-ink-3);stroke-width:1.4"/>',
                  _words((x0 + x1) / 2 + 30, yb + 4, w["start"], anchor="start", size=11)]
    cx, cy = _car_x(pos), _car_y(pos)
    slope = 1.35 * math.cos(3 * pos)   # the slope of the hill at pos, in the world's units
    deg = -math.degrees(math.atan(slope * (118 / 0.9) / (290 / 1.8)))
    parts.append(f'<g transform="translate({cx:.1f},{cy:.1f}) rotate({deg:.1f})">'
                 f'<rect x="-13" y="-15" width="26" height="10" rx="3" style="fill:var(--ea-inset);stroke:var(--ea-ink-2);stroke-width:1.6"/>'
                 f'<circle cx="-7" cy="-3.5" r="3.5" style="fill:var(--ea-page);stroke:var(--ea-ink-2);stroke-width:1.4"/>'
                 f'<circle cx="7" cy="-3.5" r="3.5" style="fill:var(--ea-page);stroke:var(--ea-ink-2);stroke-width:1.4"/></g>')
    if words:
        parts.append(_words(cx, cy - 48, w["engine"], size=11))
    return _svg(W, H, parts)


def world_picture(lang: str) -> str:
    """The guess: the two worlds side by side."""
    lang = lang if lang in LANGS else "en"
    w = _WORLD_TEXT[lang]
    return _figure([cartpole_svg(w["cart"], lang), mountain_svg(w["car"], lang)], w["alt"], 700) + _note(w["note"])


# --- drawing: bins --------------------------------------------------------------------------------------------------------

_BIN_TEXT = {
    "en": {"line": "Position −0.5, with 12 bins: 4 edges lie below it: bin 4",
           "grid": "4 bins each: row = 4 × position bin + velocity bin",
           "pos": "position bin", "vel": "velocity bin", "edge": "an edge",
           "alt": "Left: the position of MountainCar from −1.2 to 0.6, cut by 11 edges into 12 bins, numbered 0 to 11; the position −0.5 lies in bin 4, with 4 edges below it. Right: a grid of 4 position bins by 4 velocity bins, each cell with its row of the Q-table, 0 to 15, row by row; the cell of position bin 2 and velocity bin 1 is row 9."},
    "nl": {"line": "Positie −0,5, met 12 bins: 4 randen liggen eronder: bin 4",
           "grid": "4 bins per getal: rij = 4 × bin positie + bin snelheid",
           "pos": "bin positie", "vel": "bin snelheid", "edge": "een rand",
           "alt": "Links: de positie van MountainCar van −1,2 tot 0,6, door 11 randen verdeeld in 12 bins, genummerd 0 tot en met 11; de positie −0,5 ligt in bin 4, met 4 randen eronder. Rechts: een rooster van 4 bins voor de positie bij 4 bins voor de snelheid, in elke cel de rij van de Q-table, 0 tot en met 15, rij na rij; de cel van bin 2 van de positie en bin 1 van de snelheid is rij 9."},
}


def _line_svg(title: str, lang: str, x: float = -0.5, b: int = 12, rows=None, W: int = 330) -> str:
    """A number line of the position of MountainCar cut into b bins, with x marked; `rows` gives (x, bin, wrong) marks
    instead, for the picture of the learner's bin_of."""
    lo, hi = LIMITS["MountainCar-v0"][0]
    edges = np.linspace(lo, hi, b + 1)[1:-1]
    H = TOP + 96
    y = TOP + 50
    px = lambda v: 16 + (v - lo) / (hi - lo) * (W - 32)
    parts = [_title(2, 14, title),
             f'<line x1="{px(lo):.1f}" y1="{y}" x2="{px(hi):.1f}" y2="{y}" style="stroke:var(--ea-ink-3);stroke-width:2"/>']
    for v in (lo, hi):
        parts.append(f'<line x1="{px(v):.1f}" y1="{y - 5}" x2="{px(v):.1f}" y2="{y + 5}" style="stroke:var(--ea-ink-3);stroke-width:2"/>')
        parts.append(_words(px(v), y + 34, number(v, 1, lang), size=11))
    for e in edges:
        parts.append(f'<line x1="{px(e):.1f}" y1="{y - 12}" x2="{px(e):.1f}" y2="{y + 12}" style="stroke:var(--ea-ink-2);stroke-width:1.4"/>')
    bounds = [lo, *edges, hi]
    for k in range(b):
        parts.append(_words((px(bounds[k]) + px(bounds[k + 1])) / 2, y + 18, str(k), size=10.5, fill="var(--ea-ink-3)"))
    marks = [(x, bin_of(x, edges), False)] if rows is None else rows
    for v, k, wrong in marks:
        color = "var(--ea-loss-text)" if wrong else "var(--ea-ink)"
        parts.append(f'<circle cx="{px(v):.1f}" cy="{y}" r="4.5" style="fill:{color}"/>')
        parts.append(_words(px(v), y - 18, str(k), size=12, fill=color, weight=700))
    return _svg(W, H, parts)


def _grid_svg(title: str, lang: str, cells: dict | None = None, mark: tuple | None = (2, 1), W: int = 330) -> str:
    """A grid of 4 position bins (rows, from the bottom) by 4 velocity bins (columns), each cell with its row; `cells`
    gives (text, wrong) per (position bin, velocity bin) instead, for the picture of the learner's row_of."""
    w = _BIN_TEXT[lang]
    n, size = 4, 40
    left, top = 92, TOP + 16
    H = top + n * size + 46
    parts = [_title(2, 14, title)]
    for p in range(n):
        for v in range(n):
            x, y = left + v * size, top + (n - 1 - p) * size
            text_, wrong = (str(p * n + v), False) if cells is None else cells[(p, v)]
            on = mark is not None and (p, v) == tuple(mark)
            parts.append(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" style="fill:{"var(--ea-mid)" if on else "var(--ea-page)"};'
                         f'stroke:var(--ea-line);stroke-width:1"/>')
            color = "var(--ea-loss-text)" if wrong else "var(--ea-ink)"
            parts.append(_words(x + size / 2, y + size / 2 + 4.5, text_, size=13, fill=color, weight=700 if wrong or on else 400))
    for k in range(n):
        parts.append(_words(left - 8, top + (n - 1 - k) * size + size / 2 + 4, str(k), anchor="end", size=11))
        parts.append(_words(left + k * size + size / 2, top + n * size + 15, str(k), size=11))
    parts.append(_words(left - 30, top + n * size / 2, w["pos"], size=11).replace("<text ", f'<text transform="rotate(-90 {left - 30} {top + n * size / 2})" ', 1))
    parts.append(_words(left + n * size / 2, top + n * size + 33, w["vel"], size=11))
    return _svg(W, H, parts)


def bins_picture(lang: str) -> str:
    """The idea: a number cut into bins, and the bins of two numbers as the rows of a table."""
    lang = lang if lang in LANGS else "en"
    w = _BIN_TEXT[lang]
    return _figure([_line_svg(w["line"], lang), _grid_svg(w["grid"], lang)], w["alt"], 700)


# --- drawing: two benchmarks that a table cannot take ----------------------------------------------------------------------

_BENCH_TEXT = {
    "en": {"atari": "Atari: a screen of pixels", "atari2": "210 × 160 pixels, 128 colors", "atari3": "18 actions at most",
           "hopper": "MuJoCo, Hopper: a body", "hopper2": "11 numbers in", "hopper3": "3 numbers out, each from −1 to 1",
           "pixel": "one pixel",
           "alt": "Left: a game screen of Atari as a grid of pixels in shades of gray, with one pixel picked out; the agent sees 210 by 160 pixels and has at most 18 actions. Right: Hopper of MuJoCo, a body of four parts on one leg, with a curved arrow at each of its three joints: the agent sees 11 numbers and gives back 3, a turning force for each joint, each a real number from −1 to 1."},
    "nl": {"atari": "Atari: een scherm van pixels", "atari2": "210 × 160 pixels, 128 kleuren", "atari3": "hoogstens 18 actions",
           "hopper": "MuJoCo, Hopper: een lichaam", "hopper2": "11 getallen erin", "hopper3": "3 getallen eruit, elk van −1 tot 1",
           "pixel": "één pixel",
           "alt": "Links: een spelscherm van Atari als rooster van pixels in tinten grijs, met één pixel uitgelicht; de agent ziet 210 bij 160 pixels en heeft hoogstens 18 actions. Rechts: Hopper van MuJoCo, een lichaam van vier delen op één been, met een gebogen pijl bij elk van de drie gewrichten: de agent ziet 11 getallen en geeft er 3 terug, een draaiende kracht per gewricht, elk een reëel getal van −1 tot 1."},
}

_SHADES = ("var(--ea-page)", "var(--ea-inset)", "var(--ea-mid)", "var(--ea-line)", "var(--ea-axis)", "var(--ea-ink-3)")


def _atari_svg(lang: str, W: int = 330) -> str:
    w = _BENCH_TEXT[lang]
    cols, rows, size = 16, 21, 7.4   # a screen of 160 by 210, one square for every 10 by 10 pixels
    left, top = 18, TOP + 8
    rng = np.random.default_rng(3)
    shade = np.zeros((rows, cols), dtype=int)
    shade[2:4, 1:15] = 4                     # a band across the top
    shade[5:7, 1:15] = 3
    shade[rng.integers(8, 18, 6), rng.integers(1, 15, 6)] = 5   # a few things on the screen
    shade[19, 6:10] = 5                      # a bar near the bottom
    parts = [_title(2, 14, w["atari"])]
    for r in range(rows):
        for c in range(cols):
            parts.append(f'<rect x="{left + c * size:.1f}" y="{top + r * size:.1f}" width="{size:.1f}" height="{size:.1f}" '
                         f'style="fill:{_SHADES[shade[r, c]]};stroke:var(--ea-line);stroke-width:0.4"/>')
    pr, pc = 12, 11
    x, y = left + pc * size, top + pr * size
    parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{size:.1f}" height="{size:.1f}" style="fill:none;stroke:var(--ea-ink);stroke-width:1.6"/>')
    bx = left + cols * size + 26
    parts.append(f'<line x1="{x + size:.1f}" y1="{y + size / 2:.1f}" x2="{bx - 4:.1f}" y2="{y + size / 2:.1f}" style="stroke:var(--ea-ink-2);stroke-width:1.2"/>')
    parts.append(_words(bx, y + size / 2 + 4, w["pixel"], anchor="start"))
    parts.append(_words(bx, top + 40, w["atari2"], anchor="start", fill="var(--ea-ink)"))
    parts.append(_words(bx, top + 58, w["atari3"], anchor="start"))
    return _svg(W, top + rows * size + 8, parts)


def _torque(x: float, y: float, r: float = 11) -> str:
    """A curved arrow around a joint: a turning force."""
    a0, a1 = math.radians(200), math.radians(340)
    x0, y0 = x + r * math.cos(a0), y + r * math.sin(a0)
    x1, y1 = x + r * math.cos(a1), y + r * math.sin(a1)
    head = f'{x1 + 1:.1f},{y1 + 5:.1f} {x1 + 4.5:.1f},{y1 - 3:.1f} {x1 - 4:.1f},{y1 - 1:.1f}'
    return (f'<path d="M {x0:.1f} {y0:.1f} A {r} {r} 0 0 1 {x1:.1f} {y1:.1f}" style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.5"/>'
            f'<polygon points="{head}" style="fill:var(--ea-ink-2)"/>')


def _hopper_svg(lang: str, W: int = 330) -> str:
    w = _BENCH_TEXT[lang]
    ground = TOP + 160
    pts = [(78, TOP + 24), (84, TOP + 78), (66, TOP + 118), (80, ground - 4), (112, ground - 4)]   # torso top, hip, knee, ankle, toe
    widths = (13, 9, 7, 6)
    parts = [_title(2, 14, w["hopper"]),
             f'<line x1="14" y1="{ground}" x2="{W - 14}" y2="{ground}" style="stroke:var(--ea-ink-3);stroke-width:2"/>']
    for (x0, y0), (x1, y1), width in zip(pts, pts[1:], widths):
        parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" style="stroke:var(--ea-ink-2);stroke-width:{width};stroke-linecap:round"/>')
    for x, y in pts[1:4]:
        parts.append(f'<circle cx="{x}" cy="{y}" r="4.5" style="fill:var(--ea-page);stroke:var(--ea-ink);stroke-width:1.6"/>')
        parts.append(_torque(x, y, 14))
    parts.append(_words(150, TOP + 60, w["hopper2"], anchor="start", fill="var(--ea-ink)"))
    parts.append(_words(150, TOP + 80, w["hopper3"].split(", ")[0] + ",", anchor="start"))
    parts.append(_words(150, TOP + 96, w["hopper3"].split(", ")[1], anchor="start"))
    return _svg(W, ground + 8, parts)


def bench_picture(lang: str) -> str:
    """The idea: a screen of Atari and the Hopper of MuJoCo, with what goes in and comes out."""
    lang = lang if lang in LANGS else "en"
    return _figure([_atari_svg(lang), _hopper_svg(lang)], _BENCH_TEXT[lang]["alt"], 700)


# --- drawing: what the largest Q does in every bin ----------------------------------------------------------------------

_POLICY_TEXT = {
    "en": {"CartPole-v1": ("angle", "angular velocity"), "MountainCar-v0": ("position", "velocity"),
           "key_cart": "◀ push left, ▶ push right", "key_car": "◀ push left, ● nothing, ▶ push right; empty: all Q's still the same"},
    "nl": {"CartPole-v1": ("hoek", "hoeksnelheid"), "MountainCar-v0": ("positie", "snelheid"),
           "key_cart": "◀ duw naar links, ▶ duw naar rechts", "key_car": "◀ duw naar links, ● niets, ▶ duw naar rechts; leeg: alle Q's nog gelijk"},
}


def best_action(Q, r: int):
    """The action with the largest Q in row r, or None while every Q of the row is the same: nothing learned."""
    row = np.asarray(Q[r], dtype=float)
    if np.all(row == row[0]):
        return None
    return int(np.argmax(row))


def _mark(x: float, y: float, action: int, name: str, half: float) -> str:
    color = "var(--ea-param)"
    if name == "MountainCar-v0" and action == 1:
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{max(1.6, half * 0.35):.1f}" style="fill:{color}"/>'
    d = -1 if action == 0 else 1
    pts = [(x + d * half * 0.7, y), (x - d * half * 0.55, y - half * 0.6), (x - d * half * 0.55, y + half * 0.6)]
    return '<polygon points="' + " ".join(f"{px:.1f},{py:.1f}" for px, py in pts) + f'" style="fill:{color}"/>'


def policy_svg(name: str, Q, b: int, title: str, lang: str, W: int = 330) -> str:
    """The best action of every combination of the two numbers that count, b by b: the first number along the
    bottom, the second up the side."""
    w = _POLICY_TEXT[lang]
    side = 228
    cell = side / b
    left, top = 58, TOP + 6
    H = top + side + 74
    parts = [_title(2, 14, title),
             f'<rect x="{left}" y="{top}" width="{side}" height="{side}" style="fill:var(--ea-page);stroke:var(--ea-line);stroke-width:1"/>']
    for k in range(1, b):
        parts.append(f'<line x1="{left + k * cell:.1f}" y1="{top}" x2="{left + k * cell:.1f}" y2="{top + side}" style="stroke:var(--ea-line);stroke-width:0.6"/>')
        parts.append(f'<line x1="{left}" y1="{top + k * cell:.1f}" x2="{left + side}" y2="{top + k * cell:.1f}" style="stroke:var(--ea-line);stroke-width:0.6"/>')
    for i in range(b):
        for j in range(b):
            a = best_action(Q, i * b + j)
            if a is not None:
                parts.append(_mark(left + (i + 0.5) * cell, top + (b - j - 0.5) * cell, a, name, cell / 2))
    (lo0, hi0), (lo1, hi1) = LIMITS[name][-2:]
    for v, x in ((lo0, left), (hi0, left + side)):
        parts.append(_words(x, top + side + 14, number(v, 2, lang), size=10.5))
    for v, y in ((lo1, top + side), (hi1, top)):
        parts.append(_words(left - 5, y + 4, number(v, 2, lang), anchor="end", size=10.5))
    first, second = w[name]
    parts.append(_words(left + side / 2, top + side + 30, first, size=11))
    parts.append(_words(left - 38, top + side / 2, second, size=11).replace("<text ", f'<text transform="rotate(-90 {left - 38} {top + side / 2})" ', 1))
    for k, line in enumerate(w["key_cart" if name == "CartPole-v1" else "key_car"].split("; ")):   # the key, a line per part
        parts.append(_words(2, top + side + 50 + 16 * k, line, anchor="start", size=10.5))
    return _svg(W, H, parts)


# --- the demo: Q-learning with your bins ----------------------------------------------------------------------------------

WINDOW = 50   # the chart: the mean steps of the last this many episodes
STRIDE = 10   # the chart draws every this many episodes
RANGE = {"CartPole-v1": (0.0, 500.0), "MountainCar-v0": (0.0, 200.0)}
TICKS = [0, 500, 1000, 1500]   # the x-axis: a last tick at 2,000, on the right edge, would not fit

DEMO_TEXT = {
    "en": {"title": "Your agent: steps per episode", "x": "episodes", "y": "mean of the last {w}",
           "alt": "The mean number of steps of the last {w} episodes of your agent in {world}, after every episode so far.",
           "world": "World", "bins": "Bins", "run": "Run", "after": "after {n} episodes", "start": "before training",
           "cart": "With the largest Q and no random actions, the pole stays up for {m} steps on average, in {t} test episodes.",
           "car": "With the largest Q and no random actions, the car reaches the flag in {k} of {t} test episodes.",
           "car_steps": "With the largest Q and no random actions, the car reaches the flag in {k} of {t} test episodes, in {m} steps on average.",
           "first": "The first flag came in episode {f}.", "no_first": "No episode has reached the flag yet.",
           "say": "After {n} episodes, the mean of the last {w} episodes is {m} steps. {now}",
           "grid": "{world}, {b} bins: {when}"},
    "nl": {"title": "Je agent: stappen per episode", "x": "episodes", "y": "gemiddelde van de laatste {w}",
           "alt": "Het gemiddelde aantal stappen van de laatste {w} episodes van je agent in {world}, na elke episode tot nu toe.",
           "world": "Wereld", "bins": "Bins", "run": "Run", "after": "na {n} episodes", "start": "voor de training",
           "cart": "Met de grootste Q en zonder willekeurige actions blijft de paal gemiddeld {m} stappen staan, in {t} testepisodes.",
           "car": "Met de grootste Q en zonder willekeurige actions bereikt de auto de vlag in {k} van de {t} testepisodes.",
           "car_steps": "Met de grootste Q en zonder willekeurige actions bereikt de auto de vlag in {k} van de {t} testepisodes, in gemiddeld {m} stappen.",
           "first": "De eerste vlag kwam in episode {f}.", "no_first": "Nog geen episode heeft de vlag bereikt.",
           "say": "Na {n} episodes is het gemiddelde van de laatste {w} episodes {m} stappen. {now}",
           "grid": "{world}, {b} bins: {when}"},
}


def world_options(lang: str) -> dict:
    return {SHORT[name]: name for name in ENVS}


def bins_options(lang: str) -> dict:
    return {str(b): b for b in BINS}


def run_options(lang: str) -> dict:
    word = DEMO_TEXT[lang if lang in LANGS else "en"]["run"]
    return {f"{word} {k + 1}": k for k in range(RUNS)}


def chart_points(r: dict, start: int, end: int) -> list:
    """The points of the demo's chart for episodes start to end: the mean steps of the last WINDOW episodes, every
    STRIDE episodes and at the end."""
    out = []
    for e in range(max(1, start), end + 1):
        if e % STRIDE == 0 or e == end:
            out.append((e, mean_last(r["steps"], e, WINDOW)))
    return out


def now_text(r: dict, Q, lang: str, fn=row_of, first: bool = True) -> str:
    """What the largest Q does in the world now, in one or two sentences."""
    w = DEMO_TEXT[lang]
    name, b = r["name"], r["bins"]
    g = greedy(name, Q, b, fn)
    if name == "CartPole-v1":
        return w["cart"].format(m=number(g["mean"], 1, lang), t=TEST)
    say = (w["car_steps"] if g["flags"] else w["car"]).format(k=g["flags"], t=TEST, m=number(g["mean"], 1, lang))
    if first:
        say += " " + (w["first"].format(f=number(r["first"], 0, lang)) if r["first"] else w["no_first"])
    return say


def _when(n: int, lang: str) -> str:
    w = DEMO_TEXT[lang]
    return w["after"].format(n=number(n, 0, lang)) if n else w["start"]


def now_picture(r: dict, n: int, lang: str) -> str:
    """Beside the chart of the demo: the best action in every bin now."""
    lang = lang if lang in LANGS else "en"
    title = DEMO_TEXT[lang]["grid"].format(world=SHORT[r["name"]], b=r["bins"], when=_when(n, lang))
    return _figure([policy_svg(r["name"], r["Q"], r["bins"], title, lang)], title, 330)


def _curve_panel(r: dict, n: int, lang: str) -> chart.Panel:
    name = r["name"]
    w = DEMO_TEXT[lang]
    low, high = RANGE[name]
    p = chart.Panel((0, EPISODES), (low, high), title=w["title"], xlabel=w["x"], ylabel=w["y"].format(w=WINDOW), xticks=TICKS)
    points = chart_points(r, 1, len(r["steps"]))
    if points:
        p.line([x for x, _ in points], [y for _, y in points], "ink", 2.2)
    if 1 <= n <= len(r["steps"]):
        p.line([n, n], [low, high], "axis", 1.2)
        p.points([n], [mean_last(r["steps"], n, WINDOW)], "ink", r=3.6)
    return p


def rewind_frame(r: dict, n: int, lang: str, fn=row_of) -> tuple:
    """A stop of the rewind after the demo: the chart with a line at n, the best actions as the agent knew them then,
    and a sentence."""
    lang = lang if lang in LANGS else "en"
    w = DEMO_TEXT[lang]
    name = r["name"]
    n = min(r["kept"], key=lambda k: abs(k - int(n)))
    Q = r["kept"][n]
    svg = chart.figure([_curve_panel(r, n, lang)], alt=w["alt"].format(w=WINDOW, world=SHORT[name]), width=330, height=240)
    svg = chart.with_decimal_comma(svg) if lang == "nl" else svg
    title = w["grid"].format(world=SHORT[name], b=r["bins"], when=_when(n, lang))
    picture = _figure([policy_svg(name, Q, r["bins"], title, lang)], title, 330)
    now = now_text(r, Q, lang, fn, first=False)
    say = now if n == 0 else w["say"].format(n=number(n, 0, lang), w=WINDOW, m=number(mean_last(r["steps"], n, WINDOW), 1, lang), now=now)
    return live.rewound(svg, [picture], width=330), say


# --- what the check gives the learner's functions ------------------------------------------------------------------------

class NamedArray(np.ndarray):
    """An array of edges that shows its name in a check message, such as `pos_12`, instead of all its numbers. What
    is made from it, such as `edges < x` or a sum, is a plain array or number again."""

    def __new__(cls, values, name: str):
        obj = np.array(values, dtype=float).view(cls)
        obj.name = name
        return obj

    def __array_finalize__(self, obj):
        self.name = getattr(obj, "name", None)

    def __getitem__(self, key):
        return np.ndarray.__getitem__(self.view(np.ndarray), key)

    def __array_wrap__(self, array, context=None, return_scalar=False):
        out = np.asarray(array)
        return out[()] if out.ndim == 0 else out

    def __deepcopy__(self, memo):
        return NamedArray(np.array(self), self.name)

    def __reduce__(self):
        return (NamedArray, (np.array(self), self.name))

    def __repr__(self):
        return self.name if self.name else repr(np.array(self))


class NamedEdges(list):
    """The edges of every number of an observation, a list of arrays, that shows its name in a check message."""

    def __init__(self, arrays, name: str):
        super().__init__(arrays)
        self.name = name

    def __deepcopy__(self, memo):
        return NamedEdges([np.array(a) for a in self], self.name)

    def __reduce__(self):
        return (NamedEdges, ([np.array(a) for a in self], self.name))

    def __repr__(self):
        return self.name


def _edges(name: str, b: int, k: int, label: str) -> NamedArray:
    return NamedArray(make_edges(name, b)[k], label)


POS_12 = _edges("MountainCar-v0", 12, 0, "pos_12")      # the 11 edges of the position of MountainCar, 12 bins
ANGLE_12 = _edges("CartPole-v1", 12, 2, "angle_12")     # the 11 edges of the angle of CartPole, 12 bins
NO_EDGES = NamedArray([], "no_edges")                     # a number with one bin
CAR_12 = NamedEdges(make_edges("MountainCar-v0", 12), "car_12")
CAR_20 = NamedEdges(make_edges("MountainCar-v0", 20), "car_20")
CART_12 = NamedEdges(make_edges("CartPole-v1", 12), "cart_12")
CART_4 = NamedEdges(make_edges("CartPole-v1", 4), "cart_4")
DIGITS = NamedEdges([np.linspace(0, 1, 11)[1:-1] for _ in range(4)], "digits")   # four numbers from 0 to 1, 10 bins each

BIN_CASES = [(-0.5, POS_12), (-1.15, POS_12), (0.55, POS_12), (0.05, ANGLE_12), (0.3, NO_EDGES)]
ROW_CASES = [([-0.52, 0.004], CAR_12), ([0.01, -0.2, 0.05, -0.3], CART_12), ([0.35, 0.05, 0.75, 0.25], DIGITS),
             ([-0.9, -0.03], CAR_20), ([1.5, 0.4, -0.12, 0.9], CART_4)]
POLICY_CASES = [(0,), (1,), (2,), (3,), (4,)]


def _fine(x) -> bool:
    return isinstance(x, (int, float, np.floating, np.integer)) and not isinstance(x, (bool, np.bool_)) and math.isfinite(x)


def same_number(a, b) -> bool:
    return _fine(a) and _fine(b) and abs(float(a) - float(b)) < 1e-9


def same_hold(a, b) -> bool:
    """pole_policy: right when the pole stays up at least HOLD steps, whatever the course's own policy does."""
    return _fine(a) and float(a) >= HOLD


def show_value(v) -> dict:
    if _fine(v):
        shown = f"`{int(v)}`" if float(v).is_integer() else f"`{float(v)!r}`"
    else:
        text_ = repr(v)
        shown = f"`{text_ if len(text_) <= 80 else text_[:77] + '...'}`"
    return {"en": shown, "nl": shown}


def show_steps(v) -> dict:
    if _fine(v):
        return {"en": f"{int(v)} steps", "nl": f"{int(v)} stappen"}
    return show_value(v)


@dataclass
class PolicyBox(FunctionBox):
    """pole_policy in a code box: the check plays one episode of CartPole with it, seeded, and counts the steps."""

    def _run(self, fn, case):
        return hold(fn, case[0])

    def _call_text(self, case) -> str:
        return f"hold(pole_policy, {case[0]})"

    def _same_results(self, a: list, b: list) -> bool:   # a known wrong policy: the same steps in every episode
        return all(x[0] == y[0] and (x[0] == "error" or x[1] == y[1]) for x, y in zip(a, b))


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
    from everything_ai.check import Timeout, with_time_limit
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(make_, max(seconds, 0.05))
    except Timeout:
        return _SLOW
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


_YOURS_TEXT = {
    "en": {"pole_policy": {"title": "Your policy in CartPole",
                           "intro": "The angle of the pole at every step of the episode with seed 0: with your `pole_policy`, and with random pushes. The episode ends when the angle passes 12°, at the gray lines, when the cart runs off the track, or after 500 steps.",
                           "panel": "The angle of the pole", "x": "steps", "y": "angle (degrees)", "yours": "your policy", "random": "random",
                           "say": "With seeds 0 to 4, your policy keeps the pole up for {steps} steps; random pushes for {random}.",
                           "none": "Your `pole_policy` stops with an error, takes too long, or gives something other than 0 or 1, so the episode cannot be played yet.",
                           "alt": "The angle of the pole over the steps of one episode: with your policy and with random pushes, between two lines at 12° and −12°."},
           "bin_of": {"title": "Your bins",
                      "intro": "Your `bin_of` for seven positions of MountainCar, with 12 bins; in the color of mistakes, a bin that differs from the number of edges below the position.",
                      "line": "Position with 12 bins: your bin above each dot",
                      "none": "Your `bin_of` stops with an error or takes too long, so nothing can be drawn yet."},
           "row_of": {"title": "Your rows",
                      "intro": "Your `row_of` for the middle of each cell of MountainCar with 4 bins for each number; in the color of mistakes, a row that differs from the course's.",
                      "grid": "4 bins each: your row in each cell",
                      "none": "Your `row_of` stops with an error or takes too long, so nothing can be drawn yet."}},
    "nl": {"pole_policy": {"title": "Je policy in CartPole",
                           "intro": "De hoek van de paal bij elke stap van de episode met seed 0: met je `pole_policy`, en met willekeurige duwen. De episode eindigt als de hoek voorbij 12° gaat, bij de grijze lijnen, als het karretje van de baan rijdt, of na 500 stappen.",
                           "panel": "De hoek van de paal", "x": "stappen", "y": "hoek (graden)", "yours": "je policy", "random": "willekeurig",
                           "say": "Met seeds 0 tot en met 4 houdt je policy de paal {steps} stappen overeind; willekeurige duwen {random}.",
                           "none": "Je `pole_policy` stopt met een fout, duurt te lang, of geeft iets anders dan 0 of 1, dus de episode kan nog niet gespeeld worden.",
                           "alt": "De hoek van de paal over de stappen van één episode: met je policy en met willekeurige duwen, tussen twee lijnen bij 12° en −12°."},
           "bin_of": {"title": "Je bins",
                      "intro": "Je `bin_of` voor zeven posities van MountainCar, met 12 bins; in de kleur van fouten een bin die anders is dan het aantal randen onder de positie.",
                      "line": "Positie met 12 bins: je bin boven elke stip",
                      "none": "Je `bin_of` stopt met een fout of duurt te lang, dus er is nog geen afbeelding."},
           "row_of": {"title": "Je rijen",
                      "intro": "Je `row_of` voor het midden van elke cel van MountainCar met 4 bins per getal; in de kleur van fouten een rij die anders is dan bij de course.",
                      "grid": "4 bins per getal: je rij in elke cel",
                      "none": "Je `row_of` stopt met een fout of duurt te lang, dus er is nog geen afbeelding."}},
}


def _join(values: list, lang: str) -> str:
    parts = [number(v, 0, lang) for v in values]
    return ", ".join(parts[:-1]) + (" en " if lang == "nl" else " and ") + parts[-1]


def pole_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["pole_policy"]
    fn = _fn(seen)

    def make_():
        _, _, path = play("CartPole-v1", fn, 0)
        return path, [hold(fn, s) for s in range(5)]
    got = _call(make_) if fn is not None else None
    if not isinstance(got, tuple):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    path, steps = got
    _, _, rand = play("CartPole-v1", None, 0)
    deg = lambda seq: [math.degrees(float(o[2])) for o in seq]
    p = chart.Panel((0, 500), (-15, 15), title=w["panel"], xlabel=w["x"], ylabel=w["y"], xticks=[0, 100, 200, 300, 400])
    p.line([0, 500], [12, 12], "axis", 1.2).line([0, 500], [-12, -12], "axis", 1.2)
    p.line(list(range(len(rand))), deg(rand), "ink-3", 1.6)
    p.line(list(range(len(path))), deg(path), "ink", 2.2)
    # a key at the top right, above where most lines of the angle go
    p.line([352, 378], [9.6, 9.6], "ink", 2.2).text(384, 9.6, w["yours"], "ink", anchor="start", halo=True)
    p.line([352, 378], [6.0, 6.0], "ink-3", 1.6).text(384, 6.0, w["random"], "ink-3", anchor="start", halo=True)
    svg = chart.figure([p], alt=w["alt"], width=520, height=250)
    svg = chart.with_decimal_comma(svg) if lang == "nl" else svg
    say = w["say"].format(steps=_join(steps, lang), random=_join([hold(None, s) for s in range(5)], lang))
    return yours.figure(w["title"], _note(w["intro"]) + svg + _note(say), alt=w["title"])


PICTURE_POSITIONS = (-1.1, -0.82, -0.5, -0.31, 0.02, 0.33, 0.52)


def bin_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["bin_of"]
    fn = _fn(seen)
    edges = make_edges("MountainCar-v0", 12)[0]
    got = _call(lambda: [fn(x, NamedArray(edges, "pos_12")) for x in PICTURE_POSITIONS]) if fn is not None else None
    if not isinstance(got, list):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    marks = []
    for x, g in zip(PICTURE_POSITIONS, got):
        want = bin_of(x, edges)
        marks.append((x, str(int(g)) if _fine(g) and float(g).is_integer() else "?", not same_number(g, want)))
    return yours.figure(w["title"], _note(w["intro"]) + _figure([_line_svg(w["line"], lang, rows=marks, W=520)], w["line"], 520),
                        alt=w["title"])


def row_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["row_of"]
    fn = _fn(seen)
    edges = make_edges("MountainCar-v0", 4)
    (plo, phi), (vlo, vhi) = LIMITS["MountainCar-v0"]
    centers = {(p, v): [plo + (p + 0.5) * (phi - plo) / 4, vlo + (v + 0.5) * (vhi - vlo) / 4] for p in range(4) for v in range(4)}
    got = _call(lambda: {k: fn(list(c), NamedEdges(edges, "car_4")) for k, c in centers.items()}) if fn is not None else None
    if not isinstance(got, dict):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    cells = {}
    for k, g in got.items():
        want = row_of(centers[k], edges)
        cells[k] = (str(int(g)) if _fine(g) and float(g).is_integer() else "?", not same_number(g, want))
    return yours.figure(w["title"], _note(w["intro"]) + _figure([_grid_svg(w["grid"], lang, cells=cells, mark=None)], w["grid"], 330),
                        alt=w["title"])


# --- the exercises -----------------------------------------------------------------------------------------------------

REQUIRED = ["pole_policy", "bin_of", "row_of", "sparse"]


def _dots_left(got, text_) -> bool:
    """A `...` of the start code left in the box: an Ellipsis in the code, not three dots inside a string."""
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


def _not_an_action(e) -> bool:
    return isinstance(e, AssertionError) and "invalid" in str(e)


# exercise 1: pole_policy
def _pp(test: str = "obs[3] > 0") -> str:
    return ("def pole_policy(obs):\n"
            f"    if {test}:\n"
            "        return 1   # push the cart to the right\n"
            "    return 0       # push the cart to the left\n")


_PP_START = _pp("...")
_PP = _pp()
_PP_LEAN = _pp("obs[2] > 0")
_PP_AWAY = _pp("obs[3] < 0")


# exercise 2: bin_of
def _b(fill: str = "edges < x") -> str:
    return ("def bin_of(x, edges):\n"
            f"    return int(np.sum({fill}))\n")


_B_START = _b("...")
_B = _b()
_B_ABOVE = _b("edges > x")
_B_ONE_MORE = _B.replace("))\n", ")) + 1\n")
_B_ARGMAX = "def bin_of(x, edges):\n    return int(np.argmax(edges > x))\n"
B_NAMES = {"np": np}


# exercise 3: row_of
def _r(times: str = "(len(edges[i]) + 1)") -> str:
    return ("def row_of(obs, edges):\n"
            "    row = 0\n"
            "    for i in range(len(obs)):\n"
            f"        row = row * {times} + bin_of(obs[i], edges[i])\n"
            "    return row\n")


_R_START = _r("...")
_R = _r()
_R_ORDER = _r("len(edges[i]) + 1")
_R_EDGES = _r("len(edges[i])")
_R_TEN = _r("10")
_R_ADDED = _r("1")
R_NAMES = {"np": np, "bin_of": bin_of}


EXERCISES = {
    "pole_policy": PolicyBox(
        name="pole_policy",
        function="pole_policy",
        start=_PP_START,
        solution=_PP,
        compare=same_hold,
        show=show_steps,
        visible=POLICY_CASES[:1],
        hidden=POLICY_CASES[1:],
        first=[_DOTS],
        picture=pole_picture,
        mistakes=[
            Mistake("not-an-action", error=_not_an_action,
                    en="With `{call}`, CartPole gets an action it does not have: its actions are 0, push left, and 1, push right. Does every way through `pole_policy` end with `return 1` or `return 0`?",
                    nl="Met `{call}` krijgt CartPole een action die er niet is: de actions zijn 0, duw naar links, en 1, duw naar rechts. Eindigt elke weg door `pole_policy` met `return 1` of `return 0`?"),
            Mistake("lean", wrong=_PP_LEAN,
                    en="`{call}` gives {got}, and it should be at least 100: your policy pushes toward the side the pole leans, also while the pole is already swinging back, so every swing gets bigger than the one before. Which of the four numbers says which way the pole is falling right now?",
                    nl="`{call}` geeft {got}, en het moeten er minstens 100 zijn: je policy duwt naar de kant waar de paal heen helt, ook als de paal al terugzwaait, dus wordt elke zwaai groter dan de vorige. Welk van de vier getallen zegt welke kant de paal nu op valt?"),
            Mistake("away", wrong=_PP_AWAY,
                    en="`{call}` gives {got}, and it should be at least 100: your policy pushes the cart away from the side the pole falls to, so the pole falls even faster. Which way should the cart go to get under the pole again?",
                    nl="`{call}` geeft {got}, en het moeten er minstens 100 zijn: je policy duwt het karretje weg van de kant waar de paal heen valt, dus valt de paal nog sneller. Welke kant moet het karretje op om weer onder de paal te komen?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with a test that is true when the pole falls to the right.",
                      "nl": "Vervang de `...` door een test die waar is als de paal naar rechts valt."},
            "arguments": {"en": "The check calls your function with one observation, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(obs)`?",
                          "nl": "De check roept je functie aan met één observation, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(obs)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should be at least 100. Does your policy push the cart to the right when the pole falls to the right, and to the left when it falls to the left?",
                      "nl": "`{call}` geeft {got}, en het moeten er minstens 100 zijn. Duwt je policy het karretje naar rechts als de paal naar rechts valt, en naar links als de paal naar links valt?"},
            "almost": {"en": "Right with seed 0. But `{call}` gives {got}, and it should be at least 100. Does your policy look at the pole, whatever the seed?",
                       "nl": "Goed met seed 0. Maar `{call}` geeft {got}, en het moeten er minstens 100 zijn. Kijkt je policy naar de paal, wat de seed ook is?"},
            "correct": {"en": "Right in all {n} episodes: the pole stays up for at least 100 steps every time, while random pushes keep it up for 23 steps on average. In the picture, how does the angle move with your policy?",
                        "nl": "Goed in alle {n} episodes: de paal blijft elke keer minstens 100 stappen staan, terwijl willekeurige duwen de paal gemiddeld 23 stappen overeind houden. Hoe beweegt de hoek in de afbeelding met je policy?"},
        },
    ),
    "bin_of": FunctionBox(
        name="bin_of",
        function="bin_of",
        start=_B_START,
        solution=_B,
        names=B_NAMES,
        compare=same_number,
        show=show_value,
        visible=BIN_CASES[:1],
        hidden=BIN_CASES[1:],
        first=[_DOTS],
        picture=bin_picture,
        mistakes=[
            Mistake("above", wrong=_B_ABOVE,
                    en="`{call}` gives {got}, and it should give {want}: your comparison counts the edges above `x`, so the bins run the wrong way, from the top down. Which edges should the count take?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je vergelijking telt de randen boven `x`, dus lopen de bins de verkeerde kant op, van boven naar beneden. Welke randen moet de telling nemen?"),
            Mistake("one-more", wrong=_B_ONE_MORE,
                    en="`{call}` gives {got}, and it should give {want}: your bins start at 1. A row of a Q-table starts at 0: which bin should a number below every edge get?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je bins beginnen bij 1. Een rij van een Q-table begint bij 0: welke bin hoort bij een getal onder alle randen?"),
            Mistake("argmax", wrong=_B_ARGMAX,
                    en="`{call}` gives {got}, and it should give {want}: `np.argmax` gives the first edge above `x`, and when no edge lies above `x` it gives 0, the first place. What should a number above every edge get?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `np.argmax` geeft de eerste rand boven `x`, en als er geen rand boven `x` ligt, geeft `np.argmax` 0, de eerste plaats. Wat hoort er bij een getal boven alle randen?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with a comparison of `edges` and `x` whose `True`s are the edges below `x`.",
                      "nl": "Vervang de `...` door een vergelijking van `edges` en `x` waarvan de `True`'s de randen onder `x` zijn."},
            "arguments": {"en": "The check calls `{call}`, with a number and an array of edges, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(x, edges)`?",
                          "nl": "De check roept `{call}` aan, met een getal en een array van randen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(x, edges)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function count the edges that lie below `x`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Telt je functie de randen die onder `x` liggen?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for every `x` and every array of edges, also one without any edges?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie voor elke `x` en elke array van randen, ook een array zonder randen?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still start with `return`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Begint de regel nog met `return`?"},
            "correct": {"en": "Right in all {n} cases: below every edge is bin 0, above every edge is bin 11 of `pos_12`, and a number without edges always lands in bin 0. In the picture, which bin holds the position −0.5?",
                        "nl": "Goed in alle {n} gevallen: onder alle randen is bin 0, boven alle randen is bin 11 van `pos_12`, en een getal zonder randen komt altijd in bin 0. In welke bin ligt de positie −0,5 in de afbeelding?"},
        },
    ),
    "row_of": FunctionBox(
        name="row_of",
        function="row_of",
        start=_R_START,
        solution=_R,
        names=R_NAMES,
        compare=same_number,
        show=show_value,
        visible=ROW_CASES[:1],
        hidden=ROW_CASES[1:],
        first=[_DOTS],
        picture=row_picture,
        mistakes=[
            Mistake("order", wrong=_R_ORDER,
                    en="`{call}` gives {got}, and it should give {want}: without parentheses Python multiplies `row` by `len(edges[i])` first and adds the 1 after that. How do you make Python add the 1 before it multiplies?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: zonder haakjes vermenigvuldigt Python `row` eerst met `len(edges[i])` en telt pas daarna de 1 erbij op. Hoe laat je Python de 1 optellen voordat Python vermenigvuldigt?"),
            Mistake("edges", wrong=_R_EDGES,
                    en="`{call}` gives {got}, and it should give {want}: you multiply by the number of edges, and a number has one bin more than it has edges. By how much should `row` be multiplied?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je vermenigvuldigt met het aantal randen, en een getal heeft één bin meer dan randen. Met hoeveel moet `row` vermenigvuldigd worden?"),
            Mistake("ten", wrong=_R_TEN,
                    en="`{call}` gives {got}, and it should give {want}: with 10 the bins are digits, which works when every number has 10 bins, as in `digits`. With 12 bins for each number, by how much should `row` be multiplied?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: met 10 zijn de bins cijfers, en dat klopt als elk getal 10 bins heeft, zoals in `digits`. Met hoeveel moet `row` vermenigvuldigd worden met 12 bins per getal?"),
            Mistake("added", wrong=_R_ADDED,
                    en="`{call}` gives {got}, and it should give {want}: your rows add up the bins, so bins 1 and 2 land in the same row as bins 2 and 1. What should happen to `row` before the next bin is added?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je rijen tellen de bins op, dus komen bins 1 en 2 in dezelfde rij als bins 2 en 1. Wat moet er met `row` gebeuren voordat de volgende bin erbij komt?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the number that `row` is multiplied by: the bins of number `i`.",
                      "nl": "Vervang de `...` door het getal waarmee `row` vermenigvuldigd wordt: de bins van getal `i`."},
            "arguments": {"en": "The check calls `{call}`, with an observation and a list of edges, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(obs, edges)`?",
                          "nl": "De check roept `{call}` aan, met een observation en een lijst van randen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(obs, edges)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function multiply `row` by the number of bins of number `i` before it adds that number's bin?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Vermenigvuldigt je functie `row` met het aantal bins van getal `i` voordat de functie de bin van dat getal erbij optelt?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function take the number of bins from `edges`, so that it works for any number of bins?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en het moet {want} zijn. Haalt je functie het aantal bins uit `edges`, zodat de functie werkt voor elk aantal bins?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return row`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return row`?"},
            "correct": {"en": "Right in all {n} cases: with `digits`, bins 3, 0, 7 and 2 give row 3072, the bins as digits, and in CartPole the cart's two numbers, with one bin each, change nothing. In the picture, which row lies above row 2?",
                        "nl": "Goed in alle {n} gevallen: met `digits` geven bins 3, 0, 7 en 2 rij 3072, de bins als cijfers, en in CartPole veranderen de twee getallen van het karretje, met één bin elk, niets. Welke rij ligt in de afbeelding boven rij 2?"},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("pole_policy", _PP_START, "empty", None),
    ("pole_policy", _PP, "correct", None),
    ("pole_policy", _pp("obs[3] > ..."), "wrong", "dots-left"),
    ("pole_policy", _PP_LEAN, "wrong", "lean"),
    ("pole_policy", _PP_AWAY, "wrong", "away"),
    ("pole_policy", _PP.replace("        return 1   # push the cart to the right\n", "        return 2\n"), "error", "not-an-action"),
    ("pole_policy", _PP.replace("    return 0       # push the cart to the left\n", ""), "error", "not-an-action"),
    # other right answers: where it leans and where it falls together, and the same test the other way around
    ("pole_policy", _pp("3 * obs[2] + obs[3] > 0"), "correct", None),
    ("pole_policy", "def pole_policy(obs):\n    if obs[3] < 0:\n        return 0\n    return 1\n", "correct", None),
    # a policy that holds the pole up long enough with seed 0 only
    ("pole_policy", _pp("obs[3] > 0.04"), "almost", None),
    ("bin_of", _B_START, "empty", None),
    ("bin_of", _B, "correct", None),
    ("bin_of", _b("edges < ..."), "wrong", "dots-left"),
    ("bin_of", _B_ABOVE, "wrong", "above"),
    ("bin_of", _B_ONE_MORE, "wrong", "one-more"),
    ("bin_of", _B_ARGMAX, "wrong", "argmax"),
    ("bin_of", "def bin_of(x, edges):\n    int(np.sum(edges < x))\n", "wrong", "no-return"),
    # other right answers: the comparison the other way around, with <=, and a loop that counts
    ("bin_of", _b("x > edges"), "correct", None),
    ("bin_of", _b("edges <= x"), "correct", None),
    ("bin_of", "def bin_of(x, edges):\n    count = 0\n    for e in edges:\n        if e < x:\n            count += 1\n    return count\n", "correct", None),
    # the example's own x, whatever x is: right for the example only
    ("bin_of", _b("edges < -0.5"), "almost", None),
    ("row_of", _R_START, "empty", None),
    ("row_of", _R, "correct", None),
    ("row_of", _r("(len(edges[i]) + ...)"), "wrong", "dots-left"),
    ("row_of", _R_ORDER, "wrong", "order"),
    ("row_of", _R_EDGES, "wrong", "edges"),
    ("row_of", _R_TEN, "wrong", "ten"),
    ("row_of", _R_ADDED, "wrong", "added"),
    ("row_of", _R.replace("    return row\n", ""), "wrong", "no-return"),
    # other right answers: the bins first, and the size of the array of edges
    ("row_of", _r("(edges[i].size + 1)"), "correct", None),
    ("row_of", _R.replace("row = row * (len(edges[i]) + 1) +", "row = (len(edges[i]) + 1) * row +"), "correct", None),
    # 12 bins for every number: right for MountainCar and CartPole with 12 bins, not for digits
    ("row_of", _r("12"), "almost", None),
]
