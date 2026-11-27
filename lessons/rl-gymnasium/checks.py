# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rl-gymnasium: Gymnasium gives every environment the same interface, reset and step, and
spaces that say what observations and actions look like, so one Q-learner runs on FrozenLake, Taxi and CliffWalking
alike. The course's Q-learner and its runs, play_random, target and q_episode, the functions the learner writes, the
pictures of the three worlds, and the demo.

FrozenLake, Taxi and CliffWalking are environments of Gymnasium 1.4.0 (Farama Foundation, MIT); Taxi comes from
Dietterich (2000), CliffWalking from Sutton and Barto (2018, example 6.6). The code, the runs and the pictures are the
course's own."""
import ast
import copy
import html
import math
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from everything_ai import chart, live, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, kept

LESSON = "rl-gymnasium"
LANGS = ("en", "nl")
ENVS = ("FrozenLake-v1", "Taxi-v4", "CliffWalking-v1")
SHORT = {"FrozenLake-v1": "FrozenLake", "Taxi-v4": "Taxi", "CliffWalking-v1": "CliffWalking"}
ETA = 0.1           # the learning rate, the same in every world
GAMMA = 0.99        # the discount
EPSILON = 0.1       # the chance of a random action
EPISODES = {"FrozenLake-v1": 10_000, "Taxi-v4": 5_000, "CliffWalking-v1": 500}
RUNS = 10           # the course's runs in each world, seeds 0 to 9
FRACTIONS = (0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)   # the stops of the rewind, as parts of a world's episodes
WINDOW = {"FrozenLake-v1": 100, "Taxi-v4": 50, "CliffWalking-v1": 10}   # the chart: the mean of the last this many
LIMIT = 200         # the most steps a greedy test of CliffWalking takes: the world itself has no time limit


def make(name: str, **options):
    """A world of Gymnasium by its name."""
    return gym.make(name, **options)


def new_q(env) -> np.ndarray:
    """A Q-table of zeros: a row per observation, a column per action, as many as the spaces say."""
    return np.zeros((env.observation_space.n, env.action_space.n))


def looks(name: str) -> list:
    """The episodes after which the rewind shows the agent: 0, 1%, 2%, ... of all its episodes."""
    return [round(f * EPISODES[name]) for f in FRACTIONS]


# --- the functions of the exercises, as the course writes them ----------------------------------------------------------

def play_random(env, seed):
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    total, steps = 0, 0
    terminated, truncated = False, False
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
    return total, steps, terminated


def target(Q, reward, nxt, terminated, truncated, gamma):
    if terminated:
        return reward
    return reward + gamma * Q[nxt].max()


def q_episode(env, Q, rng):
    obs, info = env.reset()
    total = 0
    terminated, truncated = False, False
    while not (terminated or truncated):
        if rng.random() < EPSILON:
            action = rng.integers(env.action_space.n)
        else:
            action = np.argmax(Q[obs])
        nxt, reward, terminated, truncated, info = env.step(action)
        Q[obs, action] += ETA * (target(Q, reward, nxt, terminated, truncated, GAMMA) - Q[obs, action])
        obs = nxt
        total += reward
    return total


class _Bad(Exception):
    """The learner's function gave something that is not a number, or left a Q that is not a number."""


def _fine(x) -> bool:
    return isinstance(x, (int, float, np.floating, np.integer)) and not isinstance(x, (bool, np.bool_)) and math.isfinite(x)


def train(name: str, seed: int = 0, fn=q_episode, episodes: int | None = None, show=None,
          limit: float | None = None) -> dict | str:
    """One run of `fn` in one world: a Q-table of zeros as the spaces say, the world seeded once with `seed` and the
    random numbers of the agent from np.random.default_rng(seed), then `episodes` episodes. The return of every
    episode, the Q-table at every stop of looks(name), and the Q-table at the end. `show(n, run)` hears every episode;
    "slow" after `limit` seconds."""
    episodes = EPISODES[name] if episodes is None else episodes
    env = make(name)
    Q = new_q(env)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    stops = set(looks(name))
    out = {"name": name, "seed": seed, "Q": Q, "returns": [], "kept": {0: Q.copy()}}
    begin = time.perf_counter()
    for n in range(1, episodes + 1):
        total = fn(env, Q, rng)
        if not _fine(total) or not np.isfinite(Q).all():
            raise _Bad
        out["returns"].append(float(total))
        if n in stops:
            out["kept"][n] = Q.copy()
        if show is not None:
            show(n, out)
        if limit is not None and time.perf_counter() - begin > limit:
            return "slow"
    env.close()
    return out


# --- how good a Q-table is: the policy of the largest Q, worked out exactly ----------------------------------------------

def frozen_chance(Q, steps: int = 100) -> float:
    """The chance that the policy of the largest Q reaches G on FrozenLake within `steps` steps, from the start: the
    slips worked out from the world's own table of transitions, so no luck of the dice."""
    P = make("FrozenLake-v1").unwrapped.P
    policy = np.argmax(Q, axis=1)
    V = np.zeros(len(P))
    for _ in range(steps):
        V = np.array([sum(p * (r + (0.0 if done else V[s2])) for p, s2, r, done in P[s][int(policy[s])])
                      for s in range(len(P))])
    return float(V[0])


def frozen_best(steps: int = 100) -> float:
    """The best chance any policy has to reach G on FrozenLake within `steps` steps, worked out once."""
    def make_():
        P = make("FrozenLake-v1").unwrapped.P
        V = np.zeros(len(P))
        for _ in range(steps):
            V = np.array([max(sum(p * (r + (0.0 if done else V[s2])) for p, s2, r, done in P[s][a]) for a in range(4))
                          for s in range(len(P))])
        return float(V[0])
    return kept((LESSON, "frozen-best", steps), make_)


def taxi_starts() -> list:
    """The 300 states a Taxi episode can start in: the passenger neither in the taxi nor at the destination."""
    u = make("Taxi-v4").unwrapped
    return [s for s in range(500) if (lambda d: d[2] < 4 and d[2] != d[3])(list(u.decode(s)))]


def _greedy_walk(u, Q, s: int, limit: int) -> tuple:
    """The policy of the largest Q from state s in a world without luck: its return, its steps, and whether it got to
    an end within `limit` steps."""
    total, steps = 0, 0
    while steps < limit:
        ((p, s, r, done),) = u.P[s][int(np.argmax(Q[s]))]
        total += r
        steps += 1
        if done:
            return total, steps, True
    return total, steps, False


def taxi_greedy(Q) -> tuple:
    """The policy of the largest Q from each of the 300 starts of Taxi, at most 200 steps: the mean return, and how
    many of the 300 passengers it delivers."""
    u = make("Taxi-v4").unwrapped
    walks = [_greedy_walk(u, Q, s, 200) for s in taxi_starts()]
    return sum(w[0] for w in walks) / len(walks), sum(w[2] for w in walks)


def taxi_best() -> float:
    """The best mean return over the 300 starts of Taxi, without discount, worked out once."""
    def make_():
        u = make("Taxi-v4").unwrapped
        V = np.zeros(500)
        for _ in range(100):
            V = np.array([max(sum(p * (r + (0.0 if done else V[s2])) for p, s2, r, done in u.P[s][a]) for a in range(6))
                          for s in range(500)])
        return float(np.mean(V[taxi_starts()]))
    return kept((LESSON, "taxi-best"), make_)


def cliff_greedy(Q) -> tuple:
    """The policy of the largest Q on CliffWalking from the start: its return and its steps, or (None, LIMIT) when it
    does not reach G within LIMIT steps."""
    u = make("CliffWalking-v1").unwrapped
    total, steps, done = _greedy_walk(u, Q, 36, LIMIT)
    return (total, steps) if done else (None, LIMIT)


# --- the course's runs ---------------------------------------------------------------------------------------------------

def course_run(name: str, seed: int) -> dict:
    """The course's run k = seed + 1 in one world, made once per Python process."""
    return kept((LESSON, "run", name, seed), lambda: train(name, seed))


def demo_run(fn, name: str, seed: int, show=None, limit: float = 120.0) -> dict | str:
    """The demo: one run of the learner's q_episode in one world; "error" when it stops with an error or gives
    something that is not a number, "slow" after `limit` seconds."""
    try:
        return train(name, seed, fn, show=show, limit=limit)
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the demo says which function to look at
        return "error"


def course_q(name: str) -> np.ndarray:
    """The NumPy cell: the Q-table of the course's run 4 at the end, in one world."""
    return course_run(name, 3)["Q"].copy()


def course_results() -> dict:
    """What the guess and the texts say about the course's 10 runs in each world: for FrozenLake the chance to reach G
    within 100 steps with the largest Q, for Taxi the mean return and the passengers delivered from the 300 starts,
    for CliffWalking the steps of the way of the largest Q."""
    def make_():
        out = {}
        for name in ENVS:
            rows = []
            for seed in range(RUNS):
                Q = course_run(name, seed)["Q"]
                if name == "FrozenLake-v1":
                    rows.append(frozen_chance(Q))
                elif name == "Taxi-v4":
                    rows.append(taxi_greedy(Q))
                else:
                    rows.append(cliff_greedy(Q))
            out[name] = rows
        return out
    return kept((LESSON, "results"), make_)


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


def percent(x: float, lang: str, digits: int = 1) -> str:
    return number(100 * x, digits, lang) + "%"


# --- drawing the three worlds --------------------------------------------------------------------------------------------

FROZEN_MAP = ["SFFF", "FHFH", "FFFH", "HFFG"]
FROZEN_ENDS = frozenset(r * 4 + c for r, row in enumerate(FROZEN_MAP) for c, x in enumerate(row) if x in "HG")
# what each action number does in each world: a step (rows, columns), or a word for Taxi's pickup and drop-off
MOVES = {"FrozenLake-v1": {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)},
         "CliffWalking-v1": {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)},
         "Taxi-v4": {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}}
CLIFF_ROWS, CLIFF_COLS = 4, 12
CLIFF = frozenset(3 * 12 + c for c in range(1, 11))
CLIFF_START, CLIFF_GOAL = 36, 47
TAXI_LOCS = [(0, 0), (0, 4), (4, 0), (4, 3)]   # R, G, Y, B
TAXI_LETTERS = "RGYB"
# the walls of Taxi, read from its map: a wall between (row, col) and (row, col + 1)
_TAXI_MAP = ["|R: | : :G|", "| : | : : |", "| : : : : |", "| | : | : |", "|Y| : |B: |"]
TAXI_WALLS = frozenset((r, c) for r, line in enumerate(_TAXI_MAP) for c in range(4) if line[2 + 2 * c] == "|")
TAXI_CASE = (2, 2, 2, 1)   # the case the pictures follow: the taxi in the middle, the passenger at Y, to G
_HALO = "paint-order:stroke;stroke:var(--ea-page);stroke-width:3px;stroke-linejoin:round"


def taxi_state(row: int, col: int, passenger: int, destination: int) -> int:
    return ((row * 5 + col) * 5 + passenger) * 4 + destination


def best_actions(Q, s: int) -> list:
    """The actions with the largest Q in state s, or none while every Q of the state is the same: nothing learned."""
    row = np.asarray(Q[s], dtype=float)
    if np.all(row == row[0]):
        return []
    return [int(a) for a in np.flatnonzero(row == row.max())]


def _triangle(mx: float, my: float, dr: int, dc: int, half: float) -> str:
    tip = (mx + dc * (half - 2), my + dr * (half - 2))
    base = (mx + dc * (half - 7.5), my + dr * (half - 7.5))
    side = (dr * 3.6, dc * 3.6)
    corners = [tip, (base[0] + side[0], base[1] + side[1]), (base[0] - side[0], base[1] - side[1])]
    return ('<polygon points="' + " ".join(f"{px:.1f},{py:.1f}" for px, py in corners)
            + '" style="fill:var(--ea-ink-2)"/>')


TOP = 38   # the room above a world for its title: a name, and a line under it


def _title(x: float, y: float, words: str) -> str:
    """A title of one or two lines: the part before ": " in bold, the rest in grey under it."""
    name, _, rest = words.partition(": ")
    out = f'<text x="{x}" y="{y}" style="font-size:13.5px;font-weight:600;fill:var(--ea-ink)">{html.escape(name)}</text>'
    if rest:
        out += (f'<text x="{x}" y="{y + 16}" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(rest)}</text>')
    return out


def _letter(x: float, y: float, sign: str, fill: str = "var(--ea-ink-2)", size: float = 13, halo: bool = True) -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" style="font-size:{size}px;font-weight:700;fill:{fill};text-anchor:middle;'
            f'{_HALO if halo else ""}">{html.escape(sign)}</text>')


def _rect(x, y, w, h, fill="var(--ea-page)") -> str:
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h}" '
            f'style="fill:{fill};stroke:var(--ea-line);stroke-width:1"/>')


def _svg(width: float, height: float, parts: list) -> str:
    return (f'<svg viewBox="0 0 {width:.0f} {height:.0f}" aria-hidden="true" style="flex:0 1 auto;width:100%;'
            f'max-width:{width:.0f}px;height:auto;font-family:var(--ea-font-ui)">{"".join(parts)}</svg>')


def _figure(svgs: list, alt: str, width: int) -> str:
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(alt)}" style="margin:0.25rem 0 0.5rem;'
            f'max-width:min(100%,{width}px);display:flex;flex-wrap:wrap;gap:0.6rem 1.4rem;align-items:flex-start">'
            f'{"".join(svgs)}</figure>')


def frozen_svg(title: str, Q=None, cell: int = 36) -> str:
    """FrozenLake: S, the holes H in the color of the cliff, G; with a Q-table, the best actions as small triangles."""
    W, top = 4 * cell + 2, TOP
    parts = [_title(1, 14, title)]
    for s in range(16):
        r, c = divmod(s, 4)
        x, y = 1 + c * cell, top + r * cell
        kind = FROZEN_MAP[r][c]
        fill = {"H": "var(--ea-data)", "G": "var(--ea-data-wash)"}.get(kind, "var(--ea-page)")
        parts.append(_rect(x, y, cell, cell, fill))
        if kind in "SHG":
            parts.append(_letter(x + cell / 2, y + cell / 2 + 4.5, kind,
                                 "var(--ea-page)" if kind == "H" else "var(--ea-ink-2)", halo=kind != "H"))
        if Q is not None and s not in FROZEN_ENDS:
            for a in best_actions(Q, s):
                dr, dc = MOVES["FrozenLake-v1"][a]
                parts.append(_triangle(x + cell / 2, y + cell / 2, dr, dc, cell / 2))
    return _svg(W, top + 4 * cell + 2, parts)


def cliff_way(Q) -> list | None:
    """The cells the largest Q leads through on CliffWalking, from S to G, or None when it does not get there."""
    if Q is None:
        return None
    u = make("CliffWalking-v1").unwrapped
    s, cells = CLIFF_START, [CLIFF_START]
    for _ in range(LIMIT):
        acts = best_actions(Q, s)
        if not acts:
            return None
        ((p, s, r, done),) = u.P[s][acts[0]]
        if r == -100:
            return None
        cells.append(s)
        if done:
            return cells
    return None


_CLIFF_WORDS = {"en": "cliff: −100, back to S", "nl": "klif: −100, terug naar S"}


def cliff_svg(title: str, lang: str, Q=None, cell: int = 26) -> str:
    """CliffWalking: the cliff as one band with its words, S and G; with a Q-table, the best actions as small triangles
    and, once they lead from S to G, that way as an orange line."""
    W, top = CLIFF_COLS * cell + 2, TOP
    parts = [_title(1, 14, title)]
    for s in range(48):
        r, c = divmod(s, CLIFF_COLS)
        if s in CLIFF:
            continue
        parts.append(_rect(1 + c * cell, top + r * cell, cell, cell,
                           "var(--ea-data-wash)" if s == CLIFF_GOAL else "var(--ea-page)"))
    x0, x1, yb = 1 + cell, 1 + (CLIFF_COLS - 1) * cell, top + 3 * cell
    parts.append(_rect(x0, yb, x1 - x0, cell, "var(--ea-data)"))
    parts.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{yb + cell / 2 + 4:.1f}" style="font-size:11.5px;font-weight:600;'
                 f'fill:var(--ea-page);text-anchor:middle">{html.escape(_CLIFF_WORDS[lang])}</text>')
    for s, sign in ((CLIFF_START, "S"), (CLIFF_GOAL, "G")):
        r, c = divmod(s, CLIFF_COLS)
        parts.append(_letter(1 + c * cell + cell / 2, top + r * cell + cell / 2 + 4.5, sign))
    way = cliff_way(Q)
    if way:
        points = " ".join(f"{1 + (s % 12) * cell + cell / 2:.1f},{top + (s // 12) * cell + cell / 2:.1f}" for s in way)
        parts.append(f'<polyline points="{points}" style="fill:none;stroke:var(--ea-param);stroke-width:3;'
                     f'stroke-linecap:round;stroke-linejoin:round;opacity:0.8"/>')
    if Q is not None:
        for s in range(48):
            if s in CLIFF or s in (CLIFF_START, CLIFF_GOAL):
                continue
            r, c = divmod(s, CLIFF_COLS)
            for a in best_actions(Q, s):
                dr, dc = MOVES["CliffWalking-v1"][a]
                parts.append(_triangle(1 + c * cell + cell / 2, top + r * cell + cell / 2, dr, dc, cell / 2))
    return _svg(W, top + 4 * cell + 2, parts)


def taxi_walk(Q, steps: int = 40) -> dict:
    """The largest Q from TAXI_CASE: the cells of the taxi, where it picked the passenger up and dropped the
    passenger off, and whether it delivered within `steps` steps."""
    u = make("Taxi-v4").unwrapped
    s = taxi_state(*TAXI_CASE)
    cells, pick, drop = [TAXI_CASE[:2]], None, None
    for _ in range(steps):
        a = int(np.argmax(Q[s]))
        ((p, nxt, r, done),) = u.P[s][a]
        row, col, passenger, _ = u.decode(s)
        if a == 4 and passenger < 4 and r != -10:
            pick = (row, col)
        s = nxt
        cells.append(tuple(list(u.decode(s))[:2]))
        if done:
            drop = cells[-1]
            break
    return {"cells": cells, "pick": pick, "drop": drop}


def taxi_svg(title: str, Q=None, cell: int = 36) -> str:
    """Taxi: the 5 by 5 streets with their walls, the four stops R, G, Y and B; with a Q-table, where the largest Q
    takes the taxi from the middle with the passenger at Y and G as the destination, as an orange line, with P where
    it picks the passenger up and D where it drops the passenger off."""
    W, top = 5 * cell + 2, TOP
    parts = [_title(1, 14, title)]
    for r in range(5):
        for c in range(5):
            stop = (r, c) in TAXI_LOCS
            parts.append(_rect(1 + c * cell, top + r * cell, cell, cell, "var(--ea-inset)" if stop else "var(--ea-page)"))
    for r, c in TAXI_WALLS:
        x = 1 + (c + 1) * cell
        parts.append(f'<line x1="{x}" y1="{top + r * cell}" x2="{x}" y2="{top + (r + 1) * cell}" '
                     f'style="stroke:var(--ea-ink);stroke-width:4;stroke-linecap:round"/>')
    parts.append(f'<rect x="1" y="{top}" width="{5 * cell}" height="{5 * cell}" style="fill:none;stroke:var(--ea-ink);'
                 f'stroke-width:2"/>')
    for (r, c), letter in zip(TAXI_LOCS, TAXI_LETTERS):
        parts.append(f'<text x="{1 + c * cell + 4}" y="{top + r * cell + 13}" style="font-size:12px;font-weight:700;'
                     f'fill:var(--ea-ink-2)">{letter}</text>')
    if Q is not None:
        walk = taxi_walk(Q)
        seen = []
        for cell_ in walk["cells"]:
            if not seen or seen[-1] != cell_:
                seen.append(cell_)
        if len(seen) > 1:
            points = " ".join(f"{1 + c * cell + cell / 2:.1f},{top + r * cell + cell / 2:.1f}" for r, c in seen)
            parts.append(f'<polyline points="{points}" style="fill:none;stroke:var(--ea-param);stroke-width:3;'
                         f'stroke-linecap:round;stroke-linejoin:round;opacity:0.8"/>')
        r, c = TAXI_CASE[:2]
        parts.append(f'<circle cx="{1 + c * cell + cell / 2:.1f}" cy="{top + r * cell + cell / 2:.1f}" r="4.5" '
                     f'style="fill:var(--ea-param)"/>')
        for where, sign in ((walk["pick"], "P"), (walk["drop"], "D")):
            if where is not None:
                r, c = where
                parts.append(_letter(1 + c * cell + cell / 2, top + r * cell + cell - 6, sign, "var(--ea-param-text)", 12))
    return _svg(W, top + 5 * cell + 2, parts)


_WORLD_TEXT = {
    "en": {"FrozenLake-v1": "FrozenLake: 16 states, 4 actions", "Taxi-v4": "Taxi: 500 states, 6 actions",
           "CliffWalking-v1": "CliffWalking: 48 states, 4 actions",
           "alt": "The three worlds. FrozenLake: a grid of 4 by 4 with the start S top left, four holes H and the goal G bottom right. Taxi: streets of 5 by 5 with walls and four stops, R, G, Y and B. CliffWalking: a grid of 4 by 12 with S bottom left, G bottom right, and the cliff between them.",
           "note": "FrozenLake: the ice is slippery, so a step goes the way the agent asks in one of three times, and otherwise to one side; G gives 1, a hole ends the episode with 0. Taxi: pick the passenger up at one stop and drop the passenger off at another, +20; every step costs 1, a wrong pickup or drop-off 10. CliffWalking: every step costs 1, a step into the cliff 100, and back to S."},
    "nl": {"FrozenLake-v1": "FrozenLake: 16 states, 4 actions", "Taxi-v4": "Taxi: 500 states, 6 actions",
           "CliffWalking-v1": "CliffWalking: 48 states, 4 actions",
           "alt": "De drie werelden. FrozenLake: een rooster van 4 bij 4 met de start S linksboven, vier wakken H en het doel G rechtsonder. Taxi: straten van 5 bij 5 met muren en vier haltes, R, G, Y en B. CliffWalking: een rooster van 4 bij 12 met S linksonder, G rechtsonder, en het klif daartussen.",
           "note": "FrozenLake: het ijs is glad, dus gaat een stap één op de drie keer naar de gevraagde kant, en anders opzij; G geeft 1, een wak beëindigt de episode met 0. Taxi: haal de passagier op bij de ene halte en zet de passagier af bij een andere, +20; elke stap kost 1, een verkeerde ophaal- of afzetpoging 10. CliffWalking: elke stap kost 1, een stap in het klif 100, en terug naar S."},
}


def world_picture(lang: str) -> str:
    """The guess: the three worlds side by side, without an agent."""
    lang = lang if lang in LANGS else "en"
    w = _WORLD_TEXT[lang]
    svgs = [frozen_svg(w["FrozenLake-v1"]), taxi_svg(w["Taxi-v4"]), cliff_svg(w["CliffWalking-v1"], lang)]
    return _figure(svgs, w["alt"], 700) + _note(w["note"])


def world_svg(name: str, title: str, lang: str, Q=None) -> str:
    if name == "FrozenLake-v1":
        return frozen_svg(title, Q)
    if name == "Taxi-v4":
        return taxi_svg(title, Q)
    return cliff_svg(title, lang, Q)


# --- the picture of The idea: one interface between any agent and any world --------------------------------------------

_IDEA_TEXT = {
    "en": {"agent": "agent", "q": "Q-table, ε-greedy", "env": "environment", "worlds": "FrozenLake, Taxi,", "worlds2": "CliffWalking, …",
           "start": "at the start of an episode", "step": "at every step",
           "alt": "The agent on the left and the environment on the right, with four arrows between them. At the start of an episode the agent calls env.reset(seed=0) and gets obs, info back. At every step the agent calls env.step(action) and gets obs, reward, terminated, truncated, info back."},
    "nl": {"agent": "agent", "q": "Q-table, ε-greedy", "env": "environment", "worlds": "FrozenLake, Taxi,", "worlds2": "CliffWalking, …",
           "start": "aan het begin van een episode", "step": "bij elke stap",
           "alt": "De agent links en de environment rechts, met vier pijlen ertussen. Aan het begin van een episode roept de agent env.reset(seed=0) aan en krijgt obs, info terug. Bij elke stap roept de agent env.step(action) aan en krijgt obs, reward, terminated, truncated, info terug."},
}


def _arrow(x0: float, x1: float, y: float) -> str:
    head = 7 if x1 > x0 else -7
    return (f'<line x1="{x0}" y1="{y}" x2="{x1 - head}" y2="{y}" style="stroke:var(--ea-ink-2);stroke-width:1.6"/>'
            f'<polygon points="{x1},{y} {x1 - head},{y - 4} {x1 - head},{y + 4}" style="fill:var(--ea-ink-2)"/>')


def _code(x: float, y: float, words: str) -> str:
    return (f'<text x="{x}" y="{y}" style="font-family:var(--ea-font-code);font-size:12px;fill:var(--ea-ink);'
            f'text-anchor:middle">{html.escape(words)}</text>')


def idea_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang]
    W, H, bw = 580, 176, 128
    left, right, mid = bw, W - bw, W / 2
    parts = []
    for x, name_, line1, line2, fill in ((0, w["agent"], w["q"], "", "var(--ea-param-text)"),
                                         (W - bw, w["env"], w["worlds"], w["worlds2"], "var(--ea-ink-2)")):
        parts.append(f'<rect x="{x + 1}" y="20" width="{bw - 2}" height="{H - 40}" rx="6" style="fill:var(--ea-inset);'
                     f'stroke:var(--ea-line);stroke-width:1"/>')
        parts.append(f'<text x="{x + bw / 2}" y="{H / 2 - 6}" style="font-size:14px;font-weight:600;fill:var(--ea-ink);'
                     f'text-anchor:middle">{html.escape(name_)}</text>')
        parts.append(f'<text x="{x + bw / 2}" y="{H / 2 + 12}" style="font-size:11.5px;fill:{fill};text-anchor:middle">'
                     f'{html.escape(line1)}</text>')
        if line2:
            parts.append(f'<text x="{x + bw / 2}" y="{H / 2 + 26}" style="font-size:11.5px;fill:{fill};'
                         f'text-anchor:middle">{html.escape(line2)}</text>')
    for y, label in ((14, w["start"]), (98, w["step"])):
        parts.append(f'<text x="{mid}" y="{y}" style="font-size:11.5px;fill:var(--ea-ink-2);text-anchor:middle;'
                     f'font-style:italic">{html.escape(label)}</text>')
    parts += [_code(mid, 32, "env.reset(seed=0)"), _arrow(left + 6, right - 6, 38),
              _code(mid, 62, "obs, info"), _arrow(right - 6, left + 6, 68),
              _code(mid, 116, "env.step(action)"), _arrow(left + 6, right - 6, 122),
              _code(mid, 146, "obs, reward, terminated, truncated, info"), _arrow(right - 6, left + 6, 152)]
    svg = (f'<svg viewBox="0 0 {W} {H}" aria-hidden="true" style="width:100%;max-width:{W}px;height:auto;'
           f'font-family:var(--ea-font-ui)">{"".join(parts)}</svg>')
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(w["alt"])}" style="margin:0.25rem 0 0.5rem;'
            f'max-width:min(100%,{W}px)">{svg}</figure>')


# --- the demo: your q_episode in a world of your choice -----------------------------------------------------------------

RANGE = {"FrozenLake-v1": (0.0, 1.0), "Taxi-v4": (-800.0, 20.0), "CliffWalking-v1": (-200.0, 0.0)}
STRIDE = {"FrozenLake-v1": 50, "Taxi-v4": 25, "CliffWalking-v1": 1}   # the chart draws every this many episodes
TICKS = {"FrozenLake-v1": [0, 2000, 4000, 6000, 8000], "Taxi-v4": [0, 1000, 2000, 3000, 4000],
         "CliffWalking-v1": [0, 100, 200, 300, 400, 500]}   # the x-axis of the rewind: a last tick of 5 digits would not fit

DEMO_TEXT = {
    "en": {"title": "Your agent: mean return", "x": "episodes", "y": "mean of the last {w}",
           "alt": "The mean return of the last {w} episodes of your agent in {world}, after every episode so far.",
           "world": "World", "run": "Run", "after": "after {n} episodes", "start": "before training",
           "frozen": "With the largest Q, the agent reaches G in {p} of the episodes; the best any policy can do within 100 steps is {best}.",
           "frozen0": "Every Q is still the same, so the agent has no best action anywhere yet.",
           "taxi": "From the middle, with the passenger at Y and G as the destination, the largest Q delivers the passenger in {k} steps. From all 300 starts it delivers {d} passengers, with a mean return of {m}; the best possible is {best}.",
           "taxi_no": "From the middle, with the passenger at Y and G as the destination, the largest Q does not deliver the passenger yet. From all 300 starts it delivers {d} passengers.",
           "cliff": "The largest Q leads from S to G in {k} steps.",
           "cliff_no": "The largest Q does not lead from S to G yet.",
           "say": "After {n} episodes, the mean return of the last {w} is {m}. {now}"},
    "nl": {"title": "Je agent: gemiddelde return", "x": "episodes", "y": "gemiddelde van de laatste {w}",
           "alt": "De gemiddelde return van de laatste {w} episodes van je agent in {world}, na elke episode tot nu toe.",
           "world": "Wereld", "run": "Run", "after": "na {n} episodes", "start": "voor de training",
           "frozen": "Met de grootste Q bereikt de agent G in {p} van de episodes; het beste wat een policy binnen 100 stappen kan, is {best}.",
           "frozen0": "Elke Q is nog gelijk, dus heeft de agent nog nergens een beste action.",
           "taxi": "Vanuit het midden, met de passagier bij Y en G als bestemming, brengt de grootste Q de passagier weg in {k} stappen. Vanuit alle 300 starts brengt de grootste Q {d} passagiers weg, met een gemiddelde return van {m}; het best mogelijke is {best}.",
           "taxi_no": "Vanuit het midden, met de passagier bij Y en G als bestemming, brengt de grootste Q de passagier nog niet weg. Vanuit alle 300 starts brengt de grootste Q {d} passagiers weg.",
           "cliff": "De grootste Q leidt in {k} stappen van S naar G.",
           "cliff_no": "De grootste Q leidt nog niet van S naar G.",
           "say": "Na {n} episodes is de gemiddelde return van de laatste {w} {m}. {now}"},
}


def world_options(lang: str) -> dict:
    return {SHORT[name]: name for name in ENVS}


def run_options(lang: str) -> dict:
    word = DEMO_TEXT[lang if lang in LANGS else "en"]["run"]
    return {f"{word} {k + 1}": k for k in range(RUNS)}


def mean_last(returns: list, n: int, w: int) -> float:
    part = returns[max(0, n - w):n]
    return sum(part) / len(part) if part else 0.0


def chart_points(r: dict, start: int, end: int) -> list:
    """The points of the demo's chart for episodes start to end: the mean return of the last WINDOW episodes, within
    the range of the chart, every STRIDE episodes and at the end."""
    name = r["name"]
    low, high = RANGE[name]
    step = STRIDE[name]
    out = []
    for e in range(max(1, start), end + 1):
        if e % step == 0 or e == end:
            out.append((e, min(high, max(low, mean_last(r["returns"], e, WINDOW[name])))))
    return out


def now_text(name: str, Q, lang: str) -> str:
    """What the largest Q does in the world now, in one or two sentences."""
    w = DEMO_TEXT[lang]
    if name == "FrozenLake-v1":
        if not any(best_actions(Q, s) for s in range(16)):
            return w["frozen0"]
        return w["frozen"].format(p=percent(frozen_chance(Q), lang), best=percent(frozen_best(), lang))
    if name == "Taxi-v4":
        walk = taxi_walk(Q)
        m, d = taxi_greedy(Q)
        if walk["drop"] is None:
            return w["taxi_no"].format(d=d)
        return w["taxi"].format(k=len(walk["cells"]) - 1, d=d, m=number(m, 2, lang), best=number(taxi_best(), 2, lang))
    way = cliff_way(Q)
    return w["cliff"].format(k=len(way) - 1) if way else w["cliff_no"]


def now_picture(r: dict, n: int, lang: str) -> str:
    """Beside the chart of the demo: the world with what the largest Q does in it now."""
    lang = lang if lang in LANGS else "en"
    w = DEMO_TEXT[lang]
    title = f'{SHORT[r["name"]]}: {w["after"].format(n=number(n, 0, lang)) if n else w["start"]}'
    return _figure([world_svg(r["name"], title, lang, r["Q"])], title, 330)


def _curve_panel(r: dict, n: int, lang: str) -> chart.Panel:
    name = r["name"]
    w = DEMO_TEXT[lang]
    low, high = RANGE[name]
    p = chart.Panel((0, EPISODES[name]), (low, high), title=w["title"], xlabel=w["x"], ylabel=w["y"].format(w=WINDOW[name]),
                    xticks=TICKS[name])
    points = chart_points(r, 1, len(r["returns"]))
    if points:
        p.line([x for x, _ in points], [y for _, y in points], "ink", 2.2)
    if 1 <= n <= len(r["returns"]):
        p.line([n, n], [low, high], "axis", 1.2)
        p.points([n], [min(high, max(low, mean_last(r["returns"], n, WINDOW[name])))], "ink", r=3.6)
    return p


def rewind_frame(r: dict, n: int, lang: str) -> tuple:
    """A stop of the rewind after the demo: the chart of the run with a line at n, the world as the agent knew it
    then, and a sentence."""
    lang = lang if lang in LANGS else "en"
    w = DEMO_TEXT[lang]
    name = r["name"]
    n = min(r["kept"], key=lambda k: abs(k - int(n)))
    Q = r["kept"][n]
    svg = chart.figure([_curve_panel(r, n, lang)], alt=w["alt"].format(w=WINDOW[name], world=SHORT[name]), width=330, height=240)
    svg = chart.with_decimal_comma(svg) if lang == "nl" else svg
    title = f'{SHORT[name]}: {w["after"].format(n=number(n, 0, lang)) if n else w["start"]}'
    picture = _figure([world_svg(name, title, lang, Q)], title, 330)
    if n == 0:
        say = now_text(name, Q, lang)
    else:
        say = w["say"].format(n=number(n, 0, lang), w=WINDOW[name], m=number(mean_last(r["returns"], n, WINDOW[name]), 2, lang),
                              now=now_text(name, Q, lang))
    return live.rewound(svg, [picture], width=330), say


# --- what the check gives the learner's functions ------------------------------------------------------------------------

class NamedTable(np.ndarray):
    """A Q-table that shows its name in a check message, such as `Q_ones`, instead of all its numbers. A part of it,
    such as one row, is a plain array again."""

    def __new__(cls, values, name: str):
        obj = np.array(values, dtype=float).view(cls)
        obj.name = name
        return obj

    def __array_finalize__(self, obj):
        self.name = None

    def __getitem__(self, key):   # a row or a number of the table, as a plain array or number
        return np.ndarray.__getitem__(self.view(np.ndarray), key)

    def __array_wrap__(self, array, context=None, return_scalar=False):   # Q.max(), Q + 1: a plain number or array
        out = np.asarray(array)
        return out[()] if out.ndim == 0 else out

    def __deepcopy__(self, memo):
        return NamedTable(np.array(self), self.name)

    def __reduce__(self):
        return (NamedTable, (np.array(self), self.name))

    def __repr__(self):
        return self.name if self.name else repr(np.array(self))


def q_taxi() -> NamedTable:
    """The Q-table of the course's run 4 of Taxi at the end."""
    return NamedTable(course_q("Taxi-v4"), "Q_taxi")


Q_ONES = NamedTable(np.ones((16, 4)), "Q_ones")   # FrozenLake, every Q starting at 1: optimistic, as in an earlier lesson
TAXI_MIDDLE = taxi_state(*TAXI_CASE)   # 249: the taxi in the middle, the passenger at Y, the destination G
TAXI_LOADED = taxi_state(2, 2, 4, 1)   # 257: the taxi in the middle with the passenger in it, the destination G


def kept_table(name: str, n: int) -> NamedTable:
    """The Q-table of the course's run 4 in one world after n episodes, one of the stops of looks(name)."""
    short = {"FrozenLake-v1": "frozen", "Taxi-v4": "taxi", "CliffWalking-v1": "cliff"}[name]
    return NamedTable(course_run(name, 3)["kept"][n], f"Q_{short}_{n}")


def zeros_table(name: str) -> NamedTable:
    env = make(name)
    return NamedTable(new_q(env), "Q_zeros")


@dataclass(frozen=True)
class Episode:
    """What q_episode did with one case: the return it gave back, the Q-table it left, and the first Q where that
    table differs from the right one (state, action, got, want)."""
    total: object
    values: tuple
    off: tuple | None


def _q_sample(env, Q, rng):
    """q_episode with env.action_space.sample() for the random action: also right, with the action space's own
    random numbers, which the check seeds."""
    obs, info = env.reset()
    total = 0
    terminated, truncated = False, False
    while not (terminated or truncated):
        if rng.random() < EPSILON:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[obs])
        nxt, reward, terminated, truncated, info = env.step(action)
        Q[obs, action] += ETA * (target(Q, reward, nxt, terminated, truncated, GAMMA) - Q[obs, action])
        obs = nxt
        total += reward
    return total


def _play(fn, name: str, table, seed: int):
    """One episode of fn in a world seeded with `seed`, from a copy of `table`: the return and the table it leaves."""
    env = make(name)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    Q = np.array(table, dtype=float)
    total = fn(env, Q, np.random.default_rng(seed))
    return total, Q


def judge_episode(name: str, table, seed: int, fn) -> Episode | None:
    total, Q = _play(fn, name, table, seed)
    if total is None:
        return None
    want_total, want_Q = _play(q_episode, name, table, seed)
    good = _fine(total) and isinstance(Q, np.ndarray) and Q.shape == want_Q.shape and np.isfinite(Q).all()
    if good and float(total) == float(want_total) and np.allclose(Q, want_Q, atol=1e-9):
        return Episode(float(want_total), tuple(np.round(want_Q, 6).ravel()), None)
    if good and fn is not _q_sample:   # the random action from the action space: right as well
        other_total, other_Q = _play(_q_sample, name, table, seed)
        if float(total) == float(other_total) and np.allclose(Q, other_Q, atol=1e-9):
            return Episode(float(want_total), tuple(np.round(want_Q, 6).ravel()), None)
    if not good:
        return Episode(total if not _fine(total) else float(total), (), None)
    off = None
    diff = np.argwhere(~np.isclose(Q, want_Q, atol=1e-9))
    if len(diff):
        s, a = (int(x) for x in diff[0])
        off = (s, a, float(Q[s, a]), float(want_Q[s, a]))
    return Episode(float(total), tuple(np.round(Q, 6).ravel()), off if float(total) == float(want_total) else None)


def same_episode(a, b) -> bool:
    if not isinstance(a, Episode) or not isinstance(b, Episode):
        return a == b
    return _fine(a.total) and _fine(b.total) and float(a.total) == float(b.total) and a.values == b.values


def same_play(a, b) -> bool:
    """play_random: the same return, the same number of steps, and the same way of stopping."""
    try:
        return (len(a) == 3 and _fine(a[0]) and _fine(a[1]) and isinstance(a[2], (bool, np.bool_))
                and float(a[0]) == float(b[0]) and int(a[1]) == int(b[1]) and bool(a[2]) == bool(b[2]))
    except TypeError:
        return False


def same_number(a, b) -> bool:
    return _fine(a) and _fine(b) and abs(float(a) - float(b)) < 1e-9


def _code_number(v) -> str:
    v = round(float(v), 4)
    return str(int(v)) if v.is_integer() else repr(v)


def show_value(v) -> dict:
    if isinstance(v, tuple) and len(v) == 3 and all(_fine(x) or isinstance(x, (bool, np.bool_)) for x in v):
        shown = "`(" + ", ".join(repr(bool(x)) if isinstance(x, (bool, np.bool_)) else _code_number(x) for x in v) + ")`"
    elif _fine(v):
        shown = f"`{_code_number(v)}`"
    else:
        text_ = repr(v)
        shown = f"`{text_ if len(text_) <= 80 else text_[:77] + '...'}`"
    return {"en": shown, "nl": shown}


def show_episode(got) -> dict:
    if not isinstance(got, Episode):
        return show_value(got)
    total = _code_number(got.total) if _fine(got.total) else repr(got.total)
    if got.off is None:
        return {"en": f"the return `{total}`", "nl": f"de return `{total}`"}
    s, a, v, want = got.off
    return {"en": f"the return `{total}`, but leaves `Q[{s}, {a}]` at {number(v, 3, 'en')}, where it should be {number(want, 3, 'en')}",
            "nl": f"de return `{total}`, maar laat `Q[{s}, {a}]` achter op {number(v, 3, 'nl')}, waar {number(want, 3, 'nl')} hoort"}


@dataclass
class PlayBox(FunctionBox):
    """play_random in a code box: the check makes the world by its name and calls play_random(env, seed)."""

    def _run(self, fn, case):
        name, seed = case
        return fn(make(name), seed)

    def _call_text(self, case) -> str:
        return f'play_random(gym.make("{case[0]}"), {case[1]})'


@dataclass
class EpisodeBox(FunctionBox):
    """q_episode in a code box: the check makes the world, seeds it with env.reset(seed=seed), gives a copy of a
    Q-table and a new random generator with the same seed, and judges the return and the Q-table it leaves."""

    def _run(self, fn, case):
        return judge_episode(case[0], case[1], case[2], fn)

    def _call_text(self, case) -> str:
        return f'q_episode(gym.make("{case[0]}"), {case[1]!r}, np.random.default_rng({case[2]}))'


PLAY_CASES = [("FrozenLake-v1", 0), ("Taxi-v4", 0), ("Taxi-v4", 27), ("CliffWalking-v1", 0), ("FrozenLake-v1", 115)]


def target_cases() -> list:
    Q = q_taxi()
    return [
        (Q, -1, TAXI_MIDDLE, False, False, 0.99),   # a step like any other
        (Q, -1, TAXI_MIDDLE, False, True, 0.99),    # the 200th step: the clock runs out
        (Q_ONES, 0, 7, True, False, 0.99),          # into the hole at 7, from a table that starts at 1
        (Q_ONES, 1, 15, True, True, 0.99),          # onto G at the 100th step: both at once
        (Q, -1, TAXI_LOADED, False, False, 0.9),     # the passenger in the taxi, in the middle
    ]


def episode_cases() -> list:
    return [
        ("FrozenLake-v1", kept_table("FrozenLake-v1", 2000), 1),
        ("Taxi-v4", zeros_table("Taxi-v4"), 2),
        ("Taxi-v4", kept_table("Taxi-v4", 250), 3),
        ("CliffWalking-v1", zeros_table("CliffWalking-v1"), 4),
        ("CliffWalking-v1", kept_table("CliffWalking-v1", 50), 5),
    ]


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


def _table(head: list, rows: list) -> str:
    """A small table: rows of cells, each (text, wrong); a wrong cell in the color of mistakes."""
    th = "".join(f'<th style="text-align:left;padding:0.15rem 0.7rem 0.15rem 0;font-weight:600">{html.escape(h)}</th>'
                 for h in head)
    body = []
    for row in rows:
        cells = []
        for text_, wrong in row:
            color = "var(--ea-loss-text)" if wrong else "var(--ea-ink)"
            weight = 600 if wrong else 400
            cells.append(f'<td style="padding:0.15rem 0.7rem 0.15rem 0;color:{color};font-weight:{weight}">{_marked(text_)}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<table style="border-collapse:collapse;font-family:var(--ea-font-ui);font-size:0.85rem;'
            f'font-variant-numeric:tabular-nums;margin:0.3rem 0"><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


_YOURS_TEXT = {
    "en": {"play_random": {"title": "Your random agent",
                           "head": ["World", "seed", "return", "steps", "how it stopped"],
                           "intro": "Your `play_random` in each world with seeds 0, 1 and 2; in the color of mistakes, what differs from the course's.",
                           "ended": "ended", "cut": "cut off", "none": "Your `play_random` stops with an error, takes too long, or gives something other than a return, a number of steps and `True` or `False`, so the table cannot be drawn yet."},
           "target": {"title": "Your targets",
                      "head": ["the step", "terminated", "truncated", "your target", "right"],
                      "intro": "Your `target` for four steps; in the color of mistakes, a target that differs from the right one.",
                      "rows": ["Taxi: a step like any other, to the middle", "Taxi: the 200th step, to the middle",
                               "FrozenLake: into the hole at 7, from `Q_ones`", "FrozenLake: onto G at the 100th step, from `Q_ones`"],
                      "none": "Your `target` stops with an error or takes too long, so the table cannot be drawn yet."},
           "q_episode": {"title": "Your agent in two worlds",
                         "intro": "Run 1 with your `q_episode`: 500 episodes of CliffWalking and 2,000 of Taxi, each from a table of zeros.",
                         "cliff": "CliffWalking: after 500 episodes", "taxi": "Taxi: after 2,000 episodes",
                         "say": "On CliffWalking, {cliff} In Taxi, {taxi}",
                         "way": "the largest Q leads from S to G in {k} steps.", "no_way": "the largest Q does not lead from S to G yet.",
                         "drop": "the largest Q takes the passenger from Y to G in {k} steps, from the middle.",
                         "no_drop": "the largest Q does not deliver the passenger from the middle yet.",
                         "none": "Your `q_episode` stops with an error, takes too long, or leaves a Q that is not a number, so nothing can be drawn yet.",
                         "alt": "CliffWalking and Taxi after your agent learned them: the best actions on the cliff and its way from S to G, and the way of the taxi from the middle to Y and on to G."}},
    "nl": {"play_random": {"title": "Je willekeurige agent",
                           "head": ["Wereld", "seed", "return", "stappen", "hoe de episode stopte"],
                           "intro": "Je `play_random` in elke wereld met seeds 0, 1 en 2; in de kleur van fouten wat anders is dan bij de course.",
                           "ended": "geëindigd", "cut": "afgekapt", "none": "Je `play_random` stopt met een fout, duurt te lang, of geeft iets anders dan een return, een aantal stappen en `True` of `False`, dus er is nog geen tabel."},
           "target": {"title": "Je targets",
                      "head": ["de stap", "terminated", "truncated", "je target", "goed"],
                      "intro": "Je `target` voor vier stappen; in de kleur van fouten een target dat anders is dan het goede.",
                      "rows": ["Taxi: een gewone stap, naar het midden", "Taxi: de 200e stap, naar het midden",
                               "FrozenLake: het wak bij 7 in, vanuit `Q_ones`", "FrozenLake: op G bij de 100e stap, vanuit `Q_ones`"],
                      "none": "Je `target` stopt met een fout of duurt te lang, dus er is nog geen tabel."},
           "q_episode": {"title": "Je agent in twee werelden",
                         "intro": "Run 1 met je `q_episode`: 500 episodes CliffWalking en 2.000 Taxi, elk vanuit een tabel met nullen.",
                         "cliff": "CliffWalking: na 500 episodes", "taxi": "Taxi: na 2.000 episodes",
                         "say": "Op CliffWalking {cliff} In Taxi {taxi}",
                         "way": "leidt de grootste Q in {k} stappen van S naar G.", "no_way": "leidt de grootste Q nog niet van S naar G.",
                         "drop": "brengt de grootste Q de passagier vanuit het midden in {k} stappen van Y naar G.",
                         "no_drop": "brengt de grootste Q de passagier vanuit het midden nog niet weg.",
                         "none": "Je `q_episode` stopt met een fout, duurt te lang, of laat een Q achter die geen getal is, dus er is nog geen afbeelding.",
                         "alt": "CliffWalking en Taxi nadat je agent deze werelden leerde: de beste actions op het klif en de weg van S naar G, en de weg van de taxi vanuit het midden naar Y en verder naar G."}},
}


def play_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["play_random"]
    fn = _fn(seen)
    cases = [(name, seed) for name in ENVS for seed in range(3)]
    got = None
    if fn is not None:
        got = _call(lambda: [fn(make(name), seed) for name, seed in cases])
    if not isinstance(got, list) or not all(isinstance(g, tuple) and len(g) == 3 for g in got):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    rows = []
    for (name, seed), g in zip(cases, got):
        want = play_random(make(name), seed)
        try:
            ret = number(float(g[0]), 0, lang)
            steps = number(int(g[1]), 0, lang)
        except (TypeError, ValueError):
            ret, steps = repr(g[0]), repr(g[1])
        how = w["ended"] if g[2] is True or g[2] is np.True_ else (w["cut"] if g[2] is False or g[2] is np.False_ else repr(g[2]))
        rows.append([(SHORT[name], False), (str(seed), False), (ret, not same_number(g[0], want[0])),
                     (steps, not same_number(g[1], want[1])), (how, not isinstance(g[2], (bool, np.bool_)) or bool(g[2]) != want[2])])
    return yours.figure(w["title"], _note(w["intro"]) + _table(w["head"], rows), alt=w["title"])


def target_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["target"]
    fn = _fn(seen)
    cases = target_cases()[:4]
    got = _call(lambda: [fn(copy.deepcopy(c[0]), *c[1:]) for c in cases]) if fn is not None else None
    if not isinstance(got, list):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    rows = []
    for label, c, g in zip(w["rows"], cases, got):
        want = target(*c)
        shown = number(float(g), 2, lang) if _fine(g) else repr(g)
        rows.append([(label, False), (str(c[3]), False), (str(c[4]), False), (shown, not same_number(g, want)),
                     (number(want, 2, lang), False)])
    return yours.figure(w["title"], _note(w["intro"]) + _table(w["head"], rows), alt=w["title"])


def episode_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["q_episode"]
    fn = _fn(seen)
    got = None
    if fn is not None:
        got = _call(lambda: (train("CliffWalking-v1", 0, fn, episodes=500)["Q"], train("Taxi-v4", 0, fn, episodes=2000)["Q"]))
    if not isinstance(got, tuple):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    cliff_q, taxi_q = got
    way, walk = cliff_way(cliff_q), taxi_walk(taxi_q)
    say = w["say"].format(cliff=w["way"].format(k=len(way) - 1) if way else w["no_way"],
                          taxi=w["drop"].format(k=len(walk["cells"]) - 1) if walk["drop"] else w["no_drop"])
    body = (_note(w["intro"]) + _figure([cliff_svg(w["cliff"], lang, cliff_q), taxi_svg(w["taxi"], taxi_q)], w["alt"], 560)
            + _note(say))
    return yours.figure(w["title"], body, alt=w["title"])


# --- the exercises -----------------------------------------------------------------------------------------------------

REQUIRED = ["play_random", "target", "q_episode", "cut_off"]


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


# exercise 1: play_random
def _p(test: str = "terminated or truncated", step: str = "env.step(action)",
       names: str = "obs, reward, terminated, truncated, info") -> str:
    return ("def play_random(env, seed):\n"
            "    obs, info = env.reset(seed=seed)\n"
            "    env.action_space.seed(seed)\n"
            "    total, steps = 0, 0\n"
            "    terminated, truncated = False, False\n"
            f"    while not ({test}):\n"
            "        action = env.action_space.sample()\n"
            f"        {names} = {step}\n"
            "        total += reward\n"
            "        steps += 1\n"
            "    return total, steps, terminated\n")


_P_START = _p("...", "...")
_P = _p()
_P_NO_CUT = _p("terminated")
_P_AND = _p("terminated and truncated")
_P_SWAPPED = _p(names="obs, reward, truncated, terminated, info")


# exercise 2: target
def _t(test: str = "terminated", back: str = "Q[nxt].max()") -> str:
    return ("def target(Q, reward, nxt, terminated, truncated, gamma):\n"
            f"    if {test}:\n"
            "        return reward\n"
            f"    return reward + gamma * {back}\n")


_T_START = _t("...", "...")
_T = _t()
_T_CUT_ENDS = _t("terminated or truncated")
_T_ONLY_CUT = _t("truncated")
_T_NEVER_ENDS = _t("False")
_T_REVERSED = _t("not terminated")
_T_SUM = _t(back="Q[nxt].sum()")
_T_WHOLE = _t(back="Q.max()")
T_NAMES = {"np": np}


# exercise 3: q_episode
def _e(random_: str = "rng.integers(env.action_space.n)",
       aim: str = "target(Q, reward, nxt, terminated, truncated, GAMMA)") -> str:
    return ("def q_episode(env, Q, rng):\n"
            "    obs, info = env.reset()\n"
            "    total = 0\n"
            "    terminated, truncated = False, False\n"
            "    while not (terminated or truncated):\n"
            "        if rng.random() < EPSILON:\n"
            f"            action = {random_}\n"
            "        else:\n"
            "            action = np.argmax(Q[obs])\n"
            "        nxt, reward, terminated, truncated, info = env.step(action)\n"
            f"        Q[obs, action] += ETA * ({aim} - Q[obs, action])\n"
            "        obs = nxt\n"
            "        total += reward\n"
            "    return total\n")


_E_START = _e("rng.integers(...)", "...")
_E = _e()
_E_FOUR = _e("rng.integers(4)")
_E_REWARD_ONLY = _e(aim="reward")
_E_THIS_STATE = _e(aim="target(Q, reward, obs, terminated, truncated, GAMMA)")
E_NAMES = {"np": np, "target": target, "EPSILON": EPSILON, "ETA": ETA, "GAMMA": GAMMA}

PLAY_CASES_ = PLAY_CASES
TARGET_CASES = target_cases()
EPISODE_CASES = episode_cases()


EXERCISES = {
    "play_random": PlayBox(
        name="play_random",
        function="play_random",
        start=_P_START,
        solution=_P,
        compare=same_play,
        show=show_value,
        visible=PLAY_CASES[:1],
        hidden=PLAY_CASES[1:],
        first=[_DOTS],
        picture=play_picture,
        mistakes=[
            Mistake("no-cut", wrong=_P_NO_CUT,
                    en="`{call}` gives {got}, and it should give {want}: your loop stops only when the episode ends, so in Taxi it goes on past step 200, where the clock cut the episode off. What should stop the loop as well?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je loop stopt alleen als de episode eindigt, dus in Taxi gaat de loop door na stap 200, waar de klok de episode afkapte. Wat moet de loop ook laten stoppen?"),
            Mistake("and", wrong=_P_AND,
                    en="`{call}` gives {got}, and it should give {want}: your loop stops only when `terminated` and `truncated` are both `True`, and in CliffWalking, which has no clock, that never happens. Should the loop stop when both are true, or as soon as one of them is?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je loop stopt alleen als `terminated` en `truncated` allebei `True` zijn, en in CliffWalking, zonder klok, gebeurt dat nooit. Moet de loop stoppen als beide waar zijn, of zodra één van de twee waar is?"),
            Mistake("swapped", wrong=_P_SWAPPED,
                    en="`{call}` gives {got}, and it should give {want}: the last value says the episode ended where it was cut off, and the other way around. In what order does `env.step` give its five values?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de laatste waarde zegt dat de episode eindigde waar de episode werd afgekapt, en andersom. In welke volgorde geeft `env.step` de vijf waarden?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: the test of the `while` loop, and the call that takes one step.",
                      "nl": "Vervang de twee `...`: de test van de `while`-loop, en de aanroep die één stap zet."},
            "arguments": {"en": "The check calls `{call}`, with a world and a seed, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(env, seed)`?",
                          "nl": "De check roept `{call}` aan, met een wereld en een seed, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(env, seed)` houden?"},
            "slow": {"en": "Your function is still not done after 2 seconds. Does the loop stop as soon as the episode ends or is cut off?",
                     "nl": "Je functie is na 2 seconden nog niet klaar. Stopt de loop zodra de episode eindigt of wordt afgekapt?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does `env.step` take the `action` and give its five values to `obs, reward, terminated, truncated, info`, in that order, and does the loop stop as soon as one of the two flags is `True`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Neemt `env.step` de `action` en geeft deze aanroep de vijf waarden aan `obs, reward, terminated, truncated, info`, in deze volgorde, en stopt de loop zodra één van de twee vlaggen `True` is?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your loop stop in every world, also when the clock cuts an episode off?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en het moet {want} zijn. Stopt je loop in elke wereld, ook als de klok een episode afkapt?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return total, steps, terminated`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return total, steps, terminated`?"},
            "correct": {"en": "Right for all {n} episodes: FrozenLake ends in a hole after 2 steps, Taxi is cut off after 200 steps with −911, and a random walk on CliffWalking, which nothing cuts off, takes 3,791 steps. In the picture, which episodes were cut off?",
                        "nl": "Goed voor alle {n} episodes: FrozenLake eindigt na 2 stappen in een wak, Taxi wordt na 200 stappen afgekapt met −911, en een willekeurige wandeling op CliffWalking, die door niets wordt afgekapt, duurt 3.791 stappen. Welke episodes werden in de afbeelding afgekapt?"},
        },
    ),
    "target": FunctionBox(
        name="target",
        function="target",
        start=_T_START,
        solution=_T,
        names=T_NAMES,
        compare=same_number,
        show=show_value,
        visible=TARGET_CASES[:1],
        hidden=TARGET_CASES[1:],
        first=[_DOTS],
        picture=target_picture,
        mistakes=[
            Mistake("cut-ends", wrong=_T_CUT_ENDS,
                    en="`{call}` gives {got}, and it should give {want}: here the clock ran out, and your target says that nothing comes after this step. The taxi still stands in the middle, with the passenger waiting at Y: does its future end because the clock stopped?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: hier liep de klok af, en je target zegt dat er na deze stap niets meer komt. De taxi staat nog in het midden, en de passagier wacht bij Y: houdt de toekomst van de taxi op omdat de klok stopte?"),
            Mistake("only-cut", wrong=_T_ONLY_CUT,
                    en="`{call}` gives {got}, and it should give {want}: your test asks `truncated`, so the step that the clock cut off gets the reward alone, and a step into a hole gets a future. Which of the two flags says that nothing follows?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je test vraagt `truncated`, dus krijgt de stap die de klok afkapte alleen de reward, en een stap een wak in een toekomst. Welke van de twee vlaggen zegt dat er niets meer volgt?"),
            Mistake("never-ends", wrong=_T_NEVER_ENDS,
                    en="`{call}` gives {got}, and it should give {want}: the agent fell into a hole, and your target still adds `gamma` times the largest Q of the hole, 1 in `Q_ones`, as if something follows. In a table of zeros the Q's of a hole stay 0, since the agent never acts there, but should anything be added after an end at all?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de agent viel in een wak, en je target telt er nog `gamma` keer de grootste Q van het wak bij op, 1 in `Q_ones`, alsof er nog iets volgt. In een tabel met nullen blijven de Q's van een wak 0, omdat de agent daar nooit een action neemt, maar hoort er na een einde wel iets bij te komen?"),
            Mistake("reversed", wrong=_T_REVERSED,
                    en="`{call}` gives {got}, and it should give {want}: with `not`, an ordinary step gets the reward alone, and an end gets a future. Which way around should the test be?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: met `not` krijgt een gewone stap alleen de reward, en een einde een toekomst. Hoe moet de test andersom?"),
            Mistake("sum", wrong=_T_SUM,
                    en="`{call}` gives {got}, and it should give {want}: `.sum()` adds up all the Q's of `nxt`, while Q-learning takes the best one. Which method of an array gives its largest number?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `.sum()` telt alle Q's van `nxt` op, terwijl Q-learning de beste neemt. Welke methode van een array geeft het grootste getal?"),
            Mistake("whole-table", wrong=_T_WHOLE,
                    en="`{call}` gives {got}, and it should give {want}: `Q.max()` is the largest Q of the whole table, in any state. Which row of `Q` belongs to the next state?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `Q.max()` is de grootste Q van de hele tabel, in welke state dan ook. Welke rij van `Q` hoort bij de volgende state?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: the test for an episode that ended, and the largest Q of the next state.",
                      "nl": "Vervang de twee `...`: de test voor een episode die eindigde, en de grootste Q van de volgende state."},
            "arguments": {"en": "The check calls `{call}`, with six values, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(Q, reward, nxt, terminated, truncated, gamma)`?",
                          "nl": "De check roept `{call}` aan, met zes waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(Q, reward, nxt, terminated, truncated, gamma)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function give the reward alone only when the episode ended, and otherwise the reward plus `gamma` times the largest Q in row `nxt`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Geeft je functie alleen de reward als de episode eindigde, en anders de reward plus `gamma` keer de grootste Q in rij `nxt`?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function look at row `nxt` of `Q`, whatever `nxt` is?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en het moet {want} zijn. Kijkt je functie naar rij `nxt` van `Q`, wat `nxt` ook is?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does every way through it end with `return`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt elke weg door je functie met `return`?"},
            "unreached": {"en": "With `{call}`, the episode did not end, and your function gives nothing back: Python never gets to a `return` with a value. Does the function still end with the line that starts with `return reward + gamma`?",
                          "nl": "Met `{call}` eindigde de episode niet, en je functie geeft niets terug: Python komt nooit bij een `return` met een waarde. Eindigt de functie nog met de regel die begint met `return reward + gamma`?"},
            "correct": {"en": "Right in all {n} cases: the 200th step of Taxi keeps its future, 4.25 like any other step to the middle, and a step into a hole gives just its reward, 0, even from a table that starts at 1. In the picture, which two steps have only their reward as their target?",
                        "nl": "Goed in alle {n} gevallen: de 200e stap van Taxi houdt de toekomst, 4,25 net als elke andere stap naar het midden, en een stap een wak in geeft alleen de reward, 0, zelfs vanuit een tabel die bij 1 begint. Welke twee stappen hebben in de afbeelding alleen de reward als target?"},
        },
    ),
    "q_episode": EpisodeBox(
        name="q_episode",
        function="q_episode",
        start=_E_START,
        solution=_E,
        names=E_NAMES,
        compare=same_episode,
        show=show_episode,
        visible=EPISODE_CASES[:1],
        hidden=EPISODE_CASES[1:],
        first=[_DOTS],
        picture=episode_picture,
        mistakes=[
            Mistake("four", wrong=_E_FOUR,
                    en="`{call}` gives {got}, and it should give {want}: in FrozenLake and CliffWalking your agent does what it should, but with `rng.integers(4)` a random action in Taxi is never 4 or 5, pickup or drop-off. How does your function find out how many actions a world has?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: in FrozenLake en CliffWalking doet je agent wat de agent moet doen, maar met `rng.integers(4)` is een willekeurige action in Taxi nooit 4 of 5, ophalen of afzetten. Hoe komt je functie te weten hoeveel actions een wereld heeft?"),
            Mistake("reward-only", wrong=_E_REWARD_ONLY,
                    en="`{call}` gives {got}, and it should give {want}: your update moves `Q[obs, action]` toward the reward of this step alone, so nothing that comes after it counts. Which function of this lesson gives the target of Q-learning?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je update schuift `Q[obs, action]` op naar alleen de reward van deze stap, dus telt niets van wat erna komt mee. Welke functie van deze les geeft het target van Q-learning?"),
            Mistake("this-state", wrong=_E_THIS_STATE,
                    en="`{call}` gives {got}, and it should give {want}: the target takes the Q's of `obs`, the state the agent leaves. To which state did `env.step` take the agent?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het target neemt de Q's van `obs`, de state die de agent verlaat. Naar welke state bracht `env.step` de agent?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: how many actions `rng.integers` chooses from, and the target of the update.",
                      "nl": "Vervang de twee `...`: uit hoeveel actions `rng.integers` kiest, en het target van de update."},
            "arguments": {"en": "The check calls `{call}`, with a world, a Q-table and a random generator, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(env, Q, rng)`?",
                          "nl": "De check roept `{call}` aan, met een wereld, een Q-table en een random generator, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(env, Q, rng)` houden?"},
            "slow": {"en": "Your function is still not done after 2 seconds. Does the loop stop as soon as the episode ends or is cut off, and does `obs` move on to `nxt` after every step?",
                     "nl": "Je functie is na 2 seconden nog niet klaar. Stopt de loop zodra de episode eindigt of wordt afgekapt, en schuift `obs` na elke stap door naar `nxt`?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does the random action come from all the actions of `env`, and does the update move `Q[obs, action]` toward `target(Q, reward, nxt, terminated, truncated, GAMMA)`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Komt de willekeurige action uit alle actions van `env`, en schuift de update `Q[obs, action]` op naar `target(Q, reward, nxt, terminated, truncated, GAMMA)`?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work in a world with another number of actions and states?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie in een wereld met een ander aantal actions en states?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return total`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return total`?"},
            "correct": {"en": "Right in all {n} cases, in all three worlds, with one function that knows nothing about any of them but their spaces. In the picture your agent learns CliffWalking and Taxi; which way does it take along the cliff?",
                        "nl": "Goed in alle {n} gevallen, in alle drie de werelden, met één functie die niets weet over de werelden behalve de spaces. In de afbeelding leert je agent CliffWalking en Taxi; welke weg neemt de agent langs het klif?"},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("play_random", _P_START, "empty", None),
    ("play_random", _P, "correct", None),
    ("play_random", _p(step="..."), "wrong", "dots-left"),
    ("play_random", _P_NO_CUT, "wrong", "no-cut"),
    ("play_random", _P_AND, "wrong", "and"),
    ("play_random", _P_SWAPPED, "wrong", "swapped"),
    ("play_random", _P.replace("    return total, steps, terminated\n", ""), "wrong", "no-return"),
    # other right answers: the two flags each with their own not, or a step with all five names on their own line
    ("play_random", _p("not terminated and not truncated").replace("while not (not terminated and not truncated)", "while not terminated and not truncated"), "correct", None),
    # a loop that stops only at the clock: right where the clock stops it, as in the example and in Taxi
    ("play_random", _p("truncated or steps == 2"), "almost", None),
    ("target", _T_START, "empty", None),
    ("target", _T, "correct", None),
    ("target", _t("terminated", "..."), "wrong", "dots-left"),
    ("target", _T_CUT_ENDS, "wrong", "cut-ends"),
    ("target", _T_ONLY_CUT, "wrong", "only-cut"),
    ("target", _T_NEVER_ENDS, "wrong", "never-ends"),
    ("target", _T_REVERSED, "wrong", "reversed"),
    ("target", _T_SUM, "wrong", "sum"),
    ("target", _T_WHOLE, "wrong", "whole-table"),
    ("target", _T.replace("    return reward + gamma * Q[nxt].max()\n", ""), "wrong", "unreached"),
    # other right answers: Python's max of the row, NumPy's, and the test as a comparison
    ("target", _t(back="max(Q[nxt])"), "correct", None),
    ("target", _t("terminated == True", "np.max(Q[nxt])"), "correct", None),
    # the row of the example's state, whatever nxt is: right for the example only
    ("target", _t(back="Q[249].max()"), "almost", None),
    ("q_episode", _E_START, "empty", None),
    ("q_episode", _E, "correct", None),
    ("q_episode", _e(aim="..."), "wrong", "dots-left"),
    ("q_episode", _E_FOUR, "wrong", "four"),
    ("q_episode", _E_REWARD_ONLY, "wrong", "reward-only"),
    ("q_episode", _E_THIS_STATE, "wrong", "this-state"),
    ("q_episode", _E.replace("    return total\n", ""), "wrong", "no-return"),
    # other right answers: a random action from the action space, the columns of Q, or Q-learning's target by hand,
    # which in these tables learns the same: the Q's of an end stay 0
    ("q_episode", _e("env.action_space.sample()"), "correct", None),
    ("q_episode", _e("rng.integers(Q.shape[1])", "reward + GAMMA * Q[nxt].max()"), "correct", None),
]
