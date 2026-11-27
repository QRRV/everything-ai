# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rl-project-grid-agent: the second step of the learner's project agent. GridWorld, a grid world
as a class whose step the learner writes, with walls, puddles, pits and a goal; GridAgent, one agent for Monte Carlo
control, SARSA and Q-learning that differ only in their targets; learning_curve, the mean return of a method at every
episode over several seeds, with the learner's average of the bandit project; episode, the loop the course puts in the
project as agent/episode.py; the learner's own world, the demo in which the three methods learn it with ten seeds, and
the pictures.

The pits follow the cliff of Sutton and Barto (Reinforcement Learning: An Introduction, 2nd edition, 2018, example 6.6) as
the earlier lessons of the course used it; the puddles are the course's own, as are the code, the runs and the pictures."""
import ast
import copy
import html
import math
import numbers
from collections import deque
from pathlib import Path

import numpy as np

from everything_ai import chart, live, projects, yours
from everything_ai.check import ClassBox, FunctionBox, Mistake, Timeout, kept, run_code, with_time_limit

LESSON = "rl-project-grid-agent"
# Exercise 3 and the demo use your world and your agent of exercises 1 and 2 and your average of the bandit project:
# while one of them is missing or broken, the course's stands in.
projects.builds_on(("agent", "measure", "rl-project-bandit", "average"),
                   ("agent", "gridworld", LESSON, "world"),
                   ("agent", "gridagent", LESSON, "agent"))
projects.provide(LESSON)   # agent/episode.py, which the lesson gives, also when only the checks are loaded

ACTIONS = ["up", "down", "left", "right"]
METHODS = ("mc", "sarsa", "q")
EPSILON = 0.1      # every agent of the lesson explores with ε = 0.1
ETA = 0.2          # and learns with η = 0.2
GAMMA = 1.0        # every step costs, so the return needs no discount
LIMIT = 200        # an episode stops after this many steps, also when the agent has not reached G
SEEDS = 10         # seed k gives the agent of run k np.random.default_rng(k), for every method
EPISODES = 300
LAST = 50          # the lesson compares the methods over the last 50 episodes
WINDOW = 10        # a curve shows the mean over the last 10 episodes
LOW = -150         # the bottom of the charts of returns: the first episodes, far lower, stay at its edge
DEMO_LIMIT = 120.0   # seconds the demo gets
LOOKS = [0, 1, 5, 10, 20, 50, 100, 200, 300]   # episodes the rewind stops at
MAX_ROWS, MAX_COLS = 8, 12

# The course's world: a lane from S to G between a row of puddles above it and a row of pits below it. The lane takes
# 7 steps; the dry way over the top row takes 11.
WORLD = [
    "........",
    ".WWWWWW.",
    "S......G",
    ".XXXXXX.",
]
# A maze for exercise 4: 16 steps from S to G, and Monte Carlo never gets there.
MAZE = [
    "S..#.....",
    ".#.#.###.",
    ".#...#...",
    ".####.#X.",
    "......#.G",
]
SIGNS = ".SGXW#"


# --- the course's code: the loop it gives, and the solutions -----------------------------------------------------------

_EPISODE = (Path(__file__).parent / "project" / "agent" / "episode.py").read_text(encoding="utf-8")


def _names(code: str) -> dict:
    out = {"__name__": "__course__"}
    exec(compile(code, "<course>", "exec"), out)
    return out


episode = _names(_EPISODE)["episode"]

# exercise 1: GridWorld
_W_HEAD = ('MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}\n\n\n'
           "class GridWorld:\n"
           "    def __init__(self, rows):\n"
           "        self.rows = rows\n"
           "        self.height = len(rows)\n"
           "        self.width = len(rows[0])\n"
           "        for row in range(self.height):\n"
           "            for col in range(self.width):\n"
           '                if rows[row][col] == "S":\n'
           "                    self.start = (row, col)\n"
           "        self.state = self.start\n\n"
           "    def reset(self):\n"
           "        self.state = self.start\n"
           "        return self.state\n\n"
           "    def step(self, action):\n"
           "        # the cell the action leads to; into the edge or a wall, the agent stays where it is\n"
           "        r = self.state[0] + MOVES[action][0]\n"
           "        c = self.state[1] + MOVES[action][1]\n"
           '        if r < 0 or r >= self.height or c < 0 or c >= self.width or self.rows[r][c] == "#":\n'
           "            r, c = self.state\n"
           "        sign = self.rows[r][c]\n"
           "        # gives back where the agent is now, the reward, and whether the episode is over\n")
_W_PIT = ('        if sign == "X":\n'
          "            self.state = self.start\n"
          "            return self.state, -100, False\n")
_W_MOVE = "        self.state = (r, c)\n"
_W_GOAL = ('        if sign == "G":\n'
           "            return self.state, -1, True\n")
_W_WET = ('        if sign == "W":\n'
          "            return self.state, -10, False\n")
_W_DRY = "        return self.state, -1, False\n"
_W_START = _W_HEAD + "        ...\n"


def _w(pit: str = _W_PIT, move: str = _W_MOVE, goal: str = _W_GOAL, wet: str = _W_WET, dry: str = _W_DRY) -> str:
    return _W_HEAD + pit + move + goal + wet + dry


_W = _w()

# exercise 2: GridAgent
_G_HEAD = ("import numpy as np\n\n"
           'ACTIONS = ["up", "down", "left", "right"]\n\n\n'
           "class GridAgent:\n"
           "    def __init__(self, method, epsilon, eta, gamma, rng):\n"
           '        self.method = method   # "mc", "sarsa" or "q"\n'
           "        self.epsilon = epsilon\n"
           "        self.eta = eta\n"
           "        self.gamma = gamma\n"
           "        self.rng = rng\n"
           "        self.q = {}            # for every state the agent has seen: a dict with the Q of each action\n"
           '        self.steps = []        # for "mc": the steps of this episode so far\n\n'
           "    def qs(self, state):\n"
           "        if state not in self.q:\n"
           '            self.q[state] = {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0}\n'
           "        return self.q[state]\n\n"
           "    def act(self, state):\n"
           "        if self.rng.random() < self.epsilon:\n"
           "            return ACTIONS[self.rng.integers(0, 4)]\n"
           "        return ACTIONS[int(np.argmax(list(self.qs(state).values())))]\n\n"
           "    def move(self, state, action, target):\n"
           "        # moves the Q of the state and the action a part eta of the way toward the target\n"
           "        qs = self.qs(state)\n"
           "        qs[action] = qs[action] + self.eta * (target - qs[action])\n\n"
           "    def learn(self, state, action, reward, nxt, nxt_action):\n"
           '        if self.method == "mc":\n'
           "            self.steps.append((state, action, reward))\n"
           '        elif self.method == "sarsa":\n')
_G_SARSA = "reward + self.gamma * self.qs(nxt)[nxt_action]"
_G_Q = "reward + self.gamma * max(self.qs(nxt).values())"
_G_MC = "reward + self.gamma * g"


def _g(sarsa: str = _G_SARSA, q: str = _G_Q, mc: str = _G_MC) -> str:
    return (_G_HEAD
            + f"            self.move(state, action, {sarsa})\n"
            + "        else:\n"
            + f"            self.move(state, action, {q})\n\n"
            + "    def end(self):\n"
            + "        # Monte Carlo: the return of every step, from the last step back to the first\n"
            + "        g = 0.0\n"
            + "        for state, action, reward in reversed(self.steps):\n"
            + f"            g = {mc}\n"
            + "            self.move(state, action, g)\n"
            + "        self.steps = []\n")


_G = _g()
_G_START = _g("...", "...", "...")

# exercise 3: learning_curve
_C_HEAD = ("import numpy as np\n"
           "from agent.episode import episode\n"
           "from agent.gridagent import GridAgent\n"
           "from agent.gridworld import GridWorld\n"
           "from agent.measure import average\n\n"
           "LIMIT = 200\n\n\n"
           "def learning_curve(method, rows, seeds, episodes):\n"
           "    # the mean return at every episode over `seeds` runs, each with a new world and a new agent\n"
           "    runs = []\n"
           "    for seed in range(seeds):\n"
           "        world = GridWorld(rows)\n"
           "        agent = GridAgent(method, 0.1, 0.2, 1.0, np.random.default_rng(seed))\n")
_C_NEW = "        returns = []\n"
_C_LOOP = "        for e in range(episodes):\n"
_C_PLAY = "            returns.append(episode(agent, world, LIMIT))\n"
_C_KEEP = "        runs.append(returns)\n"
_C_TAIL = "    return average(runs)\n"
_C_START = _C_HEAD + "        ...\n" + _C_TAIL


def _c(body: str | None = None) -> str:
    return _C_HEAD + (body if body is not None else _C_NEW + _C_LOOP + _C_PLAY + _C_KEEP) + _C_TAIL


_C = _c()

_COURSE = _names(_W)
_COURSE.update(_names(_G))
GridWorld = _COURSE["GridWorld"]
GridAgent = _COURSE["GridAgent"]


def average(runs):
    """The course's average of the bandit project: the mean over the runs at every episode."""
    return [float(x) for x in np.mean(np.asarray(runs, dtype=float), axis=0)]


def learning_curve(method, rows, seeds, episodes):
    """The course's learning_curve, with the course's world, agent and average: the solution of exercise 3."""
    runs = []
    for seed in range(seeds):
        world = GridWorld(rows)
        agent = GridAgent(method, EPSILON, ETA, GAMMA, np.random.default_rng(seed))
        runs.append([episode(agent, world, LIMIT) for _ in range(episodes)])
    return average(runs)


# --- a world of the learner: what it must be, and where the agent can go -----------------------------------------------

_PROBLEMS = {
    "type": {"en": "`my_world` must be a list of strings, one string per row, such as `[\"S..G\"]`.",
             "nl": "`my_world` moet een lijst met strings zijn, één string per rij, zoals `[\"S..G\"]`."},
    "size": {"en": "Your world has {rows} rows of up to {cols} signs; the demo can use worlds of 1 to 8 rows of 2 to 12 signs.",
             "nl": "Je wereld heeft {rows} rijen van hoogstens {cols} tekens; de demo kan werelden gebruiken van 1 tot en met 8 rijen van 2 tot en met 12 tekens."},
    "ragged": {"en": "The rows have different lengths: {lengths}. Every row needs the same number of signs.",
               "nl": "De rijen zijn niet even lang: {lengths}. Elke rij heeft hetzelfde aantal tekens nodig."},
    "signs": {"en": "Your world has signs the course does not know: {signs}. Use `.`, `S`, `G`, `#`, `W` and `X`.",
              "nl": "Je wereld heeft tekens die de course niet kent: {signs}. Gebruik `.`, `S`, `G`, `#`, `W` en `X`."},
    "start": {"en": "Your world has {n} start cells `S`; it needs exactly one.",
              "nl": "Je wereld heeft {n} startvakjes `S`; er moet er precies één zijn."},
    "goal": {"en": "Your world has no goal `G`; it needs at least one.",
             "nl": "Je wereld heeft geen doel `G`; er moet er minstens één zijn."},
    "closed": {"en": "No way leads from S to a goal without a wall or a pit in it, so no agent can ever finish an episode.",
               "nl": "Er loopt geen weg van S naar een doel zonder muur of put, dus kan geen enkele agent ooit een episode afmaken."},
}


def problems(rows) -> list:
    """What is wrong with a world, as texts in both languages; empty when the course can train on it."""
    if not isinstance(rows, (list, tuple)) or not rows or not all(isinstance(r, str) for r in rows):
        return [_PROBLEMS["type"]]
    lengths = [len(r) for r in rows]
    if not 1 <= len(rows) <= MAX_ROWS or not 2 <= max(lengths) <= MAX_COLS:
        return [{k: v.format(rows=len(rows), cols=max(lengths)) for k, v in _PROBLEMS["size"].items()}]
    out = []
    if len(set(lengths)) > 1:
        out.append({k: v.format(lengths=", ".join(str(n) for n in lengths)) for k, v in _PROBLEMS["ragged"].items()})
    odd = sorted({ch for r in rows for ch in r if ch not in SIGNS})
    if odd:
        shown = ", ".join("`" + (ch if ch.strip() else repr(ch)) + "`" for ch in odd)
        out.append({k: v.format(signs=shown) for k, v in _PROBLEMS["signs"].items()})
    starts = sum(r.count("S") for r in rows)
    if starts != 1:
        out.append({k: v.format(n=starts) for k, v in _PROBLEMS["start"].items()})
    if not any("G" in r for r in rows):
        out.append(_PROBLEMS["goal"])
    if not out and shortest(rows) is None:
        out.append(_PROBLEMS["closed"])
    return out


def find(rows, sign: str) -> tuple:
    for r, row in enumerate(rows):
        if sign in row:
            return r, row.index(sign)
    return None


def shortest(rows) -> int | None:
    """The fewest steps from S to a goal, through cells that are not walls or pits; None when no goal can be reached."""
    start = find(rows, "S")
    seen, todo = {start: 0}, deque([start])
    while todo:
        r, c = todo.popleft()
        if rows[r][c] == "G":
            return seen[(r, c)]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(rows) and 0 <= nc < len(rows[0]) and rows[nr][nc] not in "#X" and (nr, nc) not in seen:
                seen[(nr, nc)] = seen[(r, c)] + 1
                todo.append((nr, nc))
    return None


# --- the runs: three methods, ten seeds, on one world ----------------------------------------------------------------------

def greedy(qs: dict) -> str:
    """The action an agent takes when it does not explore: the largest Q, the first one where several tie."""
    best, top = "up", qs["up"]
    for a in ("down", "left", "right"):
        if qs[a] > top:
            best, top = a, qs[a]
    return best


def arrows(rows, q: dict) -> dict:
    """The best actions of every cell the agent has a Q for, as small arrows: all actions with the largest Q, none where
    all four tie, such as a cell the agent has never left, and none on a goal."""
    out = {}
    for s, qs in q.items():
        r, c = s
        if rows[r][c] == "G":
            continue
        top = max(qs.values())
        best = [a for a in ACTIONS if qs[a] > top - 1e-9]
        if len(best) < len(ACTIONS):
            out[s] = best
    return out


def way(rows, q: dict) -> list | None:
    """The cells from S to G when the agent always takes its best action, the first one where several tie; None when
    that way runs into a pit, an edge or a wall, or comes back to a cell it has seen."""
    world = GridWorld(rows)
    s, cells = world.reset(), [world.start]
    for _ in range(len(rows) * len(rows[0])):
        qs = q.get(s, {a: 0.0 for a in ACTIONS})
        nxt, reward, done = world.step(greedy(qs))
        if nxt in cells:
            return None
        cells.append(nxt)
        if done:
            return cells
        s = nxt
    return None


def _snapshot(rows, agent) -> dict:
    q = {s: dict(qs) for s, qs in agent.q.items()}
    return {"arrows": arrows(rows, q), "way": way(rows, q)}


class _Broken(Exception):
    """The learner's code stopped with an error during the run, or gave something that is not a number."""


def train(rows, world_class=None, agent_class=None, average_fn=None, show=None, seeds: int = SEEDS,
          episodes: int = EPISODES) -> dict:
    """The three methods on `rows`, run k of each with np.random.default_rng(k), WINDOW episodes at a time so that a
    chart can grow: the returns of every run at every episode, the mean over the runs with `average_fn`, and the best
    actions and the way of run 1 (seed 0) at every episode of LOOKS. `show(n, r)` after every block of episodes."""
    world_class = world_class or GridWorld
    agent_class = agent_class or GridAgent
    average_fn = average_fn or average
    worlds = {m: [world_class(rows) for _ in range(seeds)] for m in METHODS}
    agents = {m: [agent_class(m, EPSILON, ETA, GAMMA, np.random.default_rng(k)) for k in range(seeds)] for m in METHODS}
    r = {"rows": list(rows), "returns": {m: [[] for _ in range(seeds)] for m in METHODS},
         "mean": {m: [] for m in METHODS}, "first": {}, "now": {}, "episodes": 0}

    def look(n):
        r["now"] = {m: _snapshot(rows, agents[m][0]) for m in METHODS}
        if n in LOOKS:
            r["first"][n] = r["now"]

    look(0)
    done = 0
    while done < episodes:
        block = min(WINDOW if done >= WINDOW else 1, episodes - done)   # the first ten episodes one at a time
        for m in METHODS:
            for k in range(seeds):
                for _ in range(block):
                    try:
                        got = episode(agents[m][k], worlds[m][k], LIMIT)
                    except Timeout:
                        raise
                    except Exception as e:   # noqa: BLE001   the demo says which part of the project stopped
                        raise _Broken("agent") from e
                    if not isinstance(got, numbers.Real) or not math.isfinite(got):
                        raise _Broken("agent")
                    r["returns"][m][k].append(float(got))
        done += block
        r["episodes"] = done
        for m in METHODS:
            try:
                mean = average_fn([list(run_) for run_ in r["returns"][m]])
                mean = [float(x) for x in mean]
            except Timeout:
                raise
            except Exception as e:   # noqa: BLE001
                raise _Broken("average") from e
            if len(mean) != done or not all(math.isfinite(x) for x in mean):
                raise _Broken("average")
            r["mean"][m] = mean
        look(done)
        if show is not None:
            show(done, r)
    return r


def course_run() -> dict:
    """The course's run on its own world: the numbers the lesson's texts give."""
    return kept("rl-project-grid-agent:world", lambda: train(WORLD))


def maze_run() -> dict:
    """The course's run on the maze of exercise 4."""
    return kept("rl-project-grid-agent:maze", lambda: train(MAZE))


def last_mean(values, n: int = LAST) -> float:
    return float(np.mean(values[-n:]))


def per_seed(r: dict, method: str, n: int = LAST) -> list:
    """The mean return of each run over its last n episodes."""
    return [last_mean(run_, n) for run_ in r["returns"][method]]


def reached(rows, method: str, seeds: int = SEEDS, episodes: int = EPISODES) -> int:
    """How many episodes of all runs of a method reach G before the limit: the runs of train, counted step by step."""
    count = 0
    for k in range(seeds):
        world = GridWorld(rows)
        agent = GridAgent(method, EPSILON, ETA, GAMMA, np.random.default_rng(k))
        for _ in range(episodes):
            state = world.reset()
            action = agent.act(state)
            for _ in range(LIMIT):
                nxt, reward, done = world.step(action)
                nxt_action = agent.act(nxt)
                agent.learn(state, action, reward, nxt, nxt_action)
                if done:
                    count += 1
                    break
                state, action = nxt, nxt_action
            agent.end()
    return count


# --- the words and the parts of the pictures ---------------------------------------------------------------------------

LANGS = ("en", "nl")
_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem"
_HALO = "paint-order:stroke;stroke:var(--ea-page);stroke-width:3px;stroke-linejoin:round"
_TOWARD = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


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


def _chart(panels, alt: str, lang: str, width: int = 330, height: int = 240) -> str:
    out = chart.figure(panels, alt=alt, width=width, height=height)
    return chart.with_decimal_comma(out) if lang == "nl" else out


_WORDS = {
    "en": {"mc": "Monte Carlo", "sarsa": "SARSA", "q": "Q-learning", "episodes": "episodes", "return": "mean return",
           "start": "start", "goal": "goal", "wall": "wall", "puddle": "puddle", "pit": "pit, back to S",
           "one": "episode", "many": "episodes"},
    "nl": {"mc": "Monte Carlo", "sarsa": "SARSA", "q": "Q-learning", "episodes": "episodes", "return": "gemiddelde return",
           "start": "start", "goal": "doel", "wall": "muur", "puddle": "plas", "pit": "put, terug naar S",
           "one": "episode", "many": "episodes"},
}
# how each method is drawn in a chart with all three: role, width, dashed; the key under the chart says which is which
_STYLE = {"sarsa": ("ink", 2.4, False), "q": ("ink", 1.4, True), "mc": ("ink-3", 1.4, True)}


def curve_key(lang: str) -> list:
    w = _WORDS[lang if lang in LANGS else "en"]
    return [(_STYLE[m][0], _STYLE[m][2], w[m]) for m in ("sarsa", "q", "mc")]


def cell_size(rows, tall: bool = False) -> int:
    """The size of a cell: at most 26 px, a picture at most 314 px wide, and in a column of three (tall) at most 150 px
    high each."""
    size = min(26, 312 // len(rows[0]))
    return min(size, 150 // len(rows)) if tall else size


def world_svg(rows, panels: list, alt: str, lang: str, cell: int | None = None) -> str:
    """One or more copies of a grid world, one under the other, in one picture. Each panel is a dict: `title`, `arrows`
    {cell: [actions]} as small ink triangles, `way`, the cells from S to G drawn as an orange line (what the agent
    learned), or None, `walk`, the cells of a walk drawn as an ink line, broken where the agent went back to S, and
    `falls`, (cell, pit) pairs: a step into a pit, drawn as a line to the edge of the pit with a dot. Walls are filled, pits filled
    with their reward, puddles shaded with their cost in the corner, left out in a panel with arrows, where the arrows
    would cover it. The picture is as wide as the longest title needs, at most 314 px."""
    lang = lang if lang in LANGS else "en"
    cell = cell or cell_size(rows, tall=len(panels) > 1)
    titles = [len(p["title"]) * 7.6 + 4 for p in panels if p.get("title")]
    W = max(len(rows[0]) * cell + 2, min(314, int(max(titles, default=0))))
    parts, y0 = [], 0
    small = max(8.5, min(11.0, cell * 0.42))
    for p in panels:
        if p.get("title"):
            parts.append(f'<text x="1" y="{y0 + 14}" style="font-size:13px;font-weight:600;fill:var(--ea-ink)">'
                         f'{html.escape(p["title"])}</text>')
            top = y0 + 22
        else:
            top = y0 + 1
        for r, row in enumerate(rows):
            for c, sign in enumerate(row):
                x, y = 1 + c * cell, top + r * cell
                fill = {"#": "var(--ea-ink-2)", "X": "var(--ea-data)", "W": "var(--ea-data-wash)"}.get(sign, "var(--ea-page)")
                parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" style="fill:{fill};'
                             f'stroke:var(--ea-line);stroke-width:1"/>')
                if sign == "X":
                    parts.append(f'<text x="{x + cell / 2:.1f}" y="{y + cell / 2 + small * 0.36:.1f}" style="font-size:{small:.1f}px;'
                                 f'font-weight:600;fill:var(--ea-page);text-anchor:middle">−100</text>')
                elif sign == "W" and not p.get("arrows"):
                    parts.append(f'<text x="{x + 2}" y="{y + small + 0.5:.1f}" style="font-size:{small * 0.9:.1f}px;'
                                 f'font-weight:600;fill:var(--ea-ink-2)">−10</text>')
                elif sign in "SG":
                    parts.append(f'<text x="{x + cell / 2:.1f}" y="{y + cell / 2 + 4.5:.1f}" style="font-size:{min(13, cell * 0.5):.1f}px;'
                                 f'font-weight:700;fill:var(--ea-ink-2);text-anchor:middle;{_HALO}">{sign}</text>')
        walk = p.get("walk") or []
        piece = []
        for a, b in zip([None] + walk, walk):
            if a is not None and abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1:   # back to S after a pit: a new piece
                parts.append(_polyline(piece, top, cell, "var(--ea-ink)", 2.2, 0.55))
                piece = []
            piece.append(b)
        parts.append(_polyline(piece, top, cell, "var(--ea-ink)", 2.2, 0.55))
        for a, b in p.get("falls") or []:   # the line stops at the edge of the pit, so the dot leaves −100 readable
            ax, ay = 1 + a[1] * cell + cell / 2, top + a[0] * cell + cell / 2
            ex, ey = ax + (b[1] - a[1]) * cell * 0.66, ay + (b[0] - a[0]) * cell * 0.66
            parts.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" style="stroke:var(--ea-ink);'
                         'stroke-width:2.2;stroke-linecap:round;opacity:0.55"/>')
            parts.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="3.4" style="fill:var(--ea-ink);stroke:var(--ea-page);stroke-width:1.4"/>')
        if p.get("way"):
            parts.append(_polyline(p["way"], top, cell, "var(--ea-param)", 3.0, 0.85))
        for (r, c), acts in sorted(p.get("arrows", {}).items()):
            if rows[r][c] in "SG":
                continue
            mx, my = 1 + c * cell + cell / 2, top + r * cell + cell / 2
            for a in acts:
                dx, dy = _TOWARD[a]
                reach = cell / 2 - 2
                tip = (mx + dx * reach, my + dy * reach)
                base = (mx + dx * (reach - cell * 0.22), my + dy * (reach - cell * 0.22))
                side = (dy * cell * 0.14, dx * cell * 0.14)
                corners = [tip, (base[0] + side[0], base[1] + side[1]), (base[0] - side[0], base[1] - side[1])]
                parts.append('<polygon points="' + " ".join(f"{px:.1f},{py:.1f}" for px, py in corners)
                             + '" style="fill:var(--ea-ink-2)"/>')
        y0 = top + len(rows) * cell + 12
    H = y0 - 10
    svg = (f'<svg viewBox="0 0 {W} {H}" aria-hidden="true" style="flex:1 1 0;width:100%;min-width:min(100%,{W}px);'
           f'max-width:{W}px;height:auto;font-family:var(--ea-font-ui)">{"".join(parts)}</svg>')
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(alt)}" style="margin:0.25rem 0 0.5rem;'
            f'max-width:min(100%,{W}px);display:flex">{svg}</figure>')


def _polyline(cells, top: float, cell: int, colour: str, width: float, opacity: float) -> str:
    if len(cells) < 2:
        return ""
    points = " ".join(f"{1 + c * cell + cell / 2:.1f},{top + r * cell + cell / 2:.1f}" for r, c in cells)
    return (f'<polyline points="{points}" style="fill:none;stroke:{colour};stroke-width:{width};'
            f'stroke-linecap:round;stroke-linejoin:round;opacity:{opacity}"/>')


_KEY_TEXT = {
    "en": "`S` start, `G` goal; a cell marked −10: a puddle, a step onto it costs −10; a cell marked −100: a pit, a step into it costs −100 and puts the agent back on S; a filled cell without a number: a wall, the agent stays in its cell. Every other step costs −1.",
    "nl": "`S` start, `G` doel; een vakje met −10: een plas, een stap erop kost −10; een vakje met −100: een put, een stap erin kost −100 en zet de agent terug op S; een gevuld vakje zonder getal: een muur, de agent blijft in hetzelfde vakje. Elke andere stap kost −1.",
}


def world_key(lang: str) -> str:
    return _note(_KEY_TEXT[lang if lang in LANGS else "en"])


_VIEW_TEXT = {
    "en": {"title": "Your world", "alt": "Your grid world, {r} rows by {c} columns.",
           "ok": "The shortest way from S to G takes {n} steps. Your agents can learn this world: press the button of the demo below.",
           "bad": "The demo cannot use this world yet:"},
    "nl": {"title": "Je wereld", "alt": "Je grid world, {r} rijen bij {c} kolommen.",
           "ok": "De kortste weg van S naar G kost {n} stappen. Je agents kunnen deze wereld leren: druk op de knop van de demo hieronder.",
           "bad": "De demo kan deze wereld nog niet gebruiken:"},
}


def _drawable(rows) -> bool:
    """A world the picture can draw: a list of 1 to 8 strings of the same length, 2 to 12 signs each."""
    return (isinstance(rows, (list, tuple)) and 1 <= len(rows) <= MAX_ROWS and all(isinstance(r, str) for r in rows)
            and len({len(r) for r in rows}) == 1 and 2 <= len(rows[0]) <= MAX_COLS)


def world_view(rows, lang: str) -> str:
    """The learner's world as the demo will see it, with what is wrong with it, or the length of its shortest way."""
    lang = lang if lang in LANGS else "en"
    w = _VIEW_TEXT[lang]
    found = problems(rows)
    drawn = ""
    if _drawable(rows):
        drawn = world_svg(rows, [{}], w["alt"].format(r=len(rows), c=len(rows[0])), lang) + world_key(lang)
    if found:
        items = "".join(f"<li>{_marked(p[lang])}</li>" for p in found)
        return yours.figure(w["title"], drawn + f'<div style="{_LOSS_UI}">{_marked(w["bad"])}<ul style="margin:0.2rem 0 0 1.1rem;padding:0">{items}</ul></div>',
                            alt=w["title"])
    return yours.figure(w["title"], drawn + _note(w["ok"].format(n=shortest(rows))), alt=w["title"])


# --- the chart of the runs -------------------------------------------------------------------------------------------

def smooth(curve: list) -> list:
    """The mean over the last WINDOW episodes at every episode."""
    out, total = [], 0.0
    for i, y in enumerate(curve):
        total += y
        if i >= WINDOW:
            total -= curve[i - WINDOW]
        out.append(total / min(i + 1, WINDOW))
    return out


def chart_points(r: dict, method: str, start: int, end: int) -> list:
    """The points of the demo's chart for episodes start to end: the mean return of the last 10 episodes, over the runs,
    never below the bottom of the chart."""
    line = smooth(r["mean"][method][:end])
    return [(e, max(LOW, line[e - 1])) for e in range(max(1, start), end + 1)]


def _curve_panel(r: dict, n: int, lang: str, title: str) -> chart.Panel:
    w = _WORDS[lang]
    p = chart.Panel((0, EPISODES), (LOW, 0), title=title, xlabel=w["episodes"], ylabel=w["return"],
                    yticks=[-150, -100, -50, 0])
    for m in ("mc", "q", "sarsa"):
        points = chart_points(r, m, 1, len(r["mean"][m]))
        if points:
            role, width, dashed = _STYLE[m]
            p.line([x for x, _ in points], [y for _, y in points], role, width, dashed=dashed)
    if 1 <= n <= len(r["mean"]["sarsa"]):
        p.line([n, n], [LOW, 0], "axis", 1.2)
    return p


def _grids(r: dict, n: int, lang: str, snap: dict) -> str:
    w, dw = _WORDS[lang], DEMO_TEXT[lang]
    word = w["one"] if n == 1 else w["many"]
    panels = [{"title": dw["grid"].format(m=w[m], n=number(n, 0, lang), word=word),
               "arrows": snap[m]["arrows"], "way": snap[m]["way"]} for m in METHODS]
    return world_svg(r["rows"], panels, dw["grid_alt"].format(n=number(n, 0, lang), word=word), lang)


DEMO_TEXT = {
    "en": {"title": "Your agents: mean return", "x": "episodes", "y": "mean return",
           "alt": "The mean return of the last 10 episodes over the 10 seeds, after every episode so far: SARSA solid, Q-learning dashed, Monte Carlo light and dashed.",
           "grid": "{m}, seed 0, after {n} {word}",
           "grid_alt": "Three copies of the world, Monte Carlo, SARSA and Q-learning: the best actions of the run with seed 0 after {n} {word} as small arrows, and where they lead from S, if they lead to G, as an orange line.",
           "say0": "Before the first episode every Q is 0, so no cell has a best action yet.",
           "rewind1": "After {n} {word}: over the 10 seeds, the first episode gave Monte Carlo a mean return of {mc}, SARSA {sarsa} and Q-learning {q}. With seed 0, {ways}",
           "rewind": "After {n} {word}: over the 10 seeds, the last {k} episodes gave Monte Carlo a mean return of {mc}, SARSA {sarsa} and Q-learning {q}. With seed 0, {ways}",
           "lead": "the arrows of {m} lead from S to G in {k} steps",
           "also": "of {m} in {k}",
           "none_after": "those of {ms} do not lead to G yet",
           "none_only": "the arrows of {ms} do not lead to G yet",
           "progress": "**Episode {n} of {total}**"},
    "nl": {"title": "Je agents: gemiddelde return", "x": "episodes", "y": "gemiddelde return",
           "alt": "De gemiddelde return van de laatste 10 episodes over de 10 seeds, na elke episode tot nu toe: SARSA doorgetrokken, Q-learning gestreept, Monte Carlo licht en gestreept.",
           "grid": "{m}, seed 0, na {n} {word}",
           "grid_alt": "Drie keer de wereld, Monte Carlo, SARSA en Q-learning: de beste actions van de run met seed 0 na {n} {word} als kleine pijlen, en, als de pijlen naar G leiden, de weg vanaf S als oranje lijn.",
           "say0": "Bij de start is elke Q 0, dus heeft nog geen enkel vakje een beste action.",
           "rewind1": "Na {n} {word}: over de 10 seeds gaf de eerste episode Monte Carlo een gemiddelde return van {mc}, SARSA {sarsa} en Q-learning {q}. Met seed 0 {ways}",
           "rewind": "Na {n} {word}: over de 10 seeds gaven de laatste {k} episodes Monte Carlo een gemiddelde return van {mc}, SARSA {sarsa} en Q-learning {q}. Met seed 0 {ways}",
           "lead": "leiden de pijlen van {m} in {k} stappen van S naar G",
           "also": "van {m} in {k}",
           "none_after": "de pijlen van {ms} leiden nog niet naar G",
           "none_only": "leiden de pijlen van {ms} nog niet naar G",
           "progress": "**Episode {n} van {total}**"},
}


def _ways(rows, snap: dict, lang: str) -> str:
    """Where the arrows of each method lead from S, as the end of the sentence after "With seed 0,"."""
    w, dw = _WORDS[lang], DEMO_TEXT[lang]
    lead = [(w[m], len(snap[m]["way"]) - 1) for m in METHODS if snap[m]["way"]]
    none = [w[m] for m in METHODS if not snap[m]["way"]]
    out = ""
    if lead:
        pieces = [dw["lead"].format(m=lead[0][0], k=lead[0][1])] + [dw["also"].format(m=m, k=k) for m, k in lead[1:]]
        out = _listed(pieces, lang)
    if none:
        out = out + "; " + dw["none_after"].format(ms=_listed(none, lang)) if out else dw["none_only"].format(ms=_listed(none, lang))
    return out + "."


def now_picture(r: dict, n: int, lang: str) -> str:
    """Beside the chart of the demo: the best actions of the runs with seed 0 as they are now."""
    lang = lang if lang in LANGS else "en"
    return _grids(r, n, lang, r["now"])


def rewind_frame(r: dict, n: int, lang: str) -> tuple:
    """A stop of the rewind after the demo: the chart of the runs with a line at n, the best actions of the runs with
    seed 0 then, and a sentence."""
    lang = lang if lang in LANGS else "en"
    w, dw = _WORDS[lang], DEMO_TEXT[lang]
    n = min(r["first"], key=lambda k: abs(k - int(n)))
    word = w["one"] if n == 1 else w["many"]
    curve = _chart([_curve_panel(r, n, lang, dw["title"])], dw["alt"], lang) + live.key_html(curve_key(lang))
    if n == 0:
        say = dw["say0"]
    else:
        k = min(WINDOW, n)
        mean = {m: number(float(np.mean(r["mean"][m][n - k:n])), 1, lang) for m in METHODS}
        say = dw["rewind1" if k == 1 else "rewind"].format(n=number(n, 0, lang), word=word, k=k, ways=_ways(r["rows"], r["first"][n], lang), **mean)
    return live.rewound(curve, [_grids(r, n, lang, r["first"][n])], width=330), say


# --- the demo: your world, your agent and your average --------------------------------------------------------------------

def _imported(code: str, name: str):
    got = run_code(code)
    value = got.names.get(name) if got.error is None else None
    return value if callable(value) else None


def project_parts() -> dict:
    """GridWorld, GridAgent and average of the learner's project when they are there and pass their contract test,
    else the course's: a dict with the three and which of them are yours."""
    out = {"world": GridWorld, "agent": GridAgent, "average": average, "yours": []}
    for key, module, name in (("world", "gridworld", "GridWorld"), ("agent", "gridagent", "GridAgent"),
                              ("average", "measure", "average")):
        if projects.state("agent", module) == "yours":
            got = _imported(f"from agent.{module} import {name}", name)
            if got is not None:
                out[key] = got
                out["yours"].append(key)
    return out


def demo(rows, show=None) -> dict | str:
    """The demo within its time limit: the three methods on `rows` with the parts of your project (project_parts), or a
    string: "agent" when your world or agent stops with an error or gives something that is not a number, "average"
    when your average does, or "slow"."""
    parts = project_parts()
    try:
        out = with_time_limit(lambda: train(rows, parts["world"], parts["agent"], parts["average"], show=show), DEMO_LIMIT)
    except Timeout:
        return "slow"
    except _Broken as e:
        return str(e)
    out["yours"] = parts["yours"]
    return out


_SUMMARY = {
    "en": {"who": {("world", "agent", "average"): "your `GridWorld`, your `GridAgent` and your `average`",
                   "partly": "{yours}, and the course's {course}, since yours {verb} not pass {its} check yet",
                   "none": "the course's `GridWorld`, `GridAgent` and `average`, since yours are not in your project yet"},
           "say": "Learned with {who}. Over the last 50 episodes, the mean return was {mc} for Monte Carlo, {sarsa} for SARSA and {q} for Q-learning. Over the 10 seeds, the runs of Monte Carlo ranged from {lo} to {hi}."},
    "nl": {"who": {("world", "agent", "average"): "je `GridWorld`, je `GridAgent` en je `average`",
                   "partly": "{yours}, en {course} van de course, omdat jouw {version} nog niet door de check {verb}",
                   "none": "de `GridWorld`, `GridAgent` en `average` van de course, omdat jouw versies nog niet in je project staan"},
           "say": "Geleerd met {who}. Over de laatste 50 episodes was de gemiddelde return {mc} voor Monte Carlo, {sarsa} voor SARSA en {q} voor Q-learning. Over de 10 seeds liepen de runs van Monte Carlo van {lo} tot {hi}."},
}
_PART_NAME = {"world": "`GridWorld`", "agent": "`GridAgent`", "average": "`average`"}


def _listed(items: list, lang: str) -> str:
    joiner = " and " if lang == "en" else " en "
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + joiner + items[-1]


def summary(r: dict, lang: str) -> str:
    """The sentence under the rewind: whose parts learned, and the three means over the last 50 episodes."""
    lang = lang if lang in LANGS else "en"
    s = _SUMMARY[lang]
    yours_ = [k for k in ("world", "agent", "average") if k in r["yours"]]
    course = [k for k in ("world", "agent", "average") if k not in r["yours"]]
    if not course:
        who = s["who"][("world", "agent", "average")]
    elif not yours_:
        who = s["who"]["none"]
    else:
        mine = _listed([("your " if lang == "en" else "je ") + _PART_NAME[k] for k in yours_], lang)
        theirs = _listed([_PART_NAME[k] for k in course], lang)
        if lang == "en":
            who = s["who"]["partly"].format(yours=mine, course=theirs, verb="do" if len(course) > 1 else "does",
                                            its="their" if len(course) > 1 else "its")
        else:
            who = s["who"]["partly"].format(yours=mine, course="de " + theirs, verb="komen" if len(course) > 1 else "komt",
                                            version="versies" if len(course) > 1 else "versie")
    seeds = per_seed(r, "mc", min(LAST, r["episodes"]))
    mean = {m: number(last_mean(r["mean"][m]), 1, lang) for m in METHODS}
    return s["say"].format(who=who, lo=number(min(seeds), 1, lang), hi=number(max(seeds), 1, lang), **mean)


# --- the pictures of the learner's work ------------------------------------------------------------------------------------

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


# exercise 1: a walk through the course's world with your step
WALK = ["left", "up", "right", "down", "right", "down"] + ["right"] * 7
_MOVE = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def _falls(rows, cells: list, got: list) -> list:
    """The steps of the walk WALK into a pit, as (cell, pit) pairs: a step that cost −100 from a cell next to a pit."""
    out = []
    for k, (_, reward, _) in enumerate(got):
        (r, c), (dr, dc) = cells[k], _MOVE[WALK[k]]
        if reward <= -100 and 0 <= r + dr < len(rows) and 0 <= c + dc < len(rows[0]) and rows[r + dr][c + dc] == "X":
            out.append(((r, c), (r + dr, c + dc)))
    return out


def _walk(world_class) -> list:
    """The walk WALK with the learner's GridWorld on the course's world, as (state, reward, done) per step, until done."""
    world = world_class(WORLD)
    out = []
    for a in WALK:
        state, reward, done = world.step(a)
        r, c = state
        if not (0 <= r < len(WORLD) and 0 <= c < len(WORLD[0])) or not isinstance(reward, numbers.Real):
            raise ValueError("outside")
        out.append(((int(r), int(c)), float(reward), bool(done)))
        if done:
            break
    return out


_PICTURE_TEXT = {
    "en": {"walk_title": "Your GridWorld on the course's world",
           "walk_head": "Your `step` with the actions {acts}, from S: where your world puts the agent, as an ink line.",
           "walk_say": "Your `step` gave the rewards {rewards}; {end} A return of {total} in {n} steps.",
           "walk_done": "the last step ended the episode.",
           "walk_open": "no step ended the episode.",
           "walk_none": "Your `step` stops with an error, takes too long, or gives something other than a cell of the world and a number, so nothing can be drawn yet.",
           "walk_alt": "The course's world with the walk of your step drawn as a line.",
           "agent_title": "Your GridAgent on the course's world",
           "agent_head": "Your `GridAgent` with each of its three targets, seed 0, after {n} episodes on the course's world with the course's `GridWorld`: its best actions as small arrows, and in orange where they lead from S.",
           "agent_say": "Over the last 50 episodes of this one run, the mean return was {mc} for Monte Carlo, {sarsa} for SARSA and {q} for Q-learning.",
           "agent_none": "Your `GridAgent` stops with an error, takes too long, or gives something that is not a number, so nothing can be drawn yet.",
           "agent_alt": "Three copies of the course's world with the best actions your agent learned with each method.",
           "panel": "{m}, seed 0",
           "curve_title": "Your learning curves on the course's world",
           "curve_head": "Your `learning_curve` for each method on the course's world, with {k} seeds and {n} episodes: the mean return of the last 10 episodes.",
           "curve_say": "Over the last 50 episodes: {mc} for Monte Carlo, {sarsa} for SARSA and {q} for Q-learning.",
           "curve_none": "Your `learning_curve` stops with an error, takes too long, or gives something other than a number for every episode, so nothing can be drawn yet.",
           "curve_alt": "Three learning curves: the mean return of the last 10 episodes after every episode, for each method."},
    "nl": {"walk_title": "Je GridWorld op de wereld van de course",
           "walk_head": "Je `step` met de actions {acts}, vanaf S: waar je wereld de agent neerzet, als lijn in inkt.",
           "walk_say": "Je `step` gaf de rewards {rewards}; {end} Een return van {total} in {n} stappen.",
           "walk_done": "de laatste stap maakte een einde aan de episode.",
           "walk_open": "geen enkele stap maakte een einde aan de episode.",
           "walk_none": "Je `step` stopt met een fout, duurt te lang, of geeft iets anders dan een vakje van de wereld en een getal, dus er is nog geen afbeelding.",
           "walk_alt": "De wereld van de course met de wandeling van je step als lijn.",
           "agent_title": "Je GridAgent op de wereld van de course",
           "agent_head": "Je `GridAgent` met elk van de drie targets, seed 0, na {n} episodes op de wereld van de course met de `GridWorld` van de course: de beste actions als kleine pijlen, en in oranje waar de pijlen vanaf S heen leiden.",
           "agent_say": "Over de laatste 50 episodes van deze ene run was de gemiddelde return {mc} voor Monte Carlo, {sarsa} voor SARSA en {q} voor Q-learning.",
           "agent_none": "Je `GridAgent` stopt met een fout, duurt te lang, of geeft iets wat geen getal is, dus er is nog geen afbeelding.",
           "agent_alt": "Drie keer de wereld van de course met de beste actions die je agent met elke methode leerde.",
           "panel": "{m}, seed 0",
           "curve_title": "Je learning curves op de wereld van de course",
           "curve_head": "Je `learning_curve` voor elke methode op de wereld van de course, met {k} seeds en {n} episodes: de gemiddelde return van de laatste 10 episodes.",
           "curve_say": "Over de laatste 50 episodes: {mc} voor Monte Carlo, {sarsa} voor SARSA en {q} voor Q-learning.",
           "curve_none": "Je `learning_curve` stopt met een fout, duurt te lang, of geeft iets anders dan een getal voor elke episode, dus er is nog geen afbeelding.",
           "curve_alt": "Drie learning curves: de gemiddelde return van de laatste 10 episodes na elke episode, voor elke methode."},
}


def walk_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _PICTURE_TEXT[lang]
    run_ = seen.mine
    cls = run_.names.get("GridWorld") if run_ is not None else None
    acts = ", ".join(f"`{a}`" for a in WALK[:6]) + ", " + ("and then 7 times `right`" if lang == "en" else "en daarna 7 keer `right`")
    head = _note(w["walk_head"].format(acts=acts))
    got = _call(lambda: _walk(cls)) if callable(cls) else None
    if not isinstance(got, list) or not got:
        return yours.figure(w["walk_title"], head + _note(w["walk_none"], _LOSS_UI), alt=w["walk_title"])
    cells = [find(WORLD, "S")] + [s for s, _, _ in got]
    picture = world_svg(WORLD, [{"walk": cells, "falls": _falls(WORLD, cells, got)}], w["walk_alt"], lang)
    rewards = ", ".join(number(rw, 0, lang) for _, rw, _ in got)
    end = w["walk_done"] if got[-1][2] else w["walk_open"]
    say = w["walk_say"].format(rewards=rewards, end=end, total=number(sum(rw for _, rw, _ in got), 0, lang), n=len(got))
    return yours.figure(w["walk_title"], head + picture + world_key(lang) + _note(say), alt=w["walk_title"])


# the pictures of the lesson's text: the course's world, and a walk through it with the course's GridWorld
_TEXT_ALT = {
    "en": {"world": "The course's world: 4 rows of 8 cells. S is at the left of the third row and G at its right; the six cells between them have a row of six puddles above them and a row of six pits below them.",
           "walk": "The course's world with a walk of 13 steps from S as a line: left into the edge, up, right onto a puddle, down, right and down into a pit, then from S seven steps to the right, to G."},
    "nl": {"world": "De wereld van de course: 4 rijen van 8 vakjes. S staat links in de derde rij en G rechts; boven de zes vakjes tussen S en G ligt een rij van zes plassen, eronder een rij van zes putten.",
           "walk": "De wereld van de course met een wandeling van 13 stappen vanaf S als lijn: links tegen de rand, omhoog, rechts op een plas, omlaag, rechts en omlaag in een put, dan vanaf S zeven stappen naar rechts, naar G."},
}


def world_picture(lang: str) -> str:
    """The course's world, without a key: beside the guess."""
    lang = lang if lang in LANGS else "en"
    return world_svg(WORLD, [{}], _TEXT_ALT[lang]["world"], lang)


def idea_picture(lang: str) -> str:
    """The idea: the walk WALK through the course's world with the course's GridWorld, and the key."""
    lang = lang if lang in LANGS else "en"
    got = _walk(GridWorld)
    cells = [find(WORLD, "S")] + [state for state, _, _ in got]
    return world_svg(WORLD, [{"walk": cells, "falls": _falls(WORLD, cells, got)}], _TEXT_ALT[lang]["walk"], lang) + world_key(lang)


# exercise 2: your agent learns the course's world, one run per method
AGENT_EPISODES = 300


def _agent_runs(agent_class) -> dict:
    out = {}
    for m in METHODS:
        world = GridWorld(WORLD)
        agent = agent_class(m, EPSILON, ETA, GAMMA, np.random.default_rng(0))
        returns = []
        for _ in range(AGENT_EPISODES):
            g = episode(agent, world, LIMIT)
            if not isinstance(g, numbers.Real) or not math.isfinite(g):
                raise ValueError("not a number")
            returns.append(float(g))
        out[m] = {"returns": returns, **_snapshot(WORLD, agent)}
    return out


def agent_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w, words = _PICTURE_TEXT[lang], _WORDS[lang]
    run_ = seen.mine
    cls = run_.names.get("GridAgent") if run_ is not None else None
    head = _note(w["agent_head"].format(n=AGENT_EPISODES))
    got = _call(lambda: _agent_runs(cls)) if callable(cls) else None
    if not isinstance(got, dict):
        return yours.figure(w["agent_title"], head + _note(w["agent_none"], _LOSS_UI), alt=w["agent_title"])
    panels = [{"title": w["panel"].format(m=words[m]), "arrows": got[m]["arrows"], "way": got[m]["way"]} for m in METHODS]
    mean = {m: number(last_mean(got[m]["returns"]), 1, lang) for m in METHODS}
    body = head + world_svg(WORLD, panels, w["agent_alt"], lang) + _note(w["agent_say"].format(**mean))
    return yours.figure(w["agent_title"], body, alt=w["agent_title"])


# exercise 3: your learning curves on the course's world
CURVE_SEEDS, CURVE_EPISODES = 3, 150


def curve_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w, words = _PICTURE_TEXT[lang], _WORDS[lang]
    run_ = seen.mine
    fn = run_.names.get("learning_curve") if run_ is not None else None
    head = _note(w["curve_head"].format(k=CURVE_SEEDS, n=CURVE_EPISODES))

    def make():
        out = {}
        for m in METHODS:
            curve = [float(x) for x in fn(m, list(WORLD), CURVE_SEEDS, CURVE_EPISODES)]
            if len(curve) != CURVE_EPISODES or not all(math.isfinite(x) for x in curve):
                raise ValueError("not a curve")
            out[m] = curve
        return out

    got = _call(make, 3 * PICTURE_SECONDS) if callable(fn) else None
    if not isinstance(got, dict):
        return yours.figure(w["curve_title"], head + _note(w["curve_none"], _LOSS_UI), alt=w["curve_title"])
    p = chart.Panel((0, CURVE_EPISODES), (LOW, 0), xlabel=words["episodes"], ylabel=words["return"], yticks=[-150, -100, -50, 0])
    for m in ("mc", "q", "sarsa"):
        role, width, dashed = _STYLE[m]
        ys = [max(LOW, y) for y in smooth(got[m])]
        p.line(list(range(1, CURVE_EPISODES + 1)), ys, role, width, dashed=dashed)
    mean = {m: number(last_mean(got[m]), 1, lang) for m in METHODS}
    body = head + _chart([p], w["curve_alt"], lang, height=220) + live.key_html(curve_key(lang)) + _note(w["curve_say"].format(**mean))
    return yours.figure(w["curve_title"], body, alt=w["curve_title"])


# --- the exercises ---------------------------------------------------------------------------------------------------------

REQUIRED = ["world", "agent", "curves", "maze"]


def _close(a, b) -> bool:
    """The same values, floats to 1e-6: tuples, lists and dicts compared item by item; a list and a tuple differ."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, numbers.Real) and isinstance(b, numbers.Real):
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    return a == b


def _curve_close(a, b) -> bool:
    """A learning curve: a list of numbers, or a NumPy array of them, the same as the course's to 1e-6."""
    try:
        a = [float(x) for x in a]
    except (TypeError, ValueError):
        return False
    return _close(a, [float(x) for x in b])


def _value(v) -> str:
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_value(k)}: {_value(x)}" for k, x in v.items()) + "}"
    if isinstance(v, (list, tuple)):
        inner = ", ".join(_value(x) for x in v)
        return f"({inner}{',' if len(v) == 1 else ''})" if isinstance(v, tuple) else f"[{inner}]"
    if isinstance(v, bool):
        return repr(v)
    if isinstance(v, float):
        return repr(round(v, 6) + 0.0)
    if isinstance(v, numbers.Real):
        return repr(round(float(v), 6) + 0.0) if not float(v).is_integer() else repr(int(v))
    return repr(v)


def _show(v):
    try:
        out = _value([float(x) for x in v])
    except (TypeError, ValueError):
        out = _value(v)
    return out if len(out) <= 300 else out[:300] + "…"


def _dots_left(got, text_) -> bool:
    """A `...` of the start code left in the box: an Ellipsis in the code, not three dots inside a string."""
    try:
        tree = ast.parse(text_)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Constant) and node.value is Ellipsis for node in ast.walk(tree))


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang de `...` door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _kept(name: str) -> dict:
    return {"en": f"Your `{name}` is right. You changed `{{saved}}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
            "nl": f"Je `{name}` klopt. Je hebt `{{saved}}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."}


_SCENARIO = {"en": "With these lines after your {what}:\n\n```python\n{{scenario}}\n```\n\nyou get `{{got}}`, and it should be `{{want}}`",
             "nl": "Met deze regels na je {what}:\n\n```python\n{{scenario}}\n```\n\nkrijg je `{{got}}`, en dat moet `{{want}}` zijn"}


def _scenario(lang: str, what: str) -> str:
    return _SCENARIO[lang].format(what=what)


_S, _S_NL = _scenario("en", "class"), _scenario("nl", "class")

# exercise 1: GridWorld
TINY = ["S#G", ".W.", "X.."]
TINY_WALK = ["right", "up", "down", "down", "down", "right", "right", "up"]
_W_SCENARIOS = [
    # the example: a wall, the edge, a pit, a puddle and the goal
    f"world = GridWorld({TINY!r})\nresults = []\nfor a in {TINY_WALK!r}:\n    results.append(world.step(a))\nresults".replace("'", '"'),
    # back to the start with reset, and the edge on the left
    f"world = GridWorld({TINY!r})\nfor a in ['down', 'right', 'right', 'up']:\n    world.step(a)\n"
    "world.reset(), world.state, world.step('left')".replace("'", '"'),
    # S in the middle, a pit above the cell to its right, the bottom and the right edge, and the edge in a puddle
    "world = GridWorld(['..X.', '#S.W', '...G'])\nresults = []\n"
    "for a in ['left', 'down', 'down', 'up', 'right', 'up', 'right', 'right', 'right', 'down']:\n"
    "    results.append(world.step(a))\nresults".replace("'", '"'),
]
_W_FORGET = (_W_HEAD + _W_PIT + _W_GOAL.replace("self.state, -1, True", "(r, c), -1, True")
             + _W_WET.replace("self.state, -10", "(r, c), -10") + _W_DRY.replace("self.state, -1", "(r, c), -1"))
_W_NO_MOVE = _w(move="")
_W_NO_PIT = _w(pit="")
_W_PIT_ENDS = _w(pit=_W_PIT.replace("-100, False", "-100, True"))
_W_GOAL_OPEN = _w(goal="")
_W_DRY_ONLY = _w(wet="")
_W_ELEVEN = _w(wet=_W_WET.replace("-10,", "-11,"))

# exercise 2: GridAgent
_A_SCENARIOS = [
    # the worked example: the same two steps for each method, with the Q of the cell (0, 1) given
    "import numpy as np\n\nout = []\nfor method in [\"sarsa\", \"q\", \"mc\"]:\n"
    "    agent = GridAgent(method, 0.1, 0.5, 0.9, np.random.default_rng(0))\n"
    "    agent.q[(0, 1)] = {\"up\": -4.0, \"down\": -6.0, \"left\": -8.0, \"right\": -2.0}\n"
    "    agent.learn((0, 0), \"right\", -1, (0, 1), \"down\")\n"
    "    agent.learn((0, 1), \"down\", -10, (1, 1), \"left\")\n"
    "    agent.end()\n"
    "    out.append((agent.qs((0, 0))[\"right\"], agent.qs((0, 1))[\"down\"]))\nout",
    # Monte Carlo with a cell that comes back in the episode, and γ = 1
    "import numpy as np\n\nagent = GridAgent(\"mc\", 0.0, 0.5, 1.0, np.random.default_rng(0))\n"
    "agent.learn((0, 0), \"up\", -1, (0, 0), \"right\")\n"
    "agent.learn((0, 0), \"right\", -1, (0, 1), \"right\")\n"
    "agent.learn((0, 1), \"right\", -1, (0, 2), \"up\")\n"
    "agent.end()\nagent.qs((0, 0)), agent.qs((0, 1))",
    # Q-learning with η = 1 and γ = 0.5: the largest Q of the next cell, then back
    "import numpy as np\n\nagent = GridAgent(\"q\", 0.0, 1.0, 0.5, np.random.default_rng(0))\n"
    "agent.qs((1, 1))[\"left\"] = 4.0\n"
    "agent.learn((1, 0), \"right\", -1, (1, 1), \"up\")\n"
    "agent.learn((1, 1), \"left\", -1, (1, 0), \"right\")\n"
    "agent.qs((1, 0)), agent.qs((1, 1))",
    # SARSA after a pit: back on S, and the next action there
    "import numpy as np\n\nagent = GridAgent(\"sarsa\", 0.0, 0.5, 0.9, np.random.default_rng(0))\n"
    "agent.qs((2, 0))[\"up\"] = -3.0\n"
    "agent.learn((2, 1), \"down\", -100, (2, 0), \"up\")\n"
    "agent.learn((2, 0), \"up\", -1, (1, 0), \"right\")\n"
    "agent.qs((2, 1)), agent.qs((2, 0))",
]
_G_SARSA_MAX = _g(sarsa=_G_Q)
_G_Q_NEXT = _g(q=_G_SARSA)
_G_HERE = _g(sarsa="reward + self.gamma * self.qs(state)[nxt_action]", q="reward + self.gamma * max(self.qs(state).values())")
_G_NO_GAMMA = _g(sarsa="reward + self.qs(nxt)[nxt_action]", q="reward + max(self.qs(nxt).values())")
_G_NO_REWARD = _g(sarsa="self.gamma * self.qs(nxt)[nxt_action]", q="self.gamma * max(self.qs(nxt).values())")
_G_MC_REWARD = _g(mc="reward")
_G_MC_NO_GAMMA = _g(mc="reward + g")
_G_MC_SWAP = _g(mc="g + self.gamma * reward")

# exercise 3: learning_curve


class _Val:
    """A test value that a check message shows as code: by a name of the lesson."""

    def __init__(self, shown: str, value):
        self.shown, self.value = shown, value

    def __repr__(self) -> str:
        return self.shown

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


SMALL = ["S..", "WX.", "..G"]
CURVE_CASES = [("q", _Val("small_world", SMALL), 3, 4), ("mc", SMALL, 2, 3), ("sarsa", list(WORLD), 2, 5),
               ("q", ["S.G"], 1, 2), ("sarsa", list(TINY), 4, 3)]
_C_OUTSIDE = _C_HEAD.replace("    runs = []\n", "    runs = []\n    returns = []\n") + _C_LOOP + _C_PLAY + _C_KEEP + _C_TAIL
_C_ONE = _c(_C_NEW + "        returns.append(episode(agent, world, LIMIT))\n" + _C_KEEP)
_C_FRESH = _c(_C_NEW + _C_LOOP + "            agent = GridAgent(method, 0.1, 0.2, 1.0, np.random.default_rng(seed))\n" + _C_PLAY + _C_KEEP)
_C_NO_KEEP = _c(_C_NEW + _C_LOOP + _C_PLAY)

EXERCISES = {
    "world": ClassBox(
        name="world",
        picture=walk_picture,
        start=_W_START,
        solution=_W,
        scenarios=_W_SCENARIOS,
        compare=_close,
        first=[_DOTS],
        save_as=("agent", "gridworld"),
        mistakes=[
            Mistake("forget", wrong=_W_FORGET,
                    en=_S + ": your `step` gives back the new cell, but the world does not keep it, so the next step starts from the old cell again. Where does `step` store where the agent is now?",
                    nl=_S_NL + ": je `step` geeft het nieuwe vakje terug, maar de wereld onthoudt het niet, dus begint de volgende stap weer bij het oude vakje. Waar bewaart `step` waar de agent nu staat?"),
            Mistake("no-move", wrong=_W_NO_MOVE,
                    en=_S + ": your agent never leaves S. `step` gives back `self.state`; where does `self.state` become the cell of `r` and `c`?",
                    nl=_S_NL + ": je agent komt nooit van S af. `step` geeft `self.state` terug; waar wordt `self.state` het vakje van `r` en `c`?"),
            Mistake("no-pit", wrong=_W_NO_PIT,
                    en=_S + ": a step into the pit costs −1, and the agent stands in the pit. A pit costs −100 and puts the agent back on S; is there a rule for the sign `\"X\"`, before the agent moves?",
                    nl=_S_NL + ": een stap in de put kost −1, en de agent staat in de put. Een put kost −100 en zet de agent terug op S; is er een regel voor het teken `\"X\"`, voordat de agent zich verplaatst?"),
            Mistake("pit-ends", wrong=_W_PIT_ENDS,
                    en=_S + ": a step into the pit ends your episode. In this world the agent goes back to S and goes on, as on the cliff; what should the third value be after a pit?",
                    nl=_S_NL + ": een stap in de put maakt een einde aan je episode. In deze wereld gaat de agent terug naar S en gaat de agent verder, zoals op het klif; wat hoort de derde waarde na een put te zijn?"),
            Mistake("goal-open", wrong=_W_GOAL_OPEN,
                    en=_S + ": the episode never ends, not even on G. Which of the three values says that the episode is over, and when should it be `True`?",
                    nl=_S_NL + ": de episode houdt nooit op, ook niet op G. Welke van de drie waarden zegt of de episode voorbij is, en wanneer hoort deze waarde `True` te zijn?"),
            Mistake("dry", wrong=_W_DRY_ONLY,
                    en=_S + ": a step onto the puddle costs −1, as every other step does. What does a step onto `\"W\"` cost?",
                    nl=_S_NL + ": een stap op de plas kost −1, net als elke andere stap. Wat kost een stap op `\"W\"`?"),
            Mistake("eleven", wrong=_W_ELEVEN,
                    en=_S + ": a step onto the puddle costs −11, the −1 of a step and −10 on top of it. In this world a step onto a puddle costs −10 in all; what does your rule for `\"W\"` give back?",
                    nl=_S_NL + ": een stap op de plas kost −11, de −1 van een stap met −10 erbovenop. In deze wereld kost een stap op een plas in totaal −10; wat geeft je regel voor `\"W\"` terug?"),
        ],
        texts={
            **_SAME_TEXTS,
            "empty": {"en": "Replace the `...` with the rules of the signs: a pit first, then the move, then the goal, a puddle and every other cell. Each rule gives back where the agent is, the reward, and whether the episode is over.",
                      "nl": "Vervang de `...` door de regels van de tekens: eerst een put, dan de verplaatsing, dan het doel, een plas en elk ander vakje. Elke regel geeft terug waar de agent staat, de reward, en of de episode voorbij is."},
            "wrong": {"en": _S + ". Does a pit put the agent back on S with −100, does the world keep the cell the agent moves to, does the goal end the episode, and does a puddle cost −10?",
                      "nl": _S_NL + ". Zet een put de agent terug op S met −100, onthoudt de wereld het vakje waar de agent heen gaat, maakt het doel een einde aan de episode, en kost een plas −10?"},
            "almost": {"en": "Right for the example. But " + _S[0].lower() + _S[1:] + ". Does your `step` also work after `reset`, when S is not in a corner, and at the bottom and the right edge?",
                       "nl": "Goed voor het voorbeeld. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Werkt je `step` ook na `reset`, als S niet in een hoek staat, en aan de onderkant en de rechterrand?"},
            "correct": {"en": "Right: eight steps past a wall, the edge, a pit and a puddle to the goal. Your `GridWorld` is now in your project, as `{saved}`. In the picture your `step` takes the agent through the course's world; where does it put the agent after the pit?",
                        "nl": "Goed: acht stappen langs een muur, de rand, een put en een plas naar het doel. Je `GridWorld` staat nu in je project, als `{saved}`. In de afbeelding stuurt je `step` de agent door de wereld van de course; waar zet je `step` de agent neer na de put?"},
            "kept": _kept("GridWorld"),
        },
    ),
    "agent": ClassBox(
        name="agent",
        picture=agent_picture,
        start=_G_START,
        solution=_G,
        scenarios=_A_SCENARIOS,
        compare=_close,
        first=[_DOTS],
        save_as=("agent", "gridagent"),
        mistakes=[
            Mistake("sarsa-max", wrong=_G_SARSA_MAX,
                    en=_S + ": your SARSA takes the largest Q in `nxt`, as Q-learning does. SARSA learns from the action it will really take next; which argument of `learn` is that action?",
                    nl=_S_NL + ": je SARSA neemt de grootste Q in `nxt`, zoals Q-learning. SARSA leert van de action die de agent echt als volgende neemt; welk argument van `learn` is deze action?"),
            Mistake("q-next", wrong=_G_Q_NEXT,
                    en=_S + ": your Q-learning takes the Q of `nxt_action`, as SARSA does. Q-learning learns as if the agent takes its best action next, whatever it really takes; which Q in `nxt` is that?",
                    nl=_S_NL + ": je Q-learning neemt de Q van `nxt_action`, zoals SARSA. Q-learning leert alsof de agent als volgende de beste action neemt, wat de agent ook echt neemt; welke Q in `nxt` is dat?"),
            Mistake("here", wrong=_G_HERE,
                    en=_S + ": your targets look at the Q of `state`, the cell where the step started. A target looks one step ahead; in which cell did the step end?",
                    nl=_S_NL + ": je targets kijken naar de Q van `state`, het vakje waar de stap begon. Een target kijkt één stap vooruit; in welk vakje eindigde de stap?"),
            Mistake("no-gamma", wrong=_G_NO_GAMMA,
                    en=_S + ": your SARSA and Q-learning leave out `self.gamma`, so the Q of the next cell counts in full, also with γ = 0.9. Where does γ go in a target?",
                    nl=_S_NL + ": je SARSA en Q-learning laten `self.gamma` weg, dus telt de Q van het volgende vakje helemaal mee, ook met γ = 0,9. Waar hoort γ in een target?"),
            Mistake("no-reward", wrong=_G_NO_REWARD,
                    en=_S + ": your targets leave out the reward of the step. A target is what the step gave plus what comes after it; what did this step give?",
                    nl=_S_NL + ": je targets laten de reward van de stap weg. Een target is wat de stap opleverde plus wat erna komt; wat leverde deze stap op?"),
            Mistake("mc-reward", wrong=_G_MC_REWARD,
                    en=_S + ": your Monte Carlo moves each Q toward the reward of its own step only. The return is that reward plus everything after it; what does `g` hold just before this line?",
                    nl=_S_NL + ": je Monte Carlo schuift elke Q alleen richting de reward van de eigen stap. De return is deze reward plus alles erna; wat bevat `g` vlak voor deze regel?"),
            Mistake("mc-no-gamma", wrong=_G_MC_NO_GAMMA,
                    en=_S + ": your Monte Carlo adds the return after a step in full, also with γ = 0.9. Where does γ go?",
                    nl=_S_NL + ": je Monte Carlo telt de return na een stap helemaal mee, ook met γ = 0,9. Waar hoort γ?"),
            Mistake("mc-swap", wrong=_G_MC_SWAP,
                    en=_S + ": your Monte Carlo multiplies the reward of the step by γ, and not the return after it. Which of the two lies one step further away?",
                    nl=_S_NL + ": je Monte Carlo vermenigvuldigt de reward van de stap met γ, en niet de return erna. Welke van de twee ligt één stap verder weg?"),
        ],
        texts={
            **_SAME_TEXTS,
            "empty": {"en": "Replace the three `...` with the targets: in `learn` the target of SARSA and the target of Q-learning, in `end` the return `g` of Monte Carlo.",
                      "nl": "Vervang de drie `...` door de targets: in `learn` het target van SARSA en het target van Q-learning, in `end` de return `g` van Monte Carlo."},
            "wrong": {"en": _S + ". Is each target the reward of the step plus γ times what comes after it: for SARSA the Q of the next action, for Q-learning the largest Q of the next cell, for Monte Carlo the return after the step?",
                      "nl": _S_NL + ". Is elk target de reward van de stap plus γ keer wat erna komt: voor SARSA de Q van de volgende action, voor Q-learning de grootste Q van het volgende vakje, voor Monte Carlo de return na de stap?"},
            "almost": {"en": "Right for the example. But " + _S[0].lower() + _S[1:] + ". Does your agent also work with γ = 1, when a cell comes back in an episode, and with η = 1?",
                       "nl": "Goed voor het voorbeeld. Maar " + _S_NL[0].lower() + _S_NL[1:] + ". Werkt je agent ook met γ = 1, als een vakje terugkomt in een episode, en met η = 1?"},
            "correct": {"en": "Right: after the same two steps, SARSA moves the Q of right in (0, 0) to −3.2, Q-learning to −1.4 and Monte Carlo to −5.0. Your `GridAgent` is now in your project, as `{saved}`. In the picture it learns the course's world with each target; which way does each orange line take?",
                        "nl": "Goed: na dezelfde twee stappen schuift SARSA de Q van right in (0, 0) naar −3,2, Q-learning naar −1,4 en Monte Carlo naar −5,0. Je `GridAgent` staat nu in je project, als `{saved}`. In de afbeelding leert je agent de wereld van de course met elk target; welke weg neemt elke oranje lijn?"},
            "kept": _kept("GridAgent"),
        },
    ),
    "curves": FunctionBox(
        name="curves",
        picture=curve_picture,
        function="learning_curve",
        start=_C_START,
        solution=_C,
        show=_show,
        compare=_curve_close,
        visible=CURVE_CASES[:1],
        hidden=CURVE_CASES[1:],
        first=[_DOTS],
        save_as=("agent", "curves"),
        mistakes=[
            Mistake("empty-runs", error=lambda e: isinstance(e, (IndexError, ZeroDivisionError)),
                    en="Python stops in `average`: `runs` is still empty when your function gives it to `average`. Does your loop add the list of returns of each seed to `runs`?",
                    nl="Python stopt in `average`: `runs` is nog leeg als je functie de lijst aan `average` geeft. Voegt je loop de lijst met returns van elke seed toe aan `runs`?"),
            Mistake("outside", wrong=_C_OUTSIDE,
                    en="`{call}` gives {got}, and it should give {want}: all your seeds add their returns to one list, so each run holds the returns of every seed one after the other. Where should `returns = []` go, so that each seed starts a list of its own?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: alle seeds voegen de returns toe aan één lijst, dus bevat elke run de returns van alle seeds achter elkaar. Waar hoort `returns = []`, zodat elke seed een eigen lijst begint?"),
            Mistake("one-episode", wrong=_C_ONE,
                    en="`{call}` gives {got}, and it should give {want}: each agent plays one episode, so your curve has one number. How many episodes should each agent play?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke agent speelt één episode, dus heeft je curve één getal. Hoeveel episodes hoort elke agent te spelen?"),
            Mistake("fresh", wrong=_C_FRESH,
                    en="`{call}` gives {got}, and it should give {want}: your agent starts again from nothing at every episode, so it never gets better. How often should each seed make an agent?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je agent begint bij elke episode opnieuw bij nul, dus wordt de agent nooit beter. Hoe vaak hoort elke seed een agent te maken?"),
        ],
        texts={
            **_SAME_TEXTS,
            "empty": {"en": "Replace the `...` with the run of one seed: a new list of returns, one `episode` after the other for every episode, and then that list into `runs`.",
                      "nl": "Vervang de `...` door de run van één seed: een nieuwe lijst met returns, één `episode` na de andere voor elke episode, en dan deze lijst in `runs`."},
            "arguments": {"en": "The check calls `{call}`, with four values, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(method, rows, seeds, episodes)`?",
                          "nl": "De check roept `{call}` aan, met vier waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(method, rows, seeds, episodes)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does each seed play all its episodes with one agent of its own, keep their returns in a list of its own, and put that list into `runs`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Speelt elke seed alle episodes met één eigen agent, bewaart elke seed de returns in een eigen lijst, en gaat deze lijst in `runs`?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for every method, any number of seeds and episodes, and any world?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie voor elke methode, elk aantal seeds en episodes, en elke wereld?"},
            "no_return": {"en": "With `{call}`, your function gives back `None`, and it should give a list. Does it still end with `return average(runs)`?",
                          "nl": "Met `{call}` geeft je functie `None` terug, en het moet een lijst zijn. Eindigt je functie nog met `return average(runs)`?"},
            "correct": {"en": "Right: three seeds of four episodes, averaged at every episode. Your `learning_curve` is now in your project, as `{saved}`, next to your `average`. In the picture it measures all three methods on the course's world; which curve rises fastest?",
                        "nl": "Goed: drie seeds van vier episodes, gemiddeld bij elke episode. Je `learning_curve` staat nu in je project, als `{saved}`, naast je `average`. In de afbeelding meet je functie alle drie de methoden op de wereld van de course; welke curve stijgt het snelst?"},
            "kept": _kept("learning_curve"),
        },
    ),
}

CASES = [
    ("world", _W_START, "empty", None),
    ("world", _W, "correct", None),
    ("world", _W_HEAD + _W_PIT + "        ...\n", "wrong", "dots-left"),
    ("world", _W_FORGET, "wrong", "forget"),
    ("world", _W_NO_MOVE, "wrong", "no-move"),
    ("world", _W_NO_PIT, "wrong", "no-pit"),
    ("world", _W_PIT_ENDS, "wrong", "pit-ends"),
    ("world", _W_GOAL_OPEN, "wrong", "goal-open"),
    ("world", _W_DRY_ONLY, "wrong", "dry"),
    ("world", _W_ELEVEN, "wrong", "eleven"),
    ("world", _W_HEAD + '        if sign == "X":\n            self.state = self.start\n            return self.start, -100, False\n'
     '        self.state = (r, c)\n        reward = -1\n        if sign == "W":\n            reward = -10\n'
     '        return self.state, reward, sign == "G"\n', "correct", None),
    ("world", _W_HEAD.replace("        sign = self.rows[r][c]\n", "        sign = self.rows[r][c]\n        old = self.state\n") + _W_PIT + _W_MOVE + _W_GOAL
     + '        if sign == "W" and self.state != old:\n            return self.state, -10, False\n' + _W_DRY, "almost", None),
    ("world", _w(wet=_W_WET.replace("-10,", "-5,")), "wrong", None),
    ("agent", _G_START, "empty", None),
    ("agent", _G, "correct", None),
    ("agent", _g(mc="..."), "wrong", "dots-left"),
    ("agent", _G_SARSA_MAX, "wrong", "sarsa-max"),
    ("agent", _G_Q_NEXT, "wrong", "q-next"),
    ("agent", _G_HERE, "wrong", "here"),
    ("agent", _G_NO_GAMMA, "wrong", "no-gamma"),
    ("agent", _G_NO_REWARD, "wrong", "no-reward"),
    ("agent", _G_MC_REWARD, "wrong", "mc-reward"),
    ("agent", _G_MC_NO_GAMMA, "wrong", "mc-no-gamma"),
    ("agent", _G_MC_SWAP, "wrong", "mc-swap"),
    ("agent", _g(q="reward + self.gamma * max(self.q[nxt].values())"), "error", None),
    ("agent", _g(sarsa="reward + self.gamma * self.qs(nxt).get(nxt_action)"), "correct", None),
    ("agent", _g(q="reward + self.gamma * max([self.qs(nxt)[a] for a in ACTIONS])"), "correct", None),
    ("agent", _g(mc="reward + self.gamma * g + 0.5"), "wrong", None),
    # 0.9 in place of self.gamma: right for the example, where γ = 0.9, not for the cases with γ = 1 and γ = 0.5
    ("agent", _g(sarsa="reward + 0.9 * self.qs(nxt)[nxt_action]", q="reward + 0.9 * max(self.qs(nxt).values())",
                 mc="reward + 0.9 * g"), "almost", None),
    ("curves", _C_START, "empty", None),
    ("curves", _C, "correct", None),
    ("curves", _c(_C_NEW + _C_LOOP + "            ...\n" + _C_KEEP), "wrong", "dots-left"),
    ("curves", _C_NO_KEEP, "error", "empty-runs"),
    ("curves", _C_OUTSIDE, "wrong", "outside"),
    ("curves", _C_ONE, "wrong", "one-episode"),
    ("curves", _C_FRESH, "wrong", "fresh"),
    ("curves", _c(_C_NEW + _C_LOOP + _C_PLAY + _C_KEEP).replace("    return average(runs)\n", ""), "wrong", "no-return"),
    ("curves", _c("        runs.append([episode(agent, world, LIMIT) for e in range(episodes)])\n"), "correct", None),
    ("curves", _c(_C_NEW + _C_LOOP + "            returns.append(episode(agent, world, 50))\n" + _C_KEEP), "almost", None),
]
