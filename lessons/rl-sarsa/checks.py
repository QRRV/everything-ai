# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rl-sarsa: on a grid of 4 by 12 with a cliff along the bottom, SARSA learns Q(s, a) after every
step toward the reward plus the Q of the action it will really take next, and Q-learning toward the reward plus the
largest Q of the next state. With ε-greedy steps, SARSA learns the safe way along the top and Q-learning the short way
along the edge, which it sometimes falls off. The cliff, the course's runs of both methods side by side, sarsa_update
and sarsa_episode, the functions the learner writes, and their pictures; the play block and the demo.

The cliff and SARSA as in Sutton and Barto (Reinforcement Learning: An Introduction, 2nd edition, 2018, section 6.4 and
example 6.6); the code, the runs and the pictures are the course's own."""
import ast
import copy
import html
import math
import random
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from everything_ai import chart, live, yours
from everything_ai.check import FunctionBox, Mistake

LANGS = ("en", "nl")
ROWS, COLS = 4, 12
START, GOAL = (3, 0), (3, 11)
CLIFF = frozenset((3, c) for c in range(1, COLS - 1))   # the bottom row between S and G
ACTIONS = ["up", "down", "left", "right"]
MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
FALL = -100         # a step into the cliff: this reward, and back to S
ETA = 0.5           # the learning rate of the play block, for both methods
EPSILON = 0.1       # the chance of a random action, for both methods
GAMMA = 1.0         # every step counts the same
EPISODES = 500
RUNS = 50           # the runs of the play block and of the demo, side by side
EPSILONS = (0.2, 0.1, 0.05, 0.01)   # the demo's menu
LOOKS = [0, 1, 10, 20, 50, 100, 200, 300, 400, 500]
WINDOW = 10         # the chart shows the mean return of the last 10 episodes
LOW = -100          # the bottom of the chart: the first episodes, far lower, stay at its edge
METHODS = ("sarsa", "q")
MAX_STEPS = 100_000   # an episode that takes longer than this has gone wrong


class Named(dict):
    """A Q-table with a name, so that a check can say which table it gave the learner's function."""

    def __init__(self, entries, name: str):
        super().__init__(entries)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return Named({s: dict(qs) for s, qs in self.items()}, self.name)


# --- the cliff and the course's code -----------------------------------------------------------------------------------

def step(state, action):
    """The cliff: one step from `state`. At the edge of the grid the agent stays where it is; a step into the cliff
    gives −100 and puts the agent back on S; every other step gives −1. Gives back where the agent is now, and the
    reward."""
    r = min(max(state[0] + MOVES[action][0], 0), ROWS - 1)
    c = min(max(state[1] + MOVES[action][1], 0), COLS - 1)
    if (r, c) in CLIFF:
        return START, FALL
    return (r, c), -1


def new_q() -> dict:
    """A Q-table of zeros: a dict per state with a Q per action, as in Q-learning."""
    return {(r, c): {a: 0.0 for a in ACTIONS} for r in range(ROWS) for c in range(COLS)}


def choose(Q, state, epsilon, rng):
    """ε-greedy, as the learner wrote it in Q-learning: a random action with probability ε, else the action with the
    largest Q, the first one where several tie."""
    actions = []
    qs = []
    for a in Q[state]:
        actions.append(a)
        qs.append(Q[state][a])
    if rng.random() < epsilon:
        return actions[rng.integers(0, 4)]
    return actions[np.argmax(qs)]


def sarsa_update(Q, state, action, reward, nxt, nxt_action, eta, gamma):
    """The course's SARSA: the new Q of the state and action, moved toward the reward plus the Q of the action the
    agent will take next."""
    target = reward + gamma * Q[nxt][nxt_action]
    return Q[state][action] + eta * (target - Q[state][action])


def q_update(Q, state, action, reward, nxt, nxt_action, eta, gamma):
    """Q-learning in the same shape: the target takes the largest Q of the next state, whatever comes next."""
    target = reward + gamma * max(Q[nxt].values())
    return Q[state][action] + eta * (target - Q[state][action])


def sarsa_episode(Q, eta, epsilon, gamma, rng):
    """The course's episode of SARSA: the solution of exercise 2."""
    state = START
    action = choose(Q, state, epsilon, rng)
    total = 0
    while state != GOAL:
        nxt, reward = step(state, action)
        total = total + reward
        nxt_action = choose(Q, nxt, epsilon, rng)
        Q[state][action] = sarsa_update(Q, state, action, reward, nxt, nxt_action, eta, gamma)
        state = nxt
        action = nxt_action
    return total


def _pick(Q, state, epsilon, rnd) -> str:
    """ε-greedy as choose does it, with Python's own random numbers: much faster for the course's many runs."""
    if epsilon > 0 and rnd.random() < epsilon:
        return ACTIONS[int(rnd.random() * 4)]
    qs = Q[state]
    best, top = "up", qs["up"]
    for a in ("down", "left", "right"):
        if qs[a] > top:
            best, top = a, qs[a]
    return best


def arrows(Q) -> dict:
    """The best action of every state, as small arrows: all actions with the largest Q, none where all four tie, such as
    a state the agent has never left."""
    out = {}
    for s, qs in Q.items():
        if s in CLIFF or s == GOAL:
            continue
        top = max(qs.values())
        best = [a for a in ACTIONS if qs[a] > top - 1e-9]
        if len(best) < len(ACTIONS):
            out[s] = best
    return out


def way(Q) -> list | None:
    """The cells from S to G when the agent always takes the best action, the first one where several tie; None when
    that way falls into the cliff or comes back to a cell it has seen."""
    s, cells = START, [START]
    for _ in range(ROWS * COLS):
        nxt, reward = step(s, _pick(Q, s, 0, None))
        if reward == FALL or nxt in cells:
            return None
        cells.append(nxt)
        s = nxt
        if s == GOAL:
            return cells
    return None


class _Bad(Exception):
    """The learner's function gave something that is not a number, or an episode never ended."""


def _fine(x) -> bool:
    return isinstance(x, (int, float, np.floating, np.integer)) and not isinstance(x, bool) and math.isfinite(x)


def _episode(Q, fn, epsilon, rnd, eta=ETA, gamma=GAMMA) -> tuple:
    """One episode with `fn` as the update: its return and how often the agent fell."""
    s = START
    a = _pick(Q, s, epsilon, rnd)
    total, falls, steps = 0, 0, 0
    while s != GOAL:
        nxt, reward = step(s, a)
        total += reward
        falls += reward == FALL
        b = _pick(Q, nxt, epsilon, rnd)
        v = fn(Q, s, a, reward, nxt, b, eta, gamma)
        if not _fine(v):
            raise _Bad
        Q[s][a] = float(v)
        s, a = nxt, b
        steps += 1
        if steps > MAX_STEPS:
            raise _Bad
    return total, falls


def run(sarsa_fn=sarsa_update, epsilon: float = EPSILON, show=None, limit: float = 60.0, runs: int = RUNS) -> dict | str:
    """SARSA and Q-learning side by side, `runs` runs of 500 episodes each, with run k of both methods on random seed k:
    the return and the falls of every episode of every run, the mean return over the runs, and the best actions of run
    1 at every stop of LOOKS. `show(n, run)` hears every episode; run["now"] holds the Q-tables of run 1 as they are.
    "sarsa" when sarsa_fn stops with an error or gives something that is not a number, "slow" after `limit` seconds."""
    fns = {"sarsa": sarsa_fn, "q": q_update}
    tables = {m: [new_q() for _ in range(runs)] for m in METHODS}
    rnds = {m: [random.Random(k) for k in range(runs)] for m in METHODS}
    out = {"epsilon": epsilon, "runs": runs,
           "each": {m: [[] for _ in range(runs)] for m in METHODS},
           "falls": {m: [[] for _ in range(runs)] for m in METHODS},
           "mean": {m: [] for m in METHODS},
           "first": {0: {m: ({}, None) for m in METHODS}},
           "now": {m: tables[m][0] for m in METHODS}}
    begin = time.perf_counter()
    for n in range(1, EPISODES + 1):
        for m in METHODS:
            for k in range(runs):
                try:
                    total, falls = _episode(tables[m][k], fns[m], epsilon, rnds[m][k])
                except Exception:   # noqa: BLE001   the demo says which function to look at
                    if m == "sarsa":
                        return "sarsa"
                    raise
                out["each"][m][k].append(total)
                out["falls"][m][k].append(falls)
            out["mean"][m].append(sum(e[-1] for e in out["each"][m]) / runs)
        if n in LOOKS:
            out["first"][n] = {m: (arrows(tables[m][0]), way(tables[m][0])) for m in METHODS}
        if show is not None:
            show(n, out)
        if time.perf_counter() - begin > limit:
            return "slow"
    out["ways"] = {m: [way(Q) for Q in tables[m]] for m in METHODS}
    return out


@lru_cache(maxsize=1)
def play_run() -> dict:
    """The run of the play block: both methods with ε = 0.1 and η = 0.5."""
    return run()


def course_returns() -> tuple:
    """The NumPy cell: the return of every episode of every run, for SARSA and for Q-learning, as two arrays of shape
    (50, 500): a row per run, a column per episode."""
    r = play_run()
    return np.array(r["each"]["sarsa"], dtype=float), np.array(r["each"]["q"], dtype=float)


def course_falls() -> tuple:
    """The NumPy cell: how often the agent fell in every episode of every run, as two arrays of shape (50, 500)."""
    r = play_run()
    return np.array(r["falls"]["sarsa"]), np.array(r["falls"]["q"])


@lru_cache(maxsize=None)
def demo_run(epsilon: float) -> dict:
    """The demo with the course's functions: what the text after the demo says."""
    return run(epsilon=epsilon)


def smooth(curve: list) -> list:
    """For every episode, the mean of the last WINDOW values up to it: the line of the chart."""
    out = []
    for e in range(1, len(curve) + 1):
        part = curve[max(0, e - WINDOW):e]
        out.append(sum(part) / len(part))
    return out


def last_mean(r: dict, method: str, n: int, k: int = WINDOW) -> float:
    """The mean return over the runs of the last k episodes up to episode n."""
    part = r["mean"][method][max(0, n - k):n]
    return sum(part) / len(part)


# --- drawing ------------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem"
_TOWARD = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
CELL = 26


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


def _dec(x: float, lang: str, digits: int = 2) -> str:
    out = f"{x:,.{digits}f}"
    if out.startswith("-") and float(out.replace(",", "")) == 0:
        out = out[1:]
    out = out.replace("-", "−")
    return out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch, and a real minus sign."""
    return _dec(x, lang if lang in LANGS else "en", digits)


def eps_text(e: float, lang: str) -> str:
    """ε as the lesson writes it: 0.1, or 0,1 in Dutch."""
    text_ = f"{e:g}"
    return text_.replace(".", ",") if lang == "nl" else text_


def eps_options(lang: str) -> dict:
    """The choices of the demo's menu: ε as written, and its value."""
    return {eps_text(e, lang): e for e in EPSILONS}


_CLIFF_WORDS = {"en": "cliff: −100, back to S", "nl": "klif: −100, terug naar S"}


def cliff_svg(panels: list, alt: str, lang: str, cell: int = CELL) -> str:
    """One or two cliffs, one under the other, in one picture. Each panel is a dict: `title`, `arrows` {cell: [actions]}
    as small ink triangles, and `way`, the cells from S to G drawn as an orange line (what the agent learned), or None.
    The cliff is one dark band with its words; S and G are marked."""
    lang = lang if lang in LANGS else "en"
    W = COLS * cell + 2
    halo = "paint-order:stroke;stroke:var(--ea-page);stroke-width:3px;stroke-linejoin:round"
    parts, y0 = [], 0
    for p in panels:
        parts.append(f'<text x="1" y="{y0 + 14}" style="font-size:13.5px;font-weight:600;fill:var(--ea-ink)">'
                     f'{html.escape(p["title"])}</text>')
        top = y0 + 22
        for r in range(ROWS):
            for c in range(COLS):
                if (r, c) in CLIFF:
                    continue
                fill = "var(--ea-data-wash)" if (r, c) == GOAL else "var(--ea-page)"
                parts.append(f'<rect x="{1 + c * cell}" y="{top + r * cell}" width="{cell}" height="{cell}" '
                             f'style="fill:{fill};stroke:var(--ea-line);stroke-width:1"/>')
        x0, x1 = 1 + cell, 1 + (COLS - 1) * cell
        yb = top + (ROWS - 1) * cell
        parts.append(f'<rect x="{x0}" y="{yb}" width="{x1 - x0}" height="{cell}" style="fill:var(--ea-data);'
                     f'stroke:var(--ea-line);stroke-width:1"/>')
        parts.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{yb + cell / 2 + 4:.1f}" style="font-size:11.5px;font-weight:600;'
                     f'fill:var(--ea-page);text-anchor:middle">{html.escape(_CLIFF_WORDS[lang])}</text>')
        for (r, c), sign in ((START, "S"), (GOAL, "G")):
            parts.append(f'<text x="{1 + c * cell + cell / 2:.1f}" y="{top + r * cell + cell / 2 + 4.5:.1f}" '
                         f'style="font-size:13px;font-weight:700;fill:var(--ea-ink-2);text-anchor:middle;{halo}">{sign}</text>')
        cells = p.get("way")
        if cells:
            points = " ".join(f"{1 + c * cell + cell / 2:.1f},{top + r * cell + cell / 2:.1f}" for r, c in cells)
            parts.append(f'<polyline points="{points}" style="fill:none;stroke:var(--ea-param);stroke-width:3;'
                         f'stroke-linecap:round;stroke-linejoin:round;opacity:0.8"/>')
        for (r, c), acts in sorted(p.get("arrows", {}).items()):
            if (r, c) in (START, GOAL):
                continue
            mx, my = 1 + c * cell + cell / 2, top + r * cell + cell / 2
            for a in acts:
                dx, dy = _TOWARD[a]
                tip = (mx + dx * (cell / 2 - 2), my + dy * (cell / 2 - 2))
                base = (mx + dx * (cell / 2 - 7.5), my + dy * (cell / 2 - 7.5))
                side = (dy * 3.6, dx * 3.6)
                corners = [tip, (base[0] + side[0], base[1] + side[1]), (base[0] - side[0], base[1] - side[1])]
                parts.append('<polygon points="' + " ".join(f"{px:.1f},{py:.1f}" for px, py in corners)
                             + '" style="fill:var(--ea-ink-2)"/>')
        y0 = top + ROWS * cell + 12
    H = y0 - 10
    svg = (f'<svg viewBox="0 0 {W} {H}" aria-hidden="true" style="flex:1 1 0;width:100%;min-width:min(100%,{W}px);'
           f'max-width:{W}px;height:auto;font-family:var(--ea-font-ui)">{"".join(parts)}</svg>')
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(alt)}" style="margin:0.25rem 0 0.5rem;'
            f'max-width:min(100%,{W}px);display:flex">{svg}</figure>')


_WORDS = {
    "en": {"episodes": "episodes", "return": "mean return", "sarsa": "SARSA", "q": "Q-learning"},
    "nl": {"episodes": "episodes", "return": "gemiddelde return", "sarsa": "SARSA", "q": "Q-learning"},
}


def _svg(panels: list, alt: str, lang: str, width: int = 330, height: int = 250) -> str:
    out = chart.figure(panels, alt=alt, width=width, height=height)
    return chart.with_decimal_comma(out) if lang == "nl" else out


def _curve_panel(r: dict, n: int, lang: str, title: str) -> chart.Panel:
    """The mean return of the last 10 episodes after every episode, over the runs: SARSA solid, Q-learning dashed, with
    a line at n."""
    w = _WORDS[lang]
    p = chart.Panel((0, EPISODES), (LOW, 0), title=title, xlabel=w["episodes"], ylabel=w["return"],
                    yticks=[-100, -75, -50, -25, 0])
    xs = list(range(1, len(r["mean"]["q"]) + 1))
    for m, width, dashed in (("q", 1.6, True), ("sarsa", 2.2, False)):
        ys = [max(LOW, v) for v in smooth(r["mean"][m])]
        if xs:
            p.line(xs, ys, "ink", width, dashed=dashed)
    if 1 <= n <= len(xs):
        p.line([n, n], [LOW, 0], "axis", 1.2)
        p.points([n, n], [max(LOW, last_mean(r, "sarsa", n)), max(LOW, last_mean(r, "q", n))], "ink", r=3.6)
    return p


def curve_key(lang: str) -> list:
    w = _WORDS[lang if lang in LANGS else "en"]
    return [("ink", False, w["sarsa"]), ("ink", True, w["q"])]


def _steps_text(cells, lang: str) -> str:
    return str(len(cells) - 1)


_PLAY_TEXT = {
    "en": {"grid": "Run 1 after {n} {word}", "one": "episode", "many": "episodes", "curve": "Mean of 50 runs",
           "sarsa": "SARSA", "q": "Q-learning",
           "grid_alt": "Two copies of the cliff, SARSA above and Q-learning below: the best action of every state in run 1 after {n} {word} as small arrows, and where they lead from S, if they lead to G, as an orange line.",
           "curve_alt": "The mean return of the last 10 episodes over 50 runs, after 1 to 500 episodes: SARSA solid, Q-learning dashed, with a line at {n}.",
           "say0": "Before the first episode every Q is 0, so no state has a best action yet.",
           "say": "After {n} {word}: over the 50 runs, the last 10 episodes gave SARSA a mean return of {s} and Q-learning {q}. In run 1, {ways}",
           "say1": "After 1 episode: over the 50 runs, the first episode gave SARSA a return of {s} on average and Q-learning {q}: knowing nothing, both agents wander for hundreds of steps and fall many times. In run 1, {ways}",
           "both": "the arrows of SARSA lead from S to G in {a} steps, and those of Q-learning in {b}.",
           "only_s": "the arrows of SARSA lead from S to G in {a} steps; those of Q-learning do not lead to G yet.",
           "only_q": "the arrows of Q-learning lead from S to G in {b} steps; those of SARSA do not lead to G yet.",
           "none": "the arrows of neither method lead from S to G yet.",
           "note": "The chart: the mean return of the last 10 episodes over the 50 runs, SARSA solid, Q-learning dashed. The arrows: the best action of each state in run 1; in orange, the way they lead from S, once they lead to G. Both methods learn with ε = 0.1 and η = 0.5."},
    "nl": {"grid": "Run 1 na {n} {word}", "one": "episode", "many": "episodes", "curve": "Gemiddelde van 50 runs",
           "sarsa": "SARSA", "q": "Q-learning",
           "grid_alt": "Twee keer het klif, SARSA boven en Q-learning onder: de beste action van elke state in run 1 na {n} {word} als kleine pijlen, en, als de pijlen naar G leiden, de weg vanaf S als oranje lijn.",
           "curve_alt": "De gemiddelde return van de laatste 10 episodes over 50 runs, na 1 tot 500 episodes: SARSA doorgetrokken, Q-learning gestreept, met een lijn bij {n}.",
           "say0": "Bij de start is elke Q 0, dus heeft nog geen enkele state een beste action.",
           "say": "Na {n} {word}: over de 50 runs gaven de laatste 10 episodes SARSA een gemiddelde return van {s} en Q-learning {q}. In run 1 {ways}",
           "say1": "Na 1 episode: over de 50 runs gaf de eerste episode SARSA gemiddeld een return van {s} en Q-learning {q}: zonder kennis dwalen beide agents honderden stappen rond en vallen de agents vaak. In run 1 {ways}",
           "both": "leiden de pijlen van SARSA in {a} stappen van S naar G, en de pijlen van Q-learning in {b}.",
           "only_s": "leiden de pijlen van SARSA in {a} stappen van S naar G; de pijlen van Q-learning leiden nog niet naar G.",
           "only_q": "leiden de pijlen van Q-learning in {b} stappen van S naar G; de pijlen van SARSA leiden nog niet naar G.",
           "none": "leiden de pijlen van geen van beide methoden al van S naar G.",
           "note": "De grafiek: de gemiddelde return van de laatste 10 episodes over de 50 runs, SARSA doorgetrokken, Q-learning gestreept. De pijlen: de beste action van elke state in run 1; in oranje de weg waar de pijlen vanaf S heen leiden, zodra de pijlen naar G leiden. Beide methoden leren met ε = 0,1 en η = 0,5."},
}


def _ways(first: dict, w: dict) -> str:
    a, b = first["sarsa"][1], first["q"][1]
    if a and b:
        return w["both"].format(a=len(a) - 1, b=len(b) - 1)
    if a:
        return w["only_s"].format(a=len(a) - 1)
    if b:
        return w["only_q"].format(b=len(b) - 1)
    return w["none"]


def _grids(first: dict, n: int, lang: str, w: dict) -> str:
    word = w["one"] if n == 1 else w["many"]
    panels = [{"title": f'{w["sarsa"]}: {w["grid"].format(n=n, word=word)}', "arrows": first["sarsa"][0], "way": first["sarsa"][1]},
              {"title": f'{w["q"]}: {w["grid"].format(n=n, word=word)}', "arrows": first["q"][0], "way": first["q"][1]}]
    return cliff_svg(panels, w["grid_alt"].format(n=n, word=word), lang)


def play_frame(n: int, lang: str) -> tuple:
    """The play block after n episodes: the best actions of run 1 beside the mean return of the 50 runs, and a
    sentence."""
    lang = lang if lang in LANGS else "en"
    w = _PLAY_TEXT[lang]
    r = play_run()
    n = min(LOOKS, key=lambda k: abs(k - int(n)))
    word = w["one"] if n == 1 else w["many"]
    grids = _grids(r["first"][n], n, lang, w)
    curve = _svg([_curve_panel(r, n, lang, w["curve"])], w["curve_alt"].format(n=n), lang)
    body = live.rewound(curve + live.key_html(curve_key(lang)), [grids], width=330)
    if n == 0:
        say = w["say0"]
    elif n == 1:
        say = w["say1"].format(s=number(r["mean"]["sarsa"][0], 0, lang), q=number(r["mean"]["q"][0], 0, lang),
                               ways=_ways(r["first"][n], w))
    else:
        say = w["say"].format(n=n, word=word, s=number(last_mean(r, "sarsa", n), 1, lang),
                              q=number(last_mean(r, "q", n), 1, lang), ways=_ways(r["first"][n], w))
    return body + _note(w["note"]), say


# --- the demo: your sarsa_update beside Q-learning, 50 runs -------------------------------------------------------------

DEMO_TEXT = {
    "en": {"title": "Your runs: mean return", "x": "episodes", "y": "mean return",
           "alt": "The mean return of the last 10 episodes over 50 runs, after every episode so far: SARSA solid, Q-learning dashed.",
           "eps": "ε for both methods",
           "sarsa": "SARSA", "q": "Q-learning", "grid": "Run 1 after {n} {word}", "one": "episode", "many": "episodes",
           "grid_alt": "Two copies of the cliff, SARSA above and Q-learning below: the best action of every state in run 1 after {n} {word} as small arrows, and where they lead from S, if they lead to G, as an orange line.",
           "rewind": "After {n} {word}: over the 50 runs, the last 10 episodes gave SARSA a mean return of {s} and Q-learning {q}. In run 1, {ways}",
           "both": "the arrows of SARSA lead from S to G in {a} steps, and those of Q-learning in {b}.",
           "only_s": "the arrows of SARSA lead from S to G in {a} steps; those of Q-learning do not lead to G yet.",
           "only_q": "the arrows of Q-learning lead from S to G in {b} steps; those of SARSA do not lead to G yet.",
           "none": "the arrows of neither method lead from S to G yet.",
           "say0": "Before the first episode every Q is 0, so no state has a best action yet."},
    "nl": {"title": "Je runs: gemiddelde return", "x": "episodes", "y": "gemiddelde return",
           "alt": "De gemiddelde return van de laatste 10 episodes over 50 runs, na elke episode tot nu toe: SARSA doorgetrokken, Q-learning gestreept.",
           "eps": "ε voor beide methoden",
           "sarsa": "SARSA", "q": "Q-learning", "grid": "Run 1 na {n} {word}", "one": "episode", "many": "episodes",
           "grid_alt": "Twee keer het klif, SARSA boven en Q-learning onder: de beste action van elke state in run 1 na {n} {word} als kleine pijlen, en, als de pijlen naar G leiden, de weg vanaf S als oranje lijn.",
           "rewind": "Na {n} {word}: over de 50 runs gaven de laatste 10 episodes SARSA een gemiddelde return van {s} en Q-learning {q}. In run 1 {ways}",
           "both": "leiden de pijlen van SARSA in {a} stappen van S naar G, en de pijlen van Q-learning in {b}.",
           "only_s": "leiden de pijlen van SARSA in {a} stappen van S naar G; de pijlen van Q-learning leiden nog niet naar G.",
           "only_q": "leiden de pijlen van Q-learning in {b} stappen van S naar G; de pijlen van SARSA leiden nog niet naar G.",
           "none": "leiden de pijlen van geen van beide methoden al van S naar G.",
           "say0": "Bij de start is elke Q 0, dus heeft nog geen enkele state een beste action."},
}


def chart_points(r: dict, method: str, start: int, end: int) -> list:
    """The points of the demo's chart for episodes start to end: the mean return of the last 10 episodes, never below
    the bottom of the chart."""
    line = smooth(r["mean"][method][:end])
    return [(e, max(LOW, line[e - 1])) for e in range(max(1, start), end + 1)]


def now_picture(r: dict, n: int, lang: str) -> str:
    """Beside the chart of the demo: the best actions of run 1 as they are now."""
    lang = lang if lang in LANGS else "en"
    first = {m: (arrows(r["now"][m]), way(r["now"][m])) for m in METHODS}
    return _grids(first, n, lang, DEMO_TEXT[lang])


def rewind_frame(r: dict, n: int, lang: str) -> tuple:
    """A stop of the rewind after the demo: the chart of the run with a line at n, the best actions of run 1 then, and
    a sentence."""
    lang = lang if lang in LANGS else "en"
    w = DEMO_TEXT[lang]
    n = min(r["first"], key=lambda k: abs(k - int(n)))
    word = w["one"] if n == 1 else w["many"]
    curve = _svg([_curve_panel(r, n, lang, w["title"])], w["alt"], lang, width=330, height=240)
    curve += live.key_html(curve_key(lang))
    if n == 0:
        say = w["say0"]
    else:
        say = w["rewind"].format(n=n, word=word, s=number(last_mean(r, "sarsa", n), 1, lang),
                                 q=number(last_mean(r, "q", n), 1, lang), ways=_ways(r["first"][n], w))
    return live.rewound(curve, [_grids(r["first"][n], n, lang, w)], width=330), say


# --- the pictures of the learner's work -----------------------------------------------------------------------------------

_STOPPED = object()   # what _call gives for a function that stopped with an error
_SLOW = object()      # what _call gives for a function that ran out of time
PICTURE_SECONDS = 3   # what the learner's function gets for all the calls of one picture together
YOUR_EPISODES = 200   # the episodes of the picture under exercise 2


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _call(make, seconds: float = PICTURE_SECONDS):
    """make() within a time limit, with what it prints thrown away; _SLOW when the time runs out, _STOPPED when it
    stops with an error."""
    import contextlib
    import io
    from everything_ai.check import Timeout, with_time_limit
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(make, max(seconds, 0.05))
    except Timeout:
        return _SLOW
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def one_run(fn=sarsa_update, method: str = "sarsa", epsilon: float = EPSILON, seed: int = 0,
            episodes: int = EPISODES) -> dict:
    """One run of one method, as run k = seed of the play block: the Q-table after every episode in `keep`, and the
    return and falls of every episode."""
    Q = new_q()
    rnd = random.Random(seed)
    returns, falls, kept = [], [], {}
    for n in range(1, episodes + 1):
        total, fell = _episode(Q, fn if method == "sarsa" else q_update, epsilon, rnd)
        returns.append(total)
        falls.append(fell)
        if n in (50, 100, episodes):
            kept[n] = Named(copy.deepcopy(Q), f"Q_{n}")
    return {"Q": Q, "returns": returns, "falls": falls, "kept": kept}


@lru_cache(maxsize=1)
def _course_one() -> dict:
    return one_run()


def _your_episodes(fn, episodes: int = YOUR_EPISODES) -> dict:
    """The learner's sarsa_episode, episode after episode, from a Q-table of zeros, with the random numbers of seed 0."""
    Q = new_q()
    rng = np.random.default_rng(0)
    returns = []
    for _ in range(episodes):
        total = fn(Q, ETA, EPSILON, GAMMA, rng)
        if not _fine(total):
            raise _Bad
        returns.append(float(total))
    return {"Q": Q, "returns": returns}


_PICTURE_TEXT = {
    "en": {"sarsa_update": {"title": "Your SARSA in run 1", "who": "`sarsa_update`",
                            "head": "Run 1 of the play block with your `sarsa_update`, ε = 0.1 and η = 0.5: the best action of every state after 500 episodes, and in orange the way they lead from S.",
                            "say": "Over the last 100 episodes your agent's mean return is {m}, and it fell {f} times; {ways}"},
           "sarsa_episode": {"title": "Your episodes", "who": "`sarsa_episode`",
                             "head": "200 episodes of your `sarsa_episode`, from a Q-table of zeros, with ε = 0.1 and η = 0.5: the best action of every state after the last one, and in orange the way they lead from S.",
                             "say": "The last 50 episodes gave a mean return of {m}; {ways}"},
           "way": "its arrows lead from S to G in {a} steps.", "no_way": "its arrows do not lead from S to G yet.",
           "panel": "After {n} episodes",
           "none": "Your {who} stops with an error, takes too long, or gives something that is not a number, so nothing can be drawn yet.",
           "alt": "The cliff with the best action of every state as small arrows, and where they lead from S, if they lead to G, as an orange line."},
    "nl": {"sarsa_update": {"title": "Je SARSA in run 1", "who": "`sarsa_update`",
                            "head": "Run 1 uit het speelblok met je `sarsa_update`, ε = 0,1 en η = 0,5: de beste action van elke state na 500 episodes, en in oranje de weg waar de pijlen vanaf S heen leiden.",
                            "say": "Over de laatste 100 episodes is de gemiddelde return van je agent {m}, en de agent viel {f} keer; {ways}"},
           "sarsa_episode": {"title": "Je episodes", "who": "`sarsa_episode`",
                             "head": "200 episodes van je `sarsa_episode`, vanaf een Q-table met nullen, met ε = 0,1 en η = 0,5: de beste action van elke state na de laatste episode, en in oranje de weg waar de pijlen vanaf S heen leiden.",
                             "say": "De laatste 50 episodes gaven een gemiddelde return van {m}; {ways}"},
           "way": "de pijlen leiden in {a} stappen van S naar G.", "no_way": "de pijlen leiden nog niet van S naar G.",
           "panel": "Na {n} episodes",
           "none": "Je {who} stopt met een fout, duurt te lang, of geeft iets wat geen getal is, dus er is nog geen afbeelding.",
           "alt": "Het klif met de beste action van elke state als kleine pijlen, en, als de pijlen naar G leiden, de weg vanaf S als oranje lijn."},
}


def _picture(seen, name: str) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _PICTURE_TEXT[lang]
    title, who, head = w[name]["title"], w[name]["who"], w[name]["head"]
    fn = _fn(seen)
    got = None
    if fn is not None:
        make = (lambda: one_run(fn)) if name == "sarsa_update" else (lambda: _your_episodes(fn))
        got = _call(make)
        if not isinstance(got, dict):
            got = None
    if got is None:
        return yours.figure(title, _note(head) + _note(w["none"].format(who=who), _LOSS_UI), alt=title)
    cells = way(got["Q"])
    ways = w["way"].format(a=len(cells) - 1) if cells else w["no_way"]
    n = EPISODES if name == "sarsa_update" else YOUR_EPISODES
    body = _note(head) + cliff_svg([{"title": w["panel"].format(n=n), "arrows": arrows(got["Q"]), "way": cells}], w["alt"], lang)
    if name == "sarsa_update":
        say = w[name]["say"].format(m=number(sum(got["returns"][-100:]) / 100, 1, lang), f=sum(got["falls"][-100:]), ways=ways)
    else:
        say = w[name]["say"].format(m=number(sum(got["returns"][-50:]) / 50, 1, lang), ways=ways)
    return yours.figure(title, body + _note(say), alt=title)


def update_picture(seen) -> str:
    return _picture(seen, "sarsa_update")


def episode_picture(seen) -> str:
    return _picture(seen, "sarsa_episode")


# --- judging ------------------------------------------------------------------------------------------------------------

def _example_q() -> Named:
    """The Q-table of the worked example and of the first case of exercise 1: zeros, except next to the cliff at (2, 5)
    and (2, 6)."""
    Q = new_q()
    Q[(2, 5)] = {"up": -9.0, "down": -110.0, "left": -9.0, "right": -7.0}
    Q[(2, 6)] = {"up": -8.0, "down": -114.0, "left": -8.0, "right": -6.0}
    return Named(Q, "Q")


EXAMPLE_Q = _example_q()


def _q_run(n: int) -> Named:
    """The Q-table of the course's run 1 of SARSA after n episodes (50, 100 or 500)."""
    return _course_one()["kept"][n]


def same_number(a, b) -> bool:
    return _fine(a) and _fine(b) and abs(float(a) - float(b)) < 1e-6


@dataclass(frozen=True)
class Episode:
    """What sarsa_episode did with one case: the return it gave back, the Q-table it left, and the first Q where that
    table differs from the right one (state, action, got, want)."""
    total: object
    values: tuple
    off: tuple | None


def _values(Q) -> tuple:
    out = []
    for s in sorted(Q):
        for a in ACTIONS:
            v = Q[s].get(a) if isinstance(Q.get(s), dict) else None
            out.append(round(float(v), 6) if _fine(v) else None)
    return tuple(out)


def judge_episode(table, eta, epsilon, gamma, seed, fn) -> Episode:
    Q = copy.deepcopy(table)
    total = fn(Q, eta, epsilon, gamma, np.random.default_rng(seed))
    want_Q = copy.deepcopy(table)
    want_total = sarsa_episode(want_Q, eta, epsilon, gamma, np.random.default_rng(seed))
    off = None
    for s in sorted(want_Q):
        for a in ACTIONS:
            v = Q.get(s, {}).get(a) if isinstance(Q.get(s), dict) else None
            if not (_fine(v) and abs(float(v) - want_Q[s][a]) < 1e-6):
                off = (s, a, v, want_Q[s][a])
                break
        if off:
            break
    total = float(total) if _fine(total) else total
    if total is None:
        return None
    return Episode(total, _values(Q), None if off is None or total != want_total else off)


def same_episode(a, b) -> bool:
    if not isinstance(a, Episode) or not isinstance(b, Episode):
        return a == b
    return same_number(a.total, b.total) and a.values == b.values


def _code_number(v) -> str:
    """A number as Python writes it, rounded to four decimals: for inside code marks, in both languages."""
    v = round(float(v), 4)
    return str(int(v)) if v.is_integer() else repr(v)


def show_number(v) -> dict:
    """A number the learner's function gave back, as code, rounded to four decimals."""
    shown = f"`{_code_number(v)}`" if _fine(v) else f"`{v!r}`"
    return {"en": shown, "nl": shown}


def show_episode(got, lang: str = "en") -> dict:
    if not isinstance(got, Episode):
        return {"en": f"`{got!r}`", "nl": f"`{got!r}`"}
    total = total_nl = _code_number(got.total) if _fine(got.total) else repr(got.total)
    if got.off is None:
        return {"en": f"the return `{total}`", "nl": f"de return `{total_nl}`"}
    s, a, v, want = got.off
    have = number(v, 2, "en") if _fine(v) else "nothing"
    have_nl = number(v, 2, "nl") if _fine(v) else "niets"
    return {"en": f"the return `{total}`, but leaves `Q[{s}][\"{a}\"]` at {have}, where it should be {number(want, 2, 'en')}",
            "nl": f"de return `{total_nl}`, maar laat `Q[{s}][\"{a}\"]` achter op {have_nl}, waar {number(want, 2, 'nl')} hoort"}


@dataclass
class EpisodeBox(FunctionBox):
    """sarsa_episode in a code box: the check gives it a copy of a Q-table and a new random generator, and judges the
    return it gives back and the Q-table it leaves, so that a known mistake is known by what it does in every case."""

    def _run(self, fn, case):
        return judge_episode(*case, fn)

    def _call_text(self, case) -> str:
        return f"sarsa_episode({case[0]!r}, {case[1]!r}, {case[2]!r}, {case[3]!r}, np.random.default_rng({case[4]!r}))"


# --- the exercises -----------------------------------------------------------------------------------------------------

REQUIRED = ["sarsa_update", "sarsa_episode", "robot"]


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


# exercise 1: sarsa_update
def _u(line: str = "reward + gamma * Q[nxt][nxt_action]") -> str:
    """sarsa_update as the box holds it, with this line: the target."""
    return ("def sarsa_update(Q, state, action, reward, nxt, nxt_action, eta, gamma):\n"
            f"    target = {line}\n"
            "    return Q[state][action] + eta * (target - Q[state][action])\n")


_U_START = _u("...")
_U = _u()
_U_MAX = _u("reward + gamma * max(Q[nxt].values())")
_U_THIS_ACTION = _u("reward + gamma * Q[nxt][action]")
_U_DOWN = _u('reward + gamma * Q[nxt]["down"]')
_U_THIS_STATE = _u("reward + gamma * Q[state][nxt_action]")
_U_NO_REWARD = _u("gamma * Q[nxt][nxt_action]")
_U_NO_GAMMA = _u("reward + Q[nxt][nxt_action]")
_U_OWN = _u("reward + gamma * Q[state][action]")
_U_ERROR = _u("reward + gamma * Q[nxt][nxt_action] - Q[state][action]")


def update_cases() -> list:
    """The steps the check gives sarsa_update: first the step of the worked example, where the random action is down;
    then four steps with the Q-table of the course's run 1 after 500 episodes: on the top row, a fall, and two with
    γ = 0.9."""
    q = _q_run(500)
    return [
        (EXAMPLE_Q, (2, 5), "right", -1, (2, 6), "down", 0.5, 1.0),
        (q, (0, 4), "right", -1, (0, 5), "right", 0.5, 1.0),
        (q, (2, 3), "down", -100, (3, 0), "up", 0.5, 1.0),
        (q, (1, 2), "up", -1, (0, 2), "left", 0.2, 0.9),
        (q, (0, 10), "right", -1, (0, 11), "left", 0.1, 0.9),
    ]


# exercise 2: sarsa_episode
def _e(line: str = "choose(Q, nxt, epsilon, rng)") -> str:
    """sarsa_episode as the box holds it, with this line: the next action."""
    return ("def sarsa_episode(Q, eta, epsilon, gamma, rng):\n"
            "    state = START\n"
            "    action = choose(Q, state, epsilon, rng)\n"
            "    total = 0\n"
            "    while state != GOAL:\n"
            "        nxt, reward = step(state, action)\n"
            "        total = total + reward\n"
            f"        nxt_action = {line}\n"
            "        Q[state][action] = sarsa_update(Q, state, action, reward, nxt, nxt_action, eta, gamma)\n"
            "        state = nxt\n"
            "        action = nxt_action\n"
            "    return total\n")


_E_START = _e("...")
_E = _e()
_E_OLD_STATE = _e("choose(Q, state, epsilon, rng)")
_E_NO_EXPLORING = _e("choose(Q, nxt, 0, rng)")
EPISODE_NAMES = {"START": START, "GOAL": GOAL, "step": step, "choose": choose, "sarsa_update": sarsa_update}


def episode_cases() -> list:
    """What the check gives sarsa_episode: a Q-table, η, ε, γ and the seed of a new random generator. First the
    Q-table of the course's run 1 after 50 episodes; then other seeds, ε = 0.2, a table of zeros, and γ = 0.9."""
    q50, q500 = _q_run(50), _q_run(500)
    zeros = Named(new_q(), "new_q()")
    return [
        (q50, 0.5, 0.1, 1.0, 1),
        (q50, 0.5, 0.2, 1.0, 2),
        (zeros, 0.5, 0.1, 1.0, 3),
        (q500, 0.1, 0.1, 0.9, 4),
        (q50, 0.3, 0.05, 0.9, 5),
    ]


UPDATE_CASES = update_cases()
EPISODE_CASES = episode_cases()


EXERCISES = {
    "sarsa_update": FunctionBox(
        name="sarsa_update",
        function="sarsa_update",
        start=_U_START,
        solution=_U,
        compare=same_number,
        show=show_number,
        visible=UPDATE_CASES[:1],
        hidden=UPDATE_CASES[1:],
        first=[_DOTS],
        picture=update_picture,
        mistakes=[
            Mistake("max", wrong=_U_MAX,
                    en="`{call}` gives {got}: that is the target of Q-learning, with the largest Q of `nxt`, as if the agent will take the best action there. SARSA takes the Q of the action the agent will really take next; which name holds that action?",
                    nl="`{call}` geeft {got}: dat is het target van Q-learning, met de grootste Q van `nxt`, alsof de agent daar de beste action neemt. SARSA neemt de Q van de action die de agent echt als volgende neemt; welke naam bevat deze action?"),
            Mistake("this-action", wrong=_U_THIS_ACTION,
                    en="`{call}` gives {got}: the target takes the Q of `action` in the next state, but `action` is the action of this step. Which action will the agent take in `nxt`?",
                    nl="`{call}` geeft {got}: het target neemt de Q van `action` in de volgende state, maar `action` is de action van deze stap. Welke action neemt de agent in `nxt`?"),
            Mistake("down", wrong=_U_DOWN,
                    en="`{call}` gives {got}: your target always takes the Q of down in `nxt`, the action of the worked example. Which name holds the action the agent will take next in this step?",
                    nl="`{call}` geeft {got}: je target neemt altijd de Q van down in `nxt`, de action uit het uitgewerkte voorbeeld. Welke naam bevat de action die de agent in deze stap als volgende neemt?"),
            Mistake("this-state", wrong=_U_THIS_STATE,
                    en="`{call}` gives {got}: the target takes the Q of the next action in `state`, the state the agent leaves. In which state will the agent take that action?",
                    nl="`{call}` geeft {got}: het target neemt de Q van de volgende action in `state`, de state die de agent verlaat. In welke state neemt de agent deze action?"),
            Mistake("no-reward", wrong=_U_NO_REWARD,
                    en="`{call}` gives {got}: the reward of the step is missing, so the −1 of every step and the −100 of a fall never reach Q. What does the step itself bring, before the Q of the next action?",
                    nl="`{call}` geeft {got}: de reward van de stap ontbreekt, dus de −1 van elke stap en de −100 van een val komen nooit in Q. Wat levert de stap zelf op, nog voor de Q van de volgende action?"),
            Mistake("no-gamma", wrong=_U_NO_GAMMA,
                    en="Right with γ = 1. But `{call}` gives {got}: the Q of the next action counts in full, without `gamma`. With γ = 0.9, how much of it should count?",
                    nl="Goed met γ = 1. Maar `{call}` geeft {got}: de Q van de volgende action telt volledig mee, zonder `gamma`. Hoeveel ervan moet er met γ = 0,9 meetellen?"),
            Mistake("own", wrong=_U_OWN,
                    en="`{call}` gives {got}: the target takes the old Q of this very state and action, so Q never learns what lies after the step. Which Q tells you what comes after it?",
                    nl="`{call}` geeft {got}: het target neemt de oude Q van deze state en action zelf, dus Q leert nooit wat er na de stap ligt. Welke Q zegt wat er daarna komt?"),
            Mistake("error", wrong=_U_ERROR,
                    en="`{call}` gives {got}: your target is already the TD error, the target minus the old Q, and the line under it takes the old Q away once more. What is the target, before anything is taken away?",
                    nl="`{call}` geeft {got}: je target is al de TD error, het target min de oude Q, en de regel eronder haalt de oude Q er nog een keer af. Wat is het target, voordat je er iets van aftrekt?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the target of SARSA: the reward plus the discounted Q of the action the agent will take next, in the next state.",
                      "nl": "Vervang de `...` door het target van SARSA: de reward plus de verdisconteerde Q van de action die de agent als volgende neemt, in de volgende state."},
            "arguments": {"en": "The check calls `sarsa_update(Q, state, action, reward, nxt, nxt_action, eta, gamma)`, and the `def` line of your function has a different number of names between its parentheses. Can you keep those eight names?",
                          "nl": "De check roept `sarsa_update(Q, state, action, reward, nxt, nxt_action, eta, gamma)` aan, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je deze acht namen houden?"},
            "slow": {"en": "Your function is still not done after 2 seconds. Does it do one step, without a loop of its own?",
                     "nl": "Je functie is na 2 seconden nog niet klaar. Doet je functie één stap, zonder eigen loop?"},
            "wrong": {"en": "`{call}` gives {got}. Is `target` the reward plus `gamma` times the Q of the next action in the next state?",
                      "nl": "`{call}` geeft {got}. Is `target` de reward plus `gamma` keer de Q van de volgende action in de volgende state?"},
            "almost": {"en": "Right for the step of the worked example. But `{call}` gives {got}. Does your function work for every step, with the values it gets?",
                       "nl": "Goed voor de stap van het uitgewerkte voorbeeld. Maar `{call}` geeft {got}. Werkt je functie voor elke stap, met de waarden die je functie krijgt?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with the line that gives back the new Q?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met de regel die de nieuwe Q teruggeeft?"},
            "correct": {"en": "Right in all {n} cases: in the worked example the random step down pulls Q((2, 5), right) from −7 to −61, where Q-learning leaves it at −7. In the picture your `sarsa_update` learns run 1 of the play block; which way do its arrows take?",
                        "nl": "Goed in alle {n} gevallen: in het uitgewerkte voorbeeld trekt de willekeurige stap omlaag Q((2, 5), right) van −7 naar −61, waar Q-learning de waarde op −7 laat. In de afbeelding leert je `sarsa_update` run 1 uit het speelblok; welke weg nemen de pijlen?"},
        },
    ),
    "sarsa_episode": EpisodeBox(
        name="sarsa_episode",
        function="sarsa_episode",
        start=_E_START,
        solution=_E,
        names=EPISODE_NAMES,
        compare=same_episode,
        show=show_episode,
        visible=EPISODE_CASES[:1],
        hidden=EPISODE_CASES[1:],
        first=[_DOTS],
        picture=episode_picture,
        mistakes=[
            Mistake("old-state", wrong=_E_OLD_STATE,
                    en="`{call}` gives {got}: the next action is chosen for `state`, the state the agent leaves, but the agent takes it in `nxt`. In which state should `choose` look at the Q's?",
                    nl="`{call}` geeft {got}: de volgende action wordt gekozen voor `state`, de state die de agent verlaat, maar de agent neemt de action in `nxt`. In welke state moet `choose` naar de Q's kijken?"),
            Mistake("no-exploring", wrong=_E_NO_EXPLORING,
                    en="`{call}` gives {got}: with 0 in place of `epsilon`, the agent never takes a random action after its first step, so SARSA learns the value of a policy that never explores. Which chance of a random action should the next action get?",
                    nl="`{call}` geeft {got}: met 0 in plaats van `epsilon` neemt de agent na de eerste stap nooit een willekeurige action, dus SARSA leert de waarde van een policy die nooit verkent. Welke kans op een willekeurige action hoort bij de volgende action?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the next action: the agent's ε-greedy choice in the state it has just landed on.",
                      "nl": "Vervang de `...` door de volgende action: de ε-greedy keuze van de agent in de state waar de agent net is beland."},
            "arguments": {"en": "The check calls `sarsa_episode(Q, eta, epsilon, gamma, rng)`, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(Q, eta, epsilon, gamma, rng)`?",
                          "nl": "De check roept `sarsa_episode(Q, eta, epsilon, gamma, rng)` aan, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(Q, eta, epsilon, gamma, rng)` houden?"},
            "slow": {"en": "Your function is still not done after 2 seconds. Does the agent get a new action at every step, chosen in the state it has just landed on, so that it can reach G?",
                     "nl": "Je functie is na 2 seconden nog niet klaar. Krijgt de agent bij elke stap een nieuwe action, gekozen in de state waar de agent net is beland, zodat de agent G kan bereiken?"},
            "wrong": {"en": "`{call}` gives {got}. Does `choose` get the same kinds of values as on the line before the loop, but with the state the agent has just landed on?",
                      "nl": "`{call}` geeft {got}. Krijgt `choose` dezelfde soort waarden als op de regel boven de loop, maar met de state waar de agent net is beland?"},
            "almost": {"en": "Right for the first call. But `{call}` gives {got}. Does your next action use the values the function gets?",
                       "nl": "Goed voor de eerste aanroep. Maar `{call}` geeft {got}. Gebruikt je volgende action de waarden die je functie krijgt?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return total`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return total`?"},
            "correct": {"en": "Right in all {n} cases: with the Q-table after 50 episodes, the episode of seed 1 gives the return −19: 19 steps without a fall. In the picture your `sarsa_episode` plays 200 episodes from a table of zeros; do its arrows reach G?",
                        "nl": "Goed in alle {n} gevallen: met de Q-table na 50 episodes geeft de episode van seed 1 de return −19: 19 stappen zonder val. In de afbeelding speelt je `sarsa_episode` 200 episodes vanaf een table met nullen; bereiken de pijlen G?"},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("sarsa_update", _U_START, "empty", None),
    ("sarsa_update", _U, "correct", None),
    ("sarsa_update", _u("reward + ..."), "wrong", "dots-left"),
    ("sarsa_update", _U_MAX, "wrong", "max"),
    ("sarsa_update", _U_THIS_ACTION, "wrong", "this-action"),
    ("sarsa_update", _U_DOWN, "wrong", "down"),
    ("sarsa_update", _U_THIS_STATE, "wrong", "this-state"),
    ("sarsa_update", _U_NO_REWARD, "wrong", "no-reward"),
    ("sarsa_update", _U_NO_GAMMA, "wrong", "no-gamma"),
    ("sarsa_update", _U_OWN, "wrong", "own"),
    ("sarsa_update", _U_ERROR, "wrong", "error"),
    ("sarsa_update", _u().replace("    return Q[state][action]", "    Q[state][action] = Q[state][action]"), "wrong", "no-return"),
    # other right answers: the TD error first, and everything in one line
    ("sarsa_update", "def sarsa_update(Q, state, action, reward, nxt, nxt_action, eta, gamma):\n    delta = reward + gamma * Q[nxt][nxt_action] - Q[state][action]\n    return Q[state][action] + eta * delta\n", "correct", None),
    ("sarsa_update", "def sarsa_update(Q, state, action, reward, nxt, nxt_action, eta, gamma):\n    return (1 - eta) * Q[state][action] + eta * (reward + gamma * Q[nxt][nxt_action])\n", "correct", None),
    # the discount on the reward too: right with γ = 1 only
    ("sarsa_update", _u("gamma * (reward + Q[nxt][nxt_action])"), "almost", None),
    ("sarsa_episode", _E_START, "empty", None),
    ("sarsa_episode", _E, "correct", None),
    ("sarsa_episode", _e("choose(Q, nxt, ...)"), "wrong", "dots-left"),
    ("sarsa_episode", _E_OLD_STATE, "wrong", "old-state"),
    ("sarsa_episode", _E_NO_EXPLORING, "wrong", "no-exploring"),
    ("sarsa_episode", _E.replace("    return total\n", ""), "wrong", "no-return"),
    # ε written as a number: right with ε = 0.1 only
    ("sarsa_episode", _e("choose(Q, nxt, 0.1, rng)"), "almost", None),
]
