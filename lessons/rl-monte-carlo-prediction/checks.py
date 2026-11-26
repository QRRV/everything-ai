# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rl-monte-carlo-prediction: the robot of rl-policy-evaluation walks from its corner to the
charger with the random policy, and the values of the tiles are estimated from those walks alone, without the table T,
by averaging the returns after each tile's first visits (first-visit Monte Carlo) or after every visit (every-visit).
The course's own walks, returns and estimates; returns and first_visit, the functions the learner writes, and their
pictures; the play block over the walks.

Monte Carlo prediction as in Sutton and Barto (Reinforcement Learning: An Introduction, 2nd edition, 2018, section
5.1). The room, the robot, the walks and the pictures are the course's own."""
import ast
import copy
import html
import math
from dataclasses import dataclass

import numpy as np

from everything_ai import chart, draw, yours
from everything_ai.check import FunctionBox, Mistake

LANGS = ("en", "nl")
ACTIONS = ["up", "down", "left", "right"]
MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
GAMMA = 1.0      # the robot's steps all count the same
SEED = 7         # the seed of the course's walks
N_WALKS = 1000   # how many walks the course plays from the corner


# --- the room, its table and the random policy --------------------------------------------------------------------------

class World(list):
    """A grid of tiles as a list of strings, one per row: '.' a tile, G the charger."""

    def __init__(self, rows, name: str):
        super().__init__(rows)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return list(self)


ROOM = World([       # the room of rl-policy-evaluation: the charger in the middle tile
    ".....",
    ".....",
    "..G..",
    ".....",
    ".....",
], "ROOM")


def table(world) -> dict:
    """The table of a world, as in rl-policy-evaluation: an action goes where it says, and into the edge the robot stays
    where it is; every step costs −1. On the charger the task is over: every action stays there, with reward 0."""
    T = {}
    for r, row in enumerate(world):
        for c, sign in enumerate(row):
            state = (r, c)
            T[state] = {}
            for a in ACTIONS:
                if sign == "G":
                    T[state][a] = [(1.0, state, 0)]
                    continue
                nr, nc = r + MOVES[a][0], c + MOVES[a][1]
                nxt = (nr, nc) if 0 <= nr < len(world) and 0 <= nc < len(world[0]) else state
                T[state][a] = [(1.0, nxt, -1)]
    return T


T_ROOM = table(ROOM)
RANDOM = {s: {a: 1 / len(acts) for a in acts} for s, acts in T_ROOM.items()}
CHARGER = (2, 2)
CORNER = (0, 0)    # where every walk of the course starts
NEXT = (1, 2)      # the tile above the charger
DIAGONAL = (1, 1)  # a tile diagonal to it
EDGE = (0, 2)      # the middle of an edge
STATES = [s for s in T_ROOM if s != CHARGER]


def _ends(T, s) -> bool:
    """Whether the task is over in s: every action stays there, with reward 0."""
    return all(outs == [(1.0, s, 0)] for outs in T[s].values())


def true_values(T, pi, gamma) -> dict:
    """The values of policy pi, solved exactly from the Bellman equation of the policy: what rl-policy-evaluation's sweeps
    come close to (−24 next to the charger, −36 in a corner)."""
    states = [s for s in T if not _ends(T, s)]
    at = {s: i for i, s in enumerate(states)}
    A, b = np.eye(len(states)), np.zeros(len(states))
    for s in states:
        for a in T[s]:
            for p, nxt, r in T[s][a]:
                b[at[s]] += pi[s][a] * p * r
                if nxt in at:
                    A[at[s], at[nxt]] -= gamma * pi[s][a] * p
    x = np.linalg.solve(A, b)
    return {s: (float(x[at[s]]) if s in at else 0.0) for s in T}


END = true_values(T_ROOM, RANDOM, GAMMA)   # −24, −30, −32, −34, −36


def step_moments(T, pi, start) -> tuple[float, float]:
    """The mean and the standard deviation of the number of steps from start to the end, solved exactly: with E[N] = 1 +
    the average E[N'] one step on, and E[N²] = 1 + 2 E[N'] + E[N'²] averaged the same way."""
    states = [s for s in T if not _ends(T, s)]
    at = {s: i for i, s in enumerate(states)}
    P = np.zeros((len(states), len(states)))
    for s in states:
        for a in T[s]:
            for p, nxt, _r in T[s][a]:
                if nxt in at:
                    P[at[s], at[nxt]] += pi[s][a] * p
    A = np.eye(len(states)) - P
    m1 = np.linalg.solve(A, np.ones(len(states)))
    m2 = np.linalg.solve(A, np.ones(len(states)) + 2 * P @ m1)
    k = at[start]
    return float(m1[k]), float(math.sqrt(m2[k] - m1[k] ** 2))


def step_chances(T, pi, start, most: int = 2000) -> list[float]:
    """chances[n]: the probability that a walk from start ends after exactly n steps, for n up to `most`."""
    states = [s for s in T if not _ends(T, s)]
    at = {s: i for i, s in enumerate(states)}
    P = np.zeros((len(states), len(states)))
    out = np.zeros(len(states))   # the probability to end with the next step
    for s in states:
        for a in T[s]:
            for p, nxt, _r in T[s][a]:
                if nxt in at:
                    P[at[s], at[nxt]] += pi[s][a] * p
                else:
                    out[at[s]] += pi[s][a] * p
    now = np.zeros(len(states))
    now[at[start]] = 1.0
    chances = [0.0]
    for _ in range(most):
        chances.append(float(now @ out))
        now = now @ P
    return chances


MEAN_STEPS, SPREAD = step_moments(T_ROOM, RANDOM, CORNER)   # 36 and 31.71: the walks from the corner


# --- episodes and the course's Monte Carlo -------------------------------------------------------------------------------

class Episode(list):
    """An episode as a list of steps (state, reward): the state the robot was on and the reward of the step it took
    from there. A check message calls it by its name, such as WALK; the learner's code gets a plain copy."""

    def __init__(self, steps, name: str = ""):
        super().__init__(steps)
        self.name = name

    def __repr__(self):
        return self.name or list.__repr__(self)

    def __deepcopy__(self, memo):
        return [tuple(step) for step in self]


class Walks(list):
    """A list of episodes that a check message calls by its name, such as TEN_WALKS."""

    def __init__(self, episodes, name: str = ""):
        super().__init__(episodes)
        self.name = name

    def __repr__(self):
        return self.name or list.__repr__(self)

    def __deepcopy__(self, memo):
        return [[tuple(step) for step in ep] for ep in self]


def walk(T, pi, start, rng) -> Episode:
    """One walk with policy pi from start until the task is over: for each step a random number picks the action, by
    the probabilities of pi in their order (and another one the outcome, where an action has more than one)."""
    steps = []
    s = start
    while not _ends(T, s):
        u = rng.random()
        acts = list(pi[s])
        a = acts[-1]
        for b in acts:
            u = u - pi[s][b]
            if u < 0:
                a = b
                break
        outs = T[s][a]
        k = 0
        if len(outs) > 1:
            u = rng.random()
            for k, (p, _nxt, _r) in enumerate(outs):
                u = u - p
                if u < 0:
                    break
        _p, nxt, r = outs[k]
        steps.append((s, r))
        s = nxt
    return Episode(steps)


def returns(episode, gamma) -> list:
    """The course's solution of exercise 1: the return from every step, worked out backward. The learner's first_visit
    gets this function."""
    out = [0] * len(episode)
    G = 0
    for t in range(len(episode) - 1, -1, -1):
        s, r = episode[t]
        G = r + gamma * G
        out[t] = G
    return out


def first_visit(episodes, gamma) -> dict:
    """The course's solution of exercise 2: the average of the returns after the first visit of each state."""
    total, count = {}, {}
    for episode in episodes:
        G = returns(episode, gamma)
        seen = set()
        for t in range(len(episode)):
            s = episode[t][0]
            if s not in seen:
                seen.add(s)
                total[s] = total.get(s, 0) + G[t]
                count[s] = count.get(s, 0) + 1
    return {s: total[s] / count[s] for s in total}


def every_visit(episodes, gamma) -> dict:
    """Every-visit Monte Carlo: the average of the returns after every visit of each state."""
    total, count = {}, {}
    for episode in episodes:
        G = returns(episode, gamma)
        for t in range(len(episode)):
            s = episode[t][0]
            total[s] = total.get(s, 0) + G[t]
            count[s] = count.get(s, 0) + 1
    return {s: total[s] / count[s] for s in total}


def course_walks(n: int = N_WALKS, seed: int = SEED) -> Walks:
    rng = np.random.default_rng(seed)
    return Walks([walk(T_ROOM, RANDOM, CORNER, rng) for _ in range(n)], "WALKS")


WALKS = course_walks()                          # the walks of the play block, all from the corner
TEN_WALKS = Walks(WALKS[:10], "TEN_WALKS")      # the example of exercise 2
FIRST_WALK = Episode(WALKS[0], "FIRST_WALK")    # walk 1 of the play block, a case of exercise 1
# the walk of exercise 1, in the corridor of the worked example: tiles 0 (the charger) to 3 (against the wall); from
# tile 3 the robot bumps into the wall, goes left twice, right once and left twice
WALK = Episode([(3, -1), (3, -1), (2, -1), (1, -1), (2, -1), (1, -1)], "WALK")
# the same corridor, where the step onto the charger pays +10
PRIZE_WALK = Episode([(2, -1), (1, -1), (2, -1), (3, -1), (3, -1), (2, -1), (1, 10)], "PRIZE_WALK")


def visits(episode, state) -> int:
    return sum(1 for s, _r in episode if s == state)


def _running(episodes, state) -> tuple[list, list]:
    """The first-visit and the every-visit estimate of state after each number of walks, from 1 on."""
    fv, ev = [], []
    f_total = f_count = e_total = e_count = 0
    for episode in episodes:
        G = returns(episode, GAMMA)
        seen = False
        for t, (s, _r) in enumerate(episode):
            if s != state:
                continue
            if not seen:
                seen = True
                f_total, f_count = f_total + G[t], f_count + 1
            e_total, e_count = e_total + G[t], e_count + 1
        fv.append(f_total / f_count if f_count else None)
        ev.append(e_total / e_count if e_count else None)
    return fv, ev


FV_CORNER, EV_CORNER = _running(WALKS, CORNER)   # [k - 1] is the estimate after k walks
LOOKS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]   # the walks of the play block
PLAY = {n: first_visit(WALKS[:n], GAMMA) for n in LOOKS}
PLAY_EVERY = {n: every_visit(WALKS[:n], GAMMA) for n in LOOKS}


def largest_gap(V: dict) -> float:
    """The largest distance between an estimate and the value of policy evaluation, over the tiles V has."""
    return max(abs(V[s] - END[s]) for s in V if s != CHARGER)


def corridor_example(walks: int = 1000, seed: int = 0) -> list[int]:
    """The worked example: the numbers of steps of the walks from tile 3 of the corridor, as the lesson's cell plays
    them."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(walks):
        s, steps = 3, 0
        while s != 0:
            if rng.random() < 0.5:
                s = s - 1
            else:
                s = min(s + 1, 3)
            steps = steps + 1
        out.append(steps)
    return out


def numpy_walks(n: int = 100000, seed: int = 0) -> np.ndarray:
    """The NumPy cell: the numbers of steps of n walks from the corner, all played at once."""
    rng = np.random.default_rng(seed)
    row = np.zeros(n, dtype=int)
    col = np.zeros(n, dtype=int)
    steps = np.zeros(n, dtype=int)
    going = np.ones(n, dtype=bool)
    while going.sum() > 0:
        move = rng.integers(0, 4, size=n)
        row = np.clip(row - (move == 0) + (move == 1), 0, 4)
        col = np.clip(col - (move == 2) + (move == 3), 0, 4)
        steps = steps + going
        going = going & ((row != 2) | (col != 2))
    return steps


# --- words and numbers for the pictures ---------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem"


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
    """A number for the lesson's text: a decimal comma in Dutch, and a real minus sign."""
    return _dec(x, lang if lang in LANGS else "en", digits)


def _svg(panels: list, alt: str, lang: str, width: int = 330, height: int = 250) -> str:
    out = chart.figure(panels, alt=alt, width=width, height=height)
    return chart.with_decimal_comma(out) if lang == "nl" else out


def _room_panel(V: dict, title: str, digits: int = 2) -> dict:
    """ROOM with a value on every tile V has but the charger, which shows its value 0."""
    tiles = {s: float(V[s]) for s in V if s != CHARGER}
    return {"rows": ROOM, "title": title, "role": "pred", "index": True, "values": tiles, "goal": "0", "digits": digits}


def _grid(panels: list, alt: str, lang: str, cell: int = 51) -> str:
    return draw.grid_world(panels, alt=alt, cell=cell, decimal="," if lang == "nl" else ".").text


_PLAY_TEXT = {
    "en": {"title": "ROOM after {n} {word}", "one": "walk", "many": "walks",
           "chart": "The corner, walk by walk", "x": "walks", "y": "estimate of the corner",
           "first": "first visit", "every": "every visit",
           "alt": "ROOM, 5 by 5 tiles with the charger in the middle, with the first-visit estimate of every tile visited in the first {n} {word}: the corner {c}, above the charger {a}; {left}.",
           "left_none": "every tile has been visited",
           "left_some": "{k} {tiles} not visited yet",
           "tile": "tile", "tiles": "tiles",
           "chart_alt": "The estimate of the corner after every walk from 1 to 1,000, on a log scale: first visit, solid, and every visit, dashed, both swinging a lot at first and then settling near the value −36, a gray line, between two dashed gray lines that close in on it; a mark at walk {n}, where first visit is at {f} and every visit at {e}.",
           "say": "After {n} {word}: the corner is at {f} with first visit and at {e} with every visit, {gap} from −36 with first visit; the tile above the charger is at {a}.",
           "note": "Gray: the charger, with the value 0; in the chart the value −36 of policy evaluation, and dashed one spread of the average above and below it, 31.7/√n for n walks. Blue: the estimate of each tile after these walks, the average of the returns after its first visits; a tile without a number has not been visited yet. In the chart the corner after every walk, solid with first visit and dashed with every visit."},
    "nl": {"title": "ROOM na {n} {word}", "one": "wandeling", "many": "wandelingen",
           "chart": "De hoek, wandeling na wandeling", "x": "wandelingen", "y": "schatting van de hoek",
           "first": "first visit", "every": "every visit",
           "alt": "ROOM, 5 bij 5 tegels met het laadstation in het midden, met de first-visit-schatting van elke tegel die in de eerste {n} {word} bezocht is: de hoek {c}, boven het laadstation {a}; {left}.",
           "left_none": "elke tegel is bezocht",
           "left_some": "{k} {tiles} nog niet bezocht",
           "tile": "tegel", "tiles": "tegels",
           "chart_alt": "De schatting van de hoek na elke wandeling van 1 tot 1.000, op een logschaal: first visit, doorgetrokken, en every visit, gestreept, allebei eerst flink schommelend en dan rustig rond de waarde −36, een grijze lijn, tussen twee gestreepte grijze lijnen die er steeds dichter omheen komen; een markering bij wandeling {n}, waar first visit op {f} staat en every visit op {e}.",
           "say": "Na {n} {word}: de hoek staat op {f} met first visit en op {e} met every visit, met first visit {gap} van −36; de tegel boven het laadstation staat op {a}.",
           "note": "Grijs: het laadstation, met de waarde 0; in de grafiek de waarde −36 van policy evaluation, en gestreept één spreiding van het gemiddelde erboven en eronder, 31,7/√n bij n wandelingen. Blauw: de schatting van elke tegel na deze wandelingen, het gemiddelde van de returns na de eerste bezoeken; een tegel zonder getal is nog niet bezocht. In de grafiek de hoek na elke wandeling, doorgetrokken met first visit en gestreept met every visit."},
}


def _walks_word(n: int, w: dict) -> str:
    return w["one"] if n == 1 else w["many"]


def _corner_panel(n: int, lang: str) -> chart.Panel:
    w = _PLAY_TEXT[lang]
    p = chart.Panel((1, N_WALKS), (-80, 0), title=w["chart"], xlabel=w["x"], ylabel=w["y"], xlog=True,
                    yticks=[-80, -60, -40, -20, 0])
    band = [round(10 ** (k / 20), 3) for k in range(61)]   # 1 to 1000, evenly on the log scale
    for side in (-1, 1):
        p.line(band, [END[CORNER] + side * SPREAD / math.sqrt(x) for x in band], "ink-3", 1.2, dashed=True)
    p.line([1, N_WALKS], [END[CORNER], END[CORNER]], "ink-3", 1.4)
    p.line([n, n], [-80, 0], "axis", 1.2)
    xs = list(range(1, N_WALKS + 1))
    p.line(xs, FV_CORNER, "pred", 2.2)
    p.line(xs, EV_CORNER, "pred", 1.8, dashed=True)
    p.points([n], [FV_CORNER[n - 1]], "pred", r=3.8)
    p.points([n], [EV_CORNER[n - 1]], "pred", r=3.8, hollow=True)
    return p


def play_frame(n: int, lang: str) -> tuple:
    """The play block after n walks: ROOM with the first-visit estimates, the chart of the corner beside it, and a
    sentence."""
    lang = lang if lang in LANGS else "en"
    w = _PLAY_TEXT[lang]
    n = min(LOOKS, key=lambda k: abs(k - int(n)))
    V = PLAY[n]
    word = _walks_word(n, w)
    missing = len(STATES) - len([s for s in V if s != CHARGER])
    left = w["left_none"] if missing == 0 else w["left_some"].format(k=missing, tiles=w["tile"] if missing == 1 else w["tiles"])
    f, e, a = FV_CORNER[n - 1], EV_CORNER[n - 1], V.get(NEXT)
    a_text = number(a, 1, lang) if a is not None else "–"
    count = _dec(n, lang, 0)
    grid = _grid([_room_panel(V, w["title"].format(n=count, word=word), digits=1)],
                 w["alt"].format(n=count, word=word, c=number(f, 1, lang), a=a_text, left=left), lang)
    line = _svg([_corner_panel(n, lang)], w["chart_alt"].format(n=count, f=number(f, 1, lang), e=number(e, 1, lang)), lang,
                width=330, height=250)
    body = (f'<div style="display:flex;flex-wrap:wrap;gap:0.5rem 1.5rem;align-items:flex-start">'
            f'<div style="flex:0 1 300px">{grid}</div><div style="flex:1 1 300px;max-width:330px">{line}</div></div>')
    say = w["say"].format(n=count, word=word, f=number(f, 1, lang), e=number(e, 1, lang),
                          gap=number(abs(f - END[CORNER]), 1, lang), a=a_text)
    return body + _note(w["note"]), say


# --- the pictures of the learner's work ---------------------------------------------------------------------------------

_STOPPED = object()   # what _call gives for a function that stopped with an error
_SLOW = object()      # what _call gives for a function that ran out of time
PICTURE_SECONDS = 3   # what the learner's function gets for all the calls of one picture together


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


def _numbers(got, n: int) -> list | None:
    """got as a list of n finite floats, or None."""
    if not isinstance(got, (list, tuple)) or len(got) != n:
        return None
    try:
        out = [float(x) for x in got]
    except (TypeError, ValueError):
        return None
    return out if all(math.isfinite(x) for x in out) else None


def _estimates(got, want: dict) -> dict | None:
    """got as {state: float} for the same states as want, or None."""
    if not isinstance(got, dict) or set(got) != set(want):
        return None
    try:
        out = {s: float(got[s]) for s in want}
    except (TypeError, ValueError):
        return None
    return out if all(math.isfinite(v) for v in out.values()) else None


RETURN_GAMMAS = (1.0, 0.9)   # the discount factors of the picture of exercise 1

_RETURNS_TEXT = {
    "en": {"title": "Your returns of walk 1",
           "head": "Your `returns` on walk 1 of the play block, {n} steps from the corner to the charger, with γ = 1 and with γ = 0.9.",
           "chart": "The return from every step", "x": "step t", "y": "return",
           "alt": "The return from every step of walk 1 as your returns gives it: with γ = 1 from {a} at step 0 to {b} at the last step, solid; with γ = 0.9, dashed, it starts at {c}.",
           "start": "From step 0: {a} with γ = 1, the whole walk, and {c} with γ = 0.9.",
           "same": "Your returns are the course's at every step.",
           "differ": "With γ = {g} your return from step {t} is {mine}, and the course's {want}.",
           "none": "Your `returns` stops with an error, takes too long or does not give one number for every step here, so nothing can be drawn yet.",
           "note": "Purple: the return from every step, solid with γ = 1 and dashed with γ = 0.9. With γ = 1 it is minus the number of steps still to come; with γ = 0.9, steps far ahead count so little that the return from the early steps stays close to −10."},
    "nl": {"title": "Je returns van wandeling 1",
           "head": "Je `returns` op wandeling 1 van het speelblok, {n} stappen van de hoek naar het laadstation, met γ = 1 en met γ = 0,9.",
           "chart": "De return vanaf elke stap", "x": "stap t", "y": "return",
           "alt": "De return vanaf elke stap van wandeling 1 zoals je returns de return geeft: met γ = 1 van {a} bij stap 0 tot {b} bij de laatste stap, doorgetrokken; met γ = 0,9, gestreept, begint de lijn bij {c}.",
           "start": "Vanaf stap 0: {a} met γ = 1, de hele wandeling, en {c} met γ = 0,9.",
           "same": "Je returns zijn bij elke stap gelijk aan de returns van de course.",
           "differ": "Met γ = {g} is je return vanaf stap {t} {mine}, en de return van de course {want}.",
           "none": "Je `returns` stopt met een fout, duurt te lang of geeft hier niet voor elke stap één getal, dus er is nog geen afbeelding.",
           "note": "Paars: de return vanaf elke stap, doorgetrokken met γ = 1 en gestreept met γ = 0,9. Met γ = 1 is de return min het aantal stappen dat nog komt; met γ = 0,9 tellen stappen ver vooruit zo weinig mee dat de return vanaf de vroege stappen dicht bij −10 blijft."},
}


def _gamma_text(g: float, lang: str) -> str:
    out = f"{g:g}"
    return out.replace(".", ",") if lang == "nl" else out


def returns_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _RETURNS_TEXT[lang]
    fn = _fn(seen)
    n = len(FIRST_WALK)
    got = None
    if fn is not None:
        got = _call(lambda: [_numbers(fn(copy.deepcopy(FIRST_WALK), g), n) for g in RETURN_GAMMAS])
        if not isinstance(got, list) or any(v is None for v in got):
            got = None
    head = _note(w["head"].format(n=n))
    if got is None:
        return yours.figure(w["title"], head + _note(w["none"], _LOSS_UI), alt=w["title"])
    lo = min(-n - 2, math.floor(min(min(v) for v in got) / 10) * 10)
    hi = max(0, math.ceil(max(max(v) for v in got) / 10) * 10)
    p = chart.Panel((0, n - 1), (lo, hi), title=w["chart"], xlabel=w["x"], ylabel=w["y"])
    xs = list(range(n))
    p.line(xs, got[0], "loss", 2.2)
    p.line(xs, got[1], "loss", 2.0, dashed=True)
    span = hi - lo
    k = int(0.12 * (n - 1))   # each line's name: γ = 0.9 above its line near the start, γ = 1 below its line further on
    p.text(k, min(got[1][k] + 0.11 * span, hi - 0.04 * span), "γ = " + _gamma_text(RETURN_GAMMAS[1], lang), "ink-2")
    k = int(0.45 * (n - 1))
    p.text(k, max(got[0][k] - 0.1 * span, lo + 0.05 * span), "γ = " + _gamma_text(RETURN_GAMMAS[0], lang), "ink-2")
    a, b, c = number(got[0][0], 2, lang), number(got[0][-1], 2, lang), number(got[1][0], 2, lang)
    body = head + _svg([p], w["alt"].format(a=a, b=b, c=c), lang, width=460, height=250)
    body += _note(w["start"].format(a=a, c=c))
    off = None
    for g, mine in zip(RETURN_GAMMAS, got):
        want = returns(FIRST_WALK, g)
        t = next((t for t in range(n) if abs(mine[t] - want[t]) > 1e-6), None)
        if t is not None:
            off = (g, t, mine[t], want[t])
            break
    if off is None:
        body += _note(w["same"])
    else:
        g, t, mine_, want_ = off
        body += _note(w["differ"].format(g=_gamma_text(g, lang), t=t, mine=number(mine_, 2, lang), want=number(want_, 2, lang)), _LOSS_UI)
    return yours.figure(w["title"], body + _note(w["note"]), alt=w["title"])


FIRST_LOOKS = ((TEN_WALKS, 10), (WALKS, N_WALKS))   # the walks of the picture of exercise 2
FIRST_WANT = {k: first_visit(walks, GAMMA) for walks, k in FIRST_LOOKS}

_FIRST_TEXT = {
    "en": {"title": "Your first_visit on ROOM",
           "head": "Your `first_visit` on the walks of the play block: the first 10, and all 1,000.",
           "panel": "after {k} walks",
           "alt": "ROOM twice, with the estimates your first_visit gives after 10 and after 1,000 walks: in the corner {a} and {b}, above the charger {c} and {d}.",
           "gaps": "From the values of policy evaluation, your estimates lie at most {g10} away after 10 walks, and at most {g} after 1,000.",
           "differ": "After {k} walks your estimate of tile {s} is {mine}, and the course's {want}.",
           "none": "Your `first_visit` stops with an error, takes too long or does not give an estimate for exactly the visited tiles here, so nothing can be drawn yet.",
           "note": "Gray: the charger, with the value 0. Blue: your estimates, the average of the returns after the first visits of each tile; policy evaluation gives −24 next to the charger, −30 diagonal to it, −32 and −34 along the edges and −36 in the corners."},
    "nl": {"title": "Je first_visit op ROOM",
           "head": "Je `first_visit` op de wandelingen van het speelblok: de eerste 10, en alle 1.000.",
           "panel": "na {k} wandelingen",
           "alt": "Twee keer ROOM, met de schattingen die je first_visit geeft na 10 en na 1.000 wandelingen: in de hoek {a} en {b}, boven het laadstation {c} en {d}.",
           "gaps": "Je schattingen liggen na 10 wandelingen hoogstens {g10} van de waarden van policy evaluation, en na 1.000 hoogstens {g}.",
           "differ": "Na {k} wandelingen is je schatting van tegel {s} {mine}, en de schatting van de course {want}.",
           "none": "Je `first_visit` stopt met een fout, duurt te lang of geeft hier niet precies voor de bezochte tegels een schatting, dus er is nog geen afbeelding.",
           "note": "Grijs: het laadstation, met de waarde 0. Blauw: je schattingen, het gemiddelde van de returns na de eerste bezoeken van elke tegel; policy evaluation geeft −24 naast het laadstation, −30 er schuin naast, −32 en −34 langs de randen en −36 in de hoeken."},
}


def first_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _FIRST_TEXT[lang]
    fn = _fn(seen)
    got = None
    if fn is not None:
        got = _call(lambda: [_estimates(fn(copy.deepcopy(walks), GAMMA), FIRST_WANT[k]) for walks, k in FIRST_LOOKS])
        if not isinstance(got, list) or any(v is None for v in got):
            got = None
    if got is None:
        return yours.figure(w["title"], _note(w["head"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    alt = w["alt"].format(a=number(got[0][CORNER], 1, lang), b=number(got[1][CORNER], 1, lang),
                          c=number(got[0][NEXT], 1, lang), d=number(got[1][NEXT], 1, lang))
    grid = _grid([_room_panel(V, w["panel"].format(k=_dec(k, lang, 0)), digits=1) for V, (_w, k) in zip(got, FIRST_LOOKS)],
                 alt, lang, cell=46)
    body = _note(w["head"]) + grid
    body += _note(w["gaps"].format(g10=number(largest_gap(got[0]), 1, lang), g=number(largest_gap(got[1]), 2, lang)))
    off = None
    for V, (_w, k) in zip(got, FIRST_LOOKS):
        want = FIRST_WANT[k]
        s = next((s for s in want if abs(V[s] - want[s]) > 1e-6), None)
        if s is not None:
            off = (k, s, V[s], want[s])
            break
    if off is not None:
        k, s, mine_, want_ = off
        body += _note(w["differ"].format(k=_dec(k, lang, 0), s=f"({s[0]}, {s[1]})", mine=number(mine_, 2, lang),
                                         want=number(want_, 2, lang)), _LOSS_UI)
    return yours.figure(w["title"], body + _note(w["note"]), alt=w["title"])


# --- how an answer is judged --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Judged:
    """What returns or first_visit gave back, as far as the check is concerned: the numbers rounded to six decimals, and
    `off`, where they first differ from the course's answer."""
    kind: str                 # "ok"; "shape": not what the function should give; "length" or "states": too few or too many
    values: tuple = ()        # returns: (G_0, G_1, ...); first_visit: ((state, estimate), ...) in the order of first visit
    off: tuple | None = None  # (place or state, its number, the course's)
    detail: tuple = ()        # "length": (got, want); "states": (missing, extra)


def _r6(x) -> float:
    return round(float(x), 6) + 0.0


def judge_returns(episode, gamma, value):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        return Judged("shape")
    if len(value) != len(episode):
        return Judged("length", detail=(len(value), len(episode)))
    try:
        values = tuple(_r6(x) for x in value)
    except (TypeError, ValueError):
        return Judged("shape")
    if not all(math.isfinite(x) for x in values):
        return Judged("shape")
    right = returns(episode, gamma)
    off = next(((t, v, right[t]) for t, v in enumerate(values) if abs(v - right[t]) > 1e-6), None)
    return Judged("ok", values, off)


def _visited(episodes) -> list:
    """The states the episodes visit, in the order of their first visit."""
    out, seen = [], set()
    for episode in episodes:
        for s, _r in episode:
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


def judge_first(episodes, gamma, value):
    if value is None:
        return None
    if not isinstance(value, dict):
        return Judged("shape")
    order = _visited(episodes)
    if set(value) != set(order):
        missing = tuple(s for s in order if s not in value)
        extra = len([s for s in value if s not in set(order)])
        return Judged("states", detail=(missing, extra))
    try:
        values = tuple((s, _r6(value[s])) for s in order)
    except (TypeError, ValueError):
        return Judged("shape")
    if not all(math.isfinite(v) for _s, v in values):
        return Judged("shape")
    right = first_visit(episodes, gamma)
    off = next(((s, v, right[s]) for s, v in values if abs(v - right[s]) > 1e-6), None)
    return Judged("ok", values, off)


def same_judged(a, b) -> bool:
    """Right when the numbers agree within a millionth, for the same places or states."""
    if not (isinstance(a, Judged) and isinstance(b, Judged)):
        return a == b
    if a.kind != b.kind or a.detail != b.detail:
        return False
    if a.kind != "ok" or len(a.values) != len(b.values):
        return a.kind != "ok"
    if a.values and isinstance(a.values[0], tuple):
        return all(sa == sb and abs(va - vb) <= 1e-6 for (sa, va), (sb, vb) in zip(a.values, b.values))
    return all(abs(va - vb) <= 1e-6 for va, vb in zip(a.values, b.values))


def _plain(x: float) -> str:
    """A number the way the check shows it: up to four decimals, a real minus sign."""
    text = f"{x:.4f}".rstrip("0").rstrip(".")
    if text in ("-0", "-0."):
        text = "0"
    return text.replace("-", "−")


def _state(s) -> str:
    return f"({s[0]}, {s[1]})" if isinstance(s, tuple) else str(s)


def show_returns(judged) -> dict:
    """What returns gave back, in words: the whole list when it is short, or where it first differs."""
    if not isinstance(judged, Judged) or judged.kind == "shape":
        return {"en": "something other than a list with a number for every step",
                "nl": "iets anders dan een list met een getal voor elke stap"}
    if judged.kind == "length":
        got, want = judged.detail
        return {"en": f"a list of {got} numbers, for an episode of {want} steps",
                "nl": f"een list van {got} getallen, voor een episode van {want} stappen"}
    if len(judged.values) <= 8:
        shown = "`[" + ", ".join(_plain(x) for x in judged.values) + "]`"
        if judged.off is None:
            return {"en": shown, "nl": shown}
        t, _got, want = judged.off
        return {"en": f"{shown}, where place {t} should be `{_plain(want)}`",
                "nl": f"{shown}, waar plek {t} `{_plain(want)}` moet zijn"}
    if judged.off is not None:
        t, got, want = judged.off
        return {"en": f"a list with `{_plain(got)}` at place {t}, where it should be `{_plain(want)}`",
                "nl": f"een list met `{_plain(got)}` op plek {t}, waar het `{_plain(want)}` moet zijn"}
    return {"en": "the right return from every step", "nl": "de goede return vanaf elke stap"}


def show_first(judged) -> dict:
    """What first_visit gave back, in words: where it first differs from the course's answer."""
    if not isinstance(judged, Judged) or judged.kind == "shape":
        return {"en": "something other than a dictionary with a number for every visited state",
                "nl": "iets anders dan een dictionary met een getal voor elke bezochte state"}
    if judged.kind == "states":
        missing, extra = judged.detail
        if missing:
            return {"en": f"a dictionary without an estimate of `{_state(missing[0])}`, which the episodes visit",
                    "nl": f"een dictionary zonder schatting van `{_state(missing[0])}`, terwijl de episodes daar komen"}
        return {"en": "a dictionary with states that the episodes do not visit",
                "nl": "een dictionary met states waar de episodes niet komen"}
    if judged.off is not None:
        s, got, want = judged.off
        return {"en": f"`V[{_state(s)}] = {_plain(got)}`, where it should be `{_plain(want)}`",
                "nl": f"`V[{_state(s)}] = {_plain(got)}`, waar het `{_plain(want)}` moet zijn"}
    if not judged.values:
        return {"en": "an empty dictionary, the right answer for no episodes",
                "nl": "een lege dictionary, het goede antwoord bij geen episodes"}
    return {"en": "the right estimate for every visited state", "nl": "de goede schatting voor elke bezochte state"}


@dataclass
class JudgedBox(FunctionBox):
    """A function in a code box whose answer is a list or dictionary of numbers: the check calls it on each case and
    judges what it gives back, so that a known mistake is known by what it does in every case."""
    judge: object = None

    def _run(self, fn, case):
        return self.judge(*case, super()._run(fn, case))


# --- the exercises -----------------------------------------------------------------------------------------------------

REQUIRED = ["returns", "first", "credit"]


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


# exercise 1: returns
def _r(line: str = "G = r + gamma * G") -> str:
    """returns as the box holds it, with this line: step 2 of the recipe, the return from step t."""
    return ("def returns(episode, gamma):\n"
            "    out = [0] * len(episode)\n"
            "    G = 0\n"
            "    for t in range(len(episode) - 1, -1, -1):\n"
            "        s, r = episode[t]\n"
            f"        {line}\n"
            "        out[t] = G\n"
            "    return out\n")


_R_START = _r("G = ...")
_R = _r()
_R_NO_GAMMA = _r("G = r + G")
_R_FIRST_TOO = _r("G = gamma * (r + G)")
_R_REWARD = _r("G = r")
_R_NEXT_PLACE = _r("G = r + gamma * out[t + 1]")

RETURNS_CASES = [
    (WALK, 0.9),
    (WALK, 1.0),
    (PRIZE_WALK, 0.9),
    (FIRST_WALK, 1.0),
    (Episode([(1, 10)], "[(1, 10)]"), 0.5),
    (Episode([], "[]"), 0.5),
]

# exercise 2: first_visit
def _f(when: str = "s not in seen", value: str = "total[s] / count[s]") -> str:
    """first_visit as the box holds it, with these two parts: when a step counts, and the estimate of a state."""
    return ("def first_visit(episodes, gamma):\n"
            "    total = {}\n"
            "    count = {}\n"
            "    for episode in episodes:\n"
            "        G = returns(episode, gamma)\n"
            "        seen = set()\n"
            "        for t in range(len(episode)):\n"
            "            s = episode[t][0]\n"
            f"            if {when}:\n"
            "                seen.add(s)\n"
            "                if s not in total:\n"
            "                    total[s] = 0\n"
            "                    count[s] = 0\n"
            "                total[s] = total[s] + G[t]\n"
            "                count[s] = count[s] + 1\n"
            "    V = {}\n"
            "    for s in total:\n"
            f"        V[s] = {value}\n"
            "    return V\n")


_F_START = _f("...", "...")
_F = _f()
_F_EVERY = _f("True")
_F_INVERTED = _f("s in seen")
_F_START_ONLY = _f("t == 0")
_F_SUM = _f(value="total[s]")
_F_PER_WALK = _f(value="total[s] / len(episodes)")
_F_EVER = _f("s not in total")   # the first visit in all walks together
_F_PAIRS = _f("episode[t] not in episode[:t]")   # right where every reward is the same, as in ROOM

FIRST_CASES = [
    (TEN_WALKS, 1.0),
    (Walks([WALK], "[WALK]"), 1.0),
    (Walks([WALK, PRIZE_WALK], "[WALK, PRIZE_WALK]"), 0.9),
    (TEN_WALKS, 0.9),
    (WALKS, 1.0),
    (Walks([], "[]"), 1.0),
]


EXERCISES = {
    "returns": JudgedBox(
        name="returns",
        picture=returns_picture,
        function="returns",
        start=_R_START,
        solution=_R,
        judge=judge_returns,
        compare=same_judged,
        show=show_returns,
        visible=RETURNS_CASES[:1],
        hidden=RETURNS_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("no-gamma", wrong=_R_NO_GAMMA,
                    en="`{call}` gives {got}: every reward counts in full, however far ahead it comes. With γ = 0.9 a reward one step later counts 0.9 times, two steps later 0.81 times; where in your line does `gamma` make the return from the next step count less?",
                    nl="`{call}` geeft {got}: elke reward telt helemaal mee, hoe ver vooruit de reward ook komt. Met γ = 0,9 telt een reward een stap later 0,9 keer mee, twee stappen later 0,81 keer; waar in je regel zorgt `gamma` dat de return vanaf de volgende stap minder telt?"),
            Mistake("first-too", wrong=_R_FIRST_TOO,
                    en="`{call}` gives {got}: your line multiplies the reward of step `t` by `gamma` too, so even the last step gives −0.9 instead of −1. The reward right after a step counts in full, and only what comes later is discounted; which part of your line should `gamma` leave alone?",
                    nl="`{call}` geeft {got}: je regel vermenigvuldigt ook de reward van stap `t` met `gamma`, dus zelfs de laatste stap geeft −0,9 in plaats van −1. De reward direct na een stap telt helemaal mee, en alleen wat later komt telt minder; welk deel van je regel moet `gamma` met rust laten?"),
            Mistake("reward-only", wrong=_R_REWARD,
                    en="`{call}` gives {got}: every step gets only its own reward, so `G` forgets what came after it. The return from step `t` is its reward plus everything that follows, discounted; what does `G` hold from the step after `t` when your line runs?",
                    nl="`{call}` geeft {got}: elke stap krijgt alleen de eigen reward, dus vergeet `G` wat erna kwam. De return vanaf stap `t` is de reward van de stap plus alles wat volgt, minder zwaar geteld; wat bevat `G` van de stap na `t` als je regel draait?"),
            Mistake("next-place", error=lambda e: isinstance(e, IndexError),
                    en="With `{call}`, your function stops with an error: at the last step, place `t + 1` lies past the end of the list. The loop goes backward, so when your line runs for step `t`, `G` still holds the return from the step after it; can your line use `G` instead?",
                    nl="Met `{call}` stopt je functie met een fout: bij de laatste stap ligt plek `t + 1` voorbij het eind van de list. De loop gaat achteruit, dus als je regel voor stap `t` draait, bevat `G` nog de return vanaf de stap erna; kan je regel `G` gebruiken?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the return from step `t`: its reward plus the return from the step after it, discounted.",
                      "nl": "Vervang de `...` door de return vanaf stap `t`: de reward van de stap plus de return vanaf de stap erna, minder zwaar geteld."},
            "arguments": {"en": "The check calls `{call}`, with an episode and a discount factor, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(episode, gamma)`?",
                          "nl": "De check roept `{call}` aan, met een episode en een discount factor, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(episode, gamma)` houden?"},
            "slow": {"en": "Your function is still not done after 2 seconds. Does the loop go over `range(len(episode) - 1, -1, -1)`, with `for`?",
                     "nl": "Je functie is na 2 seconden nog niet klaar. Gaat de loop over `range(len(episode) - 1, -1, -1)`, met `for`?"},
            "wrong": {"en": "`{call}` gives {got}. Is the return from step `t` the reward of step `t` plus `gamma` times the return from the step after it?",
                      "nl": "`{call}` geeft {got}. Is de return vanaf stap `t` de reward van stap `t` plus `gamma` keer de return vanaf de stap erna?"},
            "almost": {"en": "Right for `WALK` with γ = 0.9. But `{call}` gives {got}. Does your function work for every episode, with every reward, and every discount factor?",
                       "nl": "Goed voor `WALK` met γ = 0,9. Maar `{call}` geeft {got}. Werkt je functie voor elke episode, met elke reward, en elke discount factor?"},
            "no_return": {"en": "With `{call}`, your function gives back `None`, and it should give the list of returns. Does it still end with `return out`?",
                          "nl": "Met `{call}` geeft je functie `None` terug, en het moet de list met returns zijn. Eindigt je functie nog met `return out`?"},
            "correct": {"en": "Right in all {n} cases: in `WALK`, with γ = 0.9, the return from the first step is −4.6856, six steps of −1, each later one counting a bit less. In the picture your function works out the returns of walk 1 of the play block; why does the dashed line hardly move over the first steps?",
                        "nl": "Goed in alle {n} gevallen: in `WALK`, met γ = 0,9, is de return vanaf de eerste stap −4,6856, zes stappen van −1, elke latere stap iets minder zwaar. In de afbeelding rekent je functie de returns van wandeling 1 van het speelblok uit; waarom beweegt de gestreepte lijn de eerste stappen nauwelijks?"},
        },
    ),
    "first": JudgedBox(
        name="first",
        picture=first_picture,
        function="first_visit",
        names={"returns": returns},
        start=_F_START,
        solution=_F,
        judge=judge_first,
        compare=same_judged,
        show=show_first,
        visible=FIRST_CASES[:1],
        hidden=FIRST_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("every-visit", wrong=_F_EVERY,
                    en="`{call}` gives {got}: every visit of a state counts, also when the walk comes back to it. That is every-visit Monte Carlo, a method of its own, and over many walks it ends up close to the same values. This exercise asks for first-visit: per walk, only the return after the first visit counts; what is `seen` for?",
                    nl="`{call}` geeft {got}: elk bezoek aan een state telt, ook als de wandeling er terugkomt. Dat is every-visit Monte Carlo, een methode op zich, en over veel wandelingen komt every-visit bij dezelfde waarden uit. Deze oefening vraagt first-visit: per wandeling telt alleen de return na het eerste bezoek; waar is `seen` voor?"),
            Mistake("inverted", wrong=_F_INVERTED,
                    en="`{call}` gives {got}: no step ever counts. `seen` starts empty in every walk, and a state gets into it only inside your `if`, so `s in seen` is never true there. Which states should count: those already in `seen`, or those not in it yet?",
                    nl="`{call}` geeft {got}: geen enkele stap telt. `seen` begint leeg in elke wandeling, en een state komt er alleen in binnen je `if`, dus `s in seen` is daar nooit waar. Welke states moeten tellen: de states die al in `seen` staan, of de states die er nog niet in staan?"),
            Mistake("first-ever", wrong=_F_EVER,
                    en="`{call}` gives {got}: only the first visit of a state in all walks together counts. `total` keeps its states from walk to walk, while `seen` starts empty in every walk; which of the two says whether this walk has been in `s` before?",
                    nl="`{call}` geeft {got}: alleen het eerste bezoek aan een state in alle wandelingen samen telt. `total` houdt de states van wandeling tot wandeling bij, terwijl `seen` in elke wandeling leeg begint; welke van de twee zegt of deze wandeling al eerder in `s` was?"),
            Mistake("start-only", wrong=_F_START_ONLY,
                    en="`{call}` gives {got}: only step 0 of each walk counts, so only the start gets an estimate. Every other tile is visited too, and its first visit gives a return as well; which steps are the first visit of their state?",
                    nl="`{call}` geeft {got}: alleen stap 0 van elke wandeling telt, dus alleen de start krijgt een schatting. Elke andere tegel wordt ook bezocht, en ook het eerste bezoek daar geeft een return; welke stap van een wandeling is het eerste bezoek aan een state?"),
            Mistake("sum", wrong=_F_SUM,
                    en="`{call}` gives {got}: all returns of a state added up, not their average. The estimate of a state is the return to expect from it; how many returns has `count[s]` counted?",
                    nl="`{call}` geeft {got}: de som van alle returns van een state, geen gemiddelde. De schatting van een state is de return die je ervandaan kunt verwachten; hoeveel returns heeft `count[s]` geteld?"),
            Mistake("per-walk", wrong=_F_PER_WALK,
                    en="`{call}` gives {got}: you divide by the number of walks. That is right for the corner, where every walk starts, but a walk of 4 steps visits only 4 tiles: the other tiles got no return from it. How many returns does `count[s]` say a tile got?",
                    nl="`{call}` geeft {got}: je deelt door het aantal wandelingen. Voor de hoek klopt dat, want elke wandeling begint daar. Een wandeling van 4 stappen bezoekt echter slechts 4 tegels: de andere tegels kregen er geen return uit. Hoeveel returns kreeg een tegel volgens `count[s]`?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: when a step counts, its state's first visit in the walk, and the estimate of `s`, the average of its returns.",
                      "nl": "Vervang de twee `...`: wanneer een stap telt, het eerste bezoek aan de state in de wandeling, en de schatting van `s`, het gemiddelde van de returns."},
            "arguments": {"en": "The check calls `{call}`, with a list of episodes and a discount factor, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(episodes, gamma)`?",
                          "nl": "De check roept `{call}` aan, met een list van episodes en een discount factor, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(episodes, gamma)` houden?"},
            "slow": {"en": "Your function is still not done after 2 seconds. Does every loop go over the episodes, an episode or a dictionary, with `for`?",
                     "nl": "Je functie is na 2 seconden nog niet klaar. Gaat elke loop over de episodes, een episode of een dictionary, met `for`?"},
            "wrong": {"en": "`{call}` gives {got}. Does a step count only at the first visit of its state in the walk, and is the estimate of a state the average of its returns?",
                      "nl": "`{call}` geeft {got}. Telt een stap alleen bij het eerste bezoek aan de state in de wandeling, en is de schatting van een state het gemiddelde van de returns?"},
            "almost": {"en": "Right for `TEN_WALKS`. But `{call}` gives {got}. Does your function work for every list of episodes, with every reward, and every discount factor?",
                       "nl": "Goed voor `TEN_WALKS`. Maar `{call}` geeft {got}. Werkt je functie voor elke list van episodes, met elke reward, en elke discount factor?"},
            "no_return": {"en": "With `{call}`, your function gives back `None`, and it should give the estimates. Does it still end with `return V`?",
                          "nl": "Met `{call}` geeft je functie `None` terug, en het moeten de schattingen zijn. Eindigt je functie nog met `return V`?"},
            "correct": {"en": "Right in all {n} cases: after the ten walks, your estimate of the corner is −26.9, still 9.1 from −36. In the picture your function goes on to all 1,000 walks; how far is the corner from −36 then?",
                        "nl": "Goed in alle {n} gevallen: na de tien wandelingen is je schatting van de hoek −26,9, nog 9,1 van −36. In de afbeelding gaat je functie door tot alle 1.000 wandelingen; hoe ver zit de hoek dan van −36?"},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("returns", _R_START, "empty", None),
    ("returns", _R, "correct", None),
    ("returns", _r("G = r + ... * G"), "wrong", "dots-left"),
    ("returns", _R_NO_GAMMA, "wrong", "no-gamma"),
    ("returns", _R_FIRST_TOO, "wrong", "first-too"),
    ("returns", _R_REWARD, "wrong", "reward-only"),
    ("returns", _R_NEXT_PLACE, "error", "next-place"),
    ("returns", _R.replace("    return out\n", ""), "wrong", "no-return"),
    # another right answer: the same sum in the other order
    ("returns", _r("G = gamma * G + r"), "correct", None),
    # a reward of −1 written into the line: right for WALK, wrong once a step pays +10
    ("returns", _r("G = -1 + gamma * G"), "almost", None),
    ("first", _F_START, "empty", None),
    ("first", _F, "correct", None),
    ("first", _f(value="..."), "wrong", "dots-left"),
    ("first", _F_EVERY, "wrong", "every-visit"),
    ("first", _F_INVERTED, "wrong", "inverted"),
    ("first", _F_START_ONLY, "wrong", "start-only"),
    ("first", _F_EVER, "wrong", "first-ever"),
    ("first", _F_SUM, "wrong", "sum"),
    ("first", _F_PER_WALK, "wrong", "per-walk"),
    ("first", _F.replace("    return V\n", ""), "wrong", "no-return"),
    # other right answers: the states of the steps before t, looked up in the episode itself
    ("first", _f("s not in [step[0] for step in episode[:t]]"), "correct", None),
    # a step looked up with its reward: right where every reward is −1, wrong once a reward differs
    ("first", _F_PAIRS, "almost", None),
]
