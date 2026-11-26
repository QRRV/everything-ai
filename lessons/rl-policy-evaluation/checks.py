# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rl-policy-evaluation: a robot that looks for its charger on a grid of tiles, every step costing
−1, so that the value of a tile under a policy is minus the average number of steps from there to the charger; the
course's own policy evaluation, sweep after sweep; evaluate and until_stable, the functions the learner writes, and
their pictures; the play block over the sweeps.

Policy evaluation as in Sutton and Barto (Reinforcement Learning: An Introduction, 2nd edition, 2018, section 4.1).
The room, the robot, the runs and the pictures are the course's own."""
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
THETA = 0.01     # the stop of the example of exercise 2
END_SWEEPS = 2000   # sweeps after which the course takes the values as the end values


# --- the worlds, their tables and the policies -----------------------------------------------------------------------

class World(list):
    """A grid of tiles as a list of strings, one per row: '.' a tile, G the charger. A check message calls it by the
    name the lesson gives it."""

    def __init__(self, rows, name: str):
        super().__init__(rows)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return list(self)


class Table(dict):
    """The table T of a world: T[state][action] is the list of what can happen, as (probability, next state, reward),
    as in value iteration. A check message calls it by its name, such as T_ROOM; the learner's code gets a plain copy."""

    def __init__(self, entries, name: str, world: World):
        super().__init__(entries)
        self.name, self.world = name, world

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return {s: {a: list(outs) for a, outs in acts.items()} for s, acts in self.items()}


class Policy(dict):
    """A policy as pi[state][action] = the probability that it takes that action there. A check message calls it by its
    name, such as RANDOM; the learner's code gets a plain copy."""

    def __init__(self, entries, name: str):
        super().__init__(entries)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return {s: dict(acts) for s, acts in self.items()}


ROOM = World([       # the room of the lesson: the charger in the middle tile
    ".....",
    ".....",
    "..G..",
    ".....",
    ".....",
], "ROOM")
CORRIDOR = World(["G..."], "CORRIDOR")   # the worked example: the charger at the left end, a wall at the right end
SMALL = World([      # a check case: the charger in a corner, the last state of the table
    "...",
    "..G",
], "SMALL")


def find(world, sign: str) -> tuple[int, int]:
    """The tile (row, column) with this sign."""
    for r, row in enumerate(world):
        if sign in row:
            return (r, row.index(sign))
    raise ValueError(sign)


def table(world, actions=tuple(ACTIONS), name: str = "") -> Table:
    """The table of a world: an action goes where it says, and into the edge the robot stays where it is; every step
    costs −1. On the charger the task is over: every action stays there, with reward 0."""
    T = {}
    for r, row in enumerate(world):
        for c, sign in enumerate(row):
            state = (r, c)
            T[state] = {}
            for a in actions:
                if sign == "G":
                    T[state][a] = [(1.0, state, 0)]
                    continue
                nr, nc = r + MOVES[a][0], c + MOVES[a][1]
                nxt = (nr, nc) if 0 <= nr < len(world) and 0 <= nc < len(world[0]) else state
                T[state][a] = [(1.0, nxt, -1)]
    return Table(T, name or "T_" + world.name, world)


def uniform(T, name: str) -> Policy:
    """The random policy of a table: every action of a state with the same probability."""
    return Policy({s: {a: 1 / len(acts) for a in acts} for s, acts in T.items()}, name)


T_ROOM = table(ROOM)
T_CORRIDOR = table(CORRIDOR, actions=("left", "right"))
T_SMALL = table(SMALL)
RANDOM = uniform(T_ROOM, "RANDOM")
LEFT_RIGHT = uniform(T_CORRIDOR, "LEFT_RIGHT")
RANDOM_SMALL = uniform(T_SMALL, "RANDOM_SMALL")
# a check case: the corridor where the step onto the charger pays +10, so that values rise and fall
T_PRIZE = Table({s: {a: [(p, nxt, 10 if nxt == (0, 0) and s != (0, 0) else r) for p, nxt, r in outs]
                     for a, outs in acts.items()} for s, acts in T_CORRIDOR.items()}, "T_PRIZE", CORRIDOR)
MOSTLY_UP = Policy({s: {"up": 0.4, "down": 0.2, "left": 0.2, "right": 0.2} for s in T_ROOM}, "MOSTLY_UP")
UP = Policy({s: {"up": 1.0, "down": 0.0, "left": 0.0, "right": 0.0} for s in T_ROOM}, "UP")   # exercise 3

CHARGER = find(ROOM, "G")
NEXT = (1, 2)      # a tile next to the charger
DIAGONAL = (1, 1)  # a tile diagonal to it
EDGE = (0, 2)      # the middle of an edge
CORNER = (0, 0)    # a corner


# --- the course's policy evaluation --------------------------------------------------------------------------------------

def sweep(T, pi, gamma, V) -> dict:
    """One sweep: for every state the average over the policy's actions of the rewards and values one step on, all from
    the old values V. The learner's until_stable gets this function."""
    new = {}
    for s in T:
        v = 0
        for a in T[s]:
            q = 0
            for p, nxt, r in T[s][a]:
                q = q + p * (r + gamma * V[nxt])
            v = v + pi[s][a] * q
        new[s] = v
    return new


def evaluate(T, pi, gamma, sweeps) -> dict:
    """The course's solution of exercise 1: the values after `sweeps` sweeps from V = 0."""
    V = {s: 0 for s in T}
    for _ in range(sweeps):
        V = sweep(T, pi, gamma, V)
    return V


def until_stable(T, pi, gamma, theta):
    """The course's solution of exercise 2: sweeps until the largest change of a state in a sweep is below theta; the
    values and the number of sweeps."""
    V = {s: 0 for s in T}
    sweeps = 0
    while True:
        new = sweep(T, pi, gamma, V)
        sweeps = sweeps + 1
        change = max(abs(new[s] - V[s]) for s in T)
        V = new
        if change < theta:
            return V, sweeps


def history(T, pi, gamma, sweeps) -> list[dict]:
    """The values after every sweep, from sweep 0 (all 0) up to `sweeps`."""
    out = [{s: 0 for s in T}]
    for _ in range(sweeps):
        out.append(sweep(T, pi, gamma, out[-1]))
    return out


PLAY_SWEEPS = 300   # the play block goes from sweep 0 to here
PLAY = history(T_ROOM, RANDOM, GAMMA, PLAY_SWEEPS)
LOOKS = [0, 1, 2, 3, 4, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300]   # the sweeps of the play block
END = evaluate(T_ROOM, RANDOM, GAMMA, END_SWEEPS)          # the end values: −24, −30, −32, −34, −36
STOPS = [0.5, 0.1, 0.01, 0.001]                              # the thetas of the picture of exercise 2


def corridor_sweeps(n: int) -> list[list[float]]:
    """The worked example: the values of the four cells of the corridor after each of the first n sweeps."""
    steps = history(T_CORRIDOR, LEFT_RIGHT, GAMMA, n)
    return [[steps[k][(0, c)] for c in range(4)] for k in range(1, n + 1)]


def up_values(gamma: float, sweeps: int) -> dict:
    """Exercise 3: the values of UP after `sweeps` sweeps."""
    return evaluate(T_ROOM, UP, gamma, sweeps)


def matrix_form(T, pi, gamma, sweeps: int):
    """The same policy evaluation with NumPy, as the lesson's last cell does it: P[i, j] is the probability to go from
    state i to state j in one step of the policy, R[i] the reward to expect there; a sweep is V = R + gamma P V."""
    states = list(T)
    at = {s: i for i, s in enumerate(states)}
    P = np.zeros((len(states), len(states)))
    R = np.zeros(len(states))
    for s in states:
        for a in T[s]:
            for p, nxt, r in T[s][a]:
                P[at[s], at[nxt]] += pi[s][a] * p
                R[at[s]] += pi[s][a] * p * r
    V = np.zeros(len(states))
    for _ in range(sweeps):
        V = R + gamma * P @ V
    return {s: float(V[at[s]]) for s in states}


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


def _room_panel(V: dict, title: str) -> dict:
    """ROOM with a value on every tile but the charger, which shows its value 0."""
    tiles = {s: float(V[s]) for s in T_ROOM if s != CHARGER}
    return {"rows": ROOM, "title": title, "role": "pred", "index": True, "values": tiles, "goal": "0"}


def _grid(panels: list, alt: str, lang: str, cell: int = 52) -> str:
    return draw.grid_world(panels, alt=alt, cell=cell, decimal="," if lang == "nl" else ".").text


_PLAY_TEXT = {
    "en": {"title": "ROOM after sweep {k}",
           "chart": "Two tiles, sweep by sweep", "x": "sweep", "y": "value",
           "next": "next to the charger", "corner": "corner",
           "alt": "ROOM, 5 by 5 tiles with the charger in the middle, after sweep {k} of policy evaluation of the random policy: next to the charger {a}, diagonal to it {b}, in the middle of an edge {c}, in a corner {d}.",
           "chart_alt": "The values of two tiles after every sweep from 0 to 300: next to the charger from 0 down to −24, solid, and in a corner down to −36, dashed, each line falling fast at first and then slower; a mark at sweep {k}.",
           "say0": "Sweep 0: every value is 0, before any step is counted.",
           "say": "Sweep {k}: the corner is at {d}, the tile next to the charger at {a}.",
           "note": "Gray: the charger, with the value 0. Blue: the value of each tile after this sweep; the lines show the values of two tiles after every sweep, solid for the tile next to the charger, dashed for the corner."},
    "nl": {"title": "ROOM na sweep {k}",
           "chart": "Twee tegels, sweep na sweep", "x": "sweep", "y": "waarde",
           "next": "naast het laadstation", "corner": "hoek",
           "alt": "ROOM, 5 bij 5 tegels met het laadstation in het midden, na sweep {k} van policy evaluation van de random policy: naast het laadstation {a}, er schuin naast {b}, midden op een rand {c}, in een hoek {d}.",
           "chart_alt": "De waarden van twee tegels na elke sweep van 0 tot 300: naast het laadstation van 0 omlaag naar −24, doorgetrokken, en in een hoek naar −36, gestreept, elke lijn eerst snel dalend en dan langzamer; een markering bij sweep {k}.",
           "say0": "Sweep 0: elke waarde is 0, nog voordat er een stap geteld is.",
           "say": "Sweep {k}: de hoek staat op {d}, de tegel naast het laadstation op {a}.",
           "note": "Grijs: het laadstation, met de waarde 0. Blauw: de waarde van elke tegel na deze sweep; de lijnen tonen de waarden van twee tegels na elke sweep, doorgetrokken voor de tegel naast het laadstation, gestreept voor de hoek."},
}

_LINES = [(NEXT, False), (CORNER, True)]   # solid and dashed


def _sweeps_panel(k: int, lang: str) -> chart.Panel:
    w = _PLAY_TEXT[lang]
    p = chart.Panel((0, PLAY_SWEEPS), (-40, 0), title=w["chart"], xlabel=w["x"], ylabel=w["y"],
                    xticks=[0, 100, 200, 300], yticks=[-40, -30, -20, -10, 0])
    xs = list(range(PLAY_SWEEPS + 1))
    p.line([k, k], [-40, 0], "axis", 1.2)
    for tile, dashed in _LINES:
        p.line(xs, [PLAY[j][tile] for j in xs], "pred", 2.2, dashed=dashed)
        p.points([k], [PLAY[k][tile]], "pred", r=3.8)
    p.text(205, -21.5, w["next"], "ink-2")
    p.text(250, -33.5, w["corner"], "ink-2")
    return p


def play_frame(k: int, lang: str) -> tuple:
    """The play block at sweep k: ROOM with its values, the chart of three tiles beside it, and a sentence."""
    lang = lang if lang in LANGS else "en"
    w = _PLAY_TEXT[lang]
    V = PLAY[k]
    vals = {key: number(V[tile], 2, lang) for key, tile in (("a", NEXT), ("b", DIAGONAL), ("c", EDGE), ("d", CORNER))}
    grid = _grid([_room_panel(V, w["title"].format(k=k))], w["alt"].format(k=k, **vals), lang)
    line = _svg([_sweeps_panel(k, lang)], w["chart_alt"].format(k=k), lang, width=330, height=250)
    body = (f'<div style="display:flex;flex-wrap:wrap;gap:0.5rem 1.5rem;align-items:flex-start">'
            f'<div style="flex:0 1 300px">{grid}</div><div style="flex:1 1 300px;max-width:330px">{line}</div></div>')
    say = w["say0"] if k == 0 else w["say"].format(k=k, **vals)
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


def _values(got, T) -> dict | None:
    """got as {state: float} for every state of T, or None."""
    if not isinstance(got, dict) or set(got) != set(T):
        return None
    try:
        out = {s: float(got[s]) for s in T}
    except (TypeError, ValueError):
        return None
    return out if all(math.isfinite(v) for v in out.values()) else None


EVAL_LOOKS = (3, 300)   # the sweeps of the picture of exercise 1

_EVAL_TEXT = {
    "en": {"title": "Your evaluate on ROOM",
           "head": "Your `evaluate` on ROOM with the random policy, after 3 sweeps and after 300.",
           "panel": "after {k} sweeps",
           "alt": "ROOM twice, with the values your evaluate gives after 3 and after 300 sweeps: in a corner {a} and {b}, next to the charger {c} and {d}.",
           "same": "Your values are the course's on every tile.",
           "differ": "After {k} sweeps your value of tile {s} is {mine}, and the course's {want}.",
           "none": "Your `evaluate` stops with an error, takes too long or gives no value for every tile here, so nothing can be drawn yet.",
           "note": "Gray: the charger, with the value 0. Blue: your values; each is minus the average number of steps from that tile to the charger, as far as the sweeps so far count them."},
    "nl": {"title": "Je evaluate op ROOM",
           "head": "Je `evaluate` op ROOM met de random policy, na 3 sweeps en na 300.",
           "panel": "na {k} sweeps",
           "alt": "Twee keer ROOM, met de waarden die je evaluate geeft na 3 en na 300 sweeps: in een hoek {a} en {b}, naast het laadstation {c} en {d}.",
           "same": "Je waarden zijn op elke tegel gelijk aan de waarden van de course.",
           "differ": "Na {k} sweeps is je waarde van tegel {s} {mine}, en de waarde van de course {want}.",
           "none": "Je `evaluate` stopt met een fout, duurt te lang of geeft hier niet voor elke tegel een waarde, dus er is nog geen afbeelding.",
           "note": "Grijs: het laadstation, met de waarde 0. Blauw: je waarden; elke waarde is min het gemiddelde aantal stappen van de tegel naar het laadstation, voor zover de sweeps tot nu toe de stappen tellen."},
}


def evaluate_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _EVAL_TEXT[lang]
    fn = _fn(seen)
    got = None
    if fn is not None:
        got = _call(lambda: [_values(fn(copy.deepcopy(T_ROOM), copy.deepcopy(RANDOM), GAMMA, k), T_ROOM) for k in EVAL_LOOKS])
        if not isinstance(got, list) or any(v is None for v in got):
            got = None
    if got is None:
        return yours.figure(w["title"], _note(w["head"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    alt = w["alt"].format(a=number(got[0][CORNER], 2, lang), b=number(got[1][CORNER], 2, lang),
                          c=number(got[0][NEXT], 2, lang), d=number(got[1][NEXT], 2, lang))
    grid = _grid([_room_panel(V, w["panel"].format(k=k)) for V, k in zip(got, EVAL_LOOKS)], alt, lang, cell=46)
    body = _note(w["head"]) + grid
    off = None
    for V, k in zip(got, EVAL_LOOKS):
        want = PLAY[k]
        s = next((s for s in T_ROOM if abs(V[s] - want[s]) > 1e-6), None)
        if s is not None:
            off = (k, s, V[s], want[s])
            break
    if off is None:
        body += _note(w["same"])
    else:
        k, s, mine_, want_ = off
        body += _note(w["differ"].format(k=k, s=f"({s[0]}, {s[1]})", mine=number(mine_, 2, lang), want=number(want_, 2, lang)), _LOSS_UI)
    return yours.figure(w["title"], body + _note(w["note"]), alt=w["title"])


_STABLE_TEXT = {
    "en": {"title": "Your until_stable on ROOM",
           "head": "Your `until_stable` on ROOM with the random policy, for four values of `theta`.",
           "chart": "The corner, sweep by sweep", "x": "sweep", "y": "value of the corner",
           "end": "end value −36",
           "alt": "The value of the corner of ROOM after every sweep, from 0 down toward −36, with a dot where your until_stable stops for each theta: {stops}.",
           "stop": "θ = {t}: sweep {n}, corner {v}",
           "line": "θ = {t}: your function stops after {n} {word}, with the corner at {v}, {gap} from its end value.",
           "one": "sweep", "many": "sweeps",
           "none": "Your `until_stable` stops with an error, takes too long or gives no values and number of sweeps here, so nothing can be drawn yet.",
           "differ": "With θ = {t} the course's function stops after {want} sweeps.",
           "note": "Gray: the line of the end value. Blue: the value of the corner after every sweep, and a dot where your function stops for each θ."},
    "nl": {"title": "Je until_stable op ROOM",
           "head": "Je `until_stable` op ROOM met de random policy, voor vier waarden van `theta`.",
           "chart": "De hoek, sweep na sweep", "x": "sweep", "y": "waarde van de hoek",
           "end": "eindwaarde −36",
           "alt": "De waarde van de hoek van ROOM na elke sweep, van 0 omlaag naar −36, met een stip waar je until_stable stopt voor elke theta: {stops}.",
           "stop": "θ = {t}: sweep {n}, hoek {v}",
           "line": "θ = {t}: je functie stopt na {n} {word}, met de hoek op {v}, {gap} van de eindwaarde.",
           "one": "sweep", "many": "sweeps",
           "none": "Je `until_stable` stopt met een fout, duurt te lang of geeft hier geen waarden en aantal sweeps, dus er is nog geen afbeelding.",
           "differ": "Met θ = {t} stopt de functie van de course na {want} sweeps.",
           "note": "Grijs: de lijn van de eindwaarde. Blauw: de waarde van de hoek na elke sweep, en een stip waar je functie stopt voor elke θ."},
}


def _theta(t: float, lang: str) -> str:
    out = f"{t:g}"
    return out.replace(".", ",") if lang == "nl" else out


def _stable(got, T) -> tuple | None:
    """got as (values, sweeps) when it is two things: values for every state and a whole number of sweeps."""
    if not (isinstance(got, (tuple, list)) and len(got) == 2):
        return None
    V = _values(got[0], T)
    n = got[1]
    if V is None or isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        return None
    return V, int(n)


STABLE = {t: until_stable(T_ROOM, RANDOM, GAMMA, t) for t in STOPS}   # the course's answers, for the picture


def stable_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _STABLE_TEXT[lang]
    fn = _fn(seen)
    got = None
    if fn is not None:
        got = _call(lambda: [_stable(fn(copy.deepcopy(T_ROOM), copy.deepcopy(RANDOM), GAMMA, t), T_ROOM) for t in STOPS])
        if not isinstance(got, list) or any(v is None for v in got):
            got = None
    if got is None:
        return yours.figure(w["title"], _note(w["head"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    last = max(PLAY_SWEEPS, min(max(n for _, n in got), 600))
    curve = history(T_ROOM, RANDOM, GAMMA, last)
    p = chart.Panel((0, last), (-40, 0), title=w["chart"], xlabel=w["x"], ylabel=w["y"],
                    yticks=[-40, -30, -20, -10, 0])
    p.line([0, last], [END[CORNER], END[CORNER]], "ink-3", 1.2, dashed=True)
    p.text(last * 0.04, -38.4, w["end"], "ink-2", anchor="start")
    xs = list(range(last + 1))
    p.line(xs, [curve[j][CORNER] for j in xs], "pred", 2.2)
    shown = [(V, n, t) for (V, n), t in zip(got, STOPS) if n <= last]
    p.points([n for _, n, _ in shown], [V[CORNER] for V, _, _ in shown], "pred", r=4.2)
    named = {}   # one name for the thetas that stop at the same sweep
    for V, n, t in shown:
        named.setdefault(n, (V, []))[1].append(_theta(t, lang))
    for n, (V, ts) in named.items():   # each name up and to the right of its dot, where the falling line has gone on below it
        p.text(n + last * 0.02, min(V[CORNER] + 4, -3), "θ = " + ("; " if lang == "nl" else ", ").join(ts), "ink-2", anchor="start")
    stops = "; ".join(w["stop"].format(t=_theta(t, lang), n=n, v=number(V[CORNER], 2, lang)) for (V, n), t in zip(got, STOPS))
    body = _note(w["head"]) + _svg([p], w["alt"].format(stops=stops), lang, width=520, height=270)
    for (V, n), t in zip(got, STOPS):
        body += _note(w["line"].format(t=_theta(t, lang), n=n, word=w["one"] if n == 1 else w["many"], v=number(V[CORNER], 2, lang),
                                       gap=number(abs(V[CORNER] - END[CORNER]), 2, lang)))
    off = next(((t, STABLE[t][1]) for (V, n), t in zip(got, STOPS) if n != STABLE[t][1]), None)
    if off is not None:
        body += _note(w["differ"].format(t=_theta(off[0], lang), want=off[1]), _LOSS_UI)
    return yours.figure(w["title"], body + _note(w["note"]), alt=w["title"])


# --- how an answer is judged --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Judged:
    """What evaluate or until_stable gave back, as far as the check is concerned: the values rounded to six decimals,
    and for until_stable the number of sweeps; `off` says where the values first differ from the course's answer."""
    kind: str                     # "ok"; "shape": not what the function should give; "states": a state is missing
    world: str = ""
    values: tuple = ()            # ((state, value rounded to 6 decimals), ...)
    sweeps: int | None = None     # until_stable only
    want_sweeps: int | None = None
    off: tuple | None = None      # (state, its value, the course's)


def _rounded(V, T) -> tuple | None:
    try:
        return tuple((s, round(float(V[s]), 6) + 0.0) for s in sorted(T))
    except (TypeError, ValueError, KeyError):
        return None


def judge_evaluate(T, pi, gamma, sweeps, value):
    if value is None:
        return None
    if not isinstance(value, dict):
        return Judged("shape", T.world.name)
    if set(value) != set(T):
        return Judged("states", T.world.name)
    values = _rounded(value, T)
    if values is None:
        return Judged("shape", T.world.name)
    right = evaluate(T, pi, gamma, sweeps)
    off = next(((s, v, right[s]) for s, v in values if abs(v - right[s]) > 1e-6), None)
    return Judged("ok", T.world.name, values, off=off)


def judge_stable(T, pi, gamma, theta, value):
    if value is None:
        return None
    if isinstance(value, dict):
        return Judged("values-only", T.world.name)
    if isinstance(value, (tuple, list)) and len(value) == 2 and isinstance(value[1], dict) and not isinstance(value[0], dict):
        return Judged("swapped", T.world.name)
    if not (isinstance(value, (tuple, list)) and len(value) == 2 and isinstance(value[0], dict)):
        return Judged("shape", T.world.name)
    V, n = value
    if set(V) != set(T):
        return Judged("states", T.world.name)
    values = _rounded(V, T)
    if values is None or isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        return Judged("shape", T.world.name)
    right, want = until_stable(T, pi, gamma, theta)
    off = next(((s, v, right[s]) for s, v in values if abs(v - right[s]) > 1e-6), None)
    return Judged("ok", T.world.name, values, int(n), want, off)


def same_judged(a, b) -> bool:
    """Right when the values agree within a millionth, and for until_stable the numbers of sweeps are the same."""
    if not (isinstance(a, Judged) and isinstance(b, Judged)):
        return a == b
    if a.kind != b.kind or a.world != b.world:
        return False
    if a.kind != "ok":
        return True
    close = all(sa == sb and abs(va - vb) <= 1e-6 for (sa, va), (sb, vb) in zip(a.values, b.values))
    return close and a.sweeps == b.sweeps


def _plain(x: float) -> str:
    """A value the way the check shows it: up to four decimals, a real minus sign."""
    text = f"{x:.4f}".rstrip("0").rstrip(".")
    if text in ("-0", "-0."):
        text = "0"
    return text.replace("-", "−")


def show_values(judged) -> dict:
    """What evaluate gave back, in words: where it first differs from the course's answer."""
    if not isinstance(judged, Judged) or judged.kind == "shape":
        return {"en": "something other than a dictionary with a value for every state",
                "nl": "iets anders dan een dictionary met een waarde voor elke state"}
    if judged.kind == "states":
        return {"en": "a dictionary without a value for every state of the table",
                "nl": "een dictionary zonder waarde voor elke state van de tabel"}
    if judged.off is not None:
        s, got, want = judged.off
        return {"en": f"`V[{s}] = {_plain(got)}`, where it should be `{_plain(want)}`",
                "nl": f"`V[{s}] = {_plain(got)}`, waar het `{_plain(want)}` moet zijn"}
    return {"en": "the right value for every state", "nl": "de goede waarde voor elke state"}


def show_stable(judged) -> dict:
    """What until_stable gave back, in words."""
    if not isinstance(judged, Judged) or judged.kind == "shape":
        return {"en": "something other than two things, the values and the number of sweeps",
                "nl": "iets anders dan twee dingen, de waarden en het aantal sweeps"}
    if judged.kind == "values-only":
        return {"en": "only the values, without the number of sweeps",
                "nl": "alleen de waarden, zonder het aantal sweeps"}
    if judged.kind == "swapped":
        return {"en": "two things, but the values second",
                "nl": "twee dingen, maar de waarden als tweede"}
    if judged.kind == "states":
        return {"en": "values without a value for every state of the table",
                "nl": "waarden zonder waarde voor elke state van de tabel"}
    if judged.sweeps != judged.want_sweeps:
        return {"en": f"{judged.sweeps} {'sweep' if judged.sweeps == 1 else 'sweeps'}, where it should be {judged.want_sweeps}",
                "nl": f"{judged.sweeps} {'sweep' if judged.sweeps == 1 else 'sweeps'}, waar het er {judged.want_sweeps} moeten zijn"}
    if judged.off is not None:
        s, got, want = judged.off
        return {"en": f"the right number of sweeps, {judged.sweeps}, but `V[{s}] = {_plain(got)}`, where it should be `{_plain(want)}`",
                "nl": f"het goede aantal sweeps, {judged.sweeps}, maar `V[{s}] = {_plain(got)}`, waar het `{_plain(want)}` moet zijn"}
    return {"en": f"the right values after the right number of sweeps, {judged.sweeps}",
            "nl": f"de goede waarden na het goede aantal sweeps, {judged.sweeps}"}


@dataclass
class JudgedBox(FunctionBox):
    """A function in a code box whose answer is a dictionary of values: the check calls it on each case and judges what
    it gives back, so that a known mistake is known by what it does in every case."""
    judge: object = None

    def _run(self, fn, case):
        return self.judge(*case, super()._run(fn, case))


# --- the exercises -----------------------------------------------------------------------------------------------------

REQUIRED = ["evaluate", "stable", "forever"]


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


# exercise 1: evaluate
def _e(line: str = "v = v + pi[s][a] * q") -> str:
    """evaluate as the box holds it, with this line: step 3 of the recipe, an action's share of the new value."""
    return ("def evaluate(T, pi, gamma, sweeps):\n"
            "    V = {}\n"
            "    for s in T:\n"
            "        V[s] = 0\n"
            "    for k in range(sweeps):\n"
            "        new = {}\n"
            "        for s in T:\n"
            "            v = 0\n"
            "            for a in T[s]:\n"
            "                q = 0\n"
            "                for p, nxt, r in T[s][a]:\n"
            "                    q = q + p * (r + gamma * V[nxt])\n"
            f"                {line}\n"
            "            new[s] = v\n"
            "        V = new\n"
            "    return V\n")


_E_START = _e("v = v + ...")
_E = _e()
_E_NO_PI = _e("v = v + q")
_E_MAX = _e("v = max(v, q)")
_E_QUARTER = _e("v = v + 0.25 * q")
_E_LAST = _e("v = pi[s][a] * q")
_E_IN_PLACE = _E.replace("            new[s] = v\n", "            new[s] = v\n            V[s] = v\n")

EVALUATE_CASES = [
    (T_ROOM, RANDOM, 1.0, 2),
    (T_CORRIDOR, LEFT_RIGHT, 1.0, 3),
    (T_ROOM, RANDOM, 1.0, 40),
    (T_ROOM, MOSTLY_UP, 1.0, 5),
    (T_SMALL, RANDOM_SMALL, 0.9, 10),
    (T_ROOM, RANDOM, 0.5, 0),
]

# exercise 2: until_stable
def _u(change: str = "abs(new[s] - V[s])", stop: str = "change < theta") -> str:
    """until_stable as the box holds it, with these two parts: the change of a state, and when to stop."""
    return ("def until_stable(T, pi, gamma, theta):\n"
            "    V = {}\n"
            "    for s in T:\n"
            "        V[s] = 0\n"
            "    sweeps = 0\n"
            "    while True:\n"
            "        new = sweep(T, pi, gamma, V)\n"
            "        sweeps = sweeps + 1\n"
            "        change = 0\n"
            "        for s in T:\n"
            f"            change = max(change, {change})\n"
            "        V = new\n"
            f"        if {stop}:\n"
            "            return V, sweeps\n")


_U_START = _u("...", "...")
_U = _u()
_U_NO_ABS = _u("new[s] - V[s]")
_U_LAST = _U.replace("            change = max(change, abs(new[s] - V[s]))\n", "            change = abs(new[s] - V[s])\n")
_U_BIGGER = _u(stop="change > theta")
_U_VALUES = _U.replace("            return V, sweeps\n", "            return V\n")
_U_SWAPPED = _U.replace("            return V, sweeps\n", "            return sweeps, V\n")

STABLE_CASES = [
    (T_ROOM, RANDOM, 1.0, 0.01),
    (T_CORRIDOR, LEFT_RIGHT, 1.0, 0.001),
    (T_ROOM, RANDOM, 1.0, 0.1),
    (T_SMALL, RANDOM_SMALL, 1.0, 0.01),
    (T_ROOM, MOSTLY_UP, 0.9, 0.001),
    (T_PRIZE, LEFT_RIGHT, 0.9, 0.001),   # values that rise and fall, so a change must count both ways
]


EXERCISES = {
    "evaluate": JudgedBox(
        name="evaluate",
        picture=evaluate_picture,
        function="evaluate",
        start=_E_START,
        solution=_E,
        judge=judge_evaluate,
        compare=same_judged,
        show=show_values,
        visible=EVALUATE_CASES[:1],
        hidden=EVALUATE_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("no-pi", wrong=_E_NO_PI,
                    en="`{call}` gives {got}: you add the Q of every action in full, as if the robot took all four at once. In a state, each action counts with the probability that the policy takes it; where does `pi[s][a]` go in your line?",
                    nl="`{call}` geeft {got}: je telt de Q van elke action helemaal op, alsof de robot alle vier de actions tegelijk neemt. In een state telt elke action mee met de kans dat de policy de action neemt; waar komt `pi[s][a]` in je regel?"),
            Mistake("max", wrong=_E_MAX,
                    en="`{call}` gives {got}: `max` keeps the largest Q, as value iteration does, and with `v` starting at 0 no Q below 0 gets in. Policy evaluation follows the policy, and the random policy takes the worse actions too; how much does each action count?",
                    nl="`{call}` geeft {got}: `max` houdt de grootste Q, zoals value iteration doet, en omdat `v` bij 0 begint, komt geen Q onder 0 erin. Policy evaluation volgt de policy, en de random policy neemt ook de slechtere actions; hoeveel telt elke action mee?"),
            Mistake("quarter", wrong=_E_QUARTER,
                    en="`{call}` gives {got}: every action counts 1/4, the probability of the random policy on ROOM. The check also gives other policies, such as `LEFT_RIGHT` with two actions of 1/2 each; where can your line find the probability of each action?",
                    nl="`{call}` geeft {got}: elke action telt voor 1/4, de kans van de random policy op ROOM. De check geeft ook andere policies, zoals `LEFT_RIGHT` met twee actions van elk 1/2; waar kan je regel de kans van elke action vinden?"),
            Mistake("last-only", wrong=_E_LAST,
                    en="`{call}` gives {got}: `v = ...` throws away what `v` held, so only the last action of a state counts. Can you add the share of each action to `v`?",
                    nl="`{call}` geeft {got}: `v = ...` gooit weg wat er in `v` stond, dus telt alleen de laatste action van een state mee. Kun je het deel van elke action bij `v` optellen?"),
            Mistake("in-place", wrong=_E_IN_PLACE,
                    en="`{call}` gives {got}: your loop writes each new value into `V` during the sweep, so the states after it in the same sweep already use the new value. That in-place version reaches the same end values, often in fewer sweeps, but after a given number of sweeps its values are different ones. Step 3 keeps the new values apart in `new` until the sweep is done; where does your code change `V`?",
                    nl="`{call}` geeft {got}: je loop schrijft elke nieuwe waarde tijdens de sweep in `V`, dus gebruiken de states erna in dezelfde sweep de nieuwe waarde al. Zo'n in-place versie komt uit op dezelfde eindwaarden, vaak in minder sweeps, maar na een bepaald aantal sweeps zijn de waarden andere. Stap 3 houdt de nieuwe waarden apart in `new` tot de sweep klaar is; waar verandert je code `V`?"),
            Mistake("pi-index", error=lambda e: isinstance(e, KeyError) and isinstance(e.args[0] if e.args else None, str),
                    en="With `{call}`, your function stops with an error: it looks up an action where `pi` has states. `pi` holds a dictionary for every state, and `pi[s]` one for every action; which two keys does the probability of `a` in `s` need?",
                    nl="Met `{call}` stopt je functie met een fout: je functie zoekt een action op waar `pi` states heeft. `pi` bevat een dictionary voor elke state, en `pi[s]` een voor elke action; welke twee keys heeft de kans van `a` in `s` nodig?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the share of action `a` in the new value of `s`: its Q, counted with the probability that the policy takes it.",
                      "nl": "Vervang de `...` door het deel van action `a` in de nieuwe waarde van `s`: de Q van de action, meegeteld met de kans dat de policy de action neemt."},
            "arguments": {"en": "The check calls `{call}`, with a table, a policy, a discount factor and a number of sweeps, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(T, pi, gamma, sweeps)`?",
                          "nl": "De check roept `{call}` aan, met een tabel, een policy, een discount factor en een aantal sweeps, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(T, pi, gamma, sweeps)` houden?"},
            "slow": {"en": "Your function is still not done after 2 seconds. Does every loop go over a table, a dictionary or `range(sweeps)`, with `for`?",
                     "nl": "Je functie is na 2 seconden nog niet klaar. Gaat elke loop over een tabel, een dictionary of `range(sweeps)`, met `for`?"},
            "wrong": {"en": "`{call}` gives {got}. Is the new value of a state the Q of each action, counted with its probability `pi[s][a]`, added up over the actions?",
                      "nl": "`{call}` geeft {got}. Is de nieuwe waarde van een state de Q van elke action, meegeteld met de kans `pi[s][a]`, opgeteld over de actions?"},
            "almost": {"en": "Right for ROOM after 2 sweeps. But `{call}` gives {got}. Does your function work for every table, every policy, every discount factor and every number of sweeps?",
                       "nl": "Goed voor ROOM na 2 sweeps. Maar `{call}` geeft {got}. Werkt je functie voor elke tabel, elke policy, elke discount factor en elk aantal sweeps?"},
            "no_return": {"en": "With `{call}`, your function gives back `None`, and it should give the values. Does it still end with `return V`?",
                          "nl": "Met `{call}` geeft je functie `None` terug, en het moeten de waarden zijn. Eindigt je functie nog met `return V`?"},
            "correct": {"en": "Right in all {n} cases: after 2 sweeps, the tile next to the charger has the value −1.75, every other tile −2. In the picture your function goes on to 300 sweeps; how far is the corner then from the 4 steps of the shortest way?",
                        "nl": "Goed in alle {n} gevallen: na 2 sweeps heeft de tegel naast het laadstation de waarde −1,75, elke andere tegel −2. In de afbeelding gaat je functie door tot 300 sweeps; hoe ver zit de hoek dan van de 4 stappen van de kortste weg?"},
        },
    ),
    "stable": JudgedBox(
        name="stable",
        picture=stable_picture,
        function="until_stable",
        names={"sweep": sweep},
        start=_U_START,
        solution=_U,
        judge=judge_stable,
        compare=same_judged,
        show=show_stable,
        visible=STABLE_CASES[:1],
        hidden=STABLE_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("first-sweep", wrong=_U_NO_ABS,
                    en="`{call}` gives {got}: your function stops after the first sweep. Every value goes down, so a change counted as `new[s] - V[s]` is below 0, and `change` starts at 0. Does `change` hold how far a value moved, up or down, and does the loop stop only once that is smaller than `theta`?",
                    nl="`{call}` geeft {got}: je functie stopt na de eerste sweep. Elke waarde gaat omlaag, dus een verandering geteld als `new[s] - V[s]` ligt onder 0, en `change` begint bij 0. Houdt `change` bij hoe ver een waarde verschoof, omhoog of omlaag, en stopt de loop pas als dat kleiner is dan `theta`?"),
            Mistake("wrong-way", wrong=_U_BIGGER,
                    en="`{call}` gives {got}: your loop stops while the values still move a lot, here after the first sweep, where every value moved by 1. The change gets smaller with every sweep, and the loop should stop once it is small; which way should your comparison with `theta` point?",
                    nl="`{call}` geeft {got}: je loop stopt terwijl de waarden nog flink verschuiven, hier na de eerste sweep, waarin elke waarde 1 verschoof. De verandering wordt elke sweep kleiner, en de loop moet stoppen zodra de verandering klein is; welke kant moet je vergelijking met `theta` op wijzen?"),
            Mistake("values-only", wrong=_U_VALUES,
                    en="`{call}` gives {got}. `until_stable` gives back two things, the values and the number of sweeps it did; does your `return` give both, as `V, sweeps`?",
                    nl="`{call}` geeft {got}. `until_stable` geeft twee dingen terug, de waarden en het aantal sweeps; geeft je `return` allebei, als `V, sweeps`?"),
            Mistake("swapped", wrong=_U_SWAPPED,
                    en="`{call}` gives {got}. The check and the picture take the values first and the number of sweeps second; can you give back `V, sweeps`, in that order?",
                    nl="`{call}` geeft {got}. De check en de afbeelding nemen eerst de waarden en dan het aantal sweeps; kun je `V, sweeps` teruggeven, in deze volgorde?"),
            Mistake("last-change", wrong=_U_LAST,
                    en="`{call}` gives {got}: `change` holds only the change of the last state of the table, here the charger, whose value never changes. The stop needs the largest change of all states; what does `max` do with the old `change` and the new one?",
                    nl="`{call}` geeft {got}: `change` bevat alleen de verandering van de laatste state van de tabel, hier het laadstation, waarvan de waarde nooit verandert. De stop heeft de grootste verandering van alle states nodig; wat doet `max` met de oude `change` en de nieuwe?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: how far the value of `s` moved in this sweep, and when the loop may stop.",
                      "nl": "Vervang de twee `...`: hoe ver de waarde van `s` verschoof in deze sweep, en wanneer de loop mag stoppen."},
            "arguments": {"en": "The check calls `{call}`, with a table, a policy, a discount factor and `theta`, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(T, pi, gamma, theta)`?",
                          "nl": "De check roept `{call}` aan, met een tabel, een policy, een discount factor en `theta`, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(T, pi, gamma, theta)` houden?"},
            "slow": {"en": "Your function is still not done after 2 seconds: the loop never gets to `return`. The largest change of a sweep gets smaller and smaller; does your loop stop once it is smaller than `theta`?",
                     "nl": "Je functie is na 2 seconden nog niet klaar: de loop komt nooit bij `return`. De grootste verandering van een sweep wordt steeds kleiner; stopt je loop zodra de verandering kleiner is dan `theta`?"},
            "wrong": {"en": "`{call}` gives {got}. Is `change` the largest distance between a new and an old value, and does the loop stop once it is below `theta`?",
                      "nl": "`{call}` geeft {got}. Is `change` de grootste afstand tussen een nieuwe en een oude waarde, en stopt de loop zodra de afstand onder `theta` ligt?"},
            "almost": {"en": "Right for ROOM with θ = 0.01. But `{call}` gives {got}. Does your function work for every table, every policy, every discount factor and every `theta`?",
                       "nl": "Goed voor ROOM met θ = 0,01. Maar `{call}` geeft {got}. Werkt je functie voor elke tabel, elke policy, elke discount factor en elke `theta`?"},
            "no_return": {"en": "With `{call}`, your function gives back `None`, and it should give the values and the number of sweeps. Does it still end with `return V, sweeps`?",
                          "nl": "Met `{call}` geeft je functie `None` terug, en het moeten de waarden en het aantal sweeps zijn. Eindigt je functie nog met `return V, sweeps`?"},
            "correct": {"en": "Right in all {n} cases: with θ = 0.01, your function stops after 151 sweeps. In the picture it stops for four values of θ; how far from −36 is the corner each time?",
                        "nl": "Goed in alle {n} gevallen: met θ = 0,01 stopt je functie na 151 sweeps. In de afbeelding stopt je functie voor vier waarden van θ; hoe ver van −36 zit de hoek elke keer?"},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("evaluate", _E_START, "empty", None),
    ("evaluate", _E, "correct", None),
    ("evaluate", _e("v = v + ... * q"), "wrong", "dots-left"),
    ("evaluate", _E_NO_PI, "wrong", "no-pi"),
    ("evaluate", _E_MAX, "wrong", "max"),
    ("evaluate", _E_QUARTER, "wrong", "quarter"),
    ("evaluate", _E_LAST, "wrong", "last-only"),
    ("evaluate", _e("v = v + pi[a] * q"), "error", "pi-index"),
    ("evaluate", _E.replace("    return V\n", ""), "wrong", "no-return"),
    # other right answers: the probability after Q, and the share of each outcome straight into v
    ("evaluate", _e("v = v + q * pi[s][a]"), "correct", None),
    ("evaluate", _E.replace("                    q = q + p * (r + gamma * V[nxt])\n", "                    q = q + pi[s][a] * p * (r + gamma * V[nxt])\n")
                   .replace("v = v + pi[s][a] * q", "v = v + q"), "correct", None),
    # it updates V while it goes through the states, so later states already use new values
    ("evaluate", _E_IN_PLACE, "wrong", "in-place"),
    ("stable", _U_START, "empty", None),
    ("stable", _U, "correct", None),
    ("stable", _u(stop="..."), "wrong", "dots-left"),
    ("stable", _U_NO_ABS, "wrong", "first-sweep"),
    ("stable", _U_BIGGER, "wrong", "wrong-way"),
    ("stable", _U_LAST, "wrong", "last-change"),
    ("stable", _u(stop="change < 0"), "slow", None),
    ("stable", _U_VALUES, "wrong", "values-only"),
    ("stable", _U_SWAPPED, "wrong", "swapped"),
    # a change counted one way only: right where every value goes down, wrong once values rise too
    ("stable", _u("V[s] - new[s]"), "almost", None),
    ("stable", _u("abs(new[s]) - abs(V[s])"), "almost", None),
    # another right answer: the other order of the subtraction
    ("stable", _u("abs(V[s] - new[s])"), "correct", None),
    ("stable", _u(stop="change <= theta"), "correct", None),
]
