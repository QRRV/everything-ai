# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rl-policy-iteration: the robot of rl-policy-evaluation makes its policy better. In every
state it takes the action with the largest Q under the values of its policy (policy improvement), and evaluation and
improvement take turns until the policy no longer changes (policy iteration). ROOM is the room of the lesson before;
HALL is a hall with puddles, where a step that ends in a puddle costs −10. The course's own improvement and policy
iteration; improve and policy_iteration, the functions the learner writes, and their pictures; the play block over the
rounds on HALL.

Policy improvement and policy iteration as in Sutton and Barto (Reinforcement Learning: An Introduction, 2nd edition,
2018, sections 4.2 and 4.3), after Howard (Dynamic Programming and Markov Processes, 1960). The rooms, the robot, the
runs and the pictures are the course's own."""
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
THETA = 0.001    # the stop of every evaluation in the play block and the pictures
WET = -10        # what a step costs that ends in a puddle
TIE = 1e-9       # Q-values closer together than this count as equal


# --- the worlds, their tables, policies and values ---------------------------------------------------------------------

class World(list):
    """A grid of tiles as a list of strings, one per row: '.' a tile, S the start (a tile), W a puddle (a tile where a
    step ends at a cost of −10), G the charger. A check message calls it by the name the lesson gives it."""

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


class Named(dict):
    """A policy (pi[state][action] = probability) or values (V[state] = number) that a check message calls by name,
    such as RANDOM or V_ROOM; the learner's code gets a plain copy."""

    def __init__(self, entries, name: str):
        super().__init__(entries)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return {s: dict(v) if isinstance(v, dict) else v for s, v in self.items()}


ROOM = World([       # the room of rl-policy-evaluation: the charger in the middle tile
    ".....",
    ".....",
    "..G..",
    ".....",
    ".....",
], "ROOM")
HALL = World([       # the play block and exercise 2: the shortest way crosses a puddle, the dry way is longer
    "G.....",
    "WWWW..",
    "......",
    "..WWWW",
    "S.....",
], "HALL")
CORRIDOR = World(["G..."], "CORRIDOR")   # the corridor of rl-policy-evaluation
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
    """The table of a world: an action goes where it says, and into the edge the robot stays where it is; a step costs
    −1, and −10 when it ends in a puddle. On the charger the task is over: every action stays there, with reward 0."""
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
                T[state][a] = [(1.0, nxt, WET if world[nxt[0]][nxt[1]] == "W" else -1)]
    return Table(T, name or "T_" + world.name, world)


def uniform(T, name: str) -> Named:
    """The random policy of a table: every action of a state with the same probability."""
    return Named({s: {a: 1 / len(acts) for a in acts} for s, acts in T.items()}, name)


T_ROOM = table(ROOM)
T_HALL = table(HALL)
T_CORRIDOR = table(CORRIDOR, actions=("left", "right"))
T_SMALL = table(SMALL)
RANDOM = uniform(T_ROOM, "RANDOM")
RANDOM_HALL = uniform(T_HALL, "RANDOM_HALL")

CHARGER = find(ROOM, "G")
CORNER = (0, 0)
NEXT = (1, 2)
START = find(HALL, "S")
GOAL = find(HALL, "G")


# --- the course's policy evaluation, improvement and policy iteration -------------------------------------------------

def sweep(T, pi, gamma, V) -> dict:
    """One sweep of policy evaluation, as in rl-policy-evaluation, from the old values V."""
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


def until_stable(T, pi, gamma, theta):
    """Policy evaluation as the learner wrote it in rl-policy-evaluation: sweeps until the largest change of a state in
    a sweep is below theta; the values and the number of sweeps. The learner's policy_iteration gets this function."""
    V = {s: 0 for s in T}
    sweeps = 0
    while True:
        new = sweep(T, pi, gamma, V)
        sweeps = sweeps + 1
        change = max(abs(new[s] - V[s]) for s in T)
        V = new
        if change < theta:
            return V, sweeps


def q_values(T, V, gamma, s) -> list:
    """Q(s, a) for every action of s, in the order of T[s]."""
    return [sum(p * (r + gamma * V[nxt]) for p, nxt, r in T[s][a]) for a in T[s]]


def improve(T, V, gamma) -> dict:
    """The course's policy improvement: in every state probability 1 for the action with the largest Q under V. Of
    actions whose Q-values are within TIE of the largest, it takes the first, so that two Q-values that are equal on
    paper and differ in the last digits of the computer always give the same action. The learner's policy_iteration
    gets this function."""
    pi = {}
    for s in T:
        actions = list(T[s])
        qs = q_values(T, V, gamma, s)
        best = next(a for a, q in zip(actions, qs) if q >= max(qs) - TIE)
        pi[s] = {a: 1.0 if a == best else 0.0 for a in actions}
    return pi


def policy_iteration(T, gamma, theta):
    """The course's solution of exercise 2: from the random policy, evaluate and improve until the policy no longer
    changes; the policy, its values and the number of rounds (evaluations)."""
    pi = {s: {a: 1 / len(T[s]) for a in T[s]} for s in T}
    rounds = 0
    while True:
        V, _ = until_stable(T, pi, gamma, theta)
        rounds = rounds + 1
        new = improve(T, V, gamma)
        if new == pi:
            return pi, V, rounds
        pi = new


def rounds_of(T, gamma, theta) -> list[dict]:
    """Every round of policy iteration from the random policy: the policy it evaluates, its values, the sweeps that
    took, and how many tiles (charger aside) the improvement at the end of the round gives a new action."""
    pi = {s: {a: 1 / len(T[s]) for a in T[s]} for s in T}
    out = []
    while True:
        V, n = until_stable(T, pi, gamma, theta)
        new = improve(T, V, gamma)
        changed = sum(1 for s in T if new[s] != pi[s] and not _over(T, s))
        out.append({"pi": pi, "V": V, "sweeps": n, "changed": changed})
        if new == pi:
            return out
        pi = new


def value_iteration_sweeps(T, gamma, theta) -> tuple[dict, int]:
    """Value iteration as in rl-value-iteration, for the comparison in the text: the values and the number of sweeps."""
    V = {s: 0 for s in T}
    sweeps = 0
    while True:
        new = {s: max(q_values(T, V, gamma, s)) for s in T}
        sweeps = sweeps + 1
        change = max(abs(new[s] - V[s]) for s in T)
        V = new
        if change < theta:
            return V, sweeps


def _over(T, s) -> bool:
    """True for a state where the task is over: every action stays there with reward 0."""
    return all(nxt == s and r == 0 for a in T[s] for _, nxt, r in T[s][a])


def action_of(pi, s):
    """The action a deterministic policy takes in s."""
    return max(pi[s], key=pi[s].get)


def walk(T, pi, start, limit: int = 40) -> list:
    """The tiles the robot passes when it follows a deterministic policy from start, up to the charger or `limit` steps."""
    out = [start]
    s = start
    for _ in range(limit):
        if _over(T, s):
            break
        s = max(T[s][action_of(pi, s)], key=lambda o: o[0])[1]
        out.append(s)
    return out


V_ROOM = Named(until_stable(T_ROOM, RANDOM, GAMMA, THETA)[0], "V_ROOM")              # the random robot's values
V_HALL = Named(until_stable(T_HALL, RANDOM_HALL, GAMMA, THETA)[0], "V_HALL")
V_CORRIDOR = Named(until_stable(T_CORRIDOR, uniform(T_CORRIDOR, ""), GAMMA, THETA)[0], "V_CORRIDOR")
V_SMALL = Named(until_stable(T_SMALL, uniform(T_SMALL, ""), GAMMA, THETA)[0], "V_SMALL")
BETTER = improve(T_ROOM, V_ROOM, GAMMA)                                                 # one improvement on ROOM
V_BETTER = until_stable(T_ROOM, BETTER, GAMMA, THETA)
ROUNDS = rounds_of(T_HALL, GAMMA, THETA)                                                # the play block
FINAL, V_FINAL, N_ROUNDS = policy_iteration(T_HALL, GAMMA, THETA)
PLAY_ROUNDS = list(range(1, len(ROUNDS) + 1))                                          # the stops of the play block
VI_HALL = value_iteration_sweeps(T_HALL, GAMMA, THETA)


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


def _arrows(T, pi) -> dict:
    """The arrows of a policy: for every state but the charger, the actions it takes with a probability above 0."""
    return {s: [a for a in T[s] if pi[s][a] > 1e-9] for s in T if not _over(T, s)}


def _panel(world, T, V: dict, title: str, pi=None, path=None, digits: int = 2) -> dict:
    p = {"rows": world, "title": title, "role": "pred", "index": True, "goal": "0", "digits": digits,
         "values": {s: float(V[s]) for s in T if not _over(T, s)}}
    if pi is not None:
        p["arrows"] = _arrows(T, pi)
    if path:
        p["path"] = path
    return p


def _grid(panels: list, alt: str, lang: str, cell: int = 52) -> str:
    return draw.grid_world(panels, alt=alt, cell=cell, decimal="," if lang == "nl" else ".").text


def _wet(path) -> int:
    """How many steps of a walk end in a puddle."""
    return sum(1 for r, c in path[1:] if HALL[r][c] == "W")


_PLAY_TEXT = {
    "en": {"title": "HALL in round {k}",
           "chart": "New actions per round", "x": "round", "y": "tiles",
           "alt0": "HALL, 5 rows by 6 columns, with the charger at the top left, a row of four puddles below it and another row of four puddles lower down at the right, in round 1: the random policy, without arrows, with values from {lo} to {hi}; S at the bottom left has {v}.",
           "alt": "HALL in round {k}: an arrow on every tile for the action of the policy, its values, and the way from S: {steps} steps, {wet} of them into a puddle, value {v}.",
           "chart_alt": "Bars for rounds 1 to {n}, the number of tiles that get a new action at the end of each round: {bars}; the bar of round {k} is filled.",
           "say1": "Round 1 evaluates the random policy: S is at {v}. The improvement at the end of the round gives all {c} tiles an action.",
           "say": "Round {k}: from S the robot takes {steps} steps, {through}, value {v}. The improvement at the end of the round gives {c} {tiles} a new action.",
           "last": "Round {k}: the improvement gives no tile a new action, so the policy is stable and policy iteration stops. From S the robot takes {steps} steps, {through}, value {v}.",
           "through": "through a puddle", "around": "around the puddles", "tile": "tile", "tiles": "tiles",
           "note": "Gray: the charger, with the value 0, and the puddles, where a step ends at a cost of −10. Blue: the value of each tile under the policy of this round, in whole numbers; an arrow shows the action the policy takes, the gray line its way from S. The gray bars count the tiles that get a new action at the end of each round; the filled bar is this round."},
    "nl": {"title": "HALL in ronde {k}",
           "chart": "Nieuwe actions per ronde", "x": "ronde", "y": "tegels",
           "alt0": "HALL, 5 rijen bij 6 kolommen, met het laadstation linksboven, een rij van vier plassen eronder en nog een rij van vier plassen lager aan de rechterkant, in ronde 1: de random policy, zonder pijlen, met waarden van {lo} tot {hi}; S linksonder heeft {v}.",
           "alt": "HALL in ronde {k}: een pijl op elke tegel voor de action van de policy, de waarden, en de weg vanaf S: {steps} stappen, waarvan {wet} in een plas, waarde {v}.",
           "chart_alt": "Staven voor ronde 1 tot {n}, het aantal tegels dat aan het eind van elke ronde een nieuwe action krijgt: {bars}; de staaf van ronde {k} is gevuld.",
           "say1": "Ronde 1 evalueert de random policy: S staat op {v}. De verbetering aan het eind van de ronde geeft alle {c} tegels een action.",
           "say": "Ronde {k}: vanaf S zet de robot {steps} stappen, {through}, waarde {v}. De verbetering aan het eind van de ronde geeft {c} {tiles} een nieuwe action.",
           "last": "Ronde {k}: de verbetering geeft geen enkele tegel een nieuwe action, dus is de policy stabiel en stopt policy iteration. Vanaf S zet de robot {steps} stappen, {through}, waarde {v}.",
           "through": "door een plas", "around": "om de plassen heen", "tile": "tegel", "tiles": "tegels",
           "note": "Grijs: het laadstation, met de waarde 0, en de plassen: een stap die in een plas eindigt kost −10. Blauw: de waarde van elke tegel onder de policy van deze ronde, in hele getallen; een pijl toont de action van de policy, de grijze lijn de weg vanaf S. De grijze staven tellen de tegels die aan het eind van elke ronde een nieuwe action krijgen; de gevulde staaf is deze ronde."},
}


def _bars_panel(k: int, lang: str) -> chart.Panel:
    w = _PLAY_TEXT[lang]
    n = len(ROUNDS)
    top = max(r["changed"] for r in ROUNDS)
    p = chart.Panel((0.4, n + 0.6), (0, top + 2), title=w["chart"], xlabel=w["x"], ylabel=w["y"],
                    xticks=list(range(1, n + 1)), yticks=[0, 10, 20, 30])
    for j, r in enumerate(ROUNDS, start=1):
        if r["changed"]:
            p.rect(j - 0.32, 0, j + 0.32, r["changed"], "ink-2", solid=(j == k))
        p.text(j, r["changed"] + 1.2, str(r["changed"]), "ink-2" if j != k else "ink")
    return p


def play_frame(k: int, lang: str) -> tuple:
    """The play block in round k (1 to the last): HALL with the policy that round evaluates and its values, the bars of
    changed tiles beside it, and a sentence."""
    lang = lang if lang in LANGS else "en"
    w = _PLAY_TEXT[lang]
    k = max(1, min(int(k), len(ROUNDS)))
    r = ROUNDS[k - 1]
    V, pi = r["V"], r["pi"]
    v = number(V[START], 2 if k == 1 else 0, lang)
    if k == 1:
        tiles = [V[s] for s in T_HALL if not _over(T_HALL, s)]
        alt = w["alt0"].format(lo=number(min(tiles), 0, lang), hi=number(max(tiles), 0, lang), v=v)
        panel = _panel(HALL, T_HALL, V, w["title"].format(k=k), digits=0)
        say = w["say1"].format(v=v, c=r["changed"])
    else:
        path = walk(T_HALL, pi, START)
        alt = w["alt"].format(k=k, steps=len(path) - 1, wet=_wet(path), v=v)
        panel = _panel(HALL, T_HALL, V, w["title"].format(k=k), pi=pi, path=path, digits=0)
        through = w["through"] if _wet(path) else w["around"]
        if k == len(ROUNDS):
            say = w["last"].format(k=k, steps=len(path) - 1, through=through, v=v)
        else:
            say = w["say"].format(k=k, steps=len(path) - 1, through=through, v=v, c=r["changed"],
                                  tiles=w["tile"] if r["changed"] == 1 else w["tiles"])
    grid = _grid([panel], alt, lang, cell=51)
    bars = ", ".join(str(x["changed"]) for x in ROUNDS)
    chart_ = _svg([_bars_panel(k, lang)], w["chart_alt"].format(n=len(ROUNDS), bars=bars, k=k), lang, width=330, height=250)
    body = (f'<div style="display:flex;flex-wrap:wrap;gap:0.5rem 1.5rem;align-items:flex-start">'
            f'<div style="flex:0 1 350px">{grid}</div><div style="flex:1 1 300px;max-width:330px">{chart_}</div></div>')
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


def _policy(got, T) -> dict | None:
    """got as {state: {action: probability}} for every state and action of T, adding up to 1 in every state, or None."""
    if not isinstance(got, dict) or set(got) != set(T):
        return None
    out = {}
    for s in T:
        row = got[s]
        if not isinstance(row, dict) or set(row) != set(T[s]):
            return None
        try:
            out[s] = {a: float(row[a]) for a in T[s]}
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(x) and x >= -1e-9 for x in out[s].values()) or abs(sum(out[s].values()) - 1) > 1e-6:
            return None
    return out


def _values(got, T) -> dict | None:
    """got as {state: float} for every state of T, or None."""
    if not isinstance(got, dict) or set(got) != set(T):
        return None
    try:
        out = {s: float(got[s]) for s in T}
    except (TypeError, ValueError):
        return None
    return out if all(math.isfinite(v) for v in out.values()) else None


def _reaches(T, pi) -> bool:
    """True when the robot reaches the charger from every tile, following the actions the policy can take: from every
    tile some way of the policy leads to the charger, so that no tile keeps the robot away from it for good."""
    good = {s for s in T if _over(T, s)}
    grew = True
    while grew:
        grew = False
        for s in T:
            if s not in good and any(pi[s][a] > 1e-9 and p > 0 and nxt in good for a in T[s] for p, nxt, _ in T[s][a]):
                good.add(s)
                grew = True
    return good == set(T)


def _first_off(T, V, gamma, pi):
    """The first state where the policy puts probability on an action whose Q is below the largest, or None."""
    for s in sorted(T):
        qs = dict(zip(T[s], q_values(T, V, gamma, s)))
        best = max(qs.values())
        for a in T[s]:
            if pi[s][a] > 1e-6 and qs[a] < best - 1e-6:
                return s, a, qs[a], best
    return None


_IMPROVE_TEXT = {
    "en": {"title": "Your improve on ROOM",
           "head": "Your `improve` on ROOM with the values of the random policy, and the values of the policy it gives.",
           "before": "the random policy", "after": "your new policy",
           "alt": "ROOM twice: left the values of the random policy, from {a} next to the charger to {b} in a corner; right the arrows of your new policy and its values, from {c} next to the charger to {d} in a corner.",
           "same": "Your policy takes a best action on every tile: from a corner the robot now needs {d} steps, where the random policy needed about {b} on average.",
           "differ": "On tile {s} your policy takes `{a}`, with a Q of {q}, where the best Q there is {best}.",
           "lost": "From some tile your policy never reaches the charger, so its values never settle and the right grid shows only its arrows.",
           "none": "Your `improve` stops with an error, takes too long or gives no policy for every tile here, with probabilities that add up to 1, so nothing can be drawn yet.",
           "note": "Gray: the charger, with the value 0. Blue: values; each is minus the average number of steps from that tile to the charger under the policy above it. An arrow shows an action your policy takes."},
    "nl": {"title": "Je improve op ROOM",
           "head": "Je `improve` op ROOM met de waarden van de random policy, en de waarden van de policy die je functie geeft.",
           "before": "de random policy", "after": "je nieuwe policy",
           "alt": "Twee keer ROOM: links de waarden van de random policy, van {a} naast het laadstation tot {b} in een hoek; rechts de pijlen van je nieuwe policy en de waarden ervan, van {c} naast het laadstation tot {d} in een hoek.",
           "same": "Je policy neemt op elke tegel een beste action: vanuit een hoek heeft de robot nu {d} stappen nodig, waar de random policy er gemiddeld ongeveer {b} nodig had.",
           "differ": "Op tegel {s} neemt je policy `{a}`, met een Q van {q}, waar de beste Q {best} is.",
           "lost": "Vanaf een van de tegels komt je policy nooit bij het laadstation, dus komen de waarden nooit tot rust en toont het rechter grid alleen de pijlen.",
           "none": "Je `improve` stopt met een fout, duurt te lang of geeft hier niet voor elke tegel een policy met kansen die samen 1 zijn, dus valt er nog niets te tekenen.",
           "note": "Grijs: het laadstation, met de waarde 0. Blauw: waarden; elke waarde is min het gemiddelde aantal stappen van de tegel naar het laadstation onder de policy erboven. Een pijl toont een action van je policy."},
}


def improve_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _IMPROVE_TEXT[lang]
    fn = _fn(seen)
    pi = None
    if fn is not None:
        got = _call(lambda: fn(copy.deepcopy(T_ROOM), copy.deepcopy(V_ROOM), GAMMA))
        pi = _policy(got, T_ROOM)
    if pi is None:
        return yours.figure(w["title"], _note(w["head"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    left = _panel(ROOM, T_ROOM, V_ROOM, w["before"])
    reaches = _reaches(T_ROOM, pi)
    V2 = until_stable(T_ROOM, pi, GAMMA, THETA)[0] if reaches else None
    if V2 is not None:
        whole = all(abs(v - round(v)) < 1e-9 for v in V2.values())   # the values of a best policy: whole steps
        right = _panel(ROOM, T_ROOM, V2, w["after"], pi=pi, digits=0 if whole else 2)
    else:
        right = {"rows": ROOM, "title": w["after"], "index": True, "goal": "0", "arrows": _arrows(T_ROOM, pi)}
    nums = {"a": number(V_ROOM[NEXT], 2, lang), "b": number(V_ROOM[CORNER], 2, lang),
            "c": number(V2[NEXT], 2, lang) if V2 else "?", "d": number(V2[CORNER], 2, lang) if V2 else "?"}
    grid = _grid([left, right], w["alt"].format(**nums), lang, cell=46)
    body = _note(w["head"]) + grid
    off = _first_off(T_ROOM, V_ROOM, GAMMA, pi)
    if off is not None:
        s, a, q, best = off
        body += _note(w["differ"].format(s=f"({s[0]}, {s[1]})", a=a, q=number(q, 2, lang), best=number(best, 2, lang)), _LOSS_UI)
    elif V2 is not None:
        body += _note(w["same"].format(b=number(-V_ROOM[CORNER], 0, lang), d=number(-V2[CORNER], 0, lang)))
    if V2 is None:
        body += _note(w["lost"], _LOSS_UI)
    return yours.figure(w["title"], body + _note(w["note"]), alt=w["title"])


_ITER_TEXT = {
    "en": {"title": "Your policy_iteration on HALL",
           "head": "Your `policy_iteration` on HALL, with θ = 0.001: the policy it gives back, its values, and the way from S.",
           "panel": "after {n} {word}",
           "alt": "HALL with the arrows of the policy your policy_iteration gives back, its values, and the way from S: {steps} steps, {wet} of them into a puddle, value {v}.",
           "alt_mixed": "HALL with the arrows of the policy your policy_iteration gives back, several in some tiles, and its values; S has {v}.",
           "line": "Your function stops after {n} {word}. From S the robot takes {steps} steps, {through}, value {v}.",
           "mixed": "Your function stops after {n} {word}, with a policy that still spreads its probability over several actions in some tiles, so it has no single way from S; S has the value {v}.",
           "one": "round", "many": "rounds", "through": "through a puddle", "around": "around the puddles",
           "differ": "The course's function stops after {want} rounds, with S at {v}.",
           "none": "Your `policy_iteration` stops with an error, takes too long or gives no policy, values and number of rounds here, so nothing can be drawn yet.",
           "note": "Gray: the charger, with the value 0, and the puddles, where a step ends at a cost of −10. Blue: the values of your policy, in whole numbers; an arrow shows its action, the gray line its way from S."},
    "nl": {"title": "Je policy_iteration op HALL",
           "head": "Je `policy_iteration` op HALL, met θ = 0,001: de policy die je functie teruggeeft, de waarden ervan, en de weg vanaf S.",
           "panel": "na {n} {word}",
           "alt": "HALL met de pijlen van de policy die je policy_iteration teruggeeft, de waarden ervan, en de weg vanaf S: {steps} stappen, waarvan {wet} in een plas, waarde {v}.",
           "alt_mixed": "HALL met de pijlen van de policy die je policy_iteration teruggeeft, op sommige tegels meer dan één, en de waarden ervan; S heeft {v}.",
           "line": "Je functie stopt na {n} {word}. Vanaf S zet de robot {steps} stappen, {through}, waarde {v}.",
           "mixed": "Je functie stopt na {n} {word}, met een policy die op sommige tegels de kans nog over meer actions verdeelt, dus is er geen vaste weg vanaf S; S heeft de waarde {v}.",
           "one": "ronde", "many": "rondes", "through": "door een plas", "around": "om de plassen heen",
           "differ": "De functie van de course stopt na {want} rondes, met S op {v}.",
           "none": "Je `policy_iteration` stopt met een fout, duurt te lang of geeft hier geen policy, waarden en aantal rondes, dus valt er nog niets te tekenen.",
           "note": "Grijs: het laadstation, met de waarde 0, en de plassen: een stap die in een plas eindigt kost −10. Blauw: de waarden van je policy, in hele getallen; een pijl toont de action, de grijze lijn de weg vanaf S."},
}


def _iteration(got, T) -> tuple | None:
    """got as (policy, values, rounds) when it is those three things for T, or None."""
    if not (isinstance(got, (tuple, list)) and len(got) == 3):
        return None
    pi, V, n = _policy(got[0], T), _values(got[1], T), got[2]
    if pi is None or V is None or isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        return None
    return pi, V, int(n)


def iteration_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _ITER_TEXT[lang]
    fn = _fn(seen)
    got = None
    if fn is not None:
        got = _iteration(_call(lambda: fn(copy.deepcopy(T_HALL), GAMMA, THETA)), T_HALL)
    if got is None:
        return yours.figure(w["title"], _note(w["head"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    pi, V, n = got
    deterministic = all(max(pi[s].values()) > 1 - 1e-9 for s in T_HALL)
    path = walk(T_HALL, pi, START) if deterministic else None
    word = w["one"] if n == 1 else w["many"]
    panel = _panel(HALL, T_HALL, V, w["panel"].format(n=n, word=word), pi=pi, path=path, digits=0)
    steps = len(path) - 1 if path else "?"
    wet = _wet(path) if path else "?"
    v = number(V[START], 0, lang)
    alt = w["alt"].format(steps=steps, wet=wet, v=v) if path else w["alt_mixed"].format(v=v)
    grid = _grid([panel], alt, lang, cell=58)
    through = w["through"] if path and _wet(path) else w["around"]
    body = _note(w["head"]) + grid
    if path:
        body += _note(w["line"].format(n=n, word=word, steps=steps, through=through, v=v))
    else:
        body += _note(w["mixed"].format(n=n, word=word, v=v))
    if n != N_ROUNDS or abs(V[START] - V_FINAL[START]) > 1e-6:
        body += _note(w["differ"].format(want=N_ROUNDS, v=number(V_FINAL[START], 0, lang)), _LOSS_UI)
    return yours.figure(w["title"], body + _note(w["note"]), alt=w["title"])


# --- how an answer is judged --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Judged:
    """What improve or policy_iteration gave back, as far as the check is concerned. `kind` is "ok" or what is wrong
    with its shape; `detail` says what the check found, the same for two answers that do the same."""
    kind: str
    world: str = ""
    detail: tuple = ()


def _r6(x: float) -> float:
    return round(float(x), 6) + 0.0


def judge_improve(T, V, gamma, value):
    if value is None:
        return None
    if not isinstance(value, dict):
        return Judged("shape", T.world.name)
    if set(value) != set(T):
        return Judged("states", T.world.name)
    for s in sorted(T):
        row = value[s]
        if not isinstance(row, dict) or set(row) != set(T[s]):
            return Judged("actions", T.world.name, (s,))
        try:
            probs = {a: float(row[a]) for a in T[s]}
        except (TypeError, ValueError):
            return Judged("actions", T.world.name, (s,))
        if abs(sum(probs.values()) - 1) > 1e-6 or any(not math.isfinite(x) or x < -1e-9 for x in probs.values()):
            return Judged("sum", T.world.name, (s, _r6(sum(probs.values()))))
    off = _first_off(T, V, gamma, {s: {a: float(value[s][a]) for a in T[s]} for s in T})
    if off is not None:
        s, a, q, best = off
        return Judged("off", T.world.name, (s, a, _r6(q), _r6(best)))
    return Judged("ok", T.world.name)


_WANT = {}   # the course's answer for each case of exercise 2, worked out once


def judge_iteration(T, gamma, theta, value):
    if value is None:
        return None
    if not (isinstance(value, (tuple, list)) and len(value) == 3):
        return Judged("shape", T.world.name)
    pi, V, n = value
    if not (isinstance(pi, dict) and isinstance(V, dict)) or isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        return Judged("shape", T.world.name)
    if set(pi) != set(T) or set(V) != set(T):
        return Judged("states", T.world.name)
    mine_pi, mine_V = _policy(pi, T), _values(V, T)
    if mine_pi is None or mine_V is None:
        return Judged("shape", T.world.name)
    key = (T.name, gamma, theta)
    if key not in _WANT:
        _WANT[key] = policy_iteration(T, gamma, theta)
    want_pi, want_V, want_n = _WANT[key]
    off_pi = next(((s, action_of(mine_pi, s), action_of(want_pi, s)) for s in sorted(T)
                   if any(abs(mine_pi[s][a] - want_pi[s][a]) > 1e-6 for a in T[s])), None)
    off_V = next(((s, _r6(mine_V[s]), _r6(want_V[s])) for s in sorted(T) if abs(mine_V[s] - want_V[s]) > 1e-6), None)
    return Judged("ok", T.world.name, (int(n), want_n, off_pi, off_V))


def same_judged(a, b) -> bool:
    if not (isinstance(a, Judged) and isinstance(b, Judged)):
        return a == b
    return a == b


def _plain(x: float) -> str:
    """A value the way the check shows it: up to four decimals, a real minus sign."""
    text = f"{x:.4f}".rstrip("0").rstrip(".")
    if text in ("-0", "-0."):
        text = "0"
    return text.replace("-", "−")


def show_policy(judged) -> dict:
    """What improve gave back, in words."""
    if not isinstance(judged, Judged) or judged.kind == "shape":
        return {"en": "something other than a dictionary with a policy for every state",
                "nl": "iets anders dan een dictionary met een policy voor elke state"}
    if judged.kind == "states":
        return {"en": "a policy without an entry for every state of the table",
                "nl": "een policy zonder entry voor elke state van de tabel"}
    if judged.kind == "actions":
        s = judged.detail[0]
        return {"en": f"`pi[{s}]` without a probability for every action of state `{s}`",
                "nl": f"`pi[{s}]` zonder kans voor elke action van state `{s}`"}
    if judged.kind == "sum":
        s, total = judged.detail
        return {"en": f"probabilities that add up to {_plain(total)} in state `{s}`",
                "nl": f"kansen die in state `{s}` samen {_plain(total)} zijn"}
    if judged.kind == "off":
        s, a, q, best = judged.detail
        return {"en": f"a policy that takes `{a}` in state `{s}`, with a Q of {_plain(q)}, where the best Q there is {_plain(best)}",
                "nl": f"een policy die in state `{s}` `{a}` neemt, met een Q van {_plain(q)}, waar de beste Q {_plain(best)} is"}
    return {"en": "a policy that takes a best action in every state", "nl": "een policy die in elke state een beste action neemt"}


def show_iteration(judged) -> dict:
    """What policy_iteration gave back, in words."""
    if not isinstance(judged, Judged) or judged.kind == "shape":
        return {"en": "something other than three things: a policy, its values and the number of rounds",
                "nl": "iets anders dan drie dingen: een policy, de waarden ervan en het aantal rondes"}
    if judged.kind == "states":
        return {"en": "a policy or values without an entry for every state of the table",
                "nl": "een policy of waarden zonder entry voor elke state van de tabel"}
    n, want, off_pi, off_V = judged.detail
    rounds = {"en": "round" if n == 1 else "rounds", "nl": "ronde" if n == 1 else "rondes"}
    if n != want:
        return {"en": f"{n} {rounds['en']}, where it should be {want}", "nl": f"{n} {rounds['nl']}, waar het er {want} moeten zijn"}
    if off_pi is not None:
        s, mine, right = off_pi
        return {"en": f"the right number of rounds, {n}, but a policy that takes `{mine}` in state `{s}`, where it should take `{right}`",
                "nl": f"het goede aantal rondes, {n}, en toch een policy die in state `{s}` `{mine}` neemt, waar het `{right}` moet zijn"}
    if off_V is not None:
        s, mine, right = off_V
        return {"en": f"the right policy after {n} {rounds['en']}, but `V[{s}] = {_plain(mine)}`, where it should be `{_plain(right)}`",
                "nl": f"de goede policy na {n} {rounds['nl']}, maar `V[{s}] = {_plain(mine)}`, waar het `{_plain(right)}` moet zijn"}
    return {"en": f"the right policy and values after the right number of rounds, {n}",
            "nl": f"de goede policy en waarden na het goede aantal rondes, {n}"}


@dataclass
class JudgedBox(FunctionBox):
    """A function in a code box whose answer is a policy: the check calls it on each case and judges what it gives
    back, so that a known mistake is known by what it does in every case."""
    judge: object = None

    def _run(self, fn, case):
        return self.judge(*case, super()._run(fn, case))


# --- the exercises -----------------------------------------------------------------------------------------------------

REQUIRED = ["improve", "iteration", "ties"]


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


# exercise 1: improve
def _i(line: str = "best = actions[np.argmax(qs)]") -> str:
    """improve as the box holds it, with this line: the action with the largest Q."""
    return ("def improve(T, V, gamma):\n"
            "    pi = {}\n"
            "    for s in T:\n"
            "        actions = list(T[s])\n"
            "        qs = []\n"
            "        for a in actions:\n"
            "            q = 0\n"
            "            for p, nxt, r in T[s][a]:\n"
            "                q = q + p * (r + gamma * V[nxt])\n"
            "            qs.append(q)\n"
            f"        {line}\n"
            "        pi[s] = {}\n"
            "        for a in actions:\n"
            "            if a == best:\n"
            "                pi[s][a] = 1.0\n"
            "            else:\n"
            "                pi[s][a] = 0.0\n"
            "    return pi\n")


_I_START = _i("best = ...")
_I = _i()
_I_MIN = _i("best = actions[np.argmin(qs)]")
_I_NUMBER = _i("best = max(qs)")
_I_FIRST = _i("best = actions[0]")

IMPROVE_CASES = [
    (T_ROOM, V_ROOM, 1.0),
    (T_HALL, V_HALL, 1.0),
    (T_CORRIDOR, V_CORRIDOR, 1.0),
    (T_SMALL, V_SMALL, 0.9),
    (T_ROOM, Named(V_BETTER[0], "V_BETTER"), 1.0),
    (T_HALL, Named(ROUNDS[1]["V"], "V_ROUND_2"), 1.0),
]


# exercise 2: policy_iteration
def _p(improve_line: str = "new = improve(T, V, gamma)", stop: str = "new == pi") -> str:
    """policy_iteration as the box holds it, with these two parts: the new policy, and when to stop."""
    return ("def policy_iteration(T, gamma, theta):\n"
            "    pi = {}\n"
            "    for s in T:\n"
            "        pi[s] = {}\n"
            "        for a in T[s]:\n"
            "            pi[s][a] = 1 / len(T[s])\n"
            "    rounds = 0\n"
            "    while True:\n"
            "        V, sweeps = until_stable(T, pi, gamma, theta)\n"
            "        rounds = rounds + 1\n"
            f"        {improve_line}\n"
            f"        if {stop}:\n"
            "            return pi, V, rounds\n"
            "        pi = new\n")


_P_START = _p("new = ...", "...")
_P = _p()
_P_WRONG_WAY = _p(stop="new != pi")
_P_PI_FOR_V = _p("new = improve(T, pi, gamma)")
_P_ONE_STATE = _p(stop="new[s] == pi[s]")

ITERATION_CASES = [
    (T_ROOM, 1.0, 0.01),
    (T_HALL, 1.0, 0.001),
    (T_CORRIDOR, 1.0, 0.001),
    (T_SMALL, 1.0, 0.01),
    (T_HALL, 0.95, 0.001),   # a discount that changes the best policy: 10 rounds
    (T_ROOM, 0.5, 0.001),
]


EXERCISES = {
    "improve": JudgedBox(
        name="improve",
        picture=improve_picture,
        function="improve",
        names={"np": np},
        start=_I_START,
        solution=_I,
        judge=judge_improve,
        compare=same_judged,
        show=show_policy,
        visible=IMPROVE_CASES[:1],
        hidden=IMPROVE_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("argmin", wrong=_I_MIN,
                    en="`{call}` gives {got}: `np.argmin` finds the place of the smallest Q, the worst action. Which NumPy function finds the place of the largest?",
                    nl="`{call}` geeft {got}: `np.argmin` vindt de plek van de kleinste Q, de slechtste action. Welke NumPy-functie vindt de plek van de grootste?"),
            Mistake("number", wrong=_I_NUMBER,
                    en="`{call}` gives {got}: `best` holds a number, and no action is equal to a number, so every action gets 0. The largest Q and its place are numbers; `best` must be the action at the place of the largest Q in `actions`: which one is that?",
                    nl="`{call}` geeft {got}: in `best` staat een getal, en geen action is gelijk aan een getal, dus krijgt elke action 0. De grootste Q en de plek ervan zijn getallen; `best` moet de action op de plek van de grootste Q in `actions` zijn: welke is dat?"),
            Mistake("first-action", wrong=_I_FIRST,
                    en="`{call}` gives {got}: `actions[0]` is the first action of every state, `up`, whatever its Q. Which action has the largest Q in `qs`?",
                    nl="`{call}` geeft {got}: `actions[0]` is in elke state de eerste action, `up`, wat de Q ook is. Welke action heeft de grootste Q in `qs`?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the action of `s` that has the largest Q in `qs`.",
                      "nl": "Vervang de `...` door de action van `s` met de grootste Q in `qs`."},
            "arguments": {"en": "The check calls `{call}`, with a table, values and a discount factor, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(T, V, gamma)`?",
                          "nl": "De check roept `{call}` aan, met een tabel, waarden en een discount factor, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(T, V, gamma)` houden?"},
            "slow": {"en": "Your function is still not done after 2 seconds. Does every loop go over a table, a list or `actions`, with `for`?",
                     "nl": "Je functie is na 2 seconden nog niet klaar. Gaat elke loop over een tabel, een lijst of `actions`, met `for`?"},
            "wrong": {"en": "`{call}` gives {got}. Does `best` hold the action of `s` with the largest Q, so that it gets probability 1 and every other action 0?",
                      "nl": "`{call}` geeft {got}. Staat in `best` de action van `s` met de grootste Q, zodat deze action kans 1 krijgt en elke andere action 0?"},
            "almost": {"en": "Right for ROOM. But `{call}` gives {got}. Does your function work for every table, every set of values and every discount factor?",
                       "nl": "Goed voor ROOM. Maar `{call}` geeft {got}. Werkt je functie voor elke tabel, alle waarden en elke discount factor?"},
            "no_return": {"en": "With `{call}`, your function gives back `None`, and it should give the new policy. Does it still end with `return pi`?",
                          "nl": "Met `{call}` geeft je functie `None` terug, en het moet de nieuwe policy zijn. Eindigt je functie nog met `return pi`?"},
            "correct": {"en": "Right in all {n} cases: on ROOM, the corner's best actions are down and right, each with a Q of −34.97, and the new policy goes straight to the charger from every tile. In the picture, how many steps does it need from a corner now?",
                        "nl": "Goed in alle {n} gevallen: op ROOM zijn de beste actions in de hoek omlaag en rechts, elk met een Q van −34,97, en de nieuwe policy gaat vanaf elke tegel recht naar het laadstation. Hoeveel stappen heeft de policy in de afbeelding nu nodig vanuit een hoek?"},
        },
    ),
    "iteration": JudgedBox(
        name="iteration",
        picture=iteration_picture,
        function="policy_iteration",
        names={"until_stable": until_stable, "improve": improve},
        start=_P_START,
        solution=_P,
        judge=judge_iteration,
        compare=same_judged,
        show=show_iteration,
        visible=ITERATION_CASES[:1],
        hidden=ITERATION_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("wrong-way", wrong=_P_WRONG_WAY,
                    en="`{call}` gives {got}: your loop stops after the first round and gives back the random policy, since the improved policy is a different one. The loop should go on while the policy still changes, and stop once nothing changes; does your comparison of `new` and `pi` say that?",
                    nl="`{call}` geeft {got}: je loop stopt na de eerste ronde en geeft de random policy terug, omdat de verbeterde policy een andere is. De loop moet doorgaan zolang de policy nog verandert, en stoppen zodra er niets meer verandert; zegt je vergelijking van `new` en `pi` dat?"),
            Mistake("one-state", wrong=_P_ONE_STATE,
                    en="`{call}` gives {got}: after the first loop, `s` is still the last state of the table, so your stop looks at the policy in that one state, and the loop ends while other states still change. Which two policies should your comparison look at, as a whole?",
                    nl="`{call}` geeft {got}: na de eerste loop is `s` nog steeds de laatste state van de tabel, dus kijkt je stop naar de policy in slechts één state, en stopt de loop terwijl andere states nog veranderen. Welke twee policies moet je vergelijking als geheel bekijken?"),
            Mistake("pi-for-v", error=lambda e: isinstance(e, TypeError) and "dict" in str(e),
                    en="With `{call}`, your function stops with an error: `improve` needs values, a number for each state, to compute each Q with `V[nxt]`, and it gets a policy, a dictionary of probabilities for each state. Which of your names holds the values?",
                    nl="Met `{call}` stopt je functie met een fout: `improve` heeft waarden nodig, een getal voor elke state, om elke Q uit te rekenen met `V[nxt]`, en krijgt een policy, een dictionary met kansen voor elke state. Welke van je namen bevat de waarden?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: the improved policy, and when the loop may stop.",
                      "nl": "Vervang de twee `...`: de verbeterde policy, en wanneer de loop mag stoppen."},
            "arguments": {"en": "The check calls `{call}`, with a table, a discount factor and `theta`, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(T, gamma, theta)`?",
                          "nl": "De check roept `{call}` aan, met een tabel, een discount factor en `theta`, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(T, gamma, theta)` houden?"},
            "slow": {"en": "Your function is still not done after 2 seconds: the loop never gets to `return`. Once the improvement gives back the policy you already had, nothing changes any more. `is` asks whether two names hold the very same dictionary, and `improve` makes a new one every round: does your stop compare what `new` and `pi` hold?",
                     "nl": "Je functie is na 2 seconden nog niet klaar: de loop komt nooit bij `return`. Zodra de verbetering dezelfde policy teruggeeft als de policy waarmee de ronde begon, verandert er niets meer. `is` vraagt of twee namen precies dezelfde dictionary bevatten, en `improve` maakt elke ronde een nieuwe: vergelijkt je stop wat er in `new` en `pi` staat?"},
            "wrong": {"en": "`{call}` gives {got}. Is `new` the improved policy, from `improve` with the values and the discount factor of this round, and does the loop stop once it is the same as `pi`?",
                      "nl": "`{call}` geeft {got}. Is `new` de verbeterde policy, uit `improve` met de waarden en de discount factor van deze ronde, en stopt de loop zodra `new` gelijk is aan `pi`?"},
            "almost": {"en": "Right for ROOM. But `{call}` gives {got}. Does your function work for every table, every discount factor and every `theta`?",
                       "nl": "Goed voor ROOM. Maar `{call}` geeft {got}. Werkt je functie voor elke tabel, elke discount factor en elke `theta`?"},
            "no_return": {"en": "With `{call}`, your function gives back `None`, and it should give the policy, its values and the number of rounds. Does it still end with `return pi, V, rounds`?",
                          "nl": "Met `{call}` geeft je functie `None` terug, en het moeten de policy, de waarden en het aantal rondes zijn. Eindigt je functie nog met `return pi, V, rounds`?"},
            "correct": {"en": "Right in all {n} cases: on ROOM your function stops after 3 rounds, and on HALL after 9. In the picture, which way does the robot take from S on HALL?",
                        "nl": "Goed in alle {n} gevallen: op ROOM stopt je functie na 3 rondes, en op HALL na 9. Welke weg neemt de robot in de afbeelding vanaf S op HALL?"},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("improve", _I_START, "empty", None),
    ("improve", _I, "correct", None),
    ("improve", _i("best = actions[np.argmax(...)]"), "wrong", "dots-left"),
    ("improve", _I_MIN, "wrong", "argmin"),
    ("improve", _I_NUMBER, "wrong", "number"),
    ("improve", _i("best = np.argmax(qs)"), "wrong", "number"),
    ("improve", _I_FIRST, "wrong", "first-action"),
    ("improve", _I.replace("    return pi\n", ""), "wrong", "no-return"),
    # other right answers: the largest Q found by hand, and a policy that shares its probability over equal best actions
    ("improve", _i("best = actions[qs.index(max(qs))]"), "correct", None),
    ("improve", _I.replace("        best = actions[np.argmax(qs)]\n", "        top = [b for b, q in zip(actions, qs) if q > max(qs) - 1e-9]\n")
                  .replace("            if a == best:\n                pi[s][a] = 1.0\n            else:\n                pi[s][a] = 0.0\n",
                           "            if a in top:\n                pi[s][a] = 1 / len(top)\n            else:\n                pi[s][a] = 0.0\n"),
     "correct", None),
    ("iteration", _P_START, "empty", None),
    ("iteration", _P, "correct", None),
    ("iteration", _p(stop="..."), "wrong", "dots-left"),
    ("iteration", _P_WRONG_WAY, "wrong", "wrong-way"),
    ("iteration", _p(stop="True"), "wrong", "wrong-way"),
    ("iteration", _P_PI_FOR_V, "error", "pi-for-v"),
    ("iteration", _p(stop="new is pi"), "slow", None),
    ("iteration", _P.replace("            return pi, V, rounds\n", "            return pi, V\n"), "wrong", None),
    # another right answer: the comparison the other way round, and the new policy given back
    ("iteration", _p(stop="pi == new").replace("return pi, V, rounds", "return new, V, rounds"), "correct", None),
    # it looks only at the last state, the `s` left over from the first loop
    ("iteration", _P_ONE_STATE, "wrong", "one-state"),
]
