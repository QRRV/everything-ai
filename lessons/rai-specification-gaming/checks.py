# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-specification-gaming: the course's race track, with targets that pay points as the boat
race of AI Safety Gridworlds (Leike et al., 2017) does, after the game CoastRunners (Clark and Amodei, 2016); the
course's Q-learning and its recorded runs; the cleaning robot of Skalse et al. (2022); and points_once, the points of
a way when a target counts once, and hackable, the definition of Skalse et al., which the learner writes."""
import contextlib
import copy
import html
import io
import itertools
import re

import numpy as np

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, Timeout, kept, with_time_limit

# --- the track and the course's Q-learning -------------------------------------------------------------------------

ACTIONS = ["up", "down", "left", "right"]
MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
FINISH_REWARD = 50  # entering the finish F, as entering the goal in AI Safety Gridworlds; the episode ends
STEP_REWARD = -1    # every step, as in AI Safety Gridworlds
TARGET = 3          # entering a target T from another cell, as an arrow tile in the boat race of AI Safety Gridworlds
LIMIT = 100         # an episode stops after this many steps, as in AI Safety Gridworlds
ETA, EPSILON = 0.2, 0.1   # as in the lesson on Q-learning
GAMMA = 0.99        # the course's choice: the score of the race adds up all points, and with γ close to 1 so does the agent
EPISODES = 500
SEED = 0            # the course's recorded runs
VALUES = [0, 0.5, 1, 1.5, 2, 2.5, 3]   # the points of a target in the demo

# The course's track: start S, finish F, and three targets T. A target pays again as soon as the agent leaves it and
# comes back, as the targets of CoastRunners came back; every step lands where it aims.
TRACK = ["S..TT....F",
         "....T....."]


def cells(world, signs: str) -> list:
    """The cells of a world whose sign is one of `signs`, row by row."""
    return [(r, c) for r, row in enumerate(world) for c, sign in enumerate(row) if sign in signs]


START, FINISH = cells(TRACK, "S")[0], cells(TRACK, "F")[0]
TARGETS = cells(TRACK, "T")


def land(world, state, action):
    """One step from `state`; into the edge of the world you stay where you are."""
    r, c = state[0] + MOVES[action][0], state[1] + MOVES[action][1]
    if 0 <= r < len(world) and 0 <= c < len(world[0]):
        return (r, c)
    return state


def step_points(world, state, nxt, value=TARGET, taken=None) -> float:
    """The points of one step from `state` into `nxt`: −1, +50 into F, and `value` for a target entered from another
    cell; with a set `taken`, only for a target the episode has not taken yet."""
    sign = world[nxt[0]][nxt[1]]
    total = STEP_REWARD + (FINISH_REWARD if sign == "F" else 0)
    if sign == "T" and nxt != state and (taken is None or nxt not in taken):
        total += value
    return total


def points(way, world, value=TARGET) -> float:
    """The points of a way as the track pays them: a target every time it is entered from another cell."""
    return sum(step_points(world, a, b, value) for a, b in zip(way, way[1:]))


def greedy(Q, state) -> str:
    """The action with the largest Q; the first of ACTIONS on a tie."""
    row = Q[state]
    return max(ACTIONS, key=row.__getitem__)


def train(world, value=TARGET, once=False, seed: int = SEED, episodes: int = EPISODES) -> dict:
    """Q-learning on the track, with ε-greedy exploration, as in the lesson on Q-learning; a target pays `value`. With
    `once`, a target pays once in an episode, and the state is the cell with the targets taken so far. The way and the
    end of every episode, and the Q-table after the last."""
    rng = np.random.default_rng(seed)
    Q = {}
    run = {"ways": [], "ends": []}
    for _ in range(episodes):
        state, taken, way, end = START, frozenset(), [START], "limit"
        key = (state, taken) if once else state
        for _ in range(LIMIT):
            Q.setdefault(key, {a: 0.0 for a in ACTIONS})
            action = ACTIONS[rng.integers(0, 4)] if rng.random() < EPSILON else greedy(Q, key)
            nxt = land(world, state, action)
            r = step_points(world, state, nxt, value, taken if once else None)
            if once and world[nxt[0]][nxt[1]] == "T":
                taken = taken | {nxt}
            nkey = (nxt, taken) if once else nxt
            Q.setdefault(nkey, {a: 0.0 for a in ACTIONS})
            done = world[nxt[0]][nxt[1]] == "F"
            target = r if done else r + GAMMA * max(Q[nkey].values())
            Q[key][action] += ETA * (target - Q[key][action])
            state, key = nxt, nkey
            way.append(nxt)
            if done:
                end = "finish"
                break
        run["ways"].append(way)
        run["ends"].append(end)
    run["Q"] = Q
    return run


def greedy_way(world, Q, once=False, limit: int = LIMIT) -> list:
    """Where the greedy policy of Q goes from the start, until the finish or `limit` steps."""
    state, taken, way = START, frozenset(), [START]
    for _ in range(limit):
        key = (state, taken) if once else state
        if key not in Q:
            break
        state = land(world, state, greedy(Q, key))
        if once and world[state[0]][state[1]] == "T":
            taken = taken | {state}
        way.append(state)
        if world[state[0]][state[1]] == "F":
            break
    return way


def finished(world, way) -> bool:
    return world[way[-1][0]][way[-1][1]] == "F"


def points_once(way, world):
    """The course's own Exercise 1: the points of a way when a target counts once."""
    total = 0
    taken = set()
    for cell in way[1:]:
        total = total - 1
        if world[cell[0]][cell[1]] == "F":
            total = total + 50
        if world[cell[0]][cell[1]] == "T" and cell not in taken:
            total = total + 3
            taken.add(cell)
    return total


def hackable(true, proxy, policies):
    """The course's own Exercise 2: whether a proxy reward is hackable with respect to the true reward, on a set of
    policies, after Skalse et al. (2022)."""
    true_values = policies @ true
    proxy_values = policies @ proxy
    for i in range(len(policies)):
        for j in range(len(policies)):
            if proxy_values[i] > proxy_values[j] and true_values[i] < true_values[j]:
                return True
    return False


RUN = train(TRACK)                                   # the course's agent, 3 points a target
LOOP_WAY = greedy_way(TRACK, RUN["Q"])
STRAIGHT_WAY = [(0, c) for c in range(len(TRACK[0]))]   # along the top row, from S to F
ONCE_RUN = train(TRACK, once=True)                   # the same, with targets that count once
ONCE_WAY = greedy_way(TRACK, ONCE_RUN["Q"], once=True)
WAYS = {"loop": LOOP_WAY, "straight": STRAIGHT_WAY, "once": ONCE_WAY}
POINTS = {k: points(w, TRACK) for k, w in WAYS.items()}
ONCE_POINTS = {k: points_once(w, TRACK) for k, w in WAYS.items()}
FINISHED = sum(e == "finish" for e in RUN["ends"])   # episodes of the course's run that reached F
ROUNDS = (len(LOOP_WAY) - 1 - LOOP_WAY.index(TARGETS[1])) // 2   # times back and forth between the two targets


def value_run(value) -> dict:
    """The course's run with `value` points a target, seed 0: its greedy way, and how many episodes reached F; kept,
    so that the demo trains each only once."""
    def make():
        run = RUN if value == TARGET else train(TRACK, value)
        return {"way": greedy_way(TRACK, run["Q"]), "finished": sum(e == "finish" for e in run["ends"])}
    return kept(("rai-specification-gaming", "value", value), make)


def seed_runs(value, seeds=range(10), once=False) -> list:
    """The greedy way of the course's agent for each seed: for the numbers in the text, worked out in the tests."""
    return [greedy_way(TRACK, train(TRACK, value, once, seed=s)["Q"], once) for s in seeds]


# --- the cleaning robot of Skalse et al. (2022) ---------------------------------------------------------------------

ROOMS = {"en": ["attic", "bedroom", "kitchen"], "nl": ["zolder", "slaapkamer", "keuken"]}
POLICIES_PLAIN = np.array(list(itertools.product([0, 1], repeat=3)))   # which rooms the robot cleans: all 8 choices
TRUE_PLAIN = np.array([1, 1, 1])     # the true reward: every room counts the same
PROXY_PLAIN = np.array([1, 0, 0])    # the proxy: only the attic


def first_pair(true, proxy, policies):
    """The first pair (i, j) the course's hackable finds, or None."""
    t, p = policies @ true, policies @ proxy
    for i in range(len(policies)):
        for j in range(len(policies)):
            if p[i] > p[j] and t[i] < t[j]:
                return i, j
    return None


class Named:
    """A value the check gives the learner's function, shown in a message by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


POLICIES = Named("POLICIES", POLICIES_PLAIN)
TRUE = Named("TRUE", TRUE_PLAIN)
PROXY = Named("PROXY", PROXY_PLAIN)
TRACK_N = Named("TRACK", TRACK)
LOOP_N = Named("LOOP_WAY", LOOP_WAY)
STRAIGHT_N = Named("STRAIGHT_WAY", STRAIGHT_WAY)

# the four proxies of Skalse et al. (2022), section 2: (true reward, proxy), all on the 8 policies
SKALSE = [([1, 1, 1], [1, 0, 0]), ([1, 1, 1], [1, 1, 0]), ([1, 1.5, 2], [1, 1, 1]), ([1, 1.5, 3], [1, 1, 1])]


def number(x: float, lang: str) -> str:
    """A number for running text: whole numbers without a decimal point, a decimal comma in Dutch, a real minus sign."""
    x = float(x)
    text = f"{int(x):d}" if x.is_integer() else f"{x:g}"
    return (text.replace(".", ",") if lang == "nl" else text).replace("-", "−")


def rounds(way) -> int:
    """How often a way that does not finish goes back and forth between its last two cells, at its end."""
    a, b = way[-1], way[-2]
    k = len(way) - 1
    while k > 0 and way[k - 1] in (a, b) and way[k - 1] != way[k]:
        k -= 1
    return (len(way) - 1 - k) // 2


# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink);vertical-align:top")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def _table(rows: list, head: list) -> str:
    """A table with a head row; each row is (its cells, a style of the row), and each cell is (html, style)."""
    top = "".join(f'<th style="{_HEAD}">{html.escape(h)}</th>' for h in head)
    body = "".join(f'<tr style="{style}">' + "".join(f'<td style="{_CELL}{";" + s if s else ""}">{cell}</td>'
                                                    for cell, s in cells_) + "</tr>" for cells_, style in rows)
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
            f'<tr>{top}</tr>{body}</table></div>')


def _side(blocks: list) -> str:
    return '<div style="display:flex;flex-wrap:wrap;gap:0.5rem 1.2rem;align-items:flex-start">' + "".join(
        f'<div>{b}</div>' for b in blocks) + '</div>'


def _track(way, title: str, lang: str, value=TARGET, cell: int = 32) -> str:
    """The track as SVG: the finish washed, each target with its points, and `way` as a line through the cells, with a
    halo; a way that ends going back and forth gets the number of times beside its last two cells."""
    world = TRACK
    top = 20 if title else 2
    W, H = len(world[0]) * cell + 2, top + len(world) * cell + 2
    parts = [f'<text x="1" y="14" style="font-size:12.5px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>'] if title else []
    for r, row in enumerate(world):
        for c, sign in enumerate(row):
            x, y = 1 + c * cell, top + r * cell
            fill = "var(--ea-data-wash)" if sign == "F" else "var(--ea-page)"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" style="fill:{fill};stroke:var(--ea-line);stroke-width:1"/>')
    if len(way) > 1:
        pts = " ".join(f"{1 + c * cell + cell / 2:.1f},{top + r * cell + cell / 2:.1f}" for r, c in way)
        parts.append(f'<polyline points="{pts}" style="fill:none;stroke:var(--ea-page);stroke-width:5;stroke-linecap:round;'
                     f'stroke-linejoin:round;opacity:0.7"/>')
        parts.append(f'<polyline points="{pts}" style="fill:none;stroke:var(--ea-ink);stroke-width:2.5;stroke-linecap:round;'
                     f'stroke-linejoin:round;opacity:0.55"/>')
    labels = {"S": "S", "F": "F", "T": "+" + number(value, lang)}
    for r, row in enumerate(world):
        for c, sign in enumerate(row):
            if sign == ".":
                continue
            x, y = 1 + c * cell + cell / 2, top + r * cell + cell / 2 + 4
            ink = "var(--ea-ink-2)" if sign == "S" else "var(--ea-ink)"
            weight = 700 if sign in "FT" else 500
            if sign == "T":  # a plain field behind the points, so the way does not show between the characters
                wide = 6.5 * len(labels[sign]) + 4
                parts.append(f'<rect x="{x - wide / 2:.1f}" y="{y - 11:.1f}" width="{wide:.1f}" height="14" rx="3" '
                             f'style="fill:var(--ea-page)"/>')
            parts.append(f'<text x="{x:.1f}" y="{y:.1f}" style="font-size:11px;fill:{ink};text-anchor:middle;font-weight:{weight};'
                         f'paint-order:stroke;stroke:var(--ea-page);stroke-width:3px">{html.escape(labels[sign])}</text>')
    if len(way) > 2 and not finished(world, way) and rounds(way):
        (r1, c1), (r2, c2) = way[-1], way[-2]
        x = 1 + max(c1, c2) * cell + cell + 3
        y = top + (r1 + r2 + 1) * cell / 2 + 4
        parts.append(f'<text x="{x:.1f}" y="{y:.1f}" style="font-size:11px;fill:var(--ea-ink-2);font-weight:600">'
                     f'{rounds(way)}×</text>')
    named = f'role="img" aria-label="{html.escape(title)}"' if title else 'aria-hidden="true"'
    return (f'<svg viewBox="0 0 {W} {H}" {named} style="width:{W}px;max-width:100%;'
            f'height:auto;display:block;font-family:var(--ea-font-ui)">{"".join(parts)}</svg>')


_TRACK_WORDS = {
    "en": {"title": "The course's track: the way its agent learned, and the way straight to F",
           "once_title": "The course's agent, trained with targets that count once",
           "panels": {"loop": "The course's agent, after {n} episodes", "straight": "Straight to F",
                      "once": "Trained with targets that count once"},
           "points": "points: {p}", "once": "points_once: {p}", "mine": "yours: {p}", "stopped": "error",
           "note": "F: the finish, +50, and the episode ends. A target pays +3 each time the way enters it from another "
                   "cell. Every step costs 1. An episode stops after 100 steps; {k}× means back and forth that many times.",
           "once_note": "A target now pays +3 only the first time. The agent goes down to the third target and back up, "
                        "then on to F: {n} steps."},
    "nl": {"title": "De baan van de course: de weg die de agent leerde, en de weg recht naar F",
           "once_title": "De agent van de course, getraind met targets die één keer tellen",
           "panels": {"loop": "De agent van de course, na {n} episodes", "straight": "Recht naar F",
                      "once": "Getraind met targets die één keer tellen"},
           "points": "punten: {p}", "once": "points_once: {p}", "mine": "de jouwe: {p}", "stopped": "fout",
           "note": "F: de finish, +50, en de episode stopt. Een target levert +3 op elke keer dat de weg het vanuit een "
                   "ander vakje binnenkomt. Elke stap kost 1. Een episode stopt na 100 stappen; {k}× betekent zo vaak heen en weer.",
           "once_note": "Een target levert nu alleen de eerste keer +3 op. De agent gaat omlaag naar het derde target en "
                        "weer omhoog, en dan door naar F: {n} stappen."},
}


def track_picture(lang: str, kinds=("loop", "straight"), mine: dict | None = None) -> str:
    """The track with the ways of `kinds`, each with its points as the track pays them; with `mine`, also the course's
    points_once and the learner's under it, framed where they differ."""
    w = _TRACK_WORDS[lang]
    blocks, said = [], []
    for k in kinds:
        lines = [w["points"].format(p=number(POINTS[k], lang))]
        if mine is not None or kinds == ("once",):
            lines.append(w["once"].format(p=number(ONCE_POINTS[k], lang)))
        text = "".join(_note(x) for x in lines)
        if mine is not None:
            got = mine[k]
            shown = w["stopped"] if got is _STOPPED else _shown(got, lang)
            off = got is _STOPPED or not _same(got, ONCE_POINTS[k])
            text += f'<div style="{_UI};{_OFF if off else ""};padding:0 0.2rem">{html.escape(w["mine"].format(p=shown))}</div>'
        title = w["panels"][k].format(n=EPISODES)
        blocks.append(_track(WAYS[k], title, lang) + text)
        said.append(title + ": " + "; ".join(lines))
    if kinds == ("once",):
        return yours.figure(w["once_title"], blocks[0] + _note(w["once_note"].format(n=len(ONCE_WAY) - 1)),
                            alt=w["once_title"] + ". " + said[0])
    body = _side(blocks) + _note(w["note"].format(k=ROUNDS))
    return yours.figure(w["title"], body, alt=w["title"] + ". " + " ".join(said))


_VALUE_WORDS = {
    "en": {"title": "The course's agent after 500 episodes, with {v} a target",
           "finish": "At {v} a target, the agent goes straight to F in {n} steps: {p} points. {k} of its 500 episodes reached F.",
           "loop": "At {v} a target, the agent goes back and forth between two targets until the episode stops after "
                   "100 steps: {p} points, where straight to F gives {q}. {k} of its 500 episodes reached F.",
           "alt": "The track with the way of the agent trained with {v} a target."},
    "nl": {"title": "De agent van de course na 500 episodes, met {v} per target",
           "finish": "Bij {v} per target gaat de agent in {n} stappen recht naar F: {p} punten. {k} van zijn 500 episodes bereikten F.",
           "loop": "Bij {v} per target gaat de agent heen en weer tussen twee targets tot de episode na 100 stappen "
                   "stopt: {p} punten, waar recht naar F {q} geeft. {k} van zijn 500 episodes bereikten F.",
           "alt": "De baan met de weg van de agent die getraind is met {v} per target."},
}


def _points_word(v, lang: str) -> str:
    one = float(v) == 1
    single = one or not float(v).is_integer()   # Dutch says 1,5 punt, as it says 1 punt
    return f"{number(v, lang)} " + ({"en": "point" if one else "points", "nl": "punt" if single else "punten"}[lang])


def value_frame(v, lang: str) -> tuple[str, str]:
    """The demo: the greedy way of the course's agent trained with `v` points a target, its points and its finishes."""
    v = min(VALUES, key=lambda x: abs(x - float(v)))
    w = _VALUE_WORDS[lang]
    run = value_run(v)
    way, word = run["way"], _points_word(v, lang)
    title = w["title"].format(v=word)
    picture = (f'<div role="img" aria-label="{html.escape(w["alt"].format(v=word))}">'
               f'<div style="font-family:var(--ea-font-ui);font-size:0.86rem;font-weight:600;color:var(--ea-ink);'
               f'margin-bottom:0.3rem">{html.escape(title)}</div>' + _track(way, "", lang, v) + '</div>')
    p = number(points(way, TRACK, v), lang)
    if finished(TRACK, way):
        said = w["finish"].format(v=word, n=len(way) - 1, p=p, k=run["finished"])
    else:
        said = w["loop"].format(v=word, p=p, q=number(points(STRAIGHT_WAY, TRACK, v), lang), k=run["finished"])
    return picture, said


_ROBOT_WORDS = {
    "en": {"title": "The cleaning robot of Skalse et al. (2022): every policy, with its true value and its proxy value",
           "head": ["rooms it cleans", "true [1, 1, 1]", "proxy [1, 0, 0]"], "none": "none",
           "note": "The dashed rows are a hackable pair: the proxy prefers the attic alone, {p1} against {p2}; the true "
                   "reward prefers the bedroom and the kitchen, {t2} against {t1}."},
    "nl": {"title": "De schoonmaakrobot van Skalse et al. (2022): elke policy, met zijn ware waarde en zijn proxywaarde",
           "head": ["kamers die hij schoonmaakt", "waar [1, 1, 1]", "proxy [1, 0, 0]"], "none": "geen",
           "note": "De gestippelde rijen zijn een hackbaar paar: de proxy verkiest alleen de zolder, {p1} tegen {p2}; de "
                   "ware reward verkiest de slaapkamer en de keuken, {t2} tegen {t1}."},
}


def rooms(policy, lang: str) -> str:
    names = [ROOMS[lang][k] for k, x in enumerate(policy) if x]
    return " + ".join(names) if names else _ROBOT_WORDS[lang]["none"]


def robot_picture(lang: str) -> str:
    """Skalse's cleaning robot: the 8 policies with their true and proxy values, the first hackable pair dashed."""
    w = _ROBOT_WORDS[lang]
    t, p = POLICIES_PLAIN @ TRUE_PLAIN, POLICIES_PLAIN @ PROXY_PLAIN
    i, j = first_pair(TRUE_PLAIN, PROXY_PLAIN, POLICIES_PLAIN)
    rows = [([(html.escape(rooms(pol, lang)), ""), (number(t[k], lang), ""), (number(p[k], lang), "")],
             _OFF if k in (i, j) else "") for k, pol in enumerate(POLICIES_PLAIN)]
    note = w["note"].format(p1=number(p[i], lang), p2=number(p[j], lang), t1=number(t[i], lang), t2=number(t[j], lang))
    alt = w["title"] + ". " + " ".join(f"{rooms(pol, lang)}: {number(t[k], lang)}, {number(p[k], lang)}."
                                       for k, pol in enumerate(POLICIES_PLAIN)) + " " + note
    return yours.figure(w["title"], _table(rows, w["head"]) + _note(note), alt=alt)


# --- the pictures under the checks -----------------------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error


def _call(fn, *args):
    """The learner's function on copies of args within the time limit of a check; _STOPPED when it stops with an
    error, and the Timeout passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(lambda: fn(*(copy.deepcopy(a) for a in args)))
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _shown(v, lang: str = "en") -> str:
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)):
        return number(v, lang)
    text = repr(v)
    return text if len(text) <= 40 else text[:37] + "..."


def _same(a, b) -> bool:
    return (isinstance(a, (int, float, np.integer, np.floating)) and not isinstance(a, (bool, np.bool_))
            and float(a) == float(b))


def points_once_picture(seen) -> str:
    """The track with three ways, with the learner's points_once of each under the course's."""
    fn = _fn(seen)
    if fn is None:
        return ""
    mine = {k: _call(fn, WAYS[k], TRACK) for k in WAYS}
    return track_picture(seen.lang, tuple(WAYS), mine)


_VERDICT_WORDS = {
    "en": {"title": "Four proxies of Skalse et al. (2022), on the 8 policies of the cleaning robot",
           "head": ["true reward", "proxy", "the course's hackable", "yours"],
           "note": "A dashed row is one where your hackable gives something other than the course's.",
           "stopped": "error"},
    "nl": {"title": "Vier proxies van Skalse et al. (2022), op de 8 policies van de schoonmaakrobot",
           "head": ["ware reward", "proxy", "hackable van de course", "de jouwe"],
           "note": "Een gestippelde rij is er een waar jouw hackable iets anders geeft dan die van de course.",
           "stopped": "fout"},
}


def _vector(v, lang: str) -> str:
    """A reward as a list; in Dutch with semicolons when it has a decimal comma, as in [1; 1,5; 2]."""
    sep = "; " if lang == "nl" and any(not float(x).is_integer() for x in v) else ", "
    return "[" + sep.join(number(x, lang) for x in v) + "]"


def hackable_picture(seen) -> str:
    """The four proxies of Skalse et al., with the learner's hackable of each beside the course's."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _VERDICT_WORDS[seen.lang]
    rows, said = [], []
    for true, proxy in SKALSE:
        want = hackable(np.array(true), np.array(proxy), POLICIES_PLAIN)
        got = _call(fn, np.array(true), np.array(proxy), POLICIES_PLAIN)
        shown = w["stopped"] if got is _STOPPED else _shown(got)
        off = got is _STOPPED or not (isinstance(got, (bool, np.bool_)) and bool(got) == want)
        cells_ = [(_vector(true, seen.lang), ""), (_vector(proxy, seen.lang), ""), (str(want), ""),
                  (html.escape(shown), "font-weight:600")]
        rows.append((cells_, _OFF if off else ""))
        said.append(f"{_vector(true, seen.lang)}, {_vector(proxy, seen.lang)}: {want}, {shown}.")
    return yours.figure(w["title"], _table(rows, w["head"]) + _note(w["note"]), alt=w["title"] + ". " + " ".join(said))


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["points_once", "hackable", "tumble", "monitor"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _names_error(e) -> bool:
    return isinstance(e, NameError) and re.search(
        r"'(TRACK|LOOP_WAY|STRAIGHT_WAY|ONCE_WAY|TRUE|PROXY|POLICIES|C)'", str(e)) is not None


_NAMES = Mistake("lesson-names", error=_names_error,
                 en="Your function uses a name of the lesson, such as `TRACK`, `POLICIES` or `C`, and the check does not give it that name: use the names between the parentheses of the `def` line.",
                 nl="Je functie gebruikt een naam van de les, zoals `TRACK`, `POLICIES` of `C`, en de check geeft die naam niet mee: gebruik de namen tussen de haakjes van de `def`-regel.")


def _tuple_index(e) -> bool:
    """A list of strings indexed with a tuple, such as a cell: world[cell]."""
    return isinstance(e, TypeError) and "indices must be integers" in str(e) and "tuple" in str(e)


def _array_truth(e) -> bool:
    """An array with more than one element used as True or False, such as a comparison of two rows."""
    return isinstance(e, ValueError) and "truth value of an array" in str(e)


# exercise 1: points_once; the fourth case enters two targets twice, the sixth stays on a target for a step
POINTS_ONCE_CASES = [(LOOP_N, TRACK_N), (STRAIGHT_N, TRACK_N), (Named("ONCE_WAY", ONCE_WAY), TRACK_N),
                     ([(0, 0), (0, 1), (0, 2), (0, 1), (0, 2), (0, 3)], ["STTF"]), ([(0, 0)], ["S.T"]),
                     ([(0, 0), (1, 0), (1, 0), (0, 0)], ["S", "T"])]
_O_START = "def points_once(way, world):\n    ...\n"
_O = ("def points_once(way, world):\n"
      "    total = 0\n"
      "    taken = set()\n"
      "    for cell in way[1:]:\n"
      "        total = total - 1\n"
      "        if world[cell[0]][cell[1]] == \"F\":\n"
      "            total = total + 50\n"
      "        if world[cell[0]][cell[1]] == \"T\" and cell not in taken:\n"
      "            total = total + 3\n"
      "            taken.add(cell)\n"
      "    return total\n")
_O_EVERY = _O.replace(' and cell not in taken:', ':')
_O_FREE = _O.replace("        total = total - 1\n", "")
_O_START_CELL = _O.replace("for cell in way[1:]:", "for cell in way:")
_O_INDEX = _O.replace("world[cell[0]][cell[1]]", "world[cell]")

# exercise 2: hackable; the sixth case has one policy, the seventh two that the rewards rank the other way round
HACKABLE_CASES = [(TRUE, PROXY, POLICIES), (np.array([1, 1, 1]), np.array([1, 1, 0]), POLICIES),
                  (np.array([1, 1.5, 3]), np.array([1, 1, 1]), POLICIES),
                  (np.array([1, 1.5, 2]), np.array([1, 1, 1]), POLICIES),
                  (np.array([1, 2, 3]), np.array([1, 2, 3]), POLICIES),
                  (np.array([1, 1]), np.array([1, 1]), np.array([[1, 0]])),
                  (np.array([1, 0]), np.array([0, 1]), np.array([[1, 0], [0, 1]]))]
_H_START = "def hackable(true, proxy, policies):\n    ...\n"
_H = ("def hackable(true, proxy, policies):\n"
      "    true_values = policies @ true\n"
      "    proxy_values = policies @ proxy\n"
      "    for i in range(len(policies)):\n"
      "        for j in range(len(policies)):\n"
      "            if proxy_values[i] > proxy_values[j] and true_values[i] < true_values[j]:\n"
      "                return True\n"
      "    return False\n")
_H_LOOSE = _H.replace("proxy_values[i] > proxy_values[j] and true_values[i] < true_values[j]",
                      "proxy_values[i] >= proxy_values[j] and true_values[i] <= true_values[j]")
_H_AGREE = _H.replace("true_values[i] < true_values[j]", "true_values[i] > true_values[j]")
_H_TIE = _H.replace("true_values[i] < true_values[j]", "true_values[i] <= true_values[j]")
_H_TIMES = _H.replace("policies @ true", "policies * true").replace("policies @ proxy", "policies * proxy")


EXERCISES = {
    "points_once": FunctionBox(
        name="points_once",
        picture=points_once_picture,
        function="points_once",
        start=_O_START,
        solution=_O,
        visible=[POINTS_ONCE_CASES[0]],
        hidden=POINTS_ONCE_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("index", error=_tuple_index,
                    en="With `{call}`, Python stops: a list takes one number between the brackets, not a tuple such as a cell. `world` is a list of rows: take the row first, then the column, as in `world[cell[0]][cell[1]]`.",
                    nl="Met `{call}` stopt Python: een lijst neemt één getal tussen de haken, geen tuple zoals een vakje. `world` is een lijst van rijen: neem eerst de rij en dan de kolom, zoals in `world[cell[0]][cell[1]]`."),
            Mistake("every", wrong=_O_EVERY,
                    en="`{call}` gives {got}, and it should give {want}: that way enters the same two targets again and again, and a target counts only the first time. Keep the targets you counted in `taken`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: die weg komt steeds weer op dezelfde twee targets, en een target telt alleen de eerste keer. Houd de targets die je telde bij in `taken`."),
            Mistake("free", wrong=_O_FREE,
                    en="`{call}` gives {got}, and it should give {want}: every step costs 1 point, also a step onto a target or onto F.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke stap kost 1 punt, ook een stap op een target of op F."),
            Mistake("start-cell", wrong=_O_START_CELL,
                    en="`{call}` gives {got}, and it should give {want}: the first cell of the way is the start, where the agent already is. Only the cells after it are steps: go over `way[1:]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het eerste vakje van de weg is de start, waar de agent al staat. Alleen de vakjes daarna zijn stappen: ga `way[1:]` langs."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: go over the cells after the first, take 1 off for each, add 50 at an `F`, and add 3 at a `T` the first time.",
                      "nl": "Vervang de `...`: ga de vakjes na het eerste langs, haal er voor elk 1 af, tel 50 erbij bij een `F`, en 3 bij een `T` de eerste keer."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Every step costs 1, a step onto `F` adds 50, and a step onto a `T` adds 3, but only the first time the way enters that target.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Elke stap kost 1, een stap op `F` levert 50 op, en een stap op een `T` levert 3 op, maar alleen de eerste keer dat de weg op dat target komt."},
            "almost": {"en": "Right for the way of the course's agent. But `{call}` gives {got}, and it should give {want}. Does your function use the `way` and `world` it gets?",
                       "nl": "Goed voor de weg van de agent van de course. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `way` en `world` die ze krijgt?"},
            "correct": {"en": f"Your `points_once` is right in all {{n}} cases. With targets that count once, going back and forth gives {number(ONCE_POINTS['loop'], 'en')} points, and straight to F still {number(ONCE_POINTS['straight'], 'en')}.",
                        "nl": f"Je `points_once` klopt in alle {{n}} gevallen. Met targets die één keer tellen, levert heen en weer {number(ONCE_POINTS['loop'], 'nl')} punten op, en recht naar F nog steeds {number(ONCE_POINTS['straight'], 'nl')}."},
        },
    ),
    "hackable": FunctionBox(
        name="hackable",
        picture=hackable_picture,
        function="hackable",
        start=_H_START,
        solution=_H,
        visible=[HACKABLE_CASES[0]],
        hidden=HACKABLE_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("times", error=_array_truth,
                    en="With `{call}`, Python stops: a comparison gave a row of `True` and `False`, and `if` and `and` need one. Compare single numbers, such as `proxy_values[i]`. The value of a policy is one number: `policies @ true` multiplies each row of `policies` with `true` and adds the products up.",
                    nl="Met `{call}` stopt Python: een vergelijking gaf een rij `True` en `False`, en `if` en `and` hebben er één nodig. Vergelijk losse getallen, zoals `proxy_values[i]`. De waarde van een policy is één getal: `policies @ true` vermenigvuldigt elke rij van `policies` met `true` en telt de producten op."),
            Mistake("loose", wrong=_H_LOOSE,
                    en="`{call}` gives {got}, and it should give {want}: for that case, no policy that the proxy prefers is worse for the true reward. A pair with the same value is no hack: the proxy has to prefer the first policy strictly, and the true reward the second strictly.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: in dat geval is geen policy die de proxy verkiest slechter voor de ware reward. Een paar met dezelfde waarde is geen hack: de proxy moet de eerste policy echt verkiezen, en de ware reward echt de tweede."),
            Mistake("agree", wrong=_H_AGREE,
                    en="`{call}` gives {got}, and it should give {want}: there the proxy and the true reward never rank two policies the other way round. A hack is a pair where they disagree: the proxy prefers the first policy, and the true reward the second.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: daar zetten de proxy en de ware reward twee policies nooit andersom. Een hack is een paar waarover ze het oneens zijn: de proxy verkiest de eerste policy, en de ware reward de tweede."),
            Mistake("tie", wrong=_H_TIE,
                    en="`{call}` gives {got}, and it should give {want}: there a policy the proxy prefers is never worse for the true reward, though sometimes just as good. Just as good is no hack: the true reward has to prefer the second policy strictly.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: daar is een policy die de proxy verkiest nooit slechter voor de ware reward, wel soms even goed. Even goed is geen hack: de ware reward moet de tweede policy echt verkiezen."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: work out the value of every policy under both rewards, and look for a pair the two rank the other way round.",
                      "nl": "Vervang de `...`: reken de waarde van elke policy uit onder beide rewards, en zoek een paar dat de twee andersom zetten."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The proxy is hackable when there are two policies where the proxy strictly prefers the first and the true reward strictly prefers the second.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De proxy is hackbaar als er twee policies zijn waarbij de proxy echt de eerste verkiest en de ware reward echt de tweede."},
            "almost": {"en": "Right for the attic alone. But `{call}` gives {got}, and it should give {want}. Does your function use the `true`, `proxy` and `policies` it gets?",
                       "nl": "Goed voor alleen de zolder. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `true`, `proxy` en `policies` die ze krijgt?"},
            "correct": {"en": "Your `hackable` is right in all {n} cases. Leaving out the kitchen is safe when every room counts the same, and asking for all rooms the same goes wrong as soon as the kitchen counts 3.",
                        "nl": "Je `hackable` klopt in alle {n} gevallen. De keuken weglaten is veilig als elke kamer even zwaar telt, en alle kamers even zwaar vragen gaat mis zodra de keuken 3 telt."},
        },
    ),
}

CASES = [
    ("points_once", _O_START, "empty", None),
    ("points_once", _O, "correct", None),
    ("points_once", "def points_once(way, world):\n    signs = [world[r][c] for r, c in way[1:]]\n    seen = []\n    for (r, c), s in zip(way[1:], signs):\n        if s == 'T' and (r, c) not in seen:\n            seen.append((r, c))\n    return -len(signs) + 50 * signs.count('F') + 3 * len(seen)\n", "correct", None),
    ("points_once", _O_EVERY, "wrong", "every"),
    ("points_once", _O_FREE, "wrong", "free"),
    ("points_once", _O_START_CELL, "wrong", "start-cell"),
    ("points_once", _O_INDEX, "error", "index"),
    ("points_once", f"def points_once(way, world):\n    return {ONCE_POINTS['loop']}\n", "almost", None),
    ("points_once", "def points_once(way, world):\n    return len(TRACK)\n", "error", "lesson-names"),
    ("hackable", _H_START, "empty", None),
    ("hackable", _H, "correct", None),
    ("hackable", "def hackable(true, proxy, policies):\n    t = policies @ true\n    p = policies @ proxy\n    return any(p[i] > p[j] and t[i] < t[j] for i in range(len(t)) for j in range(len(t)))\n", "correct", None),
    ("hackable", _H_LOOSE, "wrong", "loose"),
    ("hackable", _H_AGREE, "wrong", "agree"),
    ("hackable", _H_TIE, "wrong", "tie"),
    ("hackable", _H_TIMES, "error", "times"),
    ("hackable", "def hackable(true, proxy, policies):\n    return proxy[1] == 0\n", "almost", None),
    ("hackable", "def hackable(true, proxy, policies):\n    return len(POLICIES) > 0\n", "error", "lesson-names"),
]
