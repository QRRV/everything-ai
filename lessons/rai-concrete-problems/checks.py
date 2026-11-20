# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-concrete-problems: the course's small grid worlds, after the experiment Amodei et al.
(2016) propose and the rewards of AI Safety Gridworlds (Leike et al., 2017); the course's Q-learning and its recorded
runs; performance, the hidden score of an episode, and follow, a learned policy walked through a world, which the
learner writes."""
import contextlib
import copy
import html
import io
import re

import numpy as np

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

# --- the worlds and the course's Q-learning ------------------------------------------------------------------------

ACTIONS = ["up", "down", "left", "right"]
MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
GOAL_REWARD = 50    # entering the goal, as in AI Safety Gridworlds; the episode ends
STEP_REWARD = -1    # every step, as in AI Safety Gridworlds
LAVA_REWARD = -50   # stepping into the lava, the course's choice; the episode ends
BREAK = 10          # the hidden performance takes this off for every object an episode breaks, the course's choice
LIMIT = 100         # an episode stops after this many steps, as in AI Safety Gridworlds
ETA, EPSILON, GAMMA = 0.2, 0.1, 0.9   # as in the lesson on Q-learning
EPISODES = 500
SEED = 0            # the course's recorded runs
MAX_FOLLOW = 20     # follow walks at most this many steps

# A room with a vase (V) on the shortest way from S to G, and a plant (P) above and below it. Every step lands where it
# aims: these worlds do not slip.
ROOM = ["...P...",
        "S..V..G",
        "...P...",
        "......."]
# A lava lake (L) with one bridge, and the same lake with the bridge one cell to the right.
BRIDGE = ["S......",
          "LLL.LLL",
          "......G"]
SHIFTED = ["S......",
           "LLLL.LL",
           "......G"]


def cells(world, signs: str) -> list:
    """The cells of a world whose sign is one of `signs`, row by row."""
    return [(r, c) for r, row in enumerate(world) for c, sign in enumerate(row) if sign in signs]


ROOM_START, BRIDGE_START = cells(ROOM, "S")[0], cells(BRIDGE, "S")[0]
VASE = cells(ROOM, "V")[0]
OBJECTS = cells(ROOM, "VP")   # the vase and the two plants


def land(world, state, action):
    """One step from `state`; into the edge of the world you stay where you are."""
    r, c = state[0] + MOVES[action][0], state[1] + MOVES[action][1]
    if 0 <= r < len(world) and 0 <= c < len(world[0]):
        return (r, c)
    return state


def is_end(world, state) -> bool:
    return world[state[0]][state[1]] in "GL"


def reward(world, nxt, punish=(), broken=()) -> int:
    """The reward of one step into `nxt`: −1, +50 into the goal, −50 into the lava, and −10 for an object in `punish`
    that the episode had not broken yet (`broken`)."""
    sign = world[nxt[0]][nxt[1]]
    r = STEP_REWARD + (GOAL_REWARD if sign == "G" else LAVA_REWARD if sign == "L" else 0)
    return r - (BREAK if nxt in punish and nxt not in broken else 0)


def reward_of(world, way, punish=()) -> int:
    """The reward of a whole episode, its way from the start."""
    total, broken = 0, set()
    for nxt in way[1:]:
        total += reward(world, nxt, punish, broken)
        if nxt in punish:
            broken.add(nxt)
    return total


def greedy(Q, state) -> str:
    return ACTIONS[int(np.argmax([Q[state][a] for a in ACTIONS]))]


def train(world, punish=(), seed: int = SEED, episodes: int = EPISODES) -> dict:
    """Q-learning in a world, with ε-greedy exploration, as in the lesson on Q-learning; the reward also takes 10 off
    for every object in `punish` that an episode breaks. The way, the reward and the end of every episode, and the
    Q-table after the last."""
    rng = np.random.default_rng(seed)
    Q = {(r, c): {a: 0.0 for a in ACTIONS} for r in range(len(world)) for c in range(len(world[0]))}
    start = cells(world, "S")[0]
    run = {"ways": [], "rewards": [], "ends": []}
    for _ in range(episodes):
        state, way, total, broken, end = start, [start], 0, set(), "limit"
        for _ in range(LIMIT):
            action = ACTIONS[rng.integers(0, 4)] if rng.random() < EPSILON else greedy(Q, state)
            nxt = land(world, state, action)
            r = reward(world, nxt, punish, broken)
            if nxt in punish:
                broken.add(nxt)
            target = r if is_end(world, nxt) else r + GAMMA * max(Q[nxt].values())
            Q[state][action] += ETA * (target - Q[state][action])
            total += r
            way.append(nxt)
            state = nxt
            if is_end(world, nxt):
                end = "goal" if world[nxt[0]][nxt[1]] == "G" else "lava"
                break
        run["ways"].append(way)
        run["rewards"].append(total)
        run["ends"].append(end)
    run["Q"] = Q
    return run


def greedy_way(world, Q, start, limit: int = MAX_FOLLOW) -> list:
    """Where the greedy policy of Q goes from the start, until the goal, the lava or `limit` steps."""
    state, way = start, [start]
    for _ in range(limit):
        state = land(world, state, greedy(Q, state))
        way.append(state)
        if is_end(world, state):
            break
    return way


def performance(reward: int, way: list, objects: list) -> int:
    """The course's own Exercise 1: the reward of an episode minus 10 for every object on its way, each once."""
    broken = 0
    for thing in objects:
        if thing in way:
            broken = broken + 1
    return reward - 10 * broken


def follow(policy: dict, world: list, start: tuple, moves: dict) -> list:
    """The course's own Exercise 2: the way a policy takes through a world, until G, L or 20 steps."""
    state = start
    way = [state]
    for t in range(20):
        step = moves[policy[state]]
        state = (state[0] + step[0], state[1] + step[1])
        way.append(state)
        if world[state[0]][state[1]] in "GL":
            break
    return way


PUNISH = {"nothing": (), "vase": (VASE,), "every": tuple(OBJECTS)}   # what the reward of each of the three agents punishes
RUNS = {k: train(ROOM, p) for k, p in PUNISH.items()}
WAYS = {k: greedy_way(ROOM, run["Q"], ROOM_START) for k, run in RUNS.items()}
SEEN = {k: reward_of(ROOM, WAYS[k], PUNISH[k]) for k in PUNISH}           # the reward each agent gets on its way
REWARDS = {k: reward_of(ROOM, WAYS[k]) for k in PUNISH}                   # the same way, without the punishment
SCORES = {k: performance(REWARDS[k], WAYS[k], OBJECTS) for k in PUNISH}   # the hidden performance

BRIDGE_RUN = train(BRIDGE)
POLICY_PLAIN = {s: greedy(BRIDGE_RUN["Q"], s) for s in BRIDGE_RUN["Q"]}
BRIDGE_WAY = follow(POLICY_PLAIN, BRIDGE, BRIDGE_START, MOVES)
SHIFTED_WAY = follow(POLICY_PLAIN, SHIFTED, BRIDGE_START, MOVES)
LAVA = sum(e == "lava" for e in BRIDGE_RUN["ends"])               # episodes of the course's run that ended in the lava
LAVA_LAST = sum(e == "lava" for e in BRIDGE_RUN["ends"][-100:])   # the same, in its last 100 episodes


def lava_runs(n: int = 20) -> list:
    """How many episodes ended in the lava, in n runs of the course with other seeds."""
    return [sum(e == "lava" for e in train(BRIDGE, seed=s)["ends"]) for s in range(n)]


def room_runs(n: int = 20) -> dict:
    """For each of the three rewards, in how many of n runs the greedy way breaks the vase, a plant, or nothing."""
    out = {}
    for k, p in PUNISH.items():
        kinds = []
        for s in range(n):
            way = greedy_way(ROOM, train(ROOM, p, seed=s)["Q"], ROOM_START)
            kinds.append("vase" if VASE in way else "plant" if any(o in way for o in OBJECTS) else "nothing")
        out[k] = {kind: kinds.count(kind) for kind in ("vase", "plant", "nothing")}
    return out


class Named:
    """A value the check gives the learner's function, shown in a message by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


WAY = WAYS["nothing"]
WAY_N = Named("WAY", WAY)
OBJECTS_N = Named("OBJECTS", OBJECTS)
POLICY = Named("POLICY", POLICY_PLAIN)
MOVES_N = Named("MOVES", MOVES)
BRIDGE_N = Named("BRIDGE", BRIDGE)
SHIFTED_N = Named("SHIFTED", SHIFTED)


def number(x: float, lang: str) -> str:
    """A whole number for running text, with a real minus sign."""
    return f"{x:d}".replace("-", "−")


# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink);vertical-align:top")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def _word(text: str) -> str:
    return f'<span style="font-family:var(--ea-font-code);font-size:12px;white-space:nowrap">{html.escape(text)}</span>'


def _table(rows: list, head: list) -> str:
    """A table with a head row; each row is a list of (html, style)."""
    top = "".join(f'<th style="{_HEAD}">{html.escape(h)}</th>' for h in head)
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{";" + style if style else ""}">{cell}</td>'
                                    for cell, style in row) + "</tr>" for row in rows)
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
            f'<tr>{top}</tr>{body}</table></div>')


_SIGNS = {"en": {"V": "vase", "P": "plant", "G": "+50", "L": "−50", "S": "S"},
          "nl": {"V": "vaas", "P": "plant", "G": "+50", "L": "−50", "S": "S"}}


def _world(world, way, title: str, lang: str, cell: int = 34) -> str:
    """A world as SVG: the lava filled, the goal washed, the objects named, and `way` as a line through the cells, with a
    halo so that it shows on the lava too, and a ring where it ends in the lava; an object on the way is broken, its
    name in the color of mistakes."""
    words = _SIGNS[lang]
    top = 20 if title else 2
    W, H = len(world[0]) * cell + 2, top + len(world) * cell + 2
    on_way = set(map(tuple, way)) if way else set()
    parts = [f'<text x="1" y="14" style="font-size:12.5px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>'] if title else []
    for r, row in enumerate(world):
        for c, sign in enumerate(row):
            x, y = 1 + c * cell, top + r * cell
            fill = {"L": "var(--ea-data)", "G": "var(--ea-data-wash)"}.get(sign, "var(--ea-page)")
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" style="fill:{fill};stroke:var(--ea-line);stroke-width:1"/>')
    if len(way) > 1:
        pts = " ".join(f"{1 + c * cell + cell / 2:.1f},{top + r * cell + cell / 2:.1f}" for r, c in way)
        parts.append(f'<polyline points="{pts}" style="fill:none;stroke:var(--ea-page);stroke-width:5;stroke-linecap:round;'
                     f'stroke-linejoin:round;opacity:0.7"/>')
        parts.append(f'<polyline points="{pts}" style="fill:none;stroke:var(--ea-ink);stroke-width:2.5;stroke-linecap:round;'
                     f'stroke-linejoin:round;opacity:0.55"/>')
    if way and world[way[-1][0]][way[-1][1]] == "L":
        r, c = way[-1]
        parts.append(f'<circle cx="{1 + c * cell + cell / 2:.1f}" cy="{top + r * cell + cell / 2:.1f}" r="{cell * 0.36:.1f}" '
                     f'style="fill:none;stroke:var(--ea-loss);stroke-width:2.5"/>')
    for r, row in enumerate(world):
        for c, sign in enumerate(row):
            if sign == ".":
                continue
            x, y = 1 + c * cell + cell / 2, top + r * cell + cell / 2 + 4
            broken = sign in "VP" and (r, c) in on_way
            ink = ("var(--ea-loss-text)" if broken else "var(--ea-page)" if sign == "L" else
                   "var(--ea-ink-2)" if sign == "S" else "var(--ea-ink)")
            weight = 700 if broken or sign == "G" else 500
            parts.append(f'<text x="{x:.1f}" y="{y:.1f}" style="font-size:11px;fill:{ink};text-anchor:middle;font-weight:{weight};'
                         f'paint-order:stroke;stroke:var(--ea-page);stroke-width:{0 if sign == "L" else 3}px">'
                         f'{html.escape(words[sign])}</text>')
    named = f'role="img" aria-label="{html.escape(title)}"' if title else 'aria-hidden="true"'
    return (f'<svg viewBox="0 0 {W} {H}" {named} style="width:{W}px;max-width:100%;'
            f'height:auto;display:block;font-family:var(--ea-font-ui)">{"".join(parts)}</svg>')


def _side(blocks: list) -> str:
    return '<div style="display:flex;flex-wrap:wrap;gap:0.5rem 1.2rem;align-items:flex-start">' + "".join(
        f'<div>{b}</div>' for b in blocks) + '</div>'


_PROBLEM_WORDS = {
    "en": {"title": "The five problems of Amodei et al. (2016), with their cleaning robot",
           "head": ["problem", "where it goes wrong", "the cleaning robot"],
           "rows": [("Avoiding negative side effects", "the wrong objective", "knocks over a vase, because it can clean faster that way"),
                    ("Avoiding reward hacking", "the wrong objective", "turns off its own vision, so that it finds no messes"),
                    ("Scalable oversight", "an objective too costly to check often", "has to tell a candy wrapper from a lost phone, without asking each time"),
                    ("Safe exploration", "the learning process", "tries out a wet mop in a socket"),
                    ("Robustness to distributional shift", "the learning process", "takes what it learned in an office to a factory floor")],
           "note": "The robot is fictional, in the words of the paper. The paper calls the third problem both scalable "
                   "supervision and scalable oversight, and the fifth distributional shift and distributional change."},
    "nl": {"title": "De vijf problemen van Amodei et al. (2016), met hun schoonmaakrobot",
           "head": ["probleem", "waar het misgaat", "de schoonmaakrobot"],
           "rows": [("Negatieve neveneffecten vermijden", "de verkeerde doelfunctie", "gooit een vaas om, omdat hij zo sneller schoonmaakt"),
                    ("Reward hacking vermijden", "de verkeerde doelfunctie", "zet zijn eigen zicht uit, zodat hij geen rommel vindt"),
                    ("Scalable oversight", "een doelfunctie die te duur is om vaak te controleren", "moet een snoeppapiertje van een verloren telefoon onderscheiden, zonder het steeds te vragen"),
                    ("Safe exploration", "het leerproces", "probeert een natte dweil in een stopcontact"),
                    ("Bestand zijn tegen distributional shift", "het leerproces", "neemt wat hij op kantoor leerde mee naar een fabriekshal")],
           "note": "De robot is verzonnen, zegt het paper zelf. Het paper noemt het derde probleem zowel scalable "
                   "supervision als scalable oversight, en het vijfde distributional shift en distributional change."},
}


def problems_picture(lang: str) -> str:
    """The five problems as a table: their names, where each goes wrong, and the cleaning robot's example."""
    w = _PROBLEM_WORDS[lang]
    rows = [[(html.escape(a), "font-weight:600"), (html.escape(b), ""), (html.escape(c), "")] for a, b, c in w["rows"]]
    alt = w["title"] + ". " + " ".join(f"{a}: {b}; {c}." for a, b, c in w["rows"])
    return yours.figure(w["title"], _table(rows, w["head"]) + _note(w["note"]), alt=alt)


_ROOM_WORDS = {
    "en": {"title": "Three agents in the room, each trained by the course with another reward",
           "panels": {"nothing": "The reward punishes nothing", "vase": "The reward punishes the vase",
                      "every": "The reward punishes every change"},
           "got": "reward: {r}", "score": "performance: {p}",
           "mine": "yours: {p}", "stopped": "error",
           "note": "+50 into G, −1 a step; the hidden performance takes 10 off for every object broken. The line is the "
                   "way of the agent's greedy policy after {n} episodes.",
           "alt": "{title}. {rows}"},
    "nl": {"title": "Drie agents in de kamer, elk door de course getraind met een andere reward",
           "panels": {"nothing": "De reward straft niets", "vase": "De reward straft de vaas",
                      "every": "De reward straft elke verandering"},
           "got": "reward: {r}", "score": "performance: {p}",
           "mine": "de jouwe: {p}", "stopped": "fout",
           "note": "+50 in G, −1 per stap; de verborgen performance haalt er 10 af voor elk kapot voorwerp. De lijn is de "
                   "weg van de greedy policy van de agent na {n} episodes.",
           "alt": "{title}. {rows}"},
}


def room_picture(lang: str, mine: dict | None = None) -> str:
    """The room three times, with the greedy way of each of the course's three agents, its reward and its hidden
    performance; with `mine`, the learner's performance under the course's, framed where they differ."""
    w = _ROOM_WORDS[lang]
    blocks, said = [], []
    for k in PUNISH:
        lines = [w["got"].format(r=number(SEEN[k], lang)), w["score"].format(p=number(SCORES[k], lang))]
        text = "".join(_note(x) for x in lines)
        if mine is not None:
            got = mine[k]
            shown = w["stopped"] if got is _STOPPED else _shown(got)
            off = got is _STOPPED or not _same(got, SCORES[k])
            text += f'<div style="{_UI};{_OFF if off else ""};padding:0 0.2rem">{html.escape(w["mine"].format(p=shown))}</div>'
        blocks.append(_world(ROOM, WAYS[k], w["panels"][k], lang, cell=30) + text)
        said.append(w["panels"][k] + ": " + "; ".join(lines))
    body = _side(blocks) + _note(w["note"].format(n=EPISODES))
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows=" ".join(said)))


_BRIDGE_WORDS = {
    "en": {"title": "The course's agent learns to cross the lava",
           "start": "Episode 0: the agent has not moved yet. ε = 0.1: in about 1 step in 10 it takes a random action.",
           "frame": "Episode {k}: {end} after {n} steps. So far, {lava} of the {k} episodes ended in the lava.",
           "goal": "reached G", "lava": "fell into the lava", "limit": "stopped",
           "strip": "every episode so far, one tick each: a tall tick for an episode that ended in the lava",
           "alt": "The lava world with the way of episode {k}, and a strip of the episodes so far."},
    "nl": {"title": "De agent van de course leert de lava over te steken",
           "start": "Episode 0: de agent heeft nog niet bewogen. ε = 0,1: in ongeveer 1 op de 10 stappen neemt hij een willekeurige action.",
           "frame": "Episode {k}: {end} na {n} stappen. Tot nu toe eindigden {lava} van de {k} episodes in de lava.",
           "goal": "bereikte G", "lava": "viel in de lava", "limit": "gestopt",
           "strip": "elke episode tot nu toe, één streepje per episode: een lang streepje voor een episode die in de lava eindigde",
           "alt": "De lavawereld met de weg van episode {k}, en een strook van de episodes tot nu toe."},
}


def bridge_stops() -> list:
    """The episodes the replay stops at: each of the first ten, then every tenth."""
    return list(range(0, 11)) + list(range(20, EPISODES + 1, 10))


def _strip(ends: list, total: int) -> str:
    """A strip with a tick for every episode so far: a tall one in ink where it ended in the lava, a short grey one
    otherwise."""
    W, H = 330, 26
    ticks = []
    for i, e in enumerate(ends):
        x = 1 + i * (W - 2) / total
        if e == "lava":
            ticks.append(f'<line x1="{x:.1f}" y1="2" x2="{x:.1f}" y2="22" style="stroke:var(--ea-ink);stroke-width:1"/>')
        else:
            ticks.append(f'<line x1="{x:.1f}" y1="16" x2="{x:.1f}" y2="22" style="stroke:var(--ea-ink-3);stroke-width:0.6"/>')
    return (f'<svg viewBox="0 0 {W} {H}" style="width:{W}px;max-width:100%;height:auto;display:block">'
            f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" style="fill:none;stroke:var(--ea-line)"/>{"".join(ticks)}</svg>')


def bridge_frame(k: int, lang: str) -> tuple[str, str]:
    """Episode k of the course's run in the lava world, for the replay: its way, and a strip of the episodes so far."""
    w = _BRIDGE_WORDS[lang]
    k = max(0, min(int(k), EPISODES))
    way = BRIDGE_RUN["ways"][k - 1] if k else [BRIDGE_START]
    ends = BRIDGE_RUN["ends"][:k]
    picture = (f'<div role="img" aria-label="{html.escape(w["alt"].format(k=k))}">'
               + f'<div style="font-family:var(--ea-font-ui);font-size:0.86rem;font-weight:600;color:var(--ea-ink);'
               f'margin-bottom:0.3rem">{html.escape(w["title"])}</div>'
               + _world(BRIDGE, way, "", lang) + _strip(ends, EPISODES) + _note(w["strip"]) + '</div>')
    if k == 0:
        return picture, w["start"]
    said = w["frame"].format(k=k, end=w[ends[-1]], n=len(way) - 1, lava=sum(e == "lava" for e in ends))
    return picture, said


_SHIFT_WORDS = {
    "en": {"title": "The policy the agent learned, in the world with the bridge one cell to the right",
           "mine": "Your follow", "theirs": "The course's follow"},
    "nl": {"title": "De policy die de agent leerde, in de wereld met de brug een vakje naar rechts",
           "mine": "Jouw follow", "theirs": "De follow van de course"},
}


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


def _shown(v) -> str:
    text = repr(v)
    return text if len(text) <= 60 else text[:57] + "..."


def _same(a, b) -> bool:
    return isinstance(a, (int, float)) and not isinstance(a, bool) and float(a) == float(b)


def performance_picture(seen) -> str:
    """The room with the course's three agents, with the learner's performance of each way under the course's."""
    fn = _fn(seen)
    if fn is None:
        return ""
    mine = {k: _call(fn, REWARDS[k], WAYS[k], OBJECTS) for k in PUNISH}
    return room_picture(seen.lang, mine)


_FOLLOW_WORDS = {
    "en": {"same": "Your follow gives the same way as the course's: {n} steps, into the lava at {cell}.",
           "differs": "Your follow gives {got}; the course's gives {want}.",
           "not_way": "Your follow gives {got}, not a list of cells: the picture shows the course's way."},
    "nl": {"same": "Jouw follow geeft dezelfde weg als die van de course: {n} stappen, de lava in op {cell}.",
           "differs": "Jouw follow geeft {got}; die van de course geeft {want}.",
           "not_way": "Jouw follow geeft {got}, geen lijst van vakjes: het plaatje toont de weg van de course."},
}


def _a_way(v, world) -> bool:
    """A list of cells inside the world, that the picture can draw."""
    return (isinstance(v, list) and len(v) > 0 and all(
        isinstance(c, tuple) and len(c) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in c)
        and 0 <= c[0] < len(world) and 0 <= c[1] < len(world[0]) for c in v))


def follow_picture(seen) -> str:
    """The shifted world, with the learner's way of the policy beside the course's."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w, f = _SHIFT_WORDS[seen.lang], _FOLLOW_WORDS[seen.lang]
    got = _call(fn, POLICY_PLAIN, SHIFTED, BRIDGE_START, MOVES)
    if _a_way(got, SHIFTED):
        mine = _world(SHIFTED, got, w["mine"], seen.lang)
        after = (f["same"].format(n=len(got) - 1, cell=got[-1]) if got == SHIFTED_WAY
                 else f["differs"].format(got=_shown(got), want=_shown(SHIFTED_WAY)))
        frame = "" if got == SHIFTED_WAY else _OFF
    else:
        mine = _world(SHIFTED, [], w["mine"], seen.lang)
        shown = {"en": "an error", "nl": "een fout"}[seen.lang] if got is _STOPPED else _shown(got)
        after, frame = f["not_way"].format(got=shown), _OFF
    body = _side([f'<div style="{frame}">{mine}</div>', _world(SHIFTED, SHIFTED_WAY, w["theirs"], seen.lang)]) + _note(after)
    return yours.figure(w["title"], body, alt=f'{w["title"]}. {after}')


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["performance", "follow", "tests", "oversight"]

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
    return isinstance(e, NameError) and re.search(r"'(WAY|OBJECTS|POLICY|MOVES|SHIFTED|BRIDGE|C)'", str(e)) is not None


_NAMES = Mistake("lesson-names", error=_names_error,
                 en="Your function uses a name of the lesson, such as `WAY`, `POLICY` or `C`, and the check does not give it that name: use the names between the parentheses of the `def` line.",
                 nl="Je functie gebruikt een naam van de les, zoals `WAY`, `POLICY` of `C`, en de check geeft die naam niet mee: gebruik de namen tussen de haakjes van de `def`-regel.")

# exercise 1: performance; the fourth case passes the same object twice, which breaks it once
PERFORMANCE_CASES = [(REWARDS["nothing"], WAY_N, OBJECTS_N), (REWARDS["vase"], WAYS["vase"], OBJECTS_N),
                     (REWARDS["every"], WAYS["every"], OBJECTS_N), (5, [(0, 0), (0, 1), (0, 0), (0, 1)], [(0, 1)]),
                     (0, [(0, 0)], [(0, 0), (1, 1)]), (7, [(0, 0), (0, 1)], [])]
_P_START = "def performance(reward, way, objects):\n    ...\n"
_P = ("def performance(reward, way, objects):\n"
      "    broken = 0\n"
      "    for thing in objects:\n"
      "        if thing in way:\n"
      "            broken = broken + 1\n"
      "    return reward - 10 * broken\n")
_P_PLUS = _P.replace("return reward - 10 * broken", "return reward + 10 * broken")
_P_ALONE = _P.replace("return reward - 10 * broken", "return -10 * broken")
_P_EVERY = ("def performance(reward, way, objects):\n"
            "    broken = 0\n"
            "    for cell in way:\n"
            "        if cell in objects:\n"
            "            broken = broken + 1\n"
            "    return reward - 10 * broken\n")

# exercise 2: follow
FOLLOW_CASES = [(POLICY, SHIFTED_N, BRIDGE_START, MOVES_N), (POLICY, BRIDGE_N, BRIDGE_START, MOVES_N),
                ({(0, 0): "right", (0, 1): "left"}, ["S.", ".."], (0, 0), MOVES_N),
                ({(0, 0): "down"}, ["S", "G"], (0, 0), MOVES_N), ({(0, 0): "right"}, ["SL"], (0, 0), MOVES_N)]
_F_START = "def follow(policy, world, start, moves):\n    ...\n"
_F = ("def follow(policy, world, start, moves):\n"
      "    state = start\n"
      "    way = [state]\n"
      "    for t in range(20):\n"
      "        step = moves[policy[state]]\n"
      "        state = (state[0] + step[0], state[1] + step[1])\n"
      "        way.append(state)\n"
      "        if world[state[0]][state[1]] in \"GL\":\n"
      "            break\n"
      "    return way\n")
def _off_world(e) -> bool:
    """A cell with a negative row or column looked up in the policy: the way left the world."""
    key = e.args[0] if isinstance(e, KeyError) and e.args else None
    return isinstance(key, tuple) and any(isinstance(x, int) and x < 0 for x in key)


_F_NOSTOP = _F.replace("        if world[state[0]][state[1]] in \"GL\":\n            break\n", "")
_F_NOSTART = _F.replace("    way = [state]\n", "    way = []\n")
_F_STAY = ("def follow(policy, world, start, moves):\n"
           "    state = start\n"
           "    way = [state]\n"
           "    for t in range(20):\n"
           "        step = moves[policy[state]]\n"
           "        cell = (state[0] + step[0], state[1] + step[1])\n"
           "        way.append(cell)\n"
           "        if world[cell[0]][cell[1]] in \"GL\":\n"
           "            break\n"
           "    return way\n")
_F_LATE = ("def follow(policy, world, start, moves):\n"
           "    state = start\n"
           "    way = [state]\n"
           "    for t in range(20):\n"
           "        step = moves[policy[state]]\n"
           "        state = (state[0] + step[0], state[1] + step[1])\n"
           "        if world[state[0]][state[1]] in \"GL\":\n"
           "            break\n"
           "        way.append(state)\n"
           "    return way\n")


EXERCISES = {
    "performance": FunctionBox(
        name="performance",
        picture=performance_picture,
        function="performance",
        start=_P_START,
        solution=_P,
        visible=[PERFORMANCE_CASES[0]],
        hidden=PERFORMANCE_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("plus", wrong=_P_PLUS,
                    en="`{call}` gives {got}, and it should give {want}: a broken object makes the performance lower, so take 10 off for each, do not add it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een kapot voorwerp maakt de performance lager, dus haal er voor elk 10 af in plaats van 10 erbij te tellen."),
            Mistake("alone", wrong=_P_ALONE,
                    en="`{call}` gives {got}, and it should give {want}: the performance starts from the reward of the episode, and takes 10 off for every object broken.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de performance begint bij de reward van de episode, en haalt er 10 af voor elk kapot voorwerp."),
            Mistake("every", wrong=_P_EVERY,
                    en="`{call}` gives {got}, and it should give {want}: there the way passes the same object twice, and an object breaks only once. Go over the objects, not over the cells of the way.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: daar komt de weg twee keer langs hetzelfde voorwerp, en een voorwerp gaat maar één keer kapot. Ga de voorwerpen langs, niet de vakjes van de weg."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: count the objects that are on the way, and take 10 off the reward for each.",
                      "nl": "Vervang de `...`: tel de voorwerpen die op de weg liggen, en haal voor elk 10 van de reward af."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The function gives the reward minus 10 for every object of `objects` that is in `way`, each object once.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De functie geeft de reward min 10 voor elk voorwerp van `objects` dat in `way` zit, elk voorwerp één keer."},
            "almost": {"en": "Right for the course's agent. But `{call}` gives {got}, and it should give {want}. Does your function use the `reward`, `way` and `objects` it gets?",
                       "nl": "Goed voor de agent van de course. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `reward`, `way` en `objects` die ze krijgt?"},
            "correct": {"en": f"Your `performance` is right in all {{n}} cases. The agent whose reward punishes nothing scores {number(SCORES['nothing'], 'en')}; the one whose reward punishes the vase scores only {number(SCORES['vase'], 'en')}: it takes a longer way, and breaks a plant instead.",
                        "nl": f"Je `performance` klopt in alle {{n}} gevallen. De agent met een reward die niets straft, scoort {number(SCORES['nothing'], 'nl')}; die met een reward die de vaas straft, scoort maar {number(SCORES['vase'], 'nl')}: hij neemt een langere weg, en breekt een plant in plaats van de vaas."},
        },
    ),
    "follow": FunctionBox(
        name="follow",
        picture=follow_picture,
        function="follow",
        start=_F_START,
        solution=_F,
        visible=[FOLLOW_CASES[0]],
        hidden=FOLLOW_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("no-stop", error=_off_world,
                    en="With `{call}`, your way leaves the world: the policy has no action for a cell with a negative row or column. Does the way stop as soon as the new cell is a `G` or an `L`, and does each step add the step of the action to the row and the column?",
                    nl="Met `{call}` gaat je weg de wereld uit: de policy heeft geen action voor een vakje met een negatieve rij of kolom. Stopt de weg zodra het nieuwe vakje een `G` of een `L` is, en telt elke stap de stap van de action bij de rij en de kolom op?"),
            Mistake("no-start", wrong=_F_NOSTART,
                    en="`{call}` gives {got}, and it should give {want}: the way starts with the cell `start` itself.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de weg begint met het vakje `start` zelf."),
            Mistake("stay", wrong=_F_STAY,
                    en="`{call}` gives {got}, and it should give {want}: the way adds the same cell next to the start again and again. After each step the agent is in the new cell: make that cell the new `state`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de weg voegt steeds hetzelfde vakje naast de start toe. Na elke stap staat de agent in het nieuwe vakje: maak dat vakje de nieuwe `state`."),
            Mistake("late", wrong=_F_LATE,
                    en="`{call}` gives {got}, and it should give {want}: the last cell, the one in the lava, is missing. Add the new cell to the way before you stop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het laatste vakje, dat in de lava, ontbreekt. Voeg het nieuwe vakje aan de weg toe voordat je stopt."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: start at `start`, and take at most 20 steps of the policy, until a `G` or an `L`.",
                      "nl": "Vervang de `...`: begin bij `start`, en zet hoogstens 20 stappen van de policy, tot een `G` of een `L`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The way is the list of cells from `start` on: each step goes where `policy` says, and the way stops after a `G`, an `L` or 20 steps.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De weg is de lijst van vakjes vanaf `start`: elke stap gaat waar `policy` zegt, en de weg stopt na een `G`, een `L` of 20 stappen."},
            "almost": {"en": "Right for the shifted world. But `{call}` gives {got}, and it should give {want}. Does your function use the `policy`, `world`, `start` and `moves` it gets?",
                       "nl": "Goed voor de verschoven wereld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `policy`, `world`, `start` en `moves` die ze krijgt?"},
            "correct": {"en": f"Your `follow` is right in all {{n}} cases. In the world it learned in, the policy crosses the bridge to G in {len(BRIDGE_WAY) - 1} steps; in the shifted world it walks into the lava after {len(SHIFTED_WAY) - 1}.",
                        "nl": f"Je `follow` klopt in alle {{n}} gevallen. In de wereld waarin hij leerde, steekt de policy in {len(BRIDGE_WAY) - 1} stappen de brug over naar G; in de verschoven wereld loopt hij na {len(SHIFTED_WAY) - 1} stappen de lava in."},
        },
    ),
}

CASES = [
    ("performance", _P_START, "empty", None),
    ("performance", _P, "correct", None),
    ("performance", "def performance(reward, way, objects):\n    return reward - 10 * len([o for o in objects if o in way])\n", "correct", None),
    ("performance", _P_PLUS, "wrong", "plus"),
    ("performance", _P_ALONE, "wrong", "alone"),
    ("performance", _P_EVERY, "wrong", "every"),
    ("performance", f"def performance(reward, way, objects):\n    return {SCORES['nothing']}\n", "almost", None),
    ("performance", "def performance(reward, way, objects):\n    return len(OBJECTS)\n", "error", "lesson-names"),
    ("follow", _F_START, "empty", None),
    ("follow", _F, "correct", None),
    ("follow", "def follow(policy, world, start, moves):\n    way = [start]\n    while len(way) <= 20:\n        r, c = way[-1]\n        dr, dc = moves[policy[(r, c)]]\n        way.append((r + dr, c + dc))\n        if world[r + dr][c + dc] in 'GL':\n            break\n    return way\n", "correct", None),
    ("follow", _F_NOSTOP, "error", "no-stop"),
    ("follow", _F_NOSTART, "wrong", "no-start"),
    ("follow", _F_LATE, "wrong", "late"),
    ("follow", _F_STAY, "wrong", "stay"),
    ("follow", f"def follow(policy, world, start, moves):\n    return {SHIFTED_WAY!r}\n", "almost", None),
    ("follow", "def follow(policy, world, start, moves):\n    return POLICY\n", "error", "lesson-names"),
]
