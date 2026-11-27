# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rl-behavior-cloning: an expert crosses a bridge three cells wide between two rows of pits, where
a step can slip to the side, and never falls. A clone learns from the expert's demonstrations with k-nearest neighbors:
the cell is the example, the expert's action its class. Where the expert stood, the clone does what the expert did;
where a slip takes it to a cell the expert seldom stood on, it copies the nearest cell the expert did stand on, walks
on along the pits, and falls. The bridge, the expert from value iteration, the course's demonstrate, clone and
mistakes, the functions the learner writes, the clones' runs, and the pictures.

The slips and the rewards follow the lesson on value iteration; behavior cloning and its compounding errors as in Ross
and Bagnell (Efficient Reductions for Imitation Learning, AISTATS 2010). The bridge, the code, the runs and the pictures
are the course's own."""
import html
import math

import numpy as np

from everything_ai import chart, live, yours
from everything_ai.check import FunctionBox, Mistake, kept, with_time_limit

LESSON = "rl-behavior-cloning"
ACTIONS = ["up", "down", "left", "right"]
MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
SIDES = {"up": ["left", "right"], "down": ["left", "right"], "left": ["up", "down"], "right": ["up", "down"]}
SLIP = 0.2        # the chance that a step slips: 0.1 to each side of the action, as in value iteration
GAMMA = 0.9       # the expert's discount, as in value iteration
LIMIT = 100       # an episode of a clone stops after this many steps, also when it has not reached G or a pit
NS = [1, 2, 5, 10, 20, 50]   # the numbers of demonstrations the demo goes through
SEEDS = 20        # demonstrations with seed 0 to 19 for every number
PLAYS = 500       # the episodes each clone plays; the episodes of seed s play with seed 100 + s
K = 1             # the k of the clones, unless a text says otherwise

# A bridge three cells wide, with a row of pits on each side. S and G are in the middle row.
BRIDGE = [
    "XXXXXXXXXXXX",
    "............",
    "S..........G",
    "............",
    "XXXXXXXXXXXX",
]
ROWS, COLS = len(BRIDGE), len(BRIDGE[0])
START, GOAL = (2, 0), (2, COLS - 1)
CELLS = [(r, c) for r in range(ROWS) for c in range(COLS) if BRIDGE[r][c] in ".S"]   # where the agent can stand


class Named(dict):
    """A dict that a check message calls by its name, such as EXPERT; the learner's code gets a plain copy."""

    def __init__(self, entries, name: str):
        super().__init__(entries)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return dict(self)


class NamedList(list):
    """A list that a check message calls by how it was made, such as `demonstrate(EXPERT, 1, 0)`."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return list(self)


# --- the bridge and the expert --------------------------------------------------------------------------------------

def land(state, move):
    """The cell one step from `state` in the direction of `move`; at the edge of the world the agent stays put."""
    r, c = state[0] + MOVES[move][0], state[1] + MOVES[move][1]
    return (r, c) if 0 <= r < ROWS and 0 <= c < COLS else state


def step(state, action, rng):
    """One step on the bridge: the action goes where it says with chance 0.8, and slips to each side with 0.1. Gives
    back the new cell, the reward (+1 on G, −1 in a pit, 0 elsewhere) and whether the episode is over."""
    u = rng.random()
    move = action if u < 1 - SLIP else SIDES[action][0] if u < 1 - SLIP / 2 else SIDES[action][1]
    nxt = land(state, move)
    sign = BRIDGE[nxt[0]][nxt[1]]
    return nxt, 1 if sign == "G" else -1 if sign == "X" else 0, sign in "GX"


def _table() -> dict:
    """The bridge as the table T of value iteration: T[state][action] lists (probability, next state, reward)."""
    T = {}
    for r in range(ROWS):
        for c in range(COLS):
            state, sign = (r, c), BRIDGE[r][c]
            T[state] = {}
            for a in ACTIONS:
                if sign in "GX":
                    T[state][a] = [(1.0, state, 0)]
                    continue
                outs = []
                for p, move in [(1 - SLIP, a), (SLIP / 2, SIDES[a][0]), (SLIP / 2, SIDES[a][1])]:
                    nxt = land(state, move)
                    there = BRIDGE[nxt[0]][nxt[1]]
                    outs.append((p, nxt, 1 if there == "G" else -1 if there == "X" else 0))
                T[state][a] = outs
    return T


def value_iteration(rounds: int = 100) -> tuple:
    """Value iteration on the bridge, as in the lesson on value iteration: the values, and the policy, the first action
    with the largest Q in each cell the agent can stand on."""
    T = _table()
    V = {s: 0.0 for s in T}
    for _ in range(rounds):
        Q = {s: {a: sum(p * (r + GAMMA * V[n]) for p, n, r in T[s][a]) for a in ACTIONS} for s in T}
        V = {s: max(Q[s].values()) for s in T}
    return V, {s: ACTIONS[int(np.argmax([Q[s][a] for a in ACTIONS]))] for s in CELLS}


V_STAR, _PI = value_iteration()
EXPERT = Named(_PI, "EXPERT")
# An expert that only ever goes right: a stand-in that shows whether a function uses the expert it gets.
STRAIGHT = Named({s: "right" for s in CELLS}, "STRAIGHT")


def knn_predict(X_train, y_train, x, k):
    """The k-NN of the lesson on k-nearest neighbors: the k training examples nearest to x vote."""
    dists = np.linalg.norm(X_train - x, axis=1)
    nearest = np.argsort(dists, kind="stable")[:k]
    names, counts = np.unique(y_train[nearest], return_counts=True)
    return str(names[np.argmax(counts)])


# --- the course's solutions --------------------------------------------------------------------------------------------

def demonstrate(expert, episodes, seed):
    """The course's demonstrate, as the lesson writes it: every cell the expert stands on, with the expert's action."""
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(episodes):
        state = START
        over = False
        while not over:
            action = expert[state]
            pairs.append((state, action))
            state, reward, over = step(state, action, rng)
    return pairs


def clone(pairs, k):
    """The course's clone: k-NN on the cells of the demonstrations, asked for every cell of the bridge."""
    X = []
    y = []
    for state, action in pairs:
        X.append(state)
        y.append(action)
    X = np.array(X)
    y = np.array(y)
    policy = {}
    for state in CELLS:
        policy[state] = knn_predict(X, y, state, k)
    return policy


def mistakes(policy, expert, states):
    """The course's mistakes: the share of the states where the policy does something other than the expert."""
    wrong = 0
    for state in states:
        if policy[state] != expert[state]:
            wrong = wrong + 1
    return wrong / len(states)


def play(policy, episodes, seed) -> list:
    """Episodes of a policy on the bridge, each up to LIMIT steps: for each, how it ended ("G", "X" for a pit, or
    "limit") and the cells it stood on, in order, with the cell it fell from or reached G from last."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(episodes):
        state, cells, end = START, [], "limit"
        for _ in range(LIMIT):
            cells.append(state)
            state, reward, over = step(state, policy[state], rng)
            if over:
                end = "G" if reward > 0 else "X"
                break
        out.append((end, cells))
    return out


def visits(policy, episodes, seed) -> list:
    """Every cell a policy stands on in its episodes, as often as it stands there."""
    return [s for _, cells in play(policy, episodes, seed) for s in cells]


def outcome(policy, seed: int, episodes: int = PLAYS) -> dict:
    """How the episodes of a policy end, as shares, and the cells it stood on."""
    games = play(policy, episodes, seed)
    ends = [e for e, _ in games]
    return {"falls": ends.count("X") / episodes, "goal": ends.count("G") / episodes,
            "limit": ends.count("limit") / episodes, "visits": [s for _, cells in games for s in cells],
            "games": games}


def _measure(n: int, seed: int, k: int = K) -> dict:
    pairs = demonstrate(EXPERT, n, seed)
    policy = clone(pairs, k)
    out = outcome(policy, 100 + seed)
    return {"pairs": pairs, "policy": policy, "falls": out["falls"], "goal": out["goal"], "limit": out["limit"],
            "on_demo": mistakes(policy, EXPERT, [s for s, _ in pairs]),
            "on_own": mistakes(policy, EXPERT, out["visits"]),
            "cells_wrong": sum(policy[s] != EXPERT[s] for s in CELLS),
            "demo_cells": len({s for s, _ in pairs}), "games": out["games"][:5]}


def course_runs(k: int = K) -> dict:
    """For every number of demonstrations in NS and every seed: the clone, and how it does. Kept: the lesson's texts,
    the demo and the NumPy cell all use it."""
    return kept(f"{LESSON}:runs:{k}", lambda: {n: [_measure(n, s, k) for s in range(SEEDS)] for n in NS})


def mean_of(runs: list, key: str) -> float:
    return float(np.mean([r[key] for r in runs]))


def expert_outcome() -> dict:
    return kept(f"{LESSON}:expert", lambda: {k: v for k, v in outcome(EXPERT, 100, 2000).items() if k != "games"})


def dagger(own: int = 5) -> dict:
    """One round of DAgger for every seed: the clone of one demonstration plays `own` episodes (seed 500 + s), the
    expert names its action in every cell the clone stood on, and the clone learns again from both. The mean falls
    and the mean number of pairs, beside those of 5 demonstrations."""
    def make():
        falls, sizes = [], []
        for s in range(SEEDS):
            pairs = demonstrate(EXPERT, 1, s)
            seen_ = visits(clone(pairs, K), own, 500 + s)
            more = pairs + [(c, EXPERT[c]) for c in seen_]
            falls.append(outcome(clone(more, K), 100 + s)["falls"])
            sizes.append(len(more))
        return {"falls": float(np.mean(falls)), "pairs": float(np.mean(sizes))}
    return kept(f"{LESSON}:dagger:{own}", make)


def next_mistake() -> dict:
    """Over every episode of the 20 clones of one demonstration: how often a step is a mistake (the clone does something
    other than the expert), how often the step after a mistake is one too, and how many falls come right after one."""
    def make():
        steps = wrong = after = after_wrong = falls = falls_after = 0
        for s, run in enumerate(course_runs()[1]):
            policy = run["policy"]
            for end, cells in play(policy, PLAYS, 100 + s):
                m = [policy[c] != EXPERT[c] for c in cells]
                steps, wrong = steps + len(m), wrong + sum(m)
                for a, b in zip(m, m[1:]):
                    if a:
                        after, after_wrong = after + 1, after_wrong + b
                if end == "X":
                    falls, falls_after = falls + 1, falls_after + m[-1]
        return {"share": wrong / steps, "after": after_wrong / after, "falls": falls, "falls_after": falls_after}
    return kept(f"{LESSON}:next", make)


# --- the words and the parts of the pictures ---------------------------------------------------------------------------

LANGS = ("en", "nl")
_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem"
_HALO = "paint-order:stroke;stroke:var(--ea-page);stroke-width:3px;stroke-linejoin:round"
_TOWARD = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
_GLYPH = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
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


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch, and a real minus sign."""
    out = f"{x:,.{digits}f}"
    if out.startswith("-") and float(out.replace(",", "")) == 0:
        out = out[1:]
    out = out.replace("-", "−")
    return out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out


def percent(x: float, lang: str) -> str:
    """A share as a percentage with one decimal: 43.8%, in Dutch 43,8%."""
    return number(100 * x, 1, lang) + "%"


def bridge_svg(panels: list, alt: str, lang: str) -> str:
    """One or more copies of the bridge, one under the other. Each panel is a dict: `title`; `arrows` {cell: action},
    drawn as small triangles, in ink for the expert or in orange (`learned`: True) for a clone, what it learned;
    `shade`, cells the expert stood on, washed in the color of data; `wrong`, cells where a clone does something else
    than the expert, washed in the color that counts mistakes; `walk`, cells in order, drawn as an ink line; and
    `fall`, (cell, pit) for a fall at the end of the walk, drawn as a line to the edge of the pit with a dot. Pits are
    filled with their reward."""
    cell = CELL
    titles = [len(p["title"]) * 7.6 + 4 for p in panels if p.get("title")]
    W = max(COLS * cell + 2, min(314, int(max(titles, default=0))))
    parts, y0 = [], 0
    small = 10.5
    for p in panels:
        if p.get("title"):
            parts.append(f'<text x="1" y="{y0 + 14}" style="font-size:13px;font-weight:600;fill:var(--ea-ink)">'
                         f'{html.escape(p["title"])}</text>')
            top = y0 + 22
        else:
            top = y0 + 1
        shade, wrong = set(p.get("shade") or ()), set(p.get("wrong") or ())
        for r, row in enumerate(BRIDGE):
            for c, sign in enumerate(row):
                x, y = 1 + c * cell, top + r * cell
                fill = ("var(--ea-data)" if sign == "X" else "var(--ea-loss-wash)" if (r, c) in wrong
                        else "var(--ea-data-wash)" if (r, c) in shade else "var(--ea-page)")
                parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" style="fill:{fill};'
                             f'stroke:var(--ea-line);stroke-width:1"/>')
                if (r, c) in wrong:
                    parts.append(f'<rect x="{x + 1.5}" y="{y + 1.5}" width="{cell - 3}" height="{cell - 3}" '
                                 'style="fill:none;stroke:var(--ea-loss);stroke-width:1.5"/>')
                if sign == "X":
                    parts.append(f'<text x="{x + cell / 2:.1f}" y="{y + cell / 2 + small * 0.36:.1f}" style="font-size:{small}px;'
                                 'font-weight:600;fill:var(--ea-page);text-anchor:middle">−1</text>')
                elif sign in "SG":
                    parts.append(f'<text x="{x + cell / 2:.1f}" y="{y + cell / 2 + 4.5:.1f}" style="font-size:13px;'
                                 f'font-weight:700;fill:var(--ea-ink-2);text-anchor:middle;{_HALO}">{sign}</text>')
        walk = p.get("walk") or []
        if len(walk) > 1:
            parts.append(_polyline(walk, top, cell, "var(--ea-ink)", 2.2, 0.55))
        if p.get("fall"):
            a, b = p["fall"]
            ax, ay = 1 + a[1] * cell + cell / 2, top + a[0] * cell + cell / 2
            ex, ey = ax + (b[1] - a[1]) * cell * 0.56, ay + (b[0] - a[0]) * cell * 0.56
            parts.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" style="stroke:var(--ea-ink);'
                         'stroke-width:2.2;stroke-linecap:round;opacity:0.55"/>')
            parts.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="3.4" style="fill:var(--ea-ink);stroke:var(--ea-page);stroke-width:1.4"/>')
        colour = "var(--ea-param)" if p.get("learned") else "var(--ea-ink-2)"
        for (r, c), a in sorted((p.get("arrows") or {}).items()):
            if not (0 <= r < ROWS and 0 <= c < COLS) or BRIDGE[r][c] in "SGX" or a not in _TOWARD:
                continue
            mx, my = 1 + c * cell + cell / 2, top + r * cell + cell / 2
            dx, dy = _TOWARD[a]
            reach = cell / 2 - 3
            tip = (mx + dx * reach, my + dy * reach)
            base = (mx + dx * (reach - cell * 0.3), my + dy * (reach - cell * 0.3))
            side = (dy * cell * 0.16, dx * cell * 0.16)
            corners = [tip, (base[0] + side[0], base[1] + side[1]), (base[0] - side[0], base[1] - side[1])]
            parts.append('<polygon points="' + " ".join(f"{px:.1f},{py:.1f}" for px, py in corners)
                         + f'" style="fill:{colour}"/>')
        y0 = top + ROWS * cell + 12
    H = y0 - 10
    svg = (f'<svg viewBox="0 0 {W} {H}" aria-hidden="true" style="flex:1 1 0;width:100%;min-width:min(100%,{W}px);'
           f'max-width:{W}px;height:auto;font-family:var(--ea-font-ui)">{"".join(parts)}</svg>')
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(alt)}" style="margin:0.25rem 0 0.5rem;'
            f'max-width:min(100%,{W}px);display:flex">{svg}</figure>')


def _polyline(cells, top: float, cell: int, colour: str, width: float, opacity: float) -> str:
    points = " ".join(f"{1 + c * cell + cell / 2:.1f},{top + r * cell + cell / 2:.1f}" for r, c in cells)
    return (f'<polyline points="{points}" style="fill:none;stroke:{colour};stroke-width:{width};'
            f'stroke-linecap:round;stroke-linejoin:round;opacity:{opacity}"/>')


_KEY_TEXT = {
    "en": "`S` start, `G` goal (+1); a cell marked −1: a pit, where the episode ends. Ink arrows: the expert; orange arrows: a clone. Shaded: a cell the expert stood on; outlined: a cell where the clone does something other than the expert.",
    "nl": "`S` start, `G` doel (+1); een vakje met −1: een put, waar de episode eindigt. Pijlen in inkt: de expert; oranje pijlen: een kloon. Licht ingekleurd: een vakje waar de expert stond; omlijnd: een vakje waar de kloon iets anders doet dan de expert.",
}


def bridge_key(lang: str) -> str:
    return _note(_KEY_TEXT[lang if lang in LANGS else "en"])


def _fall(cells: list, end: str) -> tuple | None:
    """The fall at the end of an episode: the last cell, and the pit beside it in the row of pits it is next to."""
    if end != "X" or not cells:
        return None
    r, c = cells[-1]
    return (r, c), (0 if r == 1 else ROWS - 1, c)


_PIC_TEXT = {
    "en": {"expert": "The expert", "expert_alt": "The bridge, three cells wide between two rows of pits, with the expert's action in every cell: right in the middle row, down in the row above it, up in the row below it.",
           "demo": "The expert's demonstration, seed 0", "clone": "The clone of that demonstration",
           "idea_alt": "Two copies of the bridge. Above, the cells of one demonstration of the expert, shaded, with its walk and its actions. Below, the clone that k-NN learns from it: an action in every cell, the cells where it differs from the expert outlined, and one episode of the clone that ends in a pit."},
    "nl": {"expert": "De expert", "expert_alt": "De brug, drie vakjes breed tussen twee rijen putten, met de action van de expert in elk vakje: rechts in de middelste rij, omlaag in de rij erboven, omhoog in de rij eronder.",
           "demo": "De demonstratie van de expert, seed 0", "clone": "De kloon van deze demonstratie",
           "idea_alt": "Twee keer de brug. Boven de vakjes van één demonstratie van de expert, licht ingekleurd, met de wandeling en de actions. Onder de kloon die k-NN eruit leert: een action in elk vakje, de vakjes waar de kloon van de expert afwijkt omlijnd, en één episode van de kloon die in een put eindigt."},
}


def world_picture(lang: str) -> str:
    """The bridge with the expert's actions, beside the guess."""
    lang = lang if lang in LANGS else "en"
    w = _PIC_TEXT[lang]
    return bridge_svg([{"title": w["expert"], "arrows": dict(EXPERT)}], w["expert_alt"], lang) + bridge_key(lang)


def idea_picture(lang: str) -> str:
    """Seed 0: the expert's one demonstration, and the clone k-NN learns from it, with its first episode."""
    lang = lang if lang in LANGS else "en"
    w = _PIC_TEXT[lang]
    run = course_runs()[1][0]
    pairs, policy = run["pairs"], run["policy"]
    end, cells = run["games"][0]
    wrong = [s for s in CELLS if policy[s] != EXPERT[s]]
    panels = [{"title": w["demo"], "arrows": {s: a for s, a in pairs}, "shade": [s for s, _ in pairs],
               "walk": [s for s, _ in pairs] + [GOAL]},
              {"title": w["clone"], "arrows": policy, "learned": True, "wrong": wrong, "walk": cells,
               "fall": _fall(cells, end)}]
    return bridge_svg(panels, w["idea_alt"], lang) + bridge_key(lang)


# --- the demo: from 1 to 50 demonstrations --------------------------------------------------------------------------

DEMO_TEXT = {
    "en": {"title": "Falls of the clones", "x": "demonstrations", "y": "falls in 100 episodes",
           "alt": "The share of episodes in which the clones fall, over 20 seeds, for 1, 2, 5, 10, 20 and 50 demonstrations, with a line at the number shown; the expert falls in none.",
           "panel": "The clone of seed 0, {n} {word}",
           "panel_alt": "The bridge with the clone that k-NN learns from {n} {word} of the expert with seed 0: its action in every cell, the cells the expert stood on shaded, and the cells where the clone differs from the expert outlined.",
           "one": "demonstration", "many": "demonstrations", "expert": "expert",
           "say": "With {n} {word}: over the 20 seeds, the clones fall in {falls} of their episodes, and on the cells they stand on they do something other than the expert in {own} of their steps, against {demo} on the cells of the demonstrations. The clone of seed 0 differs from the expert in {cells} of the 35 cells. The expert falls in none of its 2,000 episodes."},
    "nl": {"title": "Vallen van de klonen", "x": "demonstraties", "y": "vallen per 100 episodes",
           "alt": "Het deel van de episodes waarin de klonen vallen, over 20 seeds, bij 1, 2, 5, 10, 20 en 50 demonstraties, met een lijn bij het getoonde aantal; de expert valt nooit.",
           "panel": "De kloon van seed 0, {n} {word}",
           "panel_alt": "De brug met de kloon die k-NN leert uit {n} {word} van de expert met seed 0: de action in elk vakje, de vakjes waar de expert stond licht ingekleurd, en de vakjes waar de kloon van de expert afwijkt omlijnd.",
           "one": "demonstratie", "many": "demonstraties", "expert": "expert",
           "say": "Met {n} {word}: over de 20 seeds vallen de klonen in {falls} van de episodes, en op de vakjes waar de klonen zelf staan, doen de klonen bij {own} van de stappen iets anders dan de expert, tegen {demo} op de vakjes van de demonstraties. De kloon van seed 0 wijkt in {cells} van de 35 vakjes af van de expert. De expert valt in geen van de 2.000 episodes."},
}


def _demo_chart(n: int, lang: str) -> str:
    w = DEMO_TEXT[lang]
    runs = course_runs()
    falls = [100 * mean_of(runs[m], "falls") for m in NS]
    p = chart.Panel((1, 50), (0, 60), title=w["title"], xlabel=w["x"], ylabel=w["y"], xlog=True,
                    xticks=NS, xtick_labels=[str(m) for m in NS], yticks=[0, 20, 40, 60])
    p.line([1, 50], [0, 0], "ink", 1.4, dashed=True)
    p.text(1.12, 4, w["expert"], "ink", anchor="start")
    p.line([n, n], [0, 60], "axis", 1.2)
    p.line(NS, falls, "loss", 2.2)
    p.points(NS, falls, "loss", 3.5)
    out = chart.figure([p], alt=w["alt"], width=330, height=240)
    return chart.with_decimal_comma(out) if lang == "nl" else out


def demo_frame(n: int, lang: str) -> tuple:
    """A stop of the demo: the chart of the falls with a line at n, the clone of seed 0 from n demonstrations, and a
    sentence."""
    lang = lang if lang in LANGS else "en"
    w = DEMO_TEXT[lang]
    n = min(NS, key=lambda m: abs(m - int(n)))
    runs = course_runs()[n]
    word = w["one"] if n == 1 else w["many"]
    run = runs[0]
    wrong = [s for s in CELLS if run["policy"][s] != EXPERT[s]]
    grid = bridge_svg([{"title": w["panel"].format(n=n, word=word), "arrows": run["policy"], "learned": True,
                        "shade": [s for s, _ in run["pairs"]], "wrong": wrong}],
                      w["panel_alt"].format(n=n, word=word), lang)
    say = w["say"].format(n=n, word=word, falls=percent(mean_of(runs, "falls"), lang), own=percent(mean_of(runs, "on_own"), lang),
                          demo=percent(mean_of(runs, "on_demo"), lang), cells=run["cells_wrong"])
    return live.rewound(_demo_chart(n, lang), [grid], width=330), say


# --- the pictures of what the learner writes ----------------------------------------------------------------------------

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


def _pairs_ok(pairs) -> bool:
    """A list of pairs of a cell of the world (G and the pits too) and an action; a pair may also be a list."""
    grid = {(r, c) for r in range(ROWS) for c in range(COLS)}
    try:
        return (isinstance(pairs, list) and len(pairs) > 0 and all(
            isinstance(p, (tuple, list)) and len(p) == 2 and isinstance(p[0], (tuple, list, np.ndarray))
            and tuple(int(v) for v in p[0]) in grid and str(p[1]) in ACTIONS for p in pairs))
    except Exception:   # noqa: BLE001   odd contents: no picture
        return False


def _policy_ok(policy) -> bool:
    return isinstance(policy, dict) and all(s in policy and str(policy[s]) in ACTIONS for s in CELLS)


_YOURS_TEXT = {
    "en": {"demo_title": "Your demonstrations", "demo_one": "`demonstrate(EXPERT, 1, 0)`", "demo_five": "`demonstrate(EXPERT, 5, 0)`",
           "demo_head": "What your `demonstrate` gives for the expert on the bridge, with seed 0: the cells of the pairs shaded, with the expert's action in each, and the walk of the first episode.",
           "demo_say": "One episode: {n} pairs on {c} of the 35 cells. Five episodes: {n5} pairs on {c5} cells; the expert never stood on the other {rest}.",
           "demo_none": "Your `demonstrate` stops with an error, takes too long, or gives something other than a list of pairs of a cell and an action, so the picture is not drawn yet.",
           "demo_alt": "The bridge twice, with the cells of your demonstrations of one and of five episodes shaded.",
           "panel_one": "1 episode", "panel_five": "5 episodes",
           "clone_title": "Your clone",
           "clone_head": "Your `clone` of `demonstrate(EXPERT, 1, 0)` with k = 1: its action in every cell, the cells of the demonstration shaded, and the cells where it does something other than the expert outlined.",
           "clone_say": "It differs from the expert in {w} of the 35 cells, {d} of them cells the expert stood on. In {plays} episodes it falls in {falls} of them.",
           "clone_none": "Your `clone` stops with an error, takes too long, or gives something other than an action for every cell of `CELLS`, so the picture is not drawn yet.",
           "clone_alt": "The bridge with the actions of your clone in every cell.",
           "clone_panel": "Your clone, k = 1",
           "m_title": "Mistakes of the clone",
           "m_head": "Your `mistakes` for the clone of `demonstrate(EXPERT, 1, 0)`, on three lists of states:",
           "m_rows": ["the cells of the demonstration", "all 35 cells", "the cells the clone stands on in 20 episodes"],
           "m_none": "Your `mistakes` stops with an error, takes too long, or gives something other than a number, so the picture is not drawn yet.",
           "m_alt": "Three bars: the share of mistakes of the clone on the cells of the demonstration, on all cells, and on the cells it stands on itself."},
    "nl": {"demo_title": "Je demonstraties", "demo_one": "`demonstrate(EXPERT, 1, 0)`", "demo_five": "`demonstrate(EXPERT, 5, 0)`",
           "demo_head": "Wat je `demonstrate` geeft voor de expert op de brug, met seed 0: de vakjes van de paren licht ingekleurd, met de action van de expert in elk vakje, en de wandeling van de eerste episode.",
           "demo_say": "Eén episode: {n} paren op {c} van de 35 vakjes. Vijf episodes: {n5} paren op {c5} vakjes; op de andere {rest} stond de expert nooit.",
           "demo_none": "Je `demonstrate` stopt met een fout, duurt te lang, of geeft iets anders dan een lijst met paren van een vakje en een action, dus er is nog geen afbeelding.",
           "demo_alt": "Twee keer de brug, met de vakjes van je demonstraties van één en van vijf episodes licht ingekleurd.",
           "panel_one": "1 episode", "panel_five": "5 episodes",
           "clone_title": "Je kloon",
           "clone_head": "Je `clone` van `demonstrate(EXPERT, 1, 0)` met k = 1: de action in elk vakje, de vakjes van de demonstratie licht ingekleurd, en de vakjes waar je kloon iets anders doet dan de expert omlijnd.",
           "clone_say": "Je kloon wijkt in {w} van de 35 vakjes af van de expert, en {d} van deze vakjes zijn vakjes waar de expert stond. In {plays} episodes valt je kloon in {falls} van de episodes.",
           "clone_none": "Je `clone` stopt met een fout, duurt te lang, of geeft niet voor elk vakje van `CELLS` een action, dus er is nog geen afbeelding.",
           "clone_alt": "De brug met de actions van je kloon in elk vakje.",
           "clone_panel": "Je kloon, k = 1",
           "m_title": "Fouten van de kloon",
           "m_head": "Je `mistakes` voor de kloon van `demonstrate(EXPERT, 1, 0)`, op drie lijsten met states:",
           "m_rows": ["de vakjes van de demonstratie", "alle 35 vakjes", "de vakjes waar de kloon staat in 20 episodes"],
           "m_none": "Je `mistakes` stopt met een fout, duurt te lang, of geeft iets anders dan een getal, dus er is nog geen afbeelding.",
           "m_alt": "Drie balken: het deel van de fouten van de kloon op de vakjes van de demonstratie, op alle vakjes, en op de vakjes waar de kloon zelf staat."},
}


def demonstrate_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]
    run_ = seen.mine
    fn = run_.names.get("demonstrate") if run_ is not None else None
    head = _note(w["demo_head"])
    got = _call(lambda: (fn(dict(EXPERT), 1, 0), fn(dict(EXPERT), 5, 0))) if callable(fn) else None
    if not (isinstance(got, tuple) and _pairs_ok(got[0]) and _pairs_ok(got[1])):
        return yours.figure(w["demo_title"], head + _note(w["demo_none"], _LOSS_UI), alt=w["demo_title"])
    one, five = ([(tuple(int(v) for v in s), str(a)) for s, a in pairs] for pairs in got)
    walk = [s for s, _ in one]
    panels = [{"title": w["panel_one"], "arrows": dict(one), "shade": walk, "walk": walk},
              {"title": w["panel_five"], "arrows": dict(five), "shade": [s for s, _ in five]}]
    c5 = len({s for s, _ in five})
    say = w["demo_say"].format(n=len(one), c=len(set(walk)), n5=len(five), c5=c5, rest=len(CELLS) - c5)
    return yours.figure(w["demo_title"], head + bridge_svg(panels, w["demo_alt"], lang) + _note(say), alt=w["demo_title"])


PAIRS = NamedList(demonstrate(EXPERT, 1, 0), "demonstrate(EXPERT, 1, 0)")


def clone_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]
    run_ = seen.mine
    fn = run_.names.get("clone") if run_ is not None else None
    head = _note(w["clone_head"])
    got = _call(lambda: fn(list(PAIRS), 1)) if callable(fn) else None
    if not _policy_ok(got):
        return yours.figure(w["clone_title"], head + _note(w["clone_none"], _LOSS_UI), alt=w["clone_title"])
    policy = {s: str(got[s]) for s in CELLS}
    demo = {s for s, _ in PAIRS}
    wrong = [s for s in CELLS if policy[s] != EXPERT[s]]
    out = outcome(policy, 100)
    panel = {"title": w["clone_panel"], "arrows": policy, "learned": True, "shade": demo, "wrong": wrong}
    say = w["clone_say"].format(w=len(wrong), d=len(demo & set(wrong)), plays=PLAYS, falls=percent(out["falls"], lang))
    return yours.figure(w["clone_title"], head + bridge_svg([panel], w["clone_alt"], lang) + _note(say), alt=w["clone_title"])


CLONE = Named(clone(list(PAIRS), 1), "clone(demonstrate(EXPERT, 1, 0), 1)")
ALL_CELLS = NamedList(CELLS, "CELLS")
DEMO_CELLS = NamedList([s for s, _ in PAIRS], "demo_cells")
CLONE_CELLS = NamedList(visits(CLONE, 20, 100), "clone_cells")


def _bars(rows: list, values: list, alt: str, lang: str) -> str:
    """Three bars of shares, from 0 to 1, in the color that counts mistakes, each with its label and its value."""
    width, label_w, bar_w, row_h = 314, 0, 150, 34
    parts = []
    for i, (label, v) in enumerate(zip(rows, values)):
        y = i * row_h
        parts.append(f'<text x="0" y="{y + 12}" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(label)}</text>')
        parts.append(f'<rect x="0" y="{y + 17}" width="{bar_w}" height="10" style="fill:none;stroke:var(--ea-line);stroke-width:1"/>')
        parts.append(f'<rect x="0" y="{y + 17}" width="{bar_w * max(0.0, min(1.0, v)):.1f}" height="10" style="fill:var(--ea-loss)"/>')
        parts.append(f'<text x="{bar_w + 8}" y="{y + 26}" style="font-size:12px;font-weight:600;fill:var(--ea-loss-text)">{percent(v, lang)}</text>')
    H = len(rows) * row_h
    svg = (f'<svg viewBox="0 0 {width} {H}" aria-hidden="true" style="width:100%;max-width:{width}px;height:auto;'
           f'font-family:var(--ea-font-ui)">{"".join(parts)}</svg>')
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(alt)}" style="margin:0.25rem 0 0.5rem;'
            f'max-width:min(100%,{width}px)">{svg}</figure>')


def mistakes_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]
    run_ = seen.mine
    fn = run_.names.get("mistakes") if run_ is not None else None
    head = _note(w["m_head"])
    lists = [list(DEMO_CELLS), list(ALL_CELLS), list(CLONE_CELLS)]
    got = _call(lambda: [fn(dict(CLONE), dict(EXPERT), states) for states in lists]) if callable(fn) else None
    ok = isinstance(got, list) and all(isinstance(v, (int, float, np.floating)) and not isinstance(v, bool)
                                       and math.isfinite(float(v)) for v in got)
    if not ok:
        return yours.figure(w["m_title"], head + _note(w["m_none"], _LOSS_UI), alt=w["m_title"])
    return yours.figure(w["m_title"], head + _bars(w["m_rows"], [float(v) for v in got], w["m_alt"], lang), alt=w["m_title"])


# --- the exercises ----------------------------------------------------------------------------------------------------

REQUIRED = ["demonstrate", "clone", "mistakes", "bigger-k"]


def _dots_left(got, text) -> bool:
    import re
    return re.search(r"^[^#\n]*\.\.\.", text or "", re.MULTILINE) is not None


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang de `...` door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def show_pairs(v):
    """A list of pairs in a message: how many, the first and the last."""
    if isinstance(v, list) and v and all(isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], (tuple, list)) for p in v):
        first, last = (f"({tuple(p[0])!r}, {str(p[1])!r})" for p in (v[0], v[-1]))
        return {"en": f"a list of {len(v)} pairs, from `{first}` to `{last}`",
                "nl": f"een lijst met {len(v)} paren, van `{first}` tot en met `{last}`"}
    text = repr(v)
    text = text if len(text) <= 80 else text[:77] + "..."
    return {"en": f"`{text}`", "nl": f"`{text}`"}


def show_policy(v):
    """A policy in a message: the three rows of the bridge as arrows, · where a cell has no action."""
    if isinstance(v, dict):
        rows = []
        for r in range(1, ROWS - 1):
            rows.append("".join("G" if (r, c) == GOAL else _GLYPH.get(str(v.get((r, c))), "·") for c in range(COLS)))
        return {"en": "`" + " / ".join(rows) + "`", "nl": "`" + " / ".join(rows) + "`"}
    text = repr(v)
    text = text if len(text) <= 80 else text[:77] + "..."
    return {"en": f"`{text}`", "nl": f"`{text}`"}


def show_share(v):
    """A share in a message, as Python writes it, rounded to four decimals."""
    if isinstance(v, (float, np.floating)) and math.isfinite(float(v)):
        return f"`{round(float(v), 4)!r}`"
    text = repr(v)
    return f"`{text if len(text) <= 80 else text[:77] + '...'}`"


def same_policy(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(str(a[s]) == str(b[s]) for s in b)
    return a == b


def _pair(x):
    """A pair of a cell and an action as plain Python values, also when the pair is a list or the cell an array;
    anything else as it is."""
    if isinstance(x, (tuple, list)) and len(x) == 2 and isinstance(x[0], (tuple, list, np.ndarray)):
        return tuple(int(v) for v in x[0]), str(x[1])
    return x


def same_pairs(a, b) -> bool:
    try:
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and [_pair(x) for x in a] == [_pair(y) for y in b]
        return bool(a == b)
    except Exception:   # noqa: BLE001   values that cannot be compared are not the same
        return False


# exercise 1: demonstrate
_D_HEAD = ("import numpy as np\n\n\n"
           "def demonstrate(expert, episodes, seed):\n"
           "    rng = np.random.default_rng(seed)\n"
           "    pairs = []\n"
           "    for _ in range(episodes):\n"
           "        state = START\n"
           "        over = False\n"
           "        while not over:\n")
_D_TAIL = "    return pairs\n"


def _d(*lines: str) -> str:
    body = lines or ("action = expert[state]", "pairs.append((state, action))", "state, reward, over = step(state, action, rng)")
    return _D_HEAD + "".join(f"            {line}\n" for line in body) + _D_TAIL


_D_START = _d("...")
_D = _d()
_D_LATE = _d("action = expert[state]", "state, reward, over = step(state, action, rng)", "pairs.append((state, action))")
_D_ACTIONS = _d("action = expert[state]", "pairs.append(action)", "state, reward, over = step(state, action, rng)")
_D_STATES = _d("action = expert[state]", "pairs.append(state)", "state, reward, over = step(state, action, rng)")
_D_NEW_RNG = _d("action = expert[state]", "pairs.append((state, action))",
                "state, reward, over = step(state, action, np.random.default_rng(seed))")
_D_COURSE_EXPERT = _d("action = EXPERT[state]", "pairs.append((state, action))", "state, reward, over = step(state, action, rng)")
D_NAMES = {"START": START, "step": step, "EXPERT": EXPERT}

# exercise 2: clone
_C_HEAD = ("import numpy as np\n\n\n"
           "def clone(pairs, k):\n"
           "    X = []\n"
           "    y = []\n"
           "    for state, action in pairs:\n")


def _c(collect: tuple = ("X.append(state)", "y.append(action)"), predict: str = "knn_predict(X, y, state, k)") -> str:
    return (_C_HEAD + "".join(f"        {line}\n" for line in collect)
            + "    X = np.array(X)\n"
            + "    y = np.array(y)\n"
            + "    policy = {}\n"
            + "    for state in CELLS:\n"
            + f"        policy[state] = {predict}\n"
            + "    return policy\n")


_C_START = _c(("...",), "...")
_C = _c()
_C_SWAPPED = _c(("X.append(action)", "y.append(state)"))
_C_ALL = _c(predict="knn_predict(X, y, state, len(pairs))")
_C_ONLY = ("def clone(pairs, k):\n"
           "    policy = {}\n"
           "    for state, action in pairs:\n"
           "        policy[state] = action\n"
           "    return policy\n")
C_NAMES = {"CELLS": CELLS, "knn_predict": knn_predict}

# exercise 3: mistakes
_M_HEAD = ("def mistakes(policy, expert, states):\n"
           "    wrong = 0\n"
           "    for state in states:\n")


def _m(test: str = "policy[state] != expert[state]", back: str = "wrong / len(states)") -> str:
    return _M_HEAD + f"        if {test}:\n            wrong = wrong + 1\n" + f"    return {back}\n"


_M_START = _M_HEAD + "        ...\n    return ...\n"
_M = _m()
_M_COUNT = _m(back="wrong")
_M_AGREE = _m(test="policy[state] == expert[state]")
_M_CELLS = _m(back="wrong / 35")


def _pairs_of(expert, n, seed) -> NamedList:
    return NamedList(demonstrate(expert, n, seed), f"demonstrate({expert!r}, {n}, {seed})")


EXERCISES = {
    "demonstrate": FunctionBox(
        name="demonstrate",
        function="demonstrate",
        start=_D_START,
        solution=_D,
        names=D_NAMES,
        compare=same_pairs,
        show=show_pairs,
        visible=[(EXPERT, 1, 0)],
        hidden=[(EXPERT, 2, 3), (STRAIGHT, 2, 1), (EXPERT, 3, 7)],
        first=[_DOTS],
        picture=demonstrate_picture,
        mistakes=[
            Mistake("late", wrong=_D_LATE,
                    en="`{call}` gives {got}, and it should give {want}: each pair holds the cell the step led to, beside the action the expert took in the cell before it. Which cell belongs with the action: the one the expert acts in, or the one it lands on?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elk paar bevat het vakje waar de stap heen leidde, naast de action die de expert in het vorige vakje nam. Welk vakje hoort bij de action: het vakje waar de expert de action neemt, of het vakje waar de expert landt?"),
            Mistake("actions-only", wrong=_D_ACTIONS,
                    en="`{call}` gives {got}: a list of actions alone. Later the clone has to find the cells nearest to a cell of its own, so which two things does each pair need?",
                    nl="`{call}` geeft {got}: alleen een lijst met actions. Later moet de kloon de vakjes vinden die het dichtst bij een eigen vakje liggen, dus welke twee dingen heeft elk paar nodig?"),
            Mistake("states-only", wrong=_D_STATES,
                    en="`{call}` gives {got}: a list of cells alone. The clone learns what the expert did in each cell, so what else does each pair need?",
                    nl="`{call}` geeft {got}: alleen een lijst met vakjes. De kloon leert wat de expert in elk vakje deed, dus wat heeft elk paar nog meer nodig?"),
            Mistake("new-rng", wrong=_D_NEW_RNG,
                    en="`{call}` gives {got}, and it should give {want}: every step gets a new generator with the same seed, so every step draws the same random number, and slips the same way, or never. Which generator should every step use?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke stap krijgt een nieuwe generator met dezelfde seed, dus trekt elke stap hetzelfde willekeurige getal, en glijdt elke stap op dezelfde manier uit, of nooit. Welke generator hoort elke stap te gebruiken?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the three lines of one step: the expert's action in `state`, the pair into `pairs`, and the step itself.",
                      "nl": "Vervang de `...` door de drie regels van één stap: de action van de expert in `state`, het paar in `pairs`, en de stap zelf."},
            "arguments": {"en": "The check calls `{call}`, with three values, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(expert, episodes, seed)`?",
                          "nl": "De check roept `{call}` aan, met drie waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(expert, episodes, seed)` houden?"},
            "slow": {"en": "Your function is still not done after 2 seconds. Does every pass of the `while` loop call `step`, so that `state` and `over` change?",
                     "nl": "Je functie is na 2 seconden nog niet klaar. Roept elke ronde van de `while`-loop `step` aan, zodat `state` en `over` veranderen?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does each pass of the loop take the action of `expert` in `state`, add the pair of the two to `pairs`, and then let `step` move the agent, with `rng`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Neemt elke ronde van de loop de action van `expert` in `state`, voegt de ronde het paar van de twee toe aan `pairs`, en laat de ronde dan `step` de agent verplaatsen, met `rng`?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function use the expert it gets, `expert`?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie de expert die je functie krijgt, `expert`?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return pairs`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return pairs`?"},
            "correct": {"en": "Right in all {n} cases: one episode of the expert with seed 0 gives 21 pairs, on 17 of the 35 cells. In the picture, which cells did the expert never stand on, even in five episodes?",
                        "nl": "Goed in alle {n} gevallen: één episode van de expert met seed 0 geeft 21 paren, op 17 van de 35 vakjes. Op welke vakjes stond de expert in de afbeelding nooit, zelfs niet in vijf episodes?"},
        },
    ),
    "clone": FunctionBox(
        name="clone",
        function="clone",
        start=_C_START,
        solution=_C,
        names=C_NAMES,
        compare=same_policy,
        show=show_policy,
        visible=[(PAIRS, 1)],
        hidden=[(_pairs_of(EXPERT, 5, 3), 1), (_pairs_of(EXPERT, 5, 3), 3), (_pairs_of(STRAIGHT, 2, 1), 1),
                (_pairs_of(EXPERT, 2, 8), 9)],
        first=[_DOTS],
        picture=clone_picture,
        mistakes=[
            Mistake("swapped", error=lambda e: "ufunc" in str(e),
                    en="With `{call}`, `knn_predict` stops: `X` holds actions, and k-NN cannot work out the distance between two words. Which of the two goes into `X`, the features, and which into `y`, the classes?",
                    nl="Met `{call}` stopt `knn_predict`: `X` bevat actions, en k-NN kan de afstand tussen twee woorden niet uitrekenen. Welke van de twee gaat in `X`, de features, en welke in `y`, de klassen?"),
            Mistake("only-demo", wrong=_C_ONLY,
                    en="`{call}` gives {got}: an action only in the cells the expert stood on. The clone needs an action in every cell of `CELLS`, also where the expert never stood; who answers for those cells?",
                    nl="`{call}` geeft {got}: alleen een action in de vakjes waar de expert stond. De kloon heeft in elk vakje van `CELLS` een action nodig, ook waar de expert nooit stond; wie geeft daar het antwoord?"),
            Mistake("all-vote", wrong=_C_ALL,
                    en="`{call}` gives {got}, and it should give {want}: every cell gets the action most pairs have, because all the pairs vote. How many neighbors should vote?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elk vakje krijgt de action die de meeste paren hebben, omdat alle paren stemmen. Hoeveel buren horen te stemmen?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: in the first loop, the cell into `X` and the action into `y`; in the second, the action k-NN gives for `state`.",
                      "nl": "Vervang de twee `...`: in de eerste loop het vakje in `X` en de action in `y`; in de tweede de action die k-NN geeft voor `state`."},
            "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(pairs, k)`?",
                          "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(pairs, k)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does `X` get the cell of every pair and `y` its action, and does every cell of `CELLS` get what `knn_predict` gives for it, with `X`, `y` and `k`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Krijgt `X` het vakje van elk paar en `y` de action ervan, en krijgt elk vakje van `CELLS` wat `knn_predict` voor dit vakje geeft, met `X`, `y` en `k`?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function use the pairs and the `k` it gets?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie de paren en de `k` die je functie krijgt?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return policy`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return policy`?"},
            "correct": {"en": "Right in all {n} cases: the clone of one demonstration does what the expert did in all 17 cells the expert stood on, and something else in 16 of the other 18. In the picture, where are the outlined cells?",
                        "nl": "Goed in alle {n} gevallen: de kloon van één demonstratie doet in alle 17 vakjes waar de expert stond wat de expert deed, en iets anders in 16 van de andere 18. Waar liggen de omlijnde vakjes in de afbeelding?"},
        },
    ),
    "mistakes": FunctionBox(
        name="mistakes",
        function="mistakes",
        start=_M_START,
        solution=_M,
        visible=[(CLONE, EXPERT, ALL_CELLS)],
        hidden=[(CLONE, EXPERT, DEMO_CELLS), (CLONE, EXPERT, CLONE_CELLS), (STRAIGHT, EXPERT, ALL_CELLS),
                (EXPERT, EXPERT, CLONE_CELLS)],
        first=[_DOTS],
        show=show_share,
        picture=mistakes_picture,
        mistakes=[
            Mistake("count", wrong=_M_COUNT,
                    en="`{call}` gives {got}, and it should give {want}: the number of states with a mistake, not the share. What do you divide it by?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het aantal states met een fout, niet het deel. Waardoor deel je dit aantal?"),
            Mistake("agree", wrong=_M_AGREE,
                    en="`{call}` gives {got}, and it should give {want}: the share of the states where the clone does the same as the expert. Which test counts the states where the two differ?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het deel van de states waar de kloon hetzelfde doet als de expert. Welke test telt de states waar de twee verschillen?"),
            Mistake("cells", wrong=_M_CELLS,
                    en="`{call}` gives {got}, and it should give {want}: you divide by 35, the number of cells of the bridge, but `states` can hold any number of states, also the same cell more than once. How many states does it hold?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je deelt door 35, het aantal vakjes van de brug, maar `states` kan elk aantal states bevatten, ook hetzelfde vakje vaker. Hoeveel states bevat `states`?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: in the loop, add 1 to `wrong` where `policy` and `expert` differ; after it, give back the share.",
                      "nl": "Vervang de twee `...`: tel in de loop 1 op bij `wrong` waar `policy` en `expert` verschillen; geef daarna het deel terug."},
            "arguments": {"en": "The check calls `{call}`, with three values, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(policy, expert, states)`?",
                          "nl": "De check roept `{call}` aan, met drie waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(policy, expert, states)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does `wrong` count the states of `states` where `policy` and `expert` give different actions, and do you divide by the number of states?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Telt `wrong` de states van `states` waar `policy` en `expert` verschillende actions geven, en deel je door het aantal states?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for every list of states, also one with the same cell more than once?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie voor elke lijst met states, ook een lijst met hetzelfde vakje vaker?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it end with `return` and the share?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie met `return` en het deel?"},
            "correct": {"en": "Right in all {n} cases: the clone differs from the expert in 16 of the 35 cells, in none of the cells of the demonstration, and in 27.3% of the cells it stands on itself in 20 episodes. In the picture, which bar is the one that decides whether the clone falls?",
                        "nl": "Goed in alle {n} gevallen: de kloon wijkt af van de expert in 16 van de 35 vakjes, in geen enkel vakje van de demonstratie, en in 27,3% van de vakjes waar de kloon zelf staat in 20 episodes. Welke balk in de afbeelding bepaalt of de kloon valt?"},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("demonstrate", _D_START, "empty", None),
    ("demonstrate", _D, "correct", None),
    ("demonstrate", _d("action = expert[state]", "pairs.append(...)", "state, reward, over = step(state, action, rng)"), "wrong", "dots-left"),
    ("demonstrate", _D_LATE, "wrong", "late"),
    ("demonstrate", _D_ACTIONS, "wrong", "actions-only"),
    ("demonstrate", _D_STATES, "wrong", "states-only"),
    ("demonstrate", _D_NEW_RNG, "wrong", "new-rng"),
    ("demonstrate", _D.replace("    return pairs\n", ""), "wrong", "no-return"),
    # another right answer: the action straight into the pair, and the reward left unnamed
    ("demonstrate", _d("pairs.append((state, expert[state]))", "state, _, over = step(state, expert[state], rng)"), "correct", None),
    # another right answer: each pair as a list
    ("demonstrate", _d("action = expert[state]", "pairs.append([state, action])", "state, reward, over = step(state, action, rng)"), "correct", None),
    # the course's EXPERT in place of the expert the function gets: right for the example only
    ("demonstrate", _D_COURSE_EXPERT, "almost", None),
    ("clone", _C_START, "empty", None),
    ("clone", _C, "correct", None),
    ("clone", _c(predict="knn_predict(X, y, ...)"), "wrong", "dots-left"),
    ("clone", _C_SWAPPED, "error", "swapped"),
    ("clone", _C_ONLY, "wrong", "only-demo"),
    ("clone", _C_ALL, "wrong", "all-vote"),
    ("clone", _C.replace("    return policy\n", ""), "wrong", "no-return"),
    # another right answer: the cell as an array
    ("clone", _c(predict="knn_predict(X, y, np.array(state), k)"), "correct", None),
    # k = 1 written as a number: right for the example only
    ("clone", _c(predict="knn_predict(X, y, state, 1)"), "almost", None),
    ("mistakes", _M_START, "empty", None),
    ("mistakes", _M, "correct", None),
    ("mistakes", _m(back="..."), "wrong", "dots-left"),
    ("mistakes", _M_COUNT, "wrong", "count"),
    ("mistakes", _M_AGREE, "wrong", "agree"),
    ("mistakes", _M_CELLS, "wrong", "cells"),
    ("mistakes", _m(back="wrong / len(states)").replace("    return wrong / len(states)\n", ""), "wrong", "no-return"),
    # another right answer: a share that starts at 0 and grows by 1 / len(states)
    ("mistakes", _M_HEAD + "        if policy[state] != expert[state]:\n            wrong = wrong + 1 / len(states)\n    return wrong\n", "correct", None),
    # each cell counted once: right for a list without repeats only
    ("mistakes", _m(back="wrong / len(set(states))"), "almost", None),
]
