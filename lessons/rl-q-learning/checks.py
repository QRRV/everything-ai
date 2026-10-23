# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rl-q-learning: the grid world of the lesson before, the environment that rolls the dice of its
table T, the course's own Q-learning and its recorded run, which the replay goes through; update and choose, the
functions the learner completes, and which the demo trains with."""
import functools
import re
import time
from dataclasses import dataclass

import numpy as np

from everything_ai import chart, draw
from everything_ai.check import FunctionBox, Mistake

ACTIONS = ["up", "down", "left", "right"]
MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
SIDES = {"up": ["left", "right"], "down": ["left", "right"], "left": ["up", "down"], "right": ["up", "down"]}
SLIP = 0.2       # the chance that a step slips: 0.1 to each side of the action, as in the lesson before
GAMMA = 0.9
ETA = 0.2        # the learning rate of the demo and the course's run
EPSILON = 0.1    # the chance of a random action
EPISODES = 300
MAX_STEPS = 100  # an episode stops after this many steps, also when the task is not over
SEED = 2         # the course's recorded run
YOUR_SEED = 7    # the learner's run, so that it is not the course's own


class Named(dict):
    """A Q-table that a check message calls by its name; the learner's code gets a plain copy."""

    def __init__(self, entries, name: str):
        super().__init__(entries)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return {s: dict(qs) for s, qs in self.items()}


GRID = [
    "......",
    "S....G",
    "XXXXXX",
]


def find(world, sign: str) -> tuple[int, int]:
    for r, row in enumerate(world):
        if sign in row:
            return (r, row.index(sign))
    raise ValueError(sign)


START, GOAL = find(GRID, "S"), find(GRID, "G")


def land(world, state, move):
    """One step from `state` in the direction of `move`; at the edge of the world you stay where you are."""
    r, c = state[0] + MOVES[move][0], state[1] + MOVES[move][1]
    if 0 <= r < len(world) and 0 <= c < len(world[0]):
        return (r, c)
    return state


def is_end(state) -> bool:
    return GRID[state[0]][state[1]] in "GX"


def table(world, slip: float = SLIP) -> dict:
    """The MDP of the lesson before as a table: T[state][action] lists (probability, next state, reward)."""
    T = {}
    for r, row in enumerate(world):
        for c, sign in enumerate(row):
            T[(r, c)] = {}
            for a in ACTIONS:
                if sign in "GX":
                    T[(r, c)][a] = [(1.0, (r, c), 0)]
                    continue
                outs = []
                for p, move in [(1 - slip, a), (slip / 2, SIDES[a][0]), (slip / 2, SIDES[a][1])]:
                    nxt = land(world, (r, c), move)
                    there = world[nxt[0]][nxt[1]]
                    outs.append((round(p, 10), nxt, 1 if there == "G" else -1 if there == "X" else 0))
                T[(r, c)][a] = outs
    return T


T = table(GRID)
STATES = [s for s in T if not is_end(s)]


def q_star(rounds: int = 300) -> dict:
    """The Q-values of value iteration, from the lesson before: what Q-learning should come close to."""
    V = {s: 0 for s in T}
    for _ in range(rounds):
        Q = {s: {a: sum(p * (r + GAMMA * V[n]) for p, n, r in T[s][a]) for a in ACTIONS} for s in T}
        V = {s: max(Q[s].values()) for s in T}
    return Q


Q_STAR = q_star()


def best_actions(qs: dict, tie: float = 1e-9) -> list[str]:
    """Every action whose Q is the largest; none when all four are the same, as at the start."""
    top = max(qs.values())
    best = [a for a in ACTIONS if qs[a] > top - tie]
    return [] if len(best) == len(ACTIONS) else best


def step(state, action, rng):
    """The environment: it rolls the dice with the probabilities of T, and gives back where the agent landed and its
    reward. The agent's code never looks at T."""
    x, total = rng.random(), 0.0
    for p, nxt, reward in T[state][action]:
        total += p
        if x < total:
            return nxt, reward
    return nxt, reward


def update(Q, state, action, reward, nxt, eta, gamma):
    """The course's own update, as in the lesson: the new Q of the state and action."""
    qs = []
    for a in Q[nxt]:
        qs.append(Q[nxt][a])
    target = reward + gamma * max(qs)
    return Q[state][action] + eta * (target - Q[state][action])


def choose(Q, state, epsilon, rng):
    """The course's own ε-greedy choice, as in the lesson."""
    actions = []
    qs = []
    for a in Q[state]:
        actions.append(a)
        qs.append(Q[state][a])
    if rng.random() < epsilon:
        return actions[rng.integers(0, 4)]
    return actions[np.argmax(qs)]


def greedy(Q, state) -> str:
    return ACTIONS[int(np.argmax([Q[state][a] for a in ACTIONS]))]


def greedy_way(Q) -> list:
    """Where the greedy policy of Q goes from S if no step slips, until G, a pit, or 12 steps."""
    s, way = START, [START]
    for _ in range(12):
        s = land(GRID, s, greedy(Q, s))
        way.append(s)
        if is_end(s):
            break
    return way


TOP_WAY = greedy_way(Q_STAR)   # up from S, along the top row, and down into G


def stops(episodes: int = EPISODES) -> list[int]:
    """The episodes the replay stops at: each of the first ten, then every tenth."""
    return list(range(0, 11)) + list(range(20, episodes + 1, 10))


def train(update_fn, choose_fn, episodes: int = EPISODES, seed: int = SEED, eta: float = ETA, epsilon: float = EPSILON,
          gamma: float = GAMMA, show=None, pause: float = 0.0) -> dict | str:
    """Q-learning on GRID with these two functions, after the recipe of the lesson: the return, the number of steps and
    where every episode ended, and at every stop of the replay the Q-table and the way of that episode. A function that
    stops with an error, or gives back something that does not fit, ends the run: then the name of that function.
    `show(episode, return)` hears every episode; `pause` waits that long after each one, so that you can watch."""
    rng = np.random.default_rng(seed)
    Q = {s: {a: 0.0 for a in ACTIONS} for s in T}
    keep = set(stops(episodes))
    run = {"returns": [], "steps": [], "ends": [], "Q": {0: {s: dict(qs) for s, qs in Q.items()}}, "ways": {0: []}}
    for episode in range(1, episodes + 1):
        state, G, factor, way, end = START, 0.0, 1.0, [START], "limit"
        for _ in range(MAX_STEPS):
            try:
                action = choose_fn(Q, state, epsilon, rng)
            except Exception:
                return "choose"
            if action not in ACTIONS:
                return "choose"
            nxt, reward = step(state, action, rng)
            try:
                new = float(update_fn(Q, state, action, reward, nxt, eta, gamma))
            except Exception:
                return "update"
            if not np.isfinite(new):
                return "update"
            Q[state][action] = new
            G += factor * reward
            factor *= gamma
            way.append(nxt)
            state = nxt
            if is_end(state):
                end = "goal" if state == GOAL else "pit"
                break
        run["returns"].append(G)
        run["steps"].append(len(way) - 1)
        run["ends"].append(end)
        if episode in keep:
            run["Q"][episode] = {s: dict(qs) for s, qs in Q.items()}
            run["ways"][episode] = way
        if show is not None:
            show(episode, G)
        if pause:
            time.sleep(pause)
    run["final"] = Q
    return run


RUN = train(update, choose)
Q = Named(RUN["Q"][EPISODES], "Q")        # the Q-table the checks give update and choose: the course's after its run


def mean_of_runs(n: int = 20, episodes: int = EPISODES) -> list[float]:
    """The return of every episode, as the mean of n runs of the course with other seeds."""
    runs = [train(update, choose, episodes, seed=100 + k)["returns"] for k in range(n)]
    return [float(np.mean([r[i] for r in runs])) for i in range(episodes)]


def means(returns: list[float], window: int = 20) -> list[float]:
    """The mean return of the last `window` episodes, at every episode."""
    return [float(np.mean(returns[max(0, i - window + 1): i + 1])) for i in range(len(returns))]


@functools.cache
def mean_line() -> tuple[float, ...]:
    """The dashed line of the demo: the mean over the last 20 episodes, as the thick line, of the mean of 20 runs of the
    course. Worked out once, at the first press of the button, so that a press after it starts the chart at once."""
    return tuple(means(mean_of_runs()))


def top_from(run: dict) -> int | None:
    """The first stop of the replay from which the greedy way from S stays the top row of value iteration."""
    at = [k for k in stops(len(run["returns"])) if k > 0]
    for i, k in enumerate(at):
        if all(greedy_way(run["Q"][j]) == TOP_WAY for j in at[i:]):
            return k
    return None


def agree(Q) -> int:
    """In how many of the states where the task goes on the greedy action of Q is a best action of value iteration."""
    return sum(greedy(Q, s) in best_actions(Q_STAR[s]) for s in STATES)


def differ(Q) -> list:
    """The states where it is not, with the action of Q."""
    return [(s, greedy(Q, s)) for s in STATES if greedy(Q, s) not in best_actions(Q_STAR[s])]


def number(x: float, digits: int, lang: str) -> str:
    """A number for running text, with the decimal mark of the language and a real minus sign."""
    text = f"{x:.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    text = text.replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _cells(Q) -> tuple[dict, dict]:
    """The largest Q in every state, and the greedy action where one action is best."""
    values = {s: max(Q[s].values()) for s in STATES}
    arrows = {s: [greedy(Q, s)] if best_actions(Q[s]) else [] for s in STATES}
    return values, arrows


def policies(Q, titles: tuple[str, str], alt: str, decimal: str = "."):
    """The greedy policy of a Q-table next to that of value iteration, each with the largest Q in every state."""
    mine, star = _cells(Q), _cells(Q_STAR)
    return draw.grid_world([{"rows": GRID, "title": titles[0], "role": "param", "values": mine[0], "arrows": mine[1], "index": True},
                            {"rows": GRID, "title": titles[1], "role": "pred", "values": star[0], "arrows": star[1], "index": True}],
                           alt=alt, cell=50, decimal=decimal)


def returns_chart(returns: list[float], upto: int, labels: dict, lang: str) -> str:
    """The return of every episode of a run up to `upto`, thin, and its mean over 20 episodes, in ink, with a dot at
    `upto`; the x-axis has every episode of the run, so that you see how far it is."""
    p = chart.Panel((0, len(returns)), (-1.05, 1.05), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    yticks=[-1, -0.5, 0, 0.5, 1])
    xs = list(range(1, upto + 1))
    if upto > 1:
        p.line(xs, returns[:upto], "axis", 0.8)
        p.line(xs, means(returns[:upto]), "ink", 2.4)
    if upto >= 1:
        p.points([upto], [means(returns[:upto])[-1]], "ink", r=3.5)
    svg = chart.figure([p], alt=labels["alt"], width=330, height=240)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


def replay_frame(run: dict, k: int, labels: dict, lang: str) -> tuple[str, str]:
    """Episode k of a run, for the replay: the way of that episode and the greedy policy of the Q-table after it, with
    the largest Q in every state; beside it the returns so far. And the sentence under it."""
    Q = run["Q"][k]
    values, arrows = _cells(Q)
    grid = draw.grid_world([{"rows": GRID, "title": labels["grid"].format(k=k), "role": "param", "values": values,
                             "arrows": arrows, "path": run["ways"][k], "index": True}],
                           alt=labels["grid_alt"].format(k=k), cell=50, decimal="," if lang == "nl" else ".").text
    side = returns_chart(run["returns"], k, labels, lang)
    picture = f'<div style="display:flex;flex-wrap:wrap;gap:0 16px;align-items:flex-start">{grid}{side}</div>'
    if k == 0:
        return picture, labels["start"]
    way = greedy_way(Q)
    where = "top" if way == TOP_WAY else "goal" if way[-1] == GOAL else "not"
    said = labels["episode"].format(k=k, n=run["steps"][k - 1], end=labels[run["ends"][k - 1]],
                                    g=number(run["returns"][k - 1], 2, lang), way=labels["way_" + where])
    return picture, said


# --- how an answer is judged -------------------------------------------------------------------------------------

def _plain(x) -> str:
    """A number the way the lessons write it in code: 4 decimals at most, without float noise or e-notation."""
    x = x.item() if hasattr(x, "item") else x
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return repr(x)
    text = f"{x:.4f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def same_number(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) <= 1e-9
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class Picks:
    """How often a choose function picked each action, over many calls with the same Q, state and ε."""
    kind: str                       # "ok"; "other": something that is not an action
    shares: tuple = ()              # (share of up, of down, of left, of right)
    other: str = ""                 # what it gave back instead of an action, as code


CALLS = 4000   # the check calls choose this many times per case
CLOSE = 0.035  # a share at most this far from that of ε-greedy counts as ε-greedy: at 4000 calls 4.5 standard deviations or more


def epsilon_greedy(Q_, state, epsilon) -> tuple:
    """How often ε-greedy takes each action, exactly: ε/4 each, and 1 − ε more for the one with the largest Q."""
    best = greedy(Q_, state)
    return tuple(epsilon / 4 + (1 - epsilon) * (a == best) for a in ACTIONS)


def same_picks(a, b) -> bool:
    if not (isinstance(a, Picks) and isinstance(b, Picks)):
        return a == b
    if a.kind != b.kind:
        return False
    return a.kind != "ok" or all(abs(x - y) <= CLOSE for x, y in zip(a.shares, b.shares))


def _per_100(x: float, lang: str) -> str:
    """A share as times in 100, with one decimal where it has one: 92.5, 2.5, 33.1, 100."""
    text = f"{100 * x:.1f}".removesuffix(".0")
    return text.replace(".", ",") if lang == "nl" else text


def show_picks(p) -> dict:
    if not isinstance(p, Picks):
        return {"en": repr(p), "nl": repr(p)}
    if p.kind == "other":
        return {"en": f"`{p.other}`, which is not one of the four actions", "nl": f"`{p.other}`, en dat is geen van de vier actions"}
    return {lang: ", ".join(f'"{a}" {_per_100(x, lang)}' for a, x in zip(ACTIONS, p.shares)) for lang in ("en", "nl")}


@dataclass
class ChooseBox(FunctionBox):
    """choose in a code box. The check calls it CALLS times for every case, with one random number generator, and
    counts how often it picks each action. Shares that close to those of ε-greedy count as exactly those: any ε-greedy
    choice is right, whichever random numbers it draws, also with Python's own random module."""

    def _run(self, fn, case):
        Q_, state, epsilon = case
        Q_ = {s: dict(qs) for s, qs in Q_.items()}   # the learner's code gets a plain copy
        rng = np.random.default_rng(1)
        counts = dict.fromkeys(ACTIONS, 0)
        for _ in range(CALLS):
            action = fn(Q_, state, epsilon, rng)
            if action not in counts:
                plain = action.item() if hasattr(action, "item") else action
                return None if action is None else Picks("other", other=repr(plain))
            counts[action] += 1
        shares = tuple(counts[a] / CALLS for a in ACTIONS)
        exact = epsilon_greedy(Q_, state, epsilon)
        if all(abs(x - y) <= CLOSE for x, y in zip(shares, exact)):
            shares = exact   # as close as chance allows: this is ε-greedy
        return Picks("ok", shares)

    def _call_text(self, case) -> str:
        Q_, state, epsilon = case
        return f"choose({Q_!r}, {state}, {epsilon}, rng)"


# The lesson is done when these are correct. "no-exploring" is the choice question in the lesson.
REQUIRED = ["update", "choose", "no-exploring"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _u(target: str = "target = reward + gamma * max(qs)",
       back: str = "return Q[state][action] + eta * (target - Q[state][action])") -> str:
    """update as the box holds it, with these two lines: the target and the step of step 5 of the recipe."""
    return ("def update(Q, state, action, reward, nxt, eta, gamma):\n"
            "    qs = []\n"
            "    for a in Q[nxt]:\n"
            "        qs.append(Q[nxt][a])\n"
            f"    {target}\n"
            f"    {back}\n")


def _c(test: str = "rng.random() < epsilon", best: str = "return actions[np.argmax(qs)]",
       random: str = "return actions[rng.integers(0, 4)]") -> str:
    """choose as the box holds it, with these lines: the test for a random action, and the best action."""
    return ("import numpy as np\n\n\n"
            "def choose(Q, state, epsilon, rng):\n"
            "    actions = []\n"
            "    qs = []\n"
            "    for a in Q[state]:\n"
            "        actions.append(a)\n"
            "        qs.append(Q[state][a])\n"
            f"    if {test}:\n"
            f"        {random}\n"
            f"    {best}\n")


_U_START = _u("target = ...", "return ...")
_U = _u()
_C_START = _c("...", "return ...")
_C = _c()

_ZERO = Named({s: dict.fromkeys(ACTIONS, 0.0) for s in T}, "Q_ZERO")   # a Q-table before any step: all 0

EXERCISES = {
    "update": FunctionBox(
        name="update",
        function="update",
        start=_U_START,
        solution=_U,
        compare=same_number,
        show=_plain,
        visible=[(Q, START, "up", 0, (0, 0), ETA, GAMMA)],
        hidden=[(Q, (0, 5), "down", 1, GOAL, ETA, GAMMA), (Q, (1, 2), "right", -1, (2, 2), ETA, GAMMA),
                (Q, (0, 3), "up", 0, (0, 3), 0.5, 0.8), (_ZERO, START, "right", 0, (1, 1), ETA, GAMMA),
                (Q, (1, 4), "right", 0, (0, 4), 0.1, 0.9)],
        first=[_DOTS],
        mistakes=[
            Mistake("no-gamma", wrong=_u("target = reward + max(qs)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. Your target counts the best Q of the next state as much as a reward now. Multiply it by `gamma`, as in the formula of *The idea*.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Je target telt de beste Q van de volgende state even zwaar als een reward nu. Vermenigvuldig hem met `gamma`, zoals in de formule van *Het idee*."),
            Mistake("no-reward", wrong=_u("target = gamma * max(qs)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. Without the reward, nothing ever brings a Q away from 0. The target is the reward plus `gamma` times the best Q of the next state.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Zonder de reward komt er nooit een Q van 0 af. Het target is de reward plus `gamma` keer de beste Q van de volgende state."),
            Mistake("jump", wrong=_u(back="return target"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. You put the target in place of the old Q, so Q forgets everything before this step, and with slips it jumps up and down. Move it a step `eta` towards the target: `Q[state][action] + eta * (target - Q[state][action])`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Je zet het target in de plaats van de oude Q, dus vergeet Q alles van voor deze stap, en met uitglijders springt hij op en neer. Schuif hem een stap `eta` op naar het target: `Q[state][action] + eta * (target - Q[state][action])`."),
            Mistake("away", wrong=_u(back="return Q[state][action] + eta * (Q[state][action] - target)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. `Q - target` points away from the target: your Q moves the wrong way. Take `target - Q[state][action]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. `Q - target` wijst van het target af: je Q schuift de verkeerde kant op. Neem `target - Q[state][action]`."),
            Mistake("no-minus", wrong=_u(back="return Q[state][action] + eta * target"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. You add a share of the target to Q, so Q keeps growing. The step goes over the difference between the target and the old Q: `eta * (target - Q[state][action])`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Je telt een deel van het target bij Q op, dus Q blijft groeien. De stap gaat over het verschil tussen het target en de oude Q: `eta * (target - Q[state][action])`."),
            Mistake("same-action", wrong=_u("target = reward + gamma * Q[nxt][action]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. You take the Q of the same action in the next state; Q-learning takes the best one there, `max(qs)`, the value of acting as well as you can from there.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Je neemt de Q van dezelfde action in de volgende state; Q-learning neemt daar de beste, `max(qs)`, de waarde van zo goed mogelijk handelen vanaf daar."),
            Mistake("max-of-dict", error=lambda e: isinstance(e, TypeError) and "sequence" in str(e),
                    en="With `{call}`, your function stops with an error: `max(Q[nxt])` is the largest key of the dictionary, the action `\"up\"`, not a number. The loop above puts the Q-values of `nxt` in `qs`: take `max(qs)`.",
                    nl="Met `{call}` stopt je functie met een fout: `max(Q[nxt])` is de grootste key van de dictionary, de action `\"up\"`, geen getal. De loop erboven zet de Q-values van `nxt` in `qs`: neem `max(qs)`."),
        ],
        texts={**_TEXTS,
               "empty": {"en": "Replace the two `...`: the target, and the new Q, a step `eta` towards it.",
                         "nl": "Vervang de twee `...`: het target, en de nieuwe Q, een stap `eta` ernaartoe."},
               "arguments": {"en": "The check calls `{call}`, with seven values, and the `def` line of your function has another number of names between its parentheses. Keep `(Q, state, action, reward, nxt, eta, gamma)`.",
                             "nl": "De check roept `{call}` aan, met zeven waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(Q, state, action, reward, nxt, eta, gamma)`."},
               "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your two lines with step 5 of the recipe and the formula of *The idea*.",
                         "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je twee regels met stap 5 van het recept en de formule van *Het idee*."},
               "almost": {"en": "Right for the first step. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for a step into G or a pit, for a step into the edge, and for another `eta` and `gamma`?",
                          "nl": "Goed voor de eerste stap. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt het ook voor een stap op G of in een kuil, voor een stap tegen de rand, en voor een andere `eta` en `gamma`?"},
               "correct": {"en": "Your update is right in all {n} cases, also for a step into G, into a pit and into the edge.",
                           "nl": "Je update klopt in alle {n} gevallen, ook voor een stap op G, in een kuil en tegen de rand."}},
    ),
    "choose": ChooseBox(
        name="choose",
        function="choose",
        start=_C_START,
        solution=_C,
        compare=same_picks,
        show=show_picks,
        visible=[(Q, START, EPSILON)],
        hidden=[(Q, (0, 2), 0.0), (Q, (1, 4), 1.0), (Q, (0, 5), 0.5)],
        first=[_DOTS],
        mistakes=[
            Mistake("inverted", wrong=_c("rng.random() > epsilon"),
                    en="`{call}` picks, per 100 calls: {got}; it should be about {want}. `rng.random() > epsilon` is true in most calls, so your agent takes a random action most of the time. A random action comes with probability `epsilon`: `rng.random() < epsilon`.",
                    nl="`{call}` kiest, per 100 keer: {got}; dat moet ongeveer {want} zijn. `rng.random() > epsilon` is bij de meeste aanroepen waar, dus kiest je agent meestal een willekeurige action. Een willekeurige action komt met kans `epsilon`: `rng.random() < epsilon`."),
            Mistake("never-random", wrong=_c("False"),
                    en="`{call}` picks, per 100 calls: {got}; it should be about {want}. Your agent never takes a random action, so it never learns what the other actions do. With probability `epsilon` it should pick a random one: `rng.random() < epsilon`.",
                    nl="`{call}` kiest, per 100 keer: {got}; dat moet ongeveer {want} zijn. Je agent neemt nooit een willekeurige action, dus leert hij nooit wat de andere doen. Met kans `epsilon` hoort hij een willekeurige te kiezen: `rng.random() < epsilon`."),
            Mistake("smallest", wrong=_c(best="return actions[np.argmin(qs)]"),
                    en="`{call}` picks, per 100 calls: {got}; it should be about {want}. `np.argmin` gives the place of the smallest Q: the worst action. Outside the random actions, take the one with the largest Q.",
                    nl="`{call}` kiest, per 100 keer: {got}; dat moet ongeveer {want} zijn. `np.argmin` geeft de plek van de kleinste Q: de slechtste action. Buiten de willekeurige actions neem je die met de grootste Q."),
            Mistake("place-not-action", matches=lambda got, text: isinstance(got, Picks) and got.kind == "other",
                    en="`{call}` gives {got}. `np.argmax(qs)` and `rng.integers(0, 4)` are places in the list, numbers; give back the action at that place: `actions[...]`.",
                    nl="`{call}` geeft {got}. `np.argmax(qs)` en `rng.integers(0, 4)` zijn plekken in de lijst, getallen; geef de action op die plek terug: `actions[...]`."),
            Mistake("max-of-keys", wrong=_c(best="return max(Q[state])"),
                    en="`{call}` picks, per 100 calls: {got}; it should be about {want}. `max(Q[state])` is the largest key of the dictionary, `\"up\"` in the order of the alphabet, not the action with the largest Q. The loop above puts the Q-values in `qs`: take `actions[np.argmax(qs)]`.",
                    nl="`{call}` kiest, per 100 keer: {got}; dat moet ongeveer {want} zijn. `max(Q[state])` is de grootste key van de dictionary, `\"up\"` in de volgorde van het alfabet, niet de action met de grootste Q. De loop erboven zet de Q-values in `qs`: neem `actions[np.argmax(qs)]`."),
            Mistake("three", wrong=_c(random="return actions[rng.integers(0, 3)]"),
                    en="`{call}` picks, per 100 calls: {got}; it should be about {want}. `rng.integers(0, 3)` never gives 3, so a random action is never `\"right\"`. Its second number is not included: `rng.integers(0, 4)` gives 0, 1, 2 or 3.",
                    nl="`{call}` kiest, per 100 keer: {got}; dat moet ongeveer {want} zijn. `rng.integers(0, 3)` geeft nooit 3, dus is een willekeurige action nooit `\"right\"`. Het tweede getal telt niet mee: `rng.integers(0, 4)` geeft 0, 1, 2 of 3."),
        ],
        texts={**_TEXTS,
               "empty": {"en": "Replace the two `...`: when to take a random action, and which action otherwise.",
                         "nl": "Vervang de twee `...`: wanneer je een willekeurige action neemt, en welke action anders."},
               "arguments": {"en": "The check calls `{call}`, with four values, and the `def` line of your function has another number of names between its parentheses. Keep `(Q, state, epsilon, rng)`.",
                             "nl": "De check roept `{call}` aan, met vier waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(Q, state, epsilon, rng)`."},
               "slow": {"en": "Your function is still not done after 2 seconds. Does every loop go over a list or a dictionary, with `for`?",
                        "nl": "Je functie is na 2 seconden nog niet klaar. Gaat elke loop over een lijst of een dictionary, met `for`?"},
               "wrong": {"en": "`{call}` picks, per 100 calls: {got}; it should be about {want}. Compare your two lines with step 3 of the recipe.",
                         "nl": "`{call}` kiest, per 100 keer: {got}; dat moet ongeveer {want} zijn. Vergelijk je twee regels met stap 3 van het recept."},
               "almost": {"en": "Right with `epsilon` 0.1. But `{call}` picks, per 100 calls: {got}; it should be about {want}. Does it work for every `epsilon`, also 0 and 1?",
                          "nl": "Goed met `epsilon` 0.1. Maar `{call}` kiest, per 100 keer: {got}; dat moet ongeveer {want} zijn. Werkt het voor elke `epsilon`, ook 0 en 1?"},
               "correct": {"en": "Your choose is right in all {n} cases: with `epsilon` 0 always the best action, with 1 each action as often, and in between a random one in a share `epsilon` of the calls.",
                           "nl": "Je choose klopt in alle {n} gevallen: met `epsilon` 0 altijd de beste action, met 1 elke action even vaak, en daartussen een willekeurige in een deel `epsilon` van de aanroepen."}},
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("update", _U_START, "empty", None),
    ("update", _u("target = ..."), "wrong", "dots-left"),
    ("update", _u("target = reward + max(qs)"), "wrong", "no-gamma"),
    ("update", _u("target = gamma * max(qs)"), "wrong", "no-reward"),
    ("update", _u(back="return target"), "wrong", "jump"),
    ("update", _u(back="return Q[state][action] + eta * (Q[state][action] - target)"), "wrong", "away"),
    ("update", _u(back="return Q[state][action] + eta * target"), "wrong", "no-minus"),
    ("update", _u("target = reward + gamma * Q[nxt][action]"), "wrong", "same-action"),
    ("update", _u("target = reward + gamma * max(Q[nxt])"), "error", "max-of-dict"),
    # other right answers: the old Q and the target weighed, as some books write it
    ("update", _u(back="return (1 - eta) * Q[state][action] + eta * target"), "correct", None),
    ("update", _u(back="new = Q[state][action] + eta * (target - Q[state][action])\n    return new"), "correct", None),
    ("choose", _C_START, "empty", None),
    ("choose", _c(best="return ..."), "wrong", "dots-left"),
    ("choose", _c("rng.random() > epsilon"), "wrong", "inverted"),
    ("choose", _c("False"), "wrong", "never-random"),
    ("choose", _c(best="return actions[np.argmin(qs)]"), "wrong", "smallest"),
    ("choose", _c(best="return np.argmax(qs)"), "wrong", "place-not-action"),
    ("choose", _c(random="return actions[rng.integers(0, 3)]"), "wrong", "three"),
    ("choose", _c(best="return max(Q[state])"), "wrong", "max-of-keys"),
    # other right answers: another way to draw the random action, the test the other way round, and Python's own random
    ("choose", _c(random="return ACTIONS[rng.integers(0, 4)]").replace("import numpy as np\n",
                                                                        "import numpy as np\n\nACTIONS = [\"up\", \"down\", \"left\", \"right\"]\n"), "correct", None),
    ("choose", _c("rng.random() >= 1 - epsilon"), "correct", None),
    ("choose", "import random\n\nrandom.seed(3)\n" + _c("random.random() < epsilon", random="return random.choice(actions)"), "correct", None),
    # right with epsilon 0.1, but with epsilon 1 it never takes a random action
    ("choose", _c("rng.random() < epsilon and epsilon < 1"), "almost", None),
]
