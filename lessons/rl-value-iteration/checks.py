# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rl-value-iteration: the grid worlds of the lesson, the table T the course makes of each, and
the course's own value iteration, which draws the play block one round at a time, with V(S) per round below it;
value_iteration, the function the learner completes."""
import re
from dataclasses import dataclass

import numpy as np

from everything_ai import chart, draw
from everything_ai.check import FunctionBox, Mistake

ACTIONS = ["up", "down", "left", "right"]
MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
SIDES = {"up": ["left", "right"], "down": ["left", "right"], "left": ["up", "down"], "right": ["up", "down"]}
SLIP = 0.2      # the chance that a step slips: 0.1 to each side of the action
GAMMA = 0.9


class World(list):
    """A grid world as a list of strings, one per row: '.' a cell to stand on, S the start, G the goal (+1) and X a pit
    (−1). A check message calls it by the name the lesson gives it."""

    def __init__(self, rows, name: str):
        super().__init__(rows)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return list(self)


class Table(dict):
    """The table T of a world: T[state][action] is the list of what can happen, as (probability, next state, reward).
    A check message calls it by its name, such as T_SMALL; the learner's code gets a plain copy."""

    def __init__(self, entries, name: str, world: World):
        super().__init__(entries)
        self.name, self.world = name, world

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return {s: {a: list(outs) for a, outs in acts.items()} for s, acts in self.items()}


GRID = World([       # the world of the play block and of exercise 1: a ledge above a row of pits
    "......",
    "S....G",
    "XXXXXX",
], "GRID")
SMALL = World([      # the world of the worked example
    "..G",
    "S.X",
], "SMALL")
BRIDGE = World([     # a bridge over pits: every way to G runs along them
    "S..G",
    "XXXX",
], "BRIDGE")


def find(world, sign: str) -> tuple[int, int]:
    """The cell (row, column) with this sign."""
    for r, row in enumerate(world):
        if sign in row:
            return (r, row.index(sign))
    raise ValueError(sign)


START, GOAL = find(GRID, "S"), find(GRID, "G")


def land(world, state, move):
    """The cell one step from `state` in the direction of `move`; at the edge of the world you stay where you are."""
    r, c = state[0] + MOVES[move][0], state[1] + MOVES[move][1]
    if 0 <= r < len(world) and 0 <= c < len(world[0]):
        return (r, c)
    return state


def table(world, slip: float = SLIP, name: str = "") -> Table:
    """The MDP of a world as a table: T[state][action] lists what can happen, as (probability, next state, reward).
    An action goes where it says with probability 1 − slip, and slips to each side with slip / 2. Landing on G gives
    +1, in a pit −1, anywhere else 0. In G and in a pit the task is over: every action stays there, with reward 0."""
    T = {}
    for r, row in enumerate(world):
        for c, sign in enumerate(row):
            state = (r, c)
            T[state] = {}
            for a in ACTIONS:
                if sign in "GX":
                    T[state][a] = [(1.0, state, 0)]
                    continue
                outs = []
                for p, move in [(1 - slip, a), (slip / 2, SIDES[a][0]), (slip / 2, SIDES[a][1])]:
                    if p == 0:
                        continue
                    nxt = land(world, state, move)
                    there = world[nxt[0]][nxt[1]]
                    outs.append((round(p, 10), nxt, 1 if there == "G" else -1 if there == "X" else 0))
                T[state][a] = outs
    return Table(T, name or "T_" + world.name, world)


def q_values(T, V, gamma) -> dict:
    """Q(s, a) for every state and action: the sum over the outcomes of p × (r + gamma × V(next state))."""
    return {s: {a: sum(p * (r + gamma * V[nxt]) for p, nxt, r in T[s][a]) for a in ACTIONS} for s in T}


def value_iteration(T, gamma, rounds):
    """The course's own value iteration, as in the lesson: the values after `rounds` rounds, and the policy of the last
    round, the first action with the largest Q in each state."""
    V = {s: 0 for s in T}
    pi = {}
    for _ in range(rounds):
        Q = q_values(T, V, gamma)
        V = {s: max(Q[s].values()) for s in T}
        pi = {s: ACTIONS[int(np.argmax([Q[s][a] for a in ACTIONS]))] for s in T}
    return V, pi


def history(T, gamma, rounds) -> list[dict]:
    """The values after every round, from round 0 (all 0) up to `rounds`, and the Q-values they give."""
    V = {s: 0 for s in T}
    out = [{"V": dict(V), "Q": q_values(T, V, gamma)}]
    for _ in range(rounds):
        V = {s: max(qs.values()) for s, qs in out[-1]["Q"].items()}
        out.append({"V": dict(V), "Q": q_values(T, V, gamma)})
    return out


def best_actions(qs: dict, tie: float = 1e-9) -> list[str]:
    """Every action whose Q is the largest, in the order of ACTIONS; none when all are the same, as before round 1."""
    top = max(qs.values())
    best = [a for a in ACTIONS if qs[a] > top - tie]
    return [] if len(best) == len(ACTIONS) else best


def is_end(world, state) -> bool:
    return world[state[0]][state[1]] in "GX"


T = table(GRID, name="T")
T_SMALL = table(SMALL)
T_BRIDGE = table(BRIDGE)
T_NO_SLIP = table(GRID, slip=0.0, name="T_NO_SLIP")
ROUNDS = 20                      # the play block goes from round 0 to here: by then the values hardly change
STEPS = {"slip": history(T, GAMMA, ROUNDS), "still": history(T_NO_SLIP, GAMMA, ROUNDS)}
V_STAR, PI_STAR = value_iteration(T, GAMMA, 200)
V_STILL, PI_STILL = value_iteration(T_NO_SLIP, GAMMA, 200)
Q_STAR = q_values(T, V_STAR, GAMMA)


def evaluate(T, policy, gamma, rounds: int = 500) -> dict:
    """The expected return from every state if you always take the action of `policy` (policy evaluation)."""
    V = {s: 0 for s in T}
    for _ in range(rounds):
        V = {s: sum(p * (r + gamma * V[nxt]) for p, nxt, r in T[s][policy[s]]) for s in T}
    return V


STRAIGHT = evaluate(T, PI_STILL, GAMMA)   # the guess: the way along the pits, without slip the best, here with slip


def first_round(steps: list[dict], state, above: float) -> int:
    """The first round in which `state` has a value above `above`."""
    return next(k for k, s in enumerate(steps) if s["V"][state] > above)


ABOVE = {name: first_round(steps, START, 0.1) for name, steps in STEPS.items()}   # the task of the play block
# what the line of V(S) per round shows, for its alt text: without slip 0 until it is 0.66 at once; with slip it
# stays below 0.01 a little longer, then rises, and from round NEAR on it reads as its end value with 2 decimals
S_STILL = [s["V"][START] for s in STEPS["still"]]
S_SLIP = [s["V"][START] for s in STEPS["slip"]]
STILL_FROM = first_round(STEPS["still"], START, 0)
TINY_TO = max(k for k, v in enumerate(S_SLIP) if v < 0.01)
NEAR = next(k for k in range(ROUNDS + 1) if all(f"{v:.2f}" == f"{V_STAR[START]:.2f}" for v in S_SLIP[k:]))


def _panel(world, s: dict, title: str, *, values: bool = True, arrows: bool = True) -> dict:
    cells = [(r, c) for r in range(len(world)) for c in range(len(world[0])) if not is_end(world, (r, c))]
    return {"rows": world, "title": title, "role": "pred", "index": True,
            "values": {x: s["V"][x] for x in cells} if values else {},
            "arrows": {x: best_actions(s["Q"][x]) for x in cells} if arrows else {}}


def side_by_side(k: int, titles: tuple[str, str], alt: str, decimal: str = "."):
    """GRID after round k of value iteration, without slip and with slip 0.2, for the play block."""
    return draw.grid_world([_panel(GRID, STEPS["still"][k], titles[0]), _panel(GRID, STEPS["slip"][k], titles[1])],
                           alt=alt, cell=50, decimal=decimal)


def value_of_s(k: int, labels: dict, lang: str) -> str:
    """V(S) after every round of the play block, dashed without slip and solid with slip, both in the colour of a
    prediction, with a mark at round k and a dot on each line there: the whole run stays in view (lesson template,
    block 9). `labels` holds the words of the lesson; its alt text names every turn of both lines."""
    p = chart.Panel((0, ROUNDS), (0, 0.8), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    yticks=[0, 0.1, 0.2, 0.4, 0.6, 0.8])
    rounds = list(range(ROUNDS + 1))
    p.line([k, k], [0, 0.8], "axis", 1.2)
    p.line(rounds, S_STILL, "pred", 2.2, dashed=True)
    p.line(rounds, S_SLIP, "pred", 2.2)
    p.points([k], [S_STILL[k]], "pred", r=4, hollow=True)
    p.points([k], [S_SLIP[k]], "pred", r=4)
    p.text(13, 0.72, labels["still"], "ink-2")
    p.text(14, 0.31, labels["slip"], "ink-2")
    alt = labels["alt"].format(n=ROUNDS, still_to=STILL_FROM - 1, still_from=STILL_FROM, still_end=number(V_STILL[START], 2, lang),
                               tiny_to=TINY_TO, above=ABOVE["slip"], near=NEAR, slip_end=number(V_STAR[START], 2, lang),
                               k=k, a=number(S_STILL[k], 2, lang), b=number(S_SLIP[k], 2, lang))
    return chart.figure([p], alt=alt, width=330, height=220)


def worlds(alt: str, decimal: str = "."):
    """The two other worlds the check gives value_iteration, with their names."""
    return draw.grid_world([{"rows": SMALL, "title": "SMALL", "index": True}, {"rows": BRIDGE, "title": "BRIDGE", "index": True}],
                           alt=alt, cell=44, decimal=decimal)


def number(x: float, digits: int, lang: str) -> str:
    """A number for running text, with the decimal mark of the language and a real minus sign."""
    text = f"{x:.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    text = text.replace("-", "\u2212")
    return text.replace(".", ",") if lang == "nl" else text


# --- how an answer of value_iteration is judged --------------------------------------------------------------------

@dataclass(frozen=True)
class Solved:
    """What value_iteration gave back, as far as the check is concerned: the values, rounded so that a different order
    of adding up still counts as the same, and whether the policy takes a best action in every state. Where two actions
    are equally good, either one is right; a wrong policy is kept as it is, to tell one known mistake from another.
    `off` and `wrong` say where it first differs from the course's answer, for the message."""
    kind: str                       # "ok"; "shape": not two dictionaries, V and pi; "states": V or pi lacks a state
    world: str = ""
    values: tuple = ()              # ((state, value rounded to 6 decimals), ...)
    policy_ok: bool = False
    policy: tuple = ()              # ((state, action as Python shows it), ...) when policy_ok is False
    off: tuple | None = None        # the first state whose value is not the course's: (state, its value, the course's)
    wrong: tuple | None = None      # the first state whose action is not a best one: (state, its action, a best one)


def judge(T, gamma, rounds, value):
    """What `value_iteration(T, gamma, rounds)` gave back, judged against the course's own value iteration: None stays
    None, so the check can say that a return is missing."""
    if value is None:
        return None
    if not (isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], dict) and isinstance(value[1], dict)):
        return Solved("shape")
    V, pi = value
    states = sorted(T)
    if set(V) != set(T) or set(pi) != set(T):
        return Solved("states")
    try:
        values = tuple((s, round(float(V[s]), 6) + 0.0) for s in states)
    except (TypeError, ValueError):
        return Solved("shape")
    steps = history(T, gamma, rounds)
    right = steps[-1]["V"]
    off = next(((s, v, right[s]) for s, v in values if abs(v - right[s]) > 1e-6), None)
    wrong = None
    for s in states:
        qs = steps[-2]["Q"][s] if rounds > 0 else None
        best = [a for a in ACTIONS if qs and qs[a] > max(qs.values()) - 1e-9]
        if not (isinstance(pi[s], str) and pi[s] in best):
            wrong = (s, pi[s], best[0] if best else None)
            break
    policy = () if wrong is None else tuple((s, repr(pi[s])) for s in states)
    return Solved("ok", T.world.name, values, wrong is None, policy, off, wrong)


def same_solved(a, b) -> bool:
    """Right when the values agree within a millionth and both policies take a best action everywhere; a wrong policy
    only counts as the same as another one that is exactly as wrong."""
    if not (isinstance(a, Solved) and isinstance(b, Solved)):
        return a == b
    if a.kind != b.kind or a.world != b.world:
        return False
    if a.kind != "ok":
        return True
    close = all(sa == sb and abs(va - vb) <= 1e-6 for (sa, va), (sb, vb) in zip(a.values, b.values))
    return close and a.policy_ok == b.policy_ok and (a.policy_ok or a.policy == b.policy)


def _digits(a: float, b: float) -> int:
    """Enough decimals to tell two numbers apart: 2, or more when they agree to 2."""
    for d in (2, 3, 4, 6):
        if f"{a:.{d}f}" != f"{b:.{d}f}":
            return d
    return 6


def _plain(x: float, d: int) -> str:
    text = f"{x:.{d}f}"
    return text.lstrip("-") if float(text) == 0 else text


def _as_code(v) -> str:
    """An action the way code writes it; a number, also one of NumPy, as a plain number with 2 decimals at most."""
    if isinstance(v, str):
        return f'"{v}"'
    v = v.item() if hasattr(v, "item") else v
    if isinstance(v, float):
        return _plain(v, 2)
    return repr(v)


def show(solved) -> dict:
    """A judged answer in words, for the check's messages: where it first differs from the course's answer."""
    if not isinstance(solved, Solved) or solved.kind == "shape":
        return {"en": "something other than two dictionaries, `V` and `pi`",
                "nl": "iets anders dan twee dictionaries, `V` en `pi`"}
    if solved.kind == "states":
        return {"en": "a `V` or a `pi` without a value for every state of `T`",
                "nl": "een `V` of een `pi` zonder waarde voor elke state van `T`"}
    if solved.off is not None:
        s, got, want = solved.off
        d = _digits(got, want)
        return {"en": f"`V[{s}] = {_plain(got, d)}`, where it should be `{_plain(want, d)}`",
                "nl": f"`V[{s}] = {_plain(got, d)}`, waar het `{_plain(want, d)}` moet zijn"}
    if solved.wrong is not None:
        s, got, best = solved.wrong
        return {"en": f"the right values, but `pi[{s}] = {_as_code(got)}`, where `\"{best}\"` is a best action",
                "nl": f"de goede waarden, maar `pi[{s}] = {_as_code(got)}`, waar `\"{best}\"` een beste action is"}
    return {"en": "the right values, and a best action in every state",
            "nl": "de goede waarden, en een beste action in elke state"}


@dataclass
class ValueBox(FunctionBox):
    """value_iteration in a code box. The check calls it on each table and judges what it gives back (judge), so any
    best action counts where two are equally good, and a known mistake is known by what it does in every case."""
    judge: object = None

    def _run(self, fn, case):
        return self.judge(*case, super()._run(fn, case))


# The lesson is done when these are correct. "neighbor" and "markov" are choice questions in the lesson.
REQUIRED = ["value-iteration", "neighbor", "markov"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _v(q: str = "q = q + p * (r + gamma * V[nxt])", pi: str = "pi[s] = actions[np.argmax(qs)]") -> str:
    """value_iteration as the box holds it, with these two lines: the Q of step 3 of the recipe, and step 5."""
    return ("import numpy as np\n\n\n"
            "def value_iteration(T, gamma, rounds):\n"
            "    V = {}\n"
            "    for s in T:\n"
            "        V[s] = 0\n"
            "    pi = {}\n"
            "    for k in range(rounds):\n"
            "        new = {}\n"
            "        for s in T:\n"
            "            actions = []\n"
            "            qs = []\n"
            "            for a in T[s]:\n"
            "                q = 0\n"
            "                for p, nxt, r in T[s][a]:\n"
            f"                    {q}\n"
            "                actions.append(a)\n"
            "                qs.append(q)\n"
            "            new[s] = max(qs)\n"
            f"            {pi}\n"
            "        V = new\n"
            "    return V, pi\n")


_V_START = _v("q = q + ...", "pi[s] = ...")
_V = _v()

EXERCISES = {
    "value-iteration": ValueBox(
        name="value-iteration",
        function="value_iteration",
        start=_V_START,
        solution=_V,
        judge=judge,
        compare=same_solved,
        show=show,
        visible=[(T, GAMMA, 30)],
        hidden=[(T_SMALL, GAMMA, 1), (T_SMALL, GAMMA, 2), (T_SMALL, GAMMA, 30), (T_BRIDGE, 0.5, 20), (T, GAMMA, 3), (T, 0.0, 4)],
        first=[_DOTS],
        mistakes=[
            Mistake("no-gamma", wrong=_v("q = q + p * (r + V[nxt])"),
                    en="`{call}` gives {got}. Your Q counts a value one step later as much as a reward now. Multiply `V[nxt]` by `gamma`, as in the Bellman equation of *The idea*.",
                    nl="`{call}` geeft {got}. Je Q telt een waarde een stap later even zwaar als een reward nu. Vermenigvuldig `V[nxt]` met `gamma`, zoals in de Bellman equation van *Het idee*."),
            Mistake("no-p", wrong=_v("q = q + r + gamma * V[nxt]"),
                    en="`{call}` gives {got}. You add up all outcomes of an action in full, as if each of them happened. Weigh each one by its probability `p`: `p * (r + gamma * V[nxt])`.",
                    nl="`{call}` geeft {got}. Je telt alle uitkomsten van een action volledig op, alsof ze allemaal gebeuren. Weeg elke uitkomst met zijn kans `p`: `p * (r + gamma * V[nxt])`."),
            Mistake("p-only-reward", wrong=_v("q = q + p * r + gamma * V[nxt]"),
                    en="`{call}` gives {got}. `p * r + gamma * V[nxt]` weighs only the reward by its probability: `*` goes before `+`. Put the reward and the value in brackets, so that `p` weighs both.",
                    nl="`{call}` geeft {got}. `p * r + gamma * V[nxt]` weegt alleen de reward met zijn kans: `*` gaat voor `+`. Zet de reward en de waarde tussen haakjes, zodat `p` ze allebei weegt."),
            Mistake("last-only", wrong=_v("q = p * (r + gamma * V[nxt])"),
                    en="`{call}` gives {got}. `q = p * ...` throws away what `q` held, so only the last outcome of an action counts. Add each outcome to `q`: `q = q + ...`.",
                    nl="`{call}` geeft {got}. `q = p * ...` gooit weg wat er in `q` stond, dus telt alleen de laatste uitkomst van een action. Tel elke uitkomst bij `q` op: `q = q + ...`."),
            Mistake("own-value", wrong=_v("q = q + p * (r + gamma * V[s])"),
                    en="`{call}` gives {got}. You use the value of the state you are in, `V[s]`; the Bellman equation looks one step ahead, at the value of where the outcome lands: `V[nxt]`.",
                    nl="`{call}` geeft {got}. Je gebruikt de waarde van de state waar je bent, `V[s]`; de Bellman equation kijkt een stap vooruit, naar de waarde van waar de uitkomst landt: `V[nxt]`."),
            Mistake("no-reward", wrong=_v("q = q + p * gamma * V[nxt]"),
                    en="`{call}` gives {got}. Without the reward `r`, nothing ever brings a value above 0. Add the reward to the discounted value: `p * (r + gamma * V[nxt])`.",
                    nl="`{call}` geeft {got}. Zonder de reward `r` komt er nooit een waarde boven 0. Tel de reward op bij de verdisconteerde waarde: `p * (r + gamma * V[nxt])`."),
            Mistake("place-not-action", wrong=_v(pi="pi[s] = np.argmax(qs)"),
                    en="`{call}` gives {got}. `np.argmax(qs)` is the place of the largest Q in the list, a number; the policy holds the action at that place: `actions[np.argmax(qs)]`.",
                    nl="`{call}` geeft {got}. `np.argmax(qs)` is de plek van de grootste Q in de lijst, een getal; de policy bevat de action op die plek: `actions[np.argmax(qs)]`."),
            Mistake("value-not-action", wrong=_v(pi="pi[s] = max(qs)"),
                    en="`{call}` gives {got}. `max(qs)` is the largest Q itself, a number; the policy holds the action that gives it: `actions[np.argmax(qs)]`.",
                    nl="`{call}` geeft {got}. `max(qs)` is de grootste Q zelf, een getal; de policy bevat de action die hem geeft: `actions[np.argmax(qs)]`."),
            Mistake("argmax-q", wrong=_v(pi="pi[s] = actions[np.argmax(q)]"),
                    en="`{call}` gives {got}. `np.argmax(q)` looks at one number, the Q of the last action, so it always gives place 0, and the policy always `\"up\"`. Take the place of the largest number in the list `qs`: `actions[np.argmax(qs)]`.",
                    nl="`{call}` geeft {got}. `np.argmax(q)` kijkt naar één getal, de Q van de laatste action, dus geeft hij altijd plek 0, en de policy altijd `\"up\"`. Neem de plek van het grootste getal in de lijst `qs`: `actions[np.argmax(qs)]`."),
            Mistake("smallest", wrong=_v(pi="pi[s] = actions[np.argmin(qs)]"),
                    en="`{call}` gives {got}. `np.argmin` gives the place of the smallest Q: the worst action. The policy takes the action with the largest Q.",
                    nl="`{call}` geeft {got}. `np.argmin` geeft de plek van de kleinste Q: de slechtste action. De policy neemt de action met de grootste Q."),
            Mistake("last-action", wrong=_v(pi="pi[s] = a"),
                    en="`{call}` gives {got}. `a` is the last action the loop went through, `\"right\"`, whatever its Q. Take the action with the largest Q: `actions[np.argmax(qs)]`.",
                    nl="`{call}` geeft {got}. `a` is de laatste action waar de loop langs ging, `\"right\"`, wat zijn Q ook is. Neem de action met de grootste Q: `actions[np.argmax(qs)]`."),
            Mistake("two-values", matches=lambda got, text: isinstance(got, Solved) and got.kind in ("shape", "states"),
                    en="`{call}` gives {got}. Keep the line `return V, pi`: the check needs both, the values and the policy, with a value and an action for every state of `T`.",
                    nl="`{call}` geeft {got}. Houd de regel `return V, pi`: de check heeft ze allebei nodig, de waarden en de policy, met een waarde en een action voor elke state van `T`."),
            Mistake("index-pair", error=lambda e: isinstance(e, KeyError) and isinstance(e.args[0] if e.args else None, str),
                    en="With `{call}`, your function stops with an error: it looks up an action where `V` has states. `V[nxt]` is the value of the state the outcome lands in.",
                    nl="Met `{call}` stopt je functie met een fout: hij zoekt een action op waar `V` states heeft. `V[nxt]` is de waarde van de state waar de uitkomst landt."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "empty": {"en": "Replace the two `...`: the Q of an action, step 3 of the recipe, and the action of the policy, step 5.",
                      "nl": "Vervang de twee `...`: de Q van een action, stap 3 van het recept, en de action van de policy, stap 5."},
            "arguments": {"en": "The check calls `{call}`, with a table, a discount factor and a number of rounds, and the `def` line of your function has another number of names between its parentheses. Keep `(T, gamma, rounds)`.",
                          "nl": "De check roept `{call}` aan, met een tabel, een discount factor en een aantal rondes, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(T, gamma, rounds)`."},
            "slow": {"en": "Your function is still not done after 2 seconds. Does every loop go over a list or a table, with `for`?",
                     "nl": "Je functie is na 2 seconden nog niet klaar. Gaat elke loop over een lijst of een tabel, met `for`?"},
            "wrong": {"en": "`{call}` gives {got}. Compare your two lines with steps 3 and 5 of the recipe.",
                      "nl": "`{call}` geeft {got}. Vergelijk je twee regels met stap 3 en 5 van het recept."},
            "almost": {"en": "Right for GRID after 30 rounds. But `{call}` gives {got}. Does your function work for every table, every discount factor and every number of rounds?",
                       "nl": "Goed voor GRID na 30 rondes. Maar `{call}` geeft {got}. Werkt je functie voor elke tabel, elke discount factor en elk aantal rondes?"},
            "correct": {"en": "Your value iteration is right in all {n} cases: the values, and a policy with a best action in every state.",
                        "nl": "Je value iteration klopt in alle {n} gevallen: de waarden, en een policy met een beste action in elke state."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("value-iteration", _V_START, "empty", None),
    ("value-iteration", _v("q = q + ..."), "wrong", "dots-left"),
    ("value-iteration", _v(pi="pi[s] = ..."), "wrong", "dots-left"),
    ("value-iteration", _v("q = q + p * (r + V[nxt])"), "wrong", "no-gamma"),
    ("value-iteration", _v("q = q + r + gamma * V[nxt]"), "wrong", "no-p"),
    ("value-iteration", _v("q = q + p * r + gamma * V[nxt]"), "wrong", "p-only-reward"),
    ("value-iteration", _v("q = p * (r + gamma * V[nxt])"), "wrong", "last-only"),
    ("value-iteration", _v("q = q + p * (r + gamma * V[s])"), "wrong", "own-value"),
    ("value-iteration", _v("q = q + p * gamma * V[nxt]"), "wrong", "no-reward"),
    ("value-iteration", _v(pi="pi[s] = np.argmax(qs)"), "wrong", "place-not-action"),
    ("value-iteration", _v(pi="pi[s] = max(qs)"), "wrong", "value-not-action"),
    ("value-iteration", _v(pi="pi[s] = actions[np.argmax(q)]"), "wrong", "argmax-q"),
    ("value-iteration", _v(pi="pi[s] = actions[np.argmin(qs)]"), "wrong", "smallest"),
    ("value-iteration", _v(pi="pi[s] = a"), "wrong", "last-action"),
    ("value-iteration", _v("q = q + p * (r + gamma * V[a])"), "error", "index-pair"),
    ("value-iteration", _V.replace("    return V, pi\n", "    return V\n"), "wrong", "two-values"),
    # other right answers: the reward and the value apart, and the last of two equally good actions
    ("value-iteration", _v("q = q + p * r + p * gamma * V[nxt]"), "correct", None),
    ("value-iteration", _v(pi="pi[s] = actions[len(qs) - 1 - int(np.argmax(qs[::-1]))]"), "correct", None),
    # right after 30 rounds on GRID, where the values have settled, but it updates V while it goes through the states
    ("value-iteration", _V.replace("            new[s] = max(qs)\n", "            new[s] = max(qs)\n            V[s] = new[s]\n"), "almost", None),
]
