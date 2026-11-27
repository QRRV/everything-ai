# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rl-kinds-of-rl: the methods of the module on one map. Two of its questions, does a method use a
model of the environment and does it update from whole episodes or from its own estimates, give the map's corners:
dynamic programming, Monte Carlo and TD learning, and the corner of every path, measured on the random walk. A
policy-based agent on three machines that pay 1 coin with chances 0.2, 0.5 and 0.8 learns the chance of pulling each
machine instead of values, beside a value-based agent with ε-greedy. The sort of agents by their model,
nudge, the function the learner writes, its picture, the play block and the demo.

The map after Sutton and Barto (Reinforcement Learning: An Introduction, 2nd edition, 2018, section 8.13); the policy
-based agent is the linear reward-inaction scheme of learning automata (Narendra and Thathachar, 1974). The code, the
runs and the pictures are the course's own."""
import ast
import html
import math
import random
import time
from functools import lru_cache

from everything_ai import chart, live, yours
from everything_ai.check import FunctionBox, Mistake, Sort

LANGS = ("en", "nl")

# --- the random walk of the lesson TD(0): five states A to E between two ends, the random policy -------------------------
LEFT, RIGHT, CENTER = 0, 6, 3
STATES = range(1, 6)
TRUE = [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 0]   # the chance to end on the right, from each state
TOL = 0.01


def sweep(V: list) -> list:
    """One sweep of policy evaluation with the model: every state at once, from the old values, both next states with
    chance 1/2 each, a reward of 1 for the step onto the right end."""
    new = list(V)
    for s in STATES:
        new[s] = 0.5 * (0 + V[s - 1]) + 0.5 * ((1 if s + 1 == RIGHT else 0) + V[s + 1])
    return new


def evaluate(sweeps: int) -> list:
    V = [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0]
    for _ in range(sweeps):
        V = sweep(V)
    return V


def worst(V: list) -> float:
    """The largest distance of an estimate from its true value."""
    return max(abs(V[s] - TRUE[s]) for s in STATES)


def sweeps_needed(tol: float = TOL) -> int:
    k = 0
    while worst(evaluate(k)) >= tol:
        k += 1
    return k


def every_path(depth: int) -> tuple:
    """Every walk from C, each with its chance 1/2 per step: how many walks end within `depth` steps, the chance that a
    walk is longer, and the chance to end on the right within depth steps."""
    counts, finished, right = {CENTER: 1}, 0, 0.0
    for n in range(1, depth + 1):
        new = {}
        for s, c in counts.items():
            for t in (s - 1, s + 1):
                if t in (LEFT, RIGHT):
                    finished += c
                    right += c / 2 ** n if t == RIGHT else 0
                else:
                    new[t] = new.get(t, 0) + c
        counts = new
    return finished, sum(counts.values()) / 2 ** depth, right


PATH_DEPTH = 33   # every walk of up to 33 steps: V(C) is then known to within about 0.01


# --- three machines: a policy-based and a value-based agent --------------------------------------------------------------
PAYS = (0.2, 0.5, 0.8)   # the chance that each machine pays 1 coin; else it pays nothing
BEST = 2
ETA = 0.1
EPSILON = 0.1
PULLS = 1000
RUNS = 200
ETAS = (0.01, 0.05, 0.1, 0.2, 0.5)   # the demo's menu
LOOKS = [0, 10, 50, 100, 200, 500, 1000]
WINDOW = 10
SURE = 0.99   # a run has settled on a machine once its chance is above this


def nudge(p, action, reward, eta):
    """The course's policy-based update: after a pull that pays, every chance moves eta of the way toward its target,
    1 for the machine pulled and 0 for the others; after a pull that pays nothing, nothing changes."""
    if reward == 0:
        return p
    new = []
    for a in range(len(p)):
        if a == action:
            target = 1
        else:
            target = 0
        new.append(p[a] + eta * (target - p[a]))
    return new


def pick(p, rnd) -> int:
    """A machine drawn with the chances in p."""
    u, total = rnd.random(), 0.0
    for a, x in enumerate(p):
        total += x
        if u < total:
            return a
    return len(p) - 1


class _Bad(Exception):
    """The learner's nudge gave something that is not a list of chances."""


def _chances(p, n: int) -> list:
    """p as a list of n chances, or _Bad: numbers between 0 and 1 that add up to 1."""
    if not isinstance(p, (list, tuple)) or len(p) != n:
        raise _Bad
    out = []
    for x in p:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not -1e-9 <= x <= 1 + 1e-9:
            raise _Bad
        out.append(float(x))
    if abs(sum(out) - 1) > 1e-6:
        raise _Bad
    return out


def _value_step(Q, n, rnd, epsilon=EPSILON) -> int:
    """The value-based agent: ε-greedy on the mean of what each machine paid so far, ties to the first machine."""
    if rnd.random() < epsilon:
        return int(rnd.random() * len(Q))
    best, top = 0, Q[0]
    for a in range(1, len(Q)):
        if Q[a] > top:
            best, top = a, Q[a]
    return best


def run(nudge_fn=nudge, eta: float = ETA, show=None, limit: float = 60.0, runs: int = RUNS) -> dict | str:
    """Both agents on the three machines, `runs` runs of PULLS pulls, run k of each on random seed k: per pull, the
    share of runs that pulled the best machine, and the reward; where each run of the policy-based agent settled; and
    the chances and the estimates of run 1 at every stop of LOOKS. `show(n, run)` hears every 10th pull. "nudge" when
    nudge_fn stops with an error or gives no chances, "slow" after `limit` seconds."""
    k_ = len(PAYS)
    P = [[1 / k_] * k_ for _ in range(runs)]
    Q = [[0.0] * k_ for _ in range(runs)]
    N = [[0] * k_ for _ in range(runs)]
    rp = [random.Random(k) for k in range(runs)]
    rv = [random.Random(k) for k in range(runs)]
    out = {"eta": eta, "runs": runs, "share": {"policy": [], "value": []}, "reward": {"policy": 0, "value": 0},
           "first": {0: (list(P[0]), list(Q[0]))}, "now": {"p": P[0], "q": Q[0]}}
    begin = time.perf_counter()
    for t in range(1, PULLS + 1):
        best_p = best_v = 0
        for k in range(runs):
            a = pick(P[k], rp[k])
            r = 1 if rp[k].random() < PAYS[a] else 0
            try:
                P[k] = _chances(nudge_fn(list(P[k]), a, r, eta), k_)
            except Exception:   # noqa: BLE001   the demo says which function to look at
                return "nudge"
            best_p += a == BEST
            out["reward"]["policy"] += r
            b = _value_step(Q[k], N[k], rv[k])
            r = 1 if rv[k].random() < PAYS[b] else 0
            N[k][b] += 1
            Q[k][b] += (r - Q[k][b]) / N[k][b]
            best_v += b == BEST
            out["reward"]["value"] += r
        out["share"]["policy"].append(best_p / runs)
        out["share"]["value"].append(best_v / runs)
        out["now"] = {"p": P[0], "q": Q[0]}
        if t in LOOKS:
            out["first"][t] = (list(P[0]), list(Q[0]))
        if show is not None and (t % 10 == 0 or t == 1):
            show(t, out)
        if time.perf_counter() - begin > limit:
            return "slow"
    out["reward"] = {m: v / (runs * PULLS) for m, v in out["reward"].items()}
    out["settled"] = [next((a for a in range(k_) if p[a] > SURE), None) for p in P]
    out["last"] = {m: sum(out["share"][m][-100:]) / 100 for m in ("policy", "value")}
    return out


@lru_cache(maxsize=None)
def demo_run(eta: float) -> dict:
    """The demo with the course's nudge: what the text after the demo says."""
    return run(eta=eta)


def play_run() -> dict:
    return demo_run(ETA)


def smooth(curve: list) -> list:
    out = []
    for e in range(1, len(curve) + 1):
        part = curve[max(0, e - WINDOW):e]
        out.append(sum(part) / len(part))
    return out


def settled(r: dict) -> dict:
    """How many runs of the policy-based agent settled on each machine, and how many on none yet."""
    out = {a: 0 for a in range(len(PAYS))} | {None: 0}
    for a in r["settled"]:
        out[a] += 1
    return out


# --- drawing ------------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem"
_HALO = "paint-order:stroke;stroke:var(--ea-page);stroke-width:3px;stroke-linejoin:round"


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


def eta_text(e: float, lang: str) -> str:
    text_ = f"{e:g}"
    return text_.replace(".", ",") if lang == "nl" else text_


def eta_options(lang: str) -> dict:
    return {eta_text(e, lang): e for e in ETAS}


def _svg(body: str, width: int, height: int, alt: str) -> str:
    svg = (f'<svg viewBox="0 0 {width} {height}" aria-hidden="true" style="flex:1 1 0;width:100%;'
           f'min-width:min(100%,{width}px);max-width:{width}px;height:auto;font-family:var(--ea-font-ui)">{body}</svg>')
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(alt)}" style="margin:0.25rem 0 0.5rem;'
            f'max-width:min(100%,{width}px);display:flex">{svg}</figure>')


def _t(x, y, words, size=11.5, weight=400, fill="var(--ea-ink)", anchor="middle", extra="") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" style="font-size:{size}px;font-weight:{weight};fill:{fill};'
            f'text-anchor:{anchor};{extra}">{html.escape(words)}</text>')


# the map: where each method sits, in the order of the play block
METHODS = [   # (name, corner, what one update looks at, the lesson that built it)
    ("policy evaluation", "dp", "dp", "rl-policy-evaluation"),
    ("policy iteration", "dp", "improve", "rl-policy-iteration"),
    ("value iteration", "dp", "vi", "rl-value-iteration"),
    ("prediction", "mc", "mc", "rl-monte-carlo-prediction"),
    ("control", "mc", "mcq", "rl-monte-carlo-control"),
    ("TD(0)", "td", "td", "rl-td-prediction"),
    ("Q-learning", "td", "q", "rl-q-learning"),
    ("SARSA", "td", "sarsa", "rl-sarsa"),
]
STOPS = list(range(1, len(METHODS) + 1))
MAP_WORDS = {
    "en": {"cols": ("one step + estimate", "the whole episode"), "rows": ("all next states", "one next state"),
           "corners": {"dp": "dynamic programming", "mc": "Monte Carlo", "td": "TD learning", "all": "every path"},
           "every": ("every walk to its end,", "each with its chance:", "only for tiny tasks"),
           "alt": "The map of RL methods: on the left one step and then an estimate, on the right the whole episode; at the top all next states, with their chances from the model, at the bottom the one next state that happened. Dynamic programming top left, TD learning bottom left, Monte Carlo bottom right, every path top right."},
    "nl": {"cols": ("één stap + schatting", "de hele episode"), "rows": ("alle volgende states", "één volgende state"),
           "corners": {"dp": "dynamic programming", "mc": "Monte Carlo", "td": "TD learning", "all": "alle paden"},
           "every": ("elke wandeling tot", "het einde, elk met de", "eigen kans: alleen voor", "heel kleine taken"),
           "alt": "De kaart van RL-methoden: links één stap en dan een schatting, rechts de hele episode; boven alle volgende states, met de kansen uit het model, onder de ene volgende state die gebeurde. Dynamic programming linksboven, TD learning linksonder, Monte Carlo rechtsonder, alle paden rechtsboven."},
}


def map_svg(stop: int | None, lang: str) -> str:
    """The map with the method of stop `stop` (1 to 8) marked: an orange dot and ink, the others in gray."""
    w = MAP_WORDS[lang if lang in LANGS else "en"]
    W, H, x0, y0, cw, ch = 330, 252, 24, 26, 153, 112
    parts = []
    for j, words in enumerate(w["cols"]):
        parts.append(_t(x0 + cw * j + cw / 2, 15, words, size=11.5, fill="var(--ea-ink-2)"))
    for i, words in enumerate(w["rows"]):
        cy = y0 + ch * i + ch / 2
        parts.append(f'<text x="13" y="{cy:.1f}" transform="rotate(-90 13 {cy:.1f})" style="font-size:11.5px;'
                     f'fill:var(--ea-ink-2);text-anchor:middle">{html.escape(words)}</text>')
    where = {"dp": (0, 0), "all": (1, 0), "td": (0, 1), "mc": (1, 1)}
    mine = METHODS[stop - 1][0] if stop in STOPS else None
    for corner, (j, i) in where.items():
        x, y = x0 + cw * j, y0 + ch * i
        fill = "var(--ea-inset)" if corner == "all" else "var(--ea-page)"
        parts.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" style="fill:{fill};stroke:var(--ea-line);stroke-width:1"/>')
        parts.append(_t(x + 9, y + 20, w["corners"][corner], size=12.5, weight=700, anchor="start"))
        if corner == "all":
            for k, line in enumerate(w["every"]):
                parts.append(_t(x + 9, y + 44 + 15 * k, line, size=11.5, fill="var(--ea-ink-2)", anchor="start",
                                extra="font-style:italic"))
            continue
        names = [m[0] for m in METHODS if m[1] == corner]
        for k, name in enumerate(names):
            yy = y + 45 + 20 * k
            if name == mine:
                parts.append(f'<circle cx="{x + 14}" cy="{yy - 4}" r="4.5" style="fill:var(--ea-param)"/>')
            parts.append(_t(x + 25, yy, name, size=12, weight=650 if name == mine else 400,
                            fill="var(--ea-ink)" if name == mine else "var(--ea-ink-3)", anchor="start"))
    return _svg("".join(parts), W, H, w["alt"])


BACKUP_WORDS = {
    "en": {"title": "What one update looks at",
           "dp": ("V(C) ← ½ (0 + V(B)) + ½ (0 + V(D))", "both next states, with their chances from the model"),
           "improve": ("π(C) ← the action with the larger 0 + V(·)", "both actions with the model; the policy takes the best"),
           "vi": ("V(C) ← max(0 + V(B), 0 + V(D))", "the best action, with its next state from the model"),
           "mc": ("V(C) ← toward G = +1", "one walk to its end: the return of the whole walk"),
           "mcq": ("Q(C, right) ← toward G = +1", "one walk to its end, which starts with right"),
           "td": ("V(C) ← toward 0 + V(D)", "one step that happened, then its estimate of D"),
           "q": ("Q(C, right) ← toward 0 + max Q(D, ·)", "one step, then the best action in D"),
           "sarsa": ("Q(C, right) ← toward 0 + Q(D, A′)", "one step, then the action A′ it takes in D"),
           "left": "left", "right": "right", "max": "max",
           "alt": "The random walk: the states A to E between two ends, with C, whose value is updated, outlined in orange, and in ink what its update looks at."},
    "nl": {"title": "Waar één update naar kijkt",
           "dp": ("V(C) ← ½ (0 + V(B)) + ½ (0 + V(D))", "beide volgende states, met de kansen uit het model"),
           "improve": ("π(C) ← de action met de grootste 0 + V(·)", "beide actions met het model; de policy neemt de beste"),
           "vi": ("V(C) ← max(0 + V(B), 0 + V(D))", "de beste action, met de volgende state uit het model"),
           "mc": ("V(C) ← richting G = +1", "één wandeling tot het einde: de return van de hele wandeling"),
           "mcq": ("Q(C, right) ← richting G = +1", "één wandeling tot het einde, beginnend met right"),
           "td": ("V(C) ← richting 0 + V(D)", "één stap die gebeurde, dan de schatting van D"),
           "q": ("Q(C, right) ← richting 0 + max Q(D, ·)", "één stap, dan de beste action in D"),
           "sarsa": ("Q(C, right) ← richting 0 + Q(D, A′)", "één stap, dan de action A′ in D"),
           "left": "left", "right": "right", "max": "max",
           "alt": "De willekeurige wandeling: de states A tot E tussen twee uiteinden, met C, waarvan de waarde wordt bijgewerkt, oranje omlijnd, en in inkt waar de update naar kijkt."},
}
_WALK_PATH = [3, 4, 3, 4, 5, 6]   # the walk drawn for Monte Carlo: C, D, C, D, E, and off the right end


def backup_svg(kind: str, lang: str) -> str:
    """The random walk with C, whose value is updated, outlined in orange, and what its update looks at in ink."""
    w = BACKUP_WORDS[lang if lang in LANGS else "en"]
    W, H, cell, x0, y0 = 330, 196, 40, 25, 56
    parts = [_t(2, 15, w["title"], size=12.5, weight=600, anchor="start")]
    names = ["0", "A", "B", "C", "D", "E", "+1"]

    def cx(s):
        return x0 + cell * s + cell / 2

    for s in range(7):
        end = s in (LEFT, RIGHT)
        parts.append(f'<rect x="{x0 + cell * s + 3}" y="{y0}" width="{cell - 6}" height="30" rx="4" style="fill:'
                     f'{"var(--ea-inset)" if end else "var(--ea-page)"};stroke:var(--ea-line);stroke-width:1"/>')
        parts.append(_t(cx(s), y0 + 20, names[s], size=12.5, weight=400 if end else 700,
                        fill="var(--ea-ink-2)" if end else "var(--ea-ink)"))
    parts.append(f'<rect x="{x0 + cell * CENTER + 1}" y="{y0 - 2}" width="{cell - 2}" height="34" rx="5" '
                 f'style="fill:none;stroke:var(--ea-param);stroke-width:2.6"/>')

    def arc(a, b, words):
        x1, x2 = cx(a), cx(b)
        mid = (x1 + x2) / 2
        parts.append(f'<path d="M{x1:.1f},{y0 - 4} Q{mid:.1f},{y0 - 30} {x2:.1f},{y0 - 4}" style="fill:none;'
                     f'stroke:var(--ea-ink);stroke-width:1.8"/>')
        parts.append(f'<circle cx="{x2:.1f}" cy="{y0 - 4}" r="2.8" style="fill:var(--ea-ink)"/>')
        if words:
            parts.append(_t(mid, y0 - 19, words, size=11.5, weight=600, extra=_HALO))

    def ring(s):
        parts.append(f'<rect x="{x0 + cell * s + 1}" y="{y0 - 2}" width="{cell - 2}" height="34" rx="5" '
                     f'style="fill:none;stroke:var(--ea-ink);stroke-width:1.6"/>')

    def arrow(s, way, bold):
        y, x1 = y0 + 46, cx(s) + 3 * way
        x2 = x1 + 15 * way
        parts.append(f'<line x1="{x1:.1f}" y1="{y}" x2="{x2:.1f}" y2="{y}" style="stroke:var(--ea-ink);'
                     f'stroke-width:{2 if bold else 1.2}"/>')
        parts.append(f'<polygon points="{x2 + 5 * way:.1f},{y} {x2:.1f},{y - 4} {x2:.1f},{y + 4}" style="fill:var(--ea-ink)"/>')

    if kind in ("dp", "vi", "improve"):
        arc(CENTER, CENTER - 1, "½" if kind == "dp" else w["left"])
        arc(CENTER, CENTER + 1, "½" if kind == "dp" else w["right"])
        ring(CENTER - 1)
        ring(CENTER + 1)
    elif kind in ("td", "q", "sarsa"):
        arc(CENTER, CENTER + 1, "" if kind == "td" else w["right"])
        ring(CENTER + 1)
        if kind == "q":
            arrow(CENTER + 1, -1, False)
            arrow(CENTER + 1, 1, False)
            parts.append(_t(cx(CENTER + 1), y0 + 66, w["max"], size=11.5, weight=600))
        elif kind == "sarsa":
            arrow(CENTER + 1, 1, True)
            parts.append(_t(cx(CENTER + 1) + 27, y0 + 50, "A′", size=11.5, weight=600, anchor="start"))
    else:   # a whole walk, step by step downward
        y = y0 + 42
        pts = " ".join(f"{cx(s):.1f},{y + 10 * k:.1f}" for k, s in enumerate(_WALK_PATH))
        parts.append(f'<polyline points="{pts}" style="fill:none;stroke:var(--ea-ink);stroke-width:2;stroke-linejoin:round"/>')
        for k, s in enumerate(_WALK_PATH):
            parts.append(f'<circle cx="{cx(s):.1f}" cy="{y + 10 * k:.1f}" r="2.6" style="fill:var(--ea-ink)"/>')
    first, second = w[kind]
    parts.append(_t(2, H - 30, first, size=12, weight=600, anchor="start"))
    parts.append(_t(2, H - 12, second, size=11.5, fill="var(--ea-ink-2)", anchor="start"))
    return _svg("".join(parts), W, H, w["alt"])


_PLAY_TEXT = {
    "en": [
        "**Policy evaluation**, from *What is a policy worth? Policy evaluation*. Model-based: it updates with the chances of every next state, from the table `T`. Value-based: it works out $V$ of a given policy. One step and its own estimates, over all next states. On- or off-policy does not apply here: it never takes a step.",
        "**Policy iteration**, from *Better and better: policy iteration*. Model-based and value-based: it evaluates a policy with the table `T`, then takes in every state the action with the largest $Q$. One step and its own estimates, over all next states; it never takes a step itself.",
        "**Value iteration**, from *When steps can slip: value iteration*. Model-based and value-based: every update takes the best action, with the chances of all its next states from `T`. One step and its own estimates; it never takes a step itself.",
        "**Monte Carlo prediction**, from *Learning from whole games: Monte Carlo prediction*. Model-free: it needs only walks that happened. Value-based: it estimates $V$. It waits for the whole episode and uses its return. On-policy: it estimates the values of the policy that makes the walks.",
        "**Monte Carlo control**, from *Playing better from experience: Monte Carlo control*. Model-free and value-based: it learns $Q$ from blackjack hands. Whole episodes: after each hand, the return from every step on. On-policy: it improves the policy it plays with.",
        "**TD(0)**, from *Learning after every step: TD(0)*. Model-free and value-based. After every step it updates from its own estimate of the next state. On-policy: it learns the values of the policy it follows.",
        "**Q-learning**, from *Learning from your own steps: Q-learning*. Model-free and value-based; after every step, from its own estimate of the best action in the next state. Off-policy: it learns the values of always taking the best action, while it follows ε-greedy.",
        "**SARSA**, from *Careful or bold: SARSA and Q-learning*. Model-free and value-based; after every step, from its own estimate of the action it takes next. On-policy: the one question of the four on which it differs from Q-learning.",
    ],
    "nl": [
        "**Policy evaluation**, uit *Wat is een policy waard? Policy evaluation*. Model-based: werkt bij met de kansen van elke volgende state, uit de tabel `T`. Value-based: rekent $V$ uit van een gegeven policy. Eén stap en de eigen schattingen, over alle volgende states. On- of off-policy is hier niet van toepassing: de methode zet nooit een stap.",
        "**Policy iteration**, uit *Steeds beter: policy iteration*. Model-based en value-based: evalueert een policy met de tabel `T`, en neemt dan in elke state de action met de grootste $Q$. Eén stap en de eigen schattingen, over alle volgende states; de methode zet zelf nooit een stap.",
        "**Value iteration**, uit *Als stappen kunnen uitglijden: value iteration*. Model-based en value-based: elke update neemt de beste action, met de kansen van alle volgende states uit `T`. Eén stap en de eigen schattingen; de methode zet zelf nooit een stap.",
        "**Monte Carlo prediction**, uit *Leren van hele potjes: Monte Carlo prediction*. Model-free: heeft alleen wandelingen nodig die echt gebeurden. Value-based: schat $V$. Wacht op de hele episode en gebruikt de return. On-policy: schat de waarden van de policy die de wandelingen maakt.",
        "**Monte Carlo control**, uit *Beter spelen door ervaring: Monte Carlo control*. Model-free en value-based: leert $Q$ uit gespeelde handen blackjack. Hele episodes: na elke hand de return vanaf elke stap. On-policy: verbetert de policy waarmee de agent speelt.",
        "**TD(0)**, uit *Leren na elke stap: TD(0)*. Model-free en value-based. Werkt na elke stap bij vanuit de eigen schatting van de volgende state. On-policy: leert de waarden van de policy die de agent volgt.",
        "**Q-learning**, uit *Leren van je eigen stappen: Q-learning*. Model-free en value-based; na elke stap, vanuit de eigen schatting van de beste action in de volgende state. Off-policy: leert de waarden van altijd de beste action nemen, terwijl de agent ε-greedy volgt.",
        "**SARSA**, uit *Voorzichtig of gedurfd: SARSA en Q-learning*. Model-free en value-based; na elke stap, vanuit de eigen schatting van de action die de agent als volgende neemt. On-policy: de enige vraag van de vier waarop SARSA van Q-learning verschilt.",
    ],
}


def play_frame(stop: int, lang: str) -> tuple:
    """The play block at one method: the map with the method marked, beside what one of its updates looks at on the
    random walk, and a sentence with its answers to the four questions."""
    lang = lang if lang in LANGS else "en"
    n = min(STOPS, key=lambda k: abs(k - int(stop)))
    body = live.beside((map_svg(n, lang), 330), (backup_svg(METHODS[n - 1][2], lang), 330))
    return body, _PLAY_TEXT[lang][n - 1]


# --- the bandit: pictures ------------------------------------------------------------------------------------------------

BANDIT_WORDS = {
    "en": {"title": "Run 1 after {n} {word}", "one": "pull", "many": "pulls", "x": "chance that the machine pays",
           "y": "chance or estimate", "curve": "Mean of 200 runs", "pulls": "pulls", "share": "share on the best machine",
           "policy": "policy-based (nudge)", "value": "value-based (ε-greedy)",
           "bars_alt": "For each of the three machines, a filled orange bar: the chance that the policy-based agent pulls it; an open orange bar: the value-based agent's estimate of what it pays; and a gray line at the chance it really pays.",
           "curve_alt": "The share of the 200 runs that pulled the best machine, after every pull: the policy-based agent solid, the value-based agent dashed.",
           "key": "Filled: the chance that the policy-based agent pulls the machine. Open: what the value-based agent estimates it pays. Gray line: what it really pays."},
    "nl": {"title": "Run 1 na {n} {word}", "one": "keer trekken", "many": "keer trekken", "x": "kans dat de automaat betaalt",
           "y": "kans of schatting", "curve": "Gemiddelde van 200 runs", "pulls": "keer trekken", "share": "deel aan de beste automaat",
           "policy": "policy-based (nudge)", "value": "value-based (ε-greedy)",
           "bars_alt": "Voor elk van de drie automaten een gevulde oranje balk: de kans dat de policy-based agent eraan trekt; een open oranje balk: wat de value-based agent schat dat de automaat betaalt; en een grijze lijn bij de kans dat de automaat echt betaalt.",
           "curve_alt": "Het deel van de 200 runs dat aan de beste automaat trok, na elke keer trekken: de policy-based agent doorgetrokken, de value-based agent gestreept.",
           "key": "Gevuld: de kans dat de policy-based agent aan de automaat trekt. Open: wat de value-based agent schat dat de automaat betaalt. Grijze lijn: wat de automaat echt betaalt."},
}


def _chart(panels, alt, lang, width=330, height=240) -> str:
    out = chart.figure(panels, alt=alt, width=width, height=height)
    return chart.with_decimal_comma(out) if lang == "nl" else out


def bars_svg(p, q, n: int, lang: str) -> str:
    """The chances of the policy-based agent beside the estimates of the value-based one, for each machine."""
    w = BANDIT_WORDS[lang if lang in LANGS else "en"]
    word = w["one"] if n == 1 else w["many"]
    panel = chart.Panel((0, 3), (0, 1), title=w["title"].format(n=number(n, 0, lang), word=word), xlabel=w["x"], ylabel=w["y"],
                        xticks=[0.5, 1.5, 2.5], xtick_labels=[number(x, 1, lang) for x in PAYS], yticks=[0, 0.25, 0.5, 0.75, 1])
    for a in range(len(PAYS)):
        x = a + 0.5
        panel.rect(x - 0.34, 0, x - 0.03, max(0.0, p[a]), role="param", solid=True)
        panel.rect(x + 0.03, 0, x + 0.34, max(0.0, min(1.0, q[a])), role="param")
        panel.line([x + 0.01, x + 0.36], [PAYS[a], PAYS[a]], "ink-3", 1.6)
    return _chart([panel], w["bars_alt"], lang, height=230) + _note(w["key"])


def _curve_panel(r: dict, n: int, lang: str) -> chart.Panel:
    w = BANDIT_WORDS[lang]
    panel = chart.Panel((0, PULLS), (0, 1), title=w["curve"], xlabel=w["pulls"], ylabel=w["share"],
                        yticks=[0, 0.25, 0.5, 0.75, 1])
    xs = list(range(1, len(r["share"]["value"]) + 1))
    if xs:
        panel.line(xs, smooth(r["share"]["value"]), "ink", 1.6, dashed=True)
        panel.line(xs, smooth(r["share"]["policy"]), "ink", 2.2)
    if 1 <= n <= len(xs):
        panel.line([n, n], [0, 1], "axis", 1.2)
    return panel


def curve_key(lang: str) -> list:
    w = BANDIT_WORDS[lang if lang in LANGS else "en"]
    return [("ink", False, w["policy"]), ("ink", True, w["value"])]


def chart_points(r: dict, method: str, start: int, end: int) -> list:
    line = smooth(r["share"][method][:end])
    return [(t, line[t - 1]) for t in range(max(1, start), end + 1)]


def now_picture(r: dict, n: int, lang: str) -> str:
    """Beside the chart of the demo: the chances and the estimates of run 1 as they are now."""
    return bars_svg(r["now"]["p"], r["now"]["q"], n, lang if lang in LANGS else "en")


DEMO_TEXT = {
    "en": {"title": "Your runs: share on the best machine", "x": "pulls", "y": "share on the best machine",
           "alt": "The share of the 200 runs that pulled the best machine, after every pull so far: the policy-based agent solid, the value-based agent dashed.",
           "eta": "η of the policy-based agent",
           "rewind": "After {n} {word}: {p} of the 200 runs pulled the best machine with the policy-based agent, and {v} with the value-based agent.",
           "say0": "Before the first pull, each machine has chance 1/3 and every estimate is 0."},
    "nl": {"title": "Je runs: deel aan de beste automaat", "x": "keer trekken", "y": "deel aan de beste automaat",
           "alt": "Het deel van de 200 runs dat aan de beste automaat trok, na elke keer trekken tot nu toe: de policy-based agent doorgetrokken, de value-based agent gestreept.",
           "eta": "η van de policy-based agent",
           "rewind": "Na {n} {word}: {p} van de 200 runs trokken met de policy-based agent aan de beste automaat, en {v} met de value-based agent.",
           "say0": "Bij de start heeft elke automaat kans 1/3 en is elke schatting 0."},
}


def rewind_frame(r: dict, n: int, lang: str) -> tuple:
    """A stop of the rewind after the demo: the chart with a line at n, beside the bars of run 1 then, and a sentence."""
    lang = lang if lang in LANGS else "en"
    w, bw = DEMO_TEXT[lang], BANDIT_WORDS[lang]
    n = min(r["first"], key=lambda k: abs(k - int(n)))
    curve = _chart([_curve_panel(r, n, lang)], bw["curve_alt"], lang) + live.key_html(curve_key(lang))
    p, q = r["first"][n]
    if n == 0:
        say = w["say0"]
    else:
        word = bw["one"] if n == 1 else bw["many"]
        say = w["rewind"].format(n=number(n, 0, lang), word=word, p=round(r["share"]["policy"][n - 1] * RUNS),
                                 v=round(r["share"]["value"][n - 1] * RUNS))
    return live.rewound(curve, [bars_svg(p, q, n, lang)], width=330), say


# --- the picture of the learner's nudge -----------------------------------------------------------------------------------

PICTURE_SECONDS = 3
_STOPPED = object()


def _call(make, seconds: float = PICTURE_SECONDS):
    import contextlib
    import io
    from everything_ai.check import Timeout, with_time_limit
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(make, max(seconds, 0.05))
    except Timeout:
        return _STOPPED
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def one_run(fn=nudge, eta: float = ETA, seed: int = 0, pulls: int = PULLS) -> dict:
    """Run `seed` of the policy-based agent with `fn` as its update: its chances after every pull, and what it pulled."""
    rnd = random.Random(seed)
    p = [1 / len(PAYS)] * len(PAYS)
    chances, pulled = [list(p)], []
    for _ in range(pulls):
        a = pick(p, rnd)
        r = 1 if rnd.random() < PAYS[a] else 0
        p = _chances(fn(list(p), a, r, eta), len(PAYS))
        chances.append(p)
        pulled.append(a)
    return {"chances": chances, "pulled": pulled}


_PICTURE_TEXT = {
    "en": {"title": "Your policy-based agent in run 1",
           "head": "Your `nudge` in run 1 of the demo, with η = 0.1: the chance of pulling each machine after every pull.",
           "say": "After 1,000 pulls the chances are {c}; in the last 100 pulls it pulled the best machine {b} times.",
           "none": "Your `nudge` stops with an error, takes too long, or gives something other than chances between 0 and 1 that add up to 1, so nothing can be drawn yet.",
           "x": "pulls", "y": "chance of pulling", "machine": "pays {x}",
           "alt": "Three lines, one per machine: the chance that your agent pulls it, after every pull."},
    "nl": {"title": "Je policy-based agent in run 1",
           "head": "Je `nudge` in run 1 van de demo, met η = 0,1: de kans om aan elke automaat te trekken, na elke keer trekken.",
           "say": "Na 1.000 keer trekken zijn de kansen {c}; in de laatste 100 keer trok de agent {b} keer aan de beste automaat.",
           "none": "Je `nudge` stopt met een fout, duurt te lang, of geeft iets anders dan kansen tussen 0 en 1 die samen 1 zijn, dus er is nog geen afbeelding.",
           "x": "keer trekken", "y": "kans om te trekken", "machine": "betaalt {x}",
           "alt": "Drie lijnen, één per automaat: de kans dat je agent eraan trekt, na elke keer trekken."},
}


def _and(items: list, lang: str) -> str:
    return ", ".join(items[:-1]) + (" and " if lang == "en" else " en ") + items[-1]


def nudge_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _PICTURE_TEXT[lang]
    run_ = seen.mine
    fn = run_.names.get("nudge") if run_ is not None else None
    got = _call(lambda: one_run(fn)) if callable(fn) else None
    if not isinstance(got, dict):
        return yours.figure(w["title"], _note(w["head"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    panel = chart.Panel((0, PULLS), (0, 1), xlabel=w["x"], ylabel=w["y"], yticks=[0, 0.25, 0.5, 0.75, 1])
    xs = list(range(0, len(got["chances"]), 4))
    styles = {2: ("param", 2.2, False), 1: ("param", 1.5, True), 0: ("ink-3", 1.5, True)}
    for a in (0, 1, 2):
        role, width, dashed = styles[a]
        ys = [got["chances"][i][a] for i in xs]
        panel.line(xs, ys, role, width, dashed=dashed)
    key = live.key_html([(styles[a][0], styles[a][2], w["machine"].format(x=number(PAYS[a], 1, lang))) for a in (2, 1, 0)])
    last = got["chances"][-1]
    say = w["say"].format(c=_and([number(x, 2, lang) for x in last], lang), b=sum(a == BEST for a in got["pulled"][-100:]))
    body = _note(w["head"]) + _chart([panel], w["alt"], lang, height=220) + key + _note(say)
    return yours.figure(w["title"], body, alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------------

REQUIRED = ["kinds", "nudge", "paper"]

_GIVEN = {"en": "Where do its predictions of what an action leads to come from: written down by someone, or learned from its own tries?",
          "nl": "Wie vertelt de agent waar een action toe leidt: iemand die dat opschreef, of de eigen pogingen?"}
_LEARNED = {"en": "Before it plans, it builds something up from its own tries: is that values, or a prediction of what an action leads to?",
            "nl": "Voordat de agent plant, bouwt de agent iets op uit de eigen pogingen: zijn dat waarden, of een voorspelling van waar een action toe leidt?"}
_FREE = {"en": "Does it ever predict what an action will lead to, or does it only keep what its tries were worth?",
         "nl": "Voorspelt de agent ooit waar een action toe leidt, of houdt de agent alleen bij wat de pogingen waard waren?"}

KINDS = Sort(
    name="kinds",
    groups={
        "given": {"en": "Model-based, with a model it was given", "nl": "Model-based, met een model dat de agent kreeg"},
        "learned": {"en": "Model-based, with a model it learned", "nl": "Model-based, met een model dat de agent leerde"},
        "free": {"en": "Model-free", "nl": "Model-free"},
    },
    items={
        "route": {"en": "**Route planner.** It has every road with its travel time, and works out the fastest route before you leave.",
                  "nl": "**Routeplanner.** Heeft elke weg met de reistijd, en rekent de snelste route uit voordat je vertrekt."},
        "board": {"en": "**Board game program.** Before each move it plays moves ahead in its head, with the rules of the game.",
                  "nl": "**Bordspelprogramma.** Speelt voor elke zet in gedachten zetten vooruit, met de regels van het spel."},
        "heating": {"en": "**Heating.** For a week it switches on and off and measures how fast the room warms up and cools down; then it plans when to switch on.",
                    "nl": "**Verwarming.** Schakelt een week aan en uit en meet hoe snel de kamer opwarmt en afkoelt; plant daarna wanneer de verwarming aangaat."},
        "blackjack": {"en": "**Blackjack agent.** Monte Carlo control: it plays hands and learns Q from what each hand gave.",
                      "nl": "**Blackjack-agent.** Monte Carlo control: speelt handen en leert Q uit wat elke hand opleverde."},
        "cliff": {"en": "**Cliff walker.** Q-learning on the cliff: it learns from every step it takes.",
                  "nl": "**Agent op het klif.** Q-learning op het klif: leert van elke stap die de agent zet."},
        "grid": {"en": "**Hall with puddles.** Policy iteration on HALL, with the table `T`: for every tile and action, where the step goes and what it costs.",
                 "nl": "**Hal met plassen.** Policy iteration op HALL, met de tabel `T`: voor elke tegel en action waar de stap heen gaat en wat de stap kost."},
        "arm": {"en": "**Robot arm.** It learns from its own tries where its arm ends up after each motor signal, then tries plans out in that learned simulator before it moves.",
                "nl": "**Robotarm.** Leert uit de eigen pogingen waar de arm terechtkomt na elk signaal aan de motoren, en probeert daarna plannen uit in de geleerde simulator voordat de arm beweegt."},
    },
    solution={"route": "given", "board": "given", "heating": "learned", "blackjack": "free", "cliff": "free",
              "grid": "given", "arm": "learned"},
    nudge={"route": _GIVEN, "board": _GIVEN, "grid": _GIVEN, "heating": _LEARNED, "arm": _LEARNED,
           "blackjack": _FREE, "cliff": _FREE},
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the seven agents again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de zeven agents opnieuw uit."},
        "empty": {"en": "Put each of the seven agents in one of the three groups.",
                  "nl": "Zet elk van de zeven agents in een van de drie groepen."},
        "unfinished": {"en": "Everything you placed so far is right. Place the rest too ({n} to go).",
                       "nl": "Je plaatsingen tot nu toe kloppen allemaal. Plaats ook de rest (nog {n})."},
        "wrong": {"en": "“{item}” is in the wrong group.",
                  "nl": "‘{item}’ staat in de verkeerde groep."},
        "correct": {"en": "All {n} are right. The route planner, the board game program and policy iteration were given their model: the roads, the rules, the table `T`. The heating and the robot arm learn one, then plan with it. The blackjack agent and the cliff walker never predict where an action leads; they only learn what their tries were worth.",
                    "nl": "Alle {n} kloppen. De routeplanner, het bordspelprogramma en policy iteration kregen het model: de wegen, de regels, de tabel `T`. De verwarming en de robotarm leren er een, en plannen daar dan mee. De blackjack-agent en de agent op het klif voorspellen nooit waar een action toe leidt; de twee leren alleen wat de eigen pogingen waard waren."},
    },
)


# exercise 2: nudge
def _n(line: str = "p[a] + eta * (target - p[a])") -> str:
    """nudge as the box holds it, with this line in new.append(...)."""
    return ("def nudge(p, action, reward, eta):\n"
            "    if reward == 0:\n"
            "        return p\n"
            "    new = []\n"
            "    for a in range(len(p)):\n"
            "        if a == action:\n"
            "            target = 1\n"
            "        else:\n"
            "            target = 0\n"
            f"        new.append({line})\n"
            "    return new\n")


_N_START = _n("...")
_N = _n()
_N_AWAY = _n("p[a] + eta * (p[a] - target)")
_N_NO_MINUS = _n("p[a] + eta * target")
_N_ALL_UP = _n("p[a] + eta * (1 - p[a])")
_N_ACTION = _n("p[a] + eta * (target - p[action])")
_N_TARGET = _n("eta * (target - p[a])")


def same_chances(a, b) -> bool:
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)) or len(a) != len(b):
        return False
    try:
        return all(abs(float(x) - float(y)) < 1e-9 for x, y in zip(a, b))
    except (TypeError, ValueError):
        return False


def show_chances(v) -> dict:
    if isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
        shown = "`[" + ", ".join(f"{round(float(x), 4):g}" for x in v) + "]`"
    else:
        shown = f"`{v!r}`"
    return {"en": shown, "nl": shown}


NUDGE_CASES = [
    ([0.2, 0.3, 0.5], 2, 1, 0.1),
    ([0.2, 0.3, 0.5], 0, 0, 0.1),
    ([0.5, 0.25, 0.25], 0, 1, 0.5),
    ([0.1, 0.2, 0.3, 0.4], 1, 1, 0.2),
    ([0.6, 0.3, 0.1], 2, 1, 0.05),
]

def _dots_left(got, text_) -> bool:
    """A `...` of the start code left in the box: an Ellipsis in the code, not three dots inside a string."""
    try:
        tree = ast.parse(text_)
    except (SyntaxError, TypeError, ValueError):
        return False
    return any(isinstance(node, ast.Constant) and node.value is Ellipsis for node in ast.walk(tree))


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Can you replace it with your own code?",
                nl="Er staat nog een `...` in je code. Kun je de `...` vervangen door je eigen code?")

EXERCISES = {
    "kinds": KINDS,
    "nudge": FunctionBox(
        name="nudge",
        function="nudge",
        start=_N_START,
        solution=_N,
        compare=same_chances,
        show=show_chances,
        visible=NUDGE_CASES[:1],
        hidden=NUDGE_CASES[1:],
        first=[_DOTS],
        picture=nudge_picture,
        mistakes=[
            Mistake("away", wrong=_N_AWAY,
                    en="`{call}` gives {got}: the chance of the machine that paid goes down, and the others go up. Which comes first in the difference, so that each chance moves toward its target?",
                    nl="`{call}` geeft {got}: de kans van de automaat die betaalde gaat omlaag, en de andere gaan omhoog. Wat komt eerst in het verschil, zodat elke kans richting het target schuift?"),
            Mistake("no-minus", wrong=_N_NO_MINUS,
                    en="`{call}` gives {got}: the chances no longer add up to 1, since the chance of the machine that paid grows and the others stay where they were. Each chance moves η of the way toward its target: how far away from its target is it now?",
                    nl="`{call}` geeft {got}: de kansen zijn samen geen 1 meer, want de kans van de automaat die betaalde groeit en de andere blijven staan. Elke kans schuift η van de afstand richting het target op: hoe ver ligt de kans nu van het target?"),
            Mistake("all-up", wrong=_N_ALL_UP,
                    en="`{call}` gives {got}: every chance moves toward 1, also for the machines that were not pulled, so the chances add up to more than 1. Which name holds 1 for the machine pulled and 0 for the others?",
                    nl="`{call}` geeft {got}: elke kans schuift richting 1, ook bij de automaten waar de agent niet aan trok, dus zijn de kansen samen meer dan 1. Welke naam is 1 voor de automaat waar de agent aan trok, en 0 voor de andere?"),
            Mistake("action", wrong=_N_ACTION,
                    en="`{call}` gives {got}: every machine subtracts the chance of the machine pulled from its target, instead of its own chance. Which machine's chance should move toward the target on this line?",
                    nl="`{call}` geeft {got}: elke automaat haalt de kans van de automaat waar de agent aan trok van het target af, in plaats van de eigen kans. Van welke automaat moet de kans op deze regel richting het target schuiven?"),
            Mistake("step-only", wrong=_N_TARGET,
                    en="`{call}` gives {got}: that is only the step, η times the distance to the target, without the chance it starts from. What does the new chance start from?",
                    nl="`{call}` geeft {got}: dat is alleen de stap, η keer de afstand tot het target, zonder de kans waar de stap van uitgaat. Waar begint de nieuwe kans?"),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "empty": {"en": "Replace the `...` with the new chance of machine `a`: its old chance, moved η of the way toward its target.",
                      "nl": "Vervang de `...` door de nieuwe kans van automaat `a`: de oude kans, η van de afstand richting het target opgeschoven."},
            "arguments": {"en": "The check calls `nudge(p, action, reward, eta)`, and the `def` line of your function has a different number of names between its parentheses. Can you keep those four names?",
                          "nl": "De check roept `nudge(p, action, reward, eta)` aan, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je deze vier namen houden?"},
            "slow": {"en": "Your function is still not done after 2 seconds. Does its loop go over the machines once?",
                     "nl": "Je functie is na 2 seconden nog niet klaar. Gaat de loop één keer langs de automaten?"},
            "wrong": {"en": "`{call}` gives {got}. Does each new chance start from the old one and move η of the way toward its target?",
                      "nl": "`{call}` geeft {got}. Begint elke nieuwe kans bij de oude, en schuift de kans η van de afstand richting het target op?"},
            "almost": {"en": "Right for the first call. But `{call}` gives {got}. Does your function work for every list of chances, with the values it gets?",
                       "nl": "Goed voor de eerste aanroep. Maar `{call}` geeft {got}. Werkt je functie voor elke lijst met kansen, met de waarden die je functie krijgt?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return new`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return new`?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back: Python never gets to a `return` with a value. Does it still end with `return new`, with four spaces at the start, as in the start code?",
                          "nl": "Met `{call}` geeft je functie niets terug: Python komt nooit bij een `return` met een waarde. Eindigt je functie nog met `return new`, met vier spaties aan het begin, zoals in de beginregels?"},
            "correct": {"en": "Right in all {n} cases: after a pull of machine 2 that pays, `[0.2, 0.3, 0.5]` becomes `[0.18, 0.27, 0.55]`, and the chances still add up to 1. In the picture your `nudge` plays run 1 of the demo; which machine does it end up pulling?",
                        "nl": "Goed in alle {n} gevallen: als automaat 2 na een keer trekken betaalt, wordt `[0.2, 0.3, 0.5]` gelijk aan `[0.18, 0.27, 0.55]`, en zijn de kansen samen nog steeds 1. In de afbeelding speelt je `nudge` run 1 van de demo; aan welke automaat trekt de agent uiteindelijk?"},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner gave, state, known mistake).
_RIGHT = dict(KINDS.solution)
CASES = [
    ("kinds", {}, "empty", None),
    ("kinds", _RIGHT, "correct", None),
    ("kinds", {"route": "given", "board": "given"}, "wrong", "unfinished"),
    ("kinds", {**_RIGHT, "board": "free"}, "wrong", "wrong:board"),
    ("kinds", {**_RIGHT, "heating": "given"}, "wrong", "wrong:heating"),
    ("kinds", {**_RIGHT, "blackjack": "learned"}, "wrong", "wrong:blackjack"),
    ("kinds", {**_RIGHT, "grid": "free", "arm": "free"}, "wrong", "wrong:grid"),
    ("kinds", {**_RIGHT, "arm": "given"}, "wrong", "wrong:arm"),
    ("kinds", {**_RIGHT, "route": "learned"}, "wrong", "wrong:route"),
    ("kinds", {**_RIGHT, "cliff": "given"}, "wrong", "wrong:cliff"),
    ("nudge", _N_START, "empty", None),
    ("nudge", _N, "correct", None),
    ("nudge", _n("p[a] + ..."), "wrong", "dots-left"),
    ("nudge", _N_AWAY, "wrong", "away"),
    ("nudge", _N_NO_MINUS, "wrong", "no-minus"),
    ("nudge", _N_ALL_UP, "wrong", "all-up"),
    ("nudge", _N_ACTION, "wrong", "action"),
    ("nudge", _N_TARGET, "wrong", "step-only"),
    ("nudge", _N.replace("    return new\n", ""), "wrong", "unreached"),
    ("nudge", _N.replace("    return new\n", "").replace("        return p\n", "        p = p\n"), "wrong", "no-return"),
    # other right answers: the two cases written out, and the step first
    ("nudge", "def nudge(p, action, reward, eta):\n    if reward == 0:\n        return p\n    new = []\n    for a in range(len(p)):\n        if a == action:\n            new.append(p[a] + eta * (1 - p[a]))\n        else:\n            new.append(p[a] * (1 - eta))\n    return new\n", "correct", None),
    ("nudge", _n("(1 - eta) * p[a] + eta * target"), "correct", None),
    # a nudge after a pull that paid nothing too: right where the reward is 1 only
    ("nudge", _N.replace("    if reward == 0:\n        return p\n", ""), "almost", None),
]
