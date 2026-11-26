# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rl-bandits: the course's bandits of ten machines, the agent that chooses with epsilon-greedy
and keeps the sample average of every machine, the course's runs over 2,000 bandits, and the demo over 300 bandits
with the learner's estimates; estimates and best_share, the functions the learner writes, and their pictures.

The bandits follow the ten-armed testbed of Sutton and Barto (Reinforcement Learning: An Introduction, 2nd edition,
2018, section 2.3): the mean of every machine is drawn around 0 with standard deviation 1, and a reward around that
mean with standard deviation 1. The code, the runs and the pictures are the course's own."""
import ast
import copy
import html
import math
import numbers

import numpy as np

from everything_ai import chart, live, yours
from everything_ai.check import FunctionBox, Mistake, kept

K = 10                       # machines of a bandit
STEPS = 1000                 # pulls of one run
RUNS = 2000                  # bandits of the course's runs
EPSILONS = (0.0, 0.01, 0.1)  # greedy and the two epsilon-greedy agents
SEED = 1                     # the course's runs
DEMO_RUNS = 300              # bandits of the demo, with the learner's estimates
DEMO_SEED = 5                # the demo's bandits, so that they are not the course's own
LONG_STEPS = 10_000          # the run of exercise 3, measured by the lesson's test
IDEA_SEED = 3                # the bandit of the pictures


# --- the bandits and the agent -----------------------------------------------------------------------------------------

def bandits(n: int, seed: int) -> np.ndarray:
    """n bandits of K machines: the mean of every machine, drawn around 0 with standard deviation 1."""
    return np.random.default_rng(seed).normal(0.0, 1.0, (n, K))


def pull(means, action: int, rng) -> float:
    """A pull of machine `action`: a reward around its mean, with standard deviation 1."""
    return float(means[action] + rng.normal())


def choose(Q, epsilon: float, rng) -> int:
    """Epsilon-greedy: with probability epsilon a random machine, else the one with the largest estimate (the first
    of them when several are equal, as np.argmax does)."""
    if rng.random() < epsilon:
        return int(rng.integers(len(Q)))
    return int(np.argmax(Q))


def estimates(totals, counts):
    """The course's solution of exercise 1: the sample average of every machine, its total divided by its count, and
    0 for a machine that was never pulled."""
    q = []
    for i in range(len(totals)):
        if counts[i] == 0:
            q.append(0.0)
        else:
            q.append(totals[i] / counts[i])
    return q


def best_share(actions, best):
    """The course's solution of exercise 2: for every step, the share of the runs whose machine was the best one.
    `actions` has a row for every run and a column for every step; `best` is a column with the best machine of every
    run."""
    return (actions == best).mean(axis=0)


def _runs(n: int, steps: int, epsilon: float, seed: int, with_q: bool = False) -> tuple:
    """The agent on n bandits at once, with NumPy: the reward of every run at every step, the machines it pulled,
    the best machine of every run as a column and the means; `with_q` adds the estimates at the end."""
    rng = np.random.default_rng(seed)
    means = rng.normal(0.0, 1.0, (n, K))
    totals, counts, q = np.zeros((n, K)), np.zeros((n, K)), np.zeros((n, K))
    rewards, actions = np.zeros((n, steps)), np.zeros((n, steps), dtype=np.int8)
    rows = np.arange(n)
    for t in range(steps):
        greedy = q.argmax(axis=1)
        random_ = rng.integers(0, K, n)
        a = np.where(rng.random(n) < epsilon, random_, greedy)
        r = means[rows, a] + rng.normal(0.0, 1.0, n)
        totals[rows, a] += r
        counts[rows, a] += 1
        q[rows, a] = totals[rows, a] / counts[rows, a]
        rewards[:, t], actions[:, t] = r, a
    best = means.argmax(axis=1).astype(np.int8).reshape(-1, 1)
    return (rewards, actions, best, means, q) if with_q else (rewards, actions, best, means)


def course_runs() -> dict:
    """The course's runs: for every epsilon, 2,000 bandits of 1,000 pulls, with the same bandits for each (SEED).
    For each: the mean reward at every step, the share of best machines at every step, the machines pulled and the
    best machine of every bandit."""
    def make():
        out = {}
        for eps in EPSILONS:
            rewards, actions, best, means = _runs(RUNS, STEPS, eps, SEED)
            out[eps] = {"reward": rewards.mean(axis=0), "best": best_share(actions, best), "actions": actions,
                        "best_of": best, "top": float(means.max(axis=1).mean())}
        return out
    return kept("rl-bandits:runs", make)


def last(eps: float, what: str, n: int = 100) -> float:
    """The mean over the last n steps of the course's run with this epsilon: of the reward, or of the share of best
    machines."""
    return float(np.mean(course_runs()[eps][what][-n:]))


def first(eps: float, what: str, n: int = 20) -> float:
    """The mean over the first n steps of the course's run with this epsilon."""
    return float(np.mean(course_runs()[eps][what][:n]))


def at(eps: float, what: str, step: int, half: int = 10) -> float:
    """The mean of the course's run around one step: over the 2 x half steps around it, so that one step's luck does
    not count."""
    v = course_runs()[eps][what]
    return float(np.mean(v[max(0, step - half - 1): step + half - 1]))


def long_run(steps: int = LONG_STEPS, runs: int = RUNS) -> dict:
    """The run of exercise 3: 2,000 bandits of 10,000 pulls, for epsilon 0.01 and 0.1. Over the last 1,000 pulls the
    mean reward ("reward") and the share of best machines ("best"); at the end, the share of bandits whose largest
    estimate is the best machine ("knows"), and where it is not, how much less that machine pays than the best one,
    on average ("gap"). Takes some seconds: the lesson's test measures it."""
    out = {}
    for eps in (0.01, 0.1):
        rewards, actions, best, means, q = _runs(runs, steps, eps, SEED, with_q=True)
        took = q.argmax(axis=1)
        knows = took == best[:, 0]
        gap = means.max(axis=1) - means[np.arange(runs), took]
        out[eps] = {"reward": float(rewards[:, -1000:].mean()), "best": float(best_share(actions[:, -1000:], best).mean()),
                    "knows": float(knows.mean()), "gap": float(gap[~knows].mean())}
    return out


# --- the bandit of the pictures ----------------------------------------------------------------------------------------

def idea_means() -> list:
    """The ten means of the bandit in the pictures, rounded to two decimals."""
    return [round(float(m), 2) for m in bandits(1, IDEA_SEED)[0]]


def idea_rewards(per: int = 40) -> list:
    """`per` rewards of every machine of the bandit in the pictures, for the picture of the idea."""
    rng = np.random.default_rng(IDEA_SEED + 100)
    return [[pull(idea_means(), a, rng) for _ in range(per)] for a in range(K)]


def play(est, epsilon: float, pulls: int, seed: int, means=None) -> dict:
    """An agent on the bandit of the pictures (or `means`): `pulls` pulls with epsilon-greedy, the estimates from
    `est(totals, counts)` after every pull. Its totals, counts, estimates and rewards at the end."""
    means = idea_means() if means is None else means
    rng = np.random.default_rng(seed)
    totals, counts = [0.0] * len(means), [0] * len(means)
    q = list(est(list(totals), list(counts)))
    rewards = []
    for _ in range(pulls):
        a = choose(q, epsilon, rng)
        r = pull(means, a, rng)
        totals[a] = totals[a] + r
        counts[a] = counts[a] + 1
        q = list(est(list(totals), list(counts)))
        rewards.append(r)
    return {"totals": totals, "counts": counts, "q": q, "rewards": rewards}


# --- the demo: 300 bandits with the learner's estimates ----------------------------------------------------------------

LOOKS = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]   # the rewind's stops
WATCHED = 0    # the bandit of the demo that the picture beside the chart shows
DEMO_LIMIT = 120.0   # seconds the demo gets


def _ten_numbers(values) -> bool:
    """Whether the estimates are K finite numbers, of Python or of NumPy."""
    try:
        return len(values) == K and all(isinstance(x, numbers.Real) and math.isfinite(x) for x in values)
    except TypeError:
        return False


def train(est, runs: int = DEMO_RUNS, seed: int = DEMO_SEED, show=None, steps: int = STEPS):
    """The demo: greedy, epsilon 0.01 and epsilon 0.1 side by side on the same `runs` bandits, every agent with the
    estimates of `est(totals, counts)`. After every step, `show(step, means, seen)` gets the mean reward of each
    agent at that step and what the agents know of the watched bandit. Gives the run, or "estimates" when `est` stops
    with an error or gives no list of K numbers."""
    means = bandits(runs, seed).tolist()
    rng = np.random.default_rng(seed + 1)
    state = {eps: {"totals": [[0.0] * K for _ in range(runs)], "counts": [[0] * K for _ in range(runs)]} for eps in EPSILONS}
    try:
        for eps in EPSILONS:
            start = list(est([0.0] * K, [0] * K))
            if not _ten_numbers(start):
                return "estimates"
            state[eps]["q"] = [list(start) for _ in range(runs)]
    except Exception:   # noqa: BLE001   the demo says which function stopped
        return "estimates"
    best = [max(range(K), key=lambda a, m=m: m[a]) for m in means]
    curve = {eps: [] for eps in EPSILONS}
    hits = {eps: [] for eps in EPSILONS}
    looks = {}
    for t in range(1, steps + 1):
        for eps in EPSILONS:
            s = state[eps]
            total, right = 0.0, 0
            for b in range(runs):
                q = s["q"][b]
                a = choose(q, eps, rng)
                if a >= K:   # a list that grew longer than its check allows
                    return "estimates"
                r = means[b][a] + float(rng.normal())
                s["totals"][b][a] += r
                s["counts"][b][a] += 1
                try:
                    new = est(list(s["totals"][b]), list(s["counts"][b]))
                    if t <= 3 and not _ten_numbers(new):
                        return "estimates"
                except Exception:   # noqa: BLE001
                    return "estimates"
                s["q"][b] = new
                total += r
                right += a == best[b]
            curve[eps].append(total / runs)
            hits[eps].append(right / runs)
        seen = {eps: {"q": list(state[eps]["q"][WATCHED]), "counts": list(state[eps]["counts"][WATCHED])} for eps in EPSILONS}
        if t in LOOKS:
            looks[t] = seen
        if show is not None:
            show(t, {eps: curve[eps][-1] for eps in EPSILONS}, seen)
    return {"curve": curve, "best": hits, "looks": looks, "means": means[WATCHED], "runs": runs}


# --- the parts of the pictures -----------------------------------------------------------------------------------------

LANGS = ("en", "nl")
_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
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
    if out in ("-0." + "0" * digits, "-0"):
        out = out[1:]
    out = out.replace("-", "−")
    return out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out


def _pct(x: float, lang: str, digits: int = 1) -> str:
    out = f"{100 * x:.{digits}f}%"
    return out.replace(".", ",") if lang == "nl" else out


def _eps(eps: float, lang: str) -> str:
    return "ε = " + (f"{eps:g}".replace(".", ",") if lang == "nl" else f"{eps:g}")


def _svg(panels: list, alt: str, lang: str, width: int = 330, height: int = 250) -> str:
    out = chart.figure(panels, alt=alt, width=width, height=height)
    return chart.with_decimal_comma(out) if lang == "nl" else out


def _bandit_panel(means, q, counts, title: str, lang: str, low: float = -3.5, high: float = 4.5,
                  stagger: bool = False) -> chart.Panel:
    """One bandit as the pictures draw it: the mean of every machine as a short line, the agent's estimate as an
    orange dot, open for a machine it never pulled, and above each machine its number of pulls; with `stagger`, the
    numbers of the odd machines a row lower, so that numbers of three or four digits in a narrow panel stay apart."""
    w = _WORDS[lang]
    p = chart.Panel((-0.6, K - 0.4), (low, high), title=title, xlabel=w["machine"], ylabel=w["reward"],
                    xticks=list(range(K)))
    for a in range(K):
        p.line([a - 0.34, a + 0.34], [means[a], means[a]], "ink", 2.2)
    if q is not None:
        tried = [a for a in range(K) if counts[a] > 0]
        never = [a for a in range(K) if counts[a] == 0]
        if tried:
            p.points(tried, [max(low, min(high, q[a])) for a in tried], "param", r=4.2)
        if never:
            p.points(never, [max(low, min(high, q[a])) for a in never], "param", r=3.6, hollow=True)
        for a in range(K):
            p.text(a, high - (0.35 if a % 2 == 0 or not stagger else 1.0), f"{counts[a]}", "ink-2")
    return p


_WORDS = {
    "en": {"machine": "machine", "reward": "reward", "pull": "pull", "share": "share of bandits",
           "greedy": "greedy", "best": "best machine"},
    "nl": {"machine": "automaat", "reward": "reward", "pull": "trek", "share": "deel van de bandits",
           "greedy": "greedy", "best": "beste automaat"},
}


# --- the picture of the idea -------------------------------------------------------------------------------------------

_IDEA_TEXT = {
    "en": {"title": "Ten machines of one bandit: 40 pulls each",
           "alt": "Ten machines, numbered 0 to 9: for each, 40 rewards as gray dots, spread around a short line at its mean. Machine 9 has the highest mean, 3.32, machine 0 the second, 2.04.",
           "note": "Gray: 40 rewards of each machine. The short line at each machine: its mean, which the agent does not know."},
    "nl": {"title": "Tien automaten van één bandit: elk 40 keer getrokken",
           "alt": "Tien automaten, genummerd 0 tot en met 9: bij elke automaat 40 rewards als grijze stippen, verspreid rond een kort streepje bij het gemiddelde. Automaat 9 heeft het hoogste gemiddelde, 3,32, automaat 0 het op één na hoogste, 2,04.",
           "note": "Grijs: 40 rewards van elke automaat. Het korte streepje bij elke automaat: het gemiddelde, dat de agent niet kent."},
}


def idea_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang] | _WORDS[lang]
    rewards = idea_rewards()
    rng = np.random.default_rng(0)
    p = chart.Panel((-0.6, K - 0.4), (-5.5, 6.5), title="", xlabel=w["machine"], ylabel=w["reward"],
                    xticks=list(range(K)))
    for a in range(K):
        xs = (a + rng.uniform(-0.22, 0.22, len(rewards[a]))).tolist()
        p.points(xs, rewards[a], "ink-3", r=2.0)
    for a, m in enumerate(idea_means()):
        p.line([a - 0.36, a + 0.36], [m, m], "ink", 2.6)
    return yours.figure(w["title"], _svg([p], w["alt"], lang, width=560, height=280) + _note(w["note"]), alt=w["title"])


# --- the learner's functions in the pictures ---------------------------------------------------------------------------

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


def _numbers(v, n: int) -> list | None:
    """v as a list of n finite floats, or None."""
    try:
        out = [float(x) for x in v]
    except Exception:   # noqa: BLE001
        return None
    return out if len(out) == n and all(math.isfinite(x) for x in out) else None


ESTIMATE_PULLS = 200   # pulls of the agent in the picture of exercise 1
ESTIMATE_SEED = 11

_EST_TEXT = {
    "en": {"title": "Your estimates at work",
           "head": "An agent with ε = 0.1 and your `estimates` plays the bandit of *The idea*: {n} pulls. The number above each machine: how often it was pulled.",
           "alt": "The ten machines of the bandit of the idea, each with its mean as a short line and the agent's estimate as an orange dot, and the number of pulls above it.",
           "none_alt": "The ten machines of the bandit of the idea, each with its mean as a short line.",
           "mean": "Over the {n} pulls the agent got {r} per pull; the best machine, 9, pays {top}.",
           "same": "Your estimates are the course's after every pull.",
           "differ": "After {n} pulls, your estimates differ from the course's: machine {a} gets {mine}, and its total divided by its count is {want}.",
           "none": "Your `estimates` stops with an error, takes too long or gives no list of ten numbers for this bandit, so the picture shows the means alone.",
           "note": "The short line: the mean of each machine. Orange: your estimates, an open dot for a machine that was never pulled."},
    "nl": {"title": "Je estimates aan het werk",
           "head": "Een agent met ε = 0,1 en je `estimates` speelt de bandit van *Het idee*: {n} keer trekken. Het getal boven elke automaat: hoe vaak eraan getrokken is.",
           "alt": "De tien automaten van de bandit uit het idee, elk met het gemiddelde als kort streepje en de schatting van de agent als oranje stip, en erboven hoe vaak eraan getrokken is.",
           "none_alt": "De tien automaten van de bandit uit het idee, elk met het gemiddelde als kort streepje.",
           "mean": "Over de {n} keer trekken kreeg de agent {r} per keer; de beste automaat, 9, geeft {top}.",
           "same": "Je schattingen zijn na elke trek gelijk aan de schattingen van de course.",
           "differ": "Na {n} keer trekken verschillen je schattingen van de schattingen van de course: automaat {a} krijgt {mine}, en het totaal gedeeld door het aantal is {want}.",
           "none": "Je `estimates` stopt met een fout, duurt te lang of geeft voor deze bandit geen lijst van tien getallen, dus de afbeelding toont alleen de gemiddelden.",
           "note": "Het korte streepje: het gemiddelde van elke automaat. Oranje: je schattingen, een open stip voor een automaat waaraan nooit getrokken is."},
}


def estimates_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _EST_TEXT[lang]
    means = idea_means()
    fn = _fn(seen)
    course = play(estimates, 0.1, ESTIMATE_PULLS, ESTIMATE_SEED)
    mine = None
    if fn is not None:
        def safe(t, c):
            out = _numbers(fn(t, c), K)
            if out is None:
                raise ValueError("not ten numbers")
            return out
        got = _call(lambda: play(safe, 0.1, ESTIMATE_PULLS, ESTIMATE_SEED))
        mine = got if isinstance(got, dict) else None
    head = _note(w["head"].format(n=ESTIMATE_PULLS))
    if mine is None:
        p = _bandit_panel(means, None, [0] * K, "", lang)
        return yours.figure(w["title"], head + _svg([p], w["none_alt"], lang, width=520, height=260) + _note(w["none"], _LOSS_UI), alt=w["title"])
    p = _bandit_panel(means, mine["q"], mine["counts"], "", lang)
    lines = _note(w["mean"].format(n=ESTIMATE_PULLS, r=_dec(sum(mine["rewards"]) / ESTIMATE_PULLS, lang), top=_dec(means[9], lang)))
    want = [t / c if c else 0.0 for t, c in zip(mine["totals"], mine["counts"])]
    off = [a for a in range(K) if abs(mine["q"][a] - want[a]) > 1e-9]
    if mine["counts"] == course["counts"] and not off:
        lines += _note(w["same"])
    elif off:
        a = off[0]
        lines += _note(w["differ"].format(n=ESTIMATE_PULLS, a=a, mine=_dec(mine["q"][a], lang), want=_dec(want[a], lang)), _LOSS_UI)
    return yours.figure(w["title"], head + _svg([p], w["alt"], lang, width=520, height=260) + lines + _note(w["note"]), alt=w["title"])


# --- the share of best machines ----------------------------------------------------------------------------------------

def actions_of(eps: float) -> tuple:
    """The machines pulled in the course's run with this epsilon (2,000 rows, 1,000 columns) and the best machine of
    every bandit as a column, as new arrays."""
    r = course_runs()[eps]
    return r["actions"].copy(), r["best_of"].copy()


_SHARE_TEXT = {
    "en": {"title": "Your best_share on the course's runs",
           "head": "The three agents of the course on the same 2,000 bandits; your `best_share` works out, for every pull, the share of bandits where the agent pulled the best machine.",
           "alt": "Three lines over 1,000 pulls, one for greedy, epsilon 0.01 and epsilon 0.1: for every pull, the share of the 2,000 bandits in which the agent pulled the best machine, as your best_share works it out.",
           "end": "Over the last 100 pulls: greedy {g}, ε = 0.01 {a}, ε = 0.1 {b}.",
           "differ": "Your values differ from the course's: at pull {t}, {who} gets {mine} from your function, and {want} from the course's.",
           "none": "Your `best_share` stops with an error, takes too long or gives no array of one share per pull for these runs.",
           "note": "The lines are the agents: the thickest is ε = 0.1, the dashed one greedy."},
    "nl": {"title": "Je best_share op de runs van de course",
           "head": "De drie agents van de course op dezelfde 2.000 bandits; je `best_share` rekent voor elke trek uit in welk deel van de bandits de agent aan de beste automaat trok.",
           "alt": "Drie lijnen over 1.000 keer trekken, voor greedy, epsilon 0,01 en epsilon 0,1: voor elke trek het deel van de 2.000 bandits waarin de agent aan de beste automaat trok, zoals je best_share het uitrekent.",
           "end": "Over de laatste 100 keer trekken: greedy {g}, ε = 0,01 {a}, ε = 0,1 {b}.",
           "differ": "Je waarden verschillen van de waarden van de course: bij trek {t} krijgt {who} {mine} van je functie, en {want} van de functie van de course.",
           "none": "Je `best_share` stopt met een fout, duurt te lang of geeft voor deze runs geen array met één aandeel per trek.",
           "note": "De lijnen zijn de agents: de dikste is ε = 0,1, de gestreepte greedy."},
}

_STYLE = {0.0: ("ink-3", 1.8, True), 0.01: ("ink-2", 1.8, False), 0.1: ("ink", 2.6, False)}


def _share_panel(curves: dict, lang: str, title: str = "") -> chart.Panel:
    w = _WORDS[lang]
    p = chart.Panel((0, STEPS), (0, 1), title=title, xlabel=w["pull"], ylabel=w["share"])
    for eps, ys in curves.items():
        role, width, dashed = _STYLE[eps]
        p.line(list(range(1, len(ys) + 1)), list(ys), role, width, dashed=dashed)
        name = w["greedy"] if eps == 0 else _eps(eps, lang)
        p.text(STEPS * 0.98, float(np.mean(ys[-50:])) + 0.06, name, role, "end", halo=True)
    return p


def best_share_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _SHARE_TEXT[lang]
    fn = _fn(seen)
    head = _note(w["head"])
    if fn is None:
        return yours.figure(w["title"], head + _note(w["none"], _LOSS_UI), alt=w["title"])
    mine = {}
    for eps in EPSILONS:
        acts, best = actions_of(eps)
        got = _call(lambda: _numbers(fn(acts, best), STEPS))
        if not isinstance(got, list):
            return yours.figure(w["title"], head + _note(w["none"], _LOSS_UI), alt=w["title"])
        mine[eps] = got
    p = _share_panel(mine, lang)
    end = w["end"].format(g=_pct(np.mean(mine[0.0][-100:]), lang), a=_pct(np.mean(mine[0.01][-100:]), lang),
                          b=_pct(np.mean(mine[0.1][-100:]), lang))
    body = head + _svg([p], w["alt"], lang, width=520, height=260) + _note(end)
    for eps in EPSILONS:
        want = course_runs()[eps]["best"]
        off = [t for t in range(STEPS) if abs(mine[eps][t] - want[t]) > 1e-9]
        if off:
            t = off[0]
            who = _WORDS[lang]["greedy"] if eps == 0 else _eps(eps, lang)
            body += _note(w["differ"].format(t=t + 1, who=who, mine=_dec(mine[eps][t], lang, 3), want=_dec(float(want[t]), lang, 3)), _LOSS_UI)
            break
    return yours.figure(w["title"], body + _note(w["note"]), alt=w["title"])


# --- the demo's pictures -----------------------------------------------------------------------------------------------

_LIVE_TEXT = {
    "en": {"title": "Your run: the mean reward at every pull", "x": "pull", "y": "mean reward",
           "alt": "The mean reward over 300 bandits at every pull of the agent with epsilon 0.1 so far, thin, and its mean over the last 20 pulls, thick; dashed, the mean over the last 20 pulls of the greedy agent and of the agent with epsilon 0.01.",
           "rewind_alt": "The mean reward over 300 bandits of the three agents over the whole run, each as its mean over the last 20 pulls: epsilon 0.1 thick, greedy and epsilon 0.01 dashed, with a vertical line at the chosen pull.",
           "now": "The first of the 300 bandits after pull {t}",
           "now_alt": "Two panels, greedy and epsilon 0.1: the ten machines of the first bandit, each with its mean as a short line, the agent's estimate as an orange dot and its number of pulls above it."},
    "nl": {"title": "Je run: gemiddelde reward per trek", "x": "trek", "y": "gemiddelde reward",
           "alt": "De gemiddelde reward over 300 bandits bij elke trek van de agent met epsilon 0,1 tot nu toe, dun, en het gemiddelde over de laatste 20 keer trekken, dik; gestreept het gemiddelde over de laatste 20 keer trekken van de greedy agent en van de agent met epsilon 0,01.",
           "rewind_alt": "De gemiddelde reward over 300 bandits van de drie agents over de hele run, elk als het gemiddelde over de laatste 20 keer trekken: epsilon 0,1 dik, greedy en epsilon 0,01 gestreept, met een verticale lijn bij de gekozen trek.",
           "now": "De eerste van de 300 bandits na trek {t}",
           "now_alt": "Twee panelen, greedy en epsilon 0,1: de tien automaten van de eerste bandit, elk met het gemiddelde als kort streepje, de schatting van de agent als oranje stip en erboven hoe vaak eraan getrokken is."},
}


def live_words(lang: str) -> dict:
    return _LIVE_TEXT[lang if lang in LANGS else "en"]


def now_picture(means, seen: dict, step: int, lang: str) -> str:
    """What the greedy agent and the epsilon-0.1 agent know of the watched bandit after `step` pulls."""
    lang = lang if lang in LANGS else "en"
    w = _LIVE_TEXT[lang] | _WORDS[lang]
    panels = [_bandit_panel(means, seen[eps]["q"], seen[eps]["counts"], w["greedy"] if eps == 0 else _eps(eps, lang), lang,
                            stagger=True) for eps in (0.0, 0.1)]
    return yours.figure(w["now"].format(t=step), _svg(panels, w["now_alt"], lang, width=300, height=240), alt=w["now"].format(t=step))


def _smooth(ys, k: int = 20) -> list:
    out = []
    for i in range(len(ys)):
        part = ys[max(0, i - k + 1): i + 1]
        out.append(sum(part) / len(part))
    return out


def curve_picture(run: dict, lang: str, at: int | None = None) -> str:
    """The three agents of a run, each as its mean reward over the last 20 pulls, with a line at pull `at`."""
    lang = lang if lang in LANGS else "en"
    w = _LIVE_TEXT[lang] | _WORDS[lang]
    p = chart.Panel((0, STEPS), (-0.2, 1.8), title=w["title"], xlabel=w["x"], ylabel=w["y"])
    for eps in EPSILONS:
        role, width, _ = _STYLE[eps]
        dashed = eps != 0.1   # as in the live chart: greedy and 0.01 dashed
        ys = _smooth(run["curve"][eps])
        p.line(list(range(1, len(ys) + 1)), ys, role, width, dashed=dashed)
        name = w["greedy"] if eps == 0 else _eps(eps, lang)
        p.text(STEPS * 0.98, float(np.mean(run["curve"][eps][-50:])) + (0.1 if eps == 0.1 else -0.13), name, role, "end", halo=True)
    if at is not None:
        p.line([at, at], [-0.2, 1.8], "ink-2", 1.2)
    return _svg([p], w["rewind_alt"], lang, width=330, height=240)   # beside the agents in the rewind: two or three charts of 330 px


_REWIND_TEXT = {
    "en": "After pull {t}: greedy has tried {g} of the 10 machines of the first bandit and ε = 0.1 has tried {s}; ε = 0.1 pulled the best machine, {b}, {n} {times}.",
    "nl": "Na trek {t}: greedy heeft {g} van de 10 automaten van de eerste bandit geprobeerd en ε = 0,1 heeft er {s} geprobeerd; ε = 0,1 trok {n} keer aan de beste automaat, {b}.",
}


def rewind_frame(run: dict, step: int, lang: str) -> str:
    seen = run["looks"][step]
    return live.rewound(curve_picture(run, lang, at=step), [now_picture(run["means"], seen, step, lang)])


def rewind_text(run: dict, step: int, lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    seen = run["looks"][step]
    best = max(range(K), key=lambda a: run["means"][a])
    g = sum(1 for c in seen[0.0]["counts"] if c > 0)
    s = sum(1 for c in seen[0.1]["counts"] if c > 0)
    n = seen[0.1]["counts"][best]
    return _REWIND_TEXT[lang].format(t=step, g=g, s=s, b=best, n=n, times="time" if n == 1 else "times")


def watched_means() -> list:
    """The ten means of the bandit of the demo that the picture beside the chart shows."""
    return [float(m) for m in bandits(DEMO_RUNS, DEMO_SEED)[WATCHED]]


def smoothed(ys, k: int = 20) -> list:
    """ys as (pull, mean over the last k pulls), for a line beside the live chart."""
    return list(zip(range(1, len(ys) + 1), _smooth(ys, k)))


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma in Dutch, and a real minus sign."""
    return _dec(x, lang if lang in LANGS else "en", digits)


def percent(x: float, lang: str) -> str:
    return _pct(x, lang if lang in LANGS else "en")


def demo(est, show=None):
    """The demo within its time limit: the run, "estimates" when the learner's function fails, or "slow"."""
    from everything_ai.check import Timeout, with_time_limit
    try:
        return with_time_limit(lambda: train(est, show=show), DEMO_LIMIT)
    except Timeout:
        return "slow"


def mine(box_text, correct: bool):
    """The learner's estimates when its check passes, else None: the demo then uses the course's."""
    if not correct:
        return None
    from everything_ai.check import run_code
    fn = run_code(box_text, {"np": np}).names.get("estimates")
    return fn if callable(fn) else None


# --- try it: another epsilon -------------------------------------------------------------------------------------------

TRY_EPSILONS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]


def try_runs() -> dict:
    """The agent on the course's 2,000 bandits for every epsilon of the slider: its mean reward at every pull and its
    share of best machines over the last 100 pulls."""
    def make():
        out = {}
        for eps in TRY_EPSILONS:
            if eps in EPSILONS:
                r = course_runs()[eps]
                out[eps] = (r["reward"], float(np.mean(r["best"][-100:])))
            else:
                rewards, actions, best, _ = _runs(RUNS, STEPS, eps, SEED)
                out[eps] = (rewards.mean(axis=0), float(best_share(actions, best)[-100:].mean()))
        return out
    return kept("rl-bandits:try", make)


_TRY_TEXT = {
    "en": {"title": "The mean reward with ε = {e}", "alt": "The mean reward over 2,000 bandits at every pull, averaged over 20 pulls, for the chosen epsilon, with the greedy agent dashed and the mean of the best machine as a thin line.",
           "said": "ε = {e}: over the last 100 pulls {r} per pull, and the best machine in {p} of the 2,000 bandits."},
    "nl": {"title": "De gemiddelde reward met ε = {e}", "alt": "De gemiddelde reward over 2.000 bandits bij elke trek, gemiddeld over 20 keer trekken, voor de gekozen epsilon, met de greedy agent gestreept en het gemiddelde van de beste automaat als dunne lijn.",
           "said": "ε = {e}: over de laatste 100 keer trekken {r} per keer, en de beste automaat in {p} van de 2.000 bandits."},
}


def try_frame(eps: float, lang: str) -> tuple:
    lang = lang if lang in LANGS else "en"
    w = _TRY_TEXT[lang] | _WORDS[lang]
    runs = try_runs()
    eps = min(TRY_EPSILONS, key=lambda e: abs(e - eps))
    e_text = f"{eps:g}".replace(".", ",") if lang == "nl" else f"{eps:g}"
    reward, share = runs[eps]
    top = course_runs()[0.0]["top"]
    p = chart.Panel((0, STEPS), (-0.2, 1.8), title=w["title"].format(e=e_text), xlabel=_LIVE_TEXT[lang]["x"], ylabel=_LIVE_TEXT[lang]["y"])
    p.line([0, STEPS], [top, top], "ink-3", 1.0)
    p.text(STEPS * 0.02, top + 0.09, w["best"], "ink-3", "start")
    g = _smooth(list(runs[0.0][0]))
    p.line(list(range(1, STEPS + 1)), g, "ink-3", 1.6, dashed=True)
    p.text(STEPS * 0.98, float(np.mean(g[-50:])) - 0.13, w["greedy"], "ink-3", "end")
    p.line(list(range(1, STEPS + 1)), _smooth(list(reward)), "ink", 2.6)
    said = w["said"].format(e=e_text, r=_dec(float(np.mean(reward[-100:])), lang), p=_pct(share, lang))
    return _svg([p], w["alt"], lang, width=460, height=260), said


# --- the exercises -----------------------------------------------------------------------------------------------------

REQUIRED = ["estimates", "best_share", "long-run"]


class _Val:
    """A test value that a check message shows as code: written out, or by a name of the lesson."""

    def __init__(self, shown: str, value):
        self.shown, self.value = shown, value

    def __repr__(self) -> str:
        return self.shown

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


def _value(v) -> str:
    """A value the way code writes it, numbers rounded to six decimals."""
    if isinstance(v, np.ndarray):
        return _value(v.tolist())
    if isinstance(v, (list, tuple)):
        inner = ", ".join(_value(x) for x in v)
        return f"({inner})" if isinstance(v, tuple) else f"[{inner}]"
    if isinstance(v, (bool, np.bool_)):
        return repr(bool(v))
    if isinstance(v, (float, np.floating)):
        return repr(round(float(v), 6) + 0.0)
    if isinstance(v, np.integer):
        return repr(int(v))
    return repr(v)


def _written(v) -> _Val:
    if isinstance(v, np.ndarray):   # as the learner would write it, so that the call can be tried in a cell
        return _Val(f"np.array({_value(v)})", v)
    return _Val(_value(v), v)


def _show(v):
    if isinstance(v, np.ndarray) and v.size > 12:
        head = ", ".join(_value(x) for x in v.ravel()[:4].tolist())
        size = f"{v.size:,}"
        return {"en": f"an array of {size} values that starts with {head}",
                "nl": f"een array van {size.replace(',', '.')} waarden, beginnend met {head}"}
    out = _value(v)
    return out if len(out) <= 300 else out[:300] + "…"


def _dots_left(got, text_) -> bool:
    """A `...` of the start code left in the box: an Ellipsis in the code, not three dots inside a string."""
    try:
        tree = ast.parse(text_)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Constant) and node.value is Ellipsis for node in ast.walk(tree))


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang de `...` door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_PLAYED = play(estimates, 0.1, ESTIMATE_PULLS, ESTIMATE_SEED)
TOTALS_200 = [round(t, 6) for t in _PLAYED["totals"]]   # the agent of the picture after 200 pulls
COUNTS_200 = list(_PLAYED["counts"])

_E_HEAD = "def estimates(totals, counts):\n    q = []\n    for i in range(len(totals)):\n        if counts[i] == 0:\n"


def _e(zero: str, other: str) -> str:
    return _E_HEAD + f"            q.append({zero})\n        else:\n            q.append({other})\n    return q\n"


_E = _e("0.0", "totals[i] / counts[i]")
_E_START = _e("...", "...")
_E_UPSIDE = _e("0.0", "counts[i] / totals[i]")
_E_TOTAL = _e("0.0", "totals[i]")
_E_NONE = _e("None", "totals[i] / counts[i]")
_E_FLOOR = _e("0.0", "totals[i] // counts[i]")
_E_ALL = _e("0.0", "sum(totals) / sum(counts)")
_E_RETURN_IN = _E.replace("    return q\n", "        return q\n")
_E_NO_IF = "def estimates(totals, counts):\n    q = []\n    for i in range(len(totals)):\n        q.append(totals[i] / counts[i])\n    return q\n"

ESTIMATE_CASES = [
    (_written([3.0, 0.0, -1.5]), _written([2, 0, 3])),
    (_written([0.0] * K), _written([0] * K)),
    (_written([2.5, -0.8, 4.2, 0.0, 1.1, 0.0, -3.0, 0.6, 0.0, 9.9]), _written([2, 1, 3, 0, 1, 0, 4, 1, 0, 3])),
    (_written([5.0]), _written([4])),
    (_written([6, 3]), _written([3, 1])),
    (_Val("totals_200", TOTALS_200), _Val("counts_200", COUNTS_200)),
]

# exercise 2: best_share. The small runs of the lesson, then one run, a run that always takes the best machine, and
# the course's runs of the three agents
SMALL_ACTIONS = np.array([[0, 2, 2, 2], [1, 1, 0, 0], [2, 2, 2, 1]])
SMALL_BEST = np.array([[2], [0], [1]])


def _runs_val(eps: float, which: str) -> _Val:
    name = {0.0: "greedy", 0.01: "careful", 0.1: "curious"}[eps]
    acts, best = actions_of(eps)
    return _Val(f"{name}_actions", acts) if which == "actions" else _Val(f"{name}_best", best)


SHARE_CASES = [
    (_Val("small_actions", SMALL_ACTIONS), _Val("small_best", SMALL_BEST)),
    (_written(np.array([[3, 3, 1]])), _written(np.array([[3]]))),
    (_written(np.array([[4, 4], [0, 0]])), _written(np.array([[4], [0]]))),
    (_runs_val(0.0, "actions"), _runs_val(0.0, "best")),
    (_runs_val(0.1, "actions"), _runs_val(0.1, "best")),
]

_S_START = "def best_share(actions, best):\n    hits = ...\n    return ...\n"
_S = "def best_share(actions, best):\n    hits = actions == best\n    return hits.mean(axis=0)\n"
_S_ROWS = _S.replace("hits.mean(axis=0)", "hits.mean(axis=1)")
_S_ALL = _S.replace("hits.mean(axis=0)", "hits.mean()")
_S_COUNT = _S.replace("hits.mean(axis=0)", "hits.sum(axis=0)")
_S_ACTIONS = _S.replace("hits.mean(axis=0)", "actions.mean(axis=0)")
_S_HITS = _S.replace("hits.mean(axis=0)", "hits")


EXERCISES = {
    "estimates": FunctionBox(
        name="estimates",
        picture=estimates_picture,
        function="estimates",
        names={"np": np},
        start=_E_START,
        solution=_E,
        show=_show,
        visible=ESTIMATE_CASES[:1],
        hidden=ESTIMATE_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("zero", error=lambda e: isinstance(e, ZeroDivisionError),
                    en="With `{call}`, your function stops with an error: it divides by a count of 0, and nobody can divide by 0. What should a machine that was never pulled get?",
                    nl="Met `{call}` stopt je functie met een fout: je functie deelt door een aantal van 0, en delen door 0 kan niet. Wat moet een automaat krijgen waaraan nooit getrokken is?"),
            Mistake("upside-down", wrong=_E_UPSIDE,
                    en="`{call}` gives {got}, and it should give {want}: you divide the count by the total. Which of the two goes on top in an average?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je deelt het aantal door het totaal. Welke van de twee staat boven de streep in een gemiddelde?"),
            Mistake("total", wrong=_E_TOTAL,
                    en="`{call}` gives {got}, and it should give {want}: those are the totals themselves. A machine pulled three times has three rewards in its total; what do you divide by to get the reward of one pull?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: dat zijn de totalen zelf. Na drie keer trekken heeft een automaat drie rewards in het totaal; waardoor deel je om de reward van één keer trekken te krijgen?"),
            Mistake("none", wrong=_E_NONE,
                    en="`{call}` gives {got}, and it should give {want}: a machine that was never pulled gets `None`, and `np.argmax` cannot compare `None` with numbers. Which number does the course give such a machine?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: als er nooit aan een automaat getrokken is, krijgt de automaat `None`, en `np.argmax` kan `None` niet met getallen vergelijken. Welk getal geeft de course zo'n automaat?"),
            Mistake("floor", wrong=_E_FLOOR,
                    en="`{call}` gives {got}, and it should give {want}: `//` rounds down to a whole number. Can you divide with `/`, so the estimate keeps its decimals?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `//` rondt af naar beneden, op een heel getal. Kun je delen met `/`, zodat de schatting de decimalen houdt?"),
            Mistake("all-machines", wrong=_E_ALL,
                    en="`{call}` gives {got}, and it should give {want}: every machine that was pulled gets the average over all pulls of all machines. Which total and which count belong to machine `i`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke automaat krijgt, zodra eraan getrokken is, het gemiddelde over alle keren trekken van alle automaten. Welk totaal en welk aantal horen bij automaat `i`?"),
            Mistake("tuple", matches=lambda got, text_: isinstance(got, tuple),
                    en="`{call}` gives {got}, a tuple, and it should give the list {want}. The start code makes `q` a list with `q = []`; can you give back `q` itself?",
                    nl="`{call}` geeft {got}, een tuple, en het moet de lijst {want} zijn. De startcode maakt van `q` een lijst met `q = []`; kun je `q` zelf teruggeven?"),
            Mistake("return-in-loop", wrong=_E_RETURN_IN,
                    en="`{call}` gives {got}, and it should give {want}: your list stops after the first machine. The `return` stands inside the loop, so the function ends during its first pass through the loop; where should it stand?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je lijst stopt na de eerste automaat. De `return` staat in de loop, dus de functie stopt al bij de eerste keer door de loop; waar hoort de `return` te staan?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: what a machine that was never pulled gets, and the sample average of the others.",
                      "nl": "Vervang de twee keer `...`: wat een automaat krijgt waaraan nooit getrokken is, en het steekproefgemiddelde van de andere."},
            "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has another number of names between its parentheses. Can you keep `(totals, counts)`?",
                          "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(totals, counts)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does every machine get its own total divided by its own count, and a machine with count 0 the value `0.0`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Krijgt elke automaat het eigen totaal gedeeld door het eigen aantal, en een automaat met aantal 0 de waarde `0.0`?"},
            "almost": {"en": "Right for the three machines of the example. But `{call}` gives {got}, and it should give {want}. Does your function work for any number of machines, and for whole numbers?",
                       "nl": "Goed voor de drie automaten van het voorbeeld. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie voor elk aantal automaten, en voor hele getallen?"},
            "unreached": {"en": "With `{call}`, your function gives back `None`, and it should give a list. Does it end with `return q`, after the loop?",
                          "nl": "Met `{call}` geeft je functie `None` terug, en het moet een lijst zijn. Eindigt je functie met `return q`, na de loop?"},
            "correct": {"en": "Right: machine 0 gets 3.0 / 2 = 1.5, machine 1 was never pulled and gets `0.0`, machine 2 gets −1.5 / 3 = −0.5. In the picture, an agent with ε = 0.1 uses your function for 200 pulls. Machine 9 pays most: why did the agent pull machine 0 more often?",
                        "nl": "Goed: automaat 0 krijgt 3,0 / 2 = 1,5, automaat 1 krijgt `0.0`, want er is nooit aan getrokken, automaat 2 krijgt −1,5 / 3 = −0,5. In de afbeelding gebruikt een agent met ε = 0,1 je functie 200 keer. Automaat 9 geeft het meest: waarom trok de agent vaker aan automaat 0?"},
        },
    ),
    "best_share": FunctionBox(
        name="best_share",
        picture=best_share_picture,
        function="best_share",
        names={"np": np},
        start=_S_START,
        solution=_S,
        show=_show,
        visible=SHARE_CASES[:1],
        hidden=SHARE_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("rows", wrong=_S_ROWS,
                    en="`{call}` gives {got}, and it should give {want}: one share for every run, the mean along each row. The share per pull is a mean over the runs: down the rows, one result per column. Which axis does that?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: één aandeel per run, het gemiddelde langs elke rij. Het aandeel per trek is een gemiddelde over de runs: langs de rijen omlaag, één uitkomst per kolom. Welke axis doet dat?"),
            Mistake("one-number", wrong=_S_ALL,
                    en="`{call}` gives {got}, and it should give {want}: one number, the share over all pulls of all runs together. The check wants one share for every pull; along which axis do you take the mean?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: één getal, het aandeel over alle keren trekken van alle runs samen. De check wil een aandeel per trek; langs welke axis neem je het gemiddelde?"),
            Mistake("count", wrong=_S_COUNT,
                    en="`{call}` gives {got}, and it should give {want}: for every pull, the number of runs that pulled the best machine. A share is that number divided by the number of runs; which function of NumPy does both at once?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: voor elke trek het aantal runs dat aan de beste automaat trok. Een aandeel is dat aantal gedeeld door het aantal runs; welke functie van NumPy doet beide in één keer?"),
            Mistake("machines", wrong=_S_ACTIONS,
                    en="`{call}` gives {got}, and it should give {want}: the mean of the machine numbers themselves. Which array says, for every pull, whether it was the best machine?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het gemiddelde van de nummers van de automaten zelf. Welke array zegt voor elke trek of het de beste automaat was?"),
            Mistake("hits", wrong=_S_HITS,
                    en="`{call}` gives {got}, and it should give {want}: the comparison itself, True or False for every pull of every run. What do you do with it to get one share per pull?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de vergelijking zelf, True of False voor elke trek van elke run. Wat doe je ermee om één aandeel per trek te krijgen?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: compare `actions` with `best`, and give back the share of True for every pull.",
                      "nl": "Vervang de twee keer `...`: vergelijk `actions` met `best`, en geef voor elke trek het aandeel True terug."},
            "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has another number of names between its parentheses. Can you keep `(actions, best)`?",
                          "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(actions, best)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function compare every pull with the best machine of its run, and take the mean over the runs?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Vergelijkt je functie elke trek met de beste automaat van de run, en neemt je functie het gemiddelde over de runs?"},
            "almost": {"en": "Right for the three small runs. But `{call}` gives {got}, and it should give {want}. Does your function work for one run, and for 2,000 runs of 1,000 pulls?",
                       "nl": "Goed voor de drie kleine runs. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie voor één run, en voor 2.000 runs van 1.000 keer trekken?"},
            "unreached": {"en": "With `{call}`, your function gives back `None`, and it should give an array. Does it end with a `return`?",
                          "nl": "Met `{call}` geeft je functie `None` terug, en het moet een array zijn. Eindigt je functie met een `return`?"},
            "correct": {"en": "Right: at the first pull no run took its best machine, at the last pull all three did. The picture shows your function on the course's runs. Which agent finds the best machine most often by pull 1,000?",
                        "nl": "Goed: bij de eerste trek nam geen enkele run de beste automaat, bij de laatste trek alle drie. De afbeelding toont je functie op de runs van de course. Welke agent vindt bij trek 1.000 het vaakst de beste automaat?"},
        },
    ),
}

CASES = [
    ("estimates", _E_START, "empty", None),
    ("estimates", _E, "correct", None),
    ("estimates", _e("0.0", "..."), "wrong", "dots-left"),
    ("estimates", _E_NO_IF, "error", "zero"),
    ("estimates", _E_UPSIDE, "wrong", "upside-down"),
    ("estimates", _E_TOTAL, "wrong", "total"),
    ("estimates", _E_NONE, "wrong", "none"),
    ("estimates", _E_FLOOR, "wrong", "floor"),
    ("estimates", _E_ALL, "wrong", "all-machines"),
    ("estimates", _E_RETURN_IN, "wrong", "return-in-loop"),
    ("estimates", _E.replace("    return q\n", "    return tuple(q)\n"), "wrong", "tuple"),
    ("estimates", "def estimates(totals, counts):\n    return [t / c if c > 0 else 0.0 for t, c in zip(totals, counts)]\n", "correct", None),
    ("estimates", "def estimates(totals, counts):\n    t = np.array(totals, dtype=float)\n    c = np.array(counts)\n    return np.where(c > 0, t / np.maximum(c, 1), 0.0)\n", "correct", None),
    ("estimates", _e("0.0", "totals[i] / (counts[i] + 1)"), "wrong", None),
    ("estimates", _e("0.0", "totals[i] / counts[i]").replace("range(len(totals))", "range(3)"), "almost", None),
    ("best_share", _S_START, "empty", None),
    ("best_share", _S, "correct", None),
    ("best_share", "def best_share(actions, best):\n    hits = actions == best\n    return ...\n", "wrong", "dots-left"),
    ("best_share", _S_ROWS, "wrong", "rows"),
    ("best_share", _S_ALL, "wrong", "one-number"),
    ("best_share", _S_COUNT, "wrong", "count"),
    ("best_share", _S_ACTIONS, "wrong", "machines"),
    ("best_share", _S_HITS, "wrong", "hits"),
    ("best_share", "def best_share(actions, best):\n    return np.mean(actions == best, axis=0)\n", "correct", None),
    ("best_share", "def best_share(actions, best):\n    shares = []\n    for t in range(actions.shape[1]):\n        shares.append(float(np.mean(actions[:, t] == best[:, 0])))\n    return shares\n", "correct", None),
    ("best_share", "def best_share(actions, best):\n    hits = actions == best\n    return hits.sum(axis=0) / (len(actions) + 1)\n", "wrong", None),
    ("best_share", "def best_share(actions, best):\n    hits = actions == best\n    return hits.sum(axis=0) / 3\n", "almost", None),
]
