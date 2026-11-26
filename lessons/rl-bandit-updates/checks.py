# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rl-bandit-updates: the incremental update of an estimate, with a learning rate of 1/n (the
sample average) or a constant one; bandits whose means drift at every pull; optimistic initial values; update and
reward_shares, the functions the learner writes, and their pictures; the demo on drifting bandits with the learner's
update, and the slider over the start value of a greedy agent.

The bandits follow Sutton and Barto (Reinforcement Learning: An Introduction, 2nd edition, 2018, sections 2.4 to 2.6):
the drifting bandits of their exercise 2.5, here with means that start around 0 with standard deviation 1, and the
optimistic start of their figure 2.3. The code, the runs and the pictures are the course's own."""
import ast
import copy
import html
import math
import numbers

import numpy as np

from everything_ai import chart, live, yours
from everything_ai.check import FunctionBox, Mistake, kept

K = 10                    # machines of a bandit
RUNS = 2000               # bandits of the course's runs
SEED = 1                  # the course's runs
EPSILON = 0.1             # the epsilon of the agents on drifting bandits, and of the realistic agent
ETA = 0.1                 # the constant learning rate
DRIFT = 0.01              # the standard deviation of the step every mean takes at every pull
DRIFT_STEPS = 10_000      # pulls of the course's runs on drifting bandits
STEPS = 1000              # pulls of the course's runs on bandits that stay the same (optimistic start, slider)
START = 5.0               # the optimistic start of exercise 3
AGENTS = ("average", "constant")   # learning rate 1/n and learning rate 0.1


# --- the course's solutions ----------------------------------------------------------------------------------------------

def update(q, reward, eta):
    """The course's solution of exercise 1: the estimate q moved a step eta toward the reward."""
    return q + eta * (reward - q)


def reward_shares(eta, n):
    """The course's solution of exercise 2: how much each of n rewards counts in the estimate after n updates with
    the constant learning rate eta; the first reward first, the last one last."""
    i = np.arange(1, n + 1)
    return eta * (1 - eta) ** (n - i)


# --- the course's runs -------------------------------------------------------------------------------------------------

def _runs(n: int, steps: int, epsilon: float, seed: int, eta: float | None = None, start: float = 0.0,
          drift: float = 0.0) -> tuple:
    """An agent on n bandits at once, with NumPy: epsilon-greedy, every estimate starting at `start` and moved toward
    each reward with learning rate `eta`, or 1/n when `eta` is None. With `drift`, every mean takes a random step of
    that standard deviation after every pull. Gives the mean reward over the bandits at every pull and the share of
    bandits whose pull took the best machine at that moment."""
    rng = np.random.default_rng(seed)
    means = rng.normal(0.0, 1.0, (n, K))
    q, counts = np.full((n, K), float(start)), np.zeros((n, K))
    reward, best = np.zeros(steps), np.zeros(steps)
    rows = np.arange(n)
    for t in range(steps):
        greedy = q.argmax(axis=1)
        random_ = rng.integers(0, K, n)
        a = np.where(rng.random(n) < epsilon, random_, greedy)
        r = means[rows, a] + rng.normal(0.0, 1.0, n)
        counts[rows, a] += 1
        step = 1.0 / counts[rows, a] if eta is None else eta
        q[rows, a] += step * (r - q[rows, a])
        reward[t], best[t] = r.mean(), (a == means.argmax(axis=1)).mean()
        if drift:
            means += rng.normal(0.0, drift, (n, K))
    return reward, best


def course_drift() -> dict:
    """The course's runs on drifting bandits: 2,000 bandits of 10,000 pulls, epsilon 0.1, with learning rate 1/n
    ("average") and 0.1 ("constant"), on the same bandits. For each: the mean reward and the share of best machines at
    every pull."""
    def make():
        return {name: _runs(RUNS, DRIFT_STEPS, EPSILON, SEED, eta=None if name == "average" else ETA, drift=DRIFT)
                for name in AGENTS}
    return kept("rl-bandit-updates:drift", make)


def course_optimistic() -> dict:
    """The course's runs on bandits that stay the same, 2,000 bandits of 1,000 pulls, learning rate 0.1: the realistic
    agent (start 0, epsilon 0.1) and the optimistic one (start 5, greedy)."""
    def make():
        return {"realistic": _runs(RUNS, STEPS, EPSILON, SEED, eta=ETA),
                "optimistic": _runs(RUNS, STEPS, 0.0, SEED, eta=ETA, start=START)}
    return kept("rl-bandit-updates:optimistic", make)


def mean_of(values, first: int, last: int) -> float:
    """The mean of values[first - 1 : last]: from pull `first` up to and including pull `last`."""
    return float(np.mean(values[first - 1: last]))


def drift_share(name: str, first: int, last: int) -> float:
    return mean_of(course_drift()[name][1], first, last)


def drift_reward(name: str, first: int, last: int) -> float:
    return mean_of(course_drift()[name][0], first, last)


# --- the machine of the pictures: its mean jumps -----------------------------------------------------------------------

JUMP_PULLS = 200   # pulls of the machine in the pictures
JUMP_AT = 100      # after this pull its mean jumps
JUMP_FROM, JUMP_TO = 0.0, 2.0
JUMP_SEED = 4


def jump_means() -> list:
    """The mean of the machine of the pictures at every pull: 0 for the first 100 pulls, then 2."""
    return [JUMP_FROM if t <= JUMP_AT else JUMP_TO for t in range(1, JUMP_PULLS + 1)]


def jump_rewards() -> list:
    """The 200 rewards of the machine of the pictures, around its mean with standard deviation 1."""
    rng = np.random.default_rng(JUMP_SEED)
    return [round(m + float(rng.normal()), 6) for m in jump_means()]


def follow(fn, constant: bool) -> list:
    """The estimate after every reward of the machine of the pictures, moved by `fn(q, reward, eta)` with learning
    rate 1/n, or 0.1 when `constant`; it starts at 0."""
    q, out = 0.0, []
    for n, r in enumerate(jump_rewards(), start=1):
        q = float(fn(q, r, ETA if constant else 1.0 / n))
        out.append(q)
    return out


# --- the demo: drifting bandits with the learner's update ---------------------------------------------------------------

DEMO_RUNS = 300
DEMO_STEPS = 10_000
DEMO_SEED = 5
WATCHED = 23          # the bandit of the demo that the picture beside the chart shows: at the end its best machine is
                      # one that 1/n has pulled little since its mean rose, so the picture shows what the lesson says
DEMO_LIMIT = 120.0    # seconds the demo gets
LOOKS = [1, 10, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10_000]


def _number(x) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, bool) and math.isfinite(x)


def train(fn, runs: int = DEMO_RUNS, steps: int = DEMO_STEPS, seed: int = DEMO_SEED, show=None):
    """The demo: two agents with epsilon 0.1 on the same `runs` drifting bandits, one with learning rate 1/n and one
    with 0.1, both moving their estimates with `fn(q, reward, eta)`. After every pull, `show(step, shares, seen)` gets
    the share of bandits in which each agent pulled the best machine, and what the agents know of the watched bandit.
    Gives the run, or "update" when `fn` stops with an error or gives no number."""
    rng = np.random.default_rng(seed)
    means = rng.normal(0.0, 1.0, (runs, K))
    q = {a: [[0.0] * K for _ in range(runs)] for a in AGENTS}
    n = {a: [[0] * K for _ in range(runs)] for a in AGENTS}
    curve = {a: [] for a in AGENTS}
    looks = {}
    for t in range(1, steps + 1):
        m = means.tolist()
        best = means.argmax(axis=1).tolist()
        explore = (rng.random((2, runs)) < EPSILON).tolist()
        random_ = rng.integers(0, K, (2, runs)).tolist()
        noise = rng.normal(0.0, 1.0, (2, runs)).tolist()
        for j, name in enumerate(AGENTS):
            qa, na, hits = q[name], n[name], 0
            for b in range(runs):
                qb = qa[b]
                a = random_[j][b] if explore[j][b] else max(range(K), key=qb.__getitem__)
                r = m[b][a] + noise[j][b]
                na[b][a] += 1
                try:
                    new = fn(qb[a], r, 1.0 / na[b][a] if j == 0 else ETA)
                except Exception:   # noqa: BLE001   the demo says which function stopped
                    return "update"
                if isinstance(new, np.ndarray) and new.shape == ():   # a NumPy answer can give a 0-d array
                    new = new.item()
                if not _number(new):
                    return "update"
                qb[a] = float(new)
                hits += a == best[b]
            curve[name].append(hits / runs)
        w = min(WATCHED, runs - 1)   # a short run of a test may have fewer bandits
        seen = {"means": m[w], **{name: {"q": list(q[name][w]), "counts": list(n[name][w])} for name in AGENTS}}
        if t in LOOKS:
            looks[t] = seen
        if show is not None:
            show(t, {name: curve[name][-1] for name in AGENTS}, seen)
        means += rng.normal(0.0, DRIFT, (runs, K))
    return {"curve": curve, "looks": looks, "runs": runs, "steps": steps}


def demo(fn, show=None):
    """The demo within its time limit: the run, "update" when the learner's function fails, or "slow"."""
    from everything_ai.check import Timeout, with_time_limit
    try:
        return with_time_limit(lambda: train(fn, show=show), DEMO_LIMIT)
    except Timeout:
        return "slow"


def mine(box_text, correct: bool):
    """The learner's update when its check passes, else None: the demo then uses the course's."""
    if not correct:
        return None
    from everything_ai.check import run_code
    fn = run_code(box_text, {"np": np}).names.get("update")
    return fn if callable(fn) else None


def smoothed(ys, k: int = 100) -> list:
    """ys as (pull in thousands, mean over the last k pulls) at every 10th pull, for a line of the demo's charts, whose
    x-axis counts thousands of pulls."""
    return [(t / 1000, y) for t, y in zip(range(1, len(ys) + 1), _smooth(ys, k)) if t % 10 == 0]


def _smooth(ys, k: int = 100) -> list:
    out, total = [], 0.0
    for i, y in enumerate(ys):
        total += y
        if i >= k:
            total -= ys[i - k]
        out.append(total / min(i + 1, k))
    return out


# --- the slider: the start value of a greedy agent ----------------------------------------------------------------------

STARTS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]


def try_runs() -> dict:
    """The greedy agent with learning rate 0.1 on the course's 2,000 bandits of 1,000 pulls, for every start value of
    the slider: its mean reward and its share of best machines at every pull."""
    def make():
        out = {}
        for s in STARTS:
            out[s] = course_optimistic()["optimistic"] if s == START else _runs(RUNS, STEPS, 0.0, SEED, eta=ETA, start=s)
        return out
    return kept("rl-bandit-updates:starts", make)


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


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma in Dutch, and a real minus sign."""
    return _dec(x, lang if lang in LANGS else "en", digits)


def percent(x: float, lang: str) -> str:
    return _pct(x, lang if lang in LANGS else "en")


def _svg(panels: list, alt: str, lang: str, width: int = 330, height: int = 250) -> str:
    out = chart.figure(panels, alt=alt, width=width, height=height)
    return chart.with_decimal_comma(out) if lang == "nl" else out


_WORDS = {
    "en": {"machine": "machine", "reward": "reward", "pull": "pull", "share": "share of bandits",
           "average": "1/n", "constant": "η = 0.1", "estimate": "estimate"},
    "nl": {"machine": "automaat", "reward": "reward", "pull": "trek", "share": "deel van de bandits",
           "average": "1/n", "constant": "η = 0,1", "estimate": "schatting"},
}


def _jump_panel(lines: dict, lang: str, title: str = "") -> chart.Panel:
    """The machine of the pictures: its rewards as gray dots, its mean as a line, and the estimates in `lines` (1/n
    dashed, 0.1 solid) in orange, each named at its end."""
    w = _WORDS[lang]
    p = chart.Panel((0, JUMP_PULLS), (-3.0, 5.0), title=title, xlabel=w["pull"], ylabel=w["reward"])
    p.points(list(range(1, JUMP_PULLS + 1)), jump_rewards(), "ink-3", r=1.8)
    p.line([0, JUMP_AT, JUMP_AT, JUMP_PULLS], [JUMP_FROM, JUMP_FROM, JUMP_TO, JUMP_TO], "ink", 2.0)
    for name, ys in lines.items():
        p.line(list(range(1, JUMP_PULLS + 1)), ys, "param", 2.4 if name == "constant" else 2.0, dashed=name == "average")
        p.text(JUMP_PULLS * 0.985, ys[-1] + (0.42 if ys[-1] >= max(v[-1] for v in lines.values()) else -0.42), w[name], "param", "end", halo=True)
    return p


# --- the picture of the idea -------------------------------------------------------------------------------------------

_IDEA_TEXT = {
    "en": {"title": "A machine whose mean jumps from 0 to 2 after 100 pulls",
           "alt": "200 rewards of one machine as gray dots, around a mean of 0 for the first 100 pulls and of 2 after that, drawn as a line. Two orange lines are estimates: the sample average, dashed, ends near 1; the estimate with learning rate 0.1 follows the jump and ends near 2.",
           "note": "Gray: the rewards. The line that jumps: the mean, which the agent does not know. Orange: the estimates, dashed with 1/n (the sample average), solid with η = 0.1."},
    "nl": {"title": "Een automaat waarvan het gemiddelde na 100 keer trekken van 0 naar 2 springt",
           "alt": "200 rewards van één automaat als grijze stippen, rond een gemiddelde van 0 bij de eerste 100 keer trekken en van 2 daarna, getekend als lijn. Twee oranje lijnen zijn schattingen: het steekproefgemiddelde, gestreept, eindigt bij 1; de schatting met learning rate 0,1 volgt de sprong en eindigt bij 2.",
           "note": "Grijs: de rewards. De lijn die springt: het gemiddelde, dat de agent niet kent. Oranje: de schattingen, gestreept met 1/n (het steekproefgemiddelde), doorgetrokken met η = 0,1."},
}


def idea_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang]
    p = _jump_panel({"average": follow(update, False), "constant": follow(update, True)}, lang)
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


_UPDATE_TEXT = {
    "en": {"title": "Your update on the machine of The idea",
           "head": "The machine of *The idea*, with your `update`: once with η = 1/n, once with η = 0.1.",
           "alt": "200 rewards of one machine as gray dots, its mean as a line that jumps from 0 to 2 after 100 pulls, and two orange lines: the estimates of your update with learning rate 1/n, dashed, and 0.1, solid.",
           "none_alt": "200 rewards of one machine as gray dots, and its mean as a line that jumps from 0 to 2 after 100 pulls.",
           "end": "After 200 pulls: with 1/n your estimate is {a}, with η = 0.1 it is {c}; the mean is now 2.",
           "same": "Your estimates are the course's after every pull.",
           "differ": "Your estimates differ from the course's: after pull {t}, with {who}, yours is {mine} and the course's {want}.",
           "none": "Your `update` stops with an error, takes too long or gives no number here, so the picture shows the rewards alone.",
           "note": "Gray: the rewards. The line that jumps: the mean. Orange: your estimates, dashed with 1/n, solid with η = 0.1."},
    "nl": {"title": "Je update op de automaat uit Het idee",
           "head": "De automaat uit *Het idee*, met je `update`: één keer met η = 1/n, één keer met η = 0,1.",
           "alt": "200 rewards van één automaat als grijze stippen, het gemiddelde als lijn die na 100 keer trekken van 0 naar 2 springt, en twee oranje lijnen: de schattingen van je update met learning rate 1/n, gestreept, en 0,1, doorgetrokken.",
           "none_alt": "200 rewards van één automaat als grijze stippen, en het gemiddelde als lijn die na 100 keer trekken van 0 naar 2 springt.",
           "end": "Na 200 keer trekken: met 1/n is je schatting {a}, met η = 0,1 is de schatting {c}; het gemiddelde is nu 2.",
           "same": "Je schattingen zijn na elke trek gelijk aan de schattingen van de course.",
           "differ": "Je schattingen verschillen van de schattingen van de course: na trek {t}, met {who}, is je schatting {mine} en de schatting van de course {want}.",
           "none": "Je `update` stopt met een fout, duurt te lang of geeft hier geen getal, dus de afbeelding toont alleen de rewards.",
           "note": "Grijs: de rewards. De lijn die springt: het gemiddelde. Oranje: je schattingen, gestreept met 1/n, doorgetrokken met η = 0,1."},
}


def update_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _UPDATE_TEXT[lang]
    fn = _fn(seen)
    head = _note(w["head"])
    got = None
    if fn is not None:
        got = _call(lambda: {name: _numbers(follow(fn, name == "constant"), JUMP_PULLS) for name in AGENTS})
        if not isinstance(got, dict) or any(v is None for v in got.values()):
            got = None
    if got is None:
        p = _jump_panel({}, lang)
        return yours.figure(w["title"], head + _svg([p], w["none_alt"], lang, width=520, height=260) + _note(w["none"], _LOSS_UI), alt=w["title"])
    p = _jump_panel(got, lang)
    body = head + _svg([p], w["alt"], lang, width=520, height=260)
    body += _note(w["end"].format(a=_dec(got["average"][-1], lang), c=_dec(got["constant"][-1], lang)))
    off = None
    for name in AGENTS:
        want = follow(update, name == "constant")
        for t in range(JUMP_PULLS):
            if abs(got[name][t] - want[t]) > 1e-9:
                off = (t, name, got[name][t], want[t])
                break
        if off:
            break
    if off is None:
        body += _note(w["same"])
    else:
        t, name, mine_, want_ = off
        body += _note(w["differ"].format(t=t + 1, who=_WORDS[lang][name], mine=_dec(mine_, lang, 3), want=_dec(want_, lang, 3)), _LOSS_UI)
    return yours.figure(w["title"], body + _note(w["note"]), alt=w["title"])


SHARE_N = 30   # rewards in the picture of exercise 2

_SHARE_TEXT = {
    "en": {"title": "Your reward_shares: how much each of 30 rewards counts",
           "head": "Your `reward_shares(0.1, 30)`, as a bar for every reward: the first reward on the left, the last on the right.",
           "alt": "30 bars that grow from left to right: how much each of 30 rewards counts in the estimate with learning rate 0.1, according to your function, and a dashed line at 1/30, what each reward counts in the sample average.",
           "none_alt": "A dashed line at 1/30: what each of 30 rewards counts in the sample average.",
           "together": "Together your shares are {s}: the start value keeps the other {rest}. The last reward counts {last}, the first only {first}.",
           "differ": "Your shares differ from the course's: reward {i} gets {mine} from your function, and {want} from the course's.",
           "none": "Your `reward_shares` stops with an error, takes too long or does not give back 30 numbers for `reward_shares(0.1, 30)`.",
           "note": "The bars: your shares with η = 0.1. Dashed: 1/30, the share of every reward in the sample average."},
    "nl": {"title": "Je reward_shares: hoeveel elk van 30 rewards meetelt",
           "head": "Je `reward_shares(0.1, 30)`, als een staaf voor elke reward: de eerste reward links, de laatste rechts.",
           "alt": "30 staven die groeien van links naar rechts: hoeveel elk van 30 rewards meetelt in de schatting met learning rate 0,1, volgens je functie, en een gestreepte lijn bij 1/30, wat elke reward meetelt in het steekproefgemiddelde.",
           "none_alt": "Een gestreepte lijn bij 1/30: wat elk van 30 rewards meetelt in het steekproefgemiddelde.",
           "together": "Samen zijn je aandelen {s}: de startwaarde houdt de andere {rest}. De laatste reward telt {last} mee, de eerste slechts {first}.",
           "differ": "Je aandelen verschillen van de aandelen van de course: reward {i} krijgt {mine} van je functie, en {want} van de functie van de course.",
           "none": "Je `reward_shares` stopt met een fout, duurt te lang of geeft geen 30 getallen terug voor `reward_shares(0.1, 30)`.",
           "note": "De staven: je aandelen met η = 0,1. Gestreept: 1/30, het aandeel van elke reward in het steekproefgemiddelde."},
}


def shares_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _SHARE_TEXT[lang]
    fn = _fn(seen)
    head = _note(w["head"])
    got = None if fn is None else _call(lambda: _numbers(fn(ETA, SHARE_N), SHARE_N))
    p = chart.Panel((0.3, SHARE_N + 0.7), (0, 0.12), xlabel=_WORDS[lang]["reward"], ylabel="")
    p.line([0.3, SHARE_N + 0.7], [1 / SHARE_N, 1 / SHARE_N], "ink-3", 1.4, dashed=True)
    p.text(1.0, 1 / SHARE_N + 0.006, "1/30", "ink-3", "start", halo=True)
    if not isinstance(got, list):
        return yours.figure(w["title"], head + _svg([p], w["none_alt"], lang, width=520, height=240) + _note(w["none"], _LOSS_UI), alt=w["title"])
    for i, v in enumerate(got, start=1):
        top = max(0.0, min(0.12, v))
        p.line([i, i], [0, top], "ink", 7.0)
    want = reward_shares(ETA, SHARE_N).tolist()
    total = sum(got)
    body = head + _svg([p], w["alt"], lang, width=520, height=240)
    off = [i for i in range(SHARE_N) if abs(got[i] - want[i]) > 1e-9]
    if not off:   # what the shares add up to says something only about the right shares
        body += _note(w["together"].format(s=_dec(total, lang, 3), rest=_dec(1 - total, lang, 3), last=_dec(got[-1], lang, 3), first=_dec(got[0], lang, 3)))
    else:
        i = off[0]
        body += _note(w["differ"].format(i=i + 1, mine=_dec(got[i], lang, 4), want=_dec(want[i], lang, 4)), _LOSS_UI)
    return yours.figure(w["title"], body + _note(w["note"]), alt=w["title"])


# --- the demo's pictures -----------------------------------------------------------------------------------------------

_LIVE_TEXT = {
    "en": {"title": "Your run: how often the best machine", "x": "pull, in thousands", "y": "share of bandits",
           "alt": "The share of 300 drifting bandits in which the agent with learning rate 0.1 pulled the best machine, over every 10 pulls so far, thin, and its mean over the last 100 pulls, thick; dashed, the mean over the last 100 pulls of the agent with learning rate 1/n.",
           "rewind_alt": "The share of 300 drifting bandits in which each agent pulled the best machine, as its mean over the last 100 pulls over the whole run: learning rate 0.1 solid, 1/n dashed, with a vertical line at the chosen pull.",
           "now": "One of the 300 bandits after pull {t}",
           "now_alt": "Two panels, learning rate 1/n and 0.1: the ten machines of one bandit, each with its mean at this pull as a short line, the agent's estimate as an orange dot and its number of pulls above it."},
    "nl": {"title": "Je run: hoe vaak de beste automaat", "x": "trek, in duizendtallen", "y": "deel van de bandits",
           "alt": "Het deel van 300 verschuivende bandits waarin de agent met learning rate 0,1 aan de beste automaat trok, over elke 10 keer trekken tot nu toe, dun, en het gemiddelde over de laatste 100 keer trekken, dik; gestreept het gemiddelde over de laatste 100 keer trekken van de agent met learning rate 1/n.",
           "rewind_alt": "Het deel van 300 verschuivende bandits waarin elke agent aan de beste automaat trok, als het gemiddelde over de laatste 100 keer trekken over de hele run: learning rate 0,1 doorgetrokken, 1/n gestreept, met een verticale lijn bij de gekozen trek.",
           "now": "Eén van de 300 bandits na trek {t}",
           "now_alt": "Twee panelen, learning rate 1/n en 0,1: de tien automaten van één bandit, elk met het gemiddelde bij deze trek als kort streepje, de schatting van de agent als oranje stip en erboven hoe vaak eraan getrokken is."},
}


def live_words(lang: str) -> dict:
    return _LIVE_TEXT[lang if lang in LANGS else "en"]


def _bandit_panel(means, q, counts, title: str, lang: str, low: float = -3.5, high: float = 4.5) -> chart.Panel:
    """One bandit: the mean of every machine as a short line, the agent's estimate as an orange dot, open for a machine
    it never pulled, and above each machine its number of pulls, the odd machines a row lower, so that numbers of four
    digits stay apart."""
    w = _WORDS[lang]
    p = chart.Panel((-0.6, K - 0.4), (low, high), title=title, xlabel=w["machine"], ylabel=w["reward"],
                    xticks=list(range(K)))
    for a in range(K):
        p.line([a - 0.34, a + 0.34], [means[a], means[a]], "ink", 2.2)
    tried = [a for a in range(K) if counts[a] > 0]
    never = [a for a in range(K) if counts[a] == 0]
    if tried:
        p.points(tried, [max(low, min(high, q[a])) for a in tried], "param", r=4.2)
    if never:
        p.points(never, [max(low, min(high, q[a])) for a in never], "param", r=3.6, hollow=True)
    for a in range(K):
        p.text(a, high - (0.35 if a % 2 == 0 else 1.0), f"{counts[a]}", "ink-2")
    return p


def now_picture(seen: dict, step: int, lang: str) -> str:
    """What the two agents know of the watched bandit after `step` pulls, beside its means at that pull."""
    lang = lang if lang in LANGS else "en"
    w = _LIVE_TEXT[lang] | _WORDS[lang]
    panels = [_bandit_panel(seen["means"], seen[name]["q"], seen[name]["counts"], w[name], lang) for name in AGENTS]
    title = w["now"].format(t=_dec(step, lang, 0))
    return yours.figure(title, _svg(panels, w["now_alt"], lang, width=300, height=240), alt=title)


_STYLE = {"average": ("ink-2", 1.8, True), "constant": ("ink", 2.6, False)}


def curve_picture(run: dict, lang: str, at: int | None = None) -> str:
    """The two agents of a run, each as its share of best machines over the last 100 pulls, with a line at pull `at`."""
    lang = lang if lang in LANGS else "en"
    w = _LIVE_TEXT[lang] | _WORDS[lang]
    steps = run["steps"]
    p = chart.Panel((0, steps / 1000), (0, 1), title=w["title"], xlabel=w["x"], ylabel=w["y"])
    ends = {}
    for name in AGENTS:
        role, width, dashed = _STYLE[name]
        ys = smoothed(run["curve"][name])   # every 10th pull: 1,000 points for a run of 10,000
        p.line([t for t, _ in ys], [y for _, y in ys], role, width, dashed=dashed)
        ends[name] = float(np.mean(run["curve"][name][-200:]))
    for name in AGENTS:
        other = ends["average" if name == "constant" else "constant"]
        p.text(steps / 1000 * 0.98, ends[name] + (0.07 if ends[name] >= other else -0.08), w[name], _STYLE[name][0], "end", halo=True)
    if at is not None:
        p.line([at / 1000, at / 1000], [0, 1], "ink-2", 1.2)
    return _svg([p], w["rewind_alt"], lang, width=330, height=240)   # beside the agents in the rewind: two charts of 330 px


def rewind_frame(run: dict, step: int, lang: str) -> str:
    return live.rewound(curve_picture(run, lang, at=step), [now_picture(run["looks"][step], step, lang)])


_REWIND_TEXT = {
    "en": "After pull {t}: the best machine of this bandit is machine {b}, with a mean of {m}. With 1/n the agent estimates it at {qa}, with η = 0.1 at {qc}.",
    "nl": "Na trek {t}: de beste automaat van deze bandit is automaat {b}, met een gemiddelde van {m}. Met 1/n schat de agent de automaat op {qa}, met η = 0,1 op {qc}.",
}


def rewind_text(run: dict, step: int, lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    seen = run["looks"][step]
    b = max(range(K), key=lambda a: seen["means"][a])
    return _REWIND_TEXT[lang].format(t=_dec(step, lang, 0), b=b, m=_dec(seen["means"][b], lang), qa=_dec(seen["average"]["q"][b], lang),
                                     qc=_dec(seen["constant"]["q"][b], lang))


# --- the slider ----------------------------------------------------------------------------------------------------------

_TRY_TEXT = {
    "en": {"title": "Greedy with start {s}: how often the best machine",
           "alt": "The share of 2,000 bandits in which the greedy agent with the chosen start value and learning rate 0.1 pulled the best machine, at every pull of 1,000, with the agent that starts at 0 and has epsilon 0.1 dashed.",
           "realistic": "start 0, ε = 0.1",
           "said": "Start {s}: over the last 100 pulls the best machine in {p} of the bandits; over all 1,000 pulls {r} per pull."},
    "nl": {"title": "Greedy met start {s}: hoe vaak de beste automaat",
           "alt": "Het deel van 2.000 bandits waarin de greedy agent met de gekozen startwaarde en learning rate 0,1 aan de beste automaat trok, bij elk van 1.000 keer trekken, met de agent die bij 0 begint en epsilon 0,1 heeft gestreept.",
           "realistic": "start 0, ε = 0,1",
           "said": "Start {s}: over de laatste 100 keer trekken in {p} van de bandits de beste automaat; over alle 1.000 keer trekken {r} per keer."},
}


def _start_text(s: float, lang: str) -> str:
    out = f"{s:g}"
    return out.replace(".", ",") if lang == "nl" else out


def try_frame(start: float, lang: str) -> tuple:
    lang = lang if lang in LANGS else "en"
    w = _TRY_TEXT[lang]
    runs = try_runs()
    s = min(STARTS, key=lambda v: abs(v - start))
    reward, best = runs[s]
    p = chart.Panel((0, STEPS), (0, 1), title=w["title"].format(s=_start_text(s, lang)), xlabel=_WORDS[lang]["pull"], ylabel=_WORDS[lang]["share"])
    ref = _smooth(list(course_optimistic()["realistic"][1]), 10)
    p.line(list(range(1, STEPS + 1)), ref, "ink-3", 1.6, dashed=True)
    p.text(STEPS * 0.98, float(np.mean(ref[-50:])) - 0.08, w["realistic"], "ink-3", "end", halo=True)
    p.line(list(range(1, STEPS + 1)), _smooth(list(best), 10), "ink", 2.4)
    said = w["said"].format(s=_start_text(s, lang), p=_pct(mean_of(best, STEPS - 99, STEPS), lang), r=_dec(float(np.mean(reward)), lang))
    return _svg([p], w["alt"], lang, width=460, height=260), said


# --- the exercises -----------------------------------------------------------------------------------------------------

REQUIRED = ["update", "reward_shares", "spike"]


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


def _show(v):
    if isinstance(v, np.ndarray) and v.size > 12:
        head = ", ".join(_value(x) for x in v.ravel()[:4].tolist())
        return {"en": f"an array of {v.size} values that starts with {head}",
                "nl": f"een array van {v.size} waarden, beginnend met {head}"}
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

# exercise 1: update
_U_START = "def update(q, reward, eta):\n    return ...\n"
_U = "def update(q, reward, eta):\n    return q + eta * (reward - q)\n"
_U_AWAY = _U.replace("(reward - q)", "(q - reward)")
_U_NO_Q = _U.replace("q + eta * (reward - q)", "eta * (reward - q)")
_U_TARGET = _U.replace("q + eta * (reward - q)", "reward")
_U_REWARD = _U.replace("q + eta * (reward - q)", "q + eta * reward")
_U_DIVIDE = _U.replace("q + eta * (reward - q)", "q + (reward - q) / eta")
_U_MEAN = _U.replace("q + eta * (reward - q)", "(q + reward) / 2")

UPDATE_CASES = [
    (1.0, 3.0, 0.5),
    (0.0, 2.0, 1.0),
    (2.0, 0.0, 0.5),
    (1.0, 4.0, 1 / 3),
    (5.0, 1.2, 0.1),
    (-1.0, -3.0, 0.25),
    (0.4, 0.4, 0.1),
]

# exercise 2: reward_shares
_R_START = "def reward_shares(eta, n):\n    i = np.arange(1, n + 1)\n    return ...\n"
_R = "def reward_shares(eta, n):\n    i = np.arange(1, n + 1)\n    return eta * (1 - eta) ** (n - i)\n"
_R_REVERSED = _R.replace("(n - i)", "(i - 1)")
_R_ONE_MORE = _R.replace("(n - i)", "(n - i + 1)")
_R_NO_ETA = _R.replace("eta * (1 - eta) ** (n - i)", "(1 - eta) ** (n - i)")
_R_ETA_POWER = _R.replace("eta * (1 - eta) ** (n - i)", "eta ** (n - i)")
_R_EQUAL = _R.replace("eta * (1 - eta) ** (n - i)", "eta * np.ones(n)")
_R_SUM = _R.replace("eta * (1 - eta) ** (n - i)", "(eta * (1 - eta) ** (n - i)).sum()")

SHARES_CASES = [(0.5, 3), (0.1, 5), (1.0, 3), (0.25, 1), (0.1, 30), (0.2, 4)]


EXERCISES = {
    "update": FunctionBox(
        name="update",
        picture=update_picture,
        function="update",
        names={"np": np},
        start=_U_START,
        solution=_U,
        show=_show,
        visible=UPDATE_CASES[:1],
        hidden=UPDATE_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("away", wrong=_U_AWAY,
                    en="`{call}` gives {got}, and it should give {want}: your estimate moves away from the reward. The reward 3.0 lies above the estimate 1.0; in which order do you subtract, so that the step goes up?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je schatting gaat van de reward af. De reward 3,0 ligt boven de schatting 1,0; in welke volgorde trek je af, zodat de stap omhoog gaat?"),
            Mistake("no-q", wrong=_U_NO_Q,
                    en="`{call}` gives {got}, and it should give {want}: that is the step alone. The new estimate is the old one plus the step; what do you add the step to?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: dat is alleen de stap. De nieuwe schatting is de oude plus de stap; waar tel je de stap bij op?"),
            Mistake("target", wrong=_U_TARGET,
                    en="`{call}` gives {got}, and it should give {want}: the estimate jumps all the way to the reward, so it forgets every reward before. How big a part of the way does `eta` say to go?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de schatting springt helemaal naar de reward, dus vergeet de schatting alle eerdere rewards. Welk deel van de weg zegt `eta`?"),
            Mistake("reward-only", wrong=_U_REWARD,
                    en="`{call}` gives {got}, and it should give {want}: you add a part of the reward itself. The step is a part of the distance from the estimate to the reward; how do you write that distance?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je telt een deel van de reward zelf op. De stap is een deel van de afstand van de schatting tot de reward; hoe schrijf je de afstand?"),
            Mistake("divide", wrong=_U_DIVIDE,
                    en="`{call}` gives {got}, and it should give {want}: you divide by `eta`, so a small learning rate gives a huge step. With η = 1/n the step is the distance divided by n; what do you do with `eta` to get that?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je deelt door `eta`, dus een kleine learning rate geeft een enorme stap. Met η = 1/n is de stap de afstand gedeeld door n; wat doe je met `eta` om dat te krijgen?"),
            Mistake("halfway", wrong=_U_MEAN,
                    en="`{call}` gives {got}, and it should give {want}: you always go halfway, whatever `eta` is. Is `eta` in your line?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je gaat altijd halverwege, wat `eta` ook is. Staat `eta` in je regel?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the new estimate: `q` moved a step `eta` toward `reward`.",
                      "nl": "Vervang de `...` door de nieuwe schatting: `q` een stap `eta` opgeschoven naar `reward`."},
            "arguments": {"en": "The check calls `{call}`, with three values, and the `def` line of your function has another number of names between its parentheses. Can you keep `(q, reward, eta)`?",
                          "nl": "De check roept `{call}` aan, met drie waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(q, reward, eta)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is the new estimate the old one plus `eta` times the distance from the old estimate to the reward?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Is de nieuwe schatting de oude plus `eta` keer de afstand van de oude schatting tot de reward?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for every estimate, reward and learning rate?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie voor elke schatting, reward en learning rate?"},
            "correct": {"en": "Right: from 1.0 toward 3.0, half of the distance 2.0, gives 2.0. In the picture your function follows the machine of *The idea*: with 1/n your estimates are the sample average, with 0.1 they follow the jump. Which line still holds on to the first 100 pulls?",
                        "nl": "Goed: de schatting 1,0 schuift de helft van de afstand 2,0 op naar 3,0, en komt op 2,0. In de afbeelding volgt je functie de automaat uit *Het idee*: met 1/n zijn je schattingen het steekproefgemiddelde, met 0,1 volgen de schattingen de sprong. Welke lijn houdt nog vast aan de eerste 100 keer trekken?"},
        },
    ),
    "reward_shares": FunctionBox(
        name="reward_shares",
        picture=shares_picture,
        function="reward_shares",
        names={"np": np},
        start=_R_START,
        solution=_R,
        show=_show,
        visible=SHARES_CASES[:1],
        hidden=SHARES_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("reversed", wrong=_R_REVERSED,
                    en="`{call}` gives {got}, and it should give {want}: the first reward counts most and the last least. Reward `n` came last; how many updates followed it?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de eerste reward telt het zwaarst en de laatste het minst. Reward `n` kwam als laatste; hoeveel updates kwamen daarna nog?"),
            Mistake("one-more", wrong=_R_ONE_MORE,
                    en="`{call}` gives {got}, and it should give {want}: even the last reward is multiplied by 1 − η once. After the last reward no update followed; what power belongs to it?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: zelfs de laatste reward wordt één keer met 1 − η vermenigvuldigd. Na de laatste reward kwam geen update meer; welke macht hoort daarbij?"),
            Mistake("no-eta", wrong=_R_NO_ETA,
                    en="`{call}` gives {got}, and it should give {want}: the last reward counts 1, as if the estimate jumped all the way to it. An update moves only a part `eta` of the way; where does that part go?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de laatste reward telt voor 1, alsof de schatting er helemaal naartoe sprong. Een update schuift slechts een deel `eta` van de weg op; waar komt dat deel?"),
            Mistake("eta-power", wrong=_R_ETA_POWER,
                    en="`{call}` gives {got}, and it should give {want}: every later update multiplies by `eta`. What is left of an old reward's part after an update is 1 − η of it; which number goes under the power?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke latere update vermenigvuldigt met `eta`. Na een update is er van het deel van een oude reward nog 1 − η over; welk getal hoort onder de macht?"),
            Mistake("equal", wrong=_R_EQUAL,
                    en="`{call}` gives {got}, and it should give {want}: every reward counts the same. Each later update shrinks the part of an old reward; does `i` appear in your line?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke reward telt even zwaar. Elke latere update laat het deel van een oude reward krimpen; staat `i` in je regel?"),
            Mistake("sum", wrong=_R_SUM,
                    en="`{call}` gives {got}, and it should give {want}: one number, all shares added up. The check wants one share for every reward, as an array; can you leave out `.sum()`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: één getal, alle aandelen opgeteld. De check wil een aandeel voor elke reward, als array; kun je `.sum()` weglaten?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with how much each reward counts: `eta` for the last one, and a factor 1 − η for every update after a reward.",
                      "nl": "Vervang de `...` door hoeveel elke reward meetelt: `eta` bij de laatste, en een factor 1 − η voor elke update na een reward."},
            "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has another number of names between its parentheses. Can you keep `(eta, n)`?",
                          "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(eta, n)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does the last reward get `eta`, and does every update after a reward multiply its part by 1 − η?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Krijgt de laatste reward `eta`, en vermenigvuldigt elke update na een reward het deel met 1 − η?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for every learning rate and every number of rewards?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie voor elke learning rate en elk aantal rewards?"},
            "unreached": {"en": "With `{call}`, your function gives back `None`, and it should give an array. Does it end with `return`?",
                          "nl": "Met `{call}` geeft je functie `None` terug, en het moet een array zijn. Eindigt je functie met `return`?"},
            "correct": {"en": "Right: with η = 0.5 the last of three rewards counts 0.5, the middle one 0.25 and the first 0.125. In the picture, the last of 30 rewards counts three times as much as with the sample average; what is left over for the start value?",
                        "nl": "Goed: met η = 0,5 telt de laatste van drie rewards voor 0,5, de middelste voor 0,25 en de eerste voor 0,125. In de afbeelding telt de laatste van 30 rewards drie keer zo zwaar als bij het steekproefgemiddelde; wat blijft er over voor de startwaarde?"},
        },
    ),
}

CASES = [
    ("update", _U_START, "empty", None),
    ("update", _U, "correct", None),
    ("update", "def update(q, reward, eta):\n    step = eta * (reward - q)\n    return ...\n", "wrong", "dots-left"),
    ("update", _U_AWAY, "wrong", "away"),
    ("update", _U_NO_Q, "wrong", "no-q"),
    ("update", _U_TARGET, "wrong", "target"),
    ("update", _U_REWARD, "wrong", "reward-only"),
    ("update", _U_DIVIDE, "wrong", "divide"),
    ("update", _U_MEAN, "wrong", "halfway"),
    ("update", "def update(q, reward, eta):\n    return (1 - eta) * q + eta * reward\n", "correct", None),
    ("update", "def update(q, reward, eta):\n    step = eta * (reward - q)\n    q = q + step\n    return q\n", "correct", None),
    ("update", "def update(q, reward, eta):\n    return q + eta * (reward - q) + 0.01\n", "wrong", None),
    ("update", "def update(q, reward, eta):\n    return q + eta * (reward - q) if q >= 0 else reward\n", "almost", None),
    ("reward_shares", _R_START, "empty", None),
    ("reward_shares", _R, "correct", None),
    ("reward_shares", "def reward_shares(eta, n):\n    i = ...\n    return eta * i\n", "wrong", "dots-left"),
    ("reward_shares", _R_REVERSED, "wrong", "reversed"),
    ("reward_shares", _R_ONE_MORE, "wrong", "one-more"),
    ("reward_shares", _R_NO_ETA, "wrong", "no-eta"),
    ("reward_shares", _R_ETA_POWER, "wrong", "eta-power"),
    ("reward_shares", _R_EQUAL, "wrong", "equal"),
    ("reward_shares", _R_SUM, "wrong", "sum"),
    ("reward_shares", "def reward_shares(eta, n):\n    return np.array([eta * (1 - eta) ** (n - k) for k in range(1, n + 1)])\n", "correct", None),
    ("reward_shares", "def reward_shares(eta, n):\n    shares = []\n    for k in range(n):\n        shares.append(eta * (1 - eta) ** (n - 1 - k))\n    return np.array(shares)\n", "correct", None),
    ("reward_shares", "def reward_shares(eta, n):\n    i = np.arange(1, n + 1)\n    return eta * (1 - eta) ** (n - i) / 2\n", "wrong", None),
    ("reward_shares", "def reward_shares(eta, n):\n    return np.array([0.125, 0.25, 0.5])\n", "almost", None),
    ("reward_shares", "def reward_shares(eta, n):\n    i = np.arange(1, n + 1)\n    shares = eta * (1 - eta) ** (n - i)\n", "wrong", "no-return"),
]
