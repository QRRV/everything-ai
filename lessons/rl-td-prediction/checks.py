# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rl-td-prediction: on a random walk of five states, A to E, TD(0) estimates the values after
every step, from the reward and its own estimate of the next state (bootstrapping), and Monte Carlo with a fixed
learning rate after every walk, from the return. The course's walks, its runs of both methods side by side, td_update
and mc_update, the functions the learner writes, and their pictures; the play block and the demo.

TD(0) and the random walk as in Sutton and Barto (Reinforcement Learning: An Introduction, 2nd edition, 2018, sections
6.1 to 6.3, example 6.2); the code, the runs and the pictures are the course's own."""
import ast
import html
import math
import random
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from everything_ai import chart, live, yours
from everything_ai.check import FunctionBox, Mistake

LANGS = ("en", "nl")
NAMES = ("A", "B", "C", "D", "E")   # the states 1 to 5; 0 is the left end and 6 the right end
STATES = (1, 2, 3, 4, 5)
LEFT, RIGHT, CENTER = 0, 6, 3       # every walk starts in C
TRUE = (0.0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 0.0)   # the chance to end on the right, from each state
GAMMA = 1.0         # the walk's rewards all count the same
ETA = 0.1         # the learning rate of the play block, for both methods
RUNS = 100          # the runs of the play block and of the demo, side by side
N_WALKS = 100       # the walks of every run
LOOKS = [0, 1, 2, 3, 5, 10, 20, 50, 100]   # the stops of the play block and of the rewind
ETAS = (0.01, 0.02, 0.05, 0.1, 0.15)     # what the demo lets you pick


# --- the walk -----------------------------------------------------------------------------------------------------------

class Walk(list):
    """A walk as a list of steps (s, r, s_next): the state, the reward of the step, and the state it led to. A check
    message calls it by its name, such as WALK; the learner's code gets a plain copy."""

    def __init__(self, steps, name: str = ""):
        super().__init__(steps)
        self.name = name

    def __repr__(self):
        return self.name or list.__repr__(self)

    def __deepcopy__(self, memo):
        return [tuple(step) for step in self]


class Walks(list):
    """A list of walks that a check message calls by its name, such as [WALK, WALK]."""

    def __init__(self, walks, name: str = ""):
        super().__init__(walks)
        self.name = name

    def __repr__(self):
        return self.name or list.__repr__(self)

    def __deepcopy__(self, memo):
        return [[tuple(step) for step in w] for w in self]


def start_values() -> list:
    """V before the first walk: 0.5 for A to E, and 0 at both ends, where nothing more comes."""
    return [0.0] + [0.5] * 5 + [0.0]


def walk(rng) -> Walk:
    """One walk from C: a step left or right with a chance of 1/2 each, until an end; the step onto the right end pays
    1, every other step 0."""
    s, steps = CENTER, []
    while s not in (LEFT, RIGHT):
        nxt = s + (1 if rng.random() < 0.5 else -1)
        steps.append((s, 1 if nxt == RIGHT else 0, nxt))
        s = nxt
    return Walk(steps)


def error(V) -> float:
    """How far V is from the true values: the average distance over A to E."""
    return sum(abs(V[s] - TRUE[s]) for s in STATES) / len(STATES)


def td_update(V, s, r, s_next, eta, gamma):
    """The course's solution of exercise 1: one step of TD(0)."""
    delta = r + gamma * V[s_next] - V[s]
    V[s] = V[s] + eta * delta


def mc_update(V, walk, eta, gamma):
    """The course's solution of exercise 2: Monte Carlo with a fixed learning rate after a whole walk, at every visit,
    from the end back to the start."""
    G = 0
    for s, r, s_next in reversed(walk):
        G = r + gamma * G
        V[s] = V[s] + eta * (G - V[s])


def _mc_forward(V, walk, eta, gamma):
    """The same updates from the start to the end: the other right order, which the check accepts too."""
    G = [0.0] * len(walk)
    g = 0
    for t in range(len(walk) - 1, -1, -1):
        g = walk[t][1] + gamma * g
        G[t] = g
    for t, (s, _r, _n) in enumerate(walk):
        V[s] = V[s] + eta * (G[t] - V[s])


@lru_cache(maxsize=1)
def course_walks() -> tuple:
    """The walks of the play block and of the demo: run k plays its 100 walks with the seed k."""
    out = []
    for k in range(RUNS):
        rng = random.Random(k)
        out.append(tuple(walk(rng) for _ in range(N_WALKS)))
    return tuple(out)


WALK = Walk([(3, 0, 4), (4, 0, 5), (5, 1, 6)], "WALK")             # C, D, E and out on the right: the worked example
LEFT_WALK = Walk([(3, 0, 2), (2, 0, 1), (1, 0, 0)], "LEFT_WALK")   # C, B, A and out on the left
# C, D, C, D, E, D, E and out on the right: C twice, D three times, E twice
LOOP_WALK = Walk([(3, 0, 4), (4, 0, 3), (3, 0, 4), (4, 0, 5), (5, 0, 4), (4, 0, 5), (5, 1, 6)], "LOOP_WALK")


# --- the course's runs --------------------------------------------------------------------------------------------------

def _fine(V) -> bool:
    try:
        return len(V) == 7 and all(math.isfinite(float(v)) and abs(float(v)) < 1e6 for v in V)
    except (TypeError, ValueError):
        return False


def run(td_fn=td_update, mc_fn=mc_update, eta_td: float = ETA, eta_mc: float = ETA, show=None,
        limit: float = 60.0) -> dict | str:
    """TD(0) and Monte Carlo side by side on the same walks, 100 runs at once: after walk n of every run, the average
    error over the runs, the error of each run, and the estimates of the first run. `show(n, run)` hears every walk. A
    function that stops with an error, or leaves V without a number for every state, ends the run: then "td" or "mc",
    or "slow" after `limit` seconds."""
    walks = course_walks()
    tds = [start_values() for _ in range(RUNS)]
    mcs = [start_values() for _ in range(RUNS)]
    e0 = error(start_values())
    out = {"td": {0: e0}, "mc": {0: e0}, "each_td": [[e0] for _ in range(RUNS)], "each_mc": [[e0] for _ in range(RUNS)],
           "first_td": {0: start_values()}, "first_mc": {0: start_values()}, "eta_td": eta_td, "eta_mc": eta_mc}
    begin = time.perf_counter()
    for n in range(1, N_WALKS + 1):
        for k in range(RUNS):
            w = walks[k][n - 1]
            try:
                for s, r, s_next in w:
                    td_fn(tds[k], s, r, s_next, eta_td, GAMMA)
            except Exception:   # noqa: BLE001   the demo says which function to look at
                return "td"
            if not _fine(tds[k]):
                return "td"
            try:
                mc_fn(mcs[k], [tuple(step) for step in w], eta_mc, GAMMA)
            except Exception:   # noqa: BLE001
                return "mc"
            if not _fine(mcs[k]):
                return "mc"
            out["each_td"][k].append(error(tds[k]))
            out["each_mc"][k].append(error(mcs[k]))
        out["td"][n] = sum(e[n] for e in out["each_td"]) / RUNS
        out["mc"][n] = sum(e[n] for e in out["each_mc"]) / RUNS
        out["first_td"][n] = [float(v) for v in tds[0]]
        out["first_mc"][n] = [float(v) for v in mcs[0]]
        if show is not None:
            show(n, out)
        if time.perf_counter() - begin > limit:
            return "slow"
    return out


@lru_cache(maxsize=1)
def play_run() -> dict:
    """The run of the play block: both methods with η = 0.1."""
    return run()


def course_errors() -> tuple:
    """The NumPy cell: the error of every run after every walk, for TD(0) and for Monte Carlo, both with η = 0.1, as
    two arrays of shape (100, 101): a row per run, a column per number of walks from 0 to 100."""
    r = play_run()
    return np.array(r["each_td"]), np.array(r["each_mc"])


@lru_cache(maxsize=None)
def demo_run(eta_td: float, eta_mc: float) -> dict:
    """The demo with the course's functions: what the text after the demo says."""
    return run(eta_td=eta_td, eta_mc=eta_mc)


# --- drawing ------------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem"
_HEAD = "font-family:var(--ea-font-ui);font-size:0.9rem;font-weight:600;margin-bottom:0.2rem"


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


def _head(text_: str) -> str:
    return f'<div style="{_HEAD}">{html.escape(text_)}</div>'


def _dec(x: float, lang: str, digits: int = 2) -> str:
    out = f"{x:,.{digits}f}"
    if out.startswith("-") and float(out.replace(",", "")) == 0:
        out = out[1:]
    out = out.replace("-", "−")
    return out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch, and a real minus sign."""
    return _dec(x, lang if lang in LANGS else "en", digits)


def eta_text(a: float, lang: str) -> str:
    """A learning rate as the lesson writes it: 0.1, or 0,1 in Dutch."""
    text_ = f"{a:g}"
    return text_.replace(".", ",") if lang == "nl" else text_


def eta_options(lang: str) -> dict:
    """The choices of the demo's two menus: the learning rate as written, and its value."""
    return {eta_text(a, lang): a for a in ETAS}


def _svg(panels: list, alt: str, lang: str, width: int = 330, height: int = 250) -> str:
    out = chart.figure(panels, alt=alt, width=width, height=height)
    return chart.with_decimal_comma(out) if lang == "nl" else out


def _values_panel(lines: list, title: str, lang: str) -> chart.Panel:
    """Estimates of A to E against the true values: `lines` holds (V, dashed) pairs, solid with dots or dashed with open
    dots, in orange, what is learned; the true values as a gray dashed line."""
    w = _WORDS[lang]
    p = chart.Panel((0.6, 5.4), (0, 1), title=title, xlabel=w["state"], ylabel=w["value"], xticks=list(STATES),
                    xtick_labels=list(NAMES), yticks=[0, 0.25, 0.5, 0.75, 1])
    p.line(list(STATES), [TRUE[s] for s in STATES], "ink-3", 1.4, dashed=True)
    for V, dashed in lines:
        ys = [float(V[s]) for s in STATES]
        p.line(list(STATES), ys, "param", 1.6 if dashed else 2.2, dashed=dashed)
        p.points(list(STATES), ys, "param", r=3.2 if dashed else 3.6, hollow=dashed)
    return p


def _clip(V) -> list:
    """V for a chart: every estimate between −0.05 and 1.05, so that a wild one stays at the edge of the chart."""
    return [min(1.05, max(-0.05, float(v))) for v in V]


_WORDS = {
    "en": {"state": "state", "value": "estimate of V", "walks": "walks", "error": "average distance"},
    "nl": {"state": "state", "value": "schatting van V", "walks": "wandelingen", "error": "gemiddelde afstand"},
}


def _error_panel(curves: dict, n: int, lang: str, title: str) -> chart.Panel:
    """The error of TD(0), solid, and of Monte Carlo, dashed, after every number of walks, with a line at n."""
    w = _WORDS[lang]
    p = chart.Panel((0, N_WALKS), (0, 0.25), title=title, xlabel=w["walks"], ylabel=w["error"],
                    yticks=[0, 0.05, 0.1, 0.15, 0.2, 0.25])
    xs = list(range(N_WALKS + 1))
    p.line(xs, [min(0.25, curves["mc"][k]) for k in xs], "loss", 1.6, dashed=True)
    p.line(xs, [min(0.25, curves["td"][k]) for k in xs], "loss", 2.2)
    p.line([n, n], [0, 0.25], "axis", 1.2)
    p.points([n, n], [min(0.25, curves["td"][n]), min(0.25, curves["mc"][n])], "loss", r=3.6)
    return p


_PLAY_TEXT = {
    "en": {"left": "Run 1 after {n} {word}", "right": "Average of 100 runs", "one": "walk", "many": "walks",
           "key_td": "TD(0)", "key_mc": "Monte Carlo", "key_true": "true values",
           "alt": "Two charts. Left: the estimates of A to E in run 1 after {n} {word}, of TD(0) solid and of Monte Carlo dashed, against the true values from 1/6 to 5/6 as a gray dashed line. Right: the average distance to the true values over 100 runs, after 0 to 100 walks, TD(0) solid and Monte Carlo dashed, with a line at {n}.",
           "say0": "Before the first walk: every estimate is 0.5, on average {e} from the true values.",
           "say": "After {n} {word}: in run 1, TD(0) is on average {t1} from the true values and Monte Carlo {m1}; over all 100 runs, TD(0) is at {t} and Monte Carlo at {m}.",
           "note": "Left, in orange, what the two methods learned in run 1: TD(0) solid with dots, Monte Carlo dashed with open dots; the gray dashed line shows the true values, 1/6 to 5/6. Right, the average distance between the estimates and the true values over the 100 runs, TD(0) solid and Monte Carlo dashed. Both methods learn with η = 0.1, from the same walks."},
    "nl": {"left": "Run 1 na {n} {word}", "right": "Gemiddelde van 100 runs", "one": "wandeling", "many": "wandelingen",
           "key_td": "TD(0)", "key_mc": "Monte Carlo", "key_true": "echte waarden",
           "alt": "Twee grafieken. Links: de schattingen van A tot en met E in run 1 na {n} {word}, van TD(0) doorgetrokken en van Monte Carlo gestreept, afgezet tegen de echte waarden van 1/6 tot 5/6 als grijze gestreepte lijn. Rechts: de gemiddelde afstand tot de echte waarden over 100 runs, na 0 tot 100 wandelingen, TD(0) doorgetrokken en Monte Carlo gestreept, met een lijn bij {n}.",
           "say0": "Bij de start: elke schatting is 0,5, gemiddeld op {e} van de echte waarden.",
           "say": "Na {n} {word}: in run 1 zit TD(0) gemiddeld op {t1} van de echte waarden en Monte Carlo op {m1}; over alle 100 runs zit TD(0) op {t} en Monte Carlo op {m}.",
           "note": "Links, in oranje, wat de twee methoden leerden in run 1: TD(0) doorgetrokken met stippen, Monte Carlo gestreept met open stippen; de grijze gestreepte lijn toont de echte waarden, 1/6 tot 5/6. Rechts de gemiddelde afstand tussen de schattingen en de echte waarden over de 100 runs, TD(0) doorgetrokken en Monte Carlo gestreept. Beide methoden leren met η = 0,1, van dezelfde wandelingen."},
}


def _key(lang: str) -> str:
    w = _PLAY_TEXT[lang]
    return live.key_html([("param", False, w["key_td"]), ("param", True, w["key_mc"]), ("ink-3", True, w["key_true"])])


def play_frame(n: int, lang: str) -> tuple:
    """The play block after n walks: the estimates of run 1 beside the average error of the 100 runs, and a sentence."""
    lang = lang if lang in LANGS else "en"
    w = _PLAY_TEXT[lang]
    r = play_run()
    n = min(LOOKS, key=lambda k: abs(k - int(n)))
    word = w["one"] if n == 1 else w["many"]
    left = _values_panel([(_clip(r["first_td"][n]), False), (_clip(r["first_mc"][n]), True)],
                         w["left"].format(n=n, word=word), lang)
    right = _error_panel(r, n, lang, w["right"])
    body = _svg([left, right], w["alt"].format(n=n, word=word), lang) + _key(lang)
    if n == 0:
        say = w["say0"].format(e=number(r["td"][0], 2, lang))
    else:
        say = w["say"].format(n=n, word=word, t1=number(error(r["first_td"][n]), 3, lang),
                              m1=number(error(r["first_mc"][n]), 3, lang), t=number(r["td"][n], 3, lang),
                              m=number(r["mc"][n], 3, lang))
    return body + _note(w["note"]), say


# --- the demo: your td_update and mc_update on 100 runs -----------------------------------------------------------------

DEMO_TEXT = {
    "en": {"title": "Your runs: distance to the true values", "x": "walks", "y": "average distance",
           "alt": "The average distance between the estimates and the true values over 100 runs, after every walk so far: TD(0) solid, Monte Carlo dashed.",
           "eta_td": "η for TD(0)", "eta_mc": "η for Monte Carlo",
           "key_td": "TD(0), η = {a}", "key_mc": "Monte Carlo, η = {a}",
           "now": "Run 1 after {n} {word}", "one": "walk", "many": "walks",
           "values_alt": "The estimates of A to E in run 1 after {n} {word}, of TD(0) solid and of Monte Carlo dashed, against the true values from 1/6 to 5/6 as a gray dashed line.",
           "rewind_title": "Run 1 after {n} {word}",
           "rewind": "After {n} {word}: over the 100 runs, TD(0) is on average {t} from the true values and Monte Carlo {m}; in run 1, {t1} and {m1}."},
    "nl": {"title": "Je runs: afstand tot de echte waarden", "x": "wandelingen", "y": "gemiddelde afstand",
           "alt": "De gemiddelde afstand tussen de schattingen en de echte waarden over 100 runs, na elke wandeling tot nu toe: TD(0) doorgetrokken, Monte Carlo gestreept.",
           "eta_td": "η voor TD(0)", "eta_mc": "η voor Monte Carlo",
           "key_td": "TD(0), η = {a}", "key_mc": "Monte Carlo, η = {a}",
           "now": "Run 1 na {n} {word}", "one": "wandeling", "many": "wandelingen",
           "values_alt": "De schattingen van A tot en met E in run 1 na {n} {word}, van TD(0) doorgetrokken en van Monte Carlo gestreept, afgezet tegen de echte waarden van 1/6 tot 5/6 als grijze gestreepte lijn.",
           "rewind_title": "Run 1 na {n} {word}",
           "rewind": "Na {n} {word}: over de 100 runs zit TD(0) gemiddeld op {t} van de echte waarden en Monte Carlo op {m}; in run 1 op {t1} en {m1}."},
}


def demo_key(eta_td: float, eta_mc: float, lang: str) -> list:
    """What each line of the demo's chart is, as L.live takes it."""
    lang = lang if lang in LANGS else "en"
    w = DEMO_TEXT[lang]
    return [("loss", False, w["key_td"].format(a=eta_text(eta_td, lang))),
            ("loss", True, w["key_mc"].format(a=eta_text(eta_mc, lang)))]


def now_picture(r: dict, n: int, lang: str) -> str:
    """Beside the chart of the demo: the estimates of run 1 as they are now."""
    lang = lang if lang in LANGS else "en"
    w = DEMO_TEXT[lang]
    word = w["one"] if n == 1 else w["many"]
    panel = _values_panel([(_clip(r["first_td"][n]), False), (_clip(r["first_mc"][n]), True)],
                          w["now"].format(n=n, word=word), lang)
    return _svg([panel], w["values_alt"].format(n=n, word=word), lang) + _key(lang)


def rewind_frame(r: dict, n: int, lang: str) -> tuple:
    """A stop of the rewind after the demo: the chart of the run with a line at n, the estimates of run 1 then, and a
    sentence."""
    lang = lang if lang in LANGS else "en"
    w = DEMO_TEXT[lang]
    n = min(r["first_td"], key=lambda k: abs(k - int(n)))
    word = w["one"] if n == 1 else w["many"]
    curve = _svg([_error_panel(r, n, lang, w["title"])], w["alt"], lang, width=330, height=240)
    curve += live.key_html(demo_key(r["eta_td"], r["eta_mc"], lang))
    panel = _values_panel([(_clip(r["first_td"][n]), False), (_clip(r["first_mc"][n]), True)],
                          w["rewind_title"].format(n=n, word=word), lang)
    say = w["rewind"].format(n=n, word=word, t=number(r["td"][n], 3, lang), m=number(r["mc"][n], 3, lang),
                             t1=number(error(r["first_td"][n]), 3, lang), m1=number(error(r["first_mc"][n]), 3, lang))
    return live.rewound(curve, [_svg([panel], w["values_alt"].format(n=n, word=word), lang) + _key(lang)], width=330), say


# --- the pictures of the learner's work -----------------------------------------------------------------------------------

_STOPPED = object()   # what _call gives for a function that stopped with an error
_SLOW = object()      # what _call gives for a function that ran out of time
PICTURE_SECONDS = 3   # what the learner's function gets for all the calls of one picture together
PICTURE_LOOKS = (10, 100)


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


def _first_run(fn, kind: str) -> dict | None:
    """The 100 walks of run 1 with the learner's function and η = 0.1: V after 10 and after 100 walks."""
    V = start_values()
    out = {}
    for n, w in enumerate(course_walks()[0], start=1):
        if kind == "td":
            for s, r, s_next in w:
                fn(V, s, r, s_next, ETA, GAMMA)
        else:
            fn(V, [tuple(step) for step in w], ETA, GAMMA)
        if n in PICTURE_LOOKS:
            if not _fine(V):
                return None
            out[n] = [float(v) for v in V]
    return out


_PICTURE_TEXT = {
    "en": {"td": {"title": "Your TD(0) in run 1", "who": "`td_update`"},
           "mc": {"title": "Your Monte Carlo in run 1", "who": "`mc_update`"},
           "head": "The 100 walks of run 1 of the play block with your {who} and η = 0.1: the estimates after 10 walks, dashed, and after 100, solid.",
           "key10": "after 10 walks", "key100": "after 100 walks", "true": "true values",
           "say": "After 100 walks your estimates are on average {e} from the true values; after 10 walks, {e10}.",
           "none": "Your {who} stops with an error, takes too long, or leaves `V` without a number for every state, so nothing can be drawn yet.",
           "alt": "The estimates of A to E after 10 walks, dashed, and after 100 walks, solid, against the true values from 1/6 to 5/6 as a gray dashed line.",
           "note": "Orange is what is learned; the gray dashed line shows the true values, 1/6 to 5/6."},
    "nl": {"td": {"title": "Je TD(0) in run 1", "who": "`td_update`"},
           "mc": {"title": "Je Monte Carlo in run 1", "who": "`mc_update`"},
           "head": "De 100 wandelingen van run 1 uit het speelblok met je {who} en η = 0,1: de schattingen na 10 wandelingen, gestreept, en na 100, doorgetrokken.",
           "key10": "na 10 wandelingen", "key100": "na 100 wandelingen", "true": "echte waarden",
           "say": "Na 100 wandelingen zitten je schattingen gemiddeld op {e} van de echte waarden; na 10 wandelingen {e10}.",
           "none": "Je {who} stopt met een fout, duurt te lang, of laat `V` achter zonder getal voor elke state, dus er is nog geen afbeelding.",
           "alt": "De schattingen van A tot en met E na 10 wandelingen, gestreept, en na 100 wandelingen, doorgetrokken, afgezet tegen de echte waarden van 1/6 tot 5/6 als grijze gestreepte lijn.",
           "note": "Oranje is wat geleerd wordt; de grijze gestreepte lijn toont de echte waarden, 1/6 tot 5/6."},
}


def _picture(seen, kind: str) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _PICTURE_TEXT[lang]
    title, who = w[kind]["title"], w[kind]["who"]
    fn = _fn(seen)
    got = None
    if fn is not None:
        got = _call(lambda: _first_run(fn, kind))
        if not isinstance(got, dict):
            got = None
    if got is None:
        return yours.figure(title, _note(w["head"].format(who=who)) + _note(w["none"].format(who=who), _LOSS_UI), alt=title)
    panel = _values_panel([(_clip(got[10]), True), (_clip(got[100]), False)], title, lang)
    body = _note(w["head"].format(who=who)) + _svg([panel], w["alt"], lang)
    body += live.key_html([("param", True, w["key10"]), ("param", False, w["key100"]), ("ink-3", True, w["true"])])
    body += _note(w["say"].format(e=number(error(got[100]), 3, lang), e10=number(error(got[10]), 3, lang)))
    return yours.figure(title, body + _note(w["note"]), alt=title)


def td_picture(seen) -> str:
    return _picture(seen, "td")


def mc_picture(seen) -> str:
    return _picture(seen, "mc")


# --- how an answer is judged --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Judged:
    """What a function did to V over a list of steps or walks, as far as the check is concerned: V rounded to six
    decimals, and `off`, the first state where it differs from the course's answer."""
    kind: str                 # "ok"; "shape": V without a number for every state
    values: tuple = ()
    off: tuple | None = None  # (state, got, want)


def _r6(x) -> float:
    return round(float(x), 6) + 0.0


def _judged(V, wants: list) -> Judged:
    """V against the right answers (one, or two where two orders are right): the closest one decides where it is off."""
    if not _fine(V):
        return Judged("shape")
    values = tuple(_r6(v) for v in V)
    offs = []
    for want in wants:
        off = next(((s, values[s], want[s]) for s in range(7) if abs(values[s] - want[s]) > 1e-6), None)
        if off is None:
            return Judged("ok", values, None)
        offs.append(off)
    return Judged("ok", values, offs[0])


def judge_td(steps, eta, gamma, fn) -> Judged | None:
    """Runs fn as TD(0) does, `td_update(V, s, r, s_next, eta, gamma)` for every step, from V = START, and judges the
    V it leaves; None when V did not change at all."""
    V = start_values()
    for s, r, s_next in steps:
        fn(V, s, r, s_next, eta, gamma)
    if V == start_values():
        return None   # nothing changed: a function that made a new list, or that changes nothing
    want = start_values()
    for s, r, s_next in steps:
        td_update(want, s, r, s_next, eta, gamma)
    return _judged(V, [want])


def judge_mc(walks, eta, gamma, fn) -> Judged | None:
    """Runs fn as Monte Carlo does, `mc_update(V, walk, eta, gamma)` after every walk, from V = START, and judges the
    V it leaves; None when V did not change at all. Where a walk visits a state more than once, the updates from the
    start to the end are right too."""
    V = start_values()
    for w in walks:
        fn(V, [tuple(step) for step in w], eta, gamma)
    if V == start_values():
        return None
    back, forward = start_values(), start_values()
    for w in walks:
        mc_update(back, list(w), eta, gamma)
        _mc_forward(forward, list(w), eta, gamma)
    return _judged(V, [back, forward])


def same_judged(a, b) -> bool:
    """Right when both are right, in either right order; a wrong V must be the same everywhere, within a millionth."""
    if not (isinstance(a, Judged) and isinstance(b, Judged)):
        return a == b
    if a.kind != b.kind:
        return False
    if a.kind != "ok":
        return True
    if (a.off is None) != (b.off is None):
        return False
    return a.off is None or all(abs(x - y) <= 1e-6 for x, y in zip(a.values, b.values))


def _plain(x: float) -> str:
    """A number the way the check shows it: up to four decimals, a real minus sign."""
    text = f"{x:.4f}".rstrip("0").rstrip(".")
    if text in ("-0", "-0.", ""):
        text = "0"
    return text.replace("-", "−")


def show_values(judged) -> dict:
    """What the function left in V, in words: where it first differs from the course's answer."""
    if not isinstance(judged, Judged) or judged.kind == "shape":
        return {"en": "a `V` without a number for every state", "nl": "een `V` zonder getal voor elke state"}
    if judged.off is not None:
        s, got, want = judged.off
        where = {"en": f" (state {NAMES[s - 1]})" if s in STATES else " (an end)",
                 "nl": f" (state {NAMES[s - 1]})" if s in STATES else " (een einde)"}
        return {"en": f"`V[{s}] = {_plain(got)}`{where['en']}, where it should be `{_plain(want)}`",
                "nl": f"`V[{s}] = {_plain(got)}`{where['nl']}, waar het `{_plain(want)}` moet zijn"}
    shown = "`V = [" + ", ".join(_plain(v) for v in judged.values) + "]`"
    return {"en": shown, "nl": shown}


@dataclass
class TDBox(FunctionBox):
    """td_update in a code box: the check runs it over the steps of one or more walks, as TD(0) does after every step,
    and judges the V it leaves, so that a known mistake is known by what it does in every case."""

    def _run(self, fn, case):
        return judge_td(case[0], case[1], case[2], fn)

    def _call_text(self, case) -> str:
        return f"for s, r, s_next in {case[0]!r}: td_update(V, s, r, s_next, {case[1]!r}, {case[2]!r})"


@dataclass
class MCBox(FunctionBox):
    """mc_update in a code box: the check runs it after every walk of a list, and judges the V it leaves."""

    def _run(self, fn, case):
        return judge_mc(case[0], case[1], case[2], fn)

    def _call_text(self, case) -> str:
        return f"for walk in {case[0]!r}: mc_update(V, walk, {case[1]!r}, {case[2]!r})"


# --- the exercises -----------------------------------------------------------------------------------------------------

REQUIRED = ["td_update", "mc_update", "predictor"]


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


# exercise 1: td_update
def _t(line: str = "r + gamma * V[s_next] - V[s]") -> str:
    """td_update as the box holds it, with this line: the TD error."""
    return ("def td_update(V, s, r, s_next, eta, gamma):\n"
            f"    delta = {line}\n"
            "    V[s] = V[s] + eta * delta\n")


_T_START = _t("...")
_T = _t()
_T_NO_MINUS = _t("r + gamma * V[s_next]")
_T_NO_REWARD = _t("gamma * V[s_next] - V[s]")
_T_NO_NEXT = _t("r - V[s]")
_T_SWAPPED = _t("r + gamma * V[s] - V[s_next]")
_T_NO_GAMMA = _t("r + V[s_next] - V[s]")
_T_NEXT_INDEX = _t("r + gamma * V[s + 1] - V[s]")
_T_NOTHING = _t().replace("    delta =", "    V = list(V)\n    delta =")

TD_CASES = [
    (Walk(WALK + WALK, "WALK + WALK"), 0.1, 1.0),
    (LEFT_WALK, 0.1, 1.0),
    (Walk(WALK + LEFT_WALK + WALK, "WALK + LEFT_WALK + WALK"), 0.5, 0.9),
    (LOOP_WALK, 0.2, 1.0),
    (Walk(LOOP_WALK + LEFT_WALK, "LOOP_WALK + LEFT_WALK"), 0.1, 0.9),
]


# exercise 2: mc_update
def _m(line: str = "V[s] + eta * (G - V[s])") -> str:
    """mc_update as the box holds it, with this line: the new V[s]."""
    return ("def mc_update(V, walk, eta, gamma):\n"
            "    G = 0\n"
            "    for s, r, s_next in reversed(walk):\n"
            "        G = r + gamma * G\n"
            f"        V[s] = {line}\n")


_M_START = _m("...")
_M = _m()
_M_RETURN = _m("G")
_M_NO_MINUS = _m("V[s] + eta * G")
_M_REWARD = _m("V[s] + eta * (r - V[s])")
_M_TD = _m("V[s] + eta * (r + gamma * V[s_next] - V[s])")
_M_INVERTED = _m("V[s] + eta * (V[s] - G)")
_M_BRACKETS = _m("V[s] + eta * G - V[s]")
_M_NOTHING = _m().replace("    G = 0\n", "    V = list(V)\n    G = 0\n")

MC_CASES = [
    (Walks([WALK, WALK], "[WALK, WALK]"), 0.1, 1.0),
    (Walks([LEFT_WALK], "[LEFT_WALK]"), 0.1, 1.0),
    (Walks([WALK, LEFT_WALK, WALK], "[WALK, LEFT_WALK, WALK]"), 0.5, 0.9),
    (Walks([LOOP_WALK], "[LOOP_WALK]"), 0.2, 1.0),
    (Walks([LOOP_WALK, LEFT_WALK], "[LOOP_WALK, LEFT_WALK]"), 0.1, 0.9),
]


EXERCISES = {
    "td_update": TDBox(
        name="td_update",
        function="td_update",
        start=_T_START,
        solution=_T,
        compare=same_judged,
        show=show_values,
        visible=TD_CASES[:1],
        hidden=TD_CASES[1:],
        first=[_DOTS],
        picture=td_picture,
        mistakes=[
            Mistake("no-minus", wrong=_T_NO_MINUS,
                    en="`{call}` gives {got}: every step adds a tenth of the target to `V[s]`, so the estimates keep growing. The TD error is the difference between the target and what `V` said before; what do you take away from the target?",
                    nl="`{call}` geeft {got}: elke stap telt een tiende van het target op bij `V[s]`, dus de schattingen blijven groeien. De TD error is het verschil tussen het target en wat `V` eerst zei; wat haal je van het target af?"),
            Mistake("no-reward", wrong=_T_NO_REWARD,
                    en="`{call}` gives {got}: the reward of the step is missing, so the 1 on the right end never reaches `V`, and every estimate sinks toward 0. What does the step itself bring, before the value of the next state?",
                    nl="`{call}` geeft {got}: de reward van de stap ontbreekt, dus de 1 aan het rechtereinde komt nooit in `V`, en elke schatting zakt naar 0. Wat levert de stap zelf op, nog voor de waarde van de volgende state?"),
            Mistake("no-next", wrong=_T_NO_NEXT,
                    en="`{call}` gives {got}: only the reward of the step, so `V[s]` learns what the next step brings and nothing after it. In C the next step always brings 0; which estimate tells you what comes after it?",
                    nl="`{call}` geeft {got}: alleen de reward van de stap, dus `V[s]` leert wat de volgende stap oplevert en niets daarna. In C levert de volgende stap altijd 0 op; welke schatting zegt wat er daarna komt?"),
            Mistake("swapped", wrong=_T_SWAPPED,
                    en="`{call}` gives {got}: `V[s]` and `V[s_next]` have changed places. The target is the reward plus the discounted value of the state you reach; which of the two is that?",
                    nl="`{call}` geeft {got}: `V[s]` en `V[s_next]` hebben van plaats gewisseld. Het target is de reward plus de verdisconteerde waarde van de state waar je terechtkomt; welke van de twee is dat?"),
            Mistake("no-gamma", wrong=_T_NO_GAMMA,
                    en="Right with γ = 1. But `{call}` gives {got}: the value of the next state counts in full, without `gamma`. With γ = 0.9, how much of `V[s_next]` should count?",
                    nl="Goed met γ = 1. Maar `{call}` geeft {got}: de waarde van de volgende state telt volledig mee, zonder `gamma`. Hoeveel van `V[s_next]` moet er met γ = 0,9 meetellen?"),
            Mistake("next-index", wrong=_T_NEXT_INDEX,
                    en="Right for `WALK + WALK`, where every step goes one state to the right, so `V[s + 1]` happens to be the estimate of the state you reach. In `LEFT_WALK` every step goes to the left. Which name holds the state the step leads to?",
                    nl="Goed voor `WALK + WALK`: daar gaat elke stap één state naar rechts, dus `V[s + 1]` is toevallig de schatting van de state waar je terechtkomt. In `LEFT_WALK` gaat elke stap naar links. Welke naam bevat de state waar de stap naartoe gaat?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the TD error: the reward plus the discounted value of the next state, minus the value of this state.",
                      "nl": "Vervang de `...` door de TD error: de reward plus de verdisconteerde waarde van de volgende state, min de waarde van deze state."},
            "arguments": {"en": "The check calls `td_update(V, s, r, s_next, eta, gamma)` for every step, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(V, s, r, s_next, eta, gamma)`?",
                          "nl": "De check roept `td_update(V, s, r, s_next, eta, gamma)` aan voor elke stap, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(V, s, r, s_next, eta, gamma)` houden?"},
            "slow": {"en": "Your function is still not done after 2 seconds. Does it do one step, without a loop of its own?",
                     "nl": "Je functie is na 2 seconden nog niet klaar. Doet je functie één stap, zonder eigen loop?"},
            "wrong": {"en": "`{call}` gives {got}. Is `delta` the reward plus `gamma` times the value of the next state, minus the value of this state?",
                      "nl": "`{call}` geeft {got}. Is `delta` de reward plus `gamma` keer de waarde van de volgende state, min de waarde van deze state?"},
            "almost": {"en": "Right for `WALK + WALK`. But `{call}` gives {got}. Does your function work for every step, also with γ = 0.9 and in a walk that ends on the left?",
                       "nl": "Goed voor `WALK + WALK`. Maar `{call}` geeft {got}. Werkt je functie voor elke stap, ook met γ = 0,9 en in een wandeling die links eindigt?"},
            "no_return": {"en": "With `{call}`, your function changes nothing in `V`. Does it still give `V[s]` its new value, in the list `V` it got?",
                          "nl": "Met `{call}` verandert je functie niets in `V`. Geeft je functie `V[s]` nog een nieuwe waarde, in de lijst `V` die je functie kreeg?"},
            "unreached": {"en": "With `{call}`, your function changes nothing in `V`: it gives a value back with `return`, but the check looks at the list `V`. Does `V[s]` get its new value before the function stops?",
                          "nl": "Met `{call}` verandert je functie niets in `V`: je functie geeft een waarde terug met `return`, maar de check kijkt naar de lijst `V`. Krijgt `V[s]` een nieuwe waarde voordat je functie stopt?"},
            "correct": {"en": "Right in all {n} cases: after the two walks, E is at 0.595 and D at 0.505, and C is still at 0.5. In the picture your `td_update` learns from the 100 walks of run 1; how far from the true values does it end?",
                        "nl": "Goed in alle {n} gevallen: na de twee wandelingen staat E op 0,595 en D op 0,505, en C nog op 0,5. In de afbeelding leert je `td_update` van de 100 wandelingen van run 1; hoe ver van de echte waarden eindigt je functie?"},
        },
    ),
    "mc_update": MCBox(
        name="mc_update",
        function="mc_update",
        start=_M_START,
        solution=_M,
        compare=same_judged,
        show=show_values,
        visible=MC_CASES[:1],
        hidden=MC_CASES[1:],
        first=[_DOTS],
        picture=mc_picture,
        mistakes=[
            Mistake("last-return", wrong=_M_RETURN,
                    en="`{call}` gives {got}: `V[s]` becomes the return of the last walk, and everything it learned before is gone. With η = 0.1, how far should `V[s]` move toward `G`?",
                    nl="`{call}` geeft {got}: `V[s]` wordt de return van de laatste wandeling, en alles wat er eerder geleerd was is weg. Hoe ver moet `V[s]` met η = 0,1 opschuiven naar `G`?"),
            Mistake("no-minus", wrong=_M_NO_MINUS,
                    en="`{call}` gives {got}: every visit adds a part of the return to `V[s]`, so the estimates keep growing. `V[s]` should move toward `G` by a part of the distance; what is the distance between `V[s]` and `G`?",
                    nl="`{call}` geeft {got}: elk bezoek telt een deel van de return op bij `V[s]`, dus de schattingen blijven groeien. `V[s]` moet een deel van de afstand opschuiven naar `G`; wat is de afstand tussen `V[s]` en `G`?"),
            Mistake("reward-only", wrong=_M_REWARD,
                    en="`{call}` gives {got}: `V[s]` moves toward the reward of one step, not toward the return. Which number holds every reward from this step to the end of the walk?",
                    nl="`{call}` geeft {got}: `V[s]` schuift op naar de reward van één stap, niet naar de return. Welk getal bevat elke reward van deze stap tot het einde van de wandeling?"),
            Mistake("td-instead", wrong=_M_TD,
                    en="`{call}` gives {got}: that is the update of TD(0), with the value of the next state as a guess for the rest of the walk. Monte Carlo waits for the end; which number does it know then, and TD(0) not yet?",
                    nl="`{call}` geeft {got}: dat is de update van TD(0), met de waarde van de volgende state als schatting voor de rest van de wandeling. Monte Carlo wacht op het einde; welk getal kent Monte Carlo dan wel, en TD(0) nog niet?"),
            Mistake("inverted", wrong=_M_INVERTED,
                    en="`{call}` gives {got}: `V[s]` moves away from the return instead of toward it. In which order should `G` and `V[s]` stand in the brackets?",
                    nl="`{call}` geeft {got}: `V[s]` schuift weg van de return in plaats van ernaartoe. In welke volgorde moeten `G` en `V[s]` tussen de haakjes staan?"),
            Mistake("brackets", wrong=_M_BRACKETS,
                    en="`{call}` gives {got}: `V[s]` is added and taken away again, so only `eta * G` is left. Python multiplies before it subtracts; where do the brackets go, so that `eta` multiplies the whole distance?",
                    nl="`{call}` geeft {got}: `V[s]` wordt opgeteld en weer afgetrokken, dus alleen `eta * G` blijft over. Python vermenigvuldigt eerst en trekt dan af; waar komen de haakjes, zodat `eta` de hele afstand vermenigvuldigt?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the new `V[s]`: the old one, moved toward the return `G` by `eta` times the distance.",
                      "nl": "Vervang de `...` door de nieuwe `V[s]`: de oude, opgeschoven naar de return `G` met `eta` keer de afstand."},
            "arguments": {"en": "The check calls `mc_update(V, walk, eta, gamma)` after every walk, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(V, walk, eta, gamma)`?",
                          "nl": "De check roept `mc_update(V, walk, eta, gamma)` aan na elke wandeling, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(V, walk, eta, gamma)` houden?"},
            "slow": {"en": "Your function is still not done after 2 seconds. Does the loop go over the steps of the walk, with `for s, r, s_next in reversed(walk):`?",
                     "nl": "Je functie is na 2 seconden nog niet klaar. Gaat de loop over de stappen van de wandeling, met `for s, r, s_next in reversed(walk):`?"},
            "wrong": {"en": "`{call}` gives {got}. Does `V[s]` move toward the return `G` by `eta` times the distance between them?",
                      "nl": "`{call}` geeft {got}. Schuift `V[s]` op naar de return `G`, met `eta` keer de afstand ertussen?"},
            "almost": {"en": "Right for `[WALK, WALK]`. But `{call}` gives {got}. Does your function work for every walk, also one that visits a state more than once, and with γ = 0.9?",
                       "nl": "Goed voor `[WALK, WALK]`. Maar `{call}` geeft {got}. Werkt je functie voor elke wandeling, ook een wandeling die een state vaker bezoekt, en met γ = 0,9?"},
            "no_return": {"en": "With `{call}`, your function changes nothing in `V`. Does it still give `V[s]` its new value, in the list `V` it got?",
                          "nl": "Met `{call}` verandert je functie niets in `V`. Geeft je functie `V[s]` nog een nieuwe waarde, in de lijst `V` die je functie kreeg?"},
            "unreached": {"en": "With `{call}`, your function changes nothing in `V`: it gives a value back with `return`, but the check looks at the list `V`. Does `V[s]` get its new value before the function stops?",
                          "nl": "Met `{call}` verandert je functie niets in `V`: je functie geeft een waarde terug met `return`, maar de check kijkt naar de lijst `V`. Krijgt `V[s]` een nieuwe waarde voordat je functie stopt?"},
            "correct": {"en": "Right in all {n} cases: after the two walks, C, D and E are all at 0.595, also C, whose true value is 0.5. In the picture your `mc_update` learns from the 100 walks of run 1; does it end closer to the true values than TD(0), or further?",
                        "nl": "Goed in alle {n} gevallen: na de twee wandelingen staan C, D en E alle drie op 0,595, ook C, met een echte waarde van 0,5. In de afbeelding leert je `mc_update` van de 100 wandelingen van run 1; eindigt je functie dichter bij de echte waarden dan TD(0), of verder weg?"},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("td_update", _T_START, "empty", None),
    ("td_update", _T, "correct", None),
    ("td_update", _t("r + ..."), "wrong", "dots-left"),
    ("td_update", _T_NO_MINUS, "wrong", "no-minus"),
    ("td_update", _T_NO_REWARD, "wrong", "no-reward"),
    ("td_update", _T_NO_NEXT, "wrong", "no-next"),
    ("td_update", _T_SWAPPED, "wrong", "swapped"),
    ("td_update", _T_NO_GAMMA, "wrong", "no-gamma"),
    # a new list in the function: the one of the check stays as it was
    ("td_update", _T_NEXT_INDEX, "wrong", "next-index"),
    ("td_update", _T_NOTHING, "wrong", "no-return"),
    # the new value given back with return, so V stays as it was
    ("td_update", _t().replace("    V[s] = V[s] + eta * delta\n", "    return V[s] + eta * delta\n"), "wrong", "unreached"),
    # other right answers: the update in one line, and the target first
    ("td_update", _t("r + gamma * V[s_next] - V[s]").replace("    V[s] = V[s] + eta * delta\n", "    V[s] += eta * delta\n"), "correct", None),
    ("td_update", "def td_update(V, s, r, s_next, eta, gamma):\n    target = r + gamma * V[s_next]\n    V[s] = V[s] + eta * (target - V[s])\n", "correct", None),
    # the discount on the reward too: right with γ = 1 only
    ("td_update", _t("gamma * (r + V[s_next]) - V[s]"), "almost", None),
    ("mc_update", _M_START, "empty", None),
    ("mc_update", _M, "correct", None),
    ("mc_update", _m("V[s] + ..."), "wrong", "dots-left"),
    ("mc_update", _M_RETURN, "wrong", "last-return"),
    ("mc_update", _M_NO_MINUS, "wrong", "no-minus"),
    ("mc_update", _M_REWARD, "wrong", "reward-only"),
    ("mc_update", _M_TD, "wrong", "td-instead"),
    ("mc_update", _M_INVERTED, "wrong", "inverted"),
    ("mc_update", _M_BRACKETS, "wrong", "brackets"),
    ("mc_update", _M_NOTHING, "wrong", "no-return"),
    ("mc_update", _m().replace("        V[s] = V[s]", "        return V[s]"), "wrong", "unreached"),
    # another right answer: the returns first, then the updates from the start to the end
    ("mc_update", "def mc_update(V, walk, eta, gamma):\n    G = 0\n    returns = []\n    for s, r, s_next in reversed(walk):\n        G = r + gamma * G\n        returns.append(G)\n    returns.reverse()\n    for t in range(len(walk)):\n        s = walk[t][0]\n        V[s] = V[s] + eta * (returns[t] - V[s])\n", "correct", None),
    # the return without its discount: right with γ = 1 only
    ("mc_update", _m("V[s] + eta * (G - V[s])").replace("G = r + gamma * G", "G = r + G"), "almost", None),
]
