# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-off-switch: the off switch. The small world is the two-state world of Orseau and
Armstrong (2016, Figure 2): in s1, action a gives 1 and goes on to s2, action b gives 0.9 and stays in s1; in s2, a
gives 1 and b gives 0, and both go back to s1; an interruption in s2 forces b. The learner writes sarsa_update, the
SARSA step, and off_switch, what the three choices of the off-switch game (Hadfield-Menell et al., 2017) are worth to
a robot with a belief about the utility of its action. A demo trains Q-learning and SARSA side by side in the small
world, with a chart that grows while they learn and the world with what each has learned beside it."""
import contextlib
import copy
import html
import io
import re
import time

import numpy as np

from everything_ai import chart, live, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

# --- the small world ------------------------------------------------------------------------------------------------

WORLD = {("s1", "a"): ("s2", 1.0), ("s1", "b"): ("s1", 0.9),   # (state, action): (next state, reward), after
         ("s2", "a"): ("s1", 1.0), ("s2", "b"): ("s1", 0.0)}   # Orseau and Armstrong (2016), Figure 2
STATES = ["s1", "s2"]
ACTIONS = ["a", "b"]
FORCED = ("s2", "b")   # where an interruption happens, and the action it forces
GAMMA = 0.5            # the discount of the paper
ETA, EPSILON = 0.1, 0.1            # the course's choices: learning rate and exploration
STEPS, EVERY = 20000, 250          # steps of a run, and how often the demo reports
THETAS = [round(0.1 * k, 1) for k in range(9)]   # the interruption probabilities the demo offers: 0 to 0.8
THETA = 0.5            # the demo's start, and the probability under the check of exercise 1
SEED = 0               # the seed of the runs under the check and in the demo


def q_update(Q, state, action, reward, nxt, nxt_action, eta, gamma):
    """Q-learning's step, in the same form as SARSA's: the target uses the largest Q where it landed, so nxt_action
    plays no part."""
    target = reward + gamma * max(Q[nxt].values())
    return Q[state][action] + eta * (target - Q[state][action])


def sarsa_update(Q, state, action, reward, nxt, nxt_action, eta, gamma):
    """SARSA's step: the target uses the Q of the action the agent takes next, interruptions included."""
    target = reward + gamma * Q[nxt][nxt_action]
    return Q[state][action] + eta * (target - Q[state][action])


def act(Q, state, theta, rng):
    """The action the agent takes in `state`: ε-greedy on its own Q (a when the two are equal), then, in s2 and with
    probability theta, an interruption that forces b."""
    if rng.random() < EPSILON:
        action = ACTIONS[int(rng.integers(0, 2))]
    else:
        action = "a" if Q[state]["a"] >= Q[state]["b"] else "b"
    if state == FORCED[0] and rng.random() < theta:
        action = FORCED[1]
    return action


def best(Q, state: str = "s1") -> str:
    return "a" if Q[state]["a"] >= Q[state]["b"] else "b"


def run(updates: dict, theta: float, seed: int = SEED, steps: int = STEPS, show=None, pause: float = 0.0) -> dict | str:
    """The learners in `updates` (name: update function) side by side in the small world, each with its own random
    numbers from `seed`, for `steps` steps. Every EVERY steps, for each learner: the share of its choices in s1 that
    were a (going on toward s2) in those steps, and a copy of its Q-table; `show(k, now)` hears of report k, with now
    {"tables": each learner's Q-table, "shares": each learner's shares so far}, and the run then waits `pause` seconds,
    so that a person can watch it. Gives back {name: {"shares", "tables", "final"}}, or the name of the first learner whose
    update stops with an error or gives no number."""
    learners = {}
    for name in updates:
        rng = np.random.default_rng(seed)
        Q = {s: {a: 0.0 for a in ACTIONS} for s in STATES}
        learners[name] = {"rng": rng, "Q": Q, "state": "s1", "action": act(Q, "s1", theta, rng), "a": 0, "n": 0,
                          "shares": [], "tables": []}
    for t in range(1, steps + 1):
        for name, update in updates.items():
            me = learners[name]
            Q, state, action = me["Q"], me["state"], me["action"]
            if state == "s1":
                me["n"] += 1
                me["a"] += action == "a"
            nxt, reward = WORLD[(state, action)]
            nxt_action = act(Q, nxt, theta, me["rng"])
            try:
                new = update(copy.deepcopy(Q), state, action, reward, nxt, nxt_action, ETA, GAMMA)
            except Timeout:
                raise
            except Exception:   # noqa: BLE001   the demo says which update stopped
                return name
            if isinstance(new, bool) or not isinstance(new, (int, float)) or not np.isfinite(new):
                return name
            Q[state][action] = float(new)
            me["state"], me["action"] = nxt, nxt_action
        if t % EVERY == 0:
            for me in learners.values():
                me["shares"].append(me["a"] / me["n"] if me["n"] else 0.0)
                me["tables"].append(copy.deepcopy(me["Q"]))
                me["a"] = me["n"] = 0
            if show is not None:
                show(t // EVERY, {"tables": {name: me["tables"][-1] for name, me in learners.items()},
                                  "shares": {name: list(me["shares"]) for name, me in learners.items()}})
                if pause:
                    time.sleep(pause)
    return {name: {"shares": me["shares"], "tables": me["tables"], "final": me["Q"]} for name, me in learners.items()}


WINDOW = 4   # reports in the thick line of the demo's chart: the last 1,000 steps


def means(shares: list, window: int = WINDOW) -> list:
    """The share of a over the last `window` reports, 1,000 steps: the thick line of the chart."""
    return [sum(shares[max(0, i - window + 1): i + 1]) / len(shares[max(0, i - window + 1): i + 1]) for i in range(len(shares))]


def value_go(theta: float, gamma: float = GAMMA) -> float:
    """What always taking a is worth from s1, when an interruption forces b in s2 with probability theta: the
    course's own calculation, a sum of discounted rewards 1, 1 - theta, 1, 1 - theta, ..."""
    return (1 + gamma * (1 - theta)) / (1 - gamma ** 2)


def value_stay(gamma: float = GAMMA) -> float:
    """What always taking b in s1 is worth: 0.9 at every step."""
    return 0.9 / (1 - gamma)


def threshold(gamma: float = GAMMA) -> float:
    """The theta where going on and staying are worth the same: the course's own calculation."""
    return 0.1 + 0.1 / gamma


# --- the off-switch game ---------------------------------------------------------------------------------------------

ROBBIE = [(-39.5 + k, 0.01) for k in range(100)]    # Russell's example: a belief spread evenly from -40 to +60
SHIFTED = [(-59.5 + k, 0.01) for k in range(100)]   # the course's own: robbie shifted 20 down, from -60 to +40


def off_switch(belief):
    """What the robot's three choices are worth with this belief, a list of (utility, probability): acting now (the
    mean utility), switching itself off (0), and waiting for a rational human, who lets it go ahead only when the
    utility is above 0."""
    act_, wait = 0.0, 0.0
    for utility, p in belief:
        act_ = act_ + p * utility
        wait = wait + p * max(utility, 0)
    return act_, 0.0, wait


# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink);vertical-align:middle")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"
_CODE = "font-family:var(--ea-font-code);font-size:0.78rem"


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def _table(rows: list, head: list) -> str:
    """A table with a head row; each row is (its cells, a style of the row), and each cell is (html, style)."""
    top = "".join(f'<th style="{_HEAD}">{h}</th>' for h in head)
    body = "".join(f'<tr style="{style}">' + "".join(f'<td style="{_CELL}{";" + s if s else ""}">{cell}</td>'
                                                    for cell, s in cells_) + "</tr>" for cells_, style in rows)
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
            f'<tr>{top}</tr>{body}</table></div>')


def _code(text: str) -> str:
    return f'<span style="{_CODE}">{html.escape(text)}</span>'


def number(v: float, lang: str, digits: int = 2) -> str:
    text = f"{v:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def _rounded(v):
    if isinstance(v, float):
        return round(v, 6) + 0.0
    if isinstance(v, (tuple, list)):
        return type(v)(_rounded(x) for x in v)
    return v


def show_value(v) -> str:
    """A result as the check shows it: numbers rounded to six decimals, anything else as Python shows it, cut short
    when it is long."""
    text = repr(_rounded(v))
    return text if len(text) <= 200 else text[:197] + "..."


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and bool(np.isfinite(v))


def same_answer(got, want) -> bool:
    """Whether the learner's answer is the course's: a number within a millionth, or three such numbers as a tuple
    or a list."""
    if isinstance(want, tuple):
        return (isinstance(got, (tuple, list)) and len(got) == len(want)
                and all(same_answer(g, w) for g, w in zip(got, want)))
    return _number(got) and abs(got - want) < 1e-6


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _all gives for a call that stopped with an error


def _all(fn, calls: list) -> list:
    """The learner's function on copies of each list of arguments in `calls`, all within one time limit of a check:
    a list of the results, with _STOPPED for each call that stops with an error; the Timeout passes on."""
    def one(args):
        try:
            return fn(*(copy.deepcopy(a) for a in args))
        except Timeout:
            raise
        except Exception:   # noqa: BLE001   the check above says what went wrong
            return _STOPPED

    with contextlib.redirect_stdout(io.StringIO()):
        return with_time_limit(lambda: [one(args) for args in calls])


_FORCED_WORDS = {"en": "b, 0: forced by an interruption", "nl": "b, 0: gedwongen bij een onderbreking"}


def _arrowhead(x: float, y: float, dx: float, dy: float, color: str) -> str:
    """A small filled triangle at (x, y) pointing along (dx, dy)."""
    n = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    pts = [(x, y), (x - 9 * ux + 4.5 * px, y - 9 * uy + 4.5 * py), (x - 9 * ux - 4.5 * px, y - 9 * uy - 4.5 * py)]
    return f'<polygon points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" style="fill:{color}"/>'


def world_svg(lang: str, Q: dict | None = None, alt: str = "", width: int = 400) -> str:
    """The small world as two states with an arrow for every action, its reward beside it; the arrow an interruption
    forces dashed. With a Q-table, each arrow also shows its Q, and the action with the larger Q in s1, the learner's
    choice where it can stay away from s2, is drawn thick in the color of what a model learns. It is drawn at most
    `width` px wide; its words are 15 px in a drawing 420 wide, so they stay 11 px or more down to 310 px."""
    ink, ink2, ink3, param = "var(--ea-ink)", "var(--ea-ink-2)", "var(--ea-ink-3)", "var(--ea-param)"
    pick = best(Q) if Q else None

    def stroke(state: str, action: str) -> tuple:
        if state == "s1" and pick is not None:
            return (param, 3.2) if action == pick else (ink3, 1.4)
        return (ink2, 1.6)

    def q_text(state: str, action: str) -> str:
        return f" · Q {number(Q[state][action], lang)}" if Q else ""

    parts = []
    # s1 -a-> s2, over the top
    c, sw = stroke("s1", "a")
    parts.append(f'<path d="M 132 98 Q 210 40 288 98" style="fill:none;stroke:{c};stroke-width:{sw}"/>')
    parts.append(_arrowhead(288, 98, 78, 58, c))
    parts.append(f'<text x="210" y="56" text-anchor="middle" style="fill:{ink};font-size:15px">a, +1{html.escape(q_text("s1", "a"))}</text>')
    # s1 -b-> s1, a loop on the left
    c, sw = stroke("s1", "b")
    parts.append(f'<path d="M 90 96 C 40 60, 30 160, 88 124" style="fill:none;stroke:{c};stroke-width:{sw}"/>')
    parts.append(_arrowhead(88, 124, 58, -36, c))
    parts.append(f'<text x="6" y="176" text-anchor="start" style="fill:{ink};font-size:15px">b, +{number(0.9, lang, 1)}{html.escape(q_text("s1", "b"))}</text>')
    # s2 -a-> s1, below
    c, sw = stroke("s2", "a")
    parts.append(f'<path d="M 288 122 Q 210 175 132 122" style="fill:none;stroke:{c};stroke-width:{sw}"/>')
    parts.append(_arrowhead(132, 122, -78, -53, c))
    parts.append(f'<text x="210" y="129" text-anchor="middle" style="fill:{ink};font-size:15px">a, +1{html.escape(q_text("s2", "a"))}</text>')
    # s2 -b-> s1, forced by an interruption, lower and dashed
    c, sw = stroke("s2", "b")
    parts.append(f'<path d="M 300 133 Q 210 230 120 133" style="fill:none;stroke:{c};stroke-width:{sw};stroke-dasharray:5 4"/>')
    parts.append(_arrowhead(120, 133, -90, -97, c))
    parts.append(f'<text x="210" y="202" text-anchor="middle" style="fill:{ink};font-size:15px">{html.escape(_FORCED_WORDS[lang])}{html.escape(q_text("s2", "b"))}</text>')
    for x, name in ((110, "s1"), (310, "s2")):
        parts.append(f'<circle cx="{x}" cy="110" r="24" style="fill:var(--ea-page, transparent);stroke:{ink};stroke-width:1.6"/>')
        parts.append(f'<text x="{x}" y="115" text-anchor="middle" style="fill:{ink};font-size:17px;font-weight:600">{name}</text>')
    label = f' role="img" aria-label="{html.escape(alt)}"' if alt else ' aria-hidden="true"'
    return (f'<svg viewBox="0 0 420 212"{label} style="width:100%;max-width:min(100%,{width}px);height:auto;'
            f'font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums">{"".join(parts)}</svg>')


def world_picture(lang: str) -> str:
    """The small world of *The idea*, without a Q-table."""
    alt = {"en": "Two states, s1 and s2. From s1, action a gives 1 and goes to s2; action b gives 0.9 and stays in s1. "
                 "From s2, a gives 1 and b gives 0, and both go back to s1; an interruption in s2 forces b, drawn dashed.",
           "nl": "Twee states, s1 en s2. Vanuit s1 geeft action a 1 en gaat naar s2; action b geeft 0,9 en blijft in s1. "
                 "Vanuit s2 geeft a 1 en b 0, en allebei gaan terug naar s1; een onderbreking in s2 dwingt b af, gestippeld."}[lang]
    title = {"en": "The small world of Orseau and Armstrong", "nl": "De kleine wereld van Orseau en Armstrong"}[lang]
    note = {"en": "Each arrow is an action with its reward. Discount γ = 0.5. After Orseau and Armstrong (2016), Figure 2.",
            "nl": "Elke pijl is een action met de reward erbij. Discount γ = 0,5. Naar Orseau en Armstrong (2016), figuur 2."}[lang]
    return yours.figure(title, world_svg(lang, alt=alt) + _note(note), alt=title)


def _tables_html(tables: dict, words: dict, lang: str, width: int = 400) -> str:
    """The world once for each learner, side by side, with its Q-table on the arrows, each at most `width` px wide."""
    cells = []
    for name, Q in tables.items():
        head = f'<div style="font-family:var(--ea-font-ui);font-size:0.85rem;font-weight:600;color:var(--ea-ink);margin:0.2rem 0">{html.escape(words[name])}</div>'
        said = words["picks"].format(action=best(Q))
        cells.append(f'<div style="flex:1 1 {min(width, 310)}px;min-width:0;max-width:{width}px">{head}{world_svg(lang, Q, alt=words[name] + ". " + said, width=width)}'
                     f'<div style="{_UI}">{html.escape(said)}</div></div>')
    return f'<div style="display:flex;flex-wrap:wrap;gap:0.5rem 1.5rem">{"".join(cells)}</div>'


_SARSA_WORDS = {
    "en": {"title": "Your SARSA in the small world", "q": "Q-learning (the course's)", "s": "SARSA (your sarsa_update)",
           "picks": "In s1 it picks {action}.", "head": ["the call", "the course's", "yours"], "stopped": "error",
           "run_stopped": "Your sarsa_update stops with an error or gives no number during a run, so the world shows only Q-learning.",
           "note": "Both learners took 20,000 steps in the small world, with interruptions in s2 half of the time (θ = 0.5), "
                   "η = 0.1, ε = 0.1 and the same seed; each arrow shows the Q it learned, and in s1 the action with the larger Q, its choice there, is drawn thick. The table is every case of the "
                   "check; a dashed row is one where your function gives something other than the course's."},
    "nl": {"title": "Jouw SARSA in de kleine wereld", "q": "Q-learning (van de course)", "s": "SARSA (jouw sarsa_update)",
           "picks": "In s1 is de keuze {action}.", "head": ["de aanroep", "van de course", "de jouwe"], "stopped": "fout",
           "run_stopped": "Je sarsa_update stopt met een fout of geeft geen getal tijdens een run, dus de wereld laat alleen Q-learning zien.",
           "note": "Ze namen allebei 20.000 stappen in de kleine wereld, met in s2 de helft van de tijd een onderbreking (θ = 0,5), "
                   "η = 0,1, ε = 0,1 en dezelfde seed; elke pijl laat de Q zien die ze leerden, en in s1 is de action met de grootste Q, de keuze daar, dik getekend. De tabel bevat elk geval "
                   "van de check; een gestippelde rij is er een waar jouw functie iets anders geeft dan die van de course."},
}


def sarsa_picture(seen) -> str:
    """Under the check of exercise 1: Q-learning and the learner's SARSA after a run in the small world at THETA,
    each drawn as the world with its Q-table, and every case of the check in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _SARSA_WORDS[seen.lang]
    got = _all(fn, SARSA_CASES)
    rows = []
    for i, case in enumerate(SARSA_CASES):
        want = sarsa_update(*copy.deepcopy(case))
        if got[i] is _STOPPED:
            shown_, off = html.escape(w["stopped"]), True
        else:
            shown_, off = _code(show_value(got[i])), not same_answer(got[i], want)
        rows.append(([(_code(EXERCISES["sarsa"]._call_text(case)), ""), (_code(show_value(want)), ""),
                      (shown_, "font-weight:600")], _OFF if off else ""))
    with contextlib.redirect_stdout(io.StringIO()):
        result = with_time_limit(lambda: run({"q": q_update, "s": fn}, THETA), 10)
    if isinstance(result, str):
        result = run({"q": q_update}, THETA)
        top = _tables_html({"q": result["q"]["final"]}, w, seen.lang) + _note(w["run_stopped"])
    else:
        top = _tables_html({"q": result["q"]["final"], "s": result["s"]["final"]}, w, seen.lang)
    body = top + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


_OFF_WORDS = {
    "en": {"title": "Your off-switch game", "x": "utility U for the human", "y": "probability",
           "off_part": "switched off", "on_part": "goes ahead",
           "names": {"act": "act", "off": "off", "wait": "wait"}, "head": ["the call", "the course's", "yours"],
           "stopped": "error",
           "note": "The block is the first case, robbie: what the robot believes the utility of its action could be, "
                   "100 values from −39.5 to 59.5, each with probability 0.01; gray where a rational human would switch the robot off, and where the human lets it go ahead in the color of data. Under it, the "
                   "course's act, off and wait as dots, yours as rings. The table is every case; a dashed row is one where "
                   "your function gives something other than the course's."},
    "nl": {"title": "Jouw off-switch game", "x": "utiliteit U voor de mens", "y": "kans",
           "off_part": "uitgezet", "on_part": "gaat door",
           "names": {"act": "act", "off": "off", "wait": "wait"}, "head": ["de aanroep", "van de course", "de jouwe"],
           "stopped": "fout",
           "note": "Het blok is het eerste geval, robbie: wat de robot denkt dat de utiliteit van de action kan zijn, "
                   "100 waarden van −39,5 tot en met 59,5, elk met kans 0,01; grijs waar een rationele mens de robot zou uitzetten, en in de kleur van data waar de mens de robot laat doorgaan. Eronder staan "
                   "act, off en wait van de course als stippen, de jouwe als ringen. De tabel bevat elk geval; een "
                   "gestippelde rij is er een waar jouw functie iets anders geeft dan die van de course."},
}


def off_picture(seen) -> str:
    """Under the check of exercise 2: Russell's belief as bars, gray where the human would switch the robot off, with
    the course's three values as dots on the utility axis and the learner's as rings; every case in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _OFF_WORDS[seen.lang]
    got = _all(fn, OFF_CASES)
    rows = []
    for i, case in enumerate(OFF_CASES):
        want = off_switch(*copy.deepcopy(case))
        if got[i] is _STOPPED:
            shown_, off = html.escape(w["stopped"]), True
        else:
            shown_, off = _code(show_value(got[i])), not same_answer(got[i], want)
        rows.append(([(_code(EXERCISES["off_switch"]._call_text(case)), ""), (_code(show_value(want)), ""),
                      (shown_, "font-weight:600")], _OFF if off else ""))
    panel = chart.Panel(x=(-45, 65), y=(-0.0105, 0.015), xlabel=w["x"], ylabel=w["y"],
                        xticks=[-40, -20, 0, 20, 40, 60], yticks=[0, 0.01])
    panel.rect(-40, 0, 0, 0.01, role="ink-3")     # where a rational human switches the robot off
    panel.rect(0, 0, 60, 0.01, role="data")       # where the human lets it go ahead
    panel.text(-20, 0.0128, w["off_part"], role="ink-3")
    panel.text(30, 0.0128, w["on_part"], role="ink-2")
    mine = got[0]   # the learner's three values as rings, under the course's dots
    if isinstance(mine, (tuple, list)) and len(mine) == 3:
        for value in mine:
            if _number(value) and -45 <= value <= 65:
                panel.points([value], [-0.0025], role="ink", r=7, hollow=True)
    for value, key, row in zip(off_switch(ROBBIE), ("act", "off", "wait"), (-0.0088, -0.0062, -0.0062)):
        panel.points([value], [-0.0025], role="ink", r=4)
        panel.text(value, row, w["names"][key], role="ink-2")
    svg = chart.figure([panel], alt=w["title"], width=420, height=240)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


# --- the demo: Q-learning and SARSA side by side --------------------------------------------------------------------

PAUSE = 0.04   # seconds the demo waits after each of its 80 reports, so that the chart grows while you watch

DEMO_WORDS = {
    "en": {"title": "SARSA in the small world", "x": "thousand steps", "y": "share of a in s1",
           "q": "Q-learning (the course's)", "s": "SARSA", "s_yours": "SARSA (your sarsa_update)",
           "picks": "In s1 it picks {action}.",
           "alt": "For SARSA, the share of its choices in s1 that were a, going on toward s2: every 250 steps thin and gray, "
                  "over the last 1,000 steps thick; Q-learning's share over the last 1,000 steps dashed."},
    "nl": {"title": "SARSA in de kleine wereld", "x": "duizend stappen", "y": "aandeel a in s1",
           "q": "Q-learning (van de course)", "s": "SARSA", "s_yours": "SARSA (jouw sarsa_update)",
           "picks": "In s1 is de keuze {action}.",
           "alt": "Voor SARSA het aandeel van de keuzes in s1 dat a was, door naar s2: elke 250 stappen dun en grijs, over de "
                  "laatste 1000 stappen dik; het aandeel van Q-learning over de laatste 1000 stappen gestippeld."},
}


def demo_words(lang: str, yours_: bool) -> dict:
    """The words of the demo; the SARSA of the learner's own sarsa_update is called that."""
    w = dict(DEMO_WORDS[lang])
    if yours_:
        w["s"] = w["s_yours"]
    return w


THOUSANDS = STEPS // 1000   # the x-axis of the chart counts thousands of steps


def at(k: int) -> float:
    """Where report k lies on the chart's x-axis, in thousands of steps."""
    return EVERY * k / 1000


def points(shares: list) -> list:
    """The thick line of a learner's shares as (thousands of steps, share), for the chart."""
    return [(at(i + 1), m) for i, m in enumerate(means(shares))]


def steps_text(n: int, lang: str) -> str:
    """A number of steps as the lesson writes it: 1,000 and 20,000 in English, 1000 and 20.000 in Dutch."""
    if lang == "nl":
        return f"{n:,}".replace(",", ".") if n >= 10000 else str(n)
    return f"{n:,}"


def progress_text(k: int, words: dict, lang: str) -> str:
    step = steps_text(EVERY * k, lang)
    return (f"**{words['title']}** · stap {step} van 20.000" if lang == "nl"
            else f"**{words['title']}** · step {step} of 20,000")


def mine(box_text, correct: bool, name: str):
    """The learner's own function `name` when its exercise is right, else None: the demo then uses the course's."""
    if not correct or not isinstance(box_text, str):
        return None
    from everything_ai.check import run_code
    run_ = run_code(box_text, {})
    fn = run_.names.get(name) if run_.error is None else None
    return fn if callable(fn) else None


def demo(theta: float, sarsa=None, show=None, pause: float = PAUSE) -> dict | str:
    """The demo's run: Q-learning and SARSA side by side for 20,000 steps with interruption probability theta, SARSA
    with `sarsa`, the learner's sarsa_update, or the course's. Gives back the run, or "sarsa" when `sarsa` stops with
    an error or gives no number, or "slow" when the run takes more than a minute."""
    updates = {"q": q_update, "s": sarsa or sarsa_update}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            out = with_time_limit(lambda: run(updates, theta, show=show, pause=pause), 60)
    except Timeout:
        return "slow"
    return "sarsa" if isinstance(out, str) else out


def now_parts(tables: dict, words: dict, lang: str) -> list:
    """What the demo shows beside its chart: the world once for each learner, with what it has learned so far."""
    return [_tables_html(tables, words, lang, width=330)]


def chart_of(result: dict, words: dict, lang: str, upto: int | None = None) -> live.Live:
    """The demo's chart as it grew during the run, up to report `upto` (all by default): SARSA's share of a in s1,
    every 250 steps thin and over the last 1,000 steps thick, and Q-learning's dashed."""
    k = len(result["s"]["shares"]) if upto is None else upto
    drawn = live.Live({"title": words["title"], "x": words["x"], "y": words["y"], "alt": words["alt"]}, THOUSANDS,
                      role="param", lang=lang, low=0.0, high=1.0, window=WINDOW)
    drawn.reference(points(result["q"]["shares"][:k]), "Q-learning", role="param")
    drawn.xs = [at(i + 1) for i in range(k)]
    drawn.ys = list(result["s"]["shares"][:k])
    return drawn


REWIND_STEPS = [EVERY * k for k in range(1, 9)] + [1000 * k for k in range(3, 21)]   # the stops of the rewind:
# every 250 steps at the start, where SARSA changes its mind, then every 1,000


def report_of(step) -> int:
    """The report a stop of the rewind shows: the one made at that step."""
    return int(step) // EVERY


def rewind_frame(result: dict, k: int, words: dict, lang: str) -> str:
    """The rewind at report k: the whole chart with a line at that step, and beside it the world for each learner as
    it was then."""
    tables = {name: result[name]["tables"][k - 1] for name in ("q", "s")}
    return live.rewound(chart_of(result, words, lang).picture(at=at(k)), now_parts(tables, words, lang))


def rewind_said(result: dict, k: int, lang: str) -> str:
    q, s = best(result["q"]["tables"][k - 1]), best(result["s"]["tables"][k - 1])
    step = steps_text(EVERY * k, lang)
    if lang == "nl":
        return f"Na stap {step}: in s1 kiest Q-learning {q}, en SARSA {s}."
    return f"After step {step}: in s1, Q-learning picks {q}, and SARSA {s}."


def demo_said(result: dict, theta: float, lang: str) -> str:
    q, s = best(result["q"]["final"]), best(result["s"]["final"])
    sq, ss = means(result["q"]["shares"])[-1], means(result["s"]["shares"])[-1]
    if lang == "nl":
        return (f"Na 20.000 stappen met θ = {number(theta, lang, 1)}: in s1 kiest Q-learning {q}, en koos daar in de laatste "
                f"1000 stappen in {round(100 * sq)}% van de keuzes a; SARSA kiest {s}, met a in {round(100 * ss)}% van de keuzes.")
    return (f"After 20,000 steps with θ = {number(theta, lang, 1)}: in s1, Q-learning picks {q}, and chose a in "
            f"{round(100 * sq)}% of its choices there over the last 1,000 steps; SARSA picks {s}, with a in {round(100 * ss)}%.")


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["sarsa", "off_switch", "robot", "specs"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# exercise 1: sarsa_update; the first case is a step from s1 with a, where the agent then takes b in s2
Q_EXAMPLE = {"s1": {"a": 1.7, "b": 1.8}, "s2": {"a": 2.0, "b": 0.9}}
SARSA_CASES = [
    (Q_EXAMPLE, "s1", "a", 1.0, "s2", "b", 0.1, 0.5),
    (Q_EXAMPLE, "s2", "a", 1.0, "s1", "a", 0.1, 0.5),
    (Q_EXAMPLE, "s1", "b", 0.9, "s1", "a", 0.5, 0.5),
    (Q_EXAMPLE, "s1", "a", 1.0, "s2", "a", 1.0, 0.9),
    ({"s1": {"a": 0.0, "b": 0.0}, "s2": {"a": 0.0, "b": 0.0}}, "s1", "a", 1.0, "s2", "b", 0.1, 0.5),
    ({"hall": {"go": 0.5, "stay": 0.2}, "room": {"go": 1.0, "stay": 0.0}}, "hall", "go", 0.0, "room", "stay", 0.2, 0.9),
]
_S_DEF = "def sarsa_update(Q, state, action, reward, nxt, nxt_action, eta, gamma):\n"
_S_START = _S_DEF + "    ...\n"
_S = (_S_DEF +
      "    target = reward + gamma * Q[nxt][nxt_action]\n"
      "    return Q[state][action] + eta * (target - Q[state][action])\n")
_S_MAX = _S.replace("gamma * Q[nxt][nxt_action]", "gamma * max(Q[nxt].values())")
_S_SAME = _S.replace("Q[nxt][nxt_action]", "Q[nxt][action]")
_S_NOGAMMA = _S.replace("reward + gamma * Q[nxt][nxt_action]", "reward + Q[nxt][nxt_action]")
_S_NOREWARD = _S.replace("reward + gamma * Q[nxt][nxt_action]", "gamma * Q[nxt][nxt_action]")
_S_TARGET = _S.replace("    return Q[state][action] + eta * (target - Q[state][action])\n", "    return target\n")
_S_SIGN = _S.replace("eta * (target - Q[state][action])", "eta * (Q[state][action] - target)")
_S_NEXT = _S.replace("    return Q[state][action] + eta * (target - Q[state][action])\n",
                     "    return Q[nxt][nxt_action] + eta * (target - Q[nxt][nxt_action])\n")

# exercise 2: off_switch; the first case is Russell's robbie
OFF_CASES = [
    (ROBBIE,),
    ([(-2, 0.3), (1, 0.5), (3, 0.2)],),
    ([(5, 1.0)],),
    ([(-3, 1.0)],),
    ([(-1, 0.5), (1, 0.5)],),
    (SHIFTED,),
]
_O_DEF = "def off_switch(belief):\n"
_O_START = _O_DEF + "    ...\n"
_O = (_O_DEF +
      "    act, wait = 0.0, 0.0\n"
      "    for utility, p in belief:\n"
      "        act = act + p * utility\n"
      "        wait = wait + p * max(utility, 0)\n"
      "    return act, 0.0, wait\n")
_O_OUTSIDE = (_O_DEF +
              "    act = 0.0\n"
              "    for utility, p in belief:\n"
              "        act = act + p * utility\n"
              "    return act, 0.0, max(act, 0)\n")
_O_NOWEIGHT = _O.replace("wait = wait + p * max(utility, 0)", "wait = wait + max(utility, 0)")
_O_GOOD = (_O_DEF +
           "    act, good, chance = 0.0, 0.0, 0.0\n"
           "    for utility, p in belief:\n"
           "        act = act + p * utility\n"
           "        if utility > 0:\n"
           "            good = good + p * utility\n"
           "            chance = chance + p\n"
           "    return act, 0.0, good / chance\n")
_O_BOTH = _O.replace("act = act + p * utility", "act = act + p * max(utility, 0)")
_O_ORDER = _O.replace("    return act, 0.0, wait\n", "    return act, wait, 0.0\n")
_O_MIN = _O.replace("max(utility, 0)", "min(utility, 0)")
_O_FIRST = _O.replace("    return act, 0.0, wait\n", "        return act, 0.0, wait\n")

_NAMED = [(Q_EXAMPLE, "Q"), (ROBBIE, "robbie"), (SHIFTED, "shifted")]   # the data the texts call by name


def _lit(value) -> str:
    """A value the way the lesson writes it in code: Q, robbie and shifted by name, texts in double quotes."""
    for data, name in _NAMED:
        if type(value) is type(data) and value == data:
            return name
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, tuple):
        return "(" + ", ".join(_lit(v) for v in value) + ("," if len(value) == 1 else "") + ")"
    if isinstance(value, list):
        return "[" + ", ".join(_lit(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_lit(k)}: {_lit(v)}" for k, v in value.items()) + "}"
    if isinstance(value, float):
        return repr(_rounded(value))
    return repr(value)


class NamedBox(FunctionBox):
    """A FunctionBox whose messages write a call the way the lesson does: `off_switch(robbie)`."""

    def _call_text(self, case) -> str:
        return f"{self.function}({', '.join(_lit(a) for a in case)})"


EXERCISES = {
    "sarsa": NamedBox(
        name="sarsa",
        picture=sarsa_picture,
        function="sarsa_update",
        start=_S_START,
        solution=_S,
        visible=[SARSA_CASES[0]],
        hidden=SARSA_CASES[1:],
        first=[_DOTS],
        show=show_value,
        compare=same_answer,
        mistakes=[
            Mistake("max", wrong=_S_MAX,
                    en="`{call}` gives {got}, and it should give {want}: your target uses the largest Q in `nxt`. That is the target of Q-learning. SARSA uses the Q of the action it takes next, `nxt_action`, also when an interruption forced it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je target gebruikt de grootste Q in `nxt`. Dat is het target van Q-learning. SARSA gebruikt de Q van de action die daarna komt, `nxt_action`, ook als een onderbreking die action afdwong."),
            Mistake("same", wrong=_S_SAME,
                    en="`{call}` gives {got}, and it should give {want}: in `nxt` you use the Q of `action`, the action of this step. Use the action taken next, `nxt_action`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: in `nxt` gebruik je de Q van `action`, de action van deze stap. Gebruik de action die daarna komt, `nxt_action`."),
            Mistake("nogamma", wrong=_S_NOGAMMA,
                    en="`{call}` gives {got}, and it should give {want}: the Q of the next step counts in full in your target. Multiply it by `gamma` first, the discount.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de Q van de volgende stap telt in je target helemaal mee. Vermenigvuldig die eerst met `gamma`, de discount."),
            Mistake("noreward", wrong=_S_NOREWARD,
                    en="`{call}` gives {got}, and it should give {want}: your target leaves out the reward of this step. The target is `reward` plus the discounted Q of the next step.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je target laat de reward van deze stap weg. Het target is `reward` plus de Q van de volgende stap, met discount."),
            Mistake("target", wrong=_S_TARGET,
                    en="`{call}` gives {got}, and it should give {want}: you give back the target itself. Move the old Q only a step `eta` toward it, as in Q-learning.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je geeft het target zelf terug. Zet de oude Q slechts één stap `eta` in de richting ervan, zoals bij Q-learning."),
            Mistake("sign", wrong=_S_SIGN,
                    en="`{call}` gives {got}, and it should give {want}: your step moves Q away from the target. Take the target minus the old Q, not the other way round.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je stap zet Q van het target af. Neem het target min de oude Q, niet andersom."),
            Mistake("next", wrong=_S_NEXT,
                    en="`{call}` gives {got}, and it should give {want}: you move the Q of the next step. The step is about `Q[state][action]`, the state and action that just gave `reward`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je verschuift de Q van de volgende stap. De stap gaat over `Q[state][action]`, de state en action die net `reward` gaven."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: work out the target from `reward`, `gamma` and the Q of the next state and next action, then move the old Q a step `eta` toward it.",
                      "nl": "Vervang de `...`: reken het target uit met `reward`, `gamma` en de Q van de volgende state en volgende action, en zet de oude Q dan een stap `eta` in de richting ervan."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The target is `reward` plus `gamma` times the Q of the action that actually came next, in the next state; the new Q is the old one plus `eta` times the target minus the old one.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Het target is `reward` plus `gamma` keer de Q van de action die echt als volgende kwam, in de volgende state; de nieuwe Q is de oude plus `eta` keer het target min de oude."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use the `Q`, the states and the numbers it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie wel de `Q`, de states en de getallen uit de aanroep?"},
            "correct": {"en": "Your `sarsa_update` is right in all {n} cases. Under the check, SARSA ran with your step: in s1 it picks b, and stays away from where the interruptions happen.",
                        "nl": "Je `sarsa_update` klopt in alle {n} gevallen. Onder de check liep SARSA met jouw stap: in s1 kiest SARSA b, en blijft weg van waar de onderbrekingen gebeuren."},
        },
    ),
    "off_switch": NamedBox(
        name="off_switch",
        picture=off_picture,
        function="off_switch",
        start=_O_START,
        solution=_O,
        visible=[OFF_CASES[0]],
        hidden=OFF_CASES[1:],
        first=[_DOTS],
        show=show_value,
        compare=same_answer,
        mistakes=[
            Mistake("outside", wrong=_O_OUTSIDE,
                    en="`{call}` gives {got}, and it should give {want}: your wait is the larger of act and 0, as if the human decided once for the mean. The human decides for each utility: add `p` times the larger of `utility` and 0, pair by pair.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je wait is het grootste van act en 0, alsof de mens één keer beslist over het gemiddelde. De mens beslist per utiliteit: tel paar voor paar `p` keer het grootste van `utility` en 0 op."),
            Mistake("noweight", wrong=_O_NOWEIGHT,
                    en="`{call}` gives {got}, and it should give {want}: in wait, each utility counts in full. Weight each utility by its probability: `p` times it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: in wait telt elke utiliteit helemaal mee. Weeg die met de kans: `p` keer."),
            Mistake("good", wrong=_O_GOOD,
                    en="`{call}` gives {got}, and it should give {want}: your wait is the mean of the good utilities alone. In the other cases the human switches the robot off, worth 0, and those cases count too, with their probability.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je wait is het gemiddelde van alleen de goede utiliteiten. In de andere gevallen zet de mens de robot uit, met waarde 0, en die gevallen tellen ook mee, met hun kans."),
            Mistake("both", wrong=_O_BOTH,
                    en="`{call}` gives {got}, and it should give {want}: your act leaves out the bad utilities. When the robot acts at once, nobody can stop it: act is the mean of all utilities, the negative ones too.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je act laat de slechte utiliteiten weg. Als de robot meteen handelt, kan niemand dat nog tegenhouden: act is het gemiddelde van alle utiliteiten, ook de negatieve."),
            Mistake("order", wrong=_O_ORDER,
                    en="`{call}` gives {got}, and it should give {want}: the three values are in another order. Give back act, then off (0), then wait.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de drie waarden staan in een andere volgorde. Geef act terug, dan off (0), dan wait."),
            Mistake("min", wrong=_O_MIN,
                    en="`{call}` gives {got}, and it should give {want}: your wait keeps the bad utilities and drops the good ones. A rational human lets the robot go ahead when the utility is above 0: take the larger of `utility` and 0.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je wait houdt de slechte utiliteiten en laat de goede weg. Een rationele mens laat de robot doorgaan als de utiliteit boven 0 ligt: neem het grootste van `utility` en 0."),
            Mistake("first", wrong=_O_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: your `return` is inside the loop, so the function stops after the first pair. Put it after the loop, as far in as `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `return` staat in de loop, dus de functie stopt na het eerste paar. Zet die na de loop, even ver ingesprongen als `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: add up `act` and `wait` over the pairs of `belief`, and give back act, 0 and wait.",
                      "nl": "Vervang de `...`: tel `act` en `wait` op over de paren van `belief`, en geef act, 0 en wait terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Go through the pairs as `utility` and `p`: `act` adds `p` times the utility, and `wait` adds `p` times what the outcome is worth after the human's choice: 0 when the utility is below 0, and else the utility. Give back act, 0.0 and wait.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Loop door de paren als `utility` en `p`: `act` telt `p` keer de utiliteit op, en `wait` telt `p` keer op wat de uitkomst na de keuze van de mens waard is: 0 als de utiliteit onder 0 ligt, en anders de utiliteit. Geef `act`, `0.0` en `wait` terug."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use the `belief` it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie wel de `belief` uit de aanroep?"},
            "correct": {"en": "Your `off_switch` is right in all {n} cases. For robbie, waiting is worth 18 and acting 10: the robot does better if it lets the human decide.",
                        "nl": "Je `off_switch` klopt in alle {n} gevallen. Voor robbie is wachten 18 waard en handelen 10: de robot doet het beter door de mens te laten beslissen."},
        },
    ),
}

CASES = [
    ("sarsa", _S_START, "empty", None),
    ("sarsa", _S, "correct", None),
    ("sarsa", _S_DEF + "    old = Q[state][action]\n    return old + eta * (reward + gamma * Q[nxt][nxt_action] - old)\n", "correct", None),
    ("sarsa", _S_DEF + "    return (1 - eta) * Q[state][action] + eta * (reward + gamma * Q[nxt][nxt_action])\n", "correct", None),
    ("sarsa", _S_MAX, "wrong", "max"),
    ("sarsa", _S_SAME, "wrong", "same"),
    ("sarsa", _S_NOGAMMA, "wrong", "nogamma"),
    ("sarsa", _S_NOREWARD, "wrong", "noreward"),
    ("sarsa", _S_TARGET, "wrong", "target"),
    ("sarsa", _S_SIGN, "wrong", "sign"),
    ("sarsa", _S_NEXT, "wrong", "next"),
    ("sarsa", _S_DEF + "    return 1.675\n", "almost", None),
    ("off_switch", _O_START, "empty", None),
    ("off_switch", _O, "correct", None),
    ("off_switch", _O_DEF + "    act = sum(p * u for u, p in belief)\n    wait = sum(p * u for u, p in belief if u > 0)\n    return [act, 0, wait]\n", "correct", None),
    ("off_switch", _O_OUTSIDE, "wrong", "outside"),
    ("off_switch", _O_NOWEIGHT, "wrong", "noweight"),
    ("off_switch", _O_GOOD, "wrong", "good"),
    ("off_switch", _O_BOTH, "wrong", "both"),
    ("off_switch", _O_ORDER, "wrong", "order"),
    ("off_switch", _O_MIN, "wrong", "min"),
    ("off_switch", _O_FIRST, "wrong", "first"),
    ("off_switch", _O_DEF + "    return (10.0, 0.0, 18.0)\n", "almost", None),
]
