# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rl-many-agents: two Q-learners, A and B, play the repeated prisoner's dilemma, 100 rounds a
game, 300 games. Without memory each agent has one state, and both learn to betray. With memory of the last round
each agent's state is (its own move, the other's move), and most pairs learn to cooperate again, most of them with
the same rule: cooperate after a round in which both did the same, betray after one in which they differed. The game,
the course's Q-learning and its runs, rewards, is_dilemma and together, the functions the learner writes, and the
pictures.

The repeated prisoner's dilemma with Q-learners follows Sandholm and Crites (1996); the game, the code, the runs and the
pictures are the course's own."""
import html
import math
import random
import re

import numpy as np

from everything_ai import chart, live, yours
from everything_ai.check import FunctionBox, Mistake, kept, with_time_limit

LESSON = "rl-many-agents"
LANGS = ("en", "nl")
COOPERATE, BETRAY = "cooperate", "betray"
ACTIONS = [COOPERATE, BETRAY]
# my reward for (my move, the other's move)
PAYOFF = {(COOPERATE, COOPERATE): 3, (COOPERATE, BETRAY): 0, (BETRAY, COOPERATE): 5, (BETRAY, BETRAY): 1}
START = "start"                     # the state of the first round of a game, before anything happened
MEMORIES = [(COOPERATE, COOPERATE), (COOPERATE, BETRAY), (BETRAY, COOPERATE), (BETRAY, BETRAY)]
STATES = [START] + MEMORIES
ROUNDS = 100        # rounds in a game
GAMES = 300
ETA = 0.1
GAMMA = 0.9
EPSILON = 0.05
SEEDS = 20          # the course's runs, with random seeds 0 to 19
LOOKS = [0, 1, 5, 10, 20, 50, 100, 200, 300]   # the stops of the demo
KEEP = (50, 100, 300)               # the Q-tables kept of every run with memory, for exercise 3
WINDOW = 10         # the chart shows the mean of the last 10 games
KINDS = ("memory", "none", "betray_first")   # with memory; without; with memory and "betray" as the first action


class Named(dict):
    """A Q-table or a payoff table that a check message calls by its name; the learner's code gets a plain copy."""

    def __init__(self, entries, name: str):
        super().__init__(entries)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return {k: dict(v) if isinstance(v, dict) else v for k, v in self.items()}


# --- the game, the course's solutions and Q-learning ------------------------------------------------------------------

def greedy(Q, state):
    """The action with the largest Q in `state`, the first one where they tie; a state the Q-table has never seen
    counts as all zeros, so it gives "cooperate"."""
    qs = Q.get(state) if isinstance(Q, dict) else None
    if not isinstance(qs, dict):
        return ACTIONS[0]
    best, top = ACTIONS[0], qs.get(ACTIONS[0], 0.0)
    for a in ACTIONS[1:]:
        if qs.get(a, 0.0) > top:
            best, top = a, qs.get(a, 0.0)
    return best


def rewards(a, b):
    reward_a = PAYOFF[(a, b)]
    reward_b = PAYOFF[(b, a)]
    return reward_a, reward_b


def is_dilemma(payoff):
    both_cooperate = payoff[("cooperate", "cooperate")]
    both_betray = payoff[("betray", "betray")]
    tempted = payoff[("betray", "cooperate")]
    fooled = payoff[("cooperate", "betray")]
    betray_pays = tempted > both_cooperate and both_betray > fooled
    return betray_pays and both_cooperate > both_betray


def together(QA, QB):
    sa, sb = START, START
    count = 0
    for _ in range(ROUNDS):
        a = greedy(QA, sa)
        b = greedy(QB, sb)
        if a == "cooperate" and b == "cooperate":
            count += 1
        sa, sb = (a, b), (b, a)
    return count


def play(QA, QB, memory: bool = True, rounds: int = ROUNDS) -> list:
    """A game of two agents that always take the action with the largest Q: the moves (a, b) of every round."""
    sa = sb = START
    out = []
    for _ in range(rounds):
        a, b = greedy(QA, sa), greedy(QB, sb)
        out.append((a, b))
        sa, sb = ((a, b), (b, a)) if memory else (START, START)
    return out


def both_cooperate(QA, QB, memory: bool = True) -> int:
    return sum(a == COOPERATE and b == COOPERATE for a, b in play(QA, QB, memory))


def train(kind: str = "memory", seed: int = 0, games: int = GAMES, looks=(), keep=()) -> dict:
    """Two independent Q-learners in the repeated prisoner's dilemma: in every round both choose ε-greedy, each gets
    its reward from PAYOFF and moves its own Q toward r + γ times the largest Q of its next state, as in Q-learning. As
    far as the agents know the game always goes on, so the target always has the next state. With memory the next
    state of A is (a, b) and of B (b, a); without it, it stays START. "betray_first" puts "betray" first, for the
    random choice and for a tie. For every game the rounds in which both cooperated and the reward of A and of B; the
    Q-tables after the games in `looks` and `keep`."""
    memory = kind != "none"
    order = ACTIONS if kind != "betray_first" else ACTIONS[::-1]
    rnd = random.Random(seed)
    QA, QB = {}, {}

    def choose(Q, s):
        qs = Q[s]
        if rnd.random() < EPSILON:
            return order[int(rnd.random() * 2)]
        return order[1] if qs[order[1]] > qs[order[0]] else order[0]

    out = {"both": [], "reward_a": [], "reward_b": [], "kept": {}}
    for g in range(games + 1):
        if g in looks or g in keep:
            out["kept"][g] = ({s: dict(q) for s, q in QA.items()}, {s: dict(q) for s, q in QB.items()})
        if g == games:
            break
        sa = sb = START
        n_both = total_a = total_b = 0
        for _ in range(ROUNDS):
            for Q, s in ((QA, sa), (QB, sb)):
                if s not in Q:
                    Q[s] = dict.fromkeys(ACTIONS, 0.0)
            a = choose(QA, sa)
            b = choose(QB, sb)
            ra, rb = PAYOFF[(a, b)], PAYOFF[(b, a)]
            na, nb = ((a, b), (b, a)) if memory else (START, START)
            for Q, s, x, r, n in ((QA, sa, a, ra, na), (QB, sb, b, rb, nb)):
                nq = Q.get(n)
                target = r + GAMMA * (max(nq.values()) if nq else 0.0)
                Q[s][x] += ETA * (target - Q[s][x])
            sa, sb = na, nb
            n_both += a == COOPERATE and b == COOPERATE
            total_a += ra
            total_b += rb
        out["both"].append(n_both)
        out["reward_a"].append(total_a)
        out["reward_b"].append(total_b)
    return out


def smooth(curve: list) -> list:
    """For every game, the mean of the last WINDOW values up to it."""
    out, run = [], 0.0
    for e, v in enumerate(curve):
        run += v
        if e >= WINDOW:
            run -= curve[e - WINDOW]
        out.append(run / min(e + 1, WINDOW))
    return out


def course_runs(kind: str = "memory") -> dict:
    """The 20 runs of 300 games of one kind: the rounds in which both cooperated and the rewards of every game of
    every run, the mean over the runs smoothed, and the Q-tables of every run at the stops of LOOKS and KEEP. Kept per
    kind: the texts, the demo, the NumPy cell and exercise 3 use it."""
    def make():
        runs = [train(kind, seed, GAMES, looks=LOOKS, keep=KEEP) for seed in range(SEEDS)]
        mean = [sum(r["both"][g] for r in runs) / SEEDS for g in range(GAMES)]
        return {"both": [r["both"] for r in runs], "reward_a": [r["reward_a"] for r in runs],
                "reward_b": [r["reward_b"] for r in runs], "smooth": smooth(mean), "kept": [r["kept"] for r in runs]}
    return kept(f"{LESSON}:runs:{kind}", make)


def history(kind: str, what: str = "both") -> np.ndarray:
    """The NumPy cell: the rounds of 100 in which both cooperated ("both"), or the reward of A ("reward_a") or of B
    ("reward_b"), in every game of every run of one kind, as an array of shape (20, 300): a row per run, a column per
    game."""
    return np.array(course_runs(kind)[what], dtype=int)


def course_q(kind: str = "memory", n: int = GAMES, seed: int = 0) -> tuple:
    return course_runs(kind)["kept"][seed][n]


def full_pairs(kind: str, n: int) -> int:
    """How many of the 20 pairs cooperate in every round of a game after n games, with the largest Q."""
    memory = kind != "none"
    if kind == "betray_first":
        return sum(_both_first_betray(*course_q(kind, n, s)) == ROUNDS for s in range(SEEDS))
    return sum(both_cooperate(*course_q(kind, n, s), memory) == ROUNDS for s in range(SEEDS))


def _both_first_betray(QA, QB) -> int:
    """`both_cooperate` for agents that take "betray" where their Q's tie."""
    def pick(Q, s):
        qs = Q.get(s)
        return COOPERATE if qs and qs[COOPERATE] > qs[BETRAY] else BETRAY
    sa = sb = START
    n = 0
    for _ in range(ROUNDS):
        a, b = pick(QA, sa), pick(QB, sb)
        n += a == COOPERATE and b == COOPERATE
        sa, sb = (a, b), (b, a)
    return n


def last_mean(kind: str, n: int) -> float:
    """The mean over the runs of the last WINDOW games up to game n."""
    return course_runs(kind)["smooth"][n - 1]


def rule(Q) -> tuple:
    """What an agent does with the largest Q in each of the five states."""
    return tuple(greedy(Q, s) for s in STATES)


SAME_OR_NOT = (COOPERATE, COOPERATE, BETRAY, BETRAY, COOPERATE)   # cooperate after the same, betray after a difference


# --- the words and the pictures ----------------------------------------------------------------------------------------

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


def _svg(width: int, height: int, parts: list, alt: str) -> str:
    svg = (f'<svg viewBox="0 0 {width} {height}" aria-hidden="true" style="width:100%;max-width:{width}px;height:auto;'
           f'font-family:var(--ea-font-ui)">{"".join(parts)}</svg>')
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(alt)}" style="margin:0.25rem 0 0.5rem;'
            f'max-width:min(100%,{width}px)">{svg}</figure>')


def _t(x, y, words, size=12, fill="var(--ea-ink)", anchor="middle", weight=400, italic=False):
    style = f"font-size:{size}px;fill:{fill};text-anchor:{anchor};font-weight:{weight}" + (";font-style:italic" if italic else "")
    return f'<text x="{x:.1f}" y="{y:.1f}" style="{style}">{html.escape(str(words))}</text>'


def _cell(x, y, w, h, fill="var(--ea-page)"):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h}" style="fill:{fill};stroke:var(--ea-line);stroke-width:1"/>'


_WORDS = {"en": {COOPERATE: "cooperate", BETRAY: "betray"},
          "nl": {COOPERATE: "samenwerken", BETRAY: "verraden"}}


def payoff_svg(table: dict, lang: str, alt: str) -> str:
    """The payoff table as two by two cells: a row per move of A, a column per move of B, and in each cell the reward
    of A and of B. `table` maps (a, b) to (reward of A, reward of B); a value that is not a pair of numbers shows as ?.
    The cell where both cooperate and the cell where both betray are shaded."""
    w = _WORDS[lang]
    left, top, cw, ch = 100, 40, 112, 44
    parts = [_t(left + cw, 14, "B", 13, weight=700),
             _t(14, top + ch + 4, "A", 13, weight=700, anchor="start")]
    for j, b in enumerate(ACTIONS):
        parts.append(_t(left + j * cw + cw / 2, top - 8, w[b], 12, "var(--ea-ink-2)"))
    for i, a in enumerate(ACTIONS):
        parts.append(_t(left - 8, top + i * ch + ch / 2 + 4, w[a], 12, "var(--ea-ink-2)", anchor="end"))
        for j, b in enumerate(ACTIONS):
            x, y = left + j * cw, top + i * ch
            parts.append(_cell(x, y, cw, ch, "var(--ea-inset)" if a == b else "var(--ea-page)"))
            pair = table.get((a, b))
            if isinstance(pair, tuple) and len(pair) == 2 and all(_fine(v) for v in pair):
                words = f"A {_num(pair[0], lang)} · B {_num(pair[1], lang)}"
                parts.append(_t(x + cw / 2, y + ch / 2 + 5, words, 14, weight=600))
            else:
                parts.append(_t(x + cw / 2, y + ch / 2 + 5, "?", 14, "var(--ea-loss-text)", weight=700))
    return _svg(left + 2 * cw + 2, top + 2 * ch + 2, parts, alt)


def _num(v, lang: str) -> str:
    v = float(v)
    return number(v, 0 if v.is_integer() else 2, lang)


def _fine(x) -> bool:
    return isinstance(x, (int, float, np.floating, np.integer)) and not isinstance(x, bool) and math.isfinite(float(x))


_PIC_TEXT = {
    "en": {"game_alt": "The prisoner's dilemma as a table of two by two cells: a row for each move of A, a column for each move of B. Both cooperate: 3 each. A betrays while B cooperates: 5 for A, 0 for B, and the other way around. Both betray: 1 each.",
           "game_key": "A cell gives the reward of A and of B for one round. The two shaded cells are the rounds in which both do the same.",
           "memory": "with memory", "none": "without memory", "start": "start",
           "policy_alt": "A table of what agent A and agent B do with the largest Q in run 1: with memory in each of the five states, without memory in its one state.",
           "own": "own move", "other": "other's move",
           "policy_key": "Own move and other's move: the last round, as the agent saw it. In bold: the two Q's of the state differ, so the agent learned this action. In italics: the two Q's are still equal, so the agent takes the first action."},
    "nl": {"game_alt": "Het prisoner's dilemma als een tabel van twee bij twee vakjes: een rij voor elke zet van A, een kolom voor elke zet van B. Allebei samenwerken: elk 3. A verraadt terwijl B samenwerkt: 5 voor A, 0 voor B, en andersom. Allebei verraden: elk 1.",
           "game_key": "Een vakje geeft de reward van A en van B voor één ronde. De twee gekleurde vakjes zijn de rondes waarin beide hetzelfde doen.",
           "memory": "met geheugen", "none": "zonder geheugen", "start": "start",
           "policy_alt": "Een tabel van wat agent A en agent B in run 1 doen met de grootste Q: met geheugen in elk van de vijf states, zonder geheugen in de enige state.",
           "own": "eigen zet", "other": "zet ander",
           "policy_key": "Eigen zet en zet ander: de laatste ronde, zoals de agent deze zag. Vet: de twee Q's van de state verschillen, dus heeft de agent deze action geleerd. Cursief: de twee Q's zijn nog gelijk, dus neemt de agent de eerste action."},
}


def world_picture(lang: str) -> str:
    """The prisoner's dilemma, beside the guess."""
    lang = lang if lang in LANGS else "en"
    w = _PIC_TEXT[lang]
    return payoff_svg({k: rewards(*k) for k in PAYOFF}, lang, w["game_alt"]) + _note(w["game_key"])


def policy_svg(memory_pair: tuple, none_pair: tuple, lang: str, alt: str) -> str:
    """What A and B do with the largest Q: with memory a row per state, the state as its own move and the other's
    move, and without memory one row. An action is in bold where the two Q's of the state differ, what the agent
    learned, and in italics where they tie."""
    w = _PIC_TEXT[lang]
    words = _WORDS[lang]
    cw, rh, top = 82, 22, 22   # 82 fits the Dutch "samenwerken" in bold at 11 px, and keeps the table at 330 px
    left = 2 * cw
    parts = [_t(cw / 2, 15, w["own"], 11.5, "var(--ea-ink-2)"), _t(cw * 1.5, 15, w["other"], 11.5, "var(--ea-ink-2)"),
             _t(left + cw / 2, 15, "A", 13, weight=700), _t(left + cw * 1.5, 15, "B", 13, weight=700)]
    rows = [(w["memory"], None)] + [(s, ("memory", s)) for s in STATES] + [(w["none"], None), (START, ("none", START))]
    y = top
    for label, what in rows:
        if what is None:
            parts.append(_t(2, y + 15, label, 12, "var(--ea-ink-2)", anchor="start", weight=600))
            y += rh
            continue
        kind, s = what
        if s == START:
            parts.append(_cell(0, y, left, rh, "var(--ea-inset)"))
            parts.append(_t(cw, y + 15, w["start"], 11))
        else:
            for j, move in enumerate(s):
                parts.append(_cell(j * cw, y, cw, rh, "var(--ea-inset)"))
                parts.append(_t(j * cw + cw / 2, y + 15, words[move], 11))
        for j, Q in enumerate(memory_pair if kind == "memory" else none_pair):
            x = left + j * cw
            parts.append(_cell(x, y, cw, rh))
            qs = Q.get(s) if isinstance(Q, dict) else None
            learned = isinstance(qs, dict) and qs.get(COOPERATE, 0.0) != qs.get(BETRAY, 0.0)
            move = greedy(Q, s)
            if learned:
                parts.append(_t(x + cw / 2, y + 15, words[move], 11, "var(--ea-param-text)", weight=700))
            else:
                parts.append(_t(x + cw / 2, y + 15, words[move], 11, "var(--ea-ink-2)", italic=True))
        y += rh
    return _svg(left + 2 * cw + 2, y + 2, parts, alt)


def policy_key(lang: str) -> str:
    return _note(_PIC_TEXT[lang if lang in LANGS else "en"]["policy_key"])


def idea_picture(lang: str) -> str:
    """Run 1 after 300 games, with memory and without: what A and B do with the largest Q."""
    lang = lang if lang in LANGS else "en"
    return policy_svg(course_q("memory"), course_q("none"), lang, _PIC_TEXT[lang]["policy_alt"]) + policy_key(lang)


# --- the demo: the training of the pairs -----------------------------------------------------------------------------

DEMO_TEXT = {
    "en": {"title": "Rounds in which both cooperate", "x": "games", "y": "rounds of 100",
           "memory": "with memory", "none": "without memory",
           "alt": "A chart over 300 games on a log scale, the mean of 20 runs over the last 10 games, with a line at {n}: the rounds of 100 in which both agents cooperate. Both lines start near 90. Without memory the line falls to almost 0 within 80 games; with memory it falls to about 23 near game 80 and climbs back to about 70.",
           "say0": "Before the first game every Q is 0, so with a tie both agents take the first action, `\"cooperate\"`, in every state: they cooperate in every round, but nothing has been learned yet.",
           "say": "After {n} games: over the 20 runs and the last {k} games, both agents cooperated in {m} of the 100 rounds of a game with memory, and in {z} without memory. With the largest Q, {f} of the 20 pairs with memory cooperate in every round of a game, and {g} of the 20 pairs without memory."},
    "nl": {"title": "Rondes waarin beide samenwerken", "x": "games", "y": "rondes van 100",
           "memory": "met geheugen", "none": "zonder geheugen",
           "alt": "Een grafiek over 300 games op een logaritmische schaal, het gemiddelde van 20 runs over de laatste 10 games, met een lijn bij {n}: de rondes van 100 waarin beide agents samenwerken. Beide lijnen beginnen rond 90. Zonder geheugen daalt de lijn binnen 80 games naar bijna 0; met geheugen daalt de lijn rond game 80 naar ongeveer 23 en klimt terug naar ongeveer 70.",
           "say0": "Bij de start is elke Q 0, dus nemen beide agents bij een gelijke stand in elke state de eerste action, `\"cooperate\"`: de agents werken in elke ronde samen, maar er is nog niets geleerd.",
           "say": "Na {n} games: over de 20 runs en de laatste {k} games werkten beide agents met geheugen samen in {m} van de 100 rondes van een game, en zonder geheugen in {z}. Met de grootste Q werken {f} van de 20 paren met geheugen in elke ronde van een game samen, en {g} van de 20 paren zonder geheugen."},
}


def _demo_chart(n: int, lang: str) -> str:
    w = DEMO_TEXT[lang]
    p = chart.Panel((1, GAMES), (0, 100), title=w["title"], xlabel=w["x"], ylabel=w["y"], xlog=True,
                    xticks=[1, 10, 100], yticks=[0, 25, 50, 75, 100])
    if n >= 1:
        p.line([n, n], [0, 100], "axis", 1.2)
    shown = list(range(1, n + 1))
    if shown:
        for kind, width, dashed in (("none", 1.4, True), ("memory", 2.4, False)):
            p.line(shown, [course_runs(kind)["smooth"][x - 1] for x in shown], "ink", width, dashed=dashed)
    out = chart.figure([p], alt=w["alt"].format(n=n), width=330, height=230)
    out = chart.with_decimal_comma(out) if lang == "nl" else out
    return out + live.key_html([("ink", False, w["memory"]), ("ink", True, w["none"])])


def demo_frame(n: int, lang: str) -> tuple:
    """A stop of the demo: the chart with a line at n, what the agents of run 1 do after n games, and a sentence."""
    lang = lang if lang in LANGS else "en"
    w = DEMO_TEXT[lang]
    n = min(LOOKS, key=lambda m: abs(m - int(n)))
    table = policy_svg(course_q("memory", n), course_q("none", n), lang, _PIC_TEXT[lang]["policy_alt"])
    if n == 0:
        say = w["say0"]
    else:
        say = w["say"].format(n=n, k=min(n, WINDOW), m=number(last_mean("memory", n), 1, lang),
                              z=number(last_mean("none", n), 1, lang), f=full_pairs("memory", n), g=full_pairs("none", n))
    return _demo_chart(n, lang) + table + policy_key(lang), say


# --- the pictures of what the learner writes -------------------------------------------------------------------------

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


def _fn_of(seen, name: str):
    run_ = seen.mine
    fn = run_.names.get(name) if run_ is not None else None
    return fn if callable(fn) else None


_YOURS_TEXT = {
    "en": {"r_title": "Your rewards", "r_head": "Your `rewards` for every pair of moves:",
           "r_say": "If B cooperates, A gets {cc} for cooperating and {bc} for betraying. If B betrays, A gets {cb} for cooperating and {bb} for betraying.",
           "r_none": "Your `rewards` stops with an error, takes too long, or gives something that is not a pair of numbers for some moves: those cells show a ?.",
           "r_alt": "The prisoner's dilemma as your rewards give it, as a table of two by two cells.",
           "d_title": "Your verdicts", "d_head": "Your `is_dilemma` for the six games of the check: a row per game, with its four rewards under the names of the code, and your verdict.",
           "d_yes": "dilemma", "d_no": "no dilemma", "d_bad": "?",
           "d_none": "Your `is_dilemma` stops with an error, takes too long, or gives something that is not True or False, so nothing can be drawn yet.",
           "d_alt": "A table of the six games of the check: per game its four rewards and your verdict, a prisoner's dilemma or not.",
           "t_title": "Your count", "t_head": "Your `together` for the 20 pairs with memory after 300 games: a bar per run, the rounds of 100 in which both cooperate.",
           "t_say": "{f} of the 20 pairs cooperate in every round, as your function counts it.",
           "t_none": "Your `together` stops with an error, takes too long, or gives something that is not a number, so nothing can be drawn yet.",
           "t_alt": "Bars for the 20 runs: the rounds of 100 in which both agents cooperate, as your together counts them.",
           "t_x": "run", "t_y": "rounds"},
    "nl": {"r_title": "Je rewards", "r_head": "Je `rewards` voor elk paar zetten:",
           "r_say": "Werkt B samen, dan krijgt A {cc} voor samenwerken en {bc} voor verraden. Verraadt B, dan krijgt A {cb} voor samenwerken en {bb} voor verraden.",
           "r_none": "Je `rewards` stopt met een fout, duurt te lang, of geeft voor sommige zetten iets wat geen paar getallen is: in deze vakjes staat een ?.",
           "r_alt": "Het prisoner's dilemma zoals je rewards het geven, als een tabel van twee bij twee vakjes.",
           "d_title": "Je oordelen", "d_head": "Je `is_dilemma` voor de zes spellen van de check: een rij per spel, met de vier rewards onder de namen uit de code, en je oordeel.",
           "d_yes": "dilemma", "d_no": "geen dilemma", "d_bad": "?",
           "d_none": "Je `is_dilemma` stopt met een fout, duurt te lang, of geeft iets wat geen True of False is, dus er is nog geen afbeelding.",
           "d_alt": "Een tabel van de zes spellen van de check: per spel de vier rewards en je oordeel, een prisoner's dilemma of niet.",
           "t_title": "Je telling", "t_head": "Je `together` voor de 20 paren met geheugen na 300 games: een balk per run, de rondes van 100 waarin beide samenwerken.",
           "t_say": "{f} van de 20 paren werken in elke ronde samen, zoals je functie het telt.",
           "t_none": "Je `together` stopt met een fout, duurt te lang, of geeft iets wat geen getal is, dus er is nog geen afbeelding.",
           "t_alt": "Balken voor de 20 runs: de rondes van 100 waarin beide agents samenwerken, zoals je together de rondes telt.",
           "t_x": "run", "t_y": "rondes"},
}


def rewards_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]
    fn = _fn_of(seen, "rewards")
    if fn is None:
        return yours.figure(w["r_title"], _note(w["r_head"]) + _note(w["r_none"], _LOSS_UI), alt=w["r_title"])
    table = {}
    for k in PAYOFF:
        got = _call(lambda k=k: fn(*k), 1.0)
        table[k] = tuple(got) if isinstance(got, (tuple, list)) and len(got) == 2 else None
    body = _note(w["r_head"]) + payoff_svg(table, lang, w["r_alt"])
    fine = all(t is not None and all(_fine(v) for v in t) for t in table.values())
    if fine:
        mine = {k: table[k][0] for k in table}
        body += _note(w["r_say"].format(cc=_num(mine[(COOPERATE, COOPERATE)], lang), bc=_num(mine[(BETRAY, COOPERATE)], lang),
                                        cb=_num(mine[(COOPERATE, BETRAY)], lang), bb=_num(mine[(BETRAY, BETRAY)], lang)))
    else:
        body += _note(w["r_none"], _LOSS_UI)
    return yours.figure(w["r_title"], body, alt=w["r_title"])


def _game(cc, cb, bc, bb, name: str) -> Named:
    return Named({(COOPERATE, COOPERATE): cc, (COOPERATE, BETRAY): cb, (BETRAY, COOPERATE): bc, (BETRAY, BETRAY): bb}, name)


GAMES_OF_CHECK = [Named(PAYOFF, "PAYOFF"),
                  _game(2, -1, 3, 0, "GAME_2"),    # a prisoner's dilemma with other numbers
                  _game(4, 0, 3, 1, "GAME_3"),     # betraying a cooperator pays less
                  _game(3, 1, 5, 0, "GAME_4"),     # betraying a betrayer pays less
                  _game(1, 0, 3, 2, "GAME_5"),     # betraying always pays, and both betraying beats both cooperating
                  _game(5, 3, 4, 1, "GAME_6")]     # cooperating always pays


def dilemma_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]
    fn = _fn_of(seen, "is_dilemma")
    head = _note(w["d_head"])
    if fn is None:
        return yours.figure(w["d_title"], head + _note(w["d_none"], _LOSS_UI), alt=w["d_title"])
    verdicts = []
    for g in GAMES_OF_CHECK:
        got = _call(lambda g=g: fn({k: v for k, v in g.items()}), 1.0)
        verdicts.append(bool(got) if isinstance(got, (bool, np.bool_)) else None)
    if all(v is None for v in verdicts):
        return yours.figure(w["d_title"], head + _note(w["d_none"], _LOSS_UI), alt=w["d_title"])
    columns = [("both_cooperate", (COOPERATE, COOPERATE)), ("both_betray", (BETRAY, BETRAY)),
               ("tempted", (BETRAY, COOPERATE)), ("fooled", (COOPERATE, BETRAY))]
    widths = [70, 104, 90, 68, 62, 100]
    xs = [sum(widths[:k]) for k in range(len(widths) + 1)]
    rh, top = 24, 26
    mono = "font-family:var(--ea-font-code)"
    parts = [f'<text x="{xs[k + 1] + widths[k + 1] / 2:.1f}" y="17" style="font-size:11px;fill:var(--ea-ink-2);'
             f'text-anchor:middle;{mono}">{name}</text>' for k, (name, _) in enumerate(columns)]
    for i, (g, v) in enumerate(zip(GAMES_OF_CHECK, verdicts)):
        y = top + i * rh
        parts.append(f'<text x="{xs[0] + 4:.1f}" y="{y + 16:.1f}" style="font-size:11.5px;fill:var(--ea-ink);'
                     f'font-weight:700;{mono}">{g.name}</text>')
        for k, (_, key) in enumerate(columns):
            parts.append(_cell(xs[k + 1], y, widths[k + 1], rh))
            parts.append(_t(xs[k + 1] + widths[k + 1] / 2, y + 16, _num(g[key], lang), 12.5, weight=600))
        x = xs[5] + widths[5] / 2
        if v is None:
            parts.append(_t(x, y + 16, w["d_bad"], 12, "var(--ea-loss-text)", weight=700))
        else:
            parts.append(_t(x, y + 16, w["d_yes"] if v else w["d_no"], 12,
                            "var(--ea-ink)" if v else "var(--ea-ink-2)", weight=700 if v else 400))
    body = head + _svg(xs[-1], top + len(GAMES_OF_CHECK) * rh + 2, parts, w["d_alt"])
    if any(v is None for v in verdicts):
        body += _note(w["d_none"], _LOSS_UI)
    return yours.figure(w["d_title"], body, alt=w["d_title"])


def together_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]
    fn = _fn_of(seen, "together")
    head = _note(w["t_head"])
    runs = course_runs("memory")["kept"]

    def make():
        out = []
        for seed in range(SEEDS):
            QA, QB = runs[seed][GAMES]
            v = fn({s: dict(q) for s, q in QA.items()}, {s: dict(q) for s, q in QB.items()})
            if not _fine(v):
                raise ValueError
            out.append(float(v))
        return out

    got = _call(make) if fn is not None else None
    if not isinstance(got, list):
        return yours.figure(w["t_title"], head + _note(w["t_none"], _LOSS_UI), alt=w["t_title"])
    top = max(100.0, max(got))
    p = chart.Panel((0.3, SEEDS + 0.7), (0, top * 1.05), xlabel=w["t_x"], ylabel=w["t_y"], xticks=[1, 5, 10, 15, 20],
                    yticks=[0, 50, 100])
    for i, v in enumerate(got, start=1):
        p.rect(i - 0.36, 0, i + 0.36, max(0.0, min(v, top * 1.05)), "data", solid=True)
    out = chart.figure([p], alt=w["t_alt"], width=330, height=200)
    out = chart.with_decimal_comma(out) if lang == "nl" else out
    full = sum(v == ROUNDS for v in got)
    return yours.figure(w["t_title"], head + out + _note(w["t_say"].format(f=full)), alt=w["t_title"])


# --- the exercises ----------------------------------------------------------------------------------------------------

REQUIRED = ["rewards", "is_dilemma", "together", "pricing"]


def _dots_left(got, text) -> bool:
    return re.search(r"^[^#\n]*\.\.\.", text or "", re.MULTILINE) is not None


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Can you replace it with your own code?",
                nl="Er staat nog een `...` in je code. Kun je de `...` vervangen door je eigen code?")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def same_pair(a, b) -> bool:
    try:
        return len(a) == 2 and all(_fine(x) for x in a) and [float(x) for x in a] == [float(x) for x in b]
    except TypeError:
        return False


def same_truth(a, b) -> bool:
    return isinstance(a, (bool, np.bool_)) and bool(a) == bool(b)


def same_number(a, b) -> bool:
    return _fine(a) and _fine(b) and abs(float(a) - float(b)) < 1e-9


def _shown(v) -> dict:
    text_ = repr(v) if not isinstance(v, np.bool_) else repr(bool(v))
    shown = f"`{text_ if len(text_) <= 80 else text_[:77] + '...'}`"
    return {"en": shown, "nl": shown}


# exercise 1: rewards
def _r(back: str = "PAYOFF[(b, a)]") -> str:
    return ("def rewards(a, b):\n"
            "    reward_a = PAYOFF[(a, b)]\n"
            f"    reward_b = {back}\n"
            "    return reward_a, reward_b\n")


_R_START = _r("...")
_R = _r()
_R_SAME = _r("PAYOFF[(a, b)]")
_R_BOTH = _r("PAYOFF[(b, b)]")
R_NAMES = {"PAYOFF": PAYOFF}


# exercise 2: is_dilemma
def _d(pays: str = "tempted > both_cooperate and both_betray > fooled",
       back: str = "betray_pays and both_cooperate > both_betray") -> str:
    return ("def is_dilemma(payoff):\n"
            "    both_cooperate = payoff[(\"cooperate\", \"cooperate\")]\n"
            "    both_betray = payoff[(\"betray\", \"betray\")]\n"
            "    tempted = payoff[(\"betray\", \"cooperate\")]\n"
            "    fooled = payoff[(\"cooperate\", \"betray\")]\n"
            f"    betray_pays = {pays}\n"
            f"    return {back}\n")


_D_START = _d("...", "...")
_D = _d()
_D_TEMPT = _d("tempted > both_cooperate")
_D_FOOLED = _d("both_betray > fooled")
_D_OR = _d("tempted > both_cooperate or both_betray > fooled")
_D_NO_TOGETHER = _d(back="betray_pays")
_D_REVERSED = _d(back="betray_pays and both_betray > both_cooperate")


# exercise 3: together
def _g(test: str = 'a == "cooperate" and b == "cooperate"', nxt: str = "(a, b), (b, a)") -> str:
    return ("def together(QA, QB):\n"
            "    sa, sb = START, START\n"
            "    count = 0\n"
            "    for _ in range(ROUNDS):\n"
            "        a = greedy(QA, sa)\n"
            "        b = greedy(QB, sb)\n"
            f"        if {test}:\n"
            "            count += 1\n"
            f"        sa, sb = {nxt}\n"
            "    return count\n")


_G_START = _g("...", "...")
_G = _g()
_G_SAME_VIEW = _g(nxt="(a, b), (a, b)")
_G_SWAPPED = _g(nxt="(b, a), (a, b)")
_G_NO_MEMORY = _g(nxt="START, START")
_G_EITHER = _g('a == "cooperate" or b == "cooperate"')
_G_A_ONLY = _g('a == "cooperate"')
G_NAMES = {"START": START, "ROUNDS": ROUNDS, "greedy": greedy}


def _q_cases() -> list:
    """The pairs of Q-tables the check gives together: run 1 with memory after 300 games; runs 4, 7 and 12 after 300
    games; run 14 after 100 games and run 16 after 50 games, still learning, where it matters whose move comes first
    in a state."""
    out = []
    for run_, n in ((1, 300), (4, 300), (7, 300), (12, 300), (14, 100), (16, 50)):
        QA, QB = course_q("memory", n, run_ - 1)
        tag = f"{run_}" if n == GAMES else f"{run_}_{n}"
        out.append((Named(QA, f"QA_{tag}"), Named(QB, f"QB_{tag}")))
    return out


Q_CASES = _q_cases()


EXERCISES = {
    "rewards": FunctionBox(
        name="rewards",
        function="rewards",
        start=_R_START,
        solution=_R,
        names=R_NAMES,
        compare=same_pair,
        show=_shown,
        visible=[(BETRAY, COOPERATE)],
        hidden=[(COOPERATE, COOPERATE), (COOPERATE, BETRAY), (BETRAY, BETRAY)],
        first=[_DOTS],
        picture=rewards_picture,
        mistakes=[
            Mistake("same", wrong=_R_SAME,
                    en="`{call}` gives {got}, and it should give {want}: B gets what A gets, so a betrayal pays both of them. `PAYOFF` gives the reward of the player whose move comes first in the pair. Whose move comes first for B?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: B krijgt wat A krijgt, dus levert verraad allebei iets op. `PAYOFF` geeft de reward van de speler met de eerste zet in het paar. Welke zet moet voor B als eerste staan?"),
            Mistake("own-move", wrong=_R_BOTH,
                    en="`{call}` gives {got}, and it should give {want}: your `reward_b` looks only at the move of B, as if A did the same. Which two moves decide what B gets?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je `reward_b` kijkt alleen naar de zet van B, alsof A hetzelfde deed. Welke twee zetten bepalen wat B krijgt?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the reward of B: the same table, seen from the side of B.",
                      "nl": "Vervang de `...` door de reward van B: dezelfde tabel, gezien vanaf de kant van B."},
            "arguments": {"en": "The check calls `{call}`, with two moves, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(a, b)`?",
                          "nl": "De check roept `{call}` aan, met twee zetten, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(a, b)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. In the pair you give `PAYOFF` for B, does the move of B come first?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Staat de zet van B als eerste in het paar dat je voor B aan `PAYOFF` geeft?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your `reward_b` come from `PAYOFF` for every pair of moves, also when both do the same?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en het moet {want} zijn. Komt je `reward_b` voor elk paar zetten uit `PAYOFF`, ook als beide hetzelfde doen?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return reward_a, reward_b`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return reward_a, reward_b`?"},
            "correct": {"en": "Right for all {n} pairs of moves: whatever B does, A gets more by betraying, and so does B. Yet both get more when both cooperate. Can you see it in the table of the picture?",
                        "nl": "Goed voor alle {n} paren zetten: wat B ook doet, A krijgt meer door te verraden, en B net zo. Toch krijgen beide meer als beide samenwerken. Zie je het in de tabel van de afbeelding?"},
        },
    ),
    "is_dilemma": FunctionBox(
        name="is_dilemma",
        function="is_dilemma",
        start=_D_START,
        solution=_D,
        compare=same_truth,
        show=_shown,
        visible=[(GAMES_OF_CHECK[0],)],
        hidden=[(g,) for g in GAMES_OF_CHECK[1:]],
        first=[_DOTS],
        picture=dilemma_picture,
        mistakes=[
            Mistake("tempted-only", wrong=_D_TEMPT,
                    en="`{call}` gives {got}, and it should give {want}: in this game betraying pays more when the other cooperates, but less when the other betrays. Must betraying pay more in one case, or in both?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: in dit spel levert verraden meer op als de ander samenwerkt, maar minder als de ander verraadt. Moet verraden in één geval meer opleveren, of in allebei?"),
            Mistake("fooled-only", wrong=_D_FOOLED,
                    en="`{call}` gives {got}, and it should give {want}: in this game betraying pays more when the other betrays, but less when the other cooperates. Must betraying pay more in one case, or in both?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: in dit spel levert verraden meer op als de ander verraadt, maar minder als de ander samenwerkt. Moet verraden in één geval meer opleveren, of in allebei?"),
            Mistake("either", wrong=_D_OR,
                    en="`{call}` gives {got}, and it should give {want}: with `or`, betraying only has to pay more in one of the two cases. In a prisoner's dilemma, does it pay more whatever the other does?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: met `or` hoeft verraden slechts in één van de twee gevallen meer op te leveren. Levert verraden in een prisoner's dilemma meer op, wat de ander ook doet?"),
            Mistake("no-together", wrong=_D_NO_TOGETHER,
                    en="`{call}` gives {got}, and it should give {want}: here betraying always pays, but both betraying also gives more than both cooperating, so nobody loses by betraying. What else must be true for a dilemma?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: hier loont verraden altijd, maar allebei verraden geeft ook meer dan allebei samenwerken, dus gaat niemand erop achteruit door te verraden. Wat moet er nog meer waar zijn voor een dilemma?"),
            Mistake("reversed", wrong=_D_REVERSED,
                    en="`{call}` gives {got}, and it should give {want}: your last test asks whether both betraying beats both cooperating. In a dilemma, which of the two is better for both?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je laatste test vraagt of allebei verraden beter is dan allebei samenwerken. Welke van de twee is in een dilemma beter voor beide?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: `betray_pays`, whether betraying pays more whatever the other does, and then the whole test.",
                      "nl": "Vervang de twee `...`: `betray_pays`, of verraden meer oplevert wat de ander ook doet, en dan de hele test."},
            "arguments": {"en": "The check calls `{call}`, with one table, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(payoff)`?",
                          "nl": "De check roept `{call}` aan, met één tabel, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(payoff)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does `betray_pays` compare the two cases, the other cooperating and the other betraying, and does the return also ask whether both cooperating beats both betraying?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Vergelijkt `betray_pays` de twee gevallen, de ander werkt samen en de ander verraadt, en vraagt de return ook of allebei samenwerken beter is dan allebei verraden?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does each comparison in `betray_pays` compare betraying with cooperating against the same move of the other?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en het moet {want} zijn. Zet elke vergelijking in `betray_pays` verraden en samenwerken tegenover dezelfde zet van de ander?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it end with `return` and the whole test?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie met `return` en de hele test?"},
            "correct": {"en": "Right for all {n} games: `PAYOFF` and `GAME_2` are prisoner's dilemmas, the other four are not, each for its own reason. In the picture, can you say for each game which part of the test fails?",
                        "nl": "Goed voor alle {n} spellen: `PAYOFF` en `GAME_2` zijn prisoner's dilemma's, de andere vier niet, elk om een eigen reden. Kun je in de afbeelding voor elk spel zeggen welk deel van de test niet klopt?"},
        },
    ),
    "together": FunctionBox(
        name="together",
        function="together",
        start=_G_START,
        solution=_G,
        names=G_NAMES,
        compare=same_number,
        show=_shown,
        visible=Q_CASES[:1],
        hidden=Q_CASES[1:],
        first=[_DOTS],
        picture=together_picture,
        mistakes=[
            Mistake("same-view", wrong=_G_SAME_VIEW,
                    en="`{call}` gives {got}, and it should give {want}: B gets the state `(a, b)`, the round as A saw it, so B remembers a betrayal by A as one of its own. Whose move comes first in the state of B?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: B krijgt de state `(a, b)`, de ronde zoals A deze zag, dus onthoudt B een verraad van A als een eigen verraad. Welke zet staat als eerste in de state van B?"),
            Mistake("no-memory", wrong=_G_NO_MEMORY,
                    en="`{call}` gives {got}, and it should give {want}: both states go back to `START` after every round, so the agents play without memory, and these agents learned with memory. What did the round just played look like for each of them?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: beide states gaan na elke ronde terug naar `START`, dus spelen de agents zonder geheugen, terwijl deze agents met geheugen leerden. Hoe zag de ronde die net gespeeld is eruit voor elk van de twee?"),
            Mistake("either", wrong=_G_EITHER,
                    en="`{call}` gives {got}, and it should give {want}: with `or` you count every round in which at least one agent cooperates. Which rounds should count?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: met `or` tel je elke ronde waarin minstens één agent samenwerkt. Welke rondes moeten meetellen?"),
            Mistake("a-only", wrong=_G_A_ONLY,
                    en="`{call}` gives {got}, and it should give {want}: you count the rounds in which A cooperates, whatever B does. Which rounds should count?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je telt de rondes waarin A samenwerkt, wat B ook doet. Welke rondes moeten meetellen?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: the test for a round in which both cooperate, and the new states of A and B.",
                      "nl": "Vervang de twee `...`: de test voor een ronde waarin beide samenwerken, en de nieuwe states van A en B."},
            "arguments": {"en": "The check calls `{call}`, with two Q-tables, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(QA, QB)`?",
                          "nl": "De check roept `{call}` aan, met twee Q-tables, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(QA, QB)` houden?"},
            "slow": {"en": "Your function is still not done after 2 seconds. Does the loop run `ROUNDS` times?",
                     "nl": "Je functie is na 2 seconden nog niet klaar. Draait de loop `ROUNDS` keer?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your test ask that both `a` and `b` are `\"cooperate\"`, and does each agent get the round with its own move first?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Vraagt je test of `a` en `b` allebei `\"cooperate\"` zijn, en krijgt elke agent de ronde met de eigen zet als eerste?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your test look at both agents, in a pair where the two act differently?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en het moet {want} zijn. Kijkt je test naar beide agents, in een paar waarin de twee verschillend handelen?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return count`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return count`?"},
            "correct": {"en": "Right for all {n} pairs: 100 for the pair of run 1, 1, 98 and 34 for runs 4, 7 and 12, and 50 and 33 for two pairs that are still learning, where it matters whose move comes first. In the picture, your function counts all 20 pairs; how many cooperate in every round?",
                        "nl": "Goed voor alle {n} paren: 100 voor het paar van run 1, 1, 98 en 34 voor runs 4, 7 en 12, en 50 en 33 voor twee paren die nog leren, waar het uitmaakt welke zet als eerste staat. In de afbeelding telt je functie alle 20 paren; hoeveel werken er in elke ronde samen?"},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("rewards", _R_START, "empty", None),
    ("rewards", _R, "correct", None),
    ("rewards", _r("PAYOFF[(b, ...)]"), "wrong", "dots-left"),
    ("rewards", _R_SAME, "wrong", "same"),
    ("rewards", _R_BOTH, "wrong", "own-move"),
    ("rewards", _R.replace("    return reward_a, reward_b\n", ""), "wrong", "no-return"),
    # other right answers: the key without its parentheses, or the table looked up by hand
    ("rewards", _r("PAYOFF[b, a]"), "correct", None),
    ("rewards", _r("{('cooperate', 'cooperate'): 3, ('cooperate', 'betray'): 0, ('betray', 'cooperate'): 5, ('betray', 'betray'): 1}[(b, a)]"), "correct", None),
    # the two rewards add up to 5: right for the example only
    ("rewards", _r("5 - reward_a"), "almost", None),
    ("is_dilemma", _D_START, "empty", None),
    ("is_dilemma", _D, "correct", None),
    ("is_dilemma", _d(back="betray_pays and ..."), "wrong", "dots-left"),
    ("is_dilemma", _D_TEMPT, "wrong", "tempted-only"),
    ("is_dilemma", _D_FOOLED, "wrong", "fooled-only"),
    ("is_dilemma", _D_OR, "wrong", "either"),
    ("is_dilemma", _D_NO_TOGETHER, "wrong", "no-together"),
    ("is_dilemma", _D_REVERSED, "wrong", "reversed"),
    ("is_dilemma", _D.replace("    return betray_pays and both_cooperate > both_betray\n", ""), "wrong", "no-return"),
    # another right answer: the comparisons the other way around
    ("is_dilemma", _d("both_cooperate < tempted and fooled < both_betray", "both_betray < both_cooperate and betray_pays"), "correct", None),
    # betraying beats being fooled: right for the example only
    ("is_dilemma", _d("tempted > fooled"), "almost", None),
    ("together", _G_START, "empty", None),
    ("together", _G, "correct", None),
    ("together", _g(nxt="(a, b), ..."), "wrong", "dots-left"),
    ("together", _G_SAME_VIEW, "wrong", "same-view"),
    # A and B swapped: B gets the round as A saw it, as in the mistake above, and no case tells the two apart
    ("together", _G_SWAPPED, "wrong", "same-view"),
    ("together", _G_NO_MEMORY, "wrong", "no-memory"),
    ("together", _G_EITHER, "wrong", "either"),
    ("together", _G_A_ONLY, "wrong", "a-only"),
    ("together", _G.replace("    return count\n", ""), "wrong", "no-return"),
    # another right answer: a chained comparison
    ("together", _g('a == b == "cooperate"'), "correct", None),
    # the rounds in which B cooperates: right for the example only
    ("together", _g('b == "cooperate"'), "almost", None),
]
