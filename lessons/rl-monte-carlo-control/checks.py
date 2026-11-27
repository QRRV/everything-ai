# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rl-monte-carlo-control: an agent learns to play blackjack from the hands it plays. Q(s, a) is
the average of the returns after each state and action, and after every hand the policy takes the action with the
larger Q; exploring starts, or an epsilon-soft policy, make sure that both actions keep being tried. The course's
blackjack, its exact values worked out from the probabilities of the cards, its runs of Monte Carlo control, update
and soft, the functions the learner writes, and their pictures; the play block and the demo.

Monte Carlo control as in Sutton and Barto (Reinforcement Learning: An Introduction, 2nd edition, 2018, sections 5.3
and 5.4, example 5.1 for the rules of blackjack). The course plays without naturals, as Gymnasium's Blackjack-v1 does
by default; the code, the runs and the pictures are the course's own."""
import ast
import html
import math
import random
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from everything_ai import chart, draw, live, yours
from everything_ai.check import FunctionBox, Mistake

LANGS = ("en", "nl")
ACTIONS = ("stick", "hit")
CARDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10)   # 1 is an ace; 10, jack, queen and king count as 10
CHANCE = {c: CARDS.count(c) / len(CARDS) for c in range(1, 11)}
TOTALS = tuple(range(12, 22))   # the totals at which the player decides; below 12 another card can never bust
UPS = tuple(range(1, 11))       # the dealer's card that shows, 1 for an ace
STATES = tuple((t, d, a) for a in (False, True) for t in TOTALS for d in UPS)
SEED = 1           # the seed of the play block: Monte Carlo control with exploring starts
SOFT_SEED = 0      # the seed of the demo: Monte Carlo control with an epsilon-soft policy
N_HANDS = 500000   # the hands of the play block and of the demo
EPSILON = 0.1      # the epsilon of the demo


# --- the game -----------------------------------------------------------------------------------------------------------

def card(rng) -> int:
    """One card from a deck that never runs out: every card has the same chance, so a 10 comes 4 times in 13."""
    return CARDS[int(rng.random() * 13)]


def value(total: int, ace: bool) -> int:
    """What a hand counts: its cards with every ace as 1, plus 10 for an ace that can count as 11 without going over 21."""
    return total + 10 if ace and total + 10 <= 21 else total


def dealer_plays(up: int, rng) -> int:
    """The dealer turns the hidden card and takes cards until the hand counts 17 or more: what it counts then."""
    total, ace = up, up == 1
    while True:
        c = card(rng)
        total, ace = total + c, ace or c == 1
        v = value(total, ace)
        if v >= 17:
            return v


def deal(rng) -> tuple:
    """The start of a hand: two cards for the player and the dealer's card that shows. Below 12 the player always takes
    another card, since that can never go over 21: the state where the first decision comes."""
    c1, c2 = card(rng), card(rng)
    total, ace = c1 + c2, c1 == 1 or c2 == 1
    up = card(rng)
    while value(total, ace) < 12:
        c = card(rng)
        total, ace = total + c, ace or c == 1
    return value(total, ace), up, ace and total + 10 <= 21


def outcome(player: int, dealer: int) -> int:
    """The reward at the end of a hand in which the player stuck: +1 for a win, 0 for a draw, −1 for a loss."""
    if dealer > 21 or player > dealer:
        return 1
    return 0 if player == dealer else -1


def play(policy: dict, rng, start=None, first=None, choose=None) -> tuple:
    """One hand: the list of (state, action) of the player's decisions, and the reward at the end. `start` and `first`
    give an exploring start; `choose(state)` picks each action, else the policy does."""
    total, up, usable = start if start is not None else deal(rng)
    hand, a = [], first
    while True:
        s = (total, up, usable)
        if a is None:
            a = choose(s) if choose is not None else policy[s]
        hand.append((s, a))
        if a == "stick":
            break
        total += card(rng)
        if total > 21:
            if not usable:
                return hand, -1
            total, usable = total - 10, False   # the ace now counts as 1
        a = None
    return hand, outcome(total, dealer_plays(up, rng))


# --- the course's functions: the answers of the exercises ---------------------------------------------------------------

def update(Q, count, policy, hand, reward):
    """After one hand: the average of the returns of every (state, action) of the hand, and the action with the larger
    Q in that state; on a tie the policy sticks."""
    for s, a in hand:
        count[(s, a)] = count[(s, a)] + 1
        Q[(s, a)] = Q[(s, a)] + (reward - Q[(s, a)]) / count[(s, a)]
        if Q[(s, "hit")] > Q[(s, "stick")]:
            policy[s] = "hit"
        else:
            policy[s] = "stick"


def soft(policy, s, epsilon):
    """The epsilon-soft policy of epsilon-greedy: every action gets epsilon / 2, and the policy's action the rest."""
    chances = {}
    for a in ["stick", "hit"]:
        if a == policy[s]:
            chances[a] = 1 - epsilon + epsilon / 2
        else:
            chances[a] = epsilon / 2
    return chances


def start_policy() -> dict:
    """The policy the agent starts with: hit below 20, stick on 20 and 21."""
    return {s: "stick" if s[0] >= 20 else "hit" for s in STATES}


def fresh() -> tuple:
    """Q and count at the start: 0 for every state and action."""
    return {(s, a): 0.0 for s in STATES for a in ACTIONS}, {(s, a): 0 for s in STATES for a in ACTIONS}


# --- what the course works out from the probabilities of the cards ------------------------------------------------------

@lru_cache(maxsize=None)
def _dealer_ends(total: int, ace: bool) -> tuple:
    """The chance of each total the dealer ends on, from a hand that counts `total` with every ace as 1."""
    v = value(total, ace)
    if v >= 17:
        return ((v, 1.0),)
    out = {}
    for c, p in CHANCE.items():
        for end, q in _dealer_ends(total + c, ace or c == 1):
            out[end] = out.get(end, 0.0) + p * q
    return tuple(sorted(out.items()))


def stick_value(player: int, up: int) -> float:
    """The return to expect when the player sticks on `player` against the dealer's card `up`."""
    return sum(q * outcome(player, end) for end, q in _dealer_ends(up, up == 1))


def evaluate(policy: dict, epsilon: float = 0.0) -> float:
    """The return to expect per hand from a fresh deal when the player plays `policy`, with the random moves of the
    epsilon-soft policy when epsilon > 0: worked out from the probabilities of the cards, which the agent never sees."""
    @lru_cache(maxsize=None)
    def v(total: int, ace: bool, up: int) -> float:
        val = value(total, ace)
        if val > 21:
            return -1.0
        hit = sum(p * v(total + c, ace or c == 1, up) for c, p in CHANCE.items())
        if val < 12:
            return hit
        mine = policy[(val, up, ace and total + 10 <= 21)]
        share = 1 - epsilon / 2   # the share of the policy's own action
        stick = stick_value(val, up)
        return share * (stick if mine == "stick" else hit) + (1 - share) * (hit if mine == "stick" else stick)

    return sum(p1 * p2 * pu * v(c1 + c2, c1 == 1 or c2 == 1, up)
               for c1, p1 in CHANCE.items() for c2, p2 in CHANCE.items() for up, pu in CHANCE.items())


def best_policy() -> dict:
    """The best policy, worked out backward from the probabilities of the cards; on a tie it sticks."""
    @lru_cache(maxsize=None)
    def b(total: int, ace: bool, up: int) -> tuple:
        val = value(total, ace)
        if val > 21:
            return -1.0, None
        stick = stick_value(val, up)
        hit = sum(p * b(total + c, ace or c == 1, up)[0] for c, p in CHANCE.items())
        return (stick, "stick") if stick >= hit else (hit, "hit")

    return {(t, d, a): b(t - 10 if a else t, a, d)[1] for t, d, a in STATES}


BEST = best_policy()
BEST_VALUE = evaluate(BEST)                  # −0.0466: even the best play loses a little per hand
START_VALUE = evaluate(start_policy())       # −0.354: hit below 20


def differ(policy: dict) -> list:
    """The states where a policy takes another action than the best policy."""
    return [s for s in STATES if policy.get(s) != BEST[s]]


# --- the course's runs --------------------------------------------------------------------------------------------------

class Hands(list):
    """Hands with their rewards, as (hand, reward), shown by name in a check."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return [(list(h), r) for h, r in self]


def _curve_stops(n: int) -> list:
    """Where the chart of a run measures the policy: evenly on a log scale from the first hand to n."""
    return sorted({min(n, round(10 ** (k / 20))) for k in range(0, 1 + math.ceil(20 * math.log10(n)))})


LOOKS = [0, 100, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]   # the stops of the play block


def explore_run(n: int = N_HANDS, seed: int = SEED, update_fn=update, keep: int = 0) -> dict:
    """Monte Carlo control with exploring starts: every hand starts in a random state with a random first action, and
    after every hand `update_fn` updates Q and the policy. The policy at every stop, Q at the end, and the first
    `keep` hands with their rewards."""
    rng = random.Random(seed)
    Q, count = fresh()
    policy = start_policy()
    stops = set(LOOKS) | set(_curve_stops(n))
    run = {"policies": {0: dict(policy)}, "hands": []}
    for i in range(1, n + 1):
        start = (rng.randint(12, 21), rng.randint(1, 10), rng.random() < 0.5)
        hand, reward = play(policy, rng, start, ACTIONS[rng.randint(0, 1)])
        if i <= keep:
            run["hands"].append((tuple(hand), reward))
        update_fn(Q, count, policy, hand, reward)
        if i in stops:
            run["policies"][i] = dict(policy)
    run["Q"] = Q
    return run


@lru_cache(maxsize=1)
def play_run() -> dict:
    """The run of the play block, with what a hand is worth at every stop."""
    run = explore_run(keep=1000)
    run["worth"] = {k: evaluate(p) for k, p in run["policies"].items()}
    return run


def soft_run(update_fn=update, soft_fn=soft, n: int = N_HANDS, seed: int = SOFT_SEED, epsilon: float = EPSILON,
             show=None, limit: float = 120.0) -> dict | str:
    """Monte Carlo control with an epsilon-soft policy, from real deals: `soft_fn` gives the chances of the two actions,
    and after every hand `update_fn` updates Q and the policy. At every stop of the chart, the policy and what a hand
    is worth with it, without and with the random moves; `show(hands, run)` hears every stop. A function that stops
    with an error, or gives back chances that do not fit, ends the run: then its name, or "slow" after `limit`
    seconds."""
    rng = random.Random(seed)
    Q, count = fresh()
    policy = start_policy()
    stops = set(_curve_stops(n)) | set(LOOKS)
    run = {"policies": {0: dict(policy)}, "worth": {0: evaluate(policy)}, "played": {0: evaluate(policy, epsilon)}}
    begin = time.perf_counter()

    def choose(s):
        chances = soft_fn(policy, s, epsilon)
        hit = float(chances["hit"])
        if not (0 <= hit <= 1 and abs(hit + float(chances["stick"]) - 1) < 1e-9):
            raise ValueError("chances")
        return "hit" if rng.random() < hit else "stick"

    for i in range(1, n + 1):
        try:
            hand, reward = play(policy, rng, choose=choose)
        except Exception:   # noqa: BLE001   the demo says which function to look at
            return "soft"
        try:
            update_fn(Q, count, policy, hand, reward)
        except Exception:   # noqa: BLE001
            return "update"
        if any(policy.get(s) not in ACTIONS for s, _a in hand):   # before soft meets it in the next hand
            return "update"
        if i in stops:
            if any(policy.get(s) not in ACTIONS for s in STATES):
                return "update"
            run["policies"][i] = dict(policy)
            run["worth"][i] = evaluate(policy)
            run["played"][i] = evaluate(policy, epsilon)
            if show is not None:
                show(i, run)
            if time.perf_counter() - begin > limit:
                return "slow"
    return run


# --- the worked example and the NumPy cell --------------------------------------------------------------------------------

def worked_example(hands: int = 100000, seed: int = 0) -> dict:
    """The worked example of the lesson, the same draws: stick on 16 against a 10, or hit once and then stick."""
    rng = np.random.default_rng(seed)

    def draw_card():
        return min(int(rng.integers(1, 14)), 10)

    def dealer(up):
        cards = [up, draw_card()]
        while True:
            total = sum(cards)
            if 1 in cards and total + 10 <= 21:
                total = total + 10
            if total >= 17:
                return total
            cards.append(draw_card())

    def result(player, d):
        if player > 21:
            return -1
        return outcome(player, d)

    stick, hit = [], []
    for _ in range(hands):
        stick.append(result(16, dealer(10)))
        hit.append(result(16 + draw_card(), dealer(10)))
    return {"stick": stick, "hit": hit}


STICK_16 = stick_value(16, 10)                                                          # −0.576
HIT_16 = sum(p * (-1 if 16 + c > 21 else stick_value(16 + c, 10)) for c, p in CHANCE.items())   # −0.569


def _q_array(Q: dict) -> np.ndarray:
    """Q as an array of shape (2, 10, 10, 2): usable ace or not, the player's total from 12, the dealer's card from the
    ace, and the action, 0 stick and 1 hit."""
    out = np.zeros((2, 10, 10, 2))
    for ((t, d, a), act), q in Q.items():
        out[int(a), t - 12, d - 1, ACTIONS.index(act)] = q
    return out


def play_q() -> np.ndarray:
    """Q after the 500,000 hands of the play block, as an array of shape (2, 10, 10, 2): usable ace or not, the
    player's total from 12, the dealer's card from the ace, and the action, 0 stick and 1 hit."""
    return _q_array(play_run()["Q"])


def best_array() -> np.ndarray:
    """The best policy as an array of shape (2, 10, 10): 0 stick, 1 hit."""
    out = np.zeros((2, 10, 10), dtype=int)
    for (t, d, a), act in BEST.items():
        out[int(a), t - 12, d - 1] = ACTIONS.index(act)
    return out


# --- drawing ------------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem"
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
    if out.startswith("-") and float(out.replace(",", "")) == 0:
        out = out[1:]
    out = out.replace("-", "−")
    return out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch, and a real minus sign."""
    return _dec(x, lang if lang in LANGS else "en", digits)


def _svg(panels: list, alt: str, lang: str, width: int = 330, height: int = 250) -> str:
    out = chart.figure(panels, alt=alt, width=width, height=height)
    return chart.with_decimal_comma(out) if lang == "nl" else out


_MAP_TEXT = {
    "en": {"none": "No usable ace", "usable": "Usable ace", "x": "dealer's card", "y": "your total"},
    "nl": {"none": "Geen bruikbare aas", "usable": "Bruikbare aas", "x": "kaart van de dealer", "y": "je totaal"},
}
_COLS = ["A", 2, 3, 4, 5, 6, 7, 8, 9, 10]


def maps(policy: dict, lang: str, rings: bool = True) -> str:
    """The policy as two maps, without and with a usable ace: H where it hits, S where it sticks, and a ring where it
    differs from the best policy."""
    w = _MAP_TEXT[lang]
    panels = []
    for ace, title in ((False, w["none"]), (True, w["usable"])):
        on, ring = set(), set()
        for r, t in enumerate(reversed(TOTALS)):
            for c, d in enumerate(UPS):
                s = (t, d, ace)
                if policy.get(s) == "hit":
                    on.add((r, c))
                if rings and policy.get(s) != BEST[s]:
                    ring.add((r, c))
        panels.append({"title": title, "rows": list(reversed(TOTALS)), "cols": _COLS, "on": on, "ring": ring,
                       "letters": ("H", "S"), "xlabel": w["x"], "ylabel": w["y"]})
    off = differ(policy)
    alt = {"en": f"Two maps of the policy, 10 by 10, the player's total 21 down to 12 against the dealer's card from the ace to 10: H where the policy hits, S where it sticks; it differs from the best policy in {len(off)} of 200 states, each with a ring.",
           "nl": f"Twee roosters van de policy, 10 bij 10, het totaal van de speler van 21 tot 12 tegen de kaart van de dealer van de aas tot 10: H waar de policy een kaart neemt, S waar de policy stopt; de policy verschilt in {len(off)} van de 200 states van de beste policy, elk met een ring."}[lang]
    return draw.policy_grids(panels, alt=alt, cell=24).text


_WORTH_TEXT = {
    "en": {"chart": "What a hand is worth", "x": "hands", "y": "return per hand", "best": "best policy",
           "alt": "What a hand is worth with the policy after every number of hands from 1 to 500,000, on a log scale: from {a} it climbs toward the gray line of the best policy, −0.047; a mark at {n} hands, at {v}."},
    "nl": {"chart": "Wat een hand waard is", "x": "handen", "y": "return per hand", "best": "beste policy",
           "alt": "Wat een hand waard is met de policy na elk aantal handen van 1 tot 500.000, op een logschaal: vanaf {a} klimt de lijn naar de grijze lijn van de beste policy, −0,047; een markering bij {n} handen, op {v}."},
}


def _worth_panel(worth: dict, n: int, lang: str, played: dict | None = None) -> chart.Panel:
    w = _WORTH_TEXT[lang]
    p = chart.Panel((1, N_HANDS), (-0.4, 0), title=w["chart"], xlabel=w["x"], ylabel=w["y"], xlog=True,
                    yticks=[-0.4, -0.3, -0.2, -0.1, 0])
    p.line([1, N_HANDS], [BEST_VALUE, BEST_VALUE], "ink-3", 1.4, dashed=True)   # as the live chart of the demo
    xs = [k for k in sorted(worth) if k >= 1]
    p.line(xs, [worth[k] for k in xs], "ink", 2.2)
    if played:
        p.line(xs, [played[k] for k in xs if k in played], "ink", 1.6, dashed=True)
    if n >= 1:
        p.line([n, n], [-0.4, 0], "axis", 1.2)
        p.points([n], [worth[n]], "ink", r=3.8)
    return p


_PLAY_TEXT = {
    "en": {"title": "Exploring starts after {n} {word}", "one": "hand", "many": "hands",
           "say": "After {n} {word}: the policy differs from the best policy in {k} of 200 states, and a hand is worth {v} with it.",
           "say0": "Before the first hand: the agent hits below 20 and sticks on 20 and 21. That differs from the best policy in {k} of 200 states, and a hand is worth {v} with it.",
           "note": "Orange, H: the policy hits there, it takes a card; empty, S: it sticks. A ring: the policy differs there from the best policy, which the course works out from the probabilities of the cards. In the chart, the return to expect per hand when you play the policy; the gray dashed line, −0.047, is the best policy. The course works these numbers out from the probabilities; the agent itself only sees its hands."},
    "nl": {"title": "Exploring starts na {n} {word}", "one": "hand", "many": "handen",
           "say": "Na {n} {word}: de policy verschilt in {k} van de 200 states van de beste policy, en een hand is er {v} waard mee.",
           "say0": "Bij de start: de agent neemt onder 20 een kaart en stopt op 20 en 21. Dat verschilt in {k} van de 200 states van de beste policy, en een hand is er {v} waard mee.",
           "note": "Oranje, H: de policy neemt daar een kaart; leeg, S: de policy stopt. Een ring: de policy verschilt daar van de beste policy, die de course uitrekent uit de kansen van de kaarten. In de grafiek de return die je per hand kunt verwachten als je met de policy speelt; de grijze gestreepte lijn, −0,047, is de beste policy. De course rekent deze getallen uit met de kansen; de agent zelf ziet alleen de eigen handen."},
}


def _count(n: int, lang: str) -> str:
    return _dec(n, lang, 0)


def play_frame(n: int, lang: str) -> tuple:
    """The play block after n hands with exploring starts: the two maps of the policy, the chart of what a hand is worth
    beside them, and a sentence."""
    lang = lang if lang in LANGS else "en"
    w = _PLAY_TEXT[lang]
    run = play_run()
    n = min(LOOKS, key=lambda k: abs(k - int(n)))
    policy, v = run["policies"][n], run["worth"][n]
    count, word = _count(n, lang), w["one"] if n == 1 else w["many"]
    head = f'<div style="font-family:var(--ea-font-ui);font-size:0.9rem;font-weight:600;margin-bottom:0.2rem">{html.escape(w["title"].format(n=count, word=word))}</div>'
    line = _svg([_worth_panel(run["worth"], n, lang)],
                _WORTH_TEXT[lang]["alt"].format(a=number(run["worth"][1], 3, lang), n=count, v=number(v, 3, lang)), lang,
                width=460, height=230)
    body = head + maps(policy, lang) + line
    k = len(differ(policy))
    say = (w["say0"] if n == 0 else w["say"]).format(n=count, word=word, k=k, v=number(v, 3, lang))
    return body + _note(w["note"]), say


# --- the demo: Monte Carlo control with an epsilon-soft policy ------------------------------------------------------------

DEMO_TEXT = {
    "en": {"title": "Your run: what a hand is worth", "x": "hands", "y": "return per hand",
           "alt": "What a hand is worth with the policy of your run so far, after every number of hands on a log scale, solid without the random moves and dashed with them; gray, the best policy.",
           "key_policy": "the policy", "key_played": "as played, with ε = 0.1", "best": "best policy",
           "now": "Your agent after {n} hands",
           "rewind_title": "Your run after {n} hands",
           "rewind": "After {n} hands: the policy differs from the best policy in {k} of 200 states; a hand is worth {v} with it, and {p} as played, with the random moves."},
    "nl": {"title": "Je run: wat een hand waard is", "x": "handen", "y": "return per hand",
           "alt": "Wat een hand waard is met de policy van je run tot nu toe, na elk aantal handen op een logschaal, doorgetrokken zonder de willekeurige zetten en gestreept met; grijs de beste policy.",
           "key_policy": "de policy", "key_played": "zoals gespeeld, met ε = 0,1", "best": "beste policy",
           "now": "Je agent na {n} handen",
           "rewind_title": "Je run na {n} handen",
           "rewind": "Na {n} handen: de policy verschilt in {k} van de 200 states van de beste policy; een hand is er {v} waard mee, en {p} zoals gespeeld, met de willekeurige zetten."},
}


def now_picture(n: int, policy: dict, lang: str) -> str:
    """Beside the chart of the demo: the maps of the policy as it is now."""
    lang = lang if lang in LANGS else "en"
    return (f'<div style="font-family:var(--ea-font-ui);font-size:0.9rem;font-weight:600;margin-bottom:0.2rem">'
            f'{html.escape(DEMO_TEXT[lang]["now"].format(n=_count(n, lang)))}</div>' + maps(policy, lang))


def rewind_frame(run: dict, n: int, lang: str) -> tuple:
    """A stop of the rewind after the demo: the maps of the policy then, the chart with a line there, and a sentence."""
    lang = lang if lang in LANGS else "en"
    w = DEMO_TEXT[lang]
    n = min(run["policies"], key=lambda k: abs(k - int(n)))
    policy = run["policies"][n]
    count = _count(n, lang)
    head = f'<div style="font-family:var(--ea-font-ui);font-size:0.9rem;font-weight:600;margin-bottom:0.2rem">{html.escape(w["rewind_title"].format(n=count))}</div>'
    line = _svg([_worth_panel(run["worth"], n, lang, run["played"])], w["alt"], lang, width=live.CHART_WIDTH, height=240)
    key = live.key_html([("ink", False, w["key_policy"]), ("ink", True, w["key_played"]), ("ink-3", True, w["best"])])
    say = w["rewind"].format(n=count, k=len(differ(policy)), v=number(run["worth"][n], 3, lang),
                             p=number(run["played"][n], 3, lang))
    return live.rewound(line + key, [head + maps(policy, lang)], width=live.CHART_WIDTH), say


@lru_cache(maxsize=1)
def course_soft_run() -> dict:
    """The demo with the course's update and soft: what the text after the demo says."""
    return soft_run()


# --- the pictures of the learner's work -----------------------------------------------------------------------------------

_STOPPED = object()   # what _call gives for a function that stopped with an error
_SLOW = object()      # what _call gives for a function that ran out of time
PICTURE_SECONDS = 4   # what the learner's function gets for all the calls of one picture together
PICTURE_HANDS = 100000


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


_UPDATE_TEXT = {
    "en": {"title": "Your agent after 100,000 hands",
           "head": "Monte Carlo control with exploring starts and your `update`: 100,000 hands, the same hands as the play block as long as your policy takes the same actions.",
           "say": "Your policy differs from the best policy in {k} of 200 states, and a hand is worth {v} with it; the agent started at −0.354.",
           "none": "Your `update` stops with an error, takes too long, or leaves a state without the action `\"stick\"` or `\"hit\"`, so nothing can be drawn yet.",
           "note": "Orange, H: your policy hits there; empty, S: it sticks. A ring: your policy differs there from the best policy, which the course works out from the probabilities of the cards."},
    "nl": {"title": "Je agent na 100.000 handen",
           "head": "Monte Carlo control met exploring starts en je `update`: 100.000 handen, dezelfde handen als in het speelblok zolang je policy dezelfde actions kiest.",
           "say": "Je policy verschilt in {k} van de 200 states van de beste policy, en een hand is er {v} waard mee; de agent begon op −0,354.",
           "none": "Je `update` stopt met een fout, duurt te lang, of laat een state achter zonder de action `\"stick\"` of `\"hit\"`, dus er is nog geen afbeelding.",
           "note": "Oranje, H: je policy neemt daar een kaart; leeg, S: je policy stopt. Een ring: je policy verschilt daar van de beste policy, die de course uitrekent uit de kansen van de kaarten."},
}


def update_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _UPDATE_TEXT[lang]
    fn = _fn(seen)
    got = None
    if fn is not None:
        got = _call(lambda: explore_run(PICTURE_HANDS, SEED, fn)["policies"][PICTURE_HANDS])
        if not isinstance(got, dict) or any(got.get(s) not in ACTIONS for s in STATES):
            got = None
    if got is None:
        return yours.figure(w["title"], _note(w["head"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    body = _note(w["head"]) + maps(got, lang)
    body += _note(w["say"].format(k=len(differ(got)), v=number(evaluate(got), 3, lang)))
    return yours.figure(w["title"], body + _note(w["note"]), alt=w["title"])


SOFT_STATE = (20, 10, False)          # the state of the picture of exercise 2, where START sticks
SOFT_EPSILONS = (0.0, 0.1, 0.5, 1.0)

_SOFT_TEXT = {
    "en": {"title": "Your soft at 20 against a 10",
           "head": "The chances your `soft` gives in the state `(20, 10, False)`, where the policy sticks, for four values of ε.",
           "row": "ε = {e}: {a}",
           "alt": "Bars of the chances of stick and hit in the state (20, 10, False) for ε = 0, 0.1, 0.5 and 1.",
           "none": "Your `soft` stops with an error, takes too long or does not give a number for `\"stick\"` and `\"hit\"` here, so nothing can be drawn yet.",
           "added": "With ε = {e} your chances add up to {t}, not to 1.",
           "note": "With ε = 0 the policy always sticks; with ε = 1 every move is random, half stick and half hit."},
    "nl": {"title": "Je soft op 20 tegen een 10",
           "head": "De kansen die je `soft` geeft in de state `(20, 10, False)`, waar de policy stopt, bij vier waarden van ε.",
           "row": "ε = {e}: {a}",
           "alt": "Balken met de kansen op stick en hit in de state (20, 10, False) bij ε = 0, 0,1, 0,5 en 1.",
           "none": "Je `soft` stopt met een fout, duurt te lang of geeft hier geen getal voor `\"stick\"` en `\"hit\"`, dus er is nog geen afbeelding.",
           "added": "Met ε = {e} zijn je kansen samen {t}, niet 1.",
           "note": "Met ε = 0 stopt de policy altijd; met ε = 1 is elke zet willekeurig, de helft stick en de helft hit."},
}


def _chances(got) -> dict | None:
    if not isinstance(got, dict) or set(got) != set(ACTIONS):
        return None
    try:
        out = {a: float(got[a]) for a in ACTIONS}
    except (TypeError, ValueError):
        return None
    return out if all(math.isfinite(v) and abs(v) < 1e6 for v in out.values()) else None


def soft_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _SOFT_TEXT[lang]
    fn = _fn(seen)
    got = None
    if fn is not None:
        got = _call(lambda: [_chances(fn(start_policy(), SOFT_STATE, e)) for e in SOFT_EPSILONS])
        if not isinstance(got, list) or any(c is None for c in got):
            got = None
    if got is None:
        return yours.figure(w["title"], _note(w["head"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    rows = [(w["row"].format(e=number(e, 1, lang) if e not in (0.0, 1.0) else str(int(e)), a=a), max(0.0, round(c[a], 4)))
            for e, c in zip(SOFT_EPSILONS, got) for a in ACTIONS]
    bars = draw.hbars(rows, top=max(1.0, max(v for _r, v in rows)), alt=w["alt"]).text
    if lang == "nl":
        bars = chart.with_decimal_comma(bars)
    body = _note(w["head"]) + bars
    for e, c in zip(SOFT_EPSILONS, got):
        t = c["stick"] + c["hit"]
        if abs(t - 1) > 1e-9:
            body += _note(w["added"].format(e=number(e, 1, lang) if e not in (0.0, 1.0) else str(int(e)), t=number(t, 3, lang)), _LOSS_UI)
            break
    return yours.figure(w["title"], body + _note(w["note"]), alt=w["title"])


# --- how an answer is judged --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Judged:
    """What update did to Q and the policy over a list of hands, as far as the check is concerned: Q of every state and
    action rounded to six decimals, the policy, and `off`, where they first differ from the course's answer."""
    kind: str                 # "ok"; "shape": a state without "stick" or "hit", or a Q that is not a number
    q: tuple = ()             # ((state, action, Q), ...) for every pair that a hand visited
    policy: tuple = ()        # ((state, action), ...) for every state that a hand visited
    off: tuple | None = None  # ("Q", (state, action), got, want) or ("policy", state, got, want)


def _r6(x) -> float:
    return round(float(x), 6) + 0.0


def _visited(hands) -> tuple:
    pairs, states = [], []
    for hand, _r in hands:
        for s, a in hand:
            if (s, a) not in pairs:
                pairs.append((s, a))
            if s not in states:
                states.append(s)
    return tuple(pairs), tuple(states)


def _expected_action(Q: dict, s) -> set:
    """The actions a right update may choose in state s from its own Q: hit when it is larger, stick when it is smaller
    or exactly the same, and either when the two differ by less than a billionth, as two ways to work out one average
    can."""
    d = float(Q[(s, "hit")]) - float(Q[(s, "stick")])
    if d == 0 or d < -1e-9:
        return {"stick"}
    return {"hit"} if d > 1e-9 else {"stick", "hit"}


def judge_update(hands, fn) -> Judged:
    """Runs fn as the course does, `update(Q, count, policy, hand, reward)` for every hand, from Q and count at 0 and
    the start policy, and judges Q and the policy it leaves."""
    Q, count = fresh()
    policy = start_policy()
    for hand, reward in hands:
        fn(Q, count, policy, list(hand), reward)
    if len(hands) and (Q, count, policy) == (*fresh(), start_policy()):
        return None   # nothing changed: a function that made new dictionaries, or that changes nothing
    cq, cc = fresh()
    cp = start_policy()
    for hand, reward in hands:
        update(cq, cc, cp, list(hand), reward)
    pairs, states = _visited(hands)
    try:
        q = tuple((s, a, _r6(Q[(s, a)])) for s, a in pairs)
    except (TypeError, ValueError, KeyError):
        return Judged("shape")
    if not all(math.isfinite(v) for _s, _a, v in q) or any(policy.get(s) not in ACTIONS for s in states):
        return Judged("shape")
    pol = tuple((s, policy[s]) for s in states)
    off = next((("Q", (s, a), v, cq[(s, a)]) for s, a, v in q if abs(v - cq[(s, a)]) > 1e-6), None)
    if off is None:
        off = next((("policy", s, policy[s], cp[s]) for s in states if policy[s] not in _expected_action(Q, s)), None)
    return Judged("ok", q, pol, off)


def same_judged(a, b) -> bool:
    """Right when Q agrees within a millionth for the same states and actions; a right policy may differ where its Q
    of the two actions is a tie up to the last digits, a wrong one must be the same everywhere."""
    if not (isinstance(a, Judged) and isinstance(b, Judged)):
        return a == b
    if a.kind != b.kind:
        return False
    if a.kind != "ok":
        return True
    if (a.off is None) != (b.off is None) or len(a.q) != len(b.q):
        return False
    if any(sa != sb or aa != ab or abs(va - vb) > 1e-6 for (sa, aa, va), (sb, ab, vb) in zip(a.q, b.q)):
        return False
    return a.off is None or a.policy == b.policy


def _plain(x: float) -> str:
    """A number the way the check shows it: up to four decimals, a real minus sign."""
    text = f"{x:.4f}".rstrip("0").rstrip(".")
    if text in ("-0", "-0.", ""):
        text = "0"
    return text.replace("-", "−")


def show_update(judged) -> dict:
    """What update left behind, in words: where it first differs from the course's answer."""
    if not isinstance(judged, Judged) or judged.kind == "shape":
        return {"en": "a `Q` without a number for every state and action, or a state without the action `\"stick\"` or `\"hit\"`",
                "nl": "een `Q` zonder getal voor elke state en action, of een state zonder de action `\"stick\"` of `\"hit\"`"}
    if judged.off is not None:
        kind, key, got, want = judged.off
        if kind == "Q":
            s, a = key
            return {"en": f"`Q[({s}, \"{a}\")] = {_plain(got)}`, where it should be `{_plain(want)}`",
                    "nl": f"`Q[({s}, \"{a}\")] = {_plain(got)}`, waar het `{_plain(want)}` moet zijn"}
        return {"en": f"`policy[{key}] = \"{got}\"`, where it should be `\"{want}\"`",
                "nl": f"`policy[{key}] = \"{got}\"`, waar het `\"{want}\"` moet zijn"}
    if not judged.q:
        return {"en": "nothing changed, the right answer for no hands", "nl": "niets veranderd, het goede antwoord bij geen handen"}
    return {"en": "the right `Q` and policy", "nl": "de goede `Q` en policy"}


@dataclass
class UpdateBox(FunctionBox):
    """update in a code box: the check runs it over a list of hands, as Monte Carlo control does after every hand, and
    judges the Q and the policy it leaves, so that a known mistake is known by what it does in every case."""

    def _run(self, fn, case):
        return judge_update(case[0], fn)

    def _call_text(self, case) -> str:
        return f"for hand, reward in {case[0]!r}: update(Q, count, policy, hand, reward)"


def same_chances(a, b) -> bool:
    """Right when the chances of both actions agree within a millionth."""
    ca, cb = _chances(a), _chances(b)
    if ca is None or cb is None:
        return a == b
    return all(abs(ca[x] - cb[x]) <= 1e-6 for x in ACTIONS)


def show_chances(value) -> dict:
    c = _chances(value)
    if c is None:
        return {"en": "something other than a dictionary with a chance for `\"stick\"` and for `\"hit\"`",
                "nl": "iets anders dan een dictionary met een kans voor `\"stick\"` en voor `\"hit\"`"}
    shown = "`{" + ", ".join(f'"{a}": {_plain(c[a])}' for a in ACTIONS) + "}`"
    return {"en": shown, "nl": shown}


# --- the exercises -----------------------------------------------------------------------------------------------------

REQUIRED = ["update", "soft", "casino"]


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


# exercise 1: update
def _u(line: str = "Q[(s, a)] + (reward - Q[(s, a)]) / count[(s, a)]",
       when: str = 'Q[(s, "hit")] > Q[(s, "stick")]') -> str:
    """update as the box holds it, with these two parts: the new Q of (s, a), and when the policy hits."""
    return ("def update(Q, count, policy, hand, reward):\n"
            "    for s, a in hand:\n"
            "        count[(s, a)] = count[(s, a)] + 1\n"
            f"        Q[(s, a)] = {line}\n"
            f"        if {when}:\n"
            '            policy[s] = "hit"\n'
            "        else:\n"
            '            policy[s] = "stick"\n')


_U_START = _u("...", "...")
_U = _u()
_U_LAST = _u("reward")
_U_TOTAL = _u("Q[(s, a)] + reward")
_U_ETA = _u("Q[(s, a)] + 0.1 * (reward - Q[(s, a)])")
_U_DIVIDED = _u("reward / count[(s, a)]")
_U_INVERTED = _u(when='Q[(s, "hit")] < Q[(s, "stick")]')
_U_TIE = _u(when='Q[(s, "hit")] >= Q[(s, "stick")]')
_U_COPY = _u(when='a == "hit"')
_U_PLUS_ONE = _u("Q[(s, a)] + (reward - Q[(s, a)]) / (count[(s, a)] + 1)")
_U_NOTHING = _u().replace("    for s, a in hand:", "    Q, count, policy = dict(Q), dict(count), dict(policy)\n    for s, a in hand:")

S13, S17, S19 = (13, 2, False), (17, 2, False), (19, 2, False)
HANDS = Hands([
    ([(S13, "hit"), (S19, "stick")], 1),    # hit on 13, a 6 makes 19, stick: a win
    ([(S13, "hit")], -1),                   # hit on 13, a 10: over 21
    ([(S13, "hit"), (S17, "stick")], 0),    # hit on 13, a 4 makes 17, stick: the dealer gets 17 too
    ([(S19, "stick")], 0),                  # stick on 19: the dealer gets 19 too
    ([(S13, "stick")], -1),                 # stick on 13: the dealer gets 18; last, so that a policy that only
], "HANDS")                                 # repeats the last action takes the wrong one on 13
SOFT_HAND = Hands([([((15, 6, True), "hit"), ((15, 6, False), "hit"), ((21, 6, False), "stick")], 1)], "ACE_HAND")


class _LazyHands(Hands):
    """FIRST_HANDS, the first 1,000 hands of the play block, played only when a check needs them."""

    def __init__(self):
        super().__init__([], "FIRST_HANDS")
        self._ready = False

    def _fill(self):
        if not self._ready:
            self.extend(play_run()["hands"])
            self._ready = True

    def __iter__(self):
        self._fill()
        return super().__iter__()

    def __len__(self):
        self._fill()
        return super().__len__()

    def __getitem__(self, k):
        self._fill()
        return super().__getitem__(k)


FIRST_HANDS = _LazyHands()

UPDATE_CASES = [
    (HANDS,),
    (SOFT_HAND,),
    (Hands(list(HANDS) + list(SOFT_HAND), "HANDS + ACE_HAND"),),
    (FIRST_HANDS,),
    (Hands([], "[]"),),
]


# exercise 2: soft
def _s(line: str = "1 - epsilon + epsilon / 2") -> str:
    """soft as the box holds it, with this line: the chance of the policy's own action."""
    return ("def soft(policy, s, epsilon):\n"
            "    chances = {}\n"
            '    for a in ["stick", "hit"]:\n'
            "        if a == policy[s]:\n"
            f"            chances[a] = {line}\n"
            "        else:\n"
            "            chances[a] = epsilon / 2\n"
            "    return chances\n")


_S_START = _s("...")
_S = _s()
_S_MINUS = _s("1 - epsilon")
_S_ONE = _s("1")
_S_HALF = _s("epsilon / 2")


class _Policy(dict):
    """The start policy, shown by name in a check."""

    def __repr__(self):
        return "START"

    def __deepcopy__(self, memo):
        return dict(self)


START = _Policy(start_policy())
SOFT_CASES = [
    (START, (20, 10, False), 0.1),
    (START, (13, 2, False), 0.1),
    (START, (20, 10, False), 0.0),
    (START, (13, 2, False), 1.0),
    (START, (21, 1, True), 0.5),
    (START, (12, 7, True), 0.3),
]


EXERCISES = {
    "update": UpdateBox(
        name="update",
        function="update",
        start=_U_START,
        solution=_U,
        compare=same_judged,
        show=show_update,
        visible=UPDATE_CASES[:1],
        hidden=UPDATE_CASES[1:],
        first=[_DOTS],
        picture=update_picture,
        mistakes=[
            Mistake("last-return", wrong=_U_LAST,
                    en="`{call}` gives {got}: `Q` keeps only the return of the last hand through `(s, a)`. `Q[(s, a)]` should be the average of all returns after `(s, a)`, the return to expect; how many does `count[(s, a)]` say there are?",
                    nl="`{call}` geeft {got}: `Q` houdt alleen de return van de laatste hand langs `(s, a)`. `Q[(s, a)]` moet het gemiddelde zijn van alle returns na `(s, a)`, de return die je kunt verwachten; hoeveel zijn het er volgens `count[(s, a)]`?"),
            Mistake("total", wrong=_U_TOTAL,
                    en="`{call}` gives {got}: all returns after `(s, a)` added up, not their average. After a win and a draw, should `Q` say 1, or 0.5?",
                    nl="`{call}` geeft {got}: alle returns na `(s, a)` opgeteld, geen gemiddelde. Moet `Q` na een winst en een gelijkspel 1 zeggen, of 0,5?"),
            Mistake("fixed-eta", wrong=_U_ETA,
                    en="`{call}` gives {got}: a fixed learning rate of 0.1, as in Q-learning. `Q` then moves a tenth of the way toward each return and stays close to 0 at first; it is not the average. With which learning rate, for the `n`-th return, is it exactly the average?",
                    nl="`{call}` geeft {got}: een vaste learning rate van 0,1, zoals in Q-learning. `Q` schuift dan een tiende van de afstand op naar elke return en blijft eerst dicht bij 0; het is geen gemiddelde. Met welke learning rate, voor de `n`-de return, is het precies het gemiddelde?"),
            Mistake("divided", wrong=_U_DIVIDED,
                    en="`{call}` gives {got}: the last return divided by the count, and the returns before it are gone. Can you start from the old `Q[(s, a)]` and move it toward `reward`?",
                    nl="`{call}` geeft {got}: de laatste return gedeeld door het aantal, en de eerdere returns zijn weg. Kun je beginnen bij de oude `Q[(s, a)]` en deze waarde naar `reward` laten opschuiven?"),
            Mistake("inverted", wrong=_U_INVERTED,
                    en="`{call}` gives {got}: your policy takes the action that brought less. Which action should the policy take, the one with the larger `Q` or the smaller?",
                    nl="`{call}` geeft {got}: je policy kiest de action die minder opleverde. Welke action moet de policy kiezen, de action met de grotere `Q` of de kleinere?"),
            Mistake("copies-action", wrong=_U_COPY,
                    en="`{call}` gives {got}: your policy repeats the action of the last hand in each state, whether it brought much or little. Which two numbers should the `if` compare, so that the policy takes the action that brought more on average?",
                    nl="`{call}` geeft {got}: je policy herhaalt de action van de laatste hand in elke state, of deze action nu veel of weinig opleverde. Welke twee getallen moet de `if` vergelijken, zodat de policy de action kiest die gemiddeld meer opleverde?"),
            Mistake("plus-one", wrong=_U_PLUS_ONE,
                    en="`{call}` gives {got}: `count[(s, a)]` has already gone up by 1 for this hand, so dividing by one more moves `Q` too little: after a first return of 1, `Q` is 0.5. What should `Q` be after one return?",
                    nl="`{call}` geeft {got}: `count[(s, a)]` is voor deze hand al met 1 opgehoogd, dus delen door één meer laat `Q` te weinig opschuiven: na een eerste return van 1 is `Q` 0,5. Wat moet `Q` zijn na één return?"),
            Mistake("tie-hits", wrong=_U_TIE,
                    en="`{call}` gives {got}: on a tie your policy hits, also where both actions still have a `Q` of 0 because no hand tried them. The course sticks on a tie; which sign makes the `if` true only when hitting is strictly better?",
                    nl="`{call}` geeft {got}: bij gelijke `Q` neemt je policy een kaart, ook waar beide actions nog een `Q` van 0 hebben omdat nog geen hand deze actions probeerde. De course stopt bij gelijke `Q`; welk teken maakt de `if` alleen waar als een kaart nemen echt beter is?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: the new `Q[(s, a)]`, the average of its returns, and when the policy hits.",
                      "nl": "Vervang de twee `...`: de nieuwe `Q[(s, a)]`, het gemiddelde van de returns, en wanneer de policy een kaart neemt."},
            "arguments": {"en": "The check calls `update(Q, count, policy, hand, reward)` for every hand, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(Q, count, policy, hand, reward)`?",
                          "nl": "De check roept `update(Q, count, policy, hand, reward)` aan voor elke hand, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(Q, count, policy, hand, reward)` houden?"},
            "slow": {"en": "Your function is still not done after 2 seconds. Does the loop go over the steps of the hand, with `for s, a in hand:`?",
                     "nl": "Je functie is na 2 seconden nog niet klaar. Gaat de loop over de stappen van de hand, met `for s, a in hand:`?"},
            "wrong": {"en": "`{call}` gives {got}. Is `Q[(s, a)]` the average of all returns after `(s, a)`, and does the policy hit only when `Q` of hitting is larger?",
                      "nl": "`{call}` geeft {got}. Is `Q[(s, a)]` het gemiddelde van alle returns na `(s, a)`, en neemt de policy alleen een kaart als de `Q` van een kaart nemen groter is?"},
            "almost": {"en": "Right for `HANDS`. But `{call}` gives {got}. Does your function work for every list of hands, also `ACE_HAND` and `FIRST_HANDS`?",
                       "nl": "Goed voor `HANDS`. Maar `{call}` geeft {got}. Werkt je functie voor elke lijst met handen, ook `ACE_HAND` en `FIRST_HANDS`?"},
            "no_return": {"en": "With `{call}`, your function changes nothing. Does it still give `Q[(s, a)]` and `policy[s]` their new values?",
                          "nl": "Met `{call}` verandert je functie niets. Geeft je functie `Q[(s, a)]` en `policy[s]` nog een nieuwe waarde?"},
            "correct": {"en": "Right in all {n} cases: after the five hands of `HANDS`, hitting on 13 against a 2 has a `Q` of 0 and sticking −1, so the policy hits there. In the picture your `update` plays 100,000 hands; in how many states does its policy differ from the best one?",
                        "nl": "Goed in alle {n} gevallen: na de vijf handen van `HANDS` heeft een kaart nemen op 13 tegen een 2 een `Q` van 0 en stoppen −1, dus de policy neemt daar een kaart. In de afbeelding speelt je `update` 100.000 handen; in hoeveel states verschilt de policy van de beste?"},
        },
    ),
    "soft": FunctionBox(
        name="soft",
        function="soft",
        start=_S_START,
        solution=_S,
        compare=same_chances,
        show=show_chances,
        visible=SOFT_CASES[:1],
        hidden=SOFT_CASES[1:],
        first=[_DOTS],
        picture=soft_picture,
        mistakes=[
            Mistake("one-minus", wrong=_S_MINUS,
                    en="`{call}` gives {got}: the two chances add up to less than 1. A random move can also pick the policy's own action; what chance does that action get then, all in all?",
                    nl="`{call}` geeft {got}: de twee kansen zijn samen minder dan 1. Een willekeurige zet kan ook de action van de policy kiezen; welke kans krijgt deze action dan, alles bij elkaar?"),
            Mistake("no-random", wrong=_S_ONE,
                    en="`{call}` gives {got}: the two chances add up to more than 1. With probability ε the move is random; how much is left for the policy's own action, plus its share of the random moves?",
                    nl="`{call}` geeft {got}: de twee kansen zijn samen meer dan 1. Met kans ε is de zet willekeurig; hoeveel blijft er over voor de action van de policy, plus het eigen deel van de willekeurige zetten?"),
            Mistake("both-half", wrong=_S_HALF,
                    en="`{call}` gives {got}: both actions get ε/2, the share of a random move only. The policy's own action also gets every move that is not random; how large is that part?",
                    nl="`{call}` geeft {got}: beide actions krijgen ε/2, alleen het deel van een willekeurige zet. De action van de policy krijgt ook elke zet die niet willekeurig is; hoe groot is dat deel?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the chance of the policy's own action: the moves that are not random, plus its share of the random ones.",
                      "nl": "Vervang de `...` door de kans op de action van de policy: de zetten die niet willekeurig zijn, plus het eigen deel van de willekeurige zetten."},
            "arguments": {"en": "The check calls `{call}`, with a policy, a state and ε, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(policy, s, epsilon)`?",
                          "nl": "De check roept `{call}` aan, met een policy, een state en ε, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(policy, s, epsilon)` houden?"},
            "slow": {"en": "Your function is still not done after 2 seconds. Does the loop go over the two actions, with `for a in [\"stick\", \"hit\"]:`?",
                     "nl": "Je functie is na 2 seconden nog niet klaar. Gaat de loop over de twee actions, met `for a in [\"stick\", \"hit\"]:`?"},
            "wrong": {"en": "`{call}` gives {got}. Do the two chances add up to 1, with ε/2 for the other action?",
                      "nl": "`{call}` geeft {got}. Zijn de twee kansen samen 1, met ε/2 voor de andere action?"},
            "almost": {"en": "Right for `soft(START, (20, 10, False), 0.1)`. But `{call}` gives {got}. Does your function work for every ε, also 0 and 1?",
                       "nl": "Goed voor `soft(START, (20, 10, False), 0.1)`. Maar `{call}` geeft {got}. Werkt je functie voor elke ε, ook 0 en 1?"},
            "no_return": {"en": "With `{call}`, your function gives back `None`, and it should give the chances. Does it still end with `return chances`?",
                          "nl": "Met `{call}` geeft je functie `None` terug, en het moeten de kansen zijn. Eindigt je functie nog met `return chances`?"},
            "correct": {"en": "Right in all {n} cases: with ε = 0.1 the policy's action gets 0.95 and the other 0.05, so one move in twenty goes the other way. In the demo below, your `soft` plays the hands; what does a hand cost while the agent keeps trying both actions?",
                        "nl": "Goed in alle {n} gevallen: met ε = 0,1 krijgt de action van de policy 0,95 en de andere 0,05, dus één zet op de twintig gaat de andere kant op. In de demo hieronder speelt je `soft` de handen; wat kost een hand terwijl de agent beide actions blijft proberen?"},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("update", _U_START, "empty", None),
    ("update", _U, "correct", None),
    ("update", _u("Q[(s, a)] + ..."), "wrong", "dots-left"),
    ("update", _U_LAST, "wrong", "last-return"),
    ("update", _U_TOTAL, "wrong", "total"),
    ("update", _U_ETA, "wrong", "fixed-eta"),
    ("update", _U_DIVIDED, "wrong", "divided"),
    ("update", _U_INVERTED, "wrong", "inverted"),
    ("update", _U_TIE, "wrong", "tie-hits"),
    ("update", _U_COPY, "wrong", "copies-action"),
    ("update", _U_PLUS_ONE, "wrong", "plus-one"),
    # new dictionaries in the function: the ones of the check stay as they were
    ("update", _U_NOTHING, "wrong", "no-return"),
    # hitting where Q of the action taken beats sticking: wrong on the last hand of HANDS
    ("update", _u(when='Q[(s, a)] > Q[(s, "stick")]'), "wrong", None),
    # other right answers: the average worked out from the old total, and the comparison the other way around
    ("update", _u("(Q[(s, a)] * (count[(s, a)] - 1) + reward) / count[(s, a)]"), "correct", None),
    ("update", _u(when='Q[(s, "stick")] < Q[(s, "hit")]'), "correct", None),
    # an ace that stops counting is a new state: wrong only in a hand with such an ace
    ("update", _u(when='Q[(s, "hit")] > Q[(s, "stick")] or s[2]'), "almost", None),
    ("soft", _S_START, "empty", None),
    ("soft", _S, "correct", None),
    ("soft", _s("1 - ..."), "wrong", "dots-left"),
    ("soft", _S_MINUS, "wrong", "one-minus"),
    ("soft", _S_ONE, "wrong", "no-random"),
    ("soft", _S_HALF, "wrong", "both-half"),
    # another right answer: the same chance, written shorter
    ("soft", _s("1 - epsilon / 2"), "correct", None),
    # a fixed 0.95: right for ε = 0.1 only
    ("soft", _s("0.95"), "almost", None),
]
