# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson llm-preference-pairs: bt_prob, the Bradley-Terry probability that one answer is preferred
to another, from their scores; make_pairs, the preference pairs a checker gives among answers to one instruction;
bt_loss, the mean loss of scores on comparisons; and a choice. The demo is an arena: the base model of the capstone,
it after full SFT and it with LoRA answered the same test instructions, a checker says which answer is better
wherever two differ, and Bradley-Terry scores are fitted to those comparisons with the learner's functions."""
import importlib.util
import itertools
import math

from everything_ai import paths, projects
from everything_ai.check import FunctionBox, Mistake


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LO = _checks("llm-lora", "everything_ai_checks_llm_lora_for_preference")
T, S = LO.T, LO.S
BUILDS_ON = ()
projects.builds_on(*BUILDS_ON)
SAVES = (("llm", "preference_pairs", "llm-preference-pairs", "make-pairs"),)

MODELS = ("base", "sft", "lora")   # the players of the arena
STEPS, RATE = 2000, 0.5            # gradient descent on the mean loss, from scores of 0


def number(x: float, digits: int, lang: str) -> str:
    return S.number(x, digits, lang)


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def _bits(hits) -> list:
    """Hits as a list of bools, from a list or from a string of 0 and 1."""
    return [c == "1" for c in hits] if isinstance(hits, str) else [bool(h) for h in hits]


def arena(standin: bool = False) -> dict:
    """Which of the test instructions each player did: the learner's runs of llm-sft and llm-lora when both are there,
    on the same model of the capstone, else the course's runs of their demos (their MEASURED, on its model of the demo
    run of llm-pretrain-capstone). With "source", "yours" or None."""
    sft, lora = (None, None) if standin else (T.saved(), LO.saved())
    if sft is not None and lora is not None and "hits" in sft and "hits" in lora and sft["base"] == lora["base"]:
        return {"base": _bits(sft["hits"]["before"]), "sft": _bits(sft["hits"]["after"]), "lora": _bits(lora["hits"]),
                "source": "yours"}
    sft, lora = T.MEASURED["yours"]["hits"], LO.MEASURED["yours"]["hits"]
    return {"base": _bits(sft["before"]), "sft": _bits(sft["after"]), "lora": _bits(lora), "source": None}


def comparisons(hits: dict) -> list:
    """(winner, loser) for every instruction and every two players of which exactly one did it."""
    out = []
    for a, b in itertools.combinations(MODELS, 2):
        for ha, hb in zip(hits[a], hits[b]):
            if ha and not hb:
                out.append((a, b))
            elif hb and not ha:
                out.append((b, a))
    return out


def fit(games: list, bt_prob, steps: int = STEPS, rate: float = RATE) -> dict | str:
    """Bradley-Terry scores for the players of `games`, by gradient descent on the mean of -log P(winner preferred),
    with the learner's bt_prob for P; the score of "base" is held at 0, because only differences count. "prob" when
    bt_prob stops with an error or gives no probability."""
    scores = {m: 0.0 for m in MODELS}
    count = {k: n for k, n in wins(games).items() if n}   # the games of the same two players all have the same p
    for _ in range(steps):
        grad = {m: 0.0 for m in MODELS}
        for (w, l), n in count.items():
            try:
                p = float(bt_prob(scores[w], scores[l]))
            except Exception:
                return "prob"
            if not 0.0 < p <= 1.0:
                return "prob"
            grad[w] -= n * (1 - p) / len(games)   # the derivative of -log sigmoid(d) is -(1 - sigmoid(d))
            grad[l] += n * (1 - p) / len(games)
        for m in MODELS[1:]:
            scores[m] -= rate * grad[m]
    return scores


def course_fit(standin: bool = False) -> dict:
    """The fit with the course's own functions, which stands in until the learner fits with theirs; with `standin`, on
    the course's runs even when the learner has runs of their own."""
    hits = arena(standin)
    games = comparisons(hits)
    scores = fit(games, lambda a, b: sigmoid(a - b))
    loss = lambda sc: sum(-math.log(sigmoid(sc[w] - sc[l])) for w, l in games) / max(1, len(games))
    return {"hits": hits, "games": games, "scores": scores, "start": loss({m: 0.0 for m in MODELS}), "end": loss(scores)}


def wins(games: list) -> dict:
    """How often each player beat each other player: {(a, b): wins of a over b}."""
    out = {(a, b): 0 for a in MODELS for b in MODELS if a != b}
    for w, l in games:
        out[(w, l)] += 1
    return out


def curve_picture(points: list, labels: dict) -> str:
    """The Bradley-Terry probability against the difference of two scores, from -4 to 4, with a point and its words for
    every (difference, words) of `points`. The curve is a prediction, so it has the colour of predictions. A label sits
    where the rising curve leaves room: below and to the right of a point at a positive difference, above and to the
    left of one at a negative difference, and off the other labels and the dashed lines. The axis reaches far enough
    past the farthest point for its label."""
    from everything_ai import chart
    # a label of 14 px takes about 8 px a letter, of the 486 px of the axis; it starts 4% of the half-axis from its point
    far = max([4] + [math.ceil(abs(d) / max(0.2, 0.96 - 16 * len(words) / 486)) for d, words in points])
    ds = [-far + 2 * far * i / 160 for i in range(161)]
    p = chart.Panel((-far, far), (-0.05, 1.05), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    for level in (0, 0.5, 1):
        p.line([-far, far], [level, level], "ink-3", 1.2, dashed=True)
    p.line(ds, [sigmoid(d) for d in ds], "pred")
    for side in (1, -1):   # right of the points at 0 or more, from the right; left of the others, from the left
        taken = []
        for d, words in sorted((pt for pt in points if (pt[0] >= 0) == (side == 1)), key=lambda pt: -side * pt[0]):
            y = sigmoid(d) - side * 0.12
            while any(abs(y - t) < 0.09 for t in taken) or any(abs(y - level) < 0.05 for level in (0, 0.5, 1)):
                y -= side * 0.03
            taken.append(y)
            p.points([d], [sigmoid(d)], "pred", 5)
            p.text(d + side * far * 0.04, y, words, "ink", anchor="start" if side == 1 else "end")
    svg = chart.figure([p], alt=labels["alt"], width=560, height=320)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg



REQUIRED = ["bt-prob", "make-pairs", "bt-loss", "shift"]

_DOTS = S._DOTS
_TEXTS = S._TEXTS
_CALL, _CALL_NL = S._CALL, S._CALL_NL


def _close(got, want) -> bool:
    try:
        return math.isclose(float(got), float(want), rel_tol=1e-6, abs_tol=1e-9)   # also float32, as in torch
    except (TypeError, ValueError):
        return False


def _shown(v) -> str:
    return S._rounded(v)


def _same_pairs(got, want) -> bool:
    """The same pairs, in any order; every pair two answers, not one string."""
    try:
        if not isinstance(got, list):   # a list, as the task asks: the DPO project goes through it more than once
            return False
        if not all(isinstance(p, (tuple, list)) and len(p) == 2 for p in got):
            return False
        return sorted(tuple(p) for p in got) == sorted(tuple(p) for p in want)
    except TypeError:
        return False


# --- exercise 1: bt_prob -----------------------------------------------------------------------------------------------

_P_HEAD = "import math\n\n\ndef bt_prob(score_a, score_b):\n"
_P = _P_HEAD + "    return 1 / (1 + math.exp(-(score_a - score_b)))\n"
_P_START = _P_HEAD + "    # the sigmoid of the difference of the scores\n    return ...\n"

# --- exercise 2: make_pairs --------------------------------------------------------------------------------------------

_M_HEAD = "def make_pairs(answers, hits):\n"
_M = (_M_HEAD + "    pairs = []\n    for i in range(len(answers)):\n        for j in range(len(answers)):\n"
      "            if hits[i] and not hits[j]:\n                pairs.append((answers[i], answers[j]))\n    return pairs\n")
_M_START = (_M_HEAD + "    pairs = []\n    for i in range(len(answers)):\n        for j in range(len(answers)):\n"
            "            # answer i is chosen and answer j rejected when the checker passes i and not j\n            ...\n"
            "    return pairs\n")
_M_ANY = _M.replace("if hits[i] and not hits[j]:", "if hits[i] != hits[j]:")
_M_TIES = _M.replace("if hits[i] and not hits[j]:", "if i != j and (hits[i] or not hits[j]):")
_M_SELF = _M.replace("if hits[i] and not hits[j]:", "if hits[i]:")
_M_JOINED = _M.replace("pairs.append((answers[i], answers[j]))", "pairs.append(answers[i] + answers[j])")
_M_FIRST = (_M_HEAD + "    pairs = []\n    for i in range(len(answers)):\n        for j in range(len(answers)):\n"
            "            if hits[i] and not hits[j]:\n                pairs.append((answers[i], answers[j]))\n                break\n"
            "    return pairs\n")

# --- exercise 3: bt_loss -----------------------------------------------------------------------------------------------

_L_HEAD = "import math\n\n\ndef bt_loss(games, scores):\n"
_L = (_L_HEAD + "    total = 0\n    for winner, loser in games:\n"
      "        p = 1 / (1 + math.exp(-(scores[winner] - scores[loser])))\n        total += -math.log(p)\n"
      "    return total / len(games)\n")
_L_START = (_L_HEAD + "    total = 0\n    for winner, loser in games:\n"
            "        # the probability that the winner is preferred, and its loss\n        ...\n"
            "    return total / len(games)\n")
_L_SUM = _L.replace("    return total / len(games)\n", "    return total\n")
_L_SIGN = _L.replace("total += -math.log(p)", "total += math.log(p)")
_L_SWAP = _L.replace("scores[winner] - scores[loser]", "scores[loser] - scores[winner]")
_L_PROB = _L.replace("total += -math.log(p)", "total += 1 - p")

EXERCISES = {
    "bt-prob": FunctionBox(
        name="bt-prob",
        function="bt_prob",
        start=_P_START,
        solution=_P,
        visible=[(1.0, 0.0)],
        hidden=[(0.0, 0.0), (2.0, 3.0), (-1.5, -1.5), (0.4, -2.1), (10.0, 4.0)],
        show=_shown,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("no-sigmoid", wrong=_P_HEAD + "    return score_a - score_b\n",
                    en=_CALL + ": that is the difference of the scores, and it can be any number. The probability is the sigmoid of that difference: it turns any number into one between 0 and 1.",
                    nl=_CALL_NL + ": dat is het verschil van de scores, en dat kan elk getal zijn. De kans is de sigmoid van dat verschil: die maakt van elk getal een getal tussen 0 en 1."),
            Mistake("reversed", wrong=_P_HEAD + "    return 1 / (1 + math.exp(-(score_b - score_a)))\n",
                    en=_CALL + ": that is the probability that `b` is preferred. A higher score of `a` must give a higher probability for `a`: the sigmoid of `score_a - score_b`.",
                    nl=_CALL_NL + ": dat is de kans dat `b` de voorkeur krijgt. Een hogere score van `a` moet een hogere kans voor `a` geven: de sigmoid van `score_a - score_b`."),
            Mistake("ratio", wrong=_P_HEAD + "    return score_a / (score_a + score_b)\n",
                    en=_CALL + ": you divide one score by the sum of both. That is the form for strengths that are positive numbers, and for two scores of 0 it divides by zero. Scores can be any number: the probability is the sigmoid of their difference.",
                    nl=_CALL_NL + ": je deelt de ene score door de som van beide. Dat is de vorm voor sterktes die positieve getallen zijn, en voor twee scores van 0 deelt hij door nul. Scores kunnen elk getal zijn: de kans is de sigmoid van hun verschil."),
            Mistake("rounded", wrong=_P_HEAD + "    return round(1 / (1 + math.exp(-(score_a - score_b))), 4)\n",
                    en=_CALL + ": you round the probability to 4 decimals, so it is a little off. Give it back as it is: rounding is for showing a number, as in the `print` of the worked example.",
                    nl=_CALL_NL + ": je rondt de kans af op 4 decimalen, dus hij wijkt een beetje af. Geef hem terug zoals hij is: afronden is om een getal te laten zien, zoals in de `print` van het uitgewerkte voorbeeld."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: the sigmoid of the difference of the two scores.",
                      "nl": "Vervang de `...`: de sigmoid van het verschil van de twee scores."},
            "wrong": {"en": _CALL + ". The sigmoid of which number, made from both scores, and in which order?",
                      "nl": _CALL_NL + ". De sigmoid van welk getal, gemaakt uit beide scores, en in welke volgorde?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function use both scores, and no fixed numbers?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Gebruikt je functie beide scores, en geen vaste getallen?"},
            "correct": {"en": "Your function turns two scores into the probability that the first is preferred: only their difference counts.",
                        "nl": "Je functie maakt van twee scores de kans dat de eerste de voorkeur krijgt: alleen hun verschil telt."},
        },
    ),
    "make-pairs": FunctionBox(
        name="make-pairs",
        function="make_pairs",
        start=_M_START,
        solution=_M,
        visible=[(["A", "B", "C"], [True, False, False])],
        hidden=[(["A", "B"], [True, True]), (["A", "B"], [False, False]), (["P", "Q", "R", "S"], [False, True, True, False]),
                (["only"], [True]), (["x", "y", "z"], [False, False, True]),
                (["Tom flew his kite.", "The cat slept.", "A kite in the park."], [True, False, True])],
        compare=_same_pairs,
        save_as=("llm", "preference_pairs"),
        first=[_DOTS],
        mistakes=[
            Mistake("any", wrong=_M_ANY,
                    en=_CALL + ": you also make pairs whose chosen answer failed the check and whose rejected answer passed it. Only an answer that passes is chosen, and only one that fails is rejected.",
                    nl=_CALL_NL + ": je maakt ook paren waarvan het gekozen antwoord de check niet haalde en het afgewezen antwoord wel. Alleen een antwoord dat slaagt wordt gekozen, en alleen een dat zakt afgewezen."),
            Mistake("ties", wrong=_M_TIES,
                    en=_CALL + ": you also pair two answers that the checker treats the same, both passed or both failed. Such a pair says nothing about which is better: leave it out.",
                    nl=_CALL_NL + ": je maakt ook paren van twee antwoorden die de checker hetzelfde vindt, allebei geslaagd of allebei gezakt. Zo'n paar zegt niets over welk beter is: laat het weg."),
            Mistake("self", wrong=_M_SELF,
                    en=_CALL + ": an answer that passed is paired with every answer, also with itself and with others that passed. The rejected answer must fail the check.",
                    nl=_CALL_NL + ": een antwoord dat slaagde wordt gekoppeld aan elk antwoord, ook aan zichzelf en aan andere antwoorden die slaagden. Het afgewezen antwoord moet zakken voor de check."),
            Mistake("first-only", wrong=_M_FIRST,
                    en=_CALL + ": every answer that passed gets only one rejected answer, the first. Pair it with every answer that failed: there is no `break`.",
                    nl=_CALL_NL + ": elk antwoord dat slaagde krijgt maar één afgewezen antwoord, het eerste. Koppel het aan elk antwoord dat zakte: er is geen `break`."),
            Mistake("joined", matches=lambda got, text: isinstance(got, list) and bool(got) and all(isinstance(p, str) for p in got),
                    en=_CALL + ": every pair is one string. A pair keeps the two answers apart: a tuple of the chosen and the rejected answer, in that order.",
                    nl=_CALL_NL + ": elk paar is één string. Een paar houdt de twee antwoorden apart: een tuple van het gekozen en het afgewezen antwoord, in die volgorde."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: add a pair to `pairs` when answer i passed the check and answer j did not.",
                      "nl": "Vervang de `...`: voeg een paar toe aan `pairs` als antwoord i de check haalde en antwoord j niet."},
            "wrong": {"en": _CALL + ". A pair (chosen, rejected) for every answer that passed and every answer that failed?",
                      "nl": _CALL_NL + ". Een paar (gekozen, afgewezen) voor elk antwoord dat slaagde en elk antwoord dat zakte?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". All passed, all failed, and more than one of each?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Allemaal geslaagd, allemaal gezakt, en meer dan één van elk?"},
            "correct": {"en": "Your function turns a checker's verdicts into preference pairs, and leaves out the pairs that say nothing. It is in your project now, as `{saved}`.",
                        "nl": "Je functie maakt van de oordelen van een checker preference pairs, en laat de paren weg die niets zeggen. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `make_pairs` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `make_pairs` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "bt-loss": FunctionBox(
        name="bt-loss",
        function="bt_loss",
        start=_L_START,
        solution=_L,
        visible=[([("lora", "base"), ("sft", "base"), ("base", "sft"), ("lora", "sft")], {"base": 0.0, "sft": 1.0, "lora": 1.5})],
        hidden=[([("a", "b")], {"a": 0.0, "b": 0.0}), ([("b", "a"), ("b", "a")], {"a": 2.0, "b": -1.0}),
                ([("x", "y"), ("y", "z"), ("x", "z")], {"x": 0.3, "y": 0.1, "z": -0.4})],
        show=_shown,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("sum", wrong=_L_SUM,
                    en=_CALL + ": that is the sum of the losses. Divide by the number of games, `len(games)`, for their mean: then more games do not make a bigger loss.",
                    nl=_CALL_NL + ": dat is de som van de losses. Deel door het aantal partijen, `len(games)`, voor hun gemiddelde: dan maken meer partijen de loss niet groter."),
            Mistake("sign", wrong=_L_SIGN,
                    en=_CALL + ": the log of a probability is negative, so your loss is too. The loss is minus the log: `-math.log(p)`, which is 0 for a sure win and grows when the winner was unlikely.",
                    nl=_CALL_NL + ": de log van een kans is negatief, dus je loss ook. De loss is min de log: `-math.log(p)`, dat 0 is voor een zekere winst en groeit als de winnaar onwaarschijnlijk was."),
            Mistake("swap", wrong=_L_SWAP,
                    en=_CALL + ": your loss takes the log of the probability that the loser is preferred. It needs the probability that the winner is preferred: then the loss is small when the scores agree with the game.",
                    nl=_CALL_NL + ": je loss neemt de log van de kans dat de verliezer de voorkeur krijgt. Hij heeft de kans nodig dat de winnaar de voorkeur krijgt: dan is de loss klein als de scores het eens zijn met de partij."),
            Mistake("one-minus", wrong=_L_PROB,
                    en=_CALL + ": you add `1 - p`, how far the probability is from 1. The loss of Bradley-Terry is `-math.log(p)`, the cross-entropy of the game: it grows without end for a winner the scores call very unlikely.",
                    nl=_CALL_NL + ": je telt `1 - p` op, hoe ver de kans van 1 af ligt. De loss van Bradley-Terry is `-math.log(p)`, de cross-entropy van de partij: die groeit zonder grens voor een winnaar die de scores heel onwaarschijnlijk noemen."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: the probability `p` that the winner is preferred, and the loss of the game added to `total`.",
                      "nl": "Vervang de `...`: de kans `p` dat de winnaar de voorkeur krijgt, en de loss van de partij opgeteld bij `total`."},
            "wrong": {"en": _CALL + ". For every game the sigmoid of the score of the winner minus that of the loser, minus its log, and the mean over the games?",
                      "nl": _CALL_NL + ". Voor elke partij de sigmoid van de score van de winnaar min die van de verliezer, min zijn log, en het gemiddelde over de partijen?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function use the names in `games` to look up their scores?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Gebruikt je functie de namen in `games` om hun scores op te zoeken?"},
            "correct": {"en": "Your loss is small when the scores agree with who won: fitting the scores is making it as small as it gets.",
                        "nl": "Je loss is klein als de scores het eens zijn met wie won: de scores fitten is hem zo klein mogelijk maken."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("bt-prob", _P_START, "empty", None),
    ("bt-prob", _P, "correct", None),
    ("bt-prob", _P_HEAD + "    return math.exp(score_a) / (math.exp(score_a) + math.exp(score_b))\n", "correct", None),
    ("bt-prob", "import numpy as np\n\n\ndef bt_prob(score_a, score_b):\n    return float(1 / (1 + np.exp(score_b - score_a)))\n", "correct", None),
    ("bt-prob", "import numpy as np\n\n\ndef bt_prob(score_a, score_b):\n    return np.float32(1 / (1 + np.exp(score_b - score_a)))\n", "correct", None),
    ("bt-prob", _P_HEAD + "    return score_a - score_b\n", "wrong", "no-sigmoid"),
    ("bt-prob", _P_HEAD + "    return 1 / (1 + math.exp(-(score_b - score_a)))\n", "wrong", "reversed"),
    ("bt-prob", _P_HEAD + "    return score_a / (score_a + score_b)\n", "wrong", "ratio"),
    ("bt-prob", _P_HEAD + "    return round(1 / (1 + math.exp(-(score_a - score_b))), 4)\n", "wrong", "rounded"),
    ("make-pairs", _M_START, "empty", None),
    ("make-pairs", _M, "correct", None),
    ("make-pairs", _M_HEAD + "    good = [a for a, h in zip(answers, hits) if h]\n    bad = [a for a, h in zip(answers, hits) if not h]\n"
                   "    return [(g, b) for b in bad for g in good]\n", "correct", None),
    ("make-pairs", _M_ANY, "wrong", "any"),
    ("make-pairs", _M_TIES, "wrong", "ties"),
    ("make-pairs", _M_SELF, "wrong", "self"),
    ("make-pairs", _M_FIRST, "wrong", "first-only"),
    ("make-pairs", _M_JOINED, "wrong", "joined"),
    ("bt-loss", _L.replace("total += -math.log(p)", "total += -math.log(1 - p)"), "wrong", "swap"),
    ("bt-loss", _L_START, "empty", None),
    ("bt-loss", _L, "correct", None),
    ("bt-loss", _L_HEAD + "    return sum(math.log(1 + math.exp(scores[l] - scores[w])) for w, l in games) / len(games)\n", "correct", None),
    ("bt-loss", _L_SUM, "wrong", "sum"),
    ("bt-loss", _L_SIGN, "wrong", "sign"),
    ("bt-loss", _L_SWAP, "wrong", "swap"),
    ("bt-loss", _L_PROB, "wrong", "one-minus"),
]
