# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson attn-scale: score_spread, the spread of the dot product of random queries and keys; and
scaled_attention, attention whose scores are divided by the square root of the key length."""
import functools
import math
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai import chart
from everything_ai.check import FunctionBox, Mistake

SIZES = [4, 16, 64, 256, 1024]
ROWS, KEYS = 2000, 16


@functools.lru_cache(maxsize=1)
def sharpness() -> dict:
    """For every key length d_k: random queries and 16 random keys per query, numbers of spread 1, seed 0. Gives the
    spread of the scores, and per way (plain or divided by √d_k) the mean largest attention weight of a row and the
    mean size of the softmax's gradient (the Frobenius norm of its Jacobian, diag(a) − a aᵀ)."""
    rng = np.random.default_rng(0)
    out = {"spread": [], "plain": [], "scaled": [], "grad_plain": [], "grad_scaled": []}
    for d_k in SIZES:
        q = rng.normal(size=(ROWS, d_k))
        k = rng.normal(size=(ROWS, KEYS, d_k))
        s = np.einsum("nd,njd->nj", q, k)
        out["spread"].append(float(s.std()))
        for key, z in (("plain", s), ("scaled", s / math.sqrt(d_k))):
            z = z - z.max(axis=1, keepdims=True)
            a = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
            out[key].append(float(a.max(axis=1).mean()))
            jac = np.einsum("ni,ij->nij", a, np.eye(KEYS)) - np.einsum("ni,nj->nij", a, a)
            out["grad_" + key].append(float(np.linalg.norm(jac, axis=(1, 2)).mean()))
    return out


def sharpness_picture(labels: dict) -> str:
    """The mean largest attention weight of a row of 16 keys, against the key length, with and without scaling."""
    m = sharpness()
    panel = chart.Panel((3, 1400), (0, 1), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"], xticks=SIZES, xlog=True)
    panel.line(SIZES, m["plain"], "hidden")
    panel.points(SIZES, m["plain"], "hidden")
    panel.line(SIZES, m["scaled"], "hidden", dashed=True)
    panel.points(SIZES, m["scaled"], "hidden", hollow=True)
    panel.text(16, m["plain"][1] - 0.14, labels["plain"], "hidden", "start")
    panel.text(1024, m["scaled"][-1] + 0.1, labels["scaled"], "hidden", "end")
    panel.text(4, 1 / KEYS + 0.06, labels["even"], "ink-2", "start")
    panel.line([3, 1400], [1 / KEYS, 1 / KEYS], "ink-3", 1, dashed=True)
    svg = chart.figure([panel], alt=labels["alt"], width=330, height=250)
    return chart.with_decimal_comma(svg) if labels.get("decimal") == "," else svg


# The lesson is done when these are correct. "divide-too-much" is the choice question in the lesson.
REQUIRED = ["divide-too-much", "score-spread", "scaled-attention"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _grid(v):
    """A value for the messages: rounded to 4 decimals, as nested lists, whole numbers without .0."""
    try:
        a = np.asarray(v.detach() if isinstance(v, torch.Tensor) else v, dtype=float)
    except (TypeError, ValueError, AttributeError):
        return repr(v)
    if np.all(np.isfinite(a)) and np.all(a == np.round(a)):
        return repr(a.astype(int).tolist())
    return repr(np.round(a, 4).tolist())


def _same(got, want) -> bool:
    try:
        g = np.asarray(got.detach() if isinstance(got, torch.Tensor) else got, dtype=float)
        w = np.asarray(want.detach() if isinstance(want, torch.Tensor) else want, dtype=float)
    except (TypeError, ValueError, AttributeError):
        return False
    return g.shape == w.shape and bool(np.allclose(g, w, rtol=1e-4, atol=1e-4))


# --- exercise 2: score_spread ---------------------------------------------------------------------------------------------

_P_HEAD = ("import numpy as np\n\n\ndef score_spread(d_k):\n    rng = np.random.default_rng(0)\n"
           "    # 10,000 queries and 10,000 keys of d_k numbers each, with mean 0 and spread 1\n"
           "    q = rng.normal(size=(10000, d_k))\n    k = rng.normal(size=(10000, d_k))\n")
_P_START = (_P_HEAD + "    # the dot product of every query with the key in the same row\n    scores = ...\n"
            "    # how far the scores spread: their standard deviation\n    return ...\n")


def _p(scores: str = "np.sum(q * k, axis=1)", out: str = "np.std(scores)") -> str:
    return _P_HEAD + f"    scores = {scores}\n    return {out}\n"


# --- exercise 3: scaled_attention -----------------------------------------------------------------------------------------

_A_HEAD = ("import math\nimport torch\n\n\ndef scaled_attention(Q, K, V):\n"
           "    Q = torch.tensor(Q, dtype=torch.float32)\n    K = torch.tensor(K, dtype=torch.float32)\n    V = torch.tensor(V, dtype=torch.float32)\n"
           "    d_k = Q.shape[1]\n")
_A_START = (_A_HEAD + "    # the scores of every query with every key, divided by the square root of the key length\n    S = ...\n"
            "    # the weights and the new vectors\n    A = torch.softmax(S, dim=1)\n    return A @ V\n")


def _a(scores: str = "Q @ K.T / math.sqrt(d_k)", tail: str = "    A = torch.softmax(S, dim=1)\n    return A @ V\n") -> str:
    return _A_HEAD + f"    S = {scores}\n" + tail


_Q1 = [[2.0, 0.0, 2.0, 0.0], [0.0, 2.0, 0.0, 2.0], [1.0, 1.0, 1.0, 1.0]]
_V1 = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

EXERCISES = {
    "score-spread": FunctionBox(
        name="score-spread",
        function="score_spread",
        start=_P_START,
        solution=_p(),
        visible=[(64,)],
        hidden=[(1,), (16,), (256,)],
        compare=lambda got, want: _same(got, want),
        show=_grid,
        first=[_DOTS],
        mistakes=[
            Mistake("variance", wrong=_p(out="np.var(scores)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the variance, the spread squared. The spread of the scores is their standard deviation: `np.std(scores)`, about the square root of `d_k`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is de variantie, de spreiding in het kwadraat. De spreiding van de scores is hun standaardafwijking: `np.std(scores)`, ongeveer de wortel uit `d_k`."),
            Mistake("mean", wrong=_p(out="np.mean(scores)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the mean of the scores, which stays near 0, as many are negative. How far they spread is the standard deviation: `np.std(scores)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is het gemiddelde van de scores, dat dicht bij 0 blijft, omdat er veel negatief zijn. Hoe ver ze uiteenlopen, is de standaardafwijking: `np.std(scores)`."),
            Mistake("one-number", wrong=_p(scores="q * k"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you measure the spread of the separate products, not of the dot products. A dot product adds up the products of a row: `np.sum(q * k, axis=1)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je meet de spreiding van de losse producten, niet van de inproducten. Een inproduct telt de producten van een rij op: `np.sum(q * k, axis=1)`."),
        ],
        texts={
            **_TEXTS,
            "slow": {"en": "Your function is still not done after 2 seconds. Does it use `q @ k.T`? That is every query with every key, 100 million dot products, where you need only 10,000: each query with the key in its own row. Multiply number by number, and add up every row.",
                     "nl": "Je functie is na 2 seconden nog niet klaar. Gebruikt hij `q @ k.T`? Dat is elke query met elke key, 100 miljoen inproducten, terwijl je er maar 10.000 nodig hebt: elke query met de key in zijn eigen rij. Vermenigvuldig getal voor getal, en tel elke rij op."},
            "empty": {"en": "Replace the two `...`: the dot product of every query with the key in its row, and the standard deviation of those scores.",
                      "nl": "Vervang de twee `...`: het inproduct van elke query met de key in zijn rij, en de standaardafwijking van die scores."},
            "arguments": {"en": "The check calls `{call}`, with one number, and the `def` line of your function has another number of names between its parentheses. Keep `(d_k)`.",
                          "nl": "De check roept `{call}` aan, met één getal, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(d_k)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. One dot product per row, `axis=1`, and then the standard deviation of all of them?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Eén inproduct per rij, `axis=1`, en dan de standaardafwijking van allemaal?"},
            "almost": {"en": "Right for 64. But `{call}` gives `{got}`, and it should give `{want}`. Does your function use `d_k`, and no fixed numbers?",
                       "nl": "Goed voor 64. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gebruikt je functie `d_k`, en geen vaste getallen?"},
            "correct": {"en": "Your measurement shows the rule: the scores spread about √d_k, 1 for 1 number, 4 for 16, 8 for 64 and 16 for 256. Four times as many numbers, twice the spread.",
                        "nl": "Je meting laat de regel zien: de scores spreiden ongeveer √d_k, 1 bij 1 getal, 4 bij 16, 8 bij 64 en 16 bij 256. Vier keer zoveel getallen, twee keer de spreiding."},
        },
    ),
    "scaled-attention": FunctionBox(
        name="scaled-attention",
        function="scaled_attention",
        start=_A_START,
        solution=_a(),
        visible=[(_Q1, _Q1, _V1)],
        hidden=[([[1.0, 2.0], [3.0, 0.0]], [[0.0, 1.0], [2.0, 2.0], [1.0, -1.0]], [[1.0], [2.0], [3.0]]),
                ([[4.0] * 16], [[1.0] * 16, [0.0] * 16], [[1.0, 0.0], [0.0, 1.0]]),
                ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0], [0.0, 0.0, 9.0]], [[2.0, 2.0], [4.0, 0.0]])],
        compare=_same,
        show=_grid,
        first=[_DOTS],
        mistakes=[
            Mistake("sizes", error=lambda e: isinstance(e, RuntimeError) and "cannot be multiplied" in str(e),
                    en="The sizes of a matrix multiplication do not fit. `Q` and `K` have one row per word, so the score of every query with every key is `Q @ K.T`: (n, d_k) times (d_k, n).",
                    nl="De maten van een matrixvermenigvuldiging passen niet. `Q` en `K` hebben één rij per woord, dus de score van elke query met elke key is `Q @ K.T`: (n, d_k) keer (d_k, n)."),
            Mistake("by-d", wrong=_a(scores="Q @ K.T / d_k"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you divide by `d_k`, which makes the scores too small, so the weights come out flatter than they should be. Divide by its square root: `math.sqrt(d_k)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je deelt door `d_k`, waardoor de scores te klein worden, dus de weights worden vlakker dan ze moeten zijn. Deel door de wortel ervan: `math.sqrt(d_k)`."),
            Mistake("not-scaled", wrong=_a(scores="Q @ K.T"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is what the scores give when they are not scaled, with weights that are too sharp. Divide the scores by `math.sqrt(d_k)` before the softmax.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat geven de scores als ze niet geschaald zijn, met weights die te scherp zijn. Deel de scores door `math.sqrt(d_k)` vóór de softmax."),
            Mistake("words", wrong=_a(scores="Q @ K.T / math.sqrt(len(K))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you divide by the square root of the number of keys, `len(K)`. What makes the scores grow is the length of a key, `d_k`: the number of products in one dot product.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je deelt door de wortel uit het aantal keys, `len(K)`. Wat de scores laat groeien, is de lengte van een key, `d_k`: het aantal producten in één inproduct."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: the scores of every query with every key, divided by the square root of `d_k`.",
                      "nl": "Vervang de `...`: de scores van elke query met elke key, gedeeld door de wortel uit `d_k`."},
            "arguments": {"en": "The check calls `{call}`, with queries, keys and values, and the `def` line of your function has another number of names between its parentheses. Keep `(Q, K, V)`.",
                          "nl": "De check roept `{call}` aan, met queries, keys en values, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(Q, K, V)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your line with step 2 of the recipe.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je regel met stap 2 van het recept."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function use `d_k`, and no fixed numbers?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gebruikt je functie `d_k`, en geen vaste getallen?"},
            "correct": {"en": "Your function is scaled dot-product attention, the attention of the transformer: with keys of 16 numbers, every score is divided by 4, and with keys of 64 numbers, as in GPT-2, by 8.",
                        "nl": "Je functie is scaled dot-product attention, de attention van de transformer: met keys van 16 getallen wordt elke score door 4 gedeeld, en met keys van 64 getallen, zoals in GPT-2, door 8."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("score-spread", _P_START, "empty", None),
    ("score-spread", _p(scores="(q * k).sum(axis=1)", out="scores.std()"), "correct", None),
    ("score-spread", _p(out="np.var(scores)"), "wrong", "variance"),
    ("score-spread", _p(out="np.mean(scores)"), "wrong", "mean"),
    ("score-spread", _p(scores="q * k"), "wrong", "one-number"),
    ("score-spread", _p(out="np.std(scores) if d_k == 64 else 8.0"), "almost", None),
    ("scaled-attention", _A_START, "empty", None),
    ("scaled-attention", _a(scores="(Q @ K.T) / d_k ** 0.5"), "correct", None),
    ("scaled-attention", _a(scores="Q @ K.T / d_k"), "wrong", "by-d"),
    ("scaled-attention", _a(scores="Q @ K.T"), "wrong", "not-scaled"),
    ("scaled-attention", _a(scores="Q @ K.T / math.sqrt(len(K))"), "wrong", "words"),
    ("scaled-attention", _a(scores="Q @ K / math.sqrt(d_k)"), "error", "sizes"),
]
