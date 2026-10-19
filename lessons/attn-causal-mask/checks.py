# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson attn-causal-mask: future_mask, which positions lie in the future of each position; and
causal_attention, scaled dot-product attention in which every position only takes from itself and the positions
before it."""
import math
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai import draw
from everything_ai.check import FunctionBox, Mistake

# "the cat sat down" with made-up embeddings of 2 numbers, used as queries, keys and values at once.
SENTENCE = ["the", "cat", "sat", "down"]
X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5]]


def masked_weights() -> np.ndarray:
    x = torch.tensor(X)
    S = x @ x.T / math.sqrt(x.shape[1])
    future = torch.tril(torch.ones(len(x), len(x))) == 0
    return torch.softmax(S.masked_fill(future, float("-inf")), dim=1).numpy()


def mask_picture(labels: dict):
    """The attention weights of "the cat sat down" with the causal mask: zeros above the diagonal."""
    return draw.weight_grid(SENTENCE, masked_weights(), row_label=labels["rows"], col_label=labels["cols"],
                            decimal=labels["decimal"], title=labels["title"], alt=labels["alt"])


# The lesson is done when these are correct. "minus-infinity" is the choice question in the lesson.
REQUIRED = ["minus-infinity", "future-mask", "causal-attention"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _array(v):
    if isinstance(v, torch.Tensor):
        v = v.detach()
        return np.asarray(v.float() if v.dtype == torch.bool else v, dtype=float)
    return np.asarray(v, dtype=float)


def _grid(v):
    """A value for the messages: True and False as they are, numbers rounded to 4 decimals, as nested lists."""
    if isinstance(v, torch.Tensor) and v.dtype == torch.bool:
        return repr(v.tolist())
    try:
        a = _array(v)
    except (TypeError, ValueError):
        return repr(v)
    if np.all(np.isfinite(a)) and np.all(a == np.round(a)):
        return repr(a.astype(int).tolist())
    return repr(np.round(a, 4).tolist())


def _same(got, want) -> bool:
    try:
        g, w = _array(got), _array(want)
    except (TypeError, ValueError):
        return False
    return g.shape == w.shape and bool(np.allclose(g, w, rtol=1e-4, atol=1e-4, equal_nan=True))


def _is_bool(v) -> bool:
    """True and False, as masked_fill needs: a bool tensor, a NumPy bool array, or lists of True and False."""
    if isinstance(v, torch.Tensor):
        return v.dtype == torch.bool
    try:
        return np.asarray(v).dtype == bool
    except (TypeError, ValueError):
        return False


def _same_mask(got, want) -> bool:
    """The mask is right only as True and False: a mask of the numbers 0 and 1 makes masked_fill stop with an error."""
    return _is_bool(got) and _same(got, want)


def _numbers_right(got, text) -> bool:
    """The right places, but as the numbers 0 and 1 instead of True and False."""
    if _is_bool(got):
        return False
    try:
        a = _array(got)
    except (TypeError, ValueError):
        return False
    return a.ndim == 2 and a.shape[0] == a.shape[1] and bool(np.array_equal(a, np.triu(np.ones_like(a), 1)))


# --- exercise 2: future_mask ----------------------------------------------------------------------------------------------

_F_HEAD = "import torch\n\n\ndef future_mask(n):\n"
_F_START = (_F_HEAD + "    # True where column j lies after row i: the positions each position may not see\n    return ...\n")


def _f(body: str = "torch.tril(torch.ones(n, n)) == 0") -> str:
    return _F_HEAD + f"    return {body}\n"


# --- exercise 3: causal_attention -----------------------------------------------------------------------------------------

_A_HEAD = ("import math\nimport torch\n\n\ndef causal_attention(Q, K, V):\n"
           "    Q, K, V = torch.tensor(Q, dtype=torch.float32), torch.tensor(K, dtype=torch.float32), torch.tensor(V, dtype=torch.float32)\n"
           "    n, d_k = Q.shape\n    S = Q @ K.T / math.sqrt(d_k)\n")
_A_START = (_A_HEAD + "    # the scores of the future set to minus infinity\n    S = ...\n"
            "    # the weights and the new vectors\n    A = torch.softmax(S, dim=1)\n    return A @ V\n")
_A_MASK = "S.masked_fill(torch.tril(torch.ones(n, n)) == 0, float(\"-inf\"))"


def _a(mask: str = _A_MASK, tail: str = "    A = torch.softmax(S, dim=1)\n    return A @ V\n") -> str:
    return _A_HEAD + f"    S = {mask}\n" + tail


EXERCISES = {
    "future-mask": FunctionBox(
        name="future-mask",
        function="future_mask",
        start=_F_START,
        solution=_f(),
        visible=[(3,)],
        hidden=[(1,), (5,), (2,)],
        compare=_same_mask,
        show=_grid,
        first=[_DOTS],
        mistakes=[
            Mistake("past", wrong=_f("torch.tril(torch.ones(n, n)) == 1"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your True stands where a position may look, its past and itself. The mask marks the future: True above the diagonal, where column `j` comes after row `i`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: bij jou staat True waar een positie mag kijken, zijn verleden en zichzelf. De mask markeert de toekomst: True boven de diagonaal, waar kolom `j` na rij `i` komt."),
            Mistake("itself", wrong=_f("torch.tril(torch.ones(n, n), diagonal=-1) == 0"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your mask also hides every position from itself, on the diagonal. Then the first position sees nothing at all, and its softmax gives nan. Only the positions after it are the future: `True` only above the diagonal, as in step 2 of the recipe.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je mask verbergt ook elke positie voor zichzelf, op de diagonaal. Dan ziet de eerste positie helemaal niets, en geeft zijn softmax nan. Alleen de posities erna zijn de toekomst: `True` alleen boven de diagonaal, zoals in stap 2 van het recept."),
            Mistake("numbers", state="almost", matches=_numbers_right,
                    en="`{call}` gives `{got}`: the right places, but as the numbers 0 and 1, and it should give `{want}`. `masked_fill` only takes a mask of True and False. A comparison, such as `== 0` in step 2 of the recipe, turns numbers into True and False.",
                    nl="`{call}` geeft `{got}`: de goede plekken, maar als de getallen 0 en 1, en dat moet `{want}` zijn. `masked_fill` neemt alleen een mask van True en False. Een vergelijking, zoals `== 0` in stap 2 van het recept, maakt van getallen True en False."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: a matrix of n by n with True above the diagonal, and False on and below it.",
                      "nl": "Vervang de `...`: een matrix van n bij n met True boven de diagonaal, en False op en onder de diagonaal."},
            "arguments": {"en": "The check calls `{call}`, with one number, and the `def` line of your function has another number of names between its parentheses. Keep `(n)`.",
                          "nl": "De check roept `{call}` aan, met één getal, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(n)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your line with step 2 of the recipe.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je regel met stap 2 van het recept."},
            "almost": {"en": "Right for 3 positions. But `{call}` gives `{got}`, and it should give `{want}`. Does your function use `n`, and no fixed size?",
                       "nl": "Goed voor 3 posities. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gebruikt je functie `n`, en geen vaste grootte?"},
            "correct": {"en": "Your mask hides the future of every position: nothing for the last one, everything after itself for the first. Of the n × n places, n(n − 1)/2 are True: 10 for 5 positions.",
                        "nl": "Je mask verbergt de toekomst van elke positie: niets voor de laatste, alles na zichzelf voor de eerste. Van de n × n plekken zijn er n(n − 1)/2 True: 10 bij 5 posities."},
        },
    ),
    "causal-attention": FunctionBox(
        name="causal-attention",
        function="causal_attention",
        start=_A_START,
        solution=_a(),
        visible=[([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]], [[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])],
        hidden=[([[1.0, 2.0], [3.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [2.0, 2.0], [1.0, -1.0]], [[1.0], [2.0], [3.0]]),
                ([[2.0, 0.0, 1.0]], [[1.0, 1.0, 1.0]], [[5.0, 6.0]]),
                ([[1.0, 0.0], [0.0, 1.0]], [[10.0, 0.0], [0.0, 10.0]], [[1.0, 0.0], [0.0, 1.0]])],
        compare=_same,
        show=_grid,
        first=[_DOTS],
        mistakes=[
            Mistake("number-mask", error=lambda e: isinstance(e, RuntimeError) and "boolean" in str(e),
                    en="With `{call}`, `masked_fill` stops with an error: it only takes a mask of True and False, and yours has numbers. A comparison, such as `== 0` in step 2 of the recipe, turns numbers into True and False.",
                    nl="Met `{call}` stopt `masked_fill` met een fout: het neemt alleen een mask van True en False, en de jouwe heeft getallen. Een vergelijking, zoals `== 0` in stap 2 van het recept, maakt van getallen True en False."),
            Mistake("no-mask", wrong=_a(mask="S"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: every position still takes from the positions after it. Set their scores to minus infinity before the softmax, and keep the copy that `masked_fill` gives: `S = S.masked_fill(..., float(\"-inf\"))`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elke positie neemt nog van de posities erna. Zet hun scores op min oneindig vóór de softmax, en bewaar de kopie die `masked_fill` geeft: `S = S.masked_fill(..., float(\"-inf\"))`."),
            Mistake("zero", wrong=_a(mask="S.masked_fill(torch.tril(torch.ones(n, n)) == 0, 0.0)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you set the scores of the future to 0, and e to the power of 0 is 1, so the future still gets weight. With minus infinity, its weight is exactly 0.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je zet de scores van de toekomst op 0, en e tot de macht 0 is 1, dus de toekomst krijgt nog weight. Met min oneindig is zijn weight precies 0."),
            Mistake("after", wrong=_a(mask="S", tail="    A = torch.softmax(S, dim=1).masked_fill(torch.tril(torch.ones(n, n)) == 0, 0.0)\n    return A @ V\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you set the weights of the future to 0 after the softmax, so the weights of a row no longer add up to 1. Mask the scores before the softmax, with minus infinity: the softmax then shares the whole weight among the positions that remain.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je zet de weights van de toekomst op 0 na de softmax, dus de weights van een rij tellen niet meer op tot 1. Maskeer de scores vóór de softmax, met min oneindig: de softmax verdeelt dan de hele weight over de posities die overblijven."),
            Mistake("backward", wrong=_a(mask="S.masked_fill(torch.tril(torch.ones(n, n)) == 1, float(\"-inf\"))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you hide the past and the position itself, so every position looks only ahead, and the last one sees nothing. Hide the positions after it: `True` only above the diagonal, as in step 2 of the recipe.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je verbergt het verleden en de positie zelf, dus elke positie kijkt alleen vooruit, en de laatste ziet niets. Verberg de posities erna: `True` alleen boven de diagonaal, zoals in stap 2 van het recept."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: the scores with those of the future set to minus infinity.",
                      "nl": "Vervang de `...`: de scores, met die van de toekomst op min oneindig."},
            "arguments": {"en": "The check calls `{call}`, with queries, keys and values, and the `def` line of your function has another number of names between its parentheses. Keep `(Q, K, V)`.",
                          "nl": "De check roept `{call}` aan, met queries, keys en values, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(Q, K, V)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your line with steps 2 and 3 of the recipe.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je regel met stap 2 en 3 van het recept."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function use `n`, and no fixed size?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gebruikt je functie `n`, en geen vaste grootte?"},
            "correct": {"en": "Your attention is causal: the first position keeps its own value, and every later one averages over itself and what came before. This is the attention of GPT.",
                        "nl": "Je attention is causaal: de eerste positie houdt zijn eigen value, en elke latere middelt over zichzelf en wat ervoor kwam. Dit is de attention van GPT."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("future-mask", _F_START, "empty", None),
    ("future-mask", _f("torch.triu(torch.ones(n, n), diagonal=1) == 1"), "correct", None),
    ("future-mask", _f("torch.tril(torch.ones(n, n)) == 1"), "wrong", "past"),
    ("future-mask", _f("torch.tril(torch.ones(n, n), diagonal=-1) == 0"), "wrong", "itself"),
    ("future-mask", _f("torch.tril(torch.ones(3, 3)) == 0"), "almost", None),
    ("future-mask", _f("torch.triu(torch.ones(n, n), diagonal=1)"), "almost", "numbers"),
    ("future-mask", _f("1 - torch.tril(torch.ones(n, n))"), "almost", "numbers"),
    ("future-mask", _f("[[j > i for j in range(n)] for i in range(n)]"), "correct", None),
    ("causal-attention", _A_START, "empty", None),
    ("causal-attention", _a(mask="S.masked_fill(torch.triu(torch.ones(n, n), diagonal=1) == 1, -1e9)"), "correct", None),
    ("causal-attention", _a(mask="S"), "wrong", "no-mask"),
    ("causal-attention", _a(mask="S.masked_fill(torch.triu(torch.ones(n, n), diagonal=1), float(\"-inf\"))"), "error", "number-mask"),
    ("causal-attention", _a(mask="S.masked_fill(torch.tril(torch.ones(n, n)) == 0, 0.0)"), "wrong", "zero"),
    ("causal-attention", _a(mask="S", tail="    A = torch.softmax(S, dim=1).masked_fill(torch.tril(torch.ones(n, n)) == 0, 0.0)\n    return A @ V\n"), "wrong", "after"),
    ("causal-attention", _a(mask="S.masked_fill(torch.tril(torch.ones(n, n)) == 1, float(\"-inf\"))"), "wrong", "backward"),
]
