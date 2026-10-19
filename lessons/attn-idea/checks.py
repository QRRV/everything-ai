# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson attn-idea: attend, the new vector of one word after attention: the softmax of its dot
products with every word as weights, and the weighted average of all embeddings. And the picture: the made-up
embeddings of "the", "river", "bank" and "account", and the new vector of "bank" in "the river bank" and in "the bank
account"."""

import math
import re

import numpy as np

from everything_ai.check import FunctionBox, Mistake

# Embeddings the course made up for the lesson: 2 numbers each, about money and about water.
WORDS = {"the": (0.5, 0.5), "river": (0.0, 2.0), "bank": (1.0, 1.0), "account": (2.0, 0.0)}


def attend_rows(X: np.ndarray) -> np.ndarray:
    """Every row's new vector: the softmax of its dot products with all rows, times the rows."""
    scores = X @ X.T
    weights = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    return weights @ X


def new_bank(sentence: tuple[str, ...]) -> tuple[float, float]:
    X = np.array([WORDS[w] for w in sentence])
    return tuple(float(v) for v in attend_rows(X)[sentence.index("bank")])


def bank_picture(labels: dict) -> str:
    """Arrows from the origin: the made-up embeddings, and the new vector of "bank" in two sentences."""
    from everything_ai import chart
    panel = chart.Panel((-0.2, 2.8), (-0.2, 2.8), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    for word in ("river", "bank", "account"):
        a, b = WORDS[word]
        panel.arrow(0, 0, a, b, "hidden", 1.5)
    panel.text(0.12, 2.12, labels["river"], "hidden", "start")
    panel.text(1.05, 1.08, labels["bank"], "hidden", "start")
    panel.text(2.0, 0.12, labels["account"], "hidden", "middle")
    for key, sentence, (dx, dy, anchor) in (("in_river", ("the", "river", "bank"), (0.08, 0.06, "start")),
                                            ("in_account", ("the", "bank", "account"), (0.1, 0.02, "start"))):
        a, b = new_bank(sentence)
        panel.arrow(0, 0, a, b, "hidden", 2.8)
        panel.text(a + dx, b + dy, labels[key], "hidden", anchor)
    return chart.figure([panel], alt=labels["alt"], width=330, height=330)


# The lesson is done when these are correct. "weights-sum" and "transformer" are the choice questions in the lesson.
REQUIRED = ["weights-sum", "attend", "transformer"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _num(x) -> str:
    x = float(x)
    if not math.isfinite(x):
        return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")
    if abs(x) < 1e-12:
        return "0"
    return np.format_float_positional(x, precision=4, unique=False, fractional=False, trim="-").rstrip(".")


def _nested(v) -> str:
    if np.ndim(v) == 0:
        return _num(v)
    return "[" + ", ".join(_nested(x) for x in v) + "]"


class _Array(np.ndarray):
    """Embeddings that a check message writes the way the lesson's code does: np.array([...]). The learner's code
    gets a plain copy."""

    def __repr__(self):
        return f"np.array({_nested(np.asarray(self).tolist())})"

    def __deepcopy__(self, memo):
        return np.array(self)


def _a(values) -> np.ndarray:
    return np.array(values, dtype=float).view(_Array)


def _show(v) -> str:
    try:
        return "np.array(" + _nested(np.asarray(v, dtype=float).tolist()) + ")"
    except (TypeError, ValueError):
        return repr(v)


def _close(got, want) -> bool:
    try:
        g, w = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    except (TypeError, ValueError):
        return False
    return g.shape == w.shape and bool(np.allclose(g, w, rtol=0, atol=1e-4))   # the task shows 4 decimals


# --- exercise 2: the new vector of one word --------------------------------------------------------------------------------

_A_HEAD = "import numpy as np\n\n\ndef attend(X, i):\n"
_A_START = (_A_HEAD + "    # 1. the score of word i with every word: the dot products\n    scores = ...\n"
            "    # 2. the weights: the softmax of the scores\n    weights = ...\n"
            "    # 3. the new vector: the weighted average of all rows of X\n    return ...\n")


def _f(scores: str = "X @ X[i]", weights: str = "np.exp(scores) / np.sum(np.exp(scores))", out: str = "weights @ X") -> str:
    return _A_HEAD + f"    scores = {scores}\n    weights = {weights}\n    return {out}\n"


_VISIBLE = (_a([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), 0)
_HIDDEN = [
    (_a([[0.5, 0.5], [1.0, 1.0], [2.0, 0.0]]), 1),   # "bank" in "the bank account"
    (_a([[0.5, 0.5], [1.0, 1.0], [2.0, 0.0]]), 2),
    (_a([[0.5, 0.5], [1.0, 1.0], [2.0, 0.0]]), 0),
    (_a([[1.0, 0.0], [0.0, 1.0]]), 0),
    (_a([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0], [2.0, 0.0, 1.0], [0.5, 0.5, 0.5]]), 3),
    (_a([[3.0, 1.0]]), 0),   # one word: it can only look at itself
]

EXERCISES = {
    "attend": FunctionBox(
        name="attend",
        function="attend",
        start=_A_START,
        solution=_f(),
        visible=[_VISIBLE],
        hidden=_HIDDEN,
        show=_show,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("no-softmax", wrong=_f(weights="scores"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you use the scores themselves as weights, so they do not add up to 1 and the new vector is not an average. Turn them into weights with the softmax: `np.exp(scores) / np.sum(np.exp(scores))`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je gebruikt de scores zelf als weights, dus ze tellen niet op tot 1 en de nieuwe vector is geen gemiddelde. Maak er weights van met de softmax: `np.exp(scores) / np.sum(np.exp(scores))`."),
            Mistake("no-divide", wrong=_f(weights="np.exp(scores)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your weights are `np.exp(scores)` without dividing by their sum, so they do not add up to 1. Divide by `np.sum(np.exp(scores))`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je weights zijn `np.exp(scores)` zonder te delen door hun som, dus ze tellen niet op tot 1. Deel door `np.sum(np.exp(scores))`."),
            Mistake("weights-only", wrong=_f(out="weights"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you give back the weights, one for each word, not the new vector. Step 3: the weighted average of the rows of `X`, `weights @ X`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je geeft de weights terug, één per woord, niet de nieuwe vector. Stap 3: het gewogen gemiddelde van de rijen van `X`, `weights @ X`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the three `...`: the dot products of row `i` with every row, their softmax, and the weighted average of the rows of `X`.",
                      "nl": "Vervang de drie `...`: de inproducten van rij `i` met elke rij, hun softmax, en het gewogen gemiddelde van de rijen van `X`."},
            "arguments": {"en": "The check calls `{call}`, with an array and a row number, and the `def` line of your function has another number of names between its parentheses. Keep `(X, i)`.",
                          "nl": "De check roept `{call}` aan, met een array en een rijnummer, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(X, i)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Are the scores the dot products of row `i` with every row of `X`, row `i` itself included, are the weights their softmax, and is the new vector the weighted average of the rows?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Zijn de scores de inproducten van rij `i` met elke rij van `X`, rij `i` zelf ook, zijn de weights hun softmax, en is de nieuwe vector het gewogen gemiddelde van de rijen?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for every row `i`, every number of words and every number of numbers per word?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elke rij `i`, elk aantal woorden en elk aantal getallen per woord?"},
            "correct": {"en": "Your function gives the new vector of a word in all {n} cases: a weighted average of all embeddings, with the softmax of the dot products as weights.",
                        "nl": "Je functie geeft in alle {n} gevallen de nieuwe vector van een woord: een gewogen gemiddelde van alle embeddings, met de softmax van de inproducten als weights."},
        },
    ),
}

CASES = [
    ("attend", _A_START, "empty", None),
    ("attend", _f(), "correct", None),
    ("attend", _A_HEAD + "    scores = np.array([np.dot(X[i], X[j]) for j in range(len(X))])\n    e = np.exp(scores)\n"
               "    weights = e / e.sum()\n    out = np.zeros(X.shape[1])\n    for j in range(len(X)):\n        out = out + weights[j] * X[j]\n    return out\n", "correct", None),
    ("attend", _f(weights="scores"), "wrong", "no-softmax"),
    ("attend", _f(weights="np.exp(scores)"), "wrong", "no-divide"),
    ("attend", _f(out="weights"), "wrong", "weights-only"),
    ("attend", _f(scores="X[i] @ X[i]"), "error", None),
    ("attend", "def attend(X):\n    return X\n", "error", "arguments"),
]
