# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson attn-scores: softmax_rows, the softmax of every row of a matrix of scores, without overflow;
and attend_all, the new vectors of all words at once, from the attention weights of the dot products."""
import re

import numpy as np

from everything_ai import draw
from everything_ai.check import FunctionBox, Mistake

# The made-up embeddings of attn-idea: how much a word is about money, and how much about water.
WORDS = {"the": [0.5, 0.5], "river": [0.0, 2.0], "bank": [1.0, 1.0], "account": [2.0, 0.0]}


def attention_weights(X) -> np.ndarray:
    S = np.asarray(X, dtype=float) @ np.asarray(X, dtype=float).T
    S = S - S.max(axis=1, keepdims=True)
    return np.exp(S) / np.sum(np.exp(S), axis=1, keepdims=True)


def weights_picture(labels: dict):
    """The attention weights of "the river bank", one row per word."""
    sentence = ["the", "river", "bank"]
    A = attention_weights([WORDS[w] for w in sentence])
    return draw.weight_grid(sentence, A, row_label=labels["rows"], col_label=labels["cols"], decimal=labels["decimal"],
                            title=labels["title"], alt=labels["alt"])


# The lesson is done when these are correct. "symmetric" is the choice question in the lesson.
REQUIRED = ["symmetric", "softmax-rows", "self-attention"]

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
    """An array for the messages: nested lists rounded to 4 decimals, nan as nan, whole numbers without .0."""
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return repr(v)
    if np.all(np.isfinite(a)) and np.all(a == np.round(a)):
        return repr(a.astype(int).tolist())
    return repr((np.round(a, 4) + 0.0).tolist())


def _same_grid(got, want) -> bool:
    try:
        g, w = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    except (TypeError, ValueError):
        return False
    return g.shape == w.shape and bool(np.allclose(g, w, rtol=1e-4, atol=1e-4, equal_nan=True))


def _has_nan(got, text) -> bool:
    try:
        return bool(np.isnan(np.asarray(got, dtype=float)).any())
    except (TypeError, ValueError):
        return False


# --- exercise 2: softmax_rows ---------------------------------------------------------------------------------------------

_S_HEAD = "import numpy as np\n\n\ndef softmax_rows(S):\n    S = np.array(S, dtype=float)\n"
_S_SHIFT = "    S = S - S.max(axis=1, keepdims=True)\n"
_S_OUT = "    return np.exp(S) / np.sum(np.exp(S), axis=1, keepdims=True)\n"
_S_START = (_S_HEAD + "    # 2. the largest number of every row subtracted, so that no exp overflows\n    S = ...\n"
            "    # 3. e to the power of every number, divided by the sum of its row\n    return ...\n")


def _s(shift: str = _S_SHIFT, out: str = _S_OUT) -> str:
    return _S_HEAD + shift + out


# --- exercise 3: attend_all -------------------------------------------------------------------------------------------

_A_HEAD = "import numpy as np\n\n\ndef attend_all(X):\n    X = np.array(X, dtype=float)\n"
_A_MID = ("    # 2. and 3. the attention weights: the softmax of every row\n    S = S - S.max(axis=1, keepdims=True)\n"
          "    A = np.exp(S) / np.sum(np.exp(S), axis=1, keepdims=True)\n")
_A_START = (_A_HEAD + "    # 1. the score of every word with every word\n    S = ...\n" + _A_MID
            + "    # 4. the new vectors: every word takes from every word, with the weights of its row\n    return ...\n")


def _a(scores: str = "X @ X.T", out: str = "A @ X") -> str:
    return _A_HEAD + f"    S = {scores}\n" + _A_MID + f"    return {out}\n"


EXERCISES = {
    "softmax-rows": FunctionBox(
        name="softmax-rows",
        function="softmax_rows",
        start=_S_START,
        solution=_s(),
        visible=[([[1, 2, 3], [1000, 1000, 1001]],)],
        hidden=[([[0.5, 1, 1], [1, 4, 2], [1, 2, 2]],), ([[5]],), ([[-1000, 0], [3, 3]],), ([[200, 400, 400], [400, 1600, 800], [400, 800, 800]],),
                ([[1, 2], [3, 4], [5, 6]],), ([[-1000, 1000]],)],
        compare=_same_grid,
        show=_grid,
        first=[_DOTS],
        mistakes=[
            Mistake("global-max", wrong=_s(shift="    S = S - S.max()\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you subtract the largest number of the whole matrix. A row far below it then gets e to the power of about −1,000 everywhere, which is 0, and 0 divided by 0 is nan. Subtract the largest number of each row: `S.max(axis=1, keepdims=True)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je trekt het grootste getal van de hele matrix af. Een rij die daar ver onder ligt, krijgt dan overal e tot de macht ongeveer −1.000, wat 0 is, en 0 gedeeld door 0 is nan. Trek het grootste getal van elke rij af: `S.max(axis=1, keepdims=True)`."),
            Mistake("column-max", wrong=_s(shift="    S = S - S.max(axis=0, keepdims=True)\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you subtract the largest number of every column, not of every row: `axis=0` goes down the columns. A row far below the largest number of a column then gets e to the power of about −1,000 everywhere, which is 0, and 0 divided by 0 is nan. Subtract the largest number of each row: `S.max(axis=1, keepdims=True)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je trekt het grootste getal van elke kolom af, niet van elke rij: `axis=0` gaat langs de kolommen. Een rij die ver onder het grootste getal van een kolom ligt, krijgt dan overal e tot de macht ongeveer −1.000, wat 0 is, en 0 gedeeld door 0 is nan. Trek het grootste getal van elke rij af: `S.max(axis=1, keepdims=True)`."),
            Mistake("overflow", matches=_has_nan,
                    en="`{call}` gives `{got}`, and it should give `{want}`: a row with large numbers gives nan. e to the power of 1,000 is inf, and inf divided by inf is nan. Subtract the largest number of every row first: the softmax stays the same, and the largest exp is then e to the power of 0, which is 1.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: een rij met grote getallen geeft nan. e tot de macht 1.000 is inf, en inf gedeeld door inf is nan. Trek eerst het grootste getal van elke rij af: de softmax blijft hetzelfde, en de grootste exp is dan e tot de macht 0, dus 1."),
            Mistake("whole", wrong=_s(out="    return np.exp(S) / np.sum(np.exp(S))\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your weights add up to 1 over the whole matrix, not per row. Sum per row: `np.sum(..., axis=1, keepdims=True)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je weights tellen op tot 1 over de hele matrix, niet per rij. Tel per rij op: `np.sum(..., axis=1, keepdims=True)`."),
            Mistake("columns", wrong=_s(out="    return np.exp(S) / np.sum(np.exp(S), axis=0, keepdims=True)\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: every column of yours adds up to 1, not every row. `axis=0` sums down the columns; `axis=1` sums along each row.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: bij jou telt elke kolom op tot 1, niet elke rij. `axis=0` telt elke kolom van boven naar beneden op; `axis=1` telt langs elke rij."),
            Mistake("no-keepdims", error=lambda e: isinstance(e, ValueError) and "could not be broadcast" in str(e),
                    en="NumPy cannot line up your arrays. `S.max(axis=1)` and `np.sum(..., axis=1)` give one number per row in a flat array, which NumPy lines up with the columns of `S`. With `keepdims=True`, they stay a column, one number per row, so every row gets its own.",
                    nl="NumPy kan je arrays niet tegen elkaar uitzetten (broadcasting). `S.max(axis=1)` en `np.sum(..., axis=1)` geven één getal per rij in een platte array, die NumPy langs de kolommen van `S` legt. Met `keepdims=True` blijven ze een kolom, één getal per rij, zodat elke rij zijn eigen getal krijgt."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the largest number of every row subtracted, then e to the power of every number, divided by the sum of its row.",
                      "nl": "Vervang de twee `...`: het grootste getal van elke rij eraf, dan e tot de macht van elk getal, gedeeld door de som van zijn rij."},
            "arguments": {"en": "The check calls `{call}`, with one matrix, and the `def` line of your function has another number of names between its parentheses. Keep `(S)`.",
                          "nl": "De check roept `{call}` aan, met één matrix, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(S)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your lines with steps 2 and 3 of the recipe.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je regels met stap 2 en 3 van het recept."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for any number of rows and columns?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elk aantal rijen en kolommen?"},
            "correct": {"en": "Your softmax gives every row weights that are positive and add up to 1, also for scores of a thousand: a row of 1,000, 1,000 and 1,001 gives the same weights as 0, 0 and 1.",
                        "nl": "Je softmax geeft elke rij weights die positief zijn en optellen tot 1, ook bij scores van duizend: een rij van 1.000, 1.000 en 1.001 geeft dezelfde weights als 0, 0 en 1."},
        },
    ),
    "self-attention": FunctionBox(
        name="self-attention",
        function="attend_all",
        start=_A_START,
        solution=_a(),
        visible=[([[1, 0], [0, 1], [1, 1]],)],
        hidden=[([[0.5, 0.5], [0.0, 2.0], [1.0, 1.0]],), ([[1, 2, 0], [0, 1, 1]],), ([[3, 4]],),
                ([[10, 0], [0, 10], [10, 10], [5, 5]],), ([[1, 2], [2, 1]],)],
        compare=_same_grid,
        show=_grid,
        first=[_DOTS],
        mistakes=[
            Mistake("columns", syntax=r"^[ \t]*S = .*\bX\.T\s*@\s*X\b",
                    en="`{line}` compares the columns of `X`, the numbers of the embeddings, not the words. `X @ X.T` has a row and a column for every word: the dot product of every word with every word.",
                    nl="`{line}` vergelijkt de kolommen van `X`, de getallen van de embeddings, niet de woorden. `X @ X.T` heeft een rij en een kolom voor elk woord: het inproduct van elk woord met elk woord."),
            Mistake("weights", wrong=_a(out="A"),
                    en="`{call}` gives the attention weights, a matrix with a row and a column for every word. Give the new vectors: every word takes from every word with the weights of its row, `A @ X`, one row per word, as long as an embedding.",
                    nl="`{call}` geeft de attention weights, een matrix met een rij en een kolom voor elk woord. Geef de nieuwe vectoren: elk woord neemt van elk woord met de weights van zijn rij, `A @ X`, één rij zo lang als een embedding per woord."),
            Mistake("transposed", wrong=_a(out="A.T @ X"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: `A.T @ X` uses the columns of `A`. Word `i` takes from every word with the weights of row `i`, which add up to 1: `A @ X`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `A.T @ X` gebruikt de kolommen van `A`. Woord `i` neemt van elk woord met de weights van rij `i`, die optellen tot 1: `A @ X`."),
            Mistake("sizes", error=lambda e: isinstance(e, ValueError) and "matmul" in str(e),
                    en="The shapes of a matrix multiplication do not fit. `X` has shape (n, d), one row per word; the scores `X @ X.T` and the weights `A` have shape (n, n); and the new vectors `A @ X` have shape (n, d) again.",
                    nl="De shapes van een matrixvermenigvuldiging passen niet. `X` heeft shape (n, d), één rij per woord; de scores `X @ X.T` en de weights `A` hebben shape (n, n); en de nieuwe vectoren `A @ X` weer shape (n, d)."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the scores of every word with every word, and the new vectors.",
                      "nl": "Vervang de twee `...`: de scores van elk woord met elk woord, en de nieuwe vectoren."},
            "arguments": {"en": "The check calls `{call}`, with one matrix of embeddings, and the `def` line of your function has another number of names between its parentheses. Keep `(X)`.",
                          "nl": "De check roept `{call}` aan, met één matrix van embeddings, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(X)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your lines with steps 1 and 4 of the recipe.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je regels met stap 1 en 4 van het recept."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for any number of words and of numbers per word?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elk aantal woorden en getallen per woord?"},
            "correct": {"en": "Your function gives every word its new vector at once, with two matrix multiplications and a softmax. Row 0 of the example is what `attend` gave for word 0 in [Attention: every word looks at the others](/course/open/attn-idea?lang=en): the same calculation, for all words together.",
                        "nl": "Je functie geeft elk woord tegelijk zijn nieuwe vector, met twee matrixvermenigvuldigingen en een softmax. Rij 0 van het voorbeeld is wat `attend` voor woord 0 gaf in [Attention: elk woord kijkt naar de andere](/course/open/attn-idea?lang=nl): dezelfde berekening, voor alle woorden samen."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("softmax-rows", _S_START, "empty", None),
    ("softmax-rows", _s(shift="    S = S - np.max(S, axis=1, keepdims=True)\n",
                        out="    e = np.exp(S)\n    return e / e.sum(axis=1, keepdims=True)\n"), "correct", None),
    ("softmax-rows", _s(shift="    S = S - S.max()\n"), "wrong", "global-max"),
    ("softmax-rows", _s(shift="    S = S - S.max(axis=0, keepdims=True)\n"), "wrong", "column-max"),
    ("softmax-rows", _s(shift="    S = S - S.max(axis=0)\n"), "wrong", "column-max"),
    ("softmax-rows", _s(shift="    S = S - S.mean(axis=1, keepdims=True)\n"), "wrong", "overflow"),
    ("softmax-rows", _s(shift=""), "wrong", "overflow"),
    ("softmax-rows", _s(out="    return np.exp(S) / np.sum(np.exp(S))\n"), "wrong", "whole"),
    ("softmax-rows", _s(out="    return np.exp(S) / np.sum(np.exp(S), axis=0, keepdims=True)\n"), "wrong", "columns"),
    ("softmax-rows", _s(shift="    S = S - S.max(axis=1)\n"), "error", "no-keepdims"),
    ("self-attention", _A_START, "empty", None),
    ("self-attention", _a(scores="np.dot(X, X.T)", out="np.dot(A, X)"), "correct", None),
    ("self-attention", _a(scores="X.T @ X"), "error", "columns"),
    ("self-attention", _a(out="A"), "wrong", "weights"),
    ("self-attention", _a(out="A.T @ X"), "wrong", "transposed"),
    ("self-attention", _a(out="X @ A"), "error", "sizes"),
]
