# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson lm-model-families: can_see, which tokens a token may look at with attention in an encoder
and in a decoder; and the picture of who looks at whom."""
import re

from everything_ai import draw
from everything_ai.check import FunctionBox, Mistake, same_value


def sees(kind: str, i: int, j: int) -> bool:
    """The course's answer: in an encoder every token looks at all tokens, in a decoder only at itself and earlier."""
    return True if kind == "encoder" else j <= i


def look_picture(labels: dict) -> str:
    """Who looks at whom, in an encoder and in a decoder, for the five tokens in `labels["tokens"]`."""
    tokens = labels["tokens"]
    n = len(tokens)
    panels = [(labels[kind], [[sees(kind, i, j) for j in range(n)] for i in range(n)]) for kind in ("encoder", "decoder")]
    return draw.look_grids(panels, tokens, caption=labels["caption"], title=labels["title"], alt=labels["alt"]).text


# The lesson is done when these are correct. "family" and "translate" are choice questions in the lesson.
REQUIRED = ["family", "can-see", "translate"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_START = "def can_see(kind, i, j):\n    if kind == \"encoder\":\n        return ...\n    return ...\n"


def _c(encoder: str = "True", decoder: str = "j <= i") -> str:
    return f"def can_see(kind, i, j):\n    if kind == \"encoder\":\n        return {encoder}\n    return {decoder}\n"


_C = _c()

EXERCISES = {
    "can-see": FunctionBox(
        name="can-see",
        function="can_see",
        start=_START,
        solution=_C,
        visible=[("decoder", 1, 3)],
        hidden=[("decoder", 3, 1), ("decoder", 2, 2), ("decoder", 2, 3), ("encoder", 1, 3), ("encoder", 4, 0), ("encoder", 2, 2), ("decoder", 0, 4), ("decoder", 4, 0)],
        compare=same_value,
        first=[_DOTS],
        mistakes=[
            Mistake("forward", wrong=_c(decoder="j >= i"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your decoder looks at itself and the tokens after it. When a decoder writes, the tokens after it are not there yet: it looks back, at places 0 up to and including `i`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je decoder kijkt naar zichzelf en de tokens erna. Als een decoder schrijft, zijn de tokens erna er nog niet: hij kijkt terug, naar de plekken 0 tot en met `i`."),
            Mistake("not-itself", wrong=_c(decoder="j < i"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your decoder does not look at its own token. A token may always look at itself, also in a decoder: `j` may be equal to `i`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je decoder kijkt niet naar zijn eigen token. Een token mag altijd naar zichzelf kijken, ook in een decoder: `j` mag gelijk zijn aan `i`."),
            Mistake("next-token", state="almost", wrong=_c(decoder="j <= i + 1"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your decoder also looks at the token just after it. That is the token it must predict, so it could simply copy the answer. A decoder looks at places 0 up to and including `i`, not `i + 1`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je decoder kijkt ook naar het token net erna. Dat is het token dat hij moet voorspellen, dus zo kan hij het antwoord gewoon afkijken. Een decoder kijkt naar de plekken 0 tot en met `i`, niet `i + 1`."),
            Mistake("encoder-self", wrong=_c(encoder="i == j"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your encoder looks only at its own token. In an encoder, every token looks at all tokens, before and after it.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je encoder kijkt alleen naar zijn eigen token. In een encoder kijkt elk token naar alle tokens, ervoor en erna."),
            Mistake("decoder-all", wrong=_c(decoder="True"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your decoder looks at all tokens, as an encoder does. A decoder looks only at itself and the tokens before it.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je decoder kijkt naar alle tokens, zoals een encoder. Een decoder kijkt alleen naar zichzelf en de tokens ervoor."),
            Mistake("numbers", matches=lambda got, text: type(got) is int,
                    en="`{call}` gives `{got}`, a number, and it should give `{want}`. Let your function give `True` or `False`: a comparison gives one of the two itself.",
                    nl="`{call}` geeft `{got}`, een getal, en dat moet `{want}` zijn. Laat je functie `True` of `False` geven: een vergelijking geeft zelf een van de twee."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "empty": {"en": "Replace the two `...`: whether a token of an encoder may look at token `j`, and whether a token of a decoder may.",
                      "nl": "Vervang de twee `...`: of een token van een encoder naar token `j` mag kijken, en of een token van een decoder dat mag."},
            "arguments": {"en": "The check calls `{call}`, with three values, and the `def` line of your function has another number of names between its parentheses. Keep `(kind, i, j)`.",
                          "nl": "De check roept `{call}` aan, met drie waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(kind, i, j)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your two lines with steps 1 and 2 of the recipe.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je twee regels met stap 1 en 2 van het recept."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for both kinds, and for every `i` and `j`?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor beide soorten, en voor elke `i` en `j`?"},
            "correct": {"en": "Your function draws the line between the two families: an encoder token looks at all tokens, a decoder token at itself and the tokens before it. In a decoder of 5 tokens, the first looks at 1 token and the last at all 5.",
                        "nl": "Je functie trekt de grens tussen de twee families: een token van een encoder kijkt naar alle tokens, een token van een decoder naar zichzelf en de tokens ervoor. In een decoder van 5 tokens kijkt het eerste naar 1 token en het laatste naar alle 5."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("can-see", _START, "empty", None),
    ("can-see", "def can_see(kind, i, j):\n    if kind == \"decoder\":\n        return i >= j\n    return True\n", "correct", None),
    ("can-see", _c(decoder="j >= i"), "wrong", "forward"),
    ("can-see", _c(decoder="j < i"), "wrong", "not-itself"),
    ("can-see", _c(encoder="i == j"), "wrong", "encoder-self"),
    ("can-see", _c(decoder="True"), "wrong", "decoder-all"),
    ("can-see", _c(decoder="j <= i + 1"), "almost", "next-token"),
    ("can-see", _c(decoder="j < i + 1"), "correct", None),
    ("can-see", "def can_see(kind, i, j):\n    if kind == \"encoder\":\n        return 1\n    if j <= i:\n        return 1\n    return 0\n", "wrong", "numbers"),
    ("can-see", _c(decoder="j == i"), "almost", None),
    ("can-see", _c(encoder="False"), "almost", None),
]
