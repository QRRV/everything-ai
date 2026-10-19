# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson lm-perplexity: perplexity, e to the power of the average NLL of the probabilities a model gave
the tokens that really came; and char_perplexity, the perplexity per character of a model that counts in tokens.
The picture is the curve from average NLL to perplexity, with the counted bigram model of Tiny Shakespeare on it."""
import functools
import math
import re

import numpy as np

from everything_ai import paths
from everything_ai.check import FunctionBox, Mistake

DATA = "data/tiny-shakespeare.txt"   # about 1.1 MB of plays by Shakespeare: see data/README.md


@functools.lru_cache(maxsize=1)
def _bigram():
    """The counted bigram model of Tiny Shakespeare: P[i, j] is the probability that character j follows character i."""
    text = (paths.ROOT / DATA).read_text(encoding="utf-8")
    chars = sorted(set(text))
    index = {c: i for i, c in enumerate(chars)}
    ids = np.array([index[c] for c in text])
    counts = np.zeros((len(chars), len(chars)))
    np.add.at(counts, (ids[:-1], ids[1:]), 1)
    return counts / counts.sum(axis=1, keepdims=True), index, ids


def bigram_prob(a: str, b: str) -> float:
    """The probability the counted bigram model gives character b after character a."""
    P, index, _ = _bigram()
    return float(P[index[a], index[b]])


@functools.lru_cache(maxsize=1)
def bigram_nll() -> float:
    """The average NLL of the counted bigram model on all of Tiny Shakespeare, the text it counted."""
    P, _, ids = _bigram()
    return float(-np.mean(np.log(P[ids[:-1], ids[1:]])))


def ppl_picture(labels: dict) -> str:
    """Perplexity against average NLL, the curve e to the power of it, with three models on it."""
    from everything_ai import chart
    xs = [i / 50 for i in range(0, 226)]
    panel = chart.Panel((0, 4.6), (0, 72), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    panel.line(xs, [math.exp(x) for x in xs], "loss", 2.2)
    points = [(0.0, labels["perfect"], "start", 0.15, 6), (bigram_nll(), labels["bigram"], "end", -0.08, 15),
              (math.log(65), labels["uniform"], "end", -0.12, -4)]
    for x, text, anchor, dx, dy in points:
        panel.points([x], [math.exp(x)], "loss")
        shown = f"{round(math.exp(x), 1):g}".replace(".", labels["decimal"])   # 65, 11.6 or 11,6, 1
        panel.text(x + dx, math.exp(x) + dy, text.format(shown), "loss", anchor)
    return chart.figure([panel], alt=labels["alt"], width=330, height=270)


# The lesson is done when these are correct. "compare" is the choice question in the lesson.
REQUIRED = ["compare", "perplexity", "char-perplexity"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _close(got, want) -> bool:
    try:
        return math.isclose(float(got), float(want), rel_tol=1e-6, abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


def _num(v):
    """A number for the messages: at most 4 decimals, without float noise; a huge one in words, so that no e-notation
    shows (e to the power of a total NLL); anything else as Python shows it."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return repr(v)
    if math.isfinite(x) and abs(x) >= 1e6:
        digits = len(f"{abs(x):.0f}")
        return {"en": f"a number with {digits} digits", "nl": f"een getal van {digits} cijfers"}
    return round(x, 4)


# --- exercise 2: perplexity -----------------------------------------------------------------------------------------------

class _Frac(float):
    """1/65 the way a check message writes it, not as 0.015384615384615385; the learner's code gets a plain float."""

    def __repr__(self):
        return "1/65"

    def __deepcopy__(self, memo):
        return float(self)


_P_HEAD = "import numpy as np\n\n\ndef perplexity(probs):\n"
_P_START = _P_HEAD + "    nll = ...\n    return ...\n"


def _p(nll: str = "-np.mean(np.log(probs))", ret: str = "np.exp(nll)") -> str:
    return _P_HEAD + f"    nll = {nll}\n    return {ret}\n"


_P = _p()

# --- exercise 3: char_perplexity ------------------------------------------------------------------------------------------

_C_HEAD = "import numpy as np\n\n\ndef char_perplexity(avg_nll, n_tokens, n_chars):\n"
_C_START = _C_HEAD + "    nll_per_char = ...\n    return np.exp(nll_per_char)\n"


def _c(per_char: str = "avg_nll * n_tokens / n_chars") -> str:
    return _C_HEAD + f"    nll_per_char = {per_char}\n    return np.exp(nll_per_char)\n"


_C = _c()

EXERCISES = {
    "perplexity": FunctionBox(
        name="perplexity",
        function="perplexity",
        start=_P_START,
        solution=_P,
        visible=[([0.5, 0.125],)],
        hidden=[([_Frac(1 / 65)] * 3,), ([1.0, 1.0],), ([0.1],), ([0.9, 0.01, 0.5, 0.2],)],
        compare=_close,
        show=_num,
        first=[_DOTS],
        mistakes=[
            Mistake("no-minus", wrong=_p(nll="np.mean(np.log(probs))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your NLL has no minus, so it is negative, and e to the power of it is below 1. The NLL is minus the mean of the logs.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je NLL heeft geen min, dus hij is negatief, en e tot de macht ervan is kleiner dan 1. De NLL is min het gemiddelde van de logs."),
            Mistake("no-exp", wrong=_p(ret="nll"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the average NLL itself. Perplexity is e to the power of it: `np.exp(nll)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is de gemiddelde NLL zelf. Perplexity is e tot de macht ervan: `np.exp(nll)`."),
            Mistake("sum", wrong=_p(nll="-np.sum(np.log(probs))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you add up the surprises instead of taking their mean, so a longer text gets a higher perplexity. Use `np.mean`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt de verrassingen op in plaats van hun gemiddelde te nemen, dus een langere tekst krijgt een hogere perplexity. Gebruik `np.mean`."),
            Mistake("mean-prob", wrong=_p(nll="np.mean(probs)", ret="1 / nll"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: one divided by the mean probability. Perplexity takes the mean of the logs, not of the probabilities: a single tiny probability weighs heavily, as it should.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: één gedeeld door de gemiddelde kans. Perplexity neemt het gemiddelde van de logs, niet van de kansen: één piepkleine kans weegt zwaar, zoals het hoort."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the average NLL of the probabilities, and e to the power of it.",
                      "nl": "Vervang de twee `...`: de gemiddelde NLL van de kansen, en e tot de macht daarvan."},
            "arguments": {"en": "The check calls `{call}`, with one list of probabilities, and the `def` line of your function has another number of names between its parentheses. Keep `(probs)`.",
                          "nl": "De check roept `{call}` aan, met één lijst kansen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(probs)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your two lines with steps 2 and 3 of the recipe.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je twee regels met stap 2 en 3 van het recept."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for every list of probabilities?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elke lijst kansen?"},
            "correct": {"en": "Your function gives 65 for a model that gives every one of 65 characters the same probability, and 1 for a model that is always sure and always right: the number of tokens the model is choosing between.",
                        "nl": "Je functie geeft 65 voor een model dat elk van de 65 tekens dezelfde kans geeft, en 1 voor een model dat altijd zeker is en altijd gelijk heeft: het aantal tokens waartussen het model kiest."},
        },
    ),
    "char-perplexity": FunctionBox(
        name="char-perplexity",
        function="char_perplexity",
        start=_C_START,
        solution=_C,
        visible=[(3.0, 100, 400)],
        hidden=[(2.0, 50, 50), (4.0, 10, 37), (1.5, 1000, 4200)],
        compare=_close,
        show=_num,
        first=[_DOTS],
        mistakes=[
            Mistake("per-token", wrong=_c(per_char="avg_nll"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is still the perplexity per token. Spread the total NLL of the text, `avg_nll * n_tokens`, over its characters.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is nog de perplexity per token. Verdeel de totale NLL van de tekst, `avg_nll * n_tokens`, over zijn tekens."),
            Mistake("upside-down", wrong=_c(per_char="avg_nll * n_chars / n_tokens"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you multiply by characters per token, so the NLL per character grows. The total NLL, `avg_nll * n_tokens`, is divided by the number of characters.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je vermenigvuldigt met tekens per token, dus de NLL per teken groeit. De totale NLL, `avg_nll * n_tokens`, wordt gedeeld door het aantal tekens."),
            Mistake("total", wrong=_c(per_char="avg_nll * n_tokens"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is e to the power of the total NLL of the whole text. Divide it by the number of characters, to get the NLL per character.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is e tot de macht van de totale NLL van de hele tekst. Deel die door het aantal tekens, om de NLL per teken te krijgen."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: the NLL per character, from the average NLL per token, the number of tokens and the number of characters.",
                      "nl": "Vervang de `...`: de NLL per teken, uit de gemiddelde NLL per token, het aantal tokens en het aantal tekens."},
            "arguments": {"en": "The check calls `{call}`, with three numbers, and the `def` line of your function has another number of names between its parentheses. Keep `(avg_nll, n_tokens, n_chars)`.",
                          "nl": "De check roept `{call}` aan, met drie getallen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(avg_nll, n_tokens, n_chars)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. First the total NLL of the text, then spread over its characters.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Eerst de totale NLL van de tekst, dan verdeeld over zijn tekens."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function use all three of its numbers?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gebruikt je functie alle drie haar getallen?"},
            "correct": {"en": "Your function puts models with different tokens on one scale: per character. An average NLL of 3.0 per token, at 4 characters per token, gives a perplexity of 2.12 per character.",
                        "nl": "Je functie zet modellen met verschillende tokens op één schaal: per teken. Een gemiddelde NLL van 3,0 per token, bij 4 tekens per token, geeft een perplexity van 2,12 per teken."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("perplexity", _P_START, "empty", None),
    ("perplexity", _p(nll="-np.mean(np.log2(probs))", ret="2 ** nll"), "correct", None),
    ("perplexity", _P_HEAD + "    return 1 / np.prod(probs) ** (1 / len(probs))\n", "correct", None),
    ("perplexity", _p(nll="np.mean(np.log(probs))"), "wrong", "no-minus"),
    ("perplexity", _p(ret="nll"), "wrong", "no-exp"),
    ("perplexity", _p(nll="-np.sum(np.log(probs))"), "wrong", "sum"),
    ("perplexity", _p(nll="np.mean(probs)", ret="1 / nll"), "wrong", "mean-prob"),
    ("perplexity", _p(nll="-np.mean(np.log(probs)) / 2"), "wrong", None),
    ("char-perplexity", _C_START, "empty", None),
    ("char-perplexity", _C_HEAD + "    return np.exp(avg_nll) ** (n_tokens / n_chars)\n", "correct", None),
    ("char-perplexity", _c(per_char="avg_nll"), "wrong", "per-token"),
    ("char-perplexity", _c(per_char="avg_nll * n_chars / n_tokens"), "wrong", "upside-down"),
    ("char-perplexity", _c(per_char="avg_nll * n_tokens"), "wrong", "total"),
    ("char-perplexity", _c(per_char="avg_nll / 4"), "almost", None),
]
