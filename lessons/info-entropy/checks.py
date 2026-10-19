# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson info-entropy: entropy, the expected surprise of a distribution in bits, and min_bits, the
least number of bits that a code using only how often each character appears needs for a whole text. And the data:
the characters of Tiny Shakespeare, what gzip makes of them, and the species of the Palmer penguins; with the picture
of the entropy of a coin."""
import gzip
import math
import re
from collections import Counter
from functools import cache

import numpy as np

from everything_ai import paths
from everything_ai.check import FunctionBox, Mistake

PLAYS = "data/tiny-shakespeare.txt"   # about 1.1 MB of plays by Shakespeare: see data/README.md
PENGUINS = "data/penguins.csv"         # 344 penguins of the Palmer Archipelago: see data/README.md


@cache
def plays() -> str:
    return (paths.ROOT / PLAYS).read_text(encoding="utf-8")


@cache
def plays_entropy() -> float:
    """The entropy, in bits, of the characters of the plays, each taken on its own."""
    counts = np.array(list(Counter(plays()).values()), dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def kinds() -> int:
    return len(set(plays()))


@cache
def gzip_bits() -> float:
    """How many bits per character gzip, at its best setting, needs for the plays."""
    data = plays().encode("utf-8")
    return len(gzip.compress(data, 9)) * 8 / len(plays())


@cache
def species_counts() -> list:
    import pandas as pd
    d = pd.read_csv(paths.ROOT / PENGUINS)
    counts = d["species"].value_counts()
    return [int(counts[s]) for s in ("Adelie", "Gentoo", "Chinstrap")]


def uniform_bits() -> float:
    """The entropy if all characters of the plays were equally common: log2 of how many kinds there are."""
    return math.log2(kinds())


def species_probs() -> list:
    c = species_counts()
    return [n / sum(c) for n in c]


def surprise_bits(p: float) -> float:
    return -math.log2(p)


def species_entropy() -> float:
    return -sum(p * math.log2(p) for p in species_probs())


def number(x: float, digits: int, lang: str) -> str:
    s = f"{x:.{digits}f}"
    return s.replace(".", ",") if lang == "nl" else s


def grouped(n: int, lang: str) -> str:
    s = f"{n:,}"
    return s.replace(",", ".") if lang == "nl" else s


# --- the picture: the entropy of a coin that lands heads with probability p -----------------------------------------------

def coin_picture(labels: dict) -> str:
    from everything_ai import chart
    ps = [i / 200 for i in range(1, 200)]
    hs = [-(p * math.log2(p) + (1 - p) * math.log2(1 - p)) for p in ps]
    panel = chart.Panel((0, 1), (0, 1.1), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    panel.line([0] + ps + [1], [0] + hs + [0], "ink", 2.6)
    for p in (0.5, 0.9):
        h = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        panel.points([p], [h], "ink", 5)
        words = labels["point"].format(p=labels["number"](p, 1), h=labels["number"](h, 2))
        if p > 0.7:   # near the right edge: the words end at the point, so they stay inside the chart
            panel.text(p - 0.02, h + 0.07, words, "ink", "end")
        else:
            panel.text(p, h + 0.07, words, "ink")
    return chart.figure([panel], alt=labels["alt"], width=520, height=290)


# The lesson is done when these are correct. "most-entropy" is the choice question in the lesson.
REQUIRED = ["most-entropy", "entropy", "min-bits"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _num(v) -> str:
    if _several(v) or (not isinstance(v, (str, dict)) and np.ndim(v) > 0):   # a list or an array: each number on its own
        try:
            return "[" + ", ".join(_num(x) for x in np.asarray(v, dtype=float).ravel().tolist()) + "]"
        except (TypeError, ValueError):
            return repr(v)
    try:
        x = float(v)
    except (TypeError, ValueError):
        return repr(v)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    text = f"{round(x, 4) + 0.0:.4f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _several(v) -> bool:
    """Several numbers, such as an array with one number per outcome, where one number belongs."""
    try:
        return not isinstance(v, (str, dict)) and np.size(v) > 1
    except (TypeError, ValueError):
        return False


def _close(got, want) -> bool:
    """The same number; two nans count as the same, so a wrong version that gives nan can name the mistake."""
    if got is None or want is None or _several(got) or _several(want):
        return False
    try:
        a, b = np.asarray(got, dtype=float).item(), np.asarray(want, dtype=float).item()
    except (TypeError, ValueError):
        return False
    return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-9)


class _Vector(np.ndarray):
    """Probabilities that a check message writes the way the lesson's code does: np.array([...]). The learner's code
    gets a plain copy."""

    def __repr__(self):
        return f"np.array({[int(v) if float(v) == int(v) else float(v) for v in np.asarray(self).tolist()]!r})"

    def __deepcopy__(self, memo):
        return np.array(self)


def _v(values) -> np.ndarray:
    return np.array(values, dtype=float).view(_Vector)


_SEVERAL = Mistake("not-one-number", matches=lambda got, text: _several(got),
                   en="`{call}` gives `{got}`, several numbers, and it should give one number, `{want}`. Step 4 of the recipe weighs each surprise by its probability and adds them all up.",
                   nl="`{call}` geeft `{got}`, meerdere getallen, en dat moet één getal zijn, `{want}`. Stap 4 van het recept weegt elke verrassing met haar kans en telt ze allemaal op.")

_MATH_LOG = Mistake("math-log-zero", error=lambda e: isinstance(e, ValueError) and ("math domain error" in str(e) or "positive input" in str(e)), state="error",
                    en="With `{call}`, your function stops: `math` cannot take the log of 0, or of a number below 0. An outcome with probability 0 never happens and adds no surprise, so leave it out before you take the log.",
                    nl="Met `{call}` stopt je functie: `math` kan de log van 0, of van een getal onder 0, niet nemen. Een uitkomst met kans 0 gebeurt nooit en voegt geen verrassing toe, dus laat haar weg voordat je de log neemt.")

# --- exercise 2: the entropy of a distribution ---------------------------------------------------------------------------

_E_HEAD = "import numpy as np\n\n\ndef entropy(p):\n"
_E_START = _E_HEAD + "    # the entropy in bits of the probabilities in the array p\n    ...\n"


def _e(body: str = "    p = p[p > 0]\n    return -np.sum(p * np.log2(p))\n") -> str:
    return _E_HEAD + body


_NAN = Mistake("zero-probability", wrong=_e("    return -np.sum(p * np.log2(p))\n"),
               en="`{call}` gives `{got}`, short for not a number. An outcome with probability 0 gives `0 * np.log2(0)`, which NumPy works out as 0 times minus infinity. Such an outcome never happens, so it adds no surprise: leave it out.",
               nl="`{call}` geeft `{got}`, kort voor not a number: geen getal. Een uitkomst met kans 0 geeft `0 * np.log2(0)`, en dat rekent NumPy uit als 0 keer min oneindig. Zo'n uitkomst gebeurt nooit, dus voegt ze geen verrassing toe: laat haar weg.")


# --- exercise 3: the least number of bits for a whole text ---------------------------------------------------------------

_B_HEAD = "import numpy as np\n\n\ndef min_bits(text):\n"
_B_START = _B_HEAD + "    # the least number of bits a code needs for the whole text, using only how often each character appears\n    ...\n"
_B_BODY = ("    counts = {}\n    for ch in text:\n        if ch in counts:\n            counts[ch] = counts[ch] + 1\n"
           "        else:\n            counts[ch] = 1\n    probs = []\n    for ch in counts:\n"
           "        probs.append(counts[ch] / len(text))\n    p = np.array(probs)\n")


def _b(last: str = "    return -np.sum(p * np.log2(p)) * len(text)\n") -> str:
    return _B_HEAD + _B_BODY + last


EXERCISES = {
    "entropy": FunctionBox(
        name="entropy",
        function="entropy",
        start=_E_START,
        solution=_e(),
        visible=[(_v([0.5, 0.25, 0.25]),)],
        hidden=[(_v([0.5, 0.5]),), (_v([1.0]),), (_v([0.1, 0.2, 0.3, 0.4]),), (_v([0.25, 0.25, 0.25, 0.25]),), (_v([0.5, 0.5, 0.0]),)],
        show=_num,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            _SEVERAL,
            _MATH_LOG,
            _NAN,
            Mistake("nats", wrong=_e("    p = p[p > 0]\n    return -np.sum(p * np.log(p))\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the entropy in nats, with the natural log. In bits, the log has base 2: `np.log2`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is de entropie in nats, met de natuurlijke log. In bits heeft de log grondtal 2: `np.log2`."),
            Mistake("no-minus", wrong=_e("    p = p[p > 0]\n    return np.sum(p * np.log2(p))\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the sign is wrong. The log of a probability is 0 or below, so the surprise is minus the log.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: het teken klopt niet. De log van een kans is 0 of minder, dus de verrassing is min de log."),
            Mistake("not-weighted", wrong=_e("    p = p[p > 0]\n    return -np.sum(np.log2(p))\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you add up the surprises, but each should be weighted by its probability first, as in an expected value.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt de verrassingen op, maar elke verrassing moet eerst met haar kans gewogen worden, zoals bij een verwachtingswaarde."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: the surprise of each outcome in bits, weighted by its probability, added up.",
                      "nl": "Vervang de `...` door de body van de functie: de verrassing van elke uitkomst in bits, gewogen met haar kans, opgeteld."},
            "arguments": {"en": "The check calls `{call}`, with one array of probabilities, and the `def` line of your function has another number of names between its parentheses. Keep `(p)`.",
                          "nl": "De check roept `{call}` aan, met één array kansen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(p)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Work out the surprise of each outcome in bits by hand, and weigh each by its probability.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken de verrassing van elke uitkomst in bits met de hand uit, en weeg elke met haar kans."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for any distribution, also one with a probability of 0 or of 1?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elke verdeling, ook een met een kans van 0 of van 1?"},
            "correct": {"en": "Your entropy is right in all {n} cases: 1 bit for a fair coin, 2 for four equal outcomes, 0 for a sure one.",
                        "nl": "Je entropie klopt in alle {n} gevallen: 1 bit voor een eerlijke munt, 2 voor vier gelijke uitkomsten, 0 voor een zekere."},
        },
    ),
    "min-bits": FunctionBox(
        name="min-bits",
        function="min_bits",
        start=_B_START,
        solution=_b(),
        visible=[("aaab",)],
        hidden=[("abab",), ("aaaa",), ("the cat sat on the mat",), ("abcdefgh",)],
        show=_num,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("not-one-number", matches=_SEVERAL.matches,
                    en="`{call}` gives `{got}`, several numbers, and it should give one number, `{want}`. The entropy adds up the weighted surprises of all characters, and the whole text needs that times its length.",
                    nl="`{call}` geeft `{got}`, meerdere getallen, en dat moet één getal zijn, `{want}`. De entropie telt de gewogen verrassingen van alle tekens op, en de hele tekst heeft dat keer zijn lengte nodig."),
            Mistake("per-character", wrong=_b("    return -np.sum(p * np.log2(p))\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the number of bits per character, the entropy. For the whole text, multiply by the number of characters.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is het aantal bits per teken, de entropie. Voor de hele tekst vermenigvuldig je met het aantal tekens."),
            Mistake("nats", wrong=_b("    return -np.sum(p * np.log(p)) * len(text)\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is in nats, with the natural log. Bits need the log with base 2.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is in nats, met de natuurlijke log. Bits hebben de log met grondtal 2 nodig."),
            Mistake("distinct", wrong=_b("    return -np.sum(p * np.log2(p)) * len(counts)\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you multiply by the number of different characters. Every character of the text needs its bits, also the ones that repeat: multiply by the length of the text.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je vermenigvuldigt met het aantal verschillende tekens. Elk teken van de tekst heeft zijn bits nodig, ook de tekens die terugkomen: vermenigvuldig met de lengte van de tekst."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: count each character, turn the counts into probabilities, and multiply the entropy by the length of the text.",
                      "nl": "Vervang de `...` door de body van de functie: tel elk teken, zet de aantallen om in kansen, en vermenigvuldig de entropie met de lengte van de tekst."},
            "arguments": {"en": "The check calls `{call}`, with one text, and the `def` line of your function has another number of names between its parentheses. Keep `(text)`.",
                          "nl": "De check roept `{call}` aan, met één tekst, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(text)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Count the characters of the text by hand: what are their probabilities, what is the entropy, and how many characters are there?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Tel de tekens van de tekst met de hand: wat zijn hun kansen, wat is de entropie, en hoeveel tekens zijn er?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for any text, also one with a single kind of character?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elke tekst, ook een met maar één soort teken?"},
            "correct": {"en": "Your function is right in all {n} cases: the entropy of the characters, times the length of the text.",
                        "nl": "Je functie klopt in alle {n} gevallen: de entropie van de tekens, keer de lengte van de tekst."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("entropy", _E_START, "empty", None),
    ("entropy", _e(), "correct", None),
    ("entropy", _e("    total = 0\n    for q in p:\n        if q > 0:\n            total = total - q * np.log2(q)\n    return total\n"), "correct", None),
    ("entropy", _e("    return -np.sum(p * np.log2(p))\n"), "wrong", "zero-probability"),
    ("entropy", _e("    p = p[p > 0]\n    return -np.sum(p * np.log(p))\n"), "wrong", "nats"),
    ("entropy", _e("    p = p[p > 0]\n    return np.sum(p * np.log2(p))\n"), "wrong", "no-minus"),
    ("entropy", _e("    p = p[p > 0]\n    return -np.sum(np.log2(p))\n"), "wrong", "not-weighted"),
    ("entropy", _e("    p = p[p > 0]\n    return -(p * np.log2(p))\n"), "wrong", "not-one-number"),
    ("entropy", _e("    return np.sum(p * np.log2(-p))\n"), "wrong", None),   # a nan without a 0 in p: not the mask
    ("entropy", _e("    import math\n    total = 0\n    for q in p:\n        total = total - q * math.log2(q)\n    return total\n"), "error", "math-log-zero"),
    ("entropy", _e("    total = 0\n    for q in p:\n        total = total - q * np.log2(q)\n    return total\n"), "wrong", "zero-probability"),
    ("min-bits", _B_START, "empty", None),
    ("min-bits", _b(), "correct", None),
    ("min-bits", _b("    p = p[p > 0]\n    return len(text) * np.sum(p * np.log2(1 / p))\n"), "correct", None),
    ("min-bits", _b("    return -np.sum(p * np.log2(p))\n"), "wrong", "per-character"),
    ("min-bits", _b("    return -np.sum(p * np.log(p)) * len(text)\n"), "wrong", "nats"),
    ("min-bits", _b("    return -np.sum(p * np.log2(p)) * len(counts)\n"), "wrong", "distinct"),
    ("min-bits", _b("    return -(p * np.log2(p)) * len(text)\n"), "wrong", "not-one-number"),
    ("min-bits", _B_HEAD + "    from collections import Counter\n    p = np.array(list(Counter(text).values())) / len(text)\n"
     "    return -np.sum(p * np.log2(p)) * len(text)\n", "correct", None),
]
