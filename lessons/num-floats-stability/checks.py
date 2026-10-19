# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson num-floats-stability: logsumexp, the log of a sum of e-powers that never overflows, and
ce_from_logits, the cross-entropy loss worked out straight from the logits. And the picture: the gap between a float64
and the next one, for numbers from 1 to 10 to the power 16."""
import math
import re

import numpy as np

from everything_ai.check import FunctionBox, Mistake


def stable_lse(z) -> float:
    z = np.asarray(z, dtype=float)
    m = z.max()
    return float(m + np.log(np.sum(np.exp(z - m))))


def number(x: float, digits: int, lang: str) -> str:
    s = f"{x:.{digits}f}"
    return s.replace(".", ",") if lang == "nl" else s


# --- the picture: the gap between neighbouring floats grows with the number ------------------------------------------------

def gap_picture(labels: dict) -> str:
    """The gap between a float64 and the next one, against the number itself, both on a log scale: a straight line,
    because the gap grows with the number. Drawn through the powers of 2, where the gap is exactly 2.2 times 10 to the
    power -16 times the number; in between it stays the same up to the next power of 2, so points in between would
    make the line wobble. A maths object, so in ink."""
    from everything_ai import chart
    xs = [2.0 ** k for k in range(0, 54)] + [1e16]
    panel = chart.Panel((1, 1e16), (1e-16, 1e3), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"], xlog=True, ylog=True)
    panel.line(xs, [float(np.spacing(x)) for x in xs], "ink", 2.2)
    panel.points([1.0, 1e16], [float(np.spacing(1.0)), float(np.spacing(1e16))], "ink", 4.5)
    panel.text(30, 1.8e-16, labels["at_one"], "ink", "start")
    panel.text(1e15, 6, labels["at_big"], "ink", "end")
    return chart.figure([panel], alt=labels["alt"], width=520, height=300)


# The lesson is done when these are correct. "underflow" is the choice question in the lesson.
REQUIRED = ["underflow", "logsumexp", "ce-from-logits"]

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


def _close(got, want) -> bool:
    try:
        return math.isclose(float(got), float(want), rel_tol=1e-9, abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


class _Vector(np.ndarray):
    """Logits that a check message writes the way the lesson's code does: np.array([...]). The learner's code gets a
    plain copy."""

    def __repr__(self):
        return f"np.array({[int(v) if float(v) == int(v) else float(v) for v in np.asarray(self).tolist()]!r})"

    def __deepcopy__(self, memo):
        return np.array(self)


def _v(values) -> np.ndarray:
    return np.array(values, dtype=float).view(_Vector)


def _body(text: str) -> str:
    """The learner's code without comments and print lines, so that only what the function works out counts."""
    return "\n".join(line.split("#")[0] for line in text.splitlines() if not line.strip().startswith("print("))


def _not_finite(got, text) -> bool:
    try:
        return not math.isfinite(float(got))
    except (TypeError, ValueError):
        return False


def _log_of_zero(got, text) -> bool:
    """Softmax first, with the largest value off, and the log afterwards: where the probability of class c becomes 0,
    minus its log is plus infinity. A division makes the probability; code that only works out m and then the plain
    formula gives `inf` or `nan` from the overflow, and gets the overflow message."""
    try:
        infinite = float(got) == math.inf
    except (TypeError, ValueError):
        return False
    body = _body(text)
    return infinite and "max" in body and "/" in body


_OVERFLOW = Mistake(
    "overflow", matches=_not_finite, state="almost",
    en="`{call}` gives `{got}`, and it should give `{want}`: somewhere `np.exp` gave a result too large or too small for a float, which became `inf` or 0. Take the largest value off first, so that every exponent is 0 or below, and add it back at the end.",
    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: ergens gaf `np.exp` een uitkomst die te groot of te klein is voor een float, en die werd `inf` of 0. Haal eerst de grootste waarde eraf, zodat elke exponent 0 of lager is, en tel hem aan het eind er weer bij.")

# --- exercise 2: log-sum-exp -----------------------------------------------------------------------------------------------

_L_HEAD = "import numpy as np\n\n\ndef logsumexp(z):\n"
_L_START = _L_HEAD + "    # the log of the sum of e to the power of each value of z, without overflow\n    ...\n"


def _l(body: str = "    m = z.max()\n    return m + np.log(np.sum(np.exp(z - m)))\n") -> str:
    return _L_HEAD + body


# --- exercise 3: the cross-entropy straight from the logits -----------------------------------------------------------------

_C_HEAD = "import numpy as np\n\n\ndef ce_from_logits(z, c):\n"
_C_START = _C_HEAD + "    # the cross-entropy loss of the logits z when the right class is c, without overflow\n    ...\n"


def _c(last: str = "    return m + np.log(np.sum(np.exp(z - m))) - z[c]\n") -> str:
    return _C_HEAD + "    m = z.max()\n" + last


_SMALL = _v([1, 2, 3])
_BIG = _v([1000, 1001, 999])
_LOW = _v([-1000, -1001, -999])
_SPREAD = _v([500, -500, 0])
_FAR = _v([0, 800, -800])   # the largest value in the middle: taking off z[0] or the mean still overflows

EXERCISES = {
    "logsumexp": FunctionBox(
        name="logsumexp",
        function="logsumexp",
        start=_L_START,
        solution=_l(),
        visible=[(_SMALL,)],
        hidden=[(_BIG,), (_LOW,), (_SPREAD,), (_v([0, 0, 0, 0]),), (_v([800]),), (_FAR,)],
        show=_num,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("not-back", wrong=_l("    m = np.max(z)\n    return np.log(np.sum(np.exp(z - m)))\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you take the largest value off, but do not add it back. The log of the shifted sum is short by exactly that value.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je haalt de grootste waarde eraf, maar telt hem niet terug. De log van de verschoven som komt precies die waarde tekort."),
            Mistake("minimum", wrong=_l("    m = np.min(z)\n    return m + np.log(np.sum(np.exp(z - m)))\n"), state="almost",
                    en="`{call}` gives `{got}`, and it should give `{want}`: you take the smallest value off, so the others still get large exponents and overflow. Take the largest off: then every exponent is 0 or below.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je haalt de kleinste waarde eraf, dus de andere krijgen nog steeds grote exponenten en lopen over. Haal de grootste eraf: dan is elke exponent 0 of lager."),
            _OVERFLOW,
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: take the largest value off, sum the e-powers, take the log, and add the largest value back.",
                      "nl": "Vervang de `...` door de body van de functie: haal de grootste waarde eraf, tel de e-machten op, neem de log, en tel de grootste waarde er weer bij."},
            "arguments": {"en": "The check calls `{call}`, with one array, and the `def` line of your function has another number of names between its parentheses. Keep `(z)`.",
                          "nl": "De check roept `{call}` aan, met één array, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(z)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare with `np.log(np.sum(np.exp(z)))` for small values: they must give the same.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk voor kleine waarden met `np.log(np.sum(np.exp(z)))`: die moeten hetzelfde geven."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for very large and very negative values too?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie ook voor heel grote en heel negatieve waarden?"},
            "correct": {"en": "Your function is right in all {n} cases, also for values of a thousand and more, where the plain formula gives `inf`.",
                        "nl": "Je functie klopt in alle {n} gevallen, ook voor waarden van duizend en meer, waar de gewone formule `inf` geeft."},
        },
    ),
    "ce-from-logits": FunctionBox(
        name="ce-from-logits",
        function="ce_from_logits",
        start=_C_START,
        solution=_c(),
        visible=[(_v([2, 1, 0]), 0)],
        hidden=[(_BIG, 1), (_BIG, 2), (_LOW, 0), (_SPREAD, 1), (_v([3, 3]), 0), (_FAR, 2)],
        show=_num,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("sign", wrong=_c("    return z[c] - (m + np.log(np.sum(np.exp(z - m))))\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the sign is the other way round, so the loss is 0 or below. The loss is minus the log of the probability, the log-sum-exp minus the logit of the right class.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: het teken is omgedraaid, dus de loss is 0 of lager. De loss is min de log van de kans: de log-sum-exp min de logit van de goede klasse."),
            Mistake("lse-only", wrong=_c("    return m + np.log(np.sum(np.exp(z - m)))\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the log-sum-exp alone. The loss of class `c` is the log-sum-exp minus its own logit, `z[c]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is alleen de log-sum-exp. De loss van klasse `c` is de log-sum-exp min haar eigen logit, `z[c]`."),
            Mistake("log-of-zero", matches=_log_of_zero, state="almost",
                    en="`{call}` gives `{got}`, and it should give `{want}`: you take the largest value off, but then work out the probability of class `c` first. Here that probability is so small that it becomes 0, and the log of 0 is minus infinity. Stay in logs: the loss is the log-sum-exp of `z` minus `z[c]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je haalt de grootste waarde eraf, maar rekent daarna eerst de kans van klasse `c` uit. Hier is die kans zo klein dat hij 0 wordt, en de log van 0 is min oneindig. Blijf in logs: de loss is de log-sum-exp van `z` min `z[c]`."),
            _OVERFLOW,
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: the log-sum-exp of `z`, by the recipe, minus the logit of the right class.",
                      "nl": "Vervang de `...` door de body van de functie: de log-sum-exp van `z`, met het recept, min de logit van de goede klasse."},
            "arguments": {"en": "The check calls `{call}`, with the logits and the right class, and the `def` line of your function has another number of names between its parentheses. Keep `(z, c)`.",
                          "nl": "De check roept `{call}` aan, met de logits en de goede klasse, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(z, c)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. For small logits, compare with minus the log of the softmax probability of class `c`.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk voor kleine logits met min de log van de softmax-kans van klasse `c`."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for any class `c`, and for very large logits?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elke klasse `c`, en voor heel grote logits?"},
            "correct": {"en": "Your loss is right in all {n} cases, also for logits where softmax first and log afterwards would give `inf` or `nan`.",
                        "nl": "Je loss klopt in alle {n} gevallen, ook voor logits waar eerst softmax en dan de log `inf` of `nan` zou geven."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("logsumexp", _L_START, "empty", None),
    ("logsumexp", _l(), "correct", None),
    ("logsumexp", _l("    m = max(z)\n    total = 0\n    for v in z:\n        total = total + np.exp(v - m)\n    return m + np.log(total)\n"), "correct", None),
    ("logsumexp", _l("    return np.log(np.sum(np.exp(z)))\n"), "almost", "overflow"),
    ("logsumexp", _l("    # first take np.max(z) off\n    return np.log(np.sum(np.exp(z)))\n"), "almost", "overflow"),
    ("logsumexp", _l("    m = np.max(z)\n    return np.log(np.sum(np.exp(z - m)))\n"), "wrong", "not-back"),
    ("logsumexp", _l("    m = np.min(z)\n    return m + np.log(np.sum(np.exp(z - m)))\n"), "almost", "minimum"),
    ("logsumexp", _l("    m = z.mean()\n    return m + np.log(np.sum(np.exp(z - m)))\n"), "almost", "overflow"),
    ("logsumexp", _l("    m = z[0]\n    return m + np.log(np.sum(np.exp(z - m)))\n"), "almost", "overflow"),
    ("ce-from-logits", _C_START, "empty", None),
    ("ce-from-logits", _c(), "correct", None),
    ("ce-from-logits", _C_HEAD + "    e = np.exp(z - np.max(z))\n    return -np.log(e[c] / e.sum())\n", "almost", "log-of-zero"),
    ("ce-from-logits", _C_HEAD + "    return np.log(np.sum(np.exp(z - z[c])))\n", "almost", "overflow"),
    ("ce-from-logits", _C_HEAD + "    p = np.exp(z) / np.sum(np.exp(z))\n    return -np.log(p[c])\n", "almost", "overflow"),
    ("ce-from-logits", _C_HEAD + "    m = z.max()\n    return np.log(np.sum(np.exp(z))) - z[c]\n", "almost", "overflow"),
    ("ce-from-logits", _C_HEAD + "    m = z.max()\n    p = np.exp(z) / np.sum(np.exp(z))\n    return -np.log(p[c])\n", "almost", "overflow"),
    ("ce-from-logits", _C_HEAD + "    m = z.mean()\n    return m + np.log(np.sum(np.exp(z - m))) - z[c]\n", "almost", "overflow"),
    ("ce-from-logits", _c("    return z[c] - (m + np.log(np.sum(np.exp(z - m))))\n"), "wrong", "sign"),
    ("ce-from-logits", _c("    return m + np.log(np.sum(np.exp(z - m)))\n"), "wrong", "lse-only"),
]
