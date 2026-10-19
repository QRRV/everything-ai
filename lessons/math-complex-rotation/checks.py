# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson math-complex-rotation: rotate_pairs, which reads a real vector as pairs of numbers, each pair a
complex number, and turns each pair by its own angle. And the picture of a rotation in the complex plane."""
import math
import re

import numpy as np

from everything_ai.check import FunctionBox, Mistake, _show_rounded


def plane_picture(labels: dict) -> str:
    """The complex plane with the unit circle, a number z = 2 + 1i as an arrow, and z times e^(i·60°) as a second arrow:
    the same length, 60 degrees further round. Ink only: nothing here is learned or a loss. Both axes have the same
    scale, about 50 pixels per unit (258 pixels for 5.2 units across, 189 for 3.8 up), so the circle is round and the
    two arrows look as long as they are."""
    from everything_ai import chart
    z = 2 + 1j
    w = z * np.exp(1j * math.pi / 3)
    p = chart.Panel((-2.6, 2.6), (-1.2, 2.6), title=labels["title"], xlabel=labels["re"], ylabel=labels["im"],
                    xticks=[-2, -1, 0, 1, 2], yticks=[-1, 0, 1, 2])
    t = np.linspace(0, 2 * math.pi, 120)
    p.line(np.cos(t), np.sin(t), "ink-3", 1.2, dashed=True)
    p.arrow(0, 0, z.real, z.imag, "ink", 2.2)
    p.arrow(0, 0, w.real, w.imag, "ink-2", 2.2)
    a0, a1 = np.angle(z), np.angle(w)
    arc = np.linspace(a0, a1, 30)
    p.line(1.25 * np.cos(arc), 1.25 * np.sin(arc), "ink-2", 1.4)   # the angle, outside the dashed circle
    p.text(z.real + 0.1, z.imag - 0.5, labels["z"], "ink", "middle")
    p.text(w.real - 0.1, w.imag + 0.18, labels["w"], "ink-2", "middle")
    p.text(1.55 * np.cos((a0 + a1) / 2), 1.55 * np.sin((a0 + a1) / 2) - 0.05, labels["angle"], "ink-2", "middle")
    svg = chart.figure([p], alt=labels["alt"], width=330, height=269)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


class _Vec(np.ndarray):
    """A NumPy array that a check message writes as `np.array([...])`; the learner's code gets a plain array."""

    def __repr__(self):
        return "np.array([" + ", ".join(_plain(v) for v in np.asarray(self).tolist()) + "])"

    def __deepcopy__(self, memo):
        return np.array(self)


class _Angles(list):
    """Angles that a check message writes the way the lesson does, such as `[np.pi / 2, 0.3]`."""

    def __init__(self, values, shown):
        super().__init__(values)
        self.shown = shown

    def __repr__(self):
        return self.shown

    def __deepcopy__(self, memo):
        return list(self)


def _plain(v) -> str:
    return f"{v:g}"


def _v(values) -> np.ndarray:
    """Whole numbers stay an array of whole numbers, as `np.array([1, 0, 0, 1])` is when the learner tries the call."""
    return np.array(values).view(_Vec)


# --- exercise 1: rotate_pairs -------------------------------------------------------------------------------------------------

_HEAD = "import numpy as np\n\n\ndef rotate_pairs(x, angles):\n"
_START = (_HEAD + "    # pair j is x[2*j] + 1j * x[2*j + 1]: turn it by angles[j],\n"
          "    # and give back a real array with the same layout as x\n    ...\n")


def _r(turn: str = "np.exp(1j * np.array(angles))", pairs: str = "x[0::2] + 1j * x[1::2]",
       back: str = "    out = []\n    for c in z:\n        out.append(c.real)\n        out.append(c.imag)\n    return np.array(out)\n") -> str:
    return _HEAD + f"    z = {pairs}\n    z = z * {turn}\n" + back


_HALVES = (_HEAD + "    half = len(x) // 2\n    z = (x[:half] + 1j * x[half:]) * np.exp(1j * np.array(angles))\n"
           "    return np.concatenate([z.real, z.imag])\n")


def _close(got, want) -> bool:
    if np.iscomplexobj(np.asarray(got)):
        return False
    try:
        g, w = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    except (TypeError, ValueError):
        return False
    return g.shape == w.shape and np.allclose(g, w, atol=1e-9)


def _show(v) -> str:
    try:
        a = np.asarray(v)
    except Exception:
        return _show_rounded(v, 4)
    if np.iscomplexobj(a):
        return "complex numbers " + _show_rounded([complex(round(c.real, 4), round(c.imag, 4)) for c in a.ravel()], None)
    if a.ndim > 1:   # grouped, such as one row per pair: shown as it is, so it never looks the same as a flat answer
        return "[" + ", ".join(_show(row) for row in a) + "]"
    try:
        return "[" + ", ".join(f"{round(float(c), 4) + 0.0:g}" for c in a.ravel()) + "]"
    except (TypeError, ValueError):
        return _show_rounded(v, 4)


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# The example of the lesson, np.array([1, 0, 0, 1]), holds whole numbers, and so do two more cases: a function that
# writes its floats into an array made from x then loses their decimals. The other cases hold floats.
_VISIBLE = (_v([1, 0, 0, 1]), _Angles([math.pi / 2, math.pi / 2], "[np.pi / 2, np.pi / 2]"))
_HIDDEN = [
    (_v([2, 1]), _Angles([math.pi / 3], "[np.pi / 3]")),
    (_v([1, 2, 3, 4, 5, 6]), _Angles([0.3, -1.2, 2.5], "[0.3, -1.2, 2.5]")),
    (_v([0.5, -1.5, 2, 0]), _Angles([0, math.pi], "[0, np.pi]")),
    (_v([3, 4, -1, 2, 0.25, 0.75, 1, 1]), _Angles([1, 2, 3, 4], "[1, 2, 3, 4]")),
]

# A wrong version that keeps the type of x: with whole numbers in x, NumPy cuts off the decimals of the turned values.
_WHOLE = _r(back="    out = np.zeros_like(x)\n    out[0::2] = z.real\n    out[1::2] = z.imag\n    return out\n")


def _array(got):
    try:
        return np.asarray(got)
    except Exception:
        return None


def _rows(got, text) -> bool:
    """The numbers come back with rows, such as one row per pair, instead of as one flat array like x."""
    a = _array(got) if got is not None else None
    return a is not None and not np.iscomplexobj(a) and a.ndim > 1


def _complex(got, text) -> bool:
    a = _array(got) if got is not None else None
    return a is not None and np.iscomplexobj(a)


EXERCISES = {
    "rotate-pairs": FunctionBox(
        name="rotate-pairs",
        function="rotate_pairs",
        start=_START,
        solution=_r(),
        visible=[_VISIBLE],
        hidden=_HIDDEN,
        show=_show,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("list-times", error=lambda e: isinstance(e, TypeError) and "can't multiply sequence" in str(e),
                    en="Your function stops with an error: somewhere a list, such as `angles`, is multiplied by a number like `1j`. Python cannot multiply a list by `1j` or by a float; NumPy can do it with an array. Make the angles an array first: `np.array(angles)`.",
                    nl="Je functie stopt met een fout: ergens wordt een lijst, zoals `angles`, vermenigvuldigd met een getal als `1j`. Python kan een lijst niet met `1j` of met een float vermenigvuldigen; NumPy kan dat wel met een array. Maak van de hoeken eerst een array: `np.array(angles)`."),
            Mistake("j-name", error=lambda e: isinstance(e, NameError) and "'j'" in str(e),
                    en="Python does not know `j` on its own: a plain `j` is a name, and there is no such name here. The imaginary unit is `1j`, with the 1, as in `3 + 4j`.",
                    nl="Python kent `j` niet op zichzelf: een losse `j` is een naam, en die naam bestaat hier niet. De imaginaire eenheid is `1j`, met de 1, zoals in `3 + 4j`."),
            Mistake("clockwise", wrong=_r(turn="np.exp(-1j * np.array(angles))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your pairs turn the other way, clockwise. Multiplying by `np.exp(1j * angle)` turns counterclockwise, from the real axis towards the imaginary axis, as long as the first number of each pair is the real part: `x[0::2] + 1j * x[1::2]`. A minus sign in the exponent, the minus sign on the other sine of a rotation matrix, or the two numbers of each pair the other way round, turns clockwise.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je paren draaien de andere kant op, met de klok mee. Vermenigvuldigen met `np.exp(1j * angle)` draait tegen de klok in, van de reële as naar de imaginaire as, zolang het eerste getal van elk paar het reële deel is: `x[0::2] + 1j * x[1::2]`. Een minteken in de exponent, het minteken bij de andere sinus van een rotatiematrix, of de twee getallen van elk paar andersom, draait met de klok mee."),
            Mistake("degrees", wrong=_r(turn="np.exp(1j * np.radians(angles))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you read the angles as degrees. They are in radians already, as `np.cos`, `np.sin` and the angle in `np.exp(1j * angle)` need them: use them as they are.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je leest de hoeken als graden. Ze zijn al in radialen, zoals `np.cos`, `np.sin` en de hoek in `np.exp(1j * angle)` ze nodig hebben: gebruik ze zoals ze zijn."),
            Mistake("no-i", wrong=_r(turn="np.exp(np.array(angles))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your pairs grow or shrink, but do not turn. `np.exp(angle)` is e to the power of the angle, a real number. A turn needs the imaginary unit in the exponent: `np.exp(1j * angle)`, a number of length 1.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je paren worden groter of kleiner, maar draaien niet. `np.exp(angle)` is e tot de macht de hoek, een reëel getal. Een draaiing vraagt de imaginaire eenheid in de exponent: `np.exp(1j * angle)`, een getal met lengte 1."),
            Mistake("halves", wrong=_HALVES,
                    en="`{call}` gives `{got}`, and it should give `{want}`: you pair the first half of `x` with the second half. Some RoPE code does that too, but here a pair is two entries next to each other: `x[0]` with `x[1]`, `x[2]` with `x[3]`, and so on, so the real parts are `x[0::2]` and the imaginary parts `x[1::2]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je koppelt de eerste helft van `x` aan de tweede helft. Sommige RoPE-code doet dat ook, maar hier is een paar twee getallen naast elkaar: `x[0]` met `x[1]`, `x[2]` met `x[3]`, enzovoort, dus de reële delen zijn `x[0::2]` en de imaginaire delen `x[1::2]`."),
            Mistake("whole-numbers", wrong=_WHOLE,
                    en="`{call}` gives `{got}`, and it should give `{want}`: your result holds only whole numbers. It is an array made from `x`, which holds whole numbers in this call, so NumPy cuts off the decimals of every value you put into it. Collect the turned values in a new list, and make an array of that at the end.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je uitkomst bevat alleen hele getallen. Het is een array die je van `x` hebt gemaakt, en `x` bevat in deze aanroep hele getallen, dus NumPy laat de cijfers achter de komma vallen van elke waarde die je erin zet. Verzamel de gedraaide waarden in een nieuwe lijst, en maak daar aan het eind een array van."),
            Mistake("complex-out", matches=_complex,
                    en="`{call}` gives complex numbers, and it should give a real vector with the same layout as `x`. Put each turned number back as two real numbers: its `.real` at the even place of its pair, and its `.imag` right after it.",
                    nl="`{call}` geeft complexe getallen, en het moet een reële vector geven met dezelfde indeling als `x`. Zet elk gedraaid getal terug als twee reële getallen: zijn `.real` op de even plek van zijn paar, en zijn `.imag` direct daarna."),
            Mistake("rows", matches=_rows,
                    en="`{call}` gives `{got}`: the numbers come back in rows. It should give `{want}`: one flat array with the same shape as `x`, the two numbers of each pair next to each other.",
                    nl="`{call}` geeft `{got}`: de getallen komen terug in rijen. Het moet `{want}` zijn: één platte array met dezelfde shape als `x`, de twee getallen van elk paar naast elkaar."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with your function: make a complex number of each pair, multiply it by `np.exp(1j * angle)` with its own angle, and put the real and imaginary parts back in place.",
                      "nl": "Vervang de `...` door je functie: maak van elk paar een complex getal, vermenigvuldig het met `np.exp(1j * angle)` met zijn eigen hoek, en zet de reële en imaginaire delen terug op hun plek."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with a vector and a list of angles, and the `def` line of your function has another number of names between its parentheses. Keep `(x, angles)`.",
                          "nl": "De check roept `{call}` aan, met een vector en een lijst hoeken, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(x, angles)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does each pair of entries next to each other become one complex number, turned by its own angle, and come back at the same two places?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Wordt elk paar getallen naast elkaar één complex getal, gedraaid over zijn eigen hoek, en komt het terug op dezelfde twee plekken?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for any number of pairs, and for angles that differ per pair?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt het voor elk aantal paren, en voor hoeken die per paar verschillen?"},
            "correct": {"en": "Your function is right in all {n} cases: each pair turns by its own angle, and keeps its length.",
                        "nl": "Je functie klopt in alle {n} gevallen: elk paar draait over zijn eigen hoek, en houdt zijn lengte."},
        },
    ),
}

# The lesson is done when these are correct. "relative" is the choice question, "real-dot" the open question.
REQUIRED = ["rotate-pairs", "relative", "real-dot"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_ROTMAT = (_HEAD + "    out = []\n    for j, a in enumerate(angles):\n        r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])\n"
           "        out.extend(r @ x[2 * j:2 * j + 2])\n    return np.array(out)\n")

CASES = [
    ("rotate-pairs", _START, "empty", None),
    ("rotate-pairs", _r(), "correct", None),
    ("rotate-pairs", _HEAD + "    out = np.array(x, dtype=float)\n    for j, a in enumerate(angles):\n        c, s = np.cos(a), np.sin(a)\n"
     "        out[2 * j] = c * x[2 * j] - s * x[2 * j + 1]\n        out[2 * j + 1] = s * x[2 * j] + c * x[2 * j + 1]\n    return out\n", "correct", None),
    ("rotate-pairs", _ROTMAT, "correct", None),
    ("rotate-pairs", _r(back="    return np.stack([z.real, z.imag], axis=1).reshape(-1)\n"), "correct", None),
    ("rotate-pairs", _r(back="    out = np.zeros(len(x))\n    out[0::2] = z.real\n    out[1::2] = z.imag\n    return out\n"), "correct", None),
    ("rotate-pairs", _HEAD + "    z = np.asarray(x, dtype=float).view(complex) * np.exp(1j * np.asarray(angles))\n    return z.view(float)\n", "correct", None),
    ("rotate-pairs", _HEAD + "    out = []\n    for j in range(len(angles)):\n        c = complex(x[2 * j], x[2 * j + 1]) * complex(np.cos(angles[j]), np.sin(angles[j]))\n"
     "        out += [c.real, c.imag]\n    return out\n", "correct", None),
    ("rotate-pairs", _r(turn="np.exp(-1j * np.array(angles))"), "wrong", "clockwise"),
    ("rotate-pairs", _ROTMAT.replace("-np.sin(a)], [np.sin(a)", "np.sin(a)], [-np.sin(a)"), "wrong", "clockwise"),
    ("rotate-pairs", _r(pairs="x[1::2] + 1j * x[0::2]", back="    out = np.zeros(len(x))\n    out[1::2] = z.real\n    out[0::2] = z.imag\n    return out\n"), "wrong", "clockwise"),
    ("rotate-pairs", _r(turn="np.exp(1j * np.radians(angles))"), "wrong", "degrees"),
    ("rotate-pairs", _r(turn="np.exp(np.array(angles))"), "wrong", "no-i"),
    ("rotate-pairs", _HALVES, "wrong", "halves"),
    ("rotate-pairs", _WHOLE, "wrong", "whole-numbers"),
    ("rotate-pairs", _r(back="    return z\n"), "wrong", "complex-out"),
    ("rotate-pairs", _r(back="    return np.stack([z.real, z.imag], axis=1)\n"), "wrong", "rows"),
    ("rotate-pairs", _r(back="    return [(c.real, c.imag) for c in z]\n"), "wrong", "rows"),
    ("rotate-pairs", _r(turn="np.exp(1j * angles)"), "error", "list-times"),
    ("rotate-pairs", _r(pairs="x[0::2] + j * x[1::2]"), "error", "j-name"),
    ("rotate-pairs", _r(turn="np.exp(1j * angles[0])"), "almost", None),
    ("rotate-pairs", _HEAD + "    return x\n", "wrong", None),
]
