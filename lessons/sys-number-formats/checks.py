# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-number-formats: spacing, the step between the numbers a floating-point format can
store in the power of 2 that a number lies in, from its number of mantissa bits. And the picture of the bits of FP32,
TF32, BF16 and FP16."""
import html
import math
import re

from everything_ai.check import FunctionBox, Mistake

# Sign, exponent and mantissa bits of each format.
FORMATS = [("FP32", 1, 8, 23), ("TF32", 1, 8, 10), ("BF16", 1, 8, 7), ("FP16", 1, 5, 10)]


def bits_picture(labels: dict) -> str:
    """Each format as a row of boxes, one per bit: the sign filled in ink, the exponent filled in grey, the mantissa
    open. Ink and grey only, since nothing here is learned or a loss. The name and the count stand above the boxes, so
    the 32 boxes of FP32 fit in a narrow window."""
    def box(kind):
        if kind == "man":
            return '<span style="display:inline-block;width:5px;height:14px;margin-right:1px;border:1px solid var(--ea-ink-2);box-sizing:border-box"></span>'
        colour = "var(--ea-ink)" if kind == "sign" else "var(--ea-ink-3)"
        return f'<span style="display:inline-block;width:5px;height:14px;margin-right:1px;background:{colour}"></span>'
    rows = []
    for name, s, e, m in FORMATS:
        boxes = box("sign") * s + '<span style="display:inline-block;width:4px"></span>' + box("exp") * e + '<span style="display:inline-block;width:4px"></span>' + box("man") * m
        note = html.escape(labels.get(f"note_{name}", ""))
        rows.append(f'<div style="margin:0.35rem 0"><div><span style="color:var(--ea-ink);font-weight:600">{name}</span>'
                    f' <span style="color:var(--ea-ink-2);font-size:0.8rem">{s + e + m} {html.escape(labels["bits"])}{note}</span></div>'
                    f'<div style="white-space:nowrap;line-height:1">{boxes}</div></div>')
    key = "".join(f'<span style="margin-right:1rem;white-space:nowrap">{box(k)}<span style="margin-left:0.3rem">{html.escape(labels[k])}</span></span>'
                  for k in ("sign", "exp", "man"))
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(labels["alt"])}" '
            f'style="margin:0.25rem 0 0.75rem;max-width:640px;font-family:var(--ea-font-ui);font-size:0.875rem">'
            f'<div style="font-size:1rem;font-weight:600;color:var(--ea-ink);margin-bottom:0.4rem">{html.escape(labels["title"])}</div>'
            f'{"".join(rows)}<div style="color:var(--ea-ink-2);font-size:0.8rem;margin-top:0.5rem">{key}</div></figure>')


# --- exercise 1: spacing ------------------------------------------------------------------------------------------------------

_HEAD = "import math\n\n\ndef spacing(x, mantissa_bits):\n"
_START = (_HEAD + "    # the power of 2 that x lies in: 2 to the power floor(log2 |x|)\n    ...\n"
          "    # the mantissa splits that power into 2 ** mantissa_bits equal steps\n    ...\n")


def _s(power: str = "math.floor(math.log2(abs(x)))", step: str = "2.0 ** (power - mantissa_bits)") -> str:
    return _HEAD + f"    power = {power}\n    return {step}\n"


def _close(got, want) -> bool:
    try:
        return math.isclose(float(got), float(want), rel_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _show(v) -> str:
    try:
        return f"{float(v):g}"
    except (TypeError, ValueError):
        return repr(v)


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# The steps grow with x itself, smoothly: the relative step, true only at a power of 2.
_SCALED = dict(
    en="`{call}` gives `{got}`, and it should give `{want}`: you scale the step with `x` itself. All steps between `2 ** p` and `2 ** (p + 1)` are equal, and they only double at the next power of 2. Take the power of 2 that `x` lies in, rounded down.",
    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je schaalt de stap met `x` zelf. Alle stappen tussen `2 ** p` en `2 ** (p + 1)` zijn even groot, en ze verdubbelen pas bij de volgende macht van 2. Neem de macht van 2 waarin `x` ligt, naar beneden afgerond.")

EXERCISES = {
    "spacing": FunctionBox(
        name="spacing",
        function="spacing",
        start=_START,
        solution=_s(),
        visible=[(70000.0, 7)],
        hidden=[(1.0, 7), (3.0, 23), (-0.3, 10), (60000.0, 10), (0.001, 7), (1e-20, 23)],
        show=_show,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("no-abs", error=lambda e: isinstance(e, ValueError) and "math domain error" in str(e),
                    en="With a negative `x`, `math.log2` stops: the log of a negative number does not exist. The spacing depends only on the size of the number, so take `math.log2(abs(x))`.",
                    nl="Met een negatieve `x` stopt `math.log2`: de log van een negatief getal bestaat niet. De afstand hangt alleen af van de grootte van het getal, dus neem `math.log2(abs(x))`."),
            Mistake("log10", wrong=_s(power="math.floor(math.log10(abs(x)))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you take a power of 10, and a float stores its exponent as a power of 2. Use `math.log2`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je neemt een macht van 10, en een float slaat zijn exponent op als macht van 2. Gebruik `math.log2`."),
            Mistake("ceil", wrong=_s(power="math.ceil(math.log2(abs(x)))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you round the power up, so you get the steps of the next power of 2, twice as big. `x` lies between `2 ** p` and `2 ** (p + 1)` with `p` rounded down: `math.floor`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je rondt de macht naar boven af, dus krijg je de stappen van de volgende macht van 2, twee keer zo groot. `x` ligt tussen `2 ** p` en `2 ** (p + 1)` met `p` naar beneden afgerond: `math.floor`."),
            Mistake("round", wrong=_s(power="round(math.log2(abs(x)))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you round the power to the nearest whole number, so a number in the upper part of its power of 2, such as 3, gets the steps of the next power. Round down, with `math.floor`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je rondt de macht af op het dichtstbijzijnde gehele getal, dus krijgt een getal in het bovenste deel van zijn macht van 2, zoals 3, de stappen van de volgende macht. Rond naar beneden af, met `math.floor`."),
            Mistake("plus-one", wrong=_s(step="2.0 ** (power - mantissa_bits - 1)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your steps are half as big. The mantissa bits split one power of 2 into `2 ** mantissa_bits` steps, not twice as many.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je stappen zijn half zo groot. De mantissebits verdelen één macht van 2 in `2 ** mantissa_bits` stappen, niet twee keer zoveel."),
            Mistake("relative", wrong=_s(step="2.0 ** (-mantissa_bits)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the step near 1, from the table in *The idea*. Near a bigger or a smaller number the steps grow or shrink with its power of 2: multiply by `2 ** power`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is de stap bij 1, uit de tabel in *Het idee*. Bij een groter of kleiner getal groeien of krimpen de stappen met zijn macht van 2: vermenigvuldig met `2 ** power`."),
            Mistake("scaled", wrong=_s(step="abs(x) * 2.0 ** (-mantissa_bits)"), **_SCALED),
            Mistake("scaled-signed", wrong=_s(step="x * 2.0 ** (-mantissa_bits)"), **_SCALED),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: first the power of 2 that `x` lies in, then the size of one step of the mantissa in that power.",
                      "nl": "Vervang de twee `...`: eerst de macht van 2 waarin `x` ligt, dan de grootte van één stap van de mantisse in die macht."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with a number and a number of mantissa bits, and the `def` line of your function has another number of names between its parentheses. Keep `(x, mantissa_bits)`.",
                          "nl": "De check roept `{call}` aan, met een getal en een aantal mantissebits, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(x, mantissa_bits)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you take the power of 2 that `x` lies in, rounded down, and split it into `2 ** mantissa_bits` steps?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Neem je de macht van 2 waarin `x` ligt, naar beneden afgerond, en verdeel je die in `2 ** mantissa_bits` stappen?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for numbers below 1, for negative numbers, and for a power of 2 itself, such as 1?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt het voor getallen onder 1, voor negatieve getallen, en voor een macht van 2 zelf, zoals 1?"},
            "correct": {"en": "Your function is right in all {n} cases: the steps of a format grow with the power of 2 of the number, and shrink with every mantissa bit.",
                        "nl": "Je functie klopt in alle {n} gevallen: de stappen van een formaat groeien met de macht van 2 van het getal, en krimpen met elke mantissebit."},
        },
    ),
}

# The lesson is done when these are correct. "format" is the choice question, "lost-update" the open question.
REQUIRED = ["spacing", "format", "lost-update"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("spacing", _START, "empty", None),
    ("spacing", _s(), "correct", None),
    ("spacing", _HEAD + "    return 2 ** math.floor(math.log2(abs(x))) / 2 ** mantissa_bits\n", "correct", None),
    ("spacing", _HEAD + "    m, e = math.frexp(x)\n    return 2.0 ** (e - 1 - mantissa_bits)\n", "correct", None),
    # a loop instead of a log, and NumPy's own spacing of an FP32 or FP16 number where NumPy has the format
    ("spacing", _HEAD + "    p = 0\n    while 2 ** (p + 1) <= abs(x):\n        p += 1\n    while 2 ** p > abs(x):\n        p -= 1\n"
                "    return 2 ** p / 2 ** mantissa_bits\n", "correct", None),
    ("spacing", "import math\nimport numpy as np\n\n\ndef spacing(x, mantissa_bits):\n    if mantissa_bits == 23:\n        return float(np.spacing(np.float32(abs(x))))\n"
                "    if mantissa_bits == 10:\n        return float(np.spacing(np.float16(abs(x))))\n"
                "    return 2.0 ** (math.floor(math.log2(abs(x))) - mantissa_bits)\n", "correct", None),
    ("spacing", _HEAD + "    return math.ulp(abs(x)) * 2 ** (52 - mantissa_bits)\n", "correct", None),
    ("spacing", _s(power="math.floor(math.log2(x))"), "error", "no-abs"),
    ("spacing", _s(power="math.floor(math.log10(abs(x)))"), "wrong", "log10"),
    ("spacing", _s(power="math.ceil(math.log2(abs(x)))"), "wrong", "ceil"),
    ("spacing", _s(power="round(math.log2(abs(x)))"), "wrong", "round"),
    ("spacing", _s(step="2.0 ** (power - mantissa_bits - 1)"), "wrong", "plus-one"),
    ("spacing", _s(step="2.0 ** (-mantissa_bits)"), "wrong", "relative"),
    ("spacing", _HEAD + "    return abs(x) / 2 ** mantissa_bits\n", "wrong", "scaled"),
    ("spacing", _HEAD + "    return x * 2 ** -mantissa_bits\n", "wrong", "scaled-signed"),
    ("spacing", _HEAD + "    return 2.0 ** (int(math.log2(abs(x))) - mantissa_bits)\n", "almost", None),
]
