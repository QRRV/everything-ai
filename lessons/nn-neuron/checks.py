# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nn-neuron: a neuron with a step as its activation function, and choosing its weights and
bias term so that it fires for one corner of the square. And the pictures: a neuron as a diagram, and the four
corners of the square with the boundary of a neuron whose bias term moves on a slider."""
import html
import re

from everything_ai.check import CodeBox, FunctionBox, Mistake

CORNERS = [(0, 0), (0, 1), (1, 0), (1, 1)]
BIASES = [round(-2.5 + 0.1 * k, 1) for k in range(31)]   # the slider: b from -2.5 to 0.5, with w = (1, 1)


def fires(b: float, w=(1, 1)) -> list[int]:
    """1 for each corner where w · x + b is above 0, else 0."""
    return [1 if w[0] * x1 + w[1] * x2 + b > 0 else 0 for x1, x2 in CORNERS]


def diagram(labels: dict[str, str]) -> str:
    """A neuron with two inputs: the inputs on the left, an arrow with its weight from each, the sum with the bias
    term, the activation function, and the output on the right. Weights and bias term orange, the output blue."""
    W, H = 600, 200
    p = []

    def text(x, y, s, style="font-size:15px;fill:var(--ea-ink)", anchor="middle"):
        p.append(f'<text x="{x}" y="{y}" style="{style};text-anchor:{anchor}">{s}</text>')

    def arrow(x0, y0, x1, y1, colour="var(--ea-ink-2)"):
        import math
        L = math.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        hx, hy = x1 - 9 * ux, y1 - 9 * uy
        p.append(f'<line x1="{x0}" y1="{y0}" x2="{hx:.1f}" y2="{hy:.1f}" style="stroke:{colour};stroke-width:1.8"/>'
                 f'<polygon points="{x1},{y1} {hx - 4.5 * uy:.1f},{hy + 4.5 * ux:.1f} {hx + 4.5 * uy:.1f},{hy - 4.5 * ux:.1f}" style="fill:{colour}"/>')

    for y, name in [(50, "x₁"), (150, "x₂")]:   # the inputs
        p.append(f'<circle cx="40" cy="{y}" r="22" style="fill:var(--ea-inset);stroke:var(--ea-ink-3);stroke-width:1.2"/>')
        text(40, y + 5, name)
    arrow(62, 50, 238, 92, "var(--ea-param)")
    arrow(62, 150, 238, 108, "var(--ea-param)")
    text(150, 58, "w₁", "font-size:15px;font-weight:600;fill:var(--ea-param-text)")
    text(150, 156, "w₂", "font-size:15px;font-weight:600;fill:var(--ea-param-text)")
    p.append('<circle cx="270" cy="100" r="32" style="fill:var(--ea-inset);stroke:var(--ea-ink-3);stroke-width:1.2"/>')
    text(270, 106, "Σ", "font-size:22px;fill:var(--ea-ink)")
    text(270, 158, "+ b", "font-size:15px;font-weight:600;fill:var(--ea-param-text)")
    text(270, 178, html.escape(labels["sum"]), "font-size:12.5px;fill:var(--ea-ink-2)")
    arrow(302, 100, 378, 100)
    text(340, 90, "z", "font-size:15px;font-style:italic;fill:var(--ea-ink)")
    p.append('<rect x="380" y="72" width="76" height="56" rx="8" style="fill:var(--ea-inset);stroke:var(--ea-ink-3);stroke-width:1.2"/>')
    text(418, 106, "f", "font-size:20px;font-style:italic;fill:var(--ea-ink)")
    text(418, 158, html.escape(labels["activation"]), "font-size:12.5px;fill:var(--ea-ink-2)")
    arrow(456, 100, 540, 100, "var(--ea-pred)")
    text(568, 106, "ŷ", "font-size:18px;font-weight:600;fill:var(--ea-pred-text)")
    text(568, 132, html.escape(labels["output"]), "font-size:12.5px;fill:var(--ea-ink-2)")
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:min(40rem,{W}px)">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["alt"])}</title>{"".join(p)}</svg></figure>')


def square(b: float, labels: dict[str, str]) -> str:
    """The four corners of the square, filled where the neuron with w = (1, 1) and this b fires and open where it does
    not, and its boundary x₁ + x₂ + b = 0 as a line; beside it, how many corners fire for every b on the slider. Drawn
    as SVG text, once for every stop of the slider (L.explore); dragging across the second chart sets b."""
    from everything_ai import chart
    p = chart.Panel((-0.3, 1.3), (-0.3, 1.3), title=labels["title"], xlabel="x₁", ylabel="x₂", xticks=[0, 1])
    lo, hi = -0.3, 1.3
    p.line([lo, hi], [-b - lo, -b - hi], "pred", 2.4)   # x₂ = −b − x₁
    on = [c for c, f in zip(CORNERS, fires(b)) if f]
    off = [c for c, f in zip(CORNERS, fires(b)) if not f]
    p.points([x for x, _ in on], [y for _, y in on], "data", 7)
    p.points([x for x, _ in off], [y for _, y in off], "data", 7, hollow=True)
    # beside it, for every b on the slider: how many corners fire
    k = chart.Panel((-2.6, 0.6), (-0.3, 4.3), title=labels["count_title"], xlabel="b", ylabel=labels["count"], yticks=[0, 1, 2, 3, 4])
    k.line([b, b], [-0.3, 4.3], "ink-3", 1.2)
    k.points(BIASES, [sum(fires(v)) for v in BIASES], "pred", 3)   # dots, not a line: the count jumps between two stops
    k.points([b], [sum(fires(b))], "pred", 6)
    return chart.figure([p, k], alt=labels["alt"], width=330, height=320, drag=1)   # two fit side by side in a play block


# The lesson is done when these are correct. "which-activation" is the choice question in the lesson.
REQUIRED = ["which-activation", "step-neuron", "one-corner"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_S_HEAD = "import numpy as np\n\n\ndef step_neuron(x, w, b):\n"
_S_START = _S_HEAD + "    z = ...\n    ...\n"
_S_BODY = "    z = np.dot(w, x) + b\n    if z > 0:\n        return 1\n    else:\n        return 0"
_S = _S_HEAD + _S_BODY


def _s(body: str) -> str:
    return _S_HEAD + body


_C_HEAD = "import numpy as np\n\n"
_C_LOOP = ("outputs = []\nfor x in [[0, 0], [0, 1], [1, 0], [1, 1]]:\n    z = np.dot(w, x) + b\n    if z > 0:\n        outputs.append(1)\n"
           "    else:\n        outputs.append(0)\noutputs")
_C_START = _C_HEAD + "w = ...\nb = ...\n\n" + _C_LOOP + "\n"
_C = _C_HEAD + "w = np.array([1, -1])\nb = -0.5\n\n" + _C_LOOP


def _c(w: str, b: str) -> str:
    return _C_HEAD + f"w = {w}\nb = {b}\n\n" + _C_LOOP


def _outputs(want):
    return lambda got, text: isinstance(got, list) and got == want


EXERCISES = {
    "step-neuron": FunctionBox(
        name="step-neuron",
        function="step_neuron",
        start=_S_START,
        solution=_S,
        visible=[([1, 1], [1, 1], -1.5)],
        hidden=[([0, 1], [1, 1], -1.5), ([1, 0], [1, 1], -1.0), ([0.5, 2.0, 1.0], [2.0, -1.0, 2.0], 0.4), ([3, 1], [0.5, -2], 0.0),
                ([2], [-1], 1.5), ([1, 1], [0.5, -1.0], -1.0)],
        first=[_DOTS],
        mistakes=[
            Mistake("no-activation", wrong=_s("    z = np.dot(w, x) + b\n    return z"),
                    en="`{call}` gives `{got}`: the weighted sum plus the bias term, `z` itself. A neuron with a step gives back 1 when `z` is above 0, and 0 otherwise.",
                    nl="`{call}` geeft `{got}`: de gewogen som plus de bias term, `z` zelf. Een neuron met een stap geeft 1 terug als `z` boven 0 ligt, en anders 0."),
            Mistake("sigmoid", wrong=_s("    z = np.dot(w, x) + b\n    return 1 / (1 + np.exp(-z))"),
                    en="`{call}` gives `{got}`: the sigmoid of `z`, a probability. Here the activation function is a step: 1 when `z` is above 0, and 0 otherwise.",
                    nl="`{call}` geeft `{got}`: de sigmoid van `z`, een kans. Hier is de activation function een stap: 1 als `z` boven 0 ligt, en anders 0."),
            Mistake("at-zero", state="almost", wrong=_s("    z = np.dot(w, x) + b\n    if z >= 0:\n        return 1\n    else:\n        return 0"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: there `z` is exactly 0. The neuron fires only when `z` is above 0: `z > 0`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: daar is `z` precies 0. Het neuron vuurt alleen als `z` boven 0 ligt: `z > 0`."),
            Mistake("no-bias", wrong=_s("    z = np.dot(w, x)\n    if z > 0:\n        return 1\n    else:\n        return 0"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you leave out the bias term. `z` is the weighted sum plus `b`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je laat de bias term weg. `z` is de gewogen som plus `b`."),
            Mistake("minus-bias", wrong=_s("    z = np.dot(w, x) - b\n    if z > 0:\n        return 1\n    else:\n        return 0"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you subtract the bias term. `z` is the weighted sum plus `b`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je trekt de bias term af. `z` is de gewogen som plus `b`."),
            Mistake("true-false", matches=lambda got, text: type(got).__name__ in ("bool", "bool_"),
                    en="`{call}` gives `{got}`: a comparison such as `z > 0` gives `True` or `False`. A neuron with a step gives back the numbers 1 and 0: use an `if` with `return 1` and `return 0`.",
                    nl="`{call}` geeft `{got}`: een vergelijking zoals `z > 0` geeft `True` of `False`. Een neuron met een stap geeft de getallen 1 en 0 terug: gebruik een `if` met `return 1` en `return 0`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the first `...` with the weighted sum plus the bias term, and the second with the step: give back 1 when `z` is above 0, and 0 otherwise.",
                      "nl": "Vervang de eerste `...` door de gewogen som plus de bias term, en de tweede door de stap: geef 1 terug als `z` boven 0 ligt, en anders 0."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with three values, and the `def` line of your function has a different number of names between its parentheses. Keep `(x, w, b)`.",
                          "nl": "De check roept `{call}` aan, met drie waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(x, w, b)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Is `z` the dot product of `w` and `x` plus `b`, and do you give back 1 when `z` is above 0, and 0 otherwise?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Is `z` het inproduct van `w` en `x` plus `b`, en geef je 1 terug als `z` boven 0 ligt, en anders 0?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Do you work it out from `x`, `w` and `b` themselves, for any number of inputs?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken je het uit met `x`, `w` en `b` zelf, voor elk aantal inputs?"},
            "correct": {"en": "Your function is right in all {n} cases. With the weights 1 and 1 and the bias term −1.5 it fires only for (1, 1): it computes AND.",
                        "nl": "Je functie klopt in alle {n} gevallen. Met de weights 1 en 1 en de bias term −1,5 vuurt hij alleen bij (1, 1): hij rekent AND uit."},
        },
    ),
    "one-corner": CodeBox(
        name="one-corner",
        start=_C_START,
        solution=_C,
        keep=["for x in [[0, 0], [0, 1], [1, 0], [1, 1]]:", "z = np.dot(w, x) + b", "if z > 0:"],
        first=[
            _DOTS,
            Mistake("typed", matches=lambda got, text: re.search(r"^[^#\n]*outputs\s*=\s*\[\s*\d|^\s*\[\s*\d", text, re.MULTILINE) is not None,
                    en="You type the outputs in yourself. Let the neuron work them out: choose only `w` and `b`, and leave the loop as it is.",
                    nl="Je typt de uitkomsten zelf in. Laat het neuron ze uitrekenen: kies alleen `w` en `b`, en laat de loop zoals hij is."),
        ],
        mistakes=[
            Mistake("swapped", matches=_outputs([0, 1, 0, 0]),
                    en="Your last line gives `{got}`: the neuron fires for (0, 1), not for (1, 0). Swap the two weights: the first input should count for, the second against.",
                    nl="Je laatste regel geeft `{got}`: het neuron vuurt bij (0, 1), niet bij (1, 0). Wissel de twee weights om: de eerste input moet meetellen, de tweede ertegen."),
            Mistake("and", matches=_outputs([0, 0, 0, 1]),
                    en="Your last line gives `{got}`: that is AND, it fires for (1, 1). With (1, 1) the second input should pull `z` below 0: give it a negative weight.",
                    nl="Je laatste regel geeft `{got}`: dat is AND, het vuurt bij (1, 1). Bij (1, 1) moet de tweede input `z` onder 0 trekken: geef hem een negatieve weight."),
            Mistake("both", matches=_outputs([0, 0, 1, 1]),
                    en="Your last line gives `{got}`: it fires whenever the first input is 1, also for (1, 1). The second input should pull `z` below 0: give it a negative weight.",
                    nl="Je laatste regel geeft `{got}`: het vuurt zodra de eerste input 1 is, ook bij (1, 1). De tweede input moet `z` onder 0 trekken: geef hem een negatieve weight."),
            Mistake("origin", matches=lambda got, text: isinstance(got, list) and len(got) == 4 and got[0] == 1,
                    en="Your last line gives `{got}`: the neuron fires for (0, 0). There both inputs are 0, so `z` is just `b`: the bias term should be 0 or below.",
                    nl="Je laatste regel geeft `{got}`: het neuron vuurt bij (0, 0). Daar zijn beide inputs 0, dus `z` is gewoon `b`: de bias term moet 0 of lager zijn."),
            Mistake("never", matches=_outputs([0, 0, 0, 0]),
                    en="Your last line gives `{got}`: the neuron never fires. For (1, 0), `z` is the first weight plus `b`: that should be above 0.",
                    nl="Je laatste regel geeft `{got}`: het neuron vuurt nooit. Bij (1, 0) is `z` de eerste weight plus `b`: dat moet boven 0 liggen."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...` with weights `w`, two numbers, and a bias term `b`, so that the neuron fires only for (1, 0).",
                      "nl": "Vervang de twee `...` door weights `w`, twee getallen, en een bias term `b`, zodat het neuron alleen bij (1, 0) vuurt."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. Choose only `w` and `b`.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. Kies alleen `w` en `b`."},
            "no_value": {"en": "The last line of the box shows nothing. Put `outputs` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `outputs` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`, and it should be `[0, 0, 1, 0]`: only (1, 0) fires. Which input should count for, which against, and where should `b` put the line?",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet `[0, 0, 1, 0]` zijn: alleen (1, 0) vuurt. Welke input moet meetellen, welke ertegen, en waar moet `b` de lijn leggen?"},
            "correct": {"en": "Only (1, 0) fires. The first input counts for, the second against, and the bias term keeps `z` at 0 or below for (0, 0). One neuron, one straight line: the next lesson shows what that cannot do.",
                        "nl": "Alleen (1, 0) vuurt. De eerste input telt mee, de tweede ertegen, en de bias term houdt `z` bij (0, 0) op 0 of eronder. Eén neuron, één rechte lijn: de volgende les laat zien wat dat niet kan."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("step-neuron", _S_START, "empty", None),
    ("step-neuron", _S, "correct", None),
    ("step-neuron", _s("    z = w @ np.array(x) + b\n    if z > 0:\n        return 1\n    return 0"), "correct", None),
    ("step-neuron", _s("    z = np.dot(w, x) + b\n    if z <= 0:\n        return 0\n    return 1"), "correct", None),
    ("step-neuron", _s("    z = np.dot(w, x) + b\n    return z"), "wrong", "no-activation"),
    ("step-neuron", _s("    z = np.dot(w, x) + b\n    return 1 / (1 + np.exp(-z))"), "wrong", "sigmoid"),
    ("step-neuron", _s("    z = np.dot(w, x) + b\n    if z >= 0:\n        return 1\n    else:\n        return 0"), "almost", "at-zero"),
    ("step-neuron", _s("    z = np.dot(w, x)\n    if z > 0:\n        return 1\n    else:\n        return 0"), "wrong", "no-bias"),
    ("step-neuron", _s("    z = x[0] * w[0] + x[1] * w[1] + b\n    if z > 0:\n        return 1\n    else:\n        return 0"), "almost", None),
    ("step-neuron", _s("    z = np.dot(w, x) - b\n    if z > 0:\n        return 1\n    else:\n        return 0"), "wrong", "minus-bias"),
    ("step-neuron", _s("    z = np.dot(w, x) + b\n    return z > 0"), "wrong", "true-false"),
    ("one-corner", _C_START, "empty", None),
    ("one-corner", _C, "correct", None),
    ("one-corner", _c("[2, -2]", "-1"), "correct", None),
    ("one-corner", _c("np.array([1, -1])", "0"), "correct", None),
    ("one-corner", _c("np.array([-1, 1])", "-0.5"), "wrong", "swapped"),
    ("one-corner", _c("np.array([1, 1])", "-1.5"), "wrong", "and"),
    ("one-corner", _c("np.array([1, 0])", "-0.5"), "wrong", "both"),
    ("one-corner", _c("np.array([1, -1])", "0.5"), "wrong", "origin"),
    ("one-corner", _c("np.array([1, -1])", "-1.5"), "wrong", "never"),
    ("one-corner", _C.replace("w = np.array([1, -1])\nb = -0.5\n", "w = np.array([1, -1])\nb = -0.5\noutputs = [0, 0, 1, 0]\n"), "wrong", "typed"),
    ("one-corner", _c("np.array([0, 0])", "0")[:-len("outputs")] + "[0, 0, 1, 0]", "wrong", "typed"),
]
