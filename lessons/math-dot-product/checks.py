# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson math-dot-product: a prediction with two features and np.dot, and the dot product as a loop in
plain Python. And the picture: a dot product worked out box by box."""
import html
import re

from everything_ai.check import Code, CodeBox, FunctionBox, Mistake, uses_loop


def picture(w: list, x: list, labels: dict[str, str]) -> str:
    """The weights above the features, the product of each pair below them, and the sum of the products: the dot
    product. Weights are orange, as parameters; the rest is ink, as a dot product has no colour of its own."""
    BW, BH, GAP, LEFT = 74, 40, 10, 112
    n = len(w)
    W = LEFT + n * (BW + GAP) + 150
    rows = [(labels["w"], w, "param"), (labels["x"], x, "ink"), (labels["product"], [a * b for a, b in zip(w, x)], "ink")]
    parts = []
    for r, (name, values, role) in enumerate(rows):
        y = 18 + r * (BH + 22)
        colour = "var(--ea-param-text)" if role == "param" else "var(--ea-ink)"
        stroke = "var(--ea-param)" if role == "param" else "var(--ea-axis)"
        parts.append(f'<text x="{LEFT - 12}" y="{y + BH / 2 + 5}" style="font-size:14px;fill:var(--ea-ink-2);text-anchor:end">{html.escape(name)}</text>')
        for i, v in enumerate(values):
            bx = LEFT + i * (BW + GAP)
            parts.append(f'<rect x="{bx}" y="{y}" width="{BW}" height="{BH}" rx="5" style="fill:var(--ea-page);stroke:{stroke};stroke-width:1.5"/>'
                         f'<text x="{bx + BW / 2}" y="{y + BH / 2 + 5}" style="font-size:15px;font-family:var(--ea-font-code);fill:{colour};text-anchor:middle">{_num(v)}</text>')
            if r == 1:
                parts.append(f'<text x="{bx + BW / 2}" y="{y - 6}" style="font-size:14px;fill:var(--ea-ink-3);text-anchor:middle">×</text>')
    y = 18 + 2 * (BH + 22)
    total = sum(a * b for a, b in zip(w, x))
    sx = LEFT + n * (BW + GAP) + 18
    parts.append(f'<text x="{sx}" y="{y + BH / 2 + 5}" style="font-size:15px;fill:var(--ea-ink-2)">{labels["sum"]}</text>'
                 f'<text x="{sx + 40}" y="{y + BH / 2 + 5}" style="font-size:17px;font-weight:600;fill:var(--ea-ink)">{_num(total)}</text>')
    H = y + BH + 8
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:min(100%,{W}px)">'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(labels["alt"])}" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["alt"])}</title>{"".join(parts)}</svg></figure>')


def _num(v) -> str:
    return (str(int(v)) if float(v).is_integer() else str(v)).replace("-", "−")


# The lesson is done when these are correct. "by-hand" is the choice question in the lesson.
REQUIRED = ["by-hand", "grade", "dot-loop"]

_HEAD = "import numpy as np\n\nw = np.array([0.5, 0.3])\nb = 1\nx = np.array([6, 8])\n"
_START = _HEAD + "y_pred = ...\ny_pred\n"
_SOLUTION = _HEAD + "y_pred = np.dot(w, x) + b\ny_pred"


def _grade(new: str) -> str:
    return _SOLUTION.replace("y_pred = np.dot(w, x) + b", new)


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_LOOP_START = "def dot(a, b):\n    ...\n"
_LOOP = "def dot(a, b):\n    total = 0\n    for i in range(len(a)):\n        total = total + a[i] * b[i]\n    return total"


def _loop(new: str) -> str:
    return "def dot(a, b):\n" + new


EXERCISES = {
    "grade": CodeBox(
        name="grade",
        start=_START,
        solution=_SOLUTION,
        keep=["w = np.array([0.5, 0.3])", "b = 1", "x = np.array([6, 8])"],
        # The check also runs the box with another student: a number typed in by hand does not change with it.
        others=[{"x": Code("np.array([10, 7])")}],
        first=[_DOTS],
        mistakes=[
            Mistake("no-bias", wrong=_grade("y_pred = np.dot(w, x)"),
                    en="Your last line gives `{got}`: the dot product alone. A prediction also adds the bias term `b`.",
                    nl="Je laatste regel geeft `{got}`: alleen het inproduct. Een voorspelling telt ook de bias term `b` op."),
            Mistake("no-sum", wrong=_grade("y_pred = w * x + b"),
                    en="Your last line gives `{got}`: an array, with one product per feature. A dot product adds the products up to one number: see step 3 of the recipe.",
                    nl="Je laatste regel geeft `{got}`: een array, met één product per feature. Een inproduct telt de producten op tot één getal: zie stap 3 van het recept."),
            Mistake("sums", wrong=_grade("y_pred = w.sum() * x.sum() + b"),
                    en="Your last line gives 12.2: the sum of the weights times the sum of the features. Each weight goes with its own feature: multiply pairs first, then add up.",
                    nl="Je laatste regel geeft 12,2: de som van de weights keer de som van de features. Elke weight hoort bij zijn eigen feature: eerst paren vermenigvuldigen, dan optellen."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the prediction: the dot product of `w` and `x`, plus `b`.",
                      "nl": "Vervang de `...` door de voorspelling: het inproduct van `w` en `x`, plus `b`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another student by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een andere leerling."},
            "no_value": {"en": "The last line of the box shows nothing. Put `y_pred` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `y_pred` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`, and it should be the dot product of `w` and `x`, plus `b`. Do you multiply each weight by its own feature, add up, and add `b`?",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet het inproduct van `w` en `x` zijn, plus `b`. Vermenigvuldig je elke weight met zijn eigen feature, tel je alles op, en tel je er `b` bij op?"},
            "other": {"en": "Right for this student. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you work it out with `w`, `x` and `b` themselves?",
                      "nl": "Goed voor deze leerling. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je het uit met `w`, `x` en `b` zelf?"},
            "other_error": {"en": "Right for this student. But with {where}, your code stops with an error. Do you work it out with `w`, `x` and `b` themselves?",
                            "nl": "Goed voor deze leerling. Maar met {where} stopt je code met een fout. Reken je het uit met `w`, `x` en `b` zelf?"},
            "correct": {"en": "0.5 × 6 + 0.3 × 8 = 3 + 2.4 = 5.4, and with the bias term 1 the model predicts a grade of 6.4. The check also tried a student with 10 hours of study and 7 of sleep: 8.1.",
                        "nl": "0,5 × 6 + 0,3 × 8 = 3 + 2,4 = 5,4, en met de bias term 1 voorspelt het model een cijfer van 6,4. De check probeerde ook een leerling die 10 uur leerde en 7 uur sliep: 8,1."},
        },
    ),
    "dot-loop": FunctionBox(
        name="dot-loop",
        function="dot",
        start=_LOOP_START,
        solution=_LOOP,
        visible=[([1, 2, 3], [4, 5, 6])],
        hidden=[([2, 0, 1], [3, 7, 5]), ([1.5], [4]), ([1, -1, 1, -1], [5, 5, 5, 5]), ([0.5, 0.3], [6, 8]), ([], [])],
        python_values=True,
        first=[_DOTS],
        form=[
            Mistake("no-loop", state="almost", matches=lambda got, text: not uses_loop(text),
                    en="Your function gives the right number in every case, but without a loop of its own: something else does the work. Write the loop yourself: go through the places, and add each product to a total.",
                    nl="Je functie geeft in elk geval het goede getal, maar zonder eigen loop: iets anders doet het werk. Schrijf de loop zelf: loop de plekken langs, en tel elk product bij een totaal op."),
        ],
        mistakes=[
            Mistake("sums", wrong=_loop("    return sum(a) * sum(b)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the sum of `a` times the sum of `b`. Multiply the numbers in the same place, `a[i] * b[i]`, and add up those products.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de som van `a` keer de som van `b`. Vermenigvuldig de getallen op dezelfde plek, `a[i] * b[i]`, en tel die producten op."),
            Mistake("added", wrong=_loop("    total = 0\n    for i in range(len(a)):\n        total = total + a[i] + b[i]\n    return total"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your function adds the pairs. A dot product multiplies them: `a[i] * b[i]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je functie telt de paren op. Een inproduct vermenigvuldigt ze: `a[i] * b[i]`."),
            Mistake("products", wrong=_loop("    out = []\n    for i in range(len(a)):\n        out.append(a[i] * b[i])\n    return out"),
                    en="`{call}` gives `{got}`: the products, one per place. A dot product is one number: add them up.",
                    nl="`{call}` geeft `{got}`: de producten, één per plek. Een inproduct is één getal: tel ze op."),
            Mistake("last-only", wrong=_loop("    total = 0\n    for i in range(len(a)):\n        total = a[i] * b[i]\n    return total"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: only the last product. Each round should add its product to the total: `total = total + ...`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: alleen het laatste product. In elke ronde moet het product bij het totaal: `total = total + ...`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a loop that multiplies the numbers in the same place and adds the products up, and give the total back with `return`.",
                      "nl": "Vervang de `...` door een loop die de getallen op dezelfde plek vermenigvuldigt en de producten optelt, en geef het totaal terug met `return`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with two lists, and the `def` line of your function has a different number of names between its parentheses. Keep `(a, b)`.",
                          "nl": "De check roept `{call}` aan, met twee lijsten, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(a, b)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you multiply the numbers in the same place, and add all the products up?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vermenigvuldig je de getallen op dezelfde plek, en tel je alle producten op?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work it out from the numbers in `a` and `b` themselves, for lists of any length?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Rekent je functie het uit met de getallen in `a` en `b` zelf, voor lijsten van elke lengte?"},
            "correct": {"en": "Your function is right in all {n} cases. `dot([1, 2, 3], [4, 5, 6])` is 4 + 10 + 18 = 32, the same as `np.dot` gives. With no numbers at all, the dot product is 0.",
                        "nl": "Je functie klopt in alle {n} gevallen. `dot([1, 2, 3], [4, 5, 6])` is 4 + 10 + 18 = 32, wat `np.dot` ook geeft. Zonder getallen is het inproduct 0."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("grade", _START, "empty", None),
    ("grade", _SOLUTION, "correct", None),
    ("grade", _grade("y_pred = (w * x).sum() + b"), "correct", None),
    ("grade", _grade("y_pred = w[0] * x[0] + w[1] * x[1] + b"), "correct", None),
    ("grade", _grade("y_pred = x.dot(w) + b"), "correct", None),
    ("grade", _grade("y_pred = np.dot(w, x)"), "wrong", "no-bias"),
    ("grade", _grade("y_pred = w * x + b"), "wrong", "no-sum"),
    ("grade", _grade("y_pred = w.sum() * x.sum() + b"), "wrong", "sums"),
    ("grade", _grade("y_pred = 6.4"), "almost", "other-value"),
    ("grade", _HEAD + "y_pred = np.dot(w, x) + b\nprint(y_pred)", "wrong", "print-last"),
    ("dot-loop", _LOOP_START, "empty", None),
    ("dot-loop", "import numpy as np\n\ndef dot(a, b):\n    return np.dot(a, b)", "almost", "no-loop"),
    ("dot-loop", _LOOP, "correct", None),
    ("dot-loop", _loop("    total = 0\n    for p, q in zip(a, b):\n        total += p * q\n    return total"), "correct", None),
    ("dot-loop", _loop("    return sum([a[i] * b[i] for i in range(len(a))])"), "correct", None),
    ("dot-loop", _loop("    return sum(a) * sum(b)"), "wrong", "sums"),
    ("dot-loop", _loop("    total = 0\n    for i in range(len(a)):\n        total = total + a[i] + b[i]\n    return total"), "wrong", "added"),
    ("dot-loop", _loop("    out = []\n    for i in range(len(a)):\n        out.append(a[i] * b[i])\n    return out"), "wrong", "products"),
    ("dot-loop", _loop("    total = 0\n    for i in range(len(a)):\n        total = a[i] * b[i]\n    return total"), "wrong", "last-only"),
    ("dot-loop", _loop("    return 32"), "almost", None),
]
