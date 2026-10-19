# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson math-matrix-multiply: the predictions for all students at once with @, and the shape rule as a
function. And the houses of the picture: a matrix of examples and a weight vector."""
import re

from everything_ai.check import Code, CodeBox, FunctionBox, Mistake

# Four houses from the lesson ml-vocabulary: area in m², rooms, garden (1 or 0); and the model of math-dot-product.
HOUSES = [[62, 3, 0], [85, 4, 1], [110, 5, 1], [48, 2, 0]]
W = [2, 15, 20]
B = 30

# The lesson is done when these are correct. "result-shape" is the choice question in the lesson.
REQUIRED = ["result-shape", "all-grades", "shape-rule"]

_HEAD = "import numpy as np\n\nX = np.array([[6, 8], [10, 7], [2, 9]])\nw = np.array([0.5, 0.3])\nb = 1\n"
_START = _HEAD + "y_pred = ...\ny_pred\n"
_SOLUTION = _HEAD + "y_pred = X @ w + b\ny_pred"


def _grades(new: str) -> str:
    return _SOLUTION.replace("y_pred = X @ w + b", new)


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_RULE_START = "def result_shape(a, b):\n    ...\n"
_RULE = "def result_shape(a, b):\n    if a[1] != b[0]:\n        return False\n    return (a[0], b[1])"


def _rule(new: str) -> str:
    return "def result_shape(a, b):\n" + new


EXERCISES = {
    "all-grades": CodeBox(
        name="all-grades",
        start=_START,
        solution=_SOLUTION,
        keep=["w = np.array([0.5, 0.3])", "b = 1"],
        # The check also runs the box with other students: numbers typed in by hand do not change with them.
        others=[{"X": Code("np.array([[4, 6], [8, 8]])")}],
        first=[_DOTS],
        mistakes=[
            Mistake("w-first", error=lambda e: isinstance(e, ValueError) and ("matmul" in str(e) or "not aligned" in str(e)),
                    en="Python stops: the shapes do not fit. The inner numbers must be equal: `X` has shape (3, 2) and `w` shape (2,), so `X` comes first: `X @ w`.",
                    nl="Python stopt: de shapes passen niet. De binnenste getallen moeten gelijk zijn: `X` heeft shape (3, 2) en `w` shape (2,), dus `X` komt eerst: `X @ w`."),
            Mistake("star", wrong=_grades("y_pred = X * w + b"),
                    en="Your last line gives `{got}`: a matrix of separate products, nothing added up. For one prediction per row, use `X @ w`.",
                    nl="Je laatste regel geeft `{got}`: een matrix met losse producten, niets opgeteld. Voor één voorspelling per rij gebruik je `X @ w`."),
            Mistake("no-bias", wrong=_grades("y_pred = X @ w"),
                    en="Your last line gives `{got}`: the dot products alone. Each prediction also adds the bias term `b`.",
                    nl="Je laatste regel geeft `{got}`: alleen de inproducten. Elke voorspelling telt ook de bias term `b` op."),
            Mistake("first-only", wrong=_grades("y_pred = np.dot(X[0], w) + b"),
                    en="Your last line gives `{got}`: the prediction for the first student only. `X @ w` does every row at once.",
                    nl="Je laatste regel geeft `{got}`: alleen de voorspelling voor de eerste leerling. `X @ w` doet elke rij tegelijk."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the predictions for all students, with `@` and the bias term.",
                      "nl": "Vervang de `...` door de voorspellingen voor alle leerlingen, met `@` en de bias term."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other students by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere leerlingen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `y_pred` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `y_pred` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`, and it should be one prediction per student: the dot product of each row of `X` with `w`, plus `b`.",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet één voorspelling per leerling zijn: het inproduct van elke rij van `X` met `w`, plus `b`."},
            "other": {"en": "Right for these students. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you work it out with `X`, `w` and `b` themselves?",
                      "nl": "Goed voor deze leerlingen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je het uit met `X`, `w` en `b` zelf?"},
            "other_error": {"en": "Right for these students. But with {where}, your code stops with an error. Does it work for any number of students?",
                            "nl": "Goed voor deze leerlingen. Maar met {where} stopt je code met een fout. Werkt hij voor elk aantal leerlingen?"},
            "correct": {"en": "Three predictions at once: 6.4, 8.1 and 4.7. `X` has shape (3, 2) and `w` shape (2,): the inner numbers fit, and the result has shape (3,), one per student. The check also tried two other students.",
                        "nl": "Drie voorspellingen tegelijk: 6,4; 8,1 en 4,7. `X` heeft shape (3, 2) en `w` shape (2,): de binnenste getallen passen, en de uitkomst heeft shape (3,), één per leerling. De check probeerde ook twee andere leerlingen."},
        },
    ),
    "shape-rule": FunctionBox(
        name="shape-rule",
        function="result_shape",
        start=_RULE_START,
        solution=_RULE,
        visible=[((4, 2), (2, 3))],
        hidden=[((3, 2), (3, 2)), ((8, 3), (3, 1)), ((1, 4), (4, 7)), ((2, 2), (2, 2)), ((4, 6), (5, 4))],
        first=[_DOTS],
        mistakes=[
            Mistake("list", matches=lambda got, text: isinstance(got, list),
                    en="`{call}` gives `{got}`, a list, and it should give `{want}`. A shape is a tuple, with round brackets.",
                    nl="`{call}` geeft `{got}`, een lijst, en dat moet `{want}` zijn. Een shape is een tuple, met ronde haakjes."),
            Mistake("inner", wrong=_rule("    return (a[1], b[0])"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the two inner numbers. Those must be equal; the result takes the two outer ones.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de twee binnenste getallen. Die moeten gelijk zijn; de uitkomst krijgt de twee buitenste."),
            Mistake("no-test", wrong=_rule("    return (a[0], b[1])"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: these shapes do not fit, because the inner numbers differ. Check that first, and give back `False` when they do not fit.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: deze shapes passen niet, want de binnenste getallen verschillen. Controleer dat eerst, en geef `False` terug als ze niet passen."),
            Mistake("outer-test", wrong=_rule("    if a[0] != b[1]:\n        return False\n    return (a[0], b[1])"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your function compares the outer numbers. The inner ones must be equal: the columns of `a`, `a[1]`, and the rows of `b`, `b[0]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je functie vergelijkt de buitenste getallen. De binnenste moeten gelijk zijn: de kolommen van `a`, `a[1]`, en de rijen van `b`, `b[0]`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the shape rule: `False` when the inner numbers differ, otherwise the shape of the result.",
                      "nl": "Vervang de `...` door de shape-regel: `False` als de binnenste getallen verschillen, anders de shape van de uitkomst."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with two shapes, and the `def` line of your function has a different number of names between its parentheses. Keep `(a, b)`.",
                          "nl": "De check roept `{call}` aan, met twee shapes, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(a, b)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. The shape rule: (n, k) @ (k, m) gives (n, m), and the two k's must be equal.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. De shape-regel: (n, k) @ (k, m) geeft (n, m), en de twee k's moeten gelijk zijn."},
            "unreached": {"en": "With `{call}`, your function gives nothing back. Does every way through it end with a `return`: `False` when the shapes do not fit, and the new shape when they do?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt elke weg erdoor met een `return`: `False` als de shapes niet passen, en de nieuwe shape als ze wel passen?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it also give `False` when the shapes do not fit?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Geeft hij ook `False` als de shapes niet passen?"},
            "correct": {"en": "Your function is right in all {n} cases. (4, 2) @ (2, 3) gives (4, 3): the inner 2's fit, and the outer numbers remain. (3, 2) @ (3, 2) does not fit.",
                        "nl": "Je functie klopt in alle {n} gevallen. (4, 2) @ (2, 3) geeft (4, 3): de binnenste tweeën passen, en de buitenste getallen blijven over. (3, 2) @ (3, 2) past niet."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("all-grades", _START, "empty", None),
    ("all-grades", _SOLUTION, "correct", None),
    ("all-grades", _grades("y_pred = np.dot(X, w) + b"), "correct", None),
    ("all-grades", _grades("y_pred = (X * w).sum(axis=1) + b"), "correct", None),
    ("all-grades", _grades("y_pred = X.dot(w) + b"), "correct", None),
    ("all-grades", _grades("y_pred = X * w + b"), "wrong", "star"),
    ("all-grades", _grades("y_pred = X @ w"), "wrong", "no-bias"),
    ("all-grades", _grades("y_pred = np.dot(X[0], w) + b"), "wrong", "first-only"),
    ("all-grades", _grades("y_pred = w @ X + b"), "error", "w-first"),
    ("all-grades", _grades("y_pred = np.array([6.4, 8.1, 4.7])"), "almost", "other-value"),
    ("all-grades", _grades("y_pred = np.dot(w, X) + b"), "error", "w-first"),
    ("all-grades", _HEAD + "y_pred = X @ w + b\nprint(y_pred)", "wrong", "print-last"),
    ("shape-rule", _RULE_START, "empty", None),
    ("shape-rule", _RULE, "correct", None),
    ("shape-rule", _rule("    if a[1] == b[0]:\n        return (a[0], b[1])\n    return False"), "correct", None),
    ("shape-rule", _rule("    n, k = a\n    k2, m = b\n    if k != k2:\n        return False\n    return (n, m)"), "correct", None),
    ("shape-rule", _rule("    return (a[1], b[0])"), "wrong", "inner"),
    ("shape-rule", _rule("    if a[1] != b[0]:\n        return False\n    return [a[0], b[1]]"), "wrong", "list"),
    ("shape-rule", _rule("    if a[1] == b[0]:\n        return (a[0], b[1])"), "wrong", "unreached"),
    ("shape-rule", _rule("    return (a[0], b[1])"), "wrong", "no-test"),
    ("shape-rule", _rule("    if a[0] != b[1]:\n        return False\n    return (a[0], b[1])"), "wrong", "outer-test"),
]
