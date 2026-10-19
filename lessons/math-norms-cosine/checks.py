# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson math-norms-cosine: the cosine similarity of two people's music, and the distance between two
vectors in plain Python. And the picture with two vectors, which follows two sliders."""
import math
import re

from everything_ai.check import Code, CodeBox, FunctionBox, Mistake

# Vector a of the picture: length 4, at 15 degrees above the horizontal axis. Vector b turns from there.
A_LENGTH, A_ANGLE = 4, 15


def vectors(angle: float, length: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """The vectors a and b of the picture, with `angle` degrees between them and b `length` long."""
    ta, tb = math.radians(A_ANGLE), math.radians(A_ANGLE + angle)
    return (A_LENGTH * math.cos(ta), A_LENGTH * math.sin(ta)), (length * math.cos(tb), length * math.sin(tb))


def measures(angle: float, length: float) -> tuple[float, float]:
    """The distance between the heads of a and b, and their cosine similarity, worked out from the vectors."""
    (a1, a2), (b1, b2) = vectors(angle, length)
    distance = math.hypot(a1 - b1, a2 - b2)
    similarity = round((a1 * b1 + a2 * b2) / (A_LENGTH * length), 12) + 0.0   # + 0.0: never a minus zero
    return distance, similarity


def picture(angle: float, length: float, labels: dict[str, str], over: str) -> str:
    """The vectors a and b as arrows from the origin, with a dashed line between their heads. Beside them, for every
    stop of the slider that `over` names: the cosine similarity for every angle, or the distance (dashed, as the line
    between the heads) and the cosine similarity for every length. Drawn as SVG text, once for every stop of the
    slider (L.explore); dragging across the second chart sets that slider."""
    from everything_ai import chart
    (a1, a2), (b1, b2) = vectors(angle, length)
    p = chart.Panel((-6.5, 6.5), (-2, 6), title=labels["title"])
    p.line([a1, b1], [a2, b2], "ink-3", 1.6, dashed=True)
    p.arrow(0, 0, a1, a2, "ink")
    p.arrow(0, 0, b1, b2, "ink")
    # each name just past the head of its arrow, a to one side and b to the other, so they never overlap
    for (x, y), name, side in (((a1, a2), "a", -1), ((b1, b2), "b", 1)):
        r = math.hypot(x, y)
        ux, uy = x / r, y / r
        p.text(x + 0.3 * ux - side * 0.4 * uy, y + 0.3 * uy + side * 0.4 * ux, name)
    distance, similarity = measures(angle, length)
    if over == "angle":
        angles = list(range(0, 181, 5))
        k = chart.Panel((0, 180), (-1.1, 1.1), title=labels["every"], xlabel=labels["x"], ylabel=labels["y"],
                        xticks=[0, 45, 90, 135, 180], yticks=[-1, -0.5, 0, 0.5, 1])
        k.line([angle, angle], [-1.1, 1.1], "ink-3", 1.2)
        k.line(angles, [measures(v, length)[1] for v in angles], "ink", 2.2)
        k.points([angle], [similarity], "ink", 4.5)
    else:
        lengths = [1 + 0.1 * i for i in range(41)]
        k = chart.Panel((1, 5), (0, 4), title=labels["every"], xlabel=labels["x"], yticks=[0, 1, 2, 3, 4])
        # the line that marks the length runs from the axis up to the lower dot only, and each label sits above its
        # line, so that no line or dot crosses a label at any length
        k.line([length, length], [0, similarity], "ink-3", 1.2)
        k.line(lengths, [measures(angle, v)[0] for v in lengths], "ink-3", 2, dashed=True)
        k.line(lengths, [measures(angle, v)[1] for v in lengths], "ink", 2.2)
        k.points([length, length], [distance, similarity], "ink", 4.5)
        k.text(4.7, 3.75, labels["distance"], "ink-2", anchor="end")
        k.text(4.9, 1.1, labels["similarity"], "ink-2", anchor="end")
    # 330 by 239: the plot of the arrows has the same scale on both axes, so an angle looks as large as it is
    return chart.figure([p, k], alt=labels["alt"], width=330, height=239, drag=1)


# The texts of the worked example: how often each uses the words "ball", "goal" and "vote".
TEXTS = {"a": [3, 2, 0], "b": [12, 8, 1], "c": [0, 1, 4]}

# The lesson is done when these are correct. "norm" is the choice question in the lesson.
REQUIRED = ["norm", "cosine", "distance-loop"]

_A, _B = [10, 2, 5], [4, 1, 2]
_HEAD = "import numpy as np\n\na = np.array([10, 2, 5])\nb = np.array([4, 1, 2])\n"
_START = _HEAD + "similarity = ...\nsimilarity\n"
_SOLUTION = _HEAD + "similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))\nsimilarity"


def _cosine(new: str) -> str:
    return _SOLUTION.replace("similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))", new)


def _near(got, want: float, margin: float = 1e-6) -> bool:
    try:
        return abs(float(got) - want) < margin
    except (TypeError, ValueError):
        return False


_DOT = sum(p * q for p, q in zip(_A, _B))
_NORM_A, _NORM_B = sum(p * p for p in _A) ** 0.5, sum(q * q for q in _B) ** 0.5

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_DIST_START = "def distance(a, b):\n    ...\n"
_DIST = ("def distance(a, b):\n    total = 0\n    for i in range(len(a)):\n        total = total + (a[i] - b[i]) ** 2\n"
         "    return total ** 0.5")


def _dist(new: str) -> str:
    return "def distance(a, b):\n" + new


EXERCISES = {
    "cosine": CodeBox(
        name="cosine",
        start=_START,
        solution=_SOLUTION,
        keep=["a = np.array([10, 2, 5])"],
        # The check also runs the box with another person b: numbers typed in by hand do not change with it.
        others=[{"b": Code("np.array([0, 6, 1])")}],
        first=[_DOTS],
        mistakes=[
            Mistake("rounded", state="almost", matches=lambda got, text: _near(got, _DOT / (_NORM_A * _NORM_B), 1e-3),
                    en="Your last line gives `{got}`: close, but rounded. Let Python work it out from `a` and `b`, without rounding and without typing in a number.",
                    nl="Je laatste regel geeft `{got}`: dichtbij, maar afgerond. Laat Python het uitrekenen met `a` en `b`, zonder afronden en zonder zelf een getal in te typen."),
            Mistake("no-divide", wrong=_cosine("similarity = np.dot(a, b)"),
                    en="Your last line gives `{got}`: the dot product. It grows with the lengths of `a` and `b`. Divide it by the product of their norms, step 3 of the recipe.",
                    nl="Je laatste regel geeft `{got}`: het inproduct. Dat groeit mee met de lengtes van `a` en `b`. Deel het door het product van hun normen, stap 3 van het recept."),
            Mistake("one-norm", matches=lambda got, text: _near(got, _DOT / _NORM_A) or _near(got, _DOT / _NORM_B),
                    en="Your last line gives `{got}`: the dot product divided by one norm. Divide by both norms, multiplied together.",
                    nl="Je laatste regel geeft `{got}`: het inproduct gedeeld door één norm. Deel door beide normen, met elkaar vermenigvuldigd."),
            Mistake("no-brackets", matches=lambda got, text: _near(got, _DOT / _NORM_A * _NORM_B) or _near(got, _DOT / _NORM_B * _NORM_A),
                    en="Your last line gives `{got}`. Python divides by the first norm and then multiplies by the second. Put brackets around the product of the two norms.",
                    nl="Je laatste regel geeft `{got}`. Python deelt door de eerste norm en vermenigvuldigt daarna met de tweede. Zet haakjes om het product van de twee normen."),
            Mistake("sum-norms", wrong=_cosine("similarity = np.dot(a, b) / (np.linalg.norm(a) + np.linalg.norm(b))"),
                    en="Your last line gives `{got}`: you divide by the sum of the norms. The formula divides by their product.",
                    nl="Je laatste regel geeft `{got}`: je deelt door de som van de normen. De formule deelt door hun product."),
            Mistake("no-roots", wrong=_cosine("similarity = np.dot(a, b) / (np.dot(a, a) * np.dot(b, b))"),
                    en="Your last line gives `{got}`. `np.dot(a, a)` is the square of the norm: take its square root with `** 0.5`, or use `np.linalg.norm(a)`.",
                    nl="Je laatste regel geeft `{got}`. `np.dot(a, a)` is het kwadraat van de norm: neem de wortel met `** 0.5`, of gebruik `np.linalg.norm(a)`."),
            Mistake("distance", wrong=_cosine("similarity = np.linalg.norm(a - b)"),
                    en="Your last line gives `{got}`: the distance between `a` and `b`. The task asks for the cosine similarity, which looks at the direction and not at how far apart they are.",
                    nl="Je laatste regel geeft `{got}`: de afstand tussen `a` en `b`. De opdracht vraagt de cosine similarity, die naar de richting kijkt en niet naar hoe ver ze uit elkaar liggen."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the cosine similarity of `a` and `b`: their dot product, divided by the product of their norms.",
                      "nl": "Vervang de `...` door de cosine similarity van `a` en `b`: hun inproduct, gedeeld door het product van hun normen."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another person by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een andere persoon."},
            "no_value": {"en": "The last line of the box shows nothing. Put `similarity` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `similarity` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`, and it should be the dot product of `a` and `b`, divided by the product of their norms: a number between −1 and 1.",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet het inproduct van `a` en `b` zijn, gedeeld door het product van hun normen: een getal tussen −1 en 1."},
            "other": {"en": "Right for this person. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you work it out with `a` and `b` themselves?",
                      "nl": "Goed voor deze persoon. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je het uit met `a` en `b` zelf?"},
            "other_error": {"en": "Right for this person. But with {where}, your code stops with an error. Does it work for every `b`?",
                            "nl": "Goed voor deze persoon. Maar met {where} stopt je code met een fout. Werkt hij voor elke `b`?"},
            "correct": {"en": "About 0.999: the two of them play almost the same mix of music, even though the first plays more than twice as much. The check also tried another person.",
                        "nl": "Ongeveer 0,999: de twee luisteren naar bijna dezelfde mix van muziek, ook al luistert de eerste meer dan twee keer zoveel. De check probeerde ook een andere persoon."},
        },
    ),
    "distance-loop": FunctionBox(
        name="distance-loop",
        function="distance",
        start=_DIST_START,
        solution=_DIST,
        visible=[([1, 2], [4, 6])],
        python_values=True,
        hidden=[([0, 0, 0], [1, 2, 2]), ([5], [2]), ([1, 1], [1, 1]), ([2, 5, 1], [1, 3, 3]), ([-1, 2], [2, -2])],
        first=[_DOTS],
        mistakes=[
            Mistake("list-minus", error=lambda e: isinstance(e, TypeError) and "for -: 'list' and 'list'" in str(e),
                    en="`{call}` stops: Python cannot subtract two lists with `-`; that only works for NumPy arrays. Subtract place by place in a loop: `a[i] - b[i]`.",
                    nl="`{call}` stopt: Python kan twee lijsten niet van elkaar aftrekken met `-`; dat werkt alleen bij NumPy-arrays. Trek plek voor plek af in een loop: `a[i] - b[i]`."),
            Mistake("half", wrong=_dist("    total = 0\n    for i in range(len(a)):\n        total = total + (a[i] - b[i]) ** 2\n    return total ** 1 / 2"),
                    en="`{call}` gives `{got}`: half of the sum, not its square root. Python works out `** 1` first and then divides by 2. Put brackets around the power, `** (1 / 2)`, or write `** 0.5`.",
                    nl="`{call}` geeft `{got}`: de helft van de som, niet de wortel. Python rekent eerst `** 1` uit en deelt dan door 2. Zet haakjes om de macht, `** (1 / 2)`, of schrijf `** 0.5`."),
            Mistake("no-root", wrong=_dist("    total = 0\n    for i in range(len(a)):\n        total = total + (a[i] - b[i]) ** 2\n    return total"),
                    en="`{call}` gives `{got}`: the sum of the squared differences. One step is missing: the square root, with `** 0.5`.",
                    nl="`{call}` geeft `{got}`: de som van de gekwadrateerde verschillen. Er ontbreekt één stap: de wortel, met `** 0.5`."),
            Mistake("no-square", wrong=_dist("    total = 0\n    for i in range(len(a)):\n        total = total + (a[i] - b[i])\n    return total ** 0.5"),
                    en="`{call}` does not give a plain number: you add up the differences without squaring them, so the total is negative, and Python cannot take an ordinary square root of that. Square each difference first.",
                    nl="`{call}` geeft geen gewoon getal: je telt de verschillen op zonder ze te kwadrateren, dus het totaal is negatief, en daar kan Python geen gewone wortel uit trekken. Kwadrateer eerst elk verschil."),
            Mistake("lengths", wrong=_dist("    sa = 0\n    sb = 0\n    for i in range(len(a)):\n        sa = sa + a[i] ** 2\n        sb = sb + b[i] ** 2\n    return sa ** 0.5 - sb ** 0.5"),
                    en="`{call}` gives `{got}`: the difference between the two norms. The distance is the norm of the difference: first subtract place by place, then square, add up and take the root.",
                    nl="`{call}` geeft `{got}`: het verschil tussen de twee normen. De afstand is de norm van het verschil: eerst plek voor plek aftrekken, dan kwadrateren, optellen en de wortel nemen."),
            Mistake("streets", wrong=_dist("    total = 0\n    for i in range(len(a)):\n        total = total + abs(a[i] - b[i])\n    return total"),
                    en="`{call}` gives `{got}`: the differences added up, as if you walk along the streets of a grid. The distance of this lesson is the straight line: square the differences, add them up and take the root.",
                    nl="`{call}` geeft `{got}`: de verschillen opgeteld, alsof je langs de straten van een raster loopt. De afstand van deze les is de rechte lijn: kwadrateer de verschillen, tel ze op en neem de wortel."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the distance between `a` and `b`, given back with `return`.",
                      "nl": "Vervang de `...` door de afstand tussen `a` en `b`, teruggegeven met `return`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with two lists, and the `def` line of your function has a different number of names between its parentheses. Keep `(a, b)`.",
                          "nl": "De check roept `{call}` aan, met twee lijsten, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(a, b)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you subtract place by place, square each difference, add them up and take the root?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Trek je plek voor plek af, kwadrateer je elk verschil, tel je ze op en neem je de wortel?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for any length, and for negative numbers?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor elke lengte, en voor negatieve getallen?"},
            "correct": {"en": "Your function is right in all {n} cases. From (1, 2) to (4, 6) is 3 across and 4 up, and the straight line is 5: the square root of 9 + 16.",
                        "nl": "Je functie klopt in alle {n} gevallen. Van (1, 2) naar (4, 6) is 3 opzij en 4 omhoog, en de rechte lijn is 5: de wortel uit 9 + 16."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("cosine", _START, "empty", None),
    ("cosine", _SOLUTION, "correct", None),
    ("cosine", _cosine("similarity = np.dot(a, b) / (np.dot(a, a) ** 0.5 * np.dot(b, b) ** 0.5)"), "correct", None),
    ("cosine", _cosine("similarity = (a * b).sum() / ((a ** 2).sum() ** 0.5 * (b ** 2).sum() ** 0.5)"), "correct", None),
    ("cosine", _cosine("similarity = a @ b / np.linalg.norm(a) / np.linalg.norm(b)"), "correct", None),
    ("cosine", _cosine("similarity = np.dot(a, b)"), "wrong", "no-divide"),
    ("cosine", _cosine("similarity = np.dot(a, b) / np.linalg.norm(b)"), "wrong", "one-norm"),
    ("cosine", _cosine("similarity = np.dot(a, b) / np.linalg.norm(a) * np.linalg.norm(b)"), "wrong", "no-brackets"),
    ("cosine", _cosine("similarity = np.dot(a, b) / (np.linalg.norm(a) + np.linalg.norm(b))"), "wrong", "sum-norms"),
    ("cosine", _cosine("similarity = np.dot(a, b) / (np.dot(a, a) * np.dot(b, b))"), "wrong", "no-roots"),
    ("cosine", _cosine("similarity = np.linalg.norm(a - b)"), "wrong", "distance"),
    ("cosine", _cosine("similarity = round(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), 3)"), "almost", "rounded"),
    ("cosine", _cosine("similarity = 0.9990766"), "almost", "other-value"),
    ("cosine", _HEAD + "similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))\nprint(similarity)", "wrong", "print-last"),
    ("distance-loop", _DIST_START, "empty", None),
    ("distance-loop", _DIST, "correct", None),
    ("distance-loop", _dist("    total = 0\n    for p, q in zip(a, b):\n        total += (p - q) ** 2\n    return total ** 0.5"), "correct", None),
    ("distance-loop", _dist("    import math\n    return math.sqrt(sum([(a[i] - b[i]) ** 2 for i in range(len(a))]))"), "correct", None),
    ("distance-loop", _dist("    total = 0\n    for i in range(len(a)):\n        total = total + (a[i] - b[i]) ** 2\n    return total"), "wrong", "no-root"),
    ("distance-loop", _dist("    total = 0\n    for i in range(len(a)):\n        total = total + (a[i] - b[i])\n    return total ** 0.5"), "wrong", "no-square"),
    ("distance-loop", _dist("    sa = 0\n    sb = 0\n    for i in range(len(a)):\n        sa = sa + a[i] ** 2\n        sb = sb + b[i] ** 2\n    return sa ** 0.5 - sb ** 0.5"), "wrong", "lengths"),
    ("distance-loop", _dist("    total = 0\n    for i in range(len(a)):\n        total = total + abs(a[i] - b[i])\n    return total"), "wrong", "streets"),
    ("distance-loop", _dist("    return (a - b) ** 2"), "error", "list-minus"),
    ("distance-loop", _dist("    total = 0\n    for i in range(len(a)):\n        total = total + (a[i] - b[i]) ** 2\n    return total ** 1 / 2"), "wrong", "half"),
    ("distance-loop", _dist("    return 5.0"), "almost", None),
]
