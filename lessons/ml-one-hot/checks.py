# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-one-hot: the one-hot vector of one label as a list, and counting the classes of a batch of
digit labels by adding up their one-hot vectors."""
import re

from everything_ai.check import CodeBox, Code, FunctionBox, Mistake

# The lesson is done when these are correct. "how-long" is the choice question in the lesson.
REQUIRED = ["how-long", "one-hot", "count-classes"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_O_START = "def one_hot(label, k):\n    vector = []\n    ...\n    return vector\n"


def _o(test: str) -> str:
    return ("def one_hot(label, k):\n    vector = []\n    for i in range(k):\n        if " + test + ":\n            vector.append(1)\n"
            "        else:\n            vector.append(0)\n    return vector")


_O = _o("i == label")

_C_HEAD = "import numpy as np\n\nlabels = np.array([3, 0, 9, 3, 7])\n"
_C_START = _C_HEAD + "counts = ...\ncounts\n"
_C = _C_HEAD + "counts = np.eye(10)[labels].sum(axis=0)\ncounts"


def _c(new: str) -> str:
    return _C.replace("counts = np.eye(10)[labels].sum(axis=0)", new)


EXERCISES = {
    "one-hot": FunctionBox(
        name="one-hot",
        function="one_hot",
        start=_O_START,
        solution=_O,
        visible=[(1, 3)],
        hidden=[(0, 3), (9, 10), (2, 4), (0, 1)],
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("one-later", wrong=_o("i == label + 1"),
                    en="`{call}` gives `{got}`: the 1 sits one place too late. Places count from 0, like the classes: class 0 has its 1 at place 0.",
                    nl="`{call}` geeft `{got}`: de 1 staat één plek te laat. Plekken tellen vanaf 0, net als de klassen: klasse 0 heeft zijn 1 op plek 0."),
            Mistake("one-early", wrong=_o("i == label - 1"),
                    en="`{call}` gives `{got}`: the 1 sits one place too early. Places count from 0, like the classes: class 1 has its 1 at place 1.",
                    nl="`{call}` geeft `{got}`: de 1 staat één plek te vroeg. Plekken tellen vanaf 0, net als de klassen: klasse 1 heeft zijn 1 op plek 1."),
            Mistake("swapped", wrong=_o("i == k").replace("for i in range(k)", "for i in range(label)"),
                    en="`{call}` gives `{got}`: the vector should have `k` places, one per class, and the 1 at place `label`. Did `label` and `k` swap roles?",
                    nl="`{call}` geeft `{got}`: de vector moet `k` plekken hebben, één per klasse, met de 1 op plek `label`. Heb je de rollen van `label` en `k` omgedraaid?"),
            Mistake("up-to", wrong=_o("i <= label"),
                    en="`{call}` gives `{got}`: a 1 at every place up to the label. A one-hot vector has exactly one 1, at the place of the class: `==`.",
                    nl="`{call}` geeft `{got}`: een 1 op elke plek tot en met het label. Een one-hot-vector heeft precies één 1, op de plek van de klasse: `==`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a loop over the `k` places that adds a 1 at the place of `label` and a 0 everywhere else.",
                      "nl": "Vervang de `...` door een loop over de `k` plekken die een 1 toevoegt op de plek van `label` en overal anders een 0."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with a label and a number of classes, and the `def` line of your function has a different number of names between its parentheses. Keep `(label, k)`.",
                          "nl": "De check roept `{call}` aan, met een label en een aantal klassen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(label, k)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does the vector have `k` places, with a 1 at place `label` and 0 everywhere else?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Heeft de vector `k` plekken, met een 1 op plek `label` en overal anders 0?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for any label and any number of classes?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor elk label en elk aantal klassen?"},
            "correct": {"en": "Your function is right in all {n} cases. Class 9 of the ten digits becomes nine zeros and a 1 at the end: one place per class, one 1.",
                        "nl": "Je functie klopt in alle {n} gevallen. Klasse 9 van de tien cijfers wordt negen nullen en een 1 aan het eind: één plek per klasse, één 1."},
        },
    ),
    "count-classes": CodeBox(
        name="count-classes",
        start=_C_START,
        solution=_C,
        keep=[],
        # The check also runs the box with other labels: a number typed in by hand does not change with them.
        others=[{"labels": Code("np.array([1, 1, 1, 0, 8, 8])")}],
        first=[_DOTS],
        mistakes=[
            Mistake("per-row", wrong=_c("counts = np.eye(10)[labels].sum(axis=1)"),
                    en="Your last line gives `{got}`: one number per example, always 1, because each row has exactly one 1. To count per class, add up each column instead: which axis runs down the columns?",
                    nl="Je laatste regel geeft `{got}`: één getal per voorbeeld, altijd 1, omdat elke rij precies één 1 heeft. Om per klasse te tellen, tel je in plaats daarvan elke kolom op: langs welke axis loop je een kolom af?"),
            Mistake("all", wrong=_c("counts = np.eye(10)[labels].sum()"),
                    en="Your last line gives `{got}`: all the 1s together, one per example. Add up each column on its own, for one count per class: which axis runs down the columns?",
                    nl="Je laatste regel geeft `{got}`: alle enen samen, één per voorbeeld. Tel elke kolom apart op, voor één telling per klasse: langs welke axis loop je een kolom af?"),
            Mistake("matrix", wrong=_c("counts = np.eye(10)[labels]"),
                    en="Your last line gives the one-hot vectors themselves, one row per example. Now add up each column, `.sum(axis=0)`, for one count per class.",
                    nl="Je laatste regel geeft de one-hot-vectoren zelf, één rij per voorbeeld. Tel nu elke kolom op, `.sum(axis=0)`, voor één telling per klasse."),
            Mistake("wrong-size", error=lambda e: isinstance(e, IndexError) and "out of bounds for axis 0" in str(e),
                    en="Python stops: a label is too big for the identity matrix. `np.eye(n)` has n rows, one per class; with ten digits, that is `np.eye(10)`, not one row per example.",
                    nl="Python stopt: een label is te groot voor de eenheidsmatrix. `np.eye(n)` heeft n rijen, één per klasse; bij tien cijfers is dat `np.eye(10)`, niet één rij per voorbeeld."),
            Mistake("no-axis", error=lambda e: type(e).__name__ == "AxisError",
                    en="Python stops: your array has no axis with that number. The one-hot vectors form a matrix with two axes: axis 0 runs down the rows, axis 1 along each row.",
                    nl="Python stopt: je array heeft geen axis met dat nummer. De one-hot-vectoren vormen een matrix met twee axes: axis 0 loopt langs de rijen omlaag, axis 1 langs elke rij."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the one-hot vectors of `labels`, added up per class.",
                      "nl": "Vervang de `...` door de one-hot-vectoren van `labels`, opgeteld per klasse."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug."},
            "no_value": {"en": "The last line of the box shows nothing. Put `counts` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `counts` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Is it ten numbers, one per digit: the one-hot vectors of `labels`, with each column added up?",
                      "nl": "Je laatste regel geeft `{got}`. Zijn het tien getallen, één per cijfer: de one-hot-vectoren van `labels`, met elke kolom opgeteld?"},
            "other": {"en": "Right for these labels. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you work it out from `labels` itself, with a place for each of the ten digits, also the ones that do not appear?",
                      "nl": "Goed voor deze labels. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je het uit met `labels` zelf, met een plek voor elk van de tien cijfers, ook die niet voorkomen?"},
            "other_error": {"en": "Right for these labels. But with {where}, your code stops with an error. Do you work it out from `labels` itself?",
                            "nl": "Goed voor deze labels. Maar met {where} stopt je code met een fout. Reken je het uit met `labels` zelf?"},
            "correct": {"en": "Ten counts, one per digit: the 3 twice, the 0, 7 and 9 once, the others not at all. Adding up one-hot vectors counts the classes. The check also tried other labels.",
                        "nl": "Tien tellingen, één per cijfer: de 3 twee keer, de 0, 7 en 9 één keer, de rest niet. Door one-hot-vectoren op te tellen, tel je de klassen. De check probeerde ook andere labels."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("one-hot", _O_START, "empty", None),
    ("one-hot", _O, "correct", None),
    ("one-hot", "def one_hot(label, k):\n    vector = []\n    for i in range(k):\n        vector.append(0)\n    vector[label] = 1\n    return vector", "correct", None),
    ("one-hot", "def one_hot(label, k):\n    vector = [0] * k\n    vector[label] = 1\n    return vector", "correct", None),
    ("one-hot", "import numpy as np\n\n\ndef one_hot(label, k):\n    return np.eye(k)[label]", "correct", None),
    ("one-hot", _o("i == label + 1"), "wrong", "one-later"),
    ("one-hot", _o("i == label - 1"), "wrong", "one-early"),
    ("one-hot", _o("i <= label"), "wrong", "up-to"),
    ("one-hot", "def one_hot(label, k):\n    return [0, 1, 0]", "almost", None),
    ("count-classes", _C_START, "empty", None),
    ("count-classes", _C, "correct", None),
    ("count-classes", _c("onehot = np.eye(10)[labels]\ncounts = onehot.sum(axis=0)"), "correct", None),
    ("count-classes", _c("counts = np.eye(10)[labels].sum(axis=1)"), "wrong", "per-row"),
    ("count-classes", _c("counts = np.eye(10)[labels].sum()"), "wrong", "all"),
    ("count-classes", _c("counts = np.eye(10)[labels]"), "wrong", "matrix"),
    ("count-classes", _c("counts = np.eye(len(labels))[labels].sum(axis=0)"), "error", "wrong-size"),
    ("count-classes", _c("counts = np.eye(10)[labels].sum(axis=2)"), "error", "no-axis"),
    ("count-classes", _c("counts = np.array([1, 0, 0, 2, 0, 0, 0, 1, 0, 1])"), "almost", "other-value"),
]
