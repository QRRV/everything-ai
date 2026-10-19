# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-classification: train delays in minutes, a regression label, turned into the classes
"late" and "on time". And the students' table in the picture, once with a number as its label and once with a class."""
import csv
import re
from pathlib import Path

from everything_ai.check import FunctionBox, Mistake

STUDENTS = Path(__file__).parent.parent / "ml-vocabulary" / "students.csv"


def picture(labels: list[str]) -> list[tuple]:
    """The students who have a label, twice: once with the grade marked, once with passed marked."""
    with open(STUDENTS, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], [row for row in rows[1:] if row[2]]
    grade, passed = header.index("grade"), header.index("passed")
    return [
        (labels[0], header, rows, {(i, grade) for i in range(len(rows))}, 'df["grade"]'),
        (labels[1], header, rows, {(i, passed) for i in range(len(rows))}, 'df["passed"]'),
    ]


# The lesson is done when these are correct. "which-regression" and "class-numbers" are the choice questions in the lesson.
REQUIRED = ["which-regression", "class-numbers", "delay-classes"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_START = "def delay_classes(delays):\n    classes = []\n    ...\n    return classes\n"


def _d(test: str, yes: str, no: str) -> str:
    return ("def delay_classes(delays):\n    classes = []\n    for delay in delays:\n"
            f"        if {test}:\n            classes.append({yes})\n        else:\n            classes.append({no})\n"
            "    return classes")


_SOLUTION = _d("delay >= 5", '"late"', '"on time"')
_RETURN_IN_LOOP = _SOLUTION.replace("\n    return classes", "\n        return classes")
_CLASS_NAME = ("def delay_classes(delays):\n    classes = []\n    for delay in delays:\n        if delay >= 5:\n"
               "            class = \"late\"\n        else:\n            class = \"on time\"\n        classes.append(class)\n    return classes")

EXERCISES = {
    "delay-classes": FunctionBox(
        name="delay-classes",
        function="delay_classes",
        start=_START,
        solution=_SOLUTION,
        visible=[([2, 12, 0, 7],)],
        hidden=[([5],), ([4],), ([],), ([4.5, 30, 5, 1],), ([5.5, 0, 9],)],
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("bools", wrong="def delay_classes(delays):\n    classes = []\n    for delay in delays:\n        classes.append(delay >= 5)\n    return classes",
                    en="`{call}` gives `{got}`: `True` and `False`. The classes here are the words `\"late\"` and `\"on time\"`: add those to `classes`.",
                    nl="`{call}` geeft `{got}`: `True` en `False`. De klassen zijn hier de woorden `\"late\"` en `\"on time\"`: voeg die toe aan `classes`."),
            Mistake("greater", wrong=_d("delay > 5", '"late"', '"on time"'),
                    en="`{call}` gives `{got}`: a train that is exactly 5 minutes late counts as late too. Which comparison is also `True` for 5?",
                    nl="`{call}` geeft `{got}`: een trein die precies 5 minuten te laat is, telt ook als te laat. Welke vergelijking is ook `True` bij 5?"),
            Mistake("swapped", wrong=_d("delay >= 5", '"on time"', '"late"'),
                    en="`{call}` gives `{got}`: the two classes the wrong way round. A delay of 5 minutes or more is `\"late\"`.",
                    nl="`{call}` geeft `{got}`: de twee klassen andersom. Een vertraging van 5 minuten of meer is `\"late\"`."),
            Mistake("return-in-loop", wrong=_RETURN_IN_LOOP,
                    en="`{call}` gives `{got}`: the function stops after the first delay. Is `return classes` indented inside the loop? It belongs under the loop, as far in as `for`.",
                    nl="`{call}` geeft `{got}`: de functie stopt na de eerste vertraging. Staat `return classes` ingesprongen in de loop? Die hoort onder de loop, even ver ingesprongen als `for`."),
            # the exercise is about classes, so `class` is a likely name; Python keeps that word for itself
            Mistake("class-name", syntax=r"^(?:[ \t]*class[ \t]*=(?!=)|[^#\n]*\bfor[ \t]+class[ \t]+in\b|[^#\n]*\([ \t]*class[ \t]*\))",
                    en="Python cannot read the line `{line}`: `class` is a word Python keeps for itself (a later lesson shows what for), so it cannot be a name. Call the class of one delay something else, such as `delay_class`.",
                    nl="Python kan de regel `{line}` niet lezen: `class` is een woord dat Python voor zichzelf houdt (een latere les laat zien waarvoor), dus het kan geen naam zijn. Noem de klasse van één vertraging anders, zoals `delay_class`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a loop that adds a class to `classes` for each delay: `\"late\"` or `\"on time\"`.",
                      "nl": "Vervang de `...` door een loop die voor elke vertraging een klasse aan `classes` toevoegt: `\"late\"` of `\"on time\"`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with one list of delays, and the `def` line of your function has a different number of names between its parentheses. Keep `(delays)`.",
                          "nl": "De check roept `{call}` aan, met één lijst vertragingen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(delays)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Is there one class for each delay, and is each class written exactly as `\"late\"` or `\"on time\"`?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Is er één klasse voor elke vertraging, en staat elke klasse er precies als `\"late\"` of `\"on time\"`?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for any list of delays, also an empty one?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor elke lijst vertragingen, ook een lege?"},
            "correct": {"en": "Your function is right in all {n} cases. The delays were numbers, a label for regression; now each one is one of two classes, a label for classification.",
                        "nl": "Je functie klopt in alle {n} gevallen. De vertragingen waren getallen, een label voor regressie; nu hoort elke vertraging bij een van twee klassen, een label voor classificatie."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("delay-classes", _START, "empty", None),
    ("delay-classes", _SOLUTION, "correct", None),
    ("delay-classes", _d("delay < 5", '"on time"', '"late"'), "correct", None),
    ("delay-classes", "def delay_classes(delays):\n    return [\"late\" if d >= 5 else \"on time\" for d in delays]", "correct", None),
    ("delay-classes", "def delay_classes(delays):\n    classes = []\n    for delay in delays:\n        classes.append(delay >= 5)\n    return classes", "wrong", "bools"),
    ("delay-classes", _d("delay > 5", '"late"', '"on time"'), "wrong", "greater"),
    ("delay-classes", _d("delay <= 5", '"on time"', '"late"'), "wrong", "greater"),
    ("delay-classes", _d("delay >= 5", '"on time"', '"late"'), "wrong", "swapped"),
    ("delay-classes", _d("delay >= 5", '"Late"', '"on time"'), "wrong", None),
    ("delay-classes", "def delay_classes(delays):\n    return [\"on time\", \"late\", \"on time\", \"late\"]", "almost", None),
    ("delay-classes", _RETURN_IN_LOOP, "wrong", "return-in-loop"),
    ("delay-classes", _CLASS_NAME, "error", "class-name"),
    ("delay-classes", _CLASS_NAME.replace("class = \"late\"", "the_class = \"late\"").replace("class = \"on time\"", "the_class = \"on time\"")
     .replace("append(class)", "append(the_class)"), "correct", None),
]
