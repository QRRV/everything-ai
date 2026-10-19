# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson np-arrays: a table typed in as a NumPy array, and a function that reads the shape and
dtype of an array. And the numbers in the picture and the worked example."""
import re

from everything_ai.check import CodeBox, FunctionBox, Mistake, same_array

# The picture: a vector and a matrix.
VECTOR = [7, 9, 4]
MATRIX = [[1, 2, 3], [4, 5, 6]]

# The lesson is done when these are correct. "shape-of" is the choice question in the lesson.
REQUIRED = ["shape-of", "rain", "describe"]

RAIN = [[0, 3, 5, 1], [2, 0, 0, 4], [6, 1, 0, 0]]
_RAIN_START = "import numpy as np\n\nrain = ...\nrain\n"
_RAIN = "import numpy as np\n\nrain = np.array([[0, 3, 5, 1], [2, 0, 0, 4], [6, 1, 0, 0]])\nrain"


def _array(got, shape) -> bool:
    """The learner's last line gave a NumPy array of this shape."""
    return type(got).__name__ == "ndarray" and got.shape == shape


def _same_numbers(got, numbers) -> bool:
    return got.tolist() == numbers


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_NO_NP = Mistake("no-import", error=lambda e: isinstance(e, NameError) and "'np'" in str(e),
                 en="Python does not know `np` yet. Put the line `import numpy as np` at the top: it imports NumPy under the short name `np`.",
                 nl="Python kent `np` nog niet. Houd de regel `import numpy as np` bovenaan: die importeert NumPy onder de korte naam `np`.")
_NO_PARENS = Mistake("attribute-call", syntax=r"(?<!np)\.(shape|dtype)\s*\(",
                     en="`.shape` and `.dtype` have no parentheses: they are not something the array does, but something it has. Write them without `()`.",
                     nl="`.shape` en `.dtype` hebben geen haakjes: het is niet iets wat de array doet, maar iets wat hij heeft. Schrijf ze zonder `()`.")
_SEVERAL = Mistake("several-lists", error=lambda e: isinstance(e, TypeError) and any(
                       w in str(e) for w in ("positional argument", "data type", "Field elements", "not understood")),
                   en="`np.array` gets one list, and yours gets the weeks one by one. Put square brackets around all three weeks, so that they form one big list: `np.array([[...], [...], [...]])`.",
                   nl="`np.array` krijgt één lijst, en die van jou krijgt de weken één voor één. Zet vierkante haken om alle drie de weken, zodat ze één grote lijst vormen: `np.array([[...], [...], [...]])`.")
_COMMA = Mistake("missing-comma", error=lambda e: isinstance(e, TypeError) and "indices must be integers" in str(e),
                 en="A comma is missing between two weeks. Between two inner lists goes a comma: `], [`.",
                 nl="Er ontbreekt een komma tussen twee weken. Tussen twee binnenste lijsten staat een komma: `], [`.")
_UNPACK = Mistake("unpack-count", error=lambda e: isinstance(e, ValueError) and "values to unpack" in str(e),
                  en="With `{call}`, the shape has two numbers, and the left of your `=` has another number of names. Give it two names: `rows, cols = grid.shape`.",
                  nl="Met `{call}` heeft de shape twee getallen, en links van je `=` staat een ander aantal namen. Geef hem twee namen: `rows, cols = grid.shape`.")


_DESCRIBE_START = "import numpy as np\n\n\ndef describe(table):\n    grid = np.array(table)\n    ...\n"
_DESCRIBE = ("import numpy as np\n\n\ndef describe(table):\n    grid = np.array(table)\n    rows, cols = grid.shape\n"
             "    return f\"{rows} by {cols}, {grid.dtype}\"")

EXERCISES = {
    "rain": CodeBox(
        name="rain",
        start=_RAIN_START,
        solution=_RAIN,
        # This lesson is about the array itself: a list with the right numbers is not it yet (template 4.3).
        compare=same_array,
        first=[_DOTS],
        mistakes=[
            _NO_NP, _COMMA, _SEVERAL, _NO_PARENS,
            Mistake("ragged", error=lambda e: isinstance(e, ValueError) and "inhomogeneous" in str(e),
                    en="One of your weeks has a different number of values from the others. An array is a full grid: every row needs 4 numbers, Monday to Thursday.",
                    nl="Een van je weken heeft een ander aantal waarden dan de andere. Een array is een volledig rooster: elke rij heeft 4 getallen nodig, maandag tot en met donderdag."),
            Mistake("still-list", matches=lambda got, text: isinstance(got, list),
                    en="The right numbers, but still a list of lists: it has no shape and no dtype, and that is what this lesson is about. Give it to `np.array(...)`, and give the array the name `rain`.",
                    nl="De goede getallen, maar nog een lijst van lijsten: die heeft geen shape en geen dtype, en daar gaat deze les juist over. Geef hem aan `np.array(...)`, en geef de array de naam `rain`."),
            Mistake("shape-last", matches=lambda got, text: isinstance(got, tuple) and got == (3, 4),
                    en="That is the shape, `(3, 4)`, and it is right. But the check needs the array itself: make the last line just `rain`.",
                    nl="Dat is de shape, `(3, 4)`, en die klopt. Maar de check heeft de array zelf nodig: maak de laatste regel gewoon `rain`."),
            Mistake("extra-brackets", matches=lambda got, text: _array(got, (1, 3, 4)),
                    en="Your array has one pair of square brackets too many around the table: its shape is `(1, 3, 4)`. Take away the outer `[` and `]`, so that the big list holds the three weeks directly.",
                    nl="Je array heeft één paar vierkante haken te veel om de tabel: zijn shape is `(1, 3, 4)`. Haal de buitenste `[` en `]` weg, zodat de grote lijst de drie weken direct bevat."),
            Mistake("row-per-day", matches=lambda got, text: _array(got, (4, 3)) and _same_numbers(got.T, RAIN),
                    en="Your array has a row per day: its shape is `(4, 3)`. This one needs a row per week: each inner list is one week, from Monday to Thursday.",
                    nl="Je array heeft een rij per dag: zijn shape is `(4, 3)`. Hier hoort een rij per week: elke binnenste lijst is één week, van maandag tot en met donderdag."),
            Mistake("one-row", matches=lambda got, text: _array(got, (12,)) and got.tolist() == sum(RAIN, []),
                    en="Your array is a vector of 12 numbers: one long row, with shape `(12,)`. Put each week in a list of its own, inside one big list: then it gets 3 rows.",
                    nl="Je array is een vector van 12 getallen: één lange rij, met shape `(12,)`. Zet elke week in een eigen lijst, binnen één grote lijst: dan krijgt hij 3 rijen."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with an array made from the table: a row per week, Monday to Thursday.",
                      "nl": "Vervang de `...` door een array van de tabel: een rij per week, maandag tot en met donderdag."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "no_value": {"en": "The last line of the box shows nothing. Put `rain` on the last line, so the check can see your array.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `rain` op de laatste regel, zodat de check je array kan zien."},
            "wrong": {"en": "Your last line gives:\n\n```\n{got}\n```\n\nand it should be the array of the table: 3 rows, one per week, of 4 numbers each. Are all numbers there, in the order of the table?",
                      "nl": "Je laatste regel geeft:\n\n```\n{got}\n```\n\nen dat moet de array van de tabel zijn: 3 rijen, één per week, van elk 4 getallen. Staan alle getallen erin, in de volgorde van de tabel?"},
            "correct": {"en": "Your array is right: 3 rows, one per week, of 4 numbers each, so `rain.shape` is `(3, 4)`.",
                        "nl": "Je array klopt: 3 rijen, één per week, van elk 4 getallen, dus `rain.shape` is `(3, 4)`."},
        },
    ),
    "describe": FunctionBox(
        name="describe",
        function="describe",
        start=_DESCRIBE_START,
        solution=_DESCRIBE,
        visible=[([[1, 2, 3], [4, 5, 6]],)],
        hidden=[([[1.5, 2.0]],), ([[1], [2], [3], [4]],), ([[0.5, 1, 2], [3, 4, 5]],), ([[7]],)],
        python_values=True,
        first=[_DOTS],
        mistakes=[
            _NO_NP, _NO_PARENS, _UNPACK,
            Mistake("swapped", wrong=_DESCRIBE.replace("{rows} by {cols}", "{cols} by {rows}"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: rows and columns are swapped. In a shape, the rows come first: `rows, cols = grid.shape`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: rijen en kolommen zijn omgedraaid. In een shape komen de rijen eerst: `rows, cols = grid.shape`."),
            Mistake("no-dtype", wrong=_DESCRIBE.replace(", {grid.dtype}", ""),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the dtype is missing. Add it after a comma: `grid.dtype`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: het dtype ontbreekt. Zet het erachter, na een komma: `grid.dtype`."),
            Mistake("shape-itself", wrong=_DESCRIBE.replace("    return f\"{rows} by {cols}, {grid.dtype}\"", "    return grid.shape"),
                    en="`{call}` gives `{got}`: the shape itself, a tuple. The function should give back a text, such as `{want}`: build it with an f-string.",
                    nl="`{call}` geeft `{got}`: de shape zelf, een tuple. De functie moet een tekst teruggeven, zoals `{want}`: maak die met een f-string."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the lines that give back the rows, the columns and the dtype of `grid`, as one text.",
                      "nl": "Vervang de `...` door de regels die de rijen, de kolommen en het dtype van `grid` teruggeven, als één tekst."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with one table, and the `def` line of your function has another number of names between its parentheses. Keep `(table)`.",
                          "nl": "De check roept `{call}` aan, met één tabel, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(table)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does your text have the rows, then the columns, then the dtype?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Heeft je tekst de rijen, dan de kolommen, dan het dtype?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for every table, also one with a float in it?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor elke tabel, ook een met een float erin?"},
            "correct": {"en": "Your function is right in all {n} cases. One float in the table is enough to make the whole array `float64`.",
                        "nl": "Je functie klopt in alle {n} gevallen. Eén float in de tabel is genoeg om de hele array `float64` te maken."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_R = "import numpy as np\n\n"
_D = "import numpy as np\n\n\ndef describe(table):\n    grid = np.array(table)\n"
CASES = [
    ("rain", _RAIN_START, "empty", None),
    ("rain", _RAIN, "correct", None),
    ("rain", _R + "rain = np.array([[0.0, 3, 5, 1], [2, 0, 0, 4], [6, 1, 0, 0]])\nrain", "correct", None),
    ("rain", _R + "week1 = [0, 3, 5, 1]\nweek2 = [2, 0, 0, 4]\nweek3 = [6, 1, 0, 0]\nrain = np.array([week1, week2, week3])\nrain", "correct", None),
    ("rain", _R + "rain = [[0, 3, 5, 1], [2, 0, 0, 4], [6, 1, 0, 0]]\nrain", "wrong", "still-list"),
    ("rain", _R + "rain = np.array([[0, 2, 6], [3, 0, 1], [5, 0, 0], [1, 4, 0]])\nrain", "wrong", "row-per-day"),
    ("rain", _R + "rain = np.array([0, 3, 5, 1, 2, 0, 0, 4, 6, 1, 0, 0])\nrain", "wrong", "one-row"),
    ("rain", _R + "rain = np.array([[0, 3, 5, 1], [2, 0, 0], [6, 1, 0, 0]])\nrain", "error", "ragged"),
    ("rain", "rain = np.array([[0, 3, 5, 1], [2, 0, 0, 4], [6, 1, 0, 0]])\nrain", "error", "no-import"),
    ("rain", _R + "rain = np.array([[0, 3, 5, 1], [2, 0, 0, 4], [6, 1, 0, 1]])\nrain", "wrong", None),
    ("rain", _R + "rain = np.array(...)\nrain", "wrong", "dots-left"),
    ("rain", _R + "rain = np.array([[0, 3, 5, 1], [2, 0, 0, 4], [6, 1, 0, 0]])", "wrong", "no-value"),
    ("rain", _R + "rain = np.array([0, 3, 5, 1], [2, 0, 0, 4], [6, 1, 0, 0])\nrain", "error", "several-lists"),
    ("rain", _R + "w1 = [0, 3, 5, 1]\nw2 = [2, 0, 0, 4]\nrain = np.array(w1, w2)\nrain", "error", "several-lists"),
    ("rain", _R + "rain = np.array([[0, 3, 5, 1] [2, 0, 0, 4], [6, 1, 0, 0]])\nrain", "error", "missing-comma"),
    ("rain", _R + "rain = np.array([[0, 3, 5, 1], [2, 0, 0, 4], [6, 1, 0, 0]])\nprint(rain)", "wrong", "print-last"),
    ("rain", _RAIN + ".shape", "wrong", "shape-last"),
    ("rain", _R + "rain = np.array([[[0, 3, 5, 1], [2, 0, 0, 4], [6, 1, 0, 0]]])\nrain", "wrong", "extra-brackets"),
    ("rain", _RAIN + "\nrain.shape()", "error", "attribute-call"),
    ("describe", _DESCRIBE_START, "empty", None),
    ("describe", _DESCRIBE, "correct", None),
    ("describe", _D + "    return f\"{len(table)} by {len(table[0])}, {grid.dtype}\"", "correct", None),
    ("describe", _D + "    return str(grid.shape[0]) + \" by \" + str(grid.shape[1]) + \", \" + str(grid.dtype)", "correct", None),
    ("describe", _DESCRIBE.replace("{rows} by {cols}", "{cols} by {rows}"), "wrong", "swapped"),
    ("describe", _DESCRIBE.replace(", {grid.dtype}", ""), "wrong", "no-dtype"),
    ("describe", _D + "    return grid.shape", "wrong", "shape-itself"),
    ("describe", _D + "    rows, cols = grid.shape()\n    return f\"{rows} by {cols}, {grid.dtype}\"", "error", "attribute-call"),
    ("describe", _D + "    rows, cols, kind = grid.shape\n    return f\"{rows} by {cols}, {kind}\"", "error", "unpack-count"),
    ("describe", _D.replace("import numpy as np\n\n\n", "") + "    rows, cols = grid.shape\n    return f\"{rows} by {cols}, {grid.dtype}\"", "error", "no-import"),
    ("describe", _D + "    rows, cols = grid.shape\n    return f\"{rows} by {cols}, int64\"", "almost", None),
]
