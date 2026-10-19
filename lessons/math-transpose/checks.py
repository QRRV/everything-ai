# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson math-transpose: two models for all students at once, with a transpose, and the transpose of a
nested list in plain Python. And the data of the picture and the worked example."""
import re

from everything_ai.check import Code, CodeBox, FunctionBox, Mistake

# The matrix of the picture, and the houses and two models of the worked example: one row of weights per model.
A = [[1, 2, 3], [4, 5, 6]]
HOUSES = [[62, 3, 0], [85, 4, 1], [110, 5, 1], [48, 2, 0]]
MODELS = [[2, 15, 20], [0.5, 3, 1]]

# The lesson is done when these are correct. "gram" is the choice question in the lesson.
REQUIRED = ["gram", "two-models", "transpose-lists"]

_HEAD = ("import numpy as np\n\nX = np.array([[6, 8], [10, 7], [2, 9]])\n"
         "W = np.array([[0.5, 0.3], [0.2, 0.6]])\n")
_START = _HEAD + "y_pred = ...\ny_pred\n"
_SOLUTION = _HEAD + "y_pred = X @ W.T\ny_pred"


def _models(new: str) -> str:
    return _SOLUTION.replace("y_pred = X @ W.T", new)


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_LISTS_START = "def transpose(rows):\n    ...\n"
_LISTS = ("def transpose(rows):\n    columns = []\n    for j in range(len(rows[0])):\n        column = []\n"
          "        for row in rows:\n            column.append(row[j])\n        columns.append(column)\n    return columns")


def _lists(new: str) -> str:
    return "def transpose(rows):\n" + new


EXERCISES = {
    "two-models": CodeBox(
        name="two-models",
        start=_START,
        solution=_SOLUTION,
        keep=["W = np.array([[0.5, 0.3], [0.2, 0.6]])"],
        # The check also runs the box with other students: numbers typed in by hand do not change with them.
        others=[{"X": Code("np.array([[4, 6], [8, 8]])")}],
        first=[_DOTS],
        mistakes=[
            Mistake("shapes", error=lambda e: isinstance(e, ValueError) and ("matmul" in str(e) or "not aligned" in str(e)),
                    en="Python stops: the shapes do not fit. Write down both shapes: `X` is (3, 2) and `W` is (2, 2). Each student, a row of `X`, must meet each model, a row of `W`: which of the two do you transpose?",
                    nl="Python stopt: de shapes passen niet. Schrijf beide shapes op: `X` is (3, 2) en `W` is (2, 2). Elke leerling, een rij van `X`, moet elk model tegenkomen, een rij van `W`: welke van de twee transponeer je?"),
            Mistake("star", error=lambda e: isinstance(e, ValueError) and "broadcast" in str(e),
                    en="Python stops: `*` multiplies the numbers in the same place, and `X`, (3, 2), and `W`, (2, 2), do not have the same shape. Matrix multiplication is `@`.",
                    nl="Python stopt: `*` vermenigvuldigt de getallen op dezelfde plek, en `X`, (3, 2), en `W`, (2, 2), hebben niet dezelfde shape. Matrixvermenigvuldiging is `@`."),
            Mistake("no-transpose", wrong=_models("y_pred = X @ W"),
                    en="Your last line gives `{got}`. The shapes happen to fit, but each row of `X` now meets a column of `W`, not a model. The models are the rows of `W`: make them columns with `W.T`.",
                    nl="Je laatste regel geeft `{got}`. De shapes passen toevallig, maar elke rij van `X` komt nu een kolom van `W` tegen, geen model. De modellen zijn de rijen van `W`: maak er kolommen van met `W.T`."),
            Mistake("flipped", wrong=_models("y_pred = W @ X.T"),
                    en="Your last line gives `{got}`: the right numbers, but with one row per model and one column per student. The task asks for one row per student and one column per model.",
                    nl="Je laatste regel geeft `{got}`: de goede getallen, maar met één rij per model en één kolom per leerling. De opdracht vraagt om één rij per leerling en één kolom per model."),
            Mistake("one-model", wrong=_models("y_pred = X @ W[0]"),
                    en="Your last line gives `{got}`: the predictions of the first model only. With `W.T`, one multiplication gives both models at once.",
                    nl="Je laatste regel geeft `{got}`: alleen de voorspellingen van het eerste model. Met `W.T` geeft één vermenigvuldiging beide modellen tegelijk."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the predictions of both models for all students: one matrix multiplication, with a transpose.",
                      "nl": "Vervang de `...` door de voorspellingen van beide modellen voor alle leerlingen: één matrixvermenigvuldiging, met een getransponeerde."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other students by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere leerlingen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `y_pred` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `y_pred` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`, and it should have one row per student and one column per model: each number the dot product of a student's row with a model's row.",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet één rij per leerling en één kolom per model hebben: elk getal het inproduct van de rij van een leerling met de rij van een model."},
            "other": {"en": "Right for these students. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you work it out with `X` and `W` themselves?",
                      "nl": "Goed voor deze leerlingen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je het uit met `X` en `W` zelf?"},
            "other_error": {"en": "Right for these students. But with {where}, your code stops with an error. Does it work for any number of students?",
                            "nl": "Goed voor deze leerlingen. Maar met {where} stopt je code met een fout. Werkt hij voor elk aantal leerlingen?"},
            "correct": {"en": "Shape (3, 2): one row per student, one column per model. The first student gets 5.4 from the first model and 6.0 from the second. The check also tried two other students.",
                        "nl": "Shape (3, 2): één rij per leerling, één kolom per model. De eerste leerling krijgt 5,4 van het eerste model en 6,0 van het tweede. De check probeerde ook twee andere leerlingen."},
        },
    ),
    "transpose-lists": FunctionBox(
        name="transpose-lists",
        function="transpose",
        start=_LISTS_START,
        solution=_LISTS,
        visible=[([[1, 2], [3, 4], [5, 6]],)],
        hidden=[([[1, 2], [3, 4]],), ([[7, 8, 9]],), ([[1], [2], [3]],), ([[1, 2, 3], [4, 5, 6]],)],
        first=[_DOTS],
        mistakes=[
            Mistake("count-rows", error=lambda e: isinstance(e, IndexError),
                    en="With `{call}`, Python stops: your code asks a row for a place it does not have. How many columns does `rows` have? `len(rows)` counts the rows, `len(rows[0])` the numbers in one row.",
                    nl="Met `{call}` stopt Python: je code vraagt een rij om een plek die hij niet heeft. Hoeveel kolommen heeft `rows`? `len(rows)` telt de rijen, `len(rows[0])` de getallen in één rij."),
            Mistake("same", wrong=_lists("    return rows"),
                    en="`{call}` gives `{got}`: the rows unchanged. The first row should become the first column: the first new row holds the first number of every old row.",
                    nl="`{call}` geeft `{got}`: de rijen zoals ze waren. De eerste rij moet de eerste kolom worden: de eerste nieuwe rij bevat het eerste getal van elke oude rij."),
            Mistake("flat", wrong=_lists("    out = []\n    for j in range(len(rows[0])):\n        for row in rows:\n            out.append(row[j])\n    return out"),
                    en="`{call}` gives `{got}`: the right order, but in one flat list. Collect each column in a list of its own, and put those lists in the result.",
                    nl="`{call}` geeft `{got}`: de goede volgorde, maar in één platte lijst. Verzamel elke kolom in een eigen lijst, en zet die lijsten in de uitkomst."),
            Mistake("one-column-list", wrong=_lists("    columns = []\n    column = []\n    for j in range(len(rows[0])):\n        for row in rows:\n            column.append(row[j])\n        columns.append(column)\n    return columns"),
                    en="`{call}` gives `{got}`: every new row is the same, long list. Make a new, empty `column` for each `j`, inside the outer loop, so each column gets a list of its own.",
                    nl="`{call}` geeft `{got}`: elke nieuwe rij is dezelfde, lange lijst. Maak voor elke `j` een nieuwe, lege `column`, binnen de buitenste loop, zodat elke kolom een eigen lijst krijgt."),
            Mistake("return-in-loop", wrong=_lists("    columns = []\n    for j in range(len(rows[0])):\n        column = []\n        for row in rows:\n            column.append(row[j])\n        columns.append(column)\n        return columns"),
                    en="`{call}` gives `{got}`: only the first column. Your `return` stands inside the loop, so the function stops after one round. Put `return columns` after the loop, at the same indent as `for`.",
                    nl="`{call}` geeft `{got}`: alleen de eerste kolom. Je `return` staat binnen de loop, dus de functie stopt na één ronde. Zet `return columns` na de loop, even ver ingesprongen als `for`."),
            Mistake("reversed", wrong=_lists("    return rows[::-1]"),
                    en="`{call}` gives `{got}`: the rows in the other order. Transposing swaps rows and columns, not the order of the rows.",
                    nl="`{call}` geeft `{got}`: de rijen in de andere volgorde. Transponeren wisselt rijen en kolommen, niet de volgorde van de rijen."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the columns of `rows`, each as a list, given back as a list of lists with `return`.",
                      "nl": "Vervang de `...` door de kolommen van `rows`, elk als lijst, teruggegeven als lijst van lijsten met `return`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with one list of rows, and the `def` line of your function has a different number of names between its parentheses. Keep `(rows)`.",
                          "nl": "De check roept `{call}` aan, met één lijst van rijen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(rows)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does the first new row hold the first number of every old row, the second the second, and so on?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Bevat de eerste nieuwe rij het eerste getal van elke oude rij, de tweede het tweede, enzovoort?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for any number of rows and columns, also just one?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor elk aantal rijen en kolommen, ook voor maar één?"},
            "correct": {"en": "Your function is right in all {n} cases. 3 rows of 2 numbers become 2 rows of 3: shape (3, 2) becomes (2, 3), just as with `.T`.",
                        "nl": "Je functie klopt in alle {n} gevallen. 3 rijen van 2 getallen worden 2 rijen van 3: shape (3, 2) wordt (2, 3), net als bij `.T`."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("two-models", _START, "empty", None),
    ("two-models", _SOLUTION, "correct", None),
    ("two-models", _models("y_pred = np.dot(X, W.T)"), "correct", None),
    ("two-models", _models("y_pred = (W @ X.T).T"), "correct", None),
    ("two-models", _models("y_pred = X.dot(W.T)"), "correct", None),
    ("two-models", _models("y_pred = X @ W"), "wrong", "no-transpose"),
    ("two-models", _models("y_pred = W @ X.T"), "wrong", "flipped"),
    ("two-models", _models("y_pred = X @ W[0]"), "wrong", "one-model"),
    ("two-models", _models("y_pred = W @ X"), "error", "shapes"),
    ("two-models", _models("y_pred = np.dot(W, X)"), "error", "shapes"),
    ("two-models", _models("y_pred = X * W"), "error", "star"),
    ("two-models", _models("y_pred = np.array([[5.4, 6.0], [7.1, 6.2], [3.7, 5.8]])"), "almost", "other-value"),
    ("two-models", _HEAD + "y_pred = X @ W.T\nprint(y_pred)", "wrong", "print-last"),
    ("transpose-lists", _LISTS_START, "empty", None),
    ("transpose-lists", _LISTS, "correct", None),
    ("transpose-lists", _lists("    return [[row[j] for row in rows] for j in range(len(rows[0]))]"), "correct", None),
    ("transpose-lists", _lists("    return [list(col) for col in zip(*rows)]"), "correct", None),
    ("transpose-lists", _lists("    return rows"), "wrong", "same"),
    ("transpose-lists", _lists("    out = []\n    for j in range(len(rows[0])):\n        for row in rows:\n            out.append(row[j])\n    return out"), "wrong", "flat"),
    ("transpose-lists", _lists("    return rows[::-1]"), "wrong", "reversed"),
    ("transpose-lists", _lists("    columns = []\n    column = []\n    for j in range(len(rows[0])):\n        for row in rows:\n            column.append(row[j])\n        columns.append(column)\n    return columns"), "wrong", "one-column-list"),
    ("transpose-lists", _lists("    columns = []\n    for j in range(len(rows[0])):\n        column = []\n        for row in rows:\n            column.append(row[j])\n        columns.append(column)\n        return columns"), "wrong", "return-in-loop"),
    ("transpose-lists", _lists("    columns = []\n    for j in range(len(rows)):\n        columns.append([row[j] for row in rows])\n    return columns"), "error", "count-rows"),
    ("transpose-lists", _lists("    return [[1, 3, 5], [2, 4, 6]]"), "almost", None),
]
