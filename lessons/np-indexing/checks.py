# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson np-indexing: a column taken out of an array, and the amounts above a limit, picked with
a boolean mask. And the numbers in the picture."""
import re

from everything_ai.check import Code, CodeBox, FunctionBox, Mistake, uses_loop

# The picture: one matrix, and what three selections pick from it.
PICTURE = [[1, 2, 3], [4, 5, 6]]


def picture(labels: list[str], gives: str) -> list[tuple]:
    """Three panels: one number, a column, and a mask. `labels` names each and `gives` is the word for "gives",
    in the lesson's language."""
    rows = [[str(x) for x in row] for row in PICTURE]
    above = {(i, j) for i, row in enumerate(PICTURE) for j, x in enumerate(row) if x > 2}
    return [
        (labels[0], rows, ["m[1, 2]", f"{gives} 6"], {(1, 2)}),
        (labels[1], rows, ["m[:, 1]", f"{gives} array([2, 5])"], {(0, 1), (1, 1)}),
        (labels[2], rows, ["m[m > 2]", f"{gives} array([3, 4, 5, 6])"], above),
    ]


# The lesson is done when these are correct. "which-part" is the choice question in the lesson.
REQUIRED = ["which-part", "wednesdays", "wet-days"]

RAIN = [[0, 3, 5, 1], [2, 0, 0, 4], [6, 1, 0, 0]]
_RAIN_LINE = "rain = np.array([[0, 3, 5, 1], [2, 0, 0, 4], [6, 1, 0, 0]])"
_WED_START = "import numpy as np\n\n" + _RAIN_LINE + "\nwednesdays = ...\nwednesdays\n"
_WED = "import numpy as np\n\n" + _RAIN_LINE + "\nwednesdays = rain[:, 2]\nwednesdays"


def _gives(numbers):
    """The learner's last line gave a NumPy array with exactly these numbers, in this shape."""
    return lambda got, text: type(got).__name__ == "ndarray" and got.tolist() == numbers


_DAYS = [{"en": "Mondays", "nl": "maandagen"}, {"en": "Tuesdays", "nl": "dinsdagen"},
         {"en": "Wednesdays", "nl": "woensdagen"}, {"en": "Thursdays", "nl": "donderdagen"}]


def _column(got) -> int | None:
    """Which column of the rain the learner's last line gave, as a vector; None if it is no column."""
    if type(got).__name__ != "ndarray" or got.ndim != 1:
        return None
    for j in range(4):
        if got.tolist() == [row[j] for row in RAIN]:
            return j
    return None


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_PAST_END = Mistake("past-the-end", error=lambda e: isinstance(e, IndexError) and "out of bounds" in str(e),
                    en="Your index is past the end of `rain`. It has 3 rows, numbered 0 to 2, and 4 columns, numbered 0 to 3: counting starts at 0.",
                    nl="Je index ligt voorbij het eind van `rain`. Die heeft 3 rijen, genummerd 0 tot en met 2, en 4 kolommen, genummerd 0 tot en met 3: tellen begint bij 0.")

_WET_START = "import numpy as np\n\n\ndef wet_days(rain, limit):\n    grid = np.array(rain)\n    ...\n"
_WET = "import numpy as np\n\n\ndef wet_days(rain, limit):\n    grid = np.array(rain)\n    wet = grid > limit\n    return grid[wet]"

EXERCISES = {
    "wednesdays": CodeBox(
        name="wednesdays",
        start=_WED_START,
        solution=_WED,
        keep=["import numpy as np", _RAIN_LINE],
        # The check also runs the box with other rain: numbers typed in by hand do not change with it.
        others=[{"rain": Code("np.array([[1, 2, 7, 3], [4, 5, 8, 6]])")}],
        first=[_DOTS],
        mistakes=[
            _PAST_END,
            Mistake("row-not-column", matches=_gives([6, 1, 0, 0]),
                    en="`wednesdays` is row 2: all four days of week 3. A column takes one number from every row: first `:` for all rows, then a comma and the column.",
                    nl="`wednesdays` is rij 2: alle vier de dagen van week 3. Een kolom pakt één getal uit elke rij: eerst `:` voor alle rijen, dan een komma en de kolom."),
            Mistake("other-column", matches=lambda got, text: _column(got) not in (None, 2),
                    detail=lambda got, text: {"col": _column(got), "days": _DAYS[_column(got)]},
                    en="That is column {col}: the {days}. Counting starts at 0, so Monday is column 0. Which column is Wednesday?",
                    nl="Dat is kolom {col}: de {days}. Tellen begint bij 0, dus maandag is kolom 0. Welke kolom is woensdag?"),
            Mistake("one-number", matches=lambda got, text: isinstance(got, (int, float)) or type(got).__module__ == "numpy" and getattr(got, "ndim", 1) == 0,
                    en="That is one number: one week, one day. The task asks for the Wednesday of every week: put `:` where the row goes.",
                    nl="Dat is één getal: één week, één dag. De opdracht vraagt de woensdag van elke week: zet `:` waar de rij staat."),
            Mistake("slice-column", matches=_gives([[5], [0], [0]]),
                    en="The right numbers, but as a matrix of 3 rows of 1 number each. Give the column as one number instead of a slice: then it is a vector.",
                    nl="De goede getallen, maar als matrix van 3 rijen van elk 1 getal. Geef de kolom als één getal in plaats van een slice: dan is het een vector."),
        ],
        form=[
            Mistake("with-loop", state="almost", matches=lambda got, text: uses_loop(text),
                    en="The right numbers, but with a loop. A column is one step, without a loop: `rain[:, column]`.",
                    nl="De goede getallen, maar met een loop. Een kolom is één stap, zonder loop: `rain[:, kolom]`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the column of Wednesdays, taken out of `rain`.",
                      "nl": "Vervang de `...` door de kolom met woensdagen, uit `rain` gehaald."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug."},
            "no_value": {"en": "The last line of the box shows nothing. Put `wednesdays` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `wednesdays` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives:\n\n```\n{got}\n```\n\nand it should be the rain of the three Wednesdays: one number from every row, from the column of Wednesday.",
                      "nl": "Je laatste regel geeft:\n\n```\n{got}\n```\n\nen dat moet de regen van de drie woensdagen zijn: één getal uit elke rij, uit de kolom van woensdag."},
            "correct": {"en": "`rain[:, 2]` takes column 2, the Wednesdays, from every week: `array([5, 0, 0])`.",
                        "nl": "`rain[:, 2]` pakt kolom 2, de woensdagen, uit elke week: `array([5, 0, 0])`."},
        },
    ),
    "wet-days": FunctionBox(
        name="wet-days",
        function="wet_days",
        start=_WET_START,
        solution=_WET,
        visible=[(RAIN, 2)],
        hidden=[(RAIN, 0), (RAIN, 10), ([[1.5, 2.5]], 2), ([[3, 3], [1, 4]], 3)],
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("mask-on-list", syntax=r"^[^#\n]*\brain\s*(\[|[<>])",
                    en="`rain` is the list of lists the check passes in. A list cannot be compared with a number, and it cannot take a mask. Use `grid`, the array the function made of it.",
                    nl="`rain` is de lijst van lijsten die de check meegeeft. Een lijst kun je niet met een getal vergelijken, en hij kan geen mask nemen. Gebruik `grid`, de array die de functie ervan maakte."),
            Mistake("if-on-array", error=lambda e: isinstance(e, ValueError) and "truth value" in str(e),
                    en="With `{call}`, an `if` gets a comparison on a whole array. `if` needs one `True` or `False`, and a comparison on an array gives one for every number. Put the mask between the square brackets of `grid` instead.",
                    nl="Met `{call}` krijgt een `if` een vergelijking op een hele array. `if` heeft één `True` of `False` nodig, en een vergelijking op een array geeft er een voor elk getal. Zet het mask in plaats daarvan tussen de vierkante haken van `grid`."),
            Mistake("fixed-limit", state="almost", wrong=_WET.replace("grid > limit", "grid > 2"),
                    en="Right for the example. But `{call}` gives `{got}`, and it should give `{want}`: your mask compares with `2`, not with `limit`. Use the name `limit`, so the function works for every limit.",
                    nl="Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn: je mask vergelijkt met `2`, niet met `limit`. Gebruik de naam `limit`, dan werkt de functie voor elke grens."),
            Mistake("at-least", wrong=_WET.replace("grid > limit", "grid >= limit"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your mask also keeps the amounts equal to `limit`. The task asks for more than `limit`: use `>`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je mask kiest ook de hoeveelheden gelijk aan `limit`. De opdracht vraagt om meer dan `limit`: gebruik `>`."),
            Mistake("mask-itself", wrong=_WET.replace("    return grid[wet]", "    return wet"),
                    en="`{call}` gives the mask itself: `True` or `False` for every number. Put the mask between the square brackets of `grid`: that gives the numbers where it is `True`.",
                    nl="`{call}` geeft het mask zelf: `True` of `False` voor elk getal. Zet het mask tussen de vierkante haken van `grid`: dat geeft de getallen waar het `True` is."),
            Mistake("count", wrong=_WET.replace("    return grid[wet]", "    return len(grid[wet])"),
                    en="`{call}` gives `{got}`: how many there are. The function should give back the amounts themselves, `{want}`.",
                    nl="`{call}` geeft `{got}`: hoeveel het er zijn. De functie moet de hoeveelheden zelf teruggeven, `{want}`."),
            Mistake("below", wrong=_WET.replace("grid > limit", "grid < limit"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: these are the amounts below `limit`. Turn the comparison around: more than `limit`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat zijn de hoeveelheden onder `limit`. Draai de vergelijking om: meer dan `limit`."),
        ],
        form=[
            Mistake("with-loop", state="almost", matches=lambda got, text: uses_loop(text),
                    en="Right in every case, but with a loop. This lesson is about the mask, which does the same without a loop: `grid > limit` gives the mask, and `grid[...]` with the mask picks the amounts.",
                    nl="Goed in elk geval, maar met een loop. Deze les gaat over het mask, dat hetzelfde doet zonder loop: `grid > limit` geeft het mask, en `grid[...]` met het mask kiest de hoeveelheden."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a mask of the amounts above `limit`, and give back the amounts it picks.",
                      "nl": "Vervang de `...` door een mask van de hoeveelheden boven `limit`, en geef de hoeveelheden terug die het kiest."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the rain and a limit, and the `def` line of your function has another number of names between its parentheses. Keep `(rain, limit)`.",
                          "nl": "De check roept `{call}` aan, met de regen en een grens, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(rain, limit)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does your mask keep the amounts above `limit`, and do you give back the amounts it picks?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Kiest je mask de hoeveelheden boven `limit`, en geef je de hoeveelheden terug die het kiest?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for every rain and every limit?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor elke regen en elke grens?"},
            "correct": {"en": "Your function is right in all {n} cases. With a limit above every amount, the mask is `False` everywhere, and you get an empty array: `array([], dtype=int64)`.",
                        "nl": "Je functie klopt in alle {n} gevallen. Met een grens boven elke hoeveelheid is het mask overal `False`, en krijg je een lege array: `array([], dtype=int64)`."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_W = "import numpy as np\n\n" + _RAIN_LINE + "\n"
_F = "import numpy as np\n\n\ndef wet_days(rain, limit):\n    grid = np.array(rain)\n"
CASES = [
    ("wednesdays", _WED_START, "empty", None),
    ("wednesdays", _WED, "correct", None),
    ("wednesdays", _W + "wednesdays = rain[0:3, 2]\nwednesdays", "correct", None),
    ("wednesdays", _W + "wednesdays = rain[2]\nwednesdays", "wrong", "row-not-column"),
    ("wednesdays", _W + "wednesdays = rain[:, 3]\nwednesdays", "wrong", "other-column"),
    ("wednesdays", _W + "wednesdays = rain[:, 0]\nwednesdays", "wrong", "other-column"),
    ("wednesdays", _W + "wednesdays = rain[:, 2]\nprint(wednesdays)", "wrong", "print-last"),
    ("wednesdays", _W + "days = []\nfor week in rain:\n    days.append(week[2])\nwednesdays = np.array(days)\nwednesdays", "almost", "with-loop"),
    ("wednesdays", _W + "wednesdays = rain.T[2]\nwednesdays", "correct", None),
    ("wednesdays", _W + "wednesdays = rain[:, \"Wednesday\"]\nwednesdays", "error", None),
    ("wednesdays", _W + "wednesdays = rain[0, 2]\nwednesdays", "wrong", "one-number"),
    ("wednesdays", _W + "wednesdays = rain[:, 2:3]\nwednesdays", "wrong", "slice-column"),
    ("wednesdays", _W + "wednesdays = [5, 0, 0]\nwednesdays", "almost", "other-value"),
    ("wednesdays", _W + "wednesdays = list(rain[:, 2])\nwednesdays", "correct", None),
    ("wednesdays", _W + "wednesdays = np.array([5, 0, 0])\nwednesdays", "almost", "other-value"),
    ("wednesdays", _W + "wednesdays = rain[:, 4]\nwednesdays", "error", "past-the-end"),
    ("wednesdays", "import numpy as np\n\nrain = np.array([[0, 3, 5, 1], [2, 0, 0, 4], [6, 1, 0, 9]])\nwednesdays = rain[:, 2]\nwednesdays", "wrong", "changed-line"),
    ("wednesdays", _W + "wednesdays = rain[:, ...]\nwednesdays", "wrong", "dots-left"),
    ("wet-days", _WET_START, "empty", None),
    ("wet-days", _WET, "correct", None),
    ("wet-days", _F + "    return grid[grid > limit]", "correct", None),
    ("wet-days", _WET.replace("grid > limit", "grid >= limit"), "wrong", "at-least"),
    ("wet-days", _WET.replace("    return grid[wet]", "    return wet"), "wrong", "mask-itself"),
    ("wet-days", _WET.replace("    return grid[wet]", "    return len(grid[wet])"), "wrong", "count"),
    ("wet-days", _WET.replace("grid > limit", "grid < limit"), "wrong", "below"),
    ("wet-days", _WET.replace("grid > limit", "grid > 2"), "almost", "fixed-limit"),
    ("wet-days", _F + "    return rain[grid > limit]", "error", "mask-on-list"),
    ("wet-days", _F + "    return grid[rain > limit]", "error", "mask-on-list"),
    ("wet-days", _F + "    out = []\n    for row in grid:\n        if row > limit:\n            out.append(row)\n    return out", "error", "if-on-array"),
    ("wet-days", _F + "    out = []\n    for row in rain:\n        for x in row:\n            if x > limit:\n                out.append(x)\n    return out", "almost", "with-loop"),
]
