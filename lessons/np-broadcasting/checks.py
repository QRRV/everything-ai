# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson np-broadcasting: temperatures minus the mean of their time of day, and each town's
share of the votes, with a column of totals. And the numbers in the picture."""
import re

from everything_ai.check import Code, CodeBox, FunctionBox, Mistake, uses_loop

# The picture: a matrix, a vector stretched over its rows, and their sum.
MATRIX = [[1, 2, 3], [4, 5, 6]]
VECTOR = [10, 20, 30]


def picture(labels: list[str], notes: list[str]) -> list[tuple]:
    """Three panels: m, the vector stretched to the shape of m (its copies gray), and m + v."""
    rows = [[str(x) for x in row] for row in MATRIX]
    stretched = [[str(x) for x in VECTOR] for _ in MATRIX]
    total = [[str(x + y) for x, y in zip(row, VECTOR)] for row in MATRIX]
    return [
        (labels[0], rows, [f"shape ({len(MATRIX)}, {len(VECTOR)})"]),
        (labels[1], stretched, [f"shape ({len(VECTOR)},)", notes[0]], {(0, j) for j in range(len(VECTOR))}),
        (labels[2], total, [f"shape ({len(MATRIX)}, {len(VECTOR)})", notes[1]]),
    ]


# The lesson is done when these are correct. "which-fits" is the choice question in the lesson.
REQUIRED = ["which-fits", "centered", "shares"]

_TEMPS_LINE = "temps = np.array([[10, 16, 12], [8, 15, 11], [12, 18, 16], [10, 15, 13]])"
_CENTER_START = "import numpy as np\n\n" + _TEMPS_LINE + "\ncentered = ...\ncentered\n"
_CENTER = "import numpy as np\n\n" + _TEMPS_LINE + "\ncentered = temps - temps.mean(axis=0)\ncentered"


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_NO_FIT = {"en": "NumPy cannot make these shapes fit: from the right, two lengths differ, and neither is 1. ",
           "nl": "NumPy kan deze shapes niet passend maken: van rechts verschillen twee lengtes, en geen van beide is 1. "}

_SHARES_START = "import numpy as np\n\n\ndef shares(votes):\n    votes = np.array(votes)\n    ...\n"
_SHARES = ("import numpy as np\n\n\ndef shares(votes):\n    votes = np.array(votes)\n"
           "    totals = votes.sum(axis=1, keepdims=True)\n    return votes / totals")

EXERCISES = {
    "centered": CodeBox(
        name="centered",
        start=_CENTER_START,
        solution=_CENTER,
        keep=["import numpy as np", _TEMPS_LINE],
        # The check also runs the box with other temperatures: numbers typed in by hand do not change with them.
        others=[{"temps": Code("np.array([[1, 2, 6], [3, 4, 4]])")}],
        first=[_DOTS],
        mistakes=[
            Mistake("no-fit", error=lambda e: isinstance(e, ValueError) and "broadcast" in str(e),
                    en=_NO_FIT["en"] + "`temps` has shape `(4, 3)`, and the means per day `(4,)`. The task asks for the mean of each time of day: one per column.",
                    nl=_NO_FIT["nl"] + "`temps` heeft shape `(4, 3)`, en de gemiddelden per dag `(4,)`. De opdracht vraagt het gemiddelde van elk dagdeel: één per kolom."),
            Mistake("all-mean", wrong=_CENTER.replace("temps.mean(axis=0)", "temps.mean()"),
                    en="Every temperature minus the mean of all twelve. The task asks for the mean of its own time of day: one mean per column, so down the rows.",
                    nl="Elke temperatuur min het gemiddelde van alle twaalf. De opdracht vraagt het gemiddelde van zijn eigen dagdeel: één gemiddelde per kolom, dus langs de rijen omlaag."),
            Mistake("per-day", wrong=_CENTER.replace("temps.mean(axis=0)", "temps.mean(axis=1, keepdims=True)"),
                    en="Every temperature minus the mean of its day: the column of day means was stretched over the columns. The task asks for the mean of its time of day: one per column, so down the rows.",
                    nl="Elke temperatuur min het gemiddelde van zijn dag: de kolom met daggemiddelden werd over de kolommen uitgerekt. De opdracht vraagt het gemiddelde van zijn dagdeel: één per kolom, dus langs de rijen omlaag."),
            Mistake("reversed", wrong=_CENTER.replace("temps - temps.mean(axis=0)", "temps.mean(axis=0) - temps"),
                    en="Each mean minus the temperature: the other way around, so a warm day comes out below 0. Take the means away from `temps`, not `temps` from the means.",
                    nl="Elk gemiddelde min de temperatuur: andersom, dus een warme dag komt onder de 0 uit. Haal de gemiddelden af van `temps`, niet `temps` van de gemiddelden."),
            Mistake("means-only", matches=lambda got, text: type(got).__name__ == "ndarray" and got.size == 3,
                    en="These are only the means, one per time of day. Now take them away from `temps`: NumPy stretches them over the four rows.",
                    nl="Dit zijn alleen de gemiddelden, één per dagdeel. Haal ze nu af van `temps`: NumPy rekt ze uit over de vier rijen."),
        ],
        form=[
            Mistake("with-loop", state="almost", matches=lambda got, text: uses_loop(text),
                    en="The right numbers, but with a loop. Broadcasting does it without one: `temps` minus the means, in one line.",
                    nl="De goede getallen, maar met een loop. Broadcasting doet het zonder: `temps` min de gemiddelden, in één regel."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with `temps` minus the mean of each time of day.",
                      "nl": "Vervang de `...` door `temps` min het gemiddelde van elk dagdeel."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug."},
            "no_value": {"en": "The last line of the box shows nothing. Put `centered` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `centered` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives:\n\n```\n{got}\n```\n\nand it should be every temperature minus the mean of its time of day: the morning mean is 10, so the first morning, 10, becomes 0. Which mean do you take away, and from what?",
                      "nl": "Je laatste regel geeft:\n\n```\n{got}\n```\n\nen dat moet elke temperatuur zijn min het gemiddelde van zijn dagdeel: het ochtendgemiddelde is 10, dus de eerste ochtend, 10, wordt 0. Welk gemiddelde haal je ervan af, en waarvan?"},
            "other": {"en": "Right for these temperatures. But with {where}, your last line gives:\n\n```\n{got}\n```\n\nand it should give:\n\n```\n{want}\n```\n\nDo you calculate the means from `temps` itself, or did you type numbers in?",
                      "nl": "Goed voor deze temperaturen. Maar met {where} geeft je laatste regel:\n\n```\n{got}\n```\n\nen dat moet dit zijn:\n\n```\n{want}\n```\n\nBereken je de gemiddelden uit `temps` zelf, of typte je getallen in?"},
            "correct": {"en": "One mean per column fits `(4, 3)` from the right, and NumPy stretched the means over the four rows: every temperature minus the mean of its time of day.",
                        "nl": "Eén gemiddelde per kolom past van rechts op `(4, 3)`, en NumPy rekte de gemiddelden uit over de vier rijen: elke temperatuur min het gemiddelde van zijn dagdeel."},
        },
    ),
    "shares": FunctionBox(
        name="shares",
        function="shares",
        start=_SHARES_START,
        solution=_SHARES,
        visible=[([[30, 50, 20], [10, 10, 30]],)],
        hidden=[([[1, 1], [3, 1]],), ([[5, 5, 10]],), ([[2], [8]],), ([[1, 2, 1], [0, 3, 1], [4, 4, 2]],)],
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("no-fit", error=lambda e: isinstance(e, ValueError) and "broadcast" in str(e),
                    en="With `{call}`, NumPy cannot stretch the shapes to fit each other. The totals of the towns have shape `(2,)`, and `votes` has shape `(2, 3)`: compared from the right, 2 and 3 do not fit. Keep the totals as a column, with `keepdims=True`: shape `(2, 1)`.",
                    nl="Met `{call}` kan NumPy de shapes niet zo uitrekken dat ze passen. De totalen van de steden hebben shape `(2,)`, en `votes` heeft shape `(2, 3)`: van rechts vergeleken passen 2 en 3 niet. Houd de totalen als kolom, met `keepdims=True`: shape `(2, 1)`."),
            Mistake("keyword-typo", error=lambda e: isinstance(e, TypeError) and "keyword" in str(e),
                    en="With `{call}`, `.sum` gets a named value it does not know. Check the spelling: `axis=` and `keepdims=`, with an s at the end.",
                    nl="Met `{call}` krijgt `.sum` een waarde met een naam die hij niet kent. Kijk de spelling na: `axis=` en `keepdims=`, met een s aan het eind."),
            Mistake("percent", wrong=_SHARES.replace("return votes / totals", "return votes / totals * 100"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: these are percentages. A share here is a number from 0 to 1, so leave out the `* 100`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dit zijn percentages. Een aandeel is hier een getal van 0 tot 1, dus laat de `* 100` weg."),
            Mistake("per-party", wrong=_SHARES.replace("axis=1, keepdims=True", "axis=0, keepdims=True"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: each number is divided by the total of its column, its party. The task asks for each town's share: divide by the total of each row.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elk getal is gedeeld door het totaal van zijn kolom, zijn partij. De opdracht vraagt het aandeel binnen elke stad: deel door het totaal van elke rij."),
            Mistake("all-votes", wrong=_SHARES.replace("votes.sum(axis=1, keepdims=True)", "votes.sum()"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: each number is divided by all votes together. The task asks for each town's share: divide by the total of its own row.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elk getal is gedeeld door alle stemmen samen. De opdracht vraagt het aandeel binnen elke stad: deel door het totaal van zijn eigen rij."),
        ],
        form=[
            Mistake("with-loop", state="almost", matches=lambda got, text: uses_loop(text),
                    en="Right in every case, but with a loop. Broadcasting does it without one: divide `votes` by a column of totals.",
                    nl="Goed in elk geval, maar met een loop. Broadcasting doet het zonder: deel `votes` door een kolom met totalen."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with each number of `votes` divided by the total of its row.",
                      "nl": "Vervang de `...` door elk getal van `votes` gedeeld door het totaal van zijn rij."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with one table of votes, and the `def` line of your function has a different number of names between its parentheses. Keep `(votes)`.",
                          "nl": "De check roept `{call}` aan, met één tabel stemmen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(votes)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Is every number divided by the total of its own row?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Is elk getal gedeeld door het totaal van zijn eigen rij?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for every table?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor elke tabel?"},
            "correct": {"en": "Your function is right in all {n} cases. The totals are a column, shape `(rows, 1)`, so each row is divided by its own total, and each row of shares adds up to 1.",
                        "nl": "Je functie klopt in alle {n} gevallen. De totalen zijn een kolom, shape `(rijen, 1)`, dus wordt elke rij gedeeld door zijn eigen totaal, en telt elke rij aandelen op tot 1."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_T = "import numpy as np\n\n" + _TEMPS_LINE + "\n"
_S = "import numpy as np\n\n\ndef shares(votes):\n    votes = np.array(votes)\n"
CASES = [
    ("centered", _CENTER_START, "empty", None),
    ("centered", _CENTER, "correct", None),
    ("centered", _T + "means = temps.mean(axis=0)\ncentered = temps - means\ncentered", "correct", None),
    ("centered", _T + "centered = temps - temps.mean(axis=0, keepdims=True)\ncentered", "correct", None),
    ("centered", _T + "centered = temps - temps.mean(axis=1)\ncentered", "error", "no-fit"),
    ("centered", _T + "centered = temps - temps.mean()\ncentered", "wrong", "all-mean"),
    ("centered", _T + "centered = temps - temps.mean(axis=1, keepdims=True)\ncentered", "wrong", "per-day"),
    ("centered", _T + "centered = temps.mean(axis=0)\ncentered", "wrong", "means-only"),
    ("centered", _T + "centered = temps - temps.mean(axis=0)\nprint(centered)", "wrong", "print-last"),
    ("centered", _T + "centered = temps.mean(axis=0) - temps\ncentered", "wrong", "reversed"),
    ("centered", _T + "centered = temps.mean(axis=0, keepdims=True)\ncentered", "wrong", "means-only"),
    ("centered", _T + "centered = temps - [10, 16, 13]\ncentered", "almost", "other-value"),
    ("centered", _T + "rows = []\nfor row in temps:\n    rows.append(row - temps.mean(axis=0))\ncentered = np.array(rows)\ncentered", "almost", "with-loop"),
    ("shares", _SHARES_START, "empty", None),
    ("shares", _SHARES, "correct", None),
    ("shares", _S + "    return votes / votes.sum(axis=1, keepdims=True)", "correct", None),
    ("shares", _S + "    return votes / votes.sum(axis=1)", "error", "no-fit"),
    ("shares", _SHARES.replace("axis=1, keepdims=True", "axis=0, keepdims=True"), "wrong", "per-party"),
    ("shares", _SHARES.replace("votes.sum(axis=1, keepdims=True)", "votes.sum()"), "wrong", "all-votes"),
    ("shares", _SHARES.replace("keepdims=True", "keepdim=True"), "error", "keyword-typo"),
    ("shares", _SHARES.replace("return votes / totals", "return votes / totals * 100"), "wrong", "percent"),
    ("shares", _S + "    out = []\n    for row in votes:\n        out.append(row / row.sum())\n    return np.array(out)", "almost", "with-loop"),
]
