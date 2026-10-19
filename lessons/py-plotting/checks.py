# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson py-plotting: a line chart and a bar chart drawn with matplotlib in a box. The check reads
back what the figure shows (the kind of chart, its values, title and axis labels) and compares it."""
import re

from everything_ai.check import Mistake, PlotBox

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*$", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with the lines that draw the chart.",
                nl="Er staat nog een `...` in je code. Vervang die door de regels die de grafiek tekenen.")
_NO_PLT = Mistake("no-plt", syntax=r"(?s)\A(?!.*import\s+matplotlib\.pyplot\s+as\s+plt).*\bplt\s*\.",
                  en="Python does not know `plt` yet: the box needs the line `import matplotlib.pyplot as plt` at the top. Put it back.",
                  nl="Python kent `plt` nog niet: het vak heeft de regel `import matplotlib.pyplot as plt` bovenaan nodig. Zet hem terug.")
_LENGTHS = Mistake("lengths", error=lambda e: isinstance(e, ValueError) and re.search(r"same first dimension|must be the same size|shape mismatch", str(e)) is not None,
                   en="Python stops while drawing: the two lists do not have the same length. Each point needs one x value and one y value, so give both lists as many values.",
                   nl="Python stopt tijdens het tekenen: de twee lijsten zijn niet even lang. Elk punt heeft één x-waarde en één y-waarde nodig, dus geef beide lijsten evenveel waarden.")
_ONE_LIST = Mistake("one-list", error=lambda e: isinstance(e, TypeError) and "required positional argument" in str(e),
                    en="Python stops while drawing: the chart gets only one list. It needs two: first what goes along the bottom, then what goes up the side.",
                    nl="Python stopt tijdens het tekenen: de grafiek krijgt maar één lijst. Hij heeft er twee nodig: eerst wat langs de onderkant gaat, dan wat langs de zijkant omhooggaat.")
_ASSIGNED = Mistake("plt-assigned", matches=lambda got, text: re.search(r"^\s*plt\s*\.\s*\w+\s*=(?!=)", text, re.MULTILINE) is not None,
                    en="A line such as `plt.title = \"...\"` gives `plt.title` a new value, instead of using it. Call it with parentheses: `plt.title(\"...\")`, and the same for `plt.xlabel` and `plt.ylabel`.",
                    nl="Een regel zoals `plt.title = \"...\"` geeft `plt.title` een nieuwe waarde, in plaats van hem te gebruiken. Roep hem aan met haakjes: `plt.title(\"...\")`, en net zo voor `plt.xlabel` en `plt.ylabel`.")

_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "nothing": {"en": "Your code draws nothing yet. Draw the chart with `plt.plot`, `plt.scatter` or `plt.bar`, before `plt.show()`.",
                "nl": "Je code tekent nog niets. Teken de grafiek met `plt.plot`, `plt.scatter` of `plt.bar`, vóór `plt.show()`."},
    "kind": {"en": "Your code draws {kind}, but this exercise asks for {want_kind}. Draw it with `{draw}`.",
             "nl": "Je code tekent {kind}, maar deze oefening vraagt om {want_kind}. Teken die met `{draw}`."},
    "title": {"en": "Right chart, but it has no title yet. Add `plt.title(...)` with a text that says what the chart shows.",
              "nl": "Goede grafiek, maar hij heeft nog geen titel. Voeg `plt.title(...)` toe, met een tekst die zegt wat de grafiek laat zien."},
    "xlabel": {"en": "Right chart, but the x-axis has no label yet. Add `plt.xlabel(...)` with what is along the bottom.",
               "nl": "Goede grafiek, maar de x-as heeft nog geen label. Voeg `plt.xlabel(...)` toe, met wat er langs de onderkant staat."},
    "ylabel": {"en": "Right chart, but the y-axis has no label yet. Add `plt.ylabel(...)` with what goes up the side.",
               "nl": "Goede grafiek, maar de y-as heeft nog geen label. Voeg `plt.ylabel(...)` toe, met wat er langs de zijkant omhooggaat."},
}

# The lesson is done when these are correct. "which-chart" is the choice question in the lesson.
REQUIRED = ["temps-line", "which-chart", "minutes-bar"]

_HEAD = "import matplotlib.pyplot as plt\n\n"
_TEMPS_DATA = "days = [1, 2, 3, 4, 5, 6, 7]\ntemps = [12, 15, 9, 14, 16, 18, 17]\n"
_TEMPS_START = _HEAD + _TEMPS_DATA + "...\nplt.show()"
_TEMPS = _HEAD + _TEMPS_DATA + ("plt.plot(days, temps)\nplt.title(\"Temperature per day\")\nplt.xlabel(\"day\")\n"
                                "plt.ylabel(\"degrees\")\nplt.show()")
_MIN_DATA = "names = [\"read\", \"code\", \"walk\", \"sleep\"]\nminutes = [30, 45, 20, 480]\n"
_MIN_START = _HEAD + _MIN_DATA + "...\nplt.show()"
_MIN = _HEAD + _MIN_DATA + "plt.bar(names, minutes)\nplt.title(\"Minutes per activity\")\nplt.ylabel(\"minutes\")\nplt.show()"


def _first(got):
    return got["charts"][0] if got and got["charts"] else None


EXERCISES = {
    "temps-line": PlotBox(
        name="temps-line",
        start=_TEMPS_START,
        solution=_TEMPS,
        first=[_DOTS, _ASSIGNED],
        mistakes=[
            _NO_PLT, _LENGTHS, _ONE_LIST,
            Mistake("no-x", matches=lambda got, text: (c := _first(got)) is not None and c["kind"] == "line"
                    and c["y"] == [12, 15, 9, 14, 16, 18, 17] and c["x"] == list(range(7)),
                    en="Right temperatures, but your line starts at day 0. With only `temps`, Python numbers the points itself, from 0. Give the days first, then the temperatures.",
                    nl="Goede temperaturen, maar je lijn begint bij dag 0. Met alleen `temps` nummert Python de punten zelf, vanaf 0. Geef eerst de dagen, dan de temperaturen."),
            Mistake("swapped", matches=lambda got, text: (c := _first(got)) is not None
                    and c["x"] == [12, 15, 9, 14, 16, 18, 17] and c["y"] == [1, 2, 3, 4, 5, 6, 7],
                    en="Your chart has the temperatures along the bottom and the days up the side. The first list goes along the bottom: give the days first, then the temperatures.",
                    nl="Je grafiek heeft de temperaturen langs de onderkant en de dagen langs de zijkant. De eerste lijst gaat langs de onderkant: geef eerst de dagen, dan de temperaturen."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the lines that draw `temps` as a line, with `days` along the bottom, a title and a label on each axis.",
                      "nl": "Vervang de `...` door de regels die `temps` tekenen als lijn, met `days` langs de onderkant, een titel en een label bij elke as."},
            "cleared": {"en": "There is no code in the box any more. Start with `import matplotlib.pyplot as plt`, then the lists `days` and `temps`, your lines, and `plt.show()`.",
                        "nl": "Er staat geen code meer in het vak. Begin met `import matplotlib.pyplot as plt`, dan de lijsten `days` en `temps`, je regels, en `plt.show()`."},
            "values": {"en": "Your line goes through the x values `{got_x}` and the y values `{got}`, and it should go through `{want_x}` and `{want}`. Use the lists `days` and `temps` as they are.",
                       "nl": "Je lijn gaat door de x-waarden `{got_x}` en de y-waarden `{got}`, en dat moet `{want_x}` en `{want}` zijn. Gebruik de lijsten `days` en `temps` zoals ze zijn."},
            "correct": {"en": "Your line chart is right: the temperature of each day, with a title and both axis labels.",
                        "nl": "Je lijngrafiek klopt: de temperatuur van elke dag, met een titel en beide aslabels."},
        },
    ),
    "minutes-bar": PlotBox(
        name="minutes-bar",
        start=_MIN_START,
        solution=_MIN,
        first=[_DOTS, _ASSIGNED],
        mistakes=[
            _NO_PLT, _LENGTHS, _ONE_LIST,
            Mistake("bar-swapped", matches=lambda got, text: (c := _first(got)) is not None and c["kind"] == "bar" and c["x"] == [30, 45, 20, 480],
                    en="`plt.bar` gets the names first, along the bottom, and then the heights. Swap the two lists.",
                    nl="`plt.bar` krijgt eerst de namen, langs de onderkant, en dan de hoogtes. Draai de twee lijsten om."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the lines that draw `minutes` as bars, one per name, with a title and a label for the y-axis.",
                      "nl": "Vervang de `...` door de regels die `minutes` tekenen als staven, één per naam, met een titel en een label bij de y-as."},
            "cleared": {"en": "There is no code in the box any more. Start with `import matplotlib.pyplot as plt`, then the lists `names` and `minutes`, your lines, and `plt.show()`.",
                        "nl": "Er staat geen code meer in het vak. Begin met `import matplotlib.pyplot as plt`, dan de lijsten `names` en `minutes`, je regels, en `plt.show()`."},
            "values": {"en": "Your bars are `{got_x}` with heights `{got}`, and they should be `{want_x}` with heights `{want}`. Use the lists `names` and `minutes` as they are.",
                       "nl": "Je staven zijn `{got_x}` met hoogtes `{got}`, en dat moet `{want_x}` met hoogtes `{want}` zijn. Gebruik de lijsten `names` en `minutes` zoals ze zijn."},
            "correct": {"en": "Your bar chart is right: one bar per activity, as high as its minutes, with a title and a label on the y-axis.",
                        "nl": "Je staafdiagram klopt: één staaf per activiteit, zo hoog als het aantal minuten, met een titel en een label bij de y-as."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_T = _HEAD + _TEMPS_DATA
_M = _HEAD + _MIN_DATA
CASES = [
    ("temps-line", _TEMPS_START, "empty", None),
    ("temps-line", _TEMPS, "correct", None),
    ("temps-line", _TEMPS.replace("Temperature per day", "Seven days of weather"), "correct", None),
    ("temps-line", _TEMPS.replace("plt.plot(days, temps)", "plt.plot(temps)"), "wrong", "no-x"),
    ("temps-line", _TEMPS.replace("plt.plot(days, temps)", "plt.plot(temps, days)"), "wrong", "swapped"),
    ("temps-line", _TEMPS.replace("plt.plot(days, temps)", "plt.scatter(days, temps)"), "wrong", "kind"),
    ("temps-line", _TEMPS.replace("plt.plot(days, temps)", "plt.bar(days, temps)"), "wrong", "kind"),
    ("temps-line", _TEMPS.replace("plt.title(\"Temperature per day\")\n", ""), "almost", "no-title"),
    ("temps-line", _TEMPS.replace("plt.xlabel(\"day\")\n", ""), "almost", "no-xlabel"),
    ("temps-line", _TEMPS.replace("plt.ylabel(\"degrees\")\n", ""), "almost", "no-ylabel"),
    ("temps-line", _T + "plt.title(\"Temperature per day\")\nplt.show()", "wrong", "nothing"),
    ("temps-line", _TEMPS.replace("plt.plot(days, temps)", "plt.plot(days, [12, 15, 9])"), "error", "lengths"),
    ("temps-line", _TEMPS.replace("import matplotlib.pyplot as plt\n\n", ""), "error", "no-plt"),
    ("temps-line", _TEMPS.replace("plt.plot(days, temps)", "plt.plot(days, temps)\n..."), "wrong", "dots-left"),
    ("minutes-bar", _MIN_START, "empty", None),
    ("minutes-bar", _MIN, "correct", None),
    ("minutes-bar", _MIN.replace("plt.bar(names, minutes)", "plt.plot(names, minutes)"), "wrong", "kind"),
    ("minutes-bar", _MIN.replace("plt.bar(names, minutes)", "plt.bar(minutes, names)"), "wrong", "bar-swapped"),
    ("minutes-bar", _MIN.replace("plt.ylabel(\"minutes\")\n", ""), "almost", "no-ylabel"),
    ("minutes-bar", _MIN.replace("plt.bar(names, minutes)", "plt.bar(names, [30, 45, 20, 48])"), "wrong", "values"),
    ("minutes-bar", _MIN.replace("plt.bar(names, minutes)", "plt.barh(names, minutes)"), "wrong", "kind"),
    ("minutes-bar", _MIN.replace("plt.bar(names, minutes)", "plt.plot(minutes, names)"), "wrong", "kind"),
    ("minutes-bar", _MIN.replace("plt.bar(names, minutes)", "plt.bar(minutes)"), "error", "one-list"),
    ("temps-line", _TEMPS.replace("plt.title(\"Temperature per day\")", "plt.titel(\"Temperature per day\")"), "error", None),
    ("temps-line", _TEMPS.replace("plt.plot(days, temps)", "plt.plot(Days, temps)"), "error", None),
    ("temps-line", _TEMPS.replace("plt.title(\"Temperature per day\")", "plt.title(\"Temperature per day)"), "error", None),
    ("temps-line", _TEMPS.replace("plt.title(\"Temperature per day\")", "plt.title = \"Temperature per day\""), "wrong", "plt-assigned"),
    ("temps-line", _TEMPS, "correct", None),
    ("temps-line", _T + "plt.plot(days, temps)\nplt.show()\nplt.title(\"Week\")\nplt.xlabel(\"day\")\nplt.ylabel(\"degrees\")", "almost", "after-show"),
    ("temps-line", _TEMPS.replace("plt.title(\"Temperature per day\")", "plt.title(\"Temperature...\")"), "correct", None),
]
