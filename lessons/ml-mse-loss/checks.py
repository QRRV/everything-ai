# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-mse-loss: the mean squared error of a weather forecaster in plain Python, and the same as a
function with NumPy. And the picture of the residuals with their squares."""
import re

from everything_ai.check import CodeBox, FunctionBox, Mistake

# Five houses and a model, for the picture: area in m², price in thousands of euros.
AREA = [70, 95, 110, 130, 150]
PRICE = [210, 300, 265, 330, 390]
W, B = 2.2, 50


def picture(labels: dict[str, str]) -> str:
    """The houses, the line of the model, each residual as a stroke, and on each stroke a square with that length as
    its side: the squared residual. The squares are drawn square on screen."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from everything_ai import plots
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.set_xlim(55, 175)
    ax.set_ylim(150, 430)
    fig.canvas.draw()
    px = ax.transData.transform([(0, 0), (1, 1)])
    x_per_y = (px[1, 1] - px[0, 1]) / (px[1, 0] - px[0, 0])   # how many x units one y unit is wide on screen
    xs = [55, 175]
    ax.plot(xs, [W * x + B for x in xs], color=plots.PRED, linestyle="-")
    for x, y in zip(AREA, PRICE):
        y_pred = W * x + B
        side = abs(y - y_pred)
        ax.add_patch(Rectangle((x, min(y, y_pred)), side * x_per_y, side, facecolor=(plots.LOSS, 0.18), edgecolor=plots.LOSS, linewidth=1.2))
        ax.plot([x, x], [y, y_pred], color=plots.RESIDUAL, linestyle="-", linewidth=2.2, zorder=4)
    ax.scatter(AREA, PRICE, color=plots.DATA, s=16, zorder=3)
    ax.set_xlabel(labels["x"])
    ax.set_ylabel(labels["y"])
    ax.set_title(labels["title"])
    html = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return html


# The lesson is done when these are correct. "why-square" is the open question; it does not hold anyone back.
REQUIRED = ["mse-temps", "mse"]

_TEMPS = "temps_y = [12, 15, 9, 10]\ntemps_y_pred = [15, 15, 8, 14]\n"
_TEMPS_START = _TEMPS + "mse_temps = ...\nmse_temps\n"
_TEMPS_SOLUTION = (_TEMPS + "residuals = []\nfor i in range(len(temps_y)):\n    residuals.append(temps_y[i] - temps_y_pred[i])\n"
                   "squares = []\nfor r in residuals:\n    squares.append(r ** 2)\ntotal = 0\nfor s in squares:\n    total = total + s\n"
                   "mse_temps = total / len(squares)\nmse_temps")

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _temps(last: str) -> str:
    """A whole program that ends with `mse_temps = <last>`, after the residuals and their squares."""
    return (_TEMPS + "residuals = []\nfor i in range(len(temps_y)):\n    residuals.append(temps_y[i] - temps_y_pred[i])\n"
            "squares = []\nfor r in residuals:\n    squares.append(r ** 2)\n" + f"mse_temps = {last}\nmse_temps")


_MSE_START = "import numpy as np\n\n\ndef mse(y, y_pred):\n    y = np.array(y)\n    y_pred = np.array(y_pred)\n    ...\n"
_MSE = ("import numpy as np\n\n\ndef mse(y, y_pred):\n    y = np.array(y)\n    y_pred = np.array(y_pred)\n"
        "    residuals = y - y_pred\n    return (residuals ** 2).mean()")


def _mse(new: str) -> str:
    return _MSE.replace("    residuals = y - y_pred\n    return (residuals ** 2).mean()", new)


EXERCISES = {
    "mse-temps": CodeBox(
        name="mse-temps",
        start=_TEMPS_START,
        solution=_TEMPS_SOLUTION,
        keep=["temps_y = [12, 15, 9, 10]", "temps_y_pred = [15, 15, 8, 14]"],
        # The check also runs the box with two other days: a number typed in by hand does not change with them.
        others=[{"temps_y": [20, 22], "temps_y_pred": [21, 25]}],
        first=[_DOTS],
        mistakes=[
            Mistake("no-square", wrong=_temps("sum(residuals) / len(residuals)"), en="Your last line gives `{got}`: a plain mean of the residuals, without squares. Plus and minus cancel each other out there. Square each residual first.", nl="Je laatste regel geeft `{got}`: een gewoon gemiddelde van de residuen, zonder kwadraten. Plus en min vallen daarin tegen elkaar weg. Kwadrateer eerst elk residu."),
            Mistake("no-square-flipped", wrong=_temps("-sum(residuals) / len(residuals)"), en="Your last line gives `{got}`: a plain mean of the residuals, without squares. Plus and minus cancel each other out there. Square each residual first.", nl="Je laatste regel geeft `{got}`: een gewoon gemiddelde van de residuen, zonder kwadraten. Plus en min vallen daarin tegen elkaar weg. Kwadrateer eerst elk residu."),
            Mistake("no-sign", wrong=_temps("sum([abs(r) for r in residuals]) / len(residuals)"),
                    en="Your last line gives `{got}`: the mean of the residuals without their sign. That is a loss too, but not the mean squared error: square each residual instead.",
                    nl="Je laatste regel geeft `{got}`: het gemiddelde van de residuen zonder hun teken. Dat is ook een loss, maar niet de mean squared error: kwadrateer elk residu."),
            Mistake("squares-only", wrong=_temps("squares"),
                    en="Your last line gives `{got}`: one square per day. The loss is one number: take the mean of the squares, step 3 of the recipe.",
                    nl="Je laatste regel geeft `{got}`: één kwadraat per dag. De loss is één getal: neem het gemiddelde van de kwadraten, stap 3 van het recept."),
            Mistake("no-mean", wrong=_temps("sum(squares)"),
                    en="Your last line gives `{got}`: the sum of the squares. The mean squared error is their mean: divide by the number of days.",
                    nl="Je laatste regel geeft `{got}`: de som van de kwadraten. De mean squared error is hun gemiddelde: deel door het aantal dagen."),
            Mistake("mean-then-square", wrong=_temps("(sum(residuals) / len(residuals)) ** 2"),
                    en="Your last line gives `{got}`: the square of the mean residual. The order is the other way round: square each residual first, then take the mean.",
                    nl="Je laatste regel geeft `{got}`: het kwadraat van het gemiddelde residu. De volgorde is andersom: kwadrateer eerst elk residu, en neem dan het gemiddelde."),
            Mistake("one-less", wrong=_temps("sum(squares) / (len(squares) - 1)"),
                    en="Your last line gives `{got}`: you divide by one less than the number of days. The mean divides by the number of days itself.",
                    nl="Je laatste regel geeft `{got}`: je deelt door één minder dan het aantal dagen. Het gemiddelde deelt door het aantal dagen zelf."),
            Mistake("doubled", wrong=_temps("sum([r * 2 for r in residuals]) / len(residuals)"),
                    en="Your last line gives `{got}`: each residual times 2, not squared. Squared is `** 2`, or the residual times itself.",
                    nl="Je laatste regel geeft `{got}`: elk residu keer 2, niet in het kwadraat. Kwadrateren is `** 2`, of het residu keer zichzelf."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the steps of the recipe, and put the mean squared error in `mse_temps`.",
                      "nl": "Vervang de `...` door de stappen van het recept, en zet de mean squared error in `mse_temps`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other days by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere dagen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `mse_temps` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `mse_temps` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`, and it should be the mean of the squared residuals of the four days. Do you take the residual of each day, square it, and then take the mean?",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet het gemiddelde van de gekwadrateerde residuen van de vier dagen zijn. Neem je het residu van elke dag, kwadrateer je het, en neem je dan het gemiddelde?"},
            "other": {"en": "Right for these four days. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you work it out from `temps_y` and `temps_y_pred` themselves?",
                      "nl": "Goed voor deze vier dagen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je het uit met `temps_y` en `temps_y_pred` zelf?"},
            "other_error": {"en": "Right for these four days. But with {where}, your code stops with an error. Does it work for any number of days?",
                            "nl": "Goed voor deze vier dagen. Maar met {where} stopt je code met een fout. Werkt hij voor elk aantal dagen?"},
            "correct": {"en": "The residuals are −3, 0, 1 and −4; squared, 9, 0, 1 and 16; their mean is 6.5. The check also tried two other days, and your code got those right too.",
                        "nl": "De residuen zijn −3, 0, 1 en −4; in het kwadraat 9, 0, 1 en 16; hun gemiddelde is 6,5. De check probeerde ook twee andere dagen, en ook daar klopte je code."},
        },
    ),
    "mse": FunctionBox(
        name="mse",
        function="mse",
        start=_MSE_START,
        solution=_MSE,
        visible=[([3, 5, 2, 6], [4, 5, 0, 6])],
        hidden=[([300, 250, 400], [280, 260, 400]), ([1, 2, 3], [1, 2, 3]), ([10], [7]), ([0, 0, 0, 0], [1, -1, 1, -1])],
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("no-square", wrong=_mse("    residuals = y - y_pred\n    return residuals.mean()"), en="`{call}` gives `{got}`, and it should give `{want}`: your function takes the plain mean of the residuals, where plus and minus cancel out. Square each residual first: `** 2`.", nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je functie neemt het gewone gemiddelde van de residuen, waarin plus en min tegen elkaar wegvallen. Kwadrateer eerst elk residu: `** 2`."),
            Mistake("no-square-flipped", wrong=_mse("    residuals = y_pred - y\n    return residuals.mean()"), en="`{call}` gives `{got}`, and it should give `{want}`: your function takes the plain mean of the residuals, where plus and minus cancel out. Square each residual first: `** 2`.", nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je functie neemt het gewone gemiddelde van de residuen, waarin plus en min tegen elkaar wegvallen. Kwadrateer eerst elk residu: `** 2`."),
            Mistake("no-mean", wrong=_mse("    residuals = y - y_pred\n    return residuals ** 2"),
                    en="`{call}` gives `{got}`: one square per example. The loss is one number: take their mean, step 3 of the recipe.",
                    nl="`{call}` geeft `{got}`: één kwadraat per voorbeeld. De loss is één getal: neem hun gemiddelde, stap 3 van het recept."),
            Mistake("sum", wrong=_mse("    residuals = y - y_pred\n    return (residuals ** 2).sum()"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your function adds up the squares. The mean squared error takes their mean: `.mean()`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je functie telt de kwadraten op. De mean squared error neemt hun gemiddelde: `.mean()`."),
            Mistake("mean-then-square", wrong=_mse("    residuals = y - y_pred\n    return residuals.mean() ** 2"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your function squares the mean residual. Square each residual first, then take the mean.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je functie kwadrateert het gemiddelde residu. Kwadrateer eerst elk residu, en neem dan het gemiddelde."),
            Mistake("root", wrong=_mse("    residuals = y - y_pred\n    return (residuals ** 2).mean() ** 0.5"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your function takes the square root of the mean squared error. That is a measure too, but the loss of this lesson stops at the mean: leave out the root.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je functie neemt de wortel uit de mean squared error. Dat is ook een maat, maar de loss van deze les stopt bij het gemiddelde: laat de wortel weg."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the three steps of the recipe on the arrays, and give the mean squared error back with `return`.",
                      "nl": "Vervang de `...` door de drie stappen van het recept op de arrays, en geef de mean squared error terug met `return`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with two lists, and the `def` line of your function has a different number of names between its parentheses. Keep `(y, y_pred)`.",
                          "nl": "De check roept `{call}` aan, met twee lijsten, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(y, y_pred)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you take the residuals, square them, and take their mean?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Neem je de residuen, kwadrateer je ze, en neem je hun gemiddelde?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for any number of examples, or does it use a number that only fits the example?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elk aantal voorbeelden, of gebruikt hij een getal dat alleen bij het voorbeeld past?"},
            "correct": {"en": "Your function is right in all {n} cases. Right predictions give a loss of 0, and the three houses of the worked example give 166.67.",
                        "nl": "Je functie klopt in alle {n} gevallen. Goede voorspellingen geven een loss van 0, en de drie huizen uit het uitgewerkte voorbeeld geven 166,67."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_F = "import numpy as np\n\n\ndef mse(y, y_pred):\n    y = np.array(y)\n    y_pred = np.array(y_pred)\n"
CASES = [
    ("mse-temps", _TEMPS_START, "empty", None),
    ("mse-temps", _TEMPS_SOLUTION, "correct", None),
    ("mse-temps", _temps("sum(squares) / len(squares)"), "correct", None),
    ("mse-temps", _TEMPS + "import numpy as np\nmse_temps = ((np.array(temps_y) - np.array(temps_y_pred)) ** 2).mean()\nmse_temps", "correct", None),
    ("mse-temps", _temps("sum(residuals) / len(residuals)"), "wrong", "no-square"),
    ("mse-temps", _temps("sum(squares)"), "wrong", "no-mean"),
    ("mse-temps", _temps("(sum(residuals) / len(residuals)) ** 2"), "wrong", "mean-then-square"),
    ("mse-temps", _temps("sum(squares) / (len(squares) - 1)"), "wrong", "one-less"),
    ("mse-temps", _temps("sum([r * 2 for r in residuals]) / len(residuals)"), "wrong", "doubled"),
    ("mse-temps", _temps("-sum(residuals) / len(residuals)"), "wrong", "no-square-flipped"),
    ("mse-temps", _temps("sum([abs(r) for r in residuals]) / len(residuals)"), "wrong", "no-sign"),
    ("mse-temps", _temps("squares"), "wrong", "squares-only"),
    ("mse-temps", _TEMPS + "mse_temps = 6.5\nmse_temps", "almost", "other-value"),
    ("mse-temps", _temps("sum(squares) / 4"), "almost", "other-value"),
    ("mse-temps", _temps("sum(squares) / len(squares)")[:-len("mse_temps")] + "print(mse_temps)", "wrong", "print-last"),
    ("mse", _MSE_START, "empty", None),
    ("mse", _MSE, "correct", None),
    ("mse", _F + "    return np.mean((y_pred - y) ** 2)", "correct", None),
    ("mse", _F + "    return ((y - y_pred) ** 2).sum() / len(y)", "correct", None),
    ("mse", _mse("    residuals = y - y_pred\n    return residuals.mean()"), "wrong", "no-square"),
    ("mse", _mse("    residuals = y - y_pred\n    return (residuals ** 2).sum()"), "wrong", "sum"),
    ("mse", _mse("    residuals = y - y_pred\n    return residuals.mean() ** 2"), "wrong", "mean-then-square"),
    ("mse", _mse("    return (y.mean() - y_pred.mean()) ** 2"), "wrong", "mean-then-square"),
    ("mse", _mse("    residuals = y - y_pred\n    return (residuals ** 2).mean() ** 0.5"), "wrong", "root"),
    ("mse", _F + "    return np.sqrt(((y - y_pred) ** 2).mean())", "wrong", "root"),
    ("mse", _mse("    residuals = y_pred - y\n    return residuals.mean()"), "wrong", "no-square-flipped"),
    ("mse", _mse("    residuals = y - y_pred\n    return residuals ** 2"), "wrong", "no-mean"),
    ("mse", _F + "    return ((y - y_pred) ** 2).sum() / 4", "almost", None),
    ("mse", _F + "    return 1.25", "almost", None),
]
