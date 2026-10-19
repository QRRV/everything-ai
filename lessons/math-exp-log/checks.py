# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson math-exp-log: how often a sheet of paper must be folded to reach a kilometer, and the log of a
product as a sum of logs. And the picture of exp and log as mirror images."""
import math
import re

from everything_ai.check import CodeBox, FunctionBox, Mistake


def picture(x: float, labels: dict[str, str]) -> str:
    """exp and log on the same axes, with the line y = x between them, a point (x, eˣ) on exp and its mirror image
    (eˣ, x) on log. Drawn as SVG text, for every stop of the slider once (L.explore)."""
    from everything_ai import chart
    p = chart.Panel((-3, 8), (-3, 8), title=labels["title"], xlabel="x", ylabel="y")
    p.line([-3, 8], [-3, 8], "ink-3", 1.2, dashed=True)
    xs = [-3 + 5.08 * i / 120 for i in range(121)]
    p.line(xs, [math.exp(v) for v in xs], "data")
    p.line([math.exp(v) for v in xs], xs, "data")
    p.text(1.25, 6.6, "exp", "ink-2")
    p.text(6.6, 1.3, "log", "ink-2")
    y = math.exp(x)
    p.line([x, y], [y, x], "ink-3", 1.2)
    p.points([x, y], [y, x], "data", 5.5)
    return chart.figure([p], alt=labels["alt"], width=420, height=400)


# The lesson is done when these are correct. "negative" is the choice question in the lesson.
REQUIRED = ["negative", "folds", "log-product"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_F_HEAD = "import numpy as np\n\nthickness = 0.1\nheight = 1000000\n"
_F_START = _F_HEAD + "folds = ...\nfolds\n"
_F = _F_HEAD + "folds = np.log2(height / thickness)\nfolds"


def _f(new: str) -> str:
    return _F.replace("folds = np.log2(height / thickness)", new)


_FOLDS = math.log2(1000000 / 0.1)   # about 23.25: the exact number of folds for one kilometer


def _rounded(got, text) -> bool:
    """A whole or rounded number of folds, such as 24 from a loop that counts doublings, or 23.3 from round()."""
    if isinstance(got, (str, bytes, bool)):
        return False
    try:
        return abs(float(got) - _FOLDS) < 1
    except (TypeError, ValueError):
        return False


_P_START = "import numpy as np\n\n\ndef log_product(ps):\n    total = 0\n    ...\n    return total\n"
_P = "import numpy as np\n\n\ndef log_product(ps):\n    total = 0\n    for p in ps:\n        total = total + np.log(p)\n    return total"


def _p(body: str, end: str = "    return total") -> str:
    return "import numpy as np\n\n\ndef log_product(ps):\n" + body + end


EXERCISES = {
    "folds": CodeBox(
        name="folds",
        start=_F_START,
        solution=_F,
        keep=["thickness = 0.1"],
        # The check also runs the box with the distance to the moon: a number typed in by hand does not change with it.
        others=[{"height": 384400000000}],
        first=[_DOTS],
        mistakes=[
            Mistake("ratio-only", wrong=_f("folds = height / thickness"),
                    en="Your last line gives `{got}`: how many times thicker the paper must get. How many doublings make that factor? That is a logarithm with base 2.",
                    nl="Je laatste regel geeft `{got}`: hoeveel keer dikker het papier moet worden. Hoeveel keer verdubbelen geeft die factor? Dat is een logaritme met grondtal 2."),
            Mistake("no-ratio", wrong=_f("folds = np.log2(height)"),
                    en="Your last line gives `{got}`: the doublings from 1 millimeter to a kilometer. But the paper starts at `thickness`, 0.1 millimeter: by what factor must that grow?",
                    nl="Je laatste regel geeft `{got}`: het aantal keer verdubbelen van 1 millimeter tot een kilometer. Maar het papier begint bij `thickness`, 0,1 millimeter: met welke factor moet dat groeien?"),
            Mistake("swapped", wrong=_f("folds = np.log2(thickness / height)"),
                    en="Your last line gives `{got}`: negative, because the factor is the wrong way round. The paper has to grow from `thickness` to `height`: `height / thickness`.",
                    nl="Je laatste regel geeft `{got}`: negatief, omdat de factor andersom staat. Het papier moet groeien van `thickness` naar `height`: `height / thickness`."),
            Mistake("base-e", wrong=_f("folds = np.log(height / thickness)"),
                    en="Your last line gives `{got}`: the log with base e, how many times you multiply by e. Each fold multiplies by 2: which log has base 2?",
                    nl="Je laatste regel geeft `{got}`: de log met grondtal e, hoe vaak je met e vermenigvuldigt. Elke vouw vermenigvuldigt met 2: welke log heeft grondtal 2?"),
            Mistake("base-10", wrong=_f("folds = np.log10(height / thickness)"),
                    en="Your last line gives `{got}`: the log with base 10, how many times you multiply by 10. Each fold multiplies by 2: which log has base 2?",
                    nl="Je laatste regel geeft `{got}`: de log met grondtal 10, hoe vaak je met 10 vermenigvuldigt. Elke vouw vermenigvuldigt met 2: welke log heeft grondtal 2?"),
            Mistake("rounded", matches=_rounded,
                    en="Your last line gives `{got}`: a whole or rounded number of folds. Real paper does need whole folds, but the check asks for the exponent of 2 exactly as a logarithm gives it, with all its decimals.",
                    nl="Je laatste regel geeft `{got}`: een heel of afgerond aantal keer vouwen. Echt papier vouw je wel een heel aantal keer, maar de check vraagt de exponent van 2 precies zoals een logaritme hem geeft, met alle cijfers achter de komma."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the number of folds: the exponent of 2 that turns `thickness` into `height`.",
                      "nl": "Vervang de `...` door het aantal keer vouwen: de exponent van 2 die `thickness` laat groeien tot `height`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another height by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een andere hoogte."},
            "no_value": {"en": "The last line of the box shows nothing. Put `folds` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `folds` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. By what factor must the paper grow, and how many doublings give that factor?",
                      "nl": "Je laatste regel geeft `{got}`. Met welke factor moet het papier groeien, en hoeveel keer verdubbelen geeft die factor?"},
            "other": {"en": "Right for one kilometer. But with {where}, the distance to the moon in millimeters, your last line gives `{got}`, and it should give `{want}`. Do you work it out from `thickness` and `height` themselves?",
                      "nl": "Goed voor één kilometer. Maar met {where}, de afstand tot de maan in millimeter, geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je het uit met `thickness` en `height` zelf?"},
            "other_error": {"en": "Right for one kilometer. But with {where}, the distance to the moon in millimeters, your code stops with an error. Do you work it out from `thickness` and `height` themselves?",
                            "nl": "Goed voor één kilometer. Maar met {where}, de afstand tot de maan in millimeter, stopt je code met een fout. Reken je het uit met `thickness` en `height` zelf?"},
            "correct": {"en": "About 23.3 folds: after 23 folds the paper is about 840 meters thick, after 24 about 1.7 kilometers. The check also tried the distance to the moon: 41.8, so 42 folds would do.",
                        "nl": "Ongeveer 23,3 keer: na 23 keer vouwen is het papier ongeveer 840 meter dik, na 24 keer ongeveer 1,7 kilometer. De check probeerde ook de afstand tot de maan: 41,8, dus 42 keer vouwen is genoeg."},
        },
    ),
    "log-product": FunctionBox(
        name="log-product",
        function="log_product",
        start=_P_START,
        solution=_P,
        visible=[([0.5, 0.2, 0.1],)],
        hidden=[([1e-200, 1e-200],), ([0.9, 0.9, 0.9],), ([1.0],), ([],)],
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("product-first", wrong=_p("    product = 1\n    for p in ps:\n        product = product * p\n", "    return np.log(product)"),
                    en="`{call}` gives `{got}`: minus infinity. The product of these two numbers, 1e-400, is too small for the computer: it rounds it to 0, and the log of 0 is minus infinity. Add the logs instead, one number at a time.",
                    nl="`{call}` geeft `{got}`: min oneindig. Het product van deze twee getallen, 1e-400, is te klein voor de computer: die rondt het af naar 0, en de log van 0 is min oneindig. Tel in plaats daarvan de logs op, één getal tegelijk."),
            Mistake("no-log", wrong=_p("    total = 0\n    for p in ps:\n        total = total + p\n"),
                    en="`{call}` gives `{got}`: the numbers themselves added up. Add the log of each number: `np.log(p)`.",
                    nl="`{call}` geeft `{got}`: de getallen zelf opgeteld. Tel de log van elk getal op: `np.log(p)`."),
            Mistake("log-of-sum", wrong=_p("    total = 0\n    for p in ps:\n        total = total + p\n", "    return np.log(total)"),
                    en="`{call}` gives `{got}`: the log of the sum. The log of a product is the sum of the logs: take the log of each number, then add.",
                    nl="`{call}` geeft `{got}`: de log van de som. De log van een product is de som van de logs: neem de log van elk getal, en tel dan op."),
            Mistake("logs-multiplied", wrong=_p("    total = 1\n    for p in ps:\n        total = total * np.log(p)\n"),
                    en="`{call}` gives `{got}`: the logs multiplied. The log turns multiplying into adding: add the logs.",
                    nl="`{call}` geeft `{got}`: de logs vermenigvuldigd. De log maakt van vermenigvuldigen optellen: tel de logs op."),
            Mistake("base-2", wrong=_p("    total = 0\n    for p in ps:\n        total = total + np.log2(p)\n"),
                    en="`{call}` gives `{got}`: the sum of the logs with base 2. Without a base, log in this course means base e: which function is that? See step 2 of the recipe.",
                    nl="`{call}` geeft `{got}`: de som van de logs met grondtal 2. Zonder grondtal betekent log in deze course grondtal e: welke functie is dat? Zie stap 2 van het recept."),
            Mistake("base-10", wrong=_p("    total = 0\n    for p in ps:\n        total = total + np.log10(p)\n"),
                    en="`{call}` gives `{got}`: the sum of the logs with base 10. Without a base, log in this course means base e: which function is that? See step 2 of the recipe.",
                    nl="`{call}` geeft `{got}`: de som van de logs met grondtal 10. Zonder grondtal betekent log in deze course grondtal e: welke functie is dat? Zie stap 2 van het recept."),
            Mistake("return-in-loop", wrong=_p("    total = 0\n    for p in ps:\n        total = total + np.log(p)\n        return total\n", ""),
                    en="`{call}` gives `{got}`: only the log of the first number. Is your `return` inside the loop? Move it to the left, under `for`, so that the loop goes through every number first.",
                    nl="`{call}` geeft `{got}`: alleen de log van het eerste getal. Staat je `return` in de loop? Zet hem naar links, onder `for`, zodat de loop eerst elk getal langsgaat."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a loop that adds the log of each number to `total`.",
                      "nl": "Vervang de `...` door een loop die de log van elk getal bij `total` optelt."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with one list of numbers, and the `def` line of your function has a different number of names between its parentheses. Keep `(ps)`.",
                          "nl": "De check roept `{call}` aan, met één lijst getallen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(ps)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you add up the log of each number?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Tel je de log van elk getal op?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for any list of numbers, also for very small numbers, one number, or none?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor elke lijst getallen, ook voor heel kleine getallen, één getal, of geen?"},
            "correct": {"en": "Your function is right in all {n} cases. For two numbers of 1e-200, the product is too small for the computer, but the sum of their logs is an ordinary number, about −921. That is why a model works with the logs of probabilities.",
                        "nl": "Je functie klopt in alle {n} gevallen. Bij twee getallen van 1e-200 is het product te klein voor de computer, maar de som van hun logs is een gewoon getal: ongeveer −921. Daarom rekent een model met de logs van kansen."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("folds", _F_START, "empty", None),
    ("folds", _F, "correct", None),
    ("folds", _f("folds = np.log(height / thickness) / np.log(2)"), "correct", None),
    ("folds", _f("folds = np.log2(height) - np.log2(thickness)"), "correct", None),
    ("folds", _f("folds = height / thickness"), "wrong", "ratio-only"),
    ("folds", _f("folds = np.log2(height)"), "wrong", "no-ratio"),
    ("folds", _f("folds = np.log2(thickness / height)"), "wrong", "swapped"),
    ("folds", _f("folds = np.log(height / thickness)"), "wrong", "base-e"),
    ("folds", _f("folds = np.log10(height / thickness)"), "wrong", "base-10"),
    ("folds", _f("folds = np.log2(1000000 / 0.1)"), "almost", "other-value"),
    ("folds", _f("folds = 0\nt = thickness\nfor i in range(100):\n    if t < height:\n        t = t * 2\n        folds = folds + 1"), "wrong", "rounded"),
    ("folds", _f("folds = round(np.log2(height / thickness), 1)"), "wrong", "rounded"),
    ("log-product", _P_START, "empty", None),
    ("log-product", _P, "correct", None),
    ("log-product", _p("    total = 0\n    for i in range(len(ps)):\n        total = total + np.log(ps[i])\n"), "correct", None),
    ("log-product", _p("    product = 1\n    for p in ps:\n        product = product * p\n", "    return np.log(product)"), "wrong", "product-first"),
    ("log-product", _p("    total = 0\n    for p in ps:\n        total = total + p\n"), "wrong", "no-log"),
    ("log-product", _p("    total = 0\n    for p in ps:\n        total = total + p\n", "    return np.log(total)"), "wrong", "log-of-sum"),
    ("log-product", _p("    total = 1\n    for p in ps:\n        total = total * np.log(p)\n"), "wrong", "logs-multiplied"),
    ("log-product", _p("    total = 0\n    for p in ps:\n        total += np.log(p)\n"), "correct", None),
    ("log-product", _p("    total = 0\n", "    return np.sum(np.log(ps))"), "correct", None),
    ("log-product", _p("    total = 0\n    for p in ps:\n        total = total + np.log2(p)\n"), "wrong", "base-2"),
    ("log-product", _p("    total = 0\n    for p in ps:\n        total = total + np.log10(p)\n"), "wrong", "base-10"),
    ("log-product", _p("    total = 0\n    for p in ps:\n        total = total + np.log(p)\n        return total\n", ""), "wrong", "return-in-loop"),
]
