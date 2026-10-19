# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson py-errors: two functions with a bug, fixed in their own cells. And the cell and traceback in the picture."""
from everything_ai.check import BugFix, Mistake

# The picture: a cell whose function stops for an empty list, and the traceback under it, read from the bottom up.
PICTURE_CELL = [("def average(numbers):", ""), ("    total = 0", ""), ("    for n in numbers:", ""),
                ("        total = total + n", ""), ("    return total / len(numbers)", ""), ("", ""), ("print(average([]))", "")]
PICTURE_TRACEBACK = [("Traceback (most recent call last):", ""), ("  Cell, line 7, in <module>", ""), ("    print(average([]))", "3"),
                     ("  Cell, line 5, in average", ""), ("    return total / len(numbers)", "2"), ("ZeroDivisionError: division by zero", "1")]

# The lesson is done when these are correct. "traceback" is the choice question in the lesson.
REQUIRED = ["total", "traceback", "passed"]

_TOTAL_BUG = "def total_price(prices):\n    total = 0\n    for price in prices:\n        total = total + prices\n    return total"
_PASSED_BUG = "def count_passed(scores):\n    count = 0\n    for score in scores:\n        if score >= 5.5:\n            count = count + 1\n        return count"

EXERCISES = {
    "total": BugFix(
        name="total",
        function="total_price",
        start=_TOTAL_BUG,
        solution=_TOTAL_BUG.replace("total + prices", "total + price"),
        visible=[([2.5, 4.0, 1.25],)],
        hidden=[([],), ([3],), ([1, 2, 3, 4],), ([0.5, 0.25],)],
        mistakes=[
            Mistake("bug-kept", wrong=_TOTAL_BUG,
                    en="With `{call}`, your function still stops on the same line as at the start:\n\n```text\n{traceback}\n```\n\nRead the last line again: which two things does the line above it add?",
                    nl="Met `{call}` stopt je functie nog steeds op dezelfde regel als aan het begin:\n\n```text\n{traceback}\n```\n\nLees de laatste regel nog eens: welke twee dingen telt de regel erboven op?"),
            Mistake("last-only", wrong=_TOTAL_BUG.replace("total = total + prices", "total = price"),
                    en="`{call}` gives `{got}`: only the last price. Each round, `total` gets the price of that round, and what it had is lost. How can each round add to what `total` already is?",
                    nl="`{call}` geeft `{got}`: alleen de laatste prijs. Elke ronde krijgt `total` de prijs van die ronde, en wat hij al had, raakt hij kwijt. Hoe telt elke ronde op bij wat `total` al is?"),
            Mistake("count", wrong=_TOTAL_BUG.replace("total + prices", "total + 1"),
                    en="`{call}` gives `{got}`: the number of prices, not their total. Add the price of each round, not 1.",
                    nl="`{call}` geeft `{got}`: het aantal prijzen, niet hun totaal. Tel de prijs van elke ronde op, niet 1."),
        ],
        texts={
            "missing": {"en": "I cannot find `total_price`. Run the cell with the function again; if it shows an error, fix that first.",
                        "nl": "Ik vind `total_price` niet. Voer de cel met de functie opnieuw uit; laat die een fout zien, los die dan eerst op."},
            "empty_error": {"en": "This function has a bug. The check calls `{call}`, and Python stops with this error:\n\n```text\n{traceback}\n```\n\nRead it from the bottom up, find the bug in the cell above, fix it, and run the cell.",
                            "nl": "Deze functie heeft een bug. De check roept `{call}` aan, en Python stopt met deze fout:\n\n```text\n{traceback}\n```\n\nLees hem van onder naar boven, zoek de bug in de cel hierboven, los hem op, en voer de cel uit."},
            "empty_wrong": {"en": "This function has a bug: `{call}` gives `{got}`, and it should give `{want}`. Find it in the cell above, fix it, and run the cell.",
                            "nl": "Deze functie heeft een bug: `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Zoek hem in de cel hierboven, los hem op, en voer de cel uit."},
            "error": {"en": "With `{call}`, your function stops with an error:\n\n```text\n{traceback}\n```\n\nRead the last line first, then the line of your cell just above it.",
                      "nl": "Met `{call}` stopt je functie met een fout:\n\n```text\n{traceback}\n```\n\nLees eerst de laatste regel, en dan de regel uit je cel er net boven."},
            "wrong": {"en": "No error any more, but `{call}` gives `{got}`, and it should give `{want}`: 2.5 + 4.0 + 1.25. Test it on a small list, such as `[1, 2]`: which line gives the wrong answer?",
                      "nl": "Geen fout meer, maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn: 2,5 + 4,0 + 1,25. Test met een kleine lijst, zoals `[1, 2]`: welke regel geeft het verkeerde antwoord?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function add up every price, for every list?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Telt je functie elke prijs op, voor elke lijst?"},
            "correct": {"en": "Bug fixed: your function is right in all {n} cases, also for an empty list.",
                        "nl": "Bug opgelost: je functie klopt in alle {n} gevallen, ook bij een lege lijst."},
        },
    ),
    "passed": BugFix(
        name="passed",
        function="count_passed",
        start=_PASSED_BUG,
        solution=_PASSED_BUG.replace("\n        return count", "\n    return count"),
        visible=[([7.0, 4.5, 8.0],)],
        hidden=[([],), ([4.5, 8.0],), ([5.5],), ([5.8],), ([1.0, 2.0],)],
        mistakes=[
            Mistake("bug-kept", wrong=_PASSED_BUG,
                    en="`{call}` still gives `{got}`, as at the start, and it should give `{want}`: 7.0 and 8.0 both pass. Test it on `[7.0]` and on `[4.5, 8.0]`: which answer is wrong, and which line makes the function stop too early?",
                    nl="`{call}` geeft nog steeds `{got}`, net als aan het begin, en het zou `{want}` moeten zijn: 7,0 en 8,0 zijn allebei voldoende. Test met `[7.0]` en met `[4.5, 8.0]`: welk antwoord klopt niet, en welke regel laat de functie te vroeg stoppen?"),
            Mistake("return-in-if", wrong=_PASSED_BUG.replace("\n        return count", "\n            return count"),
                    en="`{call}` gives `{got}`: your function stops at the first score that passes. `return` ends the function at once. Where should it be, so it runs only after every score?",
                    nl="`{call}` geeft `{got}`: je functie stopt bij de eerste voldoende. `return` beëindigt de functie meteen. Waar hoort hij te staan, zodat hij pas na elke score draait?"),
            Mistake("greater", state="almost", wrong=_PASSED_BUG.replace("\n        return count", "\n    return count").replace(">= 5.5", "> 5.5"),
                    en="`{call}` gives `{got}`: a score of exactly 5.5 does not count now, and it is a pass. Keep `>=`.",
                    nl="`{call}` geeft `{got}`: een score van precies 5,5 telt nu niet mee, en dat is een voldoende. Laat `>=` staan."),
        ],
        texts={
            "missing": {"en": "I cannot find `count_passed`. Run the cell with the function again; if it shows an error, fix that first.",
                        "nl": "Ik vind `count_passed` niet. Voer de cel met de functie opnieuw uit; laat die een fout zien, los die dan eerst op."},
            "empty_error": {"en": "This function has a bug. The check calls `{call}`, and Python stops with this error:\n\n```text\n{traceback}\n```\n\nFind the bug in the cell above, fix it, and run the cell.",
                            "nl": "Deze functie heeft een bug. De check roept `{call}` aan, en Python stopt met deze fout:\n\n```text\n{traceback}\n```\n\nZoek de bug in de cel hierboven, los hem op, en voer de cel uit."},
            "empty_wrong": {"en": "This function has a bug, and it gives no error: `{call}` gives `{got}`, and it should give `{want}`. Test it on small lists whose answer you know, find the bug, fix it, and run the cell.",
                            "nl": "Deze functie heeft een bug, en hij geeft geen fout: `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Test hem met kleine lijsten waarvan je het antwoord weet, zoek de bug, los hem op, en voer de cel uit."},
            "error": {"en": "With `{call}`, your function stops with an error:\n\n```text\n{traceback}\n```\n\nRead the last line first, then the line of your cell just above it.",
                      "nl": "Met `{call}` stopt je functie met een fout:\n\n```text\n{traceback}\n```\n\nLees eerst de laatste regel, en dan de regel uit je cel er net boven."},
            "no_return": {"en": "With `{call}`, your function gives nothing back: Python never gets to `return count`. Where should it be, so it always runs, also when the list is empty?",
                          "nl": "Met `{call}` geeft je functie niets terug: Python komt nooit bij `return count`. Waar hoort die te staan, zodat hij altijd draait, ook bij een lege lijst?"},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: 7.0 and 8.0 both pass. Test it on `[7.0]` and on `[4.5, 8.0]`: which answer is wrong, and which line causes it?",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: 7,0 en 8,0 zijn allebei voldoende. Test met `[7.0]` en met `[4.5, 8.0]`: welk antwoord klopt niet, en welke regel veroorzaakt het?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function look at every score, and count exactly 5.5 as a pass?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Kijkt je functie naar elke score, en telt precies 5,5 als voldoende?"},
            "correct": {"en": "Bug fixed: your function is right in all {n} cases, also for an empty list and for exactly 5.5.",
                        "nl": "Bug opgelost: je functie klopt in alle {n} gevallen, ook bij een lege lijst en bij precies 5,5."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, the function as the learner left it, state, known mistake).
_FIXED_TOTAL = _TOTAL_BUG.replace("total + prices", "total + price")
_FIXED_PASSED = _PASSED_BUG.replace("\n        return count", "\n    return count")
CASES = [
    ("total", _TOTAL_BUG, "empty", None),
    ("total", "# my notes\n" + _TOTAL_BUG + "\n", "empty", None),
    ("total", _FIXED_TOTAL, "correct", None),
    ("total", "def total_price(prices):\n    total = 0\n    for p in prices:\n        total += p\n    return total", "correct", None),
    ("total", _TOTAL_BUG.replace("total = total + prices", "total = price"), "wrong", "last-only"),
    ("total", _TOTAL_BUG.replace("total + prices", "total + 1"), "wrong", "count"),
    ("total", _TOTAL_BUG.replace("total + prices", "total + prices[0]"), "wrong", None),
    ("total", "def total_price(prices):\n    sum = 0\n    for price in prices:\n        sum = sum + prices\n    return sum", "error", "bug-kept"),
    ("total", _FIXED_TOTAL.replace("    return total", "    return round(total)"), "wrong", None),
    ("total", _TOTAL_BUG.replace("total + prices", "total + prise"), "error", None),
    ("passed", _PASSED_BUG, "empty", None),
    ("passed", _FIXED_PASSED, "correct", None),
    ("passed", _PASSED_BUG.replace("\n        return count", "\n            return count"), "wrong", "return-in-if"),
    ("passed", _FIXED_PASSED.replace(">= 5.5", "> 5.5"), "almost", "greater"),
    ("passed", _PASSED_BUG.replace(">= 5.5", "> 5.5"), "wrong", None),
    ("passed", "def count_passed(scores):\n    n = 0\n    for score in scores:\n        if score >= 5.5:\n            n = n + 1\n        return n", "wrong", "bug-kept"),
    ("passed", _FIXED_PASSED.replace("    return count", "    print(count)"), "wrong", "prints"),
    ("passed", _FIXED_PASSED.replace(">= 5.5", ">= 6"), "almost", None),
    ("passed", _FIXED_PASSED.replace("count = count + 1", "count += 1"), "correct", None),
    ("passed", _FIXED_PASSED.replace("score >= 5.5", "scores >= 5.5"), "error", None),
]
