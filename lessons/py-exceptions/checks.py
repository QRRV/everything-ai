# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson py-exceptions: catching a ValueError, and raising one yourself."""
from everything_ai.check import BugFix, Mistake

# The lesson is done when these are correct. "raise-caught" is the choice question in the lesson.
REQUIRED = ["safe-int", "raise-caught", "average"]

_SAFE_START = "def safe_int(text, default):\n    ..."
_SAFE = "def safe_int(text, default):\n    try:\n        return int(text)\n    except ValueError:\n        return default"
_AVERAGE_START = "def average(numbers):\n    total = 0\n    for n in numbers:\n        total = total + n\n    return total / len(numbers)"
_AVERAGE = ("def average(numbers):\n    if len(numbers) == 0:\n        raise ValueError(\"the list is empty\")\n"
            "    total = 0\n    for n in numbers:\n        total = total + n\n    return total / len(numbers)")


def _no_value_error(got, fn) -> bool:
    """The function catches errors without naming ValueError, such as a bare `except:`."""
    return "ValueError" not in fn.__code__.co_names


EXERCISES = {
    "safe-int": BugFix(
        name="safe-int",
        function="safe_int",
        start=_SAFE_START,
        solution=_SAFE,
        visible=[("42", 0)],
        hidden=[("hi", 0), ("", -1), ("3.5", 0), ("-7", 0)],
        form=[Mistake("bare-except", matches=_no_value_error, state="almost",
                      en="Right in all cases, but your function has no `except ValueError:`. An `except` without `ValueError` also catches a typo or another bug, and hides it. Use `try` with `except ValueError:`.",
                      nl="Goed in alle gevallen, maar je functie heeft geen `except ValueError:`. Een `except` zonder `ValueError` vangt ook een typefout of een andere bug op, en verstopt die. Gebruik `try` met `except ValueError:`.")],
        mistakes=[
            Mistake("always-default", wrong="def safe_int(text, default):\n    return default",
                    en="`{call}` gives `{got}`: your function always gives back `default`. Give back `int(text)` when the text is a whole number, and `default` only when `int` stops with a `ValueError`.",
                    nl="`{call}` geeft `{got}`: je functie geeft altijd `default` terug. Geef `int(text)` terug als de tekst een geheel getal is, en `default` alleen als `int` stopt met een `ValueError`."),
            Mistake("text-back", wrong="def safe_int(text, default):\n    try:\n        int(text)\n        return text\n    except ValueError:\n        return default",
                    en="`{call}` gives `{got}`: the text, with quotes, not the number. Give back `int(text)`, the number itself.",
                    nl="`{call}` geeft `{got}`: de tekst, met aanhalingstekens, niet het getal. Geef `int(text)` terug, het getal zelf."),
            Mistake("float", state="almost", wrong="def safe_int(text, default):\n    try:\n        return float(text)\n    except ValueError:\n        return default",
                    en="`{call}` gives `{got}`: `float` also reads numbers with a decimal point, and `safe_int` is about whole numbers. Use `int(text)`.",
                    nl="`{call}` geeft `{got}`: `float` leest ook getallen met een punt, en `safe_int` gaat over gehele getallen. Gebruik `int(text)`."),
            Mistake("isdigit", state="almost", wrong="def safe_int(text, default):\n    if text.isdigit():\n        return int(text)\n    return default",
                    en="`{call}` gives `{got}`: `\"-7\"` is a whole number too, but `isdigit` says no to its minus sign. Let `int` decide, with `try` and `except ValueError:`.",
                    nl="`{call}` geeft `{got}`: `\"-7\"` is ook een geheel getal, maar `isdigit` zegt nee tegen het minteken. Laat `int` beslissen, met `try` en `except ValueError:`."),
        ],
        texts={
            "missing": {"en": "I cannot find `safe_int`. Run the cell with the function again; if it shows an error, fix that first.",
                        "nl": "Ik vind `safe_int` niet. Voer de cel met de functie opnieuw uit; laat die een fout zien, los die dan eerst op."},
            "empty_error": {"en": "Fill in `safe_int` in the cell above: try `int(text)`, and give back `default` when that stops with a `ValueError`. Then run the cell.",
                            "nl": "Vul `safe_int` in, in de cel hierboven: probeer `int(text)`, en geef `default` terug als dat stopt met een `ValueError`. Voer daarna de cel uit."},
            "empty_wrong": {"en": "Fill in `safe_int` in the cell above: try `int(text)`, and give back `default` when that stops with a `ValueError`. Then run the cell.",
                            "nl": "Vul `safe_int` in, in de cel hierboven: probeer `int(text)`, en geef `default` terug als dat stopt met een `ValueError`. Voer daarna de cel uit."},
            "error": {"en": "With `{call}`, your function stops with an error:\n\n```text\n{traceback}\n```\n\nRead the last line first, then the line of your cell just above it. If it says `ValueError`: is the line with `int` under `try:`, and does your `except` name `ValueError`?",
                      "nl": "Met `{call}` stopt je functie met een fout:\n\n```text\n{traceback}\n```\n\nLees eerst de laatste regel, en dan de regel uit je cel er net boven. Staat er `ValueError`: staat de regel met `int` onder `try:`, en noemt je `except` `ValueError`?"},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: `\"42\"` is a whole number. Give back `int(text)` under `try:`.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: `\"42\"` is een geheel getal. Geef `int(text)` terug onder `try:`."},
            "almost": {"en": "Right for `\"42\"`. But `{call}` gives `{got}`, and it should give `{want}`. Does your function give back `int(text)` whenever `int` can read the text, and `default` only when it stops with a `ValueError`?",
                       "nl": "Goed voor `\"42\"`. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Geeft je functie `int(text)` terug als `int` de tekst kan lezen, en `default` alleen als die stopt met een `ValueError`?"},
            "correct": {"en": "Your function is right in all {n} cases: a whole number comes back as a number, and anything else, also `\"3.5\"` and the empty text, gives `default`.",
                        "nl": "Je functie klopt in alle {n} gevallen: een geheel getal komt terug als getal, en al het andere, ook `\"3.5\"` en de lege tekst, geeft `default`."},
        },
    ),
    "average": BugFix(
        name="average",
        function="average",
        start=_AVERAGE_START,
        solution=_AVERAGE,
        visible=[([],)],
        hidden=[([4, 8, 6],), ([5],), ([-2, 2],), ([1, 2],)],
        texts={
            "missing": {"en": "I cannot find `average`. Run the cell with the function again; if it shows an error, fix that first.",
                        "nl": "Ik vind `average` niet. Voer de cel met de functie opnieuw uit; laat die een fout zien, los die dan eerst op."},
            "empty_error": {"en": "For an empty list, this function stops with Python's own error:\n\n```text\n{traceback}\n```\n\nThat says what Python noticed, not what is wrong with the input. Make `average` raise a `ValueError` with a message of your own when the list is empty, before the division. Then run the cell.",
                            "nl": "Bij een lege lijst stopt deze functie met de eigen fout van Python:\n\n```text\n{traceback}\n```\n\nDat zegt wat Python merkte, niet wat er mis is met de invoer. Laat `average` een `ValueError` geven met een eigen melding als de lijst leeg is, vóór de deling. Voer daarna de cel uit."},
            "empty_wrong": {"en": "Make `average` raise a `ValueError` with a message of your own when the list is empty, before the division. Then run the cell.",
                            "nl": "Laat `average` een `ValueError` geven met een eigen melding als de lijst leeg is, vóór de deling. Voer daarna de cel uit."},
            "error": {"en": "With `{call}`, your function stops with this error:\n\n```text\n{traceback}\n```\n\nFor an empty list, the check expects a `ValueError` that your function raises itself, before the division; for other lists, the average.",
                      "nl": "Met `{call}` stopt je functie met deze fout:\n\n```text\n{traceback}\n```\n\nBij een lege lijst verwacht de check een `ValueError` die je functie zelf geeft, vóór de deling; bij andere lijsten het gemiddelde."},
            "not_raised": {"en": "`{call}` gives `{got}`, and the check expects your function to stop with a `{kind}`. An empty list has no average: whatever you give back looks like an answer, and the calling code goes on with it. Stop with `raise {kind}(...)`, with a message of your own.",
                           "nl": "`{call}` geeft `{got}`, en de check verwacht dat je functie stopt met een `{kind}`. Een lege lijst heeft geen gemiddelde: wat je ook teruggeeft, het lijkt een antwoord, en de code die je functie aanriep, gaat ermee verder. Stop met `raise {kind}(...)`, met een eigen melding."},
            "no_message": {"en": "Your function raises a `{kind}` for `{call}`, but without a real message. Give it one in parentheses, such as `raise {kind}(\"the list is empty\")`.",
                           "nl": "Je functie geeft een `{kind}` bij `{call}`, maar zonder echte melding. Geef hem er een tussen haakjes, zoals `raise {kind}(\"de lijst is leeg\")`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Did you change the division? Only add the `if` for an empty list, at the top.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Heb je de deling veranderd? Zet alleen de `if` voor een lege lijst erbij, bovenaan."},
            "almost": {"en": "Right for the empty list. But `{call}` gives `{got}`, and it should give `{want}`: the average. Did you change the rest of the function? Only add the `if`, at the top.",
                       "nl": "Goed voor de lege lijst. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn: het gemiddelde. Heb je de rest van de functie veranderd? Zet alleen de `if` erbij, bovenaan."},
            "correct": {"en": "Your function is right in all {n} cases: for an empty list, it stops with your own `ValueError`, and other lists give their average.",
                        "nl": "Je functie klopt in alle {n} gevallen: bij een lege lijst stopt hij met je eigen `ValueError`, en andere lijsten geven hun gemiddelde."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, the function as the learner left it, state, known mistake).
CASES = [
    ("safe-int", _SAFE_START, "empty", None),
    ("safe-int", _SAFE, "correct", None),
    ("safe-int", "def safe_int(text, default):\n    try:\n        number = int(text)\n    except ValueError:\n        number = default\n    return number", "correct", None),
    ("safe-int", "def safe_int(text, default):\n    try:\n        return int(text)\n    except:\n        return default", "almost", "bare-except"),
    ("safe-int", "def safe_int(text, default):\n    return int(text)", "error", None),
    ("safe-int", "def safe_int(text, default):\n    try:\n        return int(text)\n    except TypeError:\n        return default", "error", None),
    ("safe-int", "def safe_int(text, default):\n    return default", "wrong", "always-default"),
    ("safe-int", "def safe_int(text, default):\n    try:\n        return float(text)\n    except ValueError:\n        return default", "almost", "float"),
    ("safe-int", "def safe_int(text, default):\n    if text.isdigit():\n        return int(text)\n    return default", "almost", "isdigit"),
    ("safe-int", "def safe_int(text, default):\n    try:\n        return int(text)\n    except Exception:\n        return default", "almost", "bare-except"),
    ("safe-int", "def safe_int(text, default):\n    try:\n        int(text)\n        return text\n    except ValueError:\n        return default", "wrong", "text-back"),
    ("average", _AVERAGE_START, "empty", None),
    ("average", _AVERAGE, "correct", None),
    ("average", _AVERAGE.replace('raise ValueError("the list is empty")', "return 0"), "wrong", "not-raised"),
    ("average", _AVERAGE.replace('raise ValueError("the list is empty")', "raise ValueError"), "almost", "no-message"),
    ("average", _AVERAGE.replace('raise ValueError("the list is empty")', 'raise ZeroDivisionError("empty")'), "error", None),
    ("average", _AVERAGE.replace("total / len(numbers)", "total / 2"), "almost", None),
    ("average", _AVERAGE.replace("raise ValueError", "return ValueError"), "wrong", "returned-error"),
    ("average", _AVERAGE.replace('"the list is empty"', '"..."'), "almost", "no-message"),
    ("average", _AVERAGE.replace("total / len(numbers)", "total // len(numbers)"), "almost", None),
    ("average", "def average(numbers):\n    if numbers == []:\n        raise ValueError(\"no numbers\")\n    return sum(numbers) / len(numbers)", "correct", None),
]
