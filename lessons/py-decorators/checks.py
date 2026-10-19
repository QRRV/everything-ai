# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson py-decorators: two decorators written in a box. The check runs the box, then
decorates small functions of its own with them, and compares what those give back with the solution."""
import re

from everything_ai.check import Mistake, ScenarioBox

# A `...` that is still there, found before the code runs: a function that is still `...` gives back nothing,
# and that should not be explained as some other mistake.
_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own lines.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen regels.")

_COLON = Mistake("colon-missing", syntax=r"^[ \t]*(def\b[^:\n]*|if\b[^:\n]*)$",
                 en="The line `{line}` must end with a colon, `:`. The colon tells Python that the indented lines below belong to it.",
                 nl="De regel `{line}` moet eindigen op een dubbele punt, `:`. Die vertelt Python dat de ingesprongen regels eronder erbij horen.")
_CALLED = Mistake("returns-call", syntax=r"^\s+return\s+wrapper\s*\(",
                  en="`return wrapper()` or `return wrapper(x)` calls the new function right away, and gives back its result. A decorator gives back the function itself: `return wrapper`, without parentheses.",
                  nl="`return wrapper()` of `return wrapper(x)` roept de nieuwe functie meteen aan, en geeft zijn uitkomst terug. Een decorator geeft de functie zelf terug: `return wrapper`, zonder haakjes.")
_NO_X = Mistake("wrapper-no-x", syntax=r"^\s+def\s+wrapper\s*\(\s*\)",
                en="`def wrapper():` has no name for the value that comes in. After the `@` line, the name of the function stands for `wrapper`, so a call such as `square(3)` gives 3 to `wrapper`. Write `def wrapper(x):`.",
                nl="`def wrapper():` heeft geen naam voor de waarde die binnenkomt. Na de `@`-regel staat de naam van de functie voor `wrapper`, dus geeft een aanroep zoals `square(3)` de 3 aan `wrapper`. Schrijf `def wrapper(x):`.")
_FN_EMPTY = Mistake("fn-no-x", syntax=r"\bfn\s*\(\s*\)",
                    en="`fn()` calls the wrapped function without a value, but it needs one. Pass `x` on: `fn(x)`.",
                    nl="`fn()` roept de ingepakte functie aan zonder waarde, maar die heeft er een nodig. Geef `x` door: `fn(x)`.")
_OUTSIDE = Mistake("wrapper-outside", syntax=r"^def\s+wrapper\b",
                   en="`def wrapper` stands at the left edge, outside the decorator, so it cannot use `fn`. Indent it under the `def` line of the decorator, so that it is written inside it.",
                   nl="`def wrapper` staat helemaal links, buiten de decorator, dus kan hij `fn` niet gebruiken. Spring hem in onder de `def`-regel van de decorator, zodat hij erin staat.")
_NO_WRAPPER = Mistake("no-wrapper", syntax=r"(?s)\A(?!.*\bdef\s+wrapper\b).*",
                      en="Your decorator has no `wrapper`, so there is no new function to give back. Write `def wrapper(x):` inside it, with the work in it, and end the decorator with `return wrapper`.",
                      nl="Je decorator heeft geen `wrapper`, dus is er geen nieuwe functie om terug te geven. Schrijf `def wrapper(x):` erin, met het werk daarin, en sluit de decorator af met `return wrapper`.")
_CALLS_EXAMPLE = Mistake("calls-example", syntax=r"\b(square|half|minus|shift|flip|same)\s*\(",
                         en="Your decorator calls a function of the check by its name. A decorator works for every function: call the one it gets, `fn`, with `fn(x)`.",
                         nl="Je decorator roept een functie van de check aan bij zijn naam. Een decorator werkt voor elke functie: roep de functie aan die hij krijgt, `fn`, met `fn(x)`.")
_RETURNS_FN = Mistake("returns-fn", matches=lambda got, text: re.search(r"^    return\s+fn\s*$", text, re.MULTILINE) is not None,
                      en="Your decorator gives back `fn`, the old function, so after the `@` line nothing has changed. Give back the new function: `return wrapper`.",
                      nl="Je decorator geeft `fn` terug, de oude functie, dus is er na de `@`-regel niets veranderd. Geef de nieuwe functie terug: `return wrapper`.")
_GOT_FUNCTION = Mistake("got-function", matches=lambda got, text: callable(got) or (isinstance(got, list) and any(callable(g) for g in got)),
                        en="You get a function back, not a number: `fn` without parentheses is the function itself. Call it: `fn(x)`.",
                        nl="Je krijgt een functie terug, geen getal: `fn` zonder haakjes is de functie zelf. Roep hem aan: `fn(x)`.")
# No `return` at the decorator's own level (4 spaces): the decorator gives back nothing, so the decorated name is None.
_NO_RETURN = Mistake("no-return-wrapper", syntax=r"(?ms)\A(?!.*^    return\b).*",
                     en="Your decorator gives back nothing, so after the `@` line the function's name stands for `None`, and calling it stops with an error. End the decorator with `return wrapper`, as far in as `def wrapper`.",
                     nl="Je decorator geeft niets terug, dus staat na de `@`-regel de naam van de functie voor `None`, en stopt Python met een fout als je hem aanroept. Sluit de decorator af met `return wrapper`, even ver ingesprongen als `def wrapper`.")
_FN_CALLED = Mistake("fn-not-called", matches=lambda got, text: re.search(r"\bfn\s*\(", text) is None,
                     en="After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Your `wrapper` never calls `fn`, the function it wraps. Call it with `fn(x)`, and use what it gives back.",
                     nl="Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Je `wrapper` roept `fn`, de functie die hij inpakt, nooit aan. Roep hem aan met `fn(x)`, en gebruik wat hij teruggeeft.")
_NOTHING = Mistake("wrapper-nothing", matches=lambda got, text: got is None or (isinstance(got, list) and None in got),
                   en="After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`: `wrapper` gives back nothing, `None`, for at least one value. Does every way through `wrapper` end with `return` and a value, and not with `print`?",
                   nl="Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`: `wrapper` geeft voor minstens één waarde niets terug, `None`. Eindigt elke weg door `wrapper` met `return` en een waarde, en niet met `print`?")

# The lesson is done when these are correct. "same-as" is the choice question in the lesson.
REQUIRED = ["plus-ten", "same-as", "no-negatives"]

_PLUS_START = "def plus_ten(fn):\n    def wrapper(x):\n        ...\n    return wrapper"
_PLUS = "def plus_ten(fn):\n    def wrapper(x):\n        return fn(x) + 10\n    return wrapper"
_NONNEG_START = "def no_negatives(fn):\n    ...\n"
_NONNEG = ("def no_negatives(fn):\n    def wrapper(x):\n        result = fn(x)\n        if result < 0:\n            return 0\n"
           "        return result\n    return wrapper")

EXERCISES = {
    "plus-ten": ScenarioBox(
        name="plus-ten",
        start=_PLUS_START,
        solution=_PLUS,
        scenarios=[
            "@plus_ten\ndef square(x):\n    return x * x\n\nsquare(3)",
            "@plus_ten\ndef half(x):\n    return x / 2\n\nhalf(5)",
            "def same(x):\n    return x\n\nsame_plus = plus_ten(same)\nsame_plus(0)",
            "@plus_ten\ndef minus(x):\n    return -x\n\n[minus(1), minus(20)]",
        ],
        first=[_DOTS],
        mistakes=[
            _COLON, _CALLED, _NO_X, _FN_EMPTY, _OUTSIDE, _CALLS_EXAMPLE, _NO_WRAPPER, _NO_RETURN,
            Mistake("adds-to-input", wrong=_PLUS.replace("return fn(x) + 10", "return fn(x + 10)"),
                    en="After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Your `wrapper` adds 10 to `x` before it calls `fn`. Add 10 to what `fn(x)` gives back instead.",
                    nl="Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Je `wrapper` telt 10 op bij `x` voordat hij `fn` aanroept. Tel 10 op bij wat `fn(x)` teruggeeft."),
            _RETURNS_FN, _GOT_FUNCTION, _NOTHING, _FN_CALLED,
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` in `wrapper` with one line: give back what `fn` gives for `x`, plus 10.",
                      "nl": "Vervang de `...` in `wrapper` door één regel: geef terug wat `fn` voor `x` geeft, plus 10."},
            "cleared": {"en": "There is no code in the box any more. Start again with `def plus_ten(fn):`, with `def wrapper(x):` inside it, and `return wrapper` at the end.",
                        "nl": "Er staat geen code meer in het vak. Begin opnieuw met `def plus_ten(fn):`, met `def wrapper(x):` erin, en `return wrapper` aan het eind."},
            "wrong": {"en": "After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: `square(3)` is 9, plus 10. Does `wrapper` give back `fn(x)` plus 10?",
                      "nl": "Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: `square(3)` is 9, plus 10. Geeft `wrapper` `fn(x)` plus 10 terug?"},
            "almost": {"en": "Right for the example. But after these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Does `wrapper` pass its own `x` to `fn`, and add 10 to exactly what `fn` gives back?",
                       "nl": "Goed voor het voorbeeld. Maar na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Geeft `wrapper` zijn eigen `x` door aan `fn`, en telt hij 10 op bij precies wat `fn` teruggeeft?"},
            "correct": {"en": "Your decorator is right in all {n} cases: every function it wraps gives back 10 more.",
                        "nl": "Je decorator klopt in alle {n} gevallen: elke functie die hij inpakt, geeft 10 meer terug."},
        },
    ),
    "no-negatives": ScenarioBox(
        name="no-negatives",
        start=_NONNEG_START,
        solution=_NONNEG,
        scenarios=[
            "@no_negatives\ndef shift(x):\n    return x - 5\n\n[shift(8), shift(2)]",
            "@no_negatives\ndef flip(x):\n    return -x\n\n[flip(-4), flip(4), flip(0)]",
            "def same(x):\n    return x\n\nsame_safe = no_negatives(same)\nsame_safe(-1)",
            "@no_negatives\ndef shift(x):\n    return x - 5\n\n[shift(-10), shift(4.5), shift(5), shift(5.5)]",
        ],
        first=[_DOTS],
        mistakes=[
            _COLON, _CALLED, _NO_X, _FN_EMPTY, _OUTSIDE, _CALLS_EXAMPLE, _NO_WRAPPER, _NO_RETURN,
            Mistake("clamps-input", wrong=_NONNEG.replace("        result = fn(x)\n        if result < 0:\n            return 0\n        return result",
                                                          "        if x < 0:\n            return 0\n        return fn(x)"),
                    en="After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Your `wrapper` looks at `x`, what goes in. Look at what `fn(x)` gives back: if that is below 0, give back 0.",
                    nl="Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Je `wrapper` kijkt naar `x`, wat erin gaat. Kijk naar wat `fn(x)` teruggeeft: is dat kleiner dan 0, geef dan 0 terug."),
            _RETURNS_FN, _GOT_FUNCTION, _NOTHING, _FN_CALLED,
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a `wrapper` inside `no_negatives`, and `return wrapper` at the end.",
                      "nl": "Vervang de `...` door een `wrapper` in `no_negatives`, en `return wrapper` aan het eind."},
            "cleared": {"en": "There is no code in the box any more. Start again with `def no_negatives(fn):`, and under it, indented, your lines.",
                        "nl": "Er staat geen code meer in het vak. Begin opnieuw met `def no_negatives(fn):`, met daaronder, ingesprongen, je regels."},
            "wrong": {"en": "After these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`: `shift(8)` is 3, and `shift(2)` is -3, which becomes 0. Does `wrapper` give back 0 when `fn(x)` is below 0, and `fn(x)` itself otherwise?",
                      "nl": "Na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn: `shift(8)` is 3, en `shift(2)` is -3, en dat wordt 0. Geeft `wrapper` 0 terug als `fn(x)` kleiner is dan 0, en anders `fn(x)` zelf?"},
            "almost": {"en": "Right for the example. But after these lines:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`. Does your `wrapper` work for every function, also for 0, for results just below 0, and for floats such as 5.5?",
                       "nl": "Goed voor het voorbeeld. Maar na deze regels:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn. Werkt je `wrapper` voor elke functie, ook voor 0, voor uitkomsten net onder 0, en voor floats zoals 5,5?"},
            "correct": {"en": "Your decorator is right in all {n} cases: a result below 0 becomes 0, and every other result stays as it was.",
                        "nl": "Je decorator klopt in alle {n} gevallen: een uitkomst onder 0 wordt 0, en elke andere uitkomst blijft zoals hij was."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("plus-ten", _PLUS_START, "empty", None),
    ("plus-ten", _PLUS, "correct", None),
    ("plus-ten", _PLUS.replace("return fn(x) + 10", "result = fn(x)\n        return result + 10"), "correct", None),
    ("plus-ten", _PLUS.replace("return fn(x) + 10", "return fn(x + 10)"), "wrong", "adds-to-input"),
    ("plus-ten", _PLUS.replace("return fn(x) + 10", "print(fn(x) + 10)"), "wrong", "wrapper-nothing"),
    ("plus-ten", _PLUS.replace("return fn(x) + 10", "return x + 10"), "wrong", "fn-not-called"),
    ("plus-ten", _PLUS.replace("    return wrapper", "    return wrapper(x)"), "error", "returns-call"),
    ("plus-ten", _PLUS.replace("\n    return wrapper", ""), "error", "no-return-wrapper"),
    ("plus-ten", _PLUS.replace("return fn(x) + 10", "return fn(x) + 1"), "wrong", None),
    ("plus-ten", _PLUS.replace("def wrapper(x):", "def wrapper(x)"), "error", "colon-missing"),
    ("plus-ten", "", "empty", None),
    ("plus-ten", _PLUS.replace("def wrapper(x):", "def wrapper():").replace("fn(x)", "fn()"), "error", "wrapper-no-x"),
    ("plus-ten", _PLUS.replace("fn(x)", "fn()"), "error", "fn-no-x"),
    ("plus-ten", "def wrapper(x):\n    return fn(x) + 10\n\ndef plus_ten(fn):\n    return wrapper", "error", "wrapper-outside"),
    ("plus-ten", _PLUS.replace("return fn(x) + 10", "return square(x) + 10"), "error", "calls-example"),
    ("plus-ten", "def plus_ten(fn):\n    return fn", "wrong", "returns-fn"),
    ("plus-ten", _PLUS.replace("    return wrapper", "    return fn"), "wrong", "returns-fn"),
    ("plus-ten", _PLUS.replace("return fn(x) + 10", "return fn"), "wrong", "got-function"),
    ("plus-ten", _PLUS.replace("return fn(x) + 10", "y = fn(x)\n        ..."), "wrong", "dots-left"),
    ("no-negatives", _NONNEG_START, "empty", None),
    ("no-negatives", _NONNEG, "correct", None),
    ("no-negatives", _NONNEG.replace("        if result < 0:\n            return 0\n        return result",
                                      "        if result < 0:\n            result = 0\n        return result"), "correct", None),
    ("no-negatives", _NONNEG.replace("        result = fn(x)\n        if result < 0:\n            return 0\n        return result",
                                      "        if x < 0:\n            return 0\n        return fn(x)"), "wrong", "clamps-input"),
    ("no-negatives", _NONNEG.replace("if result < 0:", "if result <= 0:"), "correct", None),
    ("no-negatives", _NONNEG.replace("        return result\n", ""), "wrong", "wrapper-nothing"),
    ("no-negatives", _NONNEG.replace("    return wrapper", "    return wrapper(x)"), "error", "returns-call"),
    ("no-negatives", _NONNEG.replace("\n    return wrapper", ""), "error", "no-return-wrapper"),
    ("no-negatives", _NONNEG.replace("            return 0", "            return -result"), "wrong", None),
    ("no-negatives", _NONNEG.replace("if result < 0:", "if result <= -1:"), "almost", None),
    ("no-negatives", "def no_negatives(fn):\n    result = fn(x)\n    if result < 0:\n        return 0\n    return result", "error", "no-wrapper"),
    ("no-negatives", _NONNEG_START.replace("    ...\n", "    ...\n    return fn\n"), "wrong", "dots-left"),
]
