# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson py-recursion, and the calls in its picture."""
from everything_ai.check import FunctionBox, Mistake, calls_itself

# The picture: sum_to(3), call by call. Python works out what each call gives back.


def _sum_to(n):
    return 0 if n == 0 else n + _sum_to(n - 1)


PICTURE_N = 3
PICTURE_LINES = {
    "en": [f"sum_to({k}) gives {k} + sum_to({k - 1})" for k in range(PICTURE_N, 0, -1)] + ["sum_to(0) gives 0, the base case"],
    "nl": [f"sum_to({k}) geeft {k} + sum_to({k - 1})" for k in range(PICTURE_N, 0, -1)] + ["sum_to(0) geeft 0, het basisgeval"],
}
PICTURE_BACK = [f"{_sum_to(k)}\n" for k in range(PICTURE_N, -1, -1)]

_SYNTAX = [
    Mistake("dots-left", syntax=r"^[^#\n]*\.\.\.",
            en="There is still a `...` in your function. Replace it with the base case and the call to itself.",
            nl="Er staat nog een `...` in je functie. Vervang die door het basisgeval en de aanroep van zichzelf."),
    Mistake("curly-quote", syntax=r"[“”‘’„]",
            en="Your code has a curly quote, like `“`. Python only reads straight quotes, like `\"`: type the quotes again in the box.",
            nl="Je code heeft een gekruld aanhalingsteken, zoals `“`. Python leest alleen rechte, zoals `\"`: typ de aanhalingstekens opnieuw in het vak."),
    Mistake("colon-missing", syntax=r"^[ \t]*(def\b[^:\n]*|if\b[^:\n]*|else[ \t]*)$",
            en="The line `{line}` must end with a colon, `:`. The colon tells Python that the indented lines below belong to it.",
            nl="De regel `{line}` moet eindigen op een dubbele punt, `:`. Die vertelt Python dat de ingesprongen regels eronder erbij horen."),
    Mistake("single-equals", syntax=r"^[ \t]*if\b[^\n]*[^=!<>]=[^=]",
            en="In a condition, \"equal to\" is written with two equals signs: `==`. One `=` gives a name a value.",
            nl="In een voorwaarde schrijf je \"gelijk aan\" met twee isgelijktekens: `==`. Eén `=` geeft een naam een waarde."),
    Mistake("number-called", syntax=r"\bn\s*\(",
            en="`n(...)` asks Python to call `n` as a function, but `n` is a number. To multiply, write `*`: `n * factorial(n - 1)`.",
            nl="`n(...)` vraagt Python om `n` aan te roepen als functie, maar `n` is een getal. Vermenigvuldigen schrijf je met `*`: `n * factorial(n - 1)`."),
    Mistake("text-called", syntax=r"\btext\s*\(",
            en="`text(...)` asks Python to call `text` as a function, but it is a string. To take a character, use square brackets: `text[0]`.",
            nl="`text(...)` vraagt Python om `text` aan te roepen als functie, maar het is een string. Een teken pak je met vierkante haken: `text[0]`."),
]


def _not_itself(name):
    """The function gives the right answers, but does not call itself."""
    return [Mistake("no-recursion", matches=lambda got, text: not calls_itself(text, name), state="almost",
                    en=f"Right in all cases, but your function does not call itself. This exercise is about exactly that: a base case, and a call to `{name}` on a smaller piece. A loop or another way works too, but try it this way.",
                    nl=f"Goed in alle gevallen, maar je functie roept zichzelf niet aan. Daar gaat deze oefening juist over: een basisgeval, en een aanroep van `{name}` met een kleiner stuk. Een loop of een andere manier werkt ook, maar probeer het op deze manier.")]

# The lesson is done when these are correct. "sum-four" is the choice question in the lesson.
REQUIRED = ["factorial", "sum-four", "reverse"]

EXERCISES = {
    "factorial": FunctionBox(
        name="factorial",
        function="factorial",
        start="def factorial(n):\n    ...\n",
        solution="def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)",
        visible=[(4,)],
        hidden=[(0,), (1,), (6,)],
        form=_not_itself("factorial"),
        mistakes=_SYNTAX + [
            Mistake("one-only", syntax=r"\A(?![\s\S]*\bn\s*(==\s*0|<=\s*[01]|<\s*[12])\b)[\s\S]*^\s*if\s+n\s*==\s*1\s*:",
                    en="Your base case catches 1, but the check also calls `factorial(0)`. From 0, `n - 1` goes to -1, -2, and so on: it never reaches 1, so the calls never stop. Catch 0 in the base case: its factorial is 1.",
                    nl="Je basisgeval vangt 1 af, maar de check roept ook `factorial(0)` aan. Vanaf 0 gaat `n - 1` naar -1, -2, enzovoort: het komt nooit bij 1, dus de aanroepen stoppen nooit. Vang 0 af in het basisgeval: de faculteit daarvan is 1."),
            Mistake("zero-gives-zero", state="almost", wrong="def factorial(n):\n    if n <= 1:\n        return n\n    return n * factorial(n - 1)",
                    en="`{call}` gives `{got}`, and the factorial of 0 is 1. Let your base case give back 1 for 0.",
                    nl="`{call}` geeft `{got}`, en de faculteit van 0 is 1. Laat je basisgeval bij 0 het getal 1 teruggeven."),
            Mistake("added", wrong="def factorial(n):\n    if n == 0:\n        return 0\n    return n + factorial(n - 1)",
                    en="`{call}` gives `{got}`: your function adds the numbers. The factorial multiplies them: `4 * 3 * 2 * 1`.",
                    nl="`{call}` geeft `{got}`: je functie telt de getallen op. De faculteit vermenigvuldigt ze: `4 * 3 * 2 * 1`."),
            Mistake("base-zero", wrong="def factorial(n):\n    if n == 0:\n        return 0\n    return n * factorial(n - 1)",
                    en="`{call}` gives `{got}`: somewhere your function multiplies by 0, and everything times 0 is 0. The factorial of 0 is 1: does your base case give back 1?",
                    nl="`{call}` geeft `{got}`: ergens vermenigvuldigt je functie met 0, en alles keer 0 is 0. De faculteit van 0 is 1: geeft je basisgeval 1 terug?"),
            Mistake("no-multiply", wrong="def factorial(n):\n    if n == 0:\n        return 1\n    return factorial(n - 1)",
                    en="`{call}` gives `{got}`: your function calls itself, but does not multiply by `n`. Give back `n` times the factorial of `n - 1`.",
                    nl="`{call}` geeft `{got}`: je functie roept zichzelf aan, maar vermenigvuldigt niet met `n`. Geef `n` keer de faculteit van `n - 1` terug."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a base case for 0, and a line that calls `factorial` on `n - 1`.",
                      "nl": "Vervang de `...` door een basisgeval voor 0, en een regel die `factorial` aanroept met `n - 1`."},
            "cleared": {"en": "There is no code in the box any more. Start again with `def factorial(n):`, and under it, indented, your lines.",
                        "nl": "Er staat geen code meer in het vak. Begin opnieuw met `def factorial(n):`, met daaronder, ingesprongen, je regels."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: `4 * 3 * 2 * 1`. Look at the worked example: a base case, and a call to itself with a number one smaller.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: `4 * 3 * 2 * 1`. Kijk naar het uitgewerkte voorbeeld: een basisgeval, en een aanroep van zichzelf met een getal dat één kleiner is."},
            "almost": {"en": "Right for 4. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for every number? It needs a base case for 0, whose factorial is 1, and a call to itself for the others.",
                       "nl": "Goed voor 4. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie voor elk getal? Hij heeft een basisgeval nodig voor 0, waarvan de faculteit 1 is, en een aanroep van zichzelf voor de andere."},
            "correct": {"en": "Your function is right in all {n} cases, also for 0, whose factorial is 1.",
                        "nl": "Je functie klopt in alle {n} gevallen, ook bij 0, waarvan de faculteit 1 is."},
        },
    ),
    "reverse": FunctionBox(
        name="reverse",
        function="reverse",
        start="def reverse(text):\n    ...\n",
        solution='def reverse(text):\n    if text == "":\n        return ""\n    return reverse(text[1:]) + text[0]',
        visible=[("abc",)],
        hidden=[("",), ("a",), ("stop",), ("hello world",)],
        form=_not_itself("reverse"),
        mistakes=_SYNTAX + [
            Mistake("one-char-only", syntax=r"\A(?![\s\S]*(==\s*\"\"|==\s*''|<=\s*1|<\s*1|==\s*0|not\s+text))[\s\S]*len\(\s*text\s*\)\s*==\s*1",
                    en="Your base case catches a text of one character, but the check also calls `reverse(\"\")`. `\"\"[1:]` is `\"\"` again, so the empty text never gets shorter, and the calls never stop. Catch the empty text in the base case.",
                    nl="Je basisgeval vangt een tekst van één teken af, maar de check roept ook `reverse(\"\")` aan. `\"\"[1:]` is weer `\"\"`, dus de lege tekst wordt nooit korter, en de aanroepen stoppen nooit. Vang de lege tekst af in het basisgeval."),
            Mistake("rest-only", wrong='def reverse(text):\n    if text == "":\n        return ""\n    return reverse(text[1:])',
                    en="`{call}` gives `{got}`, the empty text. Each call drops its first character and never puts it back. Add `text[0]` to the result of the call.",
                    nl="`{call}` geeft `{got}`, de lege tekst. Elke aanroep laat zijn eerste teken weg en zet het nooit terug. Zet `text[0]` bij de uitkomst van de aanroep."),
            Mistake("same-order", wrong='def reverse(text):\n    if text == "":\n        return ""\n    return text[0] + reverse(text[1:])',
                    en="`{call}` gives `{got}`, the same text. The first character should come at the end: put `text[0]` after the call, not before it.",
                    nl="`{call}` geeft `{got}`, dezelfde tekst. Het eerste teken hoort achteraan: zet `text[0]` na de aanroep, niet ervoor."),
            Mistake("first-only", wrong='def reverse(text):\n    if text == "":\n        return ""\n    return text[0]',
                    en="`{call}` gives `{got}`: only the first character. The rest of the text has to be reversed too: call `reverse` on `text[1:]`.",
                    nl="`{call}` geeft `{got}`: alleen het eerste teken. De rest van de tekst moet ook achterstevoren: roep `reverse` aan met `text[1:]`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a base case for the empty text, and a line that calls `reverse` on the rest of the text.",
                      "nl": "Vervang de `...` door een basisgeval voor de lege tekst, en een regel die `reverse` aanroept met de rest van de tekst."},
            "cleared": {"en": "There is no code in the box any more. Start again with `def reverse(text):`, and under it, indented, your lines.",
                        "nl": "Er staat geen code meer in het vak. Begin opnieuw met `def reverse(text):`, met daaronder, ingesprongen, je regels."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. The reverse of `abc` is the reverse of `bc`, followed by `a`. Which call gives the reverse of the rest?",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn. `abc` achterstevoren is `bc` achterstevoren, met daarachter `a`. Welke aanroep geeft de rest achterstevoren?"},
            "almost": {"en": "Right for `abc`. But `{call}` gives `{got}`, and it should give `{want}`. Does your base case handle the empty text, and does every call make the text one shorter?",
                       "nl": "Goed voor `abc`. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Vangt je basisgeval de lege tekst op, en maakt elke aanroep de tekst één korter?"},
            "correct": {"en": "Your function is right in all {n} cases, also for an empty text and a text with a space.",
                        "nl": "Je functie klopt in alle {n} gevallen, ook bij een lege tekst en een tekst met een spatie."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_F = "def factorial(n):\n"
_R = "def reverse(text):\n"
CASES = [
    ("factorial", _F + "    ...\n", "empty", None),
    ("factorial", _F + "    if n == 0:\n        return 1\n    return n * factorial(n - 1)", "correct", None),
    ("factorial", _F + "  if n <= 1:\n    return 1\n  else:\n    return n * factorial(n - 1)", "correct", None),
    ("factorial", _F + "    result = 1\n    for k in range(1, n + 1):\n        result = result * k\n    return result", "almost", "no-recursion"),
    ("factorial", _F + "    if n <= 1:\n        return n\n    return n * factorial(n - 1)", "almost", "zero-gives-zero"),
    ("factorial", _F + "    if n == 0:\n        print(1)\n    else:\n        print(n * factorial(n - 1))", "error", None),
    ("factorial", _F + "    if n == 0:\n        return\n    return n * factorial(n - 1)", "error", None),
    ("factorial", _F + "    if n == 0:\n        return 1\n        return n * factorial(n - 1)", "wrong", "unreached"),
    ("factorial", _F + "    if n == 0:\n        return 1\n    return n(factorial(n - 1))", "error", "number-called"),
    ("factorial", _F + "    if n == 0:\n        return 0\n    return n + factorial(n - 1)", "wrong", "added"),
    ("factorial", _F + "    if n == 0:\n        return 0\n    return n * factorial(n - 1)", "wrong", "base-zero"),
    ("factorial", _F + "    if n == 0:\n        return 1\n    return factorial(n - 1)", "wrong", "no-multiply"),
    ("factorial", _F + "    if n == 1:\n        return 1\n    return n * factorial(n - 1)", "error", "one-only"),
    ("factorial", _F + "    return n * factorial(n - 1)", "error", None),
    ("factorial", _F + "    if n = 0:\n        return 1\n    return n * factorial(n - 1)", "error", "single-equals"),
    ("reverse", _R + "    ...\n", "empty", None),
    ("reverse", _R + '    if text == "":\n        return ""\n    return reverse(text[1:]) + text[0]', "correct", None),
    ("reverse", _R + '    if len(text) <= 1:\n        return text\n    return reverse(text[1:]) + text[0]', "correct", None),
    ("reverse", _R + '    if text == "":\n        return ""\n    return text[0] + reverse(text[1:])', "wrong", "same-order"),
    ("reverse", _R + '    if text == "":\n        return ""\n    return text[0]', "wrong", "first-only"),
    ("reverse", _R + "    return reverse(text[1:]) + text[0]", "error", None),
    ("reverse", _R + '    if len(text) == 1:\n        return text\n    return reverse(text[1:]) + text[0]', "error", "one-char-only"),
    ("reverse", _R + '    if text == "":\n        return ""\n    return reverse(text[1:])', "wrong", "rest-only"),
    ("reverse", _R + '    if text == "":\n        return ""\n    return reverse(text[1:]) + text(0)', "error", "text-called"),
    ("reverse", _R + "    return text[::-1]", "almost", "no-recursion"),
]
