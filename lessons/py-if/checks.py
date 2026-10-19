# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson py-if, and the worked example it shows for two amounts."""
import re

from everything_ai.check import CodeBox, Mistake, run_code

# The worked example, with its comments per language. Python works out what it shows for each amount.
EXAMPLE = {
    "en": ("spent = 12.5\n"
           "# 1 and 2. the condition spent >= 10, after if, with a colon\n"
           "if spent >= 10:\n"
           "    # 3. indented: these lines run only when the condition is True\n"
           "    print(\"Free cookie!\")\n"
           "    cookies = 1\n"
           "# 4. else at the left edge, with its own indented lines\n"
           "else:\n"
           "    print(\"No cookie this time.\")\n"
           "    cookies = 0\n"
           "# not indented: this line always runs\n"
           "print(cookies)"),
    "nl": ("spent = 12.5\n"
           "# 1 en 2. de voorwaarde spent >= 10, na if, met een dubbele punt\n"
           "if spent >= 10:\n"
           "    # 3. ingesprongen: deze regels draaien alleen als de voorwaarde True is\n"
           "    print(\"Gratis koekje!\")\n"
           "    cookies = 1\n"
           "# 4. else helemaal links, met eigen ingesprongen regels\n"
           "else:\n"
           "    print(\"Deze keer geen koekje.\")\n"
           "    cookies = 0\n"
           "# niet ingesprongen: deze regel draait altijd\n"
           "print(cookies)"),
}
OTHER_SPENT = 7
EXAMPLE_SHOWS = {lang: run_code(code).stdout for lang, code in EXAMPLE.items()}
EXAMPLE_OTHER = {lang: run_code(code.replace("spent = 12.5", f"spent = {OTHER_SPENT}")).stdout for lang, code in EXAMPLE.items()}


def _lines(text) -> list[str]:
    return [line.rstrip() for line in str(text).rstrip("\n").splitlines()]


def _same_line(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-9
    except ValueError:
        return a.strip() == b.strip()


def same_lines(a, b) -> bool:
    """The same lines under the cell; 9 and 9.0 count as the same number."""
    return (isinstance(a, str) and isinstance(b, str) and len(_lines(a)) == len(_lines(b))
            and all(_same_line(x, y) for x, y in zip(_lines(a), _lines(b))))


def _shows(*lines):
    return lambda got, text: len(_lines(got)) == len(lines) and all(_same_line(x, y) for x, y in zip(_lines(got), lines))


def _has_if(text):
    return re.search(r"^[ \t]*if\b", text, re.MULTILINE) is not None


# Python cannot read these, or stops at them; the pattern finds out why. They hold for both exercises.
_SYNTAX = [
    Mistake("curly-quote", syntax=r"[“”‘’„]",
            en="Your program has a curly quote, like `“`. Python only reads straight quotes, like `\"`: type the quotes again in the box.",
            nl="Je programma heeft een gekruld aanhalingsteken, zoals `“`. Python leest alleen rechte, zoals `\"`: typ de aanhalingstekens opnieuw in het vak."),
    Mistake("capital", syntax=r"^[ \t]*(If|IF|Else|ELSE)\b",
            en="Python writes `if` and `else` in small letters. On the line `{line}`, change the capital.",
            nl="Python schrijft `if` en `else` met kleine letters. Verander op de regel `{line}` de hoofdletter."),
    Mistake("else-condition", syntax=r"^[ \t]*else[ \t]+[^:\s]",
            en="`else` gets no condition of its own: it runs when the condition after `if` is False. Write just `else:`.",
            nl="`else` krijgt geen eigen voorwaarde: hij draait als de voorwaarde na `if` False is. Schrijf alleen `else:`."),
    Mistake("colon-missing", syntax=r"^[ \t]*(if\b[^:\n]*|else[ \t]*)$",
            en="The line `{line}` must end with a colon, `:`. The colon tells Python that the indented lines below belong to it.",
            nl="De regel `{line}` moet eindigen op een dubbele punt, `:`. Die vertelt Python dat de ingesprongen regels eronder erbij horen."),
    Mistake("sign-order", syntax=r"^[ \t]*if\b[^\n]*(=<|=>)",
            en="On the line `{line}`, the two signs are the wrong way around: \"less than or equal to\" is `<=`, and \"greater than or equal to\" is `>=`, with the `=` second.",
            nl="Op de regel `{line}` staan de twee tekens andersom: \"kleiner dan of gelijk aan\" is `<=`, en \"groter dan of gelijk aan\" is `>=`, met de `=` als tweede."),
    Mistake("single-equals", syntax=r"^[ \t]*if\b[^\n]*[^=!<>]=[^=]",
            en="In a condition, \"equal to\" is written with two equals signs: `==`. One `=` gives a name a value, and that cannot go after `if`.",
            nl="In een voorwaarde schrijf je \"gelijk aan\" met twee isgelijktekens: `==`. Eén `=` geeft een naam een waarde, en dat kan niet na `if`."),
    Mistake("compare-not-give", syntax=r"^[ \t]+price[ \t]*==",
            en="`==` asks whether `price` is equal to a number; it does not give `price` a value. To give it a value, use one `=`.",
            nl="`==` vraagt of `price` gelijk is aan een getal; het geeft `price` geen waarde. Een waarde geven doe je met één `=`."),
    Mistake("else-indented", syntax=r"^[ \t]+else\b",
            en="`else:` starts at the left edge, right under its `if`, not indented. Only the lines under `else:` are indented.",
            nl="`else:` begint helemaal links, recht onder zijn `if`, niet ingesprongen. Alleen de regels onder `else:` springen in."),
    Mistake("dots-left", syntax=r"\.\.\.",
            en="There is still a `...` in your program. Replace it with your `if` and `else`.",
            nl="Er staat nog een `...` in je programma. Vervang die door je `if` en `else`."),
]

# The lesson is done when these are correct. "boundary" is the choice question in the lesson.
REQUIRED = ["below-zero", "boundary", "ticket"]

EXERCISES = {
    "below-zero": CodeBox(
        name="below-zero",
        start='temperature = -3\n...\n',
        keep=["temperature = -3"],
        solution='temperature = -3\nif temperature < 0:\n    print("Below zero")\nelse:\n    print("Zero or above")',
        outcome="stdout",
        expect="Below zero\n",
        compare=same_lines,
        others=[{"temperature": 0}, {"temperature": 5}, {"temperature": -0.5}],
        mistakes=_SYNTAX + [
            Mistake("both", matches=_shows("Below zero", "Zero or above"),
                    en="Both lines appear. The second `print` must go under `else:`, indented, so it only runs when the condition is False.",
                    nl="Beide regels verschijnen. De tweede `print` hoort onder `else:`, ingesprongen, zodat hij alleen draait als de voorwaarde False is."),
            Mistake("capitals", matches=lambda got, text: [l.lower().rstrip(".!") for l in _lines(got)] == ["below zero"], state="almost",
                    en="The words are right, but Python shows text exactly as you write it. Use `Below zero` and `Zero or above`, with the same capitals and nothing after them.",
                    nl="De woorden kloppen, maar Python laat tekst precies zo zien als je hem schrijft. Gebruik `Below zero` en `Zero or above`, met dezelfde hoofdletters en niets erachter."),
            Mistake("swapped", matches=lambda got, text: _lines(got) == ["Zero or above"] and _has_if(text),
                    en="Your program shows `Zero or above`, while `temperature` is -3. Look at the condition: is it True for -3? And which line belongs under `if`?",
                    nl="Je programma laat `Zero or above` zien, terwijl `temperature` -3 is. Kijk naar de voorwaarde: is die True voor -3? En welke regel hoort onder `if`?"),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with an `if` and an `else` that show `Below zero` or `Zero or above`.",
                      "nl": "Vervang de `...` door een `if` en een `else` die `Below zero` of `Zero or above` laten zien."},
            "cleared": {"en": "There is no code in the box any more. Type `temperature = -3` back on the first line, and write your `if` and `else` under it.",
                        "nl": "Er staat geen code meer in het vak. Typ `temperature = -3` terug op de eerste regel, en schrijf je `if` en `else` eronder."},
            "nothing": {"en": "Your program shows nothing for -3: no `print` ran. Is the line with `Below zero` under the `if`, and is your condition `True` for -3?",
                        "nl": "Je programma laat bij -3 niets zien: er draaide geen `print`. Staat de regel met `Below zero` onder de `if`, en is je voorwaarde `True` bij -3?"},
            "other": {"en": "Right for -3. But with {where} instead, your program shows:\n\n```\n{got}\n```\n\nand it should show `{want}`. Look at your condition: is it `True` for exactly the temperatures below zero?",
                      "nl": "Goed bij -3. Maar met {where} laat je programma dit zien:\n\n```\n{got}\n```\n\nen het zou `{want}` moeten zijn. Kijk naar je voorwaarde: is die `True` bij precies de temperaturen onder nul?"},
            "wrong": {"en": "Your program shows:\n\n```\n{got}\n```\n\nFor -3 it should show `Below zero`, and nothing else. Follow the recipe step by step.",
                      "nl": "Je programma laat dit zien:\n\n```\n{got}\n```\n\nBij -3 hoort het alleen `Below zero` te laten zien. Volg het recept stap voor stap."},
            "correct": {"en": "Your program shows `Below zero` for -3, and the right line for 0, 5 and -0.5 too: below zero only when `temperature < 0` is `True`.",
                        "nl": "Je programma laat `Below zero` zien bij -3, en ook bij 0, bij 5 en bij -0,5 de goede regel: onder nul alleen als `temperature < 0` `True` is."},
        },
    ),
    "ticket": CodeBox(
        name="ticket",
        start='age = 15\n...\nprint(price)',
        keep=["age = 15"],
        solution='age = 15\nif age < 12:\n    price = 5\nelse:\n    price = 9\nprint(price)',
        outcome="stdout",
        expect="9\n",
        compare=same_lines,
        others=[{"age": 8}, {"age": 12}, {"age": 11}],
        mistakes=_SYNTAX + [
            Mistake("price-never", syntax=r"\A(?![\s\S]*^[ \t]*price[ \t]*=(?!=))[\s\S]*print\(price\)",
                    en="Your program never gives `price` a value, so `print(price)` does not know it. Under `if` and under `else`, give `price` a value with `=`; the last line shows it.",
                    nl="Je programma geeft `price` nooit een waarde, dus kent `print(price)` hem niet. Geef `price` onder `if` en onder `else` een waarde met `=`; de laatste regel laat hem zien."),
            Mistake("no-else", syntax=r"\A(?![\s\S]*^\s*else\b)[\s\S]*^\s*if\b",
                    en="When the condition is False, your program never gives `price` a value, so `print(price)` does not know it. Add `else:` with the other price.",
                    nl="Als de voorwaarde False is, geeft je programma `price` nooit een waarde, dus kent `print(price)` hem niet. Voeg `else:` toe met de andere prijs."),
            Mistake("swapped", matches=lambda got, text: _shows("5")(got, text) and _has_if(text),
                    en="Your program shows 5 for a 15-year-old. Children under 12 pay 5; look at your condition and at which price is under `if`.",
                    nl="Je programma laat 5 zien voor iemand van 15. Kinderen onder de 12 betalen 5; kijk naar je voorwaarde en naar welke prijs onder `if` staat."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with an `if` and an `else` that give `price` the right value.",
                      "nl": "Vervang de `...` door een `if` en een `else` die `price` de goede waarde geven."},
            "cleared": {"en": "There is no code in the box any more. Type `age = 15` back on the first line and `print(price)` on the last, and write your `if` and `else` in between.",
                        "nl": "Er staat geen code meer in het vak. Typ `age = 15` terug op de eerste regel en `print(price)` op de laatste, en schrijf je `if` en `else` ertussen."},
            "nothing": {"en": "Your program shows nothing. Keep `print(price)` as the last line, at the left edge, so it runs in both cases.",
                        "nl": "Je programma laat niets zien. Houd `print(price)` als laatste regel, helemaal links, zodat hij in beide gevallen draait."},
            "other": {"en": "Right for 15. But with {where} instead, your program shows `{got}`, and it should show `{want}`. Look at your condition: is it `True` for exactly the children under 12?",
                      "nl": "Goed bij 15. Maar met {where} laat je programma `{got}` zien, en het zou `{want}` moeten zijn. Kijk naar je voorwaarde: is die `True` bij precies de kinderen onder de 12?"},
            "wrong": {"en": "Your program shows:\n\n```\n{got}\n```\n\nA 15-year-old pays 9. Follow the recipe: which condition, and which price under `if` and under `else`?",
                      "nl": "Je programma laat dit zien:\n\n```\n{got}\n```\n\nIemand van 15 betaalt 9. Volg het recept: welke voorwaarde, en welke prijs onder `if` en onder `else`?"},
            "correct": {"en": "Your program shows 9 for a 15-year-old, and the right price for the ages 8, 11 and 12 too: only children under 12 pay 5.",
                        "nl": "Je programma laat 9 zien voor iemand van 15, en ook bij 8, 11 en 12 jaar de goede prijs: alleen kinderen onder de 12 betalen 5."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_B = "temperature = -3\n"
_T = "age = 15\n"
CASES = [
    ("below-zero", "temperature = -3\n...\n", "empty", None),
    ("below-zero", "", "empty", None),
    ("below-zero", _B + 'if temperature < 0:\n    print("Below zero")\nelse:\n    print("Zero or above")', "correct", None),
    ("below-zero", _B + 'if temperature >= 0:\n    print("Zero or above")\nelse:\n    print("Below zero")', "correct", None),
    ("below-zero", _B + 'if temperature <= 0:\n    print("Below zero")\nelse:\n    print("Zero or above")', "almost", "other-value"),
    ("below-zero", _B + 'print("Below zero")', "almost", "other-value"),
    ("below-zero", _B + 'if temperature < 0:\n    print("Below zero")\nprint("Zero or above")', "wrong", "both"),
    ("below-zero", _B + 'if temperature > 0:\n    print("Below zero")\nelse:\n    print("Zero or above")', "wrong", "swapped"),
    ("below-zero", _B + 'if temperature < 0\n    print("Below zero")\nelse:\n    print("Zero or above")', "error", "colon-missing"),
    ("below-zero", _B + 'if temperature < 0:\n    print("Below zero")\nelse\n    print("Zero or above")', "error", "colon-missing"),
    ("below-zero", _B + 'If temperature < 0:\n    print("Below zero")\nelse:\n    print("Zero or above")', "error", "capital"),
    ("below-zero", _B + 'if temperature < 0:\n    print("Below zero")\nelse temperature >= 0:\n    print("Zero or above")', "error", "else-condition"),
    ("below-zero", _B + 'if temperature = -3:\n    print("Below zero")\nelse:\n    print("Zero or above")', "error", "single-equals"),
    ("below-zero", _B + 'if temperature < 0:\n    print("Below zero")\n    else:\n    print("Zero or above")', "error", "else-indented"),
    ("below-zero", _B + 'if temperature < 0:\nprint("Below zero")\nelse:\n    print("Zero or above")', "error", None),
    ("below-zero", _B + 'if temperature <= -1:\n    print("Below zero")\nelse:\n    print("Zero or above")', "almost", "other-value"),
    ("below-zero", _B + 'if temperature =< 0:\n    print("Below zero")\nelse:\n    print("Zero or above")', "error", "sign-order"),
    ("below-zero", _B + 'if temperature < 0:\n    print("below zero")\nelse:\n    print("zero or above")', "almost", "capitals"),
    ("below-zero", 'temperature = 5\nif temperature < 0:\n    print("Below zero")\nelse:\n    print("Zero or above")', "wrong", "changed-line"),
    ("below-zero", _B + 'if temperature >= 0:\n    print("Zero or above")', "wrong", "nothing"),
    ("below-zero", _B + 'print("Zero or above")', "wrong", None),
    ("ticket", "age = 15\n...\nprint(price)", "empty", None),
    ("ticket", _T + "if age < 12:\n    price = 5\nelse:\n    price = 9\nprint(price)", "correct", None),
    ("ticket", _T + "if age >= 12:\n    price = 9\nelse:\n    price = 5\nprint(price)", "correct", None),
    ("ticket", _T + "price = 9\nif age < 12:\n    price = 5\nprint(price)", "correct", None),
    ("ticket", _T + "if age <= 12:\n    price = 5\nelse:\n    price = 9\nprint(price)", "almost", "other-value"),
    ("ticket", _T + "if age < 12:\n    price = 5\nprint(price)", "error", "no-else"),
    ("ticket", _T + "if age > 12:\n    price = 5\nelse:\n    price = 9\nprint(price)", "wrong", "swapped"),
    ("ticket", _T + "price = 9\nprint(price)", "almost", "other-value"),
    ("ticket", _T + "if age < 12:\n    price = 5.0\nelse:\n    price = 9.0\nprint(price)", "correct", None),
    ("ticket", _T + "if age < 12:\n    price == 5\nelse:\n    price == 9\nprint(price)", "error", "compare-not-give"),
    ("ticket", _T + "if age < 12:\n    print(5)\nelse:\n    print(9)\nprint(price)", "error", "price-never"),
    ("ticket", "age = 8\nif age < 12:\n    price = 5\nelse:\n    price = 9\nprint(price)", "wrong", "changed-line"),
]
