# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson py-lists: a slice, and append. And the list in the picture."""
import re

from everything_ai.check import CodeBox, Mistake

# The list in the picture, and the slice it marks.
PICTURE = [7, 9, 4, 8, 6]
PICTURE_SPAN = (1, 3)


def _lines(text) -> list[str]:
    return [line.rstrip() for line in str(text).rstrip("\n").splitlines()]


def same_lines(a, b) -> bool:
    return isinstance(a, str) and isinstance(b, str) and _lines(a) == _lines(b)


def _shows(*texts):
    """The program showed exactly one of these lines."""
    return lambda got, text: _lines(got) in [[t] for t in texts]


_DOTS = [Mistake("dots-left", syntax=r"^[^#\n]*\.\.\.",
                 en="There is still a `...` in your program. Replace it with your own line.",
                 nl="Er staat nog een `...` in je programma. Vervang die door je eigen regel."),
         Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                 en="There is still a `...` in your program. Replace it with your own line.",
                 nl="Er staat nog een `...` in je programma. Vervang die door je eigen regel.")]
_CURLY = Mistake("curly-quote", syntax=r"[“”‘’„]",
                 en="Your program has a curly quote, like `“`. Python only reads straight quotes, like `\"`: type the quotes again in the box.",
                 nl="Je programma heeft een gekruld aanhalingsteken, zoals `“`. Python leest alleen rechte, zoals `\"`: typ de aanhalingstekens opnieuw in het vak.")

# The lesson is done when these are correct. "last-two" is the choice question in the lesson.
REQUIRED = ["top", "last-two", "cart"]

EXERCISES = {
    "top": CodeBox(
        name="top",
        start="scores = [7, 9, 4, 8, 6]\ntop = ...\nprint(top)",
        solution="scores = [7, 9, 4, 8, 6]\ntop = scores[:3]\nprint(top)",
        outcome="stdout",
        expect="[7, 9, 4]\n",
        compare=same_lines,
        others=[{"scores": [5, 1, 3, 2]}],
        keep=["scores = [7, 9, 4, 8, 6]"],
        mistakes=_DOTS + [
            _CURLY,
            Mistake("round-brackets", syntax=r"\w\(\s*-?\d*\s*:|scores\s*\(",
                    en="A slice goes between square brackets, `[ ]`, not round ones: `scores[1:3]`.",
                    nl="Een slice staat tussen vierkante haken, `[ ]`, niet tussen ronde haakjes: `scores[1:3]`."),
            Mistake("comma", syntax=r"\w\[[^\]\n]*,",
                    en="In a slice, the two numbers are separated by a colon, `:`, not by a comma.",
                    nl="In een slice staat tussen de twee getallen een dubbele punt, `:`, geen komma."),
            Mistake("two", matches=_shows("[7, 9]"),
                    en="Your program shows `[7, 9]`: two scores. A slice stops just before its second number: which stop gives three items?",
                    nl="Je programma laat `[7, 9]` zien: twee scores. Een slice stopt net voor zijn tweede getal: bij welke stop krijg je drie items?"),
            Mistake("four", matches=_shows("[7, 9, 4, 8]"),
                    en="Your program shows four scores. The second number of a slice is not included. Count again: which stop gives three items?",
                    nl="Je programma laat vier scores zien. Het tweede getal van een slice doet zelf niet mee. Tel nog eens: bij welke stop krijg je drie items?"),
            Mistake("from-one", matches=_shows("[9, 4, 8]", "[9, 4]"),
                    en="Your program starts at the second score, `9`. Python counts from 0: the first score is at index 0.",
                    nl="Je programma begint bij de tweede score, `9`. Python telt vanaf 0: de eerste score staat op index 0."),
            Mistake("last-three", matches=_shows("[4, 8, 6]"),
                    en="Your program shows the last three scores: a negative number counts from the end. The first three start at index 0.",
                    nl="Je programma laat de laatste drie scores zien: een negatief getal telt vanaf het eind. De eerste drie beginnen bij index 0."),
            Mistake("not-a-list", matches=lambda got, text: got.strip().startswith("("),
                    en="Your program shows the three scores between round brackets: that is not a list. One slice, `scores[start:stop]`, gives a list.",
                    nl="Je programma laat de drie scores tussen ronde haakjes zien: dat is geen lijst. Eén slice, `scores[start:stop]`, geeft een lijst."),
            Mistake("one-item", matches=_shows("7", "9", "4", "8", "6"),
                    en="Your program shows one score, not a piece of the list. A slice has a colon between square brackets: `scores[start:stop]`.",
                    nl="Je programma laat één score zien, geen stuk van de lijst. Een slice heeft een dubbele punt tussen vierkante haken: `scores[start:stop]`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a slice of `scores`: the first three.",
                      "nl": "Vervang de `...` door een slice van `scores`: de eerste drie."},
            "cleared": {"en": "There is no code in the box any more. Type the three lines back: `scores = [7, 9, 4, 8, 6]`, `top = ...` and `print(top)`.",
                        "nl": "Er staat geen code meer in het vak. Typ de drie regels terug: `scores = [7, 9, 4, 8, 6]`, `top = ...` en `print(top)`."},
            "wrong": {"en": "Your program shows:\n\n```\n{got}\n```\n\nThat is not yet `[7, 9, 4]`. Look at step 3 of the recipe: where does the slice start, and where does it stop?",
                      "nl": "Je programma laat dit zien:\n\n```\n{got}\n```\n\nDat is nog niet `[7, 9, 4]`. Kijk naar stap 3 van het recept: waar begint de slice, en waar stopt hij?"},
            "other": {"en": "Right for these scores. But with {where} instead, your program shows `{got}`, and it should show `{want}`. Did you type the numbers yourself, or count from the end? In every list, the first three are at index 0, 1 and 2.",
                      "nl": "Goed voor deze scores. Maar met {where} laat je programma `{got}` zien, en dat moet `{want}` zijn. Typte je de getallen zelf, of telde je vanaf het eind? In elke lijst staan de eerste drie op index 0, 1 en 2."},
            "correct": {"en": "Your program shows `[7, 9, 4]`, the first three scores, and it also works for another list.",
                        "nl": "Je programma laat `[7, 9, 4]` zien, de eerste drie scores, en het werkt ook voor een andere lijst."},
        },
    ),
    "cart": CodeBox(
        name="cart",
        start='cart = ["bread", "milk"]\n...\nprint(cart)',
        solution='cart = ["bread", "milk"]\ncart.append("apples")\nprint(cart)',
        outcome="stdout",
        expect="['bread', 'milk', 'apples']\n",
        compare=same_lines,
        others=[{"cart": ["eggs"]}],
        keep=['cart = ["bread", "milk"]'],
        mistakes=_DOTS + [
            _CURLY,
            Mistake("square-after-append", syntax=r"append\s*\[",
                    en="After `append` come round parentheses, with the new item inside, as in step 4 of the recipe.",
                    nl="Na `append` komen ronde haakjes, met het nieuwe item erin, zoals in stap 4 van het recept."),
            Mistake("no-quotes", syntax=r"append\(\s*apples\s*\)",
                    en="`apples` has no quotes, so Python looks for a name `apples`. Text goes between quotes: `\"apples\"`.",
                    nl="`apples` staat niet tussen aanhalingstekens, dus zoekt Python een naam `apples`. Tekst zet je tussen aanhalingstekens: `\"apples\"`."),
            Mistake("plus-text", syntax=r"cart\s*\+\s*[\"']",
                    en="A list and a string cannot be added with `+`. To add one item at the end, use `append`, as in step 4 of the recipe.",
                    nl="Een lijst en een string kun je niet optellen met `+`. Eén item achteraan toevoegen doe je met `append`, zoals in stap 4 van het recept."),
            Mistake("append-is-instruction", syntax=r"\.append\s*=",
                    en="`append` is an instruction of the list: you call it, with the new item between round parentheses, as in step 4 of the recipe.",
                    nl="`append` is een instructie van de lijst: je roept hem aan, met het nieuwe item tussen ronde haakjes, zoals in stap 4 van het recept."),
            Mistake("other-name", syntax=r"\.\s*(?!append\b)[A-Za-z_]\w*\s*\(",
                    en="A list has no instruction with that name. The one that adds at the end is `append`, in small letters: check the spelling.",
                    nl="Een lijst heeft geen instructie met die naam. De instructie die achteraan toevoegt, heet `append`, met kleine letters: kijk de spelling na."),
            Mistake("append-alone", syntax=r"^[ \t]*append\s*\(",
                    en="`append` belongs to a list: write the list's name and a dot in front of it, as in step 4 of the recipe.",
                    nl="`append` hoort bij een lijst: zet de naam van de lijst en een punt ervoor, zoals in stap 4 van het recept."),
            Mistake("new-place", syntax=r"cart\s*\[\s*\d+\s*\]\s*=",
                    en="`cart` has no place with that number yet, so you cannot fill it. To add an item at the end, use `append`.",
                    nl="`cart` heeft nog geen plek met dat nummer, dus die kun je niet vullen. Een item achteraan toevoegen doe je met `append`."),
            Mistake("assigned-none", matches=lambda got, text: "None" in [line.strip() for line in got.splitlines()],
                    en="Your program shows `None`, Python's word for nothing. `append` changes the list itself and gives nothing back, so do not use it inside `cart = ...` or `print(...)`. Write `cart.append(...)` on a line of its own.",
                    nl="Je programma laat `None` zien, het woord van Python voor niets. `append` verandert de lijst zelf en geeft niets terug, dus gebruik hem niet in `cart = ...` of `print(...)`. Schrijf `cart.append(...)` op een eigen regel."),
            Mistake("nested", matches=lambda got, text: "['apples']" in got,
                    en="Your cart now has a list inside it, `['apples']`. Between the parentheses of `append` goes the item itself, `\"apples\"`, without square brackets.",
                    nl="Je mandje heeft nu een lijst erin, `['apples']`. Tussen de haakjes van `append` komt het item zelf, `\"apples\"`, zonder vierkante haken."),
            Mistake("unchanged", matches=_shows("['bread', 'milk']"),
                    en="Your program shows the cart as it was: nothing changed `cart` before `print(cart)`. Add the item with `cart.append(...)`, on a line above the `print`.",
                    nl="Je programma laat het mandje zien zoals het was: niets veranderde `cart` vóór `print(cart)`. Voeg het item toe met `cart.append(...)`, op een regel boven de `print`."),
            Mistake("front", matches=_shows("['apples', 'bread', 'milk']"),
                    en="`\"apples\"` is at the front. `append` adds an item at the end of the list.",
                    nl="`\"apples\"` staat vooraan. `append` voegt een item achteraan de lijst toe."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a line that adds `\"apples\"` at the end of `cart`.",
                      "nl": "Vervang de `...` door een regel die `\"apples\"` achteraan `cart` toevoegt."},
            "cleared": {"en": "There is no code in the box any more. Type the three lines back: `cart = [\"bread\", \"milk\"]`, `...` and `print(cart)`.",
                        "nl": "Er staat geen code meer in het vak. Typ de drie regels terug: `cart = [\"bread\", \"milk\"]`, `...` en `print(cart)`."},
            "wrong": {"en": "Your program shows:\n\n```\n{got}\n```\n\nThat is not yet `['bread', 'milk', 'apples']`. Look at step 4 of the recipe, and check that the new item is exactly `\"apples\"`.",
                      "nl": "Je programma laat dit zien:\n\n```\n{got}\n```\n\nDat is nog niet `['bread', 'milk', 'apples']`. Kijk naar stap 4 van het recept, en kijk of het nieuwe item precies `\"apples\"` is."},
            "other": {"en": "Right for this cart. But with {where} instead, your program shows `{got}`, and it should show `{want}`. Add the item to `cart` with `append`; do not type the whole list yourself.",
                      "nl": "Goed voor dit mandje. Maar met {where} laat je programma `{got}` zien, en dat moet `{want}` zijn. Voeg het item met `append` aan `cart` toe; typ niet zelf de hele lijst."},
            "correct": {"en": "Your program shows `['bread', 'milk', 'apples']`: `\"apples\"` is now at the end, and it also works for another cart.",
                        "nl": "Je programma laat `['bread', 'milk', 'apples']` zien: `\"apples\"` staat nu achteraan, en het werkt ook voor een ander mandje."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_S = "scores = [7, 9, 4, 8, 6]\n"
_K = 'cart = ["bread", "milk"]\n'
CASES = [
    ("top", _S + "top = ...\nprint(top)", "empty", None),
    ("top", "", "empty", None),
    ("top", _S + "top = scores[:3]\nprint(top)", "correct", None),
    ("top", _S + "top = scores[0:3]\nprint(top)", "correct", None),
    ("top", _S + "top = [7, 9, 4]\nprint(top)", "almost", "other-value"),
    ("top", _S + "top = scores[:2]\nprint(top)", "wrong", "two"),
    ("top", _S + "top = scores[0:4]\nprint(top)", "wrong", "four"),
    ("top", _S + "top = scores[1:4]\nprint(top)", "wrong", "from-one"),
    ("top", _S + "top = scores[2]\nprint(top)", "wrong", "one-item"),
    ("top", _S + "top = scores(0:3)\nprint(top)", "error", "round-brackets"),
    ("top", _S + "top = scores[0, 3]\nprint(top)", "error", "comma"),
    ("top", _S + "top = scores[-3:]\nprint(top)", "wrong", "last-three"),
    ("top", _S + "top = scores[0], scores[1], scores[2]\nprint(top)", "wrong", "not-a-list"),
    ("top", _S + "top = scores(0, 3)\nprint(top)", "error", "round-brackets"),
    ("top", _S + "top = scores[:-2]\nprint(top)", "almost", "other-value"),
    ("cart", _K + "...\nprint(cart)", "empty", None),
    ("cart", _K + 'cart.append("apples")\nprint(cart)', "correct", None),
    ("cart", _K + 'cart = cart + ["apples"]\nprint(cart)', "correct", None),
    ("cart", _K + 'cart = ["bread", "milk", "apples"]\nprint(cart)', "almost", "other-value"),
    ("cart", _K + 'cart = cart.append("apples")\nprint(cart)', "wrong", "assigned-none"),
    ("cart", _K + 'cart.append(apples)\nprint(cart)', "error", "no-quotes"),
    ("cart", _K + 'cart.append["apples"]\nprint(cart)', "error", "square-after-append"),
    ("cart", _K + 'cart = cart + "apples"\nprint(cart)', "error", "plus-text"),
    ("cart", _K + 'cart.add("apples")\nprint(cart)', "error", "other-name"),
    ("cart", _K + 'cart = ["apples"] + cart\nprint(cart)', "wrong", "front"),
    ("cart", _K + 'cart.appen("apples")\nprint(cart)', "error", "other-name"),
    ("cart", _K + 'append("apples")\nprint(cart)', "error", "append-alone"),
    ("cart", _K + 'cart.append = "apples"\nprint(cart)', "error", "append-is-instruction"),
    ("cart", _K + 'cart[2] = "apples"\nprint(cart)', "error", "new-place"),
    ("cart", _K + 'cart.append(["apples"])\nprint(cart)', "wrong", "nested"),
    ("cart", _K + 'print(cart)\ncart.append("apples")', "wrong", "unchanged"),
    ("cart", _K + 'print(cart.append("apples"))', "wrong", "assigned-none"),
    ("cart", _K + "...\nprint(cart)\n", "empty", None),
]
