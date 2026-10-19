# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson py-tuples-sets: each word once and in order, and the fastest three times. And the rows in the picture."""
import ast
import re

from everything_ai.check import CodeBox, Mistake

# The picture: the same numbers as a list, a tuple, a set, and sorted.
PICTURE = [3, 1, 3, 2]


def picture_rows(notes: dict[str, str]) -> list[tuple[str, list[str], str, bool]]:
    nums = PICTURE
    return [
        (f"nums = {nums}", [str(x) for x in nums], notes["list"], True),
        (f"fixed = {tuple(nums)}", [str(x) for x in nums], notes["tuple"], True),
        ("set(nums)", [str(x) for x in sorted(set(nums))], notes["set"], False),
        ("sorted(nums)", [str(x) for x in sorted(nums)], notes["sorted"], True),
    ]


def _lines(text) -> list[str]:
    return [line.rstrip() for line in str(text).rstrip("\n").splitlines()]


def same_lines(a, b) -> bool:
    return isinstance(a, str) and isinstance(b, str) and _lines(a) == _lines(b)


def _shows(*texts):
    """The program showed exactly one of these lines."""
    return lambda got, text: _lines(got) in [[t] for t in texts]


_DOTS = [Mistake("dots-left", syntax=r"^[^#\n]*\.\.\.",
                 en="There is still a `...` in your program. Replace it with your own code.",
                 nl="Er staat nog een `...` in je programma. Vervang die door je eigen code."),
         Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                 en="There is still a `...` in your program. Replace it with your own code.",
                 nl="Er staat nog een `...` in je programma. Vervang die door je eigen code.")]
_CURLY = Mistake("curly-quote", syntax=r"[“”‘’„]",
                 en="Your program has a curly quote, like `“`. Python only reads straight quotes, like `\"`: type the quotes again in the box.",
                 nl="Je programma heeft een gekruld aanhalingsteken, zoals `“`. Python leest alleen rechte, zoals `\"`: typ de aanhalingstekens opnieuw in het vak.")
def _as_list(text):
    try:
        value = ast.literal_eval(text.strip())
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, list) else None


_SORT_ERROR = Mistake("sort-none-error", syntax=r"\.sort\(\s*\)",
                      en="`.sort()` puts the list itself in order and gives back nothing: `None`, Python's word for nothing. You cannot slice or sort `None`. Use `sorted(...)`: it gives a new list.",
                      nl="`.sort()` zet de lijst zelf op volgorde en geeft niets terug: `None`, het woord van Python voor niets. `None` kun je niet slicen of sorteren. Gebruik `sorted(...)`: dat geeft een nieuwe lijst.")
_BRACKETS = Mistake("square-call", syntax=r"\b(sorted|set)\s*\[",
                    en="`set` and `sorted` take round brackets, like `len`: `sorted(set(words))`.",
                    nl="`set` en `sorted` krijgen ronde haakjes, net als `len`: `sorted(set(words))`.")
_TUPLE_OUT = Mistake("tuple-out", matches=lambda got, text: got.strip().startswith("("),
                     en="The right kind of values, but between parentheses: a tuple. The task asks for a list, with square brackets: `sorted(...)` gives one.",
                     nl="De goede soort waarden, maar tussen ronde haakjes: een tuple. De opdracht vraagt om een lijst, met vierkante haken: `sorted(...)` geeft er een.")
_SORT_NONE = Mistake("sort-none", matches=_shows("None"),
                     en="Your program shows `None`, Python's word for nothing. `.sort()` puts the list itself in order and gives nothing back. `sorted(...)` gives a new list, in order: use that one.",
                     nl="Je programma laat `None` zien, het woord van Python voor niets. `.sort()` zet de lijst zelf op volgorde en geeft niets terug. `sorted(...)` geeft een nieuwe lijst, op volgorde: gebruik die.")

# The lesson is done when these are correct. "tuple-change" is the choice question in the lesson.
REQUIRED = ["unique", "tuple-change", "podium"]

EXERCISES = {
    "unique": CodeBox(
        name="unique",
        start='words = ["rain", "sun", "rain", "fog", "sun"]\nunique = ...\nprint(unique)',
        solution='words = ["rain", "sun", "rain", "fog", "sun"]\nunique = sorted(set(words))\nprint(unique)',
        outcome="stdout",
        expect="['fog', 'rain', 'sun']\n",
        compare=same_lines,
        others=[{"words": ["b", "a", "b", "c"]}, {"words": ["pear", "fig", "kiwi", "apple", "fig", "date", "lime"]}],
        keep=['words = ["rain", "sun", "rain", "fog", "sun"]'],
        mistakes=_DOTS + [
            _CURLY,
            Mistake("set-index", syntax=r"set\([^\n]*\)\s*\[",
                    en="A set has no places, so it has no index. To get a list, in order, use `sorted(...)` around the set.",
                    nl="Een set heeft geen plekken, dus ook geen index. Wil je een lijst, op volgorde, gebruik dan `sorted(...)` om de set."),
            Mistake("set-sort", syntax=r"set\([^\n]*\)\s*\.\s*sort\b",
                    en="A set has no `.sort()`: it has no order to keep. `sorted(...)` around the set gives a list, in order.",
                    nl="Een set heeft geen `.sort()`: hij heeft geen volgorde om te bewaren. `sorted(...)` om de set geeft een lijst, op volgorde."),
            _SORT_ERROR, _BRACKETS,
            Mistake("braces", syntax=r"\{\s*words\s*\}",
                    en="Curly braces around `words` try to make a set with the whole list as one value, and a list cannot go into a set. `set(words)` makes a set of the items in the list.",
                    nl="Accolades om `words` proberen een set te maken met de hele lijst als één waarde, en een lijst kan niet in een set. `set(words)` maakt een set van de items in de lijst."),
            Mistake("sorted-then-set", matches=lambda got, text: re.search(r"set\(\s*sorted\(", text) is not None,
                    en="You sort first and then make a set, and a set loses the order again. Make the set first, then sort it.",
                    nl="Je sorteert eerst en maakt dan een set, en een set raakt de volgorde weer kwijt. Maak eerst de set, en sorteer die dan."),
            Mistake("square-call-out", matches=lambda got, text: got.strip().startswith(("set[", "sorted[")),
                    en="`set` and `sorted` take round brackets, like `len`: `sorted(set(words))`.",
                    nl="`set` en `sorted` krijgen ronde haakjes, net als `len`: `sorted(set(words))`."),
            Mistake("set-only", matches=lambda got, text: got.strip().startswith("{"),
                    en="Your program shows a set, between curly braces: each word once, but in no fixed order, so it can differ each time. Put it in order with `sorted(...)`, which gives a list.",
                    nl="Je programma laat een set zien, tussen accolades: elk woord één keer, maar zonder vaste volgorde, dus die kan elke keer anders zijn. Zet hem op volgorde met `sorted(...)`, dat geeft een lijst."),
            Mistake("sorted-only", matches=_shows("['fog', 'rain', 'rain', 'sun', 'sun']"),
                    en="Your program shows the words in order, but `rain` and `sun` are there twice. A set keeps each value once: make a set of `words` first, then sort it.",
                    nl="Je programma laat de woorden op volgorde zien, maar `rain` en `sun` staan er twee keer in. In een set staat elke waarde maar één keer: maak eerst een set van `words`, en sorteer die dan."),
            Mistake("not-sorted", matches=lambda got, text: (_as_list(got) is not None and sorted(_as_list(got)) == ["fog", "rain", "sun"]
                                                             and _as_list(got) != ["fog", "rain", "sun"]),
                    en="Each word is there once, but not from A to Z: a set has no fixed order. Put `sorted(...)` around the set.",
                    nl="Elk woord staat er één keer in, maar niet van A tot Z: een set heeft geen vaste volgorde. Zet `sorted(...)` om de set."),
            _TUPLE_OUT,
            Mistake("count", matches=_shows("3"),
                    en="Your program shows `3`: how many different words there are. The task asks for the words themselves, each once and in order: leave out `len`.",
                    nl="Je programma laat `3` zien: hoeveel verschillende woorden er zijn. De opdracht vraagt om de woorden zelf, elk één keer en op volgorde: laat `len` weg."),
            _SORT_NONE,
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with each word of `words` once, in order: a set, then `sorted`.",
                      "nl": "Vervang de `...` door elk woord uit `words` één keer, op volgorde: een set, en dan `sorted`."},
            "cleared": {"en": "There is no code in the box any more. Type the three lines back: the line with `words`, `unique = ...` and `print(unique)`.",
                        "nl": "Er staat geen code meer in het vak. Typ de drie regels terug: de regel met `words`, `unique = ...` en `print(unique)`."},
            "wrong": {"en": "Your program shows:\n\n```\n{got}\n```\n\nThat is not yet `['fog', 'rain', 'sun']`. Look at steps 2 and 3 of the recipe: first each word once, then in order.",
                      "nl": "Je programma laat dit zien:\n\n```\n{got}\n```\n\nDat is nog niet `['fog', 'rain', 'sun']`. Kijk naar stap 2 en 3 van het recept: eerst elk woord één keer, dan op volgorde."},
            "other": {"en": "Right for these words. But with {where} instead, your program shows `{got}`, and it should show `{want}`. Does your code depend on this list, such as words you typed yourself? And a set has no fixed order: sort it with `sorted(...)`.",
                      "nl": "Goed voor deze woorden. Maar met {where} laat je programma `{got}` zien, en dat moet `{want}` zijn. Hangt je code af van deze lijst, bijvoorbeeld door woorden die je zelf typte? En een set heeft geen vaste volgorde: sorteer hem met `sorted(...)`."},
            "other_error": {"en": "Right for these words. But with {where} instead, your program stops with an error. Does your code depend on this list, such as fixed places? Make the list from `set(words)` and `sorted(...)`.",
                            "nl": "Goed voor deze woorden. Maar met {where} stopt je programma met een fout. Hangt je code af van deze lijst, bijvoorbeeld door vaste plekken? Maak de lijst met `set(words)` en `sorted(...)`."},
            "correct": {"en": "Your program shows `['fog', 'rain', 'sun']`: each word once, from A to Z. It also works for another list.",
                        "nl": "Je programma laat `['fog', 'rain', 'sun']` zien: elk woord één keer, van A tot Z. Het werkt ook voor een andere lijst."},
        },
    ),
    "podium": CodeBox(
        name="podium",
        start="times = [12.4, 11.8, 13.1, 11.9, 12.0]\npodium = ...\nprint(podium)",
        solution="times = [12.4, 11.8, 13.1, 11.9, 12.0]\npodium = sorted(times)[:3]\nprint(podium)",
        outcome="stdout",
        expect="[11.8, 11.9, 12.0]\n",
        compare=same_lines,
        others=[{"times": [9.9, 10.5, 9.7, 10.1]}],
        keep=["times = [12.4, 11.8, 13.1, 11.9, 12.0]"],
        mistakes=_DOTS + [
            _CURLY, _SORT_ERROR, _BRACKETS,
            Mistake("comma-slice", syntax=r"\)\s*\[\s*-?\d+\s*,",
                    en="A slice has a colon between its numbers, not a comma: `[start:stop]`.",
                    nl="Een slice heeft een dubbele punt tussen zijn getallen, geen komma: `[start:stop]`."),
            Mistake("slowest", matches=_shows("[12.0, 12.4, 13.1]", "[13.1, 12.4, 12.0]"),
                    en="Your program shows the three slowest times. In a race, the lowest time is the fastest, and `sorted` puts the lowest first. Take the slice from the start of the sorted list.",
                    nl="Je programma laat de drie langzaamste tijden zien. Bij een wedstrijd is de laagste tijd de snelste, en `sorted` zet de laagste vooraan. Pak de slice vanaf het begin van de gesorteerde lijst."),
            Mistake("fastest-last", matches=_shows("[12.0, 11.9, 11.8]"),
                    en="The three fastest times, but the fastest is last. `sorted(...)` without anything else puts the lowest first: then take the slice from the start.",
                    nl="De drie snelste tijden, maar de snelste staat achteraan. `sorted(...)` zonder iets erbij zet de laagste vooraan: pak dan de slice vanaf het begin."),
            Mistake("from-one", matches=_shows("[11.9, 12.0, 12.4]", "[11.9, 12.0]"),
                    en="11.8, the fastest, is missing: your slice starts at index 1. Python counts from 0, so the first time is at index 0.",
                    nl="11.8, de snelste, ontbreekt: je slice begint bij index 1. Python telt vanaf 0, dus de eerste tijd staat op index 0."),
            _TUPLE_OUT,
            Mistake("not-sorted", matches=_shows("[12.4, 11.8, 13.1]"),
                    en="Your program shows the first three times of the list, in the order they were run. Sort the list first, then take the first three.",
                    nl="Je programma laat de eerste drie tijden van de lijst zien, in de volgorde waarin ze gelopen zijn. Sorteer de lijst eerst, en pak dan de eerste drie."),
            Mistake("sorted-slice", matches=_shows("[11.8, 12.4, 13.1]"),
                    en="Your program sorts only the first three times. Sort the whole list first, and then take a slice: the slice comes after `sorted(...)`.",
                    nl="Je programma sorteert alleen de eerste drie tijden. Sorteer eerst de hele lijst, en pak daarna een slice: de slice komt na `sorted(...)`."),
            Mistake("two", matches=_shows("[11.8, 11.9]"),
                    en="Your program shows only two of the times. A slice stops just before its second number: which stop gives three?",
                    nl="Je programma laat maar twee van de tijden zien. Een slice stopt net voor zijn tweede getal: welk tweede getal geeft er drie?"),
            Mistake("whole", matches=_shows("[11.8, 11.9, 12.0, 12.4, 13.1]"),
                    en="Your program shows all times, in order. The podium has three places: take the first three with a slice.",
                    nl="Je programma laat alle tijden zien, op volgorde. Het podium heeft drie plekken: pak de eerste drie met een slice."),
            Mistake("one", matches=lambda got, text: re.fullmatch(r"-?\d+(\.\d+)?", got.strip()) is not None,
                    en="Your program shows a single time. The podium has three: a slice with a colon gives a piece of the list.",
                    nl="Je programma laat één enkele tijd zien. Het podium heeft er drie: een slice met een dubbele punt geeft een stuk van de lijst."),
            _SORT_NONE,
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the three lowest times, in order: sort, then take a slice.",
                      "nl": "Vervang de `...` door de drie laagste tijden, op volgorde: sorteer, en pak dan een slice."},
            "cleared": {"en": "There is no code in the box any more. Type the three lines back: the line with `times`, `podium = ...` and `print(podium)`.",
                        "nl": "Er staat geen code meer in het vak. Typ de drie regels terug: de regel met `times`, `podium = ...` en `print(podium)`."},
            "wrong": {"en": "Your program shows:\n\n```\n{got}\n```\n\nThat is not yet `[11.8, 11.9, 12.0]`. Look at step 4 of the recipe: sort first, then take a slice.",
                      "nl": "Je programma laat dit zien:\n\n```\n{got}\n```\n\nDat is nog niet `[11.8, 11.9, 12.0]`. Kijk naar stap 4 van het recept: eerst sorteren, dan een slice."},
            "other": {"en": "Right for these times. But with {where} instead, your program shows `{got}`, and it should show `{want}`. Does your code depend on this list, such as times you typed or fixed places? Sort `times`, then take the first three.",
                      "nl": "Goed voor deze tijden. Maar met {where} laat je programma `{got}` zien, en dat moet `{want}` zijn. Hangt je code af van deze lijst, bijvoorbeeld door tijden die je typte of vaste plekken? Sorteer `times`, en pak dan de eerste drie."},
            "other_error": {"en": "Right for these times. But with {where} instead, your program stops with an error. Does your code depend on this list, such as fixed places? Sort `times`, then take the first three.",
                            "nl": "Goed voor deze tijden. Maar met {where} stopt je programma met een fout. Hangt je code af van deze lijst, bijvoorbeeld door vaste plekken? Sorteer `times`, en pak dan de eerste drie."},
            "correct": {"en": "Your program shows `[11.8, 11.9, 12.0]`, the three fastest times, and it also works for another race.",
                        "nl": "Je programma laat `[11.8, 11.9, 12.0]` zien, de drie snelste tijden, en het werkt ook voor een andere wedstrijd."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_W = 'words = ["rain", "sun", "rain", "fog", "sun"]\n'
_T = "times = [12.4, 11.8, 13.1, 11.9, 12.0]\n"
CASES = [
    ("unique", _W + "unique = ...\nprint(unique)", "empty", None),
    ("unique", "", "empty", None),
    ("unique", _W + "unique = sorted(set(words))\nprint(unique)", "correct", None),
    ("unique", _W + "unique = set(words)\nunique = sorted(unique)\nprint(unique)", "correct", None),
    ("unique", _W + "unique = set(words)\nprint(unique)", "wrong", "set-only"),
    ("unique", _W + "unique = sorted(words)\nprint(unique)", "wrong", "sorted-only"),
    ("unique", _W + "unique = len(set(words))\nprint(unique)", "wrong", "count"),
    ("unique", _W + "unique = {words}\nprint(unique)", "error", "braces"),
    ("unique", _W + "unique = set(sorted(words))\nprint(unique)", "wrong", "sorted-then-set"),
    ("unique", _W + "unique = sorted(words.sort())\nprint(unique)", "error", "sort-none-error"),
    ("unique", _W + "unique = set(words)[0]\nprint(unique)", "error", "set-index"),
    ("unique", _W + "unique = set(words).sort()\nprint(unique)", "error", "set-sort"),
    ("unique", _W + "unique = sorted[set(words)]\nprint(unique)", "error", "square-call"),
    ("unique", _W + "unique = set[words]\nprint(unique)", "wrong", "square-call-out"),
    ("unique", _W + "unique = tuple(sorted(set(words)))\nprint(unique)", "wrong", "tuple-out"),
    ("unique", _W + 'unique = ["sun", "rain", "fog"]\nprint(unique)', "wrong", "not-sorted"),
    ("unique", _W + "unique = words.sort()\nprint(unique)", "wrong", "sort-none"),
    ("unique", _W + 'unique = ["fog", "rain", "sun"]\nprint(unique)', "almost", "other-value"),
    ("podium", _T + "podium = ...\nprint(podium)", "empty", None),
    ("podium", _T + "podium = sorted(times)[:3]\nprint(podium)", "correct", None),
    ("podium", _T + "podium = sorted(times)[0:3]\nprint(podium)", "correct", None),
    ("podium", _T + "times = sorted(times)\npodium = times[:3]\nprint(podium)", "correct", None),
    ("podium", _T + "podium = sorted(times)[-3:]\nprint(podium)", "wrong", "slowest"),
    ("podium", _T + "podium = times[:3]\nprint(podium)", "wrong", "not-sorted"),
    ("podium", _T + "podium = sorted(times[:3])\nprint(podium)", "wrong", "sorted-slice"),
    ("podium", _T + "podium = sorted(times)[:2]\nprint(podium)", "wrong", "two"),
    ("podium", _T + "podium = sorted(times)\nprint(podium)", "wrong", "whole"),
    ("podium", _T + "podium = sorted(times)[0]\nprint(podium)", "wrong", "one"),
    ("podium", _T + "podium = times.sort()\nprint(podium)", "wrong", "sort-none"),
    ("podium", _T + "podium = times.sort()[:3]\nprint(podium)", "error", "sort-none-error"),
    ("podium", _T + "podium = sorted(times)[0, 3]\nprint(podium)", "error", "comma-slice"),
    ("podium", _T + "podium = sorted(times)[1:4]\nprint(podium)", "wrong", "from-one"),
    ("podium", _T + "podium = sorted(times)[2]\nprint(podium)", "wrong", "one"),
    ("podium", _T + "podium = sorted(times, reverse=True)[-3:]\nprint(podium)", "wrong", "fastest-last"),
    ("podium", _T + "podium = (sorted(times)[0], sorted(times)[1], sorted(times)[2])\nprint(podium)", "wrong", "tuple-out"),
    ("podium", _T + "times.sort()\npodium = times[:3]\nprint(podium)", "correct", None),
    ("podium", _T + "podium = [11.8, 11.9, 12.0]\nprint(podium)", "almost", "other-value"),
]
