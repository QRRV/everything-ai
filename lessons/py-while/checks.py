# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson py-while, and the loop in its picture, check by check."""
import re

from everything_ai.check import CodeBox, Mistake

# The loop in the picture: n doubles while n < 20. For each check: what it says, and n after the round.
PICTURE_LIMIT = 20


def _checks(limit):
    n, lines, names, shown = 1, {"en": [], "nl": []}, [], []
    while True:
        go = n < limit
        lines["en"].append(f"{n} < {limit} is {go}" + ("" if go else ": stop"))
        lines["nl"].append(f"{n} < {limit} is {go}" + ("" if go else ": stop"))
        if not go:
            names.append({"n": n})
            shown.append(f"{n}\n")
            break
        n = n * 2
        names.append({"n": n})
        shown.append("")
    return lines, names, shown


PICTURE_LINES, PICTURE_NAMES, PICTURE_SHOWN = _checks(PICTURE_LIMIT)


def _lines(text) -> list[str]:
    return [line.rstrip() for line in str(text).rstrip("\n").splitlines()]


def same_lines(a, b) -> bool:
    return isinstance(a, str) and isinstance(b, str) and _lines(a) == _lines(b)


def _shows(*texts):
    return lambda got, text: _lines(got) in [[str(t)] for t in texts]


def _print_inside(got, text):
    return len(_lines(got)) > 1 and re.search(r"^[ \t]+print\(", text, re.MULTILINE) is not None


_SYNTAX = [
    Mistake("dots-left", syntax=r"^[^#\n]*\.\.\.",
            en="There is still a `...` in your program. Replace every `...` with your own code.",
            nl="Er staat nog een `...` in je programma. Vervang elke `...` door je eigen code."),
    Mistake("curly-quote", syntax=r"[“”‘’„]",
            en="Your program has a curly quote, like `“`. Python only reads straight quotes, like `\"`: type the quotes again in the box.",
            nl="Je programma heeft een gekruld aanhalingsteken, zoals `“`. Python leest alleen rechte, zoals `\"`: typ de aanhalingstekens opnieuw in het vak."),
    Mistake("colon-missing", syntax=r"^[ \t]*(while\b[^:\n]*|for\b[^:\n]*|if\b[^:\n]*|else[ \t]*)$",
            en="The line `{line}` must end with a colon, `:`. The colon tells Python that the indented lines below belong to it.",
            nl="De regel `{line}` moet eindigen op een dubbele punt, `:`. Die vertelt Python dat de ingesprongen regels eronder erbij horen."),
    Mistake("capital", syntax=r"^[ \t]*(While|For|If|Else|Break)\b",
            en="Python writes `while`, `for`, `if`, `else` and `break` in small letters. On the line `{line}`, change the capital.",
            nl="Python schrijft `while`, `for`, `if`, `else` en `break` met kleine letters. Verander op de regel `{line}` de hoofdletter."),
]
_DOTS_RAN = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                    en="There is still a `...` in your program. Replace every `...` with your own code.",
                    nl="Er staat nog een `...` in je programma. Vervang elke `...` door je eigen code.")
_PRINT_INSIDE = Mistake("print-inside", matches=_print_inside,
                        en="Your program shows more than one line: a `print` is indented, so it runs inside the loop. Keep one `print`, at the left edge, after the loop.",
                        nl="Je programma laat meer dan één regel zien: een `print` springt in, dus hij draait in de loop. Houd één `print` over, helemaal links, na de loop.")

# The lesson is done when these are correct. "break" is the choice question in the lesson.
REQUIRED = ["double", "break", "first-long"]

EXERCISES = {
    "double": CodeBox(
        name="double",
        start="target = 1000\ncount = 1\nhours = 0\nwhile ...:\n    ...\nprint(hours)",
        solution="target = 1000\ncount = 1\nhours = 0\nwhile count < target:\n    count = count * 2\n    hours = hours + 1\nprint(hours)",
        outcome="stdout",
        compare=same_lines,
        others=[{"target": 8}, {"target": 1}],
        keep=["target = 1000", "count = 1", "hours = 0"],
        first=[_DOTS_RAN],
        mistakes=_SYNTAX + [
            _DOTS_RAN, _PRINT_INSIDE,
            Mistake("once", matches=_shows(1),
                    en="Your program shows 1: the round ran only once. `if` runs its lines once; `while` repeats them as long as its condition is `True`. And `hours = hours + 1` belongs inside the loop.",
                    nl="Je programma laat 1 zien: de ronde draaide maar één keer. `if` voert zijn regels één keer uit; `while` herhaalt ze zolang de voorwaarde `True` is. En `hours = hours + 1` hoort in de loop."),
            Mistake("count-shown", matches=_shows(1024),
                    en="Your program shows 1024: that is `count`, the number of bacteria. The task asks for the number of hours: show `hours`.",
                    nl="Je programma laat 1024 zien: dat is `count`, het aantal bacteriën. De opdracht vraagt het aantal uren: laat `hours` zien."),
            Mistake("zero", matches=_shows(0),
                    en="Your program shows 0. Either the loop never ran, because its condition was `False` from the start, or `hours` did not grow. The loop should go on while `count` is still below `target`, and add 1 to `hours` each round.",
                    nl="Je programma laat 0 zien. Of de loop draaide nooit, omdat de voorwaarde meteen `False` was, of `hours` groeide niet. De loop moet doorgaan zolang `count` nog onder `target` is, en elke ronde 1 bij `hours` optellen."),
            Mistake("plus-two", matches=_shows(500),
                    en="Your program shows 500: `count` grows by 2 each round. Doubling is times 2, `* 2`.",
                    nl="Je programma laat 500 zien: `count` groeit elke ronde met 2. Verdubbelen is keer 2, `* 2`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: the condition after `while`, and what happens in one round.",
                      "nl": "Vervang de twee `...`: de voorwaarde na `while`, en wat er in één ronde gebeurt."},
            "cleared": {"en": "There is no code in the box any more. Type `target = 1000`, `count = 1` and `hours = 0` back at the top, then your loop, and `print(hours)` at the bottom.",
                        "nl": "Er staat geen code meer in het vak. Typ `target = 1000`, `count = 1` en `hours = 0` terug bovenaan, dan je loop, en `print(hours)` onderaan."},
            "slow": {"en": "Your loop does not stop: after 2 seconds, its condition is still `True`. Does each round double `count`, and does the condition become `False` once `count` is at least `target`? Look at steps 2 and 3 of the recipe.",
                     "nl": "Je loop stopt niet: na 2 seconden is de voorwaarde nog steeds `True`. Verdubbelt elke ronde `count`, en wordt de voorwaarde `False` zodra `count` minstens `target` is? Kijk naar stap 2 en 3 van het recept."},
            "wrong": {"en": "Your program shows:\n\n```\n{got}\n```\n\nThat is not yet the number of hours. Look at steps 2 and 3 of the recipe: when should the loop go on, and what changes in each round?",
                      "nl": "Je programma laat dit zien:\n\n```\n{got}\n```\n\nDat is nog niet het aantal uren. Kijk naar stap 2 en 3 van het recept: wanneer moet de loop doorgaan, en wat verandert er elke ronde?"},
            "other": {"en": "Right for 1000. But with {where} instead, your program shows `{got}`, and it should show `{want}`. Does the loop stop at the first count of at least `target`? With `<=`, it goes one round too far when `count` is exactly `target`; and a number of hours you typed yourself does not change with `target`.",
                      "nl": "Goed bij 1000. Maar met {where} laat je programma `{got}` zien, en het zou `{want}` moeten zijn. Stopt de loop bij het eerste aantal van minstens `target`? Met `<=` gaat hij één ronde te ver als `count` precies `target` is; en een aantal uren dat je zelf typte, verandert niet mee met `target`."},
            "correct": {"en": "Your program shows 10: after 10 hours there are 1024, the first count of at least 1000. It also works for other targets.",
                        "nl": "Je programma laat 10 zien: na 10 uur zijn het er 1024, het eerste aantal van minstens 1000. Het werkt ook voor andere doelen."},
        },
    ),
    "first-long": CodeBox(
        name="first-long",
        start='words = ["sun", "rain", "thunder", "fog", "drizzle"]\nfirst_long = "none"\nfor word in words:\n    ...\nprint(first_long)',
        solution='words = ["sun", "rain", "thunder", "fog", "drizzle"]\nfirst_long = "none"\nfor word in words:\n    if len(word) > 5:\n        first_long = word\n        break\nprint(first_long)',
        outcome="stdout",
        compare=same_lines,
        others=[{"words": ["hello", "planet", "galaxy"]}, {"words": ["hi", "yo"]}],
        keep=['words = ["sun", "rain", "thunder", "fog", "drizzle"]', 'first_long = "none"'],
        first=[_DOTS_RAN],
        mistakes=_SYNTAX + [
            Mistake("word-compared", syntax=r"\bword\s*[<>]",
                    en="You compare the word itself with 5. Compare its number of characters, `len(word)`, with 5.",
                    nl="Je vergelijkt het woord zelf met 5. Vergelijk het aantal tekens, `len(word)`, met 5."),
            Mistake("break-outside", syntax=r"^break\b",
                    en="`break` is at the left edge, outside the loop. Indent it under the `if`, after `first_long = word`.",
                    nl="`break` staat helemaal links, buiten de loop. Laat hem inspringen onder de `if`, na `first_long = word`."),
            _DOTS_RAN, _PRINT_INSIDE,
            Mistake("no-break", matches=_shows("drizzle"),
                    en="Your program shows `drizzle`, the last long word. Without `break`, the loop goes on after the first long word, and `first_long` gets every next one too. Stop the loop with `break`, right after the first.",
                    nl="Je programma laat `drizzle` zien, het laatste lange woord. Zonder `break` gaat de loop na het eerste lange woord door, en krijgt `first_long` ook elk volgend. Stop de loop met `break`, direct na het eerste."),
            Mistake("first-word", matches=_shows("sun"),
                    en="Your program shows `sun`, the first word: your loop stopped in the first round. `break` should only run for a word with more than 5 letters: is it inside the `if`, and is the `if` `False` for `sun`?",
                    nl="Je programma laat `sun` zien, het eerste woord: je loop stopte in de eerste ronde. `break` moet alleen draaien bij een woord met meer dan 5 letters: staat hij in de `if`, en is de `if` `False` bij `sun`?"),
            Mistake("never-set", matches=_shows("none"),
                    en="Your program shows `none`: `first_long` never got a word. Either the loop stopped before that line, or your `if` was never `True`. Inside the `if`, put `first_long = word` first and `break` after it, and check that the condition is `True` for `thunder`.",
                    nl="Je programma laat `none` zien: `first_long` kreeg nooit een woord. Of de loop stopte vóór die regel, of je `if` was nooit `True`. Zet in de `if` eerst `first_long = word` en daarna `break`, en kijk of de voorwaarde `True` is bij `thunder`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with an `if` for a long word, and inside it the line for `first_long` and a `break`.",
                      "nl": "Vervang de `...` door een `if` voor een lang woord, met daarin de regel voor `first_long` en een `break`."},
            "cleared": {"en": "There is no code in the box any more. Type the lines back: `words = [...]` with the five words, `first_long = \"none\"`, `for word in words:`, your `if`, and `print(first_long)`.",
                        "nl": "Er staat geen code meer in het vak. Typ de regels terug: `words = [...]` met de vijf woorden, `first_long = \"none\"`, `for word in words:`, je `if`, en `print(first_long)`."},
            "wrong": {"en": "Your program shows:\n\n```\n{got}\n```\n\nThe first word with more than 5 letters is `thunder`. Which condition is `True` for a long word, and when should the loop stop?",
                      "nl": "Je programma laat dit zien:\n\n```\n{got}\n```\n\nHet eerste woord met meer dan 5 letters is `thunder`. Welke voorwaarde is `True` bij een lang woord, en wanneer moet de loop stoppen?"},
            "other": {"en": "Right for these words. But with {where} instead, your program shows `{got}`, and it should show `{want}`. Is your condition `True` for exactly the words with more than 5 letters, and did you let the loop find the word instead of typing it?",
                      "nl": "Goed voor deze woorden. Maar met {where} laat je programma `{got}` zien, en het zou `{want}` moeten zijn. Is je voorwaarde `True` bij precies de woorden met meer dan 5 letters, en liet je de loop het woord zoeken in plaats van het zelf te typen?"},
            "other_error": {"en": "Right for these words. But with {where} instead, your program stops with an error. Let the loop go through `words` itself, so it works for every list.",
                            "nl": "Goed voor deze woorden. Maar met {where} stopt je programma met een fout. Laat de loop `words` zelf langsgaan, dan werkt het voor elke lijst."},
            "correct": {"en": "Your program shows `thunder`, the first word with more than 5 letters, and it also works for other lists.",
                        "nl": "Je programma laat `thunder` zien, het eerste woord met meer dan 5 letters, en het werkt ook voor andere lijsten."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_D = "target = 1000\ncount = 1\nhours = 0\n"
_W = 'words = ["sun", "rain", "thunder", "fog", "drizzle"]\nfirst_long = "none"\n'
CASES = [
    ("double", _D + "while ...:\n    ...\nprint(hours)", "empty", None),
    ("double", "", "empty", None),
    ("double", _D + "while count < target:\n    count = count * 2\n    hours = hours + 1\nprint(hours)", "correct", None),
    ("double", _D + "while count < target:\n  hours = hours + 1\n  count = count * 2\nprint(hours)", "correct", None),
    ("double", _D + "while count <= target:\n    count = count * 2\n    hours = hours + 1\nprint(hours)", "almost", "other-value"),
    ("double", _D + "hours = 10\nprint(hours)", "almost", "other-value"),
    ("double", _D + "while count < target:\n    count = count * 2\nprint(hours)", "wrong", "zero"),
    ("double", _D + "while count > target:\n    count = count * 2\n    hours = hours + 1\nprint(hours)", "wrong", "zero"),
    ("double", _D + "while count < target:\n    count = count + 2\n    hours = hours + 1\nprint(hours)", "wrong", "plus-two"),
    ("double", _D + "while count < target:\n    hours = hours + 1\nprint(hours)", "slow", None),
    ("double", _D + "while count < target:\n    count = count * 2\n    hours = hours + 1\n    print(hours)", "wrong", "print-inside"),
    ("double", _D + "while count < target\n    count = count * 2\n    hours = hours + 1\nprint(hours)", "error", "colon-missing"),
    ("first-long", _W + "for word in words:\n    ...\nprint(first_long)", "empty", None),
    ("first-long", _W + "for word in words:\n    if len(word) > 5:\n        first_long = word\n        break\nprint(first_long)", "correct", None),
    ("first-long", _W + "for word in words:\n    if len(word) > 5:\n        first_long = word\nprint(first_long)", "wrong", "no-break"),
    ("first-long", _W + "for word in words:\n    if len(word) > 5:\n        first_long = word\n    break\nprint(first_long)", "wrong", "never-set"),
    ("first-long", _W + "for word in words:\n    if len(words) > 5:\n        first_long = word\n        break\nprint(first_long)", "wrong", "never-set"),
    ("first-long", _W + "for word in words:\n    if len(word) < 5:\n        first_long = word\n        break\nprint(first_long)", "wrong", "first-word"),
    ("first-long", _W + "for word in words:\n    if word > 5:\n        first_long = word\n        break\nprint(first_long)", "error", "word-compared"),
    ("double", _D + "while count != target:\n    count = count * 2\n    hours = hours + 1\nprint(hours)", "slow", None),
    ("double", _D + "while ...:\n    count = count * 2\n    hours = hours + 1\nprint(hours)", "wrong", "dots-left"),
    ("double", _D + "if count < target:\n    count = count * 2\n    hours = hours + 1\nprint(hours)", "wrong", "once"),
    ("double", _D + "while count < target:\n    count = count * 2\n    hours = hours + 1\nprint(count)", "wrong", "count-shown"),
    ("first-long", _W + "for word in words:\n    if len(word) >= 5:\n        first_long = word\n        break\nprint(first_long)", "almost", "other-value"),
    ("first-long", _W + 'first_long = "thunder"\nprint(first_long)', "almost", "other-value"),
]
