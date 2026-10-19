# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson py-first-program, and the program in its picture."""
import re

from everything_ai.check import CodeBox, Mistake, run_code

# The program in the picture, per language, and what each line adds under the cell: Python works that out.
STEPS = {
    "en": ['print("Good morning")', 'print("Coffee first")', 'print("Then work")'],
    "nl": ['print("Goedemorgen")', 'print("Eerst koffie")', 'print("Dan aan het werk")'],
}
SHOWN = {lang: [run_code(line).stdout for line in lines] for lang, lines in STEPS.items()}


def _lines(text) -> list[str]:
    """The lines a program showed, without spaces at their ends: those do not show on the screen."""
    return [line.rstrip() for line in str(text).rstrip("\n").splitlines()]


def same_lines(a, b) -> bool:
    return isinstance(a, str) and isinstance(b, str) and _lines(a) == _lines(b)


def _order(want: str):
    """The right lines, in another order."""
    return lambda got, text: _lines(got) != _lines(want) and sorted(_lines(got)) == sorted(_lines(want))


def _first_out_of_place(want: str):
    """The first line under the cell that is not the one that belongs there."""
    def detail(got, text):
        for have, should in zip(_lines(got), _lines(want)):
            if have != should:
                return {"line": have, "should": should}
        return {"line": "", "should": ""}
    return detail


def _capitals(want: str):
    """The right lines, but with other capitals."""
    return lambda got, text: [line.lower() for line in _lines(got)] == [line.lower() for line in _lines(want)]


def _one_line(want: str):
    """All the words on one line, with or without commas between them."""
    return lambda got, text: len(_lines(got)) == 1 and re.findall(r"\w+", _lines(got)[0]) == re.findall(r"\w+", want)


def _extra(want: str):
    """The right lines in the right order, and more lines besides."""
    def test(got, text):
        rest = iter(_lines(got))
        return len(_lines(got)) > len(_lines(want)) and all(line in rest for line in _lines(want))
    return test


def _space_inside(want: str):
    """The right lines, but with spaces at their start: a space between the quote and the text."""
    return lambda got, text: _lines(got) != _lines(want) and [line.strip() for line in _lines(got)] == _lines(want)


_TOOLS = "Python\nNumPy\nPyTorch\n"
_COUNTDOWN = "3\n2\n1\nGo!\n"

# Python cannot read these, or stops at them; the pattern finds out why. They hold for both exercises.
# The first pattern that fits wins, so the most telling one comes first.
_SYNTAX = [
    Mistake("curly-quote", syntax=r"[“”‘’„]",
            en="Your program has a curly quote, like `“`. Python only reads straight quotes, like `\"`. Curly ones come from text copied out of a document or a web page: type the quotes again in the box.",
            nl="Je programma heeft een gekruld aanhalingsteken, zoals `“`. Python leest alleen rechte, zoals `\"`. Gekrulde komen uit tekst die je uit een document of webpagina hebt gekopieerd: typ de aanhalingstekens opnieuw in het vak."),
    Mistake("space-at-start", syntax=r"^[ \t]+\S",
            en="A line starts with a space. In this program, every line starts at the very left: remove the spaces before `print`.",
            nl="Een regel begint met een spatie. In dit programma begint elke regel helemaal links: haal de spaties voor `print` weg."),
    Mistake("square-brackets", syntax=r"print\s*\[",
            en="What `print` shows goes between parentheses, `( )`, not between square brackets, `[ ]`. Like `print(\"Hello\")`.",
            nl="Wat `print` laat zien, staat tussen ronde haakjes, `( )`, niet tussen vierkante, `[ ]`. Zoals in `print(\"Hallo\")`."),
    Mistake("two-on-one-line", syntax=r"\)[ \t]*print\b",
            en="Two instructions are on one line. Put each `print` on its own line.",
            nl="Er staan twee instructies op één regel. Zet elke `print` op een eigen regel."),
    Mistake("quote-missing", syntax=r"""print\(\s*(["'])(?:(?!\1)[^\n])*$|print\(\s*[^"'\s][^\n]*["']\s*\)""",
            en="A quote is missing. A string starts and ends with a quote, like `\"Hello\"` in `print(\"Hello\")`.",
            nl="Er mist een aanhalingsteken. Een string begint en eindigt met een aanhalingsteken, zoals `\"Hallo\"` in `print(\"Hallo\")`."),
    Mistake("parenthesis-missing", syntax=r"""^\s*print\s*["']|^\s*print\s*\([^\n)]*$""",
            en="A parenthesis is missing. What `print` shows goes between parentheses, like `print(\"Hello\")`.",
            nl="Er mist een haakje. Wat `print` laat zien, staat tussen ronde haakjes, zoals in `print(\"Hallo\")`."),
]

# The lesson is done when these are correct. "quotes" is the choice question in the lesson.
REQUIRED = ["tools", "countdown", "quotes"]

EXERCISES = {
    "tools": CodeBox(
        name="tools",
        start="# Python, NumPy, PyTorch\n",
        solution='print("Python")\nprint("NumPy")\nprint("PyTorch")',
        outcome="stdout",
        expect=_TOOLS,
        compare=same_lines,
        mistakes=_SYNTAX + [
            Mistake("order", matches=_order(_TOOLS), detail=_first_out_of_place(_TOOLS),
                    en="All three lines are there, but not in the order the task asks for. The first line out of place is `{line}`; there, `{should}` belongs. Python runs your program from top to bottom.",
                    nl="Alle drie de regels staan er, maar niet in de volgorde die de opdracht vraagt. De eerste regel die verkeerd staat, is `{line}`; daar hoort `{should}`. Python voert je programma van boven naar beneden uit."),
            Mistake("one-line", matches=_one_line(_TOOLS),
                    en="All three names are on one line. Each `print` shows its text on a new line, so use one `print` for each line.",
                    nl="Alle drie de namen staan op één regel. Elke `print` zet zijn tekst op een nieuwe regel, dus gebruik één `print` per regel."),
            Mistake("capitals", matches=_capitals(_TOOLS), state="almost",
                    en="The names are right, but the capitals are not. Python shows text exactly as you write it, so copy the capitals too: `Python`, `NumPy`, `PyTorch`.",
                    nl="De namen kloppen, maar de hoofdletters niet. Python laat tekst precies zien zoals je hem schrijft, dus neem ook de hoofdletters over: `Python`, `NumPy`, `PyTorch`."),
            Mistake("space-inside", matches=_space_inside(_TOOLS), state="almost",
                    en="The names are right, but a line starts with a space: there is a space between the quote and the name. Python shows that space too, so remove it.",
                    nl="De namen kloppen, maar een regel begint met een spatie: er staat een spatie tussen het aanhalingsteken en de naam. Ook die spatie laat Python zien, dus haal hem weg."),
            Mistake("extra-line", matches=_extra(_TOOLS), state="almost",
                    en="The three lines are there, in the right order, but your program shows more than that. Remove the `print` you do not need.",
                    nl="De drie regels staan er, in de goede volgorde, maar je programma laat meer zien. Haal de `print` weg die je niet nodig hebt."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Only the comment is in the box, and Python skips it. Write your program under it: one `print` for each line.",
                      "nl": "Er staat alleen het commentaar in het vak, en dat slaat Python over. Schrijf je programma eronder: één `print` voor elke regel."},
            "cleared": {"en": "There is no code in the box. Write your program in it: one `print` for each line.",
                        "nl": "Er staat geen code in het vak. Schrijf je programma erin: één `print` voor elke regel."},
            "wrong": {"en": "Your program shows:\n\n```\n{got}\n```\n\nThat is not yet the three lines `Python`, `NumPy` and `PyTorch`. Compare it line by line.",
                      "nl": "Je programma laat dit zien:\n\n```\n{got}\n```\n\nDat zijn nog niet de drie regels `Python`, `NumPy` en `PyTorch`. Vergelijk het regel voor regel."},
            "correct": {"en": "Your program shows `Python`, `NumPy` and `PyTorch`, each on its own line, in that order.",
                        "nl": "Je programma laat `Python`, `NumPy` en `PyTorch` zien, elk op een eigen regel, in die volgorde."},
        },
    ),
    "countdown": CodeBox(
        name="countdown",
        start='print("1")\nprint("Go!")\nprint("3")\nprint("2")',
        solution='print("3")\nprint("2")\nprint("1")\nprint("Go!")',
        outcome="stdout",
        expect=_COUNTDOWN,
        compare=same_lines,
        mistakes=_SYNTAX + [
            Mistake("order", matches=_order(_COUNTDOWN), detail=_first_out_of_place(_COUNTDOWN),
                    en="All four lines are there, but not yet in the order of a countdown. The first line out of place shows `{line}`; there, `{should}` belongs. Python runs from top to bottom.",
                    nl="Alle vier de regels staan er, maar nog niet in de volgorde van een countdown. De eerste regel die verkeerd staat, laat `{line}` zien; daar hoort `{should}`. Python werkt van boven naar beneden."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "The lines are still in the mixed-up order: the program shows 1, Go!, 3, 2. Move the lines so it counts down from 3.",
                      "nl": "De regels staan nog door elkaar: het programma laat 1, Go!, 3, 2 zien. Verplaats de regels zodat het aftelt vanaf 3."},
            "cleared": {"en": "There are no lines of code in the box any more. Type the four lines back, each on its own line: `print(\"1\")`, `print(\"Go!\")`, `print(\"3\")` and `print(\"2\")`. Then put them in the order of a countdown.",
                        "nl": "Er staan geen regels code meer in het vak. Typ de vier regels terug, elk op een eigen regel: `print(\"1\")`, `print(\"Go!\")`, `print(\"3\")` en `print(\"2\")`. Zet ze dan in de volgorde van een countdown."},
            "wrong": {"en": "Your program shows:\n\n```\n{got}\n```\n\nA countdown shows `3`, `2`, `1` and then `Go!`. Do not change the lines themselves; only move them.",
                      "nl": "Je programma laat dit zien:\n\n```\n{got}\n```\n\nEen countdown laat `3`, `2`, `1` en dan `Go!` zien. Verander de regels zelf niet; verplaats ze alleen."},
            "correct": {"en": "3, 2, 1, Go! The same four instructions, now in the right order, give the countdown.",
                        "nl": "3, 2, 1, Go! Dezelfde vier instructies geven in de goede volgorde de countdown."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("tools", "# Python, NumPy, PyTorch\n", "empty", None),
    ("tools", 'print("Python")\nprint("NumPy")\nprint("PyTorch")', "correct", None),
    ("tools", '# my start\nprint("Python")\nprint("NumPy")\nprint("PyTorch")  ', "correct", None),
    ("tools", 'print("PyTorch")\nprint("NumPy")\nprint("Python")', "wrong", "order"),
    ("tools", 'print("Python NumPy PyTorch")', "wrong", "one-line"),
    ("tools", 'print("python")\nprint("numpy")\nprint("pytorch")', "almost", "capitals"),
    ("tools", 'print(Python)', "error", None),
    ("tools", 'print("Python)', "error", "quote-missing"),
    ("tools", "print('Python)", "error", "quote-missing"),
    ("tools", 'print "Python"', "error", "parenthesis-missing"),
    ("tools", 'print(“Python”)', "error", "curly-quote"),
    ("tools", 'print(“Python")', "error", "curly-quote"),
    ("tools", ' print("Python")\nprint("NumPy")\nprint("PyTorch")', "error", "space-at-start"),
    ("tools", 'print["Python"]', "error", "square-brackets"),
    ("tools", 'print("Python") print("NumPy")', "error", "two-on-one-line"),
    ("tools", 'print("Python, NumPy, PyTorch")', "wrong", "one-line"),
    ("tools", 'print(" Python")\nprint("NumPy")\nprint("PyTorch")', "almost", "space-inside"),
    ("tools", 'print("Python")\nprint("NumPy")\nprint("PyTorch")\nprint("Jax")', "almost", "extra-line"),
    ("tools", '# nothing printed yet\nname = "Python"', "wrong", "nothing"),
    ("tools", "", "empty", None),
    ("tools", 'print("Python")\nprint("NumPy")', "wrong", None),
    ("countdown", 'print("1")\nprint("Go!")\nprint("3")\nprint("2")', "empty", None),
    ("countdown", 'print("3")\nprint("2")\nprint("1")\nprint("Go!")', "correct", None),
    ("countdown", 'print("3")\nprint("1")\nprint("2")\nprint("Go!")', "wrong", "order"),
    ("countdown", 'print("3")\nprint("2")\nprint("1")', "wrong", None),
    ("countdown", "", "empty", None),
]

