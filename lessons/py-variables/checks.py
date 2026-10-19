# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson py-variables, and the program and worked example the lesson shows."""
import contextlib
import io
import math
import re

from everything_ai.check import CodeBox, Mistake, run_code

# The program in the picture. For each line: the names and their values after it, and what it shows under the cell.
STEPS = ["price = 2.5", "cups = 3", "total = price * cups", "print(total)"]


def _states(lines: list[str]) -> tuple[list[dict], list[str]]:
    """Run the program one line at a time, the way Python does, and keep the names after each line and what it showed."""
    names: dict = {}
    states, shown = [], []
    for line in lines:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(line, names)
        states.append({k: v for k, v in names.items() if not k.startswith("__")})
        shown.append(out.getvalue())
    return states, shown


NAMES, SHOWN = _states(STEPS)

# The worked example, with its comments per language, and what it shows (Python works that out).
EXAMPLE = {
    "en": ("# 1. a name for each number you start with\npizza = 9.5\ncount = 2\ndelivery = 2.5\n"
           "# 2. the calculation with those names, and a name for the result\ntotal = pizza * count + delivery\n"
           "# 3. show the result\nprint(total)"),
    "nl": ("# 1. een naam voor elk getal waarmee je begint\npizza = 9.5\ncount = 2\ndelivery = 2.5\n"
           "# 2. de berekening met die namen, en een naam voor de uitkomst\ntotal = pizza * count + delivery\n"
           "# 3. laat de uitkomst zien\nprint(total)"),
}
EXAMPLE_SHOWS = run_code(EXAMPLE["en"]).stdout
assert EXAMPLE_SHOWS == run_code(EXAMPLE["nl"]).stdout

# A second worked example, in the form of exercise 2: a number of choices to the power of the number of places.
POWER = {
    "en": "# 1. a name for each number you start with\nsides = 6\ndice = 3\n"
          "# 2. 6 outcomes for each die, for 3 dice: 6 * 6 * 6, a power\noutcomes = sides ** dice\n"
          "# 3. show the result\nprint(outcomes)",
    "nl": "# 1. een naam voor elk getal waarmee je begint\nsides = 6\ndice = 3\n"
          "# 2. 6 uitkomsten per dobbelsteen, voor 3 dobbelstenen: 6 * 6 * 6, een macht\noutcomes = sides ** dice\n"
          "# 3. laat de uitkomst zien\nprint(outcomes)",
}
POWER_SHOWS = run_code(POWER["en"]).stdout
assert POWER_SHOWS == run_code(POWER["nl"]).stdout


def _lines(text) -> list[str]:
    return [line.rstrip() for line in str(text).rstrip("\n").splitlines()]


def _same_line(a: str, b: str) -> bool:
    """The same line under the cell: the same text, or the same number (7.5 and 7.50, 10000 and 10000.0)."""
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-9)
    except ValueError:
        return a.strip() == b.strip()


def same_lines(a, b) -> bool:
    return (isinstance(a, str) and isinstance(b, str) and len(_lines(a)) == len(_lines(b))
            and all(_same_line(x, y) for x, y in zip(_lines(a), _lines(b))))


def _numbers(line: str) -> list[float]:
    return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", line)]


def _shows(number):
    """The program showed exactly this number, on one line."""
    return lambda got, text: len(_lines(got)) == 1 and _numbers(_lines(got)[0]) == [number] and \
        _same_line(_lines(got)[0], str(number))


def _also(number):
    """The right number is somewhere under the cell, but there is more."""
    return lambda got, text: any(math.isclose(n, number, rel_tol=1e-9) for line in _lines(got) for n in _numbers(line))


def _printed_word(word):
    """The program printed the name itself, as text, instead of its value."""
    return lambda got, text: _lines(got)[-1:] == [word]


def _line_with(pattern):
    """{line} in a message: the first line the pattern is found on."""
    def detail(got, text):
        for line in text.splitlines():
            if re.search(pattern, line):
                return {"line": line.strip()}
        return {"line": ""}
    return detail


def _both(key, pattern, en, nl):
    """One mistake, recognized whether or not the program still runs: by the pattern in what the learner typed."""
    return [Mistake(key, syntax=pattern, en=en, nl=nl),
            Mistake(key, matches=lambda got, text: re.search(pattern, text, re.MULTILINE) is not None,
                    detail=_line_with(pattern), en=en, nl=nl)]


# Mistakes in what the learner typed. They are checked in this order; the first that fits is told.
_DOTS = _both("dots-left", r"\.\.\.",
              en="There is still a `...` in your program. Replace every `...` with a number or a calculation.",
              nl="Er staat nog een `...` in je programma. Vervang elke `...` door een getal of een berekening.")
_CURLY = [Mistake("curly-quote", syntax=r"[“”‘’„]",
                  en="Your program has a curly quote, like `“`. Python only reads straight quotes, like `\"`: type the quotes again in the box.",
                  nl="Je programma heeft een gekruld aanhalingsteken, zoals `“`. Python leest alleen rechte, zoals `\"`: typ de aanhalingstekens opnieuw in het vak.")]
_QUOTED = _both("quoted-number", r"""=\s*["']\s*-?\d[\d.,]*\s*["']""",
                en="On the line `{line}`, the number is between quotes, so it is text, a string, and not a number. Remove the quotes around the number.",
                nl="Op de regel `{line}` staat het getal tussen aanhalingstekens, dus is het tekst, een string, en geen getal. Haal de aanhalingstekens rond het getal weg.")
_COMMA = _both("decimal-comma", r"=\s*\d+,\s*\d",
               en="Python writes decimal numbers with a point, not a comma: `2.5`, not `2,5`. With a comma, Python makes two numbers out of it.",
               nl="Python schrijft kommagetallen met een punt, niet met een komma: `2.5`, niet `2,5`. Met een komma maakt Python er twee getallen van.")
_HAT = _both("hat", r"\^",
             en="In Python, `^` does not mean \"to the power of\". Write a power with two stars, like `2 ** 3`.",
             nl="In Python betekent `^` niet \"tot de macht\". Een macht schrijf je met twee sterretjes, zoals `2 ** 3`.")
# Python cannot read these at all.
_SYNTAX = [
    Mistake("nothing-after-equals", syntax=r"=[ \t]*$",
            en="The line `{line}` has nothing after `=`. After `=` comes the value the name gets, like `price = 2.5`.",
            nl="Op de regel `{line}` staat niets na `=`. Na `=` komt de waarde die de naam krijgt, zoals in `price = 2.5`."),
    Mistake("sign-at-end", syntax=r"[-+*/][ \t]*$",
            en="The line `{line}` ends with a calculation sign. A calculation needs a number or a name on both sides of the sign.",
            nl="De regel `{line}` eindigt met een rekenteken. Een berekening heeft aan beide kanten van het teken een getal of een naam nodig."),
    Mistake("times-x", syntax=r"(?<=[\w)])\s+x\s+(?=[\w(])|×",
            en="In Python, times is `*`, not `x` or `×`. Like `2.5 * 3`.",
            nl="In Python is keer `*`, niet `x` of `×`. Zoals in `2.5 * 3`."),
    Mistake("unit", syntax=r"[€$]|\d[ \t]+[A-Za-z]",
            en="On the line `{line}`, the value is more than a number. Write only the number, without a € sign or a word after it: `price = 2.5`.",
            nl="Op de regel `{line}` is de waarde meer dan een getal. Schrijf alleen het getal, zonder €-teken of een woord erachter: `price = 2.5`."),
    Mistake("double-equals", syntax=r"^\s*\w+\s*==",
            en="To give a name a value, you use one `=`: `price = 2.5`. Two, `==`, mean something else, which a later lesson explains.",
            nl="Een naam een waarde geven doe je met één `=`: `price = 2.5`. Twee, `==`, betekenen iets anders, dat een latere les uitlegt."),
    Mistake("value-left", syntax=r"^\s*[\d.]+\s*=(?!=)",
            en="In Python, the name goes on the left of `=`, and the value on the right: `price = 2.5`, not `2.5 = price`.",
            nl="In Python staat de naam links van `=`, en de waarde rechts: `price = 2.5`, niet `2.5 = price`."),
]
_FIRST = _DOTS + _CURLY + _QUOTED + _COMMA + _HAT + _SYNTAX

# The lesson is done when these are correct. "divide" is the choice question in the lesson.
REQUIRED = ["coffee", "lock", "divide"]

EXERCISES = {
    "coffee": CodeBox(
        name="coffee",
        start="price = ...\ncups = ...\ntotal = ...\nprint(total)",
        solution="price = 2.5\ncups = 3\ntotal = price * cups\nprint(total)",
        outcome="stdout",
        expect="7.5\n",
        compare=same_lines,
        mistakes=_FIRST + [
            Mistake("added", matches=_shows(5.5),
                    en="Your program shows 5.5: you added the price and the number of cups. For the total, multiply them: `*`.",
                    nl="Je programma laat 5.5 zien: je telde de prijs en het aantal koppen op. Voor het totaal vermenigvuldig je ze: `*`."),
            Mistake("word", matches=_printed_word("total"),
                    en="Your program shows the word `total`, because it is between quotes. Without quotes, `print(total)` shows the value of the name.",
                    nl="Je programma laat het woord `total` zien, omdat het tussen aanhalingstekens staat. Zonder aanhalingstekens laat `print(total)` de waarde van de naam zien."),
            Mistake("extra", matches=_also(7.5), state="almost",
                    en="The right total, 7.5, is there, but your program shows more. Show only the total: one `print`, with only `total` in it.",
                    nl="Het goede totaal, 7.5, staat er, maar je programma laat meer zien. Laat alleen het totaal zien: één `print`, met alleen `total` erin."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Fill in the three `...`: the price, the number of cups, and the calculation for the total.",
                      "nl": "Vul de drie `...` in: de prijs, het aantal koppen, en de berekening voor het totaal."},
            "cleared": {"en": "There is no code in the box any more. Type the four lines back, each on its own line: `price = ...`, `cups = ...`, `total = ...` and `print(total)`. Then fill in each `...`.",
                        "nl": "Er staat geen code meer in het vak. Typ de vier regels terug, elk op een eigen regel: `price = ...`, `cups = ...`, `total = ...` en `print(total)`. Vul dan elke `...` in."},
            "wrong": {"en": "Your program shows:\n\n```\n{got}\n```\n\nThat is not yet the total for three cups at 2.5 each. Look at step 2 of the recipe: which calculation gives the total?",
                      "nl": "Je programma laat dit zien:\n\n```\n{got}\n```\n\nDat is nog niet het totaal voor drie koppen van € 2,50. Kijk naar stap 2 van het recept: welke berekening geeft het totaal?"},
            "correct": {"en": "Your program shows 7.5: three cups at 2.5 each.",
                        "nl": "Je programma laat 7.5 zien: drie koppen van € 2,50 per stuk."},
        },
    ),
    "lock": CodeBox(
        name="lock",
        start="digits = 4\ncodes = ...\nprint(codes)",
        solution="digits = 4\ncodes = 10 ** digits\nprint(codes)",
        outcome="stdout",
        expect="10000\n",
        compare=same_lines,
        mistakes=_FIRST + [
            Mistake("times", matches=_shows(40),
                    en="Your program shows 40: that is 10 times 4. With 1 digit there are 10 codes, with 2 digits 100, with 3 digits 1000. Which symbol in the table does that?",
                    nl="Je programma laat 40 zien: dat is 10 keer 4. Met 1 cijfer zijn er 10 codes, met 2 cijfers 100, met 3 cijfers 1000. Welk teken uit de tabel doet dat?"),
            Mistake("swapped", matches=_shows(4 ** 10),
                    en="Your program shows 4 to the power 10. It is the other way round: 10 choices for each digit, and 4 digits. Swap the two numbers around `**`.",
                    nl="Je programma laat 4 tot de macht 10 zien. Het is andersom: 10 keuzes per cijfer, en 4 cijfers. Draai de twee getallen rond `**` om."),
            Mistake("thousands-point", matches=lambda got, text: re.search(r"=\s*\d{1,3}\.\d{3}\b", text) is not None,
                    en="In Python, a point in a number is a decimal point: `10.000` is ten. Write ten thousand without a point: `10000`.",
                    nl="In Python is een punt in een getal een decimale punt: `10.000` is tien. Schrijf tienduizend zonder punt: `10000`."),
            Mistake("highest", matches=_shows(9999),
                    en="Your program shows 9999: the highest code. But 0000 is a code too, so there is one more.",
                    nl="Je programma laat 9999 zien: de hoogste code. Maar 0000 is ook een code, dus er is er één meer."),
            Mistake("three-digits", matches=_shows(1000),
                    en="Your program shows 1000: the number of codes with 3 digits. This lock has 4.",
                    nl="Je programma laat 1000 zien: het aantal codes met 3 cijfers. Dit slot heeft er 4."),
            Mistake("no-zero", matches=_shows(9 ** 4),
                    en="Your program shows 6561: 9 to the power 4. Each digit can be 0 to 9, and that is 10 choices.",
                    nl="Je programma laat 6561 zien: 9 tot de macht 4. Elk cijfer kan 0 tot en met 9 zijn, en dat zijn 10 keuzes."),
            Mistake("word", matches=_printed_word("codes"),
                    en="Your program shows the word `codes`, because it is between quotes. Without quotes, `print(codes)` shows the value of the name.",
                    nl="Je programma laat het woord `codes` zien, omdat het tussen aanhalingstekens staat. Zonder aanhalingstekens laat `print(codes)` de waarde van de naam zien."),
            Mistake("extra", matches=_also(10000), state="almost",
                    en="The right number, 10000, is there, but your program shows more. Show only the number of codes: one `print`, with only `codes` in it.",
                    nl="Het goede aantal, 10000, staat er, maar je programma laat meer zien. Laat alleen het aantal codes zien: één `print`, met alleen `codes` erin."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Fill in the `...` with the calculation for the number of codes.",
                      "nl": "Vul op de plek van de `...` de berekening in voor het aantal codes."},
            "cleared": {"en": "There is no code in the box any more. Type the three lines back, each on its own line: `digits = 4`, `codes = ...` and `print(codes)`. Then fill in the `...`.",
                        "nl": "Er staat geen code meer in het vak. Typ de drie regels terug, elk op een eigen regel: `digits = 4`, `codes = ...` en `print(codes)`. Vul dan de `...` in."},
            "wrong": {"en": "Your program shows:\n\n```\n{got}\n```\n\nThat is not yet the number of codes. With 1 digit there are 10 codes, with 2 digits 100. How does it continue?",
                      "nl": "Je programma laat dit zien:\n\n```\n{got}\n```\n\nDat is nog niet het aantal codes. Met 1 cijfer zijn er 10 codes, met 2 cijfers 100. Hoe gaat dat verder?"},
            "correct": {"en": "10000 codes: 10 choices for each of the 4 digits, so 10 to the power 4.",
                        "nl": "10000 codes: 10 keuzes voor elk van de 4 cijfers, dus 10 tot de macht 4."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_C = "price = 2.5\ncups = 3\ntotal = price * cups\nprint(total)"
CASES = [
    ("coffee", "price = ...\ncups = ...\ntotal = ...\nprint(total)", "empty", None),
    ("coffee", "", "empty", None),
    ("coffee", _C, "correct", None),
    ("coffee", "price = 2.5\ncups = 3\ntotal = cups * price\nprint(total)", "correct", None),
    ("coffee", "print(2.5 * 3)", "correct", None),
    ("coffee", "price = 2.50\ncups = 3.0\ntotal = price * cups\nprint(total)", "correct", None),
    ("coffee", "price = 2.5\ncups = 3\ntotal = price + cups\nprint(total)", "wrong", "added"),
    ("coffee", "price = 2,5\ncups = 3\ntotal = price * cups\nprint(total)", "wrong", "decimal-comma"),
    ("coffee", "price = 2,5\ncups = 3\ntotal = price + cups\nprint(total)", "error", "decimal-comma"),
    ("coffee", "price = 2.5\ncups = 3\ntotal = price * cups\nprint(\"total\")", "wrong", "word"),
    ("coffee", "price = 2.5\ncups = ...\ntotal = price * cups\nprint(total)", "error", "dots-left"),
    ("coffee", "price = 2.5\ncups = 3\ntotal = ...\nprint(total)", "wrong", "dots-left"),
    ("coffee", "price = \"2.5\"\ncups = 3\ntotal = price * cups\nprint(total)", "wrong", "quoted-number"),
    ("coffee", "price = 2.5\ncups = \"3\"\ntotal = price * cups\nprint(total)", "error", "quoted-number"),
    ("coffee", "price =\ncups = 3\ntotal = price * cups\nprint(total)", "error", "nothing-after-equals"),
    ("coffee", "price = 2.5\ncups = 3\ntotal = price *\nprint(total)", "error", "sign-at-end"),
    ("coffee", "price = 2.5\ncups = 3\ntotal = price x cups\nprint(total)", "error", "times-x"),
    ("coffee", "2.5 = price\ncups = 3\ntotal = price * cups\nprint(total)", "error", "value-left"),
    ("coffee", "price = 2.5\ncups = 3\ntotal = price ^ cups\nprint(total)", "error", "hat"),
    ("coffee", "price = 2.5\ncups = 3\ntotal = price * cups\nprint(“total”)", "error", "curly-quote"),
    ("coffee", "price = 2.5\ncups = 3\ntotal = price * cups\nprint(price)\nprint(total)", "almost", "extra"),
    ("coffee", "price = 2.5\ncups = 3\ntotal = price * cups\nprint(\"Total:\", total)", "almost", "extra"),
    ("coffee", "total = price * cups\nprice = 2.5\ncups = 3\nprint(total)", "error", None),
    ("coffee", "price = 2.5 euro\ncups = 3\ntotal = price * cups\nprint(total)", "error", "unit"),
    ("coffee", "price = €2.5\ncups = 3\ntotal = price * cups\nprint(total)", "error", "unit"),
    ("coffee", "price == 2.5\ncups = 3\ntotal = price * cups\nprint(total)", "error", "double-equals"),
    ("coffee", "price = 2.5\ncups = 3\ntotal = price * 4\nprint(total)", "wrong", None),
    ("lock", "digits = 4\ncodes = ...\nprint(codes)", "empty", None),
    ("lock", "", "empty", None),
    ("lock", "digits = 4\ncodes = 10 ** digits\nprint(codes)", "correct", None),
    ("lock", "digits = 4\ncodes = 10 * 10 * 10 * 10\nprint(codes)", "correct", None),
    ("lock", "digits = 4\ncodes = 10.0 ** digits\nprint(codes)", "correct", None),
    ("lock", "digits = 4\ncodes = 10 * digits\nprint(codes)", "wrong", "times"),
    ("lock", "digits = 4\ncodes = 10 ^ digits\nprint(codes)", "wrong", "hat"),
    ("lock", "digits = 4\ncodes = digits ** 10\nprint(codes)", "wrong", "swapped"),
    ("lock", "digits = 4\ncodes = 9999\nprint(codes)", "wrong", "highest"),
    ("lock", "digits = 4\ncodes = 10.000\nprint(codes)", "wrong", "thousands-point"),
    ("lock", "digits = 4\ncodes = 10 ** 3\nprint(codes)", "wrong", "three-digits"),
    ("lock", "digits = 4\ncodes = 9 ** digits\nprint(codes)", "wrong", "no-zero"),
    ("lock", "digits = 4\ncodes = 10,000\nprint(codes)", "wrong", "decimal-comma"),
    ("lock", "digits = 4\ncodes = 10 ** digits\nprint(\"codes\")", "wrong", "word"),
    ("lock", "digits = 4\ncodes = 10 ** digits\nprint(digits)\nprint(codes)", "almost", "extra"),
]
