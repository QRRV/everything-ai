# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson py-for, and the loop in its picture, round by round."""
import re

from everything_ai.check import CodeBox, Mistake

# The loop in the picture: a total of temperatures, step by step. Python works out the names after each step.
PICTURE_ITEMS = [12, 15, 9]


def _steps(items):
    total = 0
    lines = {"en": ["total = 0"], "nl": ["total = 0"]}
    names, shown = [{"total": 0}], [""]
    for k, item in enumerate(items):
        lines["en"].append(f"round {k + 1}: total = {total} + {item}")
        lines["nl"].append(f"ronde {k + 1}: total = {total} + {item}")
        total = total + item
        names.append({"temp": item, "total": total})
        shown.append("")
    lines["en"].append("print(total)")
    lines["nl"].append("print(total)")
    names.append(dict(names[-1]))
    shown.append(f"{total}\n")
    return lines, names, shown


PICTURE_LINES, PICTURE_NAMES, PICTURE_SHOWN = _steps(PICTURE_ITEMS)


def _lines(text) -> list[str]:
    return [line.rstrip() for line in str(text).rstrip("\n").splitlines()]


def _same_line(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-9
    except ValueError:
        return a.strip() == b.strip()


def same_lines(a, b) -> bool:
    """The same lines under the cell; 11 and 11.0 count as the same number."""
    return (isinstance(a, str) and isinstance(b, str) and len(_lines(a)) == len(_lines(b))
            and all(_same_line(x, y) for x, y in zip(_lines(a), _lines(b))))


def _shows(*numbers):
    return lambda got, text: len(_lines(got)) == 1 and any(_same_line(_lines(got)[0], str(n)) for n in numbers)


def _numbers(text) -> list[float]:
    return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", str(text))]


def _print_inside(got, text):
    """More than one line under the cell, and an indented print."""
    return len(_lines(got)) > 1 and re.search(r"^[ \t]+print\(", text, re.MULTILINE) is not None


_SYNTAX = [
    Mistake("dots-left", syntax=r"^[^#\n]*\.\.\.",
            en="There is still a `...` in your program. Replace it with the line or lines that should run in each round.",
            nl="Er staat nog een `...` in je programma. Vervang die door de regel of regels die elke ronde moeten draaien."),
    Mistake("curly-quote", syntax=r"[“”‘’„]",
            en="Your program has a curly quote, like `“`. Python only reads straight quotes, like `\"`: type the quotes again in the box.",
            nl="Je programma heeft een gekruld aanhalingsteken, zoals `“`. Python leest alleen rechte, zoals `\"`: typ de aanhalingstekens opnieuw in het vak."),
    Mistake("colon-missing", syntax=r"^[ \t]*(for\b[^:\n]*|if\b[^:\n]*|else[ \t]*)$",
            en="The line `{line}` must end with a colon, `:`. The colon tells Python that the indented lines below belong to it.",
            nl="De regel `{line}` moet eindigen op een dubbele punt, `:`. Die vertelt Python dat de ingesprongen regels eronder erbij horen."),
    Mistake("capital", syntax=r"^[ \t]*(For|FOR|If|Else)\b",
            en="Python writes `for`, `if` and `else` in small letters. On the line `{line}`, change the capital.",
            nl="Python schrijft `for`, `if` en `else` met kleine letters. Verander op de regel `{line}` de hoofdletter."),
]
_DOTS_RAN = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                    en="There is still a `...` in your program. Replace it with the line or lines that should run in each round.",
                    nl="Er staat nog een `...` in je programma. Vervang die door de regel of regels die elke ronde moeten draaien.")
_RESET = Mistake("reset-inside", matches=lambda got, text: re.search(r"^[ \t]+(total|frost_days)\s*=\s*0\s*$", text, re.MULTILINE) is not None,
                 en="A line inside the loop gives the name 0 again, so the count starts over in every round. The line with `= 0` goes before the loop, at the left edge.",
                 nl="Een regel in de loop geeft de naam weer 0, dus de telling begint elke ronde opnieuw. De regel met `= 0` hoort vóór de loop, helemaal links.")
_PRINT_INSIDE = Mistake("print-inside", matches=_print_inside,
                        en="Your program shows more than one line, because a `print` is indented and runs inside the loop. Keep one `print`, at the left edge, after the loop.",
                        nl="Je programma laat meer dan één regel zien, omdat een `print` inspringt en in de loop draait. Houd één `print` over, helemaal links, na de loop.")


def _repeated(number):
    """Every line is the right number, but there is more than one line."""
    return lambda got, text: len(_lines(got)) > 1 and all(_numbers(line) == [number] for line in _lines(got))


def _with_text(number):
    """One line has the right number, and other text besides."""
    return lambda got, text: len(_lines(got)) >= 1 and any(number in _numbers(line) and not _same_line(line, str(number)) for line in _lines(got))

# The lesson is done when these are correct. "range" is the choice question in the lesson.
REQUIRED = ["total", "range", "frost"]

EXERCISES = {
    "total": CodeBox(
        name="total",
        start="prices = [2.5, 4, 1.5, 3]\ntotal = 0\nfor price in prices:\n    ...\nprint(total)",
        solution="prices = [2.5, 4, 1.5, 3]\ntotal = 0\nfor price in prices:\n    total = total + price\nprint(total)",
        outcome="stdout",
        compare=same_lines,
        others=[{"prices": [1, 2]}, {"prices": []}],
        keep=["prices = [2.5, 4, 1.5, 3]", "total = 0"],
        mistakes=_SYNTAX + [
            Mistake("range-of-list", syntax=r"range\(\s*prices\s*\)",
                    en="`range` takes a number, not a list. To go through the list itself, write `for price in prices:`.",
                    nl="`range` krijgt een getal, geen lijst. Wil je de lijst zelf langsgaan, schrijf dan `for price in prices:`."),
            Mistake("index-number", syntax=r"\bprice\s*\[",
                    en="`price` is one price, a number; it has no places. Use `price` itself, the price of this round.",
                    nl="`price` is één prijs, een getal; het heeft geen plekken. Gebruik `price` zelf, de prijs van deze ronde."),
            Mistake("whole-list", syntax=r"\bprices\s*\+|\+\s*prices\b|\+=\s*prices\b",
                    en="Inside the loop, `price` is one price; `prices` is the whole list. Add `price`, the price of this round.",
                    nl="In de loop is `price` één prijs; `prices` is de hele lijst. Tel `price` op, de prijs van deze ronde."),
            _DOTS_RAN, _RESET, _PRINT_INSIDE,
            Mistake("twice", matches=_repeated(11),
                    en="Your program shows the total more than once. Keep one `print(total)`, at the left edge, after the loop.",
                    nl="Je programma laat het totaal meer dan eens zien. Houd één `print(total)` over, helemaal links, na de loop."),
            Mistake("with-text", matches=_with_text(11),
                    en="Your program has the right total, but the check reads only the number: keep the last line as `print(total)`.",
                    nl="Je programma heeft het goede totaal, maar de check leest alleen het getal: houd de laatste regel `print(total)`."),
            Mistake("last-only", matches=lambda got, text: _shows(3)(got, text) and re.search(r"^[ \t]+total\s*=\s*price\s*$", text, re.MULTILINE) is not None,
                    en="Your program shows 3: only the last price. Each round, `total` should get what it already is, plus the price of that round.",
                    nl="Je programma laat 3 zien: alleen de laatste prijs. Elke ronde moet `total` krijgen wat hij al is, plus de prijs van die ronde."),
            Mistake("not-kept", matches=lambda got, text: _shows(0)(got, text) and re.search(r"^[ \t]+total\s*\+", text, re.MULTILINE) is not None,
                    en="Your total stays 0. `total + price` calculates a sum, but does not keep it: give `total` the new value with `=`.",
                    nl="Je totaal blijft 0. `total + price` rekent een som uit, maar bewaart hem niet: geef `total` de nieuwe waarde met `=`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the line that adds the price of each round to `total`.",
                      "nl": "Vervang de `...` door de regel die de prijs van elke ronde bij `total` optelt."},
            "cleared": {"en": "There is no code in the box any more. Type the lines back: `prices = [2.5, 4, 1.5, 3]`, `total = 0`, `for price in prices:`, the line inside the loop, and `print(total)`.",
                        "nl": "Er staat geen code meer in het vak. Typ de regels terug: `prices = [2.5, 4, 1.5, 3]`, `total = 0`, `for price in prices:`, de regel in de loop, en `print(total)`."},
            "wrong": {"en": "Your program shows:\n\n```\n{got}\n```\n\nThat is not yet the total of the four prices. Look at step 3 of the recipe: what should happen to `total` in each round?",
                      "nl": "Je programma laat dit zien:\n\n```\n{got}\n```\n\nDat is nog niet het totaal van de vier prijzen. Kijk naar stap 3 van het recept: wat moet er elke ronde met `total` gebeuren?"},
            "other": {"en": "Right for these prices. But with {where} instead, your program shows `{got}`, and it should show `{want}`. Let the loop add up the prices in `prices`; do not type the total yourself.",
                      "nl": "Goed voor deze prijzen. Maar met {where} laat je programma `{got}` zien, en het zou `{want}` moeten zijn. Laat de loop de prijzen in `prices` optellen; typ het totaal niet zelf."},
            "other_error": {"en": "Right for these prices. But with {where} instead, your program stops with an error: it asks for an item the list does not have. Let the loop go through `prices`, so it works however many prices there are.",
                            "nl": "Goed voor deze prijzen. Maar met {where} stopt je programma met een fout: het vraagt om een item dat de lijst niet heeft. Laat de loop `prices` langsgaan, dan werkt het voor elk aantal prijzen."},
            "correct": {"en": "Your program shows `{got}`, the total of the four prices, and it also works for other lists, even an empty one.",
                        "nl": "Je programma laat `{got}` zien, het totaal van de vier prijzen, en het werkt ook voor andere lijsten, zelfs een lege."},
        },
    ),
    "frost": CodeBox(
        name="frost",
        start="temps = [12, -3, 5, -1, 0, 8, 7]\nfrost_days = 0\n...\nprint(frost_days)",
        solution="temps = [12, -3, 5, -1, 0, 8, 7]\nfrost_days = 0\nfor temp in temps:\n    if temp < 0:\n        frost_days = frost_days + 1\nprint(frost_days)",
        outcome="stdout",
        compare=same_lines,
        others=[{"temps": [-4, 3, -0.5, -7]}, {"temps": [5, 6]}],
        keep=["temps = [12, -3, 5, -1, 0, 8, 7]", "frost_days = 0"],
        mistakes=_SYNTAX + [
            Mistake("whole-list", syntax=r"\btemps\s*[<>]",
                    en="Inside the loop, `temp` is one temperature; `temps` is the whole list. Compare `temp`, the temperature of this round, with 0.",
                    nl="In de loop is `temp` één temperatuur; `temps` is de hele lijst. Vergelijk `temp`, de temperatuur van deze ronde, met 0."),
            _DOTS_RAN, _RESET, _PRINT_INSIDE,
            Mistake("twice", matches=_repeated(2),
                    en="Your program shows the number of frost days more than once. Keep one `print(frost_days)`, at the left edge, after the loop.",
                    nl="Je programma laat het aantal vorstdagen meer dan eens zien. Houd één `print(frost_days)` over, helemaal links, na de loop."),
            Mistake("with-text", matches=_with_text(2),
                    en="Your program has the right count, but the check reads only the number: keep the last line as `print(frost_days)`.",
                    nl="Je programma heeft de goede telling, maar de check leest alleen het getal: houd de laatste regel `print(frost_days)`."),
            Mistake("wrong-way", matches=_shows(4, 5),
                    en="Your program counts the days above zero. `<` means less than: `temp < 0` is `True` for a temperature below zero.",
                    nl="Je programma telt de dagen boven nul. `<` betekent kleiner dan: `temp < 0` is `True` bij een temperatuur onder nul."),
            Mistake("zero-counts", matches=_shows(3),
                    en="Your program counts 3 frost days: it also counts the day with 0 degrees. Below zero means less than 0, so `< 0`, not `<= 0`.",
                    nl="Je programma telt 3 vorstdagen: het telt ook de dag met 0 graden. Onder nul betekent kleiner dan 0, dus `< 0`, niet `<= 0`."),
            Mistake("added-temps", matches=_shows(-4),
                    en="Your program shows -4: it added the temperatures of the frost days. Count the days instead: add 1 for each day below zero.",
                    nl="Je programma laat -4 zien: het telde de temperaturen van de vorstdagen op. Tel de dagen: tel 1 op voor elke dag onder nul."),
            Mistake("all-days", matches=_shows(7),
                    en="Your program counts all 7 days. Only count a day when its temperature is below zero: put an `if` inside the loop.",
                    nl="Je programma telt alle 7 de dagen. Tel een dag alleen als de temperatuur onder nul is: zet een `if` in de loop."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a loop over `temps` that adds 1 to `frost_days` for each day below zero.",
                      "nl": "Vervang de `...` door een loop over `temps` die 1 bij `frost_days` optelt voor elke dag onder nul."},
            "cleared": {"en": "There is no code in the box any more. Type `temps = [12, -3, 5, -1, 0, 8, 7]` and `frost_days = 0` back at the top, and `print(frost_days)` at the bottom.",
                        "nl": "Er staat geen code meer in het vak. Typ `temps = [12, -3, 5, -1, 0, 8, 7]` en `frost_days = 0` opnieuw bovenaan, en `print(frost_days)` onderaan."},
            "wrong": {"en": "Your program shows:\n\n```\n{got}\n```\n\nThere are 2 days below zero. Look at the worked example: a loop over the list, and inside it an `if` with a line under it that adds 1.",
                      "nl": "Je programma laat dit zien:\n\n```\n{got}\n```\n\nEr zijn 2 dagen onder nul. Kijk naar het uitgewerkte voorbeeld: een loop over de lijst, met daarin een `if` die 1 optelt."},
            "other": {"en": "Right for these temperatures. But with {where} instead, your program shows `{got}`, and it should show `{want}`. Let the loop count the days in `temps`; do not type the number yourself.",
                      "nl": "Goed voor deze temperaturen. Maar met {where} laat je programma `{got}` zien, en het zou `{want}` moeten zijn. Laat de loop de dagen in `temps` tellen; typ het aantal niet zelf."},
            "other_error": {"en": "Right for these temperatures. But with {where} instead, your program stops with an error: it asks for an item the list does not have. Let the loop go through `temps`, so it works however many days there are.",
                            "nl": "Goed voor deze temperaturen. Maar met {where} stopt je programma met een fout: het vraagt om een item dat de lijst niet heeft. Laat de loop `temps` langsgaan, dan werkt het voor elk aantal dagen."},
            "correct": {"en": "Your program counts 2 frost days, and it also works for other lists of temperatures.",
                        "nl": "Je programma telt 2 vorstdagen, en het werkt ook voor andere lijsten met temperaturen."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_P = "prices = [2.5, 4, 1.5, 3]\ntotal = 0\n"
_F = "temps = [12, -3, 5, -1, 0, 8, 7]\nfrost_days = 0\n"
CASES = [
    ("total", _P + "for price in prices:\n    ...\nprint(total)", "empty", None),
    ("total", "", "empty", None),
    ("total", _P + "for price in prices:\n    total = total + price\nprint(total)", "correct", None),
    ("total", _P + "for price in prices:\n  total = total + price\nprint(total)", "correct", None),
    ("total", _P + "for p in prices:\n    total += p\nprint(total)", "correct", None),
    ("total", _P + "total = 11\nprint(total)", "almost", "other-value"),
    ("total", _P + "for price in prices:\n    total = price\nprint(total)", "wrong", "last-only"),
    ("total", _P + "for price in prices:\n    total + price\nprint(total)", "wrong", "not-kept"),
    ("total", _P + "for price in prices:\n    total = total + price\n    print(total)", "wrong", "print-inside"),
    ("total", _P + "for price in prices:\n    total = 0\n    total = total + price\nprint(total)", "wrong", "reset-inside"),
    ("total", _P + "for price in prices:\n    total = total + prices\nprint(total)", "error", "whole-list"),
    ("total", _P + "for price in prices\n    total = total + price\nprint(total)", "error", "colon-missing"),
    ("total", _P + "For price in prices:\n    total = total + price\nprint(total)", "error", "capital"),
    ("frost", _F + "...\nprint(frost_days)", "empty", None),
    ("frost", _F + "for temp in temps:\n    if temp < 0:\n        frost_days = frost_days + 1\nprint(frost_days)", "correct", None),
    ("frost", _F + "for t in temps:\n  if t < 0:\n    frost_days += 1\nprint(frost_days)", "correct", None),
    ("frost", _F + "frost_days = 2\nprint(frost_days)", "almost", "other-value"),
    ("frost", _F + "for temp in temps:\n    if temp <= 0:\n        frost_days = frost_days + 1\nprint(frost_days)", "wrong", "zero-counts"),
    ("frost", _F + "for temp in temps:\n    if temp < 0:\n        frost_days = frost_days + temp\nprint(frost_days)", "wrong", "added-temps"),
    ("frost", _F + "for temp in temps:\n    frost_days = frost_days + 1\nprint(frost_days)", "wrong", "all-days"),
    ("frost", _F + "for temp in temps:\n    if temp < 0:\n        frost_days = frost_days + 1\n        print(frost_days)", "wrong", "print-inside"),
    ("frost", _F + "for temp in temps:\n    if temp < 0\n        frost_days = frost_days + 1\nprint(frost_days)", "error", "colon-missing"),
    ("frost", _F + "for temp in temps:\n    if temp > 0:\n        frost_days = frost_days + 1\nprint(frost_days)", "wrong", "wrong-way"),
    ("frost", _F + "for temp in temps:\n    if temps < 0:\n        frost_days = frost_days + 1\nprint(frost_days)", "error", "whole-list"),
    ("frost", _F + "for temp in temps:\n    if temp <= -1:\n        frost_days = frost_days + 1\nprint(frost_days)", "almost", "other-value"),
    ("frost", _F + "for temp in temps:\n    if temp < 0:\n        frost_days = frost_days + 1\nprint(f\"{frost_days} frost days\")", "wrong", "with-text"),
    ("total", _P + "for price in prices:\n    total = total + price\n    print(total)\nprint(total)", "wrong", "print-inside"),
    ("total", _P + "for price in prices:\n    total = total + price\nprint(total)\nprint(total)", "wrong", "twice"),
    ("total", _P + "for price in prices:\n    total = total + price\nprint(f\"Total: {total}\")", "wrong", "with-text"),
    ("total", _P + "total = prices[0] + prices[1] + prices[2] + prices[3]\nprint(total)", "almost", "other-value"),
    ("total", _P + "for i in range(prices):\n    total = total + i\nprint(total)", "error", "range-of-list"),
]
