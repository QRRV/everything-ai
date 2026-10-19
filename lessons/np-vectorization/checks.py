# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson np-vectorization: prices raised by 10% without a loop, and the speed of each trip from
two arrays. And the rows in the picture, and the measurement of a loop against NumPy."""
import re
import time

from everything_ai.check import Code, CodeBox, FunctionBox, Mistake, uses_loop

# The picture: two arrays, and what +, and * with a number, make of them. Exercise 1 asks about other numbers.
A, B = [1, 2, 3], [10, 20, 30]


def picture_rows(notes: dict[str, str]) -> list[tuple[str, list[str], str, bool]]:
    return [
        ("a", [str(x) for x in A], "", True),
        ("b", [str(x) for x in B], "", True),
        ("a + b", [str(x + y) for x, y in zip(A, B)], notes["plus"], True),
        ("a * 2", [str(x * 2) for x in A], notes["times"], True),
    ]


def measure(n: int = 1_000_000, tries: int = 3) -> tuple[float, float]:
    """Seconds for the squares of 0 to n - 1, with a Python loop and with NumPy: the fastest of a few tries,
    so that the first time NumPy wakes up does not count."""
    import numpy as np

    def loop():
        squares = []
        for i in range(n):
            squares.append(i * i)

    def whole():
        np.arange(n) ** 2

    def fastest(fn):
        times = []
        for _ in range(tries):
            start = time.perf_counter()
            fn()
            times.append(time.perf_counter() - start)
        return min(times)

    return fastest(loop), fastest(whole)


def decimal(x: float, places: int, lang: str) -> str:
    """A number with this many decimals, the way the language writes it: 1.5 in English, 1,5 in Dutch."""
    text = f"{x:.{places}f}"
    return text.replace(".", ",") if lang == "nl" else text


# The lesson is done when these are correct. "add-arrays" is the choice question in the lesson.
REQUIRED = ["add-arrays", "new-prices", "speeds"]

_PRICES_LINE = "prices = np.array([2.5, 1.0, 4.0, 12.0])"
_PRICES_START = "import numpy as np\n\n" + _PRICES_LINE + "\nnew_prices = ...\nnew_prices\n"
_PRICES = "import numpy as np\n\n" + _PRICES_LINE + "\nnew_prices = prices * 1.1\nnew_prices"


def _gives(numbers):
    """The learner's last line gave these numbers, as an array or a list (within a small margin)."""
    def test(got, text):
        values = got.tolist() if type(got).__name__ == "ndarray" else got
        if not isinstance(values, list) or len(values) != len(numbers):
            return False
        return all(isinstance(x, (int, float)) and abs(x - y) < 1e-6 for x, y in zip(values, numbers))
    return test


def _no_prices_used(text) -> bool:
    """`prices` is used nowhere but on the line the box starts with."""
    return not re.search(r"\bprices\b", text.replace(_PRICES_LINE, ""))


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SPEEDS_START = "import numpy as np\n\n\ndef speeds(km, minutes):\n    km = np.array(km)\n    minutes = np.array(minutes)\n    ...\n"
_SPEEDS = ("import numpy as np\n\n\ndef speeds(km, minutes):\n    km = np.array(km)\n    minutes = np.array(minutes)\n"
           "    hours = minutes / 60\n    return km / hours")

EXERCISES = {
    "new-prices": CodeBox(
        name="new-prices",
        start=_PRICES_START,
        solution=_PRICES,
        keep=["import numpy as np", _PRICES_LINE],
        # The check also runs the box with other prices: numbers typed in by hand do not change with them.
        others=[{"prices": Code("np.array([10.0, 20.0, 5.0])")}],
        first=[_DOTS],
        mistakes=[
            Mistake("plus-ten", matches=_gives([12.5, 11.0, 14.0, 22.0]),
                    en="Every price went up by 10, not by 10%. 10% of a price is a tenth of it; the new price is the old price plus that.",
                    nl="Elke prijs ging 10 omhoog, niet 10%. 10% van een prijs is een tiende ervan; de nieuwe prijs is de oude plus dat."),
            Mistake("only-increase", matches=_gives([0.25, 0.1, 0.4, 1.2]),
                    en="These are the increases themselves: 10% of each price. The new price is the old price plus that increase.",
                    nl="Dit zijn alleen de verhogingen: 10% van elke prijs. De nieuwe prijs is de oude prijs plus die verhoging."),
            Mistake("times-ten", matches=_gives([25.0, 10.0, 40.0, 120.0]),
                    en="Every price is now ten times as high. 10% more is the price plus a tenth of it: what do you multiply a price by to get that?",
                    nl="Elke prijs is nu tien keer zo hoog. 10% meer is de prijs plus een tiende ervan: waarmee vermenigvuldig je een prijs om dat te krijgen?"),
            Mistake("decimal-comma", matches=lambda got, text: isinstance(got, tuple),
                    en="Your last line gives two values, because of a comma. Python writes decimal numbers with a point, not a comma: `1.1`, not `1,1`.",
                    nl="Je laatste regel geeft twee waarden, door een komma. Python schrijft kommagetallen met een punt, niet met een komma: `1.1`, niet `1,1`."),
        ],
        form=[
            Mistake("with-loop", state="almost", matches=lambda got, text: uses_loop(text),
                    en="Right numbers, but with a loop. This lesson is about doing without one, because that is shorter and much faster: write the calculation once, with the whole array `prices`.",
                    nl="De goede getallen, maar met een loop. Deze les gaat erover dat het zonder kan, omdat dat korter en veel sneller is: schrijf de berekening één keer, met de hele array `prices`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a calculation on `prices` that makes every price 10% higher.",
                      "nl": "Vervang de `...` door een berekening met `prices` die elke prijs 10% hoger maakt."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug."},
            "no_value": {"en": "The last line of the box shows nothing. Put `new_prices` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `new_prices` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives:\n\n```\n{got}\n```\n\nand it should be every price 10% higher: `2.5` becomes `2.75`. What do you multiply a price by, to make it 10% higher?",
                      "nl": "Je laatste regel geeft:\n\n```\n{got}\n```\n\nen dat moet elke prijs 10% hoger zijn: `2.5` wordt `2.75`. Waarmee vermenigvuldig je een prijs om hem 10% hoger te maken?"},
            "correct": {"en": "One calculation, and NumPy does it for every price: `2.5` becomes `2.75`, and `12` becomes `13.2`.",
                        "nl": "Eén berekening, en NumPy doet hem voor elke prijs: `2.5` wordt `2.75`, en `12` wordt `13.2`."},
        },
    ),
    "speeds": FunctionBox(
        name="speeds",
        function="speeds",
        start=_SPEEDS_START,
        solution=_SPEEDS,
        visible=[([10, 3, 42], [30, 15, 120])],
        hidden=[([5], [60]), ([1.5, 100], [6, 50]), ([2, 8], [10, 30]), ([10, 7], [45, 25])],
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("per-minute", wrong=_SPEEDS.replace("    hours = minutes / 60\n    return km / hours", "    return km / minutes"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is km per minute. An hour has 60 minutes: turn the minutes into hours first.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is km per minuut. Een uur heeft 60 minuten: maak eerst uren van de minuten."),
            Mistake("lists-not-arrays", error=lambda e: isinstance(e, TypeError) and "list" in str(e),
                    en="With `{call}`, your function calculates with the lists themselves. Lists do not divide number by number; arrays do. Keep the lines that make arrays of `km` and `minutes`, and calculate with those.",
                    nl="Met `{call}` rekent je functie met de lijsten zelf. Lijsten delen niet getal voor getal; arrays wel. Houd de regels die arrays maken van `km` en `minutes`, en reken daarmee."),
            Mistake("floor-division", matches=lambda got, text: "//" in text,
                    en="`{call}` gives `{got}`, and it should give `{want}`. `//` divides and then rounds down to a whole number. Divide with one `/`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. `//` deelt en rondt daarna af naar beneden, op een geheel getal. Deel met één `/`."),
            Mistake("upside-down", wrong=_SPEEDS.replace("return km / hours", "return hours / km"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is hours per km, upside down. A speed is the distance divided by the time.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is uren per km, omgekeerd. Een snelheid is de afstand gedeeld door de tijd."),
            Mistake("minutes-per-km", wrong=_SPEEDS.replace("    hours = minutes / 60\n    return km / hours", "    return minutes / km"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is minutes per km, upside down. A speed is the distance divided by the time.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is minuten per km, omgekeerd. Een snelheid is de afstand gedeeld door de tijd."),
            Mistake("times-sixty-wrong", wrong=_SPEEDS.replace("    hours = minutes / 60\n    return km / hours", "    return km / minutes / 60"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: 3600 times too small, because the 60 went the wrong way. 30 minutes is 0.5 hours: the hours are the minutes divided by 60.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: 3600 keer te klein, omdat de 60 de verkeerde kant op ging. 30 minuten is 0,5 uur: de uren zijn de minuten gedeeld door 60."),
        ],
        form=[
            Mistake("with-loop", state="almost", matches=lambda got, text: uses_loop(text),
                    en="Right in every case, but with a loop. This lesson is about doing without one, because that is shorter and much faster: write the calculation once, with the whole arrays.",
                    nl="Goed in elk geval, maar met een loop. Deze les gaat erover dat het zonder kan, omdat dat korter en veel sneller is: schrijf de berekening één keer, met de hele arrays."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the calculation of the speed of every trip, in km per hour.",
                      "nl": "Vervang de `...` door de berekening van de snelheid van elke rit, in km per uur."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the distances and the minutes, and the `def` line of your function has another number of names between its parentheses. Keep `(km, minutes)`.",
                          "nl": "De check roept `{call}` aan, met de afstanden en de minuten, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(km, minutes)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. A speed is the distance divided by the time; is your time in hours?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Een snelheid is de afstand gedeeld door de tijd; is jouw tijd in uren?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for every trip?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor elke rit?"},
            "correct": {"en": "Your function is right in all {n} cases. Two arrays of the same shape go together number by number: each distance with its own time.",
                        "nl": "Je functie klopt in alle {n} gevallen. Twee arrays met dezelfde shape gaan getal voor getal samen: elke afstand met zijn eigen tijd."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_P = "import numpy as np\n\n" + _PRICES_LINE + "\n"
_S = "import numpy as np\n\n\ndef speeds(km, minutes):\n    km = np.array(km)\n    minutes = np.array(minutes)\n"
CASES = [
    ("new-prices", _PRICES_START, "empty", None),
    ("new-prices", _PRICES, "correct", None),
    ("new-prices", _P + "new_prices = prices + prices * 0.1\nnew_prices", "correct", None),
    ("new-prices", _P + "new_prices = prices * 110 / 100\nnew_prices", "correct", None),
    ("new-prices", _P + "new_prices = prices + 10\nnew_prices", "wrong", "plus-ten"),
    ("new-prices", _P + "new_prices = prices * 0.1\nnew_prices", "wrong", "only-increase"),
    ("new-prices", _P + "new_prices = prices * 10\nnew_prices", "wrong", "times-ten"),
    ("new-prices", _P + "new_prices = []\nfor p in prices:\n    new_prices.append(p * 1.1)\nnew_prices", "almost", "with-loop"),
    ("new-prices", _P + "new_prices = [p * 1.1 for p in prices]\nnew_prices", "almost", "with-loop"),
    ("new-prices", _P + "new_prices = np.array([2.75, 1.1, 4.4, 13.2])\nnew_prices", "almost", "other-value"),
    ("new-prices", _P + "new = []\nfor p in prices:\n    new.append(p * 1.1)\nnew_prices = np.array(new)\nnew_prices", "almost", "with-loop"),
    ("new-prices", _P + "new_prices = prices * 1.2\nnew_prices", "wrong", None),
    ("new-prices", _P + "# 10% more for every price\nnew_prices = prices * 1.1\nnew_prices", "correct", None),
    ("new-prices", _P + "higher = prices * 1.1\nnew_prices = higher\nnew_prices", "correct", None),
    ("new-prices", _P + "new_prices = prices * 1,1\nnew_prices", "wrong", "decimal-comma"),
    ("new-prices", _P + "new_prices = prices * 1.1\nprint(new_prices)", "wrong", "print-last"),
    ("speeds", _SPEEDS_START, "empty", None),
    ("speeds", _SPEEDS, "correct", None),
    ("speeds", _S + "    return km / minutes * 60", "correct", None),
    ("speeds", _S + "    return 60 * km / minutes", "correct", None),
    ("speeds", _S + "    return km / minutes", "wrong", "per-minute"),
    ("speeds", _S + "    return minutes / 60 / km", "wrong", "upside-down"),
    ("speeds", _S + "    return km / minutes / 60", "wrong", "times-sixty-wrong"),
    ("speeds", _S + "    hours = minutes * 60\n    return km / hours", "wrong", "times-sixty-wrong"),
    ("speeds", _S + "    return minutes / km", "wrong", "minutes-per-km"),
    ("speeds", _S + "    return 60 * km // minutes", "wrong", "floor-division"),
    ("speeds", "import numpy as np\n\n\ndef speeds(km, minutes):\n    hours = [m / 60 for m in minutes]\n    return km / hours", "error", "lists-not-arrays"),
    ("speeds", _S + "    # km per hour for every trip\n    return km / (minutes / 60)", "correct", None),
    ("speeds", _S + "    out = []\n    for i in range(len(km)):\n        out.append(km[i] / minutes[i] * 60)\n    return out", "almost", "with-loop"),
    ("speeds", _S + "    out = []\n    for i in range(len(km)):\n        out.append(km[i] / minutes[i] * 60)\n    return np.array(out)", "almost", "with-loop"),
]
