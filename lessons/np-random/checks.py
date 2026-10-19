# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson np-random: twenty dice rolls from a random number generator (RNG) with a seed, and a
list in a random order that is the same for the same seed. And the rows in the picture and the order in the
text of exercise 3, worked out by NumPy itself when the lesson opens, so they always match the learner's NumPy."""
import re

from everything_ai.check import CodeBox, FunctionBox, Mistake, _as_code

NAMES = ["Ada", "Bo", "Cas", "Dex", "Eva"]


def picture_rows(notes: dict[str, str]) -> list[tuple[str, list[str], str, bool]]:
    """Six dice rolls from RNGs: two new ones with seed 7, the first one asked again, and one with seed 8."""
    import numpy as np

    def six(rng):
        return [str(x) for x in rng.integers(1, 7, size=6)]
    first = np.random.default_rng(7)
    rows = [("default_rng(7)", six(first), notes["first"], True),
            ("default_rng(7)", six(np.random.default_rng(7)), notes["same"], True)]
    rows.append((notes["asked"], six(first), notes["again"], True))
    rows.append(("default_rng(8)", six(np.random.default_rng(8)), notes["other"], True))
    return rows


def shuffled_example() -> str:
    """What `shuffled(NAMES, 7)` gives with the learner's NumPy, as the lesson writes it."""
    import numpy as np
    return _shown(np.random.default_rng(7).permutation(NAMES))


# The lesson is done when these are correct. "one-number" is the choice question in the lesson.
REQUIRED = ["one-number", "rolls", "shuffled"]

_ROLLS_START = "import numpy as np\n\n...\nrolls\n"
_ROLLS = "import numpy as np\n\nrng = np.random.default_rng(2026)\nrolls = rng.integers(1, 7, size=20)\nrolls"


def _rolls(seed, low, high, n=20):
    import numpy as np
    return np.random.default_rng(seed).integers(low, high, size=n).tolist()


def _shown(value) -> str:
    """An array as a cell shows it, but without NumPy's dtype note for text, such as `dtype='<U3'`."""
    if type(value).__name__ == "ndarray":
        return f"array({_as_code(value.tolist())})"
    return _as_code(value) if isinstance(value, (str, list)) else repr(value)


def _values(got):
    return got.tolist() if type(got).__name__ == "ndarray" else got


def _gives(numbers):
    return lambda got, text: _values(got) == numbers


def _off_die(got, text) -> bool:
    """Whole numbers, but some below 1 or above 6: not a die."""
    values = _values(got)
    flat = [x for row in values for x in (row if isinstance(row, list) else [row])] if isinstance(values, list) else []
    return bool(flat) and all(isinstance(x, int) for x in flat) and any(x < 1 or x > 6 for x in flat)


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_OLD_SEED = Mistake("old-seed", matches=lambda got, text: re.search(r"np\.random\.(seed|randint|shuffle|permutation)\b", text) is not None,
                    en="That is NumPy's old way, with one hidden RNG for the whole program. Make your own random number generator with `np.random.default_rng`, and ask it for the numbers.",
                    nl="Dat is de oude manier van NumPy, met één verborgen RNG voor het hele programma. Maak je eigen random number generator met `np.random.default_rng`, en vraag hem om de getallen.")
_PYTHON_RANDOM = Mistake("python-random", matches=lambda got, text: re.search(r"^\s*import random\b|\brandom\.(randint|shuffle|sample|seed)\b", text, re.MULTILINE) is not None,
                         en="That is Python's own `random` module. In this lesson, the random numbers come from NumPy: `rng = np.random.default_rng(...)`, and then ask `rng`.",
                         nl="Dat is Pythons eigen module `random`. In deze les komen de willekeurige getallen van NumPy: `rng = np.random.default_rng(...)`, en dan vraag je het aan `rng`.")

_SHUFFLED_START = "import numpy as np\n\n\ndef shuffled(items, seed):\n    ...\n"
_SHUFFLED = ("import numpy as np\n\n\ndef shuffled(items, seed):\n    rng = np.random.default_rng(seed)\n"
             "    return rng.permutation(items)")

EXERCISES = {
    "rolls": CodeBox(
        name="rolls",
        start=_ROLLS_START,
        solution=_ROLLS,
        first=[_DOTS],
        mistakes=[
            _OLD_SEED, _PYTHON_RANDOM,
            Mistake("high-six", matches=_gives(_rolls(2026, 1, 6)),
                    en="Your rolls never give a 6: `integers` stops just before its second number. For 1 up to and including 6, the second number is 7.",
                    nl="Je worpen geven nooit een 6: `integers` stopt net voor zijn tweede getal. Voor 1 tot en met 6 is het tweede getal 7."),
            Mistake("from-zero", matches=_gives(_rolls(2026, 0, 6)),
                    en="Your rolls go from 0 to 5: `integers(0, 6)` starts at 0. A die starts at 1.",
                    nl="Je worpen gaan van 0 tot en met 5: `integers(0, 6)` begint bij 0. Een dobbelsteen begint bij 1."),
            Mistake("off-die", matches=_off_die,
                    en="Some of your rolls are below 1 or above 6, so they are not rolls of a die. Which two numbers give 1 up to and including 6, when the second one is not included?",
                    nl="Een paar van je worpen liggen onder 1 of boven 6, dus zijn het geen dobbelsteenworpen. Welke twee getallen geven 1 tot en met 6, als het tweede niet meedoet?"),
            Mistake("no-seed", matches=lambda got, text: re.search(r"default_rng\(\s*\)", text) is not None,
                    en="Your RNG has no seed, so it gives different rolls every time you run the box. Give it the seed 2026: `np.random.default_rng(2026)`.",
                    nl="Je RNG heeft geen seed, dus geeft hij elke keer dat je het vak uitvoert andere worpen. Geef hem de seed 2026: `np.random.default_rng(2026)`."),
            Mistake("other-seed", matches=lambda got, text: re.search(r"default_rng\(\s*[^)\s]", text) is not None and "2026" not in text,
                    en="These rolls come from another seed. The check expects the rolls of seed 2026, so that everyone gets the same ones: `np.random.default_rng(2026)`.",
                    nl="Deze worpen komen van een andere seed. De check verwacht de worpen van seed 2026, zodat iedereen dezelfde krijgt: `np.random.default_rng(2026)`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a random number generator with seed 2026, and twenty dice rolls from it, called `rolls`.",
                      "nl": "Vervang de `...` door een random number generator met seed 2026, en twintig dobbelsteenworpen daaruit, met de naam `rolls`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "no_value": {"en": "The last line of the box shows nothing. Put `rolls` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `rolls` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives:\n\n```\n{got}\n```\n\nand it should be twenty rolls, 1 to 6, from one RNG with seed 2026. Do you make the RNG once, with seed 2026, and ask it for twenty numbers?",
                      "nl": "Je laatste regel geeft:\n\n```\n{got}\n```\n\nen dat moeten twintig worpen zijn, 1 tot en met 6, van één RNG met seed 2026. Maak je de RNG één keer, met seed 2026, en vraag je hem om twintig getallen?"},
            "correct": {"en": "Twenty rolls from seed 2026: anyone who runs this code, with the same version of NumPy, gets the same twenty. Run the box again: still the same.",
                        "nl": "Twintig worpen van seed 2026: iedereen die deze code uitvoert, met dezelfde versie van NumPy, krijgt dezelfde twintig. Voer het vak nog eens uit: nog steeds dezelfde."},
        },
    ),
    "shuffled": FunctionBox(
        name="shuffled",
        function="shuffled",
        start=_SHUFFLED_START,
        solution=_SHUFFLED,
        visible=[(NAMES, 7)],
        hidden=[(NAMES, 8), ([1, 2, 3, 4, 5, 6, 7, 8], 1), (["x"], 3), ([10, 20, 30], 2026)],
        show=_shown,
        first=[_DOTS],
        mistakes=[
            _OLD_SEED, _PYTHON_RANDOM,
            Mistake("seed-outside", error=lambda e: isinstance(e, NameError) and "'seed'" in str(e),
                    en="Python does not know `seed` where you use it: `seed` exists only inside the function, as the value it gets. Make the RNG inside `shuffled`.",
                    nl="Python kent `seed` niet waar je hem gebruikt: `seed` bestaat alleen binnen de functie, als de waarde die hij krijgt. Maak de RNG binnen `shuffled`."),
            Mistake("rng-outside", state="almost", matches=lambda got, text: re.search(r"^\S.*default_rng\(", text, re.MULTILINE) is not None,
                    en="`{call}` gives `{got}`, and it should give `{want}`. Your RNG is made once, outside the function, so it ignores `seed`, and it keeps going with every call. Make it inside the function, with `seed`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Je RNG wordt één keer gemaakt, buiten de functie, dus negeert hij `seed`, en gaat hij bij elke aanroep verder. Maak hem binnen de functie, met `seed`."),
            Mistake("no-seed", matches=lambda got, text: re.search(r"default_rng\(\s*\)", text) is not None,
                    en="`{call}` gives `{got}`, and it should give `{want}`: your RNG has no seed, so the order changes every time. Make it with the seed the function gets: `seed`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je RNG heeft geen seed, dus verandert de volgorde elke keer. Maak hem met de seed die de functie krijgt: `seed`."),
            Mistake("fixed-seed", state="almost", matches=lambda got, text: re.search(r"default_rng\(\s*\d", text) is not None,
                    en="`{call}` gives `{got}`, and it should give `{want}`: your RNG always uses the same number as its seed. Give it the seed the function gets: `seed`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je RNG gebruikt altijd hetzelfde getal als seed. Geef hem de seed die de functie krijgt: `seed`."),
            Mistake("same-order", wrong="def shuffled(items, seed):\n    return items",
                    en="`{call}` gives the items in the order they came in, `{got}`. Ask the RNG for a random order.",
                    nl="`{call}` geeft de items in hun oorspronkelijke volgorde, `{got}`. Vraag de RNG om een willekeurige volgorde."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a random number generator made with `seed`, and give back `items` in a random order.",
                      "nl": "Vervang de `...` door een random number generator gemaakt met `seed`, en geef `items` terug in een willekeurige volgorde."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the items and a seed, and the `def` line of your function has a different number of names between its parentheses. Keep `(items, seed)`.",
                          "nl": "De check roept `{call}` aan, met de items en een seed, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(items, seed)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you make one RNG with `seed`, and ask it for the items in a random order?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Maak je één RNG met `seed`, en vraag je hem om de items in een willekeurige volgorde?"},
            "unreached": {"en": "With `{call}`, your function gives back `None`, nothing. Did you use `rng.shuffle`? It shuffles the list itself and gives back nothing; `rng.permutation(items)` gives back the items in a new, random order. Otherwise: does every way through the function end with `return` and a value?",
                          "nl": "Met `{call}` geeft je functie `None` terug, niets. Gebruikte je `rng.shuffle`? Die schudt de lijst zelf en geeft niets terug; `rng.permutation(items)` geeft de items terug in een nieuwe, willekeurige volgorde. Anders: eindigt elke weg door de functie met `return` en een waarde?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for every list and every seed?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor elke lijst en elke seed?"},
            "correct": {"en": "Your function is right in all {n} cases. The same seed gives the same order every time, and another seed usually a different one.",
                        "nl": "Je functie klopt in alle {n} gevallen. Dezelfde seed geeft elke keer dezelfde volgorde, en een andere seed meestal een andere."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_I = "import numpy as np\n\n"
_F = "import numpy as np\n\n\ndef shuffled(items, seed):\n"
CASES = [
    ("rolls", _ROLLS_START, "empty", None),
    ("rolls", _ROLLS, "correct", None),
    ("rolls", _I + "rolls = np.random.default_rng(2026).integers(1, 7, size=20)\nrolls", "correct", None),
    ("rolls", _I + "rng = np.random.default_rng(2026)\nrolls = rng.integers(1, 6, size=20)\nrolls", "wrong", "high-six"),
    ("rolls", _I + "rng = np.random.default_rng(2026)\nrolls = rng.integers(0, 6, size=20)\nrolls", "wrong", "from-zero"),
    ("rolls", _I + "rng = np.random.default_rng(2026)\nrolls = rng.integers(0, 7, size=20)\nrolls", "wrong", "off-die"),
    ("rolls", _I + "rng = np.random.default_rng(42)\nrolls = rng.integers(1, 7, size=20)\nrolls", "wrong", "other-seed"),
    ("rolls", _I + "rng = np.random.default_rng()\nrolls = rng.integers(1, 7, size=20)\nrolls", "wrong", "no-seed"),
    ("rolls", _I + "np.random.seed(2026)\nrolls = np.random.randint(1, 7, size=20)\nrolls", "wrong", "old-seed"),
    ("rolls", _I + "import random\nrandom.seed(2026)\nrolls = [random.randint(1, 6) for i in range(20)]\nrolls", "wrong", "python-random"),
    ("rolls", _I + "rng = np.random.default_rng(2026)\nrolls = rng.integers(1, 7, size=20)\nprint(rolls)", "wrong", "print-last"),
    ("shuffled", _SHUFFLED_START, "empty", None),
    ("shuffled", _SHUFFLED, "correct", None),
    ("shuffled", _F + "    return np.random.default_rng(seed).permutation(items)", "correct", None),
    ("shuffled", _F + "    rng = np.random.default_rng(seed)\n    rng.shuffle(items)\n    return items", "correct", None),
    ("shuffled", _SHUFFLED.replace("default_rng(seed)", "default_rng(7)"), "almost", "fixed-seed"),
    ("shuffled", _SHUFFLED.replace("default_rng(seed)", "default_rng(42)"), "almost", "fixed-seed"),
    ("shuffled", _SHUFFLED.replace("default_rng(seed)", "default_rng()"), "wrong", "no-seed"),
    ("shuffled", _F + "    return items", "wrong", "same-order"),
    ("shuffled", _F + "    rng = np.random.default_rng(seed)\n    return rng.shuffle(items)", "wrong", "unreached"),
    ("shuffled", "import numpy as np\n\nrng = np.random.default_rng(seed)\n\n\ndef shuffled(items, seed):\n    return rng.permutation(items)", "error", "seed-outside"),
    ("shuffled", "import numpy as np\n\nrng = np.random.default_rng(7)\n\n\ndef shuffled(items, seed):\n    return rng.permutation(items)", "almost", "rng-outside"),
]
