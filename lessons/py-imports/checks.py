# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson py-imports: a function with math, and one with random and a seed."""
import math
import re

from everything_ai.check import FunctionBox, Mistake


def same_pi(a, b) -> bool:
    """Much tighter than the usual margin: 3.14159 is close to pi, but it is not `math.pi` (template 4.3)."""
    try:
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    except TypeError:
        return a == b


# A module's names belong to the module: giving one a new value with = would change it for the whole lesson.
_ASSIGN_TO_MODULE = Mistake(
    "assign-to-module", matches=lambda got, text: re.search(r"^[^#\n]*\b(?:math|random)\s*\.\s*\w+\s*=(?!=)", text, re.MULTILINE) is not None,
    en="`random.seed` and `math.pi` belong to their module: you use them, you do not give them a new value with `=`. A function from a module is called with parentheses, such as `random.seed(seed)`.",
    nl="`random.seed` en `math.pi` horen bij hun module: je gebruikt ze, je geeft ze geen nieuwe waarde met `=`. Een functie uit een module roep je aan met haakjes, zoals `random.seed(seed)`.")
_NO_IMPORT = Mistake(
    "no-import", syntax=r"\A(?![\s\S]*^\s*(?:import|from)\s)",
    en="Your box has no `import` line. The check runs the box on its own, so it needs its own `import` at the top, even when a cell of the lesson already imports the module.",
    nl="Je vak heeft geen regel met `import`. De check voert het vak los uit, dus het heeft een eigen `import` bovenaan nodig, ook als een cel van de les de module al binnenhaalt.")
_SYNTAX = [
    Mistake("dots-left", syntax=r"^[^#\n]*\.\.\.",
            en="There is still a `...` in your function. Replace it with your own lines.",
            nl="Er staat nog een `...` in je functie. Vervang die door je eigen regels."),
    Mistake("curly-quote", syntax=r"[“”‘’„]",
            en="Your code has a curly quote, like `“`. Python only reads straight quotes, like `\"`: type the quotes again in the box.",
            nl="Je code heeft een gekruld aanhalingsteken, zoals `“`. Python leest alleen rechte, zoals `\"`: typ de aanhalingstekens opnieuw in het vak."),
    Mistake("module-capital", syntax=r"\bimport\s+(Math|Random)\b|\b(Math|Random)\s*\.",
            en="Module names are written in small letters: `math` and `random`.",
            nl="Namen van modules schrijf je met kleine letters: `math` en `random`."),
    _NO_IMPORT,
]

# The lesson is done when these are correct. "import-name" is the choice question in the lesson.
REQUIRED = ["circle", "import-name", "pick"]

_CIRCLE = "import math\n\ndef circle_area(r):\n    return math.pi * r ** 2"
_PICK = "import random\n\ndef pick(items, seed):\n    random.seed(seed)\n    return random.choice(items)"

EXERCISES = {
    "circle": FunctionBox(
        name="circle",
        function="circle_area",
        start="import math\n\ndef circle_area(r):\n    ...\n",
        solution=_CIRCLE,
        visible=[(3,)],
        hidden=[(1,), (2,), (0,), (0.5,), (10,)],
        compare=same_pi,
        first=[_ASSIGN_TO_MODULE],
        mistakes=_SYNTAX + [
            Mistake("pi-capital", syntax=r"\bmath\s*\.\s*PI\b",
                    en="Inside `math`, pi is written in small letters: `math.pi`.",
                    nl="In `math` schrijf je pi met kleine letters: `math.pi`."),
            Mistake("pi-called", syntax=r"\bmath\s*\.\s*pi\s*\(",
                    en="`math.pi` is a number, not a function: write it without parentheses.",
                    nl="`math.pi` is een getal, geen functie: schrijf het zonder haakjes."),
            Mistake("no-prefix", syntax=r"\A(?![\s\S]*^\s*from\s+math\s+import)[\s\S]*^[^#\n]*(?<![\w.])(?:pi\b|sqrt\s*\()",
                    en="After `import math`, a name from the module needs `math.` in front: `math.pi`.",
                    nl="Na `import math` zet je voor een naam uit de module `math.`: `math.pi`."),
            Mistake("rounded-pi", state="almost", matches=lambda got, text: re.search(r"3\.14|22\s*/\s*7", text) is not None,
                    en="`{call}` gives `{got}`: close, but that pi is rounded. `math.pi` is much more precise: use that.",
                    nl="`{call}` geeft `{got}`: dichtbij, maar die pi is afgerond. `math.pi` is veel preciezer: gebruik die."),
            Mistake("caret", matches=lambda got, text: re.search(r"\^", text) is not None,
                    en="`{call}` gives `{got}`. In Python, `^` is not a power: write `r ** 2` for `r` squared.",
                    nl="`{call}` geeft `{got}`. In Python is `^` geen macht: schrijf `r ** 2` voor `r` in het kwadraat."),
            Mistake("circumference", wrong="import math\n\ndef circle_area(r):\n    return 2 * math.pi * r",
                    en="`{call}` gives `{got}`: 2 times pi times r, the distance around the circle; `r * 2` doubles `r`. The area is pi times r squared: `r ** 2`, or `r * r`.",
                    nl="`{call}` geeft `{got}`: 2 keer pi keer r, de omtrek van de cirkel; `r * 2` verdubbelt `r`. De oppervlakte is pi keer r in het kwadraat: `r ** 2`, of `r * r`."),
            Mistake("no-square", wrong="import math\n\ndef circle_area(r):\n    return math.pi * r",
                    en="`{call}` gives `{got}`: pi times r. The area is pi times r squared: `r ** 2`, or `r * r`.",
                    nl="`{call}` geeft `{got}`: pi keer r. De oppervlakte is pi keer r in het kwadraat: `r ** 2`, of `r * r`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a `return` of the area: pi from `math`, times `r` squared.",
                      "nl": "Vervang de `...` door een `return` met de oppervlakte: pi uit `math`, keer `r` in het kwadraat."},
            "cleared": {"en": "There is no code in the box any more. Start again with `import math`, then `def circle_area(r):`, and under it, indented, your line.",
                        "nl": "Er staat geen code meer in het vak. Begin opnieuw met `import math`, dan `def circle_area(r):`, met daaronder, ingesprongen, je regel."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: pi times 3 times 3. Look at step 2 of the recipe for pi, and square `r`.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: pi keer 3 keer 3. Kijk naar stap 2 van het recept voor pi, en kwadrateer `r`."},
            "almost": {"en": "Right for a radius of 3. But `{call}` gives `{got}`, and it should give `{want}`. Does your function square `r` for every radius?",
                       "nl": "Goed voor een straal van 3. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Kwadrateert je functie `r` voor elke straal?"},
            "correct": {"en": "Your function is right in all {n} cases, with the precise pi from `math`.",
                        "nl": "Je functie klopt in alle {n} gevallen, met de precieze pi uit `math`."},
        },
    ),
    "pick": FunctionBox(
        name="pick",
        function="pick",
        start="import random\n\ndef pick(items, seed):\n    ...\n",
        solution=_PICK,
        visible=[(["rock", "paper", "scissors"], 1)],
        hidden=[(["a", "b", "c", "d"], 7), (["x"], 3), ([10, 20, 30], 42), (["a", "b", "c", "d"], 8)],
        first=[_ASSIGN_TO_MODULE],
        mistakes=_SYNTAX + [
            Mistake("shadowed-seed", syntax=r"^\s*from\s+random\s+import[^\n]*\bseed\b",
                    en="`from random import seed` brings in a function called `seed`, but inside `pick`, `seed` is already the argument with the number. Use `import random` and `random.seed(seed)`.",
                    nl="`from random import seed` haalt een functie `seed` binnen, maar in `pick` is `seed` al het argument met het getal. Gebruik `import random` en `random.seed(seed)`."),
            Mistake("no-prefix", syntax=r"\A(?![\s\S]*^\s*from\s+random\s+import)[\s\S]*^[^#\n]*(?<![\w.])(?:choice|randint)\s*\(",
                    en="After `import random`, a name from the module needs `random.` in front, as in `random.choice(...)`. It happens on the line `{line}`.",
                    nl="Na `import random` zet je voor een naam uit de module `random.`, zoals in `random.choice(...)`. Het gebeurt op de regel `{line}`."),
            Mistake("a-list", matches=lambda got, text: isinstance(got, list),
                    en="`{call}` gives `{got}`: a list. `random.choice`, without an s, picks one item.",
                    nl="`{call}` geeft `{got}`: een lijst. `random.choice`, zonder s, kiest één item."),
            Mistake("no-seed", matches=lambda got, text: re.search(r"^[^#\n]*\bseed\s*\(", text, re.MULTILINE) is None,
                    en="`{call}` gives `{got}`, an item from the list, but without setting the seed it can be another one each time. Look at step 4 of the recipe.",
                    nl="`{call}` geeft `{got}`, een item uit de lijst, maar zonder de seed te zetten kan het elke keer een ander zijn. Kijk naar stap 4 van het recept."),
            Mistake("fixed-seed", state="almost", matches=lambda got, text: re.search(r"\bseed\s*\(\s*\d+\s*\)", text) is not None,
                    en="`{call}` gives `{got}`, and with this seed the check expects `{want}`. Your function sets a seed you typed yourself. Give `random.seed` the argument `seed`, so each call uses its own seed.",
                    nl="`{call}` geeft `{got}`, en met deze seed verwacht de check `{want}`. Je functie zet een seed die je zelf typte. Geef `random.seed` het argument `seed`, dan gebruikt elke aanroep zijn eigen seed."),
            Mistake("other-way", matches=lambda got, text: re.search(r"^[^#\n]*\brandom\s*\.\s*seed\s*\(\s*seed\s*\)", text, re.MULTILINE) is not None,
                    en="`{call}` gives `{got}`, and with this seed the check expects `{want}`. Is the seed set first, and is the item picked right after it, with one call of `random.choice`?",
                    nl="`{call}` geeft `{got}`, en met deze seed verwacht de check `{want}`. Wordt eerst de seed gezet, en wordt het item meteen daarna gekozen, met één aanroep van `random.choice`?"),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...`: set the seed with the argument `seed`, then give back one item of `items` (step 4 of the recipe).",
                      "nl": "Vervang de `...`: zet de seed met het argument `seed`, en geef dan één item van `items` terug (stap 4 van het recept)."},
            "cleared": {"en": "There is no code in the box any more. Start again with `import random`, then `def pick(items, seed):`, and under it, indented, your lines.",
                        "nl": "Er staat geen code meer in het vak. Begin opnieuw met `import random`, dan `def pick(items, seed):`, met daaronder, ingesprongen, je regels."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`, one item of the list. Look at step 4 of the recipe: first the seed, then the pick.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn, één item uit de lijst. Kijk naar stap 4 van het recept: eerst de seed, dan de keuze."},
            "almost": {"en": "Right for the first list. But `{call}` gives `{got}`, and it should give `{want}`. Does your function use its own `items` and `seed`?",
                       "nl": "Goed voor de eerste lijst. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Gebruikt je functie zijn eigen `items` en `seed`?"},
            "correct": {"en": "Your function is right in all {n} cases: the same seed gives the same pick, every time.",
                        "nl": "Je functie klopt in alle {n} gevallen: dezelfde seed geeft elke keer dezelfde keuze."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_M = "import math\n\ndef circle_area(r):\n"
_R = "import random\n\ndef pick(items, seed):\n"
CASES = [
    ("circle", _M + "    ...\n", "empty", None),
    ("circle", _CIRCLE, "correct", None),
    ("circle", _M + "    return math.pi * r * r", "correct", None),
    ("circle", "from math import pi\n\ndef circle_area(r):\n    return pi * r ** 2", "correct", None),
    ("circle", _M + "    return 3.14 * r ** 2", "almost", "rounded-pi"),
    ("circle", _M + "    return 3.14159 * r ** 2", "almost", "rounded-pi"),
    ("circle", _M + "    return 2 * math.pi * r", "wrong", "circumference"),
    ("circle", _M + "    return math.pi * r * 2", "wrong", "circumference"),
    ("circle", _M + "    return math.pi * r", "wrong", "no-square"),
    ("circle", _M + "    return pi * r ** 2", "error", "no-prefix"),
    ("circle", _M + "    return math.PI * r ** 2", "error", "pi-capital"),
    ("circle", _M + "    return math.pi() * r ** 2", "error", "pi-called"),
    ("circle", "def circle_area(r):\n    return math.pi * r ** 2", "error", "no-import"),
    ("circle", _M + "    math.pi = 3.14\n    return math.pi * r ** 2", "wrong", "assign-to-module"),
    ("pick", _R + "    ...\n", "empty", None),
    ("pick", _PICK, "correct", None),
    ("pick", _R + "    random.seed(seed)\n    item = random.choice(items)\n    return item", "correct", None),
    ("pick", _R + "    random.seed(seed)\n    return items[random.randint(0, len(items) - 1)]", "correct", None),
    ("pick", _R + "    return random.choice(items)", "wrong", "no-seed"),
    ("pick", _R + "    random.seed(1)\n    return random.choice(items)", "almost", "fixed-seed"),
    ("pick", _R + "    random.seed(seed)\n    random.random()\n    return random.choice(items)", "wrong", "other-way"),
    ("pick", _R + "    random.seed(seed)\n    return choice(items)", "error", "no-prefix"),
    ("pick", _R + "    random.seed(seed)\n    return random.sample(items, 1)", "wrong", "a-list"),
    ("pick", _R + "    random.seed(seed)\n    print(random.choice(items))", "wrong", "prints"),
    ("pick", _R + "    random.seed = seed\n    return random.choice(items)", "wrong", "assign-to-module"),
    ("pick", "from random import seed, choice\n\ndef pick(items, seed):\n    seed(seed)\n    return choice(items)", "error", "shadowed-seed"),
]
