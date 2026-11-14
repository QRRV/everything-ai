# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson py-json: adding up the grams of a recipe read from a JSON file, and writing a weather report
as JSON text. And the dict of the picture, in Python and as JSON."""
import json
import re

from everything_ai.check import FunctionBox, Mistake
from everything_ai import yours

# The files of this lesson. Python runs in the course's own folder, so a path starts there.
FOLDER = "lessons/py-json"
PANCAKES = f"{FOLDER}/pancakes.json"
SOUP = f"{FOLDER}/soup.json"
TEA = f"{FOLDER}/tea.json"

# The picture: a dict as Python shows it, and the same dict as JSON text, with what changes on the way.
PICTURE = {"name": "pancakes", "vegetarian": True, "notes": None, "servings": 4}


def picture_lines() -> tuple[list[str], list[str]]:
    """The lines of the picture: repr of each pair as Python shows it, and json.dumps of each pair."""
    items = list(PICTURE.items())
    py, js = [], []
    for i, (k, v) in enumerate(items):
        first, last = ("{" if i == 0 else " "), ("}" if i == len(items) - 1 else ",")
        py.append(f"{first}{k!r}: {v!r}{last}")
        js.append(f"{first}{json.dumps(k)}: {json.dumps(v)}{last}")
    return py, js


def same_json(a, b) -> bool:
    """Two JSON texts with the same content: the same keys and values, also the same kind (false is not 0), in
    whatever order and spacing."""
    if not (isinstance(a, str) and isinstance(b, str)):
        return False
    try:
        return json.dumps(json.loads(a), sort_keys=True) == json.dumps(json.loads(b), sort_keys=True)
    except ValueError:
        return False


def _not_json(got, text) -> bool:
    if not isinstance(got, str):
        return False
    try:
        json.loads(got)
        return False
    except ValueError:
        return True


def _rain_text(got, text) -> bool:
    try:
        return isinstance(json.loads(got).get("rain"), str)
    except (TypeError, ValueError, AttributeError):
        return False


_SYNTAX = [
    Mistake("dots-left", syntax=r"^[^#\n]*\.\.\.",
            en="There is still a `...` in your function. Replace it with your own lines.",
            nl="Er staat nog een `...` in je functie. Vervang die door je eigen regels."),
    Mistake("curly-quote", syntax=r"[“”‘’„]",
            en="Your code has a curly quote, like `“`. Python only reads straight quotes, like `\"`: type the quotes again in the box.",
            nl="Je code heeft een gekruld aanhalingsteken, zoals `“`. Python leest alleen rechte, zoals `\"`: typ de aanhalingstekens opnieuw in het vak."),
    Mistake("no-import", syntax=r"\A(?![\s\S]*^\s*(?:import|from)\s)",
            en="Your box has no `import` line. The check runs the box on its own, so it needs its own `import json` at the top.",
            nl="Je vak heeft geen regel met `import`. De check voert het vak los uit, dus het heeft een eigen `import json` bovenaan nodig."),
]

# The lesson is done when these are correct. "true-null" is the choice question in the lesson.
REQUIRED = ["total-grams", "true-null", "report"]

_TOTAL = ("import json\n\ndef total_grams(path):\n    with open(path) as f:\n        recipe = json.load(f)\n"
          "    total = 0\n    for ingredient in recipe[\"ingredients\"]:\n        total = total + ingredient[\"grams\"]\n"
          "    return total")
_REPORT = "import json\n\ndef report(city, mm):\n    return json.dumps({\"city\": city, \"mm\": mm, \"rain\": mm > 0})"

EXERCISES = {
    "total-grams": FunctionBox(
        name="total-grams",
        picture=yours.steps(call=0),
        function="total_grams",
        start="import json\n\ndef total_grams(path):\n    ...\n",
        solution=_TOTAL,
        visible=[(PANCAKES,)],
        hidden=[(SOUP,), (TEA,)],
        mistakes=_SYNTAX + [
            Mistake("loads-path", error=lambda e: isinstance(e, json.JSONDecodeError),
                    en="Your function stops with a `JSONDecodeError`: `json.loads` reads JSON from a string, and `path` is the string of the path, not the text in the file. Open the file with `with open(path) as f:`, and read it with `json.load(f)`, without the s.",
                    nl="Je functie stopt met een `JSONDecodeError`: `json.loads` leest JSON uit een string, en `path` is de string van het pad, niet de tekst in het bestand. Open het bestand met `with open(path) as f:`, en lees het met `json.load(f)`, zonder de s."),
            Mistake("load-path", error=lambda e: isinstance(e, AttributeError) and "read" in str(e),
                    en="Your function stops with an `AttributeError`: `json.load` reads from an open file, and gets the path, a string. Open the file first, `with open(path) as f:`, and give `json.load` the open file `f`.",
                    nl="Je functie stopt met een `AttributeError`: `json.load` leest uit een open bestand, en krijgt het pad, een string. Open eerst het bestand, `with open(path) as f:`, en geef `json.load` het open bestand `f`."),
            Mistake("loads-file", error=lambda e: isinstance(e, TypeError) and "must be str" in str(e),
                    en="Your function stops with a `TypeError`: `json.loads` reads JSON from a string, and gets the open file `f`. For an open file, use `json.load(f)`, without the s.",
                    nl="Je functie stopt met een `TypeError`: `json.loads` leest JSON uit een string, en krijgt het open bestand `f`. Gebruik voor een open bestand `json.load(f)`, zonder de s."),
            Mistake("dumps-read", error=lambda e: isinstance(e, TypeError) and "not JSON serializable" in str(e),
                    en="Your function stops with a `TypeError`: `json.dumps` goes the other way, from a Python value to JSON text, and cannot write an open file as JSON. To read the JSON in the file, use `json.load(f)`.",
                    nl="Je functie stopt met een `TypeError`: `json.dumps` werkt de andere kant op, van een Python-waarde naar JSON-tekst, en kan geen open bestand als JSON schrijven. Lees de JSON in het bestand met `json.load(f)`."),
            Mistake("dump-read", error=lambda e: isinstance(e, TypeError) and "'fp'" in str(e),
                    en="Your function stops with a `TypeError`: `json.dump` goes the other way: it writes a value to a file. To read the JSON in the file, use `json.load(f)`.",
                    nl="Je functie stopt met een `TypeError`: `json.dump` werkt de andere kant op: hij schrijft een waarde naar een bestand. Lees de JSON in het bestand met `json.load(f)`."),
            Mistake("no-load", syntax=r"\A(?![\s\S]*\bloads?\s*\()",
                    en="Your function never reads the JSON with `json.load`. Without it, what you have of the file is text, one string, and not a dict, so you cannot look up `\"ingredients\"` in it. Read the open file with `json.load(f)`.",
                    nl="Je functie leest de JSON nergens met `json.load`. Zonder dat heb je van het bestand alleen tekst, één string, en geen dict, dus kun je er `\"ingredients\"` niet in opzoeken. Lees het open bestand met `json.load(f)`."),
            Mistake("list-key", error=lambda e: isinstance(e, TypeError) and "list indices" in str(e),
                    en="Your function stops with a `TypeError`: it looks up a key in a list. `recipe[\"ingredients\"]` is a list with a dict for each ingredient: go through it with a `for` loop, and look up `\"grams\"` in each dict.",
                    nl="Je functie stopt met een `TypeError`: hij zoekt een key op in een lijst. `recipe[\"ingredients\"]` is een lijst met een dict voor elk ingrediënt: ga hem langs met een `for`-loop, en zoek in elke dict `\"grams\"` op."),
            Mistake("keys-loop", error=lambda e: isinstance(e, TypeError) and "string indices" in str(e),
                    en="Your function stops with a `TypeError`: it looks up a key in a string. `for x in recipe:` goes through the keys of the dict, strings such as `\"name\"`. The list of ingredients is `recipe[\"ingredients\"]`: loop over that.",
                    nl="Je functie stopt met een `TypeError`: hij zoekt een key op in een string. `for x in recipe:` gaat de keys van de dict langs, strings zoals `\"name\"`. De lijst met ingrediënten is `recipe[\"ingredients\"]`: ga die lijst langs."),
            Mistake("wrong-key", error=lambda e: isinstance(e, KeyError),
                    en="Your function stops with a `KeyError`: it looks up a key that the dict does not have. Look at the file: the keys are written exactly as `\"ingredients\"` and `\"grams\"`.",
                    nl="Je functie stopt met een `KeyError`: hij zoekt een key op die de dict niet heeft. Kijk in het bestand: de keys staan er precies als `\"ingredients\"` en `\"grams\"`."),
            Mistake("count", wrong=_TOTAL.replace('total + ingredient["grams"]', "total + 1"),
                    en="`{call}` gives `{got}`: the number of ingredients, not their grams. Add the grams of each ingredient.",
                    nl="`{call}` geeft `{got}`: het aantal ingrediënten, niet hun grammen. Tel de grammen van elk ingrediënt op."),
            Mistake("last-only", wrong=_TOTAL.replace('total = total + ingredient["grams"]', 'total = ingredient["grams"]'),
                    en="`{call}` gives `{got}`: only the grams of the last ingredient. Each round, add to what `total` already is.",
                    nl="`{call}` geeft `{got}`: alleen de grammen van het laatste ingrediënt. Tel elke ronde op bij wat `total` al is."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...`: open the file at `path`, read it with `json.load`, and add up the grams of every ingredient.",
                      "nl": "Vervang de `...`: open het bestand op `path`, lees het met `json.load`, en tel de grammen van elk ingrediënt op."},
            "cleared": {"en": "There is no code in the box any more. Start again with `import json`, then `def total_grams(path):`, and under it, indented, your lines.",
                        "nl": "Er staat geen code meer in het vak. Begin opnieuw met `import json`, dan `def total_grams(path):`, met daaronder, ingesprongen, je regels."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: 250 + 450 + 110 + 25 grams. Does your function add up `\"grams\"` of every ingredient in `recipe[\"ingredients\"]`?",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: 250 + 450 + 110 + 25 gram. Telt je functie `\"grams\"` op van elk ingrediënt in `recipe[\"ingredients\"]`?"},
            "almost": {"en": "Right for the pancakes. But `{call}` gives `{got}`, and it should give `{want}`. Does your function read the file it gets, and add up all its ingredients?",
                       "nl": "Goed voor de pannenkoeken. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Leest je functie het bestand dat hij krijgt, en telt hij al zijn ingrediënten op?"},
            "correct": {"en": "Your function is right for all {n} recipes: it reads the JSON, and adds up the grams of every ingredient.",
                        "nl": "Je functie klopt voor alle {n} recepten: hij leest de JSON, en telt de grammen van elk ingrediënt op."},
        },
    ),
    "report": FunctionBox(
        name="report",
        picture=yours.steps(call=0),
        function="report",
        start="import json\n\ndef report(city, mm):\n    ...\n",
        solution=_REPORT,
        visible=[("Utrecht", 2.5)],
        hidden=[("Delft", 0), ("'s-Hertogenbosch", 11), ("Groningen", 0.25)],
        compare=same_json,
        show=repr,
        mistakes=_SYNTAX + [
            Mistake("dump-text", error=lambda e: isinstance(e, TypeError) and "'fp'" in str(e),
                    en="With `{call}`, your function stops with a `TypeError`: `json.dump`, without the s, writes to an open file, and gets none. For text, use `json.dumps`, with the s: *dump to a string*.",
                    nl="Met `{call}` stopt je functie met een `TypeError`: `json.dump`, zonder de s, schrijft naar een open bestand, en krijgt er geen. Gebruik voor tekst `json.dumps`, met de s: *dump to a string*."),
            Mistake("loads-dict", error=lambda e: isinstance(e, TypeError) and "must be str" in str(e),
                    en="With `{call}`, your function stops with a `TypeError`: `json.loads` goes the other way, from JSON text to Python values, and gets a dict. To write the dict as JSON text, use `json.dumps`.",
                    nl="Met `{call}` stopt je functie met een `TypeError`: `json.loads` werkt de andere kant op, van JSON-tekst naar Python-waarden, en krijgt een dict. Schrijf de dict als JSON-tekst met `json.dumps`."),
            Mistake("a-dict", matches=lambda got, text: isinstance(got, dict),
                    en="`{call}` gives a dict: a Python value, not text. Give the dict to `json.dumps`, and return the text it makes.",
                    nl="`{call}` geeft een dict: een Python-waarde, geen tekst. Geef de dict aan `json.dumps`, en geef de tekst terug die hij maakt."),
            Mistake("not-json", matches=_not_json,
                    en="`{call}` gives `{got}`, and that is not valid JSON: `json.loads` cannot read it back. Do not write the text yourself: make a dict, and let `json.dumps` write it, with double quotes and `true` or `false`.",
                    nl="`{call}` geeft `{got}`, en dat is geen geldige JSON: `json.loads` kan het niet teruglezen. Schrijf de tekst niet zelf: maak een dict, en laat `json.dumps` hem schrijven, met dubbele aanhalingstekens en `true` of `false`."),
            Mistake("rain-text", matches=_rain_text,
                    en="`{call}` gives `{got}`: there, `\"rain\"` is a string in quotes. Give `json.dumps` a bool, `True` or `False`, and it writes `true` or `false` without quotes.",
                    nl="`{call}` geeft `{got}`: daarin is `\"rain\"` een string tussen aanhalingstekens. Geef `json.dumps` een bool, `True` of `False`, dan schrijft hij zelf `true` of `false`, zonder aanhalingstekens."),
            Mistake("at-least", state="almost", wrong=_REPORT.replace("mm > 0", "mm >= 0"),
                    en="`{call}` gives `{got}`: with 0 mm it did not rain. `\"rain\"` is true only when `mm` is more than 0.",
                    nl="`{call}` geeft `{got}`: bij 0 mm heeft het niet geregend. `\"rain\"` is alleen true als `mm` meer dan 0 is."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...`: make a dict with the keys `\"city\"`, `\"mm\"` and `\"rain\"`, and return the JSON text that `json.dumps` makes of it.",
                      "nl": "Vervang de `...`: maak een dict met de keys `\"city\"`, `\"mm\"` en `\"rain\"`, en geef de JSON-tekst terug die `json.dumps` ervan maakt."},
            "cleared": {"en": "There is no code in the box any more. Start again with `import json`, then `def report(city, mm):`, and under it, indented, your lines.",
                        "nl": "Er staat geen code meer in het vak. Begin opnieuw met `import json`, dan `def report(city, mm):`, met daaronder, ingesprongen, je regels."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give JSON with the same keys and values as `{want}`. Are the keys exactly `\"city\"`, `\"mm\"` and `\"rain\"`, and is each value the right kind: a number for `\"mm\"`, true or false for `\"rain\"`?",
                      "nl": "`{call}` geeft `{got}`, en het zou JSON moeten zijn met dezelfde keys en waarden als `{want}`. Zijn de keys precies `\"city\"`, `\"mm\"` en `\"rain\"`, en is elke waarde van de goede soort: een getal voor `\"mm\"`, true of false voor `\"rain\"`?"},
            "almost": {"en": "Right for Utrecht. But `{call}` gives `{got}`, and it should give JSON with the same keys and values as `{want}`. Does your function use its own `city` and `mm`, and compute `\"rain\"` from `mm`?",
                       "nl": "Goed voor Utrecht. Maar `{call}` geeft `{got}`, en het zou JSON moeten zijn met dezelfde keys en waarden als `{want}`. Gebruikt je functie zijn eigen `city` en `mm`, en rekent hij `\"rain\"` uit met `mm`?"},
            "correct": {"en": "Your function is right in all {n} cases: valid JSON every time, which `json.loads` can read back.",
                        "nl": "Je functie klopt in alle {n} gevallen: elke keer geldige JSON, die `json.loads` kan teruglezen."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_T = "import json\n\ndef total_grams(path):\n"
_R = "import json\n\ndef report(city, mm):\n"
CASES = [
    ("total-grams", _T + "    ...\n", "empty", None),
    ("total-grams", _TOTAL, "correct", None),
    ("total-grams", _T + "    with open(path) as f:\n        recipe = json.loads(f.read())\n    return sum(i[\"grams\"] for i in recipe[\"ingredients\"])", "correct", None),
    ("total-grams", _T + "    recipe = json.loads(path)\n    total = 0\n    for i in recipe[\"ingredients\"]:\n        total = total + i[\"grams\"]\n    return total", "error", "loads-path"),
    ("total-grams", _T + "    recipe = json.load(path)\n    total = 0\n    for i in recipe[\"ingredients\"]:\n        total = total + i[\"grams\"]\n    return total", "error", "load-path"),
    ("total-grams", _T + "    with open(path) as f:\n        recipe = json.load(f)\n    total = 0\n    for i in recipe:\n        total = total + i[\"grams\"]\n    return total", "error", "keys-loop"),
    ("total-grams", _T + "    with open(path) as f:\n        recipe = json.load(f)\n    total = 0\n    for i in recipe[\"ingredient\"]:\n        total = total + i[\"grams\"]\n    return total", "error", "wrong-key"),
    ("total-grams", _T + "    with open(path) as f:\n        recipe = json.loads(f)\n    return 0", "error", "loads-file"),
    ("total-grams", _T + "    with open(path) as f:\n        recipe = json.dumps(f)\n    return 0", "error", "dumps-read"),
    ("total-grams", _T + "    with open(path) as f:\n        recipe = json.dump(f)\n    return 0", "error", "dump-read"),
    ("total-grams", _T + "    with open(path) as f:\n        text = f.read()\n    total = 0\n    for i in text[\"ingredients\"]:\n        total = total + i[\"grams\"]\n    return total", "error", "no-load"),
    ("total-grams", _T + "    with open(path) as f:\n        recipe = json.load(f)\n    return recipe[\"ingredients\"][\"grams\"]", "error", "list-key"),
    ("total-grams", _TOTAL.replace('total + ingredient["grams"]', "total + 1"), "wrong", "count"),
    ("total-grams", _TOTAL.replace('total = total + ingredient["grams"]', 'total = ingredient["grams"]'), "wrong", "last-only"),
    ("total-grams", _T + "    return 835", "almost", None),
    ("total-grams", "def total_grams(path):\n    with open(path) as f:\n        recipe = json.load(f)\n    return 0", "error", "no-import"),
    ("report", _R + "    ...\n", "empty", None),
    ("report", _REPORT, "correct", None),
    ("report", _R + "    data = {\"rain\": mm > 0, \"city\": city, \"mm\": mm}\n    return json.dumps(data, indent=2)", "correct", None),
    ("report", _R + "    return {\"city\": city, \"mm\": mm, \"rain\": mm > 0}", "wrong", "a-dict"),
    ("report", _R + "    return \"{'city': '\" + city + \"', 'mm': \" + str(mm) + \", 'rain': \" + str(mm > 0) + \"}\"", "wrong", "not-json"),
    ("report", _R + "    return json.dumps({\"city\": city, \"mm\": mm, \"rain\": str(mm > 0)})", "wrong", "rain-text"),
    ("report", _REPORT.replace("mm > 0", "mm >= 0"), "almost", "at-least"),
    ("report", _R + "    return json.dump({\"city\": city, \"mm\": mm, \"rain\": mm > 0})", "error", "dump-text"),
    ("report", _R + "    return json.loads({\"city\": city, \"mm\": mm, \"rain\": mm > 0})", "error", "loads-dict"),
    ("report", _R + "    return json.dumps({\"city\": city, \"mm\": mm, \"rain\": 1 if mm > 0 else 0})", "wrong", None),
    ("report", _R + "    return json.dumps({\"city\": \"Utrecht\", \"mm\": 2.5, \"rain\": True})", "almost", None),
    ("report", _R + "    return json.dumps({\"town\": city, \"mm\": mm, \"rain\": mm > 0})", "wrong", None),
]
