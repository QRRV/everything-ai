# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson data-provenance: the datasheet of the weather station from the lesson data-explore,
completed in a box, and a function that finds the fields a datasheet still lacks."""
import re

from everything_ai.check import CodeBox, FunctionBox, Mistake

# What every datasheet in this lesson answers, in this order.
FIELDS = ["source", "collected", "license", "consent", "limits"]

# The datasheet of Palmer penguins, as the worked example writes it: one of the datasheets the check tries.
PENGUINS = {
    "source": "Kristen Gorman and the Palmer Station research program, Antarctica",
    "collected": "2007 to 2009",
    "license": "CC0 1.0",
    "consent": "not needed: there are no people in it",
    "limits": "344 penguins of 3 species, from 3 islands; some values are missing",
}

# The lesson is done when these are correct. "cc-by" is the choice question in the lesson.
REQUIRED = ["cc-by", "station-card", "missing-fields"]

_CARD_START = '''card = {
    "source": "made up by the course, for its lessons",
    "collected": "one month, with no real date",
    ...
}
card
'''
_CARD = '''card = {
    "source": "made up by the course, for its lessons",
    "collected": "one month, with no real date",
    "license": "CC BY-SA 4.0",
    "consent": "not needed: there are no people in it",
    "limits": "one made-up station, one month; two temperatures missing, and -999 for a broken wind meter",
}
card'''

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*,?\s*$", text, re.MULTILINE) is not None,
                en="There is still a line with only `...` in your code. Replace it with your own lines.",
                nl="Er staat nog een regel met alleen `...` in je code. Vervang die door je eigen regels.")


def _plain(text: str) -> str:
    """A license name without spaces, dashes and case, and without the word international, so that CC-BY-SA-4.0 and
    Creative Commons Attribution-ShareAlike 4.0 International read alike."""
    return re.sub(r"international$", "", re.sub(r"[^a-z0-9]", "", text.lower()))


_STATION_LICENSE = ("ccbysa40", "ccbysa4", "creativecommonsattributionsharealike40")


def _card_ok(got, want) -> bool:
    """A datasheet is right with every field filled in with text, and the station's license, CC BY-SA 4.0, written
    in any of its usual ways."""
    return (isinstance(got, dict) and all(isinstance(got.get(f), str) and got[f].strip() for f in FIELDS)
            and _plain(got["license"]) in _STATION_LICENSE)


def _lacks(got) -> list[str]:
    return [f for f in FIELDS if not (isinstance(got.get(f), str) and got[f].strip())]


def _lacking(got, text):
    names = _lacks(got)
    return {"fields": ", ".join(f"`{n}`" for n in names)}


_MISSING_START = '''FIELDS = ["source", "collected", "license", "consent", "limits"]


def missing_fields(card):
    ...
'''
_MISSING = '''FIELDS = ["source", "collected", "license", "consent", "limits"]


def missing_fields(card):
    missing = []
    for field in FIELDS:
        if field in card:
            if card[field] == "":
                missing.append(field)
        else:
            missing.append(field)
    return missing'''


def _missing(new: str) -> str:
    return _MISSING.replace('''    missing = []
    for field in FIELDS:
        if field in card:
            if card[field] == "":
                missing.append(field)
        else:
            missing.append(field)
    return missing''', new)


EXERCISES = {
    "station-card": CodeBox(
        name="station-card",
        start=_CARD_START,
        solution=_CARD,
        compare=_card_ok,
        first=[_DOTS],
        mistakes=[
            Mistake("no-comma", syntax=r'"[ \t]*\n[ \t]*"\w+"[ \t]*:',
                    en="Python cannot read your dictionary: a comma is missing after the pair on the line before `{line}`. Every pair ends with a comma.",
                    nl="Python kan je dictionary niet lezen: er ontbreekt een komma na het paar op de regel vóór `{line}`. Elk paar eindigt met een komma."),
            Mistake("not-a-dictionary", matches=lambda got, text: not isinstance(got, dict),
                    en="Your last line does not give a dictionary. Keep the curly brackets of `card`, and put `card` on the last line.",
                    nl="Je laatste regel geeft geen dictionary. Houd de accolades van `card`, en zet `card` op de laatste regel."),
            Mistake("field-lacking", matches=lambda got, text: isinstance(got, dict) and bool(_lacks(got)), detail=_lacking,
                    en="Your datasheet has no text yet for {fields}. Each of the five fields gets a text between quotes, such as `\"license\": \"a text\"`, with a comma after each pair. Check the spelling of each key too: exactly `license`, `consent` and `limits`.",
                    nl="Je datasheet heeft nog geen tekst voor {fields}. Elk van de vijf velden krijgt een tekst tussen aanhalingstekens, zoals `\"license\": \"een tekst\"`, met een komma na elk paar. Kijk ook de spelling van elke key na: precies `license`, `consent` en `limits`."),
            Mistake("no-version", state="almost", matches=lambda got, text: isinstance(got, dict) and not _lacks(got) and _plain(got["license"]) == "ccbysa",
                    en="Almost: write the version too, CC BY-SA 4.0. Each version of a license has its own terms.",
                    nl="Bijna: schrijf de versie er ook bij, CC BY-SA 4.0. Elke versie van een licentie heeft zijn eigen voorwaarden."),
            Mistake("other-license", matches=lambda got, text: isinstance(got, dict) and not _lacks(got),
                    en="Every field has a text, but the license is not the one of the station data. The course shares the station data under CC BY-SA 4.0: write that as the license.",
                    nl="Elk veld heeft een tekst, maar de licentie is niet die van de stationsdata. De course deelt de stationsdata onder CC BY-SA 4.0: schrijf dat als licentie."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the three fields that are still missing: `license`, `consent` and `limits`.",
                      "nl": "Vervang de `...` door de drie velden die nog ontbreken: `license`, `consent` en `limits`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "no_value": {"en": "The last line of the box shows nothing. Put `card` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `card` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`, and it should be a datasheet with a text for each of `source`, `collected`, `license`, `consent` and `limits`.",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet een datasheet zijn met een tekst voor elk van `source`, `collected`, `license`, `consent` en `limits`."},
            "correct": {"en": "The datasheet is complete: where the data comes from, when, under which license, whether people had to agree, and what it does not cover. The check reads only whether each field has a text and the license; what you wrote in the other fields is yours.",
                        "nl": "De datasheet is compleet: waar de data vandaan komt, wanneer, onder welke licentie, of er mensen toestemming moesten geven, en wat de data niet dekt. De check kijkt alleen of elk veld een tekst heeft, en naar de licentie; wat je in de andere velden schreef, is aan jou."},
        },
    ),
    "missing-fields": FunctionBox(
        name="missing-fields",
        function="missing_fields",
        start=_MISSING_START,
        solution=_MISSING,
        visible=[({"source": "a weather station", "license": "CC0 1.0"},)],
        hidden=[(dict(PENGUINS),), ({},), ({"source": "", "collected": "2024", "license": "MIT", "consent": "", "limits": "one city"},),
                ({"limits": "one school", "license": "CC BY 4.0", "name": "test scores"},),
                ({"source": "x", "collected": "", "license": "CC0 1.0", "notes": ""},)],
        first=[_DOTS],
        mistakes=[
            Mistake("key-error", error=lambda e: isinstance(e, KeyError),
                    en="With `{call}`, your code looks up a field that the datasheet does not have, and Python stops. Check with `in` whether the field is there before you look at its text.",
                    nl="Met `{call}` zoekt je code een veld op dat de datasheet niet heeft, en Python stopt. Kijk eerst met `in` of het veld er is, voordat je naar de tekst kijkt."),
            Mistake("empty-text", wrong=_missing('''    missing = []
    for field in FIELDS:
        if field not in card:
            missing.append(field)
    return missing'''),
                    en="`{call}` gives `{got}`, and it should give `{want}`: a field with an empty text, `\"\"`, says nothing either. Count it as missing too.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: een veld met een lege tekst, `\"\"`, zegt ook niets. Tel het ook als ontbrekend."),
            Mistake("present", wrong=_missing('''    present = []
    for field in FIELDS:
        if field in card and card[field] != "":
            present.append(field)
    return present'''),
                    en="`{call}` gives `{got}`: the fields that are there. The task asks for the ones that are missing.",
                    nl="`{call}` geeft `{got}`: de velden die er wel zijn. De opdracht vraagt om de ontbrekende."),
            Mistake("how-many", wrong=_missing('''    missing = []
    for field in FIELDS:
        if field not in card or card[field] == "":
            missing.append(field)
    return len(missing)'''),
                    en="`{call}` gives `{got}`: how many fields are missing. The task asks which ones: give back the list itself.",
                    nl="`{call}` geeft `{got}`: hoeveel velden er ontbreken. De opdracht vraagt welke: geef de lijst zelf terug."),
            Mistake("card-keys", wrong=_missing('''    missing = []
    for field in card:
        if card[field] == "":
            missing.append(field)
    return missing'''),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your loop goes through the fields the datasheet has, so it never sees the ones that are not there. Go through `FIELDS` instead.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je loop gaat door de velden die de datasheet heeft, en ziet dus nooit de velden die er niet zijn. Ga door `FIELDS`."),
            Mistake("get-none", wrong=_missing('''    missing = []
    for field in FIELDS:
        if card.get(field) == "":
            missing.append(field)
    return missing'''),
                    en="`{call}` gives `{got}`, and it should give `{want}`: a field that is not in the datasheet is not counted. `card.get(field)` gives `None` for such a field, and `None` is not equal to `\"\"`. Check with `in` first whether the field is there.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: een veld dat niet in de datasheet staat, wordt niet geteld. `card.get(field)` geeft `None` voor zo'n veld, en `None` is niet gelijk aan `\"\"`. Kijk eerst met `in` of het veld er is."),
            Mistake("return-in-loop", wrong=_missing('''    for field in FIELDS:
        if field not in card or card[field] == "":
            return [field]
    return []'''),
                    en="`{call}` gives `{got}`: only the first missing field. `return` stops the function at once, inside the loop. Collect the fields in a list, and give it back after the loop.",
                    nl="`{call}` geeft `{got}`: alleen het eerste ontbrekende veld. `return` stopt de functie meteen, in de loop. Verzamel de velden in een lijst, en geef die na de loop terug."),
            Mistake("return-in-loop-2", wrong=_missing('''    missing = []
    for field in FIELDS:
        if field not in card or card[field] == "":
            missing.append(field)
        return missing'''),
                    en="`{call}` gives `{got}`: your `return` is inside the loop, so the function stops after the first field. Put `return missing` after the loop, at the same indent as `for`.",
                    nl="`{call}` geeft `{got}`: je `return` staat in de loop, dus de functie stopt na het eerste veld. Zet `return missing` na de loop, even ver ingesprongen als `for`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with code that gives back the list of fields from `FIELDS` that `card` lacks.",
                      "nl": "Vervang de `...` door code die de lijst velden uit `FIELDS` teruggeeft die `card` mist."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with one dictionary, and the `def` line of your function has a different number of names between its parentheses. Keep `(card)`.",
                          "nl": "De check roept `{call}` aan, met één dictionary, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(card)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you go through `FIELDS`, and keep each field that `card` does not have or that has an empty text?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Ga je door `FIELDS`, en houd je elk veld over dat `card` niet heeft of dat een lege tekst heeft?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for every datasheet, also a complete one or an empty one?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor elke datasheet, ook een complete of een lege?"},
            "correct": {"en": "Your function is right in all {n} cases: a complete datasheet gives `[]`, and an empty one all five fields, in the order of `FIELDS`.",
                        "nl": "Je functie klopt in alle {n} gevallen: een complete datasheet geeft `[]`, en een lege alle vijf de velden, in de volgorde van `FIELDS`."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_HEAD = 'card = {\n    "source": "made up by the course, for its lessons",\n    "collected": "one month, with no real date",\n'
_F = 'FIELDS = ["source", "collected", "license", "consent", "limits"]\n\n\ndef missing_fields(card):\n'
CASES = [
    ("station-card", _CARD_START, "empty", None),
    ("station-card", _CARD, "correct", None),
    ("station-card", _HEAD + '    "license": "CC-BY-SA-4.0",\n    "consent": "no people",\n    "limits": "made up",\n}\ncard', "correct", None),
    ("station-card", _HEAD + '    "license": "cc by-sa 4.0",\n    "consent": "no people",\n    "limits": "made up",\n}\ncard', "correct", None),
    ("station-card", _HEAD + '    "license": "CC BY 4.0",\n    "consent": "no people",\n    "limits": "made up",\n}\ncard', "wrong", "other-license"),
    ("station-card", _HEAD + '    "license": "Creative Commons Attribution-ShareAlike 4.0 International",\n    "consent": "no people",\n    "limits": "made up",\n}\ncard', "correct", None),
    ("station-card", _HEAD + '    "license": "CC BY-SA",\n    "consent": "no people",\n    "limits": "made up",\n}\ncard', "almost", "no-version"),
    ("station-card", _HEAD + '    "license": "CC BY-SA 4.0"\n    "consent": "no people",\n    "limits": "made up",\n}\ncard', "error", "no-comma"),
    ("station-card", _HEAD + '    "license": "CC BY-SA 4.0",\n    "consent": "no people",\n    "limits": "one month...",\n}\ncard', "correct", None),
    ("station-card", _HEAD + '    "license": "CC BY-SA 4.0",\n    "limits": "made up",\n}\ncard', "wrong", "field-lacking"),
    ("station-card", _HEAD + '    "license": "CC BY-SA 4.0",\n    "consent": "",\n    "limits": "made up",\n}\ncard', "wrong", "field-lacking"),
    ("station-card", _HEAD + '    "license": "CC BY-SA 4.0",\n    "consent": "no people",\n    "limits": "made up",\n}\ncard["license"]', "wrong", "not-a-dictionary"),
    ("station-card", _HEAD + '    "license": "CC BY-SA 4.0",\n    "consent": "no people",\n    "limits": "made up",\n}\nprint(card)', "wrong", "print-last"),
    ("station-card", _HEAD + '    "license": "CC BY-SA 4.0",\n    "consent": "no people",\n    "limits": "made up",\n}', "wrong", "no-value"),
    ("missing-fields", _MISSING_START, "empty", None),
    ("missing-fields", _MISSING, "correct", None),
    ("missing-fields", _F + '    missing = []\n    for field in FIELDS:\n        if field not in card or card[field] == "":\n            missing.append(field)\n    return missing', "correct", None),
    ("missing-fields", _F + '    return [f for f in FIELDS if f not in card or card[f] == ""]', "correct", None),
    ("missing-fields", _F + '    missing = []\n    for field in FIELDS:\n        if card.get(field) == "":\n            missing.append(field)\n    return missing', "wrong", "get-none"),
    ("missing-fields", _F + '    missing = []\n    for field in FIELDS:\n        if field not in card or card[field] == "":\n            missing.append(field)\n        return missing', "wrong", "return-in-loop-2"),
    ("missing-fields", _F + '    return [f for f in FIELDS if card.get(f, "") == ""]', "correct", None),
    ("missing-fields", _F + '    missing = []\n    for field in FIELDS:\n        if card[field] == "":\n            missing.append(field)\n    return missing', "error", "key-error"),
    ("missing-fields", _F + '    missing = []\n    for field in FIELDS:\n        if field not in card:\n            missing.append(field)\n    return missing', "wrong", "empty-text"),
    ("missing-fields", _F + '    return [f for f in FIELDS if f in card and card[f] != ""]', "wrong", "present"),
    ("missing-fields", _F + '    return len([f for f in FIELDS if f not in card or card[f] == ""])', "wrong", "how-many"),
    ("missing-fields", _F + '    return [f for f in card if card[f] == ""]', "wrong", "card-keys"),
    ("missing-fields", _F + '    for field in FIELDS:\n        if field not in card or card[field] == "":\n            return [field]\n    return []', "wrong", "return-in-loop"),
    ("missing-fields", _F + '    return ["collected", "consent", "limits"]', "almost", None),
]
