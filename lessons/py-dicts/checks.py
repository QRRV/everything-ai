# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson py-dicts: looking up by key, and counting with a dictionary. And the dictionary in the picture."""
import ast
import re

from everything_ai.check import CodeBox, Mistake

# The dictionary in the picture, and the key that is looked up.
PICTURE = {"apple": 0.5, "pear": 0.75, "plum": 0.3}
PICTURE_MARK = "pear"


def _lines(text) -> list[str]:
    return [line.rstrip() for line in str(text).rstrip("\n").splitlines()]


def same_lines(a, b) -> bool:
    return isinstance(a, str) and isinstance(b, str) and _lines(a) == _lines(b)


def _as_dict(text):
    try:
        value = ast.literal_eval(text.strip())
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, dict) else None


def same_dict(a, b) -> bool:
    """The same lines, or the same dictionary shown with its pairs in another order."""
    if same_lines(a, b):
        return True
    da, db = _as_dict(a) if isinstance(a, str) else None, _as_dict(b) if isinstance(b, str) else None
    return da is not None and da == db


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

# The lesson is done when these are correct. "keys" is the choice question in the lesson.
REQUIRED = ["capital", "keys", "count"]

_CAPITALS = 'capitals = {"France": "Paris", "Japan": "Tokyo", "Peru": "Lima"}'

EXERCISES = {
    "capital": CodeBox(
        name="capital",
        start=_CAPITALS + '\ncountry = "Japan"\ncity = ...\nprint(city)',
        solution=_CAPITALS + '\ncountry = "Japan"\ncity = capitals[country]\nprint(city)',
        outcome="stdout",
        expect="Tokyo\n",
        compare=same_lines,
        others=[{"country": "Peru"}],
        keep=[_CAPITALS, 'country = "Japan"'],
        mistakes=_DOTS + [
            _CURLY,
            Mistake("quoted-name", syntax=r"\[\s*[\"']country[\"']\s*\]",
                    en="With quotes, `\"country\"` is the text *country*, and no country in `capitals` is called that. Without quotes, `country` is the name, and Python uses its value, `\"Japan\"`.",
                    nl="Met aanhalingstekens is `\"country\"` de tekst *country*, en geen land in `capitals` heet zo. Zonder aanhalingstekens is `country` de naam, en gebruikt Python de waarde ervan, `\"Japan\"`."),
            Mistake("round-brackets", syntax=r"capitals\s*\(",
                    en="A key goes between square brackets, `[ ]`, not round ones, as in step 2 of the recipe.",
                    nl="Een key staat tussen vierkante haken, `[ ]`, niet tussen ronde haakjes, zoals in stap 2 van het recept."),
            Mistake("by-place", syntax=r"capitals\s*\[\s*-?\d+\s*\]",
                    en="A dictionary has no places with a number, as a list has. You look up a value by its key: here, the name of a country.",
                    nl="Een lijst heeft genummerde plekken, een dictionary niet. Je zoekt een waarde op met de key: hier de naam van een land."),
            Mistake("value-as-key", syntax=r"capitals\s*\[\s*[\"'](Tokyo|Paris|Lima)[\"']\s*\]",
                    en="`{line}` looks up a city, but the cities are the values. The keys are the countries: look up with the key in `country`.",
                    nl="`{line}` zoekt een stad op, maar de steden zijn de waarden. De keys zijn de landen: zoek op met de key in `country`."),
            Mistake("dot", syntax=r"capitals\s*\.\s*\w",
                    en="A key goes between square brackets right after the dictionary's name, not after a dot: `capitals[...]`.",
                    nl="Een key staat tussen vierkante haken direct na de naam van de dictionary, niet na een punt: `capitals[...]`."),
            Mistake("whole-dict", matches=lambda got, text: got.strip().startswith("{"),
                    en="Your program shows the whole dictionary. `city` should be one value: the one under the key in `country`.",
                    nl="Je programma laat de hele dictionary zien. `city` hoort één waarde te zijn: die onder de key in `country`."),
            Mistake("the-key", matches=_shows("Japan"),
                    en="Your program shows `Japan`: the key itself. You want the value under that key. Look it up with square brackets after `capitals`.",
                    nl="Je programma laat `Japan` zien: de key zelf. Je wilt de waarde onder die key. Zoek hem op met vierkante haken na `capitals`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the capital of `country`, looked up in `capitals`.",
                      "nl": "Vervang de `...` door de hoofdstad van `country`, opgezocht in `capitals`."},
            "cleared": {"en": "There is no code in the box any more. Type the four lines back: the line with `capitals`, `country = \"Japan\"`, `city = ...` and `print(city)`.",
                        "nl": "Er staat geen code meer in het vak. Typ de vier regels terug: de regel met `capitals`, `country = \"Japan\"`, `city = ...` en `print(city)`."},
            "wrong": {"en": "Your program shows:\n\n```\n{got}\n```\n\nThat is not yet `Tokyo`. Look at step 2 of the recipe: which dictionary do you look in, and with which key?",
                      "nl": "Je programma laat dit zien:\n\n```\n{got}\n```\n\nDat is nog niet `Tokyo`. Kijk naar stap 2 van het recept: in welke dictionary zoek je, en met welke key?"},
            "other": {"en": "Right for Japan. But with {where} instead, your program shows `{got}`, and it should show `{want}`. Did you type the country or the city yourself? Look up with the name `country`, so it works for every country.",
                      "nl": "Goed voor Japan. Maar met {where} laat je programma `{got}` zien, en dat moet `{want}` zijn. Typte je het land of de stad zelf? Zoek op met de naam `country`, dan werkt het voor elk land."},
            "other_error": {"en": "Right for Japan. But with {where} instead, your program stops with an error. Did you type the country yourself? Look up with the name `country`, so it works for every country.",
                            "nl": "Goed voor Japan. Maar met {where} stopt je programma met een fout. Typte je het land zelf? Zoek op met de naam `country`, dan werkt het voor elk land."},
            "correct": {"en": "Your program shows `Tokyo`, the value under the key `\"Japan\"`, and it also works for another country.",
                        "nl": "Je programma laat `Tokyo` zien, de waarde onder de key `\"Japan\"`, en het werkt ook voor een ander land."},
        },
    ),
    "count": CodeBox(
        name="count",
        start='words = ["rain", "sun", "rain", "fog", "rain"]\ncounts = {}\nfor word in words:\n    ...\nprint(counts)',
        solution=('words = ["rain", "sun", "rain", "fog", "rain"]\ncounts = {}\nfor word in words:\n'
                  '    if word in counts:\n        counts[word] = counts[word] + 1\n    else:\n        counts[word] = 1\nprint(counts)'),
        outcome="stdout",
        expect="{'rain': 3, 'sun': 1, 'fog': 1}\n",
        compare=same_dict,
        others=[{"words": ["a", "b", "a"]}],
        keep=['words = ["rain", "sun", "rain", "fog", "rain"]', "counts = {}"],
        mistakes=_DOTS + [
            _CURLY,
            Mistake("else-indent", syntax=r"^([ \t]*)if\b[^\n]*:[ \t]*\n(?:\1[ \t]+[^\n]*\n)*\1[ \t]+else\s*:",
                    en="Your `else:` has more spaces in front than its `if`. An `else:` lines up with its `if`: the same number of spaces. In the box, press Backspace once before `else:`.",
                    nl="Je `else:` heeft meer spaties ervoor dan zijn `if`. Een `else:` staat gelijk met zijn `if`: evenveel spaties. Druk in het vak één keer op Backspace voor `else:`."),
            Mistake("double-equals", syntax=r"^[ \t]*counts\s*\[[^\]\n]*\]\s*==",
                    en="The line `{line}` compares with `==`, and gives the key no value. To give a key a value, use one `=`: `counts[word] = 1`.",
                    nl="De regel `{line}` vergelijkt met `==`, en geeft de key geen waarde. Een key een waarde geven doe je met één `=`: `counts[word] = 1`."),
            Mistake("in-order", syntax=r"\bcounts\s+in\s+word\b|\bin\s+word\b",
                    en="The order is: the key first, then `in`, then the dictionary: `word in counts`.",
                    nl="De volgorde is: eerst de key, dan `in`, dan de dictionary: `word in counts`."),
            Mistake("whole-list", syntax=r"\bwords\s+in\s+counts\b|counts\s*\[\s*words\s*\]",
                    en="`words` is the whole list. In the loop, the word of this round is `word`, without an s.",
                    nl="`words` is de hele lijst. In de loop is het woord van deze ronde `word`, zonder s."),
            Mistake("swapped", syntax=r"else\s*:\s*\n\s*counts\s*\[\s*word\s*\]\s*=\s*counts\s*\[",
                    en="Under `else:`, the word is not a key yet, so `counts[word]` cannot be looked up there. Under `else:`, the new pair simply gets 1; the count goes up under `if word in counts:`.",
                    nl="Onder `else:` is het woord nog geen key, dus kan `counts[word]` daar niet worden opgezocht. Onder `else:` krijgt het nieuwe paar gewoon 1; de telling gaat omhoog onder `if word in counts:`."),
            Mistake("no-check", syntax=r"\A(?![\s\S]*\bin\s+counts\b)[\s\S]*counts\s*\[\s*word\s*\]\s*\+",
                    en="The first time a word comes by, it is not a key in `counts` yet, so `counts[word]` stops the program. Check with `in` first whether the word is a key. If not, it gets a new pair, with count 1.",
                    nl="De eerste keer dat een woord langskomt, is het nog geen key in `counts`, dus stopt `counts[word]` het programma. Kijk eerst met `in` of het woord een key is. Zo niet, dan krijgt het een nieuw paar, met telling 1."),
            Mistake("append", syntax=r"counts\s*\.\s*append",
                    en="A dictionary has no `append`; that is for lists. A new pair goes in with `=`: `counts[word] = 1`.",
                    nl="Een dictionary heeft geen `append`; die is voor lijsten. Een nieuw paar zet je erin met `=`: `counts[word] = 1`."),
            Mistake("else-edge", matches=lambda got, text: re.search(r"^else\s*:", text, re.MULTILINE) is not None,
                    en="Your `else:` is at the left edge, so it does not belong to the `if` inside the loop. Give it the same number of spaces in front as that `if`.",
                    nl="Je `else:` staat helemaal links, dus hoort hij niet bij de `if` in de loop. Geef hem evenveel spaties ervoor als die `if`."),
            Mistake("quoted-word", matches=lambda got, text: "'word'" in got,
                    en="Your dictionary has the key `'word'`: with quotes, `\"word\"` is the text *word*. Without quotes, `word` is the name, and Python uses its value, the word of this round.",
                    nl="Je dictionary heeft de key `'word'`: met aanhalingstekens is `\"word\"` de tekst *word*. Zonder aanhalingstekens is `word` de naam, en gebruikt Python de waarde ervan, het woord van deze ronde."),
            Mistake("loop-over-counts", matches=lambda got, text: re.search(r"^\s*for\s+\w+\s+in\s+counts\b", text, re.MULTILINE) is not None,
                    en="Your loop goes through `counts`, which starts empty, so the loop never runs. Go through the list: `for word in words:`.",
                    nl="Je loop gaat door `counts`, en die begint leeg, dus de loop draait nooit. Ga door de lijst: `for word in words:`."),
            Mistake("empty", matches=_shows("{}"),
                    en="Your program shows `{{}}`: `counts` stays empty. A word that is not a key yet has to get a new pair: write `else:` with `counts[word] = 1` under it.",
                    nl="Je programma laat `{{}}` zien: `counts` blijft leeg. Een woord dat nog geen key is, moet een nieuw paar krijgen: schrijf `else:` met daaronder `counts[word] = 1`."),
            Mistake("all-one", matches=_shows("{'rain': 1, 'sun': 1, 'fog': 1}"),
                    en="Each word has count 1, also `rain`, which comes by three times. When a word is already a key, its count has to go up: give `counts[word]` a new value, the old value plus 1. And check that your `if` is `True` when the word is already a key: `word in counts`.",
                    nl="Elk woord heeft telling 1, ook `rain`, dat drie keer langskomt. Is een woord al een key, dan moet zijn telling omhoog: geef `counts[word]` een nieuwe waarde, de oude waarde plus 1. En kijk of je `if` `True` is als het woord al een key is: `word in counts`."),
            Mistake("from-zero", matches=_shows("{'rain': 2, 'sun': 0, 'fog': 0}"),
                    en="Each count is one too low. The first time a word comes by, it already counts once: its new pair gets 1, not 0.",
                    nl="Elke telling is één te laag. De eerste keer dat een woord langskomt, telt het al één keer: zijn nieuwe paar krijgt 1, niet 0."),
            Mistake("inside-loop", matches=lambda got, text: len(_lines(got)) > 1,
                    en="Your program shows more than one line. `print(counts)` belongs at the left edge, after the loop, so it runs once.",
                    nl="Je programma laat meer dan één regel zien. `print(counts)` hoort helemaal links, na de loop, zodat hij één keer draait."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` in the loop with an `if` that checks whether `word` is a key in `counts`, and an `else`.",
                      "nl": "Vervang de `...` in de loop door een `if` die kijkt of `word` een key is in `counts`, en een `else`."},
            "cleared": {"en": "There is no code in the box any more. Type the lines back: `words = [\"rain\", \"sun\", \"rain\", \"fog\", \"rain\"]`, `counts = {{}}`, `for word in words:`, the indented `...`, and `print(counts)`.",
                        "nl": "Er staat geen code meer in het vak. Typ de regels terug: `words = [\"rain\", \"sun\", \"rain\", \"fog\", \"rain\"]`, `counts = {{}}`, `for word in words:`, de ingesprongen `...`, en `print(counts)`."},
            "wrong": {"en": "Your program shows:\n\n```\n{got}\n```\n\nThat is not yet `{{'rain': 3, 'sun': 1, 'fog': 1}}`. In each round: is `word` already a key? Then 1 more; if not, a new pair with 1.",
                      "nl": "Je programma laat dit zien:\n\n```\n{got}\n```\n\nDat is nog niet `{{'rain': 3, 'sun': 1, 'fog': 1}}`. In elke ronde: is `word` al een key? Dan 1 erbij; zo niet, dan een nieuw paar met 1."},
            "keep": {"en": "The check needs the line `{line}` as it was: put it back. The loop fills `counts`, starting from an empty dictionary, and the check tries other words by itself.",
                     "nl": "De check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De loop vult `counts`, vanaf een lege dictionary, en de check probeert zelf andere woorden."},
            "other": {"en": "Right for these words. But with {where} instead, your program shows `{got}`, and it should show `{want}`. Did you type the counts yourself? Let the loop count every word in `words`.",
                      "nl": "Goed voor deze woorden. Maar met {where} laat je programma `{got}` zien, en dat moet `{want}` zijn. Typte je de tellingen zelf? Laat de loop elk woord in `words` tellen."},
            "other_error": {"en": "Right for these words. But with {where} instead, your program stops with an error. Did you use a word from the list yourself? Use the name `word`, so it works for every word.",
                            "nl": "Goed voor deze woorden. Maar met {where} stopt je programma met een fout. Gebruikte je zelf een woord uit de lijst? Gebruik de naam `word`, dan werkt het voor elk woord."},
            "correct": {"en": "Your program shows `{got}`: `rain` three times, `sun` and `fog` once. It also works for another list.",
                        "nl": "Je programma laat `{got}` zien: `rain` drie keer, `sun` en `fog` één keer. Het werkt ook voor een andere lijst."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_C = _CAPITALS + '\ncountry = "Japan"\n'
_W = 'words = ["rain", "sun", "rain", "fog", "rain"]\ncounts = {}\nfor word in words:\n'
_P = "print(counts)"
CASES = [
    ("capital", _C + "city = ...\nprint(city)", "empty", None),
    ("capital", "", "empty", None),
    ("capital", _C + "city = capitals[country]\nprint(city)", "correct", None),
    ("capital", _C + "print(capitals[country])", "correct", None),
    ("capital", _C + 'city = capitals["Japan"]\nprint(city)', "almost", "other-value"),
    ("capital", _C + 'city = "Tokyo"\nprint(city)', "almost", "other-value"),
    ("capital", _C + 'city = capitals["country"]\nprint(city)', "error", "quoted-name"),
    ("capital", _C + "city = capitals(country)\nprint(city)", "error", "round-brackets"),
    ("capital", _C + "city = capitals[1]\nprint(city)", "error", "by-place"),
    ("capital", _C + "city = capitals.country\nprint(city)", "error", "dot"),
    ("capital", _C + "city = capitals\nprint(city)", "wrong", "whole-dict"),
    ("capital", _C + "city = country\nprint(city)", "wrong", "the-key"),
    ("capital", _C + "city = capitals[Japan]\nprint(city)", "error", None),
    ("capital", _CAPITALS + '\ncountry = "Peru"\ncity = capitals[country]\nprint(city)', "wrong", "changed-line"),
    ("count", _W + "    ...\n" + _P, "empty", None),
    ("count", _W + "    if word in counts:\n        counts[word] = counts[word] + 1\n    else:\n        counts[word] = 1\n" + _P, "correct", None),
    ("count", _W + "    if word not in counts:\n        counts[word] = 0\n    counts[word] = counts[word] + 1\n" + _P, "correct", None),
    ("count", _W + "    counts[word] = counts[word] + 1\n" + _P, "error", "no-check"),
    ("count", _W + "    counts[word] = 1\n" + _P, "wrong", "all-one"),
    ("count", _W + "    if word in counts:\n        counts[word] = counts[word] + 1\n" + _P, "wrong", "empty"),
    ("count", _W + "    if word in counts:\n        counts[word] = counts[word] + 1\n    else:\n        counts[word] = 0\n" + _P, "wrong", "from-zero"),
    ("count", _W + "    if word in counts:\n        counts[word] = 1\n    else:\n        counts[word] = counts[word] + 1\n" + _P, "error", "swapped"),
    ("count", _W + "    counts.append(word)\n" + _P, "error", "append"),
    ("count", _W + "    if counts in word:\n        counts[word] = counts[word] + 1\n    else:\n        counts[word] = 1\n" + _P, "error", "in-order"),
    ("count", _W + "    if word in counts:\n        counts[word] = counts[word] + 1\n    else:\n        counts[word] = 1\n    print(counts)", "wrong", "inside-loop"),
    ("count", _W + "    if words in counts:\n        counts[word] = counts[word] + 1\n    else:\n        counts[word] = 1\n" + _P, "error", "whole-list"),
    ("count", _W + "    if \"word\" in counts:\n        counts[\"word\"] = counts[\"word\"] + 1\n    else:\n        counts[\"word\"] = 1\n" + _P, "wrong", "quoted-word"),
    ("count", _W + "    if word in counts:\n        counts[word] += 1\n    else:\n        counts[word] = 1\n" + _P, "correct", None),
    ("count", _W + "    if word in counts:\n        counts[word] = counts[word] + 1\nelse:\n    counts[word] = 1\n" + _P, "wrong", "else-edge"),
    ("count", _W + "    if word in counts:\n        counts[word] = counts[word] + 1\n        else:\n            counts[word] = 1\n" + _P, "error", "else-indent"),
    ("count", _W + "    if word in counts:\n        counts[word] = counts[word] + 1\n    else:\n        counts[word] == 1\n" + _P, "error", "double-equals"),
    ("count", 'words = ["rain", "sun", "rain", "fog", "rain"]\ncounts = {}\nfor word in words[::-1]:\n    if word in counts:\n        counts[word] = counts[word] + 1\n    else:\n        counts[word] = 1\n' + _P, "correct", None),
    ("count", 'words = ["rain", "sun", "rain", "fog", "rain"]\ncounts = {}\nfor word in counts:\n    if word in counts:\n        counts[word] = counts[word] + 1\n    else:\n        counts[word] = 1\n' + _P, "wrong", "loop-over-counts"),
    ("capital", _C + 'city = capitals["Tokyo"]\nprint(city)', "error", "value-as-key"),
    ("count", 'words = ["rain", "sun", "rain", "fog", "rain"]\ncounts = {"rain": 3, "sun": 1, "fog": 1}\n' + _P, "wrong", "changed-line"),
]
