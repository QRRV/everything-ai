# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson py-strings: initials with indexing, and a caption with an f-string."""
import re

from everything_ai.check import CodeBox, Mistake

# The word in the picture.
WORD = "banana"


def _lines(text) -> list[str]:
    return [line.rstrip() for line in str(text).rstrip("\n").splitlines()]


def same_lines(a, b) -> bool:
    return isinstance(a, str) and isinstance(b, str) and _lines(a) == _lines(b)


def _shows(*texts):
    """The program showed exactly one of these lines."""
    return lambda got, text: _lines(got) in [[t] for t in texts]


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


_DOTS = _both("dots-left", r"^[^#\n]*\.\.\.",
              en="There is still a `...` in your program. Replace it with what the name should get.",
              nl="Er staat nog een `...` in je programma. Vervang die door wat de naam moet krijgen.")
_CURLY = [Mistake("curly-quote", syntax=r"[“”‘’„]",
                  en="Your program has a curly quote, like `“`. Python only reads straight quotes, like `\"`: type the quotes again in the box.",
                  nl="Je programma heeft een gekruld aanhalingsteken, zoals `“`. Python leest alleen rechte, zoals `\"`: typ de aanhalingstekens opnieuw in het vak.")]
_ROUND = [Mistake("round-brackets", syntax=r"(first_name|last_name|city_name)\(\s*-?\d+\s*\)",
                  en="On the line `{line}`, a place is between round parentheses. A place goes between square brackets, `[ ]`: `word[0]`, not `word(0)`.",
                  nl="Op de regel `{line}` staat een plek tussen ronde haakjes. Een plek zet je tussen vierkante haken, `[ ]`: `word[0]`, niet `word(0)`.")]
_NO_F = Mistake("no-f", matches=lambda got, text: "{" in got and "}" in got,
                en="Your program shows the braces as they are, with the names inside them. That happens when the `f` before the quote is missing: write `f\"...\"`.",
                nl="Je programma laat de accolades zien zoals ze er staan, met de namen erin. Dat gebeurt als de `f` voor het aanhalingsteken ontbreekt: schrijf `f\"...\"`.")
_LEN_SQUARE = Mistake("len-square", syntax=r"len\s*\[",
                      en="`len` takes the string between round parentheses: `len(city_name)`, not `len[city_name]`.",
                      nl="`len` krijgt de string tussen ronde haakjes: `len(city_name)`, niet `len[city_name]`.")
_PLUS_NUMBER = Mistake("plus-number", syntax=r"\+\s*len\(|len\([^)]*\)\s*\+",
                       en="`len` gives a number, and `+` cannot glue a number to a string. Put `len(city_name)` between braces in an f-string.",
                       nl="`len` geeft een getal, en `+` kan geen getal aan een string plakken. Zet `len(city_name)` tussen accolades in een f-string.")
_FIRST = _DOTS + _CURLY + _ROUND + [_LEN_SQUARE, _PLUS_NUMBER]


def _printed_word(word):
    """The program printed the name itself, as text, instead of its value."""
    return Mistake("word", matches=lambda got, text: _lines(got)[-1:] == [word],
                   en=f"Your program shows the word `{word}`, because it is between quotes. Without quotes, `print({word})` shows the value of the name.",
                   nl=f"Je programma laat het woord `{word}` zien, omdat het tussen aanhalingstekens staat. Zonder aanhalingstekens laat `print({word})` de waarde van de naam zien.")

# The lesson is done when these are correct. "spaces" is the choice question in the lesson.
REQUIRED = ["initials", "spaces", "caption"]

EXERCISES = {
    "initials": CodeBox(
        name="initials",
        start='first_name = "Grace"\nlast_name = "Hopper"\ninitials = ...\nprint(initials)',
        solution='first_name = "Grace"\nlast_name = "Hopper"\ninitials = f"{first_name[0]}{last_name[0]}"\nprint(initials)',
        outcome="stdout",
        expect="GH\n",
        compare=same_lines,
        others=[{"first_name": "Ada", "last_name": "Lovelace"}],
        keep=['first_name = "Grace"', 'last_name = "Hopper"'],
        mistakes=_FIRST + [
            _NO_F, _printed_word("initials"),
            Mistake("index-outside", matches=lambda got, text: "[0]" in got or "[-1]" in got,
                    en="Your program shows the index as text: it is outside the braces. Put it inside, right after the name, so Python takes the character.",
                    nl="Je programma laat de index als tekst zien: hij staat buiten de accolades. Zet hem erbinnen, direct achter de naam, dan pakt Python het teken."),
            Mistake("from-one", matches=_shows("ro", "r o"),
                    en="Your program shows `{got}`: the second letter of each name. Python counts places from 0, so the first letter is `[0]`.",
                    nl="Je programma laat `{got}` zien: de tweede letter van elke naam. Python telt plekken vanaf 0, dus de eerste letter is `[0]`."),
            Mistake("last-letters", matches=_shows("er", "e r"),
                    en="Your program shows `{got}`: the last letter of each name. `[-1]` counts from the end; the first letter is `[0]`.",
                    nl="Je programma laat `{got}` zien: de laatste letter van elke naam. `[-1]` telt vanaf het eind; de eerste letter is `[0]`."),
            Mistake("whole-names", matches=_shows("GraceHopper", "Grace Hopper"),
                    en="Your program shows the whole names. Take only the first character of each name, with `[0]`.",
                    nl="Je programma laat de hele namen zien. Neem van elke naam alleen het eerste teken, met `[0]`."),
            Mistake("names-as-text", matches=_shows("fl", "f l", "first_namelast_name"),
                    en="Your program takes letters from the words `first_name` and `last_name` themselves, because they are between quotes. Without quotes, Python uses the value of the name.",
                    nl="Je programma neemt letters uit de woorden `first_name` en `last_name` zelf, omdat ze tussen aanhalingstekens staan. Zonder aanhalingstekens gebruikt Python de waarde van de naam."),
            Mistake("space-between", matches=_shows("G H"), state="almost",
                    en="The letters are right, but there is a space between them. Initials go right next to each other: `GH`.",
                    nl="De letters kloppen, maar er staat een spatie tussen. Initialen staan direct naast elkaar: `GH`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the initials: the first character of each name, right next to each other.",
                      "nl": "Vervang de `...` door de initialen: het eerste teken van elke naam, direct naast elkaar."},
            "cleared": {"en": "There is no code in the box any more. Type the four lines back: `first_name = \"Grace\"`, `last_name = \"Hopper\"`, `initials = ...` and `print(initials)`.",
                        "nl": "Er staat geen code meer in het vak. Typ de vier regels terug: `first_name = \"Grace\"`, `last_name = \"Hopper\"`, `initials = ...` en `print(initials)`."},
            "wrong": {"en": "Your program shows:\n\n```\n{got}\n```\n\nThat is not yet `GH`: the first character of `first_name`, right after it the first character of `last_name`, and nothing else. Look at steps 3 and 4 of the recipe.",
                      "nl": "Je programma laat dit zien:\n\n```\n{got}\n```\n\nDat is nog niet `GH`: het eerste teken van `first_name`, direct daarachter het eerste teken van `last_name`, en verder niets. Kijk naar stap 3 en 4 van het recept."},
            "other": {"en": "Right for Grace Hopper. But with the names Ada Lovelace, your program shows `{got}`, and it should show `{want}`. Take the letters from `first_name` and `last_name`; do not type them yourself.",
                      "nl": "Goed voor Grace Hopper. Maar met de namen Ada Lovelace laat je programma `{got}` zien, en het zou `{want}` moeten zijn. Pak de letters uit `first_name` en `last_name`; typ ze niet zelf."},
            "correct": {"en": "Your program shows `GH`: the first character of each name, at place 0. It also works for other names, such as Ada Lovelace.",
                        "nl": "Je programma laat `GH` zien: het eerste teken van elke naam, op plek 0. Het werkt ook voor andere namen, zoals Ada Lovelace."},
        },
    ),
    "caption": CodeBox(
        name="caption",
        start='city_name = "Groningen"\ncaption = ...\nprint(caption)',
        solution='city_name = "Groningen"\ncaption = f"{city_name} ({len(city_name)})"\nprint(caption)',
        outcome="stdout",
        expect="Groningen (9)\n",
        compare=same_lines,
        others=[{"city_name": "Delft"}],
        keep=['city_name = "Groningen"'],
        first=[Mistake("len-of-word", matches=lambda got, text: re.search(r"len\(\s*[\"']city_name[\"']\s*\)", text) is not None,
                       en="`len(\"city_name\")` counts the characters of the word `city_name` itself, because it is between quotes. Write `len(city_name)`, without quotes.",
                       nl="`len(\"city_name\")` telt de tekens van het woord `city_name` zelf, omdat het tussen aanhalingstekens staat. Schrijf `len(city_name)`, zonder aanhalingstekens.")],
        mistakes=_FIRST + [
            _NO_F, _printed_word("caption"),
            Mistake("len-as-text", matches=lambda got, text: "len(" in got,
                    en="Your program shows `len(` as text. Put the whole `len(city_name)` between braces, so Python works it out.",
                    nl="Je programma laat `len(` als tekst zien. Zet de hele `len(city_name)` tussen accolades, dan rekent Python het uit."),
            Mistake("name-not-value", matches=lambda got, text: "city_name" in got,
                    en="Your program shows the word `city_name`. In an f-string, put the name between braces, so Python fills in its value.",
                    nl="Je programma laat het woord `city_name` zien. Zet de naam in een f-string tussen accolades, dan vult Python zijn waarde in."),
            Mistake("no-space", matches=_shows("Groningen(9)"), state="almost",
                    en="The name and the number are right, but the space before the `(` is missing. In an f-string, every space counts.",
                    nl="De naam en het getal kloppen, maar de spatie voor de `(` ontbreekt. In een f-string telt elke spatie mee."),
            Mistake("no-parentheses", matches=_shows("Groningen 9"), state="almost",
                    en="The name and the number are right, but the parentheses around the number are missing: `Groningen (9)`.",
                    nl="De naam en het getal kloppen, maar de haakjes om het getal ontbreken: `Groningen (9)`."),
            Mistake("miscounted", matches=lambda got, text: len(_lines(got)) == 1 and re.fullmatch(r"Groningen \((?!9\))\d+\)", _lines(got)[0]) is not None and "len(" not in text,
                    en="The form is right, but Groningen has another number of characters. Let `len(city_name)` count them.",
                    nl="De vorm klopt, maar Groningen heeft een ander aantal tekens. Laat `len(city_name)` ze tellen."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with an f-string that gives `Groningen (9)`.",
                      "nl": "Vervang de `...` door een f-string die `Groningen (9)` geeft."},
            "cleared": {"en": "There is no code in the box any more. Type the three lines back: `city_name = \"Groningen\"`, `caption = ...` and `print(caption)`.",
                        "nl": "Er staat geen code meer in het vak. Typ de drie regels terug: `city_name = \"Groningen\"`, `caption = ...` en `print(caption)`."},
            "wrong": {"en": "Your program shows:\n\n```\n{got}\n```\n\nThat is not yet `Groningen (9)`. Look at step 4 of the recipe: an `f` before the quote, and between braces what Python should fill in.",
                      "nl": "Je programma laat dit zien:\n\n```\n{got}\n```\n\nDat is nog niet `Groningen (9)`. Kijk naar stap 4 van het recept: een `f` voor het aanhalingsteken, en tussen accolades wat Python moet invullen."},
            "other": {"en": "Right for Groningen. But with {where} instead, your program shows `{got}`, and it should show `{want}`. Use the name `city_name`, and let `len(city_name)` count, so it works for every city.",
                      "nl": "Goed voor Groningen. Maar met {where} laat je programma `{got}` zien, en het zou `{want}` moeten zijn. Gebruik de naam `city_name`, en laat `len(city_name)` tellen, dan werkt het voor elke stad."},
            "correct": {"en": "Your program shows `Groningen (9)`: the name, and its number of characters from `len`. It also works for another city, such as Delft.",
                        "nl": "Je programma laat `Groningen (9)` zien: de naam, en het aantal tekens uit `len`. Het werkt ook voor een andere stad, zoals Delft."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_I = 'first_name = "Grace"\nlast_name = "Hopper"\n'
_L = 'city_name = "Groningen"\n'
CASES = [
    ("initials", _I + "initials = ...\nprint(initials)", "empty", None),
    ("initials", "", "empty", None),
    ("initials", _I + 'initials = f"{first_name[0]}{last_name[0]}"\nprint(initials)', "correct", None),
    ("initials", _I + "initials = first_name[0] + last_name[0]\nprint(initials)", "correct", None),
    ("initials", _I + '_a = first_name[0]\n_b = last_name[0]\ninitials = f"{_a}{_b}"\nprint(initials)', "correct", None),
    ("initials", _I + 'initials = "{first_name[0]}{last_name[0]}"\nprint(initials)', "wrong", "no-f"),
    ("initials", _I + 'initials = f"{first_name[1]}{last_name[1]}"\nprint(initials)', "wrong", "from-one"),
    ("initials", _I + 'initials = f"{first_name[-1]}{last_name[-1]}"\nprint(initials)', "wrong", "last-letters"),
    ("initials", _I + 'initials = f"{first_name}{last_name}"\nprint(initials)', "wrong", "whole-names"),
    ("initials", _I + 'initials = f"{first_name[0]} {last_name[0]}"\nprint(initials)', "almost", "space-between"),
    ("initials", _I + "initials = 'first_name'[0] + 'last_name'[0]\nprint(initials)", "wrong", "names-as-text"),
    ("initials", _I + "initials = first_name(0) + last_name(0)\nprint(initials)", "error", "round-brackets"),
    ("initials", _I + "initials = first_name[9]\nprint(initials)", "error", None),
    ("initials", _I + 'initials = f"{first_name[0]}{last_name[0]}"\nprint(“initials”)', "error", "curly-quote"),
    ("initials", _I + 'initials = f"{first_name[0]}"\nprint(initials)', "wrong", None),
    ("initials", _I + 'initials = "GH"\nprint(initials)', "almost", "other-value"),
    ("initials", _I + 'initials = f"{first_name}[0]{last_name}[0]"\nprint(initials)', "wrong", "index-outside"),
    ("initials", _I + 'initials = f"{first_name[0]}{last_name[0]}"\nprint("initials")', "wrong", "word"),
    ("initials", 'first_name = "Quinn"\nlast_name = "Hopper"\ninitials = f"{first_name[0]}{last_name[0]}"\nprint(initials)', "wrong", "changed-line"),
    ("initials", "first_name = 'Grace'\nlast_name = 'Hopper'\ninitials = first_name[0] + last_name[0]\nprint(initials)", "correct", None),
    ("initials", _I + '# initials = ...\ninitials = f"{first_name[1]}{last_name[1]}"\nprint(initials)', "wrong", "from-one"),
    ("caption", _L + "caption = ...\nprint(caption)", "empty", None),
    ("caption", _L + 'caption = f"{city_name} ({len(city_name)})"\nprint(caption)', "correct", None),
    ("caption", _L + '_n = len(city_name)\ncaption = f"{city_name} ({_n})"\nprint(caption)', "correct", None),
    ("caption", _L + 'caption = "{city_name} ({len(city_name)})"\nprint(caption)', "wrong", "no-f"),
    ("caption", _L + 'caption = f"city_name ({len(city_name)})"\nprint(caption)', "wrong", "name-not-value"),
    ("caption", _L + 'caption = f"{city_name}({len(city_name)})"\nprint(caption)', "almost", "no-space"),
    ("caption", _L + 'caption = f"{city_name} {len(city_name)}"\nprint(caption)', "almost", "no-parentheses"),
    ("caption", _L + 'caption = f"{city_name} (8)"\nprint(caption)', "wrong", "miscounted"),
    ("caption", _L + 'caption = f"{city_name} ({len(city_name) - 1})"\nprint(caption)', "wrong", None),
    ("caption", _L + 'caption = f"{city_name} (9)"\nprint(caption)', "almost", "other-value"),
    ("caption", _L + 'caption = f"{city_name} ({len("city_name")})"\nprint(caption)', "wrong", "len-of-word"),
    ("caption", _L + 'caption = f"{city_name} (len(city_name))"\nprint(caption)', "wrong", "len-as-text"),
    ("caption", _L + 'caption = city_name + " (" + len(city_name) + ")"\nprint(caption)', "error", "plus-number"),
    ("caption", _L + 'caption = f"{city_name} ({len[city_name]})"\nprint(caption)', "error", "len-square"),
]
