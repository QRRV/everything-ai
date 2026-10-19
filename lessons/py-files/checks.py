# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson py-files: counting the lines of a file, and showing some of them. And the lines in the picture."""
import re
from pathlib import Path

from everything_ai.check import CodeBox, Mistake

# The files of this lesson. Python runs in the course's own folder, so a path starts there.
FOLDER = "lessons/py-files"
POEM = f"{FOLDER}/poem.txt"
SHOPPING = f"{FOLDER}/shopping.txt"
PANTRY = f"{FOLDER}/pantry.txt"
_SHOPPING_TEXT = (Path(__file__).parent / "shopping.txt").read_text(encoding="utf-8")

# The picture: the first lines of the poem, as they are in the file and as `for line in f` gives them.
PICTURE_FILE = ["Roses are red,", "violets are blue,"]


def _lines(text) -> list[str]:
    return [line.rstrip() for line in str(text).rstrip("\n").splitlines()]


def same_lines(a, b) -> bool:
    return isinstance(a, str) and isinstance(b, str) and _lines(a) == _lines(b)


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
_PATH_QUOTED = Mistake("path-quoted", syntax=r"open\(\s*[\"']path[\"']\s*\)",
                       en="With quotes, `\"path\"` is a file called *path*, and there is none. Without quotes, `path` is the name, and Python uses its value, the real path. Remove the quotes: `open(path)`.",
                       nl="Met aanhalingstekens is `\"path\"` een bestand dat *path* heet, en dat is er niet. Zonder aanhalingstekens is `path` de naam, en gebruikt Python de waarde ervan, het echte pad. Haal de aanhalingstekens weg: `open(path)`.")
_WITH_COLON = Mistake("with-colon", syntax=r"^[ \t]*with\b[^:\n]*$",
                      en="The line `{line}` must end with a colon, `:`. The colon tells Python that the indented lines below belong to it.",
                      nl="De regel `{line}` moet eindigen op een dubbele punt, `:`. Die vertelt Python dat de ingesprongen regels eronder erbij horen.")
_TYPED_PATH = Mistake("typed-path", syntax=r"open\(\s*[\"'](?!path[\"'])",
                      en="Use the name `path`, which holds the right path: `open(path)`. A path typed between quotes has to be exactly right, and the check also tries another file.",
                      nl="Gebruik de naam `path`, die het goede pad bevat: `open(path)`. Een pad tussen aanhalingstekens moet precies kloppen, en de check probeert ook een ander bestand.")
_STRIP_CALL = Mistake("strip-no-brackets", syntax=r"\.strip\b(?!\s*\()",
                      en="`strip` needs round brackets to do its work: `line.strip()`.",
                      nl="`strip` heeft ronde haakjes nodig om zijn werk te doen: `line.strip()`.")

# The lesson is done when these are correct. "newline" is the choice question in the lesson.
REQUIRED = ["count-lines", "newline", "long-items"]

EXERCISES = {
    "count-lines": CodeBox(
        name="count-lines",
        start=f'path = "{SHOPPING}"\ncount = 0\n...\nprint(count)',
        solution=f'path = "{SHOPPING}"\ncount = 0\nwith open(path) as f:\n    for line in f:\n        count = count + 1\nprint(count)',
        outcome="stdout",
        expect="5\n",
        compare=same_lines,
        others=[{"path": PANTRY}],
        keep=[f'path = "{SHOPPING}"'],
        form=[Mistake("no-with", matches=lambda got, text: re.search(r"^\s*with\b", text, re.MULTILINE) is None, state="almost",
                      en="Right, but without `with`. Then the file stays open until Python cleans it up itself. Write `with open(path) as f:`, so it is closed again right after the indented lines.",
                      nl="Goed, maar zonder `with`. Dan blijft het bestand open tot Python het zelf opruimt. Schrijf `with open(path) as f:`, dan wordt het meteen na de ingesprongen regels weer gesloten.")],
        mistakes=_DOTS + [
            _CURLY, _PATH_QUOTED, _WITH_COLON, _TYPED_PATH,
            Mistake("path-chars", matches=_shows(str(len(SHOPPING))),
                    en="Your program shows `{got}`: the number of characters in the path itself. A loop over `path` goes through the text of the path. Open the file first: `with open(path) as f:`, and loop over `f`.",
                    nl="Je programma laat `{got}` zien: het aantal tekens in het pad zelf. Een loop over `path` gaat de tekst van het pad langs. Open eerst het bestand: `with open(path) as f:`, en loop over `f`."),
            Mistake("letters", matches=_shows(str(len(_SHOPPING_TEXT)), str(len(_SHOPPING_TEXT.replace("\n", "")))),
                    en="Your program shows `{got}`: the number of characters, not lines. Go through the file with `for line in f:`, and add 1 for each line.",
                    nl="Je programma laat `{got}` zien: het aantal tekens, geen regels. Ga het bestand langs met `for line in f:`, en tel 1 op bij elke regel."),
            Mistake("zero", matches=_shows("0"),
                    en="Your program shows `0`: the count never went up. Inside `for line in f:`, indented, add 1 to `count`.",
                    nl="Je programma laat `0` zien: de telling ging nooit omhoog. Tel in `for line in f:`, ingesprongen, 1 op bij `count`."),
            Mistake("each-round", matches=lambda got, text: _lines(got) == ["1", "2", "3", "4", "5"],
                    en="Your program shows the count after every line. `print(count)` belongs at the left edge, after the indented lines under `with`, so it runs once.",
                    nl="Je programma laat de telling na elke regel zien. `print(count)` hoort helemaal links, na de ingesprongen regels onder `with`, zodat hij één keer draait."),
            Mistake("extra-output", matches=lambda got, text: len(_lines(got)) > 1 and _lines(got)[-1] == "5",
                    en="The count is right, but your program also shows other lines. Keep only `print(count)`, at the end.",
                    nl="De telling klopt, maar je programma laat ook andere regels zien. Houd alleen `print(count)`, aan het eind."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a `with` block that opens `path`, and a loop that adds 1 to `count` for each line.",
                      "nl": "Vervang de `...` door een `with`-blok dat `path` opent, en een loop die bij elke regel 1 optelt bij `count`."},
            "cleared": {"en": "There is no code in the box any more. Type the lines back: `path = ...` with the path, `count = 0`, `...` and `print(count)`.",
                        "nl": "Er staat geen code meer in het vak. Typ de regels terug: `path = ...` met het pad, `count = 0`, `...` en `print(count)`."},
            "wrong": {"en": "Your program shows:\n\n```\n{got}\n```\n\nThat is not yet `5`, the number of lines in the file. Look at the counting loop in the worked example.",
                      "nl": "Je programma laat dit zien:\n\n```\n{got}\n```\n\nDat is nog niet `5`, het aantal regels in het bestand. Kijk naar de loop die telt in het uitgewerkte voorbeeld."},
            "other": {"en": "Right for this file. But with {where} instead, your program shows `{got}`, and it should show `{want}`. Did you type the number, or the path, yourself? Use `open(path)`, and let the loop count the lines.",
                      "nl": "Goed voor dit bestand. Maar met {where} laat je programma `{got}` zien, en dat moet `{want}` zijn. Typte je het getal, of het pad, zelf? Gebruik `open(path)`, en laat de loop de regels tellen."},
            "other_error": {"en": "Right for this file. But with {where} instead, your program stops with an error. Do you open the file in `path`, and not a path you typed yourself?",
                            "nl": "Goed voor dit bestand. Maar met {where} stopt je programma met een fout. Open je het bestand in `path`, en niet een pad dat je zelf typte?"},
            "correct": {"en": "Your program shows `5`: one for each line of the shopping list. It also works for another file.",
                        "nl": "Je programma laat `5` zien: één voor elke regel van het boodschappenlijstje. Het werkt ook voor een ander bestand."},
        },
    ),
    "long-items": CodeBox(
        name="long-items",
        start=f'path = "{SHOPPING}"\nwith open(path) as f:\n    for line in f:\n        ...',
        solution=f'path = "{SHOPPING}"\nwith open(path) as f:\n    for line in f:\n        item = line.strip()\n        if len(item) > 4:\n            print(item)',
        outcome="stdout",
        expect="bread\napples\ncheese\n",
        compare=same_lines,
        others=[{"path": PANTRY}],
        keep=[f'path = "{SHOPPING}"'],
        mistakes=_DOTS + [
            _CURLY, _PATH_QUOTED, _WITH_COLON, _TYPED_PATH, _STRIP_CALL,
            Mistake("strip-not-kept", matches=lambda got, text: re.search(r"^\s*line\.strip\(\)\s*$", text, re.MULTILINE) is not None,
                    en="`line.strip()` on a line of its own gives a new string, and then throws it away: `line` itself stays the same. Keep it under a name: `item = line.strip()`.",
                    nl="`line.strip()` op een eigen regel geeft een nieuwe string, en gooit die dan weg: `line` zelf blijft hetzelfde. Bewaar hem onder een naam: `item = line.strip()`."),
            Mistake("five", matches=lambda got, text: _lines(got) == ["apples", "cheese"],
                    en="`bread` is missing, and it has 5 letters, more than 4. Compare with `> 4`.",
                    nl="`bread` ontbreekt, en dat heeft 5 letters, meer dan 4. Vergelijk met `> 4`."),
            Mistake("at-least", matches=lambda got, text: _lines(got) == ["bread", "milk", "apples", "eggs", "cheese"] and ">=" in text,
                    en="Every item is shown, also `milk` and `eggs` with exactly 4 letters. Show only items with more than 4 letters: `>`, not `>=`.",
                    nl="Elk item staat erbij, ook `milk` en `eggs` met precies 4 letters. Laat alleen items met meer dan 4 letters zien: `>`, niet `>=`."),
            Mistake("no-filter", matches=lambda got, text: [line.strip() for line in _lines(got) if line.strip()] == ["bread", "milk", "apples", "eggs", "cheese"]
                    and re.search(r"\bif\b", text) is None,
                    en="Every item is shown. Add an `if` with `len(...) > 4`, so only the long items are shown.",
                    nl="Elk item staat erbij. Zet er een `if` met `len(...) > 4` bij, zodat alleen de lange items verschijnen."),
            Mistake("no-strip", matches=lambda got, text: "milk" in _lines(got) and re.search(r"len\(\s*line\s*\)", text) is not None,
                    en="`milk` is shown too, and it has 4 letters. Without `strip`, each line still has its newline at the end, so `len` counts one character too many. Use `line.strip()` before you count.",
                    nl="`milk` staat er ook bij, en dat heeft 4 letters. Zonder `strip` heeft elke regel nog zijn regeleinde, dus telt `len` één teken te veel. Gebruik `line.strip()` voordat je telt."),
            Mistake("blank-lines", matches=lambda got, text: "" in [line.strip() for line in str(got).rstrip("\n").splitlines()],
                    en="Your program shows an empty line after each item: the newline at the end of the line, plus the one `print` adds. Show `line.strip()` instead of `line`.",
                    nl="Je programma laat na elk item een lege regel zien: het regeleinde van de regel, plus dat van `print`. Laat `line.strip()` zien in plaats van `line`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` in the loop: strip the line, and show it when it has more than 4 letters.",
                      "nl": "Vervang de `...` in de loop: haal het regeleinde eraf, en laat de regel zien als hij meer dan 4 letters heeft."},
            "cleared": {"en": "There is no code in the box any more. Type the lines back: `path = ...` with the path, the `with` line, the `for` line, and the indented `...`.",
                        "nl": "Er staat geen code meer in het vak. Typ de regels terug: `path = ...` met het pad, de regel met `with`, de regel met `for`, en de ingesprongen `...`."},
            "wrong": {"en": "Your program shows:\n\n```\n{got}\n```\n\nThat is not yet `bread`, `apples` and `cheese`, each on its own line. Do you measure and show each line after `strip`, and keep only those longer than 4?",
                      "nl": "Je programma laat dit zien:\n\n```\n{got}\n```\n\nDat is nog niet `bread`, `apples` en `cheese`, elk op een eigen regel. Meet en laat je elke regel zien na `strip`, en houd je alleen die langer dan 4?"},
            "other": {"en": "Right for this file. But with {where} instead, your program shows:\n\n```\n{got}\n```\n\nand it should show:\n\n```\n{want}\n```\n\nDid you type the items, or the path, yourself? Let the loop go through the file with `open(path)`. And remember: the last line of a file may have no newline at all.",
                      "nl": "Goed voor dit bestand. Maar met {where} laat je programma dit zien:\n\n```\n{got}\n```\n\nen het zou dit moeten zijn:\n\n```\n{want}\n```\n\nTypte je de items, of het pad, zelf? Laat de loop het bestand langsgaan met `open(path)`. En denk eraan: de laatste regel van een bestand heeft soms helemaal geen regeleinde."},
            "correct": {"en": "Your program shows `bread`, `apples` and `cheese`: the items with more than 4 letters, without empty lines. It also works for another file.",
                        "nl": "Je programma laat `bread`, `apples` en `cheese` zien: de items met meer dan 4 letters, zonder lege regels. Het werkt ook voor een ander bestand."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_C = f'path = "{SHOPPING}"\ncount = 0\n'
_L = f'path = "{SHOPPING}"\nwith open(path) as f:\n    for line in f:\n'
CASES = [
    ("count-lines", _C + "...\nprint(count)", "empty", None),
    ("count-lines", "", "empty", None),
    ("count-lines", _C + "with open(path) as f:\n    for line in f:\n        count = count + 1\nprint(count)", "correct", None),
    ("count-lines", _C + "with open(path) as f:\n    for line in f:\n        count += 1\nprint(count)", "correct", None),
    ("count-lines", _C + "count = 5\nprint(count)", "almost", "other-value"),
    ("count-lines", _C + "with open(path) as f:\n    text = f.read()\ncount = len(text)\nprint(count)", "wrong", "letters"),
    ("count-lines", _C + "with open(path) as f:\n    for line in f:\n        count = count + 1\n        print(count)", "wrong", "each-round"),
    ("count-lines", _C + 'with open("path") as f:\n    for line in f:\n        count = count + 1\nprint(count)', "error", "path-quoted"),
    ("count-lines", _C + "with open(path) as f\n    for line in f:\n        count = count + 1\nprint(count)", "error", "with-colon"),
    ("count-lines", _C + "with open(path) as f:\n    pass\nfor line in f:\n    count = count + 1\nprint(count)", "error", None),
    ("count-lines", _C + "with open(path) as f:\n    pass\ntext = f.read()\nprint(count)", "error", None),
    ("count-lines", _C + "for line in path:\n    count = count + 1\nprint(count)", "wrong", "path-chars"),
    ("count-lines", _C + "with open(path) as f:\n    for line in f:\n        print(line.strip())\n        count = count + 1\nprint(count)", "wrong", "extra-output"),
    ("count-lines", _C + "f = open(path)\nfor line in f:\n    count = count + 1\nprint(count)", "almost", "no-with"),
    ("count-lines", _C + 'with open("lessons/py-files/shopping.txt") as f:\n    for line in f:\n        count = count + 1\nprint(count)', "almost", "other-value"),
    ("count-lines", _C + 'with open("lessons/shopping.txt") as f:\n    for line in f:\n        count = count + 1\nprint(count)', "error", "typed-path"),
    ("long-items", _L + "        ...", "empty", None),
    ("long-items", _L + "        item = line.strip()\n        if len(item) > 4:\n            print(item)", "correct", None),
    ("long-items", _L + "        if len(line.strip()) > 4:\n            print(line.strip())", "correct", None),
    ("long-items", _L + "        if len(line) > 4:\n            print(line.strip())", "wrong", "no-strip"),
    ("long-items", _L + "        if len(line.strip()) > 4:\n            print(line)", "wrong", "blank-lines"),
    ("long-items", _L + "        item = line.strip()\n        if len(item) >= 4:\n            print(item)", "wrong", "at-least"),
    ("long-items", f'path = "{SHOPPING}"\nprint("bread")\nprint("apples")\nprint("cheese")', "almost", "other-value"),
    ("long-items", _L + "        print(line.strip())", "wrong", "no-filter"),
    ("long-items", _L + "        line.strip()\n        if len(line) > 4:\n            print(line)", "wrong", "strip-not-kept"),
    ("long-items", _L + "        item = line.strip()\n        if len(item) > 5:\n            print(item)", "wrong", "five"),
    ("long-items", _L + "        item = line.strip\n        if len(item) > 4:\n            print(item)", "error", "strip-no-brackets"),
    ("long-items", _L + "        item = line.strp()\n        if len(item) > 4:\n            print(item)", "error", None),
    ("long-items", _L + "        if len(line) > 5:\n            print(line.strip())", "almost", "other-value"),
]
