# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the project lesson py-project-letter-counts: count_letters and count_pairs, each saved in the
learner's project (work/projects/llm/) when right, and a bar chart of the book drawn with the learner's own module."""
import re
from functools import cache
from pathlib import Path

from everything_ai import paths, projects
from everything_ai.check import FunctionBox, Mistake, PlotBox, run_code

LETTERS_LINE = 'LETTERS = "abcdefghijklmnopqrstuvwxyz"\n\n\n'

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own lines.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen regels.")
_NO_LETTERS = Mistake("no-letters-line", syntax=r"(?s)\A(?!.*^LETTERS\s*=).*\bLETTERS\b",
                      en="Python does not know `LETTERS`: the box needs the line `LETTERS = \"abcdefghijklmnopqrstuvwxyz\"` at the top. Put it back.",
                      nl="Python kent `LETTERS` niet: het vak heeft de regel `LETTERS = \"abcdefghijklmnopqrstuvwxyz\"` bovenaan nodig. Zet hem terug.")
# Adding 1 to a key that is not there yet stops with a KeyError; any other error, such as a typo, is something else.
_KEY = Mistake("add-before-key", error=lambda e: isinstance(e, KeyError),
               en="The first time a key comes by, it is not in `counts` yet, so `counts[...] + 1` stops. Look first with `if ... in counts:`; if it is not there, start it at 1.",
               nl="De eerste keer dat een key langskomt, staat hij nog niet in `counts`, dus stopt `counts[...] + 1`. Kijk eerst met `if ... in counts:`; staat hij er niet in, begin hem dan bij 1.")

# The lesson is done when these are correct.
REQUIRED = ["count-letters", "count-pairs", "letter-chart"]

_LETTERS_START = LETTERS_LINE + "def count_letters(text):\n    counts = {}\n    for ch in text.lower():\n        ...\n    return counts\n"
_LETTERS = LETTERS_LINE + ("def count_letters(text):\n    counts = {}\n    for ch in text.lower():\n        if ch in LETTERS:\n"
                           "            if ch in counts:\n                counts[ch] = counts[ch] + 1\n            else:\n"
                           "                counts[ch] = 1\n    return counts")
_PAIRS_START = LETTERS_LINE + ("def count_pairs(text):\n    text = text.lower()\n    counts = {}\n    for i in range(len(text) - 1):\n"
                               "        pair = text[i:i + 2]\n        ...\n    return counts\n")
_PAIRS = LETTERS_LINE + ("def count_pairs(text):\n    text = text.lower()\n    counts = {}\n    for i in range(len(text) - 1):\n"
                         "        pair = text[i:i + 2]\n        if pair[0] in LETTERS and pair[1] in LETTERS:\n"
                         "            if pair in counts:\n                counts[pair] = counts[pair] + 1\n            else:\n"
                         "                counts[pair] = 1\n    return counts")

_CHART_HEAD = ("import matplotlib.pyplot as plt\nfrom llm import letters\n\n"
               "with open(\"data/tiny-shakespeare.txt\") as f:\n    book = f.read()\n\n"
               "counts = letters.count_letters(book)\nnames = sorted(counts)\n")
_CHART_START = _CHART_HEAD + "...\nplt.show()"
_CHART_DRAW = ("heights = [counts[ch] for ch in names]\nplt.bar(names, heights)\n"
               "plt.title(\"Letters in Shakespeare\")\nplt.ylabel(\"how often\")\nplt.show()")
_CHART = _CHART_HEAD + _CHART_DRAW
# What the check compares with: the course counts by itself, so a wrong module of the learner shows up.
_CHART_COURSE = (_LETTERS + "\n\n\nimport matplotlib.pyplot as plt\n\nwith open(\"data/tiny-shakespeare.txt\") as f:\n    book = f.read()\n\n"
                 "counts = count_letters(book)\nnames = sorted(counts)\n" + _CHART_DRAW)


def _letters_file() -> Path:
    return projects.folder() / "llm" / "letters.py"


def _course(code: str, name: str):
    """The course's own version of a function, from its solution."""
    space = {}
    exec(code, space)
    return space[name]


_PANGRAM = "The quick brown fox jumps over the lazy dog, AND THEN SOME!"


def _saved_ok() -> bool | None:
    """None when the learner's module is not saved yet; else whether its count_letters still counts right."""
    if not _letters_file().exists():
        return None
    run = run_code(f"from llm import letters\nletters.count_letters({_PANGRAM!r})")
    return run.error is None and run.value == _course(_LETTERS, "count_letters")(_PANGRAM)


def pairs_ok() -> bool | None:
    """For the lesson's demo: None when count_pairs is not saved yet; else whether it still counts right."""
    if not (projects.folder() / "llm" / "pairs.py").exists():
        return None
    run = run_code(f"from llm import pairs\npairs.count_pairs({_PANGRAM!r})")
    return run.error is None and run.value == _course(_PAIRS, "count_pairs")(_PANGRAM)


@cache
def _book_counts() -> tuple[int, ...]:
    """The course's own count of the book: the height of each bar, from a to z."""
    counts = _course(_LETTERS, "count_letters")((paths.ROOT / "data" / "tiny-shakespeare.txt").read_text(encoding="utf-8"))
    return tuple(counts[ch] for ch in sorted(counts))


def _other_order(got) -> bool:
    """Bars a to z with the right counts, but not each under its own letter."""
    if not (got and got["charts"]) or got["charts"][0]["kind"] != "bar":
        return False
    x, y = got["charts"][0]["x"], got["charts"][0]["y"]
    right = list(_book_counts())
    return list(x) == list("abcdefghijklmnopqrstuvwxyz") and list(y) != right and sorted(y) == sorted(right)


_REDO = {"en": " Did you change that file? Delete it, and check exercise 1 again: its check then saves your function again.",
         "nl": " Heb je dat bestand veranderd? Verwijder het, en laat oefening 1 opnieuw nakijken: die check bewaart je functie dan opnieuw."}

_COUNT_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "arguments": {"en": "The check calls `{call}`, with one text, and the `def` line of your function has another number of names between its parentheses. Keep `(text)`.",
                  "nl": "De check roept `{call}` aan, met één tekst, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(text)`."},
}

EXERCISES = {
    "count-letters": FunctionBox(
        name="count-letters",
        function="count_letters",
        start=_LETTERS_START,
        solution=_LETTERS,
        visible=[("Hello, World",)],
        hidden=[("",), ("AAa b!",), ("To be, or not to be",), ("Zz9 zz",)],
        python_values=True,
        save_as=("llm", "letters"),
        first=[_DOTS],
        mistakes=[
            _NO_LETTERS, _KEY,
            Mistake("no-filter", wrong=_LETTERS.replace("        if ch in LETTERS:\n", "        if True:\n"),
                    en="`{call}` gives `{got}`: it also counts spaces and commas. Count a character only when it is a letter: `if ch in LETTERS:`.",
                    nl="`{call}` geeft `{got}`: hij telt ook spaties en komma's. Tel een teken alleen als het een letter is: `if ch in LETTERS:`."),
            Mistake("no-lower", wrong=_LETTERS.replace("text.lower()", "text"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the capitals are missing. `LETTERS` has only small letters, so the loop needs the text in small letters: keep `text.lower()`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de hoofdletters ontbreken. `LETTERS` heeft alleen kleine letters, dus heeft de loop de tekst in kleine letters nodig: houd `text.lower()`."),
            Mistake("always-one", wrong=_LETTERS.replace("counts[ch] = counts[ch] + 1", "counts[ch] = 1"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. Every letter has the count 1: when a letter is already in `counts`, add 1 to what it has.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Elke letter heeft de telling 1: staat een letter al in `counts`, tel dan 1 op bij wat hij heeft."),
            Mistake("start-at-zero", wrong=_LETTERS.replace("counts[ch] = 1", "counts[ch] = 0"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: every count is 1 too low. When a new letter comes by, it has been seen once already: start it at 1.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elke telling is 1 te laag. Komt er een nieuwe letter langs, dan is hij al één keer gezien: begin hem bij 1."),
            Mistake("return-in-loop", wrong=_LETTERS.replace("\n    return counts", "\n        return counts"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. Your `return` is indented under the loop, so the function stops after the first character. Put `return counts` after the loop, as far in as `for`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Je `return` is ingesprongen onder de loop, dus stopt de functie na het eerste teken. Zet `return counts` na de loop, even ver naar binnen als `for`."),
        ],
        texts={
            **_COUNT_TEXTS,
            "empty": {"en": "Replace the `...` with the lines that count `ch`, but only when it is a letter.",
                      "nl": "Vervang de `...` door de regels die `ch` tellen, maar alleen als het een letter is."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does your loop count each letter once, and only letters?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Telt je loop elke letter één keer, en alleen letters?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it also work for an empty text, and for capitals?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij ook voor een lege tekst, en voor hoofdletters?"},
            "correct": {"en": "Your function is right in all {n} cases. The check saved it in your project, as `{saved}`: exercise 3 and later lessons import it from there.",
                        "nl": "Je functie klopt in alle {n} gevallen. De check heeft hem in je project bewaard, als `{saved}`: oefening 3 en latere lessen halen hem daar binnen."},
            "kept": {"en": "Your function is right in all {n} cases. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je functie klopt in alle {n} gevallen. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren? Verwijder het bestand en laat opnieuw nakijken."},
        },
    ),
    "count-pairs": FunctionBox(
        name="count-pairs",
        function="count_pairs",
        start=_PAIRS_START,
        solution=_PAIRS,
        visible=[("Hello",)],
        hidden=[("",), ("a b",), ("aaa",), ("To be!",), ("x",)],
        python_values=True,
        save_as=("llm", "pairs"),
        first=[_DOTS],
        mistakes=[
            _NO_LETTERS, _KEY,
            Mistake("no-filter", wrong=_PAIRS.replace("        if pair[0] in LETTERS and pair[1] in LETTERS:\n", "        if True:\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: it also counts pairs with a space or a sign in them. Count a pair only when both characters are letters.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: hij telt ook paren met een spatie of een teken erin. Tel een paar alleen als beide tekens letters zijn."),
            Mistake("first-only", wrong=_PAIRS.replace("if pair[0] in LETTERS and pair[1] in LETTERS:", "if pair[0] in LETTERS:"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. Your `if` looks only at the first character of the pair. Look at both: `pair[0]` and `pair[1]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Je `if` kijkt alleen naar het eerste teken van het paar. Kijk naar allebei: `pair[0]` en `pair[1]`."),
            Mistake("second-only", wrong=_PAIRS.replace("if pair[0] in LETTERS and pair[1] in LETTERS:", "if pair[1] in LETTERS:"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. Your `if` looks only at the second character of the pair. Look at both: `pair[0]` and `pair[1]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Je `if` kijkt alleen naar het tweede teken van het paar. Kijk naar allebei: `pair[0]` en `pair[1]`."),
            Mistake("no-lower", wrong=_PAIRS.replace("    text = text.lower()\n", ""),
                    en="`{call}` gives `{got}`, and it should give `{want}`: pairs with a capital are missing. `LETTERS` has only small letters, so the text needs to be in small letters: keep `text = text.lower()`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: paren met een hoofdletter ontbreken. `LETTERS` heeft alleen kleine letters, dus moet de tekst in kleine letters staan: houd `text = text.lower()`."),
            Mistake("always-one", wrong=_PAIRS.replace("counts[pair] = counts[pair] + 1", "counts[pair] = 1"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. Every pair has the count 1: when a pair is already in `counts`, add 1 to what it has.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Elk paar heeft de telling 1: staat een paar al in `counts`, tel dan 1 op bij wat het heeft."),
            Mistake("return-in-loop", wrong=_PAIRS.replace("\n    return counts", "\n        return counts"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. Your `return` is indented under the loop, so the function stops after the first pair. Put `return counts` after the loop, as far in as `for`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Je `return` is ingesprongen onder de loop, dus stopt de functie na het eerste paar. Zet `return counts` na de loop, even ver naar binnen als `for`."),
        ],
        texts={
            **_COUNT_TEXTS,
            "empty": {"en": "Replace the `...` with the lines that count `pair`, but only when both its characters are letters.",
                      "nl": "Vervang de `...` door de regels die `pair` tellen, maar alleen als beide tekens letters zijn."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does your loop count each pair of two letters next to each other, and only pairs of letters?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Telt je loop elk paar van twee letters naast elkaar, en alleen paren van letters?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it also work for a text with spaces, an empty text, and a pair that comes back?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij ook voor een tekst met spaties, een lege tekst, en een paar dat terugkomt?"},
            "correct": {"en": "Your function is right in all {n} cases. The check saved it in your project, as `{saved}`: in a later lesson, these pair counts become your first language model.",
                        "nl": "Je functie klopt in alle {n} gevallen. De check heeft hem in je project bewaard, als `{saved}`: in een latere les worden deze paartellingen je eerste language model."},
            "kept": {"en": "Your function is right in all {n} cases. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je functie klopt in alle {n} gevallen. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren? Verwijder het bestand en laat opnieuw nakijken."},
        },
    ),
    "letter-chart": PlotBox(
        name="letter-chart",
        start=_CHART_START,
        solution=_CHART,
        reference=_CHART_COURSE,
        first=[
            _DOTS,
            Mistake("not-saved-yet", state="empty", matches=lambda got, text: "llm" in text and _saved_ok() is None,
                    en="Your module is not in your project yet. The check of exercise 1 saves it, as `work/projects/llm/letters.py`, once your `count_letters` is right. Finish exercise 1 first; this check then runs again by itself.",
                    nl="Je module staat nog niet in je project. De check van oefening 1 bewaart hem, als `work/projects/llm/letters.py`, zodra je `count_letters` klopt. Maak eerst oefening 1 af; deze check draait daarna vanzelf opnieuw."),
            Mistake("module-wrong", matches=lambda got, text: "llm" in text and _saved_ok() is False,
                    en="The `count_letters` in your module `work/projects/llm/letters.py` no longer counts right, or stops with an error." + _REDO["en"],
                    nl="De `count_letters` in je module `work/projects/llm/letters.py` telt niet meer goed, of stopt met een fout." + _REDO["nl"]),
        ],
        mistakes=[
            Mistake("bar-swapped", matches=lambda got, text: bool(got and got["charts"]) and got["charts"][0]["kind"] == "bar"
                    and all(isinstance(x, (int, float)) for x in got["charts"][0]["x"]) and len(got["charts"][0]["x"]) == 26,
                    en="`plt.bar` gets the names first, along the bottom, and then the heights: `plt.bar(names, heights)`.",
                    nl="`plt.bar` krijgt eerst de namen, langs de onderkant, en dan de hoogtes: `plt.bar(names, heights)`."),
            Mistake("other-order", matches=lambda got, text: _other_order(got),
                    en="Your bars have the right counts, but not each under its own letter: the heights are in another order than the letters in `names`. Make the heights by going through `names`, in its own order.",
                    nl="Je staven hebben de goede tellingen, maar niet elk onder zijn eigen letter: de hoogtes staan in een andere volgorde dan de letters in `names`. Maak de hoogtes door `names` langs te gaan, in zijn eigen volgorde."),
        ],
        form=[
            Mistake("own-module", state="almost",
                    matches=lambda got, text: not (re.search(r"\bllm\b", text) and re.search(r"\bcount_letters\s*\(", text)),
                    en="Right chart, but it was not counted with your own module. Keep the lines `from llm import letters` and `counts = letters.count_letters(book)`: this exercise puts your project to work.",
                    nl="Goede grafiek, maar hij is niet geteld met je eigen module. Houd de regels `from llm import letters` en `counts = letters.count_letters(book)`: in deze oefening gaat je project aan het werk."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the lines that make a list of heights, one count per letter in `names`, and draw them as bars with a title and a y-axis label.",
                      "nl": "Vervang de `...` door de regels die een lijst met hoogtes maken, één telling per letter in `names`, en teken die als staven met een titel en een label bij de y-as."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "nothing": {"en": "Your code draws nothing yet. Draw the bars with `plt.bar(names, heights)`, before `plt.show()`.",
                        "nl": "Je code tekent nog niets. Teken de staven met `plt.bar(names, heights)`, vóór `plt.show()`."},
            "kind": {"en": "Your code draws {kind}, and this exercise asks for {want_kind}: a bar per letter. Draw it with `{draw}`.",
                     "nl": "Je code tekent {kind}, en deze oefening vraagt om {want_kind}: een staaf per letter. Teken die met `{draw}`."},
            "values": {"en": "Your bars are `{got_x}` with heights `{got}`, and they should be the letters a to z, each as high as its count in the book. Does each height come from `counts`, for the letter of `names` at the same place?",
                       "nl": "Je staven zijn `{got_x}` met hoogtes `{got}`, en dat moeten de letters a tot en met z zijn, elk zo hoog als zijn telling in het boek. Komt elke hoogte uit `counts`, voor de letter van `names` op dezelfde plek?"},
            "title": {"en": "Right chart, but it has no title yet. Add `plt.title(...)` with what the chart shows.",
                      "nl": "Goede grafiek, maar hij heeft nog geen titel. Voeg `plt.title(...)` toe, met wat de grafiek laat zien."},
            "xlabel": {"en": "Right chart, but the x-axis has no label yet. Add `plt.xlabel(...)`.",
                       "nl": "Goede grafiek, maar de x-as heeft nog geen label. Voeg `plt.xlabel(...)` toe."},
            "ylabel": {"en": "Right chart, but the y-axis has no label yet. Add `plt.ylabel(...)` with what the heights mean.",
                       "nl": "Goede grafiek, maar de y-as heeft nog geen label. Voeg `plt.ylabel(...)` toe, met wat de hoogtes betekenen."},
            "correct": {"en": "Your chart is right, and it was made with your own module: `count_letters` from your project counted the whole book. The tallest bar is `e`.",
                        "nl": "Je grafiek klopt, en hij is gemaakt met je eigen module: `count_letters` uit je project telde het hele boek. De hoogste staaf is `e`."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
# The tests check every solution first, so count_letters is saved before the chart cases run.
_L = LETTERS_LINE + "def count_letters(text):\n    counts = {}\n    for ch in text.lower():\n"
_P = LETTERS_LINE + "def count_pairs(text):\n    text = text.lower()\n    counts = {}\n    for i in range(len(text) - 1):\n        pair = text[i:i + 2]\n"
CASES = [
    ("count-letters", _LETTERS_START, "empty", None),
    ("count-letters", _LETTERS, "correct", None),
    ("count-letters", _L + "        if ch in LETTERS:\n            if ch not in counts:\n                counts[ch] = 0\n            counts[ch] = counts[ch] + 1\n    return counts", "correct", None),
    ("count-letters", _LETTERS.replace("        if ch in LETTERS:\n", "        if True:\n"), "wrong", "no-filter"),
    ("count-letters", _LETTERS.replace("counts[ch] = counts[ch] + 1", "counts[ch] = 1"), "wrong", "always-one"),
    ("count-letters", _L + "        if ch in LETTERS:\n            counts[ch] = counts[ch] + 1\n    return counts", "error", "add-before-key"),
    ("count-letters", _LETTERS.replace(LETTERS_LINE, ""), "error", "no-letters-line"),
    ("count-letters", _LETTERS.replace("text.lower()", "text"), "wrong", "no-lower"),
    ("count-letters", _LETTERS.replace("counts[ch] = 1", "counts[ch] = 0"), "wrong", "start-at-zero"),
    ("count-letters", _LETTERS.replace("\n    return counts", "\n        return counts"), "wrong", "return-in-loop"),
    ("count-letters", _LETTERS.replace("                counts[ch] = 1", "                count[ch] = 1"), "error", None),
    ("count-letters", _L + "        if ch in LETTERS:\n            counts[ch] = counts.get(ch, 0) + 1\n    return counts", "correct", None),
    ("count-pairs", _PAIRS_START, "empty", None),
    ("count-pairs", _PAIRS, "correct", None),
    ("count-pairs", _PAIRS.replace("        if pair[0] in LETTERS and pair[1] in LETTERS:\n", "        if True:\n"), "wrong", "no-filter"),
    ("count-pairs", _PAIRS.replace("if pair[0] in LETTERS and pair[1] in LETTERS:", "if pair[0] in LETTERS:"), "wrong", "first-only"),
    ("count-pairs", _PAIRS.replace("if pair[0] in LETTERS and pair[1] in LETTERS:", "if pair[1] in LETTERS:"), "wrong", "second-only"),
    ("count-pairs", _PAIRS.replace("    text = text.lower()\n", ""), "wrong", "no-lower"),
    ("count-pairs", _PAIRS.replace("\n    return counts", "\n        return counts"), "wrong", "return-in-loop"),
    ("count-pairs", _P + "        if pair[0] in LETTERS:\n            if pair[1] in LETTERS:\n                if pair in counts:\n                    counts[pair] += 1\n"
                    "                else:\n                    counts[pair] = 1\n    return counts", "correct", None),
    ("count-pairs", _PAIRS.replace("counts[pair] = counts[pair] + 1", "counts[pair] = 1"), "wrong", "always-one"),
    ("count-pairs", _P + "        if pair[0] in LETTERS and pair[1] in LETTERS:\n            counts[pair] += 1\n    return counts", "error", "add-before-key"),
    ("letter-chart", _CHART_START, "empty", None),
    ("letter-chart", _CHART, "correct", None),
    ("letter-chart", _CHART.replace("plt.bar(names, heights)", "plt.plot(names, heights)"), "wrong", "kind"),
    ("letter-chart", _CHART.replace("plt.bar(names, heights)", "plt.bar(heights, names)"), "wrong", "bar-swapped"),
    ("letter-chart", _CHART.replace("plt.ylabel(\"how often\")\n", ""), "almost", "no-ylabel"),
    ("letter-chart", _CHART.replace("heights = [counts[ch] for ch in names]", "heights = [counts[ch] for ch in sorted(counts, reverse=True)]"), "wrong", "other-order"),
    ("letter-chart", _CHART.replace("heights = [counts[ch] for ch in names]", "heights = [1 for ch in names]"), "wrong", "values"),
    ("letter-chart", _CHART_COURSE, "almost", "own-module"),
]
