# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-kinds-of-learning: five tasks sorted by their kind of learning, and training pairs made from
a list of words, the way self-supervised learning makes its own labels."""
import re

from everything_ai.check import FunctionBox, Mistake, Sort

# The lesson is done when these are correct. "spam-kind" is the choice question in the lesson.
REQUIRED = ["spam-kind", "kinds", "word-pairs"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_START = "def word_pairs(words):\n    pairs = []\n    ...\n    return pairs\n"
_SOLUTION = ("def word_pairs(words):\n    pairs = []\n    for i in range(len(words) - 1):\n        pairs.append((words[i], words[i + 1]))\n"
             "    return pairs")


# A `return` inside the loop: a line deeper in than the `for` line above it, with no line in between at the depth of the `for`.
_RETURN_IN_LOOP = re.compile(r"^([ \t]*)for\b[^\n]*:[ \t]*\n(?:\1[ \t]+[^\n]*\n|[ \t]*\n)*?\1[ \t]+return\b", re.MULTILINE)


def _w(body: str) -> str:
    return "def word_pairs(words):\n    pairs = []\n" + body + "    return pairs"


def _show(value) -> str:
    """Pairs the way the lesson writes them: `[("the", "cat")]`, strings with double quotes, also inside a tuple."""
    if isinstance(value, str):
        return '"' + value + '"'
    if isinstance(value, list):
        return "[" + ", ".join(_show(v) for v in value) + "]"
    if isinstance(value, tuple):
        return "(" + ", ".join(_show(v) for v in value) + ("," if len(value) == 1 else "") + ")"
    return repr(value)


EXERCISES = {
    "kinds": Sort(
        name="kinds",
        groups={
            "supervised": {"en": "Supervised", "nl": "Supervised"},
            "unsupervised": {"en": "Unsupervised", "nl": "Unsupervised"},
            "self": {"en": "Self-supervised", "nl": "Self-supervised"},
            "reinforcement": {"en": "Reinforcement", "nl": "Reinforcement"},
        },
        items={
            "fraud": {"en": "**Fraud check.** A bank learns to spot fraud from a million card payments that its staff marked as fraud or not fraud.",
                      "nl": "**Fraudecheck.** Een bank leert fraude herkennen uit een miljoen kaartbetalingen die medewerkers als fraude of geen fraude hadden gemarkeerd."},
            "puzzle": {"en": "**Photo puzzle.** A model learns from millions of photos that nobody labeled: it covers a square of each photo, and predicts what was under it.",
                       "nl": "**Fotopuzzel.** Een model leert van miljoenen foto's die niemand labelde: het dekt in elke foto een vierkantje af, en voorspelt wat eronder zat."},
            "news": {"en": "**News topics.** A news site sorts 20,000 articles into groups about the same subject. Nobody made a list of subjects first.",
                     "nl": "**Nieuwsonderwerpen.** Een nieuwssite verdeelt 20.000 artikelen in groepen over hetzelfde onderwerp. Niemand maakte eerst een lijst met onderwerpen."},
            "chess": {"en": "**Chess.** A program learns chess by playing many games against itself. It gets +1 when it wins and −1 when it loses.",
                      "nl": "**Schaken.** Een programma leert schaken door veel partijen tegen zichzelf te spelen. Het krijgt +1 als het wint en −1 als het verliest."},
            "translate": {"en": "**Translation.** A model learns to translate from a million English sentences, each with the Dutch translation that a person wrote.",
                          "nl": "**Vertalen.** Een model leert vertalen van een miljoen Engelse zinnen, elk met de Nederlandse vertaling die een mens schreef."},
        },
        solution={"fraud": "supervised", "puzzle": "self", "news": "unsupervised", "chess": "reinforcement", "translate": "supervised"},
        nudge={
            "fraud": {"en": "Where do the answers fraud or not fraud come from?",
                      "nl": "Waar komen de antwoorden fraude of geen fraude vandaan?"},
            "puzzle": {"en": "Nobody labeled the photos. But is there a right answer to predict, and where does it come from?",
                       "nl": "Niemand labelde de foto's. Maar is er een goed antwoord om te voorspellen, en waar komt dat vandaan?"},
            "news": {"en": "Is there a right answer for each article, or only articles to find structure in?",
                     "nl": "Is er een goed antwoord bij elk artikel, of alleen artikelen om structuur in te vinden?"},
            "chess": {"en": "Does it learn from examples with answers, or from what happens after its moves?",
                      "nl": "Leert het van voorbeelden met antwoorden, of van wat er na zijn zetten gebeurt?"},
            "translate": {"en": "Where does each Dutch sentence come from: from the English sentence itself, or from a person?",
                          "nl": "Waar komt elke Nederlandse zin vandaan: uit de Engelse zin zelf, of van een mens?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the tasks again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de taken opnieuw uit."},
            "empty": {"en": "Pick a kind of learning for each task.",
                      "nl": "Kies voor elke taak een soort leren."},
            "unfinished": {"en": "Everything you picked so far is right. Pick a kind for the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe koos, klopt. Kies ook voor de rest een soort (nog {n})."},
            "wrong": {"en": "{item} belongs to another kind of learning.",
                      "nl": "{item} hoort bij een andere soort leren."},
            "correct": {"en": "All {n} tasks have their kind. Each time the question was the same: where do the right answers come from?",
                        "nl": "Alle {n} taken hebben hun soort. Elke keer was de vraag dezelfde: waar komen de goede antwoorden vandaan?"},
        },
    ),
    "word-pairs": FunctionBox(
        name="word-pairs",
        function="word_pairs",
        start=_START,
        solution=_SOLUTION,
        visible=[(["the", "cat", "sat", "down"],)],
        hidden=[(["to", "be"],), (["one"],), ([],), (["a", "b", "a", "b"],), (["the", "dog", "ran", "home", "fast"],)],
        show=_show,
        first=[_DOTS],
        mistakes=[
            Mistake("too-far", error=lambda e: isinstance(e, IndexError),
                    en="With `{call}`, your function asks for a word at a place the list does not have. Is it the word after the last word? That one is not there: stop the loop one place earlier, with `range(len(words) - 1)`.",
                    nl="Met `{call}` vraagt je functie om een woord op een plek die de lijst niet heeft. Is het het woord na het laatste woord? Dat is er niet: stop de loop één plek eerder, met `range(len(words) - 1)`."),
            Mistake("list-pairs", matches=lambda got, text: isinstance(got, list) and len(got) > 0 and all(isinstance(p, list) for p in got),
                    en="`{call}` gives `{got}`: each pair is a list, in square brackets. Make each pair a tuple, in round brackets, as in the worked example: `(words[i], words[i + 1])`.",
                    nl="`{call}` geeft `{got}`: elk paar is een lijst, met vierkante haken. Maak van elk paar een tuple, met ronde haken, zoals in het uitgewerkte voorbeeld: `(words[i], words[i + 1])`."),
            Mistake("return-in-loop", matches=lambda got, text: isinstance(got, list) and len(got) == 1 and _RETURN_IN_LOOP.search(text) is not None,
                    en="`{call}` gives `{got}`: only the first pair. Your `return` is indented under the `for` line, so the function stops in the first round of the loop. Put `return pairs` after the loop, as far in as `pairs = []`.",
                    nl="`{call}` geeft `{got}`: alleen het eerste paar. Je `return` staat ingesprongen onder de `for`-regel, dus de functie stopt in de eerste ronde van de loop. Zet `return pairs` na de loop, even ver ingesprongen als `pairs = []`."),
            Mistake("same-word", wrong=_w("    for i in range(len(words) - 1):\n        pairs.append((words[i], words[i]))\n"),
                    en="`{call}` gives `{got}`: each word is paired with itself. The label is the word after it: `words[i + 1]`.",
                    nl="`{call}` geeft `{got}`: elk woord is aan zichzelf gekoppeld. Het label is het woord erna: `words[i + 1]`."),
            Mistake("swapped", wrong=_w("    for i in range(len(words) - 1):\n        pairs.append((words[i + 1], words[i]))\n"),
                    en="`{call}` gives `{got}`: each pair the other way round. The input comes first, then the label: the word, and then the word after it.",
                    nl="`{call}` geeft `{got}`: elk paar andersom. Eerst de input, dan het label: het woord, en dan het woord erna."),
            Mistake("one-short", wrong=_w("    for i in range(len(words) - 2):\n        pairs.append((words[i], words[i + 1]))\n"),
                    en="`{call}` gives `{got}`: the last pair is missing. With `len(words) - 1`, the loop goes up to the second-to-last word, the last one with a word after it.",
                    nl="`{call}` geeft `{got}`: het laatste paar ontbreekt. Met `len(words) - 1` gaat de loop tot en met het voorlaatste woord, het laatste woord met nog een woord erna."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a loop that adds a pair to `pairs` for each word that has a word after it: the word, and the next word.",
                      "nl": "Vervang de `...` door een loop die een paar aan `pairs` toevoegt voor elk woord met nog een woord erna: het woord, en het volgende woord."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with one list of words, and the `def` line of your function has a different number of names between its parentheses. Keep `(words)`.",
                          "nl": "De check roept `{call}` aan, met één lijst woorden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(words)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Is each pair a word and the word right after it, in that order?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Is elk paar een woord en het woord direct erna, in die volgorde?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for any number of words, also for one word, or none?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor elk aantal woorden, ook voor één woord, of geen?"},
            "correct": {"en": "Your function is right in all {n} cases. Four words give three pairs: nobody wrote a label, the text itself gives them. That is self-supervised learning.",
                        "nl": "Je functie klopt in alle {n} gevallen. Vier woorden geven drie paren: niemand schreef een label, de tekst zelf geeft ze. Dat is self-supervised learning."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_KINDS = {"fraud": "supervised", "puzzle": "self", "news": "unsupervised", "chess": "reinforcement", "translate": "supervised"}
CASES = [
    ("kinds", {}, "empty", None),
    ("kinds", _KINDS, "correct", None),
    ("kinds", {"fraud": "supervised", "puzzle": "self"}, "wrong", "unfinished"),
    ("kinds", {**_KINDS, "puzzle": "unsupervised"}, "wrong", "wrong:puzzle"),
    ("kinds", {**_KINDS, "translate": "self"}, "wrong", "wrong:translate"),
    ("kinds", {**_KINDS, "chess": "supervised"}, "wrong", "wrong:chess"),
    ("kinds", {**_KINDS, "news": "self"}, "wrong", "wrong:news"),
    ("word-pairs", _START, "empty", None),
    ("word-pairs", _SOLUTION, "correct", None),
    ("word-pairs", _w("    for i in range(1, len(words)):\n        pairs.append((words[i - 1], words[i]))\n"), "correct", None),
    ("word-pairs", "def word_pairs(words):\n    return list(zip(words, words[1:]))", "correct", None),
    ("word-pairs", _w("    for i in range(len(words)):\n        pairs.append((words[i], words[i + 1]))\n"), "error", "too-far"),
    ("word-pairs", _w("    for i in range(len(words) - 1):\n        pairs.append((words[i], words[i]))\n"), "wrong", "same-word"),
    ("word-pairs", _w("    for i in range(len(words) - 1):\n        pairs.append((words[i + 1], words[i]))\n"), "wrong", "swapped"),
    ("word-pairs", _w("    for i in range(len(words) - 2):\n        pairs.append((words[i], words[i + 1]))\n"), "wrong", "one-short"),
    ("word-pairs", _w("    for i in range(len(words) - 1):\n        pairs.append([words[i], words[i + 1]])\n"), "wrong", "list-pairs"),
    ("word-pairs", _w("    for i in range(len(words) - 1):\n        pairs.append((words[i], words[i + 1]))\n        return pairs\n"), "wrong", "return-in-loop"),
    ("word-pairs", "def word_pairs(words):\n    pairs = []\n    for i in range(len(words) - 1):\n        pairs.append((words[i], words[i + 1]))\n\n        return pairs", "wrong", "return-in-loop"),
    ("word-pairs", "def word_pairs(words):\n    return [(\"the\", \"cat\"), (\"cat\", \"sat\"), (\"sat\", \"down\")]", "almost", None),
]
