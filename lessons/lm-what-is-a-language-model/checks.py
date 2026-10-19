# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson lm-what-is-a-language-model: examples, the training examples a text gives for next-token
prediction; and the course's own small language model, next_probs, made by counting in Tiny Shakespeare which
character follows each group of three characters."""
import functools
import re
from collections import Counter

from everything_ai import paths
from everything_ai.check import CodeBox, FunctionBox, Mistake, same_value

DATA = "data/tiny-shakespeare.txt"   # about 1.1 MB of plays by Shakespeare: see data/README.md
LOOK = 3                             # the model looks at the last 3 characters of the text so far


@functools.lru_cache(maxsize=1)
def _text() -> str:
    return (paths.ROOT / DATA).read_text(encoding="utf-8")


@functools.lru_cache(maxsize=1)
def _vocabulary() -> tuple[str, ...]:
    return tuple(sorted(set(_text())))


@functools.lru_cache(maxsize=1)
def _all() -> Counter:
    """How often each character occurs in the whole text: what the model gives when it looks at no characters."""
    return Counter(_text())


@functools.lru_cache(maxsize=4096)
def _counts(context: str) -> Counter:
    """How often each character follows `context` in the text."""
    text, out, start = _text(), Counter(), 0
    while True:
        i = text.find(context, start)
        if i < 0 or i + len(context) >= len(text):
            return out
        out[text[i + len(context)]] += 1
        start = i + 1


def next_probs(text: str) -> dict[str, float]:
    """The course's language model: for every character of the vocabulary, the probability that it comes next after
    `text`. It counts what follows the last 3 characters in Tiny Shakespeare; when those never
    occur, it looks at fewer."""
    for k in range(min(LOOK, len(text)), -1, -1):
        c = _counts(text[len(text) - k:]) if k else _all()
        n = sum(c.values())
        if n:
            return {ch: c[ch] / n for ch in _vocabulary()}
    return {}


def top(text: str, n: int = 5) -> list[tuple[str, float]]:
    """The n most likely next characters after text, for the picture; a space is shown as ␣."""
    probs = next_probs(text)
    best = sorted(probs, key=lambda ch: -probs[ch])[:n]
    return [("␣" if ch == " " else ch, probs[ch]) for ch in best]


# The lesson is done when these are correct. "add-up" is the choice question in the lesson.
REQUIRED = ["add-up", "examples", "real-probs"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _lists(v):
    """Tuples as lists, so that an example as a tuple or as a list counts the same."""
    if isinstance(v, (list, tuple)):
        return [_lists(x) for x in v]
    return v


def _same_pairs(a, b) -> bool:
    return same_value(_lists(a), _lists(b))


def _shown(v) -> str:
    """A list of examples for the messages, on one line, with tuples as Python shows them."""
    return repr(v)


# --- exercise 2: examples -------------------------------------------------------------------------------------------------

_E_HEAD = "def examples(text):\n    pairs = []\n"
_E_START = _E_HEAD + "    for i in range(...):\n        pairs.append((..., ...))\n    return pairs\n"


def _e(loop: str = "    for i in range(1, len(text)):\n", pair: str = "        pairs.append((text[:i], text[i]))\n") -> str:
    return _E_HEAD + loop + pair + "    return pairs\n"


_E = _e()

# --- exercise 3: real-probs -----------------------------------------------------------------------------------------------

_P_HEAD = "text = \"to be\"\nreal = []\nfor i in range(1, len(text)):\n"
_P_START = _P_HEAD + "    probs = ...\n    real.append(...)\nreal\n"


def _p(body: str = "    probs = next_probs(text[:i])\n    real.append(probs[text[i]])\n") -> str:
    return _P_HEAD + body + "real\n"


_P = _p()

EXERCISES = {
    "examples": FunctionBox(
        name="examples",
        function="examples",
        start=_E_START,
        solution=_E,
        visible=[("my lord",)],
        hidden=[("ab",), ("a",), ("to be",), ("aaa",)],
        compare=_same_pairs,
        show=_shown,
        first=[_DOTS],
        mistakes=[
            Mistake("from-zero", wrong=_e(loop="    for i in range(len(text)):\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your first example has an empty text so far, `''`. Start at 1: the first example has the first token as its input and the second as its label.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je eerste voorbeeld heeft een lege tekst tot nu toe, `''`. Begin bij 1: het eerste voorbeeld heeft het eerste token als input en het tweede als label."),
            Mistake("leak", wrong=_e(pair="        pairs.append((text[:i + 1], text[i]))\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: every input already ends with its own label. `text[:i]` stops just before place `i`, so the input does not contain the token that comes next.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elke input eindigt al op zijn eigen label. `text[:i]` stopt net voor plek `i`, zodat de input het token dat volgt niet bevat."),
            Mistake("one-short", wrong=_e(loop="    for i in range(1, len(text) - 1):\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the last example is missing. `range` stops just before its end, and the last token, at place `len(text) - 1`, is a label too.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: het laatste voorbeeld ontbreekt. `range` stopt net voor zijn eind, en het laatste token, op plek `len(text) - 1`, is ook een label."),
            Mistake("last-only", wrong=_e(pair="        pairs.append((text[i - 1], text[i]))\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: each input is only the token just before its label. In next-token prediction the input is the whole text so far, `text[:i]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elke input is alleen het token vlak voor zijn label. Bij next-token prediction is de input de hele tekst tot nu toe, `text[:i]`."),
            Mistake("swapped", wrong=_e(pair="        pairs.append((text[i], text[:i]))\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: input and label are the other way round. First the text so far, then the token that comes next.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: input en label zijn omgedraaid. Eerst de tekst tot nu toe, dan het token dat volgt."),
            Mistake("past-end", error=lambda e: isinstance(e, IndexError),
                    en="With `{call}`, your code asks for a place past the last token: the places of a text run from 0 to `len(text) - 1`. Which place does it ask for in the last round? `range` stops just before its end.",
                    nl="Met `{call}` vraagt je code om een plek voorbij het laatste token: de plekken van een tekst lopen van 0 tot en met `len(text) - 1`. Om welke plek vraagt hij in de laatste ronde? `range` stopt net voor zijn eind."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "empty": {"en": "Replace the three `...`: which places `i` the loop goes over, and for each one the text so far and the token that comes next.",
                      "nl": "Vervang de drie `...`: over welke plekken `i` de loop gaat, en voor elke plek de tekst tot nu toe en het token dat volgt."},
            "arguments": {"en": "The check calls `{call}`, with one text, and the `def` line of your function has another number of names between its parentheses. Keep `(text)`.",
                          "nl": "De check roept `{call}` aan, met één tekst, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(text)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your loop with step 2 of the recipe.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je loop met stap 2 van het recept."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for a text of any length?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor een tekst van elke lengte?"},
            "correct": {"en": "Your function makes the right examples from every text, also from a text of one token, which gives none. Every token after the first is the label of one example.",
                        "nl": "Je functie maakt de goede voorbeelden van elke tekst, ook van een tekst van één token, die er geen geeft. Elk token na het eerste is het label van één voorbeeld."},
        },
    ),
    "real-probs": CodeBox(
        name="real-probs",
        start=_P_START,
        solution=_P,
        names={"next_probs": next_probs},
        keep=["text = \"to be\""],
        others=[{"text": "my lord"}, {"text": "the king"}],
        first=[_DOTS],
        mistakes=[
            Mistake("leak", wrong=_p("    probs = next_probs(text[:i + 1])\n    real.append(probs[text[i]])\n"),
                    en="Your last line gives `{got}`: the model got a text that already ends with the token it should predict, so it is asked about the token after that one. Give it `text[:i]`, the text before place `i`.",
                    nl="Je laatste regel geeft `{got}`: het model kreeg een tekst die al eindigt op het token dat het moet voorspellen, dus het wordt gevraagd naar het token daarna. Geef het `text[:i]`, de tekst voor plek `i`."),
            Mistake("last-seen", wrong=_p("    probs = next_probs(text[:i])\n    real.append(probs[text[i - 1]])\n"),
                    en="Your last line gives `{got}`: the probability of the last token the model has already read, not of the one that really comes next. That one is `text[i]`.",
                    nl="Je laatste regel geeft `{got}`: de kans op het laatste token dat het model al gelezen heeft, niet op het token dat echt volgt. Dat is `text[i]`."),
            Mistake("c-prefix", error=lambda e: isinstance(e, NameError) and getattr(e, "name", None) == "C",
                    en="In the box, the model is called just `next_probs`, without `C.` in front of it: write `next_probs(...)`.",
                    nl="In het vak heet het model gewoon `next_probs`, zonder `C.` ervoor: schrijf `next_probs(...)`."),
            Mistake("whole-dict", matches=lambda got, text: isinstance(got, list) and len(got) > 0 and all(isinstance(x, dict) for x in got),
                    en="Your list holds a whole dictionary for every place, with a probability for every character. Look up only the probability of the token that really comes next, as `_probs[\"r\"]` in the worked example.",
                    nl="Je lijst bevat voor elke plek een hele dictionary, met een kans voor elk teken. Zoek alleen de kans op van het token dat echt volgt, zoals `_probs[\"r\"]` in het uitgewerkte voorbeeld."),
            Mistake("whole-text", wrong=_p("    probs = next_probs(text)\n    real.append(probs[text[i]])\n"),
                    en="Your last line gives `{got}`: the model got the whole text every time, so it predicts what comes after the end. Give it only the text so far, `text[:i]`.",
                    nl="Je laatste regel geeft `{got}`: het model kreeg elke keer de hele tekst, dus het voorspelt wat er na het eind komt. Geef het alleen de tekst tot nu toe, `text[:i]`."),
        ],
        digits=3,
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "empty": {"en": "Replace the two `...`: what the model gives for the text so far, and from that the probability of the token that really comes next.",
                      "nl": "Vervang de twee `...`: wat het model geeft voor de tekst tot nu toe, en daaruit de kans op het token dat echt volgt."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug."},
            "no_value": {"en": "The last line of the box shows nothing. Put `real` back on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `real` terug op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`, and it should give, for every place, the probability the model gives the token that really comes next. Compare your two lines with steps 3 and 4 of the recipe.",
                      "nl": "Je laatste regel geeft `{got}`, en hij moet voor elke plek de kans geven die het model geeft aan het token dat echt volgt. Vergelijk je twee regels met stap 3 en 4 van het recept."},
            "other": {"en": "Right for \"to be\". But with {where}, your last line gives `{got}`, and it should give `{want}`. Does every line work with `text`, and not with something that only fits \"to be\"?",
                      "nl": "Goed voor \"to be\". Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Werkt elke regel met `text`, en niet met iets wat alleen bij \"to be\" past?"},
            "other_error": {"en": "Right for \"to be\". But with {where}, your code stops with an error. Does every line work with `text`, and not with something that only fits \"to be\"?",
                            "nl": "Goed voor \"to be\". Maar met {where} stopt je code met een fout. Werkt elke regel met `text`, en niet met iets wat alleen bij \"to be\" past?"},
            "correct": {"en": "Your list is {got}. The model was surest of the space after \"to\", and least sure of the \"b\" after \"to \": many words can start there. The same code works for any text.",
                        "nl": "Je lijst is {got}. Het model was het zekerst van de spatie na \"to\", en het minst zeker van de \"b\" na \"to \": daar kunnen veel woorden beginnen. Dezelfde code werkt voor elke tekst."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("examples", _E_START, "empty", None),
    ("examples", _e(pair="        pairs.append([text[:i], text[i]])\n"), "correct", None),
    ("examples", _e(loop="    for i in range(len(text)):\n"), "wrong", "from-zero"),
    ("examples", _e(pair="        pairs.append((text[:i + 1], text[i]))\n"), "wrong", "leak"),
    ("examples", _e(loop="    for i in range(1, len(text) - 1):\n"), "wrong", "one-short"),
    ("examples", _e(pair="        pairs.append((text[i - 1], text[i]))\n"), "wrong", "last-only"),
    ("examples", _e(loop="    for i in range(len(text) - 1):\n", pair="        pairs.append((text[i], text[i + 1]))\n"), "wrong", "last-only"),
    ("examples", _e(pair="        pairs.append((text[i], text[:i]))\n"), "wrong", "swapped"),
    ("examples", _e(loop="    for i in range(1, len(text) + 1):\n"), "error", "past-end"),
    ("examples", _e(pair="        pairs.append((text[:i], text[i + 1]))\n"), "error", "past-end"),
    ("examples", _e(pair="        pairs.append((text, text[i]))\n"), "wrong", None),
    ("real-probs", _P_START, "empty", None),
    ("real-probs", _p("    real.append(next_probs(text[:i])[text[i]])\n"), "correct", None),
    ("real-probs", _p("    probs = next_probs(text[:i + 1])\n    real.append(probs[text[i]])\n"), "wrong", "leak"),
    ("real-probs", _p("    probs = next_probs(text[:i])\n    real.append(probs[text[i - 1]])\n"), "wrong", "last-seen"),
    ("real-probs", _p("    probs = next_probs(text)\n    real.append(probs[text[i]])\n"), "wrong", "whole-text"),
    ("real-probs", _p("    probs = next_probs(text[:i])\n    real.append(probs)\n"), "wrong", "whole-dict"),
    ("real-probs", _p("    probs = C.next_probs(text[:i])\n    real.append(probs[text[i]])\n"), "error", "c-prefix"),
    ("real-probs", _p("    probs = next_probs(text[i - 1])\n    real.append(probs[text[i]])\n"), "wrong", None),
    ("real-probs", _p("    probs = next_probs(\"to be\"[:i])\n    real.append(probs[\"to be\"[i]])\n"), "almost", "other-value"),
    ("real-probs", _P.replace("range(1, len(text))", "range(1, 5)"), "almost", "other-value"),
]
