# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the project lesson lm-bigram-counts: count_bigrams and bigram_probs, each saved in the learner's project
(work/projects/llm/) when right, and a demo of the learner's own bigram model on Tiny Shakespeare."""
import math
import re
from functools import cache

from everything_ai import paths, projects
from everything_ai.check import FunctionBox, Mistake, Timeout, run_code, with_time_limit

DATA = "data/tiny-shakespeare.txt"   # about 1.1 MB of plays by Shakespeare: see data/README.md

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
# Adding 1 to a key that is not there yet stops with a KeyError, the only one `counts[pair]` can give. Known by that
# error, not by the text: a typo such as `if pair in count:` stops with another error, and gets the plain message.
_KEY = Mistake("add-before-key", error=lambda e: isinstance(e, KeyError),
               en="The first time a pair comes by, it is not in `counts` yet, so `counts[pair] + 1` stops. Look first with `if pair in counts:`; if it is not there, start it at 1.",
               nl="De eerste keer dat een paar langskomt, staat het nog niet in `counts`, dus stopt `counts[pair] + 1`. Kijk eerst met `if pair in counts:`; staat het er niet in, begin het dan bij 1.")


@cache
def text() -> str:
    return (paths.ROOT / DATA).read_text(encoding="utf-8")


# Hint 1 points to count_pairs: copied whole, its `if` with LETTERS keeps only letters, and without the line
# `LETTERS = ...` it stops because Python does not know LETTERS. Both get this message.
_LETTERS = Mistake("letters-check", matches=lambda got, text: re.search(r"\bin\s+LETTERS\b", text) is not None,
                   en="Your function still checks for letters with `LETTERS`, as `count_pairs` did. A bigram model counts every pair, also with a space, a capital or a sign in it: leave out the `if` with `LETTERS`, and leave the text as it is.",
                   nl="Je functie controleert nog op letters met `LETTERS`, zoals `count_pairs` deed. Een bigram-model telt elk paar, ook met een spatie, een hoofdletter of een leesteken erin: laat de `if` met `LETTERS` weg, en laat de tekst zoals hij is.")


# --- exercise 2: count_bigrams ------------------------------------------------------------------------------------------

_C_HEAD = "def count_bigrams(text):\n    counts = {}\n    for i in range(len(text) - 1):\n        pair = text[i:i + 2]\n"
_C_START = _C_HEAD + "        ...\n    return counts\n"
_C_COUNT = "        if pair in counts:\n            counts[pair] = counts[pair] + 1\n        else:\n            counts[pair] = 1\n"


def _c(count: str = _C_COUNT, head: str = _C_HEAD) -> str:
    return head + count + "    return counts\n"


_C = _c()

# --- exercise 3: bigram_probs -------------------------------------------------------------------------------------------

_P_HEAD = "def bigram_probs(counts, a):\n    total = 0\n    for pair in counts:\n        if pair[0] == a:\n"
_P_MID = "    probs = {}\n    for pair in counts:\n        if pair[0] == a:\n"
_P_START = _P_HEAD + "            total = ...\n" + _P_MID + "            probs[pair[1]] = ...\n    return probs\n"


def _p(add: str = "            total = total + counts[pair]\n", value: str = "            probs[pair[1]] = counts[pair] / total\n",
       head: str = _P_HEAD) -> str:
    return head + add + _P_MID + value + "    return probs\n"


_P = _p()


def _close_probs(got, want) -> bool:
    """Two dictionaries of probabilities: the same keys, and each value the same up to rounding."""
    return (isinstance(got, dict) and isinstance(want, dict) and set(got) == set(want)
            and all(isinstance(got[k], (int, float)) and math.isclose(got[k], want[k], rel_tol=1e-9) for k in want))


def _shown(v) -> str:
    """A dictionary for the messages, with probabilities rounded to 4 decimals."""
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k!r}: {round(x, 4) if isinstance(x, float) else x!r}" for k, x in v.items()) + "}"
    return repr(v)


# The lesson is done when these are correct. "markov" is the choice question in the lesson.
REQUIRED = ["markov", "count-bigrams", "bigram-probs"]

_SAVED_TEXTS = {
    "correct": {"en": "Your function is right in all {n} cases. The check saved it in your project, as `{saved}`: the demo and later lessons import it from there.",
                "nl": "Je functie klopt in alle {n} gevallen. De check heeft hem in je project bewaard, als `{saved}`: de demo en latere lessen halen hem daar binnen."},
    "kept": {"en": "Your function is right in all {n} cases. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
             "nl": "Je functie klopt in alle {n} gevallen. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren? Verwijder het bestand en laat opnieuw nakijken."},
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

EXERCISES = {
    "count-bigrams": FunctionBox(
        name="count-bigrams",
        function="count_bigrams",
        start=_C_START,
        solution=_C,
        visible=[("To be, or not",)],
        hidden=[("",), ("x",), ("aaa",), ("To be, or\nnot!",), ("abab",)],
        python_values=True,
        save_as=("llm", "bigrams"),
        first=[_DOTS, _LETTERS],
        mistakes=[
            _KEY,
            Mistake("letters-only", wrong=_c(count="        if pair.isalpha():\n" + "\n".join("    " + line for line in _C_COUNT.splitlines()) + "\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: pairs with a space or a sign in them are missing. A language model has to write those too: count every pair, whatever its characters.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: paren met een spatie of een leesteken erin ontbreken. Een language model moet die ook schrijven: tel elk paar, wat de tekens ook zijn."),
            Mistake("lowered", wrong=_c(head=_C_HEAD.replace("    counts = {}\n", "    text = text.lower()\n    counts = {}\n")),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your function makes capitals small. A capital is a token of its own: a model that never saw one could not start a sentence with it. Leave the text as it is.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je functie maakt hoofdletters klein. Een hoofdletter is een eigen token: een model dat er nooit een zag, kan er geen zin mee beginnen. Laat de tekst zoals hij is."),
            Mistake("always-one", wrong=_c(count="        counts[pair] = 1\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. Every pair has the count 1: when a pair is already in `counts`, add 1 to what it has.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Elk paar heeft de telling 1: staat een paar al in `counts`, tel dan 1 op bij wat het heeft."),
            Mistake("start-at-zero", wrong=_c(count=_C_COUNT.replace("counts[pair] = 1", "counts[pair] = 0")),
                    en="`{call}` gives `{got}`, and it should give `{want}`: every count is 1 too low. When a new pair comes by, it has been seen once already: start it at 1.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elke telling is 1 te laag. Komt er een nieuw paar langs, dan is het al één keer gezien: begin het bij 1."),
        ],
        texts={
            **_SAVED_TEXTS,
            "empty": {"en": "Replace the `...` with the lines that count `pair`: one more when it is in `counts` already, else a new key that starts at 1.",
                      "nl": "Vervang de `...` door de regels die `pair` tellen: één erbij als het al in `counts` staat, anders een nieuwe key die bij 1 begint."},
            "arguments": {"en": "The check calls `{call}`, with one text, and the `def` line of your function has another number of names between its parentheses. Keep `(text)`.",
                          "nl": "De check roept `{call}` aan, met één tekst, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(text)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does your loop count every pair once, whatever its characters?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Telt je loop elk paar één keer, wat de tekens ook zijn?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it also count a pair that comes back, and pairs with a newline in them?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Telt hij ook een paar dat terugkomt, en paren met een regeleinde erin?"},
        },
    ),
    "bigram-probs": FunctionBox(
        name="bigram-probs",
        function="bigram_probs",
        start=_P_START,
        solution=_P,
        visible=[({"ab": 3, "ac": 1, "ba": 2}, "a")],
        hidden=[({"ab": 3, "ac": 1, "ba": 2}, "b"), ({"ab": 3, "ac": 1, "ba": 2}, "c"), ({"aa": 1, "ab": 2}, "a"),
                ({"th": 2, "te": 1, "t ": 1, "ht": 5}, "t"), ({"q ": 1, "qu": 4, " q": 7}, "q")],
        compare=_close_probs,
        show=_shown,
        save_as=("llm", "bigram_model"),
        first=[_DOTS],
        mistakes=[
            Mistake("all-pairs", wrong=_p(head=_P_HEAD.replace("        if pair[0] == a:\n", "        if True:\n")),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the probabilities add up to less than 1. Your `total` counts every pair in `counts`; count only the pairs that start with `a`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de kansen tellen op tot minder dan 1. Je `total` telt elk paar in `counts`; tel alleen de paren die met `a` beginnen."),
            Mistake("counts-not-probs", wrong=_p(value="            probs[pair[1]] = counts[pair]\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: those are the counts, not probabilities. Divide each count by `total`, so that together they add up to 1.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat zijn de tellingen, geen kansen. Deel elke telling door `total`, zodat ze samen op 1 uitkomen."),
            Mistake("whole-pair", wrong=_p(value="            probs[pair] = counts[pair] / total\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your keys are whole pairs. The model says which character comes next, so the key is the second character of the pair, `pair[1]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je keys zijn hele paren. Het model zegt welk teken er volgt, dus de key is het tweede teken van het paar, `pair[1]`."),
            Mistake("total-last", wrong=_p(add="            total = counts[pair]\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. Your `total` is only the count of the last pair that starts with `a`: each time, it forgets what it had. Add each count to what `total` already has.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Je `total` is alleen de telling van het laatste paar dat met `a` begint: elke keer vergeet hij wat hij al had. Tel elke telling op bij wat `total` al heeft."),
            Mistake("total-keys", wrong=_p(add="            total = total + 1\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the probabilities add up to more than 1. Your `total` counts the different pairs that start with `a`, 1 for each; add up their counts, how often each pair occurs.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de kansen tellen op tot meer dan 1. Je `total` telt de verschillende paren die met `a` beginnen, 1 per paar; tel hun tellingen op, hoe vaak elk paar voorkomt."),
            Mistake("second-char", wrong=_P.replace("pair[0] == a", "pair[1] == a"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. Your `if` looks for `a` as the second character of a pair: those pairs say what comes before `a`. The pairs that say what comes after `a` start with it: `pair[0] == a`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Je `if` zoekt `a` als tweede teken van een paar: die paren zeggen wat er voor `a` staat. De paren die zeggen wat er na `a` komt, beginnen ermee: `pair[0] == a`."),
        ],
        texts={
            **_SAVED_TEXTS,
            "empty": {"en": "Replace the two `...`: what `total` gets from each pair that starts with `a`, and then the probability of each next character.",
                      "nl": "Vervang de twee `...`: wat `total` krijgt van elk paar dat met `a` begint, en daarna de kans op elk volgend teken."},
            "arguments": {"en": "The check calls `{call}`, with the counts and a character, and the `def` line of your function has another number of names between its parentheses. Keep `(counts, a)`.",
                          "nl": "De check roept `{call}` aan, met de tellingen en een teken, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(counts, a)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your two lines with steps 3 and 4 of the recipe.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je twee regels met stap 3 en 4 van het recept."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it also work for a character that no pair starts with?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij ook voor een teken waar geen paar mee begint?"},
        },
    ),
}


# --- the demo: the learner's own bigram model on Tiny Shakespeare -----------------------------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "llm" / f"{module}.py").exists()


def demo():
    """For the demo: None when bigrams.py or bigram_model.py is not in the project yet; "counts" or "probs" when that
    saved module no longer gives what the course's gives on a small case, or stops with an error; "slow" when counting
    the book takes more than 30 seconds; else (number of different pairs, probabilities after "t", after "q")."""
    if not (_saved("bigrams") and _saved("bigram_model")):
        return None
    # a text with a newline, and with pairs that come back, so a count that never grows shows up too
    run = run_code("from llm.bigrams import count_bigrams\ncount_bigrams('To be, or not\\nto be!')")
    if run.error is not None or run.value != _course_counts("To be, or not\nto be!"):
        return "counts"
    count = run.names["count_bigrams"]
    run = run_code("from llm.bigram_model import bigram_probs\nbigram_probs({'th': 2, 'te': 1, 't ': 1, 'ht': 5}, 't')")
    if run.error is not None or not _close_probs(run.value, {"h": 0.5, "e": 0.25, " ": 0.25}):
        return "probs"
    probs = run.names["bigram_probs"]
    try:
        counts = with_time_limit(lambda: count(text()), 30)
    except Timeout:
        return "slow"
    except Exception:
        return "counts"
    try:
        return len(counts), probs(counts, "t"), probs(counts, "q")
    except Exception:
        return "probs"


def _course_counts(t: str) -> dict:
    counts = {}
    for i in range(len(t) - 1):
        counts[t[i:i + 2]] = counts.get(t[i:i + 2], 0) + 1
    return counts


@cache
def book_counts() -> dict:
    """The course's own bigram counts of Tiny Shakespeare."""
    return _course_counts(text())


def grouped(n: int, lang: str) -> str:
    """A whole number for running text, with a thousands separator: 1,403 in English, 1.403 in Dutch."""
    return f"{n:,}".replace(",", "." if lang == "nl" else ",")


def distinct() -> int:
    """How many different pairs of characters occur in the plays."""
    return len(book_counts())


def after_q() -> int:
    """How many pairs start with a q: every one of them is "qu"."""
    return sum(v for k, v in book_counts().items() if k[0] == "q")


def course_probs(a: str) -> dict:
    counts = book_counts()
    total = sum(v for k, v in counts.items() if k[0] == a)
    return {k[1]: v / total for k, v in counts.items() if k[0] == a}


def top(probs: dict, n: int = 5) -> list[tuple[str, float]]:
    """The n most likely next characters, for a picture; a space is shown as ␣ and a line break as ↵."""
    best = sorted(probs, key=lambda ch: -probs[ch])[:n]
    return [({" ": "␣", "\n": "↵"}.get(ch, ch), probs[ch]) for ch in best]


# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("count-bigrams", _C_START, "empty", None),
    ("count-bigrams", _c(count="        counts[pair] = counts.get(pair, 0) + 1\n"), "correct", None),
    ("count-bigrams", _c(count="        counts[pair] = counts[pair] + 1\n"), "error", "add-before-key"),
    ("count-bigrams", _c(count="        if pair.isalpha():\n" + "\n".join("    " + line for line in _C_COUNT.splitlines()) + "\n"), "wrong", "letters-only"),
    ("count-bigrams", _c(head=_C_HEAD.replace("    counts = {}\n", "    text = text.lower()\n    counts = {}\n")), "wrong", "lowered"),
    ("count-bigrams", _c(count="        counts[pair] = 1\n"), "wrong", "always-one"),
    ("count-bigrams", _c(count=_C_COUNT.replace("counts[pair] = 1", "counts[pair] = 0")), "wrong", "start-at-zero"),
    ("count-bigrams", _c(count="        if pair[0] in LETTERS and pair[1] in LETTERS:\n" + "\n".join("    " + line for line in _C_COUNT.splitlines()) + "\n",
                         head=_C_HEAD.replace("    counts = {}\n", "    text = text.lower()\n    counts = {}\n")), "wrong", "letters-check"),
    ("count-bigrams", 'LETTERS = "abcdefghijklmnopqrstuvwxyz"\n\n\n' + _c(count="        if pair[0] in LETTERS and pair[1] in LETTERS:\n" + "\n".join("    " + line for line in _C_COUNT.splitlines()) + "\n"), "wrong", "letters-check"),
    ("count-bigrams", _c(count=_C_COUNT.replace("if pair in counts", "if pair in count")), "error", None),
    ("bigram-probs", _P_START, "empty", None),
    ("bigram-probs", _p(value="            probs[pair[1]] = counts[pair] * (1 / total)\n"), "correct", None),
    ("bigram-probs", _p(head=_P_HEAD.replace("        if pair[0] == a:\n", "        if True:\n")), "wrong", "all-pairs"),
    ("bigram-probs", _p(value="            probs[pair[1]] = counts[pair]\n"), "wrong", "counts-not-probs"),
    ("bigram-probs", _p(value="            probs[pair] = counts[pair] / total\n"), "wrong", "whole-pair"),
    ("bigram-probs", _p(add="            total = counts[pair]\n"), "wrong", "total-last"),
    ("bigram-probs", _p(add="            total = total + 1\n"), "wrong", "total-keys"),
    ("bigram-probs", _P.replace("pair[0] == a", "pair[1] == a"), "wrong", "second-char"),
]
