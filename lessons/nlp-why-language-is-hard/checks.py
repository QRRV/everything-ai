# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-why-language-is-hard: counting the readings a sentence gets from words with several
meanings, mapping spelling variants to one form, and how fast the ways to read a sentence grow with the number of
phrases such as "in the box" (Church and Patil 1982). The meanings, sentences, variants and messages are made up by
the course."""
import contextlib
import copy
import html
import io
import math
import re

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

# --- the course's words, sentences and messages ----------------------------------------------------------------------

# Made-up and short: what a word can be, for words with more than one meaning or role. A word that is not in the dict
# has one reading.
SENSES = {
    "strikes": ["verb: hits", "noun: times that workers stop working"],
    "idle": ["verb: leaves with nothing to do", "adjective: doing nothing"],
    "kids": ["noun: children", "verb: teases"],
    "saw": ["verb: past of see", "verb: cuts with a saw", "noun: a tool for cutting"],
    "bat": ["noun: a flying animal", "noun: a stick for hitting a ball"],
    "bank": ["noun: a place that keeps money", "noun: the land beside a river"],
    "light": ["noun: what lets you see", "adjective: not heavy", "verb: sets on fire"],
    "fine": ["adjective: good", "noun: money paid as a punishment"],
}
SENSES_NL = {
    "strikes": ["werkwoord: slaat", "zelfstandig naamwoord: stakingen"],
    "idle": ["werkwoord: laat zitten zonder iets te doen", "bijvoeglijk naamwoord: werkloos, niets doend"],
    "kids": ["zelfstandig naamwoord: kinderen", "werkwoord: plaagt"],
    "saw": ["werkwoord: zag", "werkwoord: zaagt", "zelfstandig naamwoord: een zaag"],
    "bat": ["zelfstandig naamwoord: een vleermuis", "zelfstandig naamwoord: een knuppel om mee te slaan"],
    "bank": ["zelfstandig naamwoord: een bank voor geld", "zelfstandig naamwoord: de oever van een rivier"],
    "light": ["zelfstandig naamwoord: licht", "bijvoeglijk naamwoord: niet zwaar", "werkwoord: steekt aan"],
    "fine": ["bijvoeglijk naamwoord: goed", "zelfstandig naamwoord: een boete"],
}
SENTENCES = [
    ["teacher", "strikes", "idle", "kids"],
    ["we", "saw", "a", "bat"],
    ["the", "bank", "by", "the", "river"],
    ["a", "fine", "light", "bat"],
    ["the", "end"],
]
# the two readings of the headline that make sense, by the roles of strikes, idle and kids
SENSIBLE = [("noun", "verb", "noun"), ("verb", "adjective", "noun")]

# one meaning, many forms: made-up spelling variants and their standard form
VARIANTS = {"thx": "thanks", "colour": "color", "u": "you", "2moro": "tomorrow", "favourite": "favorite",
            "pls": "please"}
MESSAGES = [
    ["thx", "for", "the", "colour", "tips"],
    ["thanks", "for", "the", "color", "tips"],
    ["see", "u", "2moro"],
    ["my", "favourite", "colour", "is", "blue"],
    ["pls", "call", "me"],
]


def readings(words: list, senses: dict) -> int:
    """The course's own Exercise 1: the number of ways to read the words, one meaning for each word."""
    count = 1
    for word in words:
        if word in senses:
            count = count * len(senses[word])
    return count


def canonical(words: list, variants: dict) -> list:
    """The course's own Exercise 2: each word replaced by its standard form, if it has one."""
    out = []
    for word in words:
        if word in variants:
            out.append(variants[word])
        else:
            out.append(word)
    return out


def parses(n: int) -> int:
    """How many ways there are to attach n phrases such as "in the box" after a verb and its object: the Catalan
    number, as Church and Patil (1982) showed. Built up from smaller counts: a tree splits into a left and a right
    part."""
    counts = [1]
    for k in range(1, n + 1):
        counts.append(sum(counts[i] * counts[k - 1 - i] for i in range(k)))
    return counts[n]


# Church and Patil's own examples: 2, 3 and 4 phrases gave 2, 5 and 14 readings, and their parser still matched the
# Catalan numbers with 9 (4862, on sentences such as "It was the number of products of products ..."). The paper prints 469 for 429, the eighth number: a misprint.
CHURCH_PATIL = {2: 2, 3: 5, 4: 14, 9: 4862}
PRINTED = [1, 1, 2, 5, 14, 42, 132, 469, 1430, 4862]


def combinations(words: list, lang: str = "en") -> list:
    """Every way to read the words with SENSES: a list of rows, one meaning for each word that has more than one."""
    table = SENSES_NL if lang == "nl" else SENSES
    rows = [[]]
    for word in words:
        if word in SENSES:
            rows = [row + [(word, m)] for row in rows for m in table[word]]
    return rows


class Named:
    """A value the check gives the learner's function, shown in a message by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


SENSES_N = Named("SENSES", SENSES)
VARIANTS_N = Named("VARIANTS", VARIANTS)
SENTENCES_N = [Named(f"SENTENCES[{i}]", s) for i, s in enumerate(SENTENCES)]
MESSAGES_N = [Named(f"MESSAGES[{i}]", m) for i, m in enumerate(MESSAGES)]


# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink);vertical-align:top")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def _word(text: str) -> str:
    """A short value as Python writes it, kept on one line in a table cell."""
    return f'<span style="font-family:var(--ea-font-code);font-size:12px;white-space:nowrap">{html.escape(text)}</span>'


def _table(rows: list, head: list) -> str:
    """A table with a head row; each row is a list of (html, style)."""
    top = "".join(f'<th style="{_HEAD}">{html.escape(h)}</th>' for h in head)
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{";" + style if style else ""}">{cell}</td>'
                                    for cell, style in row) + "</tr>" for row in rows)
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
            f'<tr>{top}</tr>{body}</table></div>')


_HEADLINE_WORDS = {
    "en": {"title": "Eight ways to read four words",
           "head": ["strikes", "idle", "kids", "makes sense?"], "yes": "yes", "no": "no",
           "note": "Three words with two roles each: 2 × 2 × 2 = 8 readings. Two of them make sense: the strikes of "
                   "teachers leave children without lessons, or a teacher hits children who do nothing. A reader "
                   "picks one without noticing; a program has to choose."},
    "nl": {"title": "Acht manieren om vier woorden te lezen",
           "head": ["strikes", "idle", "kids", "zinnig?"], "yes": "ja", "no": "nee",
           "note": "Drie woorden met elk twee rollen: 2 × 2 × 2 = 8 lezingen. Twee ervan zijn zinnig: stakingen van "
                   "leraren laten kinderen zonder les zitten, of een leraar slaat kinderen die niets doen. Een lezer "
                   "kiest er een zonder het te merken; een programma moet kiezen."},
}


def _role(meaning: str) -> str:
    return meaning.split(":", 1)[0]


def headline_picture(lang: str) -> str:
    """The eight readings of the headline, with the two that make sense marked."""
    w = _HEADLINE_WORDS[lang]
    rows, said = [], []
    for row, row_en in zip(combinations(SENTENCES[0], lang), combinations(SENTENCES[0], "en")):
        roles = tuple(_role(m) for _, m in row_en)
        ok = roles in SENSIBLE
        rows.append([(html.escape(m), "") for _, m in row] + [(w["yes"] if ok else w["no"],
                                                                "font-weight:600" if ok else "color:var(--ea-ink-3)")])
        said.append(", ".join(m for _, m in row) + f" ({w['yes'] if ok else w['no']})")
    return yours.figure(w["title"], _table(rows, w["head"]) + _note(w["note"]),
                        alt=f'{w["title"]}: ' + "; ".join(said) + ". " + w["note"])


_PARSES_WORDS = {
    "en": {"title": "Ways to read a sentence with {n} {phrases} like ‘in the box’", "one": "phrase", "more": "phrases",
           "head": ["phrases", "ways to read", ""], "now": "Put the block {phrases}",
           "pp": ["in the box", "on the table", "in the kitchen", "by the window", "near the door", "on the left",
                  "in the morning", "with a smile", "for the child"],
           "church": "Church and Patil found {k} for {n} phrases.",
           "church9": "Church and Patil's parser found 4862 for nine phrases, on sentences such as ‘It was the number "
                      "of products of products …’.",
           "small": {0: "Without a phrase the sentence is not complete: ‘put’ needs a place. As a row of phrases, it "
                        "has one reading.",
                     1: "One phrase has one place to go: it says where to put the block."},
           "note": "Each extra phrase can attach to more places, so the count grows faster and faster: these are the "
                   "Catalan numbers. The bars grow with the number of digits: drawn to scale, the bar for 9 phrases "
                   "would be 4862 times as long as the one for 1. The course computed the numbers; Church and Patil's "
                   "parser found the same counts."},
    "nl": {"title": "Manieren om een zin met {n} {phrases} te lezen", "one": "voorzetselgroep", "more": "voorzetselgroepen",
           "head": ["voorzetselgroepen", "manieren", ""], "now": "Put the block {phrases}",
           "pp": ["in the box", "on the table", "in the kitchen", "by the window", "near the door", "on the left",
                  "in the morning", "with a smile", "for the child"],
           "church": "Church en Patil vonden er {k} bij {n} voorzetselgroepen.",
           "church9": "De parser van Church en Patil vond er 4862 bij negen voorzetselgroepen, in zinnen zoals It was "
                      "the number of products of products enzovoort.",
           "small": {0: "Zonder voorzetselgroep is de zin niet af: put vraagt om een plek. Als rij voorzetselgroepen "
                        "heeft de zin één lezing.",
                     1: "Eén voorzetselgroep kan maar op één plek: die zegt waar het blok heen moet."},
           "note": "Elke voorzetselgroep erbij kan op meer plekken aanhaken, dus het aantal groeit steeds sneller: dit zijn de "
                   "Catalan-getallen. De balken groeien met het aantal cijfers: op schaal zou de balk bij 9 "
                   "voorzetselgroepen 4862 keer zo lang zijn als die bij 1. De course rekende de getallen uit; de "
                   "parser van Church en Patil vond dezelfde aantallen."},
}


def parses_picture(n: int, lang: str) -> str:
    """The number of ways to read a sentence with 0 to 9 phrases, with the chosen number in ink."""
    n = max(0, min(int(n), 9))
    w = _PARSES_WORDS[lang]
    rows = []
    for k in range(10):
        width = 0.4 + 2.2 * math.log10(parses(k))
        bar = (f'<div style="height:0.7rem;width:{width:.2f}rem;background:'
               f'{"var(--ea-ink)" if k == n else "var(--ea-ink-3)"};border-radius:2px"></div>')
        style = "font-weight:600" if k == n else "color:var(--ea-ink-2)"
        rows.append([(str(k), style), (str(parses(k)), style), (bar, "")])
    sentence = w["now"].format(phrases=" ".join(w["pp"][:n])).strip() + "."
    lines = [sentence]
    if n in w["small"]:
        lines.append(w["small"][n])
    if n == 9:
        lines.append(w["church9"])
    elif n in CHURCH_PATIL:
        lines.append(w["church"].format(k=CHURCH_PATIL[n], n=n))
    lines.append(w["note"])
    title = w["title"].format(n=n, phrases=w["one"] if n == 1 else w["more"])
    return yours.figure(title, _table(rows, w["head"]) + "".join(_note(x) for x in lines),
                        alt=f'{title}. ' + "; ".join(f"{k}: {parses(k)}" for k in range(10)) + ". " + " ".join(lines))


# --- the pictures under the checks -----------------------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error


def _call(fn, *args):
    """The learner's function on copies of args within the time limit of a check; _STOPPED when it stops with an
    error, and the Timeout passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(lambda: fn(*(copy.deepcopy(a) for a in args)))
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _shown(v, lang: str) -> str:
    """What the learner's function gave, for a cell of a table, as Python writes it."""
    if v is _STOPPED:
        return {"en": "error", "nl": "fout"}[lang]
    text = repr(v)
    return text if len(text) <= 60 else text[:57] + "..."


_CHECK_WORDS = {
    "en": {"readings": {"title": "Your readings on the course's sentences",
                        "head": ["words", "yours", "the course's"],
                        "fine": "The same as the course's for all five sentences: {counts}.",
                        "differs": "It differs from the course's for {k} of the five sentences: your answers there have a dashed frame."},
           "canonical": {"title": "Your canonical on the course's messages",
                         "head": ["message", "yours", "the course's"],
                         "fine": "The same as the course's for all five messages. The first two are now the same list.",
                         "differs": "It differs from the course's for {k} of the five messages: your answers there have a dashed frame."}},
    "nl": {"readings": {"title": "Jouw readings op de zinnen van de course",
                        "head": ["woorden", "de jouwe", "die van de course"],
                        "fine": "Bij alle vijf zinnen hetzelfde als die van de course: {counts}.",
                        "differs": "Bij {k} van de vijf zinnen anders dan die van de course: jouw antwoorden daar hebben een gestreept kader."},
           "canonical": {"title": "Jouw canonical op de berichten van de course",
                         "head": ["bericht", "de jouwe", "die van de course"],
                         "fine": "Bij alle vijf berichten hetzelfde als die van de course. De eerste twee zijn nu dezelfde lijst.",
                         "differs": "Bij {k} van de vijf berichten anders dan die van de course: jouw antwoorden daar hebben een gestreept kader."}},
}


def _compare(seen, key: str, items: list, fn_course, extra) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    words = _CHECK_WORDS[seen.lang][key]
    rows, said, k = [], [], 0
    for item in items:
        got, want = _call(fn, item, extra), fn_course(item, extra)
        differs = got != want
        k += differs
        rows.append([(_word(" ".join(item)), ""), (_word(_shown(got, seen.lang)), _OFF if differs else ""),
                     (_word(repr(want)), "")])
        said.append(f"{' '.join(item)}: {_shown(got, seen.lang)}, {want!r}")
    counts = ", ".join(str(readings(s, SENSES)) for s in SENTENCES)
    after = words["differs"].format(k=k) if k else words["fine"].format(counts=counts)
    return yours.figure(words["title"], _table(rows, words["head"]) + _note(after),
                        alt=f'{words["title"]}: ' + "; ".join(said) + ". " + after)


def readings_picture(seen) -> str:
    """The course's sentences, with the learner's count beside the course's."""
    return _compare(seen, "readings", SENTENCES, readings, SENSES)


def canonical_picture(seen) -> str:
    """The course's messages, with the learner's standard forms beside the course's."""
    return _compare(seen, "canonical", MESSAGES, canonical, VARIANTS)


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["readings", "canonical", "trophy", "blossoms"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _names_error(e) -> bool:
    return isinstance(e, NameError) and re.search(r"'(SENSES|SENTENCES|VARIANTS|MESSAGES|C)'", str(e)) is not None


_NAMES = Mistake("lesson-names", error=_names_error,
                 en="Your function uses a name of the lesson, such as `SENSES`, `VARIANTS` or `C`, and the check does not give it that name: use the names between the parentheses of the `def` line.",
                 nl="Je functie gebruikt een naam van de les, zoals `SENSES`, `VARIANTS` of `C`, en de check geeft die naam niet mee: gebruik de namen tussen de haakjes van de `def`-regel.")

_KEY = Mistake("not-in-dict", error=lambda e: isinstance(e, KeyError),
               en="Your function stops with a `KeyError`: it looks up a word that is not a key of the dict, such as `\"teacher\"` or `\"for\"`. Look a word up only after `if word in ...:` says it is there.",
               nl="Je functie stopt met een `KeyError`: ze zoekt een woord op dat geen key van de dict is, zoals `\"teacher\"` of `\"for\"`. Zoek een woord pas op als `if word in ...:` zegt dat het erin staat.")

# exercise 1: readings
READINGS_CASES = [(s, SENSES_N) for s in SENTENCES_N] + [([], {"x": ["a", "b"]}), (["x", "y", "x"], {"x": ["a", "b", "c"]})]
_R_START = "def readings(words, senses):\n    ...\n"
_R = ("def readings(words, senses):\n"
      "    count = 1\n"
      "    for word in words:\n"
      "        if word in senses:\n"
      "            count = count * len(senses[word])\n"
      "    return count\n")
_R_PLUS = _R.replace("count * len(", "count + len(")
_R_ZERO = _R.replace("    count = 1\n", "    count = 0\n")
_R_EARLY = _R.replace("    return count\n", "        return count\n")

# exercise 2: canonical
CANONICAL_CASES = [(m, VARIANTS_N) for m in MESSAGES_N] + [([], {"a": "b"}), (["a", "b", "a"], {"a": "c"})]
_C_START = "def canonical(words, variants):\n    ...\n"
_C = ("def canonical(words, variants):\n"
      "    out = []\n"
      "    for word in words:\n"
      "        if word in variants:\n"
      "            out.append(variants[word])\n"
      "        else:\n"
      "            out.append(word)\n"
      "    return out\n")
_C_DROP = ("def canonical(words, variants):\n"
           "    out = []\n"
           "    for word in words:\n"
           "        if word in variants:\n"
           "            out.append(variants[word])\n"
           "    return out\n")
_C_SAME = _C.replace("            out.append(variants[word])\n", "            out.append(word)\n")
_C_EARLY = _C.replace("    return out\n", "        return out\n")


EXERCISES = {
    "readings": FunctionBox(
        name="readings",
        picture=readings_picture,
        function="readings",
        start=_R_START,
        solution=_R,
        visible=[READINGS_CASES[0]],
        hidden=READINGS_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            _KEY,
            Mistake("plus", wrong=_R_PLUS,
                    en="`{call}` gives {got}, and it should give {want}: your function adds the meanings up. Each meaning of one word goes with each meaning of the next, so the counts multiply.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw functie telt de betekenissen op. Elke betekenis van het ene woord gaat samen met elke betekenis van het volgende, dus de aantallen worden vermenigvuldigd."),
            Mistake("zero", wrong=_R_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: a count that starts at 0 stays 0, whatever you multiply it by. Words without a second meaning give one reading, so start at 1.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een telling die bij 0 begint, blijft 0, waarmee je ook vermenigvuldigt. Woorden zonder tweede betekenis geven één lezing, dus begin bij 1."),
            Mistake("early", wrong=_R_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back the count after the first word. Put the `return` after the loop, so every word counts.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw functie geeft de telling na het eerste woord terug. Zet de `return` na de loop, zodat elk woord meetelt."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: start a count, go over the words, and multiply by the number of meanings of each word in `senses`.",
                      "nl": "Vervang de `...`: begin een telling, ga de woorden langs, en vermenigvuldig met het aantal betekenissen van elk woord in `senses`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Each word in `senses` multiplies the count by its number of meanings; other words leave it as it is.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Elk woord in `senses` vermenigvuldigt de telling met het aantal betekenissen ervan; andere woorden laten de telling zoals die is."},
            "almost": {"en": "Right for the headline. But `{call}` gives {got}, and it should give {want}. Does your function use the `words` and `senses` it gets?",
                       "nl": "Goed voor de kop. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `words` en `senses` die ze krijgt?"},
            "correct": {"en": "Your `readings` is right in all {n} cases. Four words already give 8 readings, and *a fine light bat* gives 12. Most of them make no sense, but a program does not know which until it has more to go on than the words.",
                        "nl": "Je `readings` klopt in alle {n} gevallen. Vier woorden geven al 8 lezingen, en *a fine light bat* geeft er 12. De meeste kloppen niet, maar een programma weet pas welke als het meer heeft dan de woorden."},
        },
    ),
    "canonical": FunctionBox(
        name="canonical",
        picture=canonical_picture,
        function="canonical",
        start=_C_START,
        solution=_C,
        visible=[CANONICAL_CASES[0]],
        hidden=CANONICAL_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            _KEY,
            Mistake("drop", wrong=_C_DROP,
                    en="`{call}` gives {got}, and it should give {want}: the words that are already in their standard form are gone. Keep those as they are, with an `else`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de woorden die al in hun standaardvorm staan, zijn weg. Houd die zoals ze zijn, met een `else`."),
            Mistake("same", wrong=_C_SAME,
                    en="`{call}` gives {got}, and it should give {want}: every word stays as it was. A word that is a key of `variants` gets its standard form, the value under that key.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elk woord blijft zoals het was. Een woord dat een key van `variants` is, krijgt zijn standaardvorm, de waarde onder die key."),
            Mistake("early", wrong=_C_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your function stops after the first word. Put the `return` after the loop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw functie stopt na het eerste woord. Zet de `return` na de loop."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: make a new list, and add each word to it, in its standard form if `variants` has one.",
                      "nl": "Vervang de `...`: maak een nieuwe lijst, en voeg elk woord eraan toe, in zijn standaardvorm als `variants` er een heeft."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The function gives a new list of the same length: each word that is a key of `variants` replaced by its value, and the other words as they are.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De functie geeft een nieuwe lijst van dezelfde lengte: elk woord dat een key van `variants` is, vervangen door zijn waarde, en de andere woorden zoals ze zijn."},
            "almost": {"en": "Right for the first message. But `{call}` gives {got}, and it should give {want}. Does your function use the `words` and `variants` it gets?",
                       "nl": "Goed voor het eerste bericht. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `words` en `variants` die ze krijgt?"},
            "correct": {"en": "Your `canonical` is right in all {n} cases. The first two messages were written differently and meant the same; now they are the same list. A dict of variants only helps with forms someone thought of, though.",
                        "nl": "Je `canonical` klopt in alle {n} gevallen. De eerste twee berichten waren anders geschreven en bedoelden hetzelfde; nu zijn ze dezelfde lijst. Een dict met varianten helpt wel alleen bij vormen waar iemand aan dacht."},
        },
    ),
}

CASES = [
    ("readings", _R_START, "empty", None),
    ("readings", _R, "correct", None),
    ("readings", "def readings(words, senses):\n    n = 1\n    for w in words:\n        n = n * len(senses.get(w, [w]))\n    return n\n", "correct", None),
    ("readings", _R_PLUS, "wrong", "plus"),
    ("readings", _R_ZERO, "wrong", "zero"),
    ("readings", _R_EARLY, "wrong", "early"),
    ("readings", "def readings(words, senses):\n    count = 1\n    for word in words:\n        count = count * len(senses[word])\n    return count\n", "error", "not-in-dict"),
    ("readings", "def readings(words, senses):\n    return 8\n", "almost", None),
    ("readings", "def readings(words, senses):\n    return len(SENSES)\n", "error", "lesson-names"),
    ("canonical", _C_START, "empty", None),
    ("canonical", _C, "correct", None),
    ("canonical", "def canonical(words, variants):\n    return [variants.get(w, w) for w in words]\n", "correct", None),
    ("canonical", _C_DROP, "wrong", "drop"),
    ("canonical", _C_SAME, "wrong", "same"),
    ("canonical", _C_EARLY, "wrong", "early"),
    ("canonical", "def canonical(words, variants):\n    return [variants[w] for w in words]\n", "error", "not-in-dict"),
    ("canonical", "def canonical(words, variants):\n    return ['thanks', 'for', 'the', 'color', 'tips']\n", "almost", None),
    ("canonical", "def canonical(words, variants):\n    return VARIANTS\n", "error", "lesson-names"),
]
