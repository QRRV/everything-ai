# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-morphemes: cut a known ending off a word, split a word into two known words, sort words by
the way they were made (inflection, derivation, compounding), and say why derivation is riskier to merge. The lists of
endings and words, and the minimum of three letters for what is left, are the course's own."""
import contextlib
import copy
import html
import io
import re

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, with_time_limit

# --- the course's lists ----------------------------------------------------------------------------------------------

SUFFIXES = ["ing", "ed", "s"]            # English endings of inflection
SUFFIXES_NL = ["te", "en", "t"]          # Dutch endings of inflection: werk-te, boek-en, werk-t
WORDS = ["dog", "house", "door", "bell", "foot", "ball", "sun", "light", "rain", "bow", "book", "store"]
WORDS_NL = ["hond", "huis", "voor", "deur", "fiets", "bel", "zon", "licht", "regen", "boog", "boek", "winkel"]


def cut_suffix(word: str, suffixes: list) -> list:
    """The course's own Exercise 1: the rest and the first ending of the list that the word ends in, when at least 3
    letters are left; else the word as one part."""
    for suffix in suffixes:
        rest = word[:len(word) - len(suffix)]
        if rest + suffix == word and len(rest) >= 3:
            return [rest, suffix]
    return [word]


def split_compound(word: str, words: list) -> list:
    """The course's own Exercise 2: the first cut, from the left, into two words of the list; else the word as one
    part."""
    for i in range(1, len(word)):
        left = word[:i]
        right = word[i:]
        if left in words and right in words:
            return [left, right]
    return [word]


def suffixes(lang: str) -> list:
    return SUFFIXES_NL if lang == "nl" else SUFFIXES


def words(lang: str) -> list:
    return WORDS_NL if lang == "nl" else WORDS


# the words the pictures under the checks show, in the lesson's language
CUT_WORDS = {"en": ["walked", "walks", "singing", "sing", "cats", "speed", "running", "bus"],
             "nl": ["werkt", "werkte", "boeken", "kat", "open", "nacht", "lopen"]}
SPLIT_WORDS = {"en": ["doorbell", "football", "sunlight", "rainbow", "bookstore", "dogs", "house"],
               "nl": ["voordeur", "fietsbel", "zonlicht", "regenboog", "boekwinkel", "honden", "huis"]}


class Named:
    """A value the check gives the learner's function, shown in a message by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


SUFFIXES_N = Named("SUFFIXES", SUFFIXES)
SUFFIXES_NL_N = Named("SUFFIXES_NL", SUFFIXES_NL)
WORDS_N = Named("WORDS", WORDS)
WORDS_NL_N = Named("WORDS_NL", WORDS_NL)

# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink);vertical-align:top")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"
_CODE = "font-family:var(--ea-font-code);font-size:12px"
_KIND = ("font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);font-weight:600;padding-top:0.35rem")
_PART = ("display:inline-block;margin:0 0.15rem 0 0;text-align:center;vertical-align:top")
_FREE = "border:1.5px solid var(--ea-ink-3)"
_BOUND = "border:1.5px dashed var(--ea-ink-3)"
_MORPH = ("display:block;border-radius:5px;padding:0.2rem 0.5rem;font-family:var(--ea-font-code);font-size:0.95rem;"
          "color:var(--ea-ink)")
_LABEL = "display:block;font-family:var(--ea-font-ui);font-size:0.72rem;color:var(--ea-ink-3);margin-top:0.15rem"


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def _morphemes(parts: list) -> str:
    """Morphemes as boxes side by side, each with its label under it: a solid frame for a part that can be a word on
    its own, a dashed one for an affix."""
    boxes = []
    for text, label, free in parts:
        boxes.append(f'<span style="{_PART}"><span style="{_MORPH};{_FREE if free else _BOUND}">{html.escape(text)}</span>'
                     f'<span style="{_LABEL}">{html.escape(label)}</span></span>')
    return "".join(boxes)


_PARTS_WORDS = {
    "en": {"title": "Three ways to make a word",
           "rows": [("Inflection", [("walk", "root", True), ("ed", "suffix", False)],
                     "a form of walk: the past"),
                    ("Derivation", [("care", "root", True), ("ful", "suffix", False), ("ly", "suffix", False)],
                     "a new word: care, then careful, then carefully"),
                    ("Compound", [("dog", "word", True), ("house", "word", True)],
                     "two words in one: a kind of house")],
           "legend": "Solid frame: a part that can be a word on its own. Dashed frame: an affix, which cannot."},
    "nl": {"title": "Drie manieren om een woord te maken",
           "rows": [("Inflectie", [("loop", "wortel", True), ("t", "achtervoegsel", False)],
                     "een vorm van lopen: jij loopt"),
                    ("Derivatie", [("schoon", "wortel", True), ("heid", "achtervoegsel", False)],
                     "een nieuw woord: van schoon naar schoonheid"),
                    ("Samenstelling", [("voor", "woord", True), ("deur", "woord", True)],
                     "twee woorden in één: een soort deur")],
           "legend": "Doorgetrokken kader: een deel dat op zichzelf een woord kan zijn. Gestreept kader: een affix, "
                     "dat niet los kan staan."},
}


def parts_picture(lang: str) -> str:
    """One word for each way to make words, cut into its morphemes."""
    w = _PARTS_WORDS[lang]
    rows, said = [], []
    for kind, parts, note in w["rows"]:
        rows.append(f'<div style="{_KIND}">{html.escape(kind)}</div><div>{_morphemes(parts)}'
                    f'<div style="{_UI}">{html.escape(note)}</div></div>')
        said.append(f'{kind}: {" + ".join(p[0] for p in parts)} ({", ".join(p[1] for p in parts)}), {note}')
    body = (f'<div style="display:grid;grid-template-columns:minmax(6rem,8rem) minmax(0,1fr);gap:0.6rem 0.6rem">'
            f'{"".join(rows)}</div>' + _note(w["legend"]))
    return yours.figure(w["title"], body, alt=f'{w["title"]}. {"; ".join(said)}. {w["legend"]}')


def _joined(value) -> str:
    """A list of parts as Python writes it, and as the parts with + between them: ['walk', 'ed'] is walk + ed."""
    if isinstance(value, list) and value and all(isinstance(p, str) for p in value):
        return " + ".join(value)
    return repr(value)


def _parts_cell(value) -> str:
    return f'<span style="{_CODE}">{html.escape(_joined(value))}</span>'


_DEMO_WORDS = {
    "en": {"title": "What the course's functions make of {word}", "empty": "Type a word in the box.",
           "lists": "Endings: {s}. Words: {w}.",
           "one": "One part: no ending of the list fits with at least 3 letters left, and no cut gives two known words."},
    "nl": {"title": "Wat de functies van de course van {word} maken", "empty": "Typ een woord in het vak.",
           "lists": "Uitgangen: {s}. Woorden: {w}.",
           "one": "Eén deel: geen uitgang van de lijst past met minstens 3 letters over, en geen knip geeft twee bekende "
                  "woorden."},
}


def demo_picture(word, lang: str) -> str:
    """A word the learner types, through the course's cut_suffix and split_compound, with the lists of the lesson's
    language."""
    w = _DEMO_WORDS[lang]
    word = str(word or "").strip().lower()[:40]
    lists = w["lists"].format(s=", ".join(suffixes(lang)), w=", ".join(words(lang)))
    if not word:
        return yours.figure(w["empty"], _note(lists), alt=f'{w["empty"]} {lists}')
    rows, said = [], []
    for name, fn, given in (("cut_suffix", cut_suffix, "SUFFIXES_NL" if lang == "nl" else "SUFFIXES"),
                            ("split_compound", split_compound, "WORDS_NL" if lang == "nl" else "WORDS")):
        parts = fn(word, suffixes(lang) if name == "cut_suffix" else words(lang))
        bound = [name == "cut_suffix" and k == 1 for k in range(len(parts))]   # the ending is an affix
        boxes = "".join(f'<span style="{_PART}"><span style="{_MORPH};{_BOUND if b else _FREE}">{html.escape(p)}</span></span>'
                        for p, b in zip(parts, bound))
        rows.append(f'<div style="{_KIND};font-family:var(--ea-font-code);font-weight:400;font-size:12px">'
                    f'{html.escape(name)}(word, {given})</div><div>{boxes}</div>')
        said.append(f"{name}: {_joined(parts)}")
    one = cut_suffix(word, suffixes(lang)) == [word] and split_compound(word, words(lang)) == [word]
    title = w["title"].format(word=word)
    body = (f'<div style="display:grid;grid-template-columns:minmax(8rem,14rem) minmax(0,1fr);gap:0.5rem 0.6rem;'
            f'align-items:center">{"".join(rows)}</div>' + (_note(w["one"]) if one else "") + _note(lists))
    return yours.figure(title, body, alt=f'{title}. {"; ".join(said)}. {w["one"] if one else ""} {lists}')


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


def _table(rows: list, head: list) -> str:
    """A table with a head row; each row is a list of (html, style)."""
    top = "".join(f'<th style="{_HEAD}">{html.escape(h)}</th>' for h in head)
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{";" + style if style else ""}">{cell}</td>'
                                    for cell, style in row) + "</tr>" for row in rows)
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
            f'<tr>{top}</tr>{body}</table></div>')


_CHECK_WORDS = {
    "en": {"error": "error", "head": ["word", "yours", "the course's"],
           "differs": "It differs from the course's for {k} of the {n} words: they have a purple dashed frame.",
           "cut": {"title": "Your cut_suffix(word, SUFFIXES) on English words",
                   "fine": "The same as the course's for all {n} words. Both cut speed, which is one morpheme, and give "
                           "runn for running, whose root is run: a list of endings sees letters, not meaning or "
                           "spelling."},
           "split": {"title": "Your split_compound(word, WORDS) on English words",
                     "fine": "The same as the course's for all {n} words. dogs stays whole: dog is a known word, but "
                             "the right piece, s, is not."},
           "alt": "{title}: {rows}. {after}"},
    "nl": {"error": "fout", "head": ["woord", "de jouwe", "die van de course"],
           "differs": "Bij {k} van de {n} woorden anders dan die van de course: die hebben een paars gestreept kader.",
           "cut": {"title": "Jouw cut_suffix(word, SUFFIXES_NL) op Nederlandse woorden",
                   "fine": "Bij alle {n} woorden hetzelfde als die van de course. Allebei knippen ze nacht, dat één "
                           "morfeem is, en geven ze lop voor lopen, met de wortel loop: een lijst uitgangen ziet "
                           "letters, geen betekenis of spelling."},
           "split": {"title": "Jouw split_compound(word, WORDS_NL) op Nederlandse woorden",
                     "fine": "Bij alle {n} woorden hetzelfde als die van de course. honden blijft heel: hond is een "
                             "bekend woord, maar het rechterstuk (en) niet."},
           "alt": "{title}: {rows}. {after}"},
}


def _picture(seen, which: str, course_fn, given: list, shown_words: list) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _CHECK_WORDS[seen.lang]
    rows, said, k = [], [], 0
    for word in shown_words:
        got, want = _call(fn, word, given), course_fn(word, given)
        differs = got is _STOPPED or got != want or type(got) is not type(want)
        k += differs
        mine = w["error"] if got is _STOPPED else _joined(got)
        rows.append([(f'<span style="{_CODE}">{html.escape(word)}</span>', ""),
                     (f'<span style="{_CODE}">{html.escape(mine)}</span>', _OFF if differs else ""),
                     (_parts_cell(want), "")])
        said.append(f"{word} {mine}, {_joined(want)}")
    n = len(shown_words)
    after = w["differs"].format(k=k, n=n) if k else w[which]["fine"].format(n=n)
    title = w[which]["title"]
    return yours.figure(title, _table(rows, w["head"]) + _note(after),
                        alt=w["alt"].format(title=title, rows="; ".join(said), after=after))


def cut_picture(seen) -> str:
    """The learner's cut_suffix beside the course's, on words of the lesson's language with its endings."""
    return _picture(seen, "cut", cut_suffix, suffixes(seen.lang), CUT_WORDS[seen.lang])


def split_picture(seen) -> str:
    """The learner's split_compound beside the course's, on words of the lesson's language with its list of words."""
    return _picture(seen, "split", split_compound, words(seen.lang), SPLIT_WORDS[seen.lang])


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["cut_suffix", "split_compound", "kinds", "merge"]

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
    return isinstance(e, NameError) and re.search(r"'(SUFFIXES|SUFFIXES_NL|WORDS|WORDS_NL|C)'", str(e)) is not None


_NAMES = Mistake("lesson-names", error=_names_error,
                 en="Your function uses a name of the lesson, such as `SUFFIXES`, `WORDS` or `C`, and the check does not give it that name: use the names between the parentheses of the `def` line.",
                 nl="Je functie gebruikt een naam van de les, zoals `SUFFIXES`, `WORDS` of `C`, en de check geeft die naam niet mee: gebruik de namen tussen de haakjes van de `def`-regel.")

# exercise 1: cut_suffix; the check also gives a list of its own and Dutch words
CUT_CASES = [
    ("walked", SUFFIXES_N), ("walks", SUFFIXES_N), ("sing", SUFFIXES_N), ("cats", SUFFIXES_N), ("bus", SUFFIXES_N),
    ("glasses", ["es", "s"]), ("werkte", SUFFIXES_NL_N), ("lopen", SUFFIXES_NL_N), ("speed", SUFFIXES_N),
]
_C_START = "def cut_suffix(word, suffixes):\n    ...\n"
_C = ("def cut_suffix(word, suffixes):\n"
      "    for suffix in suffixes:\n"
      "        rest = word[:len(word) - len(suffix)]\n"
      "        if rest + suffix == word and len(rest) >= 3:\n"
      "            return [rest, suffix]\n"
      "    return [word]\n")
_C_EARLY = _C.replace("            return [rest, suffix]\n", "            return [rest, suffix]\n        else:\n            return [word]\n")
_C_SHORT = _C.replace(" and len(rest) >= 3", "")
_C_STRICT = _C.replace("len(rest) >= 3", "len(rest) > 3")
_C_BARE = _C.replace("    return [word]\n", "    return word\n")
_C_REST = _C.replace("            return [rest, suffix]\n", "            return rest\n")
_C_LAST = ("def cut_suffix(word, suffixes):\n"
           "    parts = [word]\n"
           "    for suffix in suffixes:\n"
           "        rest = word[:len(word) - len(suffix)]\n"
           "        if rest + suffix == word and len(rest) >= 3:\n"
           "            parts = [rest, suffix]\n"
           "    return parts\n")

# exercise 2: split_compound; the check also gives Dutch words and lists of its own
SPLIT_CASES = [
    ("doorbell", WORDS_N), ("football", WORDS_N), ("dogs", WORDS_N), ("house", WORDS_N), ("voordeur", WORDS_NL_N),
    ("honden", WORDS_NL_N), ("bluebird", ["blue", "bird"]), ("ab", ["a", "b"]), ("x", ["x"]),
]
_S_START = "def split_compound(word, words):\n    ...\n"
_S = ("def split_compound(word, words):\n"
      "    for i in range(1, len(word)):\n"
      "        left = word[:i]\n"
      "        right = word[i:]\n"
      "        if left in words and right in words:\n"
      "            return [left, right]\n"
      "    return [word]\n")
_S_LEFT = _S.replace("        if left in words and right in words:\n", "        if left in words:\n")
_S_EARLY = _S.replace("            return [left, right]\n", "            return [left, right]\n        else:\n            return [word]\n")
_S_BARE = _S.replace("    return [word]\n", "    return word\n")
_S_PAIR = _S.replace("            return [left, right]\n", "            return (left, right)\n")


EXERCISES = {
    "cut_suffix": FunctionBox(
        name="cut_suffix",
        picture=cut_picture,
        function="cut_suffix",
        start=_C_START,
        solution=_C,
        visible=[CUT_CASES[0]],
        hidden=CUT_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("early", wrong=_C_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your function gives up after the first ending of the list. Try every ending, and give back the word as one part only after the loop, when none fitted.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft het na de eerste uitgang van de lijst al op. Probeer elke uitgang, en geef het woord pas na de loop als één deel terug, als er geen paste."),
            Mistake("rest", wrong=_C_REST,
                    en="`{call}` gives {got}, and it should give {want}: that is only what is left. Give back a list of both parts: what is left, and the ending.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is alleen wat overblijft. Geef een lijst met beide delen terug: wat overblijft, en de uitgang."),
            Mistake("short", wrong=_C_SHORT,
                    en="`{call}` gives {got}, and it should give {want}: what is left has fewer than 3 letters. Cut only when at least 3 letters are left, so that a word such as `sing` stays whole.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: wat overblijft, heeft minder dan 3 letters. Knip alleen als er minstens 3 letters overblijven, zodat een woord als `sing` heel blijft."),
            Mistake("bare", wrong=_C_BARE,
                    en="`{call}` gives {got}, and it should give {want}: a word without an ending of the list is one part. Give back a list with only the word, so that the function always gives back a list of parts.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een woord zonder uitgang van de lijst is één deel. Geef een lijst met alleen het woord terug, zodat de functie steeds een lijst delen teruggeeft."),
            Mistake("strict", wrong=_C_STRICT,
                    en="`{call}` gives {got}, and it should give {want}: exactly 3 letters are left, and that is enough. Cut when 3 or more letters are left.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: er blijven precies 3 letters over, en dat is genoeg. Knip als er 3 of meer letters overblijven."),
            Mistake("last", wrong=_C_LAST,
                    en="`{call}` gives {got}, and it should give {want}: that is the last ending of the list that fits. Take the first one, and give back the parts as soon as you find it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is de laatste uitgang van de lijst die past. Neem de eerste, en geef de delen terug zodra je die vindt."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: try the endings in order, and give back what is left and the first ending that fits, or the word as one part.",
                      "nl": "Vervang de `...`: probeer de uitgangen in volgorde, en geef terug wat overblijft en de eerste uitgang die past, of het woord als één deel."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The function gives back what is left and the first ending of the list that the word ends in, with at least 3 letters left, and else a list with only the word.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De functie geeft terug wat overblijft en de eerste uitgang van de lijst waarop het woord eindigt, met minstens 3 letters over, en anders een lijst met alleen het woord."},
            "almost": {"en": "Right for the first word. But `{call}` gives {got}, and it should give {want}. Does your function use the `word` and `suffixes` it gets, and keep at least 3 letters?",
                       "nl": "Goed voor het eerste woord. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `word` en `suffixes` die ze krijgt, en houdt ze minstens 3 letters over?"},
            "correct": {"en": "Your `cut_suffix` is right in all {n} cases, also for Dutch words. Look at the picture: like the course's own, it cuts speed, which has no ending.",
                        "nl": "Je `cut_suffix` klopt in alle {n} gevallen, ook voor Engelse woorden. Kijk naar de afbeelding: net als die van de course knipt je functie nacht, dat geen uitgang heeft."},
        },
    ),
    "split_compound": FunctionBox(
        name="split_compound",
        picture=split_picture,
        function="split_compound",
        start=_S_START,
        solution=_S,
        visible=[SPLIT_CASES[0]],
        hidden=SPLIT_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("early", wrong=_S_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your function gives up after the first cut. Try every cut, and give back the word as one part only after the loop, when none worked.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft het na de eerste knip al op. Probeer elke knip, en geef het woord pas na de loop als één deel terug, als er geen werkte."),
            Mistake("left", wrong=_S_LEFT,
                    en="`{call}` gives {got}, and it should give {want}: the left piece is a known word, but the right piece is not. Split only where both pieces are in the list.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het linkerstuk is een bekend woord, maar het rechterstuk niet. Splits alleen waar beide stukken in de lijst staan."),
            Mistake("bare", wrong=_S_BARE,
                    en="`{call}` gives {got}, and it should give {want}: a word without a cut into two known words is one part. Give back a list with only the word, so that the function always gives back a list of parts.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een woord zonder knip in twee bekende woorden is één deel. Geef een lijst met alleen het woord terug, zodat de functie steeds een lijst delen teruggeeft."),
            Mistake("pair", wrong=_S_PAIR,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back the two pieces between parentheses, a tuple. Give back a list, with square brackets.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft de twee stukken tussen ronde haakjes terug, een tuple. Geef een lijst terug, met vierkante haken."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: try the cuts from left to right, and give back the first one where both pieces are in the list, or the word as one part.",
                      "nl": "Vervang de `...`: probeer de knippen van links naar rechts, en geef de eerste terug waarbij beide stukken in de lijst staan, of het woord als één deel."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The function gives back the two pieces of the first cut where both are in the list, and else a list with only the word.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De functie geeft de twee stukken terug van de eerste knip waarbij beide in de lijst staan, en anders een lijst met alleen het woord."},
            "almost": {"en": "Right for the first word. But `{call}` gives {got}, and it should give {want}. Does your function use the `word` and `words` it gets, and try every cut?",
                       "nl": "Goed voor het eerste woord. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `word` en `words` die ze krijgt, en probeert ze elke knip?"},
            "correct": {"en": "Your `split_compound` is right in all {n} cases, also for Dutch words. Look at the picture: a word whose second piece is no known word stays whole.",
                        "nl": "Je `split_compound` klopt in alle {n} gevallen, ook voor Engelse woorden. Kijk naar de afbeelding: een woord waarvan het tweede stuk geen bekend woord is, blijft heel."},
        },
    ),
    "kinds": Sort(
        name="kinds",
        groups={
            "inflection": {"en": "Inflection", "nl": "Inflectie"},
            "derivation": {"en": "Derivation", "nl": "Derivatie"},
            "compound": {"en": "Compound", "nl": "Samenstelling"},
        },
        items={
            "past": {"en": "**walked**, from *walk*", "nl": "**liep**, van *lopen*"},
            "ful": {"en": "**careful**, from *care*", "nl": "**schoonheid**, van *schoon*"},
            "pair": {"en": "**doghouse**, from *dog* and *house*", "nl": "**voordeur**, van *voor* en *deur*"},
            "plural": {"en": "**cats**, from *cat*", "nl": "**katten**, van *kat*"},
            "un": {"en": "**unhappy**, from *happy*", "nl": "**ongewoon**, van *gewoon*"},
            "spaced": {"en": "**river boat race**, from *river*, *boat* and *race*",
                       "nl": "**sciencefiction**, van *science* en *fiction*"},
            "ness": {"en": "**happiness**, from *happy*", "nl": "**blijheid**, van *blij*"},
        },
        solution={"past": "inflection", "ful": "derivation", "pair": "compound", "plural": "inflection",
                  "un": "derivation", "spaced": "compound", "ness": "derivation"},
        nudge={
            "past": {"en": "Is this a new word, or the same word in another form?",
                     "nl": "Is dit een nieuw woord, of hetzelfde woord in een andere vorm?"},
            "ful": {"en": "Can you put *the* before this word, as you can before *care*?",
                    "nl": "Kun je *de* of *het* voor dit woord zetten? En voor *schoon*?"},
            "pair": {"en": "How many words went into it?",
                     "nl": "Hoeveel woorden gingen erin?"},
            "plural": {"en": "Does the ending make a new word, or fit the word to the sentence?",
                       "nl": "Maakt de uitgang een nieuw woord, of past die het woord aan de zin aan?"},
            "un": {"en": "Does the prefix make another form of the same word, or a word with another meaning?",
                   "nl": "Maakt het voorvoegsel een andere vorm van hetzelfde woord, of een woord met een andere betekenis?"},
            "spaced": {"en": "How many words went into it, whatever the spaces?",
                       "nl": "Hoeveel woorden gingen erin, ook al komen ze uit het Engels?"},
            "ness": {"en": "Is this a form of the word it comes from, or a word of its own?",
                     "nl": "Is dit een vorm van het woord waar het van komt, of een woord op zichzelf?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the words again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de woorden opnieuw uit."},
            "empty": {"en": "Put each word in a group: inflection, derivation or compound.",
                      "nl": "Zet elk woord in een groep: inflectie, derivatie of samenstelling."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs in another group.",
                      "nl": "{item} hoort in een andere groep."},
            "correct": {"en": "All {n} are right. Inflection makes forms of one word, derivation makes new words, here also one of the same kind as its base, and a compound joins words, with or without spaces.",
                        "nl": "Alle {n} kloppen. Inflectie maakt vormen van één woord, derivatie maakt nieuwe woorden, hier ook een van hetzelfde soort als het basiswoord, en een samenstelling voegt woorden samen."},
        },
    ),
}

CASES = [
    ("cut_suffix", _C_START, "empty", None),
    ("cut_suffix", _C, "correct", None),
    ("cut_suffix", "def cut_suffix(word, suffixes):\n    for suffix in suffixes:\n        if word.endswith(suffix) and len(word) - len(suffix) >= 3:\n            return [word[:len(word) - len(suffix)], suffix]\n    return [word]\n", "correct", None),
    ("cut_suffix", _C_EARLY, "wrong", "early"),
    ("cut_suffix", _C_REST, "wrong", "rest"),
    ("cut_suffix", _C_SHORT, "wrong", "short"),
    ("cut_suffix", _C_BARE, "wrong", "bare"),
    ("cut_suffix", _C_STRICT, "wrong", "strict"),
    ("cut_suffix", _C_LAST, "wrong", "last"),
    ("cut_suffix", "def cut_suffix(word, suffixes):\n    return ['walk', 'ed']\n", "almost", None),
    ("cut_suffix", "def cut_suffix(word, suffixes):\n    return cut_suffix(word, SUFFIXES[1:])\n", "error", "lesson-names"),
    ("split_compound", _S_START, "empty", None),
    ("split_compound", _S, "correct", None),
    ("split_compound", "def split_compound(word, words):\n    for left in words:\n        for right in words:\n            if left + right == word:\n                return [left, right]\n    return [word]\n", "correct", None),
    ("split_compound", _S_EARLY, "wrong", "early"),
    ("split_compound", _S_LEFT, "wrong", "left"),
    ("split_compound", _S_BARE, "wrong", "bare"),
    ("split_compound", _S_PAIR, "wrong", "pair"),
    ("split_compound", "def split_compound(word, words):\n    return ['door', 'bell']\n", "almost", None),
    ("split_compound", "def split_compound(word, words):\n    return [w for w in WORDS if w in word]\n", "error", "lesson-names"),
    ("kinds", {}, "empty", None),
    ("kinds", {"past": "inflection", "ful": "derivation", "pair": "compound", "plural": "inflection",
               "un": "derivation", "spaced": "compound", "ness": "derivation"}, "correct", None),
    ("kinds", {"past": "inflection", "ful": "derivation"}, "wrong", "unfinished"),
    ("kinds", {"past": "inflection", "ful": "derivation", "pair": "compound", "plural": "inflection",
               "un": "inflection", "spaced": "compound", "ness": "derivation"}, "wrong", "wrong:un"),
    ("kinds", {"past": "derivation"}, "wrong", "wrong:past"),
]
