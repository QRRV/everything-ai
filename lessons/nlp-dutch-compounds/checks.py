# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-dutch-compounds: split a Dutch compound into two known words with a linking sound between
them, in every way; split a word into known words in every way, with a function that calls itself; tell a real compound
from a split that only fits the letters; and say what tells two splits apart. The list of Dutch words comes from Open
Dutch WordNet (data/wordnet-nl.tsv, made by data/tools/make_wordnet_nl.py); keeping words of three letters or more, and
the linking sounds "", "s" and "en", are the course's own choices."""
import contextlib
import copy
import functools
import html
import io
import re

from everything_ai import paths, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, with_time_limit

# --- the course's lists ----------------------------------------------------------------------------------------------

WORDNET_NL = paths.ROOT / "data" / "wordnet-nl.tsv"
LINKS = ["", "s", "en"]   # no linking sound, -s- (stadsmuur) or -en- (hondenhok)


def _dutch_words() -> set:
    """The lemmas of Open Dutch WordNet that are one word of three or more small letters."""
    words = set()
    for line in WORDNET_NL.read_text(encoding="utf-8").splitlines()[1:]:
        lemma = line.split("\t")[1]
        if lemma.isalpha() and lemma.islower() and len(lemma) >= 3:
            words.add(lemma)
    return words


DUTCH = _dutch_words()


def link_splits(word: str, words, links: list) -> list:
    """The course's own Exercise 1: every split of the word into a known word, a linking sound and a known word."""
    found = []
    for i in range(1, len(word)):
        left = word[:i]
        right = word[i:]
        for link in links:
            stem = left[:len(left) - len(link)]
            if stem + link == left and stem in words and right in words:
                found.append([stem, link, right])
    return found


def all_splits(word: str, words) -> list:
    """The course's own Exercise 2: every way to cut the word into known words, the word itself first when it is one."""
    found = []
    if word in words:
        found.append([word])
    for i in range(1, len(word)):
        left = word[:i]
        if left in words:
            for rest in all_splits(word[i:], words):
                found.append([left] + rest)
    return found


@functools.lru_cache(maxsize=4096)
def linked_splits(word: str) -> tuple:
    """For the demo: every way to cut the word into words of DUTCH, with a linking sound of LINKS between two words,
    as tuples of parts; the same as all_splits with the linking sounds of link_splits, and remembered, so that a long
    word takes no time."""
    found = []
    if word in DUTCH:
        found.append((word,))
    for i in range(1, len(word)):
        left = word[:i]
        for link in LINKS:
            stem = left[:len(left) - len(link)]
            if stem + link == left and stem in DUTCH:
                for rest in linked_splits(word[i:]):
                    found.append((stem,) + ((link,) if link else ()) + rest)
    return tuple(found)


class Named:
    """A value the check gives the learner's function, shown in a message by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


DUTCH_N = Named("C.DUTCH", DUTCH)
LINKS_N = Named("C.LINKS", LINKS)

# the words the pictures under the checks show
LINK_WORDS = ["hondenhok", "stadsmuur", "ziekenhuis", "huiswerk", "verjaardagstaart", "voordeur", "zonnebril"]
ALL_WORDS = ["voetbalveld", "valkuil", "politiekring", "kantoor", "fietsbel", "reactiestappen", "zonnebril"]

# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink);vertical-align:top")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"
_CODE = "font-family:var(--ea-font-code);font-size:12px"
_KIND = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);font-weight:600;padding-top:0.35rem"
_PART = "display:inline-block;margin:0 0.15rem 0.2rem 0;text-align:center;vertical-align:top"
_WORD = "border:1.5px solid var(--ea-ink-3)"
_LINK = "border:1.5px dashed var(--ea-ink-3)"
_MORPH = ("display:block;border-radius:5px;padding:0.2rem 0.5rem;font-family:var(--ea-font-code);font-size:0.95rem;"
          "color:var(--ea-ink)")
_LABEL = "display:block;font-family:var(--ea-font-ui);font-size:0.72rem;color:var(--ea-ink-3);margin-top:0.15rem"
_OR = "display:inline-block;margin:0 0.5rem 0 0.3rem;font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3)"


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def _boxes(parts, labels=None) -> str:
    """Parts side by side: a solid frame for a word, a dashed one for a linking sound; a label under each when given."""
    out = []
    for k, part in enumerate(parts):
        link = part in LINKS
        label = f'<span style="{_LABEL}">{html.escape(labels[k])}</span>' if labels else ""
        out.append(f'<span style="{_PART}"><span style="{_MORPH};{_LINK if link else _WORD}">{html.escape(part)}</span>'
                   f'{label}</span>')
    return "".join(out)


_PARTS_WORDS = {
    "en": {"title": "Three things a splitter meets",
           "rows": [("Parts in parts", [(["voetbal", "veld"], ["football", "field"]),
                                        (["voet", "bal", "veld"], ["foot", "ball", "field"])],
                     "voetbal is itself a compound, so the parts can be split again"),
                    ("A linking sound", [(["hond", "en", "hok"], ["dog", "linking sound", "shed"])],
                     "a doghouse: -en- between the two words"),
                    ("Two splits", [(["val", "kuil"], ["fall", "pit"]), (["valk", "uil"], ["falcon", "owl"])],
                     "a pitfall, or a falcon owl: the same letters")],
           "or": "or",
           "legend": "Solid frame: a known word. Dashed frame: a linking sound."},
    "nl": {"title": "Drie dingen die een splitser tegenkomt",
           "rows": [("Delen in delen", [(["voetbal", "veld"], ["woord", "woord"]),
                                        (["voet", "bal", "veld"], ["woord", "woord", "woord"])],
                     "voetbal is zelf een samenstelling, dus de delen kun je weer splitsen"),
                    ("Een tussenklank", [(["hond", "en", "hok"], ["woord", "tussenklank", "woord"])],
                     "een hondenhok: -en- tussen de twee woorden"),
                    ("Twee splitsingen", [(["val", "kuil"], ["woord", "woord"]), (["valk", "uil"], ["woord", "woord"])],
                     "een valkuil, of een soort uil: dezelfde letters")],
           "or": "of",
           "legend": "Doorgetrokken kader: een bekend woord. Gestreept kader: een tussenklank."},
}


def parts_picture(lang: str) -> str:
    """The three things of The idea: a part that is itself a compound, a linking sound, and two splits of one word."""
    w = _PARTS_WORDS[lang]
    rows, said = [], []
    for kind, splits, note in w["rows"]:
        shown = f'<span style="{_OR}">{html.escape(w["or"])}</span>'.join(
            f'<span style="display:inline-block">{_boxes(parts, labels)}</span>' for parts, labels in splits)
        rows.append(f'<div style="{_KIND}">{html.escape(kind)}</div><div>{shown}'
                    f'<div style="{_UI}">{html.escape(note)}</div></div>')
        sep = f' {w["or"]} '
        said.append(f'{kind}: {sep.join(" + ".join(p) for p, _ in splits)}, {note}')
    body = (f'<div style="display:grid;grid-template-columns:minmax(6rem,9rem) minmax(0,1fr);gap:0.6rem 0.6rem">'
            f'{"".join(rows)}</div>' + _note(w["legend"]))
    return yours.figure(w["title"], body, alt=f'{w["title"]}. {"; ".join(said)}. {w["legend"]}')


def _split_text(split) -> str:
    """One split the way the lesson writes it: hond + en + hok; an empty linking sound shows as ""."""
    if isinstance(split, list) and all(isinstance(p, str) for p in split):
        return " + ".join(p if p else '""' for p in split)
    return repr(split)


def _splits_text(value) -> list:
    """What a function gave back as lines of text: one line per split, or the value as Python shows it."""
    if isinstance(value, list) and all(isinstance(s, list) for s in value):
        return [_split_text(s) for s in value] or ["[]"]
    return [repr(value)]


_DEMO_WORDS = {
    "en": {"title": "Every split of {word}", "empty": "Type a word in the box.", "shown": "the first {k} of {n}",
           "none": "No split: a part is not in the list, or the spelling changed, as in zonnebank, from zon.",
           "count": "{n} ways, the fewest parts first.",
           "source": "The words come from Open Dutch WordNet, with the linking sounds -s- and -en-.",
           "parts": "{k} parts", "part": "1 part"},
    "nl": {"title": "Elke splitsing van {word}", "empty": "Typ een woord in het vak.", "shown": "de eerste {k} van {n}",
           "none": "Geen splitsing: een deel staat niet in de lijst, of de spelling veranderde, zoals in zonnebank, van "
                   "zon.",
           "count": "{n} manieren, de minste delen eerst.",
           "source": "De woorden komen uit Open Dutch WordNet, met de tussenklanken -s- en -en-.",
           "parts": "{k} delen", "part": "1 deel"},
}
DEMO_MAX = 6   # splits the demo shows


def demo_picture(word, lang: str) -> str:
    """A word the learner types, cut in every way into words of DUTCH with the linking sounds of LINKS."""
    w = _DEMO_WORDS[lang]
    word = str(word or "").strip().lower()[:40]
    if not word:
        return yours.figure(w["empty"], _note(w["source"]), alt=f'{w["empty"]} {w["source"]}')
    found = sorted(linked_splits(word), key=lambda s: len([p for p in s if p not in LINKS]))
    title = w["title"].format(word=word)
    if not found:
        return yours.figure(title, _note(w["none"]) + _note(w["source"]), alt=f'{title}. {w["none"]} {w["source"]}')
    rows, said = [], []
    for split in found[:DEMO_MAX]:
        k = len([p for p in split if p not in LINKS])
        label = w["part"] if k == 1 else w["parts"].format(k=k)
        rows.append(f'<div style="{_KIND};font-weight:400">{html.escape(label)}</div>'
                    f'<div>{_boxes(split)}</div>')
        said.append(" + ".join(split))
    count = w["count"].format(n=len(found))
    if len(found) > DEMO_MAX:
        count += f' ({w["shown"].format(k=DEMO_MAX, n=len(found))})'
    body = (f'<div style="display:grid;grid-template-columns:minmax(4rem,6rem) minmax(0,1fr);gap:0.3rem 0.6rem;'
            f'align-items:center">{"".join(rows)}</div>' + _note(count) + _note(w["source"]))
    return yours.figure(title, body, alt=f'{title}: {"; ".join(said)}. {count} {w["source"]}')


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


def _cell(lines: list) -> str:
    return "<br>".join(f'<span style="{_CODE}">{html.escape(t)}</span>' for t in lines)


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
           "link": {"title": "Your link_splits(word, C.DUTCH, C.LINKS) on Dutch words",
                    "fine": "The same as the course's for all {n} words. For zonnebril the function finds nothing, since "
                            "its linking sound comes with a spelling change, from zon to zonne. And hui (whey) in "
                            "huiswerk is a word too."},
           "all": {"title": "Your all_splits(word, C.DUTCH) on Dutch words",
                   "fine": "The same as the course's for all {n} words. Each list holds the word itself when it is a "
                           "known word, and the splits from the shortest first part to the longest."},
           "alt": "{title}: {rows}. {after}"},
    "nl": {"error": "fout", "head": ["woord", "de jouwe", "die van de course"],
           "differs": "Bij {k} van de {n} woorden anders dan die van de course: die hebben een paars gestreept kader.",
           "link": {"title": "Jouw link_splits(word, C.DUTCH, C.LINKS) op Nederlandse woorden",
                    "fine": "Bij alle {n} woorden hetzelfde als die van de course. Bij zonnebril vindt de functie niets, "
                            "omdat zon daar met de tussenklank zonne wordt. En hui (wei) in huiswerk is ook een woord."},
           "all": {"title": "Jouw all_splits(word, C.DUTCH) op Nederlandse woorden",
                   "fine": "Bij alle {n} woorden hetzelfde als die van de course. Elke lijst heeft het woord zelf als "
                           "dat een bekend woord is, en de splitsingen van het kortste eerste deel tot het langste."},
           "alt": "{title}: {rows}. {after}"},
}


def _picture(seen, which: str, course_fn, extra: tuple, shown_words: list) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _CHECK_WORDS[seen.lang]
    rows, said, k = [], [], 0
    for word in shown_words:
        got, want = _call(fn, word, *extra), course_fn(word, *extra)
        differs = got is _STOPPED or got != want
        k += differs
        mine = [w["error"]] if got is _STOPPED else _splits_text(got)
        theirs = _splits_text(want)
        rows.append([(f'<span style="{_CODE}">{html.escape(word)}</span>', ""),
                     (_cell(mine), _OFF if differs else ""), (_cell(theirs), "")])
        said.append(f"{word}: {'; '.join(mine)}, {'; '.join(theirs)}")
    n = len(shown_words)
    after = w["differs"].format(k=k, n=n) if k else w[which]["fine"].format(n=n)
    title = w[which]["title"]
    return yours.figure(title, _table(rows, w["head"]) + _note(after),
                        alt=w["alt"].format(title=title, rows="; ".join(said), after=after))


def link_picture(seen) -> str:
    """The learner's link_splits beside the course's, on Dutch words with DUTCH and LINKS."""
    return _picture(seen, "link", link_splits, (DUTCH, LINKS), LINK_WORDS)


def all_picture(seen) -> str:
    """The learner's all_splits beside the course's, on Dutch words with DUTCH."""
    return _picture(seen, "all", all_splits, (DUTCH,), ALL_WORDS)


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["link_splits", "all_splits", "real", "which"]

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
    return isinstance(e, NameError) and re.search(r"'(DUTCH|LINKS|C)'", str(e)) is not None


_NAMES = Mistake("lesson-names", error=_names_error,
                 en="Your function uses a name of the lesson, such as `DUTCH`, `LINKS` or `C`, and the check does not give it that name: use the names between the parentheses of the `def` line.",
                 nl="Je functie gebruikt een naam van de les, zoals `DUTCH`, `LINKS` of `C`, en de check geeft die naam niet mee: gebruik de namen tussen de haakjes van de `def`-regel.")

# exercise 1: link_splits; the check also gives lists of its own, and a German word with the German linking sounds
LINK_CASES = [
    ("hondenhok", DUTCH_N, LINKS_N), ("ziekenhuis", DUTCH_N, LINKS_N), ("huiswerk", DUTCH_N, LINKS_N),
    ("stadsmuur", DUTCH_N, LINKS_N), ("zonnebril", DUTCH_N, LINKS_N), ("valkuil", ["val", "kuil", "valk", "uil"], [""]),
    ("aktionsplan", ["aktion", "aktions", "plan", "akt", "ion"], ["", "s", "es"]),
    ("boekenkast", ["boek", "boeken", "kast"], ["en", ""]), ("hondhok", ["ho", "hond", "hok"], ["", "en"]),
    ("stadsmuur", ["stad", "stads", "muur", "smuur"], ["s", ""]),
    ("ab", ["a", "b"], [""]),
]
_L_START = "def link_splits(word, words, links):\n    ...\n"
_L = ("def link_splits(word, words, links):\n"
      "    found = []\n"
      "    for i in range(1, len(word)):\n"
      "        left = word[:i]\n"
      "        right = word[i:]\n"
      "        for link in links:\n"
      "            stem = left[:len(left) - len(link)]\n"
      "            if stem + link == left and stem in words and right in words:\n"
      "                found.append([stem, link, right])\n"
      "    return found\n")
_L_FIRST = _L.replace("                found.append([stem, link, right])\n", "                return [[stem, link, right]]\n")
_L_PAIR = _L.replace("found.append([stem, link, right])", "found.append([stem, right])")
_L_LEFT = _L.replace("found.append([stem, link, right])", "found.append([left, link, right])")
_L_NOCHECK = _L.replace("if stem + link == left and stem in words", "if stem in words")
_L_EARLY = _L.replace("    return found\n", "        return found\n")
_L_ORDER = ("def link_splits(word, words, links):\n"
            "    found = []\n"
            "    for link in links:\n"
            "        for i in range(1, len(word)):\n"
            "            left = word[:i]\n"
            "            right = word[i:]\n"
            "            stem = left[:len(left) - len(link)]\n"
            "            if stem + link == left and stem in words and right in words:\n"
            "                found.append([stem, link, right])\n"
            "    return found\n")

# exercise 2: all_splits; the check also gives lists of its own
ALL_CASES = [
    ("voetbalveld", DUTCH_N), ("valkuil", DUTCH_N), ("politiekring", DUTCH_N), ("zonnebril", DUTCH_N),
    ("aktionsplan", ["aktionsplan", "aktion", "aktions", "plan", "akt", "ion"]), ("abc", ["a", "b", "c", "ab", "bc"]),
    ("xyz", ["x", "z"]), ("x", ["x"]),
]
_A_START = "def all_splits(word, words):\n    ...\n"
_A = ("def all_splits(word, words):\n"
      "    found = []\n"
      "    if word in words:\n"
      "        found.append([word])\n"
      "    for i in range(1, len(word)):\n"
      "        left = word[:i]\n"
      "        if left in words:\n"
      "            for rest in all_splits(word[i:], words):\n"
      "                found.append([left] + rest)\n"
      "    return found\n")
_A_NOWHOLE = _A.replace("    if word in words:\n        found.append([word])\n", "")
_A_TWO = ("def all_splits(word, words):\n"
          "    found = []\n"
          "    if word in words:\n"
          "        found.append([word])\n"
          "    for i in range(1, len(word)):\n"
          "        left = word[:i]\n"
          "        if left in words and word[i:] in words:\n"
          "            found.append([left, word[i:]])\n"
          "    return found\n")
_A_NOLEFT = _A.replace("found.append([left] + rest)", "found.append(rest)")
_A_UNCHECKED = _A.replace("        if left in words:\n            for rest in all_splits(word[i:], words):\n"
                          "                found.append([left] + rest)\n",
                          "        for rest in all_splits(word[i:], words):\n            found.append([left] + rest)\n")
_A_WHOLE_ONLY = _A.replace("    if word in words:\n        found.append([word])\n",
                           "    if word in words:\n        return [[word]]\n")
_A_LAST = _A_NOWHOLE.replace("    return found\n", "    if word in words:\n        found.append([word])\n    return found\n")


def _string_glued(e) -> bool:
    """[left] + rest where rest is a string: the word itself went into the list without its own brackets."""
    return isinstance(e, TypeError) and 'can only concatenate list (not "str") to list' in str(e)


EXERCISES = {
    "link_splits": FunctionBox(
        name="link_splits",
        picture=link_picture,
        function="link_splits",
        start=_L_START,
        solution=_L,
        visible=[LINK_CASES[0]],
        hidden=LINK_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("first", wrong=_L_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: your function stops at the first split it finds. Add each split to the list, and give back the list after both loops.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie stopt bij de eerste splitsing die ze vindt. Voeg elke splitsing toe aan de lijst, en geef de lijst na beide loops terug."),
            Mistake("early", wrong=_L_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back the list after the first cut. Move `return found` to the left, under `for i in ...`, so that it comes after the loop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft de lijst al na de eerste knip terug. Zet `return found` meer naar links, onder `for i in ...`, zodat die na de loop komt."),
            Mistake("pair", wrong=_L_PAIR,
                    en="`{call}` gives {got}, and it should give {want}: the linking sound is missing from each split. Keep it in the middle: `[stem, link, right]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de tussenklank ontbreekt in elke splitsing. Zet die in het midden: `[stem, link, right]`."),
            Mistake("left", wrong=_L_LEFT,
                    en="`{call}` gives {got}, and it should give {want}: the first part still ends in the linking sound. Put `stem`, the left piece without the sound, first.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het eerste deel eindigt nog op de tussenklank. Zet `stem`, het linkerstuk zonder de klank, vooraan."),
            Mistake("no-check", wrong=_L_NOCHECK,
                    en="`{call}` gives {got}, and it should give {want}: there is a split where the left piece does not end in that linking sound. Check for each linking sound that the left piece really ends in it, as in the worked example.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: er is een splitsing waarbij het linkerstuk niet op die tussenklank eindigt. Controleer bij elke tussenklank of het linkerstuk er echt op eindigt, zoals in het uitgewerkte voorbeeld."),
            Mistake("order", wrong=_L_ORDER,
                    en="`{call}` gives {got}, and it should give {want}: the same splits, in another order. Go over the cuts in the outer loop, from left to right, and over the linking sounds inside it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dezelfde splitsingen, in een andere volgorde. Ga in de buitenste loop langs de knippen, van links naar rechts, en daarbinnen langs de tussenklanken."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: start with an empty list, try every cut and every linking sound, and add each split that works to the list.",
                      "nl": "Vervang de `...`: begin met een lege lijst, probeer elke knip en elke tussenklank, en voeg elke splitsing die werkt toe aan de lijst."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The function gives back a list of every split `[stem, link, right]`, by cut from left to right and by linking sound in the order of the list, where the left piece ends in the linking sound and both `stem` and `right` are known words.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De functie geeft een lijst terug met elke splitsing `[stem, link, right]`, per knip van links naar rechts en per tussenklank in de volgorde van de lijst, waarbij het linkerstuk op de tussenklank eindigt en `stem` en `right` allebei bekende woorden zijn."},
            "almost": {"en": "Right for the first word. But `{call}` gives {got}, and it should give {want}. Does your function use the `word`, `words` and `links` it gets, and keep every split it finds?",
                       "nl": "Goed voor het eerste woord. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `word`, `words` en `links` die ze krijgt, en houdt ze elke splitsing die ze vindt?"},
            "correct": {"en": "Your `link_splits` is right in all {n} cases, also for the German aktionsplan with its linking sounds -s- and -es-. Look at the picture: huiswerk has two splits, and zonnebril none.",
                        "nl": "Je `link_splits` klopt in alle {n} gevallen, ook voor het Duitse aktionsplan met de Duitse tussenklanken -s- en -es-. Kijk naar de afbeelding: huiswerk heeft twee splitsingen, en zonnebril geen."},
        },
    ),
    "all_splits": FunctionBox(
        name="all_splits",
        picture=all_picture,
        function="all_splits",
        start=_A_START,
        solution=_A,
        visible=[ALL_CASES[0]],
        hidden=ALL_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("string", error=_string_glued,
                    en="With `{call}`, `[left] + rest` glues a list and a string, and Python cannot do that: a split is a list of parts. Add the word itself to the list as `[word]`, between square brackets.",
                    nl="Met `{call}` plakt `[left] + rest` een lijst en een string aan elkaar, en dat kan Python niet: een splitsing is een lijst delen. Voeg het woord zelf als `[word]` toe aan de lijst, tussen vierkante haken."),
            Mistake("no-whole", wrong=_A_NOWHOLE,
                    en="`{call}` gives {got}, and it should give {want}: a known word on its own is a split too, one of one part. When `word` is in `words`, add `[word]` to the list first; the calls for the rest of a word need it to find anything.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een bekend woord op zichzelf is ook een splitsing, van één deel. Voeg `[word]` eerst toe aan de lijst als `word` in `words` staat; de aanroepen voor de rest van een woord hebben dat nodig om iets te vinden."),
            Mistake("two-only", wrong=_A_TWO,
                    en="`{call}` gives {got}, and it should give {want}: your function only splits into two parts. Let `all_splits` split the rest of the word too, and put the left piece before each split it gives back.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie splitst alleen in twee delen. Laat `all_splits` ook de rest van het woord splitsen, en zet het linkerstuk vóór elke splitsing die je terugkrijgt."),
            Mistake("no-left", wrong=_A_NOLEFT,
                    en="`{call}` gives {got}, and it should give {want}: the left piece is missing from the splits. Put it before each split of the rest: `[left] + rest`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het linkerstuk ontbreekt in de splitsingen. Zet het vóór elke splitsing van de rest: `[left] + rest`."),
            Mistake("unchecked", wrong=_A_UNCHECKED,
                    en="`{call}` gives {got}, and it should give {want}: there is a split whose first part is no known word. Split the rest only when the left piece is in `words`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: er is een splitsing waarvan het eerste deel geen bekend woord is. Splits de rest alleen als het linkerstuk in `words` staat."),
            Mistake("whole-only", wrong=_A_WHOLE_ONLY,
                    en="`{call}` gives {got}, and it should give {want}: your function stops as soon as the word itself is known. Add `[word]` to the list and go on with the cuts: a known word can have splits too.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie stopt zodra het woord zelf bekend is. Voeg `[word]` toe aan de lijst en ga door met de knippen: een bekend woord kan ook splitsingen hebben."),
            Mistake("whole-last", wrong=_A_LAST,
                    en="`{call}` gives {got}, and it should give {want}: the same splits, in another order, with the word itself last. Add `[word]` to the list first, before the cuts.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dezelfde splitsingen, in een andere volgorde, met het woord zelf achteraan. Voeg `[word]` eerst toe aan de lijst, vóór de knippen."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: start with an empty list, add the word itself when it is known, and for each cut whose left piece is known, add the left piece with each split of the rest.",
                      "nl": "Vervang de `...`: begin met een lege lijst, voeg het woord zelf toe als dat bekend is, en voeg voor elke knip met een bekend linkerstuk het linkerstuk toe met elke splitsing van de rest."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The function gives back a list of every way to cut the word into known words: first `[word]` when the word itself is known, then, by cut from left to right, the left piece before each split of the rest.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De functie geeft een lijst terug met elke manier om het woord in bekende woorden te knippen: eerst `[word]` als het woord zelf bekend is, en dan, per knip van links naar rechts, het linkerstuk vóór elke splitsing van de rest."},
            "almost": {"en": "Right for the first word. But `{call}` gives {got}, and it should give {want}. Does your function use the `word` and `words` it gets, and call itself with the same `words`?",
                       "nl": "Goed voor het eerste woord. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `word` en `words` die ze krijgt, en roept ze zichzelf aan met dezelfde `words`?"},
            "correct": {"en": "Your `all_splits` is right in all {n} cases, also for aktionsplan with the German words of Koehn and Knight. Look at the picture: kantoor has a split that fits the letters, not the meaning.",
                        "nl": "Je `all_splits` klopt in alle {n} gevallen, ook voor aktionsplan met de Duitse woorden van Koehn en Knight. Kijk naar de afbeelding: kantoor heeft een splitsing die bij de letters past, niet bij de betekenis."},
        },
    ),
    "real": Sort(
        name="real",
        groups={
            "real": {"en": "A real compound", "nl": "Een echte samenstelling"},
            "false": {"en": "Only the letters fit", "nl": "Alleen de letters passen"},
        },
        items={
            "veld": {"en": "**voetbal + veld** (football + field) in *voetbalveld* (football field)",
                     "nl": "**voetbal + veld** in *voetbalveld*"},
            "kantoor": {"en": "**kant + oor** (side + ear) in *kantoor* (office)",
                        "nl": "**kant + oor** in *kantoor*"},
            "bestand": {"en": "**bes + tand** (berry + tooth) in *bestand* (file)",
                        "nl": "**bes + tand** in *bestand*"},
            "arts": {"en": "**tand + arts** (tooth + doctor) in *tandarts* (dentist)",
                     "nl": "**tand + arts** in *tandarts*"},
            "hui": {"en": "**hui + -s- + werk** (whey + -s- + work) in *huiswerk* (homework)",
                    "nl": "**hui + -s- + werk** in *huiswerk* (*hui* is wei)"},
            "boog": {"en": "**regen + boog** (rain + bow) in *regenboog* (rainbow)",
                     "nl": "**regen + boog** in *regenboog*"},
            "staart": {"en": "**verjaardag + staart** (birthday + tail) in *verjaardagstaart* (birthday cake)",
                       "nl": "**verjaardag + staart** in *verjaardagstaart*"},
        },
        solution={"veld": "real", "kantoor": "false", "bestand": "false", "arts": "real", "hui": "false",
                  "boog": "real", "staart": "false"},
        nudge={
            "veld": {"en": "Is a football field a kind of field?",
                     "nl": "Is een voetbalveld een soort veld?"},
            "kantoor": {"en": "Does an office have anything to do with a side or an ear?",
                        "nl": "Heeft een kantoor iets met een kant of een oor te maken?"},
            "bestand": {"en": "Does a file on a computer have anything to do with berries or teeth?",
                        "nl": "Heeft een bestand op een computer iets met bessen of tanden te maken?"},
            "arts": {"en": "Is a dentist a doctor who looks after teeth?",
                     "nl": "Is een tandarts een arts die naar tanden kijkt?"},
            "hui": {"en": "Is homework about whey, or about work done at home, in the house (*huis*)?",
                    "nl": "Gaat huiswerk over wei, of over werk voor thuis, in huis?"},
            "boog": {"en": "Is a rainbow a bow, an arch, made by the rain?",
                     "nl": "Is een regenboog een boog die de regen maakt?"},
            "staart": {"en": "Is a birthday cake a tail? Which other split has a linking sound?",
                       "nl": "Is een verjaardagstaart een staart? Welke andere splitsing heeft een tussenklank?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the words again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de woorden opnieuw uit."},
            "empty": {"en": "Put each split in a group: a real compound, or only the letters fit.",
                      "nl": "Zet elke splitsing in een groep: een echte samenstelling, of alleen de letters passen."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs in the other group.",
                      "nl": "{item} hoort in de andere groep."},
            "correct": {"en": "All {n} are right. A list of words only sees letters: each of these splits is made of known words, but only in a real compound do the parts give the meaning.",
                        "nl": "Alle {n} kloppen. Een lijst woorden ziet alleen letters: elke splitsing hier bestaat uit bekende woorden, maar alleen in een echte samenstelling geven de delen de betekenis."},
        },
    ),
}

CASES = [
    ("link_splits", _L_START, "empty", None),
    ("link_splits", _L, "correct", None),
    ("link_splits", "def link_splits(word, words, links):\n    found = []\n    for i in range(1, len(word)):\n        for link in links:\n            if word[:i].endswith(link) and word[:i - len(link)] in words and word[i:] in words:\n                found.append([word[:i - len(link)], link, word[i:]])\n    return found\n", "correct", None),
    ("link_splits", _L_FIRST, "wrong", "first"),
    ("link_splits", _L_EARLY, "wrong", "early"),
    ("link_splits", _L_PAIR, "wrong", "pair"),
    ("link_splits", _L_LEFT, "wrong", "left"),
    ("link_splits", _L_NOCHECK, "wrong", "no-check"),
    ("link_splits", _L_ORDER, "wrong", "order"),
    ("link_splits", "def link_splits(word, words, links):\n    return [['hond', 'en', 'hok']]\n", "almost", None),
    ("link_splits", "def link_splits(word, words, links):\n    return [w for w in DUTCH if w in word]\n", "error", "lesson-names"),
    ("all_splits", _A_START, "empty", None),
    ("all_splits", _A, "correct", None),
    ("all_splits", "def all_splits(word, words):\n    found = [[word]] if word in words else []\n    for i in range(1, len(word)):\n        if word[:i] in words:\n            found = found + [[word[:i]] + rest for rest in all_splits(word[i:], words)]\n    return found\n", "correct", None),
    ("all_splits", _A.replace("        found.append([word])\n", "        found.append(word)\n"), "error", "string"),
    ("all_splits", _A_NOWHOLE, "wrong", "no-whole"),
    ("all_splits", _A_TWO, "wrong", "two-only"),
    ("all_splits", _A_NOLEFT, "wrong", "no-left"),
    ("all_splits", _A_UNCHECKED, "wrong", "unchecked"),
    ("all_splits", _A_WHOLE_ONLY, "wrong", "whole-only"),
    ("all_splits", _A_LAST, "wrong", "whole-last"),
    ("all_splits", "def all_splits(word, words):\n    if word == \"\":\n        return [[]]\n    found = []\n    for i in range(1, len(word) + 1):\n        if word[:i] in words:\n            for rest in all_splits(word[i:], words):\n                found.append([word[:i]] + rest)\n    return found\n", "wrong", "whole-last"),
    ("all_splits", "def all_splits(word, words):\n    return [['voetbalveld'], ['voet', 'bal', 'veld'], ['voetbal', 'veld']]\n", "almost", None),
    ("all_splits", "def all_splits(word, words):\n    return all_splits(word, DUTCH)\n", "error", "lesson-names"),
    ("real", {}, "empty", None),
    ("real", {"veld": "real", "kantoor": "false", "bestand": "false", "arts": "real", "hui": "false",
              "boog": "real", "staart": "false"}, "correct", None),
    ("real", {"veld": "real", "kantoor": "false"}, "wrong", "unfinished"),
    ("real", {"veld": "real", "kantoor": "false", "bestand": "false", "arts": "real", "hui": "real",
              "boog": "real", "staart": "false"}, "wrong", "wrong:hui"),
    ("real", {"kantoor": "real"}, "wrong", "wrong:kantoor"),
]
