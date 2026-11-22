# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-sounds: tell whether two words, written as lists of sounds, form a minimal pair
(minimal_pair), make the last sound of a word voiceless after the Dutch rule of final devoicing (devoice), sort
sentences under phonetics, phonology and spelling, and say why two p-sounds of English count as one phoneme.

The pronunciations are those of the English Wiktionary, as on November 22, 2026 (CC BY-SA 4.0), revision per word in
WRITTEN. The lists of sounds are the course's own reading of them: one value per sound, a length mark with its vowel,
and without the slashes, the stress mark, the dots between syllables and the final (n) that many speakers drop. The
rule follows table 1 of the e-ANS, section 1.3.1 (Geert Booij, version 3.0, September 2020). The pairs of the picture,
the sentences of the sort and the cases of the checks are the course's own."""
import contextlib
import copy
import html
import io
import re

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, with_time_limit

# --- the words -------------------------------------------------------------------------------------------------------

DUTCH = {
    "hond": ["ɦ", "ɔ", "n", "t"], "honden": ["ɦ", "ɔ", "n", "d", "ə"],
    "hand": ["ɦ", "ɑ", "n", "t"], "handen": ["ɦ", "ɑ", "n", "d", "ə"],
    "hoed": ["ɦ", "u", "t"], "hoeden": ["ɦ", "u", "d", "ə"],
    "krab": ["k", "r", "ɑ", "p"], "krabben": ["k", "r", "ɑ", "b", "ə"],
    "rib": ["r", "ɪ", "p"], "ribben": ["r", "ɪ", "b", "ə"],
    "druif": ["d", "r", "œy̯", "f"], "druiven": ["d", "r", "œy̯", "v", "ə"],
    "zaag": ["z", "aː", "x"], "zagen": ["z", "aː", "ɣ", "ə"],
    "kat": ["k", "ɑ", "t"], "katten": ["k", "ɑ", "t", "ə"],
    "boek": ["b", "u", "k"], "boeken": ["b", "u", "k", "ə"],
    "kant": ["k", "ɑ", "n", "t"], "kanten": ["k", "ɑ", "n", "t", "ə"],
    "bed": ["b", "ɛ", "t"], "pet": ["p", "ɛ", "t"],
}
ENGLISH = {
    "pat": ["p", "æ", "t"], "bat": ["b", "æ", "t"], "pin": ["p", "ɪ", "n"], "bin": ["b", "ɪ", "n"],
    "spin": ["s", "p", "ɪ", "n"], "ship": ["ʃ", "ɪ", "p"], "sheep": ["ʃ", "iː", "p"],
    "think": ["θ", "ɪ", "ŋ", "k"], "sink": ["s", "ɪ", "ŋ", "k"],
}
PAIRS = [("hond", "honden"), ("hand", "handen"), ("hoed", "hoeden"), ("krab", "krabben"), ("rib", "ribben"),
         ("druif", "druiven"), ("zaag", "zagen"), ("kat", "katten"), ("boek", "boeken"), ("kant", "kanten")]
VOICELESS = {"b": "p", "d": "t", "v": "f", "z": "s", "ɣ": "x"}
SOUNDS = DUTCH | ENGLISH

# Each word as the English Wiktionary writes it, with the revision of its page (November 22, 2026).
WRITTEN = {
    "hond": ("/ɦɔnt/", 93791251), "honden": ("/ˈɦɔn.də(n)/", 93982130), "hand": ("/ɦɑnt/", 95748189),
    "handen": ("/ˈɦɑn.də(n)/", 88394153), "hoed": ("/ɦut/", 95807846), "hoeden": ("/ˈɦudə(n)/", 81640600),
    "krab": ("/krɑp/", 95544093), "krabben": ("/ˈkrɑ.bə(n)/", 91781003), "rib": ("/rɪp/", 95642078),
    "ribben": ("/ˈrɪ.bə(n)/", 91490856), "druif": ("/drœy̯f/", 90295165), "druiven": ("/ˈdrœy̯və(n)/", 80060361),
    "zaag": ("/zaːx/", 92484842), "zagen": ("/ˈzaːɣə(n)/", 93666090), "kat": ("/kɑt/", 95717727),
    "katten": ("/ˈkɑ.tə(n)/", 83032226), "boek": ("/buk/", 95717565), "boeken": ("/ˈbu.kə(n)/", 85062428),
    "kant": ("/kɑnt/", 95541656), "kanten": ("/ˈkɑn.tə(n)/", 81683004), "bed": ("/bɛt/", 95585847),
    "pet": ("/pɛt/", 95806714), "pat": ("/pæt/", 95707986), "bat": ("/bæt/", 95766845),
    "pin": ("/pɪn/", 95801032), "bin": ("/bɪn/", 95738877), "spin": ("/spɪn/", 95717506),
    "ship": ("/ʃɪp/", 95749278), "sheep": ("/ˈʃiːp/", 95739106), "think": ("/ˈθɪŋk/", 95708400),
    "sink": ("/ˈsɪŋk/", 95432643),
}
# The pairs of the picture under exercise 1.
TRY_PAIRS = [("hond", "hand"), ("hand", "kant"), ("bed", "pet"), ("kat", "kant"), ("kat", "pet"), ("hond", "honden"),
             ("pat", "bat"), ("ship", "sheep"), ("think", "sink"), ("pin", "spin")]
# The rule's five pairs, each with the plural and singular that the e-ANS gives as its example.
RULE_EXAMPLES = [("b", "ribben", "rib"), ("d", "handen", "hand"), ("v", "brieven", "brief"), ("z", "huizen", "huis"),
                 ("ɣ", "zagen", "zaag")]


def minimal_pair(a, b):
    if len(a) != len(b):
        return False
    diff = [i for i in range(len(a)) if a[i] != b[i]]
    return len(diff) == 1


def stem(plural):
    return plural[:len(plural) - 1]


def devoice(sounds, voiceless):
    out = sounds[:len(sounds) - 1]
    last = sounds[len(sounds) - 1]
    if last in voiceless:
        last = voiceless[last]
    out.append(last)
    return out


# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.2rem 0.5rem;text-align:left;vertical-align:middle;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_SOUND = ("display:inline-block;min-width:1.15rem;text-align:center;padding:0 0.22rem;margin:0.08rem 0.08rem;"
          "border:1px solid var(--ea-line);border-radius:4px;font-family:var(--ea-font-code);font-size:0.85rem;"
          "color:var(--ea-ink)")
_MARKED = ";border:1.5px solid var(--ea-ink);font-weight:600"
_LETTER = _SOUND.replace("var(--ea-font-code)", "var(--ea-font-ui)") + ";color:var(--ea-ink-2)"


def _note(text: str) -> str:
    """A line under a picture; a word between asterisks is set in italics, as in the lesson's text."""
    marked = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", html.escape(text))
    return f'<div style="{_UI}">{marked}</div>'


def _bare(text: str) -> str:
    """The same line without its asterisks, for the alt text."""
    return text.replace("*", "")


def _code(value) -> str:
    """A value the way Python shows it, with the strings in single quotes."""
    if isinstance(value, str):
        return "'" + value + "'"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_code(v) for v in value) + "]"
    return repr(value)


def _short(value, limit: int = 40) -> str:
    text = _code(value)
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _chips(sounds, marked=(), off: bool = False) -> str:
    """A list of sounds as small boxes; a marked sound has a dark edge, and a whole list that is off a purple one."""
    boxes = "".join(f'<span style="{_SOUND}{_MARKED if i in marked else ""}">{html.escape(str(s))}</span>'
                    for i, s in enumerate(sounds))
    if off:
        return (f'<span style="display:inline-block;border-radius:6px;padding:0 0.1rem;'
                f'box-shadow:0 0 0 2px var(--ea-loss)">{boxes}</span>')
    return boxes


def _word(word: str, sounds=None, marked=(), off: bool = False) -> str:
    chips = _chips(SOUNDS[word] if sounds is None else sounds, marked, off)
    return (f'<span style="display:inline-block;min-width:4.2rem;font-family:var(--ea-font-prose);font-style:italic;'
            f'color:var(--ea-ink)">{html.escape(word)}</span>{chips}')


def _differ(a: list, b: list) -> list:
    """The places where two lists of the same length have another sound; none when the lengths differ."""
    return [i for i in range(len(a)) if a[i] != b[i]] if len(a) == len(b) else []


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error


def _call(fn, *args):
    """The learner's function within the time limit of a check; _STOPPED when it stops with an error, and the Timeout
    passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(lambda: fn(*(copy.deepcopy(a) for a in args)))
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


_RULE_WORDS = {
    "en": {"title": "Final devoicing in Dutch: five voiced obstruents and their voiceless partners",
           "head": ["Voiced", "Voiceless", "Plural, singular", "Last letter of the singular"],
           "legend": "After tables 1 and 2 of the e-ANS, section 1.3.1 (Geert Booij, 2020), with its example words. "
                     "The spelling keeps b, d and g (the letter for ɣ), but writes f and s for v and z at the end of "
                     "a syllable."},
    "nl": {"title": "Finale verscherping in het Nederlands: vijf stemhebbende obstruenten en hun stemloze partners",
           "head": ["Stemhebbend", "Stemloos", "Meervoud, enkelvoud", "Laatste letter van het enkelvoud"],
           "legend": "Naar tabel 1 en 2 van de e-ANS, paragraaf 1.3.1 (Geert Booij, 2020), met de voorbeeldwoorden "
                     "daaruit. De spelling houdt b, d en g (de letter voor ɣ), maar schrijft aan het eind van een "
                     "lettergreep f en s voor v en z."},
}


def rule_picture(lang: str) -> str:
    w = _RULE_WORDS[lang]
    head = "".join(f'<th style="{_HEAD}">{h}</th>' for h in w["head"])
    rows, alt_rows = "", []
    for voiced, plural, singular in RULE_EXAMPLES:
        letter = singular[len(singular) - 1]
        rows += (f'<tr><td style="{_CELL}"><span style="{_SOUND}">{voiced}</span></td>'
                 f'<td style="{_CELL}"><span style="{_SOUND}{_MARKED}">{VOICELESS[voiced]}</span></td>'
                 f'<td style="{_CELL};font-family:var(--ea-font-prose)"><i>{plural}</i>, <i>{singular}</i></td>'
                 f'<td style="{_CELL}"><span style="{_LETTER}">{letter}</span></td></tr>')
        alt_rows.append(f"{voiced} - {VOICELESS[voiced]}: {plural}, {singular} ({letter})")
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.4rem 0">'
             f'<tr>{head}</tr>{rows}</table></div>')
    alt = f'{w["title"]}. ' + "; ".join(alt_rows) + f". {_bare(w['legend'])}"
    return yours.figure(w["title"], table + _note(w["legend"]), alt=alt)


_MINIMAL_WORDS = {
    "en": {"title": "Your minimal_pair, at work: ten pairs of words",
           "head": ["First word", "Second word", "Your answer"],
           "key": "A sound with a dark edge differs from the sound in the same place of the other word.",
           "fine": "Your answers are right for all {n} pairs. {k} pairs are minimal pairs; the others differ in "
                   "length, or in more than one place.",
           "differs": "Where an answer is in purple, your function differs from the right answer.",
           "error": "Your function stops with an error at *{a}* and *{b}*.",
           "slow": "Your function takes too long."},
    "nl": {"title": "Jouw minimal_pair aan het werk: tien woordparen",
           "head": ["Eerste woord", "Tweede woord", "Jouw antwoord"],
           "key": "Een klank met een donkere rand verschilt van de klank op dezelfde plaats in het andere woord.",
           "fine": "Je antwoorden kloppen bij alle {n} paren. {k} paren zijn minimale paren; de andere verschillen "
                   "in lengte, of op meer dan één plaats.",
           "differs": "Waar een antwoord paars is, wijkt je functie af van het goede antwoord.",
           "error": "Je functie stopt met een fout bij *{a}* en *{b}*.",
           "slow": "Je functie doet er te lang over."},
}


def minimal_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    lang, w = seen.lang, _MINIMAL_WORDS[seen.lang]
    rows, alt_rows, end, all_right = "", [], "", True
    for a, b in TRY_PAIRS:
        try:
            got = _call(fn, SOUNDS[a], SOUNDS[b])
        except Timeout:
            end = w["slow"]
            break
        if got is _STOPPED:
            end = w["error"].format(a=a, b=b)
            break
        want = minimal_pair(SOUNDS[a], SOUNDS[b])
        off = not (type(got) is bool and got == want)
        all_right = all_right and not off
        marked = _differ(SOUNDS[a], SOUNDS[b])
        color = "var(--ea-loss-text);font-weight:600" if off else "var(--ea-ink)"
        rows += (f'<tr><td style="{_CELL}">{_word(a, marked=marked)}</td><td style="{_CELL}">{_word(b, marked=marked)}'
                 f'</td><td style="{_CELL};font-family:var(--ea-font-code);color:{color}">'
                 f'{html.escape(_short(got, 16))}</td></tr>')
        alt_rows.append(f"{a} {' '.join(SOUNDS[a])}, {b} {' '.join(SOUNDS[b])}: {_short(got, 16)}")
    head = "".join(f'<th style="{_HEAD}">{h}</th>' for h in w["head"])
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.4rem 0">'
             f'<tr>{head}</tr>{rows}</table></div>')
    k = sum(minimal_pair(SOUNDS[a], SOUNDS[b]) for a, b in TRY_PAIRS)
    after = end or (w["fine"].format(n=len(TRY_PAIRS), k=k) if all_right else w["differs"])
    alt = f'{w["title"]}. ' + "; ".join(alt_rows) + f". {_bare(w['key'])} {_bare(after)}"
    return yours.figure(w["title"], table + _note(w["key"]) + _note(after), alt=alt)


_DEVOICE_WORDS = {
    "en": {"title": "Your devoice, at work: from the plural to the singular",
           "head": ["Plural", "Without the ending", "Your devoice", "Singular in the dictionary"],
           "key": "A sound with a dark edge is one your function changed.",
           "fine": "Your devoice gives the dictionary's singular for all {n} words: {k} by making the last sound "
                   "voiceless, {m} because they end in a voiceless sound already.",
           "differs": "Where a result has a purple edge, it differs from the dictionary's singular.",
           "error": "Your function stops with an error at *{word}*.",
           "slow": "Your function takes too long.",
           "not_list": "For *{word}*, your devoice gives {got}, not a list of sounds."},
    "nl": {"title": "Jouw devoice aan het werk: van meervoud naar enkelvoud",
           "head": ["Meervoud", "Zonder de uitgang", "Jouw devoice", "Enkelvoud in het woordenboek"],
           "key": "Een klank met een donkere rand is door je functie veranderd.",
           "fine": "Je devoice geeft bij alle {n} woorden het enkelvoud van het woordenboek: {k} door de laatste "
                   "klank stemloos te maken, {m} omdat die woorden al op een stemloze klank eindigen.",
           "differs": "Waar een uitkomst een paarse rand heeft, wijkt die af van het enkelvoud in het woordenboek.",
           "error": "Je functie stopt met een fout bij *{word}*.",
           "slow": "Je functie doet er te lang over.",
           "not_list": "Bij *{word}* geeft je devoice {got}, geen lijst klanken."},
}


def devoice_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    lang, w = seen.lang, _DEVOICE_WORDS[seen.lang]
    rows, alt_rows, end, all_right = "", [], "", True
    for singular, plural in PAIRS:
        base = stem(DUTCH[plural])
        try:
            got = _call(fn, base, VOICELESS)
        except Timeout:
            end = w["slow"]
            break
        if got is _STOPPED:
            end = w["error"].format(word=plural)
            break
        if not isinstance(got, list):
            end = w["not_list"].format(word=plural, got=_short(got))
            break
        off = got != DUTCH[singular]
        all_right = all_right and not off
        changed = [i for i in range(len(got)) if i >= len(base) or got[i] != base[i]]
        rows += (f'<tr><td style="{_CELL}">{_word(plural)}</td><td style="{_CELL}">{_chips(base)}</td>'
                 f'<td style="{_CELL}">{_chips(got, changed, off)}</td><td style="{_CELL}">{_word(singular)}</td></tr>')
        alt_rows.append(f"{plural} {' '.join(DUTCH[plural])}, {' '.join(base)}, {_short(got)}, "
                        f"{singular} {' '.join(DUTCH[singular])}")
    head = "".join(f'<th style="{_HEAD}">{h}</th>' for h in w["head"])
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.4rem 0">'
             f'<tr>{head}</tr>{rows}</table></div>')
    k = sum(devoice(stem(DUTCH[p]), VOICELESS) != stem(DUTCH[p]) for _, p in PAIRS)
    after = end or (w["fine"].format(n=len(PAIRS), k=k, m=len(PAIRS) - k) if all_right else w["differs"])
    alt = f'{w["title"]}. ' + "; ".join(alt_rows) + f". {_bare(w['key'])} {_bare(after)}"
    return yours.figure(w["title"], table + _note(w["key"]) + _note(after), alt=alt)


_DEMO_WORDS = {
    "en": {"title": "{word}: letters and sounds",
           "rows": ["Letters", "As Wiktionary writes it", "Sounds, as a list"],
           "count": "{a} letters, {b} sounds. {lang}, after the English Wiktionary (revision {rev}).",
           "dutch": "Dutch", "english": "English",
           "key": "The list leaves out the marks of Wiktionary's form: the slashes, ˈ before the syllable with the "
                  "stress, a dot between syllables, and (n) for an n that many speakers leave out."},
    "nl": {"title": "{word}: letters en klanken",
           "rows": ["Letters", "Zoals Wiktionary het schrijft", "Klanken, als lijst"],
           "count": "{a} letters, {b} klanken. {lang}, naar de Engelse Wiktionary (revisie {rev}).",
           "dutch": "Nederlands", "english": "Engels",
           "key": "De lijst laat de tekens van de vorm uit Wiktionary weg: de schuine strepen, ˈ voor de lettergreep "
                  "met de klemtoon, een punt tussen lettergrepen, en (n) voor een n die veel sprekers weglaten."},
}


def demo_picture(word: str, lang: str) -> str:
    w = _DEMO_WORDS[lang]
    word = word if word in SOUNDS else "krabben"
    letters = "".join(f'<span style="{_LETTER}">{html.escape(c)}</span>' for c in word)
    form, rev = WRITTEN[word]
    cells = [letters, f'<span style="font-family:var(--ea-font-code);font-size:0.95rem">{html.escape(form)}</span>',
             _chips(SOUNDS[word])]
    rows = "".join(f'<tr><td style="{_HEAD}">{name}</td><td style="{_CELL}">{cell}</td></tr>'
                   for name, cell in zip(w["rows"], cells))
    table = f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.4rem 0">{rows}</table></div>'
    count = w["count"].format(a=len(word), b=len(SOUNDS[word]), rev=rev,
                              lang=w["dutch"] if word in DUTCH else w["english"])
    title = w["title"].format(word=word)
    alt = f"{title}. {' '.join(word)}; {form}; {' '.join(SOUNDS[word])}. {count} {w['key']}"
    return yours.figure(title, table + _note(count) + _note(w["key"]), alt=alt)


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["minimal_pair", "devoice", "fields", "allophone"]


class Named:
    """A value the check gives the learner's function, shown in a message by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


def _dutch(word: str) -> Named:
    return Named(f'DUTCH["{word}"]', DUTCH[word])


def _english(word: str) -> Named:
    return Named(f'ENGLISH["{word}"]', ENGLISH[word])


_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: "..." in "\n".join(
                    line.split("#")[0] for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# exercise 1: minimal_pair; a word shorter than the other comes before a longer one, so that leaving out the check on
# the lengths first gives a wrong answer, not an error
MINIMAL_CASES = [
    (_dutch("hond"), _dutch("hand")),
    (_dutch("kat"), _dutch("kant")),
    (_dutch("kat"), _dutch("pet")),
    (_dutch("bed"), _dutch("pet")),
    (_english("ship"), _english("sheep")),
    (_dutch("kat"), _dutch("kat")),
    (_english("pin"), _english("spin")),
    (_dutch("honden"), _dutch("hond")),
    (["a"], ["o"]),
    ([], []),
]
_M_START = "def minimal_pair(a, b):\n    ...\n"
_M = ("def minimal_pair(a, b):\n"
      "    if len(a) != len(b):\n"
      "        return False\n"
      "    diff = [i for i in range(len(a)) if a[i] != b[i]]\n"
      "    return len(diff) == 1\n")
_M_NO_LENGTH = _M.replace("    if len(a) != len(b):\n        return False\n", "")
_M_ANY = _M.replace("len(diff) == 1", "len(diff) >= 1")
_M_AT_MOST = _M.replace("len(diff) == 1", "len(diff) <= 1")
_M_SAME = _M.replace("a[i] != b[i]", "a[i] == b[i]")


def _not_a_bool(got, text) -> bool:
    return not isinstance(got, bool)


# exercise 2: devoice
DEVOICE_CASES = [
    (Named('stem(DUTCH["honden"])', stem(DUTCH["honden"])), Named("VOICELESS", VOICELESS)),
    (Named('stem(DUTCH["druiven"])', stem(DUTCH["druiven"])), Named("VOICELESS", VOICELESS)),
    (_dutch("kat"), Named("VOICELESS", VOICELESS)),
    (["z", "aː", "ɣ"], Named("VOICELESS", VOICELESS)),
    (["b"], Named("VOICELESS", VOICELESS)),
    (["d", "ɔ", "n"], Named("VOICELESS", VOICELESS)),
    (["b", "ɛ", "d"], {"d": "t"}),
    (["k", "r", "ɑ", "b"], {"d": "t"}),
]
_D_START = "def devoice(sounds, voiceless):\n    ...\n"
_D = ("def devoice(sounds, voiceless):\n"
      "    out = sounds[:len(sounds) - 1]\n"
      "    last = sounds[len(sounds) - 1]\n"
      "    if last in voiceless:\n"
      "        last = voiceless[last]\n"
      "    out.append(last)\n"
      "    return out\n")
_D_EVERY = ("def devoice(sounds, voiceless):\n"
            "    out = []\n"
            "    for s in sounds:\n"
            "        if s in voiceless:\n"
            "            out.append(voiceless[s])\n"
            "        else:\n"
            "            out.append(s)\n"
            "    return out\n")
_D_DROPS = ("def devoice(sounds, voiceless):\n"
            "    out = sounds[:len(sounds) - 1]\n"
            "    last = sounds[len(sounds) - 1]\n"
            "    if last in voiceless:\n"
            "        out.append(voiceless[last])\n"
            "    return out\n")
_D_NO_IF = ("def devoice(sounds, voiceless):\n"
            "    out = sounds[:len(sounds) - 1]\n"
            "    out.append(voiceless[sounds[len(sounds) - 1]])\n"
            "    return out\n")
_D_EARLY = _D.replace("    out.append(last)\n    return out\n", "        out.append(last)\n    return out\n")


def _a_string(got, text) -> bool:
    return isinstance(got, str)


EXERCISES = {
    "minimal_pair": FunctionBox(
        name="minimal_pair",
        picture=minimal_picture,
        function="minimal_pair",
        start=_M_START,
        solution=_M,
        visible=MINIMAL_CASES[:1],
        hidden=MINIMAL_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("not-a-bool", matches=_not_a_bool,
                    en="`{call}` should give `True` or `False`: do the two words form a minimal pair or not? Which comparison gives `True` or `False`?",
                    nl="`{call}` moet `True` of `False` geven: vormen de twee woorden een minimaal paar of niet? Welke vergelijking geeft `True` of `False`?"),
            Mistake("no-length", wrong=_M_NO_LENGTH,
                    en="`{call}` gives {got}, and it should give {want}: one word has a sound more than the other. Two words of different lengths are no minimal pair, as *kat* and *kant* show. What should your function give back for them, before it compares any sounds?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het ene woord heeft een klank meer dan het andere. Twee woorden van verschillende lengte zijn geen minimaal paar, zoals *kat* en *kant* laten zien. Wat moet je functie voor zulke woorden teruggeven, voordat de functie klanken vergelijkt?"),
            Mistake("any", wrong=_M_ANY,
                    en="`{call}` gives {got}, and it should give {want}: your function also says `True` when two words differ in more than one place, such as *kat* and *pet*. How many places may differ in a minimal pair?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie zegt ook `True` als twee woorden op meer dan één plaats verschillen, zoals *kat* en *pet*. Op hoeveel plaatsen mag een minimaal paar verschillen?"),
            Mistake("at-most", wrong=_M_AT_MOST,
                    en="`{call}` gives {got}, and it should give {want}: your function says `True` for a word and itself. A word is no minimal pair with itself. How many places should differ?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie zegt `True` bij een woord en zichzelf. Een woord vormt geen minimaal paar met zichzelf. Op hoeveel plaatsen moeten de woorden verschillen?"),
            Mistake("same", wrong=_M_SAME,
                    en="`{call}` gives {got}, and it should give {want}, though the two words have the same length and only their vowels differ. Which places does your list collect?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn, terwijl de twee woorden even lang zijn en alleen hun klinkers verschillen. Welke plaatsen verzamelt je lijst?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back `True` when the lists `a` and `b` form a minimal pair, and `False` otherwise.",
                      "nl": "Vervang de `...`: geef `True` terug als de lijsten `a` en `b` een minimaal paar vormen, en anders `False`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Do the two words have the same number of sounds? And how many of their places hold another sound?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Hebben de twee woorden evenveel klanken? En op hoeveel plaatsen staat er een andere klank?"},
            "almost": {"en": "Right for *hond* and *hand*. But `{call}` gives {got}, and it should give {want}. Does your function work for any two lists of sounds?",
                       "nl": "Goed voor *hond* en *hand*. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke twee lijsten klanken, ook van verschillende lengte?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back: it has a `return`, but Python does not get to it. Does every way through your function end with a `return`?",
                          "nl": "Met `{call}` geeft je functie niets terug: er staat een `return` in, maar Python komt er niet aan toe. Eindigt elke weg door je functie met een `return`?"},
            "correct": {"en": "Your `minimal_pair` is right in all {n} cases. Look at the picture: ten pairs of words, with your answer next to each.",
                        "nl": "Je `minimal_pair` klopt in alle {n} gevallen. Kijk naar de afbeelding: tien woordparen, met jouw antwoord bij elk paar."},
        },
    ),
    "devoice": FunctionBox(
        name="devoice",
        picture=devoice_picture,
        function="devoice",
        start=_D_START,
        solution=_D,
        visible=DEVOICE_CASES[:1],
        hidden=DEVOICE_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("a-string", matches=_a_string,
                    en="`{call}` gives {got}: one sound, not the whole list. Give back all the sounds of the word, with only the last one changed where the rule says so.",
                    nl="`{call}` geeft {got}: één klank, niet de hele lijst. Geef alle klanken van het woord terug, met alleen de laatste veranderd waar de regel dat zegt."),
            Mistake("every", wrong=_D_EVERY,
                    en="`{call}` gives {got}, and it should give {want}: your function makes every voiced sound voiceless, also the one at the start. The rule only works at the end. Which sound of the list should your function look at?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie maakt elke stemhebbende klank stemloos, ook die aan het begin. De regel werkt alleen aan het eind. Naar welke klank van de lijst moet je functie kijken?"),
            Mistake("drops", wrong=_D_DROPS,
                    en="`{call}` gives {got}, and it should give {want}: the last sound disappears when it is not a key of `voiceless`. Such a sound stays as it is. Does your function add the last sound to `out` in both cases, after the `if`?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de laatste klank verdwijnt als die geen sleutel van `voiceless` is. Zo'n klank blijft zoals die is. Voegt je functie de laatste klank in beide gevallen toe aan `out`, na de `if`?"),
            Mistake("no-if", error=lambda e: isinstance(e, KeyError),
                    en="With `{call}`, your function stops with a `KeyError`: the last sound is not a key of `voiceless`, so the dictionary has nothing to give. Look it up only when it is a key, with an `if`.",
                    nl="Met `{call}` stopt je functie met een `KeyError`: de laatste klank is geen sleutel van `voiceless`, dus de dictionary heeft niets te geven. Zoek de klank alleen op als die een sleutel is, met een `if`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back the sounds of `sounds`, with the last one made voiceless when it is a key of `voiceless`.",
                      "nl": "Vervang de `...`: geef de klanken van `sounds` terug, met de laatste stemloos gemaakt als die een sleutel van `voiceless` is."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which sound may change, and which sound takes its place?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Welke klank mag veranderen, en welke klank komt ervoor in de plaats?"},
            "almost": {"en": "Right for *honden*. But `{call}` gives {got}, and it should give {want}. Does your function work for any list of sounds and any dictionary of pairs?",
                       "nl": "Goed voor *honden*. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke lijst klanken en elke dictionary met paren?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back: it has a `return`, but Python does not get to it. Is the `return` the last line, outside the `if`?",
                          "nl": "Met `{call}` geeft je functie niets terug: er staat een `return` in, maar Python komt er niet aan toe. Is de `return` de laatste regel, buiten de `if`?"},
            "correct": {"en": "Your `devoice` is right in all {n} cases. Look at the picture: ten plurals, without their ending and through your rule, next to the singular in the dictionary.",
                        "nl": "Je `devoice` klopt in alle {n} gevallen. Kijk naar de afbeelding: tien meervouden, zonder hun uitgang en na jouw regel, naast het enkelvoud uit het woordenboek."},
        },
    ),
    "fields": Sort(
        name="fields",
        groups={
            "mouth": {"en": "How the body makes a sound",
                      "nl": "Hoe je lichaam een klank maakt"},
            "system": {"en": "Which sounds a language uses to tell words apart",
                       "nl": "Welke klanken een taal gebruikt om woorden uit elkaar te houden"},
            "letters": {"en": "How a language writes its sounds down",
                        "nl": "Hoe een taal de klanken opschrijft"},
        },
        items={
            "m": {"en": "**For an m, the lips close and the air goes out through the nose.**",
                  "nl": "**Bij een m gaan de lippen dicht en gaat de lucht door de neus naar buiten.**"},
            "th": {"en": "**For the th of *think*, the tip of the tongue touches the upper teeth.**",
                   "nl": "**Bij de th van het Engelse *think* raakt het puntje van de tong de bovenste tanden.**"},
            "takdak": {"en": "**In Dutch, swapping the t of *tak* (branch) for a d makes another word: *dak* (roof).**",
                       "nl": "**In het Nederlands wordt *tak* een ander woord, *dak*, als je de t door een d vervangt.**"},
            "count": {"en": "**Standard Dutch has 36 sounds that can change a word, by the count of the e-ANS.**",
                      "nl": "**Het Standaardnederlands heeft 36 klanken die een woord kunnen veranderen, volgens de telling van de e-ANS.**"},
            "meet": {"en": "**English writes the same vowel as *ee* in *meet* and as *ea* in *meat*.**",
                     "nl": "**Het Engels schrijft dezelfde klinker als *ee* in *meet* en als *ea* in *meat*.**"},
            "hoed": {"en": "**Dutch writes *hoed* (hat) with a d, though it ends in a t-sound.**",
                     "nl": "**Het Nederlands schrijft *hoed* met een d, al eindigt het woord op een t-klank.**"},
        },
        solution={"m": "mouth", "th": "mouth", "takdak": "system", "count": "system", "meet": "letters",
                  "hoed": "letters"},
        nudge={
            "m": {"en": "Would this be true of an m in any language?",
                  "nl": "Geldt dit voor een m in elke taal?"},
            "th": {"en": "Does this sentence say anything about words, or about letters?",
                   "nl": "Zegt deze zin iets over woorden, of over letters?"},
            "takdak": {"en": "What happens to the word here?",
                       "nl": "Wat gebeurt er hier met het woord?"},
            "count": {"en": "What does a sound have to do to be counted here?",
                      "nl": "Wat moet een klank doen om hier meegeteld te worden?"},
            "meet": {"en": "Do the two vowels sound different, or only look different?",
                     "nl": "Klinken de twee klinkers anders, of zien alleen de letters er anders uit?"},
            "hoed": {"en": "Is the d here something you hear, or something you see?",
                     "nl": "Is de d hier iets wat je hoort, of iets wat je ziet?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the sentences again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de zinnen opnieuw uit."},
            "empty": {"en": "Put each bold sentence under the group it is about.",
                      "nl": "Zet elke vetgedrukte zin bij de groep waar die zin over gaat."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs under another group.",
                      "nl": "{item} hoort bij een andere groep."},
            "correct": {"en": "All {n} are right. The first group is phonetics, the second phonology, the third spelling. The count of 36 is an analysis: in the PHOIBLE database, the course counts 37 to 43 sounds in the three inventories of Standard Dutch, because linguists count in different ways.",
                        "nl": "Alle {n} kloppen. De eerste groep is fonetiek, de tweede fonologie, de derde spelling. De 36 is een analyse: in de database PHOIBLE telt de course 37 tot 43 klanken in de drie inventarissen van het Standaardnederlands, omdat taalkundigen op verschillende manieren tellen."},
        },
    ),
}

CASES = [
    ("minimal_pair", _M_START, "empty", None),
    ("minimal_pair", _M, "correct", None),
    ("minimal_pair", "def minimal_pair(a, b):\n    if len(a) != len(b):\n        return False\n    n = 0\n    for i in range(len(a)):\n        if a[i] != b[i]:\n            n = n + 1\n    return n == 1\n", "correct", None),
    ("minimal_pair", "def minimal_pair(a, b):\n    return len(a) == len(b) and len([i for i in range(len(a)) if a[i] != b[i]]) == 1\n", "correct", None),
    ("minimal_pair", _M_NO_LENGTH, "wrong", "no-length"),
    ("minimal_pair", _M_ANY, "wrong", "any"),
    ("minimal_pair", _M_AT_MOST, "wrong", "at-most"),
    ("minimal_pair", _M_SAME, "wrong", "same"),
    ("minimal_pair", _M.replace("len(diff) == 1", "len(diff)"), "wrong", "not-a-bool"),
    ("minimal_pair", _M.replace("    return len(diff) == 1\n", "    len(diff) == 1\n"), "wrong", "unreached"),
    ("minimal_pair", "def minimal_pair(a, b):\n    return True\n", "almost", None),
    ("devoice", _D_START, "empty", None),
    ("devoice", _D, "correct", None),
    ("devoice", "def devoice(sounds, voiceless):\n    last = sounds[len(sounds) - 1]\n    if last in voiceless:\n        return sounds[:len(sounds) - 1] + [voiceless[last]]\n    return sounds\n", "correct", None),
    ("devoice", "def devoice(sounds, voiceless):\n    n = len(sounds) - 1\n    if sounds[n] in voiceless:\n        sounds[n] = voiceless[sounds[n]]\n    return sounds\n", "correct", None),
    ("devoice", _D_EVERY, "wrong", "every"),
    ("devoice", _D_DROPS, "wrong", "drops"),
    ("devoice", _D_NO_IF, "error", "no-if"),
    ("devoice", _D_EARLY, "wrong", "drops"),
    ("devoice", "def devoice(sounds, voiceless):\n    last = sounds[len(sounds) - 1]\n    if last in voiceless:\n        return voiceless[last]\n    return last\n", "wrong", "a-string"),
    ("devoice", _D.replace("    return out\n", ""), "wrong", "no-return"),
    ("devoice", "def devoice(sounds, voiceless):\n    return " + repr(devoice(stem(DUTCH["honden"]), VOICELESS)) + "\n", "almost", None),
    ("fields", {}, "empty", None),
    ("fields", {"m": "mouth", "th": "mouth", "takdak": "system", "count": "system", "meet": "letters",
                "hoed": "letters"}, "correct", None),
    ("fields", {"m": "mouth", "takdak": "system"}, "wrong", "unfinished"),
    ("fields", {"hoed": "system"}, "wrong", "wrong:hoed"),
    ("fields", {"m": "mouth", "th": "mouth", "takdak": "system", "count": "letters", "meet": "letters",
                "hoed": "letters"}, "wrong", "wrong:count"),
]
