# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-simplify: readability formulas, which score a text by the length of its sentences and its
words. syllables counts the syllables of a word by its groups of vowels; counts adds up the words, the sentences and the
syllables of a text cut into sentences; flesch is Flesch's reading ease (1948). Douma's version for Dutch (1960) has
smaller weights.

The texts are the eight books of everything_ai.books and the Dutch sentences of Tatoeba; the letters about the green bin
and the words counted by hand are the course's own. Every number in the text and the pictures is computed by the course
(tests/test_nlp_simplify.py computes them again)."""
import html
import importlib.util
import math
import re

from everything_ai import books, paths, yours
from everything_ai.check import FunctionBox, Mistake, Sort

LANGS = ("en", "nl")
_KEY = "nlp-simplify"


def _summarize():
    spec = importlib.util.spec_from_file_location("everything_ai_checks_nlp_summarize_for_simplify",
                                                  paths.LESSONS / "nlp-summarize" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _summarize()   # the parts of the pictures and the exercises

# --- cutting a text into sentences and words --------------------------------------------------------------------------

# The list of *Cutting text into sentences and words*, with English abbreviations and a few more Dutch ones added.
ABBREVIATIONS = frozenset({
    "a.s.", "bijv.", "blz.", "bv.", "ca.", "d.w.z.", "dhr.", "dr.", "drs.", "e.d.", "enz.", "etc.", "i.p.v.", "incl.",
    "ir.", "jl.", "m.b.t.", "mevr.", "mld.", "mln.", "mr.", "n.v.", "nr.", "o.a.", "o.m.", "pct.", "prof.", "resp.",
    "st.", "t.o.v.",
    "mrs.", "messrs.", "mme.", "mlle.", "vs.", "e.g.", "i.e.", "no.", "jhr.", "mej.",
})
_CLOSERS = "\"'”’)]_*»"   # what may follow the mark at the end of a sentence: quotation marks, brackets, italics


def ends_sentence(word: str, abbreviations=ABBREVIATIONS) -> bool:
    """As in *Cutting text into sentences and words*: True when a word cut at spaces ends with . ? or ! and is neither
    an abbreviation nor an initial such as J.; the course looks past closing quotation marks and brackets first."""
    word = word.rstrip(_CLOSERS)
    if word.lower() in abbreviations or re.search(r"^[A-Z]\.$", word):
        return False
    return re.search(r"[.?!]$", word) is not None


def _pieces(text: str) -> list:
    """A text cut into sentences, each as (start, end, words): where it starts and ends in the text, and its words."""
    out, words_, start, end = [], [], None, None
    for m in re.finditer(r"\S+", text):
        found = books.words(m.group())
        if found and start is None:
            start = m.start()
        words_ += found
        end = m.end()
        if words_ and ends_sentence(m.group()):
            out.append((start, end, words_))
            words_, start = [], None
    if words_:
        out.append((start, end, words_))
    return out


def sentences(text: str) -> list:
    """A text cut into sentences, each a list of its words: runs of letters, as everything_ai.books finds them."""
    return [ws for _, _, ws in _pieces(text if isinstance(text, str) else "")]


# --- the three functions, as the course writes them ---------------------------------------------------------------------

def syllables(word):
    """The course's solution of Exercise 1: the groups of vowels in a row in the word, in small letters; at least 1."""
    groups = re.findall(r"[aeiouy]+", word.lower())
    return max(1, len(groups))


def counts(sentences):
    """The course's solution of Exercise 2: the words, the sentences and the syllables of a text cut into sentences."""
    n_words = 0
    n_syllables = 0
    for sentence in sentences:
        for word in sentence:
            n_words = n_words + 1
            n_syllables = n_syllables + syllables(word)
    return n_words, len(sentences), n_syllables


def flesch(words, sentences, syllables):
    """The course's solution of Exercise 3: Flesch's reading ease, written per word instead of per 100 words."""
    return 206.835 - 1.015 * words / sentences - 84.6 * syllables / words


def douma(words: int, sentences: int, syllables: int) -> float:
    """Douma's reading ease for Dutch (1960), written per word: RE = 206,84 - 0,77 x woordlengte - 0,93 x zinslengte,
    where woordlengte is the syllables per 100 words."""
    return 206.84 - 0.93 * words / sentences - 77 * syllables / words


FORMULA = {"en": flesch, "nl": douma}   # the formula each lesson scores its own language with

# Flesch's table (1948, Table 5): from which score on each band starts, from the top.
BANDS = ((90, "very-easy"), (80, "easy"), (70, "fairly-easy"), (60, "standard"), (50, "fairly-difficult"),
         (30, "difficult"), (-math.inf, "very-difficult"))
# Douma's table (1960, p. 28): the same seven bands, but fairly difficult starts at 45 instead of 50.
DOUMA_BANDS = ((90, "very-easy"), (80, "easy"), (70, "fairly-easy"), (60, "standard"), (45, "fairly-difficult"),
               (30, "difficult"), (-math.inf, "very-difficult"))
TABLE = {"flesch": BANDS, "douma": DOUMA_BANDS}   # the table that goes with each formula, by its name
BAND_NAMES = {
    "en": {"very-easy": "very easy", "easy": "easy", "fairly-easy": "fairly easy", "standard": "standard",
           "fairly-difficult": "fairly difficult", "difficult": "difficult", "very-difficult": "very difficult"},
    "nl": {"very-easy": "zeer gemakkelijk", "easy": "gemakkelijk", "fairly-easy": "vrij gemakkelijk",
           "standard": "standaard", "fairly-difficult": "vrij moeilijk", "difficult": "moeilijk",
           "very-difficult": "zeer moeilijk"},
}


def band(score: float, table=BANDS) -> str:
    """The band of a table, Flesch's unless another is given, a score falls in; below 0 counts as very difficult and
    above 100 as very easy."""
    for start, key in table:
        if score >= start:
            return key
    return "very-difficult"


def add_counts(parts) -> tuple:
    """Counts of several texts together."""
    w = s = y = 0
    for a, b, c in parts:
        w, s, y = w + a, s + b, y + c
    return w, s, y


# --- the texts: eight books and today's Dutch sentences ----------------------------------------------------------------

TEXTS = ("alice", "pride", "holmes", "frankenstein", "moby-dick", "eline-vere", "havelaar", "camera-obscura", "tatoeba")
TEXT_LANG = {key: ("en" if i < 5 else "nl") for i, key in enumerate(TEXTS)}
TEXT_NAMES = {
    "en": {"alice": "Alice's Adventures in Wonderland", "pride": "Pride and Prejudice",
           "holmes": "The Adventures of Sherlock Holmes", "frankenstein": "Frankenstein", "moby-dick": "Moby-Dick",
           "eline-vere": "Eline Vere", "havelaar": "Max Havelaar", "camera-obscura": "Camera Obscura",
           "tatoeba": "Dutch sentences of Tatoeba"},
    "nl": {"alice": "Alice's Adventures in Wonderland", "pride": "Pride and Prejudice",
           "holmes": "The Adventures of Sherlock Holmes", "frankenstein": "Frankenstein", "moby-dick": "Moby-Dick",
           "eline-vere": "Eline Vere", "havelaar": "Max Havelaar", "camera-obscura": "Camera Obscura",
           "tatoeba": "Nederlandse zinnen van Tatoeba"},
}


def text_counts() -> dict:
    """The words, sentences and syllables of each book and of all Dutch sentences of Tatoeba, each sentence of Tatoeba
    cut on its own. It takes a few seconds, so the pictures use TEXT_COUNTS, which the tests compare with it."""
    out = {b["key"]: counts(sentences(b["text"])) for b in books.books()}
    out["tatoeba"] = add_counts(counts(sentences(t)) for _, t in books.dutch_sentences())
    return {key: out[key] for key in TEXTS}


TEXT_COUNTS = {   # text_counts(), written out
    "alice": (26776, 1634, 37747), "pride": (127858, 6156, 200528), "holmes": (105286, 6796, 152959),
    "frankenstein": (75201, 3328, 118778), "moby-dick": (216543, 10229, 328650), "eline-vere": (182872, 13036, 276472),
    "havelaar": (125009, 6863, 196963), "camera-obscura": (161370, 8056, 250591), "tatoeba": (1301039, 204977, 1893292),
}


def sentences_without_list(text: str) -> list:
    """A text cut into sentences without the list of abbreviations: every word ending with . ? or ! ends one, unless
    it is an initial such as J."""
    out, words_ = [], []
    for word in text.split():
        words_ += books.words(word)
        if words_ and ends_sentence(word, abbreviations=frozenset()):
            out.append(words_)
            words_ = []
    if words_:
        out.append(words_)
    return out


# --- the letters about the green bin ----------------------------------------------------------------------------------

LETTERS = {
    "en": {"hard": "In accordance with the regulations concerning the collection of household waste, residents are hereby "
                   "informed that, commencing on the first Monday of the forthcoming month, containers designated for "
                   "organic waste will be emptied exclusively on alternate weeks, unless circumstances necessitate an "
                   "alternative arrangement. Residents who require additional information regarding this modification "
                   "are requested to contact the municipal service department.",
           "plain": "From next month, we empty the green bin once every two weeks. We start on the first Monday of the "
                    "month. Do you have a question? Call the city service desk."},
    "nl": {"hard": "Overeenkomstig de verordening inzake de inzameling van huishoudelijk afval worden bewoners er hierbij "
                   "van in kennis gesteld dat, met ingang van de eerste maandag van de eerstvolgende maand, de "
                   "containers die bestemd zijn voor groente-, fruit- en tuinafval uitsluitend om de week zullen worden "
                   "geledigd, tenzij de omstandigheden een andere regeling noodzakelijk maken. Bewoners die aanvullende "
                   "informatie wensen met betrekking tot deze wijziging, worden verzocht contact op te nemen met de "
                   "gemeentelijke dienstverlening.",
            "plain": "Vanaf volgende maand legen we de groene container eens in de twee weken. We beginnen op de eerste "
                     "maandag van de maand. Heeft u een vraag? Bel dan de servicebalie van de gemeente."},
}
LEFT_OUT = {"en": "unless circumstances necessitate an alternative arrangement",   # what the plain letter leaves out
            "nl": "tenzij de omstandigheden een andere regeling noodzakelijk maken"}
LONG_WORD = 3   # from how many syllables on a picture puts a word on a background


def letter_counts() -> dict:
    """The counts of the four letters, by (language, "hard" or "plain")."""
    return {(lang, kind): counts(sentences(LETTERS[lang][kind])) for lang in LANGS for kind in ("hard", "plain")}


# --- words counted by hand -----------------------------------------------------------------------------------------------

# Each word with the number of syllables a reader hears, counted by hand by the course; the last four of each language
# are words where the vowel rule gives another number.
BY_HAND = {
    "en": (("simple", 2), ("readability", 5), ("information", 4), ("question", 2), ("bin", 1),
           ("make", 1), ("created", 3), ("service", 2), ("rhythm", 2)),
    "nl": (("leesbaarheid", 3), ("bijenkast", 3), ("huishoudelijk", 4), ("container", 3), ("maandag", 2),
           ("ideeën", 3), ("museum", 3), ("bioscoop", 3), ("theater", 3)),
}


# --- small parts of the pictures (those of nlp-summarize) -------------------------------------------------------------

_UI, _LABEL, _CELL, _HEAD, _NUMBER = S._UI, S._LABEL, S._CELL, S._HEAD, S._NUMBER
_DATA, _FAINT, _OUT, _PRED, _LOSS = S._DATA, S._FAINT, S._OUT, S._PRED, S._LOSS
_CHIP, _CHIP_PRED, _CHIP_DATA, _CHIP_LOSS = S._CHIP, S._CHIP_PRED, S._CHIP_DATA, S._CHIP_LOSS
_note, _colors, _label, _num, _dec, _pct = S._note, S._colors, S._label, S._num, S._dec, S._pct
_cell, _table, _bar, _short, _is_number = S._cell, S._table, S._bar, S._short, S._is_number
_fn, _call, _empty, _SLOW, _STOPPED = S._fn, S._call, S._empty, S._SLOW, S._STOPPED
_WHY_EMPTY, _WORD = S._WHY_EMPTY, S._WORD
TEXT_CAP = 3000   # characters of the text in the try-it box that the course scores


def _score(v, lang: str) -> str:
    """A score as the pictures write it: one decimal; anything else a learner's function gives, as it is."""
    return _dec(v, lang, 1) if _is_number(v) and abs(v) < 1e9 else html.escape(_short(v))


def _whole(v) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return abs(v) < 10**9
    return isinstance(v, float) and math.isfinite(v) and v.is_integer() and abs(v) < 1e9


def _is_counts(v) -> bool:
    return isinstance(v, (tuple, list)) and len(v) == 3 and all(_whole(x) for x in v)


def _close(a, b) -> bool:
    return _is_number(a) and _is_number(b) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def _chipped(part: str, chip: str, style: str = "") -> str:
    """A piece of text with each word of LONG_WORD syllables or more on a background, the whole in `style`."""
    out, pos = [], 0
    for m in _WORD.finditer(part):
        out.append(html.escape(part[pos:m.start()]))
        w = html.escape(m.group())
        out.append(f'<span style="{chip}">{w}</span>' if syllables(m.group()) >= LONG_WORD else w)
        pos = m.end()
    out.append(html.escape(part[pos:]))
    inner = "".join(out)
    return f'<span style="{style}">{inner}</span>' if style and inner else inner


def _sentence_html(text_: str, start: int, end: int, purple: tuple | None = None) -> str:
    """One sentence of a text, from `start` to `end`, with its long words on a background, and the part from
    purple[0] to purple[1] in purple."""
    a, b = purple if purple else (end, end)
    a, b = min(max(a, start), end), min(max(b, start), end)
    return (_chipped(text_[start:a], _CHIP_DATA) + _chipped(text_[a:b], _CHIP_LOSS, f"{_LOSS};font-weight:400")
            + _chipped(text_[b:end], _CHIP_DATA))


def _length_bars(pieces: list, lang: str, w: dict, most: int) -> str:
    rows = []
    for i, (_, _, ws) in enumerate(pieces, 1):
        rows.append(f'<div style="{_UI};{_DATA}">' + _bar(len(ws), most, "var(--ea-data)", 12)
                    + html.escape(w["sentence"].format(i=_num(i, lang), n=_num(len(ws), lang))) + "</div>")
    return "".join(rows)


# --- the idea: one letter, twice ---------------------------------------------------------------------------------------

_IDEA_TEXT = {
    "en": {"title": "One letter about the green bin, in official language and in plain language",
           "hard": "In official language:", "plain": "In plain language:",
           "sentence": "sentence {i}: {n} words",
           "counts": "{w} words in {s} sentences, {y} syllables: {wps} words per sentence and {spw} syllables per word.",
           "score": "Flesch's reading ease: {f}, {band}",
           "gray": "Gray: the two letters, with each word of three syllables or more on a background, and the length of each sentence.",
           "blue": "Blue: the score that Flesch's formula gives each letter.",
           "purple": "Purple: the part of the official letter that the plain letter leaves out; no formula sees that it is gone."},
    "nl": {"title": "Eén brief over de groene container, in ambtelijke taal en in eenvoudige taal",
           "hard": "In ambtelijke taal:", "plain": "In eenvoudige taal:",
           "sentence": "zin {i}: {n} woorden",
           "counts": "{w} woorden in {s} zinnen, {y} lettergrepen: {wps} woorden per zin en {spw} lettergrepen per woord.",
           "score": "Flesch-Douma: {f}, {band}",
           "gray": "Grijs: de twee brieven, met elk woord van drie lettergrepen of meer op een achtergrond, en de lengte van elke zin.",
           "blue": "Blauw: de score die de formule van Douma aan elke brief geeft.",
           "purple": "Paars: het stuk van de ambtelijke brief dat de eenvoudige brief weglaat; geen formule ziet dat het weg is."},
}


def idea_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang]
    body = ""
    all_pieces = {kind: _pieces(LETTERS[lang][kind]) for kind in ("hard", "plain")}
    most = max(len(ws) for p in all_pieces.values() for _, _, ws in p)
    for kind in ("hard", "plain"):
        text_ = LETTERS[lang][kind]
        pieces = all_pieces[kind]
        start = text_.find(LEFT_OUT[lang]) if kind == "hard" else -1
        purple = (start, start + len(LEFT_OUT[lang])) if start >= 0 else None
        body += _label(w[kind]) + f'<div style="{_DATA}">' + " ".join(
            _sentence_html(text_, a, b, purple=purple) for a, b, _ in pieces) + "</div>"
        body += _length_bars(pieces, lang, w, most)
        n_w, n_s, n_y = counts([ws for _, _, ws in pieces])
        body += _label(w["counts"].format(w=_num(n_w, lang), s=_num(n_s, lang), y=_num(n_y, lang),
                                          wps=_dec(n_w / n_s, lang, 1), spw=_dec(n_y / n_w, lang, 2)))
        score = FORMULA[lang](n_w, n_s, n_y)
        body += f'<div style="{_UI};{_PRED}">{html.escape(w["score"].format(f=_score(score, lang), band=BAND_NAMES[lang][band(score, TABLE[FORMULA[lang].__name__])]))}</div>'
    body += _colors(w, body)
    return yours.figure(w["title"], body, alt=w["title"])


# --- the pictures under the exercises ------------------------------------------------------------------------------------

_SYL_TEXT = {
    "en": {"title": "Your syllables on words counted by hand",
           "lang": {"en": "English words:", "nl": "Dutch words:"},
           "head": ["Word", "Your syllables", "By hand"],
           "agree": "Your syllables gives the number a reader hears in {k} of the {n} words.",
           "gray": "Gray: the words, and the syllables a reader hears, counted by hand. A dashed line under a word: the course's rule gives another number than the hand count.",
           "blue": "Blue: what your syllables gives, when it is what the course's syllables gives.",
           "purple": "Purple: a number that differs from what the course's syllables gives.",
           "kind": "a whole number"},
    "nl": {"title": "Je syllables op woorden die met de hand zijn geteld",
           "lang": {"en": "Engelse woorden:", "nl": "Nederlandse woorden:"},
           "head": ["Woord", "Je syllables", "Met de hand"],
           "agree": "Je syllables geeft bij {k} van de {n} woorden het aantal dat een lezer hoort.",
           "gray": "Grijs: de woorden, en de lettergrepen die een lezer hoort, met de hand geteld. Een stippellijn onder een woord: de regel van de course geeft een ander aantal dan de telling met de hand.",
           "blue": "Blauw: wat je syllables geeft, als dat is wat de syllables van de course geeft.",
           "purple": "Paars: een aantal dat anders is dan bij de syllables van de course.",
           "kind": "een geheel getal"},
}


def syllables_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _SYL_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    order = (lang, "nl" if lang == "en" else "en")
    words_ = [x for l_ in order for x, _ in BY_HAND[l_]]
    mine = _call(lambda: [fn(x) for x in words_])
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    if not all(_whole(v) for v in mine):
        return _empty(w, "odd")
    got = dict(zip(words_, mine))
    body, agree = "", 0
    for l_ in order:
        rows = []
        for x, hand in BY_HAND[l_]:
            v = int(got[x])
            good = v == syllables(x)
            agree += v == hand
            word_style = _DATA + (";text-decoration:underline dashed;text-underline-offset:0.2rem" if syllables(x) != hand else "")
            rows.append("<tr>" + _cell(html.escape(x), word_style) + _cell(_num(v, lang), _PRED if good else _LOSS, _NUMBER)
                        + _cell(_num(hand, lang), _DATA, _NUMBER) + "</tr>")
        body += _label(w["lang"][l_]) + _table(rows, w["head"], (1, 2))
    body += _note(w["agree"].format(k=_num(agree, lang), n=_num(len(words_), lang)))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


_COUNTS_TEXT = {
    "en": {"title": "Your counts on the four letters about the green bin",
           "head": ["Letter", "Words", "Sentences", "Syllables", "Words per sentence", "Syllables per word"],
           "names": {("en", "hard"): "English, official", ("en", "plain"): "English, plain",
                     ("nl", "hard"): "Dutch, official", ("nl", "plain"): "Dutch, plain"},
           "each": "Your counts of each sentence of the letter in {kind}:",
           "kinds": {"hard": "official language", "plain": "plain language"},
           "sentence": "sentence {i}: {w} words, {y} syllables",
           "gray": "Gray: the letters.",
           "blue": "Blue: what your counts gives, and the averages from it, when it is what the course's counts gives; a bar is the length of a sentence in words.",
           "purple": "Purple: a number that differs from what the course's counts gives.",
           "kind": "three whole numbers"},
    "nl": {"title": "Je counts op de vier brieven over de groene container",
           "head": ["Brief", "Woorden", "Zinnen", "Lettergrepen", "Woorden per zin", "Lettergrepen per woord"],
           "names": {("en", "hard"): "Engels, ambtelijk", ("en", "plain"): "Engels, eenvoudig",
                     ("nl", "hard"): "Nederlands, ambtelijk", ("nl", "plain"): "Nederlands, eenvoudig"},
           "each": "Je counts van elke zin van de brief in {kind}:",
           "kinds": {"hard": "ambtelijke taal", "plain": "eenvoudige taal"},
           "sentence": "zin {i}: {w} woorden, {y} lettergrepen",
           "gray": "Grijs: de brieven.",
           "blue": "Blauw: wat je counts geeft, en de gemiddelden daaruit, als dat is wat de counts van de course geeft; een balk is de lengte van een zin in woorden.",
           "purple": "Paars: een getal dat anders is dan bij de counts van de course.",
           "kind": "drie gehele getallen"},
}


def counts_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _COUNTS_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    letters = [(l_, kind) for l_ in (lang, "nl" if lang == "en" else "en") for kind in ("hard", "plain")]
    cut = {k: sentences(LETTERS[k[0]][k[1]]) for k in letters}

    def make():
        whole = {k: fn([list(s) for s in cut[k]]) for k in letters}
        each = {kind: [fn([list(s)]) for s in cut[(lang, kind)]] for kind in ("hard", "plain")}
        return whole, each

    mine = _call(make)
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    whole, each = mine
    if not all(_is_counts(v) for v in whole.values()) or not all(_is_counts(v) for vs in each.values() for v in vs):
        return _empty(w, "odd")
    rows = []
    for k in letters:
        got, want = tuple(int(x) for x in whole[k]), counts(cut[k])
        cells = [_cell(html.escape(w["names"][k]), _DATA)]
        for x, y in zip(got, want):
            cells.append(_cell(_num(x, lang), _PRED if x == y else _LOSS, _NUMBER))
        good = got == want
        style = _OUT if good else _LOSS
        cells.append(_cell(_dec(got[0] / got[1], lang, 1) if got[1] else "–", style, _NUMBER))
        cells.append(_cell(_dec(got[2] / got[0], lang, 2) if got[0] else "–", style, _NUMBER))
        rows.append("<tr>" + "".join(cells) + "</tr>")
    body = _table(rows, w["head"], (1, 2, 3, 4, 5))
    most = max([int(v[0]) for vs in each.values() for v in vs] + [1])
    for kind in ("hard", "plain"):
        body += _label(w["each"].format(kind=w["kinds"][kind]))
        for i, (v, s) in enumerate(zip(each[kind], cut[(lang, kind)]), 1):
            good = tuple(int(x) for x in v) == counts([s])
            color = "var(--ea-pred)" if good else "var(--ea-loss)"
            body += (f'<div style="{_UI};{_OUT if good else _LOSS}">' + _bar(int(v[0]), most, color, 12)
                     + html.escape(w["sentence"].format(i=_num(i, lang), w=_num(int(v[0]), lang), y=_num(int(v[2]), lang)))
                     + "</div>")
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


_FLESCH_TEXT = {
    "en": {"title": "Your flesch on eight books and today's Dutch sentences",
           "head": ["Text", "Words per sentence", "Syllables per word", "Your flesch", "Flesch's table", "Douma"],
           "hardest": "By your flesch, the hardest English book is {hard} ({h}), and the easiest {easy} ({e}).",
           "gray": "Gray: the texts, their averages, and Douma's score for the Dutch ones, which the course computes.",
           "blue": "Blue: what your flesch gives, and its band in Flesch's table, when it is what the course's flesch gives.",
           "purple": "Purple: a score that differs from what the course's flesch gives.",
           "kind": "a number"},
    "nl": {"title": "Je flesch op acht boeken en de Nederlandse zinnen van nu",
           "head": ["Tekst", "Woorden per zin", "Lettergrepen per woord", "Je flesch", "Tabel van Flesch", "Douma"],
           "hardest": "Volgens je flesch is het moeilijkste Engelse boek {hard} ({h}), en het makkelijkste {easy} ({e}).",
           "gray": "Grijs: de teksten, de gemiddelden, en de score van Douma voor de Nederlandse teksten, die de course uitrekent.",
           "blue": "Blauw: wat je flesch geeft, en de plaats in de tabel van Flesch, als dat is wat de flesch van de course geeft.",
           "purple": "Paars: een score die anders is dan bij de flesch van de course.",
           "kind": "een getal"},
}


def flesch_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _FLESCH_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    c = TEXT_COUNTS
    mine = _call(lambda: {key: fn(*c[key]) for key in TEXTS})
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    if not all(_is_number(v) and abs(v) < 1e9 for v in mine.values()):
        return _empty(w, "odd")
    rows = []
    for key in TEXTS:
        n_w, n_s, n_y = c[key]
        v, good = mine[key], _close(mine[key], flesch(*c[key]))
        style = _OUT if good else _LOSS
        rows.append("<tr>" + _cell(f"<i>{html.escape(TEXT_NAMES[lang][key])}</i>", _DATA)
                    + _cell(_dec(n_w / n_s, lang, 1), _DATA, _NUMBER) + _cell(_dec(n_y / n_w, lang, 2), _DATA, _NUMBER)
                    + _cell(_bar(v, 100, "var(--ea-pred)" if good else "var(--ea-loss)", 5) + _score(v, lang),
                            _PRED if good else _LOSS, _NUMBER)
                    + _cell(html.escape(BAND_NAMES[lang][band(v)]), style)
                    + _cell(_score(douma(*c[key]), lang) if TEXT_LANG[key] == "nl" else "–", _DATA, _NUMBER) + "</tr>")
    body = _table(rows, w["head"], (1, 2, 3, 5))
    english = [k for k in TEXTS if TEXT_LANG[k] == "en"]
    hard = min(english, key=lambda k: mine[k])
    easy = max(english, key=lambda k: mine[k])
    body += _note(w["hardest"].format(hard=f"*{TEXT_NAMES[lang][hard]}*", h=_score(mine[hard], lang),
                                      easy=f"*{TEXT_NAMES[lang][easy]}*", e=_score(mine[easy], lang)))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


# --- try it: a text of the learner's own -------------------------------------------------------------------------------

_TRY_TEXT = {
    "en": {"title": "Your text, sentence by sentence, with its scores",
           "sentence": "sentence {i}: {n} words",
           "head": ["Words", "Sentences", "Syllables", "Words per sentence", "Syllables per word"],
           "flesch": "Flesch's reading ease: {f}, {band}",
           "douma": "Flesch-Douma, for Dutch: {f}, {band}",
           "empty": "Write a text in the box, and the course scores it here.",
           "cut": "The course scores the first {n} characters of your text.",
           "more": "and {n} sentences more",
           "gray": "Gray: your text, cut into sentences, with each word of three syllables or more on a background, the length of each sentence, and the counts.",
           "blue": "Blue: the scores of the two formulas, with their band in Flesch's table."},
    "nl": {"title": "Je tekst, zin voor zin, met de scores",
           "sentence": "zin {i}: {n} woorden",
           "head": ["Woorden", "Zinnen", "Lettergrepen", "Woorden per zin", "Lettergrepen per woord"],
           "flesch": "Leesgemak van Flesch, voor Engels: {f}, {band}",
           "douma": "Flesch-Douma: {f}, {band}",
           "empty": "Schrijf een tekst in het vak, dan scoort de course de tekst hier.",
           "cut": "De course scoort de eerste {n} tekens van je tekst.",
           "more": "en nog {n} zinnen",
           "gray": "Grijs: je tekst, in zinnen geknipt, met elk woord van drie lettergrepen of meer op een achtergrond, de lengte van elke zin, en de tellingen.",
           "blue": "Blauw: de scores van de twee formules, met de plaats in de tabel van Flesch."},
}
TRY_SENTENCES = 12   # sentences the try-it picture draws


def try_picture(lang: str, text_="") -> str:
    lang = lang if lang in LANGS else "en"
    w = _TRY_TEXT[lang]
    text_ = text_ if isinstance(text_, str) else ""
    cut = len(text_) > TEXT_CAP
    text_ = text_[:TEXT_CAP]
    pieces = _pieces(text_)
    body = ""
    if not pieces:
        body = _note(w["empty"])
        return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])
    most = max(len(ws) for _, _, ws in pieces)
    for i, (a, b, ws) in enumerate(pieces[:TRY_SENTENCES], 1):
        body += (f'<div style="{_UI};{_DATA}">' + _bar(len(ws), most, "var(--ea-data)", 12)
                 + html.escape(w["sentence"].format(i=_num(i, lang), n=_num(len(ws), lang))) + "</div>"
                 + f'<div style="{_DATA};margin-left:0.5rem">{_sentence_html(text_, a, b)}</div>')
    if len(pieces) > TRY_SENTENCES:
        body += _note(w["more"].format(n=_num(len(pieces) - TRY_SENTENCES, lang)))
    n_w, n_s, n_y = counts([ws for _, _, ws in pieces])
    row = ("<tr>" + "".join(_cell(_num(x, lang), _DATA, _NUMBER) for x in (n_w, n_s, n_y))
           + _cell(_dec(n_w / n_s, lang, 1), _DATA, _NUMBER) + _cell(_dec(n_y / n_w, lang, 2), _DATA, _NUMBER) + "</tr>")
    body += _table([row], w["head"], (0, 1, 2, 3, 4))
    for key, f in (("flesch", flesch), ("douma", douma)):
        v = f(n_w, n_s, n_y)
        body += (f'<div style="{_UI};{_PRED}">' + _bar(v, 100, "var(--ea-pred)", 8)
                 + html.escape(w[key].format(f=_score(v, lang), band=BAND_NAMES[lang][band(v, TABLE[key])])) + "</div>")
    if cut:
        body += _note(w["cut"].format(n=_num(TEXT_CAP, lang)))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------------

_Val = S._Val
_DOTS = S._DOTS
_SAME_TEXTS = S._SAME_TEXTS


def _value(v) -> str:
    return S._value(v)


def _show(v):
    out = _value(v)
    return out if len(out) <= 300 else out[:300] + "…"


def _written(v):
    return _Val(_value(v), v)


def _as_counts(v):
    """What counts gives, with a list of three made a tuple: [6, 2, 7] counts as (6, 2, 7)."""
    return tuple(v) if isinstance(v, list) and len(v) == 3 else v


# The small cases the exercise texts show; the same in both lessons.
TOY_SYLLABLES = ("simple", "leesbaarheid")
TOY_COUNTS = ([["We", "start", "on", "Monday"], ["Call", "us", "today"]], [["Bel", "ons"], ["We", "beginnen", "op", "maandag"]])
TOY_FLESCH = ((31, 4, 41), (33, 4, 51))   # the plain letters, in English and in Dutch

_SY_HEAD = "def syllables(word):\n"
_SY_GROUPS = '    groups = re.findall(r"[aeiouy]+", word.lower())\n'
_SY_RET = "    return max(1, len(groups))\n"


def _sy(groups: str = _SY_GROUPS, ret: str = _SY_RET) -> str:
    return _SY_HEAD + groups + ret


_SY = _sy()
_SY_START = _sy("    groups = ...\n", "    return ...\n")
_SY_NO_MAX = _sy(ret="    return len(groups)\n")
_SY_LETTERS = _sy('    groups = re.findall(r"[aeiouy]", word.lower())\n')
_SY_NO_Y = _sy('    groups = re.findall(r"[aeiou]+", word.lower())\n')
_SY_NO_LOWER = _sy('    groups = re.findall(r"[aeiouy]+", word)\n')
_SY_CONSONANTS = _sy('    groups = re.findall(r"[^aeiouy]+", word.lower())\n')

_CO_HEAD = "def counts(sentences):\n    n_words = 0\n    n_syllables = 0\n    for sentence in sentences:\n        for word in sentence:\n"
_CO_WORD = "            n_words = n_words + 1\n"
_CO_SYL = "            n_syllables = n_syllables + syllables(word)\n"
_CO_RET = "    return n_words, len(sentences), n_syllables\n"


def _co(word: str = _CO_WORD, syl: str = _CO_SYL, ret: str = _CO_RET) -> str:
    return _CO_HEAD + word + syl + ret


_CO = _co()
_CO_START = _CO_HEAD + "            ...\n" + "    return ...\n"
_CO_LAST = _co(ret="    return n_words, len(sentence), n_syllables\n")
_CO_LETTERS = _co(word="            n_words = n_words + len(word)\n")
_CO_ONE = _co(syl="            n_syllables = n_syllables + 1\n")
_CO_ORDER = _co(ret="    return n_words, n_syllables, len(sentences)\n")
_CO_INSIDE = _CO_HEAD + _CO_WORD + _CO_SYL + "        return n_words, len(sentences), n_syllables\n"

_FL_HEAD = "def flesch(words, sentences, syllables):\n"


def _fl(ret: str) -> str:
    return _FL_HEAD + "    return " + ret + "\n"


_FL = _fl("206.835 - 1.015 * words / sentences - 84.6 * syllables / words")
_FL_START = _fl("...")
_FL_PER_100 = _fl("206.835 - 1.015 * words / sentences - 0.846 * syllables / words")
_FL_SWAPPED = _fl("206.835 - 84.6 * words / sentences - 1.015 * syllables / words")
_FL_SENTENCES_UP = _fl("206.835 - 1.015 * sentences / words - 84.6 * syllables / words")
_FL_WORDS_UP = _fl("206.835 - 1.015 * words / sentences - 84.6 * words / syllables")
_FL_PLUS = _fl("206.835 - 1.015 * words / sentences + 84.6 * syllables / words")
_FL_DOUMA = _fl("206.84 - 0.93 * words / sentences - 77 * syllables / words")
_FL_ROUNDED = _fl("round(206.835 - 1.015 * words / sentences - 84.6 * syllables / words, 1)")
_ROUNDS = re.compile(r"\b(round|int|floor|ceil|trunc)\s*\(")


def _rounded(got, text_) -> bool:
    """A number that is not the score, from code that rounds: round(x, 1), round(x, 3), round(x) or int(x)."""
    return _is_number(got) and text_ is not None and _ROUNDS.search(text_) is not None


def _not_whole(got, text_) -> bool:
    return not _whole(got)


def _not_counts(got, text_) -> bool:
    return not _is_counts(got)


def _not_number(got, text_) -> bool:
    return not _is_number(got)


REQUIRED = ["syllables", "counts", "flesch", "changes"]


def _letter(lang: str, kind: str) -> _Val:
    return _Val(f'C.sentences(C.LETTERS["{lang}"]["{kind}"])', sentences(LETTERS[lang][kind]))


EXERCISES = {
    "syllables": FunctionBox(
        name="syllables",
        picture=syllables_picture,
        function="syllables",
        start=_SY_START,
        solution=_SY,
        show=_show,
        names={"re": re},
        visible=[(_written(x),) for x in TOY_SYLLABLES],
        hidden=[(_written(x),) for x in ("Hmm", "Apple", "happy", "boot", "make", "Readability", "bijenkast", "rhythm",
                                         "queue", "a")],
        first=[_DOTS],
        mistakes=[
            Mistake("not-whole", matches=_not_whole,
                    en="`{call}` gives {got}, and it should give {want}: one whole number, the number of syllables. What does your function give back?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: één geheel getal, het aantal lettergrepen. Wat geeft je functie terug?"),
            Mistake("no-max", wrong=_SY_NO_MAX,
                    en="`{call}` gives {got}, and it should give {want}: a word without a, e, i, o, u or y, such as *Hmm*, gets 0 syllables from your function. Every word has at least one: which function gives the larger of two numbers?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een woord zonder a, e, i, o, u of y, zoals *Hmm*, krijgt van je functie 0 lettergrepen. Elk woord heeft er minstens één: welke functie geeft het grootste van twee getallen?"),
            Mistake("letters", wrong=_SY_LETTERS,
                    en="`{call}` gives {got}, and it should give {want}: your function counts each vowel on its own, so two vowels in a row, such as *ee* or *oo*, count as two syllables. Which sign after the class lets one match take a whole group of vowels in a row?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie telt elke klinker apart, dus twee klinkers op een rij, zoals *ee* of *oo*, tellen als twee lettergrepen. Welk teken na de tekenset laat één match een hele groep klinkers op een rij pakken?"),
            Mistake("no-y", wrong=_SY_NO_Y,
                    en="`{call}` gives {got}, and it should give {want}: your class leaves out the y, so *happy* gets one syllable. Which six letters count as vowels in this lesson?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je tekenset laat de y weg, dus *happy* krijgt één lettergreep. Welke zes letters tellen in deze les als klinker?"),
            Mistake("no-lower", wrong=_SY_NO_LOWER,
                    en="`{call}` gives {got}, and it should give {want}: the class holds small letters only, so the capital *A* of *Apple* is not a vowel to your function. Which method gives the word in small letters first?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de tekenset bevat alleen kleine letters, dus de hoofdletter *A* van *Apple* is voor je functie geen klinker. Welke methode geeft het woord eerst in kleine letters?"),
            Mistake("consonants", wrong=_SY_CONSONANTS,
                    en="`{call}` gives {got}, and it should give {want}: with `^` at the start of the class, your function counts the groups of letters that are not vowels. A syllable has a vowel at its heart: what is left without the `^`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: met `^` aan het begin van de tekenset telt je functie de groepen letters die geen klinker zijn. Een lettergreep heeft een klinker als kern: wat blijft er over zonder de `^`?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: the first finds the groups of vowels in the word in small letters, and the `return` gives how many there are, at least 1.",
                      "nl": "Vervang de twee `...`: de eerste vindt de groepen klinkers in het woord in kleine letters, en de `return` geeft hoeveel het er zijn, minstens 1."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function count every group of vowels in a row as one syllable, with y as a vowel, and give at least 1?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Telt je functie elke groep klinkers op een rij als één lettergreep, met de y als klinker, en geeft de functie minstens 1?"},
            "almost": {"en": "Right for the two words of the example. But `{call}` gives {got}, and it should give {want}. Does your function look at capitals too, and give 1 to a word without vowels?",
                       "nl": "Goed voor de twee woorden van het voorbeeld. Maar `{call}` geeft {got}, en het moet {want} zijn. Kijkt je functie ook naar hoofdletters, en geeft de functie 1 aan een woord zonder klinkers?"},
            "correct": {"en": "Right: your function counts syllables the way the course does. In the picture below, in which words does the rule give another number than a reader hears, and why?",
                        "nl": "Goed: je functie telt lettergrepen zoals de course het doet. Kijk in het plaatje hieronder: bij welke woorden geeft de regel een ander aantal dan een lezer hoort, en waarom?"},
        },
    ),
    "counts": FunctionBox(
        name="counts",
        picture=counts_picture,
        function="counts",
        start=_CO_START,
        solution=_CO,
        show=_show,
        names={"syllables": syllables},
        collect=_as_counts,
        visible=[(_written(x),) for x in TOY_COUNTS],
        hidden=[(_written([]),), (_written([[]]),), (_written([["Hmm"]]),),
                (_written([["a", "b"], ["c"], ["d", "e", "f"]]),)]
               + [(_letter(lang, kind),) for lang in LANGS for kind in ("hard", "plain")],
        first=[_DOTS],
        mistakes=[
            Mistake("list-word", error=lambda e: isinstance(e, AttributeError) and "'list' object" in str(e) and "lower" in str(e),
                    en="`{call}` stops with an `AttributeError`: the course's `syllables` got a whole list of words, and a list has no `.lower()`. Which one word should `syllables` get?",
                    nl="`{call}` stopt met een `AttributeError`: de `syllables` van de course kreeg een hele lijst woorden, en een lijst heeft geen `.lower()`. Welk ene woord moet `syllables` krijgen?"),
            Mistake("not-counts", matches=_not_counts,
                    en="`{call}` gives {got}, and it should give {want}: three whole numbers, the words, the sentences and the syllables. What does your function give back?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: drie gehele getallen, de woorden, de zinnen en de lettergrepen. Wat geeft je functie terug?"),
            Mistake("last-sentence", wrong=_CO_LAST,
                    en="`{call}` gives {got}, and it should give {want}: the second number is the length of the last sentence, the one the loop ended with. How many sentences does the whole text have?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het tweede getal is de lengte van de laatste zin, de zin waarmee de loop eindigde. Hoeveel zinnen heeft de hele tekst?"),
            Mistake("letters", wrong=_CO_LETTERS,
                    en="`{call}` gives {got}, and it should give {want}: the first number counts the letters of the words, not the words. How much does one word add to the number of words?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het eerste getal telt de letters van de woorden, niet de woorden. Hoeveel voegt één woord toe aan het aantal woorden?"),
            Mistake("one-each", wrong=_CO_ONE,
                    en="`{call}` gives {got}, and it should give {want}: the third number counts each word as one syllable, so it equals the number of words. Which function of the course gives the syllables of a word?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het derde getal telt elk woord als één lettergreep, dus het is gelijk aan het aantal woorden. Welke functie van de course geeft de lettergrepen van een woord?"),
            Mistake("order", wrong=_CO_ORDER,
                    en="`{call}` gives {got}, and it should give {want}: the same three numbers, but the sentences and the syllables have changed places. In which order does the function give them back?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: dezelfde drie getallen, maar de zinnen en de lettergrepen hebben van plaats gewisseld. In welke volgorde geeft de functie de getallen terug?"),
            Mistake("return-inside", wrong=_CO_INSIDE,
                    en="`{call}` gives {got}, and it should give {want}: your function stops after the first sentence. Where does your `return` stand: inside the outer `for` loop, or after it?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie stopt na de eerste zin. Waar staat je `return`: in de buitenste `for`-loop, of erna?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: the first with two lines that count the word and its syllables, and the `return` with the three numbers.",
                      "nl": "Vervang de twee `...`: de eerste door twee regels die het woord en de lettergrepen ervan tellen, en de `return` door de drie getallen."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does each word add 1 to the words and its syllables to the syllables, and does the function give back the words, the number of sentences and the syllables, in that order?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Telt elk woord 1 op bij de woorden en de lettergrepen ervan bij de lettergrepen, en geeft de functie de woorden, het aantal zinnen en de lettergrepen terug, in deze volgorde?"},
            "almost": {"en": "Right for the two small texts. But `{call}` gives {got}, and it should give {want}. Does your function give `(0, 0, 0)` for no sentences, and count an empty sentence as a sentence?",
                       "nl": "Goed voor de twee kleine teksten. Maar `{call}` geeft {got}, en het moet {want} zijn. Geeft je functie `(0, 0, 0)` bij geen zinnen, en telt de functie een lege zin als zin?"},
            "correct": {"en": "Right: your function counts what the formula needs. In the picture below, how long is the longest sentence of the letter in official language, and how long are those of the plain letter?",
                        "nl": "Goed: je functie telt wat de formule nodig heeft. Kijk in het plaatje hieronder: hoe lang is de langste zin van de brief in ambtelijke taal, en hoe lang zijn de zinnen van de eenvoudige brief?"},
        },
    ),
    "flesch": FunctionBox(
        name="flesch",
        picture=flesch_picture,
        function="flesch",
        start=_FL_START,
        solution=_FL,
        show=_show,
        visible=[tuple(_written(x) for x in case) for case in TOY_FLESCH],
        hidden=[(_written(1),) * 3, (_written(100), _written(5), _written(150))]
               + [tuple(_written(x) for x in letter_counts()[(lang, "hard")]) for lang in LANGS]
               + [tuple(_written(x) for x in TEXT_COUNTS[key]) for key in ("pride", "moby-dick", "eline-vere")],
        compare=lambda a, b: _close(a, b) or a == b,
        first=[_DOTS],
        mistakes=[
            Mistake("not-number", matches=_not_number,
                    en="`{call}` gives {got}, and it should give {want}: one number, the score. What does your function give back?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: één getal, de score. Wat geeft je functie terug?"),
            Mistake("per-100", wrong=_FL_PER_100,
                    en="`{call}` gives {got}, and it should give {want}: you use Flesch's 0.846, which goes with syllables per 100 words, on syllables per word. What does the weight become for one word instead of 100?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je gebruikt de 0,846 van Flesch, die bij lettergrepen per 100 woorden hoort, voor lettergrepen per woord. Wat wordt de factor voor één woord in plaats van 100?"),
            Mistake("swapped", wrong=_FL_SWAPPED,
                    en="`{call}` gives {got}, and it should give {want}: the two weights have changed places. Which of the two averages gets the large weight, 84.6?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de twee factoren hebben van plaats gewisseld. Welk van de twee gemiddelden krijgt de grote factor, 84,6?"),
            Mistake("sentences-up", wrong=_FL_SENTENCES_UP,
                    en="`{call}` gives {got}, and it should give {want}: you divide the sentences by the words, which gives sentences per word. The formula wants the words per sentence: which number goes on top?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je deelt de zinnen door de woorden, en dat geeft zinnen per woord. De formule wil de woorden per zin: welk getal komt boven de deelstreep?"),
            Mistake("words-up", wrong=_FL_WORDS_UP,
                    en="`{call}` gives {got}, and it should give {want}: you divide the words by the syllables, which gives words per syllable. The formula wants the syllables per word: which number goes on top?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je deelt de woorden door de lettergrepen, en dat geeft woorden per lettergreep. De formule wil de lettergrepen per woord: welk getal komt boven de deelstreep?"),
            Mistake("plus", wrong=_FL_PLUS,
                    en="`{call}` gives {got}, and it should give {want}: long words raise your score, while they should lower it. Which sign goes before the 84.6?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: lange woorden verhogen je score, terwijl lange woorden de score moeten verlagen. Welk teken hoort voor de 84,6?"),
            Mistake("douma", wrong=_FL_DOUMA,
                    en="`{call}` gives {got}, and it should give {want}: those are Douma's numbers for Dutch. Which three numbers does Flesch's formula have?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: dat zijn de getallen van Douma voor het Nederlands. Welke drie getallen heeft de formule van Flesch?"),
            Mistake("rounded", matches=_rounded,
                    en="`{call}` does not give exactly the score the check computes: your function rounds it. The pictures round the score only when they show it; can the function give the score unrounded?",
                    nl="`{call}` geeft niet precies de score die de check uitrekent: je functie rondt de score af. De plaatjes ronden de score pas af bij het tonen; kan de functie de score onafgerond geven?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with Flesch's formula: 206.835, minus 1.015 times the words per sentence, minus 84.6 times the syllables per word.",
                      "nl": "Vervang de `...` door de formule van Flesch: 206,835, min 1,015 keer de woorden per zin, min 84,6 keer de lettergrepen per woord."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Do you start from 206.835, and take off 1.015 times the words per sentence and 84.6 times the syllables per word?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Begin je bij 206,835, en haal je er 1,015 keer de woorden per zin en 84,6 keer de lettergrepen per woord af?"},
            "almost": {"en": "Right for the two plain letters. But `{call}` gives {got}, and it should give {want}. Does your function give the score as it is, also when it is above 100 or below 0?",
                       "nl": "Goed voor de twee eenvoudige brieven. Maar `{call}` geeft {got}, en het moet {want} zijn. Geeft je functie de score zoals de score is, ook als de score boven 100 of onder 0 ligt?"},
            "correct": {"en": "Right: your function is Flesch's reading ease. In the picture below, which English book is the hardest by Flesch's table, and how do the Dutch sentences of Tatoeba score?",
                        "nl": "Goed: je functie is het leesgemak van Flesch. Kijk in het plaatje hieronder: welk Engels boek is het moeilijkst volgens de tabel van Flesch, en hoe scoren de Nederlandse zinnen van Tatoeba?"},
        },
    ),
}

# The letter of Exercise 4; each change makes a new letter, and the tests check its score with both formulas.
BASE = {"en": "From next month, we empty the green bin once every two weeks, and we start on the first Monday. We never "
              "empty it on holidays. Do you have a question? Call the city service desk.",
        "nl": "Vanaf volgende maand legen we de groene container eens in de twee weken, en we beginnen op de eerste "
              "maandag. Op feestdagen legen we de container niet. Heeft u een vraag? Bel dan de servicebalie van de "
              "gemeente."}
_FIRST = {"en": "From next month, we empty the green bin once every two weeks, and we start on the first Monday.",
          "nl": "Vanaf volgende maand legen we de groene container eens in de twee weken, en we beginnen op de eerste maandag."}
_QUESTION = {"en": "Do you have a question? Call the city service desk.",
             "nl": "Heeft u een vraag? Bel dan de servicebalie van de gemeente."}
CHANGES = {   # each change as (what goes, what comes in its place)
    "en": {"split": ("once every two weeks, and we start", "once every two weeks. We start"),
           "sundays": ("on holidays", "on Sundays"),
           "call": ("Call the city service desk.", "Call us."),
           "opposite": ("We never", "We always"),
           "weekday": ("Monday", "Friday"),
           "order": (BASE["en"], _QUESTION["en"] + " " + BASE["en"].replace(" " + _QUESTION["en"], "")),
           "salad": (_FIRST["en"], "Monday first the on start we and, weeks two every once bin green the empty we month next from.")},
    "nl": {"split": ("eens in de twee weken, en we beginnen", "eens in de twee weken. We beginnen"),
           "sundays": ("Op feestdagen", "Op zondag"),
           "call": ("Bel dan de servicebalie van de gemeente.", "Bel ons."),
           "opposite": ("de container niet.", "de container ook."),
           "weekday": ("maandag", "vrijdag"),
           "order": (BASE["nl"], _QUESTION["nl"] + " " + BASE["nl"].replace(" " + _QUESTION["nl"], "")),
           "salad": (_FIRST["nl"], "Maandag eerste de op beginnen we en, weken twee de in eens container groene de we legen maand volgende vanaf.")},
}


def changed(lang: str, item: str) -> str:
    """The letter of Exercise 4 after one change."""
    old, new = CHANGES[lang][item]
    return BASE[lang].replace(old, new)


EXERCISES["changes"] = Sort(
    name="changes",
    groups={
        "up": {"en": "The score goes up", "nl": "De score gaat omhoog"},
        "same": {"en": "The score stays the same", "nl": "De score blijft gelijk"},
    },
    items={
        "split": {"en": "**Two sentences.** The first sentence becomes two: *From next month, we empty the green bin once every two weeks. We start on the first Monday.*",
                  "nl": "**Twee zinnen.** De eerste zin wordt er twee: *Vanaf volgende maand legen we de groene container eens in de twee weken. We beginnen op de eerste maandag.*"},
        "sundays": {"en": "**Another day.** *Sundays* instead of *holidays*: *We never empty it on Sundays.*",
                    "nl": "**Een andere dag.** *zondag* in plaats van *feestdagen*: *Op zondag legen we de container niet.*"},
        "call": {"en": "**Call us.** *Call us.* instead of *Call the city service desk.*",
                 "nl": "**Bel ons.** *Bel ons.* in plaats van *Bel dan de servicebalie van de gemeente.*"},
        "opposite": {"en": "**The opposite.** *always* instead of *never*: *We always empty it on holidays.*",
                     "nl": "**Het omgekeerde.** *ook* in plaats van *niet*: *Op feestdagen legen we de container ook.*"},
        "weekday": {"en": "**Another weekday.** *Friday* instead of *Monday*.",
                    "nl": "**Een andere weekdag.** *vrijdag* in plaats van *maandag*."},
        "order": {"en": "**Another order.** The two sentences about a question come first.",
                  "nl": "**Een andere volgorde.** De twee zinnen over een vraag komen eerst."},
        "salad": {"en": "**Word salad.** The words of the first sentence backward: *Monday first the on start we and, weeks two every once bin green the empty we month next from.*",
                  "nl": "**Woordensalade.** De woorden van de eerste zin achterstevoren: *Maandag eerste de op beginnen we en, weken twee de in eens container groene de we legen maand volgende vanaf.*"},
    },
    solution={"split": "up", "sundays": "up", "call": "up",
              "opposite": "same", "weekday": "same", "order": "same", "salad": "same"},
    nudge={
        "split": {"en": "The words stay nearly the same, but the letter gets one sentence more. What happens to the number of words per sentence?",
                  "nl": "De woorden blijven bijna gelijk, maar de brief krijgt een zin meer. Wat gebeurt er met het aantal woorden per zin?"},
        "sundays": {"en": "*holidays* has three groups of vowels, *Sundays* two, and the sentence keeps its length. What happens to the syllables per word?",
                    "nl": "*feestdagen* heeft drie groepen klinkers, *zondag* twee, en de zin houdt dezelfde lengte. Wat gebeurt er met de lettergrepen per woord?"},
        "call": {"en": "*Call us.* has two short words where the sentence had five, some of them long. What happens to the words per sentence and the syllables per word?",
                 "nl": "*Bel ons.* heeft twee korte woorden waar de zin er zeven had, een paar ervan lang. Wat gebeurt er met de woorden per zin en de lettergrepen per woord?"},
        "opposite": {"en": "*never* and *always* both have two groups of vowels, and the sentence keeps its length. Does the formula see that the letter now says the opposite?",
                     "nl": "*niet* en *ook* hebben allebei één groep klinkers, en de zin houdt dezelfde lengte. Ziet de formule dat de brief nu het omgekeerde zegt?"},
        "weekday": {"en": "*Monday* and *Friday* both have two groups of vowels. Which of the two averages changes?",
                    "nl": "*maandag* en *vrijdag* hebben allebei twee groepen klinkers. Welk van de twee gemiddelden verandert?"},
        "order": {"en": "The same sentences, in another order. Does an average change when you take the same numbers in another order?",
                  "nl": "Dezelfde zinnen, in een andere volgorde. Verandert een gemiddelde als je dezelfde getallen in een andere volgorde neemt?"},
        "salad": {"en": "The same words, backward, in the same sentence. Does the formula look at the order of the words?",
                  "nl": "Dezelfde woorden, achterstevoren, in dezelfde zin. Kijkt de formule naar de volgorde van de woorden?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the groups again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de groepen opnieuw uit."},
        "empty": {"en": "Put each change with what it does to the score.",
                  "nl": "Zet elke verandering bij het effect op de score."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "{item} belongs with the other group.",
                  "nl": "{item} hoort bij de andere groep."},
        "correct": {"en": "All {n} are right. The formula sees two averages: the words per sentence and the syllables per word. Shorter sentences and shorter words raise the score; the order of the sentences or of the words, and what the letter says, change nothing. The word salad scores exactly as high as the letter, and so does the letter that now says the opposite.",
                    "nl": "Alle {n} kloppen. De formule ziet twee gemiddelden: de woorden per zin en de lettergrepen per woord. Kortere zinnen en kortere woorden verhogen de score; de volgorde van de zinnen of van de woorden, en wat de brief zegt, veranderen niets. De woordensalade scoort precies even hoog als de brief, net als de brief die nu het omgekeerde zegt."},
    },
)

# What the tests feed each check, and what it must say: (exercise, code or choices, state, known mistake).
CASES = [
    ("syllables", _SY_START, "empty", None),
    ("syllables", _SY, "correct", None),
    ("syllables", _sy(ret="    return ...\n"), "wrong", "dots-left"),
    ("syllables", _sy(ret="    return groups\n"), "wrong", "not-whole"),
    ("syllables", _SY_NO_MAX, "wrong", "no-max"),
    ("syllables", _SY_LETTERS, "wrong", "letters"),
    ("syllables", _SY_NO_Y, "wrong", "no-y"),
    ("syllables", _SY_NO_LOWER, "wrong", "no-lower"),
    ("syllables", _SY_CONSONANTS, "wrong", "consonants"),
    ("syllables", _sy('    groups = re.findall("[aeiouy]+", word.lower())\n', "    return max(len(groups), 1)\n"), "correct", None),
    ("syllables", "def syllables(word):\n    return max(1, len(re.findall(r\"[aeiouy]+\", word, re.IGNORECASE)))\n",
     "correct", None),
    ("counts", _CO_START, "empty", None),
    ("counts", _CO, "correct", None),
    ("counts", _CO_HEAD + _CO_WORD + _CO_SYL + "    return ...\n", "wrong", "dots-left"),
    ("counts", _co(ret="    return n_words + n_syllables\n"), "wrong", "not-counts"),
    ("counts", _CO_LAST, "wrong", "last-sentence"),
    ("counts", _co(syl="            n_syllables = n_syllables + syllables(sentence)\n"), "error", "list-word"),
    ("counts", _CO_LETTERS, "wrong", "letters"),
    ("counts", _CO_ONE, "wrong", "one-each"),
    ("counts", _CO_ORDER, "wrong", "order"),
    ("counts", _CO_INSIDE, "wrong", "return-inside"),
    ("counts", _co("            n_words += 1\n", "            n_syllables += syllables(word)\n"), "correct", None),
    ("counts", _co(ret="    return [n_words, len(sentences), n_syllables]\n"), "correct", None),
    ("counts", "def counts(sentences):\n    words = [w for s in sentences for w in s]\n"
     "    return len(words), len(sentences), sum(syllables(w) for w in words)\n", "correct", None),
    ("flesch", _FL_START, "empty", None),
    ("flesch", _FL, "correct", None),
    ("flesch", _fl("206.835 - 1.015 * words / sentences - 84.6 * ..."), "wrong", "dots-left"),
    ("flesch", _fl("str(206.835 - 1.015 * words / sentences - 84.6 * syllables / words)"), "wrong", "not-number"),
    ("flesch", _FL_PER_100, "wrong", "per-100"),
    ("flesch", _FL_SWAPPED, "wrong", "swapped"),
    ("flesch", _FL_SENTENCES_UP, "wrong", "sentences-up"),
    ("flesch", _FL_WORDS_UP, "wrong", "words-up"),
    ("flesch", _FL_PLUS, "wrong", "plus"),
    ("flesch", _FL_DOUMA, "wrong", "douma"),
    ("flesch", _FL_ROUNDED, "wrong", "rounded"),
    ("flesch", _fl("round(206.835 - 1.015 * words / sentences - 84.6 * syllables / words, 3)"), "wrong", "rounded"),
    ("flesch", _fl("int(206.835 - 1.015 * words / sentences - 84.6 * syllables / words)"), "wrong", "rounded"),
    ("flesch", _fl("206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)"), "correct", None),
    ("flesch", _FL_HEAD + "    per_sentence = words / sentences\n    per_word = syllables / words\n"
     "    return 206.835 - 1.015 * per_sentence - 84.6 * per_word\n", "correct", None),
    ("changes", {}, "empty", None),
    ("changes", dict(EXERCISES["changes"].solution), "correct", None),
    ("changes", {"split": "up", "order": "same"}, "wrong", "unfinished"),
] + [("changes", {item: "same" if group == "up" else "up"}, "wrong", f"wrong:{item}")
     for item, group in EXERCISES["changes"].solution.items()]
