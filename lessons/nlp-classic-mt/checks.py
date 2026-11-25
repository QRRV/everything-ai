# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-classic-mt: translating Dutch into English word for word with a dictionary
(word_for_word); comparing a translation with the translations that people wrote, as the same, the same words in
another order, or other words (outcome); and one rule that moves a verb at the end of a Dutch sentence to
right after its helping verb, as English wants it (verb_after_helper).

Both the English and the Dutch lesson translate from Dutch into English: the result of word for word reads as a gloss
for both. The sentences are those of Tatoeba (https://tatoeba.org, CC BY 2.0 FR): the Dutch ones of
data/tatoeba-nl.tsv.gz, with every English translation of them in data/tatoeba-nl-en.tsv.gz, both from the export of
November 21, 2026. The dictionary is the course's own (data/course-dictionary-nl-en.tsv, CC-BY-SA-4.0): 478 common
Dutch words, each with one English word, and whether it is a helping verb or a verb form that can stand at the end.
The lesson measures on the sentences whose every word is in the dictionary, and leaves out those with an English
translation that holds he or she: the dictionary has no Dutch pronoun for them, and the course shows none. Every number
in the text and the pictures is computed by the course (tests/test_nlp_classic_mt.py computes them again)."""
import functools
import gzip
import html
import re

from everything_ai import books, paths, yours
from everything_ai.check import FunctionBox, Mistake

DICTIONARY_FILE = paths.ROOT / "data" / "course-dictionary-nl-en.tsv"
TRANSLATIONS_FILE = paths.ROOT / "data" / "tatoeba-nl-en.tsv.gz"

_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*|\d+")
_PRONOUN = re.compile(r"\b(he|she|him|her|his|hers|himself|herself)\b", re.I)

# English contractions, written out the way the dictionary's words come out: *I'm* is *i am*.
CONTRACTIONS = {
    "i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are", "it's": "it is", "that's": "that is",
    "there's": "there is", "what's": "what is", "where's": "where is", "who's": "who is", "how's": "how is",
    "here's": "here is", "let's": "let us", "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "can not", "cannot": "can not", "won't": "will not", "isn't": "is not", "aren't": "are not",
    "wasn't": "was not", "weren't": "were not", "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "wouldn't": "would not", "shouldn't": "should not", "couldn't": "could not", "mustn't": "must not",
    "needn't": "need not", "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
    "i'll": "i will", "you'll": "you will", "we'll": "we will", "it'll": "it will", "they'll": "they will",
    "i'd": "i would", "you'd": "you would", "we'd": "we would", "they'd": "they would",
}

WORKED = 6133069      # Ik zal de volgende bus nemen. / I'll take the next bus.
IDIOM = 384557        # Het regent pijpenstelen. / It's raining cats and dogs. (not every word is in the dictionary)


def words(text: str) -> list:
    """The words of a text in small letters, without punctuation; a word with an apostrophe inside stays one word."""
    return _WORD.findall(text.lower().replace("’", "'"))


def english_words(text: str) -> list:
    """The words of an English text as words() gives them, with the contractions written out: *I'll* is *i will*."""
    return " ".join(CONTRACTIONS.get(w, w) for w in words(text)).split()


@functools.cache
def _dictionary() -> tuple:
    with open(DICTIONARY_FILE, encoding="utf-8") as f:
        lines = f.read().split("\n")
    assert lines[0] == "dutch\tenglish\thelper\tverb"
    rows = [line.split("\t") for line in lines[1:] if line]
    return tuple((d, e, h == "1", v == "1") for d, e, h, v in rows)


DICTIONARY = {d: e for d, e, _, _ in _dictionary()}
HELPERS = {d for d, _, h, _ in _dictionary() if h}
VERBS = {d for d, _, _, v in _dictionary() if v}


# --- the course's own answers to the exercises ---------------------------------------------------------------------------

def word_for_word(words, dictionary):
    english = []
    for w in words:
        if w in dictionary:
            english.append(dictionary[w])
        else:
            english.append(w)
    return english


def outcome(translation, references):
    if translation in references:
        return "same"
    for r in references:
        if sorted(r) == sorted(translation):
            return "order"
    return "words"


def verb_after_helper(words, helpers, verbs):
    last = words[-1]
    if last not in verbs:
        return words
    for i in range(len(words) - 1):
        if words[i] in helpers:
            return words[:i + 1] + [last] + words[i + 1:-1]
    return words


def translate(text: str, rule: bool = False) -> list:
    """The English words of a Dutch text, word for word, after the rule of exercise 3 when rule is true."""
    ws = words(text)
    if rule and ws:
        ws = verb_after_helper(ws, HELPERS, VERBS)
    return word_for_word(ws, DICTIONARY)


# --- the sentences of Tatoeba ------------------------------------------------------------------------------------------

@functools.cache
def _pairs() -> tuple:
    """(id, Dutch text, English translations) for every Dutch sentence of Tatoeba with a translation, by id."""
    dutch = dict(books.dutch_sentences())
    english = {}
    with gzip.open(TRANSLATIONS_FILE, "rt", encoding="utf-8") as f:
        lines = f.read().split("\n")
    assert lines[0] == "nl_id\ten_id\ttext"
    for line in lines[1:]:
        if line:
            a, _, text = line.split("\t")
            english.setdefault(int(a), []).append(text)
    return tuple((sid, dutch[sid], tuple(english[sid])) for sid in sorted(english))


def _covers(text: str) -> bool:
    ws = words(text)
    return bool(ws) and all(w in DICTIONARY for w in ws)


@functools.cache
def _covered() -> tuple:
    """The sentences the lesson measures on: every word in the dictionary, and no he or she in a translation."""
    return tuple(p for p in _pairs() if _covers(p[1]) and not any(_PRONOUN.search(e) for e in p[2]))


def pairs() -> list:
    """The sentences the lesson measures on, as [id, Dutch text, [English translations]], as a new list each time."""
    return [[sid, text, list(es)] for sid, text, es in _covered()]


@functools.cache
def _covered_words() -> tuple:
    """For each sentence of _covered(): its Dutch words, and the English words of each translation."""
    return tuple((tuple(words(text)), tuple(tuple(english_words(e)) for e in es)) for _, text, es in _covered())


def dutch_words() -> list:
    """The Dutch words of each sentence the lesson measures on, as a new list each time."""
    return [list(ws) for ws, _ in _covered_words()]


def references() -> list:
    """The English translations of each sentence the lesson measures on, each as its list of words, a new list each
    time."""
    return [[list(r) for r in refs] for _, refs in _covered_words()]


def pair(sid: int) -> tuple:
    """(Dutch text, English translations) of the Tatoeba sentence with this id."""
    for s, text, es in _pairs():
        if s == sid:
            return text, list(es)
    raise KeyError(sid)


def _outcomes(translations: list) -> list:
    return [outcome(list(t), [list(r) for r in refs]) for t, (_, refs) in zip(translations, _covered_words())]


@functools.cache
def _direct() -> tuple:
    return tuple(tuple(word_for_word(list(ws), DICTIONARY)) for ws, _ in _covered_words())


@functools.cache
def _ruled() -> tuple:
    return tuple(tuple(word_for_word(verb_after_helper(list(ws), HELPERS, VERBS), DICTIONARY))
                 for ws, _ in _covered_words())


# --- the numbers the text names ----------------------------------------------------------------------------------------

@functools.cache
def numbers() -> dict:
    """The numbers the lesson's text names."""
    direct = _outcomes(_direct())
    ruled = _outcomes(_ruled())
    moved = [list(ws) != verb_after_helper(list(ws), HELPERS, VERBS) for ws, _ in _covered_words()]
    out = {
        "dictionary": len(DICTIONARY), "helpers": len(HELPERS), "verbs": len(VERBS),
        "dutch": len(_pairs()), "translations": sum(len(es) for _, _, es in _pairs()),
        "covered": len(_covered()),
        "several": sum(len(es) > 1 for _, _, es in _covered()),
        "moved": sum(moved),
        "fixed": sum(a != "same" and b == "same" for a, b in zip(direct, ruled)),
        "broken": sum(a == "same" and b != "same" for a, b in zip(direct, ruled)),
    }
    for kind in ("same", "order", "words"):
        out[kind] = direct.count(kind)
        out[kind + "_rule"] = ruled.count(kind)
    n = out["covered"]
    for kind in ("same", "order", "words", "same_rule", "order_rule", "words_rule"):
        out[kind + "_pct"] = 100 * out[kind] / n
    return out


def direct() -> list:
    """The course's word-for-word translation of each sentence the lesson measures on, as a new list each time."""
    return [list(t) for t in _direct()]


def left_out() -> int:
    """How many sentences have every word in the dictionary, but a translation with he or she: the lesson leaves them
    out."""
    return sum(1 for p in _pairs() if _covers(p[1])) - len(_covered())


# Sentences the pictures show, by their id in Tatoeba.
TIRED = 384938        # Ik ben moe. / I'm tired.
SEE = 8123878         # Ik zie je. / I can see you., I see you.
HUNGRY = 439361       # Ik heb honger. / I'm hungry.
LIVE = 397297         # Waar woon je? / Where do you live?
SWIM = 2102718        # Ik kan snel zwemmen. / I can swim fast.
HELP = 1068548        # Kan ik helpen? / Can I help?
NOTHING = 9763021     # Ik heb niets te zeggen. / I have nothing to say.
WORDS_ROWS = (TIRED, WORKED, HUNGRY, LIVE, IDIOM)       # the picture under exercise 1
OUTCOME_ROWS = (TIRED, SEE, WORKED, HUNGRY, LIVE)       # the picture under exercise 2
RULE_ROWS = (WORKED, SWIM, HELP, NOTHING, TIRED)        # the picture under exercise 3
KINDS = ("same", "order", "words")


# --- styles and small parts of the pictures ------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);margin-top:0.45rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.84rem")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap"
_DATA = "color:var(--ea-data-text)"
_FAINT = "color:var(--ea-ink-2)"
_PRED = "color:var(--ea-pred-text);font-weight:600"
_LOSS = "color:var(--ea-loss-text);font-weight:600"
_CHIP = ("display:inline-block;margin:0.08rem 0.2rem 0.08rem 0;padding:0 0.3rem;border-radius:3px;"
         "font-family:var(--ea-font-code);font-size:0.78rem;line-height:1.5")
_CHIP_PRED = _CHIP + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_CHIP_DATA = _CHIP + ";background:var(--ea-data-wash);color:var(--ea-data-text)"
_CHIP_LOSS = _CHIP + ";background:var(--ea-loss-wash);color:var(--ea-loss-text);font-weight:600"
_CHIP_NONE = _CHIP + ";border:1px dashed var(--ea-line);color:var(--ea-ink-3)"
_DASHED = ";border:1px dashed currentColor;padding:0 calc(0.3rem - 1px)"   # a word that is not in the dictionary


def _marked(text_: str) -> str:
    """Escaped text in which `name` is set as code and *word* in italics."""
    out = []
    for k, part in enumerate(html.escape(text_).split("`")):
        if k % 2:
            out.append(f"<code>{part}</code>")
        else:
            out.append("".join(f"<i>{p}</i>" if j % 2 else p for j, p in enumerate(part.split("*"))))
    return "".join(out)


def _note(text_: str) -> str:
    """A line under a picture; a word between asterisks is set in italics and a name between backticks as code."""
    return f'<div style="{_UI}">{_marked(text_)}</div>'


def _label(text_: str) -> str:
    return f'<div style="{_LABEL}">{_marked(text_)}</div>'


def _num(n, lang: str) -> str:
    """A whole number with a separator for thousands: 1,234 in English, 1.234 in Dutch."""
    out = f"{round(n):,}"
    return out.replace(",", ".") if lang == "nl" else out


def _dec(x: float, lang: str, digits: int = 1) -> str:
    out = f"{round(float(x), digits) + 0.0:.{digits}f}"
    return out.replace(".", ",") if lang == "nl" else out


def _pct(part, whole, lang: str) -> str:
    return _dec(100 * part / whole if whole else 0.0, lang) + "%"


def _chip(text_: str, style: str) -> str:
    return f'<span style="{style}">{html.escape(text_)}</span>'


def _gray(text_: str) -> str:
    return f'<span style="{_DATA}">{html.escape(text_)}</span>'


def _faint(text_: str) -> str:
    return f'<span style="{_FAINT}">{html.escape(text_)}</span>'


def _table(rows_: list, head: list, numbers_: tuple = ()) -> str:
    """A small table; rows hold HTML that is already escaped; numbers_ are the columns set to the right."""
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers_ else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers_ else ""}">{c}</td>'
                                    for j, c in enumerate(r)) + "</tr>" for r in rows_)
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem">'
            f"<thead><tr>{top}</tr></thead><tbody>{body}</tbody></table></div>")


def _dutch_chips(ws: list) -> str:
    """Dutch words as gray chips; a word that is not in the dictionary has a dashed border."""
    return "".join(_chip(w, _CHIP_DATA + ("" if w in DICTIONARY else _DASHED)) for w in ws)


def _english_chips(ws: list, dutch: list, style: str = _CHIP_PRED) -> str:
    """English words as chips in the given style; a word that stayed Dutch, as it is not in the dictionary, has a dashed
    border."""
    kept = {w for w in dutch if w not in DICTIONARY}
    return "".join(_chip(str(w), style + (_DASHED if w in kept else "")) for w in ws)


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error
_SLOW = object()      # what _call gives for a function that ran out of time
PICTURE_SECONDS = 2   # what the learner's function gets for all the calls of one picture together


def _call(make, seconds: float = PICTURE_SECONDS):
    """make() within a time limit, with what it prints thrown away; _SLOW when the time runs out, _STOPPED when it
    stops with an error."""
    import contextlib
    import io
    from everything_ai.check import Timeout, with_time_limit
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(make, max(seconds, 0.05))
    except Timeout:
        return _SLOW
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _is_words(v) -> bool:
    return isinstance(v, list) and all(isinstance(w, str) for w in v)


def _empty(w: dict, key: str) -> str:
    """The picture that stays empty, with why: the learner's function stopped, was too slow, or gave something odd."""
    text_ = w[key].format(s=PICTURE_SECONDS, kind=w.get("kind", ""))
    return yours.figure(w["title"], _note(text_), alt=text_)


_WHY_EMPTY = {
    "en": {"stopped": "Your function stops with an error, so there is nothing to draw yet. The check above says why.",
           "slow": "Your function took longer than {s} seconds, so there is nothing to draw yet.",
           "odd": "Your function gives something other than {kind}, so there is nothing to draw yet. The check above says what."},
    "nl": {"stopped": "Je functie stopt met een fout, dus er valt nog niets te tekenen. De check hierboven zegt waarom.",
           "slow": "Je functie deed er langer over dan {s} seconden, dus er valt nog niets te tekenen.",
           "odd": "Je functie geeft iets anders dan {kind}, dus er valt nog niets te tekenen. De check hierboven zegt wat."},
}

LANGS = ("en", "nl")


# --- the idea: the Vauquois triangle, and the tree of the worked example -------------------------------------------------

_IDEA_TEXT = {
    "en": {"title": "Three ways to translate: the Vauquois triangle",
           "source": "Dutch text", "target": "English text", "source_tree": "Dutch tree", "target_tree": "English tree",
           "top": "interlingua: the meaning, in no language", "direct": "direct: word for word",
           "transfer": "transfer: tree to tree", "analysis": "analysis", "generation": "generation",
           "tree": "The tree of the worked example: *zal* hangs from *nemen*, with `aux`.",
           "tree_alt": "Dependency tree of ik zal de volgende bus nemen: nemen is the root; ik hangs from it with nsubj, zal with aux, bus with obj; de and volgende hang from bus with det and amod.",
           "rows": ("The Dutch words", "Direct, word for word", "After the rule: *nemen* moves to right after *zal*"),
           "note": "Gray: the Dutch text, what goes in. Blue: the English text, what comes out. The thick arc, `aux`, links the two words that the rule of exercise 3 puts side by side."},
    "nl": {"title": "Drie manieren om te vertalen: de driehoek van Vauquois",
           "source": "Nederlandse tekst", "target": "Engelse tekst", "source_tree": "Nederlandse boom",
           "target_tree": "Engelse boom", "top": "interlingua: de betekenis, in geen enkele taal",
           "direct": "direct: woord voor woord", "transfer": "transfer: van boom naar boom", "analysis": "analyse",
           "generation": "generatie",
           "tree": "De boom van het uitgewerkte voorbeeld: *zal* hangt aan *nemen*, met `aux`.",
           "tree_alt": "Dependency tree van ik zal de volgende bus nemen: nemen is de root; ik hangt eraan met nsubj, zal met aux, bus met obj; de en volgende hangen aan bus met det en amod.",
           "rows": ("De Nederlandse woorden", "Direct, woord voor woord", "Na de regel: *nemen* schuift naar de plek direct na *zal*"),
           "note": "Grijs: de Nederlandse tekst, wat erin gaat. Blauw: de Engelse tekst, wat eruit komt. De dikke boog, `aux`, verbindt de twee woorden die de regel van oefening 3 naast elkaar zet."},
}

# The tree of the worked example in Universal Dependencies: (word, head, relation), heads counted from 1.
WORKED_TREE = (("ik", 6, "nsubj"), ("zal", 6, "aux"), ("de", 5, "det"), ("volgende", 5, "amod"), ("bus", 6, "obj"),
               ("nemen", 0, "root"))


def _head(x: float, y: float, dx: float, dy: float, color: str) -> str:
    """An arrowhead with its tip at (x, y), pointing along (dx, dy)."""
    import math
    n = math.hypot(dx, dy) or 1.0
    ux, uy = dx / n, dy / n
    bx, by = x - 8 * ux, y - 8 * uy
    return (f'<polygon points="{x:.1f},{y:.1f} {bx - 4 * uy:.1f},{by + 4 * ux:.1f} {bx + 4 * uy:.1f},{by - 4 * ux:.1f}" '
            f'style="fill:{color}"/>')


def triangle_svg(lang: str) -> str:
    """The Vauquois triangle: direct translation along the bottom, transfer from tree to tree in the middle, and an
    interlingua at the top; analysis climbs the Dutch side, generation goes down the English side."""
    import math
    w = _IDEA_TEXT[lang]
    left, right, bottom, apex_x, apex_y, mid = 100.0, 500.0, 226.0, 300.0, 40.0, 136.0
    up = (bottom - mid) / (bottom - apex_y)
    lx, rx = left + (apex_x - left) * up, right - (right - apex_x) * up
    ink, line = "var(--ea-ink-2)", "var(--ea-line)"
    data, pred = "var(--ea-data-text)", "var(--ea-pred-text)"
    font = "font-family:var(--ea-font-ui)"
    angle = math.degrees(math.atan2(bottom - apex_y, apex_x - left))

    def text(x, y, s, color, size=12, anchor="middle", extra=""):
        return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" style="{font};font-size:{size}px;fill:{color}'
                f'{extra}">{html.escape(s)}</text>')

    parts = [f'<path d="M{left},{bottom} L{apex_x},{apex_y} L{right},{bottom}" style="fill:none;stroke:{line};stroke-width:1.5"/>']
    # analysis up the left side, generation down the right side
    for x0, y0, x1, y1 in ((left, bottom, apex_x, apex_y), (apex_x, apex_y, right, bottom)):
        tx, ty = x0 + 0.78 * (x1 - x0), y0 + 0.78 * (y1 - y0)
        parts.append(_head(tx, ty, x1 - x0, y1 - y0, line))
    for x, y, s, rot in ((left + 0.2 * (apex_x - left) - 9, bottom - 0.2 * (bottom - apex_y) - 9, w["analysis"], -angle),
                         (right - 0.2 * (right - apex_x) + 9, bottom - 0.2 * (bottom - apex_y) - 9, w["generation"], angle)):
        parts.append(f'<g transform="rotate({rot:.1f} {x:.1f} {y:.1f})">{text(x, y, s, ink, 11.5, extra=";font-style:italic")}</g>')
    # direct along the bottom, transfer in the middle
    for x0, x1, y, s in ((left + 10, right - 10, bottom, w["direct"]), (lx + 10, rx - 10, mid, w["transfer"])):
        parts.append(f'<path d="M{x0:.1f},{y:.1f} L{x1 - 6:.1f},{y:.1f}" style="fill:none;stroke:{ink};stroke-width:1.6"/>')
        parts.append(_head(x1, y, 1, 0, ink))
        parts.append(text((x0 + x1) / 2, y - 7, s, ink, 12.5))
    # the corners and the trees
    for x, y, color in ((left, bottom, data), (right, bottom, pred), (lx, mid, data), (rx, mid, pred), (apex_x, apex_y, ink)):
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" style="fill:{color}"/>')
    parts += [text(left, bottom + 19, w["source"], data, 12.5, extra=";font-weight:600"),
              text(left, bottom + 35, "ik zal de volgende bus nemen", data, 11, extra=";font-style:italic"),
              text(right, bottom + 19, w["target"], pred, 12.5, extra=";font-weight:600"),
              text(right, bottom + 35, "i will take the next bus", pred, 11, extra=";font-style:italic"),
              text(lx - 10, mid + 4, w["source_tree"], data, 12, "end", ";font-weight:600"),
              text(rx + 10, mid + 4, w["target_tree"], pred, 12, "start", ";font-weight:600"),
              text(apex_x, apex_y - 12, w["top"], ink, 12.5, extra=";font-weight:600")]
    alt = w["title"]
    return (f'<div style="overflow-x:auto"><svg role="img" aria-label="{html.escape(alt)}" viewBox="0 0 600 272" '
            f'width="600" height="272" style="max-width:none;display:block">{"".join(parts)}</svg></div>')


@functools.cache
def _dependencies():
    """The checks of nlp-dependencies, for its drawing of a dependency tree."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("everything_ai_checks_nlp_dependencies_for_classic_mt",
                                                  paths.LESSONS / "nlp-dependencies" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def idea_picture(lang: str) -> str:
    """The Vauquois triangle, then the tree of the worked example and the sentence word for word and after the rule."""
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang]
    tree_words = [t[0] for t in WORKED_TREE]
    arcs = [(i, h, rel, "bold" if rel == "aux" else "ink") for i, (_, h, rel) in enumerate(WORKED_TREE, start=1)]
    tree = _dependencies().arcs_svg(tree_words, arcs, w["tree_alt"])
    dutch = words(pair(WORKED)[0])
    rows_ = [(w["rows"][0], _dutch_chips(dutch)),
             (w["rows"][1], _english_chips(translate(pair(WORKED)[0]), dutch)),
             (w["rows"][2], _english_chips(translate(pair(WORKED)[0], rule=True), dutch))]
    body = (triangle_svg(lang) + _label(w["tree"]) + tree
            + "".join(_label(label) + f"<div>{chips}</div>" for label, chips in rows_) + _note(w["note"]))
    return yours.figure(w["title"], body, alt=w["title"])


# --- the box: a sentence the learner types ---------------------------------------------------------------------------------

_BOX_TEXT = {
    "en": {"title": "Your sentence, translated by rules",
           "none": "Type a Dutch sentence in the box above.",
           "dutch": "The Dutch words:",
           "direct": "Direct, word for word:",
           "rule": "With the rule of exercise 3:",
           "moved": "*{v}* moves to right after *{h}*.",
           "no_verb": "The rule moves nothing: the last word, *{v}*, is not in `C.VERBS`.",
           "no_helper": "The rule moves nothing: no word before *{v}* is in `C.HELPERS`.",
           "already": "The rule moves nothing: *{v}* already stands right after *{h}*.",
           "people": "People of Tatoeba translate this sentence as:",
           "outcome": "Word for word gives `\"{a}\"`, with the rule `\"{b}\"`.",
           "note": "Gray: the Dutch words and the translations by people. Dashed: a word that is not in the dictionary, which stays as it is. Blue: the English words that come out."},
    "nl": {"title": "Je zin, vertaald met regels",
           "none": "Typ een Nederlandse zin in het vak hierboven.",
           "dutch": "De Nederlandse woorden:",
           "direct": "Direct, woord voor woord:",
           "rule": "Met de regel van oefening 3:",
           "moved": "*{v}* schuift naar de plek direct na *{h}*.",
           "no_verb": "De regel verplaatst niets: het laatste woord, *{v}*, staat niet in `C.VERBS`.",
           "no_helper": "De regel verplaatst niets: eerder in de zin dan *{v}* staat geen woord uit `C.HELPERS`.",
           "already": "De regel verplaatst niets: *{v}* staat al direct na *{h}*.",
           "people": "Mensen van Tatoeba vertalen deze zin zo:",
           "outcome": "Woord voor woord geeft `\"{a}\"`, met de regel `\"{b}\"`.",
           "note": "Grijs: de Nederlandse woorden en de vertalingen door mensen. Gestippeld: een woord dat niet in het woordenboek staat en blijft zoals het is. Blauw: de Engelse woorden die eruit komen."},
}


@functools.cache
def _where() -> dict:
    """For the words of each sentence the lesson measures on: its place in _covered(), the first when two are alike."""
    out = {}
    for k, (ws, _) in enumerate(_covered_words()):
        out.setdefault(ws, k)
    return out


def box_picture(lang: str, typed: str) -> str:
    """The box: the words of what the learner typed, word for word and after the rule of exercise 3; for a sentence of
    Tatoeba that the lesson measures on, also its translations by people and the outcome of both."""
    lang = lang if lang in LANGS else "en"
    w = _BOX_TEXT[lang]
    dutch = words((typed or "")[:300])
    if not dutch:
        return yours.figure(w["title"], _note(w["none"]), alt=w["none"])
    moved = verb_after_helper(dutch, HELPERS, VERBS)
    last = dutch[-1]
    if moved != dutch:
        h = next(x for x in dutch[:-1] if x in HELPERS)
        why = w["moved"].format(v=last, h=h)
    elif last not in VERBS:
        why = w["no_verb"].format(v=last)
    else:
        h = next((x for x in dutch[:-1] if x in HELPERS), None)
        why = w["already"].format(v=last, h=h) if h else w["no_helper"].format(v=last)
    plain, ruled = word_for_word(dutch, DICTIONARY), word_for_word(moved, DICTIONARY)
    body = (_label(w["dutch"]) + f"<div>{_dutch_chips(dutch)}</div>"
            + _label(w["direct"]) + f"<div>{_english_chips(plain, dutch)}</div>"
            + _label(w["rule"]) + f"<div>{_english_chips(ruled, dutch)}</div>" + _note(why))
    k = _where().get(tuple(dutch))
    if k is not None:
        refs = [list(r) for r in _covered_words()[k][1]]
        texts = "".join(f"<div>{_gray(e)}</div>" for e in _covered()[k][2])
        body += (_label(w["people"]) + texts
                 + _note(w["outcome"].format(a=outcome(plain, refs), b=outcome(ruled, refs))))
    return yours.figure(w["title"], body + _note(w["note"]), alt=w["title"])


# --- the picture under exercise 1: word for word on sentences of Tatoeba ------------------------------------------------

_WORDS_TEXT = {
    "en": {"title": "Your word_for_word on sentences of Tatoeba",
           "head": ["Dutch", "Your word_for_word", "A translation by people"],
           "course": "the course's word_for_word gives *{want}*",
           "summary": "Of the {n} sentences the lesson measures on, your word_for_word turns {k} into exactly a translation that people wrote, {pct}.",
           "differs": "With the course's word_for_word: {want}.",
           "note": "Gray: the Dutch sentences and a translation by people. Dashed: a word that is not in the dictionary. Blue: what your word_for_word gives, when it is what the course's word_for_word gives. Purple: what differs from the course's word_for_word.",
           "kind": "a list of words"},
    "nl": {"title": "Je word_for_word op zinnen van Tatoeba",
           "head": ["Nederlands", "Je word_for_word", "Een vertaling door mensen"],
           "course": "de word_for_word van de course geeft *{want}*",
           "summary": "Van de {n} zinnen waarop de les meet, maakt je word_for_word er {k} precies tot een vertaling die mensen schreven, {pct}.",
           "differs": "Met de word_for_word van de course: {want}.",
           "note": "Grijs: de Nederlandse zinnen en een vertaling door mensen. Gestippeld: een woord dat niet in het woordenboek staat. Blauw: wat je word_for_word geeft, als dat is wat de word_for_word van de course geeft. Paars: wat anders is dan bij de word_for_word van de course.",
           "kind": "een lijst met woorden"},
}


def words_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _WORDS_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    shown_ = [words(pair(sid)[0]) for sid in WORDS_ROWS]
    every = _covered_words()

    def make():
        d = dict(DICTIONARY)
        return [fn(list(ws), d) for ws in shown_], [fn(list(ws), d) for ws, _ in every]

    mine = _call(make)
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    rows_mine, all_mine = mine
    if not all(_is_words(m) for m in rows_mine + all_mine):
        return _empty(w, "odd")
    rows_ = []
    for sid, ws, m in zip(WORDS_ROWS, shown_, rows_mine):
        want = word_for_word(ws, DICTIONARY)
        if m == want:
            cell = _english_chips(m, ws)
        else:
            cell = _english_chips(m, ws, _CHIP_LOSS) + _note(w["course"].format(want=" ".join(want)))
        rows_.append([_gray(pair(sid)[0]) + f"<div>{_dutch_chips(ws)}</div>", cell, _gray(pair(sid)[1][0])])
    n = len(every)
    k = sum(1 for m, (_, refs) in zip(all_mine, every) if tuple(m) in refs)
    summary = w["summary"].format(n=_num(n, lang), k=_num(k, lang), pct=_pct(k, n, lang))
    if any(tuple(m) != t for m, t in zip(all_mine, _direct())):
        summary += " " + w["differs"].format(want=_num(numbers()["same"], lang))
    body = _table(rows_, w["head"]) + f'<div style="{_UI}">{html.escape(summary)}</div>' + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


# --- the picture under exercise 2: the outcome of word for word ----------------------------------------------------------

_OUTCOME_TEXT = {
    "en": {"title": "Your outcome on the word-for-word translations",
           "head": ["Dutch", "Word for word", "Translations by people", "Your outcome"],
           "course": "the course's outcome gives `\"{want}\"`",
           "counts": "Your outcome over all {n} sentences the lesson measures on:",
           "count_head": ["Outcome", "Sentences", "Share", ""],
           "bar_course": "the course's outcome: {want}",
           "note": "Gray: the sentences, their translations by people, and the word-for-word translations. Blue: what your outcome gives, and a count, when it is what the course's outcome gives. Purple: what differs from the course's outcome.",
           "kind": "one of the texts \"same\", \"order\" and \"words\""},
    "nl": {"title": "Je outcome op de vertalingen woord voor woord",
           "head": ["Nederlands", "Woord voor woord", "Vertalingen door mensen", "Je outcome"],
           "course": "de outcome van de course geeft `\"{want}\"`",
           "counts": "Je outcome over alle {n} zinnen waarop de les meet:",
           "count_head": ["Uitkomst", "Zinnen", "Deel", ""],
           "bar_course": "de outcome van de course: {want}",
           "note": "Grijs: de zinnen, de vertalingen door mensen, en de vertalingen woord voor woord. Blauw: wat je outcome geeft, en een aantal, als dat is wat de outcome van de course geeft. Paars: wat anders is dan bij de outcome van de course.",
           "kind": "een van de teksten \"same\", \"order\" en \"words\""},
}


def _bar(share: float, color: str) -> str:
    """A bar for a share from 0 to 1, on a track of 10rem."""
    width = max(0.0, min(1.0, share)) * 100
    return (f'<div style="width:10rem;height:0.7rem;margin-top:0.3rem;border-radius:2px;background:var(--ea-data-wash)">'
            f'<div style="width:{width:.1f}%;height:100%;border-radius:2px;background:{color}"></div></div>')


def _counts(got: dict, want: dict, n: int, lang: str, course_text: str) -> list:
    """Rows of a table of the three outcomes: the count in blue when it is the course's, else in purple with the
    course's count under it."""
    rows_ = []
    for kind in KINDS:
        same = got[kind] == want[kind]
        style, color = (_PRED, "var(--ea-pred)") if same else (_LOSS, "var(--ea-loss)")
        cell = f'<span style="{style}">{_num(got[kind], lang)}</span>'
        if not same:
            cell += _note(course_text.format(want=_num(want[kind], lang)))
        rows_.append([_chip(f'"{kind}"', _CHIP_DATA), cell, _faint(_pct(got[kind], n, lang)), _bar(got[kind] / n, color)])
    return rows_


def outcome_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _OUTCOME_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    places = [_where()[tuple(words(pair(sid)[0]))] for sid in OUTCOME_ROWS]
    every = _covered_words()
    direct_ = _direct()

    def make():
        return [fn(list(direct_[k]), [list(r) for r in every[k][1]]) for k in places], \
               [fn(list(t), [list(r) for r in refs]) for t, (_, refs) in zip(direct_, every)]

    mine = _call(make)
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    rows_mine, all_mine = mine
    if not all(isinstance(m, str) and m in KINDS for m in rows_mine + all_mine):
        return _empty(w, "odd")
    rows_ = []
    for sid, k, m in zip(OUTCOME_ROWS, places, rows_mine):
        want = outcome(list(direct_[k]), [list(r) for r in every[k][1]])
        cell = _chip(f'"{m}"', _CHIP_PRED if m == want else _CHIP_LOSS)
        if m != want:
            cell += _note(w["course"].format(want=want))
        rows_.append([_gray(pair(sid)[0]), _chip(" ".join(direct_[k]), _CHIP_DATA),
                      "".join(f"<div>{_gray(e)}</div>" for e in pair(sid)[1]), cell])
    n = len(every)
    got = {kind: all_mine.count(kind) for kind in KINDS}
    body = (_table(rows_, w["head"]) + _label(w["counts"].format(n=_num(n, lang)))
            + _table(_counts(got, {kind: numbers()[kind] for kind in KINDS}, n, lang, w["bar_course"]),
                     w["count_head"], (1, 2))
            + _note(w["note"]))
    return yours.figure(w["title"], body, alt=w["title"])


# --- the picture under exercise 3: one rule that moves a verb ------------------------------------------------------------

_RULE_TEXT = {
    "en": {"title": "Your verb_after_helper on sentences of Tatoeba",
           "head": ["Dutch", "After your verb_after_helper", "In English", "Without → with the rule"],
           "course": "the course's verb_after_helper gives *{want}*",
           "summary": "Your rule changes the order of {moved} of the {n} sentences. {fixed} that word for word got wrong now come out exactly as a translation by people, and {broken} that word for word got right no longer do.",
           "differs": "The course's rule changes {moved}, puts right {fixed} and breaks {broken}.",
           "count_head": ["Outcome", "Word for word", "With your rule", ""],
           "bar_course": "with the course's rule: {want}",
           "note": "Gray: the sentences, the outcomes, and the counts of word for word. Blue: what your verb_after_helper gives, its English, and a count, when it is what the course's verb_after_helper gives. Purple: what differs from the course's verb_after_helper.",
           "kind": "a list of words"},
    "nl": {"title": "Je verb_after_helper op zinnen van Tatoeba",
           "head": ["Nederlands", "Na je verb_after_helper", "In het Engels", "Zonder → met de regel"],
           "course": "de verb_after_helper van de course geeft *{want}*",
           "summary": "Je regel verandert de volgorde van {moved} van de {n} zinnen. {fixed} zinnen die woord voor woord fout gingen, komen er nu precies uit als een vertaling door mensen, en {broken} zinnen die woord voor woord goed gingen, niet meer.",
           "differs": "De regel van de course verandert er {moved}, zet er {fixed} goed en maakt er {broken} fout.",
           "count_head": ["Uitkomst", "Woord voor woord", "Met je regel", ""],
           "bar_course": "met de regel van de course: {want}",
           "note": "Grijs: de zinnen, de uitkomsten, en de aantallen bij woord voor woord. Blauw: wat je verb_after_helper geeft, het Engels daarvan, en een aantal, als dat is wat de verb_after_helper van de course geeft. Paars: wat anders is dan bij de verb_after_helper van de course.",
           "kind": "een lijst met woorden"},
}


def rule_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _RULE_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    shown_ = [words(pair(sid)[0]) for sid in RULE_ROWS]
    every = _covered_words()

    def make():
        h, v = set(HELPERS), set(VERBS)
        return [fn(list(ws), h, v) for ws in shown_], [fn(list(ws), h, v) for ws, _ in every]

    mine = _call(make)
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    rows_mine, all_mine = mine
    if not all(_is_words(m) for m in rows_mine + all_mine):
        return _empty(w, "odd")
    rows_ = []
    for sid, ws, m in zip(RULE_ROWS, shown_, rows_mine):
        want = verb_after_helper(list(ws), HELPERS, VERBS)
        refs = [english_words(e) for e in pair(sid)[1]]
        if m == want:
            cell = "".join(_chip(x, _CHIP_PRED) for x in m)
        else:
            cell = "".join(_chip(x, _CHIP_LOSS) for x in m) + _note(w["course"].format(want=" ".join(want)))
        before, after = outcome(word_for_word(ws, DICTIONARY), refs), outcome(word_for_word(m, DICTIONARY), refs)
        english = " ".join(word_for_word(m, DICTIONARY))
        rows_.append([_gray(pair(sid)[0]), cell,
                      f'<span style="color:var(--ea-{"pred" if m == want else "loss"}-text)">{html.escape(english)}</span>',
                      _faint(f'"{before}" → "{after}"')])
    n = len(every)
    direct_ = _outcomes(_direct())
    ruled = _outcomes([word_for_word(m, DICTIONARY) for m in all_mine])
    moved = sum(1 for m, (ws, _) in zip(all_mine, every) if tuple(m) != ws)
    fixed = sum(1 for a, b in zip(direct_, ruled) if a != "same" and b == "same")
    broken = sum(1 for a, b in zip(direct_, ruled) if a == "same" and b != "same")
    num = numbers()
    summary = w["summary"].format(moved=_num(moved, lang), n=_num(n, lang), fixed=_num(fixed, lang),
                                  broken=_num(broken, lang))
    if (moved, fixed, broken) != (num["moved"], num["fixed"], num["broken"]):
        summary += " " + w["differs"].format(moved=_num(num["moved"], lang), fixed=_num(num["fixed"], lang),
                                             broken=_num(num["broken"], lang))
    got = {kind: ruled.count(kind) for kind in KINDS}
    counts = _counts(got, {kind: num[kind + "_rule"] for kind in KINDS}, n, lang, w["bar_course"])
    for row, kind in zip(counts, KINDS):
        row.insert(1, _gray(_num(num[kind], lang)))
        del row[3]   # the share: the bar shows it
    body = (_table(rows_, w["head"]) + f'<div style="{_UI}">{html.escape(summary)}</div>'
            + _table(counts, w["count_head"], (1, 2)) + _note(w["note"]))
    return yours.figure(w["title"], body, alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------------

class _Val:
    """A test value that the check's messages write by its name, such as `C.DICTIONARY`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value, memo)


def _value(v) -> str:
    """A value the way code writes it, with double quotes; a set with its items in order."""
    if isinstance(v, bool) or v is None:
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    if isinstance(v, tuple):
        return "(" + ", ".join(_value(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    if isinstance(v, list):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    if isinstance(v, (set, frozenset)):
        return "{" + ", ".join(sorted(_value(x) for x in v)) + "}" if v else "set()"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_value(k)}: {_value(x)}" for k, x in v.items()) + "}"
    return repr(v)


def _show(v):
    """A value for a message, the way the lesson writes it."""
    return _value(v)


def _written(v) -> _Val:
    """A small test value that the messages write the way the lesson writes it, with double quotes."""
    return _Val(_value(v), v)


def _words_val(text_: str) -> _Val:
    return _Val(f"C.words({_value(text_)})", words(text_))


_DICTIONARY_VAL = _Val("C.DICTIONARY", DICTIONARY)
_HELPERS_VAL = _Val("C.HELPERS", HELPERS)
_VERBS_VAL = _Val("C.VERBS", VERBS)


def _place(sid: int) -> int:
    return _where()[tuple(words(pair(sid)[0]))]


def _direct_val(sid: int) -> _Val:
    k = _place(sid)
    return _Val(f"C.direct()[{k}]", list(_direct()[k]))


def _refs_val(sid: int) -> _Val:
    k = _place(sid)
    return _Val(f"C.references()[{k}]", [list(r) for r in _covered_words()[k][1]])


_DOTS = Mistake("dots-left", matches=lambda got, text_: any(line.split("#", 1)[0].count("...")
                                                            for line in text_.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang de `...` door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# The small cases the exercise texts show; the lessons in both languages translate from Dutch into English.
TOY_WORDS = ((["ik", "zie", "een", "kat"], {"ik": "i", "zie": "see", "een": "a"}),
             (["de", "hond", "slaapt"], {"de": "the", "hond": "dog"}))
TOY_OUTCOME = ((["the", "dog", "sleeps"], [["the", "dog", "is", "sleeping"], ["the", "dog", "sleeps"]]),
               (["i", "will", "the", "bus", "take"], [["i", "will", "take", "the", "bus"]]))
TOY_HELPERS, TOY_VERBS = {"heb", "kan"}, {"gelezen", "zwemmen"}
TOY_RULE = ((["ik", "heb", "het", "boek", "gelezen"], TOY_HELPERS, TOY_VERBS),
            (["ik", "kan", "snel", "zwemmen"], TOY_HELPERS, TOY_VERBS))

_WW_HEAD = "def word_for_word(words, dictionary):\n    english = []\n    for w in words:\n"
_WW_KNOWN = "            english.append(dictionary[w])\n"
_WW_OTHER = "            english.append(w)\n"


def _ww(known: str = _WW_KNOWN, other: str = _WW_OTHER) -> str:
    return _WW_HEAD + "        if w in dictionary:\n" + known + "        else:\n" + other + "    return english\n"


_WW = _ww()
_WW_START = _ww("            ...\n", "            ...\n")
_WW_DROPPED = _ww(other="            pass\n")
_WW_KEPT = _ww(known=_WW_OTHER)
_WW_NO_IF = _WW_HEAD + "        english.append(dictionary[w])\n    return english\n"
_WW_SWAPPED = _ww(_WW_OTHER, _WW_KNOWN)
_WW_GET = _WW_HEAD + "        english.append(dictionary.get(w))\n    return english\n"
_WW_INSIDE = _WW_HEAD + "        if w in dictionary:\n" + _WW_KNOWN + "        else:\n" + _WW_OTHER + "        return english\n"

_OC_HEAD = "def outcome(translation, references):\n"
_OC_SAME = "    if translation in references:\n"
_OC_ORDER = "        if sorted(r) == sorted(translation):\n"


def _oc(same: str = _OC_SAME, order: str = _OC_ORDER) -> str:
    return (_OC_HEAD + same + '        return "same"\n    for r in references:\n' + order
            + '            return "order"\n    return "words"\n')


_OC = _oc()
_OC_START = _oc("    if ...:\n", "        if ...:\n")
_OC_FIRST = (_OC_HEAD + "    if translation == references[0]:\n        return \"same\"\n"
             "    if sorted(references[0]) == sorted(translation):\n        return \"order\"\n    return \"words\"\n")
_OC_SET = _oc(order="        if set(r) == set(translation):\n")
_OC_NO_SORT = _oc(order="        if r == translation:\n")
_OC_NOT_SAME = _oc(same="    if translation == references:\n")
_OC_LEN = _oc(order="        if len(r) == len(translation):\n")
_OC_SORT = _oc(order="        if r.sort() == translation.sort():\n")
_OC_HALF = _oc(order="        if sorted(r) == translation:\n")

_VH_HEAD = "def verb_after_helper(words, helpers, verbs):\n    last = words[-1]\n"
_VH_IF = "    if last not in verbs:\n"
_VH_RET = "            return words[:i + 1] + [last] + words[i + 1:-1]\n"


def _vh(check: str = _VH_IF, ret: str = _VH_RET) -> str:
    return (_VH_HEAD + check + "        return words\n    for i in range(len(words) - 1):\n"
            "        if words[i] in helpers:\n" + ret + "    return words\n")


_VH = _vh()
_VH_START = _vh("    if ...:\n", "            return ...\n")
_VH_END_KEPT = _vh(ret="            return words[:i + 1] + [last] + words[i + 1:]\n")
_VH_HELPER_TWICE = _vh(ret="            return words[:i + 1] + [last] + words[i:-1]\n")
_VH_BEFORE = _vh(ret="            return words[:i] + [last] + words[i:-1]\n")
_VH_LOST = _vh(ret="            return words[:i + 1] + [last]\n")
_VH_INVERTED = _vh("    if last in verbs:\n")
_VH_ANY_LAST = _vh("    if len(words) < 2:\n")


def _not_list(got, text_) -> bool:
    return not isinstance(got, list)


def _not_kind(got, text_) -> bool:
    return got not in KINDS


REQUIRED = ["word_for_word", "outcome", "verb_after_helper", "transfers"]

EXERCISES = {
    "word_for_word": FunctionBox(
        name="word_for_word",
        picture=words_picture,
        function="word_for_word",
        start=_WW_START,
        solution=_WW,
        show=_show,
        visible=[(_written(a), _written(b)) for a, b in TOY_WORDS],
        hidden=[(_written([]), _written({"ik": "i"})),
                (_written(["kat", "kat"]), _written({"kat": "cat"})),
                (_written(["tom", "is", "thuis"]), _written({"is": "is", "thuis": "home"})),
                (_words_val(pair(WORKED)[0]), _DICTIONARY_VAL),
                (_words_val(pair(IDIOM)[0]), _DICTIONARY_VAL),
                (_words_val(pair(LIVE)[0]), _DICTIONARY_VAL)],
        first=[_DOTS],
        mistakes=[
            Mistake("keyerror", error=lambda e: isinstance(e, KeyError),
                    en="`{call}` stops with a `KeyError`: your function looks up a word that is not a key of `dictionary`. Does your function look a word up only when it is a key of `dictionary`?",
                    nl="`{call}` stopt met een `KeyError`: je functie zoekt een woord op dat geen key van `dictionary` is. Zoekt je functie een woord alleen op als het een key van `dictionary` is?"),
            Mistake("not-list", matches=_not_list,
                    en="`{call}` gives {got}, and it should give {want}: a list of words, one for each word of `words`. Does your function give back the list `english`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een lijst met woorden, één voor elk woord van `words`. Geeft je functie de lijst `english` terug?"),
            Mistake("dropped", wrong=_WW_DROPPED,
                    en="`{call}` gives {got}, and it should give {want}: your function leaves out *kat*, which is not in `dictionary`. A word without a translation stays as it is: what goes in the list for it?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie laat *kat* weg, want *kat* staat niet in `dictionary`. Een woord zonder vertaling blijft zoals het is: wat komt er dan in de lijst?"),
            Mistake("kept", wrong=_WW_KEPT,
                    en="`{call}` gives {got}, and it should give {want}: the Dutch words, unchanged. For *ik*, which is a key of `dictionary`, what is its value?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de Nederlandse woorden, onveranderd. Wat is de waarde van *ik*, dat een key van `dictionary` is?"),
            Mistake("get-none", wrong=_WW_GET,
                    en="`{call}` gives {got}, and it should give {want}: `get` gives `None` for *kat*, which is not in `dictionary`. A word without a translation stays as it is: how does it get into the list?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `get` geeft `None` voor *kat*, want *kat* staat niet in `dictionary`. Een woord zonder vertaling blijft zoals het is: hoe komt het in de lijst?"),
            Mistake("return-inside", wrong=_WW_INSIDE,
                    en="`{call}` gives {got}, and it should give {want}: only the first word. Where does your `return` stand: inside the `for` loop, or after it?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: alleen het eerste woord. Waar staat je `return`: in de `for`-loop, of erna?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: for a word that is in `dictionary`, append its translation to `english`; for any other word, append the word itself.",
                      "nl": "Vervang de twee `...`: voeg voor een woord dat in `dictionary` staat de vertaling toe aan `english`, en voor elk ander woord het woord zelf."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. For each word: is it a key of `dictionary`? Then its value goes in the list, and else the word itself. Does your list hold one word for each word of `words`, in the same order?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Is elk woord een key van `dictionary`? Dan komt de waarde in de lijst, en anders het woord zelf. Bevat je lijst één woord voor elk woord van `words`, in dezelfde volgorde?"},
            "almost": {"en": "Right for the two small cases. But `{call}` gives {got}, and it should give {want}. Does your function give one word for each word of `words`, in the same order, also when a word comes twice or the list is empty?",
                       "nl": "Goed voor de twee kleine gevallen. Maar `{call}` geeft {got}, en het moet {want} zijn. Geeft je functie één woord voor elk woord van `words`, in dezelfde volgorde, ook als een woord twee keer staat of de lijst leeg is?"},
            "correct": {"en": "Right: your function translates word for word, as a pocket dictionary would. Which sentences does that get right, in the picture below, and which not?",
                        "nl": "Goed: je functie vertaalt woord voor woord, zoals met een zakwoordenboek. Welke zinnen gaan daarmee goed, in het plaatje hieronder, en welke niet?"},
        },
    ),
    "outcome": FunctionBox(
        name="outcome",
        picture=outcome_picture,
        function="outcome",
        start=_OC_START,
        solution=_OC,
        show=_show,
        visible=[(_written(a), _written(b)) for a, b in TOY_OUTCOME],
        hidden=[(_written(["i", "have", "hunger"]), _written([["i", "am", "hungry"], ["i", "feel", "hungry"]])),
                (_written(["the", "dog", "sees", "the", "cat"]), _written([["the", "cat", "sees", "dog"]])),
                (_written(["can", "help", "i"]), _written([["i", "can", "help", "you"], ["can", "i", "help"]])),
                (_written(["i", "love", "of", "you"]), _written([["i", "love", "you"]])),
                (_direct_val(TIRED), _refs_val(TIRED)),
                (_direct_val(SEE), _refs_val(SEE)),
                (_direct_val(WORKED), _refs_val(WORKED)),
                (_direct_val(LIVE), _refs_val(LIVE))],
        first=[_DOTS],
        mistakes=[
            Mistake("not-kind", matches=_not_kind,
                    en="`{call}` gives {got}, and it should give {want}: one of the three texts `\"same\"`, `\"order\"` and `\"words\"`. Does each `return` of the start code still give one of them?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een van de drie teksten `\"same\"`, `\"order\"` en `\"words\"`. Geeft elke `return` van de startcode er nog een?"),
            Mistake("first-only", wrong=_OC_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: your function compares only with the first translation, *the dog is sleeping*, and *the dog sleeps* is the second. Which translations of `references` should count?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie vergelijkt alleen met de eerste vertaling, *the dog is sleeping*, en *the dog sleeps* is de tweede. Welke vertalingen van `references` moeten tellen?"),
            Mistake("not-same", wrong=_OC_NOT_SAME,
                    en="`{call}` gives {got}, and it should give {want}: *the dog sleeps* is exactly the second translation of `references`, and your function does not count it as `\"same\"`. How do you ask whether `translation` is one of the lists in `references`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: *the dog sleeps* is precies de tweede vertaling van `references`, en je functie rekent het niet als `\"same\"`. Hoe vraag je of `translation` een van de lijsten in `references` is?"),
            Mistake("no-sort", wrong=_OC_NO_SORT,
                    en="`{call}` gives {got}, and it should give {want}: *i will the bus take* has the same words as *i will take the bus*, in another order, and two lists in another order are not equal. What puts the words of both lists in the same order before you compare them?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: *i will the bus take* heeft dezelfde woorden als *i will take the bus*, in een andere volgorde, en twee lijsten in een andere volgorde zijn niet gelijk. Wat zet de woorden van beide lijsten in dezelfde volgorde, zodat je de twee kunt vergelijken?"),
            Mistake("half-sorted", wrong=_OC_HALF,
                    en="`{call}` gives {got}, and it should give {want}: your function sorts the words of `r` and compares them with `translation` as it is, and the words of *i will the bus take* are not in sorted order. What do the words of `translation` need too, before you compare?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie sorteert de woorden van `r` en vergelijkt de gesorteerde woorden met `translation` zoals het is, en de woorden van *i will the bus take* staan niet gesorteerd. Wat moeten de woorden van `translation` ook, zodat je kunt vergelijken?"),
            Mistake("sort-method", wrong=_OC_SORT,
                    en="`{call}` gives {got}, and it should give {want}: `sort` sorts a list itself and gives back `None`, so your comparison is `None == None`, which is always `True`. Which function gives a new list, sorted?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `sort` sorteert een lijst zelf en geeft `None` terug, dus je vergelijking is `None == None`, en dat is altijd `True`. Welke functie geeft een nieuwe, gesorteerde lijst?"),
            Mistake("length", wrong=_OC_LEN,
                    en="`{call}` gives {got}, and it should give {want}: *i have hunger* has as many words as *i am hungry*, but other words. Which comparison says that two lists hold the same words?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: *i have hunger* heeft evenveel woorden als *i am hungry*, maar andere woorden. Welke vergelijking zegt dat twee lijsten dezelfde woorden bevatten?"),
            Mistake("set", wrong=_OC_SET,
                    en="`{call}` gives {got}, and it should give {want}: a set keeps each word once, so the two times *the* of your translation count as the one *the* of the other. What keeps every word, also one that comes twice?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een set houdt elk woord één keer, dus de twee keer *the* van je vertaling tellen als de ene *the* van de andere. Wat houdt elk woord, ook een woord dat twee keer staat?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: the first asks whether `translation` is one of the translations in `references`, the second whether `r` holds the same words as `translation`, in any order.",
                      "nl": "Vervang de twee `...`: de eerste vraagt of `translation` een van de vertalingen in `references` is, de tweede of `r` dezelfde woorden bevat als `translation`, in welke volgorde ook."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is the translation one of `references`? If not, does one of them hold the same words, each as often, in another order?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Is de vertaling een van `references`? Zo niet, bevat een ervan dezelfde woorden, elk even vaak, in een andere volgorde?"},
            "almost": {"en": "Right for the two small cases. But `{call}` gives {got}, and it should give {want}. Does your function look at every translation in `references`, and count a word as often as it comes?",
                       "nl": "Goed voor de twee kleine gevallen. Maar `{call}` geeft {got}, en het moet {want} zijn. Kijkt je functie naar elke vertaling in `references`, en telt de functie een woord zo vaak als het staat?"},
            "correct": {"en": "Right: your function tells an exact translation from one with its words in the wrong order. How many of the word-for-word translations, in the picture below, are in the wrong order?",
                        "nl": "Goed: je functie onderscheidt een precieze vertaling van een vertaling met de woorden in de verkeerde volgorde. Hoeveel van de vertalingen woord voor woord, in het plaatje hieronder, staan in de verkeerde volgorde?"},
        },
    ),
    "verb_after_helper": FunctionBox(
        name="verb_after_helper",
        picture=rule_picture,
        function="verb_after_helper",
        start=_VH_START,
        solution=_VH,
        show=_show,
        visible=[(_written(a), _written(b), _written(c)) for a, b, c in TOY_RULE],
        hidden=[(_written(["ik", "kan", "zwemmen"]), _written(TOY_HELPERS), _written(TOY_VERBS)),
                (_written(["ik", "heb", "het", "boek"]), _written(TOY_HELPERS), _written(TOY_VERBS)),
                (_written(["het", "boek", "gelezen"]), _written(TOY_HELPERS), _written(TOY_VERBS)),
                (_written(["kan", "ik", "zwemmen"]), _written(TOY_HELPERS), _written(TOY_VERBS)),
                (_written(["ik", "zou", "het", "hebben", "gelezen"]), _written({"zou", "hebben"}), _written({"gelezen"})),
                (_words_val(pair(WORKED)[0]), _HELPERS_VAL, _VERBS_VAL),
                (_words_val(pair(HELP)[0]), _HELPERS_VAL, _VERBS_VAL),
                (_words_val(pair(NOTHING)[0]), _HELPERS_VAL, _VERBS_VAL)],
        first=[_DOTS],
        mistakes=[
            Mistake("not-list", matches=_not_list,
                    en="`{call}` gives {got}, and it should give {want}: a list of words. What does your function give back?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een lijst met woorden. Wat geeft je functie terug?"),
            Mistake("end-kept", wrong=_VH_END_KEPT,
                    en="`{call}` gives {got}, and it should give {want}: *gelezen* is now twice in the list, after *heb* and still at the end. Which slice gives the words after the helping verb, without the last word?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: *gelezen* staat nu twee keer in de lijst, na *heb* en nog steeds aan het eind. Welke slice geeft de woorden na het hulpwerkwoord, zonder het laatste woord?"),
            Mistake("helper-twice", wrong=_VH_HELPER_TWICE,
                    en="`{call}` gives {got}, and it should give {want}: *heb* is now twice in the list. The slice after `[last]` starts at place `i`, the place of *heb* itself: where should it start?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: *heb* staat nu twee keer in de lijst. De slice na `[last]` begint op plaats `i`, de plaats van *heb* zelf: waar moet de slice beginnen?"),
            Mistake("before-helper", wrong=_VH_BEFORE,
                    en="`{call}` gives {got}, and it should give {want}: *gelezen* now stands in front of *heb*, and English puts it after the helping verb. Which slice ends right after *heb*?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: *gelezen* staat nu eerder dan *heb*, en het Engels zet het na het hulpwerkwoord. Welke slice eindigt direct na *heb*?"),
            Mistake("lost-rest", wrong=_VH_LOST,
                    en="`{call}` gives {got}, and it should give {want}: *het boek* is gone. Which words still have to come after *gelezen*?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: *het boek* is verdwenen. Welke woorden moeten er nog na *gelezen* komen?"),
            Mistake("inverted", wrong=_VH_INVERTED,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back the words as they are, though *gelezen* is in `verbs` and *heb* in `helpers`. When should it give back `words` unchanged?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie geeft de woorden onveranderd terug, terwijl *gelezen* in `verbs` staat en *heb* in `helpers`. Wanneer moet je functie `words` onveranderd teruggeven?"),
            Mistake("any-last", wrong=_VH_ANY_LAST,
                    en="`{call}` gives {got}, and it should give {want}: your function moves *boek*, though *boek* is not in `verbs`. Which last words may move?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie verplaatst *boek*, terwijl *boek* niet in `verbs` staat. Welke laatste woorden mogen verschuiven?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: the `if` gives back `words` as they are when the last word may not move, and the `return` in the loop gives the words with `last` right after the helping verb at place `i`.",
                      "nl": "Vervang de twee `...`: de `if` geeft `words` onveranderd terug als het laatste woord niet mag verschuiven, en de `return` in de loop geeft de woorden met `last` direct na het hulpwerkwoord op plaats `i`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is the last word in `verbs`? Then which is the first word in `helpers`, and what list do you get when the last word moves to right after it?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Staat het laatste woord in `verbs`? Wat is dan het eerste woord in `helpers`, en welke lijst krijg je als het laatste woord naar de plek direct daarna schuift?"},
            "almost": {"en": "Right for the two small cases. But `{call}` gives {got}, and it should give {want}. Does your function leave a sentence alone when its last word is not in `verbs`, or when no word of `helpers` comes before it? And does it put the verb right after the first helping verb, not a later one?",
                       "nl": "Goed voor de twee kleine gevallen. Maar `{call}` geeft {got}, en het moet {want} zijn. Laat je functie een zin met rust als het laatste woord niet in `verbs` staat, of als er eerder geen woord uit `helpers` staat? En zet je functie het werkwoord direct na het eerste hulpwerkwoord, niet na een volgend hulpwerkwoord?"},
            "correct": {"en": "Right: your function moves the verb to where English wants it. Does every sentence get better from it, in the picture below?",
                        "nl": "Goed: je functie zet het werkwoord waar het Engels het wil hebben. Wordt elke zin er beter van, in het plaatje hieronder?"},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, code, state, known mistake).
CASES = [
    ("word_for_word", _WW_START, "empty", None),
    ("word_for_word", _WW, "correct", None),
    ("word_for_word", _ww(other="            ...\n"), "wrong", "dots-left"),
    ("word_for_word", _WW_NO_IF, "error", "keyerror"),
    ("word_for_word", _WW_SWAPPED, "error", "keyerror"),
    ("word_for_word", _ww(known="            english.append(dictionary[w])\n").replace("    return english\n",
                                                                                         '    return " ".join(english)\n'),
     "wrong", "not-list"),
    ("word_for_word", _WW_DROPPED, "wrong", "dropped"),
    ("word_for_word", _WW_KEPT, "wrong", "kept"),
    ("word_for_word", _WW_GET, "wrong", "get-none"),
    ("word_for_word", _WW_INSIDE, "wrong", "return-inside"),
    ("word_for_word", _WW_HEAD + "        english.append(dictionary.get(w, w))\n    return english\n", "correct", None),
    ("word_for_word", "def word_for_word(words, dictionary):\n"
     "    return [dictionary[w] if w in dictionary else w for w in words]\n", "correct", None),
    ("word_for_word", _ww(other="            english = english + [w]\n"), "correct", None),
    ("outcome", _OC_START, "empty", None),
    ("outcome", _OC, "correct", None),
    ("outcome", _oc(order="        if ...:\n"), "wrong", "dots-left"),
    ("outcome", _OC.replace('return "order"', "return True"), "wrong", "not-kind"),
    ("outcome", _OC_FIRST, "wrong", "first-only"),
    ("outcome", _OC_NOT_SAME, "wrong", "not-same"),
    ("outcome", _OC_NO_SORT, "wrong", "no-sort"),
    ("outcome", _OC_HALF, "wrong", "half-sorted"),
    ("outcome", _OC_SORT, "wrong", "sort-method"),
    ("outcome", _OC_LEN, "wrong", "length"),
    ("outcome", _OC_SET, "wrong", "set"),
    ("outcome", _oc(order="        if sorted(translation) == sorted(r):\n"), "correct", None),
    ("outcome", "def outcome(translation, references):\n    if translation in references:\n        return \"same\"\n"
     "    if sorted(translation) in [sorted(r) for r in references]:\n        return \"order\"\n    return \"words\"\n",
     "correct", None),
    ("verb_after_helper", _VH_START, "empty", None),
    ("verb_after_helper", _VH, "correct", None),
    ("verb_after_helper", _vh(ret="            return ...\n"), "wrong", "dots-left"),
    ("verb_after_helper", _vh(ret="            return tuple(words[:i + 1] + [last] + words[i + 1:-1])\n"), "wrong", "not-list"),
    ("verb_after_helper", _VH_END_KEPT, "wrong", "end-kept"),
    ("verb_after_helper", _VH_HELPER_TWICE, "wrong", "helper-twice"),
    ("verb_after_helper", _VH_BEFORE, "wrong", "before-helper"),
    ("verb_after_helper", _VH_LOST, "wrong", "lost-rest"),
    ("verb_after_helper", _VH_INVERTED, "wrong", "inverted"),
    ("verb_after_helper", _VH_ANY_LAST, "wrong", "any-last"),
    ("verb_after_helper", _vh("    if not last in verbs:\n"), "correct", None),
    ("verb_after_helper", _vh(ret="            return words[:i + 1] + [words[-1]] + words[i + 1:len(words) - 1]\n"),
     "correct", None),
]
