# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-corpora: make a concordance of a word in a book (kwic), find the particles of a sentence
by the labels of a treebank (particles), label sentences by the test of a guideline, and choose a corpus for a
question.

The books are the course's eight books of Project Gutenberg (data/gutenberg-books.json.gz, made by
data/tools/make_gutenberg.py; everything_ai.books cuts them into words). The picture under the particles uses two
treebanks of Universal Dependencies 2.19, UD_English-EWT (without the emails of Enron) and UD_Dutch-Alpino (data/ud-
en-ewt.conllu.gz and data/ud-nl-alpino.conllu.gz, made by data/tools/make_ud.py). The senses of *kid* in the first
picture are those of WordNet 3.1 (WordNet 3.1 Copyright 2011 by Princeton University; all rights reserved, under the
WordNet licence, which allows this use with the notice); the senses of *kind* are the synsets of Open Dutch WordNet in
data/wordnet-nl.tsv (CC BY-SA 4.0), with descriptions of the course. The sentences of the exercises, the sort and the
cases of the checks are the course's own; the tests of the sort follow Beatrice Santorini's guidelines for the Penn
Treebank (1990), with sentences of the course."""
import collections
import contextlib
import copy
import gzip
import html
import io

from everything_ai import books, paths, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, kept, with_time_limit

# --- the books -------------------------------------------------------------------------------------------------------


def book_words(key: str) -> list:
    """The words of one of the course's books, as a new list: the text without punctuation, cut into words."""
    return books.book_words(key)


def book_names() -> dict:
    """{"title (year)": key} of the eight books, for the menu of the demo."""
    return {f'{b["title"]} ({b["year"]})': b["key"] for b in books.books()}


def kwic(words, target, width):
    lines = []
    for i in range(len(words)):
        if words[i].lower() == target:
            start = i - width
            if start < 0:
                start = 0
            left = " ".join(words[start:i])
            right = " ".join(words[i + 1:i + 1 + width])
            lines.append((left, words[i], right))
    return lines


# --- the sentence of exercise 2 --------------------------------------------------------------------------------------

# word i of a sentence is at place i - 1: (form, head, relation); head 0 is the root. The heads and relations are the
# same in English and in Dutch, so a case on numbers gives the same answer in both languages.
SENTENCES = {
    "en": [("We", 2, "nsubj"), ("picked", 0, "root"), ("the", 4, "det"), ("kids", 2, "obj"), ("up", 2, "compound:prt"),
           ("and", 7, "cc"), ("dropped", 2, "conj"), ("them", 7, "obj"), ("off", 7, "compound:prt")],
    "nl": [("We", 2, "nsubj"), ("haalden", 0, "root"), ("de", 4, "det"), ("kinderen", 2, "obj"),
           ("op", 2, "compound:prt"), ("en", 7, "cc"), ("zetten", 2, "conj"), ("ze", 7, "obj"),
           ("af", 7, "compound:prt")],
}
SENTENCE_TAGS = ["PRON", "VERB", "DET", "NOUN", "ADP", "CCONJ", "VERB", "PRON", "ADP"]


def particles(sentence):
    found = []
    for i in range(1, len(sentence) + 1):
        form, head, relation = sentence[i - 1]
        if relation == "compound:prt":
            found.append((head, i))
    return found


# --- the treebanks of the picture under exercise 2 -------------------------------------------------------------------

UD = {"en": paths.ROOT / "data" / "ud-en-ewt.conllu.gz", "nl": paths.ROOT / "data" / "ud-nl-alpino.conllu.gz"}
TREEBANK_NAMES = {"en": "UD_English-EWT", "nl": "UD_Dutch-Alpino"}


def _read(path) -> tuple:
    """The sentences of a CoNLL-U file of the course: (sent_id, words), each word a tuple (form, head, relation)."""
    out, words, sent_id = [], [], None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# sent_id = "):
                sent_id = line[len("# sent_id = "):]
            elif not line:
                if words:
                    out.append((sent_id, tuple(words)))
                words = []
            elif not line.startswith("#"):
                cols = line.split("\t")
                words.append((cols[1], int(cols[6]), cols[7]))
    if words:
        out.append((sent_id, tuple(words)))
    return tuple(out)


# what keeps a sentence out of the picture: the posts of EWT are written by anyone, so the picture leaves out its
# weblogs and newsgroups and keeps its reviews and answers, and in both treebanks no word may be one of _SKIP or hold
# one of _STEMS (as in nlp-dependencies)
_SKIP = {"ass", "god", "sex", "war", "dead", "die", "died", "damn", "crap", "jezus", "kut", "dood", "doden", "oorlog",
         "bom", "lijk"}
_STEMS = ("fuck", "shit", "bitch", "kill", "murder", "suicid", "terror", "bomb", "retard", "piss", "rape", "nazi",
          "porn", "drug", "shoot", "weapon", "death", "blood", "moord", "gedood", "verkracht", "wapen", "neuk", "hoer")
_SKIP_SOURCES = ("weblog-", "newsgroup-")
# words that keep a sentence from being the example under the table: the course names nobody by a gendered pronoun
_PRONOUNS = {"he", "she", "him", "her", "his", "hers", "himself", "herself", "hij", "zij", "haar", "hem", "zijn"}


def _pool(lang: str) -> tuple:
    out = []
    for sent_id, words in _read(UD[lang]):
        if sent_id.startswith(_SKIP_SOURCES):
            continue
        if any(f.lower() in _SKIP or any(x in f.lower() for x in _STEMS) for f, _, _ in words):
            continue
        out.append(words)
    return tuple(out)


POOLS = {"en": kept("nlp-corpora:ewt", lambda: _pool("en")), "nl": kept("nlp-corpora:alpino", lambda: _pool("nl"))}


# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_CODE = ";font-family:var(--ea-font-code);font-size:0.78rem"
_OFF = ";color:var(--ea-loss-text);font-weight:600"
_CHIP = ("display:inline-block;padding:0 0.25rem;margin:0.05rem 0.05rem;border:1px solid var(--ea-line);"
         "border-radius:4px;font-family:var(--ea-font-ui);font-size:0.85rem;color:var(--ea-ink)")
_MARKED = ";border:1.5px solid var(--ea-ink);font-weight:600"
# how each kind of arc is drawn: stroke, width, dashes, label colour
_ARC = {
    "ink": ("var(--ea-ink-2)", 1.3, "", "var(--ea-ink-2)"),
    "bold": ("var(--ea-ink)", 2.6, "", "var(--ea-ink)"),
}


def _note(text: str) -> str:
    """A line under a picture; a word between asterisks is set in italics, as in the lesson's text."""
    parts = html.escape(text).split("*")
    marked = "".join(f"<i>{p}</i>" if k % 2 else p for k, p in enumerate(parts))
    return f'<div style="{_UI}">{marked}</div>'


def _bare(text: str) -> str:
    """The same line without its asterisks, for the alt text."""
    return text.replace("*", "")


def _num(n, lang: str) -> str:
    """A whole number with a separator for thousands: 26,776 in English, 26.776 in Dutch."""
    text = f"{round(n):,}"
    return text.replace(",", ".") if lang == "nl" else text


def arcs_svg(words: list, arcs: list, alt: str, tags=None) -> str:
    """A dependency tree as arcs above the words: each arc (d, h, label, kind) goes from head h to word d, with the
    arrow at d; head 0 is the root, drawn as an arrow from above. Words count from 1. (As in nlp-dependencies.)"""
    n = len(words)
    tags = tags if tags is not None and len(tags) == n else None
    widths = [max(30.0, 8.2 * len(w) + 14, (7.4 * len(tags[k]) + 10) if tags else 0.0) for k, w in enumerate(words)]
    xs, x = [], 10.0
    for wd in widths:
        xs.append(x + wd / 2)
        x += wd + 10
    width = x + 6
    real = [(d, h, label, kind) for d, h, label, kind in arcs if 1 <= d <= n and 0 <= h <= n and h != d]
    spans = {}
    for d, h, _, _ in real:
        if h:
            spans[(d, h)] = (min(d, h), max(d, h))
    level = {}
    for key in sorted(spans, key=lambda k: spans[k][1] - spans[k][0]):
        lo, hi = spans[key]
        inside = [level[o] for o in level if spans[o] != (lo, hi) and lo <= spans[o][0] and spans[o][1] <= hi]
        level[key] = 1 + max(inside, default=0)
    top_level = max(level.values(), default=0)
    y0 = 40 + 22 * top_level
    height = y0 + (44 if tags else 28)
    parts = []
    for d, h, label, kind in real:
        stroke, sw, dash, ink = _ARC.get(kind, _ARC["ink"])
        style = f"fill:none;stroke:{stroke};stroke-width:{sw}" + (f";stroke-dasharray:{dash}" if dash else "")
        dx = xs[d - 1]
        if not h:
            parts.append(f'<path d="M{dx:.1f},8 L{dx:.1f},{y0 - 7:.1f}" style="{style}"/>')
            top, mid, anchor = 16.0, dx + 5, "start"
        else:
            hx = xs[h - 1]
            side = 1 if d > h else -1
            hx, dx = hx + 4 * side, dx - 4 * side
            top = y0 - 12 - 22 * level[(d, h)]
            r = 6
            parts.append(f'<path d="M{hx:.1f},{y0:.1f} L{hx:.1f},{top + r:.1f} Q{hx:.1f},{top:.1f} {hx + r * side:.1f},'
                         f'{top:.1f} L{dx - r * side:.1f},{top:.1f} Q{dx:.1f},{top:.1f} {dx:.1f},{top + r:.1f} '
                         f'L{dx:.1f},{y0 - 7:.1f}" style="{style}"/>')
            mid, anchor = (hx + dx) / 2, "middle"
        parts.append(f'<polygon points="{dx - 3.6:.1f},{y0 - 8:.1f} {dx + 3.6:.1f},{y0 - 8:.1f} {dx:.1f},{y0 - 1:.1f}" '
                     f'style="fill:{stroke}"/>')
        if label:
            weight = ";font-weight:600" if kind == "bold" else ""
            parts.append(f'<text x="{mid:.1f}" y="{top - 3:.1f}" text-anchor="{anchor}" style="font-family:'
                         f'var(--ea-font-code);font-size:11px;fill:{ink};stroke:var(--ea-page);stroke-width:4;'
                         f'paint-order:stroke{weight}">{html.escape(label)}</text>')
    for k, w in enumerate(words):
        parts.append(f'<text x="{xs[k]:.1f}" y="{y0 + 17:.1f}" text-anchor="middle" style="font-family:var(--ea-font-ui);'
                     f'font-size:14px;fill:var(--ea-ink);font-weight:500">{html.escape(w)}</text>')
        if tags:
            parts.append(f'<text x="{xs[k]:.1f}" y="{y0 + 34:.1f}" text-anchor="middle" style="font-family:'
                         f'var(--ea-font-code);font-size:10.5px;fill:var(--ea-ink-3)">{html.escape(tags[k])}</text>')
    return (f'<div style="overflow-x:auto"><svg role="img" aria-label="{html.escape(alt)}" viewBox="0 0 {width:.0f} '
            f'{height:.0f}" width="{width:.0f}" height="{height:.0f}" style="max-width:none;display:block">{"".join(parts)}</svg></div>')


# The senses of the word of the third row: the English ones of WordNet 3.1 (the first words of each synset, and its
# gloss up to the first semicolon), the Dutch ones the synsets of Open Dutch WordNet with the lemma "kind" in
# data/wordnet-nl.tsv (their Dutch lemmas, and a description of the course).
SENSES = {
    "en": ("kid", [("kid", "young goat"),
                   ("child, kid, youngster, minor, shaver, nipper, …", "a young person of either sex"),
                   ("child, kid", "a human offspring (son or daughter) of any age"),
                   ("kid, kidskin", "soft smooth leather from the hide of a young goat")]),
    "nl": ("kind", [("kind, jongere, jongeling, minderjarige, ding", "een jong mens"),
                    ("kind, koter, loot, spruit", "een zoon of dochter, van welke leeftijd ook"),
                    ("kind, nakomeling, jong, jonkie, kleintje", "een jong dier")]),
}

_IDEA_WORDS = {
    "en": {"title": "One sentence in a corpus, a treebank and a lexical database",
           "rows": ["In a corpus: the text", "In a treebank: the text with its tree",
                    "In a lexical database: facts about a word"],
           "sentence": "We picked the kids up and dropped them off.",
           "senses": "The noun *{word}* has {n} senses in WordNet 3.1, each a synset of its own:",
           "legend": "The sentence and its labels are the course's own, by the guidelines of Universal Dependencies: "
                     "the part of speech under each word, and the relations on the arcs. The senses of *kid* are "
                     "those of WordNet 3.1 (Princeton University, 2011), with the first words of each synset and its "
                     "gloss."},
    "nl": {"title": "Eén zin in een corpus, een treebank en een lexicale database",
           "rows": ["In een corpus: de tekst", "In een treebank: de tekst met de boom",
                    "In een lexicale database: feiten over een woord"],
           "sentence": "We haalden de kinderen op en zetten ze af.",
           "senses": "Het zelfstandig naamwoord *{word}* staat in Open Dutch WordNet in {n} synsets:",
           "legend": "De zin en de labels zijn van de course, volgens de richtlijnen van Universal Dependencies: de "
                     "woordsoort onder elk woord, en de relaties op de bogen. De synsets van *kind* komen uit Open "
                     "Dutch WordNet (via de Open Multilingual Wordnet 2.0), met de Nederlandse woorden erin en een "
                     "omschrijving van de course."},
}


def idea_picture(lang: str) -> str:
    """The sentence of exercise 2 in three kinds of resource."""
    w = _IDEA_WORDS[lang]
    sentence = SENTENCES[lang]
    words = [s[0] for s in sentence]
    said = "; ".join(f"{rel}({sentence[h - 1][0] if h else 'ROOT'}, {form})" for form, h, rel in sentence)
    tree = arcs_svg(words, [(i, s[1], s[2], "ink") for i, s in enumerate(sentence, start=1)], said, SENTENCE_TAGS)
    word, senses = SENSES[lang]
    sense_rows = "".join(f'<tr><td style="{_CELL}">{html.escape(syn)}</td><td style="{_CELL};color:var(--ea-ink-2)">'
                         f'{html.escape(gloss)}</td></tr>' for syn, gloss in senses)
    third = (_note(w["senses"].format(word=word, n=len(senses)))
             + f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.2rem 0">{sense_rows}'
               f'</table></div>')
    text = f'<div style="font-family:var(--ea-font-prose);font-size:1rem;color:var(--ea-ink)">{html.escape(w["sentence"])}</div>'
    rows = "".join(f'<tr><th style="{_HEAD};width:10rem">{h}</th><td style="{_CELL}">{body}</td></tr>'
                   for h, body in zip(w["rows"], [text, tree, third]))
    table = f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.4rem 0">{rows}</table></div>'
    alt = (f'{w["title"]}. {w["rows"][0]}: {w["sentence"]} {w["rows"][1]}: {said}. {w["rows"][2]}: '
           f'{_bare(w["senses"].format(word=word, n=len(senses)))} '
           + "; ".join(f"{syn}: {gloss}" for syn, gloss in senses) + f". {_bare(w['legend'])}")
    return yours.figure(w["title"], table + _note(w["legend"]), alt=alt)


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


def _is_line(v) -> bool:
    return isinstance(v, tuple) and len(v) == 3 and all(isinstance(p, str) for p in v)


def _kwic_rows(lines: list, wrong=()) -> str:
    """Lines of a concordance as a table: the left side to the right, the word in the middle, the right side."""
    rows = ""
    for k, (left, key, right) in enumerate(lines):
        off = _OFF if k in wrong else ""
        rows += (f'<tr><td style="{_CELL};text-align:right;white-space:nowrap;color:var(--ea-ink-2){off}">'
                 f'{html.escape(left)}</td><td style="{_CELL};text-align:center;font-weight:600{off}">'
                 f'{html.escape(key)}</td><td style="{_CELL};white-space:nowrap;color:var(--ea-ink-2){off}">'
                 f'{html.escape(right)}</td></tr>')
    return f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.4rem 0">{rows}</table></div>'


SHOWN = 12   # lines of the picture under exercise 1

_KWIC_WORDS = {
    "en": {"title": "Your kwic, at work: rabbit in Alice's Adventures in Wonderland",
           "fine": "Your concordance has all {n} lines; the first {k} are shown. Read them from top to bottom: "
                   "*White* stands before *Rabbit* in {white} of them.",
           "count": "Your function gives {got} lines, and the book has {n} places where *rabbit* stands.",
           "differs": "A line in purple differs from the right line in that place.",
           "error": "Your function stops with an error on the words of the book.",
           "slow": "Your function takes too long on the words of the book.",
           "not_lines": "Your function does not give back a list of lines of three texts, so there is nothing to "
                        "show yet."},
    "nl": {"title": "Jouw kwic aan het werk: rabbit in Alice's Adventures in Wonderland",
           "fine": "Je concordantie heeft alle {n} regels; de eerste {k} staan hier. Lees ze van boven naar beneden: "
                   "*White* staat in {white} ervan voor *Rabbit*.",
           "count": "Je functie geeft {got} regels, en het boek heeft {n} plaatsen waar *rabbit* staat.",
           "differs": "Een paarse regel wijkt af van de goede regel op die plaats.",
           "error": "Je functie stopt met een fout op de woorden van het boek.",
           "slow": "Je functie doet er te lang over op de woorden van het boek.",
           "not_lines": "Je functie geeft geen lijst regels van drie teksten terug, dus er valt nog niets te tonen."},
}


def kwic_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _KWIC_WORDS[seen.lang]
    words = book_words("alice")
    want = kwic(words, "rabbit", 3)
    try:
        got = _call(fn, words, "rabbit", 3)
    except Timeout:
        return yours.figure(w["title"], _note(w["slow"]), alt=f'{w["title"]}. {w["slow"]}')
    if got is _STOPPED:
        return yours.figure(w["title"], _note(w["error"]), alt=f'{w["title"]}. {w["error"]}')
    if not isinstance(got, list) or not got or not all(_is_line(v) for v in got):
        return yours.figure(w["title"], _note(w["not_lines"]), alt=f'{w["title"]}. {w["not_lines"]}')
    shown = got[:SHOWN]
    wrong = [k for k in range(len(shown)) if k >= len(want) or shown[k] != want[k]]
    if got == want:
        white = sum(1 for left, key, right in want if left.endswith("White"))
        after = w["fine"].format(n=len(want), k=len(shown), white=white)
    else:
        after = w["count"].format(got=len(got), n=len(want)) + " " + w["differs"]
    alt = f'{w["title"]}. ' + "; ".join(f"{a} [{b}] {c}" for a, b, c in shown) + f". {_bare(after)}"
    return yours.figure(w["title"], _kwic_rows(shown, wrong) + _note(after), alt=alt)


DEMO_LINES = 23   # a screenful, in Sinclair's words: "in practice a screenful, some 23 lines"

_DEMO_WORDS = {
    "en": {"title": "{word} in {book}",
           "count": "{book} has {total} words. *{word}* stands at {n} of them: {rate} per million words.",
           "shown": "The first {k} lines, in the order of the book:",
           "none": "*{word}* does not occur in this book. Try another word, or another book.",
           "empty": "Type a word.",
           "edition": "The text follows {edition}."},
    "nl": {"title": "{word} in {book}",
           "count": "{book} heeft {total} woorden. *{word}* staat op {n} plaatsen: {rate} per miljoen woorden.",
           "shown": "De eerste {k} regels, in de volgorde van het boek:",
           "none": "*{word}* komt niet voor in dit boek. Probeer een ander woord, of een ander boek.",
           "empty": "Typ een woord.",
           "edition": "De tekst volgt {edition}."},
}
_EDITIONS_NL = {
    "pride": "de uitgave van George Allen, 1894, met een voorwoord van George Saintsbury",
    "frankenstein": "de herziene tekst van 1831",
    "havelaar": "de aantekeningen van de uitgave van 1875, herzien in 1881",
    "camera-obscura": "de 29e druk, Haarlem, De Erven F. Bohn, 1917",
}


def demo_picture(key: str, word: str, width: int, lang: str) -> str:
    """The course's concordance of a word in one of the eight books: how often, and the first 23 lines."""
    w = _DEMO_WORDS[lang]
    key = key if key in books.keys() else "alice"
    b = books.book(key)
    target = (word or "").strip().lower()
    title = w["title"].format(word=target or "…", book=b["title"])
    edition = _EDITIONS_NL.get(key, "") if lang == "nl" else b["edition"]
    end = _note(w["edition"].format(edition=edition)) if edition else ""
    if not target:
        return yours.figure(title, _note(w["empty"]), alt=f"{title}. {w['empty']}")
    words = book_words(key)
    lines = kwic(words, target, width)
    if not lines:
        none = w["none"].format(word=target)
        return yours.figure(title, _note(none), alt=f"{title}. {_bare(none)}")
    count = w["count"].format(book=b["title"], total=_num(len(words), lang), word=target, n=_num(len(lines), lang),
                              rate=_num(len(lines) / len(words) * 1_000_000, lang))
    shown = lines[:DEMO_LINES]
    head = w["shown"].format(k=len(shown))
    alt = f"{title}. {_bare(count)} {head} " + "; ".join(f"{a} [{k}] {c}" for a, k, c in shown) + "."
    return yours.figure(title, _note(count) + _note(head) + _kwic_rows(shown) + end, alt=alt)


_PRT_WORDS = {
    "en": {"title": "Your particles, at work: every sentence of {bank} from reviews and answers",
           "head": ["Verb", "Particle", "How often"],
           "fine": "Your function finds all {n} particles in {s} sentences, with {k} different pairs of verb and "
                   "particle; the {m} most common are shown.",
           "differs": "Your function finds {got} pairs, and the labels of the treebank give {n}. A pair in purple "
                      "is not one of them.",
           "example": "An example from the treebank, with the verb and its particle marked:",
           "error": "Your function stops with an error on a sentence of the treebank.",
           "slow": "Your function takes too long on the sentences of the treebank.",
           "not_pairs": "Your function does not give back a list of pairs of word numbers, so there is nothing to "
                        "count yet."},
    "nl": {"title": "Jouw particles aan het werk: elke zin van {bank}",
           "head": ["Werkwoord", "Partikel", "Hoe vaak"],
           "fine": "Je functie vindt alle {n} partikels in {s} zinnen, met {k} verschillende paren van werkwoord en "
                   "partikel; de {m} die het vaakst voorkomen, staan hier.",
           "differs": "Je functie vindt {got} paren, en de labels van de treebank geven er {n}. Een paars paar hoort "
                      "daar niet bij.",
           "example": "Een voorbeeld uit de treebank, met het werkwoord en het partikel gemarkeerd:",
           "error": "Je functie stopt met een fout bij een zin van de treebank.",
           "slow": "Je functie doet er te lang over op de zinnen van de treebank.",
           "not_pairs": "Je functie geeft geen lijst paren van woordnummers terug, dus er valt nog niets te tellen."},
}
TOP = 8   # pairs in the table under exercise 2


def _pairs_of(sentence, got) -> list:
    """The pairs (verb form, particle form) for a list of (head, i), in small letters; None when got is no such list."""
    if not isinstance(got, list):
        return None
    out = []
    n = len(sentence)
    for p in got:
        if not (isinstance(p, tuple) and len(p) == 2 and all(type(v) is int and 1 <= v <= n for v in p)):
            return None
        out.append((sentence[p[0] - 1][0].lower(), sentence[p[1] - 1][0].lower()))
    return out


def _example(pool, pair) -> tuple:
    """A short sentence of the pool with that pair of verb and particle, without a gendered pronoun: (words, marked)."""
    for words in sorted(pool, key=len):
        if len(words) < 5 or any(f.lower() in _PRONOUNS for f, _, _ in words):
            continue
        for i, (form, head, rel) in enumerate(words, start=1):
            if rel == "compound:prt" and (words[head - 1][0].lower(), form.lower()) == pair:
                return [f for f, _, _ in words], {head - 1, i - 1}
    return None, set()


def particles_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _PRT_WORDS[lang]
    title = w["title"].format(bank=TREEBANK_NAMES[lang])
    pool = POOLS[lang]
    sentences = [list(words) for words in pool]
    fresh = [list(words) for words in pool]   # lists of tuples: the learner's code can change its own list only
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            got = with_time_limit(lambda: [fn(s) for s in fresh])
    except Timeout:
        return yours.figure(title, _note(w["slow"]), alt=f'{title}. {w["slow"]}')
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return yours.figure(title, _note(w["error"]), alt=f'{title}. {w["error"]}')
    mine, right = collections.Counter(), collections.Counter()
    with_prt = 0
    for s, g in zip(sentences, got):
        pairs = _pairs_of(s, g)
        if pairs is None:
            return yours.figure(title, _note(w["not_pairs"]), alt=f'{title}. {w["not_pairs"]}')
        mine.update(pairs)
        theirs = _pairs_of(s, particles(s))
        right.update(theirs)
        with_prt += bool(theirs)
    top = mine.most_common(TOP)
    rows = ""
    for (verb, prt), n in top:
        off = "" if (verb, prt) in right else _OFF
        rows += (f'<tr><td style="{_CELL}{off}">{html.escape(verb)}</td><td style="{_CELL}{off}">{html.escape(prt)}'
                 f'</td><td style="{_CELL};text-align:right">{_num(n, lang)}</td></tr>')
    head = "".join(f'<th style="{_HEAD}">{h}</th>' for h in w["head"])
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.4rem 0">'
             f'<tr>{head}</tr>{rows}</table></div>')
    if mine == right:
        after = w["fine"].format(n=_num(sum(right.values()), lang), s=_num(with_prt, lang),
                                 k=_num(len(right), lang), m=len(top))
    else:
        after = w["differs"].format(got=_num(sum(mine.values()), lang), n=_num(sum(right.values()), lang))
    body = table + _note(after)
    alt_example = ""
    if top and top[0][0] in right:
        words, marked = _example(pool, top[0][0])
        if words:
            chips = " ".join(f'<span style="{_CHIP}{_MARKED if k in marked else ""}">{html.escape(f)}</span>'
                             for k, f in enumerate(words))
            body += _note(w["example"]) + f'<div style="margin:0.3rem 0;line-height:1.9">{chips}</div>'
            alt_example = f' {w["example"]} {" ".join(words)}'
    alt = (f'{title}. ' + "; ".join(f"{v} {p}: {n}" for (v, p), n in top) + f". {after}{alt_example}")
    return yours.figure(title, body, alt=alt)


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["kwic", "particles", "label", "purpose"]


class Named:
    """A value the check gives the learner's function, shown in a message by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


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


def _value(v) -> str:
    """A short value the way the lessons write it: strings in double quotes, also inside a tuple or a list."""
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, tuple):
        return "(" + ", ".join(_value(p) for p in v) + ("," if len(v) == 1 else "") + ")"
    if isinstance(v, list):
        return "[" + ", ".join(_value(p) for p in v) + "]"
    return repr(v)


def _show_lines(v):
    """{got} and {want} of kwic: a long list of lines as its length and its first two lines."""
    if isinstance(v, list) and len(v) > 3:
        first = ", ".join(_value(p) for p in v[:2])
        return {"en": f"a list of {len(v)} lines, starting with {first}",
                "nl": f"een lijst van {len(v)} regels, die begint met {first}"}
    return _value(v)


# exercise 1: kwic. Every case of the check's own is short; the word in the third place with a width of 3 shows
# whether the left side starts at 0 when i - width is below it.
class _Book:
    """The words of Alice, made when a check first needs them, and shown in a message as ALICE."""

    def __repr__(self) -> str:
        return "ALICE"

    def __deepcopy__(self, memo):
        return book_words("alice")


ALICE_N = _Book()
KWIC_CASES = [
    (ALICE_N, "rabbit", 3),
    (["down", "the", "rabbit", "hole"], "rabbit", 3),
    (["the", "cat", "sat", "on", "the", "mat"], "the", 2),
    (["the", "cat", "sat", "on", "the", "mat"], "mat", 2),
    (["The", "Rabbit", "ran", "past", "the", "door"], "rabbit", 1),
    (["up", "and", "up"], "up", 0),
    (["once", "upon", "a", "time"], "rabbit", 2),
    (ALICE_N, "hatter", 2),
]
_K_START = "def kwic(words, target, width):\n    ...\n"
_K = ("def kwic(words, target, width):\n"
      "    lines = []\n"
      "    for i in range(len(words)):\n"
      "        if words[i].lower() == target:\n"
      "            start = i - width\n"
      "            if start < 0:\n"
      "                start = 0\n"
      "            left = \" \".join(words[start:i])\n"
      "            right = \" \".join(words[i + 1:i + 1 + width])\n"
      "            lines.append((left, words[i], right))\n"
      "    return lines\n")
_K_NO_CLAMP = _K.replace("            if start < 0:\n                start = 0\n", "")
_K_LEFT_KEY = _K.replace("words[start:i]", "words[start:i + 1]")
_K_RIGHT_KEY = _K.replace("words[i + 1:i + 1 + width]", "words[i:i + 1 + width]")
_K_SHORT = _K.replace("words[i + 1:i + 1 + width]", "words[i + 1:i + width]")
_K_CASE = _K.replace("if words[i].lower() == target:", "if words[i] == target:")
_K_TARGET = _K.replace("lines.append((left, words[i], right))", "lines.append((left, target, right))")
_K_LISTS = (_K.replace("left = \" \".join(words[start:i])", "left = words[start:i]")
            .replace("right = \" \".join(words[i + 1:i + 1 + width])", "right = words[i + 1:i + 1 + width]"))


def _not_list(got, text) -> bool:
    return not isinstance(got, list)


def _lines_with_lists(got, text) -> bool:
    return isinstance(got, list) and any(isinstance(v, tuple) and any(isinstance(p, list) for p in v) for v in got)


def _not_lines(got, text) -> bool:
    return isinstance(got, list) and any(not _is_line(v) for v in got)


# exercise 2: particles. The answers are word numbers, the same in both languages; the cases of the check's own hold a
# plain compound, a preposition, a particle before its verb (Dutch om ze op te halen) and a sentence of one word.
SENTENCE_N = Named("SENTENCE", SENTENCES["en"])
PARTICLE_CASES = [
    (SENTENCE_N,),
    ([("Turn", 0, "root"), ("off", 1, "compound:prt"), ("the", 4, "det"), ("light", 1, "obj")],),
    ([("Call", 0, "root"), ("the", 4, "det"), ("phone", 4, "compound"), ("number", 1, "obj")],),
    ([("They", 2, "nsubj"), ("walked", 0, "root"), ("up", 5, "case"), ("the", 5, "det"), ("hill", 2, "obl")],),
    ([("Set", 0, "root"), ("up", 1, "compound:prt"), ("the", 5, "det"), ("phone", 5, "compound"), ("line", 1, "obj")],),
    ([("om", 5, "mark"), ("ze", 5, "obj"), ("op", 5, "compound:prt"), ("te", 5, "mark"), ("halen", 0, "root")],),
    ([("Hello", 0, "root")],),
]
_P_START = "def particles(sentence):\n    ...\n"
_P = ("def particles(sentence):\n"
      "    found = []\n"
      "    for i in range(1, len(sentence) + 1):\n"
      "        form, head, relation = sentence[i - 1]\n"
      "        if relation == \"compound:prt\":\n"
      "            found.append((head, i))\n"
      "    return found\n")
_P_SWAP = _P.replace("found.append((head, i))", "found.append((i, head))")
_P_ZERO = _P.replace("found.append((head, i))", "found.append((head, i - 1))")
_P_ANY = _P.replace("if relation == \"compound:prt\":", "if \"compound\" in relation:")
_P_FORMS = _P.replace("found.append((head, i))", "found.append((sentence[head - 1][0], form))")


def _strings_in_pairs(got, text) -> bool:
    return isinstance(got, list) and any(isinstance(p, tuple) and any(isinstance(v, str) for v in p) for p in got)


def _no_pairs(got, text) -> bool:
    return isinstance(got, list) and any(not isinstance(p, tuple) for p in got)


def _index_error(e) -> bool:
    return isinstance(e, IndexError)


EXERCISES = {
    "kwic": FunctionBox(
        name="kwic",
        picture=kwic_picture,
        function="kwic",
        start=_K_START,
        solution=_K,
        visible=KWIC_CASES[:1],
        hidden=KWIC_CASES[1:],
        first=[_DOTS],
        show=_show_lines,
        mistakes=[
            Mistake("not-a-list", matches=_not_list,
                    en="`{call}` gives {got}, and it should give a list: one line for every place where the word stands. Does your function make a list, add a line to it at every place, and give back the list after the loop?",
                    nl="`{call}` geeft {got}, en dat moet een lijst zijn: één regel voor elke plaats waar het woord staat. Maakt je functie een lijst, voegt de functie bij elke plaats een regel toe, en geeft de functie de lijst na de loop terug?"),
            Mistake("lists", matches=_lines_with_lists,
                    en="`{call}` gives lines with lists of words in them. The left and the right side should each be one text, with the words joined by spaces, as `\" \".join(...)` does in the worked example.",
                    nl="`{call}` geeft regels met lijsten woorden erin. De linker- en de rechterkant moeten elk één tekst zijn, met spaties tussen de woorden, zoals `\" \".join(...)` in het uitgewerkte voorbeeld doet."),
            Mistake("not-lines", matches=_not_lines,
                    en="`{call}` gives {got}. Each line should be a tuple of three texts: the words before, the word itself and the words after, as in `(left, words[i], right)`.",
                    nl="`{call}` geeft {got}. Elke regel moet een tuple van drie teksten zijn: de woorden ervoor, het woord zelf en de woorden erna, zoals in `(left, words[i], right)`."),
            Mistake("no-clamp", wrong=_K_NO_CLAMP,
                    en="`{call}` gives {got}, and it should give {want}: near the start of the list, your left side comes out empty. When `i - width` is below 0, a slice from there gives no words here; the left side should then start at place 0.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: vlak na het begin van de lijst blijft je linkerkant leeg. Als `i - width` onder 0 ligt, geeft een slice vanaf daar hier geen woorden; de linkerkant moet dan op plaats 0 beginnen."),
            Mistake("left-key", wrong=_K_LEFT_KEY,
                    en="`{call}` gives {got}, and it should give {want}: the word itself is also in your left side. The slice of the left side stops before place `i`: `words[a:b]` goes up to `b`, without `b` itself.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het woord zelf staat ook in je linkerkant. De slice van de linkerkant stopt vóór plaats `i`: `words[a:b]` gaat tot `b`, zonder `b` zelf."),
            Mistake("right-key", wrong=_K_RIGHT_KEY,
                    en="`{call}` gives {got}, and it should give {want}: the word itself is also in your right side. The right side starts at the place after the word.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het woord zelf staat ook in je rechterkant. De rechterkant begint op de plaats na het woord."),
            Mistake("short", wrong=_K_SHORT,
                    en="`{call}` gives {got}, and it should give {want}: your right side has one word too few. It should hold `width` words, where the text has them. Where does your slice stop?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je rechterkant heeft één woord te weinig. Daar moeten `width` woorden staan, als de tekst ze heeft. Waar stopt je slice?"),
            Mistake("case", wrong=_K_CASE,
                    en="`{call}` gives {got}, and it should give {want}: your function misses the word where it starts with a capital, as *Rabbit* in *the White Rabbit*. Compare the word in small letters with `target`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie mist het woord als het met een hoofdletter begint, zoals *Rabbit* in *the White Rabbit*. Vergelijk het woord in kleine letters met `target`."),
            Mistake("target", wrong=_K_TARGET,
                    en="`{call}` gives {got}, and it should give {want}: the middle of each line should be the word as the text writes it, with its capitals, not `target`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: in het midden van elke regel hoort het woord zoals de tekst het schrijft, met de hoofdletters, niet `target`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back a list with a line `(left, word, right)` for every place where `target` stands in `words`.",
                      "nl": "Vervang de `...`: geef een lijst terug met een regel `(left, word, right)` voor elke plaats waar `target` in `words` staat."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does each line have the `width` words before the word, the word itself, and the `width` words after it?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Heeft elke regel de `width` woorden vóór het woord, het woord zelf, en de `width` woorden erna?"},
            "almost": {"en": "Right for *rabbit* in *Alice*. But `{call}` gives {got}, and it should give {want}. Does your function work for any list of words, any word and any width?",
                       "nl": "Goed voor *rabbit* in *Alice*. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke lijst woorden, elk woord en elke breedte?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back: it has a `return`, but Python does not get to it. Is the `return` the last line, after the loop?",
                          "nl": "Met `{call}` geeft je functie niets terug: er staat een `return` in, maar Python komt er niet aan toe. Is de `return` de laatste regel, na de loop?"},
            "correct": {"en": "Your `kwic` is right in all {n} cases. Look at the picture: the first lines of your concordance of *rabbit*, with the word in the middle.",
                        "nl": "Je `kwic` klopt in alle {n} gevallen. Kijk naar de afbeelding: de eerste regels van je concordantie van *rabbit*, met het woord in het midden."},
        },
    ),
    "particles": FunctionBox(
        name="particles",
        picture=particles_picture,
        function="particles",
        start=_P_START,
        solution=_P,
        visible=PARTICLE_CASES[:1],
        hidden=PARTICLE_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("index", error=_index_error,
                    en="With `{call}`, your function looks past the end of the list. Word number `i` is at place `i - 1`: the words count from 1, the places in a list from 0.",
                    nl="Met `{call}` kijkt je functie voorbij het eind van de lijst. Woord nummer `i` staat op plaats `i - 1`: de woorden tellen vanaf 1, de plaatsen in een lijst vanaf 0."),
            Mistake("forms", matches=_strings_in_pairs,
                    en="`{call}` gives words, and it should give word numbers: {want}. The number of the verb is the head of the particle; the number of the particle is `i` itself.",
                    nl="`{call}` geeft woorden, en dat moeten woordnummers zijn: {want}. Het nummer van het werkwoord is het hoofdwoord van het partikel; het nummer van het partikel is `i` zelf."),
            Mistake("no-pairs", matches=_no_pairs,
                    en="`{call}` gives {got}, and it should give {want}: a pair for each particle, with two numbers in it, as in `(head, i)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een paar voor elk partikel, met twee nummers erin, zoals in `(head, i)`."),
            Mistake("swap", wrong=_P_SWAP,
                    en="`{call}` gives {got}, and it should give {want}: the two numbers of each pair are the other way round. The head, the number of the verb, comes first.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de twee nummers van elk paar staan andersom. Het hoofdwoord, het nummer van het werkwoord, komt eerst."),
            Mistake("zero", wrong=_P_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: the number of the particle is one too low. Word numbers count from 1, like the heads: the first word is number 1, at place 0.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het nummer van het partikel is één te laag. Woordnummers tellen vanaf 1, net als de hoofdwoorden: het eerste woord heeft nummer 1, op plaats 0."),
            Mistake("any-compound", wrong=_P_ANY,
                    en="`{call}` gives {got}, and it should give {want}: your function also takes words with the relation `compound`, such as *phone* in *phone number*. That relation joins two nouns; only `compound:prt` marks a particle.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie neemt ook woorden met de relatie `compound`, zoals *phone* in *phone number*. Die relatie verbindt twee zelfstandige naamwoorden; alleen `compound:prt` wijst een partikel aan."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back a pair `(head, i)` for every word of `sentence` whose relation is `compound:prt`.",
                      "nl": "Vervang de `...`: geef een paar `(head, i)` terug voor elk woord van `sentence` met de relatie `compound:prt`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which words have the relation `compound:prt`, and which numbers belong in their pair?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Welke woorden hebben de relatie `compound:prt`, en welke nummers horen in hun paar?"},
            "almost": {"en": "Right for SENTENCE. But `{call}` gives {got}, and it should give {want}. Does your function work for any sentence, also one without particles?",
                       "nl": "Goed voor SENTENCE. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke zin, ook een zin zonder partikels?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back: it has a `return`, but Python does not get to it. Is the `return` the last line, after the loop?",
                          "nl": "Met `{call}` geeft je functie niets terug: er staat een `return` in, maar Python komt er niet aan toe. Is de `return` de laatste regel, na de loop?"},
            "correct": {"en": "Your `particles` is right in all {n} cases. Look at the picture: your function, run on every sentence of a real treebank.",
                        "nl": "Je `particles` klopt in alle {n} gevallen. Kijk naar de afbeelding: je functie, op elke zin van een echte treebank."},
        },
    ),
    "label": Sort(
        name="label",
        groups={
            "particle": {"en": "A particle: the word belongs with the verb",
                         "nl": "Een partikel: het woord hoort bij het werkwoord"},
            "preposition": {"en": "A preposition: the word belongs with the noun after it",
                            "nl": "Een voorzetsel: het woord hoort bij het zelfstandig naamwoord erna"},
        },
        items={
            "switch": {"en": "**We switched *off* the radio.**",
                       "nl": "**We zetten de radio *uit*.**"},
            "jump": {"en": "**The cat jumped *off* the table.**",
                     "nl": "**De kat springt *uit* de boom.**"},
            "fill": {"en": "**Please fill *in* the form.**",
                     "nl": "**Vul het formulier *in*.**"},
            "wait": {"en": "**We waited *in* the hall.**",
                     "nl": "**We wachten *in* de hal.**"},
            "look": {"en": "**They looked *up* the word.**",
                     "nl": "**We zoeken het woord *op*.**"},
            "walk": {"en": "**They walked *up* the hill.**",
                     "nl": "**We lopen *op* de stoep.**"},
        },
        solution={"switch": "particle", "jump": "preposition", "fill": "particle", "wait": "preposition",
                  "look": "particle", "walk": "preposition"},
        nudge={
            "switch": {"en": "Can you also say *We switched the radio off*?",
                       "nl": "Hoe gaat de zin verder na *omdat we de radio …*?"},
            "jump": {"en": "Can you also say *The cat jumped the table off*?",
                     "nl": "Hoe gaat de zin verder na *omdat de kat …*?"},
            "fill": {"en": "Can you also say *Please fill the form in*?",
                     "nl": "Hoe gaat de zin verder na *omdat je het formulier …*?"},
            "wait": {"en": "Can you also say *We waited the hall in*?",
                     "nl": "Hoe gaat de zin verder na *omdat we …*, met *wachten* achteraan?"},
            "look": {"en": "Can you also say *They looked the word up*?",
                     "nl": "Hoe gaat de zin verder na *omdat we het woord …*?"},
            "walk": {"en": "Can you also say *They walked the hill up*?",
                     "nl": "Hoe gaat de zin verder na *omdat we …*, met *lopen* achteraan?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the sentences again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de zinnen opnieuw uit."},
            "empty": {"en": "Put each bold sentence under the group its word in italics belongs to.",
                      "nl": "Zet elke vetgedrukte zin bij de groep waar het schuingedrukte woord bij hoort."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs under another group.",
                      "nl": "{item} hoort bij een andere groep."},
            "correct": {"en": "All {n} are right. The Penn Treebank tags the particles RP and the prepositions IN. Universal Dependencies tags both ADP, and gives each particle the relation compound:prt to its verb: another guideline, just as consistent.",
                        "nl": "Alle {n} kloppen. In de bijzin plakt een partikel aan het werkwoord vast: *omdat we de radio uitzetten*, *omdat je het formulier invult*, *omdat we het woord opzoeken*. Een voorzetsel blijft bij het zelfstandig naamwoord: *omdat de kat uit de boom springt*. Universal Dependencies geeft beide de woordsoort ADP, en elk partikel de relatie compound:prt naar het werkwoord."},
        },
    ),
}

CASES = [
    ("kwic", _K_START, "empty", None),
    ("kwic", _K, "correct", None),
    ("kwic", "def kwic(words, target, width):\n    lines = []\n    for i in range(len(words)):\n        if words[i].lower() == target:\n            start = i - width\n            if i < width:\n                start = 0\n            lines.append((\" \".join(words[start:i]), words[i], \" \".join(words[i + 1:i + width + 1])))\n    return lines\n", "correct", None),
    ("kwic", "def kwic(words, target, width):\n    return [(\" \".join(words[i - width:i]) if i >= width else \" \".join(words[:i]), words[i], \" \".join(words[i + 1:i + 1 + width])) for i in range(len(words)) if words[i].lower() == target]\n", "correct", None),
    ("kwic", _K_NO_CLAMP, "wrong", "no-clamp"),
    ("kwic", _K_LEFT_KEY, "wrong", "left-key"),
    ("kwic", _K_RIGHT_KEY, "wrong", "right-key"),
    ("kwic", _K_SHORT, "wrong", "short"),
    ("kwic", _K_CASE, "wrong", "case"),
    ("kwic", _K_TARGET, "wrong", "target"),
    ("kwic", _K_LISTS, "wrong", "lists"),
    ("kwic", _K.replace("lines.append((left, words[i], right))", "lines.append([left, words[i], right])"), "wrong", "not-lines"),
    ("kwic", _K.replace("    return lines\n", "    return len(lines)\n"), "wrong", "not-a-list"),
    ("kwic", _K.replace("    return lines\n", "    len(lines)\n"), "wrong", "no-return"),
    ("kwic", "def kwic(words, target, width):\n    return " + repr(kwic(book_words("alice"), "rabbit", 3)) + "\n", "almost", None),
    ("particles", _P_START, "empty", None),
    ("particles", _P, "correct", None),
    ("particles", "def particles(sentence):\n    return [(sentence[i - 1][1], i) for i in range(1, len(sentence) + 1) if sentence[i - 1][2] == \"compound:prt\"]\n", "correct", None),
    ("particles", "def particles(sentence):\n    found = []\n    i = 0\n    for form, head, relation in sentence:\n        i = i + 1\n        if relation == \"compound:prt\":\n            found.append((head, i))\n    return found\n", "correct", None),
    ("particles", _P_SWAP, "wrong", "swap"),
    ("particles", _P_ZERO, "wrong", "zero"),
    ("particles", _P_ANY, "wrong", "any-compound"),
    ("particles", _P_FORMS, "wrong", "forms"),
    ("particles", _P.replace("found.append((head, i))", "found.append(head)"), "wrong", "no-pairs"),
    ("particles", _P.replace("sentence[i - 1]", "sentence[i]"), "error", "index"),
    ("particles", _P.replace("    return found\n", ""), "wrong", "no-return"),
    ("particles", "def particles(sentence):\n    return [(2, 5), (7, 9)]\n", "almost", None),
    ("label", {}, "empty", None),
    ("label", {"switch": "particle", "jump": "preposition", "fill": "particle", "wait": "preposition",
               "look": "particle", "walk": "preposition"}, "correct", None),
    ("label", {"switch": "particle", "jump": "preposition"}, "wrong", "unfinished"),
    ("label", {"walk": "particle"}, "wrong", "wrong:walk"),
    ("label", {"switch": "particle", "jump": "particle", "fill": "particle", "wait": "preposition",
               "look": "particle", "walk": "preposition"}, "wrong", "wrong:jump"),
]
