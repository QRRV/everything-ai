# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-normalize: fold a word to small letters without accents (fold), make only the first word
of a sentence small unless it is a name (truecase), and leave the stop words out of a list of words
(drop_stop_words), and sort tasks by what they need to keep: nothing of case and accents, the capitals, or the stop
words.

The words come from the Dutch sentences of Tatoeba (CC BY 2.0 FR) in data/tatoeba-nl.tsv.gz and from the eight books
of data/gutenberg-books.json.gz (public domain), cut into words by everything_ai.books. The stop words are the lists
of the Snowball project for English and Dutch (BSD-3-Clause) in data/snowball-stop-en.txt and data/snowball-stop-nl.txt.
The list of names is made by the course from Tatoeba: every word that the middle of a sentence writes with a capital
more often than in small letters. Every number in the pictures is counted by the course; the sentences of the idea
and of the cases, and the items of the sort, are the course's own."""
import collections
import contextlib
import copy
import html
import io
import re
import unicodedata

from everything_ai import books, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, kept, with_time_limit

# --- the course's solutions ------------------------------------------------------------------------------------------


def fold(word):
    """The course's solution of Exercise 1: the word in small letters, without accents or other marks on its
    letters: the form NFD splits a letter such as é into e and a combining mark, and the marks are left out."""
    decomposed = unicodedata.normalize("NFD", word.lower())
    return "".join([c for c in decomposed if unicodedata.combining(c) == 0])


def truecase(words, names):
    """The course's solution of Exercise 2: the words of a sentence with the first word in small letters, unless it
    is in `names`; the other words stay as they are."""
    if len(words) == 0 or words[0] in names:
        return words
    return [words[0].lower()] + words[1:]


def drop_stop_words(words, stop_words):
    """The course's solution of Exercise 3: the words, in order, without the ones whose small-letter form is in
    `stop_words`."""
    return [word for word in words if word.lower() not in stop_words]


def tokens(text):
    """The tokens of a text as in the lesson Cutting text into sentences and words: a run of letters, digits and _,
    or one character that is neither of these nor white space."""
    return re.findall(r"\w+|[^\w\s]", text)


# --- the data --------------------------------------------------------------------------------------------------------

# the course names nobody by a gendered pronoun: pictures leave out what holds one
_PRONOUNS = {"hij", "zij", "haar", "hem", "zijn", "ze", "he", "she", "him", "her", "his", "hers", "himself", "herself"}


def stop_words_kept(lang: str = "nl") -> frozenset:
    """The stop words of Snowball for Dutch (nl) or English (en), as a set that cannot be changed."""
    return kept(f"nlp-normalize:stop-{lang}", lambda: frozenset(books.stop_words(lang)))


def stop_words(lang: str = "nl") -> set:
    """The stop words of Snowball as a new set, for the lesson's STOP_WORDS and STOP_WORDS_EN."""
    return set(stop_words_kept(lang))


def _sentences() -> tuple:
    return tuple((sid, tuple(books.words(text))) for sid, text in books.dutch_sentences())


def sentences() -> tuple:
    """Every Dutch sentence of Tatoeba as (id, its words), in the order of the file; a sentence without words gives
    an empty tuple."""
    return kept("nlp-normalize:sentences", _sentences)


def words_of(sid: int) -> list:
    """The words of the sentence of Tatoeba with this id."""
    return list(kept("nlp-normalize:by-id", lambda: dict(sentences()))[sid])


def text_of(sid: int) -> str:
    return kept("nlp-normalize:text-by-id", lambda: dict(books.dutch_sentences()))[sid]


def forms() -> collections.Counter:
    """How often each word form occurs in the Dutch sentences of Tatoeba, as written: "Een" and "een" are two forms."""
    return kept("nlp-normalize:forms", lambda: collections.Counter(w for _, ws in sentences() for w in ws))


def middle() -> collections.Counter:
    """How often each word form occurs in the middle of a sentence of Tatoeba: every word but the first."""
    return kept("nlp-normalize:middle", lambda: collections.Counter(w for _, ws in sentences() for w in ws[1:]))


def _names() -> frozenset:
    mid = middle()
    return frozenset(w for w, n in mid.items() if w != w.lower() and n > mid[w.lower()])


def names_kept() -> frozenset:
    """The names of the course: every word that the middle of a sentence of Tatoeba writes with a capital more often
    than in small letters, such as Tom, Boston and IJsland."""
    return kept("nlp-normalize:names", _names)


def names() -> set:
    """The list of names as a new set, for the lesson's NAMES."""
    return set(names_kept())


def n_types(fn) -> int:
    """How many different words the forms of Tatoeba become under `fn`, such as str.lower or fold."""
    return len({fn(w) for w in forms()})


def accented(word: str) -> bool:
    """True for a form whose letters carry an accent or another mark: one that fold changes more than lower does."""
    return fold(word) != word.lower()


def share(words, stop) -> tuple:
    """(words of `words` in small letters in the set `stop`, all words)."""
    return sum(1 for w in words if w.lower() in stop), len(words)


def tatoeba_share() -> tuple:
    return kept("nlp-normalize:tatoeba-share",
                lambda: share([w for _, ws in sentences() for w in ws], stop_words_kept("nl")))


def book_shares() -> dict:
    """For each book: (stop words, words), with the list of the book's language."""
    return kept("nlp-normalize:book-shares",
                lambda: {b["key"]: share(books.book_words(b["key"]), stop_words_kept(b["lang"])) for b in books.books()})


NEGATIONS = {"niet", "geen", "niets"}


def with_negation() -> int:
    """How many sentences of Tatoeba hold niet, geen or niets, in any case."""
    return kept("nlp-normalize:negation",
                lambda: sum(1 for _, ws in sentences() if NEGATIONS & {w.lower() for w in ws}))


def all_stop_words() -> int:
    """How many sentences of Tatoeba have words, all of them stop words."""
    stop = stop_words_kept("nl")
    return kept("nlp-normalize:all-stop",
                lambda: sum(1 for _, ws in sentences() if ws and all(w.lower() in stop for w in ws)))


# --- the look of the pictures ------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums"
_LINE = "font-family:var(--ea-font-ui);font-size:0.9rem;color:var(--ea-ink);line-height:1.9"
_CODE = "font-family:var(--ea-font-code);font-size:0.82rem;color:var(--ea-ink)"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);margin-top:0.35rem"
_CHIP = ("display:inline-block;margin:0.12rem 0.2rem 0.12rem 0;padding:0 0.3rem;border-radius:3px;"
         "font-family:var(--ea-font-code);font-size:0.82rem;line-height:1.5;white-space:pre")
_CHIP_BLUE = _CHIP + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_CHIP_PURPLE = _CHIP + ";background:var(--ea-loss-wash);color:var(--ea-loss-text)"
_CHIP_DATA = _CHIP + ";border:1px solid var(--ea-line);color:var(--ea-ink)"


def _note(text: str) -> str:
    """A line under a picture; a word between asterisks is set in italics, as in the lesson's text."""
    parts = html.escape(text).split("*")
    marked = "".join(f"<i>{p}</i>" if k % 2 else p for k, p in enumerate(parts))
    return f'<div style="{_UI}">{marked}</div>'


def _num(n, lang: str) -> str:
    """A whole number with a separator for thousands: 2,319 in English, 2.319 in Dutch."""
    text = f"{round(n):,}"
    return text.replace(",", ".") if lang == "nl" else text


def _pct(part: int, whole: int, lang: str) -> str:
    """A share as a percentage with one decimal: 51.1% in English, 51,1% in Dutch."""
    text = f"{100 * part / whole:.1f}%" if whole else "–"
    return text.replace(".", ",") if lang == "nl" else text


def _table(rows_: list, head: list, numbers: tuple = ()) -> str:
    """A small table; rows hold HTML that is already escaped; numbers are the columns set to the right."""
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers else ""}">{c}</td>'
                                    for j, c in enumerate(row)) + "</tr>" for row in rows_)
    return (f'<table style="border-collapse:collapse;margin:0.3rem 0 0.2rem"><thead><tr>{top}</tr></thead>'
            f"<tbody>{body}</tbody></table>")


def _shown(text: str) -> str:
    """A string as a picture shows it: a combining mark on its own is written as its code, such as \\u0301."""
    return "".join(f"\\u{ord(c):04x}" if unicodedata.combining(c) else c for c in text)


def _chips(pieces: list, styles: list) -> str:
    return "".join(f'<span style="{s}">{html.escape(_shown(p))}</span>' for p, s in zip(pieces, styles))


# --- the idea's picture and the demo: a sentence step by step ---------------------------------------------------------

IDEA = {"en": "The US will not let us into the café.", "nl": "Tom was niet één keer in een café in Wenen."}
_STEP_WORDS = {
    "en": {"title": "One sentence, step by step", "demo": "Your text, step by step",
           "tokens": "The tokens: {n}",
           "lower": "Case folding, every letter small: {k} changed",
           "fold": "Accents and other marks removed: {k} changed",
           "stop": "Stop words left out: {n} tokens left",
           "gone": "Left out:",
           "list": {"en": "the English list of Snowball, {n} words", "nl": "the Dutch list of Snowball, {n} words"},
           "note": "Purple: a token that this step changed, or that it left out. Stop words: {lst}.",
           "nfc": "The text is put in the form NFC first, as most text already is.",
           "empty": "Type a text in the box.",
           "long": "This text is longer than {k} characters; try a shorter one.",
           "alt": "The text {s} step by step. {rows}. {note}"},
    "nl": {"title": "Eén zin, stap voor stap", "demo": "Jouw tekst, stap voor stap",
           "tokens": "De tokens: {n}",
           "lower": "Case folding, elke letter klein: {k} veranderd",
           "fold": "Accenten en andere tekens weg: {k} veranderd",
           "stop": "Stop words weggelaten: {n} tokens over",
           "gone": "Weggelaten:",
           "list": {"en": "de Engelse lijst van Snowball, {n} woorden", "nl": "de Nederlandse lijst van Snowball, {n} woorden"},
           "note": "Paars: een token dat deze stap veranderde, of dat de stap wegliet. Stop words: {lst}.",
           "nfc": "De tekst gaat eerst in de vorm NFC, zoals de meeste tekst al is.",
           "empty": "Typ een tekst in het vak.",
           "long": "Deze tekst is langer dan {k} tekens; probeer een kortere.",
           "alt": "De tekst {s} stap voor stap. {rows}. {note}"},
}
DEMO_LONGEST = 200


def steps(text: str, list_lang: str) -> dict:
    """The tokens of `text` after each step: as cut, in small letters, without accents, and without stop words, with
    the stop words that were left out."""
    cut = tokens(unicodedata.normalize("NFC", text))
    small = [t.lower() for t in cut]
    plain = [fold(t) for t in cut]
    stop = stop_words_kept(list_lang)
    return {"tokens": cut, "lower": small, "fold": plain,
            "stop": [t for t in plain if t not in stop], "gone": [t for t in plain if t in stop]}


def _steps_body(text: str, list_lang: str, lang: str) -> tuple:
    w = _STEP_WORDS[lang]
    s = steps(text, list_lang)
    rows_, alts = [], []

    def row(label, pieces, styles):
        rows_.append(f'<div style="{_LABEL}">{html.escape(label)}</div>' + _chips(pieces, styles))
        alts.append(label + ": " + " | ".join(_shown(p) for p in pieces))

    row(w["tokens"].format(n=len(s["tokens"])), s["tokens"], [_CHIP_DATA] * len(s["tokens"]))
    before = s["tokens"]
    for key in ("lower", "fold"):
        changed = [a != b for a, b in zip(before, s[key])]
        row(w[key].format(k=sum(changed)), s[key], [_CHIP_PURPLE if c else _CHIP_DATA for c in changed])
        before = s[key]
    row(w["stop"].format(n=len(s["stop"])), s["stop"], [_CHIP_DATA] * len(s["stop"]))
    if s["gone"]:
        rows_.append(f'<div><span style="{_UI}">{html.escape(w["gone"])}</span> '
                     + _chips(s["gone"], [_CHIP_PURPLE] * len(s["gone"])) + "</div>")
        alts.append(w["gone"] + " " + " | ".join(s["gone"]))
    lst = w["list"][list_lang].format(n=_num(len(stop_words_kept(list_lang)), lang))
    note = w["note"].format(lst=lst)
    return "".join(rows_), alts, note


def idea_picture(lang: str) -> str:
    w = _STEP_WORDS[lang]
    text = IDEA[lang]
    rows_, alts, note = _steps_body(text, lang, lang)
    body = f'<div style="{_LINE}">{html.escape(text)}</div>' + rows_ + _note(note)
    return yours.figure(w["title"], body, alt=w["alt"].format(s=text, rows="; ".join(alts), note=note))


def demo_picture(text, list_lang: str, lang: str) -> str:
    w = _STEP_WORDS[lang]
    text = str(text or "")
    if not text.strip():
        return yours.figure(w["demo"], _note(w["empty"]), alt=w["empty"])
    if len(text) > DEMO_LONGEST:
        msg = w["long"].format(k=DEMO_LONGEST)
        return yours.figure(w["demo"], _note(msg), alt=msg)
    list_lang = list_lang if list_lang in ("en", "nl") else "nl"
    rows_, alts, note = _steps_body(text, list_lang, lang)
    body = (f'<div style="{_LINE}">{html.escape(text)}</div>' + rows_ + _note(note) + _note(w["nfc"]))
    return yours.figure(w["demo"], body, alt=w["alt"].format(s=text, rows="; ".join(alts), note=note + " " + w["nfc"]))


# --- running the learner's function ----------------------------------------------------------------------------------


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error
PICTURE_LIMIT = 10.0   # seconds the learner's function gets for all words or sentences of a picture


def _call(make, seconds: float = PICTURE_LIMIT):
    """make() within a time limit, with what it prints thrown away; _STOPPED when it stops with an error, and the
    Timeout passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(make, seconds)
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _strings(v) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


def _run_or_say(make, w: dict, ok) -> tuple:
    """(what make() gives, None), or (None, a figure that says why there is no picture): an error, too slow, or a
    result that `ok` does not accept."""
    try:
        got = _call(make)
    except Timeout:
        text = w["slow"].format(s=PICTURE_LIMIT)
        return None, yours.figure(w["title"], _note(text), alt=text)
    if got is _STOPPED:
        return None, yours.figure(w["title"], _note(w["stopped"]), alt=w["stopped"])
    if not ok(got):
        return None, yours.figure(w["title"], _note(w["odd"]), alt=w["odd"])
    return got, None


# --- the picture under exercise 1: what your fold merges in Tatoeba ----------------------------------------------------

_FOLD_WORDS = {
    "en": {"title": "What your fold merges",
           "head": ["the forms of Tatoeba", "different words"],
           "written": "as written", "lower": "every letter small, with lower", "yours": "your fold",
           "groups": "The accented forms your fold merges most often, with how often each form occurs:",
           "note": "Blue: what your fold gives. The forms are the {n} different words of the Dutch sentences of "
                   "Tatoeba; {a} of them carry an accent or another mark, and they occur {t} times.",
           "stopped": "Your function stopped with an error on one of the words, so the picture stays empty.",
           "slow": "Your function took more than {s:g} seconds for all words, so the picture stays empty.",
           "odd": "Your function does not give back a string for every word, so the picture stays empty."},
    "nl": {"title": "Wat je fold samenvoegt",
           "head": ["de vormen van Tatoeba", "verschillende woorden"],
           "written": "zoals geschreven", "lower": "elke letter klein, met lower", "yours": "jouw fold",
           "groups": "De vormen met een accent die je fold het vaakst samenvoegt, met hoe vaak elke vorm voorkomt:",
           "note": "Blauw: wat je fold geeft. De vormen zijn de {n} verschillende woorden van de Nederlandse zinnen "
                   "van Tatoeba; {a} ervan hebben een accent of een ander teken, en samen komen deze vormen {t} keer voor.",
           "stopped": "Je functie stopte met een fout op een van de woorden, dus de afbeelding blijft leeg.",
           "slow": "Je functie deed er langer dan {s:g} seconden over voor alle woorden, dus de afbeelding blijft leeg.",
           "odd": "Je functie geeft niet voor elk woord een string terug, dus de afbeelding blijft leeg."},
}
FOLD_SHOWN = 5   # groups the picture shows


def fold_groups(results: dict) -> list:
    """The groups of forms that share a result, those with the most occurrences of accented forms first, as
    (result, [(form, count), ...] with the most common form first); groups with a pronoun stay out."""
    count = forms()
    groups = collections.defaultdict(list)
    for form, got in results.items():
        groups[got].append((form, count[form]))
    out = []
    for got, members in groups.items():
        weight = sum(n for f, n in members if accented(f))
        if weight and len(members) > 1 and not ({got.lower()} | {f.lower() for f, _ in members}) & _PRONOUNS:
            out.append((weight, got, sorted(members, key=lambda fn: (-fn[1], fn[0]))))
    out.sort(key=lambda t: (-t[0], t[1]))
    return [(got, members) for _, got, members in out]


def fold_picture(seen) -> str:
    lang = seen.lang
    w = _FOLD_WORDS[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    words = list(forms())
    got, said = _run_or_say(lambda: [fn(x) for x in words], w, _strings)
    if said:
        return said
    results = dict(zip(words, got))
    k = len(set(got))
    rows_ = [[html.escape(w["written"]), _num(len(words), lang)],
             [html.escape(w["lower"]), _num(n_types(str.lower), lang)],
             [html.escape(w["yours"]), _num(k, lang)]]
    shown = fold_groups(results)[:FOLD_SHOWN]
    lines = "".join(f'<div>{_chips([g], [_CHIP_BLUE])} '
                    + _chips([f"{f} {_num(n, lang)}" for f, n in members[:5]], [_CHIP_DATA] * min(5, len(members)))
                    + "</div>" for g, members in shown)
    n_acc = [f for f in words if accented(f)]
    note = w["note"].format(n=_num(len(words), lang), a=_num(len(n_acc), lang),
                            t=_num(sum(forms()[f] for f in n_acc), lang))
    alt = (w["title"] + ". " + "; ".join(f"{html.unescape(r[0])}: {r[1]}" for r in rows_) + ". " + w["groups"] + " "
           + "; ".join(f"{_shown(g)}: " + ", ".join(f"{f} {n}" for f, n in members[:5]) for g, members in shown)
           + ". " + note)
    body = (_table(rows_, w["head"], numbers=(1,)) + (f'<div style="{_LABEL}">{html.escape(w["groups"])}</div>' + lines
                                                      if shown else "") + _note(note))
    return yours.figure(w["title"], body, alt=alt)


# --- the picture under exercise 2: your truecase on Tatoeba ------------------------------------------------------------

# the sentences of Tatoeba the picture shows, by id, with what the course says of each
TRUECASE_SHOWN = [6617223, 686790, 711890, 7463167]
_TRUE_WORDS = {
    "en": {"title": "Your truecase on Tatoeba",
           "head": ["the words of Tatoeba", "different words"],
           "written": "as written", "all": "every word small", "first": "only the first word small",
           "yours": "your truecase",
           "examples": "Four sentences, with what your function makes of them:",
           "says": {6617223: "Tom is not home: *Tom* is a name, and keeps its capital.",
                    686790: "The lake is full of fish: *het* is a small word at the start of the sentence.",
                    711890: "Crying will not help: *wenen* is the verb, and small is right.",
                    7463167: "Vienna is the capital of Austria: the city *Wenen* now looks like the verb."},
           "note": "Purple: a word that your function changed. The list of names holds the {n} words that the middle "
                   "of a sentence of Tatoeba writes with a capital more often than in small letters.",
           "stopped": "Your function stopped with an error on one of the sentences, so the picture stays empty.",
           "slow": "Your function took more than {s:g} seconds for all sentences, so the picture stays empty.",
           "odd": "Your function does not give back a list of strings for every sentence, so the picture stays empty."},
    "nl": {"title": "Jouw truecase op Tatoeba",
           "head": ["de woorden van Tatoeba", "verschillende woorden"],
           "written": "zoals geschreven", "all": "elk woord klein", "first": "alleen het eerste woord klein",
           "yours": "jouw truecase",
           "examples": "Vier zinnen, met wat je functie ervan maakt:",
           "says": {6617223: "*Tom* is een naam, en houdt de hoofdletter.",
                    686790: "*Het* is een klein woord aan het begin van de zin.",
                    711890: "Hier is *wenen* het werkwoord, en klein klopt.",
                    7463167: "De stad *Wenen* lijkt nu op het werkwoord."},
           "note": "Paars: een woord dat je functie veranderde. De lijst met namen heeft de {n} woorden die het midden "
                   "van een zin van Tatoeba vaker met een hoofdletter schrijft dan in kleine letters.",
           "stopped": "Je functie stopte met een fout op een van de zinnen, dus de afbeelding blijft leeg.",
           "slow": "Je functie deed er langer dan {s:g} seconden over voor alle zinnen, dus de afbeelding blijft leeg.",
           "odd": "Je functie geeft niet voor elke zin een lijst met strings terug, dus de afbeelding blijft leeg."},
}


def type_counts() -> dict:
    """Different words in Tatoeba as written, with every word small, and with only the first word of each sentence
    small."""
    def count():
        all_words = [w for _, ws in sentences() for w in ws]
        first = {w for _, ws in sentences() for w in ([ws[0].lower()] + list(ws[1:]) if ws else [])}
        return {"written": len(set(all_words)), "all": len({w.lower() for w in all_words}), "first": len(first)}
    return kept("nlp-normalize:type-counts", count)


def truecase_picture(seen) -> str:
    lang = seen.lang
    w = _TRUE_WORDS[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    known = names_kept()
    sents = [list(ws) for _, ws in sentences()]
    got, said = _run_or_say(lambda: [fn(ws, known) for ws in sents], w, lambda g: all(_strings(x) for x in g))
    if said:
        return said
    t = type_counts()
    k = len({x for g in got for x in g})
    rows_ = [[html.escape(w[key]), _num(t[key], lang)] for key in ("written", "all", "first")]
    rows_.append([html.escape(w["yours"]), _num(k, lang)])
    by_id = {sid: g for (sid, _), g in zip(sentences(), got)}
    examples = []
    for sid in TRUECASE_SHOWN:
        before, after = words_of(sid), by_id[sid]
        styles = [_CHIP_PURPLE if i >= len(before) or a != before[i] else _CHIP_DATA for i, a in enumerate(after)]
        examples.append(f'<div style="border-bottom:1px solid var(--ea-line);padding:0.2rem 0">'
                        f'<div style="{_CODE};color:var(--ea-ink-2)">{html.escape(text_of(sid))}</div>'
                        f"<div>{_chips(after, styles)}</div>{_note(w['says'][sid])}</div>")
    note = w["note"].format(n=_num(len(known), lang))
    alt = (w["title"] + ". " + "; ".join(f"{html.unescape(r[0])}: {r[1]}" for r in rows_) + ". " + w["examples"] + " "
           + "; ".join(f"{text_of(sid)} {' '.join(by_id[sid])} ({w['says'][sid].replace('*', '')})" for sid in TRUECASE_SHOWN)
           + ". " + note)
    body = (_table(rows_, w["head"], numbers=(1,)) + f'<div style="{_LABEL}">{html.escape(w["examples"])}</div>'
            + "".join(examples) + _note(note))
    return yours.figure(w["title"], body, alt=alt)


# --- the picture under exercise 3: what the stop words take ------------------------------------------------------------

DROP_SHOWN = [6617223, 3524160, 686790]
_DROP_WORDS = {
    "en": {"title": "What your function leaves out",
           "head": ["text", "words", "left out", "share"],
           "tatoeba": "Tatoeba, the Dutch sentences",
           "line": "Each text with the list of its language: the Dutch list of {nl} words, or the English list of {en}.",
           "lost": "{k} of the {n} sentences of Tatoeba with *niet*, *geen* or *niets* lose that word, and {e} "
                   "sentences lose every word.",
           "lost_all": "All {n} sentences of Tatoeba with *niet*, *geen* or *niets* lose that word, and {e} sentences "
                       "lose every word.",
           "examples": "Three sentences of Tatoeba, with what is left:",
           "says": {6617223: "Tom is not home: without *niet*, the rest says the opposite.",
                    3524160: "A bee is buzzing: *bij* (bee) is on the list as *bij* (by).",
                    686790: "The lake is full of fish: *meer* (lake) is on the list as *meer* (more)."},
           "stopped": "Your function stopped with an error on one of the texts, so the picture stays empty.",
           "slow": "Your function took more than {s:g} seconds for all texts, so the picture stays empty.",
           "odd": "Your function does not give back a list of strings for every text, so the picture stays empty."},
    "nl": {"title": "Wat je functie weglaat",
           "head": ["tekst", "woorden", "weggelaten", "deel"],
           "tatoeba": "Tatoeba, de Nederlandse zinnen",
           "line": "Elke tekst met de lijst van de eigen taal: de Nederlandse lijst van {nl} woorden, of de Engelse van "
                   "{en}.",
           "lost": "{k} van de {n} zinnen van Tatoeba met *niet*, *geen* of *niets* raken dat woord kwijt, en {e} "
                   "zinnen raken elk woord kwijt.",
           "lost_all": "Alle {n} zinnen van Tatoeba met *niet*, *geen* of *niets* raken dat woord kwijt, en {e} zinnen "
                       "raken elk woord kwijt.",
           "examples": "Drie zinnen van Tatoeba, met wat overblijft:",
           "says": {6617223: "Zonder *niet* zegt de rest het omgekeerde.",
                    3524160: "*bij* (het dier) staat op de lijst als *bij* (het voorzetsel).",
                    686790: "*meer* (het water) staat op de lijst als *meer* (de hoeveelheid)."},
           "stopped": "Je functie stopte met een fout op een van de teksten, dus de afbeelding blijft leeg.",
           "slow": "Je functie deed er langer dan {s:g} seconden over voor alle teksten, dus de afbeelding blijft leeg.",
           "odd": "Je functie geeft niet voor elke tekst een lijst met strings terug, dus de afbeelding blijft leeg."},
}


def drop_picture(seen) -> str:
    lang = seen.lang
    w = _DROP_WORDS[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    nl, en = stop_words_kept("nl"), stop_words_kept("en")
    sents = [list(ws) for _, ws in sentences()]
    keys = [b["key"] for b in books.books()]

    def run():
        mine = [fn(ws, nl) for ws in sents]
        return mine, [fn(books.book_words(k), en if books.book(k)["lang"] == "en" else nl) for k in keys]

    got, said = _run_or_say(run, w, lambda g: all(_strings(x) for x in g[0]) and all(_strings(x) for x in g[1]))
    if said:
        return said
    mine, in_books = got
    total = sum(len(ws) for ws in sents)
    left = sum(len(g) for g in mine)
    rows_ = [[html.escape(w["tatoeba"]), _num(total, lang), _num(total - left, lang), _pct(total - left, total, lang)]]
    for k, g in zip(keys, in_books):
        n = len(books.book_words(k))
        rows_.append([f"<i>{html.escape(books.book(k)['title'])}</i>", _num(n, lang), _num(n - len(g), lang),
                      _pct(n - len(g), n, lang)])
    with_neg = [(ws, g) for ws, g in zip(sents, mine) if NEGATIONS & {x.lower() for x in ws}]
    lost = sum(1 for ws, g in with_neg if not (NEGATIONS & {x.lower() for x in g}))
    empty = sum(1 for ws, g in zip(sents, mine) if ws and not g)
    lost_line = w["lost_all" if lost == len(with_neg) else "lost"].format(k=_num(lost, lang), n=_num(len(with_neg), lang),
                                                                         e=_num(empty, lang))
    by_id = {sid: g for (sid, _), g in zip(sentences(), mine)}
    examples = "".join(f'<div style="border-bottom:1px solid var(--ea-line);padding:0.2rem 0">'
                       f'<div style="{_CODE};color:var(--ea-ink-2)">{html.escape(text_of(sid))}</div>'
                       f"<div>{_chips(by_id[sid], [_CHIP_BLUE] * len(by_id[sid]))}</div>{_note(w['says'][sid])}</div>"
                       for sid in DROP_SHOWN)
    line = w["line"].format(nl=len(nl), en=len(en))
    alt = (w["title"] + ". " + line + " " + "; ".join(f"{html.unescape(re.sub('<[^>]+>', '', r[0]))}: {r[1]} words, "
                                                      f"{r[2]} left out ({r[3]})" for r in rows_) + ". "
           + lost_line.replace("*", "") + " " + w["examples"] + " "
           + "; ".join(f"{text_of(sid)} {' '.join(by_id[sid])} ({w['says'][sid].replace('*', '')})" for sid in DROP_SHOWN))
    body = (f'<div style="{_LINE}">{html.escape(line)}</div>' + _table(rows_, w["head"], numbers=(1, 2, 3))
            + _note(lost_line) + f'<div style="{_LABEL}">{html.escape(w["examples"])}</div>' + examples)
    return yours.figure(w["title"], body, alt=alt)


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["fold", "truecase", "drop-stop-words", "what-to-keep"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: "..." in "\n".join(
                    line.split("#")[0] for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang het door je eigen code.")


def _value(v) -> str:
    """A short value the way the lessons write it: strings in double quotes, also inside a list or a set, and a
    combining mark as its code, such as \\u0301."""
    if isinstance(v, str):
        return '"' + _shown(v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")) + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_value(p) for p in v) + "]"
    if isinstance(v, (set, frozenset)):
        return "{" + ", ".join(_value(p) for p in sorted(v)) + "}" if v else "set()"
    return repr(v)


class Named:
    """A value of the lesson that a message shows by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


class _Lazy:
    """A value of the lesson made when a check first needs it, and shown in a message by its name. `make` gives a
    value that cannot be changed, such as a frozenset."""

    def __init__(self, name: str, make):
        self.name, self.make = name, make

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return self.make()


def _lit(v) -> Named:
    """A value of the check's own, shown in a message as Python, with strings in double quotes."""
    return Named(_value(v), v)


NAMES_N = _Lazy("NAMES", names_kept)
STOP_WORDS_N = _Lazy("STOP_WORDS", lambda: stop_words_kept("nl"))
STOP_WORDS_EN_N = _Lazy("STOP_WORDS_EN", lambda: stop_words_kept("en"))
DECOMPOSED = "cafe\u0301"   # café written with e and the combining acute accent


def _a_list(got, text) -> bool:
    return isinstance(got, list)


def _a_set(got, text) -> bool:
    return isinstance(got, (set, frozenset))


# exercise 1: fold. Besides "Café": the number één, capitals with a trema, two tremas, the capital IJ, an
# apostrophe, café written with a combining accent, a cedilla, and an empty string.
FOLD_CASES = [
    (_lit("Café"),),
    (_lit("één"),),
    (_lit("NAÏEF"),),
    (_lit("coëfficiënt"),),
    (_lit("IJsland"),),
    (_lit("zo'n"),),
    (Named('"cafe\\u0301"', DECOMPOSED),),
    (_lit("Ça"),),
    (_lit(""),),
]
_HEAD_U = "import unicodedata\n\n"
_F_START = _HEAD_U + "def fold(word):\n    ...\n"
_F = (_HEAD_U + "def fold(word):\n"
      "    decomposed = unicodedata.normalize(\"NFD\", word.lower())\n"
      "    return \"\".join([c for c in decomposed if unicodedata.combining(c) == 0])\n")
_F_NO_LOWER = _F.replace("word.lower()", "word")
_F_NO_NFD = _F.replace("unicodedata.normalize(\"NFD\", word.lower())", "word.lower()")
_F_NFC = _F.replace("\"NFD\"", "\"NFC\"")
_F_MARKS = _F.replace("== 0", "!= 0")
_F_LIST = _F.replace("return \"\".join([c for c in decomposed if unicodedata.combining(c) == 0])",
                     "return [c for c in decomposed if unicodedata.combining(c) == 0]")
_F_SPACES = _F.replace("return \"\".join(", "return \" \".join(")

# exercise 2: truecase. Besides the visible sentence: a name at the start, an empty sentence, a name in the middle
# that looks like a verb, Eén at the start, the capital IJ, a word in capitals, a list of names of its own, and an
# empty list of names.
TRUE_CASES = [
    (_lit(["Het", "regent", "in", "Wenen"]), NAMES_N),
    (_lit(["Tom", "slaapt"]), NAMES_N),
    (_lit([]), NAMES_N),
    (_lit(["Ik", "ken", "Ken"]), NAMES_N),
    (_lit(["Eén", "café", "graag"]), NAMES_N),
    (_lit(["IJsland", "is", "koud"]), NAMES_N),
    (_lit(["WAAROM"]), NAMES_N),
    (_lit(["Boston", "Boston"]), _lit({"Boston"})),
    (_lit(["Boston", "Boston"]), _lit(set())),
]
_T_START = "def truecase(words, names):\n    ...\n"
_T = ("def truecase(words, names):\n"
      "    if len(words) == 0 or words[0] in names:\n"
      "        return words\n"
      "    return [words[0].lower()] + words[1:]\n")
_T_ALL = "def truecase(words, names):\n    return [w.lower() for w in words]\n"
_T_NO_NAMES = _T.replace(" or words[0] in names", "")
_T_FIRST_ONLY = _T.replace("[words[0].lower()] + words[1:]", "[words[0].lower()]")
_T_NAMES_SMALL = _T.replace("if len(words) == 0 or words[0] in names:", "if len(words) == 0 or words[0] not in names:")

# exercise 3: drop_stop_words. Besides the visible sentence: a stop word with a capital, an empty list, a homonym of a
# stop word, a phrase of stop words only with the English list, US and us, and a punctuation mark.
DROP_CASES = [
    (_lit(["Tom", "is", "niet", "thuis"]), STOP_WORDS_N),
    (_lit(["Ik", "ben", "moe"]), STOP_WORDS_N),
    (_lit([]), STOP_WORDS_N),
    (_lit(["Een", "bij", "zoemt"]), STOP_WORDS_N),
    (_lit(["to", "be", "or", "not", "to", "be"]), STOP_WORDS_EN_N),
    (_lit(["US", "and", "us"]), STOP_WORDS_EN_N),
    (_lit(["Wat", "?"]), STOP_WORDS_N),
    (_lit(["Café", "De", "Zon"]), _lit({"de"})),
]
_D_START = "def drop_stop_words(words, stop_words):\n    ...\n"
_D = ("def drop_stop_words(words, stop_words):\n"
      "    return [word for word in words if word.lower() not in stop_words]\n")
_D_NO_LOWER = _D.replace("if word.lower() not in", "if word not in")
_D_ONLY = _D.replace(" not in stop_words", " in stop_words")
_D_LOWERED = _D.replace("[word for word", "[word.lower() for word")
_D_SET = ("def drop_stop_words(words, stop_words):\n"
          "    return {word for word in words if word.lower() not in stop_words}\n")

EXERCISES = {
    "fold": FunctionBox(
        name="fold",
        picture=fold_picture,
        function="fold",
        start=_F_START,
        solution=_F,
        visible=FOLD_CASES[:1],
        hidden=FOLD_CASES[1:],
        first=[_DOTS],
        show=_value,
        mistakes=[
            Mistake("list", matches=_a_list,
                    en="`{call}` gives {got}, and it should give {want}: a list of characters, not a string. Glue the characters together with `\"\".join(...)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een lijst met tekens, geen string. Plak de tekens aan elkaar met `\"\".join(...)`."),
            Mistake("spaces", wrong=_F_SPACES,
                    en="`{call}` gives {got}, and it should give {want}: `\" \".join` puts a space between every two characters. Join them with the empty string `\"\"`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `\" \".join` zet een spatie tussen elke twee tekens. Plak de tekens aan elkaar met de lege string `\"\"`."),
            Mistake("no-lower", wrong=_F_NO_LOWER,
                    en="`{call}` gives {got}, and it should give {want}: the accent is gone, but the capital stays. Make the word small too, with `.lower()`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het accent is weg, maar de hoofdletter blijft. Maak het woord ook klein, met `.lower()`."),
            Mistake("no-nfd", wrong=_F_NO_NFD,
                    en="`{call}` gives {got}, and it should give {want}: in `\"café\"`, é is one character, and `unicodedata.combining` gives 0 for it, so it stays. Split each letter from its marks first, with the form NFD.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: in `\"café\"` is é één teken, en `unicodedata.combining` geeft er 0 voor, dus het blijft staan. Haal eerst elke letter los van de tekens erop, met de vorm NFD."),
            Mistake("nfc", wrong=_F_NFC,
                    en="`{call}` gives {got}, and it should give {want}: the form NFC puts a letter and its accent together as one character, such as é, and `unicodedata.combining` gives 0 for that. The form that splits them is NFD.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de vorm NFC zet een letter en het accent erop samen als één teken, zoals é, en `unicodedata.combining` geeft daar 0 voor. De vorm die de letter en het accent uit elkaar haalt, is NFD."),
            Mistake("marks", wrong=_F_MARKS,
                    en="`{call}` gives {got}, and it should give {want}: your function keeps only the combining marks and leaves the letters out. Keep the letters, and leave out the combining marks.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie houdt alleen de combinerende tekens, en laat de letters weg. Houd de letters, en laat de combinerende tekens weg."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: make the word small, split each letter from its marks with the form NFD, and glue the other characters together again: those that are not a combining mark.",
                      "nl": "Vervang de `...`: maak het woord klein, haal elke letter los van de tekens erop met de vorm NFD, en plak de tekens die geen combinerend teken zijn weer aan elkaar."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function make the word small, take it apart with NFD, and leave out every combining mark?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Maakt je functie het woord klein, haalt je functie het uit elkaar met NFD, en laat je functie elk combinerend teken weg?"},
            "almost": {"en": "Right for \"Café\". But `{call}` gives {got}, and it should give {want}. Does your function work for every word: a trema, a cedilla, capitals, a word without accents?",
                       "nl": "Goed voor \"Café\". Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elk woord: een trema, een cedille, hoofdletters, een woord zonder accenten?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back. Does it end with `return`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie met `return`?"},
            "correct": {"en": "Your `fold` is right in all {n} cases. Look at the picture: your function on every word form of the Dutch sentences of Tatoeba, and which forms it merges.",
                        "nl": "Je `fold` klopt in alle {n} gevallen. Kijk naar de afbeelding: je functie op elke woordvorm van de Nederlandse zinnen van Tatoeba, en welke vormen je functie samenvoegt."},
        },
    ),
    "truecase": FunctionBox(
        name="truecase",
        picture=truecase_picture,
        function="truecase",
        start=_T_START,
        solution=_T,
        visible=TRUE_CASES[:1],
        hidden=TRUE_CASES[1:],
        first=[_DOTS],
        show=_value,
        mistakes=[
            Mistake("all-small", wrong=_T_ALL,
                    en="`{call}` gives {got}, and it should give {want}: your function makes every word small, also a name in the middle of the sentence. Only the first word gets a capital just for being first.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie maakt elk woord klein, ook een naam midden in de zin. Alleen het eerste woord krijgt een hoofdletter alleen omdat het vooraan staat."),
            Mistake("no-names", wrong=_T_NO_NAMES,
                    en="`{call}` gives {got}, and it should give {want}: a first word that is in `names` keeps its capital, as it is written. Look the first word up in `names` before you make it small.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een eerste woord dat in `names` staat, houdt de hoofdletter, zoals het geschreven is. Zoek het eerste woord op in `names` voordat je het klein maakt."),
            Mistake("names-small", wrong=_T_NAMES_SMALL,
                    en="`{call}` gives {got}, and it should give {want}: your function makes the first word small when it *is* a name, and keeps the capital when it is not. Turn the test around.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie maakt het eerste woord klein als het een naam *is*, en houdt de hoofdletter als het geen naam is. Draai de test om."),
            Mistake("first-only", wrong=_T_FIRST_ONLY,
                    en="`{call}` gives {got}, and it should give {want}: the other words of the sentence are gone. Add them after the first word, with `words[1:]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de andere woorden van de zin zijn weg. Zet de andere woorden achter het eerste woord, met `words[1:]`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back the words as they are when the list is empty or its first word is a name, and otherwise the first word small, followed by the other words.",
                      "nl": "Vervang de `...`: geef `words` onveranderd terug als de lijst leeg is of het eerste woord een naam is, en anders het eerste woord klein, gevolgd door de andere woorden."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function leave a list without words and a first word in `names` alone, and otherwise make only the first word small?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Geeft je functie `words` onveranderd terug bij een lijst zonder woorden en bij een eerste woord in `names`, en maakt je functie anders alleen het eerste woord klein?"},
            "almost": {"en": "Right for the sentence about Wenen. But `{call}` gives {got}, and it should give {want}. Does your function work for every sentence: one that starts with a name, one without words?",
                       "nl": "Goed voor de zin over Wenen. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke zin: een die met een naam begint, een zonder woorden?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back. Does every way through it end with `return`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt elke weg erdoorheen met `return`?"},
            "correct": {"en": "Your `truecase` is right in all {n} cases. Look at the picture: your function on every sentence of Tatoeba, and on four sentences that show what it can and cannot tell.",
                        "nl": "Je `truecase` klopt in alle {n} gevallen. Kijk naar de afbeelding: je functie op elke zin van Tatoeba, en op vier zinnen die laten zien wat je functie wel en niet kan zien."},
        },
    ),
    "drop-stop-words": FunctionBox(
        name="drop-stop-words",
        picture=drop_picture,
        function="drop_stop_words",
        start=_D_START,
        solution=_D,
        visible=DROP_CASES[:1],
        hidden=DROP_CASES[1:],
        first=[_DOTS],
        show=_value,
        mistakes=[
            Mistake("a-set", matches=_a_set,
                    en="`{call}` gives {got}, and it should give {want}: a set has no order and holds each word once. Give back a list, with square brackets.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een set heeft geen volgorde en heeft elk woord maar één keer. Geef een lijst terug, met vierkante haken."),
            Mistake("no-lower", wrong=_D_NO_LOWER,
                    en="`{call}` gives {got}, and it should give {want}: the lists of Snowball hold small letters, so a stop word with a capital, such as *Ik* at the start of a sentence, stays. Look the word up in small letters.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de lijsten van Snowball hebben kleine letters, dus een stop word met een hoofdletter, zoals *Ik* aan het begin van een zin, blijft staan. Zoek het woord op in kleine letters."),
            Mistake("only-stop", wrong=_D_ONLY,
                    en="`{call}` gives {got}, and it should give {want}: your function keeps the stop words and leaves the other words out. Turn the test around.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie houdt de stop words en laat de andere woorden weg. Draai de test om."),
            Mistake("lowered", wrong=_D_LOWERED,
                    en="`{call}` gives {got}, and it should give {want}: the words that stay should stay as they are written, with their capitals. Use the small letters only to look a word up.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de woorden die blijven, moeten onveranderd blijven, met hun hoofdletters. Gebruik de kleine letters alleen om een woord op te zoeken."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back, in order, the words whose small-letter form is not in `stop_words`.",
                      "nl": "Vervang de `...`: geef de woorden terug, in volgorde, waarvan de vorm in kleine letters niet in `stop_words` staat."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function keep each word whose small-letter form is not in `stop_words`, as it is written and in its order?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Houdt je functie elk woord waarvan de vorm in kleine letters niet in `stop_words` staat, zoals het geschreven is en in de volgorde?"},
            "almost": {"en": "Right for the sentence about Tom. But `{call}` gives {got}, and it should give {want}. Does your function work for every list: a capital, no words at all, the English list?",
                       "nl": "Goed voor de zin over Tom. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke lijst: een hoofdletter, een lege lijst, de Engelse lijst?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back. Does it end with `return`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie met `return`?"},
            "correct": {"en": "Your `drop_stop_words` is right in all {n} cases. Look at the picture: your function on all sentences of Tatoeba and on the eight books, and what it takes away.",
                        "nl": "Je `drop_stop_words` klopt in alle {n} gevallen. Kijk naar de afbeelding: je functie op alle zinnen van Tatoeba en op de acht boeken, en wat je functie weghaalt."},
        },
    ),
    "what-to-keep": Sort(
        name="what-to-keep",
        groups={
            "fold": {"en": "Fold case and accents: the spelling may vary",
                     "nl": "Hoofdletters en accenten weg: de spelling mag verschillen"},
            "capitals": {"en": "Keep the capitals: they tell words apart",
                         "nl": "Houd de hoofdletters: een hoofdletter houdt woorden uit elkaar"},
            "stop": {"en": "Keep the stop words: small words carry the meaning",
                     "nl": "Houd de stop words: kleine woorden dragen de betekenis"},
        },
        items={
            "ferrari": {"en": "**A search box that finds pages about a *Ferrari* when someone types *ferrari*.**",
                        "nl": "**Een zoekvak dat pagina's over een *Ferrari* vindt als iemand *ferrari* typt.**"},
            "creme": {"en": "**Someone types *creme brulee* and should find recipes for *crème brûlée*.**",
                      "nl": "**Iemand typt *creme brulee* en moet recepten voor *crème brûlée* vinden.**"},
            "naief": {"en": "**Count how often e-mails use the word *naïef*, also where someone typed *naief* or *Naïef*.**",
                      "nl": "**Tel hoe vaak e-mails het woord *naïef* gebruiken, ook waar iemand *naief* of *Naïef* typte.**"},
            "fed": {"en": "**Tell *the Fed*, the central bank of the United States, from the verb *fed*.**",
                    "nl": "**Houd *the Fed*, de centrale bank van de Verenigde Staten, apart van het Engelse werkwoord *fed*.**"},
            "zuiden": {"en": "**Tell *het Zuiden*, the South as a political and economic area, from *het zuiden van Frankrijk*, the south of France.**",
                       "nl": "**Houd *het Zuiden* als politiek en economisch gebied apart van *het zuiden van Frankrijk*.**"},
            "black": {"en": "**Tell the family name *Black* from the color *black*.**",
                      "nl": "**Houd de achternaam *Black* apart van de Engelse kleur *black*.**"},
            "letitbe": {"en": "**Search for the song title *Let It Be*.**",
                        "nl": "**Zoek de titel van het nummer *Let It Be*.**"},
            "utrecht": {"en": "**Find trains *to Utrecht*, and not trains *from Utrecht*.**",
                        "nl": "**Vind treinen *naar Utrecht*, en niet treinen *uit Utrecht*.**"},
            "review": {"en": "**Tell the review *Het eten is niet lekker* from *Het eten is lekker*.**",
                       "nl": "**Houd de recensie *Het eten is niet lekker* apart van *Het eten is lekker*.**"},
        },
        solution={"ferrari": "fold", "creme": "fold", "naief": "fold", "fed": "capitals", "zuiden": "capitals",
                  "black": "capitals", "letitbe": "stop", "utrecht": "stop", "review": "stop"},
        nudge={
            "ferrari": {"en": "Does it matter to the person searching whether the F is a capital?",
                        "nl": "Maakt het voor wie zoekt uit of de F een hoofdletter is?"},
            "creme": {"en": "Which characters does the person leave out, and should the search mind?",
                      "nl": "Welke tekens laat deze persoon weg, en moet het zoeken daar last van hebben?"},
            "naief": {"en": "Are these three different words, or one word written in three ways?",
                      "nl": "Zijn dit drie verschillende woorden, of één woord op drie manieren geschreven?"},
            "fed": {"en": "Which letter shows that the bank is meant?",
                    "nl": "Welke letter laat zien dat de bank bedoeld is?"},
            "zuiden": {"en": "Which letter shows that the area is meant, and not a direction?",
                       "nl": "Welke letter laat zien dat het gebied bedoeld is, en niet een richting?"},
            "black": {"en": "What still tells the name from the color once both are written in small letters?",
                      "nl": "Wat houdt de naam nog apart van de kleur als je allebei in kleine letters schrijft?"},
            "letitbe": {"en": "Which words of this title are very common, and what is left without them?",
                        "nl": "Welke woorden van deze titel komen heel vaak voor, en wat blijft er zonder die woorden over?"},
            "utrecht": {"en": "Which word shows the direction of the trip?",
                        "nl": "Welk woord geeft de richting van de reis aan?"},
            "review": {"en": "Which word makes the two reviews differ, and how common is it?",
                       "nl": "Welk woord maakt het verschil tussen de twee recensies, en hoe vaak komt het voor?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the cases again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de gevallen opnieuw uit."},
            "empty": {"en": "Put each bold task under what it needs.",
                      "nl": "Zet elke vetgedrukte taak bij wat de taak nodig heeft."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item}: this task goes somewhere else.",
                      "nl": "{item}: deze taak hoort ergens anders."},
            "correct": {"en": "All {n} are right. Folding helps where people write one word in more than one way, capitals tell a name from a common word, and small words such as *not* and *to* can carry the meaning.",
                        "nl": "Alle {n} kloppen. Hoofdletters en accenten weghalen helpt waar mensen één woord op meer dan één manier schrijven, hoofdletters houden een naam apart van een gewoon woord, en kleine woorden zoals *niet* en *naar* kunnen de betekenis dragen."},
        },
    ),
}

CASES = [
    ("fold", _F_START, "empty", None),
    ("fold", _F, "correct", None),
    ("fold", _HEAD_U + "def fold(word):\n    parts = unicodedata.normalize(\"NFD\", word)\n    return \"\".join([c for c in parts if not unicodedata.combining(c)]).lower()\n", "correct", None),
    ("fold", _HEAD_U + "def fold(word):\n    out = \"\"\n    for c in unicodedata.normalize(\"NFKD\", word.lower()):\n        if unicodedata.combining(c) == 0:\n            out = out + c\n    return out\n", "correct", None),
    ("fold", _F_LIST, "wrong", "list"),
    ("fold", _F_SPACES, "wrong", "spaces"),
    ("fold", _F_NO_LOWER, "wrong", "no-lower"),
    ("fold", _F_NO_NFD, "wrong", "no-nfd"),
    ("fold", _F_NFC, "wrong", "nfc"),
    ("fold", _F_MARKS, "wrong", "marks"),
    ("fold", _F.replace("    return \"\".join", "    \"\".join"), "wrong", "no-return"),
    ("fold", _HEAD_U + "def fold(word):\n    return \"cafe\"\n", "almost", None),
    ("truecase", _T_START, "empty", None),
    ("truecase", _T, "correct", None),
    ("truecase", "def truecase(words, names):\n    if words == [] or words[0] in names:\n        return words\n    return [words[0].lower()] + words[1:]\n", "correct", None),
    ("truecase", "def truecase(words, names):\n    out = list(words)\n    if out and out[0] not in names:\n        out[0] = out[0].lower()\n    return out\n", "correct", None),
    ("truecase", _T_ALL, "wrong", "all-small"),
    ("truecase", _T_NO_NAMES, "wrong", "no-names"),
    ("truecase", _T_NAMES_SMALL, "wrong", "names-small"),
    ("truecase", _T_FIRST_ONLY, "wrong", "first-only"),
    ("truecase", "def truecase(words, names):\n    return [\"het\", \"regent\", \"in\", \"Wenen\"]\n", "almost", None),
    ("drop-stop-words", _D_START, "empty", None),
    ("drop-stop-words", _D, "correct", None),
    ("drop-stop-words", "def drop_stop_words(words, stop_words):\n    kept = []\n    for w in words:\n        if not w.lower() in stop_words:\n            kept.append(w)\n    return kept\n", "correct", None),
    ("drop-stop-words", _D_SET, "wrong", "a-set"),
    ("drop-stop-words", _D_NO_LOWER, "wrong", "no-lower"),
    ("drop-stop-words", _D_ONLY, "wrong", "only-stop"),
    ("drop-stop-words", _D_LOWERED, "wrong", "lowered"),
    ("drop-stop-words", "def drop_stop_words(words, stop_words):\n    return [\"Tom\", \"thuis\"]\n", "almost", None),
    ("what-to-keep", {}, "empty", None),
    ("what-to-keep", {"ferrari": "fold", "creme": "fold", "naief": "fold", "fed": "capitals", "zuiden": "capitals",
                      "black": "capitals", "letitbe": "stop", "utrecht": "stop", "review": "stop"}, "correct", None),
    ("what-to-keep", {"ferrari": "fold", "fed": "capitals"}, "wrong", "unfinished"),
    ("what-to-keep", {"review": "fold"}, "wrong", "wrong:review"),
    ("what-to-keep", {"ferrari": "capitals", "creme": "fold", "naief": "fold", "fed": "capitals", "zuiden": "capitals",
                      "black": "capitals", "letitbe": "stop", "utrecht": "stop", "review": "stop"}, "wrong", "wrong:ferrari"),
    ("what-to-keep", {"ferrari": "fold", "creme": "fold", "naief": "fold", "fed": "capitals", "zuiden": "capitals",
                      "black": "stop", "letitbe": "stop", "utrecht": "stop", "review": "stop"}, "wrong", "wrong:black"),
]
