# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-language-change: count where a sentence switches language (switches), how often a word
occurs per 10,000 words of a corpus (per_10k), and which words follow a word (next_words), and sort statements under
borrowing, code-switching, semantic change or spelling.

The old Dutch is that of three novels of Project Gutenberg in data/gutenberg-books.json.gz (made by
data/tools/make_gutenberg.py): Camera Obscura (1839, the 29th printing of 1917), Max Havelaar (1860, an edition of 1881
or later) and Eline Vere (1889). Today's Dutch is that of the Dutch sentences of Tatoeba (CC BY 2.0 FR) in
data/tatoeba-nl.tsv.gz (made by data/tools/make_tatoeba_nl.py). The sentences with code-switching are quoted from
Poplack (1980), the title of the article, and from Peter Burger, "Het proefschrift van Ad Backus. Nederturks", Onze Taal
63 (1994), pp. 130-131; the language of each word is the course's own reading. Every number in the pictures is counted
by the course; the cases of the checks and the statements of the sort are the course's own."""
import contextlib
import copy
import html
import io
import re

from everything_ai import books, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, kept, with_time_limit

# --- the course's solutions ------------------------------------------------------------------------------------------


def switches(tags):
    """The course's solution of Exercise 1: the number of places where a word has another language tag than the word
    before it."""
    n = 0
    for i in range(1, len(tags)):
        if tags[i] != tags[i - 1]:
            n = n + 1
    return n


def per_10k(counts, word):
    """The course's solution of Exercise 2: how often `word` occurs for every 10,000 words of the corpus that `counts`
    counted; 0 for a word that is not in it."""
    total = 0
    for w in counts:
        total = total + counts[w]
    if word not in counts:
        return 0
    return counts[word] / total * 10000


def next_words(sentences, target):
    """The course's solution of Exercise 3: how often each word, in small letters, comes right after `target` in a
    sentence; `target` is compared in small letters."""
    counts = {}
    for sentence in sentences:
        for i in range(len(sentence) - 1):
            if sentence[i].lower() == target:
                w = sentence[i + 1].lower()
                if w in counts:
                    counts[w] = counts[w] + 1
                else:
                    counts[w] = 1
    return counts


def per_language(tags):
    """The worked example: how many words of a sentence each language has."""
    counts = {}
    for tag in tags:
        if tag in counts:
            counts[tag] = counts[tag] + 1
        else:
            counts[tag] = 1
    return counts


# --- the sentences with code-switching -------------------------------------------------------------------------------

# Poplack (1980), the title: a speaker's sentence that starts in English and ends in Spanish
POPLACK = ["Sometimes", "I'll", "start", "a", "sentence", "in", "Spanish", "y", "termino", "en", "español"]
POPLACK_TAGS = ["en", "en", "en", "en", "en", "en", "en", "es", "es", "es", "es"]
# Burger (1994), p. 130: two sentences of Turkish speakers in the Netherlands, from the research of Ad Backus, with the
# Dutch translations Onze Taal gives
BACKUS = ["Babama", "bir", "smoes", "verzinnen", "yaptik", "schoolfeestje", "var", "diye"]
BACKUS_TAGS = ["tr", "tr", "nl", "nl", "tr", "nl", "tr", "tr"]
BACKUS_TEXT = "Babama bir smoes verzinnen yaptik, schoolfeestje var diye."
BACKUS_NL = "Voor mijn vader hebben we een smoes verzonnen, we zeiden dat er een schoolfeestje was."
BACKUS_2 = ["Geldiğimde", "iki", "ay", "nieuwkomers'a", "gittim"]
BACKUS_2_TAGS = ["tr", "tr", "tr", "mixed", "tr"]
BACKUS_2_NL = "Twee maanden nadat ik aankwam, ging ik naar de nieuwkomers."

TAG_NAMES = {"en": {"tr": "Turkish", "nl": "Dutch", "en": "English", "es": "Spanish", "mixed": "mixed"},
             "nl": {"tr": "Turks", "nl": "Nederlands", "en": "Engels", "es": "Spaans", "mixed": "gemengd"}}

# --- the corpora: three novels of then, the sentences of Tatoeba of now -----------------------------------------------

NOVEL_KEYS = ("camera-obscura", "havelaar", "eline-vere")   # in the order of their first editions: 1839, 1860, 1889
_CUT = re.compile(r"[.!?]+")


def _read_novels() -> tuple:
    """The sentences of the three novels, each a tuple of its words: the text cut at every full stop, question mark and
    exclamation mark, then into words as books.words does."""
    out = []
    for key in NOVEL_KEYS:
        for part in _CUT.split(books.book(key)["text"]):
            words = books.words(part)
            if words:
                out.append(tuple(words))
    return tuple(out)


def _novels() -> tuple:
    return kept("nlp-language-change:novels", _read_novels)


def _tatoeba() -> tuple:
    return kept("nlp-language-change:tatoeba", lambda: tuple(tuple(books.words(s[1])) for s in books.dutch_sentences()))


def novels() -> list:
    """The sentences of the three novels as the exercises take them: a new list of lists of words."""
    return [list(s) for s in _novels()]


def tatoeba() -> list:
    """The Dutch sentences of Tatoeba as the exercises take them: a new list of lists of words."""
    return [list(s) for s in _tatoeba()]


def _count(sentences) -> dict:
    counts = {}
    for sentence in sentences:
        for word in sentence:
            w = word.lower()
            counts[w] = counts.get(w, 0) + 1
    return counts


def _novel_counts() -> dict:
    return kept("nlp-language-change:novel-counts", lambda: _count(_novels()))


def _tatoeba_counts() -> dict:
    return kept("nlp-language-change:tatoeba-counts", lambda: _count(_tatoeba()))


def novel_counts() -> dict:
    """How often each word occurs in the three novels, in small letters, as a new dict."""
    return dict(_novel_counts())


def tatoeba_counts() -> dict:
    """How often each word occurs in the Dutch sentences of Tatoeba, in small letters, as a new dict."""
    return dict(_tatoeba_counts())


def size(counts: dict) -> int:
    return sum(counts.values())


def text_count(key: str, form: str) -> int:
    """How often a form such as "to-day" stands in a book, as a whole word, in any capitals: books.words cuts a word at
    a hyphen, so the course counts these in the text."""
    return len(re.findall(r"(?<![\w-])" + re.escape(form) + r"(?![\w-])", books.book(key)["text"], flags=re.IGNORECASE))


# --- small helpers for the pictures ----------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums"
_MISS = "color:var(--ea-loss-text);font-weight:600"
_OLD = ("text-decoration:underline;text-decoration-color:var(--ea-ink-3);text-underline-offset:0.18em;"
        "text-decoration-thickness:1px")
_LINE = "font-family:var(--ea-font-ui);font-size:0.9rem;color:var(--ea-ink);line-height:1.6"
_TAG = "font-family:var(--ea-font-ui);font-size:0.72rem;color:var(--ea-ink-3)"
_RUN = ("display:inline-flex;flex-direction:column;align-items:center;border:1px solid var(--ea-line);"
        "border-radius:4px;padding:0.05rem 0.35rem;margin:0.15rem 0.1rem")
_KWIC = "font-family:var(--ea-font-code);font-size:0.78rem;color:var(--ea-ink);white-space:pre"


def _note(text: str) -> str:
    """A line under a picture; a word between asterisks is set in italics, as in the lesson's text."""
    parts = html.escape(text).split("*")
    marked = "".join(f"<i>{p}</i>" if k % 2 else p for k, p in enumerate(parts))
    return f'<div style="{_UI}">{marked}</div>'


def _num(n, lang: str) -> str:
    """A whole number with a separator for thousands: 2,319 in English, 2.319 in Dutch."""
    text = f"{round(n):,}"
    return text.replace(",", ".") if lang == "nl" else text


def _dec(x: float, lang: str, digits: int = 1) -> str:
    """A number with a fixed number of decimals, with a decimal comma in Dutch."""
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def _table(rows_: list, head: list, numbers: tuple = (), styles: dict | None = None) -> str:
    """A small table; rows hold HTML that is already escaped; numbers are the columns set to the right; styles maps
    (row, column) to extra style."""
    styles = styles or {}
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers else ""}{styles.get((i, j), "")}">'
                                    f'{c}</td>' for j, c in enumerate(row)) + "</tr>" for i, row in enumerate(rows_))
    return (f'<table style="border-collapse:collapse;margin:0.3rem 0 0.2rem"><thead><tr>{top}</tr></thead>'
            f"<tbody>{body}</tbody></table>")


def runs(words: list, tags: list) -> list:
    """The words of a sentence in runs of the same tag: a list of (tag, [words])."""
    out = []
    for word, tag in zip(words, tags):
        if out and out[-1][0] == tag:
            out[-1][1].append(word)
        else:
            out.append((tag, [word]))
    return out


def _runs_html(words: list, tags: list, lang: str) -> str:
    """A sentence with each run of one language in a frame, the language under it."""
    return "".join(f'<span style="{_RUN}"><span style="{_LINE}">{html.escape(" ".join(ws))}</span>'
                   f'<span style="{_TAG}">{html.escape(TAG_NAMES[lang][tag])}</span></span>' for tag, ws in runs(words, tags))


def _block(title: str, inner: str) -> str:
    return (f'<div><div style="{_HEAD};border:none;padding-left:0.45rem">{html.escape(title)}</div>{inner}</div>')


# --- the idea's picture: a sentence that switches, and a sentence in the old spelling ---------------------------------

# Eline Vere (1889), as the course's file prints it; the old forms with their spelling of today
OLD_SENTENCE = "De groote menschen bogen allen met een zelfbewust glimlachje voor den Spaanschen bisschop."
OLD_FORMS = {"groote": "grote", "menschen": "mensen", "den": "de", "Spaanschen": "Spaanse"}
_IDEA_WORDS = {
    "en": {"title": "Languages meet and change",
           "switch": "Turkish and Dutch in one sentence",
           "switch_note": "From the research of Ad Backus (Onze Taal, 1994). In English: we made up an excuse for my "
                          "father; we said there was a school party.",
           "old": "Eline Vere (1889), as the course's file prints it",
           "old_note": "Underlined: written another way today: {pairs}.",
           "alt": "Two sentences. A sentence of a Turkish speaker in the Netherlands, in runs of Turkish and Dutch. A "
                  "sentence of the novel Eline Vere in the old spelling, with four words that are written another way "
                  "today."},
    "nl": {"title": "Talen ontmoeten elkaar en veranderen",
           "switch": "Turks en Nederlands in één zin",
           "switch_note": "Uit het onderzoek van Ad Backus (Onze Taal, 1994). Vertaling uit Onze Taal: " + BACKUS_NL,
           "old": "Eline Vere (1889), zoals het bestand van de course het drukt",
           "old_note": "Onderstreept: nu anders geschreven: {pairs}.",
           "alt": "Twee zinnen. Een zin van een Turkse spreker in Nederland, in stukken Turks en Nederlands. Een zin "
                  "uit de roman Eline Vere in de oude spelling, met vier woorden die nu anders geschreven worden."},
}


def _old_html(text: str) -> str:
    out, at = [], 0
    for m in books._WORD.finditer(text):
        out.append(html.escape(text[at:m.start()]))
        word = html.escape(m.group())
        out.append(f'<span style="{_OLD}">{word}</span>' if m.group() in OLD_FORMS else word)
        at = m.end()
    out.append(html.escape(text[at:]))
    return "".join(out)


def idea_picture(lang: str) -> str:
    w = _IDEA_WORDS[lang]
    arrow = " → "
    pairs = ", ".join(f"{old}{arrow}{new}" for old, new in OLD_FORMS.items())
    body = (_block(w["switch"], f'<div style="margin:0.2rem 0">{_runs_html(BACKUS, BACKUS_TAGS, lang)}</div>'
                                + _note(w["switch_note"]))
            + '<div style="height:0.6rem"></div>'
            + _block(w["old"], f'<div style="{_LINE};margin:0.2rem 0.45rem">{_old_html(OLD_SENTENCE)}</div>'
                               + _note(w["old_note"].format(pairs=pairs))))
    return yours.figure(w["title"], body, alt=w["alt"] + " " + w["old_note"].format(pairs=pairs))


# --- running the learner's function ----------------------------------------------------------------------------------


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


def _a_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# --- the picture under exercise 1: sentences in runs, with your count of switches -------------------------------------

SWITCH_SENTENCES = [("Poplack (1980)", POPLACK, POPLACK_TAGS), ("Backus (Onze Taal, 1994)", BACKUS, BACKUS_TAGS),
                    ("Backus (Onze Taal, 1994)", BACKUS_2, BACKUS_2_TAGS)]
_SWITCH_WORDS = {
    "en": {"title": "Where your sentences switch",
           "count": "your switches: {got}; the course's: {want}",
           "note": "Each frame is a run of words in one language. The word *nieuwkomers'a* is Dutch with a Turkish "
                   "ending: mixed.",
           "stopped": "Your function stopped with an error on these sentences, so the picture stays empty.",
           "odd": "Your function does not give back a number, so the picture stays empty."},
    "nl": {"title": "Waar je zinnen van taal wisselen",
           "count": "jouw wissels: {got}; volgens de course: {want}",
           "note": "Elk kader is een reeks woorden in één taal. Het woord *nieuwkomers'a* is Nederlands met een Turkse "
                   "uitgang: gemengd.",
           "stopped": "Je functie stopte met een fout op deze zinnen, dus de afbeelding blijft leeg.",
           "odd": "Je functie geeft geen getal terug, dus de afbeelding blijft leeg."},
}


def switches_picture(seen) -> str:
    lang = seen.lang
    w = _SWITCH_WORDS[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    blocks, alts = [], []
    for name, words, tags in SWITCH_SENTENCES:
        got = _call(fn, tags)
        if got is _STOPPED:
            return yours.figure(w["title"], _note(w["stopped"]), alt=w["stopped"])
        if not _a_number(got):
            return yours.figure(w["title"], _note(w["odd"]), alt=w["odd"])
        want = switches(tags)
        count = html.escape(w["count"].format(got=got, want=want))
        style = _MISS if got != want else "color:var(--ea-ink-2)"
        blocks.append(f'<div style="margin:0.35rem 0"><div style="{_TAG}">{html.escape(name)}</div>'
                      f'<div>{_runs_html(words, tags, lang)}</div>'
                      f'<div style="font-family:var(--ea-font-ui);font-size:0.8rem;{style}">{count}</div></div>')
        alts.append(" ".join(words) + ": " + w["count"].format(got=got, want=want))
    return yours.figure(w["title"], "".join(blocks) + _note(w["note"]),
                        alt=w["title"] + ". " + "; ".join(alts) + ". " + w["note"].replace("*", ""))


# --- the picture under exercise 2: old and new spellings, per 10,000 words --------------------------------------------

SPELLING_PAIRS = [("zoo", "zo"), ("oogen", "ogen"), ("menschen", "mensen"), ("groote", "grote"),
                  ("tusschen", "tussen"), ("noodig", "nodig")]
_PER_WORDS = {
    "en": {"title": "Old and new spellings, per 10,000 words",
           "head": ["old", "novels", "Tatoeba", "new", "novels", "Tatoeba"],
           "note": "With your per_10k: the three novels, {a} words, and the Dutch sentences of Tatoeba, {b} words.",
           "stopped": "Your function stopped with an error on the counts, so the picture stays empty.",
           "odd": "Your function does not give back a number, so the picture stays empty."},
    "nl": {"title": "Oude en nieuwe spelling, per 10.000 woorden",
           "head": ["oud", "romans", "Tatoeba", "nieuw", "romans", "Tatoeba"],
           "note": "Met jouw per_10k: de drie romans, {a} woorden, en de Nederlandse zinnen van Tatoeba, {b} woorden.",
           "stopped": "Je functie stopte met een fout op de tellingen, dus de afbeelding blijft leeg.",
           "odd": "Je functie geeft geen getal terug, dus de afbeelding blijft leeg."},
}


def per_picture(seen) -> str:
    lang = seen.lang
    w = _PER_WORDS[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    old, new = _novel_counts(), _tatoeba_counts()
    rows_, alts = [], []
    for a, b in SPELLING_PAIRS:
        values = [_call(fn, old, a), _call(fn, new, a), _call(fn, old, b), _call(fn, new, b)]
        if any(v is _STOPPED for v in values):
            return yours.figure(w["title"], _note(w["stopped"]), alt=w["stopped"])
        if not all(_a_number(v) for v in values):
            return yours.figure(w["title"], _note(w["odd"]), alt=w["odd"])
        shown = [_dec(v, lang) for v in values]
        rows_.append([html.escape(a), shown[0], shown[1], html.escape(b), shown[2], shown[3]])
        alts.append(f"{a} {shown[0]} {shown[1]}, {b} {shown[2]} {shown[3]}")
    note = w["note"].format(a=_num(size(old), lang), b=_num(size(new), lang))
    alt = w["title"] + ". " + "; ".join(alts) + ". " + note
    return yours.figure(w["title"], _table(rows_, w["head"], numbers=(1, 2, 4, 5)) + _note(note), alt=alt)


# --- the picture under exercise 3: the words after schoon and erg, then and now ---------------------------------------

NEXT_TARGETS = ("schoon", "erg")
_NEXT_WORDS = {
    "en": {"title": "What comes after schoon and erg",
           "head": ["next word", "times"], "where": {"novels": "{t}, in the novels", "tatoeba": "{t}, in Tatoeba"},
           "note": "With your next_words, on the sentences of the three novels and of Tatoeba: the six words that "
                   "follow most often.",
           "stopped": "Your function stopped with an error on the sentences, so the picture stays empty.",
           "odd": "Your function does not give back a dict of counts, so the picture stays empty."},
    "nl": {"title": "Wat er na schoon en erg komt",
           "head": ["woord erna", "keer"], "where": {"novels": "{t}, in de romans", "tatoeba": "{t}, in Tatoeba"},
           "note": "Met jouw next_words, op de zinnen van de drie romans en van Tatoeba: de zes woorden die het vaakst "
                   "volgen.",
           "stopped": "Je functie stopte met een fout op de zinnen, dus de afbeelding blijft leeg.",
           "odd": "Je functie geeft geen dictionary met aantallen terug, dus de afbeelding blijft leeg."},
}


def top(counted: dict, k: int = 6) -> list:
    """The k words with the highest counts, the highest first, and words with the same count in alphabetical order."""
    good = [(word, n) for word, n in counted.items()
            if isinstance(word, str) and isinstance(n, int) and not isinstance(n, bool)]
    return sorted(good, key=lambda wn: (-wn[1], wn[0]))[:k]


def next_picture(seen) -> str:
    lang = seen.lang
    w = _NEXT_WORDS[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    blocks, alts = [], []
    for target in NEXT_TARGETS:
        for where, sents in (("novels", _novels()), ("tatoeba", _tatoeba())):
            got = _call(fn, sents, target)
            if got is _STOPPED:
                return yours.figure(w["title"], _note(w["stopped"]), alt=w["stopped"])
            if not isinstance(got, dict):
                return yours.figure(w["title"], _note(w["odd"]), alt=w["odd"])
            best = top(got)
            name = w["where"][where].format(t=target)
            table = _table([[html.escape(word), _num(n, lang)] for word, n in best], w["head"], numbers=(1,))
            blocks.append(_block(name, table))
            alts.append(name + ": " + ", ".join(f"{word} {_num(n, lang)}" for word, n in best))
    body = '<div style="display:flex;flex-wrap:wrap;gap:0.4rem 1.6rem">' + "".join(blocks) + "</div>"
    return yours.figure(w["title"], body + _note(w["note"]), alt=w["title"] + ". " + "; ".join(alts) + ". " + w["note"])


# --- the demo: a word of your choice, then and now --------------------------------------------------------------------

DEMO_LINES = 4
DEMO_WIDTH = 4
_DEMO_WORDS = {
    "en": {"title": "{word}: then and now",
           "rate": "{where}: {n} times, {r} per 10,000 words",
           "novels": "the three novels", "tatoeba": "the sentences of Tatoeba",
           "none": "Not in {where}.",
           "empty": "Type a word in the box.",
           "note": "A few of the places where the word stands, spread over each corpus, with {k} words on each side.",
           "alt": "How often the word occurs in the novels and in Tatoeba, and a few lines of a concordance of each."},
    "nl": {"title": "{word}: toen en nu",
           "rate": "{where}: {n} keer, {r} per 10.000 woorden",
           "novels": "de drie romans", "tatoeba": "de zinnen van Tatoeba",
           "none": "Niet in {where}.",
           "empty": "Typ een woord in het vak.",
           "note": "Een paar plaatsen waar het woord staat, verspreid over elk corpus, met {k} woorden aan elke kant.",
           "alt": "Hoe vaak het woord voorkomt in de romans en in Tatoeba, en een paar regels van een concordantie van "
                  "elk."},
}


def kwic_lines(sentences, target: str, k: int = DEMO_LINES, width: int = DEMO_WIDTH) -> list:
    """k lines (left, word, right) of a concordance of `target` in small letters, spread evenly over all places where it
    stands: the first place, the last, and places between them."""
    places = [(s, i) for s in sentences for i, word in enumerate(s) if word.lower() == target]
    if not places:
        return []
    if len(places) <= k:
        chosen = places
    else:
        chosen = [places[round(j * (len(places) - 1) / (k - 1))] for j in range(k)]
    return [(" ".join(s[max(0, i - width):i]), s[i], " ".join(s[i + 1:i + 1 + width])) for s, i in chosen]


def demo_word(text) -> str:
    """The word a learner typed, in small letters: its first word only."""
    found = books.words(str(text or ""))
    return found[0].lower() if found else ""


def demo_picture(text, lang: str) -> str:
    w = _DEMO_WORDS[lang]
    word = demo_word(text)
    if not word:
        return yours.figure(w["empty"], _note(w["empty"]), alt=w["empty"])
    body, alts = [], []
    for where, sents, counts in (("novels", _novels(), _novel_counts()), ("tatoeba", _tatoeba(), _tatoeba_counts())):
        name = w[where]
        n = counts.get(word, 0)
        if n == 0:
            line = w["none"].format(where=name)
            body.append(_block(line, ""))
            alts.append(line)
            continue
        rate = w["rate"].format(where=name, n=_num(n, lang), r=_dec(per_10k(counts, word), lang))
        lines = kwic_lines(sents, word)
        left_w = max(len(left) for left, _, _ in lines)
        rows = "".join(f'<div style="{_KWIC}">{html.escape(left.rjust(left_w))} '
                       f'<b>{html.escape(mid)}</b> {html.escape(right)}</div>' for left, mid, right in lines)
        body.append(_block(rate, f'<div style="overflow-x:auto;padding:0.15rem 0.45rem">{rows}</div>'))
        alts.append(rate)
    note = w["note"].format(k=DEMO_WIDTH)
    return yours.figure(w["title"].format(word=word), "".join(body) + _note(note),
                        alt=w["alt"] + " " + "; ".join(alts) + ".")


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["switches", "per-10k", "next-words", "borrow-switch-change"]

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
    """A short value the way the lessons write it: strings in double quotes, also inside a tuple, a list or a dict."""
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, tuple):
        return "(" + ", ".join(_value(p) for p in v) + ("," if len(v) == 1 else "") + ")"
    if isinstance(v, list):
        return "[" + ", ".join(_value(p) for p in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_value(k)}: {_value(x)}" for k, x in v.items()) + "}"
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".") if v != int(v) else f"{v:.1f}"
    return repr(v)


_MORE = {"en": "{shown}, and {k} more", "nl": "{shown}, en nog {k}"}


def _show(v):
    """{got} and {want}: a dict of counts from the highest count down, and a long dict cut short, with how many more it
    holds, in the language of the lesson."""
    if isinstance(v, dict) and len(v) > 6:
        best = top(v, 6)
        shown = "{" + ", ".join(f"{_value(k)}: {_value(n)}" for k, n in best) + ", ...}"
        return {lang: _MORE[lang].format(shown=shown, k=_num(len(v) - len(best), lang)) for lang in _MORE}
    if isinstance(v, dict) and all(isinstance(n, int) for n in v.values()) and len(top(v, len(v))) == len(v):
        return _value(dict(top(v, len(v))))
    return _value(v)


class Named:
    """A value of the lesson that a message shows by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


class _Lazy:
    """A value of the lesson made when a check first needs it, given as a new copy each time, and shown in a message
    by its name."""

    def __init__(self, name: str, make):
        self.name, self.make = name, make

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return self.make()


def _lit(v) -> Named:
    """A value of the check's own, shown in a message as Python, with strings in double quotes."""
    return Named(_value(v), v)


NOVEL_COUNTS_N = _Lazy("NOVEL_COUNTS", novel_counts)
TATOEBA_COUNTS_N = _Lazy("TATOEBA_COUNTS", tatoeba_counts)
NOVELS_N = _Lazy("NOVELS", novels)
TATOEBA_N = _Lazy("TATOEBA", tatoeba)


def _index_error(e) -> bool:
    return isinstance(e, IndexError)


def _key_error(e) -> bool:
    return isinstance(e, KeyError)


def _zero_division(e) -> bool:
    return isinstance(e, ZeroDivisionError)


def _a_list(got, text) -> bool:
    return isinstance(got, list)


# exercise 1: switches. Besides Backus's sentence: Poplack's title, which starts and ends in two languages, a sentence
# with a mixed word, one word, no words, a switch at every word, and one language only.
SWITCH_CASES = [
    (Named("BACKUS_TAGS", BACKUS_TAGS),),
    (Named("POPLACK_TAGS", POPLACK_TAGS),),
    (_lit(BACKUS_2_TAGS),),
    (_lit(["nl"]),),
    (_lit([]),),
    (_lit(["en", "es", "en", "es"]),),
    (_lit(["nl", "nl", "nl"]),),
]
_SW_START = "def switches(tags):\n    ...\n"
_SW = ("def switches(tags):\n"
       "    n = 0\n"
       "    for i in range(1, len(tags)):\n"
       "        if tags[i] != tags[i - 1]:\n"
       "            n = n + 1\n"
       "    return n\n")
_SW_WRAP = _SW.replace("range(1, len(tags))", "range(len(tags))")
_SW_LANGUAGES = "def switches(tags):\n    return len(set(tags))\n"
_SW_ONE = _SW.replace("            n = n + 1\n", "            n = 1\n")
_SW_FIRST = ("def switches(tags):\n"
             "    n = 0\n"
             "    for tag in tags:\n"
             "        if tag != tags[0]:\n"
             "            n = n + 1\n"
             "    return n\n")

# exercise 2: per_10k. Besides zoo in the novels: zoo and zo in Tatoeba, a word of the novels that Tatoeba lacks, a
# small dict of its own, a word that is not in it, and an empty dict.
SMALL = {"de": 3, "kat": 1}
PER_CASES = [
    (NOVEL_COUNTS_N, "zoo"),
    (TATOEBA_COUNTS_N, "zoo"),
    (TATOEBA_COUNTS_N, "zo"),
    (TATOEBA_COUNTS_N, "menschen"),
    (_lit(SMALL), "kat"),
    (_lit(SMALL), "hond"),
    (_lit({}), "kat"),
]
_P_START = "def per_10k(counts, word):\n    ...\n"
_P = ("def per_10k(counts, word):\n"
      "    total = 0\n"
      "    for w in counts:\n"
      "        total = total + counts[w]\n"
      "    if word not in counts:\n"
      "        return 0\n"
      "    return counts[word] / total * 10000\n")
_P_SHARE = _P.replace("    return counts[word] / total * 10000\n", "    return counts[word] / total\n")
_P_PERCENT = _P.replace("    return counts[word] / total * 10000\n", "    return counts[word] / total * 100\n")
_P_COUNT = _P.replace("    return counts[word] / total * 10000\n", "    return counts[word]\n")
_P_DIFFERENT = ("def per_10k(counts, word):\n"
                "    if word not in counts:\n"
                "        return 0\n"
                "    return counts[word] / len(counts) * 10000\n")

# exercise 3: next_words. Besides schoon in Tatoeba: schoon and erg in the novels, a word with a capital at the start
# and the word at the end of a sentence, a next word with a capital, a word that comes twice in one sentence, no
# sentences, and a sentence of one word.
NEXT_CASES = [
    (TATOEBA_N, "schoon"),
    (NOVELS_N, "schoon"),
    (NOVELS_N, "erg"),
    (_lit([["Schoon", "is", "het", "hier"], ["Het", "is", "schoon"]]), "schoon"),
    (_lit([["erg", "Goed"], ["Erg", "goed", "gedaan"]]), "erg"),
    (_lit([["de", "kat", "en", "de", "hond"]]), "de"),
    ([], "schoon"),
    (_lit([["schoon"]]), "schoon"),
]
_N_START = "def next_words(sentences, target):\n    ...\n"
_N = ("def next_words(sentences, target):\n"
      "    counts = {}\n"
      "    for sentence in sentences:\n"
      "        for i in range(len(sentence) - 1):\n"
      "            if sentence[i].lower() == target:\n"
      "                w = sentence[i + 1].lower()\n"
      "                if w in counts:\n"
      "                    counts[w] = counts[w] + 1\n"
      "                else:\n"
      "                    counts[w] = 1\n"
      "    return counts\n")
_N_CAPITALS = _N.replace("if sentence[i].lower() == target:", "if sentence[i] == target:")
_N_NEXT_CAPITALS = _N.replace("w = sentence[i + 1].lower()", "w = sentence[i + 1]")
_N_ONES = _N.replace("                    counts[w] = counts[w] + 1\n", "                    counts[w] = 1\n")
_N_INSIDE = _N.replace("    counts = {}\n    for sentence in sentences:\n",
                       "    for sentence in sentences:\n        counts = {}\n")
_N_BEFORE = ("def next_words(sentences, target):\n"
             "    counts = {}\n"
             "    for sentence in sentences:\n"
             "        for i in range(1, len(sentence)):\n"
             "            if sentence[i].lower() == target:\n"
             "                w = sentence[i - 1].lower()\n"
             "                if w in counts:\n"
             "                    counts[w] = counts[w] + 1\n"
             "                else:\n"
             "                    counts[w] = 1\n"
             "    return counts\n")


def _same_counts(a, b) -> bool:
    """Two dicts of counts are the same when they hold the same words with the same counts, in any order."""
    return isinstance(a, dict) and isinstance(b, dict) and dict(a) == dict(b)


EXERCISES = {
    "switches": FunctionBox(
        name="switches",
        picture=switches_picture,
        function="switches",
        start=_SW_START,
        solution=_SW,
        visible=SWITCH_CASES[:1],
        hidden=SWITCH_CASES[1:],
        first=[_DOTS],
        show=_show,
        mistakes=[
            Mistake("index", error=_index_error,
                    en="With `{call}`, your function asks for a place after the last tag of the list. Compare each tag with the one before it, starting at the second tag, so that every tag you ask for exists.",
                    nl="Met `{call}` vraagt je functie om een plaats na de laatste tag van de lijst. Vergelijk elke tag met de tag ervoor, vanaf de tweede tag, zodat elke tag waar je om vraagt bestaat."),
            Mistake("wrap", wrong=_SW_WRAP,
                    en="`{call}` gives {got}, and it should give {want}: your loop starts at place 0, and there it compares the first tag with `tags[-1]`, the last tag of the list. The first word has no word before it: start the loop at place 1.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je loop begint op plaats 0, en daar vergelijkt je functie de eerste tag met `tags[-1]`, de laatste tag van de lijst. Het eerste woord heeft geen woord ervoor: begin de loop op plaats 1."),
            Mistake("languages", wrong=_SW_LANGUAGES,
                    en="`{call}` gives {got}, and it should give {want}: the number of languages in the sentence, not the number of switches. A sentence can go back and forth between two languages many times. Count the places where a tag differs from the tag before it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het aantal talen in de zin, niet het aantal wissels. Een zin kan vaak heen en weer gaan tussen twee talen. Tel de plaatsen waar een tag verschilt van de tag ervoor."),
            Mistake("first", wrong=_SW_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: you count the words in another language than the first word. A run of three Dutch words is one switch, not three. Compare each tag with the tag right before it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt de woorden in een andere taal dan het eerste woord. Een reeks van drie Nederlandse woorden is één wissel, geen drie. Vergelijk elke tag met de tag er direct voor."),
            Mistake("one", wrong=_SW_ONE,
                    en="`{call}` gives {got}, and it should give {want}: your count stays at 1, however many switches there are. What has to happen to `n` at each switch?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je aantal blijft 1, hoeveel wissels er ook zijn. Wat moet er bij elke wissel met `n` gebeuren?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: go along the tags, and count the places where a tag differs from the tag before it.",
                      "nl": "Vervang de `...`: ga de tags langs, en tel de plaatsen waar een tag verschilt van de tag ervoor."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function compare each tag with the tag before it, and count every place where the tags differ?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Vergelijkt je functie elke tag met de tag ervoor, en telt de functie elke plaats waar de tags verschillen?"},
            "almost": {"en": "Right for BACKUS_TAGS. But `{call}` gives {got}, and it should give {want}. Does your function work for any list of tags, also one that starts and ends in different languages, one word, or no words at all?",
                       "nl": "Goed voor BACKUS_TAGS. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke lijst tags, ook een die in verschillende talen begint en eindigt, één woord, of helemaal geen woorden?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back: it has a `return`, but Python does not get to it. Is the `return` the last line, after the loop?",
                          "nl": "Met `{call}` geeft je functie niets terug: er staat een `return` in, maar Python komt er niet aan toe. Is de `return` de laatste regel, na de loop?"},
            "correct": {"en": "Your `switches` is right in all {n} cases. Look at the picture: three sentences in runs of one language, with your count of switches.",
                        "nl": "Je `switches` klopt in alle {n} gevallen. Kijk naar de afbeelding: drie zinnen in reeksen van één taal, met jouw aantal wissels."},
        },
    ),
    "per-10k": FunctionBox(
        name="per-10k",
        picture=per_picture,
        function="per_10k",
        start=_P_START,
        solution=_P,
        visible=PER_CASES[:1],
        hidden=PER_CASES[1:],
        first=[_DOTS],
        show=_show,
        mistakes=[
            Mistake("key", error=_key_error,
                    en="With `{call}`, your function asks `counts` for a word that is not in it. A word that the corpus does not have occurs 0 times: give back 0 for it, before you look it up.",
                    nl="Met `{call}` vraagt je functie `counts` om een woord dat er niet in staat. Een woord dat het corpus niet heeft, komt 0 keer voor: geef er 0 voor terug, voordat je het opzoekt."),
            Mistake("zero", error=_zero_division,
                    en="With `{call}`, your function divides by 0: the corpus has no words at all. Give back 0 for a word that is not in `counts`, before you divide.",
                    nl="Met `{call}` deelt je functie door 0: het corpus heeft helemaal geen woorden. Geef 0 terug voor een woord dat niet in `counts` staat, voordat je deelt."),
            Mistake("count", wrong=_P_COUNT,
                    en="`{call}` gives {got}, and it should give {want}: the bare count of the word. A count says little when the corpora differ in size: divide it by the number of all words of the corpus, and multiply by 10,000.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het kale aantal van het woord. Een aantal zegt weinig als de corpora verschillen in grootte: deel het door het aantal van alle woorden van het corpus, en vermenigvuldig met 10.000."),
            Mistake("share", wrong=_P_SHARE,
                    en="`{call}` gives {got}, and it should give {want}: the share of the word in all words, a number between 0 and 1. Per 10,000 words, multiply that share by 10,000.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het aandeel van het woord in alle woorden, een getal tussen 0 en 1. Per 10.000 woorden vermenigvuldig je dat aandeel met 10.000."),
            Mistake("percent", wrong=_P_PERCENT,
                    en="`{call}` gives {got}, and it should give {want}: a percentage, the count per 100 words. The exercise asks for the count per 10,000 words.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een percentage, het aantal per 100 woorden. De oefening vraagt om het aantal per 10.000 woorden."),
            Mistake("different", wrong=_P_DIFFERENT,
                    en="`{call}` gives {got}, and it should give {want}: you divide by `len(counts)`, the number of different words. The size of a corpus is the number of all its words, every time a word occurs: the sum of all counts.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je deelt door `len(counts)`, het aantal verschillende woorden. De grootte van een corpus is het aantal van alle woorden, elke keer dat een woord voorkomt: de som van alle aantallen."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: add up all counts to get the size of the corpus, give back 0 for a word that is not in `counts`, and otherwise its count per 10,000 words.",
                      "nl": "Vervang de `...`: tel alle aantallen op tot de grootte van het corpus, geef 0 terug voor een woord dat niet in `counts` staat, en anders het aantal per 10.000 woorden."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function divide the count of the word by the sum of all counts, and multiply by 10,000?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Deelt je functie het aantal van het woord door de som van alle aantallen, en vermenigvuldigt de functie de uitkomst met 10.000?"},
            "almost": {"en": "Right for zoo in NOVEL_COUNTS. But `{call}` gives {got}, and it should give {want}. Does your function work for any counts and any word, also a word that is not in `counts`, or no counts at all?",
                       "nl": "Goed voor zoo in NOVEL_COUNTS. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor alle tellingen en elk woord, ook een woord dat niet in `counts` staat, of helemaal geen tellingen?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back. For a word that is not in `counts`, give back 0; and does every way through the function end with a `return`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Geef 0 terug voor een woord dat niet in `counts` staat; en eindigt elke weg door de functie met een `return`?"},
            "correct": {"en": "Your `per_10k` is right in all {n} cases. Look at the picture: six old spellings and their new ones, in the novels and in Tatoeba.",
                        "nl": "Je `per_10k` klopt in alle {n} gevallen. Kijk naar de afbeelding: zes oude spellingen en hun nieuwe, in de romans en in Tatoeba."},
        },
    ),
    "next-words": FunctionBox(
        name="next-words",
        picture=next_picture,
        function="next_words",
        start=_N_START,
        solution=_N,
        visible=NEXT_CASES[:1],
        hidden=NEXT_CASES[1:],
        first=[_DOTS],
        show=_show,
        compare=_same_counts,
        mistakes=[
            Mistake("index", error=_index_error,
                    en="With `{call}`, your function asks for a word after the last word of a sentence, where `target` stands at the end, as in *De keuken was schoon*. The last word has no next word: let your loop stop one place earlier.",
                    nl="Met `{call}` vraagt je functie om een woord na het laatste woord van een zin, waar `target` aan het eind staat, zoals in *De keuken was schoon*. Het laatste woord heeft geen volgend woord: laat je loop één plaats eerder stoppen."),
            Mistake("a-list", matches=_a_list,
                    en="`{call}` gives a list, and it should give a dict from each next word to how often it follows: {want}.",
                    nl="`{call}` geeft een lijst, en dat moet een dictionary zijn van elk volgend woord naar hoe vaak het volgt: {want}."),
            Mistake("capitals", wrong=_N_CAPITALS,
                    en="`{call}` gives {got}, and it should give {want}: your function misses `target` where it starts with a capital, as *Schoon* at the start of a sentence. Compare the word in small letters with `target`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie mist `target` als het met een hoofdletter begint, zoals *Schoon* aan het begin van een zin. Vergelijk het woord in kleine letters met `target`."),
            Mistake("next-capitals", wrong=_N_NEXT_CAPITALS,
                    en="`{call}` gives {got}, and it should give {want}: your counts keep *Goed* and *goed* apart. Count each next word in small letters.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telling houdt *Goed* en *goed* uit elkaar. Tel elk volgend woord in kleine letters."),
            Mistake("before", wrong=_N_BEFORE,
                    en="`{call}` gives {got}, and it should give {want}: you count the word before `target`, not the word after it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt het woord vóór `target`, niet het woord erna."),
            Mistake("ones", wrong=_N_ONES,
                    en="`{call}` gives {got}, and it should give {want}: every count stays at 1. What has to happen to the count of a next word you meet again?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elk aantal blijft 1. Wat moet er gebeuren met het aantal van een volgend woord dat je opnieuw tegenkomt?"),
            Mistake("inside", wrong=_N_INSIDE,
                    en="`{call}` gives {got}, and it should give {want}: your counts start again at every sentence, so only the last sentence is left. Make the dict once, before the loops.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telling begint bij elke zin opnieuw, dus alleen de laatste zin blijft over. Maak de dictionary één keer, vóór de loops."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: go through the places of each sentence, and when the word there is `target` in small letters, count the word right after it, in small letters, in a dict.",
                      "nl": "Vervang de `...`: ga de plaatsen van elke zin langs, en als het woord daar in kleine letters `target` is, tel dan het woord er direct na, in kleine letters, in een dictionary."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function count the word right after every place where `target` stands, in small letters?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Telt je functie het woord direct na elke plaats waar `target` staat, in kleine letters?"},
            "almost": {"en": "Right for schoon in TATOEBA. But `{call}` gives {got}, and it should give {want}. Does your function work for any sentences and any word, also with capitals, the word twice in one sentence, or no sentences at all?",
                       "nl": "Goed voor schoon in TATOEBA. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor alle zinnen en elk woord, ook met hoofdletters, het woord twee keer in één zin, of helemaal geen zinnen?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back: it has a `return`, but Python does not get to it. Is the `return` the last line, after both loops?",
                          "nl": "Met `{call}` geeft je functie niets terug: er staat een `return` in, maar Python komt er niet aan toe. Is de `return` de laatste regel, na beide loops?"},
            "correct": {"en": "Your `next_words` is right in all {n} cases. Look at the picture: the words that follow *schoon* and *erg* in the novels and in Tatoeba.",
                        "nl": "Je `next_words` klopt in alle {n} gevallen. Kijk naar de afbeelding: de woorden die volgen op *schoon* en *erg* in de romans en in Tatoeba."},
        },
    ),
    "borrow-switch-change": Sort(
        name="borrow-switch-change",
        groups={
            "borrowing": {"en": "Borrowing: a language takes over a word from another language",
                          "nl": "Lenen: een taal neemt een woord over uit een andere taal"},
            "switching": {"en": "Code-switching: one speaker uses two languages in one conversation or sentence",
                          "nl": "Code-switching: één spreker gebruikt twee talen in één gesprek of zin"},
            "meaning": {"en": "Semantic change: a word comes to stand for something else",
                        "nl": "Betekenisverandering: een woord gaat voor iets anders staan"},
            "spelling": {"en": "Spelling: the same word is written another way",
                         "nl": "Spelling: hetzelfde woord wordt anders geschreven"},
        },
        items={
            "cookie": {"en": "**English *cookie* comes from Dutch *koekje*.**",
                       "nl": "**Het Engelse *cookie* komt van het Nederlandse *koekje*.**"},
            "tram": {"en": "**Dutch took the word *tram* from English in the 19th century.**",
                     "nl": "**Het Nederlands nam het woord *tram* in de 19e eeuw over uit het Engels.**"},
            "poplack": {"en": "**A speaker in a study of Shana Poplack said: *Sometimes I'll start a sentence in Spanish y termino en español*.**",
                        "nl": "**Iemand in een onderzoek van Shana Poplack zei: *Sometimes I'll start a sentence in Spanish y termino en español*.**"},
            "backus": {"en": "**Ad Backus studied sentences of Turkish speakers in the Netherlands such as *Babama bir smoes verzinnen yaptik, schoolfeestje var diye*.**",
                       "nl": "**Ad Backus onderzocht zinnen van Turkse sprekers in Nederland zoals *Babama bir smoes verzinnen yaptik, schoolfeestje var diye*.**"},
            "slim": {"en": "**Dutch *slim* first meant crooked, then bad; from 1602 on it is found meaning clever.**",
                     "nl": "**Het Nederlandse *slim* betekende eerst scheef, toen slecht; vanaf 1602 komt het voor in de betekenis schrander.**"},
            "deer": {"en": "**The Old English word that became *deer* could refer to many kinds of animals.**",
                     "nl": "**Het Oudengelse woord waaruit *deer* (hert) ontstond, kon veel soorten dieren aanduiden.**"},
            "menschen": {"en": "**The Dutch novels of the course write *menschen*, the sentences of Tatoeba *mensen*.**",
                         "nl": "**De Nederlandse romans van de course schrijven *menschen*, de zinnen van Tatoeba *mensen*.**"},
            "today": {"en": "**The English novels of the course have *to-day* 33 times and *today* 4 times.**",
                      "nl": "**In de Engelse romans van de course staat 33 keer *to-day* en 4 keer *today*.**"},
        },
        solution={"cookie": "borrowing", "tram": "borrowing", "poplack": "switching", "backus": "switching",
                  "slim": "meaning", "deer": "meaning", "menschen": "spelling", "today": "spelling"},
        nudge={
            "cookie": {"en": "From which language did the word come, and which language has it now?",
                       "nl": "Uit welke taal kwam het woord, en welke taal heeft het nu?"},
            "tram": {"en": "Did a speaker go back and forth here, or did a word move into a language for good?",
                     "nl": "Ging hier een spreker heen en weer, of verhuisde een woord voorgoed naar een andere taal?"},
            "poplack": {"en": "How many languages are in this one sentence?",
                        "nl": "Hoeveel talen staan er in deze ene zin?"},
            "backus": {"en": "Does this sentence stay in one language from start to end?",
                       "nl": "Blijft deze zin van begin tot eind in één taal?"},
            "slim": {"en": "Is it the letters of the word that differ, or what it stands for?",
                     "nl": "Zijn het de letters van het woord die verschillen, of waar het voor staat?"},
            "deer": {"en": "Is the point here how the word is written, or how many things it names?",
                     "nl": "Gaat het hier om hoe het woord geschreven wordt, of om hoeveel dingen het aanduidt?"},
            "menschen": {"en": "Does the word stand for something else today, or does it only look different?",
                         "nl": "Staat het woord nu voor iets anders, of ziet het er alleen anders uit?"},
            "today": {"en": "Is this a new word, or the old one with one mark less?",
                      "nl": "Is dit een nieuw woord, of het oude met één teken minder?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the statements again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de uitspraken opnieuw uit."},
            "empty": {"en": "Put each bold statement under what it shows.",
                      "nl": "Zet elke vetgedrukte uitspraak bij wat de uitspraak laat zien."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} goes somewhere else.",
                      "nl": "{item} hoort ergens anders."},
            "correct": {"en": "All {n} are right. The kinds can meet in one form: the *zoo* of the novels is the old spelling of *zo*, and the *zoo* of today's sentences is another word, for a place with animals, as in English.",
                        "nl": "Alle {n} kloppen. De soorten kunnen samenkomen in één vorm: de *zoo* van de romans is de oude spelling van *zo*, en de *zoo* van de zinnen van nu is een ander woord, voor een plek met dieren, zoals in het Engels."},
        },
    ),
}

CASES = [
    ("switches", _SW_START, "empty", None),
    ("switches", _SW, "correct", None),
    ("switches", "def switches(tags):\n    n = 0\n    for a, b in zip(tags, tags[1:]):\n        if a != b:\n            n += 1\n    return n\n", "correct", None),
    ("switches", _SW_WRAP, "wrong", "wrap"),
    ("switches", _SW_LANGUAGES, "wrong", "languages"),
    ("switches", _SW_FIRST, "wrong", "first"),
    ("switches", _SW_ONE, "wrong", "one"),
    ("switches", "def switches(tags):\n    n = 0\n    for i in range(len(tags)):\n        if tags[i] != tags[i + 1]:\n            n = n + 1\n    return n\n", "error", "index"),
    ("switches", _SW.replace("    return n\n", ""), "wrong", "no-return"),
    ("per-10k", _P_START, "empty", None),
    ("per-10k", _P, "correct", None),
    ("per-10k", "def per_10k(counts, word):\n    total = sum(counts.values())\n    return 10000 * counts.get(word, 0) / total if total else 0\n", "correct", None),
    ("per-10k", _P_COUNT, "wrong", "count"),
    ("per-10k", _P_SHARE, "wrong", "share"),
    ("per-10k", _P_PERCENT, "wrong", "percent"),
    ("per-10k", _P_DIFFERENT, "wrong", "different"),
    ("per-10k", _P.replace("    if word not in counts:\n        return 0\n", ""), "error", "key"),
    ("per-10k", "def per_10k(counts, word):\n    total = 0\n    for w in counts:\n        total = total + counts[w]\n    return counts.get(word, 0) / total * 10000\n", "error", "zero"),
    ("per-10k", "def per_10k(counts, word):\n    return 2587 / 469251 * 10000\n", "almost", None),
    ("next-words", _N_START, "empty", None),
    ("next-words", _N, "correct", None),
    ("next-words", "def next_words(sentences, target):\n    counts = {}\n    for s in sentences:\n        for a, b in zip(s, s[1:]):\n            if a.lower() == target:\n                counts[b.lower()] = counts.get(b.lower(), 0) + 1\n    return counts\n", "correct", None),
    ("next-words", _N_CAPITALS, "wrong", "capitals"),
    ("next-words", _N_NEXT_CAPITALS, "wrong", "next-capitals"),
    ("next-words", _N_BEFORE, "wrong", "before"),
    ("next-words", _N_ONES, "wrong", "ones"),
    ("next-words", _N_INSIDE, "wrong", "inside"),
    ("next-words", _N.replace("range(len(sentence) - 1)", "range(len(sentence))"), "error", "index"),
    ("next-words", "def next_words(sentences, target):\n    found = []\n    for s in sentences:\n        for i in range(len(s) - 1):\n            if s[i].lower() == target:\n                found.append(s[i + 1].lower())\n    return found\n", "wrong", "a-list"),
    ("next-words", _N.replace("    return counts\n", ""), "wrong", "no-return"),
    ("borrow-switch-change", {}, "empty", None),
    ("borrow-switch-change", {"cookie": "borrowing", "tram": "borrowing", "poplack": "switching", "backus": "switching",
                              "slim": "meaning", "deer": "meaning", "menschen": "spelling", "today": "spelling"},
     "correct", None),
    ("borrow-switch-change", {"cookie": "borrowing", "poplack": "switching"}, "wrong", "unfinished"),
    ("borrow-switch-change", {"backus": "borrowing"}, "wrong", "wrong:backus"),
    ("borrow-switch-change", {"cookie": "borrowing", "tram": "borrowing", "poplack": "switching", "backus": "switching",
                              "slim": "meaning", "deer": "spelling", "menschen": "spelling", "today": "spelling"},
     "wrong", "wrong:deer"),
    ("borrow-switch-change", {"cookie": "borrowing", "tram": "switching", "poplack": "switching", "backus": "switching",
                              "slim": "meaning", "deer": "meaning", "menschen": "spelling", "today": "spelling"},
     "wrong", "wrong:tram"),
]
