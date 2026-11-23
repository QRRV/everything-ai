# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-bio-tags: turn spans into BIO tags (to_tags), turn BIO tags back into spans (to_spans), and
say for names next to each other whether IO tags keep them apart (a sort); with the pictures of what each function
gives on the sentences of Universal NER, two pictures of the idea and a demo.

The sentences are those of UNER_English-EWT of Universal NER v1 (Mayhew and others, NAACL 2024; CC-BY-SA-4.0), without
the emails (data/uner-en-ewt.tsv.gz, made by data/tools/make_uner.py, read with everything_ai.uner), so nothing is
downloaded. Every number in the text and the pictures is computed by the course (tests/test_nlp_bio_tags.py computes
them again)."""
import contextlib
import html
import io
from collections import Counter

from everything_ai import uner, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, with_time_limit

_KEY = "nlp-bio-tags"

# --- the course's data -----------------------------------------------------------------------------------------------

UNER = tuple(s.pairs for s in uner.sentences())   # 11,722 sentences: tuples of (word, tag) pairs
_SHOWN = tuple(s.pairs for s in uner.sentences() if s.shown)
EXAMPLE = uner.sentence("answers-20090717131757AAVh6rG_ans-0007").pairs   # Miramar California is a bit north of San Diego .
YORK = uner.sentence("weblog-blogspot.com_alaindewitt_20040929103700_ENG_20040929_103700-0078").pairs
HEADLINE = uner.sentence("weblog-blogspot.com_dakbangla_20041119231111_ENG_20041119_231111-0005").pairs
EXAMPLE_SPANS = [(0, 1, "LOC"), (1, 2, "LOC"), (7, 9, "LOC")]
# the course's own tags for the example, as a tagger could give them: Miramar California as one place, and an I- after
# an O on Diego
TAGGER = ["B-LOC", "I-LOC", "O", "O", "O", "O", "O", "O", "I-LOC", "O"]
KINDS = uner.KINDS


def words_of(sentence) -> list:
    return [w for w, _ in sentence]


def tags_of(sentence) -> list:
    return [t for _, t in sentence]


def io_tags(n: int, spans: list) -> list:
    """The IO tags of n words with these spans, as in the worked example: I- on every word of a span."""
    tags = ["O"] * n
    for start, end, kind in spans:
        for i in range(start, end):
            tags[i] = "I-" + kind
    return tags


def counts() -> dict:
    """The numbers of the lesson: words, words outside every name, names, names of more than one word, per type, and
    the times a B- comes right after a span of the same type."""
    words = sum(len(s) for s in UNER)
    outside = sum(1 for s in UNER for _, t in s if t == "O")
    spans = [x for s in UNER for x in uner.spans(tags_of(s))]
    touching = 0
    for s in UNER:
        found = uner.spans(tags_of(s))
        touching += sum(1 for a, b in zip(found, found[1:]) if a[1] == b[0] and a[2] == b[2])
    return {"sentences": len(UNER), "words": words, "outside": outside, "inside": words - outside, "spans": len(spans),
            "long": sum(1 for x in spans if x[1] - x[0] > 1), "kinds": Counter(x[2] for x in spans),
            "with_names": sum(1 for s in UNER if any(t != "O" for _, t in s)), "touching": touching,
            "longest": max(x[1] - x[0] for x in spans)}


# --- pictures: words, tags and spans ---------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_HEAD_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);font-weight:600;margin-top:0.6rem"
_CODE = "font-family:var(--ea-font-code);font-size:0.78rem"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:1px"
_PART = "display:inline-block;margin:0 0.15rem 0.25rem 0;text-align:center;vertical-align:top"
_WORD = ("display:block;border-radius:5px;padding:0.15rem 0.4rem;font-family:var(--ea-font-code);font-size:0.9rem;"
         "color:var(--ea-ink);border:1px solid var(--ea-line)")
_TAG = "display:block;font-family:var(--ea-font-code);font-size:0.7rem;margin-top:0.15rem;padding:0 0.1rem"
_GROUP = "display:inline-block;padding-bottom:0.15rem;margin-right:0.25rem;vertical-align:top;text-align:center"
_GROUP_LABEL = "display:block;font-family:var(--ea-font-code);font-size:0.7rem;margin-top:0.15rem"
_ROLE = {   # the line under a span and the color of a tag: the data's, the learner's, or wrong
    "data": ("var(--ea-ink-3)", "var(--ea-ink-2)"),
    "mine": ("var(--ea-pred)", "var(--ea-pred-text)"),
    "off": ("var(--ea-loss)", "var(--ea-loss-text)"),
}


def _marked(text: str) -> str:
    """Escaped text in which `name` is set as code and *word* in italics."""
    out = []
    for k, part in enumerate(html.escape(text).split("`")):
        if k % 2:
            out.append(f"<code>{part}</code>")
        else:
            out.append("".join(f"<i>{p}</i>" if j % 2 else p for j, p in enumerate(part.split("*"))))
    return "".join(out)


def _note(text: str) -> str:
    return f'<div style="{_UI}">{_marked(text)}</div>'


def _head(text: str) -> str:
    return f'<div style="{_HEAD_UI}">{_marked(text)}</div>'


def _bare(text: str) -> str:
    return text.replace("*", "").replace("`", "")


def _n(x: int, lang: str) -> str:
    """A whole number with a separator for thousands: 11,722 in English, 11.722 in Dutch."""
    text = f"{x:,}"
    return text.replace(",", ".") if lang == "nl" else text


def _pct(x: float, lang: str, digits: int = 1) -> str:
    text = f"{100 * x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def _repr(v) -> str:
    """How Python shows a value, also for a value whose repr stops with an error."""
    try:
        return repr(v)
    except Exception:   # noqa: BLE001   a learner's value can be anything
        return "<" + type(v).__name__ + ">"


def _shown_tag(tag) -> str:
    """A tag of the learner's as text, short: anything that is not a string is shown as Python shows it."""
    text = tag if isinstance(tag, str) else _repr(tag)
    return text if len(text) <= 12 else text[:11] + "…"


def tagged_words(words, tags, roles=None) -> str:
    """Words in boxes, each with its tag under it; roles gives per word "data", "mine" or "off"."""
    parts = []
    for i, word in enumerate(words):
        tag = tags[i] if i < len(tags) else ""
        role = roles[i] if roles else "data"
        color = _ROLE[role][1]
        extra = f";{_OFF}" if role == "off" else ""
        parts.append(f'<span style="{_PART}"><span style="{_WORD}">{html.escape(word)}</span>'
                     f'<span style="{_TAG};color:{color}{extra}">{html.escape(_shown_tag(tag)) or "&nbsp;"}</span></span>')
    return "".join(parts)


def spanned_words(words, spans, roles=None) -> str:
    """Words in boxes, with a line and a type under each span; roles gives per span "data", "mine" or "off". Spans that
    overlap or lie outside the words are left out of the drawing (the note under a picture names them)."""
    marks, taken = {}, set()
    for k, (start, end, kind) in enumerate(spans):
        if not (0 <= start < end <= len(words)) or taken & set(range(start, end)):
            continue
        taken |= set(range(start, end))
        marks[start] = (end, kind, roles[k] if roles else "data")
    parts, i = [], 0
    while i < len(words):
        if i in marks:
            end, kind, role = marks[i]
            line, color = _ROLE[role]
            style = "dashed" if role == "off" else "solid"
            inner = "".join(f'<span style="{_PART}"><span style="{_WORD}">{html.escape(w)}</span></span>'
                            for w in words[i:end])
            parts.append(f'<span style="{_GROUP};border-bottom:2px {style} {line}"><span style="display:block">'
                         f'{inner}</span><span style="{_GROUP_LABEL};color:{color}">{html.escape(str(kind))}</span>'
                         f'</span>')
            i = end
        else:
            parts.append(f'<span style="{_PART}"><span style="{_WORD}">{html.escape(words[i])}</span></span>')
            i += 1
    return "".join(parts)


def _spans_code(spans, roles=None) -> str:
    """Spans as Python writes them, each in the color of its role."""
    out = []
    for k, x in enumerate(spans):
        role = roles[k] if roles else "data"
        color = _ROLE[role][1]
        extra = f";{_OFF}" if role == "off" else ""
        out.append(f'<span style="color:{color}{extra}">{html.escape(repr(x))}</span>')
    return f'<div style="{_CODE};margin-top:0.1rem">[{", ".join(out)}]</div>'


_SPANS_WORDS = {
    "en": {"title": "Names as spans", "example": "A question with its answer on the web",
           "york": "A blog",
           "legend": "A line under one or more words marks a name, a span, with its type under it: PER a person, LOC "
                     "a place, ORG an organization. Under each sentence, its spans as `(start, end, kind)`: the "
                     "position of the first word, and of the word just after the last, counted from 0."},
    "nl": {"title": "Namen als spans", "example": "Een vraag met antwoord op het web", "york": "Een blog",
           "legend": "Een lijn onder een of meer woorden markeert een naam, een span, met het type eronder: PER een "
                     "persoon, LOC een plaats, ORG een organisatie. Onder elke zin de spans als `(start, end, kind)`: "
                     "de positie van het eerste woord, en de positie van het woord net na het laatste, geteld vanaf 0."},
}


def spans_picture(lang: str) -> str:
    """The idea: two sentences of Universal NER with their names drawn as spans, and the spans as tuples."""
    w = _SPANS_WORDS[lang]
    rows, said = [], []
    for key, sentence in (("example", EXAMPLE), ("york", YORK)):
        found = uner.spans(tags_of(sentence))
        rows.append(_head(w[key]) + f'<div>{spanned_words(words_of(sentence), found)}</div>' + _spans_code(found))
        said.append(f'{w[key]}: {" ".join(words_of(sentence))} {found}')
    return yours.figure(w["title"], "".join(rows) + _note(w["legend"]),
                        alt=f'{w["title"]}. {"; ".join(said)}. {_bare(w["legend"])}')


_BIO_WORDS = {
    "en": {"title": "BIO tags and IO tags of the same sentence",
           "bio": "BIO: B- on the first word of a span, I- on the words after it", "io": "IO: I- on every word of a span",
           "read": "read back", "bio_back": "three places", "io_back": "two places",
           "legend": "Purple dashed: the span that IO tags make of *Miramar* and *California*, two places that touch. "
                     "Read back, I-LOC I-LOC is one place of two words, as *San Diego* is."},
    "nl": {"title": "BIO-tags en IO-tags van dezelfde zin",
           "bio": "BIO: B- op het eerste woord van een span, I- op de woorden erna",
           "io": "IO: I- op elk woord van een span",
           "read": "teruggelezen", "bio_back": "drie plaatsen", "io_back": "twee plaatsen",
           "legend": "Paars gestreept: de span die IO-tags maken van *Miramar* en *California*, twee plaatsen die "
                     "elkaar raken. Teruggelezen is I-LOC I-LOC één plaats van twee woorden, net als *San Diego*."},
}


def bio_picture(lang: str) -> str:
    """The idea: the example with its BIO tags and with IO tags, and the spans each gives back."""
    w = _BIO_WORDS[lang]
    words = words_of(EXAMPLE)
    bio = tags_of(EXAMPLE)
    io_ = io_tags(len(words), EXAMPLE_SPANS)
    back_bio = uner.spans(bio)
    back_io = uner.spans(io_)
    roles_io = ["data" if x in back_bio else "off" for x in back_io]
    body = (_head(w["bio"]) + f'<div>{tagged_words(words, bio)}</div>'
            + _note(f'{w["read"]}, {w["bio_back"]}:') + f'<div>{spanned_words(words, back_bio)}</div>'
            + _head(w["io"]) + f'<div>{tagged_words(words, io_)}</div>'
            + _note(f'{w["read"]}, {w["io_back"]}:') + f'<div>{spanned_words(words, back_io, roles_io)}</div>')
    return yours.figure(w["title"], body + _note(w["legend"]),
                        alt=f'{w["title"]}. BIO: {" ".join(bio)}, {w["bio_back"]} {back_bio}. IO: {" ".join(io_)}, '
                            f'{w["io_back"]} {back_io}. {_bare(w["legend"])}')


# --- the demo: one word, several types -------------------------------------------------------------------------------

_KIND_NAMES = {"en": {"PER": "person", "LOC": "place", "ORG": "organization", "O": "no name (O)"},
               "nl": {"PER": "persoon", "LOC": "plaats", "ORG": "organisatie", "O": "geen naam (O)"}}
_DEMO_WORDS = {
    "en": {"title": "{word} in the sentences of Universal NER", "empty": "Type a word in the box.",
           "none": "{word} is not in the sentences of Universal NER. Capitals count: *Chicago* and *chicago* are "
                   "different words.",
           "times": "{n} times", "one": "1 time", "examples": "Examples",
           "note": "A word counts as a person, a place or an organization when it is inside a span of that type. The demo "
                   "looks for the word exactly as you type it, capitals included. The examples leave out sentences "
                   "with an address, a phone number or a crude or violent word."},
    "nl": {"title": "{word} in de zinnen van Universal NER", "empty": "Typ een woord in het vak.",
           "none": "{word} staat niet in de zinnen van Universal NER. Hoofdletters tellen mee: *Chicago* en *chicago* "
                   "zijn verschillende woorden.",
           "times": "{n} keer", "one": "1 keer", "examples": "Voorbeelden",
           "note": "Een woord telt als persoon, plaats of organisatie als het in een span van dat type staat. De demo "
                   "zoekt het woord precies zoals je het typt, met hoofdletters. De voorbeelden laten zinnen weg met "
                   "een adres, een telefoonnummer of een grof of gewelddadig woord."},
}
DEMO_EXAMPLES = 2   # example sentences per type
DEMO_SPAN = 5       # words the demo shows on each side of the word


def word_kinds(word: str) -> dict:
    """How often a word, spelled exactly so, capitals included, is inside a span of each type, and outside every span
    (O)."""
    word = str(word)
    out = Counter()
    for s in UNER:
        for w, t in s:
            if w == word:
                out[t[2:] if t != "O" else "O"] += 1
    return dict(out)


def _examples(word: str, kind: str) -> list:
    """Up to DEMO_EXAMPLES shown sentences with the word inside a span of this type (or outside every span, for O), as
    (words, spans) around the word."""
    out = []
    for s in _SHOWN:
        tags = tags_of(s)
        for i, (w, t) in enumerate(s):
            if w == word and (t[2:] if t != "O" else "O") == kind:
                a, b = max(0, i - DEMO_SPAN), min(len(s), i + DEMO_SPAN + 1)
                found = [x for x in uner.spans(tags) if a <= x[0] and x[1] <= b]
                out.append((words_of(s)[a:b], [(x[0] - a, x[1] - a, x[2]) for x in found], a > 0, b < len(s)))
                break
        if len(out) == DEMO_EXAMPLES:
            break
    return out


def demo_picture(word, lang: str) -> str:
    """A word the learner types: how often it is a person, a place, an organization or no name, with examples."""
    w = _DEMO_WORDS[lang]
    names = _KIND_NAMES[lang]
    word = str(word or "").strip()[:40]
    if not word:
        return yours.figure(w["empty"], "", alt=w["empty"])
    found = word_kinds(word)
    title = w["title"].format(word=word)
    if not found:
        shown_, plain = _said(w["none"], word=word)
        return yours.figure(title, f'<div style="{_UI}">{shown_}</div>', alt=f"{title}. {plain}")
    top = max(found.values())
    rows, said, parts = [], [], []
    for kind in (*KINDS, "O"):
        n = found.get(kind, 0)
        if not n:
            continue
        width = 0.3 + 9 * n / top
        bar = f'<div style="height:0.7rem;width:{width:.2f}rem;background:var(--ea-ink-3);border-radius:2px"></div>'
        count = w["one"] if n == 1 else w["times"].format(n=_n(n, lang))
        rows.append(f'<div style="{_CODE};color:var(--ea-ink)">{html.escape(names[kind])}</div><div>{bar}</div>'
                    f'<div style="{_UI};margin-top:0">{html.escape(count)}</div>')
        said.append(f"{names[kind]} {count}")
        for words, spans, cut_left, cut_right in _examples(word, kind):
            shown = (["…"] if cut_left else []) + words + (["…"] if cut_right else [])
            shift = 1 if cut_left else 0
            parts.append(f'<div>{spanned_words(shown, [(a + shift, b + shift, k) for a, b, k in spans])}</div>')
    table = (f'<div style="display:grid;grid-template-columns:max-content minmax(0,10rem) auto;gap:0.2rem 0.6rem;'
             f'align-items:center">{"".join(rows)}</div>')
    return yours.figure(title, table + _head(w["examples"]) + "".join(parts) + _note(w["note"]),
                        alt=f'{title}. {"; ".join(said)}. {_bare(w["note"])}')


# --- running the learner's function in a picture ---------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error
PICTURE_SECONDS = 3   # what the learner's function gets for all the sentences of a picture together


def _quiet(fn, args):
    """The learner's function on new copies of its lists, without what it prints; _STOPPED when it stops with an
    error."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*(list(a) if isinstance(a, list) else a for a in args))
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _on_all(fn, make_args) -> list | None:
    """The learner's function on every sentence of Universal NER, or None when that takes longer than
    PICTURE_SECONDS together."""
    try:
        return with_time_limit(lambda: [_quiet(fn, make_args(s)) for s in UNER], PICTURE_SECONDS)
    except Timeout:
        return None


def as_spans(v):
    """What the learner's to_spans gives, as a list of (start, end, kind) tuples, or None when it is not that: a span
    as a list counts as the same span as a tuple."""
    if not isinstance(v, (list, tuple)):
        return None
    out = []
    for x in v:
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            return None
        start, end, kind = x
        if not (isinstance(start, int) and isinstance(end, int) and isinstance(kind, str)) or isinstance(start, bool) \
                or isinstance(end, bool):
            return None
        out.append((start, end, kind))
    return out


def as_tags(v, n: int):
    """What the learner's to_tags gives, as a list of n values, or None when it is not a list or tuple of n."""
    if not isinstance(v, (list, tuple)) or len(v) != n:
        return None
    return list(v)


_SLOW = {"en": "Your function takes too long on all 11,722 sentences together. Does a loop of yours go on longer than "
               "it needs to?",
         "nl": "Je functie duurt te lang op alle 11.722 zinnen samen. Gaat een van je loops langer door dan nodig is?"}

_TAGS_WORDS = {
    "en": {"title": "Your to_tags on sentences of Universal NER",
           "example": "The example, with the spans of the exercise", "york": "A blog", "headline": "A headline",
           "fine": "Your tags are those of Universal NER, also for the other {total} sentences: your function turns "
                   "the spans of each sentence back into its tags.",
           "off": "Purple dashed: a tag of yours that differs from the tag in Universal NER.",
           "count": "Of all {total} sentences, {same} get exactly the tags of Universal NER.",
           "stopped": "Your function stops with an error on {k} of the {total} sentences.",
           "length": "For a sentence of {n} words, your function gives back {got} instead of a list of {n} tags.",
           "blue": "Blue: your tags."},
    "nl": {"title": "Jouw to_tags op zinnen van Universal NER",
           "example": "Het voorbeeld, met de spans van de oefening", "york": "Een blog", "headline": "Een kop",
           "fine": "Je tags zijn gelijk aan de tags van Universal NER, ook voor de andere {total} zinnen: je functie "
                   "maakt van de spans van elke zin weer de tags ervan.",
           "off": "Paars gestreept: een tag van jou die verschilt van de tag in Universal NER.",
           "count": "Van de {total} zinnen krijgen er {same} precies de tags van Universal NER.",
           "stopped": "Je functie stopt met een fout op {k} van de {total} zinnen.",
           "length": "Voor een zin van {n} woorden geeft je functie {got} terug in plaats van een lijst van {n} tags.",
           "blue": "Blauw: jouw tags."},
}


def _short(v) -> str:
    text = _repr(v)
    return text if len(text) <= 60 else text[:59] + "…"


def _said(template: str, **values) -> tuple:
    """A note as HTML and as plain text, with the values filled in after the marking: a * or a ` in a value of the
    learner's stays as it is."""
    out, plain = _marked(template), _bare(template)
    for key, value in values.items():
        out = out.replace("{" + key + "}", html.escape(str(value)))
        plain = plain.replace("{" + key + "}", str(value))
    return out, plain


def _figure(title: str, rows: list, said: list, notes: list) -> str:
    body = "".join(rows) + f'<div style="{_UI}">{" ".join(h for h, _ in notes)}</div>'
    return yours.figure(title, body, alt=f'{title}. {"; ".join(said)}. {" ".join(p for _, p in notes)}')


def tags_picture(seen) -> str:
    """The learner's to_tags on three sentences, each tag beside the one in Universal NER, and how many of all the
    sentences come out right."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _TAGS_WORDS[lang]
    total = len(UNER)
    rows, notes, said, off = [], [], [], False
    for key, sentence in (("example", EXAMPLE), ("york", YORK), ("headline", HEADLINE)):
        words, want = words_of(sentence), tags_of(sentence)
        try:
            got = with_time_limit(lambda: _quiet(fn, (len(words), uner.spans(want))), PICTURE_SECONDS)
        except Timeout:
            return yours.figure(w["title"], _note(_SLOW[lang]), alt=f'{w["title"]}. {_SLOW[lang]}')
        if got is _STOPPED:
            continue
        tags = as_tags(got, len(words))
        if tags is None:
            note = _said(w["length"], n=len(words), got=_short(got))
            if note not in notes:
                notes.append(note)
            continue
        roles = ["mine" if t == want[i] else "off" for i, t in enumerate(tags)]
        off = off or "off" in roles
        rows.append(_head(w[key]) + f'<div>{tagged_words(words, tags, roles)}</div>')
        said.append(f'{w[key]}: ' + " ".join(f"{x}/{_shown_tag(t)}" for x, t in zip(words, tags)))
    mine = _on_all(fn, lambda s: (len(s), uner.spans(tags_of(s))))
    if mine is None:
        notes.append(_said(_SLOW[lang]))
    else:
        stopped = sum(1 for g in mine if g is _STOPPED)
        same = sum(1 for g, s in zip(mine, UNER) if g is not _STOPPED and as_tags(g, len(s)) == tags_of(s))
        if stopped:
            notes.append(_said(w["stopped"], k=_n(stopped, lang), total=_n(total, lang)))
        if same == total:
            notes.insert(0, _said(w["fine"], total=_n(total - 3, lang)))
        else:
            notes.insert(0, _said(w["count"], total=_n(total, lang), same=_n(same, lang)))
            if off:
                notes.insert(0, _said(w["off"]))
    return _figure(w["title"], rows, said, notes + [_said(w["blue"])])


_SPANS_SEEN_WORDS = {
    "en": {"title": "Your to_spans on sentences of Universal NER",
           "example": "The example", "headline": "A headline: the last word is part of a name",
           "tagger": "The example with the tags of a tagger: an I- after an O, on *Diego*",
           "yours": "yours", "course": "the course's",
           "fine": "Your spans are the course's, also on all {total} sentences: {spans} names, {long} of them more than "
                   "one word.",
           "off": "Purple dashed: a span of yours that the course does not have.",
           "count": "Of all {total} sentences, {same} get exactly the course's spans.",
           "stopped": "Your function stops with an error on {k} of the {total} sentences.",
           "odd": "Your function gives back {got}, and not a list of spans `(start, end, kind)`.",
           "blue": "Blue: your spans. Gray: the tags your function reads, and the course's spans."},
    "nl": {"title": "Jouw to_spans op zinnen van Universal NER",
           "example": "Het voorbeeld", "headline": "Een kop: het laatste woord hoort bij een naam",
           "tagger": "Het voorbeeld met de tags van een tagger: een I- na een O, op *Diego*",
           "yours": "van jou", "course": "van de course",
           "fine": "Je spans zijn gelijk aan de spans van de course, ook op alle {total} zinnen: {spans} namen, waarvan "
                   "{long} van meer dan één woord.",
           "off": "Paars gestreept: een span van jou die de course niet heeft.",
           "count": "Van de {total} zinnen krijgen er {same} precies de spans van de course.",
           "stopped": "Je functie stopt met een fout op {k} van de {total} zinnen.",
           "odd": "Je functie geeft {got} terug, en geen lijst met spans `(start, end, kind)`.",
           "blue": "Blauw: jouw spans. Grijs: de tags die je functie leest, en de spans van de course."},
}


def spans_seen_picture(seen) -> str:
    """The learner's to_spans on the example, a headline and the tags of a tagger, beside the course's spans, and on
    all the sentences."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _SPANS_SEEN_WORDS[lang]
    total = len(UNER)
    rows, notes, said, off = [], [], [], False
    for key, words, tags in (("example", words_of(EXAMPLE), tags_of(EXAMPLE)),
                             ("headline", words_of(HEADLINE), tags_of(HEADLINE)),
                             ("tagger", words_of(EXAMPLE), TAGGER)):
        want = uner.spans(tags)
        try:
            got = with_time_limit(lambda: _quiet(fn, (list(tags),)), PICTURE_SECONDS)
        except Timeout:
            return yours.figure(w["title"], _note(_SLOW[lang]), alt=f'{w["title"]}. {_SLOW[lang]}')
        if got is _STOPPED:
            continue
        spans = as_spans(got)
        if spans is None:
            note = _said(w["odd"], got=_short(got))
            if note not in notes:
                notes.append(note)
            continue
        roles = ["mine" if x in want else "off" for x in spans]
        off = off or "off" in roles
        rows.append(_head(w[key]) + f'<div>{tagged_words(words, tags)}</div>'
                    + f'<div>{spanned_words(words, spans, roles)}</div>'
                    + f'<div style="{_UI}">{html.escape(w["yours"])}:</div>{_spans_code(spans, roles)}'
                    + f'<div style="{_UI}">{html.escape(w["course"])}:</div>{_spans_code(want)}')
        said.append(f'{_bare(w[key])}: {w["yours"]} {spans}, {w["course"]} {want}')
    mine = _on_all(fn, lambda s: (tags_of(s),))
    if mine is None:
        notes.append(_said(_SLOW[lang]))
    else:
        stopped = sum(1 for g in mine if g is _STOPPED)
        same = sum(1 for g, s in zip(mine, UNER) if g is not _STOPPED and as_spans(g) == uner.spans(tags_of(s)))
        if stopped:
            notes.append(_said(w["stopped"], k=_n(stopped, lang), total=_n(total, lang)))
        if same == total:
            c = counts()
            notes.insert(0, _said(w["fine"], total=_n(total, lang), spans=_n(c["spans"], lang),
                                  long=_n(c["long"], lang)))
        else:
            notes.insert(0, _said(w["count"], total=_n(total, lang), same=_n(same, lang)))
            if off:
                notes.insert(0, _said(w["off"]))
    return _figure(w["title"], rows, said, notes + [_said(w["blue"])])


# --- the checks ------------------------------------------------------------------------------------------------------

REQUIRED = ["to_tags", "to_spans", "io-or-bio", "kinds"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: "..." in "\n".join(
                    line.split("#")[0] for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang de `...` door je eigen code.")


def _listed(v):
    """What to_tags gives, with a tuple of tags made a list, as the picture under the check reads it."""
    return list(v) if isinstance(v, tuple) else v


def _tuples(v):
    """What to_spans gives, with each span that is a list made a tuple: [0, 1, "LOC"] counts as (0, 1, "LOC"), and a
    tuple of spans as a list of them."""
    if isinstance(v, (list, tuple)) and all(isinstance(x, (list, tuple)) for x in v):
        return [tuple(x) for x in v]
    return v


def _tuple_changed(e) -> bool:
    """A number inside a span changed, as in spans[-1][1] = i + 1: a tuple cannot change."""
    return isinstance(e, TypeError) and "does not support item assignment" in str(e)


def _index_error(e) -> bool:
    """A tag put at a position that the list of tags does not have."""
    return isinstance(e, IndexError) and "assignment" in str(e)


def _prefixed(got, text) -> bool:
    """Spans whose type still holds the B- or I- of the tag."""
    spans = as_spans(got)
    return bool(spans) and any(k[:2] in ("B-", "I-") for _, _, k in spans)


def _not_spans(got, text) -> bool:
    return got is not None and as_spans(got) is None


# exercise 1: to_tags. The visible case: the example; then the blog and the headline, no spans, a span of three words,
# a span up to the last word, spans out of order, and one word.
TT_CASES = [
    (10, [(0, 1, "LOC"), (1, 2, "LOC"), (7, 9, "LOC")]),
    (11, [(0, 2, "PER"), (4, 6, "LOC"), (8, 10, "ORG")]),
    (5, [(0, 1, "LOC"), (3, 5, "LOC")]),
    (4, []),
    (7, [(1, 4, "ORG")]),
    (6, [(2, 6, "ORG")]),
    (3, [(2, 3, "PER"), (0, 2, "LOC")]),
    (1, [(0, 1, "PER")]),
]
_TT_HEAD = "def to_tags(n, spans):\n"
_TT_START = _TT_HEAD + "    ...\n"
_TT = (_TT_HEAD
       + '    tags = ["O"] * n\n'
       + "    for start, end, kind in spans:\n"
       + '        tags[start] = "B-" + kind\n'
       + "        for i in range(start + 1, end):\n"
       + '            tags[i] = "I-" + kind\n'
       + "    return tags\n")
_TT_IO = _TT.replace('        tags[start] = "B-" + kind\n        for i in range(start + 1, end):\n',
                     "        for i in range(start, end):\n")
_TT_ALL_B = _TT.replace('"I-" + kind', '"B-" + kind')
_TT_END = _TT.replace("range(start + 1, end)", "range(start + 1, end + 1)")
_TT_DASH = _TT.replace('"B-" + kind', '"B" + kind').replace('"I-" + kind', '"I" + kind')
_TT_FIRST = _TT.replace('        for i in range(start + 1, end):\n            tags[i] = "I-" + kind\n', "")

# exercise 2: to_spans. The visible case: the example; then the blog, the headline (a name up to the last word), the
# tagger's tags (an I- after an O), an I- of another type, no names, no words, an I- at the start, a span of four words,
# and two persons that touch.
TS_CASES = [
    (tags_of(EXAMPLE),),
    (tags_of(YORK),),
    (tags_of(HEADLINE),),
    (list(TAGGER),),
    (["B-PER", "I-LOC", "I-LOC"],),
    (["O", "O"],),
    ([],),
    (["I-ORG"],),
    (["B-ORG", "I-ORG", "I-ORG", "I-ORG", "O"],),
    (["B-PER", "B-PER", "I-PER"],),
]
_TS_HEAD = "def to_spans(tags):\n"
_TS_START = _TS_HEAD + "    ...\n"
_TS = (_TS_HEAD
       + "    spans = []\n"
       + "    for i in range(len(tags)):\n"
       + "        tag = tags[i]\n"
       + '        if tag[0] == "B":\n'
       + "            spans.append((i, i + 1, tag[2:]))\n"
       + '        elif tag[0] == "I":\n'
       + "            if len(spans) > 0 and spans[-1][1] == i and spans[-1][2] == tag[2:]:\n"
       + "                spans[-1] = (spans[-1][0], i + 1, tag[2:])\n"
       + "            else:\n"
       + "                spans.append((i, i + 1, tag[2:]))\n"
       + "    return spans\n")
_TS_END = (_TS.replace("(i, i + 1, tag[2:])", "(i, i, tag[2:])")
           .replace("(spans[-1][0], i + 1, tag[2:])", "(spans[-1][0], i, tag[2:])")
           .replace("spans[-1][1] == i and", "spans[-1][1] == i - 1 and"))
_TS_MERGE = _TS.replace('if tag[0] == "B":\n            spans.append((i, i + 1, tag[2:]))\n        elif tag[0] == "I":',
                        'if tag != "O":')
_TS_SKIP = _TS.replace("            else:\n                spans.append((i, i + 1, tag[2:]))\n", "")
_TS_ANY = _TS.replace("len(spans) > 0 and spans[-1][1] == i and spans[-1][2] == tag[2:]", "len(spans) > 0")
_TS_PREFIX = _TS.replace("tag[2:]", "tag")
_TS_BEFORE = _TS.replace("spans[-1][1] == i and", "spans[-1][1] == i - 1 and")
_TS_AFTER = _TS.replace("spans[-1][1] == i and", "spans[-1][1] == i + 1 and")
_TS_TYPE = _TS.replace(" and spans[-1][2] == tag[2:]", "")
_TS_GAP = _TS.replace(" and spans[-1][1] == i", "")
# an I- that never continues a span: i - 1 or i + 1 in place of i
_SPLIT_TEXT = {"en": "`{call}` gives {got}, and it should give {want}: an I- right after a span of the same type starts a "
                     "span of its own, so a name of two words comes back as two names. A span that stops right before "
                     "token `i`: what is its `end`, as in a slice?",
               "nl": "`{call}` geeft {got}, en dat moet {want} zijn: een I- direct na een span van hetzelfde type begint "
                     "een eigen span, dus een naam van twee woorden komt terug als twee namen. Een span die direct voor "
                     "token `i` ophoudt: wat is dan de `end`, net als bij een slice?"}

EXERCISES = {
    "to_tags": FunctionBox(
        name="to_tags",
        picture=tags_picture,
        function="to_tags",
        start=_TT_START,
        solution=_TT,
        visible=TT_CASES[:1],
        hidden=TT_CASES[1:],
        collect=_listed,
        first=[_DOTS],
        mistakes=[
            Mistake("index", error=_index_error,
                    en="With `{call}`, your function stops with an IndexError: it puts a tag at a position that its list does not have. How many tags should the list hold, and which value of the call says so?",
                    nl="Met `{call}` stopt je functie met een IndexError: je functie zet een tag op een positie die niet in de lijst bestaat. Hoeveel tags hoort de lijst te bevatten, en welke waarde van de aanroep zegt dat?"),
            Mistake("io", wrong=_TT_IO,
                    en="`{call}` gives {got}, and it should give {want}: every word of a span gets an I-, also its first word, as in the IO tags of the worked example. When two spans touch, how can someone who reads your tags see where the one ends and the other begins?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elk woord van een span krijgt een I-, ook het eerste, zoals in de IO-tags van het uitgewerkte voorbeeld. Als twee spans elkaar raken, hoe ziet iemand die je tags leest dan waar de ene ophoudt en de andere begint?"),
            Mistake("all-b", wrong=_TT_ALL_B,
                    en="`{call}` gives {got}, and it should give {want}: every word of a span gets a B-, so a name of two words reads as two names of one word. Which word of a span is the only one with a B-?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elk woord van een span krijgt een B-, dus een naam van twee woorden leest als twee namen van één woord. Welk woord van een span is het enige met een B-?"),
            Mistake("end", wrong=_TT_END,
                    en="`{call}` gives {got}, and it should give {want}: your function also tags the word at position `end`, which comes after the span. Which positions does a slice `words[start:end]` cover?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie tagt ook het woord op positie `end`, dat na de span komt. Welke posities beslaat een slice `words[start:end]`?"),
            Mistake("dash", wrong=_TT_DASH,
                    en="`{call}` gives {got}, and it should give {want}: your tags have no hyphen between the letter and the type. How is a tag of Universal NER written?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je tags hebben geen streepje tussen de letter en het type. Hoe wordt een tag van Universal NER geschreven?"),
            Mistake("first-only", wrong=_TT_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: only the first word of each span gets a tag, and the words after it stay O. What do the other words of a span get?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: alleen het eerste woord van elke span krijgt een tag, en de woorden erna blijven O. Wat krijgen de andere woorden van een span?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: start with a list of `n` times `\"O\"`, and give the words of each span their B- and I- tags.",
                      "nl": "Vervang de `...`: begin met een lijst van `n` keer `\"O\"`, en geef de woorden van elke span een B- of een I-tag."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which word of a span gets a B-, which words get an I-, and which words keep their O?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Welk woord van een span krijgt een B-, welke woorden krijgen een I-, en welke woorden houden een O?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for any number of words, for no spans at all, and for spans in any order?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elk aantal woorden, zonder spans, en voor spans in elke volgorde?"},
            "correct": {"en": "Your `to_tags` is right in all {n} cases. Look at the picture: your tags on three sentences, and on all 11,722 sentences of Universal NER.",
                        "nl": "Je `to_tags` klopt in alle {n} gevallen. Kijk naar de afbeelding: je tags op drie zinnen, en op alle 11.722 zinnen van Universal NER."},
        },
    ),
    "to_spans": FunctionBox(
        name="to_spans",
        picture=spans_seen_picture,
        function="to_spans",
        start=_TS_START,
        solution=_TS,
        visible=TS_CASES[:1],
        hidden=TS_CASES[1:],
        collect=_tuples,
        first=[_DOTS],
        mistakes=[
            Mistake("end", wrong=_TS_END,
                    en="`{call}` gives {got}, and it should give {want}: the `end` of each of your spans is the position of its last word, and not of the word after it. What does a slice `words[start:end]` give for your spans?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de `end` van elk van je spans is de positie van het laatste woord, en niet van het woord erna. Wat geeft een slice `words[start:end]` voor je spans?"),
            Mistake("merge", wrong=_TS_MERGE,
                    en="`{call}` gives {got}, and it should give {want}: a B- right after a span of the same type makes that span longer, so two names that touch become one. What does a B- say about the word it is on?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een B- direct na een span van hetzelfde type maakt die span langer, dus twee namen die elkaar raken, worden één naam. Wat zegt een B- over het woord waar de B- op staat?"),
            Mistake("skip", wrong=_TS_SKIP,
                    en="`{call}` gives {got}, and it should give {want}: an I- that does not continue a span is left out, so a name that a tagger starts with an I- gets lost. What does the lesson make of an I- after an O, or after a span of another type?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een I- die geen span voortzet, valt weg, dus een naam die een tagger met een I- begint, raakt zoek. Wat maakt de les van een I- na een O, of na een span van een ander type?"),
            Mistake("tuple", error=_tuple_changed,
                    en="With `{call}`, your function stops with a TypeError: it changes a number inside a span, and a tuple cannot change once it is made. How can you put a new span in the place of the last one?",
                    nl="Met `{call}` stopt je functie met een TypeError: je functie verandert een getal in een span, en een tuple kan na het maken niet meer veranderen. Hoe zet je een nieuwe span op de plek van de laatste?"),
            Mistake("split", wrong=_TS_AFTER, en=_SPLIT_TEXT["en"], nl=_SPLIT_TEXT["nl"]),
            Mistake("type", wrong=_TS_TYPE,
                    en="`{call}` gives {got}, and it should give {want}: an I- continues the span right before it, also when that span has another type, so a person and a place become one span. Which part of the tag has to fit the span before it?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een I- zet de span direct ervoor voort, ook als die span een ander type heeft, dus een persoon en een plaats worden één span. Welk deel van de tag moet bij de span ervoor passen?"),
            Mistake("gap", wrong=_TS_GAP,
                    en="`{call}` gives {got}, and it should give {want}: an I- makes the last span of its type longer, also when an O comes between them. When does an I- continue the span before it?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een I- maakt de laatste span van hetzelfde type langer, ook als er een O tussen zit. Wanneer zet een I- de span ervoor voort?"),
            Mistake("any-before", wrong=_TS_ANY,
                    en="`{call}` gives {got}, and it should give {want}: an I- makes the last span longer, also when an O or a span of another type comes between them. When does an I- continue the span before it?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een I- maakt de laatste span langer, ook als er een O of een span van een ander type tussen zit. Wanneer zet een I- de span ervoor voort?"),
            Mistake("prefix", matches=_prefixed,
                    en="`{call}` gives {got}, and it should give {want}: the types of your spans still start with B- or I-, such as `'B-LOC'`. Which part of a tag is its type?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de types van je spans beginnen nog met B- of I-, zoals `'B-LOC'`. Welk deel van een tag is het type?"),
            Mistake("not-spans", matches=_not_spans,
                    en="`{call}` gives {got}, and it should give {want}: a list of spans, each a tuple `(start, end, kind)`. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een lijst met spans, elk een tuple `(start, end, kind)`. Wat geeft je `return`-regel terug?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: go over the tags from left to right, start a new span at a B-, and at an I- make the span before it one word longer when the I- continues it.",
                      "nl": "Vervang de `...`: ga de tags van links naar rechts langs, begin bij een B- een nieuwe span, en maak bij een I- de span ervoor één woord langer als de I- die span voortzet."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Where does each span start, where does it end, and what is its type?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Waar begint elke span, waar houdt de span op, en wat is het type?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function also give back a span that runs up to the last word, and a span for an I- that does not continue one?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Geeft je functie ook een span terug die tot het laatste woord loopt, en een span voor een I- die geen span voortzet?"},
            "correct": {"en": "Your `to_spans` is right in all {n} cases, also for an I- after an O. Look at the picture: your spans on three sentences, and on all 11,722 sentences of Universal NER.",
                        "nl": "Je `to_spans` klopt in alle {n} gevallen, ook voor een I- na een O. Kijk naar de afbeelding: je spans op drie zinnen, en op alle 11.722 zinnen van Universal NER."},
        },
    ),
    "io-or-bio": Sort(
        name="io-or-bio",
        groups={
            "io": {"en": "IO tags keep these spans apart", "nl": "IO-tags houden deze spans uit elkaar"},
            "bio": {"en": "Only BIO tags keep these spans apart", "nl": "Alleen BIO-tags houden deze spans uit elkaar"},
        },
        items={
            "center": {"en": "We are staying at a hotel in [LOC Center City] [LOC Philadelphia] so I wanted something close.",
                       "nl": "*We are staying at a hotel in [LOC Center City] [LOC Philadelphia] so I wanted something close.* (We logeren in een hotel in Center City in Philadelphia, dus ik wilde iets dichtbij.)"},
            "lopez": {"en": "1572 [PER Ruy Lopez] defeats several eminent players in [LOC Rome].",
                      "nl": "*1572 [PER Ruy Lopez] defeats several eminent players in [LOC Rome].* (1572: Ruy Lopez verslaat in Rome een paar vooraanstaande spelers.)"},
            "rafael": {"en": "What to do in [LOC San Rafael] [LOC Ca]?",
                       "nl": "*What to do in [LOC San Rafael] [LOC Ca]?* (Wat is er te doen in San Rafael, Californië?)"},
            "defense": {"en": "Orders for the deal to go ahead were given by the [LOC US] [ORG Department of Defense].",
                        "nl": "*Orders for the deal to go ahead were given by the [LOC US] [ORG Department of Defense].* (Het Amerikaanse ministerie van Defensie gaf opdracht om de deal door te laten gaan.)"},
            "lawrence": {"en": "Absolutely my favorite store in [LOC Lawrence], [LOC KS]",
                         "nl": "*Absolutely my favorite store in [LOC Lawrence], [LOC KS]* (Echt mijn favoriete winkel in Lawrence, Kansas)"},
            "zealand": {"en": "is there any good places to get an ice-cream sundae from in [LOC Invercargill] [LOC New Zealand]?",
                        "nl": "*is there any good places to get an ice-cream sundae from in [LOC Invercargill] [LOC New Zealand]?* (zijn er goede plekken voor een sundae in Invercargill, Nieuw-Zeeland?)"},
            "richmond": {"en": "the [LOC Richmond] [ORG Ukrops] chain is known for its charity work",
                         "nl": "*the [LOC Richmond] [ORG Ukrops] chain is known for its charity work* (de keten Ukrops uit Richmond staat bekend om liefdadigheidswerk)"},
            "pancras": {"en": "Eurostar runs from [LOC London] [LOC St Pancras station].",
                        "nl": "*Eurostar runs from [LOC London] [LOC St Pancras station].* (Eurostar rijdt vanaf station St Pancras in Londen.)"},
        },
        solution={"center": "bio", "lopez": "io", "rafael": "bio", "defense": "io", "lawrence": "io", "zealand": "bio",
                  "richmond": "io", "pancras": "bio"},
        nudge={
            "center": {"en": "Do the two spans touch, and do they have the same type?",
                       "nl": "Raken de twee spans elkaar, en hebben de twee spans hetzelfde type?"},
            "lopez": {"en": "What lies between the person and the place?",
                      "nl": "Wat staat er tussen de persoon en de plaats?"},
            "rafael": {"en": "With I-LOC on all three words, how many places would you read back?",
                       "nl": "Hoeveel plaatsen lees je terug als alle drie de woorden I-LOC hebben?"},
            "defense": {"en": "The two spans touch: with IO tags, what changes at the word Department?",
                        "nl": "De twee spans raken elkaar: wat verandert er bij IO-tags op het woord Department?"},
            "lawrence": {"en": "What stands between the two places, and what tag does it get?",
                         "nl": "Wat staat er tussen de twee plaatsen, en welke tag krijgt het?"},
            "zealand": {"en": "With I-LOC on all three words, how many places would you read back?",
                        "nl": "Hoeveel plaatsen lees je terug als alle drie de woorden I-LOC hebben?"},
            "richmond": {"en": "Do the two spans that touch have the same type?",
                         "nl": "Hebben de twee spans die elkaar raken, hetzelfde type?"},
            "pancras": {"en": "Do the two places touch, and what would I-LOC on all four words read back as?",
                        "nl": "Raken de twee plaatsen elkaar, en wat lees je terug als alle vier de woorden I-LOC hebben?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the groups again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de groepen opnieuw uit."},
            "empty": {"en": "Put each sentence in the group that fits its spans.",
                      "nl": "Zet elke zin in de groep die bij de spans past."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The sentence {item} belongs in the other group. Write its IO tags: which spans do you read back?",
                      "nl": "De zin {item} hoort in de andere groep. Schrijf de IO-tags op: welke spans lees je terug?"},
            "correct": {"en": "All {n} are right. IO tags lose a span only where two spans of the same type touch, mostly a place right before a bigger place. A word between them, or another type, keeps them apart.",
                        "nl": "Alle {n} kloppen. IO-tags raken alleen een span kwijt als twee spans van hetzelfde type elkaar raken, meestal een plaats direct voor een grotere plaats. Een woord ertussen, of een ander type, houdt de spans uit elkaar."},
        },
    ),
}

CASES = [
    ("to_tags", _TT_START, "empty", None),
    ("to_tags", _TT, "correct", None),
    ("to_tags", _TT.replace("    return tags\n", "    return tuple(tags)\n"), "correct", None),
    ("to_tags", "def to_tags(n, spans):\n    tags = []\n    for i in range(n):\n        tags.append(\"O\")\n    for start, end, kind in spans:\n        for i in range(start, end):\n            if i == start:\n                tags[i] = \"B-\" + kind\n            else:\n                tags[i] = \"I-\" + kind\n    return tags\n", "correct", None),
    ("to_tags", "def to_tags(n, spans):\n    tags = [\"O\"] * n\n    for start, end, kind in spans:\n        tags[start:end] = [f\"B-{kind}\"] + [f\"I-{kind}\"] * (end - start - 1)\n    return tags\n", "correct", None),
    ("to_tags", _TT_IO, "wrong", "io"),
    ("to_tags", _TT_ALL_B, "wrong", "all-b"),
    ("to_tags", _TT_END, "wrong", "end"),
    ("to_tags", _TT_DASH, "wrong", "dash"),
    ("to_tags", _TT_FIRST, "wrong", "first-only"),
    ("to_tags", _TT.replace('["O"] * n', '["O"] * len(spans)'), "error", "index"),
    ("to_tags", "def to_tags(n, spans):\n    return ['B-LOC', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'O']\n", "almost", None),
    ("to_tags", _TT.replace("    tags = [\"O\"] * n\n", "    tags = [\"O\"] * n\n    ...\n"), "wrong", "dots-left"),
    ("to_spans", _TS_START, "empty", None),
    ("to_spans", _TS, "correct", None),
    ("to_spans", _TS.replace("(i, i + 1, tag[2:])", "[i, i + 1, tag[2:]]").replace("(spans[-1][0], i + 1, tag[2:])", "[spans[-1][0], i + 1, tag[2:]]"), "correct", None),
    ("to_spans", "def to_spans(tags):\n    spans = []\n    start = None\n    for i, tag in enumerate(tags + [\"O\"]):\n        if start is not None and (tag == \"O\" or tag.startswith(\"B-\") or tag[2:] != kind):\n            spans.append((start, i, kind))\n            start = None\n        if tag != \"O\" and start is None:\n            start, kind = i, tag[2:]\n    return spans\n", "correct", None),
    ("to_spans", _TS_END, "wrong", "end"),
    ("to_spans", _TS_MERGE, "wrong", "merge"),
    ("to_spans", _TS_SKIP, "wrong", "skip"),
    ("to_spans", _TS_ANY, "wrong", "any-before"),
    ("to_spans", _TS_PREFIX, "wrong", "prefix"),
    ("to_spans", _TS_BEFORE, "wrong", "split"),   # i - 1 and i + 1 give the same spans on every case
    ("to_spans", _TS_AFTER, "wrong", "split"),
    ("to_spans", _TS_TYPE, "wrong", "type"),
    ("to_spans", _TS_GAP, "wrong", "gap"),
    ("to_spans", _TS.replace("spans[-1] = (spans[-1][0], i + 1, tag[2:])", "spans[-1][1] = i + 1"), "error", "tuple"),
    ("to_spans", _TS.replace("    return spans\n", "    return tuple(spans)\n"), "correct", None),
    ("to_spans", "def to_spans(tags):\n    return len(tags)\n", "wrong", "not-spans"),
    ("to_spans", "def to_spans(tags):\n    return [(0, 1, 'LOC'), (1, 2, 'LOC'), (7, 9, 'LOC')]\n", "almost", None),
    ("io-or-bio", {}, "empty", None),
    ("io-or-bio", {"center": "bio", "lopez": "io", "rafael": "bio", "defense": "io", "lawrence": "io", "zealand": "bio",
                   "richmond": "io", "pancras": "bio"}, "correct", None),
    ("io-or-bio", {"center": "bio", "lopez": "io"}, "wrong", "unfinished"),
    ("io-or-bio", {"center": "io"}, "wrong", "wrong:center"),
    ("io-or-bio", {"center": "bio", "lopez": "io", "rafael": "bio", "defense": "bio", "lawrence": "io", "zealand": "bio",
                   "richmond": "io", "pancras": "bio"}, "wrong", "wrong:defense"),
]
