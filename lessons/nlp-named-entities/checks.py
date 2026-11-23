# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-named-entities: find the longest name of a gazetteer that starts at a position
(longest_at), give a word its short word shape (shape), and tag a sentence with the tag before each word
(tag_sentence); with the pictures of what each function does on the test set, two pictures of the idea and a demo.

The sentences are those of Universal NER's English-EWT (everything_ai.uner, without the emails), with its own split
into a training set and a test set, so nothing is downloaded. Every number in the text and the pictures is computed by
the course (tests/test_nlp_named_entities.py computes them again)."""
import ast
import contextlib
import copy
import gzip
import html
import io
import re
from types import MappingProxyType

from everything_ai import paths, uner, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, kept, run_code, with_time_limit

_KEY = "nlp-named-entities"

# --- the course's data -----------------------------------------------------------------------------------------------

TRAIN = uner.split("train")   # 8,773 sentences: tuples of (word, tag) pairs
TEST = uner.split("test")     # 1,471
KINDS = uner.KINDS            # PER, LOC, ORG
EXAMPLE = uner.sentence("newsgroup-groups.google.com_civilization_1201f7692b7769fb_ENG_20050908_010400-0006").pairs
# Alteration of the Mississippi River and the destruction of wetlands at its mouth have left the area around New
# Orleans abnormally vulnerable to the forces of nature .
UN = uner.sentence("weblog-blogspot.com_grandpasgripes_20060413051000_ENG_20060413_051000-0006").pairs
# They are certainly being nasty to the United Nations Security Council in connection with the anti-proliferation
# treaty .
KOREA = uner.sentence("newsgroup-groups.google.com_hiddennook_f50294175d32a8ac_ENG_20041120_152800-0003").pairs
# The talk of the day besides a more level playing field with China was North Korea .
PENTAGON = uner.sentence("weblog-juancole.com_juancole_20040722101300_ENG_20040722_101300-0019").pairs
# The Pentagon did not even have a plan for dealing with Afghanistan or al - Qaeda that it could pull off the shelf ,
# according to Bob Woodward .
PLENTY = uner.sentence("answers-20111106022931AAwpqXm_ans-0004").pairs
# Bay of Plenty - Are you even old enough to vote ?
UD_EN = paths.ROOT / "data" / "ud-en-ewt.conllu.gz"   # the treebank of the lesson about tagging word classes
START = "O"   # the tag before the first word of a sentence: no name stands before it


def words_of(sentence) -> list:
    return [w for w, _ in sentence]


def tags_of(sentence) -> list:
    return [t for _, t in sentence]


def spans(tags) -> list:
    """The course's to_spans of the lesson about BIO: the spans (start, end, kind) of a list of tags."""
    return uner.spans(list(tags))


def name_counts(sentences) -> dict:
    """For each name of the sentences, as a tuple of its words, how often it has each type."""
    counts = {}
    for sentence in sentences:
        words = words_of(sentence)
        for start, end, kind in spans(tags_of(sentence)):
            name = tuple(words[start:end])
            if name not in counts:
                counts[name] = {}
            counts[name][kind] = counts[name].get(kind, 0) + 1
    return counts


def most_frequent(counts) -> dict:
    """For each key, the value it has most often; of two that tie, the one counted first."""
    best = {}
    for key in counts:
        top = None
        for value in counts[key]:
            if top is None or counts[key][value] > counts[key][top]:
                top = value
        best[key] = top
    return best


NAME_COUNTS = kept("nlp-named-entities:names", lambda: name_counts(TRAIN))
GAZETTEER = MappingProxyType(most_frequent(NAME_COUNTS))   # (word, ...) -> the type it has most often in TRAIN


def one_word_at(words, i, gazetteer):
    """The worked example: the name of one word at position i, or None."""
    name = (words[i],)
    if name in gazetteer:
        return (i, i + 1, gazetteer[name])
    return None


def longest_at(words, i, gazetteer):
    """The course's own Exercise 1: the longest name of the gazetteer that starts at position i, as a span, or None."""
    for end in range(len(words), i, -1):
        name = tuple(words[i:end])
        if name in gazetteer:
            return (i, end, gazetteer[name])
    return None


def find_names(words, gazetteer, look=longest_at) -> list:
    """The worked example's find_names: from left to right, at each position that is not inside a name found already,
    the name that `look` finds there."""
    found = []
    free = 0
    for i in range(len(words)):
        if i >= free:
            span = look(words, i, gazetteer)
            if span is not None:
                found.append(span)
                free = span[1]
    return found


def shape(word):
    """The course's own Exercise 2: the short word shape of Jurafsky and Martin. A capital is X, a small letter x and a
    digit d; any other character stays as it is; a run of the same kind becomes one."""
    out = ""
    last = ""
    for ch in word:
        if ch.isupper():
            kind = "X"
        elif ch.islower():
            kind = "x"
        elif ch.isdigit():
            kind = "d"
        else:
            kind = ch
        if kind != last:
            out = out + kind
            last = kind
    return out


def before_counts(sentences, key) -> dict:
    """For each (key(word), tag before), how often the word has each tag: the tag before the first word is O."""
    counts = {}
    for sentence in sentences:
        before = START
        for word, tag in sentence:
            k = (key(word), before)
            if k not in counts:
                counts[k] = {}
            counts[k][tag] = counts[k].get(tag, 0) + 1
            before = tag
    return counts


WORD_COUNTS = kept("nlp-named-entities:by-word", lambda: before_counts(TRAIN, lambda w: w))
SHAPE_COUNTS = kept("nlp-named-entities:by-shape", lambda: before_counts(TRAIN, shape))
BY_WORD = MappingProxyType(most_frequent(WORD_COUNTS))     # (word, tag before) -> its most frequent tag in TRAIN
BY_SHAPE = MappingProxyType(most_frequent(SHAPE_COUNTS))   # (shape, tag before) -> its most frequent tag in TRAIN


def tag_sentence(words, by_word, by_shape) -> list:
    """The course's own Exercise 3: from left to right, the tag of (word, tag before) in by_word, else the tag of
    (shape, tag before) in by_shape, else O; the tag before the first word is O."""
    tags = []
    before = START
    for word in words:
        if (word, before) in by_word:
            tag = by_word[(word, before)]
        elif (shape(word), before) in by_shape:
            tag = by_shape[(shape(word), before)]
        else:
            tag = "O"
        tags.append(tag)
        before = tag
    return tags


def source_of(words, by_word, by_shape) -> list:
    """For each word, which table gave its tag in tag_sentence: "word", "shape" or "none"."""
    out, before = [], START
    for word, tag in zip(words, tag_sentence(words, by_word, by_shape)):
        out.append("word" if (word, before) in by_word else "shape" if (shape(word), before) in by_shape else "none")
        before = tag
    return out


# --- the numbers of the lesson ---------------------------------------------------------------------------------------

def overlaps(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def score(found_per_sentence, sentences=TEST) -> dict:
    """Spans found against the spans of Universal NER: right (same start, end and type), wrong, missed; the wrong ones
    in three groups (the same words with another type, overlapping a name, no name at all), and the right ones whose
    words the gazetteer does not have."""
    out = {"names": 0, "found": 0, "right": 0, "type": 0, "part": 0, "none": 0, "new": 0}
    for found, sentence in zip(found_per_sentence, sentences):
        words = words_of(sentence)
        gold = spans(tags_of(sentence))
        out["names"] += len(gold)
        for span in set(found):
            out["found"] += 1
            if span in gold:
                out["right"] += 1
                out["new"] += tuple(words[span[0]:span[1]]) not in GAZETTEER
            elif any(g[:2] == span[:2] for g in gold):
                out["type"] += 1
            elif any(overlaps(g, span) for g in gold):
                out["part"] += 1
            else:
                out["none"] += 1
    out["wrong"] = out["found"] - out["right"]
    out["missed"] = out["names"] - out["right"]
    return out


def numbers() -> dict:
    """The numbers of the text: the data, the gazetteer and the tagger on the test set."""
    def _go():
        test_names = [(words_of(s), sp) for s in TEST for sp in spans(tags_of(s))]
        known = sum(tuple(w[a:b]) in GAZETTEER for w, (a, b, k) in test_names)
        same_type = sum(GAZETTEER.get(tuple(w[a:b])) == k for w, (a, b, k) in test_names)
        return {"train_sentences": len(TRAIN), "test_sentences": len(TEST),
                "train_names": sum(len(spans(tags_of(s))) for s in TRAIN), "test_names": len(test_names),
                "test_words": sum(len(s) for s in TEST),
                "gazetteer": len(GAZETTEER), "longest": max(len(n) for n in GAZETTEER),
                "more_types": sum(len(c) > 1 for c in NAME_COUNTS.values()),
                "known": known, "known_same_type": same_type,
                "list": score([find_names(words_of(s), GAZETTEER) for s in TEST]),
                "one_word": score([find_names(words_of(s), GAZETTEER, one_word_at) for s in TEST]),
                "tagger": score([spans(tag_sentence(words_of(s), BY_WORD, BY_SHAPE)) for s in TEST]),
                "by_word": len(BY_WORD), "by_shape": len(BY_SHAPE),
                "train_types": len({w for s in TRAIN for w, _ in s}),
                "shapes": len({shape(w) for s in TRAIN for w, _ in s})}
    return kept("nlp-named-entities:numbers", _go)


def propn_numbers() -> dict:
    """Names and proper nouns: in the sentences that the treebank of the lesson about tagging word classes and
    Universal NER cut into the same words, how many words of names are PROPN, and how many PROPN words stand outside
    every name, with the most frequent of those."""
    def _go():
        tags, sent_id, words = {}, None, []
        with gzip.open(UD_EN, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("# sent_id = "):
                    sent_id = line[len("# sent_id = "):]
                elif not line:
                    if words:
                        tags[sent_id] = words
                    words = []
                elif not line.startswith("#"):
                    cols = line.split("\t")
                    if cols[0].isdigit():
                        words.append((cols[1], cols[3]))
        out = {"sentences": 0, "in_names": 0, "in_names_propn": 0, "propn": 0, "propn_outside": 0}
        outside = {}
        for s in uner.sentences():
            ud = tags.get(s.sent_id)
            if ud is None or words_of(ud) != words_of(s.pairs):
                continue
            out["sentences"] += 1
            for (word, ner), (_, pos) in zip(s.pairs, ud):
                out["in_names"] += ner != "O"
                out["in_names_propn"] += ner != "O" and pos == "PROPN"
                out["propn"] += pos == "PROPN"
                if ner == "O" and pos == "PROPN":
                    out["propn_outside"] += 1
                    outside[word] = outside.get(word, 0) + 1
        out["outside"] = outside
        return out
    return kept("nlp-named-entities:propn", _go)


# --- pictures: sentences with their names ----------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_HEAD_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);font-weight:600;margin-top:0.6rem"
_CODE = "font-family:var(--ea-font-code);font-size:0.78rem"
_LINE = "font-family:var(--ea-font-code);font-size:0.85rem;line-height:2.1;color:var(--ea-ink)"
_KIND = "font-family:var(--ea-font-code);font-size:0.62rem;vertical-align:sub;margin-left:2px"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:1px;border-radius:3px"
_PART = "display:inline-block;margin:0 0.2rem 0.3rem 0;text-align:center;vertical-align:top"
_WORD = ("display:block;border-radius:5px;padding:0.15rem 0.4rem;font-family:var(--ea-font-code);font-size:0.9rem;"
         "color:var(--ea-ink)")
_WORD_BORDER = {"word": "border:1px solid var(--ea-line)", "shape": "border:1.5px dotted var(--ea-ink-3)",
                "none": "border:1.5px dotted var(--ea-ink-3)"}
_TAG = "display:block;font-family:var(--ea-font-code);font-size:0.7rem;margin-top:0.15rem;padding:0 0.15rem"
_ROLE = {   # the line under a span and the color of its type: the data's, the course's, the learner's, or wrong
    "data": ("var(--ea-ink-3)", "var(--ea-ink-2)"),
    "course": ("var(--ea-ink)", "var(--ea-ink)"),
    "mine": ("var(--ea-pred)", "var(--ea-pred-text)"),
    "off": ("var(--ea-loss)", "var(--ea-loss-text)"),
}
_CELL = ("padding:0.15rem 0.5rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.8rem;color:var(--ea-ink-2);vertical-align:top")
_HEAD_CELL = _CELL + ";color:var(--ea-ink-2);font-weight:600"


def _marked(text: str) -> str:
    """Escaped text in which `name` is set as code and *word* in italics."""
    out = []
    for k, part in enumerate(html.escape(text).split("`")):
        if k % 2:
            out.append(f"<code>{part}</code>")
        else:
            out.append("".join(f"<i>{p}</i>" if j % 2 else p for j, p in enumerate(part.split("*"))))
    return "".join(out)


def _bare(text: str) -> str:
    return text.replace("*", "").replace("`", "")


def _said(template: str, **values) -> tuple:
    """A note as HTML and as plain text, with the values filled in after the marking: a * or a ` in a value of the
    learner's stays as it is."""
    out, plain = _marked(template), _bare(template)
    for key, value in values.items():
        out = out.replace("{" + key + "}", html.escape(str(value)))
        plain = plain.replace("{" + key + "}", str(value))
    return out, plain


def _note(text: str) -> str:
    return f'<div style="{_UI}">{_marked(text)}</div>'


def _head(text: str) -> str:
    return f'<div style="{_HEAD_UI}">{_marked(text)}</div>'


def _n(x: int, lang: str) -> str:
    """A whole number with a separator for thousands: 18,990 in English, 18.990 in Dutch."""
    text = f"{x:,}"
    return text.replace(",", ".") if lang == "nl" else text


def _repr(v) -> str:
    """How Python shows a value, also for a value whose repr stops with an error."""
    try:
        return repr(v)
    except Exception:   # noqa: BLE001   a learner's value can be anything
        return "<" + type(v).__name__ + ">"


def _short(v, limit: int = 60) -> str:
    text = _repr(v)
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _shown(v, limit: int = 12) -> str:
    """A tag or a shape as text, short: anything that is not a string is shown as Python shows it."""
    text = v if isinstance(v, str) else _repr(v)
    return text if len(text) <= limit else text[:limit - 1] + "…"


def inline(words, spans_, roles=None) -> str:
    """A sentence as one line of text, with a line under each span and its type after it; roles gives per span
    "data", "course", "mine" or "off". Spans that overlap or lie outside the words are left out of the drawing (the
    notes under a picture name them)."""
    marks, taken = {}, set()
    for k, (start, end, kind) in enumerate(spans_):
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
            parts.append((f'<span style="border-bottom:2px {style} {line};padding-bottom:1px">'
                          f'{html.escape(" ".join(words[i:end]))}</span>'
                          f'<span style="{_KIND};color:{color}">{html.escape(_shown(kind))}</span>', False))
            i = end
        else:
            # a token of punctuation keeps to the word before it, so a line never starts with a lone full stop
            parts.append((html.escape(words[i]), not any(ch.isalnum() for ch in words[i])))
            i += 1
    line_ = "".join((("&nbsp;" if glue else " ") if k else "") + part for k, (part, glue) in enumerate(parts))
    return f'<div style="{_LINE}">{line_}</div>'


def _plain(words, spans_) -> str:
    """A sentence with its spans between brackets, for the text of a picture."""
    out, i, by_start = [], 0, {s[0]: s for s in spans_ if 0 <= s[0] < s[1] <= len(words)}
    while i < len(words):
        if i in by_start:
            start, end, kind = by_start[i]
            out.append(f"[{_shown(kind)} {' '.join(words[start:end])}]")
            i = end
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


def _figure(title: str, rows: list, said: list, notes: list) -> str:
    body = "".join(rows) + f'<div style="{_UI}">{" ".join(h for h, _ in notes)}</div>'
    return yours.figure(title, body, alt=f'{title}. {"; ".join(said)}. {" ".join(p for _, p in notes)}')


_IDEA_WORDS = {
    "en": {"title": "The names of one sentence of the test set, three ways",
           "gold": "In Universal NER: the names people marked",
           "list": "The gazetteer, the longest name first",
           "tagger": "The tagger with the tag before",
           "legend": "A line under one or more words marks a name, with its type after it: PER a person, LOC a place, "
                     "ORG an organization. Gray: the names people marked. A line in the color of the text: a name "
                     "found with the same start, end and type. Purple dashed: a find that differs from the names "
                     "people marked.",
           "count": "The gazetteer gets {list} of the {n} names right, and the tagger {tagger}."},
    "nl": {"title": "De namen van één zin uit de test set, op drie manieren",
           "gold": "In Universal NER: de namen die mensen markeerden",
           "list": "De gazetteer, de langste naam eerst",
           "tagger": "De tagger met de tag ervoor",
           "legend": "Een lijn onder een of meer woorden markeert een naam, met het type erachter: PER een persoon, "
                     "LOC een plaats, ORG een organisatie. Grijs: de namen die mensen markeerden. Een lijn in de kleur "
                     "van de tekst: een gevonden naam met dezelfde start, end en hetzelfde type. Paars gestreept: een "
                     "vondst die afwijkt van de namen die mensen markeerden.",
           "count": "De gazetteer heeft {list} van de {n} namen goed, en de tagger {tagger}."},
}


def _roles(found, gold, good: str) -> list:
    return [good if s in gold else "off" for s in found]


def idea_picture(lang: str) -> str:
    """The example sentence with the names of Universal NER, of the gazetteer and of the tagger."""
    w = _IDEA_WORDS[lang]
    words, gold = words_of(EXAMPLE), spans(tags_of(EXAMPLE))
    listed = find_names(words, GAZETTEER)
    tagged = spans(tag_sentence(words, BY_WORD, BY_SHAPE))
    rows = [_head(w["gold"]) + inline(words, gold),
            _head(w["list"]) + inline(words, listed, _roles(listed, gold, "course")),
            _head(w["tagger"]) + inline(words, tagged, _roles(tagged, gold, "course"))]
    said = [f'{_bare(w[k])}: {_plain(words, s)}' for k, s in (("gold", gold), ("list", listed), ("tagger", tagged))]
    notes = [_said(w["legend"]),
             _said(w["count"], list=sum(s in gold for s in listed), tagger=sum(s in gold for s in tagged), n=len(gold))]
    return _figure(w["title"], rows, said, notes)


# the rows of the table of word shapes in the second picture of the idea
SHAPE_ROWS = [("Xx", "O"), ("Xx", "B-PER"), ("Xx", "B-LOC"), ("X.X.", "O"), ("d", "O")]
SHAPE_EXAMPLES = ["Villanueva", "U.S.", "2026"]
_SHAPES_WORDS = {
    "en": {"title": "A few rows of the table of word shapes",
           "examples": "Short word shapes: {examples}.",
           "shape": "word shape", "before": "tag before", "tags": "tags of such words in the training set",
           "gives": "the table gives",
           "note": "A word with a capital and then small letters, `Xx`, is not a name most of the time after `O`; right "
                   "after the first word of a name, it is the next word of that name almost every time. The table has "
                   "{rows} rows, one for each pair of a word shape and a tag before that the training set has. Gray: the "
                   "counts in the training set."},
    "nl": {"title": "Een paar rijen van de tabel van word shapes",
           "examples": "Short word shapes: {examples}.",
           "shape": "word shape", "before": "tag ervoor", "tags": "tags van zulke woorden in de training set",
           "gives": "de tabel geeft",
           "note": "Een woord met een hoofdletter en dan kleine letters, `Xx`, is na `O` meestal geen naam; direct na "
                   "het eerste woord van een naam is het bijna altijd het volgende woord van diezelfde naam. De tabel "
                   "heeft {rows} rijen, één voor elk paar van een short word shape en een tag ervoor dat in de training set "
                   "voorkomt. Grijs: de aantallen in de training set."},
}


def _counts_text(counts: dict, lang: str) -> str:
    order = sorted(counts, key=lambda t: -counts[t])
    return " · ".join(f"{t} {_n(counts[t], lang)}" for t in order)


def shapes_picture(lang: str) -> str:
    """Short word shapes of three words, and a few rows of the tagger's table of word shapes with their counts."""
    w = _SHAPES_WORDS[lang]
    examples = " · ".join(f"{x} → {shape(x)}" for x in SHAPE_EXAMPLES)
    head = "<tr>" + "".join(f'<th style="{_HEAD_CELL}">{_marked(w[k])}</th>'
                            for k in ("shape", "before", "tags", "gives")) + "</tr>"
    rows, said = [], []
    for key in SHAPE_ROWS:
        counts = SHAPE_COUNTS[key]
        rows.append(f'<tr><td style="{_CELL};{_CODE}">{html.escape(key[0])}</td>'
                    f'<td style="{_CELL};{_CODE}">{html.escape(key[1])}</td>'
                    f'<td style="{_CELL};{_CODE};color:var(--ea-ink-3)">{html.escape(_counts_text(counts, lang))}</td>'
                    f'<td style="{_CELL};{_CODE};color:var(--ea-ink)">{html.escape(BY_SHAPE[key])}</td></tr>')
        said.append(f"{key[0]} {key[1]}: {_counts_text(counts, lang)} → {BY_SHAPE[key]}")
    table = f'<table style="border-collapse:collapse;margin-top:0.3rem">{head}{"".join(rows)}</table>'
    top = _said(w["examples"], examples=examples)
    note = _said(w["note"], rows=_n(len(BY_SHAPE), lang))
    body = f'<div style="{_UI};color:var(--ea-ink-2)">{top[0]}</div>' + table + f'<div style="{_UI}">{note[0]}</div>'
    return yours.figure(w["title"], body, alt=f'{w["title"]}. {top[1]} {"; ".join(said)}. {note[1]}')


# --- the demo --------------------------------------------------------------------------------------------------------

DEMO_MAX = 30   # words the demo reads
DEMO_TEXT = "Jane Villanueva of United Airlines flew to San Diego ."
_DEMO_WORDS = {
    "en": {"title": "The gazetteer and the tagger on your sentence",
           "list": "The gazetteer, the longest name first",
           "tagger": "The tagger with the tag before",
           "tags": "The tags of the tagger, word by word",
           "legend": "A line under one or more words marks a name, with its type after it: PER a person, LOC a place, "
                     "ORG an organization.",
           "boxes": "Under each word its tag. A dotted box: the table of words does not have the word after that tag "
                    "before, so the tag comes from its word shape, under the tag; when neither table has its pair, the tag is "
                    "O.",
           "none": "Neither finds a name.",
           "cut": "The demo reads the first {max} words.",
           "empty": "Type a sentence, with spaces between the words and before punctuation."},
    "nl": {"title": "De gazetteer en de tagger op jouw zin",
           "list": "De gazetteer, de langste naam eerst",
           "tagger": "De tagger met de tag ervoor",
           "tags": "De tags van de tagger, woord voor woord",
           "legend": "Een lijn onder een of meer woorden markeert een naam, met het type erachter: PER een persoon, "
                     "LOC een plaats, ORG een organisatie.",
           "boxes": "Onder elk woord de tag. Een gestippeld vak: de tabel van woorden heeft het woord niet na die tag "
                    "ervoor, dus de tag komt van de word shape, onder de tag; heeft ook de tabel van word shapes het paar niet, "
                    "dan is de tag O.",
           "none": "Geen van beide vindt een naam.",
           "cut": "De demo leest de eerste {max} woorden.",
           "empty": "Typ een zin, met spaties tussen de woorden en vóór leestekens."},
}


def _tag_column(word: str, tag: str, source: str, role: str = "course") -> str:
    """One word in its box with its tag under it, and its shape under that when the tag came from the shape table;
    dotted when the table of words did not have the word after that tag before."""
    color = _ROLE[role][1]
    off = ";" + _OFF if role == "off" else ""
    lines = f'<span style="{_TAG};color:{color}{off}">{html.escape(_shown(tag))}</span>'
    if source == "shape":
        lines += f'<span style="{_TAG};color:var(--ea-ink-3)">{html.escape(_shown(shape(word)))}</span>'
    return (f'<span style="{_PART}"><span style="{_WORD};{_WORD_BORDER[source]}">{html.escape(word)}</span>'
            f'{lines}</span>')


def demo_picture(text, lang: str) -> str:
    """The gazetteer and the tagger on a sentence the learner types, cut at the spaces."""
    w = _DEMO_WORDS[lang]
    words = str(text or "").split()
    if not words:
        return yours.figure(w["title"], _note(w["empty"]), alt=f'{w["title"]}. {w["empty"]}')
    notes = []
    if len(words) > DEMO_MAX:
        words = words[:DEMO_MAX]
        notes.append(_said(w["cut"], max=DEMO_MAX))
    listed = find_names(words, GAZETTEER)
    tags = tag_sentence(words, BY_WORD, BY_SHAPE)
    tagged = spans(tags)
    sources = source_of(words, BY_WORD, BY_SHAPE)
    columns = "".join(_tag_column(x, t, s) for x, t, s in zip(words, tags, sources))
    rows = [_head(w["list"]) + inline(words, listed, ["course"] * len(listed)),
            _head(w["tagger"]) + inline(words, tagged, ["course"] * len(tagged)),
            _head(w["tags"]) + f"<div>{columns}</div>"]
    said = [f'{_bare(w["list"])}: {_plain(words, listed)}', f'{_bare(w["tagger"])}: {_plain(words, tagged)}',
            " ".join(f"{x}/{t}" for x, t in zip(words, tags))]
    notes[:0] = [_said(w["legend"]), _said(w["boxes"])] + ([_said(w["none"])] if not listed and not tagged else [])
    return _figure(w["title"], rows, said, notes)


# --- running the learner's function in a picture ---------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _quiet gives for a function that stopped with an error
_TOO_SLOW = object()  # what _timed gives when the learner's function takes too long; None is a value of theirs
PICTURE_SECONDS = 2   # what the learner's function gets for the whole test set of a picture


def _quiet(fn, args):
    """The learner's function without what it prints; _STOPPED when it stops with an error."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*args)
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _timed(make):
    """make() within PICTURE_SECONDS, or _TOO_SLOW when it takes longer."""
    try:
        return with_time_limit(make, PICTURE_SECONDS)
    except Timeout:
        return _TOO_SLOW


_SLOW = {"en": "Your function takes too long on the whole test set. Does a loop of yours go on longer than it needs to?",
         "nl": "Je functie duurt te lang op de hele test set. Gaat een van je loops langer door dan nodig is?"}
_SLOW_TRAIN = {"en": "Your function takes too long on all the words of the training set. Does a loop of yours go on longer than it needs to?",
               "nl": "Je functie duurt te lang op alle woorden van de training set. Gaat een van je loops langer door dan nodig is?"}
_STOP = {"en": "Your function stops with an error on {k} sentences of the test set.",
         "nl": "Je functie stopt met een fout op {k} zinnen uit de test set."}


def as_span(v, i: int, n: int):
    """What the learner's longest_at gives at position i of n words, as a span, None for no name, or _STOPPED when it
    is neither: a span starts at i and ends after it, at n at most."""
    if v is None:
        return None
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        return _STOPPED
    start, end, kind = v
    if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
        return _STOPPED
    if start != i or not i < end <= n or not isinstance(kind, str):
        return _STOPPED
    return (start, end, kind)


def sweep(fn, words, gazetteer):
    """The course's find_names with the learner's longest_at: the spans, and the first odd value with its position,
    or None. A function that stops with an error gives _STOPPED."""
    found, free, odd = [], 0, None
    for i in range(len(words)):
        if i >= free:
            got = _quiet(fn, (list(words), i, gazetteer))
            if got is _STOPPED:
                return _STOPPED, None
            span = as_span(got, i, len(words))
            if span is _STOPPED:
                odd = odd or (i, got)
                continue
            if span is not None:
                found.append(span)
                free = span[1]
    return found, odd


_LONGEST_WORDS = {
    "en": {"title": "Your longest_at in find_names, on the test set",
           "gold": "In Universal NER",
           "mine": "With your longest_at",
           "gray": "Gray: the names people marked in Universal NER.",
           "blue": "Blue: a name your function finds with the same start, end and type as in Universal NER.",
           "off": "Purple dashed: a find that differs.",
           "total": "On the whole test set, the gazetteer with your longest_at: names right {right} of the {n}, wrong "
                    "finds {wrong}, names missed {missed}. With one word at a time, as in the worked example: names "
                    "right {one}.",
           "kinds": "The wrong finds: part of a longer name or overlapping one {part}, no name at all {none}, the "
                    "words of a name with another type {type}.",
           "odd": "At position {i} of *{sentence}*, your function gives {got}. A span is a tuple (start, end, kind) "
                  "that starts at {i}, with an end after it and no further than {n}, or None for no name.",
           "differs": "Your names differ from those of the course's longest_at in {k} sentences of the test set.",
           "case": "In the check's case `{call}`, your function gives {got}, and the course's {want}."},
    "nl": {"title": "Jouw longest_at in find_names, op de test set",
           "gold": "In Universal NER",
           "mine": "Met jouw longest_at",
           "gray": "Grijs: de namen die mensen markeerden in Universal NER.",
           "blue": "Blauw: een naam die je functie vindt met dezelfde start, end en hetzelfde type als in Universal NER.",
           "off": "Paars gestreept: een vondst die afwijkt.",
           "total": "Op de hele test set, de gazetteer met jouw longest_at: namen goed {right} van de {n}, foute "
                    "vondsten {wrong}, namen gemist {missed}. Met één woord tegelijk, zoals in het uitgewerkte "
                    "voorbeeld: namen goed {one}.",
           "kinds": "De foute vondsten: deel van een langere naam of overlappend ermee {part}, helemaal geen naam "
                    "{none}, de woorden van een naam met een ander type {type}.",
           "odd": "Op positie {i} van *{sentence}* geeft je functie {got}. Een span is een tuple (start, end, kind) "
                  "met start {i}, een end daarna en niet verder dan {n}, of None voor geen naam.",
           "differs": "Je namen verschillen in {k} zinnen uit de test set van de namen van de longest_at van de course.",
           "case": "In het geval `{call}` van de check geeft je functie {got}, en de functie van de course {want}."},
}


def longest_picture(seen) -> str:
    """The learner's longest_at in the course's find_names: two sentences of the test set, and the names found right,
    wrong and missed on the whole test set."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _LONGEST_WORDS[lang]
    got = _timed(lambda: [sweep(fn, words_of(s), GAZETTEER) for s in TEST])
    if got is _TOO_SLOW:
        return _figure(w["title"], [], [], [_said(_SLOW[lang])])
    rows, said, notes, off, odd = [], [], [], False, None
    for sentence in (UN, KOREA):
        found, odd_here = sweep(fn, words_of(sentence), GAZETTEER)
        odd = odd or (odd_here and (sentence, odd_here))
        if found is _STOPPED:
            continue
        words, gold = words_of(sentence), spans(tags_of(sentence))
        roles = _roles(found, gold, "mine")
        off = off or "off" in roles
        rows.append(_head(w["gold"]) + inline(words, gold) + _head(w["mine"]) + inline(words, found, roles))
        said.append(f'{_plain(words, gold)} / {_plain(words, found)}')
    stopped = sum(f is _STOPPED for f, _ in got)
    if stopped:
        notes.append(_said(_STOP[lang], k=_n(stopped, lang)))
    if odd is None:
        odd = next(((s, o) for s, (f, o) in zip(TEST, got) if o is not None), None)
    if odd:
        sentence, (i, value) = odd
        notes.append(_said(w["odd"], i=i, sentence=" ".join(words_of(sentence)), got=_short(value),
                           n=len(sentence)))
    found_all = [f if f is not _STOPPED else [] for f, _ in got]
    sc, nums = score(found_all), numbers()
    notes.append(_said(w["total"], right=_n(sc["right"], lang), n=_n(sc["names"], lang),
                       wrong=_n(sc["wrong"], lang), missed=_n(sc["missed"], lang),
                       one=_n(nums["one_word"]["right"], lang)))
    if sc["wrong"]:
        notes.append(_said(w["kinds"], part=_n(sc["part"], lang),
                           none=_n(sc["none"], lang), type=_n(sc["type"], lang)))
    differs = sum(f != find_names(words_of(s), GAZETTEER) for f, s in zip(found_all, TEST))
    if differs:
        notes.append(_said(w["differs"], k=_n(differs, lang)))
    if differs or stopped or odd:
        for case in LA_CASES[:7]:
            args = [a.value if isinstance(a, Named) else a for a in case]
            mine = _timed(lambda: _quiet(fn, (list(args[0]), args[1], dict(args[2]))))
            want = longest_at(*args)
            if mine is not _TOO_SLOW and mine is not _STOPPED and not _same_span(mine, want):
                call = "longest_at(" + ", ".join(_case_code(a) for a in case) + ")"
                notes.append(_said(w["case"], call=call, got=_short(mine), want=_short(want)))
                break
    extra = ([_said(w["gray"]), _said(w["blue"])] if rows else []) + ([_said(w["off"])] if off else [])
    return _figure(w["title"], rows, said, extra + notes)


def _same_span(a, b) -> bool:
    if a is _STOPPED:
        return False
    return (tuple(a) if isinstance(a, list) else a) == b


# exercise 2: the words the picture shows with their shapes
SHAPE_WORDS = ["Villanueva", "L'Occitane", "I.M.F.", "DC10-30", "U.S.", "iPhone", "2026", "us", "McDonald's", "..."]
_SHAPE_WORDS = {
    "en": {"title": "Your shape on words of the training set",
           "word": "word", "yours": "your shape", "course": "the course's shape",
           "gray": "Gray: the words. In the color of the text: the course's shape.",
           "blue": "Blue: your shape, the same as the course's.",
           "off": "Purple dashed: a shape of yours that differs from the course's.",
           "total": "Of the {n} different words of the training set, your function gives {same} the course's shape. "
                    "Those words have {shapes} different shapes in all.",
           "stopped": "Your function stops with an error on {k} words of the training set, such as *{word}*."},
    "nl": {"title": "Jouw shape op woorden uit de training set",
           "word": "woord", "yours": "jouw word shape", "course": "word shape van de course",
           "gray": "Grijs: de woorden. In de kleur van de tekst: de word shape van de course.",
           "blue": "Blauw: jouw word shape, gelijk aan de word shape van de course.",
           "off": "Paars gestreept: een word shape van jou die afwijkt van de word shape van de course.",
           "total": "Van de {n} verschillende woorden in de training set geeft je functie er {same} de word shape van de "
                    "course. Die woorden hebben samen {shapes} verschillende short word shapes.",
           "stopped": "Je functie stopt met een fout bij {k} woorden uit de training set, zoals *{word}*."},
}


def shape_picture(seen) -> str:
    """The learner's shape on ten words beside the course's, and how many words of the training set get the course's
    shape."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _SHAPE_WORDS[lang]
    head = "<tr>" + "".join(f'<th style="{_HEAD_CELL}">{_marked(w[k])}</th>' for k in ("word", "yours", "course")) \
        + "</tr>"
    rows, said, off = [], [], False
    for word in SHAPE_WORDS:
        got = _timed(lambda: _quiet(fn, (word,)))
        want = shape(word)
        if got is _TOO_SLOW:
            return _figure(w["title"], [], [], [_said(_SLOW[lang])])
        fine = got == want
        off = off or not fine
        text = "…" if got is _STOPPED else _shown(got, 16)
        style = "color:var(--ea-pred-text)" if fine else f"color:var(--ea-loss-text);{_OFF}"
        rows.append(f'<tr><td style="{_CELL};{_CODE};color:var(--ea-ink-3)">{html.escape(word)}</td>'
                    f'<td style="{_CELL};{_CODE}"><span style="{style}">{html.escape(text)}</span></td>'
                    f'<td style="{_CELL};{_CODE};color:var(--ea-ink)">{html.escape(want)}</td></tr>')
        said.append(f"{word}: {text} / {want}")
    table = f'<table style="border-collapse:collapse;margin-top:0.3rem">{head}{"".join(rows)}</table>'
    words = sorted({x for s in TRAIN for x, _ in s})
    mine = _timed(lambda: [_quiet(fn, (x,)) for x in words])
    notes = []
    if mine is _TOO_SLOW:
        notes.append(_said(_SLOW_TRAIN[lang]))
    else:
        stopped = [x for x, g in zip(words, mine) if g is _STOPPED]
        if stopped:
            notes.append(_said(w["stopped"], k=_n(len(stopped), lang), word=stopped[0]))
        same = sum(g == shape(x) for x, g in zip(words, mine))
        notes.append(_said(w["total"], n=_n(len(words), lang), same=_n(same, lang),
                           shapes=_n(numbers()["shapes"], lang)))
    extra = [_said(w["gray"]), _said(w["blue"])] + ([_said(w["off"])] if off else [])
    return _figure(w["title"], [table], said, extra + notes)


# exercise 3
_TAGGER_WORDS = {
    "en": {"title": "Your tag_sentence on the test set",
           "gold": "In Universal NER",
           "mine": "The names your tags mark",
           "gray": "Gray: the names people marked in Universal NER.",
           "blue": "Blue: a name your tags mark with the same start, end and type as in Universal NER.",
           "off": "Purple dashed: a name your tags mark that differs.",
           "length": "For a sentence of {n} words, your function gives back {got} instead of a list of {n} tags.",
           "total": "On the whole test set, the names your tags mark: right {right} of the {n}, wrong {wrong}, "
                    "missed {missed}; right names that the gazetteer does not have {new}. The gazetteer: right "
                    "{list}, wrong {list_wrong}.",
           "differs": "Your tags differ from those of the course's tagger in {k} sentences of the test set.",
           "case": "In the check's case `{call}`, your function gives {got}, and the course's tagger {want}."},
    "nl": {"title": "Jouw tag_sentence op de test set",
           "gold": "In Universal NER",
           "mine": "De namen die je tags markeren",
           "gray": "Grijs: de namen die mensen markeerden in Universal NER.",
           "blue": "Blauw: een naam die je tags markeren met dezelfde start, end en hetzelfde type als in Universal NER.",
           "off": "Paars gestreept: een naam die je tags markeren en die afwijkt.",
           "length": "Voor een zin van {n} woorden geeft je functie {got} terug in plaats van een lijst van {n} tags.",
           "total": "Op de hele test set, de namen die je tags markeren: goed {right} van de {n}, fout {wrong}, "
                    "gemist {missed}; goede namen die niet in de gazetteer staan {new}. De gazetteer: goed {list}, "
                    "fout {list_wrong}.",
           "differs": "Je tags verschillen in {k} zinnen uit de test set van de tags van de tagger van de course.",
           "case": "In het geval `{call}` van de check geeft je functie {got}, en de tagger van de course {want}."},
}


def _listed(v):
    """What tag_sentence gives, with a tuple of tags made a list, as the picture under the check reads it."""
    return list(v) if isinstance(v, tuple) else v


def _as_tags(v, n: int):
    """A list of n tags as spans can read them, or None: a tag that is not a string counts as O."""
    if not isinstance(v, (list, tuple)) or len(v) != n:
        return None
    return [t if isinstance(t, str) and (t == "O" or t[:2] in ("B-", "I-")) else "O" for t in v]


def tagger_picture(seen) -> str:
    """The learner's tag_sentence on two sentences of the test set, as names beside those of Universal NER, and the
    names right, wrong and missed on the whole test set."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _TAGGER_WORDS[lang]
    got = _timed(lambda: [_quiet(fn, (words_of(s), BY_WORD, BY_SHAPE)) for s in TEST])
    if got is _TOO_SLOW:
        return _figure(w["title"], [], [], [_said(_SLOW[lang])])
    rows, said, notes, off = [], [], [], False
    for sentence in (PENTAGON, PLENTY):
        words, gold = words_of(sentence), spans(tags_of(sentence))
        mine = _quiet(fn, (list(words), BY_WORD, BY_SHAPE))
        tags = _as_tags(mine, len(words)) if mine is not _STOPPED else None
        if tags is None:
            if mine is not _STOPPED:
                note = _said(w["length"], n=len(words), got=_short(mine))
                if note not in notes:
                    notes.append(note)
            continue
        found = spans(tags)
        roles = _roles(found, gold, "mine")
        off = off or "off" in roles
        rows.append(_head(w["gold"]) + inline(words, gold) + _head(w["mine"]) + inline(words, found, roles))
        said.append(f'{_plain(words, gold)} / {_plain(words, found)}')
    stopped = sum(g is _STOPPED for g in got)
    if stopped:
        notes.append(_said(_STOP[lang], k=_n(stopped, lang)))
    found_all = []
    for g, s in zip(got, TEST):
        tags = _as_tags(g, len(s)) if g is not _STOPPED else None
        found_all.append(spans(tags) if tags is not None else [])
    sc, nums = score(found_all), numbers()
    notes.append(_said(w["total"], right=_n(sc["right"], lang), n=_n(sc["names"], lang),
                       wrong=_n(sc["wrong"], lang), missed=_n(sc["missed"], lang), new=_n(sc["new"], lang),
                       list=_n(nums["list"]["right"], lang), list_wrong=_n(nums["list"]["wrong"], lang)))
    differs = sum(_listed(g) != tag_sentence(words_of(s), BY_WORD, BY_SHAPE) for g, s in zip(got, TEST))
    if differs:
        notes.append(_said(w["differs"], k=_n(differs, lang)))
        for case in TS_CASES[:7]:
            args = [a.value if isinstance(a, Named) else a for a in case]
            mine = _timed(lambda: _quiet(fn, (list(args[0]), dict(args[1]), dict(args[2]))))
            want = tag_sentence(*args)
            if mine is not _TOO_SLOW and mine is not _STOPPED and _listed(mine) != want:
                call = "tag_sentence(" + ", ".join(_case_code(a) for a in case) + ")"
                notes.append(_said(w["case"], call=call, got=_short(mine), want=_short(want)))
                break
    extra = ([_said(w["gray"]), _said(w["blue"])] if rows else []) + ([_said(w["off"])] if off else [])
    return _figure(w["title"], rows, said, extra + notes)


def _case_code(a) -> str:
    if isinstance(a, Named):
        return a.name
    if isinstance(a, list):
        return "[" + ", ".join(_repr(x) for x in a) + "]"
    return _repr(a)


# --- the checks ------------------------------------------------------------------------------------------------------

REQUIRED = ["longest_at", "shape", "tag_sentence", "streets"]


class Named:
    """A value the check gives the learner's function, shown in a message by its name. A tuple or a read-only table
    cannot be changed by the learner's code, so it gets the value itself; anything else gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        if isinstance(self.value, (tuple, MappingProxyType)):
            return self.value
        return copy.deepcopy(self.value, memo)


_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

def _dots_left(got, text) -> bool:
    """A `...` of the start code left in the box: an Ellipsis in the code, not three dots inside a string."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None
    return any(isinstance(n, ast.Constant) and n.value is Ellipsis for n in ast.walk(tree))


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang de `...` door je eigen code.")


def _key_error(e) -> bool:
    return isinstance(e, KeyError)


def _unhashable(e) -> bool:
    return isinstance(e, TypeError) and "unhashable" in str(e)


def _unbound(e) -> bool:
    return isinstance(e, UnboundLocalError)


def _not_span(got, text) -> bool:
    return got is not None and not (isinstance(got, (list, tuple)) and len(got) == 3)


def _not_str(got, text) -> bool:
    return got is not None and not isinstance(got, str)


def _not_list(got, text) -> bool:
    return got is not None and not isinstance(got, (list, tuple))


def _as_tuple(v):
    """What longest_at gives, with a span as a list made a tuple, as the picture under the check reads it."""
    return tuple(v) if isinstance(v, list) else v


# exercise 1: longest_at. The visible case: the United Nations Security Council, with a shorter name at the same start
# and a name inside it; then a position where no name starts, a name of one word, a name at the last word, a position
# inside a name, a name longer than the rest of the sentence, the longest of two names up to the last word, and the
# words of the worked example with the course's gazetteer.
_UN_WORDS = ["the", "United", "Nations", "Security", "Council", "met"]
_UN_GAZ = {("United", "Nations"): "ORG", ("United", "Nations", "Security", "Council"): "ORG", ("Council",): "ORG"}
_UN_NAMED = Named('{("United", "Nations"): "ORG", ("United", "Nations", "Security", "Council"): "ORG", '
                  '("Council",): "ORG"}', _UN_GAZ)
_WORDS = Named("words", words_of(EXAMPLE))
_GAZ = Named("C.GAZETTEER", GAZETTEER)
LA_CASES = [
    (_UN_WORDS, 1, _UN_NAMED),
    (_UN_WORDS, 0, _UN_NAMED),
    (_UN_WORDS, 4, _UN_NAMED),
    (["in", "Chicago"], 1, Named('{("Chicago",): "LOC"}', {("Chicago",): "LOC"})),
    (["New", "York"], 1, Named('{("New", "York"): "LOC"}', {("New", "York"): "LOC"})),
    (["New"], 0, Named('{("New", "York"): "LOC"}', {("New", "York"): "LOC"})),
    (["to", "New", "York"], 1, Named('{("New",): "ORG", ("New", "York"): "LOC"}', {("New",): "ORG", ("New", "York"): "LOC"})),
    (_WORDS, 3, _GAZ),
    (_WORDS, 18, _GAZ),
    (_WORDS, 0, _GAZ),
]
_LA_HEAD = "def longest_at(words, i, gazetteer):\n"
_LA_START = _LA_HEAD + "    ...\n"
_LA = (_LA_HEAD
       + "    for end in range(len(words), i, -1):\n"
       + "        name = tuple(words[i:end])\n"
       + "        if name in gazetteer:\n"
       + "            return (i, end, gazetteer[name])\n"
       + "    return None\n")
_LA_SHORT = _LA.replace("range(len(words), i, -1)", "range(i + 1, len(words) + 1)")
_LA_NO_LAST = _LA.replace("range(len(words), i, -1)", "range(len(words) - 1, i, -1)")
_LA_NO_ONE = _LA.replace("range(len(words), i, -1)", "range(len(words), i + 1, -1)")
_LA_PAST = _LA.replace("range(len(words), i, -1)", "range(len(words) + 1, i, -1)")
_LA_LIST = _LA.replace("name = tuple(words[i:end])", "name = words[i:end]")
_LA_INCLUSIVE = _LA.replace("return (i, end, gazetteer[name])", "return (i, end - 1, gazetteer[name])")
_LA_NO_KIND = _LA.replace("return (i, end, gazetteer[name])", "return (i, end)")
_LA_NAME = _LA.replace("return (i, end, gazetteer[name])", "return name")
_LA_EARLY = _LA.replace("            return (i, end, gazetteer[name])\n",
                        "            return (i, end, gazetteer[name])\n        else:\n            return None\n")

# exercise 2: shape. The visible case: Villanueva; then an apostrophe, dots, digits and a hyphen, a small letter first,
# all small, all capitals, only digits, no characters, three dots, a letter with an accent, and a word with two capitals.
SH_CASES = [("Villanueva",), ("L'Occitane",), ("I.M.F.",), ("DC10-30",), ("iPhone",), ("us",), ("US",), ("2026",),
            ("",), ("...",), ("café",), ("McDonald's",)]
_SH_HEAD = "def shape(word):\n"
_SH_START = _SH_HEAD + "    ...\n"
_SH = (_SH_HEAD
       + '    out = ""\n'
       + '    last = ""\n'
       + "    for ch in word:\n"
       + "        if ch.isupper():\n"
       + '            kind = "X"\n'
       + "        elif ch.islower():\n"
       + '            kind = "x"\n'
       + "        elif ch.isdigit():\n"
       + '            kind = "d"\n'
       + "        else:\n"
       + "            kind = ch\n"
       + "        if kind != last:\n"
       + "            out = out + kind\n"
       + "            last = kind\n"
       + "    return out\n")
_SH_LONG = _SH.replace("        if kind != last:\n            out = out + kind\n            last = kind\n",
                       "        out = out + kind\n")
_SH_CHAR = _SH.replace("        if kind != last:\n            out = out + kind\n            last = kind\n",
                       "        if ch != last:\n            out = out + kind\n            last = ch\n")
_SH_NO_DIGIT = _SH.replace('        elif ch.isdigit():\n            kind = "d"\n', "")
_SH_PUNCT = _SH.replace("        else:\n            kind = ch\n", '        else:\n            kind = "x"\n')
_SH_NO_ELSE = _SH.replace("        else:\n            kind = ch\n", "")
_SH_SWAP = _SH.replace('kind = "X"', 'kind = "x!"').replace('kind = "x"', 'kind = "X"').replace('kind = "x!"', 'kind = "x"')
_SH_EARLY = _SH.replace("            last = kind\n    return out\n", "            last = kind\n        return out\n")

# exercise 3: tag_sentence. The visible case: Jane Villanueva, the second word from the table of shapes after B-PER;
# then no words, an unknown word, a first word from the table of shapes, the tag before that changes the tag of a known
# word, the same word first, a name of three words, and three sentences of the test set with the course's tables.
_JANE_WORD = {("Jane", "O"): "B-PER", ("said", "I-PER"): "O", ("so", "O"): "O"}
_JANE_SHAPE = {("Xx", "B-PER"): "I-PER", ("Xx", "O"): "O"}
_YORK_WORD = {("New", "O"): "B-LOC", ("York", "B-LOC"): "I-LOC", ("York", "O"): "B-PER"}
_YORK_NAMED = Named('{("New", "O"): "B-LOC", ("York", "B-LOC"): "I-LOC", ("York", "O"): "B-PER"}', _YORK_WORD)
_THREE_SHAPE = {("Xx", "B-PER"): "I-PER", ("Xx", "I-PER"): "I-PER", ("Xx", "O"): "O"}
_BY_WORD = Named("C.BY_WORD", BY_WORD)
_BY_SHAPE = Named("C.BY_SHAPE", BY_SHAPE)
TS_CASES = [
    (["Jane", "Villanueva", "said", "so"],
     Named('{("Jane", "O"): "B-PER", ("said", "I-PER"): "O", ("so", "O"): "O"}', _JANE_WORD),
     Named('{("Xx", "B-PER"): "I-PER", ("Xx", "O"): "O"}', _JANE_SHAPE)),
    ([], {}, {}),
    (["glorp"], {}, {}),
    (["U.S.", "troops"], Named('{("troops", "B-LOC"): "O"}', {("troops", "B-LOC"): "O"}),
     Named('{("X.X.", "O"): "B-LOC"}', {("X.X.", "O"): "B-LOC"})),
    (["New", "York"], _YORK_NAMED, {}),
    (["York"], _YORK_NAMED, {}),
    (["Jane", "Ann", "Villanueva"], Named('{("Jane", "O"): "B-PER"}', {("Jane", "O"): "B-PER"}),
     Named('{("Xx", "B-PER"): "I-PER", ("Xx", "I-PER"): "I-PER", ("Xx", "O"): "O"}', _THREE_SHAPE)),
    (words_of(EXAMPLE), _BY_WORD, _BY_SHAPE),
    (words_of(PLENTY), _BY_WORD, _BY_SHAPE),
    (words_of(KOREA), _BY_WORD, _BY_SHAPE),
]
_TS_HEAD = "def tag_sentence(words, by_word, by_shape):\n"
_TS_START = _TS_HEAD + "    ...\n"
_TS = (_TS_HEAD
       + "    tags = []\n"
       + '    before = "O"\n'
       + "    for word in words:\n"
       + "        if (word, before) in by_word:\n"
       + "            tag = by_word[(word, before)]\n"
       + "        elif (shape(word), before) in by_shape:\n"
       + "            tag = by_shape[(shape(word), before)]\n"
       + "        else:\n"
       + '            tag = "O"\n'
       + "        tags.append(tag)\n"
       + "        before = tag\n"
       + "    return tags\n")
_TS_NO_BEFORE = _TS.replace("        before = tag\n", "")
_TS_WORD_BEFORE = _TS.replace("        before = tag\n", "        before = word\n")
_TS_START_NONE = _TS.replace('    before = "O"\n', "    before = None\n")
_TS_SHAPE_FIRST = (_TS_HEAD
                   + "    tags = []\n"
                   + '    before = "O"\n'
                   + "    for word in words:\n"
                   + "        if (shape(word), before) in by_shape:\n"
                   + "            tag = by_shape[(shape(word), before)]\n"
                   + "        elif (word, before) in by_word:\n"
                   + "            tag = by_word[(word, before)]\n"
                   + "        else:\n"
                   + '            tag = "O"\n'
                   + "        tags.append(tag)\n"
                   + "        before = tag\n"
                   + "    return tags\n")
_TS_NO_SHAPE = _TS.replace("        elif (shape(word), before) in by_shape:\n"
                           "            tag = by_shape[(shape(word), before)]\n", "")
_TS_NO_WORD = (_TS.replace("        if (word, before) in by_word:\n            tag = by_word[(word, before)]\n"
                           "        elif (shape(word), before) in by_shape:\n",
                           "        if (shape(word), before) in by_shape:\n"))
_TS_NO_DEFAULT = (_TS_HEAD
                  + "    tags = []\n"
                  + '    before = "O"\n'
                  + "    for word in words:\n"
                  + "        if (word, before) in by_word:\n"
                  + "            tag = by_word[(word, before)]\n"
                  + "        else:\n"
                  + "            tag = by_shape[(shape(word), before)]\n"
                  + "        tags.append(tag)\n"
                  + "        before = tag\n"
                  + "    return tags\n")

EXERCISES = {
    "longest_at": FunctionBox(
        name="longest_at",
        picture=longest_picture,
        function="longest_at",
        start=_LA_START,
        solution=_LA,
        visible=LA_CASES[:1],
        hidden=LA_CASES[1:],
        collect=_as_tuple,
        first=[_DOTS],
        mistakes=[
            Mistake("list", error=_unhashable,
                    en="With `{call}`, your function stops with a TypeError: a list cannot be a key of a dict. What type are the keys of `gazetteer`, and what type is the stretch you look up?",
                    nl="Met `{call}` stopt je functie met een TypeError: een lijst kan geen key van een dict zijn. Welk type hebben de keys van `gazetteer`, en welk type heeft het stuk dat je opzoekt?"),
            Mistake("shortest", wrong=_LA_SHORT,
                    en="`{call}` gives {got}, and it should give {want}: your function tries the shortest stretch first, so a short name wins from a longer one with the same start. In which order should the ends come?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie probeert het kortste stuk eerst, dus een korte naam wint van een langere met dezelfde start. In welke volgorde moeten de ends komen?"),
            Mistake("past-end", wrong=_LA_PAST,
                    en="`{call}` gives {got}, and it should give {want}: for a name that runs up to the last word, your end lies one beyond the words. A slice past the end stops at the last word, but the end you give back does not. Where should the ends you try begin?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: bij een naam tot en met het laatste woord ligt je end één voorbij de woorden. Een slice voorbij het eind stopt bij het laatste woord, maar het end dat je teruggeeft niet. Waar moeten de ends die je probeert beginnen?"),
            Mistake("inclusive", wrong=_LA_INCLUSIVE,
                    en="`{call}` gives {got}, and it should give {want}: your end is the position of the last word of the name. In a span, as in a slice, which position is the end?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je end is de positie van het laatste woord van de naam. Welke positie is het end in een span, net als in een slice?"),
            Mistake("no-kind", wrong=_LA_NO_KIND,
                    en="`{call}` gives {got}, and it should give {want}: your span has a start and an end, but no type. Where does the type of a name stand?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je span heeft een start en een end, maar geen type. Waar staat het type van een naam?"),
            Mistake("not-span", matches=_not_span,
                    en="`{call}` gives {got}, and it should give {want}: a span `(start, end, kind)`, or None when no name starts at that position. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een span `(start, end, kind)`, of None als er op die positie geen naam begint. Wat geeft je `return`-regel terug?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: try the stretches of words that start at position `i`, the longest first, and give back the span of the first one that is in `gazetteer`.",
                      "nl": "Vervang de `...`: probeer de stukken woorden die op positie `i` beginnen, het langste eerst, en geef de span terug van het eerste stuk dat in `gazetteer` staat."},
            "no_return": {"en": "With `{call}`, your function gives None, and the gazetteer does have a name that starts there. Which ends does your loop try, and does it get to each of them before your function gives back None?",
                          "nl": "Met `{call}` geeft je functie None, en de gazetteer heeft wel een naam die daar begint. Welke ends probeert je loop, en komt je loop bij elk ervan voordat je functie None teruggeeft?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which stretches of words that start at position `i` does your function try, and in which order?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Welke stukken woorden die op positie `i` beginnen probeert je functie, en in welke volgorde?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for a name of one word, for a name up to the last word, and for a position where no name starts?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor een naam van één woord, voor een naam tot en met het laatste woord, en voor een positie waar geen naam begint?"},
            "correct": {"en": "Your `longest_at` is right in all {n} cases. Look at the picture: the gazetteer with your function on the test set.",
                        "nl": "Je `longest_at` klopt in alle {n} gevallen. Kijk naar de afbeelding: de gazetteer met jouw functie op de test set."},
        },
    ),
    "shape": FunctionBox(
        name="shape",
        picture=shape_picture,
        function="shape",
        start=_SH_START,
        solution=_SH,
        visible=SH_CASES[:1],
        hidden=SH_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("unbound", error=_unbound,
                    en="With `{call}`, your function stops with an error: a name gets used before it has a value. Which name in the line of the error is it, and where does your function first give it a value?",
                    nl="Met `{call}` stopt je functie met een fout: een naam wordt gebruikt voordat de naam een waarde heeft. Welke naam in de regel van de fout is het, en waar geeft je functie die naam voor het eerst een waarde?"),
            Mistake("no-else", wrong=_SH_NO_ELSE,
                    en="`{call}` gives {got}, and it should give {want}: a character that none of your `if` and `elif` lines catch keeps the kind of the character before it. Which kind should such a character get?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een teken dat geen van je `if`- en `elif`-regels opvangt, houdt de soort van het teken ervoor. Welke soort moet zo'n teken krijgen?"),
            Mistake("long", wrong=_SH_LONG,
                    en="`{call}` gives {got}, and it should give {want}: every character adds its kind, so a long word gets a long shape. When should a kind be left out?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elk teken voegt de eigen soort toe, dus een lang woord krijgt een lange word shape. Wanneer moet een soort wegblijven?"),
            Mistake("char", wrong=_SH_CHAR,
                    en="`{call}` gives {got}, and it should give {want}: your function compares each character with the character before it, so only a double letter, such as the *ll* in *Villanueva*, is left out. What should a kind be compared with?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie vergelijkt elk teken met het teken ervoor, dus alleen een dubbele letter, zoals de *ll* in *Villanueva*, blijft weg. Waarmee moet je een soort vergelijken?"),
            Mistake("no-digit", wrong=_SH_NO_DIGIT,
                    en="`{call}` gives {got}, and it should give {want}: a digit stays as it is. Which kind does a digit get?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een cijfer blijft zoals het is. Welke soort krijgt een cijfer?"),
            Mistake("punct", wrong=_SH_PUNCT,
                    en="`{call}` gives {got}, and it should give {want}: a character that is not a letter or a digit becomes `x`, as a small letter does. What happens to such a character in a short word shape?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een teken dat geen letter of cijfer is wordt `x`, net als een kleine letter. Wat gebeurt er met zo'n teken in een short word shape?"),
            Mistake("swap", wrong=_SH_SWAP,
                    en="`{call}` gives {got}, and it should give {want}: your capitals are `x` and your small letters `X`. Which letter stands for a capital?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je hoofdletters zijn `x` en je kleine letters `X`. Welke letter staat voor een hoofdletter?"),
            Mistake("early", wrong=_SH_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your shape holds only the first character. When does your `return` run: after the loop over all the characters, or during the first one?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je word shape bevat alleen het eerste teken. Wanneer draait je `return`: na de loop over alle tekens, of al bij het eerste teken?"),
            Mistake("not-str", matches=_not_str,
                    en="`{call}` gives {got}, and it should give {want}: one string, the short word shape. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: één string, de short word shape. Wat geeft je `return`-regel terug?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give each character its kind, `X`, `x`, `d` or the character itself, and add a kind only when it differs from the kind before it.",
                      "nl": "Vervang de `...`: geef elk teken een soort, `X`, `x`, `d` of het teken zelf, en voeg een soort alleen toe als de soort verschilt van de soort ervoor."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which kind does each character get, and which kinds do you leave out?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Welke soort krijgt elk teken, en welke soorten laat je weg?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for digits, for punctuation, for an empty word and for a word that starts with a small letter?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor cijfers, voor leestekens, voor een leeg woord en voor een woord dat met een kleine letter begint?"},
            "correct": {"en": "Your `shape` is right in all {n} cases. Look at the picture: your shapes beside the course's, and on all the words of the training set.",
                        "nl": "Je `shape` klopt in alle {n} gevallen. Kijk naar de afbeelding: jouw word shapes naast de word shapes van de course, en op alle woorden van de training set."},
        },
    ),
    "tag_sentence": FunctionBox(
        name="tag_sentence",
        picture=tagger_picture,
        function="tag_sentence",
        start=_TS_START,
        solution=_TS,
        visible=TS_CASES[:1],
        hidden=TS_CASES[1:],
        collect=_listed,
        names={"shape": shape},
        first=[_DOTS],
        mistakes=[
            Mistake("no-default", error=_key_error,
                    en="With `{call}`, your function stops with a KeyError: it looks up a key that one of the tables does not have. Which of your lookups is it, and what does your code do when neither table has its key?",
                    nl="Met `{call}` stopt je functie met een KeyError: je functie zoekt een key op die niet in een van de tabellen staat. Bij welke van je opzoekingen gebeurt dat, en wat doet je code als geen van beide tabellen de key heeft?"),
            Mistake("no-before", wrong=_TS_NO_BEFORE,
                    en="`{call}` gives {got}, and it should give {want}: every word is looked up after the tag O, as if it stood at the start of the sentence. Which tag stands before the second word?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elk woord wordt opgezocht na de tag O, alsof het aan het begin van de zin staat. Welke tag staat er vóór het tweede woord?"),
            Mistake("word-before", wrong=_TS_WORD_BEFORE,
                    en="`{call}` gives {got}, and it should give {want}: after the first word, your lookups hold the word before in the place of the tag before. What is the second item of a key of `by_word`?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: na het eerste woord staat in je opzoekingen het woord ervoor op de plek van de tag ervoor. Wat is het tweede item van een key van `by_word`?"),
            Mistake("start", wrong=_TS_START_NONE,
                    en="`{call}` gives {got}, and it should give {want}: the first word is looked up with None as its tag before, which no key of the tables has. Which tag counts as the tag before the first word?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het eerste woord wordt opgezocht met None als tag ervoor, en dat heeft geen enkele key van de tabellen. Welke tag telt als de tag vóór het eerste woord?"),
            Mistake("shape-first", wrong=_TS_SHAPE_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: your function looks in `by_shape` first, so a word that `by_word` knows after that tag gets the tag of its shape. When the word and its shape disagree, which one should win?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie kijkt eerst in `by_shape`, dus een woord dat `by_word` na die tag kent, krijgt de tag van de word shape. Als het woord en de word shape het oneens zijn, welke moet dan winnen?"),
            Mistake("no-shape", wrong=_TS_NO_SHAPE,
                    en="`{call}` gives {got}, and it should give {want}: no word gets its tag from `by_shape`. What makes up a key of `by_shape`, and when should your function look there?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: geen enkel woord krijgt de tag uit `by_shape`. Waaruit bestaat een key van `by_shape`, en wanneer moet je functie daar kijken?"),
            Mistake("no-word", wrong=_TS_NO_WORD,
                    en="`{call}` gives {got}, and it should give {want}: no word gets its tag from `by_word`. What makes up a key of `by_word`, and does your function look there first?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: geen enkel woord krijgt de tag uit `by_word`. Waaruit bestaat een key van `by_word`, en kijkt je functie daar eerst?"),
            Mistake("not-list", matches=_not_list,
                    en="`{call}` gives {got}, and it should give {want}: a list with one tag for each word. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een lijst met één tag voor elk woord. Wat geeft je `return`-regel terug?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: go over the words from left to right, give each one the tag of `(word, before)` in `by_word`, else of `(shape(word), before)` in `by_shape`, else O, and keep the tag you gave as the tag before the next word.",
                      "nl": "Vervang de `...`: ga de woorden van links naar rechts langs, geef elk woord de tag van `(word, before)` in `by_word`, anders van `(shape(word), before)` in `by_shape`, en anders O, en bewaar de gegeven tag als de tag vóór het volgende woord."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. For each word: which tag stands before it, and which table has a key for it?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Voor elk woord: welke tag staat ervoor, en welke tabel heeft er een key voor?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for no words, for a word that neither table has, and for a first word that only `by_shape` can tag?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie zonder woorden, voor een woord dat geen van beide tabellen heeft, en voor een eerste woord dat alleen `by_shape` kan taggen?"},
            "correct": {"en": "Your `tag_sentence` is right in all {n} cases. Look at the picture: the names your tags mark on the test set, beside the names people marked, with the counts of the gazetteer.",
                        "nl": "Je `tag_sentence` klopt in alle {n} gevallen. Kijk naar de afbeelding: de namen die je tags markeren op de test set, naast de namen die mensen markeerden, met de aantallen van de gazetteer."},
        },
    ),
}

CASES = [
    ("longest_at", _LA_START, "empty", None),
    ("longest_at", _LA, "correct", None),
    ("longest_at", _LA.replace("return (i, end, gazetteer[name])", "return [i, end, gazetteer[name]]"), "correct", None),
    ("longest_at", _LA.replace("    return None\n", ""), "correct", None),
    ("longest_at", "def longest_at(words, i, gazetteer):\n    ends = [e for e in range(i + 1, len(words) + 1) if tuple(words[i:e]) in gazetteer]\n    if ends:\n        return (i, max(ends), gazetteer[tuple(words[i:max(ends)])])\n", "correct", None),
    ("longest_at", _LA_SHORT, "wrong", "shortest"),
    ("longest_at", _LA_PAST, "wrong", "past-end"),
    ("longest_at", _LA_LIST, "error", "list"),
    ("longest_at", _LA_INCLUSIVE, "wrong", "inclusive"),
    ("longest_at", _LA_NO_KIND, "wrong", "no-kind"),
    ("longest_at", _LA_NAME, "wrong", "not-span"),
    ("longest_at", _LA_NO_LAST, "wrong", "no-return"),
    ("longest_at", _LA_NO_ONE, "wrong", "no-return"),
    ("longest_at", _LA_EARLY, "wrong", "no-return"),
    ("longest_at", _LA.replace("            return (i, end, gazetteer[name])\n", "            print((i, end, gazetteer[name]))\n").replace("    return None\n", ""), "wrong", "prints"),
    ("longest_at", "def longest_at(words, i, gazetteer):\n    return (1, 5, 'ORG')\n", "almost", None),
    ("longest_at", _LA.replace("    for end", "    ...\n    for end"), "wrong", "dots-left"),
    ("shape", _SH_START, "empty", None),
    ("shape", _SH, "correct", None),
    # three dots inside a string are the learner's own, not the `...` of the start code
    ("shape", _SH + "\nprint(shape(\"...\"))\n", "correct", None),
    ("shape", "def shape(word):\n    out = ''\n    for ch in word:\n        k = 'X' if ch.isupper() else 'x' if ch.islower() else 'd' if ch.isdigit() else ch\n        if out == '' or out[-1] != k:\n            out += k\n    return out\n", "correct", None),
    ("shape", _SH_LONG, "wrong", "long"),
    ("shape", _SH.replace("            last = kind\n", ""), "wrong", "long"),
    ("shape", _SH_CHAR, "wrong", "char"),
    ("shape", _SH_NO_DIGIT, "wrong", "no-digit"),
    ("shape", _SH_PUNCT, "wrong", "punct"),
    ("shape", _SH_NO_ELSE, "wrong", "no-else"),
    ("shape", _SH.replace('    last = ""\n', ""), "error", "unbound"),
    ("shape", _SH_SWAP, "wrong", "swap"),
    ("shape", _SH_EARLY, "wrong", "early"),
    ("shape", "def shape(word):\n    return list(word)\n", "wrong", "not-str"),
    ("shape", "def shape(word):\n    return 'Xx'\n", "almost", None),
    ("tag_sentence", _TS_START, "empty", None),
    ("tag_sentence", _TS, "correct", None),
    ("tag_sentence", _TS.replace("    return tags\n", "    return tuple(tags)\n"), "correct", None),
    ("tag_sentence", "def tag_sentence(words, by_word, by_shape):\n    tags = []\n    for word in words:\n        before = tags[-1] if tags else 'O'\n        tags.append(by_word.get((word, before), by_shape.get((shape(word), before), 'O')))\n    return tags\n", "correct", None),
    ("tag_sentence", _TS_NO_DEFAULT, "error", "no-default"),
    ("tag_sentence", _TS_NO_BEFORE, "wrong", "no-before"),
    ("tag_sentence", _TS_WORD_BEFORE, "wrong", "word-before"),
    ("tag_sentence", _TS_START_NONE, "wrong", "start"),
    ("tag_sentence", _TS_SHAPE_FIRST, "wrong", "shape-first"),
    ("tag_sentence", _TS_NO_SHAPE, "wrong", "no-shape"),
    ("tag_sentence", _TS_NO_WORD, "wrong", "no-word"),
    ("tag_sentence", "def tag_sentence(words, by_word, by_shape):\n    return len(words)\n", "wrong", "not-list"),
    ("tag_sentence", "def tag_sentence(words, by_word, by_shape):\n    return ['B-PER', 'I-PER', 'O', 'O']\n", "almost", None),
]
