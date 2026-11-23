# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-span-f1: the share of tokens whose tag is right (token_accuracy), precision, recall and F1
over names (span_f1), and the names of one type (of_kind); with the pictures of what each function gives for five
taggers on the test set, two pictures of the idea and a demo.

The sentences are those of Universal NER's English-EWT (everything_ai.uner, without the emails), with its own split
into a training set and a test set. The gazetteer and the tagger are those of the lesson about finding names
(nlp-named-entities), made again here from the training set, so nothing is downloaded. Every number in the text and
the pictures is computed by the course (tests/test_nlp_span_f1.py computes them again)."""
import ast
import contextlib
import html
import io
import re
from types import MappingProxyType

from everything_ai import uner, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, kept, with_time_limit

_KEY = "nlp-span-f1"

# --- the course's data -----------------------------------------------------------------------------------------------

TRAIN = uner.split("train")   # 8,773 sentences: tuples of (word, tag) pairs
TEST = uner.split("test")     # 1,471
KINDS = uner.KINDS            # PER, LOC, ORG
START = "O"   # the tag before the first word of a sentence


def words_of(sentence) -> list:
    return [w for w, _ in sentence]


def tags_of(sentence) -> list:
    return [t for _, t in sentence]


def spans(tags) -> list:
    """The course's to_spans of the lesson about BIO: the spans (start, end, kind) of a list of tags."""
    return uner.spans(list(tags))


# The gazetteer and the tagger of the lesson about finding names, made the same way from the training set.

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


GAZETTEER = MappingProxyType(most_frequent(kept("nlp-span-f1:names", lambda: name_counts(TRAIN))))


def one_word_at(words, i, gazetteer):
    """The name of one word at position i, or None."""
    name = (words[i],)
    if name in gazetteer:
        return (i, i + 1, gazetteer[name])
    return None


def longest_at(words, i, gazetteer):
    """The longest name of the gazetteer that starts at position i, as a span, or None."""
    for end in range(len(words), i, -1):
        name = tuple(words[i:end])
        if name in gazetteer:
            return (i, end, gazetteer[name])
    return None


def find_names(words, gazetteer, look=longest_at) -> list:
    """From left to right, at each position that is not inside a name found already, the name that `look` finds."""
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
    """The short word shape: a capital is X, a small letter x and a digit d; any other character stays as it is; a run
    of the same kind becomes one."""
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


BY_WORD = MappingProxyType(most_frequent(kept("nlp-span-f1:by-word", lambda: before_counts(TRAIN, lambda w: w))))
BY_SHAPE = MappingProxyType(most_frequent(kept("nlp-span-f1:by-shape", lambda: before_counts(TRAIN, shape))))


def tag_sentence(words, by_word, by_shape) -> list:
    """From left to right, the tag of (word, tag before) in by_word, else the tag of (shape, tag before) in by_shape,
    else O; the tag before the first word is O."""
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


def capitals(words) -> list:
    """The capitals rule: every word that starts with a capital and is not the first word of its sentence is part of
    a person's name, B-PER after a word outside one, I-PER after another such word."""
    tags = []
    for i, word in enumerate(words):
        if i > 0 and word[:1].isupper():
            tags.append("I-PER" if tags[-1] in ("B-PER", "I-PER") else "B-PER")
        else:
            tags.append("O")
    return tags


# --- five taggers on the test set ------------------------------------------------------------------------------------

TAGGER_KEYS = ("all_o", "one_word", "gazetteer", "tagger", "capitals")
TAGGER_NAMES = {
    "en": {"all_o": "Every token O", "one_word": "Gazetteer, one word at a time",
           "gazetteer": "Gazetteer, longest match", "tagger": "Tagger with the tag before", "capitals": "Capitals rule"},
    "nl": {"all_o": "Elk token O", "one_word": "Gazetteer, één woord tegelijk",
           "gazetteer": "Gazetteer, langste match", "tagger": "Tagger met de tag ervoor", "capitals": "Hoofdletterregel"},
}


def tag_with(key: str, words) -> list:
    """The tags that one of the five taggers gives a list of words."""
    if key == "all_o":
        return ["O"] * len(words)
    if key == "one_word":
        return uner.tags(len(words), find_names(words, GAZETTEER, one_word_at))
    if key == "gazetteer":
        return uner.tags(len(words), find_names(words, GAZETTEER))
    if key == "tagger":
        return tag_sentence(words, BY_WORD, BY_SHAPE)
    assert key == "capitals", key
    return capitals(words)


# The course's own copies, which nothing can change: the tags people gave and those of each tagger, a tuple of tags per
# sentence, and the names people marked and the finds of each tagger, a tuple of spans per sentence.
_GOLD_TAGS = tuple(tuple(tags_of(s)) for s in TEST)
_TAGS = MappingProxyType({k: kept(f"nlp-span-f1:tags-{k}", lambda k=k: tuple(tuple(tag_with(k, words_of(s)))
                                                                                   for s in TEST))
                          for k in TAGGER_KEYS})
_GOLD_NAMES = tuple(tuple(spans(t)) for t in _GOLD_TAGS)
_FINDS = MappingProxyType({k: tuple(tuple(spans(t)) for t in _TAGS[k]) for k in TAGGER_KEYS})


def _lists(rows) -> list:
    """A new list with a list per sentence: what the learner's function gets from a check or a picture, so that code
    that changes its input changes only its own copy."""
    return [list(r) for r in rows]


# What the lesson shows as C.GOLD, C.TAGGERS, C.GOLD_SPANS and C.FOUND: lists of lists, as the exercises say.
GOLD = _lists(_GOLD_TAGS)
TAGGERS = {k: _lists(v) for k, v in _TAGS.items()}
GOLD_SPANS = _lists(_GOLD_NAMES)
FOUND = {k: _lists(v) for k, v in _FINDS.items()}


# --- the course's own exercises --------------------------------------------------------------------------------------

def token_accuracy(gold, guess):
    """Exercise 1: the share of all tokens of all sentences whose tag in guess is the tag in gold."""
    same = 0
    total = 0
    for s in range(len(gold)):
        for k in range(len(gold[s])):
            if gold[s][k] == guess[s][k]:
                same = same + 1
            total = total + 1
    return same / total


def span_f1(gold, found):
    """Exercise 2: precision, recall and F1 over names; a find is right only when the same sentence has a name with
    the same start, end and type. When no find is right, all three are 0.0."""
    right = 0
    n_found = 0
    n_gold = 0
    for s in range(len(gold)):
        for span in found[s]:
            if span in gold[s]:
                right = right + 1
        n_found = n_found + len(found[s])
        n_gold = n_gold + len(gold[s])
    if right == 0:
        return (0.0, 0.0, 0.0)
    precision = right / n_found
    recall = right / n_gold
    return (precision, recall, 2 * precision * recall / (precision + recall))


def of_kind(per_sentence, kind):
    """Exercise 3: per sentence, only the spans of one type, in their order; a list per sentence, also when empty."""
    out = []
    for sentence_spans in per_sentence:
        out.append([span for span in sentence_spans if span[2] == kind])
    return out


# --- the numbers of the lesson ---------------------------------------------------------------------------------------

def overlaps(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def counts(gold, found) -> dict:
    """Finds against names: right, wrong and missed; the wrong finds in three groups: the same words with another type
    ("type"), overlapping a name ("part"), and no name at all ("none")."""
    out = {"names": 0, "found": 0, "right": 0, "type": 0, "part": 0, "none": 0}
    for g, f in zip(gold, found):
        out["names"] += len(g)
        for span in f:
            out["found"] += 1
            if span in g:
                out["right"] += 1
            elif any(x[:2] == span[:2] for x in g):
                out["type"] += 1
            elif any(overlaps(x, span) for x in g):
                out["part"] += 1
            else:
                out["none"] += 1
    out["wrong"] = out["found"] - out["right"]
    out["missed"] = out["names"] - out["right"]
    return out


def numbers() -> dict:
    """The numbers of the text: the test set, and for each tagger its accuracy per token, its counts, its precision,
    recall and span F1, and the same per type."""
    def _go():
        out = {"sentences": len(TEST), "tokens": sum(len(t) for t in _GOLD_TAGS),
               "o_tokens": sum(t.count("O") for t in _GOLD_TAGS), "names": sum(len(s) for s in _GOLD_NAMES),
               "name_tokens": sum(len(t) - t.count("O") for t in _GOLD_TAGS)}
        for k in TAGGER_KEYS:
            same = sum(a == b for g, t in zip(_GOLD_TAGS, _TAGS[k]) for a, b in zip(g, t))
            p, r, f = span_f1(_GOLD_NAMES, _FINDS[k])
            kinds = {}
            for kind in KINDS:
                gk, fk = of_kind(_GOLD_NAMES, kind), of_kind(_FINDS[k], kind)
                kinds[kind] = dict(zip(("p", "r", "f1"), span_f1(gk, fk))) | counts(gk, fk)
            out[k] = {"same": same, "accuracy": same / out["tokens"], "p": p, "r": r, "f1": f, "kinds": kinds,
                      **counts(_GOLD_NAMES, _FINDS[k])}
        return out
    return kept("nlp-span-f1:numbers", _go)


# The example of Jurafsky and Martin (their figure 18.16): a tagger that gives Villanueva O.
SLIP_WORDS = ("Jane", "Villanueva", "of", "United", "Airlines", "Holding", "discussed", "the", "Chicago", "route", ".")
SLIP_GOLD = ("B-PER", "I-PER", "O", "B-ORG", "I-ORG", "I-ORG", "O", "O", "B-LOC", "O", "O")
SLIP_FOUND = ("B-PER", "O", "O", "B-ORG", "I-ORG", "I-ORG", "O", "O", "B-LOC", "O", "O")

# The sentences of the demo, by their sent_id in Universal Dependencies: with the tagger, all names right, a name with
# another type, a name too short and with another type, a name missed, and a find where no name is.
DEMO_IDS = ("answers-20111107143529AAW8bjR_ans-0001", "reviews-314711-0001", "reviews-341750-0004",
            "reviews-256677-0003", "answers-20111108071652AA8GAZw_ans-0003")
DEMO_SENTENCES = MappingProxyType({i: uner.sentence(i).pairs for i in DEMO_IDS})


# --- pictures: helpers -----------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_HEAD_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);font-weight:600;margin-top:0.6rem"
_LINE = "font-family:var(--ea-font-code);font-size:0.85rem;line-height:2.1;color:var(--ea-ink)"
_KIND = "font-family:var(--ea-font-code);font-size:0.62rem;vertical-align:sub;margin-left:2px"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:1px;border-radius:3px"
_PART = "display:inline-block;margin:0 0.2rem 0.3rem 0;text-align:center;vertical-align:top"
_WORD = ("display:block;border-radius:5px;padding:0.15rem 0.4rem;font-family:var(--ea-font-code);font-size:0.9rem;"
         "color:var(--ea-ink);border:1px solid var(--ea-line)")
_TAG = "display:block;font-family:var(--ea-font-code);font-size:0.7rem;margin-top:0.15rem;padding:0 0.15rem"
_ROW = "display:flex;align-items:center;gap:0.5rem;margin:0.15rem 0"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);width:9.5rem;flex:none"
_TRACK = "flex:1;min-width:3rem;height:0.65rem;background:var(--ea-line);border-radius:2px"
_NUM = "font-family:var(--ea-font-code);font-size:0.8rem;width:3.6rem;flex:none;text-align:right"
_ROLE = {   # the line or bar and the color of its text: the data's, the course's, the learner's, or wrong
    "data": ("var(--ea-ink-3)", "var(--ea-ink-2)"),
    "course": ("var(--ea-ink)", "var(--ea-ink)"),
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


def _d(x: float, lang: str, digits: int = 3) -> str:
    """A number from 0 to 1 with three decimals: 0.958 in English, 0,958 in Dutch."""
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


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
    """A tag as text, short: anything that is not a string is shown as Python shows it."""
    text = v if isinstance(v, str) else _repr(v)
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _is_share(v) -> bool:
    """A number from 0 to 1, as an accuracy, a precision, a recall or an F1 score is."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1


def _bar(label: str, value, role: str, lang: str, off: bool = False) -> str:
    """A row with a label, a bar from 0 to 1 and the number; a value that is not a number from 0 to 1 gets no bar."""
    line, color = _ROLE[role]
    fine = _is_share(value)
    width = float(value) * 100 if fine else 0.0
    number = _d(float(value), lang) if fine else _short(value, 12)
    style = _NUM + f";color:{color}" + (";" + _OFF if off else "")
    return (f'<div style="{_ROW}"><span style="{_LABEL}">{_marked(label)}</span>'
            f'<span style="{_TRACK}"><span style="display:block;width:{width:.1f}%;height:100%;background:{line};'
            f'border-radius:2px"></span></span><span style="{style}">{html.escape(number)}</span></div>')


def inline(words, spans_, roles=None) -> str:
    """A sentence as one line of text, with a line under each span and its type after it; roles gives per span
    "data", "course", "mine" or "off". Spans that overlap or lie outside the words are left out of the drawing."""
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


def _roles(found, gold, good: str) -> list:
    return [good if span in gold else "off" for span in found]


def _columns(words, gold_tags, tags) -> str:
    """Each word in its box, with the tag people gave under it, and under that the tag of a tagger: dashed in purple
    when the two differ."""
    out = []
    for word, want, got in zip(words, gold_tags, tags):
        off = got != want
        color = _ROLE["off" if off else "course"][1]
        out.append(f'<span style="{_PART}"><span style="{_WORD}">{html.escape(word)}</span>'
                   f'<span style="{_TAG};color:{_ROLE["data"][1]}">{html.escape(_shown(want))}</span>'
                   f'<span style="{_TAG};color:{color}{";" + _OFF if off else ""}">{html.escape(_shown(got))}</span>'
                   f'</span>')
    return f'<div>{"".join(out)}</div>'


# --- pictures of the idea --------------------------------------------------------------------------------------------

_IDEA_WORDS = {
    "en": {"title": "Five taggers on the test set, scored two ways",
           "acc": "Accuracy per token", "f1": "Span F1",
           "legend": "Each bar runs from 0 to 1, a full bar. In the color of the text: the numbers the course measured "
                     "on the {sentences} sentences of the test set.",
           "explain": "Accuracy per token: the share of the {tokens} tokens whose tag is the tag people gave. Span F1: "
                      "the F1 score over the {names} names, where a find counts as right only with the same start, end "
                      "and type as a name people marked."},
    "nl": {"title": "Vijf taggers op de test set, op twee manieren gescoord",
           "acc": "Accuracy per token", "f1": "Span F1",
           "legend": "Elke balk loopt van 0 tot 1, een volle balk. In de kleur van de tekst: de getallen die de course "
                     "mat op de {sentences} zinnen van de test set.",
           "explain": "Accuracy per token: het deel van de {tokens} tokens waarvan de tag de tag is die mensen gaven. "
                      "Span F1: de F1 score over de {names} namen, waarbij een vondst alleen goed telt met dezelfde "
                      "start, hetzelfde end en hetzelfde type als een naam die mensen markeerden."},
}


def idea_picture(lang: str) -> str:
    """The five taggers with their accuracy per token and their span F1, each as a bar from 0 to 1."""
    w, nums, names = _IDEA_WORDS[lang], numbers(), TAGGER_NAMES[lang]
    rows, said = [], []
    for k in TAGGER_KEYS:
        rows.append(_head(names[k]) + _bar(w["acc"], nums[k]["accuracy"], "course", lang)
                    + _bar(w["f1"], nums[k]["f1"], "course", lang))
        said.append(f'{names[k]}: {w["acc"]} {_d(nums[k]["accuracy"], lang)}, {w["f1"]} {_d(nums[k]["f1"], lang)}')
    notes = [_said(w["legend"], sentences=_n(nums["sentences"], lang)),
             _said(w["explain"], tokens=_n(nums["tokens"], lang), names=_n(nums["names"], lang))]
    return _figure(w["title"], rows, said, notes)


_SLIP_WORDS = {
    "en": {"title": "One tag wrong: one token, and two errors over names",
           "tags": "Per token",
           "gold": "The names people marked",
           "found": "The names the tags of the tagger mark",
           "legend": "Under each word: first the tag people gave, in gray, then the tag of the tagger. A line under one "
                     "or more words marks a name, with its type after it. Gray: what people marked. In the color of "
                     "the text: a tag or a name of the tagger that matches. Purple dashed: a tag or a name that differs.",
           "count": "Per token: {same} of the {n} tags right, an accuracy of {acc}. Per name: {right} of the {found} "
                    "finds right and {right} of the {names} names found, so precision, recall and F1 are all {f1}."},
    "nl": {"title": "Eén tag fout: één token, en twee fouten bij de namen",
           "tags": "Per token",
           "gold": "De namen die mensen markeerden",
           "found": "De namen die de tags van de tagger markeren",
           "legend": "Onder elk woord: eerst de tag die mensen gaven, in grijs, dan de tag van de tagger. Een lijn onder "
                     "een of meer woorden markeert een naam, met het type erachter. Grijs: wat mensen markeerden. In "
                     "de kleur van de tekst: een tag of een naam van de tagger die klopt. Paars gestreept: een tag of een "
                     "naam die afwijkt.",
           "count": "Per token: {same} van de {n} tags goed, een accuracy van {acc}. Per naam: {right} van de {found} "
                    "vondsten goed en {right} van de {names} namen gevonden, dus precision, recall en F1 zijn alle "
                    "drie {f1}."},
}


def slip_picture(lang: str) -> str:
    """The example of Jurafsky and Martin: Villanueva tagged O, per token and per name."""
    w = _SLIP_WORDS[lang]
    words = list(SLIP_WORDS)
    gold, found = spans(SLIP_GOLD), spans(SLIP_FOUND)
    same = sum(a == b for a, b in zip(SLIP_GOLD, SLIP_FOUND))
    p, r, f = span_f1([gold], [found])
    right = sum(s in gold for s in found)
    rows = [_head(w["tags"]) + _columns(words, SLIP_GOLD, SLIP_FOUND),
            _head(w["gold"]) + inline(words, gold),
            _head(w["found"]) + inline(words, found, _roles(found, gold, "course"))]
    said = [" ".join(f"{x}/{a}/{b}" for x, a, b in zip(words, SLIP_GOLD, SLIP_FOUND)),
            f'{_bare(w["gold"])}: {_plain(words, gold)}', f'{_bare(w["found"])}: {_plain(words, found)}']
    notes = [_said(w["legend"]),
             _said(w["count"], same=same, n=len(words), acc=_d(same / len(words), lang), right=right,
                   found=len(found), names=len(gold), f1=_d(f, lang))]
    return _figure(w["title"], rows, said, notes)


# --- the demo --------------------------------------------------------------------------------------------------------

_DEMO_WORDS = {
    "en": {"title": "{tagger}, on one sentence of the test set",
           "tags": "Per token",
           "gold": "The names people marked",
           "found": "The names the tagger marks",
           "legend": "Under each word: first the tag people gave, in gray, then the tag of the tagger. A line under one "
                     "or more words marks a name, with its type after it. Gray: what people marked. In the color of "
                     "the text: a tag or a name of the tagger that matches. Purple dashed: a tag or a name that differs.",
           "here": "In this sentence: {same} of the {n} tags right. Names right: {right}; wrong finds: {wrong}; names "
                   "missed: {missed}.",
           "all": "On the whole test set: accuracy {acc}; precision {p}, recall {r}, span F1 {f1}."},
    "nl": {"title": "{tagger}, op één zin uit de test set",
           "tags": "Per token",
           "gold": "De namen die mensen markeerden",
           "found": "De namen die de tagger markeert",
           "legend": "Onder elk woord: eerst de tag die mensen gaven, in grijs, dan de tag van de tagger. Een lijn onder "
                     "een of meer woorden markeert een naam, met het type erachter. Grijs: wat mensen markeerden. In "
                     "de kleur van de tekst: een tag of een naam van de tagger die klopt. Paars gestreept: een tag of een "
                     "naam die afwijkt.",
           "here": "In deze zin: {same} van de {n} tags goed. Goede namen: {right}; foute vondsten: {wrong}; gemiste "
                   "namen: {missed}.",
           "all": "Op de hele test set: accuracy {acc}; precision {p}, recall {r}, span F1 {f1}."},
}


def tagger_options(lang: str) -> dict:
    """The dropdown of the demo: the name of each tagger in the lesson's language, to its key."""
    return {TAGGER_NAMES[lang][k]: k for k in TAGGER_KEYS}


def sentence_options() -> dict:
    """The dropdown of the demo: each sentence as text, to its sent_id."""
    return {" ".join(words_of(DEMO_SENTENCES[i])): i for i in DEMO_IDS}


def demo_picture(tagger, sent_id, lang: str) -> str:
    """One tagger on one sentence of the test set: per token and per name, with its scores on the whole test set."""
    w = _DEMO_WORDS[lang]
    tagger = tagger if tagger in TAGGER_KEYS else "tagger"
    sentence = DEMO_SENTENCES.get(sent_id, DEMO_SENTENCES[DEMO_IDS[0]])
    words, gold_tags = words_of(sentence), tags_of(sentence)
    tags = tag_with(tagger, words)
    gold, found = spans(gold_tags), spans(tags)
    c, nums = counts([gold], [found]), numbers()[tagger]
    title = w["title"].replace("{tagger}", TAGGER_NAMES[lang][tagger])
    rows = [_head(w["tags"]) + _columns(words, gold_tags, tags),
            _head(w["gold"]) + inline(words, gold),
            _head(w["found"]) + inline(words, found, _roles(found, gold, "course"))]
    said = [" ".join(f"{x}/{a}/{b}" for x, a, b in zip(words, gold_tags, tags)),
            f'{_bare(w["gold"])}: {_plain(words, gold)}', f'{_bare(w["found"])}: {_plain(words, found)}']
    same = sum(a == b for a, b in zip(gold_tags, tags))
    notes = [_said(w["legend"]),
             _said(w["here"], same=same, n=len(words), right=c["right"], wrong=c["wrong"], missed=c["missed"]),
             _said(w["all"], acc=_d(nums["accuracy"], lang), p=_d(nums["p"], lang), r=_d(nums["r"], lang),
                   f1=_d(nums["f1"], lang))]
    return _figure(title, rows, said, notes)


# --- running the learner's function in a picture ---------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _quiet gives for a function that stopped with an error
_TOO_SLOW = object()  # what _timed gives when the learner's function takes too long; None is a value of theirs
PICTURE_SECONDS = 2   # what the learner's function gets for all five taggers of a picture


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
_STOP = {"en": "For {tagger}, your function stops with an error on the whole test set.",
         "nl": "Voor {tagger} stopt je functie met een fout op de hele test set."}
_DIFFERS = {"en": "Purple dashed: a number that differs from the course's; the course gets {want} for {tagger}.",
            "nl": "Paars gestreept: een getal dat afwijkt van het getal van de course; de course krijgt {want} voor "
                  "{tagger}."}


def _same_number(a, b) -> bool:
    return _is_share(a) and abs(float(a) - b) < 1e-9


_ACC_WORDS = {
    "en": {"title": "Your token_accuracy on five taggers",
           "blue": "Blue: what your token_accuracy gives for each tagger on the {tokens} tokens of the test set; a full "
                   "bar is 1.",
           "odd": "For {tagger}, your function gives {got} instead of a number from 0 to 1.",
           "spread": "From {lo} to {hi}: the five lie within {gap} of each other. Every token O, which finds no name at "
                     "all, gets {all_o}, more than the capitals rule."},
    "nl": {"title": "Jouw token_accuracy op vijf taggers",
           "blue": "Blauw: wat je token_accuracy geeft voor elke tagger op de {tokens} tokens van de test set; een volle "
                   "balk is 1.",
           "odd": "Voor {tagger} geeft je functie {got} in plaats van een getal van 0 tot 1.",
           "spread": "Van {lo} tot {hi}: de vijf liggen binnen {gap} van elkaar. Elk token O, dat geen enkele naam "
                     "vindt, krijgt {all_o}, meer dan de hoofdletterregel."},
}


def accuracy_picture(seen) -> str:
    """The learner's token_accuracy on the tags of the five taggers, as bars, with the course's number beside a value
    that differs."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w, nums, names = _ACC_WORDS[lang], numbers(), TAGGER_NAMES[lang]
    got = _timed(lambda: {k: _quiet(fn, (_lists(_GOLD_TAGS), _lists(_TAGS[k]))) for k in TAGGER_KEYS})
    if got is _TOO_SLOW:
        return _figure(w["title"], [], [], [_said(_SLOW[lang])])
    rows, said, notes, differs = [], [], [], []
    for k in TAGGER_KEYS:
        v, want = got[k], nums[k]["accuracy"]
        if v is _STOPPED:
            notes.append(_said(_STOP[lang], tagger=names[k]))
            continue
        if not _is_share(v):
            notes.append(_said(w["odd"], tagger=names[k], got=_short(v)))
            continue
        off = not _same_number(v, want)
        if off:
            differs.append(k)
        rows.append(_bar(names[k], v, "mine", lang, off=off))
        said.append(f"{names[k]}: {_d(float(v), lang)}")
    extra = [_said(w["blue"], tokens=_n(nums["tokens"], lang))] if rows else []
    if differs:
        extra.append(_said(_DIFFERS[lang], want=_d(nums[differs[0]]["accuracy"], lang), tagger=names[differs[0]]))
    elif len(rows) == len(TAGGER_KEYS):
        values = [nums[k]["accuracy"] for k in TAGGER_KEYS]
        extra.append(_said(w["spread"], lo=_d(min(values), lang), hi=_d(max(values), lang),
                           gap=_d(max(values) - min(values), lang), all_o=_d(nums["all_o"]["accuracy"], lang)))
    return _figure(w["title"], rows, said, extra + notes)


_F1_WORDS = {
    "en": {"title": "Your span_f1 on five taggers",
           "acc": "Accuracy per token", "f1": "Your span F1",
           "ink": "In the color of the text: the accuracy per token, as the course measured it.",
           "blue": "Blue: the span F1 that your span_f1 gives, over the {names} names of the test set; a full bar is 1.",
           "pr": "{tagger}: precision {p}, recall {r}.",
           "odd": "For {tagger}, your function gives {got} instead of three numbers from 0 to 1.",
           "dashed": "Purple dashed: a span F1 that differs from the course's.",
           "want": "For {tagger}, the course gets precision {p}, recall {r} and span F1 {f1}.",
           "order": "Accuracy puts the capitals rule last, below Every token O; span F1 puts Every token O last, with "
                    "0. The best tagger has {right} of the {names} names right, with {wrong} wrong finds."},
    "nl": {"title": "Jouw span_f1 op vijf taggers",
           "acc": "Accuracy per token", "f1": "Jouw span F1",
           "ink": "In de kleur van de tekst: de accuracy per token, gemeten door de course.",
           "blue": "Blauw: de span F1 die je span_f1 geeft, over de {names} namen van de test set; een volle balk is 1.",
           "pr": "{tagger}: precision {p}, recall {r}.",
           "odd": "Voor {tagger} geeft je functie {got} in plaats van drie getallen van 0 tot 1.",
           "dashed": "Paars gestreept: een span F1 die afwijkt van de span F1 van de course.",
           "want": "Voor {tagger} krijgt de course precision {p}, recall {r} en span F1 {f1}.",
           "order": "Accuracy zet de hoofdletterregel onderaan, onder Elk token O; span F1 zet Elk token O onderaan, "
                    "met 0. De beste tagger heeft {right} van de {names} namen goed, met {wrong} foute vondsten."},
}


def _triple(v):
    """What span_f1 gives as three numbers from 0 to 1, or None."""
    if isinstance(v, (list, tuple)) and len(v) == 3 and all(_is_share(x) for x in v):
        return tuple(float(x) for x in v)
    return None


def f1_picture(seen) -> str:
    """The learner's span_f1 on the names of the five taggers, as bars beside the course's accuracy per token."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w, nums, names = _F1_WORDS[lang], numbers(), TAGGER_NAMES[lang]
    got = _timed(lambda: {k: _quiet(fn, (_lists(_GOLD_NAMES), _lists(_FINDS[k]))) for k in TAGGER_KEYS})
    if got is _TOO_SLOW:
        return _figure(w["title"], [], [], [_said(_SLOW[lang])])
    rows, said, notes, pr, differs = [], [], [], [], []
    for k in TAGGER_KEYS:
        v = got[k]
        if v is _STOPPED:
            notes.append(_said(_STOP[lang], tagger=names[k]))
            continue
        three = _triple(v)
        if three is None:
            notes.append(_said(w["odd"], tagger=names[k], got=_short(v)))
            continue
        want = (nums[k]["p"], nums[k]["r"], nums[k]["f1"])
        off = not _same_number(three[2], want[2])
        if not all(_same_number(a, b) for a, b in zip(three, want)):
            differs.append(k)
        rows.append(_head(names[k]) + _bar(w["acc"], nums[k]["accuracy"], "course", lang)
                    + _bar(w["f1"], three[2], "mine", lang, off=off))
        said.append(f'{names[k]}: {w["acc"]} {_d(nums[k]["accuracy"], lang)}, {w["f1"]} {_d(three[2], lang)}')
        pr.append(_said(w["pr"], tagger=names[k], p=_d(three[0], lang), r=_d(three[1], lang)))
    extra = [_said(w["ink"]), _said(w["blue"], names=_n(nums["names"], lang))] if rows else []
    if differs:
        k = differs[0]
        if any("outline" in r for r in rows):
            extra.append(_said(w["dashed"]))
        extra.append(_said(w["want"], tagger=names[k], p=_d(nums[k]["p"], lang), r=_d(nums[k]["r"], lang),
                           f1=_d(nums[k]["f1"], lang)))
    elif len(rows) == len(TAGGER_KEYS):
        best = nums["tagger"]
        extra.append(_said(w["order"], right=_n(best["right"], lang), names=_n(nums["names"], lang),
                           wrong=_n(best["wrong"], lang)))
    return _figure(w["title"], rows, said, extra + pr + notes)


_KIND_WORDS = {
    "en": {"title": "Your of_kind: span F1 per type",
           "kind": {"PER": "PER, persons", "LOC": "LOC, places", "ORG": "ORG, organizations"},
           "blue": "Blue: the span F1 of the names of one type, with your of_kind and the course's span_f1; a full bar "
                   "is 1.",
           "odd": "For the {kind} names of the test set, your function gives {got} instead of a list with {n} lists, "
                  "one for each sentence.",
           "stop": "For the {kind} names of the test set, your function stops with an error.",
           "all": "Over all three types together: {gazetteer} {g}, {tagger} {t}."},
    "nl": {"title": "Jouw of_kind: span F1 per type",
           "kind": {"PER": "PER, personen", "LOC": "LOC, plaatsen", "ORG": "ORG, organisaties"},
           "blue": "Blauw: de span F1 van de namen van één type, met jouw of_kind en de span_f1 van de course; een volle "
                   "balk is 1.",
           "odd": "Voor de {kind}-namen van de test set geeft je functie {got} in plaats van een lijst met {n} lijsten, "
                  "één voor elke zin.",
           "stop": "Voor de {kind}-namen van de test set stopt je functie met een fout.",
           "all": "Over alle drie de types samen: {gazetteer} {g}, {tagger} {t}."},
}
_KIND_TAGGERS = ("gazetteer", "tagger")


def _per_sentence(v, n: int):
    """What of_kind gives, as n lists of spans that span_f1 can read, or None."""
    if not isinstance(v, (list, tuple)) or len(v) != n:
        return None
    if not all(isinstance(x, (list, tuple)) for x in v):
        return None
    return [[tuple(s) if isinstance(s, list) else s for s in x] for x in v]


def kinds_picture(seen) -> str:
    """The learner's of_kind on the names people marked and on the finds of the gazetteer and the tagger: the span F1
    of each type, with the course's span_f1."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w, nums, names = _KIND_WORDS[lang], numbers(), TAGGER_NAMES[lang]
    n = len(TEST)

    def _all():
        out = {}
        for kind in KINDS:
            out[("gold", kind)] = _quiet(fn, (_lists(_GOLD_NAMES), kind))
            for k in _KIND_TAGGERS:
                out[(k, kind)] = _quiet(fn, (_lists(_FINDS[k]), kind))
        return out

    got = _timed(_all)
    if got is _TOO_SLOW:
        return _figure(w["title"], [], [], [_said(_SLOW[lang])])
    rows, said, notes = [], [], []
    for (who, kind), v in got.items():   # one note for the first result that is not one list per sentence
        if v is _STOPPED:
            return _figure(w["title"], [], [], [_said(w["stop"], kind=kind)])
        if _per_sentence(v, n) is None:
            return _figure(w["title"], [], [], [_said(w["odd"], kind=kind, got=_short(v), n=_n(n, lang))])
    for kind in KINDS:
        gold = _per_sentence(got[("gold", kind)], n)
        lines = []
        for k in _KIND_TAGGERS:
            found = _per_sentence(got[(k, kind)], n)
            try:
                f = span_f1(gold, found)[2]
            except Exception:   # noqa: BLE001   a span of the learner's that cannot be compared
                continue
            want = nums[k]["kinds"][kind]["f1"]
            lines.append(_bar(names[k], f, "mine", lang, off=not _same_number(f, want)))
            said.append(f'{kind} {names[k]}: {_d(f, lang)}')
        if lines:
            rows.append(_head(w["kind"][kind]) + "".join(lines))
    extra = [_said(w["blue"])] if rows else []
    if rows:
        extra.append(_said(w["all"], gazetteer=names["gazetteer"], g=_d(nums["gazetteer"]["f1"], lang),
                           tagger=names["tagger"], t=_d(nums["tagger"]["f1"], lang)))
        if any("outline" in r for r in rows):
            extra.append(_said({"en": "Purple dashed: a number that differs from the course's.",
                                "nl": "Paars gestreept: een getal dat afwijkt van het getal van de course."}[lang]))
    return _figure(w["title"], rows, said, extra + notes)


# --- the checks ------------------------------------------------------------------------------------------------------

REQUIRED = ["token_accuracy", "span_f1", "of_kind", "near-miss"]


class Named:
    """A value the check gives the learner's function, shown in a message by its name: the course's tuple of tuples,
    handed over as a new list of lists, as the exercises say."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return _lists(self.value)


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


def _zero(e) -> bool:
    return isinstance(e, ZeroDivisionError)


def _not_tuple(got, text) -> bool:
    return got is not None and not (isinstance(got, (list, tuple)) and len(got) == 3)


def _not_f1(got, text) -> bool:
    """Three numbers whose third is not the F1 score of the first two."""
    if not (isinstance(got, (list, tuple)) and len(got) == 3 and all(_is_share(x) for x in got)):
        return False
    p, r, f = (float(x) for x in got)
    return abs(f - (2 * p * r / (p + r) if p + r else 0.0)) > 1e-9


def _not_list(got, text) -> bool:
    return got is not None and not isinstance(got, (list, tuple))


def _rounded(v, depth: int = 0) -> str:
    """A value the way Python shows it, with its floats rounded to four decimals, also inside a tuple or a list; at most
    30 items of a list and three levels deep."""
    if depth > 3:
        return "…"
    if isinstance(v, float):
        text = f"{round(v, 4) + 0.0:.4f}".rstrip("0")
        return text + "0" if text.endswith(".") else text
    if type(v) is tuple:
        inner = ", ".join(_rounded(x, depth + 1) for x in v[:30]) + (", …" if len(v) > 30 else "")
        return "(" + inner + ("," if len(v) == 1 else "") + ")"
    if type(v) is list:
        return "[" + ", ".join(_rounded(x, depth + 1) for x in v[:30]) + (", …" if len(v) > 30 else "") + "]"
    return _repr(v)


def show_value(v) -> str:
    """How {got} and {want} show a value: a number as a number, rounded to four decimals; anything else as code, cut
    after 90 characters."""
    text = _rounded(v)
    if len(text) > 90:
        text = text[:89] + "…"
    if isinstance(v, (int, float)) and not isinstance(v, bool) or "`" in text:
        return text
    return f"`{text}`"


def _as_tuple(v):
    """What span_f1 gives, with three numbers in a list made a tuple, as the picture under the check reads it."""
    return tuple(v) if isinstance(v, list) else v


def _as_lists(v):
    """What of_kind gives, with a tuple of spans made a list and a span as a list made a tuple, as the picture under the
    check reads it."""
    if not isinstance(v, (list, tuple)):
        return v
    out = []
    for x in v:
        if isinstance(x, tuple) and all(isinstance(s, (list, tuple)) for s in x):
            x = list(x)   # the spans of a sentence as a tuple: the same as a list
        if isinstance(x, list):
            x = [tuple(s) if isinstance(s, list) else s for s in x]
        out.append(x)
    return out


_GOLD = Named("C.GOLD", _GOLD_TAGS)
_GOLD_SPANS = Named("C.GOLD_SPANS", _GOLD_NAMES)

# exercise 1: token_accuracy. The visible case: two sentences of three and two tokens, one tag wrong; then every tag
# right, no tag right in two sentences, a tag of one token with another type, sentences of 9 and 2 tokens in which the
# share per sentence differs from the share of all tokens, and the tags of the test set with every token O and with
# the tagger.
TA_CASES = [
    ([["B-PER", "I-PER", "O"], ["O", "B-LOC"]], [["B-PER", "O", "O"], ["O", "B-LOC"]]),
    ([["O", "B-ORG", "I-ORG"]], [["O", "B-ORG", "I-ORG"]]),
    ([["B-PER"], ["B-LOC", "I-LOC"]], [["O"], ["O", "O"]]),
    ([["B-PER"]], [["B-ORG"]]),
    ([["O", "O", "O", "O", "O", "O", "O", "O", "B-LOC"], ["B-PER", "I-PER"]],
     [["O", "O", "O", "O", "O", "O", "O", "O", "O"], ["O", "O"]]),
    (_GOLD, Named('C.TAGGERS["all_o"]', _TAGS["all_o"])),
    (_GOLD, Named('C.TAGGERS["tagger"]', _TAGS["tagger"])),
]
_TA_HEAD = "def token_accuracy(gold, guess):\n"
_TA_START = _TA_HEAD + "    ...\n"
_TA = (_TA_HEAD
       + "    same = 0\n"
       + "    total = 0\n"
       + "    for s in range(len(gold)):\n"
       + "        for k in range(len(gold[s])):\n"
       + "            if gold[s][k] == guess[s][k]:\n"
       + "                same = same + 1\n"
       + "            total = total + 1\n"
       + "    return same / total\n")
_TA_MEAN = (_TA_HEAD
            + "    shares = 0\n"
            + "    for s in range(len(gold)):\n"
            + "        same = 0\n"
            + "        for k in range(len(gold[s])):\n"
            + "            if gold[s][k] == guess[s][k]:\n"
            + "                same = same + 1\n"
            + "        shares = shares + same / len(gold[s])\n"
            + "    return shares / len(gold)\n")
_TA_WHOLE = (_TA_HEAD
             + "    same = 0\n"
             + "    for s in range(len(gold)):\n"
             + "        if gold[s] == guess[s]:\n"
             + "            same = same + 1\n"
             + "    return same / len(gold)\n")
_TA_BY_SENTENCES = _TA.replace("    return same / total\n", "    return same / len(gold)\n")
_TA_COUNT = _TA.replace("    return same / total\n", "    return same\n")
_TA_ERRORS = _TA.replace("    return same / total\n", "    return (total - same) / total\n")
_TA_NO_TOTAL = _TA.replace("            total = total + 1\n", "")
_TA_INSIDE = _TA.replace("                same = same + 1\n            total = total + 1\n",
                         "                same = same + 1\n                total = total + 1\n")

# exercise 2: span_f1. The visible case: Jane Villanueva of United Airlines flew to San Diego, with Jane found as a
# person, United Airlines right, and San Diego missed; then nothing found, every name found in two sentences, the same
# words with another type, a find that matches a name of another sentence, more finds than names, a sentence with no
# names and no finds, and the names of the test set with the gazetteer and with the tagger.
SF_CASES = [
    ([[(0, 2, "PER"), (3, 5, "ORG"), (7, 9, "LOC")]], [[(0, 1, "PER"), (3, 5, "ORG")]]),
    ([[(0, 2, "PER")]], [[]]),
    ([[(0, 2, "PER")], [(1, 2, "LOC")]], [[(0, 2, "PER")], [(1, 2, "LOC")]]),
    ([[(4, 5, "LOC")]], [[(4, 5, "ORG")]]),
    ([[(0, 1, "PER")], []], [[], [(0, 1, "PER")]]),
    ([[(2, 3, "LOC")], []], [[(2, 3, "LOC")], [(0, 1, "PER"), (4, 6, "ORG")]]),
    ([[]], [[]]),
    (_GOLD_SPANS, Named('C.FOUND["gazetteer"]', _FINDS["gazetteer"])),
    (_GOLD_SPANS, Named('C.FOUND["tagger"]', _FINDS["tagger"])),
]
_SF_HEAD = "def span_f1(gold, found):\n"
_SF_START = _SF_HEAD + "    ...\n"
_SF = (_SF_HEAD
       + "    right = 0\n"
       + "    n_found = 0\n"
       + "    n_gold = 0\n"
       + "    for s in range(len(gold)):\n"
       + "        for span in found[s]:\n"
       + "            if span in gold[s]:\n"
       + "                right = right + 1\n"
       + "        n_found = n_found + len(found[s])\n"
       + "        n_gold = n_gold + len(gold[s])\n"
       + "    if right == 0:\n"
       + "        return (0.0, 0.0, 0.0)\n"
       + "    precision = right / n_found\n"
       + "    recall = right / n_gold\n"
       + "    f1 = 2 * precision * recall / (precision + recall)\n"
       + "    return (precision, recall, f1)\n")
_SF_NO_ZERO = _SF.replace("    if right == 0:\n        return (0.0, 0.0, 0.0)\n", "")
_SF_POOLED = (_SF.replace("    right = 0\n", "    all_names = []\n    for names in gold:\n        all_names = all_names + list(names)\n    right = 0\n", 1)
              .replace("if span in gold[s]:", "if span in all_names:"))
_SF_UNTYPED = _SF.replace("if span in gold[s]:", "if span[:2] in [g[:2] for g in gold[s]]:")
_SF_OVERLAP = _SF.replace("if span in gold[s]:",
                          "if any(g[2] == span[2] and g[0] < span[1] and span[0] < g[1] for g in gold[s]):")
_SF_SWAP = (_SF.replace("    precision = right / n_found\n    recall = right / n_gold\n",
                        "    precision = right / n_gold\n    recall = right / n_found\n"))
_SF_MEAN = _SF.replace("    f1 = 2 * precision * recall / (precision + recall)\n", "    f1 = (precision + recall) / 2\n")
_SF_ONLY_F1 = _SF.replace("    return (precision, recall, f1)\n", "    return f1\n")
_SF_PER_SENTENCE = (_SF_HEAD
                    + "    ps = []\n    rs = []\n    fs = []\n"
                    + "    for s in range(len(gold)):\n"
                    + "        right = 0\n"
                    + "        for span in found[s]:\n"
                    + "            if span in gold[s]:\n"
                    + "                right = right + 1\n"
                    + "        if right == 0:\n"
                    + "            ps.append(0.0)\n            rs.append(0.0)\n            fs.append(0.0)\n"
                    + "        else:\n"
                    + "            p = right / len(found[s])\n            r = right / len(gold[s])\n"
                    + "            ps.append(p)\n            rs.append(r)\n            fs.append(2 * p * r / (p + r))\n"
                    + "    return (sum(ps) / len(ps), sum(rs) / len(rs), sum(fs) / len(fs))\n")
_SF_HARMONIC_HALF = _SF.replace("    f1 = 2 * precision * recall / (precision + recall)\n",
                                "    f1 = precision * recall / (precision + recall)\n")

# exercise 3: of_kind. The visible case: two sentences, the persons of the first kept and an empty list for the second;
# then a type that no span has, no sentences, two names of the type in one sentence with another type between them, a
# sentence without names between two others, and the organizations of the test set.
OK_CASES = [
    ([[(0, 2, "PER"), (3, 5, "ORG")], [(1, 2, "LOC")]], "PER"),
    ([[(0, 1, "LOC")], [(2, 4, "ORG")]], "PER"),
    ([], "LOC"),
    ([[(0, 1, "LOC"), (2, 3, "PER"), (4, 6, "LOC")]], "LOC"),
    ([[(0, 1, "ORG")], [], [(3, 4, "ORG")]], "ORG"),
    (_GOLD_SPANS, "ORG"),
]
_OK_HEAD = "def of_kind(per_sentence, kind):\n"
_OK_START = _OK_HEAD + "    ...\n"
_OK = (_OK_HEAD
       + "    out = []\n"
       + "    for sentence_spans in per_sentence:\n"
       + "        out.append([span for span in sentence_spans if span[2] == kind])\n"
       + "    return out\n")
_OK_FLAT = (_OK_HEAD
            + "    out = []\n"
            + "    for sentence_spans in per_sentence:\n"
            + "        for span in sentence_spans:\n"
            + "            if span[2] == kind:\n"
            + "                out.append(span)\n"
            + "    return out\n")
_OK_DROP = (_OK_HEAD
            + "    out = []\n"
            + "    for sentence_spans in per_sentence:\n"
            + "        these = [span for span in sentence_spans if span[2] == kind]\n"
            + "        if these:\n"
            + "            out.append(these)\n"
            + "    return out\n")
_OK_OTHERS = _OK.replace("span[2] == kind", "span[2] != kind")
_OK_ITEM = _OK.replace("span[2] == kind", "span[0] == kind")

EXERCISES = {
    "token_accuracy": FunctionBox(
        name="token_accuracy",
        picture=accuracy_picture,
        function="token_accuracy",
        start=_TA_START,
        solution=_TA,
        visible=TA_CASES[:1],
        hidden=TA_CASES[1:],
        show=show_value,
        first=[_DOTS],
        mistakes=[
            Mistake("zero", error=_zero,
                    en="With `{call}`, your function stops with a ZeroDivisionError: the number it divides by is 0. Which tokens should that number count?",
                    nl="Met `{call}` stopt je functie met een ZeroDivisionError: het getal waardoor je functie deelt is 0. Welke tokens moet dat getal tellen?"),
            Mistake("per-sentence", wrong=_TA_MEAN,
                    en="`{call}` gives {got}, and it should give {want}: your function takes the share of each sentence, and then the mean of those shares, so a short sentence weighs as much as a long one. What should the share be of?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie neemt het deel van elke zin, en dan het gemiddelde van die delen, dus een korte zin telt even zwaar als een lange. Waarvan moet het deel zijn?"),
            Mistake("whole-sentence", wrong=_TA_WHOLE,
                    en="`{call}` gives {got}, and it should give {want}: your function counts the sentences whose tags are all right, so one wrong tag makes a whole sentence count as wrong. What should it compare, one at a time?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt de zinnen waarvan alle tags goed zijn, dus één foute tag maakt een hele zin fout. Wat moet je functie één voor één vergelijken?"),
            Mistake("by-sentences", wrong=_TA_BY_SENTENCES,
                    en="`{call}` gives {got}, and it should give {want}: your function counts tokens, and divides by the number of sentences. What should it divide by?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt tokens, en deelt door het aantal zinnen. Waardoor moet je functie delen?"),
            Mistake("count", wrong=_TA_COUNT,
                    en="`{call}` gives {got}, and it should give {want}: the number of tokens whose tag is right. Accuracy is a share: of what?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het aantal tokens met een goede tag. Accuracy is een deel: waarvan?"),
            Mistake("errors", wrong=_TA_ERRORS,
                    en="`{call}` gives {got}, and it should give {want}: the share of tokens whose tags differ. Which tokens does accuracy count?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het deel van de tokens waarvan de tags verschillen. Welke tokens telt accuracy?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: count the tokens of all sentences, and the tokens whose tag in `guess` is the tag in `gold`, and give back the share.",
                      "nl": "Vervang de `...`: tel de tokens van alle zinnen, en de tokens waarvan de tag in `guess` de tag in `gold` is, en geef het deel terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which tokens does your function count, and what does it divide by?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Welke tokens telt je functie, en waardoor deelt je functie?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work when every tag is right, when no tag is right, and for sentences of different lengths?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie als elke tag goed is, als geen enkele tag goed is, en voor zinnen van verschillende lengte?"},
            "correct": {"en": "Your `token_accuracy` is right in all {n} cases. Look at the picture: your function on the tags of the five taggers.",
                        "nl": "Je `token_accuracy` klopt in alle {n} gevallen. Kijk naar de afbeelding: jouw functie op de tags van de vijf taggers."},
        },
    ),
    "span_f1": FunctionBox(
        name="span_f1",
        picture=f1_picture,
        function="span_f1",
        start=_SF_START,
        solution=_SF,
        visible=SF_CASES[:1],
        hidden=SF_CASES[1:],
        collect=_as_tuple,
        show=show_value,
        first=[_DOTS],
        mistakes=[
            Mistake("zero", error=_zero,
                    en="With `{call}`, your function stops with a ZeroDivisionError: the number it divides by is 0. What should your function give when no find is right?",
                    nl="Met `{call}` stopt je functie met een ZeroDivisionError: het getal waardoor je functie deelt is 0. Wat moet je functie geven als geen enkele vondst goed is?"),
            Mistake("pooled", wrong=_SF_POOLED,
                    en="`{call}` gives {got}, and it should give {want}: a find counts as right when any sentence has a name with that span. Which names may a find be compared with?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een vondst telt als goed als een willekeurige zin een naam met die span heeft. Met welke namen mag een vondst vergeleken worden?"),
            Mistake("untyped", wrong=_SF_UNTYPED,
                    en="`{call}` gives {got}, and it should give {want}: a find with the start and end of a name counts as right, also with another type. What else must match?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een vondst met de start en het end van een naam telt als goed, ook met een ander type. Wat moet er nog meer kloppen?"),
            Mistake("overlap", wrong=_SF_OVERLAP,
                    en="`{call}` gives {got}, and it should give {want}: a find that shares some words with a name of the same type counts as right. When does a find count as right in span F1?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een vondst die een paar woorden deelt met een naam van hetzelfde type telt als goed. Wanneer telt een vondst als goed bij span F1?"),
            Mistake("swapped", wrong=_SF_SWAP,
                    en="`{call}` gives {got}, and it should give {want}: your first number is the share of the names that were found, and your second the share of the finds that are right. Which of the two shares is precision?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je eerste getal is het deel van de namen dat gevonden is, en je tweede het deel van de vondsten dat goed is. Welk van de twee delen is precision?"),
            Mistake("mean", wrong=_SF_MEAN,
                    en="`{call}` gives {got}, and it should give {want}: your third number is the plain mean of precision and recall. Which mean is the F1 score?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je derde getal is het gewone gemiddelde van precision en recall. Welk gemiddelde is de F1 score?"),
            Mistake("per-sentence", wrong=_SF_PER_SENTENCE,
                    en="`{call}` gives {got}, and it should give {want}: your function scores each sentence on its own and then takes the mean of those scores, so a sentence with one name weighs as much as a sentence with five. Which counts should your function add up before it divides?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft elke zin een eigen score en neemt dan het gemiddelde van die scores, dus een zin met één naam telt even zwaar als een zin met vijf. Welke aantallen moet je functie optellen voordat je functie deelt?"),
            Mistake("not-f1", matches=_not_f1,
                    en="`{call}` gives {got}, and it should give {want}: your third number is not the F1 score of your first two. Which mean of precision and recall is the F1 score?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je derde getal is niet de F1 score van je eerste twee getallen. Welk gemiddelde van precision en recall is de F1 score?"),
            Mistake("not-tuple", matches=_not_tuple,
                    en="`{call}` gives {got}, and it should give {want}: three numbers, `(precision, recall, f1)`. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: drie getallen, `(precision, recall, f1)`. Wat geeft je `return`-regel terug?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: count the finds, the names, and the finds that are also a name in the same sentence, and give back precision, recall and F1.",
                      "nl": "Vervang de `...`: tel de vondsten, de namen, en de vondsten die ook een naam in dezelfde zin zijn, en geef precision, recall en F1 terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which finds does your function count as right, and what do precision and recall divide by?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Welke vondsten telt je functie als goed, en waardoor delen precision en recall?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work when nothing is found, when a find has another type, when a find matches a name of another sentence, and when the sentences have different numbers of names and finds?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie als er niets gevonden is, als een vondst een ander type heeft, als een vondst past bij een naam uit een andere zin, en als de zinnen verschillende aantallen namen en vondsten hebben?"},
            "correct": {"en": "Your `span_f1` is right in all {n} cases. Look at the picture: your span F1 for the five taggers, beside their accuracy per token.",
                        "nl": "Je `span_f1` klopt in alle {n} gevallen. Kijk naar de afbeelding: jouw span F1 voor de vijf taggers, naast de accuracy per token van elke tagger."},
        },
    ),
    "of_kind": FunctionBox(
        name="of_kind",
        picture=kinds_picture,
        function="of_kind",
        start=_OK_START,
        solution=_OK,
        visible=OK_CASES[:1],
        hidden=OK_CASES[1:],
        collect=_as_lists,
        show=show_value,
        first=[_DOTS],
        mistakes=[
            Mistake("flat", wrong=_OK_FLAT,
                    en="`{call}` gives {got}, and it should give {want}: all the spans end up in one list, so it no longer shows which sentence each span is in. How many lists should your function give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: alle spans komen in één lijst, dus je ziet niet meer in welke zin elke span staat. Hoeveel lijsten moet je functie teruggeven?"),
            Mistake("drop-empty", wrong=_OK_DROP,
                    en="`{call}` gives {got}, and it should give {want}: a sentence without a name of that type gets no list, so the lists no longer line up with the sentences. What should such a sentence get?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een zin zonder naam van dat type krijgt geen lijst, dus de lijsten lopen niet meer gelijk met de zinnen. Wat moet zo'n zin krijgen?"),
            Mistake("other-kinds", wrong=_OK_OTHERS,
                    en="`{call}` gives {got}, and it should give {want}: your lists hold the spans of the other types. Which spans should stay?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je lijsten bevatten de spans van de andere types. Welke spans moeten blijven?"),
            Mistake("wrong-item", wrong=_OK_ITEM,
                    en="`{call}` gives {got}, and it should give {want}: every list is empty. Which item of a span holds its type?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke lijst is leeg. Welk item van een span bevat het type?"),
            Mistake("not-list", matches=_not_list,
                    en="`{call}` gives {got}, and it should give {want}: a list with one list of spans for each sentence. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een lijst met één lijst spans voor elke zin. Wat geeft je `return`-regel terug?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back a list with, for each sentence, a list of only the spans of type `kind`.",
                      "nl": "Vervang de `...`: geef een lijst terug met voor elke zin een lijst van alleen de spans van type `kind`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does each sentence get its own list, with only the spans of type `kind`?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Krijgt elke zin een eigen lijst, met alleen de spans van type `kind`?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for no sentences, for a type that no span has, and for a sentence without names?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie zonder zinnen, voor een type dat geen enkele span heeft, en voor een zin zonder namen?"},
            "correct": {"en": "Your `of_kind` is right in all {n} cases. Look at the picture: the span F1 of each type, for the gazetteer and the tagger.",
                        "nl": "Je `of_kind` klopt in alle {n} gevallen. Kijk naar de afbeelding: de span F1 van elk type, voor de gazetteer en de tagger."},
        },
    ),
}

CASES = [
    ("token_accuracy", _TA_START, "empty", None),
    ("token_accuracy", _TA, "correct", None),
    ("token_accuracy", "def token_accuracy(gold, guess):\n    pairs = [(a, b) for s in range(len(gold)) for a, b in zip(gold[s], guess[s])]\n    return len([1 for a, b in pairs if a == b]) / len(pairs)\n", "correct", None),
    ("token_accuracy", _TA_MEAN, "wrong", "per-sentence"),
    ("token_accuracy", _TA_WHOLE, "wrong", "whole-sentence"),
    ("token_accuracy", _TA_BY_SENTENCES, "wrong", "by-sentences"),
    ("token_accuracy", _TA_COUNT, "wrong", "count"),
    ("token_accuracy", _TA_ERRORS, "wrong", "errors"),
    ("token_accuracy", _TA_INSIDE, "wrong", None),
    ("token_accuracy", _TA_NO_TOTAL, "error", "zero"),
    ("token_accuracy", _TA.replace("    return same / total\n", "    print(same / total)\n"), "wrong", "prints"),
    ("token_accuracy", "def token_accuracy(gold, guess):\n    return 0.8\n", "almost", None),
    ("token_accuracy", _TA.replace("    same = 0\n", "    same = 0\n    ...\n"), "wrong", "dots-left"),
    ("span_f1", _SF_START, "empty", None),
    ("span_f1", _SF, "correct", None),
    ("span_f1", _SF.replace("    return (precision, recall, f1)\n", "    return [precision, recall, f1]\n"), "correct", None),
    ("span_f1", "def span_f1(gold, found):\n    right = 0\n    n_found = 0\n    n_gold = 0\n    for s in range(len(gold)):\n        right = right + len([x for x in found[s] if x in gold[s]])\n        n_found = n_found + len(found[s])\n        n_gold = n_gold + len(gold[s])\n    p = right / n_found if n_found else 0.0\n    r = right / n_gold if n_gold else 0.0\n    f = 2 * right / (n_found + n_gold) if n_found + n_gold else 0.0\n    return (p, r, f)\n", "correct", None),
    ("span_f1", _SF_NO_ZERO, "error", "zero"),
    ("span_f1", _SF_POOLED, "wrong", "pooled"),
    ("span_f1", _SF_UNTYPED, "wrong", "untyped"),
    ("span_f1", _SF_OVERLAP, "wrong", "overlap"),
    ("span_f1", _SF_SWAP, "wrong", "swapped"),
    ("span_f1", _SF_MEAN, "wrong", "mean"),
    ("span_f1", _SF_ONLY_F1, "wrong", "not-tuple"),
    ("span_f1", _SF_PER_SENTENCE, "wrong", "per-sentence"),
    ("span_f1", _SF_HARMONIC_HALF, "wrong", "not-f1"),
    ("span_f1", _SF.replace("    f1 = 2 * precision * recall / (precision + recall)\n", "    f1 = precision * recall\n"), "wrong", "not-f1"),
    ("token_accuracy", "def token_accuracy(gold, guess):\n    all_gold = []\n    all_guess = []\n    for s in range(len(gold)):\n        all_gold = all_gold + gold[s]\n        all_guess = all_guess + guess[s]\n    same = 0\n    for k in range(len(all_gold)):\n        if all_gold[k] == all_guess[k]:\n            same = same + 1\n    return same / len(all_gold)\n", "correct", None),
    ("span_f1", "def span_f1(gold, found):\n    right = 0\n    n_found = 0\n    n_gold = 0\n    for s in range(len(gold)):\n        n_found = n_found + len(found[s])\n        n_gold = n_gold + len(gold[s])\n        for span in found[s]:\n            if span in gold[s]:\n                right = right + 1\n                gold[s].remove(span)\n    if right == 0:\n        return (0.0, 0.0, 0.0)\n    p = right / n_found\n    r = right / n_gold\n    return (p, r, 2 * p * r / (p + r))\n", "correct", None),
    ("of_kind", "def of_kind(per_sentence, kind):\n    for i in range(len(per_sentence)):\n        per_sentence[i] = [span for span in per_sentence[i] if span[2] == kind]\n    return per_sentence\n", "correct", None),
    ("span_f1", "def span_f1(gold, found):\n    return (0.5, 1 / 3, 0.4)\n", "almost", None),
    ("span_f1", _SF.replace("    right = 0\n", "    right = 0\n    ...\n"), "wrong", "dots-left"),
    ("of_kind", _OK_START, "empty", None),
    ("of_kind", _OK, "correct", None),
    ("of_kind", "def of_kind(per_sentence, kind):\n    return tuple(tuple(s for s in spans if s[2] == kind) for spans in per_sentence)\n", "correct", None),
    ("of_kind", "def of_kind(per_sentence, kind):\n    out = []\n    for sentence_spans in per_sentence:\n        these = []\n        for span in sentence_spans:\n            if span[2] == kind:\n                these.append(list(span))\n        out.append(these)\n    return out\n", "correct", None),
    ("of_kind", _OK_FLAT, "wrong", "flat"),
    ("of_kind", _OK_DROP, "wrong", "drop-empty"),
    ("of_kind", _OK_OTHERS, "wrong", "other-kinds"),
    ("of_kind", _OK_ITEM, "wrong", "wrong-item"),
    ("of_kind", "def of_kind(per_sentence, kind):\n    return len(per_sentence)\n", "wrong", "not-list"),
    ("of_kind", "def of_kind(per_sentence, kind):\n    return [[(0, 2, 'PER')], []]\n", "almost", None),
    ("of_kind", _OK.replace("    out = []\n", "    out = []\n    ...\n"), "wrong", "dots-left"),
]
