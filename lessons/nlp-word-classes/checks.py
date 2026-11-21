# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-word-classes: count the parts of speech in a treebank, list the tags each word gets, put
words in context under their part of speech, and say which tag follows a determiner. The sentences come from two
treebanks of Universal Dependencies 2.19, UD_English-EWT (without the emails of Enron) and UD_Dutch-Alpino
(data/ud-en-ewt.conllu.gz and data/ud-nl-alpino.conllu.gz, made by data/tools/make_ud.py); the two short sentences
of C.SAMPLE follow examples of Jurafsky and Martin, with tags the course gave them."""
import contextlib
import copy
import gzip
import html
import io
import re

from everything_ai import paths, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, kept, with_time_limit

# --- the course's data -----------------------------------------------------------------------------------------------

UD_EN = paths.ROOT / "data" / "ud-en-ewt.conllu.gz"
UD_NL = paths.ROOT / "data" / "ud-nl-alpino.conllu.gz"

# the 17 universal part-of-speech tags of Universal Dependencies, as its page puts them in three columns
OPEN = ["ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB"]
CLOSED = ["ADP", "AUX", "CCONJ", "DET", "NUM", "PART", "PRON", "SCONJ"]
OTHER = ["PUNCT", "SYM", "X"]
TAGS = OPEN + CLOSED + OTHER


def _read(path) -> tuple:
    """The sentences of a CoNLL-U file of the course: a tuple of (sent_id, sentence), each sentence a tuple of
    (word, tag) pairs."""
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
                words.append((cols[1], cols[3]))
    if words:
        out.append((sent_id, tuple(words)))
    return tuple(out)


_EN = kept("nlp-word-classes:ewt", lambda: _read(UD_EN))
_NL = kept("nlp-word-classes:alpino", lambda: _read(UD_NL))
EWT = tuple(s for _, s in _EN)        # 11,723 English sentences: tuples of (word, tag) pairs
ALPINO = tuple(s for _, s in _NL)     # 13,603 Dutch sentences
_BY_ID = {sid: s for sid, s in _EN + _NL}
EXAMPLE = _BY_ID["reviews-380048-0002"]           # We had a fantastic time .
EXAMPLE_NL = _BY_ID["cdb\\980"]                   # De eerste helft boeide weinig .
# the course's own two sentences, after the examples of Jurafsky and Martin, with the tags of UD
SAMPLE = ((("Book", "VERB"), ("that", "DET"), ("flight", "NOUN"), (".", "PUNCT")),
          (("Hand", "VERB"), ("me", "PRON"), ("that", "DET"), ("book", "NOUN"), (".", "PUNCT")))


def count_tags(sentences) -> dict:
    """The course's own Exercise 1: how many words of the sentences have each tag."""
    counts = {}
    for sentence in sentences:
        for word, tag in sentence:
            if tag in counts:
                counts[tag] = counts[tag] + 1
            else:
                counts[tag] = 1
    return counts


def tags_per_word(sentences) -> dict:
    """The course's own Exercise 2: for each word in small letters, its tags, each once, in the order they come."""
    tags = {}
    for sentence in sentences:
        for word, tag in sentence:
            small = word.lower()
            if small not in tags:
                tags[small] = []
            if tag not in tags[small]:
                tags[small].append(tag)
    return tags


def _index(sentences) -> dict:
    """For each word in small letters, how often it has each tag, the most frequent tag first."""
    counts = {}
    for s in sentences:
        for w, t in s:
            tags = counts.setdefault(w.lower(), {})
            tags[t] = tags.get(t, 0) + 1
    return {w: dict(sorted(c.items(), key=lambda kv: (-kv[1], TAGS.index(kv[0])))) for w, c in counts.items()}


def word_tags(word: str) -> dict:
    """For the demo: how often the word, in small letters, has each tag in EWT ("en") and in Alpino ("nl")."""
    index = {"en": kept("nlp-word-classes:index-en", lambda: _index(EWT)),
             "nl": kept("nlp-word-classes:index-nl", lambda: _index(ALPINO))}
    return {k: index[k].get(word.lower(), {}) for k in index}


def next_tags(sentences, first: str) -> dict:
    """How often each tag comes right after the tag `first`."""
    counts = {}
    for s in sentences:
        for (_, a), (_, b) in zip(s, s[1:]):
            if a == first:
                counts[b] = counts.get(b, 0) + 1
    return counts


class Named:
    """A value the check gives the learner's function, shown in a message by its name. A tuple of sentences cannot be
    changed by the learner's code, so it gets the value itself; a list gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return self.value if isinstance(self.value, tuple) else copy.deepcopy(self.value, memo)


SAMPLE_N = Named("C.SAMPLE", SAMPLE)
EXAMPLE_IN = Named("[C.EXAMPLE]", [EXAMPLE])            # a list with one sentence
EXAMPLE_NL_IN = Named("[C.EXAMPLE_NL]", [EXAMPLE_NL])
# small sentences of the check's own, in English and Dutch
RIVER = ((("Ik", "PRON"), ("loop", "VERB"), ("langs", "ADP"), ("de", "DET"), ("loop", "NOUN"), ("van", "ADP"),
          ("de", "DET"), ("rivier", "NOUN")),)
BACK = ((("Back", "ADV"), ("then", "ADV"), (",", "PUNCT"), ("senators", "NOUN"), ("back", "VERB"), ("the", "DET"),
         ("bill", "NOUN")), (("In", "ADP"), ("the", "DET"), ("back", "NOUN")), (("The", "DET"), ("THE", "DET")))

# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink);vertical-align:middle")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"
_CODE = "font-family:var(--ea-font-code);font-size:12px"
_PART = "display:inline-block;margin:0 0.15rem 0.2rem 0;text-align:center;vertical-align:top"
_MORPH = ("display:block;border-radius:5px;padding:0.2rem 0.45rem;font-family:var(--ea-font-code);font-size:0.95rem;"
          "color:var(--ea-ink)")
_FRAME = {"open": "border:1.5px solid var(--ea-ink-3)", "closed": "border:1.5px dashed var(--ea-ink-3)",
          "other": "border:1.5px solid transparent"}
_LABEL = "display:block;font-family:var(--ea-font-code);font-size:0.72rem;color:var(--ea-ink-2);margin-top:0.15rem"
_GROUP = ("display:inline-block;border-bottom:2px solid var(--ea-ink-3);padding-bottom:0.15rem;margin-right:0.15rem;"
          "vertical-align:top;text-align:center")
_GROUP_LABEL = "display:block;font-family:var(--ea-font-ui);font-size:0.72rem;color:var(--ea-ink-3);margin-top:0.2rem"


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def kind(tag: str) -> str:
    return "open" if tag in OPEN else "closed" if tag in CLOSED else "other"


def _word_box(word: str, tag: str) -> str:
    return (f'<span style="{_PART}"><span style="{_MORPH};{_FRAME[kind(tag)]}">{html.escape(word)}</span>'
            f'<span style="{_LABEL}">{html.escape(tag)}</span></span>')


_TAGGED_WORDS = {
    "en": {"title": "Two sentences of the treebanks, with their tags",
           "en": "English, from UD English-EWT (a review on the web)", "nl": "Dutch, from UD Dutch-Alpino",
           "phrase": "noun phrase",
           "legend": "Solid frame: an open class. Dashed frame: a closed class. No frame: another tag, such as PUNCT. "
                     "The line under a group of words marks a noun phrase."},
    "nl": {"title": "Twee zinnen uit de treebanks, met hun tags",
           "en": "Engels, uit UD English-EWT (een recensie op het web)", "nl": "Nederlands, uit UD Dutch-Alpino",
           "phrase": "naamwoordgroep",
           "legend": "Doorgetrokken kader: een open woordsoort. Gestreept kader: een gesloten woordsoort. Geen kader: "
                     "een andere tag, zoals PUNCT. De lijn onder een groep woorden markeert een naamwoordgroep."},
}
_PHRASES = {"en": (2, 5), "nl": (0, 3)}   # where the noun phrase of each example starts and ends


def tagged_picture(lang: str) -> str:
    """The two example sentences of The idea, each word with its tag, and the noun phrase marked."""
    w = _TAGGED_WORDS[lang]
    rows, said = [], []
    for key, sentence in (("en", EXAMPLE), ("nl", EXAMPLE_NL)):
        a, b = _PHRASES[key]
        before = "".join(_word_box(x, t) for x, t in sentence[:a])
        group = "".join(_word_box(x, t) for x, t in sentence[a:b])
        after = "".join(_word_box(x, t) for x, t in sentence[b:])
        shown = (f'{before}<span style="{_GROUP}"><span style="display:block">{group}</span>'
                 f'<span style="{_GROUP_LABEL}">{html.escape(w["phrase"])}</span></span>{after}')
        rows.append(f'<div style="{_UI};margin-top:0.5rem">{html.escape(w[key])}</div><div>{shown}</div>')
        said.append(f'{w[key]}: ' + " ".join(f"{x}/{t}" for x, t in sentence)
                    + f' ({w["phrase"]}: {" ".join(x for x, _ in sentence[a:b])})')
    return yours.figure(w["title"], "".join(rows) + _note(w["legend"]),
                        alt=f'{w["title"]}. {"; ".join(said)}. {w["legend"]}')


_DEMO_WORDS = {
    "en": {"title": "The tags of {word} in the two treebanks", "empty": "Type a word in the box.",
           "en": "English (EWT)", "nl": "Dutch (Alpino)", "none": "not in this treebank",
           "times": "{n} times", "one": "1 time",
           "note": "The longest bar is the tag the word has most often. Capitals do not count: The and the are the same "
                   "word here."},
    "nl": {"title": "De tags van {word} in de twee treebanks", "empty": "Typ een woord in het vak.",
           "en": "Engels (EWT)", "nl": "Nederlands (Alpino)", "none": "niet in deze treebank",
           "times": "{n} keer", "one": "1 keer",
           "note": "De langste balk is de tag die het woord het vaakst heeft. Hoofdletters tellen niet: De en de zijn "
                   "hier hetzelfde woord."},
}
DEMO_MAX = 6   # tags the demo shows per treebank


def demo_picture(word, lang: str) -> str:
    """A word the learner types, with how often it has each tag in EWT and in Alpino."""
    w = _DEMO_WORDS[lang]
    word = str(word or "").strip()[:40]
    if not word:
        return yours.figure(w["empty"], "", alt=w["empty"])
    found = word_tags(word)
    title = w["title"].format(word=word.lower())
    parts, said = [], []
    for key in ("en", "nl"):
        counts = found[key]
        parts.append(f'<div style="{_UI};margin-top:0.5rem;color:var(--ea-ink-2);font-weight:600">'
                     f'{html.escape(w[key])}</div>')
        if not counts:
            parts.append(_note(w["none"]))
            said.append(f'{w[key]}: {w["none"]}')
            continue
        top = max(counts.values())
        rows = []
        for tag, n in list(counts.items())[:DEMO_MAX]:
            width = 0.3 + 9 * n / top
            bar = (f'<div style="height:0.7rem;width:{width:.2f}rem;background:var(--ea-ink-3);'
                   f'border-radius:2px"></div>')
            count = w["one"] if n == 1 else w["times"].format(n=n)
            rows.append(f'<div style="{_CODE};color:var(--ea-ink)">{tag}</div><div>{bar}</div>'
                        f'<div style="{_UI};margin-top:0">{html.escape(count)}</div>')
        parts.append(f'<div style="display:grid;grid-template-columns:4rem minmax(0,10rem) auto;gap:0.2rem 0.5rem;'
                     f'align-items:center">{"".join(rows)}</div>')
        said.append(f'{w[key]}: ' + ", ".join(f"{t} {n}" for t, n in counts.items()))
    return yours.figure(title, "".join(parts) + _note(w["note"]), alt=f'{title}. {"; ".join(said)}. {w["note"]}')


# --- the pictures under the checks -----------------------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error


def _call(fn, *args):
    """The learner's function on the treebanks within the time limit of a check; _STOPPED when it stops with an error,
    and the Timeout passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(lambda: fn(*(copy.deepcopy(a) for a in args)))
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


_SLOW = {"en": "Your function takes too long on a whole treebank, with about 200,000 words. Does it go over all the words "
                "again for each word?",
         "nl": "Je functie duurt te lang op een hele treebank, met ongeveer 200.000 woorden. Gaat ze voor elk woord "
               "opnieuw alle woorden langs?"}


def _on_treebanks(fn):
    """The learner's function on EWT and on Alpino, or None when it does not finish within the time limit."""
    try:
        return {"en": _call(fn, Named("C.EWT", EWT)), "nl": _call(fn, Named("C.ALPINO", ALPINO))}
    except Timeout:
        return None


def _pct(x: float, lang: str) -> str:
    """A share as a number per 100, with one decimal, and a decimal comma in Dutch."""
    text = f"{100 * x:.1f}"
    return text.replace(".", ",") if lang == "nl" else text


_COUNT_WORDS = {
    "en": {"title": "Your count_tags on the two treebanks", "head": ["tag", "English (EWT)", "Dutch (Alpino)"],
           "per": "per 100 words",
           "fine": "Your counts are the course's. English has PART (to, not) and Dutch none: Alpino uses 16 of the 17 "
                   "tags. Open classes take {o_en} of 100 words in English and {o_nl} in Dutch.",
           "differs": "For {k} of the 17 tags your count differs from the course's: the numbers in those rows have a "
                      "purple dashed frame.",
           "error": "Your function stops with an error on a treebank.", "no_dict": "Your function gives back no dict.",
           "kinds": {"open": "open", "closed": "closed", "other": "other"}},
    "nl": {"title": "Jouw count_tags op de twee treebanks", "head": ["tag", "Engels (EWT)", "Nederlands (Alpino)"],
           "per": "per 100 woorden",
           "fine": "Je tellingen zijn die van de course. Het Engels heeft PART (to, not) en het Nederlands geen: Alpino "
                   "gebruikt 16 van de 17 tags. Open woordsoorten zijn in het Engels goed voor {o_en} van de 100 woorden, "
                   "in het Nederlands voor {o_nl}.",
           "differs": "Bij {k} van de 17 tags verschilt je telling van die van de course: de getallen in die rijen "
                      "hebben een paars gestreept kader.",
           "error": "Je functie stopt met een fout op een treebank.", "no_dict": "Je functie geeft geen dict terug.",
           "kinds": {"open": "open", "closed": "gesloten", "other": "overig"}},
}


def _bar(x: float, top: float) -> str:
    width = 0.15 + 6 * x / top
    return f'<span style="display:inline-block;height:0.6rem;width:{width:.2f}rem;background:var(--ea-ink-3);' \
           f'border-radius:2px;margin-right:0.4rem;vertical-align:middle"></span>'


def count_picture(seen) -> str:
    """The learner's count_tags on EWT and Alpino: each tag per 100 words, beside the course's."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _COUNT_WORDS[seen.lang]
    mine = _on_treebanks(fn)
    if mine is None:
        return yours.figure(w["title"], _note(_SLOW[seen.lang]), alt=f'{w["title"]}. {_SLOW[seen.lang]}')
    want = {"en": count_tags(EWT), "nl": count_tags(ALPINO)}
    if any(m is _STOPPED for m in mine.values()):
        return yours.figure(w["title"], _note(w["error"]), alt=f'{w["title"]}. {w["error"]}')
    if not all(isinstance(m, dict) for m in mine.values()):
        return yours.figure(w["title"], _note(w["no_dict"]), alt=f'{w["title"]}. {w["no_dict"]}')
    num = {k: {t: v for t, v in mine[k].items() if isinstance(v, int) and not isinstance(v, bool)} for k in mine}
    totals = {k: sum(num[k].values()) for k in num}
    share = {k: {t: num[k].get(t, 0) / totals[k] if totals[k] else 0.0 for t in TAGS} for k in num}
    top = max([v for k in share for v in share[k].values()] + [0.01])
    rows, said, k_off = [], [], 0
    for tag in sorted(TAGS, key=lambda t: -want["en"].get(t, 0)):
        off = any(mine[k].get(tag, 0) != want[k].get(tag, 0) for k in mine)
        k_off += off
        cells = [(f'<span style="{_CODE}">{tag}</span> <span style="{_UI}">{w["kinds"][kind(tag)]}</span>', "")]
        for k in ("en", "nl"):
            cells.append((_bar(share[k][tag], top) + f'<span style="{_CODE}">{_pct(share[k][tag], seen.lang)}</span>',
                          _OFF if off else ""))
        rows.append(cells)
        said.append(f'{tag}: {_pct(share["en"][tag], seen.lang)}, {_pct(share["nl"][tag], seen.lang)}')
    head = [w["head"][0], f'{w["head"][1]}, {w["per"]}', f'{w["head"][2]}, {w["per"]}']
    if k_off:
        after = w["differs"].format(k=k_off)
    else:
        o = {k: _pct(sum(want[k].get(t, 0) for t in OPEN) / sum(want[k].values()), seen.lang) for k in want}
        after = w["fine"].format(o_en=o["en"], o_nl=o["nl"])
    return yours.figure(w["title"], _table(rows, head) + _note(after),
                        alt=f'{w["title"]}: {"; ".join(said)}. {after}')


SHOWN_WORDS = {"en": ["back", "that", "to", "book"], "nl": ["loop", "dat", "een", "de"]}

_TAGS_WORDS = {
    "en": {"title": "Your tags_per_word on the two treebanks", "head": ["word", "yours", "the course's"],
           "share": "{k} of every 100 words in the English sentences have more than one tag somewhere in EWT, and {n} "
                    "in the Dutch sentences in Alpino, by your lists.",
           "types": "As different words: {a} of {b} in English and {c} of {d} in Dutch.",
           "differs": "For {k} of the {n} words your list differs from the course's: your lists of those words have a "
                      "purple dashed frame.",
           "error": "Your function stops with an error on a treebank.", "no_dict": "Your function gives back no dict.",
           "missing": "(not in your dict)"},
    "nl": {"title": "Jouw tags_per_word op de twee treebanks", "head": ["woord", "de jouwe", "die van de course"],
           "share": "Van elke 100 woorden in de Engelse zinnen hebben er {k} ergens in EWT meer dan één tag, en in de "
                    "Nederlandse zinnen in Alpino {n}, volgens jouw lijsten.",
           "types": "Als verschillende woorden: {a} van de {b} in het Engels en {c} van de {d} in het Nederlands.",
           "differs": "Bij {k} van de {n} woorden verschilt je lijst van die van de course: je lijsten van die woorden "
                      "hebben een paars gestreept kader.",
           "error": "Je functie stopt met een fout op een treebank.", "no_dict": "Je functie geeft geen dict terug.",
           "missing": "(niet in je dict)"},
}


def _many(value) -> int:
    try:
        return len(value)
    except TypeError:
        return 0


def tags_picture(seen) -> str:
    """The learner's tags_per_word on EWT and Alpino: the tags of some common words beside the course's, and the share
    of words with more than one tag that the learner's lists give."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _TAGS_WORDS[seen.lang]
    mine = _on_treebanks(fn)
    if mine is None:
        return yours.figure(w["title"], _note(_SLOW[seen.lang]), alt=f'{w["title"]}. {_SLOW[seen.lang]}')
    if any(m is _STOPPED for m in mine.values()):
        return yours.figure(w["title"], _note(w["error"]), alt=f'{w["title"]}. {w["error"]}')
    if not all(isinstance(m, dict) for m in mine.values()):
        return yours.figure(w["title"], _note(w["no_dict"]), alt=f'{w["title"]}. {w["no_dict"]}')
    want = {"en": kept("nlp-word-classes:tags-en", lambda: tags_per_word(EWT)),
            "nl": kept("nlp-word-classes:tags-nl", lambda: tags_per_word(ALPINO))}
    rows, said, k = [], [], 0
    shown = [(lang, word) for lang in ("en", "nl") for word in SHOWN_WORDS[lang]]
    for lang, word in shown:
        got, right = mine[lang].get(word, None), want[lang][word]
        differs = got != right
        k += differs
        got_text = w["missing"] if got is None else (", ".join(str(t) for t in got) if isinstance(got, list) else repr(got))
        rows.append([(f'<span style="{_CODE}">{html.escape(word)}</span>', ""),
                     (f'<span style="{_CODE}">{html.escape(got_text)}</span>', _OFF if differs else ""),
                     (f'<span style="{_CODE}">{html.escape(", ".join(right))}</span>', "")])
        said.append(f"{word}: {got_text}; {', '.join(right)}")
    share = {lang: _pct(_safe_share(lang, mine[lang]), seen.lang) for lang in mine}
    types = {lang: (sum(1 for v in mine[lang].values() if _many(v) > 1), len(mine[lang])) for lang in mine}
    lines = [w["share"].format(k=share["en"], n=share["nl"]),
             w["types"].format(a=_num(types["en"][0], seen.lang), b=_num(types["en"][1], seen.lang),
                               c=_num(types["nl"][0], seen.lang), d=_num(types["nl"][1], seen.lang))]
    if k:
        lines.insert(0, w["differs"].format(k=k, n=len(shown)))
    return yours.figure(w["title"], _table(rows, w["head"]) + "".join(_note(x) for x in lines),
                        alt=f'{w["title"]}: {"; ".join(said)}. {" ".join(lines)}')


def _safe_share(lang: str, tags: dict) -> float:
    """The share of the treebank's words with more than one tag in the learner's dict, whatever its values."""
    sentences = EWT if lang == "en" else ALPINO
    total = sum(len(s) for s in sentences)
    many = 0
    for s in sentences:
        for word, _ in s:
            try:
                many += _many(tags.get(word.lower(), ())) > 1
            except TypeError:   # a key that cannot be looked up, such as a list
                pass
    return many / total


def _num(n: int, lang: str) -> str:
    return f"{n:,}".replace(",", "." if lang == "nl" else ",")


def _table(rows: list, head: list) -> str:
    """A table with a head row; each row is a list of (html, style)."""
    top = "".join(f'<th style="{_HEAD}">{html.escape(h)}</th>' for h in head)
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{";" + style if style else ""}">{cell}</td>'
                                    for cell, style in row) + "</tr>" for row in rows)
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
            f'<tr>{top}</tr>{body}</table></div>')


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["count_tags", "tags_per_word", "which", "after"]

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
    return isinstance(e, NameError) and re.search(r"'(EWT|ALPINO|SAMPLE|C)'", str(e)) is not None


_NAMES = Mistake("lesson-names", error=_names_error,
                 en="Your function uses a name of the lesson, such as `C` or `EWT`, and the check does not give it that name: use `sentences`, the name between the parentheses of the `def` line.",
                 nl="Je functie gebruikt een naam van de les, zoals `C` of `EWT`, en de check geeft die naam niet mee: gebruik `sentences`, de naam tussen de haakjes van de `def`-regel.")

# exercise 1: count_tags; the check also gives sentences of its own
COUNT_CASES = [(SAMPLE_N,), (RIVER,), (BACK,), (EXAMPLE_NL_IN,), ((),)]
_C_START = "def count_tags(sentences):\n    ...\n"
_C = ("def count_tags(sentences):\n"
      "    counts = {}\n"
      "    for sentence in sentences:\n"
      "        for word, tag in sentence:\n"
      "            if tag in counts:\n"
      "                counts[tag] = counts[tag] + 1\n"
      "            else:\n"
      "                counts[tag] = 1\n"
      "    return counts\n")
_C_WORDS = _C.replace("if tag in counts:\n                counts[tag] = counts[tag] + 1\n            else:\n"
                      "                counts[tag] = 1\n",
                      "if word in counts:\n                counts[word] = counts[word] + 1\n            else:\n"
                      "                counts[word] = 1\n")
_C_ONE = _C.replace("counts[tag] = counts[tag] + 1", "counts[tag] = 1")
_C_ZERO = _C.replace("            else:\n                counts[tag] = 1\n", "            else:\n                counts[tag] = 0\n")
_C_RESET = ("def count_tags(sentences):\n"
            "    for sentence in sentences:\n"
            "        counts = {}\n"
            "        for word, tag in sentence:\n"
            "            if tag in counts:\n"
            "                counts[tag] = counts[tag] + 1\n"
            "            else:\n"
            "                counts[tag] = 1\n"
            "    return counts\n")
_C_EARLY = _C.replace("    return counts\n", "        return counts\n")

# exercise 2: tags_per_word
TAGS_CASES = [(SAMPLE_N,), (RIVER,), (BACK,), (EXAMPLE_IN,), ((),)]
_T_START = "def tags_per_word(sentences):\n    ...\n"
_T = ("def tags_per_word(sentences):\n"
      "    tags = {}\n"
      "    for sentence in sentences:\n"
      "        for word, tag in sentence:\n"
      "            small = word.lower()\n"
      "            if small not in tags:\n"
      "                tags[small] = []\n"
      "            if tag not in tags[small]:\n"
      "                tags[small].append(tag)\n"
      "    return tags\n")
_T_NOLOWER = ("def tags_per_word(sentences):\n"
              "    tags = {}\n"
              "    for sentence in sentences:\n"
              "        for word, tag in sentence:\n"
              "            if word not in tags:\n"
              "                tags[word] = []\n"
              "            if tag not in tags[word]:\n"
              "                tags[word].append(tag)\n"
              "    return tags\n")
_T_DOUBLE = _T.replace("            if tag not in tags[small]:\n                tags[small].append(tag)\n",
                       "            tags[small].append(tag)\n")
_T_NEW = ("def tags_per_word(sentences):\n"
          "    tags = {}\n"
          "    for sentence in sentences:\n"
          "        for word, tag in sentence:\n"
          "            small = word.lower()\n"
          "            tags[small] = [tag]\n"
          "    return tags\n")
_T_FIRST = ("def tags_per_word(sentences):\n"
            "    tags = {}\n"
            "    for sentence in sentences:\n"
            "        for word, tag in sentence:\n"
            "            small = word.lower()\n"
            "            if small not in tags:\n"
            "                tags[small] = [tag]\n"
            "    return tags\n")
_T_EARLY = _T.replace("    return tags\n", "        return tags\n")


def _sets(got, text) -> bool:
    """The tags as a set and not as a list."""
    return isinstance(got, dict) and any(isinstance(v, set) for v in got.values())


EXERCISES = {
    "count_tags": FunctionBox(
        name="count_tags",
        picture=count_picture,
        function="count_tags",
        start=_C_START,
        solution=_C,
        visible=COUNT_CASES[:1],
        hidden=COUNT_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("words", wrong=_C_WORDS,
                    en="`{call}` gives {got}, and it should give {want}: your dict counts the words. Count the tags: each tag is a key, and its value is how many words have it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je dict telt de woorden. Tel de tags: elke tag is een key, en de waarde is hoeveel woorden die tag hebben."),
            Mistake("one", wrong=_C_ONE,
                    en="`{call}` gives {got}, and it should give {want}: every count stays 1. When the tag is already a key, the new count is the old count and one more.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke telling blijft 1. Is de tag al een key, dan is de nieuwe telling de oude plus één."),
            Mistake("zero", wrong=_C_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: each count is one too low. The first word with a tag already counts: give a new tag the count 1.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke telling is één te laag. Het eerste woord met een tag telt al mee: geef een nieuwe tag de telling 1."),
            Mistake("reset", wrong=_C_RESET,
                    en="`{call}` gives {got}, and it should give {want}: only the last sentence counts. Make the empty dict once, before the loop over the sentences.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: alleen de laatste zin telt mee. Maak de lege dict één keer, vóór de loop over de zinnen."),
            Mistake("early", wrong=_C_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back the dict after the first sentence. Move `return counts` to the left, under `for sentence in ...`, so that it comes after the loop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft de dict al na de eerste zin terug. Zet `return counts` meer naar links, onder `for sentence in ...`, zodat die na de loop komt."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: start with an empty dict, go over every word of every sentence, and count its tag.",
                      "nl": "Vervang de `...`: begin met een lege dict, ga elk woord van elke zin langs, en tel de tag ervan."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The function gives back a dict with each tag that occurs as a key, and as its value the number of words in all the sentences with that tag.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De functie geeft een dict terug met elke tag die voorkomt als key, en als waarde het aantal woorden met die tag in alle zinnen."},
            "almost": {"en": "Right for C.SAMPLE. But `{call}` gives {got}, and it should give {want}. Does your function use the `sentences` it gets, and count every word of every sentence?",
                       "nl": "Goed voor C.SAMPLE. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `sentences` die ze krijgt, en telt ze elk woord van elke zin?"},
            "correct": {"en": "Your `count_tags` is right in all {n} cases, also for no sentences at all. Look at the picture: your function on both treebanks, per 100 words.",
                        "nl": "Je `count_tags` klopt in alle {n} gevallen, ook zonder zinnen. Kijk naar de afbeelding: je functie op beide treebanks, per 100 woorden."},
        },
    ),
    "tags_per_word": FunctionBox(
        name="tags_per_word",
        picture=tags_picture,
        function="tags_per_word",
        start=_T_START,
        solution=_T,
        visible=TAGS_CASES[:1],
        hidden=TAGS_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("set", matches=_sets,
                    en="`{call}` gives {got}, and it should give {want}: the tags of a word are a set. Make them a list, with each tag once, in the order the tags first come.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de tags van een woord zijn een set. Maak er een lijst van, met elke tag één keer, in de volgorde waarin de tags voor het eerst komen."),
            Mistake("no-lower", wrong=_T_NOLOWER,
                    en="`{call}` gives {got}, and it should give {want}: Book and book are two keys. Use each word in small letters, with `lower()`, as the key.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: Book en book zijn twee keys. Gebruik elk woord in kleine letters, met `lower()`, als key."),
            Mistake("double", wrong=_T_DOUBLE,
                    en="`{call}` gives {got}, and it should give {want}: a tag is in a word's list more than once. Add a tag only when it is not in the list yet.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een tag staat meer dan één keer in de lijst van een woord. Voeg een tag alleen toe als die nog niet in de lijst staat."),
            Mistake("new", wrong=_T_NEW,
                    en="`{call}` gives {got}, and it should give {want}: each word keeps only its last tag. Make the empty list only for a word that is not a key yet, and then add to it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elk woord houdt alleen de laatste tag. Maak de lege lijst alleen voor een woord dat nog geen key is, en voeg er daarna aan toe."),
            Mistake("first", wrong=_T_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: each word keeps only its first tag. Also add a later tag of a word that is already a key, when its list does not have that tag yet.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elk woord houdt alleen de eerste tag. Voeg ook een latere tag toe van een woord dat al een key is, als de lijst die tag nog niet heeft."),
            Mistake("early", wrong=_T_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back the dict after the first sentence. Move `return tags` to the left, under `for sentence in ...`, so that it comes after the loop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft de dict al na de eerste zin terug. Zet `return tags` meer naar links, onder `for sentence in ...`, zodat die na de loop komt."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: start with an empty dict, and give each word in small letters a list of its tags, each tag once.",
                      "nl": "Vervang de `...`: begin met een lege dict, en geef elk woord in kleine letters een lijst met de tags ervan, elke tag één keer."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The function gives back a dict with each word in small letters as a key, and as its value the list of the tags it has in the sentences, each tag once, in the order they first come.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De functie geeft een dict terug met elk woord in kleine letters als key, en als waarde de lijst met de tags die het in de zinnen heeft, elke tag één keer, in de volgorde waarin ze voor het eerst komen."},
            "almost": {"en": "Right for C.SAMPLE. But `{call}` gives {got}, and it should give {want}. Does your function use the `sentences` it gets, and add every new tag of a word?",
                       "nl": "Goed voor C.SAMPLE. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `sentences` die ze krijgt, en voegt ze elke nieuwe tag van een woord toe?"},
            "correct": {"en": "Your `tags_per_word` is right in all {n} cases. Look at the picture: your lists on both treebanks, and how many words in running text have more than one tag.",
                        "nl": "Je `tags_per_word` klopt in alle {n} gevallen. Kijk naar de afbeelding: je lijsten op beide treebanks, en hoeveel woorden in lopende tekst meer dan één tag hebben."},
        },
    ),
    "which": Sort(
        name="which",
        groups={
            "NOUN": {"en": "NOUN", "nl": "NOUN"},
            "VERB": {"en": "VERB", "nl": "VERB"},
            "ADV": {"en": "ADV", "nl": "ADV"},
        },
        items={
            "set-verb": {"en": "In *we set the table*, the word **set**",
                         "nl": "In *we set the table* (wij dekken de tafel), het woord **set**"},
            "set-noun": {"en": "In *a new chess set*, the word **set**",
                         "nl": "In *a new chess set* (een nieuw schaakspel), het woord **set**"},
            "back-adv": {"en": "In *I was twenty-one back then*, the word **back**",
                         "nl": "In *I was twenty-one back then* (ik was toen eenentwintig), het woord **back**"},
            "back-verb": {"en": "In *a clear majority of senators back the bill*, the word **back**",
                          "nl": "In *a clear majority of senators back the bill* (een duidelijke meerderheid van de senatoren steunt het wetsvoorstel), het woord **back**"},
            "back-noun": {"en": "In *a small building in the back*, the word **back**",
                          "nl": "In *a small building in the back* (een klein gebouw achteraan), het woord **back**"},
            "loop-verb": {"en": "In *ik loop naar huis* (I walk home), the word **loop**",
                          "nl": "In *ik loop naar huis*, het woord **loop**"},
            "loop-noun": {"en": "In *de loop van de rivier* (the course of the river), the word **loop**",
                          "nl": "In *de loop van de rivier*, het woord **loop**"},
        },
        solution={"set-verb": "VERB", "set-noun": "NOUN", "back-adv": "ADV", "back-verb": "VERB", "back-noun": "NOUN",
                  "loop-verb": "VERB", "loop-noun": "NOUN"},
        nudge={
            "set-verb": {"en": "What do we do with the table?",
                         "nl": "Wat doen wij met de tafel?"},
            "set-noun": {"en": "Is set here a thing you can buy?",
                         "nl": "Is set hier een ding dat je kunt kopen?"},
            "back-adv": {"en": "Does back here say when, together with then?",
                         "nl": "Zegt back hier samen met then wanneer?"},
            "back-verb": {"en": "What do the senators do with the bill?",
                          "nl": "Wat doen de senatoren met het wetsvoorstel?"},
            "back-noun": {"en": "Is back here a place, coming after the?",
                          "nl": "Is back hier een plek, na the?"},
            "loop-verb": {"en": "What does ik do here?",
                          "nl": "Wat doet ik hier?"},
            "loop-noun": {"en": "Does de come before a thing here, as in de weg?",
                          "nl": "Staat de hier vóór een ding, zoals in de weg?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the words again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de woorden opnieuw uit."},
            "empty": {"en": "Put each word in bold under its tag: NOUN, VERB or ADV.",
                      "nl": "Zet elk vetgedrukt woord onder de tag ervan: NOUN, VERB of ADV."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs under another tag.",
                      "nl": "{item} hoort onder een andere tag."},
            "correct": {"en": "All {n} are right. The same letters were a noun, a verb or an adverb: the words around them decided.",
                        "nl": "Alle {n} kloppen. Dezelfde letters waren een zelfstandig naamwoord, een werkwoord of een bijwoord: de woorden eromheen beslisten."},
        },
    ),
}

CASES = [
    ("count_tags", _C_START, "empty", None),
    ("count_tags", _C, "correct", None),
    ("count_tags", "def count_tags(sentences):\n    counts = {}\n    for sentence in sentences:\n        for pair in sentence:\n            tag = pair[1]\n            if tag not in counts:\n                counts[tag] = 0\n            counts[tag] = counts[tag] + 1\n    return counts\n", "correct", None),
    ("count_tags", "def count_tags(sentences):\n    tags = [tag for sentence in sentences for word, tag in sentence]\n    return {tag: tags.count(tag) for tag in tags}\n", "correct", None),
    ("count_tags", _C_WORDS, "wrong", "words"),
    ("count_tags", _C_ONE, "wrong", "one"),
    ("count_tags", _C_ZERO, "wrong", "zero"),
    ("count_tags", _C_RESET, "wrong", "reset"),
    ("count_tags", _C_EARLY, "wrong", "early"),
    ("count_tags", "def count_tags(sentences):\n    return {'VERB': 2, 'DET': 2, 'NOUN': 2, 'PUNCT': 2, 'PRON': 1}\n", "almost", None),
    ("count_tags", "def count_tags(sentences):\n    return count_tags(EWT)\n", "error", "lesson-names"),
    ("tags_per_word", _T_START, "empty", None),
    ("tags_per_word", _T, "correct", None),
    ("tags_per_word", "def tags_per_word(sentences):\n    tags = {}\n    for sentence in sentences:\n        for word, tag in sentence:\n            if word.lower() in tags:\n                if tag not in tags[word.lower()]:\n                    tags[word.lower()] = tags[word.lower()] + [tag]\n            else:\n                tags[word.lower()] = [tag]\n    return tags\n", "correct", None),
    ("tags_per_word", "def tags_per_word(sentences):\n    tags = {}\n    for sentence in sentences:\n        for word, tag in sentence:\n            small = word.lower()\n            if small not in tags:\n                tags[small] = set()\n            tags[small].add(tag)\n    return tags\n", "wrong", "set"),
    ("tags_per_word", _T_NOLOWER, "wrong", "no-lower"),
    ("tags_per_word", _T_DOUBLE, "wrong", "double"),
    ("tags_per_word", _T_NEW, "wrong", "new"),
    ("tags_per_word", _T_FIRST, "wrong", "first"),
    ("tags_per_word", _T_EARLY, "wrong", "early"),
    ("tags_per_word", "def tags_per_word(sentences):\n    return {'book': ['VERB', 'NOUN'], 'that': ['DET'], 'flight': ['NOUN'], '.': ['PUNCT'], 'hand': ['VERB'], 'me': ['PRON']}\n", "almost", None),
    ("tags_per_word", "def tags_per_word(sentences):\n    return tags_per_word(C.EWT)\n", "error", "lesson-names"),
    ("which", {}, "empty", None),
    ("which", {"set-verb": "VERB", "set-noun": "NOUN", "back-adv": "ADV", "back-verb": "VERB", "back-noun": "NOUN",
               "loop-verb": "VERB", "loop-noun": "NOUN"}, "correct", None),
    ("which", {"set-verb": "VERB", "set-noun": "NOUN"}, "wrong", "unfinished"),
    ("which", {"set-verb": "NOUN"}, "wrong", "wrong:set-verb"),
    ("which", {"set-verb": "VERB", "set-noun": "NOUN", "back-adv": "NOUN", "back-verb": "VERB", "back-noun": "NOUN",
               "loop-verb": "VERB", "loop-noun": "NOUN"}, "wrong", "wrong:back-adv"),
]
