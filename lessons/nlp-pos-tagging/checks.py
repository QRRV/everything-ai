# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-pos-tagging: keep the most frequent tag of each word (best_tags), measure the share of
words a tagger gets right (accuracy), and tag a sentence with the word before each word (tag_words); with the pictures
of what each function gives on the treebanks, two pictures of the idea and a demo.

The sentences come from two treebanks of Universal Dependencies 2.19, UD_English-EWT (without the emails of Enron)
and UD_Dutch-Alpino (data/ud-en-ewt.conllu.gz and data/ud-nl-alpino.conllu.gz, made by data/tools/make_ud.py), with
their own split into a training set and a test set, so nothing is downloaded. Every number in the text and the
pictures is computed by the course (tests/test_nlp_pos_tagging.py computes them again)."""
import contextlib
import copy
import gzip
import html
import io
from types import MappingProxyType

from everything_ai import paths, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, kept, with_time_limit

_KEY = "nlp-pos-tagging"

# --- the course's data -----------------------------------------------------------------------------------------------

UD_EN = paths.ROOT / "data" / "ud-en-ewt.conllu.gz"
UD_NL = paths.ROOT / "data" / "ud-nl-alpino.conllu.gz"
UNKNOWN = "NOUN"   # the tag of a word the training set does not have: the most frequent tag in both training sets


def _read(path) -> tuple:
    """The sentences of a CoNLL-U file of the course: a tuple of (sent_id, split, sentence), each sentence a tuple of
    (word, tag) pairs."""
    out, words, sent_id, split = [], [], None, None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# sent_id = "):
                sent_id = line[len("# sent_id = "):]
            elif line.startswith("# split = "):
                split = line[len("# split = "):]
            elif not line:
                if words:
                    out.append((sent_id, split, tuple(words)))
                words = []
            elif not line.startswith("#"):
                cols = line.split("\t")
                words.append((cols[1], cols[3]))
    if words:
        out.append((sent_id, split, tuple(words)))
    return tuple(out)


_EN = kept("nlp-pos-tagging:ewt", lambda: _read(UD_EN))
_NL = kept("nlp-pos-tagging:alpino", lambda: _read(UD_NL))
TRAIN_EN = tuple(s for _, split, s in _EN if split == "train")   # 8,774 English sentences: tuples of (word, tag) pairs
TEST_EN = tuple(s for _, split, s in _EN if split == "test")     # 1,471
TRAIN_NL = tuple(s for _, split, s in _NL if split == "train")   # 12,289 Dutch sentences
TEST_NL = tuple(s for _, split, s in _NL if split == "test")     # 596
_BY_ID = {sid: s for sid, _, s in _EN + _NL}
TRAIN = {"en": TRAIN_EN, "nl": TRAIN_NL}
TEST = {"en": TEST_EN, "nl": TEST_NL}
TREEBANK_NAMES = {"en": "UD_English-EWT", "nl": "UD_Dutch-Alpino"}


def words_of(sentence) -> list:
    return [w for w, _ in sentence]


def tags_of(sentence) -> list:
    return [t for _, t in sentence]


def tag_counts(sentences) -> dict:
    """The worked example: for each word, how often it has each tag, with the words as they are written."""
    counts = {}
    for sentence in sentences:
        for word, tag in sentence:
            if word not in counts:
                counts[word] = {}
            if tag not in counts[word]:
                counts[word][tag] = 0
            counts[word][tag] = counts[word][tag] + 1
    return counts


def pair_counts(sentences) -> dict:
    """For each pair (previous word, word), how often the word has each tag after that previous word. The first word
    of a sentence has no word before it, so it is in no pair."""
    counts = {}
    for sentence in sentences:
        for i in range(1, len(sentence)):
            key = (sentence[i - 1][0], sentence[i][0])
            tag = sentence[i][1]
            if key not in counts:
                counts[key] = {}
            counts[key][tag] = counts[key].get(tag, 0) + 1
    return counts


def next_counts(sentences) -> dict:
    """As pair_counts, with the word after in place of the word before: for each pair (word, next word), how often
    the word has each tag before that next word. The last word of a sentence is in no pair."""
    counts = {}
    for sentence in sentences:
        for i in range(len(sentence) - 1):
            key = (sentence[i][0], sentence[i + 1][0])
            tag = sentence[i][1]
            if key not in counts:
                counts[key] = {}
            counts[key][tag] = counts[key].get(tag, 0) + 1
    return counts


def best_tags(counts) -> dict:
    """The course's own Exercise 1: for each word, its most frequent tag; of two tags that tie, the one counted
    first."""
    best = {}
    for word in counts:
        top = None
        for tag in counts[word]:
            if top is None or counts[word][tag] > counts[word][top]:
                top = tag
        best[word] = top
    return best


def accuracy(sentences, best) -> float:
    """The course's own Exercise 2: the share of words whose tag in best, or NOUN for a word best does not have, is the
    tag of the treebank."""
    right = 0
    total = 0
    for sentence in sentences:
        for word, tag in sentence:
            if word in best:
                guess = best[word]
            else:
                guess = UNKNOWN
            if guess == tag:
                right = right + 1
            total = total + 1
    return right / total


def tag_words(words, best, pairs) -> list:
    """The course's own Exercise 3: the tag of the pair (previous word, word) when pairs has it, else the tag of the
    word in best, else NOUN."""
    tags = []
    for i in range(len(words)):
        word = words[i]
        if i > 0 and (words[i - 1], word) in pairs:
            tags.append(pairs[(words[i - 1], word)])
        elif word in best:
            tags.append(best[word])
        else:
            tags.append(UNKNOWN)
    return tags


def tag_next(words, best, nexts) -> list:
    """As tag_words, with the word after in place of the word before."""
    tags = []
    for i in range(len(words)):
        word = words[i]
        if i + 1 < len(words) and (word, words[i + 1]) in nexts:
            tags.append(nexts[(word, words[i + 1])])
        elif word in best:
            tags.append(best[word])
        else:
            tags.append(UNKNOWN)
    return tags


def tag_first(words, best) -> list:
    """The first tagger: the most frequent tag of each word, NOUN for a word it does not know."""
    return [best[w] if w in best else UNKNOWN for w in words]


COUNTS = {"en": kept("nlp-pos-tagging:counts-en", lambda: tag_counts(TRAIN_EN)),
          "nl": kept("nlp-pos-tagging:counts-nl", lambda: tag_counts(TRAIN_NL))}
BEST_EN = MappingProxyType(best_tags(COUNTS["en"]))   # word -> its most frequent tag in TRAIN_EN
BEST_NL = MappingProxyType(best_tags(COUNTS["nl"]))
PAIRS_EN = MappingProxyType(best_tags(kept("nlp-pos-tagging:pairs-en", lambda: pair_counts(TRAIN_EN))))
PAIRS_NL = MappingProxyType(best_tags(kept("nlp-pos-tagging:pairs-nl", lambda: pair_counts(TRAIN_NL))))
BEST = {"en": BEST_EN, "nl": BEST_NL}
PAIRS = {"en": PAIRS_EN, "nl": PAIRS_NL}
_NEXT = {"en": best_tags(kept("nlp-pos-tagging:next-en", lambda: next_counts(TRAIN_EN))),
         "nl": best_tags(kept("nlp-pos-tagging:next-nl", lambda: next_counts(TRAIN_NL)))}

# the course's own two sentences, after the examples of Jurafsky and Martin, with the tags of UD, and a small table of
# most frequent tags that has no Hand and gives that the tag SCONJ
SAMPLE = ((("Book", "VERB"), ("that", "DET"), ("flight", "NOUN"), (".", "PUNCT")),
          (("Hand", "VERB"), ("me", "PRON"), ("that", "DET"), ("book", "NOUN"), (".", "PUNCT")))
SAMPLE_BEST = MappingProxyType({"Book": "VERB", "that": "SCONJ", "flight": "NOUN", ".": "PUNCT", "me": "PRON",
                                "book": "NOUN"})


def kind_of(word: str, lang: str) -> str:
    """unknown: not in the training set; one: one tag there; more: more than one tag there."""
    counts = COUNTS[lang].get(word)
    if counts is None:
        return "unknown"
    return "one" if len(counts) == 1 else "more"


def numbers(lang: str) -> dict:
    """The numbers of the lesson for one treebank."""
    train, test, counts, best, pairs = TRAIN[lang], TEST[lang], COUNTS[lang], BEST[lang], PAIRS[lang]
    total_tags = {}
    for c in counts.values():
        for t, n in c.items():
            total_tags[t] = total_tags.get(t, 0) + n
    out = {"train_sentences": len(train), "train_words": sum(len(s) for s in train),
           "test_sentences": len(test), "test_words": sum(len(s) for s in test),
           "types": len(counts), "more_types": sum(1 for c in counts.values() if len(c) > 1),
           "more_train_words": sum(n for c in counts.values() if len(c) > 1 for n in c.values()),
           "top_tag": max(total_tags, key=total_tags.get), "top_tag_words": max(total_tags.values()),
           "ties": sum(1 for c in counts.values() if list(c.values()).count(max(c.values())) > 1)}
    groups = {k: [0, 0, 0] for k in ("unknown", "one", "more")}   # words, right with the first tagger, with pairs
    whole = [0, 0]   # sentences with every tag right: first tagger, with pairs
    for s in test:
        words, tags = words_of(s), tags_of(s)
        first, paired = tag_first(words, best), tag_words(words, best, pairs)
        whole[0] += first == tags
        whole[1] += paired == tags
        for w, t, f, p in zip(words, tags, first, paired):
            g = groups[kind_of(w, lang)]
            g[0] += 1
            g[1] += f == t
            g[2] += p == t
    n = out["test_words"]
    out |= {"groups": groups, "first": sum(g[1] for g in groups.values()) / n,
            "paired": sum(g[2] for g in groups.values()) / n, "whole_first": whole[0], "whole_paired": whole[1]}
    return out


def word_in_test(word: str, lang: str) -> tuple:
    """How often the first tagger, the tagger with the word before and the tagger with the word after get `word`
    right in the test set, and how often it occurs there."""
    n = first = before = after = 0
    best, pairs, nexts = BEST[lang], PAIRS[lang], _NEXT[lang]
    for s in TEST[lang]:
        words, tags = words_of(s), tags_of(s)
        a, b, c = tag_first(words, best), tag_words(words, best, pairs), tag_next(words, best, nexts)
        for i, w in enumerate(words):
            if w == word:
                n += 1
                first += a[i] == tags[i]
                before += b[i] == tags[i]
                after += c[i] == tags[i]
    return n, first, before, after


# --- pictures: words and their tags ----------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_HEAD_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);font-weight:600;margin-top:0.6rem"
_CODE = "font-family:var(--ea-font-code);font-size:0.78rem"
_PART = "display:inline-block;margin:0 0.2rem 0.3rem 0;text-align:center;vertical-align:top"
_WORD = ("display:block;border-radius:5px;padding:0.15rem 0.4rem;font-family:var(--ea-font-code);font-size:0.9rem;"
         "color:var(--ea-ink)")
_WORD_BORDER = {"known": "border:1px solid var(--ea-line)", "unknown": "border:1.5px dotted var(--ea-ink-3)"}
_TAG = "display:block;font-family:var(--ea-font-code);font-size:0.7rem;margin-top:0.15rem;padding:0 0.15rem"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:1px;border-radius:3px"
_ROLE = {   # the color of a tag: the treebank's, the course's tagger, the learner's, or wrong
    "data": "var(--ea-ink-3)",
    "course": "var(--ea-ink)",
    "changed": "var(--ea-ink);font-weight:700",
    "mine": "var(--ea-pred-text)",
    "off": "var(--ea-loss-text)",
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
    """A whole number with a separator for thousands: 16,599 in English, 16.599 in Dutch."""
    text = f"{x:,}"
    return text.replace(",", ".") if lang == "nl" else text


def _pct(x: float, lang: str, digits: int = 1) -> str:
    """A share as a number of every 100: 86.2 in English, 86,2 in Dutch."""
    text = f"{100 * x:.{digits}f}"
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


def _shown_tag(tag) -> str:
    """A tag as text, short: anything that is not a string is shown as Python shows it."""
    text = tag if isinstance(tag, str) else _repr(tag)
    return text if len(text) <= 12 else text[:11] + "…"


def _column(word: str, known: bool, tags: list) -> str:
    """One word in its box, dotted when the training set does not have it, with tags under it: each (tag, role)."""
    lines = []
    for tag, role in tags:
        off = ";" + _OFF if role == "off" else ""
        lines.append(f'<span style="{_TAG};color:{_ROLE[role]}{off}">{html.escape(_shown_tag(tag))}</span>')
    border = _WORD_BORDER["known" if known else "unknown"]
    return (f'<span style="{_PART}"><span style="{_WORD};{border}">{html.escape(word)}</span>'
            + "".join(lines) + "</span>")


def _row(words: list, known: list, tag_rows: list) -> str:
    """A sentence as columns: each word with its tags under it, one list of (tag, role) per row of tags."""
    return "<div>" + "".join(_column(w, k, [r[i] for r in tag_rows]) for i, (w, k) in enumerate(zip(words, known))) \
        + "</div>"


def _figure(title: str, rows: list, said: list, notes: list) -> str:
    body = "".join(rows) + f'<div style="{_UI}">{" ".join(h for h, _ in notes)}</div>'
    return yours.figure(title, body, alt=f'{title}. {"; ".join(said)}. {" ".join(p for _, p in notes)}')


# the test sentences of the pictures: the idea's first tagger, and the learner's tag_words
IDEA = {"en": _BY_ID["reviews-061079-0001"],                                              # Friendliest place I have ever stayed !
        "nl": _BY_ID["WR-P-P-H-0000000055\\WR-P-P-H-0000000055.p.3.s.5"]}                 # Meestal wil je weten of ...
SHOWN = {"en": [_BY_ID["weblog-blogspot.com_marketview_20050511222700_ENG_20050511_222700-0006"],   # Is that a money maker ?
                _BY_ID["newsgroup-groups.google.com_JokeEruption_df151b356f94881c_ENG_20050819_155700-0023"]],  # I should have asked for a jury .
         "nl": [_BY_ID["WR-P-P-L-0000000003\\WR-P-P-L-0000000003.p.21.s.2"],             # In dat platform zullen ...
                _BY_ID["WR-P-P-H-0000000073\\WR-P-P-H-0000000073.p.4.s.3"]]}             # Gezamenlijk , zoals dat ...

_IDEA_WORDS = {
    "en": {"title": "The first tagger on a sentence of the test set",
           "rows": "Under each word: the tag in the treebank (gray), then under it the tag of the first tagger.",
           "off": "Purple dashed: a tag of the tagger that differs from the treebank.",
           "unknown": "Dotted box: a word the training set does not have, so it gets NOUN.",
           "count": "{right} of the {n} tags are right."},
    "nl": {"title": "De eerste tagger op een zin uit de test set",
           "rows": "Onder elk woord: de tag in de treebank (grijs), en daaronder de tag van de eerste tagger.",
           "off": "Paars gestreept: een tag van de tagger die verschilt van de treebank.",
           "unknown": "Gestippeld vak: een woord dat niet in de training set staat, en dus NOUN krijgt.",
           "count": "{right} van de {n} tags kloppen."},
}


def idea_picture(lang: str) -> str:
    """The first tagger on one sentence of the test set: an unknown word and a word with more than one tag go wrong."""
    w = _IDEA_WORDS[lang]
    sentence = IDEA[lang]
    words, want = words_of(sentence), tags_of(sentence)
    got = tag_first(words, BEST[lang])
    known = [x in BEST[lang] for x in words]
    rows = [_row(words, known, [[(t, "data") for t in want], [(g, "course" if g == t else "off") for g, t in zip(got, want)]])]
    right = sum(g == t for g, t in zip(got, want))
    notes = [_said(w["rows"]), _said(w["off"]), _said(w["unknown"]), _said(w["count"], right=right, n=len(words))]
    said = [" ".join(f"{x}/{t}/{g}" for x, t, g in zip(words, want, got))]
    return _figure(w["title"], rows, said, notes)


# the words before *to* and *dat* in the picture of pairs, and the word itself
PAIRS_WORD = {"en": "to", "nl": "dat"}
PAIRS_BEFORE = {"en": ["want", "have", "going", "go", "it"], "nl": ["zei", "zonder", "en", "maar", "op"]}
_PAIRS_WORDS = {
    "en": {"title": "The tags of ‘{word}’ in the training set, after the word before it",
           "before": "word before", "tags": "tags of *{word}* after it", "gets": "the pair gets",
           "alone": "(any word)",
           "note": "The last row counts *{word}* after any word: the first tagger gives it {tag} every time. The rows "
                   "above count it after one word only, and each pair gets its own most frequent tag."},
    "nl": {"title": "De tags van ‘{word}’ in de training set, na het woord ervoor",
           "before": "woord ervoor", "tags": "tags van *{word}* erna", "gets": "het paar krijgt",
           "alone": "(elk woord)",
           "note": "De laatste rij telt *{word}* na elk woord: de eerste tagger geeft het steeds {tag}. De rijen erboven "
                   "tellen het alleen na één woord, en elk paar krijgt de eigen vaakste tag."},
}


def _counts_text(counts: dict, lang: str) -> str:
    order = sorted(counts, key=lambda t: -counts[t])
    return " · ".join(f"{t} {_n(counts[t], lang)}" for t in order)


def pairs_picture(lang: str) -> str:
    """How often *to* (English) or *dat* (Dutch) has each tag after a few words before it, and after any word."""
    w = _PAIRS_WORDS[lang]
    word = PAIRS_WORD[lang]
    pc = kept(f"nlp-pos-tagging:pairs-{lang}", lambda: pair_counts(TRAIN[lang]))
    head = (f'<tr><th style="{_HEAD_CELL}">{_marked(w["before"])}</th>'
            f'<th style="{_HEAD_CELL}">{_marked(w["tags"].replace("{word}", word))}</th>'
            f'<th style="{_HEAD_CELL}">{_marked(w["gets"])}</th></tr>')
    rows, said = [], []
    for before in PAIRS_BEFORE[lang]:
        counts = pc[(before, word)]
        rows.append(f'<tr><td style="{_CELL};{_CODE}">{html.escape(before)}</td><td style="{_CELL};{_CODE}">'
                    f'{html.escape(_counts_text(counts, lang))}</td><td style="{_CELL};{_CODE};color:var(--ea-ink)">'
                    f'{PAIRS[lang][(before, word)]}</td></tr>')
        said.append(f"{before} {word}: {_counts_text(counts, lang)}")
    counts = COUNTS[lang][word]
    rows.append(f'<tr><td style="{_CELL};color:var(--ea-ink-3)">{html.escape(w["alone"])}</td>'
                f'<td style="{_CELL};{_CODE}">{html.escape(_counts_text(counts, lang))}</td>'
                f'<td style="{_CELL};{_CODE};color:var(--ea-ink)">{BEST[lang][word]}</td></tr>')
    said.append(f'{w["alone"]} {word}: {_counts_text(counts, lang)}')
    table = f'<table style="border-collapse:collapse;margin-top:0.3rem">{head}{"".join(rows)}</table>'
    note = _said(w["note"], word=word, tag=BEST[lang][word])
    title = _bare(w["title"]).replace("{word}", word)
    return yours.figure(title, table + f'<div style="{_UI}">{note[0]}</div>', alt=f'{title}. {"; ".join(said)}. {note[1]}')


# --- the demo --------------------------------------------------------------------------------------------------------

DEMO_MAX = 30   # words the demo tags
DEMO_TEXT = {"en": "I want to book a flight to Paris .", "nl": "In dat geval denk ik dat het werkt ."}
_DEMO_WORDS = {
    "en": {"title": "The two taggers on your sentence",
           "rows": "Under each word: the tag of the first tagger, then the tag of the tagger with the word before. A "
                   "bold tag is one the word before changed.",
           "unknown": "Dotted box: a word the training set does not have.",
           "changed": "The word before changes {k} of the {n} tags.",
           "same": "The word before changes none of the {n} tags.",
           "cut": "The demo tags the first {max} words.",
           "empty": "Type a sentence, with spaces between the words and before punctuation."},
    "nl": {"title": "De twee taggers op jouw zin",
           "rows": "Onder elk woord: de tag van de eerste tagger, dan de tag van de tagger met het woord ervoor. Een "
                   "vetgedrukte tag is een tag die door het woord ervoor veranderde.",
           "unknown": "Gestippeld vak: een woord dat niet in de training set staat.",
           "changed": "Het woord ervoor verandert {k} van de {n} tags.",
           "same": "Het woord ervoor verandert geen van de {n} tags.",
           "cut": "De demo tagt de eerste {max} woorden.",
           "empty": "Typ een zin, met spaties tussen de woorden en voor leestekens."},
}


def demo_picture(text, lang: str) -> str:
    """The first tagger and the tagger with the word before on a sentence the learner types, cut at the spaces, with
    the tables of the lesson's own treebank."""
    w = _DEMO_WORDS[lang]
    words = str(text or "").split()
    if not words:
        return yours.figure(w["title"], _note(w["empty"]), alt=f'{w["title"]}. {w["empty"]}')
    notes = []
    if len(words) > DEMO_MAX:
        words = words[:DEMO_MAX]
        notes.append(_said(w["cut"], max=DEMO_MAX))
    best, pairs = BEST[lang], PAIRS[lang]
    first, paired = tag_first(words, best), tag_words(words, best, pairs)
    known = [x in best for x in words]
    rows = [_row(words, known, [[(t, "course") for t in first],
                                [(p, "changed" if p != f else "course") for f, p in zip(first, paired)]])]
    changed = sum(f != p for f, p in zip(first, paired))
    notes.insert(0, _said(w["changed"], k=changed, n=len(words)) if changed else _said(w["same"], n=len(words)))
    notes[:0] = [_said(w["rows"])]
    if not all(known):
        notes.append(_said(w["unknown"]))
    said = [" ".join(f"{x}/{f}/{p}" for x, f, p in zip(words, first, paired))]
    return _figure(w["title"], rows, said, notes)


# --- running the learner's function in a picture ---------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _quiet gives for a function that stopped with an error
PICTURE_SECONDS = 3   # what the learner's function gets for one treebank of a picture


def _quiet(fn, args):
    """The learner's function without what it prints; _STOPPED when it stops with an error."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*args)
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


_TOO_SLOW = object()   # what _timed gives when the learner's function takes too long; None is a value of theirs


def _timed(make):
    """make() within PICTURE_SECONDS, or _TOO_SLOW when it takes longer."""
    try:
        return with_time_limit(make, PICTURE_SECONDS)
    except Timeout:
        return _TOO_SLOW


def _number(v):
    """A number of the learner's as a float, or None when it is not a plain number."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        return float(v)
    except OverflowError:   # a whole number too large for a float lies far outside 0 to 1
        return float("inf") if v > 0 else float("-inf")


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 + 1e-6 * abs(b)


_SLOW = {"en": "Your function takes too long on the {name} treebank. Does a loop of yours go on longer than it needs to?",
         "nl": "Je functie duurt te lang op de treebank {name}. Gaat een van je loops langer door dan nodig is?"}
_STOP = {"en": "Your function stops with an error on the {name} treebank.",
         "nl": "Je functie stopt met een fout op de treebank {name}."}
_ORDER = {"en": ("en", "nl"), "nl": ("nl", "en")}   # the lesson's own treebank first

# exercise 1: the words whose counts the picture shows
SHOWN_WORDS = {"en": ["that", "to", "like", "back"], "nl": ["dat", "te", "er", "een"]}
_BEST_WORDS = {
    "en": {"title": "Your best_tags on the counts of the training sets",
           "word": "word", "counts": "its tags in the training set", "yours": "your tag",
           "missing": "(none)",
           "fine": "Your table gives every one of the {n} words of {name} a most frequent tag; for {ties} of them two "
                   "or more tags tie, and any of them counts.",
           "count": "Your table gives {good} of the {n} words of {name} a most frequent tag.",
           "not_dict": "On {name}, your function gives back {got} instead of a dict.",
           "blue": "Blue: your tag, a most frequent one. Gray: the counts of the training set.",
           "off": "Purple dashed: a tag of yours that is not the most frequent one, or no tag at all."},
    "nl": {"title": "Jouw best_tags op de tellingen van de training sets",
           "word": "woord", "counts": "de tags in de training set", "yours": "jouw tag",
           "missing": "(geen)",
           "fine": "Je tabel geeft elk van de {n} woorden van {name} een vaakste tag; bij {ties} woorden staan twee of "
                   "meer tags gelijk, en dan telt elk ervan.",
           "count": "Je tabel geeft {good} van de {n} woorden van {name} een vaakste tag.",
           "not_dict": "Op {name} geeft je functie {got} terug in plaats van een dict.",
           "blue": "Blauw: jouw tag, een van de vaakste. Grijs: de tellingen van de training set.",
           "off": "Paars gestreept: een tag van jou die niet de vaakste is, of helemaal geen tag."},
}


def _top(counts: dict) -> set:
    most = max(counts.values())
    return {t for t, n in counts.items() if n == most}


def best_picture(seen) -> str:
    """The learner's best_tags on the counts of both training sets: four words with their counts and the learner's
    tag, and how many words get a most frequent tag."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _BEST_WORDS[lang]
    notes, said, rows, off = [], [], [], False
    for which in _ORDER[lang]:
        name = TREEBANK_NAMES[which]
        counts = COUNTS[which]
        got = _timed(lambda: _quiet(fn, (copy.deepcopy(counts),)))
        if got is _TOO_SLOW:
            notes.append(_said(_SLOW[lang], name=name))
            continue
        if got is _STOPPED:
            notes.append(_said(_STOP[lang], name=name))
            continue
        if not isinstance(got, dict):
            notes.append(_said(w["not_dict"], name=name, got=_short(got)))
            continue
        good = 0
        for word, c in counts.items():
            try:
                good += word in got and got[word] in _top(c)
            except Exception:   # noqa: BLE001   a tag of the learner's that cannot be compared
                pass
        ties = sum(1 for c in counts.values() if len(_top(c)) > 1)
        if good == len(counts):
            notes.append(_said(w["fine"], n=_n(len(counts), lang), name=name, ties=_n(ties, lang)))
        else:
            notes.append(_said(w["count"], good=_n(good, lang), n=_n(len(counts), lang), name=name))
        if which != lang:
            continue
        head = (f'<tr><th style="{_HEAD_CELL}">{_marked(w["word"])}</th><th style="{_HEAD_CELL}">'
                f'{_marked(w["counts"])}</th><th style="{_HEAD_CELL}">{_marked(w["yours"])}</th></tr>')
        lines = []
        for word in SHOWN_WORDS[lang]:
            c = counts[word]
            tag = got.get(word, _STOPPED) if hasattr(got, "get") else _STOPPED
            fine = tag is not _STOPPED and isinstance(tag, str) and tag in _top(c)
            off = off or not fine
            shown = w["missing"] if tag is _STOPPED else _shown_tag(tag)
            style = f"color:{_ROLE['mine']}" if fine else f"color:{_ROLE['off']};{_OFF}"
            lines.append(f'<tr><td style="{_CELL};{_CODE}">{html.escape(word)}</td>'
                         f'<td style="{_CELL};{_CODE};color:var(--ea-ink-3)">{html.escape(_counts_text(c, lang))}</td>'
                         f'<td style="{_CELL};{_CODE}"><span style="{style};padding:0 0.2rem">{html.escape(shown)}'
                         f'</span></td></tr>')
            said.append(f"{word}: {_counts_text(c, lang)}; {shown}")
        rows.append(f'<table style="border-collapse:collapse;margin-top:0.3rem">{head}{"".join(lines)}</table>')
    extra = [_said(w["blue"])] + ([_said(w["off"])] if off else [])
    return _figure(w["title"], rows, said, notes + extra)


# exercise 2
_ACC_WORDS = {
    "en": {"title": "Your accuracy for the first tagger on the test sets",
           "bar": "{name}: {value}", "course": "the course: {value}",
           "not_number": "On {name}, your function gives back {got} instead of a number.",
           "outside": "On {name}, your function gives {got}: a share lies between 0 and 1, so the bar stops at the end.",
           "blue": "Blue: your share, drawn from 0 to 1, with the number of every 100 beside it.",
           "off": "Purple dashed: your share differs from the course's; the gray line marks the course's.",
           "groups": "The words of the {name} test set in three groups, with what the first tagger gets right (gray: the "
                     "course's count)",
           "group": "group", "words": "words", "share": "of every 100 words", "right": "of these right",
           "one": "one tag in the training set", "more": "more than one tag", "unknown": "not in the training set"},
    "nl": {"title": "Jouw accuracy voor de eerste tagger op de test sets",
           "bar": "{name}: {value}", "course": "de course: {value}",
           "not_number": "Op {name} geeft je functie {got} terug in plaats van een getal.",
           "outside": "Op {name} geeft je functie {got}: een aandeel ligt tussen 0 en 1, dus de balk stopt aan het eind.",
           "blue": "Blauw: jouw aandeel, getekend van 0 tot 1, met het aantal per 100 ernaast.",
           "off": "Paars gestreept: je aandeel verschilt van dat van de course; de grijze streep geeft dat van de course aan.",
           "groups": "De woorden van de test set van {name} in drie groepen, met wat de eerste tagger goed heeft (grijs: "
                     "de telling van de course)",
           "group": "groep", "words": "woorden", "share": "van elke 100 woorden", "right": "daarvan goed",
           "one": "één tag in de training set", "more": "meer dan één tag", "unknown": "niet in de training set"},
}
_BAR = "position:relative;height:0.9rem;width:100%;max-width:420px;background:var(--ea-line);border-radius:3px;margin:0.15rem 0 0.4rem"


def _bar(v: float, want: float, fine: bool) -> str:
    width = 100 * min(max(v, 0.0), 1.0)
    fill = (f"background:var(--ea-pred)" if fine else f"background:var(--ea-loss);{_OFF}")
    tick = "" if fine else (f'<span style="position:absolute;left:{100 * want:.2f}%;top:-0.15rem;bottom:-0.15rem;'
                            f'width:2px;background:var(--ea-ink-3)"></span>')
    return (f'<div style="{_BAR}"><span style="position:absolute;left:0;top:0;bottom:0;width:{width:.2f}%;'
            f'border-radius:3px;{fill}"></span>{tick}</div>')


def groups_table(lang: str) -> tuple:
    """The course's three groups of the words of a test set, as an HTML table and as plain text."""
    w = _ACC_WORDS[lang]
    g = numbers(lang)["groups"]
    n = sum(x[0] for x in g.values())
    head = (f'<tr><th style="{_HEAD_CELL}">{_marked(w["group"])}</th><th style="{_HEAD_CELL}">{_marked(w["words"])}</th>'
            f'<th style="{_HEAD_CELL}">{_marked(w["share"])}</th><th style="{_HEAD_CELL}">{_marked(w["right"])}</th></tr>')
    lines, said = [], []
    for key in ("one", "more", "unknown"):
        words, right, _ = g[key]
        cells = [w[key], _n(words, lang), _pct(words / n, lang), _pct(right / words, lang)]
        lines.append("<tr>" + "".join(f'<td style="{_CELL};color:var(--ea-ink-3)">{html.escape(c)}</td>' for c in cells)
                     + "</tr>")
        said.append(f"{cells[0]}: {cells[1]}, {cells[2]}, {cells[3]}")
    title = _bare(w["groups"]).replace("{name}", TREEBANK_NAMES[lang])
    return (_head(w["groups"].replace("{name}", TREEBANK_NAMES[lang]))
            + f'<table style="border-collapse:collapse;margin-top:0.2rem">{head}{"".join(lines)}</table>',
            f'{title}: {"; ".join(said)}')


def accuracy_picture(seen) -> str:
    """The learner's accuracy for the first tagger on both test sets, as bars, and the course's three groups of
    words."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _ACC_WORDS[lang]
    rows, said, notes, off = [], [], [], False
    for which in _ORDER[lang]:
        name = TREEBANK_NAMES[which]
        got = _timed(lambda: _quiet(fn, (TEST[which], BEST[which])))
        if got is _TOO_SLOW:
            notes.append(_said(_SLOW[lang], name=name))
            continue
        if got is _STOPPED:
            notes.append(_said(_STOP[lang], name=name))
            continue
        v = _number(got)
        if v is None or v != v:
            notes.append(_said(w["not_number"], name=name, got=_short(got)))
            continue
        want = accuracy(TEST[which], BEST[which])
        fine = _close(v, want)
        off = off or not fine
        if not 0 <= v <= 1:
            notes.append(_said(w["outside"], name=name, got=_short(got)))
        value = _pct(v, lang) if 0 <= v <= 1 else _short(got)
        label = _said(w["bar"], name=name, value=value)
        if not fine:
            label = (label[0] + " · " + _said(w["course"], value=_pct(want, lang))[0],
                     label[1] + "; " + _said(w["course"], value=_pct(want, lang))[1])
        rows.append(f'<div style="{_UI};color:var(--ea-ink-2)">{label[0]}</div>' + _bar(v, want, fine))
        said.append(label[1])
    table, plain = groups_table(lang)
    rows.append(table)
    said.append(plain)
    extra = [_said(w["blue"])] + ([_said(w["off"])] if off else [])
    return _figure(w["title"], rows, said, notes + extra)


# exercise 3
_TAG_WORDS = {
    "en": {"title": "Your tag_words on sentences of the test set",
           "rows": "Under each word: the tag in the treebank (gray), then your tag.",
           "length": "For a sentence of {n} words, your function gives back {got} instead of a list of {n} tags.",
           "total": "On the whole {name} test set, {right} of the {n} words get the tag of the treebank: {pct} of every "
                    "100, and {first} with the first tagger. Of the words with more than one tag in the training set: "
                    "{more} of every 100, and {more_first} with the first tagger.",
           "differs": "On {name}, your tags differ from those of the course's tagger for {k} words.",
           "case": "In the check's case `{call}`, your function gives {got}, and the course's tagger {want}.",
           "blue": "Blue: your tag, the same as in the treebank.",
           "off": "Purple dashed: a tag of yours that differs from the treebank."},
    "nl": {"title": "Jouw tag_words op zinnen uit de test set",
           "rows": "Onder elk woord: de tag in de treebank (grijs), dan jouw tag.",
           "length": "Voor een zin van {n} woorden geeft je functie {got} terug in plaats van een lijst van {n} tags.",
           "total": "Op de hele test set van {name} krijgen {right} van de {n} woorden de tag van de treebank: {pct} van "
                    "elke 100, en {first} met de eerste tagger. Van de woorden met meer dan één tag in de training set: "
                    "{more} van elke 100, en {more_first} met de eerste tagger.",
           "differs": "Op {name} verschillen je tags voor {k} woorden van de tags van de tagger van de course.",
           "case": "In het geval `{call}` van de check geeft je functie {got}, en de tagger van de course {want}.",
           "blue": "Blauw: jouw tag, gelijk aan de tag in de treebank.",
           "off": "Paars gestreept: een tag van jou die verschilt van de treebank."},
}


def _as_tags(v, n: int):
    if not isinstance(v, (list, tuple)) or len(v) != n:
        return None
    return list(v)


def tagwords_picture(seen) -> str:
    """The learner's tag_words on two sentences of the test set, each tag beside the treebank's, and the share right on
    both whole test sets."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _TAG_WORDS[lang]
    rows, said, notes, off, drawn, off_course = [], [], [], False, False, False
    for sentence in SHOWN[lang]:
        words, want = words_of(sentence), tags_of(sentence)
        got = _timed(lambda: _quiet(fn, (list(words), BEST[lang], PAIRS[lang])))
        if got is _TOO_SLOW:
            notes.append(_said(_SLOW[lang], name=TREEBANK_NAMES[lang]))
            break
        if got is _STOPPED:
            continue
        tags = _as_tags(got, len(words))
        if tags is None:
            note = _said(w["length"], n=len(words), got=_short(got))
            if note not in notes:
                notes.append(note)
            continue
        roles = ["mine" if t == want[i] else "off" for i, t in enumerate(tags)]
        off = off or "off" in roles
        off_course = off_course or tags != tag_words(words, BEST[lang], PAIRS[lang])
        drawn = True
        known = [x in BEST[lang] for x in words]
        rows.append(_row(words, known, [[(t, "data") for t in want], list(zip(tags, roles))]))
        said.append(" ".join(f"{x}/{t}/{_shown_tag(g)}" for x, t, g in zip(words, want, tags)))
    for which in _ORDER[lang]:
        name = TREEBANK_NAMES[which]
        best, pairs = BEST[which], PAIRS[which]
        mine = _timed(lambda: [_quiet(fn, (words_of(s), best, pairs)) for s in TEST[which]])
        if mine is _TOO_SLOW:
            notes.append(_said(_SLOW[lang], name=name))
            continue
        if any(g is _STOPPED for g in mine):
            notes.append(_said(_STOP[lang], name=name))
        right = more = more_first = more_n = n = differs = 0
        for g, s in zip(mine, TEST[which]):
            words, want = words_of(s), tags_of(s)
            tags = _as_tags(g, len(words)) if g is not _STOPPED else None
            course, first = tag_words(words, best, pairs), tag_first(words, best)
            for i, x in enumerate(words):
                n += 1
                t = tags[i] if tags is not None else _STOPPED
                right += t == want[i]
                differs += t != course[i]
                if kind_of(x, which) == "more":
                    more_n += 1
                    more += t == want[i]
                    more_first += first[i] == want[i]
        firsts = numbers(which)["first"]
        notes.append(_said(w["total"], name=name, right=_n(right, lang), n=_n(n, lang), pct=_pct(right / n, lang),
                           first=_pct(firsts, lang), more=_pct(more / more_n, lang),
                           more_first=_pct(more_first / more_n, lang)))
        if differs:
            notes.append(_said(w["differs"], name=name, k=_n(differs, lang)))
    if not off_course:   # the two sentences show no difference from the course's tagger: the first small case that does
        for case in TW_CASES[:6]:
            args = [a.value if isinstance(a, Named) else a for a in case]
            got = _timed(lambda: _quiet(fn, (list(args[0]), dict(args[1]), dict(args[2]))))
            want = tag_words(*args)
            if got is not _TOO_SLOW and got is not _STOPPED and _listed(got) != want:
                call = "tag_words(" + ", ".join(_case_code(a) for a in case) + ")"
                notes.append(_said(w["case"], call=call, got=_short(got), want=_short(want)))
                break
    extra = ([_said(w["rows"]), _said(w["blue"])] if drawn else []) + ([_said(w["off"])] if off else [])
    return _figure(w["title"], rows, said, extra + notes)


def _case_code(a) -> str:
    if isinstance(a, Named):
        return a.name
    if isinstance(a, list):
        return "[" + ", ".join(f'"{x}"' for x in a) + "]"
    return _repr(a)


# --- the checks ------------------------------------------------------------------------------------------------------

REQUIRED = ["best_tags", "accuracy", "tag_words", "after"]


class Named:
    """A value the check gives the learner's function, shown in a message by its name. A tuple of sentences or a
    read-only table cannot be changed by the learner's code, so it gets the value itself; anything else gets a copy."""

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

_DOTS = Mistake("dots-left", matches=lambda got, text: "..." in "\n".join(
                    line.split("#")[0] for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang de `...` door je eigen code.")


def _key_error(e) -> bool:
    return isinstance(e, KeyError)


def _not_dict(got, text) -> bool:
    return got is not None and not isinstance(got, dict)


def _not_number(got, text) -> bool:
    return got is not None and _number(got) is None


def _not_list(got, text) -> bool:
    return got is not None and not isinstance(got, (list, tuple))


def _listed(v):
    """What tag_words gives, with a tuple of tags made a list, as the picture under the check reads it."""
    return list(v) if isinstance(v, tuple) else v


# exercise 1: best_tags. The visible case: that and book, with the most frequent tag neither first nor last nor last in
# the alphabet; then no words, one word with one tag, the most frequent tag first, and two words where the second has
# lower counts than the first.
BT_CASES = [
    (Named('{"that": {"PRON": 641, "SCONJ": 733, "DET": 123}, "book": {"VERB": 6, "NOUN": 18}}',
           {"that": {"PRON": 641, "SCONJ": 733, "DET": 123}, "book": {"VERB": 6, "NOUN": 18}}),),
    ({},),
    ({"the": {"DET": 6730}},),
    ({"a": {"DET": 3000, "X": 2, "NOUN": 1}},),
    ({"run": {"VERB": 40, "NOUN": 12}, "walk": {"NOUN": 30, "VERB": 9}},),
]
_BT_HEAD = "def best_tags(counts):\n"
_BT_START = _BT_HEAD + "    ...\n"
_BT = (_BT_HEAD
       + "    best = {}\n"
       + "    for word in counts:\n"
       + '        top = ""\n'
       + "        most = 0\n"
       + "        for tag in counts[word]:\n"
       + "            if counts[word][tag] > most:\n"
       + "                most = counts[word][tag]\n"
       + "                top = tag\n"
       + "        best[word] = top\n"
       + "    return best\n")
_BT_CARRY = (_BT_HEAD
             + "    best = {}\n"
             + '    top = ""\n'
             + "    most = 0\n"
             + "    for word in counts:\n"
             + "        for tag in counts[word]:\n"
             + "            if counts[word][tag] > most:\n"
             + "                most = counts[word][tag]\n"
             + "                top = tag\n"
             + "        best[word] = top\n"
             + "    return best\n")
_BT_COUNT = _BT.replace("        best[word] = top\n", "        best[word] = most\n")
_BT_LAST = _BT.replace("                most = counts[word][tag]\n", "")
_BT_ABC = (_BT_HEAD
           + "    best = {}\n"
           + "    for word in counts:\n"
           + "        best[word] = max(counts[word])\n"
           + "    return best\n")
_BT_EARLY = _BT.replace("        best[word] = top\n    return best\n", "        best[word] = top\n        return best\n")

# exercise 2: accuracy. The visible case: the two sentences of C.SAMPLE with C.SAMPLE_BEST; then no table at all, a
# word in the table, an unknown word that is a noun, two sentences of different length, and both whole test sets.
AC_CASES = [
    (Named("C.SAMPLE", SAMPLE), Named("C.SAMPLE_BEST", SAMPLE_BEST)),
    (Named("C.SAMPLE", SAMPLE), {}),
    ([[("Paris", "PROPN")]], {"Paris": "PROPN"}),
    ([[("glorp", "NOUN")]], {}),
    ([[("go", "VERB"), ("home", "ADV")], [("ok", "INTJ")]], {"go": "VERB", "home": "NOUN"}),
    (Named("C.TEST_EN", TEST_EN), Named("C.BEST_EN", BEST_EN)),
    (Named("C.TEST_NL", TEST_NL), Named("C.BEST_NL", BEST_NL)),
]
_AC_HEAD = "def accuracy(sentences, best):\n"
_AC_START = _AC_HEAD + "    ...\n"
_AC = (_AC_HEAD
       + "    right = 0\n"
       + "    total = 0\n"
       + "    for sentence in sentences:\n"
       + "        for word, tag in sentence:\n"
       + "            if word in best:\n"
       + "                guess = best[word]\n"
       + "            else:\n"
       + '                guess = "NOUN"\n'
       + "            if guess == tag:\n"
       + "                right = right + 1\n"
       + "            total = total + 1\n"
       + "    return right / total\n")
_AC_NO_DEFAULT = (_AC_HEAD
                  + "    right = 0\n"
                  + "    total = 0\n"
                  + "    for sentence in sentences:\n"
                  + "        for word, tag in sentence:\n"
                  + "            if best[word] == tag:\n"
                  + "                right = right + 1\n"
                  + "            total = total + 1\n"
                  + "    return right / total\n")
_AC_SKIP = (_AC_HEAD
            + "    right = 0\n"
            + "    total = 0\n"
            + "    for sentence in sentences:\n"
            + "        for word, tag in sentence:\n"
            + "            if word in best:\n"
            + "                if best[word] == tag:\n"
            + "                    right = right + 1\n"
            + "                total = total + 1\n"
            + "    return right / total\n")
_AC_UNKNOWN_WRONG = (_AC_HEAD
                     + "    right = 0\n"
                     + "    total = 0\n"
                     + "    for sentence in sentences:\n"
                     + "        for word, tag in sentence:\n"
                     + "            if word in best and best[word] == tag:\n"
                     + "                right = right + 1\n"
                     + "            total = total + 1\n"
                     + "    return right / total\n")
_AC_COUNT = _AC.replace("    return right / total\n", "    return right\n")
_AC_PERCENT = _AC.replace("    return right / total\n", "    return 100 * right / total\n")
_AC_ERRORS = _AC.replace("    return right / total\n", "    return (total - right) / total\n")
_AC_SENTENCES = _AC.replace("    return right / total\n", "    return right / len(sentences)\n")
_AC_RESET = _AC.replace("    total = 0\n    for sentence in sentences:\n",
                        "    for sentence in sentences:\n        total = 0\n")
_AC_PER_SENTENCE = (_AC_HEAD
                    + "    shares = 0\n"
                    + "    for sentence in sentences:\n"
                    + "        right = 0\n"
                    + "        for word, tag in sentence:\n"
                    + "            if word in best:\n"
                    + "                guess = best[word]\n"
                    + "            else:\n"
                    + '                guess = "NOUN"\n'
                    + "            if guess == tag:\n"
                    + "                right = right + 1\n"
                    + "        shares = shares + right / len(sentence)\n"
                    + "    return shares / len(sentences)\n")

# exercise 3: tag_words. The visible case: go to school, with a pair for (go, to); then to at the start, a pair for
# the word after, no words, an unknown word, a pair for a word the table of words does not have, and a sentence of each
# treebank with the tables of the course.
_GO = {"go": "VERB", "to": "PART"}
TW_CASES = [
    (["go", "to", "school"], Named('{"go": "VERB", "to": "PART"}', _GO),
     Named('{("go", "to"): "ADP"}', {("go", "to"): "ADP"})),
    (["to", "go"], Named('{"to": "PART", "go": "VERB"}', {"to": "PART", "go": "VERB"}),
     Named('{("go", "to"): "ADP"}', {("go", "to"): "ADP"})),
    (["I", "go", "to", "work"], Named('{"I": "PRON", "go": "VERB", "to": "PART", "work": "NOUN"}',
                                      {"I": "PRON", "go": "VERB", "to": "PART", "work": "NOUN"}),
     Named('{("go", "to"): "ADP", ("to", "work"): "VERB"}', {("go", "to"): "ADP", ("to", "work"): "VERB"})),
    ([], {}, {}),
    (["glorp"], {}, {}),
    (["will", "glorp"], {"will": "AUX"}, Named('{("will", "glorp"): "VERB"}', {("will", "glorp"): "VERB"})),
    (["I", "want", "to", "book", "a", "flight", "to", "Paris", "."], Named("C.BEST_EN", BEST_EN),
     Named("C.PAIRS_EN", PAIRS_EN)),
    (["In", "dat", "geval", "denk", "ik", "dat", "het", "werkt", "."], Named("C.BEST_NL", BEST_NL),
     Named("C.PAIRS_NL", PAIRS_NL)),
]
_TW_HEAD = "def tag_words(words, best, pairs):\n"
_TW_START = _TW_HEAD + "    ...\n"
_TW = (_TW_HEAD
       + "    tags = []\n"
       + "    for i in range(len(words)):\n"
       + "        word = words[i]\n"
       + "        if i > 0 and (words[i - 1], word) in pairs:\n"
       + "            tags.append(pairs[(words[i - 1], word)])\n"
       + "        elif word in best:\n"
       + "            tags.append(best[word])\n"
       + "        else:\n"
       + '            tags.append("NOUN")\n'
       + "    return tags\n")
_TW_NO_PAIRS = _TW.replace("        if i > 0 and (words[i - 1], word) in pairs:\n"
                           "            tags.append(pairs[(words[i - 1], word)])\n"
                           "        elif word in best:\n",
                           "        if word in best:\n")
_TW_WRAP = _TW.replace("if i > 0 and (words[i - 1], word) in pairs:", "if (words[i - 1], word) in pairs:")
_TW_REVERSED = _TW.replace("(words[i - 1], word)", "(word, words[i - 1])")
_TW_NEXT = (_TW.replace("if i > 0 and (words[i - 1], word) in pairs:", "if i + 1 < len(words) and (word, words[i + 1]) in pairs:")
            .replace("tags.append(pairs[(words[i - 1], word)])", "tags.append(pairs[(word, words[i + 1])])"))
_TW_ORDER = (_TW_HEAD
             + "    tags = []\n"
             + "    for i in range(len(words)):\n"
             + "        word = words[i]\n"
             + "        if word in best:\n"
             + "            tags.append(best[word])\n"
             + "        elif i > 0 and (words[i - 1], word) in pairs:\n"
             + "            tags.append(pairs[(words[i - 1], word)])\n"
             + "        else:\n"
             + '            tags.append("NOUN")\n'
             + "    return tags\n")
_TW_NO_DEFAULT = (_TW_HEAD
                  + "    tags = []\n"
                  + "    for i in range(len(words)):\n"
                  + "        word = words[i]\n"
                  + "        if i > 0 and (words[i - 1], word) in pairs:\n"
                  + "            tags.append(pairs[(words[i - 1], word)])\n"
                  + "        else:\n"
                  + "            tags.append(best[word])\n"
                  + "    return tags\n")

EXERCISES = {
    "best_tags": FunctionBox(
        name="best_tags",
        picture=best_picture,
        function="best_tags",
        start=_BT_START,
        solution=_BT,
        visible=BT_CASES[:1],
        hidden=BT_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("key", error=_key_error,
                    en="With `{call}`, your function stops with a KeyError: it looks up a key that a dict does not have. Which word or tag do you look up there, and does that dict have it?",
                    nl="Met `{call}` stopt je functie met een KeyError: je functie zoekt een key op die niet in een dict staat. Welk woord of welke tag zoek je daar op, en staat dat in die dict?"),
            Mistake("count", wrong=_BT_COUNT,
                    en="`{call}` gives {got}, and it should give {want}: your table holds a count for each word, and not a tag. What does the tagger have to look up for a word?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je tabel bevat voor elk woord een aantal, en geen tag. Wat moet de tagger voor een woord kunnen opzoeken?"),
            Mistake("carry", wrong=_BT_CARRY,
                    en="`{call}` gives {got}, and it should give {want}: a word whose counts are all lower than the highest count of a word before it keeps the tag of that earlier word. When do your best tag so far and its count start over?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een woord met tellingen die allemaal lager zijn dan de hoogste telling van een woord ervoor, houdt de tag van dat eerdere woord. Wanneer beginnen je beste tag tot nu toe en de telling ervan opnieuw?"),
            Mistake("last", wrong=_BT_LAST,
                    en="`{call}` gives {got}, and it should give {want}: each word gets the last tag of its counts, whatever its count. When a tag becomes the best so far, what happens to the count that the next tags have to beat?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elk woord krijgt de laatste tag van de tellingen, wat het aantal ook is. Als een tag de beste tot nu toe wordt, wat gebeurt er dan met het aantal dat de volgende tags moeten verslaan?"),
            Mistake("abc", wrong=_BT_ABC,
                    en="`{call}` gives {got}, and it should give {want}: `max` of a dict compares its keys, the tags, by the alphabet, so VERB wins from NOUN whatever the counts. Which numbers should decide?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `max` van een dict vergelijkt de keys, de tags, op het alfabet, dus VERB wint van NOUN, wat de aantallen ook zijn. Welke getallen moeten beslissen?"),
            Mistake("early", wrong=_BT_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your table holds only the first word. When does your `return` run: after the loop over all the words, or during the first one?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je tabel bevat alleen het eerste woord. Wanneer draait je `return`: na de loop over alle woorden, of al bij het eerste woord?"),
            Mistake("not-dict", matches=_not_dict,
                    en="`{call}` gives {got}, and it should give {want}: a dict with a tag for each word. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een dict met een tag voor elk woord. Wat geeft je `return`-regel terug?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: go over the words of `counts`, and give each word the tag with its highest count.",
                      "nl": "Vervang de `...`: ga de woorden van `counts` langs, en geef elk woord de tag met de hoogste telling."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. For each word, which tag has the highest count?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Welke tag heeft bij elk woord de hoogste telling?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for any number of words, also none, and wherever the highest count stands?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elk aantal woorden, ook nul, en waar de hoogste telling ook staat?"},
            "correct": {"en": "Your `best_tags` is right in all {n} cases. Look at the picture: your table on the counts of both training sets.",
                        "nl": "Je `best_tags` klopt in alle {n} gevallen. Kijk naar de afbeelding: je tabel op de tellingen van beide training sets."},
        },
    ),
    "accuracy": FunctionBox(
        name="accuracy",
        picture=accuracy_picture,
        function="accuracy",
        start=_AC_START,
        solution=_AC,
        visible=AC_CASES[:1],
        hidden=AC_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("no-default", error=_key_error,
                    en="With `{call}`, your function stops with a KeyError: some words of the sentences are not in `best`. What does the tagger guess for a word it never saw?",
                    nl="Met `{call}` stopt je functie met een KeyError: sommige woorden van de zinnen staan niet in `best`. Wat gokt de tagger voor een onbekend woord?"),
            Mistake("floor", wrong=_AC.replace("right / total", "right // total"),
                    en="`{call}` gives {got}, and it should give {want}: `//` rounds down to a whole number, so every share below 1 becomes 0. Which division keeps the part after the point?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `//` rondt af naar beneden op een heel getal, dus elk aandeel onder 1 wordt 0. Welke deling houdt het deel na de komma?"),
            Mistake("skip", wrong=_AC_SKIP,
                    en="`{call}` gives {got}, and it should give {want}: your function leaves out the words that `best` does not have, so the hardest words of the tagger do not count. Which words of the sentences should count?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie laat de woorden weg die niet in `best` staan, dus de moeilijkste woorden van de tagger tellen niet mee. Welke woorden van de zinnen moeten meetellen?"),
            Mistake("unknown-wrong", wrong=_AC_UNKNOWN_WRONG,
                    en="`{call}` gives {got}, and it should give {want}: a word that `best` does not have always counts as wrong, also when the treebank gives it NOUN. What does the tagger guess for such a word?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een woord dat niet in `best` staat, telt altijd als fout, ook als de treebank het NOUN geeft. Wat gokt de tagger voor zo'n woord?"),
            Mistake("count", wrong=_AC_COUNT,
                    en="`{call}` gives {got}, and it should give {want}: your function gives how many words are right, and not which part of all the words that is. How many words are there in all?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft hoeveel woorden goed zijn, en niet welk deel van alle woorden dat is. Hoeveel woorden zijn er in totaal?"),
            Mistake("percent", wrong=_AC_PERCENT,
                    en="`{call}` gives {got}, and it should give {want}: your share is a number of every 100, and the check wants a share between 0 and 1. Which number in your `return` line makes it 100 times bigger?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je aandeel is een aantal per 100, en de check wil een aandeel tussen 0 en 1. Welk getal in je `return`-regel maakt het 100 keer zo groot?"),
            Mistake("errors", wrong=_AC_ERRORS,
                    en="`{call}` gives {got}, and it should give {want}: your function gives the share of words the tagger gets wrong. Which words should it count?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft het aandeel woorden dat de tagger fout heeft. Welke woorden moet je functie tellen?"),
            Mistake("sentences", wrong=_AC_SENTENCES,
                    en="`{call}` gives {got}, and it should give {want}: your function divides by the number of sentences, and accuracy is about words. How many words did you count?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie deelt door het aantal zinnen, en accuracy gaat over woorden. Hoeveel woorden heb je geteld?"),
            Mistake("reset", wrong=_AC_RESET,
                    en="`{call}` gives {got}, and it should give {want}: your count of all the words starts over at each sentence, so at the end it holds only the words of the last sentence. Where should that count start at 0?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telling van alle woorden begint bij elke zin opnieuw, dus aan het eind bevat de telling alleen de woorden van de laatste zin. Waar moet die telling op 0 beginnen?"),
            Mistake("per-sentence", wrong=_AC_PER_SENTENCE,
                    en="`{call}` gives {got}, and it should give {want}: your function takes the share of each sentence, and then the mean of those shares, so a short sentence weighs as much as a long one. What do you get when you count the words of all the sentences together?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie neemt het aandeel van elke zin, en dan het gemiddelde van die aandelen, dus een korte zin weegt even zwaar als een lange. Wat krijg je als je de woorden van alle zinnen samen telt?"),
            Mistake("not-number", matches=_not_number,
                    en="`{call}` gives {got}, and it should give {want}: one number, the share of words the tagger gets right. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: één getal, het aandeel woorden dat de tagger goed heeft. Wat geeft je `return`-regel terug?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: go over every word of every sentence, guess its tag with `best`, and give back the share of words whose guess is the tag of the treebank.",
                      "nl": "Vervang de `...`: ga elk woord van elke zin langs, gok de tag met `best`, en geef het aandeel woorden terug waarvan de gok de tag van de treebank is."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. How many words does the tagger get right, and out of how many?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Hoeveel woorden heeft de tagger goed, en van hoeveel?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function count every word, also the words that `best` does not have, and sentences of any length?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Telt je functie elk woord, ook de woorden die niet in `best` staan, en zinnen van elke lengte?"},
            "correct": {"en": "Your `accuracy` is right in all {n} cases, also on both whole test sets. Look at the picture: your share for each test set, and the words in three groups.",
                        "nl": "Je `accuracy` klopt in alle {n} gevallen, ook op beide hele test sets. Kijk naar de afbeelding: je aandeel voor elke test set, en de woorden in drie groepen."},
        },
    ),
    "tag_words": FunctionBox(
        name="tag_words",
        picture=tagwords_picture,
        function="tag_words",
        start=_TW_START,
        solution=_TW,
        visible=TW_CASES[:1],
        hidden=TW_CASES[1:],
        collect=_listed,
        first=[_DOTS],
        mistakes=[
            Mistake("no-default", error=_key_error,
                    en="With `{call}`, your function stops with a KeyError: it looks up a word or a pair that its dict does not have. Which of your lookups is it, and what does your code do when that table does not have the key?",
                    nl="Met `{call}` stopt je functie met een KeyError: je functie zoekt een woord of een paar op dat niet in de dict staat. Bij welke van je opzoekingen gebeurt dat, en wat doet je code als die tabel de key niet heeft?"),
            Mistake("no-pairs", wrong=_TW_NO_PAIRS,
                    en="`{call}` gives {got}, and it should give {want}: your function never looks at `pairs`, so every word gets its most frequent tag, as with the first tagger. When should the pair with the word before decide?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie kijkt nooit in `pairs`, dus elk woord krijgt de vaakste tag, zoals bij de eerste tagger. Wanneer moet het paar met het woord ervoor beslissen?"),
            Mistake("wrap", wrong=_TW_WRAP,
                    en="`{call}` gives {got}, and it should give {want}: the first word gets the tag of a pair too, with the last word of the list as its word before, since `words[-1]` is the last word. Which word has no word before it?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het eerste woord krijgt ook de tag van een paar, met het laatste woord van de lijst als woord ervoor, want `words[-1]` is het laatste woord. Welk woord heeft geen woord ervoor?"),
            Mistake("reversed", wrong=_TW_REVERSED,
                    en="`{call}` gives {got}, and it should give {want}: your pair has the word first and the word before it second. In which order do the two words stand in a key of `pairs`?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: in je paar staat het woord eerst en het woord ervoor als tweede. In welke volgorde staan de twee woorden in een key van `pairs`?"),
            Mistake("next", wrong=_TW_NEXT,
                    en="`{call}` gives {got}, and it should give {want}: your function looks at the word after each word, and not at the word before. Which neighbor of the word does a key of `pairs` hold?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie kijkt naar het woord na elk woord, en niet naar het woord ervoor. Welke buur van het woord staat in een key van `pairs`?"),
            Mistake("order", wrong=_TW_ORDER,
                    en="`{call}` gives {got}, and it should give {want}: your function looks in `best` first, so a word that `best` has never gets the tag of its pair. When the pair and the word alone disagree, which one should win?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie kijkt eerst in `best`, dus een woord dat in `best` staat, krijgt nooit de tag van het paar. Als het paar en het woord alleen het oneens zijn, welke moet dan winnen?"),
            Mistake("not-list", matches=_not_list,
                    en="`{call}` gives {got}, and it should give {want}: a list with one tag for each word. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een lijst met één tag voor elk woord. Wat geeft je `return`-regel terug?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give each word the tag of its pair with the word before when `pairs` has that pair, else its tag in `best`, else NOUN.",
                      "nl": "Vervang de `...`: geef elk woord de tag van het paar met het woord ervoor als `pairs` dat paar heeft, anders de tag in `best`, en anders NOUN."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. For each word: is its pair with the word before in `pairs`, and is the word in `best`?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Voor elk woord: staat het paar met het woord ervoor in `pairs`, en staat het woord in `best`?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for the first word, for no words, and for a word that only `pairs` can tag?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor het eerste woord, zonder woorden, en voor een woord dat alleen `pairs` kan taggen?"},
            "correct": {"en": "Your `tag_words` is right in all {n} cases. Look at the picture: your tags on two sentences of the test set, and the share right on both test sets.",
                        "nl": "Je `tag_words` klopt in alle {n} gevallen. Kijk naar de afbeelding: je tags op twee zinnen uit de test set, en het aandeel goed op beide test sets."},
        },
    ),
}

CASES = [
    ("best_tags", _BT_START, "empty", None),
    ("best_tags", _BT, "correct", None),
    ("best_tags", _BT.replace("> most", ">= most"), "correct", None),   # either tag of a tie counts, and no case has one
    ("best_tags", "def best_tags(counts):\n    best = {}\n    for word in counts:\n        best[word] = max(counts[word], key=counts[word].get)\n    return best\n", "correct", None),
    ("best_tags", "def best_tags(counts):\n    return {w: sorted(c.items(), key=lambda kv: -kv[1])[0][0] for w, c in counts.items()}\n", "correct", None),
    ("best_tags", _BT_COUNT, "wrong", "count"),
    ("best_tags", _BT.replace("        best[word] = top\n", "        best[word] = max(counts[word].values())\n"), "wrong", "count"),
    ("best_tags", _BT_CARRY, "wrong", "carry"),
    ("best_tags", _BT_LAST, "wrong", "last"),
    ("best_tags", _BT_ABC, "wrong", "abc"),
    ("best_tags", _BT_EARLY, "wrong", "early"),
    ("best_tags", "def best_tags(counts):\n    best = {}\n    top = None\n    for word in counts:\n        for tag in counts[word]:\n            if top is None or counts[word][tag] > counts[word][top]:\n                top = tag\n        best[word] = top\n    return best\n", "error", "key"),
    ("best_tags", "def best_tags(counts):\n    return len(counts)\n", "wrong", "not-dict"),
    ("best_tags", "def best_tags(counts):\n    return {'that': 'SCONJ', 'book': 'NOUN'}\n", "almost", None),
    ("best_tags", _BT.replace("    best = {}\n", "    best = {}\n    ...\n"), "wrong", "dots-left"),
    ("accuracy", _AC_START, "empty", None),
    ("accuracy", _AC, "correct", None),
    ("accuracy", "def accuracy(sentences, best):\n    pairs = [(w, t) for s in sentences for w, t in s]\n    return sum(best.get(w, 'NOUN') == t for w, t in pairs) / len(pairs)\n", "correct", None),
    ("accuracy", _AC_NO_DEFAULT, "error", "no-default"),
    ("accuracy", _AC.replace("right / total", "right // total"), "wrong", "floor"),
    ("accuracy", _AC_SKIP, "wrong", "skip"),
    ("accuracy", _AC_UNKNOWN_WRONG, "wrong", "unknown-wrong"),
    ("accuracy", _AC_COUNT, "wrong", "count"),
    ("accuracy", _AC_PERCENT, "wrong", "percent"),
    ("accuracy", _AC_ERRORS, "wrong", "errors"),
    ("accuracy", _AC_SENTENCES, "wrong", "sentences"),
    ("accuracy", _AC_RESET, "wrong", "reset"),
    ("accuracy", _AC_PER_SENTENCE, "wrong", "per-sentence"),
    ("accuracy", "def accuracy(sentences, best):\n    return [1, 2]\n", "wrong", "not-number"),
    ("accuracy", "def accuracy(sentences, best):\n    return 6 / 9\n", "almost", None),
    ("tag_words", _TW_START, "empty", None),
    ("tag_words", _TW, "correct", None),
    ("tag_words", _TW.replace("    return tags\n", "    return tuple(tags)\n"), "correct", None),
    ("tag_words", "def tag_words(words, best, pairs):\n    tags = []\n    before = None\n    for word in words:\n        if (before, word) in pairs:\n            tags.append(pairs[(before, word)])\n        elif word in best:\n            tags.append(best[word])\n        else:\n            tags.append('NOUN')\n        before = word\n    return tags\n", "correct", None),
    ("tag_words", _TW_NO_DEFAULT, "error", "no-default"),
    ("tag_words", _TW.replace("        if i > 0 and (words[i - 1], word) in pairs:\n            tags.append(pairs[(words[i - 1], word)])\n", "        if i > 0 and pairs[(words[i - 1], word)]:\n            tags.append(pairs[(words[i - 1], word)])\n"), "error", "no-default"),
    ("tag_words", _TW_NO_PAIRS, "wrong", "no-pairs"),
    ("tag_words", _TW_WRAP, "wrong", "wrap"),
    ("tag_words", _TW_REVERSED, "wrong", "reversed"),
    ("tag_words", _TW_NEXT, "wrong", "next"),
    ("tag_words", _TW_ORDER, "wrong", "order"),
    ("tag_words", "def tag_words(words, best, pairs):\n    return len(words)\n", "wrong", "not-list"),
    ("tag_words", "def tag_words(words, best, pairs):\n    return ['VERB', 'ADP', 'NOUN']\n", "almost", None),
]
