# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""The eight books of data/gutenberg-books.json.gz, the small corpus of domain taal: five in English and three in
Dutch, from Project Gutenberg, without Gutenberg's header and footer (data/tools/make_gutenberg.py). `book(key)` gives
one book as a dict (key, title, author, year, lang, edition, text), and `book_words(key)` its words as a list, so that
a lesson can hand a learner a book before the learner knows how to cut a text into words.

Next to the books, `dutch_sentences()` gives the Dutch sentences of Tatoeba in data/tatoeba-nl.tsv.gz, today's Dutch
written by volunteers (data/tools/make_tatoeba_nl.py), and `dutch_sentence_words()` all their words as one list.
`dutch_pairs()` gives the sentences of Tatoeba in Frisian, Gronings and Limburgish in data/tatoeba-nl-pairs.tsv.gz,
each with a Dutch translation (data/tools/make_tatoeba_pairs.py), and `language_sentences()` 500 sentences of
Tatoeba in each of twelve languages in data/tatoeba-langs.tsv.gz (data/tools/make_tatoeba_langs.py). `stop_words(lang)` gives the stop words of Snowball
for English or Dutch in data/snowball-stop-en.txt and data/snowball-stop-nl.txt (data/tools/make_snowball_stopwords.py).
`review_sentences()` gives the 3,000 labelled review sentences of data/uci-sentiment.tsv.gz
(data/tools/make_uci_sentiment.py), `dutch_review_sentences()` the course's Dutch translation of them in
data/uci-sentiment-nl.tsv.gz, and `course_reviews()` the short reviews of a few sentences that the course wrote in
English and Dutch, in data/course-reviews.tsv. `english_chapters()` cuts the five English books into their 251 chapters, stories
and letters, so that a lesson can use them as documents."""
import functools
import gzip
import json
import re

from everything_ai import paths

FILE = paths.ROOT / "data" / "gutenberg-books.json.gz"
TATOEBA_NL = paths.ROOT / "data" / "tatoeba-nl.tsv.gz"
TATOEBA_PAIRS = paths.ROOT / "data" / "tatoeba-nl-pairs.tsv.gz"
TATOEBA_LANGS = paths.ROOT / "data" / "tatoeba-langs.tsv.gz"
UCI_SENTIMENT = paths.ROOT / "data" / "uci-sentiment.tsv.gz"
UCI_SENTIMENT_NL = paths.ROOT / "data" / "uci-sentiment-nl.tsv.gz"
COURSE_REVIEWS = paths.ROOT / "data" / "course-reviews.tsv"
PATTERN_SENTIMENT = paths.ROOT / "data" / "pattern-sentiment.tsv.gz"
MASSIVE = paths.ROOT / "data" / "massive-en-nl.tsv.gz"
STOP_WORDS = {"en": paths.ROOT / "data" / "snowball-stop-en.txt", "nl": paths.ROOT / "data" / "snowball-stop-nl.txt"}
# a word: letters, with an apostrophe inside it as in "don't" or "Alice's"; digits, punctuation and the underscores
# that mark italics in the Gutenberg files are no part of a word
_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")


@functools.cache
def books() -> tuple:
    """Every book, in the order of the file: the English ones first."""
    with gzip.open(FILE, "rt", encoding="utf-8") as f:
        return tuple(json.load(f))


def keys() -> list:
    return [b["key"] for b in books()]


def book(key: str) -> dict:
    for b in books():
        if b["key"] == key:
            return b
    raise KeyError(key)


def words(text: str) -> list:
    """The words of a text in their order, as the text writes them: "The" and "the" stay two forms."""
    return _WORD.findall(text)


@functools.cache
def _words(key: str) -> tuple:
    return tuple(words(book(key)["text"]))


def book_words(key: str) -> list:
    """The words of one book, as a new list each time, so that a learner's code cannot change the course's copy."""
    return list(_words(key))


# --- the English books in chapters ----------------------------------------------------------------------------------

_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
_HOLMES = ["A Scandal in Bohemia", "The Red-Headed League", "A Case of Identity", "The Boscombe Valley Mystery",
           "The Five Orange Pips", "The Man with the Twisted Lip", "The Adventure of the Blue Carbuncle",
           "The Adventure of the Speckled Band", "The Adventure of the Engineer’s Thumb",
           "The Adventure of the Noble Bachelor", "The Adventure of the Beryl Coronet",
           "The Adventure of the Copper Beeches"]


def _roman(numeral: str) -> int:
    values = [_ROMAN[c] for c in numeral]
    return sum(-v if i + 1 < len(values) and values[i + 1] > v else v for i, v in enumerate(values))


def _heads(key: str, lines: list) -> list:
    """(line, kind, number, title) for each heading of an English book, after its list of contents. The file is pinned
    by its SHA-256, so the first line of the text proper is a fixed number, and the asserts say when a rule misses."""
    heads = []
    for i, line in enumerate(lines):
        s = line.strip()
        if key == "alice" and i > 24 and (m := re.match(r"^CHAPTER ([IVXL]+)\.$", s)):
            heads.append((i, "chapter", _roman(m.group(1)), lines[i + 1].strip()))
        elif key == "pride" and (m := re.match(r"^(?:CHAPTER|Chapter) ([IVXLC]+)\.?\]?$", s)):
            heads.append((i, "chapter", _roman(m.group(1)), ""))
        elif key == "holmes" and i > 18 and (m := re.match(r"^([IVX]+)\. ([A-Z’ \-]+)$", s)):
            number = _roman(m.group(1))
            assert _HOLMES[number - 1].upper() == m.group(2), s
            heads.append((i, "story", number, _HOLMES[number - 1]))
        elif key == "frankenstein" and i > 36 and (m := re.match(r"^(Letter|Chapter) (\d+)$", s)):
            heads.append((i, m.group(1).lower(), int(m.group(2)), ""))
        elif key == "moby-dick" and i > 296:
            if s == "ETYMOLOGY." or s.startswith("EXTRACTS. (Supplied") or s == "Epilogue":
                heads.append((i, s.split(".")[0].lower(), 0, ""))
            elif m := re.match(r"^CHAPTER (\d+)\. (.*?)\.?$", s):
                heads.append((i, "chapter", int(m.group(1)), m.group(2)))
    numbers = [(kind, n) for _, kind, n, _ in heads]
    expected = {"alice": [("chapter", n) for n in range(1, 13)],
                "pride": [("chapter", n) for n in range(1, 62)],
                "holmes": [("story", n) for n in range(1, 13)],
                "frankenstein": [("letter", n) for n in range(1, 5)] + [("chapter", n) for n in range(1, 25)],
                "moby-dick": [("etymology", 0), ("extracts", 0)] + [("chapter", n) for n in range(1, 136)]
                + [("epilogue", 0)]}[key]
    assert numbers == expected, (key, numbers[:5])
    return heads


# what the printed editions add to a chapter and the author did not write: the illustrations of Pride and Prejudice
# with their captions and copyright lines (one bracket may sit inside another; a picture whose block holds the next
# heading is still open at the end of a chapter), its printer's colophon, the numbers I. to III. of the parts of a
# Sherlock Holmes story, and the second heading of the extracts of Moby-Dick
_ILLUSTRATION = re.compile(r"\[Illustration(?:[^\[\]]|\[[^\[\]]*\])*(?:\]|\Z)")
_PRINTER = re.compile(r"^ *(CHISWICK PRESS:--CHARLES WHITTINGHAM AND CO\.|TOOKS COURT, CHANCERY LANE, LONDON\.)$", re.M)
_PART = re.compile(r"^ *[IVX]+\.$", re.M)


def _body(key: str, text: str) -> str:
    if key == "pride":
        return _PRINTER.sub("", _ILLUSTRATION.sub("", text))
    if key == "holmes":
        return _PART.sub("", text)
    if key == "moby-dick":
        return re.sub(r"^ *EXTRACTS\.$", "", text, flags=re.M)
    return text


@functools.cache
def _chapters() -> tuple:
    out = []
    for key in ("alice", "pride", "holmes", "frankenstein", "moby-dick"):
        lines = book(key)["text"].split("\n")
        heads = _heads(key, lines)
        for j, (i, kind, number, title) in enumerate(heads):
            end = heads[j + 1][0] if j + 1 < len(heads) else len(lines)
            start = i + 2 if key == "alice" else i + 1   # after the heading, and after Alice's title line
            out.append((key, kind, number, title, tuple(words(_body(key, "\n".join(lines[start:end]))))))
    return tuple(out)


def english_chapters() -> list:
    """The five English books cut into 251 parts, each from its heading to the next: (key, kind, number, title, words),
    with kind "chapter", "story" (the twelve stories of Sherlock Holmes), "letter" (the four letters that open
    Frankenstein), or "etymology", "extracts" and "epilogue" (in Moby-Dick, number 0). The title is "" where the book
    gives none. The words are those after the heading line (and after the title line in Alice), as the text writes them;
    what comes before the first heading (a title page, a list of contents, the preface of Pride and Prejudice) is left
    out, and so is what the printed editions add: the illustrations of Pride and Prejudice with their captions, its
    printer's colophon, the part numbers of the Sherlock Holmes stories and a second heading in Moby-Dick."""
    return [(key, kind, number, title, list(ws)) for key, kind, number, title, ws in _chapters()]


@functools.cache
def dutch_sentences() -> tuple:
    """The Dutch sentences of Tatoeba as (id, text), in the order of the file."""
    with gzip.open(TATOEBA_NL, "rt", encoding="utf-8") as f:
        lines = f.read().split("\n")
    assert lines[0] == "id\ttext"
    return tuple((int(sid), text) for sid, text in (line.split("\t") for line in lines[1:] if line))


@functools.cache
def _sentence_words() -> tuple:
    return tuple(w for _, text in dutch_sentences() for w in words(text))


def dutch_sentence_words() -> list:
    """The words of all Dutch sentences of Tatoeba, one after another, as a new list each time."""
    return list(_sentence_words())


@functools.cache
def dutch_pairs() -> tuple:
    """The sentences of Tatoeba in Frisian (fry), Gronings (gos) and Limburgish (lim), each as (lang, id, text, nl_id,
    nl_text) with its Dutch translation, in the order of the file: by language, then by id."""
    with gzip.open(TATOEBA_PAIRS, "rt", encoding="utf-8") as f:
        lines = f.read().split("\n")
    assert lines[0] == "lang\tid\ttext\tnl_id\tnl_text"
    out = []
    for line in lines[1:]:
        if line:
            lang, sid, text, nl_id, nl_text = line.split("\t")
            out.append((lang, int(sid), text, int(nl_id), nl_text))
    return tuple(out)


@functools.cache
def language_sentences() -> tuple:
    """The sentences of Tatoeba in twelve languages, each as (lang, id, text) with the two-letter code of its language
    (nl, af, fy, de, en, da, nb, sv, fr, es, pt, it), in the order of the file: by language, then by id."""
    with gzip.open(TATOEBA_LANGS, "rt", encoding="utf-8") as f:
        lines = f.read().split("\n")
    assert lines[0] == "lang\tid\ttext"
    return tuple((lang, int(sid), text) for lang, sid, text in (line.split("\t") for line in lines[1:] if line))



@functools.cache
def review_sentences() -> tuple:
    """The 3,000 sentences of "Sentiment Labelled Sentences" (UCI), each as (source, line, label, shown, text): the
    website (amazon, imdb or yelp), the line of the sentence in its file, its label (1 positive, 0 negative), whether a
    lesson shows it as an example (1) or not (0), and the sentence. In the order of the file: amazon, imdb, yelp."""
    with gzip.open(UCI_SENTIMENT, "rt", encoding="utf-8") as f:
        lines = f.read().split("\n")
    assert lines[0] == "source\tline\tlabel\tshown\ttext"
    return tuple((source, int(line), int(label), int(shown), text)
                 for source, line, label, shown, text in (row.split("\t") for row in lines[1:] if row))


@functools.cache
def dutch_review_sentences() -> tuple:
    """The course's Dutch translation of the 3,000 sentences of review_sentences(), in the same order and the same
    form (source, line, label, shown, text): the same source, line, label and shown, with the Dutch sentence."""
    with gzip.open(UCI_SENTIMENT_NL, "rt", encoding="utf-8") as f:
        lines = f.read().split("\n")
    assert lines[0] == "source\tline\tlabel\tshown\ttext"
    return tuple((source, int(line), int(label), int(shown), text)
                 for source, line, label, shown, text in (row.split("\t") for row in lines[1:] if row))


@functools.cache
def pattern_sentiment() -> tuple:
    """The sentiment lexicons of Pattern (De Smedt and Daelemans, PDDL), one row per sense of a word, English first and
    then Dutch, as (lang, form, pos, sense, polarity, subjectivity, intensity, shown): the three scores as floats, shown
    0 for a row the course does not show."""
    with gzip.open(PATTERN_SENTIMENT, "rt", encoding="utf-8") as f:
        lines = f.read().split("\n")
    assert lines[0] == "lang\tform\tpos\tsense\tpolarity\tsubjectivity\tintensity\tconfidence\twordnet_id\tshown"
    return tuple((lang, form, pos, sense, float(p), float(s), float(i), int(shown))
                 for lang, form, pos, sense, p, s, i, _, _, shown in (row.split("\t") for row in lines[1:] if row))


@functools.cache
def course_reviews() -> tuple:
    """The 18 short reviews that the course wrote, each as (kind, label, sentences): product, film or restaurant, the
    label of the whole review (1 positive, 0 negative), and its sentences as (label, English, Dutch), in their order."""
    lines = COURSE_REVIEWS.read_text(encoding="utf-8").split("\n")
    assert lines[0] == "review\tkind\treview_label\tsentence\tlabel\ten\tnl"
    out = {}
    for row in lines[1:]:
        if row:
            review, kind, review_label, _, label, en, nl = row.split("\t")
            out.setdefault(int(review), (kind, int(review_label), []))[2].append((int(label), en, nl))
    return tuple((kind, label, tuple(sentences)) for kind, label, sentences in out.values())


@functools.cache
def assistant_commands() -> tuple:
    """The 16,521 commands to a voice assistant of MASSIVE 1.1, each as (id, partition, scenario, intent, shown, en, nl,
    en_slots, nl_slots): the id of the record, train, dev or test, the scenario (such as alarm) and the intent (such as
    alarm_set), whether a lesson shows it as an example (1) or not (0), the command in English and in Dutch, and both
    with their slots marked, as `wake me up at [time : five am]`. In the order of the file."""
    with gzip.open(MASSIVE, "rt", encoding="utf-8") as f:
        lines = f.read().split("\n")
    assert lines[0] == "id\tpartition\tscenario\tintent\tshown\ten\tnl\ten_slots\tnl_slots"
    return tuple((int(r[0]), r[1], r[2], r[3], int(r[4]), r[5], r[6], r[7], r[8])
                 for r in (line.split("\t") for line in lines[1:] if line))

@functools.cache
def _stop_words(lang: str) -> tuple:
    out = []
    for line in STOP_WORDS[lang].read_text(encoding="utf-8").splitlines():
        parts = line.split("|", 1)[0].split()   # a comment starts with |
        if parts:
            out.append(parts[0])
    return tuple(out)


def stop_words(lang: str) -> set:
    """The stop words of Snowball for English (en) or Dutch (nl), in small letters, as a new set each time: the first
    item of each line of the list, before any comment. The Dutch list is ranked from the commonest word to the rarest;
    `stop_word_list(lang)` keeps that order."""
    return set(_stop_words(lang))


def stop_word_list(lang: str) -> list:
    """The stop words of Snowball for English (en) or Dutch (nl) in the order of the file, as a new list each time."""
    return list(_stop_words(lang))
