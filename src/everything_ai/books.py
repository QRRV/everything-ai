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
(data/tools/make_uci_sentiment.py)."""
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
