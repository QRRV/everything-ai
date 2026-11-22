# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""The eight books of data/gutenberg-books.json.gz, the small corpus of domain taal: five in English and three in
Dutch, from Project Gutenberg, without Gutenberg's header and footer (data/tools/make_gutenberg.py). `book(key)` gives
one book as a dict (key, title, author, year, lang, edition, text), and `book_words(key)` its words as a list, so that
a lesson can hand a learner a book before the learner knows how to cut a text into words.

Next to the books, `dutch_sentences()` gives the Dutch sentences of Tatoeba in data/tatoeba-nl.tsv.gz, today's Dutch
written by volunteers (data/tools/make_tatoeba_nl.py), and `dutch_sentence_words()` all their words as one list."""
import functools
import gzip
import json
import re

from everything_ai import paths

FILE = paths.ROOT / "data" / "gutenberg-books.json.gz"
TATOEBA_NL = paths.ROOT / "data" / "tatoeba-nl.tsv.gz"
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
