# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""The SMS Spam Collection v.1 of Tiago A. Almeida and José María Gómez Hidalgo (2011): 5,574 text messages in
English, each labeled spam or ham, the messages that are not spam. 747 are spam. The collection is under CC BY 4.0 at
the UCI Machine Learning Repository (dataset 228). Its ham messages are real private messages, with first names and at
times a phone number, so the course hosts none of it (course/route/bronnen.yaml, sms-spam-collection):
everything_ai.hub downloads the zip once, 203 KB, from UCI, and keeps it only when its SHA-256 is the pinned one.
`lines` reads the one text file the lessons need from the zip, as text: nothing in it is ever run. The lessons show
only messages they chose, without names and with long numbers masked."""
import functools
import zipfile

from everything_ai import books, hub

FILE = ("archive.ics.uci.edu/dataset/228", "sms+spam+collection.zip")
MEMBER = "SMSSpamCollection"   # the file in the zip with the messages; the other one is its readme
_MOST = 2_000_000   # bytes the course reads of MEMBER at most; the file has 477,907


def ready() -> bool:
    """Whether the zip is downloaded."""
    return hub.pinned(*FILE) is not None


@functools.cache
def lines() -> tuple[tuple[int, str], ...]:
    """Every line of the collection as (label, text), in the order of the file: label 1 for spam, 0 for ham. It
    raises FileNotFoundError when the zip is not downloaded yet: hub.download(*FILE) gets it."""
    path = hub.pinned(*FILE)
    if path is None:
        raise FileNotFoundError("the SMS Spam Collection is not downloaded yet: everything_ai.hub.download(*everything_ai.sms.FILE)")
    with zipfile.ZipFile(path) as z:
        if z.getinfo(MEMBER).file_size > _MOST:
            raise ValueError(f"{MEMBER} is larger than the course expects")
        text = z.read(MEMBER).decode("utf-8")
    out = []
    for line in text.split("\n"):
        if line:
            label, message = line.split("\t", 1)
            out.append((1 if label == "spam" else 0, message))
    return tuple(out)


def words(text: str) -> list:
    """The words of a message, in small letters: runs of letters, with an apostrophe inside a word, as in
    *Logistic regression on words*; digits and signs are left out."""
    return books.words(text.lower())


@functools.cache
def _split() -> tuple:
    """The different texts of the collection, each once, with the label of its first line, in the order of their first
    line; every fifth of them is a test message. Gives (training, test, line), with line the number of the first line
    of each different text, counted from 1."""
    first: dict[str, tuple[int, int]] = {}
    for number, (label, text) in enumerate(lines(), start=1):
        if text not in first:
            first[text] = (label, number)
    training, test = [], []
    for k, (text, (label, _)) in enumerate(first.items()):
        (test if k % 5 == 4 else training).append((tuple(words(text)), label))
    return tuple(training), tuple(test), {text: number for text, (_, number) in first.items()}


def training() -> list:
    """The messages to train on, 4,137, as (words, label): a list of words and 1 for spam, 0 for ham. A new list at
    every call, so a lesson that changes it changes nothing for the next."""
    return [(list(w), label) for w, label in _split()[0]]


def test() -> list:
    """The test messages, 1,034, as training() gives them: every fifth different text, so that no test message is also
    a message to train on."""
    return [(list(w), label) for w, label in _split()[1]]


def first_line(text: str) -> int:
    """The number of the first line with this text, counted from 1; a KeyError for a text that is not in the
    collection."""
    return _split()[2][text]
