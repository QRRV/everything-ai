# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Real word embeddings for the lessons of domain taal, and analogy questions to measure them with.

English: the vectors of 50 numbers of GloVe 6B (Pennington, Socher and Manning, 2014; PDDL 1.0) for the 50,000 most
frequent words of the letters a to z, which the course ships (data/glove-6b-50d-50k.npz, made by
data/tools/make_glove.py), and the 19,544 analogy questions of word2vec (Mikolov and others, 2013; Apache-2.0,
data/word2vec-questions-words.txt.gz).

Dutch: the vectors of 300 numbers of fastText for Dutch, cc.nl.300 (Grave and others, 2018; CC BY-SA 3.0). Its file
lists two million words from the most to the least frequent, 1.3 GB, so a learner downloads only its first 20 MB
(everything_ai.hub, pinned on those bytes): 30,355 complete lines, of which `fasttext_nl` keeps the 28,209 words of
letters only, with their capitals, as fastText has them. The Dutch analogy questions are the course's own, made from
word pairs in ten of word2vec's groups (data/course-word-pairs-nl.tsv)."""
import functools
import gzip
import zlib

import numpy as np

from everything_ai import hub, paths

GLOVE = paths.ROOT / "data" / "glove-6b-50d-50k.npz"
QUESTIONS = paths.ROOT / "data" / "word2vec-questions-words.txt.gz"
PAIRS_NL = paths.ROOT / "data" / "course-word-pairs-nl.tsv"
FASTTEXT = ("dl.fbaipublicfiles.com/fasttext", "cc.nl.300.vec.gz")
_MOST = 100_000_000   # bytes the course unpacks of the 20 MB at most; they hold 68,698,184


@functools.cache
def _glove() -> tuple:
    with np.load(GLOVE, allow_pickle=False) as f:
        words = tuple(str(w) for w in f["words"])
        vectors = f["vectors"].astype(np.float32)
    vectors.flags.writeable = False   # one array for every lesson: nobody changes it by accident
    return words, vectors


def glove() -> tuple[list, np.ndarray]:
    """GloVe's 50,000 words, from the most to the least frequent, as a new list, and their vectors, an array of 50,000
    rows of 50 numbers (float32, read-only): row i is the vector of word i."""
    words, vectors = _glove()
    return list(words), vectors


def rows(words: list) -> dict:
    """A dictionary from each word to its row: the first row where it stands."""
    out = {}
    for i, w in enumerate(words):
        out.setdefault(w, i)
    return out


@functools.cache
def _questions() -> tuple:
    groups, name = {}, None
    for line in gzip.decompress(QUESTIONS.read_bytes()).decode("utf-8").split("\n"):
        if line.startswith(":"):
            name = line[1:].strip()
            groups[name] = []
        elif line:
            groups[name].append(tuple(line.lower().split()))
    return tuple((g, tuple(q)) for g, q in groups.items())


def questions() -> dict:
    """word2vec's analogy questions by group, in the order of the file: a dictionary from the name of a group, such as
    "capital-common-countries", to a list of questions (a, b, c, d), each word in small letters as GloVe has them: a
    is to b as c is to d."""
    return {g: list(q) for g, q in _questions()}


@functools.cache
def _pairs_nl() -> tuple:
    groups = {}
    for line in PAIRS_NL.read_text(encoding="utf-8").split("\n")[1:]:
        if line:
            group, first, second = line.split("\t")
            groups.setdefault(group, []).append((first, second))
    return tuple((g, tuple(p)) for g, p in groups.items())


def pairs_nl() -> dict:
    """The course's Dutch word pairs by group, with the group names of word2vec: "capital-common-countries" holds
    ("Frankrijk", "Parijs") and more."""
    return {g: list(p) for g, p in _pairs_nl()}


def questions_nl() -> dict:
    """The Dutch analogy questions, made as word2vec's are: every two different pairs of a group, in both orders, give
    a question (a, b, c, d) from pair (a, b) and pair (c, d)."""
    return {g: [(a, b, c, d) for a, b in pairs for c, d in pairs if (a, b) != (c, d)] for g, pairs in pairs_nl().items()}


def fasttext_ready() -> bool:
    """Whether the first 20 MB of fastText's Dutch vectors are downloaded."""
    return hub.pinned(*FASTTEXT) is not None


@functools.cache
def _fasttext_nl() -> tuple:
    path = hub.pinned(*FASTTEXT)
    if path is None:
        raise FileNotFoundError("fastText's Dutch vectors are not downloaded yet: everything_ai.hub.download(*everything_ai.wordvectors.FASTTEXT)")
    text = zlib.decompressobj(wbits=31).decompress(path.read_bytes(), _MOST)   # the start of a gzip file: no end
    lines = text.decode("utf-8").split("\n")
    count, size = (int(x) for x in lines[0].split())
    assert size == 300 and count == 2_000_000, "not fastText's cc.nl.300"
    complete = lines[1:-1]   # the last line breaks off at 20 MB
    kept = [line for line in complete if line.split(" ", 1)[0].isalpha()]
    vectors = np.empty((len(kept), size), dtype=np.float32)
    words = []
    for i, line in enumerate(kept):
        word, numbers = line.split(" ", 1)
        words.append(word)
        vectors[i] = np.array(numbers.split(), dtype=np.float32)
    vectors.flags.writeable = False
    return tuple(words), vectors


def fasttext_nl() -> tuple[list, np.ndarray]:
    """fastText's 28,209 most frequent Dutch words of letters only, from the most to the least frequent, as a new list,
    with their capitals (*koning* and *Koning* are two words), and their vectors, an array of 300 numbers per word
    (float32, read-only). It raises FileNotFoundError when the 20 MB are not downloaded yet."""
    words, vectors = _fasttext_nl()
    return list(words), vectors
