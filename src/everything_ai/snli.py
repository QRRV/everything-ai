# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Pairs of sentences for the lesson of domain taal about natural language inference: a sample of SNLI 1.0 (Bowman,
Angeli, Potts and Manning, EMNLP 2015; CC-BY-SA-4.0) with the pairs of 10,000 premises of its training set and the
labeled pairs of its test set (data/snli-sample.tsv.gz, made by data/tools/make_snli_sample.py).

A premise is a caption of a photo; the hypothesis is a sentence a crowd worker wrote for it; the label says whether
the hypothesis follows from the premise (entailment), contradicts it (contradiction) or neither (neutral)."""
import functools
import gzip
import re
from typing import NamedTuple

from everything_ai import paths

FILE = paths.ROOT / "data" / "snli-sample.tsv.gz"
LABELS = ("entailment", "neutral", "contradiction")
SPLITS = ("train", "test")
_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?")


class Pair(NamedTuple):
    split: str          # train or test
    shown: bool         # False for a pair that a lesson does not show as an example (see make_snli_sample.py)
    label: str          # entailment, neutral or contradiction
    premise: str        # as SNLI writes it, such as "Two dogs are running through a field."
    hypothesis: str


def words(text: str) -> list:
    """The words of a sentence in lower case, without punctuation: "Nobody's outside." gives ["nobody's", "outside"]."""
    return _WORD.findall(text.lower())


@functools.cache
def rows() -> tuple:
    """Every pair of the file, in its order: 36,788 train and 9,824 test."""
    lines = gzip.decompress(FILE.read_bytes()).decode("utf-8").split("\n")
    assert lines[0].split("\t") == ["split", "shown", "label", "premise", "hypothesis"], lines[0]
    out = []
    for line in lines[1:]:
        if not line:
            continue
        split, shown, label, premise, hypothesis = line.split("\t")
        assert split in SPLITS and label in LABELS, line
        out.append(Pair(split, shown == "1", label, premise, hypothesis))
    return tuple(out)


def split(name: str) -> tuple:
    """The pairs of one part, train or test, each as a tuple (premise words, hypothesis words, label)."""
    assert name in SPLITS, name
    return tuple((tuple(words(r.premise)), tuple(words(r.hypothesis)), r.label) for r in rows() if r.split == name)
