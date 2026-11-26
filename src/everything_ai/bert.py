# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""BERT-Tiny, the smallest of Google's 24 BERT miniatures (Turc and others, described 2019 and released 2020, arXiv:1908.08962; Apache-2.0),
written out so that nothing beyond PyTorch is needed. It is BERT (Devlin and others, 2019) in small: 2 transformer
blocks of 128 numbers per token with 2 heads, 4.4 million parameters, 3.9 million of them in the token embedding.
It learned English only, in lowercase, from English Wikipedia and BookCorpus, by filling in masked words.

How it works: a tokenizer cuts the text into pieces from a vocabulary of 30,522 (WordPiece), with [CLS] in front and
[SEP] at the end. Every piece becomes 128 numbers, to which a vector of 128 numbers for its place, and one for the segment, is added. Two transformer blocks let every
token look at every other, on both sides, with the layer normalization after each branch is added. `layers` gives
what every token holds before the first block and after each block: from the second on, the numbers of a token
depend on the whole sentence. The head that fills in a [MASK] turns the last numbers back into a score per piece of
the vocabulary, with the token embedding as its weights (`piece_scores`, `fill`).

The tokenizer is the course's own too (everything_ai.wordpiece), as Hugging Face's BertTokenizer of the model does it:
it removes control characters, puts a space around every Chinese character, makes everything lowercase and drops
accents, splits at spaces and punctuation, and cuts every word into the longest pieces of the vocabulary from left to
right; a piece inside a word starts with ##. A word it cannot cut, or of more than 100 characters, becomes [UNK].

The course does not ship it (17.7 MB): everything_ai.hub downloads model.safetensors and vocab.txt of
google/bert_uncased_L-2_H-128_A-2 once, at a fixed commit (FILES). The model has short names of the course's own;
the file has the names of Hugging Face's code, and `load` renames them. On English texts this code gives the tokens
of Hugging Face's tokenizer and the numbers of its BertForMaskedLM to within 0.0001 (tests/test_bert.py). Import
this module only where the model is needed: it imports PyTorch."""
import functools
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from everything_ai import hub, models
from everything_ai.wordpiece import LONGEST, SPECIAL, Tokenizer, _chinese, _clean, _punctuation, _words  # noqa: F401

REPO = "google/bert_uncased_L-2_H-128_A-2"
FILES = ["model.safetensors", "vocab.txt"]
VOCAB, D, HEADS, FFN, LAYERS = 30_522, 128, 2, 512, 2   # from the model's config.json
EPS = 1e-12   # of every layer normalization
MAX = 512     # tokens at most, [CLS] and [SEP] included: the rest of a longer text is left out (truncation)
PAD, UNK, CLS, SEP, MASK = 0, 100, 101, 102, 103   # [PAD], [UNK], [CLS], [SEP] and [MASK] in vocab.txt

# The tokenizer is in everything_ai.wordpiece (Tokenizer, and the steps before it), which needs no PyTorch.


# --- the model --------------------------------------------------------------------------------------------------------

class Block(nn.Module):
    """A transformer block of BERT: every token looks at every other with 2 heads, except at the pads, and then goes
    through a small network of its own with GELU. The layer normalization comes after each branch is added."""

    def __init__(self):
        super().__init__()
        self.query, self.key, self.value, self.out = (nn.Linear(D, D) for _ in range(4))
        self.norm1 = nn.LayerNorm(D, eps=EPS)
        self.fc1, self.fc2 = nn.Linear(D, FFN), nn.Linear(FFN, D)
        self.norm2 = nn.LayerNorm(D, eps=EPS)

    def forward(self, x, mask):
        n, t, size = x.shape[0], x.shape[1], D // HEADS
        q, k, v = (f(x).view(n, t, HEADS, size).transpose(1, 2) for f in (self.query, self.key, self.value))
        scores = (q @ k.transpose(-2, -1) / math.sqrt(size)).masked_fill(~mask[:, None, None, :], float("-inf"))
        x = self.norm1(x + self.out((torch.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(n, t, D)))
        return self.norm2(x + self.fc2(F.gelu(self.fc1(x))))


class Bert(nn.Module):
    """BERT-Tiny: 4.4 million parameters, 3.9 million of them in the token embedding, and the head that fills in
    masked words. `forward` gives the numbers of every token before the first block and after each block."""

    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D)
        self.pos = nn.Embedding(MAX, D)
        self.segment = nn.Embedding(2, D)   # BERT's two segments, for a pair of sentences; one text is segment 0
        self.norm = nn.LayerNorm(D, eps=EPS)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        # the head that fills in [MASK]: a layer of its own, then the token embedding again, with a bias per piece
        self.head = nn.Linear(D, D)
        self.head_norm = nn.LayerNorm(D, eps=EPS)
        self.head_bias = nn.Parameter(torch.zeros(VOCAB))

    def forward(self, ids, mask=None, segments=None):
        """ids: (n, t) token ids, padded with PAD; mask: (n, t), True for a real token (all, when left out);
        segments: (n, t) 0 or 1 (all 0, when left out). Gives a list of LAYERS + 1 tensors of (n, t, 128)."""
        if mask is None:
            mask = torch.ones_like(ids, dtype=torch.bool)
        if segments is None:
            segments = torch.zeros_like(ids)
        places = torch.arange(ids.shape[1], device=ids.device)
        x = self.norm(self.tok(ids) + self.pos(places) + self.segment(segments))
        out = [x]
        for block in self.blocks:
            x = block(x, mask)
            out.append(x)
        return out

    def scores(self, x):
        """The score of every piece of the vocabulary for the last numbers of a token: (..., 128) to (..., 30,522)."""
        return self.head_norm(F.gelu(self.head(x))) @ self.tok.weight.T + self.head_bias


def _rename(name: str) -> str | None:
    """The course's name for a tensor in the file, or None for the pooler and the next-sentence head, which the
    course does not use."""
    if name.startswith(("bert.pooler.", "cls.seq_relationship.")):
        return None
    for old, new in (("cls.predictions.transform.dense", "head"), ("cls.predictions.transform.LayerNorm", "head_norm"),
                     ("cls.predictions.bias", "head_bias"), ("bert.", ""),
                     ("embeddings.word_embeddings", "tok"), ("embeddings.position_embeddings", "pos"),
                     ("embeddings.token_type_embeddings", "segment"), ("embeddings.LayerNorm", "norm"),
                     ("encoder.layer.", "blocks."), ("attention.self.", ""), ("attention.output.dense", "out"),
                     ("attention.output.LayerNorm", "norm1"), ("intermediate.dense", "fc1"),
                     ("output.dense", "fc2"), ("output.LayerNorm", "norm2")):
        name = name.replace(old, new)
    return name


# --- downloading, loading and using -----------------------------------------------------------------------------------

def downloaded() -> bool:
    return all(hub.pinned(REPO, f) is not None for f in FILES)


def download(progress=None) -> None:
    """Download the model's two files once; `progress(done, total)` hears about every megabyte of both."""
    total = sum(hub.PINNED[(REPO, f)]["size"] for f in FILES)
    done = 0
    for f in FILES:
        hub.download(REPO, f, None if progress is None else lambda d, t, before=done: progress(before + d, total))
        done += hub.PINNED[(REPO, f)]["size"]


def _missing():
    return FileNotFoundError("BERT-Tiny is not downloaded yet: everything_ai.bert.download()")


@functools.cache
def load() -> Bert:
    """The model in float32 and eval mode, nothing trainable, from the file everything_ai.hub downloaded; loaded
    once per process. It raises FileNotFoundError when the file is not there yet: download() gets it."""
    path = hub.pinned(REPO, "model.safetensors")
    if path is None:
        raise _missing()
    tensors = models.read_safetensors(path, skip=lambda name: _rename(name) is None)
    with torch.device("meta"):
        model = Bert()
    model.load_state_dict({_rename(k): v.float() for k, v in tensors.items()}, assign=True, strict=True)
    model.requires_grad_(False)
    return model.eval()


@functools.cache
def tokenizer() -> Tokenizer:
    """The tokenizer, from the downloaded vocab.txt; read once per process."""
    path = hub.pinned(REPO, "vocab.txt")
    if path is None:
        raise _missing()
    return Tokenizer(path.read_text(encoding="utf-8").split("\n")[:VOCAB])


def pieces(text: str) -> list[str]:
    """The pieces the model sees of a text: [CLS], the pieces of the text, [SEP]."""
    return ["[CLS]"] + tokenizer().pieces(text)[:MAX - 2] + ["[SEP]"]


def layers(text: str) -> list[np.ndarray]:
    """What every piece of pieces(text) holds, before the first block and after each block: a list of 3 arrays of
    one row of 128 numbers (float32) per piece. Row i of every array belongs to piece i."""
    with torch.no_grad():
        out = load()(torch.tensor([tokenizer().encode(text)]))
    return [x[0].numpy() for x in out]


def piece_scores(text: str) -> np.ndarray:
    """The score of every piece of the vocabulary at every place of pieces(text): an array of one row of 30,522
    numbers (float32) per piece, [CLS] and [SEP] included. Row i belongs to piece i; the higher a score, the more the
    model likes that piece there."""
    with torch.no_grad():
        model = load()
        return model.scores(model(torch.tensor([tokenizer().encode(text)]))[-1][0]).numpy()


def fill(text: str, top: int = 5, before_only: bool = False) -> list[list[tuple[str, float]]]:
    """For every [MASK] in the text, the `top` pieces the model likes best there, each with its probability. With
    `before_only`, the model sees only the pieces before each [MASK]: every piece after it, up to [SEP], becomes [MASK]
    too, so that the sentence keeps its length."""
    ids = tokenizer().encode(text)
    with torch.no_grad():
        model = load()
        last = model(torch.tensor([ids]))[-1][0]
        out = []
        for i, piece in enumerate(ids):
            if piece == MASK:
                if before_only:
                    hidden = ids[:i] + [MASK] * (len(ids) - 1 - i) + [SEP]
                    p = torch.softmax(model.scores(model(torch.tensor([hidden]))[-1][0][i]), dim=-1)
                else:
                    p = torch.softmax(model.scores(last[i]), dim=-1)
                best = torch.topk(p, top)
                out.append([(tokenizer().vocab[j], float(q)) for q, j in zip(best.values, best.indices)])
    return out
