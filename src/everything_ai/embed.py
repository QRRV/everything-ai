# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Granite Embedding 107M multilingual, a text embedding model of IBM's Granite Embedding Team (technical report,
2025, arXiv:2502.20204; Apache-2.0), written out so that nothing beyond PyTorch is needed. It turns a text into 384
numbers, its embedding: texts with the same meaning get embeddings that point the same way, also when one is English
and the other Dutch. It knows 12 languages, English and Dutch among them, and learned from pairs of texts that belong
together, such as a question and its answer (contrastive training), and from a larger model (knowledge distillation).

How it works: a tokenizer cuts the text into pieces from the 250,002 of XLM-RoBERTa (Conneau and others, 2019,
arXiv:1911.02116), with <s> in front and </s> at the end. Every piece becomes 384 numbers, and its place adds 384 more.
Six transformer blocks let every token look at every other, as in DETR's encoder, with the layer normalization after
each branch is added. What <s> holds after the last block, scaled to length 1, is the text's embedding (CLS pooling):
the dot product of two embeddings is their cosine similarity.

The tokenizer is the course's own too. The file tokenizer.json holds a score for every piece, and the rules that
first normalize a text (SentencePiece's nmt_nfkc, such as full-width Ａ to A and every kind of space to an ordinary
one) as a compiled trie. `Tokenizer` reads both, normalizes as Hugging Face's tokenizers does, and picks the pieces
with the highest total score (the Viterbi algorithm), as SentencePiece's unigram model does.

The course does not ship it (214 MB): everything_ai.hub downloads model.safetensors and tokenizer.json of
ibm-granite/granite-embedding-107m-multilingual once, at a fixed commit (FILES). The model has short names of the
course's own; the file has the names of Hugging Face's code, and `load` renames them. On English and Dutch texts this
code gives the tokens of Hugging Face's tokenizers and the embeddings of sentence-transformers to within 0.0001
(tests/test_embed.py). Import this module only where the model is needed: it imports PyTorch."""
import base64
import functools
import json
import math
import re
import struct
import unicodedata

import torch
import torch.nn.functional as F
from torch import nn

from everything_ai import hub, models

REPO = "ibm-granite/granite-embedding-107m-multilingual"
FILES = ["model.safetensors", "tokenizer.json"]
VOCAB, D, HEADS, FFN, LAYERS = 250_002, 384, 12, 1536, 6   # from the model's config.json
EPS = 1e-12   # of every layer normalization
MAX = 512     # tokens at most, <s> and </s> included: the rest of a longer text is left out (truncation)
# Places count from 2, as in XLM-RoBERTa, where a <pad> has place 1: 514 rows for 512 tokens.
PLACES = MAX + 2
BOS, PAD, EOS, UNK, MASK = 0, 1, 2, 3, 250_001   # <s>, <pad>, </s>, <unk> and <mask>


# --- the tokenizer ----------------------------------------------------------------------------------------------------

SPACE = "\u2581"   # ▁, how a piece shows a space: "▁the" is "the" after a space
SPECIAL = {"<s>": BOS, "<pad>": PAD, "</s>": EOS, "<unk>": UNK}   # found in the text as it is, before normalizing
_SPECIAL = re.compile("(" + "|".join(SPECIAL) + ")")
# <mask> is found after normalizing, with the spaces before it (lstrip in tokenizer.json): Unicode's White_Space
_MASK = re.compile("[\t\n\x0b\x0c\r \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]*<mask>")
# Characters that start a grapheme and take the next character with them (Unicode's Prepend)
_PREPEND = {*range(0x0600, 0x0606), 0x06DD, 0x070F, 0x0890, 0x0891, 0x08E2, 0x0D4E, 0x110BD, 0x110CD, 0x111C2,
            0x111C3, 0x1193F, 0x11941, 0x11A3A, *range(0x11A84, 0x11A8A), 0x11D46, 0x11F02}
# Spacing vowel signs (Mc) that start a grapheme of their own all the same, in Myanmar, Tai Tham, Tai Viet and Ahom
_ALONE = {0x102B, 0x102C, 0x1038, *range(0x1062, 0x1065), *range(0x1067, 0x106E), 0x1083, *range(0x1087, 0x108D),
          0x108F, *range(0x109A, 0x109D), 0x1A61, 0x1A63, 0x1A64, 0xAA7B, 0xAA7D, 0x11720, 0x11721}
_BREAK = (None, "\r", "\n", "control")   # nothing joins a control character, \r or \n, before or after it


def _kind(ch: str) -> str:
    """What a character is to the rules of graphemes: "\\r", "\\n", "control", "mark" (it belongs to the character
    before it, such as a combining accent, a vowel sign, a zero width joiner or a skin tone), "prepend" or "other"."""
    if ch in "\r\n":
        return ch
    cp, category = ord(ch), unicodedata.category(ch)
    if (category in ("Mn", "Me", "Mc") and cp not in _ALONE or cp in (0x200C, 0x200D, 0x0E33, 0x0EB3)
            or 0xFF9E <= cp <= 0xFF9F or 0x1F3FB <= cp <= 0x1F3FF or 0xE0020 <= cp <= 0xE007F):
        return "mark"
    if cp in _PREPEND:
        return "prepend"
    # code points not used yet, kept for controls (default ignorable), count as controls already
    unused = category == "Cn" and (cp == 0x2065 or 0xFFF0 <= cp <= 0xFFF8 or 0xE0000 <= cp <= 0xE0FFF)
    return "control" if category in ("Cc", "Cf", "Zl", "Zp") or unused else "other"


def graphemes(text: str) -> list[str]:
    """The text cut into graphemes, what a reader sees as one character, such as e with a combining accent, by the
    rules of Unicode (UAX #29): a mark stays with the character before it, a prepend with the one after it, \\r with
    \\n, and a control character stands alone. The rules for Korean syllables and for flags are left out: the
    graphemes they make have 6 bytes or more, which the normalization below treats character by character anyway. So
    are the rules that join emoji with a zero width joiner and Indic consonants with a virama: they change the
    outcome only when such a sequence is followed by a character with a rule and a combining mark of another script."""
    out, before = [], None
    for ch in text:
        kind = _kind(ch)
        if before == "\r" and kind == "\n":
            out[-1] += ch
        elif before not in _BREAK and kind not in _BREAK and (kind == "mark" or before == "prepend"):
            out[-1] += ch
        else:
            out.append(ch)
        before = kind
    return out


def _offset(unit: int) -> int:
    return (unit >> 10) << ((unit & 0x200) >> 6)


class Charsmap:
    """The rules that normalize a text before it is cut into pieces: NFKC, such as ﬁ to fi and ½ to 1⁄2, and more,
    such as a tab, a new line or a no-break space to a space. tokenizer.json holds them as `precompiled_charsmap`: the
    size of a trie in bytes, the trie, and then every replacement, ending in a zero byte. The trie is a double array
    (Darts), over the bytes of the texts that have a rule: from a node, a byte leads to the node at (the node's offset
    XOR the byte), which is right when that node's label is the byte."""

    def __init__(self, blob: bytes):
        size = int.from_bytes(blob[:4], "little")
        self.units = struct.unpack_from(f"<{size // 4}I", blob, 4)
        self.replacements = blob[4 + size:]
        self.seen = {}

    def rule(self, chunk: str) -> str | None:
        """The replacement for the shortest beginning of `chunk` that has a rule, or None when no beginning has one."""
        if chunk not in self.seen:
            self.seen[chunk] = self._look_up(chunk)
        return self.seen[chunk]

    def _look_up(self, chunk: str) -> str | None:
        units = self.units
        node = _offset(units[0])
        for byte in chunk.encode():
            if byte == 0:
                return None
            node ^= byte
            unit = units[node]
            if unit & 0x800000FF != byte:   # its label: no rule starts with these bytes
                return None
            node ^= _offset(unit)
            if unit >> 8 & 1:   # a rule ends here: the next unit holds where its replacement starts
                start = units[node] & 0x7FFFFFFF
                return self.replacements[start:self.replacements.index(0, start)].decode()
        return None

    def normalize(self, text: str) -> str:
        """The text with every rule applied, as Hugging Face's tokenizers does it (normalizers/precompiled.rs): a
        grapheme of fewer than 6 bytes is first looked up whole, and when a beginning of it has a rule, that rule
        replaces all of it; else every character is looked up on its own. So e with two combining accents becomes
        é, without the second accent, where SentencePiece itself would keep it."""
        out = []
        for g in graphemes(text):
            whole = self.rule(g) if len(g) > 1 and len(g.encode()) < 6 else None
            if whole is not None:
                out.append(whole)
                continue
            for ch in g:
                found = self.rule(ch)
                out.append(ch if found is None else found)
        return "".join(out)


class Tokenizer:
    """The tokenizer of tokenizer.json, as Hugging Face's tokenizers runs it. First <s>, <pad>, </s> and <unk> are
    found in the text; every part between them is normalized (Charsmap), two spaces or more become one, and <mask> is
    found. Then every space becomes ▁, ▁ is put in front when the part does not start with one, and the part is cut
    before every ▁ into words. Each word is cut into the pieces with the highest total score (`pieces`), and the ids
    get <s> in front and </s> at the end. transformers 4 does the same; transformers 5 builds a pipeline of its own from
    the vocabulary, which leaves out the ▁ of a space at the end of a text or before a special token."""

    def __init__(self, spec: dict):
        model = spec["model"]
        if model["type"] != "Unigram" or model.get("byte_fallback"):
            raise ValueError("not a unigram tokenizer without byte fallback")
        self.vocab = [piece for piece, _ in model["vocab"]]
        self.scores = {piece: (i, score) for i, (piece, score) in enumerate(model["vocab"])}
        self.longest = max(map(len, self.vocab))
        self.unknown = min(score for _, score in model["vocab"]) - 10.0   # the score of a character not in the vocab
        rules = next(n for n in spec["normalizer"]["normalizers"] if n["type"] == "Precompiled")
        self.charsmap = Charsmap(base64.b64decode(rules["precompiled_charsmap"]))

    def pieces(self, word: str) -> list[int]:
        """The ids of the pieces that make up `word` with the highest total score. best[i] is the best score for the
        first i characters, and back[i] its last piece: where it starts, and its id. A character that is in no piece
        becomes <unk> with a low score, and <unk>s in a row become one."""
        n = len(word)
        best, back = [0.0] + [None] * n, [None] * (n + 1)
        for start in range(n):
            single = False
            for end in range(start + 1, min(start + self.longest, n) + 1):
                found = self.scores.get(word[start:end])
                if found is None:
                    continue
                score = best[start] + found[1]
                if back[end] is None or score > best[end]:
                    best[end], back[end] = score, (start, found[0])
                single = single or end == start + 1
            if not single:
                score = best[start] + self.unknown
                if back[start + 1] is None or score > best[start + 1]:
                    best[start + 1], back[start + 1] = score, (start, UNK)
        ids, end, unknown_end = [], n, None
        while end > 0:
            start, i = back[end]
            if i == UNK:
                unknown_end = unknown_end or end
            else:
                if unknown_end:
                    ids.append(self.scores.get(word[end:unknown_end], (UNK,))[0])
                    unknown_end = None
                ids.append(i)
            end = start
        if unknown_end:
            ids.append(self.scores.get(word[:unknown_end], (UNK,))[0])
        return ids[::-1]

    def words(self, part: str) -> list[int]:
        """The ids of a normalized part of the text: ▁ for every space and in front, then the pieces of every word."""
        if not part:
            return []
        part = part.replace(" ", SPACE)
        if not part.startswith(SPACE):
            part = SPACE + part
        return [i for word in re.findall(f"{SPACE}[^{SPACE}]*", part) for i in self.pieces(word)]

    def encode(self, text: str) -> list[int]:
        """The token ids of `text`, with <s> in front and </s> at the end, at most MAX of them."""
        ids = []
        for k, part in enumerate(_SPECIAL.split(text)):   # the special tokens are every second part
            if k % 2:
                ids.append(SPECIAL[part])
                continue
            normal = re.sub(" {2,}", " ", self.charsmap.normalize(part))
            for j, between in enumerate(_MASK.split(normal)):
                if j:
                    ids.append(MASK)
                ids += self.words(between)
        return [BOS] + ids[:MAX - 2] + [EOS]


# --- the model --------------------------------------------------------------------------------------------------------

class Block(nn.Module):
    """A transformer block as in DETR's encoder: every token looks at every other with 12 heads, except at the pads,
    and then goes through a small network of its own with GELU. The layer normalization comes after each branch is
    added, as in the original transformer and in BERT."""

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


class Granite(nn.Module):
    """Granite Embedding 107M multilingual: 106.8 million parameters, 96 million of them in the token embedding.
    `forward` gives the vector of every token after the last block."""

    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D)
        self.pos = nn.Embedding(PLACES, D)
        self.segment = nn.Embedding(2, D)   # XLM-RoBERTa's two segments of a text; a text is always segment 0
        self.norm = nn.LayerNorm(D, eps=EPS)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))

    def forward(self, ids, mask):
        """ids: (n, t) token ids, padded with PAD; mask: (n, t), True for a real token. Gives (n, t, 384)."""
        real = ids != PAD   # as in XLM-RoBERTa: a <pad> gets place 1, the tokens 2, 3, 4 and so on
        places = torch.cumsum(real, dim=1) * real + PAD
        x = self.norm(self.tok(ids) + self.pos(places) + self.segment.weight[0])
        for block in self.blocks:
            x = block(x, mask)
        return x


def _rename(name: str) -> str | None:
    """The course's name for a tensor in the file, or None for the pooler, a layer the embedding does not use."""
    if name.startswith("pooler."):
        return None
    for old, new in (("embeddings.word_embeddings", "tok"), ("embeddings.position_embeddings", "pos"),
                     ("embeddings.token_type_embeddings", "segment"), ("embeddings.LayerNorm", "norm"),
                     ("encoder.layer.", "blocks."), ("attention.self.", ""), ("attention.output.dense", "out"),
                     ("attention.output.LayerNorm", "norm1"), ("intermediate.dense", "fc1"),
                     ("output.dense", "fc2"), ("output.LayerNorm", "norm2")):
        name = name.replace(old, new)
    return name


# --- downloading, loading and embedding -------------------------------------------------------------------------------

def downloaded() -> bool:
    return all(hub.pinned(REPO, f) is not None for f in FILES)


def download(progress=None) -> None:
    """Download the model's two files once; `progress(done, total)` hears about every megabyte of both."""
    total = sum(hub.PINNED[(REPO, f)]["size"] for f in FILES)
    done = 0
    for f in FILES:
        hub.download(REPO, f, None if progress is None else lambda d, t, before=done: progress(before + d, total))
        done += hub.PINNED[(REPO, f)]["size"]


@functools.cache
def load() -> Granite:
    """The model in float32 and eval mode, nothing trainable, from the file everything_ai.hub downloaded; loaded
    once per process. It raises FileNotFoundError when the file is not there yet: download() gets it."""
    path = hub.pinned(REPO, "model.safetensors")
    if path is None:
        raise FileNotFoundError("Granite Embedding is not downloaded yet: everything_ai.embed.download()")
    tensors = models.read_safetensors(path, skip=lambda name: _rename(name) is None)
    with torch.device("meta"):
        model = Granite()
    model.load_state_dict({_rename(k): v.float() for k, v in tensors.items()}, assign=True, strict=True)
    model.requires_grad_(False)
    return model.eval()


@functools.cache
def tokenizer() -> Tokenizer:
    """The tokenizer, from the downloaded tokenizer.json; read once per process."""
    path = hub.pinned(REPO, "tokenizer.json")
    if path is None:
        raise FileNotFoundError("Granite Embedding is not downloaded yet: everything_ai.embed.download()")
    return Tokenizer(json.loads(path.read_text(encoding="utf-8")))


def encode(text: str) -> list[int]:
    """The token ids of `text`, as Hugging Face's tokenizer of the model gives them: <s>, the pieces, </s>."""
    return tokenizer().encode(text)


def embed(texts: list[str], batch: int = 16) -> torch.Tensor:
    """The embeddings of `texts`: shape (len(texts), 384), every row of length 1, so that the dot product of two rows
    is their cosine similarity. The texts go through the model `batch` at a time, those of about the same length
    together; each is padded with <pad> to the longest of its batch, and the mask keeps the pads out of the attention,
    so a text gets the same numbers in a batch as alone (to within rounding)."""
    if isinstance(texts, str):
        raise TypeError("embed takes a list of texts: embed([text])")
    return embed_ids([encode(t) for t in texts], batch)


def embed_ids(ids: list[list[int]], batch: int = 16) -> torch.Tensor:
    """The embeddings of texts that are already token ids, each as encode() gives them; see embed()."""
    model = load()
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    out = torch.zeros(len(ids), D)
    with torch.no_grad():
        for first in range(0, len(order), batch):
            group = order[first:first + batch]
            longest = max(len(ids[i]) for i in group)
            x = torch.full((len(group), longest), PAD)
            mask = torch.zeros(len(group), longest, dtype=torch.bool)
            for row, i in enumerate(group):
                x[row, :len(ids[i])] = torch.tensor(ids[i])
                mask[row, :len(ids[i])] = True
            out[group] = F.normalize(model(x, mask)[:, 0], dim=-1)
    return out
