# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""The BPE tokenizers of Hugging Face's tokenizer.json files, written out so that nothing beyond `regex` (which
tiktoken already needs) is needed: lesson nlp-dutch-models cuts Dutch with the tokenizers of RobBERT, Mistral 7B
(under GEITje) and EuroLLM. Two kinds occur, and `Tokenizer` reads both from the file:

- byte-level BPE, as GPT-2 made it (RobBERT, Phi-2 under Fietje): the text is split into words and spaces first, every
  byte of a word becomes one character of GPT-2's byte alphabet (a space is Ġ), and the merges join neighbors;
- SentencePiece-style BPE (Mistral, EuroLLM, Llama 2): every space becomes ▁, a ▁ is put in front, the merges join
  characters over the whole text, and a character outside the vocabulary becomes its bytes, <0xC3> and so on (byte
  fallback).

The merges are applied as Hugging Face's library does: again and again the neighboring pair that was learned first,
until no learned pair is left. Only the parts of tokenizer.json that these models use are read; anything else raises
ValueError, so that a new file is never cut differently without notice. tests/test_bpe.py holds the tokens of Hugging
Face's own library for a set of sentences."""
import gzip
import json
import unicodedata
from pathlib import Path

import regex

GPT2_SPLIT = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
WHITESPACE_SPLIT = regex.compile(r"\w+|[^\w\s]+")   # Hugging Face's pre-tokenizer "Whitespace"
SPACE = "▁"   # U+2581, SentencePiece's sign for a space


def _byte_alphabet() -> dict:
    """GPT-2's alphabet: every byte as a printable character; the bytes without one get chr(256), chr(257), ..."""
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    out, extra = {b: chr(b) for b in printable}, 0
    for b in range(256):
        if b not in printable:
            out[b] = chr(256 + extra)
            extra += 1
    return out


BYTE_CHAR = _byte_alphabet()
CHAR_BYTE = {c: b for b, c in BYTE_CHAR.items()}


class Tokenizer:
    """A BPE tokenizer from a tokenizer.json, or a gzipped one (.json.gz). `pieces(text)` gives the tokens as strings
    of the vocabulary, `encode(text)` their ids, without the special tokens that a model puts around a text (<s>,
    </s>), `piece_bytes(piece)` the bytes a token stands for, and `decode(ids)` the text of a list of ids."""

    def __init__(self, path: Path):
        path = Path(path)
        raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
        spec = json.loads(raw.decode("utf-8"))
        model = spec["model"]
        if model.get("type") != "BPE" or model.get("dropout") or model.get("continuing_subword_prefix") or \
                model.get("end_of_word_suffix") or model.get("ignore_merges"):
            raise ValueError("not a BPE model of a kind this code knows")
        self.vocab = model["vocab"]
        self.ranks = {}
        for rank, merge in enumerate(model["merges"]):
            left, right = merge.split(" ") if isinstance(merge, str) else merge
            self.ranks.setdefault((left, right), rank)
        self.byte_fallback = bool(model.get("byte_fallback"))
        self.unk = model.get("unk_token")
        added = spec.get("added_tokens", [])
        if any(t.get("single_word") for t in added):
            raise ValueError("an added token for single words only")
        self.added = {t["content"]: t["id"] for t in added}
        self.piece = {i: p for p, i in self.vocab.items()} | {i: p for p, i in self.added.items()}   # id -> token
        # the added tokens, such as <s> and Phi-2's runs of spaces, are found in the text first, the longest first;
        # an lstrip or rstrip one takes the spaces on that side with it (RoBERTa's <mask>)
        alternatives = [(r"\s*" if t["lstrip"] else "") + regex.escape(t["content"]) + (r"\s*" if t["rstrip"] else "")
                        for t in sorted(added, key=lambda t: -len(t["content"]))]
        self._added_split = regex.compile("(" + "|".join(alternatives) + ")") if added else None
        self.kind, self.steps = self._read_steps(spec.get("normalizer"), spec.get("pre_tokenizer"))
        self._cache = {}

    def _read_steps(self, normalizer, pre) -> tuple[str, list]:
        """What happens before the merges, as a kind ("bytes" or "spaces") and a list of steps."""
        def flat(step, key):
            if step is None:
                return []
            return [s for part in step[key] for s in flat(part, key)] if step["type"] == "Sequence" else [step]
        norms = [(n["type"], n.get("prepend"), n.get("pattern", {}).get("String"), n.get("content"))
                 for n in flat(normalizer, "normalizers")]
        pres = flat(pre, "pretokenizers")
        if norms == [("Prepend", SPACE, None, None), ("Replace", None, " ", SPACE)] and not pres:
            return "spaces", ["prepend"]   # EuroLLM, Llama 2: always a ▁ in front
        if not norms and [p["type"] for p in pres] == ["Metaspace"] and pres[0]["replacement"] == SPACE \
                and pres[0].get("prepend_scheme") == "first" and not pres[0].get("split", True):
            return "spaces", ["prepend unless"]   # Mistral: a ▁ in front, unless the text starts with a space
        if not norms and pres and all(p["type"] in ("Whitespace", "ByteLevel", "Punctuation") for p in pres):
            steps = []
            for p in pres:
                if p["type"] == "Whitespace":
                    steps.append("whitespace")
                elif p["type"] == "ByteLevel":
                    if not p.get("use_regex", True):
                        raise ValueError("ByteLevel without its regex")
                    steps.append("bytes+space" if p.get("add_prefix_space") else "bytes")
                elif p.get("behavior") == "Isolated":
                    steps.append("punctuation")
                else:
                    raise ValueError("Punctuation other than Isolated")
            return "bytes", steps
        raise ValueError("a normalizer or pre-tokenizer this code does not know")

    # --- before the merges ---------------------------------------------------------------------------------------

    def _words(self, text: str) -> list[str]:
        """The parts that the merges work on, each already in the characters of the vocabulary."""
        if self.kind == "spaces":
            if not text:
                return []
            front = "" if self.steps == ["prepend unless"] and text.startswith(" ") else SPACE
            return [front + text.replace(" ", SPACE)]
        parts = [text]
        for step in self.steps:
            if step == "whitespace":
                parts = [w for p in parts for w in WHITESPACE_SPLIT.findall(p)]
            elif step == "punctuation":
                parts = [w for p in parts for w in _isolate_punctuation(p)]
            else:
                if step == "bytes+space":
                    parts = [p if p.startswith(" ") else " " + p for p in parts]
                parts = [w for p in parts for w in GPT2_SPLIT.findall(p)]
                parts = ["".join(BYTE_CHAR[b] for b in w.encode("utf-8")) for w in parts]
        return parts

    # --- the merges --------------------------------------------------------------------------------------------

    def _merge(self, word: str) -> list[str]:
        """The tokens of one part: its characters, then again and again the pair learned first joined, wherever it
        stands, until no learned pair is left. A character outside the vocabulary becomes its bytes (byte fallback),
        else the unknown token, else nothing."""
        if word in self._cache:
            return self._cache[word]
        symbols = []
        for ch in word:
            if ch in self.vocab:
                symbols.append(ch)
            elif self.byte_fallback and all(f"<0x{b:02X}>" in self.vocab for b in ch.encode("utf-8")):
                symbols += [f"<0x{b:02X}>" for b in ch.encode("utf-8")]
            elif self.unk is not None:
                symbols.append(self.unk)
            # else: left out, as Hugging Face's library does; RobBERT-2023 has 145 of the 256 bytes of GPT-2's alphabet
        while len(symbols) > 1:
            pairs = [(self.ranks.get((a, b)), i) for i, (a, b) in enumerate(zip(symbols, symbols[1:]))]
            best = min((r for r, _ in pairs if r is not None), default=None)
            if best is None:
                break
            out, i = [], 0
            while i < len(symbols):
                if i + 1 < len(symbols) and self.ranks.get((symbols[i], symbols[i + 1])) == best:
                    out.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            symbols = out
        if len(self._cache) < 100_000:
            self._cache[word] = symbols
        return symbols

    def pieces(self, text: str) -> list[str]:
        """The tokens of a text, as strings of the vocabulary (Ġ for a space in byte-level BPE, ▁ in SentencePiece)."""
        out = []
        for i, part in enumerate(self._added_split.split(text) if self._added_split else [text]):
            if i % 2:   # an added token
                out.append(next(t for t in (part, part.lstrip(), part.rstrip(), part.strip()) if t in self.added))
            elif part:
                out += [t for word in self._words(part) for t in self._merge(word)]
        return out

    def encode(self, text: str) -> list[int]:
        """The ids of the tokens of a text."""
        return [self.added[t] if t in self.added else self.vocab[t] for t in self.pieces(text)]

    def piece_bytes(self, piece: str) -> bytes:
        """The bytes that a token stands for. In byte-level BPE every character is one byte of GPT-2's alphabet (Ġ is a
        space); in SentencePiece-style BPE a ▁ is a space and <0xC3> the byte C3. An added token is its own text."""
        if piece in self.added:
            return piece.encode("utf-8")
        if self.kind == "bytes":
            return bytes(CHAR_BYTE[ch] for ch in piece)
        if self.byte_fallback and len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
            return bytes([int(piece[3:5], 16)])
        return piece.replace(SPACE, " ").encode("utf-8")

    def decode(self, ids: list[int]) -> str:
        """The text of a list of ids: the bytes of their tokens one after the other, read as UTF-8. Where a list ends
        in the middle of a character, as one token of a letter of two bytes can, that part becomes U+FFFD. The ▁ that
        SentencePiece-style BPE puts in front stays a space."""
        return b"".join(self.piece_bytes(self.piece[i]) for i in ids).decode("utf-8", errors="replace")


def _isolate_punctuation(text: str) -> list[str]:
    """Hugging Face's pre-tokenizer Punctuation, "Isolated": every punctuation character a part of its own."""
    out, current = [], ""
    for ch in text:
        if _is_punctuation(ch):
            if current:
                out.append(current)
            out.append(ch)
            current = ""
        else:
            current += ch
    if current:
        out.append(current)
    return out


def _is_punctuation(ch: str) -> bool:
    """Punctuation as Hugging Face's library sees it: an ASCII sign that is not a letter, digit or space, or a
    character that Unicode calls punctuation."""
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")
