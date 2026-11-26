# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""BERT's WordPiece tokenizer, as Hugging Face's BertTokenizer does it, without PyTorch: it removes control
characters, puts a space around every Chinese character, for an uncased model makes everything lowercase and drops
accents, splits at spaces and punctuation, and cuts every word into the longest pieces of the vocabulary from left to
right; a piece inside a word starts with ##. A word it cannot cut, or of more than 100 characters, becomes [UNK].
everything_ai.bert uses it for BERT-Tiny (uncased), lesson nlp-dutch-models for BERTje (cased)."""
import unicodedata

SPECIAL = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")   # found in the text as they are, before anything else
LONGEST = 100   # characters of the longest word the tokenizer cuts into pieces; a longer one is [UNK]
MAX = 512       # tokens at most, [CLS] and [SEP] included: the rest of a longer text is left out (truncation)


def _chinese(cp: int) -> bool:
    """A CJK ideograph, which gets a space on both sides: Chinese has no spaces between words."""
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF or 0x2A700 <= cp <= 0x2B73F
            or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF or 0xF900 <= cp <= 0xFAFF
            or 0x2F800 <= cp <= 0x2FA1F)


def _punctuation(ch: str) -> bool:
    """Punctuation, which becomes a word of its own: every ASCII sign that is not a letter, digit or space (also $, +
    and ^), and every character that Unicode calls punctuation."""
    return ch.isascii() and ch.isprintable() and not ch.isalnum() and ch != " " or unicodedata.category(ch)[0] == "P"


def _clean(text: str, lower: bool = True) -> str:
    """Normalize as BERT does: control characters out, every space a plain space, a space around every Chinese
    character, and for an uncased model lowercase and no accents (NFD, then every combining mark out)."""
    out = []
    for ch in text:
        cp = ord(ch)
        if cp in (0, 0xFFFD) or unicodedata.category(ch)[0] == "C" and ch not in "\t\n\r":
            continue
        if ch.isspace():
            out.append(" ")
        elif _chinese(cp):
            out.append(f" {ch} ")
        else:
            out.append(ch)
    if not lower:
        return "".join(out)
    text = unicodedata.normalize("NFD", "".join(out))
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn").lower()


def _words(text: str) -> list[str]:
    """The words of a cleaned text: split at spaces, and every punctuation sign a word of its own."""
    words = []
    for chunk in text.split():
        current = ""
        for ch in chunk:
            if _punctuation(ch):
                if current:
                    words.append(current)
                words.append(ch)
                current = ""
            else:
                current += ch
        if current:
            words.append(current)
    return words


class Tokenizer:
    """BERT's WordPiece tokenizer for vocab.txt: one piece per line, its line number its id. `lower` False for a
    cased model, such as BERTje: then it keeps capitals and accents."""

    def __init__(self, vocab: list[str], lower: bool = True):
        self.vocab, self.lower = list(vocab), lower
        self.ids = {piece: i for i, piece in enumerate(self.vocab)}

    def word(self, word: str) -> list[str]:
        """The pieces of one word: from left to right the longest piece in the vocabulary, a piece inside the word
        with ## in front. When some part has no piece, or the word is longer than 100 characters, it is [UNK]."""
        if len(word) > LONGEST:
            return ["[UNK]"]
        pieces, start = [], 0
        while start < len(word):
            end = len(word)
            while end > start:
                piece = word[start:end] if start == 0 else "##" + word[start:end]
                if piece in self.ids:
                    break
                end -= 1
            else:
                return ["[UNK]"]
            pieces.append(piece)
            start = end
        return pieces

    def pieces(self, text: str) -> list[str]:
        """The pieces of a text, without [CLS] and [SEP]. [MASK] and the other special tokens stay as they are."""
        out, rest = [], text
        while rest:
            found = [(rest.find(s), s) for s in SPECIAL if s in rest]
            if not found:
                break
            at, special = min(found)
            for w in _words(_clean(rest[:at], self.lower)):
                out += self.word(w)
            out.append(special)
            rest = rest[at + len(special):]
        for w in _words(_clean(rest, self.lower)):
            out += self.word(w)
        return out

    def encode(self, text: str) -> list[int]:
        """The token ids of a text, as Hugging Face's tokenizer of the model gives them: [CLS], the pieces, [SEP]."""
        return [self.ids["[CLS]"]] + [self.ids[p] for p in self.pieces(text)][:MAX - 2] + [self.ids["[SEP]"]]
