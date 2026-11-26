# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""The tokenizers of seven models, for lesson nlp-dutch-models: what each makes of Dutch and of English. Every one has
`encode(text)`, the ids of the tokens of a text without the special tokens a model puts around it, and `shown(text)`,
the tokens themselves: bytes for a tokenizer that works on bytes, a string for BERTje's WordPiece, whose piece inside a
word starts with ##.

- gpt2: GPT-2's own (tiktoken, the course's copy, everything_ai.gpt2_tokenizer), 50,257 tokens;
- fietje: Fietje 2's tokenizer.json, which is Phi-2's: GPT-2's tokens and 38 tokens of spaces and tabs for code;
- mistral: Mistral 7B's (under GEITje, which kept it), 32,000 tokens, SentencePiece-style BPE with byte fallback;
- eurollm: EuroLLM-1.7B's, 128,000 tokens, of the same kind;
- qwen3: Qwen3's (everything_ai.qwen3), 151,669 tokens;
- bertje: BERTje's WordPiece vocabulary, 30,073 pieces, cased (everything_ai.wordpiece);
- robbert: RobBERT v2's byte-level BPE, 40,000 tokens, made for Dutch; its file is in data/, so it works offline.

The others are downloaded once from Hugging Face, pinned in everything_ai.hub (FILES). everything_ai.bpe and
everything_ai.wordpiece read them as data; tests/test_bpe.py holds what Hugging Face's own library makes of a set of
sentences."""
import unicodedata

from everything_ai import bpe, gpt2_tokenizer, hub, paths, wordpiece

ROBBERT_FILE = paths.ROOT / "data" / "robbert-v2-tokenizer.json.gz"
QWEN_REPO = "Qwen/Qwen3-0.6B-Base"   # everything_ai.qwen3.REPO, without loading PyTorch, which qwen3 imports
PINS = {
    "bertje": ("GroNLP/bert-base-dutch-cased", "vocab.txt"),
    "mistral": ("mistralai/Mistral-7B-v0.1", "tokenizer.json"),
    "fietje": ("BramVanroy/fietje-2", "tokenizer.json"),
    "eurollm": ("utter-project/EuroLLM-1.7B", "tokenizer.json"),
}
FILES = list(PINS.values()) + [(QWEN_REPO, "vocab.json"), (QWEN_REPO, "tokenizer_config.json")]   # what is downloaded
KEYS = ("gpt2", "fietje", "mistral", "eurollm", "qwen3", "bertje", "robbert")
OFFLINE = ("gpt2", "robbert")   # the two that need no download
SIZES = {"gpt2": 50_257, "fietje": 50_295, "mistral": 32_000, "eurollm": 128_000, "qwen3": 151_669, "bertje": 30_073,
         "robbert": 40_000}   # the tokens of each vocabulary, added tokens included


class Tiktoken:
    """A tiktoken Encoding (GPT-2's, Qwen3's) with `shown`; Qwen3's puts the text in Unicode's form NFC first, as
    Qwen's own code does."""

    def __init__(self, enc, nfc: bool = False):
        self.enc, self.nfc = enc, nfc

    def encode(self, text: str) -> list:
        if self.nfc:
            return self.enc.encode(unicodedata.normalize("NFC", text), allowed_special="all")
        return self.enc.encode(text)

    def decode(self, ids: list) -> str:
        return self.enc.decode(ids)

    def shown(self, text: str) -> list:
        return [self.enc.decode_single_token_bytes(t) for t in self.encode(text)]


class Bpe:
    """A tokenizer of everything_ai.bpe with `shown`."""

    def __init__(self, tok: bpe.Tokenizer):
        self.tok = tok

    def encode(self, text: str) -> list:
        return self.tok.encode(text)

    def decode(self, ids: list) -> str:
        return self.tok.decode(ids)

    def shown(self, text: str) -> list:
        return [self.tok.piece_bytes(p) for p in self.tok.pieces(text)]


class WordPiece:
    """BERTje's tokenizer: `encode` gives the ids of the pieces without [CLS] and [SEP], `shown` the pieces."""

    def __init__(self, tok: wordpiece.Tokenizer):
        self.tok = tok

    def encode(self, text: str) -> list:
        return [self.tok.ids[p] for p in self.tok.pieces(text)]

    def decode(self, ids: list) -> str:
        out = ""
        for i in ids:
            piece = self.tok.vocab[i]
            out += piece[2:] if piece.startswith("##") else (" " if out else "") + piece
        return out

    def shown(self, text: str) -> list:
        return self.tok.pieces(text)


def ready() -> bool:
    """Whether the tokenizers that need a download are downloaded."""
    return all(hub.pinned(*pin) is not None for pin in FILES)


def robbert() -> Bpe:
    """RobBERT v2's tokenizer, from the course's copy in data/."""
    return Bpe(bpe.Tokenizer(ROBBERT_FILE))


def gpt2() -> Tiktoken:
    import tiktoken
    gpt2_tokenizer.offline()
    return Tiktoken(tiktoken.get_encoding("gpt2"))


def tokenizer(key: str):
    """One of the seven tokenizers, by its key in KEYS. FileNotFoundError when its file is not downloaded yet."""
    if key == "gpt2":
        return gpt2()
    if key == "robbert":
        return robbert()
    if key == "qwen3":
        from everything_ai import qwen3
        return Tiktoken(qwen3.tokenizer(), nfc=True)
    path = hub.pinned(*PINS[key])
    if path is None:
        raise FileNotFoundError(f"the tokenizer of {key} is not downloaded yet")
    if key == "bertje":
        return WordPiece(wordpiece.Tokenizer(path.read_text(encoding="utf-8").split("\n"), lower=False))
    return Bpe(bpe.Tokenizer(path))
