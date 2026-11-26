# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/ntrex-token-counts.tsv: for the first 100 sentences of NTREX-128 in each of the 48 languages the course
pins (everything_ai.ntrex), how many words, characters and bytes they have, and how many tokens the tokenizers of GPT-2
and Qwen3 make of them. Lesson nlp-low-resource draws its picture of what a language costs from it, also before the
learner downloads the sentences themselves. The file holds numbers only, none of the text.

Download the sentences and Qwen3's tokenizer once in the lesson, or with
`uv run python -c "from everything_ai import hub, ntrex, qwen3; [hub.download(*p) for p in ntrex.FILES];
[hub.download(qwen3.REPO, f) for f in ('vocab.json', 'tokenizer_config.json')]"`, and run, from the course folder:

    uv run python data/tools/make_ntrex_counts.py

A header line, then one line per language in the order of everything_ai.ntrex.LANGUAGES, with tabs between:
`code`, the language's code in NTREX; `joshi`, its class in the taxonomy of Joshi and others (2020), from 0, almost no
data, to 5, the most; `words`, the words of the 100 sentences as `str.split` cuts them; `characters`; `bytes`, in UTF-8;
`gpt2` and `qwen3`, the tokens of the 100 sentences, each sentence encoded on its own; and `gpt2_continued` and
`qwen3_continued`, how many of the words the tokenizer cuts into more than one token, each word encoded with a space
before it, as it stands in a sentence. Qwen3's tokenizer first puts the text in Unicode's form NFC, as Qwen's own code
does (everything_ai.qwen3.encode): of the Yoruba sentences, which write some accents as separate characters, it then
makes 8,984 tokens instead of 11,206.
"""
import tiktoken

from everything_ai import gpt2_tokenizer, ntrex, paths, qwen3

OUT = paths.ROOT / "data" / "ntrex-token-counts.tsv"

# The class of each language in the taxonomy of Pratik Joshi, Sebastin Santy, Amar Budhiraja, Kalika Bali and Monojit
# Choudhury, *The State and Fate of Linguistic Diversity and Inclusion in the NLP World* (ACL 2020), as their list
# lang2tax.txt gives it (microsoft.github.io/linguisticdiversity/assets/lang2tax.txt; Chinese is "mandarin" there).
JOSHI = {
    "eng": 5, "nld": 4, "afr": 3, "amh": 2, "arb": 5, "ben": 3, "bod": 1, "cym": 1, "deu": 5, "ell": 3, "eus": 4,
    "ewe": 1, "fao": 1, "fas": 4, "fin": 4, "fra": 5, "gle": 2, "hau": 2, "heb": 3, "hin": 4, "hun": 4, "hye": 1,
    "ind": 3, "isl": 2, "jpn": 5, "kat": 3, "khm": 1, "kor": 4, "lao": 2, "ltz": 1, "mri": 1, "pol": 4, "rus": 4,
    "smo": 1, "som": 1, "spa": 5, "swa": 2, "tam": 3, "tel": 1, "tha": 3, "tir": 2, "tur": 4, "urd": 3, "vie": 4,
    "wol": 2, "yor": 2, "zho-CN": 5, "zul": 2,
}
COLUMNS = ["code", "joshi", "words", "characters", "bytes", "gpt2", "qwen3", "gpt2_continued", "qwen3_continued"]


def counts(sentences: list, encode) -> tuple[int, int]:
    """The tokens of the sentences, and how many of their words are cut into more than one token, by `encode`, a
    function from a text to its token IDs."""
    tokens = sum(len(encode(s)) for s in sentences)
    continued = sum(1 for s in sentences for w in s.split() if len(encode(" " + w)) > 1)
    return tokens, continued


def main() -> None:
    gpt2_tokenizer.offline()
    gpt2, qwen = tiktoken.get_encoding("gpt2"), qwen3.tokenizer()
    lines = ["\t".join(COLUMNS)]
    for code in ntrex.LANGUAGES:
        sentences = ntrex.sentences(code)
        g, gc = counts(sentences, gpt2.encode)
        q, qc = counts(sentences, lambda text: qwen3.encode(qwen, text))
        row = [code, JOSHI[code], sum(len(s.split()) for s in sentences), sum(len(s) for s in sentences),
               sum(len(s.encode("utf-8")) for s in sentences), g, q, gc, qc]
        lines.append("\t".join(str(x) for x in row))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
