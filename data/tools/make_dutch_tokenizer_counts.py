# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/dutch-tokenizer-counts.tsv: how many tokens seven tokenizers make of all 1,997 sentences of NTREX-128 in
Dutch and in the English source, and how many words those sentences have. Lesson nlp-dutch-models draws its picture of
what Dutch costs each model from it, also before the learner downloads the tokenizers. The file holds numbers only,
none of the text.

Download the sentences and the tokenizers once in the lessons nlp-low-resource and nlp-dutch-models, or with
`uv run python -c "from everything_ai import dutch_models, hub, ntrex; [hub.download(*p) for p in ntrex.FILES +
dutch_models.FILES]"`, and run, from the course folder:

    uv run python data/tools/make_dutch_tokenizer_counts.py

A header line, then one line per tokenizer in the order of everything_ai.dutch_models.KEYS, with tabs between:
`tokenizer`, its key; `size`, the tokens of its vocabulary; `nld_words` and `eng_words`, the words of the sentences as
`str.split` cuts them; `nld_tokens` and `eng_tokens`, the tokens, each sentence encoded on its own, without the special
tokens a model puts around a text.
"""
from everything_ai import dutch_models, ntrex, paths

OUT = paths.ROOT / "data" / "dutch-tokenizer-counts.tsv"
COLUMNS = ["tokenizer", "size", "nld_words", "nld_tokens", "eng_words", "eng_tokens"]


def main() -> None:
    texts = {code: ntrex.sentences(code, 1997) for code in ("nld", "eng")}
    assert len(texts["nld"]) == len(texts["eng"]) == 1997
    lines = ["\t".join(COLUMNS)]
    for key in dutch_models.KEYS:
        enc = dutch_models.tokenizer(key)
        row = [key, dutch_models.SIZES[key]]
        for code in ("nld", "eng"):
            row += [sum(len(s.split()) for s in texts[code]), sum(len(enc.encode(s)) for s in texts[code])]
        lines.append("\t".join(str(x) for x in row))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
