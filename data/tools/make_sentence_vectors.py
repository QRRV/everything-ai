# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/course-sentence-vectors.npz: what two models make of the 32 sentences of the course's 16 pairs in
data/course-sentence-pairs.tsv, so that lesson nlp-sentence-embeddings works without downloading either model.

- Granite Embedding 107M multilingual (ibm-granite/granite-embedding-107m-multilingual, Apache-2.0), the course's
  embedding model, gives each sentence one embedding of 384 numbers, of length 1 (CLS pooling): `granite`.
- BERT-Tiny (google/bert_uncased_L-2_H-128_A-2, Turc and others 2019, Apache-2.0) gives each piece of a sentence a
  vector of 128 numbers after its second block. The 32 sentences go through the model as one batch, as the Sentence
  Transformers library sends 32 texts at a time: every sentence is filled up with [PAD] to the length of the longest,
  and the mask keeps the [PAD] places out of the attention. `tiny` holds every place, [PAD] places too, and `mask` says
  which places hold a real piece ([CLS], the pieces of the sentence, [SEP]): 1, or [PAD]: 0.

Both models are the course's own code (everything_ai.embed and everything_ai.bert), downloaded at a fixed commit by
everything_ai.hub. Download them once in the lessons, or with
`uv run python -c "from everything_ai import bert, embed; bert.download(); embed.download()"`, and run, from the
course folder:

    uv run python data/tools/make_sentence_vectors.py

What the file holds: `granite`, an array of shape (32, 384) in float32; `tiny`, of shape (32, T, 128) in float32, with
T the number of places of the longest sentence; and `mask`, of shape (32, T), whole numbers 0 and 1. Row 2(p - 1) is
the first sentence of pair p, row 2(p - 1) + 1 the second. The zip is written with a fixed date, so the same models
give the same file.
"""
import csv
import io
import zipfile

import numpy as np
import torch

from everything_ai import bert, embed, paths

PAIRS = paths.ROOT / "data" / "course-sentence-pairs.tsv"
OUT = paths.ROOT / "data" / "course-sentence-vectors.npz"


def sentences() -> list:
    with open(PAIRS, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [s for row in rows for s in (row["first"], row["second"])]


def main() -> None:
    texts = sentences()
    granite = embed.embed(texts).numpy().astype(np.float32)
    ids = [bert.tokenizer().encode(t) for t in texts]
    longest = max(len(i) for i in ids)
    x = torch.full((len(ids), longest), bert.PAD)
    mask = torch.zeros(len(ids), longest, dtype=torch.bool)
    for row, i in enumerate(ids):
        x[row, :len(i)] = torch.tensor(i)
        mask[row, :len(i)] = True
    with torch.no_grad():
        tiny = bert.load()(x, mask)[-1].numpy().astype(np.float32)
    arrays = {"granite": granite, "tiny": tiny, "mask": mask.numpy().astype(np.int64)}
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, buffer.getvalue())
    print(f"{OUT.name}: {len(texts)} sentences, granite {granite.shape}, tiny {tiny.shape}")


if __name__ == "__main__":
    main()
