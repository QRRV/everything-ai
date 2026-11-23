# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/bert-tiny-bank.npz: what BERT-Tiny holds for the word bank in each of the course's 44 sentences of
data/course-bank-sentences.tsv, before its first block and after each of its two blocks. Lesson
nlp-static-vs-contextual measures with them how far the uses of bank lie apart, also without the model.

BERT-Tiny is google/bert_uncased_L-2_H-128_A-2 (Turc and others, 2019, Apache-2.0), which everything_ai.bert writes
out and everything_ai.hub downloads at a fixed commit. Download it once in the lesson, or with
`uv run python -c "from everything_ai import bert; bert.download()"`, and run, from the course folder:

    uv run python data/tools/make_bert_bank.py

What the file holds: `layers`, an array of shape (44, 3, 128) in float32, and `place`, an array of 44 whole numbers.
Row i belongs to line i + 2 of the TSV file (after its header): `place[i]` is the place of bank among the pieces the
model sees, [CLS] at place 0, and `layers[i, k]` what bank holds there before the first block (k = 0) and after block
k. The zip is written with a fixed date, so the same model gives the same file.
"""
import csv
import io
import zipfile

import numpy as np

from everything_ai import bert, paths

SENTENCES = paths.ROOT / "data" / "course-bank-sentences.tsv"
OUT = paths.ROOT / "data" / "bert-tiny-bank.npz"


def main() -> None:
    with open(SENTENCES, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    layers, place = [], []
    for row in rows:
        pieces = bert.pieces(row["sentence"])
        i = pieces.index("bank")
        layers.append([layer[i] for layer in bert.layers(row["sentence"])])
        place.append(i)
    arrays = {"layers": np.array(layers, dtype=np.float32), "place": np.array(place, dtype=np.int64)}
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, buffer.getvalue())
    print(f"{OUT.name}: {len(rows)} sentences, layers {arrays['layers'].shape}")


if __name__ == "__main__":
    main()
