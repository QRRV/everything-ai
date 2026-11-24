# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/bert-tiny-fill.npz: what BERT-Tiny guesses for hidden words, so that lesson nlp-bert works without the
model. Three parts:

- `blanks`, eight sentences of the course with one [MASK] each, `answers`, the word the course hid there, and
  `blank_scores`, the 30,522 scores BERT-Tiny gives at the [MASK], one row per sentence;
- `week`, one sentence of the course with two [MASK]s, `week_answers`, the two words hidden there, and `week_scores`,
  the 30,522 scores at every place of its pieces, [CLS] and [SEP] included, one row per place;
- `ranks`, what the course measured on the 500 English sentences of data/tatoeba-langs.tsv.gz: every word of letters
  that BERT-Tiny keeps as one piece, hidden in turn with [MASK]. Column 0 is the place of the hidden word in
  BERT-Tiny's list of guesses with the whole sentence, column 1 the same when every piece after the [MASK] is a
  [MASK] too, so that the model sees only the words before it. A place is the number of pieces with a higher score,
  plus 1. Row by row: sentence by sentence in the order of the file, place by place;
- `piece_ids` and `piece_names`, the pieces of the vocabulary the lesson shows without the model: the ten with the
  highest scores in every row of `blank_scores` and `week_scores`, the hidden words, and the pieces of `week`.

The scores are stored as float16, to keep the file small; the places in `ranks` come from the scores in float32.
BERT-Tiny is google/bert_uncased_L-2_H-128_A-2 (Turc and others, 2019, Apache-2.0), which everything_ai.bert writes
out and everything_ai.hub downloads at a fixed commit. Download it once in the lesson, or with
`uv run python -c "from everything_ai import bert; bert.download()"`, and run, from the course folder:

    uv run python data/tools/make_bert_fill.py

The zip is written with a fixed date, so the same model gives the same file.
"""
import gzip
import io
import zipfile

import numpy as np
import torch

from everything_ai import bert, paths

OUT = paths.ROOT / "data" / "bert-tiny-fill.npz"
TATOEBA = paths.ROOT / "data" / "tatoeba-langs.tsv.gz"

BLANKS = [   # the course's own sentences, and the word hidden in each
    ("We opened the [MASK] and walked in.", "door"),
    ("I drink a cup of [MASK] every morning.", "coffee"),
    ("Monday comes after [MASK].", "sunday"),
    ("The opposite of hot is [MASK].", "cold"),
    ("The keys are in my [MASK].", "pocket"),
    ("The birds built a [MASK] in the tree.", "nest"),
    ("Paris is the [MASK] of France.", "capital"),
    ("It started to [MASK], so we opened our umbrellas.", "rain"),
]
WEEK = ("A week has seven [MASK] and a year has twelve [MASK].", ["days", "months"])


def english() -> list:
    with gzip.open(TATOEBA, "rt", encoding="utf-8") as f:
        rows = [line.rstrip("\n").split("\t") for line in f][1:]
    return [text for lang, _, text in rows if lang == "en"]


def measured(ids: list) -> list:
    """The places of ids that the measurement hides: a word of letters that is one piece."""
    vocab = bert.tokenizer().vocab
    return [i for i in range(1, len(ids) - 1) if vocab[ids[i]].isalpha() and not vocab[ids[i + 1]].startswith("##")]


def ranks() -> np.ndarray:
    model = bert.load()
    out = []
    for text in english():
        ids = bert.tokenizer().encode(text)
        places = measured(ids)
        if not places:
            continue
        both = [ids[:i] + [bert.MASK] + ids[i + 1:] for i in places]
        before = [ids[:i] + [bert.MASK] * (len(ids) - 1 - i) + [bert.SEP] for i in places]
        found = []
        for batch in (both, before):
            with torch.no_grad():
                last = model(torch.tensor(batch))[-1]
                rows = model.scores(last[torch.arange(len(places)), torch.tensor(places)])
            right = rows[torch.arange(len(places)), torch.tensor([ids[i] for i in places])]
            found.append(((rows > right[:, None]).sum(dim=1) + 1).tolist())
        out += list(zip(*found))
    return np.array(out, dtype=np.int32)


def main() -> None:
    blank_scores = []
    for text, answer in BLANKS:
        pieces = bert.pieces(text)
        assert pieces.count("[MASK]") == 1 and answer in bert.tokenizer().ids, text
        blank_scores.append(bert.piece_scores(text)[pieces.index("[MASK]")])
    week, week_answers = WEEK
    assert bert.pieces(week).count("[MASK]") == len(week_answers)
    week_scores = bert.piece_scores(week)
    ids = bert.tokenizer().ids
    shown = {ids[a] for _, a in BLANKS} | {ids[a] for a in week_answers} | {ids[p] for p in bert.pieces(week)}
    for row in list(blank_scores) + list(week_scores):
        shown |= {int(j) for j in np.argsort(-row, kind="stable")[:10]}
    shown = sorted(shown)
    arrays = {
        "blanks": np.array([t for t, _ in BLANKS]),
        "answers": np.array([a for _, a in BLANKS]),
        "blank_scores": np.array(blank_scores, dtype=np.float16),
        "week": np.array([week]),
        "week_answers": np.array(week_answers),
        "week_scores": week_scores.astype(np.float16),
        "ranks": ranks(),
        "piece_ids": np.array(shown, dtype=np.int32),
        "piece_names": np.array([bert.tokenizer().vocab[j] for j in shown]),
    }
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, buffer.getvalue())
    r = arrays["ranks"]
    print(f"{OUT.name}: {len(BLANKS)} blanks, week {arrays['week_scores'].shape}, {len(r)} measured places;"
          f" first guess right {(r[:, 0] == 1).sum()} with both sides, {(r[:, 1] == 1).sum()} with the words before")


if __name__ == "__main__":
    main()
