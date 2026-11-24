# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/bert-tiny-fine-tune.npz: what lesson nlp-fine-tune-bert shows of BERT-Tiny without the model, and
without training it for minutes. Two parts.

Sentences (the sentiment of a sentence, from the vector at [CLS]). Six held-out sentences of
data/uci-sentiment.tsv.gz (every fifth sentence of the file is held out, as in lesson nlp-document-vectors):

- `watch_texts`, `watch_labels` (1 positive, 0 negative), `watch_pieces` (the pieces of each sentence, [CLS] and
  [SEP] included, joined by spaces), `watch_ids` and `watch_mask` (the token ids, padded with 0, and True at a real
  piece), and `watch_layers`: what BERT-Tiny gives for them, before the first block and after each of its 2 blocks,
  as float16 of (3, 6, places, 128);
- `head_weight` and `head_bias`: the layer from 128 numbers to 2 scores (negative, positive) that the course trains
  on the vector at [CLS] after the last block, with BERT-Tiny frozen: on the 2,400 training sentences, AdamW with
  learning rate 0.001, batches of 32 sentences in a random order, 4 epochs, seed 0, as the demo of the lesson does.

Names (token classification). BERT-Tiny fine-tuned by the course on the 8,773 training sentences of
UNER_English-EWT (data/uner-en-ewt.tsv.gz), with one linear layer from 128 numbers to the 7 tags O, B-PER, I-PER,
B-LOC, I-LOC, B-ORG and I-ORG on top of every piece. Every word is cut into pieces on its own; only the first piece
of a word gets the word's tag as its label, and the other pieces, [CLS] and [SEP] get -100, which
nn.CrossEntropyLoss leaves out. AdamW with learning rate 0.0001 and weight decay 0.01, batches of 32 sentences in a
random order, 3 epochs, seed 0. As a comparison, the same with BERT-Tiny frozen: only the layer learns, with
learning rate 0.01. For the 1,471 test sentences, in the order of the file:

- `sent_ids`, `shown` (False for a sentence a lesson does not show), `word_counts` (words per sentence), `words`
  (all words one after the other, in one string with a line break between two words), `gold` (the number of each
  word's tag in `tags`), `piece_counts` (pieces per word) and `pieces` (all pieces one after the other);
- `scores`: the 7 scores of the fine-tuned model at every place of every sentence, [CLS], its pieces and [SEP], one
  row per place, sentence after sentence, as float16;
- `tags`: the 7 tags; `curve`: per epoch the mean loss of its batches and the span F1 on the 1,478 dev and the
  1,471 test sentences, with each word's tag read at its first piece; `frozen`: the same for the frozen model.

Everything runs on one thread. BERT-Tiny is google/bert_uncased_L-2_H-128_A-2 (Turc and others, 2019, Apache-2.0),
which everything_ai.bert writes out and everything_ai.hub downloads at a fixed commit. Download it once in a lesson,
or with `uv run python -c "from everything_ai import bert; bert.download()"`, and run, from the course folder (a few
minutes):

    uv run python data/tools/make_bert_fine_tune.py

The zip is written with a fixed date, so the same model and the same training give the same file.
"""
import copy
import io
import zipfile

import numpy as np
import torch
from torch import nn

from everything_ai import bert, books, paths, uner

OUT = paths.ROOT / "data" / "bert-tiny-fine-tune.npz"
TAGS = ("O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG")
SEED, BATCH = 0, 32
NAMES_EPOCHS, NAMES_LR, NAMES_FROZEN_LR = 3, 1e-4, 1e-2
SENTENCE_EPOCHS, SENTENCE_FROZEN_LR = 4, 1e-3
WATCHED = [("yelp", 240), ("yelp", 310), ("amazon", 320), ("imdb", 670), ("amazon", 205), ("imdb", 155)]   # (source, line)


# --- sentences ---------------------------------------------------------------------------------------------------------

def held_out_and_training() -> tuple:
    """The 2,400 training and 600 held-out sentences, as (label, text): every fifth sentence of the file is held out."""
    rows = books.review_sentences()
    train = [(label, text) for k, (_, _, label, _, text) in enumerate(rows) if k % 5 != 4]
    held = [(label, text) for k, (_, _, label, _, text) in enumerate(rows) if k % 5 == 4]
    return train, held


def sentence_batch(items) -> tuple:
    """Token ids padded with 0, the mask, and the labels, for (label, text) pairs."""
    ids = [bert.tokenizer().encode(text) for _, text in items]
    n = max(len(i) for i in ids)
    x = torch.zeros(len(ids), n, dtype=torch.long)
    mask = torch.zeros(len(ids), n, dtype=torch.bool)
    for k, row in enumerate(ids):
        x[k, :len(row)], mask[k, :len(row)] = torch.tensor(row), True
    return x, mask, torch.tensor([label for label, _ in items])


def sentence_head() -> nn.Linear:
    """The head on the vector at [CLS] after the last block, trained with BERT-Tiny frozen."""
    train, _ = held_out_and_training()
    torch.manual_seed(SEED)
    model = copy.deepcopy(bert.load())
    head = nn.Linear(bert.D, 2)
    opt = torch.optim.AdamW(head.parameters(), lr=SENTENCE_FROZEN_LR, weight_decay=0.0)
    order_from = torch.Generator().manual_seed(SEED)
    for _ in range(SENTENCE_EPOCHS):
        order = torch.randperm(len(train), generator=order_from).tolist()
        for s in range(0, len(order), BATCH):
            x, mask, y = sentence_batch([train[i] for i in order[s:s + BATCH]])
            loss = nn.CrossEntropyLoss()(head(model(x, mask)[-1][:, 0]), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head


def watched() -> dict:
    rows = books.review_sentences()
    found = {(source, line): (label, text) for source, line, label, _, text in rows}
    items = [found[w] for w in WATCHED]
    x, mask, y = sentence_batch(items)
    with torch.no_grad():
        layers = torch.stack(bert.load()(x, mask)).numpy()
    vocab = bert.tokenizer().vocab
    head = sentence_head()
    return {
        "watch_texts": np.array([text for _, text in items]),
        "watch_labels": y.numpy().astype(np.int8),
        "watch_pieces": np.array([" ".join(vocab[int(i)] for i, m in zip(row, keep) if m) for row, keep in zip(x, mask)]),
        "watch_ids": x.numpy().astype(np.int32),
        "watch_mask": mask.numpy(),
        "watch_layers": layers.astype(np.float16),
        "head_weight": head.weight.detach().numpy().astype(np.float32),
        "head_bias": head.bias.detach().numpy().astype(np.float32),
    }


# --- names -------------------------------------------------------------------------------------------------------------

def word_pieces(pairs) -> list:
    """The pieces of each word of a sentence, cut word by word; a word without pieces becomes [UNK]."""
    return [bert.tokenizer().pieces(word) or ["[UNK]"] for word, _ in pairs]


def example(pairs) -> tuple:
    """The token ids of a sentence, [CLS], its pieces and [SEP], and the label at every place: the word's tag at
    its first piece, -100 elsewhere."""
    ids, labels = bert.tokenizer().ids, [-100]
    pieces = [ids["[CLS]"]]
    for (_, tag), ps in zip(pairs, word_pieces(pairs)):
        for j, p in enumerate(ps):
            pieces.append(ids[p])
            labels.append(TAGS.index(tag) if j == 0 else -100)
    assert len(pieces) < bert.MAX, len(pieces)
    return pieces + [ids["[SEP]"]], labels + [-100]


def batch(examples) -> tuple:
    n = max(len(i) for i, _ in examples)
    x = torch.zeros(len(examples), n, dtype=torch.long)
    mask = torch.zeros(len(examples), n, dtype=torch.bool)
    y = torch.full((len(examples), n), -100, dtype=torch.long)
    for k, (i, labels) in enumerate(examples):
        x[k, :len(i)], mask[k, :len(i)], y[k, :len(i)] = torch.tensor(i), True, torch.tensor(labels)
    return x, mask, y


def scores(model, head, sentences) -> list:
    """The scores at every place of each sentence, as float32 arrays of (places, 7)."""
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(sentences), 64):
            examples = [example(p) for p in sentences[s:s + 64]]
            x, mask, _ = batch(examples)
            got = head(model(x, mask)[-1]).numpy()
            out += [got[k, :len(i)] for k, (i, _) in enumerate(examples)]
    return out


def read_tags(rows, pieces) -> list:
    """The tag of each word: the tag with the highest score at its first piece."""
    tags, place = [], 1
    for ps in pieces:
        tags.append(TAGS[int(np.argmax(rows[place]))])
        place += len(ps)
    return tags


def span_f1(model, head, sentences) -> float:
    right = found = names = 0
    for pairs, rows in zip(sentences, scores(model, head, sentences)):
        gold = set(uner.spans([t for _, t in pairs]))
        guess = set(uner.spans(read_tags(rows, word_pieces(pairs))))
        right, found, names = right + len(gold & guess), found + len(guess), names + len(gold)
    return 2 * right / (found + names)


def fine_tune_names(frozen: bool) -> tuple:
    torch.manual_seed(SEED)
    model = copy.deepcopy(bert.load())
    head = nn.Linear(bert.D, len(TAGS))
    if frozen:
        opt = torch.optim.AdamW(head.parameters(), lr=NAMES_FROZEN_LR, weight_decay=0.0)
    else:
        model.requires_grad_(True)
        opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=NAMES_LR, weight_decay=0.01)
    train, dev, test = (uner.split(name) for name in uner.SPLITS)
    examples = [example(p) for p in train]
    order_from = torch.Generator().manual_seed(SEED)
    loss_fn = nn.CrossEntropyLoss()   # its ignore_index is -100
    curve = []
    for epoch in range(1, NAMES_EPOCHS + 1):
        model.train(not frozen)
        order, losses = torch.randperm(len(examples), generator=order_from).tolist(), []
        for s in range(0, len(order), BATCH):
            x, mask, y = batch([examples[i] for i in order[s:s + BATCH]])
            loss = loss_fn(head(model(x, mask)[-1]).reshape(-1, len(TAGS)), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        curve.append((epoch, float(np.mean(losses)), span_f1(model, head, dev), span_f1(model, head, test)))
        print(("frozen" if frozen else "fine-tuned"), curve[-1], flush=True)
    return model, head, np.array(curve)


def names() -> dict:
    test = [s for s in uner.sentences() if s.split == "test"]
    _, _, frozen_curve = fine_tune_names(frozen=True)
    model, head, curve = fine_tune_names(frozen=False)
    pieces = [word_pieces(s.pairs) for s in test]
    rows = scores(model, head, [s.pairs for s in test])
    return {
        "sent_ids": np.array([s.sent_id for s in test]),
        "shown": np.array([s.shown for s in test]),
        "word_counts": np.array([len(s.pairs) for s in test], dtype=np.int32),
        "words": np.array("\n".join(w for s in test for w, _ in s.pairs)),   # one string: one long web address would make
                                                                              # an array of strings 36 MB
        "gold": np.array([TAGS.index(t) for s in test for _, t in s.pairs], dtype=np.int8),
        "piece_counts": np.array([len(ps) for sentence in pieces for ps in sentence], dtype=np.int32),
        "pieces": np.array([p for sentence in pieces for ps in sentence for p in ps]),
        "scores": np.concatenate(rows).astype(np.float16),
        "tags": np.array(TAGS),
        "curve": curve,
        "frozen": frozen_curve,
    }


def main() -> None:
    torch.set_num_threads(1)
    arrays = {**watched(), **names()}
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, buffer.getvalue())
    curve, frozen = arrays["curve"], arrays["frozen"]
    print(f"{OUT.name}: {len(arrays['sent_ids'])} test sentences, {len(arrays['gold'])} words,"
          f" {len(arrays['scores'])} places; span F1 on test {curve[-1][3]:.4f} fine-tuned, {frozen[-1][3]:.4f} frozen")


if __name__ == "__main__":
    main()
