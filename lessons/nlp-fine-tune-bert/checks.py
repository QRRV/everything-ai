# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-fine-tune-bert: two scores for a sentence from the vector at [CLS] (classify), the
labels of the pieces of a sentence for token classification, with -100 where the loss leaves a place out
(piece_labels), and the tag of each word read at its first piece (word_tags); with a picture of the idea, the
pictures of what each function gives, the demo that fine-tunes BERT-Tiny on the learner's computer, its rewind and
the explore.

What the exercises and their pictures need of BERT-Tiny the course ships (data/bert-tiny-fine-tune.npz, made by
data/tools/make_bert_fine_tune.py): BERT-Tiny's numbers for six held-out sentences with a head the course trained on
the vector at [CLS], and the scores of BERT-Tiny fine-tuned by the course to find names, on the 1,471 test sentences
of Universal NER. BERT-Tiny itself (18 MB, everything_ai.bert) the learner downloads once for the worked example, the
demo and the explore. Every number in the text and the pictures is computed by the course (tests/
test_nlp_fine_tune_bert.py computes them again)."""
import ast
import contextlib
import copy
import functools
import html
import io
import math
import re
from typing import NamedTuple

import numpy as np
import torch
from torch import nn

from everything_ai import books, lesson as lesson_helpers, live, netview, paths, uner, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, kept, run_code, same_value, with_time_limit
from everything_ai.lesson import download_size

_KEY = "nlp-fine-tune-bert"
REPO = "google/bert_uncased_L-2_H-128_A-2"   # everything_ai.bert.REPO: that module loads the model, so not here
BERT_FILES = [(REPO, "model.safetensors"), (REPO, "vocab.txt")]
DATA = paths.ROOT / "data" / "bert-tiny-fine-tune.npz"
TAGS = ("O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG")   # the 7 tags, numbered 0 to 6
IGNORE = -100   # the label nn.CrossEntropyLoss leaves out: its ignore_index
LABELS = {"en": {1: "positive", 0: "negative"}, "nl": {1: "positief", 0: "negatief"}}

# The demo: BERT-Tiny with a new head on the vector at [CLS], on the 2,400 training sentences of the reviews, first
# frozen with only the head learning, then all of it; 4 epochs of batches of 32 sentences in a random order each.
EPOCHS, BATCH, SEED = 4, 32, 0
RUNS = ("frozen", "tuned")
RATES = {"frozen": 1e-3, "tuned": 1e-4}
STEPS = EPOCHS * 75   # 2,400 sentences in batches of 32: 75 steps per epoch
LOOK_EVERY = 25       # the steps between two looks the rewind keeps: 13 looks, at the end of every epoch too
DEMO_SECONDS = 600    # what both runs of the demo may take together on a slow computer
# What the course measured with its own code over 5 seeds (0 to 4): the held-out sentences each run labels right after
# 4 epochs, and what logistic regression on the words got in the lesson Logistic regression on words.
REFERENCE = {"frozen": (430, 449), "tuned": (496, 502), "words": 491}
TAGGER = {"right": 343, "found": 443, "names": 802, "f1": 0.551}   # the tagger of Finding names, in Scoring a tagger
LARGE_F1 = 0.858   # Mayhew and others, a far larger model trained on the same English training set (Scoring a tagger)

DOWNLOAD_LINE = {   # what the lesson downloads, for the line under the header
    "en": f"Works on every computer. The worked example, the demo and the explore download BERT-Tiny, {download_size(BERT_FILES, 'en')}, once from Hugging Face, when you click its button in the lesson; the demo then fine-tunes it on your computer, in about a minute. The exercises and the rest of the lesson work without it.",
    "nl": f"Werkt op elke computer. Het uitgewerkte voorbeeld, de demo en het onderdeel om zelf te proberen downloaden één keer BERT-Tiny, {download_size(BERT_FILES, 'nl')}, van Hugging Face, als je in de les op de knop ervoor klikt; de demo finetunet het model daarna op je computer, in ongeveer een minuut. De oefeningen en de rest van de les werken zonder.",
}


# --- the shipped data ----------------------------------------------------------------------------------------------------

@functools.cache
def _data() -> dict:
    """The arrays of data/bert-tiny-fine-tune.npz, read once and not writable: the functions below hand out copies."""
    with np.load(DATA, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    for a in out.values():
        a.setflags(write=False)
    return out


class Sentence(NamedTuple):
    words: tuple        # the words of a test sentence of Universal NER
    gold: tuple         # the number of each word's tag in TAGS
    pieces: tuple       # the pieces of each word, as a tuple per word
    start: int          # the first row of its scores in the file's `scores`
    shown: bool         # False for a sentence the lesson does not show


@functools.cache
def test_sentences() -> tuple:
    """The 1,471 test sentences, in the order of the file."""
    d = _data()
    words = str(d["words"]).split("\n")
    out, w, p, row = [], 0, 0, 0
    for n, shown in zip(d["word_counts"], d["shown"]):
        n = int(n)
        counts = [int(c) for c in d["piece_counts"][w:w + n]]
        pieces, at = [], p
        for c in counts:
            pieces.append(tuple(str(x) for x in d["pieces"][at:at + c]))
            at += c
        out.append(Sentence(tuple(words[w:w + n]), tuple(int(t) for t in d["gold"][w:w + n]),
                            tuple(pieces), row, bool(shown)))
        row += sum(counts) + 2
        w, p = w + n, at
    return tuple(out)


def word_pieces(k: int) -> list:
    """The pieces of each word of test sentence k, as a fresh list of lists."""
    return [list(ps) for ps in test_sentences()[k].pieces]


def test_scores(k: int) -> np.ndarray:
    """The 7 scores of the course's fine-tuned BERT-Tiny at every place of test sentence k, [CLS] and [SEP]
    included, as a fresh float32 array of (places, 7)."""
    s = test_sentences()[k]
    n = sum(len(ps) for ps in s.pieces) + 2
    return _data()["scores"][s.start:s.start + n].astype(np.float32)


def gold_tags(k: int) -> list:
    return list(test_sentences()[k].gold)


def course_names() -> dict:
    """The span F1 of the course's fine-tuned BERT-Tiny and of the frozen one on the dev and the test sentences, per
    epoch, as the tool measured them."""
    d = _data()
    return {"tuned": d["curve"].tolist(), "frozen": d["frozen"].tolist()}


def spans_of(tags: list) -> list:
    """The spans of a list of tag numbers, as uner.spans reads tags."""
    return uner.spans([TAGS[t] if isinstance(t, int) and 0 <= t < len(TAGS) else "O" for t in tags])


def span_counts(tag_lists: list) -> tuple:
    """(names found right, names found, names) over the test sentences, for one list of tag numbers per sentence."""
    right = found = names = 0
    for s, tags in zip(test_sentences(), tag_lists):
        gold, guess = set(spans_of(list(s.gold))), set(spans_of(tags))
        right, found, names = right + len(gold & guess), found + len(guess), names + len(gold)
    return right, found, names


def f1(counts: tuple) -> float:
    right, found, names = counts
    return 2 * right / (found + names) if found + names else 0.0


@functools.cache
def course_tags() -> tuple:
    """The tags the course's word_tags reads from the shipped scores, one list per test sentence."""
    return tuple(word_tags(test_scores(k), word_pieces(k)) for k in range(len(test_sentences())))


@functools.cache
def course_counts() -> tuple:
    return span_counts(list(course_tags()))


def watched() -> list:
    """The six held-out sentences the classify picture and the demo watch, as (text, label, pieces)."""
    d = _data()
    return [(str(t), int(l), str(p).split(" ")) for t, l, p in zip(d["watch_texts"], d["watch_labels"], d["watch_pieces"])]


class Shipped(nn.Module):
    """BERT-Tiny for the six watched sentences, from the shipped numbers: called with their padded token ids it gives
    the list of 3 tensors BERT-Tiny gives, before the first block and after each block. The exercise's pictures work
    with it without the model. A batch of other sentences it does not know."""

    def __init__(self):
        super().__init__()
        d = _data()
        self.ids = torch.tensor(d["watch_ids"], dtype=torch.long)
        self.mask = torch.tensor(d["watch_mask"])
        self.layers = [torch.tensor(a, dtype=torch.float32) for a in d["watch_layers"]]
        self.unmasked = False   # True once it was called without the mask of these sentences: it knows no numbers for that

    def forward(self, ids, mask=None, segments=None):
        rows = [next(k for k in range(len(self.ids)) if torch.equal(self.ids[k], r)) for r in ids]
        try:
            same = mask is not None and torch.equal(torch.as_tensor(mask).bool(), self.mask[rows])
        except Exception:   # noqa: BLE001   a learner's mask can be anything
            same = False
        self.unmasked = self.unmasked or not same
        return [a[rows].clone() for a in self.layers]


def course_head() -> nn.Linear:
    """The head the course trained on the vector at [CLS] after the last block, with BERT-Tiny frozen."""
    d = _data()
    head = nn.Linear(128, 2)
    with torch.no_grad():
        head.weight.copy_(torch.tensor(d["head_weight"]))
        head.bias.copy_(torch.tensor(d["head_bias"]))
    return head


def watched_batch() -> tuple:
    """The padded token ids and the mask of the six watched sentences."""
    d = _data()
    return torch.tensor(d["watch_ids"], dtype=torch.long), torch.tensor(d["watch_mask"])


def bert_or_stop(button, texts: dict):
    """everything_ai.bert for the lesson's cells, once BERT-Tiny is downloaded. Until then the cell stops with
    `texts["ask"]` and the button; a click downloads it, with the progress in the cell."""
    lesson_helpers.pins_or_stop(button, texts, BERT_FILES)
    from everything_ai import bert
    return bert


# --- the course's own versions of the exercises ------------------------------------------------------------------------

def classify(model, head, ids, mask):
    return head(model(ids, mask)[-1][:, 0])


def piece_labels(word_pieces, tags):
    labels = [-100]
    for i in range(len(word_pieces)):
        labels.append(tags[i])
        for piece in word_pieces[i][1:]:
            labels.append(-100)
    labels.append(-100)
    return labels


def word_tags(scores, word_pieces):
    tags = []
    place = 1
    for pieces in word_pieces:
        tags.append(int(np.argmax(scores[place])))
        place = place + len(pieces)
    return tags


# --- the demo: fine-tuning BERT-Tiny on the reviews -----------------------------------------------------------------------

def sentences() -> tuple:
    """The 2,400 training and the 600 held-out sentences of the reviews, as (label, text): every fifth is held out, as
    in the lesson A document as a vector."""
    rows = books.review_sentences()
    train = [(label, text) for k, (_, _, label, _, text) in enumerate(rows) if k % 5 != 4]
    held = [(label, text) for k, (_, _, label, _, text) in enumerate(rows) if k % 5 == 4]
    return train, held


def batch(bert, items) -> tuple:
    """Token ids padded with 0, the mask and the labels of (label, text) pairs."""
    ids = [bert.tokenizer().encode(text) for _, text in items]
    n = max(len(i) for i in ids)
    x = torch.zeros(len(ids), n, dtype=torch.long)
    mask = torch.zeros(len(ids), n, dtype=torch.bool)
    for k, row in enumerate(ids):
        x[k, :len(row)], mask[k, :len(row)] = torch.tensor(row), True
    return x, mask, torch.tensor([label for label, _ in items])


def held_out(bert, model, head, classify_, held) -> tuple:
    """(sentences labeled right, mean loss) on the held-out sentences, in eval mode without a graph."""
    with netview.looking(model), netview.looking(head):
        right, total = 0, 0.0
        for s in range(0, len(held), 100):
            x, mask, y = batch(bert, held[s:s + 100])
            scores = classify_(model, head, x, mask)
            right += int((scores.argmax(1) == y).sum())
            total += float(nn.CrossEntropyLoss(reduction="sum")(scores, y))
    return right, total / len(held)


@functools.cache
def cls_scale() -> float:
    """The largest size of a number of the frozen BERT-Tiny's vectors at [CLS] of the watched sentences: the strips of
    the demo use it, so that the same number has the same shade at every step."""
    return float(np.abs(_data()["watch_layers"][2][:, 0].astype(np.float32)).max())


def _look(model, head, classify_) -> dict:
    """What the model makes now of the six watched sentences: the probability of positive for each, and its vector at
    [CLS] after the last block. In eval mode without a graph, so the run stays the same (decision 1.17)."""
    x, mask = watched_batch()
    with netview.looking(model), netview.looking(head):
        p = torch.softmax(classify_(model, head, x, mask), dim=1)[:, 1]
        cls = model(x, mask)[-1][:, 0]
    return {"p": [float(v) for v in p], "cls": cls.numpy().astype(np.float32)}


def _run(bert, name: str, classify_, show=None) -> dict:
    """One run of 4 epochs: "frozen", only the new head learns; "tuned", the head and all of BERT-Tiny. The loss of
    every step, a look every LOOK_EVERY steps for the rewind, and the held-out sentences before the first step and
    after every epoch; `show` hears every step."""
    train, held = sentences()
    torch.manual_seed(SEED)
    model = copy.deepcopy(bert.load())
    head = nn.Linear(128, 2)
    if name == "tuned":
        model.requires_grad_(True)
        opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=RATES[name], weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(head.parameters(), lr=RATES[name], weight_decay=0.0)
    order_from = torch.Generator().manual_seed(SEED)
    out = {"losses": [], "looks": [(0, _look(model, head, classify_))],
           "held": [(0, *held_out(bert, model, head, classify_, held))]}
    step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train(name == "tuned")
        order = torch.randperm(len(train), generator=order_from).tolist()
        for s in range(0, len(order), BATCH):
            x, mask, y = batch(bert, [train[i] for i in order[s:s + BATCH]])
            loss = nn.CrossEntropyLoss()(classify_(model, head, x, mask), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            out["losses"].append((step, loss.item()))
            report = {"run": name, "step": step, "steps": STEPS, "epoch": epoch, "epochs": EPOCHS, "loss": out["losses"][-1][1],
                      "look": lambda: _look(model, head, classify_)}
            if step % LOOK_EVERY == 0:
                out["looks"].append((step, _look(model, head, classify_)))
            if step == 1:   # the held-out loss before the first step, where the dashed line starts
                report["held_start"] = out["held"][0][2]
            if s + BATCH >= len(order):
                out["held"].append((step, *held_out(bert, model, head, classify_, held)))
                report["right"], report["held_loss"] = out["held"][-1][1:]
            if show is not None:
                show(report)
    out["model"], out["head"] = model, head
    return out


def demo(text, show=None) -> dict:
    """The demo with the learner's classify once it is right, else the course's: both runs, first "frozen" and then
    "tuned", and whose classify it was. If the learner's classify fails here after all, the course's runs instead,
    from step 1 again."""
    from everything_ai import bert
    ran = run_code(text) if text else None
    fn = ran.names.get("classify") if ran is not None and ran.error is None else None
    mine = callable(fn)

    def runs(classify_):
        return {name: _run(bert, name, classify_, show) for name in RUNS}

    try:
        return {"runs": with_time_limit(lambda: runs(fn if mine else classify), DEMO_SECONDS), "yours": mine}
    except Exception:   # noqa: BLE001   the learner's classify fails on the real model: the course's runs
        if not mine:
            raise
        return {"runs": with_time_limit(lambda: runs(classify), DEMO_SECONDS), "yours": False}


def probability(bert, model, head, text: str) -> float:
    """The probability of positive a trained model gives a sentence, with the course's classify."""
    x, mask, _ = batch(bert, [(0, text)])
    with netview.looking(model), netview.looking(head):
        return float(torch.softmax(classify(model, head, x, mask), dim=1)[0, 1])


# --- pictures: helpers ---------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_HEAD_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);font-weight:600;margin-top:0.6rem"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:1px;border-radius:3px"
_ROW = "display:flex;align-items:center;gap:0.5rem;margin:0.15rem 0"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);width:15rem;flex:none"
_TRACK = "flex:1;min-width:3rem;height:0.65rem;background:var(--ea-line);border-radius:2px"
_NUM = "font-family:var(--ea-font-code);font-size:0.8rem;width:3.6rem;flex:none;text-align:right"
_PART = ("display:inline-flex;flex-direction:column;align-items:center;gap:0.15rem;margin:0 0.35rem 0.5rem 0;"
         "vertical-align:top")
_CHIP = ("display:block;border-radius:4px;padding:0 0.3rem;font-family:var(--ea-font-code);font-size:0.78rem;"
         "line-height:1.5;white-space:nowrap")
_ROLE = {   # the line or bar and the color of its text: the data's, the course's, the learner's, or wrong
    "data": ("var(--ea-ink-3)", "var(--ea-ink-2)"),
    "course": ("var(--ea-ink)", "var(--ea-ink)"),
    "mine": ("var(--ea-pred)", "var(--ea-pred-text)"),
    "off": ("var(--ea-loss)", "var(--ea-loss-text)"),
}


def _marked(text: str) -> str:
    """Escaped text in which `name` is set as code and *word* in italics."""
    out = []
    for k, part in enumerate(html.escape(text).split("`")):
        if k % 2:
            out.append(f"<code>{part}</code>")
        else:
            out.append("".join(f"<i>{p}</i>" if j % 2 else p for j, p in enumerate(part.split("*"))))
    return "".join(out)


def _bare(text: str) -> str:
    return text.replace("*", "").replace("`", "")


def _said(template: str, **values) -> tuple:
    """A note as HTML and as plain text, with the values filled in after the marking: a * or a ` in a value stays as
    it is."""
    out, plain = _marked(template), _bare(template)
    for key, value in values.items():
        out = out.replace("{" + key + "}", html.escape(str(value)))
        plain = plain.replace("{" + key + "}", str(value))
    return out, plain


def _head(text: str) -> str:
    return f'<div style="{_HEAD_UI}">{_marked(text)}</div>'


def _n(x: int, lang: str) -> str:
    """A whole number with a separator for thousands: 1,471 in English, 1.471 in Dutch."""
    text = f"{x:,}"
    return text.replace(",", ".") if lang == "nl" else text


def _d(x: float, lang: str, digits: int = 3) -> str:
    """A number with three decimals: 0.602 in English, 0,602 in Dutch."""
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def _repr(v) -> str:
    """How Python shows a value, also for a value whose repr stops with an error."""
    try:
        return repr(v)
    except Exception:   # noqa: BLE001   a learner's value can be anything
        return "<" + type(v).__name__ + ">"


def _short(v, limit: int = 60) -> str:
    text = _rounded(v)
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _bar(label: str, value, role: str, lang: str, off: bool = False, width_label: str = "") -> str:
    """A row with a label, a bar from 0 to 1 and the number."""
    line, color = _ROLE[role]
    width = max(0.0, min(1.0, float(value))) * 100
    style = _NUM + f";color:{color}" + (";" + _OFF if off else "")
    label_style = _LABEL + (f";width:{width_label}" if width_label else "")
    return (f'<div style="{_ROW}"><span style="{label_style}">{label}</span>'
            f'<span style="{_TRACK}"><span style="display:block;width:{width:.1f}%;height:100%;background:{line};'
            f'border-radius:2px"></span></span><span style="{style}">{html.escape(_d(float(value), lang, 2))}</span></div>')


def _figure(title: str, rows: list, said: list, notes: list) -> str:
    body = "".join(rows) + "".join(f'<div style="{_UI}">{h}</div>' for h, _ in notes)
    return yours.figure(title, body, alt=f'{title}. {"; ".join(said)}. {" ".join(p for _, p in notes)}')


def _chip(text: str, role: str, off: bool = False, faint: bool = False) -> str:
    """A small box: gray for the data, in the color of the text for the course's, blue for the learner's, purple
    dashed where it differs."""
    if role == "data":
        style = _CHIP + ";border:1px solid var(--ea-ink-3);color:var(--ea-ink)"
    elif role == "course":
        style = _CHIP + ";border:1px solid var(--ea-ink);color:var(--ea-ink);font-weight:600"
    else:
        style = _CHIP + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
    if faint:
        style += ";opacity:0.55"
    if off:
        style += ";" + _OFF
    return f'<span style="{style}">{html.escape(text)}</span>'


def _column(*parts: str) -> str:
    return f'<span style="{_PART}">{"".join(parts)}</span>'


# --- running the learner's function in a picture -------------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _quiet gives for a function that stopped with an error
_TOO_SLOW = object()  # what _timed gives when the learner's function takes too long; None is a value of theirs
PICTURE_SECONDS = 2   # what the learner's function gets for all the calls of a picture


def _quiet_many(fn, calls) -> list:
    """The learner's function on many calls, each with fresh copies of its arguments, without what it prints;
    _STOPPED for a call that stops with an error."""
    out = []
    with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
        for args in calls:
            try:
                out.append(fn(*(copy.deepcopy(a) for a in args)))
            except Timeout:
                raise
            except Exception:   # noqa: BLE001   the check above says what went wrong
                out.append(_STOPPED)
    return out


def _timed(make):
    """make() within PICTURE_SECONDS, or _TOO_SLOW when it takes longer."""
    try:
        return with_time_limit(make, PICTURE_SECONDS)
    except Timeout:
        return _TOO_SLOW


def _tag_list(got) -> list | None:
    """The learner's outcome as a list of whole numbers, or None."""
    if not isinstance(got, (list, tuple)):
        return None
    out = []
    for g in got:
        if isinstance(g, (bool, np.bool_)):
            return None
        if isinstance(g, (int, np.integer)):
            out.append(int(g))
        elif isinstance(g, torch.Tensor) and g.numel() == 1 and not g.dtype.is_floating_point and g.dtype != torch.bool:
            out.append(int(g.item()))
        else:
            return None
    return out


def _tag_name(t) -> str:
    if t == IGNORE:
        return "-100"
    if isinstance(t, int) and 0 <= t < len(TAGS):
        return TAGS[t]
    return _short(t, 8)


# --- the picture of the idea: two heads on BERT-Tiny ----------------------------------------------------------------------

IDEA_WORDS = [["ada"], ["love", "##lace"], ["lived"], ["in"], ["london"], ["."]]
IDEA_TAGS = [1, 2, 0, 0, 3, 0]

_IDEA_TEXT = {
    "en": {"title": "Two heads on the same vectors",
           "pieces": "The pieces BERT-Tiny reads",
           "sentence": "A sentence: one head reads only the vector at [CLS]",
           "out": "2 scores: negative, positive",
           "tokens": "Token classification: one head at every place, trained with these labels",
           "legend": "Gray: the pieces of *Ada Lovelace lived in London.* as BERT-Tiny cuts them, each word on its "
                     "own. In the color of the text: the places a head learns from. Faint: a place the head leaves out. "
                     "In the middle row, every piece but [CLS]; in the last row, a place with -100 as its label, which "
                     "the loss leaves out: [CLS], [SEP], and *##lace*, the second piece of *Lovelace*."},
    "nl": {"title": "Twee heads op dezelfde vectoren",
           "pieces": "De stukjes die BERT-Tiny leest",
           "sentence": "Een zin: één head leest alleen de vector bij [CLS]",
           "out": "2 scores: negatief, positief",
           "tokens": "Token classification: één head op elke plek, getraind met deze labels",
           "legend": "Grijs: de stukjes van *Ada Lovelace lived in London.* zoals BERT-Tiny de zin knipt, elk woord "
                     "apart. In de kleur van de tekst: de plekken waar een head van leert. Vaag: een plek die de head "
                     "overslaat. In de middelste rij elk stukje behalve [CLS]; in de onderste rij een plek met -100 als "
                     "label, die de loss overslaat: [CLS], [SEP], en *##lace*, het tweede stukje van *Lovelace*."},
}


def heads_picture(lang: str) -> str:
    """The pieces of a sentence of the course, the place the sentence head reads, and the labels of token
    classification: a tag at the first piece of each word, -100 at the other places."""
    w = _IDEA_TEXT[lang]
    pieces = ["[CLS]"] + [p for ps in IDEA_WORDS for p in ps] + ["[SEP]"]
    labels = piece_labels(IDEA_WORDS, IDEA_TAGS)
    row = "".join(_column(_chip(p, "data")) for p in pieces)
    head_row = "".join(_column(_chip(p, "course") if k == 0 else _chip(p, "data", faint=True))
                       for k, p in enumerate(pieces))
    tag_row = "".join(_column(_chip(p, "data", faint=lab == IGNORE),
                              _chip(_tag_name(lab), "course", faint=lab == IGNORE)) for p, lab in zip(pieces, labels))
    rows = [_head(w["pieces"]), f"<div>{row}</div>", _head(w["sentence"]), f"<div>{head_row}</div>",
            f'<div style="{_UI}">{_marked("[CLS] → " + w["out"])}</div>', _head(w["tokens"]), f"<div>{tag_row}</div>"]
    said = [" ".join(pieces), " ".join(f"{p}: {_tag_name(lab)}" for p, lab in zip(pieces, labels))]
    return _figure(w["title"], rows, said, [_said(w["legend"])])


# --- the picture under exercise 1: what your classify gives for six held-out sentences --------------------------------

_CLASSIFY_TEXT = {
    "en": {"title": "Your classify on six held-out sentences, with BERT-Tiny frozen",
           "read": "Your function reads the vector at place {place} {layer}, with the head the course trained on "
                   "the vector at place 0, [CLS], after the last block.",
           "right": "Your function reads the vector at [CLS] after the last block: the vector the head was trained on.",
           "other": "Your function gives the head something other than one vector per sentence at one place of one of "
                    "the 3 tensors.",
           "layer": {0: "before the first block", 1: "after the first block", 2: "after the last block"},
           "count": "With these numbers, {right} of the 6 sentences get the right label.",
           "legend": "Each row: a held-out sentence of the reviews, its label in gray, and the probability of "
                     "positive from the 2 scores your function gives, after softmax, in blue. Purple dashed: a "
                     "probability that differs from the course's.",
           "odd": "Your function gives back {got}, and not 2 scores for each of the 6 sentences.",
           "no_mask": "Your function gives BERT-Tiny no `mask`, or another one: BERT-Tiny would then read the [PAD]s of the "
                      "shorter sentences as pieces. The course has BERT-Tiny's numbers for these sentences only with "
                      "their mask, so the picture shows nothing more.",
           "stop": "Your function stops with an error on the six sentences.",
           "slow": "Your function takes too long on the six sentences."},
    "nl": {"title": "Je classify op zes achtergehouden zinnen, met BERT-Tiny bevroren",
           "read": "Je functie leest de vector op plek {place} {layer}, met de head die de course trainde op de "
                   "vector op plek 0, [CLS], na het laatste block.",
           "right": "Je functie leest de vector bij [CLS] na het laatste block: de vector waarop de head getraind is.",
           "other": "Je functie geeft de head iets anders dan één vector per zin op één plek van één van de 3 tensors.",
           "layer": {0: "vóór het eerste block", 1: "na het eerste block", 2: "na het laatste block"},
           "count": "Met deze getallen krijgen {right} van de 6 zinnen het goede label.",
           "legend": "Elke rij: een achtergehouden zin uit de recensies, het label in grijs, en de kans op positief "
                     "uit de 2 scores die je functie geeft, na softmax, in blauw. Paars gestreept: een kans die afwijkt "
                     "van de kans van de course.",
           "odd": "Je functie geeft {got} terug, en geen 2 scores voor elk van de 6 zinnen.",
           "no_mask": "Je functie geeft BERT-Tiny geen `mask`, of een andere: BERT-Tiny zou de [PAD]s van de kortere "
                      "zinnen dan als stukjes lezen. De course heeft de getallen van BERT-Tiny voor deze zinnen alleen "
                      "met de bijbehorende mask, dus de afbeelding laat verder niets zien.",
           "stop": "Je functie stopt met een fout op de zes zinnen.",
           "slow": "Je functie duurt te lang op de zes zinnen."},
}


def _which_place(scores: torch.Tensor, model, head, ids, mask):
    """(layer, place) whose vectors give these scores through the head, or None."""
    layers = model(ids, mask)
    with torch.no_grad():
        for layer in (2, 1, 0):
            for place in range(layers[layer].shape[1]):
                if torch.allclose(head(layers[layer][:, place]), scores, atol=1e-4):
                    return layer, place
    return None


def classify_picture(seen) -> str:
    """The learner's classify on the six watched sentences with the shipped numbers of BERT-Tiny and the course's
    head: the probability of positive for each, and which vector it read."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _CLASSIFY_TEXT[seen.lang]
    lang = seen.lang
    model, head = Shipped(), course_head()
    ids, mask = watched_batch()
    got = _timed(lambda: _quiet_many(lambda: fn(model, head, ids.clone(), mask.clone()), [()])[0])
    if got is _TOO_SLOW:
        return _figure(w["title"], [], [], [_said(w["slow"])])
    if got is _STOPPED:
        return _figure(w["title"], [], [], [_said(w["stop"])])
    if model.unmasked:
        return _figure(w["title"], [], [], [_said(w["no_mask"])])
    if not isinstance(got, torch.Tensor) or tuple(got.shape) != (6, 2) or not torch.isfinite(got).all():
        return _figure(w["title"], [], [], [_said(w["odd"], got=_short(got, 40))])
    with torch.no_grad():
        mine = torch.softmax(got.detach().float(), dim=1)[:, 1]
        want = torch.softmax(classify(model, head, ids, mask), dim=1)[:, 1]
    rows, said, right = [], [], 0
    for (text, label, _), p, q in zip(watched(), mine.tolist(), want.tolist()):
        right += (p >= 0.5) == (label == 1)
        label_html = (f'{html.escape(text)} <span style="color:var(--ea-ink-3)">({html.escape(LABELS[lang][label])})'
                      f'</span>')
        rows.append(_bar(label_html, p, "mine", lang, off=abs(p - q) > 0.005))
        said.append(f"{text} ({LABELS[lang][label]}): {_d(p, lang, 2)}")
    where = _which_place(got.detach().float(), model, head, ids, mask)
    if where == (2, 0):
        notes = [_said(w["right"])]
    elif where is None:
        notes = [_said(w["other"])]
    else:
        notes = [_said(w["read"], place=where[1], layer=w["layer"][where[0]])]
    notes += [_said(w["count"], right=right), _said(w["legend"])]
    return _figure(w["title"], rows, said, notes)


# --- the picture under exercise 2: your labels for the pieces of three test sentences ---------------------------------

LABEL_SENTENCES = (466, 383, 1036)   # three shown test sentences whose names are words of more than one piece

_LABELS_TEXT = {
    "en": {"title": "Your labels for the pieces of three test sentences",
           "fine": "Your labels are the course's at all {n} places of the three sentences: {kept} of them get a tag, "
                   "the other {left} get -100.",
           "differ": "At {n} of the {total} places, your label differs from the course's. The first is *{piece}*, "
                     "place {k} of the sentence *{sentence}*, where the course has {want}.",
           "length": "For *{sentence}*, your function gives {got} labels, and the sentence has {want} places, [CLS] "
                     "and [SEP] included.",
           "legend": "In each column, from top to bottom: a piece of the sentence (gray), and your label (blue), as a "
                     "tag, or -100 faint. Purple dashed: a label that differs from the course's.",
           "odd": "Your function gives back {got}, and not a list of whole numbers.",
           "stop": "Your function stops with an error on the sentence *{sentence}*.",
           "slow": "Your function takes too long on the three sentences."},
    "nl": {"title": "Jouw labels voor de stukjes van drie testzinnen",
           "fine": "Je labels zijn op alle {n} plekken van de drie zinnen gelijk aan de labels van de course: {kept} "
                   "ervan krijgen een tag, de andere {left} krijgen -100.",
           "differ": "Op {n} van de {total} plekken wijkt je label af van het label van de course. De eerste is "
                     "*{piece}*, plek {k} van de zin *{sentence}*, waar de course {want} heeft.",
           "length": "Voor *{sentence}* geeft je functie {got} labels, en de zin heeft {want} plekken, [CLS] en "
                     "[SEP] meegeteld.",
           "legend": "In elke kolom, van boven naar beneden: een stukje van de zin (grijs), en je label (blauw), als "
                     "tag, of vaag -100. Paars gestreept: een label dat afwijkt van het label van de course.",
           "odd": "Je functie geeft {got} terug, en geen lijst van hele getallen.",
           "stop": "Je functie stopt met een fout op de zin *{sentence}*.",
           "slow": "Je functie duurt te lang op de drie zinnen."},
}


def labels_picture(seen) -> str:
    """The learner's labels for every place of three test sentences, under their pieces."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _LABELS_TEXT[seen.lang]
    calls = [(word_pieces(k), gold_tags(k)) for k in LABEL_SENTENCES]
    got = _timed(lambda: _quiet_many(fn, calls))
    if got is _TOO_SLOW:
        return _figure(w["title"], [], [], [_said(w["slow"])])
    rows, said, end, first, wrong, total, kept_ = [], [], None, None, 0, 0, 0
    for k, mine in zip(LABEL_SENTENCES, got):
        s = test_sentences()[k]
        sentence = " ".join(s.words)
        if mine is _STOPPED:
            end = _said(w["stop"], sentence=sentence)
            break
        labels = _tag_list(mine)
        if labels is None:
            end = _said(w["odd"], got=_short(mine, 40))
            break
        want = piece_labels(word_pieces(k), gold_tags(k))
        pieces = ["[CLS]"] + [p for ps in s.pieces for p in ps] + ["[SEP]"]
        if len(labels) != len(want):
            end = _said(w["length"], sentence=sentence, got=len(labels), want=len(want))
            break
        cols = []
        for place, (piece, lab, ok) in enumerate(zip(pieces, labels, want)):
            off = lab != ok
            if off and first is None:
                first = (piece, place, sentence, _tag_name(ok))
            wrong += off
            total += 1
            kept_ += ok != IGNORE
            cols.append(_column(_chip(piece, "data"), _chip(_tag_name(lab), "mine", off=off, faint=lab == IGNORE)))
        rows += [_head(sentence), f"<div>{''.join(cols)}</div>"]
        said.append(sentence + ": " + " ".join(_tag_name(lab) for lab in labels))
    if end is not None:
        notes = [end]
    elif first is None:
        notes = [_said(w["fine"], n=total, kept=kept_, left=total - kept_)]
    else:
        notes = [_said(w["differ"], n=wrong, total=total, piece=first[0], k=first[1], sentence=first[2], want=first[3])]
    notes.append(_said(w["legend"]))
    return _figure(w["title"], rows, said, notes)


# --- the picture under exercise 4: your tags on the 1,471 test sentences ----------------------------------------------------

TAG_SENTENCES = (164, 1036, 383)   # the sentences the picture shows word by word: all right, a wrong kind, two misses

_TAGS_TEXT = {
    "en": {"title": "Your tags from the course's fine-tuned BERT-Tiny on the test sentences",
           "total": "With your word_tags, the course's fine-tuned BERT-Tiny finds {found} names in the {sentences} "
                    "test sentences, {right} of them right, of the {names} names there: a span F1 of {f1}.",
           "course": "The course: {found} names found, {right} right, a span F1 of {f1}.",
           "same": "Your tags are the course's in every sentence. Where a tag differs from the one in the test set, "
                   "BERT-Tiny is wrong, not your function.",
           "legend": "Each word in gray, with under it your tag (blue) and the tag in the test set (in the color of "
                     "the text). Purple dashed: a tag of yours that differs from the course's.",
           "odd": "For *{sentence}*, your function gives back {got}, and not one whole number per word.",
           "stop": "Your function stops with an error on the sentence *{sentence}*.",
           "slow": "Your function takes too long on the 1,471 test sentences. Does it go through the pieces more than "
                   "once?"},
    "nl": {"title": "Jouw tags uit de gefinetunede BERT-Tiny van de course op de testzinnen",
           "total": "Met je word_tags vindt de gefinetunede BERT-Tiny van de course {found} namen in de {sentences} "
                    "testzinnen, waarvan {right} goed, van de {names} namen die erin staan: een span F1 van {f1}.",
           "course": "De course: {found} namen gevonden, {right} goed, een span F1 van {f1}.",
           "same": "Je tags zijn in elke zin gelijk aan de tags van de course. Waar een tag afwijkt van de tag in de "
                   "testset, zit BERT-Tiny ernaast, en niet je functie.",
           "legend": "Elk woord in grijs, met eronder je tag (blauw) en de tag in de testset (in de kleur van de "
                     "tekst). Paars gestreept: een tag van jou die afwijkt van de tag van de course.",
           "odd": "Voor *{sentence}* geeft je functie {got} terug, en niet één heel getal per woord.",
           "stop": "Je functie stopt met een fout op de zin *{sentence}*.",
           "slow": "Je functie duurt te lang op de 1.471 testzinnen. Gaat je functie meer dan één keer door de "
                   "stukjes?"},
}


def tags_picture(seen) -> str:
    """The learner's word_tags on the shipped scores of every test sentence: the span F1 that gives, and three
    sentences word by word."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w, lang = _TAGS_TEXT[seen.lang], seen.lang
    n = len(test_sentences())
    calls = [(test_scores(k), word_pieces(k)) for k in range(n)]
    got = _timed(lambda: _quiet_many(fn, calls))
    if got is _TOO_SLOW:
        return _figure(w["title"], [], [], [_said(w["slow"])])
    tag_lists = []
    for k, mine in enumerate(got):
        sentence = " ".join(test_sentences()[k].words)
        if mine is _STOPPED:
            return _figure(w["title"], [], [], [_said(w["stop"], sentence=sentence), _said(w["legend"])])
        tags = _tag_list(mine)
        if tags is None or len(tags) != len(test_sentences()[k].words):
            return _figure(w["title"], [], [], [_said(w["odd"], sentence=sentence, got=_short(mine, 40))])
        tag_lists.append(tags)
    rows, said = [], []
    for k in TAG_SENTENCES:
        s = test_sentences()[k]
        want = course_tags()[k]
        cols = "".join(_column(_chip(word, "data"), _chip(_tag_name(t), "mine", off=t != c), _chip(TAGS[g], "course"))
                       for word, t, c, g in zip(s.words, tag_lists[k], want, s.gold))
        rows += [_head(" ".join(s.words)), f"<div>{cols}</div>"]
        said.append(" ".join(f"{word} {_tag_name(t)}" for word, t in zip(s.words, tag_lists[k])))
    counts, course = span_counts(tag_lists), course_counts()
    notes = [_said(w["total"], found=_n(counts[1], lang), sentences=_n(n, lang), right=_n(counts[0], lang),
                   names=_n(counts[2], lang), f1=_d(f1(counts), lang))]
    if tag_lists == list(course_tags()):
        notes.append(_said(w["same"]))
    else:
        notes.append(_said(w["course"], found=_n(course[1], lang), right=_n(course[0], lang), f1=_d(f1(course), lang)))
    notes.append(_said(w["legend"]))
    return _figure(w["title"], rows, said, notes)


# --- the demo's pictures: what the model makes now, and the rewind -----------------------------------------------------------

_NOW_TEXT = {
    "en": {"p": "The probability of positive for six held-out sentences",
           "cls": "The vectors at [CLS] after the last block: {state}",
           "same": "frozen, so the same at every step", "moving": "changing as BERT-Tiny learns",
           "after": "after step {s}",
           "legend_p": "Blue: the probability of positive, with the sentence's label in gray. Purple dashed: a "
                       "sentence on the wrong side of 0.5.",
           "legend_cls": "One row of cells per sentence, in the order of the bars: the 128 numbers of its vector at "
                         "[CLS], always in the same order, in magenta, the color of what the model works out inside. The darker a cell, the further its number is from 0."},
    "nl": {"p": "De kans op positief voor zes achtergehouden zinnen",
           "cls": "De vectoren bij [CLS] na het laatste block: {state}",
           "same": "bevroren, dus bij elke stap gelijk", "moving": "veranderen terwijl BERT-Tiny leert",
           "after": "na stap {s}",
           "legend_p": "Blauw: de kans op positief, met het label van de zin in grijs. Paars gestreept: een zin aan de "
                       "verkeerde kant van 0,5.",
           "legend_cls": "Eén rij vakjes per zin, in de volgorde van de balken: de 128 getallen van de vector bij "
                         "[CLS], steeds in dezelfde volgorde, in magenta, de kleur van wat het model binnenin uitrekent. Hoe donkerder een vakje, hoe verder het getal van 0 af "
                         "ligt."},
}


def _strips(vectors: np.ndarray) -> str:
    """The vectors at [CLS] of the six sentences as rows of 128 cells, one SVG: the further a number is from 0, the
    darker its cell, on the scale of the frozen model's vectors."""
    scale = cls_scale()
    cells = []
    for r, row in enumerate(vectors):
        for i, v in enumerate(row):
            o = min(1.0, abs(float(v)) / scale)
            if o >= 0.05:
                cells.append(f'<rect x="{i}" y="{r * 1.25:g}" width="1" height="1" opacity="{o:.2f}"/>')
    h = 1.25 * len(vectors) - 0.25
    return (f'<svg viewBox="0 0 128 {h:g}" width="100%" preserveAspectRatio="none" aria-hidden="true" '
            f'style="display:block;max-width:26rem;height:{12.5 * len(vectors) - 2.5:g}px">'
            f'<rect x="0" y="0" width="128" height="{h:g}" style="fill:var(--ea-page)"/>'
            f'<g style="fill:var(--ea-hidden)">{"".join(cells)}</g></svg>')


def now_parts(seen: dict, run: str, lang: str, step: int | None = None) -> list:
    """What the demo shows beside its chart: the probability of positive for the six watched sentences, and their
    vectors at [CLS] after the last block, now or after `step`, for the rewind."""
    w = _NOW_TEXT[lang]
    title = w["p"] + ("" if step is None else ", " + w["after"].format(s=step))
    rows, said = [], []
    for (text, label, _), p in zip(watched(), seen["p"]):
        label_html = (f'{html.escape(text)} <span style="color:var(--ea-ink-3)">({html.escape(LABELS[lang][label])})'
                      f'</span>')
        rows.append(_bar(label_html, p, "mine", lang, off=(p >= 0.5) != (label == 1)))
        said.append(f"{text} ({LABELS[lang][label]}): {_d(p, lang, 2)}")
    cls_title = w["cls"].format(state=w["same"] if run == "frozen" else w["moving"])
    first = yours.figure(title, "".join(rows) + f'<div style="{_UI}">{_marked(w["legend_p"])}</div>',
                         alt=f'{title}. {"; ".join(said)}. {_bare(w["legend_p"])}')
    second = yours.figure(cls_title, _strips(seen["cls"]) + f'<div style="{_UI}">{_marked(w["legend_cls"])}</div>',
                          alt=f'{cls_title}. {_bare(w["legend_cls"])}')
    return [first, second]


def curves_picture(result: dict, labels: dict, lang: str, at: int | None = None) -> str:
    """The loss of every step of the run with all of BERT-Tiny learning, as the live chart draws it, with its
    held-out loss dashed after every epoch, and that of the run with only the head learning thin and dashed in gray,
    named labels["frozen"]; with `at`, a line at that step, for the rewind."""
    runs = result["runs"]
    chart_ = live.Live(labels, STEPS, lang=lang, low=0.0)
    chart_.reference([(s, v) for s, _, v in runs["frozen"]["held"]], labels["frozen"], role="ink-2")
    chart_.reference([(s, v) for s, _, v in runs["tuned"]["held"]], "", role="loss", replace=True)
    chart_.xs = [s for s, _ in runs["tuned"]["losses"]]
    chart_.ys = [v for _, v in runs["tuned"]["losses"]]
    return chart_.picture(at=at)


def rewind_frame(result: dict, step: int, labels: dict, lang: str) -> str:
    """The rewind at a step the run kept: the loss with a line at that step, and the look then (decision 1.17)."""
    looks = dict(result["runs"]["tuned"]["looks"])
    return live.rewound(curves_picture(result, labels, lang, at=step), now_parts(looks[step], "tuned", lang, step))


def rewind_steps(result: dict) -> list:
    return [s for s, _ in result["runs"]["tuned"]["looks"]]


def held_at(result: dict, step: int, run: str = "tuned") -> int | None:
    """The held-out sentences the run labeled right at `step`, when it counted them then (before the first step and
    after every epoch), else None."""
    return {s: r for s, r, _ in result["runs"][run]["held"]}.get(step)


# --- the explore: your own sentence ----------------------------------------------------------------------------------------

_EXPLORE_TEXT = {
    "en": {"title": "What the two runs of the demo say of your sentence",
           "frozen": "Only the head trained", "tuned": "All of BERT-Tiny trained",
           "pieces": "The pieces BERT-Tiny sees: {pieces}",
           "legend": "Each bar: the probability of positive the model of that run of the demo gives, after 4 epochs, "
                     "where a full bar is 1."},
    "nl": {"title": "Wat de twee runs van de demo van je zin zeggen",
           "frozen": "Alleen de head getraind", "tuned": "Heel BERT-Tiny getraind",
           "pieces": "De stukjes die BERT-Tiny ziet: {pieces}",
           "legend": "Elke balk: de kans op positief die het model van de run van de demo geeft, na 4 epochs, "
                     "waarbij een volle balk 1 is."},
}
LONGEST = 300   # characters of the explore's sentence


def explore_picture(bert, result: dict, text: str, lang: str) -> str:
    """The probability of positive both runs of the demo give a sentence the learner types."""
    w = _EXPLORE_TEXT[lang]
    text = (text or "").strip()[:LONGEST] or "."
    runs = result["runs"]
    p_frozen = probability(bert, runs["frozen"]["model"], runs["frozen"]["head"], text)
    p_tuned = probability(bert, runs["tuned"]["model"], runs["tuned"]["head"], text)
    pieces = " ".join(bert.pieces(text))
    rows = [_bar(html.escape(w["frozen"]), p_frozen, "mine", lang), _bar(html.escape(w["tuned"]), p_tuned, "mine", lang)]
    said = [f'{w["frozen"]}: {_d(p_frozen, lang, 2)}', f'{w["tuned"]}: {_d(p_tuned, lang, 2)}']
    return _figure(w["title"], rows, said, [_said(w["pieces"], pieces=pieces), _said(w["legend"])])


# --- the exercises ---------------------------------------------------------------------------------------------------------

REQUIRED = ["classify", "piece_labels", "first-piece", "word_tags"]


class Named:
    """A value the check gives the learner's function, shown in a message by its name or its code, as the lesson
    writes it. The learner's function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _dots_left(got, text) -> bool:
    """A `...` of the start code left in the box: an Ellipsis in the code, not three dots inside a string."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None
    return any(isinstance(n, ast.Constant) and n.value is Ellipsis for n in ast.walk(tree))


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang de `...` door je eigen code.")


def _number(got):
    """The learner's outcome as one finite float, or None: a number, a tensor or an array with one value."""
    if isinstance(got, (bool, np.bool_, str, bytes)):
        return None
    try:
        if isinstance(got, torch.Tensor):
            if got.numel() != 1:
                return None
            got = got.detach().reshape(()).item()
        value = np.asarray(got, dtype=float)
    except (TypeError, ValueError, OverflowError, RuntimeError):
        return None
    if value.shape != () or not np.isfinite(value):
        return None
    return float(value)


def _rounded(v, depth: int = 0) -> str:
    """A value the way Python shows it, with its words in double quotes, also inside a dict, a tuple or a list; at most
    30 items and three levels deep. A tensor of more numbers as its rows, rounded to 4 decimals."""
    if depth > 3:
        return "…"
    if isinstance(v, np.bool_):
        return str(bool(v))
    if isinstance(v, torch.Tensor) and v.dtype == torch.bool and v.numel() == 1:
        return str(bool(v.item()))
    if isinstance(v, (torch.Tensor, np.ndarray, np.generic)) and not isinstance(v, np.str_):
        value = _number(v)
        if value is not None:
            if float(value).is_integer() and (isinstance(v, np.integer) or getattr(v, "dtype", None) is not None
                                              and str(v.dtype).startswith(("int", "torch.int"))):
                return str(int(value))
            v = value
        elif isinstance(v, torch.Tensor) and v.dtype.is_floating_point:   # its numbers, without grad_fn
            return "tensor(" + _rounded(v.detach().tolist(), depth) + ")"
        elif isinstance(v, np.ndarray) and v.dtype.kind == "f":
            return "array(" + _rounded(v.tolist(), depth) + ")"
        else:
            return " ".join(_repr(v).split())[:120]
    if isinstance(v, float):
        if not math.isfinite(v):
            return repr(v)
        text = f"{round(v, 4) + 0.0:.4f}".rstrip("0")
        return text + "0" if text.endswith(".") else text
    if type(v) is dict:
        items = list(v.items())
        inner = ", ".join(f"{_rounded(k, depth + 1)}: {_rounded(x, depth + 1)}" for k, x in items[:30])
        return "{" + inner + (", …" if len(items) > 30 else "") + "}"
    if type(v) is tuple:
        inner = ", ".join(_rounded(x, depth + 1) for x in v[:30]) + (", …" if len(v) > 30 else "")
        return "(" + inner + ("," if len(v) == 1 else "") + ")"
    if type(v) is list:
        return "[" + ", ".join(_rounded(x, depth + 1) for x in v[:30]) + (", …" if len(v) > 30 else "") + "]"
    if isinstance(v, str) and v.isprintable() and '"' not in v and "\\" not in v:
        return f'"{v}"'   # double quotes, as in the exercise texts
    return _repr(v)


def _is_number(v) -> bool:
    """A number, also a NumPy number or a tensor or array of one number, but not True or False."""
    if isinstance(v, (bool, np.bool_)):
        return False
    if isinstance(v, (int, float, np.integer, np.floating)):
        return True
    if isinstance(v, (torch.Tensor, np.ndarray)) and "bool" not in str(v.dtype):
        return _number(v) is not None
    return False


def show_value(v) -> str:
    """How {got} and {want} show a value: as code, cut after 160 characters."""
    text = _rounded(v)
    if len(text) > 160:
        text = text[:159] + "…"
    if _is_number(v) or "`" in text:
        return text
    return f"`{text}`"


# exercise 1: classify. The check uses a small stand-in for BERT-Tiny: 3 numbers per place in place of 128, and three
# stages as BERT-Tiny gives them, where every place mixes in the mean of the real places of its sentence, so that the
# mask matters. A head of its own from 3 numbers to 2 scores.

class Toy(nn.Module):
    """A stand-in for BERT-Tiny with 3 numbers per place: `forward(ids, mask)` gives a list of 3 tensors of (n, t, 3),
    before the first block and after each of 2 blocks. Without a mask every place counts as real, as in BERT-Tiny."""

    def __init__(self):
        super().__init__()
        self.table = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0], [0.0, 1.0, -1.0], [2.0, -1.0, 0.0],
                                   [-1.0, 2.0, 1.0], [0.5, 0.5, 0.5], [1.0, 1.0, -2.0], [-2.0, 0.0, 1.0]])

    def forward(self, ids, mask=None, segments=None):
        if mask is None:
            mask = torch.ones_like(ids, dtype=torch.bool)
        x = self.table[ids]
        out = [x]
        for k in (1, 2):
            real = mask.unsqueeze(-1).float()
            mean = (x * real).sum(1, keepdim=True) / real.sum(1, keepdim=True)
            x = x + k * mean + 0.25 * torch.arange(ids.shape[1]).float()[None, :, None]
            out.append(x)
        return out

    def __repr__(self) -> str:
        return "model"


class ToyHead(nn.Linear):
    def __init__(self):
        super().__init__(3, 2)
        with torch.no_grad():
            self.weight.copy_(torch.tensor([[1.0, -1.0, 0.5], [-0.5, 1.0, 1.0]]))
            self.bias.copy_(torch.tensor([0.1, -0.2]))

    def __repr__(self) -> str:
        return "head"


_IDS_TWO = Named("ids", torch.tensor([[1, 4, 2, 5], [1, 3, 0, 0]]))
_MASK_TWO = Named("mask", torch.tensor([[True, True, True, True], [True, True, False, False]]))
_IDS_ONE = Named("ids", torch.tensor([[1, 6, 7, 3, 2]]))
_MASK_ONE = Named("mask", torch.tensor([[True, True, True, True, True]]))
_IDS_THREE = Named("ids", torch.tensor([[1, 2, 0, 0, 0], [1, 7, 4, 4, 0], [1, 5, 6, 3, 2]]))
_MASK_THREE = Named("mask", torch.tensor([[True, True, False, False, False], [True, True, True, True, False],
                                          [True, True, True, True, True]]))
CL_CASES = [
    (Toy(), ToyHead(), _IDS_TWO, _MASK_TWO),
    (Toy(), ToyHead(), _IDS_ONE, _MASK_ONE),
    (Toy(), ToyHead(), _IDS_THREE, _MASK_THREE),
]
_CL_HEAD = "def classify(model, head, ids, mask):\n"
_CL_START = _CL_HEAD + "    ...\n"
_CL = _CL_HEAD + "    return head(model(ids, mask)[-1][:, 0])\n"
_CL_FIRST = _CL.replace("[-1]", "[0]")
_CL_MIDDLE = _CL.replace("[-1]", "[1]")
_CL_LAST_PLACE = _CL.replace("[:, 0]", "[:, -1]")
_CL_MEAN = _CL.replace("[-1][:, 0]", "[-1].mean(dim=1)")
_CL_NO_MASK = _CL.replace("model(ids, mask)", "model(ids)")
_CL_EVERY = _CL.replace("[-1][:, 0]", "[-1]")
_CL_ROW = _CL.replace("[-1][:, 0]", "[-1][0]")
_CL_VECTOR = _CL_HEAD + "    return model(ids, mask)[-1][:, 0]\n"


def _same_scores(got, want) -> bool:
    """The same scores, and like the course's still tied to the head's parameters: the demo trains through them."""
    return (isinstance(got, torch.Tensor) and got.shape == want.shape and got.dtype.is_floating_point
            and got.requires_grad == want.requires_grad
            and torch.allclose(got.detach().float(), want.detach().float(), atol=1e-4))


def _no_gradient(got, text) -> bool:
    return isinstance(got, torch.Tensor) and got.dim() == 2 and got.shape[-1] == 2 and not got.requires_grad


def _shape_of(got) -> tuple | None:
    return tuple(got.shape) if isinstance(got, torch.Tensor) else None


def _every_place(got, text) -> bool:
    return isinstance(got, torch.Tensor) and got.dim() == 3


def _not_scores(got, text) -> bool:
    return got is not None and not isinstance(got, torch.Tensor)


# exercise 2: piece_labels. The visible cases: Ada Lovelace and Marie Curie; then a word of three pieces, a sentence of
# one word, a sentence of single pieces with an organization, and a test sentence of Universal NER.
PL_CASES = [
    ([["ada"], ["love", "##lace"], ["lived"], ["in"], ["london"], ["."]], [1, 2, 0, 0, 3, 0]),
    ([["marie"], ["cu", "##rie"], ["moved"], ["to"], ["paris"], ["."]], [1, 2, 0, 0, 3, 0]),
    ([["the"], ["travel", "##oc", "##ity"], ["trip"]], [0, 5, 0]),
    ([["hello"]], [0]),
    ([["they"], ["joined"], ["google"], ["in"], ["2004"]], [0, 0, 5, 0, 0]),
    (Named("C.word_pieces(913)", word_pieces(913)), Named("C.gold_tags(913)", gold_tags(913))),
]
_PL_HEAD = "def piece_labels(word_pieces, tags):\n"
_PL_START = _PL_HEAD + "    labels = [-100]\n    ...\n    return labels\n"
_PL = (_PL_HEAD + "    labels = [-100]\n    for i in range(len(word_pieces)):\n        labels.append(tags[i])\n"
       "        for piece in word_pieces[i][1:]:\n            labels.append(-100)\n    labels.append(-100)\n"
       "    return labels\n")
_PL_EVERY = _PL.replace("        for piece in word_pieces[i][1:]:\n            labels.append(-100)\n",
                        "        for piece in word_pieces[i][1:]:\n            labels.append(tags[i])\n")
_PL_NO_SEP = _PL.replace("    labels.append(-100)\n    return labels\n", "    return labels\n")
_PL_NO_CLS = _PL.replace("    labels = [-100]\n", "    labels = []\n")
_PL_NO_ENDS = _PL_NO_CLS.replace("    labels.append(-100)\n    return labels\n", "    return labels\n")
_PL_LAST = (_PL_HEAD + "    labels = [-100]\n    for i in range(len(word_pieces)):\n"
            "        for piece in word_pieces[i][1:]:\n            labels.append(-100)\n        labels.append(tags[i])\n"
            "    labels.append(-100)\n    return labels\n")
_PL_ZERO = _PL.replace("            labels.append(-100)\n", "            labels.append(0)\n")
_PL_PER_WORD = _PL_HEAD + "    return list(tags)\n"


def _ints(got) -> bool:
    return _tag_list(got) is not None


def _same_ints(got, want) -> bool:
    mine = _tag_list(got)
    return mine is not None and mine == list(want)


def _not_ints(got, text) -> bool:
    return got is not None and _tag_list(got) is None


# exercise 4: word_tags. The visible cases: Ada Lovelace and Marie Curie with scores of the course's own, then a word of
# three pieces, a sentence of one word, a tie at a first piece, and two test sentences with the scores of the course's
# fine-tuned BERT-Tiny.

def _scores(rows: list) -> np.ndarray:
    """Scores of the course's own: at each place a list of 7, with the given tag 2.0 higher than the others."""
    out = np.zeros((len(rows), 7), dtype=np.float32)
    for k, t in enumerate(rows):
        out[k, t] = 2.0
    return out


_ADA = _scores([0, 1, 2, 3, 0, 0, 3, 0, 0])   # [CLS] ada love ##lace lived in london . [SEP]
_CURIE = _scores([0, 1, 4, 2, 0, 0, 3, 0, 0])   # [CLS] marie cu ##rie moved to paris . [SEP]
_TRAVEL = _scores([0, 0, 5, 6, 6, 0, 0])        # [CLS] the travel ##oc ##ity trip [SEP]
_TIE = np.zeros((3, 7), dtype=np.float32)
_TIE[1, 1] = _TIE[1, 3] = 1.0
WT_CASES = [
    (Named("ada_scores", _ADA), [["ada"], ["love", "##lace"], ["lived"], ["in"], ["london"], ["."]]),
    (Named("curie_scores", _CURIE), [["marie"], ["cu", "##rie"], ["moved"], ["to"], ["paris"], ["."]]),
    (Named("travel_scores", _TRAVEL), [["the"], ["travel", "##oc", "##ity"], ["trip"]]),
    (Named("tie_scores", _TIE), [["hello"]]),
    (Named("C.test_scores(379)", test_scores(379)), Named("C.word_pieces(379)", word_pieces(379))),
    (Named("C.test_scores(913)", test_scores(913)), Named("C.word_pieces(913)", word_pieces(913))),
]
_WT_HEAD = "import numpy as np\n\n\ndef word_tags(scores, word_pieces):\n"
_WT_START = _WT_HEAD + "    tags = []\n    place = 1\n    ...\n    return tags\n"
_WT = (_WT_HEAD + "    tags = []\n    place = 1\n    for pieces in word_pieces:\n"
       "        tags.append(int(np.argmax(scores[place])))\n        place = place + len(pieces)\n    return tags\n")
_WT_CLS = _WT.replace("    place = 1\n", "    place = 0\n")
_WT_EVERY = (_WT_HEAD + "    tags = []\n    for place in range(1, len(scores) - 1):\n"
             "        tags.append(int(np.argmax(scores[place])))\n    return tags\n")
_WT_LAST = _WT.replace("        tags.append(int(np.argmax(scores[place])))\n        place = place + len(pieces)\n",
                       "        place = place + len(pieces)\n        tags.append(int(np.argmax(scores[place - 1])))\n")
_WT_NO_MOVE = _WT.replace("        place = place + len(pieces)\n", "        place = place + 1\n")
_WT_NAMES = _WT.replace("tags.append(int(np.argmax(scores[place])))", "tags.append(TAGS[int(np.argmax(scores[place]))])"
                        ).replace("    tags = []\n", '    TAGS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]\n    tags = []\n')
_WT_MAX = _WT.replace("int(np.argmax(scores[place]))", "float(np.max(scores[place]))")
_WT_LAST_TIE = (_WT_HEAD + "    tags = []\n    place = 1\n    for pieces in word_pieces:\n        row = scores[place]\n"
                "        best = 0\n        for k in range(len(row)):\n            if row[k] >= row[best]:\n                best = k\n"
                "        tags.append(best)\n        place = place + len(pieces)\n    return tags\n")


def _tag_names(got, text) -> bool:
    return isinstance(got, list) and bool(got) and all(isinstance(g, str) for g in got)


def _same_tags(got, want) -> bool:
    """The same whole numbers as the course's tags; for a wrong version whose outcome is not tags, such as the
    highest scores, the same numbers one by one."""
    if _tag_list(want) is not None:
        return _same_ints(got, want)
    return (isinstance(got, list) and isinstance(want, list) and len(got) == len(want)
            and all(_is_number(g) and _is_number(v) and float(g) == float(v) for g, v in zip(got, want)))


EXERCISES = {
    "classify": FunctionBox(
        name="classify",
        picture=classify_picture,
        function="classify",
        start=_CL_START,
        solution=_CL,
        visible=CL_CASES[:2],
        hidden=CL_CASES[2:],
        compare=_same_scores,
        show=show_value,
        first=[_DOTS],
        mistakes=[
            Mistake("no-gradient", matches=_no_gradient,
                    en="`{call}` gives {got}: the right numbers, but cut off from the head, so no gradient flows back through them and the demo could not train with them. Which part of your function stops the gradient?",
                    nl="`{call}` geeft {got}: de goede getallen, maar losgemaakt van de head, dus er loopt geen gradient door terug en de demo zou er niet mee kunnen trainen. Welk deel van je functie houdt de gradient tegen?"),
            Mistake("every-place", matches=_every_place,
                    en="`{call}` gives 2 scores at every place of every sentence, and it should give {want}: 2 scores per sentence. The vector of which place does the head read?",
                    nl="`{call}` geeft 2 scores op elke plek van elke zin, en dat moet {want} zijn: 2 scores per zin. De vector van welke plek leest de head?"),
            Mistake("one-sentence", wrong=_CL_ROW,
                    en="`{call}` gives {got}, and it should give {want}: a row of scores for every place of the first sentence, and not one row per sentence. Which of the three positions in the square brackets of a tensor of shape (n, t, 3) is the sentence, and which the place?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een rij scores voor elke plek van de eerste zin, en niet één rij per zin. Welke van de drie posities tussen de vierkante haken van een tensor met de vorm (n, t, 3) is de zin, en welke de plek?"),
            Mistake("first-layer", wrong=_CL_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: your function reads the numbers before the first block. Which of the 3 tensors comes after the last block?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie leest de getallen vóór het eerste block. Welke van de 3 tensors komt na het laatste block?"),
            Mistake("middle-layer", wrong=_CL_MIDDLE,
                    en="`{call}` gives {got}, and it should give {want}: your function reads the numbers after the first block. Which of the 3 tensors comes after the last block?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie leest de getallen na het eerste block. Welke van de 3 tensors komt na het laatste block?"),
            Mistake("last-place", wrong=_CL_LAST_PLACE,
                    en="`{call}` gives {got}, and it should give {want}: your function reads the last place of each sentence, a [PAD] in a short one. At which place stands [CLS]?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie leest de laatste plek van elke zin, een [PAD] in een korte zin. Op welke plek staat [CLS]?"),
            Mistake("mean", wrong=_CL_MEAN,
                    en="`{call}` gives {got}, and it should give {want}: your function averages the vectors of all places, the [PAD]s included. Jurafsky and Martin name the mean as one way, but which vector does this exercise give the head?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie neemt het gemiddelde van de vectoren van alle plekken, met de [PAD]s erbij. Jurafsky en Martin noemen het gemiddelde als één manier, maar welke vector geeft deze oefening aan de head?"),
            Mistake("no-mask", wrong=_CL_NO_MASK,
                    en="`{call}` gives {got}, and it should give {want}: for the shorter sentence, the model counts the [PAD]s as real pieces. What does the model need to know which places are real?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: bij de kortere zin telt het model de [PAD]s mee als echte stukjes. Wat heeft het model nodig om te weten welke plekken echt zijn?"),
            Mistake("vector", wrong=_CL_VECTOR,
                    en="`{call}` gives {got}, and it should give {want}: the vectors at [CLS], 3 numbers each, without the head. What turns a vector into 2 scores?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de vectoren bij [CLS], elk 3 getallen, zonder de head. Wat maakt van een vector 2 scores?"),
            Mistake("not-scores", matches=_not_scores,
                    en="`{call}` gives {got}, and it should give {want}: a tensor with 2 scores per sentence. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een tensor met 2 scores per zin. Wat geeft je `return`-regel terug?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back the 2 scores the head gives for the vector at [CLS] of every sentence, after the last block.",
                      "nl": "Vervang de `...`: geef de 2 scores terug die de head geeft voor de vector bij [CLS] van elke zin, na het laatste block."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. From which of the 3 tensors and which place come the vectors the head turns into scores?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Uit welke van de 3 tensors en van welke plek komen de vectoren waar de head scores van maakt?"},
            "almost": {"en": "Right for the first two cases. But `{call}` gives {got}, and it should give {want}. Does your function also work for three sentences of different lengths?",
                       "nl": "Goed voor de eerste twee gevallen. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie ook voor drie zinnen van verschillende lengte?"},
            "unreached": {"en": "With `{call}`, your function gives `None`. Does it end with a `return`?",
                          "nl": "Met `{call}` geeft je functie `None`. Eindigt je functie met een `return`?"},
            "correct": {"en": "Your `classify` is right in all {n} cases. The picture runs it on BERT-Tiny's numbers for six held-out sentences, and the demo below fine-tunes BERT-Tiny with it.",
                        "nl": "Je `classify` klopt in alle {n} gevallen. De afbeelding gebruikt je functie op de getallen van BERT-Tiny voor zes achtergehouden zinnen, en de demo hieronder finetunet BERT-Tiny ermee."},
        },
    ),
    "piece_labels": FunctionBox(
        name="piece_labels",
        picture=labels_picture,
        function="piece_labels",
        start=_PL_START,
        solution=_PL,
        visible=PL_CASES[:2],
        hidden=PL_CASES[2:],
        compare=_same_ints,
        show=show_value,
        first=[_DOTS],
        mistakes=[
            Mistake("every-piece", wrong=_PL_EVERY,
                    en="`{call}` gives {got}, and it should give {want}: your function gives the tag of a word to every piece of it. Jurafsky and Martin describe that way too, but which label does this exercise ask for at a piece after the first?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft de tag van een woord aan elk stukje ervan. Jurafsky en Martin beschrijven deze manier ook, maar welk label vraagt deze oefening bij een stukje na het eerste?"),
            Mistake("no-ends", wrong=_PL_NO_ENDS,
                    en="`{call}` gives {got}, and it should give {want}: a label for every piece of the words, but none for [CLS] and [SEP]. Which two places does BERT-Tiny read besides the pieces?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een label voor elk stukje van de woorden, maar geen voor [CLS] en [SEP]. Welke twee plekken leest BERT-Tiny naast de stukjes?"),
            Mistake("no-sep", wrong=_PL_NO_SEP,
                    en="`{call}` gives {got}, and it should give {want}: one label short, at the end. Which piece comes after the last word?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: één label te weinig, aan het eind. Welk stukje komt na het laatste woord?"),
            Mistake("no-cls", wrong=_PL_NO_CLS,
                    en="`{call}` gives {got}, and it should give {want}: one label short, at the start. Which piece comes before the first word?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: één label te weinig, aan het begin. Welk stukje komt vóór het eerste woord?"),
            Mistake("last-piece", wrong=_PL_LAST,
                    en="`{call}` gives {got}, and it should give {want}: your function puts the tag at the last piece of a word. Which piece of a word carries its tag?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie zet de tag bij het laatste stukje van een woord. Welk stukje van een woord draagt de tag?"),
            Mistake("zero", wrong=_PL_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: a later piece of a word gets 0, the number of the tag O, so the loss would teach the model that it is outside every name. Which label does the loss leave out?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een later stukje van een woord krijgt 0, het nummer van de tag O, dus de loss zou het model leren dat het buiten elke naam valt. Welk label slaat de loss over?"),
            Mistake("per-word", wrong=_PL_PER_WORD,
                    en="`{call}` gives {got}, and it should give {want}: one label per word. BERT-Tiny gives a vector per place: how many places does the sentence have?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: één label per woord. BERT-Tiny geeft een vector per plek: hoeveel plekken heeft de zin?"),
            Mistake("not-ints", matches=_not_ints,
                    en="`{call}` gives {got}, and it should give {want}: a list of whole numbers, one per place. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een lijst met hele getallen, één per plek. Wat geeft je `return`-regel terug?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: add a label to `labels` for every piece of every word, and -100 for [SEP] at the end.",
                      "nl": "Vervang de `...`: voeg aan `labels` een label toe voor elk stukje van elk woord, en -100 voor [SEP] aan het eind."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which places get a tag, and which get -100?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Welke plekken krijgen een tag, en welke -100?"},
            "almost": {"en": "Right for the first two cases. But `{call}` gives {got}, and it should give {want}. Does your function also work for a word of three pieces, and for a sentence of one word?",
                       "nl": "Goed voor de eerste twee gevallen. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie ook voor een woord van drie stukjes, en voor een zin van één woord?"},
            "unreached": {"en": "With `{call}`, your function gives `None`. Does it end with a `return`?",
                          "nl": "Met `{call}` geeft je functie `None`. Eindigt je functie met een `return`?"},
            "correct": {"en": "Your `piece_labels` is right in all {n} cases, also for a test sentence of Universal NER. Look at the picture: the labels of three test sentences, place by place.",
                        "nl": "Je `piece_labels` klopt in alle {n} gevallen, ook voor een testzin van Universal NER. Kijk naar de afbeelding: de labels van drie testzinnen, plek voor plek."},
        },
    ),
    "word_tags": FunctionBox(
        name="word_tags",
        picture=tags_picture,
        function="word_tags",
        start=_WT_START,
        solution=_WT,
        visible=WT_CASES[:2],
        hidden=WT_CASES[2:],
        compare=_same_tags,
        show=show_value,
        first=[_DOTS],
        mistakes=[
            Mistake("cls", wrong=_WT_CLS,
                    en="`{call}` gives {got}, and it should give {want}: the first tag comes from place 0, [CLS], and every word after it reads a place too early. Where is the first piece of the first word?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de eerste tag komt van plek 0, [CLS], en elk woord daarna leest een plek te vroeg. Waar staat het eerste stukje van het eerste woord?"),
            Mistake("every-place", wrong=_WT_EVERY,
                    en="`{call}` gives {got}, and it should give {want}: a tag for every piece. How many words does the sentence have, and how many tags does your function give?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een tag voor elk stukje. Hoeveel woorden heeft de zin, en hoeveel tags geeft je functie?"),
            Mistake("last-piece", wrong=_WT_LAST,
                    en="`{call}` gives {got}, and it should give {want}: your function reads the last piece of each word. Which piece of a word carries its tag?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie leest het laatste stukje van elk woord. Welk stukje van een woord draagt de tag?"),
            Mistake("one-step", wrong=_WT_NO_MOVE,
                    en="`{call}` gives {got}, and it should give {want}: after a word of more pieces, your function reads the next piece of that word. How many places does a word take?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: na een woord van meer stukjes leest je functie het volgende stukje van dat woord. Hoeveel plekken neemt een woord in?"),
            Mistake("names", matches=_tag_names,
                    en="`{call}` gives {got}, and it should give {want}: the names of the tags, and not their numbers. What does `np.argmax` give?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de namen van de tags, en niet de nummers ervan. Wat geeft `np.argmax`?"),
            Mistake("highest-score", wrong=_WT_MAX,
                    en="`{call}` gives {got}, and it should give {want}: the highest score at each first piece, and not the number of its tag. Which NumPy function gives the place of the highest number?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de hoogste score bij elk eerste stukje, en niet het nummer van de tag. Welke NumPy-functie geeft de plek van het hoogste getal?"),
            Mistake("last-tie", wrong=_WT_LAST_TIE,
                    en="`{call}` gives {got}, and it should give {want}: where two tags share the highest score at a first piece, your function keeps the higher number. Which of the two numbers does the exercise ask for?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: waar twee tags bij een eerste stukje dezelfde hoogste score hebben, houdt je functie het hoogste nummer. Welk van de twee nummers vraagt de oefening?"),
            Mistake("not-ints", matches=_not_ints,
                    en="`{call}` gives {got}, and it should give {want}: a list of whole numbers, one per word. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een lijst met hele getallen, één per woord. Wat geeft je `return`-regel terug?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: for every word, add to `tags` the number of the tag with the highest score at its first piece.",
                      "nl": "Vervang de `...`: voeg voor elk woord aan `tags` het nummer toe van de tag met de hoogste score bij het eerste stukje."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. At which place is the first piece of each word?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Op welke plek staat het eerste stukje van elk woord?"},
            "almost": {"en": "Right for the first two cases. But `{call}` gives {got}, and it should give {want}. Does your function also work for a word of three pieces, and for a sentence of one word?",
                       "nl": "Goed voor de eerste twee gevallen. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie ook voor een woord van drie stukjes, en voor een zin van één woord?"},
            "unreached": {"en": "With `{call}`, your function gives `None`. Does it end with a `return`?",
                          "nl": "Met `{call}` geeft je functie `None`. Eindigt je functie met een `return`?"},
            "correct": {"en": "Your `word_tags` is right in all {n} cases, also for the scores of two test sentences. Look at the picture: your function read the tags of all 1,471 test sentences.",
                        "nl": "Je `word_tags` klopt in alle {n} gevallen, ook voor de scores van twee testzinnen. Kijk naar de afbeelding: je functie las de tags van alle 1.471 testzinnen."},
        },
    ),
}

CASES = [
    ("classify", _CL_START, "empty", None),
    ("classify", _CL, "correct", None),
    ("classify", _CL_HEAD + "    layers = model(ids, mask)\n    return head(layers[2][:, 0])\n", "correct", None),
    ("classify", _CL_HEAD + "    last = model(ids, mask)[-1]\n    cls = last[:, 0, :]\n    return head(cls)\n", "correct", None),
    ("classify", _CL_FIRST, "wrong", "first-layer"),
    ("classify", _CL_MIDDLE, "wrong", "middle-layer"),
    ("classify", _CL_LAST_PLACE, "wrong", "last-place"),
    ("classify", _CL_MEAN, "wrong", "mean"),
    ("classify", _CL_NO_MASK, "wrong", "no-mask"),
    ("classify", _CL_EVERY, "wrong", "every-place"),
    ("classify", _CL_ROW, "wrong", "one-sentence"),
    ("classify", _CL_VECTOR, "wrong", "vector"),
    ("classify", _CL_HEAD + "    return [0.0, 0.0]\n", "wrong", "not-scores"),
    ("classify", _CL.replace("[:, 0])", "[:, 0]).detach()"), "wrong", "no-gradient"),
    ("classify", "import torch\n\n\n" + _CL_HEAD + "    with torch.no_grad():\n        return head(model(ids, mask)[-1][:, 0])\n", "wrong", "no-gradient"),
    ("classify", _CL_HEAD + "    if ids.shape[0] == 3:\n        return None\n    return head(model(ids, mask)[-1][:, 0])\n", "wrong", "unreached"),
    ("classify", _CL_HEAD + "    if ids.shape[0] == 3:\n        return head(model(ids)[-1][:, 0])\n    return head(model(ids, mask)[-1][:, 0])\n", "almost", None),
    ("piece_labels", _PL_START, "empty", None),
    ("piece_labels", _PL, "correct", None),
    ("piece_labels", _PL_HEAD + "    labels = [-100]\n    for i in range(len(word_pieces)):\n        labels.append(tags[i])\n"
     "        labels = labels + [-100] * (len(word_pieces[i]) - 1)\n    return labels + [-100]\n", "correct", None),
    ("piece_labels", _PL_HEAD + "    labels = [-100]\n    for i in range(len(word_pieces)):\n        for j in range(len(word_pieces[i])):\n"
     "            if j == 0:\n                labels.append(tags[i])\n            else:\n                labels.append(-100)\n"
     "    labels.append(-100)\n    return labels\n", "correct", None),
    ("piece_labels", _PL_EVERY, "wrong", "every-piece"),
    ("piece_labels", _PL_NO_ENDS, "wrong", "no-ends"),
    ("piece_labels", _PL_NO_SEP, "wrong", "no-sep"),
    ("piece_labels", _PL_NO_CLS, "wrong", "no-cls"),
    ("piece_labels", _PL_LAST, "wrong", "last-piece"),
    ("piece_labels", _PL_ZERO, "wrong", "zero"),
    ("piece_labels", _PL_PER_WORD, "wrong", "per-word"),
    ("piece_labels", _PL_HEAD + "    return str(tags)\n", "wrong", "not-ints"),
    ("piece_labels", _PL.replace("    labels.append(-100)\n    return labels\n", "    labels.append(-100)\n    if len(tags) == 3:\n        return None\n    return labels\n"), "wrong", "unreached"),
    ("piece_labels", _PL.replace("        for piece in word_pieces[i][1:]:\n", "        for piece in word_pieces[i][1:2]:\n"), "almost", None),
    ("word_tags", _WT_START, "empty", None),
    ("word_tags", _WT, "correct", None),
    ("word_tags", _WT.replace("int(np.argmax(scores[place]))", "np.argmax(scores[place])"), "correct", None),
    ("word_tags", _WT_HEAD + "    tags = []\n    place = 1\n    for i in range(len(word_pieces)):\n        row = list(scores[place])\n"
     "        tags.append(row.index(max(row)))\n        place += len(word_pieces[i])\n    return tags\n", "correct", None),
    ("word_tags", _WT_CLS, "wrong", "cls"),
    ("word_tags", _WT_EVERY, "wrong", "every-place"),
    ("word_tags", _WT_LAST, "wrong", "last-piece"),
    ("word_tags", _WT_NO_MOVE, "wrong", "one-step"),
    ("word_tags", _WT_NAMES, "wrong", "names"),
    ("word_tags", _WT_MAX, "wrong", "highest-score"),
    ("word_tags", _WT_MAX.replace("float(np.max(scores[place]))", "np.max(scores[place])"), "wrong", "highest-score"),
    ("word_tags", _WT_LAST_TIE, "wrong", "last-tie"),
    ("word_tags", _WT.replace("int(np.argmax(scores[place]))", "float(np.argmax(scores[place]))"), "wrong", "not-ints"),
    ("word_tags", _WT_HEAD + "    return 'O'\n", "wrong", "not-ints"),
    ("word_tags", _WT.replace("    return tags\n", "    if len(word_pieces) == 1:\n        return None\n    return tags\n"), "wrong", "unreached"),
    ("word_tags", _WT.replace("place = place + len(pieces)", "place = place + min(len(pieces), 2)"), "almost", None),
]
