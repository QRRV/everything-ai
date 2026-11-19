# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-data-poisoning: the 6,000 training digits of QMNIST with some of their labels flipped,
at random or only for one digit, and the DigitsMLP(64) of torch-mlp-digits trained on them. The three functions the
learner writes: flip_digit (the first share of one digit's labels become another digit, with NumPy), per_digit (the
share of the test digits of each kind that a network reads right) and verdict (whether a drop is larger than the
margin of error of a score on n test digits). The demo trains the network on labels the learner flips, at random or only the
7s, and shows while it trains what the network reads (decision 1.17)."""
import copy
import functools
import html
import math
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from everything_ai import chart, live, netview, paths, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, _show_rounded, with_time_limit

DATA = "data/qmnist-small.npz"    # the first 6,000 training and 1,000 test images of QMNIST: see data/README.md
HIDDEN, LR, BATCH, EPOCHS = 64, 0.5, 64, 10   # DigitsMLP(64), as in the demo of torch-mlp-digits
EVERY = 10                        # the demo measures the network on the test digits every 10 mini-batches
SEED = 1                          # the seed of flip_random in the worked example and the demo
RANDOM_SHARE = 0.3                # the worked example flips 30% of the labels at random
SOURCE, TARGET, SHARE = 7, 1, 0.8   # exercise 1: the first 80% of the 7s become 1s
SWEEP = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]   # the shares the text reports for flips at random
torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)   # the first optimizer in a process takes a second: not in a check


class Named:
    """A value the check calls by the name the lesson gives it, such as LABELS, so that a message stays short. The
    learner's code gets its own copy."""

    def __init__(self, make, name: str):
        self.make, self.name, self._value = make, name, None

    @property
    def value(self):
        if self._value is None:
            self._value = self.make()
        return self._value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.value, dtype=dtype)


# --- the digits ------------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _data() -> dict:
    d = np.load(paths.ROOT / DATA)
    return {k: d[k] for k in ("train_images", "train_labels", "test_images", "test_labels")}


@functools.lru_cache(maxsize=1)
def tensors():
    """The training images and the test images and labels as tensors; every pixel divided by 255."""
    d = _data()
    return (torch.tensor(d["train_images"], dtype=torch.float32) / 255,
            torch.tensor(d["test_images"], dtype=torch.float32) / 255, torch.tensor(d["test_labels"], dtype=torch.int64))


LABELS = Named(lambda: _data()["train_labels"].astype(np.int64), "LABELS")             # the 6,000 training labels
TEST_LABELS = Named(lambda: _data()["test_labels"].astype(np.int64), "TEST_LABELS")    # the 1,000 test labels
# exercise 2: twenty made-up guesses for twenty test images, two of each digit; the network misreads one 7 and one 9
GUESSES = Named(lambda: np.array([0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 4]), "GUESSES")
TRUTH = Named(lambda: np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), "TRUTH")
# exercise 1, in the check only: five 2s, so that 0.7 of them is 3.5 and round and int differ
SMALL = Named(lambda: np.array([2, 0, 2, 2, 1, 2, 9, 2, 5]), "SMALL")


# --- the course's own solutions --------------------------------------------------------------------------------------

def flip_random(labels, share, seed):
    """The worked example's: a copy of labels in which each label, with chance share, becomes one of the nine other
    digits, each as likely."""
    rng = np.random.default_rng(seed)
    flipped = labels.copy()
    picked = rng.random(len(labels)) < share
    shifts = rng.integers(1, 10, size=len(labels))
    flipped[picked] = (labels[picked] + shifts[picked]) % 10
    return flipped


def flip_digit(labels, source, target, share):
    flipped = labels.copy()
    places = np.arange(len(labels))[labels == source]
    count = round(share * len(places))
    flipped[places[:count]] = target
    return flipped


def per_digit(pred, labels):
    shares = []
    for digit in range(10):
        mine = labels == digit
        shares.append(float(np.mean(pred[mine] == digit)))
    return shares


def margin(score, n):
    """Two standard errors of a share measured on n test digits: the margin of error QMNIST's paper reports with a score."""
    return 2 * math.sqrt(score * (1 - score) / n)


def verdict(clean, poisoned, n):
    margin = 2 * math.sqrt(clean * (1 - clean) / n)
    if clean - poisoned > margin:
        return "worse"
    return "chance"


# --- the network and its training ------------------------------------------------------------------------------------

class DigitsMLP(nn.Module):
    """The MLP of torch-mlp-digits."""
    def __init__(self, hidden):
        super().__init__()
        self.hidden = nn.Linear(784, hidden)
        self.out = nn.Linear(hidden, 10)

    def forward(self, x):
        return self.out(torch.relu(self.hidden(x)))


def predict(model) -> np.ndarray:
    """What the network reads for each of the 1,000 test digits, in eval mode without a graph."""
    _, Xt, _ = tensors()
    with netview.looking(model):
        return model(Xt).argmax(dim=1).numpy()


def train(labels, show=None) -> tuple:
    """DigitsMLP(64) trained on the 6,000 training images with these labels: 10 epochs of mini-batches of 64, with SGD
    and a learning rate of 0.5, the same seeds every time. Gives back (curve, epochs): every 10 mini-batches the epoch
    so far and the share of the test digits read wrong, and what it reads for each test digit after every epoch.
    `show` hears every point of the curve as a dict with "x", "wrong", "epoch" and "pred"; a picture that fails never
    stops the run."""
    X, _, yt = tensors()
    y = torch.as_tensor(np.asarray(labels), dtype=torch.int64)
    torch.manual_seed(0)
    model = DigitsMLP(HIDDEN)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    torch.manual_seed(2)
    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH, shuffle=True)
    truth = yt.numpy()
    curve, epochs = [], []
    for epoch in range(EPOCHS):
        for b, (xb, yb) in enumerate(loader, 1):
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if b % EVERY == 0 or b == len(loader):
                pred = predict(model)
                curve.append((epoch + b / len(loader), float(np.mean(pred != truth))))
                if show is not None:
                    try:
                        show({"x": curve[-1][0], "wrong": curve[-1][1], "epoch": epoch + 1, "pred": pred})
                    except Timeout:
                        raise
                    except Exception:
                        pass
        epochs.append(predict(model))
    return curve, epochs


def flipped(kind: str, share: float, flip=None) -> np.ndarray:
    """The training labels of a run: at random with flip_random and seed 1, or the first share of the 7s as 1s, with
    `flip` (the learner's flip_digit) or the course's."""
    labels = LABELS.value.copy()
    if kind == "random":
        return flip_random(labels, share, SEED)
    return np.asarray((flip or flip_digit)(labels, SOURCE, TARGET, share))


@functools.lru_cache(maxsize=16)
def course_run(kind: str, share: float) -> tuple:
    """A run of the course on labels flipped by the course's functions, kept once trained."""
    return train(flipped(kind, share))


def clean() -> np.ndarray:
    """What the network trained on the clean labels reads for each test digit."""
    return course_run("random", 0.0)[1][-1]


def accuracy(pred) -> float:
    return float(np.mean(np.asarray(pred) == TEST_LABELS.value))


def counts() -> list:
    """How many test digits of each kind there are."""
    return [int(np.sum(TEST_LABELS.value == d)) for d in range(10)]


# --- the pictures ----------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = "padding:0.15rem 0.45rem;text-align:right;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);font-size:0.82rem"
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_MARK = "background:var(--ea-data-wash);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"


def _decimal(v: float, lang: str, digits: int) -> str:
    text = f"{v:.{digits}f}".replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def _fn(seen):
    run = seen.mine
    if run is None:
        return None
    fn = run.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _call(fn, *args):
    """The learner's function on copies of args within the time limit of a check; None when it stops with an error,
    and the Timeout passes on."""
    try:
        return with_time_limit(lambda: fn(*(copy.deepcopy(a) for a in args)))
    except Timeout:
        raise
    except Exception:
        return None


def one_each() -> list:
    """The first test digit of each kind, 0 to 9: what the demo shows the network read."""
    labels = TEST_LABELS.value
    return [int(np.flatnonzero(labels == d)[0]) for d in range(10)]


def now_picture(pred, title: str, lang: str) -> str:
    """What the network reads for the first test digit of each kind, under `title`, in ink where it is right, in the
    color of mistakes with the right digit in brackets where it is not."""
    _, Xt, _ = tensors()
    shown = one_each()
    right = [str(int(TEST_LABELS.value[p])) for p in shown]
    said = [str(int(pred[p])) for p in shown]
    alt = {"en": "The first test digit of each kind, each with the digit the network reads, and in brackets the right digit where it is wrong: ",
           "nl": "Het eerste testcijfer van elke soort, elk met het cijfer dat het netwerk leest, en tussen haakjes het goede cijfer waar het fout is: "}
    return yours.guesses([Xt[p].reshape(28, 28).numpy() for p in shown], said, right,
                         alt[lang] + "; ".join(f"{r}: {g}" for r, g in zip(right, said)) + ".", title=title)


def _bar(v: float, role: str, lang: str) -> str:
    """One bar of a share from 0 to 1, with the share beside it."""
    width = max(0.0, min(1.0, v)) * 100
    return (f'<span style="display:inline-flex;align-items:center;gap:0.4rem;width:100%">'
            f'<span style="flex:1;height:0.6rem;background:var(--ea-line);border-radius:3px;min-width:5rem">'
            f'<span style="display:block;height:100%;width:{width:.1f}%;background:var(--ea-{role});border-radius:3px"></span></span>'
            f'<span style="min-width:2.6rem;text-align:right">{_decimal(v, lang, 2)}</span></span>')


def digit_bars(columns: list, lang: str, head: str) -> str:
    """A table with a row for each digit and a bar for each column: columns are (title, ten shares or None, role)."""
    rows = [f'<tr><th style="{_HEAD};text-align:left">{html.escape(head)}</th>'
            + "".join(f'<th style="{_HEAD};text-align:left">{html.escape(t)}</th>' for t, _, _ in columns) + "</tr>"]
    for d in range(10):
        cells = []
        for _, shares, role in columns:
            ok = shares is not None and d < len(shares) and isinstance(shares[d], (int, float)) and math.isfinite(shares[d])
            cells.append(f'<td style="{_CELL};min-width:9rem">{_bar(float(shares[d]), role, lang) if ok else ""}</td>')
        rows.append(f'<tr><td style="{_CELL};text-align:left">{d}</td>' + "".join(cells) + "</tr>")
    return f'<table style="border-collapse:collapse;margin:0.3rem 0">{"".join(rows)}</table>'


_FLIP_WORDS = {
    "en": {"title": "Your flip_digit(LABELS, 7, 1, 0.8)",
           "digit": "label", "before": "in LABELS", "after": "after your flip_digit",
           "sevens": "The first 14 images labeled 7 in LABELS, each with the label your flip_digit gives it",
           "said": "Each column counts the training images with that label, before and after your flip_digit. The "
                   "images are the first 14 sevens of the training digits: a label in the color of mistakes is one "
                   "your function flipped, with the right digit in brackets.",
           "fine": "The same as the course's flip_digit: 521 of the 651 sevens are now labeled 1.",
           "wrong": "A dashed purple frame marks each count that differs from the course's flip_digit, which leaves "
                    "{s} sevens and makes {o} ones.",
           "stops": "Your flip_digit stops with an error on flip_digit(LABELS, 7, 1, 0.8): the check above says where.",
           "off": "Your flip_digit gives {got}, not 6,000 labels from 0 to 9, so there is nothing to count."},
    "nl": {"title": "Jouw flip_digit(LABELS, 7, 1, 0.8)",
           "digit": "label", "before": "in LABELS", "after": "na jouw flip_digit",
           "sevens": "De eerste 14 afbeeldingen met label 7 in LABELS, elk met het label dat jouw flip_digit ze geeft",
           "said": "Elke kolom telt de trainingsafbeeldingen met dat label, voor en na jouw flip_digit. De "
                   "afbeeldingen zijn de eerste 14 zevens van de trainingscijfers: een label in de kleur van fouten "
                   "heeft je functie omgedraaid, met het goede cijfer tussen haakjes.",
           "fine": "Hetzelfde als de flip_digit van de course: 521 van de 651 zevens hebben nu het label 1.",
           "wrong": "Een gestreept paars kader wijst elk aantal aan dat verschilt van de flip_digit van de course, "
                    "die {s} zevens overlaat en {o} enen maakt.",
           "stops": "Jouw flip_digit stopt met een fout bij flip_digit(LABELS, 7, 1, 0.8): de check hierboven zegt waar.",
           "off": "Jouw flip_digit geeft {got}, geen 6000 labels van 0 tot en met 9, dus er is niets te tellen."},
}


def _cut(text: str, n: int = 80) -> str:
    return text if len(text) <= n else text[:n - 1] + "…"


def _labels_of(got):
    """got as 6,000 whole labels from 0 to 9, or None."""
    try:
        a = np.asarray(got)
    except Exception:
        return None
    if a.shape != (len(LABELS.value),) or not np.issubdtype(a.dtype, np.integer) or a.min() < 0 or a.max() > 9:
        return None
    return a.astype(np.int64)


def flip_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _FLIP_WORDS[seen.lang]
    got = _call(fn, LABELS.value, SOURCE, TARGET, SHARE)
    if got is None:
        return yours.figure(w["title"], _note(w["stops"]))
    mine = _labels_of(got)
    if mine is None:
        return yours.figure(w["title"], _note(w["off"].format(got=_cut(_show_rounded(got, 3)))))
    want = flip_digit(LABELS.value, SOURCE, TARGET, SHARE)
    before = np.bincount(LABELS.value, minlength=10)
    after, should = np.bincount(mine, minlength=10), np.bincount(want, minlength=10)
    head = "".join(f'<th style="{_HEAD}">{d}</th>' for d in range(10))
    row_before = "".join(f'<td style="{_CELL}">{int(k)}</td>' for k in before)
    row_after = "".join(f'<td style="{_CELL}{";" + _OFF if a != s else ""}">{int(a)}</td>' for a, s in zip(after, should))
    table = (f'<table style="border-collapse:collapse;margin:0.3rem 0"><tr><th style="{_HEAD};text-align:left">{w["digit"]}</th>{head}</tr>'
             f'<tr><td style="{_CELL};text-align:left">{w["before"]}</td>{row_before}</tr>'
             f'<tr><td style="{_CELL};text-align:left">{w["after"]}</td>{row_after}</tr></table>')
    X = _data()["train_images"]
    sevens = np.flatnonzero(LABELS.value == SOURCE)[:14]
    pics = yours.guesses([X[p].reshape(28, 28) for p in sevens], [str(int(mine[p])) for p in sevens], ["7"] * len(sevens),
                         w["sevens"] + ": " + ", ".join(str(int(mine[p])) for p in sevens) + ".", title=w["sevens"])
    same = bool(np.array_equal(mine, want))
    note = w["said"] + " " + (w["fine"] if same else w["wrong"].format(s=int(should[SOURCE]), o=int(should[TARGET])))
    return yours.figure(w["title"], table + pics + _note(note))


_PER_WORDS = {
    "en": {"title": "Your per_digit on two networks",
           "digit": "digit", "clean": "clean labels", "sevens": "80% of the 7s labeled 1",
           "said": "Each row is one digit: the share of its test images that each network reads right, as your "
                   "per_digit gives it. The first network trained on the clean labels, the second on the labels of "
                   "exercise 1.",
           "fine": "The same as the course's per_digit.",
           "wrong": "Your per_digit gives other shares than the course's: the check above says what goes wrong.",
           "stops": "Your per_digit stops with an error on the guesses of the networks: the check above says where.",
           "off": "Your per_digit gives {got}, not ten shares, so there is nothing to draw."},
    "nl": {"title": "Jouw per_digit op twee netwerken",
           "digit": "cijfer", "clean": "schone labels", "sevens": "80% van de 7'en met label 1",
           "said": "Elke rij is één cijfer: het deel van de testafbeeldingen ervan dat elk netwerk goed leest, zoals "
                   "jouw per_digit het geeft. Het eerste netwerk trainde op de schone labels, het tweede op de labels "
                   "van oefening 1.",
           "fine": "Hetzelfde als de per_digit van de course.",
           "wrong": "Jouw per_digit geeft andere aandelen dan die van de course: de check hierboven zegt wat er misgaat.",
           "stops": "Jouw per_digit stopt met een fout op de gokken van de netwerken: de check hierboven zegt waar.",
           "off": "Jouw per_digit geeft {got}, geen tien aandelen, dus er is niets te tekenen."},
}


def _shares_of(got):
    """got as ten finite numbers, or None."""
    try:
        values = [float(v) for v in list(got)]
    except Exception:
        return None
    return values if len(values) == 10 and all(math.isfinite(v) for v in values) else None


def per_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _PER_WORDS[seen.lang]
    preds = [clean(), course_run("sevens", SHARE)[1][-1]]
    got = [_call(fn, p, TEST_LABELS.value) for p in preds]
    if any(g is None for g in got):
        return yours.figure(w["title"], _note(w["stops"]))
    shares = [_shares_of(g) for g in got]
    if any(s is None for s in shares):
        return yours.figure(w["title"], _note(w["off"].format(got=_cut(_show_rounded(got[0], 3)))))
    want = [per_digit(p, TEST_LABELS.value) for p in preds]
    same = all(math.isclose(a, b, abs_tol=1e-9) for s, t in zip(shares, want) for a, b in zip(s, t))
    table = digit_bars([(w["clean"], shares[0], "ink-3"), (w["sevens"], shares[1], "ink")], seen.lang, w["digit"])
    return yours.figure(w["title"], table + _note(w["said"] + " " + (w["fine"] if same else w["wrong"])))


_VERDICT_WORDS = {
    "en": {"title": "Your verdict, digit by digit",
           "digit": "digit", "n": "test images", "clean": "clean", "sevens": "7s as 1s", "random": "30% at random",
           "yours": "your verdict",
           "said": "Each row is one digit: how many test images it has, and the share of them each network reads "
                   "right. Your verdict compares each poisoned network with the clean one, with n the number of test "
                   "images of that digit; a shaded verdict says worse.",
           "fine": "The same as the course's verdict.",
           "wrong": "A dashed purple frame marks each verdict that differs from the course's.",
           "stops": "Your verdict stops with an error on one of these digits: the check above says where."},
    "nl": {"title": "Jouw verdict, cijfer voor cijfer",
           "digit": "cijfer", "n": "testafbeeldingen", "clean": "schoon", "sevens": "7'en als 1",
           "random": "30% willekeurig", "yours": "jouw verdict",
           "said": "Elke rij is één cijfer: hoeveel testafbeeldingen het heeft, en het deel daarvan dat elk netwerk "
                   "goed leest. Jouw verdict vergelijkt elk vergiftigd netwerk met het schone, met n het aantal "
                   "testafbeeldingen van dat cijfer; een gekleurd verdict zegt worse.",
           "fine": "Hetzelfde als het verdict van de course.",
           "wrong": "Een gestreept paars kader wijst elk verdict aan dat verschilt van dat van de course.",
           "stops": "Jouw verdict stopt met een fout bij een van deze cijfers: de check hierboven zegt waar."},
}


def verdict_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _VERDICT_WORDS[seen.lang]
    base = per_digit(clean(), TEST_LABELS.value)
    runs = [per_digit(course_run("sevens", SHARE)[1][-1], TEST_LABELS.value),
            per_digit(course_run("random", RANDOM_SHARE)[1][-1], TEST_LABELS.value)]
    n = counts()
    head = "".join(f'<th style="{_HEAD}">{html.escape(t)}</th>' for t in
                   (w["digit"], w["n"], w["clean"], w["sevens"], w["yours"], w["random"], w["yours"]))
    rows, same = [f"<tr>{head}</tr>"], True
    for d in range(10):
        cells = [f'<td style="{_CELL}">{d}</td>', f'<td style="{_CELL}">{n[d]}</td>',
                 f'<td style="{_CELL}">{_decimal(base[d], seen.lang, 2)}</td>']
        for shares in runs:
            got = _call(fn, base[d], shares[d], n[d])
            if got is None:
                return yours.figure(w["title"], _note(w["stops"]))
            want = verdict(base[d], shares[d], n[d])
            style = (";" + _MARK if got == "worse" else "") + (";" + _OFF if got != want else "")
            same = same and got == want
            cells.append(f'<td style="{_CELL}">{_decimal(shares[d], seen.lang, 2)}</td>')
            cells.append(f'<td style="{_CELL}{style}">{html.escape(_cut(str(got), 12))}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    table = f'<table style="border-collapse:collapse;margin:0.3rem 0">{"".join(rows)}</table>'
    return yours.figure(w["title"], table + _note(w["said"] + " " + (w["fine"] if same else w["wrong"])))


def idea_picture(lang: str) -> str:
    """Flips at random: for every share p of flipped labels, the share of a digit's training images that still carry
    its own label, 1 − p, and the share that carries each one of the nine other digits, p / 9. The lines cross at 0.9."""
    words = {"en": {"title": "Flipped at random: the right label and each wrong one", "x": "share of labels flipped",
                    "y": "share of a digit's images", "right": "its own digit, 1 − p", "wrong": "each other digit, p / 9",
                    "alt": "Two straight lines over the share p of labels flipped at random, from 0 to 1: the share of "
                           "a digit's training images that still carry its own label falls from 1 to 0, and the share "
                           "that carries any one other digit rises from 0 to 0.11. They cross at p = 0.9, where the "
                           "right label is no more common than each wrong one."},
             "nl": {"title": "Willekeurig omgedraaid: het goede label en elk fout label", "x": "deel van de labels omgedraaid",
                    "y": "deel van de afbeeldingen van een cijfer", "right": "het eigen cijfer, 1 − p",
                    "wrong": "elk ander cijfer, p / 9",
                    "alt": "Twee rechte lijnen boven het deel p van de labels dat willekeurig is omgedraaid, van 0 tot "
                           "1: het deel van de trainingsafbeeldingen van een cijfer dat nog het eigen label heeft, daalt "
                           "van 1 naar 0, en het deel met één bepaald ander cijfer stijgt van 0 naar 0,11. Ze kruisen "
                           "bij p = 0,9, waar het goede label niet vaker voorkomt dan elk fout label."}}[lang]
    p = chart.Panel((0, 1), (0, 1), title=words["title"], xlabel=words["x"], ylabel=words["y"])
    p.line([0.9, 0.9], [0, 1], "ink-3", 1.0, dashed=True)
    p.line([0, 1], [1, 0], "ink", 2.2)
    p.line([0, 1], [0, 1 / 9], "loss", 2.2)
    p.text(0.08, 0.86, words["right"], "ink", "start", halo=True)
    p.text(0.08, 0.16, words["wrong"], "loss", "start", halo=True)
    p.text(0.885, 0.6, "p = " + _decimal(0.9, lang, 1), "ink-2", "end", halo=True)
    svg = chart.figure([p], alt=words["alt"], width=520, height=280)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


# --- the demo --------------------------------------------------------------------------------------------------------

def mine(box_text, correct: bool, name: str):
    """The learner's own function `name` when its exercise is right, else None: the demo then uses the course's."""
    if not correct or not isinstance(box_text, str):
        return None
    from everything_ai.check import run_code
    run = run_code(box_text, {})
    fn = run.names.get(name) if run.error is None else None
    return fn if callable(fn) else None


def demo(kind: str, share: float, flip=None, show=None) -> tuple | str:
    """The demo's run: the labels flipped at random (flip_random, seed 1) or the first share of the 7s as 1s (with
    `flip`, the learner's flip_digit, or the course's), and the network trained on them. Gives back (labels, curve,
    epochs), or "flip" when `flip` stops with an error or gives no 6,000 labels from 0 to 9, "slow-flip" when it takes
    more than ten seconds, or "slow" when the training takes more than two minutes."""
    try:
        labels = _labels_of(with_time_limit(lambda: flipped(kind, share, flip), 10))
    except Timeout:
        return "slow-flip"
    except Exception:
        return "flip"
    if labels is None:
        return "flip"
    try:
        curve, epochs = with_time_limit(lambda: train(labels, show), 120)
    except Timeout:
        return "slow"
    return labels, curve, epochs


def curve_picture(curve: list, words: dict, lang: str, at: int | None = None) -> str:
    """The share of the test digits read wrong during the run, with the clean run dashed; with `at`, a line at that
    epoch, for the rewind."""
    p = chart.Panel((0, EPOCHS), (0, 1), title=words["title"], xlabel=words["x"], ylabel=words["y"])
    if at is not None:
        p.line([at, at], [0, 1], "ink-2", 1.2)
    base = course_run("random", 0.0)[0]
    p.line([x for x, _ in base], [y for _, y in base], "loss", 1.6, dashed=True)
    p.line([x for x, _ in curve], [y for _, y in curve], "loss", 2.2)
    p.text(EPOCHS - 0.2, min(0.95, curve[-1][1] + 0.07), words["run"], "loss", "end", halo=True)
    p.text(EPOCHS - 0.2, max(0.04, base[-1][1] - 0.04), words["clean"], "loss", "end", halo=True)
    svg = chart.figure([p], alt=words["alt"], width=live.CHART_WIDTH if at is not None else 520,
                       height=240 if at is not None else 300)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


def now_parts(pred, words: dict, lang: str) -> list:
    """What the demo shows beside its chart: the share of each digit the network reads right now, and its guesses for
    the first test digit of each kind."""
    shares = per_digit(np.asarray(pred), TEST_LABELS.value)
    base = per_digit(clean(), TEST_LABELS.value)
    return [digit_bars([(words["clean"], base, "ink-3"), (words["now"], shares, "ink")], lang, words["digit"]),
            now_picture(pred, words["guess"], lang)]


def rewind_frame(curve: list, epochs: list, e: int, words: dict, lang: str) -> str:
    """The rewind at epoch e: the curve with a line at e, and beside it what the network read after that epoch."""
    parts = now_parts(epochs[e - 1], words | {"now": words["after"].format(e=e), "guess": words["guess_after"].format(e=e)}, lang)
    return live.rewound(curve_picture(curve, words, lang, at=e), parts)


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["flip-digit", "per-digit", "verdict", "noise", "count"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# exercise 1: flip_digit
_D_HEAD = ("import numpy as np\n\n\n"
           "def flip_digit(labels, source, target, share):\n"
           "    # a copy of labels in which the first share of the images labeled source are labeled target\n"
           "    flipped = labels.copy()\n")
_D_START = _D_HEAD + "    ...\n    return flipped\n"
_D_BODY = ("    places = np.arange(len(labels))[labels == source]\n"
           "    count = round(share * len(places))\n"
           "    flipped[places[:count]] = target\n"
           "    return flipped\n")
_D = _D_HEAD + _D_BODY
_D_ALL = _D_HEAD + _D_BODY.replace("round(share * len(places))", "round(share * len(labels))")
_D_INT = _D_HEAD + _D_BODY.replace("round(share * len(places))", "int(share * len(places))")
_D_LAST = _D_HEAD + _D_BODY.replace("flipped[places[:count]]", "flipped[places[-count:]]")
_D_EVERY = _D_HEAD + "    flipped[labels == source] = target\n    return flipped\n"
_D_INPUT = _D_HEAD + _D_BODY.replace("    flipped[places[:count]] = target\n", "    labels[places[:count]] = target\n")
_D_SWAP = _D_HEAD + _D_BODY.replace("labels == source", "labels == target").replace("= target\n", "= source\n")

# exercise 2: per_digit
_P_HEAD = ("import numpy as np\n\n\n"
           "def per_digit(pred, labels):\n"
           "    # for each digit 0 to 9: the share of the images labeled with that digit that pred reads right\n")
_P_START = _P_HEAD + "    shares = []\n    ...\n    return shares\n"
_P_BODY = ("    shares = []\n"
           "    for digit in range(10):\n"
           "        mine = labels == digit\n"
           "        shares.append(float(np.mean(pred[mine] == digit)))\n"
           "    return shares\n")
_P = _P_HEAD + _P_BODY
_P_ALL = _P_HEAD + _P_BODY.replace("np.mean(pred[mine] == digit)", "np.mean(pred == labels)")
_P_SAID = _P_HEAD + _P_BODY.replace("mine = labels == digit", "mine = pred == digit").replace(
    "np.mean(pred[mine] == digit)", "np.mean(labels[mine] == digit)")
_P_SHARE = _P_HEAD + _P_BODY.replace("np.mean(pred[mine] == digit)", "np.mean(pred == digit)")
_P_COUNT = _P_HEAD + _P_BODY.replace("float(np.mean(pred[mine] == digit))", "int(np.sum(pred[mine] == digit))")
_P_NINE = _P_HEAD + _P_BODY.replace("range(10)", "range(9)")

# exercise 3: verdict
_V_HEAD = ("import math\n\n\n"
           "def verdict(clean, poisoned, n):\n"
           "    # \"worse\" if poisoned is below clean by more than the margin of error of clean on n images, else \"chance\"\n")
_V_START = _V_HEAD + "    margin = ...\n    ...\n    return \"chance\"\n"
_V_BODY = ("    margin = 2 * math.sqrt(clean * (1 - clean) / n)\n"
           "    if clean - poisoned > margin:\n"
           "        return \"worse\"\n"
           "    return \"chance\"\n")
_V = _V_HEAD + _V_BODY
_V_ONE = _V_HEAD + _V_BODY.replace("2 * math.sqrt", "math.sqrt")
_V_ROOT = _V_HEAD + _V_BODY.replace("2 * math.sqrt(clean * (1 - clean) / n)", "2 * clean * (1 - clean) / n")
_V_BOTH = _V_HEAD + _V_BODY.replace("if clean - poisoned > margin:", "if abs(clean - poisoned) > margin:")
_V_SIGN = _V_HEAD + _V_BODY.replace("if clean - poisoned > margin:", "if poisoned - clean > margin:")
_V_OTHER = _V_HEAD + _V_BODY.replace("clean * (1 - clean)", "poisoned * (1 - poisoned)")
_V_BOOL = _V_HEAD + ("    margin = 2 * math.sqrt(clean * (1 - clean) / n)\n"
                                                                   "    return clean - poisoned > margin\n")


def _not_text(got, text) -> bool:
    return isinstance(got, (bool, np.bool_))


def _not_list(got, text) -> bool:
    return not isinstance(got, (list, tuple, np.ndarray))


def _floats(got):
    """got as a list of finite numbers, or None."""
    if isinstance(got, (str, dict, bytes)):
        return None
    try:
        values = [float(v) for v in list(got)]
    except Exception:
        return None
    return values if all(math.isfinite(v) for v in values) else None


def _same_shares(got, want) -> bool:
    """The same shares, as a list or a NumPy array, each within a billionth of the course's."""
    values, wanted = _floats(got), _floats(want)
    return (values is not None and wanted is not None and len(values) == len(wanted)
            and all(abs(a - b) <= 1e-9 for a, b in zip(values, wanted)))


def _same_labels(got, want) -> bool:
    try:
        a = np.asarray(got)
    except Exception:
        return False
    return a.shape == np.shape(want) and np.issubdtype(a.dtype, np.integer) and bool(np.array_equal(a, want))


def _labels_show(v) -> str:
    """{got} and {want} of flip_digit: how many of each label, for a NumPy array of labels; the picture shows more."""
    try:
        a = np.asarray(v)
        if a.ndim == 1 and np.issubdtype(a.dtype, np.integer) and len(a) > 12:
            k = np.bincount(a.clip(0), minlength=10)
            return "an array with " + ", ".join(f"{int(k[d])}×{d}" for d in range(len(k)))
    except Exception:
        pass
    return _show_rounded(v, 3)


def _names_error(e) -> bool:
    return isinstance(e, NameError) and any(n in str(e) for n in ("LABELS", "TEST_LABELS", "GUESSES", "TRUTH", "C"))


_NAMES = Mistake("lesson-names", error=_names_error,
                 en="Your function uses a name of the lesson, such as `LABELS`, and the check does not give it that name: use the names between the parentheses of the `def` line.",
                 nl="Je functie gebruikt een naam van de les, zoals `LABELS`, en de check geeft die naam niet mee: gebruik de namen tussen de haakjes van de `def`-regel.")


EXERCISES = {
    "flip-digit": FunctionBox(
        name="flip-digit",
        picture=flip_picture,
        function="flip_digit",
        start=_D_START,
        solution=_D,
        visible=[(LABELS, SOURCE, TARGET, SHARE)],
        hidden=[(LABELS, 3, 8, 0.5), (LABELS, 7, 1, 0.05), (SMALL, 2, 5, 0.7), (SMALL, 2, 5, 0.0), (SMALL, 2, 5, 1.0), (LABELS, 0, 6, 0.25)],
        compare=_same_labels,
        show=_labels_show,
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("every", wrong=_D_EVERY,
                    en="`{call}` gives {got}: every image labeled `source` now says `target`. Flip only the first part of them: `share` of the places where the label is `source`.",
                    nl="`{call}` geeft {got}: elke afbeelding met label `source` zegt nu `target`. Draai alleen het eerste deel om: `share` van de plaatsen waar het label `source` is."),
            Mistake("all-labels", wrong=_D_ALL,
                    en="`{call}` gives {got}, and it should give {want}. Your count is `share` of all labels, and that is more than there are images labeled `source`. Take `share` of the places where the label is `source`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je aantal is `share` van alle labels, en dat zijn er meer dan er afbeeldingen met label `source` zijn. Neem `share` van de plaatsen waar het label `source` is."),
            Mistake("int", wrong=_D_INT,
                    en="`{call}` gives {got}, and it should give {want}. `int` cuts 0.8 × 651 = 520.8 down to 520; `round` gives 521. Round the count with `round`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. `int` maakt van 0,8 × 651 = 520,8 een 520; `round` geeft 521. Rond het aantal af met `round`."),
            Mistake("last", wrong=_D_LAST,
                    en="`{call}` gives the right number of flips, at the wrong places: your function flips the last ones. Flip the first: the places from the start up to `count`.",
                    nl="`{call}` geeft het goede aantal omgedraaide labels, op de verkeerde plaatsen: je functie draait de laatste om. Draai de eerste om: de plaatsen vanaf het begin tot `count`."),
            Mistake("input", wrong=_D_INPUT,
                    en="`{call}` gives the labels as they were: your function changes `labels`, and gives back `flipped`, the copy it made first. Change the copy, so that the course's labels stay clean.",
                    nl="`{call}` geeft de labels zoals ze waren: je functie verandert `labels`, en geeft `flipped` terug, de kopie die ze eerst maakte. Verander de kopie, zodat de labels van de course schoon blijven."),
            Mistake("swap", wrong=_D_SWAP,
                    en="`{call}` gives {got}: your function turns `target` into `source`. It should be the other way round: images labeled `source` get the label `target`.",
                    nl="`{call}` geeft {got}: je functie maakt van `target` een `source`. Het moet andersom: afbeeldingen met label `source` krijgen het label `target`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: find the places where the label is `source`, and give the first `share` of them the label `target` in `flipped`.",
                      "nl": "Vervang de `...`: zoek de plaatsen waar het label `source` is, en geef het eerste deel `share` daarvan het label `target` in `flipped`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Of the images labeled `source`, the first `share`, rounded with `round`, get the label `target`; the rest stays as it was.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Van de afbeeldingen met label `source` krijgt het eerste deel `share`, afgerond met `round`, het label `target`; de rest blijft zoals het was."},
            "almost": {"en": "Right for `flip_digit(LABELS, 7, 1, 0.8)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `labels`, `source`, `target` and `share` it gets?",
                       "nl": "Goed voor `flip_digit(LABELS, 7, 1, 0.8)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `labels`, `source`, `target` en `share` die ze krijgt?"},
            "correct": {"en": "Your `flip_digit` is right in all {n} cases. Of the 651 images labeled 7, the first 521 now say 1: 8.7% of the 6,000 labels. The picture shows the first sevens with their new labels.",
                        "nl": "Je `flip_digit` klopt in alle {n} gevallen. Van de 651 afbeeldingen met label 7 zeggen de eerste 521 nu 1: 8,7% van de 6000 labels. Het plaatje laat de eerste zevens zien met hun nieuwe labels."},
        },
    ),
    "per-digit": FunctionBox(
        name="per-digit",
        picture=per_picture,
        function="per_digit",
        start=_P_START,
        solution=_P,
        visible=[(GUESSES, TRUTH)],
        hidden=[(TRUTH, TRUTH), (Named(lambda: np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), "OTHER"), TRUTH),
                (Named(lambda: np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 0]), "PAIRS"),
                 Named(lambda: np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]), "PAIRED"))],
        compare=_same_shares,
        show=lambda v: _show_rounded(v, 3),
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("counts", wrong=_P_COUNT,
                    en="`{call}` gives `{got}`: how many images of each digit it reads right. Give the share: the right ones divided by all images with that label, which `np.mean` does for you.",
                    nl="`{call}` geeft `{got}`: hoeveel afbeeldingen van elk cijfer goed gelezen zijn. Geef het deel: de goede gedeeld door alle afbeeldingen met dat label, wat `np.mean` voor je doet."),
            Mistake("all", wrong=_P_ALL,
                    en="`{call}` gives `{got}`: the same share for every digit, the share of all images read right. Look only at the images with that label, with a mask.",
                    nl="`{call}` geeft `{got}`: hetzelfde deel voor elk cijfer, het deel van alle afbeeldingen dat goed gelezen is. Kijk alleen naar de afbeeldingen met dat label, met een mask."),
            Mistake("said", wrong=_P_SAID,
                    en="`{call}` gives `{got}`, and it should give `{want}`. Your mask keeps the images the network reads as that digit; keep the images whose label is that digit, and see how many of them it reads right.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Je mask houdt de afbeeldingen over die het netwerk als dat cijfer leest; houd de afbeeldingen over waarvan het label dat cijfer is, en kijk hoeveel daarvan het goed leest."),
            Mistake("share-of-all", wrong=_P_SHARE,
                    en="`{call}` gives `{got}`: for each digit, the share of all images the network reads as that digit. Divide by the images with that label only: use the mask on `pred` too.",
                    nl="`{call}` geeft `{got}`: voor elk cijfer het deel van alle afbeeldingen dat het netwerk als dat cijfer leest. Deel alleen door de afbeeldingen met dat label: gebruik de mask ook op `pred`."),
            Mistake("nine", wrong=_P_NINE,
                    en="`{call}` gives nine shares, and there are ten digits: `range(9)` stops at 8. Go from 0 to 9.",
                    nl="`{call}` geeft negen aandelen, en er zijn tien cijfers: `range(9)` stopt bij 8. Ga van 0 tot en met 9."),
            Mistake("not-a-list", matches=_not_list,
                    en="`{call}` gives `{got}`, not a list of ten shares. Add one share for each digit to `shares`, and give back `shares`.",
                    nl="`{call}` geeft `{got}`, geen lijst van tien aandelen. Voeg voor elk cijfer één aandeel toe aan `shares`, en geef `shares` terug."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: for each digit from 0 to 9, add to `shares` the share of the images labeled with that digit that `pred` reads right.",
                      "nl": "Vervang de `...`: voeg voor elk cijfer van 0 tot en met 9 aan `shares` het deel toe van de afbeeldingen met dat label dat `pred` goed leest."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. For each digit: of the images whose label is that digit, the share whose guess in `pred` is that digit too.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Voor elk cijfer: van de afbeeldingen waarvan het label dat cijfer is, het deel waarvan de gok in `pred` ook dat cijfer is."},
            "almost": {"en": "Right for `per_digit(GUESSES, TRUTH)`. But `{call}` gives `{got}`, and it should give `{want}`. Does your function use the `pred` and `labels` it gets?",
                       "nl": "Goed voor `per_digit(GUESSES, TRUTH)`. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gebruikt je functie de `pred` en `labels` die ze krijgt?"},
            "correct": {"en": "Your `per_digit` is right in all {n} cases: GUESSES reads the 7s and the 9s half right, and every other digit all right. The picture runs it on two real networks.",
                        "nl": "Je `per_digit` klopt in alle {n} gevallen: GUESSES leest de 7'en en de 9'en half goed, en elk ander cijfer helemaal goed. Het plaatje gebruikt je functie op twee echte netwerken."},
        },
    ),
    "verdict": FunctionBox(
        name="verdict",
        picture=verdict_picture,
        function="verdict",
        start=_V_START,
        solution=_V,
        visible=[(0.929, 0.881, 1000)],
        hidden=[(0.929, 0.92, 1000), (0.939, 0.03, 99), (0.9, 0.85, 110), (0.85, 0.95, 100), (0.98, 0.9, 50), (0.9, 0.89, 100),
                (0.929, 0.9, 1000)],
        first=[_DOTS],
        mistakes=[
            Mistake("bool", matches=_not_text,
                    en="`{call}` gives `{got}`, and it should give the word `{want}`: `\"worse\"` or `\"chance\"`, as text.",
                    nl="`{call}` geeft `{got}`, en het moet het woord `{want}` zijn: `\"worse\"` of `\"chance\"`, als tekst."),
            Mistake("sign", wrong=_V_SIGN,
                    en="`{call}` gives `{got}`: your function looks for a rise, poisoned minus clean. A drop is clean minus poisoned.",
                    nl="`{call}` geeft `{got}`: je functie zoekt een stijging, poisoned min clean. Een daling is clean min poisoned."),
            Mistake("both-ways", wrong=_V_BOTH,
                    en="`{call}` gives `{got}`: the poisoned network scores higher here, 0.95 against 0.85, and your function calls that worse. Only a drop counts: leave out `abs`.",
                    nl="`{call}` geeft `{got}`: het vergiftigde netwerk scoort hier hoger, 0,95 tegen 0,85, en je functie noemt dat worse. Alleen een daling telt: laat `abs` weg."),
            Mistake("one-se", wrong=_V_ONE,
                    en="`{call}` gives `{got}`, and it should give `{want}`: your margin of error is one standard error. The margin of error is two: `2 *` the square root.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je foutmarge is één standaardfout. De foutmarge is er twee: `2 *` de wortel."),
            Mistake("root", wrong=_V_ROOT,
                    en="`{call}` gives `{got}`, and it should give `{want}`: your margin of error has no square root, so it is far too small. Take `math.sqrt` of `clean * (1 - clean) / n`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je foutmarge heeft geen wortel, en is dus veel te klein. Neem `math.sqrt` van `clean * (1 - clean) / n`."),
            Mistake("other-margin", wrong=_V_OTHER,
                    en="`{call}` gives `{got}`, and it should give `{want}`: your margin of error is that of the poisoned score. The question is whether poisoned lies outside the margin of error of the clean score: use `clean` in the margin of error.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je foutmarge is die van de vergiftigde score. De vraag is of poisoned buiten de foutmarge van de schone score ligt: gebruik `clean` in de foutmarge."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: the margin of error of `clean` on `n` test images, and `\"worse\"` when `poisoned` is lower than `clean` by more than that.",
                      "nl": "Vervang de twee `...`: de foutmarge van `clean` op `n` testafbeeldingen, en `\"worse\"` als `poisoned` meer dan dat lager is dan `clean`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. The margin of error is 2 times the square root of clean times 1 minus clean, divided by n; worse when clean minus poisoned is larger than that.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. De foutmarge is 2 keer de wortel van clean keer 1 min clean, gedeeld door n; worse als clean min poisoned groter is dan dat."},
            "almost": {"en": "Right for `verdict(0.929, 0.881, 1000)`. But `{call}` gives `{got}`, and it should give `{want}`. Does your function use the `clean`, `poisoned` and `n` it gets?",
                       "nl": "Goed voor `verdict(0.929, 0.881, 1000)`. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gebruikt je functie de `clean`, `poisoned` en `n` die ze krijgt?"},
            "correct": {"en": "Your `verdict` is right in all {n} cases. On 1,000 test images the margin of error of 0.929 is 0.016, so a drop to 0.881 is worse, and one to 0.92 may be chance. The picture runs it digit by digit.",
                        "nl": "Je `verdict` klopt in alle {n} gevallen. Op 1000 testafbeeldingen is de foutmarge van 0,929 gelijk aan 0,016, dus een daling naar 0,881 is worse, en een daling naar 0,92 kan toeval zijn. Het plaatje gebruikt je functie cijfer voor cijfer."},
        },
    ),
}

CASES = [
    ("flip-digit", _D_START, "empty", None),
    ("flip-digit", _D, "correct", None),
    ("flip-digit", _D_HEAD + "    count = round(share * np.sum(labels == source))\n    for i in range(len(labels)):\n"
                   "        if labels[i] == source and count > 0:\n            flipped[i] = target\n"
                   "            count = count - 1\n    return flipped\n", "correct", None),
    ("flip-digit", _D_EVERY, "wrong", "every"),
    ("flip-digit", _D_ALL, "wrong", "all-labels"),
    ("flip-digit", _D_INT, "wrong", "int"),
    ("flip-digit", _D_LAST, "wrong", "last"),
    ("flip-digit", _D_INPUT, "wrong", "input"),
    ("flip-digit", _D_SWAP, "wrong", "swap"),
    ("flip-digit", _D_HEAD + "    flipped = LABELS.copy()\n    return flipped\n", "error", "lesson-names"),
    ("flip-digit", _D_HEAD + "    flipped[labels == 7] = 1\n    return flipped\n", "wrong", None),
    ("per-digit", _P_START, "empty", None),
    ("per-digit", _P, "correct", None),
    ("per-digit", _P_HEAD + "    shares = []\n    for digit in range(10):\n        right = 0\n        total = 0\n"
                  "        for i in range(len(labels)):\n            if labels[i] == digit:\n                total = total + 1\n"
                  "                if pred[i] == digit:\n                    right = right + 1\n"
                  "        shares.append(right / total)\n    return shares\n", "correct", None),
    ("per-digit", _P_HEAD + "    return np.array([np.mean(pred[labels == d] == d) for d in range(10)])\n", "correct", None),
    ("per-digit", _P_COUNT, "wrong", "counts"),
    ("per-digit", _P_ALL, "wrong", "all"),
    ("per-digit", _P_SAID, "wrong", "said"),
    ("per-digit", _P_SHARE, "wrong", "share-of-all"),
    ("per-digit", _P_NINE, "wrong", "nine"),
    ("per-digit", _P_HEAD + "    return 0.9\n", "wrong", "not-a-list"),
    ("per-digit", _P_HEAD + "    return [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 0.5]\n", "almost", None),
    ("per-digit", _P_HEAD + "    labels = TRUTH\n    return []\n", "error", "lesson-names"),
    ("verdict", _V_START, "empty", None),
    ("verdict", _V, "correct", None),
    ("verdict", _V_HEAD + "    if poisoned < clean - 2 * (clean * (1 - clean) / n) ** 0.5:\n        return \"worse\"\n"
                "    return \"chance\"\n", "correct", None),
    ("verdict", _V_BOOL, "wrong", "bool"),
    ("verdict", _V_SIGN, "wrong", "sign"),
    ("verdict", _V_BOTH, "wrong", "both-ways"),
    ("verdict", _V_ONE, "wrong", "one-se"),
    ("verdict", _V_ROOT, "wrong", "root"),
    ("verdict", _V_OTHER, "wrong", "other-margin"),
    ("verdict", _V_HEAD + "    return \"worse\"\n", "almost", None),
    ("verdict", _V_HEAD + "    return \"chance\"\n", "wrong", None),
]
