# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-federated-learning: the 6,000 training digits of QMNIST spread over ten phones of 600
each, in two ways (a bit of every digit, or sorted by digit so that each phone holds two or three digits), and a
server that trains DigitsMLP(64) with FedAvg without ever seeing a digit. The three functions the learner writes:
fedavg (the weighted mean of one weight, in Python), fedavg_rows (the same for every weight at once, with NumPy)
and fedavg_states (for whole state_dicts, with PyTorch); with the pictures of what they give. The demo runs FedAvg
with the learner's own fedavg_states on both kinds of phones, and shows while it trains what the server's model
guesses (decision 1.17)."""
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
from everything_ai.check import FunctionBox, Mistake, Timeout, _show_rounded, same_value, with_time_limit

DATA = "data/qmnist-small.npz"   # the first 6,000 training and 1,000 test images of QMNIST: see data/README.md
PHONES, PER_PHONE = 10, 600      # ten phones of 600 training digits each
SHARDS = 20                      # sorted by digit: 20 pieces of 300, two for each phone
CHOSEN, ROUNDS = 5, 20           # each round the server picks 5 of the 10 phones; the demo runs 20 rounds
HIDDEN, LR, BATCH = 64, 0.1, 10  # DigitsMLP(64); on a phone one epoch of mini-batches of 10, with a learning rate of 0.1
torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)   # the first optimizer in a process takes a second: not in a check


class Named:
    """A value the check calls by the name the lesson gives it, such as ROWS, so that a message stays short. The
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

    def __array__(self, dtype=None, copy=None):   # so that the picture under exercise 2 can draw ROWS and SIZES
        return np.asarray(self.value, dtype=dtype)


# --- the course's own solutions -------------------------------------------------------------------------------------

def fedavg(weights, sizes):
    total = 0
    examples = 0
    for k in range(len(weights)):
        total = total + sizes[k] * weights[k]
        examples = examples + sizes[k]
    return total / examples


def fedavg_rows(W, sizes):
    return sizes @ W / np.sum(sizes)


def fedavg_states(states, sizes):
    examples = 0
    for n in sizes:
        examples = examples + n
    new = {}
    for name in states[0]:
        total = 0
        for k in range(len(states)):
            total = total + sizes[k] * states[k][name]
        new[name] = total / examples
    return new


# --- the values of the exercises ------------------------------------------------------------------------------------

WEIGHTS, SIZES = [0.2, 0.8, 0.6], [100, 300, 600]   # exercise 1: one weight of three phones, and their examples
ROWS = Named(lambda: np.array([[0.2, -1.0, 3.0, 0.0], [0.8, 0.0, 3.0, 0.0], [0.6, 1.0, 3.0, 1.0]]), "ROWS")
SIZES_NP = Named(lambda: np.array(SIZES), "SIZES")


def _states() -> list:
    """Exercise 3: the same three phones, each with a tiny model of two tensors, w and b; the first column of w holds
    the weights of exercise 1."""
    return [{"w": torch.tensor([[0.2, -1.0], [3.0, 0.0]]), "b": torch.tensor([0.0, 1.0])},
            {"w": torch.tensor([[0.8, 0.0], [3.0, 0.0]]), "b": torch.tensor([0.0, 1.0])},
            {"w": torch.tensor([[0.6, 1.0], [3.0, 1.0]]), "b": torch.tensor([1.0, 1.0])}]


STATES = Named(_states, "STATES")


def _linears() -> list:
    """Two state_dicts of nn.Linear(3, 2), as PyTorch gives them, for a hidden case."""
    out = []
    for seed in (3, 4):
        torch.manual_seed(seed)
        out.append(nn.Linear(3, 2).state_dict())
    return out


LINEARS = Named(_linears, "LINEARS")


# --- the digits, the phones and the model ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def tensors():
    """The training images and labels and the test images and labels as tensors; every pixel divided by 255."""
    d = np.load(paths.ROOT / DATA)
    return (torch.tensor(d["train_images"], dtype=torch.float32) / 255, torch.tensor(d["train_labels"], dtype=torch.int64),
            torch.tensor(d["test_images"], dtype=torch.float32) / 255, torch.tensor(d["test_labels"], dtype=torch.int64))


def mixed() -> np.ndarray:
    """Ten phones with a bit of every digit, as in the worked example: the 6,000 digits in a random order, 600 each."""
    return np.random.default_rng(0).permutation(PHONES * PER_PHONE).reshape(PHONES, PER_PHONE)


def by_digit() -> np.ndarray:
    """Ten phones that each hold two or three digits, as McMahan et al. split MNIST, at a tenth of the size: the
    digits sorted by label, cut into 20 pieces of 300, and two pieces at random for each phone."""
    _, y, _, _ = tensors()
    pieces = np.argsort(y.numpy(), kind="stable").reshape(SHARDS, -1)
    pick = np.random.default_rng(0).permutation(SHARDS).reshape(PHONES, -1)
    return pieces[pick].reshape(PHONES, PER_PHONE)


def digits_on(split: np.ndarray) -> list:
    """The digits each phone holds."""
    _, y, _, _ = tensors()
    return [sorted(set(y[torch.tensor(p)].tolist())) for p in split]


class DigitsMLP(nn.Module):
    """The MLP of torch-mlp-digits."""
    def __init__(self, hidden):
        super().__init__()
        self.hidden = nn.Linear(784, hidden)
        self.out = nn.Linear(hidden, 10)

    def forward(self, x):
        return self.out(torch.relu(self.hidden(x)))


def start() -> dict:
    """The server's first weights: DigitsMLP(64) with seed 0, as in the worked example."""
    torch.manual_seed(0)
    return DigitsMLP(HIDDEN).state_dict()


def train_on_phone(state: dict, idx) -> dict:
    """One phone's part of a round, as in the worked example: load the server's weights, one epoch of mini-batches of
    10 over its own digits, and give back the new weights."""
    X, y, _, _ = tensors()
    idx = torch.as_tensor(np.asarray(idx))
    model = DigitsMLP(HIDDEN)
    model.load_state_dict(state)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    for xb, yb in DataLoader(TensorDataset(X[idx], y[idx]), batch_size=BATCH, shuffle=True):
        loss = loss_fn(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.state_dict()


def model_of(state) -> DigitsMLP:
    """A DigitsMLP(64) with these weights; load_state_dict stops with an error when a name or a shape is missing."""
    model = DigitsMLP(HIDDEN)
    model.load_state_dict({k: torch.as_tensor(v, dtype=torch.float32) for k, v in dict(state).items()})
    return model


def accuracy(state) -> float:
    """The share of the 1,000 test digits a model with these weights reads right, in eval mode without a graph."""
    _, _, Xt, yt = tensors()
    model = model_of(state)
    with netview.looking(model):
        return (model(Xt).argmax(dim=1) == yt).float().mean().item()


def one_each() -> list:
    """The first test digit of each kind, 0 to 9: what the demo shows the server's model guess."""
    _, _, _, yt = tensors()
    return [int((yt == k).nonzero()[0]) for k in range(10)]


def guesses(state) -> list:
    """What a model with these weights guesses for the first test digit of each kind."""
    _, _, Xt, _ = tensors()
    model = model_of(state)
    with netview.looking(model):
        return [int(g) for g in model(Xt[one_each()]).argmax(dim=1)]


def now_picture(said: list, title: str, lang: str) -> str:
    """What the server's model guesses at one moment: its guess for the first test digit of each kind, under `title`,
    in ink where it is right, in the color of mistakes where it is not."""
    _, _, Xt, yt = tensors()
    shown = one_each()
    right = [str(int(yt[place])) for place in shown]
    said = [str(g) for g in said]
    alt = {"en": "The first test digit of each kind, each with the digit the model guessed, and in brackets the right digit where it is wrong: ",
           "nl": "Het eerste testcijfer van elke soort, elk met het cijfer dat het model gokte, en tussen haakjes het goede cijfer waar het fout is: "}
    return yours.guesses([Xt[place].reshape(28, 28).numpy() for place in shown], said, right,
                         alt[lang] + "; ".join(f"{r}: {g}" for r, g in zip(right, said)) + ".", title=title)


@functools.lru_cache(maxsize=1)
def example() -> tuple:
    """The worked example again: the server's first weights, and phones 0 and 1 of `mixed` after one epoch each."""
    phones = mixed()
    server = start()
    torch.manual_seed(2)
    a = train_on_phone(server, phones[0])
    b = train_on_phone(server, phones[1])
    return server, a, b


# the picture under exercise 3: three phones with 600, 300 and 100 digits, each one epoch from the server's weights
THREE = [600, 300, 100]


@functools.lru_cache(maxsize=1)
def three_phones() -> list:
    phones = mixed()
    server = start()
    torch.manual_seed(5)
    return [train_on_phone(server, phones[k][:n]) for k, n in enumerate(THREE)]


# --- the demo: FedAvg with the learner's fedavg_states ---------------------------------------------------------------

def federated(split: np.ndarray, average, show=None, rounds: int = ROUNDS) -> tuple:
    """FedAvg on these phones, with `average` as the server's step: every round the server picks 5 of the 10 phones
    at random, each trains from the server's weights, and `average(states, sizes)` gives the new weights. The same
    seeds every time, so the same run. Gives back, after every round, the share of the test digits the server's model
    reads wrong, and its guesses for the first test digit of each kind. `show` hears every round as a dict with
    "round", "rounds", "wrong" and "said"; a picture that fails never stops the run."""
    state = start()
    rng = np.random.default_rng(1)
    torch.manual_seed(2)
    wrong, said = [], []
    for r in range(1, rounds + 1):
        chosen = rng.choice(PHONES, size=CHOSEN, replace=False)
        states, sizes = [], []
        for k in chosen:
            states.append(train_on_phone(state, split[k]))
            sizes.append(len(split[k]))
        state = average(states, sizes)
        wrong.append(1 - accuracy(state))
        said.append(guesses(state))
        if show is not None:
            try:
                show({"round": r, "rounds": rounds, "wrong": wrong[-1], "said": said[-1]})
            except Timeout:
                raise
            except Exception:
                pass
    return wrong, said


def mine(box_text, correct: bool):
    """The learner's own fedavg_states when exercise 3 is right, else None: the demo then uses the course's."""
    if not correct or not isinstance(box_text, str):
        return None
    from everything_ai.check import run_code
    run = run_code(box_text, {})
    fn = run.names.get("fedavg_states") if run.error is None else None
    return fn if callable(fn) else None


def demo(average, show=None) -> dict | str:
    """Both runs of the demo, phones with a bit of every digit first, with `average` (the learner's fedavg_states, or
    the course's). Gives back {"mixed": (wrong, said), "by_digit": (wrong, said)}, or "average" when `average`
    stops with an error or gives weights that DigitsMLP(64) cannot load, or "slow" after three minutes. `show`
    hears the run's name with every round."""
    out = {}

    def work():
        for name, split in (("mixed", mixed()), ("by_digit", by_digit())):
            out[name] = federated(split, average, None if show is None else (lambda p, n=name: show(n, p)))

    try:
        with_time_limit(work, 180)
    except Timeout:
        return "slow"
    except Exception:
        return "average"
    return out


def curves_picture(runs: dict, words: dict, lang: str, at: int | None = None) -> str:
    """The share of the test digits read wrong after every round, of both runs: a bit of every digit dashed, two or
    three digits per phone solid, both in the color of mistakes; with `at`, a line at that round, for the rewind."""
    p = chart.Panel((0.5, ROUNDS + 0.5), (0, 1), title=words["title"], xlabel=words["x"], ylabel=words["y"])
    xs = list(range(1, ROUNDS + 1))
    if at is not None:
        p.line([at, at], [0, 1], "ink-2", 1.2)
    for name, dashed in (("mixed", True), ("by_digit", False)):
        ys = runs[name][0]
        p.line(xs[:len(ys)], ys, "loss", 2.2, dashed=dashed)
        p.points(xs[:len(ys)], ys, "loss", 2.5, hollow=dashed)
        if name == "by_digit":   # its name above its end; that of the lower line under its middle, away from the rewind
            p.text(ROUNDS + 0.3, ys[-1] + 0.06, words[name], "loss", "end", halo=True)
        else:
            p.text((ROUNDS + 1) / 2, min(ys) - 0.06, words[name], "loss", "middle", halo=True)
    svg = chart.figure([p], alt=words["alt"], width=live.CHART_WIDTH if at is not None else 520,
                       height=240 if at is not None else 300)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg


def rewind_frame(runs: dict, r: int, words: dict, lang: str) -> str:
    """The rewind at round r: both curves with a line at r, and beside them what the server's model of each run
    guessed after that round (decision 1.17)."""
    pics = [now_picture(runs[name][1][r - 1], words["guess_" + name].format(r=r), lang) for name in ("mixed", "by_digit")]
    return live.rewound(curves_picture(runs, words, lang, at=r), pics)


# --- the pictures under the checks ----------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"


def _fn(seen):
    run = seen.mine
    if run is None:
        return None
    fn = run.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _limited(work):
    return with_time_limit(work)


def _number(v):
    try:
        if isinstance(v, bool):
            return None
        out = float(v)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _decimal(v: float, lang: str, digits: int) -> str:
    text = f"{v:.{digits}f}".replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


_MEAN_WORDS = {
    "en": {"title": "Your fedavg(weights, sizes) on the three phones",
           "x": "the weight", "plain": "plain mean {v}", "yours": "yours {v}",
           "key": "Each dot is one phone's weight, larger the more examples it has: 100, 300 and 600. The dashed "
                  "line is the plain mean of the three weights, {m}.",
           "inside": "The solid line is what your fedavg gives, {v}.",
           "outside": "Your fedavg gives {v}, outside the axis, which runs from 0 to 1.",
           "none": "Your fedavg gives no number here.",
           "wrong": "The course's fedavg gives {w}."},
    "nl": {"title": "Jouw fedavg(weights, sizes) op de drie telefoons",
           "x": "de weight", "plain": "gewoon gemiddelde {v}", "yours": "jouw {v}",
           "key": "Elke stip is de weight van één telefoon, groter naarmate die telefoon meer voorbeelden heeft: 100, 300 en 600. De "
                  "gestreepte lijn is het gewone gemiddelde van de drie weights, {m}.",
           "inside": "De doorgetrokken lijn is wat jouw functie fedavg geeft, {v}.",
           "outside": "Jouw functie fedavg geeft {v}, en dat ligt buiten de as, die van 0 tot 1 loopt.",
           "none": "Jouw functie fedavg geeft hier geen getal.",
           "wrong": "De fedavg van de course geeft {w}."},
}


def mean_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _MEAN_WORDS[seen.lang]
    try:
        got = _number(_limited(lambda: fn(list(WEIGHTS), list(SIZES))))
    except Exception:
        got = None
    want, plain = fedavg(WEIGHTS, SIZES), sum(WEIGHTS) / len(WEIGHTS)
    p = chart.Panel((0, 1), (0, 1), title="", xlabel=w["x"], yticks=[])
    p.line([plain, plain], [0.08, 0.92], "ink-2", 1.4, dashed=True)
    p.text(plain - 0.02, 0.9, w["plain"].format(v=_decimal(plain, seen.lang, 2)), "ink-2", "end", halo=True)
    for weight, n in zip(WEIGHTS, SIZES):
        p.points([weight], [0.45], "data", r=3 + 9 * math.sqrt(n / max(SIZES)), hollow=True)
        p.text(weight, 0.2, str(n), "ink-2", halo=True)
    note = w["key"].format(m=_decimal(plain, seen.lang, 2))
    if got is not None and 0 <= got <= 1:
        p.line([got, got], [0.3, 0.75], "param", 2.4)
        p.text(got + 0.02, 0.72, w["yours"].format(v=_decimal(got, seen.lang, 2)), "param", "start", halo=True)
        note += " " + w["inside"].format(v=_decimal(got, seen.lang, 2))
    elif got is not None:
        note += " " + w["outside"].format(v=_decimal(got, seen.lang, 2))
    else:
        note += " " + w["none"]
    if got is None or not math.isclose(got, want, rel_tol=1e-6):
        note += " " + w["wrong"].format(w=_decimal(want, seen.lang, 2))
    svg = chart.figure([p], alt=note, width=520, height=200)
    svg = chart.with_decimal_comma(svg) if seen.lang == "nl" else svg
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(note)}</div>')


_STATES_WORDS = {
    "en": {"title": "Your fedavg_states(states, sizes) on three phones' DigitsMLP(64)",
           "phone": "phone with {n} digits", "yours": "your average",
           "right": "{k} of 1,000 test digits right",
           "key": "Each phone trained one epoch on its own digits, from the same weights of the server. The bars count "
                  "the test digits each model reads right; under them, what the model of your average guesses for "
                  "the first test digit of each kind.",
           "guess": "What the model of your average guesses",
           "load": "What your fedavg_states gives back is no state_dict that DigitsMLP(64) can load: {e}",
           "stops": "Your fedavg_states stops with an error on the weights of these three phones.",
           "kind": "it is a {t}, not a dictionary.", "missing": "it has no {n}.",
           "shape": "{n} has the shape {a}, where DigitsMLP(64) needs {b}.", "other": "loading it stops with an error.",
           "differs": "The course's weighted average reads {k} right."},
    "nl": {"title": "Jouw fedavg_states(states, sizes) op het DigitsMLP(64) van drie telefoons",
           "phone": "telefoon met {n} cijfers", "yours": "jouw gemiddelde",
           "right": "{k} van de 1000 testcijfers goed",
           "key": "Elke telefoon trainde één epoch op de eigen cijfers, vanaf dezelfde weights van de server. De balken "
                  "tellen de testcijfers die elk model goed leest; daaronder wat het model van jouw gemiddelde gokt "
                  "voor het eerste testcijfer van elke soort.",
           "guess": "Wat het model van jouw gemiddelde gokt",
           "load": "Wat je fedavg_states teruggeeft, is geen state_dict die DigitsMLP(64) kan laden: {e}",
           "stops": "Je fedavg_states stopt met een fout op de weights van deze drie telefoons.",
           "kind": "het is een {t}, geen dictionary.", "missing": "er staat geen {n} in.",
           "shape": "{n} heeft de vorm {a}, en DigitsMLP(64) heeft {b} nodig.", "other": "het laden stopt met een fout.",
           "differs": "Het gewogen gemiddelde van de course leest er {k} goed."},
}


def _bar(label: str, k: int, role: str) -> str:
    return (f'<div style="display:flex;align-items:center;gap:0.6rem;margin:0.2rem 0;font-family:var(--ea-font-ui);'
            f'font-size:0.85rem;color:var(--ea-ink)"><span style="flex:0 0 12rem">{html.escape(label)}</span>'
            f'<span style="flex:1;height:0.7rem;background:var(--ea-line);border-radius:3px;max-width:14rem">'
            f'<span style="display:block;height:100%;width:{k / 10:.1f}%;background:var(--ea-{role});border-radius:3px">'
            f'</span></span><span style="min-width:3rem;text-align:right">{k}</span></div>')


def _why_not(got, w: dict) -> str:
    """Why DigitsMLP(64) cannot load what the learner's fedavg_states gave back, in a few words."""
    if not isinstance(got, dict):
        return w["kind"].format(t=type(got).__name__)
    need = DigitsMLP(HIDDEN).state_dict()
    for name, t in need.items():
        if name not in got:
            return w["missing"].format(n=name)
        shape = tuple(getattr(got[name], "shape", ()))
        if shape != tuple(t.shape):
            return w["shape"].format(n=name, a=shape, b=tuple(t.shape))
    return w["other"]


def states_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _STATES_WORDS[seen.lang]
    states = three_phones()
    rights = [round(1000 * accuracy(s)) for s in states]
    body = "".join(_bar(w["phone"].format(n=n), k, "ink-3") for n, k in zip(THREE, rights))
    want = round(1000 * accuracy(fedavg_states(states, THREE)))
    got = None
    try:
        got = _limited(lambda: fn(copy.deepcopy(states), list(THREE)))
        k = round(1000 * accuracy(got))
        said = guesses(got)
    except Timeout:
        return ""
    except Exception:
        said = w["stops"] if got is None else w["load"].format(e=_why_not(got, w))   # no result at all, or one it cannot load
        return yours.figure(w["title"], body + f'<div style="{_UI}">{html.escape(said)}</div>')
    body += _bar(w["yours"], k, "param")
    note = w["key"] + ("" if k == want else " " + w["differs"].format(k=want))
    pic = now_picture(said, w["guess"], seen.lang)
    return yours.figure(w["title"], body + pic + f'<div style="{_UI}">{html.escape(note)}</div>')


# --- how the check shows a state_dict -------------------------------------------------------------------------------

def _rounded(v):
    if isinstance(v, list):
        return [_rounded(x) for x in v]
    return round(float(v), 4) if isinstance(v, float) else v


def show_state(v) -> str:
    """A dictionary of tensors, short: each name with its numbers as lists, to four decimals."""
    if isinstance(v, dict):
        parts = []
        for k, t in v.items():
            value = _rounded(t.tolist()) if isinstance(t, torch.Tensor) else t
            parts.append(f"{k!r}: {value}")
        return "`{" + ", ".join(parts) + "}`"
    return _show_rounded(v, 4)


def same_state(got, want) -> bool:
    return isinstance(got, dict) and same_value(dict(got), dict(want))


# --- the exercises --------------------------------------------------------------------------------------------------

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There are still `...` in your code. Replace each of them with code.",
                nl="Er staan nog `...` in je code. Vervang ze allemaal door code.")

# exercise 1: fedavg
_F_HEAD = ("def fedavg(weights, sizes):\n"
           "    # each weight times its phone's number of examples, added up, divided by all examples\n")
_F_START = _F_HEAD + "    total = 0\n    examples = 0\n    ...\n    return ...\n"
_F_LOOP = "    for k in range(len(weights)):\n"
_F = (_F_HEAD + "    total = 0\n    examples = 0\n" + _F_LOOP + "        total = total + sizes[k] * weights[k]\n"
      "        examples = examples + sizes[k]\n    return total / examples\n")
_F_MEAN = (_F_HEAD + "    total = 0\n" + _F_LOOP + "        total = total + weights[k]\n"
           "    return total / len(weights)\n")
_F_COUNT = (_F_HEAD + "    total = 0\n" + _F_LOOP + "        total = total + sizes[k] * weights[k]\n"
            "    return total / len(weights)\n")
_F_SUM = (_F_HEAD + "    total = 0\n" + _F_LOOP + "        total = total + sizes[k] * weights[k]\n    return total\n")
_F_LAST = (_F_HEAD + "    total = 0\n    examples = 0\n" + _F_LOOP + "        total = total + sizes[k] * weights[k]\n"
           "        examples = sizes[k]\n    return total / examples\n")
_F_FIRST = (_F_HEAD + "    total = 0\n    examples = 0\n" + _F_LOOP + "        total = total + sizes[k] * weights[k]\n"
            "        examples = examples + sizes[k]\n        return total / examples\n")

# exercise 2: fedavg_rows
_W_HEAD = ("import numpy as np\n\n\n"
           "def fedavg_rows(W, sizes):\n"
           "    # every column at once: the sizes times the rows, added up, divided by all examples\n")
_W_START = _W_HEAD + "    return ...\n"
_W = _W_HEAD + "    return sizes @ W / np.sum(sizes)\n"
_W_MEAN = _W_HEAD + "    return np.mean(W, axis=0)\n"
_W_SUM = _W_HEAD + "    return sizes @ W\n"
_W_COUNT = _W_HEAD + "    return sizes @ W / len(sizes)\n"
_W_TURNED = _W_HEAD + "    return W @ sizes / np.sum(sizes)\n"

# exercise 3: fedavg_states
_S_HEAD = ("def fedavg_states(states, sizes):\n"
           "    # every name: the phones' tensors, each times its number of examples, added up, divided by all examples\n")
_S_START = _S_HEAD + "    examples = 0\n    for n in sizes:\n        examples = examples + n\n    new = {}\n    ...\n    return new\n"
_S_EX = "    examples = 0\n    for n in sizes:\n        examples = examples + n\n"
_S_NAMES = "    new = {}\n    for name in states[0]:\n"
_S = (_S_HEAD + _S_EX + _S_NAMES + "        total = 0\n        for k in range(len(states)):\n"
      "            total = total + sizes[k] * states[k][name]\n        new[name] = total / examples\n    return new\n")
_S_MEAN = (_S_HEAD + _S_NAMES + "        total = 0\n        for k in range(len(states)):\n"
           "            total = total + states[k][name]\n        new[name] = total / len(states)\n    return new\n")
_S_SUM = (_S_HEAD + _S_NAMES + "        total = 0\n        for k in range(len(states)):\n"
          "            total = total + sizes[k] * states[k][name]\n        new[name] = total\n    return new\n")
_S_COUNT = (_S_HEAD + _S_NAMES + "        total = 0\n        for k in range(len(states)):\n"
            "            total = total + sizes[k] * states[k][name]\n        new[name] = total / len(states)\n    return new\n")
_S_FIRST = _S_HEAD + "    return states[0]\n"
_S_ONCE = (_S_HEAD + _S_EX + "    new = {}\n    total = 0\n    for name in states[0]:\n        for k in range(len(states)):\n"
           "            total = total + sizes[k] * states[k][name]\n        new[name] = total / examples\n    return new\n")
_S_LIST = (_S_HEAD + _S_EX + _S_NAMES + "        total = 0\n        for k in range(len(states)):\n"
           "            total = total + sizes[k] * states[name]\n        new[name] = total / examples\n    return new\n")

REQUIRED = ["fedavg", "fedavg-rows", "fedavg-states", "two-digits", "server-learns"]

EXERCISES = {
    "fedavg": FunctionBox(
        name="fedavg",
        picture=mean_picture,
        function="fedavg",
        show=lambda v: _show_rounded(v, 4),
        start=_F_START,
        solution=_F,
        visible=[(list(WEIGHTS), list(SIZES))],
        hidden=[([1.0], [7]), ([0.0, 1.0], [1, 3]), ([-2.0, 4.0, 1.0, 0.5], [50, 50, 200, 700]), ([3, 5], [600, 600]),
                ([0.5, 1.0], [300, 100])],
        first=[_DOTS],
        mistakes=[
            Mistake("mean", wrong=_F_MEAN,
                    en="`{call}` gives {got}: the plain mean of the weights, as if every phone held as many examples. In FedAvg a phone counts by its number of examples: multiply each weight by `sizes[k]`, and divide by all examples together.",
                    nl="`{call}` geeft {got}: het gewone gemiddelde van de weights, alsof elke telefoon evenveel voorbeelden had. Bij FedAvg telt een telefoon mee naar het eigen aantal voorbeelden: vermenigvuldig elke weight met `sizes[k]`, en deel door alle voorbeelden samen."),
            Mistake("count", wrong=_F_COUNT,
                    en="`{call}` gives {got}, and it should give {want}. Your code divides by the number of phones. Divide by the number of examples of all phones together: add every `sizes[k]` up in `examples`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je code deelt door het aantal telefoons. Deel door het aantal voorbeelden van alle telefoons samen: tel elke `sizes[k]` op in `examples`."),
            Mistake("sum", wrong=_F_SUM,
                    en="`{call}` gives {got}: the weights times the sizes, added up, but not yet divided by the number of examples of all phones together.",
                    nl="`{call}` geeft {got}: de weights keer de aantallen, opgeteld, maar nog niet gedeeld door het aantal voorbeelden van alle telefoons samen."),
            Mistake("last", wrong=_F_LAST,
                    en="`{call}` gives {got}, and it should give {want}. Your code divides by the examples of the last phone only. Add every phone's examples up: `examples = examples + sizes[k]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je code deelt alleen door de voorbeelden van de laatste telefoon. Tel de voorbeelden van elke telefoon op: `examples = examples + sizes[k]`."),
            Mistake("first", wrong=_F_FIRST,
                    en="`{call}` gives {got}: the weight of the first phone only. Is your `return` inside the loop? Then the function stops after the first phone. Put it after the loop, at the level of `for`.",
                    nl="`{call}` geeft {got}: alleen de weight van de eerste telefoon. Staat je `return` in de loop? Dan stopt de functie na de eerste telefoon. Zet hem na de loop, op de hoogte van `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the first `...` with a loop over the phones that adds up, in `total`, each weight counted as often as its phone has examples, and in `examples` the numbers of examples; and the second `...` with the result.",
                      "nl": "Vervang de eerste `...` door een loop over de telefoons die in `total` elke weight optelt, zo vaak meegeteld als de telefoon voorbeelden heeft, en in `examples` de aantallen voorbeelden; en de tweede `...` door het resultaat."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Multiply each weight by its phone's number of examples, add those up, and divide by all examples together.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Vermenigvuldig elke weight met het aantal voorbeelden van de telefoon, tel dat op, en deel door alle voorbeelden samen."},
            "almost": {"en": "Right for `fedavg([0.2, 0.8, 0.6], [100, 300, 600])`. But `{call}` gives {got}, and it should give {want}. Does your function use every phone in `weights` and `sizes`, whatever their number?",
                       "nl": "Goed voor `fedavg([0.2, 0.8, 0.6], [100, 300, 600])`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie elke telefoon in `weights` en `sizes`, hoeveel het er ook zijn?"},
            "correct": {"en": "Your `fedavg` is right in all {n} cases. The picture puts its result beside the plain mean.",
                        "nl": "Je `fedavg` klopt in alle {n} gevallen. Het plaatje zet de uitkomst naast het gewone gemiddelde."},
        },
    ),
    "fedavg-rows": FunctionBox(
        name="fedavg_rows",
        picture=yours.arrays(0, title=yours.Text(en="Your fedavg_rows(W, sizes) on ROWS and SIZES",
                                                 nl="Jouw fedavg_rows(W, sizes) op ROWS en SIZES")),
        function="fedavg_rows",
        show=lambda v: _show_rounded(v, 4),
        start=_W_START,
        solution=_W,
        visible=[(ROWS, SIZES_NP)],
        hidden=[(np.array([[1.0, 2.0]]), np.array([5])), (np.array([[0.0, 0.0], [1.0, 2.0]]), np.array([1, 3])),
                (np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [4.0, 4.0, 4.0], [1.0, 1.0, 1.0]]), np.array([100, 100, 200, 600]))],
        first=[_DOTS],
        mistakes=[
            Mistake("turned", error=lambda e: isinstance(e, ValueError) and "matmul" in str(e),
                    en="With `{call}`, `@` gets two arrays whose sizes do not fit: in `a @ b`, the length of `a` must be the number of rows of `b`. `sizes` holds one number per phone, and `W` one row per phone.",
                    nl="Met `{call}` krijgt `@` twee arrays waarvan de maten niet passen: bij `a @ b` moet de lengte van `a` gelijk zijn aan het aantal rijen van `b`. `sizes` heeft één getal per telefoon, en `W` één rij per telefoon."),
            Mistake("mean", wrong=_W_MEAN,
                    en="`{call}` gives {got}: the plain mean of each column, as if every phone held as many examples. Weight each row by its phone's number of examples, as in exercise 1, and divide by all examples together.",
                    nl="`{call}` geeft {got}: het gewone gemiddelde van elke kolom, alsof elke telefoon evenveel voorbeelden had. Weeg elke rij naar het aantal voorbeelden van de telefoon, zoals in oefening 1, en deel door alle voorbeelden samen."),
            Mistake("sum", wrong=_W_SUM,
                    en="`{call}` gives {got}: the rows times the sizes, added up per column, but not yet divided by the number of examples of all phones together.",
                    nl="`{call}` geeft {got}: de rijen keer de aantallen, per kolom opgeteld, maar nog niet gedeeld door het aantal voorbeelden van alle telefoons samen."),
            Mistake("count", wrong=_W_COUNT,
                    en="`{call}` gives {got}, and it should give {want}. Your code divides by the number of phones. Divide by the number of examples of all phones together.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je code deelt door het aantal telefoons. Deel door het aantal voorbeelden van alle telefoons samen."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the weighted average of every column at once: a matrix product, divided by all examples together.",
                      "nl": "Vervang de `...` door het gewogen gemiddelde van elke kolom tegelijk: een matrixproduct, gedeeld door alle voorbeelden samen."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Each row is one phone: multiply it by that phone's number of examples, add the rows up, and divide by all examples together.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Elke rij is één telefoon: vermenigvuldig hem met het aantal voorbeelden van die telefoon, tel de rijen op, en deel door alle voorbeelden samen."},
            "almost": {"en": "Right for `fedavg_rows(ROWS, SIZES)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `W` and the `sizes` it gets, whatever their size?",
                       "nl": "Goed voor `fedavg_rows(ROWS, SIZES)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `W` en de `sizes` die hij krijgt, hoe groot ze ook zijn?"},
            "correct": {"en": "Your `fedavg_rows` is right in all {n} cases. The picture shows `ROWS` and `SIZES`, and what your function makes of them.",
                        "nl": "Je `fedavg_rows` klopt in alle {n} gevallen. Het plaatje laat `ROWS` en `SIZES` zien, en wat je functie ervan maakt."},
        },
    ),
    "fedavg-states": FunctionBox(
        name="fedavg_states",
        picture=states_picture,
        function="fedavg_states",
        show=show_state,
        compare=same_state,
        start=_S_START,
        solution=_S,
        visible=[(STATES, list(SIZES))],
        hidden=[([{"w": torch.tensor([1.0, -2.0])}], [5]), (LINEARS, [600, 200]),
                ([{"w": torch.tensor(1.0), "b": torch.tensor([0.0])}, {"w": torch.tensor(3.0), "b": torch.tensor([4.0])}], [1, 3])],
        first=[_DOTS],
        mistakes=[
            Mistake("list", error=lambda e: isinstance(e, TypeError) and "list indices" in str(e),
                    en="With `{call}`, your function stops: a list, such as `states` or `sizes`, takes a number between its brackets, not a name. Phone k's tensor of that name is `states[k][name]`, and its number of examples `sizes[k]`.",
                    nl="Met `{call}` stopt je functie: een lijst, zoals `states` of `sizes`, wil een getal tussen de haken, geen naam. De tensor met die naam van telefoon k is `states[k][name]`, en het aantal voorbeelden `sizes[k]`."),
            Mistake("mean", wrong=_S_MEAN,
                    en="`{call}` gives {got}: for every name the plain mean of the tensors, as if every phone held as many examples. Multiply phone k's tensor by `sizes[k]`, and divide by all examples together.",
                    nl="`{call}` geeft {got}: voor elke naam het gewone gemiddelde van de tensors, alsof elke telefoon evenveel voorbeelden had. Vermenigvuldig de tensor van telefoon k met `sizes[k]`, en deel door alle voorbeelden samen."),
            Mistake("sum", wrong=_S_SUM,
                    en="`{call}` gives {got}: for every name the tensors times the sizes, added up, but not yet divided by the number of examples of all phones together: `examples`.",
                    nl="`{call}` geeft {got}: voor elke naam de tensors keer de aantallen, opgeteld, maar nog niet gedeeld door het aantal voorbeelden van alle telefoons samen: `examples`."),
            Mistake("count", wrong=_S_COUNT,
                    en="`{call}` gives {got}, and it should give {want}. Your code divides by the number of phones. Divide by the number of examples of all phones together: `examples`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je code deelt door het aantal telefoons. Deel door het aantal voorbeelden van alle telefoons samen: `examples`."),
            Mistake("first", wrong=_S_FIRST,
                    en="`{call}` gives {got}: the state_dict of the first phone, unchanged. For every name, add the tensors of all phones, each times its number of examples.",
                    nl="`{call}` geeft {got}: de state_dict van de eerste telefoon, onveranderd. Tel voor elke naam de tensors van alle telefoons op, elk keer zijn aantal voorbeelden."),
            Mistake("once", wrong=_S_ONCE,
                    en="`{call}` gives {got}, and it should give {want}. Your `total` starts at 0 only once, before the names, so the second name goes on adding to the tensors of the first. Set `total = 0` for every name, inside the loop over the names.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je `total` begint maar één keer bij 0, vóór de namen, dus de tweede naam telt verder op bij de tensors van de eerste. Zet `total = 0` voor elke naam, in de loop over de namen."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with a loop over the names in `states[0]`: for each name, add up the phones' tensors of that name, each counted as often as its phone has examples, and put the total divided by `examples` in `new[name]`.",
                      "nl": "Vervang de `...` door een loop over de namen in `states[0]`: tel voor elke naam de tensors met die naam van de telefoons op, elk zo vaak meegeteld als de telefoon voorbeelden heeft, en zet het totaal gedeeld door `examples` in `new[name]`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. For every name: the tensors of all phones, each times its number of examples, added up and divided by all examples together.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Voor elke naam: de tensors van alle telefoons, elk keer zijn aantal voorbeelden, opgeteld en gedeeld door alle voorbeelden samen."},
            "almost": {"en": "Right for `fedavg_states(STATES, [100, 300, 600])`. But `{call}` gives {got}, and it should give {want}. Does your function use every name of `states[0]` and every phone in `states`, whatever their number?",
                       "nl": "Goed voor `fedavg_states(STATES, [100, 300, 600])`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie elke naam van `states[0]` en elke telefoon in `states`, hoeveel het er ook zijn?"},
            "correct": {"en": "Your `fedavg_states` is right in all {n} cases. The picture averages the models of three phones with it.",
                        "nl": "Je `fedavg_states` klopt in alle {n} gevallen. Het plaatje middelt er de modellen van drie telefoons mee."},
        },
    ),
}

CASES = [
    ("fedavg", _F_START, "empty", None),
    ("fedavg", _F, "correct", None),
    ("fedavg", _F_HEAD + "    total = 0\n    for k in range(len(weights)):\n        total = total + sizes[k] * weights[k]\n"
               "    return total / sum(sizes)\n", "correct", None),
    ("fedavg", "import numpy as np\n\n\n" + _F_HEAD + "    return np.sum(np.array(sizes) * np.array(weights)) / np.sum(sizes)\n", "correct", None),
    ("fedavg", _F_MEAN, "wrong", "mean"),
    ("fedavg", _F_COUNT, "wrong", "count"),
    ("fedavg", _F_SUM, "wrong", "sum"),
    ("fedavg", _F_LAST, "wrong", "last"),
    ("fedavg", _F_FIRST, "wrong", "first"),
    ("fedavg", _F_HEAD + "    return weights[-1]\n", "wrong", None),
    ("fedavg-rows", _W_START, "empty", None),
    ("fedavg-rows", _W, "correct", None),
    ("fedavg-rows", _W_HEAD + "    return np.sum(sizes.reshape(-1, 1) * W, axis=0) / np.sum(sizes)\n", "correct", None),
    ("fedavg-rows", _W_HEAD + "    return np.average(W, axis=0, weights=sizes)\n", "correct", None),
    ("fedavg-rows", _W_TURNED, "error", "turned"),
    ("fedavg-rows", _W_MEAN, "wrong", "mean"),
    ("fedavg-rows", _W_SUM, "wrong", "sum"),
    ("fedavg-rows", _W_COUNT, "wrong", "count"),
    ("fedavg-rows", _W_HEAD + "    return W[0]\n", "wrong", None),
    ("fedavg-states", _S_START, "empty", None),
    ("fedavg-states", _S, "correct", None),
    ("fedavg-states", _S_HEAD + _S_NAMES + "        new[name] = 0\n        for k in range(len(states)):\n"
                      "            new[name] = new[name] + sizes[k] / sum(sizes) * states[k][name]\n    return new\n", "correct", None),
    ("fedavg-states", _S_LIST, "error", "list"),
    ("fedavg-states", _S_MEAN, "wrong", "mean"),
    ("fedavg-states", _S_SUM, "wrong", "sum"),
    ("fedavg-states", _S_COUNT, "wrong", "count"),
    ("fedavg-states", _S_FIRST, "wrong", "first"),
    ("fedavg-states", _S_ONCE, "wrong", "once"),
    ("fedavg-states", _S_HEAD + _S_EX + _S_NAMES + "        new[name] = states[-1][name]\n    return new\n", "wrong", None),
]
