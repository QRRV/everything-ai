# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the project lesson lm-bigram-neural: loss_of, the cross-entropy of a weight table on pairs of characters
given as one-hot rows, and train, gradient descent on that table with an L2 penalty; each saved in the learner's
project (work/projects/llm/) when right. And the picture of training on the start of the plays, and a demo that
trains with the learner's code and lets the learned table write."""
import html
import math
import re
from functools import cache

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
import torch.nn.functional as F

from everything_ai import paths, projects
from everything_ai.check import FunctionBox, Mistake, run_code, with_time_limit, Timeout

# The demo writes with the learner's generate of lm-sampling, or the course's version.
projects.builds_on(("llm", "sampling", "lm-sampling", "generate"))

DATA = "data/tiny-shakespeare.txt"   # about 1.1 MB of plays by Shakespeare: see data/README.md
START = 20_000                       # the network trains on the pairs of the first 20,000 characters: fast enough here
STEPS, LR = 200, 100


@cache
def text() -> str:
    return (paths.ROOT / DATA).read_text(encoding="utf-8")


@cache
def chars() -> tuple[str, ...]:
    return tuple(sorted(set(text())))


@cache
def _start_codes() -> list[int]:
    cs = chars()
    return [cs.index(c) for c in text()[:START + 1]]


def start_pairs() -> tuple[list[int], list[int]]:
    """The pairs of the first 20,000 characters, as two lists of character numbers: x the one before, y the one after."""
    codes = _start_codes()
    return codes[:-1], codes[1:]


@cache
def counted_nll() -> float:
    """The average NLL of the counted table on the first 20,000 characters, counted on those same characters."""
    x, y = start_pairs()
    n = np.zeros((len(chars()), len(chars())))
    np.add.at(n, (x, y), 1)
    p = n / np.maximum(n.sum(axis=1, keepdims=True), 1)
    return float(-np.log(p[x, y]).mean())


@cache
def counted_prob(a: str, b: str) -> float:
    x, y = start_pairs()
    n = np.zeros((len(chars()), len(chars())))
    np.add.at(n, (x, y), 1)
    i, j = chars().index(a), chars().index(b)
    return float(n[i, j] / n[i].sum())


def _course_train(x, y, steps, lr, lam, record=False):
    x, y = torch.tensor(x), torch.tensor(y)
    W = torch.zeros((len(chars()), len(chars())), requires_grad=True)
    opt = torch.optim.SGD([W], lr=lr)
    losses = []
    for step in range(steps):
        loss = F.cross_entropy(W[x], y) + lam * (W ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if record:
            losses.append(loss.item())
    return (W, losses) if record else W


@cache
def course_run(lam: float = 0.0):
    """The course's own training on the first 20,000 characters: the table, and the loss after every step."""
    x, y = start_pairs()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)   # on small tensors, one thread is faster than several
    try:
        W, losses = _course_train(x, y, STEPS, LR, lam, record=True)
    finally:
        torch.set_num_threads(threads)
    return W.detach(), losses


def net_prob(a: str, b: str, lam: float = 0.0) -> float:
    W, _ = course_run(lam)
    i, j = chars().index(a), chars().index(b)
    return float(torch.softmax(W, dim=1)[i, j])


def net_nll(lam: float = 0.0) -> float:
    W, _ = course_run(lam)
    x, y = start_pairs()
    return float(F.cross_entropy(W[torch.tensor(x)], torch.tensor(y)))


def number(x: float, digits: int, lang: str) -> str:
    s = f"{x:.{digits}f}"
    return s.replace(".", ",") if lang == "nl" else s


def grouped(n: int, lang: str) -> str:
    s = f"{n:,}"
    return s.replace(",", ".") if lang == "nl" else s


def text_block(s: str) -> str:
    """A text the model wrote, in a box that wraps its lines, so that a long line stays readable on a phone."""
    return (f'<pre style="white-space:pre-wrap;word-break:break-word;margin:0.25rem 0 0.75rem;padding:0.6rem 0.8rem;'
            f'border-radius:6px;background:var(--ea-inset);color:var(--ea-ink);font-family:var(--ea-font-code);'
            f'font-size:0.875rem;line-height:1.45">{html.escape(s)}</pre>')


# --- the picture: the loss while training, against the NLL of the counted table ---------------------------------------

def loss_curve(labels: dict) -> str:
    """The loss after every step of training on the first 20,000 characters (purple, the loss), and the NLL of the
    counted table on the same characters as a dashed line: what the network can at best reach."""
    _, losses = course_run(0.0)
    target = counted_nll()
    W, LEFT, TOP, PH, BOTTOM, RIGHT = 540, 40, 30, 170, 34, 16
    lo, hi = 2.0, 4.5
    def X(i):
        return LEFT + (W - LEFT - RIGHT) * i / (len(losses) - 1)
    def Y(v):
        return TOP + PH * (hi - v) / (hi - lo)
    p = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    for v in (2.0, 2.5, 3.0, 3.5, 4.0, 4.5):
        p.append(f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" style="stroke:var(--ea-line)"/>'
                 f'<text x="{LEFT - 6}" y="{Y(v) + 4:.1f}" style="font-size:11px;fill:var(--ea-ink-2);text-anchor:end">'
                 f'{html.escape(labels["fmt"](v))}</text>')
    for s in (0, 50, 100, 150, 200):
        i = min(s, len(losses) - 1)
        p.append(f'<text x="{X(i):.1f}" y="{TOP + PH + 16}" style="font-size:11px;fill:var(--ea-ink-2);text-anchor:middle">{s}</text>')
    p.append(f'<text x="{(LEFT + W) / 2:.0f}" y="{TOP + PH + 30}" style="font-size:12px;fill:var(--ea-ink-2);text-anchor:middle">{html.escape(labels["steps"])}</text>')
    path = " ".join(f'{"M" if i == 0 else "L"}{X(i):.1f} {Y(v):.1f}' for i, v in enumerate(losses))
    p.append(f'<path d="{path}" style="fill:none;stroke:var(--ea-loss);stroke-width:2"/>')
    p.append(f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{Y(target):.1f}" y2="{Y(target):.1f}" style="stroke:var(--ea-loss);stroke-width:1.5;stroke-dasharray:5 4"/>'
             f'<text x="{W - RIGHT}" y="{Y(target) + 16:.1f}" style="font-size:12px;fill:var(--ea-loss-text);text-anchor:end">{html.escape(labels["counted"])}</text>')
    p.append(f'<text x="{X(0) + 8:.1f}" y="{Y(losses[0]) + 4:.1f}" style="font-size:12px;fill:var(--ea-loss-text)">{html.escape(labels["net"])}</text>')
    H = TOP + PH + BOTTOM
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["alt"])}</title>{"".join(p)}</svg></figure>')


# The lesson is done when these are correct. "one-hot-row" is the choice question in the lesson.
REQUIRED = ["one-hot-row", "loss-of", "train"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


class _Shown(torch.Tensor):
    """A tensor that a check message writes the way the lesson's code does: torch.tensor([...]). The learner's code
    gets a plain copy."""

    def __repr__(self):
        return f"torch.tensor({self.as_subclass(torch.Tensor).tolist()!r})"

    def __deepcopy__(self, memo):
        return self.as_subclass(torch.Tensor).clone()


def _shown_tensor(values):
    return torch.tensor(values).as_subclass(_Shown)


def _value(v):
    """A loss for comparing and for the messages: a plain number."""
    try:
        return float(v)
    except (TypeError, ValueError, RuntimeError):
        return v


def _close(got, want) -> bool:
    try:
        return math.isclose(float(got), float(want), rel_tol=1e-4, abs_tol=1e-6)
    except (TypeError, ValueError, RuntimeError):
        return False


def _num(v):
    """A loss for the messages, rounded to 4 decimals; a loss that ran away, as words, so that no e-notation shows."""
    try:
        x = float(v)
    except (TypeError, ValueError, RuntimeError):
        return repr(v)
    if not math.isfinite(x) or abs(x) >= 1e6:
        return {"en": "a number that ran away to infinity", "nl": "een getal dat naar oneindig is weggelopen"}
    return repr(round(x, 4) + 0.0)


_SAVED = {
    "correct": {"en": "Your function is right in all {n} cases. The check saved it in your project, as `{saved}`: the demo and later lessons import it from there.",
                "nl": "Je functie klopt in alle {n} gevallen. De check heeft hem in je project bewaard, als `{saved}`: de demo en latere lessen halen hem daar binnen."},
    "kept": {"en": "Your function is right in all {n} cases. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
             "nl": "Je functie klopt in alle {n} gevallen. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren? Verwijder het bestand en laat opnieuw nakijken."},
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# --- exercise 2: loss_of ------------------------------------------------------------------------------------------------

_L_HEAD = ("import torch\nimport torch.nn.functional as F\n\n\ndef loss_of(W, x, y):\n"
           "    X = torch.eye(len(W))[x]   # one one-hot row per pair\n")
_L_END = "    return F.cross_entropy(logits, y)\n"
_L_START = _L_HEAD + "    logits = ...\n" + _L_END


def _l(logits: str = "X @ W") -> str:
    return _L_HEAD + f"    logits = {logits}\n" + _L_END


def _loss_cases():
    W3 = _shown_tensor([[0.0, 1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.5, 1.0]])
    W2 = _shown_tensor([[1.0, -1.0], [0.0, 3.0]])
    visible = [(W3, _shown_tensor([0, 1, 2, 0]), _shown_tensor([1, 0, 2, 2]))]
    hidden = [(W3, _shown_tensor([2, 2]), _shown_tensor([1, 2])), (W2, _shown_tensor([0, 1, 1]), _shown_tensor([0, 1, 0])),
              (W2, _shown_tensor([1]), _shown_tensor([1]))]
    return visible, hidden


# --- exercise 3: train --------------------------------------------------------------------------------------------------

_T_HEAD = ("import torch\nimport torch.nn.functional as F\n\n\ndef train(x, y, steps, lr, lam):\n"
           "    x, y = torch.tensor(x), torch.tensor(y)\n    W = torch.zeros((65, 65), requires_grad=True)\n"
           "    opt = torch.optim.SGD([W], lr=lr)\n    for step in range(steps):\n")
_T_END = "    return W\n"
_T_START = _T_HEAD + "        ...\n" + _T_END
_T_BODY = ("        loss = F.cross_entropy(W[x], y) + lam * (W ** 2).mean()\n        opt.zero_grad()\n"
           "        loss.backward()\n        opt.step()\n")


def _t(body: str = _T_BODY) -> str:
    return _T_HEAD + body + _T_END


_SHORT = "the cat sat on the mat"


def _short_pairs():
    cs = chars()
    codes = [cs.index(c) for c in _SHORT]
    return codes[:-1], codes[1:]


def _fingerprint(W):
    """What the check compares of a trained table: its average NLL on the pairs of the short text, a plain number."""
    x, y = _short_pairs()
    try:
        W = torch.as_tensor(W).detach().float()
        return float(F.cross_entropy(W[torch.tensor(x)], torch.tensor(y)))
    except Exception:
        return W


def _train_cases():
    x, y = _short_pairs()
    return [(x, y, 20, 10, 0)], [(x, y, 5, 1, 0), (x, y, 20, 10, 0.05), (x, y, 20, 10, 0.5), (x, y, 10, 30, 2)]


_LV, _LH = _loss_cases()
_TV, _TH = _train_cases()

EXERCISES = {
    "loss-of": FunctionBox(
        name="loss-of",
        function="loss_of",
        start=_L_START,
        solution=_l(),
        visible=_LV,
        hidden=_LH,
        collect=_value,
        compare=_close,
        show=_num,
        save_as=("llm", "bigram_net"),
        first=[_DOTS],
        mistakes=[
            Mistake("transposed", wrong=_l("X @ W.T"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you take the columns of `W` instead of its rows. The one-hot row of a character picks its row: `X @ W`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je neemt de kolommen van `W` in plaats van de rijen. De one-hot-rij van een teken kiest zijn rij: `X @ W`."),
            Mistake("order", error=lambda e: isinstance(e, RuntimeError) and "shapes cannot be multiplied" in str(e),
                    en="With `{call}`, PyTorch stops: the shapes do not fit together. The `X` of the first line has one row per pair and one column for each row of `W`, so `X @ W` fits: the one-hot rows first, then the table.",
                    nl="Met `{call}` stopt PyTorch: de shapes passen niet op elkaar. De `X` van de eerste regel heeft één rij per paar en één kolom voor elke rij van `W`, dus `X @ W` past: eerst de one-hot-rijen, dan de tabel."),
        ],
        texts={
            **_SAVED,
            "empty": {"en": "Replace the `...` with the logits of every pair: the one-hot rows `X` times the table `W`.",
                      "nl": "Vervang de `...` door de logits van elk paar: de one-hot-rijen `X` maal de tabel `W`."},
            "arguments": {"en": "The check calls `{call}`, with a table and two tensors of character numbers, and the `def` line of your function has another number of names between its parentheses. Keep `(W, x, y)`.",
                          "nl": "De check roept `{call}` aan, met een tabel en twee tensors met tekennummers, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(W, x, y)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Are your logits the rows of `W`, one for each pair, picked by the one-hot rows?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Zijn je logits de rijen van `W`, één per paar, gekozen door de one-hot-rijen?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for a table of any size?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor een tabel van elke grootte?"},
        },
    ),
    "train": FunctionBox(
        name="train",
        function="train",
        start=_T_START,
        solution=_t(),
        visible=_TV,
        hidden=_TH,
        collect=_fingerprint,
        compare=_close,
        show=_num,
        save_as=("llm", "bigram_train"),
        first=[_DOTS],
        mistakes=[
            Mistake("no-step", wrong=_t("        loss = F.cross_entropy(W[x], y) + lam * (W ** 2).mean()\n        opt.zero_grad()\n        loss.backward()\n"),
                    en="With `{call}`, your table did not change: its NLL on these pairs is still `{got}`, the log of 65, as at the start, so no step moved `W`. `opt.step()` moves it along the slopes that `loss.backward()` has just worked out: in the loop, `opt.zero_grad()`, then `loss.backward()`, then `opt.step()`.",
                    nl="Met `{call}` is je tabel niet veranderd: de NLL op deze paren is nog `{got}`, de log van 65, zoals aan het begin, dus geen enkele stap heeft `W` verschoven. `opt.step()` verschuift hem langs de hellingen die `loss.backward()` net heeft uitgerekend: in de loop eerst `opt.zero_grad()`, dan `loss.backward()`, dan `opt.step()`."),
            Mistake("no-zero", wrong=_t("        loss = F.cross_entropy(W[x], y) + lam * (W ** 2).mean()\n        loss.backward()\n        opt.step()\n"),
                    en="With `{call}`, your table gets an NLL of `{got}` on these pairs, and the course's `{want}`. Without `opt.zero_grad()`, the slopes of every step add up to those of the steps before. Clear them before `loss.backward()`.",
                    nl="Met `{call}` krijgt je tabel een NLL van `{got}` op deze paren, en die van de course `{want}`. Zonder `opt.zero_grad()` tellen de hellingen van elke stap op bij die van de stappen ervoor. Wis ze vóór `loss.backward()`."),
            Mistake("no-penalty", wrong=_t("        loss = F.cross_entropy(W[x], y)\n        opt.zero_grad()\n        loss.backward()\n        opt.step()\n"),
                    en="With `{call}`, your table gets an NLL of `{got}` on these pairs, and the course's `{want}`. That is what a table without a penalty gets: add `lam * (W ** 2).mean()` to the loss, so that `lam` does something.",
                    nl="Met `{call}` krijgt je tabel een NLL van `{got}` op deze paren, en die van de course `{want}`. Dat krijgt een tabel zonder straf: tel `lam * (W ** 2).mean()` bij de loss op, zodat `lam` iets doet."),
            Mistake("sum-penalty", wrong=_t("        loss = F.cross_entropy(W[x], y) + lam * (W ** 2).sum()\n        opt.zero_grad()\n        loss.backward()\n        opt.step()\n"),
                    en="With `{call}`, your table gets an NLL of `{got}` on these pairs, and the course's `{want}`. Your penalty adds up the squares of all 4,225 weights, as in *Regularization and weight decay*; the recipe of this lesson takes their mean: `(W ** 2).mean()`.",
                    nl="Met `{call}` krijgt je tabel een NLL van `{got}` op deze paren, en die van de course `{want}`. Je straf telt de kwadraten van alle 4.225 weights op, zoals in *Regularisatie en weight decay*; het recept van deze les neemt hun gemiddelde: `(W ** 2).mean()`."),
        ],
        texts={
            **_SAVED,
            "empty": {"en": "Replace the `...` with step 3 of the recipe: the loss with its penalty, then zero, backward and step.",
                      "nl": "Vervang de `...` door stap 3 van het recept: de loss met de straf, dan op nul zetten, backward en de stap."},
            "arguments": {"en": "The check calls `{call}`, and the `def` line of your function has another number of names between its parentheses. Keep `(x, y, steps, lr, lam)`.",
                          "nl": "De check roept `{call}` aan, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(x, y, steps, lr, lam)`."},
            "wrong": {"en": "With `{call}`, your table gets an NLL of `{got}` on these pairs, and the course's `{want}`. Compare the four lines in your loop with step 3 of the recipe.",
                      "nl": "Met `{call}` krijgt je tabel een NLL van `{got}` op deze paren, en die van de course `{want}`. Vergelijk de vier regels in je loop met stap 3 van het recept."},
            "almost": {"en": "Right for the example. But with `{call}`, your table gets an NLL of `{got}`, and the course's `{want}`. Does your loop also work when `lam` is not 0?",
                       "nl": "Goed voor het voorbeeld. Maar met `{call}` krijgt je tabel een NLL van `{got}`, en die van de course `{want}`. Werkt je loop ook als `lam` geen 0 is?"},
        },
    ),
}




# --- the demo: the learner's training on the start of the plays, and the learned table writing ---------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "llm" / f"{module}.py").exists()


def demo():
    """For the demo: None when bigram_net.py or bigram_train.py is not in the project yet; "train" when the saved
    train no longer gives what the course's gives on the short text, or stops with an error; "net" when the saved
    loss_of does not give the NLL of the learned table, or stops with an error; "sampling" when generate stops with an
    error; "slow" when training takes more than 30 seconds; else a dict with the NLL of the learned table (worked out
    with the learner's loss_of), the probability of "h" after "t", a text it wrote, and whether generate was the
    learner's own."""
    if not (_saved("bigram_net") and _saved("bigram_train")):
        return None
    run = run_code("from llm.bigram_train import train")
    if run.error is not None:
        return "train"
    train = run.names["train"]
    x, y = _short_pairs()
    try:
        if not _close(_fingerprint(with_time_limit(lambda: train(x, y, 20, 10, 0), 10)), _fingerprint(_course_train(x, y, 20, 10, 0))):
            return "train"
    except Exception:
        return "train"
    sx, sy = start_pairs()
    try:
        W = with_time_limit(lambda: train(sx, sy, STEPS, LR, 0.0), 30)
    except Timeout:
        return "slow"
    except Exception:
        return "train"
    W = torch.as_tensor(W).detach().float()
    run = run_code("from llm.bigram_net import loss_of")
    if run.error is not None:
        return "net"
    loss_of = run.names["loss_of"]
    tx, ty = torch.tensor(sx), torch.tensor(sy)
    try:
        nll = float(with_time_limit(lambda: loss_of(W, tx, ty), 10))
    except Exception:
        return "net"
    if not _close(nll, F.cross_entropy(W[tx], ty)):
        return "net"
    probs = torch.softmax(W, dim=1).numpy().astype(float)
    probs = probs / probs.sum(axis=1, keepdims=True)
    cs = list(chars())
    run = run_code("from llm.sampling import generate")   # the learner's, or the course's when theirs is missing or broken
    if run.error is not None:
        return "sampling"
    yours = all(module != "sampling" for _, module, _, _ in projects.stand_ins())
    try:
        written = with_time_limit(lambda: run.names["generate"](probs, cs, "T", 200, 1), 10)
    except Exception:
        return "sampling"
    return {"nll": nll, "th": float(probs[cs.index("t"), cs.index("h")]), "text": written, "yours": yours}


# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("loss-of", _L_START, "empty", None),
    ("loss-of", _l(), "correct", None),
    ("loss-of", _l("W[x]"), "correct", None),
    ("loss-of", _l("X @ W.T"), "wrong", "transposed"),
    ("loss-of", _l("W @ X"), "error", "order"),
    ("loss-of", _l("torch.eye(65)[x] @ W"), "error", "order"),
    ("train", _T_START, "empty", None),
    ("train", _t(), "correct", None),
    ("train", _t(_T_BODY.replace("F.cross_entropy(W[x], y)", "F.cross_entropy(torch.eye(65)[x] @ W, y)")), "correct", None),
    ("train", _t("        loss = F.cross_entropy(W[x], y) + lam * (W ** 2).mean()\n        opt.zero_grad()\n        loss.backward()\n"), "wrong", "no-step"),
    ("train", _t("        loss = F.cross_entropy(W[x], y) + lam * (W ** 2).mean()\n        loss.backward()\n        opt.zero_grad()\n        opt.step()\n"), "wrong", "no-step"),
    ("train", _t("        loss = F.cross_entropy(W[x], y) + lam * (W ** 2).mean()\n        loss.backward()\n        opt.step()\n        opt.zero_grad()\n"), "correct", None),
    ("train", _t(_T_BODY.replace("(W ** 2).mean()", "(W * W).mean()")), "correct", None),
    ("train", _t("        loss = F.cross_entropy(W[x], y) + lam * (W ** 2).mean()\n        loss.backward()\n        opt.step()\n"), "wrong", "no-zero"),
    ("train", _t("        loss = F.cross_entropy(W[x], y)\n        opt.zero_grad()\n        loss.backward()\n        opt.step()\n"), "wrong", "no-penalty"),
    ("train", _t("        loss = F.cross_entropy(W[x], y) + lam * (W ** 2).sum()\n        opt.zero_grad()\n        loss.backward()\n        opt.step()\n"), "wrong", "sum-penalty"),
]
