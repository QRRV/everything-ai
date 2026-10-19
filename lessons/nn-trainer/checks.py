# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the project lesson nn-trainer: train_epoch, one epoch with gradient accumulation, clipping and a schedule,
saved as work/projects/nn/train_epoch.py; and Trainer, which trains with it epoch after epoch, evaluates, keeps a log
and saves the best parameters, saved as work/projects/nn/trainer.py. And the picture: the log of the Trainer on the
MLP for the handwritten digits of QMNIST."""

import functools
import re
import traceback

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from everything_ai import paths, projects
from everything_ai.check import ClassBox, Mistake

# Trainer imports train_epoch from the learner's project; when that file is missing or wrong, the course's own
# version stands in, and the lesson says so.
projects.builds_on(("nn", "train_epoch", "nn-trainer", "train-epoch"))

DATA = "data/qmnist-small.npz"   # the first 6,000 training and 1,000 test images of QMNIST: see data/README.md
EPOCHS, LR, WARMUP_EPOCHS, HIDDEN = 8, 0.003, 1, 64

# --- exercise 2: one epoch -------------------------------------------------------------------------------------------------

_E_HEAD = ("import torch\nimport torch.nn.functional as F\n\n\ndef train_epoch(model, loader, opt, sched, max_norm, accum):\n"
           "    model.train()\n    total = 0.0\n    k = 0\n    opt.zero_grad()\n    for xb, yb in loader:\n        logits = model(xb)\n"
           "        # one loss for every prediction, also when logits has a time axis, as in a language model\n"
           "        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))\n        total = total + loss.item()\n"
           "        k = k + 1\n")
_E_TAIL = "    return total / len(loader)\n"
_E_START = (_E_HEAD + "        # backward with loss / accum; after every accum mini-batches, and after the last one:\n"
            "        # clip when max_norm is not None, step, zero the gradients, and step the schedule when sched is not None\n"
            "        ...\n" + _E_TAIL)
_BACKWARD = "        (loss / accum).backward()\n"
_WHEN = "k % accum == 0 or k == len(loader)"
_CLIP = "            if max_norm is not None:\n                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)\n"
_STEP = "            opt.step()\n            opt.zero_grad()\n"
_SCHED = "            if sched is not None:\n                sched.step()\n"


def _e(backward: str = _BACKWARD, when: str = _WHEN, clip: str = _CLIP, step: str = _STEP, sched: str = _SCHED, extra: str = "") -> str:
    return _E_HEAD + extra + backward + f"        if {when}:\n" + clip + step + sched + _E_TAIL


_DATA = ("import torch\nfrom torch import nn\nfrom torch.utils.data import DataLoader, TensorDataset\n\n"
         "torch.manual_seed(0)\nX = torch.randn(40, 4)\ny = (X[:, 0] > 0).long() + (X[:, 1] > 0).long()\nmodel = nn.Linear(4, 3)\n"
         "loader = DataLoader(TensorDataset(X, y), batch_size=8)\nopt = torch.optim.SGD(model.parameters(), lr=0.5)\n")
_E_SCENARIOS = [
    # 5 mini-batches, a step after every 2 and after the last
    _DATA + "loss = train_epoch(model, loader, opt, None, None, 2)\n[round(loss, 4), model.weight.tolist()]",
    # clipping and a schedule that halves the learning rate at every step
    _DATA + "sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)\nloss = train_epoch(model, loader, opt, sched, 0.1, 2)\n"
            "[round(loss, 4), model.bias.tolist(), opt.param_groups[0][\"lr\"]]",
    # logits with a time axis, as a language model gives them: shape (4, 3, 5)
    "import torch\nfrom torch import nn\nfrom torch.utils.data import DataLoader, TensorDataset\n\ntorch.manual_seed(1)\n"
    "ids = torch.randint(0, 5, (8, 3))\nnext_ids = torch.randint(0, 5, (8, 3))\nmodel = nn.Embedding(5, 5)\n"
    "loader = DataLoader(TensorDataset(ids, next_ids), batch_size=4)\nopt = torch.optim.SGD(model.parameters(), lr=1.0)\n"
    "loss = train_epoch(model, loader, opt, None, 1.0, 1)\n[round(loss, 4), model.weight[0].tolist()]",
]

# --- exercise 3: the Trainer -----------------------------------------------------------------------------------------------

_T_HEAD = ("import math\nimport torch\nimport torch.nn.functional as F\nfrom nn.train_epoch import train_epoch\n\n\n"
           "def lr_factor(step, warmup, total):\n    \"\"\"Warmup and cosine decay, as a factor on the peak learning rate.\"\"\"\n"
           "    if step < warmup:\n        return (step + 1) / warmup\n    if step >= total:\n        return 0.0\n"
           "    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))\n\n\n"
           "class Trainer:\n    def __init__(self, model, train_loader, val_loader, epochs, lr,\n                 warmup=0, max_norm=1.0, accum=1, path=None):\n"
           "        self.model = model\n        self.train_loader = train_loader\n        self.val_loader = val_loader\n"
           "        self.epochs = epochs\n        self.max_norm = max_norm\n        self.accum = accum\n        self.path = path\n"
           "        self.warmup = warmup\n        # the number of steps of the optimizer; math.ceil rounds up\n        self.total = epochs * math.ceil(len(train_loader) / accum)\n"
           "        self.opt = torch.optim.AdamW(model.parameters(), lr=lr)\n"
           "        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, self.factor)\n        self.history = []\n\n"
           "    def factor(self, step):\n        return lr_factor(step, self.warmup, self.total)\n\n")
_EVALUATE = ("    def evaluate(self):\n        self.model.eval()\n        total = 0.0\n        with torch.no_grad():\n"
             "            for xb, yb in self.val_loader:\n                logits = self.model(xb)\n"
             "                total = total + F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1)).item()\n"
             "        self.model.train()\n        return total / len(self.val_loader)\n\n")
_FIT_HEAD = "    def fit(self):\n        best = math.inf\n        for epoch in range(self.epochs):\n"
_FIT_BODY = ("            train_loss = train_epoch(self.model, self.train_loader, self.opt,\n"
             "                                     self.sched, self.max_norm, self.accum)\n"
             "            val_loss = self.evaluate()\n"
             "            lr = self.opt.param_groups[0][\"lr\"]\n"
             "            self.history.append({\"epoch\": epoch + 1, \"train_loss\": train_loss,\n"
             "                                 \"val_loss\": val_loss, \"lr\": lr})\n")
_SAVE = ("            if val_loss < best:\n                best = val_loss\n                if self.path is not None:\n"
         "                    torch.save(self.model.state_dict(), self.path)\n")
_FIT_TAIL = "        return self.history\n"
_T_START = (_T_HEAD + "    def evaluate(self):\n        # the mean loss on val_loader, in eval mode and without a graph; back in train mode at the end\n        ...\n\n"
            + _FIT_HEAD + "            # train_epoch, evaluate, a dict in self.history with epoch, train_loss, val_loss and lr,\n"
            "            # and the parameters saved to self.path when the validation loss is the lowest so far\n            ...\n" + _FIT_TAIL)


def _t(evaluate: str = _EVALUATE, body: str = _FIT_BODY, save: str = _SAVE) -> str:
    return _T_HEAD + evaluate + _FIT_HEAD + body + save + _FIT_TAIL


_T_DATA = ("import os\nimport tempfile\nimport torch\nfrom torch import nn\nfrom torch.utils.data import DataLoader, TensorDataset\n\n"
           "torch.manual_seed(0)\nX = torch.randn(48, 4)\ny = (X[:, 0] + 0.5 * torch.randn(48) > 0).long() + (X[:, 1] > 0).long()\n"
           "train = DataLoader(TensorDataset(X[:32], y[:32]), batch_size=8)\nval = DataLoader(TensorDataset(X[32:], y[32:]), batch_size=8)\n")
_T_SCENARIOS = [
    # the log
    _T_DATA + "model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))\n"
              "trainer = Trainer(model, train, val, epochs=4, lr=0.05, warmup=2, max_norm=1.0, accum=2)\nhistory = trainer.fit()\n"
              "[[h[\"epoch\"], round(h[\"train_loss\"], 4), round(h[\"val_loss\"], 4), round(h[\"lr\"], 5)] for h in history]",
    # the checkpoint: is the saved model the one with the lowest validation loss?
    _T_DATA + "model = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 3))\npath = os.path.join(tempfile.mkdtemp(), \"best.pt\")\n"
              "trainer = Trainer(model, train, val, epochs=12, lr=0.2, path=path)\nhistory = trainer.fit()\n"
              "model.load_state_dict(torch.load(path))\n"
              "[round(min(h[\"val_loss\"] for h in history), 4), round(history[-1][\"val_loss\"], 4), round(trainer.evaluate(), 4)]",
    # with dropout: does evaluate measure in eval mode, and leave the model in train mode?
    _T_DATA + "model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Dropout(0.5), nn.Linear(16, 3))\n"
              "trainer = Trainer(model, train, val, epochs=1, lr=0.01)\na = trainer.evaluate()\nb = trainer.evaluate()\n"
              "[a == b, model.training]",
]

# --- the picture: the Trainer on the digits -------------------------------------------------------------------------------


class _DigitsMLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.hidden = nn.Linear(784, hidden)
        self.out = nn.Linear(hidden, 10)

    def forward(self, x):
        return self.out(torch.relu(self.hidden(x)))


def _course_trainer():
    """The course's Trainer and train_epoch, from the solutions of exercises 2 and 3."""
    names: dict = {}
    exec(compile(_e(), "<train_epoch>", "exec"), names)
    trainer = _t().replace("from nn.train_epoch import train_epoch\n", "")
    exec(compile(trainer, "<trainer>", "exec"), names)
    return names["Trainer"]


@functools.lru_cache(maxsize=1)
def digits_history():
    """The Trainer on the MLP of torch-mlp-digits: 8 epochs of AdamW with warmup and cosine decay, clipping at 1."""
    d = np.load(paths.ROOT / DATA)
    X = torch.tensor(d["train_images"], dtype=torch.float32) / 255
    y = torch.tensor(d["train_labels"], dtype=torch.int64)
    X_test = torch.tensor(d["test_images"], dtype=torch.float32) / 255
    y_test = torch.tensor(d["test_labels"], dtype=torch.int64)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)   # the same numbers on every computer
    try:
        torch.manual_seed(0)
        model = _DigitsMLP(HIDDEN)
        torch.manual_seed(1)
        train = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
        val = DataLoader(TensorDataset(X_test, y_test), batch_size=500)
        trainer = _course_trainer()(model, train, val, epochs=EPOCHS, lr=LR, warmup=WARMUP_EPOCHS * len(train), max_norm=1.0)
        history = trainer.fit()
        with torch.no_grad():
            right = int((model(X_test).argmax(dim=1) == y_test).sum())
    finally:
        torch.set_num_threads(threads)
    return history, right


def history_picture(labels: dict) -> str:
    """Left: the training and validation loss after every epoch. Right: the learning rate at the end of every epoch."""
    from everything_ai import chart
    history, _ = digits_history()
    epochs = [h["epoch"] for h in history]
    left = chart.Panel((0.5, EPOCHS + 0.5), (0, 1.4), title=labels["loss_title"], xlabel=labels["x"], ylabel=labels["loss"])
    left.line(epochs, [h["train_loss"] for h in history], "loss", 2.4)
    left.points(epochs, [h["train_loss"] for h in history], "loss")
    left.line(epochs, [h["val_loss"] for h in history], "loss", 1.6, dashed=True)
    left.points(epochs, [h["val_loss"] for h in history], "loss", hollow=True)
    left.text(EPOCHS, history[-1]["train_loss"] - 0.06, labels["train"], "loss", "end")
    left.text(EPOCHS, history[-1]["val_loss"] + 0.13, labels["val"], "loss", "end")
    # epoch 0: the learning rate of the first step, 1 / 94 of the peak (a warmup of one epoch, 94 mini-batches of 64)
    starts = [0] + epochs
    lrs = [LR / 94] + [h["lr"] for h in history]
    right = chart.Panel((-0.5, EPOCHS + 0.5), (0, LR * 1.1), title=labels["lr_title"], xlabel=labels["x"], ylabel=labels["lr"])
    right.line(starts, lrs, "hyper", 2.4)
    right.points(starts, lrs, "hyper")
    return chart.figure([left, right], alt=labels["alt"], width=330, height=270)


# The lesson is done when these are correct. "why-divide" is the choice question in the lesson.
REQUIRED = ["why-divide", "train-epoch", "trainer"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _in_frames(e: BaseException, name: str) -> bool:
    """The error happened inside a function of that name, such as `clip_grad_norm_` or `save` of torch."""
    return any(f.name == name for f in traceback.extract_tb(e.__traceback__))


# Errors that the plain words of the app would tell wrongly: a `None` the check passes on purpose, a file never saved,
# a key of the log the check cannot find.
_NO_IF_CLIP = Mistake("no-if-clip", error=lambda e: isinstance(e, TypeError) and "NoneType" in str(e) and _in_frames(e, "clip_grad_norm_"),
                      en="The first try is `train_epoch(model, loader, opt, None, None, 2)`: `max_norm` is `None`, so no clipping, and `clip_grad_norm_` cannot clip at `None`. Clip only `if max_norm is not None:`.",
                      nl="De eerste poging is `train_epoch(model, loader, opt, None, None, 2)`: `max_norm` is `None`, dus niet clippen, en `clip_grad_norm_` kan niet clippen op `None`. Clip alleen `if max_norm is not None:`.")
_NO_IF_SCHED = Mistake("no-if-sched", error=lambda e: isinstance(e, AttributeError) and "'NoneType' object has no attribute 'step'" in str(e),
                       en="The first try is `train_epoch(model, loader, opt, None, None, 2)`: `sched` is `None`, so no schedule, and `None` has no `step()`. Step the schedule only `if sched is not None:`.",
                       nl="De eerste poging is `train_epoch(model, loader, opt, None, None, 2)`: `sched` is `None`, dus geen schedule, en `None` heeft geen `step()`. Laat de schedule alleen een stap zetten `if sched is not None:`.")
_PATH_NONE = Mistake("path-none", error=lambda e: isinstance(e, AttributeError) and _in_frames(e, "save"),
                     en="In the first try, the Trainer gets no `path`, so `self.path` is `None`, and `torch.save` cannot save to `None`. Save only when `self.path is not None`.",
                     nl="In de eerste poging krijgt de Trainer geen `path`, dus `self.path` is `None`, en `torch.save` kan niet opslaan naar `None`. Sla alleen op als `self.path is not None`.")
_LOG_KEY = Mistake("log-key", error=lambda e: isinstance(e, KeyError) and bool(e.args) and e.args[0] in ("epoch", "train_loss", "val_loss", "lr"),
                   en="The check reads `\"epoch\"`, `\"train_loss\"`, `\"val_loss\"` and `\"lr\"` from every dict in `self.history`, and one of them is not in yours. Check these four keys, spelling included.",
                   nl="De check leest `\"epoch\"`, `\"train_loss\"`, `\"val_loss\"` en `\"lr\"` uit elke dict in `self.history`, en een daarvan staat niet in de jouwe. Kijk die vier keys na, ook de spelling.")
_NO_FILE = Mistake("no-file", error=lambda e: isinstance(e, FileNotFoundError) and str(getattr(e, "filename", "") or "").endswith("best.pt"),
                   en="The check gives the Trainer a `path`, and after `fit()` there is no file there: your `fit` never saved the parameters to `self.path`. Save them when `val_loss` is lower than `best`, and let `best` start above any loss, so that the first epoch already saves.",
                   nl="De check geeft de Trainer een `path`, en na `fit()` staat daar geen bestand: je `fit` heeft de parameters nooit naar `self.path` opgeslagen. Sla ze op als `val_loss` lager is dan `best`, en laat `best` boven elke loss beginnen, zodat de eerste epoch al opslaat.")
_SCENARIO = "With these lines after your {what}:\n\n```python\n{{scenario}}\n```\n\nyou get `{{got}}`, and it should be `{{want}}`"
_SCENARIO_NL = "Met deze regels na je {what}:\n\n```python\n{{scenario}}\n```\n\nkrijg je `{{got}}`, en dat moet `{{want}}` zijn"
_F_EN, _F_NL = _SCENARIO.format(what="function"), _SCENARIO_NL.format(what="functie")
_C_EN, _C_NL = _SCENARIO.format(what="class"), _SCENARIO_NL.format(what="class")


def _close(got, want) -> bool:
    try:
        g, w = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    except (TypeError, ValueError):
        return got == want
    return g.shape == w.shape and bool(np.allclose(g, w, rtol=1e-4, atol=1e-6))


def _close_nested(got, want) -> bool:
    """Lists that hold numbers and lists of numbers, or bools: equal up to float32 noise."""
    if isinstance(want, bool) or isinstance(got, bool):
        return got == want
    if isinstance(want, (list, tuple)):
        return isinstance(got, (list, tuple)) and len(got) == len(want) and all(_close_nested(g, w) for g, w in zip(got, want))
    return _close(got, want)


EXERCISES = {
    "train-epoch": ClassBox(
        name="train-epoch",
        start=_E_START,
        solution=_e(),
        scenarios=_E_SCENARIOS,
        compare=_close_nested,
        digits=4,
        save_as=("nn", "train_epoch"),
        first=[_DOTS],
        mistakes=[
            _NO_IF_CLIP,
            _NO_IF_SCHED,
            Mistake("no-divide", wrong=_e(backward="        loss.backward()\n"),
                    en=_F_EN + ": your steps are `accum` times too large. Each mini-batch adds its gradient to the others, so divide its loss by `accum` before `backward()`: `(loss / accum).backward()`.",
                    nl=_F_NL + ": je stappen zijn `accum` keer te groot. Elke mini-batch telt zijn gradient bij de andere op, dus deel zijn loss door `accum` vóór `backward()`: `(loss / accum).backward()`."),
            Mistake("every-batch", wrong=_e(when="True"),
                    en=_F_EN + ": you take a step after every mini-batch, so nothing is accumulated. Step only when `k % accum == 0`, and after the last mini-batch.",
                    nl=_F_NL + ": je zet na elke mini-batch een stap, dus er wordt niets opgespaard. Zet alleen een stap als `k % accum == 0`, en na de laatste mini-batch."),
            Mistake("no-last", wrong=_e(when="k % accum == 0"),
                    en=_F_EN + ": the gradients of the last mini-batches are lost when their number does not divide by `accum`. Also take a step after the last one: `or k == len(loader)`.",
                    nl=_F_NL + ": de gradients van de laatste mini-batches gaan verloren als hun aantal niet deelbaar is door `accum`. Zet ook een stap na de laatste: `or k == len(loader)`."),
            Mistake("zero-each", wrong=_e(extra="        opt.zero_grad()\n"),
                    en=_F_EN + ": you set the gradients to zero for every mini-batch, so each step only has the gradient of the last one. Set them to zero only after a step.",
                    nl=_F_NL + ": je zet de gradients bij elke mini-batch op nul, dus elke stap heeft alleen de gradient van de laatste. Zet ze alleen na een stap op nul."),
            Mistake("sched-each", wrong=_e(sched="", extra="        if sched is not None:\n            sched.step()\n"),
                    en=_F_EN + ": your schedule steps after every mini-batch, also those without a step of the optimizer. Step it with the optimizer, once per step.",
                    nl=_F_NL + ": je schedule zet na elke mini-batch een stap, ook na die zonder stap van de optimizer. Laat hem met de optimizer meelopen, één keer per stap."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: `backward()` of the loss divided by `accum`, and after every `accum` mini-batches and after the last one, the clipping, the step, zeroing the gradients and the step of the schedule.",
                      "nl": "Vervang de `...`: `backward()` van de loss gedeeld door `accum`, en na elke `accum` mini-batches en na de laatste het clippen, de stap, de gradients op nul en de stap van de schedule."},
            "wrong": {"en": _F_EN + ": the mean loss and the parameters after one epoch. Do you divide the loss by `accum`, step after every `accum` mini-batches and after the last one, and only then set the gradients to zero?",
                      "nl": _F_NL + ": de gemiddelde loss en de parameters na één epoch. Deel je de loss door `accum`, zet je een stap na elke `accum` mini-batches en na de laatste, en zet je pas daarna de gradients op nul?"},
            "almost": {"en": "Right without clipping and a schedule. But " + _F_EN[0].lower() + _F_EN[1:] + ". Do you clip when `max_norm` is not `None`, before the step, and step the schedule after each step of the optimizer?",
                       "nl": "Goed zonder clippen en schedule. Maar " + _F_NL[0].lower() + _F_NL[1:] + ". Clip je als `max_norm` niet `None` is, vóór de stap, en laat je de schedule na elke stap van de optimizer een stap zetten?"},
            "correct": {"en": "One epoch with gradient accumulation, clipping and a schedule, also for logits with a time axis. Your `train_epoch` is now in your project, as `work/projects/nn/train_epoch.py`.",
                        "nl": "Eén epoch met gradient accumulation, clippen en een schedule, ook voor logits met een tijdas. Je `train_epoch` staat nu in je project, als `work/projects/nn/train_epoch.py`."},
            "kept": {"en": "One epoch with gradient accumulation, clipping and a schedule. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Eén epoch met gradient accumulation, clippen en een schedule. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "trainer": ClassBox(
        name="trainer",
        start=_T_START,
        solution=_t(),
        scenarios=_T_SCENARIOS,
        compare=_close_nested,
        digits=4,
        save_as=("nn", "trainer"),
        first=[_DOTS],
        mistakes=[
            _PATH_NONE,
            _LOG_KEY,
            _NO_FILE,
            Mistake("every-epoch", wrong=_t(save="            if self.path is not None:\n                torch.save(self.model.state_dict(), self.path)\n"),
                    en=_C_EN + ": the saved parameters are those of the last epoch, not of the one with the lowest validation loss. Save only when `val_loss` is lower than the best so far, and keep that best.",
                    nl=_C_NL + ": de bewaarde parameters zijn die van de laatste epoch, niet die van de epoch met de laagste validation loss. Bewaar alleen als `val_loss` lager is dan de beste tot nu toe, en houd die beste bij."),
            Mistake("no-eval", wrong=_t(evaluate=_EVALUATE.replace("        self.model.eval()\n", "")),
                    en=_C_EN + ": two evaluations give two different losses, because dropout is still on. Measure in eval mode: `self.model.eval()` before the loop.",
                    nl=_C_NL + ": twee keer evalueren geeft twee verschillende losses, omdat dropout nog aanstaat. Meet in eval mode: `self.model.eval()` vóór de loop."),
            Mistake("stays-eval", wrong=_t(evaluate=_EVALUATE.replace("        self.model.train()\n", "")),
                    en=_C_EN + ": after `evaluate`, your model stays in eval mode. Put it back with `self.model.train()` at the end, so that whoever calls `evaluate` between epochs does not train on without dropout.",
                    nl=_C_NL + ": na `evaluate` blijft je model in eval mode. Zet het aan het eind terug met `self.model.train()`, zodat wie `evaluate` tussen epochs aanroept, niet zonder dropout verder traint."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: in `evaluate` the mean loss on `val_loader` in eval mode, without a graph; in `fit` for every epoch `train_epoch`, `evaluate`, a dict in `self.history`, and the parameters saved when the validation loss is the lowest so far.",
                      "nl": "Vervang de twee `...`: in `evaluate` de gemiddelde loss op `val_loader` in eval mode, zonder graaf; in `fit` voor elke epoch `train_epoch`, `evaluate`, een dict in `self.history`, en de parameters bewaard als de validation loss de laagste tot nu toe is."},
            "wrong": {"en": _C_EN + ". For every epoch: does `fit` call `train_epoch` with the optimizer, the schedule, `max_norm` and `accum` of the Trainer, then `evaluate`, and log a dict with `epoch`, `train_loss`, `val_loss` and the `lr` of the optimizer?",
                      "nl": _C_NL + ". Voor elke epoch: roept `fit` `train_epoch` aan met de optimizer, de schedule, `max_norm` en `accum` van de Trainer, dan `evaluate`, en bewaart het een dict met `epoch`, `train_loss`, `val_loss` en de `lr` van de optimizer?"},
            "almost": {"en": "The log is right. But " + _C_EN[0].lower() + _C_EN[1:] + ". Do you save the parameters only at a new lowest validation loss, and does `evaluate` measure in eval mode and go back to train mode?",
                       "nl": "Het logboek klopt. Maar " + _C_NL[0].lower() + _C_NL[1:] + ". Bewaar je de parameters alleen bij een nieuwe laagste validation loss, en meet `evaluate` in eval mode en gaat hij daarna terug naar train mode?"},
            "correct": {"en": "Your Trainer trains, evaluates, keeps a log and saves the best parameters. It is now in your project, as `work/projects/nn/trainer.py`: `from nn.trainer import Trainer`.",
                        "nl": "Je Trainer traint, evalueert, houdt een logboek bij en bewaart de beste parameters. Hij staat nu in je project, als `work/projects/nn/trainer.py`: `from nn.trainer import Trainer`."},
            "kept": {"en": "Your Trainer trains, evaluates, keeps a log and saves the best parameters. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je Trainer traint, evalueert, houdt een logboek bij en bewaart de beste parameters. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

CASES = [
    ("train-epoch", _E_START, "empty", None),
    ("train-epoch", _e(), "correct", None),
    ("train-epoch", _e(backward="        loss.backward()\n"), "wrong", "no-divide"),
    ("train-epoch", _e(when="True"), "wrong", "every-batch"),
    ("train-epoch", _e(when="k % accum == 0"), "wrong", "no-last"),
    ("train-epoch", _e(extra="        opt.zero_grad()\n"), "wrong", "zero-each"),
    ("train-epoch", _e(sched="", extra="        if sched is not None:\n            sched.step()\n"), "wrong", "sched-each"),
    ("train-epoch", _e(clip=""), "almost", None),
    ("train-epoch", _e(clip="            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)\n"), "error", "no-if-clip"),
    ("train-epoch", _e(sched="            sched.step()\n"), "error", "no-if-sched"),
    ("trainer", _T_START, "empty", None),
    ("trainer", _t(), "correct", None),
    ("trainer", _t(save="            if self.path is not None:\n                torch.save(self.model.state_dict(), self.path)\n"), "wrong", "every-epoch"),
    ("trainer", _t(evaluate=_EVALUATE.replace("        self.model.eval()\n", "")), "wrong", "no-eval"),
    ("trainer", _t(evaluate=_EVALUATE.replace("        self.model.train()\n", "")), "wrong", "stays-eval"),
    ("trainer", _t(save="            if val_loss < best:\n                best = val_loss\n                torch.save(self.model.state_dict(), self.path)\n"), "error", "path-none"),
    ("trainer", _t(body=_FIT_BODY.replace('"val_loss": val_loss', '"val loss": val_loss')), "error", "log-key"),
    ("trainer", _t().replace("best = math.inf", "best = 0"), "error", "no-file"),
]
