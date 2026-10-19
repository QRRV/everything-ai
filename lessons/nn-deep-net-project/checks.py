# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nn-deep-net-project: overfit_one_batch, the test whether a network can learn one mini-batch
by heart; and DeepMLP, a deep MLP with He initialization, batch normalization and residual connections as options.
The picture trains the same 10 layers five ways on 6,000 images of clothes from Fashion-MNIST."""
import functools
import importlib.util
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from everything_ai import draw, paths, projects
from everything_ai.check import ClassBox, Mistake

DATA = "data/fashion-mnist-small.npz"   # 6,000 training and 1,000 test images of Fashion-MNIST: see data/README.md
DEPTH, WIDTH, EPOCHS = 10, 64, 3
SGD_LR, ADAMW_LR = 0.1, 0.003
OVERFIT_STEPS, OVERFIT_LR = 300, 0.05


@functools.lru_cache(maxsize=1)
def fashion():
    """The training and test images as tensors of 0 to 1, one row of 784 pixels per image, and their labels."""
    d = np.load(paths.ROOT / DATA)
    X = torch.tensor(d["train_images"], dtype=torch.float32) / 255
    y = torch.tensor(d["train_labels"], dtype=torch.int64)
    X_test = torch.tensor(d["test_images"], dtype=torch.float32) / 255
    y_test = torch.tensor(d["test_labels"], dtype=torch.int64)
    return X, y, X_test, y_test


def one_batch():
    """The first 64 training images and their labels: the mini-batch of the overfit test."""
    X, y, _, _ = fashion()
    return X[:64], y[:64]


# --- exercise 2: overfit_one_batch ----------------------------------------------------------------------------------------

_O_HEAD = ("import torch\nimport torch.nn.functional as F\n\n\ndef overfit_one_batch(model, xb, yb, steps, lr):\n"
           "    opt = torch.optim.SGD(model.parameters(), lr=lr)\n    for step in range(steps):\n")
_O_TAIL = "    return loss.item()\n"
_O_START = (_O_HEAD + "        # the loss on the same mini-batch every time,\n        # then the gradients back to 0, backward, and a step of opt\n"
            "        ...\n" + _O_TAIL)


def _o(loss: str = "        loss = F.cross_entropy(model(xb), yb)\n", zero: str = "        opt.zero_grad()\n",
       backward: str = "        loss.backward()\n", step: str = "        opt.step()\n", tail: str = _O_TAIL) -> str:
    return _O_HEAD + loss + zero + backward + step + tail


_O_DATA = ("import torch\nfrom torch import nn\n\ntorch.manual_seed(0)\nxb = torch.randn(16, 5)\nyb = torch.randint(0, 3, (16,))\n"
           "model = nn.Linear(5, 3)\n")
_O_SCENARIOS = [
    # many steps on one mini-batch: the loss of the last step
    _O_DATA + "loss = overfit_one_batch(model, xb, yb, 50, 0.5)\nround(loss, 4)",
    # one step: the loss before that step, and the weights after it
    _O_DATA + "loss = overfit_one_batch(model, xb, yb, 1, 0.5)\n[round(loss, 4), model.weight[0].tolist()]",
    # the model itself has learned: its loss afterwards, measured again
    _O_DATA + "overfit_one_batch(model, xb, yb, 20, 0.5)\nround(nn.functional.cross_entropy(model(xb), yb).item(), 4)",
]

# --- exercise 3: DeepMLP --------------------------------------------------------------------------------------------------

_D_HEAD = ("import torch\nfrom torch import nn\n\n\nclass DeepMLP(nn.Module):\n"
           "    def __init__(self, depth, width, he=False, norm=False, residual=False):\n        super().__init__()\n"
           "        self.residual = residual\n        self.inp = nn.Linear(784, width)\n"
           "        # the depth - 2 hidden layers, in a list that PyTorch sees,\n        # with a BatchNorm for each when norm is True\n"
           "        self.hidden = nn.ModuleList()\n        for _ in range(depth - 2):\n            self.hidden.append(nn.Linear(width, width))\n"
           "        self.norms = None\n        if norm:\n            self.norms = nn.ModuleList()\n            for _ in range(depth - 2):\n"
           "                self.norms.append(nn.BatchNorm1d(width))\n        self.out = nn.Linear(width, 10)\n        if he:\n")
_HE = ("            for layer in [self.inp] + list(self.hidden) + [self.out]:\n"
       "                nn.init.kaiming_normal_(layer.weight, nonlinearity=\"relu\")\n                nn.init.zeros_(layer.bias)\n")
_F_HEAD = "\n    def forward(self, x):\n        h = torch.relu(self.inp(x))\n        for k in range(len(self.hidden)):\n"
_F_TAIL = "        return self.out(h)\n"
_BODY = ("            z = self.hidden[k](h)\n            if self.norms is not None:\n                z = self.norms[k](z)\n"
         "            z = torch.relu(z)\n            if self.residual:\n                h = h + z\n            else:\n                h = z\n")
_D_START = (_D_HEAD + "            # He initialization for every nn.Linear, with the bias terms at 0\n            ...\n" + _F_HEAD
            + "            # hidden layer k, then its BatchNorm when there is one, then ReLU;\n"
            "            # with residual, add the result to h, and without it, the result replaces h\n            ...\n" + _F_TAIL)


def _d(he: str = _HE, body: str = _BODY) -> str:
    return _D_HEAD + he + _F_HEAD + body + _F_TAIL


def _course(name: str):
    """A class or function from the course's solution of an exercise."""
    names: dict = {}
    exec(compile(_d() if name == "DeepMLP" else _o(), f"<{name}>", "exec"), names)
    return names[name]


DeepMLP = _course("DeepMLP")

_D_DATA = "import torch\n\ntorch.manual_seed(0)\nx = torch.rand(3, 784)\n"
_D_SCENARIOS = [
    # the plain network: nn.Linear's own starting weights, no norm, no residual
    _D_DATA + "model = DeepMLP(4, 8)\n[round(v, 4) for v in model(x)[0].tolist()]",
    # with residual connections
    _D_DATA + "model = DeepMLP(4, 8, residual=True)\n[round(v, 4) for v in model(x)[0].tolist()]",
    # with batch norm, in train mode
    _D_DATA + "model = DeepMLP(4, 8, norm=True)\n[round(v, 4) for v in model(x)[0].tolist()]",
    # He initialization: the spread of the weights, and the bias terms
    _D_DATA + "model = DeepMLP(10, 64, he=True)\n"
              "spreads = [model.inp.weight.std(), model.hidden[7].weight.std(), model.out.weight.std()]\n"
              "[round(v.item(), 3) for v in spreads] + [model.out.bias.abs().sum().item()]",
    # all of it together
    _D_DATA + "model = DeepMLP(6, 16, he=True, norm=True, residual=True)\n[round(v, 4) for v in model(x)[1].tolist()]",
]


_HE_SPREADS = [(2 / 784) ** 0.5, (2 / 64) ** 0.5, (2 / 64) ** 0.5]   # sigma of He initialization: inp, a hidden layer, out


def _he_other_draws(got, text) -> bool:
    """The He scenario: the spreads of He initialization and bias terms at 0, but other random numbers than the
    course's, such as the layers in another order, or kaiming_uniform_."""
    try:
        g = np.asarray(got, dtype=float)
    except (TypeError, ValueError):
        return False
    return g.shape == (4,) and g[3] == 0 and bool(np.all(np.abs(g[:3] / np.array(_HE_SPREADS) - 1) < 0.1))


def _close(got, want) -> bool:
    try:
        g, w = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    except (TypeError, ValueError):
        return got == want
    return g.shape == w.shape and bool(np.allclose(g, w, rtol=1e-4, atol=1e-4))


# --- the picture ----------------------------------------------------------------------------------------------------------

def _course_trainer():
    """The Trainer of nn-trainer, from the course's solutions there: the same as the learner's, when that is right."""
    spec = importlib.util.spec_from_file_location("everything_ai_checks_nn_trainer_for_deep", paths.LESSONS / "nn-trainer" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    ours = dict(projects._BUILDS_ON)   # nn-trainer's checks say what they build on; this lesson builds on nothing
    try:
        spec.loader.exec_module(mod)
    finally:
        projects._BUILDS_ON.clear()
        projects._BUILDS_ON.update(ours)
    return mod._course_trainer()


def _sgd(model, train) -> None:
    opt = torch.optim.SGD(model.parameters(), lr=SGD_LR)
    for _ in range(EPOCHS):
        for xb, yb in train:
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()


@functools.lru_cache(maxsize=8)
def run(he: bool, norm: bool, residual: bool, adamw: bool) -> int:
    """The 10 layers trained for 3 epochs, seed 0: with SGD at 0.1, or with the Trainer of nn-trainer (AdamW at 0.003,
    a warmup of one epoch, cosine decay, clipping at 1). Gives the number of the 1,000 test images it reads right."""
    X, y, X_test, y_test = fashion()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)   # faster on these small layers, and the numbers no longer depend on the number of cores
    try:
        torch.manual_seed(0)
        model = DeepMLP(DEPTH, WIDTH, he=he, norm=norm, residual=residual)
        torch.manual_seed(1)
        train = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
        if adamw:
            Trainer = _course_trainer()
            val = DataLoader(TensorDataset(X_test, y_test), batch_size=500)
            Trainer(model, train, val, epochs=EPOCHS, lr=ADAMW_LR, warmup=len(train), max_norm=1.0).fit()
        else:
            _sgd(model, train)
        model.eval()
        with torch.no_grad():
            return int((model(X_test).argmax(dim=1) == y_test).sum())
    finally:
        torch.set_num_threads(threads)


SETTINGS = [("plain", False, False, False, False), ("he", True, False, False, False), ("norm", False, True, False, False),
            ("residual", False, False, True, False), ("all", True, True, True, True)]


def fixes_picture(labels: dict) -> str:
    """For each way of training the 10 layers, the number of test images it reads right after 3 epochs."""
    rows = [(labels[key], run(he, norm, residual, adamw)) for key, he, norm, residual, adamw in SETTINGS]
    return draw.hbars(rows, top=1000, title=labels["title"], note=labels["note"], alt=labels["alt"]).text


# The lesson is done when these are correct. "overfit-first" is the choice question in the lesson.
REQUIRED = ["overfit-first", "overfit", "deep-mlp"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_SCENARIO = "With these lines after your {what}:\n\n```python\n{{scenario}}\n```\n\nyou get `{{got}}`, and it should be `{{want}}`"
_SCENARIO_NL = "Met deze regels na je {what}:\n\n```python\n{{scenario}}\n```\n\nkrijg je `{{got}}`, en dat moet `{{want}}` zijn"
_F_EN, _F_NL = _SCENARIO.format(what="function"), _SCENARIO_NL.format(what="functie")
_C_EN, _C_NL = _SCENARIO.format(what="class"), _SCENARIO_NL.format(what="class")

EXERCISES = {
    "overfit": ClassBox(
        name="overfit",
        start=_O_START,
        solution=_o(),
        scenarios=_O_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("nn", "overfit"),
        first=[_DOTS],
        mistakes=[
            Mistake("no-zero", wrong=_o(zero=""),
                    en=_F_EN + ": the gradients are never set back to 0, so every step also takes all gradients before it. Call `opt.zero_grad()` before `loss.backward()`.",
                    nl=_F_NL + ": de gradients gaan nooit terug naar 0, dus elke stap neemt ook alle gradients ervoor mee. Roep `opt.zero_grad()` aan vóór `loss.backward()`."),
            Mistake("no-step", wrong=_o(step=""),
                    en=_F_EN + ": the model never changes, so the loss stays where it started. A step needs the gradients of its round: first `loss.backward()`, then `opt.step()`, and `opt.zero_grad()` before `loss.backward()`, not between the two.",
                    nl=_F_NL + ": het model verandert nooit, dus de loss blijft waar hij begon. Een stap heeft de gradients van zijn ronde nodig: eerst `loss.backward()`, dan `opt.step()`, en `opt.zero_grad()` vóór `loss.backward()`, niet tussen die twee."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: the loss of the model on `xb` and `yb`, then `opt.zero_grad()`, `loss.backward()` and `opt.step()`.",
                      "nl": "Vervang de `...`: de loss van het model op `xb` en `yb`, dan `opt.zero_grad()`, `loss.backward()` en `opt.step()`."},
            "wrong": {"en": _F_EN + ": the loss of the last step. Does every round compute the loss on the same `xb` and `yb`, set the gradients to 0, and take a step after `backward()`?",
                      "nl": _F_NL + ": de loss van de laatste stap. Berekent elke ronde de loss op dezelfde `xb` en `yb`, zet hij de gradients op 0, en zet hij een stap na `backward()`?"},
            "almost": {"en": "Right for many steps. But " + _F_EN[0].lower() + _F_EN[1:] + ". Is the loss you give back that of the last step, measured before its update?",
                       "nl": "Goed voor veel stappen. Maar " + _F_NL[0].lower() + _F_NL[1:] + ". Is de loss die je teruggeeft die van de laatste stap, gemeten vóór zijn update?"},
            "correct": {"en": "Your overfit test trains a model on one mini-batch and gives the last loss. Your `overfit_one_batch` is now in your project, as `work/projects/nn/overfit.py`.",
                        "nl": "Je overfit-test traint een model op één mini-batch en geeft de laatste loss. Je `overfit_one_batch` staat nu in je project, als `work/projects/nn/overfit.py`."},
            "kept": {"en": "Your overfit test is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je overfit-test is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "deep-mlp": ClassBox(
        name="deep-mlp",
        start=_D_START,
        solution=_d(),
        scenarios=_D_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("nn", "deep_mlp"),
        first=[_DOTS],
        mistakes=[
            Mistake("relu-first", wrong=_d(body=_BODY.replace("            if self.norms is not None:\n                z = self.norms[k](z)\n            z = torch.relu(z)\n",
                                                             "            z = torch.relu(z)\n            if self.norms is not None:\n                z = self.norms[k](z)\n")),
                    en=_C_EN + ": your BatchNorm comes after the ReLU. It goes between the `nn.Linear` and the ReLU, as in [Batch normalization](/course/open/nn-batchnorm?lang=en).",
                    nl=_C_NL + ": je BatchNorm komt na de ReLU. Die hoort tussen de `nn.Linear` en de ReLU, zoals in [Batch normalization](/course/open/nn-batchnorm?lang=nl)."),
            Mistake("always-residual", wrong=_d(body=_BODY.replace("            if self.residual:\n                h = h + z\n            else:\n                h = z\n", "            h = h + z\n")),
                    en=_C_EN + ": your network adds `z` to `h` also when `residual` is `False`. Without residual connections, `z` replaces `h`: `h = z`.",
                    nl=_C_NL + ": je netwerk telt `z` ook bij `h` op als `residual` `False` is. Zonder residual connections vervangt `z` de `h`: `h = z`."),
            Mistake("relu-after-add", wrong=_d(body=_BODY.replace("            z = torch.relu(z)\n            if self.residual:\n                h = h + z\n            else:\n                h = z\n",
                                                                 "            if self.residual:\n                h = torch.relu(h + z)\n            else:\n                h = torch.relu(z)\n")),
                    en=_C_EN + ": your ReLU comes after the addition. In this block, the ReLU belongs to the layer: `z = torch.relu(z)`, and then `h + z`, so that `h` itself passes on unchanged.",
                    nl=_C_NL + ": je ReLU komt na het optellen. In dit blok hoort de ReLU bij de layer: `z = torch.relu(z)`, en dan `h + z`, zodat `h` zelf ongewijzigd doorgaat."),
            Mistake("he-hidden-only", wrong=_d(he=_HE.replace("[self.inp] + list(self.hidden) + [self.out]", "self.hidden")),
                    en=_C_EN + ": the first and the last layer keep the starting weights of `nn.Linear`. Give every `nn.Linear` He initialization: also `self.inp` and `self.out`.",
                    nl=_C_NL + ": de eerste en de laatste layer houden de startgewichten van `nn.Linear`. Geef elke `nn.Linear` He initialization: ook `self.inp` en `self.out`."),
            Mistake("he-bias", wrong=_d(he=_HE.replace("                nn.init.zeros_(layer.bias)\n", "")),
                    en=_C_EN + ": the bias terms keep the random starting values of `nn.Linear`. With He initialization they start at 0: `nn.init.zeros_(layer.bias)`.",
                    nl=_C_NL + ": de bias terms houden de willekeurige startwaarden van `nn.Linear`. Bij He initialization beginnen ze op 0: `nn.init.zeros_(layer.bias)`."),
            Mistake("he-other-draws", matches=_he_other_draws, state="almost",
                    en=_C_EN + ": your weights have the spread of He initialization, but other random numbers than the course's. The check compares the numbers themselves, so draw them as the course does: with `nn.init.kaiming_normal_`, first `self.inp`, then the hidden layers in order, then `self.out`.",
                    nl=_C_NL + ": je weights hebben de spreiding van He initialization, maar andere willekeurige getallen dan die van de course. De check vergelijkt de getallen zelf, dus trek ze zoals de course: met `nn.init.kaiming_normal_`, eerst `self.inp`, dan de hidden layers op volgorde, dan `self.out`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: He initialization for the layers in `__init__`, and in `forward` the steps of one hidden layer.",
                      "nl": "Vervang de twee `...`: He initialization voor de layers in `__init__`, en in `forward` de stappen van één hidden layer."},
            "wrong": {"en": _C_EN + ": what the plain network gives for the first image. In every round: the hidden layer, its BatchNorm when there is one, ReLU, and then `h + z` with `residual`, or `z` without it?",
                      "nl": _C_NL + ": wat het gewone netwerk geeft voor het eerste plaatje. In elke ronde: de hidden layer, zijn BatchNorm als die er is, ReLU, en dan `h + z` met `residual`, of `z` zonder?"},
            "almost": {"en": "Right for the plain network. But " + _C_EN[0].lower() + _C_EN[1:] + ". Does your class do what `he`, `norm` and `residual` ask?",
                       "nl": "Goed voor het gewone netwerk. Maar " + _C_NL[0].lower() + _C_NL[1:] + ". Doet je class wat `he`, `norm` en `residual` vragen?"},
            "correct": {"en": "One class for all versions of the deep network: plain, with He initialization, with BatchNorm, with residual connections, or all at once. Your `DeepMLP` is now in your project, as `work/projects/nn/deep_mlp.py`.",
                        "nl": "Eén class voor alle versies van het diepe netwerk: gewoon, met He initialization, met BatchNorm, met residual connections, of alles tegelijk. Je `DeepMLP` staat nu in je project, als `work/projects/nn/deep_mlp.py`."},
            "kept": {"en": "Your `DeepMLP` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `DeepMLP` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("overfit", _O_START, "empty", None),
    ("overfit", _o(loss="        loss = F.cross_entropy(model(xb), yb)\n", zero="        opt.zero_grad()\n"), "correct", None),
    ("overfit", _o(zero=""), "wrong", "no-zero"),
    ("overfit", _o(step=""), "wrong", "no-step"),
    # a step before backward: the gradients are None after zero_grad, so the model never changes either
    ("overfit", _o(zero="        opt.zero_grad()\n        opt.step()\n", step=""), "wrong", "no-step"),
    ("deep-mlp", _D_START, "empty", None),
    ("deep-mlp", _d(he=_HE.replace("nonlinearity=\"relu\"", "")), "correct", None),
    ("deep-mlp", _d(body=_BODY.replace("            if self.norms is not None:\n                z = self.norms[k](z)\n            z = torch.relu(z)\n",
                                       "            z = torch.relu(z)\n            if self.norms is not None:\n                z = self.norms[k](z)\n")), "wrong", "relu-first"),
    ("deep-mlp", _d(body=_BODY.replace("            if self.residual:\n                h = h + z\n            else:\n                h = z\n", "            h = h + z\n")), "wrong", "always-residual"),
    ("deep-mlp", _d(he=_HE.replace("[self.inp] + list(self.hidden) + [self.out]", "self.hidden")), "wrong", "he-hidden-only"),
    ("deep-mlp", _d(he=_HE.replace("                nn.init.zeros_(layer.bias)\n", "")), "wrong", "he-bias"),
    # He initialization right, but the layers in another order, or from a uniform distribution: other random numbers
    ("deep-mlp", _d(he=_HE.replace("[self.inp] + list(self.hidden) + [self.out]", "list(self.hidden) + [self.inp, self.out]")), "almost", "he-other-draws"),
    ("deep-mlp", _d(he=_HE.replace("kaiming_normal_", "kaiming_uniform_")), "almost", "he-other-draws"),
    # Xavier: another spread, so not the known mistake
    ("deep-mlp", _d(he=_HE.replace('kaiming_normal_(layer.weight, nonlinearity="relu")', "xavier_normal_(layer.weight)")), "almost", None),
    ("overfit", _o(backward=""), "wrong", "no-step"),
]
