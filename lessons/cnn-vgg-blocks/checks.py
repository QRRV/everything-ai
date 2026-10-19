# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the project lesson cnn-vgg-blocks: the VGG block of Simonyan and Zisserman (2014), n convolutions of 3 by
3 with a ReLU each and then max pooling, saved in the learner's project (work/projects/cnn/vgg.py) when right; and the
parameters of a stack of small filters against one large filter that sees as far. The picture compares those two; the
demo builds networks of two blocks from the learner's VGGBlock, with 1 to 4 convolutions per block, and trains each
with their Trainer."""
import functools
import re

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch.utils.data import DataLoader, TensorDataset

from everything_ai import paths, projects
from everything_ai.check import ClassBox, FunctionBox, Mistake, Timeout, run_code, same_value, with_time_limit

# The demo trains with the learner's Trainer of nn-trainer and counts with their count_params of cnn-pytorch; where one
# is missing or broken, the course's version stands in (template 1.3).
projects.builds_on(("nn", "train_epoch", "nn-trainer", "train-epoch"), ("nn", "trainer", "nn-trainer", "trainer"),
                   ("cnn", "count", "cnn-pytorch", "count-params"))

DATA = "data/fashion-mnist-small.npz"   # 6,000 training and 1,000 test images of Fashion-MNIST: see data/README.md
EPOCHS, LR = 5, 0.01
DEPTHS = (1, 2, 4, 8)   # convolutions per block in the demo
WIDTHS = (8, 16)        # the channels of block 1 and block 2, as in SmallCNN


@functools.lru_cache(maxsize=1)
def fashion():
    """The training and test images as tensors of 0 to 1, each image as one channel of 28 by 28, and their labels."""
    d = np.load(paths.ROOT / DATA)
    X = torch.tensor(d["train_images"], dtype=torch.float32).reshape(-1, 1, 28, 28) / 255
    y = torch.tensor(d["train_labels"], dtype=torch.int64)
    X_test = torch.tensor(d["test_images"], dtype=torch.float32).reshape(-1, 1, 28, 28) / 255
    y_test = torch.tensor(d["test_labels"], dtype=torch.int64)
    return X, y, X_test, y_test


def weights_picture(labels: dict) -> str:
    """The weights of one convolution from C to C channels, in units of C², against how wide a piece it sees: one
    filter of k by k (k², grey, dashed) and a stack of 3 by 3 filters that sees as far ((k − 1) / 2 × 9: param)."""
    from everything_ai import chart
    ks = [3, 5, 7, 9, 11]
    one = [k * k for k in ks]
    stack = [(k - 1) // 2 * 9 for k in ks]
    p = chart.Panel((2, 12), (0, 130), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    xticks=ks, yticks=[0, 40, 80, 120])
    p.line(ks, one, "ink-3", 2.2, dashed=True)
    p.points(ks, one, "ink-3")
    p.line(ks, stack, "param", 2.2)
    p.points(ks, stack, "param")
    p.text(2.4, 118, labels["one"], "ink-2", "start")
    p.text(2.4, 102, labels["stack"], "param", "start")
    p.text(6.6, 52, "49", "ink-2", "end")
    p.text(7.0, 14, "27", "param", "middle")
    return chart.figure([p], alt=labels["alt"], width=330, height=260)


def grouped(n: int, lang: str) -> str:
    """A whole number with its thousands marked, as the language writes it: 17,810 or 17.810."""
    return f"{n:,}".replace(",", "." if lang == "nl" else ",")


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


# The lesson is done when these are correct. "vgg-shape" is the choice question in the lesson.
REQUIRED = ["vgg-shape", "stack-params", "vgg-block"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _close(got, want) -> bool:
    if isinstance(want, (list, tuple)):
        return isinstance(got, (list, tuple)) and len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want))
    if isinstance(want, float):
        return isinstance(got, (int, float)) and not isinstance(got, bool) and abs(got - want) <= 1e-4 * max(1.0, abs(want))
    return got == want


# --- exercise 2: a stack of small filters against one large one ---------------------------------------------------------

_P_HEAD = "def stack_vs_one(c, n):\n"
_P_BODY = ("    stack = n * (3 * 3 * c * c + c)\n    width = 2 * n + 1\n    one = width * width * c * c + c\n"
           "    return stack, one\n")
_P_START = (_P_HEAD + "    # stack: n convolutions of 3 by 3 from c to c channels, each with c bias terms;\n"
            "    # one: one convolution from c to c channels that sees the same piece, 2 * n + 1 wide\n    ...\n")


def _p(body: str = _P_BODY) -> str:
    return _P_HEAD + body


_P_WRONG = {
    "no-bias": _p(_P_BODY.replace(" + c)\n", ")\n").replace(" * c * c + c\n", " * c * c\n")),
    "channels": _p(_P_BODY.replace("3 * 3 * c * c + c", "3 * 3 * c + c").replace("width * width * c * c + c", "width * width * c + c")),
    "width": _p(_P_BODY.replace("width = 2 * n + 1", "width = 3 * n")),
    "one-bias": _p(_P_BODY.replace("n * (3 * 3 * c * c + c)", "n * 3 * 3 * c * c + c")),
    "swapped": _p(_P_BODY.replace("return stack, one", "return one, stack")),
}
_P_CALL = "`{call}` gives `{got}`, and it should give `{want}`"


def _same_pair(got, want) -> bool:
    """The two counts as a tuple or as a list: the same numbers are the same answer."""
    return same_value(tuple(got) if isinstance(got, list) else got, want)
_P_CALL_NL = "`{call}` geeft `{got}`, en dat moet `{want}` zijn"


# --- exercise 3: the VGG block -------------------------------------------------------------------------------------------

_V_HEAD = ("import torch\nfrom torch import nn\n\n\nclass VGGBlock(nn.Module):\n    def __init__(self, c_in, c_out, n):\n"
           "        super().__init__()\n")
_V_INIT = ("        self.convs = nn.ModuleList()\n        for _ in range(n):\n"
           "            self.convs.append(nn.Conv2d(c_in, c_out, 3, padding=1))\n            c_in = c_out\n"
           "        self.pool = nn.MaxPool2d(2)\n")
_V_FORWARD = "        for conv in self.convs:\n            x = torch.relu(conv(x))\n        return self.pool(x)\n"
_V_START = (_V_HEAD + "        # n convolutions of 3 by 3 with padding 1, in an nn.ModuleList: the first from c_in to c_out\n"
            "        # channels, the others from c_out to c_out; and max pooling of 2 by 2\n        ...\n\n"
            "    def forward(self, x):\n        # every convolution with a ReLU after it, then the pooling\n        ...\n")


def _v(init: str = _V_INIT, forward: str = _V_FORWARD) -> str:
    return _V_HEAD + init + "\n    def forward(self, x):\n" + forward


_V = _v()
_V_SCENARIOS = [
    # a block of 2 convolutions from 1 to 4 channels on 2 images of 8 by 8: the shape, and 4 numbers of the first image
    "import torch\n\ntorch.manual_seed(0)\nblock = VGGBlock(1, 4, 2)\ny = block(torch.rand(2, 1, 8, 8))\n"
    "[list(y.shape), y[0, :, 0, 0].tolist()]",
    # 3 convolutions from 3 to 8 channels on an image of 6 by 6
    "import torch\n\ntorch.manual_seed(0)\nblock = VGGBlock(3, 8, 3)\ny = block(torch.rand(1, 3, 6, 6))\n"
    "[list(y.shape), round(y.sum().item(), 4)]",
    # what PyTorch sees of a block of 3 convolutions from 2 to 5 channels: tensors of parameters, and numbers in them
    "block = VGGBlock(2, 5, 3)\n[len(list(block.parameters())), sum(p.numel() for p in block.parameters())]",
]
_V_WRONG = {
    "plain-list": _v(init=_V_INIT.replace("nn.ModuleList()", "[]")),
    "no-padding": _v(init=_V_INIT.replace(", padding=1)", ")")),
    "no-relu": _v(forward=_V_FORWARD.replace("torch.relu(conv(x))", "conv(x)")),
    "pool-each": _v(forward="        for conv in self.convs:\n            x = self.pool(torch.relu(conv(x)))\n        return x\n"),
    "one-conv": _v(init=("        self.convs = nn.ModuleList()\n        self.convs.append(nn.Conv2d(c_in, c_out, 3, padding=1))\n"
                         "        self.pool = nn.MaxPool2d(2)\n")),
}
# The right block with other starting values: the course's parameters, copied into the learner's block in order,
# give what the course's block gives.
_V_PROBE = ("\n\n" + _V.replace("class VGGBlock(", "class _CourseBlock(") + "\n\ntorch.manual_seed(0)\n"
            "_mine, _course = VGGBlock(3, 8, 3), _CourseBlock(3, 8, 3)\n"
            "_ps, _qs = list(_mine.parameters()), list(_course.parameters())\n"
            "_same = [list(p.shape) for p in _ps] == [list(q.shape) for q in _qs]\nif _same:\n    with torch.no_grad():\n"
            "        for _p, _q in zip(_ps, _qs):\n            _p.copy_(_q)\n    _x = torch.rand(2, 3, 6, 6)\n"
            "    _same = torch.allclose(_mine(_x), _course(_x), atol=1e-5)\n_same\n")


def _other_numbers(got, text) -> bool:
    run = run_code(text + _V_PROBE)
    return run.error is None and run.value is True


_C_EN = "With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`"
_C_NL = "Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn"


EXERCISES = {
    "stack-params": FunctionBox(
        name="stack-params",
        function="stack_vs_one",
        start=_P_START,
        solution=_p(),
        visible=[(64, 3)],
        hidden=[(1, 1), (16, 2), (3, 4), (8, 1)],
        python_values=True,
        compare=_same_pair,
        first=[_DOTS],
        mistakes=[
            Mistake("no-bias", wrong=_P_WRONG["no-bias"],
                    en=_P_CALL + ": you count the weights only. Every convolution also has a bias term for each of its `c` output channels: `c` more per layer.",
                    nl=_P_CALL_NL + ": je telt alleen de weights. Elke convolutie heeft ook een bias term voor elk van zijn `c` output channels: `c` meer per layer."),
            Mistake("channels", wrong=_P_WRONG["channels"],
                    en=_P_CALL + ": you give each output channel a filter of one channel. A filter covers every input channel: `3 * 3 * c` weights, and there are `c` filters, one per output channel.",
                    nl=_P_CALL_NL + ": je geeft elk output channel een filter van één channel. Een filter beslaat elk input channel: `3 * 3 * c` weights, en er zijn `c` filters, één per output channel."),
            Mistake("width", wrong=_P_WRONG["width"],
                    en=_P_CALL + ": your one filter is `3 * n` wide. The pieces of stacked filters overlap: the first sees 3, and every next one adds 2, so `n` of them see `2 * n + 1`.",
                    nl=_P_CALL_NL + ": je ene filter is `3 * n` breed. De stukken van gestapelde filters overlappen: de eerste ziet 3, en elke volgende voegt er 2 toe, dus `n` ervan zien `2 * n + 1`."),
            Mistake("swapped", wrong=_P_WRONG["swapped"],
                    en=_P_CALL + ": the right two numbers, the other way round. Give the stack first, then the one convolution: `return stack, one`.",
                    nl=_P_CALL_NL + ": de goede twee getallen, andersom. Geef eerst de stapel, dan de ene convolutie: `return stack, one`."),
            Mistake("one-bias", wrong=_P_WRONG["one-bias"],
                    en=_P_CALL + ": your stack has `c` bias terms in all, and each of its `n` convolutions has `c` of its own.",
                    nl=_P_CALL_NL + ": je stapel heeft in totaal `c` bias terms, en elk van zijn `n` convoluties heeft er zelf `c`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: the parameters of the stack, those of the one convolution, and both as a pair.",
                      "nl": "Vervang de `...`: de parameters van de stapel, die van de ene convolutie, en allebei als paar."},
            "arguments": {"en": "The check calls `{call}`, with the number of channels and the number of convolutions, and the `def` line of your function has another number of names between its parentheses. Keep `(c, n)`.",
                          "nl": "De check roept `{call}` aan, met het aantal channels en het aantal convoluties, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(c, n)`."},
            "wrong": {"en": _P_CALL + ". Count the weights and the bias terms of every convolution, in the stack and in the one convolution; and check how wide the one convolution must be to see as far as the stack.",
                      "nl": _P_CALL_NL + ". Tel de weights en de bias terms van elke convolutie, in de stapel en in de ene convolutie; en kijk hoe breed de ene convolutie moet zijn om even ver te zien als de stapel."},
            "almost": {"en": "Right for the example. But " + _P_CALL[0].lower() + _P_CALL[1:] + ". Does it work for 1 channel and for 1 convolution too?",
                       "nl": "Goed voor het voorbeeld. Maar " + _P_CALL_NL[0].lower() + _P_CALL_NL[1:] + ". Werkt het ook voor 1 channel en voor 1 convolutie?"},
            "correct": {"en": "Your count is right in all {n} cases. For three of 3 by 3 against one of 7 by 7, at 64 channels: 110,784 against 200,768 parameters.",
                        "nl": "Je telling klopt in alle {n} gevallen. Voor drie van 3 bij 3 tegen één van 7 bij 7, bij 64 channels: 110.784 tegen 200.768 parameters."},
        },
    ),
    "vgg-block": ClassBox(
        name="vgg-block",
        start=_V_START,
        solution=_V,
        scenarios=_V_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("cnn", "vgg"),
        first=[_DOTS],
        mistakes=[
            Mistake("same-in", error=lambda e: isinstance(e, RuntimeError) and re.search(r"to have \d+ channels, but got \d+ channels", str(e)) is not None,
                    en="A convolution gets another number of channels than it was made for. Only the first convolution of the block gets `c_in` channels; it gives `c_out`, so every next one goes from `c_out` to `c_out`.",
                    nl="Een convolutie krijgt een ander aantal channels dan waarvoor hij gemaakt is. Alleen de eerste convolutie van het blok krijgt `c_in` channels; hij geeft er `c_out`, dus elke volgende gaat van `c_out` naar `c_out`."),
            Mistake("plain-list", wrong=_V_WRONG["plain-list"],
                    en=_C_EN + ": PyTorch sees no parameters in your block. Your convolutions are in a plain Python list; in an `nn.ModuleList` PyTorch finds them, and the Trainer trains them.",
                    nl=_C_NL + ": PyTorch ziet geen parameters in je blok. Je convoluties staan in een gewone Python-lijst; in een `nn.ModuleList` vindt PyTorch ze, en traint de Trainer ze."),
            Mistake("no-padding", wrong=_V_WRONG["no-padding"],
                    en=_C_EN + ": without padding, every convolution takes 2 off the height and width. With `padding=1`, a convolution of 3 by 3 keeps the size, and only the pooling halves it.",
                    nl=_C_NL + ": zonder padding haalt elke convolutie 2 van de hoogte en breedte af. Met `padding=1` houdt een convolutie van 3 bij 3 de grootte, en halveert alleen de pooling hem."),
            Mistake("no-relu", wrong=_V_WRONG["no-relu"],
                    en=_C_EN + ": your convolutions have no ReLU after them, so together they are one linear step. Put `torch.relu` around every convolution.",
                    nl=_C_NL + ": na je convoluties komt geen ReLU, dus samen zijn ze één lineaire stap. Zet `torch.relu` om elke convolutie."),
            Mistake("pool-each", wrong=_V_WRONG["pool-each"],
                    en=_C_EN + ": you pool after every convolution, so the block halves the size `n` times. A VGG block pools once, at the end.",
                    nl=_C_NL + ": je doet pooling na elke convolutie, dus het blok halveert de grootte `n` keer. Een VGG block doet één keer pooling, aan het eind."),
            Mistake("one-conv", wrong=_V_WRONG["one-conv"],
                    en=_C_EN + ": your block has one convolution, whatever `n` is. Make `n` of them, in a `for` loop.",
                    nl=_C_NL + ": je blok heeft één convolutie, wat `n` ook is. Maak er `n`, in een `for`-loop."),
            Mistake("other-numbers", matches=_other_numbers,
                    en=_C_EN + ": your block is built right, but its convolutions start from other random numbers than the course's, as with He initialization or with the convolutions made in another order. The check compares the numbers that come out: keep PyTorch's own starting values, and make the convolutions in the order in which the data passes them.",
                    nl=_C_NL + ": je blok is goed gebouwd, maar zijn convoluties beginnen met andere willekeurige getallen dan die van de course, zoals met He initialization of met de convoluties in een andere volgorde gemaakt. De check vergelijkt de getallen die eruit komen: houd de eigen beginwaarden van PyTorch, en maak de convoluties in de volgorde waarin de data erdoor gaat."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: in `__init__` the `n` convolutions in an `nn.ModuleList` and the pooling, and in `forward` every convolution with a ReLU, then the pooling.",
                      "nl": "Vervang de twee `...`: in `__init__` de `n` convoluties in een `nn.ModuleList` en de pooling, en in `forward` elke convolutie met een ReLU, dan de pooling."},
            "wrong": {"en": _C_EN + ". `n` convolutions of 3 by 3 with padding 1, the first from `c_in` channels and the others from `c_out`, each with a ReLU, and one max pooling at the end: steps 1 and 2 of the recipe?",
                      "nl": _C_NL + ". `n` convoluties van 3 bij 3 met padding 1, de eerste van `c_in` channels en de andere van `c_out`, elk met een ReLU, en één keer max pooling aan het eind: stap 1 en 2 van het recept?"},
            "almost": {"en": "Right for the first block. But " + _C_EN[0].lower() + _C_EN[1:] + ". Does your block work for every `c_in`, `c_out` and `n`, and does PyTorch see all its convolutions?",
                       "nl": "Goed voor het eerste blok. Maar " + _C_NL[0].lower() + _C_NL[1:] + ". Werkt je blok voor elke `c_in`, `c_out` en `n`, en ziet PyTorch al zijn convoluties?"},
            "correct": {"en": "Your VGG block is right: `n` convolutions of 3 by 3 with a ReLU each, and one max pooling. Your `VGGBlock` is now in your project, as `{saved}`.",
                        "nl": "Je VGG block klopt: `n` convoluties van 3 bij 3 met elk een ReLU, en één keer max pooling. Je `VGGBlock` staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `VGGBlock` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `VGGBlock` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("stack-params", _P_START, "empty", None),
    ("stack-params", _p(), "correct", None),
    ("stack-params", _p("    return n * (9 * c ** 2 + c), (2 * n + 1) ** 2 * c ** 2 + c\n"), "correct", None),
    ("stack-params", _p("    return [n * (9 * c * c + c), (2 * n + 1) ** 2 * c * c + c]\n"), "correct", None),
] + [("stack-params", code, "wrong", key) for key, code in _P_WRONG.items()] + [
    ("vgg-block", _V_START, "empty", None),
    ("vgg-block", _V, "correct", None),
    ("vgg-block", _v(forward="        for k in range(len(self.convs)):\n            x = torch.relu(self.convs[k](x))\n"
                             "        return self.pool(x)\n"), "correct", None),
    ("vgg-block", _v(init=_V_INIT.replace("            c_in = c_out\n", "")), "error", "same-in"),
    # the right block with other starting values: He initialization, or the convolutions made in another order
    ("vgg-block", _v(init=_V_INIT + "        for conv in self.convs:\n            nn.init.kaiming_normal_(conv.weight, nonlinearity=\"relu\")\n"
                                    "            nn.init.zeros_(conv.bias)\n"), "wrong", "other-numbers"),
    ("vgg-block", _v(init=("        self.convs = nn.ModuleList()\n        for _ in range(n - 1):\n"
                           "            self.convs.append(nn.Conv2d(c_out, c_out, 3, padding=1))\n"
                           "        self.convs.insert(0, nn.Conv2d(c_in, c_out, 3, padding=1))\n        self.pool = nn.MaxPool2d(2)\n")),
     "wrong", "other-numbers"),
    # bias=False: other parameters, not other starting values
    ("vgg-block", _v(init=_V_INIT.replace("padding=1)", "padding=1, bias=False)")), "wrong", None),
] + [("vgg-block", code, "wrong", key) for key, code in _V_WRONG.items()]


# --- the demo: networks of two VGG blocks, from 1 to 4 convolutions per block ------------------------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "cnn" / f"{module}.py").exists()


def _accuracy(model, X, y) -> int:
    model.eval()
    with torch.no_grad():
        return int((model(X).argmax(dim=1) == y).sum())


def _network(VGGBlock, n: int):
    """Two of the learner's VGG blocks of n convolutions, 1 to 8 to 16 channels, and an output head: for n = 1, the
    same network as SmallCNN, with the same starting values."""
    class VGGNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = VGGBlock(1, WIDTHS[0], n)
            self.block2 = VGGBlock(WIDTHS[0], WIDTHS[1], n)
            self.out = torch.nn.Linear(WIDTHS[1] * 7 * 7, 10)

        def forward(self, x):
            x = self.block2(self.block1(x))
            return self.out(x.reshape(len(x), -1))
    return VGGNet()


def _demo():
    """For the demo: None when vgg.py is not in the project yet; "vgg" when a network of the learner's VGGBlock stops
    with an error; "trainer" when the Trainer or count_params does; "slow" when a training run takes too long; else a
    dict with, per number of convolutions per block, the parameters and the test images read right, one run each,
    seed 0, and the modules the course's versions stood in for."""
    if not _saved("vgg"):
        return None
    run = run_code("from cnn.vgg import VGGBlock")
    if run.error is not None:
        return "vgg"
    VGGBlock = run.names["VGGBlock"]
    run = run_code("from nn.trainer import Trainer\nfrom cnn.count import count_params")
    if run.error is not None:
        return "trainer"
    Trainer, count_params = run.names["Trainer"], run.names["count_params"]
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]
    X, y, X_test, y_test = fashion()

    def train(model):
        torch.manual_seed(1)
        loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
        val = DataLoader(TensorDataset(X_test, y_test), batch_size=500)
        Trainer(model, loader, val, epochs=EPOCHS, lr=LR, warmup=len(loader), max_norm=1.0).fit()
        return _accuracy(model, X_test, y_test)

    rows = []
    for n in DEPTHS:
        try:
            torch.manual_seed(0)
            model = _network(VGGBlock, n)
        except Exception:
            return "vgg"
        try:
            params = int(count_params(model))
        except Exception:
            return "trainer"
        try:
            rows.append((n, params, with_time_limit(lambda: train(model), 300)))
        except Timeout:
            return "slow"
        except Exception:
            return "trainer"
    return {"rows": rows, "stand_ins": stand_ins}


_DEMO_RUNS: dict = {}


def _project_texts() -> tuple:
    """The texts of the project files the demo uses, as they are now (None for a missing file)."""
    texts = []
    for package, module in (("cnn", "vgg"), ("cnn", "count"), ("nn", "train_epoch"), ("nn", "trainer")):
        path = projects.folder() / package / f"{module}.py"
        texts.append(path.read_text(encoding="utf-8") if path.exists() else None)
    return tuple(texts)


def demo():
    """The demo's result, worked out once for the project files as they are."""
    key = _project_texts()
    if key not in _DEMO_RUNS:
        _DEMO_RUNS.clear()
        _DEMO_RUNS[key] = _demo()
    return _DEMO_RUNS[key]
