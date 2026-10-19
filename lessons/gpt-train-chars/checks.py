# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson gpt-train-chars: gpt_loaders, the mini-batches of windows of ids to train and to validate on,
on a device; and gpt_run, a Trainer that is ready to train the learner's GPT on the characters of a text. The demo
trains it on Tiny Shakespeare behind a button, on a GPU when there is one, shows the loss as it goes, and keeps the
best parameters in the project for the next lesson."""
import json
import math
import re
import time

import torch
from torch import nn

from everything_ai import paths, projects
from everything_ai.check import Mistake, ScenarioBox, run_code

# The run uses the learner's GPT of gpt-assemble, with their Block and attention, their gpt_examples, and their Trainer
# of nn-trainer, and gpt_run imports this lesson's own gpt_loaders; where one is missing or broken, the course's
# version stands in (template 1.3).
projects.builds_on(("llm", "attention", "attn-transformer-block", "causal-self-attention"),
                   ("llm", "block", "attn-transformer-block", "block"), ("llm", "gpt", "gpt-assemble", "gpt"),
                   ("llm", "gpt_data", "gpt-assemble", "gpt-examples"),
                   ("nn", "train_epoch", "nn-trainer", "train-epoch"), ("nn", "trainer", "nn-trainer", "trainer"),
                   ("llm", "gpt_loaders", "gpt-train-chars", "gpt-loaders"))

DATA = "data/tiny-shakespeare.txt"   # about 1.1 MB of plays by Shakespeare: see data/README.md
# What the demo trains: the sizes of the GPT (d, h, n_layers, block), the mini-batch and the epochs. On a GPU the GPT of
# gpt-count, 10.8 million parameters; on a CPU a GPT of about 1 million. The epochs are fixed, so that every run does the
# same work and can be held against the course's runs below; the time differs from computer to computer.
PLANS = {"gpu": {"sizes": (384, 6, 6, 256), "batch": 64, "epochs": 20},
         "cpu": {"sizes": (128, 4, 5, 128), "batch": 32, "epochs": 10}}
MIN_GPU_MEMORY = 7 * 2 ** 30   # the bigger GPT with mini-batches of 64 needs about 5 GB on the GPU
VOCAB = 65                     # the characters of Tiny Shakespeare
MLP_LOSS = 2.20                # the validation loss of the MLP of lm-mlp-context, with 3 characters of context
# The course's own runs of both plans, with the course's solutions, PyTorch 2.14.1 on the CPU of the course's computer
# (4 threads): the validation and training loss after every epoch, per seed. The GPU plan ran on that CPU too, which
# gives nearly the same numbers as a GPU, only slower: its minutes are those of the CPU, and only one seed ran, since
# one run of it takes the CPU about two hours. Each run is the demo's train with that seed.
REFERENCE = {
    "cpu": [
        {"seed": 0, "minutes": 7.3, "history": [
            (1, 2.7209, 2.3806), (2, 2.1497, 2.0571), (3, 1.8817, 1.9307), (4, 1.7285, 1.8302), (5, 1.637, 1.7749),
            (6, 1.5722, 1.7214), (7, 1.5289, 1.6938), (8, 1.4952, 1.6733), (9, 1.4753, 1.6632), (10, 1.467, 1.6618),
        ]},
        {"seed": 1, "minutes": 7.2, "history": [
            (1, 2.7245, 2.4143), (2, 2.1785, 2.0915), (3, 1.9008, 1.9431), (4, 1.7395, 1.8384), (5, 1.6392, 1.7762),
            (6, 1.5731, 1.7309), (7, 1.5253, 1.6944), (8, 1.4931, 1.6732), (9, 1.4724, 1.6619), (10, 1.464, 1.66),
        ]},
        {"seed": 2, "minutes": 7.3, "history": [
            (1, 2.7357, 2.4028), (2, 2.1671, 2.0859), (3, 1.8999, 1.9294), (4, 1.7461, 1.8437), (5, 1.6538, 1.7731),
            (6, 1.5866, 1.7319), (7, 1.5247, 1.706), (8, 1.5072, 1.683), (9, 1.4879, 1.6759), (10, 1.4754, 1.6737),
        ]},
    ],
    "gpu": [
        {"seed": 0, "minutes": 104.4, "history": [
            (1, 2.9392, 2.5213), (2, 2.4652, 2.4446), (3, 2.3173, 2.2022), (4, 2.0122, 2.0144), (5, 1.8073, 1.8833),
            (6, 1.6672, 1.7919), (7, 1.5759, 1.7232), (8, 1.5082, 1.674), (9, 1.4559, 1.6379), (10, 1.4104, 1.6062),
            (11, 1.3772, 1.5828), (12, 1.3452, 1.5712), (13, 1.3139, 1.5577), (14, 1.2921, 1.5459), (15, 1.2733, 1.5337),
            (16, 1.251, 1.5303), (17, 1.2377, 1.5276), (18, 1.2239, 1.5251), (19, 1.2174, 1.5245), (20, 1.2177, 1.524),
        ]},
    ],
}
# A run that ends at or below this validation loss has learned as the course's runs did: the highest end of those runs,
# plus a margin for another computer, another PyTorch, and float rounding on a GPU.
THRESHOLD = {"cpu": 1.75, "gpu": 1.60}   # cpu: the runs ended at 1.660 to 1.674; gpu: the run ended at 1.524, one seed
# Where the next-character guesses of the trained GPT are shown: places near the end of the book, in the text the GPT
# validates on and never trains on. Each is the text up to that place; the GPT reads the last `block` characters.
GUESS_AT = ["SEBASTIAN:\nWhat, art thou wakin", "It is a sleepy langu", "Do you not hear me speak?\n\n"]


def reference(plan: str, i: int = 0) -> list:
    """Run i of the course's runs of a plan, as the history the Trainer gives."""
    return [{"epoch": e, "train_loss": tr, "val_loss": va} for e, tr, va in REFERENCE[plan][i]["history"]]


def text() -> str:
    return (paths.ROOT / DATA).read_text(encoding="utf-8")


def n_params(vocab: int, d: int, n_layers: int, block: int) -> int:
    """The parameters of the GPT of gpt-assemble, counted as in gpt-count."""
    return vocab * d + block * d + n_layers * (12 * d * d + 10 * d) + 2 * d + d * vocab + vocab


def sci(x: float, lang: str) -> str:
    """A big number for KaTeX, with one digit before the power of ten: 2 \\cdot 10^{15}, and 1,1 in Dutch."""
    power = math.floor(math.log10(x))
    digits = round(x / 10 ** power, 1)
    if digits >= 10:
        digits, power = digits / 10, power + 1
    lead = f"{digits:g}".replace(".", "{,}" if lang == "nl" else ".")
    return f"{lead} \\cdot 10^{{{power}}}"


def grouped(n: int, lang: str) -> str:
    return f"{n:,}".replace(",", "." if lang == "nl" else ",")


def number(x: float, digits: int, lang: str) -> str:
    text_ = f"{x:,.{digits}f}"
    if lang == "nl":
        text_ = text_.replace(",", "_").replace(".", ",").replace("_", ".")
    return text_


# The lesson is done when these are correct. "best-epoch" is the choice question in the lesson.
REQUIRED = ["best-epoch", "gpt-loaders", "gpt-run"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_SCENARIO = "With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`"
_SCENARIO_NL = "Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn"


def _close(got, want) -> bool:
    if isinstance(want, (list, tuple)):
        return isinstance(got, (list, tuple)) and len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want))
    if isinstance(want, float) and isinstance(got, (int, float)) and not isinstance(got, bool):
        return math.isclose(got, want, rel_tol=1e-4, abs_tol=1e-4)
    if isinstance(want, int) and not isinstance(want, bool):
        return type(got) is int and got == want   # an id is a whole number: 45.0 is not an id
    return got == want


def _floats(got) -> bool:
    """Numbers with a decimal point where whole numbers belong, such as windows of ids made with dtype=torch.float32."""
    if isinstance(got, (list, tuple)):
        return any(_floats(g) for g in got)
    return isinstance(got, float)


def _on_cpu(got, text) -> bool:
    """The device scenario, with a tensor that stayed on the CPU."""
    return isinstance(got, list) and "cpu" in got


# --- exercise 2: gpt_loaders ---------------------------------------------------------------------------------------------

_L_HEAD = ("import torch\nfrom torch.utils.data import DataLoader, TensorDataset\nfrom llm.gpt_data import gpt_examples\n\n\n"
           "def gpt_loaders(ids, block, batch_size, device):\n")
_L_CUT = "    cut = int(0.9 * len(ids))\n"
_L_WINDOWS = "    X, Y = gpt_examples(ids[:cut], block)\n    X_val, Y_val = gpt_examples(ids[cut:], block)\n"
_L_DATA = ("    train = TensorDataset(torch.tensor(X).to(device), torch.tensor(Y).to(device))\n"
           "    val = TensorDataset(torch.tensor(X_val).to(device), torch.tensor(Y_val).to(device))\n")
_L_LOADERS = ("    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)\n"
              "    val_loader = DataLoader(val, batch_size=batch_size)\n")
_L_RETURN = "    return train_loader, val_loader\n"
_L_START = (_L_HEAD + "    # the first 90% of the ids to train on, the last 10% to validate on\n    cut = ...\n"
            "    # windows of block ids from each part, with your gpt_examples, as tensors on device\n    ...\n"
            "    # mini-batches of batch_size windows, shuffled for training only\n    ...\n" + _L_RETURN)


def _l(cut: str = _L_CUT, windows: str = _L_WINDOWS, data: str = _L_DATA, loaders: str = _L_LOADERS) -> str:
    return _L_HEAD + cut + windows + data + loaders + _L_RETURN


_L_SCENARIOS = [
    # the lesson's example: 45 ids to train on give 11 windows of 4, in 6 mini-batches of 2; the last 5 ids give one window
    "ids = list(range(50))\ntrain_loader, val_loader = gpt_loaders(ids, 4, 2, \"cpu\")\nbatches = []\n"
    "for xb, yb in val_loader:\n    batches.append([xb.tolist(), yb.tolist()])\n[len(train_loader), batches]",
    # the order of the windows: shuffled for training, in order for validation
    "import torch\n\nids = list(range(200))\ntrain_loader, val_loader = gpt_loaders(ids, 4, 8, \"cpu\")\ntorch.manual_seed(0)\n"
    "starts = []\nfor xb, yb in train_loader:\n    starts.append(xb[:, 0].tolist())\nfor xb, yb in val_loader:\n"
    "    starts.append(xb[:, 0].tolist())\nstarts",
    # the device: PyTorch's "meta" device stands in for a GPU, as in torch-gpu
    "train_loader, val_loader = gpt_loaders(list(range(50)), 4, 2, \"meta\")\nwhere = []\nfor xb, yb in train_loader:\n"
    "    where = [xb.device.type, yb.device.type]\nfor xb, yb in val_loader:\n    where = where + [xb.device.type, yb.device.type]\nwhere",
]
_LISTS = Mistake("lists", error=lambda e: isinstance(e, (AttributeError, TypeError)) and "'list' object has no attribute 'size'" in str(e),
                 en="`TensorDataset` needs tensors, and gets lists: your `gpt_examples` gives lists of windows. Make each a tensor first, with `torch.tensor(...)`, and move it to `device`.",
                 nl="`TensorDataset` heeft tensors nodig, en krijgt lijsten: je `gpt_examples` geeft lijsten van vensters. Maak van elk eerst een tensor, met `torch.tensor(...)`, en verplaats hem naar `device`.")

# --- exercise 3: gpt_run -------------------------------------------------------------------------------------------------------

_R_HEAD = ("from llm.gpt import GPT\nfrom llm.gpt_loaders import gpt_loaders\nfrom nn.trainer import Trainer\n\n\n"
           "def gpt_run(text, sizes, batch_size, epochs, device, path):\n    d, h, n_layers, block = sizes\n"
           "    # the characters of the text, sorted: character i has id i\n    chars = sorted(set(text))\n    index = {}\n"
           "    for i in range(len(chars)):\n        index[chars[i]] = i\n    ids = []\n    for ch in text:\n        ids.append(index[ch])\n")
_R_LOADERS = "    train_loader, val_loader = gpt_loaders(ids, block, batch_size, device)\n"
_R_MODEL = "    model = GPT(len(chars), d, h, n_layers, block).to(device)\n"
_R_TRAINER = ("    return Trainer(model, train_loader, val_loader, epochs, lr=0.001, warmup=100, max_norm=1.0,\n"
              "                   path=path)\n")
_R_START = (_R_HEAD + "    # your mini-batches, and your GPT for these characters, on device\n    ...\n"
            "    # your Trainer: AdamW at 0.001, a warmup of 100 steps, clipping at 1.0,\n"
            "    # and the best parameters saved at path\n    return ...\n")


def _r(loaders: str = _R_LOADERS, model: str = _R_MODEL, trainer: str = _R_TRAINER) -> str:
    return _R_HEAD + loaders + model + trainer


_R_TEXT = "text = \"to be or not to be, that is the question\\n\" * 10\n"
_R_SCENARIOS = [
    # the lesson's example: the sizes of the GPT, the logits for 3 ids, and what the Trainer got
    "import os\nimport tempfile\nimport torch\n\n" + _R_TEXT + "path = os.path.join(tempfile.mkdtemp(), \"best.pt\")\n"
    "trainer = gpt_run(text, (16, 2, 1, 8), 4, 3, \"cpu\", path)\nlogits = trainer.model(torch.tensor([[0, 1, 2]]))\n"
    "[list(logits.shape), len(trainer.model.blocks), trainer.epochs, trainer.total, trainer.warmup, trainer.max_norm, trainer.path == path]",
    # the learning rate, and a short training: the loss before and after
    "import torch\n\n" + _R_TEXT + "torch.manual_seed(0)\ntrainer = gpt_run(text, (16, 2, 1, 8), 4, 2, \"cpu\", None)\n"
    "before = trainer.evaluate()\nhistory = trainer.fit()\n"
    "[trainer.opt.param_groups[0][\"initial_lr\"], round(before, 4), round(history[-1][\"val_loss\"], 4)]",
    # the device: PyTorch's "meta" device stands in for a GPU
    _R_TEXT + "trainer = gpt_run(text, (16, 2, 1, 6), 5, 2, \"meta\", None)\nwhere = [trainer.model.head.weight.device.type]\n"
    "for xb, yb in trainer.train_loader:\n    where = where + [xb.device.type]\nwhere[:2] + [len(trainer.train_loader)]",
]

EXERCISES = {
    "gpt-loaders": ScenarioBox(
        name="gpt-loaders",
        start=_L_START,
        solution=_l(),
        scenarios=_L_SCENARIOS,
        compare=_close,
        save_as=("llm", "gpt_loaders"),
        first=[_DOTS],
        mistakes=[
            _LISTS,
            Mistake("floats", matches=lambda got, text: _floats(got),
                    en=_SCENARIO + ": your windows hold numbers with a decimal point, such as `45.0`. Ids are whole numbers, and the `nn.Embedding` of your GPT only looks up whole numbers. Make the tensors with `torch.tensor(X)`, without a `dtype`: a list of whole numbers gives a tensor of whole numbers.",
                    nl=_SCENARIO_NL + ": je vensters bevatten getallen met een komma, zoals `45.0`. Ids zijn gehele getallen, en de `nn.Embedding` van je GPT zoekt alleen gehele getallen op. Maak de tensors met `torch.tensor(X)`, zonder `dtype`: een lijst gehele getallen geeft een tensor van gehele getallen."),
            Mistake("train-all", wrong=_l(windows="    X, Y = gpt_examples(ids, block)\n    X_val, Y_val = gpt_examples(ids[cut:], block)\n"),
                    en=_SCENARIO + ": you make the windows to train on from all the ids, so the GPT also trains on the text it validates on. Train on `ids[:cut]` only.",
                    nl=_SCENARIO_NL + ": je maakt de vensters om op te trainen van alle ids, dus de GPT traint ook op de tekst waarop hij valideert. Train alleen op `ids[:cut]`."),
            Mistake("ten-percent", wrong=_l(cut="    cut = int(0.1 * len(ids))\n"),
                    en=_SCENARIO + ": you train on the first 10% of the ids and validate on the other 90%. The model learns from the big part: `cut` is at 90% of the ids.",
                    nl=_SCENARIO_NL + ": je traint op de eerste 10% van de ids en valideert op de andere 90%. Het model leert van het grote deel: `cut` ligt op 90% van de ids."),
            Mistake("split-windows", wrong=_l(windows=("    X, Y = gpt_examples(ids, block)\n    k = int(0.9 * len(X))\n"
                                                       "    X_val, Y_val = X[k:], Y[k:]\n    X, Y = X[:k], Y[:k]\n")),
                    en=_SCENARIO + ": you make the windows of all the ids, and then split the windows. Here the ids are split, as the comment says: `ids[:cut]` to train on and `ids[cut:]` to validate on, each part with windows of its own.",
                    nl=_SCENARIO_NL + ": je maakt de vensters van alle ids, en splitst daarna de vensters. Hier worden de ids gesplitst, zoals het commentaar zegt: `ids[:cut]` om op te trainen en `ids[cut:]` om op te valideren, elk deel met eigen vensters."),
            Mistake("no-shuffle", wrong=_l(loaders=_L_LOADERS.replace(", shuffle=True", "")),
                    en=_SCENARIO + ": your mini-batches to train on come in the same order every epoch, window after window of the book. Shuffle them: `shuffle=True` in the `DataLoader` of the training windows.",
                    nl=_SCENARIO_NL + ": je mini-batches om op te trainen komen elke epoch in dezelfde volgorde, venster na venster uit het boek. Schud ze: `shuffle=True` in de `DataLoader` van de trainingsvensters."),
            Mistake("val-shuffled", wrong=_l(loaders=_L_LOADERS.replace("DataLoader(val, batch_size=batch_size)", "DataLoader(val, batch_size=batch_size, shuffle=True)")),
                    en=_SCENARIO + ": your mini-batches to validate on are shuffled too. The validation loss is nearly the same in any order, and every shuffle draws from the random generator, so with the same seed the training gets other mini-batches. Shuffle the training windows only.",
                    nl=_SCENARIO_NL + ": je mini-batches om op te valideren worden ook geschud. De validation loss is in elke volgorde bijna hetzelfde, en elke keer schudden trekt uit de random generator, dus met dezelfde seed krijgt de training andere mini-batches. Schud alleen de trainingsvensters."),
            Mistake("not-moved", matches=_on_cpu,
                    en=_SCENARIO + ": some of your tensors stay on the CPU. Move every tensor of windows and targets, of both parts, with `.to(device)`.",
                    nl=_SCENARIO_NL + ": een deel van je tensors blijft op de CPU. Verplaats elke tensor met vensters en targets, van beide delen, met `.to(device)`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the three `...`: where the ids are cut, the windows of each part as tensors on `device`, and the two `DataLoader`s.",
                      "nl": "Vervang de drie `...`: waar de ids worden geknipt, de vensters van elk deel als tensors op `device`, en de twee `DataLoader`s."},
            "wrong": {"en": _SCENARIO + ". The ids cut at 90%, windows from each part, tensors on `device`, and mini-batches, shuffled only to train: steps 2 and 3 of the recipe?",
                      "nl": _SCENARIO_NL + ". De ids geknipt op 90%, vensters uit elk deel, tensors op `device`, en mini-batches, alleen om te trainen geschud: stap 2 en 3 van het recept?"},
            "almost": {"en": "Right for the first example. But " + _SCENARIO[0].lower() + _SCENARIO[1:] + ". Does your function use `block`, `batch_size` and `device`, and no fixed numbers? Are the windows to train on made of `ids[:cut]` only, and shuffled, and those to validate on in order?",
                       "nl": "Goed voor het eerste voorbeeld. Maar " + _SCENARIO_NL[0].lower() + _SCENARIO_NL[1:] + ". Gebruikt je functie `block`, `batch_size` en `device`, en geen vaste getallen? Komen de vensters om op te trainen alleen uit `ids[:cut]`, en worden ze geschud, en die om op te valideren niet?"},
            "correct": {"en": "Your function turns any list of ids into mini-batches to train and to validate on, already on the device: on a GPU, no mini-batch has to move during training. It is in your project now, as `{saved}`.",
                        "nl": "Je functie maakt van elke lijst ids mini-batches om op te trainen en te valideren, al op het device: op een GPU hoeft tijdens het trainen geen mini-batch meer te verhuizen. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `gpt_loaders` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `gpt_loaders` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "gpt-run": ScenarioBox(
        name="gpt-run",
        start=_R_START,
        solution=_r(),
        scenarios=_R_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("llm", "gpt_run"),
        first=[_DOTS],
        mistakes=[
            Mistake("vocab-text", wrong=_r(model=_R_MODEL.replace("len(chars)", "len(text)")),
                    en=_SCENARIO + ": your GPT gives a score for each of `len(text)` tokens, one for every character of the text. It needs one for every different character: `len(chars)`.",
                    nl=_SCENARIO_NL + ": je GPT geeft een score voor elk van `len(text)` tokens, één voor elk teken van de tekst. Hij heeft er één nodig voor elk verschillend teken: `len(chars)`."),
            Mistake("vocab-65", wrong=_r(model=_R_MODEL.replace("len(chars)", "65")),
                    en=_SCENARIO + ": your GPT always has 65 tokens, the characters of Tiny Shakespeare. This text has other characters: `len(chars)`.",
                    nl=_SCENARIO_NL + ": je GPT heeft altijd 65 tokens, de tekens van Tiny Shakespeare. Deze tekst heeft andere tekens: `len(chars)`."),
            Mistake("sizes-order", wrong=_r(model=_R_MODEL.replace("d, h, n_layers, block)", "d, n_layers, h, block)")),
                    en=_SCENARIO + ": your GPT has another number of blocks. `GPT` takes its sizes in the order of its `def` line: `vocab, d, h, n_layers, block`.",
                    nl=_SCENARIO_NL + ": je GPT heeft een ander aantal blocks. `GPT` krijgt zijn maten in de volgorde van zijn `def`-regel: `vocab, d, h, n_layers, block`."),
            Mistake("no-path", wrong=_r(trainer=_R_TRAINER.replace("path=path", "path=None")),
                    en=_SCENARIO + ": your Trainer does not get `path`, so it saves nothing. Give it `path=path`: then it keeps the parameters with the lowest validation loss, for the next lesson.",
                    nl=_SCENARIO_NL + ": je Trainer krijgt `path` niet, dus hij bewaart niets. Geef hem `path=path`: dan bewaart hij de parameters met de laagste validation loss, voor de volgende les."),
            Mistake("not-moved", matches=lambda got, text: isinstance(got, list) and got[:1] == ["cpu"],
                    en=_SCENARIO + ": your GPT stays on the CPU, and its mini-batches are on the device. Move the model too: `.to(device)`.",
                    nl=_SCENARIO_NL + ": je GPT blijft op de CPU, en zijn mini-batches staan op het device. Verplaats het model ook: `.to(device)`."),
            Mistake("batches-on-cpu", matches=lambda got, text: isinstance(got, list) and got[1:2] == ["cpu"],
                    en=_SCENARIO + ": your mini-batches stay on the CPU, and your GPT is on the device. Give your `gpt_loaders` the `device` that `gpt_run` gets.",
                    nl=_SCENARIO_NL + ": je mini-batches blijven op de CPU, en je GPT staat op het device. Geef je `gpt_loaders` het `device` dat `gpt_run` krijgt."),
            Mistake("fit-inside", wrong=_r(trainer=_R_TRAINER.replace("    return Trainer(", "    trainer = Trainer(") + "    trainer.fit()\n    return trainer\n"),
                    en=_SCENARIO + ": your `gpt_run` already trains, with `fit()`. It only sets up the run: the demo calls `fit()` itself, and shows how far it is. Give back the Trainer without training.",
                    nl=_SCENARIO_NL + ": je `gpt_run` traint al, met `fit()`. Hij zet de run alleen klaar: de demo roept `fit()` zelf aan, en laat zien hoe ver hij is. Geef de Trainer terug zonder te trainen."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: your mini-batches and your GPT on `device`, and the `Trainer` that will train it.",
                      "nl": "Vervang de twee `...`: je mini-batches en je GPT op `device`, en de `Trainer` die hem gaat trainen."},
            "wrong": {"en": _SCENARIO + ". Your `gpt_loaders`, a GPT of `len(chars)` tokens with the sizes in `sizes`, on `device`, and a `Trainer` with the settings of the comment: step 4 of the recipe?",
                      "nl": _SCENARIO_NL + ". Je `gpt_loaders`, een GPT van `len(chars)` tokens met de maten uit `sizes`, op `device`, en een `Trainer` met de instellingen uit het commentaar: stap 4 van het recept?"},
            "almost": {"en": "Right for the first example. But " + _SCENARIO[0].lower() + _SCENARIO[1:] + ". Does the Trainer get AdamW's learning rate of 0.001, and does your function pass `block`, `batch_size` and `device` on as it gets them?",
                       "nl": "Goed voor het eerste voorbeeld. Maar " + _SCENARIO_NL[0].lower() + _SCENARIO_NL[1:] + ". Krijgt de Trainer de learning rate van 0,001 voor AdamW, en geeft je functie `block`, `batch_size` en `device` door zoals hij ze krijgt?"},
            "correct": {"en": "Your run is ready: your mini-batches and your GPT on the device, and your Trainer, which starts at the loss of guessing and comes down. It is in your project now, as `{saved}`.",
                        "nl": "Je run staat klaar: je mini-batches en je GPT op het device, en je Trainer, die begint bij de loss van gokken en daalt. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `gpt_run` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `gpt_run` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# --- the demo: the run on Tiny Shakespeare --------------------------------------------------------------------------------------


def _saved(module: str) -> bool:
    return (projects.folder() / "llm" / f"{module}.py").exists()


def device() -> dict:
    """The device the demo trains on and its name: an NVIDIA GPU with enough memory, else the GPU of an Apple chip,
    else the CPU. `small` names an NVIDIA GPU with too little memory for the bigger GPT."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        if props.total_memory >= MIN_GPU_MEMORY:
            return {"device": "cuda", "name": props.name, "small": None}
        return {"device": "cpu", "name": "CPU", "small": props.name}
    if torch.backends.mps.is_available():
        return {"device": "mps", "name": "Apple GPU", "small": None}
    return {"device": "cpu", "name": "CPU", "small": None}


def plan() -> dict:
    """What the demo trains on this computer: the device, the sizes, the mini-batch, the epochs and the parameters."""
    where = device()
    p = PLANS["cpu" if where["device"] == "cpu" else "gpu"]
    d, h, n_layers, block = p["sizes"]
    return {**where, **p, "params": n_params(VOCAB, d, n_layers, block)}


def _paths(name: str = "gpt_chars") -> tuple:
    folder = projects.folder() / "llm"
    return folder / f"{name}.pt", folder / f"{name}.json"


def last_run(name: str = "gpt_chars") -> dict | None:
    """The last run the demo saved under this name, or None. res-ablation saves its runs under another name."""
    _, info = _paths(name)
    try:
        return json.loads(info.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class _OnDevice(nn.Module):
    """The learner's GPT, run inside `with torch.device(device):`, so that the tensors its forward makes, such as the
    causal mask and the places, are made on the device too. Its state_dict is the GPT's own, so the checkpoint loads
    straight into a GPT."""

    def __init__(self, model, where):
        super().__init__()
        self.inner = model
        self.where = where

    def forward(self, ids):
        with torch.device(self.where):
            return self.inner(ids)

    def state_dict(self, *args, **kwargs):
        return self.inner.state_dict(*args, **kwargs)


class _Counted:
    """The training mini-batches of a run, counted as they go, so the demo can show how far it is. With `fresh`, every
    epoch after the first takes its mini-batches from `fresh(epoch)`: windows that start at another place of the book."""

    def __init__(self, loader, step, fresh=None):
        self.loader = loader
        self.step = step
        self.fresh = fresh
        self.epoch = 0

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        if self.fresh is not None and self.epoch > 0:
            self.loader = self.fresh(self.epoch)
        self.epoch += 1
        for batch in self.loader:
            self.step()
            yield batch


def _shifted(gpt_loaders, book: str, block: int, batch: int, where: str, epochs: int, seed: int):
    """For every epoch, the training mini-batches of the learner's gpt_loaders on windows that start `offset` characters
    later in the book, a random offset under `block` per epoch: the same text, cut at other places. Without this, every
    epoch shows the same windows, and the bigger GPT learns them by heart within 8 epochs (the course's run). The piece of
    ids it gets ends so that its first 90%, the part gpt_loaders trains on, stops where the run's own split does: no
    window to validate on is ever trained on."""
    chars = sorted(set(book))
    index = {ch: i for i, ch in enumerate(chars)}
    ids = [index[ch] for ch in book]
    cut = int(0.9 * len(ids))
    g = torch.Generator().manual_seed(seed)
    offsets = [0] + torch.randint(1, block, (max(epochs - 1, 0),), generator=g).tolist()

    def fresh(epoch: int):
        off = offsets[epoch % len(offsets)]
        n = math.ceil((cut - off) / 0.9)
        while int(0.9 * n) > cut - off:
            n -= 1
        while int(0.9 * n) < cut - off:
            n += 1
        train_loader, _ = gpt_loaders(ids[off:off + n], block, batch, where)
        return train_loader

    return fresh


def _sync(where: str) -> None:
    if where == "cuda":
        torch.cuda.synchronize()
    elif where == "mps":
        torch.mps.synchronize()


def train(show, seed: int = 0, epochs: int | None = None, ablate=None, name: str = "gpt_chars",
          own: bool = True) -> dict | str | None:
    """Train the learner's GPT with their gpt_run on Tiny Shakespeare. `show` gets a dict with how far the run is,
    every few seconds. None when gpt_loaders.py or gpt_run.py is not in the project yet; "run" when the run stops with
    an error; "memory" when the GPU runs out of memory; "stopped" when the stop button ended the run before its first
    epoch did; else the run, with `kept`: None when its checkpoint became gpt_chars.pt, or the earlier run that stays
    there because its validation loss was lower. The marimo stop button ends the run early, with the best parameters
    so far kept. A new run trains into gpt_chars_new.pt, so stopping it never destroys a finished run. res-ablation
    passes `ablate`, a function that changes the GPT before it trains, its own `name` for the files, and `own=False`:
    a later lesson does not wait for gpt_loaders.py and gpt_run.py, since the course's versions stand in (template 1.3)."""
    if own and not (_saved("gpt_loaders") and _saved("gpt_run")):
        return None
    run = run_code("from llm.gpt_run import gpt_run\nfrom llm.gpt_loaders import gpt_loaders")
    if run.error is not None:
        return "run"
    gpt_run, gpt_loaders = run.names["gpt_run"], run.names["gpt_loaders"]
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]
    p = plan()
    where = p["device"]
    epochs = p["epochs"] if epochs is None else epochs
    if where == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True   # TF32 on NVIDIA GPUs since the RTX 30 series: much faster, nearly as precise
    checkpoint, info = _paths(name)
    fresh = checkpoint.with_name(f"{name}_new.pt")
    fresh.unlink(missing_ok=True)
    book = text()
    try:
        torch.manual_seed(seed)
        trainer = gpt_run(book, p["sizes"], p["batch"], epochs, where, str(fresh))
        if ablate is not None:
            ablate(trainer.model)
        trainer.model = _OnDevice(trainer.model, where)
    except Exception:
        return "run"
    steps = len(trainer.train_loader)
    windows = len(getattr(trainer.train_loader, "dataset", [])) or steps * p["batch"]
    state = {"step": 0, "shown": 0.0, "start": time.time()}

    def step():
        state["step"] += 1
        now = time.time()
        if now - state["shown"] >= 2 or state["step"] == 1:
            state["shown"] = now
            done = state["step"] - 1
            seconds = now - state["start"]
            left = seconds / done * (epochs * steps - done) if done >= 10 else None
            show({"epoch": done // steps + 1, "epochs": epochs, "step": done, "steps": epochs * steps,
                  "seconds": seconds, "left": left, "history": list(trainer.history)})

    shift = _shifted(gpt_loaders, book, p["sizes"][3], p["batch"], where, epochs, seed)
    trainer.train_loader = _Counted(trainer.train_loader, step, shift)
    stopped = False
    try:
        trainer.fit()
        _sync(where)
    except KeyboardInterrupt:   # the stop button of marimo
        stopped = True
    except RuntimeError as e:   # also torch.OutOfMemoryError, and the out-of-memory error of a Mac's GPU
        return "memory" if "out of memory" in str(e).lower() else "run"
    except Exception:
        return "run"
    history = [{k: int(v) if k == "epoch" else float(v) for k, v in h.items()} for h in trainer.history]
    if not history or not fresh.exists():   # stopped during the first epoch: nothing was saved
        fresh.unlink(missing_ok=True)
        return "stopped"
    result = {"device": where, "name": p["name"], "sizes": list(p["sizes"]), "batch": p["batch"], "epochs": epochs,
              "params": p["params"], "seed": seed, "stopped": stopped, "minutes": (time.time() - state["start"]) / 60,
              "history": history, "chars": "".join(sorted(set(book))), "stand_ins": stand_ins,
              "tokens": len(history) * windows * p["sizes"][3],
              # training on one token as in gpt-count: 2 FLOPs per parameter and attention, forward, three times that in all
              "flops": len(history) * windows * p["sizes"][3] * 3 * (2 * p["params"] + 2 * p["sizes"][2] * p["sizes"][3] * p["sizes"][0]),
              "best": None, "best_epoch": None}
    best = min(history, key=lambda h: h["val_loss"])
    result["best"], result["best_epoch"] = best["val_loss"], best["epoch"]
    result["guesses"] = guesses(result, fresh)
    old = last_run(name)
    if old is None or not checkpoint.exists() or result["best"] <= old["best"]:
        fresh.replace(checkpoint)
        info.write_text(json.dumps(result, indent=1), encoding="utf-8")
        result["kept"] = None
    else:   # the earlier run was better: it stays in the project
        fresh.unlink()
        result["kept"] = old
    return result


def guesses(result: dict, checkpoint) -> list | None:
    """The three likeliest next characters of the GPT in `checkpoint` at the places of GUESS_AT, on the CPU: for each,
    the end of the text it read (its last 16 characters), the character that really follows, and the three guesses
    with their chances."""
    run = run_code("from llm.gpt import GPT")
    if run.error is not None or not checkpoint.exists():
        return None
    d, h, n_layers, block = result["sizes"]
    chars = result["chars"]
    try:
        model = run.names["GPT"](len(chars), d, h, n_layers, block)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.eval()
        book = text()
        out = []
        for until in GUESS_AT:
            end = book.rindex(until) + len(until)
            ids = [chars.index(ch) for ch in book[end - block:end]]
            with torch.no_grad():
                probs = torch.softmax(model(torch.tensor([ids]))[0, -1], dim=0)
            top = torch.topk(probs, 3)
            out.append({"read": book[end - 16:end], "next": book[end], "top": [(chars[i], float(pr)) for pr, i in zip(top.values.tolist(), top.indices.tolist())]})
        return out
    except Exception:
        return None


def shown_text(text_: str) -> str:
    """A piece of text as the demo shows it: a newline as a sign, spaces as they are, so that a table cell can wrap."""
    return text_.replace("\n", "↵")


def shown_char(ch: str) -> str:
    """A character as the demo shows it: a space and a newline get a sign of their own."""
    return {" ": "␣", "\n": "↵"}.get(ch, ch)


def history_picture(history: list, labels: dict, epochs: int | None = None) -> str:
    """The loss of the run after every epoch: training solid, validation dashed and open, as in nn-trainer; a grey
    line at the loss of guessing among 65 characters, and one at the MLP of lm-mlp-context."""
    from everything_ai import chart
    n = max(epochs or 0, history[-1]["epoch"], 2)   # the x-axis has every epoch of the run, also while it trains
    epochs = [h["epoch"] for h in history]
    low = min([h["train_loss"] for h in history] + [1.2])
    panel = chart.Panel((0.5, n + 0.5), (math.floor(low * 2 - 0.4) / 2, 4.5), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    panel.line([0.5, n + 0.5], [math.log(VOCAB)] * 2, "ink-3", 1, dashed=True)
    panel.text(n + 0.4, math.log(VOCAB) + 0.22, labels["guess"], "ink-3", "end")
    panel.line([0.5, n + 0.5], [MLP_LOSS] * 2, "ink-3", 1, dashed=True)
    panel.text(n + 0.4, MLP_LOSS + 0.22, labels["mlp"], "ink-3", "end")
    panel.line(epochs, [h["train_loss"] for h in history], "loss", 2.4)
    panel.points(epochs, [h["train_loss"] for h in history], "loss")
    panel.line(epochs, [h["val_loss"] for h in history], "loss", 1.6, dashed=True)
    panel.points(epochs, [h["val_loss"] for h in history], "loss", hollow=True)
    # a key under the line of guessing, at the left, where the curves never come: filled for training, open for validation
    panel.text(0.6, 3.75, f"● {labels['train']}   ○ {labels['val']}", "loss", "start")
    svg = chart.figure([panel], alt=labels["alt"], width=330, height=260)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("gpt-loaders", _L_START, "empty", None),
    ("gpt-loaders", _l(), "correct", None),
    ("gpt-loaders", _l(cut="    cut = len(ids) * 9 // 10\n"), "correct", None),
    ("gpt-loaders", _l(data=("    train = TensorDataset(torch.tensor(X, device=device), torch.tensor(Y, device=device))\n"
                             "    val = TensorDataset(torch.tensor(X_val, device=device), torch.tensor(Y_val, device=device))\n")), "correct", None),
    ("gpt-loaders", _l(data=("    train = TensorDataset(torch.tensor(X), torch.tensor(Y))\n"
                             "    val = TensorDataset(torch.tensor(X_val), torch.tensor(Y_val))\n")), "wrong", "not-moved"),
    ("gpt-loaders", _l(data=("    train = TensorDataset(torch.tensor(X).to(device), torch.tensor(Y).to(device))\n"
                             "    val = TensorDataset(torch.tensor(X_val), torch.tensor(Y_val))\n")), "wrong", "not-moved"),
    ("gpt-loaders", _l(data="    train = TensorDataset(X, Y)\n    val = TensorDataset(X_val, Y_val)\n"), "error", "lists"),
    ("gpt-loaders", _l(data=("    train = TensorDataset(torch.tensor(X, dtype=torch.float32).to(device), torch.tensor(Y).to(device))\n"
                             "    val = TensorDataset(torch.tensor(X_val, dtype=torch.float32).to(device), torch.tensor(Y_val).to(device))\n")), "wrong", "floats"),
    ("gpt-loaders", _l(windows="    X, Y = gpt_examples(ids, block)\n    X_val, Y_val = gpt_examples(ids[cut:], block)\n"), "wrong", "train-all"),
    ("gpt-loaders", _l(cut="    cut = round(0.9 * len(ids))\n"), "correct", None),
    ("gpt-loaders", _l(cut="    cut = int(0.1 * len(ids))\n"), "wrong", "ten-percent"),
    ("gpt-loaders", _l(windows=("    X, Y = gpt_examples(ids, block)\n    k = int(0.9 * len(X))\n"
                                "    X_val, Y_val = X[k:], Y[k:]\n    X, Y = X[:k], Y[:k]\n")), "wrong", "split-windows"),
    ("gpt-loaders", _l(loaders=_L_LOADERS.replace(", shuffle=True", "")), "wrong", "no-shuffle"),
    ("gpt-loaders", _l(loaders=_L_LOADERS.replace("DataLoader(val, batch_size=batch_size)", "DataLoader(val, batch_size=batch_size, shuffle=True)")), "wrong", "val-shuffled"),
    ("gpt-loaders", _l(loaders=_L_LOADERS.replace("batch_size=batch_size, shuffle", "batch_size=2, shuffle").replace("DataLoader(val, batch_size=batch_size)", "DataLoader(val, batch_size=2)")), "almost", None),
    ("gpt-run", _R_START, "empty", None),
    ("gpt-run", _r(), "correct", None),
    ("gpt-run", _r(trainer="    trainer = Trainer(model, train_loader, val_loader, epochs, 0.001, 100, 1.0, 1, path)\n    return trainer\n"), "correct", None),
    ("gpt-run", _r(model="    model = GPT(len(chars), d, h, n_layers, block)\n    model.to(device)\n"), "correct", None),
    ("gpt-run", _r(model=_R_MODEL.replace("len(chars)", "len(text)")), "wrong", "vocab-text"),
    ("gpt-run", _r(model=_R_MODEL.replace("len(chars)", "65")), "wrong", "vocab-65"),
    ("gpt-run", _r(model=_R_MODEL.replace("d, h, n_layers, block)", "d, n_layers, h, block)")), "wrong", "sizes-order"),
    ("gpt-run", _r(trainer=_R_TRAINER.replace("path=path", "path=None")), "wrong", "no-path"),
    ("gpt-run", _r(model=_R_MODEL.replace(".to(device)", "")), "wrong", "not-moved"),
    ("gpt-run", _r(loaders=_R_LOADERS.replace("batch_size, device)", "batch_size, \"cpu\")")), "wrong", "batches-on-cpu"),
    ("gpt-run", _r(trainer=_R_TRAINER.replace("    return Trainer(", "    trainer = Trainer(") + "    trainer.fit()\n    return trainer\n"), "wrong", "fit-inside"),
    ("gpt-run", _r(loaders=_R_LOADERS.replace("ids, block,", "ids, 8,")), "almost", None),
    ("gpt-run", _r(loaders=_R_LOADERS.replace("block, batch_size,", "block, 4,")), "almost", None),
    ("gpt-run", _r(trainer=_R_TRAINER.replace("lr=0.001", "lr=1e-3")), "correct", None),
    ("gpt-run", _r(trainer=_R_TRAINER.replace("lr=0.001", "lr=0.01")), "almost", None),
]
